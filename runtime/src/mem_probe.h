#pragma once

// LOCAL DIAGNOSTIC (not for upstream): mem probe. Where the game process's memory goes.
// LEONIDA_MEM_PROBE=<seconds> turns it on (absent or 0: off, and nothing below is installed). Every <seconds> it logs
//   [rt] mem-probe t=...: the live Vulkan objects the runtime made (pipelines by kind, shader modules and their SPIR-V
//                         bytes, VkDeviceMemory per memory type, split into allocated and imported), what the device
//                         adds (pipeline cache bytes, arena, title memory), and
//   [rt] mem-probe smaps: /proc/self/smaps of the Wine process, Rss and Swap per kind of mapping: title memory,
//                         other PE private memory (runtime + title heaps), PE images, PE sections, memory the Vulkan
//                         driver mapped for us, /dev/nvidia* mappings, other native (Unix-side) anonymous memory
//                         (driver heaps, glibc arenas), and files.
//   [rt] mem-probe breakdown: the same allocations by consumer (src/memory_breakdown.h): the arena with its reserved,
//                         bound and in-use bytes, geometry copies, title windows and shadows, upload heaps, staging,
//                         texture blocks, sampled textures, render targets and depth, and other, which add up to the
//                         allocated bytes of the t= line.
// The Vulkan counts come from wrapping the runtime's own function table (install()); helpers that load their own
// pointers (depth24, predicate, xdxr, swapchain) are not counted: they create a handful of objects.

#include <windows.h>
#include <vulkan/vulkan.h>

#include "memory_breakdown.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace leonida::mem_probe {

struct Allocation {
    std::uint32_t type = 0;
    std::uint64_t bytes = 0;
    bool imported = false;
    std::uintptr_t mapped = 0;
    std::uint64_t mapped_bytes = 0;
    // Who asked for it (an Owner scope at the allocation), or what was bound to it later (bind_image_memory).
    memory_breakdown::Consumer consumer = memory_breakdown::Consumer::other;
};

struct State {
    PFN_vkAllocateMemory allocate = nullptr;
    PFN_vkFreeMemory free = nullptr;
    PFN_vkMapMemory map = nullptr;
    PFN_vkUnmapMemory unmap = nullptr;
    PFN_vkCreateShaderModule create_module = nullptr;
    PFN_vkDestroyShaderModule destroy_module = nullptr;
    PFN_vkCreateGraphicsPipelines create_graphics = nullptr;
    PFN_vkCreateComputePipelines create_compute = nullptr;
    PFN_vkDestroyPipeline destroy_pipeline = nullptr;
    PFN_vkCreateImage create_image = nullptr;
    PFN_vkDestroyImage destroy_image = nullptr;
    PFN_vkBindImageMemory bind_image = nullptr;

    std::mutex mutex;
    std::unordered_map<VkDeviceMemory, Allocation> allocations;
    std::unordered_map<VkShaderModule, std::uint64_t> modules;
    std::unordered_map<VkPipeline, std::uint8_t> pipelines;  // 0 graphics, 1 compute
    std::unordered_map<VkImage, std::uint32_t> image_usage;   // VkImageUsageFlags of each live image
    std::uint64_t module_bytes = 0, modules_made = 0, module_bytes_made = 0;
    std::uint64_t graphics_live = 0, compute_live = 0, graphics_made = 0, compute_made = 0;
    std::uint64_t graphics_ns = 0, compute_ns = 0;  // time inside vkCreate*Pipelines
    std::uint64_t installed = 0;
    ULONGLONG started = 0;
};

inline State& state() {
    static State s;
    return s;
}

inline unsigned interval_seconds() {
    static const unsigned seconds = [] {
        char text[16] = {};
        DWORD n = GetEnvironmentVariableA("LEONIDA_MEM_PROBE", text, sizeof text);
        if (!n || n >= sizeof text) return 0u;
        return unsigned(std::strtoul(text, nullptr, 10));
    }();
    return seconds;
}

inline bool imported_memory(const VkMemoryAllocateInfo* info) {
    for (auto* p = static_cast<const VkBaseInStructure*>(info->pNext); p; p = p->pNext)
        if (p->sType == VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT ||
            p->sType == VK_STRUCTURE_TYPE_IMPORT_MEMORY_WIN32_HANDLE_INFO_KHR ||
            p->sType == VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR)
            return true;
    return false;
}

inline VKAPI_ATTR VkResult VKAPI_CALL allocate_memory(VkDevice device, const VkMemoryAllocateInfo* info,
                                                      const VkAllocationCallbacks* callbacks, VkDeviceMemory* out) {
    State& s = state();
    const VkResult result = s.allocate(device, info, callbacks, out);
    if (result == VK_SUCCESS && out && *out) {
        std::lock_guard lock(s.mutex);
        s.allocations[*out] = {info->memoryTypeIndex, info->allocationSize, imported_memory(info), 0, 0,
                               memory_breakdown::current_owner()};
    }
    return result;
}

inline VKAPI_ATTR void VKAPI_CALL free_memory(VkDevice device, VkDeviceMemory memory,
                                              const VkAllocationCallbacks* callbacks) {
    State& s = state();
    if (memory) {
        std::lock_guard lock(s.mutex);
        s.allocations.erase(memory);
    }
    s.free(device, memory, callbacks);
}

inline VKAPI_ATTR VkResult VKAPI_CALL map_memory(VkDevice device, VkDeviceMemory memory, VkDeviceSize offset,
                                                 VkDeviceSize size, VkMemoryMapFlags flags, void** data) {
    State& s = state();
    const VkResult result = s.map(device, memory, offset, size, flags, data);
    if (result == VK_SUCCESS && data && *data) {
        std::lock_guard lock(s.mutex);
        if (auto it = s.allocations.find(memory); it != s.allocations.end()) {
            it->second.mapped = reinterpret_cast<std::uintptr_t>(*data);
            it->second.mapped_bytes = size == VK_WHOLE_SIZE ? it->second.bytes - offset : size;
        }
    }
    return result;
}

inline VKAPI_ATTR void VKAPI_CALL unmap_memory(VkDevice device, VkDeviceMemory memory) {
    State& s = state();
    {
        std::lock_guard lock(s.mutex);
        if (auto it = s.allocations.find(memory); it != s.allocations.end()) it->second.mapped = 0;
    }
    s.unmap(device, memory);
}

inline VKAPI_ATTR VkResult VKAPI_CALL create_image(VkDevice device, const VkImageCreateInfo* info,
                                                   const VkAllocationCallbacks* callbacks, VkImage* out) {
    State& s = state();
    const VkResult result = s.create_image(device, info, callbacks, out);
    if (result == VK_SUCCESS && out && *out) {
        std::lock_guard lock(s.mutex);
        s.image_usage[*out] = info->usage;
    }
    return result;
}

inline VKAPI_ATTR void VKAPI_CALL destroy_image(VkDevice device, VkImage image, const VkAllocationCallbacks* callbacks) {
    State& s = state();
    if (image) {
        std::lock_guard lock(s.mutex);
        s.image_usage.erase(image);
    }
    s.destroy_image(device, image, callbacks);
}

// Memory nobody named that an image is bound to is that image's: a render target or depth buffer, or a sampled
// texture. A named block (the texture blocks) keeps its name, whatever is bound into it.
inline VKAPI_ATTR VkResult VKAPI_CALL bind_image_memory(VkDevice device, VkImage image, VkDeviceMemory memory,
                                                        VkDeviceSize offset) {
    State& s = state();
    const VkResult result = s.bind_image(device, image, memory, offset);
    if (result == VK_SUCCESS) {
        std::lock_guard lock(s.mutex);
        const auto a = s.allocations.find(memory);
        const auto u = s.image_usage.find(image);
        if (a != s.allocations.end() && u != s.image_usage.end() &&
            a->second.consumer == memory_breakdown::Consumer::other)
            a->second.consumer = memory_breakdown::image_consumer(u->second);
    }
    return result;
}

inline VKAPI_ATTR VkResult VKAPI_CALL create_shader_module(VkDevice device, const VkShaderModuleCreateInfo* info,
                                                           const VkAllocationCallbacks* callbacks,
                                                           VkShaderModule* out) {
    State& s = state();
    const VkResult result = s.create_module(device, info, callbacks, out);
    if (result == VK_SUCCESS && out && *out) {
        std::lock_guard lock(s.mutex);
        s.modules[*out] = info->codeSize;
        s.module_bytes += info->codeSize;
        ++s.modules_made;
        s.module_bytes_made += info->codeSize;
    }
    return result;
}

inline VKAPI_ATTR void VKAPI_CALL destroy_shader_module(VkDevice device, VkShaderModule module,
                                                        const VkAllocationCallbacks* callbacks) {
    State& s = state();
    if (module) {
        std::lock_guard lock(s.mutex);
        if (auto it = s.modules.find(module); it != s.modules.end()) {
            s.module_bytes -= it->second;
            s.modules.erase(it);
        }
    }
    s.destroy_module(device, module, callbacks);
}

inline std::uint64_t now_ns() {
    return std::uint64_t(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
            .count());
}

inline void note_pipelines(std::uint8_t kind, std::uint32_t count, const VkPipeline* made, std::uint64_t ns) {
    State& s = state();
    std::lock_guard lock(s.mutex);
    (kind ? s.compute_ns : s.graphics_ns) += ns;
    for (std::uint32_t i = 0; i < count; ++i) {
        if (!made[i]) continue;
        s.pipelines[made[i]] = kind;
        ++(kind ? s.compute_live : s.graphics_live);
        ++(kind ? s.compute_made : s.graphics_made);
    }
}

inline double mib(std::uint64_t bytes) { return double(bytes) / 1048576.0; }

// One line per pipeline call that takes 1 s or more: how long, and how big its SPIR-V was.
inline void note_slow(const char* kind, std::uint64_t ns, std::uint64_t spirv_bytes, std::uint32_t stages) {
    if (ns < 1000000000ull) return;
    std::fprintf(stderr, "[rt] mem-probe slow %s pipeline: %.1f s, %u stages, %.1f MiB SPIR-V\n", kind,
                 double(ns) / 1e9, stages, mib(spirv_bytes));
}

inline std::uint64_t module_size(VkShaderModule module) {
    State& s = state();
    std::lock_guard lock(s.mutex);
    const auto it = s.modules.find(module);
    return it == s.modules.end() ? 0 : it->second;
}

inline VKAPI_ATTR VkResult VKAPI_CALL create_graphics_pipelines(VkDevice device, VkPipelineCache cache,
                                                                std::uint32_t count,
                                                                const VkGraphicsPipelineCreateInfo* infos,
                                                                const VkAllocationCallbacks* callbacks,
                                                                VkPipeline* out) {
    const std::uint64_t t0 = now_ns();
    const VkResult result = state().create_graphics(device, cache, count, infos, callbacks, out);
    const std::uint64_t ns = now_ns() - t0;
    if (result >= 0 && out) note_pipelines(0, count, out, ns);
    if (ns >= 1000000000ull && count == 1) {
        std::uint64_t bytes = 0;
        for (std::uint32_t i = 0; i < infos[0].stageCount; ++i) bytes += module_size(infos[0].pStages[i].module);
        note_slow("graphics", ns, bytes, infos[0].stageCount);
    }
    return result;
}

inline VKAPI_ATTR VkResult VKAPI_CALL create_compute_pipelines(VkDevice device, VkPipelineCache cache,
                                                               std::uint32_t count,
                                                               const VkComputePipelineCreateInfo* infos,
                                                               const VkAllocationCallbacks* callbacks,
                                                               VkPipeline* out) {
    const std::uint64_t t0 = now_ns();
    const VkResult result = state().create_compute(device, cache, count, infos, callbacks, out);
    const std::uint64_t ns = now_ns() - t0;
    if (result >= 0 && out) note_pipelines(1, count, out, ns);
    if (ns >= 1000000000ull && count == 1) note_slow("compute", ns, module_size(infos[0].stage.module), 1);
    return result;
}

inline VKAPI_ATTR void VKAPI_CALL destroy_pipeline(VkDevice device, VkPipeline pipeline,
                                                   const VkAllocationCallbacks* callbacks) {
    State& s = state();
    if (pipeline) {
        std::lock_guard lock(s.mutex);
        if (auto it = s.pipelines.find(pipeline); it != s.pipelines.end()) {
            --(it->second ? s.compute_live : s.graphics_live);
            s.pipelines.erase(it);
        }
    }
    s.destroy_pipeline(device, pipeline, callbacks);
}

// Wraps the runtime's function table (leonida::VulkanDeviceFunctions). Once per process: the first device wins.
template<typename Functions>
void install(Functions& vk) {
    if (!interval_seconds()) return;
    State& s = state();
    std::lock_guard lock(s.mutex);
    if (s.installed++) return;
    if (!vk.vkAllocateMemory || !vk.vkFreeMemory || !vk.vkMapMemory || !vk.vkUnmapMemory ||
        !vk.vkCreateShaderModule || !vk.vkDestroyShaderModule || !vk.vkCreateGraphicsPipelines ||
        !vk.vkCreateComputePipelines || !vk.vkDestroyPipeline || !vk.vkCreateImage || !vk.vkDestroyImage ||
        !vk.vkBindImageMemory)
        return;
    s.started = GetTickCount64();
    s.allocate = vk.vkAllocateMemory, vk.vkAllocateMemory = allocate_memory;
    s.free = vk.vkFreeMemory, vk.vkFreeMemory = free_memory;
    s.map = vk.vkMapMemory, vk.vkMapMemory = map_memory;
    s.unmap = vk.vkUnmapMemory, vk.vkUnmapMemory = unmap_memory;
    s.create_module = vk.vkCreateShaderModule, vk.vkCreateShaderModule = create_shader_module;
    s.destroy_module = vk.vkDestroyShaderModule, vk.vkDestroyShaderModule = destroy_shader_module;
    s.create_graphics = vk.vkCreateGraphicsPipelines, vk.vkCreateGraphicsPipelines = create_graphics_pipelines;
    s.create_compute = vk.vkCreateComputePipelines, vk.vkCreateComputePipelines = create_compute_pipelines;
    s.destroy_pipeline = vk.vkDestroyPipeline, vk.vkDestroyPipeline = destroy_pipeline;
    s.create_image = vk.vkCreateImage, vk.vkCreateImage = create_image;
    s.destroy_image = vk.vkDestroyImage, vk.vkDestroyImage = destroy_image;
    s.bind_image = vk.vkBindImageMemory, vk.vkBindImageMemory = bind_image_memory;
    std::fprintf(stderr, "[rt] mem-probe: installed, every %u s\n", interval_seconds());
}

// ---- /proc/self/smaps, classified ----
enum Bucket {
    b_title,        // the title's memory (XMem / title VirtualAlloc ranges)
    b_pe_private,   // other PE private memory: the runtime's and the title's heaps, stacks
    b_pe_high,      // PE private memory at 0x7f0000000000 and above (where native mmaps land: a check on Wine views)
    b_pe_image,     // PE images (exe/dll)
    b_pe_mapped,    // PE sections (file mappings, pagefile-backed sections)
    b_vk_mapped,    // memory the Vulkan driver mapped for vkMapMemory
    b_nvidia,       // other /dev/nvidia* mappings
    b_native_anon,  // native anonymous memory Wine does not know: driver heaps, glibc arenas, [heap], [stack]
    b_file,         // native file mappings (.so, ...)
    b_count
};
inline const char* bucket_name(int b) {
    static const char* names[b_count] = {"title",  "pe-private", "pe-private@7f", "pe-image", "pe-section",
                                         "vk-mapped", "nvidia-dev", "native-anon", "native-file"};
    return names[b];
}

struct Usage {
    std::uint64_t rss = 0, anon = 0, swap = 0, vmas = 0;
};

using Ranges = std::vector<std::pair<std::uint64_t, std::uint64_t>>;  // [begin, end), sorted

inline bool in_ranges(const Ranges& ranges, std::uint64_t va) {
    auto it = std::upper_bound(ranges.begin(), ranges.end(), std::make_pair(va, UINT64_MAX));
    if (it == ranges.begin()) return false;
    --it;
    return va >= it->first && va < it->second;
}

inline std::string read_small(const std::string& path) {
    HANDLE file = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_EXISTING, 0,
                              nullptr);
    if (file == INVALID_HANDLE_VALUE) return {};
    std::string text;
    std::vector<char> buffer(16384);
    DWORD got = 0;
    while (ReadFile(file, buffer.data(), DWORD(buffer.size()), &got, nullptr) && got) text.append(buffer.data(), got);
    CloseHandle(file);
    return text;
}

// The Unix process of this Wine process. Wine opens files in wineserver, so Z:\proc\self is the server: the game is
// the /proc entry whose command line names gta6rt and whose address space is the largest (one boot at a time).
inline std::string proc_dir() {
    static std::string found;
    if (!found.empty()) return found;
    WIN32_FIND_DATAA data{};
    HANDLE find = FindFirstFileA("Z:\\proc\\*", &data);
    if (find == INVALID_HANDLE_VALUE) return {};
    std::uint64_t best = 0;
    do {
        if (data.cFileName[0] < '1' || data.cFileName[0] > '9') continue;
        const std::string dir = std::string("Z:\\proc\\") + data.cFileName;
        std::string cmdline = read_small(dir + "\\cmdline");
        for (char& c : cmdline) c = char(std::tolower(static_cast<unsigned char>(c)));
        if (cmdline.find("gta6rt") == std::string::npos) continue;
        const std::string status = read_small(dir + "\\status");
        const std::size_t at = status.find("VmSize:");
        const std::uint64_t vm = at == std::string::npos ? 0 : std::strtoull(status.c_str() + at + 7, nullptr, 10);
        if (vm > best) {
            best = vm;
            found = dir;
        }
    } while (FindNextFileA(find, &data));
    FindClose(find);
    if (!found.empty()) std::fprintf(stderr, "[rt] mem-probe: this process is %s (VmSize %llu kB)\n", found.c_str(),
                                     (unsigned long long)best);
    return found;
}

// Parses /proc/<pid>/smaps (Wine reaches the Unix /proc through Z:). Classifies each VMA by its start address.
inline bool read_smaps(const Ranges& title, const Ranges& vk_mapped, Usage (&usage)[b_count], std::string& top_files,
                       std::string& error) {
    const std::string dir = proc_dir();
    if (dir.empty()) {
        error = "no /proc entry names gta6rt";
        return false;
    }
    HANDLE file = CreateFileA((dir + "\\smaps").c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr,
                              OPEN_EXISTING, 0, nullptr);
    if (file == INVALID_HANDLE_VALUE) {
        error = "open failed " + std::to_string(GetLastError());
        return false;
    }
    std::vector<char> buffer(1 << 20);
    std::string carry;
    int bucket = -1;
    std::string path;
    std::map<std::string, Usage> files;
    const auto field = [](const char* line, const char* name, std::uint64_t& out) {
        const std::size_t n = std::strlen(name);
        if (std::strncmp(line, name, n) != 0) return false;
        out = std::strtoull(line + n, nullptr, 10) * 1024;
        return true;
    };
    const auto header = [&](const char* line) {
        // 7f0000000000-7f0000001000 rw-p 00000000 00:00 0      [path]
        char* end = nullptr;
        const std::uint64_t begin = std::strtoull(line, &end, 16);
        if (!end || *end != '-') return false;
        const char* p = line;
        for (int fields = 0; fields < 5 && *p; ++fields) {  // skip range perms offset dev inode
            while (*p && *p != ' ') ++p;
            while (*p == ' ') ++p;
        }
        path = p;
        while (!path.empty() && (path.back() == '\n' || path.back() == '\r' || path.back() == ' ')) path.pop_back();
        if (in_ranges(title, begin)) {
            bucket = b_title;
        } else if (in_ranges(vk_mapped, begin)) {
            bucket = b_vk_mapped;
        } else if (path.rfind("/dev/nvidia", 0) == 0) {
            bucket = b_nvidia;
        } else {
            MEMORY_BASIC_INFORMATION region{};
            const bool known = VirtualQuery(reinterpret_cast<const void*>(begin), &region, sizeof region) &&
                               region.State == MEM_COMMIT;
            if (known && region.Type == MEM_IMAGE) bucket = b_pe_image;
            else if (known && region.Type == MEM_MAPPED) bucket = b_pe_mapped;
            else if (known && region.Type == MEM_PRIVATE) bucket = begin >= 0x7f0000000000ull ? b_pe_high : b_pe_private;
            else if (!path.empty() && path[0] == '/' ) bucket = b_file;
            else bucket = b_native_anon;
        }
        ++usage[bucket].vmas;
        return true;
    };
    const auto line_of = [&](const char* line) {
        std::uint64_t value = 0;
        if ((line[0] >= '0' && line[0] <= '9') || (line[0] >= 'a' && line[0] <= 'f')) {
            if (header(line)) return;
        }
        if (bucket < 0) return;
        if (field(line, "Rss:", value)) {
            usage[bucket].rss += value;
            if (bucket == b_file || bucket == b_pe_image || bucket == b_pe_mapped) files[path].rss += value;
        } else if (field(line, "Anonymous:", value)) {
            usage[bucket].anon += value;
        } else if (field(line, "Swap:", value)) {
            usage[bucket].swap += value;
            if (bucket == b_file || bucket == b_pe_image || bucket == b_pe_mapped) files[path].swap += value;
        }
    };
    DWORD got = 0;
    while (ReadFile(file, buffer.data(), DWORD(buffer.size()), &got, nullptr) && got) {
        carry.append(buffer.data(), got);
        std::size_t start = 0;
        for (std::size_t nl; (nl = carry.find('\n', start)) != std::string::npos; start = nl + 1) {
            carry[nl] = 0;
            line_of(carry.c_str() + start);
        }
        carry.erase(0, start);
    }
    CloseHandle(file);
    std::vector<std::pair<std::uint64_t, std::string>> ranked;
    for (auto& [name, u] : files) ranked.push_back({u.rss + u.swap, name});
    std::sort(ranked.rbegin(), ranked.rend());
    char text[256];
    for (std::size_t i = 0; i < ranked.size() && i < 4; ++i) {
        std::string name = ranked[i].second.empty() ? "(none)" : ranked[i].second;
        if (name.size() > 60) name = "..." + name.substr(name.size() - 57);
        std::snprintf(text, sizeof text, "%s%s %.0f", i ? ", " : "", name.c_str(), mib(ranked[i].first));
        top_files += text;
    }
    return true;
}

// Sums the Linux-side totals the kernel keeps (/proc/<pid>/smaps_rollup), for a check of the buckets.
inline std::string read_rollup() {
    const std::string dir = proc_dir();
    const std::string text = dir.empty() ? std::string() : read_small(dir + "\\smaps_rollup");
    if (text.empty()) return "rollup: open failed";
    std::string out = "rollup MiB:";
    for (const char* name : {"Rss:", "Anonymous:", "Swap:", "Pss_File:", "Pss_Shmem:"}) {
        const std::size_t at = text.find(std::string("\n") + name);
        if (at == std::string::npos) continue;
        const std::uint64_t kb = std::strtoull(text.c_str() + at + 1 + std::strlen(name), nullptr, 10);
        char item[64];
        std::snprintf(item, sizeof item, " %.*s %.0f", int(std::strlen(name) - 1), name, double(kb) / 1024.0);
        out += item;
    }
    return out;
}

// What the device adds to the line: pipeline cache bytes, arena use, title memory ranges.
struct DeviceFacts {
    std::function<std::string()> describe;  // one text fragment
    std::function<Ranges()> title_ranges;   // the title's memory
    std::function<memory_breakdown::DeviceShare()> share;  // the arena and the bindless tables, for the breakdown
    VkPhysicalDeviceMemoryProperties memory{};
};

inline std::string type_flags(VkMemoryPropertyFlags f) {
    std::string s;
    if (f & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) s += "DL";
    if (f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) s += s.empty() ? "HV" : "+HV";
    if (f & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) s += "+HC";
    if (f & VK_MEMORY_PROPERTY_HOST_CACHED_BIT) s += "+HCa";
    return s.empty() ? "none" : s;
}

inline std::string wall_clock() {
    SYSTEMTIME now{};
    GetLocalTime(&now);
    char text[16];
    std::snprintf(text, sizeof text, "%02u:%02u:%02u", now.wHour, now.wMinute, now.wSecond);
    return text;
}

inline void sample(const DeviceFacts& facts) {
    State& s = state();
    const double t = double(GetTickCount64() - s.started) / 1000.0;
    std::uint64_t type_count[VK_MAX_MEMORY_TYPES] = {}, type_bytes[VK_MAX_MEMORY_TYPES] = {};
    std::uint64_t import_count[VK_MAX_MEMORY_TYPES] = {}, import_bytes[VK_MAX_MEMORY_TYPES] = {};
    Ranges vk_mapped;
    memory_breakdown::Tally tally;
    std::uint64_t modules = 0, module_bytes = 0, modules_made = 0, module_bytes_made = 0;
    std::uint64_t graphics = 0, compute = 0, graphics_made = 0, compute_made = 0, graphics_ns = 0, compute_ns = 0;
    {
        std::lock_guard lock(s.mutex);
        for (const auto& [memory, a] : s.allocations) {
            const std::uint32_t i = a.type < VK_MAX_MEMORY_TYPES ? a.type : 0;
            if (a.imported) {
                ++import_count[i];
                import_bytes[i] += a.bytes;
            } else {
                ++type_count[i];
                type_bytes[i] += a.bytes;
            }
            if (a.mapped && !a.imported) vk_mapped.push_back({a.mapped, a.mapped + a.mapped_bytes});
            tally.add(a.consumer, a.bytes, a.imported);
        }
        modules = s.modules.size();
        module_bytes = s.module_bytes;
        modules_made = s.modules_made;
        module_bytes_made = s.module_bytes_made;
        graphics = s.graphics_live;
        compute = s.compute_live;
        graphics_made = s.graphics_made;
        compute_made = s.compute_made;
        graphics_ns = s.graphics_ns;
        compute_ns = s.compute_ns;
    }
    std::sort(vk_mapped.begin(), vk_mapped.end());
    std::string memory_text;
    char item[256];
    for (std::uint32_t i = 0; i < facts.memory.memoryTypeCount && i < VK_MAX_MEMORY_TYPES; ++i) {
        if (!type_count[i] && !import_count[i]) continue;
        std::snprintf(item, sizeof item, " type %u %s heap %u: %llu x %.0f MiB", i,
                      type_flags(facts.memory.memoryTypes[i].propertyFlags).c_str(),
                      facts.memory.memoryTypes[i].heapIndex, (unsigned long long)type_count[i], mib(type_bytes[i]));
        memory_text += item;
        if (import_count[i]) {
            std::snprintf(item, sizeof item, " (+%llu imported, %.0f MiB)", (unsigned long long)import_count[i],
                          mib(import_bytes[i]));
            memory_text += item;
        }
        memory_text += ';';
    }
    std::fprintf(stderr,
                 "[rt] mem-probe t=%.0fs (%s): pipelines live graphics %llu compute %llu (made %llu / %llu, %.1f s / %.1f s "
                 "in vkCreate*Pipelines); shader modules live %llu, %.0f MiB SPIR-V (made %llu, %.0f MiB); "
                 "vk memory:%s %s\n",
                 t, wall_clock().c_str(), (unsigned long long)graphics, (unsigned long long)compute, (unsigned long long)graphics_made,
                 (unsigned long long)compute_made, double(graphics_ns) / 1e9, double(compute_ns) / 1e9,
                 (unsigned long long)modules, mib(module_bytes), (unsigned long long)modules_made,
                 mib(module_bytes_made), memory_text.c_str(), facts.describe ? facts.describe().c_str() : "");
    std::fprintf(stderr, "[rt] mem-probe breakdown t=%.0fs: %s\n", t,
                 memory_breakdown::format(tally, facts.share ? facts.share() : memory_breakdown::DeviceShare{},
                                          module_bytes)
                     .c_str());
    Ranges title = facts.title_ranges ? facts.title_ranges() : Ranges{};
    std::sort(title.begin(), title.end());
    Usage usage[b_count];
    std::string top_files, error;
    const std::uint64_t t0 = now_ns();
    if (!read_smaps(title, vk_mapped, usage, top_files, error)) {
        std::fprintf(stderr, "[rt] mem-probe smaps t=%.0fs: %s\n", t, error.c_str());
        return;
    }
    std::string text;
    std::uint64_t rss = 0, swap = 0;
    for (int b = 0; b < b_count; ++b) {
        rss += usage[b].rss;
        swap += usage[b].swap;
        std::snprintf(item, sizeof item, " %s %.0f+%.0f (anon %.0f, %llu vmas);", bucket_name(b), mib(usage[b].rss),
                      mib(usage[b].swap), mib(usage[b].anon), (unsigned long long)usage[b].vmas);
        text += item;
    }
    std::fprintf(stderr,
                 "[rt] mem-probe smaps t=%.0fs (%.0f ms): total rss %.0f + swap %.0f MiB; rss+swap per kind:%s top "
                 "files: %s; %s\n",
                 t, double(now_ns() - t0) / 1e6, mib(rss), mib(swap), text.c_str(), top_files.c_str(),
                 read_rollup().c_str());
}

// Starts the sampling thread (once). The thread runs until the process ends.
inline void start(DeviceFacts facts) {
    const unsigned seconds = interval_seconds();
    if (!seconds) return;
    static std::once_flag once;
    std::call_once(once, [&] {
        std::thread([facts = std::move(facts), seconds] {
            for (;;) {
                Sleep(seconds * 1000);
                sample(facts);
            }
        }).detach();
    });
}

}  // namespace leonida::mem_probe
