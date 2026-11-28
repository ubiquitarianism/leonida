// NVIDIA Nsight Aftermath GPU crash dumps for the Vulkan device (vulkan_aftermath.h). Without LEONIDA_HAVE_AFTERMATH
// (no -DLEONIDA_AFTERMATH_SDK at configure time) this file only has the do-nothing versions.
#include "vulkan_aftermath.h"

#include "vulkan_device_fault.h"

#ifdef LEONIDA_HAVE_AFTERMATH

#include "shader_library.h"  // leonida::Json, leonida::JsonReader

#include <windows.h>

#include <GFSDK_Aftermath.h>
#include <GFSDK_Aftermath_GpuCrashDump.h>
#include <GFSDK_Aftermath_GpuCrashDumpDecoding.h>

#include <atomic>
#include <chrono>
#include <cinttypes>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#ifndef LEONIDA_AFTERMATH_DLL_DEFAULT
#define LEONIDA_AFTERMATH_DLL_DEFAULT ""
#endif
#ifndef LEONIDA_AFTERMATH_APP_VERSION
#define LEONIDA_AFTERMATH_APP_VERSION "unknown"
#endif

namespace leonida::aftermath {
namespace {

using Clock = std::chrono::steady_clock;
constexpr auto wait_limit = std::chrono::seconds(10);

std::wstring env_w(const wchar_t* name) {
    wchar_t buffer[32768];
    const DWORD n = GetEnvironmentVariableW(name, buffer, DWORD(std::size(buffer)));
    return n && n < std::size(buffer) ? std::wstring(buffer, n) : std::wstring();
}

bool env_is(const wchar_t* name, wchar_t value) {
    const std::wstring v = env_w(name);
    return v.size() == 1 && v[0] == value;
}

std::filesystem::path from_utf8(std::string_view text) {
    return std::filesystem::path(std::u8string(reinterpret_cast<const char8_t*>(text.data()), text.size()));
}

std::string narrow(const std::filesystem::path& path) {
    const std::u8string s = path.u8string();
    return std::string(s.begin(), s.end());
}

struct Library {
    HMODULE module = nullptr;
    PFN_GFSDK_Aftermath_EnableGpuCrashDumps EnableGpuCrashDumps = nullptr;
    PFN_GFSDK_Aftermath_GetCrashDumpStatus GetCrashDumpStatus = nullptr;
    PFN_GFSDK_Aftermath_GetShaderHashSpirv GetShaderHashSpirv = nullptr;
    PFN_GFSDK_Aftermath_GetShaderDebugInfoIdentifier GetShaderDebugInfoIdentifier = nullptr;
    PFN_GFSDK_Aftermath_GpuCrashDump_CreateDecoder CreateDecoder = nullptr;
    PFN_GFSDK_Aftermath_GpuCrashDump_DestroyDecoder DestroyDecoder = nullptr;
    PFN_GFSDK_Aftermath_GpuCrashDump_GetBaseInfo GetBaseInfo = nullptr;
    PFN_GFSDK_Aftermath_GpuCrashDump_GetDeviceInfo GetDeviceInfo = nullptr;
    PFN_GFSDK_Aftermath_GpuCrashDump_GetPageFaultInfo GetPageFaultInfo = nullptr;
    PFN_GFSDK_Aftermath_GpuCrashDump_GetPageFaultResourceInfo GetPageFaultResourceInfo = nullptr;
    PFN_GFSDK_Aftermath_GpuCrashDump_GetActiveShadersInfoCount GetActiveShadersInfoCount = nullptr;
    PFN_GFSDK_Aftermath_GpuCrashDump_GetActiveShadersInfo GetActiveShadersInfo = nullptr;
    PFN_GFSDK_Aftermath_GetShaderHashForShaderInfo GetShaderHashForShaderInfo = nullptr;
    PFN_GFSDK_Aftermath_GpuCrashDump_GenerateJSON GenerateJSON = nullptr;
    PFN_GFSDK_Aftermath_GpuCrashDump_GetJSON GetJSON = nullptr;

    bool load(const std::wstring& path) {
        module = path.find_first_of(L"\\/") != std::wstring::npos
                     ? LoadLibraryExW(path.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH)
                     : LoadLibraryW(path.c_str());
        if (!module) return false;
        bool ok = true;
        const auto get = [&](auto& fn, const char* name) {
            fn = reinterpret_cast<std::remove_reference_t<decltype(fn)>>(GetProcAddress(module, name));
            if (!fn) {
                std::fprintf(stderr, "[rt] aftermath: %s is missing from the library\n", name);
                ok = false;
            }
        };
        get(EnableGpuCrashDumps, "GFSDK_Aftermath_EnableGpuCrashDumps");
        get(GetCrashDumpStatus, "GFSDK_Aftermath_GetCrashDumpStatus");
        get(GetShaderHashSpirv, "GFSDK_Aftermath_GetShaderHashSpirv");
        get(GetShaderDebugInfoIdentifier, "GFSDK_Aftermath_GetShaderDebugInfoIdentifier");
        get(CreateDecoder, "GFSDK_Aftermath_GpuCrashDump_CreateDecoder");
        get(DestroyDecoder, "GFSDK_Aftermath_GpuCrashDump_DestroyDecoder");
        get(GetBaseInfo, "GFSDK_Aftermath_GpuCrashDump_GetBaseInfo");
        get(GetDeviceInfo, "GFSDK_Aftermath_GpuCrashDump_GetDeviceInfo");
        get(GetPageFaultInfo, "GFSDK_Aftermath_GpuCrashDump_GetPageFaultInfo");
        get(GetPageFaultResourceInfo, "GFSDK_Aftermath_GpuCrashDump_GetPageFaultResourceInfo");
        get(GetActiveShadersInfoCount, "GFSDK_Aftermath_GpuCrashDump_GetActiveShadersInfoCount");
        get(GetActiveShadersInfo, "GFSDK_Aftermath_GpuCrashDump_GetActiveShadersInfo");
        get(GetShaderHashForShaderInfo, "GFSDK_Aftermath_GetShaderHashForShaderInfo");
        get(GenerateJSON, "GFSDK_Aftermath_GpuCrashDump_GenerateJSON");
        get(GetJSON, "GFSDK_Aftermath_GpuCrashDump_GetJSON");
        return ok;
    }
};

struct Shader {
    std::string name;
    std::vector<std::uint32_t> words;  // only with LEONIDA_AFTERMATH_KEEP_SPIRV=1
};

struct State {
    Library lib;
    std::filesystem::path dir;
    std::filesystem::path shader_dir;  // LEONIDA_SHADERS: where a module's SPIR-V is read back for the decoder
    bool keep_spirv = false;
    VkDeviceDiagnosticsConfigFlagsNV extra_flags = 0;
    std::atomic<bool> enabled{false};
    std::atomic<PFN_vkCreateShaderModule> real_create{nullptr};
    std::atomic<PFN_vkGetDeviceProcAddr> real_get{nullptr};

    std::mutex shaders_mutex;  // leaf
    std::unordered_map<std::uint64_t, Shader> shaders;
    std::atomic<std::uint64_t> registered{0}, hash_failures{0};

    std::mutex debug_mutex;  // leaf
    std::map<std::pair<std::uint64_t, std::uint64_t>, std::vector<std::uint8_t>> debug_info;

    std::mutex dump_mutex;  // leaf, guards everything below
    std::condition_variable dump_cv;
    std::vector<std::uint8_t> dump;
    std::filesystem::path dump_path;
    bool have_dump = false, decoded = false;
    std::string summary;

    std::mutex wait_mutex;  // held by the first device-loss waiter; later ones queue behind it
    bool waited = false;
};

State& state() {
    static State* s = new State;  // never destroyed: Aftermath's threads may call back during exit
    return *s;
}

std::string hex64(std::uint64_t v) {
    char buffer[24];
    std::snprintf(buffer, sizeof buffer, "%016" PRIx64, v);
    return buffer;
}

std::string stamp() {
    SYSTEMTIME t{};
    GetLocalTime(&t);
    char buffer[64];
    std::snprintf(buffer, sizeof buffer, "%04u%02u%02u-%02u%02u%02u-%03u-pid%lu", t.wYear, t.wMonth, t.wDay, t.wHour,
                  t.wMinute, t.wSecond, t.wMilliseconds, GetCurrentProcessId());
    return buffer;
}

bool write_file(const std::filesystem::path& path, const void* data, std::size_t size) {
    std::error_code ec;
    std::filesystem::create_directories(path.parent_path(), ec);
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write(static_cast<const char*>(data), std::streamsize(size));
    return bool(out);
}

std::uint64_t spirv_hash(const void* code, std::size_t bytes) {
    State& s = state();
    GFSDK_Aftermath_SpirvCode spirv{};
    spirv.pData = code;
    spirv.size = std::uint32_t(bytes);
    GFSDK_Aftermath_ShaderBinaryHash hash{};
    if (!GFSDK_Aftermath_SUCCEED(s.lib.GetShaderHashSpirv(GFSDK_Aftermath_Version_API, &spirv, &hash))) return 0;
    return hash.hash;
}

const char* status_name(GFSDK_Aftermath_Device_Status status) {
    switch (status) {
    case GFSDK_Aftermath_Device_Status_Active: return "Active";
    case GFSDK_Aftermath_Device_Status_Timeout: return "Timeout";
    case GFSDK_Aftermath_Device_Status_OutOfMemory: return "OutOfMemory";
    case GFSDK_Aftermath_Device_Status_PageFault: return "PageFault";
    case GFSDK_Aftermath_Device_Status_Stopped: return "Stopped";
    case GFSDK_Aftermath_Device_Status_Reset: return "Reset";
    case GFSDK_Aftermath_Device_Status_DmaFault: return "DmaFault";
    case GFSDK_Aftermath_Device_Status_DeviceRemovedNoGpuFault: return "DeviceRemovedNoGpuFault";
    default: return "Unknown";
    }
}

const char* fault_name(GFSDK_Aftermath_FaultType type) {
    switch (type) {
    case GFSDK_Aftermath_FaultType_AddressTranslationError: return "AddressTranslationError";
    case GFSDK_Aftermath_FaultType_IllegalAccessError: return "IllegalAccessError";
    default: return "Unknown";
    }
}

const char* access_name(GFSDK_Aftermath_AccessType type) {
    switch (type) {
    case GFSDK_Aftermath_AccessType_Read: return "Read";
    case GFSDK_Aftermath_AccessType_Write: return "Write";
    case GFSDK_Aftermath_AccessType_Atomic: return "Atomic";
    default: return "Unknown";
    }
}

const char* engine_name(GFSDK_Aftermath_Engine engine) {
    switch (engine) {
    case GFSDK_Aftermath_Engine_Graphics: return "Graphics";
    case GFSDK_Aftermath_Engine_GraphicsCompute: return "GraphicsCompute";
    case GFSDK_Aftermath_Engine_Display: return "Display";
    case GFSDK_Aftermath_Engine_CopyEngine: return "CopyEngine";
    case GFSDK_Aftermath_Engine_VideoDecoder: return "VideoDecoder";
    case GFSDK_Aftermath_Engine_VideoEncoder: return "VideoEncoder";
    case GFSDK_Aftermath_Engine_Other: return "Other";
    default: return "Unknown";
    }
}

const char* client_name(GFSDK_Aftermath_Client client) {
    switch (client) {
    case GFSDK_Aftermath_Client_HostInterface: return "HostInterface";
    case GFSDK_Aftermath_Client_FrontEnd: return "FrontEnd";
    case GFSDK_Aftermath_Client_PrimitiveDistributor: return "PrimitiveDistributor";
    case GFSDK_Aftermath_Client_GraphicsProcessingCluster: return "GraphicsProcessingCluster";
    case GFSDK_Aftermath_Client_PolymorphEngine: return "PolymorphEngine";
    case GFSDK_Aftermath_Client_RasterEngine: return "RasterEngine";
    case GFSDK_Aftermath_Client_Rasterizer2D: return "Rasterizer2D";
    case GFSDK_Aftermath_Client_RenderOutputUnit: return "RenderOutputUnit";
    case GFSDK_Aftermath_Client_TextureProcessingCluster: return "TextureProcessingCluster";
    case GFSDK_Aftermath_Client_CopyEngine: return "CopyEngine";
    case GFSDK_Aftermath_Client_VideoDecoder: return "VideoDecoder";
    case GFSDK_Aftermath_Client_VideoEncoder: return "VideoEncoder";
    case GFSDK_Aftermath_Client_Other: return "Other";
    default: return "Unknown";
    }
}

const char* shader_type_name(GFSDK_Aftermath_ShaderType type) {
    switch (type) {
    case GFSDK_Aftermath_ShaderType_Vertex: return "vertex";
    case GFSDK_Aftermath_ShaderType_Tessellation_Control: return "hull";
    case GFSDK_Aftermath_ShaderType_Tessellation_Evaluation: return "domain";
    case GFSDK_Aftermath_ShaderType_Geometry: return "geometry";
    case GFSDK_Aftermath_ShaderType_Fragment: return "pixel";
    case GFSDK_Aftermath_ShaderType_Compute: return "compute";
    case GFSDK_Aftermath_ShaderType_Mesh: return "mesh";
    case GFSDK_Aftermath_ShaderType_Task: return "task";
    default: return type >= GFSDK_Aftermath_ShaderType_RayTracing_RayGeneration &&
                            type <= GFSDK_Aftermath_ShaderType_RayTracing_Internal
                        ? "ray tracing"
                        : "unknown";
    }
}

const char* residency_name(GFSDK_Aftermath_ResourceResidency residency) {
    switch (residency) {
    case GFSDK_Aftermath_ResourceResidency_FullyResident: return "resident";
    case GFSDK_Aftermath_ResourceResidency_Evicted: return "evicted";
    case GFSDK_Aftermath_ResourceResidency_MemoryFreed: return "memory freed";
    case GFSDK_Aftermath_ResourceResidency_MemoryUnbound: return "memory unbound";
    default: return "unknown";
    }
}

std::string shader_name(std::uint64_t hash) {
    State& s = state();
    std::lock_guard lock(s.shaders_mutex);
    const auto it = s.shaders.find(hash);
    return it == s.shaders.end() ? "unregistered" : it->second.name;
}

// ---- decoding ----

struct DecodeContext {
    std::map<std::uint64_t, std::vector<std::uint32_t>> binaries;  // alive until the decoder is destroyed
    std::vector<std::string> written;
};

// The module's SPIR-V for the decoder: the kept copy, else <LEONIDA_SHADERS>/<name>.spv when its hash matches (a
// patched module, such as a depth24 pixel shader, does not).
std::vector<std::uint32_t> find_binary(std::uint64_t hash) {
    State& s = state();
    std::string name;
    {
        std::lock_guard lock(s.shaders_mutex);
        const auto it = s.shaders.find(hash);
        if (it == s.shaders.end()) return {};
        if (!it->second.words.empty()) return it->second.words;
        name = it->second.name;
    }
    if (s.shader_dir.empty() || name.empty() || name.front() == '(') return {};
    const std::filesystem::path path = s.shader_dir / from_utf8(name + ".spv");
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) return {};
    const auto size = std::size_t(file.tellg());
    if (!size || size % 4 || size > (256u << 20)) return {};
    std::vector<std::uint32_t> words(size / 4);
    file.seekg(0);
    file.read(reinterpret_cast<char*>(words.data()), std::streamsize(size));
    if (!file || spirv_hash(words.data(), size) != hash) return {};
    return words;
}

void GFSDK_AFTERMATH_CALL lookup_debug_info(const GFSDK_Aftermath_ShaderDebugInfoIdentifier* id,
                                            PFN_GFSDK_Aftermath_SetData set, void*) {
    State& s = state();
    std::lock_guard lock(s.debug_mutex);
    const auto it = s.debug_info.find({id->id[0], id->id[1]});
    if (it != s.debug_info.end()) set(it->second.data(), std::uint32_t(it->second.size()));
}

void GFSDK_AFTERMATH_CALL lookup_shader(const GFSDK_Aftermath_ShaderBinaryHash* hash, PFN_GFSDK_Aftermath_SetData set,
                                        void* user) {
    auto& context = *static_cast<DecodeContext*>(user);
    auto it = context.binaries.find(hash->hash);
    if (it == context.binaries.end()) {
        it = context.binaries.emplace(hash->hash, find_binary(hash->hash)).first;
        if (!it->second.empty()) {
            // Next to the dump, for nv-aftermath-format -b <dir> and Nsight Graphics.
            const auto path = state().dir / (hex64(hash->hash) + ".spv");
            if (write_file(path, it->second.data(), it->second.size() * 4)) context.written.push_back(narrow(path));
        }
    }
    if (!it->second.empty()) set(it->second.data(), std::uint32_t(it->second.size() * 4));
}

std::string lower(std::string_view text) {
    std::string out(text);
    for (char& c : out) c = char(std::tolower(static_cast<unsigned char>(c)));
    return out;
}

std::string scalar(const Json& value) {
    switch (value.kind) {
    case Json::Kind::string: return value.string;
    case Json::Kind::boolean: return value.boolean ? "true" : "false";
    case Json::Kind::number: {
        char buffer[32];
        if (value.number >= 0 && value.number < 18446744073709551616.0 && value.number == double(std::uint64_t(value.number)))
            std::snprintf(buffer, sizeof buffer, "%" PRIu64, std::uint64_t(value.number));
        else
            std::snprintf(buffer, sizeof buffer, "%g", value.number);
        return buffer;
    }
    default: return {};
    }
}

// One line per object: its scalar fields, and the scalars of objects nested one level down.
std::string flatten(const Json& object) {
    std::string out;
    const auto add = [&out](std::string_view key, std::string value) {
        for (char& c : value)
            if (c == '\n' || c == '\r') c = ' ';
        if (!value.empty()) out += (out.empty() ? "" : " ") + std::string(key) + "=" + value;
    };
    for (const auto& [key, value] : object.object) {
        if (value.kind == Json::Kind::object)
            for (const auto& [inner_key, inner] : value.object) add(key + "." + inner_key, scalar(inner));
        else if (value.kind != Json::Kind::array)
            add(key, scalar(value));
    }
    return out;
}

// The decoder writes 64-bit values (shader hashes, pointers) as JSON integers, which a double cannot hold. Every
// unsigned integer literal of 16 or more digits is quoted, so it reaches the reader exactly, as a string.
std::string quote_big_integers(std::string_view json) {
    std::string out;
    out.reserve(json.size() + 256);
    bool in_string = false, escaped = false;
    for (std::size_t i = 0; i < json.size();) {
        const char c = json[i];
        if (in_string) {
            out += c;
            ++i;
            if (escaped) escaped = false;
            else if (c == '\\') escaped = true;
            else if (c == '"') in_string = false;
            continue;
        }
        if (c == '"') {
            in_string = true;
            out += c;
            ++i;
            continue;
        }
        if (c >= '0' && c <= '9') {
            std::size_t j = i;
            while (j < json.size() && json[j] >= '0' && json[j] <= '9') ++j;
            const bool integer = j == json.size() || (json[j] != '.' && json[j] != 'e' && json[j] != 'E');
            const bool quote = integer && j - i >= 16 && (i == 0 || json[i - 1] != '-');
            if (quote) out += '"';
            out.append(json.substr(i, j - i));
            if (quote) out += '"';
            i = j;
            continue;
        }
        out += c;
        ++i;
    }
    return out;
}

const Json* field(const Json& object, std::string_view needle) {
    for (const auto& [key, value] : object.object)
        if (lower(key).find(needle) != std::string::npos) return &value;
    return nullptr;
}

// Records under a key naming warps ("Faulted Warps", "Active Warps"), and the shader table entries ("Shader infos":
// objects with a "Shader name" and a "Shader hash"). The SDK does not specify the JSON layout; this is the one the
// 2026.3 decoder writes, read loosely.
void walk(const Json& node, std::string_view key, std::vector<std::pair<std::string, const Json*>>& warps,
          std::map<std::string, std::uint64_t>& shader_hashes, int depth = 0) {
    if (depth > 16) return;
    if (node.kind == Json::Kind::array) {
        if (lower(key).find("warp") != std::string::npos)
            for (const Json& item : node.array)
                if (item.kind == Json::Kind::object) warps.emplace_back(std::string(key), &item);
        for (const Json& item : node.array) walk(item, key, warps, shader_hashes, depth + 1);
    } else if (node.kind == Json::Kind::object) {
        const Json* name = node.find("Shader name");
        const Json* hash = node.find("Shader hash");
        if (name && hash) shader_hashes[scalar(*name)] = std::strtoull(scalar(*hash).c_str(), nullptr, 0);
        for (const auto& [k, v] : node.object) walk(v, k, warps, shader_hashes, depth + 1);
    }
}

void decode_and_report() noexcept {
    State& s = state();
    std::vector<std::uint8_t> dump;
    std::filesystem::path dump_path;
    {
        std::lock_guard lock(s.dump_mutex);
        if (s.decoded || !s.have_dump) return;
        dump = s.dump;
        dump_path = s.dump_path;
    }
    std::string summary;
    const auto line = [&summary](const char* format, auto... args) {
        char buffer[1024];
        std::snprintf(buffer, sizeof buffer, format, args...);
        summary += "[rt] aftermath: ";
        summary += buffer;
        summary += '\n';
    };
    try {
        GFSDK_Aftermath_GpuCrashDump_Decoder decoder = nullptr;
        const auto created =
            s.lib.CreateDecoder(GFSDK_Aftermath_Version_API, dump.data(), std::uint32_t(dump.size()), &decoder);
        if (!GFSDK_Aftermath_SUCCEED(created)) {
            line("GPU crash dump %s could not be decoded (0x%x)", narrow(dump_path).c_str(), unsigned(created));
        } else {
            GFSDK_Aftermath_GpuCrashDump_DeviceInfo device{};
            const bool have_device = GFSDK_Aftermath_SUCCEED(s.lib.GetDeviceInfo(decoder, &device));
            line("GPU crash dump %s (%zu bytes): device status %s, adapter reset %s, engine reset %s",
                 narrow(dump_path).c_str(), dump.size(), have_device ? status_name(device.status) : "unknown",
                 have_device && device.adapterReset ? "yes" : "no", have_device && device.engineReset ? "yes" : "no");

            // Active shaders, by our module names.
            std::uint32_t count = 0;
            std::vector<GFSDK_Aftermath_GpuCrashDump_ShaderInfo> shaders;
            if (GFSDK_Aftermath_SUCCEED(s.lib.GetActiveShadersInfoCount(decoder, &count)) && count) {
                shaders.resize(count);
                if (!GFSDK_Aftermath_SUCCEED(s.lib.GetActiveShadersInfo(decoder, count, shaders.data()))) shaders.clear();
            }
            struct Named {
                std::string name;    // our module name
                std::string binary;  // Aftermath shader binary hash, hex
            };
            std::unordered_map<std::uint64_t, Named> by_dump_hash;  // ShaderInfo::shaderHash -> module
            std::vector<std::string> active;
            for (const auto& info : shaders) {
                GFSDK_Aftermath_ShaderBinaryHash hash{};
                const bool mapped = GFSDK_Aftermath_SUCCEED(s.lib.GetShaderHashForShaderInfo(decoder, &info, &hash));
                const std::string name = info.isInternal ? "(driver internal)"
                                         : mapped        ? shader_name(hash.hash)
                                                         : "unmapped";
                by_dump_hash[info.shaderHash] = {name, mapped ? hex64(hash.hash) : "-"};
                char buffer[512];
                std::snprintf(buffer, sizeof buffer, "%s %s hash=%s", shader_type_name(info.shaderType), name.c_str(),
                              mapped ? hex64(hash.hash).c_str() : "-");
                active.emplace_back(buffer);
            }

            // Warps, from the JSON (the C API has no per-warp pc). Shader mapping info asks lookup_shader for binaries.
            DecodeContext context;
            std::uint32_t json_size = 0;
            std::string json;
            const auto generated = s.lib.GenerateJSON(
                decoder, GFSDK_Aftermath_GpuCrashDumpDecoderFlags_ALL_INFO,
                GFSDK_Aftermath_GpuCrashDumpFormatterFlags_UTF8_OUTPUT, lookup_debug_info, lookup_shader, nullptr,
                &context, &json_size);
            if (GFSDK_Aftermath_SUCCEED(generated) && json_size) {
                json.resize(json_size);
                if (!GFSDK_Aftermath_SUCCEED(s.lib.GetJSON(decoder, json_size, json.data()))) json.clear();
                while (!json.empty() && json.back() == '\0') json.pop_back();
            }
            std::filesystem::path json_path = dump_path;
            json_path.replace_extension(".json");
            if (!json.empty()) write_file(json_path, json.data(), json.size());

            Json root;
            std::vector<std::pair<std::string, const Json*>> warps;
            std::map<std::string, std::uint64_t> shader_hashes;  // the dump's shader name ("compute_02") -> dump hash
            if (!json.empty() && JsonReader(quote_big_integers(json)).parse(root))
                walk(root, {}, warps, shader_hashes);
            std::stable_partition(warps.begin(), warps.end(),
                                  [](const auto& warp) { return lower(warp.first).find("fault") != std::string::npos; });
            if (warps.empty()) line("no warp records in the dump (%s)", json.empty() ? "no JSON" : "see the JSON");
            std::size_t shown = 0;
            for (const auto& [key, warp] : warps) {
                if (shown++ == 16) {
                    line("... %zu more warp records in %s", warps.size() - 16, narrow(json_path).c_str());
                    break;
                }
                // "GPU PC Address": "<dump shader name> @ 0x<pc>", the name keyed in the shader table.
                const Json* pc_field = field(*warp, "pc address");
                const std::string address = pc_field ? scalar(*pc_field) : std::string();
                const auto at = address.find(" @ ");
                const std::string dump_name = at == std::string::npos ? address : address.substr(0, at);
                const std::string pc = at == std::string::npos ? "?" : address.substr(at + 3);
                Named module{"?", "-"};
                if (const auto h = shader_hashes.find(dump_name); h != shader_hashes.end())
                    if (const auto m = by_dump_hash.find(h->second); m != by_dump_hash.end()) module = m->second;
                const Json* mapping = field(*warp, "mapping");
                const Json* warp_count = field(*warp, "count");
                const Json* fault_name = field(*warp, "fault name");
                const bool faulted = lower(key).find("fault") != std::string::npos;
                if (!pc_field) {
                    line("%s: %s", key.c_str(), flatten(*warp).c_str());
                } else if (faulted) {
                    line("faulting shader %s (hash %s, %s) pc %s spirv %s: %s", module.name.c_str(),
                         module.binary.c_str(), dump_name.c_str(), pc.c_str(),
                         mapping ? scalar(*mapping).c_str() : "-", fault_name ? scalar(*fault_name).c_str() : key.c_str());
                } else {
                    line("%s: shader %s (hash %s, %s) pc %s spirv %s warps %s", key.c_str(), module.name.c_str(),
                         module.binary.c_str(), dump_name.c_str(), pc.c_str(),
                         mapping ? scalar(*mapping).c_str() : "-", warp_count ? scalar(*warp_count).c_str() : "?");
                }
            }

            GFSDK_Aftermath_GpuCrashDump_PageFaultInfo fault{};
            if (GFSDK_Aftermath_SUCCEED(s.lib.GetPageFaultInfo(decoder, &fault))) {
                line("page fault va=0x%016" PRIx64 " fault=%s access=%s engine=%s client=%s resources=%u",
                     fault.faultingGpuVA, fault_name(fault.faultType), access_name(fault.accessType),
                     engine_name(fault.engine), client_name(fault.client), fault.resourceInfoCount);
                std::vector<GFSDK_Aftermath_GpuCrashDump_ResourceInfo> resources(fault.resourceInfoCount);
                if (!resources.empty() &&
                    GFSDK_Aftermath_SUCCEED(s.lib.GetPageFaultResourceInfo(decoder, fault.resourceInfoCount,
                                                                           resources.data())))
                    for (std::size_t i = 0; i < resources.size(); ++i) {
                        const auto& r = resources[i];
                        line("page fault resource[%zu] va=0x%016" PRIx64 " size=0x%" PRIx64
                             " %ux%ux%u mips=%u format=%u handle=0x%016" PRIx64
                             " name=\"%s\" %s%s%s%s",
                             i, r.gpuVa, r.size, r.width, r.height, r.depth, r.mipLevels, r.format, r.apiResource,
                             r.debugName, residency_name(r.residency), r.bWasDestroyed ? ", destroyed" : "",
                             r.bPlacedResource ? ", placed" : "", r.bIsBufferHeap ? ", buffer heap" : "");
                    }
            } else {
                line("no page fault information in the dump");
            }
            for (std::size_t i = 0; i < active.size(); ++i) line("active shader[%zu] %s", i, active[i].c_str());
            {
                // The hash -> name table next to the dump, for matching it offline (nv-aftermath-format, Nsight).
                std::string table;
                std::lock_guard lock(s.shaders_mutex);
                for (const auto& [hash, shader] : s.shaders) table += hex64(hash) + '\t' + shader.name + '\n';
                std::filesystem::path table_path = dump_path;
                table_path.replace_extension(".shaders.tsv");
                const bool table_written = write_file(table_path, table.data(), table.size());
                line("%llu SPIR-V modules registered (%zu distinct, table %s), %llu not hashed",
                     static_cast<unsigned long long>(s.registered.load()), s.shaders.size(),
                     table_written ? narrow(table_path).c_str() : "not written",
                     static_cast<unsigned long long>(s.hash_failures.load()));
            }
            for (const auto& path : context.written) line("wrote %s", path.c_str());
            if (!json.empty()) line("decoded JSON %s", narrow(json_path).c_str());
            s.lib.DestroyDecoder(decoder);
        }
    } catch (...) {
        line("decoding failed");
    }
    std::fputs(summary.c_str(), stderr);
    std::fflush(stderr);
    std::lock_guard lock(s.dump_mutex);
    s.summary = std::move(summary);
    s.decoded = true;
    s.dump_cv.notify_all();
}

// ---- callbacks (free-threaded, on Aftermath's threads) ----

void GFSDK_AFTERMATH_CALL on_crash_dump(const void* data, const std::uint32_t size, void*) {
    State& s = state();
    const auto path = s.dir / (stamp() + ".nv-gpudmp");
    const bool written = write_file(path, data, size);
    std::fprintf(stderr, "[rt] aftermath: GPU crash dump, %u bytes%s %s\n", size, written ? " written to" : ", cannot write",
                 narrow(path).c_str());
    {
        std::lock_guard lock(s.dump_mutex);
        if (s.have_dump) return;  // one decode per process: the first crash is the one that matters
        const auto bytes = static_cast<const std::uint8_t*>(data);
        s.dump.assign(bytes, bytes + size);
        s.dump_path = path;
        s.have_dump = true;
    }
    // Decode once Aftermath is done (the deferred shader debug info callbacks may still follow this one).
    std::thread([] {
        State& st = state();
        const auto start = Clock::now();
        for (;;) {
            GFSDK_Aftermath_CrashDump_Status status = GFSDK_Aftermath_CrashDump_Status_Unknown;
            st.lib.GetCrashDumpStatus(&status);
            if (status == GFSDK_Aftermath_CrashDump_Status_Finished ||
                status == GFSDK_Aftermath_CrashDump_Status_CollectingDataFailed ||
                status == GFSDK_Aftermath_CrashDump_Status_NotStarted || Clock::now() - start > std::chrono::seconds(5))
                break;
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        decode_and_report();
    }).detach();
}

void GFSDK_AFTERMATH_CALL on_shader_debug_info(const void* data, const std::uint32_t size, void*) {
    State& s = state();
    GFSDK_Aftermath_ShaderDebugInfoIdentifier id{};
    if (!GFSDK_Aftermath_SUCCEED(s.lib.GetShaderDebugInfoIdentifier(GFSDK_Aftermath_Version_API, data, size, &id)))
        return;
    write_file(s.dir / ("shader-" + hex64(id.id[0]) + "-" + hex64(id.id[1]) + ".nvdbg"), data, size);
    const auto bytes = static_cast<const std::uint8_t*>(data);
    std::lock_guard lock(s.debug_mutex);
    s.debug_info[{id.id[0], id.id[1]}].assign(bytes, bytes + size);
}

void GFSDK_AFTERMATH_CALL on_description(PFN_GFSDK_Aftermath_AddGpuCrashDumpDescription add, void*) {
    add(GFSDK_Aftermath_GpuCrashDumpDescriptionKey_ApplicationName, "Leonida gta6rt");
    add(GFSDK_Aftermath_GpuCrashDumpDescriptionKey_ApplicationVersion, LEONIDA_AFTERMATH_APP_VERSION);
}

VKAPI_ATTR VkResult VKAPI_CALL create_shader_module(VkDevice device, const VkShaderModuleCreateInfo* info,
                                                    const VkAllocationCallbacks* allocator, VkShaderModule* module) {
    const VkResult result = state().real_create.load(std::memory_order_relaxed)(device, info, allocator, module);
    if (result == VK_SUCCESS && info) register_spirv(info->pCode, info->codeSize, detail::shader_name);
    return result;
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL get_device_proc_addr(VkDevice device, const char* name) {
    const PFN_vkVoidFunction function = state().real_get.load(std::memory_order_relaxed)(device, name);
    if (!function || !name || std::strcmp(name, "vkCreateShaderModule") != 0) return function;
    return reinterpret_cast<PFN_vkVoidFunction>(
        wrap_create_shader_module(reinterpret_cast<PFN_vkCreateShaderModule>(function)));
}

void device_lost() noexcept { wait_for_dump(); }

}  // namespace

bool available() noexcept { return true; }

bool enable() noexcept {
    static const bool on = []() noexcept {
        if (!env_is(L"LEONIDA_AFTERMATH", L'1')) return false;
        try {
            State& s = state();
            std::wstring dll = env_w(L"LEONIDA_AFTERMATH_DLL");
            if (dll.empty()) {
                const std::filesystem::path built = from_utf8(LEONIDA_AFTERMATH_DLL_DEFAULT);
                std::error_code ec;
                dll = !built.empty() && std::filesystem::exists(built, ec) ? built.wstring()
                                                                           : L"GFSDK_Aftermath_Lib.x64.dll";
            }
            if (!s.lib.load(dll)) {
                std::fprintf(stderr, "[rt] aftermath: cannot load %s (error %lu); crash dumps are off\n",
                             narrow(dll).c_str(), GetLastError());
                return false;
            }
            const std::wstring dir = env_w(L"LEONIDA_AFTERMATH_DIR");
            s.dir = dir.empty() ? std::filesystem::temp_directory_path() / L"leonida-aftermath" : std::filesystem::path(dir);
            std::error_code ec;
            std::filesystem::create_directories(s.dir, ec);
            s.shader_dir = env_w(L"LEONIDA_SHADERS");
            s.keep_spirv = env_is(L"LEONIDA_AFTERMATH_KEEP_SPIRV", L'1');
            if (env_is(L"LEONIDA_AFTERMATH_SHADER_ERRORS", L'1'))
                s.extra_flags |= VK_DEVICE_DIAGNOSTICS_CONFIG_ENABLE_SHADER_ERROR_REPORTING_BIT_NV;
            const auto result = s.lib.EnableGpuCrashDumps(
                GFSDK_Aftermath_Version_API, GFSDK_Aftermath_GpuCrashDumpWatchedApiFlags_Vulkan,
                GFSDK_Aftermath_GpuCrashDumpFeatureFlags_DeferDebugInfoCallbacks, on_crash_dump, on_shader_debug_info,
                on_description, nullptr, nullptr);
            if (!GFSDK_Aftermath_SUCCEED(result)) {
                std::fprintf(stderr, "[rt] aftermath: GFSDK_Aftermath_EnableGpuCrashDumps failed (0x%x)\n",
                             unsigned(result));
                return false;
            }
            s.enabled.store(true, std::memory_order_release);
            device_lost_hook.store(device_lost, std::memory_order_release);
            std::fprintf(stderr, "[rt] aftermath: GPU crash dumps on (%s), written to %s%s\n", narrow(dll).c_str(),
                         narrow(s.dir).c_str(), s.keep_spirv ? ", SPIR-V kept" : "");
            return true;
        } catch (...) {
            std::fprintf(stderr, "[rt] aftermath: setup failed\n");
            return false;
        }
    }();
    return on;
}

bool enabled() noexcept { return state().enabled.load(std::memory_order_acquire); }

VkDeviceDiagnosticsConfigFlagsNV extra_config_flags() noexcept { return enabled() ? state().extra_flags : 0; }

PFN_vkCreateShaderModule wrap_create_shader_module(PFN_vkCreateShaderModule real) noexcept {
    if (!enabled() || !real || real == create_shader_module) return real;
    PFN_vkCreateShaderModule expected = nullptr;
    State& s = state();
    if (!s.real_create.compare_exchange_strong(expected, real) && expected != real) return real;
    return create_shader_module;
}

PFN_vkGetDeviceProcAddr wrap_get_device_proc_addr(PFN_vkGetDeviceProcAddr real) noexcept {
    if (!enabled() || !real || real == get_device_proc_addr) return real;
    PFN_vkGetDeviceProcAddr expected = nullptr;
    State& s = state();
    if (!s.real_get.compare_exchange_strong(expected, real) && expected != real) return real;
    return get_device_proc_addr;
}

void register_spirv(const void* code, std::size_t bytes, const char* name) noexcept {
    if (!enabled() || !code || !bytes) return;
    try {
        State& s = state();
        const std::uint64_t hash = spirv_hash(code, bytes);
        if (!hash) {
            s.hash_failures.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        s.registered.fetch_add(1, std::memory_order_relaxed);
        std::vector<std::uint32_t> words;
        // The runtime's own modules are few and small, and not in LEONIDA_SHADERS: always kept for the decoder.
        if (s.keep_spirv || (name && std::strncmp(name, "(runtime)", 9) == 0)) {
            const auto first = static_cast<const std::uint32_t*>(code);
            words.assign(first, first + bytes / 4);
        }
        std::lock_guard lock(s.shaders_mutex);
        auto& shader = s.shaders[hash];
        if (shader.name.empty() || name) shader.name = name ? name : "(runtime)";
        if (!words.empty()) shader.words = std::move(words);
    } catch (...) {
    }
}

void wait_for_dump() noexcept {
    if (!enabled()) return;
    State& s = state();
    std::lock_guard wait(s.wait_mutex);
    if (s.waited) return;
    s.waited = true;
    std::fprintf(stderr, "[rt] aftermath: device lost, waiting for the GPU crash dump (at most %lld s)\n",
                 static_cast<long long>(wait_limit.count()));
    const auto deadline = Clock::now() + wait_limit;
    GFSDK_Aftermath_CrashDump_Status status = GFSDK_Aftermath_CrashDump_Status_Unknown;
    for (;;) {
        {
            std::unique_lock lock(s.dump_mutex);
            if (s.dump_cv.wait_for(lock, std::chrono::milliseconds(50), [&s] { return s.decoded; })) break;
            if (Clock::now() >= deadline) break;
            if (status == GFSDK_Aftermath_CrashDump_Status_Finished && !s.have_dump) break;
        }
        s.lib.GetCrashDumpStatus(&status);
        if (status == GFSDK_Aftermath_CrashDump_Status_CollectingDataFailed) break;
    }
    std::lock_guard lock(s.dump_mutex);
    if (!s.decoded)
        std::fprintf(stderr, "[rt] aftermath: no GPU crash dump after the wait (status %u%s)\n", unsigned(status),
                     s.have_dump ? ", dump written but not decoded yet" : "");
}

std::string last_dump_path() {
    State& s = state();
    std::lock_guard lock(s.dump_mutex);
    return s.have_dump ? narrow(s.dump_path) : std::string();
}

std::string last_summary() {
    State& s = state();
    std::lock_guard lock(s.dump_mutex);
    return s.summary;
}

}  // namespace leonida::aftermath

#else  // !LEONIDA_HAVE_AFTERMATH

namespace leonida::aftermath {

bool available() noexcept { return false; }
bool enable() noexcept { return false; }
bool enabled() noexcept { return false; }
VkDeviceDiagnosticsConfigFlagsNV extra_config_flags() noexcept { return 0; }
PFN_vkCreateShaderModule wrap_create_shader_module(PFN_vkCreateShaderModule real) noexcept { return real; }
PFN_vkGetDeviceProcAddr wrap_get_device_proc_addr(PFN_vkGetDeviceProcAddr real) noexcept { return real; }
void register_spirv(const void*, std::size_t, const char*) noexcept {}
void wait_for_dump() noexcept {}
std::string last_dump_path() { return {}; }
std::string last_summary() { return {}; }

}  // namespace leonida::aftermath

#endif
