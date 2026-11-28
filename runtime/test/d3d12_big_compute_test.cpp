// Large pipelines that a submission compiles (LEONIDA_SERIAL_BIG_COMPILES, LEONIDA_BIG_COMPILE_BUDGET_MB), through the D3D12
// layer on a GPU: real compute modules of more than 128 KiB (big_compute_module.h), small ones, and graphics pipelines of more than
// 1 MiB. Both settings are read once for each process, so the test runs once for each: with a fixed budget, one that holds
// nothing, the one read from the machine, one at a time and without a limit. Every run
//   - stores what the CPU computes for each dispatch, and draws the graphics pipelines' cells;
//   - compiles as many large ones at once as the setting allows and no more, and the small ones meanwhile;
//   - gives a pipeline the driver refuses the result it has always had, and its place to the next;
//   - compiles each pipeline once when two queues need the same ones;
//   - ends when the device's last references go while its workers compile and wait for memory.
// `warmup`: a submission's compiles go before those of the warm-up thread, which waits without holding a pipeline state.
// `eager`: pipeline states compiled when they are created (LEONIDA_LAZY_COMPUTE=0), by several threads, keep the budget too.
// The calls that create pipelines are watched, and held back, through the device's vkGetDeviceProcAddr.
// Self-contained: builds its SPIR-V in memory. Set LEONIDA_VK_VALIDATION=1 to run under VK_LAYER_KHRONOS_validation.
#include "big_compute_module.h"
#include "vk_test_device.h"
#include "d3d12_test_shaders.h"
#include "test_temp_dir.h"
#include "../shims/d3d12_com.h"
#include "../shims/d3d12_pipeline_compile.h"
#include "../src/large_module_gate.h"
#include "../src/shader_library.h"

#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

using namespace std::chrono_literals;
using d3d12test::Ref;
using leonida::ReplayResult;
using bigmodule::Words;
using bigmodule::op;

constexpr unsigned kBig = 6, kSmall = 4, kCompute = kBig + kSmall, kGraphics = 2;
constexpr std::size_t kBigBytes = 132 * 1024, kSmallBytes = 12 * 1024;
constexpr unsigned kExtent = 4;  // the render target; each graphics pipeline draws one 2x2 cell
constexpr unsigned kWords = bigmodule::kLoadBase + bigmodule::kLoadWords;

#define CHECK(expr)                                                                  \
    do {                                                                             \
        if (!(expr)) {                                                               \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #expr);     \
            return false;                                                            \
        }                                                                            \
    } while (0)

// ---- What the settings of this process say. ----

std::string setting(const char* name) {
    char text[64]{};
    std::size_t length = 0;
    return getenv_s(&length, text, name) == 0 && length > 1 ? text : "";
}
const bool serial = setting("LEONIDA_SERIAL_BIG_COMPILES") == "1";
const bool unlimited = setting("LEONIDA_SERIAL_BIG_COMPILES") == "0";
const bool budgeted = !serial && !unlimited;
const auto fixed_budget = leonida::large_compile_budget(setting("LEONIDA_BIG_COMPILE_BUDGET_MB").c_str()).fixed;

// ---- The driver's pipeline calls, watched. ----

struct Start {
    std::thread::id thread;
    int program = -1;  // compute: the index of its module
    bool large = false;
};

struct Hooks {
    std::mutex mutex;
    std::condition_variable changed;
    std::map<std::uint64_t, int> programs;             // the hash of a compute module's words: its index
    std::map<VkShaderModule, std::pair<std::size_t, int>> modules;  // bytes, program
    bool closed = false;  // large creations wait while it is
    std::chrono::milliseconds linger{0};  // and each takes at least this long, as does what follows it on its thread
    std::vector<Start> started;
    int large_in = 0, most_large = 0, small_done = 0, large_compute_in = 0, most_large_compute = 0;
    std::uint64_t reserved = 0;  // the estimates of the large ones in the driver
    int over = 0;                // how often several were in past the budget, or two where one is allowed
    int fail_large = -1;         // the large compute create (counted from 0) the driver refuses
    int large_compute = 0;
    bool track = false;  // threads that create a pipeline are counted until they end
    int live = 0, in_flight = 0, cache_reads = 0;
    bool cache_read_during_create = false, cache_read_with_workers = false;

    void reset() {
        std::lock_guard lock(mutex);
        closed = track = cache_read_during_create = cache_read_with_workers = false;
        started.clear();
        most_large = small_done = most_large_compute = over = large_compute = cache_reads = 0;
        fail_large = -1;
        linger = 0ms;
    }
    void open() {
        std::lock_guard lock(mutex);
        closed = false;
        changed.notify_all();
    }
    template<typename Until>
    bool wait(Until&& until) {
        std::unique_lock lock(mutex);
        return changed.wait_for(lock, 20s, until);
    }
    bool wait_large(int count) {
        return wait([&] { return large_in >= count; });
    }
    std::size_t creates() {
        std::lock_guard lock(mutex);
        return started.size();
    }
} hooks;

struct Tracked {
    bool on = false;
    ~Tracked() {
        if (!on) return;
        std::lock_guard lock(hooks.mutex);
        --hooks.live;
    }
};
thread_local Tracked tracked;

PFN_vkGetInstanceProcAddr real_instance_proc = nullptr;
PFN_vkGetDeviceProcAddr real_device_proc = nullptr;
PFN_vkCreateShaderModule real_create_module = nullptr;
PFN_vkDestroyShaderModule real_destroy_module = nullptr;
PFN_vkCreateGraphicsPipelines real_create_graphics = nullptr;
PFN_vkCreateComputePipelines real_create_compute = nullptr;
PFN_vkGetPipelineCacheData real_cache_data = nullptr;

std::uint64_t hash(const std::uint32_t* words, std::size_t count) {
    std::uint64_t h = 0xcbf29ce484222325u;
    for (std::size_t i = 0; i < count; ++i) h = (h ^ words[i]) * 0x100000001b3u;
    return h;
}

struct Creation {
    bool large = false, compute = false, refused = false;
    std::uint64_t estimate = 0;
};

Creation enter_create(bool compute, std::initializer_list<VkShaderModule> modules) {
    std::unique_lock lock(hooks.mutex);
    Creation c;
    c.compute = compute;
    std::size_t bytes = 0;
    Start start{std::this_thread::get_id(), -1, false};
    for (const VkShaderModule module : modules) {
        const auto found = hooks.modules.find(module);
        if (found == hooks.modules.end()) continue;
        bytes += found->second.first;
        if (compute) start.program = found->second.second;
    }
    c.large = start.large = bytes > (compute ? 128 * 1024 : leonida::PipelineCompiles::large_spirv_bytes - 1);
    c.estimate = leonida::large_module_estimate(bytes);
    ++hooks.in_flight;
    hooks.started.push_back(start);
    if (hooks.track && !tracked.on) {
        tracked.on = true;
        ++hooks.live;
    }
    if (c.large) {
        ++hooks.large_in;
        if (compute) ++hooks.large_compute_in;
        hooks.most_large = std::max(hooks.most_large, hooks.large_in);
        hooks.most_large_compute = std::max(hooks.most_large_compute, hooks.large_compute_in);
        hooks.reserved += c.estimate;
        // One large compute module at a time; with a budget, several only within it.
        if (serial && hooks.large_compute_in > 1) ++hooks.over;
        if (budgeted && fixed_budget && hooks.large_in > 1 && hooks.reserved > *fixed_budget) ++hooks.over;
        if (compute && hooks.large_compute++ == hooks.fail_large) c.refused = true;
    }
    hooks.changed.notify_all();
    if (c.large) hooks.changed.wait(lock, [] { return !hooks.closed; });
    const auto linger = hooks.linger;
    lock.unlock();
    if (c.large && linger > 0ms) std::this_thread::sleep_for(linger);
    return c;
}

void leave_create(const Creation& c) {
    std::lock_guard lock(hooks.mutex);
    --hooks.in_flight;
    if (c.large) {
        --hooks.large_in;
        if (c.compute) --hooks.large_compute_in;
        hooks.reserved -= c.estimate;
    } else {
        ++hooks.small_done;
    }
    hooks.changed.notify_all();
}

VKAPI_ATTR VkResult VKAPI_CALL create_module(VkDevice device, const VkShaderModuleCreateInfo* info,
                                             const VkAllocationCallbacks* allocator, VkShaderModule* out) {
    const VkResult result = real_create_module(device, info, allocator, out);
    if (result == VK_SUCCESS) {
        std::lock_guard lock(hooks.mutex);
        const auto found = hooks.programs.find(hash(info->pCode, info->codeSize / 4));
        hooks.modules[*out] = {info->codeSize, found == hooks.programs.end() ? -1 : found->second};
    }
    return result;
}

// Called between the place a compile gives back and the next one its thread asks for: with a linger, time for another
// thread to take that place.
VKAPI_ATTR void VKAPI_CALL destroy_module(VkDevice device, VkShaderModule module, const VkAllocationCallbacks* allocator) {
    std::chrono::milliseconds linger;
    {
        std::lock_guard lock(hooks.mutex);
        linger = hooks.linger;
    }
    if (linger > 0ms) std::this_thread::sleep_for(linger);
    real_destroy_module(device, module, allocator);
}

VKAPI_ATTR VkResult VKAPI_CALL create_graphics(VkDevice device, VkPipelineCache cache, std::uint32_t count,
                                               const VkGraphicsPipelineCreateInfo* infos,
                                               const VkAllocationCallbacks* allocator, VkPipeline* out) {
    VkShaderModule modules[2]{};
    for (std::uint32_t i = 0; i < std::min(infos->stageCount, 2u); ++i) modules[i] = infos->pStages[i].module;
    const Creation c = enter_create(false, {modules[0], modules[1]});
    const VkResult result = real_create_graphics(device, cache, count, infos, allocator, out);
    leave_create(c);
    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL create_compute(VkDevice device, VkPipelineCache cache, std::uint32_t count,
                                              const VkComputePipelineCreateInfo* infos,
                                              const VkAllocationCallbacks* allocator, VkPipeline* out) {
    const Creation c = enter_create(true, {infos->stage.module});
    const VkResult result = c.refused ? VK_ERROR_UNKNOWN
                                      : real_create_compute(device, cache, count, infos, allocator, out);
    leave_create(c);
    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL cache_data(VkDevice device, VkPipelineCache cache, std::size_t* size, void* data) {
    {
        std::lock_guard lock(hooks.mutex);
        ++hooks.cache_reads;
        if (hooks.in_flight) hooks.cache_read_during_create = true;
        if (hooks.live > (tracked.on ? 1 : 0)) hooks.cache_read_with_workers = true;
    }
    return real_cache_data(device, cache, size, data);
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL device_proc(VkDevice device, const char* name) {
    const PFN_vkVoidFunction real = real_device_proc(device, name);
    if (!real) return nullptr;
    const auto watch = [&](const char* which, auto& kept, auto* mine) -> PFN_vkVoidFunction {
        if (std::strcmp(name, which)) return nullptr;
        kept = reinterpret_cast<std::remove_reference_t<decltype(kept)>>(real);
        return reinterpret_cast<PFN_vkVoidFunction>(mine);
    };
    if (const auto f = watch("vkCreateShaderModule", real_create_module, create_module)) return f;
    if (const auto f = watch("vkDestroyShaderModule", real_destroy_module, destroy_module)) return f;
    if (const auto f = watch("vkCreateGraphicsPipelines", real_create_graphics, create_graphics)) return f;
    if (const auto f = watch("vkCreateComputePipelines", real_create_compute, create_compute)) return f;
    if (const auto f = watch("vkGetPipelineCacheData", real_cache_data, cache_data)) return f;
    return real;
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL instance_proc(VkInstance instance, const char* name) {
    if (!std::strcmp(name, "vkGetDeviceProcAddr")) {
        real_device_proc = reinterpret_cast<PFN_vkGetDeviceProcAddr>(real_instance_proc(instance, name));
        return reinterpret_cast<PFN_vkVoidFunction>(device_proc);
    }
    return real_instance_proc(instance, name);
}

// ---- The shaders. ----

std::uint32_t bits(float f) { return std::bit_cast<std::uint32_t>(f); }

// 1.25 MiB of OpString, which compile to nothing: a graphics pipeline that is large.
void pad(Words& w, std::uint32_t first_id) {
    for (std::uint32_t i = 0; i < 5; ++i) {
        const std::size_t at = w.size();
        w.resize(at + 65535, 0x20202020u);
        w[at] = 65535u << 16 | 7;
        w[at + 1] = first_id + i;
        w.back() = 0;
    }
}

// A full-screen triangle from VertexIndex.
Words vertex_words() {
    Words w = {0x07230203u, 0x00010500u, 0u, 42u, 0u};
    op(w, 17, {1});                              // Capability Shader
    op(w, 14, {0, 1});                           // MemoryModel Logical GLSL450
    op(w, 15, {0, 4, 0x6e69616du, 0u, 10, 11});  // EntryPoint Vertex "main"
    op(w, 71, {10, 11, 0});                      // BuiltIn Position
    op(w, 71, {11, 11, 42});                     // BuiltIn VertexIndex
    op(w, 19, {1});
    op(w, 33, {2, 1});
    op(w, 22, {3, 32});
    op(w, 23, {6, 3, 4});
    op(w, 21, {15, 32, 1});
    op(w, 32, {7, 3, 6});
    op(w, 32, {8, 1, 15});
    op(w, 59, {7, 10, 3});
    op(w, 59, {8, 11, 1});
    op(w, 43, {15, 24, 1});
    op(w, 43, {15, 25, 2});
    op(w, 43, {3, 26, bits(2.0f)});
    op(w, 43, {3, 27, bits(-1.0f)});
    op(w, 43, {3, 28, bits(0.0f)});
    op(w, 43, {3, 29, bits(1.0f)});
    op(w, 54, {1, 4, 0, 2});
    op(w, 248, {30});
    op(w, 61, {15, 31, 11});
    op(w, 196, {15, 32, 31, 24});  // index << 1
    op(w, 199, {15, 33, 32, 25});  // & 2
    op(w, 199, {15, 34, 31, 25});  // index & 2
    op(w, 111, {3, 35, 33});
    op(w, 111, {3, 36, 34});
    op(w, 133, {3, 37, 35, 26});
    op(w, 133, {3, 38, 36, 26});
    op(w, 129, {3, 39, 37, 27});
    op(w, 129, {3, 40, 38, 27});
    op(w, 80, {6, 41, 39, 40, 28, 29});
    op(w, 62, {10, 41});
    op(w, 253, {});
    op(w, 56, {});
    return w;
}

// The colour (n / 255, 1 - n / 255, 0, 1).
Words fragment_words(unsigned n) {
    Words w = {0x07230203u, 0x00010500u, 0u, 40u, 0u};
    op(w, 17, {1});
    op(w, 14, {0, 1});
    op(w, 15, {4, 4, 0x6e69616du, 0u, 9});  // EntryPoint Fragment "main"
    op(w, 16, {4, 7});                      // OriginUpperLeft
    pad(w, 30);
    op(w, 71, {9, 30, 0});  // Location 0
    op(w, 19, {1});
    op(w, 33, {2, 1});
    op(w, 22, {3, 32});
    op(w, 23, {6, 3, 4});
    op(w, 32, {10, 3, 6});
    op(w, 59, {10, 9, 3});
    op(w, 43, {3, 20, bits(float(n) / 255.0f)});
    op(w, 43, {3, 21, bits(float(255 - n) / 255.0f)});
    op(w, 43, {3, 22, bits(0.0f)});
    op(w, 43, {3, 23, bits(1.0f)});
    op(w, 44, {6, 24, 20, 21, 22, 23});
    op(w, 54, {1, 4, 0, 2});
    op(w, 248, {11});
    op(w, 62, {9, 24});
    op(w, 253, {});
    op(w, 56, {});
    return w;
}

std::vector<std::uint8_t> program(unsigned stage, unsigned n) {
    return d3d12test::fssm_program({0x42434f00u + stage, n, 0xbf810000u});
}

std::vector<bigmodule::Program> programs(kCompute);  // what each compute module runs
std::vector<std::size_t> module_bytes(kCompute);

bool write_library(const std::filesystem::path& dir) {
    const auto write = [&](const std::string& name, const Words& words) {
        std::ofstream f(dir / (name + ".spv"), std::ios::binary | std::ios::trunc);
        f.write(reinterpret_cast<const char*>(words.data()), std::streamsize(words.size() * 4));
        return bool(f);
    };
    CHECK(write("vs", vertex_words()));
    std::ofstream index(dir / "index.json", std::ios::binary | std::ios::trunc);
    index << "{\"vs\": {\"stage\": \"vs\", \"fssm_sha256\": \"" << leonida::sha256_hex(program(0, 0))
          << "\", \"bindings\": [], \"push_constant_bytes\": 144, \"input_signature\": []}";
    for (unsigned n = 0; n < kGraphics; ++n) {
        const std::string name = "ps" + std::to_string(n);
        CHECK(write(name, fragment_words(n)));
        index << ",\n\"" << name << "\": {\"stage\": \"ps\", \"fssm_sha256\": \"" << leonida::sha256_hex(program(1, n))
              << "\", \"bindings\": [], \"push_constant_bytes\": 144, \"inputs\": [], \"outputs\": [0]}";
    }
    for (unsigned n = 0; n < kCompute; ++n) {
        const std::string name = "cs" + std::to_string(n);
        // Every large one a little larger than the one before, so that the order by size is known.
        const Words words = bigmodule::make(n < kBig ? kBigBytes + n * 1024 : kSmallBytes, 20251125 + n, &programs[n]);
        module_bytes[n] = words.size() * 4;
        CHECK((module_bytes[n] > 128 * 1024) == (n < kBig));
        hooks.programs[hash(words.data(), words.size())] = int(n);
        CHECK(write(name, words));
        index << ",\n\"" << name << "\": {\"stage\": \"cs\", \"fssm_sha256\": \"" << leonida::sha256_hex(program(2, n))
              << "\", \"bindings\": [], \"push_constant_bytes\": 144, \"local_size\": [1, 1, 1], \"wave64\": false}";
    }
    index << "}";
    return bool(index);
}

// ---- The device and what it runs. ----

struct Setup {
    vktest::Device* vk;
    std::string library;

    bool device(const char* threads, Ref<ID3D12Device>& out) const {
        _putenv_s("LEONIDA_PIPELINE_THREADS", threads);
        rt::d3d12::VulkanDeviceInfo info{};
        info.get_instance_proc = instance_proc;
        info.instance = vk->instance;
        info.physical = vk->physical;
        info.device = vk->device;
        info.queue = vk->queue;
        info.queue_family = vk->family;
        info.buffer_device_address = true;
        info.subgroup = vk->subgroup;
        info.features = vk->enabled;
        info.shader_dir = library;
        CHECK(SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&out.p))));
        return true;
    }
};

struct Pipelines {
    Ref<ID3D12RootSignature> root;
    std::vector<ID3D12PipelineState*> compute, graphics;
    ~Pipelines() { release(); }
    void release() {
        for (auto* s : compute)
            if (s) s->Release();
        for (auto* s : graphics) s->Release();
        compute.clear();
        graphics.clear();
    }

    static ID3D12PipelineState* compute_state(ID3D12Device* dev, ID3D12RootSignature* root, unsigned n) {
        const auto cs = d3d12test::shader_container(program(2, n));
        D3D12_COMPUTE_PIPELINE_STATE_DESC pd{};
        pd.pRootSignature = root;
        pd.CS = {cs.data(), cs.size()};
        ID3D12PipelineState* state = nullptr;
        return SUCCEEDED(dev->CreateComputePipelineState(&pd, IID_PPV_ARGS(&state))) ? state : nullptr;
    }

    // The compute states on `threads` threads.
    bool create(ID3D12Device* dev, unsigned threads = 1) {
        D3D12_ROOT_PARAMETER params[2]{};
        params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
        params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        params[1].Constants.Num32BitValues = 1;
        D3D12_VERSIONED_ROOT_SIGNATURE_DESC desc{};
        desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_0;
        desc.Desc_1_0 = {2, params, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
        const auto blob = d3d12test::serialize_root_signature(desc);
        CHECK(SUCCEEDED(dev->CreateRootSignature(0, blob.data(), blob.size(), IID_PPV_ARGS(&root.p))));
        compute.assign(kCompute, nullptr);
        std::vector<std::thread> creating;
        for (unsigned t = 1; t < threads; ++t)
            creating.emplace_back([&, t] {
                for (unsigned n = t; n < kCompute; n += threads) compute[n] = compute_state(dev, root.p, n);
            });
        for (unsigned n = 0; n < kCompute; n += threads) compute[n] = compute_state(dev, root.p, n);
        for (auto& t : creating) t.join();
        CHECK(std::find(compute.begin(), compute.end(), nullptr) == compute.end());
        const auto vs = d3d12test::shader_container(program(0, 0));
        for (unsigned n = 0; n < kGraphics; ++n) {
            const auto ps = d3d12test::shader_container(program(1, n));
            D3D12_GRAPHICS_PIPELINE_STATE_DESC pd{};
            pd.pRootSignature = root.p;
            pd.VS = {vs.data(), vs.size()};
            pd.PS = {ps.data(), ps.size()};
            for (auto& rt : pd.BlendState.RenderTarget) {
                rt.SrcBlend = rt.SrcBlendAlpha = D3D12_BLEND_ONE;
                rt.DestBlend = rt.DestBlendAlpha = D3D12_BLEND_ZERO;
                rt.BlendOp = rt.BlendOpAlpha = D3D12_BLEND_OP_ADD;
                rt.LogicOp = D3D12_LOGIC_OP_NOOP;
                rt.RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
            }
            pd.SampleMask = UINT_MAX;
            pd.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
            pd.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
            pd.RasterizerState.DepthClipEnable = TRUE;
            pd.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_ALWAYS;
            pd.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
            pd.NumRenderTargets = 1;
            pd.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
            pd.SampleDesc = {1, 0};
            ID3D12PipelineState* state = nullptr;
            CHECK(SUCCEEDED(dev->CreateGraphicsPipelineState(&pd, IID_PPV_ARGS(&state))));
            graphics.push_back(state);
        }
        return true;
    }
};

// A queue and a list that dispatches `compute` (and draws the graphics pipelines), then copies what they wrote back.
struct Frame {
    std::vector<unsigned> compute;
    bool graphics = false;
    Ref<ID3D12Resource> target, buffer, readback;
    Ref<ID3D12DescriptorHeap> rtvs;
    Ref<ID3D12CommandQueue> queue;
    Ref<ID3D12CommandAllocator> allocator;
    Ref<ID3D12GraphicsCommandList> list;
    Ref<ID3D12Fence> fence;
    UINT64 fence_value = 0;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT foot{};

    bool create(ID3D12Device* dev, const Pipelines& p, std::vector<unsigned> dispatched, bool draws) {
        compute = std::move(dispatched);
        graphics = draws;
        const D3D12_HEAP_PROPERTIES def{D3D12_HEAP_TYPE_DEFAULT}, back{D3D12_HEAP_TYPE_READBACK};
        D3D12_RESOURCE_DESC td{};
        td.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        td.Width = td.Height = kExtent;
        td.DepthOrArraySize = td.MipLevels = 1;
        td.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        td.SampleDesc = {1, 0};
        td.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
        CHECK(SUCCEEDED(dev->CreateCommittedResource(&def, D3D12_HEAP_FLAG_NONE, &td, D3D12_RESOURCE_STATE_RENDER_TARGET,
                                                     nullptr, IID_PPV_ARGS(&target.p))));
        const auto bd = d3d12test::buffer_desc(4 * kWords, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        CHECK(SUCCEEDED(dev->CreateCommittedResource(&def, D3D12_HEAP_FLAG_NONE, &bd,
                                                     D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
                                                     IID_PPV_ARGS(&buffer.p))));
        UINT64 image_bytes = 0;
        dev->GetCopyableFootprints(&td, 0, 1, 0, &foot, nullptr, nullptr, &image_bytes);
        CHECK(image_bytes <= 4096);
        const auto rd = d3d12test::buffer_desc(4096 + 4 * kWords);
        CHECK(SUCCEEDED(dev->CreateCommittedResource(&back, D3D12_HEAP_FLAG_NONE, &rd, D3D12_RESOURCE_STATE_COPY_DEST,
                                                     nullptr, IID_PPV_ARGS(&readback.p))));
        const D3D12_DESCRIPTOR_HEAP_DESC hd{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1};
        CHECK(SUCCEEDED(dev->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&rtvs.p))));
        const auto rtv = rtvs->GetCPUDescriptorHandleForHeapStart();
        dev->CreateRenderTargetView(target.p, nullptr, rtv);
        const D3D12_COMMAND_QUEUE_DESC qd{D3D12_COMMAND_LIST_TYPE_DIRECT};
        CHECK(SUCCEEDED(dev->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue.p))));
        CHECK(SUCCEEDED(dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))));
        CHECK(SUCCEEDED(dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p))));
        CHECK(SUCCEEDED(dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, nullptr,
                                               IID_PPV_ARGS(&list.p))));
        const FLOAT clear[4] = {0, 0, 0, 0};
        list->ClearRenderTargetView(rtv, clear, 0, nullptr);
        list->SetComputeRootSignature(p.root.p);
        list->SetComputeRootUnorderedAccessView(0, buffer->GetGPUVirtualAddress());
        for (const unsigned n : compute) {
            list->SetPipelineState(p.compute[n]);
            list->SetComputeRoot32BitConstant(1, n, 0);
            list->Dispatch(1, 1, 1);
        }
        if (graphics) {
            list->SetGraphicsRootSignature(p.root.p);
            list->OMSetRenderTargets(1, &rtv, FALSE, nullptr);
            list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            const D3D12_VIEWPORT vp{0, 0, float(kExtent), float(kExtent), 0, 1};
            list->RSSetViewports(1, &vp);
            for (unsigned n = 0; n < kGraphics; ++n) {
                const D3D12_RECT cell{LONG(2 * n), 0, LONG(2 * n + 2), 2};
                list->RSSetScissorRects(1, &cell);
                list->SetPipelineState(p.graphics[n]);
                list->DrawInstanced(3, 1, 0, 0);
            }
        }
        const D3D12_RESOURCE_BARRIER to_source[2] = {
            d3d12test::transition(target.p, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE),
            d3d12test::transition(buffer.p, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE)};
        list->ResourceBarrier(2, to_source);
        D3D12_TEXTURE_COPY_LOCATION src{target.p, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
        src.SubresourceIndex = 0;
        D3D12_TEXTURE_COPY_LOCATION dst{readback.p, D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
        dst.PlacedFootprint = foot;
        list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
        list->CopyBufferRegion(readback.p, 4096, buffer.p, 0, 4 * kWords);
        const D3D12_RESOURCE_BARRIER back_again[2] = {
            d3d12test::transition(target.p, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_RENDER_TARGET),
            d3d12test::transition(buffer.p, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)};
        list->ResourceBarrier(2, back_again);
        CHECK(SUCCEEDED(list->Close()));
        return true;
    }

    ReplayResult submit() {
        ID3D12CommandList* lists[] = {list.p};
        queue->ExecuteCommandLists(1, lists);
        const ReplayResult result = rt::d3d12::last_execute_result(queue.p);
        queue->Signal(fence.p, ++fence_value);
        fence->SetEventOnCompletion(fence_value, nullptr);
        return result;
    }

    // What the shaders stored and drew; `dropped` is the compute program that was not built, whose word stays 0. The
    // readback buffer is cleared, so stale bytes cannot pass.
    bool right(int dropped = -1) {
        void* mapped = nullptr;
        CHECK(SUCCEEDED(readback->Map(0, nullptr, &mapped)) && mapped);
        auto* bytes = static_cast<std::uint8_t*>(mapped);
        bool ok = true;
        std::uint32_t words[kWords];
        std::memcpy(words, bytes + 4096, sizeof words);
        for (const unsigned n : compute) {
            const std::uint32_t want = int(n) == dropped ? 0 : programs[n].run(n);
            if (words[n] != want) {
                std::fprintf(stderr, "compute program %u stored 0x%08x, not 0x%08x\n", n, words[n], want);
                ok = false;
            }
        }
        for (unsigned n = 0; graphics && n < kGraphics; ++n)
            for (unsigned y = 0; y < 2; ++y)
                for (unsigned x = 2 * n; x < 2 * n + 2; ++x) {
                    const auto* pixel = bytes + foot.Offset + y * foot.Footprint.RowPitch + 4 * x;
                    if (pixel[0] != n || pixel[1] != 255 - n || pixel[2] != 0 || pixel[3] != 255) {
                        std::fprintf(stderr, "graphics pipeline %u drew %u %u %u %u\n", n, pixel[0], pixel[1], pixel[2],
                                     pixel[3]);
                        ok = false;
                    }
                }
        std::memset(bytes, 0xee, 4096 + 4 * kWords);
        readback->Unmap(0, nullptr);
        return ok;
    }
};

std::vector<unsigned> all_compute() {
    std::vector<unsigned> out;
    for (unsigned n = 0; n < kCompute; ++n) out.push_back(n);
    return out;
}

std::uint64_t estimate(unsigned n) { return leonida::large_module_estimate(module_bytes[n]); }

// How many of the large compute modules are in the driver at once, with five threads to compile them; 0: not known (the
// budget is read from the machine).
int expected_at_once() {
    if (serial) return 1;
    if (unlimited) return 5;
    if (!fixed_budget) return 0;
    // The first wave is the largest, in order, while they fit.
    std::uint64_t sum = 0;
    int count = 0;
    for (unsigned n = kBig; n-- > 0 && sum + estimate(n) <= *fixed_budget; ++count) sum += estimate(n);
    return std::clamp(count, 1, 5);
}

// The large compute modules are held back in the driver until as many as may be are in, and no more come.
bool side_by_side(const Setup& setup) {
    Ref<ID3D12Device> dev;
    CHECK(setup.device("4", dev));
    Pipelines pipelines;
    Frame frame;
    hooks.reset();
    CHECK(pipelines.create(dev.p) && frame.create(dev.p, pipelines, all_compute(), false));
    CHECK(hooks.creates() == 0);  // nothing is compiled at creation
    const int at_once = expected_at_once();
    hooks.closed = true;
    ReplayResult result = ReplayResult::unsupported;
    std::thread submitting([&] { result = frame.submit(); });
    const bool filled = hooks.wait_large(std::max(at_once, 1));
    // With a thread to spare, the small ones are compiled while the large ones wait (unlimited: every thread has a large one).
    const bool small_ones = unlimited || !at_once || hooks.wait([] { return hooks.small_done == int(kSmall); });
    std::this_thread::sleep_for(300ms);
    int held = 0;
    {
        std::lock_guard lock(hooks.mutex);
        held = hooks.large_in;
    }
    hooks.open();
    submitting.join();
    CHECK(filled);
    CHECK(!at_once || held == at_once);
    CHECK(small_ones);
    CHECK(result == ReplayResult::success && frame.right());
    CHECK(hooks.over == 0);
    CHECK(hooks.creates() == kCompute);  // each once
    CHECK(!at_once || hooks.most_large <= at_once || unlimited);
    CHECK(frame.submit() == ReplayResult::success && frame.right() && hooks.creates() == kCompute);
    return true;
}

// The same values whatever compiles them, without workers (each where prepare() meets it) and with them.
bool same_values(const Setup& setup) {
    for (const char* threads : {"0", "16"}) {
        Ref<ID3D12Device> dev;
        CHECK(setup.device(threads, dev));
        Pipelines pipelines;
        Frame frame;
        hooks.reset();
        CHECK(pipelines.create(dev.p) && frame.create(dev.p, pipelines, all_compute(), true));
        CHECK(frame.submit() == ReplayResult::success && frame.right());
        CHECK(hooks.over == 0);
        CHECK(hooks.creates() == kCompute + kGraphics);
        CHECK(*threads != '0' || hooks.most_large == 1);
    }
    return true;
}

// A graphics pipeline whose estimate is past the budget (or the one at a time of serial) is in alone among the large ones,
// and the small ones are compiled beside it.
bool graphics_among_the_large(const Setup& setup) {
    Ref<ID3D12Device> dev;
    CHECK(setup.device("4", dev));
    Pipelines pipelines;
    Frame frame;
    hooks.reset();
    CHECK(pipelines.create(dev.p) && frame.create(dev.p, pipelines, all_compute(), true));
    hooks.closed = true;
    ReplayResult result = ReplayResult::unsupported;
    std::thread submitting([&] { result = frame.submit(); });
    const bool one = hooks.wait_large(1);
    const bool small_ones = unlimited || !expected_at_once() ||
                            hooks.wait([] { return hooks.small_done == int(kSmall); });
    std::this_thread::sleep_for(300ms);
    int held = 0;
    bool first_is_graphics = false;
    {
        std::lock_guard lock(hooks.mutex);
        held = hooks.large_in;
        for (const Start& s : hooks.started)
            if (s.large) {
                first_is_graphics = s.program < 0;
                break;
            }
    }
    hooks.open();
    submitting.join();
    CHECK(one && small_ones);
    if (budgeted && fixed_budget) CHECK(held == 1 && first_is_graphics);  // the largest first
    if (serial) CHECK(held == 1);
    CHECK(result == ReplayResult::success && frame.right());
    CHECK(hooks.over == 0);
    CHECK(hooks.creates() == kCompute + kGraphics);
    return true;
}

// Two large graphics pipelines and nothing else: one after the other, or both at once without a limit.
bool graphics_only(const Setup& setup) {
    Ref<ID3D12Device> dev;
    CHECK(setup.device("4", dev));
    Pipelines pipelines;
    Frame frame;
    hooks.reset();
    CHECK(pipelines.create(dev.p) && frame.create(dev.p, pipelines, {}, true));
    hooks.closed = true;
    ReplayResult result = ReplayResult::unsupported;
    std::thread submitting([&] { result = frame.submit(); });
    const int at_once = unlimited ? 2 : 1;
    const bool filled = hooks.wait_large(at_once);
    std::this_thread::sleep_for(300ms);
    int held = 0;
    {
        std::lock_guard lock(hooks.mutex);
        held = hooks.large_in;
    }
    hooks.open();
    submitting.join();
    CHECK(filled);
    if (!budgeted || fixed_budget) CHECK(held == at_once);
    CHECK(result == ReplayResult::success && frame.right());
    CHECK(hooks.over == 0 && hooks.creates() == kGraphics);
    return true;
}

// The driver refuses one large compute module: its dispatch is dropped, the rest runs, and it is not tried again. Its place
// goes to the next.
bool refused(const Setup& setup) {
    for (const char* threads : {"0", "4"}) {
        Ref<ID3D12Device> dev;
        CHECK(setup.device(threads, dev));
        Pipelines pipelines;
        Frame frame;
        hooks.reset();
        CHECK(pipelines.create(dev.p) && frame.create(dev.p, pipelines, all_compute(), false));
        hooks.fail_large = 2;
        CHECK(frame.submit() == ReplayResult::success);
        int dropped = -1;
        {
            std::lock_guard lock(hooks.mutex);
            int seen = 0;
            for (const Start& s : hooks.started)
                if (s.large && s.program >= 0 && seen++ == 2) dropped = s.program;
        }
        CHECK(dropped >= 0);
        CHECK(frame.right(dropped));
        CHECK(hooks.creates() == kCompute);
        CHECK(frame.submit() == ReplayResult::success && frame.right(dropped));
        CHECK(hooks.creates() == kCompute);
    }
    return true;
}

// Two queues submit the same pipelines at once: each is compiled once, the budget is kept, and nobody waits for ever.
bool two_queues(const Setup& setup) {
    Ref<ID3D12Device> dev;
    CHECK(setup.device("4", dev));
    Pipelines pipelines;
    Frame frames[2];
    CHECK(pipelines.create(dev.p));
    auto backwards = all_compute();
    std::reverse(backwards.begin(), backwards.end());
    CHECK(frames[0].create(dev.p, pipelines, all_compute(), true));
    CHECK(frames[1].create(dev.p, pipelines, backwards, true));
    hooks.reset();
    ReplayResult results[2]{};
    std::thread threads[2];
    for (int i = 0; i < 2; ++i) threads[i] = std::thread([&, i] { results[i] = frames[i].submit(); });
    for (auto& t : threads) t.join();
    for (int i = 0; i < 2; ++i) CHECK(results[i] == ReplayResult::success && frames[i].right());
    CHECK(hooks.creates() == kCompute + kGraphics);
    CHECK(hooks.over == 0);
    return true;
}

// The device's last references are released by the thread whose submission compiles, with large ones waiting for room.
bool device_destroyed(const Setup& setup) {
    _putenv_s("LEONIDA_PIPELINE_CACHE_BYTES", "1");  // read the cache at teardown, write no file
    Ref<ID3D12Device> dev;
    CHECK(setup.device("4", dev));
    auto pipelines = std::make_unique<Pipelines>();
    auto frame = std::make_unique<Frame>();
    hooks.reset();
    CHECK(pipelines->create(dev.p) && frame->create(dev.p, *pipelines, all_compute(), true));
    hooks.track = true;
    hooks.closed = true;
    ReplayResult result = ReplayResult::unsupported;
    bool ok = false;
    std::thread submitting([&] {
        result = frame->submit();
        ok = frame->right();
        frame.reset();
        pipelines.reset();
    });
    const bool compiling = hooks.wait_large(1);
    std::this_thread::sleep_for(100ms);
    dev.reset();
    hooks.open();
    submitting.join();
    _putenv_s("LEONIDA_PIPELINE_CACHE_BYTES", "0");
    CHECK(compiling);
    CHECK(result == ReplayResult::success && ok);
    CHECK(hooks.cache_reads > 0);
    CHECK(!hooks.cache_read_during_create);
    CHECK(!hooks.cache_read_with_workers);
    CHECK(hooks.live == 0);
    return true;
}

// One place for large compiles. The warm-up thread has it, and wants the next one after it; a submission that needs four
// others, one of which the warm-up thread waits to build, gets them all first, and builds that one itself.
bool warmup(const Setup& setup) {
    Ref<ID3D12Device> dev;
    CHECK(setup.device("4", dev));
    Pipelines pipelines;
    Frame frame;
    hooks.reset();
    hooks.closed = true;
    hooks.linger = 100ms;
    CHECK(pipelines.create(dev.p));
    if (!hooks.wait_large(1)) {
        hooks.open();
        CHECK(!"the warm-up thread compiles a large one");
    }
    std::thread::id warm;
    int first = -1;
    {
        std::lock_guard lock(hooks.mutex);
        for (const Start& s : hooks.started)
            if (s.large) warm = s.thread, first = s.program;
    }
    CHECK(first == 0);  // their sizes rise with their numbers
    const std::vector<unsigned> wanted = {4, 5, 3, 1};  // 1 is what the warm-up thread builds after 0
    CHECK(frame.create(dev.p, pipelines, wanted, false));
    const std::size_t before = hooks.creates();
    ReplayResult result = ReplayResult::unsupported;
    std::thread submitting([&] { result = frame.submit(); });
    std::this_thread::sleep_for(500ms);
    const bool waits = hooks.creates() == before;  // the place is taken
    hooks.open();
    submitting.join();
    const bool done = result == ReplayResult::success && frame.right();
    rt::d3d12::PipelineCompiler::instance().wait_idle();  // the warm-up thread builds the rest
    std::vector<Start> started;
    {
        std::lock_guard lock(hooks.mutex);
        started = hooks.started;
    }
    CHECK(waits);
    CHECK(done);
    CHECK(hooks.over == 0 && hooks.most_large == 1);
    std::set<int> once;
    for (const Start& s : started) once.insert(s.program);
    CHECK(started.size() == kCompute && once.size() == kCompute);  // each once
    std::erase_if(started, [](const Start& s) { return !s.large; });
    std::set<int> built;
    for (std::size_t i = 1; i <= wanted.size(); ++i) {
        CHECK(started[i].thread != warm);  // none of the warm-up's between two of the submission's
        built.insert(started[i].program);
    }
    CHECK(built == std::set<int>(wanted.begin(), wanted.end()));
    for (std::size_t i = wanted.size() + 1; i < started.size(); ++i) CHECK(started[i].thread == warm);
    return true;
}

// The large ones that four threads create at once are held back in the driver until as many as may be are in.
bool eager(const Setup& setup) {
    Ref<ID3D12Device> dev;
    CHECK(setup.device("4", dev));
    Pipelines pipelines;
    Frame frame;
    hooks.reset();
    hooks.closed = true;
    bool created = false;
    std::thread creating([&] { created = pipelines.create(dev.p, 4); });
    const bool one = hooks.wait_large(1);
    std::this_thread::sleep_for(300ms);
    int held = 0;
    {
        std::lock_guard lock(hooks.mutex);
        held = hooks.large_in;
    }
    hooks.open();
    creating.join();
    CHECK(one && created);
    CHECK(held == expected_at_once());
    CHECK(hooks.over == 0 && hooks.creates() == kCompute);
    CHECK(frame.create(dev.p, pipelines, all_compute(), false));
    CHECK(frame.submit() == ReplayResult::success && frame.right());
    CHECK(hooks.creates() == kCompute);
    return true;
}

bool run(const Setup& setup, const std::string& scenario) {
    if (scenario == "warmup") return warmup(setup);
    if (scenario == "eager") return eager(setup);
    return side_by_side(setup) && same_values(setup) && graphics_among_the_large(setup) && graphics_only(setup) &&
           refused(setup) && two_queues(setup) && device_destroyed(setup);
}

}  // namespace

int main(int argc, char** argv) {
    const std::string scenario = argc > 1 ? argv[1] : "";
    vktest::Device vk;
    vk.request_buffer_device_address = true;
    const auto status = vk.initialize();
    if (status == vktest::DeviceStatus::unavailable) return 77;
    if (status != vktest::DeviceStatus::ready) return 1;
    VkPhysicalDeviceProperties properties{};
    vk.instance_fn<PFN_vkGetPhysicalDeviceProperties>("vkGetPhysicalDeviceProperties")(vk.physical, &properties);
    if (!vk.buffer_device_address || !vk.enabled.shaderInt64) {
        std::printf("SKIP: the GPU lacks bufferDeviceAddress or shaderInt64\n");
        return 77;
    }
    if (properties.limits.maxPushConstantsSize < 144) {
        std::printf("SKIP: maxPushConstantsSize %u, below the 144-byte block\n", properties.limits.maxPushConstantsSize);
        return 77;
    }
    real_instance_proc = vk.instance_proc;
    _putenv_s("LEONIDA_PIPELINE_CACHE_BYTES", "0");  // no session file in or out
    if (scenario != "warmup") _putenv_s("LEONIDA_COMPUTE_WARMUP", "0");  // every compile is a submission's
    const testfiles::ScratchDir library("leonida_big_compute_test");
    if (!write_library(library.path())) return 1;
    if (!run({&vk, library.path().string()}, scenario)) return 1;
    if (vk.validation_errors()) {
        std::fprintf(stderr, "FAIL: %u validation errors\n", vk.validation_errors());
        return 1;
    }
    std::printf("d3d12_big_compute_test: passed on %s\n", properties.deviceName);
    return 0;
}
