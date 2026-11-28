// Pipeline layouts shared by the PSOs of one device (DeviceState::pipeline_layout, RT-SHARED-PIPELINE-LAYOUTS). A
// session makes ~50k PSOs from a handful of layout shapes; one layout each cost ~2.3 KB of driver memory with the
// bindless set 0. Through the D3D12 layer on a GPU, with vkCreatePipelineLayout and vkDestroyPipelineLayout watched
// through the device's vkGetDeviceProcAddr:
//   - 64 PSOs of each shape (graphics and compute, each with and without set-0 bindings; the set-0 ones only where
//     the device has the bindless set), made on four threads at once: one layout per shape;
//   - the PSOs released with the device alive: no layout destroyed; new PSOs of the same shapes: no layout made;
//   - a dispatch with a shared layout (set 0 bound, constants pushed) run before its PSO and the device go;
//   - the device released: every layout it made destroyed exactly once, and nothing destroyed twice or unknown;
//   - a second device makes its own layouts, and destroys them at its teardown.
// With LEONIDA_LAZY_PIPELINES=0 and LEONIDA_LAZY_COMPUTE=0 (the _eager test) every pipeline is built at creation with
// the shared layout. Self-contained: builds its SPIR-V in memory. Set LEONIDA_VK_VALIDATION=1 to run under
// VK_LAYER_KHRONOS_validation (a layout left at the test's vkDestroyDevice is VUID-vkDestroyDevice-device-05137).
#include "vk_test_device.h"
#include "d3d12_test_shaders.h"
#include "test_temp_dir.h"
#include "../shims/d3d12_com.h"
#include "../src/shader_library.h"

#include <bit>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

namespace {

using d3d12test::Ref;
using Words = std::vector<std::uint32_t>;

constexpr unsigned kPerShape = 64, kThreads = 4;

#define CHECK(expr)                                                                  \
    do {                                                                             \
        if (!(expr)) {                                                               \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #expr);     \
            return false;                                                            \
        }                                                                            \
    } while (0)

// ---- The driver's layout calls, watched. ----

struct Hooks {
    std::mutex mutex;
    unsigned creates = 0, destroys = 0;
    unsigned pipelines = 0;                 // made by vkCreate{Graphics,Compute}Pipelines (all with these layouts)
    unsigned bad_destroys = 0;              // of a handle not made, or made and destroyed already
    std::set<VkPipelineLayout> live;
    std::map<VkPipelineLayout, unsigned> destroyed;  // each handle made while `watch` was set: times destroyed
    bool watch = false;
} hooks;

PFN_vkGetInstanceProcAddr real_instance_proc = nullptr;
PFN_vkGetDeviceProcAddr real_device_proc = nullptr;
PFN_vkCreatePipelineLayout real_create_layout = nullptr;
PFN_vkDestroyPipelineLayout real_destroy_layout = nullptr;
PFN_vkCreateGraphicsPipelines real_create_graphics = nullptr;
PFN_vkCreateComputePipelines real_create_compute = nullptr;

void count_pipelines(VkResult result, std::uint32_t count, const VkPipeline* out) {
    if (result < 0) return;
    std::lock_guard lock(hooks.mutex);
    for (std::uint32_t i = 0; i < count; ++i) hooks.pipelines += out[i] != VK_NULL_HANDLE;
}

VKAPI_ATTR VkResult VKAPI_CALL create_graphics(VkDevice device, VkPipelineCache cache, std::uint32_t count,
                                               const VkGraphicsPipelineCreateInfo* infos,
                                               const VkAllocationCallbacks* allocator, VkPipeline* out) {
    const VkResult result = real_create_graphics(device, cache, count, infos, allocator, out);
    count_pipelines(result, count, out);
    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL create_compute(VkDevice device, VkPipelineCache cache, std::uint32_t count,
                                              const VkComputePipelineCreateInfo* infos,
                                              const VkAllocationCallbacks* allocator, VkPipeline* out) {
    const VkResult result = real_create_compute(device, cache, count, infos, allocator, out);
    count_pipelines(result, count, out);
    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL create_layout(VkDevice device, const VkPipelineLayoutCreateInfo* info,
                                             const VkAllocationCallbacks* allocator, VkPipelineLayout* out) {
    const VkResult result = real_create_layout(device, info, allocator, out);
    if (result == VK_SUCCESS) {
        std::lock_guard lock(hooks.mutex);
        ++hooks.creates;
        hooks.live.insert(*out);
        if (hooks.watch) hooks.destroyed[*out] = 0;
    }
    return result;
}

VKAPI_ATTR void VKAPI_CALL destroy_layout(VkDevice device, VkPipelineLayout layout,
                                          const VkAllocationCallbacks* allocator) {
    if (layout) {
        std::lock_guard lock(hooks.mutex);
        ++hooks.destroys;
        if (!hooks.live.erase(layout)) ++hooks.bad_destroys;
        if (const auto it = hooks.destroyed.find(layout); it != hooks.destroyed.end()) ++it->second;
    }
    real_destroy_layout(device, layout, allocator);
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL device_proc(VkDevice device, const char* name) {
    const PFN_vkVoidFunction real = real_device_proc(device, name);
    if (!real) return nullptr;
    if (!std::strcmp(name, "vkCreatePipelineLayout")) {
        real_create_layout = reinterpret_cast<PFN_vkCreatePipelineLayout>(real);
        return reinterpret_cast<PFN_vkVoidFunction>(create_layout);
    }
    if (!std::strcmp(name, "vkDestroyPipelineLayout")) {
        real_destroy_layout = reinterpret_cast<PFN_vkDestroyPipelineLayout>(real);
        return reinterpret_cast<PFN_vkVoidFunction>(destroy_layout);
    }
    if (!std::strcmp(name, "vkCreateGraphicsPipelines")) {
        real_create_graphics = reinterpret_cast<PFN_vkCreateGraphicsPipelines>(real);
        return reinterpret_cast<PFN_vkVoidFunction>(create_graphics);
    }
    if (!std::strcmp(name, "vkCreateComputePipelines")) {
        real_create_compute = reinterpret_cast<PFN_vkCreateComputePipelines>(real);
        return reinterpret_cast<PFN_vkVoidFunction>(create_compute);
    }
    return real;
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL instance_proc(VkInstance instance, const char* name) {
    if (!std::strcmp(name, "vkGetDeviceProcAddr")) {
        real_device_proc = reinterpret_cast<PFN_vkGetDeviceProcAddr>(real_instance_proc(instance, name));
        return reinterpret_cast<PFN_vkVoidFunction>(device_proc);
    }
    return real_instance_proc(instance, name);
}

// ---- The shaders: the smallest modules of each stage. ----

void op(Words& out, std::uint16_t code, std::initializer_list<std::uint32_t> operands) {
    out.push_back((std::uint32_t(operands.size() + 1) << 16) | code);
    out.insert(out.end(), operands.begin(), operands.end());
}

// Position = (0, 0, 0, 1) (a vertex stage, or with `fragment` the colour at location 0).
Words constant_words(bool fragment) {
    Words w = {0x07230203u, 0x00010500u, 0u, 32u, 0u};
    op(w, 17, {1});                                              // OpCapability Shader
    op(w, 14, {0, 1});                                           // OpMemoryModel Logical GLSL450
    op(w, 15, {fragment ? 4u : 0u, 4, 0x6e69616du, 0u, 10});     // OpEntryPoint Fragment/Vertex %4 "main" %10
    if (fragment) op(w, 16, {4, 7});                             // OpExecutionMode OriginUpperLeft
    if (fragment) op(w, 71, {10, 30, 0});                        // OpDecorate %10 Location 0
    else op(w, 71, {10, 11, 0});                                 // OpDecorate %10 BuiltIn Position
    op(w, 19, {1});                                              // %1 = void
    op(w, 33, {2, 1});                                           // %2 = fn void
    op(w, 22, {3, 32});                                          // %3 = float
    op(w, 23, {6, 3, 4});                                        // %6 = vec4
    op(w, 32, {7, 3, 6});                                        // %7 = Output vec4*
    op(w, 59, {7, 10, 3});                                       // %10 = OpVariable Output
    op(w, 43, {3, 20, 0});                                       // 0.0
    op(w, 43, {3, 21, std::bit_cast<std::uint32_t>(1.0f)});      // 1.0
    op(w, 44, {6, 22, 20, 20, 20, 21});                          // (0, 0, 0, 1)
    op(w, 54, {1, 4, 0, 2});                                     // OpFunction
    op(w, 248, {23});                                            // OpLabel
    op(w, 62, {10, 22});                                         // OpStore
    op(w, 253, {});                                              // OpReturn
    op(w, 56, {});                                               // OpFunctionEnd
    return w;
}

Words compute_words() {
    Words w = {0x07230203u, 0x00010500u, 0u, 8u, 0u};
    op(w, 17, {1});
    op(w, 14, {0, 1});
    op(w, 15, {5, 4, 0x6e69616du, 0u});  // OpEntryPoint GLCompute %4 "main"
    op(w, 16, {4, 17, 1, 1, 1});          // OpExecutionMode LocalSize 1 1 1
    op(w, 19, {1});
    op(w, 33, {2, 1});
    op(w, 54, {1, 4, 0, 2});
    op(w, 248, {5});
    op(w, 253, {});
    op(w, 56, {});
    return w;
}

// The shapes: the graphics ones vary in the pixel stage's set-0 bindings, the compute ones in the compute stage's.
enum Shape : unsigned { graphics_plain, graphics_set0, compute_plain, compute_set0, shape_count };
bool set0(unsigned shape) { return shape == graphics_set0 || shape == compute_set0; }
bool compute(unsigned shape) { return shape >= compute_plain; }

// Program n of a shape (n == kPerShape: the vertex stage every graphics PSO shares). Each has its own hash, index
// entry and module, so no two PSOs are the same one.
std::vector<std::uint8_t> program(unsigned shape, unsigned n) {
    return d3d12test::fssm_program({0x534c5000u + shape, n, 0xbf810000u});
}

bool write_library(const std::filesystem::path& dir) {
    const auto write = [&](const std::string& name, const Words& words) {
        std::ofstream f(dir / (name + ".spv"), std::ios::binary | std::ios::trunc);
        f.write(reinterpret_cast<const char*>(words.data()), std::streamsize(words.size() * 4));
        return bool(f);
    };
    CHECK(write("vs", constant_words(false)));
    std::ofstream index(dir / "index.json", std::ios::binary | std::ios::trunc);
    index << "{\"vs\": {\"stage\": \"vs\", \"fssm_sha256\": \"" << leonida::sha256_hex(program(graphics_plain, kPerShape))
          << "\", \"bindings\": [], \"push_constant_bytes\": 144, \"input_signature\": []}";
    for (unsigned shape = 0; shape < shape_count; ++shape) {
        for (unsigned n = 0; n < kPerShape; ++n) {
            const std::string name = "s" + std::to_string(shape) + "_" + std::to_string(n);
            CHECK(write(name, compute(shape) ? compute_words() : constant_words(true)));
            index << ",\n\"" << name << "\": {\"stage\": \"" << (compute(shape) ? "cs" : "ps")
                  << "\", \"fssm_sha256\": \"" << leonida::sha256_hex(program(shape, n)) << "\", \"bindings\": "
                  << (set0(shape) ? "[0]" : "[]") << ", \"push_constant_bytes\": 144, "
                  << (compute(shape) ? "\"local_size\": [1, 1, 1], \"wave64\": false}" : "\"inputs\": [], \"outputs\": [0]}");
        }
    }
    index << "}";
    return bool(index);
}

// ---- The device and its PSOs. ----

struct Setup {
    vktest::Device* vk;
    std::string library;
    unsigned shapes;  // the shapes this device can make: 4 with the bindless set, else the two without set 0

    bool has(unsigned shape) const { return shapes == shape_count || !set0(shape); }

    bool device(Ref<ID3D12Device>& out) const {
        rt::d3d12::VulkanDeviceInfo info{};
        info.get_instance_proc = instance_proc;
        info.instance = vk->instance;
        info.physical = vk->physical;
        info.device = vk->device;
        info.queue = vk->queue;
        info.queue_family = vk->family;
        info.buffer_device_address = true;
        info.bindless = vk->bindless;
        info.bindless_partially_bound = vk->partially_bound;
        info.subgroup = vk->subgroup;
        info.features = vk->enabled;
        info.shader_dir = library;
        CHECK(SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&out.p))));
        return true;
    }
};

ID3D12PipelineState* make_pso(ID3D12Device* dev, ID3D12RootSignature* root, unsigned shape, unsigned n) {
    const auto code = d3d12test::shader_container(program(shape, n));
    ID3D12PipelineState* out = nullptr;
    if (compute(shape)) {
        D3D12_COMPUTE_PIPELINE_STATE_DESC pd{};
        pd.pRootSignature = root;
        pd.CS = {code.data(), code.size()};
        return SUCCEEDED(dev->CreateComputePipelineState(&pd, IID_PPV_ARGS(&out))) ? out : nullptr;
    }
    const auto vs = d3d12test::shader_container(program(graphics_plain, kPerShape));
    D3D12_GRAPHICS_PIPELINE_STATE_DESC pd{};
    pd.pRootSignature = root;
    pd.VS = {vs.data(), vs.size()};
    pd.PS = {code.data(), code.size()};
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
    return SUCCEEDED(dev->CreateGraphicsPipelineState(&pd, IID_PPV_ARGS(&out))) ? out : nullptr;
}

bool make_root(ID3D12Device* dev, Ref<ID3D12RootSignature>& root) {
    D3D12_ROOT_PARAMETER param{};
    param.ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    param.Constants.Num32BitValues = 1;
    D3D12_VERSIONED_ROOT_SIGNATURE_DESC desc{};
    desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_0;
    desc.Desc_1_0 = {1, &param, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
    const auto blob = d3d12test::serialize_root_signature(desc);
    CHECK(!blob.empty());
    CHECK(SUCCEEDED(dev->CreateRootSignature(0, blob.data(), blob.size(), IID_PPV_ARGS(&root.p))));
    return true;
}

void release(std::vector<ID3D12PipelineState*>& psos) {
    for (auto* pso : psos)
        if (pso) pso->Release();
    psos.clear();
}

unsigned creates() {
    std::lock_guard lock(hooks.mutex);
    return hooks.creates;
}

// The layouts made while `watch` was set that were destroyed `times` times; and how many there are.
unsigned watched(unsigned times, unsigned* total = nullptr) {
    std::lock_guard lock(hooks.mutex);
    unsigned n = 0;
    for (const auto& [layout, destroyed] : hooks.destroyed) n += destroyed == times;
    if (total) *total = unsigned(hooks.destroyed.size());
    return n;
}

bool run(const Setup& setup) {
    {
        Ref<ID3D12Device> dev;
        CHECK(setup.device(dev));
        Ref<ID3D12RootSignature> root;
        CHECK(make_root(dev.p, root));
        std::vector<ID3D12PipelineState*> psos(shape_count * kPerShape, nullptr);
        {
            std::lock_guard lock(hooks.mutex);
            hooks.watch = true;
        }
        const unsigned before = creates();
        // Four threads, each making a quarter of every shape, interleaved, as a title's job threads do: the first
        // PSOs of a shape race for its layout.
        std::vector<std::thread> threads;
        for (unsigned t = 0; t < kThreads; ++t)
            threads.emplace_back([&, t] {
                for (unsigned n = t; n < kPerShape; n += kThreads)
                    for (unsigned shape = 0; shape < shape_count; ++shape)
                        if (setup.has(shape)) psos[shape * kPerShape + n] = make_pso(dev.p, root.p, shape, n);
            });
        for (auto& thread : threads) thread.join();
        const unsigned made = creates() - before;
        {
            std::lock_guard lock(hooks.mutex);
            hooks.watch = false;
        }
        unsigned pso_count = 0;
        for (unsigned shape = 0; shape < shape_count; ++shape)
            for (unsigned n = 0; n < kPerShape; ++n)
                if (setup.has(shape)) {
                    CHECK(psos[shape * kPerShape + n]);
                    ++pso_count;
                }
        std::printf("%u PSOs of %u shapes made %u pipeline layout(s)\n", pso_count, setup.shapes, made);
        CHECK(made == setup.shapes);

        // Released with the device alive: their layouts stay, and the same shapes find them again.
        release(psos);
        CHECK(watched(0) == setup.shapes);
        const unsigned again_before = creates();
        std::vector<ID3D12PipelineState*> again;
        for (unsigned shape = 0; shape < shape_count; ++shape)
            if (setup.has(shape)) {
                again.push_back(make_pso(dev.p, root.p, shape, 0));
                CHECK(again.back());
            }
        CHECK(creates() == again_before);
        release(again);

        // A dispatch that binds set 0 and pushes constants with a shared layout, run before its PSO and then the
        // device are released (D3D12 asks the title to wait for the GPU before it releases a PSO it used), so
        // validation sees the shared layout used by a command buffer and destroyed at the device's teardown.
        Ref<ID3D12CommandQueue> queue;
        Ref<ID3D12CommandAllocator> allocator;
        Ref<ID3D12GraphicsCommandList> list;
        Ref<ID3D12Fence> fence;
        {
            Ref<ID3D12PipelineState> pso;
            pso.p = make_pso(dev.p, root.p, setup.has(compute_set0) ? compute_set0 : compute_plain, 0);
            CHECK(pso.p);
            CHECK(creates() == again_before);
            const D3D12_COMMAND_QUEUE_DESC queue_desc{};
            CHECK(SUCCEEDED(dev->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue.p))));
            CHECK(SUCCEEDED(dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p))));
            CHECK(SUCCEEDED(dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, pso.p,
                                                   IID_PPV_ARGS(&list.p))));
            list->SetComputeRootSignature(root.p);
            list->SetComputeRoot32BitConstant(0, 7, 0);
            list->Dispatch(1, 1, 1);
            CHECK(list->Close() == S_OK);
            ID3D12CommandList* lists[] = {list.p};
            queue->ExecuteCommandLists(1, lists);
            CHECK(rt::d3d12::last_execute_result(queue.p) == leonida::ReplayResult::success);
            CHECK(SUCCEEDED(dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))));
            CHECK(SUCCEEDED(queue->Signal(fence.p, 1)));
            CHECK(SUCCEEDED(fence->SetEventOnCompletion(1, nullptr)));
        }
        root.reset();
        CHECK(watched(0) == setup.shapes);
    }  // the device's last reference: its teardown
    {
        unsigned total = 0;
        CHECK(watched(1, &total) == setup.shapes && total == setup.shapes);
        std::lock_guard lock(hooks.mutex);
        std::printf("after the first device: %u layouts made, %u destroyed, %u live, %u bad destroys; %u pipelines "
                    "built\n",
                    hooks.creates, hooks.destroys, unsigned(hooks.live.size()), hooks.bad_destroys, hooks.pipelines);
        CHECK(hooks.live.empty() && hooks.creates == hooks.destroys && !hooks.bad_destroys);
        hooks.destroyed.clear();
    }

    // A second device over the same VkDevice: its own layouts, gone with it.
    {
        Ref<ID3D12Device> dev;
        CHECK(setup.device(dev));
        Ref<ID3D12RootSignature> root;
        CHECK(make_root(dev.p, root));
        {
            std::lock_guard lock(hooks.mutex);
            hooks.watch = true;
        }
        const unsigned before = creates();
        std::vector<ID3D12PipelineState*> psos;
        for (unsigned n = 0; n < 2; ++n)
            for (unsigned shape = 0; shape < shape_count; ++shape)
                if (setup.has(shape)) {
                    psos.push_back(make_pso(dev.p, root.p, shape, n));
                    CHECK(psos.back());
                }
        CHECK(creates() - before == setup.shapes);
        {
            std::lock_guard lock(hooks.mutex);
            hooks.watch = false;
        }
        release(psos);
    }
    unsigned total = 0;
    CHECK(watched(1, &total) == setup.shapes && total == setup.shapes);
    std::lock_guard lock(hooks.mutex);
    std::printf("after the second device: %u layouts made, %u destroyed, %u live, %u bad destroys; %u pipelines "
                "built\n",
                hooks.creates, hooks.destroys, unsigned(hooks.live.size()), hooks.bad_destroys, hooks.pipelines);
    CHECK(hooks.live.empty() && hooks.creates == hooks.destroys && !hooks.bad_destroys);
    return true;
}

}  // namespace

int main() {
    vktest::Device vk;
    vk.request_buffer_device_address = true;
    const auto status = vk.initialize();
    if (status == vktest::DeviceStatus::unavailable) return 77;
    if (status != vktest::DeviceStatus::ready) return 1;
    VkPhysicalDeviceProperties properties{};
    vk.instance_fn<PFN_vkGetPhysicalDeviceProperties>("vkGetPhysicalDeviceProperties")(vk.physical, &properties);
    if (!vk.buffer_device_address) {
        std::printf("SKIP: the GPU lacks bufferDeviceAddress\n");
        return 77;
    }
    if (properties.limits.maxPushConstantsSize < 144) {
        std::printf("SKIP: maxPushConstantsSize %u, below the 144-byte block\n", properties.limits.maxPushConstantsSize);
        return 77;
    }
    if (!vk.bindless) std::printf("the device lacks the bindless set 0: only the shapes without set-0 bindings\n");
    real_instance_proc = vk.instance_proc;
    _putenv_s("LEONIDA_PIPELINE_CACHE_BYTES", "0");  // no session file in or out
    const testfiles::ScratchDir library("leonida_shared_pipeline_layouts_test");
    if (!write_library(library.path())) return 1;
    const Setup setup{&vk, library.path().string(), vk.bindless ? unsigned(shape_count) : 2u};
    if (!run(setup)) return 1;
    if (vk.validation_errors()) {
        std::fprintf(stderr, "FAIL: %u validation errors\n", vk.validation_errors());
        return 1;
    }
    std::printf("d3d12_shared_pipeline_layouts_test: passed on %s\n", properties.deviceName);
    return 0;
}
