// Exercise one production compute PSO in each invocation. The same authored module, root signature and device
// should give the same journal key in separate cold/warm processes. The readback proves the dispatch actually ran;
// a journal hit or write is a separate observation from the production runtime, not inferred from these words.
#include "vk_test_device.h"
#include "d3d12_test_shaders.h"
#include "shader_fixture_path.h"
#include "test_temp_dir.h"
#include "../shims/d3d12_com.h"
#include "../src/shader_library.h"

#include <array>
#include <cstdio>
#include <cstring>
#include <fstream>

using namespace d3d12test;

#define CHECK(expr) do { if (!(expr)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #expr); return 1; } } while (0)

namespace {
int dispatch_once(ID3D12Device* dev, const std::vector<std::uint8_t>& container, UINT expected_sum) {
    D3D12_ROOT_PARAMETER param{};
    param.ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
    D3D12_VERSIONED_ROOT_SIGNATURE_DESC root_desc{};
    root_desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_0;
    root_desc.Desc_1_0 = {1, &param, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
    const auto bytes = serialize_root_signature(root_desc);
    CHECK(!bytes.empty());
    Ref<ID3D12RootSignature> root;
    CHECK(SUCCEEDED(dev->CreateRootSignature(0, bytes.data(), bytes.size(), IID_PPV_ARGS(&root.p))));
    D3D12_COMPUTE_PIPELINE_STATE_DESC desc{};
    desc.pRootSignature = root.p;
    desc.CS = {container.data(), container.size()};
    Ref<ID3D12PipelineState> pso;
    CHECK(SUCCEEDED(dev->CreateComputePipelineState(&desc, IID_PPV_ARGS(&pso.p))));

    using Counters = std::array<UINT, 2>;
    const auto buffer = [&](Ref<ID3D12Resource>& dst, D3D12_HEAP_TYPE type, D3D12_RESOURCE_STATES state,
                            D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE) {
        const D3D12_HEAP_PROPERTIES heap{type};
        const auto resource = buffer_desc(sizeof(Counters), flags);
        return SUCCEEDED(dev->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &resource, state, nullptr,
                                                     IID_PPV_ARGS(&dst.p)));
    };
    Ref<ID3D12Resource> upload, result, readback;
    CHECK(buffer(upload, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ));
    CHECK(buffer(result, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COPY_DEST,
                 D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS));
    CHECK(buffer(readback, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST));
    const D3D12_RANGE none{0, 0}, both{0, sizeof(Counters)};
    void* mapped = nullptr;
    CHECK(SUCCEEDED(upload->Map(0, &none, &mapped)) && mapped);
    std::memset(mapped, 0, sizeof(Counters));
    upload->Unmap(0, &both);

    constexpr auto type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
    const D3D12_COMMAND_QUEUE_DESC queue_desc{type};
    Ref<ID3D12CommandQueue> queue;
    Ref<ID3D12CommandAllocator> allocator;
    Ref<ID3D12GraphicsCommandList> list;
    Ref<ID3D12Fence> fence;
    CHECK(SUCCEEDED(dev->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue.p))));
    CHECK(SUCCEEDED(dev->CreateCommandAllocator(type, IID_PPV_ARGS(&allocator.p))));
    CHECK(SUCCEEDED(dev->CreateCommandList(0, type, allocator.p, nullptr, IID_PPV_ARGS(&list.p))));
    CHECK(SUCCEEDED(dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))));

    list->CopyBufferRegion(result.p, 0, upload.p, 0, sizeof(Counters));
    const auto ready = transition(result.p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    list->ResourceBarrier(1, &ready);
    list->SetComputeRootSignature(root.p);
    list->SetPipelineState(pso.p);
    const auto va = result->GetGPUVirtualAddress();
    CHECK(va != 0);
    list->SetComputeRootUnorderedAccessView(0, va);
    list->Dispatch(4, 1, 1);  // each group adds 1 to counter 0 and groupID.x + the authored term to counter 1
    const auto copy = transition(result.p, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
    list->ResourceBarrier(1, &copy);
    list->CopyBufferRegion(readback.p, 0, result.p, 0, sizeof(Counters));
    CHECK(SUCCEEDED(list->Close()));
    ID3D12CommandList* lists[] = {list.p};
    queue->ExecuteCommandLists(1, lists);
    CHECK(rt::d3d12::last_execute_result(queue.p) == leonida::ReplayResult::success);
    CHECK(SUCCEEDED(queue->Signal(fence.p, 1)));
    CHECK(SUCCEEDED(fence->SetEventOnCompletion(1, nullptr)));
    mapped = nullptr;
    CHECK(SUCCEEDED(readback->Map(0, &both, &mapped)) && mapped);
    Counters got{};
    std::memcpy(got.data(), mapped, sizeof(got));
    readback->Unmap(0, &none);
    if (got[0] != 4 || got[1] != expected_sum) {
        std::fprintf(stderr, "compute journal: expected (4,%u), got (%u,%u)\n", expected_sum, got[0], got[1]);
        return 1;
    }
    return 0;
}
}  // namespace

int main(int argc, char** argv) {
    const bool second = argc == 2 && std::strcmp(argv[1], "--second-shader") == 0;
    if (argc != 1 && !second) {
        std::fputs("usage: d3d12_compute_journal_test [--second-shader]\n", stderr);
        return 2;
    }
    // One additional constant per group in the second module: (1+2+3+4) versus (2+3+4+5).
    const UINT expected_sum = second ? 14u : 10u;
    const testfiles::ScratchDir scratch("leonida_compute_journal_test");
    // Default CTest runs use disposable caches. An explicit cache directory lets the host preserve the exact
    // same state for the second process; never infer a hit from runtime duration.
    if (!GetEnvironmentVariableW(L"LEONIDA_PIPELINE_CACHE_DIR", nullptr, 0)) {
        CHECK(SetEnvironmentVariableW(L"LEONIDA_PIPELINE_CACHE_DIR", (scratch.path() / "cache").c_str()));
    }
    {
        vktest::Device vk;
        vk.request_buffer_device_address = true;
        const auto status = vk.initialize();
        if (status == vktest::DeviceStatus::unavailable) return 77;
        CHECK(status == vktest::DeviceStatus::ready);
        if (!vk.buffer_device_address || !vk.pipeline_cache_control_enabled()) {
            std::puts("SKIP: requires bufferDeviceAddress and enabled Vulkan 1.3 pipelineCreationCacheControl");
            return 77;
        }
        const auto program = fssm_program({second ? 0x4a435332u : 0x4a435331u, 0x10203040u});
        const auto container = shader_container(program);
        const auto shader = vktest::shader_fixture_path(second ? LEONIDA_TEST_COUNTER_V2_SPV : LEONIDA_TEST_COUNTER_SPV);
        CHECK(testfiles::copy_file(shader, scratch.path() / "counter.spv"));
        std::ofstream index(scratch.path() / "index.json", std::ios::binary);
        index << "{\"counter\":{\"stage\":\"cs\",\"fssm_sha256\":\"" << leonida::sha256_hex(program)
              << "\",\"bindings\":[],\"push_constant_bytes\":144,\"local_size\":[1,1,1],"
                 "\"uses_code_base\":false,\"wave64\":false}}";
        index.close();
        CHECK(index);
        rt::d3d12::VulkanDeviceInfo info{};
        info.get_instance_proc = vk.instance_proc;
        info.instance = vk.instance;
        info.physical = vk.physical;
        info.device = vk.device;
        info.queue = vk.queue;
        info.queue_family = vk.family;
        info.buffer_device_address = vk.buffer_device_address;
        info.pipeline_cache_control = vk.pipeline_cache_control_enabled();
        info.shader_dir = scratch.path().string();
        Ref<ID3D12Device> dev;
        CHECK(SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&dev.p))));
        if (rt::d3d12::device_max_push_constants_size(dev.p) < 144) {
            std::puts("SKIP: authored shader requires 144 push-constant bytes");
            return 77;
        }
        CHECK(dispatch_once(dev.p, container, expected_sum) == 0);
    }
    CHECK(vktest::validation_error_count() == 0);  // includes device teardown
    std::printf("compute journal: direct Dispatch(4,1,1) copied both expected words (4,%u)\n", expected_sum);
    return 0;
}
