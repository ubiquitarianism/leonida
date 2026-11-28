// D3D12 ExecuteIndirect with a GPU count buffer and one DISPATCH argument per record.
// The same synthetic workload runs on native WARP and on the Vulkan layer, with actual GPU readback.
// Under Wine the WARP runs are skipped (host_wine.h) and the layer is checked against the expected result alone.
// Set LEONIDA_VK_VALIDATION=1 to enable Vulkan core and synchronization validation.
#include "vk_test_device.h"
#include "host_wine.h"
#include "shader_fixture_path.h"
#include "d3d12_test_shaders.h"
#include "../shims/d3d12_com.h"
#include "../src/shader_library.h"
#include <dxgi1_4.h>
#include <d3dcompiler.h>
#include <winver.h>
#include <array>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#ifndef LEONIDA_TEST_INDIRECT_COUNT_SPV
#error "LEONIDA_TEST_INDIRECT_COUNT_SPV must name the synthetic GLSL compute module"
#endif

using namespace d3d12test;
namespace {
constexpr UINT kMaximum = 3, kStride = 20, kArgsOffset = 32, kCountOffset = 44;
const char native_shader[] = R"(
RWByteAddressBuffer output : register(u0);
[numthreads(1,1,1)]
void main(uint3 group : SV_GroupID) {
    uint old;
    output.InterlockedAdd(0, 1, old);
    output.InterlockedAdd(4, group.x + 1, old);
}
)";

bool run(ID3D12Device* dev, const std::vector<std::uint8_t>& shader, UINT count, bool gpu_update,
         std::array<UINT, 2>& result, bool shim, bool fixed = false, bool clamp_fixture = false) {
    Ref<ID3D12RootSignature> root;
    D3D12_ROOT_PARAMETER param{};
    param.ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
    param.Descriptor = {0, 0};
    param.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    D3D12_VERSIONED_ROOT_SIGNATURE_DESC root_desc{};
    root_desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_0;
    root_desc.Desc_1_0 = {1, &param, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
    const auto serialized = serialize_root_signature(root_desc);
    if (serialized.empty() ||
        FAILED(dev->CreateRootSignature(0, serialized.data(), serialized.size(), IID_PPV_ARGS(&root.p))))
        return false;
    D3D12_COMPUTE_PIPELINE_STATE_DESC desc{};
    desc.pRootSignature = root.p;
    desc.CS = {shader.data(), shader.size()};
    Ref<ID3D12PipelineState> pso;
    if (FAILED(dev->CreateComputePipelineState(&desc, IID_PPV_ARGS(&pso.p)))) return false;
    const D3D12_INDIRECT_ARGUMENT_DESC argument{D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH};
    const D3D12_COMMAND_SIGNATURE_DESC signature_desc{kStride, 1, &argument, 0};
    Ref<ID3D12CommandSignature> signature;
    if (FAILED(dev->CreateCommandSignature(&signature_desc, nullptr, IID_PPV_ARGS(&signature.p))))
        return false;
    const D3D12_HEAP_PROPERTIES upload_heap{D3D12_HEAP_TYPE_UPLOAD};
    const D3D12_HEAP_PROPERTIES default_heap{D3D12_HEAP_TYPE_DEFAULT};
    const D3D12_HEAP_PROPERTIES readback_heap{D3D12_HEAP_TYPE_READBACK};
    Ref<ID3D12Resource> source, args, counts, output, readback;
    const auto create = [&](Ref<ID3D12Resource>& dest, D3D12_HEAP_PROPERTIES heap,
                            D3D12_RESOURCE_STATES state, D3D12_RESOURCE_FLAGS flags) {
        const auto buffer = buffer_desc(256, flags);
        return SUCCEEDED(dev->CreateCommittedResource(
            &heap, D3D12_HEAP_FLAG_NONE, &buffer, state, nullptr, IID_PPV_ARGS(&dest.p)));
    };
    if (!create(source, upload_heap, D3D12_RESOURCE_STATE_GENERIC_READ, D3D12_RESOURCE_FLAG_NONE) ||
        !create(args, default_heap, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_FLAG_NONE) ||
        !create(counts, default_heap, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_FLAG_NONE) ||
        !create(output, default_heap, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS) ||
        !create(readback, readback_heap, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_FLAG_NONE))
        return false;
    std::array<UINT, 64> input{};
    for (UINT i = 0; i < kMaximum; ++i) {
        const UINT base = (kArgsOffset + i * kStride) / 4;
        input[base] = i + 1;   // number of groups in the record, 1 then 2 then 3
        input[base + 1] = input[base + 2] = 1;
        if (clamp_fixture) {
            // Budget 2: (1,4,4)->(1,2,1), (2,0,1)->empty, (3,1,1)->(2,1,1).
            if (i == 0) input[base + 1] = input[base + 2] = 4;
            if (i == 1) input[base + 1] = 0;
        }
        input[base + 3] = 0xa5a5a5a5;  // padding: must be ignored
        input[base + 4] = 0x5a5a5a5a;
    }
    input[40] = count;      // source offset 160, copied to destination +44
    input[41] = 3;          // update count from source offset 164
    void* mapped = nullptr;
    if (FAILED(source->Map(0, nullptr, &mapped)) || !mapped) return false;
    std::memcpy(mapped, input.data(), sizeof(input));
    source->Unmap(0, nullptr);

    const D3D12_COMMAND_QUEUE_DESC queue_desc{D3D12_COMMAND_LIST_TYPE_COMPUTE};
    Ref<ID3D12CommandQueue> queue;
    Ref<ID3D12CommandAllocator> allocator;
    Ref<ID3D12GraphicsCommandList> list;
    Ref<ID3D12Fence> fence;
    if (FAILED(dev->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue.p))) ||
        FAILED(dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE, IID_PPV_ARGS(&allocator.p))) ||
        FAILED(dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, allocator.p,
                                     pso.p, IID_PPV_ARGS(&list.p))) ||
        FAILED(dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))))
        return false;
    list->CopyBufferRegion(args.p, kArgsOffset, source.p, kArgsOffset, kMaximum * kStride);
    list->CopyBufferRegion(counts.p, kCountOffset, source.p, 160, 4);
    list->CopyBufferRegion(output.p, 0, source.p, 0, 8); // initialize the UAV to zero
    const D3D12_RESOURCE_BARRIER ready[] = {
        transition(args.p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT),
        transition(counts.p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT),
        transition(output.p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)};
    list->ResourceBarrier(3, ready);
    list->SetComputeRootSignature(root.p);
    list->SetPipelineState(pso.p);
    list->SetComputeRootUnorderedAccessView(0, output->GetGPUVirtualAddress());
    list->ExecuteIndirect(signature.p, kMaximum, args.p, kArgsOffset, fixed ? nullptr : counts.p, fixed ? 0 : kCountOffset);
    if (gpu_update) {
        auto to_copy = transition(counts.p, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
                                  D3D12_RESOURCE_STATE_COPY_DEST);
        list->ResourceBarrier(1, &to_copy);
        list->CopyBufferRegion(counts.p, kCountOffset, source.p, 164, 4);
        auto to_indirect = transition(counts.p, D3D12_RESOURCE_STATE_COPY_DEST,
                                      D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT);
        list->ResourceBarrier(1, &to_indirect);
        // Repeat with a count produced by a GPU copy after the previous indirect dispatch.
        list->ExecuteIndirect(signature.p, kMaximum, args.p, kArgsOffset, fixed ? nullptr : counts.p, fixed ? 0 : kCountOffset);
    }
    const auto to_copy = transition(output.p, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                    D3D12_RESOURCE_STATE_COPY_SOURCE);
    list->ResourceBarrier(1, &to_copy);
    list->CopyBufferRegion(readback.p, 0, output.p, 0, 8);
    if (FAILED(list->Close())) return false;
    ID3D12CommandList* lists[] = {list.p};
    queue->ExecuteCommandLists(1, lists);
    if (shim && rt::d3d12::last_execute_result(queue.p) != leonida::ReplayResult::success) return false;
    if (FAILED(queue->Signal(fence.p, 1))) return false;
    const HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!event) return false;
    const bool completed = SUCCEEDED(fence->SetEventOnCompletion(1, event)) &&
                           WaitForSingleObject(event, 20000) == WAIT_OBJECT_0;
    CloseHandle(event);
    if (!completed || FAILED(readback->Map(0, nullptr, &mapped)) || !mapped) return false;
    std::memcpy(result.data(), mapped, sizeof(result));
    readback->Unmap(0, nullptr);
    return true;
}

// The second dispatch of a gpu_update run reads the count written by the copy before it (3).
// With second_count == count instead, this is what a host that reuses the first count gives.
std::array<UINT, 2> expected(UINT count, bool update, UINT second_count = 3) {
    auto result = std::array<UINT, 2>{};
    const auto accumulate = [&](UINT c) {
        for (UINT i = 0; i < std::min(c, kMaximum); ++i) {
            result[0] += i + 1;
            result[1] += (i + 1) * (i + 2) / 2;
        }
    };
    accumulate(count);
    if (update) accumulate(second_count);
    return result;
}

// --bundle: a fixed-count DISPATCH ExecuteIndirect recorded in a bundle and executed twice by a direct list. The
// bundle starts with the compute pipeline it was created with and has no root signature or root UAV of its own; the
// parent binds the root UAV at slot 0 for the first ExecuteBundle and at slot 1 for the second, then dispatches one
// group directly at slot 2. Each slot holds the shader's two counters (8 bytes, 64 bytes apart), so a slot shows
// which root arguments the indirect records ran with: three records of 1, 2 and 3 groups give 6/10 in slot 0. After
// the second ExecuteBundle a UAV barrier and a direct Dispatch(1, 1, 1) run before any rebind, so slot 1 holds 7/11
// only if the parent's root UAV survived the bundle; the last direct Dispatch(1, 1, 1) gives 1/1 in slot 2 (the
// bundle's pipeline stays on the parent, the parent's root signature and arguments stay as they were).
// `same_root`: the bundle sets the parent's root signature before the call, which keeps the parent's root arguments
// (docs/d3d12_vulkan.md, bundles).
// --bundle-direct (`direct`): the same bundle records Dispatch(3, 1, 1) instead of the indirect call, so slot 0
// holds 3/6 (groups 0, 1 and 2 add 1, 2 and 3), slot 1 holds 4/7 (the direct dispatch after the second bundle
// adds 1/1) and slot 2 still holds 1/1.
constexpr UINT kSlot = 64;  // bytes between the slots' counters
using BundleResult = std::array<UINT, 6>;

bool run_bundle(ID3D12Device* dev, const std::vector<std::uint8_t>& shader, bool same_root, BundleResult& result,
                bool shim, bool direct = false) {
    Ref<ID3D12RootSignature> root;
    D3D12_ROOT_PARAMETER param{};
    param.ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
    param.Descriptor = {0, 0};
    param.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    D3D12_VERSIONED_ROOT_SIGNATURE_DESC root_desc{};
    root_desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_0;
    root_desc.Desc_1_0 = {1, &param, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
    const auto serialized = serialize_root_signature(root_desc);
    if (serialized.empty() ||
        FAILED(dev->CreateRootSignature(0, serialized.data(), serialized.size(), IID_PPV_ARGS(&root.p))))
        return false;
    D3D12_COMPUTE_PIPELINE_STATE_DESC desc{};
    desc.pRootSignature = root.p;
    desc.CS = {shader.data(), shader.size()};
    Ref<ID3D12PipelineState> pso;
    if (FAILED(dev->CreateComputePipelineState(&desc, IID_PPV_ARGS(&pso.p)))) return false;
    const D3D12_INDIRECT_ARGUMENT_DESC argument{D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH};
    const D3D12_COMMAND_SIGNATURE_DESC signature_desc{kStride, 1, &argument, 0};
    Ref<ID3D12CommandSignature> signature;
    if (FAILED(dev->CreateCommandSignature(&signature_desc, nullptr, IID_PPV_ARGS(&signature.p)))) return false;
    const D3D12_HEAP_PROPERTIES upload_heap{D3D12_HEAP_TYPE_UPLOAD};
    const D3D12_HEAP_PROPERTIES default_heap{D3D12_HEAP_TYPE_DEFAULT};
    const D3D12_HEAP_PROPERTIES readback_heap{D3D12_HEAP_TYPE_READBACK};
    Ref<ID3D12Resource> source, args, output, readback;
    const auto create = [&](Ref<ID3D12Resource>& dest, D3D12_HEAP_PROPERTIES heap, D3D12_RESOURCE_STATES state,
                            D3D12_RESOURCE_FLAGS flags) {
        const auto buffer = buffer_desc(512, flags);
        return SUCCEEDED(dev->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &buffer, state, nullptr,
                                                      IID_PPV_ARGS(&dest.p)));
    };
    if (!create(source, upload_heap, D3D12_RESOURCE_STATE_GENERIC_READ, D3D12_RESOURCE_FLAG_NONE) ||
        !create(args, default_heap, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_FLAG_NONE) ||
        !create(output, default_heap, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS) ||
        !create(readback, readback_heap, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_FLAG_NONE))
        return false;
    std::array<UINT, 128> input{};  // bytes 256..511 stay zero: the output's initial value
    for (UINT i = 0; i < kMaximum; ++i) {
        const UINT base = (kArgsOffset + i * kStride) / 4;
        input[base] = i + 1;
        input[base + 1] = input[base + 2] = 1;
        input[base + 3] = 0xa5a5a5a5;  // padding: must be ignored
        input[base + 4] = 0x5a5a5a5a;
    }
    void* mapped = nullptr;
    if (FAILED(source->Map(0, nullptr, &mapped)) || !mapped) return false;
    std::memcpy(mapped, input.data(), sizeof(input));
    source->Unmap(0, nullptr);

    const D3D12_COMMAND_QUEUE_DESC queue_desc{D3D12_COMMAND_LIST_TYPE_DIRECT};
    Ref<ID3D12CommandQueue> queue;
    Ref<ID3D12CommandAllocator> allocator, bundle_allocator;
    Ref<ID3D12GraphicsCommandList> list, bundle;
    Ref<ID3D12Fence> fence;
    if (FAILED(dev->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue.p))) ||
        FAILED(dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p))) ||
        FAILED(dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_BUNDLE, IID_PPV_ARGS(&bundle_allocator.p))) ||
        FAILED(dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_BUNDLE, bundle_allocator.p, pso.p,
                                      IID_PPV_ARGS(&bundle.p))) ||
        FAILED(dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, nullptr,
                                      IID_PPV_ARGS(&list.p))) ||
        FAILED(dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))))
        return false;
    if (same_root) bundle->SetComputeRootSignature(root.p);
    if (direct) bundle->Dispatch(kMaximum, 1, 1);
    else bundle->ExecuteIndirect(signature.p, kMaximum, args.p, kArgsOffset, nullptr, 0);
    if (FAILED(bundle->Close())) return false;

    list->CopyBufferRegion(args.p, kArgsOffset, source.p, kArgsOffset, kMaximum * kStride);
    list->CopyBufferRegion(output.p, 0, source.p, 256, 3 * kSlot);
    const D3D12_RESOURCE_BARRIER ready[] = {
        transition(args.p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT),
        transition(output.p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)};
    list->ResourceBarrier(2, ready);
    list->SetComputeRootSignature(root.p);
    const D3D12_GPU_VIRTUAL_ADDRESS va = output->GetGPUVirtualAddress();
    list->SetComputeRootUnorderedAccessView(0, va);
    list->ExecuteBundle(bundle.p);
    list->SetComputeRootUnorderedAccessView(0, va + kSlot);
    list->ExecuteBundle(bundle.p);
    // No rebind before this direct dispatch: it must still find the parent's root UAV (slot 1) after the bundle.
    D3D12_RESOURCE_BARRIER uav{};
    uav.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    uav.UAV.pResource = output.p;
    list->ResourceBarrier(1, &uav);
    list->Dispatch(1, 1, 1);
    list->SetComputeRootUnorderedAccessView(0, va + 2 * kSlot);
    list->Dispatch(1, 1, 1);
    const auto to_copy = transition(output.p, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
    list->ResourceBarrier(1, &to_copy);
    list->CopyBufferRegion(readback.p, 0, output.p, 0, 3 * kSlot);
    if (FAILED(list->Close())) return false;
    ID3D12CommandList* lists[] = {list.p};
    queue->ExecuteCommandLists(1, lists);
    if (shim && rt::d3d12::last_execute_result(queue.p) != leonida::ReplayResult::success) return false;
    if (FAILED(queue->Signal(fence.p, 1))) return false;
    const HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!event) return false;
    const bool completed = SUCCEEDED(fence->SetEventOnCompletion(1, event)) &&
                           WaitForSingleObject(event, 20000) == WAIT_OBJECT_0;
    CloseHandle(event);
    if (!completed || FAILED(dev->GetDeviceRemovedReason()) || FAILED(readback->Map(0, nullptr, &mapped)) || !mapped)
        return false;
    const auto* words = static_cast<const UINT*>(mapped);
    for (UINT slot = 0; slot < 3; ++slot) {
        result[2 * slot] = words[slot * kSlot / 4];
        result[2 * slot + 1] = words[slot * kSlot / 4 + 1];
    }
    readback->Unmap(0, nullptr);
    return true;
}

// What a bundle's Close returns for the ExecuteIndirect calls a bundle may not make: a count buffer, and a signature
// that writes a root constant before its DISPATCH. Only Close is called: executing a bundle whose Close failed
// crashes the PC runtime (d3d12_bundle_forbidden_test).
bool bundle_refusals(ID3D12Device* dev, HRESULT& with_count, HRESULT& with_constant) {
    D3D12_ROOT_PARAMETER params[2] = {};
    params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
    params[0].Descriptor = {0, 0};
    params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    params[1].Constants = {0, 0, 1};
    D3D12_VERSIONED_ROOT_SIGNATURE_DESC root_desc{};
    root_desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_0;
    root_desc.Desc_1_0 = {2, params, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
    const auto serialized = serialize_root_signature(root_desc);
    Ref<ID3D12RootSignature> root;
    if (serialized.empty() ||
        FAILED(dev->CreateRootSignature(0, serialized.data(), serialized.size(), IID_PPV_ARGS(&root.p))))
        return false;
    const D3D12_INDIRECT_ARGUMENT_DESC dispatch{D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH};
    D3D12_INDIRECT_ARGUMENT_DESC compound[2] = {};
    compound[0].Type = D3D12_INDIRECT_ARGUMENT_TYPE_CONSTANT;
    compound[0].Constant = {1, 0, 1};
    compound[1].Type = D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH;
    const D3D12_COMMAND_SIGNATURE_DESC simple_desc{kStride, 1, &dispatch, 0};
    const D3D12_COMMAND_SIGNATURE_DESC compound_desc{kStride, 2, compound, 0};
    Ref<ID3D12CommandSignature> simple, constant;
    if (FAILED(dev->CreateCommandSignature(&simple_desc, nullptr, IID_PPV_ARGS(&simple.p))) ||
        FAILED(dev->CreateCommandSignature(&compound_desc, root.p, IID_PPV_ARGS(&constant.p))))
        return false;
    const D3D12_HEAP_PROPERTIES default_heap{D3D12_HEAP_TYPE_DEFAULT};
    const auto buffer = buffer_desc(256);
    Ref<ID3D12Resource> args;
    if (FAILED(dev->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &buffer,
                                            D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT, nullptr, IID_PPV_ARGS(&args.p))))
        return false;
    const auto close = [&](ID3D12CommandSignature* sig, ID3D12Resource* count, HRESULT& hr) {
        Ref<ID3D12CommandAllocator> allocator;
        Ref<ID3D12GraphicsCommandList> bundle;
        if (FAILED(dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_BUNDLE, IID_PPV_ARGS(&allocator.p))) ||
            FAILED(dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_BUNDLE, allocator.p, nullptr,
                                          IID_PPV_ARGS(&bundle.p))))
            return false;
        if (sig == constant.p) bundle->SetComputeRootSignature(root.p);
        bundle->ExecuteIndirect(sig, 1, args.p, 0, count, count ? 128 : 0);
        hr = bundle->Close();
        return true;
    };
    return close(simple.p, args.p, with_count) && close(constant.p, nullptr, with_constant);
}

// Build and revision of a loaded system DLL ("22621.5415"; Windows shows it as 10.0.22621.5415), or "unknown".
std::string module_version(const wchar_t* name) {
    const HMODULE module = GetModuleHandleW(name);
    wchar_t path[MAX_PATH]{};
    if (!module || !GetModuleFileNameW(module, path, MAX_PATH)) return "unknown";
    DWORD ignored = 0;
    const DWORD size = GetFileVersionInfoSizeW(path, &ignored);
    std::vector<std::uint8_t> data(size);
    VS_FIXEDFILEINFO* info = nullptr;
    UINT length = 0;
    if (!size || !GetFileVersionInfoW(path, 0, size, data.data()) ||
        !VerQueryValueW(data.data(), L"\\", reinterpret_cast<void**>(&info), &length) || !info)
        return "unknown";
    char text[64];
    std::snprintf(text, sizeof text, "%u.%u", HIWORD(info->dwFileVersionLS), LOWORD(info->dwFileVersionLS));
    return text;
}
}  // namespace

int make_host(Ref<ID3D12Device>& host, std::vector<std::uint8_t>& native_code) {
    using Compile = decltype(&D3DCompile);
    const HMODULE compiler = LoadLibraryExW(L"d3dcompiler_47.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    if (!compiler) return 77;
    const auto compile = reinterpret_cast<Compile>(GetProcAddress(compiler, "D3DCompile"));
    if (!compile) return 77;
    Ref<ID3DBlob> native, errors;
    if (FAILED(compile(native_shader, std::strlen(native_shader), nullptr, nullptr, nullptr,
                       "main", "cs_5_0", 0, 0, &native.p, &errors.p)) || !native.p)
        return 1;
    const auto* native_bytes = static_cast<const std::uint8_t*>(native->GetBufferPointer());
    native_code.assign(native_bytes, native_bytes + native->GetBufferSize());
    Ref<IDXGIFactory4> factory;
    Ref<IDXGIAdapter> warp;

    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory.p))) ||
        FAILED(factory->EnumWarpAdapter(IID_PPV_ARGS(&warp.p))) ||
        FAILED(D3D12CreateDevice(warp.p, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&host.p))))
    return 77;

    return 0;
}

int main(int argc, char** argv) {
    const bool clamp = argc == 2 && !std::strcmp(argv[1], "--clamp");
    const bool direct = argc == 2 && !std::strcmp(argv[1], "--bundle-direct");
    const bool bundle = direct || (argc == 2 && !std::strcmp(argv[1], "--bundle"));
    if (argc != 1 && !clamp && !bundle) return 2;
    // Wine has no WARP (its host is vkd3d on the GPU); the layer is still checked against expected().
    const bool compare_host = !hostwine::host_is_wine();
    Ref<ID3D12Device> host;
    std::vector<std::uint8_t> native_code;
    if (!clamp && compare_host) {
        if (const int status = make_host(host, native_code)) return status;
    }
    // --bundle and --bundle-direct run on WARP and, outside Wine, on the default adapter too: both are held to the
    // same values.
    const BundleResult bundle_want = direct ? BundleResult{3, 6, 4, 7, 1, 1} : BundleResult{6, 10, 7, 11, 1, 1};
    const char* const want_text = direct ? "3/6 4/7 1/1" : "6/10 7/11 1/1";
    unsigned bundle_hosts = 0;
    if (bundle && compare_host) {
        Ref<ID3D12Device> adapter_default;
        ID3D12Device* devices[] = {host.p, nullptr};
        const char* names[] = {"WARP", "the default adapter"};
        if (SUCCEEDED(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&adapter_default.p))))
            devices[1] = adapter_default.p;
        for (int i = 0; i < 2; ++i) {
            if (!devices[i]) continue;
            if (!direct) {
                HRESULT with_count = S_OK, with_constant = S_OK;
                if (!bundle_refusals(devices[i], with_count, with_constant)) return 1;
                std::printf("%s: bundle ExecuteIndirect Close with a count buffer 0x%08lx, with a root constant "
                            "0x%08lx\n", names[i], static_cast<unsigned long>(with_count),
                            static_cast<unsigned long>(with_constant));
                if (SUCCEEDED(with_count) || SUCCEEDED(with_constant)) return 1;
            }
            for (const bool same_root : {false, true}) {
                BundleResult got{};
                if (!run_bundle(devices[i], native_code, same_root, got, false, direct) || got != bundle_want) {
                    std::fprintf(stderr, "%s same_root=%d: %u/%u %u/%u %u/%u, expected %s\n", names[i],
                                 int(same_root), got[0], got[1], got[2], got[3], got[4], got[5], want_text);
                    return 1;
                }
            }
            ++bundle_hosts;
        }
    }
    vktest::Device vk;
    vk.request_buffer_device_address = true;
    const auto status = vk.initialize();
    if (status == vktest::DeviceStatus::unavailable) return 77;
    if (status != vktest::DeviceStatus::ready) return 1;
    if (!vk.buffer_device_address) return 77;
    std::vector<std::uint8_t> program = fssm_program({0x1100u, 0x44u, 0x55u, 0x66u, 0x77u, 0x88u});
    for (UINT i = 0; i < 12; ++i) put32(program, 0xbf810000 + i);
    const auto container = shader_container(program);
    const auto fixture = std::filesystem::temp_directory_path() /
                         ("leonida_indirect_dispatch_count_" + std::to_string(GetCurrentProcessId()));
    std::filesystem::create_directories(fixture);
    std::filesystem::copy_file(vktest::shader_fixture_path(LEONIDA_TEST_INDIRECT_COUNT_SPV),
                               fixture / "count.spv", std::filesystem::copy_options::overwrite_existing);
    {
        std::ofstream index(fixture / "index.json", std::ios::binary);
        index << "{\"count\":{\"stage\":\"cs\",\"fssm_sha256\":\"" << leonida::sha256_hex(program)
              << "\",\"bindings\":[],\"push_constant_bytes\":144,\"local_size\":[1,1,1],"
                 "\"uses_code_base\":false,\"wave64\":false}}";
    }
    rt::d3d12::VulkanDeviceInfo info{};
    info.get_instance_proc = vk.instance_proc;
    info.instance = vk.instance;
    info.physical = vk.physical;
    info.device = vk.device;
    info.queue = vk.queue;
    info.queue_family = vk.family;
    info.buffer_device_address = vk.buffer_device_address;
    info.shader_dir = fixture.string();
    Ref<ID3D12Device> layer;
    if (FAILED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&layer.p)))) return 1;

    if (bundle) {
        HRESULT with_count = E_INVALIDARG, with_constant = E_INVALIDARG;
        if ((!direct && !bundle_refusals(layer.p, with_count, with_constant)) || with_count != E_INVALIDARG ||
            with_constant != E_INVALIDARG) {
            std::fprintf(stderr, "layer: bundle ExecuteIndirect Close with a count buffer 0x%08lx, with a root "
                         "constant 0x%08lx, expected E_INVALIDARG for both\n",
                         static_cast<unsigned long>(with_count), static_cast<unsigned long>(with_constant));
            return 1;
        }
        for (const bool same_root : {false, true}) {
            BundleResult got{};
            if (!run_bundle(layer.p, container, same_root, got, true, direct) || got != bundle_want) {
                std::fprintf(stderr, "layer same_root=%d: %u/%u %u/%u %u/%u, expected %s\n",
                             int(same_root), got[0], got[1], got[2], got[3], got[4], got[5], want_text);
                return 1;
            }
        }
        if (vk.validation_errors()) {
            std::fprintf(stderr, "validation errors: %u\n", vk.validation_errors());
            return 1;
        }
        if (direct)
            std::printf("bundle Dispatch: layer and %u host adapter(s) gave 3/6 4/7 1/1 in both cases\n", bundle_hosts);
        else
            std::printf("bundle ExecuteIndirect(DISPATCH): layer and %u host adapter(s) gave 6/10 7/11 1/1 in both "
                        "cases; count buffer and root constant refused\n", bundle_hosts);
        return 0;
    }
    if (clamp) {
        for (bool fixed : {false, true}) {
            for (UINT count : {0u, 1u, 7u}) {
                const auto want = fixed || count >= 3 ? std::array<UINT, 2>{4, 5} :
                                  count ? std::array<UINT, 2>{2, 2} : std::array<UINT, 2>{};
                std::array<UINT, 2> result{};
                if (!run(layer.p, container, count, false, result, true, fixed, true) || result != want) {
                    std::fprintf(stderr, "group budget: fixed=%d count=%u got %u/%u expected %u/%u\n",
                                 int(fixed), count, result[0], result[1], want[0], want[1]);
                    return 1;
                }
            }
        }
        std::puts("6 indirect group budget cases passed");
        return vk.validation_errors() ? 1 : 0;
    }
    // The layer is checked against expected(), the D3D12 rule that the count is read when the GPU executes the
    // command. WARP must match it too, with one measured exception: in-box WARP 10.0.22621.5415 (Windows 11 23H2)
    // reuses the first count for the second ExecuteIndirect after the GPU copy overwrites it (0 -> 3 gives 0/0,
    // 1 -> 3 gives 2/2). That case is reported, not failed, when WARP gives exactly the stale-count result.
    unsigned cases = 0, stale_host = 0;
    for (UINT count : {0u, 1u, 3u, 7u}) {
        for (bool update : {false, true}) {
            std::array<UINT, 2> native_result{}, layer_result{};
            const auto want = expected(count, update);
            if (!compare_host) native_result = want;  // no host run: the layer alone is held to `want`
            if ((compare_host && !run(host.p, native_code, count, update, native_result, false)) ||
                !run(layer.p, container, count, update, layer_result, true) || layer_result != want ||
                (native_result != want && !(update && native_result == expected(count, update, count)))) {
                std::fprintf(stderr, "count %u update %d: host %u/%u%s, layer %u/%u, expected %u/%u\n",
                             count, int(update), native_result[0], native_result[1],
                             compare_host ? "" : " (not run)", layer_result[0], layer_result[1], want[0], want[1]);
                return 1;
            }
            if (native_result != want) {
                std::printf("count %u update 1: host WARP build %s reused the first count (%u/%u, expected %u/%u); "
                            "the layer matches the expected result\n",
                            count, module_version(L"d3d10warp.dll").c_str(), native_result[0], native_result[1],
                            want[0], want[1]);
                ++stale_host;
            }
            ++cases;
        }
    }
    if (stale_host)
        std::printf("%u GPU-count-overwrite case(s) checked against the expected result only (stale WARP count)\n",
                    stale_host);
    if (vk.validation_errors()) {
        std::fprintf(stderr, "validation errors: %u\n", vk.validation_errors());
        return 1;
    }
    std::printf("counted ExecuteIndirect(DISPATCH): %u %s passed; "
                "count=0,1,max,>max, nonzero offsets and GPU count overwrite, no validation errors\n", cases,
                compare_host ? "native WARP / Vulkan GPU comparisons"
                             : "Vulkan GPU results against the expected values (no WARP runs under Wine)");
    return 0;
}
