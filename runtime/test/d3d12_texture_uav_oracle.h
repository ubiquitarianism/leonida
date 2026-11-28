#pragma once

// Native Windows D3D12/WARP reference for synthetic texture UAV dispatches.
// Link d3d12 and dxgi; the HLSL compiler is loaded from System32.
#include <windows.h>
#include <d3d12.h>
#include <d3dcompiler.h>
#include <dxgi1_4.h>
#include <wrl/client.h>

#include <array>
#include <bit>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <span>

namespace uav_oracle {
enum class Status { ready, unavailable, failed };

namespace detail {
using Microsoft::WRL::ComPtr;

inline bool fail(const char* message) {
    std::fprintf(stderr, "WARP texture UAV: %s\n", message);
    return false;
}

inline bool check(HRESULT hr, const char* action, ID3DBlob* errors = nullptr) {
    if (SUCCEEDED(hr)) return true;
    std::fprintf(stderr, "WARP texture UAV: %s failed (0x%08lx)\n", action, static_cast<unsigned long>(hr));
    if (errors) std::fputs(static_cast<const char*>(errors->GetBufferPointer()), stderr);
    return false;
}

struct View {
    unsigned shader;
    UINT mip;
};

inline View select_view(const D3D12_RESOURCE_DESC& resource, const D3D12_UNORDERED_ACCESS_VIEW_DESC* desc) {
    if (desc) {
        switch (desc->ViewDimension) {
            case D3D12_UAV_DIMENSION_TEXTURE1D:      return {0, desc->Texture1D.MipSlice};
            case D3D12_UAV_DIMENSION_TEXTURE1DARRAY: return {1, desc->Texture1DArray.MipSlice};
            case D3D12_UAV_DIMENSION_TEXTURE2D:      return {2, desc->Texture2D.MipSlice};
            case D3D12_UAV_DIMENSION_TEXTURE2DARRAY: return {3, desc->Texture2DArray.MipSlice};
            case D3D12_UAV_DIMENSION_TEXTURE3D:      return {4, desc->Texture3D.MipSlice};
            default:                               return {5, 0};
        }
    }
    switch (resource.Dimension) {
        case D3D12_RESOURCE_DIMENSION_TEXTURE1D: return {resource.DepthOrArraySize > 1 ? 1u : 0u, 0};
        case D3D12_RESOURCE_DIMENSION_TEXTURE2D: return {resource.DepthOrArraySize > 1 ? 3u : 2u, 0};
        case D3D12_RESOURCE_DIMENSION_TEXTURE3D: return {4, 0};
        default:                              return {5, 0};
    }
}

inline D3D12_HEAP_PROPERTIES heap_properties(D3D12_HEAP_TYPE type) {
    D3D12_HEAP_PROPERTIES heap{};
    heap.Type = type;
    heap.CreationNodeMask = heap.VisibleNodeMask = 1;
    return heap;
}

struct Context {
    ComPtr<ID3D12Device> device;
    ComPtr<ID3D12RootSignature> root;
    std::array<ComPtr<ID3D12PipelineState>, 5> pipelines;

    Status prepare_device() {
        if (device) return Status::ready;
        ComPtr<IDXGIFactory4> factory;
        ComPtr<IDXGIAdapter> warp;
        HRESULT hr = CreateDXGIFactory1(IID_PPV_ARGS(factory.GetAddressOf()));
        if (hr == E_NOINTERFACE || hr == DXGI_ERROR_UNSUPPORTED) return Status::unavailable;
        if (!check(hr, "CreateDXGIFactory1")) return Status::failed;
        hr = factory->EnumWarpAdapter(IID_PPV_ARGS(warp.GetAddressOf()));
        if (hr == DXGI_ERROR_NOT_FOUND || hr == DXGI_ERROR_UNSUPPORTED || hr == E_NOTIMPL)
            return Status::unavailable;
        if (!check(hr, "EnumWarpAdapter")) return Status::failed;
        hr = D3D12CreateDevice(warp.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(device.GetAddressOf()));
        if (hr == DXGI_ERROR_UNSUPPORTED || hr == E_NOTIMPL) return Status::unavailable;
        return check(hr, "D3D12CreateDevice") ? Status::ready : Status::failed;
    }

    bool init() {
        if (root) return true;
        if (prepare_device() != Status::ready) return false;
        const D3D12_DESCRIPTOR_RANGE range{D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0, 0, 0};
        D3D12_ROOT_PARAMETER parameter{};
        parameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        parameter.DescriptorTable = {1, &range};
        parameter.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        const D3D12_ROOT_SIGNATURE_DESC desc{1, &parameter, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
        ComPtr<ID3DBlob> blob, errors;
        const HRESULT hr = D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1_0,
                                                       blob.GetAddressOf(), errors.GetAddressOf());
        return check(hr, "D3D12SerializeRootSignature", errors.Get()) &&
               check(device->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(),
                                                 IID_PPV_ARGS(root.GetAddressOf())), "CreateRootSignature");
    }

    bool create_pipeline(unsigned index) {
        if (pipelines[index]) return true;
        static constexpr const char* types[] = {
            "RWTexture1D", "RWTexture1DArray", "RWTexture2D", "RWTexture2DArray", "RWTexture3D",
        };
        static constexpr const char* coordinates[] = {"id.x", "id.xz", "id.xy", "id.xyz", "id.xyz"};
        static constexpr char source[] =
            "UAV_TYPE<uint> dst : register(u0);\n"
            "[numthreads(1, 1, 1)]\n"
            "void main(uint3 id : SV_DispatchThreadID) {\n"
            "    dst[UAV_COORD] += 1000u + id.x + 16u * id.y + 256u * id.z;\n"
            "}\n";
        struct Compiler {
            HMODULE module = LoadLibraryExW(L"d3dcompiler_47.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
            ~Compiler() { if (module) FreeLibrary(module); }
        } compiler;
        if (!compiler.module) return check(HRESULT_FROM_WIN32(GetLastError()), "load d3dcompiler_47.dll");
        auto compile = std::bit_cast<pD3DCompile>(GetProcAddress(compiler.module, "D3DCompile"));
        if (!compile) return check(HRESULT_FROM_WIN32(GetLastError()), "find D3DCompile");
        const D3D_SHADER_MACRO macros[] = {
            {"UAV_TYPE", types[index]}, {"UAV_COORD", coordinates[index]}, {nullptr, nullptr},
        };
        ComPtr<ID3DBlob> blob, errors;
        const HRESULT hr = compile(source, sizeof(source) - 1, "texture_uav_oracle.hlsl", macros, nullptr, "main",
                                   "cs_5_1", D3DCOMPILE_ENABLE_STRICTNESS, 0, blob.GetAddressOf(), errors.GetAddressOf());
        if (!check(hr, "D3DCompile", errors.Get())) return false;
        D3D12_COMPUTE_PIPELINE_STATE_DESC desc{};
        desc.pRootSignature = root.Get();
        desc.CS = {blob->GetBufferPointer(), blob->GetBufferSize()};
        return check(device->CreateComputePipelineState(&desc, IID_PPV_ARGS(pipelines[index].GetAddressOf())),
                     "CreateComputePipelineState");
    }

    HRESULT create_buffer(UINT64 bytes, D3D12_HEAP_TYPE heap_type, D3D12_RESOURCE_STATES state, ID3D12Resource** out) {
        const auto heap = heap_properties(heap_type);
        D3D12_RESOURCE_DESC desc{};
        desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        desc.Width = bytes;
        desc.Height = desc.DepthOrArraySize = desc.MipLevels = 1;
        desc.SampleDesc.Count = 1;
        desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        return device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, state, nullptr, IID_PPV_ARGS(out));
    }
};

inline Context& context() {
    static thread_local Context ctx;
    return ctx;
}
} // namespace detail

inline Status prepare() {
    return detail::context().prepare_device();
}

// Provide a valid R32_UINT or R32_TYPELESS texture descriptor and a valid UINT UAV view; a null view
// asks D3D12 to create the default view of a typed R32_UINT resource. Both spans cover the entire
// selected mip (x fastest, then y, then array layer or 3D depth). Other mips are neither read nor written.
inline bool run(const D3D12_RESOURCE_DESC& resource_desc, const D3D12_UNORDERED_ACCESS_VIEW_DESC* desc,
                std::span<const std::uint32_t> before, std::span<std::uint32_t> after, UINT dispatch_depth = 0) {
    using namespace detail;
    const View view = select_view(resource_desc, desc);
    auto& ctx = context();
    if (view.shader >= ctx.pipelines.size() ||
        (resource_desc.Format != DXGI_FORMAT_R32_UINT && resource_desc.Format != DXGI_FORMAT_R32_TYPELESS))
        return fail("expected an R32_UINT or R32_TYPELESS texture and a texture UAV");
    if (!ctx.init() || !ctx.create_pipeline(view.shader)) return false;

    ComPtr<ID3D12Resource> texture, upload, readback;
    const auto heap = heap_properties(D3D12_HEAP_TYPE_DEFAULT);
    if (!check(ctx.device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &resource_desc,
               D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(texture.GetAddressOf())),
               "CreateCommittedResource (texture)")) return false;

    const auto actual_desc = texture->GetDesc(); // MipLevels = 0 in the input requests the full chain.
    if (view.mip >= actual_desc.MipLevels) return fail("UAV mip slice is out of range");
    const bool volume = actual_desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D;
    const UINT subresources = volume ? 1 : actual_desc.DepthOrArraySize;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
    UINT64 bytes = 0;
    ctx.device->GetCopyableFootprints(&actual_desc, view.mip, 1, 0, &footprint, nullptr, nullptr, &bytes);
    if (bytes == (std::numeric_limits<UINT64>::max)()) return fail("GetCopyableFootprints rejected the texture");
    const auto& fp = footprint.Footprint;
    const UINT depth = volume ? fp.Depth : subresources;
    const std::uint64_t texels = std::uint64_t(fp.Width) * fp.Height * depth;
    if (before.size() != texels || after.size() != texels)
        return fail("input and output must span the whole selected mip");
    // Array layers are separate subresources; a 3D mip is one subresource with depth slices at rowPitch * height.
    constexpr UINT64 alignment = D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT;
    const UINT64 layer_pitch = (bytes + alignment - 1) & ~(alignment - 1);
    const UINT64 total = bytes + (subresources - 1) * layer_pitch;
    if (total > (std::numeric_limits<SIZE_T>::max)()) return fail("staging buffers exceed the host address space");
    const auto z_pitch = static_cast<SIZE_T>(volume ? UINT64(fp.RowPitch) * fp.Height : layer_pitch);
    const SIZE_T row_bytes = SIZE_T(fp.Width) * sizeof(std::uint32_t);
    if (!check(ctx.create_buffer(total, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ,
                                 upload.GetAddressOf()), "CreateCommittedResource (upload)") ||
        !check(ctx.create_buffer(total, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST,
                                 readback.GetAddressOf()), "CreateCommittedResource (readback)")) return false;
    const auto each_row = [&](auto copy) {
        for (UINT z = 0; z < depth; ++z)
            for (UINT y = 0; y < fp.Height; ++y)
                copy((SIZE_T(z) * fp.Height + y) * fp.Width, z * z_pitch + SIZE_T(y) * fp.RowPitch);
    };
    const D3D12_RANGE none{}, all{0, static_cast<SIZE_T>(total)};
    void* mapped = nullptr;
    if (!check(upload->Map(0, &none, &mapped), "Map upload")) return false;
    each_row([&](SIZE_T word, SIZE_T offset) {
        std::memcpy(static_cast<char*>(mapped) + offset, before.data() + word, row_bytes);
    });
    upload->Unmap(0, &all);

    ComPtr<ID3D12DescriptorHeap> uavs;
    const D3D12_DESCRIPTOR_HEAP_DESC heap_desc{
        D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0,
    };
    if (!check(ctx.device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(uavs.GetAddressOf())), "CreateDescriptorHeap"))
        return false;
    ctx.device->CreateUnorderedAccessView(texture.Get(), nullptr, desc, uavs->GetCPUDescriptorHandleForHeapStart());

    // Use fresh submission objects per run, so no allocator or descriptor heap is reset while in use.
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> list;
    ComPtr<ID3D12Fence> fence;
    constexpr auto direct = D3D12_COMMAND_LIST_TYPE_DIRECT;
    const D3D12_COMMAND_QUEUE_DESC queue_desc{direct};
    if (!check(ctx.device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(queue.GetAddressOf())), "CreateCommandQueue") ||
        !check(ctx.device->CreateCommandAllocator(direct, IID_PPV_ARGS(allocator.GetAddressOf())), "CreateCommandAllocator") ||
        !check(ctx.device->CreateCommandList(0, direct, allocator.Get(), ctx.pipelines[view.shader].Get(),
                                            IID_PPV_ARGS(list.GetAddressOf())), "CreateCommandList") ||
        !check(ctx.device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(fence.GetAddressOf())), "CreateFence"))
        return false;
    const auto copy_mip = [&](ID3D12Resource* buffer, bool to_texture) {
        D3D12_TEXTURE_COPY_LOCATION tex{}, staging{};
        tex.pResource = texture.Get();
        tex.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        staging.pResource = buffer;
        staging.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        staging.PlacedFootprint = footprint;
        for (UINT layer = 0; layer < subresources; ++layer) {
            tex.SubresourceIndex = view.mip + layer * actual_desc.MipLevels;
            staging.PlacedFootprint.Offset = layer * layer_pitch;
            list->CopyTextureRegion(to_texture ? &tex : &staging, 0, 0, 0, to_texture ? &staging : &tex, nullptr);
        }
    };
    const auto transition = [&](D3D12_RESOURCE_STATES from, D3D12_RESOURCE_STATES to) {
        D3D12_RESOURCE_BARRIER barrier{};
        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier.Transition = {texture.Get(), D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, from, to};
        list->ResourceBarrier(1, &barrier);
    };
    copy_mip(upload.Get(), true);
    transition(D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    ID3D12DescriptorHeap* heaps[] = {uavs.Get()};
    list->SetDescriptorHeaps(1, heaps);
    list->SetComputeRootSignature(ctx.root.Get());
    list->SetComputeRootDescriptorTable(0, uavs->GetGPUDescriptorHandleForHeapStart());
    list->Dispatch(fp.Width, fp.Height, dispatch_depth ? dispatch_depth : depth);
    transition(D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
    copy_mip(readback.Get(), false);
    if (!check(list->Close(), "Close command list")) return false;
    ID3D12CommandList* lists[] = {list.Get()};
    queue->ExecuteCommandLists(1, lists);
    if (!check(queue->Signal(fence.Get(), 1), "Signal queue fence") ||
        !check(fence->SetEventOnCompletion(1, nullptr), "wait for queue fence") ||
        !check(ctx.device->GetDeviceRemovedReason(), "D3D12 execution")) return false;
    if (!check(readback->Map(0, &all, &mapped), "Map readback")) return false;
    each_row([&](SIZE_T word, SIZE_T offset) {
        std::memcpy(after.data() + word, static_cast<const char*>(mapped) + offset, row_bytes);
    });
    readback->Unmap(0, &none);
    return true;
}
} // namespace uav_oracle
