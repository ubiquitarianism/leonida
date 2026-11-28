// Slot-133 aliases share GPU contents and retain individual COM identities.
#pragma once
#include "d3d12_copy_compare.h"
#include "../shims/d3d12_xs_device.h"

namespace placeddepthtest {

inline bool check_alias_lifetime(ID3D12Device* device, ID3D12Device* xbox, bool opaque = false) {
    using copycmp::Ptr;
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = 256;
    desc.Height = 128;
    desc.DepthOrArraySize = desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Format = DXGI_FORMAT_R32G8X24_TYPELESS;
    desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
    const auto allocation = device->GetResourceAllocationInfo(0, 1, &desc);
    if (!allocation.Alignment || allocation.SizeInBytes == UINT64_MAX || allocation.SizeInBytes <= allocation.Alignment)
        return false;
    struct Pages {
        void* ptr;
        ~Pages() { if (ptr) VirtualFree(ptr, 0, MEM_RELEASE); }
    } pages{VirtualAlloc(nullptr, SIZE_T(allocation.SizeInBytes * 2), MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE)};
    if (!pages.ptr) return false;
    const UINT64 address = reinterpret_cast<UINT64>(pages.ptr);
    auto xbox_desc = desc;
    xbox_desc.Flags |= D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    xbox_desc.Layout = opaque ? D3D12_TEXTURE_LAYOUT_UNKNOWN : static_cast<D3D12_TEXTURE_LAYOUT>(0x101);
    using Create = HRESULT(WINAPI*)(void*, UINT64, const D3D12_RESOURCE_DESC*, UINT,
                                    const D3D12_CLEAR_VALUE*, REFIID, void**);
    const auto create = reinterpret_cast<Create>((*reinterpret_cast<void***>(xbox))[133]);
    const auto place = [&](UINT64 va, const D3D12_RESOURCE_DESC& shape, Ptr<ID3D12Resource>& result,
                           const D3D12_CLEAR_VALUE* clear = nullptr) {
        return create(xbox, va, &shape, UINT_MAX, clear, IID_PPV_ARGS(&result.p));
    };
    if (opaque) {
        auto packed = xbox_desc;
        packed.Format = static_cast<DXGI_FORMAT>(118);
        packed.Width = 1280;
        packed.Height = 720;
        Ptr<ID3D12Resource> target;
        // The image is privately backed; reserve a full placement for its title VA.
        Pages packed_pages{VirtualAlloc(nullptr, 8u << 20, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE)};
        const auto packed_va = reinterpret_cast<UINT64>(packed_pages.ptr);
        if (!packed_va || FAILED(place(packed_va, packed, target)) ||
            target->GetGPUVirtualAddress() != packed_va) return false;
        target.reset();
    }
    Ptr<ID3D12Resource> first, alias, next, invalid;
    if (FAILED(place(address, xbox_desc, first)) || FAILED(place(address, xbox_desc, alias)) ||
        FAILED(place(address + allocation.SizeInBytes, xbox_desc, next))) return false;
    if (first.p == alias.p || alias->GetGPUVirtualAddress() != address ||
        next->GetGPUVirtualAddress() != address + allocation.SizeInBytes) return false;
    // Independent COM private data helps catch accidental reuse of the object itself as the alias.
    constexpr GUID cookie_key{0x650cf350, 0xa07e, 0x428d, {0xb4, 0x14, 0xdc, 0xaf, 0xc6, 0x97, 0xe8, 0x4b}};
    const UINT cookie = 12;
    UINT bytes = 0;
    if (FAILED(first->SetPrivateData(cookie_key, sizeof(cookie), &cookie)) ||
        alias->GetPrivateData(cookie_key, &bytes, nullptr) != DXGI_ERROR_NOT_FOUND) return false;
    auto smaller = xbox_desc;
    smaller.Width /= 2;
    Ptr<ID3D12Resource> other_shape;
    if (opaque && (FAILED(place(address, smaller, other_shape)) || other_shape->GetDesc().Width != smaller.Width ||
                   other_shape->GetGPUVirtualAddress() != address)) return false;
    D3D12_CLEAR_VALUE optimized{};
    optimized.Format = DXGI_FORMAT_D32_FLOAT_S8X24_UINT;
    optimized.DepthStencil = {1.0f, 0};
    if (place(address, xbox_desc, invalid, &optimized) != E_NOTIMPL || invalid.p) return false;
    if ((!opaque && (place(address, smaller, invalid) != E_NOTIMPL || invalid.p)) ||
        place(address + 1, xbox_desc, invalid) != E_INVALIDARG || invalid.p) return false;
    // A placement that starts inside another one's PC allocation: the title's own blocks can be smaller than the
    // PC sizes them (a D16 target on RADV), so the next block starts inside the range. Opaque layouts get a backing
    // of their own there; a layout whose bytes would have to be shared is still refused.
    if (!opaque) {
        if (place(address + allocation.Alignment, xbox_desc, invalid) != E_NOTIMPL || invalid.p) return false;
    } else {
        Ptr<ID3D12Resource> inside;
        if (FAILED(place(address + allocation.Alignment, xbox_desc, inside)) ||
            inside->GetGPUVirtualAddress() != address + allocation.Alignment ||
            rt::d3d12_xs::texture_backing(inside.p) == rt::d3d12_xs::texture_backing(first.p)) {
            std::fprintf(stderr, "slot-133 alias: a placement inside another range was refused or shared\n");
            return false;
        }
        // Placing the first shape again still finds its own backing, with the inner range registered.
        Ptr<ID3D12Resource> again;
        if (FAILED(place(address, xbox_desc, again)) ||
            rt::d3d12_xs::texture_backing(again.p) != rt::d3d12_xs::texture_backing(first.p)) {
            std::fprintf(stderr, "slot-133 alias: the outer range lost its backing to the inner one\n");
            return false;
        }
    }

    Ptr<ID3D12DescriptorHeap> heap;
    D3D12_DESCRIPTOR_HEAP_DESC heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_DSV, 2};
    if (FAILED(device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&heap.p)))) return false;
    D3D12_DEPTH_STENCIL_VIEW_DESC dsv{};
    dsv.Format = DXGI_FORMAT_D32_FLOAT_S8X24_UINT;
    dsv.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
    const auto handle = heap->GetCPUDescriptorHandleForHeapStart();
    device->CreateDepthStencilView(first.p, &dsv, handle);
    auto other_view = handle;
    other_view.ptr += device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_DSV);
    if (opaque) device->CreateDepthStencilView(other_shape.p, &dsv, other_view);
    copycmp::Run commands;
    if (!commands.init(device, true)) return false;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT layouts[2]{};
    UINT64 total = 0;
    device->GetCopyableFootprints(&desc, 0, 2, 0, layouts, nullptr, nullptr, &total);
    Ptr<ID3D12Resource> readback;
    const UINT64 sample_offset = (total + 511) & ~UINT64(511);
    if (!commands.buffer(D3D12_HEAP_TYPE_READBACK, sample_offset + 256, D3D12_RESOURCE_STATE_COPY_DEST, readback)) return false;
    commands.begin();
    if (opaque) {
        D3D12_RESOURCE_BARRIER use{};
        use.Type = D3D12_RESOURCE_BARRIER_TYPE_ALIASING;
        use.Aliasing = {nullptr, other_shape.p};
        commands.list->ResourceBarrier(1, &use);
        commands.list->ClearDepthStencilView(other_view, D3D12_CLEAR_FLAG_DEPTH | D3D12_CLEAR_FLAG_STENCIL, 0.625f, 0x34, 0, nullptr);
        commands.barrier(other_shape.p, D3D12_RESOURCE_STATE_DEPTH_WRITE, D3D12_RESOURCE_STATE_COPY_SOURCE);
        D3D12_TEXTURE_COPY_LOCATION src{}, dst{};
        src.pResource = other_shape.p;
        dst.pResource = readback.p;
        dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        dst.PlacedFootprint = {sample_offset,{DXGI_FORMAT_R32_TYPELESS,1,1,1,256}};
        const D3D12_BOX pixel{0,0,0,1,1,1};
        commands.list->CopyTextureRegion(&dst,0,0,0,&src,&pixel);
        use.Aliasing = {other_shape.p, first.p};
        commands.list->ResourceBarrier(1,&use);
    }
    commands.list->ClearDepthStencilView(handle, D3D12_CLEAR_FLAG_DEPTH | D3D12_CLEAR_FLAG_STENCIL,
                                         0.375f, 0xb7, 0, nullptr);
    D3D12_RESOURCE_BARRIER activate{};
    activate.Type = D3D12_RESOURCE_BARRIER_TYPE_ALIASING;
    activate.Aliasing = {first.p, alias.p};
    commands.list->ResourceBarrier(1, &activate);
    if (!commands.submit().empty()) return false;
    first.reset();
    if (!opaque && (place(address, smaller, invalid) != E_NOTIMPL || invalid.p)) return false;
    commands.begin();
    commands.barrier(alias.p, D3D12_RESOURCE_STATE_DEPTH_WRITE, D3D12_RESOURCE_STATE_COPY_SOURCE);
    for (UINT plane = 0; plane < 2; ++plane) {
        D3D12_TEXTURE_COPY_LOCATION src{}, dst{};
        src.pResource = alias.p;
        src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        src.SubresourceIndex = plane;
        dst.pResource = readback.p;
        dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        dst.PlacedFootprint = layouts[plane];
        commands.list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    }
    if (!commands.submit().empty()) return false;
    unsigned char* mapped = nullptr;
    if (FAILED(readback->Map(0, nullptr, reinterpret_cast<void**>(&mapped)))) return false;
    float earlier = 0;
    if (opaque) std::memcpy(&earlier, mapped + sample_offset, sizeof(earlier));
    bool valid = !opaque || earlier == 0.625f;
    for (UINT y = 0; y < desc.Height && valid; ++y) {
        for (UINT x = 0; x < desc.Width; ++x) {
            float depth;
            std::memcpy(&depth, mapped + layouts[0].Offset + y * layouts[0].Footprint.RowPitch + x * sizeof(depth),
                        sizeof(depth));
            if (depth != 0.375f || mapped[layouts[1].Offset + y * layouts[1].Footprint.RowPitch + x] != 0xb7) {
                std::fprintf(stderr, "slot-133 alias: wrong readback at (%u, %u)\n", x, y);
                valid = false;
                break;
            }
        }
    }
    readback->Unmap(0, nullptr);
    alias.reset();
    return valid && SUCCEEDED(place(address, smaller, invalid)) && invalid->GetGPUVirtualAddress() == address;
}

} // namespace placeddepthtest
