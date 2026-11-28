#include "vk_test_device.h"
#include "d3d12_test_shaders.h"
#include "../shims/d3d12_com.h"
#include "../shims/d3d12_xs_device.h"
#include "../src/depth24.h"
#include "../src/title_va_alias.h"
#include <array>
#include <cstdio>
#include <cstring>
#include <utility>

using namespace d3d12test;
#define VERIFY(expr) do { if (!(expr)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #expr); return 1; } } while (0)

namespace {
void* leased_pages{};
unsigned released_leases{};
void* allocate_texture_pages(std::size_t bytes, std::uint64_t) {
    leased_pages = VirtualAlloc(nullptr, bytes, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    if (leased_pages) leonida::title_va_aliases().add_title_memory(reinterpret_cast<UINT64>(leased_pages), bytes);
    return leased_pages;
}
void release_texture_pages(void* p, std::uint64_t) {
    leonida::title_va_aliases().remove_title_memory(reinterpret_cast<UINT64>(p));
    VirtualFree(p, 0, MEM_RELEASE);
    ++released_leases;
}
int check_lease(ID3D12Device* dev, ID3D12Device* xbox) {
    D3D12_RESOURCE_DESC shape{};
    shape.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    shape.Width = shape.Height = 256;
    shape.DepthOrArraySize = shape.MipLevels = 1;
    shape.Format = DXGI_FORMAT_R32G32_FLOAT;
    shape.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET | D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    shape.SampleDesc.Count = 1;
    D3D12_HEAP_PROPERTIES heap{D3D12_HEAP_TYPE_DEFAULT};
    Ref<ID3D12Resource> pc;
    VERIFY(SUCCEEDED(dev->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &shape, D3D12_RESOURCE_STATE_COMMON,
                                                  nullptr, IID_PPV_ARGS(&pc.p))));
    VERIFY(pc->GetGPUVirtualAddress() == 0);
    rt::d3d12_xs::set_descriptor_heap_allocator(allocate_texture_pages, release_texture_pages);
    using Create = HRESULT(WINAPI*)(void*, const D3D12_HEAP_PROPERTIES*, UINT, const D3D12_RESOURCE_DESC*,
                                    UINT, const D3D12_CLEAR_VALUE*, const IID*, void**);
    const auto create = reinterpret_cast<Create>(slot(xbox, 76));
    {
        Ref<ID3D12Resource> console;
        VERIFY(SUCCEEDED(create(xbox, &heap, 0, &shape, 0, nullptr, &__uuidof(ID3D12Resource),
                                reinterpret_cast<void**>(&console.p))));
        VERIFY(leased_pages && console->GetGPUVirtualAddress() == reinterpret_cast<UINT64>(leased_pages));
        VERIFY(released_leases == 0);
        auto* words = static_cast<UINT*>(leased_pages);
        words[0] = 0x365f;
        words[256 * 256 * 2 - 1] = 0x9814;
        VERIFY(console->GetGPUVirtualAddress() == reinterpret_cast<UINT64>(leased_pages));
    }
    VERIFY(released_leases == 1);
    rt::d3d12_xs::set_descriptor_heap_allocator(nullptr, nullptr);
    return 0;
}

// With the console D16S8 family made as D24S8 (LEONIDA_CONSOLE_D16S8=d24; the default on a GPU with D24S8
// attachments, every NVIDIA card), format 118 keeps its UAVs through the widened R16_TYPELESS + R8_UINT storage
// planes and Depth24Converter. Check slot 133 (the retail 1280x720 placement), CreateCommittedResource3 with a
// castable list, R16_UNORM float + R16_UINT uint + R8_UINT stencil UAV clears, and direct readback of the parent
// D24S8 attachment image.
int check_d16s8_d24_uav(ID3D12Device* dev, ID3D12Device* xbox) {
    constexpr auto xbox_d16s8 = static_cast<DXGI_FORMAT>(118);
    constexpr auto xbox_d16_plane = static_cast<DXGI_FORMAT>(119);
    constexpr auto xbox_s8_plane = static_cast<DXGI_FORMAT>(120);
    D3D12_RESOURCE_DESC tex{};
    tex.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    tex.Width = 1280;
    tex.Height = 720;
    tex.DepthOrArraySize = tex.MipLevels = 1;
    tex.SampleDesc.Count = 1;
    tex.Format = xbox_d16s8;
    tex.Layout = static_cast<D3D12_TEXTURE_LAYOUT>(0x101);
    tex.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL | D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    D3D12_HEAP_PROPERTIES heap{D3D12_HEAP_TYPE_DEFAULT};
    D3D12_CLEAR_VALUE clear{};
    clear.Format = tex.Format;
    clear.DepthStencil.Depth = 1;

    // Slot 133 placement at 1280x720 with all-ones layout keeps ALLOW_UNORDERED_ACCESS and valid plane UAVs.
    using Place = HRESULT(WINAPI*)(void*, UINT64, const D3D12_RESOURCE_DESC*, UINT, const D3D12_CLEAR_VALUE*,
                                   const IID*, void**);
    constexpr SIZE_T pages = 16u << 20;
    void* va = VirtualAlloc(nullptr, pages, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    VERIFY(va);
    Ref<ID3D12Resource> placed;
    const HRESULT place_hr = reinterpret_cast<Place>(slot(xbox, 133))(xbox, reinterpret_cast<UINT64>(va), &tex,
                                                                      0xFFFFFFFFu, &clear, &__uuidof(ID3D12Resource),
                                                                      reinterpret_cast<void**>(&placed.p));
    const bool placed_ok = SUCCEEDED(place_hr) && placed.p && placed->GetDesc().Flags == tex.Flags &&
                           placed->GetDesc().Format == DXGI_FORMAT_D24_UNORM_S8_UINT;
    bool placed_views_ok = false;
    if (placed_ok) {
        Ref<ID3D12DescriptorHeap> heap_uav;
        D3D12_DESCRIPTOR_HEAP_DESC hd{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 3};
        if (SUCCEEDED(dev->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&heap_uav.p)))) {
            const SIZE_T step = dev->GetDescriptorHandleIncrementSize(hd.Type);
            const SIZE_T base = heap_uav->GetCPUDescriptorHandleForHeapStart().ptr;
            const std::pair<DXGI_FORMAT, UINT> plane_views[3] = {
                {DXGI_FORMAT_R16_UINT, 0}, {DXGI_FORMAT_R16_UNORM, 0}, {DXGI_FORMAT_R8_UINT, 1}};
            placed_views_ok = true;
            for (UINT i = 0; i < 3; ++i) {
                D3D12_UNORDERED_ACCESS_VIEW_DESC uv{};
                uv.Format = plane_views[i].first;
                uv.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
                uv.Texture2D.PlaneSlice = plane_views[i].second;
                dev->CreateUnorderedAccessView(placed.p, nullptr, &uv, {base + i * step});
                leonida::gfx10::ImageWords words{};
                std::memcpy(words.data(), reinterpret_cast<const void*>(base + i * step), sizeof(words));
                leonida::gfx10::Image image{};
                if (!leonida::gfx10::decode(words, image) || !rt::d3d12::uav_image_resource(placed.p, image.address))
                    placed_views_ok = false;
            }
        }
    }
    placed.reset();
    VirtualFree(va, 0, MEM_RELEASE);
    VERIFY(placed_ok && placed_views_ok);

    // CreateCommittedResource3 with a castable format list also keeps ALLOW_UNORDERED_ACCESS (direct and slot 127),
    // while plain PC D24_UNORM_S8_UINT refuses R16/R8 castable views and refuses ALLOW_UNORDERED_ACCESS.
    {
        Ref<ID3D12Device10> dev10;
        VERIFY(SUCCEEDED(dev->QueryInterface(IID_PPV_ARGS(&dev10.p))));
        D3D12_RESOURCE_DESC1 desc1{tex.Dimension, 0, 32, 32, 1, 1, xbox_d16s8, tex.SampleDesc,
                                   D3D12_TEXTURE_LAYOUT_UNKNOWN, tex.Flags, {}};
        const DXGI_FORMAT castable[] = {xbox_d16s8, xbox_d16_plane, xbox_s8_plane,
                                        DXGI_FORMAT_D24_UNORM_S8_UINT, DXGI_FORMAT_R24_UNORM_X8_TYPELESS,
                                        DXGI_FORMAT_X24_TYPELESS_G8_UINT, DXGI_FORMAT_R16_TYPELESS,
                                        DXGI_FORMAT_R16_UNORM, DXGI_FORMAT_R16_UINT, DXGI_FORMAT_R8_UINT};
        Ref<ID3D12Resource> c3;
        VERIFY(SUCCEEDED(dev10->CreateCommittedResource3(&heap, D3D12_HEAP_FLAG_NONE, &desc1,
                                                         D3D12_BARRIER_LAYOUT_COMMON, &clear, nullptr,
                                                         UINT(std::size(castable)), castable, IID_PPV_ARGS(&c3.p))));
        VERIFY(c3.p && c3->GetDesc().Flags == tex.Flags && c3->GetDesc().Format == DXGI_FORMAT_D24_UNORM_S8_UINT);
        const DXGI_FORMAT bad_view = DXGI_FORMAT_R32_FLOAT;
        Ref<ID3D12Resource> bad_c3;
        VERIFY(dev10->CreateCommittedResource3(&heap, D3D12_HEAP_FLAG_NONE, &desc1, D3D12_BARRIER_LAYOUT_COMMON,
                                               &clear, nullptr, 1, &bad_view, IID_PPV_ARGS(&bad_c3.p)) == E_INVALIDARG &&
               !bad_c3.p);

        // Xbox slot 127 forwards to CreateCommittedResource3 with the Xbox layout word replaced.
        using Slot127 = HRESULT(WINAPI*)(void*, const D3D12_HEAP_PROPERTIES*, D3D12_HEAP_FLAGS,
                                         const D3D12_RESOURCE_DESC1*, D3D12_BARRIER_LAYOUT, const D3D12_CLEAR_VALUE*,
                                         ID3D12ProtectedResourceSession*, UINT, const DXGI_FORMAT*, REFIID, void**);
        D3D12_RESOURCE_DESC1 xbox_desc1 = desc1;
        xbox_desc1.Layout = static_cast<D3D12_TEXTURE_LAYOUT>(0x101);
        Ref<ID3D12Resource> c3x;
        VERIFY(SUCCEEDED(reinterpret_cast<Slot127>(slot(xbox, 127))(
            xbox, &heap, D3D12_HEAP_FLAG_NONE, &xbox_desc1, D3D12_BARRIER_LAYOUT_COMMON, &clear, nullptr,
            UINT(std::size(castable)), castable, IID_PPV_ARGS(&c3x.p))));
        VERIFY(c3x.p && c3x->GetDesc().Flags == tex.Flags && c3x->GetDesc().Format == DXGI_FORMAT_D24_UNORM_S8_UINT);

        // Plain PC D24_UNORM_S8_UINT: refuses ALLOW_UNORDERED_ACCESS and refuses R16/R8 castable formats.
        D3D12_RESOURCE_DESC1 pc_d24 = desc1;
        pc_d24.Format = DXGI_FORMAT_D24_UNORM_S8_UINT;
        D3D12_CLEAR_VALUE pc_clear = clear;
        pc_clear.Format = DXGI_FORMAT_D24_UNORM_S8_UINT;
        Ref<ID3D12Resource> refused;
        VERIFY(dev10->CreateCommittedResource3(&heap, D3D12_HEAP_FLAG_NONE, &pc_d24, D3D12_BARRIER_LAYOUT_COMMON,
                                               &pc_clear, nullptr, 0, nullptr, IID_PPV_ARGS(&refused.p)) == E_INVALIDARG &&
               !refused.p);
        pc_d24.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
        for (const DXGI_FORMAT narrow : {DXGI_FORMAT_R16_TYPELESS, DXGI_FORMAT_R16_UNORM, DXGI_FORMAT_R16_UINT,
                                         DXGI_FORMAT_R8_UINT}) {
            VERIFY(dev10->CreateCommittedResource3(&heap, D3D12_HEAP_FLAG_NONE, &pc_d24, D3D12_BARRIER_LAYOUT_COMMON,
                                                   &pc_clear, nullptr, 1, &narrow, IID_PPV_ARGS(&refused.p)) ==
                       E_INVALIDARG &&
                   !refused.p);
        }

        // CreateCommittedResource2, CreatePlacedResource/2 (in a heap), and CreateReservedResource with format 118.
        Ref<ID3D12Device8> dev8;
        VERIFY(SUCCEEDED(dev->QueryInterface(IID_PPV_ARGS(&dev8.p))));
        D3D12_RESOURCE_DESC1 d16_no_uav = desc1;
        d16_no_uav.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
        Ref<ID3D12Resource> c2;
        VERIFY(SUCCEEDED(dev8->CreateCommittedResource2(&heap, D3D12_HEAP_FLAG_NONE, &d16_no_uav,
                                                        D3D12_RESOURCE_STATE_DEPTH_WRITE, &clear, nullptr,
                                                        IID_PPV_ARGS(&c2.p))));
        VERIFY(c2.p && c2->GetDesc().Format == DXGI_FORMAT_D24_UNORM_S8_UINT);
        VERIFY(dev8->CreateCommittedResource2(&heap, D3D12_HEAP_FLAG_NONE, &desc1, D3D12_RESOURCE_STATE_DEPTH_WRITE,
                                              &clear, nullptr, IID_PPV_ARGS(&refused.p)) == E_INVALIDARG &&
               !refused.p);

        D3D12_RESOURCE_DESC placed_desc{};
        placed_desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        placed_desc.Width = placed_desc.Height = 32;
        placed_desc.DepthOrArraySize = placed_desc.MipLevels = 1;
        placed_desc.Format = xbox_d16s8;
        placed_desc.SampleDesc.Count = 1;
        placed_desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
        const auto placed_alloc = dev->GetResourceAllocationInfo(0, 1, &placed_desc);
        VERIFY(placed_alloc.SizeInBytes > 0 && placed_alloc.SizeInBytes != UINT64_MAX);
        D3D12_HEAP_DESC heap_desc{placed_alloc.SizeInBytes, heap, placed_alloc.Alignment, D3D12_HEAP_FLAG_NONE};
        Ref<ID3D12Heap> placed_heap;
        VERIFY(SUCCEEDED(dev->CreateHeap(&heap_desc, IID_PPV_ARGS(&placed_heap.p))));
        Ref<ID3D12Resource> heap_placed;
        VERIFY(SUCCEEDED(dev->CreatePlacedResource(placed_heap.p, 0, &placed_desc, D3D12_RESOURCE_STATE_DEPTH_WRITE,
                                                   &clear, IID_PPV_ARGS(&heap_placed.p))));
        VERIFY(heap_placed.p && heap_placed->GetDesc().Format == DXGI_FORMAT_D24_UNORM_S8_UINT);
        D3D12_RESOURCE_DESC placed_uav = placed_desc;
        placed_uav.Flags |= D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        VERIFY(dev->CreatePlacedResource(placed_heap.p, 0, &placed_uav, D3D12_RESOURCE_STATE_DEPTH_WRITE, &clear,
                                         IID_PPV_ARGS(&refused.p)) == E_INVALIDARG &&
               !refused.p);
        Ref<ID3D12Resource> heap_placed2;
        VERIFY(SUCCEEDED(dev10->CreatePlacedResource2(placed_heap.p, 0, &d16_no_uav, D3D12_BARRIER_LAYOUT_COMMON,
                                                      &clear, 0, nullptr, IID_PPV_ARGS(&heap_placed2.p))));
        VERIFY(heap_placed2.p && heap_placed2->GetDesc().Format == DXGI_FORMAT_D24_UNORM_S8_UINT);
        for (const DXGI_FORMAT narrow : {DXGI_FORMAT_R16_UNORM, DXGI_FORMAT_R8_UINT}) {
            VERIFY(dev10->CreatePlacedResource2(placed_heap.p, 0, &pc_d24, D3D12_BARRIER_LAYOUT_COMMON, &pc_clear,
                                                1, &narrow, IID_PPV_ARGS(&refused.p)) == E_INVALIDARG &&
                   !refused.p);
        }

        // Reserved resources reject stencil formats (including format 118) because they have no tile shape.
        D3D12_RESOURCE_DESC reserved_desc = placed_desc;
        reserved_desc.Layout = D3D12_TEXTURE_LAYOUT_64KB_UNDEFINED_SWIZZLE;
        VERIFY(dev->CreateReservedResource(&reserved_desc, D3D12_RESOURCE_STATE_DEPTH_WRITE, &clear,
                                           IID_PPV_ARGS(&refused.p)) == E_INVALIDARG &&
               !refused.p);
    }

    // Direct parent D24S8 attachment readback after R16_UNORM float + R16_UINT uint + R8_UINT stencil UAV clears.
    constexpr UINT w = 32, layers = 2;
    tex.Width = tex.Height = w;
    tex.DepthOrArraySize = layers;
    using Create = HRESULT(WINAPI*)(void*, const D3D12_HEAP_PROPERTIES*, UINT, const D3D12_RESOURCE_DESC*, UINT,
                                    const D3D12_CLEAR_VALUE*, const IID*, void**);
    Ref<ID3D12Resource> depth;
    VERIFY(SUCCEEDED(reinterpret_cast<Create>(slot(xbox, 76))(xbox, &heap, 0, &tex, D3D12_RESOURCE_STATE_DEPTH_WRITE,
                                                               &clear, &__uuidof(ID3D12Resource),
                                                               reinterpret_cast<void**>(&depth.p))));
    const D3D12_RESOURCE_DESC parent_desc = depth->GetDesc();
    VERIFY(parent_desc.Flags == tex.Flags && parent_desc.Format == DXGI_FORMAT_D24_UNORM_S8_UINT);

    Ref<ID3D12DescriptorHeap> dsv, uav;
    D3D12_DESCRIPTOR_HEAP_DESC dsv_hd{D3D12_DESCRIPTOR_HEAP_TYPE_DSV, layers};
    D3D12_DESCRIPTOR_HEAP_DESC uav_hd{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 3};
    VERIFY(SUCCEEDED(dev->CreateDescriptorHeap(&dsv_hd, IID_PPV_ARGS(&dsv.p))));
    VERIFY(SUCCEEDED(dev->CreateDescriptorHeap(&uav_hd, IID_PPV_ARGS(&uav.p))));
    const auto hslot = [&](ID3D12DescriptorHeap* h, D3D12_DESCRIPTOR_HEAP_TYPE t, UINT i) {
        auto handle = h->GetCPUDescriptorHandleForHeapStart();
        handle.ptr += i * dev->GetDescriptorHandleIncrementSize(t);
        return handle;
    };
    const UINT16 seed_d[layers] = {0x2000, 0xc000};
    const UINT8 seed_s[layers] = {0x41, 0x42};
    for (UINT i = 0; i < layers; ++i) {
        D3D12_DEPTH_STENCIL_VIEW_DESC dv{};
        dv.Format = xbox_d16s8;
        dv.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2DARRAY;
        dv.Texture2DArray = {0, i, 1};
        dev->CreateDepthStencilView(depth.p, &dv, hslot(dsv.p, dsv_hd.Type, i));
    }
    const std::pair<DXGI_FORMAT, UINT> uv_cfg[3] = {
        {DXGI_FORMAT_R16_UNORM, 0}, {DXGI_FORMAT_R16_UINT, 0}, {DXGI_FORMAT_R8_UINT, 1}};
    for (UINT i = 0; i < 3; ++i) {
        D3D12_UNORDERED_ACCESS_VIEW_DESC uv{};
        uv.Format = uv_cfg[i].first;
        uv.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2DARRAY;
        uv.Texture2DArray = {0, 0, 1, uv_cfg[i].second};
        dev->CreateUnorderedAccessView(depth.p, nullptr, &uv, hslot(uav.p, uav_hd.Type, i));
    }
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp[2 * layers]{};
    UINT64 total = 0;
    dev->GetCopyableFootprints(&parent_desc, 0, 2 * layers, 0, fp, nullptr, nullptr, &total);
    auto rb_desc = buffer_desc(total);
    D3D12_HEAP_PROPERTIES host{D3D12_HEAP_TYPE_READBACK};
    Ref<ID3D12Resource> readback;
    VERIFY(SUCCEEDED(dev->CreateCommittedResource(&host, D3D12_HEAP_FLAG_NONE, &rb_desc,
                                                  D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&readback.p))));
    Ref<ID3D12CommandQueue> queue;
    Ref<ID3D12CommandAllocator> alloc;
    Ref<ID3D12GraphicsCommandList> list;
    Ref<ID3D12Fence> fence;
    D3D12_COMMAND_QUEUE_DESC qd{};
    VERIFY(SUCCEEDED(dev->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue.p))));
    VERIFY(SUCCEEDED(dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&alloc.p))));
    VERIFY(SUCCEEDED(dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc.p, nullptr, IID_PPV_ARGS(&list.p))));
    VERIFY(SUCCEEDED(dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))));

    for (UINT i = 0; i < layers; ++i)
        list->ClearDepthStencilView(hslot(dsv.p, dsv_hd.Type, i),
                                    D3D12_CLEAR_FLAG_DEPTH | D3D12_CLEAR_FLAG_STENCIL,
                                    float(seed_d[i]) / 65535.f, seed_s[i], 0, nullptr);
    auto to_uav = transition(depth.p, D3D12_RESOURCE_STATE_DEPTH_WRITE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    list->ResourceBarrier(1, &to_uav);
    const D3D12_RECT r_unorm{2, 3, 10, 11}, r_uint{14, 5, 22, 13}, r_stencil{4, 4, 12, 12};
    const FLOAT fval[4]{0.25f};
    const UINT uval[4]{0x9123u}, sval[4]{0xa7u};
    list->ClearUnorderedAccessViewFloat({}, hslot(uav.p, uav_hd.Type, 0), depth.p, fval, 1, &r_unorm);
    D3D12_RESOURCE_BARRIER uav_bar{};
    uav_bar.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    uav_bar.UAV.pResource = depth.p;
    list->ResourceBarrier(1, &uav_bar);
    list->ClearUnorderedAccessViewUint({}, hslot(uav.p, uav_hd.Type, 1), depth.p, uval, 1, &r_uint);
    list->ClearUnorderedAccessViewUint({}, hslot(uav.p, uav_hd.Type, 2), depth.p, sval, 1, &r_stencil);
    auto to_copy = transition(depth.p, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
    list->ResourceBarrier(1, &to_copy);
    for (UINT sub = 0; sub < 2 * layers; ++sub) {
        D3D12_TEXTURE_COPY_LOCATION src{}, dst{};
        src.pResource = depth.p;
        src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        src.SubresourceIndex = sub;
        dst.pResource = readback.p;
        dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        dst.PlacedFootprint = fp[sub];
        list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    }
    VERIFY(SUCCEEDED(list->Close()));
    ID3D12CommandList* batch[]{list.p};
    queue->ExecuteCommandLists(1, batch);
    VERIFY(SUCCEEDED(queue->Signal(fence.p, 1)));
    const auto end = GetTickCount64() + 5000;
    while (fence->GetCompletedValue() < 1 && GetTickCount64() < end) Sleep(1);
    VERIFY(fence->GetCompletedValue() == 1);

    void* mapped{};
    VERIFY(SUCCEEDED(readback->Map(0, nullptr, &mapped)));
    const auto* bytes = static_cast<const std::uint8_t*>(mapped);
    for (UINT i = 0; i < layers; ++i) {
        for (int y = 0; y < int(w); ++y) {
            for (int x = 0; x < int(w); ++x) {
                UINT16 exp_u16 = seed_d[i];
                if (i == 0 && x >= r_unorm.left && x < r_unorm.right && y >= r_unorm.top && y < r_unorm.bottom)
                    exp_u16 = 0x4000u;
                if (i == 0 && x >= r_uint.left && x < r_uint.right && y >= r_uint.top && y < r_uint.bottom)
                    exp_u16 = 0x9123u;
                const std::uint32_t exp_d24 = leonida::depth24_quantize(float(exp_u16) / 65535.f);
                std::uint32_t got_word = 0;
                std::memcpy(&got_word, bytes + fp[i].Offset + y * fp[i].Footprint.RowPitch + x * 4, 4);
                const std::uint32_t got_d24 = got_word & 0xffffffu;
                if (got_d24 != exp_d24) {
                    std::fprintf(stderr, "D24 parent slice %u (%d,%d): expected %06x got %06x\n",
                                 i, x, y, exp_d24, got_d24);
                    readback->Unmap(0, nullptr);
                    return 1;
                }
                const UINT8 exp_s = (i == 0 && x >= r_stencil.left && x < r_stencil.right &&
                                     y >= r_stencil.top && y < r_stencil.bottom) ? 0xa7u : seed_s[i];
                const UINT8 got_s = bytes[fp[layers + i].Offset + y * fp[layers + i].Footprint.RowPitch + x];
                if (got_s != exp_s) {
                    std::fprintf(stderr, "D24 parent stencil slice %u (%d,%d): expected %02x got %02x\n",
                                 i, x, y, exp_s, got_s);
                    readback->Unmap(0, nullptr);
                    return 1;
                }
            }
        }
    }
    readback->Unmap(0, nullptr);
    std::printf("Xbox D16S8 made as D24S8: slot 133, CreateCommittedResource3, R16_UNORM/R16_UINT/R8_UINT UAVs "
                "and D24 parent readback exact\n");
    return 0;
}
}

int main() {
    vktest::Device gpu;
    const auto status = gpu.initialize();
    if (status == vktest::DeviceStatus::unavailable) return 77;
    VERIFY(status == vktest::DeviceStatus::ready);
    const bool with_stencil = std::getenv("LEONIDA_TEST_D16S8_UAV") != nullptr;
    constexpr auto xbox_d16s8 = static_cast<DXGI_FORMAT>(118);
    rt::d3d12::VulkanDeviceInfo info{};
    info.bindless = gpu.bindless;
    info.bindless_partially_bound = gpu.partially_bound;
    info.get_instance_proc = gpu.instance_proc;
    info.instance = gpu.instance;
    info.physical = gpu.physical;
    info.device = gpu.device;
    info.queue = gpu.queue;
    info.queue_family = gpu.family;
    info.buffer_device_address = true;
    ID3D12Device* dev{};
    VERIFY(SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&dev))));
    constexpr UINT layers = 6, width = 32;
    std::array<UINT16, layers> original{0x1095, 0x568c, 0x242a, 0x93db, 0xb832, 0x0e69};
    D3D12_RESOURCE_DESC tex{};
    tex.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    tex.Width = tex.Height = 768;
    tex.DepthOrArraySize = layers;
    tex.MipLevels = 1;
    tex.SampleDesc.Count = 1;
    tex.Format = with_stencil ? xbox_d16s8 : DXGI_FORMAT_R16_TYPELESS;
    tex.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL | D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    D3D12_HEAP_PROPERTIES heap{D3D12_HEAP_TYPE_DEFAULT};
    D3D12_CLEAR_VALUE clear{};
    clear.Format = with_stencil ? xbox_d16s8 : DXGI_FORMAT_D16_UNORM;
    clear.DepthStencil.Depth = 1;
    Ref<ID3D12Resource> depth;
    VERIFY(dev->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &tex, D3D12_RESOURCE_STATE_DEPTH_WRITE,
                                        &clear, IID_PPV_ARGS(&depth.p)) == E_INVALIDARG && !depth.p);
    Ref<ID3D12Device> proxy;
    VERIFY(SUCCEEDED(rt::d3d12_xs::create_device_proxy(dev, __uuidof(ID3D12Device),
                                                       reinterpret_cast<void**>(&proxy.p), true)));
    rt::d3d12::set_device_outer(dev, proxy.p);
    VERIFY(check_lease(dev, proxy.p) == 0);
    const auto allocation = dev->GetResourceAllocationInfo(0, 1, &tex);
    VERIFY(allocation.SizeInBytes >= 768u * 768 * layers * 2 && allocation.SizeInBytes != UINT64_MAX);
    using XboxCreate = HRESULT(WINAPI*)(void*, const D3D12_HEAP_PROPERTIES*, UINT, const D3D12_RESOURCE_DESC*,
                                          UINT, const D3D12_CLEAR_VALUE*, const IID*, void**);
    const auto create = reinterpret_cast<XboxCreate>(slot(proxy.p, 76));
    tex.Width = tex.Height = width;
    if (with_stencil) {
        VERIFY(SUCCEEDED(create(proxy.p, &heap, 0, &tex, D3D12_RESOURCE_STATE_DEPTH_WRITE, &clear,
                                &__uuidof(ID3D12Resource), reinterpret_cast<void**>(&depth.p))));
    } else {
        VERIFY(SUCCEEDED(dev->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &tex, D3D12_RESOURCE_STATE_DEPTH_WRITE,
                                                     &clear, __uuidof(ID3D12Resource), nullptr)));
        VERIFY(SUCCEEDED(dev->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &tex, D3D12_RESOURCE_STATE_DEPTH_WRITE,
                                                     &clear, IID_PPV_ARGS(&depth.p))));
    }
    VERIFY(depth->GetDesc().Flags == tex.Flags);
    Ref<ID3D12DescriptorHeap> dsv, uav;
    D3D12_DESCRIPTOR_HEAP_DESC hd{D3D12_DESCRIPTOR_HEAP_TYPE_DSV, layers};
    VERIFY(SUCCEEDED(dev->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&dsv.p))));
    hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    hd.NumDescriptors = layers + 1;
    VERIFY(SUCCEEDED(dev->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&uav.p))));
    const auto slot = [&](ID3D12DescriptorHeap* h, D3D12_DESCRIPTOR_HEAP_TYPE t, UINT i) {
        auto handle = h->GetCPUDescriptorHandleForHeapStart();
        handle.ptr += i * dev->GetDescriptorHandleIncrementSize(t);
        return handle;
    };
    Ref<ID3D12CommandQueue> queue;
    Ref<ID3D12CommandAllocator> allocator;
    Ref<ID3D12GraphicsCommandList7> list;
    Ref<ID3D12Fence> fence;
    D3D12_COMMAND_QUEUE_DESC queue_desc{};
    VERIFY(SUCCEEDED(dev->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue.p))));
    VERIFY(SUCCEEDED(dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p))));
    VERIFY(SUCCEEDED(dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, nullptr, IID_PPV_ARGS(&list.p))));
    VERIFY(SUCCEEDED(dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))));
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprints[2 * layers]{};
    UINT64 total{};
    dev->GetCopyableFootprints(&tex, 0, with_stencil ? 2 * layers : layers, 0, footprints, nullptr, nullptr, &total);
    auto buffer = buffer_desc(total);
    D3D12_HEAP_PROPERTIES host{D3D12_HEAP_TYPE_READBACK};
    Ref<ID3D12Resource> readback;
    VERIFY(SUCCEEDED(dev->CreateCommittedResource(&host, D3D12_HEAP_FLAG_NONE, &buffer,
                                                  D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&readback.p))));
    for (UINT i = 0; i < layers; ++i) {
        const auto d = slot(dsv.p, D3D12_DESCRIPTOR_HEAP_TYPE_DSV, i);
        D3D12_DEPTH_STENCIL_VIEW_DESC dv{};
        dv.Format = with_stencil ? xbox_d16s8 : DXGI_FORMAT_D16_UNORM;
        dv.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2DARRAY;
        dv.Texture2DArray.FirstArraySlice = i;
        dv.Texture2DArray.ArraySize = 1;
        dev->CreateDepthStencilView(depth.p, &dv, d);
        const auto flags = with_stencil ? D3D12_CLEAR_FLAG_DEPTH | D3D12_CLEAR_FLAG_STENCIL : D3D12_CLEAR_FLAG_DEPTH;
        list->ClearDepthStencilView(d, flags, float(original[i]) / 65535.f, UINT8(0x30 + i), 0, nullptr);
        D3D12_UNORDERED_ACCESS_VIEW_DESC uv{};
        uv.Format = DXGI_FORMAT_R16_UINT;
        uv.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2DARRAY;
        uv.Texture2DArray.FirstArraySlice = i;
        uv.Texture2DArray.ArraySize = 1;
        dev->CreateUnorderedAccessView(depth.p, nullptr, &uv, slot(uav.p, hd.Type, i));
    }
    // Mix legacy and enhanced transitions. Leave slice five entirely on the depth image.
    D3D12_RECT rect{3, 4, 19, 26};
    for (UINT i = 0; i < layers - 1; ++i) {
        D3D12_TEXTURE_BARRIER enhanced{};
        enhanced.SyncBefore = D3D12_BARRIER_SYNC_DEPTH_STENCIL;
        enhanced.SyncAfter = D3D12_BARRIER_SYNC_CLEAR_UNORDERED_ACCESS_VIEW;
        enhanced.AccessBefore = D3D12_BARRIER_ACCESS_DEPTH_STENCIL_WRITE;
        enhanced.AccessAfter = D3D12_BARRIER_ACCESS_UNORDERED_ACCESS;
        enhanced.LayoutBefore = D3D12_BARRIER_LAYOUT_DEPTH_STENCIL_WRITE;
        enhanced.LayoutAfter = D3D12_BARRIER_LAYOUT_UNORDERED_ACCESS;
        enhanced.pResource = depth.p;
        enhanced.Subresources = {0, 1, i, 1, 0, 1};
        D3D12_BARRIER_GROUP group{D3D12_BARRIER_TYPE_TEXTURE, 1};
        group.pTextureBarriers = &enhanced;
        auto legacy = transition(depth.p, D3D12_RESOURCE_STATE_DEPTH_WRITE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        legacy.Transition.Subresource = i;
        if (i % 2) list->Barrier(1, &group); else list->ResourceBarrier(1, &legacy);
        const UINT values[4]{0x8040u + i};
        list->ClearUnorderedAccessViewUint({}, slot(uav.p, hd.Type, i), depth.p, values, 1, &rect);
        std::swap(legacy.Transition.StateBefore, legacy.Transition.StateAfter);
        std::swap(enhanced.SyncBefore, enhanced.SyncAfter);
        std::swap(enhanced.AccessBefore, enhanced.AccessAfter);
        std::swap(enhanced.LayoutBefore, enhanced.LayoutAfter);
        if (i % 2) list->Barrier(1, &group); else list->ResourceBarrier(1, &legacy);
    }
    ID3D12Resource *raw_depth = depth.p, *raw_stencil = nullptr;
    if (with_stencil) {
        D3D12_UNORDERED_ACCESS_VIEW_DESC st{};
        st.Format = DXGI_FORMAT_R8_UINT;
        st.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2DARRAY;
        st.Texture2DArray.ArraySize = layers;
        st.Texture2DArray.PlaneSlice = 1;
        dev->CreateUnorderedAccessView(depth.p, nullptr, &st, slot(uav.p, hd.Type, layers));
        const auto inspected_plane = [&](UINT index) -> ID3D12Resource* {
            leonida::gfx10::ImageWords words{};
            std::memcpy(words.data(), reinterpret_cast<const void*>(slot(uav.p,hd.Type,index).ptr), sizeof(words));
            leonida::gfx10::Image image{};
            if (!leonida::gfx10::decode(words,image)) return nullptr;
            return reinterpret_cast<ID3D12Resource*>(rt::d3d12::uav_image_resource(depth.p,image.address));
        };
        raw_depth = inspected_plane(0);
        raw_stencil = inspected_plane(layers);
        VERIFY(raw_depth && raw_stencil && raw_depth != depth.p && raw_stencil != raw_depth);
    }
    auto ready = transition(depth.p, D3D12_RESOURCE_STATE_DEPTH_WRITE,
                             with_stencil ? D3D12_RESOURCE_STATE_UNORDERED_ACCESS : D3D12_RESOURCE_STATE_COPY_SOURCE);
    list->ResourceBarrier(1, &ready);
    for (UINT i = 0; i < (with_stencil ? 2 * layers : layers); ++i) {
        D3D12_TEXTURE_COPY_LOCATION from{}, to{};
        from.pResource = i < layers ? raw_depth : raw_stencil; from.SubresourceIndex = i % layers;
        to.pResource = readback.p; to.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT; to.PlacedFootprint = footprints[i];
        list->CopyTextureRegion(&to, 0, 0, 0, &from, nullptr);
    }
    VERIFY(SUCCEEDED(list->Close()));
    ID3D12CommandList* batch[]{list.p};
    queue->ExecuteCommandLists(1, batch);
    VERIFY(SUCCEEDED(queue->Signal(fence.p, 1)));
    const auto end = GetTickCount64() + 5000;
    while (fence->GetCompletedValue() < 1 && GetTickCount64() < end) Sleep(1);
    VERIFY(fence->GetCompletedValue() == 1);
    void* mapped{};
    VERIFY(SUCCEEDED(readback->Map(0, nullptr, &mapped)));
    auto* bytes = static_cast<const std::uint8_t*>(mapped);
    for (UINT i = 0; i < layers; ++i) for (int y = 0; y < width; ++y) for (int x = 0; x < width; ++x) {
        const auto offset = footprints[i].Offset + y * footprints[i].Footprint.RowPitch + x * 2;
        const auto value = *reinterpret_cast<const UINT16*>(bytes + offset);
        const bool changed = i < layers-1 && x >= rect.left && x < rect.right && y >= rect.top && y < rect.bottom;
        const UINT expected = changed ? 0x8040u+i : original[i];
        if (value != expected) {
            std::fprintf(stderr, "slice %u pixel (%d,%d) expected %04x got %04x\n", i, x, y, expected, value);
            return 1;
        }
    }
    if (with_stencil) {
        for (UINT i=0; i<layers; ++i) for (UINT y=0; y<width; ++y) for (UINT x=0; x<width; ++x)
            VERIFY(bytes[footprints[layers+i].Offset + y*footprints[layers+i].Footprint.RowPitch + x] == 0x30+i);
        std::printf("Xbox D16S8 depth and stencil: %u depth words and %u stencil bytes exact after UAV round trip\n", width*width*layers,width*width*layers);
    }
    readback->Unmap(0, nullptr);
    const char* d16s8_mode = std::getenv("LEONIDA_CONSOLE_D16S8");
    if (with_stencil && d16s8_mode && std::strcmp(d16s8_mode, "d24") == 0)
        VERIFY(check_d16s8_d24_uav(dev, proxy.p) == 0);
    VERIFY(gpu.validation_errors() == 0);
    std::printf("Depth D16 / storage R16_UINT, six layers and %u exact texels\n", width * width * layers);
    return 0;
}
