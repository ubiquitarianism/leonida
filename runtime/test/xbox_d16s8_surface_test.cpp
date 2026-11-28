#include "../shims/d3d12_xs_device.h"
#include "vk_test_device.h"
#include "d3d12_test_shaders.h"
#include "../shims/d3d12_com.h"
#include <cstdio>
#define CHECK(v)                                                                                                       \
    do {                                                                                                               \
        if (!(v)) {                                                                                                    \
            std::fprintf(stderr, "line %d: %s\n", __LINE__, #v);                                                       \
            return 1;                                                                                                  \
        }                                                                                                              \
    } while (0)
using namespace d3d12test;
int main() {
    vktest::Device gpu;
    const auto status = gpu.initialize();
    if (status == vktest::DeviceStatus::unavailable) return 77;
    CHECK(status == vktest::DeviceStatus::ready);
    rt::d3d12::VulkanDeviceInfo info{};
    info.get_instance_proc = gpu.instance_proc;
    info.instance = gpu.instance;
    info.physical = gpu.physical;
    info.device = gpu.device;
    info.queue = gpu.queue;
    info.queue_family = gpu.family;
    info.buffer_device_address = true;
    Ref<ID3D12Device> dev;
    CHECK(SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&dev.p))));
    constexpr DXGI_FORMAT format = static_cast<DXGI_FORMAT>(118);
    constexpr UINT extent = 32;
    D3D12_RESOURCE_DESC tex{};
    tex.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    tex.Width = extent;
    tex.Height = extent;
    tex.DepthOrArraySize = 1;
    tex.MipLevels = 1;
    tex.Format = format;
    tex.SampleDesc.Count = 1;
    tex.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
    const auto required = dev->GetResourceAllocationInfo(0, 1, &tex);
    CHECK(required.SizeInBytes > 0 && required.SizeInBytes != ~UINT64{0} && required.Alignment);
    D3D12_FEATURE_DATA_FORMAT_INFO planes{format, 255};
    CHECK(dev->CheckFeatureSupport(D3D12_FEATURE_FORMAT_INFO, &planes, sizeof(planes)) == E_INVALIDARG);
    CHECK(planes.PlaneCount == 255);
    {
        Ref<ID3D12Device> inner, console;
        CHECK(SUCCEEDED(dev->QueryInterface(IID_PPV_ARGS(&inner.p))));
        CHECK(SUCCEEDED(rt::d3d12_xs::create_device_proxy(inner.p, __uuidof(ID3D12Device),
                                                          reinterpret_cast<void**>(&console.p), true)));
        inner.p = nullptr;  // the proxy now owns this reference
        rt::d3d12_xs::set_console_caller([](std::uint64_t) { return true; });
        for (unsigned alias = 118; alias <= 120; ++alias) {
            planes = {static_cast<DXGI_FORMAT>(alias), 255};
            CHECK(SUCCEEDED(console->CheckFeatureSupport(D3D12_FEATURE_FORMAT_INFO, &planes, sizeof(planes))));
            CHECK(planes.PlaneCount == 2);
        }
        rt::d3d12_xs::set_console_caller(nullptr);
    }
    D3D12_HEAP_PROPERTIES def{D3D12_HEAP_TYPE_DEFAULT}, rd{D3D12_HEAP_TYPE_READBACK};
    D3D12_CLEAR_VALUE clear{};
    clear.Format = format;
    clear.DepthStencil = {0.25f, 0x3c};
    Ref<ID3D12Resource> texture, readback;
    CHECK(SUCCEEDED(dev->CreateCommittedResource(&def, D3D12_HEAP_FLAG_NONE, &tex, D3D12_RESOURCE_STATE_DEPTH_WRITE,
                                                 &clear, IID_PPV_ARGS(&texture.p))));
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp[2]{};
    UINT rows[2]{};
    UINT64 bytes[2]{}, total = 0;
    dev->GetCopyableFootprints(&tex, 0, 2, 0, fp, rows, bytes, &total);
    CHECK(bytes[0] == extent * 2 && bytes[1] == extent && rows[0] == extent && rows[1] == extent && total > 0);
    CHECK(fp[0].Footprint.Format == DXGI_FORMAT_R16_TYPELESS && fp[1].Footprint.Format == DXGI_FORMAT_R8_TYPELESS);
    auto b = buffer_desc(total);
    CHECK(SUCCEEDED(dev->CreateCommittedResource(&rd, D3D12_HEAP_FLAG_NONE, &b, D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                 IID_PPV_ARGS(&readback.p))));
    Ref<ID3D12DescriptorHeap> dsv;
    D3D12_DESCRIPTOR_HEAP_DESC hd{D3D12_DESCRIPTOR_HEAP_TYPE_DSV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
    CHECK(SUCCEEDED(dev->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&dsv.p))));
    const auto handle = dsv->GetCPUDescriptorHandleForHeapStart();
    D3D12_DEPTH_STENCIL_VIEW_DESC view{};
    view.Format = format;
    view.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
    dev->CreateDepthStencilView(texture.p, &view, handle);
    Ref<ID3D12CommandQueue> queue;
    Ref<ID3D12CommandAllocator> alloc;
    Ref<ID3D12GraphicsCommandList> list;
    Ref<ID3D12Fence> fence;
    D3D12_COMMAND_QUEUE_DESC q{};
    CHECK(SUCCEEDED(dev->CreateCommandQueue(&q, IID_PPV_ARGS(&queue.p))));
    CHECK(SUCCEEDED(dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&alloc.p))));
    CHECK(
        SUCCEEDED(dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc.p, nullptr, IID_PPV_ARGS(&list.p))));
    CHECK(SUCCEEDED(dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))));
    const auto both = D3D12_CLEAR_FLAG_DEPTH | D3D12_CLEAR_FLAG_STENCIL;
    list->ClearDepthStencilView(handle, both, .25f, 0x3c, 0, nullptr);
    D3D12_RECT rect{8, 8, 24, 24};
    list->ClearDepthStencilView(handle, both, .75f, 0xa5, 1, &rect);
    auto bar = transition(texture.p, D3D12_RESOURCE_STATE_DEPTH_WRITE, D3D12_RESOURCE_STATE_COPY_SOURCE);
    list->ResourceBarrier(1, &bar);
    D3D12_TEXTURE_COPY_LOCATION from{};
    from.pResource = texture.p;
    from.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    from.SubresourceIndex = 1;
    D3D12_TEXTURE_COPY_LOCATION to{};
    to.pResource = readback.p;
    to.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    to.PlacedFootprint = fp[1];
    list->CopyTextureRegion(&to, 0, 0, 0, &from, nullptr);
    CHECK(SUCCEEDED(list->Close()));
    ID3D12CommandList *batch[] = {list.p};
    queue->ExecuteCommandLists(1, batch);
    CHECK(SUCCEEDED(queue->Signal(fence.p, 1)));
    const auto end = GetTickCount64() + 3000;
    while (fence->GetCompletedValue() < 1 && GetTickCount64() < end)
        Sleep(1);
    CHECK(fence->GetCompletedValue() >= 1);
    void *mapped = nullptr;
    CHECK(SUCCEEDED(readback->Map(0, nullptr, &mapped)));
    const auto *pixels = static_cast<const unsigned char *>(mapped) + fp[1].Offset;
    for (unsigned y = 0; y < extent; ++y)
        for (unsigned x = 0; x < extent; ++x)
            CHECK(pixels[y * fp[1].Footprint.RowPitch + x] == (x >= 8 && x < 24 && y >= 8 && y < 24 ? 0xa5 : 0x3c));
    readback->Unmap(0, nullptr);
    CHECK(!gpu.validation_errors());
    std::printf("Xbox D16+S8 allocation %llu; both plane footprints and all %u stencil pixels passed\n",
                (unsigned long long)required.SizeInBytes, extent * extent);
    return 0;
}
