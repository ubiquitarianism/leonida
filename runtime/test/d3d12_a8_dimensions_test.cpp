// A8_UNORM textures in every dimension: 1D, 2D and 3D, compared with Windows D3D12. The host adapter and WARP both
// report TEXTURE1D, TEXTURE2D and TEXTURE3D support for A8_UNORM, size those textures and create them; the layer must
// too (it holds them as R8 images read through the alpha swizzle, the same for every dimension). A byte pattern copied
// into each texture and back out must come back unchanged. Render targets, depth and UAVs stay refused.
// No game data or shaders. Run explicitly (needs Vulkan hardware); LEONIDA_VK_VALIDATION=1 checks Vulkan usage.
#include "vk_test_device.h"
#include "host_adapter.h"
#include "host_wine.h"
#include "../shims/d3d12_com.h"
#include "../src/gfx10_descriptors.h"
#include <dxgi1_4.h>
#include <cstdint>
#include <cstring>
#include <string>
#define CHECK(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return false; } \
} while (0)

namespace {
template<class T>
struct Ref {
    T* p = nullptr;
    ~Ref() { if (p) p->Release(); }
    T* operator->() const { return p; }
};
struct Event {
    HANDLE handle = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    ~Event() { if (handle) CloseHandle(handle); }
};

struct Shape {
    const char* name;
    D3D12_RESOURCE_DIMENSION dimension;
    UINT64 width;
    UINT height;
    UINT16 depth;
    UINT16 mips;
};

// The 1D shape is the one the retail title sizes through device slot 25 (128 texels, one mip).
constexpr Shape kShapes[] = {
    {"1D 128", D3D12_RESOURCE_DIMENSION_TEXTURE1D, 128, 1, 1, 1},
    {"1D 300, 3 mips", D3D12_RESOURCE_DIMENSION_TEXTURE1D, 300, 1, 1, 3},
    {"1D array 64 x4", D3D12_RESOURCE_DIMENSION_TEXTURE1D, 64, 1, 4, 1},
    {"2D 13x7", D3D12_RESOURCE_DIMENSION_TEXTURE2D, 13, 7, 1, 1},
    {"3D 16x8x4, 2 mips", D3D12_RESOURCE_DIMENSION_TEXTURE3D, 16, 8, 4, 2},
    {"3D 5x3x7", D3D12_RESOURCE_DIMENSION_TEXTURE3D, 5, 3, 7, 1},
};

D3D12_RESOURCE_DESC describe(const Shape& shape, UINT64 alignment, D3D12_RESOURCE_FLAGS flags) {
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = shape.dimension;
    desc.Alignment = alignment;
    desc.Width = shape.width;
    desc.Height = shape.height;
    desc.DepthOrArraySize = shape.depth;
    desc.MipLevels = shape.mips;
    desc.Format = DXGI_FORMAT_A8_UNORM;
    desc.SampleDesc.Count = 1;
    desc.Flags = flags;
    return desc;
}

HRESULT create(ID3D12Device* device, const D3D12_RESOURCE_DESC& desc, D3D12_RESOURCE_STATES state,
               ID3D12Resource** out) {
    // A null `out` only asks whether the description is valid, as the D3D12 documentation allows.
    const D3D12_HEAP_PROPERTIES heap{D3D12_HEAP_TYPE_DEFAULT};
    return device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, state, nullptr,
                                           __uuidof(ID3D12Resource), reinterpret_cast<void**>(out));
}

D3D12_RESOURCE_BARRIER transition(ID3D12Resource* resource, D3D12_RESOURCE_STATES before,
                                  D3D12_RESOURCE_STATES after) {
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = resource;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    barrier.Transition.StateBefore = before;
    barrier.Transition.StateAfter = after;
    return barrier;
}

// Copies a byte pattern into every subresource of the texture, copies it back out into a readback buffer and compares
// every texel of every row and slice.
bool round_trip(ID3D12Device* device, const D3D12_RESOURCE_DESC& desc, unsigned& texels) {
    Ref<ID3D12Resource> texture;
    CHECK(SUCCEEDED(create(device, desc, D3D12_RESOURCE_STATE_COPY_DEST, &texture.p)));
    const UINT subresources = desc.MipLevels *
                              (desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D ? 1u : desc.DepthOrArraySize);
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT layouts[16]{};
    UINT rows[16]{};
    UINT64 row_bytes[16]{}, total = 0;
    CHECK(subresources <= 16);
    device->GetCopyableFootprints(&desc, 0, subresources, 0, layouts, rows, row_bytes, &total);
    CHECK(total != 0 && total != UINT64_MAX);

    D3D12_RESOURCE_DESC buffer{};
    buffer.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    buffer.Width = total;
    buffer.Height = 1;
    buffer.DepthOrArraySize = 1;
    buffer.MipLevels = 1;
    buffer.SampleDesc.Count = 1;
    buffer.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    const D3D12_HEAP_PROPERTIES upload_heap{D3D12_HEAP_TYPE_UPLOAD}, readback_heap{D3D12_HEAP_TYPE_READBACK};
    Ref<ID3D12Resource> upload, readback;
    CHECK(SUCCEEDED(device->CreateCommittedResource(&upload_heap, D3D12_HEAP_FLAG_NONE, &buffer,
                                                    D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                                    IID_PPV_ARGS(&upload.p))));
    CHECK(SUCCEEDED(device->CreateCommittedResource(&readback_heap, D3D12_HEAP_FLAG_NONE, &buffer,
                                                    D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                    IID_PPV_ARGS(&readback.p))));
    auto pattern = [](UINT s, UINT z, UINT y, UINT x) { return std::uint8_t(7 + 13 * s + 5 * z + 3 * y + x); };
    std::uint8_t* bytes = nullptr;
    CHECK(SUCCEEDED(upload->Map(0, nullptr, reinterpret_cast<void**>(&bytes))));
    std::memset(bytes, 0, static_cast<size_t>(total));
    for (UINT s = 0; s < subresources; ++s)
        for (UINT z = 0; z < layouts[s].Footprint.Depth; ++z)
            for (UINT y = 0; y < rows[s]; ++y)
                for (UINT x = 0; x < row_bytes[s]; ++x)
                    bytes[layouts[s].Offset + (z * rows[s] + y) * layouts[s].Footprint.RowPitch + x] = pattern(s, z, y, x);
    upload->Unmap(0, nullptr);

    Ref<ID3D12CommandQueue> queue;
    Ref<ID3D12CommandAllocator> allocator;
    Ref<ID3D12GraphicsCommandList> list;
    Ref<ID3D12Fence> fence;
    const D3D12_COMMAND_QUEUE_DESC queue_desc{};
    CHECK(SUCCEEDED(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue.p))));
    CHECK(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p))));
    CHECK(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, nullptr,
                                             IID_PPV_ARGS(&list.p))));
    CHECK(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))));
    for (UINT s = 0; s < subresources; ++s) {
        D3D12_TEXTURE_COPY_LOCATION to{texture.p, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
        to.SubresourceIndex = s;
        D3D12_TEXTURE_COPY_LOCATION from{upload.p, D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
        from.PlacedFootprint = layouts[s];
        list->CopyTextureRegion(&to, 0, 0, 0, &from, nullptr);
    }
    const auto to_source = transition(texture.p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE);
    list->ResourceBarrier(1, &to_source);
    for (UINT s = 0; s < subresources; ++s) {
        D3D12_TEXTURE_COPY_LOCATION to{readback.p, D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
        to.PlacedFootprint = layouts[s];
        D3D12_TEXTURE_COPY_LOCATION from{texture.p, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
        from.SubresourceIndex = s;
        list->CopyTextureRegion(&to, 0, 0, 0, &from, nullptr);
    }
    CHECK(SUCCEEDED(list->Close()));
    ID3D12CommandList* lists[] = {list.p};
    queue->ExecuteCommandLists(1, lists);
    Event done;
    CHECK(SUCCEEDED(queue->Signal(fence.p, 1)));
    CHECK(SUCCEEDED(fence->SetEventOnCompletion(1, done.handle)));
    CHECK(WaitForSingleObject(done.handle, 10000) == WAIT_OBJECT_0);

    CHECK(SUCCEEDED(readback->Map(0, nullptr, reinterpret_cast<void**>(&bytes))));
    bool same = true;
    for (UINT s = 0; s < subresources; ++s)
        for (UINT z = 0; z < layouts[s].Footprint.Depth; ++z)
            for (UINT y = 0; y < rows[s]; ++y)
                for (UINT x = 0; x < row_bytes[s]; ++x, ++texels) {
                    const std::uint8_t got =
                        bytes[layouts[s].Offset + (z * rows[s] + y) * layouts[s].Footprint.RowPitch + x];
                    if (got != pattern(s, z, y, x) && same) {
                        std::fprintf(stderr, "subresource %u texel (%u, %u, %u): %u, expected %u\n", s, x, y, z,
                                     unsigned(got), unsigned(pattern(s, z, y, x)));
                        same = false;
                    }
                }
    readback->Unmap(0, nullptr);
    return same;
}

// The default SRV of a 1D (128) and a 3D (8x8x4) texture: a T# of the right image type, in R8_UNORM's GFX10 format,
// with the swizzle (0, 0, 0, X). A check of the layer alone, so it runs where there is no Windows reference.
bool default_srv_reads_alpha(ID3D12Device* device) {
    namespace g = leonida::gfx10;
    Ref<ID3D12DescriptorHeap> cpu_heap;
    const D3D12_DESCRIPTOR_HEAP_DESC cpu_desc{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 2, D3D12_DESCRIPTOR_HEAP_FLAG_NONE};
    CHECK(SUCCEEDED(device->CreateDescriptorHeap(&cpu_desc, IID_PPV_ARGS(&cpu_heap.p))));
    const UINT step = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    CHECK(step >= sizeof(g::ImageWords));
    const std::uint32_t r8_code = g::format_from_vulkan(VK_FORMAT_R8_UNORM);
    CHECK(r8_code != 0);
    struct Case {
        Shape shape;
        g::ImageType type;
    };
    const Case cases[] = {
        {{"1D 128", D3D12_RESOURCE_DIMENSION_TEXTURE1D, 128, 1, 1, 1}, g::ImageType::image_1d},
        {{"3D 8x8x4", D3D12_RESOURCE_DIMENSION_TEXTURE3D, 8, 8, 4, 1}, g::ImageType::image_3d},
    };
    for (UINT c = 0; c < 2; ++c) {
        const Case& k = cases[c];
        Ref<ID3D12Resource> texture;
        CHECK(SUCCEEDED(create(device, describe(k.shape, 0, D3D12_RESOURCE_FLAG_NONE), D3D12_RESOURCE_STATE_COMMON,
                               &texture.p)));
        D3D12_CPU_DESCRIPTOR_HANDLE handle = cpu_heap->GetCPUDescriptorHandleForHeapStart();
        handle.ptr += SIZE_T(c) * step;
        std::memset(reinterpret_cast<void*>(handle.ptr), 0x77, step);
        device->CreateShaderResourceView(texture.p, nullptr, handle);
        g::ImageWords words{};
        std::memcpy(words.data(), reinterpret_cast<const void*>(handle.ptr), sizeof(words));
        g::Image image{};
        CHECK(g::is_image(words) && g::decode(words, image));
        CHECK(image.type == k.type && image.width == k.shape.width && image.format == r8_code);
        CHECK(k.type != g::ImageType::image_3d || (image.height == k.shape.height && image.depth == k.shape.depth));
        const g::Swizzle alpha_only{g::select_zero, g::select_zero, g::select_zero, g::select_x};
        CHECK(image.swizzle == alpha_only);
        std::printf("A8_UNORM %s: the default SRV reads (0, 0, 0, A)\n", k.shape.name);
    }
    return true;
}
}  // namespace

int main() {
    vktest::Device vk;
    const auto device_status = vk.initialize();
    if (device_status == vktest::DeviceStatus::unavailable) return 77;  // no Vulkan GPU: ctest skips
    if (device_status != vktest::DeviceStatus::ready) return 1;
    unsigned failures = 0, references = 0, sized = 0, texels = 0;
    {
        Ref<ID3D12Device> ours;
        rt::d3d12::VulkanDeviceInfo info{};
        info.get_instance_proc = vk.instance_proc;
        info.instance = vk.instance;
        info.physical = vk.physical;
        info.device = vk.device;
        info.queue = vk.queue;
        info.queue_family = vk.family;
        info.conditional_rendering = vk.conditional_rendering;
        info.storage_buffer_8bit = vk.storage_buffer_8bit;
        info.buffer_device_address = vk.buffer_device_address;
        if (FAILED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&ours.p)))) return 1;
        if (!default_srv_reads_alpha(ours.p)) ++failures;
        if (hostwine::host_is_wine()) return failures ? 1 : 77;  // every other check is a comparison with Windows D3D12
        Ref<IDXGIFactory4> factory;
        if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory.p)))) {
            std::fprintf(stderr, "no DXGI factory\n");
            return 1;
        }
        Ref<ID3D12Device> hosts[2];  // the layer's own adapter, then WARP
        std::string names[2];
        for (int warp = 0; warp < 2; ++warp) {
            Ref<IDXGIAdapter> adapter;
            if (!hostadapter::reference(factory.p, vk, warp == 1, &adapter.p, names[warp])) continue;
            if (FAILED(D3D12CreateDevice(adapter.p, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&hosts[warp].p))))
                hosts[warp].p = nullptr;
            else
                ++references;
        }
        if (!hosts[1].p) {
            std::fprintf(stderr, "WARP reference unavailable\n");
            return 1;
        }

        // Format support: the dimension bits match every reference.
        constexpr UINT dimension_bits =
            D3D12_FORMAT_SUPPORT1_TEXTURE1D | D3D12_FORMAT_SUPPORT1_TEXTURE2D | D3D12_FORMAT_SUPPORT1_TEXTURE3D;
        D3D12_FEATURE_DATA_FORMAT_SUPPORT ours_support{DXGI_FORMAT_A8_UNORM};
        if (FAILED(ours->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &ours_support, sizeof(ours_support)))) {
            std::fprintf(stderr, "A8_UNORM format support query failed\n");
            ++failures;
        }
        for (int h = 0; h < 2; ++h) {
            if (!hosts[h].p) continue;
            D3D12_FEATURE_DATA_FORMAT_SUPPORT host_support{DXGI_FORMAT_A8_UNORM};
            hosts[h]->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &host_support, sizeof(host_support));
            if ((UINT(ours_support.Support1) & dimension_bits) != (UINT(host_support.Support1) & dimension_bits)) {
                std::fprintf(stderr, "A8_UNORM dimension support 0x%x, %s reports 0x%x\n",
                             UINT(ours_support.Support1) & dimension_bits, names[h].c_str(),
                             UINT(host_support.Support1) & dimension_bits);
                ++failures;
            }
        }

        // Sizing and creation: where every reference sizes and creates the texture, the layer must too, with the
        // reference's alignment (64 KiB, or 4 KiB when the description asks for it and qualifies).
        for (const Shape& shape : kShapes) {
            for (UINT64 alignment : {UINT64(0), UINT64(D3D12_SMALL_RESOURCE_PLACEMENT_ALIGNMENT)}) {
                const D3D12_RESOURCE_DESC desc = describe(shape, alignment, D3D12_RESOURCE_FLAG_NONE);
                bool every_host = true;
                UINT64 host_alignment = 0;
                for (int h = 0; h < 2; ++h) {
                    if (!hosts[h].p) continue;
                    const D3D12_RESOURCE_ALLOCATION_INFO host = hosts[h]->GetResourceAllocationInfo(0, 1, &desc);
                    D3D12_RESOURCE_DESC created = desc;
                    created.Alignment = 0;
                    if (host.SizeInBytes == UINT64_MAX ||
                        FAILED(create(hosts[h].p, created, D3D12_RESOURCE_STATE_COMMON, nullptr)))
                        every_host = false;
                    else if (h == 1)
                        host_alignment = host.Alignment;
                }
                if (!every_host) {
                    std::printf("%s, alignment %llu: a reference refuses it; skipped\n", shape.name,
                                static_cast<unsigned long long>(alignment));
                    continue;
                }
                const D3D12_RESOURCE_ALLOCATION_INFO mine = ours->GetResourceAllocationInfo(0, 1, &desc);
                if (mine.SizeInBytes == UINT64_MAX || mine.Alignment != host_alignment) {
                    std::fprintf(stderr, "%s, alignment %llu: size %llu alignment %llu, WARP alignment %llu\n",
                                 shape.name, static_cast<unsigned long long>(alignment),
                                 static_cast<unsigned long long>(mine.SizeInBytes),
                                 static_cast<unsigned long long>(mine.Alignment),
                                 static_cast<unsigned long long>(host_alignment));
                    ++failures;
                    continue;
                }
                ++sized;
                if (alignment) continue;  // creation ignores the alignment; one round trip per shape
                if (!round_trip(ours.p, desc, texels)) {
                    std::fprintf(stderr, "%s: the round trip failed\n", shape.name);
                    ++failures;
                }
            }
        }

        // Render targets, depth and unordered access stay refused, in every dimension.
        for (const Shape& shape : kShapes) {
            for (auto flag : {D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL,
                              D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS}) {
                const D3D12_RESOURCE_DESC desc = describe(shape, 0, flag);
                if (ours->GetResourceAllocationInfo(0, 1, &desc).SizeInBytes != UINT64_MAX ||
                    SUCCEEDED(create(ours.p, desc, D3D12_RESOURCE_STATE_COMMON, nullptr))) {
                    std::fprintf(stderr, "%s with flag 0x%x: accepted\n", shape.name, unsigned(flag));
                    ++failures;
                }
            }
        }
    }
    std::printf("A8_UNORM dimensions: %u references, %u descriptions sized, %u texels round-tripped, %u failures\n",
                references, sized, texels, failures);
    return failures ? 1 : 0;
}
