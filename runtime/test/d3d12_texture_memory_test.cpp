// Textures share blocks of device memory (src/image_memory.h) on the Vulkan-backed device: many small textures take a
// few driver allocations, each keeps its own texels, a released texture's range is used again without touching its
// neighbours, and a texture made in a used range reads as zero before anything writes it, as a committed resource does.
// Render targets and large textures keep their own memory. No game data. LEONIDA_VK_VALIDATION=1 adds
// VK_LAYER_KHRONOS_validation.
#include "host_wine.h"
#include "../shims/d3d12_com.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {
int failures = 0;
void check(bool ok, const char* what) {
    if (ok) return;
    std::printf("FAIL: %s\n", what);
    ++failures;
}

template<class T>
struct Ref {
    T* p = nullptr;
    Ref() = default;
    Ref(const Ref&) = delete;
    Ref& operator=(const Ref&) = delete;
    Ref(Ref&& other) noexcept : p(other.p) { other.p = nullptr; }
    ~Ref() { reset(); }
    T* operator->() const { return p; }
    void reset() {
        if (p) p->Release();
        p = nullptr;
    }
};

constexpr UINT kSide = 64;
constexpr UINT kPitch = kSide * 4;  // a multiple of D3D12_TEXTURE_DATA_PITCH_ALIGNMENT
constexpr UINT kBytes = kPitch * kSide;

D3D12_RESOURCE_DESC texture_desc(UINT side, UINT16 mips, D3D12_RESOURCE_FLAGS flags) {
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = desc.Height = side;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = mips;
    desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    desc.SampleDesc.Count = 1;
    desc.Flags = flags;
    return desc;
}

D3D12_RESOURCE_DESC buffer_desc(UINT64 bytes) {
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = bytes;
    desc.Height = desc.DepthOrArraySize = desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    return desc;
}

struct Gpu {
    Ref<ID3D12Device> device;
    Ref<ID3D12CommandQueue> queue;
    Ref<ID3D12CommandAllocator> allocator;
    Ref<ID3D12GraphicsCommandList> list;
    Ref<ID3D12Fence> fence;
    HANDLE event = nullptr;
    UINT64 value = 0;

    ~Gpu() {
        if (event) CloseHandle(event);
    }

    bool init() {
        if (FAILED(rt::d3d12::create_vulkan_device_standalone(IID_PPV_ARGS(&device.p))) || !device.p) return false;
        D3D12_COMMAND_QUEUE_DESC desc{};
        event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        return event && SUCCEEDED(device->CreateCommandQueue(&desc, IID_PPV_ARGS(&queue.p))) &&
               SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p))) &&
               SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, nullptr,
                                                   IID_PPV_ARGS(&list.p))) &&
               SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p)));
    }

    // Runs what was recorded and waits for it, so that what was released before it is gone.
    bool run() {
        if (FAILED(list->Close())) return false;
        ID3D12CommandList* lists[] = {list.p};
        queue->ExecuteCommandLists(1, lists);
        if (FAILED(queue->Signal(fence.p, ++value)) || FAILED(fence->SetEventOnCompletion(value, event))) return false;
        if (WaitForSingleObject(event, 30000) != WAIT_OBJECT_0) return false;
        return SUCCEEDED(allocator->Reset()) && SUCCEEDED(list->Reset(allocator.p, nullptr));
    }

    HRESULT create(D3D12_HEAP_TYPE heap, const D3D12_RESOURCE_DESC& desc, D3D12_RESOURCE_STATES state,
                   Ref<ID3D12Resource>& out, D3D12_HEAP_FLAGS flags = D3D12_HEAP_FLAG_NONE) {
        D3D12_HEAP_PROPERTIES properties{};
        properties.Type = heap;
        return device->CreateCommittedResource(&properties, flags, &desc, state, nullptr, IID_PPV_ARGS(&out.p));
    }
};

D3D12_TEXTURE_COPY_LOCATION place(ID3D12Resource* resource, UINT64 offset) {
    D3D12_TEXTURE_COPY_LOCATION location{};
    location.pResource = resource;
    location.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    location.PlacedFootprint.Offset = offset;
    location.PlacedFootprint.Footprint = {DXGI_FORMAT_R8G8B8A8_UNORM, kSide, kSide, 1, kPitch};
    return location;
}

D3D12_TEXTURE_COPY_LOCATION whole(ID3D12Resource* resource) {
    D3D12_TEXTURE_COPY_LOCATION location{};
    location.pResource = resource;
    location.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    return location;
}

void transition(ID3D12GraphicsCommandList* list, ID3D12Resource* resource, D3D12_RESOURCE_STATES from,
                D3D12_RESOURCE_STATES to) {
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition = {resource, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, from, to};
    list->ResourceBarrier(1, &barrier);
}

// Reads every texture back and compares each with `expected(index)` (one byte, repeated).
template<typename Expected>
int mismatches(Gpu& gpu, const std::vector<Ref<ID3D12Resource>>& textures, Expected&& expected) {
    int bad = 0;
    constexpr std::size_t kBatch = 64;
    Ref<ID3D12Resource> readback;
    if (FAILED(gpu.create(D3D12_HEAP_TYPE_READBACK, buffer_desc(UINT64(kBytes) * kBatch),
                          D3D12_RESOURCE_STATE_COPY_DEST, readback)))
        return int(textures.size());
    for (std::size_t first = 0; first < textures.size(); first += kBatch) {
        const std::size_t count = std::min(kBatch, textures.size() - first);
        for (std::size_t i = 0; i < count; ++i) {
            ID3D12Resource* texture = textures[first + i].p;
            if (!texture) continue;
            transition(gpu.list.p, texture, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE);
            const auto from = whole(texture);
            const auto to = place(readback.p, UINT64(kBytes) * i);
            gpu.list->CopyTextureRegion(&to, 0, 0, 0, &from, nullptr);
            transition(gpu.list.p, texture, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);
        }
        if (!gpu.run()) return int(textures.size());
        const std::uint8_t* data = nullptr;
        D3D12_RANGE range{0, kBytes * count};
        if (FAILED(readback->Map(0, &range, reinterpret_cast<void**>(const_cast<std::uint8_t**>(&data))))) return int(textures.size());
        for (std::size_t i = 0; i < count; ++i) {
            if (!textures[first + i].p) continue;
            const std::uint8_t value = expected(first + i);
            for (UINT b = 0; b < kBytes; ++b)
                if (data[UINT64(kBytes) * i + b] != value) {
                    ++bad;
                    break;
                }
        }
        readback->Unmap(0, nullptr);
    }
    return bad;
}
}  // namespace

int main() {
    Gpu gpu;
    if (!gpu.init()) {
        std::printf("SKIP: no Vulkan device\n");
        return 77;  // ctest skips
    }
    constexpr std::size_t kTextures = 256;
    const auto desc = texture_desc(kSide, 1, D3D12_RESOURCE_FLAG_NONE);
    const auto before = rt::d3d12::image_memory_stats(gpu.device.p);
    check(before.textures == 0, "no texture is in a block at first");

    std::vector<Ref<ID3D12Resource>> textures(kTextures);
    for (auto& texture : textures)
        check(SUCCEEDED(gpu.create(D3D12_HEAP_TYPE_DEFAULT, desc, D3D12_RESOURCE_STATE_COMMON, texture)),
              "create a texture");
    const auto all = rt::d3d12::image_memory_stats(gpu.device.p);
    std::printf("%zu textures: %llu in %llu blocks of %llu bytes\n", kTextures, (unsigned long long)all.textures,
                (unsigned long long)all.blocks, (unsigned long long)all.block_bytes);
    check(all.blocks != 0, "small textures share blocks");
    check(all.textures == kTextures, "every small texture is in a block");
    check(all.blocks <= 2, "256 small textures take at most two driver allocations");
    const auto value_of = [](std::size_t i) { return std::uint8_t(1 + i % 250); };

    // A value of its own in each, through an upload buffer, then read back: neighbours must not overwrite each other.
    Ref<ID3D12Resource> upload;
    check(SUCCEEDED(gpu.create(D3D12_HEAP_TYPE_UPLOAD, buffer_desc(UINT64(kBytes) * kTextures),
                               D3D12_RESOURCE_STATE_GENERIC_READ, upload)),
          "create the upload buffer");
    std::uint8_t* mapped = nullptr;
    if (!upload.p || FAILED(upload->Map(0, nullptr, reinterpret_cast<void**>(&mapped)))) return 1;
    for (std::size_t i = 0; i < kTextures; ++i) std::memset(mapped + UINT64(kBytes) * i, value_of(i), kBytes);
    for (std::size_t i = 0; i < kTextures; ++i) {
        transition(gpu.list.p, textures[i].p, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
        const auto to = whole(textures[i].p);
        const auto from = place(upload.p, UINT64(kBytes) * i);
        gpu.list->CopyTextureRegion(&to, 0, 0, 0, &from, nullptr);
        transition(gpu.list.p, textures[i].p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON);
    }
    check(gpu.run(), "upload");
    check(mismatches(gpu, textures, value_of) == 0, "each texture keeps its own texels");

    // Release every other one; its range goes back when the next submission has completed.
    for (std::size_t i = 1; i < kTextures; i += 2) textures[i].reset();
    check(gpu.run() && gpu.run(), "submit after the release");
    const auto half = rt::d3d12::image_memory_stats(gpu.device.p);
    check(half.textures == kTextures / 2, "a released texture leaves its block");

    // New textures in those ranges: no new block, zero before anything writes them, and the others are untouched.
    std::vector<Ref<ID3D12Resource>> made(kTextures / 2);
    for (auto& texture : made)
        check(SUCCEEDED(gpu.create(D3D12_HEAP_TYPE_DEFAULT, desc, D3D12_RESOURCE_STATE_COMMON, texture)),
              "create a texture in a used range");
    const auto again = rt::d3d12::image_memory_stats(gpu.device.p);
    check(again.textures == kTextures && again.blocks == half.blocks, "the released ranges are used again");
    check(mismatches(gpu, made, [](std::size_t) { return std::uint8_t(0); }) == 0,
          "a texture in a used range reads as zero");
    check(mismatches(gpu, textures, value_of) == 0, "its neighbours keep their texels");

    // Render targets and large textures have their own allocations.
    Ref<ID3D12Resource> target, large;
    const auto counted = rt::d3d12::image_memory_stats(gpu.device.p).textures;
    check(SUCCEEDED(gpu.create(D3D12_HEAP_TYPE_DEFAULT,
                               texture_desc(kSide, 1, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET),
                               D3D12_RESOURCE_STATE_COMMON, target)),
          "create a render target");
    check(SUCCEEDED(gpu.create(D3D12_HEAP_TYPE_DEFAULT, texture_desc(4096, 1, D3D12_RESOURCE_FLAG_NONE),
                               D3D12_RESOURCE_STATE_COMMON, large)),
          "create a large texture");
    check(rt::d3d12::image_memory_stats(gpu.device.p).textures == counted,
          "a render target and a large texture are not in a block");

    textures.clear();
    made.clear();
    check(gpu.run() && gpu.run(), "submit after releasing every texture");
    const auto none = rt::d3d12::image_memory_stats(gpu.device.p);
    check(none.textures == 0 && none.blocks <= 1, "with every texture gone, at most one spare block is left");
    if (!failures) std::printf("texture memory: textures share blocks, keep their texels, and start zeroed\n");
    return failures ? 1 : 0;
}
