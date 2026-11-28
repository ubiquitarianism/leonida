// A vertex or index view that names the title's own memory gets a buffer made over those pages (make_title_buffer,
// TitleVaAliases::want_buffer), the way slot 133 makes one: the pages imported, or a shadow copied from them. When
// neither can be made, slot 133's last resort is an ordinary DEFAULT buffer, which has neither the title's VA nor its
// bytes. A view must not keep such a buffer: it does not cover the view, so the next IASetVertexBuffers of the same
// view would make another, each holding device memory and a device reference until the device goes, until
// LEONIDA_TITLE_BUFFERS_MB is used up and no new title view is resolved for the rest of the session. Nor is such a
// buffer made only to be let go: it would take device-local memory at every try, under the pressure that refused the
// shadow.
//   - The arena is filled first (LEONIDA_ARENA_MB=16, filled with 64 KiB buffers, so no slice is left for a shadow)
//     and host-visible allocations are refused (the test wraps vkAllocateMemory behind the instance-proc pointer it
//     hands the layer), with LEONIDA_SLOT133_IMPORT=0: no import and no shadow can be made, but a device-local DEFAULT
//     buffer could.
//   - The same 2 MiB view of title memory bound 16 times, as vertex and index views, keeps no buffer (every buffer
//     made over title memory holds a device reference, and the device's count does not grow) and is tried once: one
//     buffer of its size is created, the shadow's, and no device-local memory of its size is allocated.
//   - A second, 512 KiB view bound right after that refusal is not tried: after a refusal no range is tried for a
//     second, whichever range it is.
//   - After that second the first view is tried again, once, and refused again. 1.5 s later the second view is tried,
//     but the first is not: a range refused again waits twice as long. After that it is tried once more.
// Needs a device-local memory type that is not host-visible (skipped otherwise). Synthetic data only. Set
// LEONIDA_VK_VALIDATION=1 to run under VK_LAYER_KHRONOS_validation.
#include "vk_test_device.h"
#include "../shims/d3d12_com.h"
#include "../src/title_va_alias.h"

#include <atomic>
#include <cstdio>
#include <cstring>
#include <vector>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

namespace {

template<class T>
struct Ref {
    T* p = nullptr;
    Ref() = default;
    Ref(const Ref&) = delete;  // a copy would release the same interface twice
    Ref& operator=(const Ref&) = delete;
    ~Ref() { reset(); }
    void reset() {
        if (p) p->Release();
        p = nullptr;
    }
    T* operator->() const { return p; }
};

constexpr UINT large_bytes = 2u << 20, small_bytes = 512u << 10;  // the two views; no other buffer has these sizes

// The Vulkan calls the layer makes, seen through the function pointers it loads.
PFN_vkGetInstanceProcAddr real_instance_proc = nullptr;
PFN_vkGetDeviceProcAddr real_device_proc = nullptr;
PFN_vkAllocateMemory real_allocate = nullptr;
PFN_vkCreateBuffer real_create_buffer = nullptr;
VkPhysicalDeviceMemoryProperties memory{};
std::atomic<bool> refuse_host_visible{false};
std::atomic<unsigned> large_buffers{0}, small_buffers{0};  // buffers of each view's size created
std::atomic<unsigned> device_allocations{0};  // device-local allocations of at least the smaller view's size, made
                                              // while host-visible ones are refused

VKAPI_ATTR VkResult VKAPI_CALL allocate(VkDevice device, const VkMemoryAllocateInfo* info,
                                        const VkAllocationCallbacks* allocator, VkDeviceMemory* out) {
    if (refuse_host_visible.load() && info->memoryTypeIndex < memory.memoryTypeCount) {
        const VkMemoryPropertyFlags flags = memory.memoryTypes[info->memoryTypeIndex].propertyFlags;
        if (flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) return VK_ERROR_OUT_OF_DEVICE_MEMORY;
        if ((flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) && info->allocationSize >= small_bytes) ++device_allocations;
    }
    return real_allocate(device, info, allocator, out);
}

VKAPI_ATTR VkResult VKAPI_CALL create_buffer(VkDevice device, const VkBufferCreateInfo* info,
                                             const VkAllocationCallbacks* allocator, VkBuffer* buffer) {
    if (info->size == large_bytes) ++large_buffers;
    if (info->size == small_bytes) ++small_buffers;
    return real_create_buffer(device, info, allocator, buffer);
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL device_proc(VkDevice device, const char* name) {
    if (std::strcmp(name, "vkAllocateMemory") == 0) {
        real_allocate = reinterpret_cast<PFN_vkAllocateMemory>(real_device_proc(device, name));
        return real_allocate ? reinterpret_cast<PFN_vkVoidFunction>(&allocate) : nullptr;
    }
    if (std::strcmp(name, "vkCreateBuffer") == 0) {
        real_create_buffer = reinterpret_cast<PFN_vkCreateBuffer>(real_device_proc(device, name));
        return real_create_buffer ? reinterpret_cast<PFN_vkVoidFunction>(&create_buffer) : nullptr;
    }
    return real_device_proc(device, name);
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL instance_proc(VkInstance instance, const char* name) {
    if (std::strcmp(name, "vkGetDeviceProcAddr") == 0) {
        real_device_proc = reinterpret_cast<PFN_vkGetDeviceProcAddr>(real_instance_proc(instance, name));
        return real_device_proc ? reinterpret_cast<PFN_vkVoidFunction>(&device_proc) : nullptr;
    }
    return real_instance_proc(instance, name);
}

D3D12_RESOURCE_DESC buffer_desc(UINT64 bytes) {
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = bytes;
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    return desc;
}

// The device's reference count: every buffer the device keeps over title memory holds one.
ULONG references(ID3D12Device* device) {
    device->AddRef();
    return device->Release();
}

}  // namespace

int main() {
    vktest::Device vk;
    if (!vk.create()) return 77;
    real_instance_proc = vk.instance_proc;
    reinterpret_cast<PFN_vkGetPhysicalDeviceMemoryProperties>(
        vk.instance_proc(vk.instance, "vkGetPhysicalDeviceMemoryProperties"))(vk.physical, &memory);
    bool device_only = false;
    for (std::uint32_t i = 0; i < memory.memoryTypeCount; ++i)
        device_only |= (memory.memoryTypes[i].propertyFlags &
                        (VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) ==
                       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    if (!device_only) {
        std::fprintf(stderr, "SKIP: every device-local memory type is host-visible\n");
        return 77;
    }

    // Title memory of 4 MiB: the large view is its first 2 MiB, the small one starts 3 MiB in (another range).
    constexpr UINT title_bytes = 4u << 20;
    auto& aliases = leonida::title_va_aliases();
    void* pages = VirtualAlloc(nullptr, title_bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    EXPECT(pages);
    const auto title_va = reinterpret_cast<std::uintptr_t>(pages);
    const auto small_va = title_va + (3u << 20);
    aliases.add_title_memory(title_va, title_bytes);
    int result = 0;
    {
        rt::d3d12::VulkanDeviceInfo info{};
        info.get_instance_proc = &instance_proc;
        info.instance = vk.instance;
        info.physical = vk.physical;
        info.device = vk.device;
        info.queue = vk.queue;
        info.queue_family = vk.family;
        info.buffer_device_address = true;
        Ref<ID3D12Device> device;
        EXPECT(SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&device.p))));
        const std::uint64_t arena = rt::d3d12::device_arena_size(device.p);
        if (!arena) {
            std::fprintf(stderr, "SKIP: the device has no GPU address arena\n");
            result = 77;
        } else {
            // Fill the arena: a 64 KiB buffer (its placement alignment) that is not given an arena VA found no room
            // there, so no shadow will find any either.
            const D3D12_HEAP_PROPERTIES default_heap{D3D12_HEAP_TYPE_DEFAULT};
            const D3D12_RESOURCE_DESC filler_desc = buffer_desc(64u << 10);
            std::vector<Ref<ID3D12Resource>> fillers(arena >> 16);
            bool full = false;
            for (auto& filler : fillers) {
                EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &filler_desc,
                                                                 D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                                 IID_PPV_ARGS(&filler.p))));
                if ((full = filler->GetGPUVirtualAddress() >= arena)) break;
            }
            EXPECT(full);

            Ref<ID3D12CommandAllocator> allocator;
            Ref<ID3D12GraphicsCommandList> list;
            EXPECT(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                                            IID_PPV_ARGS(&allocator.p))));
            EXPECT(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, nullptr,
                                                       IID_PPV_ARGS(&list.p))));
            const D3D12_VERTEX_BUFFER_VIEW large_vertices{title_va, large_bytes, 16};
            const D3D12_INDEX_BUFFER_VIEW large_indices{title_va, large_bytes, DXGI_FORMAT_R16_UINT};
            const D3D12_VERTEX_BUFFER_VIEW small_vertices{small_va, small_bytes, 16};
            const D3D12_INDEX_BUFFER_VIEW small_indices{small_va, small_bytes, DXGI_FORMAT_R16_UINT};
            const auto bind_large = [&](unsigned times) {
                for (unsigned i = 0; i < times; ++i) {
                    if (i & 1) list->IASetIndexBuffer(&large_indices);
                    else list->IASetVertexBuffers(0, 1, &large_vertices);
                }
                return large_buffers.load();
            };
            const auto bind_small = [&](unsigned times) {
                for (unsigned i = 0; i < times; ++i) {
                    if (i & 1) list->IASetIndexBuffer(&small_indices);
                    else list->IASetVertexBuffers(0, 1, &small_vertices);
                }
                return small_buffers.load();
            };

            // No import, no shadow: nothing over the title's pages is kept, and the view is tried once, with no
            // device-local stand-in made for it. The other view, bound right after, is not tried.
            const ULONG before = references(device.p);
            refuse_host_visible = true;
            const unsigned first = bind_large(1);
            const unsigned tried = bind_large(15);
            const unsigned small_held = bind_small(8);
            const ULONG refused = references(device.p);
            std::printf("16 binds with no import or shadow: %lu buffers kept, %u buffers of the view's size created "
                        "(%u by the first bind), %u device-local allocations of its size; 8 binds of another view "
                        "right after: %u buffers of its size created\n",
                        refused - before, tried, first, device_allocations.load(), small_held);
            EXPECT(refused == before);
            EXPECT(device_allocations == 0);
            EXPECT(first == 1);
            EXPECT(tried == first);
            EXPECT(small_held == 0);
            EXPECT(aliases.shader_va(title_va) == title_va);

            // After the second, the large view is tried again, once, and refused again. Host-visible memory is still
            // refused here and below: a buffer the device keeps over title memory holds a reference to it, so the
            // device would never be destroyed, and its Vulkan objects would be reported as leaked under the
            // validation layer.
            Sleep(1500);
            const unsigned retried = bind_large(1);
            const unsigned again = bind_large(7);
            std::printf("after a second: %u buffers of the view's size created by the next bind, %u by 7 further "
                        "binds\n",
                        retried - tried, again - retried);
            EXPECT(retried == tried + 1);
            EXPECT(again == retried);

            // 1.5 s after that second refusal the other view is tried (no range waits for the device any more), but
            // the large view is not: refused twice, it waits 2 s.
            Sleep(1500);
            const unsigned waited = bind_large(8);
            const unsigned small_tried = bind_small(8);
            std::printf("1.5 s later: %u buffers of the large view's size created by 8 binds, %u of the other "
                        "view's by 8 binds\n",
                        waited - again, small_tried - small_held);
            EXPECT(waited == again);
            EXPECT(small_tried == small_held + 1);

            // Once its 2 s have passed (and the other view's refusal is a second old), the large view is tried again.
            Sleep(1500);
            const unsigned third = bind_large(8);
            refuse_host_visible = false;
            const ULONG after = references(device.p);
            std::printf("1.5 s later: %u buffers of the large view's size created by 8 binds; %lu buffers kept, %u "
                        "device-local allocations of a view's size in all\n",
                        third - waited, after - before, device_allocations.load());
            EXPECT(third == waited + 1);
            EXPECT(after == before);
            EXPECT(device_allocations == 0);
            EXPECT(aliases.shader_va(title_va) == title_va);
            EXPECT(aliases.shader_va(small_va) == small_va);
            EXPECT(SUCCEEDED(list->Close()));
        }
    }
    aliases.remove_title_memory(title_va);
    VirtualFree(pages, 0, MEM_RELEASE);
    if (vk.validation_errors()) {
        std::fprintf(stderr, "%u validation errors\n", vk.validation_errors());
        return 1;
    }
    if (!result) std::printf("ok\n");
    return result;
}
