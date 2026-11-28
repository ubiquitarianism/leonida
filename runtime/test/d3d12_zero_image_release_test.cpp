// A sampled BC texture the title places on its own memory (Xbox device slot 133) waits for memory of its own until
// the title writes it or a command uses it; until then its views name one shared, zeroed image of its exact shape
// (DeviceState::zero_sampled_image). Those images sit in host memory on a discrete GPU and count against a 256 MiB
// cap. One that no placed texture shows any more must go, after the submissions that may still read it, instead of
// staying for the device's life:
//   - Placements of many distinct shapes, one shape twice, each show a shared image (no memory of their own yet), or
//     the test is skipped. The device then keeps one image per shape.
//   - Releasing one of the two textures of a shape keeps that shape's image.
//   - Releasing the rest leaves no shared image and no bytes counted against the cap; after a submission has run and
//     the next one starts, every one of those images has been destroyed, and none while a view of it was left (two
//     placements had a shader-resource view, which is a view of the shared image, sampled by nothing but alive).
//   - A placement of a shape seen before gets a shared image again. Once a copy into it gives it memory of its own,
//     its shape's image goes while the texture lives on: after the next submission no shared image or byte is left,
//     and the image is destroyed after the views of it. That copy's replay on the direct queue gives the claim back
//     in its prepare and submits later: a compute queue's replay begun meanwhile (from inside the copy's
//     vkEndCommandBuffer) must not release the image before the copy's replay has recorded its submission, or the
//     image's deferred deletion would be due once the compute queue's submission completes, with the copy's pending.
//   - Shapes placed until one does not fit in the 256 MiB: once one of them is released, the one that did not fit
//     gets an image.
// The device has no host image copies, so a placement takes the shared image whatever its driver allows for host-copy
// images. Needs the bindless set (skipped otherwise). Synthetic data only. Set LEONIDA_VK_VALIDATION=1 to run under
// VK_LAYER_KHRONOS_validation.
#include "vk_test_device.h"
#include "../shims/d3d12_com.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <map>
#include <mutex>
#include <set>
#include <thread>
#include <utility>
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

constexpr UINT shapes = 12, first_width = 64, height = 52;  // no other image of the test has this height

// The Vulkan calls the layer makes, seen through the function pointers it loads. The first image made for a shape
// is its shared image: it is made before the placement's own.
PFN_vkGetInstanceProcAddr real_instance_proc = nullptr;
PFN_vkGetDeviceProcAddr real_device_proc = nullptr;
PFN_vkCreateImage real_create_image = nullptr;
PFN_vkDestroyImage real_destroy_image = nullptr;
PFN_vkCreateImageView real_create_view = nullptr;
PFN_vkDestroyImageView real_destroy_view = nullptr;
std::mutex hook_mutex;
std::set<UINT> widths_seen;     // shapes that had an image made
std::set<VkImage> shared_live;  // shared images made and not destroyed yet
std::map<VkImageView, VkImage> shared_views;  // views of those, not destroyed yet
unsigned shared_made = 0, shared_destroyed = 0, views_made = 0, destroyed_under_view = 0;

// A placement waiting on the shared zero image has no memory of its own yet.
bool shared(ID3D12Resource* texture) { return !rt::d3d12::vulkan_memory_flags(texture); }

// The copy's replay ending its command buffer, after its prepare gave the texture memory of its own (and gave its
// claim back), starts the other queue's replay and watches for up to half a second whether the shape's image is
// released meanwhile: its own submission is not recorded yet.
PFN_vkEndCommandBuffer real_end_command_buffer = nullptr;
std::atomic<ID3D12Resource*> copy_target{nullptr};  // armed: the texture the copy gives memory of its own
ID3D12Device* hook_device = nullptr;
ID3D12CommandQueue* other_queue = nullptr;
ID3D12CommandList* other_list = nullptr;
std::thread other_replay;
bool released_during_copy = false;

VKAPI_ATTR VkResult VKAPI_CALL end_command_buffer(VkCommandBuffer command_buffer) {
    ID3D12Resource* texture = copy_target.load();
    if (texture && !shared(texture) && copy_target.compare_exchange_strong(texture, nullptr)) {
        other_replay = std::thread([] {
            ID3D12CommandList* lists[] = {other_list};
            other_queue->ExecuteCommandLists(1, lists);
        });
        const auto until = std::chrono::steady_clock::now() + std::chrono::milliseconds(500);
        while (std::chrono::steady_clock::now() < until && !released_during_copy) {
            released_during_copy = rt::d3d12::shared_zero_images(hook_device).images == 0;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
    return real_end_command_buffer(command_buffer);
}

VKAPI_ATTR VkResult VKAPI_CALL create_image(VkDevice device, const VkImageCreateInfo* info,
                                            const VkAllocationCallbacks* allocator, VkImage* image) {
    const VkResult result = real_create_image(device, info, allocator, image);
    if (result == VK_SUCCESS && info->extent.height == height) {
        std::lock_guard lock(hook_mutex);
        if (widths_seen.insert(info->extent.width).second) {
            shared_live.insert(*image);
            ++shared_made;
        }
    }
    return result;
}

VKAPI_ATTR void VKAPI_CALL destroy_image(VkDevice device, VkImage image, const VkAllocationCallbacks* allocator) {
    {
        std::lock_guard lock(hook_mutex);
        if (shared_live.erase(image)) {
            ++shared_destroyed;
            for (const auto& [view, of] : shared_views) destroyed_under_view += of == image;
        }
    }
    real_destroy_image(device, image, allocator);
}

VKAPI_ATTR VkResult VKAPI_CALL create_view(VkDevice device, const VkImageViewCreateInfo* info,
                                           const VkAllocationCallbacks* allocator, VkImageView* view) {
    const VkResult result = real_create_view(device, info, allocator, view);
    if (result == VK_SUCCESS) {
        std::lock_guard lock(hook_mutex);
        if (shared_live.count(info->image)) {
            shared_views[*view] = info->image;
            ++views_made;
        }
    }
    return result;
}

VKAPI_ATTR void VKAPI_CALL destroy_view(VkDevice device, VkImageView view, const VkAllocationCallbacks* allocator) {
    {
        std::lock_guard lock(hook_mutex);
        shared_views.erase(view);
    }
    real_destroy_view(device, view, allocator);
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL device_proc(VkDevice device, const char* name) {
    if (std::strcmp(name, "vkCreateImage") == 0) {
        real_create_image = reinterpret_cast<PFN_vkCreateImage>(real_device_proc(device, name));
        return real_create_image ? reinterpret_cast<PFN_vkVoidFunction>(&create_image) : nullptr;
    }
    if (std::strcmp(name, "vkDestroyImage") == 0) {
        real_destroy_image = reinterpret_cast<PFN_vkDestroyImage>(real_device_proc(device, name));
        return real_destroy_image ? reinterpret_cast<PFN_vkVoidFunction>(&destroy_image) : nullptr;
    }
    if (std::strcmp(name, "vkCreateImageView") == 0) {
        real_create_view = reinterpret_cast<PFN_vkCreateImageView>(real_device_proc(device, name));
        return real_create_view ? reinterpret_cast<PFN_vkVoidFunction>(&create_view) : nullptr;
    }
    if (std::strcmp(name, "vkDestroyImageView") == 0) {
        real_destroy_view = reinterpret_cast<PFN_vkDestroyImageView>(real_device_proc(device, name));
        return real_destroy_view ? reinterpret_cast<PFN_vkVoidFunction>(&destroy_view) : nullptr;
    }
    if (std::strcmp(name, "vkEndCommandBuffer") == 0) {
        real_end_command_buffer = reinterpret_cast<PFN_vkEndCommandBuffer>(real_device_proc(device, name));
        return real_end_command_buffer ? reinterpret_cast<PFN_vkVoidFunction>(&end_command_buffer) : nullptr;
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

D3D12_RESOURCE_DESC bc1(UINT w, UINT h) {
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = w;
    desc.Height = h;
    desc.DepthOrArraySize = desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Format = DXGI_FORMAT_BC1_UNORM;
    return desc;
}

}  // namespace

int main() {
    vktest::Device vk;
    if (!vk.create()) return 77;
    real_instance_proc = vk.instance_proc;
    constexpr unsigned slots = shapes + 2;
    void* pages = VirtualAlloc(nullptr, slots * 65536, MEM_RESERVE, PAGE_NOACCESS);
    EXPECT(pages);
    const auto at = [&](unsigned i) { return static_cast<char*>(pages) + i * 65536; };
    // The shapes that fill the cap: BC1 8192 x (8192 - 4 i), about 32 MiB each, so it is full after about eight.
    constexpr unsigned big_slots = 16;
    constexpr UINT big_width = 8192, big_height = 8192;
    constexpr SIZE_T big_bytes = SIZE_T(big_width) * big_height / 2;
    void* big_pages = VirtualAlloc(nullptr, big_slots * big_bytes, MEM_RESERVE, PAGE_NOACCESS);
    EXPECT(big_pages);
    const auto big_at = [&](unsigned i) { return static_cast<char*>(big_pages) + i * big_bytes; };
    int result = 0;
    {
        rt::d3d12::VulkanDeviceInfo info{};
        info.get_instance_proc = &instance_proc;
        info.instance = vk.instance;
        info.physical = vk.physical;
        info.device = vk.device;
        info.queue = vk.queue;
        info.queue_family = vk.family;
        info.bindless = vk.bindless;
        info.bindless_partially_bound = vk.partially_bound;
        Ref<ID3D12Device> device;
        EXPECT(SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&device.p))));
        Ref<ID3D12Resource> textures[shapes], twin, again, upload;
        Ref<ID3D12DescriptorHeap> heap;
        D3D12_COMMAND_QUEUE_DESC queue_desc{};
        queue_desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        Ref<ID3D12CommandQueue> queue;
        Ref<ID3D12CommandAllocator> allocator;
        Ref<ID3D12GraphicsCommandList> list;
        Ref<ID3D12Fence> fence;
        Ref<ID3D12CommandQueue> compute_queue;  // the other queue, with an empty list
        Ref<ID3D12CommandAllocator> compute_allocator;
        Ref<ID3D12GraphicsCommandList> compute_list;
        UINT64 round = 0;
        // One list with what `record` puts in it, run to completion.
        const auto submit = [&](auto&& record) {
            if (FAILED(allocator->Reset()) || FAILED(list->Reset(allocator.p, nullptr))) return false;
            record();
            if (list->Close() != S_OK) return false;
            ID3D12CommandList* lists[] = {list.p};
            queue->ExecuteCommandLists(1, lists);
            ++round;
            return SUCCEEDED(queue->Signal(fence.p, round)) && SUCCEEDED(fence->SetEventOnCompletion(round, nullptr));
        };
        const auto empty = [] {};
        const auto destroyed = [] {
            std::lock_guard lock(hook_mutex);
            return shared_destroyed;
        };
        if (!rt::d3d12::bindless_set_layout(device.p)) {
            std::fprintf(stderr, "SKIP: the device has no bindless set\n");
            result = 77;
        } else if (FAILED(rt::d3d12::create_buffer_at_host_address(device.p, at(0), bc1(first_width, height), 0,
                                                                   IID_PPV_ARGS(&textures[0].p))) ||
                   !shared(textures[0].p)) {
            std::fprintf(stderr, "SKIP: a placed BC texture does not take a shared zero image on this device\n");
            result = 77;
        } else {
            EXPECT(SUCCEEDED(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue.p))));
            EXPECT(SUCCEEDED(device->CreateCommandAllocator(queue_desc.Type, IID_PPV_ARGS(&allocator.p))));
            EXPECT(SUCCEEDED(device->CreateCommandList(0, queue_desc.Type, allocator.p, nullptr,
                                                       IID_PPV_ARGS(&list.p))));
            EXPECT(list->Close() == S_OK);
            EXPECT(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))));
            D3D12_COMMAND_QUEUE_DESC compute_desc{};
            compute_desc.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
            EXPECT(SUCCEEDED(device->CreateCommandQueue(&compute_desc, IID_PPV_ARGS(&compute_queue.p))));
            EXPECT(SUCCEEDED(device->CreateCommandAllocator(compute_desc.Type,
                                                            IID_PPV_ARGS(&compute_allocator.p))));
            EXPECT(SUCCEEDED(device->CreateCommandList(0, compute_desc.Type, compute_allocator.p, nullptr,
                                                       IID_PPV_ARGS(&compute_list.p))));
            EXPECT(compute_list->Close() == S_OK);
            hook_device = device.p;
            other_queue = compute_queue.p;
            other_list = compute_list.p;
            D3D12_DESCRIPTOR_HEAP_DESC heap_desc{};
            heap_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
            heap_desc.NumDescriptors = 4;
            heap_desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
            EXPECT(SUCCEEDED(device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&heap.p))));
            const UINT increment = device->GetDescriptorHandleIncrementSize(heap_desc.Type);
            const auto slot = [&](UINT i) {
                D3D12_CPU_DESCRIPTOR_HANDLE handle = heap->GetCPUDescriptorHandleForHeapStart();
                handle.ptr += SIZE_T(i) * increment;
                return handle;
            };

            for (UINT i = 1; i < shapes; ++i) {
                EXPECT(SUCCEEDED(rt::d3d12::create_buffer_at_host_address(
                    device.p, at(i), bc1(first_width + 4 * i, height), 0, IID_PPV_ARGS(&textures[i].p))));
                EXPECT(shared(textures[i].p));
            }
            EXPECT(SUCCEEDED(rt::d3d12::create_buffer_at_host_address(device.p, at(shapes), bc1(first_width, height),
                                                                      0, IID_PPV_ARGS(&twin.p))));
            EXPECT(shared(twin.p));
            const auto placed = rt::d3d12::shared_zero_images(device.p);
            std::printf("%u shapes placed (one twice): %llu shared images, %llu bytes; %u made\n", shapes,
                        static_cast<unsigned long long>(placed.images), static_cast<unsigned long long>(placed.bytes),
                        shared_made);
            EXPECT(placed.images == shapes);
            EXPECT(placed.bytes > 0);
            EXPECT(shared_made == shapes);

            // Views of a placement name its shape's shared image while it has no memory of its own.
            device->CreateShaderResourceView(textures[1].p, nullptr, slot(0));
            device->CreateShaderResourceView(twin.p, nullptr, slot(1));
            {
                std::lock_guard lock(hook_mutex);
                std::printf("views of the shared images: %u\n", views_made);
                EXPECT(views_made > 0);
            }
            EXPECT(shared(textures[1].p) && shared(twin.p));

            // The shape's other texture still shows its image.
            textures[0].reset();
            const auto one_gone = rt::d3d12::shared_zero_images(device.p);
            EXPECT(one_gone.images == shapes);
            EXPECT(one_gone.bytes == placed.bytes);

            twin.reset();
            for (auto& texture : textures) texture.reset();
            const auto released = rt::d3d12::shared_zero_images(device.p);
            std::printf("all released: %llu shared images, %llu bytes\n",
                        static_cast<unsigned long long>(released.images),
                        static_cast<unsigned long long>(released.bytes));
            EXPECT(released.images == 0);
            EXPECT(released.bytes == 0);

            // The images go through the deferred queue: a submission that has finished, then the next one, which
            // drains what the finished one retired.
            EXPECT(submit(empty));
            EXPECT(submit(empty));
            std::printf("after two submissions: %u of the %u shared images destroyed\n", destroyed(), shared_made);
            EXPECT(destroyed() == shapes);

            // A shape seen before gets a shared image again.
            {
                std::lock_guard lock(hook_mutex);
                widths_seen.clear();
            }
            constexpr UINT again_width = first_width + 4;
            EXPECT(SUCCEEDED(rt::d3d12::create_buffer_at_host_address(device.p, at(shapes + 1),
                                                                      bc1(again_width, height), 0,
                                                                      IID_PPV_ARGS(&again.p))));
            const auto later = rt::d3d12::shared_zero_images(device.p);
            std::printf("a shape placed again: %s, %llu shared images\n",
                        shared(again.p) ? "shared" : "with memory of its own",
                        static_cast<unsigned long long>(later.images));
            EXPECT(shared(again.p));
            EXPECT(later.images == 1);
            EXPECT(later.bytes > 0 && later.bytes < placed.bytes);
            EXPECT(shared_made == shapes + 1);
            device->CreateShaderResourceView(again.p, nullptr, slot(2));

            // A copy into it gives it memory of its own. Its shape's image is released before the next submission,
            // with the texture still alive, and destroyed once that one has finished and another starts.
            constexpr UINT pitch = 256, rows = height / 4;  // BC1: 8 bytes a 4x4 block, 17 blocks a row
            const D3D12_HEAP_PROPERTIES upload_heap{D3D12_HEAP_TYPE_UPLOAD};
            D3D12_RESOURCE_DESC buffer{};
            buffer.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
            buffer.Width = pitch * rows;
            buffer.Height = buffer.DepthOrArraySize = buffer.MipLevels = 1;
            buffer.SampleDesc.Count = 1;
            buffer.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
            EXPECT(SUCCEEDED(device->CreateCommittedResource(&upload_heap, D3D12_HEAP_FLAG_NONE, &buffer,
                                                             D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                                             IID_PPV_ARGS(&upload.p))));
            copy_target.store(again.p);
            const bool copied = submit([&] {
                D3D12_TEXTURE_COPY_LOCATION to{}, from{};
                to.pResource = again.p;
                to.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
                from.pResource = upload.p;
                from.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
                from.PlacedFootprint.Footprint = {DXGI_FORMAT_BC1_UNORM, again_width, height, 1, pitch};
                list->CopyTextureRegion(&to, 0, 0, 0, &from, nullptr);
            });
            copy_target.store(nullptr);
            const bool other_ran = other_replay.joinable();
            if (other_ran) other_replay.join();
            EXPECT(copied);
            std::printf("after the copy: %s; the compute queue's replay begun during it %s the shared image\n",
                        shared(again.p) ? "shared" : "with memory of its own",
                        !other_ran ? "did not start, nor release" : released_during_copy ? "released" : "kept");
            EXPECT(!shared(again.p));
            EXPECT(other_ran);  // the copy's replay gave it memory of its own before it ended its command buffer
            EXPECT(!released_during_copy);
            // The compute queue's replay has ended; the image is released by the direct queue's next one.
            EXPECT(rt::d3d12::shared_zero_images(device.p).images == 1);
            EXPECT(submit(empty));
            const auto filled = rt::d3d12::shared_zero_images(device.p);
            std::printf("the texture with memory of its own, alive: %llu shared images, %llu bytes\n",
                        static_cast<unsigned long long>(filled.images), static_cast<unsigned long long>(filled.bytes));
            EXPECT(filled.images == 0);
            EXPECT(filled.bytes == 0);
            EXPECT(submit(empty));
            EXPECT(submit(empty));
            std::printf("then: %u of the %u shared images destroyed\n", destroyed(), shared_made);
            EXPECT(destroyed() == shapes + 1);
            again.reset();
            EXPECT(rt::d3d12::shared_zero_images(device.p).bytes == 0);

            // Shapes until one no longer fits; once one goes, the one that did not fit gets an image.
            std::vector<Ref<ID3D12Resource>> big(big_slots);
            unsigned fitted = 0, refused = big_slots;
            for (unsigned i = 0; i + 1 < big_slots; ++i) {
                EXPECT(SUCCEEDED(rt::d3d12::create_buffer_at_host_address(
                    device.p, big_at(i), bc1(big_width, big_height - 4 * i), 0, IID_PPV_ARGS(&big[i].p))));
                if (!shared(big[i].p)) {
                    refused = i;
                    break;
                }
                ++fitted;
            }
            const auto full = rt::d3d12::shared_zero_images(device.p);
            std::printf("%u large shapes fit (%llu bytes), the next %s\n", fitted,
                        static_cast<unsigned long long>(full.bytes), refused < big_slots ? "did not" : "fit too");
            EXPECT(fitted > 0 && refused < big_slots);
            EXPECT(full.images == fitted);
            big[refused].reset();
            big[0].reset();
            EXPECT(rt::d3d12::shared_zero_images(device.p).images == fitted - 1);
            EXPECT(SUCCEEDED(rt::d3d12::create_buffer_at_host_address(device.p, big_at(big_slots - 1),
                                                                      bc1(big_width, big_height - 4 * refused), 0,
                                                                      IID_PPV_ARGS(&big[big_slots - 1].p))));
            std::printf("the shape that did not fit, placed again: %s\n",
                        shared(big[big_slots - 1].p) ? "shared" : "with memory of its own");
            EXPECT(shared(big[big_slots - 1].p));
            big.clear();
            EXPECT(rt::d3d12::shared_zero_images(device.p).bytes == 0);
            EXPECT(submit(empty));
            EXPECT(submit(empty));
            {
                std::lock_guard lock(hook_mutex);
                std::printf("shared images destroyed with a view of them left: %u\n", destroyed_under_view);
                EXPECT(destroyed_under_view == 0);
            }
        }
    }
    VirtualFree(big_pages, 0, MEM_RELEASE);
    VirtualFree(pages, 0, MEM_RELEASE);
    if (vk.validation_errors()) {
        std::fprintf(stderr, "%u validation errors\n", vk.validation_errors());
        return 1;
    }
    if (!result) std::printf("ok\n");
    return result;
}
