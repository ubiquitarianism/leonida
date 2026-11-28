// The context's deferred deletion queue after a device loss (runtime/src/vulkan_backend.h,
// VulkanContext::defer_destroy). Once the device is lost no retire ticket completes again and acquire() returns before
// its drain, so an object queued then would stay until the context goes, and the queue would grow with every object
// the title releases.
//   1. Buffers deferred before the loss wait for a submission that never comes.
//   2. After the loss, the next deferred destruction frees them and every later one is freed at once: all 1000 buffers
//      are gone while the context lives, and the context's own teardown destroys none of them twice.
//   3. A bindless slot and the image view released with it (DeviceState::release in d3d12_vulkan_device.cpp defers
//      the slot, then its views), queued before the loss or after it, both wait in the queue. Releasing the slot
//      takes the device's descriptor lock, which the caller holds; and until the slot is released the set's pending
//      writes and shadow copy may still name the view, which a flush hands to vkUpdateDescriptorSets. At teardown the
//      slot is released first, then its view is destroyed.
//   4. A batch that takes the tracker lock after the loss (another queue's vkQueueSubmit found it while this one
//      waited for that lock) stops before it flushes the bindless set or records its lists: the loss is marked from
//      the interposed vkBeginCommandBuffer, after the batch's first check of it.
// vkDestroyBuffer, vkDestroyImageView and vkBeginCommandBuffer are interposed on a real device's table; only the
// buffers and views the test made are counted. Skips with exit code 77 when no Vulkan 1.3 device is available;
// LEONIDA_VK_VALIDATION=1 runs it under the validation layer. No game data.
#include "vk_test_device.h"
#include "../src/vulkan_backend.h"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <set>
#include <string_view>
#include <vector>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); std::_Exit(1); } \
} while (0)

namespace {

template<typename Handle>
std::uint64_t key(Handle handle) {
    return static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(handle));
}

namespace driver {
PFN_vkGetDeviceProcAddr real_get_device = nullptr;
PFN_vkDestroyBuffer real_destroy_buffer = nullptr;
PFN_vkDestroyImageView real_destroy_view = nullptr;
PFN_vkBeginCommandBuffer real_begin = nullptr;

std::mutex mutex;
std::set<std::uint64_t> ours;        // the buffers and views the test made
unsigned destroyed_buffers = 0;      // of ours
unsigned destroyed_views = 0;        // of ours
std::vector<std::uint64_t> events;   // bindless slot releases and destructions of our views, in order
std::atomic<leonida::VulkanContext*> lose_at_begin{nullptr};

void adopt(std::uint64_t handle) {
    std::lock_guard lock(mutex);
    ours.insert(handle);
}

unsigned buffers() {
    std::lock_guard lock(mutex);
    return destroyed_buffers;
}

unsigned views() {
    std::lock_guard lock(mutex);
    return destroyed_views;
}

VKAPI_ATTR void VKAPI_CALL destroy_buffer(VkDevice device, VkBuffer buffer, const VkAllocationCallbacks* allocator) {
    {
        std::lock_guard lock(mutex);
        if (ours.erase(key(buffer))) ++destroyed_buffers;
    }
    real_destroy_buffer(device, buffer, allocator);
}

VKAPI_ATTR void VKAPI_CALL destroy_view(VkDevice device, VkImageView view, const VkAllocationCallbacks* allocator) {
    {
        std::lock_guard lock(mutex);
        if (ours.erase(key(view))) {
            ++destroyed_views;
            events.push_back(key(view));
        }
    }
    real_destroy_view(device, view, allocator);
}

// As another queue's vkQueueSubmit returning VK_ERROR_DEVICE_LOST would, while this batch is past its own checks.
VKAPI_ATTR VkResult VKAPI_CALL begin(VkCommandBuffer command_buffer, const VkCommandBufferBeginInfo* info) {
    const VkResult result = real_begin(command_buffer, info);
    if (auto* context = lose_at_begin.exchange(nullptr)) context->mark_device_lost("the test marks the device lost");
    return result;
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL get_device(VkDevice device, const char* name) {
    const std::string_view n(name);
    if (n == "vkDestroyBuffer") return reinterpret_cast<PFN_vkVoidFunction>(destroy_buffer);
    if (n == "vkDestroyImageView") return reinterpret_cast<PFN_vkVoidFunction>(destroy_view);
    if (n == "vkBeginCommandBuffer") return reinterpret_cast<PFN_vkVoidFunction>(begin);
    return real_get_device(device, name);
}

void install(PFN_vkGetDeviceProcAddr device_proc, VkDevice device) {
    real_get_device = device_proc;
    real_destroy_buffer = reinterpret_cast<PFN_vkDestroyBuffer>(device_proc(device, "vkDestroyBuffer"));
    real_destroy_view = reinterpret_cast<PFN_vkDestroyImageView>(device_proc(device, "vkDestroyImageView"));
    real_begin = reinterpret_cast<PFN_vkBeginCommandBuffer>(device_proc(device, "vkBeginCommandBuffer"));
}
}  // namespace driver

}  // namespace

int main() {
    using namespace leonida;

    vktest::Device d;
    if (!d.create(false)) {
        std::fprintf(stderr, "vk_deferred_destroy_lost_test: no Vulkan 1.3 device, skipped\n");
        return 77;
    }
    driver::install(d.device_proc, d.device);
    const auto create_buffer = d.fn<PFN_vkCreateBuffer>("vkCreateBuffer");
    const auto create_view = d.fn<PFN_vkCreateImageView>("vkCreateImageView");
    EXPECT(create_buffer && create_view);
    const auto make_buffer = [&] {
        VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        info.size = 256;
        info.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VkBuffer buffer = VK_NULL_HANDLE;
        EXPECT(create_buffer(d.device, &info, nullptr, &buffer) == VK_SUCCESS && buffer);
        driver::adopt(key(buffer));
        return buffer;
    };
    const VkImage image = d.image(VK_FORMAT_R8G8B8A8_UNORM, 4, 4, 1, VK_IMAGE_USAGE_SAMPLED_BIT);  // d destroys it
    EXPECT(image);
    const auto make_view = [&] {
        VkImageViewCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        info.image = image;
        info.viewType = VK_IMAGE_VIEW_TYPE_2D;
        info.format = VK_FORMAT_R8G8B8A8_UNORM;
        info.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        VkImageView view = VK_NULL_HANDLE;
        EXPECT(create_view(d.device, &info, nullptr, &view) == VK_SUCCESS && view);
        driver::adopt(key(view));
        return view;
    };

    constexpr unsigned before_loss = 500;
    constexpr unsigned total = 1000;
    constexpr std::uint64_t slot_before = (std::uint64_t(1) << 32) | 3u;
    constexpr std::uint64_t slot_after = (std::uint64_t(1) << 32) | 7u;
    const VkImageView view_before = make_view(), view_after = make_view();
    {
        VulkanContext context(driver::get_device, d.device, true);
        EXPECT(context.loaded());
        context.set_bindless_release(
            [](void*, std::uint32_t dimension, std::uint32_t index) {
                std::lock_guard lock(driver::mutex);
                driver::events.push_back((std::uint64_t(dimension) << 32) | index);
            },
            nullptr);

        // 1, with a bindless slot and its view among the buffers
        for (unsigned i = 0; i < before_loss; ++i) {
            context.defer_destroy(DeferredDeletionQueue::Kind::buffer, make_buffer());
            if (i == before_loss / 2) {
                context.defer_destroy(DeferredDeletionQueue::Kind::bindless_slot, slot_before);
                context.defer_destroy(DeferredDeletionQueue::Kind::image_view, view_before);
            }
        }
        EXPECT(driver::buffers() == 0);
        EXPECT(driver::views() == 0);

        // 2
        context.mark_device_lost("the test marks the device lost");
        for (unsigned i = before_loss; i < total; ++i)
            context.defer_destroy(DeferredDeletionQueue::Kind::buffer, make_buffer());
        EXPECT(driver::buffers() == total);
        EXPECT(driver::views() == 0);  // the view queued before the loss still waits behind its slot

        // 3
        context.defer_destroy(DeferredDeletionQueue::Kind::bindless_slot, slot_after);
        context.defer_destroy(DeferredDeletionQueue::Kind::image_view, view_after);
        context.defer_destroy(DeferredDeletionQueue::Kind::buffer, make_buffer());  // still freed at once
        EXPECT(driver::buffers() == total + 1);
        EXPECT(driver::views() == 0);
        std::lock_guard lock(driver::mutex);
        EXPECT(driver::events.empty());
    }
    EXPECT(driver::buffers() == total + 1);
    EXPECT(driver::views() == 2);
    {
        std::lock_guard lock(driver::mutex);
        const std::vector<std::uint64_t> expected{slot_before, key(view_before), slot_after, key(view_after)};
        EXPECT(driver::events == expected);
    }
    std::puts("deferred deletion after a loss: buffers freed at once; bindless slots and their views kept, in order");

    // 4
    {
        VulkanContext context(driver::get_device, d.device, true);
        EXPECT(context.loaded());
        unsigned prepared = 0;
        context.set_bindless_prepare([&](VkCommandBuffer) {
            ++prepared;
            return true;
        });
        constexpr VkDeviceSize size = 4096;
        const VkBufferUsageFlags usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        void* map = nullptr;
        const VkBuffer a = d.buffer(size, usage, &map), b = d.buffer(size, usage, &map);  // d destroys them
        EXPECT(a && b);
        ResourceBinding binding;
        binding.size = size;
        binding.buffer = a;
        context.add_resource(1, binding);
        binding.buffer = b;
        context.add_resource(2, binding);
        CommandList copy;
        EXPECT(copy.copy_buffer({2, 0, 1, 0, size}));
        EXPECT(copy.close());
        const CommandList* lists[] = {&copy};
        auto queue = std::make_unique<VulkanQueue>(context, d.queue, d.family);
        EXPECT(queue->ok());
        EXPECT(queue->execute(lists) == ReplayResult::success);
        EXPECT(prepared == 1);  // a batch on a live device flushes the set

        driver::lose_at_begin = &context;
        EXPECT(queue->execute(lists) == ReplayResult::unsupported);
        EXPECT(driver::lose_at_begin.load() == nullptr);  // the interposer marked the loss, past the first check
        EXPECT(context.is_device_lost());
        EXPECT(prepared == 1);
        queue.reset();
        context.remove_resource(1);
        context.remove_resource(2);
    }
    std::puts("a batch that takes the tracker lock after the loss stops before the bindless flush");

    EXPECT(d.validation_errors() == 0);
    std::puts("vk_deferred_destroy_lost: ok");
    return 0;
}
