// Title texture uploads must keep a queue submission from racing the host copy.
#include "../src/vulkan_backend.h"

#include <cstdio>
#include <thread>

namespace {
unsigned waits = 0;
VkResult wait_result = VK_SUCCESS;
std::uint64_t last_timeout = 0;
// The writes wait for the queue timelines (vkWaitSemaphores, bounded), not for the device.
VkResult VKAPI_CALL wait_semaphores(VkDevice, const VkSemaphoreWaitInfo* info, std::uint64_t timeout) {
    ++waits;
    last_timeout = timeout;
    return info && info->semaphoreCount ? wait_result : VK_ERROR_UNKNOWN;
}
VkResult VKAPI_CALL device_idle(VkDevice) { return VK_ERROR_UNKNOWN; }  // must not be called any more
PFN_vkVoidFunction VKAPI_CALL no_proc(VkDevice, const char*) { return nullptr; }
}

int main() {
    leonida::VulkanContext ctx(no_proc, VK_NULL_HANDLE, false);
    ctx.vk.vkWaitSemaphores = wait_semaphores;
    ctx.vk.vkDeviceWaitIdle = device_idle;
    leonida::ResourceBinding binding{};
    binding.image = reinterpret_cast<VkImage>(1);
    binding.mip_levels = 1;
    ctx.add_resource(1, binding, VK_IMAGE_LAYOUT_GENERAL);
    ctx.add_resource(2, binding, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    unsigned copies = 0;
    const auto copy = [&] {
        ++copies;
        bool raced = false;
        std::thread contender([&] {
            if (ctx.submit_mutex().try_lock()) {
                raced = true;
                ctx.submit_mutex().unlock();
            }
        });
        contender.join();
        return !raced;
    };
    const auto progress = reinterpret_cast<VkSemaphore>(std::uintptr_t{0x10});
    const auto submit = [&] {  // a queue submission in flight: a retirement ticket, and image work pending
        std::lock_guard lock(ctx.submit_mutex());
        ctx.record_submit_locked(progress, 1);
    };
    submit();
    if (ctx.write_host_texture(2, 0, copy) != VK_NOT_READY || copies || waits) return 1;  // wrong layout: no wait
    if (ctx.write_host_texture(1, 0, copy) != VK_SUCCESS || copies != 1 || waits != 1) return 2;
    if (last_timeout == 0 || last_timeout == UINT64_MAX) return 3;  // bounded
    if (ctx.write_host_texture(1, 0, copy) != VK_SUCCESS || copies != 2 || waits != 1) return 4;  // wait reused
    submit();
    wait_result = VK_TIMEOUT;  // the submitted work is not done in time: skipped as not ready, not a failure
    if (ctx.write_host_texture(1, 0, copy) != VK_NOT_READY || copies != 2 || waits != 2) return 5;
    wait_result = VK_ERROR_DEVICE_LOST;
    if (ctx.write_host_texture(1, 0, copy) != VK_ERROR_DEVICE_LOST || copies != 2 || waits != 3) return 6;
    const auto stats = ctx.host_texture_stats();
    if (stats.copies != 2 || stats.waits != 3) return 7;
    std::puts("ordered texture upload: layout, submit exclusion, bounded timeline wait, reuse, timeout, loss passed");
}
