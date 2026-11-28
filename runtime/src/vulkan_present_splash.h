#pragma once

#include "vulkan_present.h"

namespace leonida {

inline bool VulkanPresenter::present_splash(ResourceId source, std::shared_ptr<void> owner) {
    std::lock_guard lock(present_mutex_);
    if (!deferred_.empty() || context_.holds().any_held()) return false;
    {
        std::shared_lock resources(context_.resource_mutex_);
        const auto* binding = context_.resolve(source);
        if (!binding || !binding->image) return false;
        splash_binding_ = *binding;
    }
    if (!capture_) splash_capture_only_ = enable_capture_storage();
    const auto before = presented();
    if (present(source, true) != VK_SUCCESS || presented() == before) return false;
    const Frame& frame = frames_[(frame_index_ - 1) % frames_.size()];
    if (vkWaitForFences_(context_.device(), 1, &frame.done, VK_TRUE, UINT64_MAX) != VK_SUCCESS ||
        finish_pending_capture() != VK_SUCCESS) return false;
    if (capture_ptr_) {
        splash_source_ = source;
        splash_owner_ = std::move(owner);
    }
    return true;
}

// A changed surface or colour format may not support readback. Keep the title able to present on it.
inline bool VulkanPresenter::recreate_for_present(bool vsync, bool srgb, bool hdr) {
    if (recreate(vsync, srgb, hdr)) return true;
    if (!splash_capture_only_) return false;
    splash_owner_.reset();
    splash_capture_only_ = false;
    capture_ = false;
    return recreate(vsync, srgb, hdr);
}

// Called only during the launch handoff, with the tracker and queue locked. Check the composed pixels so a
// transparent overlay or a clear with nonzero alpha cannot dismiss the splash. Later black frames are genuine.
inline VkResult VulkanPresenter::finish_startup_frame(Frame& frame, std::uint32_t image, bool& show_title) {
    const VkDevice device = context_.device();
    VkResult result = vkWaitForFences_(device, 1, &frame.done, VK_TRUE, UINT64_MAX);
    if (result != VK_SUCCESS) return result;
    const bool ten_bit = format_ == VK_FORMAT_A2B10G10R10_UNORM_PACK32 || format_ == VK_FORMAT_A2R10G10B10_UNORM_PACK32;
    const std::uint32_t rgb = ten_bit ? 0x3fffffffu : 0xffffffu;
    // Every 8th pixel of every 8th row: the capture is read back from host-visible memory, and reading all of it at
    // every present while the title's frames are black took 75-90 % of the present thread under Proton and doubled
    // the title's start-up (speed-3: 2 M pixels a present, under the submit lock). Content smaller than 8x8 pixels
    // keeps the splash a few frames longer.
    const auto* pixels = reinterpret_cast<const std::uint32_t*>(capture_ptr_);
    show_title = false;
    for (std::uint32_t y = 0; y < extent_.height && !show_title; y += 8) {
        const std::uint32_t* row = pixels + std::size_t(y) * extent_.width;
        for (std::uint32_t x = 0; x < extent_.width; x += 8)
            if (row[x] & rgb) {
                show_title = true;
                break;
            }
    }
    VkSubmitInfo ready{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    ready.signalSemaphoreCount = 1;
    ready.pSignalSemaphores = &ready_[image];
    if (show_title) return context_.queue_submit(queue_, 1, &ready, VK_NULL_HANDLE);
    // The acquire was consumed and the title commands finished. Overwrite the same image before presenting it.
    // The inspected copy contains the black title frame. Overwrite staging with the
    // splash actually shown; this is the copy an active disk/readback consumer must see.
    record(frame.commands, splash_binding_, splash_source_, image, nullptr, true);
    result = vkResetFences_(device, 1, &frame.done);
    if (result != VK_SUCCESS) return result;
    ready.commandBufferCount = 1;
    ready.pCommandBuffers = &frame.commands;
    return context_.queue_submit(queue_, 1, &ready, frame.done);
}

}  // namespace leonida
