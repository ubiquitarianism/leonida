#pragma once

#include <vulkan/vulkan.h>

#include <algorithm>
#include <bit>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <vector>

namespace leonida {

// Only optimal-tiled, non-sparse images go in a page. A slot is returned after the owning
// image is destroyed by the device's normal GPU retirement queue.
struct TitleImagePage {
    VkDevice device{};
    VkDeviceMemory memory{};
    PFN_vkFreeMemory free_memory{};
    std::uint32_t type = 0, capacity = 0, allocated = 0;
    std::mutex mutex;
    std::vector<std::uint32_t> returned;

    ~TitleImagePage() {
        if (memory) free_memory(device, memory, nullptr);
    }

    bool take(std::uint32_t& slot) {
        std::lock_guard lock(mutex);
        if (!returned.empty()) {
            slot = returned.back();
            returned.pop_back();
        } else if (allocated < capacity) {
            slot = allocated++;
        } else {
            return false;
        }
        return true;
    }

    void release(std::uint32_t slot) {
        std::lock_guard lock(mutex);
        returned.push_back(slot);  // the page reserves capacity at creation
    }
};

struct PooledTitleImage {
    PooledTitleImage() = default;
    PooledTitleImage(const PooledTitleImage&) = delete;
    PooledTitleImage& operator=(const PooledTitleImage&) = delete;

    std::shared_ptr<TitleImagePage> page;
    std::uint32_t slot = 0;
    VkImage image{};
    PFN_vkDestroyImage destroy_image{};

    ~PooledTitleImage() {
        if (image) destroy_image(page->device, image, nullptr);
        if (page) page->release(slot);
    }
};

class TitleImagePool {
public:
    // The caller has already excluded dedicated-only images. All allocations in one page have
    // identical stride, compatible memory types and non-overlapping, individually aligned ranges.
    template<class Allocate>
    std::unique_ptr<PooledTitleImage> bind(VkDevice device, VkImage image, const VkMemoryRequirements& req,
                                           PFN_vkBindImageMemory bind_image, PFN_vkDestroyImage destroy_image,
                                           PFN_vkFreeMemory free_memory, Allocate allocate) {
        constexpr VkDeviceSize largest_slot = VkDeviceSize{2} << 20;
        const VkDeviceSize needed = std::max(req.size, req.alignment);
        if (!needed || needed > largest_slot || !req.memoryTypeBits) return {};
        // Eight size classes per power of two keep padding below one eighth (besides required alignment).
        const VkDeviceSize quantum = std::max(req.alignment, std::max(VkDeviceSize{1}, std::bit_floor(needed) / 8));
        const VkDeviceSize stride = (needed + quantum - 1) & ~(quantum - 1);
        const VkDeviceSize page_size = std::min(VkDeviceSize{8} << 20, 256 * stride);
        auto allocation = std::make_unique<PooledTitleImage>();
        std::lock_guard lock(mutex_);
        auto& pages = classes_[stride];
        std::erase_if(pages, [](const auto& page) { return page.expired(); });
        for (const auto& weak : pages) {
            auto page = weak.lock();
            if (page && (req.memoryTypeBits & (1u << page->type)) && page->take(allocation->slot)) {
                allocation->page = std::move(page);
                break;
            }
        }
        if (!allocation->page) {
            auto page = std::make_shared<TitleImagePage>();
            page->device = device;
            page->free_memory = free_memory;
            page->capacity = static_cast<std::uint32_t>(page_size / stride);
            page->returned.reserve(page->capacity);
            page->memory = allocate(page_size, page->type);
            if (!page->memory) return {};
            pages.push_back(page);
            page->take(allocation->slot);
            allocation->page = std::move(page);
        }
        if (bind_image(device, image, allocation->page->memory, stride * allocation->slot) != VK_SUCCESS) return {};
        allocation->destroy_image = destroy_image;
        allocation->image = image;
        return allocation;
    }

private:
    std::mutex mutex_;
    std::map<VkDeviceSize, std::vector<std::weak_ptr<TitleImagePage>>> classes_;
};

}  // namespace leonida
