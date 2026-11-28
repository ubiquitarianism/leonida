#pragma once

#include <vulkan/vulkan.h>
#include <algorithm>
#include <cstdint>
#include <memory>
#include <mutex>
#include <span>
#include <unordered_map>
#include <vector>

namespace leonida {
class GraphicsShaderModules {
  public:
    struct Module {
        VkDevice device;
        PFN_vkDestroyShaderModule destroy;
        std::vector<std::uint32_t> words;
        VkShaderModule handle = VK_NULL_HANDLE;
        Module(VkDevice d, PFN_vkDestroyShaderModule fn, std::span<const std::uint32_t> input)
            : device(d), destroy(fn), words(input.begin(), input.end()) {}
        ~Module() {
            if (handle) destroy(device, handle, nullptr);
        }
    };
    using Reference = std::shared_ptr<const Module>;
    Reference acquire(VkDevice device, PFN_vkCreateShaderModule create, PFN_vkDestroyShaderModule destroy,
                      std::span<const std::uint32_t> words) {
        if (words.empty()) return {};
        std::uint64_t hash = 14695981039346656037ull;
        for (std::uint32_t word : words)
            hash = (hash ^ word) * 1099511628211ull;
        {
            std::lock_guard lock(mutex_);
            if (auto found = find_locked(hash, words)) return found;
        }
        auto result = std::make_shared<Module>(device, destroy, words);
        VkShaderModuleCreateInfo info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        info.pCode = words.data();
        info.codeSize = words.size_bytes();
        if (create(device, &info, nullptr, &result->handle) != VK_SUCCESS) {
            result->handle = VK_NULL_HANDLE;
            return {};
        }
        std::lock_guard lock(mutex_);
        if (auto found = find_locked(hash, words)) return found;
        if ((++created_ % 1024) == 0) std::erase_if(modules_, [](const auto &entry) { return entry.second.expired(); });
        modules_.emplace(hash, result);
        return result;
    }

  private:
    Reference find_locked(std::uint64_t hash, std::span<const std::uint32_t> words) {
        const auto [first, last] = modules_.equal_range(hash);
        for (auto i = first; i != last; ++i)
            if (auto live = i->second.lock();
                live && std::equal(words.begin(), words.end(), live->words.begin(), live->words.end()))
                return live;
        return {};
    }
    std::mutex mutex_;
    std::unordered_multimap<std::uint64_t, std::weak_ptr<const Module>> modules_;
    std::uint64_t created_ = 0;
};
} // namespace leonida
