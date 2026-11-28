#pragma once

// VK_EXT_memory_budget: how much GPU memory this process uses and may use, read from the Vulkan device itself. The
// device publishes it (memory_figures.h) for the title HUD (shims/title_hud.h, LEONIDA_TITLE_HUD) and the dev overlay
// (dev_overlay.h, LEONIDA_DEV_OVERLAY); DXGI's QueryVideoMemoryInfo, which the HUD read before, is 0 under Wine. The
// sums depend only on the two Vulkan structures, so a test can feed them made-up heaps.

#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
#include <vulkan/vulkan.h>

#include <cstdint>

#include "memory_figures.h"  // VramBudget

namespace leonida {

inline VramBudget sum_vram_budget(const VkPhysicalDeviceMemoryProperties& memory,
                                  const VkPhysicalDeviceMemoryBudgetPropertiesEXT& budget) {
    VramBudget out;
    for (std::uint32_t i = 0; i < memory.memoryHeapCount && i < VK_MAX_MEMORY_HEAPS; ++i) {
        if (memory.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
            out.local_usage += budget.heapUsage[i];
            out.local_budget += budget.heapBudget[i];
        } else {
            out.shared_usage += budget.heapUsage[i];
        }
    }
    out.available = out.local_budget != 0;
    return out;
}

// Reads the budget now. `physical` must have VK_EXT_memory_budget (enabled on the logical device, as the runtime's
// device does). One vkGetPhysicalDeviceMemoryProperties2 call, which needs no external synchronization, so another
// thread (the HUD's or the overlay's) may call it while the device is in use.
inline VramBudget query_vram_budget(PFN_vkGetPhysicalDeviceMemoryProperties2 get_memory_properties2,
                                    VkPhysicalDevice physical) {
    if (!get_memory_properties2 || physical == VK_NULL_HANDLE) return {};
    VkPhysicalDeviceMemoryBudgetPropertiesEXT budget{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT};
    VkPhysicalDeviceMemoryProperties2 properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2, &budget};
    get_memory_properties2(physical, &properties);
    return sum_vram_budget(properties.memoryProperties, budget);
}

}  // namespace leonida
