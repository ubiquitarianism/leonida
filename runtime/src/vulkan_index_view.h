// Device creation policy for bounded D3D12 indexed-indirect draws. No work per draw.
#pragma once

#include <vulkan/vulkan.h>

namespace leonida {

inline bool can_enable_robust_index_view(VkPhysicalDevice device, PFN_vkGetPhysicalDeviceProperties2 properties,
                                         bool sized_indirect, bool robust_access, bool storage_update_after_bind) {
    if (!sized_indirect || !robust_access) return false;
    if (!storage_update_after_bind) return true;
    // VUID-VkDeviceCreateInfo-robustBufferAccess-10247: preserve bindless rather than disabling
    // descriptorBindingStorageBufferUpdateAfterBind on a device that cannot combine the two.
    if (!properties) return false;
    VkPhysicalDeviceVulkan12Properties indexed{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_PROPERTIES};
    VkPhysicalDeviceProperties2 query{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &indexed};
    properties(device, &query);
    return indexed.robustBufferAccessUpdateAfterBind == VK_TRUE;
}

} // namespace leonida
