// The kind bits BindlessSet gives an image index (runtime/src/vulkan_bindless.h; descriptor_table.h's kind bits),
// which a tagged module (descriptor_table: tagged) trusts to index the arrays they name:
//  - a storage bit only where a storage descriptor was written, so never at an index past the storage arrays (an
//    image added without storage views can get one, and update_image may give it storage views later);
//  - no sampled bit for a 1D-array image, as that dimension has no sampled array.
// From kevincommits's finding in #1608 (RT-BINDLESS-KIND-BITS). No Vulkan loader, GPU, or game data: a fake device
// gives the set its handles and memory.
#include "../src/vulkan_bindless.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>

namespace {

std::uintptr_t next_handle = 0x1000;
template<typename T>
T make_handle() {
    return reinterpret_cast<T>(next_handle += 0x10);
}

// Mapped memory for the table buffers: each allocation gets its bytes.
std::vector<std::unique_ptr<std::uint8_t[]>> allocations;

template<typename T>
VKAPI_ATTR VkResult VKAPI_CALL create(VkDevice, const void*, const VkAllocationCallbacks*, T* out) {
    *out = make_handle<T>();
    return VK_SUCCESS;
}
template<typename T>
VKAPI_ATTR void VKAPI_CALL destroy(VkDevice, T, const VkAllocationCallbacks*) {}

VKAPI_ATTR VkResult VKAPI_CALL reset_pool(VkDevice, VkDescriptorPool, VkDescriptorPoolResetFlags) { return VK_SUCCESS; }
VKAPI_ATTR VkResult VKAPI_CALL allocate_sets(VkDevice, const VkDescriptorSetAllocateInfo* info, VkDescriptorSet* sets) {
    for (std::uint32_t i = 0; i < info->descriptorSetCount; ++i) sets[i] = make_handle<VkDescriptorSet>();
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL update_sets(VkDevice, std::uint32_t, const VkWriteDescriptorSet*, std::uint32_t,
                                       const VkCopyDescriptorSet*) {}
VKAPI_ATTR void VKAPI_CALL buffer_requirements(VkDevice, VkBuffer, VkMemoryRequirements* r) {
    *r = {1u << 20, 256, 1};
}
VKAPI_ATTR void VKAPI_CALL image_requirements(VkDevice, VkImage, VkMemoryRequirements* r) { *r = {4096, 256, 1}; }
VKAPI_ATTR VkResult VKAPI_CALL bind_buffer(VkDevice, VkBuffer, VkDeviceMemory, VkDeviceSize) { return VK_SUCCESS; }
VKAPI_ATTR VkResult VKAPI_CALL bind_image(VkDevice, VkImage, VkDeviceMemory, VkDeviceSize) { return VK_SUCCESS; }
VKAPI_ATTR VkResult VKAPI_CALL allocate_memory(VkDevice, const VkMemoryAllocateInfo* info,
                                               const VkAllocationCallbacks*, VkDeviceMemory* memory) {
    allocations.push_back(std::make_unique<std::uint8_t[]>(std::size_t(info->allocationSize)));
    *memory = reinterpret_cast<VkDeviceMemory>(allocations.back().get());
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL free_memory(VkDevice, VkDeviceMemory, const VkAllocationCallbacks*) {}
VKAPI_ATTR VkResult VKAPI_CALL map_memory(VkDevice, VkDeviceMemory memory, VkDeviceSize offset, VkDeviceSize,
                                          VkMemoryMapFlags, void** data) {
    *data = reinterpret_cast<std::uint8_t*>(memory) + offset;
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL cmd_barrier(VkCommandBuffer, VkPipelineStageFlags, VkPipelineStageFlags, VkDependencyFlags,
                                       std::uint32_t, const VkMemoryBarrier*, std::uint32_t,
                                       const VkBufferMemoryBarrier*, std::uint32_t, const VkImageMemoryBarrier*) {}
VKAPI_ATTR void VKAPI_CALL cmd_copy(VkCommandBuffer, VkBuffer, VkBuffer, std::uint32_t, const VkBufferCopy*) {}
VKAPI_ATTR void VKAPI_CALL cmd_update(VkCommandBuffer, VkBuffer, VkDeviceSize, VkDeviceSize, const void*) {}
VKAPI_ATTR void VKAPI_CALL cmd_clear(VkCommandBuffer, VkImage, VkImageLayout, const VkClearColorValue*, std::uint32_t,
                                     const VkImageSubresourceRange*) {}

#define FAKE(name, fn)                                                                                  \
    if (!std::strcmp(n, #name)) return reinterpret_cast<PFN_vkVoidFunction>(static_cast<PFN_##name>(fn));
VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL get_proc(VkDevice, const char* n) {
    FAKE(vkCreateDescriptorSetLayout, reinterpret_cast<PFN_vkCreateDescriptorSetLayout>(&create<VkDescriptorSetLayout>))
    FAKE(vkDestroyDescriptorSetLayout, &destroy<VkDescriptorSetLayout>)
    FAKE(vkCreateDescriptorPool, reinterpret_cast<PFN_vkCreateDescriptorPool>(&create<VkDescriptorPool>))
    FAKE(vkDestroyDescriptorPool, &destroy<VkDescriptorPool>)
    FAKE(vkResetDescriptorPool, &reset_pool)
    FAKE(vkAllocateDescriptorSets, &allocate_sets)
    FAKE(vkUpdateDescriptorSets, &update_sets)
    FAKE(vkCreateBuffer, reinterpret_cast<PFN_vkCreateBuffer>(&create<VkBuffer>))
    FAKE(vkDestroyBuffer, &destroy<VkBuffer>)
    FAKE(vkGetBufferMemoryRequirements, &buffer_requirements)
    FAKE(vkBindBufferMemory, &bind_buffer)
    FAKE(vkAllocateMemory, &allocate_memory)
    FAKE(vkFreeMemory, &free_memory)
    FAKE(vkMapMemory, &map_memory)
    FAKE(vkCreateImage, reinterpret_cast<PFN_vkCreateImage>(&create<VkImage>))
    FAKE(vkDestroyImage, &destroy<VkImage>)
    FAKE(vkGetImageMemoryRequirements, &image_requirements)
    FAKE(vkBindImageMemory, &bind_image)
    FAKE(vkCreateImageView, reinterpret_cast<PFN_vkCreateImageView>(&create<VkImageView>))
    FAKE(vkDestroyImageView, &destroy<VkImageView>)
    FAKE(vkCreateSampler, reinterpret_cast<PFN_vkCreateSampler>(&create<VkSampler>))
    FAKE(vkDestroySampler, &destroy<VkSampler>)
    FAKE(vkCmdPipelineBarrier, &cmd_barrier)
    FAKE(vkCmdCopyBuffer, &cmd_copy)
    FAKE(vkCmdUpdateBuffer, &cmd_update)
    FAKE(vkCmdClearColorImage, &cmd_clear)
    return nullptr;
}
#undef FAKE

int failures = 0;
void expect(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}

VkImageView view() { return make_handle<VkImageView>(); }

}  // namespace

int main() {
    using namespace leonida;
    using D = ImageDimension;
    const auto sampled_bit = [](D d) { return kind_sampled_bit(std::uint32_t(d)); };
    const auto storage_bit = [](D d) { return kind_storage_bit(std::uint32_t(d)); };

    VkPhysicalDeviceMemoryProperties memory{};
    memory.memoryTypeCount = 1;
    memory.memoryTypes[0].propertyFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    memory.memoryHeapCount = 1;
    memory.memoryHeaps[0].size = 1ull << 32;
    BindlessSet set(get_proc, make_handle<VkDevice>(), memory);  // the default capacities
    if (!set.ok()) {
        std::fprintf(stderr, "FAIL: the set was not made on the fake device\n");
        return 1;
    }
    const std::uint32_t storage_2d = set.storage_capacity(D::dim_2d);
    expect(storage_2d < set.image_capacity(D::dim_2d), "2D: the storage arrays are shorter than the sampled one");

    // 1. A 2D sampled-only image gets the first index past the storage arrays; storage views given to it later are
    //    not written there, so its kind has no storage bit.
    {
        std::uint32_t kind = 0;
        const std::uint32_t index =
            set.add_image(D::dim_2d, view(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, {}, 0, nullptr, &kind);
        expect(index == storage_2d, "2D sampled-only: the first index past the storage arrays");
        expect(kind == sampled_bit(D::dim_2d), "2D sampled-only: sampled bit only");
        StorageViews storage;
        storage.as_float = view();
        set.update_image(D::dim_2d, index, view(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, storage, 0, nullptr,
                         &kind);
        expect(kind == sampled_bit(D::dim_2d), "2D past the storage arrays, updated with a storage view: no storage bit");
        expect(set.image_kind_at(D::dim_2d, index) == kind, "image_kind_at agrees");
        // Inside the storage arrays an update does add the bit.
        std::uint32_t low_kind = 0;
        const std::uint32_t low = set.add_image(D::dim_2d, view(), VK_IMAGE_LAYOUT_GENERAL, storage, 0, nullptr,
                                                &low_kind);
        expect(low && low < storage_2d && low_kind == (sampled_bit(D::dim_2d) | storage_bit(D::dim_2d)),
               "2D with a storage view: inside the storage arrays, both bits");
        set.update_image(D::dim_2d, low, view(), VK_IMAGE_LAYOUT_GENERAL, {}, 0, nullptr, &low_kind);
        expect(low_kind == sampled_bit(D::dim_2d), "2D updated without storage views: the storage bit goes");
        set.update_image(D::dim_2d, low, view(), VK_IMAGE_LAYOUT_GENERAL, storage, 0, nullptr, &low_kind);
        expect(low_kind == (sampled_bit(D::dim_2d) | storage_bit(D::dim_2d)),
               "2D updated with a storage view inside the storage arrays: the storage bit comes back");
    }

    // 2. A mirrored 2D-array sampled-only image past both storage arrays, updated with storage views for itself and
    //    its 2D mirror: no storage bit for either.
    {
        BindlessSet::MirrorViews mirror;
        mirror.sampled = view();
        mirror.sampled_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        std::uint32_t kind = 0;
        const std::uint32_t index =
            set.add_image(D::dim_2d_array, view(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, {}, 0, &mirror, &kind);
        expect(index >= set.storage_capacity(D::dim_2d_array) && index >= storage_2d,
               "2D array sampled-only, mirrored: past both storage arrays");
        expect(kind == (sampled_bit(D::dim_2d_array) | sampled_bit(D::dim_2d)), "mirrored: both sampled bits");
        StorageViews storage;
        storage.as_float = view();
        mirror.storage.as_float = view();
        set.update_image(D::dim_2d_array, index, view(), VK_IMAGE_LAYOUT_GENERAL, storage, 0, &mirror, &kind);
        expect(kind == (sampled_bit(D::dim_2d_array) | sampled_bit(D::dim_2d)),
               "mirrored, past the storage arrays, updated with storage views: no storage bits");
        expect(set.image_kind_at(D::dim_2d_array, index) == kind, "image_kind_at agrees (mirrored)");
    }

    // 3. A storage-only 1D-array image: the dimension has no sampled array, so no sampled bit.
    {
        expect(bindless::sampled_binding(D::dim_1d_array) == bindless::no_binding, "1D array: no sampled binding");
        StorageViews storage;
        storage.as_float = view();
        std::uint32_t kind = 0;
        const std::uint32_t index = set.add_image(D::dim_1d_array, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_GENERAL, storage, 0,
                                                  nullptr, &kind);
        expect(index == 1, "1D array storage-only: index 1");
        expect(kind == storage_bit(D::dim_1d_array), "1D array storage-only: storage bit only (add_image)");
        expect(set.image_kind_at(D::dim_1d_array, index) == storage_bit(D::dim_1d_array),
               "1D array storage-only: storage bit only (image_kind_at)");
        set.update_image(D::dim_1d_array, index, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_GENERAL, storage, 0, nullptr, &kind);
        expect(kind == storage_bit(D::dim_1d_array), "1D array storage-only: storage bit only (update_image)");
    }

    if (failures) {
        std::fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    std::printf("bindless kind bits: storage only inside the storage arrays, no sampled bit for 1D arrays\n");
    return 0;
}
