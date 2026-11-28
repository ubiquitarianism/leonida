// The depth24 converter's scratch chunks (src/vulkan_depth24.h) are reused by copies on every D3D12 queue, direct,
// compute and copy, which may be three Vulkan queue families: with two or more families a chunk is CONCURRENT over
// all of them, as the device's other shared pools are (ScratchBuffers, IndirectArgumentClamp, BindlessSet); with one
// family, or none given, it stays EXCLUSIVE. On the CPU: the driver is faked and the chunk's VkBufferCreateInfo is
// logged. No GPU, no game data.
#include "../src/vulkan_depth24.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <type_traits>
#include <vector>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

using namespace leonida;

namespace {

template<typename Handle>
Handle fake_handle(std::uintptr_t value) {
    if constexpr (std::is_pointer_v<Handle>) return reinterpret_cast<Handle>(value);
    else return Handle(value);
}

struct Created {
    VkDeviceSize size;
    VkSharingMode sharing;
    std::vector<std::uint32_t> families;
};
std::vector<Created> buffers;
std::uintptr_t next_handle = 0x1000;

VKAPI_ATTR VkResult VKAPI_CALL create_buffer(VkDevice, const VkBufferCreateInfo* info, const VkAllocationCallbacks*,
                                             VkBuffer* buffer) {
    Created c{info->size, info->sharingMode, {}};
    if (info->sharingMode == VK_SHARING_MODE_CONCURRENT)
        c.families.assign(info->pQueueFamilyIndices, info->pQueueFamilyIndices + info->queueFamilyIndexCount);
    buffers.push_back(c);
    *buffer = fake_handle<VkBuffer>(next_handle++);
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL destroy_buffer(VkDevice, VkBuffer, const VkAllocationCallbacks*) {}
VKAPI_ATTR void VKAPI_CALL requirements(VkDevice, VkBuffer, VkMemoryRequirements* out) { *out = {4u << 20, 256, 1}; }
VKAPI_ATTR VkResult VKAPI_CALL bind(VkDevice, VkBuffer, VkDeviceMemory, VkDeviceSize) { return VK_SUCCESS; }
VKAPI_ATTR void VKAPI_CALL free_memory(VkDevice, VkDeviceMemory, const VkAllocationCallbacks*) {}
VKAPI_ATTR VkResult VKAPI_CALL create_module(VkDevice, const VkShaderModuleCreateInfo*, const VkAllocationCallbacks*,
                                             VkShaderModule* out) {
    *out = fake_handle<VkShaderModule>(next_handle++);
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL destroy_module(VkDevice, VkShaderModule, const VkAllocationCallbacks*) {}
VKAPI_ATTR VkResult VKAPI_CALL create_set_layout(VkDevice, const VkDescriptorSetLayoutCreateInfo*,
                                                 const VkAllocationCallbacks*, VkDescriptorSetLayout* out) {
    *out = fake_handle<VkDescriptorSetLayout>(next_handle++);
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL destroy_set_layout(VkDevice, VkDescriptorSetLayout, const VkAllocationCallbacks*) {}
VKAPI_ATTR VkResult VKAPI_CALL create_pipeline_layout(VkDevice, const VkPipelineLayoutCreateInfo*,
                                                      const VkAllocationCallbacks*, VkPipelineLayout* out) {
    *out = fake_handle<VkPipelineLayout>(next_handle++);
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL destroy_pipeline_layout(VkDevice, VkPipelineLayout, const VkAllocationCallbacks*) {}
VKAPI_ATTR VkResult VKAPI_CALL create_pipelines(VkDevice, VkPipelineCache, std::uint32_t count,
                                                const VkComputePipelineCreateInfo*, const VkAllocationCallbacks*,
                                                VkPipeline* out) {
    for (std::uint32_t i = 0; i < count; ++i) out[i] = fake_handle<VkPipeline>(next_handle++);
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL destroy_pipeline(VkDevice, VkPipeline, const VkAllocationCallbacks*) {}
VKAPI_ATTR VkResult VKAPI_CALL create_pool(VkDevice, const VkDescriptorPoolCreateInfo*, const VkAllocationCallbacks*,
                                           VkDescriptorPool* out) {
    *out = fake_handle<VkDescriptorPool>(next_handle++);
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL destroy_pool(VkDevice, VkDescriptorPool, const VkAllocationCallbacks*) {}
VKAPI_ATTR VkResult VKAPI_CALL allocate_sets(VkDevice, const VkDescriptorSetAllocateInfo* info, VkDescriptorSet* out) {
    for (std::uint32_t i = 0; i < info->descriptorSetCount; ++i) out[i] = fake_handle<VkDescriptorSet>(next_handle++);
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL update_sets(VkDevice, std::uint32_t, const VkWriteDescriptorSet*, std::uint32_t,
                                       const VkCopyDescriptorSet*) {}
VKAPI_ATTR void VKAPI_CALL bind_pipeline(VkCommandBuffer, VkPipelineBindPoint, VkPipeline) {}
VKAPI_ATTR void VKAPI_CALL bind_sets(VkCommandBuffer, VkPipelineBindPoint, VkPipelineLayout, std::uint32_t,
                                     std::uint32_t, const VkDescriptorSet*, std::uint32_t, const std::uint32_t*) {}
VKAPI_ATTR void VKAPI_CALL push_constants(VkCommandBuffer, VkPipelineLayout, VkShaderStageFlags, std::uint32_t,
                                          std::uint32_t, const void*) {}
VKAPI_ATTR void VKAPI_CALL dispatch(VkCommandBuffer, std::uint32_t, std::uint32_t, std::uint32_t) {}
VKAPI_ATTR VkResult VKAPI_CALL counter_value(VkDevice, VkSemaphore, std::uint64_t* value) {
    *value = ~std::uint64_t{0};
    return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL wait_idle(VkDevice) { return VK_SUCCESS; }

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL get_proc(VkDevice, const char* name) {
#define LEONIDA_FAKE(entry, fn) if (!std::strcmp(name, entry)) return reinterpret_cast<PFN_vkVoidFunction>(fn)
    LEONIDA_FAKE("vkCreateBuffer", create_buffer);
    LEONIDA_FAKE("vkDestroyBuffer", destroy_buffer);
    LEONIDA_FAKE("vkGetBufferMemoryRequirements", requirements);
    LEONIDA_FAKE("vkBindBufferMemory", bind);
    LEONIDA_FAKE("vkFreeMemory", free_memory);
    LEONIDA_FAKE("vkCreateShaderModule", create_module);
    LEONIDA_FAKE("vkDestroyShaderModule", destroy_module);
    LEONIDA_FAKE("vkCreateDescriptorSetLayout", create_set_layout);
    LEONIDA_FAKE("vkDestroyDescriptorSetLayout", destroy_set_layout);
    LEONIDA_FAKE("vkCreatePipelineLayout", create_pipeline_layout);
    LEONIDA_FAKE("vkDestroyPipelineLayout", destroy_pipeline_layout);
    LEONIDA_FAKE("vkCreateComputePipelines", create_pipelines);
    LEONIDA_FAKE("vkDestroyPipeline", destroy_pipeline);
    LEONIDA_FAKE("vkCreateDescriptorPool", create_pool);
    LEONIDA_FAKE("vkDestroyDescriptorPool", destroy_pool);
    LEONIDA_FAKE("vkAllocateDescriptorSets", allocate_sets);
    LEONIDA_FAKE("vkUpdateDescriptorSets", update_sets);
    LEONIDA_FAKE("vkCmdBindPipeline", bind_pipeline);
    LEONIDA_FAKE("vkCmdBindDescriptorSets", bind_sets);
    LEONIDA_FAKE("vkCmdPushConstants", push_constants);
    LEONIDA_FAKE("vkCmdDispatch", dispatch);
    LEONIDA_FAKE("vkGetSemaphoreCounterValue", counter_value);
    LEONIDA_FAKE("vkDeviceWaitIdle", wait_idle);
#undef LEONIDA_FAKE
    return nullptr;
}

// A readback of a 64x64 D24 footprint: prepare() makes one scratch chunk for it.
Depth24Job job() {
    Depth24Job j;
    j.app = fake_handle<VkBuffer>(0x10);
    j.app_size = 64 * 64 * 4;
    j.width = j.height = 64;
    j.slices = 1;
    j.app_row_words = 64;
    j.app_slice_words = 64 * 64;
    return j;
}

// The chunk prepare() makes for one copy with the converter given `families`.
int chunk_sharing(std::vector<std::uint32_t> families, VkSharingMode mode, std::vector<std::uint32_t> expect) {
    buffers.clear();
    Depth24Converter converter(get_proc, fake_handle<VkDevice>(0x1),
                               [](const VkMemoryRequirements&) { return fake_handle<VkDeviceMemory>(0x20); },
                               1u << 27, nullptr, std::move(families));
    EXPECT(converter.loaded());
    Depth24Job whole = job();
    std::vector<Depth24Job> parts;
    EXPECT(converter.prepare(whole, parts) && whole.scratch && parts.size() == 1);
    EXPECT(buffers.size() == 1);
    EXPECT(buffers[0].sharing == mode && buffers[0].families == expect);
    return 0;
}

}  // namespace

int main() {
    // Graphics, async compute and DMA families: the chunk is shared by all three.
    if (int failed = chunk_sharing({0, 1, 2}, VK_SHARING_MODE_CONCURRENT, {0, 1, 2})) return failed;
    // Copies on the async-compute family (no DMA family).
    if (int failed = chunk_sharing({0, 2}, VK_SHARING_MODE_CONCURRENT, {0, 2})) return failed;
    // One family (every D3D12 queue on the graphics family), or none given: EXCLUSIVE, as before.
    if (int failed = chunk_sharing({0}, VK_SHARING_MODE_EXCLUSIVE, {})) return failed;
    if (int failed = chunk_sharing({}, VK_SHARING_MODE_EXCLUSIVE, {})) return failed;
    std::puts("depth24_scratch_sharing: ok");
    return 0;
}
