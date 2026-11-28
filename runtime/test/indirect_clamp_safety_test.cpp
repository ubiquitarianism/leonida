// Pure-CPU mock-Vulkan tests for runtime/src/vulkan_indirect_clamp.h and runtime/src/vulkan_scratch.h:
//   - multi-queue-family sharing (VK_SHARING_MODE_CONCURRENT) on both scratch chunks and HostArea counter buffers;
//   - freeing VkDeviceMemory before retrying the next memory type when vkBindBufferMemory fails in make_area();
//   - zeroing released counter slots so an aborted/unsubmitted batch never double-counts stale counters;
//   - rejecting non-4-byte-aligned argument_offset and count_offset;
//   - thread-safe concurrent prepare() + release() while areas_ grows (TSan clean).
// No GPU, Vulkan loader, or game files required.
#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
#include "../src/vulkan_indirect_clamp.h"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

#define CHECK(cond)                                                                           \
    do {                                                                                      \
        if (!(cond)) {                                                                        \
            std::fprintf(stderr, "line %d: CHECK failed: %s\n", __LINE__, #cond);             \
            return 1;                                                                         \
        }                                                                                     \
    } while (0)

namespace {

struct BufferRecord {
    VkBufferUsageFlags usage = 0;
    VkSharingMode sharing_mode = VK_SHARING_MODE_EXCLUSIVE;
    std::vector<std::uint32_t> families;
    VkDeviceSize size = 0;
};

struct MockState {
    std::mutex mutex;
    std::uint64_t next_handle = 1;
    std::uint32_t fail_bind_type = UINT32_MAX;
    int alloc_memory_calls = 0;
    int free_memory_calls = 0;
    int create_buffer_calls = 0;
    int destroy_buffer_calls = 0;
    std::unordered_map<std::uintptr_t, BufferRecord> buffers;
    std::unordered_map<std::uintptr_t, std::uint32_t> mem_types;
    std::unordered_map<std::uintptr_t, std::unique_ptr<std::uint8_t[]>> host_maps;
    std::unordered_map<std::uintptr_t, std::uint8_t*> buffer_host;
};

MockState* g_mock = nullptr;

template <typename Handle>
Handle make_handle(std::uint64_t id) {
    return reinterpret_cast<Handle>(static_cast<std::uintptr_t>(id));
}

VKAPI_ATTR VkResult VKAPI_CALL mock_create_buffer(VkDevice, const VkBufferCreateInfo* info,
                                                  const VkAllocationCallbacks*, VkBuffer* out) {
    std::lock_guard lock(g_mock->mutex);
    ++g_mock->create_buffer_calls;
    const auto id = ++g_mock->next_handle;
    BufferRecord rec{};
    rec.usage = info->usage;
    rec.sharing_mode = info->sharingMode;
    rec.size = info->size;
    if (info->pQueueFamilyIndices && info->queueFamilyIndexCount)
        rec.families.assign(info->pQueueFamilyIndices, info->pQueueFamilyIndices + info->queueFamilyIndexCount);
    g_mock->buffers[static_cast<std::uintptr_t>(id)] = std::move(rec);
    *out = make_handle<VkBuffer>(id);
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL mock_destroy_buffer(VkDevice, VkBuffer buffer, const VkAllocationCallbacks*) {
    std::lock_guard lock(g_mock->mutex);
    ++g_mock->destroy_buffer_calls;
    g_mock->buffers.erase(reinterpret_cast<std::uintptr_t>(buffer));
    g_mock->buffer_host.erase(reinterpret_cast<std::uintptr_t>(buffer));
}

VKAPI_ATTR void VKAPI_CALL mock_get_buffer_memory_requirements(VkDevice, VkBuffer buffer,
                                                               VkMemoryRequirements* req) {
    std::lock_guard lock(g_mock->mutex);
    const auto it = g_mock->buffers.find(reinterpret_cast<std::uintptr_t>(buffer));
    req->size = it != g_mock->buffers.end() ? it->second.size : 256;
    req->alignment = 256;
    req->memoryTypeBits = 0x3u;
}

VKAPI_ATTR VkResult VKAPI_CALL mock_allocate_memory(VkDevice, const VkMemoryAllocateInfo* info,
                                                    const VkAllocationCallbacks*, VkDeviceMemory* out) {
    std::lock_guard lock(g_mock->mutex);
    ++g_mock->alloc_memory_calls;
    const auto id = ++g_mock->next_handle;
    g_mock->mem_types[static_cast<std::uintptr_t>(id)] = info->memoryTypeIndex;
    g_mock->host_maps[static_cast<std::uintptr_t>(id)] = std::make_unique<std::uint8_t[]>(size_t(info->allocationSize));
    std::memset(g_mock->host_maps[static_cast<std::uintptr_t>(id)].get(), 0, size_t(info->allocationSize));
    *out = make_handle<VkDeviceMemory>(id);
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL mock_free_memory(VkDevice, VkDeviceMemory mem, const VkAllocationCallbacks*) {
    std::lock_guard lock(g_mock->mutex);
    ++g_mock->free_memory_calls;
    g_mock->mem_types.erase(reinterpret_cast<std::uintptr_t>(mem));
    g_mock->host_maps.erase(reinterpret_cast<std::uintptr_t>(mem));
}

VKAPI_ATTR VkResult VKAPI_CALL mock_bind_buffer_memory(VkDevice, VkBuffer buffer, VkDeviceMemory mem, VkDeviceSize) {
    std::lock_guard lock(g_mock->mutex);
    const auto it = g_mock->mem_types.find(reinterpret_cast<std::uintptr_t>(mem));
    if (it != g_mock->mem_types.end() && it->second == g_mock->fail_bind_type)
        return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    const auto mit = g_mock->host_maps.find(reinterpret_cast<std::uintptr_t>(mem));
    if (mit != g_mock->host_maps.end()) g_mock->buffer_host[reinterpret_cast<std::uintptr_t>(buffer)] = mit->second.get();
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL mock_map_memory(VkDevice, VkDeviceMemory mem, VkDeviceSize offset, VkDeviceSize,
                                               VkMemoryMapFlags, void** ppData) {
    std::lock_guard lock(g_mock->mutex);
    const auto it = g_mock->host_maps.find(reinterpret_cast<std::uintptr_t>(mem));
    if (it == g_mock->host_maps.end()) return VK_ERROR_MEMORY_MAP_FAILED;
    *ppData = it->second.get() + offset;
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL mock_unmap_memory(VkDevice, VkDeviceMemory) {}

VKAPI_ATTR VkDeviceAddress VKAPI_CALL mock_get_buffer_device_address(VkDevice, const VkBufferDeviceAddressInfo* info) {
    return static_cast<VkDeviceAddress>(0x10000000ull + reinterpret_cast<std::uintptr_t>(info->buffer) * 0x1000ull);
}

VKAPI_ATTR VkResult VKAPI_CALL mock_get_semaphore_counter_value(VkDevice, VkSemaphore, std::uint64_t* value) {
    *value = UINT64_MAX;
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL mock_device_wait_idle(VkDevice) { return VK_SUCCESS; }

VKAPI_ATTR VkResult VKAPI_CALL mock_create_shader_module(VkDevice, const VkShaderModuleCreateInfo*,
                                                         const VkAllocationCallbacks*, VkShaderModule* out) {
    std::lock_guard lock(g_mock->mutex);
    *out = make_handle<VkShaderModule>(++g_mock->next_handle);
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL mock_destroy_shader_module(VkDevice, VkShaderModule, const VkAllocationCallbacks*) {}

VKAPI_ATTR VkResult VKAPI_CALL mock_create_pipeline_layout(VkDevice, const VkPipelineLayoutCreateInfo*,
                                                           const VkAllocationCallbacks*, VkPipelineLayout* out) {
    std::lock_guard lock(g_mock->mutex);
    *out = make_handle<VkPipelineLayout>(++g_mock->next_handle);
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL mock_destroy_pipeline_layout(VkDevice, VkPipelineLayout, const VkAllocationCallbacks*) {}

VKAPI_ATTR VkResult VKAPI_CALL mock_create_compute_pipelines(VkDevice, VkPipelineCache, std::uint32_t,
                                                             const VkComputePipelineCreateInfo*,
                                                             const VkAllocationCallbacks*, VkPipeline* out) {
    std::lock_guard lock(g_mock->mutex);
    *out = make_handle<VkPipeline>(++g_mock->next_handle);
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL mock_destroy_pipeline(VkDevice, VkPipeline, const VkAllocationCallbacks*) {}

VKAPI_ATTR void VKAPI_CALL mock_cmd_pipeline_barrier(VkCommandBuffer, VkPipelineStageFlags, VkPipelineStageFlags,
                                                     VkDependencyFlags, std::uint32_t, const VkMemoryBarrier*,
                                                     std::uint32_t, const VkBufferMemoryBarrier*, std::uint32_t,
                                                     const VkImageMemoryBarrier*) {}
VKAPI_ATTR void VKAPI_CALL mock_cmd_bind_pipeline(VkCommandBuffer, VkPipelineBindPoint, VkPipeline) {}
VKAPI_ATTR void VKAPI_CALL mock_cmd_push_constants(VkCommandBuffer, VkPipelineLayout, VkShaderStageFlags, std::uint32_t,
                                                   std::uint32_t, const void*) {}
VKAPI_ATTR void VKAPI_CALL mock_cmd_dispatch(VkCommandBuffer, std::uint32_t, std::uint32_t, std::uint32_t) {}
VKAPI_ATTR void VKAPI_CALL mock_cmd_fill_buffer(VkCommandBuffer, VkBuffer, VkDeviceSize, VkDeviceSize, std::uint32_t) {}
VKAPI_ATTR void VKAPI_CALL mock_cmd_copy_buffer(VkCommandBuffer, VkBuffer, VkBuffer, std::uint32_t,
                                                const VkBufferCopy*) {}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL mock_get_device_proc_addr(VkDevice, const char* name) {
#define ENTRY(fn_name, impl) \
    if (std::strcmp(name, #fn_name) == 0) return reinterpret_cast<PFN_vkVoidFunction>(impl)
    ENTRY(vkCreateBuffer, mock_create_buffer);
    ENTRY(vkDestroyBuffer, mock_destroy_buffer);
    ENTRY(vkGetBufferMemoryRequirements, mock_get_buffer_memory_requirements);
    ENTRY(vkAllocateMemory, mock_allocate_memory);
    ENTRY(vkFreeMemory, mock_free_memory);
    ENTRY(vkBindBufferMemory, mock_bind_buffer_memory);
    ENTRY(vkMapMemory, mock_map_memory);
    ENTRY(vkUnmapMemory, mock_unmap_memory);
    ENTRY(vkGetBufferDeviceAddress, mock_get_buffer_device_address);
    ENTRY(vkGetSemaphoreCounterValue, mock_get_semaphore_counter_value);
    ENTRY(vkDeviceWaitIdle, mock_device_wait_idle);
    ENTRY(vkCreateShaderModule, mock_create_shader_module);
    ENTRY(vkDestroyShaderModule, mock_destroy_shader_module);
    ENTRY(vkCreatePipelineLayout, mock_create_pipeline_layout);
    ENTRY(vkDestroyPipelineLayout, mock_destroy_pipeline_layout);
    ENTRY(vkCreateComputePipelines, mock_create_compute_pipelines);
    ENTRY(vkDestroyPipeline, mock_destroy_pipeline);
    ENTRY(vkCmdPipelineBarrier, mock_cmd_pipeline_barrier);
    ENTRY(vkCmdBindPipeline, mock_cmd_bind_pipeline);
    ENTRY(vkCmdPushConstants, mock_cmd_push_constants);
    ENTRY(vkCmdDispatch, mock_cmd_dispatch);
    ENTRY(vkCmdFillBuffer, mock_cmd_fill_buffer);
    ENTRY(vkCmdCopyBuffer, mock_cmd_copy_buffer);
#undef ENTRY
    return nullptr;
}

VkPhysicalDeviceMemoryProperties make_memory_props() {
    VkPhysicalDeviceMemoryProperties mem{};
    mem.memoryHeapCount = 2;
    mem.memoryHeaps[0] = {VkDeviceSize(1) << 30, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT};
    mem.memoryHeaps[1] = {VkDeviceSize(4) << 30, 0};
    mem.memoryTypeCount = 2;
    mem.memoryTypes[0] = {VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 0};
    mem.memoryTypes[1] = {VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 1};
    return mem;
}

VkPhysicalDeviceLimits make_limits() {
    VkPhysicalDeviceLimits limits{};
    limits.maxComputeWorkGroupCount[0] = 65535u;
    return limits;
}

leonida::ScratchBuffers::Allocate make_scratch_allocator() {
    return [](const VkMemoryRequirements& req) -> VkDeviceMemory {
        VkMemoryAllocateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        info.allocationSize = req.size;
        info.memoryTypeIndex = 1;  // never blocked by fail_bind_type = 0
        VkDeviceMemory mem = VK_NULL_HANDLE;
        mock_allocate_memory(VK_NULL_HANDLE, &info, nullptr, &mem);
        return mem;
    };
}

}  // namespace

int main() {
    using namespace leonida;
    const VkDevice dummy_device = make_handle<VkDevice>(1);
    const VkBuffer args_buf = make_handle<VkBuffer>(0x100);
    const VkBuffer count_buf = make_handle<VkBuffer>(0x200);

    // 1. Multi-family sharing: both scratch chunks and HostArea counter buffers must use VK_SHARING_MODE_CONCURRENT.
    {
        MockState state;
        g_mock = &state;
        const std::vector<std::uint32_t> families = {0u, 2u};
        IndirectArgumentClamp clamp(mock_get_device_proc_addr, dummy_device, make_memory_props(), make_limits(),
                                    make_scratch_allocator(), nullptr, families);
        CHECK(clamp.loaded());
        auto batch = clamp.batch();
        ClampedIndirect job{};
        CHECK(batch->prepare(args_buf, 0, count_buf, 0, indirect_clamp_draw, 16, 4, job));
        CHECK(job.buffer != VK_NULL_HANDLE && job.stats_buffer != VK_NULL_HANDLE);
        const auto scratch_it = state.buffers.find(reinterpret_cast<std::uintptr_t>(job.buffer));
        const auto area_it = state.buffers.find(reinterpret_cast<std::uintptr_t>(job.stats_buffer));
        CHECK(scratch_it != state.buffers.end() && area_it != state.buffers.end());
        CHECK(scratch_it->second.sharing_mode == VK_SHARING_MODE_CONCURRENT);
        CHECK(scratch_it->second.families == families);
        CHECK(area_it->second.sharing_mode == VK_SHARING_MODE_CONCURRENT);
        CHECK(area_it->second.families == families);
        batch.reset();
        clamp.retire(VK_NULL_HANDLE, 0);
        clamp.collect();
    }

    // 1b. Single-family stays VK_SHARING_MODE_EXCLUSIVE.
    {
        MockState state;
        g_mock = &state;
        IndirectArgumentClamp clamp(mock_get_device_proc_addr, dummy_device, make_memory_props(), make_limits(),
                                    make_scratch_allocator(), nullptr, {0u});
        auto batch = clamp.batch();
        ClampedIndirect job{};
        CHECK(batch->prepare(args_buf, 0, VK_NULL_HANDLE, 0, indirect_clamp_dispatch, 12, 2, job));
        CHECK(state.buffers.at(reinterpret_cast<std::uintptr_t>(job.buffer)).sharing_mode ==
              VK_SHARING_MODE_EXCLUSIVE);
        CHECK(state.buffers.at(reinterpret_cast<std::uintptr_t>(job.stats_buffer)).sharing_mode ==
              VK_SHARING_MODE_EXCLUSIVE);
    }

    // 2. vkBindBufferMemory failure on the first memory type in make_area() must free that VkDeviceMemory before
    //    retrying the next type (zero leaked allocations).
    {
        MockState state;
        state.fail_bind_type = 0u;
        g_mock = &state;
        {
            IndirectArgumentClamp clamp(mock_get_device_proc_addr, dummy_device, make_memory_props(), make_limits(),
                                        make_scratch_allocator());
            auto batch = clamp.batch();
            ClampedIndirect job{};
            CHECK(batch->prepare(args_buf, 0, VK_NULL_HANDLE, 0, indirect_clamp_draw, 16, 1, job));
            // 2 host-area memory attempts (type 0 failed bind & freed immediately, type 1 succeeded) + 1 scratch alloc.
            CHECK(state.alloc_memory_calls == 3);
            CHECK(state.free_memory_calls == 1);
        }
        CHECK(state.alloc_memory_calls == state.free_memory_calls);
        CHECK(state.create_buffer_calls == state.destroy_buffer_calls);
    }

    // 3. Released counter slots are zeroed so a subsequent aborted batch reusing the same slot never double-counts
    //    the prior submission's counters.
    {
        MockState state;
        g_mock = &state;
        IndirectArgumentClamp clamp(mock_get_device_proc_addr, dummy_device, make_memory_props(), make_limits(),
                                    make_scratch_allocator());
        VkBuffer area_buf = VK_NULL_HANDLE;
        VkDeviceSize slot_off = 0;
        {
            auto batch1 = clamp.batch();
            ClampedIndirect job1{};
            CHECK(batch1->prepare(args_buf, 0, VK_NULL_HANDLE, 0, indirect_clamp_draw, 16, 1, job1));
            area_buf = job1.stats_buffer;
            slot_off = job1.stats_offset;
            std::uint8_t* host = state.buffer_host.at(reinterpret_cast<std::uintptr_t>(area_buf));
            const std::uint32_t gpu_written[12] = {5u, 1u, 2u, 3u, 0u, 0u, 0u, 0u, 7u, 4u, 3u, 2u};
            std::memcpy(host + slot_off, gpu_written, sizeof(gpu_written));
        }
        const auto after1 = clamp.counters();
        CHECK(after1.records_clamped == 5 && after1.counts_clamped == 1 && after1.over_budget == 2 &&
              after1.zeroed == 3);
        CHECK(after1.mesh_records == 7 && after1.mesh_yz_records == 4 &&
              after1.mesh_max_y == 3 && after1.mesh_max_z == 2);

        // Batch 2 reuses the exact same slot (LIFO free_slots_) and is destroyed without GPU submission.
        {
            auto batch2 = clamp.batch();
            ClampedIndirect job2{};
            CHECK(batch2->prepare(args_buf, 0, VK_NULL_HANDLE, 0, indirect_clamp_draw, 16, 1, job2));
            CHECK(job2.stats_buffer == area_buf && job2.stats_offset == slot_off);
        }
        const auto after2 = clamp.counters();
        CHECK(after2.records_clamped == 5 && after2.counts_clamped == 1 && after2.over_budget == 2 &&
              after2.zeroed == 3);
        CHECK(after2.mesh_records == 7 && after2.mesh_yz_records == 4 &&
              after2.mesh_max_y == 3 && after2.mesh_max_z == 2);
    }

    // 4. Non-4-byte-aligned argument_offset or count_offset is rejected before allocating scratch.
    {
        MockState state;
        g_mock = &state;
        IndirectArgumentClamp clamp(mock_get_device_proc_addr, dummy_device, make_memory_props(), make_limits(),
                                    make_scratch_allocator());
        auto batch = clamp.batch();
        ClampedIndirect job{};
        CHECK(!batch->prepare(args_buf, 2, VK_NULL_HANDLE, 0, indirect_clamp_draw, 16, 1, job));
        CHECK(!batch->prepare(args_buf, 0, count_buf, 6, indirect_clamp_draw, 16, 1, job));
        CHECK(batch->calls() == 0);
        // An uncounted call ignores count_offset == 0 and still succeeds on 4-byte-aligned argument_offset.
        CHECK(batch->prepare(args_buf, 4, VK_NULL_HANDLE, 0, indirect_clamp_draw, 16, 1, job));
        CHECK(batch->calls() == 1);
    }

    // 5. Concurrent area growth during prepare() + background batch release() + counters() under TSan.
    {
        MockState state;
        g_mock = &state;
        IndirectArgumentClamp clamp(mock_get_device_proc_addr, dummy_device, make_memory_props(), make_limits(),
                                    make_scratch_allocator(), nullptr, {0u, 1u});
        std::mutex queue_mutex;
        std::vector<std::unique_ptr<IndirectArgumentClamp::Batch>> completed;
        std::atomic<bool> done{false};

        std::thread releaser([&] {
            while (!done.load(std::memory_order_acquire)) {
                std::unique_ptr<IndirectArgumentClamp::Batch> b;
                {
                    std::lock_guard lock(queue_mutex);
                    if (!completed.empty()) {
                        b = std::move(completed.back());
                        completed.pop_back();
                    }
                }
                if (b) b.reset();
                else std::this_thread::yield();
                (void)clamp.counters();
            }
            for (;;) {
                std::unique_ptr<IndirectArgumentClamp::Batch> b;
                {
                    std::lock_guard lock(queue_mutex);
                    if (completed.empty()) break;
                    b = std::move(completed.back());
                    completed.pop_back();
                }
                b.reset();
            }
        });

        // Allocate > 128 slots across several batches so areas_ grows and reallocates while releaser runs.
        for (int b = 0; b < 8; ++b) {
            auto batch = clamp.batch();
            for (int i = 0; i < 40; ++i) {
                ClampedIndirect job{};
                CHECK(batch->prepare(args_buf, 0, VK_NULL_HANDLE, 0, indirect_clamp_draw, 16, 1, job));
                CHECK(job.stats_buffer != VK_NULL_HANDLE);
            }
            clamp.retire(VK_NULL_HANDLE, 0);
            clamp.collect();
            {
                std::lock_guard lock(queue_mutex);
                completed.push_back(std::move(batch));
            }
        }
        done.store(true, std::memory_order_release);
        releaser.join();
    }

    std::printf("indirect_clamp_safety_test: all checks passed\n");
    return 0;
}
