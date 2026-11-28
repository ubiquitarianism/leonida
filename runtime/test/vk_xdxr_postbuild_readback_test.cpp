// Real Vulkan GPU regression test: XDXR AccelerationStructurePostbuild writes compacted/current
// acceleration-structure sizes from a compute shader into dest_va (commonly a host-readable READBACK buffer).
// VulkanQueue::execute_now() must emit a COMPUTE_SHADER_BIT -> HOST_BIT visibility barrier for the batch,
// and the host readback must observe the exact GPU-computed BLAS size.
// Exit 77 when there is no Vulkan device with bufferDeviceAddress.
#include "vk_test_device.h"
#include "../src/vulkan_backend.h"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

namespace {

PFN_vkGetDeviceProcAddr g_real_get_device_proc_addr = nullptr;
PFN_vkCmdPipelineBarrier2 g_real_barrier2 = nullptr;
PFN_vkCmdPipelineBarrier g_real_barrier = nullptr;
std::atomic<std::uint32_t> g_compute_to_host_barriers{0};

void VKAPI_PTR hooked_barrier2(VkCommandBuffer cb, const VkDependencyInfo* dep) {
    if (dep) {
        for (std::uint32_t i = 0; i < dep->memoryBarrierCount; ++i) {
            const VkMemoryBarrier2& mb = dep->pMemoryBarriers[i];
            const bool src_compute = (mb.srcStageMask & VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT) != 0;
            const bool src_write =
                (mb.srcAccessMask & (VK_ACCESS_2_MEMORY_WRITE_BIT | VK_ACCESS_2_SHADER_WRITE_BIT)) != 0;
            const bool dst_host = (mb.dstStageMask & VK_PIPELINE_STAGE_2_HOST_BIT) != 0 &&
                                  (mb.dstAccessMask & VK_ACCESS_2_HOST_READ_BIT) != 0;
            if (src_compute && src_write && dst_host) ++g_compute_to_host_barriers;
        }
    }
    g_real_barrier2(cb, dep);
}

void VKAPI_PTR hooked_barrier(VkCommandBuffer cb, VkPipelineStageFlags srcStageMask, VkPipelineStageFlags dstStageMask,
                              VkDependencyFlags dependencyFlags, std::uint32_t memoryBarrierCount,
                              const VkMemoryBarrier* pMemoryBarriers, std::uint32_t bufferMemoryBarrierCount,
                              const VkBufferMemoryBarrier* pBufferMemoryBarriers, std::uint32_t imageMemoryBarrierCount,
                              const VkImageMemoryBarrier* pImageMemoryBarriers) {
    const bool src_ok =
        (srcStageMask & (VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_ALL_COMMANDS_BIT)) != 0;
    const bool dst_ok = (dstStageMask & VK_PIPELINE_STAGE_HOST_BIT) != 0;
    if (src_ok && dst_ok) {
        for (std::uint32_t i = 0; i < memoryBarrierCount; ++i) {
            const VkMemoryBarrier& mb = pMemoryBarriers[i];
            const bool src_write = (mb.srcAccessMask & (VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT)) != 0;
            const bool dst_read = (mb.dstAccessMask & VK_ACCESS_HOST_READ_BIT) != 0;
            if (src_write && dst_read) ++g_compute_to_host_barriers;
        }
    }
    g_real_barrier(cb, srcStageMask, dstStageMask, dependencyFlags, memoryBarrierCount, pMemoryBarriers,
                   bufferMemoryBarrierCount, pBufferMemoryBarriers, imageMemoryBarrierCount, pImageMemoryBarriers);
}

PFN_vkVoidFunction VKAPI_PTR wrapped_get_device_proc_addr(VkDevice device, const char* name) {
    PFN_vkVoidFunction fn = g_real_get_device_proc_addr(device, name);
    if (!fn || !name) return fn;
    if (std::strcmp(name, "vkCmdPipelineBarrier2") == 0) {
        g_real_barrier2 = reinterpret_cast<PFN_vkCmdPipelineBarrier2>(fn);
        return reinterpret_cast<PFN_vkVoidFunction>(hooked_barrier2);
    }
    if (std::strcmp(name, "vkCmdPipelineBarrier") == 0) {
        g_real_barrier = reinterpret_cast<PFN_vkCmdPipelineBarrier>(fn);
        return reinterpret_cast<PFN_vkVoidFunction>(hooked_barrier);
    }
    return fn;
}

}  // namespace

int main() {
    using namespace leonida;

    vktest::Device gpu;
    gpu.request_buffer_device_address = true;
    if (!gpu.create() || !gpu.buffer_device_address) {
        std::printf("vk_xdxr_postbuild_readback: skipped (no Vulkan device with bufferDeviceAddress)\n");
        return 77;
    }

    g_real_get_device_proc_addr = gpu.device_proc;
    VulkanContext ctx(wrapped_get_device_proc_addr, gpu.device, /*separate_depth_stencil_layouts=*/false,
                      /*conditional_rendering=*/false, /*ray_tracing=*/false, /*xdxr=*/true);
    EXPECT(ctx.loaded());

    VulkanQueue queue(ctx, gpu.queue, gpu.family);
    EXPECT(queue.ok());

    // Build a minimal 1-triangle XDXR BLAS on the real GPU.
    void* vtx_map = nullptr;
    const VkBuffer vtx_buf = gpu.buffer(256, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &vtx_map);
    EXPECT(vtx_buf && vtx_map);
    const float tri[9] = {0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f};
    std::memcpy(vtx_map, tri, sizeof(tri));
    const std::uint64_t vtx_va = gpu.device_address(vtx_buf);

    XdxrLayout layout;
    EXPECT(layout.compute(1, /*is_tlas=*/false, /*box16=*/false));
    void* blas_map = nullptr;
    const VkBuffer blas_buf = gpu.buffer(layout.size + 512, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &blas_map);
    void* scratch_map = nullptr;
    const VkBuffer scratch_buf =
        gpu.buffer(std::size_t(layout.scratch) + 512, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &scratch_map);
    void* postbuild_map = nullptr;
    const VkBuffer postbuild_buf = gpu.buffer(256, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &postbuild_map);
    EXPECT(blas_buf && blas_map && scratch_buf && scratch_map && postbuild_buf && postbuild_map);

    const std::uint64_t blas_raw_va = gpu.device_address(blas_buf);
    const std::uint64_t blas_va = (blas_raw_va + 255ull) & ~255ull;
    const std::uint64_t scratch_va = (gpu.device_address(scratch_buf) + 255ull) & ~255ull;
    const std::uint64_t postbuild_va = (gpu.device_address(postbuild_buf) + 7ull) & ~7ull;
    const std::size_t postbuild_off = std::size_t(postbuild_va - gpu.device_address(postbuild_buf));

    // Poison the postbuild readback buffer so we verify real GPU writeback.
    constexpr std::uint64_t poison = 0xDEADBEEFCAFEBABEull;
    std::memcpy(static_cast<char*>(postbuild_map) + postbuild_off, &poison, sizeof(poison));

    // Register the host-readable postbuild buffer in VulkanContext so prepare() resolves dest_va via buffer_at.
    ResourceBinding rb{};
    rb.buffer = postbuild_buf;
    rb.size = 256;
    rb.va = gpu.device_address(postbuild_buf);
    rb.host_readable = true;
    ctx.add_resource(1, rb);

    // Step 1: Build the BLAS on the real GPU.
    BuildAccelerationStructure build{};
    build.is_tlas = false;
    build.dst_va = blas_va;
    build.scratch_va = scratch_va;
    RtTriangleGeometry geom{};
    geom.vertex_va = vtx_va;
    geom.vertex_count = 3;
    geom.vertex_stride = 12;
    geom.vertex_format = VK_FORMAT_R32G32B32_SFLOAT;
    geom.opaque = true;
    geom.flags = 1;
    build.triangles.push_back(geom);

    CommandList build_list;
    EXPECT(build_list.build_acceleration_structure(build) && build_list.close());
    const CommandList* build_ptrs[1] = {&build_list};
    EXPECT(queue.execute(build_ptrs) == ReplayResult::success);
    EXPECT(queue.idle());

    // Step 2: Execute a command list containing ONLY XDXR AccelerationStructurePostbuild into the READBACK buffer.
    g_compute_to_host_barriers.store(0);
    CommandList postbuild_list;
    EXPECT(postbuild_list.acceleration_structure_postbuild({postbuild_va, /*type=*/0, {blas_va}}) &&
           postbuild_list.close());
    const CommandList* postbuild_ptrs[1] = {&postbuild_list};
    EXPECT(queue.execute(postbuild_ptrs) == ReplayResult::success);
    EXPECT(queue.idle());

    const std::uint32_t barriers = g_compute_to_host_barriers.load();
    std::uint64_t reported_size = 0;
    std::memcpy(&reported_size, static_cast<const char*>(postbuild_map) + postbuild_off, sizeof(reported_size));

    std::uint32_t header_units = 0;
    std::memcpy(&header_units, static_cast<const char*>(blas_map) + (blas_va - blas_raw_va) + 4, sizeof(header_units));
    const std::uint64_t expected_size = (std::uint64_t(header_units) + 1ull) * 64ull;

    EXPECT(barriers >= 1);
    EXPECT(reported_size != poison);
    EXPECT(reported_size > 0 && reported_size <= layout.size);
    EXPECT(reported_size == expected_size);

    std::printf("vk_xdxr_postbuild_readback: PASS (compute_to_host_barriers=%u, reported_size=%llu, expected=%llu)\n",
                barriers, static_cast<unsigned long long>(reported_size),
                static_cast<unsigned long long>(expected_size));
    return 0;
}
