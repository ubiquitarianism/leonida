// Indexed console quad lists (D3D_PRIMITIVE_TOPOLOGY 19) on Vulkan, which has no quad primitive.
//
// A quad is four indices and Vulkan can only draw it as triangles. A triangle fan per quad reads the application's
// index buffer in place, but gets two things wrong: a fan takes a flat input from each triangle's own first vertex
// (the quad's second and third), where the quad takes it from its first, and one draw per quad runs every instance
// of quad 0 before quad 1, where D3D12 draws instance by instance (visible when instances overlap). So the indices
// are rewritten instead: a compute pass (quad_index_expand.comp) reads the application's index buffer on the GPU and
// writes (i0 i1 i2) (i0 i2 i3) for each quad as 32-bit indices into scratch that lives until the submission
// completes (vulkan_scratch.h), and the draw is one indexed triangle list over that scratch with the application's
// instance count, base vertex and first instance. Both triangles start at the quad's first vertex, so a flat input
// is the first vertex's over the whole quad, and one draw keeps D3D12's instance order.
//
// The indices are read on the GPU, never on the host: an index buffer can be written by a copy or a shader earlier
// in the same list. Index values are copied as they are (a 16-bit index is zero-extended), so VertexIndex -
// BaseVertex is still the application's index. That includes the strip-cut value: 0xFFFF and 0xFFFFFFFF are
// ordinary indices in a quad list, as in every list topology (the replay enables primitive restart for strips only),
// and 0xFFFF stays vertex 65535 after the widening. An index past the end of the index buffer view reads 0, as
// D3D12 defines it, and a draw's last one to three indices that do not make a quad are dropped.
//
// Needs bufferDeviceAddress (the pass reads and writes through device addresses, no descriptors) and scratch with
// INDEX_BUFFER and SHADER_DEVICE_ADDRESS usage. Non-indexed quad lists do not come here: they draw from a shared
// constant index buffer (ReplayFunctions::quad_indices).
#pragma once
#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
#include <vulkan/vulkan.h>
#include <algorithm>
#include <cstdint>
#include "quad_index_expand_spv.h"

namespace leonida {

// One indexed quad-list draw's expansion. `buffer` is null for a draw that needs none.
struct QuadIndexExpansion {
    VkBuffer buffer = VK_NULL_HANDLE;  // the triangle-list indices, 32-bit: six for each quad
    VkDeviceSize offset = 0;
    VkDeviceAddress source = 0;     // the 4-byte aligned word that holds the draw's first index
    VkDeviceAddress triangles = 0;  // `buffer` + `offset`
    std::uint64_t readable = 0;     // indices from the draw's first one that the index buffer view holds
    std::uint32_t first_half = 0;   // 16-bit indices: 1 when the first index is the high half of its word
    std::uint32_t index_bits = 0;   // 16 or 32
    std::uint32_t quads = 0;
};

class QuadIndexExpander {
public:
    static constexpr std::uint32_t indices_per_quad = 6;
    // vkCmdDrawIndexed counts the expanded indices in 32 bits.
    static constexpr std::uint32_t max_quads = UINT32_MAX / indices_per_quad;
    // The scratch bytes `quads` quads expand into.
    static constexpr VkDeviceSize bytes(std::uint32_t quads) {
        return VkDeviceSize(quads) * indices_per_quad * sizeof(std::uint32_t);
    }

    QuadIndexExpander(PFN_vkGetDeviceProcAddr get, VkDevice device, const VkPhysicalDeviceLimits& limits)
        : get_(get), device_(device), limits_(limits) {}
    ~QuadIndexExpander() {
        if (pipeline_) fn<PFN_vkDestroyPipeline>("vkDestroyPipeline")(device_, pipeline_, nullptr);
        if (layout_) fn<PFN_vkDestroyPipelineLayout>("vkDestroyPipelineLayout")(device_, layout_, nullptr);
    }
    QuadIndexExpander(const QuadIndexExpander&) = delete;
    QuadIndexExpander& operator=(const QuadIndexExpander&) = delete;

    // `quads` quads from the indices at `index_offset` in `indices` (the draw's first index), of which the view holds
    // `readable`, into bytes(quads) of `scratch` at `scratch_offset`. The pipeline is made on the first call; the
    // caller serializes calls. False, with `out` cleared, when the draw cannot be expanded.
    bool prepare(VkBuffer indices, VkDeviceSize index_offset, VkIndexType type, std::uint64_t readable,
                 std::uint32_t quads, VkBuffer scratch, VkDeviceSize scratch_offset, QuadIndexExpansion& out) {
        out = {};
        const std::uint32_t index_bytes = type == VK_INDEX_TYPE_UINT16 ? 2 : type == VK_INDEX_TYPE_UINT32 ? 4 : 0;
        if (!indices || !scratch || !index_bytes || !quads || quads > max_quads || index_offset % index_bytes ||
            scratch_offset % sizeof(std::uint32_t) || !limits_.maxComputeWorkGroupCount[0] || !make_pipeline())
            return false;
        const auto address = fn<PFN_vkGetBufferDeviceAddress>("vkGetBufferDeviceAddress");
        VkBufferDeviceAddressInfo info{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
        info.buffer = indices;
        const VkDeviceAddress source = address(device_, &info);
        info.buffer = scratch;
        const VkDeviceAddress target = address(device_, &info);
        if (!source || !target || index_offset > UINT64_MAX - source || scratch_offset > UINT64_MAX - target ||
            (source % sizeof(std::uint32_t)) || (target % sizeof(std::uint32_t)))
            return false;
        out.buffer = scratch;
        out.offset = scratch_offset;
        // The shader reads whole words: a 16-bit index at an address of 2 modulo 4 is the high half of its word.
        out.source = (source + index_offset) & ~VkDeviceAddress(3);
        out.first_half = std::uint32_t((index_offset & 3) / 2);
        out.triangles = target + scratch_offset;
        out.readable = readable;
        out.index_bits = index_bytes * 8;
        out.quads = quads;
        return true;
    }

    // Records the pass: outside a render pass, before the draw that binds job.buffer as its index buffer.
    void emit(VkCommandBuffer cb, const QuadIndexExpansion& job) const {
        const auto barrier = fn<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier");
        // Whatever wrote the application's indices earlier in the command buffer (a copy, a shader) is visible.
        VkMemoryBarrier before{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        before.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
        before.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        barrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0, 1, &before, 0, nullptr, 0, nullptr);
        fn<PFN_vkCmdBindPipeline>("vkCmdBindPipeline")(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
        struct Push {
            VkDeviceAddress source, triangles;
            std::uint32_t first_half, index_bits, limit, quads;
        };
        static_assert(sizeof(Push) == push_bytes);
        // A dispatch indexes its quads and their indices in 32 bits: 4 * quad + 4 and 6 * quad + 5 must fit.
        constexpr VkDeviceSize index_limit = 1u << 28;
        const VkDeviceSize chunk_limit =
            std::min(VkDeviceSize(limits_.maxComputeWorkGroupCount[0]) * group_size, index_limit);
        for (VkDeviceSize first = 0; first < job.quads;) {
            const std::uint32_t quads = std::uint32_t(std::min<VkDeviceSize>(chunk_limit, job.quads - first));
            const VkDeviceSize first_index = first * 4;
            // 16-bit indices: `first_index` is even, so the chunk starts at the same half of a word as the draw.
            const VkDeviceSize source_bytes = first_index * (job.index_bits / 8);
            const std::uint64_t left = job.readable > first_index ? job.readable - first_index : 0;
            const Push push{job.source + source_bytes, job.triangles + first * indices_per_quad * 4,
                            job.first_half, job.index_bits, std::uint32_t(std::min<std::uint64_t>(left, UINT32_MAX)),
                            quads};
            fn<PFN_vkCmdPushConstants>("vkCmdPushConstants")(cb, layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                                                             sizeof(push), &push);
            fn<PFN_vkCmdDispatch>("vkCmdDispatch")(cb, (quads - 1u) / group_size + 1u, 1, 1);
            first += quads;
        }
        VkMemoryBarrier after{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        after.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        after.dstAccessMask = VK_ACCESS_INDEX_READ_BIT;
        barrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,
                0, 1, &after, 0, nullptr, 0, nullptr);
    }

private:
    static constexpr std::uint32_t group_size = 64;  // local_size_x of quad_index_expand.comp
    static constexpr std::uint32_t push_bytes = 32;
    template<typename T> T fn(const char* name) const { return reinterpret_cast<T>(get_(device_, name)); }
    bool make_pipeline() {
        if (pipeline_) return true;
        for (const char* name : {"vkGetBufferDeviceAddress", "vkCreateShaderModule", "vkDestroyShaderModule",
             "vkCreatePipelineLayout", "vkDestroyPipelineLayout", "vkCreateComputePipelines", "vkDestroyPipeline",
             "vkCmdPipelineBarrier", "vkCmdBindPipeline", "vkCmdPushConstants", "vkCmdDispatch"})
            if (!get_(device_, name)) return false;
        VkShaderModule module = VK_NULL_HANDLE;
        VkPipelineLayout layout = VK_NULL_HANDLE;
        VkPipeline pipeline = VK_NULL_HANDLE;
        VkShaderModuleCreateInfo shader{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        shader.codeSize = sizeof(quad_index_expand_spirv);
        shader.pCode = quad_index_expand_spirv;
        bool ok = fn<PFN_vkCreateShaderModule>("vkCreateShaderModule")(device_, &shader, nullptr, &module) == VK_SUCCESS;
        const VkPushConstantRange range{VK_SHADER_STAGE_COMPUTE_BIT, 0, push_bytes};
        VkPipelineLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        layout_info.pushConstantRangeCount = 1;
        layout_info.pPushConstantRanges = &range;
        ok = ok && fn<PFN_vkCreatePipelineLayout>("vkCreatePipelineLayout")(device_, &layout_info, nullptr, &layout) == VK_SUCCESS;
        VkComputePipelineCreateInfo info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        info.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        info.stage.module = module;
        info.stage.pName = "main";
        info.layout = layout;
        ok = ok && fn<PFN_vkCreateComputePipelines>("vkCreateComputePipelines")(
            device_, VK_NULL_HANDLE, 1, &info, nullptr, &pipeline) == VK_SUCCESS;
        if (module) fn<PFN_vkDestroyShaderModule>("vkDestroyShaderModule")(device_, module, nullptr);
        if (!ok) {
            if (pipeline) fn<PFN_vkDestroyPipeline>("vkDestroyPipeline")(device_, pipeline, nullptr);
            if (layout) fn<PFN_vkDestroyPipelineLayout>("vkDestroyPipelineLayout")(device_, layout, nullptr);
            return false;
        }
        layout_ = layout;
        pipeline_ = pipeline;
        return true;
    }
    PFN_vkGetDeviceProcAddr get_;
    VkDevice device_;
    VkPhysicalDeviceLimits limits_;
    VkPipelineLayout layout_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
};
}  // namespace leonida
