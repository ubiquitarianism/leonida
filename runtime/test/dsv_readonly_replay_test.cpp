// CPU-only replay regression for TargetView::dsv_flags (dsv_read_only_depth / dsv_read_only_stencil):
// checks dynamic-rendering attachment layouts, storeOp, barrier access masks, LayoutTracker transitions (with and
// without separate_depth_stencil_layouts and sampled_in_general), pipeline write-suppression variant requests, and
// ClearDepthStencil skipping read-only aspects. No Vulkan loader, GPU, shaders or game data.
#include "../src/vulkan_replay.h"

#include <cstdio>
#include <vector>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

namespace {

struct Recorded {
    std::vector<VkImageMemoryBarrier> image_barriers;
    std::vector<VkPipelineStageFlags> image_barrier_dst_stages;
    unsigned memory_barriers = 0;
    unsigned image_view_calls = 0;
    bool saw_rendering = false;
    bool has_depth = false;
    bool has_stencil = false;
    VkRenderingAttachmentInfo depth{};
    VkRenderingAttachmentInfo stencil{};
    unsigned clear_depth_calls = 0;
    VkImageAspectFlags cleared_aspects = 0;
    std::vector<std::uint32_t> variant_requests;
    VkPipeline bound_pipeline = VK_NULL_HANDLE;

    void reset() {
        image_barriers.clear();
        image_barrier_dst_stages.clear();
        memory_barriers = 0;
        image_view_calls = 0;
        saw_rendering = has_depth = has_stencil = false;
        depth = {};
        stencil = {};
        clear_depth_calls = 0;
        cleared_aspects = 0;
        variant_requests.clear();
        bound_pipeline = VK_NULL_HANDLE;
    }
};

Recorded g;

VKAPI_ATTR void VKAPI_CALL rec_barrier(VkCommandBuffer, VkPipelineStageFlags, VkPipelineStageFlags dst_stage,
                                       VkDependencyFlags, std::uint32_t mem_count, const VkMemoryBarrier*,
                                       std::uint32_t, const VkBufferMemoryBarrier*, std::uint32_t img_count,
                                       const VkImageMemoryBarrier* imgs) {
    g.memory_barriers += mem_count;
    for (std::uint32_t i = 0; i < img_count; ++i) {
        g.image_barriers.push_back(imgs[i]);
        g.image_barrier_dst_stages.push_back(dst_stage);
    }
}

VKAPI_ATTR void VKAPI_CALL rec_begin(VkCommandBuffer, const VkRenderingInfo* info) {
    g.saw_rendering = true;
    g.has_depth = info->pDepthAttachment != nullptr;
    g.has_stencil = info->pStencilAttachment != nullptr;
    if (g.has_depth) g.depth = *info->pDepthAttachment;
    if (g.has_stencil) g.stencil = *info->pStencilAttachment;
}

VKAPI_ATTR void VKAPI_CALL rec_end(VkCommandBuffer) {}
VKAPI_ATTR void VKAPI_CALL rec_bind_pipe(VkCommandBuffer, VkPipelineBindPoint, VkPipeline p) { g.bound_pipeline = p; }
VKAPI_ATTR void VKAPI_CALL rec_push(VkCommandBuffer, VkPipelineLayout, VkShaderStageFlags, std::uint32_t, std::uint32_t,
                                    const void*) {}
VKAPI_ATTR void VKAPI_CALL rec_vp(VkCommandBuffer, std::uint32_t, std::uint32_t, const VkViewport*) {}
VKAPI_ATTR void VKAPI_CALL rec_sc(VkCommandBuffer, std::uint32_t, std::uint32_t, const VkRect2D*) {}
VKAPI_ATTR void VKAPI_CALL rec_topo(VkCommandBuffer, VkPrimitiveTopology) {}
VKAPI_ATTR void VKAPI_CALL rec_restart(VkCommandBuffer, VkBool32) {}
VKAPI_ATTR void VKAPI_CALL rec_stencil_ref(VkCommandBuffer, VkStencilFaceFlags, std::uint32_t) {}
VKAPI_ATTR void VKAPI_CALL rec_blend(VkCommandBuffer, const float[4]) {}
VKAPI_ATTR void VKAPI_CALL rec_draw(VkCommandBuffer, std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t) {}
VKAPI_ATTR void VKAPI_CALL rec_draw_indexed(VkCommandBuffer, std::uint32_t, std::uint32_t, std::uint32_t,
                                            std::int32_t, std::uint32_t) {}
VKAPI_ATTR void VKAPI_CALL rec_vb2(VkCommandBuffer, std::uint32_t, std::uint32_t, const VkBuffer*,
                                   const VkDeviceSize*, const VkDeviceSize*, const VkDeviceSize*) {}
VKAPI_ATTR void VKAPI_CALL rec_ib(VkCommandBuffer, VkBuffer, VkDeviceSize, VkIndexType) {}
VKAPI_ATTR void VKAPI_CALL rec_clear_ds(VkCommandBuffer, VkImage, VkImageLayout, const VkClearDepthStencilValue*,
                                        std::uint32_t count, const VkImageSubresourceRange* ranges) {
    ++g.clear_depth_calls;
    if (count) g.cleared_aspects |= ranges[0].aspectMask;
}

leonida::ReplayFunctions make_vk(bool separate, leonida::GraphicsPipeline* pso) {
    leonida::ReplayFunctions vk{};
    vk.separate_depth_stencil_layouts = separate;
    vk.pipeline_barrier = rec_barrier;
    vk.begin_rendering = rec_begin;
    vk.end_rendering = rec_end;
    vk.bind_pipeline = rec_bind_pipe;
    vk.push_constants = rec_push;
    vk.bind_vertex_buffers2 = rec_vb2;
    vk.bind_index_buffer = rec_ib;
    vk.set_viewport = rec_vp;
    vk.set_scissor = rec_sc;
    vk.set_primitive_topology = rec_topo;
    vk.set_primitive_restart_enable = rec_restart;
    vk.set_stencil_reference = rec_stencil_ref;
    vk.set_blend_constants = rec_blend;
    vk.draw = rec_draw;
    vk.draw_indexed = rec_draw_indexed;
    vk.clear_depth_stencil_image = rec_clear_ds;
    vk.buffer_at = [](std::uint64_t, std::uint64_t&, VkBuffer&, VkDeviceSize&) { return false; };
    vk.image_view = [](const auto&, const auto&) {
        ++g.image_view_calls;
        return reinterpret_cast<VkImageView>(std::uintptr_t{0x10});
    };
    vk.graphics_pipeline = [pso](auto) { return pso; };
    return vk;
}

leonida::ResourceBinding depth_binding(bool stencil, bool sampled_in_general = false) {
    leonida::ResourceBinding b{};
    b.image = reinterpret_cast<VkImage>(std::uintptr_t{0x1});
    b.format = stencil ? VK_FORMAT_D24_UNORM_S8_UINT : VK_FORMAT_D32_SFLOAT;
    b.aspects = VK_IMAGE_ASPECT_DEPTH_BIT | (stencil ? VK_IMAGE_ASPECT_STENCIL_BIT : 0u);
    b.width = b.height = 16;
    b.not_zeroed = true;
    b.sampled_in_general = sampled_in_general;
    return b;
}

leonida::GraphicsPipeline make_pso(bool stencil, bool depth_writes, bool stencil_writes) {
    leonida::GraphicsPipeline p{};
    p.pipeline = reinterpret_cast<VkPipeline>(std::uintptr_t{0x100});
    p.layout = reinterpret_cast<VkPipelineLayout>(std::uintptr_t{0x200});
    p.push_stages = VK_SHADER_STAGE_VERTEX_BIT;
    p.depth_format = stencil ? VK_FORMAT_D24_UNORM_S8_UINT : VK_FORMAT_D32_SFLOAT;
    p.depth_aspects = VK_IMAGE_ASPECT_DEPTH_BIT | (stencil ? VK_IMAGE_ASPECT_STENCIL_BIT : 0u);
    p.depth_writes = depth_writes;
    p.stencil_writes = stencil_writes;
    p.variant = [](std::uint32_t flags) {
        g.variant_requests.push_back(flags);
        return reinterpret_cast<VkPipeline>(std::uintptr_t{0x1000u + flags});
    };
    return p;
}

leonida::CommandList draw_list(std::uint32_t dsv_flags) {
    leonida::CommandList list;
    leonida::Draw d{};
    d.pipeline = 1;
    d.count = 3;
    d.instances = 1;
    d.viewport = {0.0f, 0.0f, 16.0f, 16.0f, 0.0f, 1.0f};
    d.scissor = {0, 0, 16, 16};
    d.depth = {1, 0, 0, 1, leonida::ColorViewFormat::resource, dsv_flags};
    list.draw(d);
    list.close();
    return list;
}

}  // namespace

int main() {
    using namespace leonida;
    unsigned checks = 0;

    // 1. Combined D24S8 with both aspects read-only (separate = true and false): stays in DEPTH_STENCIL_READ_ONLY,
    // no layout transition when already there, STORE_OP_NONE on both, no write variant if PSO already disables writes.
    for (bool separate : {true, false}) {
        auto b = depth_binding(true, false);
        auto pso = make_pso(true, false, false);
        auto vk = make_vk(separate, &pso);
        LayoutTracker tracker(separate);
        tracker.track(1, b, VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL);
        auto list = draw_list(dsv_read_only_depth | dsv_read_only_stencil);
        g.reset();
        EXPECT(replay(list, VK_NULL_HANDLE, [&](auto) { return &b; }, vk, &tracker) == ReplayResult::success);
        EXPECT(g.saw_rendering && g.has_depth && g.has_stencil);
        EXPECT(g.image_barriers.empty() && g.memory_barriers == 0);
        EXPECT(g.depth.imageLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL);
        EXPECT(g.stencil.imageLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL);
        EXPECT(g.depth.storeOp == VK_ATTACHMENT_STORE_OP_NONE && g.stencil.storeOp == VK_ATTACHMENT_STORE_OP_NONE);
        EXPECT(g.variant_requests.empty() && g.bound_pipeline == pso.pipeline);
        ++checks;
    }

    // 2. Mixed READ_ONLY_DEPTH + writable stencil with separate = true: depth stays READ_ONLY (no barrier) with
    // STORE_OP_NONE, stencil stays ATTACHMENT_OPTIMAL with STORE_OP_STORE, and write-enabled PSO requests read_only_depth.
    {
        auto b = depth_binding(true, false);
        auto pso = make_pso(true, true, true);
        auto vk = make_vk(true, &pso);
        LayoutTracker tracker(true);
        tracker.track(1, b, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
        tracker.set(1, {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1}, VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL);
        auto list = draw_list(dsv_read_only_depth);
        g.reset();
        EXPECT(replay(list, VK_NULL_HANDLE, [&](auto) { return &b; }, vk, &tracker) == ReplayResult::success);
        EXPECT(g.saw_rendering && g.has_depth && g.has_stencil);
        EXPECT(g.image_barriers.empty());
        EXPECT(g.depth.imageLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL);
        EXPECT(g.stencil.imageLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
        EXPECT(g.depth.storeOp == VK_ATTACHMENT_STORE_OP_NONE);
        EXPECT(g.stencil.storeOp == VK_ATTACHMENT_STORE_OP_STORE);
        EXPECT(g.variant_requests.size() == 1 && g.variant_requests[0] == GraphicsPipeline::read_only_depth);
        ++checks;
    }

    // 3. Mixed writable depth + READ_ONLY_STENCIL with separate = true.
    {
        auto b = depth_binding(true, false);
        auto pso = make_pso(true, true, true);
        auto vk = make_vk(true, &pso);
        LayoutTracker tracker(true);
        tracker.track(1, b, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
        tracker.set(1, {VK_IMAGE_ASPECT_STENCIL_BIT, 0, 1, 0, 1}, VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL);
        auto list = draw_list(dsv_read_only_stencil);
        g.reset();
        EXPECT(replay(list, VK_NULL_HANDLE, [&](auto) { return &b; }, vk, &tracker) == ReplayResult::success);
        EXPECT(g.image_barriers.empty());
        EXPECT(g.depth.imageLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
        EXPECT(g.stencil.imageLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL);
        EXPECT(g.depth.storeOp == VK_ATTACHMENT_STORE_OP_STORE);
        EXPECT(g.stencil.storeOp == VK_ATTACHMENT_STORE_OP_NONE);
        EXPECT(g.variant_requests.size() == 1 && g.variant_requests[0] == GraphicsPipeline::read_only_stencil);
        ++checks;
    }

    // 4. Mixed read-only with separate = false uses combined Vulkan 1.1 maintenance2 layouts.
    for (const bool ro_depth : {true, false}) {
        auto b = depth_binding(true, false);
        auto pso = make_pso(true, true, true);
        auto vk = make_vk(false, &pso);
        LayoutTracker tracker(false);
        tracker.track(1, b, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
        auto list = draw_list(ro_depth ? dsv_read_only_depth : dsv_read_only_stencil);
        g.reset();
        EXPECT(replay(list, VK_NULL_HANDLE, [&](auto) { return &b; }, vk, &tracker) == ReplayResult::success);
        const VkImageLayout expected = ro_depth ? VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_STENCIL_ATTACHMENT_OPTIMAL
                                                : VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_STENCIL_READ_ONLY_OPTIMAL;
        EXPECT(g.image_barriers.size() == 1 && g.image_barriers[0].newLayout == expected);
        EXPECT(g.depth.imageLayout == expected && g.stencil.imageLayout == expected);
        EXPECT(g.depth.storeOp == (ro_depth ? VK_ATTACHMENT_STORE_OP_NONE : VK_ATTACHMENT_STORE_OP_STORE));
        EXPECT(g.stencil.storeOp == (ro_depth ? VK_ATTACHMENT_STORE_OP_STORE : VK_ATTACHMENT_STORE_OP_NONE));
        ++checks;
    }

    // 5. sampled_in_general = true (bindless set 0): read-only depth stays in GENERAL so sampled descriptors match.
    {
        auto b = depth_binding(false, true);
        auto pso = make_pso(false, true, false);
        auto vk = make_vk(true, &pso);
        LayoutTracker tracker(true);
        tracker.track(1, b, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
        CommandList list;
        EXPECT(list.transition({1, 0, d3d12_state::depth_write,
                                d3d12_state::depth_read | d3d12_state::pixel_shader_resource}));
        Draw d{};
        d.pipeline = 1;
        d.count = 3;
        d.instances = 1;
        d.viewport = {0.0f, 0.0f, 16.0f, 16.0f, 0.0f, 1.0f};
        d.scissor = {0, 0, 16, 16};
        d.depth = {1, 0, 0, 1, ColorViewFormat::resource, dsv_read_only_depth};
        EXPECT(list.draw(d) && list.close());
        g.reset();
        EXPECT(replay(list, VK_NULL_HANDLE, [&](auto) { return &b; }, vk, &tracker) == ReplayResult::success);
        // Only the TransitionBarrier runs (DEPTH_STENCIL_ATTACHMENT -> GENERAL); the draw needs no layout change.
        EXPECT(g.image_barriers.size() == 1 && g.image_barriers[0].newLayout == VK_IMAGE_LAYOUT_GENERAL);
        EXPECT(g.depth.imageLayout == VK_IMAGE_LAYOUT_GENERAL);
        EXPECT(g.depth.storeOp == VK_ATTACHMENT_STORE_OP_NONE);
        EXPECT(g.variant_requests.size() == 1 && g.variant_requests[0] == GraphicsPipeline::read_only_depth);
        ++checks;
    }

    // 6. ClearDepthStencil skips read-only aspects and clears only writable ones.
    {
        auto b = depth_binding(true, false);
        auto pso = make_pso(true, false, false);
        auto vk = make_vk(true, &pso);
        CommandList list;
        EXPECT(list.clear_depth_stencil({1, 0, 0, 1, ColorViewFormat::resource, dsv_read_only_depth | dsv_read_only_stencil},
                                        clear_depth | clear_stencil, 0.5f, 0x11));
        EXPECT(list.clear_depth_stencil({1, 0, 0, 1, ColorViewFormat::resource, dsv_read_only_depth},
                                        clear_depth | clear_stencil, 0.5f, 0x22));
        EXPECT(list.close());
        g.reset();
        EXPECT(replay(list, VK_NULL_HANDLE, [&](auto) { return &b; }, vk) == ReplayResult::success);
        EXPECT(g.clear_depth_calls == 1 && g.cleared_aspects == VK_IMAGE_ASPECT_STENCIL_BIT);
        ++checks;
    }

    // 7. Draw-time transition of a read-only DSV to GENERAL (both separate and coupled fallback) must include
    // VK_ACCESS_SHADER_READ_BIT and VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT so shaders sampling it are synchronized.
    for (bool separate : {true, false}) {
        auto b = depth_binding(true, true);
        auto pso = make_pso(true, true, true);
        auto vk = make_vk(separate, &pso);
        vk.separate_depth_stencil_attachment_access = true;
        LayoutTracker tracker(separate);
        tracker.track(1, b, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
        auto list = draw_list(dsv_read_only_depth);
        g.reset();
        EXPECT(replay(list, VK_NULL_HANDLE, [&](auto) { return &b; }, vk, &tracker) == ReplayResult::success);
        EXPECT(g.image_barriers.size() == 1);
        EXPECT(g.image_barriers[0].newLayout == VK_IMAGE_LAYOUT_GENERAL);
        EXPECT((g.image_barriers[0].dstAccessMask & VK_ACCESS_SHADER_READ_BIT) != 0);
        EXPECT((g.image_barriers[0].dstAccessMask & VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT) != 0);
        EXPECT((g.image_barrier_dst_stages[0] & VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT) != 0);
        ++checks;
    }

    // 8. Packed depth/stencil sibling STORE vs GENERAL sampling: without separateDepthStencilAttachmentAccess, a mixed
    // read-only draw whose writable sibling writes keeps the read-only aspect out of GENERAL; when the sibling does
    // not write (STORE_OP_NONE) or separateDepthStencilAttachmentAccess is true, GENERAL is allowed.
    {
        auto b = depth_binding(true, true);
        auto pso_writes_s = make_pso(true, true, true);
        auto pso_no_s = make_pso(true, true, false);
        auto list = draw_list(dsv_read_only_depth);

        // No maintenance7, stencil writes -> depth stays out of GENERAL.
        {
            auto vk = make_vk(true, &pso_writes_s);
            vk.separate_depth_stencil_attachment_access = false;
            LayoutTracker tracker(true);
            tracker.track(1, b, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
            g.reset();
            EXPECT(replay(list, VK_NULL_HANDLE, [&](auto) { return &b; }, vk, &tracker) == ReplayResult::success);
            EXPECT(g.depth.imageLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL);
            EXPECT(g.depth.storeOp == VK_ATTACHMENT_STORE_OP_NONE);
            EXPECT(g.stencil.storeOp == VK_ATTACHMENT_STORE_OP_STORE);
        }
        for (const bool separate_layouts : {false, true}) {
            auto vk = make_vk(separate_layouts, &pso_writes_s);
            vk.separate_depth_stencil_attachment_access = false;
            LayoutTracker tracker(separate_layouts);
            b.sampled = true;
            const VkImageAspectFlags sampled_aspects = separate_layouts
                ? VkImageAspectFlags(VK_IMAGE_ASPECT_DEPTH_BIT)
                : b.aspects;
            tracker.track(1, b, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
            tracker.set(1, {sampled_aspects, 0, 1, 0, 1}, VK_IMAGE_LAYOUT_GENERAL);
            g.reset();
            EXPECT(replay(list, VK_NULL_HANDLE, [&](auto) { return &b; }, vk, &tracker) == ReplayResult::success);
            EXPECT(g.depth.imageLayout != VK_IMAGE_LAYOUT_GENERAL);
            EXPECT(tracker.layout(1, 0) == VK_IMAGE_LAYOUT_GENERAL);
            EXPECT(tracker.layout(1, 1) == (separate_layouts ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL
                                                            : VK_IMAGE_LAYOUT_GENERAL));
            EXPECT(g.image_barriers.size() >= 2);
            const auto& restore = g.image_barriers.back();
            EXPECT(restore.oldLayout == g.depth.imageLayout && restore.newLayout == VK_IMAGE_LAYOUT_GENERAL);
            EXPECT(restore.subresourceRange.aspectMask == sampled_aspects);
            EXPECT((restore.dstAccessMask & VK_ACCESS_SHADER_READ_BIT) != 0);
            b.sampled = false;
        }
        // With maintenance7, stencil writes -> depth stays in GENERAL for concurrent SRV sampling.
        {
            auto vk = make_vk(true, &pso_writes_s);
            vk.separate_depth_stencil_attachment_access = true;
            LayoutTracker tracker(true);
            tracker.track(1, b, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
            g.reset();
            EXPECT(replay(list, VK_NULL_HANDLE, [&](auto) { return &b; }, vk, &tracker) == ReplayResult::success);
            EXPECT(g.depth.imageLayout == VK_IMAGE_LAYOUT_GENERAL);
            EXPECT(g.depth.storeOp == VK_ATTACHMENT_STORE_OP_NONE);
            EXPECT(g.stencil.storeOp == VK_ATTACHMENT_STORE_OP_STORE);
        }
        // No maintenance7, stencil does NOT write -> stencil gets STORE_OP_NONE, so depth may use GENERAL safely.
        {
            auto vk = make_vk(true, &pso_no_s);
            vk.separate_depth_stencil_attachment_access = false;
            LayoutTracker tracker(true);
            tracker.track(1, b, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
            g.reset();
            EXPECT(replay(list, VK_NULL_HANDLE, [&](auto) { return &b; }, vk, &tracker) == ReplayResult::success);
            EXPECT(g.depth.imageLayout == VK_IMAGE_LAYOUT_GENERAL);
            EXPECT(g.depth.storeOp == VK_ATTACHMENT_STORE_OP_NONE);
            EXPECT(g.stencil.storeOp == VK_ATTACHMENT_STORE_OP_NONE);
        }
        ++checks;
    }

    // 9. Fully suppressed rectangular ClearDepthStencil skips rect_view (no image view allocated, no rendering hooks).
    {
        auto b = depth_binding(true, false);
        auto pso = make_pso(true, false, false);
        auto vk = make_vk(true, &pso);
        vk.begin_rendering = nullptr;
        vk.end_rendering = nullptr;
        vk.clear_attachments = nullptr;
        CommandList list;
        const Rect rect{0, 0, 8, 8};
        EXPECT(list.clear_depth_stencil({1, 0, 0, 1, ColorViewFormat::resource, dsv_read_only_depth | dsv_read_only_stencil},
                                        clear_depth | clear_stencil, 0.5f, 0x11, {&rect, 1}));
        EXPECT(list.close());
        g.reset();
        EXPECT(replay(list, VK_NULL_HANDLE, [&](auto) { return &b; }, vk) == ReplayResult::success);
        EXPECT(g.image_view_calls == 0 && g.clear_depth_calls == 0);
        ++checks;
    }

    std::printf("dsv_readonly_replay_test: %u checks passed\n", checks);
    return 0;
}
