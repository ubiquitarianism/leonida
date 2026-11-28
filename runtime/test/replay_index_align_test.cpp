// CPU-only replay test for index buffer offset alignment in prepare_draw()
// (VUID-vkCmdBindIndexBuffer-offset-08783: 2-byte aligned for UINT16, 4-byte aligned for UINT32).
#include "../src/vulkan_replay.h"

#include <cstdint>
#include <cstdio>

#define EXPECT(cond)                                                                      \
    do {                                                                                  \
        if (!(cond)) {                                                                    \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);          \
            return 1;                                                                     \
        }                                                                                 \
    } while (0)

namespace {

unsigned bind_index_calls = 0;

VKAPI_ATTR void VKAPI_CALL nop_begin_rendering(VkCommandBuffer, const VkRenderingInfo*) {}
VKAPI_ATTR void VKAPI_CALL nop_end_rendering(VkCommandBuffer) {}
VKAPI_ATTR void VKAPI_CALL nop_bind_vb2(VkCommandBuffer, std::uint32_t, std::uint32_t, const VkBuffer*,
                                        const VkDeviceSize*, const VkDeviceSize*, const VkDeviceSize*) {}
VKAPI_ATTR void VKAPI_CALL track_bind_index(VkCommandBuffer, VkBuffer, VkDeviceSize, VkIndexType) {
    ++bind_index_calls;
}
VKAPI_ATTR void VKAPI_CALL nop_set_viewport(VkCommandBuffer, std::uint32_t, std::uint32_t, const VkViewport*) {}
VKAPI_ATTR void VKAPI_CALL nop_set_scissor(VkCommandBuffer, std::uint32_t, std::uint32_t, const VkRect2D*) {}
VKAPI_ATTR void VKAPI_CALL nop_set_topology(VkCommandBuffer, VkPrimitiveTopology) {}
VKAPI_ATTR void VKAPI_CALL nop_set_restart(VkCommandBuffer, VkBool32) {}
VKAPI_ATTR void VKAPI_CALL nop_set_stencil_ref(VkCommandBuffer, VkStencilFaceFlags, std::uint32_t) {}
VKAPI_ATTR void VKAPI_CALL nop_set_blend(VkCommandBuffer, const float[4]) {}
VKAPI_ATTR void VKAPI_CALL nop_draw(VkCommandBuffer, std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t) {}
VKAPI_ATTR void VKAPI_CALL nop_draw_indexed(VkCommandBuffer, std::uint32_t, std::uint32_t, std::uint32_t, std::int32_t,
                                            std::uint32_t) {}
VKAPI_ATTR void VKAPI_CALL nop_bind_pipeline(VkCommandBuffer, VkPipelineBindPoint, VkPipeline) {}
VKAPI_ATTR void VKAPI_CALL nop_push_constants(VkCommandBuffer, VkPipelineLayout, VkShaderStageFlags, std::uint32_t,
                                              std::uint32_t, const void*) {}

}  // namespace

int main() {
    using namespace leonida;

    GraphicsPipeline gp{};
    gp.pipeline = reinterpret_cast<VkPipeline>(std::uintptr_t{1});

    ReplayFunctions vk{};
    vk.graphics_pipeline = [&](std::uint64_t) -> const GraphicsPipeline* { return &gp; };
    vk.image_view = [](const ResourceBinding&, const VkImageSubresourceRange&) -> VkImageView {
        return reinterpret_cast<VkImageView>(std::uintptr_t{1});
    };
    vk.begin_rendering = nop_begin_rendering;
    vk.end_rendering = nop_end_rendering;
    vk.bind_vertex_buffers2 = nop_bind_vb2;
    vk.bind_index_buffer = track_bind_index;
    vk.set_viewport = nop_set_viewport;
    vk.set_scissor = nop_set_scissor;
    vk.set_primitive_topology = nop_set_topology;
    vk.set_primitive_restart_enable = nop_set_restart;
    vk.set_stencil_reference = nop_set_stencil_ref;
    vk.set_blend_constants = nop_set_blend;
    vk.draw = nop_draw;
    vk.draw_indexed = nop_draw_indexed;
    vk.bind_pipeline = nop_bind_pipeline;
    vk.push_constants = nop_push_constants;
    vk.buffer_at = [](std::uint64_t va, std::uint64_t&, VkBuffer& buf, VkDeviceSize& off) {
        buf = reinterpret_cast<VkBuffer>(std::uintptr_t{2});
        off = va;
        return true;
    };

    const auto no_resolve = [](ResourceId) -> const ResourceBinding* { return nullptr; };

    auto run_indexed = [&](std::uint32_t format, std::uint64_t offset, ReplayResult expected) -> int {
        bind_index_calls = 0;
        CommandList list;
        Draw d{};
        d.pipeline = 1;
        d.indexed = true;
        d.count = 3;
        d.instances = 1;
        d.index.va = offset;
        d.index.size = 64;
        d.index.format = format;
        EXPECT(list.draw(d));
        EXPECT(list.close());
        EXPECT(replay(list, VK_NULL_HANDLE, no_resolve, vk) == expected);
        EXPECT(bind_index_calls == (expected == ReplayResult::success ? 1u : 0u));
        return 0;
    };

    // DXGI_FORMAT_R16_UINT (57): 2-byte aligned offset succeeds; odd offset fails before vkCmdBindIndexBuffer.
    if (run_indexed(57, 2, ReplayResult::success)) return 1;
    if (run_indexed(57, 4, ReplayResult::success)) return 1;
    if (run_indexed(57, 1, ReplayResult::invalid_draw)) return 1;
    if (run_indexed(57, 3, ReplayResult::invalid_draw)) return 1;

    // DXGI_FORMAT_R32_UINT (42): 4-byte aligned offset succeeds; 2-byte and odd offsets fail before vkCmdBindIndexBuffer.
    if (run_indexed(42, 4, ReplayResult::success)) return 1;
    if (run_indexed(42, 8, ReplayResult::success)) return 1;
    if (run_indexed(42, 2, ReplayResult::invalid_draw)) return 1;
    if (run_indexed(42, 1, ReplayResult::invalid_draw)) return 1;

    // Unbound index buffer (va == 0) using zero_buffer with misaligned zero_offset is also rejected.
    vk.zero_buffer = reinterpret_cast<VkBuffer>(std::uintptr_t{3});
    vk.zero_offset = 2;
    if (run_indexed(42, 0, ReplayResult::invalid_draw)) return 1;
    vk.zero_offset = 4;
    if (run_indexed(42, 0, ReplayResult::success)) return 1;

    std::printf("replay_index_align: 10/10 cases OK\n");
    return 0;
}
