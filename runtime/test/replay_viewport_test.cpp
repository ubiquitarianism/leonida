// The viewports and scissors emit() sets for a draw, on both dynamic-state paths: one viewport (a pipeline without
// multiViewport) and VK_DYNAMIC_STATE_VIEWPORT_WITH_COUNT.
//   - An empty D3D12 viewport (width or height 0) draws nothing. Vulkan forbids a zero-size viewport
//     (VUID-VkViewport-width-01770, VUID-VkViewport-height-01773), so it becomes a 1x1 viewport with an empty
//     scissor. On the one-viewport path viewport 0 can be empty while a later slot is not, and the draw is kept.
//   - Viewport depths are held to [0, 1] (VUID-VkViewport-minDepth-01234, VUID-VkViewport-maxDepth-01235), NaN to
//     the range's own end, unless VK_EXT_depth_range_unrestricted is on.
// No GPU: the Vulkan calls are recorded, and every recorded viewport is checked against those VUIDs.
#include "../src/vulkan_replay.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <memory>
#include <vector>

namespace {
using namespace leonida;

std::vector<VkViewport> viewports;
std::vector<VkRect2D> scissors;
bool unrestricted = false;  // what the recorded viewports may hold
int vuid_failures = 0;

bool check(bool condition, const char* description) {
    if (!condition) std::fprintf(stderr, "FAIL: %s\n", description);
    return condition;
}

void record(std::uint32_t count, const VkViewport* v, const VkRect2D* s) {
    for (std::uint32_t i = 0; i < count; ++i) {
        if (v) {
            const VkViewport& x = v[i];
            const bool in_range = unrestricted || (x.minDepth >= 0.0f && x.minDepth <= 1.0f && x.maxDepth >= 0.0f &&
                                                   x.maxDepth <= 1.0f);
            if (!(x.width > 0.0f) || x.height == 0.0f || !in_range) {
                std::fprintf(stderr, "invalid VkViewport: x %g y %g w %g h %g depth [%g, %g]\n", x.x, x.y, x.width,
                             x.height, x.minDepth, x.maxDepth);
                ++vuid_failures;
            }
            viewports.push_back(x);
        }
        if (s) scissors.push_back(s[i]);
    }
}

bool same(const VkViewport& a, const VkViewport& b) {
    return a.x == b.x && a.y == b.y && a.width == b.width && a.height == b.height && a.minDepth == b.minDepth &&
           a.maxDepth == b.maxDepth;
}
bool same(const VkRect2D& a, const VkRect2D& b) {
    return a.offset.x == b.offset.x && a.offset.y == b.offset.y && a.extent.width == b.extent.width &&
           a.extent.height == b.extent.height;
}

const VkViewport dummy{0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f};
const VkRect2D empty_scissor{{0, 0}, {0, 0}};
}

int main() {
    ReplayFunctions vk{};
    vk.begin_rendering = +[](VkCommandBuffer, const VkRenderingInfo*) {};
    vk.end_rendering = +[](VkCommandBuffer) {};
    vk.bind_pipeline = +[](VkCommandBuffer, VkPipelineBindPoint, VkPipeline) {};
    vk.push_constants = +[](VkCommandBuffer, VkPipelineLayout, VkShaderStageFlags, std::uint32_t, std::uint32_t,
                            const void*) {};
    vk.set_viewport = +[](VkCommandBuffer, std::uint32_t first, std::uint32_t count, const VkViewport* v) {
        if (first != 0) ++vuid_failures;
        record(count, v, nullptr);
    };
    vk.set_scissor = +[](VkCommandBuffer, std::uint32_t first, std::uint32_t count, const VkRect2D* s) {
        if (first != 0) ++vuid_failures;
        record(count, nullptr, s);
    };
    vk.set_viewport_with_count = +[](VkCommandBuffer, std::uint32_t count, const VkViewport* v) {
        record(count, v, nullptr);
    };
    vk.set_scissor_with_count = +[](VkCommandBuffer, std::uint32_t count, const VkRect2D* s) {
        record(count, nullptr, s);
    };
    vk.set_primitive_topology = +[](VkCommandBuffer, VkPrimitiveTopology) {};
    vk.set_primitive_restart_enable = +[](VkCommandBuffer, VkBool32) {};
    vk.set_stencil_reference = +[](VkCommandBuffer, VkStencilFaceFlags, std::uint32_t) {};
    vk.set_blend_constants = +[](VkCommandBuffer, const float*) {};
    vk.draw = +[](VkCommandBuffer, std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t) {};

    GraphicsPipeline graphics{};
    graphics.pipeline = reinterpret_cast<VkPipeline>(std::uintptr_t(3));
    const auto collect = [&](const Draw& draw, bool with_count, bool depth_unrestricted) {
        viewports.clear();
        scissors.clear();
        vuid_failures = 0;
        graphics.viewport_with_count = with_count;
        vk.depth_range_unrestricted = depth_unrestricted;
        unrestricted = depth_unrestricted;
        CommandList list;
        if (!list.draw(draw) || !list.close()) return false;
        PreparedList prepared;
        prepared.list = &list;
        prepared.commands.resize(1);
        prepared.commands[0].draw = 0;
        prepared.draws.emplace_back().graphics = &graphics;
        emit(prepared, VK_NULL_HANDLE, vk);
        return true;
    };

    Draw draw{};
    draw.pipeline = 1;
    draw.topology = 4;  // D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST
    draw.count = 3;
    draw.instances = 1;
    draw.scissor = {0, 0, 64, 64};
    bool ok = true;

    // One viewport, empty (kevincommits' repro in #1608): a valid 1x1 viewport and an empty scissor, not 0x0 with the
    // 64x64 scissor still on.
    draw.viewport = {0.0f, 0.0f, 0.0f, 0.0f, -0.5f, 2.0f};
    ok &= check(collect(draw, false, false) && viewports.size() == 1 && scissors.size() == 1,
                "one viewport: an empty viewport still sets one viewport and one scissor");
    ok &= check(vuid_failures == 0, "one viewport: no zero-size viewport");
    ok &= check(viewports.size() == 1 && same(viewports[0], dummy) && scissors.size() == 1 &&
                    same(scissors[0], empty_scissor),
                "one viewport: an empty viewport is a 1x1 viewport with an empty scissor");
    draw.viewport = {8.0f, 8.0f, 32.0f, 0.0f, 0.0f, 1.0f};  // zero height alone
    ok &= check(collect(draw, false, false) && vuid_failures == 0 && viewports.size() == 1 &&
                    same(viewports[0], dummy) && same(scissors[0], empty_scissor),
                "one viewport: a zero-height viewport is empty too");

    // A drawn viewport keeps its rectangle and scissor; its depths are held to [0, 1].
    draw.viewport = {0.0f, 0.0f, 64.0f, 64.0f, -0.5f, 2.0f};
    const VkViewport held{0.0f, 64.0f, 64.0f, -64.0f, 0.0f, 1.0f};
    const VkRect2D full{{0, 0}, {64, 64}};
    ok &= check(collect(draw, false, false) && vuid_failures == 0 && viewports.size() == 1 &&
                    same(viewports[0], held) && same(scissors[0], full),
                "one viewport: depths outside [0, 1] are clamped");
    ok &= check(collect(draw, true, false) && vuid_failures == 0 && viewports.size() == 1 &&
                    same(viewports[0], held) && same(scissors[0], full),
                "with count: depths outside [0, 1] are clamped");
    const float nan = std::numeric_limits<float>::quiet_NaN();
    draw.viewport = {0.0f, 0.0f, 64.0f, 64.0f, nan, nan};
    ok &= check(collect(draw, false, false) && vuid_failures == 0 && viewports.size() == 1 &&
                    same(viewports[0], held),
                "one viewport: NaN depths become 0 and 1");
    ok &= check(collect(draw, true, false) && vuid_failures == 0 && viewports.size() == 1 &&
                    same(viewports[0], held),
                "with count: NaN depths become 0 and 1");
    draw.viewport = {0.0f, 0.0f, 64.0f, 64.0f, 0.25f, 0.75f};
    ok &= check(collect(draw, false, false) && viewports.size() == 1 &&
                    same(viewports[0], VkViewport{0.0f, 64.0f, 64.0f, -64.0f, 0.25f, 0.75f}),
                "depths inside [0, 1] are kept");

    // With VK_EXT_depth_range_unrestricted the depths go through as given.
    draw.viewport = {0.0f, 0.0f, 64.0f, 64.0f, -0.5f, 2.0f};
    const VkViewport kept{0.0f, 64.0f, 64.0f, -64.0f, -0.5f, 2.0f};
    ok &= check(collect(draw, false, true) && vuid_failures == 0 && viewports.size() == 1 &&
                    same(viewports[0], kept),
                "one viewport, unrestricted: depths are kept");
    ok &= check(collect(draw, true, true) && vuid_failures == 0 && viewports.size() == 1 && same(viewports[0], kept),
                "with count, unrestricted: depths are kept");

    // Several viewports, the first one empty (d3d12_multi_viewport_test's case): on the counted path slot 0 is the
    // dummy and slot 1 draws, with its depths held.
    auto several = std::make_shared<DrawViewports>();
    several->viewport_count = 2;
    several->scissor_count = 2;
    several->viewports[0] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f};
    several->viewports[1] = {0.0f, 0.0f, 64.0f, 64.0f, -1.0f, 3.0f};
    several->scissors[0] = {0, 0, 64, 64};
    several->scissors[1] = {0, 0, 64, 64};
    draw.viewport = several->viewports[0];
    draw.viewports = several;
    ok &= check(collect(draw, true, false) && vuid_failures == 0 && viewports.size() == 2 && scissors.size() == 2 &&
                    same(viewports[0], dummy) && same(scissors[0], empty_scissor) && same(viewports[1], held) &&
                    same(scissors[1], full),
                "with count: an empty slot 0 is the dummy, slot 1 draws with its depths held");
    ok &= check(collect(draw, false, false) && vuid_failures == 0 && viewports.size() == 1 &&
                    same(viewports[0], dummy) && same(scissors[0], empty_scissor),
                "one viewport: an empty viewport 0 with later slots set is still valid");

    return ok ? 0 : 1;
}
