// Quad lists are triangle-list draws. A switch back to Vulkan triangle fans changes the provoking vertex for flat
// attributes and the order of overlapping blended instances.
//   - Without an index buffer: one draw over the shared triangle indices.
//   - With one: one compute dispatch that expands the application's indices into scratch (src/quad_index_expand.h),
//     then one draw over that scratch. The dispatch runs before the draw's rendering scope and outside a
//     pipeline-statistics query, reads no index past the view, and a draw with more quads than one dispatch may
//     cover is split into several.
// No GPU: the Vulkan calls are recorded.
#include "../src/vulkan_replay.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace {
using namespace leonida;

struct DrawCall {
    VkPrimitiveTopology topology;
    VkBuffer buffer;
    VkDeviceSize offset;
    VkIndexType index_type;
    std::uint32_t count, instances, first_index;
    std::int32_t base_vertex;
    std::uint32_t first_instance;
    bool operator==(const DrawCall&) const = default;
};

// What quad_index_expand.comp reads from its push constants.
struct Push {
    VkDeviceAddress source, triangles;
    std::uint32_t first_half, index_bits, limit, quads;
    bool operator==(const Push&) const = default;
};

struct Barrier {
    VkPipelineStageFlags src_stages, dst_stages;
    VkAccessFlags src_access, dst_access;
    bool operator==(const Barrier&) const = default;
};

DrawCall recorded{};
std::vector<DrawCall> calls;
std::vector<Push> pushes;
std::vector<std::uint32_t> dispatches;  // group counts
std::vector<Barrier> barriers;
std::vector<std::string> events;  // the order of everything recorded
std::vector<VkDeviceSize> scratch_requests;

template<class T>
T handle(std::uintptr_t n) {
    return reinterpret_cast<T>(n);
}

bool check(bool condition, const char* description) {
    if (!condition) std::fprintf(stderr, "FAIL: %s\n", description);
    return condition;
}

// The device the expander sees: every buffer's address is 0x1000 times its handle.
VKAPI_ATTR VkDeviceAddress VKAPI_CALL device_address(VkDevice, const VkBufferDeviceAddressInfo* info) {
    return VkDeviceAddress(reinterpret_cast<std::uintptr_t>(info->buffer)) * 0x1000;
}
VKAPI_ATTR VkResult VKAPI_CALL create_module(VkDevice, const VkShaderModuleCreateInfo* info,
                                             const VkAllocationCallbacks*, VkShaderModule* out) {
    *out = handle<VkShaderModule>(20);
    return info->codeSize && info->pCode[0] == 0x07230203u ? VK_SUCCESS : VK_ERROR_INITIALIZATION_FAILED;
}
VKAPI_ATTR void VKAPI_CALL destroy_module(VkDevice, VkShaderModule, const VkAllocationCallbacks*) {}
VKAPI_ATTR VkResult VKAPI_CALL create_layout(VkDevice, const VkPipelineLayoutCreateInfo* info,
                                             const VkAllocationCallbacks*, VkPipelineLayout* out) {
    *out = handle<VkPipelineLayout>(21);
    return info->pushConstantRangeCount == 1 && info->pPushConstantRanges[0].size == sizeof(Push)
               ? VK_SUCCESS : VK_ERROR_INITIALIZATION_FAILED;
}
VKAPI_ATTR void VKAPI_CALL destroy_layout(VkDevice, VkPipelineLayout, const VkAllocationCallbacks*) {}
VKAPI_ATTR VkResult VKAPI_CALL create_pipelines(VkDevice, VkPipelineCache, std::uint32_t,
                                                const VkComputePipelineCreateInfo*, const VkAllocationCallbacks*,
                                                VkPipeline* out) {
    *out = handle<VkPipeline>(22);
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL destroy_pipeline(VkDevice, VkPipeline, const VkAllocationCallbacks*) {}
VKAPI_ATTR void VKAPI_CALL cmd_barrier(VkCommandBuffer, VkPipelineStageFlags src, VkPipelineStageFlags dst,
                                       VkDependencyFlags, std::uint32_t count, const VkMemoryBarrier* memory,
                                       std::uint32_t, const VkBufferMemoryBarrier*, std::uint32_t,
                                       const VkImageMemoryBarrier*) {
    barriers.push_back({src, dst, count == 1 ? memory->srcAccessMask : 0, count == 1 ? memory->dstAccessMask : 0});
    events.push_back("barrier");
}
VKAPI_ATTR void VKAPI_CALL cmd_bind_pipeline(VkCommandBuffer, VkPipelineBindPoint point, VkPipeline) {
    events.push_back(point == VK_PIPELINE_BIND_POINT_COMPUTE ? "bind_compute" : "bind_graphics");
}
VKAPI_ATTR void VKAPI_CALL cmd_push_constants(VkCommandBuffer, VkPipelineLayout, VkShaderStageFlags stages,
                                              std::uint32_t, std::uint32_t size, const void* values) {
    if (stages != VK_SHADER_STAGE_COMPUTE_BIT || size != sizeof(Push)) return;  // the draw's own user data
    std::memcpy(&pushes.emplace_back(), values, sizeof(Push));
}
VKAPI_ATTR void VKAPI_CALL cmd_dispatch(VkCommandBuffer, std::uint32_t x, std::uint32_t y, std::uint32_t z) {
    dispatches.push_back(y == 1 && z == 1 ? x : 0);
    events.push_back("dispatch");
}
VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL device_proc(VkDevice, const char* name) {
#define PROC(vk_name, function) \
    if (std::strcmp(name, vk_name) == 0) return reinterpret_cast<PFN_vkVoidFunction>(function)
    PROC("vkGetBufferDeviceAddress", device_address);
    PROC("vkCreateShaderModule", create_module);
    PROC("vkDestroyShaderModule", destroy_module);
    PROC("vkCreatePipelineLayout", create_layout);
    PROC("vkDestroyPipelineLayout", destroy_layout);
    PROC("vkCreateComputePipelines", create_pipelines);
    PROC("vkDestroyPipeline", destroy_pipeline);
    PROC("vkCmdPipelineBarrier", cmd_barrier);
    PROC("vkCmdBindPipeline", cmd_bind_pipeline);
    PROC("vkCmdPushConstants", cmd_push_constants);
    PROC("vkCmdDispatch", cmd_dispatch);
#undef PROC
    return nullptr;
}

VKAPI_ATTR void VKAPI_CALL begin_query(VkCommandBuffer, VkQueryPool, std::uint32_t index, VkQueryControlFlags) {
    events.push_back("begin_query " + std::to_string(index));
}
VKAPI_ATTR void VKAPI_CALL end_query(VkCommandBuffer, VkQueryPool, std::uint32_t index) {
    events.push_back("end_query " + std::to_string(index));
}
VKAPI_ATTR void VKAPI_CALL destroy_query_pool(VkDevice, VkQueryPool, const VkAllocationCallbacks*) {}

void reset() {
    calls.clear();
    pushes.clear();
    dispatches.clear();
    barriers.clear();
    events.clear();
    scratch_requests.clear();
}
}

int main() {
    using namespace leonida;
    ReplayFunctions vk{};
    vk.begin_rendering = +[](VkCommandBuffer, const VkRenderingInfo*) { events.push_back("begin_rendering"); };
    vk.end_rendering = +[](VkCommandBuffer) { events.push_back("end_rendering"); };
    vk.bind_pipeline = cmd_bind_pipeline;
    vk.push_constants = cmd_push_constants;
    vk.set_viewport = +[](VkCommandBuffer, std::uint32_t, std::uint32_t, const VkViewport*) {};
    vk.set_scissor = +[](VkCommandBuffer, std::uint32_t, std::uint32_t, const VkRect2D*) {};
    vk.set_primitive_topology = +[](VkCommandBuffer, VkPrimitiveTopology topology) { recorded.topology = topology; };
    vk.set_primitive_restart_enable = +[](VkCommandBuffer, VkBool32 enable) {
        if (enable) events.push_back("primitive_restart");
    };
    vk.set_stencil_reference = +[](VkCommandBuffer, VkStencilFaceFlags, std::uint32_t) {};
    vk.set_blend_constants = +[](VkCommandBuffer, const float*) {};
    vk.bind_index_buffer = +[](VkCommandBuffer, VkBuffer buffer, VkDeviceSize offset, VkIndexType type) {
        recorded.buffer = buffer;
        recorded.offset = offset;
        recorded.index_type = type;
    };
    vk.draw_indexed = +[](VkCommandBuffer, std::uint32_t count, std::uint32_t instances, std::uint32_t first,
                         std::int32_t base, std::uint32_t first_instance) {
        recorded.count = count;
        recorded.instances = instances;
        recorded.first_index = first;
        recorded.base_vertex = base;
        recorded.first_instance = first_instance;
        calls.push_back(recorded);
        events.push_back("draw_indexed");
    };

    vk.quad_indices = handle<VkBuffer>(1);
    vk.quad_indices_offset = 64;
    vk.quad_index_count = 12;
    const auto app_buffer = handle<VkBuffer>(2);

    GraphicsPipeline graphics{};
    graphics.pipeline = handle<VkPipeline>(3);
    graphics.strip_cut = 1;  // 0xFFFF cuts strips: it must not make a quad list restart
    const auto collect = [&](const Draw& draw) {
        reset();
        CommandList list;
        if (!list.draw(draw) || !list.close()) return false;
        PreparedList prepared;
        prepared.list = &list;
        prepared.commands.resize(1);
        prepared.commands[0].draw = 0;
        auto& entry = prepared.draws.emplace_back();
        entry.graphics = &graphics;
        entry.index_buffer = app_buffer;
        entry.index_offset = 32;
        emit(prepared, VK_NULL_HANDLE, vk);
        return true;
    };

    Draw draw{};
    draw.pipeline = 1;
    draw.topology = topology_quad_list;
    draw.count = 8;
    draw.instances = 2;
    draw.first = 13;
    draw.first_instance = 7;

    const DrawCall triangles{VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, vk.quad_indices, 64, VK_INDEX_TYPE_UINT32,
                             12, 2, 0, 13, 7};
    bool ok = check(collect(draw) && calls == std::vector{triangles},
                    "non-indexed quads use one triangle-list draw and preserve draw bases");
    ok &= check(dispatches.empty(), "non-indexed quads need no index expansion");
    draw.count = 13;
    ok &= check(collect(draw) && calls == std::vector{triangles},
                "a draw is limited to whole quads in the generated index buffer");
    draw.count = 3;
    ok &= check(collect(draw) && calls.empty(), "an incomplete quad emits no draw");

    // Indexed quads, through prepare(): the real expander on the recording device above.
    VkPhysicalDeviceLimits limits{};
    limits.maxComputeWorkGroupCount[0] = 65535;
    QuadIndexExpander expander(device_proc, handle<VkDevice>(9), limits);
    const auto scratch = handle<VkBuffer>(5), zero_page = handle<VkBuffer>(6);
    constexpr VkDeviceSize scratch_offset = 512;
    std::uint64_t app_bytes = 64;  // the application's buffer past the view's start
    vk.graphics_pipeline = [&](std::uintptr_t) -> const GraphicsPipeline* { return &graphics; };
    vk.buffer_at = [&](std::uint64_t va, std::uint64_t& size, VkBuffer& buffer, VkDeviceSize& offset) {
        buffer = app_buffer;
        offset = va;  // the console VA is the offset in this test
        size = std::min(size, app_bytes);
        return true;
    };
    vk.image_view = [](const ResourceBinding&, const VkImageSubresourceRange&) { return VkImageView{}; };
    vk.bind_vertex_buffers2 = +[](VkCommandBuffer, std::uint32_t, std::uint32_t, const VkBuffer*,
                                  const VkDeviceSize*, const VkDeviceSize*, const VkDeviceSize*) {};
    vk.draw = +[](VkCommandBuffer, std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t) {};
    vk.zero_buffer = zero_page;
    vk.zero_size = 65536;
    vk.copy_scratch = [&](VkDeviceSize bytes, VkBuffer& buffer, VkDeviceSize& offset) {
        scratch_requests.push_back(bytes);
        buffer = scratch;
        offset = scratch_offset;
        return true;
    };
    const auto wire = [](ReplayFunctions& functions, QuadIndexExpander& to) {
        functions.prepare_quad_indices = [&to](VkBuffer indices, VkDeviceSize offset, VkIndexType type,
                                               std::uint64_t readable, std::uint32_t quads, VkBuffer target,
                                               VkDeviceSize target_offset, QuadIndexExpansion& out) {
            return to.prepare(indices, offset, type, readable, quads, target, target_offset, out);
        };
        functions.emit_quad_indices = [&to](VkCommandBuffer cb, const QuadIndexExpansion& job) { to.emit(cb, job); };
    };
    wire(vk, expander);
    const auto resolve = [](ResourceId) -> const ResourceBinding* { return nullptr; };
    const auto run = [&](const Draw& indexed, const ReplayFunctions& functions) {
        reset();
        CommandList list;
        if (!list.draw(indexed) || !list.close()) return ReplayResult::list_open;
        PreparedList prepared;
        const auto result = prepare(list, resolve, functions, prepared);
        if (result == ReplayResult::success) emit(prepared, VK_NULL_HANDLE, functions);
        return result;
    };

    constexpr VkDeviceAddress app_address = 2 * 0x1000, scratch_address = 5 * 0x1000 + scratch_offset;
    const Barrier before{VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT};
    const Barrier after{VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,
                        VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_INDEX_READ_BIT};
    const std::vector<std::string> one_dispatch_one_draw{
        "barrier", "bind_compute", "dispatch", "barrier", "begin_rendering", "bind_graphics", "draw_indexed",
        "end_rendering"};

    // 16-bit indices: two quads and a ninth index, from index 5 of a view at byte 32 (the first index is at byte 42,
    // the high half of the word at 40). The view holds 32 indices, 27 of them from index 5 on.
    draw.indexed = true;
    draw.index = {32, 64, 57};  // DXGI_FORMAT_R16_UINT
    draw.count = 9;
    draw.first = 5;
    draw.base_vertex = -2;
    const DrawCall expanded{VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, scratch, scratch_offset, VK_INDEX_TYPE_UINT32,
                            12, 2, 0, -2, 7};
    ok &= check(run(draw, vk) == ReplayResult::success && events == one_dispatch_one_draw,
                "indexed quads are one dispatch before the rendering scope and one draw inside it");
    ok &= check(calls == std::vector{expanded},
                "indexed quads draw the expanded indices once, with the draw's instances and bases");
    ok &= check(dispatches == std::vector<std::uint32_t>{1} && scratch_requests == std::vector<VkDeviceSize>{48},
                "two quads are one work group and 48 bytes of scratch");
    ok &= check(pushes == std::vector{Push{app_address + 40, scratch_address, 1, 16, 27, 2}},
                "16-bit indices are read from the word that holds the first one, up to the view's end");
    ok &= check(barriers == std::vector{before, after},
                "earlier writes are visible to the pass, and its indices to the draw");

    // 32-bit indices through a view of six: the last two of the second quad read 0.
    draw.index = {64, 24, 42};  // DXGI_FORMAT_R32_UINT
    draw.count = 8;
    draw.first = 0;
    ok &= check(run(draw, vk) == ReplayResult::success && events == one_dispatch_one_draw &&
                    pushes == std::vector{Push{app_address + 64, scratch_address, 0, 32, 6, 2}},
                "32-bit indices past the view are not read");
    // A view that runs past its buffer ends where the buffer does.
    draw.index = {64, 4096, 42};
    app_bytes = 20;
    ok &= check(run(draw, vk) == ReplayResult::success && pushes.size() == 1 && pushes[0].limit == 5,
                "a view is cut at the end of its buffer");
    app_bytes = 64;
    // The first index past the view, and no index buffer at all: every index reads 0.
    draw.index = {64, 24, 42};
    draw.first = 6;
    ok &= check(run(draw, vk) == ReplayResult::success && calls.size() == 1 && pushes.size() == 1 &&
                    pushes[0].limit == 0 && pushes[0].source == app_address + 64,
                "a draw that starts past the view reads no index");
    draw.index = {0, 0, 57};
    draw.first = 0;
    ok &= check(run(draw, vk) == ReplayResult::success && calls.size() == 1 && pushes.size() == 1 &&
                    pushes[0].limit == 0 && pushes[0].source == 6 * 0x1000,
                "without an index buffer every index reads 0");

    // Fewer than four indices: nothing to expand, nothing to draw.
    draw.index = {32, 64, 57};
    draw.count = 3;
    ok &= check(run(draw, vk) == ReplayResult::success && calls.empty() && dispatches.empty() &&
                    scratch_requests.empty(),
                "an incomplete indexed quad needs no scratch and emits no draw");

    // What cannot be expanded is refused when the list is prepared.
    draw.count = 8;
    {
        ReplayFunctions without = vk;
        without.prepare_quad_indices = {};
        ok &= check(run(draw, without) == ReplayResult::unsupported && events.empty(),
                    "indexed quads are unsupported without the expander");
        without = vk;
        without.copy_scratch = [](VkDeviceSize, VkBuffer&, VkDeviceSize&) { return false; };
        ok &= check(run(draw, without) == ReplayResult::unsupported && events.empty(),
                    "indexed quads are unsupported without scratch");
        Draw odd = draw;
        odd.index = {33, 64, 57};
        ok &= check(run(odd, vk) == ReplayResult::unsupported && events.empty(),
                    "16-bit indices at an odd address are refused");
        Draw huge = draw;
        huge.count = UINT32_MAX;  // 6 indices for each quad would not fit vkCmdDrawIndexed's count
        ok &= check(run(huge, vk) == ReplayResult::unsupported && scratch_requests.empty(),
                    "a draw whose expanded index count overflows 32 bits is refused before taking scratch");
    }

    // More quads than one dispatch may cover: 300 quads with two work groups (128 quads) for each dispatch. Every
    // dispatch starts 512 indices (1024 bytes) and 128 * 24 bytes of scratch after the one before, and the view's
    // 1000 indices from the first one run out in the second.
    {
        VkPhysicalDeviceLimits small{};
        small.maxComputeWorkGroupCount[0] = 2;
        QuadIndexExpander chunked(device_proc, handle<VkDevice>(9), small);
        ReplayFunctions split = vk;
        wire(split, chunked);
        Draw many = draw;
        many.index = {32, 2010, 57};
        many.count = 1200;
        many.first = 5;
        app_bytes = 4096;
        const std::vector<Push> want{{app_address + 40, scratch_address, 1, 16, 1000, 128},
                                     {app_address + 40 + 1024, scratch_address + 128 * 24, 1, 16, 488, 128},
                                     {app_address + 40 + 2048, scratch_address + 256 * 24, 1, 16, 0, 44}};
        ok &= check(run(many, split) == ReplayResult::success && pushes == want &&
                        dispatches == std::vector<std::uint32_t>{2, 2, 1},
                    "a large draw is expanded by several dispatches that continue where the last one stopped");
        ok &= check(calls.size() == 1 && calls[0].count == 1800 && scratch_requests == std::vector<VkDeviceSize>{7200},
                    "a large draw is still one draw over one piece of scratch");
        app_bytes = 64;
    }

    // Inside a pipeline-statistics query the dispatch must not be counted (CSInvocations): the query's segment ends
    // before it and a new one begins after it, for the draw. Without a segment pool the list is refused.
    {
        QueryPool heap{handle<VkQueryPool>(7), d3d12_query_heap::pipeline_statistics, 1, false, {false}, {}};
        ReplayFunctions queries = vk;
        queries.reset_query_pool = +[](VkCommandBuffer, VkQueryPool, std::uint32_t, std::uint32_t) {};
        queries.begin_query = begin_query;
        queries.end_query = end_query;
        queries.write_timestamp = +[](VkCommandBuffer, VkPipelineStageFlagBits, VkQueryPool, std::uint32_t) {};
        queries.copy_query_pool_results = +[](VkCommandBuffer, VkQueryPool, std::uint32_t, std::uint32_t, VkBuffer,
                                              VkDeviceSize, VkDeviceSize, VkQueryResultFlags) {};
        queries.fill_buffer = +[](VkCommandBuffer, VkBuffer, VkDeviceSize, VkDeviceSize, std::uint32_t) {};
        queries.query_pool = [&](QueryHeapId) { return &heap; };
        reset();
        CommandList list;
        ok &= check(list.begin_query({6, d3d12_query::pipeline_statistics, 0}) && list.draw(draw) &&
                        list.end_query({6, d3d12_query::pipeline_statistics, 0}) && list.close(),
                    "record a quad draw inside a statistics query");
        PreparedList prepared;
        ok &= check(prepare(list, resolve, queries, prepared) == ReplayResult::unsupported &&
                        scratch_requests.empty(),
                    "without a segment pool the expansion inside a statistics query is refused before its scratch");
        ok &= check(prepare(list, resolve, queries, prepared, true) == ReplayResult::success &&
                        prepared.statistics_segment_count == 2,
                    "the expansion splits the statistics query in two segments");
        auto pool = std::make_shared<OcclusionSegmentPool>();
        pool->pool = handle<VkQueryPool>(8);
        pool->destroy = destroy_query_pool;
        QueryReplay plan;
        ok &= check(plan.plan(prepared, {}, 0, true, pool, 4) == ReplayResult::success, "plan the split query");
        emit(prepared, VK_NULL_HANDLE, queries);
        const std::vector<std::string> want{
            "begin_query 4", "end_query 4", "barrier", "bind_compute", "dispatch", "barrier", "begin_query 5",
            "begin_rendering", "bind_graphics", "draw_indexed", "end_rendering", "end_query 5"};
        ok &= check(events == want, "the dispatch runs between the query's two segments, the draw in the second");
    }
    return ok ? 0 : 1;
}
