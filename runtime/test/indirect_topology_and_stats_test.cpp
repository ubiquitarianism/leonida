// Portable CPU-only tests for stateful indirect draw topology coverage and per-reason indirect_stats reporting.
// No GPU, driver, or game data required.
#include "../src/vulkan_replay.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#define CHECK(cond)                                                                                           \
    do {                                                                                                      \
        if (!(cond)) {                                                                                        \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);                              \
            std::exit(1);                                                                                     \
        }                                                                                                     \
    } while (0)

namespace {

template <class T>
T fake_handle(std::uintptr_t value) {
    return reinterpret_cast<T>(value);
}

struct RecordedDraw {
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkPrimitiveTopology topology = VK_PRIMITIVE_TOPOLOGY_MAX_ENUM;
    VkBool32 primitive_restart = VK_FALSE;
    bool indexed = false;
    bool counted = false;
    std::uint32_t max_count = 0;
};

struct Harness;
Harness* g_harness = nullptr;

struct Harness {
    leonida::GraphicsPipeline pipeline{};
    leonida::ResourceBinding args{};
    leonida::ResourceBinding counts{};
    leonida::ReplayFunctions vk{};
    std::vector<RecordedDraw> draws;
    std::uint32_t last_variant_flags = 0;
    std::uint32_t variant_calls = 0;

    VkPipeline bound_pipeline = VK_NULL_HANDLE;
    VkPrimitiveTopology bound_topology = VK_PRIMITIVE_TOPOLOGY_MAX_ENUM;
    VkBool32 bound_restart = VK_FALSE;

    static VKAPI_ATTR void VKAPI_CALL cb_begin_rendering(VkCommandBuffer, const VkRenderingInfo*) {}
    static VKAPI_ATTR void VKAPI_CALL cb_end_rendering(VkCommandBuffer) {}
    static VKAPI_ATTR void VKAPI_CALL cb_bind_vb2(VkCommandBuffer, std::uint32_t, std::uint32_t, const VkBuffer*,
                                                  const VkDeviceSize*, const VkDeviceSize*, const VkDeviceSize*) {}
    static VKAPI_ATTR void VKAPI_CALL cb_bind_ib(VkCommandBuffer, VkBuffer, VkDeviceSize, VkIndexType) {}
    static VKAPI_ATTR void VKAPI_CALL cb_bind_ib2(VkCommandBuffer, VkBuffer, VkDeviceSize, VkDeviceSize, VkIndexType) {}
    static VKAPI_ATTR void VKAPI_CALL cb_set_vp(VkCommandBuffer, std::uint32_t, std::uint32_t, const VkViewport*) {}
    static VKAPI_ATTR void VKAPI_CALL cb_set_sc(VkCommandBuffer, std::uint32_t, std::uint32_t, const VkRect2D*) {}
    static VKAPI_ATTR void VKAPI_CALL cb_set_topo(VkCommandBuffer, VkPrimitiveTopology t) {
        g_harness->bound_topology = t;
    }
    static VKAPI_ATTR void VKAPI_CALL cb_set_restart(VkCommandBuffer, VkBool32 en) {
        g_harness->bound_restart = en;
    }
    static VKAPI_ATTR void VKAPI_CALL cb_set_stencil(VkCommandBuffer, VkStencilFaceFlags, std::uint32_t) {}
    static VKAPI_ATTR void VKAPI_CALL cb_set_blend(VkCommandBuffer, const float[4]) {}
    static VKAPI_ATTR void VKAPI_CALL cb_bind_pipeline(VkCommandBuffer, VkPipelineBindPoint, VkPipeline p) {
        g_harness->bound_pipeline = p;
    }
    static VKAPI_ATTR void VKAPI_CALL cb_push_constants(VkCommandBuffer, VkPipelineLayout, VkShaderStageFlags,
                                                        std::uint32_t, std::uint32_t, const void*) {}
    static VKAPI_ATTR void VKAPI_CALL cb_draw(VkCommandBuffer, std::uint32_t, std::uint32_t, std::uint32_t,
                                              std::uint32_t) {}
    static VKAPI_ATTR void VKAPI_CALL cb_draw_indexed(VkCommandBuffer, std::uint32_t, std::uint32_t, std::uint32_t,
                                                      std::int32_t, std::uint32_t) {}
    static VKAPI_ATTR void VKAPI_CALL cb_draw_indirect(VkCommandBuffer, VkBuffer, VkDeviceSize, std::uint32_t count,
                                                       std::uint32_t) {
        g_harness->draws.push_back(
            {g_harness->bound_pipeline, g_harness->bound_topology, g_harness->bound_restart, false, false, count});
    }
    static VKAPI_ATTR void VKAPI_CALL cb_draw_indexed_indirect(VkCommandBuffer, VkBuffer, VkDeviceSize,
                                                               std::uint32_t count, std::uint32_t) {
        g_harness->draws.push_back(
            {g_harness->bound_pipeline, g_harness->bound_topology, g_harness->bound_restart, true, false, count});
    }
    static VKAPI_ATTR void VKAPI_CALL cb_draw_indirect_count(VkCommandBuffer, VkBuffer, VkDeviceSize, VkBuffer,
                                                             VkDeviceSize, std::uint32_t max_count, std::uint32_t) {
        g_harness->draws.push_back(
            {g_harness->bound_pipeline, g_harness->bound_topology, g_harness->bound_restart, false, true, max_count});
    }
    static VKAPI_ATTR void VKAPI_CALL cb_draw_indexed_indirect_count(VkCommandBuffer, VkBuffer, VkDeviceSize, VkBuffer,
                                                                     VkDeviceSize, std::uint32_t max_count,
                                                                     std::uint32_t) {
        g_harness->draws.push_back(
            {g_harness->bound_pipeline, g_harness->bound_topology, g_harness->bound_restart, true, true, max_count});
    }

    Harness() {
        g_harness = this;
        pipeline.pipeline = fake_handle<VkPipeline>(0x1001);
        pipeline.layout = fake_handle<VkPipelineLayout>(0x2001);
        pipeline.push_stages = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
        pipeline.rectangle_variant_topology = leonida::topology_rect_list;
        pipeline.variant = [this](std::uint32_t flags) {
            ++variant_calls;
            last_variant_flags = flags;
            if (flags & leonida::GraphicsPipeline::rectangle_stage)
                return fake_handle<VkPipeline>(0x1000 + flags);
            return fake_handle<VkPipeline>(0x1001);
        };

        args.buffer = fake_handle<VkBuffer>(0x3001);
        args.buffer_offset = 64;
        args.size = 256;
        args.indirect_arguments = true;

        counts.buffer = fake_handle<VkBuffer>(0x3002);
        counts.buffer_offset = 128;
        counts.size = 64;
        counts.indirect_arguments = true;

        vk.graphics_pipeline = [this](std::uintptr_t id) -> const leonida::GraphicsPipeline* {
            return id == 7 ? &pipeline : nullptr;
        };
        vk.buffer_at = [](std::uint64_t va, std::uint64_t& size, VkBuffer& buf, VkDeviceSize& off) {
            if (va != 0x10000) return false;
            buf = fake_handle<VkBuffer>(0x3003);
            off = 16;
            size = 256;
            return true;
        };
        vk.image_view = [](const leonida::ResourceBinding&, const VkImageSubresourceRange&) {
            return fake_handle<VkImageView>(0x4001);
        };
        vk.begin_rendering = &cb_begin_rendering;
        vk.end_rendering = &cb_end_rendering;
        vk.bind_vertex_buffers2 = &cb_bind_vb2;
        vk.bind_index_buffer = &cb_bind_ib;
        vk.bind_robust_index_view = &cb_bind_ib2;
        vk.set_viewport = &cb_set_vp;
        vk.set_scissor = &cb_set_sc;
        vk.set_primitive_topology = &cb_set_topo;
        vk.set_primitive_restart_enable = &cb_set_restart;
        vk.set_stencil_reference = &cb_set_stencil;
        vk.set_blend_constants = &cb_set_blend;
        vk.bind_pipeline = &cb_bind_pipeline;
        vk.push_constants = &cb_push_constants;
        vk.draw = &cb_draw;
        vk.draw_indexed = &cb_draw_indexed;
        vk.draw_indirect = &cb_draw_indirect;
        vk.draw_indexed_indirect = &cb_draw_indexed_indirect;
        vk.draw_indirect_count = &cb_draw_indirect_count;
        vk.draw_indexed_indirect_count = &cb_draw_indexed_indirect_count;
        vk.draw_indirect_first_instance = true;
        vk.multi_draw_indirect = true;
        vk.draw_indirect_count_enabled = true;
        vk.max_draw_indirect_count = UINT32_MAX;
    }

    ~Harness() {
        if (g_harness == this) g_harness = nullptr;
    }

    leonida::ReplayResult run_indirect(const leonida::Draw& draw, const leonida::IndirectDrawArgs& arg) {
        g_harness = this;
        draws.clear();
        bound_pipeline = VK_NULL_HANDLE;
        bound_topology = VK_PRIMITIVE_TOPOLOGY_MAX_ENUM;
        bound_restart = VK_FALSE;
        leonida::CommandList list;
        CHECK(list.draw_indirect(leonida::Draw{draw}, arg));
        CHECK(list.close());
        const auto resolve = [this](leonida::ResourceId id) -> const leonida::ResourceBinding* {
            if (id == 1) return &args;
            if (id == 2) return &counts;
            return nullptr;
        };
        return leonida::replay(list, VK_NULL_HANDLE, resolve, vk);
    }
};

leonida::Draw make_base_draw(std::uint32_t topology, bool indexed, std::uint32_t index_format = 57) {
    leonida::Draw d{};
    d.pipeline = 7;
    d.topology = topology;
    d.viewport = {0.0f, 0.0f, 64.0f, 64.0f, 0.0f, 1.0f};
    d.scissor = {0, 0, 64, 64};
    d.indexed = indexed;
    if (indexed) d.index = {0x10000, 64, index_format};  // 57 = R16_UINT, 42 = R32_UINT
    return d;
}

void test_topology_predicate() {
    using leonida::indirect_stats::topology_supported;
    for (std::uint32_t t = 0; t <= 100; ++t) {
        const bool expected = (t == 1 || t == 2 || t == 3 || t == 4 || t == 5 || t == 10 || t == 11 || t == 12 ||
                               t == 13 || t == leonida::topology_rect_list);
        CHECK(topology_supported(t) == expected);
    }
    CHECK(!topology_supported(0));
    CHECK(!topology_supported(leonida::topology_quad_list));
    CHECK(!topology_supported(33));  // D3D_PRIMITIVE_TOPOLOGY_1_CONTROL_POINT_PATCHLIST
    CHECK(!topology_supported(64));  // D3D_PRIMITIVE_TOPOLOGY_32_CONTROL_POINT_PATCHLIST
}

void test_supported_topologies_emit() {
    struct Case {
        std::uint32_t d3d_topology;
        VkPrimitiveTopology vk_topology;
        bool is_strip;
        bool expects_rect_stage;
    };
    constexpr Case cases[] = {
        {1, VK_PRIMITIVE_TOPOLOGY_POINT_LIST, false, false},
        {2, VK_PRIMITIVE_TOPOLOGY_LINE_LIST, false, false},
        {3, VK_PRIMITIVE_TOPOLOGY_LINE_STRIP, true, false},
        {4, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, false, false},
        {5, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP, true, false},
        {10, VK_PRIMITIVE_TOPOLOGY_LINE_LIST_WITH_ADJACENCY, false, false},
        {11, VK_PRIMITIVE_TOPOLOGY_LINE_STRIP_WITH_ADJACENCY, true, false},
        {12, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST_WITH_ADJACENCY, false, false},
        {13, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP_WITH_ADJACENCY, true, false},
        {leonida::topology_rect_list, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, false, true},
    };

    Harness h;
    for (const auto& c : cases) {
        for (const bool indexed : {false, true}) {
            for (const bool counted : {false, true}) {
                for (const std::uint32_t strip_cut : {0u, 1u, 2u}) {
                    for (const std::uint32_t idx_fmt : {57u, 42u}) {
                        if (!indexed && idx_fmt == 42u) continue;
                        h.pipeline.strip_cut = strip_cut;
                        h.variant_calls = 0;
                        h.last_variant_flags = 0;
                        const leonida::Draw d = make_base_draw(c.d3d_topology, indexed, idx_fmt);
                        leonida::IndirectDrawArgs arg{};
                        arg.argument_buffer = 1;
                        arg.argument_offset = 0;
                        arg.stride = indexed ? 20 : 16;
                        arg.max_count = 3;
                        if (counted) {
                            arg.count_buffer = 2;
                            arg.count_offset = 4;
                        }
                        CHECK(h.run_indirect(d, arg) == leonida::ReplayResult::success);
                        CHECK(h.draws.size() == 1);
                        const auto& got = h.draws[0];
                        CHECK(got.topology == c.vk_topology);
                        CHECK(got.indexed == indexed);
                        CHECK(got.counted == counted);
                        CHECK(got.max_count == 3);
                        const bool cut_matches = (strip_cut == 1 && idx_fmt == 57) || (strip_cut == 2 && idx_fmt == 42);
                        const VkBool32 want_restart = (indexed && c.is_strip && cut_matches) ? VK_TRUE : VK_FALSE;
                        CHECK(got.primitive_restart == want_restart);
                        if (c.expects_rect_stage) {
                            CHECK(h.variant_calls == 1);
                            CHECK((h.last_variant_flags & leonida::GraphicsPipeline::rectangle_stage) != 0);
                            CHECK(got.pipeline == fake_handle<VkPipeline>(0x1000 + h.last_variant_flags));
                        } else {
                            CHECK(h.variant_calls == 0);
                            CHECK(got.pipeline == h.pipeline.pipeline);
                        }
                    }
                }
            }
        }
    }
}

void test_rect_list_builtin_and_fallback() {
    // 1. PSO created with topology_type_rect_list (19) already contains the GS -> rectangle_variant_topology == 0.
    {
        Harness h;
        h.pipeline.rectangle_variant_topology = 0;
        const leonida::Draw d = make_base_draw(leonida::topology_rect_list, false);
        leonida::IndirectDrawArgs arg{1, 0, 16, 1, 0, 0};
        CHECK(h.run_indirect(d, arg) == leonida::ReplayResult::success);
        CHECK(h.draws.size() == 1);
        CHECK(h.variant_calls == 0);
        CHECK(h.draws[0].pipeline == h.pipeline.pipeline);
        CHECK(h.draws[0].topology == VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
    }

    // 2. Indirect RECTLIST when the pipeline cannot build a rectangle_stage variant -> skips cleanly with topology,
    //    never fails the command list (both without and with vk.compiles).
    for (const bool with_compiles : {false, true}) {
        Harness h;
        h.pipeline.variant = [](std::uint32_t) { return VkPipeline(VK_NULL_HANDLE); };
        h.pipeline.built = [](std::uint32_t) { return VkPipeline(VK_NULL_HANDLE); };
        leonida::PipelineCompiles compiles;
        if (with_compiles) h.vk.compiles = &compiles;

        const auto before = leonida::indirect_stats::reason_skips(leonida::indirect_stats::topology);
        const leonida::Draw d = make_base_draw(leonida::topology_rect_list, false);
        leonida::IndirectDrawArgs arg{1, 0, 16, 2, 0, 0};
        CHECK(h.run_indirect(d, arg) == leonida::ReplayResult::success);
        CHECK(h.draws.empty());
        CHECK(compiles.empty());
        CHECK(leonida::indirect_stats::reason_skips(leonida::indirect_stats::topology) == before + 1);
    }

    // 3. Direct RECTLIST when rectangle_stage variant cannot be built still returns unsupported (unchanged).
    {
        Harness h;
        h.pipeline.variant = [](std::uint32_t) { return VkPipeline(VK_NULL_HANDLE); };
        leonida::CommandList list;
        leonida::Draw d = make_base_draw(leonida::topology_rect_list, false);
        d.count = 3;
        d.instances = 1;
        CHECK(list.draw(d) && list.close());
        const auto res = leonida::replay(list, VK_NULL_HANDLE,
                                         [](leonida::ResourceId) -> const leonida::ResourceBinding* { return nullptr; },
                                         h.vk);
        CHECK(res == leonida::ReplayResult::unsupported);
    }
}

void test_unsupported_topologies_skip() {
    Harness h;
    constexpr std::uint32_t unsupported[] = {0, 6, 7, 9, 14, 18, leonida::topology_quad_list, 33, 64};
    for (const std::uint32_t topo : unsupported) {
        const auto before = leonida::indirect_stats::reason_skips(leonida::indirect_stats::topology);
        const leonida::Draw d = make_base_draw(topo, false);
        leonida::IndirectDrawArgs arg{1, 0, 16, 1, 0, 0};
        CHECK(h.run_indirect(d, arg) == leonida::ReplayResult::success);
        CHECK(h.draws.empty());
        CHECK(leonida::indirect_stats::reason_skips(leonida::indirect_stats::topology) == before + 1);
    }
}

std::string capture_report(const leonida::indirect_stats::Counts& counts) {
    std::FILE* f = std::tmpfile();
    CHECK(f != nullptr);
    counts.report(f);
    std::rewind(f);
    char buf[1536]{};
    const std::size_t n = std::fread(buf, 1, sizeof(buf) - 1, f);
    std::fclose(f);
    return std::string(buf, n);
}

void test_counts_report_and_concurrency() {
    using namespace leonida::indirect_stats;

    // 1. Clean prefix with no bracketed breakdown when all per-reason counts are zero.
    {
        Counts c;
        c.values[static_cast<unsigned>(Event::recorded)].store(16);
        c.values[static_cast<unsigned>(Event::emitted)].store(15);
        const std::string line = capture_report(c);
        CHECK(line == "[rt] indirect graphics: recorded=16 emitted=15 record_skips=0 replay_skips=0\n");
    }

    // 2. Non-zero per-reason counts appear in stable enum order inside [...].
    {
        Counts c;
        c.values[static_cast<unsigned>(Event::recorded)].store(4096);
        c.values[static_cast<unsigned>(Event::emitted)].store(3918);
        c.values[static_cast<unsigned>(Event::record_skip)].store(53405);
        c.values[static_cast<unsigned>(Event::replay_skip)].store(7);
        c.by_reason[compound].store(2);
        c.by_reason[mesh].store(53000);
        c.by_reason[topology].store(403);
        c.by_reason[pending_pipeline].store(7);
        const std::string line = capture_report(c);
        CHECK(line ==
              "[rt] indirect graphics: recorded=4096 emitted=3918 record_skips=53405 replay_skips=7 "
              "[compound=2, mesh=53000, topology=403, pending_pipeline=7]\n");
    }

    // 3. Out-of-range reason is ignored safely without corrupting counts.
    {
        const auto before_total = counts().values[static_cast<unsigned>(Event::record_skip)].load();
        skip(Event::record_skip, reason_count);
        CHECK(counts().values[static_cast<unsigned>(Event::record_skip)].load() == before_total);
        CHECK(reason_skips(reason_count) == 0);
    }

    // 4. Multi-threaded skip increments are race-free and exact.
    {
        const auto mesh_before = reason_skips(mesh);
        const auto div_before = reason_skips(divisor);
        std::vector<std::thread> threads;
        for (int t = 0; t < 4; ++t) {
            threads.emplace_back([t] {
                for (int i = 0; i < 250; ++i) {
                    skip(Event::record_skip, (t % 2 == 0) ? mesh : divisor);
                }
            });
        }
        for (auto& th : threads) th.join();
        CHECK(reason_skips(mesh) == mesh_before + 500);
        CHECK(reason_skips(divisor) == div_before + 500);
    }
}

}  // namespace

int main() {
    test_topology_predicate();
    test_supported_topologies_emit();
    test_rect_list_builtin_and_fallback();
    test_unsupported_topologies_skip();
    test_counts_report_and_concurrency();
    std::printf("indirect_topology_and_stats_test: PASS\n");
    return 0;
}
