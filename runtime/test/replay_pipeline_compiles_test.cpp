// Pipelines that are not built when a submission is prepared (PipelineCompiles, build_pipelines): the same commands are
// prepared twice, once with every pipeline built where prepare() meets it and once with them left to PipelineCompiles
// and built on PipelineWorkers. Both have to give the same result, the same pipeline to every draw and dispatch, the same
// commands dropped, and the same pipelines built, each once. Fixed cases, then random ones. Run a second time with
// LEONIDA_STRICT_REPLAY=1, where a dispatch without a pipeline fails its list. Also which large pipelines compile at once
// within a budget (PipelineCompiles::build). No GPU: pipelines and resources are fakes.
#include "../src/large_module_gate.h"
#include "../src/pipeline_workers.h"
#include "../src/vulkan_replay.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <set>
#include <string>
#include <thread>
#include <vector>

namespace {
using namespace leonida;

int failures = 0;
void check(bool ok, const std::string& what) {
    if (ok) return;
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++failures;
}

template<class T>
T handle(std::uintptr_t n) {
    return reinterpret_cast<T>(n);
}

constexpr ResourceId color_id = 1, depth_id = 2, arguments_id = 3;
constexpr std::uint64_t bad_va = 0xbad0000;

// The large pipelines being built now, and how often there were two, where they are to compile one after another.
std::atomic<int> large_now = 0;
std::atomic<int> large_together = 0;
std::atomic<bool> large_one_at_a_time = true;

// What a pipeline state owns: its variants, built on demand under its own lock, and what was asked of it.
struct Owner {
    std::uintptr_t id = 0;
    bool fails = false;
    bool large = false;
    bool compute = false;
    bool failed = false;
    std::mutex mutex;
    std::map<std::uint32_t, VkPipeline> variants;
    std::vector<std::uint32_t> attempts;  // the flags of every build, also the failed ones

    VkPipeline build(std::uint32_t flags) {
        std::lock_guard lock(mutex);
        if (const auto it = variants.find(flags); it != variants.end()) return it->second;
        if (failed && compute) return VK_NULL_HANDLE;  // a refused compute pipeline is not tried again
        attempts.push_back(flags);
        if (large) {
            if (++large_now > 1 && large_one_at_a_time) ++large_together;
            const auto end = std::chrono::steady_clock::now() + std::chrono::microseconds(20);
            while (std::chrono::steady_clock::now() < end) std::this_thread::yield();
            --large_now;
        }
        if (fails) {
            failed = true;
            return VK_NULL_HANDLE;
        }
        return variants[flags] = handle<VkPipeline>(id * 0x1000 + flags + 1);
    }
    VkPipeline find(std::uint32_t flags) {
        std::lock_guard lock(mutex);
        const auto it = variants.find(flags);
        return it == variants.end() ? VK_NULL_HANDLE : it->second;
    }
};

struct PipelineSpec {
    bool compute = false;
    bool pending = true;   // no pipeline yet: built on first use
    bool fails = false;
    bool with_built = true;
    bool large = false;
    std::uint64_t estimate = 0;
    bool skipped = false;   // compute: never built, its dispatches are dropped
    bool bindless = false;  // compute: needs a set 0, which these functions cannot bind
    bool async = false;     // its null handle may be waiting for another worker: never left to PipelineCompiles
    bool mesh = false;
    bool rectangle = false;
    bool vertices = false;
    int colors = 1;
    bool depth = false;
    std::set<std::uint32_t> ready;  // built before
};

// One run's registered pipelines, made from the same specs for both runs.
struct Registry {
    std::vector<std::unique_ptr<Owner>> owners;
    std::map<std::uintptr_t, GraphicsPipeline> graphics;
    std::map<std::uintptr_t, ComputePipeline> compute;

    explicit Registry(const std::vector<PipelineSpec>& specs) {
        for (std::size_t i = 0; i < specs.size(); ++i) {
            const PipelineSpec& spec = specs[i];
            auto& owner = *owners.emplace_back(std::make_unique<Owner>());
            owner.id = i + 1;
            owner.fails = spec.fails;
            owner.large = spec.large;
            owner.compute = spec.compute;
            for (const auto flags : spec.ready) owner.variants[flags] = handle<VkPipeline>(owner.id * 0x1000 + flags + 1);
            if (spec.compute) {
                ComputePipeline p{};
                p.layout = handle<VkPipelineLayout>(7);
                p.large = spec.large;
                p.estimate = spec.estimate;
                p.async_build = spec.async;
                p.skipped = spec.skipped;
                if (spec.bindless) p.bindless_set = handle<VkDescriptorSet>(6);
                if (spec.pending) {
                    p.build = [](void* context) { return static_cast<Owner*>(context)->build(0); };
                    if (spec.with_built) p.built = [](void* context) { return static_cast<Owner*>(context)->find(0); };
                    p.build_context = &owner;
                } else {
                    p.pipeline = handle<VkPipeline>(owner.id * 0x1000);
                }
                compute[owner.id] = p;
                continue;
            }
            GraphicsPipeline p{};
            p.layout = handle<VkPipelineLayout>(7);
            p.async_build = spec.async;
            if (!spec.pending) p.pipeline = handle<VkPipeline>(owner.id * 0x1000);
            p.color_formats.assign(std::size_t(spec.colors), VK_FORMAT_R8G8B8A8_UNORM);
            if (spec.depth) {
                p.depth_format = VK_FORMAT_D32_SFLOAT;
                p.depth_aspects = VK_IMAGE_ASPECT_DEPTH_BIT;
            }
            p.mesh = spec.mesh;
            p.large = spec.large;
            p.estimate = spec.estimate;
            p.vertex_slots = spec.vertices ? 1 : 0;
            if (spec.rectangle) p.rectangle_variant_topology = 17;
            p.variant = [&owner](std::uint32_t flags) { return owner.build(flags); };
            if (spec.with_built) p.built = [&owner](std::uint32_t flags) { return owner.find(flags); };
            graphics[owner.id] = p;
        }
    }
    std::multiset<std::pair<std::uintptr_t, std::uint32_t>> built() const {
        std::multiset<std::pair<std::uintptr_t, std::uint32_t>> out;
        for (const auto& o : owners)
            for (const auto flags : o->attempts) out.insert({o->id, flags});
        return out;
    }
};

struct CommandSpec {
    enum Kind { draw, dispatch, indirect_dispatch } kind = draw;
    std::uintptr_t pipeline = 1;
    int colors = 1;            // how many render targets are bound
    bool depth = false;
    bool rectangle = false;    // the topology of PipelineSpec::rectangle
    bool bad_vertices = false;  // a vertex buffer that resolves to nothing
    bool mesh = false;
};

std::vector<CommandList> record(const std::vector<std::vector<CommandSpec>>& specs) {
    std::vector<CommandList> lists;
    for (const auto& commands : specs) {
        CommandList list;
        for (const CommandSpec& c : commands) {
            if (c.kind == CommandSpec::dispatch) {
                list.dispatch({c.pipeline, {}, 1, 1, 1});
            } else if (c.kind == CommandSpec::indirect_dispatch) {
                ExecuteIndirect e{};
                e.kind = IndirectKind::dispatch;
                e.stride = 12;
                e.max_count = 1;
                e.argument_buffer = arguments_id;
                e.pipeline = c.pipeline;
                list.execute_indirect(e);
            } else {
                Draw d{};
                d.pipeline = c.pipeline;
                d.topology = c.rectangle ? 17 : 4;
                d.count = 3;
                d.instances = 1;
                d.first = 1;
                d.mesh = c.mesh;
                auto views = std::make_shared<DrawViews>();
                for (int i = 0; i < c.colors; ++i) views->render_targets[std::size_t(i)] = {color_id, 0, 0, 1};
                views->vertex_buffers[0].va = c.bad_vertices ? bad_va : 0x1000;
                views->vertex_buffers[0].size = 64;
                d.views = views;
                if (c.depth) d.depth = {depth_id, 0, 0, 1};
                list.draw(d);
            }
        }
        list.close();
        lists.push_back(std::move(list));
    }
    return lists;
}

ReplayFunctions functions(const Registry& registry) {
    ReplayFunctions vk{};
    vk.begin_rendering = +[](VkCommandBuffer, const VkRenderingInfo*) {};
    vk.end_rendering = +[](VkCommandBuffer) {};
    vk.bind_pipeline = +[](VkCommandBuffer, VkPipelineBindPoint, VkPipeline) {};
    vk.push_constants = +[](VkCommandBuffer, VkPipelineLayout, VkShaderStageFlags, std::uint32_t, std::uint32_t,
                            const void*) {};
    vk.dispatch = +[](VkCommandBuffer, std::uint32_t, std::uint32_t, std::uint32_t) {};
    vk.dispatch_indirect = +[](VkCommandBuffer, VkBuffer, VkDeviceSize) {};
    vk.set_viewport = +[](VkCommandBuffer, std::uint32_t, std::uint32_t, const VkViewport*) {};
    vk.set_scissor = +[](VkCommandBuffer, std::uint32_t, std::uint32_t, const VkRect2D*) {};
    vk.set_primitive_topology = +[](VkCommandBuffer, VkPrimitiveTopology) {};
    vk.set_primitive_restart_enable = +[](VkCommandBuffer, VkBool32) {};
    vk.set_stencil_reference = +[](VkCommandBuffer, VkStencilFaceFlags, std::uint32_t) {};
    vk.set_blend_constants = +[](VkCommandBuffer, const float*) {};
    vk.bind_index_buffer = +[](VkCommandBuffer, VkBuffer, VkDeviceSize, VkIndexType) {};
    vk.bind_vertex_buffers2 = +[](VkCommandBuffer, std::uint32_t, std::uint32_t, const VkBuffer*, const VkDeviceSize*,
                                  const VkDeviceSize*, const VkDeviceSize*) {};
    vk.draw = +[](VkCommandBuffer, std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t) {};
    vk.draw_indexed = +[](VkCommandBuffer, std::uint32_t, std::uint32_t, std::uint32_t, std::int32_t, std::uint32_t) {};
    vk.draw_mesh_tasks = +[](VkCommandBuffer, std::uint32_t, std::uint32_t, std::uint32_t) {};
    vk.image_view = [](const ResourceBinding&, const VkImageSubresourceRange&) { return handle<VkImageView>(8); };
    vk.buffer_at = [](std::uint64_t va, std::uint64_t&, VkBuffer& buffer, VkDeviceSize& offset) {
        if (va == bad_va) return false;
        buffer = handle<VkBuffer>(9);
        offset = 0;
        return true;
    };
    vk.graphics_pipeline = [&registry](std::uintptr_t id) -> const GraphicsPipeline* {
        const auto it = registry.graphics.find(id);
        return it == registry.graphics.end() ? nullptr : &it->second;
    };
    vk.compute_pipeline = [&registry](std::uintptr_t id) -> const ComputePipeline* {
        const auto it = registry.compute.find(id);
        return it == registry.compute.end() ? nullptr : &it->second;
    };
    return vk;
}

const ResourceBinding* resolve(ResourceId id) {
    static const auto binding = [](VkImageAspectFlags aspects, VkFormat format) {
        ResourceBinding b{};
        b.aspects = aspects;
        b.format = format;
        b.width = b.height = 16;
        return b;
    };
    static const ResourceBinding color = [] {
        auto b = binding(VK_IMAGE_ASPECT_COLOR_BIT, VK_FORMAT_R8G8B8A8_UNORM);
        b.image = handle<VkImage>(10);
        return b;
    }();
    static const ResourceBinding depth = [] {
        auto b = binding(VK_IMAGE_ASPECT_DEPTH_BIT, VK_FORMAT_D32_SFLOAT);
        b.image = handle<VkImage>(11);
        return b;
    }();
    static const ResourceBinding arguments = [] {
        auto b = binding(0, VK_FORMAT_UNDEFINED);
        b.buffer = handle<VkBuffer>(12);
        b.size = 4096;
        return b;
    }();
    return id == color_id ? &color : id == depth_id ? &depth : id == arguments_id ? &arguments : nullptr;
}

struct Outcome {
    ReplayResult result = ReplayResult::success;
    std::vector<VkPipeline> pipelines;  // of every draw and dispatch, when the lists were prepared; null: dropped
    std::multiset<std::pair<std::uintptr_t, std::uint32_t>> built;
    std::size_t jobs = 0;
    std::size_t runs = 0;
    std::size_t units = 0;  // what run() was given: the large pipelines together are one (no budget)
};

// `workers` null: prepare() builds each pipeline itself, as without workers.
Outcome prepare_all(const std::vector<CommandList>& lists, const std::vector<PipelineSpec>& specs,
                    PipelineWorkers* workers, const std::function<std::uint64_t()>& budget = {}) {
    Registry registry(specs);
    ReplayFunctions vk = functions(registry);
    PipelineCompiles compiles;
    if (workers) vk.compiles = &compiles;
    Outcome out;
    std::vector<PreparedList> prepared(lists.size());
    for (std::size_t i = 0; i < lists.size() && out.result == ReplayResult::success; ++i)
        out.result = prepare(lists[i], resolve, vk, prepared[i]);
    out.jobs = compiles.size();
    if (workers)
        out.result = build_pipelines(compiles, std::span(prepared), out.result,
                                     [&](std::size_t count, const std::function<void(std::size_t)>& job) {
                                         ++out.runs;
                                         out.units = count;
                                         workers->run(count, job);
                                     },
                                     budget);
    if (out.result == ReplayResult::success)
        for (std::size_t i = 0; i < lists.size(); ++i) {
            for (std::size_t c = 0; c < prepared[i].commands.size(); ++c) {
                const auto& command = lists[i].commands()[c];
                if (std::holds_alternative<Draw>(command)) {
                    const auto& draw = prepared[i].draws[prepared[i].commands[c].draw];
                    out.pipelines.push_back(draw.skipped ? VkPipeline(VK_NULL_HANDLE) : draw.pipeline);
                    check(draw.skipped || draw.pipeline, "a draw neither dropped nor with a pipeline");
                } else
                    out.pipelines.push_back(prepared[i].commands[c].pipeline.pipeline);
            }
        }
    out.built = registry.built();
    return out;
}

Outcome compare(const std::string& name, const std::vector<PipelineSpec>& specs,
                const std::vector<std::vector<CommandSpec>>& commands, PipelineWorkers& workers,
                const std::function<std::uint64_t()>& budget = {}) {
    const auto lists = record(commands);
    const Outcome serial = prepare_all(lists, specs, nullptr);
    large_one_at_a_time = !budget;
    const Outcome together = prepare_all(lists, specs, &workers, budget);
    large_one_at_a_time = true;
    check(serial.result == together.result, name + ": another result");
    check(serial.pipelines == together.pipelines, name + ": another pipeline for a draw or dispatch");
    if (serial.result == ReplayResult::success)
        check(serial.built == together.built, name + ": other pipelines built");
    else  // serial stops at the first pipeline that fails; the others were already wanted by commands before
        check(std::includes(together.built.begin(), together.built.end(), serial.built.begin(), serial.built.end()),
              name + ": a pipeline built serially was not built");
    for (const auto& one : together.built)  // one another worker builds is asked for again while it is refused
        check(together.built.count(one) == 1 || (specs[one.first - 1].async && specs[one.first - 1].fails),
              name + ": a pipeline built twice");
    return together;
}

using Commands = std::vector<std::vector<CommandSpec>>;

void cases(PipelineWorkers& workers) {
    const bool strict = strict_replay();
    const PipelineSpec pending{}, eager{.pending = false};
    const CommandSpec draw{};
    const auto use = [](std::uintptr_t id) { CommandSpec c{}; c.pipeline = id; return c; };

    auto out = compare("one pending pipeline in many draws", {pending}, {{draw, draw, draw}, {draw}}, workers);
    check(out.jobs == 1 && out.built.size() == 1, "one pipeline is one job");
    out = compare("several pending pipelines", {pending, pending, pending},
                  {{use(1), use(2), use(1)}, {use(3), use(2)}}, workers);
    check(out.jobs == 3 && out.pipelines.size() == 5 && out.runs == 1, "three pipelines, all in one batch");
    out = compare("an eager pipeline", {eager}, {{draw, draw}}, workers);
    check(out.jobs == 0 && out.runs == 0, "an eager pipeline with everything bound needs no job");
    out = compare("a built variant", {{.ready = {0}}}, {{draw, draw}}, workers);
    check(out.jobs == 0 && out.runs == 0 && out.built.empty(), "a built pipeline is found, not built again");

    // The variant flags: colour slots left unbound, depth left unbound, the rectangle topology.
    PipelineSpec targets{.pending = false, .rectangle = true, .colors = 3, .depth = true};
    CommandSpec full{.colors = 3, .depth = true}, two = full, no_depth = full, rectangle = full, all = full;
    two.colors = 1;
    no_depth.depth = false;
    rectangle.rectangle = true;
    all.colors = 0;
    all.depth = false;
    all.rectangle = true;
    out = compare("variants", {targets}, {{full, two, no_depth, rectangle, all, two, all}}, workers);
    const std::multiset<std::pair<std::uintptr_t, std::uint32_t>> flags{
        {1, 0b110}, {1, GraphicsPipeline::unbound_depth}, {1, GraphicsPipeline::rectangle_stage},
        {1, 0b111 | GraphicsPipeline::unbound_depth | GraphicsPipeline::rectangle_stage}};
    check(out.built == flags && out.jobs == 4, "the variants a draw needs, each once");
    targets.pending = true;
    out = compare("base and variants pending", {targets}, {{full, two, full}}, workers);
    check(out.jobs == 2, "the base pipeline and one variant");

    // Pipelines that cannot say whether they are built are built on the spot, as before.
    out = compare("without built", {{.with_built = false}, pending}, {{use(1), use(2)}}, workers);
    check(out.jobs == 1, "only the pipeline with `built` is left to PipelineCompiles");

    // Compute, directly and through ExecuteIndirect.
    const PipelineSpec compute{.compute = true}, ready_compute{.compute = true, .ready = {0}},
        eager_compute{.compute = true, .pending = false};
    CommandSpec dispatch{.kind = CommandSpec::dispatch}, indirect{.kind = CommandSpec::indirect_dispatch};
    const auto with = [](CommandSpec c, std::uintptr_t id) { c.pipeline = id; return c; };
    out = compare("first-use dispatches", {compute, compute, eager_compute, ready_compute},
                  {{with(dispatch, 1), with(indirect, 2), with(dispatch, 1), with(dispatch, 3), with(dispatch, 4),
                    with(indirect, 1)}},
                  workers);
    check(out.jobs == 2 && out.pipelines.size() == 6, "a compute pipeline is compiled once for both kinds of dispatch");
    out = compare("draws and dispatches", {pending, compute}, {{use(1), with(dispatch, 2)}, {with(indirect, 2), use(1)}},
                  workers);
    check(out.jobs == 2 && out.runs == 1, "graphics and compute share one batch");

    // Failures: the one a list would have stopped at.
    const PipelineSpec failing{.fails = true}, failing_compute{.compute = true, .fails = true};
    out = compare("a pipeline that cannot be built", {failing}, {{draw}}, workers);
    check(out.result == ReplayResult::unsupported, "it is unsupported");
    out = compare("a compute pipeline that cannot be built", {failing_compute, compute},
                  {{with(dispatch, 1), with(dispatch, 2)}}, workers);
    check(out.result == (strict ? ReplayResult::unresolved_resource : ReplayResult::success), "its dispatch is dropped");
    check(strict || (!out.pipelines[0] && out.pipelines[1]), "and no other");
    out = compare("an indirect dispatch of one", {failing_compute}, {{with(indirect, 1)}}, workers);
    check(out.result == ReplayResult::unresolved_resource, "it fails its list");
    out = compare("a refused draw after a refused dispatch", {failing_compute, failing},
                  {{with(dispatch, 1), use(2)}}, workers);
    check(out.result == (strict ? ReplayResult::unresolved_resource : ReplayResult::unsupported), "the first that fails");
    out = compare("a refused indirect dispatch after a refused draw", {failing_compute, failing},
                  {{use(2), with(indirect, 1)}}, workers);
    check(out.result == ReplayResult::unsupported, "the draw comes first");
    out = compare("a refused draw after a refused indirect dispatch", {failing_compute, failing},
                  {{with(indirect, 1), use(2)}}, workers);
    check(out.result == ReplayResult::unresolved_resource, "the indirect dispatch comes first");
    const PipelineSpec skipped{.compute = true, .fails = true, .skipped = true};
    out = compare("a skipped program", {skipped, compute}, {{with(dispatch, 1), with(indirect, 1), with(dispatch, 2)}},
                  workers);
    check(out.result == ReplayResult::success && out.jobs == 1, "a skipped program is dropped, strictly too");
    out = compare("a set 0 that cannot be bound", {{.compute = true, .bindless = true}}, {{with(dispatch, 1)}}, workers);
    check(out.result == ReplayResult::unsupported, "it is unsupported once the pipeline is there");
    out = compare("and a pipeline that cannot be built", {{.compute = true, .fails = true, .bindless = true}},
                  {{with(dispatch, 1)}}, workers);
    check(out.result == (strict ? ReplayResult::unresolved_resource : ReplayResult::success), "it is dropped first");
    const PipelineSpec async{.async = true}, async_compute{.compute = true, .async = true};
    out = compare("pipelines that another worker builds", {async, async_compute}, {{use(1), with(dispatch, 2)}}, workers);
    check(out.jobs == 0, "async_build is never left to PipelineCompiles");
    compare("and cannot", {{.fails = true, .async = true}, {.compute = true, .fails = true, .async = true}},
            {{use(1), with(dispatch, 2)}, {with(indirect, 2)}}, workers);
    out = compare("a failure among others", {pending, failing, pending}, {{use(1), use(2)}, {use(3)}}, workers);
    check(out.result == ReplayResult::unsupported, "one failure fails the submission");
    out = compare("a failure before a missing pipeline", {failing}, {{use(1), use(9)}}, workers);
    check(out.result == ReplayResult::unsupported, "the failed build comes first");
    out = compare("a missing pipeline before a failure", {failing}, {{use(9), use(1)}}, workers);
    check(out.result == (strict ? ReplayResult::unresolved_resource : ReplayResult::unsupported) &&
              out.built.empty() == strict,
          "a missing pipeline drops its draw; strictly it comes first");
    out = compare("a missing pipeline after one that builds", {pending}, {{use(1), use(9)}}, workers);
    check(out.result == (strict ? ReplayResult::unresolved_resource : ReplayResult::success) && out.built.size() == 1,
          "the pipeline is built, as before");
    out = compare("a missing pipeline in a later list", {pending, pending}, {{use(1)}, {use(2), use(9)}, {use(1)}}, workers);
    check(out.result == (strict ? ReplayResult::unresolved_resource : ReplayResult::success) && out.built.size() == 2,
          "lists before the failure");
    PipelineSpec vertices{.fails = true, .vertices = true};
    CommandSpec bad = draw;
    bad.bad_vertices = true;
    out = compare("a failure in a draw whose vertex buffer is missing too", {vertices}, {{bad}}, workers);
    check(out.result == ReplayResult::unsupported, "the pipeline is looked at before the vertex buffers");
    vertices.fails = false;
    out = compare("a missing vertex buffer", {vertices}, {{bad}}, workers);
    check(out.result == (strict ? ReplayResult::unresolved_resource : ReplayResult::success), "the vertex buffer");
    CommandSpec mesh = draw;
    mesh.mesh = true;
    out = compare("a mesh draw with a vertex pipeline", {pending}, {{mesh}}, workers);
    check(out.result == ReplayResult::invalid_draw && out.built.empty(), "refused before its pipeline is built");
}

// More than one worker's worth of pipelines, so that the batch really runs on several threads.
void many(PipelineWorkers& workers) {
    std::vector<PipelineSpec> specs;
    Commands commands(3);
    for (std::uintptr_t i = 0; i < 300; ++i) {
        PipelineSpec spec{};
        spec.compute = i % 3 == 2;
        spec.large = i % 7 == 0;  // 43 of them, graphics and compute
        specs.push_back(spec);
        for (std::size_t repeat = 0; repeat < 2; ++repeat) {
            CommandSpec c{};
            c.pipeline = i + 1;
            if (spec.compute) c.kind = repeat ? CommandSpec::indirect_dispatch : CommandSpec::dispatch;
            commands[(i + repeat) % 3].push_back(c);
        }
    }
    const auto out = compare("300 pipelines", specs, commands, workers);
    check(out.jobs == 300 && out.pipelines.size() == 600, "300 pipelines: every command has its own");
    check(out.units == 300 - 43 + 1, "the large pipelines are one unit, the others one each");
    const auto stats = workers.stats();
    check(stats.batches > 0 && stats.jobs >= out.units, "the batch went through the workers");
}

// A queue prepares every submission into the same PreparedLists: what one left in them is not used by the next.
void reused_lists(PipelineWorkers& workers) {
    Registry registry({{}, {}, {}, {.compute = true}, {}, {.compute = true}});
    ReplayFunctions vk = functions(registry);
    std::vector<PreparedList> prepared(2);
    const auto submit = [&](const Commands& commands, std::size_t jobs) {
        const auto lists = record(commands);
        PipelineCompiles compiles;
        vk.compiles = &compiles;
        ReplayResult result = ReplayResult::success;
        for (std::size_t i = 0; i < lists.size(); ++i)
            check(prepare(lists[i], resolve, vk, prepared[i]) == ReplayResult::success, "reused lists: prepare");
        check(compiles.size() == jobs, "reused lists: " + std::to_string(compiles.size()) + " jobs");
        result = build_pipelines(compiles, std::span(prepared).first(lists.size()), result,
                                 [&](std::size_t count, const std::function<void(std::size_t)>& job) {
                                     workers.run(count, job);
                                 });
        check(result == ReplayResult::success, "reused lists: result");
        for (std::size_t i = 0; i < lists.size(); ++i)
            for (std::size_t c = 0; c < commands[i].size(); ++c) {
                const VkPipeline want = handle<VkPipeline>(commands[i][c].pipeline * 0x1000 + 1);
                const auto& command = prepared[i].commands[c];
                check((commands[i][c].kind == CommandSpec::draw ? prepared[i].draws[command.draw].pipeline
                                                                : command.pipeline.pipeline) == want,
                      "reused lists: another pipeline");
            }
    };
    const auto draw = [](std::uintptr_t id) { CommandSpec c{}; c.pipeline = id; return c; };
    const auto dispatch = [](std::uintptr_t id) { CommandSpec c{.kind = CommandSpec::dispatch}; c.pipeline = id; return c; };
    submit({{draw(1), draw(2), dispatch(4), draw(3)}, {draw(2), draw(1)}}, 4);
    submit({{draw(3), draw(5), dispatch(4), draw(1), dispatch(6)}, {draw(1), draw(5), draw(2)}}, 2);
    submit({{draw(5)}, {draw(1)}}, 0);
    // Lists a shorter submission leaves over keep the tickets of an older one; nothing looks at them.
    submit({{draw(2)}}, 0);
}

void random_submissions(PipelineWorkers& workers) {
    std::mt19937 rng(20251125);
    const auto chance = [&](unsigned percent) { return rng() % 100 < percent; };
    unsigned succeeded = 0;
    for (int round = 0; round < 3000; ++round) {
        std::vector<PipelineSpec> specs(1 + rng() % 8);
        for (auto& s : specs) {
            s.compute = chance(30);
            s.pending = chance(75);
            s.fails = chance(4);
            s.with_built = chance(90);
            s.async = chance(8);
            s.large = chance(15);
            s.estimate = rng() % 12;
            s.bindless = chance(2);
            s.skipped = s.compute && chance(4);
            if (s.skipped) s.fails = s.pending = true;
            s.mesh = chance(3);
            s.rectangle = chance(20);
            s.vertices = chance(30);
            s.colors = int(rng() % 4);
            s.depth = chance(40);
            if (chance(30)) s.ready.insert(rng() % 4);
        }
        Commands commands(1 + rng() % 4);
        for (auto& list : commands)
            for (unsigned i = rng() % 8; i; --i) {
                CommandSpec c{};
                c.pipeline = 1 + (chance(3) ? specs.size() : rng() % specs.size());
                const bool compute = c.pipeline <= specs.size() && specs[c.pipeline - 1].compute;
                if (compute) c.kind = chance(50) ? CommandSpec::dispatch : CommandSpec::indirect_dispatch;
                c.colors = int(rng() % 4);
                c.depth = chance(50);
                c.rectangle = chance(25);
                c.bad_vertices = chance(2);
                c.mesh = c.pipeline <= specs.size() && specs[c.pipeline - 1].mesh;
                list.push_back(c);
            }
        const std::uint64_t memory = rng() % 24;  // 0: no budget
        const auto budget = [memory] { return memory; };
        const bool limited = chance(50);
        succeeded += compare("random submission " + std::to_string(round), specs, commands, workers,
                             limited ? std::function<std::uint64_t()>(budget) : std::function<std::uint64_t()>{})
                         .result == ReplayResult::success;
    }
    check(succeeded > 1000 && succeeded < 2900, "the random submissions both succeed and fail: " +
                                                    std::to_string(succeeded));
}

// Re-preparing after a prior compile pass reuses prior outcomes (both built handles and non-strict failed compute
// dispatches) without re-invoking build() or failing the submission.
void prior_compiles_reuse(PipelineWorkers& workers) {
    Registry registry({
        {.compute = true, .pending = true, .fails = false, .with_built = true},
        {.compute = true, .pending = true, .fails = true, .with_built = true},
    });
    const std::vector<CommandList> lists = record({{
        CommandSpec{.kind = CommandSpec::dispatch, .pipeline = 1},
        CommandSpec{.kind = CommandSpec::dispatch, .pipeline = 2},
    }});
    std::vector<PreparedList> prepared(1);
    ReplayFunctions vk = functions(registry);
    PipelineCompiles first;
    vk.compiles = &first;
    const ReplayResult r0 = prepare(lists[0], resolve, vk, prepared[0]);
    check(r0 == ReplayResult::success && first.size() == 2, "prior_compiles_reuse: pass 0 collected both");
    first.build([&](std::size_t count, const std::function<void(std::size_t)>& job) { workers.run(count, job); });
    const ReplayResult c0 = check_compiled_pipelines(first, r0);
    if (strict_replay()) {
        check(c0 == ReplayResult::unresolved_resource, "prior_compiles_reuse: strict fails on failed dispatch");
        return;
    }
    check(c0 == ReplayResult::success, "prior_compiles_reuse: non-strict tolerates failed dispatch");
    check(registry.owners[0]->attempts.size() == 1 && registry.owners[1]->attempts.size() == 1,
          "prior_compiles_reuse: each pipeline built once on pass 0");
    // Reset the failed flag on owner 2 so any redundant build() call on pass 1 would be caught in attempts.
    registry.owners[1]->failed = false;

    PipelineCompiles second;
    vk.compiles = &second;
    const ReplayResult r1 = prepare(lists[0], resolve, vk, prepared[0]);
    const ReplayResult b1 = build_pipelines(
        second, std::span(prepared), r1,
        [&](std::size_t count, const std::function<void(std::size_t)>& job) { workers.run(count, job); },
        {}, &first);
    check(b1 == ReplayResult::success, "prior_compiles_reuse: pass 1 succeeded");
    check(registry.owners[0]->attempts.size() == 1 && registry.owners[1]->attempts.size() == 1,
          "prior_compiles_reuse: pass 1 did not re-invoke build() on prior failure");
    check(prepared[0].commands[0].pipeline.pipeline == handle<VkPipeline>(0x1001),
          "prior_compiles_reuse: built compute handle attached");
    check(prepared[0].commands[1].pipeline.pipeline == VK_NULL_HANDLE,
          "prior_compiles_reuse: failed compute dispatch stays dropped");
}

// Serial compilation stops at the first refused graphics draw (matching pre-unlock serial semantics and
// d3d12_pipeline_workers::refused_graphics) instead of compiling later pipelines in the same submission.
void serial_stops_at_refused_draw() {
    Registry registry({
        {.compute = false, .pending = true, .fails = false},
        {.compute = false, .pending = true, .fails = true},
        {.compute = false, .pending = true, .fails = false},
        {.compute = false, .pending = true, .fails = false},
    });
    const std::vector<CommandList> lists = record({{
        CommandSpec{.kind = CommandSpec::draw, .pipeline = 1},
        CommandSpec{.kind = CommandSpec::draw, .pipeline = 2},
        CommandSpec{.kind = CommandSpec::draw, .pipeline = 3},
        CommandSpec{.kind = CommandSpec::draw, .pipeline = 4},
    }});
    std::vector<PreparedList> prepared(1);
    ReplayFunctions vk = functions(registry);
    PipelineCompiles compiles;
    vk.compiles = &compiles;
    const ReplayResult r = prepare(lists[0], resolve, vk, prepared[0]);
    check(r == ReplayResult::success && compiles.size() == 4, "serial_stops: collected all 4 draws");
    const ReplayResult out = build_pipelines_serial(compiles, std::span(prepared), r);
    check(out == ReplayResult::unsupported, "serial_stops: refused draw returns unsupported");
    check(registry.owners[0]->attempts.size() == 1 && registry.owners[1]->attempts.size() == 1,
          "serial_stops: built first draw and attempted refused second draw");
    check(registry.owners[2]->attempts.empty() && registry.owners[3]->attempts.empty(),
          "serial_stops: did not compile draws past the first refused draw");
}

// ---- Large pipelines within a budget. ----

// Pipelines that note the order they start in. One with `hold` does not end before `release` is called.
struct Shop {
    struct Item {
        Shop* shop = nullptr;
        ComputePipeline pipeline{};
        std::string name;
        bool hold = false;
    };
    std::mutex mutex;
    std::condition_variable changed;
    std::vector<std::string> started;
    std::vector<std::unique_ptr<Item>> items;
    bool released = false;
    int large_in = 0, most_large_in = 0;
    LargeModuleGate* gate = nullptr;  // set: a large one takes its place there, as in the layer

    ComputePipeline& add(const std::string& name, std::uint64_t estimate, bool large, bool hold = false) {
        auto& item = *items.emplace_back(std::make_unique<Item>());
        item.shop = this;
        item.name = name;
        item.hold = hold;
        item.pipeline.large = large;
        item.pipeline.estimate = estimate;
        item.pipeline.build_context = &item;
        item.pipeline.build = [](void* context) {
            auto& i = *static_cast<Item*>(context);
            Shop& shop = *i.shop;
            LargeModuleGate::Ticket place;
            if (shop.gate && i.pipeline.large) place = LargeModuleGate::Ticket(*shop.gate, i.pipeline.estimate);
            std::unique_lock lock(shop.mutex);
            shop.started.push_back(i.name);
            if (i.pipeline.large) shop.most_large_in = std::max(shop.most_large_in, ++shop.large_in);
            shop.changed.notify_all();
            if (i.hold) shop.changed.wait_for(lock, std::chrono::seconds(10), [&] { return shop.released; });
            if (i.pipeline.large) --shop.large_in;
            return handle<VkPipeline>(std::uintptr_t(&i) | 1);
        };
        return item.pipeline;
    }
    void fill(PipelineCompiles& compiles) {
        for (auto& item : items) compiles.add(item->pipeline, PipelineCompiles::User::dispatch);
    }
    // Whether `count` have started, within 10 s.
    bool wait_started(std::size_t count) {
        std::unique_lock lock(mutex);
        return changed.wait_for(lock, std::chrono::seconds(10), [&] { return started.size() >= count; });
    }
    void release() {
        std::lock_guard lock(mutex);
        released = true;
        changed.notify_all();
    }
};

// On `threads` threads, which take the jobs in order.
void on_threads(unsigned threads, std::size_t count, const std::function<void(std::size_t)>& job) {
    std::atomic<std::size_t> next = 0;
    std::vector<std::thread> workers;
    for (unsigned t = 0; t < threads; ++t)
        workers.emplace_back([&] {
            for (std::size_t i; (i = next++) < count;) job(i);
        });
    for (auto& w : workers) w.join();
}

using Names = std::vector<std::string>;

// The order and the jobs of the run, on one thread: the largest first, then the others, and the budget read once.
void budget_order() {
    struct Case {
        std::vector<std::uint64_t> large;
        std::uint64_t budget;
        std::size_t lanes;
        Names order;  // of the large ones
    };
    for (const Case& c : {Case{{5, 2, 3, 2, 1}, 5, 3, {"5", "3", "2", "2", "1"}},
                          Case{{5, 2, 3, 2, 1}, 4, 2, {"5", "3", "2", "2", "1"}},
                          Case{{5, 2, 3, 2, 1}, 0, 1, {"5", "2", "3", "2", "1"}},  // no budget: in the order they came
                          Case{{5, 2, 3, 2, 1}, 100, 5, {"5", "3", "2", "2", "1"}},
                          Case{{9}, 5, 1, {"9"}},        // more than the budget: one, alone
                          Case{{9, 9}, 5, 1, {"9", "9"}},
                          Case{{9, 1}, 5, 1, {"9", "1"}},
                          Case{{4, 4}, 8, 2, {"4", "4"}}}) {
        Shop shop;
        for (const auto size : c.large) shop.add(std::to_string(size), size, true);
        for (int i = 0; i < 3; ++i) shop.add("s" + std::to_string(i), 0, false);
        PipelineCompiles compiles;
        shop.fill(compiles);
        std::size_t units = 0;
        unsigned reads = 0;
        compiles.build([&](std::size_t count, const std::function<void(std::size_t)>& job) {
            units = count;
            for (std::size_t i = 0; i < count; ++i) job(i);
        }, [&] { ++reads; return c.budget; });
        Names want = c.order;
        for (int i = 0; i < 3; ++i) want.push_back("s" + std::to_string(i));
        const std::string name = "budget " + std::to_string(c.budget) + " of " + std::to_string(c.large.size());
        check(shop.started == want, name + ": another order");
        check(units == c.lanes + 3, name + ": " + std::to_string(units) + " jobs, not " + std::to_string(c.lanes + 3));
        check(reads == 1, name + ": the budget read " + std::to_string(reads) + " times");
        for (std::size_t i = 0; i < shop.items.size(); ++i)
            check(compiles.built(PipelineCompiles::Ticket(i + 1)) != VK_NULL_HANDLE, name + ": not built");
    }
    // Without a large pipeline the budget is not read, and the jobs are the pipelines.
    Shop shop;
    for (int i = 0; i < 3; ++i) shop.add("s" + std::to_string(i), 0, false);
    PipelineCompiles compiles;
    shop.fill(compiles);
    std::size_t units = 0;
    unsigned reads = 0;
    compiles.build([&](std::size_t count, const std::function<void(std::size_t)>& job) {
        units = count;
        for (std::size_t i = 0; i < count; ++i) job(i);
    }, [&] { ++reads; return 100u; });
    check(units == 3 && reads == 0, "the budget read for pipelines that are not large");
}

// A large one that does not fit beside the one in is not waited for while others are left: they come first.
void budget_others_first() {
    Shop shop;
    shop.add("8", 8, true, true);
    shop.add("4a", 4, true);
    shop.add("4b", 4, true);
    for (int i = 0; i < 3; ++i) shop.add("s" + std::to_string(i), 0, false);
    PipelineCompiles compiles;
    shop.fill(compiles);
    std::thread building([&] {
        compiles.build([&](std::size_t count, const std::function<void(std::size_t)>& job) {
            check(count == 2 + 3, "two lanes: the two of 4 fit 8 together");
            on_threads(2, count, job);
        }, [] { return 8; });
    });
    const bool all = shop.wait_started(6);  // "8" is held: the other lane does the rest
    Names seen;
    {
        std::lock_guard lock(shop.mutex);
        seen = shop.started;
    }
    shop.release();
    building.join();
    check(all && seen.size() == 6 && seen[0] == "8", "the large one first");
    check(all && std::set<std::string>(seen.begin() + 1, seen.begin() + 4) == std::set<std::string>{"s0", "s1", "s2"},
          "the others before the large ones that do not fit");
    check(all && seen[4] == "4a" && seen[5] == "4b", "then the next large ones");
}

// Those that fit together start together, before the others.
void budget_fits_together() {
    Shop shop;
    shop.add("3", 3, true, true);
    shop.add("5", 5, true, true);
    for (int i = 0; i < 2; ++i) shop.add("s" + std::to_string(i), 0, false);
    PipelineCompiles compiles;
    shop.fill(compiles);
    std::thread building([&] {
        compiles.build([&](std::size_t count, const std::function<void(std::size_t)>& job) { on_threads(2, count, job); },
                       [] { return 8; });
    });
    const bool both = shop.wait_started(2);
    Names seen;
    {
        std::lock_guard lock(shop.mutex);
        seen = shop.started;
    }
    shop.release();
    building.join();
    check(both && std::set<std::string>(seen.begin(), seen.begin() + 2) == std::set<std::string>{"3", "5"},
          "two that fit the budget are in together");
    check(shop.most_large_in == 2, "two that fit the budget at once");
}

// The room that a large one leaves is the next large one's, before the others.
void budget_room_is_given_back() {
    Shop shop;
    shop.add("5", 5, true, true);
    for (const char* name : {"3a", "3b", "3c"}) shop.add(name, 3, true);
    for (int i = 0; i < 3; ++i) shop.add("s" + std::to_string(i), 0, false);
    PipelineCompiles compiles;
    shop.fill(compiles);
    std::thread building([&] {
        compiles.build([&](std::size_t count, const std::function<void(std::size_t)>& job) {
            check(count == 2 + 3, "two lanes: two of 3 fit 8, three do not");
            on_threads(2, count, job);
        }, [] { return 8; });
    });
    const bool all = shop.wait_started(7);  // "5" is held: the other lane has room for one of 3 at a time
    Names seen;
    {
        std::lock_guard lock(shop.mutex);
        seen = shop.started;
    }
    shop.release();
    building.join();
    check(all && Names(seen.begin(), seen.begin() + 4) == Names{"5", "3a", "3b", "3c"},
          "a large one does not leave its room");
}

// Graphics and compute pipelines share the budget: what each is expected to take counts alike.
void budget_graphics_and_compute() {
    Shop shop;
    shop.add("3", 3, true);
    shop.add("4", 4, true);
    GraphicsPipeline graphics{};
    graphics.large = true;
    graphics.estimate = 5;
    graphics.variant = [&](std::uint32_t) {
        std::lock_guard lock(shop.mutex);
        shop.started.push_back("5");
        return handle<VkPipeline>(5);
    };
    PipelineCompiles compiles;
    shop.fill(compiles);
    compiles.add(graphics, 0);
    std::size_t units = 0;
    compiles.build([&](std::size_t count, const std::function<void(std::size_t)>& job) {
        units = count;
        for (std::size_t i = 0; i < count; ++i) job(i);
    }, [] { return 8; });
    check(shop.started == Names{"5", "4", "3"}, "the largest first, graphics or compute");
    check(units == 2, "3 and 4 fit 8, and 5 does not beside them");
}

// The lanes are no more than could be in: a thread is not kept waiting for room there is none of.
void budget_lanes_are_bounded() {
    Shop shop;
    for (int i = 0; i < 3; ++i) shop.add("5" + std::to_string(i), 5, true);
    for (int i = 0; i < 6; ++i) shop.add("s" + std::to_string(i), 0, false);
    PipelineCompiles compiles;
    shop.fill(compiles);
    compiles.build([&](std::size_t count, const std::function<void(std::size_t)>& job) {
        check(count == 2 + 6, "10 bytes hold two of 5");
        on_threads(4, count, job);
    }, [] { return 10; });
    check(shop.started.size() == 9 && shop.most_large_in <= 2, "no more than two large ones at once");
}

// Every one of them is built once, with the gate of the runtime holding what does not fit, whatever sizes and budget.
void budget_random(PipelineWorkers& workers) {
    std::mt19937 rng(20251127);
    for (int round = 0; round < 300; ++round) {
        Shop shop;
        const std::uint64_t budget = rng() % 20;
        LargeModuleGate gate([budget] { return budget; });
        shop.gate = &gate;
        const unsigned large = rng() % 10, others = rng() % 10;
        for (unsigned i = 0; i < large; ++i) shop.add("l" + std::to_string(i), rng() % 12, true);
        for (unsigned i = 0; i < others; ++i) shop.add("s" + std::to_string(i), 0, false);
        PipelineCompiles compiles;
        shop.fill(compiles);
        compiles.build([&](std::size_t count, const std::function<void(std::size_t)>& job) { workers.run(count, job); },
                       [budget] { return budget; });
        Names names = shop.started;
        std::sort(names.begin(), names.end());
        check(std::adjacent_find(names.begin(), names.end()) == names.end() && names.size() == large + others,
              "random budget " + std::to_string(round) + ": a pipeline built twice or not at all");
        check(gate.inflight() == 0, "random budget " + std::to_string(round) + ": a place not given back");
    }
}

}  // namespace

int main() {
    for (const unsigned threads : {0u, 1u, 4u}) {
        PipelineWorkers workers(threads);
        cases(workers);
        if (threads) many(workers);
        reused_lists(workers);
        prior_compiles_reuse(workers);
        random_submissions(workers);
        budget_random(workers);
    }
    serial_stops_at_refused_draw();
    budget_order();
    budget_others_first();
    budget_fits_together();
    budget_room_is_given_back();
    budget_graphics_and_compute();
    budget_lanes_are_bounded();
    check(large_together == 0, "two large pipelines were built at once");
    if (failures) return 1;
    std::printf("replay_pipeline_compiles_test: passed\n");
    return 0;
}
