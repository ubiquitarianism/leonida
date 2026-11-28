// Process-wide counts of calls recorded and replay attempts. A command list can replay more than once.
#pragma once
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>

namespace leonida::indirect_stats {
enum class Event { recorded, emitted, record_skip, replay_skip };
// One flag per reason, shared across resolver instantiations.
enum Reason { compound, disabled, pipeline, root_layout, viewport, mesh, topology, first_instance, entry, robust_index,
              count_limit, multi_draw, count_feature, raw_count_limit, argument_usage, count_usage,
              divisor, pending_pipeline, index_view, index_range, stateless, reason_count };

// D3D_PRIMITIVE_TOPOLOGY values whose Vulkan dynamic topology (or rectangle-list geometry-stage variant) needs no CPU
// vertex/index-count expansion at prepare time: POINTLIST(1), LINELIST(2), LINESTRIP(3), TRIANGLELIST(4),
// TRIANGLESTRIP(5), *_ADJ(10..13), and console RECTLIST(17). UNDEFINED(0), QUADLIST(19), and patch lists remain skipped.
inline constexpr bool topology_supported(std::uint32_t value) {
    return (value >= 1 && value <= 5) || (value >= 10 && value <= 13) || value == 17;
}

inline constexpr const char* reason_name(Reason reason) {
    constexpr const char* names[]{"compound signature", "opt-out", "missing pipeline", "missing root signature",
        "empty viewport", "mesh indirect", "topology", "drawIndirectFirstInstance", "missing Vulkan entry point",
        "robust sized index binding", "maxDrawIndirectCount", "multiDrawIndirect", "drawIndirectCount",
        "counted draw requires maxDrawIndirectCount == UINT32_MAX", "argument buffer lacks indirect creation usage",
        "count buffer lacks indirect creation usage", "nonunit vertex divisor", "pipeline pending",
        "invalid index view", "index view outside buffer or misaligned", "legacy command has no graphics state"};
    static_assert(sizeof(names) / sizeof(*names) == reason_count);
    return static_cast<unsigned>(reason) < reason_count ? names[reason] : "unknown";
}

inline constexpr const char* reason_tag(Reason reason) {
    constexpr const char* tags[]{"compound", "disabled", "pipeline", "root_layout", "viewport", "mesh", "topology",
        "first_instance", "entry", "robust_index", "count_limit", "multi_draw", "count_feature", "raw_count_limit",
        "argument_usage", "count_usage", "divisor", "pending_pipeline", "index_view", "index_range", "stateless"};
    static_assert(sizeof(tags) / sizeof(*tags) == reason_count);
    return static_cast<unsigned>(reason) < reason_count ? tags[reason] : "unknown";
}

struct Counts {
    std::atomic<unsigned long long> values[4]{};
    std::atomic<unsigned long long> by_reason[reason_count]{};
    void report(std::FILE* out = stderr) const {
        char line[1280];
        int pos = std::snprintf(
            line, sizeof(line),
            "[rt] indirect graphics: recorded=%llu emitted=%llu record_skips=%llu replay_skips=%llu",
            values[0].load(std::memory_order_relaxed), values[1].load(std::memory_order_relaxed),
            values[2].load(std::memory_order_relaxed), values[3].load(std::memory_order_relaxed));
        if (pos < 0) return;
        if (static_cast<std::size_t>(pos) >= sizeof(line)) pos = static_cast<int>(sizeof(line) - 1);
        bool first_reason = true;
        for (unsigned i = 0; i < reason_count; ++i) {
            const auto count = by_reason[i].load(std::memory_order_relaxed);
            if (!count) continue;
            const int rem = static_cast<int>(sizeof(line)) - pos;
            if (rem <= 2) break;
            const int n = std::snprintf(line + pos, static_cast<std::size_t>(rem), "%s%s=%llu",
                                        first_reason ? " [" : ", ",
                                        reason_tag(static_cast<Reason>(i)), count);
            if (n <= 0) break;
            pos += (n < rem) ? n : (rem - 1);
            first_reason = false;
        }
        if (!first_reason && static_cast<std::size_t>(pos) + 1 < sizeof(line)) line[pos++] = ']';
        std::fprintf(out, "%.*s\n", pos, line);
    }
    ~Counts() {
        for (const auto& value : values) {
            if (value.load(std::memory_order_relaxed)) {
                report();
                break;
            }
        }
    }
};

inline Counts& counts() {
    static Counts instance;
    return instance;
}

inline unsigned long long reason_skips(Reason reason) {
    return static_cast<unsigned>(reason) < reason_count
               ? counts().by_reason[reason].load(std::memory_order_relaxed)
               : 0ull;
}

inline void note(Event event) {
    auto& c = counts();
    const auto n = 1 + c.values[static_cast<unsigned>(event)].fetch_add(1, std::memory_order_relaxed);
    if ((n & (n - 1)) == 0) c.report();
}

inline void skip(Event event, Reason reason) {
    if (static_cast<unsigned>(reason) >= reason_count) return;
    static std::atomic_flag shown[reason_count]{};
    if (!shown[reason].test_and_set(std::memory_order_relaxed))
        std::fprintf(stderr, "[rt] indirect graphics skipped: %s\n", reason_name(reason));
    counts().by_reason[reason].fetch_add(1, std::memory_order_relaxed);
    note(event);
}
}
