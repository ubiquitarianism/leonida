#pragma once
// Work-aware timeout clock and cycle-safe producer dependency analysis for console command-list memory waits
// (runtime/shims/d3d12_xs_list_values.h, docs/d3d12_xs_console_abi.md §5.5).
//
// A memory wait gate needs a bounded timeout so an unmodelled write (one the runtime never lands) cannot hang a queue
// forever. Counting wall-clock time blindly cuts waits short whenever the runtime is busy running held work (e.g.
// multi-second CPU shadow/texture copies inside the releaser) or while an in-flight producer write has not finished on
// the GPU yet, running dependent queues out of order.
//
// This header separates the policy from Windows/D3D12 handles so it can be verified portably under sanitizers:
// - `pause_reason()` decides whether submitted or held work can still satisfy a gate without creating a deadlock cycle:
//   1. `releaser_busy`: held work from an already-opened gate is currently being released/submitted;
//   2. `matching_producer`: an in-flight WriteGroup whose final write to a waited address can satisfy an unmet wait, and
//      every pending gate preceding that WriteGroup is cycle-free with respect to the waiting queue;
//   3. `active_work`: unrelated unblocked GPU work (a WriteGroup or action gate with no pending gate ahead of it) is
//      still executing, paused only up to `active_pause_cap_us` so a continuously streaming queue cannot starve an
//      unmodelled wait.
// - `GateTimeoutState` tracks idle wait time (`elapsed_us`) separately from paused time, expiring once `elapsed_us`
//   reaches `budget_us`. Every pause reason (`releaser_busy` and `matching_producer` included) is bounded by a finite
//   `max_pause_cap_us` so a producer stuck behind an untracked dependency (e.g. a title fence wait or a hung Signal)
//   still expires instead of hanging forever, and demoted (`cut_short`) gates scale both caps with their short budget.
//
// Pure C++20 standard library only.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace leonida::list_value_wait {

enum class PauseReason : std::uint8_t {
    none = 0,               // no submitted or held work can still satisfy this gate -> timeout clock ticks
    releaser_busy = 1,      // releaser thread is executing or draining opened gates
    matching_producer = 2,  // an acyclic in-flight WriteGroup can satisfy an unmet wait of this gate
    active_work = 3,        // unrelated unblocked submitted work is still executing (bounded by active_pause_cap_us)
};

struct WaitPredicate {
    std::uint64_t address = 0;
    std::uint64_t reference = 0;
    std::uint32_t comparison = 0;
    bool wide = false;
};

struct PendingWrite {
    std::uint64_t address = 0;
    std::uint64_t value = 0;
    std::uint8_t bytes = 0;
    bool timestamp = false;  // GPU/system timestamp: final value is unknown until landing
};

struct PendingWriteGroup {
    const void* queue = nullptr;
    std::uint64_t after_gate = 0;  // highest pending gate value on `queue` ahead of this group when submitted (0: none)
    std::span<const PendingWrite> writes;
};

struct PendingGate {
    const void* queue = nullptr;
    std::uint64_t value = 0;
    bool is_action = false;                      // waits only on earlier GPU work on `queue`, never on memory
    std::span<const WaitPredicate> unmet_waits;  // waits not currently satisfied in memory (empty for action gates)
};

struct PauseCaps {
    std::uint64_t active_us = 0;  // cap on pausing solely for unrelated active_work
    std::uint64_t max_us = 0;     // hard finite cap across ALL pause reasons (releaser_busy, matching_producer, active_work)
};

// Pause caps scaled with the gate's own timeout budget (`gate_timeout_ms`) and demotion state (`cut_short`):
// - Normal gates: `active_us = 300x` and `max_us = 1500x` of `gate_timeout_ms`, capped at 10 min. At the default
//   200 ms timeout this allows up to 60 s for unrelated GPU work and up to 300 s (5 min) while the releaser or an
//   acyclic matching producer is working (covering multi-minute world-phase pipeline builds), while guaranteeing a
//   finite exit if a producer is held behind an untracked dependency (title fence wait or stuck Signal).
// - Demoted (`cut_short`) gates: `active_us = 1x` and `max_us = 4x` of `gate_timeout_ms` (4 ms / 16 ms at the 4 ms
//   demoted budget), so an unmodelled wait only pays a few milliseconds instead of inheriting the full timeout's cap.
inline constexpr PauseCaps default_pause_caps(unsigned gate_timeout_ms, bool cut_short = false) {
    constexpr std::uint64_t kMaxCapMs = 600000u;  // 10 min hard ceiling
    const std::uint64_t base_ms = gate_timeout_ms ? std::uint64_t{gate_timeout_ms} : 1u;
    const auto scale_ms = [base_ms](std::uint64_t mult) -> std::uint64_t {
        return base_ms > kMaxCapMs / mult ? kMaxCapMs : base_ms * mult;
    };
    const std::uint64_t active_ms = cut_short ? (base_ms < kMaxCapMs ? base_ms : kMaxCapMs) : scale_ms(300u);
    const std::uint64_t max_ms = cut_short ? scale_ms(4u) : scale_ms(1500u);
    return {active_ms * 1000u, max_ms * 1000u};
}

inline constexpr std::uint64_t default_active_pause_cap_us(unsigned gate_timeout_ms, bool cut_short = false) {
    return default_pause_caps(gate_timeout_ms, cut_short).active_us;
}

inline constexpr std::uint64_t default_max_pause_cap_us(unsigned gate_timeout_ms, bool cut_short = false) {
    return default_pause_caps(gate_timeout_ms, cut_short).max_us;
}

struct GateTimeoutState {
    std::uint64_t budget_us = 0;
    std::uint64_t elapsed_us = 0;
    std::uint64_t active_paused_us = 0;
    std::uint64_t active_pause_cap_us = 0;
    std::uint64_t total_paused_us = 0;
    std::uint64_t max_pause_cap_us = 0;
    bool paused_ever = false;

    void reset(std::uint64_t budget, std::uint64_t active_cap, std::uint64_t max_cap) {
        budget_us = budget;
        elapsed_us = 0;
        active_paused_us = 0;
        total_paused_us = 0;
        max_pause_cap_us = max_cap;
        active_pause_cap_us = active_cap < max_cap ? active_cap : max_cap;
        paused_ever = false;
    }

    // Advances the gate by `dt_us` under `reason`. Returns true once the idle timeout budget is exhausted.
    // Every pause reason is bounded by `max_pause_cap_us`; unrelated `active_work` is additionally bounded by
    // `active_pause_cap_us`.
    bool advance(std::uint64_t dt_us, PauseReason reason) {
        std::uint64_t room = 0;
        if (reason == PauseReason::releaser_busy || reason == PauseReason::matching_producer) {
            room = total_paused_us < max_pause_cap_us ? max_pause_cap_us - total_paused_us : 0u;
        } else if (reason == PauseReason::active_work) {
            const std::uint64_t total_room =
                total_paused_us < max_pause_cap_us ? max_pause_cap_us - total_paused_us : 0u;
            const std::uint64_t active_room =
                active_paused_us < active_pause_cap_us ? active_pause_cap_us - active_paused_us : 0u;
            room = total_room < active_room ? total_room : active_room;
        }
        if (room > 0) {
            if (dt_us == 0) return false;
            paused_ever = true;
            const std::uint64_t absorb = dt_us < room ? dt_us : room;
            total_paused_us += absorb;
            if (reason == PauseReason::active_work) active_paused_us += absorb;
            dt_us -= absorb;
        }
        if (dt_us > UINT64_MAX - elapsed_us)
            elapsed_us = UINT64_MAX;
        else
            elapsed_us += dt_us;
        return elapsed_us >= budget_us;
    }
};

namespace detail {

// True when `group`'s final write touching [wait.address, wait.address + need) lands a value that meets `wait`
// (or is a timestamp write to the exact address whose runtime value is unknown until landing).
template <class MetFn>
inline bool write_group_matches_wait(const PendingWriteGroup& group, const WaitPredicate& wait, MetFn&& met) {
    const std::uint64_t need = wait.wide ? 8u : 4u;
    if (!wait.address || wait.address > UINT64_MAX - need) return false;
    const std::uint64_t wait_end = wait.address + need;
    for (std::size_t i = group.writes.size(); i-- > 0;) {
        const PendingWrite& w = group.writes[i];
        if (!w.bytes || w.address > UINT64_MAX - w.bytes) continue;
        const std::uint64_t write_end = w.address + w.bytes;
        if (w.address >= wait_end || wait.address >= write_end) continue;
        if (w.address == wait.address && w.bytes >= need &&
            (w.timestamp || met(wait.comparison, w.value, wait.reference, wait.wide)))
            return true;
        // A later write in the same group overwrote or partially overlapped the waited range with a non-matching value.
        return false;
    }
    return false;
}

// True when no pending gate in `gates` precedes `(queue, after_gate)`.
inline bool is_unblocked(std::span<const PendingGate> gates, const void* queue, std::uint64_t after_gate) {
    if (!after_gate) return true;
    for (const PendingGate& g : gates)
        if (g.queue == queue && g.value <= after_gate) return false;
    return true;
}

// True when `gate_idx` (directly or transitively via matching in-flight writes) depends on a gate on `target_queue`.
template <class MetFn>
inline bool gate_can_reach_queue(std::size_t gate_idx, const void* target_queue,
                                 std::span<const PendingWriteGroup> writes, std::span<const PendingGate> gates,
                                 MetFn&& met, std::vector<std::size_t>& visiting) {
    const PendingGate& g = gates[gate_idx];
    if (g.queue == target_queue) return true;
    if (g.is_action) return false;
    if (std::find(visiting.begin(), visiting.end(), gate_idx) != visiting.end()) return false;
    visiting.push_back(gate_idx);
    for (const WaitPredicate& wait : g.unmet_waits) {
        for (const PendingWriteGroup& w : writes) {
            if (!write_group_matches_wait(w, wait, met)) continue;
            if (!w.after_gate) continue;
            for (std::size_t b = 0; b < gates.size(); ++b) {
                if (gates[b].queue != w.queue || gates[b].value > w.after_gate) continue;
                if (gates[b].queue == target_queue ||
                    gate_can_reach_queue(b, target_queue, writes, gates, met, visiting)) {
                    visiting.pop_back();
                    return true;
                }
            }
        }
    }
    visiting.pop_back();
    return false;
}

}  // namespace detail

// Decides whether the timeout clock of `gates[target_idx]` should pause on this pump pass.
// `met(comparison, value, reference, wide)` must implement the same predicate as `console_value_met`.
template <class MetFn>
inline PauseReason pause_reason(bool releaser_busy, std::span<const PendingWriteGroup> writes,
                                std::span<const PendingGate> gates, std::size_t target_idx, MetFn&& met) {
    if (releaser_busy) return PauseReason::releaser_busy;
    if (target_idx >= gates.size()) return PauseReason::none;
    const PendingGate& target = gates[target_idx];
    if (target.is_action) return PauseReason::none;

    // 1. A matching in-flight WriteGroup whose preceding gates (if any) do not cycle back to `target.queue`.
    std::vector<std::size_t> visiting;
    for (const WaitPredicate& wait : target.unmet_waits) {
        for (const PendingWriteGroup& w : writes) {
            if (!detail::write_group_matches_wait(w, wait, met)) continue;
            bool cyclic = false;
            if (w.after_gate) {
                for (std::size_t b = 0; b < gates.size(); ++b) {
                    if (gates[b].queue != w.queue || gates[b].value > w.after_gate) continue;
                    visiting.clear();
                    if (gates[b].queue == target.queue ||
                        detail::gate_can_reach_queue(b, target.queue, writes, gates, met, visiting)) {
                        cyclic = true;
                        break;
                    }
                }
            }
            if (!cyclic) return PauseReason::matching_producer;
        }
    }

    // 2. Unrelated unblocked submitted GPU work (an unblocked WriteGroup on any queue, or an unblocked action gate on
    //    another queue) that could still write memory via a shader/copy or unblock a CPU producer.
    for (const PendingWriteGroup& w : writes) {
        if (detail::is_unblocked(gates, w.queue, w.after_gate)) return PauseReason::active_work;
    }
    for (const PendingGate& g : gates) {
        if (!g.is_action || g.queue == target.queue) continue;
        if (g.value == 0 || detail::is_unblocked(gates, g.queue, g.value - 1)) return PauseReason::active_work;
    }
    return PauseReason::none;
}

}  // namespace leonida::list_value_wait
