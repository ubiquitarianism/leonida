// Portable CPU-only unit tests for runtime/src/list_value_wait_budget.h (RT-LIST-VALUE-WAIT-WORK-AWARE-TIMEOUT):
// work-aware timeout pausing, finite pause caps across all pause reasons (including releaser_busy and
// matching_producer), demoted-gate cap scaling, cycle detection across multi-queue write/gate chains, overwrite-aware
// write matching, action gates, and a mock Pump tick driver. No Windows, D3D12, or Vulkan headers; no game files.
#include "../src/list_value_wait_budget.h"

#include <climits>
#include <cstdio>
#include <vector>

#define CHECK(cond) do {                                                        \
    if (!(cond)) {                                                              \
        std::fprintf(stderr, "line %d: CHECK failed: %s\n", __LINE__, #cond);   \
        return 1;                                                               \
    }                                                                           \
} while (0)

namespace {

constexpr bool test_met(std::uint32_t comparison, std::uint64_t value, std::uint64_t reference, bool wide) {
    if (!wide) {
        value &= 0xffffffffu;
        reference &= 0xffffffffu;
    }
    switch (comparison) {
    case 2: return value < reference;
    case 3: return value == reference;
    case 4: return value <= reference;
    case 5: return value > reference;
    case 6: return value != reference;
    case 7: return value >= reference;
    default: return true;
    }
}

}  // namespace

int main() {
    using namespace leonida::list_value_wait;

    int q1_tag = 0, q2_tag = 0, q3_tag = 0;
    const void* const q1 = &q1_tag;
    const void* const q2 = &q2_tag;
    const void* const q3 = &q3_tag;

    // 1. Idle unmodelled wait: PauseReason::none -> expires at exact budget_us.
    {
        const WaitPredicate waits[] = {{0x1000, 5, 7, true}};  // >= 5
        const PendingGate gates[] = {{q1, 1, false, waits}};
        CHECK(pause_reason(false, {}, gates, 0, test_met) == PauseReason::none);

        GateTimeoutState t;
        const PauseCaps caps = default_pause_caps(200, false);
        t.reset(200'000, caps.active_us, caps.max_us);
        CHECK(!t.advance(100'000, PauseReason::none));
        CHECK(t.elapsed_us == 100'000 && !t.paused_ever);
        CHECK(!t.advance(99'999, PauseReason::none));
        CHECK(t.advance(1, PauseReason::none));
        CHECK(t.elapsed_us == 200'000);
    }

    // 2. Releaser actively executing held work pauses the timeout clock (covering multi-minute world-phase pipeline
    //    builds without burning idle budget), yet still expires at the finite max_pause_cap_us if a Signal hangs.
    {
        const WaitPredicate waits[] = {{0x1000, 5, 7, true}};
        const PendingGate gates[] = {{q1, 1, false, waits}};
        CHECK(pause_reason(true, {}, gates, 0, test_met) == PauseReason::releaser_busy);

        GateTimeoutState t;
        const PauseCaps caps = default_pause_caps(200, false);
        CHECK(caps.active_us == 60'000'000ull && caps.max_us == 300'000'000ull);
        t.reset(200'000, caps.active_us, caps.max_us);
        // 120 seconds of releaser executing pipeline builds/copies burns 0 us of idle timeout budget.
        for (int i = 0; i < 120; ++i) CHECK(!t.advance(1'000'000, PauseReason::releaser_busy));
        CHECK(t.elapsed_us == 0 && t.total_paused_us == 120'000'000ull && t.paused_ever);
        // Once idle again, still gets its full 200 ms budget before expiring.
        CHECK(!t.advance(199'000, PauseReason::none));
        CHECK(t.advance(1'000, PauseReason::none));

        // Hung Signal (releaser_busy forever) hits the finite max_pause_cap_us and then drains budget_us.
        GateTimeoutState hung;
        hung.reset(200'000, caps.active_us, caps.max_us);
        for (int i = 0; i < 300; ++i) CHECK(!hung.advance(1'000'000, PauseReason::releaser_busy));
        CHECK(hung.total_paused_us == 300'000'000ull && hung.elapsed_us == 0);
        CHECK(!hung.advance(199'000, PauseReason::releaser_busy));
        CHECK(hung.advance(1'000, PauseReason::releaser_busy));
        CHECK(hung.elapsed_us == 200'000);
    }

    // 3. Matching unblocked producer write on another queue (Wait-before-Signal) -> PauseReason::matching_producer.
    {
        const WaitPredicate waits[] = {{0x2000, 10, 7, true}};  // >= 10
        const PendingWrite w_ok[] = {{0x2000, 12, 8, false}};
        const PendingWriteGroup groups[] = {{q2, 0, w_ok}};
        const PendingGate gates[] = {{q1, 1, false, waits}};
        CHECK(pause_reason(false, groups, gates, 0, test_met) == PauseReason::matching_producer);

        // Timestamp write to the same address also matches (final value unknown until landing).
        const PendingWrite w_ts[] = {{0x2000, 0, 8, true}};
        const PendingWriteGroup ts_groups[] = {{q2, 0, w_ts}};
        CHECK(pause_reason(false, ts_groups, gates, 0, test_met) == PauseReason::matching_producer);

        // Matching write overwritten later in the same WriteGroup by a non-matching value does NOT match,
        // falling back to active_work (since q2's group is unblocked).
        const PendingWrite w_overwritten[] = {{0x2000, 12, 8, false}, {0x2000, 3, 8, false}};
        const PendingWriteGroup ow_groups[] = {{q2, 0, w_overwritten}};
        CHECK(pause_reason(false, ow_groups, gates, 0, test_met) == PauseReason::active_work);

        // Partial overlap by a later write in the same group invalidates the earlier matching write.
        const PendingWrite w_partial[] = {{0x2000, 12, 8, false}, {0x2004, 0, 4, false}};
        const PendingWriteGroup part_groups[] = {{q2, 0, w_partial}};
        CHECK(pause_reason(false, part_groups, gates, 0, test_met) == PauseReason::active_work);

        // 4-byte write cannot satisfy an 8-byte (wide) wait.
        const PendingWrite w_narrow[] = {{0x2000, 12, 4, false}};
        const PendingWriteGroup nar_groups[] = {{q2, 0, w_narrow}};
        CHECK(pause_reason(false, nar_groups, gates, 0, test_met) == PauseReason::active_work);

        // 8-byte write CAN satisfy a 4-byte wait (low 32 bits compared).
        const WaitPredicate wait32[] = {{0x2000, 10, 7, false}};
        const PendingWrite w_hi_lo[] = {{0x2000, 0x100000000ull | 12ull, 8, false}};
        const PendingWriteGroup hilo_groups[] = {{q2, 0, w_hi_lo}};
        const PendingGate gates32[] = {{q1, 1, false, wait32}};
        CHECK(pause_reason(false, hilo_groups, gates32, 0, test_met) == PauseReason::matching_producer);
    }

    // 4. Same-queue ordering: a write submitted BEFORE the gate (after_gate < gate.value) IS a live producer;
    //    a write submitted AFTER the gate (after_gate >= gate.value) is a self-deadlock and MUST NOT pause the gate.
    {
        const WaitPredicate waits[] = {{0x3000, 1, 7, true}};
        const PendingWrite w[] = {{0x3000, 1, 8, false}};
        const PendingGate gates[] = {{q1, 2, false, waits}};

        const PendingWriteGroup before_gate[] = {{q1, 1, w}};  // after_gate 1 < gate.value 2 (unblocked!)
        CHECK(pause_reason(false, before_gate, gates, 0, test_met) == PauseReason::matching_producer);

        const PendingWriteGroup behind_gate[] = {{q1, 2, w}};  // after_gate 2 == gate.value 2 (held by itself!)
        CHECK(pause_reason(false, behind_gate, gates, 0, test_met) == PauseReason::none);
    }

    // 5. Mutual 2-queue cycle (Q1 waits on Q2's write behind G2; Q2 waits on Q1's write behind G1):
    //    NEITHER gate pauses, so the cycle breaks cleanly instead of deadlocking.
    {
        const WaitPredicate w1[] = {{0x4000, 1, 7, true}};
        const WaitPredicate w2[] = {{0x5000, 1, 7, true}};
        const PendingWrite pw1[] = {{0x5000, 1, 8, false}};  // Q1 writes what Q2 waits for, behind Q1's gate 1
        const PendingWrite pw2[] = {{0x4000, 1, 8, false}};  // Q2 writes what Q1 waits for, behind Q2's gate 1
        const PendingWriteGroup groups[] = {{q1, 1, pw1}, {q2, 1, pw2}};
        const PendingGate gates[] = {{q1, 1, false, w1}, {q2, 1, false, w2}};
        CHECK(pause_reason(false, groups, gates, 0, test_met) == PauseReason::none);
        CHECK(pause_reason(false, groups, gates, 1, test_met) == PauseReason::none);
    }

    // 6. 3-queue cycle involving a SECOND (non-front) gate on Q2:
    //    Q1 gate 1 waits on W2 (behind Q2 gates 1 & 2); Q2 gate 2 waits on W1 (behind Q1 gate 1).
    //    Checking all gates <= after_gate catches this indirect cycle!
    {
        const WaitPredicate w_q1[] = {{0x6000, 1, 7, true}};
        const WaitPredicate w_q2_g1[] = {{0x6100, 1, 7, true}};  // unrelated front gate on Q2
        const WaitPredicate w_q2_g2[] = {{0x6200, 1, 7, true}};  // second gate on Q2 waits on Q1
        const PendingWrite pw_q1[] = {{0x6200, 1, 8, false}};    // behind Q1 gate 1
        const PendingWrite pw_q2[] = {{0x6000, 1, 8, false}};    // behind Q2 gate 2 (after_gate = 2)
        const PendingWriteGroup groups[] = {{q1, 1, pw_q1}, {q2, 2, pw_q2}};
        const PendingGate gates[] = {
            {q1, 1, false, w_q1},
            {q2, 1, false, w_q2_g1},
            {q2, 2, false, w_q2_g2},
        };
        CHECK(pause_reason(false, groups, gates, 0, test_met) == PauseReason::none);
        CHECK(pause_reason(false, groups, gates, 2, test_met) == PauseReason::none);
    }

    // 7. Acyclic 3-queue chain (Q1 -> Q2 -> Q3 where Q3's gate has no producer):
    //    Q3 does NOT pause (times out first), while Q2 and Q1 wait for their upstream producers to run in order!
    {
        const WaitPredicate w_q1[] = {{0x7000, 1, 7, true}};
        const WaitPredicate w_q2[] = {{0x7100, 1, 7, true}};
        const WaitPredicate w_q3[] = {{0x7200, 1, 7, true}};  // unmodelled wait on Q3
        const PendingWrite pw_q2[] = {{0x7000, 1, 8, false}};  // Q2 feeds Q1 (behind Q2 gate 1)
        const PendingWrite pw_q3[] = {{0x7100, 1, 8, false}};  // Q3 feeds Q2 (behind Q3 gate 1)
        const PendingWriteGroup groups[] = {{q2, 1, pw_q2}, {q3, 1, pw_q3}};
        const PendingGate gates[] = {
            {q1, 1, false, w_q1},
            {q2, 1, false, w_q2},
            {q3, 1, false, w_q3},
        };
        CHECK(pause_reason(false, groups, gates, 0, test_met) == PauseReason::matching_producer);
        CHECK(pause_reason(false, groups, gates, 1, test_met) == PauseReason::matching_producer);
        CHECK(pause_reason(false, groups, gates, 2, test_met) == PauseReason::none);
    }

    // 8. Action gate on another queue counts as active_work when unblocked, and does NOT count when blocked behind a
    //    memory gate on its own queue.
    {
        const WaitPredicate w_q1[] = {{0x8000, 1, 7, true}};
        const WaitPredicate w_q2[] = {{0x8100, 1, 7, true}};
        const PendingGate unblocked_action[] = {
            {q1, 1, false, w_q1},
            {q2, 1, true, {}},  // unblocked action gate on Q2
        };
        CHECK(pause_reason(false, {}, unblocked_action, 0, test_met) == PauseReason::active_work);

        const PendingGate blocked_action[] = {
            {q1, 1, false, w_q1},
            {q2, 1, false, w_q2},  // memory gate blocking Q2
            {q2, 2, true, {}},     // action gate behind Q2's memory gate
        };
        CHECK(pause_reason(false, {}, blocked_action, 0, test_met) == PauseReason::none);
    }

    // 9. Bounded pause caps: active_work, matching_producer, releaser_busy, combined alternating reasons, and
    //    demoted-gate cap scaling.
    {
        CHECK(default_active_pause_cap_us(200) == 60'000'000ull);
        CHECK(default_max_pause_cap_us(200) == 300'000'000ull);
        CHECK(default_active_pause_cap_us(30) == 9'000'000ull);
        CHECK(default_max_pause_cap_us(30) == 45'000'000ull);
        CHECK(default_active_pause_cap_us(1) == 300'000ull);
        CHECK(default_max_pause_cap_us(1) == 1'500'000ull);
        CHECK(default_active_pause_cap_us(30000) == 600'000'000ull);
        CHECK(default_max_pause_cap_us(30000) == 600'000'000ull);
        CHECK(default_active_pause_cap_us(UINT_MAX) == 600'000'000ull);
        CHECK(default_max_pause_cap_us(UINT_MAX) == 600'000'000ull);

        // Demoted (cut_short) gates scale caps with their short 4 ms budget instead of inheriting 60 s / 300 s.
        const PauseCaps dem = default_pause_caps(4, true);
        CHECK(dem.active_us == 4'000ull && dem.max_us == 16'000ull);
        GateTimeoutState dem_active;
        dem_active.reset(4'000, dem.active_us, dem.max_us);
        CHECK(!dem_active.advance(4'000, PauseReason::active_work));  // absorbs 4 ms cap
        CHECK(dem_active.active_paused_us == 4'000 && dem_active.elapsed_us == 0);
        CHECK(!dem_active.advance(3'999, PauseReason::active_work));
        CHECK(dem_active.advance(1, PauseReason::active_work));       // expires at 8 ms total, not 60 s!

        GateTimeoutState dem_prod;
        dem_prod.reset(4'000, dem.active_us, dem.max_us);
        CHECK(!dem_prod.advance(16'000, PauseReason::matching_producer));
        CHECK(dem_prod.total_paused_us == 16'000 && dem_prod.elapsed_us == 0);
        CHECK(dem_prod.advance(4'000, PauseReason::matching_producer));  // expires at 20 ms total

        // Normal gate active_work cap (5 s) vs larger max cap (20 s).
        GateTimeoutState t;
        t.reset(200'000, 5'000'000, 20'000'000);
        for (int i = 0; i < 5; ++i) CHECK(!t.advance(1'000'000, PauseReason::active_work));
        CHECK(t.active_paused_us == 5'000'000 && t.total_paused_us == 5'000'000 && t.elapsed_us == 0 && t.paused_ever);
        CHECK(!t.advance(199'000, PauseReason::active_work));
        CHECK(t.advance(1'000, PauseReason::active_work));
        CHECK(t.elapsed_us == 200'000);

        // Stuck matching_producer + alternating releaser_busy share total_paused_us <= max_pause_cap_us.
        GateTimeoutState stuck;
        stuck.reset(200'000, 2'000'000, 10'000'000);
        CHECK(!stuck.advance(2'000'000, PauseReason::active_work));        // total = 2 s
        CHECK(!stuck.advance(5'000'000, PauseReason::matching_producer));  // total = 7 s
        CHECK(!stuck.advance(3'100'000, PauseReason::releaser_busy));      // 3 s paused (hits 10 s cap), 100 ms elapsed
        CHECK(stuck.total_paused_us == 10'000'000 && stuck.elapsed_us == 100'000);
        CHECK(!stuck.advance(99'999, PauseReason::matching_producer));
        CHECK(stuck.advance(1, PauseReason::matching_producer));
        CHECK(stuck.elapsed_us == 200'000);

        // Partial step straddling active_pause_cap_us splits cleanly between active_paused_us and elapsed_us.
        GateTimeoutState t2;
        t2.reset(200'000, 500'000, 1'000'000);
        CHECK(!t2.advance(600'000, PauseReason::active_work));  // 500k paused, 100k elapsed
        CHECK(t2.active_paused_us == 500'000 && t2.total_paused_us == 500'000 && t2.elapsed_us == 100'000);
        CHECK(t2.advance(100'000, PauseReason::active_work));

        // active_cap > max_cap is clamped to max_cap by reset().
        GateTimeoutState clamped;
        clamped.reset(100'000, 9'000'000, 3'000'000);
        CHECK(clamped.active_pause_cap_us == 3'000'000 && clamped.max_pause_cap_us == 3'000'000);
    }

    // 10. Overflow-safe address / dt_us guards and zero-dt paused tick.
    {
        const WaitPredicate overflow_wait[] = {{UINT64_MAX - 2, 1, 7, true}};
        const PendingWrite overflow_write[] = {{UINT64_MAX - 2, 1, 8, false}};
        const PendingWriteGroup groups[] = {{q2, 0, overflow_write}};
        const PendingGate gates[] = {{q1, 1, false, overflow_wait}};
        // Write doesn't match due to address wrap guard, falls back to active_work for the unblocked group.
        CHECK(pause_reason(false, groups, gates, 0, test_met) == PauseReason::active_work);

        GateTimeoutState t;
        t.reset(1000, 0, 0);
        CHECK(t.advance(UINT64_MAX, PauseReason::none));
        CHECK(t.elapsed_us == UINT64_MAX);

        GateTimeoutState z;
        z.reset(0, 1000, 1000);
        CHECK(!z.advance(0, PauseReason::releaser_busy));
        CHECK(!z.paused_ever);
    }

    // 11. Mock Pump tick loop driving start_gate + pause_reason + GateTimeoutState end-to-end across 1 ms steps.
    {
        // (a) 200 ms gate held while releaser runs a 110 s pipeline build, then producer lands at 110.5 s: met!
        const WaitPredicate waits[] = {{0x9000, 7, 7, true}};
        const PendingWrite prod_write[] = {{0x9000, 7, 8, false}};
        const PendingWriteGroup prod_groups[] = {{q2, 0, prod_write}};
        const PendingGate gates[] = {{q1, 1, false, waits}};

        GateTimeoutState gate;
        const PauseCaps full_caps = default_pause_caps(200, false);
        gate.reset(200'000, full_caps.active_us, full_caps.max_us);
        for (int ms = 0; ms < 110'000; ++ms) {
            const PauseReason r = pause_reason(true, prod_groups, gates, 0, test_met);
            CHECK(r == PauseReason::releaser_busy && !gate.advance(1'000, r));
        }
        for (int ms = 0; ms < 500; ++ms) {
            const PauseReason r = pause_reason(false, prod_groups, gates, 0, test_met);
            CHECK(r == PauseReason::matching_producer && !gate.advance(1'000, r));
        }
        CHECK(gate.elapsed_us == 0 && gate.total_paused_us == 110'500'000ull);

        // (b) Same gate with a matching producer stuck behind an untracked title fence wait: expires at 300.2 s!
        GateTimeoutState stuck_gate;
        stuck_gate.reset(200'000, full_caps.active_us, full_caps.max_us);
        int expired_at_ms = -1;
        for (int ms = 1; ms <= 305'000; ++ms) {
            const PauseReason r = pause_reason(false, prod_groups, gates, 0, test_met);
            if (stuck_gate.advance(1'000, r)) {
                expired_at_ms = ms;
                break;
            }
        }
        CHECK(expired_at_ms == 300'200);

        // (c) Demoted gate (4 ms budget) with continuous unrelated GPU work on q2: expires at 8 ms!
        const PendingWrite unrelated[] = {{0xA000, 1, 8, false}};
        const PendingWriteGroup unrel_groups[] = {{q2, 0, unrelated}};
        GateTimeoutState dem_gate;
        const PauseCaps dem_caps = default_pause_caps(4, true);
        dem_gate.reset(4'000, dem_caps.active_us, dem_caps.max_us);
        int dem_expired_ms = -1;
        for (int ms = 1; ms <= 100; ++ms) {
            const PauseReason r = pause_reason(false, unrel_groups, gates, 0, test_met);
            CHECK(r == PauseReason::active_work);
            if (dem_gate.advance(1'000, r)) {
                dem_expired_ms = ms;
                break;
            }
        }
        CHECK(dem_expired_ms == 8);
    }

    std::printf("list_value_wait_budget_test: all checks passed\n");
    return 0;
}
