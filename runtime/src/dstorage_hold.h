// Counters of the DirectStorage queues' rule "reads into physical pages wait for the title's Submit"
// (shims/dstorage.cpp, docs/directstorage.md). Process-wide, so the log line and the test read the same numbers.
#pragma once

#include <atomic>
#include <cstdint>

namespace leonida::dstorage {

struct HoldStats {
    std::atomic<uint64_t> self_submits{0};    // a queue half full submitted itself
    std::atomic<uint64_t> held_reads{0};      // reads into physical pages such a submit left for the title's Submit
    std::atomic<uint64_t> full_releases{0};   // a full queue submitted its held entries (the title waits in an enqueue)
    std::atomic<uint64_t> timed_releases{0};  // LEONIDA_DSTORAGE_HOLD_MS ran out before the title's Submit
    std::atomic<uint64_t> page0_entries{0};   // destination page lists that named page 0 when their data was written
};

inline HoldStats& hold_stats() {
    static HoldStats stats;
    return stats;
}

}  // namespace leonida::dstorage
