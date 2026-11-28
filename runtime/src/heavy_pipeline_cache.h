// A pipeline cache that survives a crash, for the pipelines the offline corpus does not have (RT-HEAVY-PIPELINE-CACHE).
//
// The offline seed (pipeline_cache.bin, several GiB) covers what the `pipelines` build stage creates. A pipeline it
// misses is compiled on the spot, and some of those take the driver seconds each. pipeline_cache.session.bin cannot
// keep them across a crash: it is written at device teardown only, and a cache seeded from the corpus returns the
// whole corpus, over its write limit, so it is not written at all.
//
// So a pipeline is first created against the corpus cache with VK_PIPELINE_CREATE_FAIL_ON_PIPELINE_COMPILE_REQUIRED.
// When the driver would have to compile, it is created again against a cache of its own, seeded from
// pipeline_cache.heavy.bin. That cache holds only the corpus's misses, so it stays small. After a compile that took
// at least LEONIDA_HEAVY_PIPELINE_CACHE_MS (default 500 ms), it is written back to disk on a thread of its own: a
// staging file renamed over the old one, so a crash at any point leaves the last complete file.
//
// This header holds the pure decisions (the threshold and when to write). Vulkan and the file live in the device.
#pragma once

#include <cstdint>
#include <cstdlib>

namespace leonida::heavy_pipeline_cache {

inline constexpr std::uint32_t default_threshold_ms = 500;

struct Threshold {
    std::uint32_t ms = default_threshold_ms;  // 0: off
    bool valid = true;                        // false: the text was not a number; ms stays the default
};

// LEONIDA_HEAVY_PIPELINE_CACHE_MS: a compile at least this long schedules a write; 0 turns the heavy cache off.
inline Threshold parse_threshold(const char* text) {
    if (!text || !*text) return {};
    std::uint64_t value = 0;
    for (const char* p = text; *p; ++p) {
        if (*p < '0' || *p > '9') return {default_threshold_ms, false};
        value = value * 10 + std::uint64_t(*p - '0');
        if (value > 3'600'000) return {default_threshold_ms, false};  // an hour: past it the text is a mistake
    }
    return {std::uint32_t(value), true};
}

// When the writer thread writes. A heavy compile marks the cache dirty; the write waits `settle_ms` after the
// newest one (heavy compiles come in bursts at a load, and each write copies the whole heavy cache) but never more
// than `max_delay_ms` after the first unwritten one, so a long burst is still saved as it goes. A write that fails
// keeps the cache dirty and is tried again `retry_ms` later, the delay doubling with each failure in a row up to
// `max_retry_ms`, so a disk that stays full or a driver that keeps failing does not make the writer spin.
class WriteSchedule {
public:
    static constexpr std::uint64_t settle_ms = 3000;
    static constexpr std::uint64_t max_delay_ms = 20000;
    static constexpr std::uint64_t retry_ms = 5000;
    static constexpr std::uint64_t max_retry_ms = 120000;

    void note_heavy(std::uint64_t now_ms) {
        if (!dirty_) first_ = now_ms;
        dirty_ = true;
        last_ = now_ms;
        ++generation_;
    }
    bool dirty() const { return dirty_; }
    // The time to write at; only meaningful while dirty.
    std::uint64_t due_ms() const {
        const std::uint64_t settled = last_ + settle_ms, latest = first_ + max_delay_ms;
        const std::uint64_t at = settled < latest ? settled : latest;
        return at > retry_at_ ? at : retry_at_;
    }
    bool due(std::uint64_t now_ms) const { return dirty_ && now_ms >= due_ms(); }
    // Taken when a write starts; the write covers every heavy compile noted up to then.
    std::uint64_t generation() const { return generation_; }
    // A write that started at `generation` and `started_ms` is done. A heavy compile noted after it started, even in
    // the same millisecond tick, keeps the cache dirty for the next write, due as if it were first noted then.
    void written(std::uint64_t generation, std::uint64_t started_ms) {
        failures_ = 0;
        retry_at_ = 0;
        if (generation_ == generation) dirty_ = false;
        else first_ = started_ms;
    }
    // A write that started at `started_ms` failed: the cache stays dirty and the next try waits out the backoff.
    void failed(std::uint64_t started_ms) {
        std::uint64_t delay = retry_ms;
        for (std::uint32_t i = 0; i < failures_ && delay < max_retry_ms; ++i) delay *= 2;
        if (delay > max_retry_ms) delay = max_retry_ms;
        if (failures_ < 64) ++failures_;
        retry_at_ = started_ms + delay;
    }
    std::uint32_t failures() const { return failures_; }

private:
    bool dirty_ = false;
    std::uint64_t first_ = 0, last_ = 0, generation_ = 0;
    std::uint32_t failures_ = 0;   // failed writes in a row
    std::uint64_t retry_at_ = 0;   // no write before this time after a failure
};

}  // namespace leonida::heavy_pipeline_cache
