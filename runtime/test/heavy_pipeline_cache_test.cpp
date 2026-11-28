// src/heavy_pipeline_cache.h: the LEONIDA_HEAVY_PIPELINE_CACHE_MS threshold and when the heavy cache is written.
// No Vulkan, no GPU, no files.
#include "../src/heavy_pipeline_cache.h"

#include <cstdio>
#include <initializer_list>

#define CHECK(condition) do {                                        \
    if (!(condition)) {                                              \
        std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); \
        return 1;                                                    \
    }                                                                \
} while (0)

namespace heavy = leonida::heavy_pipeline_cache;

static int test_threshold() {
    CHECK(heavy::parse_threshold(nullptr).ms == heavy::default_threshold_ms);
    CHECK(heavy::parse_threshold("").ms == heavy::default_threshold_ms);
    CHECK(heavy::parse_threshold("0").ms == 0 && heavy::parse_threshold("0").valid);  // off
    CHECK(heavy::parse_threshold("2000").ms == 2000);
    CHECK(heavy::parse_threshold("0750").ms == 750);
    // Not a number, a sign, a unit or an absurd value: the default, reported as invalid.
    for (const char* bad : {"-1", "+5", "1s", "1.5", "abc", "3600001", "99999999999999999999"}) {
        const auto t = heavy::parse_threshold(bad);
        CHECK(!t.valid && t.ms == heavy::default_threshold_ms);
    }
    CHECK(heavy::parse_threshold("3600000").valid);
    return 0;
}

static int test_schedule() {
    using S = heavy::WriteSchedule;
    S s;
    CHECK(!s.dirty() && !s.due(1'000'000));
    // One heavy compile: written once it has settled.
    s.note_heavy(1000);
    CHECK(s.dirty() && !s.due(1000 + S::settle_ms - 1) && s.due(1000 + S::settle_ms));
    s.written(s.generation(), 1000 + S::settle_ms);
    CHECK(!s.dirty());
    // A burst, one compile a second for a minute: the write does not wait for the burst to end.
    S burst;
    for (std::uint64_t t = 10'000; t <= 70'000; t += 1000) {
        burst.note_heavy(t);
        if (burst.due(t)) {
            CHECK(t - 10'000 >= S::max_delay_ms);  // first write no later than max_delay after the first compile
            burst.written(burst.generation(), t);
            break;
        }
    }
    CHECK(!burst.dirty());
    // A compile that finishes while a write is running is not lost: the cache stays dirty for the next write.
    S racing;
    racing.note_heavy(5000);
    const std::uint64_t start = 5000 + S::settle_ms;
    const std::uint64_t taken = racing.generation();
    racing.note_heavy(start + 10);  // during the write
    racing.written(taken, start);
    CHECK(racing.dirty());
    CHECK(racing.due(start + 10 + S::settle_ms));
    // The same, inside one tick of a coarse clock: the compile noted at the write's own start time is still kept.
    S tick;
    tick.note_heavy(9000);
    const std::uint64_t at = 9000 + S::settle_ms;
    const std::uint64_t before = tick.generation();
    tick.note_heavy(at);
    tick.written(before, at);
    CHECK(tick.dirty());
    return 0;
}

static int test_failed_write() {
    using S = heavy::WriteSchedule;
    S s;
    s.note_heavy(1000);
    std::uint64_t now = 1000 + S::settle_ms;
    CHECK(s.due(now));
    // A failed write keeps the cache dirty and waits out the backoff before the next try.
    s.failed(now);
    CHECK(s.dirty() && s.failures() == 1);
    CHECK(!s.due(now) && !s.due(now + S::retry_ms - 1) && s.due(now + S::retry_ms));
    // Failures in a row double the delay up to the cap, so a write that keeps failing does not spin.
    std::uint64_t expect = S::retry_ms;
    for (int i = 0; i < 12; ++i) {
        now = s.due_ms();
        s.failed(now);
        expect = expect * 2 > S::max_retry_ms ? S::max_retry_ms : expect * 2;
        CHECK(s.due_ms() == now + expect);
    }
    CHECK(s.due_ms() - now == S::max_retry_ms);
    // A heavy compile during the backoff does not bring the retry forward.
    s.note_heavy(now + 1);
    CHECK(!s.due(now + S::settle_ms + 1) && s.due(now + S::max_retry_ms));
    // A successful write clears the dirty flag and the backoff.
    now = s.due_ms();
    s.written(s.generation(), now);
    CHECK(!s.dirty() && s.failures() == 0);
    s.note_heavy(now + 10);
    CHECK(s.due(now + 10 + S::settle_ms));
    return 0;
}

int main() {
    if (const int r = test_threshold()) return r;
    if (const int r = test_schedule()) return r;
    if (const int r = test_failed_write()) return r;
    std::printf("heavy_pipeline_cache: threshold parsing, the write schedule and retries after a failed write\n");
    return 0;
}
