#pragma once
// Where a submission's CPU time goes, always on and cheap: each phase of VulkanQueue::execute_now adds its time here
// (two steady_clock reads and two relaxed atomic adds), and every LEONIDA_SUBMIT_PHASES_SECONDS (5 by default, 0 turns
// the line off) one line says, since the last line, how many submissions there were, how many per second, the
// milliseconds per submission of each phase with the largest single one, and the share of the wall time the
// submitting threads spent in them. The time is the process's age (GetProcessTimes), so it lines up with the boot.
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace leonida::submit_phases {

enum Phase : unsigned {
    wait_room,  // VulkanQueue::wait_for_room: the queue at its cap waits for the GPU
    heaps,      // DeviceState::refresh_console_heaps
    shadows,    // slot-133 shadow buffers and title windows (ShadowPages::refresh)
    geometry,   // GeometryViews::refresh
    textures,   // Device::refresh_title_textures
    pipelines,  // no longer timed (the pre-pass that waited for pipelines is gone); kept for the column order
    replay,     // the rest of execute_now: preparing, recording and submitting the lists
    kCount
};

inline const char* name(unsigned p) {
    static const char* const names[kCount] = {"wait", "heaps", "shadows", "geometry", "textures", "pipes",
                                                    "replay"};
    return p < kCount ? names[p] : "?";
}

struct Totals {
    std::atomic<std::uint64_t> ns[kCount]{};
    std::atomic<std::uint64_t> max_ns[kCount]{};
    std::atomic<std::uint64_t> submissions{0}, all_submissions{0};
    std::atomic<std::int64_t> last_log{0};
    std::atomic<std::uint64_t> presents{0};  // frames the presenter showed (VulkanPresenter), since the last line
    std::atomic<std::uint64_t> present_serial{0};  // the same, never reset (LEONIDA_REFRESH_PER_FRAME)
    // Threads whose innermost Scope is this phase now, and when the first of them entered it (steady_clock). Counted
    // per phase, since queues submit from several threads: one value would go stale when their scopes overlap.
    std::atomic<unsigned> active[kCount]{};
    std::atomic<std::int64_t> active_since[kCount]{};
    std::atomic<std::uint32_t> thread{0};  // the thread of the last submission (the thread sampler reports it apart)
};
inline Totals& totals() {
    static Totals t;
    return t;
}

inline std::int64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

// Seconds since the process started.
inline double process_age() {
#ifdef _WIN32
    FILETIME created{}, exited{}, kernel{}, user{}, now{};
    if (!GetProcessTimes(GetCurrentProcess(), &created, &exited, &kernel, &user)) return 0;
    GetSystemTimeAsFileTime(&now);
    const auto ticks = [](const FILETIME& f) { return (std::uint64_t(f.dwHighDateTime) << 32) | f.dwLowDateTime; };
    return double(ticks(now) - ticks(created)) * 1e-7;
#else
    return 0;
#endif
}

inline std::int64_t period_ns() {
    static const std::int64_t value = [] {
        const char* text = std::getenv("LEONIDA_SUBMIT_PHASES_SECONDS");
        const unsigned long long s = text && *text ? std::strtoull(text, nullptr, 10) : 5;
        return std::int64_t(s) * 1000000000;
    }();
    return value;
}

inline void add(Phase p, std::uint64_t ns) {
    Totals& t = totals();
    t.ns[p].fetch_add(ns, std::memory_order_relaxed);
    std::uint64_t m = t.max_ns[p].load(std::memory_order_relaxed);
    while (ns > m && !t.max_ns[p].compare_exchange_weak(m, ns, std::memory_order_relaxed)) {
    }
}

// Times its own lifetime into a phase.
struct Scope {
    Phase phase;
    std::int64_t start = now_ns();
    int outer;  // the phase of this thread's Scope around this one, -1 for none
    explicit Scope(Phase p) : phase(p), outer(innermost()) {
        if (outer >= 0) leave(outer);
        enter(int(p), start);
        innermost() = int(p);
    }
    ~Scope() {
        const std::int64_t end = now_ns();
        add(phase, std::uint64_t(end - start));
        leave(int(phase));
        if (outer >= 0) enter(outer, end);
        innermost() = outer;
    }
    static int& innermost() {
        thread_local int phase_of_this_thread = -1;
        return phase_of_this_thread;
    }
    static void enter(int p, std::int64_t at) {
        Totals& t = totals();
        if (t.active[p].fetch_add(1, std::memory_order_relaxed) == 0) t.active_since[p].store(at, std::memory_order_relaxed);
    }
    static void leave(int p) { totals().active[p].fetch_sub(1, std::memory_order_relaxed); }
    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;
};

// The phase that some thread has been in longest, -1 when none is in one.
struct Running {
    int phase = -1;
    std::int64_t since = 0;
};
inline Running running() {
    Totals& t = totals();
    Running r;
    for (unsigned p = 0; p < kCount; ++p) {
        if (!t.active[p].load(std::memory_order_relaxed)) continue;
        const std::int64_t since = t.active_since[p].load(std::memory_order_relaxed);
        if (r.phase < 0 || since < r.since) r = {int(p), since};
    }
    return r;
}

// After each submission: counts it, and prints the line when the period has passed. One thread wins the line.
inline void submitted() {
    Totals& t = totals();
    t.submissions.fetch_add(1, std::memory_order_relaxed);
    t.all_submissions.fetch_add(1, std::memory_order_relaxed);
#ifdef _WIN32
    t.thread.store(GetCurrentThreadId(), std::memory_order_relaxed);
#endif
    const std::int64_t period = period_ns();
    if (!period) return;
    const std::int64_t now = now_ns();
    std::int64_t last = t.last_log.load(std::memory_order_relaxed);
    if (!last) {
        t.last_log.compare_exchange_strong(last, now, std::memory_order_relaxed);
        return;
    }
    if (now - last < period || !t.last_log.compare_exchange_strong(last, now, std::memory_order_relaxed)) return;
    const double seconds = double(now - last) * 1e-9;
    const std::uint64_t n = t.submissions.exchange(0, std::memory_order_relaxed);
    const std::uint64_t frames = t.presents.exchange(0, std::memory_order_relaxed);
    double ms[kCount], max_ms[kCount], all = 0;
    for (unsigned p = 0; p < kCount; ++p) {
        const double total = double(t.ns[p].exchange(0, std::memory_order_relaxed)) * 1e-6;
        max_ms[p] = double(t.max_ns[p].exchange(0, std::memory_order_relaxed)) * 1e-6;
        ms[p] = n ? total / double(n) : 0;
        all += total;
    }
    std::fprintf(stderr,
                 "[rt] submit phases t=%.0fs: %llu submissions in %.1f s (%.2f/s, %llu in all), %.2f frames/s; ms per "
                 "submission (max): %s %.2f (%.0f) %s %.2f (%.0f) %s %.2f (%.0f) %s %.2f (%.0f) %s %.2f (%.0f) %s %.2f "
                 "(%.0f) %s %.2f (%.0f); "
                 "%.0f%% of the wall time\n",
                 process_age(), static_cast<unsigned long long>(n), seconds, double(n) / seconds,
                 static_cast<unsigned long long>(t.all_submissions.load(std::memory_order_relaxed)),
                 double(frames) / seconds, name(0), ms[0],
                 max_ms[0], name(1), ms[1], max_ms[1], name(2), ms[2], max_ms[2], name(3), ms[3], max_ms[3], name(4),
                 ms[4], max_ms[4], name(5), ms[5], max_ms[5], name(6), ms[6], max_ms[6], 100.0 * all * 1e-3 / seconds);
}

}  // namespace leonida::submit_phases
