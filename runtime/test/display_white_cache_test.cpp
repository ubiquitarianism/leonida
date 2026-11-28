// DisplayWhiteCache (src/display_white_cache.h): XboxPresentation's Windows SDR white level, read on a worker thread.
// The first HDR10 present asks for the value, and the worker must read it then, not when its once-a-period wait runs
// out (djrolfes' finding in #1608: the wake-up was notified but the wait's predicate ignored it, so the first read
// came up to a second late). No window, no GPU: the read is a counter.
#include "../src/display_white_cache.h"

#include <chrono>
#include <cstdio>
#include <thread>

namespace {

bool check(bool condition, const char* description) {
    if (!condition) std::fprintf(stderr, "FAIL: %s\n", description);
    return condition;
}

// Polls `done` for up to `limit`.
template <class F>
bool within(std::chrono::milliseconds limit, F done) {
    const auto end = std::chrono::steady_clock::now() + limit;
    while (!done()) {
        if (std::chrono::steady_clock::now() >= end) return false;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return true;
}

}  // namespace

int main() {
    using namespace std::chrono_literals;
    bool ok = true;

    // A period far longer than the test: only the first get()'s wake-up can bring the first read.
    {
        std::atomic<float> windows{240.0f};
        leonida::DisplayWhiteCache cache([&] { return windows.load(); }, std::chrono::hours(1));
        std::this_thread::sleep_for(20ms);  // the worker is in its wait by now
        ok &= check(cache.reads() == 0, "nothing is read before the first HDR10 present asks");
        ok &= check(cache.get() == 0.0f, "the first get() has no value yet (the caller falls back to 203)");
        ok &= check(within(2s, [&] { return cache.reads() == 1; }), "the first get() wakes the worker to read now");
        ok &= check(within(2s, [&] { return cache.get() == 240.0f; }), "and the value it read is cached");
        windows = 300.0f;
        std::this_thread::sleep_for(50ms);
        ok &= check(cache.reads() == 1 && cache.get() == 240.0f, "later get()s do not read again before the period");
    }

    // get() before the worker reaches its wait: the request is not lost.
    for (int i = 0; i < 50; ++i) {
        leonida::DisplayWhiteCache cache([] { return 203.0f; }, std::chrono::hours(1));
        cache.get();
        if (!check(within(2s, [&] { return cache.reads() == 1; }), "a get() right after construction is read")) {
            ok = false;
            break;
        }
    }

    // Once wanted, the worker reads once a period, and a failed read keeps the last good value.
    {
        std::atomic<int> calls{0};
        leonida::DisplayWhiteCache cache([&] { return ++calls == 1 ? 180.0f : 0.0f; }, 20ms);
        cache.get();
        ok &= check(within(2s, [&] { return cache.reads() >= 4; }), "reads repeat about once a period");
        ok &= check(cache.get() == 180.0f, "a failed read keeps the last good value");
    }

    // Destruction stops a worker that waits out a long period at once.
    {
        const auto start = std::chrono::steady_clock::now();
        {
            leonida::DisplayWhiteCache cache([] { return 1.0f; }, std::chrono::hours(1));
            cache.get();
            within(2s, [&] { return cache.reads() == 1; });
        }
        ok &= check(std::chrono::steady_clock::now() - start < 2s, "destruction wakes and joins the worker");
    }
    if (ok) std::printf("display_white_cache_test: PASS\n");
    return ok ? 0 : 1;
}
