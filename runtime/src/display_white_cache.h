// The Windows SDR white level for XboxPresentation (shims/xbox_present.cpp), read off the present path.
//
// Reading it costs a QueryDisplayConfig and two DisplayConfigGetDeviceInfo calls (measured in xbox_present_test, docs
// section 4.5): too much for the present path, which would hitch once a second and feed the pacer. So a worker thread
// reads it: at once when the first HDR10 present asks for it, then about once a period. PresentX reads the cached value.
// Kept apart from XboxPresentation so the wake-up can be tested without a window or a GPU (display_white_cache_test).
#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <utility>

namespace leonida {

class DisplayWhiteCache {
public:
    // `read` returns the white level in nits, or 0 when it cannot be read (a failed read keeps the last good value). It
    // runs on the worker thread only, and only after the first get().
    explicit DisplayWhiteCache(std::function<float()> read,
                               std::chrono::milliseconds period = std::chrono::milliseconds(1000))
        : read_(std::move(read)), period_(period), thread_([this] { run(); }) {}

    ~DisplayWhiteCache() {
        {
            std::lock_guard lock(mutex_);
            stop_ = true;
        }
        wake_.notify_all();
        thread_.join();
    }

    DisplayWhiteCache(const DisplayWhiteCache&) = delete;
    DisplayWhiteCache& operator=(const DisplayWhiteCache&) = delete;

    // The last good value (0: none yet). The first call wakes the worker to read now, instead of at the end of the
    // period it is waiting out; later reads come once a period.
    float get() {
        if (!wanted_.exchange(true)) {
            std::lock_guard lock(mutex_);
            read_now_ = true;  // what the worker's wait checks: a notify alone ends no wait whose predicate is false
            wake_.notify_all();
        }
        return value_.load();
    }

    // Number of reads the worker has made (tests).
    unsigned reads() const { return reads_.load(); }

private:
    void run() {
        std::unique_lock lock(mutex_);
        while (!stop_) {
            if (wanted_.load()) {
                read_now_ = false;
                lock.unlock();
                const float nits = read_();
                if (nits > 0.0f) value_ = nits;
                ++reads_;
                lock.lock();
            }
            // read_now_ also covers a get() that came before this wait began: its notify found no waiter.
            wake_.wait_for(lock, period_, [this] { return stop_ || read_now_; });
        }
    }

    std::function<float()> read_;
    std::chrono::milliseconds period_;
    std::atomic<float> value_{0.0f};
    std::atomic<bool> wanted_{false};
    std::atomic<unsigned> reads_{0};
    std::mutex mutex_;
    std::condition_variable wake_;
    bool stop_ = false;      // under mutex_
    bool read_now_ = false;  // under mutex_: the first get() asks for a read before the period ends
    std::thread thread_;     // last: it starts once every other member is made
};

}  // namespace leonida
