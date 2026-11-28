#pragma once
// What the runtime is building now, for the dev overlay (dev_overlay.h). Always on: two relaxed atomics a build, and a
// short copy under a lock for the name.
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <string>

namespace leonida::dev_status {

struct Builds {
    std::atomic<unsigned> active{0};
    std::atomic<std::uint64_t> done{0};

    std::string last() {
        std::lock_guard lock(mutex_);
        return name_;
    }
    void name(const char* text) {
        std::lock_guard lock(mutex_);
        const std::size_t n = std::min(std::strlen(text), sizeof name_ - 1);
        std::memcpy(name_, text, n);
        name_[n] = 0;
    }

private:
    std::mutex mutex_;
    char name_[96]{};
};
inline Builds& builds() {
    static Builds instance;
    return instance;
}

// One pipeline build, from construction to destruction.
struct Build {
    explicit Build(const char* name) {
        builds().name(name);
        builds().active.fetch_add(1, std::memory_order_relaxed);
    }
    ~Build() {
        builds().active.fetch_sub(1, std::memory_order_relaxed);
        builds().done.fetch_add(1, std::memory_order_relaxed);
    }
    Build(const Build&) = delete;
    Build& operator=(const Build&) = delete;
};

}  // namespace leonida::dev_status
