#pragma once

#include <atomic>
#include <mutex>

namespace leonida {

// A placed image can be named concurrently by command replay and descriptor creation.
// Vulkan permits one successful bind for the lifetime of an ordinary image.
class DeferredImageBinding {
public:
    using Bind = bool (*)(void*);
    DeferredImageBinding(void* owner, Bind bind) : owner_(owner), bind_(bind) {}
    bool bound() const { return bound_.load(std::memory_order_acquire); }
    bool ensure() {
        if (bound()) return true;
        std::lock_guard lock(mutex_);
        if (bound()) return true;
        if (!bind_(owner_)) return false;
        bound_.store(true, std::memory_order_release);
        return true;
    }

private:
    void* owner_;
    Bind bind_;
    std::mutex mutex_;
    std::atomic<bool> bound_{false};
};

}  // namespace leonida
