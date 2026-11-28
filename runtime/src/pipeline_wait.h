#pragma once
// When the title last submitted, for the compute warm-ups (runtime/shims/d3d12_vulkan_device.cpp,
// LEONIDA_COMPUTE_WARMUP_LIVE_MAX_BYTES, opt-in): while the title submits, a warm-up of a big module is left to the
// pipeline's first use, since a create holds the compile gate for its whole length and every submission waits behind it.
// No windows.h here: vulkan_backend.h includes this, and translation units without NOMINMAX include that.
#include <atomic>
#include <chrono>
#include <cstdint>

namespace leonida::pipeline_wait {

// Steady-clock nanoseconds (the clock of submit_phases::now_ns).
inline std::int64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

// The last submission (VulkanQueue::execute).
inline std::atomic<std::int64_t>& last_submission() {
    static std::atomic<std::int64_t> value{0};
    return value;
}
inline void note_submission() { last_submission().store(now_ns(), std::memory_order_relaxed); }
inline bool title_submitting(std::int64_t within_ns = 1000000000) {
    const std::int64_t last = last_submission().load(std::memory_order_relaxed);
    return last && now_ns() - last < within_ns;
}

// Big warm-ups left to their first use while the title submits.
inline std::atomic<std::uint64_t>& warmups_left() {
    static std::atomic<std::uint64_t> value{0};
    return value;
}

}  // namespace leonida::pipeline_wait
