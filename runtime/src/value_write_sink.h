#pragma once
// Where a console value write (list slots 71 and 72) lands when its target is not memory a plain host store reaches:
// - the GPU address arena: the title names a default-heap buffer by its arena offset, which is not a host address, so
//   a host store at that number would write whatever host memory is there. The D3D12 device that owns the arena
//   answers for its range (set_arena) and writes through its own mapping.
// - a page of the title's memory that xmem maps PAGE_NOACCESS (a GPU-only window): the title's view refuses the CPU,
//   and xmem writes through its own mapping of the physical page (set_title_pages, rt::xmem::write_title_bytes).
// Two writers use it: the list values' pump (shims/d3d12_xs_list_values.h), which lands the writes of the lists it
// tracks, and the GPU-label path (vulkan_gpu_labels.h, LEONIDA_GPU_LABELS=1), for lists it does not track.
// Portable: no Windows or Vulkan headers, so both sides and the tests without a GPU can include it.

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>

namespace leonida::value_sink {

enum class Landing {
    elsewhere,  // not the arena's: the caller stores it as before
    landed,     // the arena took it
    refused,    // the arena's, but its bytes cannot be written there (reserved, sparse, past the end): dropped
};

// `device` is the context set_arena was given; `address` the title's address of the value, `size` 4 or 8.
using ArenaWrite = Landing (*)(void* device, std::uint64_t address, const void* bytes, unsigned size);
// True when the bytes were written; false when the range is not one of the shim's pages.
using TitlePagesWrite = bool (*)(const void* address, const void* bytes, std::size_t size);

struct State {
    // Held across an arena write, so a device that goes away (clear_arena) waits for one in progress.
    std::mutex mutex;
    ArenaWrite arena = nullptr;
    void* device = nullptr;
    std::atomic<TitlePagesWrite> title_pages{nullptr};
};

inline State& state() {
    static State s;
    return s;
}

// The device that owns the arena. The last device to set it answers; clear_arena(device) ends its turn.
inline void set_arena(ArenaWrite write, void* device) {
    State& s = state();
    const std::lock_guard lock(s.mutex);
    s.arena = write;
    s.device = device;
}

inline void clear_arena(void* device) {
    State& s = state();
    const std::lock_guard lock(s.mutex);
    if (s.device != device) return;
    s.arena = nullptr;
    s.device = nullptr;
}

// Asked before any host store of a value write: an arena offset must never reach one.
inline Landing arena(std::uint64_t address, const void* bytes, unsigned size) {
    State& s = state();
    const std::lock_guard lock(s.mutex);
    if (!s.arena) return Landing::elsewhere;
    return s.arena(s.device, address, bytes, size);
}

// xmem sets it when it is linked into the process (shims/xmem.cpp).
inline void set_title_pages(TitlePagesWrite write) { state().title_pages.store(write, std::memory_order_release); }

// The fallback after a host store into the title's memory failed: the page may be one of xmem's that the title's view
// keeps PAGE_NOACCESS. False without xmem, or when the page is not xmem's.
inline bool title_pages(const void* address, const void* bytes, std::size_t size) {
    const TitlePagesWrite write = state().title_pages.load(std::memory_order_acquire);
    return write && write(address, bytes, size);
}

}  // namespace leonida::value_sink
