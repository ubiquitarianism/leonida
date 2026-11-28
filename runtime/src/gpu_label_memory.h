#pragma once
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <limits>

namespace leonida::label {
struct Region {
    std::uintptr_t begin, end, allocation_base;
    bool heap;
    std::uint64_t generation = 1;
};
struct Target {
    std::weak_ptr<const Region> owner;
    std::uintptr_t address = 0;
    unsigned bytes = 0;
    std::uint64_t generation = 0;
};
struct Write {
    Target target;
    std::uint64_t value{};
    unsigned clock{}; // 0: supplied value; 1: GPU timestamp; 2: system timestamp
};
class Memory {
    std::mutex mutex_;
    std::map<std::uintptr_t, std::shared_ptr<Region>> regions_;

  public:
    void track(std::uintptr_t base, std::uintptr_t bytes, std::uintptr_t allocation_base, bool heap) {
        // A base of 0 is the GPU address arena: its address space starts at 0 (runtime/shims/d3d12_vulkan_device.cpp).
        if (!bytes || !allocation_base || bytes > std::numeric_limits<std::uintptr_t>::max() - base) return;
        const auto end = base + bytes;
        std::lock_guard lock(mutex_);
        auto previous = regions_.upper_bound(base);
        if (!heap && previous != regions_.begin()) {
            --previous;
            const auto &r = *previous->second;
            if (!r.heap && r.allocation_base == allocation_base && base >= r.begin && end <= r.end) return;
        }
        regions_[base] = std::make_shared<Region>(Region{base, end, allocation_base, heap});
    }
    void release_heap(std::uintptr_t base) {
        std::lock_guard lock(mutex_);
        regions_.erase(base);
    }
    template <class Release>
    bool release_pages(std::uintptr_t base, std::uintptr_t bytes, bool entire, Release &&release) {
        std::lock_guard lock(mutex_);
        if (!release()) return false;
        for (auto i = regions_.begin(); i != regions_.end();) {
            const auto &r = *i->second;
            if (r.heap || (entire ? r.allocation_base != base
                                  : (!bytes ? r.allocation_base != base
                                            : (base >= r.end || (base < r.begin && r.begin - base >= bytes))))) {
                ++i;
                continue;
            }
            if (entire || i->second->generation == std::numeric_limits<std::uint64_t>::max())
                i = regions_.erase(i);
            else {
                ++i->second->generation;
                ++i;
            }
        }
        return true;
    }
    Target capture(std::uintptr_t address, unsigned bytes) {
        if ((bytes != 4 && bytes != 8) || (address & (bytes - 1))) return {};
        std::lock_guard lock(mutex_);
        auto it = regions_.upper_bound(address);
        while (it != regions_.begin()) {
            --it;
            const auto &r = *it->second;
            if (address < r.end && r.end - address >= bytes) return {it->second, address, bytes, r.generation};
        }
        return {};
    }
    template <class Store> bool publish(const Target &target, Store &&store) {
        std::lock_guard lock(mutex_);
        const auto region = target.owner.lock();
        if (!region || target.generation != region->generation || (target.bytes != 4 && target.bytes != 8) ||
            (target.address & (target.bytes - 1)) || target.address < region->begin || target.address >= region->end ||
            region->end - target.address < target.bytes)
            return false;
        const auto it = regions_.find(region->begin);
        if (it == regions_.end() || it->second != region) return false;
        return store(target.address, target.bytes, region->allocation_base);
    }
};
inline Memory &memory() {
    static Memory m;
    return m;
}
} // namespace leonida::label
