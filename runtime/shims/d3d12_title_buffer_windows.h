#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <utility>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <iterator>
#include <map>
#include <vector>
#include <vulkan/vulkan.h>

namespace rt::d3d12 {

// Storage for shader-addressed mirrors when the main arena has no usable offset.
// Sub-ranges freed via free() are coalesced for reuse; one empty standard 4 MiB block is kept as a spare and any
// other empty block releases its Vulkan buffer and memory.
class TitleBufferWindows {
    static constexpr VkDeviceSize kBlockSize = 4ull << 20;
    struct Block {
        VkBuffer buffer{};
        VkDeviceMemory memory{};
        VkDeviceAddress address{};
        VkDeviceSize capacity{}, used{}, reserved_bytes{};
        std::uint8_t* mapped{};
        std::size_t live{};
        std::map<VkDeviceSize, VkDeviceSize> free_ranges;  // offset -> size, coalesced
    };
    std::vector<Block> blocks_;
    VkDeviceSize reserved_{};
    // A new block's allocation failed at this time: none is tried again for kRetryAfter. After the story load the
    // driver refused every 4 MiB block (VK_ERROR_OUT_OF_HOST_MEMORY), and each V# that wanted a copy tried again,
    // through DeviceState::allocate's sleeping retries under the device's memory lock: one submission spent 28-35 s
    // there (speed-8, speed-9). A view refused meanwhile goes without a copy, as one past the cap does.
    static constexpr auto kRetryAfter = std::chrono::seconds(2);
    std::chrono::steady_clock::time_point failed_at_{};
    bool failed_ = false;
    bool local_refused_ = false;  // device-local blocks refused once: host memory from then on
    // SystemMemory: where a device_local allocation goes instead (set_system_memory; GeometryViews from
    // LEONIDA_GEOMETRY_HOST).
    int system_memory_ = 0;
    bool system_memory_logged_ = false;
    std::uint64_t quick_refusals_ = 0;
    // The blocks' (device address, capacity), readable without the owner's lock: published in order, a slot never
    // reused before release(). A destroyed block's slot keeps its other fields and only its address goes to 0 (the
    // readers skip it), so a reader never sees a slot half rewritten. Blocks are destroyed only once their last
    // range was freed, which GeometryViews does after the GPU work that used them completed.
    static constexpr std::size_t kPublished = 4096;
    struct Published {
        std::atomic<VkDeviceAddress> address{0};
        VkDeviceSize capacity{};
        std::uint8_t* mapped{};
        VkBuffer buffer{};
    };
    std::array<Published, kPublished> published_{};
    std::atomic<std::size_t> published_count_{0};
    std::atomic<std::uint64_t> layout_generation_{0};  // bumped by every block made or destroyed

    template<class Device>
    void destroy_block(Device& d, std::vector<Block>::iterator it) {
        // Unpublished first, so no lock-free reader picks it up again; then the owner is told, so tables built from
        // for_each_block (the console heap pass's backed table) cut it before the next submission.
        for (std::size_t i = 0, n = published_count_.load(std::memory_order_relaxed); i < n; ++i)
            if (published_[i].buffer == it->buffer && published_[i].address.load(std::memory_order_relaxed)) {
                published_[i].address.store(0, std::memory_order_release);
                break;
            }
        layout_generation_.fetch_add(1, std::memory_order_release);
        if constexpr (requires { d.title_block_gone(VkDeviceAddress{}, VkDeviceSize{}); })
            d.title_block_gone(it->address, it->capacity);
        d.context->vk.vkUnmapMemory(d.info.device, it->memory);
        d.context->vk.vkDestroyBuffer(d.info.device, it->buffer, nullptr);
        d.context->vk.vkFreeMemory(d.info.device, it->memory, nullptr);
        reserved_ -= it->reserved_bytes;
        blocks_.erase(it);
    }

    // The memory types a system-memory block may take: host-visible ones outside device-local memory (the GPU reads
    // them over the bus), the write-back (cached) ones only when `cached`, else the write-combined ones only. None of
    // them: every type `bits` allows (an integrated GPU, whose memory is all device-local).
    template<class Device>
    static std::uint32_t system_memory_types(const Device& d, std::uint32_t bits, bool cached) {
        if constexpr (requires { d.memory.memoryTypes[0].propertyFlags; d.memory.memoryTypeCount; }) {
            std::uint32_t host = 0, wanted = 0;
            for (std::uint32_t i = 0; i < d.memory.memoryTypeCount && i < 32; ++i) {
                if (!(bits & (1u << i))) continue;
                const VkMemoryPropertyFlags f = d.memory.memoryTypes[i].propertyFlags;
                if ((f & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) || !(f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) ||
                    !(f & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))
                    continue;
                host |= 1u << i;
                if (bool(f & VK_MEMORY_PROPERTY_HOST_CACHED_BIT) == cached) wanted |= 1u << i;
            }
            return wanted ? wanted : host ? host : bits;
        } else {
            (void)d;
            (void)cached;
            return bits;
        }
    }

public:
    // Where blocks asked for as device_local go: kDeviceLocal (Resizable BAR while the driver gives it, the default),
    // kSystemCached (host memory outside VRAM, write-back where the device offers it: what VIctory gives its copies
    // of title memory) or kSystemUncached (host memory outside VRAM, write-combined). The GPU reads the system-memory
    // kinds over the bus. Set before the first allocation.
    static constexpr int kDeviceLocal = 0, kSystemCached = 1, kSystemUncached = 2;
    void set_system_memory(int kind) { system_memory_ = kind; }
    int system_memory() const { return system_memory_; }
    VkDeviceSize reserved() const { return reserved_; }
    // f(device address, capacity) for each live block; from any thread.
    template <class F>
    void for_each_block(F&& f) const {
        const std::size_t n = published_count_.load(std::memory_order_acquire);
        for (std::size_t i = 0; i < n; ++i)
            if (const VkDeviceAddress address = published_[i].address.load(std::memory_order_acquire))
                f(address, published_[i].capacity);
    }
    // Changes whenever a block is made or destroyed; from any thread.
    std::uint64_t layout_generation() const { return layout_generation_.load(std::memory_order_acquire); }
    // The block whose mapping holds [p, p + bytes): its buffer and the offset of p in it. From any thread.
    bool find_mapped(const std::uint8_t* p, std::size_t bytes, VkBuffer& buffer, VkDeviceSize& offset) const {
        const std::size_t n = published_count_.load(std::memory_order_acquire);
        for (std::size_t i = 0; i < n; ++i) {
            const Published& b = published_[i];
            if (!b.address.load(std::memory_order_acquire)) continue;
            if (p >= b.mapped && p - b.mapped < std::ptrdiff_t(b.capacity) &&
                bytes <= std::size_t(b.capacity) - std::size_t(p - b.mapped)) {
                buffer = b.buffer;
                offset = VkDeviceSize(p - b.mapped);
                return true;
            }
        }
        return false;
    }
    std::size_t block_count() const { return blocks_.size(); }
    std::size_t empty_blocks() const {
        std::size_t count = 0;
        for (const auto& b : blocks_)
            if (b.live == 0) ++count;
        return count;
    }

    template<class Device>
    bool allocate(Device& d, VkDeviceSize bytes, VkDeviceSize& offset, std::uint8_t*& mapped,
                  bool device_local = false, VkDeviceSize cap = UINT64_MAX, bool* at_cap = nullptr) {
        if (!bytes) return false;
        const VkDeviceSize alignment = device_local ? 16 : 65536;
        auto take = [&](Block& b) {
            for (auto it = b.free_ranges.begin(); it != b.free_ranges.end(); ++it) {
                const VkDeviceSize range_start = it->first;
                const VkDeviceSize range_end = range_start + it->second;
                const VkDeviceSize start = (range_start + alignment - 1) & ~(alignment - 1);
                if (start < range_start || start > range_end || bytes > range_end - start) continue;
                b.free_ranges.erase(it);
                if (start > range_start) b.free_ranges.emplace(range_start, start - range_start);
                const VkDeviceSize end = start + bytes;
                if (end < range_end) b.free_ranges.emplace(end, range_end - end);
                if (end > b.used) b.used = end;
                ++b.live;
                mapped = b.mapped + start;
                offset = b.address - d.arena.address + start;
                return true;
            }
            return false;
        };
        if (!blocks_.empty() && take(blocks_.back())) return true;
        // A large view may start a new slab while smaller views still fit earlier tails or freed holes.
        // Use those even after the budget can no longer accommodate another slab.
        if (device_local && blocks_.size() > 1)
            for (std::size_t i = blocks_.size() - 1; i != 0; --i)
                if (take(blocks_[i - 1])) return true;
        auto& vk = d.context->vk;
        if (!vk.vkGetBufferDeviceAddress) return false;
        if (failed_ && std::chrono::steady_clock::now() - failed_at_ < kRetryAfter) {
            ++quick_refusals_;
            return false;
        }
        const auto failed = [&] {
            failed_ = true;
            failed_at_ = std::chrono::steady_clock::now();
            return false;
        };
        auto drop_unfitting_spare = [&] {
            for (auto it = blocks_.begin(); it != blocks_.end(); ++it) {
                if (it->live == 0 && bytes > it->capacity) {
                    destroy_block(d, it);
                    return true;
                }
            }
            return false;
        };
        VkDeviceSize remaining_cap = reserved_ < cap ? cap - reserved_ : 0;
        if (bytes > remaining_cap) {
            if (drop_unfitting_spare()) remaining_cap = reserved_ < cap ? cap - reserved_ : 0;
            if (bytes > remaining_cap) {
                if (at_cap) *at_cap = true;
                return false;
            }
        }
        Block b{};
        b.capacity = std::max(bytes, std::min(kBlockSize, remaining_cap));
        VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        info.size = b.capacity;
        info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
        d.share(info);
        if (vk.vkCreateBuffer(d.info.device, &info, nullptr, &b.buffer) != VK_SUCCESS) return failed();
        VkMemoryRequirements req{};
        vk.vkGetBufferMemoryRequirements(d.info.device, b.buffer, &req);
        if (req.size > remaining_cap) {
            if (drop_unfitting_spare()) remaining_cap = reserved_ < cap ? cap - reserved_ : 0;
            if (req.size > remaining_cap) {
                if (at_cap) *at_cap = true;
                vk.vkDestroyBuffer(d.info.device, b.buffer, nullptr);
                return false;
            }
        }
        // Device-local (Resizable BAR) when asked and while the driver gives it; host memory otherwise. After the
        // story load the driver refused every 4 MiB device-local, host-visible block with a device address
        // (VK_ERROR_OUT_OF_HOST_MEMORY, speed-8/-9) while host memory still served: once that happens, later blocks
        // go straight to host memory instead of through DeviceState::allocate's sleeping retries first.
        // A device_local request under set_system_memory takes system memory instead, out of the types that are not
        // device-local (memory_type_order would otherwise give the first host-visible type, which can be the BAR).
        VkMemoryPropertyFlags chosen = 0;
        const bool system = device_local && system_memory_ != kDeviceLocal;
        const bool want_local = device_local && !local_refused_ && !system;
        VkMemoryRequirements types = req;
        if (system) types.memoryTypeBits = system_memory_types(d, req.memoryTypeBits, system_memory_ == kSystemCached);
        b.memory = d.allocate(types, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                              want_local ? VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
                              : system && system_memory_ == kSystemCached ? VK_MEMORY_PROPERTY_HOST_CACHED_BIT
                                                                          : 0,
                              &chosen, true);
        if (b.memory && system && !system_memory_logged_) {
            system_memory_logged_ = true;
            std::fprintf(stderr, "[rt] d3d12: title buffer copies in system memory (LEONIDA_GEOMETRY_HOST): memory "
                                 "flags 0x%x (%s, %s)\n",
                         unsigned(chosen),
                         (chosen & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) ? "device-local: no other type" : "not VRAM",
                         (chosen & VK_MEMORY_PROPERTY_HOST_CACHED_BIT) ? "write-back" : "write-combined");
        }
        if (b.memory && want_local && !(chosen & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
            local_refused_ = true;
            std::fprintf(stderr, "[rt] d3d12: title buffer windows: device-local memory refused a %llu KiB block; "
                                 "this and later blocks are in host memory\n",
                         static_cast<unsigned long long>(req.size >> 10));
        }
        void* host{};
        if (b.memory && vk.vkBindBufferMemory(d.info.device, b.buffer, b.memory, 0) == VK_SUCCESS &&
            vk.vkMapMemory(d.info.device, b.memory, 0, VK_WHOLE_SIZE, 0, &host) == VK_SUCCESS) {
            VkBufferDeviceAddressInfo address{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
            address.buffer = b.buffer;
            b.address = vk.vkGetBufferDeviceAddress(d.info.device, &address);
            const auto biased = b.address - d.arena.address;
            // A positive 47-bit offset works with both generations of the translated V# decoder.
            if (b.address && biased < (1ull << 47) && b.capacity <= (1ull << 47) - biased) {
                b.mapped = static_cast<std::uint8_t*>(host);
                b.reserved_bytes = req.size;
                b.free_ranges.emplace(0, b.capacity);
                if (const std::size_t n = published_count_.load(std::memory_order_relaxed); n < kPublished) {
                    published_[n].capacity = b.capacity;
                    published_[n].mapped = b.mapped;
                    published_[n].buffer = b.buffer;
                    published_[n].address.store(b.address, std::memory_order_relaxed);
                    published_count_.store(n + 1, std::memory_order_release);
                }
                layout_generation_.fetch_add(1, std::memory_order_release);
                blocks_.push_back(std::move(b));
                reserved_ += req.size;
                if (blocks_.size() == 1 || !(blocks_.size() & (blocks_.size() - 1)))
                    std::fprintf(stderr, "[rt] d3d12: title buffer shadows outside the arena: %zu blocks, "
                                         "new block %llu KiB at shader VA 0x%llx\n", blocks_.size(),
                                 static_cast<unsigned long long>(blocks_.back().capacity >> 10),
                                 static_cast<unsigned long long>(biased));
                return take(blocks_.back());
            }
            vk.vkUnmapMemory(d.info.device, b.memory);
        }
        vk.vkDestroyBuffer(d.info.device, b.buffer, nullptr);
        if (b.memory) vk.vkFreeMemory(d.info.device, b.memory, nullptr);
        return failed();
    }

    // Returns [offset, offset + bytes) to its block for reuse; keeps one empty 4 MiB block as a spare and destroys
    // any other empty block. Caller must guarantee no in-flight GPU work references this range.
    template<class Device>
    bool free(Device& d, VkDeviceSize offset, VkDeviceSize bytes) {
        if (!bytes) return false;
        const VkDeviceAddress bda = d.arena.address + offset;
        for (auto it = blocks_.begin(); it != blocks_.end(); ++it) {
            Block& b = *it;
            if (bda < b.address || bda - b.address >= b.capacity) continue;
            const VkDeviceSize start = bda - b.address;
            if (bytes > b.capacity - start) return false;
            VkDeviceSize merged_begin = start;
            VkDeviceSize merged_end = start + bytes;
            auto next = b.free_ranges.lower_bound(start);
            if (next != b.free_ranges.begin()) {
                auto prev = std::prev(next);
                if (prev->first + prev->second >= merged_begin) {
                    merged_begin = prev->first;
                    merged_end = std::max(merged_end, prev->first + prev->second);
                    next = b.free_ranges.erase(prev);
                }
            }
            while (next != b.free_ranges.end() && next->first <= merged_end) {
                merged_end = std::max(merged_end, next->first + next->second);
                next = b.free_ranges.erase(next);
            }
            b.free_ranges.emplace(merged_begin, merged_end - merged_begin);
            if (b.live > 0) --b.live;
            if (b.live == 0) {
                if (b.capacity == kBlockSize && empty_blocks() == 1) {
                    b.free_ranges.clear();
                    b.free_ranges.emplace(0, b.capacity);
                    b.used = 0;
                } else {
                    destroy_block(d, it);
                }
            }
            return true;
        }
        return false;
    }
    // Allocations refused at once because a new block failed less than kRetryAfter ago.
    std::uint64_t quick_refusals() const { return quick_refusals_; }

    bool lookup(VkDeviceAddress address, std::uint64_t& host, std::uint64_t& remaining) const {
        for (const auto& b : blocks_) {
            const auto delta = address - b.address;
            if (delta < b.used) {
                host = reinterpret_cast<std::uintptr_t>(b.mapped) + delta;
                remaining = b.used - delta;
                return true;
            }
        }
        return false;
    }

    // Device idle. No tracker writes into the blocks after this point.
    template<class Device>
    void release(Device& d) {
        for (auto& b : blocks_) {
            d.context->vk.vkUnmapMemory(d.info.device, b.memory);
            d.context->vk.vkDestroyBuffer(d.info.device, b.buffer, nullptr);
            d.context->vk.vkFreeMemory(d.info.device, b.memory, nullptr);
        }
        blocks_.clear();
        for (std::size_t i = 0, n = published_count_.load(std::memory_order_relaxed); i < n; ++i)
            published_[i].address.store(0, std::memory_order_relaxed);
        published_count_.store(0, std::memory_order_release);
        layout_generation_.fetch_add(1, std::memory_order_release);
        reserved_ = 0;
    }
};
}  // namespace rt::d3d12
