// Scratch buffers that live until the submission that used them has completed: device-local chunks handed out
// while a list is prepared, retired with the submission's timeline value, and reused (or freed) once the GPU is past
// it. The replay stages copies Vulkan cannot do in one command through them (vulkan_replay.h, depth <-> colour
// copies), and the ExecuteIndirect argument clamp keeps its record scratch in one too (vulkan_indirect_clamp.h).
// Single-threaded: the caller holds the context's lock. The same life cycle as the depth24 converter's own
// scratch (vulkan_depth24.h), which also needs descriptor pools and exists only on GPUs without D24.
// The chunks are transfer sources and destinations; `extra_usage` adds what another user needs (the expanded indices
// of quad lists, quad_index_expand.h: an index buffer written through its device address).
#pragma once

#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
#include <vulkan/vulkan.h>
#include "queue_families.h"

#include "vulkan_device_fault.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <utility>
#include <vector>

namespace leonida {

class ScratchBuffers {
public:
    // A device-local allocation for a chunk; null on failure.
    using Allocate = std::function<VkDeviceMemory(const VkMemoryRequirements&)>;
    // What the copy staging needs: transfer both ways.
    static constexpr VkBufferUsageFlags transfer_usage =
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

    // Legacy copy-scratch constructor. `extra_usage` is added to the transfer usage, and
    // `families` keeps retired chunks shareable across queue families.
    ScratchBuffers(PFN_vkGetDeviceProcAddr get, VkDevice device, Allocate allocate, VkBufferUsageFlags extra_usage = 0,
                   std::vector<std::uint32_t> families = {}, DeviceFaultDiagnostics* diagnostics = nullptr)
        : device_(device), diagnostics_(diagnostics), allocate_(std::move(allocate)),
          usage_(transfer_usage | extra_usage), families_(std::move(families)) {
        loaded_ = vk_.load(get, device, false);
        addressable_ = (usage_ & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) && vk_.vkGetBufferDeviceAddress;
    }

    // `usage` is what the chunks are created with. A chunk that declares VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
    // gets its device address from get() as well, and its memory must have been allocated with
    // VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT (the allocate callback's job). `families` keeps retired chunks shareable
    // across queue families (graphics + async compute).
    ScratchBuffers(PFN_vkGetDeviceProcAddr get, VkDevice device, Allocate allocate,
                   DeviceFaultDiagnostics* diagnostics, VkBufferUsageFlags usage,
                   std::vector<std::uint32_t> families = {})
        : device_(device), diagnostics_(diagnostics), allocate_(std::move(allocate)), usage_(usage),
          families_(std::move(families)),
          addressable_((usage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) != 0) {
        loaded_ = vk_.load(get, device, addressable_);
    }

    ~ScratchBuffers() {
        if (!loaded_) return;
        DeviceFaultDiagnostics::observe(diagnostics_, vk_.vkDeviceWaitIdle(device_), "vkDeviceWaitIdle");
        release(current_);
        for (auto& batch : retired_) release(batch);
        for (auto& chunk : spare_) destroy(chunk);
    }

    ScratchBuffers(const ScratchBuffers&) = delete;
    ScratchBuffers& operator=(const ScratchBuffers&) = delete;

    bool loaded() const { return loaded_; }

    // `bytes` of scratch for the next submission, at a 256-byte aligned offset (more than any copy needs).
    bool get(std::uint64_t bytes, VkBuffer& buffer, VkDeviceSize& offset) {
        return claim(bytes, buffer, offset, nullptr);
    }

    // The same for chunks created with a device address: `address` is the device address of the returned offset.
    bool get(std::uint64_t bytes, VkBuffer& buffer, VkDeviceSize& offset, VkDeviceAddress& address) {
        return claim(bytes, buffer, offset, &address);
    }

    // The chunks handed out since the last call belong to a submission that `timeline` reaches `ticket` for
    // (ticket 0: nothing was submitted, they are free once `collect` runs).
    void retire(VkSemaphore timeline, std::uint64_t ticket) {
        if (current_.chunks.empty()) return;
        current_.timeline = timeline;
        current_.ticket = ticket;
        retired_.push_back(std::move(current_));
        current_ = {};
    }

    // The submissions on `timeline` are over and the semaphore is about to go (its queue is idle).
    void forget(VkSemaphore timeline) {
        for (std::size_t i = 0; i < retired_.size();) {
            if (retired_[i].timeline != timeline) {
                ++i;
                continue;
            }
            release(retired_[i]);
            retired_.erase(retired_.begin() + std::ptrdiff_t(i));
        }
    }

    // Keeps for reuse (or frees) the chunks of finished submissions.
    void collect() {
        for (std::size_t i = 0; i < retired_.size();) {
            if (retired_[i].ticket != 0) {
                std::uint64_t done = 0;
                if (!retired_[i].timeline ||
                    DeviceFaultDiagnostics::observe(diagnostics_,
                        vk_.vkGetSemaphoreCounterValue(device_, retired_[i].timeline, &done),
                        "vkGetSemaphoreCounterValue (scratch)") != VK_SUCCESS ||
                    done < retired_[i].ticket) {
                    ++i;
                    continue;
                }
            }
            for (auto& chunk : retired_[i].chunks) {
                chunk.used = 0;
                if (spare_.size() < 4) spare_.push_back(chunk);
                else destroy(chunk);
            }
            retired_[i].chunks.clear();
            retired_.erase(retired_.begin() + std::ptrdiff_t(i));
        }
    }

private:
    struct Chunk {
        VkBuffer buffer = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkDeviceSize size = 0, used = 0;
    };
    struct Batch {
        std::vector<Chunk> chunks;
        VkSemaphore timeline = VK_NULL_HANDLE;
        std::uint64_t ticket = 0;
    };

    struct Functions {
#define LEONIDA_SCRATCH_FUNCTIONS(X)                                                                           \
    X(vkCreateBuffer) X(vkDestroyBuffer) X(vkGetBufferMemoryRequirements) X(vkBindBufferMemory) X(vkFreeMemory) \
    X(vkGetSemaphoreCounterValue) X(vkDeviceWaitIdle)
#define LEONIDA_SCRATCH_MEMBER(name) PFN_##name name = nullptr;
        LEONIDA_SCRATCH_FUNCTIONS(LEONIDA_SCRATCH_MEMBER)
#undef LEONIDA_SCRATCH_MEMBER
        PFN_vkGetBufferDeviceAddress vkGetBufferDeviceAddress = nullptr;  // optional: addressable pools only

        bool load(PFN_vkGetDeviceProcAddr get, VkDevice device, bool addressable) {
            bool all = true;
#define LEONIDA_SCRATCH_LOAD(name)                              \
    name = reinterpret_cast<PFN_##name>(get(device, #name));    \
    all = all && name != nullptr;
            LEONIDA_SCRATCH_FUNCTIONS(LEONIDA_SCRATCH_LOAD)
#undef LEONIDA_SCRATCH_LOAD
#undef LEONIDA_SCRATCH_FUNCTIONS
            // Only an addressable pool (the indirect clamp's) needs the addresses: a Vulkan 1.0/1.1 device without
            // bufferDeviceAddress keeps the transfer scratch.
            this->vkGetBufferDeviceAddress =
                reinterpret_cast<PFN_vkGetBufferDeviceAddress>(get(device, "vkGetBufferDeviceAddress"));
            return all && (!addressable || this->vkGetBufferDeviceAddress != nullptr);
        }
    };

    // Hands out `bytes` from this submission's chunks, making a new chunk where none fits.
    bool claim(std::uint64_t bytes, VkBuffer& buffer, VkDeviceSize& offset, VkDeviceAddress* address) {
        constexpr VkDeviceSize alignment = 256, minimum = 4u << 20;
        if (!loaded_ || !bytes) return false;
        if (address && !addressable_) return false;
        for (auto& chunk : current_.chunks) {
            const VkDeviceSize rem = chunk.used % alignment;
            const VkDeviceSize pad = rem ? alignment - rem : 0;
            if (pad > chunk.size - chunk.used) continue;
            const VkDeviceSize start = chunk.used + pad;
            if (bytes <= chunk.size - start) {
                chunk.used = start + bytes;
                buffer = chunk.buffer;
                offset = start;
                if (address) *address = device_address(chunk.buffer) + start;
                return true;
            }
        }
        Chunk chunk;
        for (std::size_t i = 0; i < spare_.size(); ++i)
            if (spare_[i].size >= bytes) {
                chunk = spare_[i];
                spare_.erase(spare_.begin() + std::ptrdiff_t(i));
                break;
            }
        if (!chunk.buffer && !make_chunk(std::max<std::uint64_t>(bytes, minimum), chunk)) return false;
        chunk.used = bytes;
        buffer = chunk.buffer;
        offset = 0;
        current_.chunks.push_back(chunk);
        if (address) *address = device_address(buffer);
        return true;
    }

    VkDeviceAddress device_address(VkBuffer buffer) const {
        VkBufferDeviceAddressInfo info{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
        info.buffer = buffer;
        return vk_.vkGetBufferDeviceAddress(device_, &info);
    }

    bool make_chunk(std::uint64_t size, Chunk& out) {
        VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        info.size = size;
        info.usage = usage_;
        // A retired chunk can be reused by a submission on a different queue family.
        share_across_queue_families(info, families_);
        if (vk_.vkCreateBuffer(device_, &info, nullptr, &out.buffer) != VK_SUCCESS) return false;
        VkMemoryRequirements requirements{};
        vk_.vkGetBufferMemoryRequirements(device_, out.buffer, &requirements);
        out.memory = allocate_ ? allocate_(requirements) : VK_NULL_HANDLE;
        if (!out.memory || vk_.vkBindBufferMemory(device_, out.buffer, out.memory, 0) != VK_SUCCESS) {
            destroy(out);
            return false;
        }
        out.size = size;
        return true;
    }
    void destroy(Chunk& chunk) {
        if (chunk.buffer) vk_.vkDestroyBuffer(device_, chunk.buffer, nullptr);
        if (chunk.memory) vk_.vkFreeMemory(device_, chunk.memory, nullptr);
        chunk = {};
    }
    void release(Batch& batch) {
        for (auto& chunk : batch.chunks) destroy(chunk);
        batch.chunks.clear();
    }

    VkDevice device_;
    DeviceFaultDiagnostics* diagnostics_;
    Allocate allocate_;
    VkBufferUsageFlags usage_ = transfer_usage;
    std::vector<std::uint32_t> families_;
    bool addressable_ = false;
    Functions vk_;
    bool loaded_ = false;
    Batch current_;               // handed out, not yet submitted
    std::vector<Batch> retired_;  // submitted, waiting for the GPU
    std::vector<Chunk> spare_;    // finished chunks kept for reuse
};

}  // namespace leonida
