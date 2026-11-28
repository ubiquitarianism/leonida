// The bindless set (src/vulkan_bindless.h) under churn and under memory pressure, at the Vulkan level, no shaders.
//
//   1. Churn: 2D image views are added, keyed in the hash table, and removed again, frame after frame, with one live
//      key rebound each frame so every flush makes a new generation. Past its initial capacity the table must stop
//      growing (erased slots and freed indices are reused), and after a warm-up the generations come from the
//      spares: no further table buffers or descriptor pools are allocated.
//   2. Device-local memory runs out while the table grows: the new table goes to host memory and the flush works.
//   3. No memory at all and no spare, nothing running: the table is rewritten in place.
//   4. No memory at all, work still running, and the table must grow: the flush does not fail (the submission would
//      be refused): it keeps the previous table and catches up once memory is back. An index removed meanwhile is
//      not given out again until then, since the stale table may still name it.
// vkAllocateMemory and vkCreateDescriptorPool are wrapped through the PFN_vkGetDeviceProcAddr the set loads with,
// to count them and to fail them on demand. Exit 77 when the device lacks the set's descriptor-indexing features.
// LEONIDA_VK_VALIDATION=1 runs under VK_LAYER_KHRONOS_validation; a validation error fails the test.
#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
#include "vk_test_device.h"
#include "../src/vulkan_bindless.h"

#include <cstdio>
#include <cstring>
#include <deque>

#define EXPECT(condition)                                                    \
    do {                                                                     \
        if (!(condition)) {                                                  \
            std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition);     \
            return 1;                                                        \
        }                                                                    \
    } while (0)

namespace {

using namespace leonida;

PFN_vkGetDeviceProcAddr g_get = nullptr;
PFN_vkAllocateMemory g_allocate = nullptr;
PFN_vkCreateDescriptorPool g_create_pool = nullptr;
VkPhysicalDeviceMemoryProperties g_memory{};

enum class Fail { none, device_local, all };
Fail g_fail = Fail::none;
bool g_fail_pools = false;
unsigned g_allocations = 0, g_pools = 0, g_refused = 0;

VKAPI_ATTR VkResult VKAPI_CALL allocate_memory(VkDevice device, const VkMemoryAllocateInfo* info,
                                               const VkAllocationCallbacks* callbacks, VkDeviceMemory* memory) {
    const bool local =
        (g_memory.memoryTypes[info->memoryTypeIndex].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0;
    if (g_fail == Fail::all || (g_fail == Fail::device_local && local)) {
        ++g_refused;
        return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    }
    ++g_allocations;
    return g_allocate(device, info, callbacks, memory);
}

VKAPI_ATTR VkResult VKAPI_CALL create_pool(VkDevice device, const VkDescriptorPoolCreateInfo* info,
                                           const VkAllocationCallbacks* callbacks, VkDescriptorPool* pool) {
    if (g_fail_pools) {
        ++g_refused;
        return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    }
    ++g_pools;
    return g_create_pool(device, info, callbacks, pool);
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL get_proc(VkDevice device, const char* name) {
    if (!std::strcmp(name, "vkAllocateMemory")) return reinterpret_cast<PFN_vkVoidFunction>(allocate_memory);
    if (!std::strcmp(name, "vkCreateDescriptorPool")) return reinterpret_cast<PFN_vkVoidFunction>(create_pool);
    return g_get(device, name);
}

DescriptorKey numbered(std::uint32_t n) {
    return {n, 0x9e3779b9u * n, n ^ 0x55555555u, n >> 3, 7, n * n, 0x80000000u, ~n};
}

}  // namespace

int main() {
    vktest::Device device;
    if (const auto status = device.initialize(); status != vktest::DeviceStatus::ready)
        return status == vktest::DeviceStatus::unavailable ? 77 : 1;
    if (!device.bindless) {
        std::printf("unsupported: the device lacks the bindless set's descriptor-indexing features\n");
        return 77;
    }
    g_get = device.device_proc;
    g_allocate = device.fn<PFN_vkAllocateMemory>("vkAllocateMemory");
    g_create_pool = device.fn<PFN_vkCreateDescriptorPool>("vkCreateDescriptorPool");
    device.instance_fn<PFN_vkGetPhysicalDeviceMemoryProperties>("vkGetPhysicalDeviceMemoryProperties")(device.physical,
                                                                                                       &g_memory);
    bool host_type = false;  // a HOST_VISIBLE | HOST_COHERENT type outside DEVICE_LOCAL memory
    for (std::uint32_t i = 0; i < g_memory.memoryTypeCount; ++i) {
        const auto flags = g_memory.memoryTypes[i].propertyFlags;
        host_type |= (flags & (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) ==
                         (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) &&
                     !(flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    }
    const auto create_view = device.fn<PFN_vkCreateImageView>("vkCreateImageView");
    const auto destroy_view = device.fn<PFN_vkDestroyImageView>("vkDestroyImageView");
    const VkImage image = device.image(VK_FORMAT_R8G8B8A8_UNORM, 1, 1, 1, VK_IMAGE_USAGE_SAMPLED_BIT);
    EXPECT(image);

    int failed = 0;
    {
        BindlessFeatures features;
        features.update_after_bind = true;
        features.partially_bound = device.partially_bound;
        BindlessSet set(get_proc, device.device, g_memory, features);
        EXPECT(set.ok());
        const DescriptorTable& table = set.table();
        const std::uint32_t initial_free = set.free_images(ImageDimension::dim_2d);
        const std::size_t initial_slots = table.capacity();

        // One submission per frame; run() waits for it, so `submitted` is also what has completed. busy: the
        // flush is told one more submission may still read the current set.
        std::uint64_t submitted = 0;
        const auto frame = [&](bool busy = false) {
            bool flushed = false;
            const bool ran = device.run([&](VkCommandBuffer command_buffer) {
                if (!set.fallbacks_ready()) set.record_fallback_init(command_buffer);
                set.release_retired(submitted);
                flushed = set.flush(command_buffer, submitted + (busy ? 1 : 0), [&] { return submitted; });
                return true;
            });
            set.table_upload_result(ran);
            ++submitted;
            return ran && flushed;
        };

        struct Live {
            DescriptorKey key;
            std::uint32_t index;
            VkImageView view;
        };
        std::deque<Live> live;
        std::uint32_t next_key = 1;
        const auto make_view = [&] {
            VkImageViewCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
            info.image = image;
            info.viewType = VK_IMAGE_VIEW_TYPE_2D;
            info.format = VK_FORMAT_R8G8B8A8_UNORM;
            info.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            VkImageView view = VK_NULL_HANDLE;
            return create_view(device.device, &info, nullptr, &view) == VK_SUCCESS ? view : VK_NULL_HANDLE;
        };
        const auto add = [&]() -> bool {
            const VkImageView view = make_view();
            if (!view) return false;
            const std::uint32_t index = set.add_image(ImageDimension::dim_2d, view, VK_IMAGE_LAYOUT_GENERAL);
            if (!index) return false;
            const DescriptorKey key = numbered(next_key++);
            set.insert_key(key, index);
            live.push_back({key, index, view});
            return true;
        };
        // What a texture's destruction does: its key leaves the table, its index the set, then its view goes. The
        // last flush ran to completion, so nothing reads the view (the device defers this until then).
        const auto remove = [&](const Live& entry) {
            set.erase_key(entry.key);
            set.remove_image(ImageDimension::dim_2d, entry.index);
            destroy_view(device.device, entry.view, nullptr);
        };

        // 1. Churn.
        constexpr unsigned frames = 400, per_frame = 8, max_live = 150;
        std::size_t warm_slots = 0;
        unsigned warm_allocations = 0, warm_pools = 0;
        std::uint64_t warm_generations = 0;
        for (unsigned f = 0; f < frames; ++f) {
            for (unsigned i = 0; i < per_frame; ++i) EXPECT(add());
            while (live.size() > max_live) {
                remove(live.front());
                live.pop_front();
            }
            // A live texture gets another view (the read-only entry of a T# replaced by one with storage, say): its
            // key is rebound, a change to an entry running work may probe, so the flush makes a new generation.
            Live& rebound = live[f % live.size()];
            const VkImageView view = make_view();
            EXPECT(view);
            const std::uint32_t index = set.add_image(ImageDimension::dim_2d, view, VK_IMAGE_LAYOUT_GENERAL);
            EXPECT(index);
            set.insert_key(rebound.key, index);
            EXPECT(set.flush_replaces_table());
            set.remove_image(ImageDimension::dim_2d, rebound.index);
            destroy_view(device.device, rebound.view, nullptr);
            rebound.index = index;
            rebound.view = view;
            EXPECT(frame());
            EXPECT(!set.last_flush_degraded() && set.retired_count() <= 1);
            if (f + 1 == frames / 2) {
                warm_slots = table.capacity();
                warm_allocations = g_allocations;
                warm_pools = g_pools;
                warm_generations = set.generations_made();
            }
        }
        const unsigned churn_allocations = g_allocations - warm_allocations, churn_pools = g_pools - warm_pools;
        std::printf("churn: %u frames, %u keys added, %zu live; table %zu -> %zu slots (%zu at half time), %llu "
                    "generations, %llu from spares; second half: %u allocations, %u descriptor pools\n",
                    frames, next_key - 1, live.size(), initial_slots, table.capacity(), warm_slots,
                    static_cast<unsigned long long>(set.generations_made()),
                    static_cast<unsigned long long>(set.spares_reused()), churn_allocations, churn_pools);
        EXPECT(table.capacity() > initial_slots);  // driven past its initial capacity
        EXPECT(table.size() == live.size());
        // Bounded: 3200 keys went through a table that holds 150, and it stays where the warm-up left it, at most
        // one more doubling (probe clustering can force one below half load).
        EXPECT(table.capacity() <= 2 * warm_slots && table.capacity() <= 16 * max_live);
        EXPECT(set.free_images(ImageDimension::dim_2d) == initial_free - live.size());
        EXPECT(set.generations_made() - warm_generations >= frames / 2);
        EXPECT(set.spares_reused() >= frames / 2 - 4);
        EXPECT(churn_allocations <= 2 && churn_pools <= 1);  // a doubling at most allocates; the spares serve the rest
        EXPECT(set.table_host_fallbacks() == 0 && set.degraded_flushes() == 0);

        // 2. Device-local memory is gone and the table must grow past every spare: host memory.
        const std::size_t before_growth = table.capacity();
        std::vector<DescriptorKey> extra;
        for (std::uint32_t k = 0; table.capacity() == before_growth; ++k) {
            extra.push_back(numbered(1000000 + k));
            set.insert_key(extra.back(), live.front().index);
        }
        // Without a host memory type outside device-local memory (llvmpipe has one type for everything) refusing
        // device-local memory refuses all of it, which is case 4 below, not this one: the table then grows normally.
        if (host_type) g_fail = Fail::device_local;
        const unsigned refused_before = g_refused;
        EXPECT(frame());
        EXPECT(!set.last_flush_degraded() && set.table_buffer_bytes() >= table.size_bytes());
        if (host_type) {
            EXPECT(g_refused > refused_before && set.table_host_fallbacks() == 1);
            EXPECT(!set.table_in_device_local_memory());
        } else {
            std::printf("no host memory outside device-local memory: the host fallback is not checked\n");
        }
        EXPECT(frame());  // releases the old generation; it is too small to keep as a spare
        EXPECT(set.spare_count() == 0 && set.retired_count() == 0);

        // 3. Nothing can be allocated, no spare fits, nothing runs: the table is rewritten in place.
        g_fail = Fail::all;
        g_fail_pools = true;
        const std::size_t bytes = set.table_buffer_bytes();
        set.insert_key(live.back().key, live.front().index);  // rebinds a live key: needs a new generation
        EXPECT(set.flush_replaces_table());
        EXPECT(frame());
        EXPECT(set.in_place_rewrites() == 1 && !set.last_flush_degraded() && set.table_buffer_bytes() == bytes);
        EXPECT(!set.flush_replaces_table());

        // 4. Nothing can be allocated, work still runs, and the table must grow: the flush keeps the old table and
        // still succeeds, with the descriptors of the new image written; the table catches up once memory is back.
        EXPECT(add());
        const std::size_t old_slots = table.capacity();
        for (std::uint32_t k = 0; table.capacity() == old_slots; ++k) {
            extra.push_back(numbered(2000000 + k));
            set.insert_key(extra.back(), live.back().index);
        }
        EXPECT(set.flush_replaces_table());
        EXPECT(frame(true));
        EXPECT(set.last_flush_degraded() && set.degraded_flushes() == 1);
        EXPECT(set.last_failure() != nullptr && set.table_buffer_bytes() == bytes);
        EXPECT(set.flush_replaces_table());  // still owed
        // A texture goes while the GPU's table is stale: that table still maps its key to its index, so the index is
        // not given out again (to answer that key with another image) until the table has caught up.
        const Live gone = live.front();
        live.pop_front();
        const std::uint32_t free_before = set.free_images(ImageDimension::dim_2d);
        remove(gone);
        EXPECT(set.free_images(ImageDimension::dim_2d) == free_before);
        EXPECT(frame(true));  // and again: never a failed flush
        EXPECT(set.degraded_flushes() == 2 && set.free_images(ImageDimension::dim_2d) == free_before);
        EXPECT(add() && live.back().index != gone.index);
        g_fail = Fail::none;
        g_fail_pools = false;
        const std::uint64_t made = set.generations_made();
        EXPECT(frame(true));
        EXPECT(!set.last_flush_degraded() && set.generations_made() == made + 1);
        EXPECT(set.free_images(ImageDimension::dim_2d) == free_before);  // the held index is back; add() took one
        EXPECT(set.table_buffer_bytes() >= table.size_bytes() && !set.flush_replaces_table());
        std::printf("pressure: %llu host fallbacks, %llu in-place rewrites, %llu degraded flushes (%s), table %zu "
                    "slots\n",
                    static_cast<unsigned long long>(set.table_host_fallbacks()),
                    static_cast<unsigned long long>(set.in_place_rewrites()),
                    static_cast<unsigned long long>(set.degraded_flushes()),
                    set.last_failure() ? set.last_failure() : "-", table.capacity());

        for (const Live& entry : live) remove(entry);
        live.clear();
        EXPECT(frame());
        EXPECT(set.free_images(ImageDimension::dim_2d) == initial_free);
        set.release_retired();
    }
    {
        // 5. A first flush under full memory (review of #64): no table ever reached the GPU, so there is nothing to
        // keep. Degrading would leave shaders reading uninitialised buffers (word 0 is the probe mask): the flush
        // refuses instead, and succeeds once memory is back.
        BindlessFeatures features;
        features.update_after_bind = true;
        features.partially_bound = device.partially_bound;
        BindlessSet set(get_proc, device.device, g_memory, features);
        EXPECT(set.ok());
        const std::size_t initial_slots = set.table().capacity();
        for (std::uint32_t k = 0; set.table().capacity() == initial_slots; ++k)
            set.insert_key(numbered(3000000 + k), 0);
        g_fail = Fail::all;
        g_fail_pools = true;
        bool flushed = true;
        device.run([&](VkCommandBuffer command_buffer) {
            if (!set.fallbacks_ready()) set.record_fallback_init(command_buffer);
            flushed = set.flush(command_buffer, 1, [] { return std::uint64_t(0); });
            return true;
        });
        set.table_upload_result(true);
        EXPECT(!flushed && set.degraded_flushes() == 0);
        g_fail = Fail::none;
        g_fail_pools = false;
        device.run([&](VkCommandBuffer command_buffer) {
            flushed = set.flush(command_buffer, 1, [] { return std::uint64_t(0); });
            return true;
        });
        set.table_upload_result(true);
        EXPECT(flushed && !set.last_flush_degraded() && set.table_buffer_bytes() >= set.table().size_bytes());
        set.release_retired();
        std::printf("first flush under full memory: refused, then uploaded once memory is back\n");
    }
    if (device.validation_errors()) {
        std::printf("%u validation errors\n", device.validation_errors());
        failed = 1;
    }
    if (!failed) std::printf("vk_bindless_grow: ok\n");
    return failed;
}
