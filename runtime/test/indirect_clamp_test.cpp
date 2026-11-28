// GPU test for runtime/src/vulkan_indirect_clamp.h: the D3D12 ExecuteIndirect argument clamp.
//
//   - in-range records of every kind (draw, draw indexed, dispatch, mesh dispatch) come back bit-identical, and the
//     pass counts nothing as clamped;
//   - a garbage record is bounded (2^26 vertex instances, 65 535 groups per dimension, 2^20 groups per dispatch)
//     instead of asking for billions of groups, and a draw record of 2^23 vertices or more runs empty and free;
//   - a counted batch clamps the count to max_command_count and empties the records past it;
//   - a batch whose records fit but whose total work passes the 2^27 budget runs all but the records that no longer
//     fit, so thousands of garbage records cannot run for seconds;
//   - a dispatch is bounded by the device's own maxComputeWorkGroupCount, not by Vulkan's minimum of 65 535: a real
//     indirect dispatch of 70 000 x 1 x 1 groups (and of 65 535 in x or y, the old limit) through the clamped scratch
//     runs every group exactly once (cs_group_marker.comp marks each one). The idea is from codec-d/cool_fox
//     docs/findings.md ("blocky patches: indirect dispatches cut at 65,535 groups"); no code of theirs is used.
//
// A record that asks for billions of groups or vertices would run until the host resets the driver, so this test is
// about the values the pass writes, never about running such a dispatch here: every case ends in a copy of the
// clamped scratch back to the host. No game data. Run with LEONIDA_VK_VALIDATION=1 to count VUIDs.
#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
#include "vk_test_device.h"

#include "../src/vulkan_indirect_clamp.h"
#include "cs_group_marker_spv.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

int failures = 0;

#define CHECK(cond)                                                                          \
    do {                                                                                     \
        if (!(cond)) {                                                                       \
            std::fprintf(stderr, "line %d: CHECK failed: %s\n", __LINE__, #cond);            \
            ++failures;                                                                      \
        }                                                                                    \
    } while (0)

// The clamped records and the pass's counters of one ExecuteIndirect call.
struct Clamped {
    std::vector<std::uint32_t> records;  // records_out, `records` * words
    std::uint32_t count = 0;             // the clamped count word
    leonida::IndirectArgumentClamp::Counters counters{};  // what this call added
};

struct Env {
    vktest::Device& device;
    leonida::IndirectArgumentClamp& clamp;
    // Host-visible argument, count and readback buffers, one set reused by every case.
    VkBuffer arguments = VK_NULL_HANDLE;
    void* arguments_map = nullptr;
    VkBuffer counts = VK_NULL_HANDLE;
    void* counts_map = nullptr;
    VkBuffer readback = VK_NULL_HANDLE;
    void* readback_map = nullptr;
    VkDeviceSize argument_bytes = 4096, readback_bytes = 4096;

    // Runs one clamp call as a submission of its own: `records` argument records of `kind` at `stride` bytes, with
    // `count_value` bounding them when given. Reads the clamped scratch back once the queue is idle, and the
    // counters once that call's batch has been released, exactly as a completed submission does.
    bool run(std::uint32_t kind, std::uint32_t stride, std::uint32_t records, std::uint32_t words,
             const std::uint32_t* values, const std::uint32_t* count_value, Clamped& out) {
        std::memset(arguments_map, 0, size_t(argument_bytes));
        for (std::uint32_t r = 0; r < records; ++r)
            std::memcpy(static_cast<std::uint8_t*>(arguments_map) + r * stride, values + r * words, words * 4);
        if (count_value) std::memcpy(counts_map, count_value, 4);
        std::memset(readback_map, 0xcd, size_t(readback_bytes));
        const auto before = clamp.counters();
        {
            auto batch = clamp.batch();
            leonida::ClampedIndirect job{};
            if (!batch->prepare(arguments, 0, count_value ? counts : VK_NULL_HANDLE, 0, kind, stride, records, job))
                return false;
            const bool ok = device.run([&](VkCommandBuffer cb) {
                clamp.emit(cb, job);
                // The clamp's last barrier orders its writes before anything that follows, this copy included. The
                // records start the slice; a counted call has its clamped count word after them.
                const VkDeviceSize end = count_value ? job.count_offset + sizeof(std::uint32_t)
                                                     : job.offset + VkDeviceSize(records) * job.stride;
                VkBufferCopy copy{job.offset, 0, end - job.offset};
                device.fn<PFN_vkCmdCopyBuffer>("vkCmdCopyBuffer")(cb, job.buffer, readback, 1, &copy);
                VkMemoryBarrier host{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
                host.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                host.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
                device.fn<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier")(cb, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                                                           VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &host, 0,
                                                                           nullptr, 0, nullptr);
                return true;
            });
            if (!ok) return false;
            const auto* words_out = static_cast<const std::uint32_t*>(readback_map);
            out.count = count_value ? words_out[(job.count_offset - job.offset) / 4] : 0;
            out.records.assign(words_out, words_out + records * job.stride / 4);
        }
        const auto after = clamp.counters();
        out.counters.calls = after.calls - before.calls;
        out.counters.records = after.records - before.records;
        out.counters.records_clamped = after.records_clamped - before.records_clamped;
        out.counters.counts_clamped = after.counts_clamped - before.counts_clamped;
        out.counters.over_budget = after.over_budget - before.over_budget;
        out.counters.dispatch_records = after.dispatch_records - before.dispatch_records;
        out.counters.dispatch_wide = after.dispatch_wide - before.dispatch_wide;
        out.counters.dispatch_widest = after.dispatch_widest;  // the largest so far: cases go from narrow to wide
        out.counters.dispatch_cut = after.dispatch_cut - before.dispatch_cut;
        // The scratch is idle once the queue is; hand it back for the next case like a completed submission does.
        clamp.retire(VK_NULL_HANDLE, 0);
        clamp.collect();
        return true;
    }
};

// The words of one record in the clamped output, as the replay would hand them to Vulkan.
std::vector<std::uint32_t> record(const Clamped& c, std::uint32_t index, std::uint32_t words) {
    return {c.records.begin() + index * words, c.records.begin() + (index + 1) * words};
}

bool all_zero(const std::vector<std::uint32_t>& words) {
    return std::all_of(words.begin(), words.end(), [](std::uint32_t w) { return w == 0; });
}

}  // namespace

int main() {
    vktest::Device device;
    device.request_buffer_device_address = true;
    if (!device.create()) {
        std::printf("indirect_clamp_test: SKIP (no Vulkan device)\n");
        return 77;
    }
    VkPhysicalDeviceProperties properties{};
    device.instance_fn<PFN_vkGetPhysicalDeviceProperties>("vkGetPhysicalDeviceProperties")(device.physical,
                                                                                          &properties);
    VkPhysicalDeviceMemoryProperties memory{};
    device.instance_fn<PFN_vkGetPhysicalDeviceMemoryProperties>("vkGetPhysicalDeviceMemoryProperties")(
        device.physical, &memory);
    // Device-local memory with a device address, as gta6rt allocates the clamp's record scratch.
    const auto allocate = [&](const VkMemoryRequirements& requirements) -> VkDeviceMemory {
        VkMemoryAllocateFlagsInfo flags{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
        flags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
        VkMemoryAllocateInfo info{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, &flags};
        info.allocationSize = requirements.size;
        for (const std::uint32_t type : leonida::memory_type_order(
                 memory, requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0, requirements.size)) {
            info.memoryTypeIndex = type;
            VkDeviceMemory out = VK_NULL_HANDLE;
            if (device.fn<PFN_vkAllocateMemory>("vkAllocateMemory")(device.device, &info, nullptr, &out) == VK_SUCCESS)
                return out;
        }
        return VK_NULL_HANDLE;
    };
    leonida::IndirectArgumentClamp clamp(device.device_proc, device.device, memory, properties.limits, allocate);
    if (!clamp.loaded()) {
        std::printf("indirect_clamp_test: SKIP (no bufferDeviceAddress or no compute)\n");
        return 77;
    }

    Env env{device, clamp};
    const VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;
    env.arguments = device.buffer(env.argument_bytes, usage, &env.arguments_map);
    env.counts = device.buffer(env.argument_bytes, usage, &env.counts_map);
    env.readback = device.buffer(env.readback_bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &env.readback_map);
    CHECK(env.arguments && env.arguments_map && env.counts && env.counts_map && env.readback && env.readback_map);
    if (failures) return 1;

    // 1. A draw of 5 vertices in 2 instances: every word comes back untouched.
    {
        const std::uint32_t values[4] = {5, 2, 7, 3};
        Clamped out{};
        CHECK(env.run(leonida::indirect_clamp_draw, 16, 1, 4, values, nullptr, out));
        CHECK(record(out, 0, 4) == std::vector<std::uint32_t>(values, values + 4));
        CHECK(out.counters.records_clamped == 0 && out.counters.counts_clamped == 0);
    }

    // 2. A draw that asks for four billion vertices in four billion instances is an unfilled slot: it runs empty,
    //    keeping its locations, and charges nothing.
    {
        const std::uint32_t values[4] = {0xFFFFFFF0u, 0xFFFFFFFFu, 4, 6};
        Clamped out{};
        CHECK(env.run(leonida::indirect_clamp_draw, 16, 1, 4, values, nullptr, out));
        const auto words = record(out, 0, 4);
        CHECK(words[0] == 0 && words[1] == 0);
        CHECK(words[2] == 4 && words[3] == 6);
        CHECK(out.counters.records_clamped == 1 && out.counters.over_budget == 0);
    }

    // 2b. Just under the unfilled count a draw is a draw: its vertices stay, and its instances are cut to what keeps
    //     count * instances within 2^26. At the unfilled count it runs empty.
    {
        const std::uint32_t most = leonida::indirect_clamp_unfilled_draw - 1;
        const std::uint32_t values[4 * 2] = {most, 100, 0, 0, leonida::indirect_clamp_unfilled_draw, 1, 0, 0};
        Clamped out{};
        CHECK(env.run(leonida::indirect_clamp_draw, 16, 2, 4, values, nullptr, out));
        const auto under = record(out, 0, 4);
        CHECK(under[0] == most && under[1] == leonida::indirect_clamp_max_vertex_instances / most);
        CHECK(record(out, 1, 4)[0] == 0 && record(out, 1, 4)[1] == 0);
        CHECK(out.counters.records_clamped == 2);
    }

    // 2c. Garbage counts ahead of a real draw in one list: before, each was cut to 2^24 vertices and charged them, so
    //     eight used up the 2^27 budget and every later draw ran empty. Now they are free and the real draw runs.
    {
        std::vector<std::uint32_t> values;
        for (std::uint32_t r = 0; r < 16; ++r) values.insert(values.end(), {0x01000000u + r, 1, 0, 0});
        const std::uint32_t valid[] = {300, 2, 3, 4};
        values.insert(values.end(), valid, valid + 4);
        Clamped out{};
        CHECK(env.run(leonida::indirect_clamp_draw, 16, 17, 4, values.data(), nullptr, out));
        CHECK(out.counters.over_budget == 0 && out.counters.records_clamped == 16);
        CHECK(record(out, 16, 4) == std::vector<std::uint32_t>(valid, valid + 4));
    }

    // 3. A counted draw: the count is clamped to max_command_count, and the records past min(count, max) are empty.
    {
        std::uint32_t values[4 * 4] = {};
        for (std::uint32_t r = 0; r < 4; ++r) {
            values[r * 4 + 0] = 3 * (r + 1);
            values[r * 4 + 1] = 1;
            values[r * 4 + 2] = r;
            values[r * 4 + 3] = 0;
        }
        const std::uint32_t count = 2;
        Clamped out{};
        CHECK(env.run(leonida::indirect_clamp_draw, 16, 4, 4, values, &count, out));
        CHECK(out.count == 2);
        CHECK(record(out, 0, 4)[0] == 3 && record(out, 1, 4)[0] == 6);
        CHECK(all_zero(record(out, 2, 4)) && all_zero(record(out, 3, 4)));
        CHECK(out.counters.counts_clamped == 0);  // 2 is a count within max_command_count: nothing to clamp
        CHECK(out.counters.records_clamped == 0);  // the emptied tail is not a clamped value
    }

    // 3b. A count larger than max_command_count is clamped to it: a *Count draw never reads a count past its records
    //     (VUID-vkCmdDrawIndirectCount-countBuffer-02717).
    {
        const std::uint32_t values[4] = {3, 1, 0, 0};
        const std::uint32_t count = 0x7FFFFFFFu;
        Clamped out{};
        CHECK(env.run(leonida::indirect_clamp_draw, 16, 1, 4, values, &count, out));
        CHECK(out.count == 1);
        CHECK(out.counters.counts_clamped == 1);
    }

    // 4. An indexed draw keeps its two locations and its first instance: an unfilled index count runs empty, and a
    //    real one keeps its indices with the instances bounded.
    {
        const std::uint32_t values[5] = {0xFFFFFFFFu, 0xFFFFFFFFu, 10, 20, 30};
        Clamped out{};
        CHECK(env.run(leonida::indirect_clamp_draw_indexed, 20, 1, 5, values, nullptr, out));
        const auto words = record(out, 0, 5);
        CHECK(words[0] == 0 && words[1] == 0);
        CHECK(words[2] == 10 && words[3] == 20 && words[4] == 30);
        CHECK(out.counters.records_clamped == 1);
    }
    {
        const std::uint32_t values[5] = {3000, 0xFFFFFFFFu, 10, 20, 30};
        Clamped out{};
        CHECK(env.run(leonida::indirect_clamp_draw_indexed, 20, 1, 5, values, nullptr, out));
        const auto words = record(out, 0, 5);
        CHECK(words[0] == 3000 && words[1] == leonida::indirect_clamp_max_vertex_instances / 3000);
        CHECK(words[2] == 10 && words[3] == 20 && words[4] == 30);
        CHECK(out.counters.records_clamped == 1);
    }

    // 5. An in-range dispatch is copied untouched; a record with no groups costs nothing and is not clamped.
    {
        const std::uint32_t values[3 * 2] = {4, 2, 1, 0, 0, 0};
        Clamped out{};
        CHECK(env.run(leonida::indirect_clamp_dispatch, 12, 2, 3, values, nullptr, out));
        CHECK(record(out, 0, 3) == std::vector<std::uint32_t>(values, values + 3));
        CHECK(all_zero(record(out, 1, 3)));
        CHECK(out.counters.records_clamped == 0);
    }

    // 6. A dispatch asking for billions of groups in every dimension runs empty: x * y overflows the group limit.
    {
        const std::uint32_t values[3] = {0xFFFFFFF0u, 0xFFFFFFFFu, 1};
        Clamped out{};
        CHECK(env.run(leonida::indirect_clamp_dispatch, 12, 1, 3, values, nullptr, out));
        CHECK(all_zero(record(out, 0, 3)));
        CHECK(out.counters.records_clamped == 1);
    }

    // 6b. A dispatch over the device's per-dimension limit is cut there (y here; a device whose y limit is above 2^20
    //     runs it empty instead, as more than 2^20 groups), and one over 2^20 groups in all runs empty.
    {
        const std::uint32_t max_y = leonida::indirect_clamp_dispatch_limit(properties.limits, 1);
        const std::uint32_t values[3 * 2] = {1, 0x10000000u, 1, 65535, 65535, 1};
        Clamped out{};
        CHECK(env.run(leonida::indirect_clamp_dispatch, 12, 2, 3, values, nullptr, out));
        const auto per_dimension = record(out, 0, 3);
        if (max_y <= leonida::indirect_clamp_max_groups)
            CHECK(per_dimension[0] == 1 && per_dimension[1] == max_y && per_dimension[2] == 1);
        else
            CHECK(all_zero(per_dimension));
        CHECK(all_zero(record(out, 1, 3)));
        CHECK(out.counters.records_clamped == 2);
        // The dispatch counters: both records, neither wider than 65 535 in x, and the first cut at the device's
        // limit where it still runs.
        CHECK(out.counters.dispatch_records == 2 && out.counters.dispatch_wide == 0);
        CHECK(out.counters.dispatch_cut == (max_y <= leonida::indirect_clamp_max_groups ? 1u : 0u));
    }

    // 6c. x within the device's limit is not cut at 65 535: 70 000 groups (fewer than 2^20) come back as asked on a
    //     device that runs them, and are cut at its limit on one that does not.
    {
        const std::uint32_t max_x = leonida::indirect_clamp_dispatch_limit(properties.limits, 0);
        const std::uint32_t values[3] = {70000, 1, 1};
        Clamped out{};
        CHECK(env.run(leonida::indirect_clamp_dispatch, 12, 1, 3, values, nullptr, out));
        const auto words = record(out, 0, 3);
        CHECK(words[0] == std::min<std::uint32_t>(70000, max_x) && words[1] == 1 && words[2] == 1);
        CHECK(out.counters.records_clamped == (max_x >= 70000 ? 0u : 1u));
        // Counted as wider than 65 535 where it runs so, and as cut at the device's limit where it is cut.
        CHECK(out.counters.dispatch_records == 1 && out.counters.dispatch_wide == (max_x >= 70000 ? 1u : 0u));
        CHECK(out.counters.dispatch_widest == std::min<std::uint32_t>(70000, max_x));
        CHECK(out.counters.dispatch_cut == (max_x >= 70000 ? 0u : 1u));
    }

    // 7. A console mesh dispatch uses only x.
    {
        const std::uint32_t values[3] = {0x12345678u, 3, 1};
        Clamped out{};
        CHECK(env.run(leonida::indirect_clamp_mesh, 12, 1, 3, values, nullptr, out));
        const auto words = record(out, 0, 3);
        CHECK(words[0] == leonida::indirect_clamp_max_group_dimension && words[1] == 1 && words[2] == 1);
        CHECK(out.counters.records_clamped == 1);
    }

    // Unfilled slots leave the list budget available to a later valid record.
    {
        std::vector<std::uint32_t> values(65 * 4, UINT32_MAX);
        const std::uint32_t valid[] = {30, 2, 3, 4};
        std::copy_n(valid, 4, values.end() - 4);
        Clamped out{};
        CHECK(env.run(leonida::indirect_clamp_draw, 16, 65, 4, values.data(), nullptr, out));
        CHECK(out.counters.records_clamped == 64 && out.counters.over_budget == 0);
        CHECK(all_zero(record(out, 0, 4)));
        CHECK(record(out, 64, 4) == std::vector<std::uint32_t>(valid, valid + 4));
    }

    // 8. The batch budget: 33 records of 65 535 groups are 33 * 65 535 * 64 units, one record past 2^27, so exactly
    //    one record is emptied - which one is the GPU's to decide, as the atomic charge is a race.
    {
        constexpr std::uint32_t kRecords = 33;
        std::vector<std::uint32_t> values(kRecords * 3);
        for (std::uint32_t r = 0; r < kRecords; ++r) {
            values[r * 3 + 0] = 65535;
            values[r * 3 + 1] = 1;
            values[r * 3 + 2] = 1;
        }
        Clamped out{};
        CHECK(env.run(leonida::indirect_clamp_dispatch, 12, kRecords, 3, values.data(), nullptr, out));
        std::uint32_t ran = 0, empty = 0, other = 0;
        for (std::uint32_t r = 0; r < kRecords; ++r) {
            const auto words = record(out, r, 3);
            if (all_zero(words)) ++empty;
            else if (words[0] == 65535 && words[1] == 1 && words[2] == 1) ++ran;
            else ++other;
        }
        CHECK(ran == 32 && empty == 1 && other == 0);
        CHECK(out.counters.over_budget == 1);
    }

    // Three calls on one list consume two full draw allowances. A fresh list starts at zero.
    {
        const std::uint32_t args[] = {1u << 22, 16, 0, 0};  // 2^26 units: half the list budget, a real draw
        std::memcpy(env.arguments_map, args, sizeof(args));
        auto batch = clamp.batch();
        leonida::ClampedIndirect jobs[4]{};
        VkDeviceAddress shared = 0;
        for (unsigned i = 0; i < 4; ++i) {
            CHECK(batch->prepare(env.arguments, 0, VK_NULL_HANDLE, 0, leonida::indirect_clamp_draw, 16, 1, jobs[i]));
            if (i < 3) jobs[i].share_list_budget(shared);
        }
        CHECK(device.run([&](VkCommandBuffer cb) {
            for (unsigned i = 0; i < 4; ++i) {
                clamp.emit(cb, jobs[i]);
                VkBufferCopy copy{jobs[i].offset, i * sizeof(args), sizeof(args)};
                device.fn<PFN_vkCmdCopyBuffer>("vkCmdCopyBuffer")(cb, jobs[i].buffer, env.readback, 1, &copy);
            }
            VkMemoryBarrier host{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            host.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            host.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
            device.fn<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier")(cb, VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &host, 0, nullptr, 0, nullptr);
            return true;
        }));
        const auto* result = static_cast<const std::uint32_t*>(env.readback_map);
        CHECK(result[0] == args[0] && result[4] == args[0] && result[8] == 0 && result[12] == args[0]);
        CHECK(std::memcmp(env.arguments_map, args, sizeof(args)) == 0);
    }

    // 9. More ExecuteIndirect calls in one submission than there are counter slots (4,096): every call is still
    //    clamped, none falls back to the application's buffer; the calls past the slots only lose their counters.
    {
        std::memset(env.arguments_map, 0, size_t(env.argument_bytes));
        const std::uint32_t huge[4] = {0x00400000u, 0xFFFFFFFFu, 4, 6};  // 2^22 vertices: a draw, its instances cut
        std::memcpy(env.arguments_map, huge, sizeof huge);
        std::memset(env.readback_map, 0xcd, size_t(env.readback_bytes));
        const auto before = clamp.counters();
        {
            auto batch = clamp.batch();
            std::vector<leonida::ClampedIndirect> jobs(4096 + 2);
            bool all = true;
            for (auto& job : jobs)
                all = batch->prepare(env.arguments, 0, VK_NULL_HANDLE, 0, leonida::indirect_clamp_draw, 16, 1, job) &&
                      all;
            const auto during = clamp.counters();
            CHECK(all);
            CHECK(during.unclamped_calls == before.unclamped_calls);
            CHECK(during.uncounted_calls - before.uncounted_calls >= 2);
            const leonida::ClampedIndirect& last = jobs.back();
            CHECK(last.stats_buffer == VK_NULL_HANDLE && jobs.front().stats_buffer != VK_NULL_HANDLE);
            const bool ok = device.run([&](VkCommandBuffer cb) {
                clamp.emit(cb, last);
                VkBufferCopy copy{last.offset, 0, last.stride};
                device.fn<PFN_vkCmdCopyBuffer>("vkCmdCopyBuffer")(cb, last.buffer, env.readback, 1, &copy);
                VkMemoryBarrier host{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
                host.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                host.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
                device.fn<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier")(cb, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                                                           VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &host, 0,
                                                                           nullptr, 0, nullptr);
                return true;
            });
            CHECK(ok);
            const auto* words = static_cast<const std::uint32_t*>(env.readback_map);
            CHECK(words[0] == 0x00400000u &&
                  words[1] == leonida::indirect_clamp_max_vertex_instances / 0x00400000u && words[2] == 4 &&
                  words[3] == 6);
        }
        clamp.retire(VK_NULL_HANDLE, 0);
        clamp.collect();
    }

    // 10. Real indirect dispatches through the clamped scratch: every group runs exactly once. 70 000 x 1 x 1 is past
    //     the old fixed 65 535 (the groups past it never ran, leaving their output stale); 65 535 in x and in y are
    //     exactly at it. A size past this device's own limit is skipped: the clamp cuts it there by design.
    {
        const auto vkfn = [&](const char* name) { return device.device_proc(device.device, name); };
        VkShaderModuleCreateInfo module_info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        module_info.codeSize = sizeof(leonida::cs_group_marker_spirv);
        module_info.pCode = leonida::cs_group_marker_spirv;
        VkShaderModule module = VK_NULL_HANDLE;
        CHECK(reinterpret_cast<PFN_vkCreateShaderModule>(vkfn("vkCreateShaderModule"))(device.device, &module_info,
                                                                                       nullptr, &module) == VK_SUCCESS);
        struct MarkerPush {
            VkDeviceAddress marks;
            std::uint32_t groups;
            std::uint32_t pad;
        };
        const VkPushConstantRange range{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(MarkerPush)};
        VkPipelineLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        layout_info.pushConstantRangeCount = 1;
        layout_info.pPushConstantRanges = &range;
        VkPipelineLayout layout = VK_NULL_HANDLE;
        CHECK(reinterpret_cast<PFN_vkCreatePipelineLayout>(vkfn("vkCreatePipelineLayout"))(
                  device.device, &layout_info, nullptr, &layout) == VK_SUCCESS);
        VkComputePipelineCreateInfo pipeline_info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        pipeline_info.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        pipeline_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        pipeline_info.stage.module = module;
        pipeline_info.stage.pName = "main";
        pipeline_info.layout = layout;
        VkPipeline pipeline = VK_NULL_HANDLE;
        CHECK(reinterpret_cast<PFN_vkCreateComputePipelines>(vkfn("vkCreateComputePipelines"))(
                  device.device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pipeline) == VK_SUCCESS);

        constexpr std::uint32_t kMostGroups = 70000;
        void* marks_map = nullptr;
        const VkBuffer marks = device.buffer(VkDeviceSize(kMostGroups + 1) * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                             &marks_map);
        CHECK(marks && marks_map && pipeline);
        struct Case {
            std::uint32_t x, y, z;
        };
        const Case cases[] = {{70000, 1, 1}, {65535, 1, 1}, {1, 65535, 1}};
        for (const Case& c : cases) {
            if (!marks || !pipeline) break;
            const std::uint32_t groups = c.x * c.y * c.z;
            if (c.x > leonida::indirect_clamp_dispatch_limit(properties.limits, 0) ||
                c.y > leonida::indirect_clamp_dispatch_limit(properties.limits, 1) ||
                c.z > leonida::indirect_clamp_dispatch_limit(properties.limits, 2)) {
                std::printf("indirect_clamp_test: %u x %u x %u is past this device's maxComputeWorkGroupCount; "
                            "skipped\n", c.x, c.y, c.z);
                continue;
            }
            const std::uint32_t args[3] = {c.x, c.y, c.z};
            std::memset(env.arguments_map, 0, size_t(env.argument_bytes));
            std::memcpy(env.arguments_map, args, sizeof args);
            std::memset(marks_map, 0, size_t(kMostGroups + 1) * 4);
            {
                auto batch = clamp.batch();
                leonida::ClampedIndirect job{};
                CHECK(batch->prepare(env.arguments, 0, VK_NULL_HANDLE, 0, leonida::indirect_clamp_dispatch, 12, 1,
                                     job));
                const MarkerPush push{device.device_address(marks), groups, 0};
                CHECK(device.run([&](VkCommandBuffer cb) {
                    clamp.emit(cb, job);  // its last barrier orders the records before the indirect read
                    reinterpret_cast<PFN_vkCmdBindPipeline>(vkfn("vkCmdBindPipeline"))(
                        cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
                    reinterpret_cast<PFN_vkCmdPushConstants>(vkfn("vkCmdPushConstants"))(
                        cb, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof push, &push);
                    reinterpret_cast<PFN_vkCmdDispatchIndirect>(vkfn("vkCmdDispatchIndirect"))(cb, job.buffer,
                                                                                                job.offset);
                    VkMemoryBarrier host{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
                    host.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
                    host.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
                    device.fn<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier")(
                        cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &host, 0, nullptr,
                        0, nullptr);
                    return true;
                }));
            }
            clamp.retire(VK_NULL_HANDLE, 0);
            clamp.collect();
            const auto* m = static_cast<const std::uint32_t*>(marks_map);
            std::uint32_t once = 0, never = 0, twice = 0;
            for (std::uint32_t g = 0; g < groups; ++g) {
                if (m[g] == 1) ++once;
                else if (m[g] == 0) ++never;
                else ++twice;
            }
            if (once != groups || m[groups] != 0)
                std::fprintf(stderr, "indirect_clamp_test: %u x %u x %u: %u group(s) ran once, %u never, %u more than "
                             "once, %u past the end\n", c.x, c.y, c.z, once, never, twice, m[groups]);
            CHECK(once == groups && never == 0 && twice == 0 && m[groups] == 0);
        }
        reinterpret_cast<PFN_vkDestroyPipeline>(vkfn("vkDestroyPipeline"))(device.device, pipeline, nullptr);
        reinterpret_cast<PFN_vkDestroyPipelineLayout>(vkfn("vkDestroyPipelineLayout"))(device.device, layout, nullptr);
        reinterpret_cast<PFN_vkDestroyShaderModule>(vkfn("vkDestroyShaderModule"))(device.device, module, nullptr);
    }

    // Every case above ran under validation when LEONIDA_VK_VALIDATION=1.
    CHECK(device.validation_errors() == device.known_validation_errors());

    if (failures) {
        std::fprintf(stderr, "indirect_clamp_test: %d failure(s)\n", failures);
        return 1;
    }
    const auto counters = clamp.counters();
    std::printf("indirect_clamp_test: every kind clamped, in-range records bit-identical, the budget saturating; "
                "%llu record(s) clamped in %llu call(s), %llu count(s), validation errors %u\n",
                static_cast<unsigned long long>(counters.records_clamped),
                static_cast<unsigned long long>(counters.calls),
                static_cast<unsigned long long>(counters.counts_clamped), device.validation_errors());
    return 0;
}
