// The console heap translation pass (src/console_heap_fix.comp, src/console_heap_gpu.h) against its CPU reference:
// synthetic descriptor words only (no game data). Checks the key table mirror on the CPU first, then that the GPU
// writes every slot exactly as console_heap_translate_slot does and records the same misses, through key table
// updates, a new range table, a submission that failed (its work queued again) and a miss list that overflows.
// Exits 77 without a Vulkan device with bufferDeviceAddress.
#include "vk_test_device.h"
#include "../src/console_heap_gpu.h"

#include <cstdio>
#include <cstring>
#include <random>
#include <array>
#include <set>
#include <vector>

#define EXPECT(x)                                                       \
    do {                                                                \
        if (!(x)) {                                                     \
            std::fprintf(stderr, "line %d: %s\n", __LINE__, #x);        \
            return 1;                                                   \
        }                                                               \
    } while (0)

namespace {

using leonida::ConsoleHeapKeys;
using leonida::ConsoleHeapRange;
using leonida::ConsoleHeapTables;

// A buffer V# half: base, stride, records and dword 3 with TYPE bits 30-31 clear.
void vsharp(std::uint32_t* w, std::uint64_t base, std::uint32_t stride, std::uint32_t records, std::uint32_t dword3) {
    w[0] = std::uint32_t(base);
    w[1] = std::uint32_t((base >> 32) & 0xffff) | (stride << 16);
    w[2] = records;
    w[3] = dword3 & 0x3fffffffu;
}

int keys_on_cpu() {
    ConsoleHeapKeys keys(6);  // 64 entries: probe chains overlap
    const ConsoleHeapKeys::Key a{1, 2, 3, 4}, b{5, 6, 7, 8};
    EXPECT(keys.find(a) < 0);
    EXPECT(keys.put(a, 0x1000, 0x100, false));
    EXPECT(keys.put(b, 0, 0x40, true));
    std::uint64_t to = 0;
    bool none = true;
    EXPECT(keys.lookup(a, to, none) && to == 0x1000 && !none);
    EXPECT(keys.lookup(b, to, none) && none);
    EXPECT(keys.size() == 2);
    // Only positive answers fall in a released shader range; the edge is exclusive.
    EXPECT(keys.erase_overlapping(0x1100, 0x2000) == 0);
    EXPECT(keys.erase_overlapping(0x10ff, 0x1100) == 1);
    EXPECT(keys.find(a) < 0 && keys.find(b) >= 0);
    // Replacing an answer keeps one entry and moves its index.
    EXPECT(keys.put(b, 0x5000, 0x10, false));
    EXPECT(keys.size() == 1);
    EXPECT(keys.erase_overlapping(0x5008, 0x5009) == 1);
    EXPECT(keys.size() == 0);
    // Fill one probe chain: the 17th key with the same start finds no slot.
    // V#s whose words differ only in high bits (bases megabytes apart): the hash must still spread them. Bare FNV-1a
    // put all of these in one probe chain.
    {
        ConsoleHeapKeys big(20);
        for (std::uint32_t i = 0; i < 20000; ++i) {
            const ConsoleHeapKeys::Key k{(i & 0xfffu) << 20, ((i >> 12) << 8) | (64u << 16), 256, 0x2000};
            EXPECT(big.put(k, 0x100000ull * i, 256, false));
        }
        EXPECT(big.size() == 20000);
    }
    std::vector<ConsoleHeapKeys::Key> same;
    for (std::uint32_t i = 0; same.size() < ConsoleHeapKeys::kProbes + 1 && i < 1u << 20; ++i) {
        const ConsoleHeapKeys::Key k{i, 0, 1, 0};
        if ((ConsoleHeapKeys::hash(k) & keys.mask()) == 7) same.push_back(k);
    }
    EXPECT(same.size() == ConsoleHeapKeys::kProbes + 1);
    for (std::size_t i = 0; i < ConsoleHeapKeys::kProbes; ++i) EXPECT(keys.put(same[i], 0x100 * i, 1, false));
    EXPECT(!keys.put(same.back(), 1, 1, false));
    std::vector<std::uint32_t> updates;
    keys.take_updates(updates);
    EXPECT(!updates.empty() && updates.size() % ConsoleHeapKeys::kUpdateWords == 0);
    EXPECT(!keys.has_updates());
    // The reference: ranges first (checked only), then keys, else a miss.
    ConsoleHeapTables ranges;
    ranges.ranges = {{0x10000, 0x20000, 0x400000, true}, {0x30000, 0x40000, 0x500000, false}};
    ConsoleHeapKeys k2(8);
    std::uint32_t w[8] = {};
    vsharp(w, 0x10100, 16, 4, 0);
    EXPECT(leonida::console_heap_translate_slot(w, ranges, k2) == 0);
    EXPECT(w[0] == 0x400100 && (w[1] & 0xffff) == 0 && w[2] == 4);
    vsharp(w, 0x1ff00, 16, 32, 0);  // 512 bytes past a range with 256 left: no records
    EXPECT(leonida::console_heap_translate_slot(w, ranges, k2) == 0 && !w[0] && !w[1] && !w[2]);
    vsharp(w, 0x30000, 0, 64, 0);  // an unchecked range: the key table, then a miss
    EXPECT(leonida::console_heap_translate_slot(w, ranges, k2) == 1 && !w[2]);
    // Backed spans leave a base as it is (before the key table); outside the ask table nothing is a miss.
    ranges.backed = {{0x80000, 0x90000}};
    ranges.ask = {{0x30000, 0x40000}};
    ranges.use_ask = true;
    vsharp(w, 0x80100, 0, 64, 0);
    EXPECT(leonida::console_heap_translate_slot(w, ranges, k2) == 0 && w[0] == 0x80100 && w[2] == 64);
    vsharp(w, 0x8ffc0, 0, 128, 0);  // runs past the span: not backed, outside ask: no records, no miss
    EXPECT(leonida::console_heap_translate_slot(w, ranges, k2) == 0 && !w[2]);
    vsharp(w, 0x30000, 0, 64, 0);  // in ask: a miss
    EXPECT(leonida::console_heap_translate_slot(w, ranges, k2) == 1);
    // Signed offsets: a base with bit 47 set is the negative shader VA it names.
    ranges.signed_offsets = true;
    ranges.backed = {{0xffff800000000000ull, 0xffff800000010000ull}};
    vsharp(w, 0x800000000100ull, 0, 64, 0);
    EXPECT(leonida::console_heap_translate_slot(w, ranges, k2) == 0 && w[2] == 64);
    std::vector<std::pair<std::uint64_t, std::uint64_t>> spans{{10, 20}, {15, 30}, {30, 40}, {50, 60}, {5, 5}};
    leonida::merge_console_heap_spans(spans);
    EXPECT(spans.size() == 2 && spans[0].first == 10 && spans[0].second == 40 && spans[1].first == 50);
    return 0;
}

struct Case {
    ConsoleHeapTables t;
    std::vector<ConsoleHeapRange>& ranges = t.ranges;
    ConsoleHeapKeys keys{14};
};

// Random slot words over a mix of classes; bases drawn from the ranges and keys so every path is taken.
void make_slots(std::mt19937& rng, std::uint32_t* words, std::size_t slots, Case& c,
                std::vector<ConsoleHeapKeys::Key>& entered) {
    std::uniform_int_distribution<std::uint32_t> any;
    for (std::size_t s = 0; s < slots; ++s) {
        std::uint32_t* w = words + 8 * s;
        for (int i = 0; i < 8; ++i) w[i] = any(rng);
        const std::uint32_t kind = any(rng) % 10;
        if (kind == 0) {  // T#: TYPE >= 8
            w[3] = (w[3] & 0x0fffffffu) | 0x80000000u;
            continue;
        }
        if (kind == 1) {  // other words with bits 30-31 set
            w[3] |= 0x40000000u;
            continue;
        }
        if (kind == 2) {
            std::memset(w, 0, 32);
            continue;
        }
        for (int half = 0; half < 2; ++half) {
            std::uint32_t* k = w + 4 * half;
            const std::uint32_t pick = any(rng) % 8;
            const std::uint32_t stride = any(rng) % 3 == 0 ? 0 : any(rng) % 64;
            const std::uint32_t records = 1 + any(rng) % 256;
            if (pick < 3 && !c.ranges.empty()) {
                const auto& r = c.ranges[any(rng) % c.ranges.size()];
                vsharp(k, r.title_va + any(rng) % (r.end - r.title_va), stride, records, any(rng));
            } else if (pick < 5) {
                vsharp(k, 0x200000000ull + (any(rng) % 4096) * 64, stride, records, any(rng));
                if (pick == 3) {
                    const ConsoleHeapKeys::Key key{k[0], k[1], k[2], k[3]};
                    const bool none = any(rng) % 4 == 0;
                    if (c.keys.put(key, 0x10000000ull + (any(rng) % 65536) * 256, 4096, none)) entered.push_back(key);
                }
            } else if (pick == 5) {
                vsharp(k, 0, stride, records, any(rng));
            } else if (pick == 6) {
                vsharp(k, 0x300000000ull + any(rng), stride, 0, any(rng));  // no records: left as it is
            } else if (half == 1) {
                k[3] |= 0x40000000u;  // a counter half that is no V#
            } else {
                vsharp(k, 0x400000000ull + any(rng), stride, records, any(rng));  // nowhere: a miss
            }
        }
    }
}

}  // namespace

int main() {
    if (const int failed = keys_on_cpu()) return failed;
    std::puts("console heap keys (CPU): ok");

    vktest::Device d;
    d.request_buffer_device_address = true;
    const auto status = d.initialize();
    if (status == vktest::DeviceStatus::unavailable || (status == vktest::DeviceStatus::ready && !d.buffer_device_address)) {
        std::printf("vk_console_heap_gpu_test: no Vulkan device with bufferDeviceAddress, skipped\n");
        return 77;
    }
    if (status != vktest::DeviceStatus::ready) return 1;
    VkPhysicalDeviceMemoryProperties memory{};
    d.instance_fn<PFN_vkGetPhysicalDeviceMemoryProperties>("vkGetPhysicalDeviceMemoryProperties")(d.physical, &memory);

    leonida::ConsoleHeapGpu gpu(d.device_proc, d.device, memory, 14);
    EXPECT(gpu.init());

    constexpr std::uint32_t kSlots = 6000;
    void* input_data = nullptr;
    void* copy_data = nullptr;
    const VkBuffer input = d.buffer(VkDeviceSize(kSlots) * 32, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &input_data);
    const VkBuffer copy = d.buffer(VkDeviceSize(kSlots) * 32, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &copy_data);
    EXPECT(input && copy && input_data && copy_data);
    const int heap = gpu.add_heap(d.device_address(input), d.device_address(copy), kSlots);
    EXPECT(heap >= 0);
    auto* in = static_cast<std::uint32_t*>(input_data);
    auto* out = static_cast<std::uint32_t*>(copy_data);

    std::mt19937 rng(1234);
    Case c;
    c.ranges = {{0x10000, 0x30000, 0x100000, true},
                {0x40000, 0x48000, 0x0000ffff00000000ull, true},  // bits 32-47 of the answer
                {0x50000, 0x60000, 0x200000, false},
                {0x100000000ull, 0x100100000ull, 0x300000, true}};
    std::vector<ConsoleHeapKeys::Key> entered;
    make_slots(rng, in, kSlots, c, entered);
    // The GPU's mirror gets the same keys.
    gpu.with_keys([&](ConsoleHeapKeys& k) {
        for (const auto& key : entered) {
            std::uint64_t to = 0;
            bool none = false;
            c.keys.lookup(key, to, none);
            k.put(key, to, 4096, none);
        }
        return 0;
    });
    gpu.set_ranges(c.ranges);

    // One round: the listed slots are written as the reference says, the others keep the marker, and the misses
    // are the reference's.
    auto round = [&](const std::vector<std::uint32_t>& listed, std::uint64_t ticket, bool expect_overflow) -> int {
        std::memset(out, 0xcd, std::size_t(kSlots) * 32);
        gpu.queue(heap, listed.begin(), listed.end());
        bool recorded = false;
        EXPECT(d.run([&](VkCommandBuffer cb) { return recorded = gpu.record(cb); }));
        EXPECT(recorded);
        gpu.commit(true, ticket);
        std::set<std::uint32_t> misses;
        bool overflowed = false;
        gpu.drain(ticket, [&](int h, std::uint32_t slot, std::uint32_t half) {
            if (h == heap) misses.insert(slot * 2 + half);
        }, [&](int) { overflowed = true; });
        EXPECT(gpu.in_flight() == 0);
        EXPECT(overflowed == expect_overflow);
        std::set<std::uint32_t> expected_misses;
        std::vector<std::uint8_t> is_listed(kSlots, 0);
        for (const std::uint32_t s : listed) is_listed[s] = 1;
        for (std::uint32_t s = 0; s < kSlots; ++s) {
            std::uint32_t w[8];
            std::memcpy(w, in + 8 * s, 32);
            if (!is_listed[s]) {
                for (int i = 0; i < 8; ++i) EXPECT(out[8 * s + i] == 0xcdcdcdcdu);
                continue;
            }
            const unsigned missed = leonida::console_heap_translate_slot(w, c.t, c.keys);
            if (missed & 1) expected_misses.insert(s * 2);
            if (missed & 2) expected_misses.insert(s * 2 + 1);
            if (std::memcmp(w, out + 8 * s, 32)) {
                std::fprintf(stderr, "slot %u: in %08x %08x %08x %08x %08x %08x %08x %08x\n", s, in[8 * s],
                             in[8 * s + 1], in[8 * s + 2], in[8 * s + 3], in[8 * s + 4], in[8 * s + 5], in[8 * s + 6],
                             in[8 * s + 7]);
                std::fprintf(stderr, "  gpu %08x %08x %08x %08x %08x %08x %08x %08x\n", out[8 * s], out[8 * s + 1],
                             out[8 * s + 2], out[8 * s + 3], out[8 * s + 4], out[8 * s + 5], out[8 * s + 6],
                             out[8 * s + 7]);
                std::fprintf(stderr, "  cpu %08x %08x %08x %08x %08x %08x %08x %08x\n", w[0], w[1], w[2], w[3], w[4],
                             w[5], w[6], w[7]);
                return 1;
            }
        }
        if (!expect_overflow) EXPECT(misses == expected_misses);
        else EXPECT(misses.size() == leonida::ConsoleHeapGpu::kMissCapacity);
        std::printf("  %zu slots, %zu misses\n", listed.size(), misses.size());
        return 0;
    };

    std::vector<std::uint32_t> all(kSlots);
    for (std::uint32_t s = 0; s < kSlots; ++s) all[s] = s;
    EXPECT(round(all, 1, false) == 0);
    std::puts("console heap pass, every slot: ok");

    // Key table updates and a new range table: erase some keys, enter answers for some misses, move a range.
    std::vector<std::uint32_t> some;
    for (std::uint32_t s = 0; s < kSlots; s += 3) some.push_back(s);
    std::size_t erased = 0, added = 0;
    for (std::size_t i = 0; i < entered.size(); i += 2) {
        c.keys.erase(entered[i]);
        gpu.with_keys([&](ConsoleHeapKeys& k) { return k.erase(entered[i]); });
        ++erased;
    }
    for (std::uint32_t s = 0; s < kSlots && added < 200; ++s) {
        std::uint32_t w[8];
        std::memcpy(w, in + 8 * s, 32);
        std::uint32_t copy_w[8];
        std::memcpy(copy_w, w, 32);
        const unsigned missed = leonida::console_heap_translate_slot(copy_w, c.t, c.keys);
        if (!(missed & 1)) continue;
        const ConsoleHeapKeys::Key key{w[0], w[1], w[2], w[3]};
        c.keys.put(key, 0x7000000000ull + added * 64, 64, added % 5 == 0);
        gpu.with_keys([&](ConsoleHeapKeys& k) { return k.put(key, 0x7000000000ull + added * 64, 64, added % 5 == 0); });
        ++added;
    }
    c.ranges[0].shader_va = 0x900000;
    c.ranges[2].checked = true;
    gpu.set_ranges(c.ranges);
    EXPECT(round(some, 2, false) == 0);
    std::printf("console heap pass, %zu keys erased, %zu entered, ranges moved: ok\n", erased, added);

    // A submission that did not reach the queue: its slots and updates come back for the next pass.
    {
        const ConsoleHeapKeys::Key key{in[8 * 5], in[8 * 5 + 1], in[8 * 5 + 2], in[8 * 5 + 3]};
        gpu.queue(heap, 5);
        bool recorded = false;
        EXPECT(d.run([&](VkCommandBuffer cb) { return recorded = gpu.record(cb); }));
        EXPECT(recorded && !gpu.pending());
        gpu.commit(false, 0);
        EXPECT(gpu.pending() && gpu.in_flight() == 0);
        (void)key;
    }
    EXPECT(round({5}, 3, false) == 0);
    // Backed and ask tables: part of the keyed bases become backed (answered before the key table), and only bases
    // in the ask span miss.
    c.t.backed = {{0x200000000ull, 0x200008000ull}, {0x10000ull, 0x18000ull}};
    c.t.ask = {{0x400000000ull, 0x500000000ull}};
    c.t.use_ask = true;
    gpu.set_spans(leonida::ConsoleHeapGpu::backed_table, c.t.backed);
    gpu.set_spans(leonida::ConsoleHeapGpu::ask_table, c.t.ask);
    EXPECT(round(all, 7, false) == 0);
    std::puts("console heap pass, backed and ask tables: ok");
    {
        // kZeroFirst: words that are no buffer pair and no T# get dwords 0-3 zeroed; the last queue() decides.
        std::vector<std::uint32_t> flagged;
        for (std::uint32_t s2 = 0; s2 < kSlots && flagged.size() < 64; ++s2)
            if (in[8 * s2 + 3] >> 30) flagged.push_back(s2);
        EXPECT(flagged.size() == 64);
        std::memset(out, 0xcd, std::size_t(kSlots) * 32);
        for (std::size_t i = 0; i < flagged.size(); ++i) {
            gpu.queue(heap, flagged[i] | leonida::ConsoleHeapGpu::kZeroFirst);
            if (i % 2) gpu.queue(heap, flagged[i]);  // queued again without the bit: copied as it is
        }
        bool recorded = false;
        EXPECT(d.run([&](VkCommandBuffer cb) { return recorded = gpu.record(cb); }));
        gpu.commit(true, 8);
        gpu.drain(8, [](int, std::uint32_t, std::uint32_t) {}, [](int) {});
        EXPECT(recorded);
        for (std::size_t i = 0; i < flagged.size(); ++i) {
            std::uint32_t w[8];
            std::memcpy(w, in + 8 * flagged[i], 32);
            leonida::console_heap_translate_slot(w, c.t, c.keys, i % 2 == 0);
            EXPECT(!std::memcmp(w, out + 8 * flagged[i], 32));
            if (i % 2 == 0) EXPECT(!out[8 * flagged[i]] && !out[8 * flagged[i] + 3]);
        }
    }
    std::puts("console heap pass, words that are no descriptor: ok");
    {
        // rescan(): one range dispatch rewrites the buffer pairs of [0, n) and leaves every other slot as it was.
        constexpr std::uint32_t n = 4000;
        std::memset(out, 0xcd, std::size_t(kSlots) * 32);
        gpu.rescan(heap, n);
        EXPECT(gpu.pending());
        bool recorded = false;
        EXPECT(d.run([&](VkCommandBuffer cb) { return recorded = gpu.record(cb); }));
        gpu.commit(true, 9);
        std::set<std::uint32_t> misses;
        gpu.drain(9, [&](int h, std::uint32_t slot, std::uint32_t half) { if (h == heap) misses.insert(slot * 2 + half); },
                  [](int) {});
        EXPECT(recorded);
        std::set<std::uint32_t> expected;
        std::size_t pairs = 0;
        for (std::uint32_t s2 = 0; s2 < kSlots; ++s2) {
            std::uint32_t w[8];
            std::memcpy(w, in + 8 * s2, 32);
            if (s2 >= n || (w[3] >> 30)) {
                for (int i = 0; i < 8; ++i) EXPECT(out[8 * s2 + i] == 0xcdcdcdcdu);
                continue;
            }
            ++pairs;
            const unsigned missed = leonida::console_heap_translate_slot(w, c.t, c.keys);
            if (missed & 1) expected.insert(s2 * 2);
            if (missed & 2) expected.insert(s2 * 2 + 1);
            EXPECT(!std::memcmp(w, out + 8 * s2, 32));
        }
        EXPECT(misses == expected && pairs > 1000);
        EXPECT(!gpu.pending());
    }
    std::puts("console heap pass, range rescan: ok");
    {
        // rescan_filtered(): only the slots whose metadata (left by the last pass over them, the rescan above) meets a
        // change span are translated again: a title span holding one slot's base, a shader span holding another's
        // result. The rest keep what they hold, without the title's words being read.
        constexpr std::uint32_t n = 4000;
        struct Half {
            std::uint64_t base, to, span;
        };
        std::vector<std::array<Half, 2>> meta(n);
        std::vector<bool> pair(n, false);
        for (std::uint32_t s2 = 0; s2 < n; ++s2) {
            std::uint32_t w[8];
            std::memcpy(w, in + 8 * s2, 32);
            if (w[3] >> 30) continue;
            pair[s2] = true;
            std::uint32_t before[8];
            std::memcpy(before, w, 32);
            leonida::console_heap_translate_slot(w, c.t, c.keys);
            for (int h = 0; h < 2; ++h) {
                const std::uint32_t* b = before + 4 * h;
                const std::uint32_t* a = w + 4 * h;
                if (!b[2]) {
                    meta[s2][h] = {};
                    continue;
                }
                const std::uint64_t stride = (b[1] >> 16) & 0x3fff;
                const std::uint64_t span = std::min<std::uint64_t>(std::uint64_t(b[2]) * std::max<std::uint64_t>(stride, 1),
                                                                   0xffffffffull);
                meta[s2][h] = {b[0] | (std::uint64_t(b[1] & 0xffff) << 32),
                               a[2] ? a[0] | (std::uint64_t(a[1] & 0xffff) << 32) : 0, span};
            }
        }
        std::uint32_t by_title = UINT32_MAX, by_result = UINT32_MAX;
        for (std::uint32_t s2 = 0; s2 < n; ++s2) {
            if (!pair[s2] || !meta[s2][0].span) continue;
            if (by_title == UINT32_MAX) by_title = s2;
            else if (meta[s2][0].to && meta[s2][0].to != meta[by_title][0].base) {
                by_result = s2;
                break;
            }
        }
        EXPECT(by_title != UINT32_MAX && by_result != UINT32_MAX);
        const leonida::ConsoleHeapSpans title{{meta[by_title][0].base, meta[by_title][0].base + 1}};
        const leonida::ConsoleHeapSpans shader{{meta[by_result][0].to, meta[by_result][0].to + 1}};
        const auto meets = [](const leonida::ConsoleHeapSpans& spans, std::uint64_t v, std::uint64_t bytes) {
            for (const auto& [a, b] : spans)
                if (v < b && a < v + bytes) return true;
            return false;
        };
        std::set<std::uint32_t> touched;
        for (std::uint32_t s2 = 0; s2 < n; ++s2)
            for (const Half& h : meta[s2]) {
                if (!pair[s2] || !h.span) continue;
                if (meets(title, h.base, 1) || meets(shader, h.base, h.span) || (h.to && meets(shader, h.to, h.span)))
                    touched.insert(s2);
            }
        EXPECT(touched.count(by_title) && touched.count(by_result) && touched.size() < 1000);
        std::memset(out, 0xcd, std::size_t(kSlots) * 32);
        const auto before = gpu.take_stats();
        (void)before;
        gpu.rescan_filtered(heap, n, shader, title);
        EXPECT(gpu.pending());
        bool recorded = false;
        EXPECT(d.run([&](VkCommandBuffer cb) { return recorded = gpu.record(cb); }));
        gpu.commit(true, 10);
        gpu.drain(10, [](int, std::uint32_t, std::uint32_t) {}, [](int) {});
        EXPECT(recorded && !gpu.pending());
        for (std::uint32_t s2 = 0; s2 < kSlots; ++s2) {
            if (!touched.count(s2)) {
                for (int i = 0; i < 8; ++i) EXPECT(out[8 * s2 + i] == 0xcdcdcdcdu);
                continue;
            }
            std::uint32_t w[8];
            std::memcpy(w, in + 8 * s2, 32);
            leonida::console_heap_translate_slot(w, c.t, c.keys);
            EXPECT(!std::memcmp(w, out + 8 * s2, 32));
        }
        EXPECT(gpu.take_stats().filtered == n);
        // A failed submission: the filtered rescan and its spans come back, and the next pass does them.
        std::memset(out, 0xcd, std::size_t(kSlots) * 32);
        gpu.rescan_filtered(heap, n, shader, title);
        EXPECT(d.run([&](VkCommandBuffer cb) { return recorded = gpu.record(cb); }));
        gpu.commit(false, 0);
        EXPECT(gpu.pending());
        EXPECT(d.run([&](VkCommandBuffer cb) { return recorded = gpu.record(cb); }));
        gpu.commit(true, 11);
        gpu.drain(11, [](int, std::uint32_t, std::uint32_t) {}, [](int) {});
        std::size_t rewritten = 0;
        for (std::uint32_t s2 = 0; s2 < n; ++s2) rewritten += out[8 * s2] != 0xcdcdcdcdu || out[8 * s2 + 3] != 0xcdcdcdcdu;
        EXPECT(rewritten == touched.size());
        std::printf("console heap pass, filtered rescan: ok (%zu of %u slots)\n", touched.size(), n);
    }
    {
        // queue_copy(): buffer copies recorded at the start of the pass; their keepalive goes once the pass retires.
        constexpr VkDeviceSize kBytes = 64 * 1024;
        void* src_data = nullptr;
        void* dst_data = nullptr;
        const VkBuffer src = d.buffer(kBytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, &src_data);
        const VkBuffer dst = d.buffer(kBytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &dst_data);
        EXPECT(src && dst && src_data && dst_data);
        auto* sw = static_cast<std::uint32_t*>(src_data);
        for (std::size_t i = 0; i < kBytes / 4; ++i) sw[i] = std::uint32_t(i * 2654435761u);
        std::memset(dst_data, 0, kBytes);
        auto keep = std::make_shared<int>(7);
        std::weak_ptr<int> watch = keep;
        gpu.queue_copy(src, 4096, dst, 0, 8192, keep);
        gpu.queue_copy(src, 0, dst, 32768, 4096, keep);
        keep.reset();
        EXPECT(gpu.pending() && !watch.expired());
        bool recorded = false;
        EXPECT(d.run([&](VkCommandBuffer cb) { return recorded = gpu.record(cb); }));
        EXPECT(recorded && !watch.expired());
        gpu.commit(true, 10);
        gpu.drain(10, [](int, std::uint32_t, std::uint32_t) {}, [](int) {});
        EXPECT(watch.expired());
        const auto* dw = static_cast<const std::uint8_t*>(dst_data);
        EXPECT(!std::memcmp(dw, static_cast<const std::uint8_t*>(src_data) + 4096, 8192));
        EXPECT(!std::memcmp(dw + 32768, src_data, 4096));
        EXPECT(dw[8192] == 0 && dw[32768 + 4096] == 0);
    }
    std::puts("console heap pass, queued copies: ok");
    c.t.backed.clear();
    c.t.ask = {{0, ~0ull}};  // everything asks again: the overflow cases below need their misses
    gpu.set_spans(leonida::ConsoleHeapGpu::backed_table, c.t.backed);
    gpu.set_spans(leonida::ConsoleHeapGpu::ask_table, c.t.ask);
    std::puts("console heap pass, failed submission queued again: ok");

    // More misses than the list holds: the overflow is reported, and the slots are still written.
    {
        constexpr std::uint32_t kStride = 0;
        for (std::uint32_t s = 0; s < kSlots; ++s) {
            vsharp(in + 8 * s, 0x500000000ull + s * 0x100, kStride, 16, 0);
            vsharp(in + 8 * s + 4, 0x600000000ull + s * 0x100, kStride, 16, 0);
        }
        std::vector<std::uint32_t> many;
        for (std::uint32_t s = 0; s < kSlots; ++s) many.push_back(s);
        // 12,000 misses fit; list them twice over with a second heap view of the same buffers to pass 65,536.
        const int heap2 = gpu.add_heap(d.device_address(input), d.device_address(copy), kSlots);
        EXPECT(heap2 >= 0);
        std::size_t total = 0;
        for (int h = 0; h < 6; ++h) {  // 6 x 12,000 = 72,000 misses in one pass over the two heaps
            gpu.queue(h % 2 ? heap2 : heap, many.begin(), many.end());
            total += 2 * many.size();
        }
        (void)total;
        bool recorded = false;
        EXPECT(d.run([&](VkCommandBuffer cb) { return recorded = gpu.record(cb); }));
        // Queued twice per heap only counts once: 2 heaps x 12,000 = 24,000 misses, under the list. Queue a heap of
        // its own instead to pass it.
        gpu.commit(true, 4);
        std::size_t misses = 0;
        bool overflowed = false;
        gpu.drain(4, [&](int, std::uint32_t, std::uint32_t) { ++misses; }, [&](int) { overflowed = true; });
        EXPECT(recorded && misses == 24000 && !overflowed);
        gpu.remove_heap(heap2);
    }
    std::puts("console heap pass, a slot queued twice is translated once: ok");
    {
        // Six heap views of the same buffers: 6 x 12,000 = 72,000 misses, past the list's 65,536.
        std::vector<int> views;
        for (int i = 0; i < 5; ++i) views.push_back(gpu.add_heap(d.device_address(input), d.device_address(copy), kSlots));
        std::vector<std::uint32_t> many;
        for (std::uint32_t s = 0; s < kSlots; ++s) many.push_back(s);
        gpu.queue(heap, many.begin(), many.end());
        for (const int v : views) {
            EXPECT(v >= 0);
            gpu.queue(v, many.begin(), many.end());
        }
        bool recorded = false;
        EXPECT(d.run([&](VkCommandBuffer cb) { return recorded = gpu.record(cb); }));
        gpu.commit(true, 5);
        std::size_t misses = 0;
        bool overflowed = false;
        gpu.drain(5, [&](int, std::uint32_t, std::uint32_t) { ++misses; }, [&](int) { overflowed = true; });
        EXPECT(recorded && overflowed && misses == leonida::ConsoleHeapGpu::kMissCapacity);
        for (std::uint32_t s = 0; s < kSlots; ++s) EXPECT(out[8 * s] == 0 && out[8 * s + 2] == 0);
        for (const int v : views) gpu.remove_heap(v);
    }
    std::puts("console heap pass, miss list overflow: ok");
    {
        // Clustered words (bases a megabyte apart) entered in both tables: every slot answered, none missed.
        Case clustered;
        for (std::uint32_t s2 = 0; s2 < 3000; ++s2) {
            for (int half = 0; half < 2; ++half) {
                const std::uint32_t i = 2 * s2 + std::uint32_t(half);
                std::uint32_t* k = in + 8 * s2 + 4 * half;
                vsharp(k, 0x700000000ull + (std::uint64_t(i) << 20), 16, 8, 0x2000);
                const ConsoleHeapKeys::Key key{k[0], k[1], k[2], k[3]};
                EXPECT(clustered.keys.put(key, 0x1000000ull + i * 256ull, 128, (i % 7) == 0));
                EXPECT(gpu.with_keys([&](ConsoleHeapKeys& keys) {
                    return keys.put(key, 0x1000000ull + i * 256ull, 128, (i % 7) == 0);
                }));
            }
        }
        std::swap(c.keys, clustered.keys);
        c.ranges.clear();
        gpu.set_ranges(c.ranges);
        EXPECT(round(all, 6, false) == 0);
    }
    std::puts("console heap pass, clustered words: ok");
    const auto stats = gpu.take_stats();
    std::printf("console heap pass: %llu passes, %llu slots, %llu updates, %llu misses, %llu overflows, %llu failed\n",
                (unsigned long long)stats.passes, (unsigned long long)stats.slots, (unsigned long long)stats.updates,
                (unsigned long long)stats.misses, (unsigned long long)stats.overflows,
                (unsigned long long)stats.failed);
    EXPECT(d.validation_errors() == 0);
    std::puts("vk_console_heap_gpu_test: ok");
    return 0;
}
