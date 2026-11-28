// Console descriptor heap translation on the GPU (LEONIDA_CONSOLE_HEAP_GPU=1): the buffer V#s the title writes into
// its console descriptor heaps are rebased by a compute pass (console_heap_fix.comp) at the start of the submission
// that reads them, instead of one by one on the submitting thread (DeviceState::fix_console_slot).
//
// What stays on the CPU (shims/d3d12_vulkan_device.cpp): finding the changed slots (the heap's ShadowPages and a
// compare against the bytes each slot was last made from), registering the T#s (they are copied as they are; shaders
// find them by their words), and every lookup that can make something: a V# the pass cannot answer is written with no
// records and recorded in the pass's miss list; once the submission is done the CPU resolves its words with
// vsharp_address (which may make a geometry copy), enters the answer in the key table, and queues the slot again.
//
// The pass answers a V# half from four tables, in vsharp_address's order:
//   - the range table: the title-VA ranges of the alias table (src/title_va_alias.h), sorted, each with the shader VA
//     of its copy and whether the CPU found that copy backed. Rebuilt when the alias table or the backings change;
//   - the backed table: the shader-VA spans of the live GPU allocations (DeviceState's gpu_backings, merged) and of
//     the geometry copies' blocks. A base already inside one (a V# our API wrote, or the title copied) is left as it
//     is, as vsharp_address's backed() leaves it. Merged neighbours make it a superset of what backed() accepts, but
//     never of memory that is not there;
//   - the key table: the exact 16 bytes of a V# half and what vsharp_address answered for them (a shader VA, or no
//     backing), open addressed with up to kProbes slots per key (FNV-1a and fmix32). The CPU keeps the
//     mirror below; its changes reach the GPU as updates the pass applies first, so the table changes in GPU order;
//   - the ask table: the title's memory and its GPU windows. A base outside them that nothing above answered gets no
//     records (vsharp_address could only answer 0); one inside them is a miss, for the CPU.
// When a backing or an alias range goes, every key entry whose answer falls in its shader VAs is dropped and every V#
// slot of the heap is translated again.
//
// ConsoleHeapKeys and console_heap_translate_slot are the CPU reference of the pass (vk_console_heap_gpu_test checks
// the GPU against them); ConsoleHeapGpu records the pass.
#pragma once
#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
#include <vulkan/vulkan.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

#include "console_heap_fix_spv.h"
#include "vulkan_aftermath.h"
#include "vulkan_memory.h"

namespace leonida {

struct ConsoleHeapRange {
    std::uint64_t title_va = 0, end = 0, shader_va = 0;
    bool checked = false;  // the whole [shader_va, shader_va + end - title_va) is backed
};

// Sorted, disjoint [start, end) spans: the backed and ask tables.
using ConsoleHeapSpans = std::vector<std::pair<std::uint64_t, std::uint64_t>>;

// Sorts and merges spans (overlapping or touching ones become one); empty and wrapping ones are dropped.
inline void merge_console_heap_spans(ConsoleHeapSpans& spans) {
    std::erase_if(spans, [](const auto& s) { return s.second <= s.first; });
    std::sort(spans.begin(), spans.end());
    std::size_t out = 0;
    for (std::size_t i = 0; i < spans.size(); ++i) {
        if (out && spans[i].first <= spans[out - 1].second) spans[out - 1].second = std::max(spans[out - 1].second, spans[i].second);
        else spans[out++] = spans[i];
    }
    spans.resize(out);
}

// [v, v + bytes) inside one span (bytes 0 counts as 1).
inline bool console_heap_spans_hold(const ConsoleHeapSpans& spans, std::uint64_t v, std::uint64_t bytes) {
    auto it = std::upper_bound(spans.begin(), spans.end(), v,
                               [](std::uint64_t x, const auto& s) { return x < s.first; });
    if (it == spans.begin()) return false;
    --it;
    const std::uint64_t n = std::max<std::uint64_t>(bytes, 1);
    return v < it->second && n <= it->second - v;
}

// What the reference (and the pass) answers from, besides the key table.
struct ConsoleHeapTables {
    std::vector<ConsoleHeapRange> ranges;
    ConsoleHeapSpans backed;
    ConsoleHeapSpans ask;
    bool use_ask = false;         // without it every unanswered half is a miss
    bool signed_offsets = false;  // LEONIDA_IMPORT_BELOW_ARENA: a base with bit 47 set is a negative offset
};

// The key table's CPU mirror. An entry's GPU form is 8 dwords: the key, the answer's low 32 bits, its bits 32-47 and
// the flags in bits 16-31 of the sixth, two pad dwords.
class ConsoleHeapKeys {
public:
    using Key = std::array<std::uint32_t, 4>;
    static constexpr std::uint32_t kProbes = 16;
    static constexpr std::uint32_t kUsed = 1, kNone = 2;
    static constexpr std::uint32_t kEntryWords = 8, kUpdateWords = 12;

    explicit ConsoleHeapKeys(std::uint32_t log2_entries = 20)
        : mask_((1u << std::min<std::uint32_t>(log2_entries, 26)) - 1), entries_(std::size_t(mask_) + 1),
          dirty_flag_(std::size_t(mask_) + 1, 0) {}

    // FNV-1a over the four dwords, then MurmurHash3's fmix32 (console_heap_fix.comp key_hash): the finalizer makes
    // the low bits, which index the table, depend on every bit of the key. Bare FNV-1a put every V# whose base
    // differed only above bit 20 into one probe chain, and the 17th of them found no slot.
    static std::uint32_t hash(const Key& k) {
        std::uint32_t h = 2166136261u;
        for (const std::uint32_t w : k) h = (h ^ w) * 16777619u;
        h ^= h >> 16;
        h *= 0x85EBCA6Bu;
        h ^= h >> 13;
        h *= 0xC2B2AE35u;
        h ^= h >> 16;
        return h;
    }

    std::uint32_t mask() const { return mask_; }
    std::size_t size() const { return used_; }
    std::size_t capacity() const { return entries_.size(); }

    // The entry's index, or -1.
    std::int64_t find(const Key& k) const {
        const std::uint32_t h = hash(k);
        for (std::uint32_t p = 0; p < kProbes; ++p) {
            const std::uint32_t at = (h + p) & mask_;
            if (entries_[at].flags & kUsed && entries_[at].key == k) return at;
        }
        return -1;
    }
    // True with the answer when the key is entered: `none` for no backing.
    bool lookup(const Key& k, std::uint64_t& to, bool& none) const {
        const std::int64_t at = find(k);
        if (at < 0) return false;
        to = entries_[std::size_t(at)].to;
        none = entries_[std::size_t(at)].flags & kNone;
        return true;
    }

    // Enters (or replaces) the answer for `k`: the shader VA `to` reaching `span` bytes, or no backing. False when all
    // of the key's probe slots hold other keys.
    bool put(const Key& k, std::uint64_t to, std::uint64_t span, bool none) {
        std::int64_t at = find(k);
        if (at < 0) {
            const std::uint32_t h = hash(k);
            for (std::uint32_t p = 0; p < kProbes && at < 0; ++p)
                if (!(entries_[(h + p) & mask_].flags & kUsed)) at = (h + p) & mask_;
            if (at < 0) return false;
            ++used_;
        } else {
            unindex(std::uint32_t(at));
        }
        Entry& e = entries_[std::size_t(at)];
        e.key = k;
        e.to = to;
        e.span = span;
        e.flags = kUsed | (none ? kNone : 0);
        if (!none) {
            by_to_.emplace(to, std::uint32_t(at));
            longest_ = std::max(longest_, span);
        }
        touch(std::uint32_t(at));
        return true;
    }

    bool erase(const Key& k) {
        const std::int64_t at = find(k);
        if (at < 0) return false;
        erase_at(std::uint32_t(at));
        return true;
    }

    // Every answer reaching into [start, end) of the shader VAs: dropped. Returns how many.
    std::size_t erase_overlapping(std::uint64_t start, std::uint64_t end) {
        if (end <= start || by_to_.empty()) return 0;
        const std::uint64_t from = start > longest_ ? start - longest_ : 0;
        std::vector<std::uint32_t> gone;
        for (auto it = by_to_.lower_bound(from); it != by_to_.end() && it->first < end; ++it) {
            const Entry& e = entries_[it->second];
            if (e.to + std::max<std::uint64_t>(e.span, 1) > start) gone.push_back(it->second);
        }
        for (const std::uint32_t at : gone) erase_at(at);
        return gone.size();
    }

    // Every "no backing" answer, for a caller that asks again after something changed: their keys.
    void take_none(std::vector<Key>& out, std::size_t at_most) {
        out.clear();
        for (std::uint32_t at = 0; at < entries_.size() && out.size() < at_most; ++at) {
            if ((entries_[at].flags & (kUsed | kNone)) != (kUsed | kNone)) continue;
            out.push_back(entries_[at].key);
            erase_at(at);
        }
    }

    void clear() {
        for (std::uint32_t at = 0; at < entries_.size(); ++at)
            if (entries_[at].flags & kUsed) erase_at(at);
        by_to_.clear();
        longest_ = 0;
    }
    // Everything gone with no updates: for a GPU table that is cleared as a whole (ConsoleHeapGpu::reset_keys).
    void clear_silent() {
        std::fill(entries_.begin(), entries_.end(), Entry{});
        for (const std::uint32_t at : dirty_) dirty_flag_[at] = 0;
        dirty_.clear();
        by_to_.clear();
        used_ = 0;
        longest_ = 0;
    }

    void gpu_words(std::uint32_t at, std::uint32_t out[kEntryWords]) const {
        const Entry& e = entries_[at];
        std::memset(out, 0, kEntryWords * 4);
        if (!(e.flags & kUsed)) return;
        for (int i = 0; i < 4; ++i) out[i] = e.key[i];
        out[4] = std::uint32_t(e.to);
        out[5] = std::uint32_t((e.to >> 32) & 0xffffu) | (std::uint32_t(e.flags) << 16);
    }

    // The entries changed since the last take, as update records (index, 3 pad dwords, the entry's 8 dwords).
    void take_updates(std::vector<std::uint32_t>& out) {
        out.clear();
        out.reserve(dirty_.size() * kUpdateWords);
        for (const std::uint32_t at : dirty_) {
            dirty_flag_[at] = 0;
            const std::size_t base = out.size();
            out.resize(base + kUpdateWords, 0);
            out[base] = at;
            gpu_words(at, out.data() + base + 4);
        }
        dirty_.clear();
    }
    bool has_updates() const { return !dirty_.empty(); }
    // Every entry again (a new GPU table, or updates that never reached the GPU).
    void touch_all() {
        for (std::uint32_t at = 0; at < entries_.size(); ++at)
            if (entries_[at].flags & kUsed) touch(at);
    }
    void touch(std::uint32_t at) {
        if (!dirty_flag_[at]) {
            dirty_flag_[at] = 1;
            dirty_.push_back(at);
        }
    }

private:
    struct Entry {
        Key key{};
        std::uint64_t to = 0;
        std::uint64_t span = 0;
        std::uint32_t flags = 0;
    };
    void unindex(std::uint32_t at) {
        const Entry& e = entries_[at];
        if (!(e.flags & kUsed) || (e.flags & kNone)) return;
        auto [lo, hi] = by_to_.equal_range(e.to);
        for (auto it = lo; it != hi; ++it)
            if (it->second == at) {
                by_to_.erase(it);
                break;
            }
    }
    void erase_at(std::uint32_t at) {
        if (!(entries_[at].flags & kUsed)) return;
        unindex(at);
        entries_[at] = Entry{};
        --used_;
        touch(at);
    }

    std::uint32_t mask_;
    std::vector<Entry> entries_;
    std::size_t used_ = 0;
    std::multimap<std::uint64_t, std::uint32_t> by_to_;  // positive answers by shader VA
    std::uint64_t longest_ = 0;
    std::vector<std::uint32_t> dirty_;
    std::vector<std::uint8_t> dirty_flag_;
};

// What the pass does to one V# half (console_heap_fix.comp translate()): 0 left as it is, 1 answered by a table
// (rebased, written with no records, or left as it is), 2 missed (written with no records).
inline unsigned console_heap_translate_half(std::uint32_t k[4], const ConsoleHeapTables& t,
                                            const ConsoleHeapKeys& keys) {
    if ((k[3] >> 30) || !k[2]) return 0;
    const auto no_records = [k] { k[0] = k[1] = k[2] = 0; };
    const auto rebase = [k](std::uint64_t to) {
        k[0] = std::uint32_t(to);
        k[1] = (k[1] & 0xffff0000u) | std::uint32_t((to >> 32) & 0xffffu);
    };
    const std::uint64_t base = std::uint64_t(k[0]) | (std::uint64_t(k[1] & 0xffffu) << 32);
    if (!base) return no_records(), 1;
    const std::uint32_t stride = (k[1] >> 16) & 0x3fffu;
    const std::uint64_t span = std::uint64_t(k[2]) * (stride ? stride : 1);
    auto it = std::upper_bound(t.ranges.begin(), t.ranges.end(), base,
                               [](std::uint64_t va, const ConsoleHeapRange& r) { return va < r.title_va; });
    if (it != t.ranges.begin()) {
        const ConsoleHeapRange& r = *std::prev(it);
        if (base < r.end && r.checked) {
            if (r.end - base < span) no_records();
            else rebase(r.shader_va + (base - r.title_va));
            return 1;
        }
    }
    std::uint64_t offset = base;
    if (t.signed_offsets && (base & (1ull << 47))) offset |= ~((1ull << 48) - 1);
    if (console_heap_spans_hold(t.backed, offset, span)) return 1;
    const ConsoleHeapKeys::Key key{k[0], k[1], k[2], k[3]};
    std::uint64_t to = 0;
    bool none = false;
    if (keys.lookup(key, to, none)) {
        if (none) no_records();
        else rebase(to);
        return 1;
    }
    no_records();
    if (t.use_ask && !console_heap_spans_hold(t.ask, base, 1)) return 1;
    return 2;
}

// One 32-byte slot as the pass writes it; the bit of each half that missed (1: dwords 0-3, 2: dwords 4-7).
// `zero_first`: the work item's bit 31 (ConsoleHeapGpu::kZeroFirst).
inline unsigned console_heap_translate_slot(std::uint32_t w[8], const ConsoleHeapTables& t,
                                            const ConsoleHeapKeys& keys, bool zero_first = false) {
    if (w[3] >> 30) {
        if (zero_first) w[0] = w[1] = w[2] = w[3] = 0;
        else return 0;
    }
    unsigned missed = 0;
    if (console_heap_translate_half(w, t, keys) == 2) missed |= 1;
    if (console_heap_translate_half(w + 4, t, keys) == 2) missed |= 2;
    return missed;
}

// The pass itself: the key table on the GPU, the range table's versions, the slots to translate per heap, and the
// passes in flight with their miss lists. Every method takes the object's own lock (a leaf: it takes no other).
class ConsoleHeapGpu {
public:
    static constexpr unsigned kMaxHeaps = 8;
    // A queued slot's bit 31: the slot is neither a buffer pair nor a T# the CPU could decode, and the pass writes its
    // dwords 0-3 as zeros. The last queue() of a slot before a pass decides it.
    static constexpr std::uint32_t kZeroFirst = 0x80000000u;
    static constexpr std::uint32_t kMissCapacity = 1u << 16;
    static constexpr std::uint32_t kMissHeader = 4;

    struct Stats {
        std::uint64_t passes = 0, slots = 0, updates = 0, misses = 0, overflows = 0, tables = 0, resets = 0;
        std::uint64_t rescanned = 0;  // slots of rescan() dispatches
        std::uint64_t filtered = 0;   // slots of rescan_filtered() dispatches
        std::uint64_t copies = 0, copy_bytes = 0;  // queue_copy()
        std::uint64_t failed = 0;  // passes whose submission did not reach the queue (their work went back)
    };
    enum TableKind : unsigned { range_table, backed_table, ask_table, change_shader_table, change_title_table, table_count };

    ConsoleHeapGpu(PFN_vkGetDeviceProcAddr get, VkDevice device, const VkPhysicalDeviceMemoryProperties& memory,
                   std::uint32_t log2_keys = 20)
        : get_(get), device_(device), memory_(memory), keys_(log2_keys) {}
    ~ConsoleHeapGpu() {
        std::lock_guard lock(mutex_);
        for (auto& pass : in_flight_) free_buffer(pass.buffer);
        for (Heap& h : heaps_) free_buffer(h.meta);
        for (auto& b : retired_) free_buffer(b);
        for (auto& buffer : pool_) free_buffer(buffer);
        for (Table& t : tables_) {
            for (auto& [version, buffer] : t.old) free_buffer(buffer);
            for (auto& buffer : t.pool) free_buffer(buffer);
            if (t.current) free_buffer(*t.current);
        }
        free_buffer(key_table_);
        if (pipeline_) fn<PFN_vkDestroyPipeline>("vkDestroyPipeline")(device_, pipeline_, nullptr);
        if (layout_) fn<PFN_vkDestroyPipelineLayout>("vkDestroyPipelineLayout")(device_, layout_, nullptr);
    }
    ConsoleHeapGpu(const ConsoleHeapGpu&) = delete;
    ConsoleHeapGpu& operator=(const ConsoleHeapGpu&) = delete;

    // The pipeline and the GPU key table (device-local, zeroed by the first pass). False: the pass cannot run.
    bool init() {
        std::lock_guard lock(mutex_);
        if (pipeline_) return true;
        if (!make_pipeline()) return false;
        const VkDeviceSize bytes = VkDeviceSize(keys_.capacity()) * ConsoleHeapKeys::kEntryWords * 4;
        if (!make_buffer(bytes, false, key_table_)) return false;
        clear_keys_ = true;
        return true;
    }

    // A heap the pass writes: its title bytes (`input`) and the copy shaders read (`copy`), `slots` 32-byte slots.
    // -1 when there is no room.
    // Each heap also gets device-local metadata, 32 bytes a slot (what the pass last made of each V# half), which
    // lets a filtered rescan pass over the slots no change touches without reading their words (rescan_filtered).
    int add_heap(VkDeviceAddress input, VkDeviceAddress copy, std::uint32_t slots) {
        std::lock_guard lock(mutex_);
        for (unsigned i = 0; i < kMaxHeaps; ++i) {
            if (heaps_[i].slots) continue;
            Heap& h = heaps_[i];
            h = Heap{};
            h.input = input;
            h.copy = copy;
            h.slots = slots;
            h.queued.assign(slots, 0);
            if (make_buffer(VkDeviceSize(slots) * 32, false, h.meta)) h.meta_clear = true;
            return int(i);
        }
        return -1;
    }
    void remove_heap(int heap) {
        std::lock_guard lock(mutex_);
        if (heap < 0 || unsigned(heap) >= kMaxHeaps) return;
        if (heaps_[heap].meta.buffer) retired_.push_back(heaps_[heap].meta);  // a pass in flight may still write it
        heaps_[heap] = Heap{};
    }

    void queue(int heap, std::uint32_t slot) {
        std::lock_guard lock(mutex_);
        queue_locked(heap, slot);
    }
    // A buffer copy for the start of the next pass, before the heap translation (LEONIDA_SHADOW_GPU: a slot-133 shadow's
    // pages copied from an import of the title's pages). `keep` (the source import) lives until that pass retires.
    void queue_copy(VkBuffer src, VkDeviceSize src_offset, VkBuffer dst, VkDeviceSize dst_offset, VkDeviceSize bytes,
                    std::shared_ptr<void> keep) {
        std::lock_guard lock(mutex_);
        copies_.push_back({src, dst, {src_offset, dst_offset, bytes}});
        if (keep) keeps_.push_back(std::move(keep));
    }

    // Queued copies into `dst` dropped (its buffer is about to go); copies already recorded stay with their pass.
    void drop_copies_to(VkBuffer dst) {
        std::lock_guard lock(mutex_);
        std::erase_if(copies_, [dst](const Copy& c) { return c.dst == dst; });
    }

    // Every buffer pair of slots [0, slots) translated again by the next pass: one dispatch over the range, no list
    // (the pass reads each slot's words and leaves the slots that hold no pair as they are).
    void rescan(int heap, std::uint32_t slots) {
        std::lock_guard lock(mutex_);
        if (heap < 0 || unsigned(heap) >= kMaxHeaps) return;
        heaps_[heap].rescan = std::max(heaps_[heap].rescan, std::min(slots, heaps_[heap].slots));
    }
    // Like rescan(), but only the slots whose metadata meets a change span: `shader` spans of shader VAs (backings and
    // alias copies that went: a half whose title base or result reaches into one) and `title` spans of title VAs
    // (alias ranges that went, words answered since: a half whose title base is in one). Without metadata, every
    // slot of the range, as rescan().
    void rescan_filtered(int heap, std::uint32_t slots, const ConsoleHeapSpans& shader, const ConsoleHeapSpans& title) {
        std::lock_guard lock(mutex_);
        if (heap < 0 || unsigned(heap) >= kMaxHeaps) return;
        heaps_[heap].rescan_filtered = std::max(heaps_[heap].rescan_filtered, std::min(slots, heaps_[heap].slots));
        change_shader_.insert(change_shader_.end(), shader.begin(), shader.end());
        change_title_.insert(change_title_.end(), title.begin(), title.end());
    }
    template <class It>
    void queue(int heap, It first, It last) {
        std::lock_guard lock(mutex_);
        for (; first != last; ++first) queue_locked(heap, *first);
    }

    // A new range table for the next pass (sorted by title VA, not overlapping).
    void set_ranges(const std::vector<ConsoleHeapRange>& ranges) {
        std::lock_guard lock(mutex_);
        Table& t = tables_[range_table];
        t.words.clear();
        for (const ConsoleHeapRange& r : ranges)
            t.words.insert(t.words.end(), {std::uint32_t(r.title_va), std::uint32_t(r.title_va >> 32),
                                           std::uint32_t(r.end), std::uint32_t(r.end >> 32),
                                           std::uint32_t(r.shader_va), std::uint32_t(r.shader_va >> 32),
                                           r.checked ? 1u : 0u, 0u});
        t.count = std::uint32_t(ranges.size());
        t.dirty = true;
    }
    // New backed or ask spans (merged) for the next pass.
    void set_spans(TableKind kind, const ConsoleHeapSpans& spans) {
        std::lock_guard lock(mutex_);
        spans_locked(kind, spans);
    }
    void spans_locked(TableKind kind, const ConsoleHeapSpans& spans) {
        Table& t = tables_[kind];
        t.words.clear();
        for (const auto& [start, end] : spans)
            t.words.insert(t.words.end(), {std::uint32_t(start), std::uint32_t(start >> 32), std::uint32_t(end),
                                           std::uint32_t(end >> 32)});
        t.count = std::uint32_t(spans.size());
        t.dirty = true;
        if (kind == ask_table) use_ask_ = true;
    }
    void set_signed_offsets(bool on) {
        std::lock_guard lock(mutex_);
        signed_offsets_ = on;
    }
    // Every key entry gone at once (the GPU table is cleared by the next pass), for a table filling up.
    void reset_keys() {
        std::lock_guard lock(mutex_);
        keys_.clear_silent();
        clear_keys_ = true;
        ++stats_.resets;
    }

    template <class F>
    auto with_keys(F&& f) {
        std::lock_guard lock(mutex_);
        return f(keys_);
    }

    bool pending() {
        std::lock_guard lock(mutex_);
        return pending_locked();
    }

    // Records the pass into `cb`, ahead of the submission's lists: the key table updates, then each heap's queued
    // slots, between barriers that order it after earlier work on the queue and before everything after it (and the
    // host's read of the miss list). True when something was recorded; commit() must follow the submit.
    bool record(VkCommandBuffer cb) {
        std::lock_guard lock(mutex_);
        if (!pipeline_ || !pending_locked() || in_flight_.size() >= 64) return false;
        if (clear_keys_) keys_.touch_all();
        std::vector<std::uint32_t> updates;
        keys_.take_updates(updates);
        const std::uint32_t update_count = std::uint32_t(updates.size() / ConsoleHeapKeys::kUpdateWords);
        std::size_t work = 0;
        for (const Heap& h : heaps_) work += h.work.size();
        const VkDeviceSize bytes = VkDeviceSize(kMissHeader + kMissCapacity) * 4 + VkDeviceSize(work) * 4 +
                                   VkDeviceSize(updates.size()) * 4 + 64;
        Pass pass{};
        if (!acquire_buffer(bytes, pass.buffer)) {
            for (std::size_t i = 0; i < updates.size(); i += ConsoleHeapKeys::kUpdateWords) keys_.touch(updates[i]);
            return false;
        }
        // A filtered rescan's change spans, merged, as this pass's change tables.
        bool filtered = false;
        for (const Heap& h : heaps_) filtered |= h.rescan_filtered && !h.rescan;
        ConsoleHeapSpans change_shader, change_title;
        if (filtered) {
            change_shader = change_shader_;
            change_title = change_title_;
            merge_console_heap_spans(change_shader);
            merge_console_heap_spans(change_title);
            spans_locked(change_shader_table, change_shader);
            spans_locked(change_title_table, change_title);
        }
        for (unsigned kind = 0; kind < table_count; ++kind) {
            if (tables_[kind].dirty && !publish(tables_[kind], kind == range_table ? 32 : 16)) {
                pool_.push_back(pass.buffer);
                for (std::size_t i = 0; i < updates.size(); i += ConsoleHeapKeys::kUpdateWords) keys_.touch(updates[i]);
                return false;
            }
            pass.versions[kind] = tables_[kind].version;
        }
        auto* words = static_cast<std::uint32_t*>(pass.buffer.mapped);
        words[0] = 0;
        words[1] = kMissCapacity;
        words[2] = 0;
        words[3] = 0;
        std::size_t at = kMissHeader + kMissCapacity;
        const VkDeviceAddress base = pass.buffer.address;
        const VkDeviceAddress update_address = base + at * 4;
        if (!updates.empty()) std::memcpy(words + at, updates.data(), updates.size() * 4);
        at += updates.size();
        const auto barrier = fn<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier");
        const auto dispatch = fn<PFN_vkCmdDispatch>("vkCmdDispatch");
        const auto push = fn<PFN_vkCmdPushConstants>("vkCmdPushConstants");
        // The copies first (another kind of work in the same prelude), then what reads their destinations may run.
        if (!copies_.empty()) {
            std::sort(copies_.begin(), copies_.end(), [](const Copy& a, const Copy& b) {
                return a.src != b.src ? a.src < b.src : a.dst < b.dst;
            });
            std::vector<VkBufferCopy> regions;
            const auto copy = fn<PFN_vkCmdCopyBuffer>("vkCmdCopyBuffer");
            for (std::size_t i = 0; i < copies_.size();) {
                std::size_t j = i;
                regions.clear();
                while (j < copies_.size() && copies_[j].src == copies_[i].src && copies_[j].dst == copies_[i].dst) {
                    regions.push_back(copies_[j].region);
                    stats_.copy_bytes += copies_[j].region.size;
                    ++j;
                }
                copy(cb, copies_[i].src, copies_[i].dst, std::uint32_t(regions.size()), regions.data());
                stats_.copies += regions.size();
                i = j;
            }
            VkMemoryBarrier copied{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            copied.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            copied.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
            barrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &copied, 0, nullptr, 0,
                    nullptr);
            pass.copies = std::move(copies_);
            pass.keeps = std::move(keeps_);
            copies_.clear();
            keeps_.clear();
        }
        // No barrier against the queue's earlier work: one from ALL_COMMANDS made every submission wait for the one
        // before it to finish.
        // The pass may then rewrite a slot an earlier submission still reads, as the CPU path's memcpy into the copy
        // always could.
        fn<PFN_vkCmdBindPipeline>("vkCmdBindPipeline")(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
        Push p{};
        const auto address = [&](TableKind kind) { return tables_[kind].current ? tables_[kind].current->address : 0; };
        p.ranges = address(range_table);
        p.backed = address(backed_table);
        p.ask = address(ask_table);
        p.keys = key_table_.address;
        p.misses = base;
        p.range_count = tables_[range_table].current ? tables_[range_table].count : 0;
        p.backed_count = tables_[backed_table].current ? tables_[backed_table].count : 0;
        p.ask_count = tables_[ask_table].current ? tables_[ask_table].count : 0;
        p.flags = (signed_offsets_ ? 1u : 0u) | (use_ask_ ? 2u : 0u);
        p.key_mask = keys_.mask();
        if (clear_keys_) {
            fn<PFN_vkCmdFillBuffer>("vkCmdFillBuffer")(cb, key_table_.buffer, 0, VK_WHOLE_SIZE, 0);
            VkMemoryBarrier filled{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            filled.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            filled.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            barrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &filled, 0,
                    nullptr, 0, nullptr);
        }
        // New heaps' metadata starts at zero (no half known).
        bool meta_filled = false;
        for (Heap& h : heaps_)
            if (h.meta_clear && h.meta.buffer) {
                fn<PFN_vkCmdFillBuffer>("vkCmdFillBuffer")(cb, h.meta.buffer, 0, VK_WHOLE_SIZE, 0);
                h.meta_clear = false;
                meta_filled = true;
                pass.filled.push_back(int(&h - heaps_.data()));
            }
        if (meta_filled) {
            VkMemoryBarrier filled{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            filled.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            filled.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            barrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &filled, 0,
                    nullptr, 0, nullptr);
        }
        if (update_count) {
            p.work = update_address;  // MODE_UPDATE's records
            p.count = update_count;
            p.mode = 1;
            push(cb, layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(p), &p);
            dispatch(cb, (update_count + 63) / 64, 1, 1);
            VkMemoryBarrier written{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            written.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            written.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            barrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &written, 0,
                    nullptr, 0, nullptr);
        }
        p.mode = 0;
        p.cshader = filtered ? address(change_shader_table) : 0;
        p.ctitle = filtered ? address(change_title_table) : 0;
        p.cshader_count = filtered && tables_[change_shader_table].current ? tables_[change_shader_table].count : 0;
        p.ctitle_count = filtered && tables_[change_title_table].current ? tables_[change_title_table].count : 0;
        for (unsigned i = 0; i < kMaxHeaps; ++i) {
            Heap& h = heaps_[i];
            const bool full = h.rescan != 0;
            const std::uint32_t slots = full ? std::max(h.rescan, h.rescan_filtered) : h.rescan_filtered;
            if (!slots) continue;
            // Before the listed slots, so a slot both rewritten and rescanned ends with its listed (newer) form.
            p.input_slots = h.input;
            p.copy_slots = h.copy;
            p.meta = h.meta.address;
            p.heap = i;
            const std::uint32_t flags = p.flags;
            p.flags |= 4u;               // FLAG_RANGE
            if (!full) p.flags |= 8u;    // FLAG_FILTER
            for (std::uint32_t first = 0; first < slots; first += 65535u * 64u) {
                const std::uint32_t n = std::min<std::uint32_t>(slots - first, 65535u * 64u);
                p.work = first;
                p.count = n;
                push(cb, layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(p), &p);
                dispatch(cb, (n + 63) / 64, 1, 1);
                (full ? stats_.rescanned : stats_.filtered) += n;
            }
            p.flags = flags;
            pass.rescans.push_back({int(i), slots, !full});
            h.rescan = h.rescan_filtered = 0;
        }
        if (filtered) {
            pass.change_shader = std::move(change_shader);
            pass.change_title = std::move(change_title);
        }
        change_shader_.clear();
        change_title_.clear();
        if (!pass.rescans.empty()) {  // the listed slots' writes land after the rescan's
            VkMemoryBarrier rescanned{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            rescanned.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            rescanned.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            barrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &rescanned,
                    0, nullptr, 0, nullptr);
        }
        for (unsigned i = 0; i < kMaxHeaps; ++i) {
            Heap& h = heaps_[i];
            if (h.work.empty()) continue;
            std::memcpy(words + at, h.work.data(), h.work.size() * 4);
            pass.work.push_back({int(i), std::move(h.work)});
            const auto& list = pass.work.back().second;
            for (const std::uint32_t item : list) h.queued[item & ~kZeroFirst] = 0;
            h.work.clear();
            p.input_slots = h.input;
            p.copy_slots = h.copy;
            p.meta = h.meta.address;
            p.heap = i;
            // At most 65535 groups a dispatch (the guaranteed limit).
            for (std::size_t first = 0; first < list.size(); first += 65535ull * 64) {
                const std::uint32_t n = std::uint32_t(std::min<std::size_t>(list.size() - first, 65535ull * 64));
                p.work = base + (at + first) * 4;
                p.count = n;
                push(cb, layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(p), &p);
                dispatch(cb, (n + 63) / 64, 1, 1);
                stats_.slots += n;
            }
            at += list.size();
        }
        VkMemoryBarrier after{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        after.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        after.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_HOST_READ_BIT;
        barrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_HOST_BIT,
                0, 1, &after, 0, nullptr, 0, nullptr);
        pass.updates = std::move(updates);
        pass.cleared = clear_keys_;
        clear_keys_ = false;
        stats_.updates += update_count;
        ++stats_.passes;
        in_flight_.push_back(std::move(pass));
        return true;
    }

    // After the submit of the command buffer record() wrote into: `ticket` is a retire ticket at or after that
    // submission. Not submitted: its slots and updates are queued again.
    void commit(bool submitted, std::uint64_t ticket) {
        std::lock_guard lock(mutex_);
        for (auto it = in_flight_.rbegin(); it != in_flight_.rend(); ++it) {
            if (it->committed) continue;
            if (submitted) {
                it->committed = true;
                it->ticket = ticket;
                return;
            }
            ++stats_.failed;
            for (const auto& [heap, list] : it->work)
                if (heaps_[heap].slots)
                    for (const std::uint32_t item : list) queue_locked(heap, item);
            for (const auto& r : it->rescans) {
                if (!heaps_[r.heap].slots) continue;
                auto& to = r.filtered ? heaps_[r.heap].rescan_filtered : heaps_[r.heap].rescan;
                to = std::max(to, r.slots);
            }
            change_shader_.insert(change_shader_.end(), it->change_shader.begin(), it->change_shader.end());
            change_title_.insert(change_title_.end(), it->change_title.begin(), it->change_title.end());
            for (const int heap : it->filled)
                if (heaps_[heap].slots) heaps_[heap].meta_clear = true;
            copies_.insert(copies_.end(), it->copies.begin(), it->copies.end());
            keeps_.insert(keeps_.end(), it->keeps.begin(), it->keeps.end());
            for (std::size_t i = 0; i < it->updates.size(); i += ConsoleHeapKeys::kUpdateWords)
                keys_.touch(it->updates[i]);
            if (it->cleared) clear_keys_ = true;
            pool_.push_back(it->buffer);
            in_flight_.erase(std::next(it).base());
            return;
        }
    }

    // The passes done by `completed` (a completed retire ticket): `miss(heap, slot, half)` for each V# half they
    // could not answer, `overflow(heap)` for a heap whose misses ran past the list (-1: every heap).
    template <class Miss, class Overflow>
    void drain(std::uint64_t completed, Miss&& miss, Overflow&& overflow) {
        std::vector<Pass> done;
        {
            std::lock_guard lock(mutex_);
            for (auto it = in_flight_.begin(); it != in_flight_.end();) {
                if (!it->committed || it->ticket > completed) {
                    ++it;
                    continue;
                }
                done.push_back(std::move(*it));
                it = in_flight_.erase(it);
            }
            // Tables no pass in flight still reads go back to their pool.
            for (unsigned kind = 0; kind < table_count; ++kind) {
                Table& t = tables_[kind];
                std::uint64_t oldest = t.version;
                for (const Pass& pass : in_flight_) oldest = std::min(oldest, pass.versions[kind]);
                for (auto it = t.old.begin(); it != t.old.end();) {
                    if (it->first < oldest) {
                        if (t.pool.size() < 8) t.pool.push_back(it->second);
                        else free_buffer(it->second);
                        it = t.old.erase(it);
                    } else {
                        ++it;
                    }
                }
            }
        }
        for (Pass& pass : done) {
            const auto* words = static_cast<const std::uint32_t*>(pass.buffer.mapped);
            const std::uint32_t count = std::min(words[0], kMissCapacity);
            for (std::uint32_t i = 0; i < count; ++i) {
                const std::uint32_t code = words[kMissHeader + i];
                miss(int(code >> 29), (code >> 1) & ((1u << 28) - 1), code & 1u);
            }
            if (words[2]) overflow(-1);
            std::lock_guard lock(mutex_);
            stats_.misses += count;
            stats_.overflows += words[2] ? 1 : 0;
            pool_.push_back(pass.buffer);
        }
    }

    std::size_t in_flight() {
        std::lock_guard lock(mutex_);
        return in_flight_.size();
    }
    // A pass recorded and not yet committed: its command buffer is on its way to the queue.
    bool uncommitted() {
        std::lock_guard lock(mutex_);
        return std::any_of(in_flight_.begin(), in_flight_.end(), [](const Pass& p) { return !p.committed; });
    }
    Stats take_stats() {
        std::lock_guard lock(mutex_);
        return std::exchange(stats_, Stats{});
    }

private:
    struct Buffer {
        VkBuffer buffer = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkDeviceAddress address = 0;
        VkDeviceSize size = 0;
        void* mapped = nullptr;
    };
    struct Heap {
        VkDeviceAddress input = 0, copy = 0;
        std::uint32_t slots = 0;
        std::vector<std::uint32_t> work;
        std::vector<std::uint32_t> queued;  // per slot: its position in `work` + 1, 0 when not queued
        std::uint32_t rescan = 0;           // slots [0, rescan) to translate again (rescan())
        std::uint32_t rescan_filtered = 0;  // the same, filtered by the change spans (rescan_filtered())
        Buffer meta;                        // device-local, 32 bytes a slot (console_heap_fix.comp meta_of)
        bool meta_clear = false;
    };
    struct Copy {
        VkBuffer src, dst;
        VkBufferCopy region;
    };
    struct Pass {
        Buffer buffer;
        std::array<std::uint64_t, table_count> versions{};
        struct Rescan {
            int heap;
            std::uint32_t slots;
            bool filtered;
        };
        std::vector<Rescan> rescans;
        ConsoleHeapSpans change_shader, change_title;
        std::vector<int> filled;  // heaps whose metadata this pass zeroed
        std::vector<Copy> copies;
        std::vector<std::shared_ptr<void>> keeps;
        std::vector<std::pair<int, std::vector<std::uint32_t>>> work;
        std::vector<std::uint32_t> updates;
        bool cleared = false, committed = false;
        std::uint64_t ticket = 0;
    };
    struct Push {
        VkDeviceAddress input_slots, copy_slots, work, ranges, keys, misses, backed, ask, meta, cshader, ctitle;
        std::uint32_t count, range_count, key_mask, mode, heap, backed_count, ask_count, flags, cshader_count,
            ctitle_count;
    };
    static_assert(sizeof(Push) == 128);  // the guaranteed maxPushConstantsSize
    // A table the CPU rebuilds whole (range, backed, ask): each version is its own buffer, kept until no pass in
    // flight reads it, then reused.
    struct Table {
        std::vector<std::uint32_t> words;
        std::uint32_t count = 0;
        bool dirty = false;
        std::unique_ptr<Buffer> current;
        std::uint64_t version = 0;
        std::vector<std::pair<std::uint64_t, Buffer>> old;
        std::vector<Buffer> pool;
    };

    template <typename T>
    T fn(const char* name) const {
        return reinterpret_cast<T>(get_(device_, name));
    }

    void queue_locked(int heap, std::uint32_t item) {
        if (heap < 0 || unsigned(heap) >= kMaxHeaps) return;
        Heap& h = heaps_[heap];
        const std::uint32_t slot = item & ~kZeroFirst;
        if (slot >= h.slots) return;
        if (const std::uint32_t at = h.queued[slot]) {
            h.work[at - 1] = item;
            return;
        }
        h.work.push_back(item);
        h.queued[slot] = std::uint32_t(h.work.size());
    }
    bool pending_locked() const {
        if (keys_.has_updates() || clear_keys_ || !copies_.empty()) return true;
        for (const Heap& h : heaps_)
            if (!h.work.empty() || h.rescan || h.rescan_filtered) return true;
        return false;
    }

    bool publish(Table& t, std::size_t entry_bytes) {
        Buffer table{};
        if (t.count) {
            // A version no pass reads any more, when one is big enough: the tables change at most submissions.
            const VkDeviceSize bytes = VkDeviceSize(t.count) * entry_bytes;
            for (auto it = t.pool.begin(); it != t.pool.end(); ++it)
                if (it->size >= bytes) {
                    table = *it;
                    t.pool.erase(it);
                    break;
                }
            // Device-local and host-visible where there is such memory: the pass's binary searches read it a few
            // dozen times per slot, and from system memory each step was a trip over the bus.
            if (!table.buffer && !make_buffer(std::max<VkDeviceSize>(bytes + bytes / 2, 64 * 1024), true, table, true))
                return false;
            std::memcpy(table.mapped, t.words.data(), t.words.size() * 4);
        }
        if (t.current) t.old.emplace_back(t.version, *t.current);
        t.current.reset();
        if (table.buffer) t.current = std::make_unique<Buffer>(table);
        ++t.version;
        t.dirty = false;
        ++stats_.tables;
        return true;
    }

    bool acquire_buffer(VkDeviceSize bytes, Buffer& out) {
        for (auto it = pool_.begin(); it != pool_.end(); ++it) {
            if (it->size < bytes) continue;
            out = *it;
            pool_.erase(it);
            return true;
        }
        // Room to grow: the next pass of about this size reuses it.
        return make_buffer(std::max<VkDeviceSize>(bytes + bytes / 2, 1 << 20), true, out);
    }

    bool make_buffer(VkDeviceSize bytes, bool host, Buffer& out, bool host_device_local = false) {
        out = {};
        VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        info.size = bytes;
        info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                     VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        if (fn<PFN_vkCreateBuffer>("vkCreateBuffer")(device_, &info, nullptr, &out.buffer) != VK_SUCCESS) return false;
        VkMemoryRequirements requirements{};
        fn<PFN_vkGetBufferMemoryRequirements>("vkGetBufferMemoryRequirements")(device_, out.buffer, &requirements);
        const VkMemoryPropertyFlags required =
            host ? VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT : 0;
        const VkMemoryPropertyFlags preferred = host ? 0 : VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        VkMemoryAllocateFlagsInfo flags{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
        flags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
        VkMemoryAllocateInfo allocate{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, &flags};
        allocate.allocationSize = requirements.size;
        // Host-visible: system memory first (not the ReBAR heap the title's windows need).
        std::vector<std::uint32_t> order;
        if (host && host_device_local) {
            order = memory_type_order(memory_, requirements.memoryTypeBits, required | VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                      0, requirements.size);
            for (const std::uint32_t t : memory_type_order(memory_, requirements.memoryTypeBits, required, 0,
                                                           requirements.size))
                if (std::find(order.begin(), order.end(), t) == order.end()) order.push_back(t);
        } else if (host) {
            order = memory_type_order(memory_, requirements.memoryTypeBits, required | VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
                                      0, requirements.size);
            for (const std::uint32_t t : memory_type_order(memory_, requirements.memoryTypeBits, required, 0,
                                                           requirements.size))
                if (std::find(order.begin(), order.end(), t) == order.end()) order.push_back(t);
        } else {
            order = memory_type_order(memory_, requirements.memoryTypeBits, required, preferred, requirements.size);
        }
        const auto allocate_memory = fn<PFN_vkAllocateMemory>("vkAllocateMemory");
        for (const std::uint32_t type : order) {
            allocate.memoryTypeIndex = type;
            if (allocate_memory(device_, &allocate, nullptr, &out.memory) == VK_SUCCESS) break;
            out.memory = VK_NULL_HANDLE;
        }
        if (!out.memory ||
            fn<PFN_vkBindBufferMemory>("vkBindBufferMemory")(device_, out.buffer, out.memory, 0) != VK_SUCCESS ||
            (host && fn<PFN_vkMapMemory>("vkMapMemory")(device_, out.memory, 0, VK_WHOLE_SIZE, 0, &out.mapped) !=
                         VK_SUCCESS)) {
            free_buffer(out);
            return false;
        }
        VkBufferDeviceAddressInfo address{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
        address.buffer = out.buffer;
        out.address = fn<PFN_vkGetBufferDeviceAddress>("vkGetBufferDeviceAddress")(device_, &address);
        out.size = bytes;
        if (!out.address) {
            free_buffer(out);
            return false;
        }
        return true;
    }
    void free_buffer(Buffer& b) {
        if (b.buffer) fn<PFN_vkDestroyBuffer>("vkDestroyBuffer")(device_, b.buffer, nullptr);
        if (b.memory) fn<PFN_vkFreeMemory>("vkFreeMemory")(device_, b.memory, nullptr);  // unmaps
        b = {};
    }

    bool make_pipeline() {
        for (const char* name :
             {"vkCreateBuffer", "vkDestroyBuffer", "vkGetBufferMemoryRequirements", "vkAllocateMemory", "vkFreeMemory",
              "vkBindBufferMemory", "vkMapMemory", "vkGetBufferDeviceAddress", "vkCreateShaderModule",
              "vkDestroyShaderModule", "vkCreatePipelineLayout", "vkDestroyPipelineLayout", "vkCreateComputePipelines",
              "vkDestroyPipeline", "vkCmdPipelineBarrier", "vkCmdBindPipeline", "vkCmdPushConstants", "vkCmdDispatch",
              "vkCmdFillBuffer", "vkCmdCopyBuffer"})
            if (!get_(device_, name)) return false;
        VkShaderModule module = VK_NULL_HANDLE;
        VkShaderModuleCreateInfo shader{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        shader.codeSize = sizeof(console_heap_fix_spirv);
        shader.pCode = console_heap_fix_spirv;
        const aftermath::ShaderName aftermath_name("(runtime) console_heap_fix");
        bool ok =
            fn<PFN_vkCreateShaderModule>("vkCreateShaderModule")(device_, &shader, nullptr, &module) == VK_SUCCESS;
        const VkPushConstantRange range{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(Push)};
        VkPipelineLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        layout_info.pushConstantRangeCount = 1;
        layout_info.pPushConstantRanges = &range;
        VkPipelineLayout layout = VK_NULL_HANDLE;
        VkPipeline pipeline = VK_NULL_HANDLE;
        ok = ok && fn<PFN_vkCreatePipelineLayout>("vkCreatePipelineLayout")(device_, &layout_info, nullptr, &layout) ==
                       VK_SUCCESS;
        VkComputePipelineCreateInfo info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        info.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        info.stage.module = module;
        info.stage.pName = "main";
        info.layout = layout;
        ok = ok && fn<PFN_vkCreateComputePipelines>("vkCreateComputePipelines")(device_, VK_NULL_HANDLE, 1, &info,
                                                                                nullptr, &pipeline) == VK_SUCCESS;
        if (module) fn<PFN_vkDestroyShaderModule>("vkDestroyShaderModule")(device_, module, nullptr);
        if (!ok) {
            if (pipeline) fn<PFN_vkDestroyPipeline>("vkDestroyPipeline")(device_, pipeline, nullptr);
            if (layout) fn<PFN_vkDestroyPipelineLayout>("vkDestroyPipelineLayout")(device_, layout, nullptr);
            return false;
        }
        layout_ = layout;
        pipeline_ = pipeline;
        return true;
    }

    PFN_vkGetDeviceProcAddr get_;
    VkDevice device_;
    VkPhysicalDeviceMemoryProperties memory_;
    std::mutex mutex_;
    VkPipelineLayout layout_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
    ConsoleHeapKeys keys_;
    Buffer key_table_;
    bool clear_keys_ = false;
    std::array<Heap, kMaxHeaps> heaps_{};
    std::array<Table, table_count> tables_{};
    bool use_ask_ = false, signed_offsets_ = false;
    std::vector<Pass> in_flight_;
    std::vector<Buffer> pool_;
    std::vector<Copy> copies_;
    std::vector<std::shared_ptr<void>> keeps_;
    ConsoleHeapSpans change_shader_, change_title_;  // pending rescan_filtered() spans
    std::vector<Buffer> retired_;                    // removed heaps' metadata
    Stats stats_;
};

}  // namespace leonida
