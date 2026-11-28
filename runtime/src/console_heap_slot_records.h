#pragma once
// What each buffer V# of a console resource heap's GPU copy resolved to (docs/d3d12_vulkan.md section 3.4), kept in
// host RAM so that a change of the address space re-fixes exactly the slots it touches.
//
// A slot of the copy is the title's descriptor with its V# bases rewritten (DeviceState::fix_console_slot). What a
// base became depends on things that change after the slot was written: a buffer or window released, an alias range
// removed, a geometry copy retired, or (for a base nothing backed yet) a buffer, alias or title memory added. Without
// a record of which slot resolved into which range, a change could only be applied by walking every live slot. Here
// each V# half (the buffer V# and the UAV counter V# of a slot) records:
//   - base and span: the bytes the title's V# names, [base, base + span) (the base is the V#'s 48 bits);
//   - target: the device address its GPU copy names, [target, target + span); for a half that resolved to nothing,
//     the device address of a copy that was found but refused (not backed yet), or 0;
//   - state: resolved, waiting (written with no base and no records until something backs it) or none.
// Two indexes find the halves a change touches: by base ("source") and by target. Each holds a half once, under the
// power-of-two class of its span and the granule of its start (a granule is at least 64 KiB and at least the class),
// so a range [lo, hi) is answered by visiting, per class in use, the granules from lo - 2^class to hi: the cost is the
// halves near the range, never the heap. An entry is checked against the record when visited (a half that moved
// leaves a stale entry behind; a bucket drops its stale entries whenever it has doubled since it was last cleaned).
// A bucket counts the halves it holds and goes when the last one leaves it, so the buckets held are those of the live
// halves: without that, every granule a session ever wrote a V# into kept its bucket until the heap was destroyed, and
// the main heap lives as long as the session (many of its V#s are data the title keeps in a slot, with a new base
// whenever the data changes).
// Portable, standard library only: console_heap_slot_records_test exercises it without a GPU.
#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <unordered_map>
#include <vector>

namespace leonida {

// One change since the last submission, as a range; hi is exclusive and greater than lo.
struct SlotChange {
    enum Kind : std::uint8_t {
        target_gone,   // device addresses: a backing released, a copy's block destroyed, an alias range's copy gone
        source_gone,   // V# bases: an alias range removed or replaced, or a copy that answered for it gone
        target_added,  // device addresses: a backing added (a refused alias target may be backed now)
        source_added,  // V# bases: an alias range or title memory added, or a backing added at those arena offsets
    };
    Kind kind;
    std::uint64_t lo, hi;
};

class ConsoleHeapSlotRecords {
public:
    enum State : std::uint8_t { none = 0, resolved = 1, waiting = 2 };

    void reset(std::size_t slots) {
        slots_ = slots;
        // calloc: the pages of slots never written are never touched (the retail heap uses a tenth of its slots).
        halves_.reset(slots ? static_cast<Half*>(std::calloc(slots * 2, sizeof(Half))) : nullptr);
        if (slots && !halves_) slots_ = 0;
        for (auto& index : indexes_) index = Index{};
        recorded_ = waiting_ = 0;
        looked_at_ = 0;
    }
    std::size_t slots() const { return slots_; }
    std::size_t recorded() const { return recorded_; }  // halves in state resolved or waiting
    std::size_t waiting_count() const { return waiting_; }  // halves in state waiting
    std::uint64_t looked_at() const { return looked_at_; }  // index entries visited by touched(), all calls
    // The buckets both indexes hold, counted over their maps: a diagnostic (the heap log line and
    // console_heap_slot_records_test, which bounds it by the live halves).
    std::size_t buckets() const {
        std::size_t n = 0;
        for (const auto& index : indexes_)
            for (const auto& buckets : index.classes) n += buckets.size();
        return n;
    }

    // Half `half` (0: the buffer V#, 1: the UAV counter V#) of `slot` now names [base, base + span) and resolved as
    // `state` says, its copy at `target` (see above). span 0 counts as 1.
    void set(std::size_t slot, unsigned half, State state, std::uint64_t base, std::uint64_t span,
             std::uint64_t target) {
        if (slot >= slots_ || half > 1) return;
        if (state == none) base = span = target = 0;
        if (!span) span = 1;
        if (span > kMaxSpan) span = kMaxSpan;
        const std::size_t key = slot * 2 + half;
        Half& h = halves_[key];
        const Half old = h;
        count(old.state(), -1);
        h.base = base;
        h.target = target;
        h.span_state = span | (std::uint64_t(state) << 56);
        count(state, +1);
        for (const int which : {kSource, kTarget}) {
            unsigned was_c = 0, is_c = 0;
            std::uint64_t was_granule = 0, is_granule = 0;
            const bool was = held(which, old, was_c, was_granule), is = held(which, h, is_c, is_granule);
            // Unchanged key: the entry already there stands for it.
            if (was && is && was_c == is_c && was_granule == is_granule) continue;
            if (was) leave(which, was_c, was_granule);  // the entry left behind is stale now
            if (is) push(which, is_c, is_granule, std::uint32_t(key));
        }
    }

    void clear(std::size_t slot) {
        set(slot, 0, none, 0, 0, 0);
        set(slot, 1, none, 0, 0, 0);
    }
    void clear_range(std::size_t first, std::size_t end) {
        end = (std::min)(end, slots_);
        for (std::size_t slot = first; slot < end; ++slot)
            for (unsigned half = 0; half < 2; ++half)
                if (halves_[slot * 2 + half].state() != none) set(slot, half, none, 0, 0, 0);
    }

    State state(std::size_t slot, unsigned half) const {
        return slot < slots_ && half < 2 ? halves_[slot * 2 + half].state() : none;
    }
    std::uint64_t target(std::size_t slot, unsigned half) const {
        return slot < slots_ && half < 2 ? halves_[slot * 2 + half].target : 0;
    }

    // The slots `changes` touch, each once, ascending, appended to `out` (cleared first):
    //   target_gone: halves whose [target, target + span) overlaps the range (whatever their state);
    //   source_gone: halves whose [base, base + span) overlaps it;
    //   target_added, source_added: the same, waiting halves only (a resolved one stays valid until something goes).
    void touched(const SlotChange* changes, std::size_t count, std::vector<std::uint32_t>& out) {
        out.clear();
        for (std::size_t i = 0; i < count; ++i) {
            const SlotChange& change = changes[i];
            if (change.hi <= change.lo) continue;
            const bool target = change.kind == SlotChange::target_gone || change.kind == SlotChange::target_added;
            const bool waiting_only = change.kind == SlotChange::target_added || change.kind == SlotChange::source_added;
            find(target ? kTarget : kSource, change.lo, change.hi, waiting_only, out);
        }
        std::sort(out.begin(), out.end());
        out.erase(std::unique(out.begin(), out.end()), out.end());
    }

private:
    static constexpr int kSource = 0, kTarget = 1;
    static constexpr std::uint64_t kMaxSpan = (std::uint64_t(1) << 56) - 1;
    static constexpr unsigned kMinGranule = 16;  // 64 KiB
    static constexpr unsigned kClasses = 57;

    struct Half {
        std::uint64_t base, target, span_state;
        std::uint64_t span() const { return span_state & kMaxSpan; }
        State state() const { return State(span_state >> 56); }
    };
    struct Free {
        void operator()(Half* p) const { std::free(p); }
    };
    struct Bucket {
        std::vector<std::uint32_t> keys;
        std::size_t clean = 0;  // its size when it was last cleaned
        std::size_t live = 0;   // the halves held here now: each has a current entry in keys
    };
    struct Index {
        std::array<std::unordered_map<std::uint64_t, Bucket>, kClasses> classes;
        std::uint64_t used = 0;  // a bit per class that ever held an entry
    };

    // The smallest c with span <= 2^c.
    static unsigned span_class(std::uint64_t span) {
        return span <= 1 ? 0u : unsigned(64 - std::countl_zero(span - 1));
    }
    static unsigned granule_shift(unsigned c) { return c < kMinGranule ? kMinGranule : c; }

    void count(State state, int delta) {
        if (state != none) recorded_ += std::size_t(std::ptrdiff_t(delta));
        if (state == waiting) waiting_ += std::size_t(std::ptrdiff_t(delta));
    }

    // Whether index `which` holds a half recorded as `h`, and if so under which class and granule.
    static bool held(int which, const Half& h, unsigned& c, std::uint64_t& granule) {
        if (h.state() == none || (which == kTarget && !h.target)) return false;
        c = span_class(h.span());
        granule = (which == kSource ? h.base : h.target) >> granule_shift(c);
        return true;
    }

    // Whether index `which` still holds `key` under (c, granule).
    bool current(int which, std::uint32_t key, unsigned c, std::uint64_t granule) const {
        unsigned hc = 0;
        std::uint64_t hg = 0;
        return held(which, halves_[key], hc, hg) && hc == c && hg == granule;
    }

    // A half no longer held under (c, granule): its entry there is stale, and with the last live half the bucket
    // goes, its stale entries with it (a lookup skips them, so it answers the same without them).
    void leave(int which, unsigned c, std::uint64_t granule) {
        auto& buckets = indexes_[which].classes[c];
        const auto it = buckets.find(granule);
        if (it != buckets.end() && !--it->second.live) buckets.erase(it);
    }

    void push(int which, unsigned c, std::uint64_t granule, std::uint32_t key) {
        Index& index = indexes_[which];
        index.used |= std::uint64_t(1) << c;
        Bucket& bucket = index.classes[c][granule];
        ++bucket.live;
        bucket.keys.push_back(key);
        if (bucket.keys.size() < 32 || bucket.keys.size() < 2 * bucket.clean) return;
        auto& keys = bucket.keys;
        keys.erase(std::remove_if(keys.begin(), keys.end(),
                                  [&](std::uint32_t k) { return !current(which, k, c, granule); }),
                   keys.end());
        std::sort(keys.begin(), keys.end());
        keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
        bucket.clean = keys.size();
    }

    void find(int which, std::uint64_t lo, std::uint64_t hi, bool waiting_only, std::vector<std::uint32_t>& out) {
        Index& index = indexes_[which];
        for (std::uint64_t used = index.used; used; used &= used - 1) {
            const unsigned c = unsigned(std::countr_zero(used));
            auto& buckets = index.classes[c];
            if (buckets.empty()) continue;
            const unsigned shift = granule_shift(c);
            // A half of this class overlaps [lo, hi) only if it starts in (lo - 2^c, hi).
            const std::uint64_t reach = std::uint64_t(1) << c;
            const std::uint64_t first = (lo >= reach ? lo - reach + 1 : 0) >> shift;
            const std::uint64_t last = (hi - 1) >> shift;
            const auto visit = [&](std::uint64_t granule, const Bucket& bucket) {
                for (const std::uint32_t key : bucket.keys) {
                    ++looked_at_;
                    if (!current(which, key, c, granule)) continue;
                    const Half& h = halves_[key];
                    if (waiting_only && h.state() != waiting) continue;
                    const std::uint64_t start = which == kSource ? h.base : h.target;
                    const std::uint64_t span = h.span();
                    const std::uint64_t end = start + span < start ? UINT64_MAX : start + span;
                    if (start < hi && end > lo) out.push_back(key >> 1);
                }
            };
            if (last - first >= buckets.size()) {
                for (const auto& [granule, bucket] : buckets)
                    if (granule >= first && granule <= last) visit(granule, bucket);
            } else {
                for (std::uint64_t granule = first;; ++granule) {
                    const auto it = buckets.find(granule);
                    if (it != buckets.end()) visit(granule, it->second);
                    if (granule == last) break;
                }
            }
        }
    }

    std::size_t slots_ = 0;
    std::unique_ptr<Half[], Free> halves_;
    std::array<Index, 2> indexes_;
    std::size_t recorded_ = 0, waiting_ = 0;
    std::uint64_t looked_at_ = 0;
};

}  // namespace leonida
