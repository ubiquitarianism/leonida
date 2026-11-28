#pragma once
// The GPU address arena's free-block map (docs/d3d12_vulkan.md section 3.3).
//
// One map of free blocks, offset -> size, first fit at a requested alignment. Every access is made under the
// device's arena mutex, and a walk holds no iterator across a call: each step reads a block's offset and size and
// lets its iterator go, so a block cut under the walk cannot leave the next step reading a freed node. A freed
// node reads as garbage (seen once in arena_allocate: an access violation reading the nil flag of a null map child
// at 0x19 -- the freed block's first two pointers reused by the heap's free list, while the rest of the node still
// looked like a node). A map access outside the lock does that to a concurrent walk: the arena's first block used
// to be inserted by create_arena with no lock, and an insert rebalances the tree, writing child links under a
// walker on another thread. It is locked now, and the walk is immune either way.
//
// The class keeps no lock of its own (the device's arena mutex is the lock), so a test can drive it directly.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <utility>
#include <vector>

namespace leonida {

class ArenaBlocks {
public:
    struct Block {
        std::uint64_t offset;
        std::uint64_t size;
    };

    // What a release found wrong with the range it was given: a range the arena never had, or bytes that are free
    // already (a double free, which would otherwise double-count them and let two live ranges overlap).
    enum class Fault { none, outside_arena, double_free, bytes_mismatch, broken_map };
    static constexpr const char* kFaultNames[] = {"none", "range outside the arena", "double free",
                                                  "free bytes disagree with the live bytes", "block map broken"};

    // [reserved, size) is the one free block: the arena's first bytes hold the zero page and the quad index list
    // (kArenaReserved in the device), which no resource may have.
    void reset(std::uint64_t reserved, std::uint64_t size) {
        blocks_.clear();
        reserved_ = reserved;
        size_ = size;
        free_bytes_ = size > reserved ? size - reserved : 0;
        if (free_bytes_) blocks_.emplace(reserved, free_bytes_);
        cuts_ = 0;
        fault_ = Fault::none;
        faults_ = 0;
        interrupted_ = false;
    }

    // First fit of `bytes` at `alignment`: `keep_out(lo, hi)` returns the end of a range of the arena no
    // allocation may cover that overlaps [lo, hi), or 0 when there is none (the device passes the title's own
    // memory through src/title_va_alias.h). False when no free block holds the request.
    template<typename KeepOut>
    bool allocate(std::uint64_t bytes, std::uint64_t alignment, KeepOut&& keep_out, std::uint64_t& out) {
        if (!bytes || !alignment || bytes > size_) return false;
        interrupted_ = false;
        // Blocks in offset order. A step carries only a block's offset and size; the iterator is gone before
        // keep_out() runs and the next block is looked up from the map again, so a cut made under the walk
        // (cuts() changed) is seen, and the search starts over on the map as it is now.
        for (int attempt = 0; attempt < kAttempts; ++attempt) {
            const std::uint64_t cuts = cuts_;
            std::uint64_t cursor = reserved_;
            for (;;) {
                const auto next = blocks_.lower_bound(cursor);
                if (next == blocks_.end()) return false;
                const std::uint64_t block = next->first;
                const std::uint64_t block_size = next->second;
                const std::uint64_t end = block + block_size;
                std::uint64_t start = 0;
                if (align_up(block, alignment, start)) {
                    while (start < end && bytes <= end - start) {
                        const std::uint64_t keep_out_end = keep_out(start, start + bytes);
                        if (!keep_out_end) break;
                        if (keep_out_end <= start || !align_up(keep_out_end, alignment, start)) {
                            start = end;
                            break;
                        }
                    }
                } else {
                    start = end;
                }
                if (cuts_ != cuts) break;  // the map moved under the walk: search the new one
                if (start < end && bytes <= end - start) {
                    const auto target = blocks_.find(block);  // nothing held between steps
                    if (target == blocks_.end() || target->second != block_size) break;
                    cut(target, start, bytes);
                    out = start;
                    return true;
                }
                cursor = end;
            }
            if (cuts_ == cuts) return false;  // nothing moved: a genuine refusal
        }
        interrupted_ = true;  // cut under the search kAttempts times: reported by the device
        return false;
    }

    // [offset, offset + bytes) exactly, inside one free block (a console descriptor heap whose address shaders
    // already use). False when any of it is taken or past the arena.
    bool allocate_at(std::uint64_t offset, std::uint64_t bytes) {
        if (!bytes || offset < reserved_ || offset > size_ || bytes > size_ - offset) return false;
        auto it = blocks_.upper_bound(offset);
        if (it == blocks_.begin()) return false;
        --it;
        if (offset + bytes > it->first + it->second) return false;
        cut(it, offset, bytes);
        return true;
    }

    // Gives [offset, offset + bytes) back and merges it with the blocks it touches. A range that is partly free
    // already is merged as its union and reported as a double free, so the accounting cannot count a byte twice.
    Fault release(std::uint64_t offset, std::uint64_t bytes) {
        if (!bytes) return Fault::none;
        if (offset < reserved_ || offset > size_ || bytes > size_ - offset) return note_fault(Fault::outside_arena);
        const std::uint64_t released_end = offset + bytes;
        std::uint64_t begin = offset;
        std::uint64_t end = released_end;
        bool already_free = false;
        auto it = blocks_.lower_bound(offset);
        if (it != blocks_.begin()) {
            const auto previous = std::prev(it);
            const std::uint64_t previous_end = previous->first + previous->second;
            if (previous_end >= offset) {  // the released bytes start inside a free block, or right after it
                if (previous_end > offset) already_free = true;
                begin = previous->first;
                end = std::max(end, previous_end);
                free_bytes_ -= previous->second;
                it = blocks_.erase(previous);
            }
        }
        while (it != blocks_.end() && it->first <= end) {
            if (it->first < released_end) already_free = true;
            end = std::max(end, it->first + it->second);
            free_bytes_ -= it->second;
            it = blocks_.erase(it);
        }
        blocks_.emplace(begin, end - begin);
        free_bytes_ += end - begin;
        ++cuts_;
        return note_fault(already_free ? Fault::double_free : Fault::none);
    }

    // The arena's bytes: the size it was created with and the first block no resource may have.
    std::uint64_t size() const { return size_; }
    std::uint64_t reserved() const { return reserved_; }
    std::uint64_t free_bytes() const { return free_bytes_; }
    std::size_t count() const { return blocks_.size(); }

    // Every free block in offset order: a report, or a diagnostic that prints what is free.
    std::vector<Block> blocks() const {
        std::vector<Block> out;
        out.reserve(blocks_.size());
        for (const auto& [offset, size] : blocks_) out.push_back({offset, size});
        return out;
    }

    // The free block that holds `offset` (if one does) and the offset of the next block after it: what the compute
    // input probe reads to tell a free range from a live one without walking the map itself
    // (d3d12_compute_input_probe.h backing()).
    bool free_at(std::uint64_t offset, std::uint64_t& block_offset, std::uint64_t& block_size,
                 std::uint64_t& next_offset) const {
        auto it = blocks_.upper_bound(offset);
        bool inside = false;
        if (it != blocks_.begin()) {
            const auto previous = std::prev(it);
            if (offset >= previous->first && offset < previous->first + previous->second) {
                inside = true;
                block_offset = previous->first;
                block_size = previous->second;
            }
        }
        next_offset = it == blocks_.end() ? size_ : it->first;
        return inside;
    }

    // The largest free block, for the high-water report (D3D-55).
    Block largest() const {
        Block best{0, 0};
        for (const auto& [offset, size] : blocks_)
            if (size > best.size) best = {offset, size};
        return best;
    }

    // The block map is a partition of the arena's free bytes: sorted, inside the arena, no overlap, and summing to
    // free_bytes(). Walked for a report, and by the device when its cheap check (free_bytes + live == size -
    // reserved) fails or LEONIDA_ARENA_CHECK=1 is set.
    Fault check() const {
        std::uint64_t sum = 0, last_end = reserved_;
        for (const auto& [offset, size] : blocks_) {
            if (!size || offset < last_end || offset > size_ || size > size_ - offset) return Fault::broken_map;
            sum += size;
            last_end = offset + size;
        }
        return sum == free_bytes_ ? Fault::none : Fault::bytes_mismatch;
    }

    // Whether the map's free bytes and the device's live bytes add up to the arena: a byte released twice, or a
    // released range that was never allocated, breaks this. O(1), so the device checks it on every call.
    bool consistent(std::uint64_t live_bytes) const {
        const std::uint64_t total = size_ > reserved_ ? size_ - reserved_ : 0;
        return live_bytes <= total && free_bytes_ == total - live_bytes;
    }

    // Faults reported so far, and the last one this call reported.
    std::uint64_t faults() const { return faults_; }
    Fault last_fault() const { return fault_; }
    // Cuts made so far: how often a block was split, taken whole or merged. A cut changes it under a walk.
    std::uint64_t cuts() const { return cuts_; }
    // The last allocate gave up because the map was cut under its search kAttempts times; the request may have been
    // refusable anyway, so the device reports it rather than counting it as a fault.
    bool interrupted() const { return interrupted_; }
    static constexpr int kAttempts = 8;

private:
    static bool align_up(std::uint64_t value, std::uint64_t alignment, std::uint64_t& out) {
        const std::uint64_t rem = value % alignment;
        if (!rem) {
            out = value;
            return true;
        }
        const std::uint64_t add = alignment - rem;
        if (value > std::numeric_limits<std::uint64_t>::max() - add) return false;
        out = value + add;
        return true;
    }

    Fault note_fault(Fault fault) {
        fault_ = fault;
        if (fault != Fault::none) ++faults_;
        return fault;
    }

    using Map = std::map<std::uint64_t, std::uint64_t>;
    void cut(Map::iterator it, std::uint64_t start, std::uint64_t bytes) {
        const std::uint64_t offset = it->first;
        const std::uint64_t end = offset + it->second;
        blocks_.erase(it);
        free_bytes_ -= end - offset;
        if (start > offset) {
            blocks_.emplace(offset, start - offset);
            free_bytes_ += start - offset;
        }
        if (start + bytes < end) {
            blocks_.emplace(start + bytes, end - (start + bytes));
            free_bytes_ += end - (start + bytes);
        }
        ++cuts_;
    }

    Map blocks_;  // offset -> size
    std::uint64_t reserved_ = 0, size_ = 0, free_bytes_ = 0, cuts_ = 0, faults_ = 0;
    Fault fault_ = Fault::none;
    bool interrupted_ = false;
};

}  // namespace leonida
