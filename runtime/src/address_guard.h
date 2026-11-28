// The address guard's page table (docs/d3d12_vulkan.md, "Address guard"): what the runtime backs around heap_base,
// for shaders translated with `tools/rdna2spirv.py --address-guard` (LEONIDA_ADDRESS_GUARD=1). Header-only, no
// Vulkan: the device writes the words this computes into a mapped buffer. tools/address_guard.py is the Python
// reference of the same format and of the check a guarded access makes.
//
// Layout. A header at arena offset kGuardHeader (64 bytes, AddressGuardHeader), then two sinks: refused loads read
// kGuardZero, refused stores and atomics write kGuardDump, each at the access's offset within its 4 KiB page. The
// page table is its own buffer; the header holds its address minus heap_base. Table words: kGuardLevel1 level-1
// words indexed by bits 47..32 of the 48-bit heap offset (one per 4 GiB chunk), naming the chunk's level-2 block;
// then the blocks, kGuardBlockWords words each, one bit per 4 KiB page (bits 31..12). Block 0 is empty (nothing
// backed), block 1 full (everything passes: a chunk the table has no room for), blocks 2.. are allocated.
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

namespace leonida::address_guard {

// Shared with tools/rdna2spirv.py (GUARD_*) and tools/address_guard.py; tests/semantic/test_address_guard.py reads
// these lines.
constexpr std::uint64_t kGuardHeader = 0x70000;
constexpr std::uint64_t kGuardZero = 0x71000;
constexpr std::uint64_t kGuardDump = 0x78000;
constexpr std::uint64_t kGuardEnd = 0x80000;
constexpr std::uint32_t kGuardLevel1 = 1u << 16;
constexpr std::uint32_t kGuardBlockWords = 1u << 15;
constexpr char kGuardMagicText[] = "LGUARD01";
constexpr std::uint64_t kMask48 = (1ull << 48) - 1;
constexpr std::uint32_t kEmptyBlock = 0, kFullBlock = 1, kFirstBlock = 2;

constexpr std::uint64_t magic() {
    std::uint64_t v = 0;
    for (int i = 7; i >= 0; --i) v = v << 8 | std::uint8_t(kGuardMagicText[i]);
    return v;
}

struct AddressGuardHeader {
    std::uint64_t magic;         // magic(): the shaders check only when it is there
    std::uint64_t limit;         // the arena's size: an access below it passes without a lookup
    std::uint64_t table;         // the page table's device address minus heap_base (mod 2^64)
    std::uint32_t refused;       // the shaders add 1 per refused access
    std::uint32_t last_module;   // the last refusing module's id: FNV-1a of its file stem (shaders write it; diagnostics)
    std::uint64_t last_refused;  // the last refused access's heap offset (written by the shaders, not atomically)
    std::uint8_t pad[24];
};
static_assert(sizeof(AddressGuardHeader) == 64);
static_assert(kGuardHeader + sizeof(AddressGuardHeader) <= kGuardZero && kGuardZero < kGuardDump &&
              kGuardDump < kGuardEnd);

// The table's contents and the words that changed. Blocks are given to chunks for good: a chunk that loses its last
// backing keeps its (now empty) block, so a submission still in flight never reads a block that came to mean
// another chunk. When the blocks run out, a new chunk gets the full block (logged by the caller through
// overflowed()): its accesses then pass unchecked, as without the guard.
class AddressGuardTable {
public:
    explicit AddressGuardTable(std::uint32_t blocks = 62) : blocks_(blocks) {
        words_.assign(word_count(), 0);
        std::fill_n(words_.begin() + kGuardLevel1 + std::size_t(kFullBlock) * kGuardBlockWords, kGuardBlockWords,
                    0xFFFFFFFFu);
    }

    std::size_t word_count() const { return kGuardLevel1 + std::size_t(kFirstBlock + blocks_) * kGuardBlockWords; }
    std::size_t bytes() const { return word_count() * sizeof(std::uint32_t); }
    // The whole table, for the first upload.
    const std::vector<std::uint32_t>& words() const { return words_; }
    std::uint32_t blocks_used() const { return used_; }
    std::uint32_t overflowed() const { return overflowed_; }

    // Make the table say exactly `spans` (heap offset [start, end), 48-bit; a span may wrap through 2^48 for a
    // negative offset). A page any byte of a span touches is backed: the GPU maps memory in whole pages. Each word
    // that changes is passed to `write(index, value)`, level-2 words before level-1 ones, so a reader that sees a
    // chunk's new block also sees its bits. Returns the number of words written.
    template <class Write>
    std::size_t assign(const std::vector<std::pair<std::uint64_t, std::uint64_t>>& spans, Write&& write) {
        next_.assign(std::size_t(used_) * kGuardBlockWords, 0);
        std::vector<std::uint32_t> level1(words_.begin(), words_.begin() + kGuardLevel1);
        for (const auto& [start, end] : spans) {
            if (end == start) continue;
            const std::uint64_t first = (start & kMask48) >> 12;
            const std::uint64_t last = ((end - 1) & kMask48) >> 12;
            if (first <= last) {
                mark(level1, first, last);
            } else {  // wraps through 2^48: two runs
                mark(level1, first, (kMask48 >> 12));
                mark(level1, 0, last);
            }
        }
        std::size_t written = 0;
        const std::size_t base = kGuardLevel1 + std::size_t(kFirstBlock) * kGuardBlockWords;
        for (std::size_t i = 0; i < next_.size(); ++i) {
            if (words_[base + i] == next_[i]) continue;
            words_[base + i] = next_[i];
            write(base + i, next_[i]);
            ++written;
        }
        for (std::uint32_t i = 0; i < kGuardLevel1; ++i) {
            if (words_[i] == level1[i]) continue;
            words_[i] = level1[i];
            write(std::size_t(i), level1[i]);
            ++written;
        }
        return written;
    }

    // The table's bit for the page of heap offset `offset`.
    bool page_backed(std::uint64_t offset) const {
        offset &= kMask48;
        const std::uint32_t block = words_[(offset >> 32) & (kGuardLevel1 - 1)];
        const std::uint64_t page = (offset >> 12) & 0xFFFFF;
        return (words_[kGuardLevel1 + std::size_t(block) * kGuardBlockWords + (page >> 5)] >> (page & 31)) & 1;
    }

    // What a guarded access of `extent` bytes at heap offset `offset` does (the shaders' %guard_fn): passes when
    // its first and last byte are below `limit` or on backed pages. `offset` is sign-extended from 48 bits.
    bool passes(std::uint64_t offset, std::uint64_t extent, std::uint64_t limit) const {
        if (offset & (1ull << 47)) offset |= ~kMask48;
        else offset &= kMask48;
        const std::uint64_t end = offset + extent - 1;
        if (offset < limit && end < limit) return true;
        return page_backed(offset) && page_backed(end);
    }

private:
    // Set the bits of pages [first, last] (48-bit page numbers), giving their chunks blocks as needed.
    void mark(std::vector<std::uint32_t>& level1, std::uint64_t first, std::uint64_t last) {
        for (std::uint64_t chunk = first >> 20; chunk <= last >> 20; ++chunk) {
            std::uint32_t& block = level1[chunk];
            if (block == kEmptyBlock) {
                if (used_ < blocks_) {
                    block = kFirstBlock + used_++;
                    next_.resize(std::size_t(used_) * kGuardBlockWords, 0);
                } else {
                    block = kFullBlock;
                    ++overflowed_;
                }
            }
            if (block < kFirstBlock) continue;
            std::uint32_t* bits = next_.data() + std::size_t(block - kFirstBlock) * kGuardBlockWords;
            const std::uint64_t lo = std::max(first, chunk << 20) & 0xFFFFF;
            const std::uint64_t hi = std::min(last, (chunk << 20) | 0xFFFFF) & 0xFFFFF;
            for (std::uint64_t page = lo; page <= hi;) {
                if (!(page & 31) && page + 31 <= hi) {
                    bits[page >> 5] = 0xFFFFFFFFu;
                    page += 32;
                } else {
                    bits[page >> 5] |= 1u << (page & 31);
                    ++page;
                }
            }
        }
    }

    std::uint32_t blocks_;
    std::uint32_t used_ = 0;
    std::uint32_t overflowed_ = 0;
    std::vector<std::uint32_t> words_;  // what the GPU's copy holds
    std::vector<std::uint32_t> next_;   // the allocated blocks being built by assign()
};

}  // namespace leonida::address_guard
