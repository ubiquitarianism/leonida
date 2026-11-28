// leonida::address_guard::AddressGuardTable (src/address_guard.h): the address guard's page table. Pages any byte
// of a span touches are backed, a chunk keeps its level-2 block for good, a released span's bits go, negative
// offsets (a buffer below the arena) and spans across 4 GiB chunks work, only changed words are written, level-2
// before level-1, and a chunk past the table's room gets the full block. No GPU, no device.
#include "../src/address_guard.h"

#include <cstdint>
#include <cstdio>
#include <utility>
#include <vector>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

namespace {

using leonida::address_guard::AddressGuardTable;
using Spans = std::vector<std::pair<std::uint64_t, std::uint64_t>>;
namespace ag = leonida::address_guard;

constexpr std::uint64_t kLimit = 0x80000;  // the "arena"
constexpr std::uint64_t kGiB = 1ull << 30;

// A copy of the GPU's table, written only through assign()'s callback, and the order of the writes.
struct Gpu {
    std::vector<std::uint32_t> words;
    std::vector<std::size_t> order;
    explicit Gpu(const AddressGuardTable& t) : words(t.words()) {}
    auto writer() {
        return [this](std::size_t i, std::uint32_t v) {
            words[i] = v;
            order.push_back(i);
        };
    }
};

int basics() {
    AddressGuardTable t(4);
    Gpu gpu(t);
    EXPECT(t.word_count() == ag::kGuardLevel1 + 6u * ag::kGuardBlockWords);
    EXPECT(!t.passes(0x90000, 4, kLimit));
    EXPECT(t.passes(0x7fffc, 4, kLimit));      // below the limit: no lookup
    EXPECT(!t.passes(0x7fffc, 8, kLimit));     // its last byte is past it, on an unbacked page

    const Spans spans{{0x90000, 0x90001}, {5 * 4 * kGiB + 0x1000, 5 * 4 * kGiB + 0x3000}};
    const std::size_t written = t.assign(spans, gpu.writer());
    EXPECT(written > 0);
    EXPECT(gpu.words == t.words());
    EXPECT(t.blocks_used() == 2);
    EXPECT(t.passes(0x90000, 4, kLimit));
    EXPECT(t.passes(0x90ffc, 4, kLimit));      // one byte of the span marks the whole page
    EXPECT(!t.passes(0x90ffc, 8, kLimit));     // the last byte is on the next page
    EXPECT(t.passes(5 * 4 * kGiB + 0x2ff0, 16, kLimit));
    EXPECT(!t.passes(5 * 4 * kGiB + 0x3000, 4, kLimit));
    EXPECT(!t.passes(6 * 4 * kGiB, 4, kLimit));

    // Level-2 words are written before level-1 ones.
    bool seen_level1 = false;
    for (const std::size_t i : gpu.order) {
        if (i < ag::kGuardLevel1) seen_level1 = true;
        else EXPECT(!seen_level1);
    }

    // The same spans again: nothing to write.
    gpu.order.clear();
    EXPECT(t.assign(spans, gpu.writer()) == 0);

    // A release: its bits go, its chunk keeps its block.
    const std::uint32_t chunk5 = t.words()[5];
    EXPECT(chunk5 >= ag::kFirstBlock);
    EXPECT(t.assign(Spans{{0x90000, 0x91000}}, gpu.writer()) > 0);
    EXPECT(gpu.words == t.words());
    EXPECT(!t.passes(5 * 4 * kGiB + 0x2ff0, 4, kLimit));
    EXPECT(t.words()[5] == chunk5);
    EXPECT(t.passes(0x90000, 4, kLimit));
    return 0;
}

int negative_and_wide() {
    AddressGuardTable t(4);
    Gpu gpu(t);
    // A buffer 8 KiB below heap_base (LEONIDA_IMPORT_BELOW_ARENA=1), as the 48-bit offset a V# holds.
    const std::uint64_t below = (0ull - 0x2000) & ag::kMask48;
    // One that runs from below heap_base into the arena's first page: wraps through 2^48.
    const std::uint64_t wrap = (0ull - 0x1000) & ag::kMask48;
    // A 6 GiB buffer across two chunks: whole words of bits.
    const std::uint64_t wide = 8 * kGiB + 0x10000;
    t.assign(Spans{{below, below + 0x1000}, {wrap, 0x1000}, {wide, wide + 6 * kGiB}}, gpu.writer());
    EXPECT(gpu.words == t.words());
    EXPECT(t.passes(below, 4, kLimit));
    EXPECT(t.passes(0ull - 0x2000, 4, kLimit));   // the same offset sign-extended, as the shaders pass it
    EXPECT(t.passes(0ull - 4, 8, kLimit));        // across zero: both pages backed
    EXPECT(!t.passes(0ull - 0x3000, 4, kLimit));
    EXPECT(t.passes(wide, 4, kLimit));
    EXPECT(t.passes(wide + 4 * kGiB, 4, kLimit));
    EXPECT(t.passes(wide + 6 * kGiB - 4, 4, kLimit));
    EXPECT(!t.passes(wide + 6 * kGiB, 4, kLimit));
    EXPECT(!t.passes(wide - 4, 8, kLimit));
    EXPECT(t.blocks_used() == 4);                 // chunks 0xffff, 0 (the wrap), 2 and 3
    return 0;
}

int out_of_room() {
    AddressGuardTable t(1);
    Gpu gpu(t);
    t.assign(Spans{{4 * kGiB, 4 * kGiB + 1}, {8 * kGiB, 8 * kGiB + 1}}, gpu.writer());
    EXPECT(t.blocks_used() == 1);
    EXPECT(t.overflowed() == 1);
    EXPECT(t.passes(4 * kGiB, 4, kLimit));
    EXPECT(!t.passes(4 * kGiB + 0x1000, 4, kLimit));
    EXPECT(t.passes(8 * kGiB + 0x5000, 4, kLimit));  // the full block: all of that chunk passes
    EXPECT(t.words()[2] == ag::kFullBlock);
    return 0;
}

}  // namespace

int main() {
    // "LGUARD01" little-endian, as tools/address_guard.py packs it and the shaders compare it.
    static_assert(ag::magic() == 0x313044524155474Cull);
    if (int r = basics()) return r;
    if (int r = negative_and_wide()) return r;
    if (int r = out_of_room()) return r;
    std::puts("address_guard_test: ok");
    return 0;
}
