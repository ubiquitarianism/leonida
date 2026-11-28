// leonida::ArenaBlocks (src/arena_blocks.h): the GPU address arena's free-block map, its first fit at an alignment,
// its coalescing, and a walk that survives the map being cut under it. No GPU, no device.
//
// The crash this covers is an access violation in arena_allocate: a walk over the map stepped into a node that had
// been freed under it (the nil flag of a null child at 0x19). A map freed or rebalanced by a write outside the
// arena's lock does that to a walk on another thread; the walk here also re-reads the map every step instead of
// keeping an iterator, so a cut made under it (a nested release or allocation, as a test can do) is survived too.
#include "../src/arena_blocks.h"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <thread>
#include <vector>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

namespace {

constexpr std::uint64_t kReserved = 0x10000;  // the zero page and the quad index list
constexpr std::uint64_t kSize = 1 << 20;
constexpr std::uint64_t kAlign = 0x1000;

using Blocks = leonida::ArenaBlocks;

// The device's keep-out: no title range in these tests, so nothing is excluded.
std::uint64_t no_keep_out(std::uint64_t, std::uint64_t) { return 0; }

bool one_block_of(const Blocks& blocks, std::uint64_t offset, std::uint64_t size) {
    const auto all = blocks.blocks();
    return all.size() == 1 && all.front().offset == offset && all.front().size == size;
}

// First fit, alignment, coalescing and exact placement.
int basics() {
    Blocks b;
    b.reset(kReserved, kSize);
    EXPECT(b.count() == 1 && b.free_bytes() == kSize - kReserved);
    EXPECT(b.check() == Blocks::Fault::none && b.consistent(0));
    std::uint64_t scratch = 0;
    EXPECT(!b.allocate(0, kAlign, no_keep_out, scratch));
    EXPECT(!b.allocate(kSize, kAlign, no_keep_out, scratch));  // larger than the arena
    std::uint64_t a = 0, c = 0, d = 0;
    EXPECT(b.allocate(0x1000, kAlign, no_keep_out, a) && a == kReserved);
    EXPECT(b.free_bytes() == kSize - kReserved - 0x1000 && b.consistent(0x1000));
    EXPECT(b.allocate(0x3000, kAlign, no_keep_out, c) && c == kReserved + 0x1000);
    // An unaligned free start is rounded up: the request lands on the next boundary.
    EXPECT(b.release(a, 0x1000) == Blocks::Fault::none);
    EXPECT(b.allocate(0x800, kAlign, no_keep_out, d) && d == kReserved);
    // Placement at a console VA only where a free block holds the range exactly.
    EXPECT(!b.allocate_at(c, 0x800));  // a live range
    EXPECT(!b.allocate_at(kReserved, 0x2000));
    EXPECT(b.release(c, 0x3000) == Blocks::Fault::none);
    EXPECT(b.allocate_at(kReserved + 0x1000, 0x1000));
    EXPECT(b.check() == Blocks::Fault::none);
    EXPECT(b.consistent(0x800 + 0x1000));
    // Giving everything back merges the arena into the one block it started as.
    EXPECT(b.release(kReserved, 0x800) == Blocks::Fault::none);
    EXPECT(b.release(kReserved + 0x1000, 0x1000) == Blocks::Fault::none);
    EXPECT(b.check() == Blocks::Fault::none && b.consistent(0));
    EXPECT(one_block_of(b, kReserved, kSize - kReserved));
    return 0;
}

// A keep-out the device passes for the title's own memory: a request that starts inside a title range is moved past
// it, and a free block the range covers entirely is skipped for the next one (first fit moves on).
int keep_out() {
    Blocks b;
    b.reset(kReserved, kSize);
    std::uint64_t a = 0, c = 0, d = 0;
    EXPECT(b.allocate(0x1000, kAlign, no_keep_out, a));  // kReserved
    EXPECT(b.allocate(0x1000, kAlign, no_keep_out, c));  // kReserved + 0x1000
    EXPECT(b.allocate(0x1000, kAlign, no_keep_out, d));  // kReserved + 0x2000
    EXPECT(b.release(a, 0x1000) == Blocks::Fault::none);
    EXPECT(b.release(c, 0x1000) == Blocks::Fault::none);  // one free block [kReserved, kReserved + 0x2000)
    // A title range over the block's first 0x800 bytes pushes the start to the next 0x1000 boundary, where the
    // request still fits the same block.
    const std::uint64_t title_lo = kReserved, title_hi = kReserved + 0x800;
    const auto keep_out_first = [&](std::uint64_t lo, std::uint64_t hi) -> std::uint64_t {
        return lo < title_hi && hi > title_lo ? title_hi : 0;
    };
    std::uint64_t out = 0;
    EXPECT(b.allocate(0x1000, kAlign, keep_out_first, out) && out == kReserved + 0x1000);
    // A title range over the whole free block leaves it unusable: first fit moves on to the tail block.
    EXPECT(b.release(out, 0x1000) == Blocks::Fault::none);
    const auto keep_out_all = [&](std::uint64_t lo, std::uint64_t hi) -> std::uint64_t {
        return lo < kReserved + 0x2000 && hi > kReserved ? kReserved + 0x2000 : 0;
    };
    EXPECT(b.allocate(0x1000, kAlign, keep_out_all, out) && out >= kReserved + 0x3000);
    EXPECT(b.check() == Blocks::Fault::none && b.consistent(0x2000));
    return 0;
}

// The crash's shape: the walk stands on a free block, and a call it makes releases live memory that borders that
// block, so the block is merged away -- the very node the walk stands on is erased and freed. The walk re-reads the
// map every step, so it carries on from the new map instead of stepping into the freed node.
int cut_under_the_walk() {
    Blocks b;
    b.reset(kReserved, kSize);
    std::uint64_t a = 0, c = 0, d = 0, e = 0;
    EXPECT(b.allocate(0x1000, kAlign, no_keep_out, a));
    EXPECT(b.allocate(0x1000, kAlign, no_keep_out, c));
    EXPECT(b.allocate(0x1000, kAlign, no_keep_out, d));
    EXPECT(b.allocate(0x1000, kAlign, no_keep_out, e));
    EXPECT(b.release(a, 0x1000) == Blocks::Fault::none);
    EXPECT(b.release(c, 0x1000) == Blocks::Fault::none);
    EXPECT(b.release(d, 0x1000) == Blocks::Fault::none);
    EXPECT(b.count() == 2);  // the coalesced gap run and the tail: the walk visits both
    int visits = 0;
    bool released_under_the_walk = false;
    const auto keep_out = [&](std::uint64_t, std::uint64_t) -> std::uint64_t {
        ++visits;
        if (visits > 8 || released_under_the_walk) return 0;  // once, to bound the nesting
        // `e` sits between the free run the walk is in and the tail: giving it back merges both blocks and erases
        // the node being walked.
        released_under_the_walk = b.release(e, 0x1000) == Blocks::Fault::none;
        return 0;
    };
    std::uint64_t out = 0;
    const bool found = b.allocate(0x1000, kAlign, keep_out, out);
    EXPECT(visits > 0 && released_under_the_walk);
    EXPECT(b.check() == Blocks::Fault::none);
    // Found or refused, the map is a partition of the arena and the result is a range of it.
    if (found) {
        EXPECT(out >= kReserved && out + 0x1000 <= kSize);
        for (const auto& block : b.blocks()) EXPECT(out + 0x1000 <= block.offset || block.offset + block.size <= out);
    }
    EXPECT(b.consistent(found ? 0x1000 : 0));
    return 0;
}

// A nested allocation takes the very block the walk stands on: the walk re-reads the map and carries on instead of
// cutting a node it has already left behind.
int taken_under_the_walk() {
    Blocks b;
    b.reset(kReserved, kSize);
    std::uint64_t a = 0, c = 0;
    EXPECT(b.allocate(0x1000, kAlign, no_keep_out, a));
    EXPECT(b.allocate(0x1000, kAlign, no_keep_out, c));
    EXPECT(b.release(a, 0x1000) == Blocks::Fault::none);
    bool nested = false;
    std::uint64_t taken = 0;
    const auto keep_out = [&](std::uint64_t, std::uint64_t) -> std::uint64_t {
        if (!nested) {
            nested = true;
            (void)b.allocate(0x1000, kAlign, no_keep_out, taken);  // the first free block, where the walk is
        }
        return 0;
    };
    std::uint64_t out = 0;
    EXPECT(b.allocate(0x1000, kAlign, keep_out, out));
    EXPECT(nested && out >= kReserved && out + 0x1000 <= kSize);
    EXPECT(taken != 0 && (out != taken || out + 0x1000 <= taken || taken + 0x1000 <= out));
    EXPECT(b.check() == Blocks::Fault::none && b.consistent(0x3000));
    return 0;
}

// A search the map is cut under on every step gives up at its limit and says so, rather than looking like a full
// arena.
int walk_interrupted() {
    Blocks b;
    b.reset(kReserved, kSize);
    std::uint64_t live = 0;
    EXPECT(b.allocate(0x1000, kAlign, no_keep_out, live));
    int cuts = 0;
    const auto keep_out = [&](std::uint64_t, std::uint64_t) -> std::uint64_t {
        ++cuts;
        // Give the live block back and take it again: the map is cut under every search of the walk.
        (void)b.release(live, 0x1000);
        (void)b.allocate(0x1000, kAlign, no_keep_out, live);
        return 0;
    };
    std::uint64_t out = 0;
    EXPECT(!b.allocate(0x1000, kAlign, keep_out, out));
    EXPECT(b.interrupted() && cuts >= Blocks::kAttempts);
    EXPECT(b.check() == Blocks::Fault::none && b.consistent(0x1000));
    // A search with nothing cutting it clears the report.
    EXPECT(b.allocate(0x1000, kAlign, no_keep_out, out) && !b.interrupted());
    return 0;
}

// Releases that are wrong are reported and cannot make two live ranges overlap.
int bad_releases() {
    Blocks b;
    b.reset(kReserved, kSize);
    std::uint64_t a = 0;
    EXPECT(b.allocate(0x2000, kAlign, no_keep_out, a));
    EXPECT(b.release(a, 0x1000) == Blocks::Fault::none);
    EXPECT(b.count() == 2 && b.blocks().front().offset == a && b.blocks().front().size == 0x1000);
    // The same bytes again: the union is inserted once, so they are free once.
    const std::uint64_t free_before = b.free_bytes();
    EXPECT(b.release(a, 0x1000) == Blocks::Fault::double_free);
    EXPECT(b.free_bytes() == free_before);
    EXPECT(b.count() == 2 && b.check() == Blocks::Fault::none);
    // A range the arena never had: refused, and the map is untouched.
    EXPECT(b.release(kSize, 0x1000) == Blocks::Fault::outside_arena);
    EXPECT(b.release(kReserved / 2, 0x1000) == Blocks::Fault::outside_arena);
    EXPECT(b.count() == 2);
    // The rest, then everything, merges into one block.
    EXPECT(b.release(a + 0x1000, 0x1000) == Blocks::Fault::none);
    EXPECT(b.count() == 1 && b.free_bytes() == kSize - kReserved);
    EXPECT(b.check() == Blocks::Fault::none && b.consistent(0));
    return 0;
}

// The diagnostic reader in d3d12_compute_input_probe.h: is an address inside a free block, and where does the next
// one start.
int reader() {
    Blocks b;
    b.reset(kReserved, kSize);
    std::uint64_t a = 0, c = 0, offset = 0, size = 0, next = 0;
    EXPECT(b.allocate(0x1000, kAlign, no_keep_out, a));
    EXPECT(b.allocate(0x1000, kAlign, no_keep_out, c));
    EXPECT(!b.free_at(a, offset, size, next) && next == c + 0x1000);
    EXPECT(b.free_at(c + 0x1000, offset, size, next) && offset == c + 0x1000 && next == kSize);
    return 0;
}

// Many threads, one lock -- the device's shape: every call to the block map under one mutex, no call re-entering it.
int threads() {
    Blocks b;
    b.reset(kReserved, kSize);
    std::mutex mutex;  // the device's arena mutex
    std::atomic<bool> failed{false};
    std::vector<std::thread> workers;
    constexpr int kThreads = 4, kRounds = 2000;
    for (int t = 0; t < kThreads; ++t) {
        workers.emplace_back([&b, &mutex, &failed, t] {
            std::uint64_t live[8] = {};
            for (int round = 0; round < kRounds && !failed.load(); ++round) {
                const int slot = (t * 3 + round) % 8;
                std::lock_guard lock(mutex);
                if (live[slot]) {
                    b.release(live[slot], 0x400);
                    live[slot] = 0;
                } else {
                    std::uint64_t out = 0;
                    if (b.allocate(0x400, 64, no_keep_out, out)) live[slot] = out;
                }
                if (b.check() != Blocks::Fault::none) failed.store(true);
            }
            std::lock_guard lock(mutex);
            for (std::uint64_t& block : live) {
                if (!block) continue;
                b.release(block, 0x400);
                block = 0;
            }
        });
    }
    for (auto& worker : workers) worker.join();
    EXPECT(!failed.load());
    EXPECT(b.check() == Blocks::Fault::none && b.consistent(0));
    EXPECT(one_block_of(b, kReserved, kSize - kReserved));
    return 0;
}

// High alignment or keep-out ends near UINT64_MAX, non-advancing keep-out callbacks, and inverted reset bounds must
// never wrap around or spin forever.
int overflow_and_bounds() {
    Blocks b;
    b.reset(kReserved, kSize);
    std::uint64_t out = 0;

    // Alignment that would overflow value + alignment - 1: refused cleanly without wrapping to 0.
    EXPECT(!b.allocate(0x1000, UINT64_MAX - 0x100, no_keep_out, out));

    // A keep_out returning an end near UINT64_MAX where (keep_out_end + alignment - 1) wraps to 0 must skip the block
    // instead of looping forever.
    int calls = 0;
    const auto wrapping_keep_out = [&](std::uint64_t, std::uint64_t) -> std::uint64_t {
        ++calls;
        return UINT64_MAX - 0x100;
    };
    EXPECT(!b.allocate(0x1000, kAlign, wrapping_keep_out, out) && calls == 1);

    // A non-advancing keep_out (returning <= start) must not spin forever.
    calls = 0;
    const auto stagnant_keep_out = [&](std::uint64_t lo, std::uint64_t) -> std::uint64_t {
        ++calls;
        return lo;
    };
    EXPECT(!b.allocate(0x1000, kAlign, stagnant_keep_out, out) && calls == 1);

    // consistent() must not overflow on huge live_bytes or underflow when reserved > size.
    EXPECT(!b.consistent(UINT64_MAX));
    b.reset(kSize, kReserved);
    EXPECT(b.free_bytes() == 0 && b.count() == 0);
    EXPECT(b.check() == Blocks::Fault::none);
    EXPECT(b.consistent(0) && !b.consistent(1));
    EXPECT(!b.allocate(0x1000, kAlign, no_keep_out, out));
    return 0;
}

}  // namespace

int main() {
    if (int r = basics()) return r;
    if (int r = keep_out()) return r;
    if (int r = cut_under_the_walk()) return r;
    if (int r = taken_under_the_walk()) return r;
    if (int r = walk_interrupted()) return r;
    if (int r = bad_releases()) return r;
    if (int r = reader()) return r;
    if (int r = threads()) return r;
    if (int r = overflow_and_bounds()) return r;
    std::fprintf(stderr, "arena_blocks: ok\n");
    return 0;
}
