// src/shadow_keep.h: a shadow copy keeps a GPU write until the title's bytes under it change. No Windows headers, no
// GPU, no game data: the "title" and its "copy" are arrays here, and the values are made up.
#include "../src/shadow_keep.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <utility>
#include <vector>

namespace {

int failures = 0;
#define EXPECT(condition)                                                        \
    do {                                                                         \
        if (!(condition)) {                                                      \
            std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #condition);    \
            ++failures;                                                          \
        }                                                                        \
    } while (0)

using leonida::shadow_keep::kPiece;

// A title range and its copy, as ShadowPages has them: the copy of the title byte at address a is copy[a - base].
struct Pages {
    static constexpr std::uintptr_t base = 0x10000;  // kPiece-aligned, like a tracker's first page
    std::vector<std::uint8_t> title, copy;
    std::vector<std::uint64_t> sums;
    std::vector<std::pair<std::uintptr_t, std::uintptr_t>> copied;  // the ranges the last refresh copied

    explicit Pages(std::size_t bytes) : title(bytes), copy(bytes), sums(bytes / kPiece) {
        for (std::size_t i = 0; i < bytes; ++i) title[i] = std::uint8_t(i * 7 + 3);
    }
    // A refresh of [va, va + n): what ShadowPages::copy_changed does for one page.
    std::size_t refresh(std::uintptr_t va, std::size_t n) {
        copied.clear();
        std::size_t bytes = 0;
        leonida::shadow_keep::copy_changed(
            title.data() + (va - base), va, n, sums.data(), base,
            [&](std::uintptr_t from, std::uintptr_t to, const std::uint8_t* source) {
                std::memcpy(copy.data() + (from - base), source, to - from);
                copied.emplace_back(from, to);
                bytes += to - from;
            });
        return bytes;
    }
    std::size_t refresh() { return refresh(base, title.size()); }
    bool same(std::size_t from, std::size_t to) const {
        return !std::memcmp(title.data() + from, copy.data() + from, to - from);
    }
};

void unchanged_title_keeps_gpu_write() {
    Pages p(4096);
    EXPECT(p.refresh() == 4096);  // nothing known yet: all of it
    EXPECT(p.same(0, 4096));
    EXPECT(p.refresh() == 0);  // nothing changed
    // A shader stores 16 bytes in the copy; the title's bytes there stay as they were.
    const std::uint8_t gpu[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    std::memcpy(p.copy.data() + 0x340, gpu, sizeof(gpu));
    EXPECT(p.refresh() == 0);
    EXPECT(!std::memcmp(p.copy.data() + 0x340, gpu, sizeof(gpu)));  // kept: the old code overwrote it here
    EXPECT(p.same(0, 0x340) && p.same(0x350, 4096));
}

void changed_title_piece_is_copied() {
    Pages p(4096);
    p.refresh();
    const std::uint8_t gpu[16] = {0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA,
                                  0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA};
    std::memcpy(p.copy.data() + 0x340, gpu, sizeof(gpu));  // GPU write in piece 3
    std::memcpy(p.copy.data() + 0x800, gpu, sizeof(gpu));  // and in piece 8
    p.title[0x810] ^= 0x5A;                                 // the title writes piece 8
    EXPECT(p.refresh() == kPiece);
    EXPECT(p.copied.size() == 1 && p.copied[0].first == Pages::base + 0x800 &&
           p.copied[0].second == Pages::base + 0x900);
    EXPECT(p.same(0x800, 0x900));                                     // the title's new bytes win there
    EXPECT(!std::memcmp(p.copy.data() + 0x340, gpu, sizeof(gpu)));  // the other GPU write stays
    EXPECT(p.refresh() == 0);
}

void partial_first_and_last_pieces() {
    Pages p(4096);
    // A target that starts and ends inside pieces: [base + 100, base + 1000).
    const std::uintptr_t va = Pages::base + 100;
    const std::size_t n = 900;
    EXPECT(p.refresh(va, n) == n);
    EXPECT(p.copied.size() == 4);
    EXPECT(p.copied.front().first == va && p.copied.front().second == Pages::base + kPiece);
    EXPECT(p.copied.back().first == Pages::base + 3 * kPiece && p.copied.back().second == va + n);
    EXPECT(p.same(100, 1000));
    EXPECT(p.copy[99] == 0 && p.copy[1000] == 0);  // nothing outside the range
    EXPECT(p.refresh(va, n) == 0);
    p.title[999] ^= 1;  // last byte of the last, partial piece
    EXPECT(p.refresh(va, n) == 1000 - 3 * kPiece);
    p.title[100] ^= 1;  // first byte of the first, partial piece
    EXPECT(p.refresh(va, n) == kPiece - 100);
    EXPECT(p.same(100, 1000));
}

void forgotten_sums_copy_once() {
    Pages p(4096);
    p.refresh();
    // The pages were not committed for a refresh: the copies were zeroed and their sums forgotten.
    std::memset(p.copy.data() + 0x400, 0, 0x200);
    leonida::shadow_keep::forget(p.sums.data(), Pages::base, Pages::base + 0x400, Pages::base + 0x600);
    EXPECT(p.refresh() == 0x200);  // the same title bytes back: copied once all the same
    EXPECT(p.same(0, 4096));
    EXPECT(p.refresh() == 0);
    // A partial range forgets every piece it touches.
    leonida::shadow_keep::forget(p.sums.data(), Pages::base, Pages::base + 0x410, Pages::base + 0x501);
    EXPECT(p.refresh() == 0x200);
}

void sums_tell_bytes_apart() {
    std::uint8_t a[kPiece] = {}, b[kPiece] = {};
    EXPECT(leonida::shadow_keep::sum_of(a, kPiece) != 0);
    EXPECT(leonida::shadow_keep::sum_of(a, kPiece) == leonida::shadow_keep::sum_of(b, kPiece));
    for (std::size_t i = 0; i < kPiece; ++i) {
        b[i] = 1;
        EXPECT(leonida::shadow_keep::sum_of(a, kPiece) != leonida::shadow_keep::sum_of(b, kPiece));
        b[i] = 0;
    }
    EXPECT(leonida::shadow_keep::sum_of(a, 16) != leonida::shadow_keep::sum_of(a, 17));  // the length counts
}

}  // namespace

int main() {
    unchanged_title_keeps_gpu_write();
    changed_title_piece_is_copied();
    partial_first_and_last_pieces();
    forgotten_sums_copy_once();
    sums_tell_bytes_apart();
    if (failures) {
        std::fprintf(stderr, "shadow_keep: %d check(s) failed\n", failures);
        return 1;
    }
    std::puts("shadow_keep: OK");
    return 0;
}
