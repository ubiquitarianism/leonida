// What a shadow copy of the title's pages keeps of the GPU's writes (LEONIDA_SHADOW_KEEP, shims/
// d3d12_vulkan_device.cpp ShadowPages). Some pages are copied at every refresh without a sign that the title wrote
// them (mapped at two addresses, unguarded, reopened with the 64 KiB block around a written page, write-watched or
// pinned): copied whole, their unchanged bytes overwrote what a shader had stored in the copy since. Here the title's
// bytes are compared, per 256-byte piece of the title's address space, with a 64-bit sum of those copied last, and
// only a piece that differs is copied. Portable: no Windows headers, no Vulkan.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace leonida::shadow_keep {

inline constexpr std::size_t kPiece = 256;

// The sum of `n` bytes, never 0 (0 is "not known": the piece is copied at its next look).
// Four independent lanes over 32 bytes a step, so the multiplies do not wait on each other: about the speed of a
// copy (one chained lane read about 3.6 GB/s on an RTX 5080 machine's CPU, a copy about 18 GB/s).
inline std::uint64_t sum_of(const std::uint8_t* bytes, std::size_t n) {
    constexpr std::uint64_t k = 0xff51afd7ed558ccdull;
    const auto rotl = [](std::uint64_t x, int r) { return (x << r) | (x >> (64 - r)); };
    std::uint64_t a0 = 0x9e3779b97f4a7c15ull ^ n, a1 = 0xc2b2ae3d27d4eb4full, a2 = 0x165667b19e3779f9ull,
                  a3 = 0x27d4eb2f165667c5ull;
    std::size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        std::uint64_t v[4];
        std::memcpy(v, bytes + i, 32);
        a0 = rotl(a0 ^ v[0], 31) * k;
        a1 = rotl(a1 ^ v[1], 31) * k;
        a2 = rotl(a2 ^ v[2], 31) * k;
        a3 = rotl(a3 ^ v[3], 31) * k;
    }
    for (; i < n; i += 8) {
        std::uint64_t v = 0;
        std::memcpy(&v, bytes + i, n - i < 8 ? n - i : 8);
        a0 = rotl(a0 ^ v, 31) * k;
    }
    std::uint64_t h = a0 ^ rotl(a1, 17) ^ rotl(a2, 29) ^ rotl(a3, 47);
    h ^= h >> 33;
    h *= 0xc4ceb9fe1a85ec53ull;
    h ^= h >> 29;
    return h | 1;  // 0: not known
}

// `bytes` holds the title's bytes [va, va + n) as read now. sums[k] is the sum of the piece at base + k * kPiece
// (base kPiece-aligned) as last copied. Each piece, cut at kPiece boundaries of the address (the first and last may be
// partial), whose bytes differ from its sum takes the new sum and is handed to copy(from, to, source).
template <class Copy>
void copy_changed(const std::uint8_t* bytes, std::uintptr_t va, std::size_t n, std::uint64_t* sums,
                  std::uintptr_t base, Copy&& copy) {
    for (std::size_t off = 0; off < n;) {
        const std::size_t piece = std::min<std::size_t>(n - off, kPiece - ((va + off) & (kPiece - 1)));
        std::uint64_t& known = sums[(va + off - base) / kPiece];
        const std::uint64_t sum = sum_of(bytes + off, piece);
        if (sum != known) {
            known = sum;
            copy(va + off, va + off + piece, bytes + off);
        }
        off += piece;
    }
}

// The pieces over [from, to) are not known any more (the copies were changed another way, as zeroed): each is copied
// at its next look.
inline void forget(std::uint64_t* sums, std::uintptr_t base, std::uintptr_t from, std::uintptr_t to) {
    if (from >= to) return;
    std::fill(sums + (from - base) / kPiece, sums + (to - base + kPiece - 1) / kPiece, std::uint64_t(0));
}

}  // namespace leonida::shadow_keep
