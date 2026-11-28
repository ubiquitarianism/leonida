// Texture layouts of the console GPU (RDNA2, GFX10.3): a texture's size and alignment, where each mip and array slice
// starts, the address of each element, and copies between tiled memory and linear rows. Used by the xg_xs.dll shim
// (runtime/shims/xg_xs.cpp), whose texture computer answers the title's layout and copy calls from it.
//
// Ported from Marvisak/VIctory (GPL-3.0-or-later; translated from Rust to C++ and changed for Leonida),
// crates/victory-formats/src/texture.rs at commit d2a4bae1f, which follows
// AMD's address library (addrlib) as shipped in Mesa under src/amd/addrlib: block dimensions, mip placement and mip
// tails from gfx10addrlib.cpp and addrlib2.cpp (ComputeSurfaceInfoMacroTiled, ComputeThinBlockDimension,
// GetMipTailDim, GetMaxNumMipsInTail, HwlComputeSurfaceInfoLinear, ComputeSurfaceAddrFromCoordMacroTiled), and the
// 256-byte micro tile patterns in kMicro from gfx10SwizzlePattern.h (the RB+ tables, as on RDNA2). addrlib carries
// this notice:
//
//   Copyright (C) 2007-2026 Advanced Micro Devices, Inc. All rights reserved.
//   SPDX-License-Identifier: MIT
//
//   Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
//   documentation files (the "Software"), to deal in the Software without restriction, including without limitation
//   the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and
//   to permit persons to whom the Software is furnished to do so, subject to the following conditions: the above
//   copyright notice and this permission notice shall be included in all copies or substantial portions of the
//   Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED.
//
// Units: an element is a texel, or a 4x4 block of a block-compressed format. Mips of one array slice (or one depth
// slab of a thick 3D texture) are stored together, smallest first: the mip tail (the small mips sharing one block)
// at offset 0, mip 0 last.
#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <vector>

namespace leonida::console_texture {

// A swizzle mode, by its hardware number (addrlib's ADDR_SW_* order).
enum : uint32_t {
    kLinear = 0,
    kSw256bS = 1,
    kSw256bD = 2,
    kSw4kbS = 5,
    kSw4kbD = 6,
    kSw64kbS = 9,
    kSw64kbD = 10,
    kSw64kbST = 17,
    kSw64kbDT = 18,
    kSw4kbSX = 21,
    kSw4kbDX = 22,
    kSw4kbRX = 23,
    kSw64kbZX = 24,
    kSw64kbSX = 25,
    kSw64kbDX = 26,
    kSw64kbRX = 27,
};

// log2 of the block size in bytes (0 for linear); nullopt for modes GFX10 does not have.
inline std::optional<uint32_t> block_log2(uint32_t swizzle) {
    switch (swizzle) {
    case 0: return 0;
    case 1: case 2: return 8;
    case 5: case 6: case 21: case 22: case 23: return 12;
    case 9: case 10: case 17: case 18: case 24: case 25: case 26: case 27: return 16;
    default: return std::nullopt;
    }
}
inline bool is_standard(uint32_t s) { return s == 1 || s == 5 || s == 9 || s == 17 || s == 21 || s == 25; }
inline bool is_display(uint32_t s) { return s == 2 || s == 6 || s == 10 || s == 18 || s == 22 || s == 26; }
inline bool is_z_order(uint32_t s) { return s == 24; }
// Pipe and bank XOR modes (_T, _X): their addresses depend on the GPU's pipe configuration.
inline bool is_xor(uint32_t s) { return s >= 16; }

enum class Dimension { tex1d, tex2d, tex3d };

enum class Error {
    ok,
    invalid,      // the description makes no texture, or a coordinate or subresource lies outside it
    unsupported,  // a valid layout whose addressing is not implemented here
    too_small,    // a buffer is smaller than the layout or the rows need
};

// One plane of a texture.
struct Desc {
    Dimension dimension = Dimension::tex2d;
    uint32_t width = 1, height = 1;  // in texels
    uint32_t depth = 1;              // depth of a 3D texture, else the array size
    uint32_t mips = 1;
    uint32_t element_bytes = 4;  // 1, 2, 4, 8 or 16
    uint32_t block = 1;          // texels per element side: 1, or 4 for block compression
    uint32_t samples = 1;
    uint32_t swizzle = kLinear;
};

struct Tail {
    uint32_t offset = 0;  // byte offset inside the tail block
    uint32_t x = 0, y = 0;  // element position the mip starts at
};

// Where one mip lives inside a slice.
struct Mip {
    uint32_t width = 0, height = 0, depth = 0;  // size in elements (depth 1 unless the texture is 3D)
    uint32_t pitch = 0, rows = 0, depth_pitch = 0;  // padded size in elements: whole blocks, or the tail's share
    uint64_t offset = 0;                            // byte offset of its first block in the slice (0 in the tail)
    std::optional<Tail> tail;
};

namespace detail {

constexpr uint32_t div_ceil(uint32_t a, uint32_t b) { return (a + b - 1) / b; }
constexpr uint32_t round_up(uint32_t a, uint32_t b) { return div_ceil(a, b) * b; }

// Elements along one side of mip `mip`, as data and copies see it: texels halve rounding down (to at least 1), then
// round up to whole elements.
constexpr uint32_t dim(uint32_t texels, uint32_t mip, uint32_t block) {
    return div_ceil(std::max(texels >> mip, 1u), block);
}
// Elements along one side of mip `mip` for placement (padding, tail, offsets): addrlib halves mip 0's elements
// rounding up, which can be one more than dim() for odd sizes.
constexpr uint32_t padded_dim(uint32_t texels, uint32_t mip, uint32_t block) {
    return div_ceil(div_ceil(std::max(texels, 1u), block), 1u << mip);
}

// Micro tile (256 bytes) patterns by log2 of the element size: for each address bit, the element coordinate bit it
// takes (x0, y1, z0...), "--" for the bits inside an element. Thin S, thin D, thick S (also the D 3D modes). From
// addrlib's GFX10_SW_256_S_RBPLUS, GFX10_SW_256_D_RBPLUS and GFX10_SW_4K_S3_RBPLUS patterns; the larger blocks add
// bits Y, X (thin) or Y, X, Z (thick) in turn.
inline constexpr const char* kMicro[3][5] = {
    {"x0x1x2x3y0y1y2y3", "--x0x1x2y0y1y2x3", "----x0x1y0y1y2x2", "------x0y0y1x1x2", "--------y0y1x0x1"},
    {"x0x1x2y1y0y2x3y3", "--x0x1x2y0y1y2x3", "----x0x1y0y1x2y2", "------x0y0x1x2y1", "--------x0y0x1y1"},
    {"x0x1z0y0z1y1x2z2", "--x0z0y0z1y1x1z2", "----x0y0z0y1x1z1", "------x0z0y0x1z1", "--------z0y0x0z1"},
};

// For each axis, the address bit each coordinate bit lands on.
inline std::array<std::vector<uint32_t>, 3> pattern(int kind, uint32_t element_log2, uint32_t block_log2) {
    std::array<std::vector<uint32_t>, 3> axes;
    const char* micro = kMicro[kind][element_log2];
    for (uint32_t bit = 0; bit < 8; ++bit) {
        const char a = micro[2 * bit];
        const int axis = a == 'x' ? 0 : a == 'y' ? 1 : a == 'z' ? 2 : -1;
        if (axis < 0) continue;
        const size_t index = static_cast<size_t>(micro[2 * bit + 1] - '0');
        auto& to = axes[static_cast<size_t>(axis)];
        if (to.size() <= index) to.resize(index + 1, 0);
        to[index] = bit;
    }
    static constexpr int kThin[2] = {1, 0};
    static constexpr int kThick[3] = {1, 0, 2};
    const int* order = kind == 2 ? kThick : kThin;
    const uint32_t n = kind == 2 ? 3u : 2u;
    for (uint32_t bit = 8, i = 0; bit < block_log2; ++bit, ++i) axes[static_cast<size_t>(order[i % n])].push_back(bit);
    return axes;
}

// Byte offsets inside a block for coordinates 0..2^bits of one axis.
inline std::vector<uint32_t> spread(const std::vector<uint32_t>& bits) {
    std::vector<uint32_t> out(size_t{1} << bits.size());
    for (uint32_t c = 0; c < out.size(); ++c) {
        uint32_t v = 0;
        for (size_t i = 0; i < bits.size(); ++i) v |= ((c >> i) & 1u) << bits[i];
        out[c] = v;
    }
    return out;
}

}  // namespace detail

constexpr uint32_t kMaxMips = 16;
// The largest width, height and depth (or array size) accepted, in texels. GFX10's image descriptors hold at most
// 16384 texels per side, so no texture the GPU can use is refused. The bound also keeps every size computed below
// within uint64_t (and its uint32_t steps), which unchecked values from a title would otherwise overflow.
constexpr uint32_t kMaxExtent = 1u << 14;

// 256-byte micro tile dimensions by log2 element size, 2D and 3D.
inline constexpr uint32_t kBlock256_2d[5][2] = {{16, 16}, {16, 8}, {8, 8}, {8, 4}, {4, 4}};
inline constexpr uint32_t kBlock256_3d[5][3] = {{8, 4, 8}, {4, 4, 8}, {4, 4, 4}, {4, 2, 4}, {2, 2, 4}};
inline constexpr uint32_t kBlock1k_3d[5][3] = {{16, 8, 8}, {8, 8, 8}, {8, 8, 4}, {8, 4, 4}, {4, 4, 4}};

class Layout {
public:
    Desc desc;
    std::array<uint32_t, 3> block{};  // block dimensions in elements (linear: the pitch alignment, 1, 1)
    bool thick = false;               // 3D blocks span block[2] depth slices (GFX10 3D S and D modes)
    // Array slices, depth slabs of block[2] slices (thick), or depth slices (thin and linear 3D).
    uint32_t slices = 0;
    uint64_t slice_size = 0;  // bytes of one slice: its whole mip chain
    uint64_t size = 0;
    uint64_t alignment = 0;
    uint32_t first_tail_mip = 0;  // the first mip in the tail; mips.size() without one
    std::vector<Mip> mips;

    static std::optional<Layout> make(const Desc& d) {
        const auto log2 = block_log2(d.swizzle);
        if (!log2) return std::nullopt;
        const auto in = [](uint32_t v, uint32_t lo, uint32_t hi) { return v >= lo && v <= hi; };
        const bool valid =
            in(d.width, 1, kMaxExtent) && in(d.height, 1, kMaxExtent) && in(d.depth, 1, kMaxExtent) &&
            in(d.mips, 1, kMaxMips) && std::has_single_bit(d.element_bytes) && d.element_bytes <= 16 &&
            (d.block == 1 || d.block == 4) && std::has_single_bit(d.samples) && d.samples <= 8 &&
            (d.samples == 1 || (d.dimension == Dimension::tex2d && d.mips == 1)) &&
            (d.dimension != Dimension::tex1d || d.height == 1) &&
            (d.dimension != Dimension::tex3d || d.swizzle == 0 || d.swizzle == 5 || d.swizzle == 9 ||
             d.swizzle == 17 || d.swizzle == 21 || (d.swizzle >= 24 && d.swizzle <= 27));
        if (!valid) return std::nullopt;
        return *log2 == 0 ? linear(d) : tiled(d, *log2);
    }

    // Whether offset() and the copies work for this layout: linear and the modes without pipe or bank XOR,
    // single-sampled. The XOR modes (_X, _T) and MSAA need the GPU's pipe and bank configuration (Series X and S
    // differ); titles use them for render targets and depth, which the host renders.
    bool addressable() const { return desc.swizzle == kLinear || !pattern_[0].empty(); }

    // Depth slices a subresource of mip `mip` covers: the mip's depth for 3D textures, else 1 (one array slice).
    uint32_t subresource_depth(uint32_t mip) const { return mip < mips.size() ? mips[mip].depth : 0; }

    // Byte offset of element (x, y) of mip `mip` in array slice or depth slice `z`. Coordinates must lie inside.
    Error offset(uint32_t mip, uint32_t x, uint32_t y, uint32_t z, uint64_t& out) const {
        if (mip >= mips.size()) return Error::invalid;
        const Mip& m = mips[mip];
        const uint32_t depth = desc.dimension == Dimension::tex3d ? m.depth : slices;
        const bool inside = x < m.width && y < m.height && z < depth;
        if (!inside) return Error::invalid;
        if (!addressable()) return Error::unsupported;
        const auto [slice, inner] = split(z);
        const uint64_t base = uint64_t{slice} * slice_size + m.offset;
        if (desc.swizzle == kLinear) {
            out = base + (uint64_t{y} * m.pitch + x) * desc.element_bytes;
            return Error::ok;
        }
        out = base + row(m, y, inner) + column(m, x);
        return Error::ok;
    }

    // Copies one subresource out of tiled memory (the whole texture, `size` bytes) into linear rows: mip `mip` of
    // array slice `slice` (3D: every depth slice of the mip, slice 0). Rows are row_pitch bytes apart, depth slices
    // depth_pitch.
    Error detile(const uint8_t* tiled, size_t tiled_len, uint32_t mip, uint32_t slice, uint8_t* linear,
                 size_t linear_len, size_t row_pitch, size_t depth_pitch) const {
        return walk(tiled_len, mip, slice, linear_len, row_pitch, depth_pitch, [&](size_t t, size_t l, size_t n) {
            std::memcpy(linear + l, tiled + t, n);
        });
    }
    // The reverse of detile(): linear rows into tiled memory.
    Error tile(uint8_t* tiled, size_t tiled_len, uint32_t mip, uint32_t slice, const uint8_t* linear,
               size_t linear_len, size_t row_pitch, size_t depth_pitch) const {
        return walk(tiled_len, mip, slice, linear_len, row_pitch, depth_pitch, [&](size_t t, size_t l, size_t n) {
            std::memcpy(tiled + t, linear + l, n);
        });
    }

    // Byte offsets inside a block of each element coordinate bit, per axis; empty for linear and XOR/MSAA modes.
    const std::array<std::vector<uint32_t>, 3>& pattern() const { return pattern_; }

private:
    std::array<std::vector<uint32_t>, 3> pattern_;

    // HwlComputeSurfaceInfoLinear: rows aligned to 256 bytes, mips packed smallest first, a 3D texture's depth slices
    // each hold the chain.
    static Layout linear(const Desc& d) {
        using namespace detail;
        Layout l;
        l.desc = d;
        const uint64_t bpe = d.element_bytes;
        const uint32_t align = 256 / d.element_bytes;
        const bool three_d = d.dimension == Dimension::tex3d;
        uint64_t offset = 0;
        for (uint32_t i = d.mips; i-- > 0;) {
            const uint32_t h = padded_dim(d.height, i, d.block);
            const uint32_t pitch = round_up(padded_dim(d.width, i, d.block), align);
            Mip m;
            m.width = dim(d.width, i, d.block);
            m.height = dim(d.height, i, d.block);
            m.depth = three_d ? dim(d.depth, i, 1) : 1;
            m.pitch = pitch;
            m.rows = h;
            m.depth_pitch = three_d ? d.depth : 1;
            m.offset = offset;
            l.mips.push_back(m);
            offset += uint64_t{pitch} * h * bpe;
        }
        std::reverse(l.mips.begin(), l.mips.end());
        l.block = {align, 1, 1};
        l.slices = d.depth;
        l.slice_size = offset;
        l.size = offset * d.depth;
        l.alignment = 256;
        l.first_tail_mip = d.mips;
        return l;
    }

    static Layout tiled(const Desc& d, uint32_t block_log2) {
        using namespace detail;
        Layout l;
        l.desc = d;
        const uint32_t element_log2 = static_cast<uint32_t>(std::countr_zero(d.element_bytes));
        const uint64_t bpe = d.element_bytes;
        const uint64_t block_bytes = uint64_t{1} << block_log2;
        const bool thick = d.dimension == Dimension::tex3d && (is_standard(d.swizzle) || is_display(d.swizzle));
        std::array<uint32_t, 3> block{};
        if (thick) {  // ComputeThickBlockDimension
            const uint32_t amp = (block_log2 - 10) / 3, rest = (block_log2 - 10) % 3;
            const auto* b = kBlock1k_3d[element_log2];
            block = {b[0] << amp, b[1] << (amp + rest / 2), b[2] << (amp + (rest != 0 ? 1u : 0u))};
        } else {  // ComputeThinBlockDimension
            const uint32_t samples_log2 = static_cast<uint32_t>(std::countr_zero(d.samples));
            const uint32_t elements_log2 = block_log2 - element_log2 - samples_log2;
            const bool wide = (samples_log2 & 1) == 0 || (block_log2 & 1) != 0;
            const uint32_t w = (elements_log2 + (wide ? 1u : 0u)) / 2;
            block = {1u << w, 1u << (elements_log2 - w), 1};
        }
        uint32_t slices = round_up(d.depth, block[2]);
        const bool three_d = d.dimension == Dimension::tex3d;
        // Mip depths placed by a thick layout (thin ones store depth as slices).
        const auto mip_depth = [&](uint32_t i) { return thick ? padded_dim(d.depth, i, 1) : 1u; };
        const auto depth_of = [&](uint32_t i) { return three_d ? dim(d.depth, i, 1) : 1u; };
        const auto width = [&](uint32_t i) { return dim(d.width, i, d.block); };
        const auto height = [&](uint32_t i) { return dim(d.height, i, d.block); };
        uint32_t first_tail_mip = d.mips;
        uint64_t slice_size = 0;
        if (d.mips == 1) {
            Mip m;
            m.width = width(0);
            m.height = height(0);
            m.depth = depth_of(0);
            m.pitch = round_up(width(0), block[0]);
            m.rows = round_up(height(0), block[1]);
            m.depth_pitch = three_d ? slices : 1;
            l.mips.push_back(m);
            slice_size = uint64_t{m.pitch} * m.rows * bpe * d.samples;
        } else {
            // GetMipTailDim: the largest mip the tail takes.
            std::array<uint32_t, 3> tail_max = block;
            if (thick) {
                switch (block_log2 % 3) {
                case 0: tail_max[1] >>= 1; break;
                case 1: tail_max[0] >>= 1; break;
                default: tail_max[2] >>= 1; break;
                }
            } else {
                tail_max[0] >>= 1;
            }
            // GetMaxNumMipsInTail; 256-byte blocks have no tail (ComputeSurfaceInfoMicroTiled).
            const uint32_t effective = thick ? block_log2 - (block_log2 - 8) / 3 : block_log2;
            const uint32_t max_in_tail = effective == 8                       ? 0
                                         : (effective >= 9 && effective <= 11) ? 1 + (1u << (effective - 9))
                                                                               : effective - 4;
            // dsMipmapHtileFix: 8- and 16-bit depth enter the tail as 32-bit.
            std::array<uint32_t, 3> fixed = tail_max;
            if (is_z_order(d.swizzle) && element_log2 <= 1) {
                const auto* b = kBlock256_2d[element_log2];
                fixed[0] /= b[0] / kBlock256_2d[2][0];
                fixed[1] /= b[1] / kBlock256_2d[2][1];
            }
            std::vector<uint64_t> sizes;
            for (uint32_t i = 0; i < d.mips; ++i) {
                const uint32_t w = padded_dim(d.width, i, d.block), h = padded_dim(d.height, i, d.block);
                if (w <= fixed[0] && h <= fixed[1] && d.mips - i <= max_in_tail) {
                    first_tail_mip = i;
                    slice_size += block_bytes / block[2];
                    break;
                }
                const uint32_t pitch = round_up(w, block[0]), rows = round_up(h, block[1]);
                const uint32_t depth = round_up(mip_depth(i), block[2]);
                const uint64_t one = uint64_t{pitch} * rows * bpe;
                slice_size += one;
                sizes.push_back(one * block[2]);
                Mip m;
                m.width = width(i);
                m.height = height(i);
                m.depth = depth_of(i);
                m.pitch = pitch;
                m.rows = rows;
                m.depth_pitch = three_d ? depth : 1;
                l.mips.push_back(m);
            }
            // Standard mips from the smallest up, after the tail block.
            uint64_t offset = first_tail_mip < d.mips ? block_bytes : 0;
            for (uint32_t i = first_tail_mip; i-- > 0;) {
                l.mips[i].offset = offset;
                offset += sizes[i];
            }
            uint32_t micro[2];
            if (thick) {
                micro[0] = kBlock256_3d[element_log2][0];
                micro[1] = kBlock256_3d[element_log2][1];
            } else {
                micro[0] = kBlock256_2d[element_log2][0];
                micro[1] = kBlock256_2d[element_log2][1];
            }
            uint32_t pitch = tail_max[0], rows = tail_max[1];
            for (uint32_t i = first_tail_mip; i < d.mips; ++i) {
                const uint32_t m = max_in_tail - 1 - (i - first_tail_mip);
                const uint32_t offset_in_tail = m > 6 ? 16u << m : m << 8;
                // Address bits 8.. of the offset, odd ones x, even ones y.
                const auto bits = [&](uint32_t from) {
                    uint32_t v = 0;
                    for (uint32_t b = 0; b < 6; ++b) v |= ((offset_in_tail >> (from + 2 * b)) & 1u) << b;
                    return v;
                };
                uint32_t x = bits(9), y = bits(8);
                if ((block_log2 & 1) != 0) {
                    std::swap(x, y);
                    if ((element_log2 & 1) != 0) {
                        y = (y << 1) | (x & 1);
                        x >>= 1;
                    }
                }
                Mip t;
                t.width = width(i);
                t.height = height(i);
                t.depth = depth_of(i);
                t.pitch = pitch;
                t.rows = rows;
                t.depth_pitch = thick ? round_up(mip_depth(i), kBlock256_3d[element_log2][2]) : 1;
                t.offset = 0;
                t.tail = Tail{offset_in_tail, x * micro[0], y * micro[1]};
                l.mips.push_back(t);
                pitch = std::max(pitch >> 1, micro[0]);
                rows = std::max(rows >> 1, micro[1]);
            }
        }
        // Thick layouts store each depth slab's chain together.
        const uint64_t slab = thick ? block[2] : 1;
        slice_size *= slab;
        slices /= block[2];
        if (!is_xor(d.swizzle) && d.samples == 1) {
            const int kind = thick ? 2 : (is_display(d.swizzle) ? 1 : 0);
            const auto bits = detail::pattern(kind, element_log2, block_log2);
            for (size_t a = 0; a < 3; ++a) l.pattern_[a] = detail::spread(bits[a]);
        }
        l.block = block;
        l.thick = thick;
        l.slices = slices;
        l.slice_size = slice_size;
        l.size = slice_size * slices;
        l.alignment = block_bytes;
        l.first_tail_mip = first_tail_mip;
        return l;
    }

    // The slice and the depth inside it for z (an array slice or a 3D depth slice).
    std::pair<uint32_t, uint32_t> split(uint32_t z) const {
        return thick ? std::pair{z / block[2], z % block[2]} : std::pair{z, 0u};
    }

    // The part of an address that depends on the row (and depth in a thick block).
    uint64_t row(const Mip& m, uint32_t y, uint32_t z) const {
        const uint32_t bw = block[0], bh = block[1];
        const int log2 = std::countr_zero(alignment);
        // In the tail the pattern of the shifted position gives the mip's offset inside the tail block.
        y += m.tail ? m.tail->y : 0;
        const uint64_t blocks = uint64_t{y / bh} * (m.pitch / bw);
        return (blocks << log2) + (pattern_[1][y % bh] | pattern_[2][z]);
    }

    // The part of an address that depends on the column.
    uint64_t column(const Mip& m, uint32_t x) const {
        const uint32_t bw = block[0];
        x += m.tail ? m.tail->x : 0;
        return (uint64_t{x / bw} << std::countr_zero(alignment)) + pattern_[0][x % bw];
    }

    // Visits contiguous runs as (tiled byte offset, linear byte offset, length).
    template <class Copy>
    Error walk(size_t tiled_len, uint32_t mip, uint32_t slice, size_t linear_len, size_t row_pitch,
               size_t depth_pitch, Copy&& copy) const {
        if (mip >= mips.size()) return Error::invalid;
        const Mip& m = mips[mip];
        const uint32_t depth = m.depth;
        uint32_t first = slice;
        if (desc.dimension == Dimension::tex3d) {
            if (slice != 0) return Error::invalid;
            first = 0;
        }
        if (first >= slices) return Error::invalid;
        if (!addressable()) return Error::unsupported;
        const size_t n = desc.element_bytes;
        const size_t bytes = size_t{m.width} * n;
        // need = (depth - 1) * depth_pitch + (height - 1) * row_pitch + bytes, refused on overflow.
        const size_t dz = depth - 1u, dy = m.height - 1u;
        if ((dz && depth_pitch > SIZE_MAX / dz) || (dy && row_pitch > SIZE_MAX / dy)) return Error::too_small;
        const size_t depth_offset = dz * depth_pitch, row_offset = dy * row_pitch;
        if (depth_offset > SIZE_MAX - row_offset || depth_offset + row_offset > SIZE_MAX - bytes)
            return Error::too_small;
        const size_t need = depth_offset + row_offset + bytes;
        if (tiled_len < size || linear_len < need || row_pitch < bytes) return Error::too_small;
        const bool is_linear = desc.swizzle == kLinear;
        // The row and depth contribute one common base to every column, so the same contiguous runs repeat on
        // every row, including mip tails.
        struct Run {
            size_t tiled, linear, len;
        };
        std::vector<Run> runs;
        if (!is_linear) {
            size_t first_linear = 0, first_tiled = static_cast<size_t>(column(m, 0));
            for (uint32_t x = 1; x < m.width; ++x) {
                const size_t off = static_cast<size_t>(column(m, x));
                const size_t at = x * n;
                if (off != first_tiled + (at - first_linear)) {
                    runs.push_back({first_tiled, first_linear, at - first_linear});
                    first_tiled = off;
                    first_linear = at;
                }
            }
            runs.push_back({first_tiled, first_linear, bytes - first_linear});
        }
        for (uint32_t z = 0; z < depth; ++z) {
            const auto [s, inner] = split(first + z);
            const uint64_t base = uint64_t{s} * slice_size + m.offset;
            for (uint32_t y = 0; y < m.height; ++y) {
                const size_t linear_row = z * depth_pitch + y * row_pitch;
                if (is_linear) {
                    copy(static_cast<size_t>(base + uint64_t{y} * m.pitch * n), linear_row, bytes);
                } else {
                    const size_t tiled_row = static_cast<size_t>(base + row(m, y, inner));
                    for (const Run& r : runs) copy(tiled_row + r.tiled, linear_row + r.linear, r.len);
                }
            }
        }
        return Error::ok;
    }
};

// What a DXGI format stores per element: bytes, texels per element side (4 for block compression), and whether its
// stencil is a second plane (the GPU keeps depth and stencil apart). nullopt for 96-bit, 1-bit, packed 4:2:2 and
// video formats, which have no layout here.
struct Format {
    uint32_t bytes, block;
    bool stencil;
};
inline std::optional<Format> format(uint32_t dxgi) {
    const auto in = [&](uint32_t lo, uint32_t hi) { return dxgi >= lo && dxgi <= hi; };
    if (in(1, 4)) return Format{16, 1, false};
    if (in(9, 18)) return Format{8, 1, false};
    if (in(19, 22) || in(44, 47)) return Format{4, 1, true};
    if (in(23, 43) || dxgi == 67 || in(87, 93) || dxgi == 116 || dxgi == 117) return Format{4, 1, false};
    if (in(48, 59) || dxgi == 85 || dxgi == 86 || dxgi == 115) return Format{2, 1, false};
    if (in(118, 120)) return Format{2, 1, true};
    if (in(60, 65)) return Format{1, 1, false};
    if (in(70, 72) || in(79, 81)) return Format{8, 4, false};
    if (in(73, 78) || in(82, 84) || in(94, 99)) return Format{16, 4, false};
    return std::nullopt;
}

// A texture as the console allocates it: a D3D12-style description with a swizzle mode.
struct TextureDesc {
    Dimension dimension = Dimension::tex2d;
    uint32_t width = 1, height = 1;  // in texels
    uint32_t depth_or_array = 1;     // depth of a 3D texture, else the array size
    uint32_t mips = 1;               // 0: the full chain
    uint32_t format = 0;             // DXGI format number
    uint32_t samples = 1;
    uint32_t swizzle = kLinear;
};

struct Plane {
    uint64_t offset = 0;  // from the start of the texture
    bool stencil = false;
    Layout layout;
};

// Every data plane of a texture (depth, then stencil), placed one after the other. No compression metadata planes
// (HTILE, CMASK, FMASK, DCC): the console adds them to render targets and depth buffers unless the description
// denies them, so those are smaller here than on the console; the host GPU renders them, so nothing reads that
// memory.
struct Texture {
    std::vector<Plane> planes;
    uint64_t size = 0;
    uint64_t alignment = 0;

    static std::optional<Texture> make(const TextureDesc& desc) {
        const auto f = format(desc.format);
        if (!f) return std::nullopt;
        const uint32_t depth3d = desc.dimension == Dimension::tex3d ? desc.depth_or_array : 1;
        const uint32_t mips =
            desc.mips ? desc.mips
                      : 32u - static_cast<uint32_t>(std::countl_zero(
                                  std::max({desc.width, desc.height, depth3d, 1u})));
        const auto plane = [&](uint32_t bytes) {
            Desc d;
            d.dimension = desc.dimension;
            d.width = desc.width;
            d.height = desc.height;
            d.depth = desc.depth_or_array;
            d.mips = mips;
            d.element_bytes = bytes;
            d.block = f->block;
            d.samples = std::max(desc.samples, 1u);
            d.swizzle = desc.swizzle;
            return d;
        };
        Texture t;
        auto main = Layout::make(plane(f->bytes));
        if (!main) return std::nullopt;
        t.planes.push_back({0, false, std::move(*main)});
        if (f->stencil) {
            auto s = Layout::make(plane(1));
            if (!s) return std::nullopt;
            const uint64_t a = s->alignment;
            const uint64_t offset = (t.planes[0].layout.size + a - 1) / a * a;
            t.planes.push_back({offset, true, std::move(*s)});
        }
        const Plane& last = t.planes.back();
        t.size = last.offset + last.layout.size;
        for (const Plane& p : t.planes) t.alignment = std::max(t.alignment, p.layout.alignment);
        return t;
    }
};

}  // namespace leonida::console_texture
