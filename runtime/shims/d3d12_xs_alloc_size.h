// Console allocation sizes for render targets, for the Xbox-ABI device proxy's slots 25 and 56 (GetResourceAllocationInfo and GetResourceAllocationInfo1).
//
// Why: the retail title checks a fixed memory budget against what its first full-screen render targets used and asserts
// when more than 2 MiB is left over (docs/retail_boot_chain.md, stop 17, #6982). The PC device sizes a 2D render target
// from the Vulkan driver's memory requirements, which is about width x height x element bytes. The console lays a
// render target out in 64 KiB tiles (the RDNA2 SW_64KB layouts), so its allocation is the surface padded to whole
// tiles: for 4-byte elements a tile is 128 x 128 pixels, and 1080 rows take nine tile rows, not 8.4. The PC size is
// smaller by that padding, summed over every full-screen target.
//
// What this is: the tile shapes are the RDNA2 SW_64KB_D shapes of runtime/src/gfx10_tiling.h (Mesa addrlib patterns,
// validated against real textures in docs/textures.md). That a console render target uses these shapes and no extra
// (DCC, CMask) memory in its allocation size is an assumption: the title's own sizes are not measured. The rule only
// ever makes a size larger than the PC device's own answer, never smaller. The mip tail follows AMD's addrlib
// (gfx10, 2D): a mip extent is rounded up (ShiftCeil), and the tail starts at the first mip that fits in half a tile's
// width and a whole tile's height; the tail is one tile.
#pragma once

#include "../src/gfx10_tiling.h"
#include "xg_xs_layout.h"

#include <windows.h>
#include <d3d12.h>

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace rt::d3d12_xs {

constexpr UINT64 kConsoleTileBytes = 64 * 1024;  // one SW_64KB tile

// Bytes per element of the colour formats a render target takes, or 0 for any other format (the PC size then stays).
inline std::size_t render_target_element_bytes(DXGI_FORMAT format) {
    switch (format) {
        case DXGI_FORMAT_R8_TYPELESS: case DXGI_FORMAT_R8_UNORM: case DXGI_FORMAT_R8_UINT:
        case DXGI_FORMAT_R8_SNORM: case DXGI_FORMAT_R8_SINT: case DXGI_FORMAT_A8_UNORM:
            return 1;
        case DXGI_FORMAT_R8G8_TYPELESS: case DXGI_FORMAT_R8G8_UNORM: case DXGI_FORMAT_R8G8_UINT:
        case DXGI_FORMAT_R8G8_SNORM: case DXGI_FORMAT_R8G8_SINT: case DXGI_FORMAT_R16_TYPELESS:
        case DXGI_FORMAT_R16_FLOAT: case DXGI_FORMAT_R16_UNORM: case DXGI_FORMAT_R16_UINT:
        case DXGI_FORMAT_R16_SNORM: case DXGI_FORMAT_R16_SINT: case DXGI_FORMAT_B5G6R5_UNORM:
        case DXGI_FORMAT_B5G5R5A1_UNORM:
            return 2;
        case DXGI_FORMAT_R10G10B10A2_TYPELESS: case DXGI_FORMAT_R10G10B10A2_UNORM:
        case DXGI_FORMAT_R10G10B10A2_UINT: case DXGI_FORMAT_R11G11B10_FLOAT:
        case DXGI_FORMAT_R8G8B8A8_TYPELESS: case DXGI_FORMAT_R8G8B8A8_UNORM:
        case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB: case DXGI_FORMAT_R8G8B8A8_UINT:
        case DXGI_FORMAT_R8G8B8A8_SNORM: case DXGI_FORMAT_R8G8B8A8_SINT:
        case DXGI_FORMAT_B8G8R8A8_TYPELESS: case DXGI_FORMAT_B8G8R8A8_UNORM:
        case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB: case DXGI_FORMAT_B8G8R8X8_TYPELESS:
        case DXGI_FORMAT_B8G8R8X8_UNORM: case DXGI_FORMAT_B8G8R8X8_UNORM_SRGB:
        case DXGI_FORMAT_R16G16_TYPELESS: case DXGI_FORMAT_R16G16_FLOAT: case DXGI_FORMAT_R16G16_UNORM:
        case DXGI_FORMAT_R16G16_UINT: case DXGI_FORMAT_R16G16_SNORM: case DXGI_FORMAT_R16G16_SINT:
        case DXGI_FORMAT_R32_TYPELESS: case DXGI_FORMAT_R32_FLOAT: case DXGI_FORMAT_R32_UINT:
        case DXGI_FORMAT_R32_SINT:
            return 4;
        case DXGI_FORMAT_R16G16B16A16_TYPELESS: case DXGI_FORMAT_R16G16B16A16_FLOAT:
        case DXGI_FORMAT_R16G16B16A16_UNORM: case DXGI_FORMAT_R16G16B16A16_UINT:
        case DXGI_FORMAT_R16G16B16A16_SNORM: case DXGI_FORMAT_R16G16B16A16_SINT:
        case DXGI_FORMAT_R32G32_TYPELESS: case DXGI_FORMAT_R32G32_FLOAT: case DXGI_FORMAT_R32G32_UINT:
        case DXGI_FORMAT_R32G32_SINT:
            return 8;
        case DXGI_FORMAT_R32G32B32A32_TYPELESS: case DXGI_FORMAT_R32G32B32A32_FLOAT:
        case DXGI_FORMAT_R32G32B32A32_UINT: case DXGI_FORMAT_R32G32B32A32_SINT:
            return 16;
        default:
            return 0;
    }
}

// The console's allocation size for a single-sampled 2D render target, or 0 when the rule does not apply (not a 2D
// texture, not a render target, more than one sample, an unlisted format, an empty extent, or a size that overflows).
// MipLevels 0 is the full chain, as on the PC device. A mip extent is rounded up. Mips before the tail are padded to
// whole tiles; the first mip that fits in half a tile's width and the whole tile height, and every mip after it, share
// one tail tile. Slices add up.
inline UINT64 console_render_target_size(const D3D12_RESOURCE_DESC& desc) {
    if (desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D) return 0;
    if (!(desc.Flags & D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET)) return 0;
    if (desc.SampleDesc.Count > 1) return 0;
    const std::size_t element_bytes = render_target_element_bytes(desc.Format);
    if (!element_bytes || !desc.Width || !desc.Height || !desc.DepthOrArraySize) return 0;

    const UINT full_chain = static_cast<UINT>(std::bit_width(std::max<UINT64>(desc.Width, desc.Height)));
    const UINT mips = desc.MipLevels ? std::min<UINT>(desc.MipLevels, full_chain) : full_chain;
    UINT64 slice = 0;
    bool tail = false;
    for (UINT mip = 0; mip < mips; ++mip) {
        const UINT64 width = std::max<UINT64>((desc.Width + (UINT64{1} << mip) - 1) >> mip, 1);
        const UINT64 height = std::max<UINT64>((UINT64{desc.Height} + (UINT64{1} << mip) - 1) >> mip, 1);
        leonida::gfx10::TiledSurface2D surface;
        if (width > UINT32_MAX ||
            !leonida::gfx10::TiledSurface2D::make(leonida::gfx10::TileMode::sw_64kb_d, element_bytes,
                                                  static_cast<std::size_t>(width), static_cast<std::size_t>(height),
                                                  surface))
            return 0;
        if (width <= surface.tile_width() / 2 && height <= surface.tile_height()) {
            tail = true;
            break;
        }
        slice += surface.tiled_size();
    }
    if (tail) slice += kConsoleTileBytes;
    const UINT64 slices = desc.DepthOrArraySize;
    if (slice > UINT64_MAX / slices) return 0;
    return slice * slices;
}

// The planes of a depth-stencil format as the console stores them, in bytes per element (0: no such plane), or
// {0, 0} for any other format. The console keeps depth and stencil in separate planes: 4 + 1 bytes for the 32-bit
// float depth formats and for D24S8, 2 + 1 for its own D16_UNORM_S8_UINT (DXGI 118), 4 or 2 alone for D32 and D16.
struct DepthPlanes {
    std::size_t depth = 0, stencil = 0;
};
inline DepthPlanes console_depth_planes(DXGI_FORMAT format) {
    switch (static_cast<int>(format)) {
        case DXGI_FORMAT_R32G8X24_TYPELESS: case DXGI_FORMAT_D32_FLOAT_S8X24_UINT:
        case DXGI_FORMAT_R24G8_TYPELESS: case DXGI_FORMAT_D24_UNORM_S8_UINT:
            return {4, 1};
        case 118: return {2, 1};  // the console's D16_UNORM_S8_UINT
        case DXGI_FORMAT_R32_TYPELESS: case DXGI_FORMAT_D32_FLOAT: return {4, 0};
        case DXGI_FORMAT_R16_TYPELESS: case DXGI_FORMAT_D16_UNORM: return {2, 0};
        default: return {};
    }
}

// The console's allocation size for a single-sampled 2D depth-stencil target: each plane laid out in 64 KiB tiles
// like a colour target of that element size, the planes added up. 0 when the rule does not apply.
//
// Why this one may be SMALLER than the PC answer, unlike the render-target rule: a PC driver stores D32_FLOAT_S8 as
// 8 bytes per texel (a 3840 x 2160 target is 63.75 MiB on NVIDIA; the console's two planes are 40.3 MiB), and the
// title checks the peak of its transient render-target pool, 4752 x 64 KiB on a Series S, against these sizes and
// stops on an int3 when it does not fit (measured: peak 320.0 MiB against 297 MiB with the PC sizes, two such
// targets alive at the peak; docs/re/answers.md). The title never writes a depth target's memory with the CPU and
// this runtime does not keep the texture in the title's allocation, so the smaller number is only its bookkeeping.
// No HiZ/HTile memory is added: whether the console counts it in the allocation is not measured.
inline UINT64 console_depth_stencil_size(const D3D12_RESOURCE_DESC& desc) {
    if (desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D) return 0;
    if (!(desc.Flags & D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL)) return 0;
    if (desc.SampleDesc.Count > 1 || !desc.Width || !desc.Height || !desc.DepthOrArraySize) return 0;
    const DepthPlanes planes = console_depth_planes(desc.Format);
    if (!planes.depth) return 0;
    const UINT full_chain = static_cast<UINT>(std::bit_width(std::max<UINT64>(desc.Width, desc.Height)));
    const UINT mips = desc.MipLevels ? std::min<UINT>(desc.MipLevels, full_chain) : full_chain;
    UINT64 slice = 0;
    for (const std::size_t element_bytes : {planes.depth, planes.stencil}) {
        if (!element_bytes) continue;
        bool tail = false;
        for (UINT mip = 0; mip < mips; ++mip) {
            const UINT64 width = std::max<UINT64>((desc.Width + (UINT64{1} << mip) - 1) >> mip, 1);
            const UINT64 height = std::max<UINT64>((UINT64{desc.Height} + (UINT64{1} << mip) - 1) >> mip, 1);
            leonida::gfx10::TiledSurface2D surface;
            if (width > UINT32_MAX ||
                !leonida::gfx10::TiledSurface2D::make(leonida::gfx10::TileMode::sw_64kb_d, element_bytes,
                                                      static_cast<std::size_t>(width),
                                                      static_cast<std::size_t>(height), surface))
                return 0;
            if (width <= surface.tile_width() / 2 && height <= surface.tile_height()) {
                tail = true;
                break;
            }
            slice += surface.tiled_size();
        }
        if (tail) slice += kConsoleTileBytes;
    }
    const UINT64 slices = desc.DepthOrArraySize;
    if (!slice || slice > UINT64_MAX / slices) return 0;
    return slice * slices;
}

// The console's size and alignment of a texture nothing renders or stores to (no render-target, depth-stencil or
// unordered-access flag): XG's layout of the description (xg_xs_layout.h; the default swizzle mode when the layout
// word names none). False when XG's model does not take the description: the PC answer then stands.
//
// Why exact: the title keeps this size in its streaming tables and reads exactly that many bytes of the texture
// from its archives through DirectStorage; a compressed chunk must be read whole, and the title stops on an int3
// when the size it was given is smaller than the chunk. Measured with a debugger at that int3: a
// request of 0x2000 bytes for a chunk that spans 0x4000, the PC answer for the texture being half the console's.
inline bool console_texture_layout(const D3D12_RESOURCE_DESC& desc, UINT64& size, UINT64& alignment) {
    constexpr UINT gpu_written = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET | D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL |
                                 D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    if (desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE1D && desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D &&
        desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE3D)
        return false;
    if ((desc.Flags & gpu_written) || desc.SampleDesc.Count != 1) return false;
    static_assert(sizeof(rt::xg_xs::ResourceDesc) == sizeof(D3D12_RESOURCE_DESC));
    rt::xg_xs::ResourceDesc xg{};
    std::memcpy(&xg, &desc, sizeof(xg));
    xg.alignment = 0;
    xg.quality = 0;
    xg.flags = 0;
    if (xg.layout != 1 && (xg.layout & ~0x1fu) != rt::xg_xs::kSwizzled) xg.layout = 0;  // XG's default mode
    static thread_local rt::xg_xs::Texture texture;  // 7 KiB: not on a title thread's stack
    texture = {};
    if (!rt::xg_xs::build_texture(xg, texture) || !texture.layout.size || !texture.layout.alignment) return false;
    size = texture.layout.size;
    alignment = texture.layout.alignment;
    return true;
}

// The console's size of a texture the GPU writes through unordered access (no render-target or depth-stencil flag,
// which have their own rules above) whose layout word names its swizzle mode: row-major (1), or 0x100 | mode. That
// is XG's layout of it; 0 when the rule does not apply (any other description, or a mode XG's model lacks).
// `title_layout` is the layout word as the title wrote it: the PC form of the description (pc_resource_desc) has
// lost it.
//
// Why: the title also writes such a texture with the CPU, through its texture lock, at the offsets
// and over the bytes XG's layout gives. Measured: a 32x32x36 R32_UINT 3D texture with layout 0x100 (linear) is
// 0x48000 bytes on the console (256-byte rows, 36 slices of 8 KiB), the PC device sizes it 0x30000, the title placed
// the next resource 0x30000 after it, and its lock clears all 0x48000 bytes - 0x18000 of them in the neighbour.
// Like the render-target rule this only ever makes a size larger than the PC device's own answer, never smaller
// (console_placement).
inline UINT64 console_unordered_access_size(const D3D12_RESOURCE_DESC& desc, UINT title_layout) {
    if (desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE1D && desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D &&
        desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE3D)
        return 0;
    constexpr UINT targets = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET | D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
    if (!(desc.Flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS) || (desc.Flags & targets) ||
        desc.SampleDesc.Count != 1)
        return 0;
    if (title_layout != 1 && (title_layout & ~0x1fu) != rt::xg_xs::kSwizzled) return 0;  // XG picks the mode
    static_assert(sizeof(rt::xg_xs::ResourceDesc) == sizeof(D3D12_RESOURCE_DESC));
    rt::xg_xs::ResourceDesc xg{};
    std::memcpy(&xg, &desc, sizeof(xg));
    xg.alignment = 0;
    xg.quality = 0;
    xg.flags = 0;
    xg.layout = title_layout;
    static thread_local rt::xg_xs::Texture texture;  // 7 KiB: not on a title thread's stack
    texture = {};
    if (!rt::xg_xs::build_texture(xg, texture) || !texture.layout.size) return 0;
    return texture.layout.size;
}

// Rounds up to a multiple of `to`, stopping at the last multiple below 2^64 instead of wrapping (as the PC device's
// totals do).
inline UINT64 align_up_saturating(UINT64 value, UINT64 to) {
    return value > UINT64_MAX - (to - 1) ? UINT64_MAX / to * to : (value + to - 1) / to * to;
}

// Lays the descriptions out as the PC device does (one after another, each at its own alignment, the total rounded up
// to the largest alignment), with a render target's size taken as its console size when that is larger than the PC
// size. pc_one(i) is the PC device's answer for description i alone. Sizes are rounded to each description's own
// alignment, so a multisampled neighbour at 4 MiB keeps its place, and sums saturate instead of wrapping. When
// `each` is given it receives every description's offset, alignment and size.
//
// Returns false, and writes nothing, when no description is a console render target that grows, or when the PC
// device refuses one: the PC answer then stands. A true answer is never below what the PC device gave for the same
// descriptions laid out in that way.
//
// `title_descs` is the title's own array when `descs` is its PC form, for the layout words the PC form drops
// (console_unordered_access_size); null when `descs` is the title's array.
template <class PcOne>
bool console_placement(const D3D12_RESOURCE_DESC* descs, UINT count, PcOne pc_one,
                       D3D12_RESOURCE_ALLOCATION_INFO& total, D3D12_RESOURCE_ALLOCATION_INFO1* each,
                       const D3D12_RESOURCE_DESC* title_descs = nullptr) {
    if (!descs) return false;
    const auto unordered_access = [&](UINT i) {
        return console_unordered_access_size(descs[i], static_cast<UINT>((title_descs ? title_descs : descs)[i].Layout));
    };
    bool any_target = false;
    for (UINT i = 0; i < count && !any_target; ++i) {
        UINT64 xg_size = 0, xg_alignment = 0;
        any_target = console_render_target_size(descs[i]) != 0 || console_depth_stencil_size(descs[i]) != 0 ||
                     console_texture_layout(descs[i], xg_size, xg_alignment) || unordered_access(i) != 0;
    }
    if (!any_target) return false;
    // Two passes when `each` is wanted: the first decides, the second writes, so a refusal or a list that does not
    // grow leaves the array as the PC device filled it.
    for (const bool write : {false, true}) {
        if (write && !each) break;
        bool grows = false;
        UINT64 end = 0, alignment = 1;
        for (UINT i = 0; i < count; ++i) {
            D3D12_RESOURCE_ALLOCATION_INFO one = pc_one(i);
            if (one.SizeInBytes == UINT64_MAX || !one.Alignment) return false;
            UINT64 size = one.SizeInBytes;
            // A texture only the CPU fills takes XG's size and alignment, whichever way they differ.
            if (UINT64 xg_size = 0, xg_alignment = 0; console_texture_layout(descs[i], xg_size, xg_alignment)) {
                if (xg_size != size || xg_alignment != one.Alignment) grows = true;
                size = xg_size;
                one.Alignment = xg_alignment;
            }
            const UINT64 console = console_render_target_size(descs[i]);
            if (console > size) {
                size = align_up_saturating(console, one.Alignment);
                grows = true;
            }
            // A texture the title also locks and writes with the CPU takes its console layout when that is larger.
            if (const UINT64 written = unordered_access(i); written > size) {
                size = align_up_saturating(written, one.Alignment);
                grows = true;
            }
            // A depth-stencil target takes its console size whichever way it differs (console_depth_stencil_size).
            if (const UINT64 depth = console_depth_stencil_size(descs[i])) {
                const UINT64 planes = align_up_saturating(depth, one.Alignment);
                if (planes != size) {
                    size = planes;
                    grows = true;
                }
            }
            const UINT64 offset = align_up_saturating(end, one.Alignment);
            if (write) each[i] = {offset, one.Alignment, size};
            end = size > UINT64_MAX - offset ? UINT64_MAX : offset + size;
            alignment = std::max(alignment, one.Alignment);
        }
        if (!grows) return false;
        total = {align_up_saturating(end, alignment), alignment};
    }
    return true;
}

}  // namespace rt::d3d12_xs
