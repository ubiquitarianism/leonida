// xg_xs.dll: the console's texture layout library. Titles ask it which swizzle mode a texture should use, how large it
// is, where each mip and slice lies, and to copy linear data into tiled memory and back. The game executable
// (1.0.2.1) imports two functions from it. The object has an Xbox vtable (AddRef and Release first, with no
// QueryInterface).
//
// Measured from the call sites in the executable, read statically (shapes and checks only):
//   HRESULT XGSuggestSwizzleMode(const desc*, 0, uint32_t* mode)
//     2 call sites, both in the game's helper that fills a texture description before XGCreateTextureComputer. The
//     out pointer is a 32-bit field of that description, which the helper first sets to 0 and, when the mode is
//     known, writes itself as 0x100 | mode. The HRESULT is only passed back up; no caller branches on it.
//   HRESULT XGCreateTextureComputer(const desc*, computer** out)
//     4 call sites. Two test the HRESULT and take their failure path when it is negative; two ignore it and call
//     through *out at once. On the computer the callers use vtable slot 1 after they are done (3 sites), slot 2 with
//     one out pointer to a large stack structure (2 sites), slot 8 with six arguments in a loop over subresources
//     (1 site) and slot 10 with seven arguments (1 site).
//
// Two layout models answer, in this order:
//   1. xg_xs_layout.h (single-plane textures that are not targets, in the linear, 4 KiB and 64 KiB standard and
//      64 KiB display modes). The D3D12 layer lays the title's streamed and placed textures out with the same header,
//      so what the title is told here is what the layer reads back.
//   2. runtime/src/console_texture_layout.h, for a description the first cannot lay out (targets, depth with stencil,
//      the other swizzle modes, multisampling): namespace `port` below. Its methods, their arguments and the layout
//      structure slot 2 fills are taken from Marvisak/VIctory (GPL-3.0-or-later; translated from Rust to C++ and
//      changed for Leonida), crates/victory-hle/src/modules/xg.rs at
//      commit d2a4bae1f, which serves the same title: 0 AddRef, 1 Release, 2 the layout, 6 an element's byte offset,
//      8 a copy into and 9 a copy out of a subresource, 10 the 64 KiB tiling. They match the slots measured above.
//      The layouts come from runtime/src/console_texture_layout.h, ported from the same project.
//
// Kept from the measured callers: a description neither model can lay out still gets a computer in *out (with a
// negative HRESULT), because two callers use *out without checking the HRESULT. That computer answers every slot
// with E_NOTIMPL (0 for the slots that return a number) and writes nothing. COM says *out is null on failure; this
// breaks that on purpose.
//
// Not modelled by the second model: compression metadata planes (HTILE, CMASK, FMASK, DCC), addressing and copies for
// the pipe/bank XOR modes (_X, _T) and MSAA (they need the GPU's pipe configuration; the host renders those targets),
// and the slots no caller reaches (element coordinate, view descriptor), which log once and return E_NOTIMPL.
#include <windows.h>
#include <intrin.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <optional>
#include <utility>

#include "../src/console_texture_layout.h"
#include "shims.h"
#include "xg_xs_layout.h"
#include "../src/title_page_watch.h"

namespace {

using namespace rt::xg_xs;
constexpr size_t kSlots = 32;

struct Computer {
    const void* const* vtable;
    std::atomic<ULONG> refs{1};
    Texture texture;
};

void trace(std::atomic<unsigned>& calls, const char* name, const void* caller, const char* format, ...) {
    static const bool enabled = [] {
        char value[2]{};
        return GetEnvironmentVariableA("LEONIDA_XG_TRACE", value, sizeof(value)) == 1 && value[0] == '1';
    }();
    if (!enabled) return;
    const unsigned call = calls.fetch_add(1, std::memory_order_relaxed) + 1;
    if (call > 1024) return;
    char details[512];
    va_list args;
    va_start(args, format);
    std::vsnprintf(details, sizeof(details), format, args);
    va_end(args);
    std::fprintf(stderr, "[rt] xg_xs: LOCAL DIAGNOSTIC: %s #%u caller=%s %s\n",
                 name, call, rt::shims::module_offset(reinterpret_cast<uint64_t>(caller)).c_str(), details);
}

void trace_desc(std::atomic<unsigned>& calls, const char* name, const void* caller,
                const ResourceDesc* desc, const void* out, uint32_t options,
                HRESULT hr, uint32_t mode, uint64_t bytes) {
    if (!desc) {
        trace(calls, name, caller, "desc=null out=%p options=%#x hr=0x%08x", out, options, uint32_t(hr));
        return;
    }
    trace(calls, name, caller,
          "dim=%u %llux%ux%u mips=%u fmt=%u samples=%u quality=%u align=%llu layout=%#x flags=%#x "
          "options=%#x out=%p hr=0x%08x mode=%#x bytes=%llu",
          desc->dimension, static_cast<unsigned long long>(desc->width), desc->height, unsigned(desc->depth_or_array),
          unsigned(desc->mips), desc->format, desc->samples, desc->quality, static_cast<unsigned long long>(desc->alignment),
          desc->layout, desc->flags, options, out, uint32_t(hr), mode, static_cast<unsigned long long>(bytes));
}

ULONG __stdcall add_ref(Computer* self) {
    static std::atomic<unsigned> calls{0};
    const ULONG refs = self->texture.layout.planes ? self->refs.fetch_add(1, std::memory_order_relaxed) + 1 : 1;
    trace(calls, "slot 0", _ReturnAddress(), "this=%p refs=%lu", self, refs);
    return refs;
}

ULONG __stdcall release(Computer* self) {
    static std::atomic<unsigned> calls{0};
    const ULONG refs = self->texture.layout.planes ? self->refs.fetch_sub(1, std::memory_order_acq_rel) - 1 : 1;
    trace(calls, "slot 1", _ReturnAddress(), "this=%p refs=%lu", self, refs);
    if (!refs) delete self;
    return refs;
}

HRESULT __stdcall get_layout(Computer* self, ResourceLayout* out) {
    static std::atomic<unsigned> calls{0};
    const auto& layout = self->texture.layout;
    const HRESULT hr = !layout.planes ? E_NOTIMPL : !out ? E_POINTER : S_OK;
    trace(calls, "slot 2", _ReturnAddress(), "this=%p out=%p hr=0x%08x bytes=%llu align=%llu mips=%u planes=%u",
          self, out, uint32_t(hr), static_cast<unsigned long long>(layout.size),
          static_cast<unsigned long long>(layout.alignment), layout.mips, layout.planes);
    if (SUCCEEDED(hr)) {
        *out = {layout.size, layout.alignment, layout.mips, layout.planes, {layout.plane[0]}, layout.dimension, 0};
    }
    return hr;
}

uint64_t __stdcall get_size(Computer* self) {
    static std::atomic<unsigned> calls{0};
    const uint64_t bytes = self->texture.layout.size;
    trace(calls, "slot 3", _ReturnAddress(), "this=%p bytes=%llu", self, static_cast<unsigned long long>(bytes));
    return bytes;
}

uint64_t __stdcall get_alignment(Computer* self) {
    static std::atomic<unsigned> calls{0};
    const uint64_t bytes = self->texture.layout.alignment;
    trace(calls, "slot 4", _ReturnAddress(), "this=%p bytes=%llu", self, static_cast<unsigned long long>(bytes));
    return bytes;
}

uint64_t __stdcall get_mip_offset(Computer* self, uint32_t plane, uint32_t level) {
    static std::atomic<unsigned> calls{0};
    const auto& layout = self->texture.layout;
    const uint64_t offset = plane < layout.planes && level < layout.mips ? layout.plane[plane].mips[level].offset : 0;
    trace(calls, "slot 5", _ReturnAddress(), "this=%p plane=%u mip=%u offset=%llu",
          self, plane, level, static_cast<unsigned long long>(offset));
    return offset;
}

uint64_t __stdcall get_element_offset(Computer* self, uint32_t plane, uint32_t level,
                                      uint64_t x, uint32_t y, uint32_t z, uint32_t sample, uint32_t* bit) {
    static std::atomic<unsigned> calls{0};
    const auto& t = self->texture;
    uint64_t offset = 0;
    if (bit) *bit = 0;
    if (plane < t.layout.planes && level < t.layout.mips && sample == 0) {
        const auto& mip = t.layout.plane[plane].mips[level];
        if (x < mip.width_elements && y < mip.height_elements && z < mip.depth) {
            const ElementAddress address(t);
            offset = address.offset(mip, address.tail_origin(mip), uint32_t(x), y, z);
        }
    }
    trace(calls, "slot 6", _ReturnAddress(), "this=%p plane=%u mip=%u x=%llu y=%u z=%u sample=%u bit=%p offset=%llu",
          self, plane, level, static_cast<unsigned long long>(x), y, z, sample, bit,
          static_cast<unsigned long long>(offset));
    return offset;
}

template<bool Into>
HRESULT copy_subresource(const Texture& t, uint32_t plane, uint32_t subresource, uint32_t sample,
                         const void* src, void* dst, uint32_t row_pitch, uint32_t slice_pitch) {
    const auto& layout = t.layout;
    if (!layout.planes) return E_NOTIMPL;
    if (!src || !dst) return E_POINTER;
    if (plane >= layout.planes || subresource >= layout.mips * t.array || sample) return E_INVALIDARG;
    const auto& mip = layout.plane[plane].mips[subresource % layout.mips];
    const uint32_t slice = subresource / layout.mips;
    const uint32_t row_bytes = mip.width_elements * t.format.bytes;
    if (row_pitch < row_bytes) return E_INVALIDARG;
    const uint32_t depth = t.layout.dimension == kTexture3D ? mip.depth : 1;
    const uint64_t one_slice = uint64_t(row_pitch) * (mip.height_elements - 1) + row_bytes;
    if (depth > 1 && slice_pitch < one_slice) return E_INVALIDARG;
    const uint64_t linear_size = uint64_t(slice_pitch) * (depth - 1) + one_slice;
    const auto source = static_cast<const std::byte*>(src);
    auto destination = static_cast<std::byte*>(dst);
    if (leonida::gfx10::tiling_detail::overlap(
            {source, Into ? linear_size : layout.size}, {destination, Into ? layout.size : linear_size}))
        return E_INVALIDARG;
    std::vector<uint64_t> columns;
    copy_elements<Into>(t, subresource % layout.mips, slice, source, destination, row_pitch, slice_pitch, columns);
    return S_OK;
}

HRESULT __stdcall copy_into(Computer* self, void* dst, uint32_t plane, uint32_t sub, const void* src,
                            uint32_t row, uint32_t slice) {
    static std::atomic<unsigned> calls{0};
    const HRESULT hr = copy_subresource<true>(self->texture, plane, sub, 0, src, dst, row, slice);
    // The texture's image on the PC GPU is filled from this memory: tell the D3D12 layer it changed.
    if (SUCCEEDED(hr)) leonida::title_texel_writes::note(dst, size_t(self->texture.layout.size));
    trace(calls, "slot 8", _ReturnAddress(), "this=%p dst=%p plane=%u sub=%u src=%p row=%u slice=%u hr=0x%08x",
          self, dst, plane, sub, src, row, slice, uint32_t(hr));
    return hr;
}

HRESULT __stdcall copy_from(Computer* self, void* dst, uint32_t plane, uint32_t sub, uint32_t sample,
                            const void* src, uint32_t row, uint32_t slice) {
    static std::atomic<unsigned> calls{0};
    const HRESULT hr = copy_subresource<false>(self->texture, plane, sub, sample, src, dst, row, slice);
    trace(calls, "slot 9", _ReturnAddress(), "this=%p dst=%p plane=%u sub=%u sample=%u src=%p row=%u slice=%u hr=0x%08x",
          self, dst, plane, sub, sample, src, row, slice, uint32_t(hr));
    return hr;
}

struct PackedMips {
    uint8_t standard, packed;
    uint16_t reserved;
    uint32_t tiles, start;
};
struct TileShape { uint32_t width, height, depth; };
struct SubresourceTiling { uint32_t width; uint16_t height, depth; uint32_t start; };
static_assert(sizeof(PackedMips) == 12 && sizeof(TileShape) == 12 && sizeof(SubresourceTiling) == 12);

HRESULT __stdcall get_tiling(Computer* self, uint32_t* total_tiles, PackedMips* packed, TileShape* shape,
                             uint32_t* count, uint32_t first, SubresourceTiling* tilings) {
    static std::atomic<unsigned> calls{0};
    const auto& t = self->texture;
    const auto& layout = t.layout;
    const auto& p = layout.plane[0];
    const unsigned requested = count ? *count : 0;
    const bool supported = layout.alignment == kTileBytes && layout.dimension == kTexture2D;
    const HRESULT hr = !layout.planes ? E_NOTIMPL : !supported ? E_FAIL : S_OK;
    const uint32_t tiles = supported ? uint32_t(layout.size / kTileBytes) : 0;
    unsigned written = 0;
    if (SUCCEEDED(hr)) {
        if (total_tiles) *total_tiles = tiles;
        if (packed) *packed = {uint8_t(p.standard_mips), uint8_t(p.tail_mips), 0, p.tail_mips ? 1u : 0u, 0};
        if (shape) *shape = {t.tile_width * t.format.side, t.tile_height * t.format.side, 1};
        const uint32_t total_subresources = t.array * layout.mips;
        if (count && first < total_subresources) {
            const uint32_t end = std::min(requested, total_subresources - first);
            for (uint32_t i = 0; i < end; ++i) {
                const uint32_t sub = first + i;
                const uint32_t level = sub % layout.mips;
                if (level >= p.standard_mips) continue;
                ++written;
                if (!tilings) continue;
                const auto& mip = p.mips[level];
                const uint64_t start_bytes = (sub / layout.mips) * p.slice_bytes + mip.offset;
                tilings[i] = {mip.padded_width / t.tile_width, uint16_t(mip.padded_height / t.tile_height), 1,
                              uint32_t(start_bytes / kTileBytes)};
            }
        }
        if (count) *count = written;
    }
    trace(calls, "slot 10", _ReturnAddress(),
          "this=%p total=%p packed=%p shape=%p count=%p first=%u out=%p requested=%u written=%u "
          "hr=0x%08x tiles=%u standard=%u tail=%u tile=%ux%u",
          self, total_tiles, packed, shape, count, first, tilings, requested, written, uint32_t(hr),
          tiles, p.standard_mips, p.tail_mips, t.tile_width * t.format.side, t.tile_height * t.format.side);
    return hr;
}

template<size_t Slot>
HRESULT __stdcall unimplemented_slot(Computer* self) {
    static std::atomic<unsigned> calls{0};
    trace(calls, "unknown slot", _ReturnAddress(), "this=%p slot=%zu hr=0x80004001", self, Slot);
    return E_NOTIMPL;
}

template<size_t... Slot>
auto make_vtable(std::index_sequence<Slot...>) {
    std::array<void*, sizeof...(Slot)> vtable{reinterpret_cast<void*>(&unimplemented_slot<Slot>)...};
    vtable[0] = reinterpret_cast<void*>(&add_ref);
    vtable[1] = reinterpret_cast<void*>(&release);
    vtable[2] = reinterpret_cast<void*>(&get_layout);
    vtable[3] = reinterpret_cast<void*>(&get_size);
    vtable[4] = reinterpret_cast<void*>(&get_alignment);
    vtable[5] = reinterpret_cast<void*>(&get_mip_offset);
    vtable[6] = reinterpret_cast<void*>(&get_element_offset);
    vtable[8] = reinterpret_cast<void*>(&copy_into);
    vtable[9] = reinterpret_cast<void*>(&copy_from);
    vtable[10] = reinterpret_cast<void*>(&get_tiling);
    return vtable;
}
const auto kVtable = make_vtable(std::make_index_sequence<kSlots>{});
Computer unavailable{kVtable.data(), {1}, {}};

// ---- The second model: runtime/src/console_texture_layout.h. ----

namespace port {

namespace ct = leonida::console_texture;

constexpr HRESULT kNotImpl = static_cast<HRESULT>(rt::shims::kENotImpl);
constexpr HRESULT kInvalidArg = static_cast<HRESULT>(0x80070057u);
constexpr HRESULT kFail = static_cast<HRESULT>(0x80004005u);

// D3D12_RESOURCE_DESC, whose layout field carries 0x100 | swizzle mode on the console (1 row major, 0 "pick one").
struct ResourceDesc {
    uint32_t dimension;
    uint64_t alignment;
    uint64_t width;
    uint32_t height;
    uint16_t depth_or_array;
    uint16_t mips;
    uint32_t format;
    uint32_t samples;
    uint32_t quality;
    uint32_t layout;
    uint32_t flags;
};
static_assert(sizeof(ResourceDesc) == 0x38);

constexpr uint32_t kSwizzled = 0x100;
constexpr uint32_t kRowMajor = 1;
constexpr uint32_t kAllowRenderTarget = 1;
constexpr uint32_t kAllowDepthStencil = 2;
constexpr uint32_t kTexture1d = 2, kTexture2d = 3, kTexture3d = 4;

// The layout structure slot 2 fills. Offsets 0x00, 0x08, 0x10 and 0x1c of a mip are the ones the title reads; the
// rest follow the library's older public shape (VIctory's reading).
constexpr size_t kMips = 15;
struct MipOut {
    uint64_t size;
    uint64_t offset;
    uint64_t slice_size;
    uint32_t pitch_texels;
    uint32_t pitch_bytes;
    uint32_t alignment;
    uint32_t padded[3];  // width, height in elements, depth or array size
    uint32_t elements[2];
    uint32_t texels[2];
    uint32_t depth_or_array;
    uint32_t samples;
    uint32_t swizzle;
    uint64_t bank_rotation[2];
    uint32_t slice_depth;
    uint32_t in_tail;
    uint32_t tail_position;  // where a tail mip starts in elements: x bits 0-10, y 11-21
};
struct PlaneOut {
    uint32_t usage;  // 1 data, 7 depth, 8 stencil (metadata planes are not made)
    uint64_t size;
    uint64_t slice_size;
    uint64_t offset;
    uint64_t alignment;
    uint32_t element_bytes;
    uint32_t standard_mips;
    uint32_t tail_mips;
    MipOut mips[kMips];
};
struct LayoutOut {
    uint64_t size;
    uint64_t alignment;
    uint32_t mips;
    uint32_t planes;
    PlaneOut plane[4];
    uint32_t dimension;
};
static_assert(sizeof(MipOut) == 0x70);
static_assert(sizeof(PlaneOut) == 0x6c8);
static_assert(sizeof(LayoutOut) == 0x1b40);

struct PackedMips {
    uint8_t standard;
    uint8_t packed;
    uint32_t tiles;
    uint32_t start;
};
struct SubresourceTiling {
    uint32_t width;
    uint16_t height;
    uint16_t depth;
    uint32_t start;
};
constexpr uint64_t kTile = 64u << 10;

// Prints once per call site kind; `slot` is the vtable index for a computer method, or -1 for an export.
void warn_once(std::atomic<bool>& warned, const char* what, int slot, const void* caller, const char* why) {
    if (warned.exchange(true, std::memory_order_relaxed)) return;
    char name[64];
    if (slot < 0) std::snprintf(name, sizeof(name), "%s", what);
    else std::snprintf(name, sizeof(name), "%s%d", what, slot);
    std::fprintf(stderr, "[rt] xg_xs: %s: %s (caller %s, logged once)\n", name, why,
                 rt::shims::module_offset(reinterpret_cast<uint64_t>(caller)).c_str());
}

// The mode the console's library picks for a description: linear for 1D, the XOR render and depth modes for targets,
// else 4 KiB standard. VIctory's reading of the library's older optimal-mode function (any size, 2D, 3D and cubes
// alike); revisit if streamed textures come out scrambled.
uint32_t suggested(const ResourceDesc& d) {
    if (d.dimension == kTexture1d) return ct::kLinear;
    if (d.flags & kAllowDepthStencil) return ct::kSw64kbZX;
    if (d.flags & kAllowRenderTarget) return ct::kSw64kbRX;
    return ct::kSw4kbS;
}

std::optional<ct::Texture> texture_of(const ResourceDesc& d) {
    ct::TextureDesc t;
    switch (d.dimension) {
    case kTexture1d: t.dimension = ct::Dimension::tex1d; break;
    case kTexture2d: t.dimension = ct::Dimension::tex2d; break;
    case kTexture3d: t.dimension = ct::Dimension::tex3d; break;
    default: return std::nullopt;
    }
    if (d.layout == 0) t.swizzle = suggested(d);
    else if (d.layout == kRowMajor) t.swizzle = ct::kLinear;
    else if ((d.layout & ~0xffu) == kSwizzled) t.swizzle = d.layout & 0xffu;
    else return std::nullopt;
    if (d.width > std::numeric_limits<uint32_t>::max()) return std::nullopt;
    t.width = static_cast<uint32_t>(d.width);
    t.height = d.height;
    t.depth_or_array = d.depth_or_array;
    t.mips = d.mips;
    t.format = d.format;
    t.samples = d.samples ? d.samples : 1;
    auto texture = ct::Texture::make(t);
    // The layout structure has room for 15 mips.
    if (!texture || texture->planes[0].layout.mips.size() > kMips) return std::nullopt;
    return texture;
}

// ---- The computer with a layout. ----

struct Computer {
    const void* const* vtable;
    std::atomic<ULONG> refs{1};
    ResourceDesc desc{};
    uint32_t mips = 0;
    ct::Texture texture;
};

ULONG __stdcall add_ref(Computer* c) { return c->refs.fetch_add(1, std::memory_order_relaxed) + 1; }

ULONG __stdcall release(Computer* c) {
    const ULONG left = c->refs.fetch_sub(1, std::memory_order_acq_rel) - 1;
    if (left == 0) delete c;
    return left;
}

// The mode a tail mip is reported in: the 256-byte one of its kind (3 for the render kind, which GFX10 lacks), the
// depth mode itself.
uint32_t tail_swizzle(uint32_t s) {
    switch (s) {
    case 2: case 6: case 10: case 18: case 22: case 26: return ct::kSw256bD;
    case 23: case 27: return 3;
    case 24: return s;
    default: return ct::kSw256bS;
    }
}

MipOut mip_out(const ResourceDesc& desc, const ct::Layout& l, size_t i) {
    const ct::Mip& m = l.mips[i];
    const uint64_t bpe = l.desc.element_bytes;
    const uint32_t block = l.desc.block;
    const bool three_d = desc.dimension == kTexture3d;
    const uint32_t array = desc.depth_or_array ? desc.depth_or_array : 1u;
    MipOut out{};
    out.pitch_texels = m.pitch * block;
    out.pitch_bytes = m.pitch * l.desc.element_bytes;
    out.padded[0] = m.pitch;
    out.padded[1] = m.rows;
    out.padded[2] = three_d ? m.depth_pitch : array;
    out.elements[0] = m.width;
    out.elements[1] = m.height;
    const uint64_t w = desc.width >> i;
    out.texels[0] = w ? static_cast<uint32_t>(w) : 1u;
    out.texels[1] = (desc.height >> i) ? (desc.height >> i) : 1u;
    out.depth_or_array = three_d ? m.depth : array;
    out.samples = l.desc.samples;
    out.slice_depth = l.block[2];
    const uint64_t slice = uint64_t{m.pitch} * m.rows * bpe;
    const uint64_t depth = three_d ? m.depth_pitch : 1u;
    if (!m.tail) {
        out.offset = m.offset;
        out.slice_size = slice * l.desc.samples;
        out.size = out.slice_size * depth;
        out.alignment = static_cast<uint32_t>(l.alignment);
        out.swizzle = l.desc.swizzle;
    } else {
        out.in_tail = 1;
        out.alignment = 256;
        out.swizzle = tail_swizzle(l.desc.swizzle);
        if (l.thick) {  // a 3D tail mip is placed by position, not by offset
            out.slice_size = l.alignment / l.block[2];
            out.tail_position = m.tail->x | (m.tail->y << 11);
        } else {
            out.offset = m.tail->offset;
            out.slice_size = slice;
        }
        out.size = out.slice_size * depth;
    }
    return out;
}

HRESULT __stdcall layout(Computer* c, LayoutOut* out) {
    if (!out) return kInvalidArg;
    std::memset(out, 0, sizeof(*out));
    out->size = c->texture.size;
    out->alignment = c->texture.alignment;
    out->mips = c->mips;
    out->planes = static_cast<uint32_t>(c->texture.planes.size());
    out->dimension = c->desc.dimension;
    // The library calls a depth plane "depth" only beside stencil or without mips.
    const bool depth = (c->desc.flags & kAllowDepthStencil) && (c->texture.planes.size() > 1 || c->mips == 1);
    for (size_t p = 0; p < c->texture.planes.size() && p < 4; ++p) {
        const ct::Plane& src = c->texture.planes[p];
        const ct::Layout& l = src.layout;
        PlaneOut& plane = out->plane[p];
        plane.usage = src.stencil ? 8u : depth ? 7u : 1u;
        plane.size = l.size;
        plane.slice_size = l.slice_size;
        plane.offset = src.offset;
        plane.alignment = l.alignment;
        plane.element_bytes = l.desc.element_bytes;
        plane.standard_mips = l.first_tail_mip;
        plane.tail_mips = c->mips - l.first_tail_mip;
        for (size_t i = 0; i < l.mips.size() && i < kMips; ++i) plane.mips[i] = mip_out(c->desc, l, i);
    }
    return S_OK;
}

// The plane, its mip and array slice for a subresource index, and the tiled bytes and the linear bytes a copy touches.
struct CopyArgs {
    const ct::Plane* plane;
    uint32_t mip, slice;
    size_t tiled, linear;
};

std::optional<CopyArgs> copy_args(const Computer* c, uint32_t plane, uint32_t subresource, uint32_t row,
                                  uint32_t depth) {
    if (plane >= c->texture.planes.size() || c->mips == 0) return std::nullopt;
    const ct::Plane& p = c->texture.planes[plane];
    const ct::Layout& l = p.layout;
    const uint32_t mip = subresource % c->mips, slice = subresource / c->mips;
    if (mip >= l.mips.size()) return std::nullopt;
    const ct::Mip& m = l.mips[mip];
    const uint64_t bytes = uint64_t{m.width} * l.desc.element_bytes;
    const uint64_t depths = l.subresource_depth(mip);
    // Every factor is at most 2^32, so each product fits in 64 bits; the sum is checked against size_t below.
    const uint64_t linear = (depths - 1) * depth + uint64_t{m.height - 1} * row + bytes;
    const uint64_t tiled = p.offset + l.size;
    if (linear > std::numeric_limits<size_t>::max() || tiled > std::numeric_limits<size_t>::max())
        return std::nullopt;
    return CopyArgs{&p, mip, slice, static_cast<size_t>(tiled), static_cast<size_t>(linear)};
}

HRESULT result(ct::Error e) {
    switch (e) {
    case ct::Error::ok: return S_OK;
    case ct::Error::unsupported: return kNotImpl;
    default: return kInvalidArg;
    }
}

void warn_copy(const Computer* c, const char* what, uint32_t subresource, ct::Error e) {
    static std::atomic<int> warned{0};
    if (warned.fetch_add(1, std::memory_order_relaxed) >= 8) return;
    std::fprintf(stderr,
                 "[rt] xg_xs: %s subresource %u failed (%s): dim=%u %llux%u depth/array=%u mips=%u format=%u "
                 "layout=0x%x\n",
                 what, subresource,
                 e == ct::Error::unsupported ? "addressing not modelled" : e == ct::Error::too_small ? "too small"
                                                                                                     : "invalid",
                 c->desc.dimension, static_cast<unsigned long long>(c->desc.width), c->desc.height,
                 c->desc.depth_or_array, c->desc.mips, c->desc.format, c->desc.layout);
}

// (tiled base, plane, subresource, linear, row pitch, depth pitch).
HRESULT __stdcall copy_into(Computer* c, uint8_t* tiled, uint32_t plane, uint32_t subresource, const uint8_t* linear,
                            uint32_t row, uint32_t depth) {
    const auto a = copy_args(c, plane, subresource, row, depth);
    if (!a || !tiled || !linear) return kInvalidArg;
    const size_t offset = static_cast<size_t>(a->plane->offset);
    const ct::Error e = a->plane->layout.tile(tiled + offset, a->tiled - offset, a->mip, a->slice, linear, a->linear,
                                              row, depth);
    if (e != ct::Error::ok) warn_copy(c, "copy into", subresource, e);
    // The texture's image on the PC GPU is filled from this memory: tell the D3D12 layer it changed.
    if (e == ct::Error::ok) leonida::title_texel_writes::note(tiled, a->tiled);
    return result(e);
}

// (linear, plane, subresource, sample, tiled base, row pitch, depth pitch).
HRESULT __stdcall copy_from(Computer* c, uint8_t* linear, uint32_t plane, uint32_t subresource, uint32_t /*sample*/,
                            const uint8_t* tiled, uint32_t row, uint32_t depth) {
    const auto a = copy_args(c, plane, subresource, row, depth);
    if (!a || !tiled || !linear) return kInvalidArg;
    const size_t offset = static_cast<size_t>(a->plane->offset);
    const ct::Error e = a->plane->layout.detile(tiled + offset, a->tiled - offset, a->mip, a->slice, linear, a->linear,
                                                row, depth);
    if (e != ct::Error::ok) warn_copy(c, "copy out of", subresource, e);
    return result(e);
}

// (total tiles, packed mips, tile shape in texels, in: room / out: count, first, subresource tilings), in 64 KiB
// tiles of plane 0, for 2D 64 KiB modes only (others fail, as on the console). The mip tail is the packed part, tile 0
// of each slice; tilings list the standard mips of every slice in turn. With a tail the console's library lists two
// slices at most and zeroes all but the first (VIctory's reading; the title reads slice 0 only).
HRESULT __stdcall tiling(Computer* c, uint32_t* tiles, PackedMips* packed, uint32_t* shape, uint32_t* count,
                         uint32_t first, SubresourceTiling* out) {
    const ct::Layout& l = c->texture.planes[0].layout;
    if (l.alignment != kTile) return kFail;
    if (l.thick) return kNotImpl;
    const uint32_t standard = l.first_tail_mip;
    const bool tail = standard < c->mips;
    const uint64_t tail_tiles = standard == 0 ? l.slice_size / kTile
                                : tail        ? l.mips[standard - 1].offset / kTile
                                              : 0;
    const uint32_t block = l.desc.block;
    if (tiles) *tiles = static_cast<uint32_t>(l.size / kTile);
    if (packed) {
        packed->standard = static_cast<uint8_t>(standard);
        packed->packed = static_cast<uint8_t>(c->mips - standard);
        packed->tiles = static_cast<uint32_t>(tail_tiles);
        packed->start = 0;
    }
    if (shape) {
        shape[0] = l.block[0] * block;
        shape[1] = l.block[1] * block;
        shape[2] = l.block[2];
    }
    if (count) {
        const uint32_t slices = tail ? std::min(l.slices, 2u) : l.slices;
        const uint64_t total = uint64_t{standard} * slices;
        const uint32_t wanted =
            static_cast<uint32_t>(std::min<uint64_t>(total > first ? total - first : 0, *count));
        for (uint32_t i = 0; i < wanted && out; ++i) {
            const uint32_t slice = (first + i) / standard, mip = (first + i) % standard;
            const ct::Mip& m = l.mips[mip];
            SubresourceTiling entry{};
            if (!(tail && slice > 0)) {
                entry.width = m.pitch / l.block[0];
                entry.height = static_cast<uint16_t>(m.rows / l.block[1]);
                entry.depth = 1;
                entry.start = static_cast<uint32_t>((uint64_t{slice} * l.slice_size + m.offset) / kTile);
            }
            out[i] = entry;
        }
        *count = out ? wanted : 0;
    }
    return S_OK;
}

// (plane, mip, x, y, z or slice, sample, bit position): an element's byte offset from the texture's start, 0 if
// outside. VIctory measured the title reading texels of a 128x4 4KB_S texture on the CPU through it.
uint64_t __stdcall element_offset(Computer* c, uint32_t plane, uint32_t mip, uint64_t x, uint32_t y, uint32_t z,
                                  uint32_t /*sample*/, uint32_t* bit) {
    if (bit) *bit = 0;
    if (plane >= c->texture.planes.size()) return 0;
    const ct::Plane& p = c->texture.planes[plane];
    const uint32_t x32 = x > std::numeric_limits<uint32_t>::max() ? std::numeric_limits<uint32_t>::max()
                                                                  : static_cast<uint32_t>(x);
    uint64_t at = 0;
    if (p.layout.offset(mip, x32, y, z, at) != ct::Error::ok) return 0;
    return p.offset + at;
}

// 3, 4: the size and the alignment slot 2 reports.
uint64_t __stdcall total_size(Computer* c) { return c->texture.size; }
uint64_t __stdcall total_alignment(Computer* c) { return c->texture.alignment; }

// 5 (plane, mip): the offset slot 2 reports for the mip, 0 outside.
uint64_t __stdcall mip_offset(Computer* c, uint32_t plane, uint32_t mip) {
    if (plane >= c->texture.planes.size() || plane >= 4) return 0;
    const ct::Layout& l = c->texture.planes[plane].layout;
    if (mip >= l.mips.size() || mip >= kMips) return 0;
    return mip_out(c->desc, l, mip).offset;
}

template <size_t Slot>
HRESULT __stdcall unused_slot(Computer*) {
    static std::atomic<bool> warned{false};
    warn_once(warned, "texture computer slot ", static_cast<int>(Slot), _ReturnAddress(),
              "method not implemented; E_NOTIMPL");
    return kNotImpl;
}

template <size_t... Slot>
constexpr auto make_vtable(std::index_sequence<Slot...>) {
    return std::array<void*, sizeof...(Slot)>{
        (Slot == 0    ? reinterpret_cast<void*>(&add_ref)
         : Slot == 1  ? reinterpret_cast<void*>(&release)
         : Slot == 2  ? reinterpret_cast<void*>(&layout)
         : Slot == 3  ? reinterpret_cast<void*>(&total_size)
         : Slot == 4  ? reinterpret_cast<void*>(&total_alignment)
         : Slot == 5  ? reinterpret_cast<void*>(&mip_offset)
         : Slot == 6  ? reinterpret_cast<void*>(&element_offset)
         : Slot == 8  ? reinterpret_cast<void*>(&copy_into)
         : Slot == 9  ? reinterpret_cast<void*>(&copy_from)
         : Slot == 10 ? reinterpret_cast<void*>(&tiling)
                      : reinterpret_cast<void*>(&unused_slot<Slot>))...};
}
const auto kVtable = make_vtable(std::make_index_sequence<kSlots>{});

// A computer for `desc`; on failure *out is left as it was.
HRESULT create(const ResourceDesc& desc, void** out, const void* caller) {
    std::optional<ct::Texture> texture = texture_of(desc);
    if (!texture) {
        static std::atomic<int> warned{0};
        if (warned.fetch_add(1, std::memory_order_relaxed) < 8)
            std::fprintf(stderr,
                         "[rt] xg_xs: XGCreateTextureComputer: no layout for dim=%u %llux%u depth/array=%u "
                         "mips=%u format=%u samples=%u layout=0x%x flags=0x%x (caller %s)\n",
                         desc.dimension, static_cast<unsigned long long>(desc.width), desc.height,
                         desc.depth_or_array, desc.mips, desc.format, desc.samples, desc.layout, desc.flags,
                         rt::shims::module_offset(reinterpret_cast<uint64_t>(caller)).c_str());
        return kNotImpl;
    }
    auto* c = new (std::nothrow) Computer;
    if (!c) return E_OUTOFMEMORY;
    c->vtable = kVtable.data();
    c->desc = desc;
    c->mips = static_cast<uint32_t>(texture->planes[0].layout.mips.size());
    c->texture = std::move(*texture);
    *out = c;
    return S_OK;
}

}  // namespace port

HRESULT __stdcall XGCreateTextureComputer(const void* desc, void** out) {
    static std::atomic<unsigned> calls{0};
    ResourceDesc d{};
    if (desc) std::memcpy(&d, desc, sizeof(d));
    Texture t;
    HRESULT hr = E_NOTIMPL;
    if (!out) hr = E_POINTER;
    else {
        // Two measured callers dereference the computer even after a failed creation.
        *out = &unavailable;
        if (!desc) hr = E_INVALIDARG;
        else if (build_texture(d, t)) {
            auto* computer = new (std::nothrow) Computer{kVtable.data(), {1}, t};
            hr = computer ? S_OK : E_OUTOFMEMORY;
            if (computer) *out = computer;
            // How the title lays out a texture of this shape (title_xg_layouts): what its CPU writes use.
            leonida::title_xg_layouts::note({d.dimension, d.format, d.height, d.depth_or_array, d.mips, d.width},
                                            d.layout);
        } else {
            // What xg_xs_layout.h cannot lay out, the second model may: targets, depth with stencil, other modes.
            port::ResourceDesc p;
            std::memcpy(&p, desc, sizeof(p));
            hr = port::create(p, out, _ReturnAddress());
        }
    }
    trace_desc(calls, "create", _ReturnAddress(), desc ? &d : nullptr, out, 0, hr, t.mode, t.layout.size);
    // A refused computer is not harmless: the title's texture lock reads the layout it would have written
    // without checking, so say which descriptions are refused (at most 32 lines).
    static std::atomic<unsigned> refused{0};
    if (desc && FAILED(hr) && refused.fetch_add(1, std::memory_order_relaxed) < 32)
        std::fprintf(stderr,
                     "[rt] xg_xs: XGCreateTextureComputer refuses dim=%u %llux%ux%u mips=%u fmt=%u samples=%u "
                     "layout=%#x flags=%#x (hr 0x%08x), caller %s\n",
                     d.dimension, static_cast<unsigned long long>(d.width), d.height, unsigned(d.depth_or_array),
                     unsigned(d.mips), d.format, d.samples, d.layout, d.flags, uint32_t(hr),
                     rt::shims::module_offset(reinterpret_cast<uint64_t>(_ReturnAddress())).c_str());
    // LOCAL DIAGNOSTIC: LEONIDA_XG_TRACE_FORMAT=<DXGI format> logs the first 256 computers of that format whatever
    // LEONIDA_XG_TRACE's cap, with the layout the title asked for (which tiling its CPU writes use).
    static const uint32_t only = [] {
        char value[8]{};
        return GetEnvironmentVariableA("LEONIDA_XG_TRACE_FORMAT", value, sizeof(value)) ? uint32_t(atoi(value)) : 0u;
    }();
    static std::atomic<unsigned> only_logged{0};
    if (only && desc && d.format == only && only_logged.fetch_add(1, std::memory_order_relaxed) < 256)
        std::fprintf(stderr,
                     "[rt] xg_xs: LOCAL DIAGNOSTIC: computer for format %u: dim=%u %llux%ux%u mips=%u layout=%#x "
                     "flags=%#x -> mode %u, %llu bytes, hr 0x%08x, caller %s\n",
                     d.format, d.dimension, static_cast<unsigned long long>(d.width), d.height,
                     unsigned(d.depth_or_array), unsigned(d.mips), d.layout, d.flags, t.mode,
                     static_cast<unsigned long long>(t.layout.size), uint32_t(hr),
                     rt::shims::module_offset(reinterpret_cast<uint64_t>(_ReturnAddress())).c_str());
    return hr;
}

HRESULT __stdcall XGSuggestSwizzleMode(const void* desc, uint32_t options, uint32_t* out) {
    static std::atomic<unsigned> calls{0};
    ResourceDesc d{};
    if (desc) std::memcpy(&d, desc, sizeof(d));
    HRESULT hr = E_INVALIDARG;
    uint32_t mode = 0;
    if (desc && out && d.dimension >= kTexture1D && d.dimension <= kTexture3D && d.width) {
        hr = E_NOTIMPL;
        if (!options) {
            mode = kSwizzled | select_mode(d);
            *out = mode;
            hr = S_OK;
        }
    }
    trace_desc(calls, "suggest", _ReturnAddress(), desc ? &d : nullptr, out, options, hr, mode, 0);
    return hr;
}

}  // namespace

SHIM_TABLE(xg_xs,
    {"xg_xs.dll", "XGCreateTextureComputer", reinterpret_cast<void*>(&XGCreateTextureComputer)},
    {"xg_xs.dll", "XGSuggestSwizzleMode", reinterpret_cast<void*>(&XGSuggestSwizzleMode)},
);
