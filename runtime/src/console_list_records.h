#pragma once

// Replays the console command list's inline records as the vtable calls they replace (BL-3, docs/boot_leads.md).
//
// While the list has room, a console DLL writes a command into the list object instead of calling the vtable
// (docs/d3d12_command_objects.md §3.5a). Measured forms and their fallbacks:
//
//   0x4002C400, x, y, z                    14  Dispatch
//   0x4003CC00, x, y, z, w                 80  dispatch with an extra word
//   header + 4*offset, value               37  SetComputeRoot32BitConstant
//   header + ((n-1) << 16), n dwords       39  SetComputeRoot32BitConstants
//   header, 32-bit table address           35  SetComputeRootDescriptorTable
//   header, 64-bit VA                      41/43/45  root CBV/SRV/UAV
//   28-byte pipeline record                25  SetPipelineState
//
// `header` is the list's table entry at +0x344 + 4*root. The runtime fills that table, so root_header() defines it
// and decode() inverts it. Pending records must be replayed before any vtable call on the same list. An unknown
// record stops decoding rather than being skipped.

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <optional>
#include <span>
#include <vector>

namespace leonida::console_list {

inline constexpr std::uint32_t dispatch_header = 0x4002C400;   // + x, y, z
inline constexpr std::uint32_t dispatch_w_header = 0x4003CC00; // + x, y, z, w
inline constexpr std::size_t pipeline_record_bytes = 28;
inline constexpr std::uint32_t inline_blocked_bits = 0xC0000000; // set in an entry: the DLL calls the vtable

// Fallback slots (docs/d3d12_command_objects.md §3.3a, §3.5a).
namespace slot {
inline constexpr unsigned dispatch = 14;
inline constexpr unsigned set_pipeline_state = 25;
inline constexpr unsigned set_compute_root_descriptor_table = 35;
inline constexpr unsigned set_compute_root_32bit_constant = 37;
inline constexpr unsigned set_compute_root_32bit_constants = 39;
inline constexpr unsigned set_compute_root_cbv = 41;
inline constexpr unsigned set_compute_root_srv = 43;
inline constexpr unsigned set_compute_root_uav = 45;
inline constexpr unsigned dispatch_w = 80;
} // namespace slot

enum class RootKind : std::uint32_t { constants = 1, table = 2, cbv = 3, srv = 4, uav = 5 };

inline constexpr unsigned max_root_parameters = 64;
inline constexpr unsigned max_root_constants = 64;

// Table entry for `root`. Bits 2..7 and 16..21 are left 0 for the DLL's offset and count adjustments;
// 8..10 kind, 22..27 root index, 28 tag; 30..31 clear so the DLL may write inline.
inline constexpr std::uint32_t root_tag = 1u << 28;
inline constexpr std::uint32_t root_header(unsigned root, RootKind kind) {
    return root_tag | (static_cast<std::uint32_t>(root & 0x3F) << 22) | (static_cast<std::uint32_t>(kind) << 8);
}

// A fallback vtable call.
struct Call {
    unsigned slot = 0;
    std::array<std::uint64_t, 4> args{};
    std::vector<std::uint32_t> constants;  // slot 39: the pointed-to values
    std::uintptr_t pipeline = 0;           // slot 25
    bool operator==(const Call&) const = default;
};

struct DecodeError {
    std::size_t offset;  // of the record that stopped decoding
    const char* what;
};

struct Decoded {
    std::vector<Call> calls;
    std::optional<DecodeError> error;
};

// Pipeline record -> pipeline object, or 0 if not one.
using PipelineResolver = std::function<std::uintptr_t(std::span<const std::uint8_t, pipeline_record_bytes>)>;

namespace detail {
inline std::uint32_t word(std::span<const std::uint8_t> bytes, std::size_t at) {
    std::uint32_t w;
    std::memcpy(&w, bytes.data() + at, sizeof(w));
    return w;
}
} // namespace detail

// `bytes`: list start up to the cursor.
inline Decoded decode(std::span<const std::uint8_t> bytes, const PipelineResolver& resolve_pipeline = {}) {
    Decoded out;
    std::size_t at = 0;
    auto fail = [&](const char* what) { out.error = DecodeError{at, what}; return out; };
    auto have = [&](std::size_t n) { return bytes.size() - at >= n; };
    if (bytes.size() % 4) {
        at = bytes.size() & ~std::size_t{3};
        return fail("record bytes are not a whole number of dwords");
    }
    while (at < bytes.size()) {
        const std::uint32_t head = detail::word(bytes, at);
        if (head == dispatch_header || head == dispatch_w_header) {
            const bool w = head == dispatch_w_header;
            const std::size_t size = w ? 20 : 16;
            if (!have(size)) return fail("truncated dispatch record");
            Call c;
            c.slot = w ? slot::dispatch_w : slot::dispatch;
            for (unsigned i = 0; i < (w ? 4u : 3u); ++i) c.args[i] = detail::word(bytes, at + 4 + 4 * i);
            out.calls.push_back(std::move(c));
            at += size;
            continue;
        }
        if ((head & inline_blocked_bits) == 0 && (head & root_tag)) {
            const std::uint32_t root = (head >> 22) & 0x3F;
            const std::uint32_t kind = (head >> 8) & 0x7;
            const std::uint32_t count = ((head >> 16) & 0x3F) + 1;
            const std::uint32_t offset_bytes = head & 0xFC;
            if (head & 0x2000F803u) return fail("root-argument header has bits this runtime never sets");
            if (count > 1 && offset_bytes) return fail("root-argument header combines a count and an offset");
            Call c;
            c.args[0] = root;
            switch (static_cast<RootKind>(kind)) {
            case RootKind::constants:
                if (count == 1) {
                    if (!have(8)) return fail("truncated root-constant record");
                    c.slot = slot::set_compute_root_32bit_constant;
                    c.args[1] = detail::word(bytes, at + 4);
                    c.args[2] = offset_bytes / 4;
                    at += 8;
                } else {
                    if (!have(4 + 4 * std::size_t{count})) return fail("truncated root-constants record");
                    c.slot = slot::set_compute_root_32bit_constants;
                    c.args[1] = count;
                    for (std::uint32_t i = 0; i < count; ++i) c.constants.push_back(detail::word(bytes, at + 4 + 4 * i));
                    at += 4 + 4 * std::size_t{count};
                }
                break;
            case RootKind::table:
                if (count > 1 || offset_bytes) return fail("descriptor-table header carries a constant adjustment");
                if (!have(8)) return fail("truncated descriptor-table record");
                c.slot = slot::set_compute_root_descriptor_table;
                c.args[1] = detail::word(bytes, at + 4);  // 32-bit on the console; widening is BL-4
                at += 8;
                break;
            case RootKind::cbv:
            case RootKind::srv:
            case RootKind::uav: {
                if (count > 1 || offset_bytes) return fail("root-descriptor header carries a constant adjustment");
                if (!have(12)) return fail("truncated root-descriptor record");
                c.slot = kind == static_cast<std::uint32_t>(RootKind::cbv)   ? slot::set_compute_root_cbv
                         : kind == static_cast<std::uint32_t>(RootKind::srv) ? slot::set_compute_root_srv
                                                                             : slot::set_compute_root_uav;
                c.args[1] = detail::word(bytes, at + 4) | std::uint64_t{detail::word(bytes, at + 8)} << 32;
                at += 12;
                break;
            }
            default:
                return fail("root-argument header names no root-parameter kind");
            }
            out.calls.push_back(std::move(c));
            continue;
        }
        if (resolve_pipeline && have(pipeline_record_bytes)) {
            const std::uintptr_t pipeline =
                resolve_pipeline(std::span<const std::uint8_t, pipeline_record_bytes>(bytes.data() + at,
                                                                                      pipeline_record_bytes));
            if (pipeline) {
                Call c;
                c.slot = slot::set_pipeline_state;
                c.args[0] = pipeline;
                c.pipeline = pipeline;
                out.calls.push_back(std::move(c));
                at += pipeline_record_bytes;
                continue;
            }
        }
        return fail("unknown record header");
    }
    return out;
}

} // namespace leonida::console_list
