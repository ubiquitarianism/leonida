// The blend state of the slot 57 record's state blob A (RT-SLOT57-BLEND-STATE): the shape check, the CB_TARGET_MASK
// and CB_BLENDn_CONTROL reads at their vertex and mesh offsets, and the RDNA2 -> D3D12 mapping. Synthetic blobs only:
// no game data, no D3D12. The register words are the forms docs/fspx.md "State blob A" counts in 1.0.2.1.
#include "../src/native_pso.h"

#include <cstdio>
#include <cstring>
#include <vector>

namespace np = leonida::native_pso;

namespace {

int g_failures = 0;

void check(bool ok, const char* what) {
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) ++g_failures;
}

void put(std::vector<std::uint8_t>& b, std::size_t at, std::uint32_t v) { std::memcpy(b.data() + at, &v, 4); }

// A blob of `size` bytes with the tag, CB_TARGET_MASK and CB_BLEND0..7 at the measured offsets for its family.
std::vector<std::uint8_t> blob_a(std::size_t size, std::uint32_t tag, std::uint32_t mask, std::uint32_t blend) {
    std::vector<std::uint8_t> b(size, 0);
    const std::size_t shift = size > 420 ? np::kStateAMeshShift : 0;
    put(b, 0, tag);
    put(b, np::kStateATargetMask + shift, mask);
    for (std::size_t i = 0; i < 8; ++i) put(b, np::kStateABlend + shift + 4 * i, blend + static_cast<std::uint32_t>(i));
    return b;
}

// D3D12_BLEND / D3D12_BLEND_OP values (d3d12.h).
enum : std::uint8_t { ZERO = 1, ONE = 2, SRC_ALPHA = 5, INV_SRC_ALPHA = 6, ADD = 1, SUBTRACT = 2, REV_SUBTRACT = 3,
                      MIN = 4, MAX = 5 };

bool same(const np::TargetBlend& t, bool enable, std::uint8_t s, std::uint8_t d, std::uint8_t op, std::uint8_t sa,
          std::uint8_t da, std::uint8_t opa, std::uint8_t mask) {
    return t.enable == enable && t.src == s && t.dst == d && t.op == op && t.src_alpha == sa && t.dst_alpha == da &&
           t.op_alpha == opa && t.write_mask == mask;
}

void decode() {
    // Both families and every measured size; the low bits 3 and 4 of the tag vary (0x4005d800/08/10).
    for (std::size_t size : {380u, 388u, 420u, 464u, 488u, 520u}) {
        const auto b = blob_a(size, 0x4005d808, 0x00ff, 0xe5010504);
        const auto r = np::decode_state_a_blend(b);
        check(r && r->target_mask == 0x00ff && r->blend[0] == 0xe5010504 && r->blend[7] == 0xe501050b,
              size > 420 ? "a mesh-family blob reads its registers 84 bytes further on"
                         : "a vertex-family blob reads CB_TARGET_MASK at +0x78 and CB_BLEND0..7 at +0xd0");
    }
    const char* why = nullptr;
    check(!np::decode_state_a_blend(blob_a(400, 0x4005d800, 0xf, 0x80000000), &why) && why &&
              std::strstr(why, "size"),
          "a size between the two families is not decoded");
    check(!np::decode_state_a_blend(blob_a(372, 0x4005d800, 0xf, 0x80000000)), "a shorter blob is not decoded");
    check(!np::decode_state_a_blend(blob_a(388, 0x4005d900, 0xf, 0x80000000), &why) && std::strstr(why, "tag"),
          "another tag is not decoded");
}

void mapping() {
    // The forms the library uses most (docs/fspx.md): off, alpha over, its straight-alpha variant, MAX, MIN.
    check(same(*np::d3d12_target_blend(0x80000000, 0xf), false, ONE, ZERO, ADD, ONE, ZERO, ADD, 0xf),
          "0x80000000 (ROP3 off only): blending off, every component written");
    check(same(*np::d3d12_target_blend(0xe5010504, 0xf), true, SRC_ALPHA, INV_SRC_ALPHA, ADD, ONE, INV_SRC_ALPHA, ADD,
               0xf),
          "0xe5010504: colour SRC_ALPHA/INV_SRC_ALPHA, alpha ONE/INV_SRC_ALPHA");
    check(same(*np::d3d12_target_blend(0xe5040504, 0xf), true, SRC_ALPHA, INV_SRC_ALPHA, ADD, SRC_ALPHA, INV_SRC_ALPHA,
               ADD, 0xf),
          "0xe5040504: alpha SRC_ALPHA/INV_SRC_ALPHA too");
    check(same(*np::d3d12_target_blend(0xe0010161, 0xf), true, ONE, ONE, MAX, ONE, ZERO, ADD, 0xf),
          "0xe0010161: colour MAX, alpha ONE/ZERO ADD");
    check(same(*np::d3d12_target_blend(0xe1410141, 0x3), true, ONE, ONE, MIN, ONE, ONE, MIN, 0x3),
          "0xe1410141: MIN on both, and the target's mask nibble is its write mask");
    // COMB_FCN 1 is source minus destination, 4 destination minus source.
    check(np::d3d12_target_blend(0x60000121, 0xf)->op == SUBTRACT, "COMB_FCN 1 is D3D12_BLEND_OP_SUBTRACT");
    check(np::d3d12_target_blend(0x60000181, 0xf)->op == REV_SUBTRACT, "COMB_FCN 4 is D3D12_BLEND_OP_REV_SUBTRACT");
    // Without SEPARATE_ALPHA_BLEND (bit 29) alpha takes the colour fields, a colour factor naming its alpha.
    check(same(*np::d3d12_target_blend(0x40000504, 0xf), true, SRC_ALPHA, INV_SRC_ALPHA, ADD, SRC_ALPHA, INV_SRC_ALPHA,
               ADD, 0xf),
          "no SEPARATE_ALPHA_BLEND: alpha uses the colour factors");
    const auto colour = np::d3d12_target_blend(0x40000302, 0xf);
    check(colour && colour->src == 3 && colour->dst == 4 && colour->src_alpha == SRC_ALPHA &&
              colour->dst_alpha == INV_SRC_ALPHA,
          "SRC_COLOR/INV_SRC_COLOR stay colour factors, and alpha reads them as SRC_ALPHA/INV_SRC_ALPHA");
    // Dual-source (RT-DUAL-SOURCE-BLEND): 0xf1010f01 is colour ONE/SRC1_COLOR, alpha ONE/SRC1_ALPHA.
    check(same(*np::d3d12_target_blend(0xf1010f01, 0xf), true, ONE, 16, ADD, ONE, 18, ADD, 0xf),
          "0xf1010f01: D3D12_BLEND_SRC1_COLOR and D3D12_BLEND_SRC1_ALPHA");
    check(np::d3d12_target_blend(0xf2011001, 0xf)->dst == 17 && np::d3d12_target_blend(0xf2011001, 0xf)->dst_alpha == 19,
          "0xf2011001: INV_SRC1_COLOR and INV_SRC1_ALPHA");
    check(np::d3d12_target_blend(0x40000f0f, 0xf)->src_alpha == 18,
          "without SEPARATE_ALPHA_BLEND, alpha reads SRC1_COLOR as SRC1_ALPHA");
    // Not built here: BOTH_SRC_ALPHA (not on gfx10) and a COMB_FCN past 4.
    check(!np::d3d12_target_blend(0x6000010b, 0xf), "BOTH_SRC_ALPHA is not built");
    check(!np::d3d12_target_blend(0x600001a1, 0xf), "COMB_FCN 5 is not built");
    // A disabled target with a dual-source pattern left in it is still a plain disabled target.
    check(np::d3d12_target_blend(0xb1010f01, 0xf) && !np::d3d12_target_blend(0xb1010f01, 0xf)->enable,
          "fields of a disabled target are not read");
}

void pipeline() {
    np::BlendRegisters r;
    r.target_mask = 0x01f;  // target 0: R G B A, target 1: R, target 2: none
    r.blend = {0xe5010504, 0x80000000, 0x80000000, 0x6000010b, 0x6000010b, 0x6000010b, 0x6000010b, 0x6000010b};
    const auto t = np::d3d12_blend(r, 3);
    check(t && (*t)[0].enable && (*t)[0].write_mask == 0xf && !(*t)[1].enable && (*t)[1].write_mask == 0x1 &&
              (*t)[2].write_mask == 0,
          "each target gets its own CB_BLENDn and its CB_TARGET_MASK nibble");
    check(t && !(*t)[3].enable && (*t)[3].write_mask == 0xf, "targets past the pipeline's count keep the defaults");
    check(!np::d3d12_blend(r, 4), "one target this path cannot build keeps the whole pipeline on the defaults");

    // A library pipeline writes R32_FLOAT, B4G4R4A4_UNORM and R8_UINT with MAX on every target (CB_BLEND 0xe1610161):
    // the integer target keeps its write mask and does not blend, which D3D12 would refuse.
    np::BlendRegisters mixed;
    mixed.target_mask = 0xff1;
    mixed.blend.fill(0xe1610161);
    const auto m = np::d3d12_blend(mixed, 3, {41, 115, 62, 0, 0, 0, 0, 0});
    check(m && (*m)[0].enable && (*m)[1].enable && !(*m)[2].enable && (*m)[2].write_mask == 0xf &&
              (*m)[0].write_mask == 0x1,
          "an integer target (R8_UINT) does not blend; the float and unorm ones do");
    check(np::dxgi_integer_format(62) && np::dxgi_integer_format(43) && !np::dxgi_integer_format(41) &&
              !np::dxgi_integer_format(28) && !np::dxgi_integer_format(115),
          "R8_UINT and R32_SINT are integer formats; R32_FLOAT, R8G8B8A8_UNORM and B4G4R4A4_UNORM are not");

    // plan() reads blob A beside blob B. A description without stages is refused as unsupported, after both blobs.
    std::vector<std::uint8_t> b(np::kStateBHeader, 0);
    b[0x25] = 1;
    const auto a = blob_a(388, 0x4005d800, 0xf, 0xe5010504);
    np::GraphicsDesc desc{};
    desc.state_a = {a.data(), a.size()};
    desc.state_b = {b.data(), b.size()};
    np::HashMap map;
    auto p = np::plan(desc, map);
    check(p.state.blend && p.state.blend->blend[0] == 0xe5010504, "plan() decodes blob A into FixedState::blend");
    desc.state_a = {};
    p = np::plan(desc, map);
    check(!p.state.blend, "without blob A, FixedState::blend stays empty");
}

}  // namespace

int main() {
    decode();
    mapping();
    pipeline();
    std::printf("%s\n", g_failures ? "FAILED" : "all passed");
    return g_failures ? 1 : 0;
}
