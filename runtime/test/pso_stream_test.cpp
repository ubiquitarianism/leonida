// Pipeline state stream copies and the derived-state patch (device slot 55) with synthetic streams only: no D3D12
// device, no game data. The layout rule is the one measured on the console's slot 86 stream (see the offsets check).
#include <d3d12.h>
#include <psapi.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <set>
#include <thread>
#include <vector>

#include "d3d12_xs_pso_stream.h"

namespace {

int g_failures = 0;

void check(bool ok, const char* what) {
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) ++g_failures;
}

std::size_t align_up(std::size_t v, std::size_t a) { return (v + a - 1) / a * a; }

// Builds a stream the way the console lays it out: u32 type, payload at its natural alignment, next entry 8-aligned.
struct StreamBuilder {
    std::vector<unsigned char> bytes;
    void entry(unsigned type, const void* payload, std::size_t size, std::size_t payload_align) {
        const std::size_t at = bytes.size();
        bytes.resize(at + 4);
        std::memcpy(bytes.data() + at, &type, 4);
        const std::size_t pay = align_up(at + 4, payload_align);
        bytes.resize(pay, 0);
        const auto* p = static_cast<const unsigned char*>(payload);
        bytes.insert(bytes.end(), p, p + size);
        bytes.resize(align_up(bytes.size(), 8), 0);
    }
};

// The test's own copy of the layout rule: payload offset of the first subobject of `wanted`, or npos.
std::size_t payload_of_type(const std::vector<unsigned char>& s, unsigned wanted) {
    std::size_t off = 0;
    while (off + 4 <= s.size()) {
        unsigned type;
        std::memcpy(&type, s.data() + off, 4);
        std::size_t size = sizeof(D3D12_SHADER_BYTECODE), align = 8;
        if (type == D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_BLEND) { size = sizeof(D3D12_BLEND_DESC); align = 4; }
        else if (type == D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_MASK) { size = 4; align = 4; }
        else if (type == D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER) { size = sizeof(D3D12_RASTERIZER_DESC); align = 4; }
        else if (type == D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL1) { size = sizeof(D3D12_DEPTH_STENCIL_DESC1); align = 4; }
        else if (type == D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL2) { size = sizeof(D3D12_DEPTH_STENCIL_DESC2); align = 4; }
        else if (type == D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER1) { size = sizeof(D3D12_RASTERIZER_DESC1); align = 4; }
        else if (type == D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER2) { size = sizeof(D3D12_RASTERIZER_DESC2); align = 4; }
        else if (type == D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE) { size = 8; align = 8; }
        else if (type == D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_INPUT_LAYOUT) { size = sizeof(D3D12_INPUT_LAYOUT_DESC); align = 8; }
        const std::size_t pay = align_up(off + 4, align);
        if (type == wanted) return pay;
        off = align_up(pay + size, 8);
    }
    return static_cast<std::size_t>(-1);
}

const unsigned char kVsCode[32] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                                   17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32};
const char kSemantic[] = "POSITION";

D3D12_INPUT_ELEMENT_DESC make_element(const char* name) {
    D3D12_INPUT_ELEMENT_DESC e{};
    e.SemanticName = name;
    e.Format = DXGI_FORMAT_R32G32B32_FLOAT;
    return e;
}

// Root signature (null), VS, BLEND, SAMPLE_MASK, RASTERIZER, DEPTH_STENCIL1, INPUT_LAYOUT.
std::vector<unsigned char> base_stream(const D3D12_INPUT_ELEMENT_DESC* elements, unsigned element_count) {
    StreamBuilder s;
    ID3D12RootSignature* root = nullptr;
    s.entry(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE, &root, 8, 8);
    D3D12_SHADER_BYTECODE vs{kVsCode, sizeof(kVsCode)};
    s.entry(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VS, &vs, sizeof(vs), 8);
    D3D12_BLEND_DESC blend{};
    s.entry(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_BLEND, &blend, sizeof(blend), 4);
    UINT mask = 0xffffffffu;
    s.entry(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_MASK, &mask, 4, 4);
    D3D12_RASTERIZER_DESC raster{};
    raster.FillMode = D3D12_FILL_MODE_SOLID;
    s.entry(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER, &raster, sizeof(raster), 4);
    D3D12_DEPTH_STENCIL_DESC1 ds1{};
    ds1.DepthEnable = TRUE;
    s.entry(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL1, &ds1, sizeof(ds1), 4);
    D3D12_INPUT_LAYOUT_DESC layout{elements, element_count};
    s.entry(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_INPUT_LAYOUT, &layout, sizeof(layout), 8);
    return s.bytes;
}

// One derived record: tag at +0, payload at +4.
std::vector<unsigned char> record(unsigned tag, const void* payload, std::size_t size) {
    std::vector<unsigned char> r(rt::d3d12_xs::kDerivedRecordSize, 0);
    std::memcpy(r.data(), &tag, 4);
    if (size) std::memcpy(r.data() + 4, payload, size);
    return r;
}

void test_layout_matches_console_offsets() {
    // Console slot 86 stream: types 0..5 then BLEND at +0x88 (payload +0x8c), SAMPLE_MASK at +0x1d8, RASTERIZER at
    // +0x1e0, DEPTH_STENCIL1 at +0x210 (measured in the notes, mesure sur l'exe).
    StreamBuilder s;
    ID3D12RootSignature* root = nullptr;
    s.entry(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE, &root, 8, 8);
    D3D12_SHADER_BYTECODE sb{};
    for (unsigned i = 0; i < 5; ++i) s.entry(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VS + i, &sb, sizeof(sb), 8);
    check(s.bytes.size() == 0x88, "six entries (types 0..5) end at +0x88");
    D3D12_BLEND_DESC blend{};
    s.entry(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_BLEND, &blend, sizeof(blend), 4);
    check(payload_of_type(s.bytes, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_BLEND) == 0x8c,
          "BLEND payload at +0x8c");
    check(s.bytes.size() == 0x1d8, "BLEND entry ends at +0x1d8");
    UINT mask = 1;
    s.entry(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_MASK, &mask, 4, 4);
    check(s.bytes.size() == 0x1e0, "SAMPLE_MASK entry is 8 bytes, next entry at +0x1e0");
}

void test_capture_and_deep_copy() {
    D3D12_INPUT_ELEMENT_DESC elements[1] = {make_element(kSemantic)};
    std::vector<unsigned char> base = base_stream(elements, 1);
    D3D12_PIPELINE_STATE_STREAM_DESC desc{base.size(), base.data()};
    auto rec = rt::d3d12_xs::PsoStreamRecord::capture(desc);
    check(rec != nullptr, "a synthetic base stream is captured");
    if (!rec) return;
    check(rec->bytes().size() == base.size(), "the copy has the stream's size");

    // Scribble over the source and its pointed-to data: the copy must still hold the original content.
    std::memset(base.data(), 0xee, base.size());
    elements[0].SemanticName = "XXXXXXXX";
    D3D12_SHADER_BYTECODE vs_copy{};
    std::memcpy(&vs_copy, rec->bytes().data() + payload_of_type(rec->bytes(), D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VS),
                sizeof(vs_copy));
    check(vs_copy.pShaderBytecode != kVsCode && vs_copy.BytecodeLength == sizeof(kVsCode) &&
              std::memcmp(vs_copy.pShaderBytecode, kVsCode, sizeof(kVsCode)) == 0,
          "shader bytecode is copied, not referenced");
    D3D12_INPUT_LAYOUT_DESC il{};
    std::memcpy(&il, rec->bytes().data() + payload_of_type(rec->bytes(), D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_INPUT_LAYOUT),
                sizeof(il));
    check(il.NumElements == 1 && std::strcmp(il.pInputElementDescs[0].SemanticName, kSemantic) == 0,
          "input layout semantic names are copied");
}

void test_derive_single_and_all_tags() {
    D3D12_INPUT_ELEMENT_DESC elements[1] = {make_element(kSemantic)};
    std::vector<unsigned char> base = base_stream(elements, 1);
    auto rec = rt::d3d12_xs::PsoStreamRecord::capture({base.size(), base.data()});
    if (!rec) { check(false, "base captured for derive tests"); return; }

    D3D12_BLEND_DESC blend{};
    blend.RenderTarget[0].RenderTargetWriteMask = 0x5;
    auto r2 = record(2, &blend, sizeof(blend));
    std::vector<unsigned char> out;
    check(rt::d3d12_xs::derive_stream(*rec, r2.data(), 1, out) == rt::d3d12_xs::DeriveStatus::ok,
          "tag 2 (blend) derives");
    const std::size_t bp = payload_of_type(out, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_BLEND);
    check(std::memcmp(out.data() + bp, &blend, sizeof(blend)) == 0, "tag 2 payload lands in the BLEND subobject");
    check(out.size() == rec->bytes().size(), "derived stream keeps the base size");

    D3D12_RASTERIZER_DESC raster{};
    raster.FillMode = D3D12_FILL_MODE_WIREFRAME;
    D3D12_DEPTH_STENCIL_DESC ds{};
    ds.DepthEnable = FALSE;
    UINT mask = 0x3;
    std::vector<unsigned char> three;
    auto r0 = record(0, &ds, sizeof(ds));
    auto r1 = record(1, &raster, sizeof(raster));
    auto r3 = record(3, &mask, 4);
    three.insert(three.end(), r0.begin(), r0.end());
    three.insert(three.end(), r1.begin(), r1.end());
    three.insert(three.end(), r3.begin(), r3.end());
    check(rt::d3d12_xs::derive_stream(*rec, three.data(), 3, out) == rt::d3d12_xs::DeriveStatus::ok,
          "three records (tags 0, 1, 3) derive");
    const std::size_t rp = payload_of_type(out, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER);
    const std::size_t mp = payload_of_type(out, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_MASK);
    const std::size_t dp = payload_of_type(out, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL1);
    check(std::memcmp(out.data() + rp, &raster, sizeof(raster)) == 0, "tag 1 payload lands in RASTERIZER");
    check(std::memcmp(out.data() + mp, &mask, 4) == 0, "tag 3 payload lands in SAMPLE_MASK");
    check(std::memcmp(out.data() + dp, &ds, sizeof(ds)) == 0, "tag 0 patches the DS1 prefix");
    const std::size_t tail = dp + sizeof(ds);
    check(out[tail] == 0 && out[tail + 1] == 0 && out[tail + 2] == 0 && out[tail + 3] == 0,
          "tag 0 keeps the DS1 depth-bounds field of the base");
}

void test_stencil_mask_record() {
    // Tag 5 carries the two stencil mask bytes at the DEPTH_STENCIL offsets 0x10/0x11 and zeros elsewhere (read from the
    // title's writes). It overlays only those two bytes of the base's DEPTH_STENCIL (or DEPTH_STENCIL1) payload.
    D3D12_INPUT_ELEMENT_DESC elements[1] = {make_element(kSemantic)};
    std::vector<unsigned char> base = base_stream(elements, 1);
    auto rec = rt::d3d12_xs::PsoStreamRecord::capture({base.size(), base.data()});
    if (!rec) { check(false, "base captured for the tag 5 tests"); return; }
    const std::size_t dp = payload_of_type(rec->bytes(), D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL1);

    unsigned char mask_payload[0x16] = {};
    mask_payload[0x10] = 0x0f;  // read mask
    mask_payload[0x11] = 0x3c;  // write mask
    auto r5 = record(5, mask_payload, sizeof(mask_payload));
    std::vector<unsigned char> out;
    check(rt::d3d12_xs::derive_stream(*rec, r5.data(), 1, out) == rt::d3d12_xs::DeriveStatus::ok,
          "tag 5 alone derives on a base with DEPTH_STENCIL1");
    check(out[dp + 0x10] == 0x0f && out[dp + 0x11] == 0x3c, "tag 5 overlays the two stencil mask bytes");
    bool rest_same = true;
    for (std::size_t i = 0; i < 0x38; ++i)
        if (i != 0x10 && i != 0x11 && out[dp + i] != rec->bytes()[dp + i]) rest_same = false;
    check(rest_same, "tag 5 leaves the other DEPTH_STENCIL1 bytes unchanged");

    // Order: records apply in the order they are given. Tag 0 writes its own masks, tag 5 overrides them.
    D3D12_DEPTH_STENCIL_DESC ds{};
    ds.StencilReadMask = 0x01;
    ds.StencilWriteMask = 0x02;
    auto r0 = record(0, &ds, sizeof(ds));
    std::vector<unsigned char> seq;
    seq.insert(seq.end(), r0.begin(), r0.end());
    seq.insert(seq.end(), r5.begin(), r5.end());
    check(rt::d3d12_xs::derive_stream(*rec, seq.data(), 2, out) == rt::d3d12_xs::DeriveStatus::ok,
          "tag 0 then tag 5 derive together");
    check(out[dp + 0x10] == 0x0f && out[dp + 0x11] == 0x3c, "tag 5 after tag 0 wins (record order)");

    seq.clear();
    seq.insert(seq.end(), r5.begin(), r5.end());
    seq.insert(seq.end(), r0.begin(), r0.end());
    check(rt::d3d12_xs::derive_stream(*rec, seq.data(), 2, out) == rt::d3d12_xs::DeriveStatus::ok,
          "tag 5 then tag 0 derive together");
    check(out[dp + 0x10] == 0x01 && out[dp + 0x11] == 0x02, "tag 0 after tag 5 wins (record order)");
}

void test_view_instancing_is_captured() {
    // The creation parser accepts a VIEW_INSTANCING subobject whose count is 0 (it refuses only a non-zero count).
    // Such a stream must also be captured: otherwise the PSO carries no kept stream and every derived PSO on it is
    // refused with "base PSO has no kept stream".
    D3D12_INPUT_ELEMENT_DESC elements[1] = {make_element(kSemantic)};
    std::vector<unsigned char> base = base_stream(elements, 1);
    const std::size_t off = base.size();
    const unsigned type = D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VIEW_INSTANCING;
    const std::size_t payload = align_up(off + 4, 8);
    base.resize(payload + sizeof(D3D12_VIEW_INSTANCING_DESC));
    std::memcpy(base.data() + off, &type, sizeof(type));
    D3D12_VIEW_INSTANCING_DESC vi{};
    vi.ViewInstanceCount = 0;
    vi.Flags = D3D12_VIEW_INSTANCING_FLAG_NONE;
    std::memcpy(base.data() + payload, &vi, sizeof(vi));

    auto rec = rt::d3d12_xs::PsoStreamRecord::capture({base.size(), base.data()});
    check(rec != nullptr, "a stream with a zero-count VIEW_INSTANCING subobject is captured");
    if (!rec) return;
    check(rec->bytes().size() == base.size(), "the copy has the stream's size");
    check(payload_of_type(rec->bytes(), type) == payload, "the VIEW_INSTANCING payload keeps its offset");

    // And a derived record still resolves against it, which is what the PSO needs.
    D3D12_RASTERIZER_DESC raster{};
    raster.FillMode = D3D12_FILL_MODE_WIREFRAME;
    auto r1 = record(1, &raster, sizeof(raster));
    std::vector<unsigned char> out;
    check(rt::d3d12_xs::derive_stream(*rec, r1.data(), 1, out) == rt::d3d12_xs::DeriveStatus::ok,
          "a derived stream still derives on a base with VIEW_INSTANCING");
}

void test_refusals() {
    D3D12_INPUT_ELEMENT_DESC elements[1] = {make_element(kSemantic)};
    std::vector<unsigned char> base = base_stream(elements, 1);
    auto rec = rt::d3d12_xs::PsoStreamRecord::capture({base.size(), base.data()});
    if (!rec) { check(false, "base captured for refusal tests"); return; }
    std::vector<unsigned char> out;

    auto r6 = record(6, nullptr, 0);
    check(rt::d3d12_xs::derive_stream(*rec, r6.data(), 1, out) == rt::d3d12_xs::DeriveStatus::unknown_tag,
          "tag 6 is refused (meaning not known)");

    // A base without a rasterizer subobject cannot take a tag-1 record.
    StreamBuilder s;
    ID3D12RootSignature* root = nullptr;
    s.entry(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE, &root, 8, 8);
    D3D12_BLEND_DESC blend{};
    s.entry(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_BLEND, &blend, sizeof(blend), 4);
    auto norast = rt::d3d12_xs::PsoStreamRecord::capture({s.bytes.size(), s.bytes.data()});
    if (!norast) { check(false, "stream without rasterizer captured"); return; }
    D3D12_RASTERIZER_DESC raster{};
    auto r1 = record(1, &raster, sizeof(raster));
    check(rt::d3d12_xs::derive_stream(*norast, r1.data(), 1, out) == rt::d3d12_xs::DeriveStatus::missing_state,
          "tag 1 on a base without RASTERIZER is refused");

    // A subobject type this module does not walk: VIEW_INSTANCING (payload with a pointer, not modelled).
    StreamBuilder v;
    unsigned char junk[32] = {};
    v.entry(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VIEW_INSTANCING, junk, sizeof(junk), 8);
    check(rt::d3d12_xs::PsoStreamRecord::capture({v.bytes.size(), v.bytes.data()}) == nullptr,
          "a stream with an unmodelled subobject type is not captured");
}

// A hand-built container of `size` bytes: the DXBC magic, 16 checksum bytes made from `checksum` (all zero when it is
// 0, as a container without a checksum has) and a body filled from `seed`. The record parses none of it.
std::vector<unsigned char> container(std::size_t size, unsigned char checksum, unsigned seed) {
    std::vector<unsigned char> bytes(size);
    std::memcpy(bytes.data(), "DXBC", 4);
    for (std::size_t i = 4; i < 20; ++i) bytes[i] = static_cast<unsigned char>(checksum ? checksum + i : 0);
    for (std::size_t i = 20; i < size; ++i) bytes[i] = static_cast<unsigned char>(seed * 131 + i * 7 + (i >> 8));
    return bytes;
}

// Root signature (null), VS and PS, as the title's slot 86 streams start.
std::vector<unsigned char> shader_stream(const std::vector<unsigned char>& vs, const std::vector<unsigned char>& ps) {
    StreamBuilder s;
    ID3D12RootSignature* root = nullptr;
    s.entry(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE, &root, 8, 8);
    D3D12_SHADER_BYTECODE code{vs.data(), vs.size()};
    s.entry(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VS, &code, sizeof(code), 8);
    code = {ps.data(), ps.size()};
    s.entry(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS, &code, sizeof(code), 8);
    return s.bytes;
}

// Captures a stream the test built; the record reads the stream and never writes it.
std::unique_ptr<rt::d3d12_xs::PsoStreamRecord> keep(const std::vector<unsigned char>& stream) {
    return rt::d3d12_xs::PsoStreamRecord::capture({stream.size(), const_cast<unsigned char*>(stream.data())});
}

D3D12_SHADER_BYTECODE stage_of(const rt::d3d12_xs::PsoStreamRecord& record, unsigned type) {
    D3D12_SHADER_BYTECODE code{};
    std::memcpy(&code, record.bytes().data() + payload_of_type(record.bytes(), type), sizeof(code));
    return code;
}

bool holds(const D3D12_SHADER_BYTECODE& code, const std::vector<unsigned char>& bytes) {
    return code.pShaderBytecode && code.BytecodeLength == bytes.size() &&
           std::memcmp(code.pShaderBytecode, bytes.data(), bytes.size()) == 0;
}

std::uint64_t private_bytes() {
    PROCESS_MEMORY_COUNTERS_EX counters{};
    counters.cb = sizeof(counters);
    if (!GetProcessMemoryInfo(GetCurrentProcess(), reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&counters),
                              sizeof(counters)))
        return 0;
    return counters.PrivateUsage;
}

void test_equal_bytecode_is_kept_once() {
    // The title creates tens of thousands of PSOs from far fewer shaders, and every slot 86 stream points at the
    // bytecode it was made from. Equal bytecode in many streams, from different source buffers, is kept once.
    constexpr std::size_t kMeanContainer = 9800;  // the corpus mean is about 9.6 KB per container
    const auto vs_a = container(kMeanContainer, 0x40, 1);
    const auto vs_a_again = vs_a;  // another buffer with the same bytes, as a second load of the shader would be
    const auto ps = container(kMeanContainer + 512, 0x80, 2);
    const auto ps_again = ps;
    const auto first_stream = shader_stream(vs_a, ps);
    const auto second_stream = shader_stream(vs_a_again, ps_again);

    auto first = keep(first_stream);
    auto second = keep(second_stream);
    if (!first || !second) { check(false, "shader streams captured"); return; }
    const auto vs1 = stage_of(*first, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VS);
    const auto vs2 = stage_of(*second, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VS);
    const auto ps1 = stage_of(*first, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS);
    const auto ps2 = stage_of(*second, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS);
    check(holds(vs1, vs_a) && holds(vs2, vs_a) && holds(ps1, ps) && holds(ps2, ps),
          "every stage of the kept streams holds the source bytes");
    check(vs1.pShaderBytecode != vs_a.data() && vs2.pShaderBytecode != vs_a_again.data(),
          "the kept bytecode is not the caller's buffer");
    check(vs1.pShaderBytecode == vs2.pShaderBytecode && ps1.pShaderBytecode == ps2.pShaderBytecode,
          "two streams with equal bytecode in different buffers share one copy of each stage");
    check(vs1.pShaderBytecode != ps1.pShaderBytecode, "stages with different bytecode keep different copies");

    // Bytecode with the same DXBC checksum and size but other bytes (hand-built containers can share one) stays apart.
    auto vs_b = vs_a;
    vs_b[kMeanContainer / 2] ^= 0xff;
    const auto third_stream = shader_stream(vs_b, ps);
    auto third = keep(third_stream);
    if (!third) { check(false, "the equal-checksum stream is captured"); return; }
    const auto vs3 = stage_of(*third, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VS);
    check(vs3.pShaderBytecode != vs1.pShaderBytecode && holds(vs3, vs_b) && holds(vs1, vs_a),
          "equal checksums with different bytes are not merged, and each keeps its own bytes");
    check(stage_of(*third, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS).pShaderBytecode == ps1.pShaderBytecode,
          "the third stream's PS, equal to the others, is still shared");

    // Containers without a checksum (all zero, as hand-built ones have) are told apart by their bytes too.
    const auto bare = container(4096, 0, 3);
    const auto bare_again = bare;
    auto bare_other = bare;
    bare_other.back() ^= 0x01;
    const auto s4 = shader_stream(bare, ps), s5 = shader_stream(bare_again, ps), s6 = shader_stream(bare_other, ps);
    auto r4 = keep(s4);
    auto r5 = keep(s5);
    auto r6 = keep(s6);
    if (!r4 || !r5 || !r6) { check(false, "the checksum-less streams are captured"); return; }
    const auto b4 = stage_of(*r4, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VS);
    const auto b5 = stage_of(*r5, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VS);
    const auto b6 = stage_of(*r6, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VS);
    check(b4.pShaderBytecode == b5.pShaderBytecode && holds(b4, bare), "equal checksum-less bytecode is shared");
    check(b6.pShaderBytecode != b4.pShaderBytecode && holds(b6, bare_other),
          "checksum-less bytecode that differs in its last byte is not merged");

    // A derived PSO keeps the base's copies, and a record keeps the shared bytes alive after the others are gone.
    std::vector<unsigned char> patched = first->bytes();
    auto derived = rt::d3d12_xs::PsoStreamRecord::derived(*first, std::move(patched));
    first.reset();
    second.reset();
    const auto vs_derived = stage_of(*derived, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VS);
    check(vs_derived.pShaderBytecode == vs1.pShaderBytecode && holds(vs_derived, vs_a),
          "a derived record shares the base's bytecode and keeps it after the base records are freed");
    derived.reset();
    auto again = keep(first_stream);
    check(again && holds(stage_of(*again, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VS), vs_a) &&
              holds(stage_of(*again, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS), ps),
          "bytecode captured again after every record that held it was freed holds the source bytes");
}

void test_many_streams_share_their_bytecode() {
    // N records of one VS and one PS: the streams' bytes are each record's own, the bytecode is kept once.
    constexpr unsigned kRecords = 10000;
    const auto vs = container(9800, 0x11, 4);
    const auto ps = container(9400, 0x22, 5);
    const auto stream = shader_stream(vs, ps);
    std::vector<std::unique_ptr<rt::d3d12_xs::PsoStreamRecord>> records;
    records.reserve(kRecords);
    const std::uint64_t before = private_bytes();
    for (unsigned i = 0; i < kRecords; ++i)
        if (auto record = keep(stream))
            records.push_back(std::move(record));
    const std::uint64_t after = private_bytes();
    check(records.size() == kRecords, "10000 streams are captured");
    if (records.size() != kRecords) return;
    const void* vs_kept = stage_of(*records[0], D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VS).pShaderBytecode;
    const void* ps_kept = stage_of(*records[0], D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS).pShaderBytecode;
    std::size_t vs_shared = 0, ps_shared = 0, intact = 0;
    for (const auto& record : records) {
        const auto v = stage_of(*record, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VS);
        const auto p = stage_of(*record, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS);
        vs_shared += v.pShaderBytecode == vs_kept;
        ps_shared += p.pShaderBytecode == ps_kept;
        intact += holds(v, vs) && holds(p, ps) && record->bytes().size() == stream.size();
    }
    check(intact == kRecords, "every one of the 10000 records holds the source bytecode and stream size");
    check(vs_shared == kRecords && ps_shared == kRecords, "the 10000 records share one copy of each stage");
    std::printf("INFO 10000 records of a %zu + %zu byte VS/PS: private bytes %+.1f MiB (%.1f MiB if each record kept "
                "its own bytecode)\n",
                vs.size(), ps.size(), (double(after) - double(before)) / (1024.0 * 1024.0),
                double(kRecords) * double(vs.size() + ps.size()) / (1024.0 * 1024.0));
}

void test_streams_from_many_threads() {
    // PSOs are created on many threads. Records made and freed side by side from the same three containers each hold
    // their own source bytes, and the ones still alive at the end share one copy per content.
    const std::vector<unsigned char> shaders[3] = {container(6000, 0x31, 6), container(7000, 0x32, 7),
                                                   container(8000, 0x33, 8)};
    const std::vector<unsigned char> streams[3] = {shader_stream(shaders[0], shaders[1]),
                                                   shader_stream(shaders[1], shaders[2]),
                                                   shader_stream(shaders[2], shaders[0])};
    constexpr unsigned kThreads = 8, kRounds = 2000;
    std::vector<std::vector<std::unique_ptr<rt::d3d12_xs::PsoStreamRecord>>> kept(kThreads);
    std::atomic<unsigned> wrong{0};
    std::vector<std::thread> threads;
    for (unsigned t = 0; t < kThreads; ++t)
        threads.emplace_back([&, t] {
            for (unsigned i = 0; i < kRounds; ++i) {
                const unsigned which = (t + i) % 3;
                auto record = keep(streams[which]);
                if (!record || !holds(stage_of(*record, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VS), shaders[which]) ||
                    !holds(stage_of(*record, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS), shaders[(which + 1) % 3]))
                    ++wrong;
                if (i % 16 == t % 16) kept[t].push_back(std::move(record));  // the others are freed at once
            }
        });
    for (auto& thread : threads) thread.join();
    std::set<const void*> copies;
    bool intact = true;
    for (const auto& list : kept)
        for (const auto& record : list) {
            const auto v = stage_of(*record, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VS);
            const auto p = stage_of(*record, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS);
            copies.insert(v.pShaderBytecode);
            copies.insert(p.pShaderBytecode);
            intact = intact && (holds(v, shaders[0]) || holds(v, shaders[1]) || holds(v, shaders[2]));
        }
    check(wrong.load() == 0 && intact, "records made and freed on 8 threads each hold their source bytecode");
    check(copies.size() == 3, "the records alive after 8 threads share one copy per container (3 in all)");
}

// Tags 0, 1 and 5 on a base whose depth-stencil is DEPTH_STENCIL2 and whose rasterizer is RASTERIZER1 or RASTERIZER2:
// the record's D3D12 v0 description is converted into the later version (RT-STATE-PSO-OVERLAY-DS2-RS12).
void test_later_subobject_versions() {
    for (const bool raster2 : {false, true}) {
        StreamBuilder s;
        ID3D12RootSignature* root = nullptr;
        s.entry(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE, &root, 8, 8);
        D3D12_DEPTH_STENCIL_DESC2 base_ds{};
        base_ds.DepthEnable = TRUE;
        base_ds.FrontFace.StencilReadMask = 0x11;
        base_ds.BackFace.StencilReadMask = 0x22;
        base_ds.DepthBoundsTestEnable = TRUE;
        s.entry(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL2, &base_ds, sizeof(base_ds), 4);
        unsigned raster_type;
        if (raster2) {
            D3D12_RASTERIZER_DESC2 r{};
            r.FillMode = D3D12_FILL_MODE_SOLID;
            raster_type = D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER2;
            s.entry(raster_type, &r, sizeof(r), 4);
        } else {
            D3D12_RASTERIZER_DESC1 r{};
            r.FillMode = D3D12_FILL_MODE_SOLID;
            raster_type = D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER1;
            s.entry(raster_type, &r, sizeof(r), 4);
        }
        auto rec = rt::d3d12_xs::PsoStreamRecord::capture({s.bytes.size(), s.bytes.data()});
        if (!rec) { check(false, "base with DEPTH_STENCIL2 and RASTERIZER1/2 captured"); return; }
        const std::size_t dp = payload_of_type(rec->bytes(), D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL2);
        const std::size_t rp = payload_of_type(rec->bytes(), raster_type);

        D3D12_DEPTH_STENCIL_DESC ds{};
        ds.DepthEnable = FALSE;
        ds.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ZERO;
        ds.DepthFunc = D3D12_COMPARISON_FUNC_GREATER;
        ds.StencilEnable = TRUE;
        ds.StencilReadMask = 0x0f;
        ds.StencilWriteMask = 0xf0;
        ds.FrontFace = {D3D12_STENCIL_OP_ZERO, D3D12_STENCIL_OP_INCR, D3D12_STENCIL_OP_REPLACE, D3D12_COMPARISON_FUNC_EQUAL};
        ds.BackFace = {D3D12_STENCIL_OP_INVERT, D3D12_STENCIL_OP_DECR, D3D12_STENCIL_OP_KEEP, D3D12_COMPARISON_FUNC_LESS};
        D3D12_RASTERIZER_DESC raster{};
        raster.FillMode = D3D12_FILL_MODE_WIREFRAME;
        raster.CullMode = D3D12_CULL_MODE_FRONT;
        raster.FrontCounterClockwise = TRUE;
        raster.DepthBias = 7;
        raster.DepthBiasClamp = 0.5f;
        raster.SlopeScaledDepthBias = 2.0f;
        raster.DepthClipEnable = TRUE;
        raster.AntialiasedLineEnable = TRUE;
        raster.ForcedSampleCount = 4;
        auto r0 = record(0, &ds, sizeof(ds));
        auto r1 = record(1, &raster, sizeof(raster));
        std::vector<unsigned char> both;
        both.insert(both.end(), r0.begin(), r0.end());
        both.insert(both.end(), r1.begin(), r1.end());
        std::vector<unsigned char> out;
        check(rt::d3d12_xs::derive_stream(*rec, both.data(), 2, out) == rt::d3d12_xs::DeriveStatus::ok,
              raster2 ? "tags 0 and 1 derive on DEPTH_STENCIL2 and RASTERIZER2"
                      : "tags 0 and 1 derive on DEPTH_STENCIL2 and RASTERIZER1");
        if (out.size() != rec->bytes().size()) continue;

        D3D12_DEPTH_STENCIL_DESC2 d2;
        std::memcpy(&d2, out.data() + dp, sizeof(d2));
        check(!d2.DepthEnable && d2.DepthWriteMask == D3D12_DEPTH_WRITE_MASK_ZERO &&
                  d2.DepthFunc == D3D12_COMPARISON_FUNC_GREATER && d2.StencilEnable,
              "tag 0 sets the DEPTH_STENCIL2 depth fields");
        check(d2.FrontFace.StencilFailOp == D3D12_STENCIL_OP_ZERO && d2.FrontFace.StencilPassOp == D3D12_STENCIL_OP_REPLACE &&
                  d2.FrontFace.StencilFunc == D3D12_COMPARISON_FUNC_EQUAL &&
                  d2.BackFace.StencilFailOp == D3D12_STENCIL_OP_INVERT &&
                  d2.BackFace.StencilDepthFailOp == D3D12_STENCIL_OP_DECR &&
                  d2.BackFace.StencilFunc == D3D12_COMPARISON_FUNC_LESS,
              "tag 0 sets the DEPTH_STENCIL2 face ops");
        check(d2.FrontFace.StencilReadMask == 0x0f && d2.BackFace.StencilReadMask == 0x0f &&
                  d2.FrontFace.StencilWriteMask == 0xf0 && d2.BackFace.StencilWriteMask == 0xf0,
              "tag 0 puts its one pair of masks on both DEPTH_STENCIL2 faces");
        check(d2.DepthBoundsTestEnable == TRUE, "tag 0 keeps the DEPTH_STENCIL2 depth-bounds field of the base");

        if (raster2) {
            D3D12_RASTERIZER_DESC2 r2;
            std::memcpy(&r2, out.data() + rp, sizeof(r2));
            check(r2.FillMode == D3D12_FILL_MODE_WIREFRAME && r2.CullMode == D3D12_CULL_MODE_FRONT &&
                      r2.FrontCounterClockwise && r2.DepthBias == 7.0f && r2.DepthBiasClamp == 0.5f &&
                      r2.SlopeScaledDepthBias == 2.0f && r2.DepthClipEnable && r2.ForcedSampleCount == 4 &&
                      r2.LineRasterizationMode == D3D12_LINE_RASTERIZATION_MODE_ALPHA_ANTIALIASED,
                  "tag 1 is converted into RASTERIZER2 (float bias, line mode from AntialiasedLineEnable)");
        } else {
            D3D12_RASTERIZER_DESC1 r1d;
            std::memcpy(&r1d, out.data() + rp, sizeof(r1d));
            check(r1d.FillMode == D3D12_FILL_MODE_WIREFRAME && r1d.CullMode == D3D12_CULL_MODE_FRONT &&
                      r1d.FrontCounterClockwise && r1d.DepthBias == 7.0f && r1d.DepthBiasClamp == 0.5f &&
                      r1d.SlopeScaledDepthBias == 2.0f && r1d.DepthClipEnable && r1d.AntialiasedLineEnable &&
                      r1d.ForcedSampleCount == 4,
                  "tag 1 is converted into RASTERIZER1 (float bias)");
        }

        // Tag 5 sets both faces' masks and nothing else.
        unsigned char mask_payload[0x16] = {};
        mask_payload[0x10] = 0x3c;
        mask_payload[0x11] = 0xc3;
        auto r5 = record(5, mask_payload, sizeof(mask_payload));
        check(rt::d3d12_xs::derive_stream(*rec, r5.data(), 1, out) == rt::d3d12_xs::DeriveStatus::ok,
              "tag 5 derives on a base with DEPTH_STENCIL2");
        if (out.size() != rec->bytes().size()) continue;
        std::memcpy(&d2, out.data() + dp, sizeof(d2));
        check(d2.FrontFace.StencilReadMask == 0x3c && d2.BackFace.StencilReadMask == 0x3c &&
                  d2.FrontFace.StencilWriteMask == 0xc3 && d2.BackFace.StencilWriteMask == 0xc3,
              "tag 5 sets both DEPTH_STENCIL2 faces' masks");
        check(d2.DepthEnable == TRUE && !d2.StencilEnable && d2.DepthBoundsTestEnable == TRUE &&
                  d2.FrontFace.StencilFailOp == base_ds.FrontFace.StencilFailOp &&
                  d2.BackFace.StencilFunc == base_ds.BackFace.StencilFunc,
              "tag 5 leaves the other DEPTH_STENCIL2 fields unchanged");
    }
}

}  // namespace

int main() {
    std::printf("pso_stream_test: derived pipeline state streams\n");
    test_layout_matches_console_offsets();
    test_capture_and_deep_copy();
    test_derive_single_and_all_tags();
    test_stencil_mask_record();
    test_view_instancing_is_captured();
    test_refusals();
    test_equal_bytecode_is_kept_once();
    test_many_streams_share_their_bytecode();
    test_streams_from_many_threads();
    test_later_subobject_versions();
    std::printf("%s: %d failure(s)\n", g_failures ? "FAILED" : "OK", g_failures);
    return g_failures ? 1 : 0;
}
