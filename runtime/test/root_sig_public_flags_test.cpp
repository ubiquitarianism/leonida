// The d3d12_xs root-signature flag mask leaves every public D3D12_ROOT_SIGNATURE_FLAGS value alone (old upstream
// PR 7140).
//
// The shims clear bits 24-31 (kXboxRootSignatureFlagHighByte) of a root signature's flags before the host serializer
// or deserializer sees them, on every descriptor and every RTS0 blob, whoever built it. Every public flag in d3d12.h
// is in the low 16 bits (the highest is SAMPLER_HEAP_DIRECTLY_INDEXED, 0x800), so a descriptor whose flags fit in 16
// bits must come out of the shims exactly as it went in. That is the case for the retail executable's root
// signatures (old upstream PR 7140: a 1.1 descriptor with Flags 0x80, and a decoder whose Flags is a zero-extended
// 16-bit read).
//
// Checked here:
// - the helpers, for every flag value in [0, 0xFFFF]: versioned 1.0/1.1/1.2 and legacy descriptors keep the value, and
//   an RTS0 blob carrying it is not rewritten;
// - the registered shims against the host d3d12.dll, for each public flag, 0x80 alone and all of them: the serializers
//   give the host's result byte for byte (1.0 and 1.1) wherever the host takes the description, the deserializer
//   returns the same flags. Where the host refuses a description only for the heap-indexed flags (Windows 10's in-box
//   d3d12.dll, kDirectlyIndexedFlags), the shims serialize it on purpose: their blob is the host's blob for the
//   flags without those bits, with the full flags in RTS0 and a valid container hash;
// - a blob carrying 0x03000000 on top of 0x80 has exactly those two bits cleared and nothing else changed.
#include <windows.h>
#include <d3d12.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "d3d12_xs_rootsig.h"
#include "shims.h"

namespace {

int g_failures = 0;

void check(bool condition, const char* what) {
    std::printf("%s %s\n", condition ? "PASS" : "FAIL", what);
    if (!condition) ++g_failures;
}

struct Blobs {
    ID3DBlob* blob = nullptr;
    ID3DBlob* error = nullptr;
    ~Blobs() {
        if (blob) blob->Release();
        if (error) error->Release();
    }
};

// Every D3D12_ROOT_SIGNATURE_FLAGS value in d3d12.h (Windows SDK 10.0.26100, DirectX-Headers 1.721).
constexpr D3D12_ROOT_SIGNATURE_FLAGS kPublicFlags[] = {
    D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT,
    D3D12_ROOT_SIGNATURE_FLAG_DENY_VERTEX_SHADER_ROOT_ACCESS,
    D3D12_ROOT_SIGNATURE_FLAG_DENY_HULL_SHADER_ROOT_ACCESS,
    D3D12_ROOT_SIGNATURE_FLAG_DENY_DOMAIN_SHADER_ROOT_ACCESS,
    D3D12_ROOT_SIGNATURE_FLAG_DENY_GEOMETRY_SHADER_ROOT_ACCESS,
    D3D12_ROOT_SIGNATURE_FLAG_DENY_PIXEL_SHADER_ROOT_ACCESS,
    D3D12_ROOT_SIGNATURE_FLAG_ALLOW_STREAM_OUTPUT,
    D3D12_ROOT_SIGNATURE_FLAG_LOCAL_ROOT_SIGNATURE,
    D3D12_ROOT_SIGNATURE_FLAG_DENY_AMPLIFICATION_SHADER_ROOT_ACCESS,
    D3D12_ROOT_SIGNATURE_FLAG_DENY_MESH_SHADER_ROOT_ACCESS,
    D3D12_ROOT_SIGNATURE_FLAG_CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED,
    D3D12_ROOT_SIGNATURE_FLAG_SAMPLER_HEAP_DIRECTLY_INDEXED,
};

constexpr uint32_t all_public_flags() {
    uint32_t all = 0;
    for (D3D12_ROOT_SIGNATURE_FLAGS f : kPublicFlags) all |= static_cast<uint32_t>(f);
    return all;
}

static_assert(all_public_flags() == 0xFFFu, "the public flags are the twelve bits 0x1..0x800");
static_assert((all_public_flags() & rt::d3d12_xs::kXboxRootSignatureFlagHighByte) == 0,
              "no public root-signature flag is in the byte the shims clear");

// The executable's direct call site (old upstream PR 7140): version 1.1, no parameters, no samplers, Flags 0x80.
constexpr uint32_t kExecutableFlags = 0x80u;
static_assert(kExecutableFlags == D3D12_ROOT_SIGNATURE_FLAG_LOCAL_ROOT_SIGNATURE, "0x80 is a public flag");

constexpr uint32_t kXboxFlags = 0x03000000u;

D3D12_ROOT_SIGNATURE_FLAGS as_flags(uint32_t v) { return static_cast<D3D12_ROOT_SIGNATURE_FLAGS>(v); }

D3D12_VERSIONED_ROOT_SIGNATURE_DESC versioned(D3D_ROOT_SIGNATURE_VERSION version, uint32_t flags) {
    D3D12_VERSIONED_ROOT_SIGNATURE_DESC d{};
    d.Version = version;
    switch (version) {
        case D3D_ROOT_SIGNATURE_VERSION_1_0: d.Desc_1_0.Flags = as_flags(flags); break;
        case D3D_ROOT_SIGNATURE_VERSION_1_1: d.Desc_1_1.Flags = as_flags(flags); break;
        default: d.Desc_1_2.Flags = as_flags(flags); break;
    }
    return d;
}

uint32_t flags_of(const D3D12_VERSIONED_ROOT_SIGNATURE_DESC& d) {
    switch (d.Version) {
        case D3D_ROOT_SIGNATURE_VERSION_1_0: return static_cast<uint32_t>(d.Desc_1_0.Flags);
        case D3D_ROOT_SIGNATURE_VERSION_1_1: return static_cast<uint32_t>(d.Desc_1_1.Flags);
        default: return static_cast<uint32_t>(d.Desc_1_2.Flags);
    }
}

// Version and every member of the descriptor the version selects (the union arms share their layout up to Flags).
bool same_fields(const D3D12_VERSIONED_ROOT_SIGNATURE_DESC& a, const D3D12_VERSIONED_ROOT_SIGNATURE_DESC& b) {
    if (a.Version != b.Version) return false;
    switch (a.Version) {
        case D3D_ROOT_SIGNATURE_VERSION_1_0:
            return a.Desc_1_0.NumParameters == b.Desc_1_0.NumParameters &&
                   a.Desc_1_0.pParameters == b.Desc_1_0.pParameters &&
                   a.Desc_1_0.NumStaticSamplers == b.Desc_1_0.NumStaticSamplers &&
                   a.Desc_1_0.pStaticSamplers == b.Desc_1_0.pStaticSamplers && a.Desc_1_0.Flags == b.Desc_1_0.Flags;
        case D3D_ROOT_SIGNATURE_VERSION_1_1:
            return a.Desc_1_1.NumParameters == b.Desc_1_1.NumParameters &&
                   a.Desc_1_1.pParameters == b.Desc_1_1.pParameters &&
                   a.Desc_1_1.NumStaticSamplers == b.Desc_1_1.NumStaticSamplers &&
                   a.Desc_1_1.pStaticSamplers == b.Desc_1_1.pStaticSamplers && a.Desc_1_1.Flags == b.Desc_1_1.Flags;
        default:
            return a.Desc_1_2.NumParameters == b.Desc_1_2.NumParameters &&
                   a.Desc_1_2.pParameters == b.Desc_1_2.pParameters &&
                   a.Desc_1_2.NumStaticSamplers == b.Desc_1_2.NumStaticSamplers &&
                   a.Desc_1_2.pStaticSamplers == b.Desc_1_2.pStaticSamplers && a.Desc_1_2.Flags == b.Desc_1_2.Flags;
    }
}

// Byte offset of the RTS0 flags dword in a DXBC container, or 0 when there is none.
size_t rts0_flags_offset(const std::vector<uint8_t>& blob) {
    if (blob.size() < 32 || std::memcmp(blob.data(), "DXBC", 4) != 0) return 0;
    uint32_t count = 0;
    std::memcpy(&count, blob.data() + 28, 4);
    for (uint32_t i = 0; i < count && 32 + i * 4 + 4 <= blob.size(); ++i) {
        uint32_t off = 0;
        std::memcpy(&off, blob.data() + 32 + i * 4, 4);
        if (off <= blob.size() && blob.size() - off >= 8 + 24 && std::memcmp(blob.data() + off, "RTS0", 4) == 0)
            return off + 8 + 20;
    }
    return 0;
}

std::vector<uint8_t> bytes_of(ID3DBlob* blob) {
    const auto* p = static_cast<const uint8_t*>(blob->GetBufferPointer());
    return {p, p + blob->GetBufferSize()};
}

void rehash(std::vector<uint8_t>& blob) {
    uint32_t hash[4];
    rt::d3d12_xs::dxbc_checksum(blob.data(), blob.size(), hash);
    std::memcpy(blob.data() + 4, hash, sizeof(hash));
}

// The host refused flags `f` and took them without kDirectlyIndexedFlags (`host_plain`, its blob for those). The shims
// serialize again without the two bits and put them back (serialize_*_keeping_heap_flags, d3d12_xs_rootsig.h), so
// their blob must be `host_plain` byte for byte except the RTS0 flags dword, which holds all of `f`, and the
// container hash, which must be valid.
bool is_heap_flag_fallback(const std::vector<uint8_t>& host_plain, const std::vector<uint8_t>& ours, uint32_t f) {
    const size_t at = rts0_flags_offset(ours);
    if (!at || ours.size() != host_plain.size() || rts0_flags_offset(host_plain) != at) return false;
    uint32_t stored = 0;
    std::memcpy(&stored, ours.data() + at, 4);
    if (stored != f) return false;
    for (size_t i = 0; i < ours.size(); ++i)
        if ((i < 4 || i >= 20) && (i < at || i >= at + 4) && ours[i] != host_plain[i]) return false;
    uint32_t hash[4];
    rt::d3d12_xs::dxbc_checksum(ours.data(), ours.size(), hash);
    return std::memcmp(hash, ours.data() + 4, sizeof(hash)) == 0;
}

// The helpers, over the whole range the executable's decoder can produce.
void test_helpers_keep_16_bit_flags(const std::vector<uint8_t>& clean_blob) {
    const D3D_ROOT_SIGNATURE_VERSION versions[] = {D3D_ROOT_SIGNATURE_VERSION_1_0, D3D_ROOT_SIGNATURE_VERSION_1_1,
                                                   D3D_ROOT_SIGNATURE_VERSION_1_2};
    const char* names[] = {"1.0", "1.1", "1.2"};
    for (int v = 0; v < 3; ++v) {
        uint32_t changed = 0;
        for (uint32_t f = 0; f <= 0xFFFFu; ++f) {
            const D3D12_VERSIONED_ROOT_SIGNATURE_DESC in = versioned(versions[v], f);
            D3D12_VERSIONED_ROOT_SIGNATURE_DESC out{};
            const bool known = rt::d3d12_xs::copy_versioned_root_signature_desc_clear_xbox_flags(in, out);
            if (!known || !same_fields(in, out) || flags_of(out) != f) ++changed;
        }
        char what[160];
        std::snprintf(what, sizeof(what),
                      "versioned %s helper: all 65536 flag values in [0, 0xFFFF] come out unchanged (%u changed)",
                      names[v], changed);
        check(changed == 0, what);
    }

    uint32_t legacy_changed = 0;
    for (uint32_t f = 0; f <= 0xFFFFu; ++f) {
        D3D12_ROOT_SIGNATURE_DESC in{};
        in.Flags = as_flags(f);
        D3D12_ROOT_SIGNATURE_DESC out{};
        rt::d3d12_xs::copy_root_signature_desc_clear_xbox_flags(in, out);
        if (out.NumParameters != in.NumParameters || out.pParameters != in.pParameters ||
            out.NumStaticSamplers != in.NumStaticSamplers || out.pStaticSamplers != in.pStaticSamplers ||
            out.Flags != in.Flags)
            ++legacy_changed;
    }
    check(legacy_changed == 0, "legacy helper: all 65536 flag values in [0, 0xFFFF] come out unchanged");

    const size_t at = rts0_flags_offset(clean_blob);
    check(at != 0, "the host serializer's blob has an RTS0 part");
    if (!at) return;
    std::vector<uint8_t> blob = clean_blob;
    uint32_t rewritten = 0;
    for (uint32_t f = 0; f <= 0xFFFFu; ++f) {
        std::memcpy(blob.data() + at, &f, 4);
        std::vector<uint8_t> out;
        if (rt::d3d12_xs::copy_rts0_blob_clear_xbox_flags(blob.data(), blob.size(), out) || !out.empty()) ++rewritten;
    }
    check(rewritten == 0, "blob helper: an RTS0 blob with flags in [0, 0xFFFF] is never rewritten");
}

void test_serializer_shim_matches_host() {
    using SerializeFn = HRESULT(WINAPI*)(const D3D12_VERSIONED_ROOT_SIGNATURE_DESC*, ID3DBlob**, ID3DBlob**);
    using LegacyFn = HRESULT(WINAPI*)(const D3D12_ROOT_SIGNATURE_DESC*, D3D_ROOT_SIGNATURE_VERSION, ID3DBlob**,
                                      ID3DBlob**);
    auto shim = reinterpret_cast<SerializeFn>(rt::shims::find("d3d12_xs.dll", "D3D12SerializeVersionedRootSignature"));
    auto legacy = reinterpret_cast<LegacyFn>(rt::shims::find("d3d12_xs.dll", "D3D12SerializeRootSignature"));
    check(shim && legacy, "shim registry exposes both d3d12_xs serializers");
    if (!shim || !legacy) return;

    std::vector<uint32_t> values;
    for (D3D12_ROOT_SIGNATURE_FLAGS f : kPublicFlags) values.push_back(static_cast<uint32_t>(f));
    values.push_back(all_public_flags());
    values.push_back(0);

    // Probed once, as the shim does: does the host serialize an empty 1.2 description?
    bool host_serializes_1_2 = false;
    {
        const D3D12_VERSIONED_ROOT_SIGNATURE_DESC empty12 = versioned(D3D_ROOT_SIGNATURE_VERSION_1_2, 0);
        Blobs b;
        host_serializes_1_2 = SUCCEEDED(D3D12SerializeVersionedRootSignature(&empty12, &b.blob, &b.error));
        std::printf("host serializer %s root signature 1.2\n", host_serializes_1_2 ? "takes" : "refuses");
    }

    const D3D_ROOT_SIGNATURE_VERSION versions[] = {D3D_ROOT_SIGNATURE_VERSION_1_0, D3D_ROOT_SIGNATURE_VERSION_1_1,
                                                   D3D_ROOT_SIGNATURE_VERSION_1_2};
    const char* names[] = {"1.0", "1.1", "1.2"};
    for (int v = 0; v < 3; ++v) {
        int differ = 0, same_bytes = 0, host_refused = 0, heap_fallback = 0, lost_flags = 0, touched_input = 0;
        for (uint32_t f : values) {
            const D3D12_VERSIONED_ROOT_SIGNATURE_DESC desc = versioned(versions[v], f);
            const D3D12_VERSIONED_ROOT_SIGNATURE_DESC before = desc;
            Blobs host, ours;
            const HRESULT hr_host = D3D12SerializeVersionedRootSignature(&desc, &host.blob, &host.error);
            const HRESULT hr_ours = shim(&desc, &ours.blob, &ours.error);
            if (!same_fields(desc, before)) ++touched_input;
            if (SUCCEEDED(hr_host)) {
                if (FAILED(hr_ours) || !ours.blob || !host.blob || bytes_of(host.blob) != bytes_of(ours.blob)) ++differ;
                else ++same_bytes;
            } else {
                ++host_refused;
                // A host that serializes only up to 1.1 refuses every 1.2 description, and the shim rewrites those
                // as 1.1. A host that refuses the heap-indexed flags (Windows 10) gets them serialized anyway when it
                // takes the description without them. Otherwise (1.0, 1.1, or 1.2 on a host that takes 1.2) the shim
                // must refuse what the host refuses.
                const bool rewritten_as_1_1 = versions[v] == D3D_ROOT_SIGNATURE_VERSION_1_2 && !host_serializes_1_2;
                const uint32_t plain_flags = f & ~rt::d3d12_xs::kDirectlyIndexedFlags;
                bool host_takes_plain = false;
                if (!rewritten_as_1_1 && plain_flags != f) {
                    const D3D12_VERSIONED_ROOT_SIGNATURE_DESC plain = versioned(versions[v], plain_flags);
                    Blobs host_plain;
                    host_takes_plain =
                        SUCCEEDED(D3D12SerializeVersionedRootSignature(&plain, &host_plain.blob, &host_plain.error)) &&
                        host_plain.blob;
                    if (host_takes_plain) {
                        ++heap_fallback;
                        if (hr_ours != S_OK || !ours.blob ||
                            !is_heap_flag_fallback(bytes_of(host_plain.blob), bytes_of(ours.blob), f))
                            ++differ;
                    }
                }
                if (!rewritten_as_1_1 && !host_takes_plain && SUCCEEDED(hr_ours)) ++differ;
            }
            if (SUCCEEDED(hr_ours) && ours.blob) {
                ID3D12VersionedRootSignatureDeserializer* de = nullptr;
                if (FAILED(D3D12CreateVersionedRootSignatureDeserializer(ours.blob->GetBufferPointer(),
                                                                         ours.blob->GetBufferSize(),
                                                                         IID_PPV_ARGS(&de))) || !de) {
                    ++lost_flags;
                } else {
                    const D3D12_VERSIONED_ROOT_SIGNATURE_DESC* got = de->GetUnconvertedRootSignatureDesc();
                    if (!got || flags_of(*got) != f) ++lost_flags;
                    de->Release();
                }
            }
        }
        char what[240];
        std::snprintf(what, sizeof(what),
                      "versioned %s: the shim serializes each public flag, 0x80 and all of them as the host does "
                      "(%d byte-identical, %d refused by the host, %d of those kept with the heap-indexed flags, "
                      "%d different)",
                      names[v], same_bytes, host_refused, heap_fallback, differ);
        check(differ == 0, what);
        std::snprintf(what, sizeof(what), "versioned %s: every blob the shim returns deserializes with the same flags",
                      names[v]);
        check(lost_flags == 0, what);
        std::snprintf(what, sizeof(what), "versioned %s: the caller's descriptor is never modified", names[v]);
        check(touched_input == 0, what);
    }

    // The legacy entry point, 1.0 only (it has no 1.1 descriptor), with the same heap-indexed fallback.
    int differ = 0, legacy_fallback = 0;
    for (uint32_t f : values) {
        D3D12_ROOT_SIGNATURE_DESC desc{};
        desc.Flags = as_flags(f);
        Blobs host, ours;
        const HRESULT hr_host = D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1_0, &host.blob,
                                                            &host.error);
        const HRESULT hr_ours = legacy(&desc, D3D_ROOT_SIGNATURE_VERSION_1_0, &ours.blob, &ours.error);
        if (SUCCEEDED(hr_host)) {
            if (FAILED(hr_ours) || !host.blob || !ours.blob || bytes_of(host.blob) != bytes_of(ours.blob)) ++differ;
            continue;
        }
        const uint32_t plain_flags = f & ~rt::d3d12_xs::kDirectlyIndexedFlags;
        bool host_takes_plain = false;
        if (plain_flags != f) {
            D3D12_ROOT_SIGNATURE_DESC plain{};
            plain.Flags = as_flags(plain_flags);
            Blobs host_plain;
            host_takes_plain = SUCCEEDED(D3D12SerializeRootSignature(&plain, D3D_ROOT_SIGNATURE_VERSION_1_0,
                                                                     &host_plain.blob, &host_plain.error)) &&
                               host_plain.blob;
            if (host_takes_plain) {
                ++legacy_fallback;
                if (hr_ours != S_OK || !ours.blob ||
                    !is_heap_flag_fallback(bytes_of(host_plain.blob), bytes_of(ours.blob), f))
                    ++differ;
            }
        }
        if (!host_takes_plain && SUCCEEDED(hr_ours)) ++differ;
    }
    char legacy_what[200];
    std::snprintf(legacy_what, sizeof(legacy_what),
                  "legacy serializer shim gives the host's result for each public flag, 0x80 and all of them "
                  "(%d kept with the heap-indexed flags the host refuses)",
                  legacy_fallback);
    check(differ == 0, legacy_what);

    // Call site 1 of old upstream PR 7140, by name.
    const D3D12_VERSIONED_ROOT_SIGNATURE_DESC exe = versioned(D3D_ROOT_SIGNATURE_VERSION_1_1, kExecutableFlags);
    Blobs host, ours;
    const HRESULT hr_host = D3D12SerializeVersionedRootSignature(&exe, &host.blob, &host.error);
    const HRESULT hr_ours = shim(&exe, &ours.blob, &ours.error);
    check(SUCCEEDED(hr_host) && SUCCEEDED(hr_ours) && host.blob && ours.blob &&
              bytes_of(host.blob) == bytes_of(ours.blob),
          "1.1, no parameters, Flags 0x80 (the executable's direct call): the shim's blob is the host's, byte for byte");
    if (ours.blob) {
        const std::vector<uint8_t> b = bytes_of(ours.blob);
        const size_t at = rts0_flags_offset(b);
        uint32_t stored = 0;
        if (at) std::memcpy(&stored, b.data() + at, 4);
        check(at && stored == kExecutableFlags, "the shim's blob stores RTS0 flags 0x80");
    }
}

void test_deserializer(const std::vector<uint8_t>& clean_blob) {
    using DeserializeFn = HRESULT(WINAPI*)(const void*, SIZE_T, REFIID, void**);
    auto shim = reinterpret_cast<DeserializeFn>(rt::shims::find("d3d12_xs.dll", "D3D12CreateRootSignatureDeserializer"));
    check(shim != nullptr, "shim registry exposes d3d12_xs!D3D12CreateRootSignatureDeserializer");
    const size_t at = rts0_flags_offset(clean_blob);
    if (!shim || !at) return;

    auto flags_through = [&](const std::vector<uint8_t>& blob, bool through_shim, uint32_t& flags) {
        ID3D12RootSignatureDeserializer* de = nullptr;
        const HRESULT hr = through_shim ? shim(blob.data(), blob.size(), IID_PPV_ARGS(&de))
                                        : D3D12CreateRootSignatureDeserializer(blob.data(), blob.size(),
                                                                               IID_PPV_ARGS(&de));
        if (FAILED(hr) || !de) return false;
        flags = static_cast<uint32_t>(de->GetRootSignatureDesc()->Flags);
        de->Release();
        return true;
    };

    // Each public flag, 0x80 and all of them, written into a correctly hashed 1.1 blob.
    int differ = 0;
    std::vector<uint32_t> values;
    for (D3D12_ROOT_SIGNATURE_FLAGS f : kPublicFlags) values.push_back(static_cast<uint32_t>(f));
    values.push_back(all_public_flags());
    for (uint32_t f : values) {
        std::vector<uint8_t> blob = clean_blob;
        std::memcpy(blob.data() + at, &f, 4);
        rehash(blob);
        uint32_t host_flags = 0, shim_flags = 0;
        const bool host_ok = flags_through(blob, false, host_flags);
        const bool shim_ok = flags_through(blob, true, shim_flags);
        if (host_ok != shim_ok || (shim_ok && (shim_flags != f || host_flags != f))) ++differ;
    }
    check(differ == 0, "the deserializer shim returns each public flag, 0x80 and all of them as the host does");

    // A blob carrying the Xbox bits on top of 0x80: exactly 0x03000000 is cleared, nothing else changes.
    std::vector<uint8_t> xbox = clean_blob;
    const uint32_t xbox_flags = kXboxFlags | kExecutableFlags;
    std::memcpy(xbox.data() + at, &xbox_flags, 4);
    rehash(xbox);
    std::vector<uint8_t> cleared;
    const bool rewritten = rt::d3d12_xs::copy_rts0_blob_clear_xbox_flags(xbox.data(), xbox.size(), cleared);
    check(rewritten && cleared.size() == xbox.size(), "a blob with flags 0x03000080 is rewritten");
    if (!rewritten || cleared.size() != xbox.size()) return;
    uint32_t got = 0;
    std::memcpy(&got, cleared.data() + at, 4);
    check(got == kExecutableFlags && (xbox_flags ^ got) == kXboxFlags,
          "0x03000080 becomes 0x80: only bits 24 and 25 are cleared");
    bool rest_same = true;
    for (size_t i = 0; i < xbox.size(); ++i)
        if ((i < 4 || i >= 20) && (i < at || i >= at + 4) && xbox[i] != cleared[i]) rest_same = false;
    check(rest_same, "every byte outside the flags dword and the container hash is unchanged");
    uint32_t hash[4];
    rt::d3d12_xs::dxbc_checksum(cleared.data(), cleared.size(), hash);
    check(std::memcmp(hash, cleared.data() + 4, sizeof(hash)) == 0, "the rewritten blob carries a valid container hash");
    uint32_t shim_flags = 0;
    check(flags_through(xbox, true, shim_flags) && shim_flags == kExecutableFlags,
          "the deserializer shim hands back 0x80 for a blob with flags 0x03000080");

    // The mask is the documented byte (d3d12_xs_rootsig.h): the other high bits go too, the low 24 stay.
    const uint32_t high_and_low = 0xFC000000u | 0x00FF0000u | kExecutableFlags;
    std::memcpy(xbox.data() + at, &high_and_low, 4);
    rehash(xbox);
    const uint32_t low_kept = 0x00FF0000u | kExecutableFlags;
    check(rt::d3d12_xs::copy_rts0_blob_clear_xbox_flags(xbox.data(), xbox.size(), cleared) &&
              std::memcmp(cleared.data() + at, &low_kept, 4) == 0,
          "flags 0xFCFF0080 become 0x00FF0080: bits 24-31 are cleared, bits 0-23 kept");
}

}  // namespace

int main() {
    std::printf("root_sig_public_flags_test: the d3d12_xs flag mask keeps every public root-signature flag\n");
    // A clean 1.1 blob from the host serializer: no parameters, no samplers, flags 0.
    std::vector<uint8_t> clean;
    {
        const D3D12_VERSIONED_ROOT_SIGNATURE_DESC empty = versioned(D3D_ROOT_SIGNATURE_VERSION_1_1, 0);
        Blobs b;
        const HRESULT hr = D3D12SerializeVersionedRootSignature(&empty, &b.blob, &b.error);
        check(SUCCEEDED(hr) && b.blob, "the host serializes an empty 1.1 root signature");
        if (b.blob) clean = bytes_of(b.blob);
    }
    if (!clean.empty()) test_helpers_keep_16_bit_flags(clean);
    test_serializer_shim_matches_host();
    if (!clean.empty()) test_deserializer(clean);
    std::printf("root_sig_public_flags_test: %s (%d failure%s)\n", g_failures ? "FAILED" : "OK", g_failures,
                g_failures == 1 ? "" : "s");
    return g_failures ? 1 : 0;
}
