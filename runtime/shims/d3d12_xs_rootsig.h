// Helpers for forwarding Xbox root-signature descriptions to the PC d3d12.dll.
//
// The measured Xbox descriptions carry type bits in the high byte of D3D12_ROOT_SIGNATURE_FLAGS (observed value
// 0x03000000). The PC serializer rejects those bits, so the shims pass a shallow copy with bits 24-31 cleared and
// every other flag bit preserved. The pointed-to parameter/sampler arrays are not modified.
#pragma once

#include <d3d12.h>
#include <cstdint>
#include <vector>

namespace rt::d3d12_xs {

constexpr uint32_t kXboxRootSignatureFlagHighByte = 0xFF000000u;

// The two flags that let a shader index a descriptor heap directly. The in-box d3d12.dll of Windows 10 (measured on
// 10.0.19041.4355) refuses either of them in the serializer, at every descriptor version, with E_INVALIDARG
// ("Unsupported bit-flag set"), although its deserializer and its devices take a blob that carries them. The wrappers
// below serialize again without these bits and put them back into the produced RTS0 chunk.
constexpr uint32_t kDirectlyIndexedFlags =
    static_cast<uint32_t>(D3D12_ROOT_SIGNATURE_FLAG_CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED) |
    static_cast<uint32_t>(D3D12_ROOT_SIGNATURE_FLAG_SAMPLER_HEAP_DIRECTLY_INDEXED);

// Always copies `in` to `out`. Returns true for the known 1.0/1.1/1.2 descriptor versions and clears the high flag
// byte in the copy. Returns false for an unknown version, leaving the copy byte-for-byte identical to `in` so the
// caller can forward it unchanged and log the unsupported version.
bool copy_versioned_root_signature_desc_clear_xbox_flags(const D3D12_VERSIONED_ROOT_SIGNATURE_DESC& in,
                                                         D3D12_VERSIONED_ROOT_SIGNATURE_DESC& out);

// Windows 10's built-in d3d12.dll serializes root signatures up to version 1.1: it rejects every 1.2 description,
// even an empty one, with E_INVALIDARG, and reports HighestVersion 1.1. Version 1.2 differs from 1.1 only in
// D3D12_STATIC_SAMPLER_DESC1::Flags, so a 1.2 description whose static samplers all have Flags == 0 is the same
// root signature written as 1.1. This rewrites it that way: `out` becomes a 1.1 description that shares `in`'s
// parameters and points its samplers at `samplers` (resized here; keep it alive while `out` is used). Returns false,
// with `out` and `samplers` untouched, when `in` is not version 1.2 or a static sampler has non-zero Flags (a UINT
// border colour or non-normalized coordinates, which 1.1 can't express).
bool downgrade_root_signature_1_2_to_1_1(const D3D12_VERSIONED_ROOT_SIGNATURE_DESC& in,
                                         std::vector<D3D12_STATIC_SAMPLER_DESC>& samplers,
                                         D3D12_VERSIONED_ROOT_SIGNATURE_DESC& out);

// Legacy D3D12SerializeRootSignature descriptor (version 1.0): copies and clears the high flag byte.
void copy_root_signature_desc_clear_xbox_flags(const D3D12_ROOT_SIGNATURE_DESC& in, D3D12_ROOT_SIGNATURE_DESC& out);

// D3D12CreateRootSignatureDeserializer path: the host deserializer accepts a correctly hashed blob whose RTS0
// flags carry the Xbox high byte and hands the byte back, and CreateRootSignature then refuses it (E_FAIL on WARP
// and an RTX 5070). So copy the blob, clear bits 24-31 of the RTS0 flags, recompute the DXBC hash the host
// deserializer checks (a stale one is E_INVALIDARG; an all-zero hash is accepted as-is and kept zeroed), and return
// true with the copy in `out`. Returns false (and leaves `out` empty) when no rewrite is needed or the container is
// not valid to re-sign: no RTS0 chunk, no Xbox high byte, a malformed/truncated header, total_size != size, or a
// non-zero incoming DXBC hash that does not match the input bytes.
bool copy_rts0_blob_clear_xbox_flags(const void* data, size_t size, std::vector<uint8_t>& out);

// The DXBC container hash of `blob` (bytes 20..size), as stored at bytes 4..19. Zeroes `out` when `blob` is null,
// `size < 20`, or the hashed payload exceeds the 32-bit bit-count limit.
void dxbc_checksum(const uint8_t* blob, size_t size, uint32_t out[4]);

using VersionedSerializeFn = HRESULT(WINAPI*)(const D3D12_VERSIONED_ROOT_SIGNATURE_DESC*, ID3DBlob**, ID3DBlob**);
using LegacySerializeFn = HRESULT(WINAPI*)(const D3D12_ROOT_SIGNATURE_DESC*, D3D_ROOT_SIGNATURE_VERSION, ID3DBlob**,
                                           ID3DBlob**);

// Serialize through `fn` (the host entry point), cooperating with a host that refuses kDirectlyIndexedFlags: when the
// first call fails and the description carries any of those bits, it calls `fn` again with them cleared and, if that
// works, restores them in the produced RTS0 chunk and recomputes the container hash. A host that takes the flags
// itself (Windows 11) gets one call and its own blob back unchanged.
//
// Returns S_OK with a blob that carries the flags, S_FALSE when the retry worked but the blob could not be patched
// (usable, flags lost), or the host's failure from the retry. `blob`/`error` may be null and receive what a direct
// call would.
HRESULT serialize_versioned_keeping_heap_flags(VersionedSerializeFn fn, const D3D12_VERSIONED_ROOT_SIGNATURE_DESC& desc,
                                               ID3DBlob** blob, ID3DBlob** error);

// The same for the legacy D3D12SerializeRootSignature entry point and its version-1.0 descriptor.
HRESULT serialize_legacy_keeping_heap_flags(LegacySerializeFn fn, const D3D12_ROOT_SIGNATURE_DESC& desc,
                                            D3D_ROOT_SIGNATURE_VERSION version, ID3DBlob** blob, ID3DBlob** error);

}  // namespace rt::d3d12_xs
