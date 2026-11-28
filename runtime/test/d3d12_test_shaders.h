// Helpers for the D3D12 pipeline tests: wrap a synthetic FSSM program in a shader container shaped like the game's
// (DXBC { DXIL { stub, XDAT { 0x0D: XBCH + zlib(fssm) } } }, the layout tools/fssm_unpack.py reads), serialize a
// root signature with the host PC runtime, and write a translator-style index next to compiled test modules.
// No game data: the programs are a few marker words, and the modules are GLSL written to the translator's contract.
#pragma once

#include <windows.h>
#include <d3d12.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "../shims/d3d12_xs_rootsig.h"

namespace d3d12test {

inline void put32(std::vector<std::uint8_t>& out, std::uint32_t v) {
    for (int i = 0; i < 4; ++i) out.push_back(std::uint8_t(v >> (8 * i)));
}

// zlib with one stored block (RFC 1950/1951), enough to wrap a test program.
inline std::vector<std::uint8_t> zlib_stored(const std::vector<std::uint8_t>& data) {
    std::vector<std::uint8_t> out = {0x78, 0x01, 0x01};
    const auto length = std::uint16_t(data.size());
    out.insert(out.end(), {std::uint8_t(length), std::uint8_t(length >> 8), std::uint8_t(~length),
                           std::uint8_t(~length >> 8)});
    out.insert(out.end(), data.begin(), data.end());
    std::uint32_t a = 1, b = 0;
    for (const std::uint8_t x : data) {
        a = (a + x) % 65521;
        b = (b + a) % 65521;
    }
    const std::uint32_t adler = (b << 16) | a;
    out.insert(out.end(), {std::uint8_t(adler >> 24), std::uint8_t(adler >> 16), std::uint8_t(adler >> 8),
                           std::uint8_t(adler)});
    return out;
}

// A synthetic FSSM program: the magic, then `words` (distinct per test shader so their hashes differ).
inline std::vector<std::uint8_t> fssm_program(std::initializer_list<std::uint32_t> words) {
    std::vector<std::uint8_t> out = {'F', 'S', 'S', 'M'};
    for (const std::uint32_t w : words) put32(out, w);
    return out;
}

inline std::vector<std::uint8_t> shader_container(const std::vector<std::uint8_t>& fssm,
                                                  const std::array<std::uint8_t, 16>* digest = nullptr) {
    const auto stream = zlib_stored(fssm);
    std::vector<std::uint8_t> xbch = {'X', 'B', 'C', 'H'};
    for (const std::uint32_t v : {1u, std::uint32_t(stream.size()), 0u, std::uint32_t(fssm.size()), 0u}) put32(xbch, v);
    xbch.insert(xbch.end(), stream.begin(), stream.end());
    std::vector<std::uint8_t> dxil(16, 0);
    dxil.insert(dxil.end(), {'X', 'D', 'A', 'T'});
    put32(dxil, 1);
    put32(dxil, 12);
    put32(dxil, 0x0d);
    put32(dxil, 16 + 12 + 12);
    put32(dxil, std::uint32_t(xbch.size()));
    dxil.insert(dxil.end(), xbch.begin(), xbch.end());
    std::vector<std::uint8_t> out = {'D', 'X', 'B', 'C'};
    out.resize(28, 0);
    put32(out, digest ? 2 : 1);
    put32(out, digest ? 40 : 36);
    if (digest) put32(out, 40 + 8 + std::uint32_t(dxil.size()));
    out.insert(out.end(), {'D', 'X', 'I', 'L'});
    put32(out, std::uint32_t(dxil.size()));
    out.insert(out.end(), dxil.begin(), dxil.end());
    if (digest) {
        out.insert(out.end(), {'H', 'A', 'S', 'H'});
        put32(out, 20);
        put32(out, 0);
        out.insert(out.end(), digest->begin(), digest->end());
    }
    return out;
}

using SerializeFn = HRESULT(WINAPI*)(const D3D12_VERSIONED_ROOT_SIGNATURE_DESC*, ID3DBlob**, ID3DBlob**);

inline std::uint32_t root_signature_flags(const D3D12_VERSIONED_ROOT_SIGNATURE_DESC& desc) {
    switch (desc.Version) {
    case D3D_ROOT_SIGNATURE_VERSION_1_0: return static_cast<std::uint32_t>(desc.Desc_1_0.Flags);
    case D3D_ROOT_SIGNATURE_VERSION_1_1: return static_cast<std::uint32_t>(desc.Desc_1_1.Flags);
    case D3D_ROOT_SIGNATURE_VERSION_1_2: return static_cast<std::uint32_t>(desc.Desc_1_2.Flags);
    default: return 0;
    }
}

inline void set_root_signature_flags(D3D12_VERSIONED_ROOT_SIGNATURE_DESC& desc, std::uint32_t flags) {
    switch (desc.Version) {
    case D3D_ROOT_SIGNATURE_VERSION_1_0:
        desc.Desc_1_0.Flags = static_cast<D3D12_ROOT_SIGNATURE_FLAGS>(flags);
        break;
    case D3D_ROOT_SIGNATURE_VERSION_1_1:
        desc.Desc_1_1.Flags = static_cast<D3D12_ROOT_SIGNATURE_FLAGS>(flags);
        break;
    default:
        desc.Desc_1_2.Flags = static_cast<D3D12_ROOT_SIGNATURE_FLAGS>(flags);
        break;
    }
}

// The in-box d3d12.dll of Windows 10 refuses the two heap-directly-indexed root-signature flags at every descriptor
// version ("Unsupported bit-flag set") although its deserializer and devices take a blob that carries them. Retry
// without them and put them back, like the shim does. The DXBC hash is zeroed rather than recomputed: the host
// accepts an all-zero checksum, and the device these blobs go to here parses the container itself.
inline std::vector<std::uint8_t> serialize_root_signature(const D3D12_VERSIONED_ROOT_SIGNATURE_DESC& desc) {
    const HMODULE d3d12 = LoadLibraryExA("d3d12.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    const auto fn = d3d12 ? reinterpret_cast<SerializeFn>(
                                reinterpret_cast<void*>(GetProcAddress(d3d12, "D3D12SerializeVersionedRootSignature")))
                          : nullptr;
    std::vector<std::uint8_t> out;
    ID3DBlob* blob = nullptr;
    if (!fn) return out;

    const std::uint32_t flags = root_signature_flags(desc);
    HRESULT hr = fn(&desc, &blob, nullptr);
    bool retried = false;
    if (FAILED(hr) && blob == nullptr && (flags & rt::d3d12_xs::kDirectlyIndexedFlags)) {
        D3D12_VERSIONED_ROOT_SIGNATURE_DESC stripped = desc;
        set_root_signature_flags(stripped, flags & ~rt::d3d12_xs::kDirectlyIndexedFlags);
        hr = fn(&stripped, &blob, nullptr);
        retried = SUCCEEDED(hr) && blob != nullptr;
    }
    if (SUCCEEDED(hr) && blob) {
        const auto* p = static_cast<const std::uint8_t*>(blob->GetBufferPointer());
        out.assign(p, p + blob->GetBufferSize());
        if (retried) {
            for (std::size_t i = 32; i + 8 <= out.size(); ++i) {
                if (std::memcmp(out.data() + i, "RTS0", 4) != 0) continue;
                std::memcpy(out.data() + i + 8 + 20, &flags, sizeof(flags));
                std::memset(out.data() + 4, 0, 16);
                break;
            }
        }
        blob->Release();
    }
    return out;
}

inline D3D12_RESOURCE_DESC buffer_desc(UINT64 bytes, D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE) {
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = bytes;
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    desc.Flags = flags;
    return desc;
}

inline D3D12_RESOURCE_BARRIER transition(ID3D12Resource* resource, D3D12_RESOURCE_STATES before,
                                         D3D12_RESOURCE_STATES after) {
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = resource;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    barrier.Transition.StateBefore = before;
    barrier.Transition.StateAfter = after;
    return barrier;
}

template<class T>
struct Ref {
    T* p = nullptr;
    ~Ref() { reset(); }
    void reset() {
        if (p) p->Release();
        p = nullptr;
    }
    T* operator->() const { return p; }
};

inline void* slot(void* object, unsigned index) { return (*reinterpret_cast<void***>(object))[index]; }

}  // namespace d3d12test
