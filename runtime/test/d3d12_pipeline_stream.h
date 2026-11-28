// Build a pipeline state stream the way d3dx12.h lays one out (CD3DX12_PIPELINE_STATE_STREAM_SUBOBJECT: a subobject
// type followed by its description, pointer-aligned), and create the PSO from it. A mesh-shader PSO has no
// D3D12_GRAPHICS_PIPELINE_STATE_DESC form, so the tests that build one use this.
#pragma once

#include <windows.h>
#include <d3d12.h>

#include <cstring>
#include <vector>

#include "d3d12_test_shaders.h"  // Ref

namespace d3d12test {

// One stream subobject as d3dx12.h declares it (CD3DX12_PIPELINE_STATE_STREAM_SUBOBJECT).
template <D3D12_PIPELINE_STATE_SUBOBJECT_TYPE Type, typename Inner>
struct alignas(void*) Subobject {
    D3D12_PIPELINE_STATE_SUBOBJECT_TYPE type = Type;
    Inner inner;
};
using RootSubobject = Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE, ID3D12RootSignature*>;
using VsSubobject = Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VS, D3D12_SHADER_BYTECODE>;
using PsSubobject = Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS, D3D12_SHADER_BYTECODE>;
using CsSubobject = Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_CS, D3D12_SHADER_BYTECODE>;
using AsSubobject = Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_AS, D3D12_SHADER_BYTECODE>;
using MsSubobject = Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS, D3D12_SHADER_BYTECODE>;
using TopologySubobject = Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PRIMITIVE_TOPOLOGY,
                                    D3D12_PRIMITIVE_TOPOLOGY_TYPE>;
using TargetsSubobject = Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RENDER_TARGET_FORMATS, D3D12_RT_FORMAT_ARRAY>;
using DepthStencil1Subobject = Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL1, D3D12_DEPTH_STENCIL_DESC1>;
using DepthStencilSubobject = Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL, D3D12_DEPTH_STENCIL_DESC>;
using RasterizerSubobject = Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER, D3D12_RASTERIZER_DESC>;
using Rasterizer2Subobject = Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER2, D3D12_RASTERIZER_DESC2>;
using MaskSubobject = Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_MASK, UINT>;

// Subobjects appended one after another: each is pointer-aligned and a multiple of the pointer size, so the bytes
// are what a struct of the same members would hold.
struct Stream {
    std::vector<std::uint8_t> bytes;
    template <typename S>
    Stream& add(const S& subobject) {
        static_assert(sizeof(S) % sizeof(void*) == 0);
        const std::size_t at = bytes.size();
        bytes.resize(at + sizeof(S));
        std::memcpy(bytes.data() + at, &subobject, sizeof(S));
        return *this;
    }
    D3D12_PIPELINE_STATE_STREAM_DESC desc(std::size_t trim = 0) { return {bytes.size() - trim, bytes.data()}; }
};

// `count` render targets of one format.
inline D3D12_RT_FORMAT_ARRAY rt_formats(UINT count, DXGI_FORMAT format) {
    D3D12_RT_FORMAT_ARRAY formats{};
    for (UINT i = 0; i < count && i < D3D12_SIMULTANEOUS_RENDER_TARGET_COUNT; ++i) formats.RTFormats[i] = format;
    formats.NumRenderTargets = count;
    return formats;
}

inline D3D12_RT_FORMAT_ARRAY one_target() { return rt_formats(1, DXGI_FORMAT_R8G8B8A8_UNORM); }

inline HRESULT create(ID3D12Device* device, Stream& stream, Ref<ID3D12PipelineState>& out, std::size_t trim = 0) {
    Ref<ID3D12Device2> device2;
    if (FAILED(device->QueryInterface(IID_PPV_ARGS(&device2.p)))) return E_NOINTERFACE;
    const D3D12_PIPELINE_STATE_STREAM_DESC desc = stream.desc(trim);
    out.reset();
    return device2->CreatePipelineState(&desc, IID_PPV_ARGS(&out.p));
}

}  // namespace d3d12test
