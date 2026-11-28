#pragma once

// Source: adapted from enx/leonida, commit 5c175db901f8486bcdefd03746b578b657e3151f on the original author's git
// mirror (https://git.notrexed.dev/enx/leonida), a fork of this project. Licence: GPL-3.0-or-later as a fork of this
// project. That server could not be reached on 2026-11-25, when this file was merged, so the source and its licence
// could not be read there; the maintainer accepted the file because the source is a fork of this same
// GPL-3.0-or-later project (the README links the server as the original author's mirror) and because
// tools/native_pso_hashes.py is adapted from the same commit. Changed, as the pull request that brought it describes
// it (the source could not be compared): the state and stage resolution moved to runtime/src/native_pso.h, and the
// stage bytecode handed on is read from the user's own carved DXBC containers, not from the description.
//
// Xbox native graphics pipeline state, adapted from enx/leonida's slot-57 implementation.
// The state and stage resolution are in native_pso.h. All stage bytecode handed to D3D12
// comes from the user's carved DXBC containers and is translated by the normal Vulkan PSO path.
#include "../src/native_pso.h"
#include "d3d12_xs_native_pso_fallback.h"
#include "d3d12_xs_pso_stream.h"
#include "../src/shader_library.h"

#include <d3d12.h>
#include <atomic>
#include <climits>
#include <cstring>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <mutex>

namespace rt::d3d12_xs {
namespace native_pipeline_detail {
namespace np = leonida::native_pso;

// The blend state of blob A's colour-target registers (RT-SLOT57-BLEND-STATE). Without it, or for a target this path
// cannot express (np::d3d12_blend), every target keeps blending off and writes every component, as before.
// Blending off and every component written on every target: the factory's state before blob A was read.
inline void default_blend_state(D3D12_BLEND_DESC& out) {
    for (auto& rt : out.RenderTarget) {
        rt = {};
        rt.SrcBlend = rt.SrcBlendAlpha = D3D12_BLEND_ONE;
        rt.DestBlend = rt.DestBlendAlpha = D3D12_BLEND_ZERO;
        rt.BlendOp = rt.BlendOpAlpha = D3D12_BLEND_OP_ADD;
        rt.LogicOp = D3D12_LOGIC_OP_NOOP;
        rt.RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
    }
    out.AlphaToCoverageEnable = FALSE;
    out.IndependentBlendEnable = FALSE;
}

inline void fill_blend_state(const np::FixedState& s, D3D12_BLEND_DESC& out) {
    default_blend_state(out);
    const auto targets = s.blend ? np::d3d12_blend(*s.blend, s.rt_count, s.rtv_formats) : std::nullopt;
    if (!targets) {
        static std::atomic<int> logged{0};
        if (logged.fetch_add(1, std::memory_order_relaxed) < 4)
            std::fprintf(stderr, "[rt] d3d12_xs: slot 57: %s, so this pipeline keeps blending off (CB_BLEND0 %08x)\n",
                         s.blend ? "a target's blend state is not one this path builds (an unknown factor or function)"
                                 : "blob A is absent or not the measured shape",
                         s.blend ? static_cast<unsigned>(s.blend->blend[0]) : 0u);
        return;
    }
    for (unsigned i = 0; i < s.rt_count; ++i) {
        const np::TargetBlend& t = (*targets)[i];
        D3D12_RENDER_TARGET_BLEND_DESC& rt = out.RenderTarget[i];
        rt.BlendEnable = t.enable;
        rt.SrcBlend = static_cast<D3D12_BLEND>(t.src);
        rt.DestBlend = static_cast<D3D12_BLEND>(t.dst);
        rt.BlendOp = static_cast<D3D12_BLEND_OP>(t.op);
        rt.SrcBlendAlpha = static_cast<D3D12_BLEND>(t.src_alpha);
        rt.DestBlendAlpha = static_cast<D3D12_BLEND>(t.dst_alpha);
        rt.BlendOpAlpha = static_cast<D3D12_BLEND_OP>(t.op_alpha);
        rt.RenderTargetWriteMask = t.write_mask;
        if (i && std::memcmp(&rt, &out.RenderTarget[0], sizeof(rt)) != 0) out.IndependentBlendEnable = TRUE;
    }
}

// The depth, stencil and raster state of blob A's registers (RT-SLOT57-DEPTH-RASTER). Without them, or for a state
// this path cannot express (np::d3d12_depth_raster), the defaults below stay, as before: depth ALWAYS with writes
// where there is a DSV, no stencil, no culling, depth clip on. Depth bias is not decoded and stays 0.
inline void fill_depth_raster_state(const np::FixedState& s, D3D12_GRAPHICS_PIPELINE_STATE_DESC& pc) {
    const auto dr = s.depth_raster ? np::d3d12_depth_raster(*s.depth_raster) : std::nullopt;
    if (!dr) {
        static std::atomic<int> logged{0};
        if (logged.fetch_add(1, std::memory_order_relaxed) < 4)
            std::fprintf(stderr,
                         "[rt] d3d12_xs: slot 57: %s, so this pipeline keeps the default depth and raster state "
                         "(DB_DEPTH_CONTROL %08x, PA_SU_SC_MODE_CNTL %08x)\n",
                         s.depth_raster ? "blob A's depth, stencil or raster state is not one this path builds"
                                        : "blob A is absent or not the measured shape",
                         s.depth_raster ? static_cast<unsigned>(s.depth_raster->depth_control) : 0u,
                         s.depth_raster ? static_cast<unsigned>(s.depth_raster->mode_cntl) : 0u);
    }
    const np::DepthRaster state = dr ? *dr : np::DepthRaster{};
    auto& r = pc.RasterizerState;
    r.FillMode = D3D12_FILL_MODE_SOLID;
    r.CullMode = static_cast<D3D12_CULL_MODE>(state.cull);
    r.FrontCounterClockwise = state.front_ccw;
    r.DepthClipEnable = state.depth_clip;
    auto& ds = pc.DepthStencilState;
    ds.DepthEnable = state.depth_enable;
    ds.DepthWriteMask = state.depth_write ? D3D12_DEPTH_WRITE_MASK_ALL : D3D12_DEPTH_WRITE_MASK_ZERO;
    ds.DepthFunc = static_cast<D3D12_COMPARISON_FUNC>(state.depth_func);
    ds.StencilEnable = state.stencil_enable;
    ds.StencilReadMask = state.read_mask;
    ds.StencilWriteMask = state.write_mask;
    const auto face = [](const np::StencilFace& f) {
        return D3D12_DEPTH_STENCILOP_DESC{static_cast<D3D12_STENCIL_OP>(f.fail), static_cast<D3D12_STENCIL_OP>(f.depth_fail),
                                          static_cast<D3D12_STENCIL_OP>(f.pass), static_cast<D3D12_COMPARISON_FUNC>(f.func)};
    };
    ds.FrontFace = face(state.front);
    ds.BackFace = face(state.back);
    // Without a DSV the console has nothing to test or write (27 records enable Z with none).
    if (pc.DSVFormat == DXGI_FORMAT_UNKNOWN) ds.DepthEnable = ds.StencilEnable = FALSE;
}

inline void fill_fixed_state(const np::FixedState& s, D3D12_GRAPHICS_PIPELINE_STATE_DESC& pc) {
    fill_blend_state(s, pc.BlendState);
    pc.SampleMask = s.sample_mask ? s.sample_mask : UINT_MAX;
    pc.DSVFormat = static_cast<DXGI_FORMAT>(s.dsv_format);
    fill_depth_raster_state(s, pc);
    pc.PrimitiveTopologyType = s.topology_type ? static_cast<D3D12_PRIMITIVE_TOPOLOGY_TYPE>(s.topology_type)
                                               : D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    pc.NumRenderTargets = s.rt_count;
    for (unsigned i = 0; i < s.rt_count; ++i) pc.RTVFormats[i] = static_cast<DXGI_FORMAT>(s.rtv_formats[i]);
    pc.SampleDesc.Count = s.sample_count;
}

template <D3D12_PIPELINE_STATE_SUBOBJECT_TYPE T, typename P>
struct alignas(void*) Subobject {
    D3D12_PIPELINE_STATE_SUBOBJECT_TYPE type = T;
    P value{};
};

struct MeshStream {
    Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE, ID3D12RootSignature*> root;
    Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS, D3D12_SHADER_BYTECODE> ms;
    Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS, D3D12_SHADER_BYTECODE> ps;
    Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_BLEND, D3D12_BLEND_DESC> blend;
    Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_MASK, UINT> sample_mask;
    Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER, D3D12_RASTERIZER_DESC> raster;
    Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL, D3D12_DEPTH_STENCIL_DESC> depth;
    Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RENDER_TARGET_FORMATS, D3D12_RT_FORMAT_ARRAY> rtvs;
    Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL_FORMAT, DXGI_FORMAT> dsv;
    Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_DESC, DXGI_SAMPLE_DESC> sample;
};

inline HRESULT create_mesh(ID3D12Device* real, const D3D12_GRAPHICS_PIPELINE_STATE_DESC& pc,
                           D3D12_SHADER_BYTECODE ms, const IID& riid, void** out) {
    ID3D12Device2* device2 = nullptr;
    if (FAILED(real->QueryInterface(__uuidof(ID3D12Device2), reinterpret_cast<void**>(&device2))) || !device2)
        return E_NOTIMPL;
    MeshStream s;
    s.root.value = pc.pRootSignature;
    s.ms.value = ms;
    s.ps.value = pc.PS;
    s.blend.value = pc.BlendState;
    s.sample_mask.value = pc.SampleMask;
    s.raster.value = pc.RasterizerState;
    s.depth.value = pc.DepthStencilState;
    s.rtvs.value.NumRenderTargets = pc.NumRenderTargets;
    for (unsigned i = 0; i < 8; ++i) s.rtvs.value.RTFormats[i] = pc.RTVFormats[i];
    s.dsv.value = pc.DSVFormat;
    s.sample.value = pc.SampleDesc;
    const D3D12_PIPELINE_STATE_STREAM_DESC stream{sizeof(s), &s};
    const HRESULT hr = device2->CreatePipelineState(&stream, riid, out);
    device2->Release();
    return hr;
}

// A pipeline state stream: each subobject is a type word, then its payload at its own alignment, padded to 8 bytes
// (d3dx12's alignas(void*) subobjects).
class Stream {
public:
    template <typename P>
    void add(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE type, const P& payload) {
        const std::size_t at = size_;
        const std::size_t offset = (at + sizeof(type) + alignof(P) - 1) & ~(alignof(P) - 1);
        size_ = (offset + sizeof(P) + 7) & ~std::size_t(7);
        words_.resize(size_ / 8);
        auto* bytes = reinterpret_cast<unsigned char*>(words_.data());
        std::memcpy(bytes + at, &type, sizeof(type));
        std::memcpy(bytes + offset, &payload, sizeof(P));
    }
    D3D12_PIPELINE_STATE_STREAM_DESC desc() { return {size_, words_.data()}; }

private:
    std::vector<std::uint64_t> words_;
    std::size_t size_ = 0;
};

// The stream of a graphics description; an MS stage makes a mesh pipeline (no input layout or topology).
inline void build_stream(const D3D12_GRAPHICS_PIPELINE_STATE_DESC& pc, D3D12_SHADER_BYTECODE ms, Stream& s) {
    s.add(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE, pc.pRootSignature);
    const bool mesh = ms.pShaderBytecode != nullptr;
    if (mesh) s.add(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS, ms);
    if (pc.VS.pShaderBytecode) s.add(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VS, pc.VS);
    if (pc.GS.pShaderBytecode) s.add(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_GS, pc.GS);
    if (pc.PS.pShaderBytecode) s.add(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS, pc.PS);
    s.add(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_BLEND, pc.BlendState);
    s.add(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_MASK, pc.SampleMask);
    s.add(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER, pc.RasterizerState);
    s.add(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL, pc.DepthStencilState);
    if (!mesh) {
        s.add(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_INPUT_LAYOUT, pc.InputLayout);
        s.add(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PRIMITIVE_TOPOLOGY, pc.PrimitiveTopologyType);
    }
    D3D12_RT_FORMAT_ARRAY rtvs{};
    rtvs.NumRenderTargets = pc.NumRenderTargets;
    for (unsigned i = 0; i < 8; ++i) rtvs.RTFormats[i] = pc.RTVFormats[i];
    s.add(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RENDER_TARGET_FORMATS, rtvs);
    s.add(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL_FORMAT, pc.DSVFormat);
    s.add(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_DESC, pc.SampleDesc);
}


}  // namespace native_pipeline_detail

// One instance per Vulkan device. The catalog is immutable once loaded; repeated pipeline
// descriptions can share the source containers without sharing mutable COM pipeline objects.
class NativeGraphicsPipelines {
public:
    template<class Create>
    HRESULT create(const std::filesystem::path& folder, const leonida::native_pso::GraphicsDesc& desc,
                   void** out, Create&& create_pipeline) {
        using namespace native_pipeline_detail;
        *out = nullptr;
        try {
            std::call_once(load_, [&] {
                std::ifstream file(folder / L"native_hashes.txt", std::ios::binary);
                if (!file) {
                    std::fprintf(stderr, "[rt] d3d12_xs: slot 57 needs native_hashes.txt from the shaders stage\n");
                    return;
                }
                const std::string text{std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
                std::string error;
                if (!catalog_.parse(text, &error)) std::fprintf(stderr, "[rt] d3d12_xs: slot 57: %s\n", error.c_str());
            });
            const auto plan = np::plan(desc, catalog_);
            if (plan.status == np::PlanStatus::malformed) return E_INVALIDARG;
            if (plan.status != np::PlanStatus::ok) return E_NOTIMPL;

            D3D12_GRAPHICS_PIPELINE_STATE_DESC pc{};
            pc.pRootSignature = static_cast<ID3D12RootSignature*>(const_cast<void*>(desc.root_signature));
            fill_fixed_state(plan.state, pc);
            std::vector<D3D12_INPUT_ELEMENT_DESC> elements;
            for (const auto& e : plan.state.elements) {
                elements.push_back({e.name.c_str(), e.semantic_index, static_cast<DXGI_FORMAT>(e.format), e.input_slot,
                                    e.aligned_byte_offset, static_cast<D3D12_INPUT_CLASSIFICATION>(e.classification),
                                    e.step_rate});
            }
            pc.InputLayout = {elements.data(), static_cast<UINT>(elements.size())};

            std::array<Container, np::kStageBlocks> code;
            D3D12_SHADER_BYTECODE mesh{};
            for (const auto& stage : plan.stages) {
                auto& bytes = code[stage.block];
                bytes = container(folder, *stage.shader);
                if (!bytes) return E_INVALIDARG;
                const D3D12_SHADER_BYTECODE view{bytes->data(), bytes->size()};
                if (stage.shader->stage == "vs") pc.VS = view;
                else if (stage.shader->stage == "ps") pc.PS = view;
                else if (stage.shader->stage == "gs") pc.GS = view;
                else mesh = view;  // plan() admits only VS/PS/GS or MS/PS.
            }
            HRESULT hr = create_pipeline(pc, mesh, code);
            // The decoded blend state can ask for something this device's layer does not do: blending a B4G4R4A4 target
            // that is an RGBA8 stand-in (without_stand_in_blend). Only a pipeline that blends such a target is retried.
            // E_NOTIMPL also stands for refusals that have nothing to do with blending (stream_unsupported: a rect list
            // without its geometry stage, mesh limits, FullyCovered; a dual-source output that does not fit Location 0
            // Index 1): those are returned as they are, never with a blend the title did not ask for.
            if (hr == E_NOTIMPL && plan.state.blend) {
                D3D12_GRAPHICS_PIPELINE_STATE_DESC retry = pc;
                const HRESULT again = without_stand_in_blend(retry) ? create_pipeline(retry, mesh, code) : hr;
                if (SUCCEEDED(again)) {
                    hr = again;
                    static std::atomic<int> logged{0};
                    if (logged.fetch_add(1, std::memory_order_relaxed) < 4)
                        std::fprintf(stderr, "[rt] d3d12_xs: slot 57: the device refused blending on this pipeline's "
                                             "B4G4R4A4 stand-in targets, so it was built with blending off on those "
                                             "targets\n");
                }
            }
            return hr;
        } catch (const std::bad_alloc&) {
            return E_OUTOFMEMORY;
        } catch (const std::exception&) {
            return E_INVALIDARG;
        }
    }

private:
    using Container = std::shared_ptr<const std::vector<std::uint8_t>>;
    Container container(const std::filesystem::path& folder, const leonida::native_pso::ShaderRef& ref) {
        {
            std::lock_guard lock(mutex_);
            if (const auto it = containers_.find(ref.digest); it != containers_.end()) return it->second;
        }
        // A catalog names files inside the shader workspace. Checking only its components lets a
        // reader with access to the two shader directories work without opening their ancestors.
        if (ref.path.find_first_of("\\:\r\n\t") != std::string::npos || ref.path.find('\0') != std::string::npos)
            return {};
        const auto path = std::filesystem::u8path(ref.path);
        if (path.has_root_path() || path.extension() != ".dxbc") return {};
        const auto workspace = folder.parent_path();
        const auto resolved = (folder / path).lexically_normal();
        const auto relative = resolved.lexically_relative(workspace);
        if (relative.empty() || *relative.begin() == "..") return {};
        auto component = workspace;
        for (const auto& part : relative) {
            component /= part;
            const DWORD attrs = GetFileAttributesW(component.c_str());
            if (attrs == INVALID_FILE_ATTRIBUTES || (attrs & FILE_ATTRIBUTE_REPARSE_POINT)) return {};
        }
        std::ifstream in(resolved, std::ios::binary | std::ios::ate);
        const std::streamoff size = in ? static_cast<std::streamoff>(in.tellg()) : 0;
        if (size <= 0 || size > (64 << 20)) return {};
        auto result = std::make_shared<std::vector<std::uint8_t>>(static_cast<std::size_t>(size));
        in.seekg(0);
        if (!in.read(reinterpret_cast<char*>(result->data()), size)) return {};
        const auto hash = leonida::dxbc_part(*result, "HASH");
        if (hash.size() < 20 || std::memcmp(hash.data() + 4, ref.digest.data(), ref.digest.size())) return {};
        std::lock_guard lock(mutex_);
        if (const auto it = containers_.find(ref.digest); it != containers_.end()) return it->second;
        constexpr std::size_t budget = std::size_t{256} << 20;
        if (result->size() <= budget - cached_bytes_) {
            containers_.emplace(ref.digest, result);
            cached_bytes_ += result->size();
        }
        return result;
    }

    std::once_flag load_;
    leonida::native_pso::HashMap catalog_;
    std::mutex mutex_;
    std::map<leonida::native_pso::Digest, Container> containers_;
    std::size_t cached_bytes_ = 0;
};

}  // namespace rt::d3d12_xs
