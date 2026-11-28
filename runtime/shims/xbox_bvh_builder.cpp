// XboxBvhBuilder.dll served by the layer's own acceleration-structure builder (RT-5, boot stop 18).
//
// The package's XboxBvhBuilder.dll builds the game's ray-tracing structures with 52 console compute programs that
// its constructor creates as pipelines on the title's thread. Translated, they take 25-60 s to compile on a cold
// driver cache (31.2 s for one of them on an RTX 4080, 59.5 s for all 52 on an RX 9070 XT, measured in #7231 and
// #7049), longer than the title's startup watchdog (about 30 s), so a first boot ends at the builder. The layer
// already builds XDXR BVH4 structures itself (BuildRaytracingAccelerationStructure on the Vulkan device, D3D-178,
// src/xdxr_build.*), in the node format the game's translated traversal shaders read, and its one pipeline compiles
// in well under a second. So this file serves the DLL's exports with the layer's builder and the DLL is never
// loaded: the loader binds an import to a shim before a bundled module (pe_loader.cpp, Loader::resolve).
//
// The exports, as the executable imports them (7 of the DLL's 12, measured statically on 1.0.2.1: names and call
// counts only). Member functions: `this` comes first (rcx), then the arguments, as for a free function on x64.
//   ??0  XboxBvhBuilder(ID3D12Device11*, bool)                                      1 call site
//   ??1  ~XboxBvhBuilder()                                                         1
//   Set64BitMortonCodes(bool)                                                       8
//   GetRaytracingAccelerationStructurePrebuildInfo(const INPUTS*, const XDXR_STREAM_DESC*, PREBUILD_INFO*)  3
//   BuildRaytracingAccelerationStructureBatched2X(GCL6*, UINT, const BATCHED_DESC2*)  6
//   CopyRaytracingAccelerationStructure(GCL4*, UINT64 dest, UINT64 source, COPY_MODE)  2
//   RehydrateRaytracingAccelerationStructureBatched2X(GCL8*, UINT, const BUILD_DESC*, const UINT64*) -> HRESULT  4
// The Xbox descriptions are the measured layouts of src/xdxr_descs.h (docs/xdxr.md).
//
// What differs from the console builder, on purpose:
//   - No programs are created: the constructor only remembers the device.
//   - The XDXR stream records (docs/xdxr.md §3) tune the console builder's own heuristics. The layer's builder has
//     none of those settings, so a non-empty stream is logged once and otherwise ignored; the structure is still a
//     valid BVH of the same geometry. Set64BitMortonCodes is recorded and ignored for the same reason.
//   - The DLL throws its validation errors as C++ exceptions (docs/xdxr.md, "Errors are thrown, not returned"). A
//     shim must not throw into the title, so an element the DLL would reject is logged and skipped instead. The one
//     rule a correct title meets every time is kept: the offline-statistics postbuild type (0x10) is never written.
//   - Rehydration: RehydrateRaytracingAccelerationStructureBatched2X rebuilds each structure from the source
//     geometry its description carries (BuildRaytracingAccelerationStructure) (docs/bvh.md §4a). The leaf frame
//     compressionParams gives is not applied yet: the leaves are built from the geometry as the description has it.
//     The GPU cannot read offline dehydrated console streams directly (docs/bvh.md §4/§4a),
//     so source-only streams without geometry are refused with E_NOTIMPL. The title passes Scratch = 0, so the shim
//     supplies scratch: one buffer per call, sized via GetRaytracingAccelerationStructurePrebuildInfo, in which each
//     element has a region of its own (rehydrate_scratch).
//
// The builder object belongs to the title, which allocates it with a size this file doesn't know, so nothing is
// written into it: what the shim keeps (the device, the Morton option) lives in a table keyed by its address.
//
// LEONIDA_BVH_BUILDER=console turns the shim off, so the imports bind to the bundled DLL as before (for work on the
// console programs themselves, #5680 and #7318). The variable is read once, when the shim registers.
#include <windows.h>

#include <d3d12.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <vector>

#ifdef LEONIDA_VULKAN_DEVICE
#include "d3d12_com.h"   // rt::d3d12::hold_until_reset
#endif
#include "d3d12_xs_device.h"
#include "shims.h"
#include "../src/xdxr_descs.h"

namespace {

namespace xdxr = leonida::xdxr;

struct BuilderState {
    ID3D12Device5* device = nullptr;   // a reference held for the builder's lifetime; null if it has no DXR methods
    bool morton64 = false;
    // Rehydration scratch made for a list that cannot hold it (not the layer's own, rehydrate_scratch): kept until the
    // builder is destroyed, since the shim never learns when the GPU is done with it.
    std::vector<ID3D12Resource*> kept;
};

void release_state(BuilderState& state) {
    for (ID3D12Resource* scratch : state.kept) scratch->Release();
    state.kept.clear();
    if (state.device) state.device->Release();
    state.device = nullptr;
}

std::mutex g_mutex;
std::unordered_map<const void*, BuilderState> g_builders;

void log_once(std::atomic<bool>& flag, const char* text) {
    if (!flag.exchange(true)) std::fprintf(stderr, "[rt] XboxBvhBuilder shim: %s\n", text);
}

// The PC device behind what the title passed: the real device behind a d3d12_xs device proxy, or the pointer
// itself when it is already a device (a test, or a backend that hands out unwrapped devices).
ID3D12Device5* device5_of(void* device) {
    if (!device) return nullptr;
    IUnknown* base = rt::d3d12_xs::device_proxy_real_device(device);
    if (!base) base = static_cast<IUnknown*>(device);
    ID3D12Device5* out = nullptr;
    if (FAILED(base->QueryInterface(IID_PPV_ARGS(&out)))) return nullptr;
    return out;
}

// The command list a title passes, as the backend's own list: a d3d12_xs proxy is unwrapped first, so the PC
// interfaces below are asked of the real list and never of a console-shaped proxy (whose vtable is not the PC one;
// docs/d3d12_command_objects.md §3.5a). Lists aren't wrapped on main yet; a console list proxy joins
// proxy_real_object, which real_object_behind_proxy follows.
IUnknown* real_list(void* list) {
    if (!list) return nullptr;
    IUnknown* real = rt::d3d12_xs::real_object_behind_proxy(list);
    return real ? real : static_cast<IUnknown*>(list);
}

ID3D12GraphicsCommandList4* list4_of(void* list) {
    IUnknown* real = real_list(list);
    if (!real) return nullptr;
    ID3D12GraphicsCommandList4* out = nullptr;
    if (FAILED(real->QueryInterface(IID_PPV_ARGS(&out)))) return nullptr;
    return out;
}

// ??0XboxBvhBuilder@@QEAA@PEAUID3D12Device11@@_N@Z: XboxBvhBuilder(ID3D12Device11* device, bool).
// Returns `this`, as an MSVC constructor does. The bool's meaning is not measured; it is not needed here.
void* builder_construct(void* self, void* device, bool) {
    ID3D12Device5* device5 = device5_of(device);
    if (!device5) {
        static std::atomic<bool> s_logged{false};
        log_once(s_logged, "the device has no ID3D12Device5 (DXR) interface; builds will be skipped");
    }
    std::lock_guard<std::mutex> lock(g_mutex);
    BuilderState& state = g_builders[self];
    release_state(state);   // the same storage constructed twice without a destructor
    state = BuilderState{device5, false, {}};
    return self;
}

// ??1XboxBvhBuilder@@QEAA@XZ: ~XboxBvhBuilder().
void builder_destruct(void* self) {
    std::lock_guard<std::mutex> lock(g_mutex);
    auto it = g_builders.find(self);
    if (it == g_builders.end()) return;
    release_state(it->second);
    g_builders.erase(it);
}

// ?Set64BitMortonCodes@XboxBvhBuilder@@QEAAX_N@Z: a console-builder option (64-bit Morton keys for its sort).
void builder_set_64bit_morton_codes(void* self, bool enable) {
    std::lock_guard<std::mutex> lock(g_mutex);
    auto it = g_builders.find(self);
    if (it != g_builders.end()) it->second.morton64 = enable;
}

// The builder's device with a reference of the caller's own, taken under the lock, so a destructor on another
// thread can't release it during the call. The caller releases it.
ID3D12Device5* builder_device(void* self) {
    std::lock_guard<std::mutex> lock(g_mutex);
    auto it = g_builders.find(self);
    if (it == g_builders.end() || !it->second.device) return nullptr;
    it->second.device->AddRef();
    return it->second.device;
}

// Scratch for one RehydrateRaytracingAccelerationStructureBatched2X call, where the title passes
// ScratchAccelerationStructureData = 0 (measured on the retail title): a new buffer of `bytes`, in which each element
// that needs scratch has a region of its own. It is never shared with another call: two calls on one list (the title
// puts no barrier between them for scratch it does not know of) or lists recorded on several threads never write the
// same bytes. The list holds the buffer until it is Reset or released (rt::d3d12::hold_until_reset), and the layer
// destroys it after the submission that used it has retired; a list that cannot hold it leaves it to the builder,
// which keeps it until its destruction. Returns the buffer's GPU address, 0 when it could not be made.
UINT64 rehydrate_scratch(void* self, ID3D12Device5* device, ID3D12GraphicsCommandList4* list, UINT64 bytes) {
    if (!device || !bytes) return 0;
    D3D12_HEAP_PROPERTIES heap_props{};
    heap_props.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC res_desc{};
    res_desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    res_desc.Alignment = 0;
    res_desc.Width = bytes;
    res_desc.Height = 1;
    res_desc.DepthOrArraySize = 1;
    res_desc.MipLevels = 1;
    res_desc.Format = DXGI_FORMAT_UNKNOWN;
    res_desc.SampleDesc.Count = 1;
    res_desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    res_desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

    ID3D12Resource* res = nullptr;
    HRESULT hr = device->CreateCommittedResource(
        &heap_props, D3D12_HEAP_FLAG_NONE, &res_desc,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
        IID_PPV_ARGS(&res));
    if (FAILED(hr) || !res) return 0;
    const UINT64 va = res->GetGPUVirtualAddress();
    bool held = false;
#ifdef LEONIDA_VULKAN_DEVICE
    held = rt::d3d12::hold_until_reset(list, res);
#else
    (void)list;
#endif
    if (held) {
        res->Release();   // the list's reference keeps it from here
        return va;
    }
    static std::atomic<bool> s_logged{false};
    log_once(s_logged, "RehydrateRaytracingAccelerationStructureBatched2X: the list cannot hold its scratch; the "
                       "builder keeps it until it is destroyed");
    std::lock_guard<std::mutex> lock(g_mutex);
    auto it = g_builders.find(self);
    if (it == g_builders.end()) {   // destroyed during the call: nothing is recorded with this scratch
        res->Release();
        return 0;
    }
    it->second.kept.push_back(res);
    return va;
}

bool has_geometry(const D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS& inputs) {
    return inputs.NumDescs > 0 && (inputs.pGeometryDescs || inputs.InstanceDescs);
}

void note_stream(const xdxr::StreamDesc& stream) {
    if (stream.size_in_bytes == 0 || !stream.stream) return;
    static std::atomic<bool> s_logged{false};
    log_once(s_logged, "XDXR stream records (console builder settings) are ignored by the layer's builder");
}

// ?GetRaytracingAccelerationStructurePrebuildInfo@XboxBvhBuilder@@QEAAXPEBUD3D12_BUILD_RAYTRACING_ACCELERATION_
// STRUCTURE_INPUTS@@PEBUD3D12XBOX_XDXR_STREAM_DESC@@PEAUD3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO@@@Z
// The sizes the layer's own builder needs, which are the sizes the layer's build will use.
void builder_prebuild_info(void* self, const D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS* inputs,
                           const xdxr::StreamDesc* stream, D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO* info) {
    if (!info) return;
    *info = {};
    if (!inputs) return;
    if (stream) note_stream(*stream);
    ID3D12Device5* device = builder_device(self);
    if (!device) {
        static std::atomic<bool> s_logged{false};
        log_once(s_logged, "GetRaytracingAccelerationStructurePrebuildInfo on a builder without a DXR device; sizes are 0");
        return;
    }
    device->GetRaytracingAccelerationStructurePrebuildInfo(inputs, info);
    device->Release();
}

// ?BuildRaytracingAccelerationStructureBatched2X@XboxBvhBuilder@@QEAAXPEAUID3D12GraphicsCommandList6@@IPEBUD3D12XBOX_
// BUILD_RAYTRACING_ACCELERATION_STRUCTURE_BATCHED_DESC2@@@Z
// Each element is a public build description followed by its postbuild descriptions and a stream (xdxr_descs.h), so
// it is one BuildRaytracingAccelerationStructure on the list.
void builder_build_batched2x(void*, void* list, UINT count, const xdxr::BatchedDesc2* descs) {
    static_assert(sizeof(D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC) == 0x30, "the BATCHED_DESC2 prefix");
    static_assert(sizeof(D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC) == sizeof(xdxr::PostbuildInfoDesc),
                  "postbuild descriptions are the public ones");
    if (count == 0) return;
    if (!descs) return;
    ID3D12GraphicsCommandList4* list4 = list4_of(list);
    if (!list4) {
        static std::atomic<bool> s_logged{false};
        log_once(s_logged, "BuildRaytracingAccelerationStructureBatched2X: the list has no ID3D12GraphicsCommandList4");
        return;
    }
    for (UINT i = 0; i < count; ++i) {
        const xdxr::BatchedDesc2& element = descs[i];
        note_stream(element.stream);
        D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC build;
        std::memcpy(&build, &element, sizeof(build));
        // The postbuild descriptions, without the offline-statistics type the DLL refuses (docs/xdxr.md §4).
        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC kept[16];
        UINT n_kept = 0;
        bool skipped_statistics = false;
        const auto* post = reinterpret_cast<const D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC*>(
            element.postbuild_info_descs);
        for (UINT p = 0; post && p < element.num_postbuild_info_descs; ++p) {
            if (static_cast<uint32_t>(post[p].InfoType) == xdxr::kPostbuildInfoOfflineStatistics) {
                skipped_statistics = true;
                continue;
            }
            if (n_kept == 16) {
                static std::atomic<bool> s_logged{false};
                log_once(s_logged, "more than 16 postbuild descriptions in one element; the rest are dropped");
                break;
            }
            kept[n_kept++] = post[p];
        }
        if (skipped_statistics) {
            // The DLL throws E_INVALIDARG for this element before any work; the shim skips it the same way.
            static std::atomic<bool> s_logged{false};
            log_once(s_logged, "an element asked for offline statistics (0x10), which the DLL refuses; element skipped");
            continue;
        }
        list4->BuildRaytracingAccelerationStructure(&build, n_kept, n_kept ? kept : nullptr);
    }
    list4->Release();
}

// ?CopyRaytracingAccelerationStructure@XboxBvhBuilder@@QEAAXPEAUID3D12GraphicsCommandList4@@_K1W4D3D12_RAYTRACING_
// ACCELERATION_STRUCTURE_COPY_MODE@@@Z
void builder_copy(void*, void* list, UINT64 dest, UINT64 source, D3D12_RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE mode) {
    ID3D12GraphicsCommandList4* list4 = list4_of(list);
    if (!list4) {
        static std::atomic<bool> s_logged{false};
        log_once(s_logged, "CopyRaytracingAccelerationStructure: the list has no ID3D12GraphicsCommandList4");
        return;
    }
    list4->CopyRaytracingAccelerationStructure(dest, source, mode);
    list4->Release();
}

// ?RehydrateRaytracingAccelerationStructureBatched2X@XboxBvhBuilder@@QEAAJPEAUID3D12GraphicsCommandList8@@IPEBUD3D12_
// BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC@@PEB_K@Z -> HRESULT.
// Rebuilds each acceleration structure from geometry on the command list.
HRESULT builder_rehydrate_batched2x(void* self, void* list, UINT count,
                                    const D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC* descs,
                                    const UINT64* source_data) {
    if (count == 0) return S_OK;
    if (!descs) return E_INVALIDARG;
    if (!list) return E_INVALIDARG;
    ID3D12GraphicsCommandList4* list4 = list4_of(list);
    if (!list4) {
        static std::atomic<bool> s_logged{false};
        log_once(s_logged, "RehydrateRaytracingAccelerationStructureBatched2X: the list has no ID3D12GraphicsCommandList4");
        return E_NOINTERFACE;
    }
    ID3D12Device5* device = builder_device(self);
    // First pass: the descriptions as they are recorded, and for each element that builds without scratch of the
    // title's a region of its own in one scratch buffer for the batch, at D3D12's acceleration-structure alignment
    // (256 bytes, which the layer checks scratch addresses against). The XDXR recorder leaves the dependency between
    // builds to the caller's barrier (src/xdxr_build.h, dispatch()) and the title puts none between the elements of
    // a batch, so elements must not share scratch bytes. The first source-only element ends the batch (below).
    constexpr UINT64 kAlign = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BYTE_ALIGNMENT;
    constexpr UINT64 kNoRegion = ~0ull;
    std::vector<D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC> builds;
    std::vector<UINT64> regions;
    builds.reserve(count);
    regions.reserve(count);
    UINT64 scratch_bytes = 0;
    bool source_only = false;
    for (UINT i = 0; i < count; ++i) {
        D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC build = descs[i];
        const UINT64 src = source_data ? source_data[i] : build.SourceAccelerationStructureData;
        // A rehydrate is a fresh build from geometry. The title sets SourceAccelerationStructureData to the
        // dehydrated stream (measured on the retail title), and the layer reads that address for an update
        // (d3d12_com.cpp, BuildRaytracingAccelerationStructure), so with PERFORM_UPDATE kept it would refit from
        // the stream's bytes. Both are cleared.
        if (build.Inputs.Flags & D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PERFORM_UPDATE) {
            static std::atomic<bool> s_update_logged{false};
            log_once(s_update_logged,
                     "RehydrateRaytracingAccelerationStructureBatched2X: an element asks for PERFORM_UPDATE; it is "
                     "built afresh from its geometry instead (the source is the dehydrated stream)");
            build.Inputs.Flags &= ~D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PERFORM_UPDATE;
        }
        build.SourceAccelerationStructureData = 0;
        UINT64 region = kNoRegion;
        if (has_geometry(build.Inputs)) {
            if (!build.ScratchAccelerationStructureData && device) {
                D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO prebuild{};
                device->GetRaytracingAccelerationStructurePrebuildInfo(&build.Inputs, &prebuild);
                const UINT64 size = prebuild.ScratchDataSizeInBytes ? prebuild.ScratchDataSizeInBytes : 1;
                region = scratch_bytes;
                scratch_bytes += (size + kAlign - 1) / kAlign * kAlign;
            }
        } else if (src) {
            source_only = true;
            break;
        }
        builds.push_back(build);
        regions.push_back(region);
    }
    const UINT64 scratch = scratch_bytes ? rehydrate_scratch(self, device, list4, scratch_bytes) : 0;

    for (std::size_t i = 0; i < builds.size(); ++i) {
        D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC& build = builds[i];
        if (has_geometry(build.Inputs)) {
            // The title passes ScratchAccelerationStructureData = 0 (measured on the retail title).
            // Supply a scratch region to prevent d3d12_com.cpp from poisoning the command list.
            if (regions[i] != kNoRegion && scratch) build.ScratchAccelerationStructureData = scratch + regions[i];
            if (!build.ScratchAccelerationStructureData) {
                static std::atomic<bool> s_scratch_logged{false};
                log_once(s_scratch_logged,
                         "RehydrateRaytracingAccelerationStructureBatched2X: ScratchAccelerationStructureData is 0 "
                         "and could not be allocated; skipping build to avoid command list rejection");
                continue;
            }
            list4->BuildRaytracingAccelerationStructure(&build, 0, nullptr);
        } else {
            static std::atomic<bool> s_empty_logged{false};
            log_once(s_empty_logged,
                     "RehydrateRaytracingAccelerationStructureBatched2X: element has NumDescs == 0 with no geometry; "
                     "skipping empty structure");
        }
    }
    if (source_only) {
        // docs/bvh.md §4/§4a: the rehydrate source is a dehydrated console stream (element chunks,
        // level table, box overrides), not a D3D12 serialized blob. On the XDXR path, a DESERIALIZE
        // copy is replayed as a byte clone, so the destination would hold dehydrated bytes the GPU
        // traversal shaders cannot read. Rehydration requires geometry reconstruction (docs/bvh.md §4a).
        // Refusing source-only rehydration without geometry.
        static std::atomic<bool> s_source_only_logged{false};
        log_once(s_source_only_logged,
                 "RehydrateRaytracingAccelerationStructureBatched2X: source-only dehydrated stream without "
                 "geometry cannot be deserialized as a D3D12 BLAS (docs/bvh.md §4/§4a); refusing element");
    }
    if (device) device->Release();
    list4->Release();
    return source_only ? E_NOTIMPL : S_OK;
}

constexpr const char kDll[] = "xboxbvhbuilder.dll";

const rt::shims::Entry kEntries[] = {
    {kDll, "??0XboxBvhBuilder@@QEAA@PEAUID3D12Device11@@_N@Z", reinterpret_cast<void*>(&builder_construct)},
    {kDll, "??1XboxBvhBuilder@@QEAA@XZ", reinterpret_cast<void*>(&builder_destruct)},
    {kDll, "?Set64BitMortonCodes@XboxBvhBuilder@@QEAAX_N@Z", reinterpret_cast<void*>(&builder_set_64bit_morton_codes)},
    {kDll,
     "?GetRaytracingAccelerationStructurePrebuildInfo@XboxBvhBuilder@@QEAAXPEBUD3D12_BUILD_RAYTRACING_ACCELERATION_"
     "STRUCTURE_INPUTS@@PEBUD3D12XBOX_XDXR_STREAM_DESC@@PEAUD3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO@@@Z",
     reinterpret_cast<void*>(&builder_prebuild_info)},
    {kDll,
     "?BuildRaytracingAccelerationStructureBatched2X@XboxBvhBuilder@@QEAAXPEAUID3D12GraphicsCommandList6@@IPEBUD3D12"
     "XBOX_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_BATCHED_DESC2@@@Z",
     reinterpret_cast<void*>(&builder_build_batched2x)},
    {kDll,
     "?CopyRaytracingAccelerationStructure@XboxBvhBuilder@@QEAAXPEAUID3D12GraphicsCommandList4@@_K1W4D3D12_"
     "RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE@@@Z",
     reinterpret_cast<void*>(&builder_copy)},
    {kDll,
     "?RehydrateRaytracingAccelerationStructureBatched2X@XboxBvhBuilder@@QEAAJPEAUID3D12GraphicsCommandList8@@IPEBU"
     "D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC@@PEB_K@Z",
     reinterpret_cast<void*>(&builder_rehydrate_batched2x)},
};

bool shim_enabled() {
    char value[16] = {};
    size_t length = 0;
    if (getenv_s(&length, value, "LEONIDA_BVH_BUILDER") != 0 || length <= 1) return true;
    return _stricmp(value, "console") != 0;
}

// Registers the table unless LEONIDA_BVH_BUILDER=console asks for the bundled DLL.
struct Registration {
    Registration() {
        if (!shim_enabled()) return;
        static rt::shims::Table table(kEntries, sizeof(kEntries) / sizeof(kEntries[0]));
    }
} g_registration;

}  // namespace

namespace rt::shims {
// For tests: the builders constructed and not destroyed.
size_t xbox_bvh_builder_live_count() {
    std::lock_guard<std::mutex> lock(g_mutex);
    return g_builders.size();
}
}  // namespace rt::shims
