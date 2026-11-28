// The XboxBvhBuilder.dll shim (shims/xbox_bvh_builder.cpp) on the standalone Vulkan device, called the way the
// title's imports bind: rt::shims::find("xboxbvhbuilder.dll", <mangled export>), `this` first. A BLAS and a TLAS are
// built through BuildRaytracingAccelerationStructureBatched2X, the BLAS is compacted through
// CopyRaytracingAccelerationStructure, and the three are written to <out dir> as <case>.bin + <case>.json for
// `python -I tools/xdxr_bvh.py check-build <out dir>` (ctest: xbox_bvh_builder + xbox_bvh_builder_check), which
// checks the records and traces rays against brute force. Also checked: the shim's prebuild sizes are the layer's,
// an element asking for offline statistics is skipped and leaves its destination untouched, a non-empty XDXR stream
// is accepted, rehydration rebuilds from geometry with S_OK (with the title's scratch, with none, and two elements of
// one batch without scratch, which the shim gives regions of their own; their structures go to check-build too) and
// refuses a source-only element, and builders are released.
//
//   xbox_bvh_builder_test <out dir>        the shim
//   xbox_bvh_builder_test --console        run with LEONIDA_BVH_BUILDER=console: nothing is registered
//
// Exit 77 without a Vulkan device. No game files.
#include "../shims/d3d12_com.h"
#include "../shims/d3d12_xs_device.h"
#include "../shims/shims.h"
#include "../src/xdxr_descs.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

namespace rt::shims {
size_t xbox_bvh_builder_live_count();
}

namespace {

namespace xdxr = leonida::xdxr;

int g_checks = 0;
int g_failures = 0;

void check(bool condition, const char* what) {
    ++g_checks;
    std::printf("%s %s\n", condition ? "PASS" : "FAIL", what);
    if (!condition) ++g_failures;
}

template<class T> struct Ref {
    T* p = nullptr;
    ~Ref() { if (p) p->Release(); }
    T* operator->() const { return p; }
    T** put() { return &p; }
};

int fail(const char* what, HRESULT hr = S_OK) {
    std::fprintf(stderr, "FAIL: %s (hr 0x%08lx)\n", what, static_cast<unsigned long>(hr));
    return 1;
}

HRESULT buffer(ID3D12Device* device, D3D12_HEAP_TYPE heap, UINT64 size, D3D12_RESOURCE_STATES state,
               D3D12_RESOURCE_FLAGS flags, ID3D12Resource** out) {
    D3D12_HEAP_PROPERTIES props{};
    props.Type = heap;
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = size ? size : 1;
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    desc.Flags = flags;
    return device->CreateCommittedResource(&props, D3D12_HEAP_FLAG_NONE, &desc, state, nullptr, IID_PPV_ARGS(out));
}

std::string num(double x) {
    char s[64];
    std::snprintf(s, sizeof s, "%.9g", x);
    return s;
}

bool write_file(const std::string& path, const void* data, std::size_t bytes) {
    std::ofstream f(path, std::ios::binary);
    f.write(static_cast<const char*>(data), std::streamsize(bytes));
    return bool(f);
}

// The exports, as the executable imports them.
constexpr const char kDll[] = "xboxbvhbuilder.dll";
constexpr const char kConstruct[] = "??0XboxBvhBuilder@@QEAA@PEAUID3D12Device11@@_N@Z";
constexpr const char kDestruct[] = "??1XboxBvhBuilder@@QEAA@XZ";
constexpr const char kMorton[] = "?Set64BitMortonCodes@XboxBvhBuilder@@QEAAX_N@Z";
constexpr const char kPrebuild[] =
    "?GetRaytracingAccelerationStructurePrebuildInfo@XboxBvhBuilder@@QEAAXPEBUD3D12_BUILD_RAYTRACING_ACCELERATION_"
    "STRUCTURE_INPUTS@@PEBUD3D12XBOX_XDXR_STREAM_DESC@@PEAUD3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO@@@Z";
constexpr const char kBatched[] =
    "?BuildRaytracingAccelerationStructureBatched2X@XboxBvhBuilder@@QEAAXPEAUID3D12GraphicsCommandList6@@IPEBUD3D12"
    "XBOX_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_BATCHED_DESC2@@@Z";
constexpr const char kCopy[] =
    "?CopyRaytracingAccelerationStructure@XboxBvhBuilder@@QEAAXPEAUID3D12GraphicsCommandList4@@_K1W4D3D12_"
    "RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE@@@Z";
constexpr const char kRehydrate[] =
    "?RehydrateRaytracingAccelerationStructureBatched2X@XboxBvhBuilder@@QEAAJPEAUID3D12GraphicsCommandList8@@IPEBU"
    "D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC@@PEB_K@Z";

using Construct = void*(*)(void*, void*, bool);
using Destruct = void (*)(void*);
using Morton = void (*)(void*, bool);
using Prebuild = void (*)(void*, const D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS*, const xdxr::StreamDesc*,
                          D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO*);
using Batched = void (*)(void*, void*, UINT, const xdxr::BatchedDesc2*);
using Copy = void (*)(void*, void*, UINT64, UINT64, D3D12_RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE);
using Rehydrate = HRESULT (*)(void*, void*, UINT, const D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC*,
                              const UINT64*);

template <class F> F shim(const char* name) { return reinterpret_cast<F>(rt::shims::find(kDll, name)); }

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: xbox_bvh_builder_test <out dir> | --console\n");
        return 2;
    }
    if (std::strcmp(argv[1], "--console") == 0) {
        // LEONIDA_BVH_BUILDER=console: the imports must bind to the bundled DLL, so the shim registers nothing.
        check(shim<Construct>(kConstruct) == nullptr && shim<Batched>(kBatched) == nullptr &&
                  !rt::shims::owns_dll(kDll),
              "LEONIDA_BVH_BUILDER=console registers none of the exports");
        std::printf("xbox_bvh_builder_test: %d of %d checks failed\n", g_failures, g_checks);
        return g_failures ? 1 : 0;
    }
    const std::string dir = argv[1];
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    for (const auto& entry : std::filesystem::directory_iterator(dir, ec)) {
        const auto ext = entry.path().extension();
        if (ext == ".bin" || ext == ".json") std::filesystem::remove(entry.path(), ec);
    }

    // ---- the seven exports are registered ----
    const auto construct = shim<Construct>(kConstruct);
    const auto destruct = shim<Destruct>(kDestruct);
    const auto morton = shim<Morton>(kMorton);
    const auto prebuild = shim<Prebuild>(kPrebuild);
    const auto batched = shim<Batched>(kBatched);
    const auto copy = shim<Copy>(kCopy);
    const auto rehydrate = shim<Rehydrate>(kRehydrate);
    check(construct && destruct && morton && prebuild && batched && copy && rehydrate && rt::shims::owns_dll(kDll),
          "the seven exports the executable imports are served by the shim");
    if (!(construct && destruct && morton && prebuild && batched && copy && rehydrate)) return 1;

    char rt_as[8] = {};
    std::size_t rt_as_length = 0;
    if (getenv_s(&rt_as_length, rt_as, "LEONIDA_RT_AS") == 0 && rt_as_length > 1 && std::strcmp(rt_as, "khr") == 0) {
        std::puts("SKIP: LEONIDA_RT_AS=khr");
        return 77;
    }
    Ref<ID3D12Device5> device;
    if (FAILED(rt::d3d12::create_vulkan_device_standalone(IID_PPV_ARGS(device.put())))) {
        std::puts("SKIP: no Vulkan D3D12 device");
        return 77;
    }

    // ---- the builder object: the title's storage, larger than anything the shim could write ----
    alignas(16) unsigned char builder[256];
    std::memset(builder, 0xab, sizeof builder);
    check(construct(builder, device.p, true) == builder, "the constructor returns `this`");
    bool untouched = true;
    for (unsigned char b : builder) untouched = untouched && b == 0xab;
    check(untouched && rt::shims::xbox_bvh_builder_live_count() == 1,
          "nothing is written into the title's builder object; its state is kept aside");
    morton(builder, true);

    // ---- geometry: a triangle soup ----
    std::mt19937 rng(23);
    std::uniform_real_distribution<float> spread(-4.0f, 4.0f), jitter(-0.6f, 0.6f);
    const UINT tris = 400;
    std::vector<float> soup;
    for (UINT t = 0; t < tris; ++t) {
        const float c[3] = {spread(rng), spread(rng), spread(rng)};
        for (int k = 0; k < 9; ++k) soup.push_back(c[k % 3] + jitter(rng));
    }
    const UINT64 soup_bytes = soup.size() * 4;
    Ref<ID3D12Resource> upload;
    HRESULT hr = buffer(device.p, D3D12_HEAP_TYPE_UPLOAD, 256 + soup_bytes + 3 * 64, D3D12_RESOURCE_STATE_GENERIC_READ,
                        D3D12_RESOURCE_FLAG_NONE, upload.put());
    if (FAILED(hr)) return fail("upload buffer", hr);
    char* mapped = nullptr;
    if (FAILED(upload->Map(0, nullptr, reinterpret_cast<void**>(&mapped)))) return fail("Map upload");
    std::memcpy(mapped, soup.data(), std::size_t(soup_bytes));
    const UINT64 soup_va = upload->GetGPUVirtualAddress();

    D3D12_RAYTRACING_GEOMETRY_DESC geometry{};
    geometry.Type = D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES;
    geometry.Flags = D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE;
    geometry.Triangles.VertexFormat = DXGI_FORMAT_R32G32B32_FLOAT;
    geometry.Triangles.VertexCount = tris * 3;
    geometry.Triangles.VertexBuffer = {soup_va, 12};
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS blas_inputs{};
    blas_inputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
    blas_inputs.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE |
                        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_COMPACTION;
    blas_inputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
    blas_inputs.NumDescs = 1;
    blas_inputs.pGeometryDescs = &geometry;

    // ---- GetRaytracingAccelerationStructurePrebuildInfo: the layer's sizes, with or without a stream ----
    const xdxr::StreamRecord2 record{2, 0, -1.0f, 0.0f};   // a type-2 record with its defaults
    const xdxr::StreamDesc stream{sizeof record, &record};
    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO blas_info{}, layer_info{};
    prebuild(builder, &blas_inputs, &stream, &blas_info);
    device->GetRaytracingAccelerationStructurePrebuildInfo(&blas_inputs, &layer_info);
    check(blas_info.ResultDataMaxSizeInBytes == layer_info.ResultDataMaxSizeInBytes &&
              blas_info.ScratchDataSizeInBytes == layer_info.ScratchDataSizeInBytes,
          "the shim's prebuild sizes are the layer's own");
    if (!blas_info.ResultDataMaxSizeInBytes) {
        std::puts("SKIP: the device does not build XDXR acceleration structures (no bufferDeviceAddress)");
        return 77;
    }
    // The title's device is a d3d12_xs device proxy (D3D12XboxCreateDevice): a builder constructed on one reaches the
    // real device behind it.
    {
        Ref<ID3D12Device> proxy;
        alignas(16) unsigned char proxied[64];
        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO through_proxy{};
        device->AddRef();   // create_device_proxy takes one reference to the real device on success
        const bool wrapped = SUCCEEDED(rt::d3d12_xs::create_device_proxy(device.p, IID_PPV_ARGS(proxy.put())));
        if (!wrapped) device->Release();
        const bool made = wrapped && rt::d3d12_xs::device_proxy_real_device(proxy.p) != nullptr;
        if (made) {
            construct(proxied, proxy.p, true);
            prebuild(proxied, &blas_inputs, nullptr, &through_proxy);
            destruct(proxied);
        }
        check(made && through_proxy.ResultDataMaxSizeInBytes == layer_info.ResultDataMaxSizeInBytes &&
                  through_proxy.ScratchDataSizeInBytes == layer_info.ScratchDataSizeInBytes,
              "a builder constructed on a d3d12_xs device proxy uses the real device behind it");
    }
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS tlas_inputs{};
    tlas_inputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL;
    tlas_inputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
    tlas_inputs.NumDescs = 2;
    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO tlas_info{};
    prebuild(builder, &tlas_inputs, nullptr, &tlas_info);
    if (!tlas_info.ResultDataMaxSizeInBytes) return fail("TLAS prebuild info is empty");

    // ---- buffers ----
    const D3D12_RESOURCE_FLAGS uav = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    const D3D12_RESOURCE_STATES as_state = D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE;
    Ref<ID3D12Resource> blas, refused, tlas, scratch, scratch2, info, blas_rb, refused_rb, tlas_rb, info_rb;
    const UINT64 scratch_size = blas_info.ScratchDataSizeInBytes > tlas_info.ScratchDataSizeInBytes
                                    ? blas_info.ScratchDataSizeInBytes : tlas_info.ScratchDataSizeInBytes;
    if (FAILED(hr = buffer(device.p, D3D12_HEAP_TYPE_DEFAULT, blas_info.ResultDataMaxSizeInBytes, as_state, uav, blas.put())) ||
        FAILED(hr = buffer(device.p, D3D12_HEAP_TYPE_DEFAULT, blas_info.ResultDataMaxSizeInBytes, as_state, uav, refused.put())) ||
        FAILED(hr = buffer(device.p, D3D12_HEAP_TYPE_DEFAULT, tlas_info.ResultDataMaxSizeInBytes, as_state, uav, tlas.put())) ||
        FAILED(hr = buffer(device.p, D3D12_HEAP_TYPE_DEFAULT, scratch_size, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, uav, scratch.put())) ||
        FAILED(hr = buffer(device.p, D3D12_HEAP_TYPE_DEFAULT, scratch_size, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, uav, scratch2.put())) ||
        FAILED(hr = buffer(device.p, D3D12_HEAP_TYPE_DEFAULT, 16, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, uav, info.put())) ||
        FAILED(hr = buffer(device.p, D3D12_HEAP_TYPE_READBACK, blas_info.ResultDataMaxSizeInBytes,
                           D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_FLAG_NONE, blas_rb.put())) ||
        FAILED(hr = buffer(device.p, D3D12_HEAP_TYPE_READBACK, blas_info.ResultDataMaxSizeInBytes,
                           D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_FLAG_NONE, refused_rb.put())) ||
        FAILED(hr = buffer(device.p, D3D12_HEAP_TYPE_READBACK, tlas_info.ResultDataMaxSizeInBytes,
                           D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_FLAG_NONE, tlas_rb.put())) ||
        FAILED(hr = buffer(device.p, D3D12_HEAP_TYPE_READBACK, 16, D3D12_RESOURCE_STATE_COPY_DEST,
                           D3D12_RESOURCE_FLAG_NONE, info_rb.put())))
        return fail("acceleration-structure buffers", hr);

    // ---- instances ----
    const float c = std::cos(0.9f), s = std::sin(0.9f);
    const float xforms[2][12] = {{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0}, {c, 0, s, 10, 0, 1, 0, -2, -s, 0, c, 1}};
    D3D12_RAYTRACING_INSTANCE_DESC instances[2]{};
    for (int i = 0; i < 2; ++i) {
        std::memcpy(instances[i].Transform, xforms[i], sizeof xforms[i]);
        instances[i].InstanceID = 40 + i;
        instances[i].InstanceMask = 0xff;
        instances[i].AccelerationStructure = blas->GetGPUVirtualAddress();
    }
    const UINT64 instances_at = (soup_bytes + 255) / 256 * 256;
    std::memcpy(mapped + instances_at, instances, sizeof instances);
    upload->Unmap(0, nullptr);
    tlas_inputs.InstanceDescs = upload->GetGPUVirtualAddress() + instances_at;

    D3D12_COMMAND_QUEUE_DESC qd{};
    qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    Ref<ID3D12CommandQueue> queue;
    Ref<ID3D12CommandAllocator> allocator;
    Ref<ID3D12GraphicsCommandList4> list;
    Ref<ID3D12Fence> fence;
    if (FAILED(device->CreateCommandQueue(&qd, IID_PPV_ARGS(queue.put()))) ||
        FAILED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(allocator.put()))) ||
        FAILED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, nullptr, IID_PPV_ARGS(list.put()))) ||
        FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(fence.put()))))
        return fail("queue, allocator, list or fence");
    UINT64 fence_value = 0;
    auto submit = [&]() {
        if (FAILED(list->Close())) return false;
        ID3D12CommandList* lists[] = {list.p};
        queue->ExecuteCommandLists(1, lists);
        if (FAILED(queue->Signal(fence.p, ++fence_value))) return false;
        HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        const bool done = event && SUCCEEDED(fence->SetEventOnCompletion(fence_value, event)) &&
                          WaitForSingleObject(event, 10000) == WAIT_OBJECT_0;
        if (event) CloseHandle(event);
        return done && SUCCEEDED(device->GetDeviceRemovedReason());
    };
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;

    // ---- BuildRaytracingAccelerationStructureBatched2X: the BLAS (with a stream and a COMPACTED_SIZE postbuild),
    // and an element asking for offline statistics, which the DLL refuses and the shim skips ----
    const xdxr::PostbuildInfoDesc compacted_post{info->GetGPUVirtualAddress(),
                                                 D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_COMPACTED_SIZE, 0};
    const xdxr::PostbuildInfoDesc statistics_post{info->GetGPUVirtualAddress() + 8,
                                                  xdxr::kPostbuildInfoOfflineStatistics, 0};
    xdxr::BatchedDesc2 elements[2]{};
    for (xdxr::BatchedDesc2& e : elements) {
        std::memcpy(&e.inputs, &blas_inputs, sizeof e.inputs);
        e.stream = stream;
    }
    elements[0].dest_acceleration_structure_data = blas->GetGPUVirtualAddress();
    elements[0].scratch_acceleration_structure_data = scratch->GetGPUVirtualAddress();
    elements[0].num_postbuild_info_descs = 1;
    elements[0].postbuild_info_descs = &compacted_post;
    elements[1].dest_acceleration_structure_data = refused->GetGPUVirtualAddress();
    elements[1].scratch_acceleration_structure_data = scratch2->GetGPUVirtualAddress();
    elements[1].num_postbuild_info_descs = 1;
    elements[1].postbuild_info_descs = &statistics_post;
    batched(builder, list.p, 2, elements);
    list->ResourceBarrier(1, &barrier);
    xdxr::BatchedDesc2 top{};
    top.dest_acceleration_structure_data = tlas->GetGPUVirtualAddress();
    std::memcpy(&top.inputs, &tlas_inputs, sizeof top.inputs);
    top.scratch_acceleration_structure_data = scratch->GetGPUVirtualAddress();
    batched(builder, list.p, 1, &top);
    batched(builder, list.p, 0, nullptr);   // an empty batch does nothing
    list->ResourceBarrier(1, &barrier);
    list->CopyBufferRegion(blas_rb.p, 0, blas.p, 0, blas_info.ResultDataMaxSizeInBytes);
    list->CopyBufferRegion(refused_rb.p, 0, refused.p, 0, blas_info.ResultDataMaxSizeInBytes);
    list->CopyBufferRegion(tlas_rb.p, 0, tlas.p, 0, tlas_info.ResultDataMaxSizeInBytes);
    list->CopyBufferRegion(info_rb.p, 0, info.p, 0, 16);
    if (!submit()) return fail("the batched builds did not complete");

    UINT64 post[2] = {};
    {
        void* p = nullptr;
        if (FAILED(info_rb->Map(0, nullptr, &p))) return fail("Map postbuild");
        std::memcpy(post, p, sizeof post);
        info_rb->Unmap(0, nullptr);
    }
    check(post[0] && post[0] % 64 == 0 && post[0] <= blas_info.ResultDataMaxSizeInBytes,
          "the BLAS element's COMPACTED_SIZE postbuild is written");
    check(post[1] == 0, "the offline-statistics element is skipped: its postbuild destination is untouched");
    {
        void* p = nullptr;
        std::uint32_t units = 0;
        if (SUCCEEDED(refused_rb->Map(0, nullptr, &p))) {
            std::memcpy(&units, static_cast<char*>(p) + 4, 4);
            refused_rb->Unmap(0, nullptr);
        }
        check(units == 0, "and no structure was built into its destination");
    }

    // ---- CopyRaytracingAccelerationStructure COMPACT into a buffer of exactly COMPACTED_SIZE ----
    Ref<ID3D12Resource> compacted, compacted_rb;
    if (FAILED(hr = buffer(device.p, D3D12_HEAP_TYPE_DEFAULT, post[0], as_state, uav, compacted.put())) ||
        FAILED(hr = buffer(device.p, D3D12_HEAP_TYPE_READBACK, post[0], D3D12_RESOURCE_STATE_COPY_DEST,
                           D3D12_RESOURCE_FLAG_NONE, compacted_rb.put())))
        return fail("compacted buffers", hr);
    if (FAILED(allocator->Reset()) || FAILED(list->Reset(allocator.p, nullptr))) return fail("Reset for compaction");
    copy(builder, list.p, compacted->GetGPUVirtualAddress(), blas->GetGPUVirtualAddress(),
         D3D12_RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE_COMPACT);
    list->ResourceBarrier(1, &barrier);
    list->CopyBufferRegion(compacted_rb.p, 0, compacted.p, 0, post[0]);
    if (!submit()) return fail("the compaction did not complete");

    // ---- RehydrateRaytracingAccelerationStructureBatched2X: rebuilds from geometry ----
    Ref<ID3D12Resource> rehydrated_as, rehydrated_rb;
    if (FAILED(hr = buffer(device.p, D3D12_HEAP_TYPE_DEFAULT, blas_info.ResultDataMaxSizeInBytes, as_state, uav,
                           rehydrated_as.put())) ||
        FAILED(hr = buffer(device.p, D3D12_HEAP_TYPE_READBACK, blas_info.ResultDataMaxSizeInBytes,
                           D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_FLAG_NONE, rehydrated_rb.put())))
        return fail("rehydrated buffers", hr);
    if (FAILED(allocator->Reset()) || FAILED(list->Reset(allocator.p, nullptr))) return fail("Reset for rehydrate");

    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC rehydrate_desc{};
    rehydrate_desc.DestAccelerationStructureData = rehydrated_as->GetGPUVirtualAddress();
    rehydrate_desc.Inputs = blas_inputs;
    rehydrate_desc.ScratchAccelerationStructureData = scratch->GetGPUVirtualAddress();
    const UINT64 source = blas->GetGPUVirtualAddress();

    check(rehydrate(builder, nullptr, 1, &rehydrate_desc, &source) == E_INVALIDARG,
          "RehydrateRaytracingAccelerationStructureBatched2X rejects null list");
    check(rehydrate(builder, list.p, 1, nullptr, &source) == E_INVALIDARG,
          "RehydrateRaytracingAccelerationStructureBatched2X rejects null descs");
    check(rehydrate(builder, list.p, 0, nullptr, nullptr) == S_OK,
          "RehydrateRaytracingAccelerationStructureBatched2X with count 0 returns S_OK");

    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC source_only_desc{};
    source_only_desc.DestAccelerationStructureData = rehydrated_as->GetGPUVirtualAddress();
    check(rehydrate(builder, list.p, 1, &source_only_desc, &source) == E_NOTIMPL,
          "RehydrateRaytracingAccelerationStructureBatched2X refuses source-only stream without geometry");

    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC empty_desc{};
    check(rehydrate(builder, list.p, 1, &empty_desc, nullptr) == S_OK,
          "RehydrateRaytracingAccelerationStructureBatched2X logs and skips empty structure");

    check(rehydrate(builder, list.p, 1, &rehydrate_desc, &source) == S_OK,
          "RehydrateRaytracingAccelerationStructureBatched2X rebuilds BLAS and returns S_OK");

    // The title passes ScratchAccelerationStructureData = 0: the shim supplies scratch. Each of these builds has a
    // destination of its own, read back below, so a build the shim skipped fails its check.
    Ref<ID3D12Resource> zero_scratch_as, zero_scratch_rb;
    if (FAILED(hr = buffer(device.p, D3D12_HEAP_TYPE_DEFAULT, blas_info.ResultDataMaxSizeInBytes, as_state, uav,
                           zero_scratch_as.put())) ||
        FAILED(hr = buffer(device.p, D3D12_HEAP_TYPE_READBACK, blas_info.ResultDataMaxSizeInBytes,
                           D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_FLAG_NONE, zero_scratch_rb.put())))
        return fail("zero-scratch buffers", hr);
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC zero_scratch_desc = rehydrate_desc;
    zero_scratch_desc.DestAccelerationStructureData = zero_scratch_as->GetGPUVirtualAddress();
    zero_scratch_desc.ScratchAccelerationStructureData = 0;
    check(rehydrate(builder, list.p, 1, &zero_scratch_desc, &source) == S_OK,
          "RehydrateRaytracingAccelerationStructureBatched2X supplies scratch when ScratchAccelerationStructureData == 0");

    // Two elements of one batch, both without scratch, and no barrier between them (the title puts none): each needs
    // a scratch region of its own. The second element has half the triangles, and asks for PERFORM_UPDATE with a
    // source, as the title's dehydrated stream would be: a rehydrate builds it afresh all the same.
    const UINT pair_tris[2] = {tris, tris / 2};
    D3D12_RAYTRACING_GEOMETRY_DESC pair_geometry[2] = {geometry, geometry};
    pair_geometry[1].Triangles.VertexCount = pair_tris[1] * 3;
    Ref<ID3D12Resource> pair_as[2], pair_rb[2];
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC pair_desc[2]{};
    for (int i = 0; i < 2; ++i) {
        if (FAILED(hr = buffer(device.p, D3D12_HEAP_TYPE_DEFAULT, blas_info.ResultDataMaxSizeInBytes, as_state, uav,
                               pair_as[i].put())) ||
            FAILED(hr = buffer(device.p, D3D12_HEAP_TYPE_READBACK, blas_info.ResultDataMaxSizeInBytes,
                               D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_FLAG_NONE, pair_rb[i].put())))
            return fail("batch-of-two buffers", hr);
        pair_desc[i].DestAccelerationStructureData = pair_as[i]->GetGPUVirtualAddress();
        pair_desc[i].Inputs = blas_inputs;
        pair_desc[i].Inputs.pGeometryDescs = &pair_geometry[i];
    }
    pair_desc[1].Inputs.Flags |= D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PERFORM_UPDATE;
    pair_desc[1].SourceAccelerationStructureData = source;
    check(rehydrate(builder, list.p, 2, pair_desc, nullptr) == S_OK,
          "RehydrateRaytracingAccelerationStructureBatched2X builds a batch of two elements without scratch");

    list->ResourceBarrier(1, &barrier);
    list->CopyBufferRegion(rehydrated_rb.p, 0, rehydrated_as.p, 0, blas_info.ResultDataMaxSizeInBytes);
    list->CopyBufferRegion(zero_scratch_rb.p, 0, zero_scratch_as.p, 0, blas_info.ResultDataMaxSizeInBytes);
    for (int i = 0; i < 2; ++i)
        list->CopyBufferRegion(pair_rb[i].p, 0, pair_as[i].p, 0, blas_info.ResultDataMaxSizeInBytes);
    if (!submit()) return fail("the rehydrate build did not complete");
    auto header_units = [](ID3D12Resource* rb) {
        void* p = nullptr;
        std::uint32_t units = 0;
        if (SUCCEEDED(rb->Map(0, nullptr, &p))) {
            std::memcpy(&units, static_cast<char*>(p) + 4, 4);
            rb->Unmap(0, nullptr);
        }
        return units;
    };
    check(header_units(rehydrated_rb.p) > 0, "the rehydrated structure has non-zero totalSizeIn64BNodes header");
    check(header_units(zero_scratch_rb.p) > 0, "the structure rehydrated without the title's scratch was built");
    check(header_units(pair_rb[0].p) > 0 && header_units(pair_rb[1].p) > 0,
          "both elements of the batch without scratch were built");

    // ---- dump for xdxr_bvh.py check-build ----
    auto dump = [&](ID3D12Resource* rb, UINT64 size, const std::string& name, const std::string& json) {
        void* p = nullptr;
        if (FAILED(rb->Map(0, nullptr, &p))) return false;
        std::uint32_t units = 0;
        std::memcpy(&units, static_cast<char*>(p) + 4, 4);   // header totalSizeIn64BNodes
        const UINT64 used = (UINT64(units) + 1) * 64;        // the field does not count the 64-byte header
        const bool ok = units && used <= size && write_file(dir + "/" + name + ".bin", p, std::size_t(used)) &&
                        write_file(dir + "/" + name + ".json", json.data(), json.size());
        rb->Unmap(0, nullptr);
        return ok;
    };
    auto triangle_list = [&](UINT count) {
        std::string out = "[";
        for (UINT t = 0; t < count; ++t) {
            out += t ? ",\n  [" : "\n  [";
            for (int k = 0; k < 3; ++k)
                out += (k ? ", [" : "[") + num(soup[9 * t + 3 * k]) + ", " + num(soup[9 * t + 3 * k + 1]) + ", " +
                       num(soup[9 * t + 3 * k + 2]) + "]";
            out += ", 0, " + std::to_string(t) + "]";
        }
        return out + "]";
    };
    const std::string triangles = triangle_list(tris);
    check(dump(blas_rb.p, blas_info.ResultDataMaxSizeInBytes, "shim_blas",
               "{\"kind\": \"blas\", \"address\": " + std::to_string(blas->GetGPUVirtualAddress()) +
                   ", \"triangles\": " + triangles + "}\n"),
          "the BLAS built through the shim is written for check-build");
    std::string inst = "[";
    for (int i = 0; i < 2; ++i) {
        inst += i ? ", {\"transform\": [" : "{\"transform\": [";
        for (int k = 0; k < 12; ++k) inst += (k ? ", " : "") + num(xforms[i][k]);
        inst += "], \"mask\": 255, \"blas\": 0}";
    }
    inst += "]";
    check(dump(tlas_rb.p, tlas_info.ResultDataMaxSizeInBytes, "shim_tlas",
               "{\"kind\": \"tlas\", \"address\": " + std::to_string(tlas->GetGPUVirtualAddress()) +
                   ", \"bias\": 0, \"blas\": [\"shim_blas\"], \"instances\": " + inst + "}\n"),
          "the TLAS built through the shim is written for check-build");
    check(dump(compacted_rb.p, post[0], "shim_blas_compacted",
               "{\"kind\": \"blas\", \"address\": " + std::to_string(compacted->GetGPUVirtualAddress()) +
                   ", \"triangles\": " + triangles + "}\n"),
          "the BLAS compacted through the shim is written for check-build");
    check(dump(zero_scratch_rb.p, blas_info.ResultDataMaxSizeInBytes, "shim_blas_rehydrated_zero_scratch",
               "{\"kind\": \"blas\", \"address\": " + std::to_string(zero_scratch_as->GetGPUVirtualAddress()) +
                   ", \"triangles\": " + triangles + "}\n"),
          "the BLAS rehydrated with the shim's scratch is written for check-build");
    for (int i = 0; i < 2; ++i) {
        const std::string name = "shim_blas_rehydrated_batch" + std::to_string(i);
        check(dump(pair_rb[i].p, blas_info.ResultDataMaxSizeInBytes, name,
                   "{\"kind\": \"blas\", \"address\": " + std::to_string(pair_as[i]->GetGPUVirtualAddress()) +
                       ", \"triangles\": " + triangle_list(pair_tris[i]) + "}\n"),
              i ? "the batch's second element (its own scratch region) is written for check-build"
                : "the batch's first element (its own scratch region) is written for check-build");
    }

    // ---- the destructor releases the builder ----
    destruct(builder);
    destruct(builder);   // a second destruction of the same storage is ignored
    check(rt::shims::xbox_bvh_builder_live_count() == 0, "the destructor releases the builder's state");

    std::printf("xbox_bvh_builder_test: %d of %d checks failed\n", g_failures, g_checks);
    return g_failures ? 1 : 0;
}
