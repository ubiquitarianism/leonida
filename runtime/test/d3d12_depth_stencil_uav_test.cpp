// Xbox depth/stencil creation, shader read-modify-write and exhaustive readback across queue submissions.
// A synthetic compute shader looks up the T#s that the device writes. No game data.
#include "d3d12_copy_compare.h"
#include "d3d12_depth_stencil_uav_views.h"
#include "d3d12_placed_depth_test.h"
#include "d3d12_test_shaders.h"
#include "shader_fixture_path.h"
#include "test_temp_dir.h"
#include "vk_test_device.h"
#include "../shims/d3d12_xs_device.h"
#include "../src/shader_library.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>

#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #x); return 1; } } while (0)

namespace {
using d3d12test::Ref;

void enhanced_transition(ID3D12GraphicsCommandList7* list, ID3D12Resource* texture,
                         D3D12_BARRIER_SYNC before_sync, D3D12_BARRIER_ACCESS before_access,
                         D3D12_BARRIER_LAYOUT before_layout, D3D12_BARRIER_SYNC after_sync,
                         D3D12_BARRIER_ACCESS after_access, D3D12_BARRIER_LAYOUT after_layout) {
    D3D12_TEXTURE_BARRIER barrier{};
    barrier.SyncBefore = before_sync;
    barrier.AccessBefore = before_access;
    barrier.LayoutBefore = before_layout;
    barrier.SyncAfter = after_sync;
    barrier.AccessAfter = after_access;
    barrier.LayoutAfter = after_layout;
    barrier.Subresources = {D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, 0, 0, 0, 0, 0};
    barrier.pResource = texture;
    D3D12_BARRIER_GROUP group{};
    group.Type = D3D12_BARRIER_TYPE_TEXTURE;
    group.NumBarriers = 1;
    group.pTextureBarriers = &barrier;
    list->Barrier(1, &group);
}

enum class Handoff { direct, graphics_transitions, compute_transitions };

struct Batch {
    Ref<ID3D12CommandAllocator> allocator;
    Ref<ID3D12GraphicsCommandList> list;
    bool open(ID3D12Device* device, D3D12_COMMAND_LIST_TYPE type) {
        return SUCCEEDED(device->CreateCommandAllocator(type, IID_PPV_ARGS(&allocator.p))) &&
               SUCCEEDED(device->CreateCommandList(0, type, allocator.p, nullptr, IID_PPV_ARGS(&list.p)));
    }
    bool submit(ID3D12CommandQueue* queue, ID3D12Fence* done, UINT64 value) {
        if (FAILED(list->Close())) return false;
        ID3D12CommandList* lists[] = {list.p};
        queue->ExecuteCommandLists(1, lists);
        return rt::d3d12::last_execute_result(queue) == leonida::ReplayResult::success &&
               SUCCEEDED(queue->Signal(done, value));
    }
};

bool compute_transfers(const vktest::Device& vk, VkFormat format, bool stencil) {
    if (vk.compute_family == vk.family) return true;
#ifdef VK_KHR_maintenance10
    if (!vk.maintenance10) return false;
    VkFormatProperties3 flags{VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_3};
    VkFormatProperties2 properties{VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2, &flags};
    vk.instance_fn<PFN_vkGetPhysicalDeviceFormatProperties2>("vkGetPhysicalDeviceFormatProperties2")(
        vk.physical, format, &properties);
    const VkFormatFeatureFlags2 required = VK_FORMAT_FEATURE_2_DEPTH_COPY_ON_COMPUTE_QUEUE_BIT_KHR |
        (stencil ? VK_FORMAT_FEATURE_2_STENCIL_COPY_ON_COMPUTE_QUEUE_BIT_KHR : 0);
    return (flags.optimalTilingFeatures & required) == required;
#else
    return false;
#endif
}

struct GuestAllocation {
    void* address = nullptr;
    ~GuestAllocation() { if (address) VirtualFree(address, 0, MEM_RELEASE); }
};

enum class Texture { committed, committed3, placed };

// The boot stack's Xbox slot 127 forwards this call to the real device's CreateCommittedResource3.
// Test the actual thunk when present, or the same normalized input at its destination on a standalone build.
HRESULT create_committed3(ID3D12Device* device, ID3D12Device* xbox, const D3D12_HEAP_PROPERTIES& heap,
                          const D3D12_RESOURCE_DESC& desc, D3D12_BARRIER_LAYOUT initial,
                          const D3D12_CLEAR_VALUE& clear, ID3D12Resource** out,
                          UINT castable_count = 0, const DXGI_FORMAT* castable = nullptr) {
    D3D12_RESOURCE_DESC1 input{desc.Dimension, desc.Alignment, desc.Width, desc.Height, desc.DepthOrArraySize,
        desc.MipLevels, desc.Format, desc.SampleDesc, desc.Layout, desc.Flags, {}};
    if (xbox && rt::d3d12_xs::device_slot_overridden(127)) {
        using Slot127 = HRESULT(WINAPI*)(void*, const D3D12_HEAP_PROPERTIES*, D3D12_HEAP_FLAGS,
            const D3D12_RESOURCE_DESC1*, D3D12_BARRIER_LAYOUT, const D3D12_CLEAR_VALUE*,
            ID3D12ProtectedResourceSession*, UINT, const DXGI_FORMAT*, REFIID, void**);
        const auto create = reinterpret_cast<Slot127>(d3d12test::slot(xbox, 127));
        return create(xbox, &heap, D3D12_HEAP_FLAG_NONE, &input, initial, &clear, nullptr,
                      castable_count, castable, IID_PPV_ARGS(out));
    }
    Ref<ID3D12Device10> device10;
    const HRESULT hr = device->QueryInterface(IID_PPV_ARGS(&device10.p));
    if (FAILED(hr)) return hr;
    input.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    return device10->CreateCommittedResource3(&heap, D3D12_HEAP_FLAG_NONE, &input, initial, &clear, nullptr,
                                              castable_count, castable, IID_PPV_ARGS(out));
}

int run(vktest::Device& vk, UINT width, UINT height, bool enhanced, Texture kind, UINT16 layers,
        DXGI_FORMAT format, Handoff handoff) {
    const bool array = layers > 1;
    // Array views cover the last two layers so the shader reaches slice 5 in the six-layer case.
    const UINT first_slice = array ? layers - 2u : 0;
    const bool placed = kind == Texture::placed;
    const bool modern = kind == Texture::committed3;
    const bool depth16 = format == DXGI_FORMAT_R16_TYPELESS;
    const UINT planes = depth16 ? 1 : 2;
    const UINT subresources = planes * layers;
    const char* fixture = array ? LEONIDA_DEPTH_STENCIL_ARRAY_SPV : LEONIDA_DEPTH_STENCIL_2D_SPV;
    const char* bindings = array ? "2,11,12" : "2,8,9";
    if (depth16) {
        fixture = array ? LEONIDA_DEPTH16_ARRAY_SPV : LEONIDA_DEPTH16_2D_SPV;
        bindings = array ? "2,12" : "2,9";
    }
    const auto program = d3d12test::fssm_program({0x44535541, 0xbf810000});
    const auto shader = d3d12test::shader_container(program);
    const testfiles::ScratchDir library("leonida_depth_stencil_uav");
    CHECK(testfiles::copy_file(vktest::shader_fixture_path(fixture), library.path() / "write.spv"));
    {
        std::ofstream index(library.path() / "index.json", std::ios::binary);
        index << "{\"write\":{\"stage\":\"cs\",\"fssm_sha256\":\"" << leonida::sha256_hex(program)
              << "\",\"bindings\":[" << bindings << "],\"push_constant_bytes\":80,\"local_size\":[1,1,1],\"wave64\":false}}";
        CHECK(index.good());
    }

    rt::d3d12::VulkanDeviceInfo info{};
    info.get_instance_proc = vk.instance_proc;
    info.instance = vk.instance;
    info.physical = vk.physical;
    info.device = vk.device;
    info.queue = vk.queue;
    info.queue_family = vk.family;
    info.compute_family = vk.compute_family;
    info.compute_queues = {vk.compute_queue};
    info.maintenance10 = vk.maintenance10;
    info.bindless = vk.bindless;
    info.bindless_partially_bound = vk.partially_bound;
    info.buffer_device_address = vk.buffer_device_address;
    info.shader_dir = library.path().string();
    ID3D12Device* device = nullptr;
    CHECK(SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&device))));
    Ref<ID3D12Device> xbox;
    CHECK(SUCCEEDED(rt::d3d12_xs::create_device_proxy(device, IID_PPV_ARGS(&xbox.p)))); // takes ownership
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = width;
    desc.Height = height;
    desc.MipLevels = 1;
    desc.DepthOrArraySize = layers;
    desc.SampleDesc.Count = 1;
    desc.Format = format;
    desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL | D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    desc.Layout = static_cast<D3D12_TEXTURE_LAYOUT>(0x101);
    D3D12_HEAP_PROPERTIES props{};
    props.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_CLEAR_VALUE optimized{};
    optimized.Format = depth16 ? DXGI_FORMAT_D16_UNORM : DXGI_FORMAT_D32_FLOAT_S8X24_UINT;
    optimized.DepthStencil.Depth = 1.0f;
    auto normalized = desc;
    normalized.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    if (modern) {
        // The standalone PC Device10 retains its ordinary rejection of the combined creation flags.
        Ref<ID3D12Resource> refused;
        CHECK(create_committed3(device, nullptr, props, desc, D3D12_BARRIER_LAYOUT_COMMON,
                                optimized, &refused.p) == E_INVALIDARG && !refused.p);
    }
    rt::d3d12::set_device_outer(device, xbox.p);
    if (!enhanced && !array && !depth16) CHECK(placeddepthtest::check_alias_lifetime(device, xbox.p) && placeddepthtest::check_alias_lifetime(device, xbox.p, true));
    using CreateFn = HRESULT(WINAPI*)(void*, const D3D12_HEAP_PROPERTIES*, D3D12_HEAP_FLAGS,
                                      const D3D12_RESOURCE_DESC*, D3D12_RESOURCE_STATES, const D3D12_CLEAR_VALUE*,
                                      REFIID, void**);
    // The baseline uses PC sizing. In builds with the Xbox slot-25 dependency, size the placement through
    // the console's return-buffer-first ABI; keep native footprints for the host pixel readback.
    auto native_depth = normalized;
    native_depth.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
    auto size = device->GetResourceAllocationInfo(0, 1, &native_depth);
    CHECK(size.SizeInBytes > 0 && size.SizeInBytes != UINT64_MAX);
    if (modern) {
        const auto logical = device->GetResourceAllocationInfo(0, 1, &normalized);
        CHECK(logical.SizeInBytes == size.SizeInBytes && logical.Alignment == size.Alignment);
    }
    if (rt::d3d12_xs::device_slot_overridden(25)) {
        using SizeFn = D3D12_RESOURCE_ALLOCATION_INFO*(WINAPI*)(D3D12_RESOURCE_ALLOCATION_INFO*, void*,
                                                                UINT, UINT, const D3D12_RESOURCE_DESC*);
        D3D12_RESOURCE_ALLOCATION_INFO xbox_size{};
        const auto get_size = reinterpret_cast<SizeFn>(d3d12test::slot(xbox.p, 25));
        CHECK(get_size(&xbox_size, xbox.p, 0, 1, &desc) == &xbox_size);
        // The console's size of a depth-stencil target may be smaller than the PC device's (console_depth_stencil_size
        // in d3d12_xs_alloc_size.h), so the guest range below takes the larger of the two.
        CHECK(xbox_size.SizeInBytes && xbox_size.SizeInBytes != UINT64_MAX && xbox_size.Alignment);
        size.SizeInBytes = std::max(size.SizeInBytes, xbox_size.SizeInBytes);
        size.Alignment = std::max(size.Alignment, xbox_size.Alignment);
    }
    GuestAllocation guest;
    Ref<ID3D12Resource> texture, alias;
    const auto create_texture = reinterpret_cast<CreateFn>(d3d12test::slot(xbox.p, 76));
    if (placed) {
        using PlaceFn = HRESULT(WINAPI*)(void*, UINT64, const D3D12_RESOURCE_DESC*, D3D12_BARRIER_LAYOUT,
                                         const D3D12_CLEAR_VALUE*, REFIID, void**);
        const auto place = reinterpret_cast<PlaceFn>(d3d12test::slot(xbox.p, 133));
        guest.address = VirtualAlloc(nullptr, SIZE_T(size.SizeInBytes), MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
        CHECK(guest.address);
        const auto va = reinterpret_cast<UINT64>(guest.address);
        const auto layout = enhanced ? D3D12_BARRIER_LAYOUT_UNORDERED_ACCESS : D3D12_BARRIER_LAYOUT_DEPTH_STENCIL_WRITE;
        CHECK(SUCCEEDED(place(xbox.p, va, &desc, layout, nullptr,
                              IID_PPV_ARGS(&texture.p))));
        Ref<ID3D12Resource> rejected;
        CHECK(place(xbox.p, va, &desc, D3D12_BARRIER_LAYOUT_RENDER_TARGET, nullptr,
                    IID_PPV_ARGS(&rejected.p)) == E_INVALIDARG && !rejected.p);
        CHECK(SUCCEEDED(place(xbox.p, va, &desc, D3D12_BARRIER_LAYOUT_SHADER_RESOURCE, nullptr,
                              IID_PPV_ARGS(&alias.p))));
        CHECK(texture.p != alias.p && texture->GetGPUVirtualAddress() == va && alias->GetGPUVirtualAddress() == va);
    } else if (modern) {
        // Exercise the exact no-list overload as well as lists naming both the attachment and its writable planes.
        const DXGI_FORMAT d16_views[] = {DXGI_FORMAT_D16_UNORM, DXGI_FORMAT_R16_UNORM, DXGI_FORMAT_R16_UINT};
        const DXGI_FORMAT d32s8_views[] = {DXGI_FORMAT_D32_FLOAT_S8X24_UINT, DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS,
            DXGI_FORMAT_X32_TYPELESS_G8X24_UINT, DXGI_FORMAT_R32_FLOAT, DXGI_FORMAT_R8_UINT};
        const UINT count = layers == 3 ? UINT(depth16 ? std::size(d16_views) : std::size(d32s8_views)) : 0;
        const DXGI_FORMAT* castable = count ? (depth16 ? d16_views : d32s8_views) : nullptr;
        CHECK(SUCCEEDED(create_committed3(device, xbox.p, props, desc, D3D12_BARRIER_LAYOUT_COMMON,
                                          optimized, &texture.p, count, castable)) && texture.p);
        if (!array) {
            Ref<ID3D12Resource> refused;
            CHECK(create_committed3(device, xbox.p, props, desc, D3D12_BARRIER_LAYOUT_RENDER_TARGET,
                                    optimized, &refused.p) == E_INVALIDARG && !refused.p);
            auto wrong_clear = optimized;
            wrong_clear.Format = DXGI_FORMAT_D32_FLOAT;
            CHECK(create_committed3(device, xbox.p, props, desc, D3D12_BARRIER_LAYOUT_COMMON,
                                    wrong_clear, &refused.p) == E_INVALIDARG && !refused.p);
            const DXGI_FORMAT incompatible = DXGI_FORMAT_R32G32_FLOAT;
            CHECK(create_committed3(device, xbox.p, props, desc, D3D12_BARRIER_LAYOUT_COMMON,
                                    optimized, &refused.p, 1, &incompatible) == E_INVALIDARG && !refused.p);
            // The PC Device10 endpoint returns S_FALSE for a successful validation-only call.
            CHECK(create_committed3(device, nullptr, props, desc, D3D12_BARRIER_LAYOUT_COMMON,
                                    optimized, nullptr) == S_FALSE);
        }
    } else {
        const auto heap_flags = !enhanced && !array && !depth16
            ? D3D12_HEAP_FLAG_ALLOW_SHADER_ATOMICS : D3D12_HEAP_FLAG_NONE;
        CHECK(SUCCEEDED(create_texture(xbox.p, &props, heap_flags, &desc, D3D12_RESOURCE_STATE_DEPTH_WRITE,
                                       nullptr, IID_PPV_ARGS(&texture.p))));
        D3D12_HEAP_FLAGS reported{};
        CHECK(SUCCEEDED(texture->GetHeapProperties(nullptr, &reported)));
        CHECK((reported & D3D12_HEAP_FLAG_ALLOW_SHADER_ATOMICS) == heap_flags);
        if (heap_flags) {
            Ref<ID3D12Resource> depth_only;
            CHECK(device->CreateCommittedResource(&props, heap_flags, &native_depth, D3D12_RESOURCE_STATE_DEPTH_WRITE,
                    nullptr, IID_PPV_ARGS(&depth_only.p)) == E_INVALIDARG && !depth_only.p);
            for (auto type : {D3D12_HEAP_TYPE_UPLOAD, D3D12_HEAP_TYPE_READBACK}) {
                auto cpu_heap = props;
                cpu_heap.Type = type;
                Ref<ID3D12Resource> invalid;
                CHECK(create_texture(xbox.p, &cpu_heap, heap_flags, &desc, D3D12_RESOURCE_STATE_COMMON,
                                     nullptr, IID_PPV_ARGS(&invalid.p)) == E_INVALIDARG && !invalid.p);
            }
        }
    }
    CHECK(texture->GetDesc().Flags == desc.Flags);
    if (array) CHECK(check_depth_uav_array_views(device, texture.p, depth16));

    const D3D12_DESCRIPTOR_HEAP_DESC dsv_desc{D3D12_DESCRIPTOR_HEAP_TYPE_DSV, layers, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
    const D3D12_DESCRIPTOR_HEAP_DESC views_desc{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 4,
                                               D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
    Ref<ID3D12DescriptorHeap> dsv, views;
    CHECK(SUCCEEDED(device->CreateDescriptorHeap(&dsv_desc, IID_PPV_ARGS(&dsv.p))));
    CHECK(SUCCEEDED(device->CreateDescriptorHeap(&views_desc, IID_PPV_ARGS(&views.p))));
    D3D12_DEPTH_STENCIL_VIEW_DESC depth_view{};
    depth_view.Format = depth16 ? DXGI_FORMAT_D16_UNORM : DXGI_FORMAT_D32_FLOAT_S8X24_UINT;
    depth_view.ViewDimension = array ? D3D12_DSV_DIMENSION_TEXTURE2DARRAY : D3D12_DSV_DIMENSION_TEXTURE2D;
    const SIZE_T dsv_step = device->GetDescriptorHandleIncrementSize(dsv_desc.Type);
    const auto depth_handle = dsv->GetCPUDescriptorHandleForHeapStart();
    for (UINT layer = 0; layer < layers; ++layer) {
        if (array) depth_view.Texture2DArray = {0, layer, 1};
        device->CreateDepthStencilView(texture.p, &depth_view, {depth_handle.ptr + dsv_step * layer});
    }
    const SIZE_T base = views->GetCPUDescriptorHandleForHeapStart().ptr;
    const SIZE_T step = device->GetDescriptorHandleIncrementSize(views_desc.Type);
    CHECK(base && step >= 32);
    std::uint32_t keys[2][8]{};
    for (UINT plane = 0; plane < planes; ++plane) {
        D3D12_UNORDERED_ACCESS_VIEW_DESC view{};
        view.ViewDimension = array ? D3D12_UAV_DIMENSION_TEXTURE2DARRAY : D3D12_UAV_DIMENSION_TEXTURE2D;
        view.Format = depth16 ? DXGI_FORMAT_R16_UINT : plane == 0 ? DXGI_FORMAT_R32_FLOAT : DXGI_FORMAT_R8_UINT;
        if (array) view.Texture2DArray = {0, first_slice, 2, plane};
        else view.Texture2D.PlaneSlice = plane;
        const D3D12_CPU_DESCRIPTOR_HANDLE handle{base + plane * step};
        device->CreateUnorderedAccessView(texture.p, nullptr, &view, handle);
        std::memcpy(keys[plane], reinterpret_cast<const void*>(handle.ptr), sizeof(keys[plane]));
        CHECK(rt::d3d12::bindless_hash_lookup(device, keys[plane]) != 0);
    }
    if (planes == 2) CHECK(std::memcmp(keys[0], keys[1], sizeof(keys[0])) != 0);
    D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
    srv.Format = depth16 ? DXGI_FORMAT_R16_UNORM : DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS;
    srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srv.ViewDimension = array ? D3D12_SRV_DIMENSION_TEXTURE2DARRAY : D3D12_SRV_DIMENSION_TEXTURE2D;
    if (array) srv.Texture2DArray = {0, 1, first_slice, 2, 0, 0.0f};
    else srv.Texture2D.MipLevels = 1;
    const D3D12_CPU_DESCRIPTOR_HANDLE sampled{base + 2 * step};
    device->CreateShaderResourceView(texture.p, &srv, sampled);
    std::uint32_t sampled_key[8]{};
    std::memcpy(sampled_key, reinterpret_cast<const void*>(sampled.ptr), sizeof(sampled_key));
    CHECK(rt::d3d12::bindless_hash_lookup(device, sampled_key) != 0);
    CHECK(std::memcmp(sampled_key, keys[0], sizeof(sampled_key)) != 0);
    const D3D12_CPU_DESCRIPTOR_HANDLE invalid{base + 3 * step};
    std::uint32_t zero[8]{};
    for (UINT plane = 0; plane < 2; ++plane) {
        std::memset(reinterpret_cast<void*>(invalid.ptr), 0x55, sizeof(zero));
        D3D12_UNORDERED_ACCESS_VIEW_DESC view{};
        view.ViewDimension = array ? D3D12_UAV_DIMENSION_TEXTURE2DARRAY : D3D12_UAV_DIMENSION_TEXTURE2D;
        if (array) view.Texture2DArray = {0, first_slice, 2, plane};
        else view.Texture2D.PlaneSlice = plane;
        view.Format = plane == 0 ? DXGI_FORMAT_R8_UINT : DXGI_FORMAT_R32_FLOAT;
        device->CreateUnorderedAccessView(texture.p, nullptr, &view, invalid);
        CHECK(std::memcmp(reinterpret_cast<const void*>(invalid.ptr), zero, sizeof(zero)) == 0);
    }

    // ClearUAV takes matching handles to two copies of the descriptor and the original Xbox resource.
    const D3D12_DESCRIPTOR_HEAP_DESC visible_desc{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 2,
                                                  D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
    Ref<ID3D12DescriptorHeap> visible;
    CHECK(SUCCEEDED(device->CreateDescriptorHeap(&visible_desc, IID_PPV_ARGS(&visible.p))));
    device->CopyDescriptorsSimple(planes, visible->GetCPUDescriptorHandleForHeapStart(), {base}, visible_desc.Type);
    const auto gpu = visible->GetGPUDescriptorHandleForHeapStart();
    const FLOAT depth_clear[] = {0.75f, 0, 0, 0};
    const UINT stencil_clear[] = {0x31, 0, 0, 0};
    if (!array) {
        // Both parents have a real R32_FLOAT storage plane, but this T# belongs to the first parent only.
        auto other_desc = desc;
        other_desc.Width = other_desc.Height = 2;
        Ref<ID3D12Resource> other;
        CHECK(SUCCEEDED(create_texture(xbox.p, &props, D3D12_HEAP_FLAG_NONE, &other_desc,
                                       D3D12_RESOURCE_STATE_DEPTH_WRITE, nullptr, IID_PPV_ARGS(&other.p))));
        copycmp::Run rejected;
        CHECK(rejected.init(device, true));
        rejected.begin();
        rejected.barrier(other.p, D3D12_RESOURCE_STATE_DEPTH_WRITE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        ID3D12DescriptorHeap* heaps[] = {visible.p};
        rejected.list->SetDescriptorHeaps(1, heaps);
        rejected.list->ClearUnorderedAccessViewFloat(gpu, {base}, other.p, depth_clear, 0, nullptr);
        CHECK(rejected.list->Close() == E_INVALIDARG);
    }

    D3D12_ROOT_PARAMETER param{};
    param.ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    param.Constants.Num32BitValues = 16;
    D3D12_VERSIONED_ROOT_SIGNATURE_DESC rs{};
    rs.Version = D3D_ROOT_SIGNATURE_VERSION_1_0;
    rs.Desc_1_0 = {1, &param, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
    const auto root_blob = d3d12test::serialize_root_signature(rs);
    Ref<ID3D12RootSignature> root;
    CHECK(!root_blob.empty());
    CHECK(SUCCEEDED(device->CreateRootSignature(0, root_blob.data(), root_blob.size(), IID_PPV_ARGS(&root.p))));
    D3D12_COMPUTE_PIPELINE_STATE_DESC pipeline_desc{};
    pipeline_desc.pRootSignature = root.p;
    pipeline_desc.CS = {shader.data(), shader.size()};
    Ref<ID3D12PipelineState> pipeline;
    CHECK(SUCCEEDED(device->CreateComputePipelineState(&pipeline_desc, IID_PPV_ARGS(&pipeline.p))));

    // Three submissions. Only the final fence is waited on by the CPU.
    copycmp::Run graphics;
    CHECK(graphics.init(device, true));
    const auto worker_type = handoff == Handoff::direct ? D3D12_COMMAND_LIST_TYPE_DIRECT : D3D12_COMMAND_LIST_TYPE_COMPUTE;
    D3D12_COMMAND_QUEUE_DESC worker_desc{};
    worker_desc.Type = worker_type;
    Ref<ID3D12CommandQueue> worker;
    CHECK(SUCCEEDED(device->CreateCommandQueue(&worker_desc, IID_PPV_ARGS(&worker.p))));
    Batch shader_batch, read_batch;
    CHECK(shader_batch.open(device, worker_type) && read_batch.open(device, D3D12_COMMAND_LIST_TYPE_DIRECT));
    std::vector<D3D12_PLACED_SUBRESOURCE_FOOTPRINT> layouts(subresources);
    std::vector<UINT64> row_bytes(subresources);
    UINT64 total = 0;
    device->GetCopyableFootprints(&native_depth, 0, subresources, 0, layouts.data(), nullptr, row_bytes.data(), &total);
    CHECK(row_bytes[0] == width * (depth16 ? 2 : 4));
    if (planes == 2) CHECK(row_bytes[layers] == width);
    copycmp::Ptr<ID3D12Resource> readback;
    CHECK(graphics.buffer(D3D12_HEAP_TYPE_READBACK, total, D3D12_RESOURCE_STATE_COPY_DEST, readback));

    const float depth_seed[] = {0.5f, 0.25f, 1.0f};
    const UINT8 stencil_seed[] = {0x5a, 0x37, 0xcb};
    const UINT16 d16_seed[] = {0, 0xffff, 257, 514, 771, 1028};
    const auto flags = depth16 ? D3D12_CLEAR_FLAG_DEPTH : D3D12_CLEAR_FLAG_DEPTH | D3D12_CLEAR_FLAG_STENCIL;
    if (modern) {
        Ref<ID3D12GraphicsCommandList7> list;
        CHECK(SUCCEEDED(graphics.list->QueryInterface(IID_PPV_ARGS(&list.p))));
        enhanced_transition(list.p, texture.p, D3D12_BARRIER_SYNC_ALL,
            D3D12_BARRIER_ACCESS_COMMON, D3D12_BARRIER_LAYOUT_COMMON, D3D12_BARRIER_SYNC_DEPTH_STENCIL,
            D3D12_BARRIER_ACCESS_DEPTH_STENCIL_WRITE, D3D12_BARRIER_LAYOUT_DEPTH_STENCIL_WRITE);
    }
    for (UINT layer = 0; layer < layers; ++layer)
        graphics.list->ClearDepthStencilView({depth_handle.ptr + dsv_step * layer}, flags,
            depth16 ? float(d16_seed[layer]) / 65535.0f : depth_seed[layer], depth16 ? 0 : stencil_seed[layer],
            0, nullptr);

    const bool compute_copy = handoff == Handoff::compute_transitions;
    // Graphics-side transfers use ordinary cross-queue UAV layout. With maintenance10 the compute family can
    // also perform the transfers, with COMMON as the shared handoff layout at both fences.
    if (enhanced) {
        Ref<ID3D12GraphicsCommandList7> list;
        CHECK(SUCCEEDED(graphics.list->QueryInterface(IID_PPV_ARGS(&list.p))));
        enhanced_transition(list.p, texture.p, D3D12_BARRIER_SYNC_DEPTH_STENCIL,
            D3D12_BARRIER_ACCESS_DEPTH_STENCIL_WRITE, D3D12_BARRIER_LAYOUT_DEPTH_STENCIL_WRITE,
            D3D12_BARRIER_SYNC_ALL, compute_copy ? D3D12_BARRIER_ACCESS_COMMON : D3D12_BARRIER_ACCESS_UNORDERED_ACCESS,
            compute_copy ? D3D12_BARRIER_LAYOUT_COMMON : D3D12_BARRIER_LAYOUT_UNORDERED_ACCESS);
    } else {
        graphics.barrier(texture.p, D3D12_RESOURCE_STATE_DEPTH_WRITE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    }
    CHECK(SUCCEEDED(graphics.list->Close()));
    ID3D12CommandList* write_lists[] = {graphics.list.p};
    graphics.queue->ExecuteCommandLists(1, write_lists);
    CHECK(rt::d3d12::last_execute_result(graphics.queue.p) == leonida::ReplayResult::success);
    CHECK(SUCCEEDED(graphics.queue->Signal(graphics.fence.p, 1)));
    CHECK(SUCCEEDED(worker->Wait(graphics.fence.p, 1)));
    Ref<ID3D12GraphicsCommandList7> compute_list;
    if (compute_copy) {
        CHECK(SUCCEEDED(shader_batch.list->QueryInterface(IID_PPV_ARGS(&compute_list.p))));
        enhanced_transition(compute_list.p, texture.p, D3D12_BARRIER_SYNC_ALL, D3D12_BARRIER_ACCESS_COMMON,
            D3D12_BARRIER_LAYOUT_COMMON, D3D12_BARRIER_SYNC_COMPUTE_SHADING, D3D12_BARRIER_ACCESS_UNORDERED_ACCESS,
            D3D12_BARRIER_LAYOUT_UNORDERED_ACCESS);
    }
    const D3D12_RECT depth_rect{1, 3, 2, 4}, stencil_rect{3, 2, 4, 3};
    ID3D12DescriptorHeap* clear_heaps[] = {visible.p};
    auto* list = shader_batch.list.p;
    list->SetDescriptorHeaps(1, clear_heaps);
    const UINT d16_clear[] = {0x4512, 0, 0, 0};
    if (depth16) list->ClearUnorderedAccessViewUint(gpu, {base}, texture.p, d16_clear, 1, &depth_rect);
    else {
        list->ClearUnorderedAccessViewFloat(gpu, {base}, texture.p, depth_clear, 1, &depth_rect);
        list->ClearUnorderedAccessViewUint({gpu.ptr + step}, {base + step}, texture.p, stencil_clear, 1, &stencil_rect);
    }
    D3D12_RESOURCE_BARRIER order{};
    order.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    order.UAV.pResource = texture.p;
    list->ResourceBarrier(1, &order);
    list->SetComputeRootSignature(root.p);
    list->SetPipelineState(pipeline.p);
    list->SetComputeRoot32BitConstants(0, 16, keys, 0);
    list->Dispatch(1, 1, 1);
    if (compute_copy)
        enhanced_transition(compute_list.p, texture.p, D3D12_BARRIER_SYNC_COMPUTE_SHADING,
            D3D12_BARRIER_ACCESS_UNORDERED_ACCESS, D3D12_BARRIER_LAYOUT_UNORDERED_ACCESS,
            D3D12_BARRIER_SYNC_ALL, D3D12_BARRIER_ACCESS_COMMON, D3D12_BARRIER_LAYOUT_COMMON);
    CHECK(shader_batch.submit(worker.p, graphics.fence.p, 2));
    CHECK(SUCCEEDED(graphics.queue->Wait(graphics.fence.p, 2)));

    // The second placement reads the first one's writes at the same guest address.
    auto* reader = placed ? alias.p : texture.p;
    if (placed) {
        D3D12_RESOURCE_BARRIER activate{};
        activate.Type = D3D12_RESOURCE_BARRIER_TYPE_ALIASING;
        activate.Aliasing = {texture.p, reader};
        read_batch.list->ResourceBarrier(1, &activate);
    }
    if (enhanced) {
        Ref<ID3D12GraphicsCommandList7> read_list;
        CHECK(SUCCEEDED(read_batch.list->QueryInterface(IID_PPV_ARGS(&read_list.p))));
        enhanced_transition(read_list.p, reader, D3D12_BARRIER_SYNC_ALL,
            compute_copy ? D3D12_BARRIER_ACCESS_COMMON : D3D12_BARRIER_ACCESS_UNORDERED_ACCESS,
            compute_copy ? D3D12_BARRIER_LAYOUT_COMMON : D3D12_BARRIER_LAYOUT_UNORDERED_ACCESS,
            D3D12_BARRIER_SYNC_COPY, D3D12_BARRIER_ACCESS_COPY_SOURCE, D3D12_BARRIER_LAYOUT_COPY_SOURCE);
    } else {
        D3D12_RESOURCE_BARRIER copy{};
        copy.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        copy.Transition = {reader, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                           D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE};
        read_batch.list->ResourceBarrier(1, &copy);
    }
    for (UINT sub = 0; sub < subresources; ++sub) {
        const auto src = copycmp::sub_loc(reader, sub);
        const auto dst = copycmp::foot_loc(readback.p, layouts[sub]);
        read_batch.list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    }
    CHECK(read_batch.submit(graphics.queue.p, graphics.fence.p, 3));
    CHECK(SUCCEEDED(graphics.fence->SetEventOnCompletion(3, nullptr)));
    CHECK(graphics.fence->GetCompletedValue() >= 3 && SUCCEEDED(device->GetDeviceRemovedReason()));
    unsigned char* mapped = nullptr;
    CHECK(SUCCEEDED(readback->Map(0, nullptr, reinterpret_cast<void**>(&mapped))) && mapped);
    for (UINT sub = 0; sub < subresources; ++sub) {
        const UINT layer = sub % layers;
        const UINT plane = sub / layers;
        const UINT bytes = plane ? 1 : depth16 ? 2 : 4;
        for (UINT y = 0; y < height; ++y) {
            for (UINT x = 0; x < width; ++x) {
                const bool in_view = layer >= first_slice;
                const bool cleared = in_view && (plane ? x == 3 && y == 2 : x == 1 && y == 3);
                UINT32 expected = 0;
                if (depth16) {
                    expected = cleared ? d16_clear[0] : d16_seed[layer];
                    if (layer == layers - 1u && x == 2 && y == 1) expected ^= 0x1357u;
                } else if (plane == 0) {
                    float depth = cleared ? depth_clear[0] : depth_seed[layer];
                    if (layer == first_slice && x == 2 && y == 1) depth *= 0.25f;
                    std::memcpy(&expected, &depth, sizeof(depth));
                } else {
                    expected = cleared ? stencil_clear[0] : stencil_seed[layer];
                    if (layer == layers - 1u && x == 5 && y == 3) expected ^= 0xffu;
                }
                UINT32 actual = 0;
                std::memcpy(&actual, mapped + layouts[sub].Offset + y * layouts[sub].Footprint.RowPitch + x * bytes, bytes);
                if (actual != expected) {
                    std::fprintf(stderr, "subresource %u pixel (%u,%u): got %#x, expected %#x\n", sub, x, y, actual, expected);
                    readback->Unmap(0, nullptr);
                    return 1;
                }
            }
        }
    }
    readback->Unmap(0, nullptr);
    const char* path = placed ? "slot 133" : "slot 76";
    if (modern) path = rt::d3d12_xs::device_slot_overridden(127) ? "slot 127" : "CreateCommittedResource3";
    std::printf("%s, %ux%u, %u layers, %u planes, %s barriers, graphics family %u, shader family %u, "
                "plane transfers on %s: every pixel matched\n", path, width, height, UINT(layers), planes,
                enhanced ? "enhanced" : "legacy", vk.family, handoff == Handoff::direct ? vk.family : vk.compute_family,
                compute_copy ? "compute" : "graphics");
    return 0;
}
} // namespace

int main() {
    vktest::Device vk;
    vk.request_buffer_device_address = true;
    vk.all_queue_types = true;
    const auto status = vk.initialize();
    if (status == vktest::DeviceStatus::unavailable) return 77;
    if (status != vktest::DeviceStatus::ready) return 1;
    if (!vk.bindless || !vk.storage_image_extended_formats) {
        std::fprintf(stderr, "SKIP: shader requires bindless images and typed r8ui storage\n");
        return 77;
    }
    const auto d32s8 = DXGI_FORMAT_R32G8X24_TYPELESS, d16 = DXGI_FORMAT_R16_TYPELESS;
    if (run(vk, 9, 7, false, Texture::committed, 1, d32s8, Handoff::direct) ||
        run(vk, 2560, 1440, true, Texture::committed, 1, d32s8, Handoff::direct) ||
        run(vk, 9, 7, true, Texture::placed, 1, d16, Handoff::graphics_transitions)) return 1;
    for (const auto format : {d32s8, d16}) {
        if (run(vk, 9, 7, false, Texture::committed, 3, format, Handoff::graphics_transitions) ||
            run(vk, 13, 9, true, Texture::placed, 3, format, Handoff::graphics_transitions)) return 1;
        const auto vk_format = format == d16 ? VK_FORMAT_D16_UNORM : VK_FORMAT_D32_SFLOAT_S8_UINT;
        if (compute_transfers(vk, vk_format, format != d16)) {
            if (run(vk, 9, 7, true, Texture::placed, 3, format, Handoff::compute_transitions)) return 1;
        } else {
            std::printf("compute-side depth transfers unavailable for VkFormat %u; graphics-side handoff tested\n",
                        unsigned(vk_format));
        }
    }
    // Array lengths reported by the Xbox boot trace. The texels are synthetic.
    constexpr UINT16 reported_layers[] = {1, 3, 6};
    for (const UINT16 layers : reported_layers)
        if (run(vk, 9, 7, true, Texture::committed3, layers, d16, Handoff::graphics_transitions)) return 1;
    if (run(vk, 9, 7, true, Texture::committed3, 3, d32s8, Handoff::graphics_transitions)) return 1;
    return vk.validation_errors() ? 1 : 0;
}
