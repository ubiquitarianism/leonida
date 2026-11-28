// D3D12_DSV_FLAG_READ_ONLY_DEPTH and D3D12_DSV_FLAG_READ_ONLY_STENCIL (RT-STATE-AUDIT-FORMATS):
//   1. Unknown DSV flags are rejected by CreateDepthStencilView (ClearDepthStencilView fails Close).
//   2. D24_UNORM_S8_UINT with READ_ONLY_DEPTH (depth plane in DEPTH_READ | PIXEL_SHADER_RESOURCE, stencil plane in
//      DEPTH_WRITE): depth tests run against cleared 0.75f and stencil increments on pass, while depth writes from a
//      PSO with DepthWriteMask = ALL are suppressed so depth stays 0.75f.
//   3. D24_UNORM_S8_UINT with READ_ONLY_STENCIL (depth in DEPTH_WRITE, stencil in DEPTH_READ): depth updates to 0.25f
//      while stencil writes from StencilPassOp = INCR_SAT are suppressed so stencil stays 0x12.
//   4. READ_ONLY_DEPTH | READ_ONLY_STENCIL suppresses both depth/stencil draws and ClearDepthStencilView on read-only
//      planes; ClearDepthStencilView on READ_ONLY_DEPTH-only clears stencil while keeping depth intact.
//   5. Depth-only D32_FLOAT with READ_ONLY_DEPTH suppresses depth writes while keeping depth in DEPTH_READ | PSR.
//
// Reuses test/glsl/vs_depth.vert and test/glsl/ps_depth_empty.frag. No game data; skips with 77 if no Vulkan GPU.
// Set LEONIDA_VK_VALIDATION=1 to run under VK_LAYER_KHRONOS_validation.
#include "vk_test_device.h"
#include "d3d12_test_shaders.h"
#include "../shims/d3d12_com.h"
#include "../src/depth24.h"
#include "../src/shader_library.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#if !defined(LEONIDA_TEST_VS_DEPTH_SPV) || !defined(LEONIDA_TEST_PS_DEPTH_EMPTY_SPV)
#error "LEONIDA_TEST_VS_DEPTH_SPV and LEONIDA_TEST_PS_DEPTH_EMPTY_SPV must be defined"
#endif

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

namespace {

using namespace d3d12test;

constexpr UINT kSide = 32;

std::uint32_t f2u(float f) {
    std::uint32_t u = 0;
    std::memcpy(&u, &f, sizeof(u));
    return u;
}

D3D12_RESOURCE_BARRIER subresource_transition(ID3D12Resource* resource, UINT subresource,
                                              D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after) {
    D3D12_RESOURCE_BARRIER b{};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition = {resource, subresource, before, after};
    return b;
}

D3D12_RESOURCE_DESC depth_desc(DXGI_FORMAT format) {
    D3D12_RESOURCE_DESC d{};
    d.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    d.Width = kSide;
    d.Height = kSide;
    d.DepthOrArraySize = 1;
    d.MipLevels = 1;
    d.Format = format;
    d.SampleDesc = {1, 0};
    d.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
    return d;
}

D3D12_RESOURCE_DESC buf_desc(UINT64 bytes) {
    D3D12_RESOURCE_DESC d{};
    d.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    d.Width = bytes;
    d.Height = 1;
    d.DepthOrArraySize = 1;
    d.MipLevels = 1;
    d.SampleDesc = {1, 0};
    d.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    return d;
}

bool make_pso(ID3D12Device* device, ID3D12RootSignature* root, const std::vector<std::uint8_t>& vs,
              const std::vector<std::uint8_t>& ps, DXGI_FORMAT dsv_format, D3D12_COMPARISON_FUNC depth_func,
              bool stencil_enable, D3D12_COMPARISON_FUNC stencil_func, ID3D12PipelineState** out) {
    D3D12_GRAPHICS_PIPELINE_STATE_DESC pd{};
    pd.pRootSignature = root;
    pd.VS = {vs.data(), vs.size()};
    pd.PS = {ps.data(), ps.size()};
    pd.SampleMask = UINT_MAX;
    pd.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    pd.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
    pd.RasterizerState.DepthClipEnable = TRUE;
    pd.DepthStencilState.DepthEnable = TRUE;
    pd.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ALL;
    pd.DepthStencilState.DepthFunc = depth_func;
    pd.DepthStencilState.StencilEnable = stencil_enable ? TRUE : FALSE;
    pd.DepthStencilState.StencilReadMask = 0xff;
    pd.DepthStencilState.StencilWriteMask = 0xff;
    const D3D12_DEPTH_STENCILOP_DESC op{D3D12_STENCIL_OP_KEEP, D3D12_STENCIL_OP_KEEP,
                                        D3D12_STENCIL_OP_INCR_SAT, stencil_func};
    pd.DepthStencilState.FrontFace = pd.DepthStencilState.BackFace = op;
    pd.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    pd.DSVFormat = dsv_format;
    pd.SampleDesc = {1, 0};
    return SUCCEEDED(device->CreateGraphicsPipelineState(&pd, IID_PPV_ARGS(out)));
}

void draw_at_z(ID3D12GraphicsCommandList* list, ID3D12RootSignature* root, ID3D12PipelineState* pso,
               D3D12_CPU_DESCRIPTOR_HANDLE dsv, float z, UINT stencil_ref = 0) {
    const D3D12_VIEWPORT vp{0.0f, 0.0f, float(kSide), float(kSide), 0.0f, 1.0f};
    const D3D12_RECT sc{0, 0, LONG(kSide), LONG(kSide)};
    list->SetGraphicsRootSignature(root);
    list->SetPipelineState(pso);
    const std::uint32_t constants[3] = {f2u(z), f2u(0.0f), f2u(0.0f)};
    list->SetGraphicsRoot32BitConstants(0, 3, constants, 0);
    list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    list->RSSetViewports(1, &vp);
    list->RSSetScissorRects(1, &sc);
    list->OMSetRenderTargets(0, nullptr, FALSE, &dsv);
    list->OMSetStencilRef(stencil_ref);
    list->DrawInstanced(3, 1, 0, 0);
}

bool read_d24s8(ID3D12CommandQueue* queue, ID3D12CommandAllocator* alloc,
                ID3D12GraphicsCommandList* list, ID3D12Fence* fence, UINT64& fence_value,
                ID3D12Resource* depth, D3D12_RESOURCE_STATES depth_state, D3D12_RESOURCE_STATES stencil_state,
                ID3D12Resource* readback, const D3D12_PLACED_SUBRESOURCE_FOOTPRINT foot[2],
                std::uint32_t& out_depth24, std::uint8_t& out_stencil) {
    if (FAILED(alloc->Reset()) || FAILED(list->Reset(alloc, nullptr))) return false;
    D3D12_RESOURCE_BARRIER to_copy[2] = {
        subresource_transition(depth, 0, depth_state, D3D12_RESOURCE_STATE_COPY_SOURCE),
        subresource_transition(depth, 1, stencil_state, D3D12_RESOURCE_STATE_COPY_SOURCE),
    };
    list->ResourceBarrier(2, to_copy);
    for (UINT plane = 0; plane < 2; ++plane) {
        D3D12_TEXTURE_COPY_LOCATION dst{readback, D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
        dst.PlacedFootprint = foot[plane];
        D3D12_TEXTURE_COPY_LOCATION src{depth, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
        src.SubresourceIndex = plane;
        list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    }
    D3D12_RESOURCE_BARRIER back[2] = {
        subresource_transition(depth, 0, D3D12_RESOURCE_STATE_COPY_SOURCE, depth_state),
        subresource_transition(depth, 1, D3D12_RESOURCE_STATE_COPY_SOURCE, stencil_state),
    };
    list->ResourceBarrier(2, back);
    if (FAILED(list->Close())) return false;
    ID3D12CommandList* lists[] = {list};
    queue->ExecuteCommandLists(1, lists);
    if (rt::d3d12::last_execute_result(queue) != leonida::ReplayResult::success) return false;
    if (FAILED(queue->Signal(fence, ++fence_value)) || FAILED(fence->SetEventOnCompletion(fence_value, nullptr)))
        return false;
    std::uint8_t* mapped = nullptr;
    if (FAILED(readback->Map(0, nullptr, reinterpret_cast<void**>(&mapped))) || !mapped) return false;
    const UINT cx = kSide / 2, cy = kSide / 2;
    std::uint32_t word = 0;
    std::memcpy(&word, mapped + foot[0].Offset + cy * foot[0].Footprint.RowPitch + cx * 4, 4);
    out_depth24 = word & leonida::depth24_max;
    out_stencil = mapped[foot[1].Offset + cy * foot[1].Footprint.RowPitch + cx];
    readback->Unmap(0, nullptr);
    return true;
}

bool depth24_near(std::uint32_t got, float expected) {
    const std::uint32_t want = leonida::depth24_quantize(expected);
    const std::uint32_t diff = got > want ? got - want : want - got;
    return diff <= 1;
}

}  // namespace

int main() {
    vktest::Device vk;
    vk.request_buffer_device_address = true;
    const auto status = vk.initialize();
    if (status == vktest::DeviceStatus::unavailable) return 77;
    if (status != vktest::DeviceStatus::ready) return 1;

    const auto vs_program = fssm_program({0x56532464, 0xbf810000});
    const auto ps_program = fssm_program({0x50532464, 0xbf810000});
    const auto library = std::filesystem::temp_directory_path() /
                         ("leonida_dsv_readonly_test_" + std::to_string(GetCurrentProcessId()));
    std::filesystem::create_directories(library);
    const auto copy = std::filesystem::copy_options::overwrite_existing;
    std::filesystem::copy_file(LEONIDA_TEST_VS_DEPTH_SPV, library / "vs_depth.spv", copy);
    std::filesystem::copy_file(LEONIDA_TEST_PS_DEPTH_EMPTY_SPV, library / "ps_depth_empty.spv", copy);
    {
        std::ofstream index(library / "index.json", std::ios::binary);
        index << "{\"vs_depth\": {\"stage\": \"vs\", \"fssm_sha256\": \"" << leonida::sha256_hex(vs_program)
              << "\", \"bindings\": [], \"push_constant_bytes\": 144, \"input_signature\": []},\n"
                 " \"ps_depth_empty\": {\"stage\": \"ps\", \"fssm_sha256\": \"" << leonida::sha256_hex(ps_program)
              << "\", \"bindings\": [], \"push_constant_bytes\": 144, \"outputs\": []}}";
    }

    int failures = 0;
    {
        rt::d3d12::VulkanDeviceInfo info{};
        info.get_instance_proc = vk.instance_proc;
        info.instance = vk.instance;
        info.physical = vk.physical;
        info.device = vk.device;
        info.queue = vk.queue;
        info.queue_family = vk.family;
        info.bindless = vk.bindless;
        info.bindless_partially_bound = vk.partially_bound;
        info.buffer_device_address = vk.buffer_device_address;
        info.occlusion_query_precise = vk.occlusion_query_precise;
        info.subgroup = vk.subgroup;
        info.shader_dir = library.string();
        info.features = vk.enabled;
        Ref<ID3D12Device> device;
        EXPECT(SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&device.p))));

        D3D12_ROOT_PARAMETER param{};
        param.ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        param.Constants = {0, 0, 3};
        D3D12_VERSIONED_ROOT_SIGNATURE_DESC root_desc{};
        root_desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_0;
        root_desc.Desc_1_0 = {1, &param, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
        const auto root_blob = serialize_root_signature(root_desc);
        EXPECT(!root_blob.empty());
        Ref<ID3D12RootSignature> root;
        EXPECT(SUCCEEDED(device->CreateRootSignature(0, root_blob.data(), root_blob.size(), IID_PPV_ARGS(&root.p))));

        const auto vs_blob = shader_container(vs_program);
        const auto ps_blob = shader_container(ps_program);
        Ref<ID3D12PipelineState> pso_less_always, pso_less_eq, pso_always_always, pso_d32;
        EXPECT(make_pso(device.p, root.p, vs_blob, ps_blob, DXGI_FORMAT_D24_UNORM_S8_UINT,
                        D3D12_COMPARISON_FUNC_LESS, true, D3D12_COMPARISON_FUNC_ALWAYS, &pso_less_always.p));
        EXPECT(make_pso(device.p, root.p, vs_blob, ps_blob, DXGI_FORMAT_D24_UNORM_S8_UINT,
                        D3D12_COMPARISON_FUNC_LESS, true, D3D12_COMPARISON_FUNC_EQUAL, &pso_less_eq.p));
        EXPECT(make_pso(device.p, root.p, vs_blob, ps_blob, DXGI_FORMAT_D24_UNORM_S8_UINT,
                        D3D12_COMPARISON_FUNC_ALWAYS, true, D3D12_COMPARISON_FUNC_ALWAYS, &pso_always_always.p));
        EXPECT(make_pso(device.p, root.p, vs_blob, ps_blob, DXGI_FORMAT_D32_FLOAT,
                        D3D12_COMPARISON_FUNC_LESS, false, D3D12_COMPARISON_FUNC_ALWAYS, &pso_d32.p));

        const D3D12_HEAP_PROPERTIES def_heap{D3D12_HEAP_TYPE_DEFAULT};
        const D3D12_HEAP_PROPERTIES rb_heap{D3D12_HEAP_TYPE_READBACK};
        const auto d24_desc = depth_desc(DXGI_FORMAT_D24_UNORM_S8_UINT);
        const auto d32_desc = depth_desc(DXGI_FORMAT_D32_FLOAT);
        Ref<ID3D12Resource> d24, d32, rb24, rb32;
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&def_heap, D3D12_HEAP_FLAG_NONE, &d24_desc,
                                                         D3D12_RESOURCE_STATE_DEPTH_WRITE, nullptr,
                                                         IID_PPV_ARGS(&d24.p))));
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&def_heap, D3D12_HEAP_FLAG_NONE, &d32_desc,
                                                         D3D12_RESOURCE_STATE_DEPTH_WRITE, nullptr,
                                                         IID_PPV_ARGS(&d32.p))));
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT foot24[2]{}, foot32{};
        UINT64 total24 = 0, total32 = 0;
        device->GetCopyableFootprints(&d24_desc, 0, 2, 0, foot24, nullptr, nullptr, &total24);
        device->GetCopyableFootprints(&d32_desc, 0, 1, 0, &foot32, nullptr, nullptr, &total32);
        const auto rb24_desc = buf_desc(total24);
        const auto rb32_desc = buf_desc(total32);
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&rb_heap, D3D12_HEAP_FLAG_NONE, &rb24_desc,
                                                         D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                         IID_PPV_ARGS(&rb24.p))));
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&rb_heap, D3D12_HEAP_FLAG_NONE, &rb32_desc,
                                                         D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                         IID_PPV_ARGS(&rb32.p))));

        Ref<ID3D12DescriptorHeap> dsv_heap;
        const D3D12_DESCRIPTOR_HEAP_DESC hd{D3D12_DESCRIPTOR_HEAP_TYPE_DSV, 7, D3D12_DESCRIPTOR_HEAP_FLAG_NONE};
        EXPECT(SUCCEEDED(device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&dsv_heap.p))));
        const UINT step = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_DSV);
        const auto slot = [&](UINT i) {
            auto h = dsv_heap->GetCPUDescriptorHandleForHeapStart();
            h.ptr += SIZE_T(i) * step;
            return h;
        };
        const auto d24_rw = slot(0), d24_ro_d = slot(1), d24_ro_s = slot(2), d24_ro_both = slot(3),
                   d24_bad = slot(4), d32_rw = slot(5), d32_ro_d = slot(6);

        D3D12_DEPTH_STENCIL_VIEW_DESC vd{};
        vd.Format = DXGI_FORMAT_D24_UNORM_S8_UINT;
        vd.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
        vd.Flags = D3D12_DSV_FLAG_NONE;
        device->CreateDepthStencilView(d24.p, &vd, d24_rw);
        vd.Flags = D3D12_DSV_FLAG_READ_ONLY_DEPTH;
        device->CreateDepthStencilView(d24.p, &vd, d24_ro_d);
        vd.Flags = D3D12_DSV_FLAG_READ_ONLY_STENCIL;
        device->CreateDepthStencilView(d24.p, &vd, d24_ro_s);
        vd.Flags = D3D12_DSV_FLAG_READ_ONLY_DEPTH | D3D12_DSV_FLAG_READ_ONLY_STENCIL;
        device->CreateDepthStencilView(d24.p, &vd, d24_ro_both);
        vd.Flags = static_cast<D3D12_DSV_FLAGS>(0x4);
        device->CreateDepthStencilView(d24.p, &vd, d24_bad);

        vd.Format = DXGI_FORMAT_D32_FLOAT;
        vd.Flags = D3D12_DSV_FLAG_NONE;
        device->CreateDepthStencilView(d32.p, &vd, d32_rw);
        vd.Flags = D3D12_DSV_FLAG_READ_ONLY_DEPTH;
        device->CreateDepthStencilView(d32.p, &vd, d32_ro_d);

        Ref<ID3D12CommandQueue> queue;
        const D3D12_COMMAND_QUEUE_DESC qd{D3D12_COMMAND_LIST_TYPE_DIRECT};
        EXPECT(SUCCEEDED(device->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue.p))));
        Ref<ID3D12CommandAllocator> alloc;
        EXPECT(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&alloc.p))));
        Ref<ID3D12GraphicsCommandList> list;
        EXPECT(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc.p, nullptr,
                                                   IID_PPV_ARGS(&list.p))));
        Ref<ID3D12Fence> fence;
        EXPECT(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))));
        UINT64 fence_value = 0;

        // 1. Unknown DSV flags must be rejected (descriptor left invalid, ClearDepthStencilView fails Close).
        list->ClearDepthStencilView(d24_bad, D3D12_CLEAR_FLAG_DEPTH, 0.5f, 0, 0, nullptr);
        if (list->Close() != E_INVALIDARG) {
            std::fprintf(stderr, "FAIL: unknown DSV flag 0x4 was not rejected\n");
            ++failures;
        }

        // 2. READ_ONLY_DEPTH + writable stencil: depth stays 0.75f across three draws while stencil increments twice.
        EXPECT(SUCCEEDED(alloc->Reset()) && SUCCEEDED(list->Reset(alloc.p, nullptr)));
        list->ClearDepthStencilView(d24_rw, D3D12_CLEAR_FLAG_DEPTH | D3D12_CLEAR_FLAG_STENCIL, 0.75f, 0x10, 0, nullptr);
        const auto d_to_ro = subresource_transition(
            d24.p, 0, D3D12_RESOURCE_STATE_DEPTH_WRITE,
            D3D12_RESOURCE_STATE_DEPTH_READ | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
        list->ResourceBarrier(1, &d_to_ro);
        draw_at_z(list.p, root.p, pso_less_always.p, d24_ro_d, 0.25f);   // passes (< 0.75), must NOT overwrite depth
        draw_at_z(list.p, root.p, pso_less_always.p, d24_ro_d, 0.50f);   // passes (< 0.75), fails if depth became 0.25
        draw_at_z(list.p, root.p, pso_less_always.p, d24_ro_d, 0.875f);  // fails (< 0.75)
        EXPECT(SUCCEEDED(list->Close()));
        {
            ID3D12CommandList* lists[] = {list.p};
            queue->ExecuteCommandLists(1, lists);
            EXPECT(rt::d3d12::last_execute_result(queue.p) == leonida::ReplayResult::success);
            EXPECT(SUCCEEDED(queue->Signal(fence.p, ++fence_value)) &&
                   SUCCEEDED(fence->SetEventOnCompletion(fence_value, nullptr)));
        }
        std::uint32_t got_d = 0;
        std::uint8_t got_s = 0;
        EXPECT(read_d24s8(queue.p, alloc.p, list.p, fence.p, fence_value, d24.p,
                          D3D12_RESOURCE_STATE_DEPTH_READ | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                          D3D12_RESOURCE_STATE_DEPTH_WRITE, rb24.p, foot24, got_d, got_s));
        if (!depth24_near(got_d, 0.75f) || got_s != 0x12) {
            std::fprintf(stderr, "FAIL READ_ONLY_DEPTH: depth 0x%06x (want ~0x%06x), stencil 0x%02x (want 0x12)\n",
                         got_d, leonida::depth24_quantize(0.75f), got_s);
            ++failures;
        }

        // 3. Writable depth + READ_ONLY_STENCIL: depth updates to 0.25f while stencil stays 0x12.
        EXPECT(SUCCEEDED(alloc->Reset()) && SUCCEEDED(list->Reset(alloc.p, nullptr)));
        const D3D12_RESOURCE_BARRIER swap_ro[2] = {
            subresource_transition(d24.p, 0,
                                   D3D12_RESOURCE_STATE_DEPTH_READ | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                                   D3D12_RESOURCE_STATE_DEPTH_WRITE),
            subresource_transition(d24.p, 1, D3D12_RESOURCE_STATE_DEPTH_WRITE, D3D12_RESOURCE_STATE_DEPTH_READ),
        };
        list->ResourceBarrier(2, swap_ro);
        draw_at_z(list.p, root.p, pso_less_eq.p, d24_ro_s, 0.25f, 0x12);
        EXPECT(SUCCEEDED(list->Close()));
        {
            ID3D12CommandList* lists[] = {list.p};
            queue->ExecuteCommandLists(1, lists);
            EXPECT(rt::d3d12::last_execute_result(queue.p) == leonida::ReplayResult::success);
            EXPECT(SUCCEEDED(queue->Signal(fence.p, ++fence_value)) &&
                   SUCCEEDED(fence->SetEventOnCompletion(fence_value, nullptr)));
        }
        EXPECT(read_d24s8(queue.p, alloc.p, list.p, fence.p, fence_value, d24.p,
                          D3D12_RESOURCE_STATE_DEPTH_WRITE, D3D12_RESOURCE_STATE_DEPTH_READ,
                          rb24.p, foot24, got_d, got_s));
        if (!depth24_near(got_d, 0.25f) || got_s != 0x12) {
            std::fprintf(stderr, "FAIL READ_ONLY_STENCIL: depth 0x%06x (want ~0x%06x), stencil 0x%02x (want 0x12)\n",
                         got_d, leonida::depth24_quantize(0.25f), got_s);
            ++failures;
        }

        // 4. Both read-only: draw and clear leave depth 0.25f and stencil 0x12 intact; READ_ONLY_DEPTH-only clear
        // updates stencil to 0x55 while keeping depth 0.25f.
        EXPECT(SUCCEEDED(alloc->Reset()) && SUCCEEDED(list->Reset(alloc.p, nullptr)));
        const auto d_to_ro2 = subresource_transition(d24.p, 0, D3D12_RESOURCE_STATE_DEPTH_WRITE,
                                                     D3D12_RESOURCE_STATE_DEPTH_READ);
        list->ResourceBarrier(1, &d_to_ro2);
        list->ClearDepthStencilView(d24_ro_both, D3D12_CLEAR_FLAG_DEPTH | D3D12_CLEAR_FLAG_STENCIL, 0.875f, 0x77, 0,
                                    nullptr);
        draw_at_z(list.p, root.p, pso_always_always.p, d24_ro_both, 0.50f);
        EXPECT(SUCCEEDED(list->Close()));
        {
            ID3D12CommandList* lists[] = {list.p};
            queue->ExecuteCommandLists(1, lists);
            EXPECT(rt::d3d12::last_execute_result(queue.p) == leonida::ReplayResult::success);
            EXPECT(SUCCEEDED(queue->Signal(fence.p, ++fence_value)) &&
                   SUCCEEDED(fence->SetEventOnCompletion(fence_value, nullptr)));
        }
        EXPECT(read_d24s8(queue.p, alloc.p, list.p, fence.p, fence_value, d24.p,
                          D3D12_RESOURCE_STATE_DEPTH_READ, D3D12_RESOURCE_STATE_DEPTH_READ,
                          rb24.p, foot24, got_d, got_s));
        if (!depth24_near(got_d, 0.25f) || got_s != 0x12) {
            std::fprintf(stderr, "FAIL READ_ONLY_BOTH: depth 0x%06x (want ~0x%06x), stencil 0x%02x (want 0x12)\n",
                         got_d, leonida::depth24_quantize(0.25f), got_s);
            ++failures;
        }

        EXPECT(SUCCEEDED(alloc->Reset()) && SUCCEEDED(list->Reset(alloc.p, nullptr)));
        const auto s_to_rw = subresource_transition(d24.p, 1, D3D12_RESOURCE_STATE_DEPTH_READ,
                                                    D3D12_RESOURCE_STATE_DEPTH_WRITE);
        list->ResourceBarrier(1, &s_to_rw);
        list->ClearDepthStencilView(d24_ro_d, D3D12_CLEAR_FLAG_DEPTH | D3D12_CLEAR_FLAG_STENCIL, 0.125f, 0x55, 0,
                                    nullptr);
        EXPECT(SUCCEEDED(list->Close()));
        {
            ID3D12CommandList* lists[] = {list.p};
            queue->ExecuteCommandLists(1, lists);
            EXPECT(rt::d3d12::last_execute_result(queue.p) == leonida::ReplayResult::success);
            EXPECT(SUCCEEDED(queue->Signal(fence.p, ++fence_value)) &&
                   SUCCEEDED(fence->SetEventOnCompletion(fence_value, nullptr)));
        }
        EXPECT(read_d24s8(queue.p, alloc.p, list.p, fence.p, fence_value, d24.p,
                          D3D12_RESOURCE_STATE_DEPTH_READ, D3D12_RESOURCE_STATE_DEPTH_WRITE,
                          rb24.p, foot24, got_d, got_s));
        if (!depth24_near(got_d, 0.25f) || got_s != 0x55) {
            std::fprintf(stderr, "FAIL partial clear on READ_ONLY_DEPTH: depth 0x%06x (want ~0x%06x), stencil 0x%02x (want 0x55)\n",
                         got_d, leonida::depth24_quantize(0.25f), got_s);
            ++failures;
        }

        // 5. D32_FLOAT (no stencil attachment) paired control: identical draws at 0.25f against cleared 0.75f with
        // read-only DSV vs writable DSV produce matching occlusion query counts while depth changes only in the
        // writable arm.
        Ref<ID3D12QueryHeap> qheap;
        const D3D12_QUERY_HEAP_DESC qhd{D3D12_QUERY_HEAP_TYPE_OCCLUSION, 2, 0};
        EXPECT(SUCCEEDED(device->CreateQueryHeap(&qhd, IID_PPV_ARGS(&qheap.p))));
        Ref<ID3D12Resource> rb_query;
        const auto rb_query_desc = buf_desc(2 * sizeof(std::uint64_t));
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&rb_heap, D3D12_HEAP_FLAG_NONE, &rb_query_desc,
                                                         D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                         IID_PPV_ARGS(&rb_query.p))));

        const auto read_d32 = [&](float& out_f) -> bool {
            std::uint8_t* mapped32 = nullptr;
            if (FAILED(rb32->Map(0, nullptr, reinterpret_cast<void**>(&mapped32))) || !mapped32) return false;
            std::memcpy(&out_f, mapped32 + foot32.Offset + (kSide / 2) * foot32.Footprint.RowPitch + (kSide / 2) * 4, 4);
            rb32->Unmap(0, nullptr);
            return true;
        };

        // 5a. Read-only arm: depth stays 0.75f, occlusion query 0 records all passing fragments.
        EXPECT(SUCCEEDED(alloc->Reset()) && SUCCEEDED(list->Reset(alloc.p, nullptr)));
        list->ClearDepthStencilView(d32_rw, D3D12_CLEAR_FLAG_DEPTH, 0.75f, 0, 0, nullptr);
        const auto d32_to_ro = subresource_transition(
            d32.p, 0, D3D12_RESOURCE_STATE_DEPTH_WRITE,
            D3D12_RESOURCE_STATE_DEPTH_READ | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
        list->ResourceBarrier(1, &d32_to_ro);
        list->BeginQuery(qheap.p, D3D12_QUERY_TYPE_OCCLUSION, 0);
        draw_at_z(list.p, root.p, pso_d32.p, d32_ro_d, 0.25f);
        list->EndQuery(qheap.p, D3D12_QUERY_TYPE_OCCLUSION, 0);
        const auto d32_to_copy = subresource_transition(
            d32.p, 0, D3D12_RESOURCE_STATE_DEPTH_READ | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
            D3D12_RESOURCE_STATE_COPY_SOURCE);
        list->ResourceBarrier(1, &d32_to_copy);
        D3D12_TEXTURE_COPY_LOCATION dst32{rb32.p, D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
        dst32.PlacedFootprint = foot32;
        D3D12_TEXTURE_COPY_LOCATION src32{d32.p, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
        src32.SubresourceIndex = 0;
        list->CopyTextureRegion(&dst32, 0, 0, 0, &src32, nullptr);
        EXPECT(SUCCEEDED(list->Close()));
        {
            ID3D12CommandList* lists[] = {list.p};
            queue->ExecuteCommandLists(1, lists);
            EXPECT(rt::d3d12::last_execute_result(queue.p) == leonida::ReplayResult::success);
            EXPECT(SUCCEEDED(queue->Signal(fence.p, ++fence_value)) &&
                   SUCCEEDED(fence->SetEventOnCompletion(fence_value, nullptr)));
        }
        float got_f_ro = 0.0f;
        EXPECT(read_d32(got_f_ro));
        if (got_f_ro != 0.75f) {
            std::fprintf(stderr, "FAIL D32_FLOAT READ_ONLY_DEPTH: got %g, want 0.75\n", double(got_f_ro));
            ++failures;
        }

        // 5b. Writable control arm: same clear + draw with d32_rw updates depth to 0.25f and matches query 0.
        EXPECT(SUCCEEDED(alloc->Reset()) && SUCCEEDED(list->Reset(alloc.p, nullptr)));
        const auto d32_to_rw = subresource_transition(d32.p, 0, D3D12_RESOURCE_STATE_COPY_SOURCE,
                                                      D3D12_RESOURCE_STATE_DEPTH_WRITE);
        list->ResourceBarrier(1, &d32_to_rw);
        list->ClearDepthStencilView(d32_rw, D3D12_CLEAR_FLAG_DEPTH, 0.75f, 0, 0, nullptr);
        list->BeginQuery(qheap.p, D3D12_QUERY_TYPE_OCCLUSION, 1);
        draw_at_z(list.p, root.p, pso_d32.p, d32_rw, 0.25f);
        list->EndQuery(qheap.p, D3D12_QUERY_TYPE_OCCLUSION, 1);
        const auto d32_rw_to_copy = subresource_transition(d32.p, 0, D3D12_RESOURCE_STATE_DEPTH_WRITE,
                                                           D3D12_RESOURCE_STATE_COPY_SOURCE);
        list->ResourceBarrier(1, &d32_rw_to_copy);
        list->CopyTextureRegion(&dst32, 0, 0, 0, &src32, nullptr);
        list->ResolveQueryData(qheap.p, D3D12_QUERY_TYPE_OCCLUSION, 0, 2, rb_query.p, 0);
        EXPECT(SUCCEEDED(list->Close()));
        {
            ID3D12CommandList* lists[] = {list.p};
            queue->ExecuteCommandLists(1, lists);
            EXPECT(rt::d3d12::last_execute_result(queue.p) == leonida::ReplayResult::success);
            EXPECT(SUCCEEDED(queue->Signal(fence.p, ++fence_value)) &&
                   SUCCEEDED(fence->SetEventOnCompletion(fence_value, nullptr)));
        }
        float got_f_rw = 0.0f;
        EXPECT(read_d32(got_f_rw));
        if (got_f_rw != 0.25f) {
            std::fprintf(stderr, "FAIL D32_FLOAT writable control: got %g, want 0.25\n", double(got_f_rw));
            ++failures;
        }
        std::uint64_t* mapped_q = nullptr;
        EXPECT(SUCCEEDED(rb_query->Map(0, nullptr, reinterpret_cast<void**>(&mapped_q))) && mapped_q);
        const std::uint64_t q_ro = mapped_q[0], q_rw = mapped_q[1];
        rb_query->Unmap(0, nullptr);
        const std::uint64_t want_q = std::uint64_t(kSide) * kSide;
        const bool queries_match = vk.occlusion_query_precise
            ? q_ro == want_q && q_rw == want_q
            : q_ro != 0 && q_rw != 0;
        if (!queries_match) {
            std::fprintf(stderr, "FAIL D32_FLOAT occlusion control: ro=%llu rw=%llu (want %llu)\n",
                         static_cast<unsigned long long>(q_ro), static_cast<unsigned long long>(q_rw),
                         static_cast<unsigned long long>(want_q));
            ++failures;
        }
    }

    std::filesystem::remove_all(library);
    if (vk.validation_errors()) {
        std::fprintf(stderr, "%u Vulkan validation errors\n", vk.validation_errors());
        return 1;
    }
    if (failures) {
        std::fprintf(stderr, "d3d12_dsv_readonly_test: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("d3d12_dsv_readonly_test: OK (unknown flags rejected, READ_ONLY_DEPTH, READ_ONLY_STENCIL, "
                "READ_ONLY_DEPTH|STENCIL, partial clear, and D32_FLOAT matched, 0 validation errors)\n");
    return 0;
}
