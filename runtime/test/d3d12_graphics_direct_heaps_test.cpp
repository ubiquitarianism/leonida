// Directly indexed descriptor heaps for graphics draws, through the Xbox-ABI proxy over the Vulkan device.
//
// A root signature with CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED and/or SAMPLER_HEAP_DIRECTLY_INDEXED gives the console's
// vertex, pixel and mesh shaders the bound heaps' 32-bit shader VAs in the user-data dwords right after the explicit
// root arguments (from s8, one dword per table): resource heap first, then sampler heap. The vertex and pixel shaders
// here (test/glsl/vs_user_data_window.vert, ps_user_data_window.frag) both read s10..s13 and write what they saw to
// two UINT render targets, one pixel per draw:
//   0. three constants (s8..s10), both flags:        s10 = constant, s11 = resource heap, s12 = sampler heap;
//   1. the same signature without the flags:         s11..s13 stay 0;
//   2. a table (s8) and three constants (s9..s11):    s12 = resource heap, s13 = sampler heap;
//   3. three constants, the sampler flag alone:      s11 = sampler heap;
//   4. as 0 after SetDescriptorHeaps with the resource heap alone: the sampler heap's dword is 0 for that draw only
//      (each draw snapshots the heaps, as a dispatch does);
//   5. a mesh pipeline (test/glsl/ms_user_data_window.mesh and ps_mesh_user_data_window.frag) after both heaps are
//      bound again: the mesh shader uses the graphics root window (push dword i -> s(i+8)), while the pixel shader
//      uses the same shared block. Target 0 gets the mesh stage's words and target 1 the pixel stage's, so one draw
//      shows both stages of the pipeline got the heaps. Skipped on a device without VK_EXT_mesh_shader.
// The recorded Draw commands are checked too. With --off (ctest d3d12_graphics_direct_heaps_off sets
// LEONIDA_GRAPHICS_DIRECT_HEAPS=0) no draw gets a heap dword. No game data.
// --indirect-resources changes the pixel shader to read slot 1 of the directly indexed resource heap. Five draws
// on one list (direct A, indirect B, indirect A, direct B, indirect B without a sampler) check actual payloads.
// Set LEONIDA_VK_VALIDATION=1 to run under VK_LAYER_KHRONOS_validation.
#include "shader_fixture_path.h"
#include "test_temp_dir.h"
#include "vk_test_device.h"
#include "d3d12_test_shaders.h"
#include "d3d12_pipeline_stream.h"
#include "../shims/d3d12_com.h"
#include "../shims/d3d12_xs_device.h"
#include "../src/shader_library.h"

#include <array>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <variant>
#include <vector>

#if !defined(LEONIDA_TEST_VS_SPV) || !defined(LEONIDA_TEST_PS_SPV) || !defined(LEONIDA_TEST_MS_SPV) || \
    !defined(LEONIDA_TEST_PS_MESH_SPV)
#error "LEONIDA_TEST_VS_SPV, LEONIDA_TEST_PS_SPV, LEONIDA_TEST_MS_SPV and LEONIDA_TEST_PS_MESH_SPV must name the compiled test shaders"
#endif

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

using namespace d3d12test;
using XboxHeapHandleFn = std::uint64_t(WINAPI*)(void* heap);  // heap slots 9/10: CPU/GPU handle in RAX

int main(int argc, char** argv) {
    const bool off = argc == 2 && std::strcmp(argv[1], "--off") == 0;
    const bool indirect_resources = argc == 2 && std::strcmp(argv[1], "--indirect-resources") == 0;
    if (off) {
        const char* v = std::getenv("LEONIDA_GRAPHICS_DIRECT_HEAPS");
        EXPECT(v && std::strcmp(v, "0") == 0);
    }
    vktest::Device d;
    d.request_mesh_shader = !indirect_resources;  // the mesh draw of this test skips when the GPU has no VK_EXT_mesh_shader
    d.request_draw_indirect_first_instance = indirect_resources;
    const auto device_status = d.initialize();
    if (device_status == vktest::DeviceStatus::unavailable) {
        std::fprintf(stderr, "SKIP: no usable Vulkan 1.3 device\n");
        return 77;
    }
    EXPECT(device_status == vktest::DeviceStatus::ready);
    if (indirect_resources && !d.enabled.drawIndirectFirstInstance) {
        std::fprintf(stderr, "SKIP: no enabled indirect first-instance capability\n");
        return 77;
    }

    const auto vs_program = fssm_program({0x56530011, 0xbf810000});
    const auto ps_program = fssm_program({0x50530011, 0xbf810000});
    const auto ms_program = fssm_program({0x4d530011, 0xbf810000});
    const auto ps_mesh_program = fssm_program({0x50530012, 0xbf810000});
    const auto vs_container = shader_container(vs_program), ps_container = shader_container(ps_program);
    auto ms_container = shader_container(ms_program);
    ms_container[36 + 8 + 2] = 13;  // the DXIL mesh shader kind, as the game's mesh containers carry
    const auto ps_mesh_container = shader_container(ps_mesh_program);
    const testfiles::ScratchDir scratch("leonida_graphics_direct_heaps_test");
    const auto& library = scratch.path();
    EXPECT(testfiles::copy_file(vktest::shader_fixture_path(LEONIDA_TEST_VS_SPV), library / "vs_user_data_window.spv"));
    EXPECT(testfiles::copy_file(vktest::shader_fixture_path(indirect_resources ? LEONIDA_TEST_PS_INDIRECT_HEAP
                                                                               : LEONIDA_TEST_PS_SPV),
                                library / "ps_user_data_window.spv"));
    EXPECT(testfiles::copy_file(vktest::shader_fixture_path(LEONIDA_TEST_MS_SPV), library / "ms_user_data_window.spv"));
    EXPECT(testfiles::copy_file(vktest::shader_fixture_path(LEONIDA_TEST_PS_MESH_SPV),
                                library / "ps_mesh_user_data_window.spv"));
    {
        std::ofstream index(library / "index.json", std::ios::binary);
        index << "{\"vs_user_data_window\": {\"stage\": \"vs\", \"fssm_sha256\": \"" << leonida::sha256_hex(vs_program)
              << "\", \"bindings\": [], \"push_constant_bytes\": 144, \"input_signature\": []},\n"
                 " \"ps_user_data_window\": {\"stage\": \"ps\", \"fssm_sha256\": \"" << leonida::sha256_hex(ps_program)
              << "\", \"bindings\": [], \"push_constant_bytes\": 144, \"outputs\": [0, 1]},\n"
                 " \"ms_user_data_window\": {\"stage\": \"ms\", \"fssm_sha256\": \"" << leonida::sha256_hex(ms_program)
              << "\", \"bindings\": [], \"push_constant_bytes\": 144, \"local_size\": [32, 1, 1], "
                 "\"root_window\": {\"sgpr\": 16, \"dword\": 8}, "
                 "\"max_vertices\": 3, \"max_primitives\": 1},\n"
                 " \"ps_mesh_user_data_window\": {\"stage\": \"ps\", \"fssm_sha256\": \""
              << leonida::sha256_hex(ps_mesh_program)
              << "\", \"bindings\": [], \"push_constant_bytes\": 144, \"outputs\": [0, 1]}}";
    }

    constexpr UINT draws = 6;              // five vertex draws, then one mesh draw
    const UINT used = d.mesh_shader && !indirect_resources ? draws : draws - 1;  // without VK_EXT_mesh_shader the mesh draw is skipped
    constexpr UINT mesh_draw = draws - 1;
    constexpr std::uint32_t mesh_marker = 0x4d53u;  // 'MS': what the mesh shader puts beside the heaps it read
    constexpr UINT pixel_bytes = 16;                // R32G32B32A32_UINT
    constexpr UINT user_dwords = 32;                // the user-data block: s0..s31
    std::array<std::array<std::uint32_t, 4>, draws> vs_seen{}, ps_seen{}, want_vs{}, want_ps{};
    std::array<std::array<std::uint32_t, user_dwords>, draws> want_user{};
    // The user-data dwords a recorded draw carries, from `first` on (the rest stay 0).
    const auto set_user = [](std::array<std::uint32_t, user_dwords>& data, UINT first,
                             std::initializer_list<std::uint32_t> values) {
        UINT dword = first;
        for (const std::uint32_t value : values) data[dword++] = value;
    };
    constexpr bool use_indirect[]{false, true, true, false, true};
    {
        rt::d3d12::VulkanDeviceInfo info{};
        info.get_instance_proc = d.instance_proc;
        info.instance = d.instance;
        info.physical = d.physical;
        info.device = d.device;
        info.queue = d.queue;
        info.queue_family = d.family;
        info.buffer_device_address = true;
        info.draw_indirect_first_instance = d.enabled.drawIndirectFirstInstance == VK_TRUE;
        info.max_draw_indirect_count = d.max_draw_indirect_count;
        info.subgroup = d.subgroup;
        info.shader_dir = library.string();
        info.mesh_shader = d.mesh_shader;  // the mesh draw's PSO needs VK_EXT_mesh_shader
        ID3D12Device* inner = nullptr;
        EXPECT(SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&inner))));
        Ref<ID3D12Device> device;
        EXPECT(SUCCEEDED(rt::d3d12_xs::create_device_proxy(inner, __uuidof(ID3D12Device),
                                                           reinterpret_cast<void**>(&device.p), /*console_lists=*/false)));
        rt::d3d12::set_device_outer(inner, device.p);

        // The shader-visible heaps whose VAs the draws receive.
        const D3D12_DESCRIPTOR_HEAP_DESC resource_desc{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 8,
                                                       D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
        const D3D12_DESCRIPTOR_HEAP_DESC sampler_desc{D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, 8,
                                                      D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
        Ref<ID3D12DescriptorHeap> resources, samplers;
        EXPECT(SUCCEEDED(device->CreateDescriptorHeap(&resource_desc, IID_PPV_ARGS(&resources.p))));
        EXPECT(SUCCEEDED(device->CreateDescriptorHeap(&sampler_desc, IID_PPV_ARGS(&samplers.p))));
        const std::uint64_t res = reinterpret_cast<XboxHeapHandleFn>(slot(resources.p, 10))(resources.p);
        const std::uint64_t smp = reinterpret_cast<XboxHeapHandleFn>(slot(samplers.p, 10))(samplers.p);
        EXPECT(res >= 0x10000 && res < (1ull << 32) && smp >= 0x10000 && smp < (1ull << 32) && res != smp);

        // Root signatures: three constants (s8..s10), or a table (s8) and three constants (s9..s11).
        D3D12_DESCRIPTOR_RANGE range{D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0, 0, 0};
        D3D12_ROOT_PARAMETER three[1] = {};
        three[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        three[0].Constants = {0, 0, 3};
        D3D12_ROOT_PARAMETER table_first[2] = {};
        table_first[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        table_first[0].DescriptorTable = {1, &range};
        table_first[1] = three[0];
        constexpr auto both = D3D12_ROOT_SIGNATURE_FLAG_CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED |
                              D3D12_ROOT_SIGNATURE_FLAG_SAMPLER_HEAP_DIRECTLY_INDEXED;
        const auto make_root = [&](const D3D12_ROOT_PARAMETER* params, UINT count, D3D12_ROOT_SIGNATURE_FLAGS flags,
                                   Ref<ID3D12RootSignature>& out) {
            D3D12_VERSIONED_ROOT_SIGNATURE_DESC desc{};
            desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_0;
            desc.Desc_1_0 = {count, params, 0, nullptr, flags};
            const auto blob = serialize_root_signature(desc);
            return !blob.empty() &&
                   SUCCEEDED(device->CreateRootSignature(0, blob.data(), blob.size(), IID_PPV_ARGS(&out.p)));
        };
        Ref<ID3D12RootSignature> three_both, three_none, table_both, three_sampler;
        EXPECT(make_root(three, 1, both, three_both));
        EXPECT(make_root(three, 1, D3D12_ROOT_SIGNATURE_FLAG_NONE, three_none));
        EXPECT(make_root(table_first, 2, both, table_both));
        EXPECT(make_root(three, 1, D3D12_ROOT_SIGNATURE_FLAG_SAMPLER_HEAP_DIRECTLY_INDEXED, three_sampler));

        D3D12_GRAPHICS_PIPELINE_STATE_DESC pso_desc{};
        pso_desc.VS = {vs_container.data(), vs_container.size()};
        pso_desc.PS = {ps_container.data(), ps_container.size()};
        pso_desc.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
        pso_desc.BlendState.RenderTarget[1].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
        pso_desc.SampleMask = 0xffffffff;
        pso_desc.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
        pso_desc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
        pso_desc.RasterizerState.DepthClipEnable = TRUE;
        pso_desc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        pso_desc.NumRenderTargets = 2;
        pso_desc.RTVFormats[0] = pso_desc.RTVFormats[1] = DXGI_FORMAT_R32G32B32A32_UINT;
        pso_desc.SampleDesc.Count = 1;
        const auto make_pso = [&](ID3D12RootSignature* root, Ref<ID3D12PipelineState>& out) {
            auto desc = pso_desc;
            desc.pRootSignature = root;
            return SUCCEEDED(device->CreateGraphicsPipelineState(&desc, IID_PPV_ARGS(&out.p)));
        };
        Ref<ID3D12PipelineState> pso_three_both, pso_three_none, pso_table_both, pso_three_sampler;
        EXPECT(make_pso(three_both.p, pso_three_both));
        EXPECT(make_pso(three_none.p, pso_three_none));
        EXPECT(make_pso(table_both.p, pso_table_both));
        EXPECT(make_pso(three_sampler.p, pso_three_sampler));

        // The mesh PSO of draw 5: a mesh shader has no D3D12_GRAPHICS_PIPELINE_STATE_DESC field, so it comes from a
        // pipeline state stream. Back-face culling is off so the triangle's winding does not matter here.
        Ref<ID3D12PipelineState> pso_mesh;
        if (d.mesh_shader) {
            D3D12_RASTERIZER_DESC raster{};
            raster.FillMode = D3D12_FILL_MODE_SOLID;
            raster.CullMode = D3D12_CULL_MODE_NONE;
            raster.DepthClipEnable = TRUE;
            Stream stream;
            stream.add(RootSubobject{.inner = three_both.p})
                  .add(MsSubobject{.inner = {ms_container.data(), ms_container.size()}})
                  .add(PsSubobject{.inner = {ps_mesh_container.data(), ps_mesh_container.size()}})
                  .add(RasterizerSubobject{.inner = raster})
                  .add(TargetsSubobject{.inner = rt_formats(2, DXGI_FORMAT_R32G32B32A32_UINT)});
            const HRESULT hr = create(device.p, stream, pso_mesh);
            if (FAILED(hr)) std::fprintf(stderr, "mesh PSO: 0x%08lx\n", static_cast<unsigned long>(hr));
            EXPECT(SUCCEEDED(hr));
        }

        // Two 6x1 UINT targets (what the VS or mesh stage saw, what the PS saw) and their RTVs.
        const D3D12_HEAP_PROPERTIES default_heap{D3D12_HEAP_TYPE_DEFAULT};
        const D3D12_HEAP_PROPERTIES readback_heap{D3D12_HEAP_TYPE_READBACK};
        D3D12_RESOURCE_DESC rt_desc{};
        rt_desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        rt_desc.Width = draws;
        rt_desc.Height = 1;
        rt_desc.DepthOrArraySize = 1;
        rt_desc.MipLevels = 1;
        rt_desc.Format = DXGI_FORMAT_R32G32B32A32_UINT;
        rt_desc.SampleDesc.Count = 1;
        rt_desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
        std::array<Ref<ID3D12Resource>, 2> targets;
        std::array<Ref<ID3D12DescriptorHeap>, 2> rtv_heaps;
        std::array<D3D12_CPU_DESCRIPTOR_HANDLE, 2> rtvs{};
        const D3D12_DESCRIPTOR_HEAP_DESC rtv_desc{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
                                                  0};
        for (std::size_t t = 0; t < targets.size(); ++t) {
            EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &rt_desc,
                                                             D3D12_RESOURCE_STATE_RENDER_TARGET, nullptr,
                                                             IID_PPV_ARGS(&targets[t].p))));
            EXPECT(SUCCEEDED(device->CreateDescriptorHeap(&rtv_desc, IID_PPV_ARGS(&rtv_heaps[t].p))));
            rtvs[t] = {SIZE_T(reinterpret_cast<XboxHeapHandleFn>(slot(rtv_heaps[t].p, 9))(rtv_heaps[t].p))};
            device->CreateRenderTargetView(targets[t].p, nullptr, rtvs[t]);
        }
        Ref<ID3D12Resource> readback;
        const auto readback_desc = buffer_desc(1024);
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&readback_heap, D3D12_HEAP_FLAG_NONE, &readback_desc,
                                                         D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                         IID_PPV_ARGS(&readback.p))));

        D3D12_COMMAND_QUEUE_DESC queue_desc{};
        Ref<ID3D12CommandQueue> queue;
        Ref<ID3D12CommandAllocator> allocator;
        Ref<ID3D12GraphicsCommandList> list;
        Ref<ID3D12GraphicsCommandList6> list6;  // DispatchMesh
        Ref<ID3D12Fence> fence;
        EXPECT(SUCCEEDED(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue.p))));
        EXPECT(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p))));
        EXPECT(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p,
                                                   pso_three_both.p, IID_PPV_ARGS(&list.p))));
        if (d.mesh_shader) EXPECT(SUCCEEDED(list->QueryInterface(IID_PPV_ARGS(&list6.p))));
        EXPECT(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))));

        const FLOAT seven[4] = {7, 7, 7, 7};  // a pixel no draw reached reads 7s
        for (const auto& rtv : rtvs) list->ClearRenderTargetView(rtv, seven, 0, nullptr);
        list->OMSetRenderTargets(2, rtvs.data(), FALSE, nullptr);
        list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        const D3D12_RECT scissor{0, 0, LONG(draws), 1};
        list->RSSetScissorRects(1, &scissor);
        ID3D12DescriptorHeap* both_heaps[] = {resources.p, samplers.p};
        list->SetDescriptorHeaps(2, both_heaps);
        const std::uint32_t r = off ? 0 : std::uint32_t(res), s = off ? 0 : std::uint32_t(smp);
        const UINT a[3] = {0xA0, 0xA1, 0xA2}, b[3] = {0xB0, 0xB1, 0xB2};
        const auto draw_at = [&](UINT pixel, ID3D12PipelineState* pso, ID3D12RootSignature* root, bool table) {
            const D3D12_VIEWPORT viewport{float(pixel), 0, 1, 1, 0, 1};
            list->RSSetViewports(1, &viewport);
            list->SetPipelineState(pso);
            list->SetGraphicsRootSignature(root);
            if (table) {
                list->SetGraphicsRootDescriptorTable(0, {res});
                list->SetGraphicsRoot32BitConstants(1, 3, b, 0);
            } else {
                list->SetGraphicsRoot32BitConstants(0, 3, a, 0);
            }
            list->DrawInstanced(3, 1, 0, 0);
        };
        Ref<ID3D12Resource> args, inputs[2];
        Ref<ID3D12CommandSignature> signature;
        Ref<ID3D12DescriptorHeap> alternate;
        if (indirect_resources) {
            EXPECT(SUCCEEDED(device->CreateDescriptorHeap(&resource_desc, IID_PPV_ARGS(&alternate.p))));
            ID3D12DescriptorHeap* heaps[]{resources.p, alternate.p};
            const D3D12_HEAP_PROPERTIES upload{D3D12_HEAP_TYPE_UPLOAD};
            const auto desc = buffer_desc(256);
            constexpr std::array<std::array<UINT, 4>, 2> values{{{11, 13, 17, 19}, {101, 103, 107, 109}}};
            const auto write_buffer = [&](Ref<ID3D12Resource>& dst, const void* src, std::size_t bytes) {
                if (FAILED(device->CreateCommittedResource(&upload, {}, &desc, D3D12_RESOURCE_STATE_GENERIC_READ,
                                                           nullptr, IID_PPV_ARGS(&dst.p)))) return false;
                void* data{};
                if (FAILED(dst->Map(0, nullptr, &data))) return false;
                std::memcpy(data, src, bytes);
                dst->Unmap(0, nullptr);
                return true;
            };
            D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
            srv.Format = DXGI_FORMAT_R32_TYPELESS;
            srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            srv.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
            srv.Buffer.NumElements = 4;
            srv.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
            for (UINT i = 0; i < 2; ++i) {
                EXPECT(write_buffer(inputs[i], values[i].data(), sizeof(values[i])));
                const auto cpu = reinterpret_cast<XboxHeapHandleFn>(slot(heaps[i], 9))(heaps[i]);
                device->CreateShaderResourceView(inputs[i].p, &srv, {SIZE_T(cpu + 32)}); // slot 1
            }
            const D3D12_DRAW_ARGUMENTS record{3, 1, 0, 0};
            EXPECT(write_buffer(args, &record, sizeof(record)));
            D3D12_INDIRECT_ARGUMENT_DESC action{};
            action.Type = D3D12_INDIRECT_ARGUMENT_TYPE_DRAW;
            const D3D12_COMMAND_SIGNATURE_DESC description{sizeof(record), 1, &action, 0};
            EXPECT(SUCCEEDED(inner->CreateCommandSignature(&description, nullptr, IID_PPV_ARGS(&signature.p))));
            constexpr UINT which[]{0, 1, 0, 1, 1};  // both sources have a direct control on this list
            for (UINT i = 0; i < used; ++i) {
                const auto h = which[i];
                auto* heap = heaps[h];
                const auto gpu = reinterpret_cast<XboxHeapHandleFn>(slot(heap, 10))(heap);
                EXPECT(gpu >= 0x10000 && gpu <= 0xffffffffu - 64u);
                if (h) EXPECT(gpu != res);
                const auto va = inputs[h]->GetGPUVirtualAddress();
                EXPECT(va);
                const UINT allowed[]{UINT(gpu), UINT(va), UINT(va >> 32)};
                ID3D12DescriptorHeap* bound[]{heap, samplers.p};
                list->SetDescriptorHeaps(i + 1 == used ? 1 : 2, bound);
                list->SetGraphicsRootSignature(three_both.p);
                list->SetPipelineState(pso_three_both.p);
                list->SetGraphicsRoot32BitConstants(0, 3, allowed, 0);
                const D3D12_VIEWPORT vp{float(i), 0, 1, 1, 0, 1};
                list->RSSetViewports(1, &vp);
                if (use_indirect[i]) list->ExecuteIndirect(signature.p, 1, args.p, 0, nullptr, 0);
                else list->DrawInstanced(3, 1, 0, 0);
                want_vs[i] = {allowed[2], UINT(gpu), i + 1 == used ? 0u : s, 0};
                want_ps[i] = values[h];
                set_user(want_user[i], 8, {allowed[0], allowed[1], allowed[2]});
                set_user(want_user[i], 11, {UINT(gpu), i + 1 == used ? 0u : s});
            }
        } else {
            // The heap dwords a draw of this signature carries after the explicit arguments (`pixel_slot`). The same
            // shared graphics block is used by vertex, pixel and mesh stages; root_window maps mesh push dword i to s(i+8).
            const auto set_heap_windows = [&](std::array<std::uint32_t, user_dwords>& data, UINT pixel_slot, bool resource,
                                              bool sampler) {
                UINT slot = pixel_slot;
                if (resource) data[slot++] = r;
                if (sampler) data[slot] = s;
            };
            draw_at(0, pso_three_both.p, three_both.p, false);
            want_vs[0] = want_ps[0] = {0xA2, r, s, 0};
            set_user(want_user[0], 8, {0xA0, 0xA1, 0xA2});
            set_heap_windows(want_user[0], 11, true, true);
            draw_at(1, pso_three_none.p, three_none.p, false);
            want_vs[1] = want_ps[1] = {0xA2, 0, 0, 0};
            set_user(want_user[1], 8, {0xA0, 0xA1, 0xA2});  // no flags: no heap dword in either window
            draw_at(2, pso_table_both.p, table_both.p, true);
            want_vs[2] = want_ps[2] = {0xB1, 0xB2, r, s};
            set_user(want_user[2], 9, {0xB0, 0xB1, 0xB2});  // s8 is the table's handle, not a constant
            set_heap_windows(want_user[2], 12, true, true);  // a table (s8) and three constants (s9..s11)
            draw_at(3, pso_three_sampler.p, three_sampler.p, false);
            want_vs[3] = want_ps[3] = {0xA2, s, 0, 0};
            set_user(want_user[3], 8, {0xA0, 0xA1, 0xA2});
            set_heap_windows(want_user[3], 11, false, true);
            list->SetDescriptorHeaps(1, &resources.p);  // the sampler heap is unbound for the next draw
            draw_at(4, pso_three_both.p, three_both.p, false);
            want_vs[4] = want_ps[4] = {0xA2, r, 0, 0};
            set_user(want_user[4], 8, {0xA0, 0xA1, 0xA2});
            set_heap_windows(want_user[4], 11, true, false);  // 0 for the unbound sampler heap, this draw only
            if (d.mesh_shader) {
                // Both heaps bound again: both stages consume the same graphics user-data block.
                list->SetDescriptorHeaps(2, both_heaps);
                const D3D12_VIEWPORT viewport{float(mesh_draw), 0, 1, 1, 0, 1};
                list->RSSetViewports(1, &viewport);
                list->SetPipelineState(pso_mesh.p);
                list->SetGraphicsRootSignature(three_both.p);
                list->SetGraphicsRoot32BitConstants(0, 3, a, 0);
                list6->DispatchMesh(1, 1, 1);
                want_vs[mesh_draw] = {r, s, mesh_marker, 0};  // the mesh stage's own words
                want_ps[mesh_draw] = {r, s, 0, 0};
                set_user(want_user[mesh_draw], 8, {0xA0, 0xA1, 0xA2});
                set_heap_windows(want_user[mesh_draw], 11, true, true);
            }
        }

        for (std::size_t t = 0; t < targets.size(); ++t) {
            const auto to_copy = transition(targets[t].p, D3D12_RESOURCE_STATE_RENDER_TARGET,
                                            D3D12_RESOURCE_STATE_COPY_SOURCE);
            list->ResourceBarrier(1, &to_copy);
            D3D12_TEXTURE_COPY_LOCATION dst{};
            dst.pResource = readback.p;
            dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
            dst.PlacedFootprint = {512 * t, {DXGI_FORMAT_R32G32B32A32_UINT, draws, 1, 1, 256}};
            D3D12_TEXTURE_COPY_LOCATION src{};
            src.pResource = targets[t].p;
            src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
            list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
        }
        EXPECT(list->Close() == S_OK);

        // The recorded draws carry the same words, snapshotted per draw: explicit arguments and shared heap dwords.
        std::vector<leonida::Command> commands;
        EXPECT(rt::d3d12::recorded_commands(list.p, commands));
        UINT recorded = 0, recorded_mismatches = 0;
        for (const auto& command : commands) {
            const auto* draw = std::get_if<leonida::Draw>(&command);
            const auto* raster = std::get_if<leonida::IndirectRasterDraw>(&command);
            if (raster) {
                EXPECT(indirect_resources && *raster && recorded < used && use_indirect[recorded]);
                draw = &(*raster)->state;
            } else if (draw && indirect_resources) EXPECT(recorded < used && !use_indirect[recorded]);
            if (!draw) continue;
            EXPECT(recorded < used);
            const UINT index = recorded++;
            const auto& want_dwords = want_user[index];
            for (UINT k = 9; k < user_dwords && k < draw->user_data.size(); ++k) {
                if (draw->user_data[k] == want_dwords[k]) continue;
                ++recorded_mismatches;
                std::fprintf(stderr, "recorded draw %u s%u: got %08x want %08x\n", index, k, draw->user_data[k],
                             want_dwords[k]);
            }
        }
        EXPECT(recorded == used);
        EXPECT(recorded_mismatches == 0);

        ID3D12CommandList* lists[] = {list.p};
        queue->ExecuteCommandLists(1, lists);
        EXPECT(rt::d3d12::last_execute_result(queue.p) == leonida::ReplayResult::success);
        EXPECT(SUCCEEDED(queue->Signal(fence.p, 1)));
        EXPECT(SUCCEEDED(fence->SetEventOnCompletion(1, nullptr)));
        void* mapped = nullptr;
        EXPECT(SUCCEEDED(readback->Map(0, nullptr, &mapped)));
        const auto* bytes = static_cast<const std::uint8_t*>(mapped);
        for (UINT pixel = 0; pixel < used; ++pixel) {
            std::memcpy(vs_seen[pixel].data(), bytes + pixel * pixel_bytes, pixel_bytes);
            std::memcpy(ps_seen[pixel].data(), bytes + 512 + pixel * pixel_bytes, pixel_bytes);
        }
        readback->Unmap(0, nullptr);

        pso_mesh.reset();
        for (auto* p : {&pso_three_both, &pso_three_none, &pso_table_both, &pso_three_sampler}) p->reset();
        for (auto* p : {&three_both, &three_none, &table_both, &three_sampler}) p->reset();
        for (auto& t : targets) t.reset();
        for (auto& h : rtv_heaps) h.reset();
        args.reset();
        signature.reset();
        for (auto& input : inputs) input.reset();
        alternate.reset();
        readback.reset();
        resources.reset();
        samplers.reset();
        list6.reset();
        list.reset();
        allocator.reset();
        queue.reset();
        fence.reset();
        EXPECT(rt::d3d12_xs::device_proxy_ref_count(device.p) == 1);
    }

    unsigned mismatches = 0;
    for (UINT pixel = 0; pixel < used; ++pixel) {
        for (int which = 0; which < 2; ++which) {
            const auto& got = which == 0 ? vs_seen[pixel] : ps_seen[pixel];
            const auto& want_words = which == 0 ? want_vs[pixel] : want_ps[pixel];
            if (got == want_words) continue;
            ++mismatches;
            std::fprintf(stderr, "draw %u %s: got %08x %08x %08x %08x want %08x %08x %08x %08x\n", pixel,
                         which == 0 ? "VS/MS words" : "PS words", got[0], got[1], got[2], got[3], want_words[0],
                         want_words[1], want_words[2], want_words[3]);
        }
    }
    EXPECT(!mismatches);
    if (d.validation_errors()) {
        std::fprintf(stderr, "%u validation errors\n", d.validation_errors());
        return 1;
    }
    if (indirect_resources) {
        std::puts("3 indirect draws and 2 direct controls: exact VS heap state and PS resource payloads across heap changes");
        return 0;
    }
    std::printf("graphics direct heaps%s: %u draws, the VS, PS and%s read the heap dwords as recorded (after the "
                "explicit arguments at s11/s12 or s12/s13, the sampler alone at s11, none without the flags, per-draw "
                "snapshot%s)\n",
                off ? " (LEONIDA_GRAPHICS_DIRECT_HEAPS=0: none supplied)" : "", used,
                d.mesh_shader ? " mesh stage" : "", d.mesh_shader ? ", the mesh stage at s19/s20" : "");
    if (!d.mesh_shader) std::fprintf(stderr, "no VK_EXT_mesh_shader on this device: the mesh draw was skipped\n");
    return 0;
}
