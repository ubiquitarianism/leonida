// Indexed console quad lists (IASetPrimitiveTopology(19) with DrawIndexedInstanced) through the D3D12-on-Vulkan
// layer (src/quad_index_expand.h).
//
// Built like d3d12_rect_list_test.cpp, with two translated shaders of its own: vs_quad_inst (vertex fetch through a
// V#, each instance after the first moved by a root constant and its red and green swapped) and ps_quad_flat (the
// colour as a Flat input, so a quad shows the colour of the vertex it takes flat inputs from: its first). Vertices
// 0..3 and 4..7 are full-screen quads with a different colour at every corner; 8..11 and 12..15 are the left and the
// right half of the target, red at every corner. Most cases use a DEFAULT heap index buffer written by a copy in
// the first command list, so nothing but the GPU can read it when the draw is recorded. Seven 64x64 frames:
//   frame 0, control: a non-indexed quad, DrawInstanced(4) over vertices 0..3. Red (vertex 0) everywhere.
//   frame 1: 16-bit indices 0 1 2 3. Red everywhere; a triangle fan shows vertices 1 and 2 (black and blue).
//   frame 2: 32-bit indices 7 4 5 6 from StartIndexLocation 3. Green (vertex 7) everywhere.
//   frame 3: 16-bit indices 0 1 2 3 from StartIndexLocation 5 (an address of 2 modulo 4) with BaseVertexLocation 3.
//     The translated vertex shader's vertex id is the index without the base vertex, so this is frame 1 again; the
//     base added into the indices would fetch vertices 3..6 instead (white).
//   frame 4: two quads (left half, right half) and two instances; instance 1 is green and one half to the right, so
//     its first quad lands on instance 0's second. D3D12 draws instance by instance: the right half ends green.
//     Drawn quad by quad, instance 0's second quad comes last and leaves it red.
//   frame 5: eight 16-bit indices, 0 1 2 3 and 7 4 5 6, through a view that ends after the sixth. The last two read
//     0 (D3D12: an index outside the view), which makes the second quad empty: red everywhere. Read past the view,
//     the second quad covers the target in green.
//   frame 6: a standalone 10-byte UPLOAD index buffer, first index 1. All four 16-bit indices are readable, with
//     the last one at the end of the resource. The backing buffer must accommodate the full word load.
// Every pixel is compared, bit for bit. LEONIDA_VK_VALIDATION=1 runs it under the validation layer.
#include "vk_test_device.h"
#include "d3d12_test_shaders.h"
#include "../shims/d3d12_com.h"
#include "../shims/d3d12_xs_device.h"
#include "../src/gfx10_descriptors.h"
#include "../src/shader_library.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iterator>

#ifndef LEONIDA_TEST_QUAD_DIR
#error "LEONIDA_TEST_QUAD_DIR must name the folder build.py wrote (spirv/index.json, fssm/*.fssm)"
#endif

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

using namespace d3d12test;
using XboxHeapHandleFn = std::uint64_t(WINAPI*)(void* heap);  // heap slots 9/10: handle in RAX

namespace {

std::vector<std::uint8_t> read_file(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

struct Vertex {  // 32 bytes, as vs_quad_inst.asm reads it
    float position[4];
    float color[4];
};

using Pixel = std::array<float, 4>;
constexpr UINT target_extent = 64;
constexpr UINT pitch = target_extent * sizeof(Pixel);
constexpr D3D_PRIMITIVE_TOPOLOGY quad_list = D3D_PRIMITIVE_TOPOLOGY(19);
constexpr Pixel red = {1, 0, 0, 1}, green = {0, 1, 0, 1}, blue = {0, 0, 1, 1}, black = {0, 0, 0, 1},
                white = {1, 1, 1, 1};

Vertex vertex(float x, float y, const Pixel& c) { return Vertex{{x, y, 0.0f, 1.0f}, {c[0], c[1], c[2], c[3]}}; }

struct Frame {
    const char* name;
    DXGI_FORMAT index_format;  // DXGI_FORMAT_UNKNOWN: DrawInstanced
    UINT view_offset, view_bytes;  // the index buffer view, in the index buffer
    UINT count, instances, start;
    INT base_vertex;
    Pixel left, right;  // what the left and the right half of the target must hold
    bool use_upload_indices = false;
};

}  // namespace

int main() {
    vktest::Device d;
    const auto device_status = d.initialize();
    if (device_status == vktest::DeviceStatus::unavailable) return 77;  // no Vulkan GPU: ctest skips
    EXPECT(device_status == vktest::DeviceStatus::ready);
    const std::filesystem::path root_dir = LEONIDA_TEST_QUAD_DIR;
    const auto vs_program = read_file(root_dir / "fssm" / "vs_quad_inst.fssm");
    const auto ps_program = read_file(root_dir / "fssm" / "ps_quad_flat.fssm");
    EXPECT(vs_program.size() >= 4 && ps_program.size() >= 4);
    const auto vs_container = shader_container(vs_program), ps_container = shader_container(ps_program);

    const Vertex vertices[16] = {
        vertex(-1, 1, red), vertex(1, 1, black), vertex(1, -1, blue), vertex(-1, -1, white),
        vertex(-1, 1, black), vertex(1, 1, blue), vertex(1, -1, white), vertex(-1, -1, green),
        vertex(-1, 1, red), vertex(0, 1, red), vertex(0, -1, red), vertex(-1, -1, red),
        vertex(0, 1, red), vertex(1, 1, red), vertex(1, -1, red), vertex(0, -1, red)};
    static_assert(sizeof(Vertex) == 32 && sizeof(vertices) == 512);
    // Index 15 fills the gaps: a draw that starts at the wrong index shows it.
    const std::uint16_t indices16[28] = {0, 1, 2, 3, 15, 0, 1, 2, 3, 15, 8, 9, 10, 11, 12, 13, 14, 15, 15, 15,
                                         0, 1, 2, 3, 7, 4, 5, 6};
    const std::uint32_t indices32[7] = {15, 15, 15, 7, 4, 5, 6};
    const std::uint16_t upload_indices16[] = {15, 0, 1, 2, 3};
    constexpr UINT indices32_offset = 64, index_bytes = 128;
    static_assert(sizeof(indices16) <= indices32_offset && indices32_offset + sizeof(indices32) <= index_bytes);
    const Frame frames[] = {
        {"non-indexed (control)", DXGI_FORMAT_UNKNOWN, 0, 0, 4, 1, 0, 0, red, red},
        {"16-bit indices", DXGI_FORMAT_R16_UINT, 0, sizeof(indices16), 4, 1, 0, 0, red, red},
        {"32-bit indices, first index 3", DXGI_FORMAT_R32_UINT, indices32_offset, sizeof(indices32), 4, 1, 3, 0,
         green, green},
        {"16-bit indices, first index 5, base vertex 3", DXGI_FORMAT_R16_UINT, 0, sizeof(indices16), 4, 1, 5, 3,
         red, red},
        {"two overlapping instances", DXGI_FORMAT_R16_UINT, 0, sizeof(indices16), 8, 2, 10, 0, red, green},
        {"view ends inside the second quad", DXGI_FORMAT_R16_UINT, 20 * sizeof(std::uint16_t),
         6 * sizeof(std::uint16_t), 8, 1, 0, 0, red, red},
        {"16-bit indices at the end of an upload buffer", DXGI_FORMAT_R16_UINT, 0,
         sizeof(upload_indices16), 4, 1, 1, 0, red, red, true},
    };
    constexpr std::size_t frame_count = std::size(frames);
    std::vector<Pixel> pixels[frame_count];
    for (auto& p : pixels) p.resize(target_extent * target_extent);
    {
        rt::d3d12::VulkanDeviceInfo info{};
        info.get_instance_proc = d.instance_proc;
        info.instance = d.instance;
        info.physical = d.physical;
        info.device = d.device;
        info.queue = d.queue;
        info.queue_family = d.family;
        info.buffer_device_address = true;
        info.features = d.enabled;
        info.shader_dir = (root_dir / "spirv").string();
        ID3D12Device* inner = nullptr;
        EXPECT(SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&inner))));
        Ref<ID3D12Device> device;
        EXPECT(SUCCEEDED(rt::d3d12_xs::create_device_proxy(inner, __uuidof(ID3D12Device),
                                                           reinterpret_cast<void**>(&device.p))));
        rt::d3d12::set_device_outer(inner, device.p);

        D3D12_DESCRIPTOR_RANGE range{D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0, 0, 0};
        D3D12_ROOT_PARAMETER params[2] = {};
        params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        params[0].DescriptorTable = {1, &range};
        params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        params[1].Constants = {0, 0, 4};
        D3D12_VERSIONED_ROOT_SIGNATURE_DESC root_desc{};
        root_desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_0;
        root_desc.Desc_1_0 = {2, params, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
        const auto root_blob = serialize_root_signature(root_desc);
        EXPECT(!root_blob.empty());
        Ref<ID3D12RootSignature> root;
        EXPECT(SUCCEEDED(device->CreateRootSignature(0, root_blob.data(), root_blob.size(), IID_PPV_ARGS(&root.p))));

        D3D12_GRAPHICS_PIPELINE_STATE_DESC pso_desc{};
        pso_desc.pRootSignature = root.p;
        pso_desc.VS = {vs_container.data(), vs_container.size()};
        pso_desc.PS = {ps_container.data(), ps_container.size()};
        pso_desc.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
        pso_desc.SampleMask = 0xffffffff;
        pso_desc.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
        pso_desc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
        pso_desc.RasterizerState.DepthClipEnable = TRUE;
        pso_desc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;  // quads use the triangle type
        pso_desc.NumRenderTargets = 1;
        pso_desc.RTVFormats[0] = DXGI_FORMAT_R32G32B32A32_FLOAT;
        pso_desc.SampleDesc.Count = 1;
        Ref<ID3D12PipelineState> pso;
        EXPECT(SUCCEEDED(device->CreateGraphicsPipelineState(&pso_desc, IID_PPV_ARGS(&pso.p))));

        const D3D12_HEAP_PROPERTIES default_heap{D3D12_HEAP_TYPE_DEFAULT};
        const D3D12_HEAP_PROPERTIES upload_heap{D3D12_HEAP_TYPE_UPLOAD};
        const D3D12_HEAP_PROPERTIES readback_heap{D3D12_HEAP_TYPE_READBACK};
        Ref<ID3D12Resource> vertex_buffer, index_buffer, upload_index_buffer, upload, readback, target;
        const auto vb_desc = buffer_desc(sizeof(vertices));
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &vb_desc,
                                                         D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                         IID_PPV_ARGS(&vertex_buffer.p))));
        const auto ib_desc = buffer_desc(index_bytes);
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &ib_desc,
                                                         D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                         IID_PPV_ARGS(&index_buffer.p))));
        const auto upload_ib_desc = buffer_desc(sizeof(upload_indices16));
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&upload_heap, D3D12_HEAP_FLAG_NONE, &upload_ib_desc,
                                                         D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                                         IID_PPV_ARGS(&upload_index_buffer.p))));
        EXPECT(upload_index_buffer->GetDesc().Width == sizeof(upload_indices16));
        void* index_map = nullptr;
        EXPECT(SUCCEEDED(upload_index_buffer->Map(0, nullptr, &index_map)));
        std::memcpy(index_map, upload_indices16, sizeof(upload_indices16));
        upload_index_buffer->Unmap(0, nullptr);
        constexpr UINT upload_indices = 512;  // where the index bytes sit in the upload buffer
        const auto upload_desc = buffer_desc(1024);
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&upload_heap, D3D12_HEAP_FLAG_NONE, &upload_desc,
                                                         D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                                         IID_PPV_ARGS(&upload.p))));
        const auto readback_desc = buffer_desc(pitch * target_extent);
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&readback_heap, D3D12_HEAP_FLAG_NONE, &readback_desc,
                                                         D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                         IID_PPV_ARGS(&readback.p))));
        D3D12_RESOURCE_DESC rt_desc{};
        rt_desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        rt_desc.Width = rt_desc.Height = target_extent;
        rt_desc.DepthOrArraySize = 1;
        rt_desc.MipLevels = 1;
        rt_desc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
        rt_desc.SampleDesc.Count = 1;
        rt_desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &rt_desc,
                                                         D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&target.p))));
        std::uint8_t* up = nullptr;
        EXPECT(SUCCEEDED(upload->Map(0, nullptr, reinterpret_cast<void**>(&up))));
        std::memcpy(up, vertices, sizeof(vertices));
        std::memset(up + upload_indices, 0xee, index_bytes);  // an index read from a gap is far out of the V#
        std::memcpy(up + upload_indices, indices16, sizeof(indices16));
        std::memcpy(up + upload_indices + indices32_offset, indices32, sizeof(indices32));

        // One V# over the sixteen vertices.
        const D3D12_DESCRIPTOR_HEAP_DESC srv_heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1,
                                                       D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
        Ref<ID3D12DescriptorHeap> srv_heap;
        EXPECT(SUCCEEDED(device->CreateDescriptorHeap(&srv_heap_desc, IID_PPV_ARGS(&srv_heap.p))));
        const std::uint64_t cpu = reinterpret_cast<XboxHeapHandleFn>(slot(srv_heap.p, 9))(srv_heap.p);
        const std::uint64_t gpu = reinterpret_cast<XboxHeapHandleFn>(slot(srv_heap.p, 10))(srv_heap.p);
        EXPECT(cpu && gpu >= 0x10000 && gpu < (1ull << 32));
        const std::uint64_t buffer_va = vertex_buffer->GetGPUVirtualAddress();
        const std::uint64_t index_va = index_buffer->GetGPUVirtualAddress();
        const std::uint64_t upload_index_va = upload_index_buffer->GetGPUVirtualAddress();
        EXPECT(buffer_va >= 0x10000 && index_va >= 0x10000 && index_va % 4 == 0);
        EXPECT(upload_index_va && upload_index_va % 4 == 0);
        leonida::gfx10::Buffer vsharp;
        vsharp.address = buffer_va;
        vsharp.stride = 32;
        vsharp.num_records = std::uint32_t(std::size(vertices));
        vsharp.format = 77;  // GFX10_FORMAT_32_32_32_32_FLOAT
        vsharp.oob_select = leonida::gfx10::OobSelect::structured;
        const std::array<std::uint32_t, 4> vsharp_words = leonida::gfx10::encode(vsharp);
        std::memcpy(reinterpret_cast<void*>(cpu), vsharp_words.data(), 16);

        const D3D12_DESCRIPTOR_HEAP_DESC heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
        Ref<ID3D12DescriptorHeap> rtv_heap;
        EXPECT(SUCCEEDED(device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&rtv_heap.p))));
        const D3D12_CPU_DESCRIPTOR_HANDLE rtv{
            SIZE_T(reinterpret_cast<XboxHeapHandleFn>(slot(rtv_heap.p, 9))(rtv_heap.p))};
        device->CreateRenderTargetView(target.p, nullptr, rtv);

        D3D12_COMMAND_QUEUE_DESC queue_desc{};
        Ref<ID3D12CommandQueue> queue;
        Ref<ID3D12CommandAllocator> allocator;
        Ref<ID3D12GraphicsCommandList> list;
        Ref<ID3D12Fence> fence;
        EXPECT(SUCCEEDED(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue.p))));
        EXPECT(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p))));
        EXPECT(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, pso.p,
                                                   IID_PPV_ARGS(&list.p))));
        EXPECT(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))));

        // The copies that fill the vertex and index buffers are in the first frame's list, before its draw.
        list->CopyBufferRegion(vertex_buffer.p, 0, upload.p, 0, sizeof(vertices));
        list->CopyBufferRegion(index_buffer.p, 0, upload.p, upload_indices, index_bytes);
        D3D12_RESOURCE_BARRIER barriers[3] = {
            transition(vertex_buffer.p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
            transition(index_buffer.p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_INDEX_BUFFER),
            transition(target.p, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_RENDER_TARGET)};
        list->ResourceBarrier(3, barriers);

        const auto render_frame = [&](UINT64 number, const Frame& frame, std::vector<Pixel>& out) -> int {
            if (number > 1) {
                EXPECT(SUCCEEDED(allocator->Reset()));
                EXPECT(SUCCEEDED(list->Reset(allocator.p, pso.p)));
                auto to_target = transition(target.p, D3D12_RESOURCE_STATE_COPY_SOURCE,
                                            D3D12_RESOURCE_STATE_RENDER_TARGET);
                list->ResourceBarrier(1, &to_target);
            }
            const FLOAT clear[4] = {0.5f, 0.125f, 0.25f, 0.5f};
            list->ClearRenderTargetView(rtv, clear, 0, nullptr);
            list->SetGraphicsRootSignature(root.p);
            ID3D12DescriptorHeap* heaps[] = {srv_heap.p};
            list->SetDescriptorHeaps(1, heaps);
            list->IASetPrimitiveTopology(quad_list);
            const D3D12_VIEWPORT viewport{0, 0, float(target_extent), float(target_extent), 0, 1};
            const D3D12_RECT scissor{0, 0, LONG(target_extent), LONG(target_extent)};
            list->RSSetViewports(1, &viewport);
            list->RSSetScissorRects(1, &scissor);
            list->OMSetRenderTargets(1, &rtv, FALSE, nullptr);
            const float constants[4] = {1, 0, 0, 0};  // each later instance: half the target to the right
            list->SetGraphicsRootDescriptorTable(0, D3D12_GPU_DESCRIPTOR_HANDLE{gpu});
            list->SetGraphicsRoot32BitConstants(1, 4, constants, 0);
            if (frame.index_format == DXGI_FORMAT_UNKNOWN) {
                list->DrawInstanced(frame.count, frame.instances, frame.start, 0);
            } else {
                const std::uint64_t frame_index_va = frame.use_upload_indices ? upload_index_va : index_va;
                const D3D12_INDEX_BUFFER_VIEW view{frame_index_va + frame.view_offset, frame.view_bytes, frame.index_format};
                list->IASetIndexBuffer(&view);
                list->DrawIndexedInstanced(frame.count, frame.instances, frame.start, frame.base_vertex, 0);
            }
            D3D12_RESOURCE_BARRIER to_copy = transition(target.p, D3D12_RESOURCE_STATE_RENDER_TARGET,
                                                        D3D12_RESOURCE_STATE_COPY_SOURCE);
            list->ResourceBarrier(1, &to_copy);
            D3D12_TEXTURE_COPY_LOCATION dst{};
            dst.pResource = readback.p;
            dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
            dst.PlacedFootprint = {0, {DXGI_FORMAT_R32G32B32A32_FLOAT, target_extent, target_extent, 1, pitch}};
            D3D12_TEXTURE_COPY_LOCATION src{};
            src.pResource = target.p;
            src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
            list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
            EXPECT(list->Close() == S_OK);
            ID3D12CommandList* lists[] = {list.p};
            queue->ExecuteCommandLists(1, lists);
            EXPECT(rt::d3d12::last_execute_result(queue.p) == leonida::ReplayResult::success);
            EXPECT(SUCCEEDED(queue->Signal(fence.p, number)));
            EXPECT(SUCCEEDED(fence->SetEventOnCompletion(number, nullptr)));
            void* mapped = nullptr;
            EXPECT(SUCCEEDED(readback->Map(0, nullptr, &mapped)));
            std::memcpy(out.data(), mapped, out.size() * sizeof(Pixel));
            readback->Unmap(0, nullptr);
            return 0;
        };
        for (std::size_t i = 0; i < frame_count; ++i) EXPECT(render_frame(i + 1, frames[i], pixels[i]) == 0);
        for (auto* r : {&vertex_buffer, &index_buffer, &upload_index_buffer, &upload, &readback, &target}) r->reset();
        pso.reset();
        root.reset();
        srv_heap.reset();
        rtv_heap.reset();
        list.reset();
        allocator.reset();
        queue.reset();
        fence.reset();
        EXPECT(rt::d3d12_xs::device_proxy_ref_count(device.p) == 1);
    }

    constexpr unsigned all = target_extent * target_extent;
    bool ok = true;
    for (std::size_t i = 0; i < frame_count; ++i) {
        unsigned exact = 0, reported = 0;
        for (UINT row = 0; row < target_extent; ++row) {
            for (UINT col = 0; col < target_extent; ++col) {
                const Pixel& want = col < target_extent / 2 ? frames[i].left : frames[i].right;
                const Pixel& got = pixels[i][row * target_extent + col];
                if (std::memcmp(&got, &want, sizeof(Pixel)) == 0) {
                    ++exact;
                } else if (reported++ < 2) {
                    std::fprintf(stderr, "frame %zu (%s) pixel (%u, %u): got %g,%g,%g,%g want %g,%g,%g,%g\n", i,
                                 frames[i].name, col, row, got[0], got[1], got[2], got[3], want[0], want[1],
                                 want[2], want[3]);
                }
            }
        }
        std::printf("frame %zu, %s: %u of %u pixels exact\n", i, frames[i].name, exact, all);
        ok = ok && exact == all;
    }
    if (d.validation_errors()) {
        std::fprintf(stderr, "%u validation errors\n", d.validation_errors());
        return 1;
    }
    if (!ok) return 1;
    std::printf("indexed quad lists: %zu frames of %u pixels exact\n", frame_count, all);
    return 0;
}
