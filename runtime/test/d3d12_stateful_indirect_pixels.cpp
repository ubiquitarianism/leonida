// Positive graphics replay only: the direct and indirect submissions each draw into a fresh offscreen target.
// The quadrant depends on the indexed vertices; red requires firstInstance=1. No title, presentation or display.
#include "d3d12_test_shaders.h"
#include "../shims/d3d12_com.h"

#include <array>
#include <cstdio>
#include <cstring>

using namespace d3d12test;

#define CHECK(x)                                                                                                       \
    do {                                                                                                               \
        if (!(x)) {                                                                                                    \
            std::fprintf(stderr, "indirect pixels line %d: %s\n", __LINE__, #x);                                       \
            return 1;                                                                                                  \
        }                                                                                                              \
    } while (0)

int check_stateful_indirect_pixels(ID3D12Device* device, ID3D12RootSignature* root,
                                   D3D12_GRAPHICS_PIPELINE_STATE_DESC desc, bool multi, bool counted, bool nonindexed_only) {
    constexpr UINT side = 64, pitch = side * 4;
    using Image = std::array<std::uint8_t, side * pitch>;
    constexpr float xy[][2] = {
        {0, 0}, {-1, 1}, {0, 1}, {-1, 0}, {-1, 0}, {0, 1}, {0, 0}, // DRAW: left quad, StartVertexLocation=1
        {0, 1}, {1, 1},  {0, 0}, {0, 0},  {1, 1},  {1, 0}          // second DRAW: right quad
    };
    constexpr std::uint16_t indices[] = {0, 0, 6, 7, 8, 8, 7, 11, 0, 1, 2, 2, 1, 5}; // right, then left (base vertex=1)
    constexpr std::uint8_t colors[][4] = {{0, 255, 0, 255}, {255, 0, 0, 255}, {255, 0, 255, 255}};
    constexpr float red_tint[3] = {1, 0, 1}, other_tint[3] = {0, 1, 0};
    // The third record is always valid but beyond maxCount=2. If it executed, magenta would overwrite the first red
    // quad.
    constexpr D3D12_DRAW_ARGUMENTS draw[] = {{6, 1, 1, 1}, {6, 1, 7, 2}, {6, 1, 1, 2}};
    constexpr D3D12_DRAW_INDEXED_ARGUMENTS indexed_draw[] = {{6, 1, 2, 1, 1}, {6, 1, 8, 1, 2}, {6, 1, 2, 1, 2}};
    constexpr UINT args_at = 1280, indexed_at = 1536, padded_stride = 32;
    constexpr UINT count_at[] = {24, 44, 92, 108};
    // Valid decoy at offset zero lives in a separate argument resource. It draws nothing if offset handling is lost.
    const std::array<D3D12_INPUT_ELEMENT_DESC, 2> layout{
        {{"POSITION", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
         {"COLOR", 0, DXGI_FORMAT_R8G8B8A8_UNORM, 1, 0, D3D12_INPUT_CLASSIFICATION_PER_INSTANCE_DATA, 1}}};
    desc.InputLayout = {layout.data(), UINT(layout.size())};
    Ref<ID3D12PipelineState> pso;
    CHECK(SUCCEEDED(device->CreateGraphicsPipelineState(&desc, IID_PPV_ARGS(&pso.p))));

    const D3D12_HEAP_PROPERTIES upload_heap{D3D12_HEAP_TYPE_UPLOAD}, read_heap{D3D12_HEAP_TYPE_READBACK},
        def_heap{D3D12_HEAP_TYPE_DEFAULT};
    const auto upload_desc = buffer_desc(2048), rb_desc = buffer_desc(sizeof(Image));
    Ref<ID3D12Resource> input, args, counts;
    CHECK(SUCCEEDED(device->CreateCommittedResource(&upload_heap, {}, &upload_desc, D3D12_RESOURCE_STATE_GENERIC_READ,
                                                    nullptr, IID_PPV_ARGS(&input.p))));
    CHECK(SUCCEEDED(device->CreateCommittedResource(&upload_heap, {}, &upload_desc, D3D12_RESOURCE_STATE_GENERIC_READ,
                                                    nullptr, IID_PPV_ARGS(&args.p))));
    std::uint8_t* data{};
    CHECK(SUCCEEDED(input->Map(0, nullptr, reinterpret_cast<void**>(&data))));
    std::memcpy(data, xy, sizeof(xy));
    std::memcpy(data + 256, indices, sizeof(indices));
    std::memcpy(data + 512, red_tint, sizeof(red_tint));
    std::memcpy(data + 768, other_tint, sizeof(other_tint));
    std::memcpy(data + 1024, colors, sizeof(colors));
    input->Unmap(0, nullptr);
    const auto va = input->GetGPUVirtualAddress();
    const D3D12_VERTEX_BUFFER_VIEW vbs[] = {{va, sizeof(xy), sizeof(xy[0])},
                                            {va + 1024, sizeof(colors), sizeof(colors[0])}};
    const D3D12_INDEX_BUFFER_VIEW ib{va + 256, sizeof(indices), DXGI_FORMAT_R16_UINT};
    constexpr UINT compound_stride = 80, compound_size = 64, compound_at = 2048 - compound_stride - compound_size;
    CHECK(SUCCEEDED(args->Map(0, nullptr, reinterpret_cast<void**>(&data))));
    std::memset(data, 0, 2048);
    for (UINT i = 0; i < 3; ++i) {
        std::memcpy(data + args_at + padded_stride * i, &draw[i], sizeof(draw[i]));
        std::memcpy(data + indexed_at + padded_stride * i, &indexed_draw[i], sizeof(indexed_draw[i]));
    }
    const D3D12_GPU_VIRTUAL_ADDRESS cb = va + 512;
    for (UINT i = 0; i < 2; ++i) {
        auto* rec = data + compound_at + i * compound_stride;  // first DWORD is the already-zero root constant
        std::memcpy(rec + 4, &cb, sizeof(cb));
        std::memcpy(rec + 12, &vbs[0], sizeof(vbs[0]));
        std::memcpy(rec + 28, &ib, sizeof(ib));
        std::memcpy(rec + 44, &indexed_draw[i], sizeof(indexed_draw[i]));
    }
    args->Unmap(0, nullptr);
    if (counted) {
        const auto count_desc = buffer_desc(128);
        CHECK(SUCCEEDED(device->CreateCommittedResource(
            &upload_heap, {}, &count_desc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&counts.p))));
        CHECK(SUCCEEDED(counts->Map(0, nullptr, reinterpret_cast<void**>(&data))));
        std::memset(data, 0, 128);
        const UINT wrong_offset = 2;
        std::memcpy(data, &wrong_offset, sizeof(wrong_offset));
        for (UINT n = 0; n <= 3; ++n)
            std::memcpy(data + count_at[n], &n, sizeof(n));
        counts->Unmap(0, nullptr);
    }
    const D3D12_VIEWPORT viewport{0, 0, float(side), float(side), 0, 1};
    const D3D12_RECT clip{8, 4, 56, 40}, later_clip{0, 0, 1, 1};
    const float blue[]{0, 0, 1, 1}, offset[]{0, -0.5f}, later_offset[]{0, 2};
    D3D12_RESOURCE_DESC tex{};
    tex.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    tex.Width = tex.Height = side;
    tex.DepthOrArraySize = tex.MipLevels = 1;
    tex.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    tex.SampleDesc.Count = 1;
    tex.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    const D3D12_DESCRIPTOR_HEAP_DESC heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1};
    const D3D12_COMMAND_QUEUE_DESC queue_desc{};
    Ref<ID3D12CommandQueue> queue;
    CHECK(SUCCEEDED(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue.p))));
    // Identical padded byte stride and terminal kind do not make a compound signature a simple draw.
    // Create and inspect both signatures; the compound is deliberately never recorded or executed.
    {
        D3D12_INDIRECT_ARGUMENT_DESC members[2]{};
        members[0].Type = D3D12_INDIRECT_ARGUMENT_TYPE_VERTEX_BUFFER_VIEW;
        members[0].VertexBuffer.Slot = 0;
        members[1].Type = D3D12_INDIRECT_ARGUMENT_TYPE_DRAW;
        for (const bool compound : {false, true}) {
            const D3D12_COMMAND_SIGNATURE_DESC sd{48, compound ? 2u : 1u, compound ? members : members + 1, 0};
            Ref<ID3D12CommandSignature> sig;
            CHECK(SUCCEEDED(device->CreateCommandSignature(&sd, nullptr, IID_PPV_ARGS(&sig.p))));
            auto kind = leonida::IndirectKind::dispatch;
            UINT stride = 0;
            bool terminal_only = compound;
            CHECK(rt::d3d12::command_signature_info(sig.p, kind, stride, terminal_only));
            CHECK(kind == leonida::IndirectKind::draw && stride == 48 && terminal_only == !compound);
        }
    }
    // Record-only: an unsupported compound draw still resets its written bindings. Nothing from
    // this list is submitted. The ordinary draws either side expose the two immutable state snapshots.
    for (UINT scenario = 0; !multi && !counted && scenario < 3; ++scenario) {
        Ref<ID3D12CommandAllocator> allocator;
        Ref<ID3D12GraphicsCommandList> list;
        CHECK(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p))));
        CHECK(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, pso.p,
                                                  IID_PPV_ARGS(&list.p))));
        Ref<ID3D12CommandSignature> compound;
        {
            D3D12_INDIRECT_ARGUMENT_DESC members[5]{};
            members[0].Type = D3D12_INDIRECT_ARGUMENT_TYPE_CONSTANT;
            members[0].Constant = {0, 1, 1};
            members[1].Type = D3D12_INDIRECT_ARGUMENT_TYPE_CONSTANT_BUFFER_VIEW;
            members[1].ConstantBufferView.RootParameterIndex = 1;
            members[2].Type = D3D12_INDIRECT_ARGUMENT_TYPE_VERTEX_BUFFER_VIEW;
            members[2].VertexBuffer.Slot = 5;
            members[3].Type = D3D12_INDIRECT_ARGUMENT_TYPE_INDEX_BUFFER_VIEW;
            members[4].Type = D3D12_INDIRECT_ARGUMENT_TYPE_DRAW_INDEXED;
            const D3D12_COMMAND_SIGNATURE_DESC sd{compound_stride, 5, members, 0};
            CHECK(SUCCEEDED(device->CreateCommandSignature(&sd, root, IID_PPV_ARGS(&compound.p))));
        }
        const UINT words[] = {37, 73};
        list->SetGraphicsRootSignature(root);
        list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        list->IASetVertexBuffers(0, 2, vbs);
        list->IASetVertexBuffers(5, 1, vbs);
        list->IASetIndexBuffer(&ib);
        list->RSSetViewports(1, &viewport);
        list->SetGraphicsRoot32BitConstants(0, 2, words, 0);
        list->SetGraphicsRootConstantBufferView(1, va + 512);
        list->DrawInstanced(3, 1, 0, 0);
        // The two packed records fit exactly, without trailing padding. Fixed zero and GPU count zero
        // have the same postcondition. The count DWORD at offset zero is separate from the records.
        list->ExecuteIndirect(compound.p, scenario == 1 ? 0 : 2, args.p, compound_at,
                              scenario == 2 ? args.p : nullptr, 0);
        list->DrawInstanced(3, 1, 0, 0);
        CHECK(list->Close() == S_OK);
        std::vector<leonida::Command> recorded;
        CHECK(rt::d3d12::recorded_commands(list.p, recorded) && recorded.size() == 2);
        const auto* before = std::get_if<leonida::Draw>(&recorded[0]);
        const auto* after = std::get_if<leonida::Draw>(&recorded[1]);
        CHECK(before && after && before->views && after->views && before->views != after->views);
        CHECK(before->views->vertex_buffers[5].va == va && after->views->vertex_buffers[5].va == 0);
        CHECK(after->views->vertex_buffers[0].va == va && after->views->vertex_buffers[1].va == va + 1024);
        CHECK(before->index.va == ib.BufferLocation && after->index.va == 0 && after->index.size == 0);
        // These are packed shader words, not D3D root-parameter indices. The graphics packer can reserve a prefix.
        const auto* packed = rt::d3d12::root_signature_layout(root);
        CHECK(packed && packed->parameters.size() == 2);
        const auto c0 = packed->parameters[0].user_offset, view = packed->parameters[1].user_offset;
        CHECK(c0 < before->user_data.size() - 1 && view < before->user_data.size() - 1);
        CHECK(before->user_data[c0] == 37 && before->user_data[c0 + 1] == 73);
        CHECK(before->user_data[view] || before->user_data[view + 1]);
        auto expected_words = before->user_data;
        expected_words[c0 + 1] = expected_words[view] = expected_words[view + 1] = 0;
        CHECK(after->user_data == expected_words);
        if (scenario == 0) {
            // Recording check only, never submitted: omit the last DWORD of the compound record.
            CHECK(SUCCEEDED(list->Reset(allocator.p, pso.p)));
            list->SetGraphicsRootSignature(root);
            list->ExecuteIndirect(compound.p, 2, args.p, compound_at + 4, nullptr, 0);
            CHECK(list->Close() == E_INVALIDARG);
        }
    }
    for (const bool indexed : {false, true}) {
        D3D12_INDIRECT_ARGUMENT_DESC argument{};
        if (nonindexed_only && indexed) continue;
        argument.Type = indexed ? D3D12_INDIRECT_ARGUMENT_TYPE_DRAW_INDEXED : D3D12_INDIRECT_ARGUMENT_TYPE_DRAW;
        const UINT stride = multi ? padded_stride : indexed ? UINT(sizeof(indexed_draw[0])) : UINT(sizeof(draw[0]));
        const D3D12_COMMAND_SIGNATURE_DESC sd{stride, 1, &argument, 0};
        Ref<ID3D12CommandSignature> sig;
        CHECK(SUCCEEDED(device->CreateCommandSignature(&sd, nullptr, IID_PPV_ARGS(&sig.p))));
        // Uncounted covers maxCount=1 or 2; counted clamps each execution to the GPU count at its own byte offset.
        const UINT max_records = multi ? 2u : 1u;
        for (UINT raw_count = counted ? 0u : max_records; raw_count <= (counted ? 3u : max_records); ++raw_count) {
            const UINT records = raw_count < max_records ? raw_count : max_records;
            Image reference{};
            for (const bool indirect : {false, true}) {
                Ref<ID3D12Resource> target, readback, copied_args, copied_counts;
                Ref<ID3D12DescriptorHeap> heap;
                Ref<ID3D12CommandAllocator> allocator;
                Ref<ID3D12GraphicsCommandList> list;
                Ref<ID3D12Fence> fence;
                CHECK(SUCCEEDED(device->CreateCommittedResource(&def_heap, {}, &tex, D3D12_RESOURCE_STATE_COMMON,
                                                                nullptr, IID_PPV_ARGS(&target.p))));
                CHECK(SUCCEEDED(device->CreateCommittedResource(
                    &read_heap, {}, &rb_desc, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&readback.p))));
                CHECK(SUCCEEDED(device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&heap.p))));
                using XboxHeapHandleFn = std::uint64_t(WINAPI*)(void*);
                const D3D12_CPU_DESCRIPTOR_HANDLE rtv{
                    SIZE_T(reinterpret_cast<XboxHeapHandleFn>(slot(heap.p, 9))(heap.p))};
                device->CreateRenderTargetView(target.p, nullptr, rtv);
                CHECK(SUCCEEDED(
                    device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p))));
                CHECK(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, pso.p,
                                                          IID_PPV_ARGS(&list.p))));
                CHECK(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))));
                if (indirect && counted) {
                    // ExecuteIndirect sees only fresh, never-CPU-mapped DEFAULT resources. Their argument and count
                    // bytes become available when these copies execute, *after* CPU command recording.
                    const auto count_desc = counts->GetDesc();
                    CHECK(SUCCEEDED(device->CreateCommittedResource(&def_heap, {}, &upload_desc,
                                                                    D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                                    IID_PPV_ARGS(&copied_args.p))));
                    CHECK(SUCCEEDED(device->CreateCommittedResource(&def_heap, {}, &count_desc,
                                                                    D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                                    IID_PPV_ARGS(&copied_counts.p))));
                    list->CopyBufferRegion(copied_args.p, 0, args.p, 0, upload_desc.Width);
                    list->CopyBufferRegion(copied_counts.p, 0, counts.p, 0, count_desc.Width);
                    const D3D12_RESOURCE_BARRIER ready[] = {transition(copied_args.p, D3D12_RESOURCE_STATE_COPY_DEST,
                                                                       D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT),
                                                            transition(copied_counts.p, D3D12_RESOURCE_STATE_COPY_DEST,
                                                                       D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT)};
                    list->ResourceBarrier(2, ready);
                }
                auto b = transition(target.p, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_RENDER_TARGET);
                list->ResourceBarrier(1, &b);
                list->ClearRenderTargetView(rtv, blue, 0, nullptr);
                list->SetGraphicsRootSignature(root);
                list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
                list->IASetVertexBuffers(0, 2, vbs);
                if (indexed) list->IASetIndexBuffer(&ib);
                list->RSSetViewports(1, &viewport);
                list->RSSetScissorRects(1, &clip);
                list->OMSetRenderTargets(1, &rtv, FALSE, nullptr);
                list->SetGraphicsRoot32BitConstants(0, 2, offset, 0);
                list->SetGraphicsRootConstantBufferView(1, va + 512);
                if (indirect) {
                    list->ExecuteIndirect(sig.p, max_records, counted ? copied_args.p : args.p,
                                          indexed ? indexed_at : args_at, counted ? copied_counts.p : nullptr,
                                          counted ? count_at[raw_count] : 0);
                } else {
                    for (UINT i = 0; i < records; ++i) {
                        if (indexed) {
                            const auto& cmd = indexed_draw[i];
                            list->DrawIndexedInstanced(cmd.IndexCountPerInstance, cmd.InstanceCount,
                                                       cmd.StartIndexLocation, cmd.BaseVertexLocation,
                                                       cmd.StartInstanceLocation);
                        } else {
                            const auto& cmd = draw[i];
                            list->DrawInstanced(cmd.VertexCountPerInstance, cmd.InstanceCount, cmd.StartVertexLocation,
                                                cmd.StartInstanceLocation);
                        }
                    }
                }
                // Legal state changes, not another draw. Taking a late reference to list state instead of a snapshot
                // would miss the nonempty region or change its tint.
                list->RSSetScissorRects(1, &later_clip);
                list->SetGraphicsRoot32BitConstants(0, 2, later_offset, 0);
                list->SetGraphicsRootConstantBufferView(1, va + 768);

                b = transition(target.p, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE);
                list->ResourceBarrier(1, &b);
                D3D12_TEXTURE_COPY_LOCATION dst{}, src{};
                dst.pResource = readback.p;
                dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
                dst.PlacedFootprint = {0, {tex.Format, side, side, 1, pitch}};
                src.pResource = target.p;
                src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
                list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
                CHECK(list->Close() == S_OK);
                ID3D12CommandList* lists[]{list.p};
                queue->ExecuteCommandLists(1, lists);
                CHECK(rt::d3d12::last_execute_result(queue.p) == leonida::ReplayResult::success);
                CHECK(SUCCEEDED(queue->Signal(fence.p, 1)));
                CHECK(SUCCEEDED(fence->SetEventOnCompletion(1, nullptr)));
                void* mapped{};
                CHECK(SUCCEEDED(readback->Map(0, nullptr, &mapped)));
                Image image{};
                std::memcpy(image.data(), mapped, image.size());
                readback->Unmap(0, nullptr);
                unsigned wrong = 0, red = 0, magenta = 0;
                for (UINT y = 0; y < side; ++y) {
                    for (UINT x = 0; x < side; ++x) {
                        const bool left = y >= 16 && y < 40 && x >= 8 && x < 32;
                        const bool right = y >= 16 && y < 40 && x >= 32 && x < 56;
                        const bool first = records >= 1 && (indexed ? right : left);
                        const bool second = records == 2 && (indexed ? left : right);
                        const std::array<std::uint8_t, 4> expected{std::uint8_t(first || second ? 255 : 0), 0,
                                                                   std::uint8_t(first ? 0 : 255), 255};
                        const bool equal = !std::memcmp(expected.data(), image.data() + y * pitch + 4 * x, 4);
                        wrong += !equal;
                        red += equal && first;
                        magenta += equal && second;
                    }
                }
                const bool same = !indirect || image == reference;
                std::printf("%s %s %s (input=%u, max=%u): %u red / %u magenta, %u/4096 RGBA mismatches%s\n",
                            indexed ? "DRAW_INDEXED" : "DRAW", indirect ? "indirect" : "direct",
                            counted ? "count-buffer" : "uncounted", raw_count, max_records, red, magenta, wrong,
                            indirect ? (same ? ", direct image identical" : ", direct image differs") : "");
                CHECK(!wrong && same);
                if (!indirect) reference = image;
            }
        }
    }
    return 0;
}
