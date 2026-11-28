// Positive graphics replay only: the direct and indirect submissions each draw into a fresh offscreen target.
// The quadrant depends on the indexed vertices; red requires firstInstance=1. No title, presentation or display.
// The compound-signature fixture below spends its own offscreen targets the same way.
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

// The same, for a helper that answers false and leaves the failure to its caller.
#define CHECK_IN(x)                                                                                                    \
    do {                                                                                                               \
        if (!(x)) {                                                                                                    \
            std::fprintf(stderr, "indirect pixels line %d: %s\n", __LINE__, #x);                                       \
            return false;                                                                                              \
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
    // Record-only, nothing from these lists is submitted. A compound record is expanded into the state calls and the
    // draw it stands for, and every binding it wrote is reset afterwards (D3D12's state-leakage rule). A record list
    // the CPU cannot read keeps the reset only. The ordinary draws either side expose the two state snapshots.
    for (UINT scenario = 0; !multi && !counted && scenario < 4; ++scenario) {
        Ref<ID3D12CommandAllocator> allocator;
        Ref<ID3D12GraphicsCommandList> list;
        Ref<ID3D12Resource> gpu_args;
        CHECK(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p))));
        CHECK(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, pso.p,
                                                  IID_PPV_ARGS(&list.p))));
        // Scenario 3: the same records in a device-local buffer. The CPU cannot read it and the GPU may have written
        // it, so the layer may not expand it: nothing is drawn and only the written bindings are reset.
        if (scenario == 3) {
            const auto gpu_desc = buffer_desc(2048);
            CHECK(SUCCEEDED(device->CreateCommittedResource(&def_heap, {}, &gpu_desc, D3D12_RESOURCE_STATE_COMMON,
                                                            nullptr, IID_PPV_ARGS(&gpu_args.p))));
        }
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
        list->ExecuteIndirect(compound.p, scenario == 1 ? 0 : 2, scenario == 3 ? gpu_args.p : args.p, compound_at,
                              scenario == 2 ? args.p : nullptr, 0);
        list->DrawInstanced(3, 1, 0, 0);
        CHECK(list->Close() == S_OK);
        std::vector<leonida::Command> recorded;
        // Scenario 0 draws the two readable records; 1 has no record, 2 a GPU count of zero, 3 an unreadable list.
        CHECK(rt::d3d12::recorded_commands(list.p, recorded) && recorded.size() == (scenario == 0 ? 4u : 2u));
        const auto* before = std::get_if<leonida::Draw>(&recorded.front());
        const auto* after = std::get_if<leonida::Draw>(&recorded.back());
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
            // The two expanded records carry the state the argument buffer names: slot 5 and the index buffer come
            // from the record, and its single root constant lands in the second of the list's two words.
            for (std::size_t i = 1; i <= 2; ++i) {
                const auto* record = std::get_if<leonida::Draw>(&recorded[i]);
                const auto& expected = indexed_draw[i - 1];
                CHECK(record && record->indexed && record->count == expected.IndexCountPerInstance &&
                      record->instances == expected.InstanceCount && record->first == expected.StartIndexLocation &&
                      record->base_vertex == expected.BaseVertexLocation &&
                      record->first_instance == expected.StartInstanceLocation);
                CHECK(record->views && record->views->vertex_buffers[5].va == va);
                CHECK(record->index.va == ib.BufferLocation);
                // The record sets the second dword of the pair (DestOffsetIn32BitValues 1): the first keeps the value
                // the list already had.
                CHECK(record->user_data[c0] == 37 && record->user_data[c0 + 1] == 0);
            }
        }
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

// A compound graphics signature: every record writes its own state (two root constants, a root CBV, both vertex
// buffer slots and the index buffer) and then draws. The picture must come from the record, never from what the list
// had bound when ExecuteIndirect was recorded. Two records place the same indexed quad at two clip-space offsets, so a
// layer that applies one record's state to both of them, or none at all, cannot match the same calls made by hand.
int check_compound_indirect_pixels(ID3D12Device* device, ID3D12RootSignature* root,
                                   D3D12_GRAPHICS_PIPELINE_STATE_DESC desc) {
    constexpr UINT side = 64, pitch = side * 4;
    using Image = std::array<std::uint8_t, side * pitch>;
    // Indices 2..7 with base vertex 1 select vertices 7, 8, 9, 9, 8 and 12: the unit square of the top-right quadrant.
    constexpr float xy[][2] = {
        {0, 0}, {-1, 1}, {0, 1}, {-1, 0}, {-1, 0}, {0, 1}, {0, 0}, // the left square, which this record does not draw
        {0, 1}, {1, 1},  {0, 0}, {0, 0},  {1, 1},  {1, 0}          // the right square: the record's own vertices
    };
    // The same vertices half a unit lower, for the buffer the list has bound before the call: an ignored record draws
    // those instead. Only translated, so the winding that culling tests is the one the reference draws with.
    constexpr float xy_decoy[][2] = {
        {0, -0.5f}, {-1, 0.5f}, {0, 0.5f}, {-1, -0.5f}, {-1, -0.5f}, {0, 0.5f}, {0, -0.5f},
        {0, 0.5f},  {1, 0.5f},  {0, -0.5f}, {0, -0.5f}, {1, 0.5f},  {1, -0.5f}
    };
    constexpr std::uint16_t indices[] = {0, 0, 6, 7, 8, 8, 7, 11, 0, 1, 2, 2, 1, 5};
    constexpr std::uint16_t decoy_indices[] = {0, 0, 0, 1, 2, 3, 4, 5};
    constexpr std::uint8_t colors[][4] = {{0, 255, 0, 255}, {255, 0, 0, 255}, {255, 0, 255, 255}};
    constexpr float record_tint[3] = {0, 1, 0}, decoy_tint[3] = {1, 0, 1};  // a root CBV each: green, then magenta
    constexpr float record_offset[2][2] = {{-0.5f, 0}, {0.5f, 0}}, decoy_offset[2] = {0.25f, 0.25f};
    constexpr float blue[]{0, 0, 1, 1};
    constexpr UINT record_count = 2, stride = 96, arguments_at = 512;
    const std::array<D3D12_INPUT_ELEMENT_DESC, 2> layout{
        {{"POSITION", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
         {"COLOR", 0, DXGI_FORMAT_R8G8B8A8_UNORM, 1, 0, D3D12_INPUT_CLASSIFICATION_PER_INSTANCE_DATA, 1}}};
    desc.InputLayout = {layout.data(), UINT(layout.size())};
    Ref<ID3D12PipelineState> pso;
    CHECK(SUCCEEDED(device->CreateGraphicsPipelineState(&desc, IID_PPV_ARGS(&pso.p))));

    const D3D12_HEAP_PROPERTIES upload_heap{D3D12_HEAP_TYPE_UPLOAD}, read_heap{D3D12_HEAP_TYPE_READBACK},
        def_heap{D3D12_HEAP_TYPE_DEFAULT};
    const auto upload_desc = buffer_desc(2048), rb_desc = buffer_desc(sizeof(Image));
    Ref<ID3D12Resource> input, arguments;
    CHECK(SUCCEEDED(device->CreateCommittedResource(&upload_heap, {}, &upload_desc, D3D12_RESOURCE_STATE_GENERIC_READ,
                                                    nullptr, IID_PPV_ARGS(&input.p))));
    CHECK(SUCCEEDED(device->CreateCommittedResource(&upload_heap, {}, &upload_desc, D3D12_RESOURCE_STATE_GENERIC_READ,
                                                    nullptr, IID_PPV_ARGS(&arguments.p))));
    std::uint8_t* data{};
    CHECK(SUCCEEDED(input->Map(0, nullptr, reinterpret_cast<void**>(&data))));
    std::memcpy(data, xy, sizeof(xy));
    std::memcpy(data + 256, indices, sizeof(indices));
    std::memcpy(data + 512, record_tint, sizeof(record_tint));
    std::memcpy(data + 768, decoy_tint, sizeof(decoy_tint));
    std::memcpy(data + 1024, colors, sizeof(colors));
    std::memcpy(data + 1280, xy_decoy, sizeof(xy_decoy));
    std::memcpy(data + 1536, decoy_indices, sizeof(decoy_indices));
    input->Unmap(0, nullptr);
    const auto va = input->GetGPUVirtualAddress();
    const D3D12_VERTEX_BUFFER_VIEW record_views[] = {{va, sizeof(xy), sizeof(xy[0])},
                                                     {va + 1024, sizeof(colors), sizeof(colors[0])}};
    const D3D12_VERTEX_BUFFER_VIEW decoy_views[] = {{va + 1280, sizeof(xy_decoy), sizeof(xy_decoy[0])},
                                                    {va + 1024, sizeof(colors), sizeof(colors[0])}};
    const D3D12_INDEX_BUFFER_VIEW record_index{va + 256, sizeof(indices), DXGI_FORMAT_R16_UINT};
    const D3D12_INDEX_BUFFER_VIEW decoy_index{va + 1536, sizeof(decoy_indices), DXGI_FORMAT_R16_UINT};
    const D3D12_GPU_VIRTUAL_ADDRESS record_cbv = va + 512;
    const D3D12_DRAW_INDEXED_ARGUMENTS draw{6, 1, 2, 1, 0};  // six indices, base vertex 1, instance 0
    CHECK(SUCCEEDED(arguments->Map(0, nullptr, reinterpret_cast<void**>(&data))));
    std::memset(data, 0, 2048);
    for (UINT i = 0; i < record_count; ++i) {
        // One record, packed in the signature's argument order: two constants, the root CBV, both vertex buffer
        // views, the index buffer view, then the draw. ByteStride pads each record to 96.
        std::uint8_t* record = data + arguments_at + i * stride;
        std::memcpy(record, record_offset[i], sizeof(record_offset[i]));
        std::memcpy(record + 8, &record_cbv, sizeof(record_cbv));
        std::memcpy(record + 16, &record_views[0], sizeof(record_views[0]));
        std::memcpy(record + 32, &record_views[1], sizeof(record_views[1]));
        std::memcpy(record + 48, &record_index, sizeof(record_index));
        std::memcpy(record + 64, &draw, sizeof(draw));
    }
    arguments->Unmap(0, nullptr);
    D3D12_INDIRECT_ARGUMENT_DESC members[6]{};
    members[0].Type = D3D12_INDIRECT_ARGUMENT_TYPE_CONSTANT;
    members[0].Constant = {0, 0, 2};
    members[1].Type = D3D12_INDIRECT_ARGUMENT_TYPE_CONSTANT_BUFFER_VIEW;
    members[1].ConstantBufferView.RootParameterIndex = 1;
    members[2].Type = D3D12_INDIRECT_ARGUMENT_TYPE_VERTEX_BUFFER_VIEW;
    members[2].VertexBuffer.Slot = 0;
    members[3].Type = D3D12_INDIRECT_ARGUMENT_TYPE_VERTEX_BUFFER_VIEW;
    members[3].VertexBuffer.Slot = 1;
    members[4].Type = D3D12_INDIRECT_ARGUMENT_TYPE_INDEX_BUFFER_VIEW;
    members[5].Type = D3D12_INDIRECT_ARGUMENT_TYPE_DRAW_INDEXED;
    const D3D12_COMMAND_SIGNATURE_DESC signature_desc{stride, 6, members, 0};
    Ref<ID3D12CommandSignature> compound;
    CHECK(SUCCEEDED(device->CreateCommandSignature(&signature_desc, root, IID_PPV_ARGS(&compound.p))));
    auto kind = leonida::IndirectKind::dispatch;
    UINT byte_stride = 0;
    bool terminal_only = true;
    CHECK(rt::d3d12::command_signature_info(compound.p, kind, byte_stride, terminal_only));
    CHECK(kind == leonida::IndirectKind::draw_indexed && byte_stride == stride && !terminal_only);

    const D3D12_VIEWPORT viewport{0, 0, float(side), float(side), 0, 1};
    const D3D12_RECT scissor{0, 0, side, side};
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
    using XboxHeapHandleFn = std::uint64_t(WINAPI*)(void*);
    // mode 0: the record list through ExecuteIndirect. mode 1: the calls a record stands for. mode 2: those calls
    // with only the state the list had bound before the indirect call.
    const auto render = [&](UINT mode, Image& image) -> bool {
        Ref<ID3D12Resource> target, readback;
        Ref<ID3D12DescriptorHeap> heap;
        Ref<ID3D12CommandAllocator> allocator;
        Ref<ID3D12GraphicsCommandList> list;
        Ref<ID3D12Fence> fence;
        CHECK_IN(SUCCEEDED(device->CreateCommittedResource(&def_heap, {}, &tex, D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                           IID_PPV_ARGS(&target.p))));
        CHECK_IN(SUCCEEDED(device->CreateCommittedResource(&read_heap, {}, &rb_desc, D3D12_RESOURCE_STATE_COPY_DEST,
                                                           nullptr, IID_PPV_ARGS(&readback.p))));
        CHECK_IN(SUCCEEDED(device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&heap.p))));
        const D3D12_CPU_DESCRIPTOR_HANDLE rtv{
            SIZE_T(reinterpret_cast<XboxHeapHandleFn>(slot(heap.p, 9))(heap.p))};
        device->CreateRenderTargetView(target.p, nullptr, rtv);
        CHECK_IN(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p))));
        CHECK_IN(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, pso.p,
                                                     IID_PPV_ARGS(&list.p))));
        CHECK_IN(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))));
        auto b = transition(target.p, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_RENDER_TARGET);
        list->ResourceBarrier(1, &b);
        list->ClearRenderTargetView(rtv, blue, 0, nullptr);
        list->SetGraphicsRootSignature(root);
        list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        list->RSSetViewports(1, &viewport);
        list->RSSetScissorRects(1, &scissor);
        list->OMSetRenderTargets(1, &rtv, FALSE, nullptr);
        // What the list has bound before the call: a mirrored vertex buffer, another index buffer and the other root
        // arguments. No record may draw with any of it.
        list->IASetVertexBuffers(0, 2, decoy_views);
        list->IASetIndexBuffer(&decoy_index);
        list->SetGraphicsRoot32BitConstants(0, 2, decoy_offset, 0);
        list->SetGraphicsRootConstantBufferView(1, va + 768);
        if (mode == 0) {
            list->ExecuteIndirect(compound.p, record_count, arguments.p, arguments_at, nullptr, 0);
        } else {
            for (UINT i = 0; i < record_count; ++i) {
                if (mode == 1) {
                    list->IASetVertexBuffers(0, 2, record_views);
                    list->IASetIndexBuffer(&record_index);
                    list->SetGraphicsRoot32BitConstants(0, 2, record_offset[i], 0);
                    list->SetGraphicsRootConstantBufferView(1, record_cbv);
                }
                list->DrawIndexedInstanced(draw.IndexCountPerInstance, draw.InstanceCount, draw.StartIndexLocation,
                                           draw.BaseVertexLocation, draw.StartInstanceLocation);
            }
        }
        b = transition(target.p, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE);
        list->ResourceBarrier(1, &b);
        D3D12_TEXTURE_COPY_LOCATION dst{}, src{};
        dst.pResource = readback.p;
        dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        dst.PlacedFootprint = {0, {tex.Format, side, side, 1, pitch}};
        src.pResource = target.p;
        src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
        CHECK_IN(list->Close() == S_OK);
        ID3D12CommandList* lists[]{list.p};
        queue->ExecuteCommandLists(1, lists);
        CHECK_IN(rt::d3d12::last_execute_result(queue.p) == leonida::ReplayResult::success);
        CHECK_IN(SUCCEEDED(queue->Signal(fence.p, 1)));
        CHECK_IN(SUCCEEDED(fence->SetEventOnCompletion(1, nullptr)));
        void* mapped{};
        CHECK_IN(SUCCEEDED(readback->Map(0, nullptr, &mapped)));
        std::memcpy(image.data(), mapped, image.size());
        readback->Unmap(0, nullptr);
        return true;
    };
    Image indirect_image{}, reference{}, decoy_image{};
    CHECK(render(0, indirect_image));
    CHECK(render(1, reference));
    CHECK(render(2, decoy_image));
    constexpr std::uint8_t clear[4] = {0, 0, 255, 255};
    const auto lit = [&clear](const Image& image) {
        unsigned count = 0;
        for (std::size_t i = 0; i < image.size(); i += 4) count += std::memcmp(image.data() + i, clear, 4) != 0;
        return count;
    };
    unsigned wrong = 0;
    for (std::size_t i = 0; i < reference.size(); i += 4)
        wrong += std::memcmp(reference.data() + i, indirect_image.data() + i, 4) != 0;
    std::printf("compound DRAW_INDEXED x%u (two vertex buffer views, the index buffer, two root constants and a root "
                "CBV per record): %u lit pixels, %u/4096 RGBA pixel mismatches against the same calls made by hand; "
                "the indirect list alone lights %u and the state the list had bound alone lights %u\n",
                record_count, lit(reference), wrong, lit(indirect_image), lit(decoy_image));
    // The record list must draw (a layer that only resets leaves the clear colour), must not use what the list had
    // bound, and must match the calls it stands for pixel for pixel.
    CHECK(lit(reference) > 0 && lit(decoy_image) > 0);
    CHECK(!wrong);
    CHECK(indirect_image != decoy_image);
    return 0;
}

