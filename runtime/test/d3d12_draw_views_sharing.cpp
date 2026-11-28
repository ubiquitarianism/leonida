// The views block a list records with each draw (leonida::DrawViews: 8 render targets and 32 vertex buffer views,
// about 700 bytes, kept until the list's Reset) is shared by the draws that keep the same render targets and vertex
// buffers. A title that binds the same vertex buffer, or the same render targets, again before every draw gets one
// block, not one per draw. A changed vertex buffer slot, a different render target, or a new descriptor behind the
// same RTV handle still starts a new block that holds the new views. Recording only: nothing is submitted.
// Run by d3d12_draw_test --draw-views-sharing, on its device, root signature and PSO description. No game data.
#include "d3d12_test_shaders.h"
#include "../shims/d3d12_com.h"

#include <cstdio>
#include <set>
#include <variant>
#include <vector>

using namespace d3d12test;

#define CHECK(x)                                                                                                       \
    do {                                                                                                               \
        if (!(x)) {                                                                                                    \
            std::fprintf(stderr, "draw views line %d: %s\n", __LINE__, #x);                                            \
            return 1;                                                                                                  \
        }                                                                                                              \
    } while (0)

int check_draw_views_sharing(ID3D12Device* device, ID3D12RootSignature* root,
                             const D3D12_GRAPHICS_PIPELINE_STATE_DESC& desc) {
    Ref<ID3D12PipelineState> pso;
    CHECK(SUCCEEDED(device->CreateGraphicsPipelineState(&desc, IID_PPV_ARGS(&pso.p))));
    const D3D12_HEAP_PROPERTIES upload_heap{D3D12_HEAP_TYPE_UPLOAD}, default_heap{D3D12_HEAP_TYPE_DEFAULT};
    const auto upload_desc = buffer_desc(2048);
    Ref<ID3D12Resource> vertices;
    CHECK(SUCCEEDED(device->CreateCommittedResource(&upload_heap, {}, &upload_desc, D3D12_RESOURCE_STATE_GENERIC_READ,
                                                    nullptr, IID_PPV_ARGS(&vertices.p))));
    const auto va = vertices->GetGPUVirtualAddress();
    const D3D12_VERTEX_BUFFER_VIEW first{va, 1024, 12}, second{va + 1024, 1024, 12};

    // Two render targets, each named by the RTV of a heap of its own.
    D3D12_RESOURCE_DESC tex{};
    tex.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    tex.Width = tex.Height = 64;
    tex.DepthOrArraySize = tex.MipLevels = 1;
    tex.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    tex.SampleDesc.Count = 1;
    tex.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    const D3D12_DESCRIPTOR_HEAP_DESC heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1};
    using XboxHeapHandleFn = std::uint64_t(WINAPI*)(void*);  // heap slot 9: the CPU handle in RAX
    Ref<ID3D12Resource> targets[2];
    Ref<ID3D12DescriptorHeap> heaps[2];
    D3D12_CPU_DESCRIPTOR_HANDLE rtvs[2]{};
    for (int i = 0; i < 2; ++i) {
        CHECK(SUCCEEDED(device->CreateCommittedResource(&default_heap, {}, &tex, D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                        IID_PPV_ARGS(&targets[i].p))));
        CHECK(SUCCEEDED(device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&heaps[i].p))));
        rtvs[i].ptr = SIZE_T(reinterpret_cast<XboxHeapHandleFn>(slot(heaps[i].p, 9))(heaps[i].p));
        device->CreateRenderTargetView(targets[i].p, nullptr, rtvs[i]);
    }
    const D3D12_VIEWPORT viewport{0, 0, 64, 64, 0, 1};

    Ref<ID3D12CommandAllocator> allocator;
    Ref<ID3D12GraphicsCommandList> list;
    CHECK(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p))));
    CHECK(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, pso.p,
                                              IID_PPV_ARGS(&list.p))));
    list->SetGraphicsRootSignature(root);
    list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    list->RSSetViewports(1, &viewport);
    list->OMSetRenderTargets(1, &rtvs[0], FALSE, nullptr);
    constexpr std::size_t repeats = 1000;
    for (std::size_t i = 0; i < repeats; ++i) {
        list->IASetVertexBuffers(0, 1, &first);  // the view slot 0 already holds after the first pass
        list->DrawInstanced(3, 1, 0, 0);
    }
    list->IASetVertexBuffers(0, 1, &second);  // [0] slot 0 changes
    list->DrawInstanced(3, 1, 0, 0);
    list->IASetVertexBuffers(0, 0, nullptr);  // [1] no slot written, and the same render target bound again
    list->OMSetRenderTargets(1, &rtvs[0], FALSE, nullptr);
    list->DrawInstanced(3, 1, 0, 0);
    list->IASetVertexBuffers(1, 1, &first);  // [2] another slot is bound
    list->DrawInstanced(3, 1, 0, 0);
    list->IASetVertexBuffers(1, 1, nullptr);  // [3] and unbound again
    list->DrawInstanced(3, 1, 0, 0);
    list->OMSetRenderTargets(1, &rtvs[1], FALSE, nullptr);  // [4] another render target
    list->DrawInstanced(3, 1, 0, 0);
    // [5] The same handle, whose descriptor now names the first target. D3D12 reads the descriptor at
    // OMSetRenderTargets, so this is a change as well.
    device->CreateRenderTargetView(targets[0].p, nullptr, rtvs[1]);
    list->OMSetRenderTargets(1, &rtvs[1], FALSE, nullptr);
    list->DrawInstanced(3, 1, 0, 0);
    CHECK(list->Close() == S_OK);

    std::vector<leonida::Command> recorded;
    CHECK(rt::d3d12::recorded_commands(list.p, recorded) && recorded.size() == repeats + 6);
    std::vector<const leonida::DrawViews*> views;
    for (const auto& command : recorded) {
        const auto* draw = std::get_if<leonida::Draw>(&command);
        CHECK(draw && draw->views);
        views.push_back(draw->views.get());
    }
    const std::set<const leonida::DrawViews*> repeated(views.begin(), views.begin() + repeats);
    if (repeated.size() != 1)
        std::fprintf(stderr, "draw views: %zu blocks for %zu draws that bind the same vertex buffer again\n",
                     repeated.size(), repeats);
    CHECK(repeated.size() == 1);

    const leonida::DrawViews& base = *views[0];
    CHECK(base.render_target_count == 1 && base.render_targets[0].resource);
    CHECK(base.vertex_buffers[0].va == va && base.vertex_buffers[0].size == 1024 &&
          base.vertex_buffers[0].stride == 12);
    CHECK(!base.vertex_buffers[1].va);
    const leonida::DrawViews* const* changed = views.data() + repeats;
    CHECK(changed[0] != &base && changed[0]->vertex_buffers[0].va == va + 1024);
    CHECK(changed[0]->render_targets[0].resource == base.render_targets[0].resource);
    CHECK(changed[1] == changed[0]);
    CHECK(changed[2] != changed[1] && changed[2]->vertex_buffers[1].va == va &&
          changed[2]->vertex_buffers[0].va == va + 1024);
    CHECK(changed[3] != changed[2] && !changed[3]->vertex_buffers[1].va &&
          changed[3]->vertex_buffers[0].va == va + 1024);
    CHECK(changed[4] != changed[3] && changed[4]->render_targets[0].resource &&
          changed[4]->render_targets[0].resource != base.render_targets[0].resource);
    CHECK(changed[4]->render_target_count == 1 && changed[4]->vertex_buffers[0].va == va + 1024);
    CHECK(changed[5] != changed[4] && changed[5]->render_targets[0].resource == base.render_targets[0].resource);
    const std::set<const leonida::DrawViews*> all(views.begin(), views.end());
    CHECK(all.size() == 6);
    std::printf("draw views: %zu draws that bind the same vertex buffer again share 1 block; a changed slot, render "
                "target or RTV descriptor starts a new one (%zu blocks for %zu draws)\n",
                repeats, all.size(), views.size());
    return 0;
}
