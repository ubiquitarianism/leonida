// First light with a texture (TEST-524, docs/first_light.md): the real model, the game's VS and a game pixel shader
// that samples one texture, drawn through the Xbox-ABI device proxy over the Vulkan device. Package-gated like
// d3d12_first_light_test: everything is read at run time from LEONIDA_FIRST_LIGHT_TEX_DIR (a tools/first_light.py
// output folder made with --texture), and the test prints SKIP and exits 77 when that is unset.
//   - The PS s_loads its T# through a 32-bit table pointer in user SGPR s8 and builds its S# from literals. The root
//     signature is [0] eight 32-bit constants (s0..s7, zero) and [1] a one-SRV descriptor table, which the layer's
//     packing (docs/d3d12_vulkan.md 3.5) puts at s8:s9. The literal S# is registered by the layer from the index.
//   - texture.bin (u32 width, height, DXGI format, then tightly packed rows) is uploaded to a DEFAULT texture with
//     CopyTextureRegion, and CreateShaderResourceView writes its T# into a shader-visible heap.
//   - expected.raw (first_light.py's CPU reference: the texture sampled bilinearly with clamp at each covered pixel's
//     interpolated UV, -1 where uncovered, alpha -2 on a UV seam where the edge rule picks the triangle, in vk_harness
//     row order) is compared pixel by pixel with a tolerance; seam pixels count for coverage only.
// Set LEONIDA_VK_VALIDATION=1 to run under VK_LAYER_KHRONOS_validation.
#include "vk_test_device.h"
#include "d3d12_test_shaders.h"
#include "../shims/d3d12_com.h"
#include "../shims/d3d12_xs_device.h"
#include "../src/shader_library.h"

#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

using namespace d3d12test;
using XboxHeapHandleFn = std::uint64_t(WINAPI*)(void* heap);  // heap slot 9: CPU handle, 10: GPU handle, in RAX
using Pixel = std::array<float, 4>;

namespace {

std::vector<std::uint8_t> read_file(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

double number(const leonida::Json& object, std::string_view key, double fallback = -1) {
    const leonida::Json* value = object.find(key);
    return value && value->kind == leonida::Json::Kind::number ? value->number : fallback;
}

std::string stage_hash(const leonida::Json& manifest, std::string_view stage) {
    const leonida::Json* entry = manifest.find(stage);
    const leonida::Json* hash = entry ? entry->find("fssm_sha256") : nullptr;
    return hash && hash->kind == leonida::Json::Kind::string ? hash->string : std::string();
}

// A texture.bin format: its block edge (1 for plain texels, 4 for BC) and the bytes of one texel or block. These are
// the formats the game's TTPF pools use (docs/first_light.md section 9).
struct TexelLayout {
    UINT block = 0;
    UINT bytes = 0;
};
TexelLayout texel_layout(DXGI_FORMAT format) {
    switch (format) {
    case DXGI_FORMAT_R8G8B8A8_UNORM:
    case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
    case DXGI_FORMAT_B8G8R8A8_UNORM:
    case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB: return {1, 4};
    case DXGI_FORMAT_R16G16B16A16_FLOAT:
    case DXGI_FORMAT_R16G16B16A16_UNORM: return {1, 8};
    case DXGI_FORMAT_R16_FLOAT: return {1, 2};
    case DXGI_FORMAT_R8_UNORM:
    case DXGI_FORMAT_A8_UNORM: return {1, 1};
    case DXGI_FORMAT_BC1_UNORM:
    case DXGI_FORMAT_BC1_UNORM_SRGB:
    case DXGI_FORMAT_BC4_UNORM: return {4, 8};
    case DXGI_FORMAT_BC2_UNORM:
    case DXGI_FORMAT_BC3_UNORM:
    case DXGI_FORMAT_BC3_UNORM_SRGB:
    case DXGI_FORMAT_BC5_UNORM:
    case DXGI_FORMAT_BC6H_UF16:
    case DXGI_FORMAT_BC6H_SF16:
    case DXGI_FORMAT_BC7_UNORM:
    case DXGI_FORMAT_BC7_UNORM_SRGB: return {4, 16};
    default: return {};
    }
}

}  // namespace

int main() {
    // _dupenv_s, as pe_loader_test reads its environment: MSVC warns about std::getenv (C4996).
    std::filesystem::path dir;
    {
        char* env = nullptr;
        std::size_t env_size = 0;
        if (_dupenv_s(&env, &env_size, "LEONIDA_FIRST_LIGHT_TEX_DIR") != 0 || !env || !*env) {
            std::free(env);
            std::printf("SKIP: LEONIDA_FIRST_LIGHT_TEX_DIR is not set (run tools/first_light.py --texture first)\n");
            return 77;
        }
        dir = env;
        std::free(env);
    }
    vktest::Device d;
    const auto device_status = d.initialize();
    if (device_status == vktest::DeviceStatus::unavailable) {
        std::fprintf(stderr, "SKIP: no usable Vulkan 1.3 device\n");
        return 77;
    }
    EXPECT(device_status == vktest::DeviceStatus::ready);
    if (!d.bindless) {
        std::printf("SKIP: the device lacks the descriptor indexing features of the bindless set 0\n");
        return 77;
    }

    const auto manifest_text = read_file(dir / "manifest.json");
    leonida::Json manifest;
    EXPECT(leonida::JsonReader(std::string_view(reinterpret_cast<const char*>(manifest_text.data()),
                                                manifest_text.size()))
               .parse(manifest));
    for (const char* name : {"vs.dxbc", "ps.dxbc", "index.json", "vs.spv", "ps.spv", "mesh.bin", "texture.bin",
                             "expected.raw"}) {
        if (!std::filesystem::is_regular_file(dir / name)) {
            std::fprintf(stderr, "%s has no %s: re-run tools/first_light.py with --texture\n", dir.string().c_str(),
                         name);
            return 1;
        }
    }
    const UINT width = UINT(number(manifest, "width")), height = UINT(number(manifest, "height"));
    EXPECT(width > 0 && width <= 4096 && height > 0 && height <= 4096);
    const UINT pitch = (width * sizeof(Pixel) + 255) / 256 * 256;

    const auto vs_container = read_file(dir / "vs.dxbc"), ps_container = read_file(dir / "ps.dxbc");
    const std::string vs_hash = leonida::sha256_hex(leonida::fssm_program(vs_container));
    const std::string ps_hash = leonida::sha256_hex(leonida::fssm_program(ps_container));
    EXPECT(vs_hash == stage_hash(manifest, "vs") && ps_hash == stage_hash(manifest, "ps"));
    leonida::ShaderLibrary library;
    EXPECT(library.load(dir.string()));
    EXPECT(library.find("vs", vs_hash) && library.find("ps", ps_hash));

    const auto mesh = read_file(dir / "mesh.bin");
    EXPECT(mesh.size() >= 8);
    std::uint32_t vertex_count = 0, index_count = 0;
    std::memcpy(&vertex_count, mesh.data(), 4);
    std::memcpy(&index_count, mesh.data() + 4, 4);
    constexpr UINT stride = 5 * sizeof(float);
    const std::size_t vertex_bytes = std::size_t(vertex_count) * stride, index_bytes = std::size_t(index_count) * 4;
    EXPECT(vertex_count && index_count && mesh.size() == 8 + vertex_bytes + index_bytes);
    const std::size_t index_offset = (vertex_bytes + 255) / 256 * 256;

    // texture.bin: u32 width, height, DXGI format, then rows with no padding.
    const auto texture = read_file(dir / "texture.bin");
    EXPECT(texture.size() >= 12);
    std::uint32_t tex_w = 0, tex_h = 0, tex_format = 0;
    std::memcpy(&tex_w, texture.data(), 4);
    std::memcpy(&tex_h, texture.data() + 4, 4);
    std::memcpy(&tex_format, texture.data() + 8, 4);
    const TexelLayout texel = texel_layout(DXGI_FORMAT(tex_format));
    EXPECT(texel.bytes && tex_w && tex_h && tex_w <= 16384 && tex_h <= 16384);
    // Rows of texels, or of 4x4 blocks for a BC format: the payload is tightly packed either way.
    const UINT tex_rows = (tex_h + texel.block - 1) / texel.block;
    const UINT tex_row = (tex_w + texel.block - 1) / texel.block * texel.bytes, tex_pitch = (tex_row + 255) / 256 * 256;
    EXPECT(texture.size() == 12 + std::size_t(tex_rows) * tex_row);
    const std::size_t tex_offset = (index_offset + index_bytes + 511) / 512 * 512;

    std::vector<Pixel> pixels(std::size_t(width) * height);
    {
        rt::d3d12::VulkanDeviceInfo info{};
        info.get_instance_proc = d.instance_proc;
        info.instance = d.instance;
        info.physical = d.physical;
        info.device = d.device;
        info.queue = d.queue;
        info.queue_family = d.family;
        info.buffer_device_address = true;
        info.bindless = d.bindless;
        info.bindless_partially_bound = d.partially_bound;
        info.image_view_min_lod = d.image_view_min_lod;
        info.subgroup = d.subgroup;
        info.shader_dir = dir.string();
        ID3D12Device* inner = nullptr;
        EXPECT(SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&inner))));
        Ref<ID3D12Device> device;
        EXPECT(SUCCEEDED(rt::d3d12_xs::create_device_proxy(inner, __uuidof(ID3D12Device),
                                                           reinterpret_cast<void**>(&device.p))));
        rt::d3d12::set_device_outer(inner, device.p);

        // [0] s0..s(p-1): zero constants; [1] s(p):s(p+1): the SRV table whose low dword the PS reads as its
        // pointer. p is the PS's pointer SGPR from the index (layer_ptr_sgpr; s8 for $PS_PassthroughCopy).
        const UINT ptr_sgpr = UINT(number(manifest, "layer_ptr_sgpr", 8));
        EXPECT(ptr_sgpr <= 30);
        // The PS s_loads its T# at a byte offset in the table (layer_tsharp_offset, from its listing): the table
        // holds that many descriptors before it, and the texture's SRV goes in that slot.
        const UINT tsharp_offset = UINT(number(manifest, "layer_tsharp_offset", 0));
        EXPECT(tsharp_offset % 32 == 0 && tsharp_offset <= 32 * 63);
        const UINT srv_slot = tsharp_offset / 32;
        D3D12_DESCRIPTOR_RANGE srv_range{D3D12_DESCRIPTOR_RANGE_TYPE_SRV, srv_slot + 1, 0, 0, 0};
        D3D12_ROOT_PARAMETER params[2] = {};
        params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        params[0].Constants = {0, 0, ptr_sgpr};
        params[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        params[1].DescriptorTable = {1, &srv_range};
        params[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        const UINT table_param = ptr_sgpr ? 1 : 0;  // with the pointer at s0 there is nothing to pad
        D3D12_VERSIONED_ROOT_SIGNATURE_DESC root_desc{};
        root_desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_0;
        root_desc.Desc_1_0 = {table_param + 1, ptr_sgpr ? params : params + 1, 0, nullptr,
                              D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT};
        const auto root_blob = serialize_root_signature(root_desc);
        EXPECT(!root_blob.empty());
        Ref<ID3D12RootSignature> root;
        EXPECT(SUCCEEDED(device->CreateRootSignature(0, root_blob.data(), root_blob.size(), IID_PPV_ARGS(&root.p))));

        const D3D12_INPUT_ELEMENT_DESC elements[2] = {
            {"POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
            {"TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 12, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0}};
        D3D12_GRAPHICS_PIPELINE_STATE_DESC pso_desc{};
        pso_desc.pRootSignature = root.p;
        pso_desc.VS = {vs_container.data(), vs_container.size()};
        pso_desc.PS = {ps_container.data(), ps_container.size()};
        pso_desc.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
        pso_desc.SampleMask = 0xffffffff;
        pso_desc.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
        pso_desc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
        pso_desc.RasterizerState.DepthClipEnable = TRUE;
        pso_desc.InputLayout = {elements, 2};
        pso_desc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        pso_desc.NumRenderTargets = 1;
        pso_desc.RTVFormats[0] = DXGI_FORMAT_R32G32B32A32_FLOAT;
        pso_desc.SampleDesc.Count = 1;
        Ref<ID3D12PipelineState> pso;
        EXPECT(SUCCEEDED(device->CreateGraphicsPipelineState(&pso_desc, IID_PPV_ARGS(&pso.p))));

        const D3D12_HEAP_PROPERTIES default_heap{D3D12_HEAP_TYPE_DEFAULT};
        const D3D12_HEAP_PROPERTIES upload_heap{D3D12_HEAP_TYPE_UPLOAD};
        const D3D12_HEAP_PROPERTIES readback_heap{D3D12_HEAP_TYPE_READBACK};
        Ref<ID3D12Resource> vertex_buffer, index_buffer, upload, readback, target, tex;
        const auto vb_desc = buffer_desc(vertex_bytes), ib_desc = buffer_desc(index_bytes);
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &vb_desc,
                                                         D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                         IID_PPV_ARGS(&vertex_buffer.p))));
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &ib_desc,
                                                         D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                         IID_PPV_ARGS(&index_buffer.p))));
        const auto upload_desc = buffer_desc(tex_offset + std::size_t(tex_pitch) * tex_rows);
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&upload_heap, D3D12_HEAP_FLAG_NONE, &upload_desc,
                                                         D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                                         IID_PPV_ARGS(&upload.p))));
        const auto readback_desc = buffer_desc(UINT64(pitch) * height);
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&readback_heap, D3D12_HEAP_FLAG_NONE, &readback_desc,
                                                         D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                         IID_PPV_ARGS(&readback.p))));
        D3D12_RESOURCE_DESC rt_desc{};
        rt_desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        rt_desc.Width = width;
        rt_desc.Height = height;
        rt_desc.DepthOrArraySize = 1;
        rt_desc.MipLevels = 1;
        rt_desc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
        rt_desc.SampleDesc.Count = 1;
        rt_desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &rt_desc,
                                                         D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&target.p))));
        D3D12_RESOURCE_DESC tex_desc = rt_desc;
        tex_desc.Width = tex_w;
        tex_desc.Height = tex_h;
        tex_desc.Format = DXGI_FORMAT(tex_format);
        tex_desc.Flags = D3D12_RESOURCE_FLAG_NONE;
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &tex_desc,
                                                         D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&tex.p))));

        std::uint8_t* up = nullptr;
        EXPECT(SUCCEEDED(upload->Map(0, nullptr, reinterpret_cast<void**>(&up))));
        std::memcpy(up, mesh.data() + 8, vertex_bytes);
        std::memcpy(up + index_offset, mesh.data() + 8 + vertex_bytes, index_bytes);
        for (UINT row = 0; row < tex_rows; ++row)
            std::memcpy(up + tex_offset + std::size_t(row) * tex_pitch, texture.data() + 12 + std::size_t(row) * tex_row,
                        tex_row);

        const D3D12_DESCRIPTOR_HEAP_DESC rtv_heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1,
                                                       D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
        Ref<ID3D12DescriptorHeap> rtv_heap, srv_heap;
        EXPECT(SUCCEEDED(device->CreateDescriptorHeap(&rtv_heap_desc, IID_PPV_ARGS(&rtv_heap.p))));
        const D3D12_CPU_DESCRIPTOR_HANDLE rtv{
            SIZE_T(reinterpret_cast<XboxHeapHandleFn>(slot(rtv_heap.p, 9))(rtv_heap.p))};
        device->CreateRenderTargetView(target.p, nullptr, rtv);
        const D3D12_DESCRIPTOR_HEAP_DESC srv_heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, srv_slot + 1,
                                                       D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
        EXPECT(SUCCEEDED(device->CreateDescriptorHeap(&srv_heap_desc, IID_PPV_ARGS(&srv_heap.p))));
        const std::uint64_t srv_cpu = reinterpret_cast<XboxHeapHandleFn>(slot(srv_heap.p, 9))(srv_heap.p);
        const std::uint64_t srv_gpu = reinterpret_cast<XboxHeapHandleFn>(slot(srv_heap.p, 10))(srv_heap.p);
        // The PS keeps only the low dword of the table pointer (s9 is zeroed), so the heap must sit below 4 GiB.
        EXPECT(srv_gpu >> 32 == 0);
        D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
        srv.Format = DXGI_FORMAT(tex_format);
        srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srv.Texture2D.MipLevels = 1;
        const UINT srv_step = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        EXPECT(srv_step == 32);  // the console's T# is 32 bytes: a byte offset in the table is a slot
        device->CreateShaderResourceView(tex.p, &srv,
                                         D3D12_CPU_DESCRIPTOR_HANDLE{SIZE_T(srv_cpu + std::uint64_t(srv_slot) * srv_step)});

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

        list->CopyBufferRegion(vertex_buffer.p, 0, upload.p, 0, vertex_bytes);
        list->CopyBufferRegion(index_buffer.p, 0, upload.p, index_offset, index_bytes);
        {
            D3D12_TEXTURE_COPY_LOCATION dst{};
            dst.pResource = tex.p;
            dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
            D3D12_TEXTURE_COPY_LOCATION src{};
            src.pResource = upload.p;
            src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
            src.PlacedFootprint = {tex_offset, {DXGI_FORMAT(tex_format), tex_w, tex_h, 1, tex_pitch}};
            list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
        }
        D3D12_RESOURCE_BARRIER barriers[4] = {
            transition(vertex_buffer.p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER),
            transition(index_buffer.p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_INDEX_BUFFER),
            transition(target.p, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_RENDER_TARGET),
            transition(tex.p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE)};
        list->ResourceBarrier(4, barriers);
        const FLOAT clear[4] = {-1, -1, -1, -1};
        list->ClearRenderTargetView(rtv, clear, 0, nullptr);

        list->SetGraphicsRootSignature(root.p);
        ID3D12DescriptorHeap* heaps[] = {srv_heap.p};
        list->SetDescriptorHeaps(1, heaps);
        list->SetGraphicsRootDescriptorTable(table_param, D3D12_GPU_DESCRIPTOR_HANDLE{srv_gpu});
        list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        const D3D12_VERTEX_BUFFER_VIEW vbv{vertex_buffer->GetGPUVirtualAddress(), UINT(vertex_bytes), stride};
        list->IASetVertexBuffers(0, 1, &vbv);
        const D3D12_INDEX_BUFFER_VIEW ibv{index_buffer->GetGPUVirtualAddress(), UINT(index_bytes),
                                          DXGI_FORMAT_R32_UINT};
        list->IASetIndexBuffer(&ibv);
        const D3D12_VIEWPORT viewport{0, 0, float(width), float(height), 0, 1};
        const D3D12_RECT scissor{0, 0, LONG(width), LONG(height)};
        list->RSSetViewports(1, &viewport);
        list->RSSetScissorRects(1, &scissor);
        list->OMSetRenderTargets(1, &rtv, FALSE, nullptr);
        list->DrawIndexedInstanced(index_count, 1, 0, 0, 0);

        D3D12_RESOURCE_BARRIER to_copy = transition(target.p, D3D12_RESOURCE_STATE_RENDER_TARGET,
                                                    D3D12_RESOURCE_STATE_COPY_SOURCE);
        list->ResourceBarrier(1, &to_copy);
        D3D12_TEXTURE_COPY_LOCATION dst{};
        dst.pResource = readback.p;
        dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        dst.PlacedFootprint = {0, {DXGI_FORMAT_R32G32B32A32_FLOAT, width, height, 1, pitch}};
        D3D12_TEXTURE_COPY_LOCATION src{};
        src.pResource = target.p;
        src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
        EXPECT(list->Close() == S_OK);
        ID3D12CommandList* lists[] = {list.p};
        queue->ExecuteCommandLists(1, lists);
        EXPECT(rt::d3d12::last_execute_result(queue.p) == leonida::ReplayResult::success);
        EXPECT(SUCCEEDED(queue->Signal(fence.p, 1)));
        EXPECT(SUCCEEDED(fence->SetEventOnCompletion(1, nullptr)));
        std::uint8_t* mapped = nullptr;
        EXPECT(SUCCEEDED(readback->Map(0, nullptr, reinterpret_cast<void**>(&mapped))));
        for (UINT row = 0; row < height; ++row)
            std::memcpy(&pixels[std::size_t(row) * width], mapped + std::size_t(row) * pitch, width * sizeof(Pixel));

        for (auto* r : {&vertex_buffer, &index_buffer, &upload, &readback, &target, &tex}) r->reset();
        pso.reset();
        root.reset();
        rtv_heap.reset();
        srv_heap.reset();
        list.reset();
        allocator.reset();
        queue.reset();
        fence.reset();
        EXPECT(rt::d3d12_xs::device_proxy_ref_count(device.p) == 1);
    }

    // The image as drawn, rows in vk_harness order, next to the inputs (d3d12_textured.raw), so a run on one GPU can
    // be compared with another's byte for byte; nothing in the repository reads it.
    {
        std::ofstream out(dir / "d3d12_textured.raw", std::ios::binary);
        for (UINT vk_y = 0; vk_y < height; ++vk_y)
            out.write(reinterpret_cast<const char*>(&pixels[std::size_t(height - 1 - vk_y) * width]),
                      std::streamsize(width * sizeof(Pixel)));
    }

    // expected.raw is in vk_harness row order; row r here is row H-1-r there (the layer flips the viewport).
    const auto expected = read_file(dir / "expected.raw");
    EXPECT(expected.size() == pixels.size() * sizeof(Pixel));
    const double tolerance = number(manifest, "texture_tolerance", 4.0 / 255.0);
    // expected_band.raw (optional): per pixel and channel, how much the reference itself changes when the sample
    // position moves by the GPU's sub-pixel snapping; it is added to the tolerance. A float format's tolerance is
    // relative to the expected value (HDR texels go well above 1).
    const auto band = read_file(dir / "expected_band.raw");
    EXPECT(band.empty() || band.size() == expected.size());
    const leonida::Json* relative_key = manifest.find("texture_tolerance_relative");
    const bool relative = relative_key && relative_key->kind == leonida::Json::Kind::boolean && relative_key->boolean;
    unsigned covered = 0, compared = 0, coverage_mismatch = 0, over = 0, ambiguous = 0;
    double worst = 0.0, sum = 0.0;
    for (UINT vk_y = 0; vk_y < height; ++vk_y) {
        for (UINT x = 0; x < width; ++x) {
            Pixel want;
            std::memcpy(want.data(), expected.data() + (std::size_t(vk_y) * width + x) * sizeof(Pixel), sizeof(Pixel));
            const Pixel& got = pixels[std::size_t(height - 1 - vk_y) * width + x];
            const bool got_covered = got[3] != -1.0f, want_covered = want[3] != -1.0f;
            covered += got_covered;
            if (got_covered != want_covered) {
                ++coverage_mismatch;
                continue;
            }
            if (!want_covered) continue;
            if (want[3] == -2.0f) {  // on a UV seam: the edge rule picks the triangle, so the colour is not compared
                ++ambiguous;
                continue;
            }
            ++compared;
            Pixel slack{};
            if (!band.empty())
                std::memcpy(slack.data(), band.data() + (std::size_t(vk_y) * width + x) * sizeof(Pixel), sizeof(Pixel));
            for (int c = 0; c < 4; ++c) {
                if (std::isnan(want[c])) continue;  // a channel the reference can't predict (an unknown alpha)
                const double e = std::fabs(double(got[c]) - double(want[c]));
                worst = std::max(worst, e);
                sum += e;
                const double scale = relative ? std::max(1.0, std::fabs(double(want[c]))) : 1.0;
                if (e > tolerance * scale + double(slack[c])) ++over;
            }
        }
    }
    std::printf("covered %u/%u; %u pixels compared with the CPU reference: worst %.5f, mean %.6f, %u channels over "
                "%.5f; %u coverage mismatches; %u on a UV seam (colour not compared)\n", covered, width * height,
                compared, worst, compared ? sum / (compared * 4.0) : 0.0, over, tolerance, coverage_mismatch,
                ambiguous);
    EXPECT(compared > 0);
    // Seams are edges, so a large count means the reference is wrong. A flattened 3D model stacks its front and back
    // layers, and their edges cross many pixel centres within the snapping distance (1.4% on one game model).
    EXPECT(ambiguous <= covered / 20);
    // Coverage is the same rasterisation as d3d12_first_light; the CPU reference marks pixel centres inside a
    // triangle, so only edge pixels may disagree.
    EXPECT(coverage_mismatch <= covered / 100);
    EXPECT(over <= compared / 1000);
    if (d.validation_errors()) {
        std::fprintf(stderr, "%u validation errors\n", d.validation_errors());
        return 1;
    }
    std::printf("first light with a texture: the game's VS + texture-sampling PS drew %u triangles\n", index_count / 3);
    return 0;
}
