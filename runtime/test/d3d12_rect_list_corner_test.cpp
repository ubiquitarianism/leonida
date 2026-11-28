// RT-RECT-LIST-CORNER: edge cases for console rectangle-list fourth-corner selection and reconstruction
// (src/rect_list_geometry.h), tested through the D3D12-on-Vulkan layer without llvm-mc or game data.
//
// Cases exercised on a real Vulkan GPU (skips with 77 without a Vulkan 1.3 GPU or geometryShader):
//   1. Uniform w != 1 (w = 2, and large w = 1e20 that overflows pre-divide squared edge lengths to +Inf)
//      with right angle at each of v0, v1, v2.
//   2. Differing w across vertices: w = (1.875, 1, 1) with right angle at v0 and w = (1, 1, 1.875) with
//      right angle at v2 (pre-divide clip-space distances pick the wrong corner and leave half the target
//      uncovered; post-divide NDC selects the right corner and clip-space v4 = other_a + other_b - corner
//      with w4 = 0.125 covers the whole target, including with a non-dyadic constant/flat attribute).
//   3. Thin axis-aligned strips (5000:1 aspect, dx = 10000, dy = 2, where float32 dx^2 + dy^2 == dx^2
//      breaks the squared-length tie-break) with right angle at v1 and at v2.
//   4. Near-square right triangles (1 vs 1 + 1e-6) with right angle at v0, v1, v2.
//   5. Rotated 45-degree rectangle (no shared x or y coordinates between vertices): fourth corner still
//      closes the diamond.
//   6. Non-dyadic constant/flat attribute ({1/3, 0.7, 1 + 1 ULP, 1}): bit-exact across all pixels with
//      zero ULP drift from (((v0 + v1) + v2) - corner) - corner.
//   7. Degenerate triangles (coincident vertices, collinear vertices, and w == 0): draw cleanly with no
//      pixels written and no validation errors.
#include "test_temp_dir.h"
#include "vk_test_device.h"
#include "d3d12_test_shaders.h"
#include "../shims/d3d12_com.h"
#include "../shims/d3d12_xs_device.h"
#include "../src/rect_list_geometry.h"
#include "../src/shader_library.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <initializer_list>
#include <string>
#include <vector>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

using namespace d3d12test;
using XboxHeapHandleFn = std::uint64_t(WINAPI*)(void* heap);

namespace {

struct Vertex {
    float position[4];
    float color[4];
};
static_assert(sizeof(Vertex) == 32);

using Pixel = std::array<float, 4>;
constexpr UINT kExtent = 64;
constexpr UINT kPitch = kExtent * sizeof(Pixel);
constexpr D3D_PRIMITIVE_TOPOLOGY kRectList = D3D_PRIMITIVE_TOPOLOGY(leonida::topology_rect_list);
const std::uint32_t kRectTypeValue = leonida::topology_type_rect_list;

void emit_spv(std::vector<std::uint32_t>& to, std::uint32_t op, std::initializer_list<std::uint32_t> args) {
    to.push_back((std::uint32_t(args.size()) + 1u) << 16 | op);
    to.insert(to.end(), args.begin(), args.end());
}

// Minimal SPIR-V 1.0 pass-through vertex shader following the translator's output contract:
//   Input Location 0 (vec4) -> Output BuiltIn Position (vec4)
//   Input Location 1 (vec4) -> Output Location 0 (vec4)
std::vector<std::uint32_t> build_passthrough_vs() {
    enum : std::uint32_t {
        op_memory_model = 14, op_entry_point = 15, op_capability = 17,
        op_type_void = 19, op_type_float = 22, op_type_vector = 23,
        op_type_pointer = 32, op_type_function = 33,
        op_function = 54, op_function_end = 56, op_variable = 59, op_load = 61, op_store = 62,
        op_decorate = 71, op_label = 248, op_return = 253,
        deco_builtin = 11, deco_location = 30, builtin_position = 0,
        storage_input = 1, storage_output = 3,
    };
    std::vector<std::uint32_t> w{0x07230203u, 0x00010000u, 0, 15, 0};
    emit_spv(w, op_capability, {1});        // Shader
    emit_spv(w, op_memory_model, {0, 1});   // Logical, GLSL450
    emit_spv(w, op_entry_point, {0, 11, 0x6e69616du, 0, 7, 8, 9, 10});  // Vertex %main "main"
    emit_spv(w, op_decorate, {7, deco_location, 0});
    emit_spv(w, op_decorate, {8, deco_location, 1});
    emit_spv(w, op_decorate, {9, deco_builtin, builtin_position});
    emit_spv(w, op_decorate, {10, deco_location, 0});
    emit_spv(w, op_type_void, {1});
    emit_spv(w, op_type_function, {2, 1});
    emit_spv(w, op_type_float, {3, 32});
    emit_spv(w, op_type_vector, {4, 3, 4});
    emit_spv(w, op_type_pointer, {5, storage_input, 4});
    emit_spv(w, op_type_pointer, {6, storage_output, 4});
    emit_spv(w, op_variable, {5, 7, storage_input});
    emit_spv(w, op_variable, {5, 8, storage_input});
    emit_spv(w, op_variable, {6, 9, storage_output});
    emit_spv(w, op_variable, {6, 10, storage_output});
    emit_spv(w, op_function, {1, 11, 0, 2});
    emit_spv(w, op_label, {12});
    emit_spv(w, op_load, {4, 13, 7});
    emit_spv(w, op_store, {9, 13});
    emit_spv(w, op_load, {4, 14, 8});
    emit_spv(w, op_store, {10, 14});
    emit_spv(w, op_return, {});
    emit_spv(w, op_function_end, {});
    return w;
}

// Minimal SPIR-V 1.0 pass-through fragment shader: Input Location 0 (vec4) -> Output Location 0 (vec4).
std::vector<std::uint32_t> build_passthrough_fs() {
    enum : std::uint32_t {
        op_memory_model = 14, op_entry_point = 15, op_execution_mode = 16, op_capability = 17,
        op_type_void = 19, op_type_float = 22, op_type_vector = 23,
        op_type_pointer = 32, op_type_function = 33,
        op_function = 54, op_function_end = 56, op_variable = 59, op_load = 61, op_store = 62,
        op_decorate = 71, op_label = 248, op_return = 253,
        deco_location = 30, storage_input = 1, storage_output = 3,
    };
    std::vector<std::uint32_t> w{0x07230203u, 0x00010000u, 0, 12, 0};
    emit_spv(w, op_capability, {1});        // Shader
    emit_spv(w, op_memory_model, {0, 1});   // Logical, GLSL450
    emit_spv(w, op_entry_point, {4, 9, 0x6e69616du, 0, 7, 8});  // Fragment %main "main"
    emit_spv(w, op_execution_mode, {9, 7}); // OriginUpperLeft
    emit_spv(w, op_decorate, {7, deco_location, 0});
    emit_spv(w, op_decorate, {8, deco_location, 0});
    emit_spv(w, op_type_void, {1});
    emit_spv(w, op_type_function, {2, 1});
    emit_spv(w, op_type_float, {3, 32});
    emit_spv(w, op_type_vector, {4, 3, 4});
    emit_spv(w, op_type_pointer, {5, storage_input, 4});
    emit_spv(w, op_type_pointer, {6, storage_output, 4});
    emit_spv(w, op_variable, {5, 7, storage_input});
    emit_spv(w, op_variable, {6, 8, storage_output});
    emit_spv(w, op_function, {1, 9, 0, 2});
    emit_spv(w, op_label, {10});
    emit_spv(w, op_load, {4, 11, 7});
    emit_spv(w, op_store, {8, 11});
    emit_spv(w, op_return, {});
    emit_spv(w, op_function_end, {});
    return w;
}

bool write_spv(const std::filesystem::path& path, const std::vector<std::uint32_t>& words) {
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(words.data()), std::streamsize(words.size() * sizeof(std::uint32_t)));
    return bool(out);
}

Vertex make_v(float x, float y, float w, const Pixel& c) {
    return Vertex{{x * w, y * w, 0.0f, w}, {c[0], c[1], c[2], c[3]}};
}

bool check_plane(const char* name, const std::vector<Pixel>& pixels, const Vertex t[3], float tol = 2e-5f) {
    const float x0 = t[0].position[0] / t[0].position[3], y0 = t[0].position[1] / t[0].position[3];
    const float x1 = t[1].position[0] / t[1].position[3], y1 = t[1].position[1] / t[1].position[3];
    const float x2 = t[2].position[0] / t[2].position[3], y2 = t[2].position[1] / t[2].position[3];
    const float ax = x1 - x0, ay = y1 - y0, bx = x2 - x0, by = y2 - y0;
    const float det = ax * by - ay * bx;
    const float iw0 = 1.0f / t[0].position[3], iw1 = 1.0f / t[1].position[3], iw2 = 1.0f / t[2].position[3];
    unsigned bad = 0;
    float worst = 0.0f;
    for (UINT row = 0; row < kExtent; ++row) {
        for (UINT col = 0; col < kExtent; ++col) {
            const float px = 2.0f * (float(col) + 0.5f) / kExtent - 1.0f - x0;
            const float py = 1.0f - 2.0f * (float(row) + 0.5f) / kExtent - y0;
            const float s = (px * by - py * bx) / det;
            const float r = (ax * py - ay * px) / det;
            const float u0 = 1.0f - s - r;
            const float inv_w = u0 * iw0 + s * iw1 + r * iw2;
            const Pixel& got = pixels[row * kExtent + col];
            bool ok = true;
            for (int c = 0; c < 4; ++c) {
                const float want = (u0 * t[0].color[c] * iw0 + s * t[1].color[c] * iw1 + r * t[2].color[c] * iw2) / inv_w;
                const float err = std::fabs(got[c] - want);
                worst = std::fmax(worst, err);
                if (!(err <= tol)) ok = false;
            }
            if (!ok && bad++ < 6) {
                std::fprintf(stderr, "%s: pixel (%u,%u) got %g,%g,%g,%g\n",
                             name, col, row, got[0], got[1], got[2], got[3]);
            }
        }
    }
    if (bad) {
        std::fprintf(stderr, "%s: %u of %u pixels off plane (worst error %g)\n",
                     name, bad, kExtent * kExtent, double(worst));
        return false;
    }
    return true;
}

}  // namespace

int main() {
    vktest::Device d;
    const auto status = d.initialize();
    if (status == vktest::DeviceStatus::unavailable) return 77;
    EXPECT(status == vktest::DeviceStatus::ready);
    if (!d.enabled.geometryShader) {
        std::printf("SKIP: no geometryShader on this GPU\n");
        return 77;
    }

    const auto vs_words = build_passthrough_vs();
    const auto fs_words = build_passthrough_fs();
    std::string why;
    const auto gs_words = leonida::rect_list_geometry(vs_words, why);
    if (gs_words.empty()) {
        std::fprintf(stderr, "rect_list_geometry failed: %s\n", why.c_str());
        return 1;
    }

    const auto vs_program = fssm_program({0x56531701, 0xbf810000});
    const auto ps_program = fssm_program({0x50531701, 0xbf810000});
    const auto vs_container = shader_container(vs_program);
    const auto ps_container = shader_container(ps_program);

    const testfiles::ScratchDir scratch("leonida_rect_list_corner_test");
    const auto& lib = scratch.path();
    EXPECT(write_spv(lib / "vs_rect.spv", vs_words));
    EXPECT(write_spv(lib / "ps_rect.spv", fs_words));
    {
        std::ofstream index(lib / "index.json", std::ios::binary);
        index << "{\"vs_rect\": {\"stage\": \"vs\", \"fssm_sha256\": \"" << leonida::sha256_hex(vs_program)
              << "\", \"bindings\": [], \"push_constant_bytes\": 0, \"input_signature\": ["
                 "{\"semantic\": \"POSITION\", \"semantic_index\": 0, \"register\": 0, \"mask\": 15, \"rw_mask\": 15, "
                 "\"system_value\": 0, \"component_type\": \"float\"}, "
                 "{\"semantic\": \"COLOR\", \"semantic_index\": 0, \"register\": 1, \"mask\": 15, \"rw_mask\": 15, "
                 "\"system_value\": 0, \"component_type\": \"float\"}]},\n"
                 " \"ps_rect\": {\"stage\": \"ps\", \"fssm_sha256\": \"" << leonida::sha256_hex(ps_program)
              << "\", \"bindings\": [], \"push_constant_bytes\": 0, \"outputs\": [0]}}";
    }

    const Pixel ca{0.0f, 0.0f, 0.0f, 1.0f};
    const Pixel cb{1.0f, 0.0f, 0.5f, 1.0f};
    const Pixel cc{0.0f, 0.5f, 1.0f, 1.0f};
    const Pixel flat_color{1.0f / 3.0f, 0.7f, 1.0f + 1.1920929e-7f, 1.0f};
    const Pixel solid{0.25f, 0.75f, 0.5f, 1.0f};
    const Pixel clear_px{0.5f, 0.125f, 0.25f, 0.5f};

    struct Case {
        const char* name;
        Vertex v[3];
    };
    const std::vector<Case> cases = {
        // 0..2: uniform w = 2 with right angle at v0, v1, v2
        {"w=2 corner v0", {make_v(-1, 1, 2, ca), make_v(1, 1, 2, cb), make_v(-1, -1, 2, cc)}},
        {"w=2 corner v1", {make_v(-1, 1, 2, ca), make_v(1, 1, 2, cb), make_v(1, -1, 2, cc)}},
        {"w=2 corner v2", {make_v(-1, 1, 2, ca), make_v(1, -1, 2, cb), make_v(-1, -1, 2, cc)}},
        // 3: large uniform w = 1e20 with right angle at v2 (overflows pre-divide dx^2+dy^2 to +Inf on main)
        {"w=1e20 corner v2", {make_v(-1, 1, 1e20f, ca), make_v(1, -1, 1e20f, cb), make_v(-1, -1, 1e20f, cc)}},
        // 4..5: differing w with right angle at v0 / v2 (pre-divide squared lengths pick the wrong corner on main;
        // post-divide NDC picks the right corner and clip-space other_a + other_b - corner yields w4 = 0.125 > 0)
        {"differing w=(1.875,1,1) corner v0",
         {make_v(-1, 1, 1.875f, solid), make_v(1, 1, 1, solid), make_v(-1, -1, 1, solid)}},
        {"differing w=(1,1,1.875) corner v2",
         {make_v(-1, 1, 1, solid), make_v(1, -1, 1, solid), make_v(-1, -1, 1.875f, solid)}},
        // 6..7: thin 5000:1 axis-aligned strip (dx=10000, dy=2 -> float32 dx^2+dy^2 == dx^2 on main)
        {"thin strip 5000:1 corner v2",
         {make_v(-5000, -1, 1, solid), make_v(5000, 1, 1, solid), make_v(5000, -1, 1, solid)}},
        {"thin strip 5000:1 corner v1",
         {make_v(5000, 1, 1, solid), make_v(5000, -1, 1, solid), make_v(-5000, -1, 1, solid)}},
        // 8..10: near-square right triangles (1 vs 1 + 1e-6) at v0, v1, v2
        {"near-square corner v0",
         {make_v(-0.5f, 0.5f, 1, solid), make_v(0.5f + 1e-6f, 0.5f, 1, solid), make_v(-0.5f, -0.5f, 1, solid)}},
        {"near-square corner v1",
         {make_v(-0.5f, 0.5f, 1, solid), make_v(0.5f + 1e-6f, 0.5f, 1, solid), make_v(0.5f + 1e-6f, -0.5f, 1, solid)}},
        {"near-square corner v2",
         {make_v(-0.5f, 0.5f, 1, solid), make_v(0.5f + 1e-6f, -0.5f, 1, solid), make_v(-0.5f, -0.5f, 1, solid)}},
        // 11: rotated 45-degree rectangle (no vertex shares x or y with another)
        {"rotated 45 deg",
         {make_v(0, 1, 1, ca), make_v(1, 0, 1, cb), make_v(-1, 0, 1, cc)}},
        // 12: non-dyadic constant/flat attribute (drifts 2 ULPs on v4 on main)
        {"flat non-dyadic exact",
         {make_v(-1, 1, 1, flat_color), make_v(1, 1, 1, flat_color), make_v(-1, -1, 1, flat_color)}},
        // 13..15: degenerate triangles (coincident, collinear, and w == 0 strictly outside clip bounds)
        {"degenerate coincident", {make_v(0, 0, 1, solid), make_v(0, 0, 1, solid), make_v(0, 0, 1, solid)}},
        {"degenerate collinear", {make_v(-1, 0, 1, solid), make_v(0, 0, 1, solid), make_v(1, 0, 1, solid)}},
        {"degenerate w=0",
         {Vertex{{1, 1, 0, 0}, {1, 0, 0, 1}}, Vertex{{2, 1, 0, 0}, {1, 0, 0, 1}}, Vertex{{1, 2, 0, 0}, {1, 0, 0, 1}}}},
        // 16: differing w=(1.875,1,1) with non-dyadic constant/flat color (full coverage, exact fourth color)
        {"differing w=(1.875,1,1) flat color",
         {make_v(-1, 1, 1.875f, flat_color), make_v(1, 1, 1, flat_color), make_v(-1, -1, 1, flat_color)}},
    };

    std::vector<Vertex> vb_data;
    vb_data.reserve(cases.size() * 3);
    for (const auto& c : cases) {
        vb_data.push_back(c.v[0]);
        vb_data.push_back(c.v[1]);
        vb_data.push_back(c.v[2]);
    }

    std::vector<std::vector<Pixel>> results(cases.size(), std::vector<Pixel>(kExtent * kExtent));
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
        info.shader_dir = lib.string();
        ID3D12Device* inner = nullptr;
        EXPECT(SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&inner))));
        Ref<ID3D12Device> device;
        EXPECT(SUCCEEDED(rt::d3d12_xs::create_device_proxy(inner, __uuidof(ID3D12Device),
                                                           reinterpret_cast<void**>(&device.p))));
        rt::d3d12::set_device_outer(inner, device.p);

        D3D12_VERSIONED_ROOT_SIGNATURE_DESC root_desc{};
        root_desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_0;
        root_desc.Desc_1_0 = {0, nullptr, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT};
        const auto root_blob = serialize_root_signature(root_desc);
        EXPECT(!root_blob.empty());
        Ref<ID3D12RootSignature> root;
        EXPECT(SUCCEEDED(device->CreateRootSignature(0, root_blob.data(), root_blob.size(), IID_PPV_ARGS(&root.p))));

        const D3D12_INPUT_ELEMENT_DESC layout[] = {
            {"POSITION", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
            {"COLOR", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 16, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
        };
        D3D12_GRAPHICS_PIPELINE_STATE_DESC pso_desc{};
        pso_desc.pRootSignature = root.p;
        pso_desc.VS = {vs_container.data(), vs_container.size()};
        pso_desc.PS = {ps_container.data(), ps_container.size()};
        pso_desc.InputLayout = {layout, 2};
        pso_desc.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
        pso_desc.SampleMask = 0xffffffff;
        pso_desc.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
        pso_desc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
        pso_desc.RasterizerState.DepthClipEnable = TRUE;
        pso_desc.NumRenderTargets = 1;
        pso_desc.RTVFormats[0] = DXGI_FORMAT_R32G32B32A32_FLOAT;
        pso_desc.SampleDesc.Count = 1;
        std::memcpy(&pso_desc.PrimitiveTopologyType, &kRectTypeValue, sizeof(kRectTypeValue));
        Ref<ID3D12PipelineState> pso;
        EXPECT(SUCCEEDED(device->CreateGraphicsPipelineState(&pso_desc, IID_PPV_ARGS(&pso.p))));

        const D3D12_HEAP_PROPERTIES default_heap{D3D12_HEAP_TYPE_DEFAULT};
        const D3D12_HEAP_PROPERTIES upload_heap{D3D12_HEAP_TYPE_UPLOAD};
        const D3D12_HEAP_PROPERTIES readback_heap{D3D12_HEAP_TYPE_READBACK};
        Ref<ID3D12Resource> vb, readback, target;
        const auto vb_bytes = UINT64(vb_data.size() * sizeof(Vertex));
        const auto vb_desc = buffer_desc(vb_bytes);
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&upload_heap, D3D12_HEAP_FLAG_NONE, &vb_desc,
                                                         D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                                         IID_PPV_ARGS(&vb.p))));
        void* vb_map = nullptr;
        EXPECT(SUCCEEDED(vb->Map(0, nullptr, &vb_map)));
        std::memcpy(vb_map, vb_data.data(), std::size_t(vb_bytes));
        vb->Unmap(0, nullptr);

        const auto rb_desc = buffer_desc(kPitch * kExtent);
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&readback_heap, D3D12_HEAP_FLAG_NONE, &rb_desc,
                                                         D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                         IID_PPV_ARGS(&readback.p))));
        D3D12_RESOURCE_DESC rt_desc{};
        rt_desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        rt_desc.Width = rt_desc.Height = kExtent;
        rt_desc.DepthOrArraySize = 1;
        rt_desc.MipLevels = 1;
        rt_desc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
        rt_desc.SampleDesc.Count = 1;
        rt_desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &rt_desc,
                                                         D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                         IID_PPV_ARGS(&target.p))));

        const D3D12_DESCRIPTOR_HEAP_DESC heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
        Ref<ID3D12DescriptorHeap> rtv_heap;
        EXPECT(SUCCEEDED(device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&rtv_heap.p))));
        const D3D12_CPU_DESCRIPTOR_HANDLE rtv{
            SIZE_T(reinterpret_cast<XboxHeapHandleFn>(slot(rtv_heap.p, 9))(rtv_heap.p))};
        device->CreateRenderTargetView(target.p, nullptr, rtv);

        D3D12_COMMAND_QUEUE_DESC qdesc{};
        Ref<ID3D12CommandQueue> queue;
        Ref<ID3D12CommandAllocator> allocator;
        Ref<ID3D12GraphicsCommandList> list;
        Ref<ID3D12Fence> fence;
        EXPECT(SUCCEEDED(device->CreateCommandQueue(&qdesc, IID_PPV_ARGS(&queue.p))));
        EXPECT(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p))));
        EXPECT(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, pso.p,
                                                   IID_PPV_ARGS(&list.p))));
        EXPECT(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))));

        const D3D12_VERTEX_BUFFER_VIEW vbv{vb->GetGPUVirtualAddress(), UINT(vb_bytes), sizeof(Vertex)};
        const D3D12_VIEWPORT vp{0, 0, float(kExtent), float(kExtent), 0, 1};
        const D3D12_RECT scissor{0, 0, LONG(kExtent), LONG(kExtent)};

        for (std::size_t idx = 0; idx < cases.size(); ++idx) {
            if (idx > 0) {
                EXPECT(SUCCEEDED(allocator->Reset()));
                EXPECT(SUCCEEDED(list->Reset(allocator.p, pso.p)));
                auto to_rt = transition(target.p, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_RENDER_TARGET);
                list->ResourceBarrier(1, &to_rt);
            } else {
                auto to_rt = transition(target.p, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_RENDER_TARGET);
                list->ResourceBarrier(1, &to_rt);
            }
            list->ClearRenderTargetView(rtv, clear_px.data(), 0, nullptr);
            list->SetGraphicsRootSignature(root.p);
            list->IASetPrimitiveTopology(kRectList);
            list->IASetVertexBuffers(0, 1, &vbv);
            list->RSSetViewports(1, &vp);
            list->RSSetScissorRects(1, &scissor);
            list->OMSetRenderTargets(1, &rtv, FALSE, nullptr);
            list->DrawInstanced(3, 1, UINT(idx * 3), 0);
            auto to_copy = transition(target.p, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE);
            list->ResourceBarrier(1, &to_copy);
            D3D12_TEXTURE_COPY_LOCATION dst{};
            dst.pResource = readback.p;
            dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
            dst.PlacedFootprint = {0, {DXGI_FORMAT_R32G32B32A32_FLOAT, kExtent, kExtent, 1, kPitch}};
            D3D12_TEXTURE_COPY_LOCATION src{};
            src.pResource = target.p;
            src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
            list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
            EXPECT(list->Close() == S_OK);
            ID3D12CommandList* lists[] = {list.p};
            queue->ExecuteCommandLists(1, lists);
            EXPECT(rt::d3d12::last_execute_result(queue.p) == leonida::ReplayResult::success);
            const UINT64 ticket = UINT64(idx + 1);
            EXPECT(SUCCEEDED(queue->Signal(fence.p, ticket)));
            EXPECT(SUCCEEDED(fence->SetEventOnCompletion(ticket, nullptr)));
            void* mapped = nullptr;
            EXPECT(SUCCEEDED(readback->Map(0, nullptr, &mapped)));
            std::memcpy(results[idx].data(), mapped, results[idx].size() * sizeof(Pixel));
            readback->Unmap(0, nullptr);
        }
    }

    int failures = 0;
    // Cases 0..3: full-screen color plane (uniform w=2 at v0/v1/v2 and large w=1e20 at v2)
    for (std::size_t idx = 0; idx <= 3; ++idx) {
        if (!check_plane(cases[idx].name, results[idx], cases[idx].v)) ++failures;
    }
    // Cases 4..7: full-screen solid coverage (differing w=(1.875,1,1) and (1,1,1.875), thin 5000:1 strip at v2 and v1).
    // For differing w (cases 4..5), smooth perspective interpolation can round by 1 ULP even on constant attributes;
    // uniform-w thin strips (cases 6..7) must match bit-exact.
    for (std::size_t idx = 4; idx <= 7; ++idx) {
        unsigned wrong = 0;
        for (const Pixel& px : results[idx]) {
            bool ok = true;
            if (idx <= 5) {
                for (int c = 0; c < 4; ++c)
                    if (!(std::fabs(px[c] - solid[c]) <= 1e-5f)) ok = false;
            } else {
                ok = std::memcmp(&px, &solid, sizeof(Pixel)) == 0;
            }
            if (!ok) ++wrong;
        }
        if (wrong) {
            std::fprintf(stderr, "%s: %u of %u pixels missed solid fill\n",
                         cases[idx].name, wrong, kExtent * kExtent);
            ++failures;
        }
    }
    // Cases 8..10: near-square [-0.5, 0.5]x[-0.5, 0.5] coverage (center 32x32 pixels strictly inside)
    for (std::size_t idx = 8; idx <= 10; ++idx) {
        unsigned inside_miss = 0, outside_hit = 0;
        for (UINT row = 0; row < kExtent; ++row) {
            for (UINT col = 0; col < kExtent; ++col) {
                const bool drawn = std::memcmp(&results[idx][row * kExtent + col], &solid, sizeof(Pixel)) == 0;
                const bool clear = std::memcmp(&results[idx][row * kExtent + col], &clear_px, sizeof(Pixel)) == 0;
                if (row >= 17 && row < 47 && col >= 17 && col < 47) {
                    if (!drawn) ++inside_miss;
                } else if (row < 15 || row >= 49 || col < 15 || col >= 49) {
                    if (!clear) ++outside_hit;
                }
            }
        }
        if (inside_miss || outside_hit) {
            std::fprintf(stderr, "%s: inside_miss=%u outside_hit=%u\n", cases[idx].name, inside_miss, outside_hit);
            ++failures;
        }
    }
    // Case 11: rotated 45-degree diamond |x| + |y| <= 1 across all 4 quadrants
    {
        unsigned inside_miss = 0, outside_hit = 0;
        const auto& px = results[11];
        for (UINT row = 0; row < kExtent; ++row) {
            for (UINT col = 0; col < kExtent; ++col) {
                const float x = 2.0f * (float(col) + 0.5f) / kExtent - 1.0f;
                const float y = 1.0f - 2.0f * (float(row) + 0.5f) / kExtent;
                const float d1 = std::fabs(x) + std::fabs(y);
                const bool is_clear = std::memcmp(&px[row * kExtent + col], &clear_px, sizeof(Pixel)) == 0;
                if (d1 < 0.94f) {
                    if (is_clear) ++inside_miss;
                } else if (d1 > 1.06f) {
                    if (!is_clear) ++outside_hit;
                }
            }
        }
        if (inside_miss || outside_hit) {
            std::fprintf(stderr, "%s: inside_miss=%u outside_hit=%u\n", cases[11].name, inside_miss, outside_hit);
            ++failures;
        }
    }
    // Case 12: non-dyadic constant/flat attribute with uniform w=1 must be bit-exact on every pixel (0 ULP drift).
    // Case 16: differing w=(1.875,1,1) with non-dyadic constant color covers the whole target within 1e-5f (smooth
    // hardware perspective interpolation divides by 1/w per pixel, which can round by 1 ULP even when all vertices match).
    for (const std::size_t idx : {std::size_t(12), std::size_t(16)}) {
        unsigned drifted = 0;
        for (UINT i = 0; i < kExtent * kExtent; ++i) {
            bool ok = true;
            if (idx == 12) {
                ok = std::memcmp(&results[idx][i], &flat_color, sizeof(Pixel)) == 0;
            } else {
                for (int c = 0; c < 4; ++c)
                    if (!(std::fabs(results[idx][i][c] - flat_color[c]) <= 1e-5f)) ok = false;
            }
            if (!ok) {
                if (drifted++ < 4) {
                    const auto& g = results[idx][i];
                    std::fprintf(stderr, "%s: pixel (%u,%u) drifted: got %.9g,%.9g,%.9g,%.9g want %.9g,%.9g,%.9g,%.9g\n",
                                 cases[idx].name, i % kExtent, i / kExtent,
                                 double(g[0]), double(g[1]), double(g[2]), double(g[3]),
                                 double(flat_color[0]), double(flat_color[1]), double(flat_color[2]), double(flat_color[3]));
                }
            }
        }
        if (drifted) {
            std::fprintf(stderr, "%s: %u of %u pixels drifted from exact constant\n",
                         cases[idx].name, drifted, kExtent * kExtent);
            ++failures;
        }
    }
    // Cases 13..15: degenerate triangles draw zero pixels (target stays exact clear color)
    for (std::size_t idx = 13; idx <= 15; ++idx) {
        unsigned hit = 0;
        for (const Pixel& px : results[idx])
            if (std::memcmp(&px, &clear_px, sizeof(Pixel)) != 0) ++hit;
        if (hit) {
            std::fprintf(stderr, "%s: drew %u unexpected pixels\n", cases[idx].name, hit);
            ++failures;
        }
    }

    if (d.validation_errors()) {
        std::fprintf(stderr, "%u validation errors\n", d.validation_errors());
        ++failures;
    }
    if (failures) {
        std::fprintf(stderr, "d3d12_rect_list_corner_test: %d case(s) failed\n", failures);
        return 1;
    }
    std::printf("d3d12_rect_list_corner_test: all %zu corner-rule cases passed on GPU\n", cases.size());
    return 0;
}
