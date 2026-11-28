// The arena copies of the programs shaders read their code base from (s_getpc_b64 and the constant tables in the
// shader's own FSSM program, docs/design.md): the PSOs made from one program share one copy, which lives while any of
// them does. Through the Vulkan device alone, with hand-built programs and GLSL modules written to the translator's
// contract:
//   - four compute PSOs of program A (uses_code_base: true) take A's bytes of the arena once, and all four read their
//     code base at the same console VA, where A's bytes are;
//   - a PSO of program B takes B's bytes at another VA; one of program C, whose index entry has no uses_code_base key
//     (an index from before the key, whose shader may read its code base), takes C's; one of program D
//     (uses_code_base: false) takes none;
//   - two graphics PSOs whose pixel shader is program E (uses_code_base: true), with different blend states, take E's
//     bytes once;
//   - releasing three of the four A PSOs keeps A's copy, its bytes intact, for the fourth; releasing that one returns
//     A's bytes, and a new A PSO makes a new copy.
// With --per-pso (LEONIDA_SHARE_CODE_BASE=0) every PSO has a copy of its own, as before the copies were shared.
// No game data. Set LEONIDA_VK_VALIDATION=1 to run under VK_LAYER_KHRONOS_validation (0 errors required).
#include "shader_fixture_path.h"
#include "test_temp_dir.h"
#include "vk_test_device.h"
#include "d3d12_test_shaders.h"
#include "../shims/d3d12_com.h"
#include "../src/shader_library.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#ifndef LEONIDA_TEST_CS_SPV
#error "LEONIDA_TEST_CS_SPV must name the compiled compute test shader"
#endif
#if !defined(LEONIDA_TEST_VS_FULLSCREEN_SPV) || !defined(LEONIDA_TEST_PS_SPV)
#error "LEONIDA_TEST_VS_FULLSCREEN_SPV and LEONIDA_TEST_PS_SPV must name the compiled draw test shaders"
#endif

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

using namespace d3d12test;

namespace {

// An FSSM program of `words` dwords after the magic, each distinct, so two programs of one seed and size are equal
// and any other two differ.
std::vector<std::uint8_t> program(std::uint32_t seed, std::uint32_t words) {
    std::vector<std::uint8_t> out = {'F', 'S', 'S', 'M'};
    for (std::uint32_t i = 0; i < words; ++i) put32(out, seed + i);
    return out;
}

// A copy's console VA and bytes, or {0, 0} when the PSO has none.
struct Copy {
    std::uint64_t va = 0, bytes = 0;
};
Copy copy_of(ID3D12PipelineState* pso) {
    Copy c;
    if (!rt::d3d12::pipeline_code_base(pso, c.va, c.bytes)) return {};
    return c;
}

// True when the arena holds `expected` at `va`.
bool arena_holds(ID3D12Device* device, std::uint64_t va, const std::vector<std::uint8_t>& expected) {
    std::vector<std::uint8_t> got(expected.size());
    return rt::d3d12::read_arena(device, va, got.data(), got.size()) && got == expected;
}

// A released PSO whose pipeline a compile worker is building (the warm-up thread builds compute ones ahead of their
// first dispatch) is deleted when that build returns (PipelineState::last_release), and its copy goes with it. Waits
// up to 60 s for the PSOs not yet deleted to be `live`.
bool settle(long live) {
    for (int i = 0; i < 6000 && rt::d3d12::live_pipeline_states() != live; ++i) Sleep(10);
    return rt::d3d12::live_pipeline_states() == live;
}

std::string entry(const char* name, const char* stage, const std::vector<std::uint8_t>& fssm, const char* extra) {
    return std::string("\"") + name + "\": {\"stage\": \"" + stage + "\", \"fssm_sha256\": \"" +
           leonida::sha256_hex(fssm) + "\", \"bindings\": [], \"push_constant_bytes\": 144" + extra + "}";
}

}  // namespace

int main(int argc, char** argv) {
    const bool per_pso = argc == 2 && std::strcmp(argv[1], "--per-pso") == 0;
    if (argc != 1 && !per_pso) {
        std::fputs("usage: d3d12_code_base_share_test [--per-pso]\n", stderr);
        return 2;
    }
    if (per_pso) {
        char value[4] = {};
        if (GetEnvironmentVariableA("LEONIDA_SHARE_CODE_BASE", value, sizeof value) != 1 || value[0] != '0') {
            std::fputs("--per-pso needs LEONIDA_SHARE_CODE_BASE=0\n", stderr);
            return 2;
        }
    }

    // The programs, of distinct sizes so each count below has one reading, and the index that maps them to modules.
    const auto a = program(0x41000000u, 1000), b = program(0x42000000u, 600), c = program(0x43000000u, 300),
               d = program(0x44000000u, 200), e = program(0x45000000u, 500), vs = program(0x56000000u, 8);
    const testfiles::ScratchDir scratch("leonida_code_base_share_test");
    const auto& library = scratch.path();
    // The pipeline cache and journal of the warm-up's builds go to the scratch folder, not the working directory.
    if (!GetEnvironmentVariableW(L"LEONIDA_PIPELINE_CACHE_DIR", nullptr, 0))
        EXPECT(SetEnvironmentVariableW(L"LEONIDA_PIPELINE_CACHE_DIR", (library / "cache").c_str()));
    EXPECT(testfiles::copy_file(vktest::shader_fixture_path(LEONIDA_TEST_CS_SPV), library / "cs_a.spv"));
    EXPECT(testfiles::copy_file(vktest::shader_fixture_path(LEONIDA_TEST_CS_SPV), library / "cs_b.spv"));
    EXPECT(testfiles::copy_file(vktest::shader_fixture_path(LEONIDA_TEST_CS_SPV), library / "cs_c.spv"));
    EXPECT(testfiles::copy_file(vktest::shader_fixture_path(LEONIDA_TEST_CS_SPV), library / "cs_d.spv"));
    EXPECT(testfiles::copy_file(vktest::shader_fixture_path(LEONIDA_TEST_VS_FULLSCREEN_SPV), library / "vs_e.spv"));
    EXPECT(testfiles::copy_file(vktest::shader_fixture_path(LEONIDA_TEST_PS_SPV), library / "ps_e.spv"));
    {
        std::ofstream index(library / "index.json", std::ios::binary);
        index << "{" << entry("cs_a", "cs", a, ", \"local_size\": [64, 1, 1], \"uses_code_base\": true") << ",\n "
              << entry("cs_b", "cs", b, ", \"local_size\": [64, 1, 1], \"uses_code_base\": true") << ",\n "
              << entry("cs_c", "cs", c, ", \"local_size\": [64, 1, 1]") << ",\n "
              << entry("cs_d", "cs", d, ", \"local_size\": [64, 1, 1], \"uses_code_base\": false") << ",\n "
              << entry("vs_e", "vs", vs, ", \"input_signature\": []") << ",\n "
              << entry("ps_e", "ps", e, ", \"outputs\": [0], \"uses_code_base\": true") << "}";
        index.close();
        EXPECT(index);
    }
    // The index as the runtime reads it: C's entry has no uses_code_base key, so its use is not known.
    {
        leonida::ShaderLibrary check;
        EXPECT(check.load(library.string()));
        const auto* entry_c = check.find("cs", leonida::sha256_hex(c));
        EXPECT(entry_c && !entry_c->code_base_known && !entry_c->uses_code_base);
        const auto* entry_d = check.find("cs", leonida::sha256_hex(d));
        EXPECT(entry_d && entry_d->code_base_known && !entry_d->uses_code_base);
    }

    {
        vktest::Device vk;
        vk.request_buffer_device_address = true;
        const auto status = vk.initialize();
        if (status == vktest::DeviceStatus::unavailable) return 77;  // no Vulkan GPU: ctest skips
        EXPECT(status == vktest::DeviceStatus::ready);
        if (!vk.buffer_device_address) {
            std::puts("SKIP: the device lacks bufferDeviceAddress, which the GPU address arena needs");
            return 77;
        }
        rt::d3d12::VulkanDeviceInfo info{};
        info.get_instance_proc = vk.instance_proc;
        info.instance = vk.instance;
        info.physical = vk.physical;
        info.device = vk.device;
        info.queue = vk.queue;
        info.queue_family = vk.family;
        info.buffer_device_address = vk.buffer_device_address;
        info.subgroup = vk.subgroup;
        info.shader_dir = library.string();
        Ref<ID3D12Device> device;
        EXPECT(SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&device.p))));
        if (rt::d3d12::device_push_constants_need_fallback(device.p)) {
            std::printf("SKIP: the PSOs need 144 push-constant bytes and this GPU has %u\n",
                        rt::d3d12::device_max_push_constants_size(device.p));
            return 77;
        }
        EXPECT(rt::d3d12::device_arena_size(device.p) > 0);
        EXPECT(rt::d3d12::live_pipeline_states() == 0);  // settle() counts this test's PSOs only
        const std::uint64_t before = rt::d3d12::arena_shader_bytes(device.p);
        const auto used = [&] { return rt::d3d12::arena_shader_bytes(device.p) - before; };

        // Root signatures: one root UAV for compute; two constants and a root CBV for the draw shaders.
        D3D12_ROOT_PARAMETER uav{};
        uav.ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
        D3D12_VERSIONED_ROOT_SIGNATURE_DESC compute_root_desc{};
        compute_root_desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_0;
        compute_root_desc.Desc_1_0 = {1, &uav, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
        D3D12_ROOT_PARAMETER draw_params[2] = {};
        draw_params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        draw_params[0].Constants = {0, 0, 2};
        draw_params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
        draw_params[1].Descriptor = {1, 0};
        D3D12_VERSIONED_ROOT_SIGNATURE_DESC draw_root_desc{};
        draw_root_desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_0;
        draw_root_desc.Desc_1_0 = {2, draw_params, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
        Ref<ID3D12RootSignature> compute_root, draw_root;
        for (auto [desc, root] : {std::pair{&compute_root_desc, &compute_root}, std::pair{&draw_root_desc, &draw_root}}) {
            const auto blob = serialize_root_signature(*desc);
            EXPECT(!blob.empty());
            EXPECT(SUCCEEDED(device->CreateRootSignature(0, blob.data(), blob.size(), IID_PPV_ARGS(&root->p))));
        }

        const auto container_a = shader_container(a), container_b = shader_container(b),
                   container_c = shader_container(c), container_d = shader_container(d),
                   container_e = shader_container(e), container_vs = shader_container(vs);
        const auto make_compute = [&](const std::vector<std::uint8_t>& container, Ref<ID3D12PipelineState>& pso) {
            D3D12_COMPUTE_PIPELINE_STATE_DESC desc{};
            desc.pRootSignature = compute_root.p;
            desc.CS = {container.data(), container.size()};
            return SUCCEEDED(device->CreateComputePipelineState(&desc, IID_PPV_ARGS(&pso.p))) && pso.p;
        };
        const auto make_graphics = [&](bool blend, Ref<ID3D12PipelineState>& pso) {
            D3D12_GRAPHICS_PIPELINE_STATE_DESC desc{};
            desc.pRootSignature = draw_root.p;
            desc.VS = {container_vs.data(), container_vs.size()};
            desc.PS = {container_e.data(), container_e.size()};
            auto& target = desc.BlendState.RenderTarget[0];
            target.RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
            if (blend) {
                target.BlendEnable = TRUE;
                target.SrcBlend = target.SrcBlendAlpha = D3D12_BLEND_ONE;
                target.DestBlend = target.DestBlendAlpha = D3D12_BLEND_ONE;
                target.BlendOp = target.BlendOpAlpha = D3D12_BLEND_OP_ADD;
            }
            desc.SampleMask = 0xffffffff;
            desc.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
            desc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
            desc.RasterizerState.DepthClipEnable = TRUE;
            desc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
            desc.NumRenderTargets = 1;
            desc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
            desc.SampleDesc.Count = 1;
            return SUCCEEDED(device->CreateGraphicsPipelineState(&desc, IID_PPV_ARGS(&pso.p))) && pso.p;
        };
        const std::uint64_t copies = per_pso ? 4 : 1;  // of A while four PSOs of it live
        const std::uint64_t draw_copies = per_pso ? 2 : 1;

        // Four PSOs of A: one copy of A's bytes, at one VA, which all four push as their code base.
        Ref<ID3D12PipelineState> pso_a[4];
        for (auto& pso : pso_a) EXPECT(make_compute(container_a, pso));
        std::printf("4 compute PSOs of one %zu-byte program: %llu bytes of the arena hold program copies\n", a.size(),
                    static_cast<unsigned long long>(used()));
        EXPECT(used() == copies * a.size());
        const Copy first = copy_of(pso_a[0].p);
        EXPECT(first.bytes == a.size());
        for (const auto& pso : pso_a) {
            const Copy copy = copy_of(pso.p);
            EXPECT(copy.bytes == a.size() && arena_holds(device.p, copy.va, a));
            EXPECT(per_pso ? (&pso == &pso_a[0] || copy.va != first.va) : copy.va == first.va);
        }

        // B: a copy of its own, elsewhere. C: copied too, since its entry does not say. D: no copy.
        Ref<ID3D12PipelineState> pso_b, pso_c, pso_d;
        EXPECT(make_compute(container_b, pso_b) && make_compute(container_c, pso_c) && make_compute(container_d, pso_d));
        const Copy copy_b = copy_of(pso_b.p), copy_c = copy_of(pso_c.p);
        EXPECT(copy_b.bytes == b.size() && copy_b.va != first.va && arena_holds(device.p, copy_b.va, b));
        EXPECT(copy_c.bytes == c.size() && arena_holds(device.p, copy_c.va, c));
        EXPECT(copy_of(pso_d.p).bytes == 0);
        EXPECT(used() == copies * a.size() + b.size() + c.size());

        // Two graphics PSOs whose pixel shader reads its code base: one copy of E's bytes (one per PSO with --per-pso),
        // none of the VS's.
        Ref<ID3D12PipelineState> draw[2];
        EXPECT(make_graphics(false, draw[0]) && make_graphics(true, draw[1]));
        const Copy draw_copy = copy_of(draw[0].p), draw_copy2 = copy_of(draw[1].p);
        EXPECT(draw_copy.bytes == e.size() && arena_holds(device.p, draw_copy.va, e));
        EXPECT(draw_copy2.bytes == e.size() && arena_holds(device.p, draw_copy2.va, e));
        EXPECT(per_pso ? draw_copy2.va != draw_copy.va : draw_copy2.va == draw_copy.va);
        EXPECT(used() == copies * a.size() + b.size() + c.size() + draw_copies * e.size());
        draw[0].reset();
        EXPECT(settle(8) && arena_holds(device.p, draw_copy2.va, e));
        draw[1].reset();
        EXPECT(settle(7) && used() == copies * a.size() + b.size() + c.size());

        // Three of A's PSOs go: the fourth keeps its copy, A's bytes still where it reads them.
        const Copy last = copy_of(pso_a[3].p);
        for (int i = 0; i < 3; ++i) pso_a[i].reset();
        EXPECT(settle(4) && used() == a.size() + b.size() + c.size());
        EXPECT(copy_of(pso_a[3].p).va == last.va && arena_holds(device.p, last.va, a));
        // The last one goes: A's bytes are returned to the arena.
        pso_a[3].reset();
        EXPECT(settle(3) && used() == b.size() + c.size());
        // A new PSO of A makes a new copy, which holds A's bytes.
        Ref<ID3D12PipelineState> again;
        EXPECT(make_compute(container_a, again));
        EXPECT(used() == a.size() + b.size() + c.size() && arena_holds(device.p, copy_of(again.p).va, a));
        again.reset();
        pso_b.reset();
        pso_c.reset();
        pso_d.reset();
        EXPECT(settle(0) && used() == 0);
        std::printf("%s: 4 PSOs of one program held %llu copies, 2 graphics PSOs %llu; all bytes returned\n",
                    per_pso ? "per PSO (LEONIDA_SHARE_CODE_BASE=0)" : "shared", static_cast<unsigned long long>(copies),
                    static_cast<unsigned long long>(draw_copies));
    }
    EXPECT(vktest::validation_error_count() == 0);  // includes device teardown
    std::puts("d3d12_code_base_share_test: PASS");
    return 0;
}
