#pragma once

// A native pipeline description and catalog for the vertex and mesh pixel readback tests. The source
// programs are built by the tests. The extra row names a container whose actual HASH does not match.
#include "d3d12_test_shaders.h"
#include "../src/native_pso.h"

namespace d3d12test {
namespace np = leonida::native_pso;
inline constexpr np::Digest producer_digest{1}, ps_digest{2}, stale_digest{3};

inline bool write_native_catalog(const std::filesystem::path& dest, const char* producer_stage,
                                 const std::vector<std::uint8_t>& producer, const std::vector<std::uint8_t>& ps) {
    std::error_code ec;
    if (!std::filesystem::create_directory(dest / "dxbc", ec)) return false;
    std::ofstream index(dest / "spirv" / "native_hashes.txt");
    index << np::HashMap::kHeader << '\n';
    const auto add = [&](const char* stage, const np::Digest& digest, const std::vector<std::uint8_t>& program) {
        const std::string name = std::string(stage) + ".dxbc";
        const auto container = shader_container(program, &digest);
        std::ofstream file(dest / "dxbc" / name, std::ios::binary);
        file.write(reinterpret_cast<const char*>(container.data()), container.size());
        index << np::hex(digest) << '\t' << stage << "\t../dxbc/" << name << '\n';
        return file.good() && index.good();
    };
    if (!add(producer_stage, producer_digest, producer) || !add("ps", ps_digest, ps)) return false;
    index << np::hex(stale_digest) << '\t' << producer_stage << "\t../dxbc/" << producer_stage << ".dxbc\n";
    return index.good();
}

inline HRESULT create_native_pso(ID3D12Device* device, ID3D12RootSignature* root, DXGI_FORMAT target,
                                 const np::Digest& producer, ID3D12PipelineState** out) {
    std::array<std::uint8_t, np::kStateBHeader> state{};
    state[0x10] = static_cast<std::uint8_t>(target);
    std::memset(state.data() + 0x1c, 0xff, sizeof(UINT));
    state[0x25] = 1;
    state[0x26] = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    state[0x28] = 1;
    const auto record = [](const np::Digest& d) {
        std::array<std::uint8_t, 20> bytes{0x43};
        std::memcpy(bytes.data() + 4, d.data(), d.size());
        return bytes;
    };
    const auto v = record(producer), p = record(ps_digest);
    np::GraphicsDesc desc{};
    desc.root_signature = root;
    desc.state_b = {state.data(), state.size()};
    desc.stages[5].records = {v.data(), v.size()};
    desc.stages[2].records = {p.data(), p.size()};
    using Create = HRESULT(WINAPI*)(void*, const np::GraphicsDesc*, const IID*, void**);
    return reinterpret_cast<Create>(slot(device, 57))(device, &desc, &__uuidof(ID3D12PipelineState),
                                                      reinterpret_cast<void**>(out));
}

}  // namespace d3d12test
