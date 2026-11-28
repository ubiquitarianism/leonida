#include "vk_test_device.h"
#include "d3d12_test_shaders.h"
#include "../shims/d3d12_com.h"
#include "../shims/d3d12_xs_device.h"
#include <cstdio>
#include <fstream>
using namespace d3d12test;
#define CHECK(x)                                                                                                       \
    do {                                                                                                               \
        if (!(x)) {                                                                                                    \
            std::fprintf(stderr, "line %d: %s\n", __LINE__, #x);                                                       \
            return 1;                                                                                                  \
        }                                                                                                              \
    } while (0)
int main(int argc, char **argv) {
    vktest::Device vk;
    const auto status = vk.initialize();
    if (status == vktest::DeviceStatus::unavailable) return 77;
    CHECK(status == vktest::DeviceStatus::ready);
    rt::d3d12::VulkanDeviceInfo info{};
    info.get_instance_proc = vk.instance_proc;
    info.instance = vk.instance;
    info.physical = vk.physical;
    info.device = vk.device;
    info.queue = vk.queue;
    info.queue_family = vk.family;
    info.buffer_device_address = true;
    ID3D12Device *dev = nullptr;
    CHECK(SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&dev))));
    Ref<ID3D12Device> xbox;
    CHECK(SUCCEEDED(
        rt::d3d12_xs::create_device_proxy(dev, __uuidof(ID3D12Device), reinterpret_cast<void **>(&xbox.p), true)));
    rt::d3d12::set_device_outer(dev, xbox.p);
    D3D12_HEAP_PROPERTIES heap{D3D12_HEAP_TYPE_DEFAULT};
    auto desc = buffer_desc(256);
    Ref<ID3D12Resource> constants;
    CHECK(SUCCEEDED(dev->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON,
                                                 nullptr, IID_PPV_ARGS(&constants.p))));
    Ref<ID3D12DescriptorHeap> descriptors;
    D3D12_DESCRIPTOR_HEAP_DESC hd{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 2, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
    CHECK(SUCCEEDED(dev->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&descriptors.p))));
    auto dst = descriptors->GetCPUDescriptorHandleForHeapStart();
    const UINT stride = dev->GetDescriptorHandleIncrementSize(hd.Type);
    D3D12_CONSTANT_BUFFER_VIEW_DESC cb{constants->GetGPUVirtualAddress(), 256};
    CHECK(cb.BufferLocation && cb.BufferLocation < (8ull << 20));
    xbox->CreateConstantBufferView(&cb, dst);
    dev->CreateConstantBufferView(&cb, {dst.ptr + stride});
    // The device writes every constant-buffer V# in the console's form of sixteen-byte rows; with LEONIDA_CBV_RAW=1
    // only a view made through the console device has rows and one made on the device itself is a byte-address view.
    wchar_t raw_env[2]{};
    const bool raw_cbv = GetEnvironmentVariableW(L"LEONIDA_CBV_RAW", raw_env, 2) == 1 && raw_env[0] == L'1';
    for (unsigned variant = 0; variant < 2; ++variant) {
        leonida::gfx10::BufferWords words{};
        std::memcpy(words.data(), reinterpret_cast<const void *>(dst.ptr + variant * stride), sizeof(words));
        leonida::gfx10::Buffer decoded{};
        CHECK(leonida::gfx10::decode(words, decoded) && decoded.address == cb.BufferLocation);
        const bool bytes = variant && raw_cbv;
        CHECK(decoded.stride == (bytes ? 0u : 16u) && decoded.num_records == (bytes ? 256u : 16u));
        CHECK(decoded.oob_select ==
              (bytes ? leonida::gfx10::OobSelect::raw : leonida::gfx10::OobSelect::structured_with_offset));
        if (argc == 3) {
            std::vector<std::uint8_t> image(static_cast<std::size_t>(cb.BufferLocation) + 256);
            std::memcpy(image.data() + 64, words.data(), sizeof(words));
            for (unsigned row = 0; row < 16; ++row)
                for (unsigned col = 0; col < 4; ++col) {
                    const std::uint32_t value = 100 + 10 * row + col;
                    std::memcpy(image.data() + cb.BufferLocation + 16 * row + 4 * col, &value, 4);
                }
            std::ofstream file(argv[1 + variant], std::ios::binary);
            CHECK(file.write(reinterpret_cast<const char *>(image.data()), image.size()));
        }
    }
    CHECK(!vk.validation_errors());
    std::puts(raw_cbv ? "Xbox CBV has sixteen complete 16-byte rows; byte-address view retains 256 bytes"
                      : "Xbox CBV and device CBV both have sixteen complete 16-byte rows");
    return 0;
}
