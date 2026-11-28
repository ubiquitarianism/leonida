// Tests for console descriptor heap T# binding rules and per-reason refusal tracking (RT-UNBOUND-TSHARP-REASONS).
// Verifies:
//   1. 1D rule: dimension and layer count compatibility ("T# type or samples do not fit the texture").
//   2. Array rule: layer range bounds ("T# names layers the texture lacks").
//   3. Cube rule: minimum layer requirement ("cube T# on fewer than six layers").
//   4. MSAA rule: sample count matching ("T# type or samples do not fit the texture").
//   5. Unbound T# reason tracking: distinct slots and total refusal counts tracked per reason.
#include "../shims/d3d12_com.h"
#include "../shims/d3d12_xs_device.h"
#include "../src/gfx10_descriptors.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace {

int g_checks = 0;
int g_failures = 0;

void check(bool condition, const char* what) {
    ++g_checks;
    if (!condition) {
        ++g_failures;
        std::fprintf(stderr, "FAIL: %s\n", what);
    }
}

template <class T>
struct Ref {
    T* p = nullptr;
    ~Ref() {
        if (p) p->Release();
    }
    T** put() {
        if (p) {
            p->Release();
            p = nullptr;
        }
        return &p;
    }
    T* operator->() const { return p; }
};

}  // namespace

int main() {
    namespace g = leonida::gfx10;

    // Test 1: Unbound T# statistics and per-reason tracking
    {
        rt::d3d12::reset_unbound_tsharp_stats();
        std::array<std::uint32_t, 8> dummy_words{};

        const char* reason_type = "T# type or samples do not fit the texture";
        const char* reason_layers = "T# names layers the texture lacks";

        // Slot 0x00 refused for type
        rt::d3d12::note_refused_tsharp_for_test(0x10000, 0x00, dummy_words, reason_type);
        // Slot 0x20 refused for type
        rt::d3d12::note_refused_tsharp_for_test(0x10000, 0x20, dummy_words, reason_type);
        // Slot 0x00 refused again for type (repeated scan)
        rt::d3d12::note_refused_tsharp_for_test(0x10000, 0x00, dummy_words, reason_type);
        // Slot 0x40 refused for layers
        rt::d3d12::note_refused_tsharp_for_test(0x10000, 0x40, dummy_words, reason_layers);

        const auto stats = rt::d3d12::unbound_tsharp_stats();
        check(stats.total_distinct_slots == 3, "total distinct slots is 3");
        check(stats.total_refusals == 4, "total refusals is 4");

        const auto it_type = stats.by_reason.find(reason_type);
        check(it_type != stats.by_reason.end(), "reason_type recorded in stats");
        if (it_type != stats.by_reason.end()) {
            check(it_type->second.distinct_slots == 2, "type reason distinct slots is 2");
            check(it_type->second.total_refusals == 3, "type reason total refusals is 3");
        }

        const auto it_layers = stats.by_reason.find(reason_layers);
        check(it_layers != stats.by_reason.end(), "reason_layers recorded in stats");
        if (it_layers != stats.by_reason.end()) {
            check(it_layers->second.distinct_slots == 1, "layers reason distinct slots is 1");
            check(it_layers->second.total_refusals == 1, "layers reason total refusals is 1");
        }
    }

    // Initialize standalone Vulkan D3D12 device for resource binding rule tests
    Ref<ID3D12Device> device;
    if (FAILED(rt::d3d12::create_vulkan_device_standalone(IID_PPV_ARGS(device.put())))) {
        std::puts("SKIP: no Vulkan D3D12 device");
        return 77;
    }

    const D3D12_HEAP_PROPERTIES heap_props{
        .Type = D3D12_HEAP_TYPE_DEFAULT,
        .CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
        .MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN,
        .CreationNodeMask = 1,
        .VisibleNodeMask = 1,
    };

    // Test 2: 1D Texture binding rules
    {
        D3D12_RESOURCE_DESC desc{};
        desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE1D;
        desc.Width = 64;
        desc.Height = 1;
        desc.DepthOrArraySize = 1;
        desc.MipLevels = 1;
        desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        desc.SampleDesc.Count = 1;
        desc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;

        Ref<ID3D12Resource> res_1d;
        check(SUCCEEDED(device->CreateCommittedResource(&heap_props, D3D12_HEAP_FLAG_NONE, &desc,
                                                        D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                        IID_PPV_ARGS(res_1d.put()))),
              "created 1D texture");

        const std::uint64_t addr = rt::d3d12::resource_image_address(res_1d.p);
        check(addr != 0, "1D texture has registered image address");

        // Case 2a: T# specifies 2D type for a 1D texture -> refuse type_fits
        {
            g::Image img{};
            img.address = addr;
            img.type = g::ImageType::image_2d;
            img.format = g::format_from_vulkan(VK_FORMAT_R8G8B8A8_UNORM);
            img.width = 64;
            img.height = 1;
            img.layer_count = 1;
            img.mip_count = 1;
            const auto words = g::encode(img);

            const char* why = nullptr;
            const auto result = rt::d3d12::test_bind_title_image(device.p, words, &why);
            check(result == rt::d3d12::TitleImage::refused, "2D T# on 1D texture refused");
            check(why && std::string(why) == "T# type or samples do not fit the texture",
                  "2D T# on 1D texture explains type mismatch");
        }

        // Case 2b: T# specifies 1D type but layer_count > 1 -> refuse type_fits
        {
            g::Image img{};
            img.address = addr;
            img.type = g::ImageType::image_1d;
            img.format = g::format_from_vulkan(VK_FORMAT_R8G8B8A8_UNORM);
            img.width = 64;
            img.height = 1;
            img.layer_count = 2;
            img.mip_count = 1;
            const auto words = g::encode(img);

            const char* why = nullptr;
            const auto result = rt::d3d12::test_bind_title_image(device.p, words, &why);
            check(result == rt::d3d12::TitleImage::refused, "1D T# with layer_count > 1 on 1D texture refused");
            check(why && std::string(why) == "T# type or samples do not fit the texture",
                  "1D T# with layer_count > 1 explains type mismatch");
        }

        // Case 2c: T# specifies 1D type and layer_count == 1 -> bound
        {
            g::Image img{};
            img.address = addr;
            img.type = g::ImageType::image_1d;
            img.format = g::format_from_vulkan(VK_FORMAT_R8G8B8A8_UNORM);
            img.width = 64;
            img.height = 1;
            img.layer_count = 1;
            img.mip_count = 1;
            const auto words = g::encode(img);

            const char* why = nullptr;
            const auto result = rt::d3d12::test_bind_title_image(device.p, words, &why);
            check(result == rt::d3d12::TitleImage::bound, "matching 1D T# bound successfully");
        }
    }

    // Test 3: 2D Array and Cube binding rules
    {
        D3D12_RESOURCE_DESC desc{};
        desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        desc.Width = 64;
        desc.Height = 64;
        desc.DepthOrArraySize = 4;  // 4 layers
        desc.MipLevels = 2;
        desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        desc.SampleDesc.Count = 1;
        desc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;

        Ref<ID3D12Resource> res_array;
        check(SUCCEEDED(device->CreateCommittedResource(&heap_props, D3D12_HEAP_FLAG_NONE, &desc,
                                                        D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                        IID_PPV_ARGS(res_array.put()))),
              "created 2D array texture");

        const std::uint64_t addr = rt::d3d12::resource_image_address(res_array.p);
        check(addr != 0, "2D array texture has registered image address");

        // Case 3a: T# names layers the texture lacks (first_layer + layer_count > 4) -> refuse layers
        {
            g::Image img{};
            img.address = addr;
            img.type = g::ImageType::image_2d_array;
            img.format = g::format_from_vulkan(VK_FORMAT_R8G8B8A8_UNORM);
            img.width = 64;
            img.height = 64;
            img.first_layer = 2;
            img.layer_count = 3;  // 2 + 3 = 5 > 4
            img.mip_count = 1;
            const auto words = g::encode(img);

            const char* why = nullptr;
            const auto result = rt::d3d12::test_bind_title_image(device.p, words, &why);
            check(result == rt::d3d12::TitleImage::refused, "T# with excessive layers refused");
            check(why && std::string(why) == "T# names layers the texture lacks",
                  "excessive layers explains layers mismatch");
        }

        // Case 3b: Cube T# on texture with fewer than six layers (desc.DepthOrArraySize = 4 < 6) -> refuse cube
        {
            g::Image img{};
            img.address = addr;
            img.type = g::ImageType::cube;
            img.format = g::format_from_vulkan(VK_FORMAT_R8G8B8A8_UNORM);
            img.width = 64;
            img.height = 64;
            img.first_layer = 0;
            img.layer_count = 6;  // cube needs at least 6 faces, but texture only has 4 layers
            img.mip_count = 1;
            const auto words = g::encode(img);

            const char* why = nullptr;
            const auto result = rt::d3d12::test_bind_title_image(device.p, words, &why);
            check(result == rt::d3d12::TitleImage::refused, "cube T# on 4 layers refused");
            check(why && std::string(why) == "cube T# on fewer than six layers",
                  "cube T# explains fewer than six layers");
        }

        // Case 3c: T# names valid sub-array (first_layer = 1, layer_count = 2 <= 4) -> bound
        {
            g::Image img{};
            img.address = addr;
            img.type = g::ImageType::image_2d_array;
            img.format = g::format_from_vulkan(VK_FORMAT_R8G8B8A8_UNORM);
            img.width = 64;
            img.height = 64;
            img.first_layer = 1;
            img.layer_count = 2;
            img.mip_count = 1;
            const auto words = g::encode(img);

            const char* why = nullptr;
            const auto result = rt::d3d12::test_bind_title_image(device.p, words, &why);
            check(result == rt::d3d12::TitleImage::bound, "valid sub-array T# bound successfully");
        }

        // Case 3d: T# names mips the texture lacks (first_mip + mip_count > 2) -> refuse mips
        {
            g::Image img{};
            img.address = addr;
            img.type = g::ImageType::image_2d_array;
            img.format = g::format_from_vulkan(VK_FORMAT_R8G8B8A8_UNORM);
            img.width = 64;
            img.height = 64;
            img.first_layer = 0;
            img.layer_count = 1;
            img.first_mip = 1;
            img.mip_count = 2;  // 1 + 2 = 3 > 2
            const auto words = g::encode(img);

            const char* why = nullptr;
            const auto result = rt::d3d12::test_bind_title_image(device.p, words, &why);
            check(result == rt::d3d12::TitleImage::refused, "T# with excessive mips refused");
            check(why && std::string(why) == "T# names mips the texture lacks",
                  "excessive mips explains mips mismatch");
        }
    }

    // Test 4: MSAA Texture binding rules
    {
        D3D12_RESOURCE_DESC desc{};
        desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        desc.Width = 64;
        desc.Height = 64;
        desc.DepthOrArraySize = 1;
        desc.MipLevels = 1;
        desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        desc.SampleDesc.Count = 4;  // 4x MSAA
        desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
        desc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;

        Ref<ID3D12Resource> res_msaa;
        check(SUCCEEDED(device->CreateCommittedResource(&heap_props, D3D12_HEAP_FLAG_NONE, &desc,
                                                        D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                        IID_PPV_ARGS(res_msaa.put()))),
              "created 4x MSAA texture");

        const std::uint64_t addr = rt::d3d12::resource_image_address(res_msaa.p);
        check(addr != 0, "MSAA texture has registered image address");

        // Case 4a: Non-MSAA T# on MSAA texture -> refuse type_fits
        {
            g::Image img{};
            img.address = addr;
            img.type = g::ImageType::image_2d;
            img.format = g::format_from_vulkan(VK_FORMAT_R8G8B8A8_UNORM);
            img.width = 64;
            img.height = 64;
            img.layer_count = 1;
            img.mip_count = 1;
            const auto words = g::encode(img);

            const char* why = nullptr;
            const auto result = rt::d3d12::test_bind_title_image(device.p, words, &why);
            check(result == rt::d3d12::TitleImage::refused, "non-MSAA T# on MSAA texture refused");
            check(why && std::string(why) == "T# type or samples do not fit the texture",
                  "non-MSAA T# explains type or samples mismatch");
        }

        // Case 4b: MSAA T# with mismatched samples (samples = 2 != 4) -> refuse type_fits
        {
            g::Image img{};
            img.address = addr;
            img.type = g::ImageType::image_2d_msaa;
            img.format = g::format_from_vulkan(VK_FORMAT_R8G8B8A8_UNORM);
            img.width = 64;
            img.height = 64;
            img.samples = 2;
            img.layer_count = 1;
            img.mip_count = 1;
            const auto words = g::encode(img);

            const char* why = nullptr;
            const auto result = rt::d3d12::test_bind_title_image(device.p, words, &why);
            check(result == rt::d3d12::TitleImage::refused, "mismatched samples T# refused");
            check(why && std::string(why) == "T# type or samples do not fit the texture",
                  "mismatched samples explains type or samples mismatch");
        }

        // Case 4c: Matching 4x MSAA T# -> bound
        {
            g::Image img{};
            img.address = addr;
            img.type = g::ImageType::image_2d_msaa;
            img.format = g::format_from_vulkan(VK_FORMAT_R8G8B8A8_UNORM);
            img.width = 64;
            img.height = 64;
            img.samples = 4;
            img.layer_count = 1;
            img.mip_count = 1;
            const auto words = g::encode(img);

            const char* why = nullptr;
            const auto result = rt::d3d12::test_bind_title_image(device.p, words, &why);
            check(result == rt::d3d12::TitleImage::bound, "matching 4x MSAA T# bound successfully");
        }
    }

    // Test 5: Unknown texture address and invalid T# words
    {
        // Unknown address
        {
            g::Image img{};
            img.address = 0xdeadbeef00ull;
            img.type = g::ImageType::image_2d;
            img.format = g::format_from_vulkan(VK_FORMAT_R8G8B8A8_UNORM);
            img.width = 64;
            img.height = 64;
            img.layer_count = 1;
            img.mip_count = 1;
            const auto words = g::encode(img);

            const char* why = nullptr;
            const auto result = rt::d3d12::test_bind_title_image(device.p, words, &why);
            check(result == rt::d3d12::TitleImage::unknown, "unknown address returns TitleImage::unknown");
            check(why && std::string(why) == "no texture at T# address",
                  "unknown address explains no texture at T# address");
        }

        // Not a valid T# (decode failure)
        {
            std::array<std::uint32_t, 8> raw_words{0x1, 0x2, 0x3, 0x0, 0x0, 0x0, 0x0, 0x0};
            const char* why = nullptr;
            const auto result = rt::d3d12::test_bind_title_image(device.p, raw_words, &why);
            check(result == rt::d3d12::TitleImage::not_tsharp, "invalid words return TitleImage::not_tsharp");
            check(why && std::string(why) == "not a valid T# (swizzle, levels or layers)",
                  "invalid words explains not a valid T#");
        }
    }

    std::printf("d3d12_unbound_tsharp_rules_test: %d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
