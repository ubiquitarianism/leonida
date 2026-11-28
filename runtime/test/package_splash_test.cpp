// Exercise manifest lookup, WIC scaling and upload with a synthetic 2x1 red bitmap.
#include "vk_test_device.h"
#include "../shims/d3d12_com.h"
#include "../shims/package_splash.h"

#include <wrl/client.h>
#include <array>
#include <cstdio>
#include <filesystem>
#include <fstream>

#define EXPECT(expr) do { if (!(expr)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #expr); return 1; } } while (0)

int main() {
    namespace fs = std::filesystem;
    namespace splash = rt::package_splash;
    using Microsoft::WRL::ComPtr;
    const auto dir = fs::current_path() / (L"splash-test-" + std::to_wstring(GetCurrentProcessId()));
    EXPECT(fs::create_directory(dir));
    struct Cleanup {
        fs::path dir;
        ~Cleanup() { std::error_code ec; fs::remove_all(dir, ec); }
    } cleanup{dir};

    SetEnvironmentVariableW(L"LEONIDA_PACKAGE_SPLASH", nullptr);
    splash::configure(dir);
    EXPECT(!splash::configured());
    BITMAPINFOHEADER info{};
    info.biSize = sizeof(info);
    info.biWidth = 2;
    info.biHeight = info.biPlanes = 1;
    info.biBitCount = 24;
    BITMAPFILEHEADER file{};
    file.bfType = 0x4d42;
    file.bfOffBits = sizeof(file) + sizeof(info);
    file.bfSize = file.bfOffBits + 8;
    const unsigned char row[8]{0, 0, 255, 0, 0, 255, 0, 0};
    std::ofstream(dir / L"art.bmp", std::ios::binary).write(reinterpret_cast<const char*>(&file), sizeof(file))
        .write(reinterpret_cast<const char*>(&info), sizeof(info)).write(reinterpret_cast<const char*>(row), sizeof(row));
    const auto config = dir / L"MicrosoftGame.Config";
    std::ofstream(config) << "<Game><ShellVisuals SplashScreenImage='../" << dir.filename().string()
                          << "/art.bmp'/></Game>";
    splash::configure(dir);
    EXPECT(!splash::configured());
    std::ofstream(config) << "<Game><!-- <ShellVisuals SplashScreenImage='missing'/> -->"
                             "<ShellVisuals SplashScreenImage='art.bmp'/></Game>";
    splash::configure(dir);
    EXPECT(splash::configured());
    SetEnvironmentVariableW(L"LEONIDA_PACKAGE_SPLASH", L"0");
    splash::configure(dir);
    EXPECT(!splash::configured());
    SetEnvironmentVariableW(L"LEONIDA_PACKAGE_SPLASH", nullptr);
    fs::remove(config);
    std::ofstream(dir / L"appxmanifest.xml") << "<Package xmlns:u='urn:test'><u:SplashScreen Image='art.bmp'/></Package>";
    splash::configure(dir);
    EXPECT(splash::configured());

    vktest::Device vk;
    const auto status = vk.initialize();
    if (status == vktest::DeviceStatus::unavailable) return 77;
    EXPECT(status == vktest::DeviceStatus::ready);
    rt::d3d12::VulkanDeviceInfo gpu{};
    gpu.get_instance_proc = vk.instance_proc;
    gpu.instance = vk.instance;
    gpu.physical = vk.physical;
    gpu.device = vk.device;
    gpu.queue = vk.queue;
    gpu.queue_family = vk.family;
    gpu.host_image_copy = vk.host_image_copy;
    ComPtr<ID3D12Device> device;
    EXPECT(SUCCEEDED(rt::d3d12::create_vulkan_device(gpu, IID_PPV_ARGS(&device))));
    ComPtr<ID3D12Resource> texture;
    texture.Attach(splash::create_texture(device.Get(), 4, 4));
    EXPECT(texture);
    std::array<UINT32, 16> pixels{};
    EXPECT(SUCCEEDED(texture->ReadFromSubresource(pixels.data(), 16, 64, 0, nullptr)));
    for (unsigned i = 0; i < pixels.size(); ++i)
        EXPECT((pixels[i] & 0xffffff) == ((i >= 4 && i < 12) ? 0xff0000u : 0u));
    texture.Attach(splash::create_texture(device.Get(), 6, 2));
    EXPECT(texture);
    EXPECT(SUCCEEDED(texture->ReadFromSubresource(pixels.data(), 24, 48, 0, nullptr)));
    for (unsigned i = 0; i < 12; ++i)
        EXPECT((pixels[i] & 0xffffff) == ((i % 6 >= 1 && i % 6 <= 4) ? 0xff0000u : 0u));
    std::puts("package splash: manifest, rollback, aspect fit and Vulkan readback passed");
    return 0;
}
