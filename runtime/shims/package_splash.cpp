#include "package_splash.h"
#include "title_window.h"

#include <Shlwapi.h>
#include <wincodec.h>
#include <wrl/client.h>
#include <xmllite.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <string_view>
#include <vector>

namespace rt::package_splash {
namespace {
using Microsoft::WRL::ComPtr;
std::filesystem::path image_path;

std::filesystem::path declared_image(const std::filesystem::path& package, const wchar_t* manifest,
                                     const wchar_t* element, const wchar_t* attribute) {
    ComPtr<IStream> stream;
    ComPtr<IXmlReader> reader;
    const auto config = package / manifest;
    if (FAILED(SHCreateStreamOnFileEx(config.c_str(), STGM_READ | STGM_SHARE_DENY_NONE, 0, FALSE, nullptr, &stream)) ||
        FAILED(CreateXmlReader(IID_PPV_ARGS(&reader), nullptr)) || FAILED(reader->SetInput(stream.Get())))
        return {};
    XmlNodeType type{};
    while (reader->Read(&type) == S_OK) {
        const wchar_t* name = nullptr;
        const wchar_t* value = nullptr;
        if (type != XmlNodeType_Element || FAILED(reader->GetLocalName(&name, nullptr)) ||
            std::wstring_view(name) != element ||
            reader->MoveToAttributeByName(attribute, nullptr) != S_OK ||
            FAILED(reader->GetValue(&value, nullptr)) || !*value)
            continue;
        const std::filesystem::path asset(value);
        if (asset.has_root_path() || asset.native().find(L':') != std::wstring::npos) return {};
        for (const auto& part : asset)
            if (part == L"..") return {};
        std::error_code ec;
        const auto root = std::filesystem::canonical(package, ec);
        if (ec) return {};
        const auto file = std::filesystem::canonical(root / asset, ec);
        if (ec) return {};
        const auto relative = file.lexically_relative(root);
        if (relative.empty() || *relative.begin() == L".." || !std::filesystem::is_regular_file(file, ec)) return {};
        return file;
    }
    return {};
}

std::vector<BYTE> decode(UINT width, UINT height) {
    ComPtr<IWICImagingFactory> factory;
    ComPtr<IWICBitmapDecoder> decoder;
    ComPtr<IWICBitmapFrameDecode> frame;
    UINT w = 0, h = 0;
    if (FAILED(CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&factory))) ||
        FAILED(factory->CreateDecoderFromFilename(image_path.c_str(), nullptr, GENERIC_READ, WICDecodeMetadataCacheOnDemand, &decoder)) ||
        FAILED(decoder->GetFrame(0, &frame)) || FAILED(frame->GetSize(&w, &h)) || !w || !h)
        return {};

    // Centre the package art at its original aspect ratio in the actual client area.
    const UINT draw_w = UINT(std::clamp<std::uint64_t>(std::uint64_t(height) * w / h, 1, width));
    const UINT draw_h = UINT(std::clamp<std::uint64_t>(std::uint64_t(width) * h / w, 1, height));
    ComPtr<IWICBitmapScaler> scaler;
    ComPtr<IWICFormatConverter> converter;
    if (FAILED(factory->CreateBitmapScaler(&scaler)) ||
        FAILED(scaler->Initialize(frame.Get(), draw_w, draw_h, WICBitmapInterpolationModeFant)) ||
        FAILED(factory->CreateFormatConverter(&converter)) ||
        FAILED(converter->Initialize(scaler.Get(), GUID_WICPixelFormat32bppPBGRA, WICBitmapDitherTypeNone, nullptr, 0,
                                     WICBitmapPaletteTypeCustom)))
        return {};
    const UINT pitch = width * 4;
    const UINT bytes = pitch * height;
    const UINT offset = (height - draw_h) / 2 * pitch + (width - draw_w) / 2 * 4;
    std::vector<BYTE> pixels(bytes);
    if (FAILED(converter->CopyPixels(nullptr, pitch, bytes - offset, pixels.data() + offset))) return {};

    return pixels;
}

std::vector<BYTE> load_pixels(UINT width, UINT height) {
    if (image_path.empty() || !width || !height || std::uint64_t(width) * height > (64u << 20)) return {};
    const HRESULT apartment = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    if (FAILED(apartment) && apartment != RPC_E_CHANGED_MODE) return {};
    std::vector<BYTE> pixels;
    try {
        pixels = decode(width, height);
    } catch (const std::bad_alloc&) {
        // Optional startup art must leave the first title frame available on low-memory hosts.
    }
    if (SUCCEEDED(apartment)) CoUninitialize();
    return pixels;
}

}  // namespace

void configure(const std::filesystem::path& package) {
    image_path.clear();
    wchar_t value[4]{};
    if (GetEnvironmentVariableW(L"LEONIDA_PACKAGE_SPLASH", value, 4) == 1 && value[0] == L'0') return;
    wchar_t window_mode[16]{};
    const DWORD length = GetEnvironmentVariableW(L"LEONIDA_PRESENT_WINDOW", window_mode, 16);
    if (length && length < 16 && (std::wstring_view(window_mode) == L"own" || std::wstring_view(window_mode) == L"hidden"))
        return;
    image_path = declared_image(package, L"MicrosoftGame.Config", L"ShellVisuals", L"SplashScreenImage");
    if (image_path.empty()) image_path = declared_image(package, L"appxmanifest.xml", L"SplashScreen", L"Image");
}

bool configured() { return !image_path.empty(); }

bool paint_window(HWND window) {
    static std::atomic<bool> painted{false};
    if (painted.load() || !configured() || !window || rt::d3d12::find_title_window(GetCurrentProcessId()) != window)
        return false;
    RECT client{};
    if (!GetClientRect(window, &client) || client.right <= 0 || client.bottom <= 0 || painted.exchange(true))
        return false;
    auto pixels = load_pixels(UINT(client.right), UINT(client.bottom));
    if (pixels.empty()) return false;
    BITMAPINFO bitmap{};
    bitmap.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    bitmap.bmiHeader.biWidth = client.right;
    bitmap.bmiHeader.biHeight = -client.bottom;
    bitmap.bmiHeader.biPlanes = 1;
    bitmap.bmiHeader.biBitCount = 32;
    bitmap.bmiHeader.biCompression = BI_RGB;
    if (GetWindowThreadProcessId(window, nullptr) == GetCurrentThreadId()) UpdateWindow(window);
    HDC dc = GetDC(window);
    if (!dc) return false;
    int lines = SetDIBitsToDevice(dc, 0, 0, client.right, client.bottom, 0, 0, 0, client.bottom,
                                  pixels.data(), &bitmap, DIB_RGB_COLORS);
    ReleaseDC(window, dc);
    return lines > 0;
}

ID3D12Resource* create_texture(ID3D12Device* device, UINT width, UINT height) {
    if (!device) return nullptr;
    const auto pixels = load_pixels(width, height);
    if (pixels.empty()) return nullptr;
    const UINT pitch = width * 4;
    const UINT bytes = pitch * height;
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = width;
    desc.Height = height;
    desc.DepthOrArraySize = desc.MipLevels = 1;
    desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    desc.SampleDesc.Count = 1;
    const D3D12_HEAP_PROPERTIES heap{D3D12_HEAP_TYPE_CUSTOM, D3D12_CPU_PAGE_PROPERTY_WRITE_BACK, D3D12_MEMORY_POOL_L0, 1, 1};
    ComPtr<ID3D12Resource> texture;
    if (FAILED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON,
                                               nullptr, IID_PPV_ARGS(&texture))) ||
        FAILED(texture->WriteToSubresource(0, nullptr, pixels.data(), pitch, bytes)))
        return nullptr;
    return texture.Detach();
}
}  // namespace rt::package_splash
