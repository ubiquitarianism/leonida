// rt::d3d12::vulkan_vram_budget on the standalone Vulkan-backed device: the GPU memory in use and the budget the
// title HUD shows (shims/title_hud.h), from VK_EXT_memory_budget. The device enables the extension exactly when the
// GPU has it; then, with a 64 MiB default-heap buffer alive, the device-local budget and the usage are non-zero.
// The device publishes that figure for the HUD and the dev overlay (src/memory_figures.h) and withdraws it when it
// is released. Without the extension the figure is not available, and the test skips (77) after checking that.
// Also skips without a Vulkan GPU. No shaders or game data.
#include "../shims/d3d12_com.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

namespace {

template<class T>
struct Ref {
    T* p = nullptr;
    ~Ref() {
        if (p) p->Release();
    }
};

constexpr std::uint64_t kMiB = 1ull << 20;

bool physical_has_memory_budget(const rt::d3d12::VulkanDeviceInfo& info) {
    const auto enumerate = reinterpret_cast<PFN_vkEnumerateDeviceExtensionProperties>(
        info.get_instance_proc(info.instance, "vkEnumerateDeviceExtensionProperties"));
    if (!enumerate) return false;
    std::uint32_t count = 0;
    if (enumerate(info.physical, nullptr, &count, nullptr) != VK_SUCCESS) return false;
    std::vector<VkExtensionProperties> extensions(count);
    if (enumerate(info.physical, nullptr, &count, extensions.data()) != VK_SUCCESS) return false;
    for (const auto& e : extensions)
        if (!std::strcmp(e.extensionName, VK_EXT_MEMORY_BUDGET_EXTENSION_NAME)) return true;
    return false;
}

void print(const char* when, const leonida::VramBudget& b) {
    std::printf("%s: %s, device-local %llu of %llu MiB, other heaps %llu MiB\n", when,
                b.available ? "available" : "not available", (unsigned long long)(b.local_usage / kMiB),
                (unsigned long long)(b.local_budget / kMiB), (unsigned long long)(b.shared_usage / kMiB));
}

}  // namespace

int main() {
    Ref<ID3D12Device> device;
    if (FAILED(rt::d3d12::create_vulkan_device_standalone(IID_PPV_ARGS(&device.p))) || !device.p) {
        std::printf("SKIP: no Vulkan device\n");
        return 77;
    }
    const rt::d3d12::VulkanDeviceInfo* info = rt::d3d12::vulkan_device_info(device.p);
    EXPECT(info && info->get_instance_proc && info->instance && info->physical);
    EXPECT(!rt::d3d12::vulkan_vram_budget(nullptr).available);

    // Enabled exactly when the GPU has it.
    const bool supported = physical_has_memory_budget(*info);
    EXPECT(info->memory_budget == supported);
    // What the title HUD and the dev overlay show (src/memory_figures.h): this device's figure once it has one.
    EXPECT(leonida::memory_figures::latest_vram().available == supported);
    if (!supported) {
        EXPECT(!rt::d3d12::vulkan_vram_budget(device.p).available);  // the HUD and the overlay show "n/a"
        std::printf("SKIP: the GPU has no VK_EXT_memory_budget; the figure is not available\n");
        return 77;
    }

    print("before", rt::d3d12::vulkan_vram_budget(device.p));
    D3D12_HEAP_PROPERTIES heap{};
    heap.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = 64 * kMiB;
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    Ref<ID3D12Resource> buffer;
    EXPECT(SUCCEEDED(device.p->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON,
                                                       nullptr, IID_PPV_ARGS(&buffer.p))));
    const leonida::VramBudget after = rt::d3d12::vulkan_vram_budget(device.p);
    print("with a 64 MiB buffer", after);
    EXPECT(after.available);
    EXPECT(after.local_budget > 0);
    EXPECT(after.local_usage + after.shared_usage > 0);
    const leonida::VramBudget published = leonida::memory_figures::latest_vram();
    EXPECT(published.available && published.local_budget > 0);  // may be the read before the buffer: cached

    // A released device withdraws its figure: nothing reads a device that is gone.
    buffer.p->Release();
    buffer.p = nullptr;
    device.p->Release();
    device.p = nullptr;
    EXPECT(!leonida::memory_figures::latest_vram().available);
    std::printf("d3d12_vram_budget: VK_EXT_memory_budget enabled; usage and budget read from the device and published"
                " while it lives\n");
    return 0;
}
