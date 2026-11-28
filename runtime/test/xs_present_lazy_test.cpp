// The presentation gta6rt's build of the device proxy makes for itself (shims/d3d12_xs_device.cpp, present_state_for
// and queue_present_x), driven through the slots in the order a title uses them. This test compiles the proxy with
// LEONIDA_VULKAN_DEVICE, as gta6rt does: a build of the proxy without it builds no presenter.
//
// Checked, on the Vulkan-backed device behind a device proxy:
//   - the frame events before the first present are the device's own: slots 78 and 79 answer S_OK and slot 80 hands
//     out a token at once;
//   - nothing is built before the first PresentX; the first one (a blank present at queue slot 30, PresentX on a queue
//     in the older layout, which a queue is until a fence reaches slot 16 or 17) builds the runtime's window, once,
//     and answers S_OK;
//   - the frame events stay the device's: slot 80 still answers at once, with the next token of the same source (a
//     pacer that never saw the interval would refuse the wait);
//   - that source is the presentation's: PresentX refuses a token the device has not handed out (E_INVALIDARG) and
//     accepts the one it has. The accepted one is shown by a source PresentX refuses after its token check
//     (block-compressed: DXGI_ERROR_UNSUPPORTED), so no frame has to be drawn here;
//   - a second PresentX builds no second window, and the window goes with the device proxy: the presenter holds no
//     reference on the proxy that owns it. The window is destroyed by its own thread, so the test waits for that.
// Needs a window and a Vulkan GPU with presentation: exits 77 (skipped) without a Vulkan device, and also where the
// device cannot present to a window (there the slot answers S_OK and shows nothing, which is checked before the skip).
#include "vk_test_device.h"
#include "../shims/d3d12_com.h"
#include "../shims/d3d12_xs_device.h"
#include "../shims/xbox_present.h"
#include "../src/frame_pacer.h"

#include <cstdint>
#include <cstdio>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

namespace {

template <typename T>
struct Ref {
    T* p = nullptr;
    ~Ref() { reset(); }
    void reset() {
        if (p) p->Release();
        p = nullptr;
    }
    T* operator->() const { return p; }
};

using RawFn = std::uint64_t(WINAPI*)(void*, std::uint64_t, std::uint64_t, std::uint64_t, std::uint64_t,
                                     std::uint64_t);

HRESULT raw(void* object, unsigned slot, std::uint64_t a1 = 0, std::uint64_t a2 = 0, std::uint64_t a3 = 0,
            std::uint64_t a4 = 0, std::uint64_t a5 = 0) {
    const auto fn = reinterpret_cast<RawFn>((*reinterpret_cast<void***>(object))[slot]);
    return static_cast<HRESULT>(static_cast<std::uint32_t>(fn(object, a1, a2, a3, a4, a5)));
}

template <typename T>
std::uint64_t word(T* pointer) { return reinterpret_cast<std::uint64_t>(pointer); }

// The windows of the class the device proxy registers for its presenter that belong to this process.
unsigned own_present_windows() {
    unsigned count = 0;
    for (HWND window = nullptr; (window = FindWindowExW(nullptr, window, L"LeonidaPresentX", nullptr)) != nullptr;) {
        DWORD process = 0;
        GetWindowThreadProcessId(window, &process);
        if (process == GetCurrentProcessId()) ++count;
    }
    return count;
}

constexpr unsigned kSetFrameIntervalX = 78, kScheduleFrameEventX = 79, kWaitFrameEventX = 80;  // device slots
constexpr unsigned kPresentX = 30;                                                            // queue slot

// The runtime's window belongs to a thread of its own, which destroys it when the device proxy lets go of it.
bool present_windows_gone() {
    for (int i = 0; i < 200; ++i) {
        if (own_present_windows() == 0) return true;
        Sleep(10);
    }
    return false;
}

}  // namespace

int main() {
    vktest::Device d;
    if (!d.create(true)) {  // no Vulkan loader, device or presentation extensions: nothing to test (ctest: skipped)
        std::printf("no Vulkan device with presentation: skipped\n");
        return 77;
    }
    bool presenting = false;
    {
        rt::d3d12::VulkanDeviceInfo info{};
        info.get_instance_proc = d.instance_proc;
        info.instance = d.instance;
        info.physical = d.physical;
        info.device = d.device;
        info.queue = d.queue;
        info.queue_family = d.family;
        info.presentation = true;
        info.swapchain_maintenance1 = d.swapchain_maintenance1;
        ID3D12Device* inner = nullptr;
        EXPECT(SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&inner))));
        Ref<ID3D12Device> device;
        EXPECT(SUCCEEDED(rt::d3d12_xs::create_device_proxy(inner, __uuidof(ID3D12Device),
                                                           reinterpret_cast<void**>(&device.p))));
        rt::d3d12::set_device_outer(inner, device.p);
        D3D12_COMMAND_QUEUE_DESC queue_desc{};
        Ref<ID3D12CommandQueue> queue;
        EXPECT(SUCCEEDED(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue.p))));
        EXPECT(rt::d3d12_xs::is_queue_proxy(queue.p));

        // What a title does before its first frame: the interval, the origin event, a wait. All the device's own.
        const auto tokens = rt::d3d12_xs::device_frame_tokens(device.p);
        EXPECT(tokens && tokens->issued() == 0);
        EXPECT(raw(device.p, kSetFrameIntervalX, 0, 16667, 1, 0) == S_OK);
        EXPECT(raw(device.p, kScheduleFrameEventX, 0, 0, 0, 0) == S_OK);
        std::uint64_t first = 0;
        EXPECT(raw(device.p, kWaitFrameEventX, 0, INFINITE, 0, 0, word(&first)) == S_OK && first == 1);
        EXPECT(own_present_windows() == 0);  // nothing is built before the first PresentX

        // The first PresentX builds the presentation. A blank present needs no token.
        EXPECT(raw(queue.p, kPresentX, 0, 0, 0) == S_OK);
        EXPECT(own_present_windows() == 1);

        // Still the device's WaitFrameEventX, at once, from the same source.
        std::uint64_t second = 0;
        EXPECT(raw(device.p, kWaitFrameEventX, 0, INFINITE, 0, 0, word(&second)) == S_OK && second == first + 1);
        EXPECT(tokens->issued() == second);

        // A source PresentX refuses after it has checked the token, so the answer tells the two tokens apart.
        const D3D12_HEAP_PROPERTIES default_heap{D3D12_HEAP_TYPE_DEFAULT};
        D3D12_RESOURCE_DESC desc{};
        desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        desc.Width = 64;
        desc.Height = 64;
        desc.DepthOrArraySize = 1;
        desc.MipLevels = 1;
        desc.Format = DXGI_FORMAT_BC1_UNORM;
        desc.SampleDesc.Count = 1;
        Ref<ID3D12Resource> compressed;
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_ALLOW_DISPLAY, &desc,
                                                         D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                         IID_PPV_ARGS(&compressed.p))));
        ID3D12Resource* resources[1] = {compressed.p};
        rt::d3d12::XboxPresentPlaneParameters plane{};
        plane.ResourceCount = 1;
        plane.ppResources = resources;
        plane.Token = second + 1;  // not handed out yet
        const HRESULT unissued = raw(queue.p, kPresentX, 1, word(&plane), 0);
        if (unissued == S_OK) {
            // This device cannot present to a window (logged by the proxy): the slot answers S_OK whatever it is
            // given and shows nothing, so the title's frame loop would run. The rest needs a presenter.
            plane.Token = second;
            EXPECT(raw(queue.p, kPresentX, 1, word(&plane), 0) == S_OK);
            EXPECT(own_present_windows() == 1);
        } else {
            presenting = true;
            EXPECT(unissued == E_INVALIDARG);
            plane.Token = second;  // the device's token: accepted, and then the source is refused
            EXPECT(raw(queue.p, kPresentX, 1, word(&plane), 0) == DXGI_ERROR_UNSUPPORTED);
            // Once handed out by the device, the refused token is accepted as well.
            std::uint64_t third = 0;
            EXPECT(raw(device.p, kWaitFrameEventX, 0, 0, 0, 0, word(&third)) == S_OK && third == second + 1);
            plane.Token = third;
            EXPECT(raw(queue.p, kPresentX, 1, word(&plane), 0) == DXGI_ERROR_UNSUPPORTED);
            EXPECT(raw(queue.p, kPresentX, 0, 0, 0) == S_OK);  // a second blank present, on the same window
            EXPECT(own_present_windows() == 1);
        }

        // The window and the presenter go with the device proxy.
        compressed.reset();
        queue.reset();
        device.reset();
        EXPECT(present_windows_gone());
    }
    if (d.validation_errors() != d.known_validation_errors()) {
        std::fprintf(stderr, "%u validation errors (%u known)\n", d.validation_errors(), d.known_validation_errors());
        return 1;
    }
    if (!presenting) {
        std::printf("this device cannot present to a window: PresentX answers S_OK and shows nothing; skipped\n");
        return 77;
    }
    std::printf("xs_present_lazy_test: all cases pass\n");
    return 0;
}
