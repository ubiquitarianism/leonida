// A sampled BC texture the title places on its own memory (Xbox device slot 133) waits for memory of its own until
// the title writes it or a command uses it; until then its views name one shared, zeroed image of its exact shape
// (DeviceState::zero_sampled_image), kept for the device's life. Under VRAM pressure that image cannot always be made
// (vkCreateImage or its memory refused). Such a shape must get its shared image at a later placement once the memory is
// back, not stay without one for the whole session, every placement of it taking memory of its own at once.
//   - A control shape gets its shared image (the placement has no memory of its own), or the test is skipped.
//   - The test wraps vkCreateImage behind the instance-proc pointer it hands the layer and refuses the next image of
//     the shape under test: that is the shared image, made before the placement's own. That placement, and one right
//     after it, have memory of their own: the shape is not tried again at every placement.
//   - After the retry delay (a second), a third placement of the shape tries again, and that image is refused too.
//   - The delay doubles at each refusal: 1.5 s later a fourth placement does not try (memory of its own, one image of
//     the shape made for it), and 1.5 s after that (past the 2 s) a fifth gets its shared image (no memory of its
//     own): two images of the shape are made for it, the shared one and the placement's.
// The device has no host image copies, so a placement takes the shared image whatever its driver allows for host-copy
// images (d3d12_existing_heap's shared slot-133 BC images do the same). Needs the bindless set (skipped otherwise).
// Synthetic data only. Set LEONIDA_VK_VALIDATION=1 to run under VK_LAYER_KHRONOS_validation.
#include "vk_test_device.h"
#include "../shims/d3d12_com.h"

#include <atomic>
#include <cstdio>
#include <cstring>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

namespace {

template<class T>
struct Ref {
    T* p = nullptr;
    Ref() = default;
    Ref(const Ref&) = delete;  // a copy would release the same interface twice
    Ref& operator=(const Ref&) = delete;
    ~Ref() { reset(); }
    void reset() {
        if (p) p->Release();
        p = nullptr;
    }
    T* operator->() const { return p; }
};

constexpr UINT width = 64, height = 48;  // the shape under test; no other image of the test has this extent

// The Vulkan calls the layer makes, seen through the function pointers it loads.
PFN_vkGetInstanceProcAddr real_instance_proc = nullptr;
PFN_vkGetDeviceProcAddr real_device_proc = nullptr;
PFN_vkCreateImage real_create_image = nullptr;
std::atomic<bool> refuse_next{false};
std::atomic<unsigned> shape_images{0}, refused{0};  // vkCreateImage calls for the shape under test, and refusals

VKAPI_ATTR VkResult VKAPI_CALL create_image(VkDevice device, const VkImageCreateInfo* info,
                                            const VkAllocationCallbacks* allocator, VkImage* image) {
    if (info->extent.width == width && info->extent.height == height) {
        ++shape_images;
        if (refuse_next.exchange(false)) {
            ++refused;
            return VK_ERROR_OUT_OF_DEVICE_MEMORY;
        }
    }
    return real_create_image(device, info, allocator, image);
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL device_proc(VkDevice device, const char* name) {
    if (std::strcmp(name, "vkCreateImage") == 0) {
        real_create_image = reinterpret_cast<PFN_vkCreateImage>(real_device_proc(device, name));
        return real_create_image ? reinterpret_cast<PFN_vkVoidFunction>(&create_image) : nullptr;
    }
    return real_device_proc(device, name);
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL instance_proc(VkInstance instance, const char* name) {
    if (std::strcmp(name, "vkGetDeviceProcAddr") == 0) {
        real_device_proc = reinterpret_cast<PFN_vkGetDeviceProcAddr>(real_instance_proc(instance, name));
        return real_device_proc ? reinterpret_cast<PFN_vkVoidFunction>(&device_proc) : nullptr;
    }
    return real_instance_proc(instance, name);
}

D3D12_RESOURCE_DESC bc1(UINT w, UINT h) {
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = w;
    desc.Height = h;
    desc.DepthOrArraySize = desc.MipLevels = desc.SampleDesc.Count = 1;
    desc.Format = DXGI_FORMAT_BC1_UNORM;
    return desc;
}

// A placement waiting on the shared zero image has no memory of its own yet.
bool shared(ID3D12Resource* texture) { return !rt::d3d12::vulkan_memory_flags(texture); }

}  // namespace

int main() {
    vktest::Device vk;
    if (!vk.create()) return 77;
    real_instance_proc = vk.instance_proc;
    void* pages = VirtualAlloc(nullptr, 6 * 65536, MEM_RESERVE, PAGE_NOACCESS);
    EXPECT(pages);
    const auto at = [&](unsigned i) { return static_cast<char*>(pages) + i * 65536; };
    int result = 0;
    {
        rt::d3d12::VulkanDeviceInfo info{};
        info.get_instance_proc = &instance_proc;
        info.instance = vk.instance;
        info.physical = vk.physical;
        info.device = vk.device;
        info.queue = vk.queue;
        info.queue_family = vk.family;
        info.bindless = vk.bindless;
        info.bindless_partially_bound = vk.partially_bound;
        Ref<ID3D12Device> device;
        EXPECT(SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&device.p))));
        Ref<ID3D12Resource> control, first, second, third, fourth, fifth;
        const D3D12_RESOURCE_DESC control_desc = bc1(32, 32), desc = bc1(width, height);
        if (!rt::d3d12::bindless_set_layout(device.p)) {
            std::fprintf(stderr, "SKIP: the device has no bindless set\n");
            result = 77;
        } else if (FAILED(rt::d3d12::create_buffer_at_host_address(device.p, at(0), control_desc, 0,
                                                                   IID_PPV_ARGS(&control.p))) ||
                   !shared(control.p)) {
            std::fprintf(stderr, "SKIP: a placed BC texture does not take a shared zero image on this device\n");
            result = 77;
        } else {
            // The shared image of the shape is refused: the first placement, and the next one, get memory of their own.
            refuse_next = true;
            EXPECT(SUCCEEDED(rt::d3d12::create_buffer_at_host_address(device.p, at(1), desc, 0,
                                                                      IID_PPV_ARGS(&first.p))));
            EXPECT(refused == 1);
            const unsigned by_first = shape_images.load();
            EXPECT(SUCCEEDED(rt::d3d12::create_buffer_at_host_address(device.p, at(2), desc, 0,
                                                                      IID_PPV_ARGS(&second.p))));
            const unsigned by_second = shape_images.load() - by_first;
            std::printf("shared image refused: placements %s and %s; %u and %u images of the shape made for them\n",
                        shared(first.p) ? "shared" : "with memory of their own",
                        shared(second.p) ? "shared" : "with memory of their own", by_first, by_second);
            EXPECT(!shared(first.p));
            EXPECT(!shared(second.p));
            EXPECT(by_second == 1);

            // After the retry delay the shape is tried again, and its image is refused again.
            Sleep(1500);
            refuse_next = true;
            const unsigned before_third = shape_images.load();
            EXPECT(SUCCEEDED(rt::d3d12::create_buffer_at_host_address(device.p, at(3), desc, 0,
                                                                      IID_PPV_ARGS(&third.p))));
            const unsigned by_third = shape_images.load() - before_third;
            std::printf("after the retry delay, refused again: placement %s; %u images of the shape made for it\n",
                        shared(third.p) ? "shared" : "with memory of their own", by_third);
            EXPECT(refused == 2);
            EXPECT(!shared(third.p));
            EXPECT(by_third == 2);

            // Refused twice, the shape waits 2 s: 1.5 s later it is not tried, and after the 2 s it gets its image.
            Sleep(1500);
            const unsigned before_fourth = shape_images.load();
            EXPECT(SUCCEEDED(rt::d3d12::create_buffer_at_host_address(device.p, at(4), desc, 0,
                                                                      IID_PPV_ARGS(&fourth.p))));
            const unsigned by_fourth = shape_images.load() - before_fourth;
            std::printf("1.5 s later: placement %s; %u images of the shape made for it\n",
                        shared(fourth.p) ? "shared" : "with memory of their own", by_fourth);
            EXPECT(!shared(fourth.p));
            EXPECT(by_fourth == 1);
            Sleep(1500);
            const unsigned before_fifth = shape_images.load();
            EXPECT(SUCCEEDED(rt::d3d12::create_buffer_at_host_address(device.p, at(5), desc, 0,
                                                                      IID_PPV_ARGS(&fifth.p))));
            const unsigned by_fifth = shape_images.load() - before_fifth;
            std::printf("1.5 s later: placement %s; %u images of the shape made for it\n",
                        shared(fifth.p) ? "shared" : "with memory of their own", by_fifth);
            EXPECT(refused == 2);
            EXPECT(shared(fifth.p));
            EXPECT(by_fifth == 2);
        }
    }
    VirtualFree(pages, 0, MEM_RELEASE);
    if (vk.validation_errors()) {
        std::fprintf(stderr, "%u validation errors\n", vk.validation_errors());
        return 1;
    }
    if (!result) std::printf("ok\n");
    return result;
}
