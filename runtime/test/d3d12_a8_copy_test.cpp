// Copies between the two backings of A8_UNORM (#428 review): an A8_UNORM render target is a
// VK_FORMAT_A8_UNORM_KHR image where the device has it (VulkanDeviceInfo::a8_render_target), any other A8_UNORM
// texture an R8_UNORM image. D3D12 copies between the two as between any two A8_UNORM textures, so:
//   - CopyResource and CopyTextureRegion (a 3x3 box at a nonzero source offset, to a nonzero destination offset)
//     from the A8-backed texture to the R8-backed one and back put each source byte, the texel's alpha, where the
//     destination keeps its alpha: the bytes read back through a footprint are the source's, and the texels outside
//     the box keep the destination's own bytes;
//   - no vkCmdCopyImage pairs a VK_FORMAT_A8_UNORM_KHR image with an image of another format: Vulkan does not count
//     the two as size-compatible (VUID-vkCmdCopyImage-srcImage-01548), whatever bytes a driver writes. The test
//     wraps vkCreateImage and vkCmdCopyImage behind the instance-proc pointer it hands the layer, so it sees this
//     without the validation layer; with LEONIDA_VK_VALIDATION=1 and the layer installed, the layer's errors also
//     fail it;
//   - the same-backing copies (A8 -> A8, R8 -> R8) are the controls and still copy image to image;
//   - a device without the A8 path (a8_render_target false, which VK_KHR_maintenance5 not being enabled means)
//     takes the documented fallback: A8_UNORM render targets refused, other A8_UNORM textures R8_UNORM images that
//     copy as before. tests/test_vulkan_extension_guards.py checks that the extension is enabled wherever the path
//     is taken.
// Synthetic data only.
#include "vk_test_device.h"
#include "../shims/d3d12_com.h"
#include <cstdio>
#include <cstring>
#include <map>
#include <mutex>
#include <vector>

namespace {
template<class T>
struct Ref {
    T* p = nullptr;
    ~Ref() { if (p) p->Release(); }
    T* operator->() const { return p; }
};

// The Vulkan calls the layer makes, seen through the function pointers it loads.
PFN_vkGetInstanceProcAddr real_instance_proc = nullptr;
PFN_vkGetDeviceProcAddr real_device_proc = nullptr;
PFN_vkCreateImage real_create_image = nullptr;
PFN_vkCmdCopyImage real_copy_image = nullptr;
std::mutex seen_mutex;
std::map<VkImage, VkFormat> image_formats;
unsigned a8_images = 0, image_copies = 0, mixed_copies = 0, unknown_copies = 0;

#ifdef VK_KHR_MAINTENANCE_5_EXTENSION_NAME
constexpr VkFormat a8_format = VK_FORMAT_A8_UNORM_KHR;
#else
constexpr VkFormat a8_format = VK_FORMAT_UNDEFINED;
#endif

VKAPI_ATTR VkResult VKAPI_CALL create_image(VkDevice device, const VkImageCreateInfo* info,
                                            const VkAllocationCallbacks* allocator, VkImage* image) {
    const VkResult result = real_create_image(device, info, allocator, image);
    if (result == VK_SUCCESS) {
        std::lock_guard<std::mutex> lock(seen_mutex);
        image_formats[*image] = info->format;
        if (a8_format != VK_FORMAT_UNDEFINED && info->format == a8_format) ++a8_images;
    }
    return result;
}

VKAPI_ATTR void VKAPI_CALL copy_image(VkCommandBuffer command_buffer, VkImage src, VkImageLayout src_layout,
                                      VkImage dst, VkImageLayout dst_layout, std::uint32_t count,
                                      const VkImageCopy* regions) {
    {
        std::lock_guard<std::mutex> lock(seen_mutex);
        ++image_copies;
        const auto s = image_formats.find(src), d = image_formats.find(dst);
        if (s == image_formats.end() || d == image_formats.end()) {
            ++unknown_copies;
        } else if (s->second != d->second && a8_format != VK_FORMAT_UNDEFINED &&
                   (s->second == a8_format || d->second == a8_format)) {
            ++mixed_copies;
            std::fprintf(stderr, "vkCmdCopyImage between VK_FORMAT_A8_UNORM_KHR and format %d\n",
                         int(s->second == a8_format ? d->second : s->second));
        }
    }
    real_copy_image(command_buffer, src, src_layout, dst, dst_layout, count, regions);
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL device_proc(VkDevice device, const char* name) {
    if (std::strcmp(name, "vkCreateImage") == 0) {
        real_create_image = reinterpret_cast<PFN_vkCreateImage>(real_device_proc(device, name));
        return real_create_image ? reinterpret_cast<PFN_vkVoidFunction>(&create_image) : nullptr;
    }
    if (std::strcmp(name, "vkCmdCopyImage") == 0) {
        real_copy_image = reinterpret_cast<PFN_vkCmdCopyImage>(real_device_proc(device, name));
        return real_copy_image ? reinterpret_cast<PFN_vkVoidFunction>(&copy_image) : nullptr;
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

D3D12_RESOURCE_DESC texture(D3D12_RESOURCE_FLAGS flags) {
    D3D12_RESOURCE_DESC d{};
    d.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    d.Width = d.Height = 4;
    d.DepthOrArraySize = d.MipLevels = 1;
    d.Format = DXGI_FORMAT_A8_UNORM;
    d.SampleDesc.Count = 1;
    d.Flags = flags;
    return d;
}

D3D12_RESOURCE_DESC buffer(UINT64 bytes) {
    D3D12_RESOURCE_DESC d{};
    d.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    d.Width = bytes;
    d.Height = d.DepthOrArraySize = d.MipLevels = 1;
    d.SampleDesc.Count = 1;
    d.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    return d;
}

D3D12_RESOURCE_BARRIER transition(ID3D12Resource* r, D3D12_RESOURCE_STATES a, D3D12_RESOURCE_STATES b) {
    D3D12_RESOURCE_BARRIER x{};
    x.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    x.Transition.pResource = r;
    x.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    x.Transition.StateBefore = a;
    x.Transition.StateAfter = b;
    return x;
}

// The source's and the destination's own bytes before the copy: distinct, nonzero, and different from each other.
std::uint8_t source_byte(UINT x, UINT y) { return std::uint8_t(0x11 + 0x10 * y + x); }
std::uint8_t destination_byte(UINT x, UINT y) { return std::uint8_t(0xc0 + 4 * y + x); }

constexpr D3D12_BOX copy_box{1, 0, 0, 4, 3, 1};  // 3x3 texels from (1, 0)
constexpr UINT copy_x = 0, copy_y = 1;           // to (0, 1)

// Uploads both patterns, copies `src` -> `dst` (CopyResource, or CopyTextureRegion of copy_box), reads `dst` back
// and compares every texel with what D3D12 defines. Returns false (and says why) on any difference.
bool copy_case(ID3D12Device* device, const char* name, const D3D12_RESOURCE_DESC& src_desc,
               const D3D12_RESOURCE_DESC& dst_desc, bool whole) {
    Ref<ID3D12CommandQueue> queue;
    Ref<ID3D12CommandAllocator> allocator;
    Ref<ID3D12GraphicsCommandList> list;
    Ref<ID3D12Fence> fence;
    const D3D12_COMMAND_QUEUE_DESC queue_desc{};
    if (FAILED(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue.p))) ||
        FAILED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p))) ||
        FAILED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, nullptr,
                                         IID_PPV_ARGS(&list.p))) ||
        FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p)))) {
        std::fprintf(stderr, "%s: queue, list or fence not created\n", name);
        return false;
    }
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
    UINT64 bytes = 0;
    device->GetCopyableFootprints(&dst_desc, 0, 1, 0, &footprint, nullptr, nullptr, &bytes);
    const D3D12_HEAP_PROPERTIES default_heap{D3D12_HEAP_TYPE_DEFAULT}, upload_heap{D3D12_HEAP_TYPE_UPLOAD},
        readback_heap{D3D12_HEAP_TYPE_READBACK};
    const UINT64 pitch = footprint.Footprint.RowPitch, second = 2 * D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT;
    const D3D12_RESOURCE_DESC upload_desc = buffer(second * 2), readback_desc = buffer(bytes);
    Ref<ID3D12Resource> src, dst, upload, readback;
    if (FAILED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &src_desc,
                                               D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&src.p))) ||
        FAILED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &dst_desc,
                                               D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&dst.p))) ||
        FAILED(device->CreateCommittedResource(&upload_heap, D3D12_HEAP_FLAG_NONE, &upload_desc,
                                               D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&upload.p))) ||
        FAILED(device->CreateCommittedResource(&readback_heap, D3D12_HEAP_FLAG_NONE, &readback_desc,
                                               D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&readback.p)))) {
        std::fprintf(stderr, "%s: resources not created\n", name);
        return false;
    }
    std::uint8_t* mapped = nullptr;
    if (FAILED(upload->Map(0, nullptr, reinterpret_cast<void**>(&mapped)))) return false;
    for (UINT y = 0; y < 4; ++y)
        for (UINT x = 0; x < 4; ++x) {
            mapped[y * pitch + x] = source_byte(x, y);
            mapped[second + y * pitch + x] = destination_byte(x, y);
        }
    upload->Unmap(0, nullptr);

    D3D12_TEXTURE_COPY_LOCATION from{upload.p, D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, {}};
    from.PlacedFootprint = footprint;
    D3D12_TEXTURE_COPY_LOCATION to_src{src.p, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {}};
    list->CopyTextureRegion(&to_src, 0, 0, 0, &from, nullptr);
    from.PlacedFootprint.Offset = second;
    D3D12_TEXTURE_COPY_LOCATION to_dst{dst.p, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {}};
    list->CopyTextureRegion(&to_dst, 0, 0, 0, &from, nullptr);
    D3D12_RESOURCE_BARRIER barrier =
        transition(src.p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE);
    list->ResourceBarrier(1, &barrier);
    if (whole) {
        list->CopyResource(dst.p, src.p);
    } else {
        D3D12_TEXTURE_COPY_LOCATION s{src.p, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {}};
        list->CopyTextureRegion(&to_dst, copy_x, copy_y, 0, &s, &copy_box);
    }
    barrier = transition(dst.p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE);
    list->ResourceBarrier(1, &barrier);
    D3D12_TEXTURE_COPY_LOCATION back{readback.p, D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, {}};
    back.PlacedFootprint = footprint;
    back.PlacedFootprint.Offset = 0;
    D3D12_TEXTURE_COPY_LOCATION from_dst{dst.p, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {}};
    list->CopyTextureRegion(&back, 0, 0, 0, &from_dst, nullptr);
    if (FAILED(list->Close())) {
        std::fprintf(stderr, "%s: Close failed\n", name);
        return false;
    }
    ID3D12CommandList* lists[] = {list.p};
    queue->ExecuteCommandLists(1, lists);
    if (rt::d3d12::last_execute_result(queue.p) != leonida::ReplayResult::success) {
        std::fprintf(stderr, "%s: the replay refused the list (%d)\n", name,
                     int(rt::d3d12::last_execute_result(queue.p)));
        return false;
    }
    if (FAILED(queue->Signal(fence.p, 1)) || FAILED(fence->SetEventOnCompletion(1, nullptr))) return false;
    if (FAILED(readback->Map(0, nullptr, reinterpret_cast<void**>(&mapped)))) return false;
    bool ok = true;
    for (UINT y = 0; y < 4; ++y)
        for (UINT x = 0; x < 4; ++x) {
            const bool inside = whole || (x >= copy_x && x < copy_x + (copy_box.right - copy_box.left) &&
                                          y >= copy_y && y < copy_y + (copy_box.bottom - copy_box.top));
            const std::uint8_t want = !inside ? destination_byte(x, y)
                                      : whole ? source_byte(x, y)
                                              : source_byte(x - copy_x + copy_box.left, y - copy_y + copy_box.top);
            const std::uint8_t got = mapped[y * pitch + x];
            if (got != want) {
                if (ok) std::fprintf(stderr, "%s: texel (%u,%u) got 0x%02x, want 0x%02x\n", name, x, y, got, want);
                ok = false;
            }
        }
    readback->Unmap(0, nullptr);
    if (ok) std::printf("%s: all 16 texels as D3D12 defines them\n", name);
    return ok;
}

bool make_device(const vktest::Device& vk, bool a8_render_target, Ref<ID3D12Device>& device) {
    rt::d3d12::VulkanDeviceInfo info{};
    info.get_instance_proc = &instance_proc;
    info.instance = vk.instance;
    info.physical = vk.physical;
    info.device = vk.device;
    info.queue = vk.queue;
    info.queue_family = vk.family;
    info.a8_render_target = a8_render_target;
    return SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&device.p)));
}
}  // namespace

int main() {
    vktest::Device vk;
    if (!vk.create()) return 77;
    real_instance_proc = vk.instance_proc;
    std::printf("device: %s VK_FORMAT_A8_UNORM_KHR render targets; validation layer %s\n",
                vk.a8_render_target ? "has" : "does NOT have", vk.validation ? "on" : "off");
    int failures = 0;
    const D3D12_RESOURCE_DESC target = texture(D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET);
    const D3D12_RESOURCE_DESC plain = texture(D3D12_RESOURCE_FLAG_NONE);

    if (vk.a8_render_target) {
        Ref<ID3D12Device> device;
        if (!make_device(vk, true, device)) return 1;
        struct Case {
            const char* name;
            const D3D12_RESOURCE_DESC *src, *dst;
            bool whole;
        };
        const Case cases[] = {
            {"CopyResource A8 render target -> A8 texture", &target, &plain, true},
            {"CopyResource A8 texture -> A8 render target", &plain, &target, true},
            {"CopyTextureRegion A8 render target -> A8 texture (3x3 box)", &target, &plain, false},
            {"CopyTextureRegion A8 texture -> A8 render target (3x3 box)", &plain, &target, false},
            {"control: CopyResource A8 render target -> A8 render target", &target, &target, true},
            {"control: CopyTextureRegion A8 texture -> A8 texture (3x3 box)", &plain, &plain, false},
        };
        for (const Case& c : cases) {
            const unsigned mixed_before = mixed_copies, copies_before = image_copies;
            if (!copy_case(device.p, c.name, *c.src, *c.dst, c.whole)) ++failures;
            if (mixed_copies != mixed_before) {
                std::fprintf(stderr, "%s: %u vkCmdCopyImage between VK_FORMAT_A8_UNORM_KHR and R8_UNORM\n", c.name,
                             mixed_copies - mixed_before);
                ++failures;
            }
            const bool control = std::strncmp(c.name, "control", 7) == 0;
            if (control && image_copies == copies_before) {  // the controls still copy image to image
                std::fprintf(stderr, "%s: no vkCmdCopyImage seen; the wrapper does not see the layer's calls\n",
                             c.name);
                ++failures;
            }
        }
        if (!a8_images) {
            std::fprintf(stderr, "no VK_FORMAT_A8_UNORM_KHR image was created for the A8 render targets\n");
            ++failures;
        }
        // 1D and 3D (RT-A8-DIMENSIONS): only a 2D render target is A8-backed, so a 1D or 3D A8_UNORM texture is an
        // R8_UNORM image on this device too and never one side of a mixed-backing copy; a 1D render target is
        // refused rather than made in either backing.
        {
            const unsigned a8_before = a8_images;
            const D3D12_HEAP_PROPERTIES default_heap{D3D12_HEAP_TYPE_DEFAULT};
            D3D12_RESOURCE_DESC one_d = plain, three_d = plain, one_d_target = target;
            one_d.Dimension = one_d_target.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE1D;
            one_d.Width = one_d_target.Width = 128;
            one_d.Height = one_d_target.Height = 1;
            three_d.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE3D;
            three_d.Width = three_d.Height = 8;
            three_d.DepthOrArraySize = 4;
            Ref<ID3D12Resource> r1, r3, rt1;
            const bool made = SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &one_d,
                                                                        D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                                        IID_PPV_ARGS(&r1.p))) &&
                              SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &three_d,
                                                                        D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                                        IID_PPV_ARGS(&r3.p)));
            const bool refused = FAILED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE,
                                                                        &one_d_target, D3D12_RESOURCE_STATE_COMMON,
                                                                        nullptr, IID_PPV_ARGS(&rt1.p))) &&
                                 !rt1.p;
            if (!made || !refused || a8_images != a8_before) {
                std::fprintf(stderr, "1D/3D A8_UNORM: created %d, 1D render target refused %d, %u A8 images\n",
                             int(made), int(refused), a8_images - a8_before);
                ++failures;
            } else {
                std::printf("1D/3D A8_UNORM textures are R8_UNORM images next to A8 render targets; a 1D render "
                            "target is refused\n");
            }
        }
    } else {
        std::printf("A8 render targets: not on this device, only the fallback below runs\n");
    }

    // The documented fallback of a device without VK_FORMAT_A8_UNORM_KHR (or without VK_KHR_maintenance5 enabled,
    // a8_render_target false; forced here on any device): A8_UNORM render targets are refused at creation and in
    // the format support, and every other A8_UNORM texture is an R8_UNORM image that copies as before.
    {
        Ref<ID3D12Device> device;
        if (!make_device(vk, false, device)) return 1;
        const unsigned a8_before = a8_images;
        D3D12_FEATURE_DATA_FORMAT_SUPPORT support{DXGI_FORMAT_A8_UNORM};
        if (FAILED(device->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &support, sizeof(support))) ||
            !(support.Support1 & D3D12_FORMAT_SUPPORT1_TEXTURE2D) ||
            (support.Support1 & (D3D12_FORMAT_SUPPORT1_RENDER_TARGET | D3D12_FORMAT_SUPPORT1_BLENDABLE))) {
            std::fprintf(stderr, "fallback: A8_UNORM format support 0x%08x (want TEXTURE2D, no RENDER_TARGET or "
                                 "BLENDABLE)\n", unsigned(support.Support1));
            ++failures;
        }
        const D3D12_RESOURCE_ALLOCATION_INFO size = device->GetResourceAllocationInfo(0, 1, &target);
        const D3D12_HEAP_PROPERTIES default_heap{D3D12_HEAP_TYPE_DEFAULT};
        Ref<ID3D12Resource> refused;
        const HRESULT hr = device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &target,
                                                           D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                           IID_PPV_ARGS(&refused.p));
        if (SUCCEEDED(hr) || refused.p || size.SizeInBytes != UINT64_MAX) {
            std::fprintf(stderr, "fallback: an A8_UNORM render target was not refused (hr 0x%08lx, size %llu)\n",
                         static_cast<unsigned long>(hr), static_cast<unsigned long long>(size.SizeInBytes));
            ++failures;
        } else {
            std::printf("fallback: A8_UNORM render target refused (hr 0x%08lx), format support 0x%08x\n",
                        static_cast<unsigned long>(hr), unsigned(support.Support1));
        }
        if (!copy_case(device.p, "fallback: CopyResource A8 texture -> A8 texture", plain, plain, true) ||
            !copy_case(device.p, "fallback: CopyTextureRegion A8 texture -> A8 texture (3x3 box)", plain, plain,
                       false))
            ++failures;
        if (a8_images != a8_before) {
            std::fprintf(stderr, "fallback: %u VK_FORMAT_A8_UNORM_KHR images created\n", a8_images - a8_before);
            ++failures;
        }
    }

    if (unknown_copies) {
        std::fprintf(stderr, "%u vkCmdCopyImage on images the wrapper did not see created\n", unknown_copies);
        ++failures;
    }
    if (vk.validation_errors()) {
        std::fprintf(stderr, "%u Vulkan validation errors\n", vk.validation_errors());
        ++failures;
    }
    std::printf("%u vkCmdCopyImage seen, %u between the two A8 backings; %s\n", image_copies, mixed_copies,
                failures ? "FAILED" : "passed");
    return failures ? 1 : 0;
}
