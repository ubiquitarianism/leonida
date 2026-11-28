// Synthetic texture UAVs through the Vulkan D3D12 device. It has no compute PSOs yet, so dispatch the translated
// cs_texture_uav.asm through its actual Vulkan set 0. Non-atomic UINT cases also compare in-view writes with
// Windows WARP. Build the shader with runtime/test/shaders/build.py and pass the .spv as argv[1].
// LEONIDA_VK_VALIDATION=1 enables the Khronos validation layer. No game files are used.
#include "vk_test_device.h"
#include "vk_pc_tail.h"
#include "../shims/d3d12_com.h"
#include "../src/gfx10_descriptors.h"
#include "d3d12_texture_uav_oracle.h"

#include <wrl/client.h>

#include <array>
#include <bit>
#include <cstring>
#include <fstream>
#include <span>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return false; } \
} while (0)

namespace {

using Microsoft::WRL::ComPtr;
namespace g = leonida::gfx10;
constexpr auto uav_heap = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
constexpr auto formatless_read = VK_FORMAT_FEATURE_2_STORAGE_READ_WITHOUT_FORMAT_BIT;
constexpr auto formatless_write = VK_FORMAT_FEATURE_2_STORAGE_WRITE_WITHOUT_FORMAT_BIT;

// Only the layer sees the masked capabilities. The real device and validation layer see the unmodified driver.
PFN_vkGetInstanceProcAddr real_instance_proc = nullptr;
PFN_vkGetPhysicalDeviceFormatProperties2 real_format_properties = nullptr;
VkFormatFeatureFlags2 missing_format_features = 0;

VKAPI_ATTR void VKAPI_CALL masked_format_properties(VkPhysicalDevice physical, VkFormat format,
                                                   VkFormatProperties2* properties) {
    real_format_properties(physical, format, properties);
    if (format != VK_FORMAT_R32_UINT) return;
    for (auto* next = static_cast<VkBaseOutStructure*>(properties->pNext); next; next = next->pNext)
        if (next->sType == VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_3)
            reinterpret_cast<VkFormatProperties3*>(next)->optimalTilingFeatures &= ~missing_format_features;
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL masked_instance_proc(VkInstance instance, const char* name) {
    if (std::strcmp(name, "vkGetPhysicalDeviceFormatProperties2") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(masked_format_properties);
    return real_instance_proc(instance, name);
}

// Numeric values are also the branches selected by cs_texture_uav.asm.
enum class Kind : UINT { one_d, array, volume, two_d, float_volume, float_array, atomic_volume };

D3D12_RESOURCE_DESC texture(Kind kind, UINT layers, DXGI_FORMAT format = DXGI_FORMAT_R32_UINT) {
    using enum Kind;
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = kind == one_d || kind == array || kind == float_array ? D3D12_RESOURCE_DIMENSION_TEXTURE1D :
                     kind == two_d ? D3D12_RESOURCE_DIMENSION_TEXTURE2D : D3D12_RESOURCE_DIMENSION_TEXTURE3D;
    desc.Width = 8;
    desc.Height = desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE1D ? 1 : 4;
    desc.DepthOrArraySize = UINT16(layers);
    desc.MipLevels = 3;
    desc.Format = format;
    desc.SampleDesc.Count = 1;
    desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    return desc;
}

D3D12_UNORDERED_ACCESS_VIEW_DESC uav(Kind kind, UINT mip, UINT first, UINT count,
                                     DXGI_FORMAT format = DXGI_FORMAT_R32_UINT) {
    using enum Kind;
    D3D12_UNORDERED_ACCESS_VIEW_DESC view{};
    view.Format = format;
    switch (kind) {
    case one_d: view.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE1D; view.Texture1D = {mip}; break;
    case array:
    case float_array: view.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE1DARRAY; view.Texture1DArray = {mip, first, count}; break;
    case two_d: view.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D; view.Texture2D = {mip, 0}; break;
    default: view.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE3D; view.Texture3D = {mip, first, count}; break;
    }
    return view;
}

struct Push {
    g::ImageWords key;
    UINT kind;
};

std::uint32_t push_bytes() {
    return leonida::vk_test::pc_tail_requested() ? 128u : 144u;
}

class Compute {
public:
    explicit Compute(vktest::Device& vk) : vk_(vk) {}
    ~Compute() {
        if (pipeline_) vk_.fn<PFN_vkDestroyPipeline>("vkDestroyPipeline")(vk_.device, pipeline_, nullptr);
        if (layout_) vk_.fn<PFN_vkDestroyPipelineLayout>("vkDestroyPipelineLayout")(vk_.device, layout_, nullptr);
        if (module_) vk_.fn<PFN_vkDestroyShaderModule>("vkDestroyShaderModule")(vk_.device, module_, nullptr);
    }

    bool create(ID3D12Device* device, const char* path) {
        const auto set_layout = rt::d3d12::bindless_set_layout(device);
        set_ = rt::d3d12::bindless_set(device);
        EXPECT(set_layout && set_);
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        EXPECT(file && file.tellg() > 0 && file.tellg() % 4 == 0);
        std::vector<std::uint32_t> code(std::size_t(file.tellg()) / 4);
        file.seekg(0);
        EXPECT(file.read(reinterpret_cast<char*>(code.data()), std::streamsize(code.size() * 4)));
        // This fixture reads only the descriptor and case selector in s0..s8, not the heap/code-base tail.
        EXPECT(!leonida::vk_test::spirv_declares_set(code.data(), code.size() * 4, leonida::vk_test::pc_tail_set));
        VkShaderModuleCreateInfo shader{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        shader.codeSize = code.size() * 4;
        shader.pCode = code.data();
        EXPECT(vk_.fn<PFN_vkCreateShaderModule>("vkCreateShaderModule")(vk_.device, &shader, nullptr, &module_) ==
               VK_SUCCESS);
        const VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT, 0, push_bytes()};
        VkPipelineLayoutCreateInfo layout{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        layout.setLayoutCount = 1;
        layout.pSetLayouts = &set_layout;
        layout.pushConstantRangeCount = 1;
        layout.pPushConstantRanges = &push;
        EXPECT(vk_.fn<PFN_vkCreatePipelineLayout>("vkCreatePipelineLayout")(vk_.device, &layout, nullptr, &layout_) ==
               VK_SUCCESS);
        VkComputePipelineCreateInfo pipeline{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        pipeline.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        pipeline.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        pipeline.stage.module = module_;
        pipeline.stage.pName = "main";
        pipeline.layout = layout_;
        EXPECT(vk_.fn<PFN_vkCreateComputePipelines>("vkCreateComputePipelines")(
                   vk_.device, VK_NULL_HANDLE, 1, &pipeline, nullptr, &pipeline_) == VK_SUCCESS);
        return true;
    }

    bool run(ID3D12Resource* resource, UINT mip, VkExtent3D extent, const Push& push,
             std::span<const std::uint32_t> input, std::span<std::uint32_t> output) {
        const auto desc = resource->GetDesc();
        const bool volume = desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D;
        const bool check_base_mip = push.kind == UINT(Kind::float_volume) && mip != 0;
        const std::size_t base_texels = check_base_mip ? std::size_t(desc.Width) * desc.Height * desc.DepthOrArraySize : 0;
        constexpr std::uint32_t base_marker = std::bit_cast<std::uint32_t>(4096.0f);
        const VkDeviceSize bytes = input.size_bytes() + base_texels * sizeof(base_marker);
        const VkImage image = rt::d3d12::vulkan_image(resource);
        EXPECT(image);
        void *src = nullptr, *dst = nullptr;
        const VkBuffer upload = vk_.buffer(bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, &src);
        const VkBuffer readback = vk_.buffer(bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &dst);
        EXPECT(upload && readback);
        std::memcpy(src, input.data(), input.size_bytes());
        std::fill_n(static_cast<std::uint32_t*>(src) + input.size(), base_texels, base_marker);

        EXPECT(vk_.run([&](VkCommandBuffer cb) {
            const auto barrier = vk_.fn<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier");
            VkImageMemoryBarrier texture_barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
            texture_barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            texture_barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
            texture_barrier.srcQueueFamilyIndex = texture_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            texture_barrier.image = image;
            texture_barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, check_base_mip ? 0u : mip,
                                                 check_base_mip ? mip + 1 : 1, 0, volume ? 1 : extent.depth};
            texture_barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr,
                    1, &texture_barrier);
            VkBufferImageCopy copies[2]{};
            copies[0].imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, mip, 0, volume ? 1 : extent.depth};
            copies[0].imageExtent = {extent.width, extent.height, volume ? extent.depth : 1};
            copies[1].bufferOffset = input.size_bytes();
            copies[1].imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            copies[1].imageExtent = {UINT(desc.Width), desc.Height, desc.DepthOrArraySize};
            const UINT copy_count = check_base_mip ? 2 : 1;
            vk_.fn<PFN_vkCmdCopyBufferToImage>("vkCmdCopyBufferToImage")(cb, upload, image, VK_IMAGE_LAYOUT_GENERAL,
                                                                        copy_count, copies);
            texture_barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
            texture_barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            texture_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            barrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0,
                    nullptr, 1, &texture_barrier);
            vk_.fn<PFN_vkCmdBindPipeline>("vkCmdBindPipeline")(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
            vk_.fn<PFN_vkCmdBindDescriptorSets>("vkCmdBindDescriptorSets")(cb, VK_PIPELINE_BIND_POINT_COMPUTE, layout_,
                                                                           0, 1, &set_, 0, nullptr);
            std::array<std::uint32_t, 36> sgprs{};
            std::memcpy(sgprs.data(), &push, sizeof(push));
            vk_.fn<PFN_vkCmdPushConstants>("vkCmdPushConstants")(cb, layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                                                                  push_bytes(), sgprs.data());
            const UINT dispatch_depth = push.kind == UINT(Kind::one_d) ? 1 : extent.depth;
            vk_.fn<PFN_vkCmdDispatch>("vkCmdDispatch")(cb, extent.width, extent.height, dispatch_depth);
            texture_barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
            texture_barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            barrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                    VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &texture_barrier);
            vk_.fn<PFN_vkCmdCopyImageToBuffer>("vkCmdCopyImageToBuffer")(cb, image, VK_IMAGE_LAYOUT_GENERAL, readback,
                                                                        copy_count, copies);
            VkMemoryBarrier host{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_TRANSFER_WRITE_BIT,
                                 VK_ACCESS_HOST_READ_BIT};
            barrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &host, 0, nullptr, 0,
                    nullptr);
            return true;
        }));
        std::memcpy(output.data(), dst, output.size_bytes());
        const auto* base = static_cast<const std::uint32_t*>(dst) + output.size();
        EXPECT(std::all_of(base, base + base_texels, [&](std::uint32_t word) { return word == base_marker; }));
        return true;
    }

private:
    vktest::Device& vk_;
    VkShaderModule module_ = VK_NULL_HANDLE;
    VkPipelineLayout layout_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
    VkDescriptorSet set_ = VK_NULL_HANDLE;
};

bool create_texture(ID3D12Device* device, const D3D12_RESOURCE_DESC& desc, ID3D12Resource** resource) {
    const D3D12_HEAP_PROPERTIES heap{D3D12_HEAP_TYPE_DEFAULT};
    // The refusal test needs a valid read-only resource; only a UAV-capable resource can start in UAV state.
    const auto state = desc.Flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS ? D3D12_RESOURCE_STATE_UNORDERED_ACCESS
                                                                             : D3D12_RESOURCE_STATE_COMMON;
    return SUCCEEDED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc,
                    state, nullptr, IID_PPV_ARGS(resource)));
}

auto read_slot(D3D12_CPU_DESCRIPTOR_HANDLE h) {
    g::ImageWords words{};
    std::memcpy(words.data(), reinterpret_cast<const void*>(h.ptr), sizeof(words));
    return words;
}

bool flush_descriptors(ID3D12Device* device) {
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> list;
    const D3D12_COMMAND_QUEUE_DESC desc{};
    EXPECT(SUCCEEDED(device->CreateCommandQueue(&desc, IID_PPV_ARGS(&queue))));
    EXPECT(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator))));
    EXPECT(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.Get(), nullptr,
                                               IID_PPV_ARGS(&list))));
    EXPECT(SUCCEEDED(list->Close()));
    ID3D12CommandList* lists[] = {list.Get()};
    queue->ExecuteCommandLists(1, lists);
    // Queue destruction waits for the device-local descriptor upload before the raw Vulkan dispatch.
    return rt::d3d12::last_execute_result(queue.Get()) == leonida::ReplayResult::success;
}

bool check_partial_clear(ID3D12Device* device, ID3D12Resource* resource, D3D12_CPU_DESCRIPTOR_HANDLE whole,
                         D3D12_CPU_DESCRIPTOR_HANDLE partial, const D3D12_RESOURCE_DESC& shape,
                         UINT mip, UINT first, UINT writable) {
    ComPtr<ID3D12CommandQueue> queue;
    const D3D12_COMMAND_QUEUE_DESC qdesc{D3D12_COMMAND_LIST_TYPE_DIRECT};
    EXPECT(SUCCEEDED(device->CreateCommandQueue(&qdesc, IID_PPV_ARGS(&queue))));
    ComPtr<ID3D12Fence> fence;
    EXPECT(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence))));
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> list;
    EXPECT(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator))));
    EXPECT(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.Get(), nullptr,
                                               IID_PPV_ARGS(&list))));
    const UINT clear_value[4] = {42, 0, 0, 0};
    ID3D12CommandList* lists[] = {list.Get()};
    if (shape.Format == DXGI_FORMAT_R32_TYPELESS) {
        // The view is UINT and the backing is SFLOAT. 3D clear format reinterpretation is still unsupported.
        list->ClearUnorderedAccessViewUint({}, partial, resource, clear_value, 0, nullptr);
        EXPECT(SUCCEEDED(list->Close()));
        queue->ExecuteCommandLists(1, lists);
        return rt::d3d12::last_execute_result(queue.Get()) == leonida::ReplayResult::unsupported;
    }
    const UINT w = std::max(1u, UINT(shape.Width) >> mip);
    const UINT h = std::max(1u, shape.Height >> mip);
    const UINT d = std::max(1u, UINT(shape.DepthOrArraySize) >> mip);
    const UINT pitch = (w * 4 + 255) / 256 * 256;
    const UINT64 total = UINT64(pitch) * h * d;
    D3D12_HEAP_PROPERTIES rb_heap{D3D12_HEAP_TYPE_READBACK};
    D3D12_RESOURCE_DESC rb_desc{};
    rb_desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    rb_desc.Width = total;
    rb_desc.Height = rb_desc.DepthOrArraySize = rb_desc.MipLevels = 1;
    rb_desc.SampleDesc.Count = 1;
    rb_desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    ComPtr<ID3D12Resource> readback;
    EXPECT(SUCCEEDED(device->CreateCommittedResource(&rb_heap, D3D12_HEAP_FLAG_NONE, &rb_desc,
                                                     D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                     IID_PPV_ARGS(&readback))));
    // A known, nonzero sentinel makes loss of a neighboring slice observable even on a newly allocated image.
    constexpr UINT sentinel = std::bit_cast<UINT>(1.0f);
    const UINT seed[4] = {sentinel, 0, 0, 0};
    list->ClearUnorderedAccessViewUint({}, whole, resource, seed, 0, nullptr);
    D3D12_RESOURCE_BARRIER order{};
    order.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    order.UAV.pResource = resource;
    list->ResourceBarrier(1, &order);
    const bool floating = shape.Format == DXGI_FORMAT_R32_FLOAT;
    if (floating) {
        const FLOAT float_value[4] = {42.0f, 0, 0, 0};
        list->ClearUnorderedAccessViewFloat({}, partial, resource, float_value, 0, nullptr);
    } else {
        list->ClearUnorderedAccessViewUint({}, partial, resource, clear_value, 0, nullptr);
    }
    D3D12_RESOURCE_BARRIER to_src{};
    to_src.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    to_src.Transition = {resource, mip, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE};
    list->ResourceBarrier(1, &to_src);
    D3D12_TEXTURE_COPY_LOCATION dst{};
    dst.pResource = readback.Get();
    dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    dst.PlacedFootprint = {0, {shape.Format, w, h, d, pitch}};
    D3D12_TEXTURE_COPY_LOCATION src{};
    src.pResource = resource;
    src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    src.SubresourceIndex = mip;
    list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    D3D12_RESOURCE_BARRIER to_uav = to_src;
    std::swap(to_uav.Transition.StateBefore, to_uav.Transition.StateAfter);
    list->ResourceBarrier(1, &to_uav);
    EXPECT(list->Close() == S_OK);
    queue->ExecuteCommandLists(1, lists);
    EXPECT(rt::d3d12::last_execute_result(queue.Get()) == leonida::ReplayResult::success);
    EXPECT(SUCCEEDED(queue->Signal(fence.Get(), 1)) && SUCCEEDED(fence->SetEventOnCompletion(1, nullptr)));
    void* mapped = nullptr;
    EXPECT(SUCCEEDED(readback->Map(0, nullptr, &mapped)));
    const auto* bytes = static_cast<const std::uint8_t*>(mapped);
    const UINT cleared = floating ? std::bit_cast<UINT>(42.0f) : clear_value[0];
    for (UINT z = 0; z < d; ++z) {
        const UINT want = (z >= first && z < first + writable) ? cleared : sentinel;
        for (UINT y = 0; y < h; ++y) {
            const auto* row = reinterpret_cast<const UINT*>(bytes + (UINT64(z) * h + y) * pitch);
            for (UINT x = 0; x < w; ++x)
                EXPECT(row[x] == want);
        }
    }
    readback->Unmap(0, nullptr);
    return true;
}

bool check_case(ID3D12Device* device, Compute* compute, Kind kind, const D3D12_RESOURCE_DESC& shape,
                D3D12_CPU_DESCRIPTOR_HANDLE source, D3D12_CPU_DESCRIPTOR_HANDLE copy, bool sliced,
                UINT mip, UINT first, UINT count, bool use_default = false) {
    const bool volume = kind == Kind::volume || kind == Kind::float_volume || kind == Kind::atomic_volume;
    const bool floating = kind == Kind::float_volume || kind == Kind::float_array;
    const VkExtent3D extent{UINT(shape.Width) >> mip, std::max(1u, shape.Height >> mip),
                            volume ? std::max(1u, UINT(shape.DepthOrArraySize) >> mip) : shape.DepthOrArraySize};
    const UINT writable = count == ~0u ? extent.depth - first : count;
    const auto desc = uav(kind, mip, first, count, floating ? DXGI_FORMAT_R32_FLOAT : DXGI_FORMAT_R32_UINT);
    ComPtr<ID3D12Resource> resource;
    EXPECT(create_texture(device, shape, &resource));
    device->CreateUnorderedAccessView(resource.Get(), nullptr, use_default ? nullptr : &desc, source);
    const auto key = read_slot(source);
    const bool partial = volume && (first || writable != extent.depth);
    if (partial && !sliced) {
        EXPECT(key == g::ImageWords{});
        return true;
    }
    g::Image decoded{};
    EXPECT(g::decode(key, decoded) && decoded.first_mip == mip && decoded.mip_count == 1);
    EXPECT(bool(key[4] & g::volume_uav_flag) == volume);
    EXPECT(key[7] == (volume ? g::volume_uav_word(first, writable) : 0));
    EXPECT(g::partial_volume_uav(key, decoded) == partial);
    EXPECT(!volume || (decoded.depth == shape.DepthOrArraySize && decoded.first_layer == 0 && decoded.layer_count == 1));
    EXPECT((kind != Kind::array && kind != Kind::float_array) ||
           (decoded.first_layer == first && decoded.layer_count == writable));
    device->CopyDescriptorsSimple(1, copy, source, uav_heap);
    EXPECT(read_slot(copy) == key);
    if (partial) {
        // Reuse the original slot for the whole-mip seed; the copied descriptor must still select its W range.
        auto whole = desc;
        whole.Texture3D.FirstWSlice = 0;
        whole.Texture3D.WSize = ~0u;
        device->CreateUnorderedAccessView(resource.Get(), nullptr, &whole, source);
        EXPECT(check_partial_clear(device, resource.Get(), source, copy, shape, mip, first, writable));
    }
    if (partial && first > 0) {
        // A second view of the same volume and mip must not collide with the copied T#.
        auto other = desc;
        other.Texture3D.FirstWSlice = 0;
        device->CreateUnorderedAccessView(resource.Get(), nullptr, &other, source);
        EXPECT(read_slot(source) != key && g::is_image(read_slot(source)));
    }
    if (partial && kind == Kind::float_volume && shape.Format == DXGI_FORMAT_R32_FLOAT) {
        // The typed SRV of this same mip must remain unsliced and must not replace the live partial UAV.
        D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
        srv.Format = shape.Format;
        srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE3D;
        srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srv.Texture3D = {mip, 1, 0.0f};
        device->CreateShaderResourceView(resource.Get(), &srv, source);
        const auto srv_key = read_slot(source);
        g::Image sampled{};
        EXPECT(g::decode(srv_key, sampled) && sampled.type == g::ImageType::image_3d &&
               sampled.address == decoded.address && sampled.first_mip == mip && sampled.mip_count == 1 &&
               sampled.depth == decoded.depth && sampled.first_layer == 0 && sampled.layer_count == 1);
        EXPECT(!(srv_key[4] & g::volume_uav_flag) && srv_key[7] == 0 && srv_key != read_slot(copy));
    }
    EXPECT(flush_descriptors(device));
    if (!compute) return true;  // CTest exercises the complete descriptor setup, without translated SPIR-V.

    std::vector<std::uint32_t> input(extent.width * extent.height * extent.depth);
    auto bits = [&](UINT value) { return floating ? std::bit_cast<UINT>(float(value)) : value; };
    for (UINT i = 0; i < input.size(); ++i) input[i] = bits(17 + i);
    std::vector<std::uint32_t> expected = input, output(input.size()), host(input.size());
    for (UINT z = first; z < first + writable; ++z)
        for (UINT y = 0; y < extent.height; ++y)
            for (UINT x = 0; x < extent.width; ++x) {
                const UINT i = (z * extent.height + y) * extent.width + x;
                const UINT initial = 17 + i;
                const UINT add = 1000 + x + 16 * y + 256 * (z - first);
                if (kind == Kind::atomic_volume)
                    expected[i] = add + (initial << 16) + ((initial + 5) << 8) + 7 + 99;
                else
                    expected[i] = bits(initial + add + (kind == Kind::float_volume && x % 2 == 0 ? initial : 0));
            }
    EXPECT(compute->run(resource.Get(), mip, extent, {read_slot(copy), UINT(kind)}, input, output));
    for (UINT i = 0; i < output.size(); ++i) {
        if (output[i] == expected[i]) continue;
        std::fprintf(stderr, "kind %u mip %u first %u count %u texel %u: got %#x, expected %#x\n",
                     UINT(kind), mip, first, count, i, output[i], expected[i]);
        return false;
    }
    if (!floating && kind != Kind::atomic_volume) {
        // WARP did not clip out-of-range W accesses to WSize in the native probe. Compare documented in-view
        // D3D12 accesses and independently require Vulkan to preserve every non-view slice, even under overdispatch.
        EXPECT(uav_oracle::run(shape, use_default ? nullptr : &desc, input, host,
                               volume ? writable : kind == Kind::one_d ? 1 : 0));
        EXPECT(output == host);
    }
    return true;
}

bool check_refusals(ID3D12Device* device, D3D12_CPU_DESCRIPTOR_HANDLE slot) {
    using enum Kind;
    const auto bad = [&](ID3D12Resource* resource, const D3D12_UNORDERED_ACCESS_VIEW_DESC* view) {
        std::memset(reinterpret_cast<void*>(slot.ptr), 0x55, sizeof(g::ImageWords));
        device->CreateUnorderedAccessView(resource, nullptr, view, slot);
        return read_slot(slot) == g::ImageWords{};
    };
    ComPtr<ID3D12Resource> line, layers, depth, typeless, read_only;
    EXPECT(create_texture(device, texture(one_d, 1), &line));
    EXPECT(create_texture(device, texture(array, 4), &layers));
    EXPECT(create_texture(device, texture(volume, 8), &depth));
    EXPECT(create_texture(device, texture(volume, 8, DXGI_FORMAT_R32_TYPELESS), &typeless));
    auto shape = texture(one_d, 1);
    shape.Flags = D3D12_RESOURCE_FLAG_NONE;
    EXPECT(create_texture(device, shape, &read_only));
    EXPECT(bad(read_only.Get(), nullptr) && bad(typeless.Get(), nullptr));
    const auto good = uav(one_d, 0, 0, 1);
    EXPECT(bad(nullptr, &good) && bad(depth.Get(), &good));
    for (const auto view : {uav(one_d, 3, 0, 1), uav(one_d, 0, 0, 1, DXGI_FORMAT_R16_FLOAT)})
        EXPECT(bad(line.Get(), &view));
    for (const auto view : {uav(array, 0, 4, 1), uav(array, 0, 1, 0), uav(array, 0, 2, 3), uav(array, 3, 0, 1)})
        EXPECT(bad(layers.Get(), &view));
    for (const auto view : {uav(volume, 3, 0, 1), uav(volume, 2, 2, 1), uav(volume, 1, 1, 0), uav(volume, 1, 2, 3)})
        EXPECT(bad(depth.Get(), &view));
    auto rgba = texture(volume, 4, DXGI_FORMAT_R8G8B8A8_TYPELESS);
    ComPtr<ID3D12Resource> srgb;
    EXPECT(create_texture(device, rgba, &srgb));
    const auto srgb_view = uav(volume, 0, 0, ~0u, DXGI_FORMAT_R8G8B8A8_UNORM_SRGB);
    EXPECT(bad(srgb.Get(), &srgb_view));
    ComPtr<ID3D12Resource> refused;
    const auto typed_srgb = texture(volume, 4, DXGI_FORMAT_R8G8B8A8_UNORM_SRGB);
    ComPtr<ID3D12Resource> typed, warp_typed;
    uav_oracle::detail::Context host;
    const bool created = create_texture(device, typed_srgb, &typed);
    if (host.prepare_device() == uav_oracle::Status::ready)
        EXPECT(created == create_texture(host.device.Get(), typed_srgb, &warp_typed));
    if (created) {
        EXPECT(bad(typed.Get(), &srgb_view));
        const auto raw_view = uav(volume, 0, 0, ~0u, DXGI_FORMAT_R8G8B8A8_UNORM);
        EXPECT(!bad(typed.Get(), &raw_view));
    }
    auto invalid_shape = texture(volume, 4);
    invalid_shape.MipLevels = 8;
    EXPECT(!create_texture(device, invalid_shape, &refused) && !refused);
    invalid_shape = texture(one_d, 1);
    invalid_shape.Width = (1ull << 32) + 8;
    EXPECT(!create_texture(device, invalid_shape, &refused) && !refused);
    return true;
}

bool check_formatless_features(ID3D12Device* device, D3D12_CPU_DESCRIPTOR_HANDLE slot, bool sliced) {
    using enum Kind;
    const auto check = [&](Kind kind, const D3D12_RESOURCE_DESC& shape, UINT first, UINT count) {
        ComPtr<ID3D12Resource> resource;
        EXPECT(create_texture(device, shape, &resource));
        const auto view = uav(kind, 1, first, count);
        device->CreateUnorderedAccessView(resource.Get(), nullptr, &view, slot);
        const auto valid = read_slot(slot);
        g::Image decoded{};
        EXPECT(g::decode(valid, decoded) && decoded.first_mip == 1 && decoded.mip_count == 1);
        for (const auto missing : {formatless_read, formatless_write, formatless_read | formatless_write}) {
            std::memcpy(reinterpret_cast<void*>(slot.ptr), valid.data(), sizeof(valid));
            missing_format_features = missing;
            device->CreateUnorderedAccessView(resource.Get(), nullptr, &view, slot);
            missing_format_features = 0;
            if (read_slot(slot) != g::ImageWords{}) {
                std::fprintf(stderr, "kind %u first slice %u: accepted mip 1 UAV without format features %#llx\n",
                             UINT(kind), first, static_cast<unsigned long long>(missing));
                return false;
            }
        }
        return true;
    };
    EXPECT(check(one_d, texture(one_d, 1), 0, 1));
    EXPECT(check(array, texture(array, 4), 1, 2));
    // The mask applies to the R32_UINT view, not its typeless resource's backing format.
    const auto volume_shape = texture(volume, 8, DXGI_FORMAT_R32_TYPELESS);
    EXPECT(check(volume, volume_shape, 0, ~0u));
    if (sliced) EXPECT(check(volume, volume_shape, 1, 2));
    return true;
}

bool check_packed_formatless_exemption(ID3D12Device* device, vktest::Device& vk,
                                       D3D12_CPU_DESCRIPTOR_HANDLE slot, bool sliced) {
    VkFormatProperties features{};
    vk.instance_fn<PFN_vkGetPhysicalDeviceFormatProperties>("vkGetPhysicalDeviceFormatProperties")(
        vk.physical, VK_FORMAT_E5B9G9R9_UFLOAT_PACK32, &features);
    if (features.optimalTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT) {
        std::puts("Packed R32_UINT exemption not exercised: native RGB9E5 storage");
        return true;
    }
    constexpr auto format = DXGI_FORMAT_R9G9B9E5_SHAREDEXP;
    ComPtr<ID3D12Resource> resource;
    if (!create_texture(device, texture(Kind::volume, 8, format), &resource)) {
        std::puts("Packed R32_UINT exemption not exercised: no emulated RGB9E5 3D storage");
        return true;
    }
    const UINT first = sliced ? 1 : 0, count = sliced ? 2 : 4;
    const auto view = uav(Kind::volume, 1, first, count, format);
    missing_format_features = formatless_read | formatless_write;
    device->CreateUnorderedAccessView(resource.Get(), nullptr, &view, slot);
    missing_format_features = 0;
    const auto key = read_slot(slot);
    g::Image decoded{};
    EXPECT(g::decode(key, decoded) && decoded.first_mip == 1 && (key[4] & g::volume_uav_flag));
    EXPECT(key[7] == (g::volume_uav_word(first, count) | g::packed_uav_flag));
    return true;
}

bool check_standalone(bool sliced) {
    ComPtr<ID3D12Device> device;
    EXPECT(SUCCEEDED(rt::d3d12::create_vulkan_device_standalone(IID_PPV_ARGS(&device))));
    ComPtr<ID3D12Resource> volume;
    EXPECT(create_texture(device.Get(), texture(Kind::volume, 8), &volume));
    const D3D12_DESCRIPTOR_HEAP_DESC heap_desc{uav_heap, 1, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
    ComPtr<ID3D12DescriptorHeap> heap;
    EXPECT(SUCCEEDED(device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&heap))));
    const auto slot = heap->GetCPUDescriptorHandleForHeapStart();
    device->CreateUnorderedAccessView(volume.Get(), nullptr, nullptr, slot);
    EXPECT(g::is_image(read_slot(slot)));
    const auto view = uav(Kind::volume, 1, 1, 2);
    device->CreateUnorderedAccessView(volume.Get(), nullptr, &view, slot);
    EXPECT(read_slot(slot)[7] == (sliced ? g::volume_uav_word(1, 2) : 0));
    return true;
}

bool check_cpu_clear(ID3D12Device* device, vktest::Device& vk, D3D12_CPU_DESCRIPTOR_HANDLE slot) {
    EXPECT(!rt::d3d12::bindless_set(device));
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> list;
    const D3D12_COMMAND_QUEUE_DESC queue_desc{};
    EXPECT(SUCCEEDED(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue))));
    EXPECT(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator))));
    EXPECT(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.Get(), nullptr,
                                               IID_PPV_ARGS(&list))));
    ComPtr<ID3D12Resource> image;
    EXPECT(create_texture(device, texture(Kind::two_d, 1), &image));
    device->CreateUnorderedAccessView(image.Get(), nullptr, nullptr, slot);
    EXPECT(g::is_image(read_slot(slot)));
    const UINT value[] = {42, 0, 0, 0};
    list->ClearUnorderedAccessViewUint({}, slot, image.Get(), value, 0, nullptr);
    EXPECT(SUCCEEDED(list->Close()));
    ID3D12CommandList* lists[] = {list.Get()};
    queue->ExecuteCommandLists(1, lists);
    EXPECT(rt::d3d12::last_execute_result(queue.Get()) == leonida::ReplayResult::success);
    void* mapped = nullptr;
    const VkBuffer output = vk.buffer(32 * sizeof(UINT), VK_BUFFER_USAGE_TRANSFER_DST_BIT, &mapped);
    EXPECT(output);
    EXPECT(vk.run([&](VkCommandBuffer cb) {
        const VkMemoryBarrier memory{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_TRANSFER_WRITE_BIT,
                                     VK_ACCESS_TRANSFER_READ_BIT};
        vk.fn<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier")(cb, VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &memory, 0, nullptr, 0, nullptr);
        vk.read_back(cb, rt::d3d12::vulkan_image(image.Get()), VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_ASPECT_COLOR_BIT,
                     8, 4, 1, output, 0);
        return true;
    }));
    const auto* pixels = static_cast<const UINT*>(mapped);
    EXPECT(std::all_of(pixels, pixels + 32, [](UINT pixel) { return pixel == 42; }));

    // A typeless backing may reinterpret bits in a transfer clear; an unsupported clear must not be submitted.
    EXPECT(SUCCEEDED(allocator->Reset()));
    EXPECT(SUCCEEDED(list->Reset(allocator.Get(), nullptr)));
    ComPtr<ID3D12Resource> volume;
    EXPECT(create_texture(device, texture(Kind::volume, 4, DXGI_FORMAT_R32_TYPELESS), &volume));
    const auto view = uav(Kind::volume, 0, 0, ~0u);
    device->CreateUnorderedAccessView(volume.Get(), nullptr, &view, slot);
    EXPECT(g::is_image(read_slot(slot)));
    list->ClearUnorderedAccessViewUint({}, slot, volume.Get(), value, 0, nullptr);
    EXPECT(SUCCEEDED(list->Close()));
    queue->ExecuteCommandLists(1, lists);
    EXPECT(rt::d3d12::last_execute_result(queue.Get()) == leonida::ReplayResult::unsupported);
    return true;
}

bool test(vktest::Device& vk, const char* shader, bool enable_slices) {
    rt::d3d12::VulkanDeviceInfo info{};
    real_instance_proc = vk.instance_proc;
    real_format_properties = vk.instance_fn<PFN_vkGetPhysicalDeviceFormatProperties2>(
        "vkGetPhysicalDeviceFormatProperties2");
    EXPECT(real_format_properties);
    info.get_instance_proc = masked_instance_proc;
    info.instance = vk.instance;
    info.physical = vk.physical;
    info.device = vk.device;
    info.queue = vk.queue;
    info.queue_family = vk.family;
    info.bindless = vk.bindless;
    info.bindless_partially_bound = vk.partially_bound;
    info.robust_image_access = vk.robust_image_access;
    info.sliced_3d_uav = enable_slices && vk.sliced_3d_uav;
    ComPtr<ID3D12Device> device;
    EXPECT(SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&device))));
    const D3D12_DESCRIPTOR_HEAP_DESC heap_desc{uav_heap, 2, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
    ComPtr<ID3D12DescriptorHeap> heap;
    EXPECT(SUCCEEDED(device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&heap))));
    const auto source = heap->GetCPUDescriptorHandleForHeapStart();
    const D3D12_CPU_DESCRIPTOR_HANDLE copy{source.ptr + device->GetDescriptorHandleIncrementSize(uav_heap)};
    if (!enable_slices) {
        ComPtr<ID3D12Resource> volume;
        EXPECT(create_texture(device.Get(), texture(Kind::volume, 8), &volume));
        device->CreateUnorderedAccessView(volume.Get(), nullptr, nullptr, source);
        EXPECT(g::is_image(read_slot(source)));
        const auto view = uav(Kind::volume, 1, 1, 2);
        device->CreateUnorderedAccessView(volume.Get(), nullptr, &view, source);
        EXPECT(read_slot(source) == g::ImageWords{});
        info.bindless = false;
        ComPtr<ID3D12Device> cpu_only;
        EXPECT(SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&cpu_only))));
        ComPtr<ID3D12DescriptorHeap> cpu_heap;
        EXPECT(SUCCEEDED(cpu_only->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&cpu_heap))));
        return check_cpu_clear(cpu_only.Get(), vk, cpu_heap->GetCPUDescriptorHandleForHeapStart());
    }

    using enum Kind;
    EXPECT(check_refusals(device.Get(), source));
    EXPECT(check_formatless_features(device.Get(), source, info.sliced_3d_uav));
    EXPECT(check_packed_formatless_exemption(device.Get(), vk, source, info.sliced_3d_uav));
    Compute compute(vk);
    if (shader) EXPECT(compute.create(device.Get(), shader));
    auto* dispatch = shader ? &compute : nullptr;
    EXPECT(check_case(device.Get(), dispatch, one_d, texture(one_d, 1), source, copy, false, 0, 0, 1, true));
    EXPECT(check_case(device.Get(), dispatch, one_d, texture(one_d, 1), source, copy, false, 2, 0, 1));
    EXPECT(check_case(device.Get(), dispatch, one_d, texture(array, 4), source, copy, false, 1, 0, 1));
    EXPECT(check_case(device.Get(), dispatch, array, texture(array, 4), source, copy, false, 0, 0, 4, true));
    EXPECT(check_case(device.Get(), dispatch, array, texture(array, 4), source, copy, false, 1, 1, 2));
    EXPECT(check_case(device.Get(), dispatch, array, texture(array, 4), source, copy, false, 1, 1, ~0u));
    EXPECT(check_case(device.Get(), dispatch, float_array, texture(float_array, 4, DXGI_FORMAT_R32_FLOAT),
                      source, copy, false, 1, 1, 2));
    EXPECT(check_case(device.Get(), dispatch, volume, texture(volume, 8), source, copy, vk.sliced_3d_uav, 0, 0, 8, true));
    EXPECT(check_case(device.Get(), dispatch, volume, texture(volume, 8, DXGI_FORMAT_R32_TYPELESS), source, copy,
                      vk.sliced_3d_uav, 1, 1, 2));
    EXPECT(check_case(device.Get(), dispatch, volume, texture(volume, 8), source, copy, vk.sliced_3d_uav, 1, 2, ~0u));
    EXPECT(check_case(device.Get(), dispatch, float_volume, texture(float_volume, 8, DXGI_FORMAT_R32_FLOAT),
                      source, copy, vk.sliced_3d_uav, 1, 1, 2));
    EXPECT(check_case(device.Get(), dispatch, atomic_volume, texture(atomic_volume, 8),
                      source, copy, vk.sliced_3d_uav, 1, 0, 4));
    EXPECT(check_case(device.Get(), dispatch, atomic_volume, texture(atomic_volume, 8),
                      source, copy, vk.sliced_3d_uav, 1, 1, 2));
    EXPECT(check_case(device.Get(), dispatch, two_d, texture(two_d, 1), source, copy, false, 1, 0, 1));
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: d3d12_texture_uav_test <cs_texture_uav.spv>|--descriptors-only\n");
        return 2;
    }
    vktest::Device vk;
    // Exercise Vulkan 1.3 per-format access even on a logical device without the legacy feature bits.
    const auto status = vk.initialize(true, false);
    if (status == vktest::DeviceStatus::unavailable && !vk.validation_errors()) {
        std::printf("SKIP: no Vulkan 1.3 device with a graphics queue\n");
        return 77;
    }
    if (status != vktest::DeviceStatus::ready) return 1;
    const char* shader = std::strcmp(argv[1], "--descriptors-only") == 0 ? nullptr : argv[1];
    VkPhysicalDeviceProperties props{};
    vk.instance_fn<PFN_vkGetPhysicalDeviceProperties>("vkGetPhysicalDeviceProperties")(vk.physical, &props);
    if (!vk.bindless || !vk.robust_image_access || (shader && props.limits.maxPushConstantsSize < push_bytes())) {
        std::printf("SKIP: requires bindless descriptors and robustImageAccess%s\n",
                    shader ? " and 144 bytes of push constants" : "");
        return vk.validation_errors() ? 1 : 77;
    }
    if (shader) {
        const auto warp = uav_oracle::prepare();
        if (warp == uav_oracle::Status::unavailable) {
            std::printf("SKIP: no D3D12 WARP adapter\n");
            return vk.validation_errors() ? 1 : 77;
        }
        if (warp != uav_oracle::Status::ready) return 1;
    }
    if (!test(vk, shader, false) || !test(vk, shader, true) || !check_standalone(vk.sliced_3d_uav) ||
        vk.validation_errors())
        return 1;
    if (!shader) {
        std::puts("texture UAV descriptors: creation, copy, range/refusal and clear checks passed");
        return 0;
    }
    if (!vk.sliced_3d_uav) std::printf("SKIP: sliced 3D UAV GPU writes (feature unavailable); rejection verified\n");
    std::printf("texture UAVs: %u GPU cases (%u compared with WARP), format/range/refusal checks passed\n",
                vk.sliced_3d_uav ? 14u : 10u, vk.sliced_3d_uav ? 10u : 8u);
    return 0;
}
