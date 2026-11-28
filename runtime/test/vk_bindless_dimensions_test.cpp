// More live images of each sampled dimension than 4096, through the bindless set 0 (src/vulkan_bindless.h), at the
// Vulkan level, and the index allocator behind them.
//
// An image keeps its index in its dimension's array for as long as it lives, so the array of a dimension must hold
// every live image of it a title has. A world run kept 4090 live 1D textures and filled the 4096 1D indices the set
// once stopped at (bindless::size_bindings capped 1D and 3D at 4096 whatever the device held); every later 1D T# then
// read the fallback's zeros. For 1D, 2D-array and 3D images alike this test sizes the set from the device's limits
// as the D3D12 device does, gives 5000 distinct images (one texel each, holding its own number) an index each,
// and has test/glsl/cs_bindless_dimensions.comp read every one of them through its index: each must read its own
// number. Then, on the same set:
//   - the rest of the array is filled: the index after the last is refused, and image_refusal says the array of
//     that dimension is full (the D3D12 device's "unbound T#s by reason" line names that reason);
//   - an image removed while the table has no changes the GPU lacks gives its index back at once, and one removed
//     while it has (an erased key) gives it back at the next flush, not before.
// Exit 77 without a Vulkan device with the bindless features or without the shader (no glslangValidator), and for
// a dimension whose device limits hold fewer than 5000 images (the test says so).
// LEONIDA_VK_VALIDATION=1 runs under VK_LAYER_KHRONOS_validation; a validation error fails the test.
#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
#include "vk_test_device.h"
#include "shader_fixture_path.h"
#include "../src/vulkan_bindless.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <vector>

#define EXPECT(condition)                                                    \
    do {                                                                     \
        if (!(condition)) {                                                  \
            std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition);     \
            return 1;                                                        \
        }                                                                    \
    } while (0)

namespace {

using namespace leonida;

constexpr std::uint32_t live_images = 5000;  // more than the 4096 1D and 3D indices of the old sizes

std::vector<std::uint32_t> load_spirv(const char* path) {
    std::ifstream file(vktest::shader_fixture_path(path), std::ios::binary);
    const std::vector<char> bytes{std::istreambuf_iterator<char>(file), {}};
    std::vector<std::uint32_t> words(bytes.size() / 4);
    if (!words.empty()) std::memcpy(words.data(), bytes.data(), words.size() * 4);
    return words;
}

const char* name(ImageDimension dimension) {
    return dimension == ImageDimension::dim_1d ? "1D" : dimension == ImageDimension::dim_2d_array ? "2D array" : "3D";
}

// The images of one dimension, `count` texels in all, each in its own image (3D) or array layer (1D, 2D array), and a
// view of each: what the test reads through the set. Destroys what it made.
struct Images {
    vktest::Device& d;
    std::vector<VkImage> images;
    std::vector<VkDeviceMemory> memory;
    std::vector<VkImageView> views;
    struct Texel {
        VkImage image;
        std::uint32_t layer;
    };
    std::vector<Texel> texels;

    explicit Images(vktest::Device& device) : d(device) {}
    Images(const Images&) = delete;
    Images& operator=(const Images&) = delete;
    ~Images() {
        for (const VkImageView view : views) d.fn<PFN_vkDestroyImageView>("vkDestroyImageView")(d.device, view, nullptr);
        for (const VkImage image : images) d.fn<PFN_vkDestroyImage>("vkDestroyImage")(d.device, image, nullptr);
        for (const VkDeviceMemory m : memory) d.fn<PFN_vkFreeMemory>("vkFreeMemory")(d.device, m, nullptr);
    }

    bool make(ImageDimension dimension, std::uint32_t count, std::uint32_t max_layers,
              const VkPhysicalDeviceMemoryProperties& properties) {
        const bool volume = dimension == ImageDimension::dim_3d;
        const std::uint32_t per_image = volume ? 1 : std::min(max_layers, count);
        // Images in groups that share one allocation (each a few MiB at most).
        constexpr std::uint32_t group = 512;
        for (std::uint32_t first = 0; first < count;) {
            std::vector<VkImage> made;
            std::uint32_t in_group = 0;
            for (; in_group < group && first < count; ++in_group) {
                VkImageCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
                info.imageType = volume ? VK_IMAGE_TYPE_3D
                                 : dimension == ImageDimension::dim_1d ? VK_IMAGE_TYPE_1D : VK_IMAGE_TYPE_2D;
                info.format = VK_FORMAT_R32_SFLOAT;
                info.extent = {1, 1, 1};
                info.mipLevels = 1;
                info.arrayLayers = std::min(per_image, count - first);
                info.samples = VK_SAMPLE_COUNT_1_BIT;
                info.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
                VkImage image = VK_NULL_HANDLE;
                if (d.fn<PFN_vkCreateImage>("vkCreateImage")(d.device, &info, nullptr, &image) != VK_SUCCESS)
                    return false;
                images.push_back(image);
                made.push_back(image);
                for (std::uint32_t layer = 0; layer < info.arrayLayers; ++layer) texels.push_back({image, layer});
                first += info.arrayLayers;
            }
            VkMemoryRequirements requirements{};
            d.fn<PFN_vkGetImageMemoryRequirements>("vkGetImageMemoryRequirements")(d.device, made[0], &requirements);
            const VkDeviceSize stride =
                (requirements.size + requirements.alignment - 1) / requirements.alignment * requirements.alignment;
            VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
            allocation.allocationSize = stride * made.size();
            allocation.memoryTypeIndex = 0xffffffffu;
            for (std::uint32_t t = 0; t < properties.memoryTypeCount; ++t)
                if (requirements.memoryTypeBits & (1u << t)) {
                    allocation.memoryTypeIndex = t;
                    break;
                }
            VkDeviceMemory block = VK_NULL_HANDLE;
            if (allocation.memoryTypeIndex == 0xffffffffu ||
                d.fn<PFN_vkAllocateMemory>("vkAllocateMemory")(d.device, &allocation, nullptr, &block) != VK_SUCCESS)
                return false;
            memory.push_back(block);
            for (std::size_t k = 0; k < made.size(); ++k) {
                VkMemoryRequirements each{};
                d.fn<PFN_vkGetImageMemoryRequirements>("vkGetImageMemoryRequirements")(d.device, made[k], &each);
                if (each.size > stride || !(each.memoryTypeBits & (1u << allocation.memoryTypeIndex)) ||
                    d.fn<PFN_vkBindImageMemory>("vkBindImageMemory")(d.device, made[k], block, stride * k) !=
                        VK_SUCCESS)
                    return false;
            }
        }
        for (const Texel& texel : texels) {
            VkImageViewCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
            info.image = texel.image;
            info.viewType = volume ? VK_IMAGE_VIEW_TYPE_3D
                            : dimension == ImageDimension::dim_1d ? VK_IMAGE_VIEW_TYPE_1D : VK_IMAGE_VIEW_TYPE_2D_ARRAY;
            info.format = VK_FORMAT_R32_SFLOAT;
            info.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, texel.layer, 1};
            VkImageView view = VK_NULL_HANDLE;
            if (d.fn<PFN_vkCreateImageView>("vkCreateImageView")(d.device, &info, nullptr, &view) != VK_SUCCESS)
                return false;
            views.push_back(view);
        }
        return true;
    }

    // Texel i holds i + 1, in GENERAL (the layout the set's descriptors name).
    void record_fill(VkCommandBuffer cb) const {
        std::vector<VkImageMemoryBarrier> barriers;
        for (const VkImage image : images) {
            VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
            barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.image = image;
            barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, VK_REMAINING_ARRAY_LAYERS};
            barriers.push_back(barrier);
        }
        d.fn<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier")(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                                               VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                                                               nullptr, std::uint32_t(barriers.size()), barriers.data());
        for (std::size_t i = 0; i < texels.size(); ++i) {
            VkClearColorValue value{};
            value.float32[0] = float(i + 1);
            const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, texels[i].layer, 1};
            d.fn<PFN_vkCmdClearColorImage>("vkCmdClearColorImage")(cb, texels[i].image, VK_IMAGE_LAYOUT_GENERAL,
                                                                   &value, 1, &range);
        }
        VkMemoryBarrier filled{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        filled.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        filled.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        d.fn<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier")(cb, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                                               VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &filled, 0,
                                                               nullptr, 0, nullptr);
    }
};

// The compute pipeline over set 0 (the bindless set's layout) and set 1 (result and indices).
struct Reader {
    vktest::Device& d;
    VkShaderModule module = VK_NULL_HANDLE;
    VkDescriptorSetLayout set1_layout = VK_NULL_HANDLE;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    VkDescriptorSet set1 = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    void* result_map = nullptr;
    void* index_map = nullptr;

    explicit Reader(vktest::Device& device) : d(device) {}
    Reader(const Reader&) = delete;
    Reader& operator=(const Reader&) = delete;
    ~Reader() {
        const VkDevice dev = d.device;
        if (pipeline) d.fn<PFN_vkDestroyPipeline>("vkDestroyPipeline")(dev, pipeline, nullptr);
        if (layout) d.fn<PFN_vkDestroyPipelineLayout>("vkDestroyPipelineLayout")(dev, layout, nullptr);
        if (pool) d.fn<PFN_vkDestroyDescriptorPool>("vkDestroyDescriptorPool")(dev, pool, nullptr);
        if (set1_layout) d.fn<PFN_vkDestroyDescriptorSetLayout>("vkDestroyDescriptorSetLayout")(dev, set1_layout, nullptr);
        if (module) d.fn<PFN_vkDestroyShaderModule>("vkDestroyShaderModule")(dev, module, nullptr);
    }

    bool make(VkDescriptorSetLayout set0_layout, const std::vector<std::uint32_t>& code) {
        const VkDevice dev = d.device;
        VkShaderModuleCreateInfo module_info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        module_info.codeSize = code.size() * 4;
        module_info.pCode = code.data();
        if (d.fn<PFN_vkCreateShaderModule>("vkCreateShaderModule")(dev, &module_info, nullptr, &module) != VK_SUCCESS)
            return false;
        const VkBuffer result = d.buffer(live_images * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &result_map);
        const VkBuffer indices = d.buffer(live_images * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &index_map);
        if (!result || !indices) return false;
        const VkDescriptorSetLayoutBinding bindings[2] = {
            {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
            {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}};
        VkDescriptorSetLayoutCreateInfo set1_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        set1_info.bindingCount = 2;
        set1_info.pBindings = bindings;
        if (d.fn<PFN_vkCreateDescriptorSetLayout>("vkCreateDescriptorSetLayout")(dev, &set1_info, nullptr,
                                                                                 &set1_layout) != VK_SUCCESS)
            return false;
        const VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2};
        VkDescriptorPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        pool_info.maxSets = 1;
        pool_info.poolSizeCount = 1;
        pool_info.pPoolSizes = &pool_size;
        if (d.fn<PFN_vkCreateDescriptorPool>("vkCreateDescriptorPool")(dev, &pool_info, nullptr, &pool) != VK_SUCCESS)
            return false;
        VkDescriptorSetAllocateInfo allocate{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        allocate.descriptorPool = pool;
        allocate.descriptorSetCount = 1;
        allocate.pSetLayouts = &set1_layout;
        if (d.fn<PFN_vkAllocateDescriptorSets>("vkAllocateDescriptorSets")(dev, &allocate, &set1) != VK_SUCCESS)
            return false;
        const VkDescriptorBufferInfo infos[2] = {{result, 0, VK_WHOLE_SIZE}, {indices, 0, VK_WHOLE_SIZE}};
        VkWriteDescriptorSet writes[2]{};
        for (std::uint32_t b = 0; b < 2; ++b) {
            writes[b] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            writes[b].dstSet = set1;
            writes[b].dstBinding = b;
            writes[b].descriptorCount = 1;
            writes[b].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[b].pBufferInfo = &infos[b];
        }
        d.fn<PFN_vkUpdateDescriptorSets>("vkUpdateDescriptorSets")(dev, 2, writes, 0, nullptr);
        const VkDescriptorSetLayout layouts[2] = {set0_layout, set1_layout};
        const VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT, 0, 8};
        VkPipelineLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        layout_info.setLayoutCount = 2;
        layout_info.pSetLayouts = layouts;
        layout_info.pushConstantRangeCount = 1;
        layout_info.pPushConstantRanges = &push;
        if (d.fn<PFN_vkCreatePipelineLayout>("vkCreatePipelineLayout")(dev, &layout_info, nullptr, &layout) !=
            VK_SUCCESS)
            return false;
        VkComputePipelineCreateInfo pipeline_info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        pipeline_info.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        pipeline_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        pipeline_info.stage.module = module;
        pipeline_info.stage.pName = "main";
        pipeline_info.layout = layout;
        return d.fn<PFN_vkCreateComputePipelines>("vkCreateComputePipelines")(dev, VK_NULL_HANDLE, 1, &pipeline_info,
                                                                               nullptr, &pipeline) == VK_SUCCESS;
    }

    void record(VkCommandBuffer cb, VkDescriptorSet set0, std::uint32_t dimension, std::uint32_t count) const {
        d.fn<PFN_vkCmdBindPipeline>("vkCmdBindPipeline")(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        const VkDescriptorSet sets[2] = {set0, set1};
        d.fn<PFN_vkCmdBindDescriptorSets>("vkCmdBindDescriptorSets")(cb, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 2,
                                                                     sets, 0, nullptr);
        const std::uint32_t pass[2] = {dimension, count};
        d.fn<PFN_vkCmdPushConstants>("vkCmdPushConstants")(cb, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, pass);
        d.fn<PFN_vkCmdDispatch>("vkCmdDispatch")(cb, count, 1, 1);
        VkMemoryBarrier host{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        host.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        host.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        d.fn<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier")(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                                               VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &host, 0, nullptr, 0,
                                                               nullptr);
    }
};

int run(vktest::Device& d, const VkPhysicalDeviceMemoryProperties& memory, const BindlessFeatures& features,
        std::uint32_t max_layers, const std::vector<std::uint32_t>& code, ImageDimension dimension) {
    BindlessSet set(d.device_proc, d.device, memory, features);
    EXPECT(set.ok());
    const std::uint32_t capacity = set.image_capacity(dimension);
    std::printf("%s: %u indices in the set\n", name(dimension), capacity);
    // Every live image needs an index, and index 0 is the fallback.
    EXPECT(capacity > live_images);

    Images images(d);
    EXPECT(images.make(dimension, live_images, max_layers, memory));
    std::vector<std::uint32_t> indices;
    for (const VkImageView view : images.views) {
        const std::uint32_t index = set.add_image(dimension, view, VK_IMAGE_LAYOUT_GENERAL);
        EXPECT(index != 0);
        indices.push_back(index);
    }
    Reader reader(d);
    EXPECT(reader.make(set.layout(), code));
    std::memcpy(reader.index_map, indices.data(), indices.size() * 4);
    std::memset(reader.result_map, 0xcd, live_images * 4);
    const std::uint32_t pass = dimension == ImageDimension::dim_1d ? 0 : dimension == ImageDimension::dim_2d_array ? 1 : 2;
    bool flushed = false;
    const bool ran = d.run([&](VkCommandBuffer cb) {
        set.record_fallback_init(cb);
        flushed = set.flush(cb, 0, [] { return std::uint64_t(0); });
        images.record_fill(cb);
        reader.record(cb, set.set(), pass, live_images);
        return true;
    });
    set.table_upload_result(ran);
    EXPECT(ran && flushed);
    std::vector<std::uint32_t> result(live_images);
    std::memcpy(result.data(), reader.result_map, live_images * 4);
    std::uint32_t wrong = 0;
    for (std::uint32_t i = 0; i < live_images; ++i) {
        float red = 0;
        std::memcpy(&red, &result[i], 4);
        if (red != float(i + 1) && wrong++ < 4)
            std::fprintf(stderr, "%s image %u at index %u read %g, expected %u\n", name(dimension), i, indices[i],
                         double(red), i + 1);
    }
    EXPECT(wrong == 0);
    std::printf("%s: %u live images, each read its own texel through its index (the last at %u)\n", name(dimension),
                live_images, indices.back());

    // The rest of the array (the same view again: an index each, as distinct T#s of one texture get), then one
    // more, refused for the reason that the array of its dimension is full.
    std::uint32_t more = 0;
    while (set.add_image(dimension, images.views[0], VK_IMAGE_LAYOUT_GENERAL)) ++more;
    EXPECT(live_images + more == capacity - 1 && set.free_images(dimension) == 0);
    EXPECT(set.image_refusal(dimension, false) == BindlessSet::ImageRefusal::array_full);
    // Freed with nothing pending in the table: the index is free again at once.
    set.remove_image(dimension, indices[7]);
    EXPECT(set.free_images(dimension) == 1);
    EXPECT(set.add_image(dimension, images.views[7], VK_IMAGE_LAYOUT_GENERAL) == indices[7]);
    // Freed after its key is erased (the GPU's table may still name it): held until a flush brings the table up to
    // date, then free.
    const DescriptorKey key{0x1234u, 0, 0, 8u << 28, 0, 0, 0, 0};
    set.insert_key(key, indices[9], dimension);
    set.erase_key(key);
    set.remove_image(dimension, indices[9]);
    EXPECT(set.free_images(dimension) == 0 && set.add_image(dimension, images.views[9], VK_IMAGE_LAYOUT_GENERAL) == 0);
    EXPECT(set.image_refusal(dimension, false) == BindlessSet::ImageRefusal::array_full);
    const bool reflushed = d.run([&](VkCommandBuffer cb) { return set.flush(cb, 1, [] { return std::uint64_t(1); }); });
    set.table_upload_result(reflushed);
    EXPECT(reflushed && set.free_images(dimension) == 1);
    EXPECT(set.add_image(dimension, images.views[9], VK_IMAGE_LAYOUT_GENERAL) == indices[9]);
    std::printf("%s: %u more filled the array; the next was refused as full, and removed indices came back\n",
                name(dimension), more);

    d.fn<PFN_vkDeviceWaitIdle>("vkDeviceWaitIdle")(d.device);
    set.release_retired();
    return 0;
}

}  // namespace

int main() {
#ifndef LEONIDA_TEST_BINDLESS_DIMENSIONS_SPV
    std::printf("unsupported: the test shader was not built (no glslangValidator)\n");
    return 77;
#else
    vktest::Device d;
    if (const auto status = d.initialize(); status != vktest::DeviceStatus::ready)
        return status == vktest::DeviceStatus::unavailable ? 77 : 1;
    if (!d.bindless) {
        std::printf("unsupported: the device lacks the bindless set's descriptor-indexing features\n");
        return 77;
    }
    VkPhysicalDeviceMemoryProperties memory{};
    d.instance_fn<PFN_vkGetPhysicalDeviceMemoryProperties>("vkGetPhysicalDeviceMemoryProperties")(d.physical, &memory);
    VkPhysicalDeviceVulkan12Properties v12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_PROPERTIES};
    VkPhysicalDeviceVulkan11Properties v11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_PROPERTIES, &v12};
    VkPhysicalDeviceProperties2 properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &v11};
    d.instance_fn<PFN_vkGetPhysicalDeviceProperties2>("vkGetPhysicalDeviceProperties2")(d.physical, &properties);
    std::printf("device: %s\n", properties.properties.deviceName);

    // Sized as the D3D12 device sizes it (d3d12_vulkan_device.cpp): from the update-after-bind limits, past
    // maxPerSetDescriptors where vkGetDescriptorSetLayoutSupport accepts the layout.
    BindlessFeatures features;
    features.update_after_bind = true;
    features.partially_bound = d.partially_bound;
    const auto layout_support =
        d.fn<PFN_vkGetDescriptorSetLayoutSupport>("vkGetDescriptorSetLayoutSupport");
    const auto layout_supported = [&](const BindlessFeatures& candidate) {
        if (!layout_support) return false;
        const bindless::SetLayout layout(candidate);
        VkDescriptorSetLayoutSupport support{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_SUPPORT};
        layout_support(d.device, &layout.info, &support);
        return support.supported == VK_TRUE;
    };
    bindless::size_bindings(features, properties.properties.limits, v11, v12, layout_supported);
    const std::uint32_t sampled_limit = std::min(v12.maxPerStageDescriptorUpdateAfterBindSampledImages,
                                                 v12.maxDescriptorSetUpdateAfterBindSampledImages);
    // Room for 5000 images in each of the four sampled arrays besides 2D's: below that, the device itself holds
    // fewer, and the test has nothing to say.
    if (sampled_limit < bindless::max_images + 3 * (live_images + 1) + 2 * bindless::growth_headroom) {
        std::printf("unsupported: the device holds %u sampled images in a set, too few for %u of each dimension\n",
                    sampled_limit, live_images);
        return 77;
    }

    const std::vector<std::uint32_t> code = load_spirv(LEONIDA_TEST_BINDLESS_DIMENSIONS_SPV);
    EXPECT(!code.empty());
    const std::uint32_t max_layers = properties.properties.limits.maxImageArrayLayers;
    for (const ImageDimension dimension :
         {ImageDimension::dim_1d, ImageDimension::dim_2d_array, ImageDimension::dim_3d})
        EXPECT(run(d, memory, features, max_layers, code, dimension) == 0);

    if (d.validation_errors()) {
        std::printf("%u validation errors\n", d.validation_errors());
        return 1;
    }
    std::printf("vk_bindless_dimensions: ok\n");
    return 0;
#endif
}
