#pragma once

// The bindless descriptor set of translated shaders: set 0 of the contract in docs/design.md and
// docs/d3d12_vulkan.md section 5, as tools/rdna2spirv.py declares it.
//
//   binding  0  sampled 2D images          4096        5  sampled 1D images          256
//            1  samplers                   1024        6  sampled 3D images          256
//            2  the descriptor hash table     1        7  3D r32ui storage images    256
//            3  sampled 2D-array images    1024        8 + 3 * dimension + class     256 each:
//            4  2D r32ui storage images    1024           storage images by component class
//   dimension: 0 = 2D, 1 = 2D array, 2 = 3D, 3 = 1D (the first four STORE_DIMS); class: 0 = float, 1 = uint, 2 = sint.
//   bindings 20/21: sampled multisample 2D images / 2D arrays, 256 each (runtime SRVs).
//   bindings 22/23 and 24/25 are their UINT and SINT equivalents on devices supporting those classes.
//   bindings 26-28: storage-only 1D-array images by component class, 256 each (STORE_DIMS index 4).
// These are the default counts (BindlessFeatures::counts). size_bindings() sizes the sampled arrays of every
// single-sample dimension, the samplers and the storage arrays from the device's limits instead; the shaders declare
// runtime arrays, so any count works for them.
//
// A shader turns a T# or S# into an index through the hash table (descriptor_table.h) and uses that index in
// whichever binding the instruction calls for: the sampled array of the image's dimension, or one of its storage
// arrays. So an image has one index, the same in every binding of its dimension, and each dimension has its own
// index space. The storage arrays are shorter (storage_capacity()), so an image with storage views gets an index
// they hold; one without takes a higher index while there is one. Index 0 of every array is a fallback, because a
// miss gives 0: a 1x1 image of zeros, a nearest clamping sampler, and 1x1 storage images. The caller owns the views
// and samplers it adds and keeps them alive while they are in the set; this class owns only the fallbacks and the
// table's buffer.
//
// The T#'s own type is the truth: the index was given out in ITS dimension's space, so a shader reading it through
// another dimension's array (a 2D instruction on a 2D-array or cube T#, say) must not use it blindly. Every entry
// therefore carries, next to the index, which arrays the index is bound in (descriptor_table.h's kind bits), and a
// translated module of the tagged contract reads that word: it uses the index only in an array the entry names and
// otherwise takes that array's fallback. For the one mismatch the hardware reads rather than drops - a 2D
// instruction on a 2D-array or cube T#, which addresses slice 0 - add_image's MirrorViews bind a view of that slice
// at the same index in the 2D arrays and name 2D in the kind, so the instruction finds it.
//
// The fallbacks stand for the console's null descriptors (a zero T#, a T# whose image the layer refused or never
// saw): each is of the kind the instruction asks for, since the translator picks the binding by the instruction's
// dimension and the view by the T#'s numeric class, and a miss at any of them reads zeros. On the console a store or
// atomic through a null descriptor is dropped and a later read still gives zeros. So the storage fallbacks never
// share an image with the sampled ones (a store through a null UAV would otherwise show in every null SRV of that
// dimension), and with BindlessFeatures::null_descriptor (VK_EXT_robustness2 nullDescriptor) they are null
// descriptors, whose stores Vulkan discards and whose reads are zero.
//
// Device features: runtimeDescriptorArray for the shaders. With BindlessFeatures::update_after_bind
// (descriptorBindingSampledImageUpdateAfterBind, ...StorageImage..., ...StorageBuffer...) flush() may run while
// command buffers that use the set are pending, as long as no descriptor they use changes and
// flush_replaces_table() is false; without it the caller flushes only when none is. When it is true the flush
// does not touch what pending work reads: it makes a new generation (table buffers and descriptor set, the
// descriptors copied from a CPU shadow) and retires the old one, which release_retired() frees once the owner
// says the submissions that could have used it are done (D3D-314). With partially_bound (descriptorBindingPartiallyBound) only index 0 is written at
// creation; otherwise every element of every array starts as the fallback.
//
// The table of binding 2 lives in DEVICE_LOCAL memory: every divergent T#/S# lookup reads it, and a HOST_VISIBLE
// table measured 19-46x slower there at 100k entries on an RTX 2060 SUPER (vk_descriptor_bench, D3D-56, #639).
// flush() takes the command buffer that will use the set: it uploads only the slots that changed, from a mapped
// staging buffer, and barriers the copy before the draws. The table at creation is empty until that first flush.
//
// When device-local memory runs out (a title that fills VRAM), a new generation's table goes to HOST_VISIBLE memory
// instead: slower lookups beat no table. Released generations are kept as spares (at most max_spares) and reused,
// pool and buffers, by the next replacing flush, so steady churn allocates nothing. If a generation still cannot be
// made, flush() rewrites the current table in place when every submission that could read it is done, and otherwise
// degrades: the descriptors are written, the table keeps its last uploaded contents (keys added since find the
// fallback, index 0; keys rebound since keep their previous index, whose views stay alive; removed indices are not
// given out again meanwhile) and the change waits for a later flush. A flush never fails the submission for want of
// memory (last_flush_degraded(), degraded_flushes()).
//
// Not thread-safe: the owner serialises calls.

#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
#include "descriptor_table.h"
#include "queue_families.h"

#include <vulkan/vulkan.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <functional>
#include <unordered_map>
#include <utility>
#include <vector>

namespace leonida {

// The first five dimensions are the translator's STORE_DIMS indices. A cube is a 2D array of six layers.
enum class ImageDimension : std::uint32_t { dim_2d, dim_2d_array, dim_3d, dim_1d, dim_1d_array, dim_2d_ms, dim_2d_ms_array };

// Storage views of one image by the component type a shader reads or writes it with (the translator picks by the
// T# format at run time). r32ui is for the atomic instructions, whose image type is R32ui (2D and 3D only).
// A null view leaves the fallback in that binding.
struct StorageViews {
    VkImageView as_float = VK_NULL_HANDLE;
    VkImageView as_uint = VK_NULL_HANDLE;
    VkImageView as_sint = VK_NULL_HANDLE;
    VkImageView r32ui = VK_NULL_HANDLE;
};

namespace bindless {
inline constexpr std::uint32_t binding_count = 29;
constexpr std::array<std::uint32_t, binding_count> default_counts();
} // namespace bindless

struct BindlessFeatures {
    bool update_after_bind = false;
    bool partially_bound = false;
    // VK_EXT_robustness2's nullDescriptor is enabled on the device: the storage arrays' fallbacks are null
    // descriptors (stores and atomics through them are discarded, loads read zero). Without it they are 1x1 images
    // of their own, apart from the sampled fallbacks, which no store reaches.
    bool null_descriptor = false;
    // Fallback sample counts for RGBA8_UNORM, R32_UINT, R32_SINT. A single sample omits that class's MS bindings.
    std::array<VkSampleCountFlagBits, 3> ms_samples{
        VK_SAMPLE_COUNT_1_BIT, VK_SAMPLE_COUNT_1_BIT, VK_SAMPLE_COUNT_1_BIT};
    // Descriptors in each binding, by binding number: bindless::bindings by default, or what size_bindings() chose.
    std::array<std::uint32_t, bindless::binding_count> counts = bindless::default_counts();
};

namespace bindless {

struct Binding {
    VkDescriptorType type;
    std::uint32_t count;
};

inline constexpr std::uint32_t table_binding = 2;
inline constexpr std::uint32_t sampler_binding = 1;
inline constexpr std::uint32_t first_ms_binding = 20;
inline constexpr std::uint32_t ms_binding_count = 6;
inline constexpr std::uint32_t first_1d_array_storage_binding = first_ms_binding + ms_binding_count;
inline constexpr std::uint32_t no_binding = 0xffffffffu;

// Indexed by binding number. Keep the literal extent for tools/spirv_metal_set0.py.
inline constexpr std::array<Binding, 29> bindings = {{
    {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 4096}, {VK_DESCRIPTOR_TYPE_SAMPLER, 1024},
    {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1},   {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1024},
    {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1024}, {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 256},
    {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 256},  {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 256},
    {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 256},  {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 256},
    {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 256},  {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 256},
    {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 256},  {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 256},
    {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 256},  {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 256},
    {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 256},  {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 256},
    {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 256},  {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 256},
    {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 256},  {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 256},
    {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 256},  {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 256},
    {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 256},  {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 256},
    {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 256},  {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 256},
    {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 256},
}};

static_assert(bindings.size() == binding_count);

constexpr std::array<std::uint32_t, binding_count> default_counts() {
    std::array<std::uint32_t, binding_count> counts{};
    for (std::uint32_t b = 0; b < binding_count; ++b) counts[b] = bindings[b].count;
    return counts;
}

constexpr bool ms_binding(std::uint32_t binding) {
    return binding >= first_ms_binding && binding < first_ms_binding + ms_binding_count;
}

constexpr bool binding_enabled(std::uint32_t binding, const BindlessFeatures& features) {
    return !ms_binding(binding) || features.ms_samples[(binding - first_ms_binding) / 2] != VK_SAMPLE_COUNT_1_BIT;
}

// Every binding is visible to every shader stage. Preserve the single-sample layout on devices whose
// per-stage or per-set limits have no room for additional pairs of MS arrays.
constexpr void limit_ms_bindings(BindlessFeatures& features, std::uint32_t sampled_limit,
                                 std::uint32_t resource_limit) {
    const auto& counts = features.counts;
    std::uint32_t sampled = 0, total = 0;
    for (std::uint32_t b = 0; b < binding_count; ++b) {
        if (ms_binding(b)) continue;
        if (bindings[b].type == VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE) sampled += counts[b];
        if (bindings[b].type != VK_DESCRIPTOR_TYPE_SAMPLER) total += counts[b];
    }
    for (std::uint32_t c = 0; c < features.ms_samples.size(); ++c) {
        if (features.ms_samples[c] == VK_SAMPLE_COUNT_1_BIT) continue;
        const auto b = first_ms_binding + 2 * c;
        const auto pair = counts[b] + counts[b + 1];
        if (sampled + pair > sampled_limit || total + pair > resource_limit) {
            features.ms_samples[c] = VK_SAMPLE_COUNT_1_BIT;
        } else {
            sampled += pair;
            total += pair;
        }
    }
}

constexpr bool multisampled(ImageDimension dimension) {
    return dimension == ImageDimension::dim_2d_ms || dimension == ImageDimension::dim_2d_ms_array;
}

constexpr std::uint32_t sampled_binding(ImageDimension dimension, std::uint32_t component_class = 0) {
    constexpr std::uint32_t by_dimension[] = {0, 3, 6, 5, no_binding, first_ms_binding, first_ms_binding + 1};
    return by_dimension[std::uint32_t(dimension)] + (multisampled(dimension) ? component_class * 2 : 0);
}

// component_class: 0 = float, 1 = uint, 2 = sint.
constexpr std::uint32_t storage_binding(ImageDimension dimension, std::uint32_t component_class) {
    if (dimension == ImageDimension::dim_1d_array) return first_1d_array_storage_binding + component_class;
    return multisampled(dimension) ? no_binding : 8 + 3 * std::uint32_t(dimension) + component_class;
}

constexpr std::uint32_t r32ui_binding(ImageDimension dimension) {
    return dimension == ImageDimension::dim_2d ? 4 : dimension == ImageDimension::dim_3d ? 7 : no_binding;
}

// Set 0's descriptor set layout at features' counts: what create_layout creates and what size_bindings'
// layout_supported asks vkGetDescriptorSetLayoutSupport about. It points into itself, so it is not copied.
struct SetLayout {
    std::array<VkDescriptorSetLayoutBinding, binding_count> entries{};
    std::array<VkDescriptorBindingFlags, binding_count> flags{};
    VkDescriptorSetLayoutBindingFlagsCreateInfo flags_info{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO};
    VkDescriptorSetLayoutCreateInfo info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};

    explicit SetLayout(const BindlessFeatures& features) {
        std::uint32_t count = 0;
        for (std::uint32_t b = 0; b < binding_count; ++b) {
            if (!binding_enabled(b, features)) continue;
            entries[count] = {b, bindings[b].type, features.counts[b], VK_SHADER_STAGE_ALL, nullptr};
            flags[count] = (features.update_after_bind ? VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT : 0) |
                           (features.partially_bound ? VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT : 0);
            ++count;
        }
        flags_info.bindingCount = count;
        flags_info.pBindingFlags = flags.data();
        info.pNext = &flags_info;
        info.flags = features.update_after_bind ? VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT : 0;
        info.bindingCount = count;
        info.pBindings = entries.data();
    }
    SetLayout(const SetLayout&) = delete;
    SetLayout& operator=(const SetLayout&) = delete;
};

// What size_bindings() grows the sampled array of every single-sample dimension (2D, 2D array, 3D and 1D) to where
// the device's limits allow: one size for all of them. Which dimensions its textures have is the title's choice, and
// an image keeps its index (one for each distinct T# of it) for as long as the texture lives, so no dimension gets
// less room than another: a world run kept 4090 live 1D textures, the whole of the 4096 1D indices the set once
// stopped at, while the 2D arrays had room to spare. The bound is the set's memory, not the device's descriptor
// limits (8388606 sampled images on RADV): every element of every array is in each generation's descriptor pool,
// used or not. At 32-64 bytes a descriptor the whole set is 10-20 MiB with 325729 descriptors when all MS classes
// and arrays reach their limits (an RTX 5070).
inline constexpr std::uint32_t max_images = 65536;
inline constexpr std::uint32_t max_storage_indices = 4096;  // in each storage array, at most its dimension's images
// A full D3D12 shader-visible sampler heap (2048) and room for literal and static S#s, within what the device can
// create (maxSamplerAllocationCount, at least 4000) less the samplers that are not in the set.
inline constexpr std::uint32_t max_samplers = 4096;
inline constexpr std::uint32_t reserved_samplers = 64;
// Growth leaves this many descriptors of each limit to the other sets of a pipeline layout.
inline constexpr std::uint32_t growth_headroom = 64;

// Sizes the 2D image, sampler and storage arrays from the device's limits: the update-after-bind ones with
// features.update_after_bind, the plain ones otherwise. Every binding is visible to every stage, so the per-stage
// limits apply to the whole set; fragment outputs count as stage resources (D3D12 allows eight render targets) and
// standalone samplers do not, though they count in an update-after-bind pool.
//   1. The default counts, or as much of them as the device holds: samplers, then storage, then 2D images.
//   2. The MS pairs that fit beside them (limit_ms_bindings, which may turn classes off).
//   3. Growth towards the maximums above, in the same order, then the 2D-array, 3D and 1D images together (one
//      count for the three, so a device whose limits stop growth short of max_images favours none of them), within
//      maxPerSetDescriptors. That limit only says what is sure to work (vkGetDescriptorSetLayoutSupport may allow
//      more), so it never shrinks step 1.
// A storage array holds the smaller of the storage count and its dimension's sampled count: add_image never
// gives a storage view an index past the sampled array. The storage-only 1D array is capped at its default of 256.
// The other bindings keep their default counts.
// layout_supported, when given, answers vkGetDescriptorSetLayoutSupport for set 0 at the counts in its argument:
// step 3 then also grows past maxPerSetDescriptors where the device says the layout works (Intel's Xe-LP reports
// 6159 and supports the full sizes, D3D-219). It is asked only once the sum is above maxPerSetDescriptors.
// Sets features.counts; returns false when a count is below its default because the device's limits are.
inline bool size_bindings(BindlessFeatures& features, const VkPhysicalDeviceLimits& limits,
                          const VkPhysicalDeviceVulkan11Properties& v11,
                          const VkPhysicalDeviceVulkan12Properties& v12,
                          const std::function<bool(const BindlessFeatures&)>& layout_supported = {}) {
    const bool uab = features.update_after_bind;
    const std::uint32_t sampled_limit =
        uab ? std::min(v12.maxPerStageDescriptorUpdateAfterBindSampledImages,
                       v12.maxDescriptorSetUpdateAfterBindSampledImages)
            : std::min(limits.maxPerStageDescriptorSampledImages, limits.maxDescriptorSetSampledImages);
    const std::uint32_t sampler_limit =
        uab ? std::min(v12.maxPerStageDescriptorUpdateAfterBindSamplers, v12.maxDescriptorSetUpdateAfterBindSamplers)
            : std::min(limits.maxPerStageDescriptorSamplers, limits.maxDescriptorSetSamplers);
    const std::uint32_t storage_limit =
        uab ? std::min(v12.maxPerStageDescriptorUpdateAfterBindStorageImages,
                       v12.maxDescriptorSetUpdateAfterBindStorageImages)
            : std::min(limits.maxPerStageDescriptorStorageImages, limits.maxDescriptorSetStorageImages);
    std::uint32_t resource_limit = uab ? v12.maxPerStageUpdateAfterBindResources : limits.maxPerStageResources;
    resource_limit -= std::min(resource_limit, std::min(limits.maxColorAttachments, 8u));
    const std::uint32_t pool_limit = uab ? v12.maxUpdateAfterBindDescriptorsInAllPools : 0xffffffffu;

    auto& counts = features.counts;
    counts = default_counts();
    std::uint32_t images = 1, samplers = 1, storage = 1;
    std::uint32_t arrays = counts[sampled_binding(ImageDimension::dim_2d_array)];
    std::uint32_t volumes = counts[sampled_binding(ImageDimension::dim_3d)];
    std::uint32_t lines = counts[sampled_binding(ImageDimension::dim_1d)];
    // Until the end the 2D storage arrays are not capped by the 2D images, which grow after them: capping can only
    // take descriptors away.
    const auto apply = [&](bool final) {
        counts[sampled_binding(ImageDimension::dim_2d)] = images;
        counts[sampled_binding(ImageDimension::dim_2d_array)] = arrays;
        counts[sampled_binding(ImageDimension::dim_3d)] = volumes;
        counts[sampled_binding(ImageDimension::dim_1d)] = lines;
        counts[sampler_binding] = samplers;
        for (std::uint32_t d = 0; d <= std::uint32_t(ImageDimension::dim_1d_array); ++d) {
            const auto dimension = ImageDimension(d);
            const auto sampled_at = sampled_binding(dimension);
            const auto capacity = sampled_at == no_binding ? bindings[storage_binding(dimension, 0)].count
                                                            : counts[sampled_at];
            const bool capped = final || dimension != ImageDimension::dim_2d;
            const std::uint32_t held = capped ? std::min(storage, capacity) : storage;
            for (std::uint32_t c = 0; c < 3; ++c) counts[storage_binding(dimension, c)] = held;
            if (r32ui_binding(dimension) != no_binding) counts[r32ui_binding(dimension)] = held;
        }
    };
    const auto fits = [&](std::uint32_t headroom, std::uint64_t set_limit) {
        apply(false);
        std::uint64_t sampled = 0, sampler = 0, stored = 0, buffers = 0;
        for (std::uint32_t b = 0; b < binding_count; ++b) {
            if (!binding_enabled(b, features)) continue;
            const auto type = bindings[b].type;
            (type == VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE   ? sampled
             : type == VK_DESCRIPTOR_TYPE_SAMPLER       ? sampler
             : type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE ? stored
                                                        : buffers) += counts[b];
        }
        const auto within = [&](std::uint64_t n, std::uint32_t limit) { return n + headroom <= limit; };
        const std::uint64_t all = sampled + sampler + stored + buffers;
        return within(sampled, sampled_limit) && within(sampler, sampler_limit) && within(stored, storage_limit) &&
               within(sampled + stored + buffers, resource_limit) && all <= pool_limit &&
               (all <= set_limit || (layout_supported && layout_supported(features)));
    };
    // Raises n to the largest value up to `high` that fits; leaves it where it is when none does.
    const auto grow = [&](std::uint32_t& n, std::uint32_t high, std::uint32_t headroom, std::uint64_t set_limit) {
        std::uint32_t low = n;
        while (low < high) {
            n = low + (high - low + 1) / 2;
            if (fits(headroom, set_limit)) low = n;
            else high = n - 1;
        }
        n = low;
    };

    constexpr std::uint64_t no_limit = ~0ull;
    const auto ms_samples = features.ms_samples;
    features.ms_samples.fill(VK_SAMPLE_COUNT_1_BIT);
    grow(samplers, bindings[sampler_binding].count, 0, no_limit);
    grow(storage, bindings[storage_binding(ImageDimension::dim_2d, 0)].count, 0, no_limit);
    grow(images, bindings[sampled_binding(ImageDimension::dim_2d)].count, 0, no_limit);
    apply(false);
    const bool full = images == bindings[0].count && samplers == bindings[sampler_binding].count &&
                      storage == bindings[storage_binding(ImageDimension::dim_2d, 0)].count;
    features.ms_samples = ms_samples;
    limit_ms_bindings(features, sampled_limit, std::min(resource_limit, pool_limit - std::min(pool_limit, samplers)));

    std::uint64_t set_limit = 0;
    for (std::uint32_t b = 0; b < binding_count; ++b)
        if (binding_enabled(b, features)) set_limit += counts[b];
    set_limit = std::max(set_limit, std::uint64_t(v11.maxPerSetDescriptors));
    const std::uint32_t creatable =
        limits.maxSamplerAllocationCount - std::min(limits.maxSamplerAllocationCount, reserved_samplers);
    grow(samplers, std::max(samplers, std::min(max_samplers, creatable)), growth_headroom, set_limit);
    grow(images, std::max(images, max_images), growth_headroom, set_limit);
    grow(storage, std::max(storage, max_storage_indices), growth_headroom, set_limit);
    // The other three dimensions grow as one: each gets the larger of its count so far and the shared one.
    const std::uint32_t arrays_before = arrays, volumes_before = volumes, lines_before = lines;
    std::uint32_t together = 0;
    const auto set_together = [&](std::uint32_t n) {
        arrays = std::max(arrays_before, n);
        volumes = std::max(volumes_before, n);
        lines = std::max(lines_before, n);
    };
    for (std::uint32_t high = max_images; together < high;) {
        const std::uint32_t n = together + (high - together + 1) / 2;
        set_together(n);
        if (fits(growth_headroom, set_limit)) together = n;
        else high = n - 1;
    }
    set_together(together);
    apply(true);
    return full;
}

} // namespace bindless

class BindlessSet {
public:
    // `descriptor_hash` is the finalizer the shaders of this device probe the table with (descriptor_table.h); it is
    // handed to the table before its first key is inserted, so every insert, lookup and rehash agrees with them.
    BindlessSet(PFN_vkGetDeviceProcAddr get, VkDevice device, const VkPhysicalDeviceMemoryProperties& memory,
                BindlessFeatures features = {}, DescriptorHashKind descriptor_hash = DescriptorHashKind::fnv1a,
                std::vector<std::uint32_t> families = {})
        : device_(device), memory_(memory), features_(features), families_(std::move(families)),
          table_(64, descriptor_hash) {
        ok_ = vk_.load(get, device) && create_layout() && create_fallbacks() && create_table_buffers() && fill();
    }

    ~BindlessSet() {
        release_retired();
        for (Generation& spare : spares_) destroy(spare);
        spares_.clear();
        destroy(table_buffers_);
        if (fallback_sampler_) vk_.vkDestroySampler(device_, fallback_sampler_, nullptr);
        for (const VkImageView view : fallback_views_) vk_.vkDestroyImageView(device_, view, nullptr);
        const auto destroy_image = [&](const FallbackImage& image) {
            if (image.image) vk_.vkDestroyImage(device_, image.image, nullptr);
            if (image.memory) vk_.vkFreeMemory(device_, image.memory, nullptr);
        };
        for (const auto& image : fallback_images_) destroy_image(image);
        for (const auto& image : storage_fallback_images_) destroy_image(image);
        if (pool_) vk_.vkDestroyDescriptorPool(device_, pool_, nullptr);  // frees the set
        if (layout_) vk_.vkDestroyDescriptorSetLayout(device_, layout_, nullptr);
    }

    BindlessSet(const BindlessSet&) = delete;
    BindlessSet& operator=(const BindlessSet&) = delete;

    bool ok() const { return ok_; }
    // What the set layout was made from (the pipeline journal names compute pipelines by it).
    const BindlessFeatures& features() const { return features_; }
    bool supports_ms_class(std::uint32_t component_class) const {
        return component_class < classes && features_.ms_samples[component_class] != VK_SAMPLE_COUNT_1_BIT;
    }
    // How many images of a dimension the set holds, the fallback included, and how many of the lowest of those
    // indices its storage arrays hold (0 for the MS dimensions, which have none). The same for samplers.
    std::uint32_t image_capacity(ImageDimension dimension) const {
        const auto binding = bindless::sampled_binding(dimension);
        return binding == bindless::no_binding ? storage_capacity(dimension) : features_.counts[binding];
    }
    std::uint32_t storage_capacity(ImageDimension dimension) const {
        return bindless::multisampled(dimension) ? 0 : features_.counts[bindless::storage_binding(dimension, 0)];
    }
    std::uint32_t sampler_capacity() const { return features_.counts[bindless::sampler_binding]; }
    // How many more images of a dimension, or samplers, add_image and add_sampler can take before they return 0.
    std::uint32_t free_images(ImageDimension dimension) const {
        const auto& used = image_used_[std::uint32_t(dimension)];
        return std::uint32_t(std::count(used.begin(), used.end(), std::uint8_t(0)));
    }
    std::uint32_t free_samplers() const {
        return std::uint32_t(std::count(sampler_used_.begin(), sampler_used_.end(), std::uint8_t(0)));
    }
    // Set 0 of every pipeline layout made for translated shaders.
    VkDescriptorSetLayout layout() const { return layout_; }
    // Bound at set 0. flush() updates it in place until flush_replaces_table() is true: then the flush makes a new
    // set (and table buffers), so read set() again after it, when recording the commands that use it. A submission
    // that bound the previous set keeps it, intact, until release_retired() frees it.
    VkDescriptorSet set() const { return set_; }

    // The fallback images are created with undefined contents: record this once, in a command buffer that executes
    // before anything uses the set. It moves them to GENERAL, the layout their descriptors name, and zeroes them.
    bool fallbacks_ready() const { return fallbacks_ready_; }
    void record_fallback_init(VkCommandBuffer command_buffer) {
        fallbacks_ready_ = true;
        std::vector<VkImageMemoryBarrier> barriers;
        std::vector<const FallbackImage*> images;
        for (const auto& image : fallback_images_) images.push_back(&image);
        for (const auto& image : storage_fallback_images_) images.push_back(&image);
        for (const FallbackImage* fallback : images) {
            const FallbackImage& image = *fallback;
            if (!image.image) continue;  // an unused class, or a storage fallback that is a null descriptor
            VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
            barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.image = image.image;
            barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            barriers.push_back(barrier);
        }
        vk_.vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0,
                                 nullptr, 0, nullptr, std::uint32_t(barriers.size()), barriers.data());
        const VkClearColorValue zero{};
        for (const auto& barrier : barriers)
            vk_.vkCmdClearColorImage(command_buffer, barrier.image, VK_IMAGE_LAYOUT_GENERAL, &zero, 1,
                                     &barrier.subresourceRange);
        VkMemoryBarrier done{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        done.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        done.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
        vk_.vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1,
                                 &done, 0, nullptr, 0, nullptr);
    }

    // A second view of the same image in the 2D arrays: the slice a 2D instruction reads through a 2D-array or cube
    // T# (its first layer). The hardware addresses such a read as slice 0, and the translated shader reads binding 0
    // with the identity the table gives, so the runtime binds this copy at the image's own index and records that
    // index as valid in the 2D array too (kind_sampled_bit), owned by this image (mirror_of): another dimension's
    // image with the same slot number does not inherit it. The storage views follow the same rule when the image has
    // them, so a 2D store or atomic lands on layer 0.
    struct MirrorViews {
        VkImageView sampled = VK_NULL_HANDLE;
        VkImageLayout sampled_layout = VK_IMAGE_LAYOUT_GENERAL;
        StorageViews storage{};
    };

    // Gives an image an index in its dimension's bindings: 0 when the dimension is full (image_refusal says why). `sampled` goes to the
    // sampled array in `sampled_layout` (the layout the image is in when shaders read it); the storage views go to
    // the storage arrays in GENERAL. For MS views, sampled_class selects the float, uint or sint sampled array.
    // `mirror` adds the same index in the 2D arrays (a 2D-array or cube image's slice 0): the slot must then be free
    // in both index spaces. `kind_out`, when given, receives what the index is bound in (descriptor_table.h's kind
    // bits), which the caller records with insert_key. The caller maps the T# to the index with insert_key(). Takes
    // effect at flush().
    std::uint32_t add_image(ImageDimension dimension, VkImageView sampled, VkImageLayout sampled_layout,
                            const StorageViews& storage = {}, std::uint32_t sampled_class = 0,
                            const MirrorViews* mirror = nullptr, std::uint32_t* kind_out = nullptr) {
        const bool stores = storage.as_float || storage.as_uint || storage.as_sint || storage.r32ui;
        if (!stores && bindless::sampled_binding(dimension) == bindless::no_binding) return 0;
        if (bindless::multisampled(dimension) && (stores || !supports_ms_class(sampled_class))) return 0;
        auto& used = image_used_[std::uint32_t(dimension)];
        const std::uint32_t low_end = std::min(std::uint32_t(used.size()), storage_capacity(dimension));
        std::vector<std::uint8_t>* used_2d = nullptr;   // the mirror's index space, when there is one
        if (mirror && dimension != ImageDimension::dim_2d && mirror->sampled) {
            used_2d = &image_used_[std::uint32_t(ImageDimension::dim_2d)];
            if (mirror->sampled_layout == VK_IMAGE_LAYOUT_UNDEFINED) used_2d = nullptr;
        }
        // A store through the array T# and one through its 2D slice both have to land on a view, so an index with
        // storage views must be inside this dimension's storage arrays and, when the mirror has storage views too,
        // inside the 2D arrays' (their range can be the shorter of the two).
        const bool any_stores = stores || mirror_storage(mirror);
        const std::uint32_t mirrored_storage_end = std::min(stores ? low_end : std::uint32_t(used.size()),
                                                            storage_capacity(ImageDimension::dim_2d));
        std::uint32_t index = 0;
        if (used_2d) {
            index = any_stores ? allocate_pair(used, *used_2d, 1, mirrored_storage_end)
                               : allocate_pair(used, *used_2d, low_end, std::uint32_t(used.size()));
            if (!index && !any_stores) index = allocate_pair(used, *used_2d, 1, low_end);
            if (!index) {
                // No shared slot: the image still gets its own dimension's slot, unmirrored (a 2D read then finds
                // the fallback). A full array is what returns 0.
                used_2d = nullptr;
            }
        }
        if (!used_2d) {
            index = stores ? allocate(used, 1, low_end) : allocate(used, low_end, std::uint32_t(used.size()));
            if (!index) index = allocate(used, 1, low_end);
        }
        if (!index) return 0;
        assert(!used_2d || !any_stores || index < storage_capacity(ImageDimension::dim_2d));
        write_image(dimension, index, sampled, sampled_layout, storage, sampled_class);
        if (index < storage_used_[std::uint32_t(dimension)].size())
            storage_used_[std::uint32_t(dimension)][index] = stores ? 1 : 0;
        if (used_2d) {
            write_image(ImageDimension::dim_2d, index, mirror->sampled, mirror->sampled_layout, mirror->storage, 0);
            used_2d->at(index) = in_use;
            // The 2D slot is this image's mirror, and this image is its owner: the slot number alone does not say
            // so, because every dimension's index space is dense and independent.
            if (index < mirror_owner_.size()) {
                mirror_owner_[index] = std::uint8_t(std::uint32_t(dimension) + 1);
                mirror_storage_used_[index] = mirror_storage(mirror) ? 1 : 0;
            }
            auto& storage_2d = storage_used_[std::uint32_t(ImageDimension::dim_2d)];
            if (index < storage_2d.size()) storage_2d[index] = mirror_storage(mirror) ? 1 : 0;
        }
        if (kind_out) {
            *kind_out = kind_sampled_bit(std::uint32_t(dimension)) |
                        (stores ? kind_storage_bit(std::uint32_t(dimension)) : 0) |
                        (used_2d ? (kind_sampled_bit(std::uint32_t(ImageDimension::dim_2d)) |
                                    (mirror_storage(mirror) ? kind_storage_bit(std::uint32_t(ImageDimension::dim_2d))
                                                            : 0))
                                 : 0);
        }
        return index;
    }

    // Why add_image returned 0 for an image of `dimension` (with storage views when `stores`; sampled_class for the
    // MS dimensions): the set has no array for it; every index of its dimension is taken (array_full: in use, or
    // removed and held until the next flush that brings the table up to date); or only the storage arrays' shorter
    // range is, which an image with storage views needs an index in (storage_full).
    enum class ImageRefusal { no_array, array_full, storage_full };
    ImageRefusal image_refusal(ImageDimension dimension, bool stores, std::uint32_t sampled_class = 0) const {
        if (!stores && bindless::sampled_binding(dimension) == bindless::no_binding) return ImageRefusal::no_array;
        if (bindless::multisampled(dimension) && (stores || !supports_ms_class(sampled_class)))
            return ImageRefusal::no_array;
        const auto& used = image_used_[std::uint32_t(dimension)];
        const std::size_t low_end = std::min<std::size_t>(used.size(), storage_capacity(dimension));
        const auto storage_end = used.begin() + std::ptrdiff_t(low_end);
        const bool storage_free = low_end > 1 && std::find(used.begin() + 1, storage_end, std::uint8_t(0)) != storage_end;
        return stores && !storage_free && free_images(dimension) ? ImageRefusal::storage_full
                                                                 : ImageRefusal::array_full;
    }

    // New views or a new layout for an index add_image returned (a texture's state changed, say). `mirror` is
    // refreshed too when the image was added with one (mirror_of says whose 2D slot it is). The slot bookkeeping
    // follows what was written, and `kind_out`, when given, receives the index's kind word afterwards for a
    // re-insert: the table keeps the kind it was given, so a caller that adds or drops storage views re-records it.
    void update_image(ImageDimension dimension, std::uint32_t index, VkImageView sampled, VkImageLayout sampled_layout,
                      const StorageViews& storage = {}, std::uint32_t sampled_class = 0,
                      const MirrorViews* mirror = nullptr, std::uint32_t* kind_out = nullptr) {
        const auto& used = image_used_[std::uint32_t(dimension)];
        const bool live = index && index < used.size() && used[index] == in_use;
        if (live) {
            write_image(dimension, index, sampled, sampled_layout, storage, sampled_class);
            const bool stores = storage.as_float || storage.as_uint || storage.as_sint || storage.r32ui;
            if (index < storage_used_[std::uint32_t(dimension)].size())
                storage_used_[std::uint32_t(dimension)][index] = stores ? 1 : 0;
        }
        if (live && mirror && mirror->sampled && mirror_of(dimension, index)) {
            write_image(ImageDimension::dim_2d, index, mirror->sampled, mirror->sampled_layout, mirror->storage, 0);
            if (index < mirror_storage_used_.size()) mirror_storage_used_[index] = mirror_storage(mirror) ? 1 : 0;
            auto& storage_2d = storage_used_[std::uint32_t(ImageDimension::dim_2d)];
            if (index < storage_2d.size()) storage_2d[index] = mirror_storage(mirror) ? 1 : 0;
        }
        if (kind_out) *kind_out = image_kind_at(dimension, index);
    }

    // Returns the index to the fallback and frees it. The caller removes the image's keys with erase_key() first.
    // While the table has changes the GPU's copy lacks (an erased key may still name the index there), the index is
    // held until a flush has brought the table up to date (one that did not degrade), so that no new image answers
    // that key; otherwise it is free at once.
    void remove_image(ImageDimension dimension, std::uint32_t index) {
        auto& used = image_used_[std::uint32_t(dimension)];
        if (!index || index >= used.size() || used[index] != in_use) return;
        write_image(dimension, index, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_GENERAL, {}, 0);
        used[index] = table_dirty_ ? held : 0;
        if (index < storage_used_[std::uint32_t(dimension)].size())
            storage_used_[std::uint32_t(dimension)][index] = 0;
        if (table_dirty_) held_images_.push_back({dimension, index});
        // Its 2D copy goes back to the fallback as well (add_image's mirror), when this image is the one that owns
        // that slot: another dimension's image at the same slot number has its own index space and keeps its 2D
        // descriptor (and, if it is a 2D-array image, its own mirror there).
        if (mirror_of(dimension, index)) {
            auto& used_2d = image_used_[std::uint32_t(ImageDimension::dim_2d)];
            write_image(ImageDimension::dim_2d, index, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_GENERAL, {}, 0);
            used_2d[index] = table_dirty_ ? held : 0;
            mirror_owner_[index] = 0;
            mirror_storage_used_[index] = 0;
            auto& storage_2d = storage_used_[std::uint32_t(ImageDimension::dim_2d)];
            if (index < storage_2d.size()) storage_2d[index] = 0;
            if (table_dirty_) held_images_.push_back({ImageDimension::dim_2d, index});
        }
    }

    // 0 when the sampler array is full.
    std::uint32_t add_sampler(VkSampler sampler) {
        const std::uint32_t index = allocate(sampler_used_, 1, std::uint32_t(sampler_used_.size()));
        if (index) write_sampler(index, sampler);
        return index;
    }

    void remove_sampler(std::uint32_t index) {
        if (!index || index >= sampler_used_.size() || sampler_used_[index] != in_use) return;
        write_sampler(index, fallback_sampler_);
        sampler_used_[index] = table_dirty_ ? held : 0;
        if (table_dirty_) held_samplers_.push_back(index);
    }

    // What an index of `dimension` is bound in, from what add_image wrote: that dimension's sampled bit, its storage
    // bit when the slot holds storage views, and the 2D bits when this image owns the slot at the same number in the
    // 2D arrays as its mirror (mirror_of).
    std::uint32_t image_kind_at(ImageDimension dimension, std::uint32_t index) const {
        const auto set = [&](const std::vector<std::uint8_t>& used) {
            return index < used.size() && used[index] != 0;
        };
        const std::uint32_t d = std::uint32_t(dimension);
        std::uint32_t kind = kind_sampled_bit(d) | (set(storage_used_[d]) ? kind_storage_bit(d) : 0);
        if (mirror_of(dimension, index)) {
            kind |= kind_sampled_bit(std::uint32_t(ImageDimension::dim_2d));
            if (set(mirror_storage_used_)) kind |= kind_storage_bit(std::uint32_t(ImageDimension::dim_2d));
        }
        return kind;
    }

    // T# and S# keys to the indices above: the table of binding 2 (descriptor_table.h). `kind` is what the index is
    // bound in, for the modules that read it (descriptor_table.h's kind bits); a sampler's entry has none. Changes
    // reach the shaders at flush().
    bool insert_key(const DescriptorKey& key, std::uint32_t index, std::uint32_t kind = 0) {
        const auto result = table_.insert_or_update(key.data(), index, kind);
        table_dirty_ |= result != DescriptorTable::InsertResult::unchanged;
        existing_entries_changed_ |= result == DescriptorTable::InsertResult::replaced ||
                                     result == DescriptorTable::InsertResult::added_with_relocation;
        return true;
    }
    // The T#'s own dimension, whose kind the set derives from the index it gave out (image_kind_at): a caller that
    // knows the T#'s TYPE needs to pass no bits.
    bool insert_key(const DescriptorKey& key, std::uint32_t index, ImageDimension dimension) {
        return insert_key(key, index, image_kind_at(dimension, index));
    }
    // Erasing empties only the key's own slot; no other entry moves (DescriptorTable::erase), so it needs no idle.
    // The caller erases a key once no submitted work uses it (its texture is gone), like D3D12 requires.
    bool erase_key(const DescriptorKey& key) {
        if (!table_.erase(key)) return false;
        table_dirty_ = true;
        return true;
    }
    const DescriptorTable& table() const { return table_; }

    // True when the next flush() would change entries a running shader may be probing (a live key's index replaced
    // or a live key moved by a relocating insert), or replace the table's buffer. flush() then does not write in
    // place: it uploads the whole table to new buffers, names them in a new descriptor set, and retires the old
    // generation, so running work keeps reading an unchanged table and nobody waits for it. Without it the flush only
    // adds entries or empties the slots of erased keys, and the copy of an entry is not atomic: a shader probing a
    // different key could in principle see a torn entry that matches it, which is accepted as vanishingly unlikely
    // because the added or erased key is not live (no running work uses it).
    bool flush_replaces_table() const {
        return table_dirty_ && (existing_entries_changed_ || force_full_upload_ ||
                                table_.size_bytes() > table_buffers_.table.size);
    }

    // True when the table's buffer was allocated in DEVICE_LOCAL memory (the fallback for a device without it is
    // HOST_VISIBLE, which still works, just slower).
    bool table_in_device_local_memory() const { return table_device_local_; }

    // True when the command buffer that carried the last table upload reached the queue; false keeps the upload
    // obligation, and the next flush() rebuilds the whole table (the changed-slot list was consumed when the copy
    // was recorded). A failed vkQueueSubmit is the caller's signal, not ours.
    void table_upload_result(bool submitted) {
        if (!upload_in_flight_) return;
        upload_in_flight_ = false;
        if (submitted) {
            table_on_gpu_ = true;  // the current buffers hold a table the GPU was given
            return;
        }
        table_dirty_ = true;
        force_full_upload_ = true;
    }

    // Writes the pending descriptor changes and uploads the table, in `command_buffer`, before the commands that use
    // the set are recorded after it. When flush_replaces_table() was true this starts a new generation (table
    // buffers, descriptor set, binding 2 and every descriptor written so far) and puts the previous one on the retire
    // list, stamped `retire_ticket`: the owner's number for "every submission made so far", which release_retired()
    // compares with what has completed.
    // When the new generation cannot be made (no memory for its buffers or descriptor pool) and `completed_ticket` is
    // given, it is asked what has completed: the generations retired up to there are released (to the spares) and
    // the generation is tried again; when everything up to `retire_ticket` is done, nothing reads the current table
    // and it is rewritten in place if it is large enough. Failing both, the flush degrades instead of failing: the
    // descriptors are written in place, as on a flush that does not replace the table (add_image writes new indices,
    // which no key of the GPU's table reaches; remove_image writes the fallback, and the index stays held until a
    // flush that does not degrade), the table keeps what it had, and its changes wait for a later flush
    // (last_flush_degraded()). False only without a command buffer, when nothing can be recorded.
    bool flush(VkCommandBuffer command_buffer, std::uint64_t retire_ticket = 0,
               const std::function<std::uint64_t()>& completed_ticket = {}) {
        if (!command_buffer) return false;
        bool replaced = false;
        bool written = true;
        const bool must_replace = flush_replaces_table();
        last_flush_degraded_ = false;
        if (table_dirty_ || !table_uploaded_) written = upload_table(command_buffer, retire_ticket, replaced);
        if (!written && must_replace && completed_ticket) {
            const std::uint64_t completed = completed_ticket();
            if (!retired_.empty()) {
                release_retired(completed);
                written = upload_table(command_buffer, retire_ticket, replaced);
            }
            if (!written && completed >= retire_ticket) written = rewrite_table_in_place(command_buffer);
        }
        last_flush_replaced_ = replaced;
        // Degrading keeps what the current buffers hold, so only once they hold a table that reached the queue: a
        // first flush under full memory, or one after a failed submission of the only upload, has nothing on the GPU
        // (word 0 is the probe mask) and refuses as before (review of #64).
        if (!written && must_replace && !table_on_gpu_) return false;
        if (!written && must_replace) {
            // The current set is the one running work reads, and the table's changes may move or rebind live keys:
            // they stay for a later flush, and keys added since find the fallback until then.
            ++degraded_flushes_;
            last_flush_degraded_ = true;
        }
        for (const Pending& pending : pending_) {
            if (pending.binding >= shadow_.size() || pending.index >= shadow_[pending.binding].size()) continue;
            shadow_[pending.binding][pending.index] = pending.info;
            shadow_written_[pending.binding][pending.index] = 1;
        }
        if (replaced) {
            write_shadow();  // includes the pending writes just folded in
        } else if (!pending_.empty()) {
            std::vector<VkWriteDescriptorSet> writes(pending_.size());
            for (std::size_t i = 0; i < pending_.size(); ++i) {
                writes[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
                writes[i].dstSet = set_;
                writes[i].dstBinding = pending_[i].binding;
                writes[i].dstArrayElement = pending_[i].index;
                writes[i].descriptorCount = 1;
                writes[i].descriptorType = bindless::bindings[pending_[i].binding].type;
                writes[i].pImageInfo = &pending_[i].info;
            }
            vk_.vkUpdateDescriptorSets(device_, std::uint32_t(writes.size()), writes.data(), 0, nullptr);
        }
        pending_.clear();
        pending_slots_.clear();
        if (!last_flush_degraded_) {
            // The table this flush leaves on the GPU names no removed index any more: they can be given out again.
            for (const auto& [dimension, index] : held_images_) image_used_[std::uint32_t(dimension)][index] = 0;
            for (const std::uint32_t index : held_samplers_) sampler_used_[index] = 0;
            held_images_.clear();
            held_samplers_.clear();
        }
        return true;
    }

    // Releases the generations (table buffers and descriptor set) that flush() retired with a ticket up to
    // `completed`: up to max_spares of them, large enough for the table, are kept for the next replacing flush to
    // reuse; the others are freed. The default releases all of them: when nothing submitted before the last flush()
    // still runs.
    void release_retired(std::uint64_t completed = UINT64_MAX) {
        std::size_t kept = 0;
        for (auto& generation : retired_) {
            if (generation.ticket <= completed) {
                if (spares_.size() < max_spares && generation.pool && generation.buffers.staging.mapped &&
                    generation.buffers.table.size >= table_.size_bytes())
                    spares_.push_back(generation);
                else
                    destroy(generation);
            } else {
                retired_[kept++] = generation;
            }
        }
        retired_.resize(kept);
    }
    bool has_retired() const { return !retired_.empty(); }
    // True when the last flush() started a new generation: its table upload is in that flush's command buffer, so
    // work on other queues that binds the new set must run after it (the owner orders them).
    bool last_flush_replaced() const { return last_flush_replaced_; }
    // The ticket of the oldest retired generation (0 when there is none).
    std::uint64_t oldest_retired_ticket() const { return retired_.empty() ? 0 : retired_.front().ticket; }
    // How many generations flush() has made after the first, and how many wait for release_retired().
    std::uint64_t generations_made() const { return generations_made_; }
    std::size_t retired_count() const { return retired_.size(); }

    // Released generations kept for reuse, and how many replacing flushes reused one instead of allocating. One: the
    // current generation and a spare are the two a replacing flush has at its peak anyway (the new one, and the one
    // retired until its work completes), and a grown table can be large (64 MiB at a million slots).
    static constexpr std::size_t max_spares = 1;
    std::size_t spare_count() const { return spares_.size(); }
    std::uint64_t spares_reused() const { return spares_reused_; }
    // Flushes that could not make the generation they needed and kept the previous table (see flush()), and whether
    // the last one did; flushes that rewrote the table in place instead, every submission being done.
    std::uint64_t degraded_flushes() const { return degraded_flushes_; }
    bool last_flush_degraded() const { return last_flush_degraded_; }
    std::uint64_t in_place_rewrites() const { return in_place_rewrites_; }
    // Table buffers placed in HOST_VISIBLE memory because the DEVICE_LOCAL allocation failed.
    std::uint64_t table_host_fallbacks() const { return table_host_fallbacks_; }
    // What the last generation that could not be made failed to get: "table buffer", "staging buffer" or
    // "descriptor set" (nullptr before any failure), and the Vulkan result of that call.
    const char* last_failure() const { return last_failure_; }
    VkResult last_failure_result() const { return last_failure_result_; }
    // Bytes of the current table buffer (the table grows by doubling its slots: size_bytes() of table()).
    std::size_t table_buffer_bytes() const { return table_buffers_.table.size; }

private:
    struct Functions {
#define LEONIDA_BINDLESS_FUNCTIONS(X)                                                                          \
    X(vkCreateDescriptorSetLayout) X(vkDestroyDescriptorSetLayout) X(vkCreateDescriptorPool)                   \
    X(vkDestroyDescriptorPool) X(vkResetDescriptorPool) X(vkAllocateDescriptorSets) X(vkUpdateDescriptorSets)  \
    X(vkCreateBuffer)                                                                                          \
    X(vkDestroyBuffer) X(vkGetBufferMemoryRequirements) X(vkBindBufferMemory) X(vkAllocateMemory)              \
    X(vkFreeMemory) X(vkMapMemory) X(vkCreateImage) X(vkDestroyImage) X(vkGetImageMemoryRequirements)          \
    X(vkBindImageMemory) X(vkCreateImageView) X(vkDestroyImageView) X(vkCreateSampler) X(vkDestroySampler)     \
    X(vkCmdPipelineBarrier) X(vkCmdCopyBuffer) X(vkCmdUpdateBuffer) X(vkCmdClearColorImage)
#define LEONIDA_BINDLESS_MEMBER(name) PFN_##name name = nullptr;
        LEONIDA_BINDLESS_FUNCTIONS(LEONIDA_BINDLESS_MEMBER)
#undef LEONIDA_BINDLESS_MEMBER

        bool load(PFN_vkGetDeviceProcAddr get, VkDevice device) {
            bool all = true;
#define LEONIDA_BINDLESS_LOAD(name)                                      \
    name = reinterpret_cast<PFN_##name>(get(device, #name));             \
    all = all && name != nullptr;
            LEONIDA_BINDLESS_FUNCTIONS(LEONIDA_BINDLESS_LOAD)
#undef LEONIDA_BINDLESS_LOAD
#undef LEONIDA_BINDLESS_FUNCTIONS
            return all;
        }
    };

    struct FallbackImage {
        VkImage image = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
    };

    struct TableBuffer {
        VkBuffer buffer = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        void* mapped = nullptr;
        std::size_t size = 0;
        bool device_local = false;  // the memory is DEVICE_LOCAL
    };

    // The table and the mapped staging copy the CPU writes before vkCmdCopyBuffer moves the changes.
    struct TableBuffers {
        TableBuffer table;    // DEVICE_LOCAL when the device has it; what binding 2 names
        TableBuffer staging;  // HOST_VISIBLE | HOST_COHERENT
    };

    struct Generation {  // what a replaced table leaves for release_retired()
        TableBuffers buffers;
        VkDescriptorPool pool = VK_NULL_HANDLE;  // owns the descriptor set
        VkDescriptorSet set = VK_NULL_HANDLE;
        std::uint64_t ticket = 0;
    };

    struct Pending {
        std::uint32_t binding;
        std::uint32_t index;
        VkDescriptorImageInfo info;
    };

    static constexpr std::uint32_t dimensions = 7, classes = 3;
    // image_used_ and sampler_used_: 0 free, in_use, or held (removed while the table was dirty: free after the next
    // flush that does not degrade).
    static constexpr std::uint8_t in_use = 1, held = 2;

    // The first free index in [begin, end), marked used; 0 when there is none.
    static std::uint32_t allocate(std::vector<std::uint8_t>& used, std::uint32_t begin, std::uint32_t end) {
        for (std::uint32_t i = begin; i < end; ++i) {
            if (used[i]) continue;
            used[i] = 1;
            return i;
        }
        return 0;
    }

    // The first index in [begin, end) free in both arrays, marked used in the first; 0 when there is none. The
    // second array's size bounds the range: a mirrored index must exist in it (its 2D slot).
    static std::uint32_t allocate_pair(std::vector<std::uint8_t>& used, std::vector<std::uint8_t>& other,
                                       std::uint32_t begin, std::uint32_t end) {
        end = std::min(end, std::uint32_t(other.size()));
        for (std::uint32_t i = begin; i < end; ++i) {
            if (used[i] || other[i]) continue;
            used[i] = 1;
            return i;
        }
        return 0;
    }

    // A mirror is worth binding only when it has a view; its storage views make the 2D storage arrays valid too.
    static bool mirror_storage(const MirrorViews* mirror) {
        return mirror && (mirror->storage.as_float || mirror->storage.as_uint || mirror->storage.as_sint ||
                          mirror->storage.r32ui);
    }

    // True when the 2D slot at `index` is the mirror of the image whose index is `index` in `dimension` (add_image's
    // MirrorViews). A bare per-slot flag cannot say so: every dimension's index space is dense and independent, so
    // the Nth image of one dimension shares its slot number with the Nth 2D-array image's mirror slot in the 2D
    // arrays. The owner is the dimension + 1 (0: no mirror).
    bool mirror_of(ImageDimension dimension, std::uint32_t index) const {
        if (dimension == ImageDimension::dim_2d) return false;
        return index < mirror_owner_.size() && mirror_owner_[index] == std::uint8_t(std::uint32_t(dimension) + 1);
    }

    std::uint32_t memory_type(std::uint32_t allowed, VkMemoryPropertyFlags wanted) const {
        for (std::uint32_t i = 0; i < memory_.memoryTypeCount; ++i)
            if ((allowed & (1u << i)) && (memory_.memoryTypes[i].propertyFlags & wanted) == wanted) return i;
        return 0xffffffffu;
    }

    bool create_layout() {
        const bindless::SetLayout layout(features_);
        const std::uint32_t count = layout.info.bindingCount;
        pool_size_count_ = count;
        for (std::uint32_t i = 0; i < count; ++i)
            pool_sizes_[i] = {layout.entries[i].descriptorType, layout.entries[i].descriptorCount};
        if (vk_.vkCreateDescriptorSetLayout(device_, &layout.info, nullptr, &layout_) != VK_SUCCESS) return false;
        if (!make_set(pool_, set_)) return false;
        for (std::uint32_t b = 0; b < bindless::bindings.size(); ++b) {
            if (b == bindless::table_binding || !bindless::binding_enabled(b, features_)) continue;
            shadow_[b].assign(features_.counts[b], VkDescriptorImageInfo{});
            shadow_written_[b].assign(features_.counts[b], 0);
        }

        for (std::uint32_t d = 0; d < dimensions; ++d) {
            image_used_[d].assign(image_capacity(ImageDimension(d)), 0);
            image_used_[d][0] = 1;
        }
        for (std::uint32_t d = 0; d < dimensions; ++d)
            storage_used_[d].assign(image_capacity(ImageDimension(d)), 0);
        mirror_owner_.assign(image_capacity(ImageDimension::dim_2d), 0);
        mirror_storage_used_.assign(image_capacity(ImageDimension::dim_2d), 0);
        sampler_used_.assign(sampler_capacity(), 0);
        sampler_used_[0] = 1;
        return true;
    }

    // A one-set pool and its set at the layout's sizes: the first generation's, and each replacement's.
    bool make_set(VkDescriptorPool& pool, VkDescriptorSet& set) const {
        VkDescriptorPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        pool_info.flags = features_.update_after_bind ? VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT : 0;
        pool_info.maxSets = 1;
        pool_info.poolSizeCount = pool_size_count_;
        pool_info.pPoolSizes = pool_sizes_.data();
        if (const VkResult result = vk_.vkCreateDescriptorPool(device_, &pool_info, nullptr, &pool);
            result != VK_SUCCESS) {
            pool = VK_NULL_HANDLE;
            note_failure("descriptor set", result);
            return false;
        }
        if (allocate_set(pool, set)) return true;
        vk_.vkDestroyDescriptorPool(device_, pool, nullptr);
        pool = VK_NULL_HANDLE;
        return false;
    }

    bool allocate_set(VkDescriptorPool pool, VkDescriptorSet& set) const {
        VkDescriptorSetAllocateInfo set_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        set_info.descriptorPool = pool;
        set_info.descriptorSetCount = 1;
        set_info.pSetLayouts = &layout_;
        const VkResult result = vk_.vkAllocateDescriptorSets(device_, &set_info, &set);
        if (result == VK_SUCCESS) return true;
        set = VK_NULL_HANDLE;
        note_failure("descriptor set", result);
        return false;
    }

    void note_failure(const char* what, VkResult result) const {
        last_failure_ = what;
        last_failure_result_ = result;
    }

    // Writes every descriptor written so far (the shadow) into the current set, one update per run of written
    // elements: the set was just made, and unwritten elements stay unwritten, as partially_bound allows.
    void write_shadow() const {
        std::vector<VkWriteDescriptorSet> writes;
        for (std::uint32_t b = 0; b < bindless::bindings.size(); ++b) {
            const auto& written = shadow_written_[b];
            for (std::size_t i = 0; i < written.size();) {
                if (!written[i]) {
                    ++i;
                    continue;
                }
                std::size_t end = i;
                while (end < written.size() && written[end]) ++end;
                VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
                write.dstSet = set_;
                write.dstBinding = b;
                write.dstArrayElement = std::uint32_t(i);
                write.descriptorCount = std::uint32_t(end - i);
                write.descriptorType = bindless::bindings[b].type;
                write.pImageInfo = &shadow_[b][i];
                writes.push_back(write);
                i = end;
            }
        }
        if (!writes.empty())
            vk_.vkUpdateDescriptorSets(device_, std::uint32_t(writes.size()), writes.data(), 0, nullptr);
    }

    // The fallbacks of every binding (the class comment): a 1x1 sampled image for each image type (each MS class
    // its own), and the storage arrays' 1x1 images of each type and component class, or null descriptors with
    // features_.null_descriptor. A sampled and a storage fallback never share an image.
    bool create_fallbacks() {
        // Each MS class shares one sampled-only fallback between its 2D and 2D-array views; so does each single-sample
        // type between its 2D and 2D-array views, sampled and storage alike.
        constexpr VkImageType types[] = {VK_IMAGE_TYPE_2D, VK_IMAGE_TYPE_3D, VK_IMAGE_TYPE_1D, VK_IMAGE_TYPE_2D};
        constexpr std::uint32_t ms_type = 3;
        constexpr VkFormat formats[classes] = {VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R32_UINT, VK_FORMAT_R32_SINT};
        // dimension -> (index into types, view type)
        constexpr std::uint32_t type_of[dimensions] = {0, 0, 1, 2, 2, ms_type, ms_type};
        constexpr VkImageViewType view_types[dimensions] = {VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_VIEW_TYPE_2D_ARRAY,
                                                            VK_IMAGE_VIEW_TYPE_3D, VK_IMAGE_VIEW_TYPE_1D,
                                                            VK_IMAGE_VIEW_TYPE_1D_ARRAY, VK_IMAGE_VIEW_TYPE_2D,
                                                            VK_IMAGE_VIEW_TYPE_2D_ARRAY};
        const auto create_image = [&](FallbackImage& image, std::uint32_t t, std::uint32_t c, VkImageUsageFlags usage) {
            VkImageCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
            info.imageType = types[t];
            info.format = formats[c];
            info.extent = {1, 1, 1};
            info.mipLevels = 1;
            info.arrayLayers = 1;
            info.samples = t == ms_type ? features_.ms_samples[c] : VK_SAMPLE_COUNT_1_BIT;
            info.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | usage;
            share_across_queue_families(info, families_);
            if (vk_.vkCreateImage(device_, &info, nullptr, &image.image) != VK_SUCCESS) return false;
            VkMemoryRequirements requirements{};
            vk_.vkGetImageMemoryRequirements(device_, image.image, &requirements);
            VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
            allocation.allocationSize = requirements.size;
            allocation.memoryTypeIndex = memory_type(requirements.memoryTypeBits, 0);
            return allocation.memoryTypeIndex != 0xffffffffu &&
                   vk_.vkAllocateMemory(device_, &allocation, nullptr, &image.memory) == VK_SUCCESS &&
                   vk_.vkBindImageMemory(device_, image.image, image.memory, 0) == VK_SUCCESS;
        };
        const auto create_view = [&](const FallbackImage& image, std::uint32_t d, std::uint32_t c, VkImageView& view) {
            VkImageViewCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
            info.image = image.image;
            info.viewType = view_types[d];
            info.format = formats[c];
            info.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            if (vk_.vkCreateImageView(device_, &info, nullptr, &view) != VK_SUCCESS) return false;
            fallback_views_.push_back(view);
            return true;
        };
        for (std::uint32_t t = 0; t < std::size(types); ++t) {
            for (std::uint32_t c = 0; c < classes; ++c) {
                // Sampled: class 0 of the single-sample types (the translator reads a miss through the float view),
                // every supported class of MS.
                if (t == ms_type ? supports_ms_class(c) : c == 0) {
                    if (!create_image(fallback_images_[t * classes + c], t, c, VK_IMAGE_USAGE_SAMPLED_BIT)) return false;
                }
                if (t != ms_type && !features_.null_descriptor &&
                    !create_image(storage_fallback_images_[t * classes + c], t, c, VK_IMAGE_USAGE_STORAGE_BIT))
                    return false;
            }
        }
        for (std::uint32_t d = 0; d < dimensions; ++d) {
            const ImageDimension dimension{d};
            for (std::uint32_t c = 0; c < classes; ++c) {
                const FallbackImage& sampled = fallback_images_[type_of[d] * classes + c];
                if (sampled.image) {
                    VkImageView view = VK_NULL_HANDLE;
                    if (!create_view(sampled, d, c, view)) return false;
                    const auto sampled_at = bindless::sampled_binding(dimension, c);
                    if (sampled_at != bindless::no_binding && (c == 0 || bindless::multisampled(dimension)))
                        fallback_by_binding_[sampled_at] = view;
                }
                const auto storage_at = bindless::storage_binding(dimension, c);
                if (storage_at == bindless::no_binding) continue;  // MS: no storage arrays
                // With null_descriptor the storage fallbacks stay VK_NULL_HANDLE: null descriptors.
                VkImageView view = VK_NULL_HANDLE;
                if (!features_.null_descriptor &&
                    !create_view(storage_fallback_images_[type_of[d] * classes + c], d, c, view))
                    return false;
                fallback_by_binding_[storage_at] = view;
                const auto r32ui_at = bindless::r32ui_binding(dimension);
                if (c == 1 && r32ui_at != bindless::no_binding) fallback_by_binding_[r32ui_at] = view;
            }
        }
        VkSamplerCreateInfo sampler{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        sampler.magFilter = sampler.minFilter = VK_FILTER_NEAREST;
        sampler.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        sampler.addressModeU = sampler.addressModeV = sampler.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sampler.maxLod = VK_LOD_CLAMP_NONE;
        return vk_.vkCreateSampler(device_, &sampler, nullptr, &fallback_sampler_) == VK_SUCCESS;
    }

    // device_local: the table (DEVICE_LOCAL, else HOST_VISIBLE when the device has no such type or that heap is out
    // of memory); otherwise the mapped staging buffer (HOST_VISIBLE | HOST_COHERENT).
    bool create(TableBuffer& out, std::size_t bytes, bool device_local) {
        constexpr VkMemoryPropertyFlags host = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        const char* what = device_local ? "table buffer" : "staging buffer";
        VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        info.size = bytes;
        info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                     (device_local ? VK_BUFFER_USAGE_TRANSFER_DST_BIT : VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
        TableBuffer made;
        made.size = bytes;
        share_across_queue_families(info, families_);
        if (const VkResult result = vk_.vkCreateBuffer(device_, &info, nullptr, &made.buffer); result != VK_SUCCESS) {
            made.buffer = VK_NULL_HANDLE;
            note_failure(what, result);
            return false;
        }
        VkMemoryRequirements requirements{};
        vk_.vkGetBufferMemoryRequirements(device_, made.buffer, &requirements);
        VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        allocation.allocationSize = requirements.size;
        const auto allocate = [&](std::uint32_t type) {
            if (type == 0xffffffffu) return VK_ERROR_OUT_OF_DEVICE_MEMORY;
            allocation.memoryTypeIndex = type;
            const VkResult result = vk_.vkAllocateMemory(device_, &allocation, nullptr, &made.memory);
            if (result != VK_SUCCESS) made.memory = VK_NULL_HANDLE;
            return result;
        };
        const std::uint32_t local = device_local
                                        ? memory_type(requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
                                        : 0xffffffffu;
        VkResult result = device_local ? allocate(local) : allocate(memory_type(requirements.memoryTypeBits, host));
        if (result != VK_SUCCESS && device_local) {
            const std::uint32_t fallback = memory_type(requirements.memoryTypeBits, host);
            if (fallback != local) {
                result = allocate(fallback);
                if (result == VK_SUCCESS && local != 0xffffffffu) ++table_host_fallbacks_;
            }
        }
        if (result == VK_SUCCESS) {
            made.device_local = (memory_.memoryTypes[allocation.memoryTypeIndex].propertyFlags &
                                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0;
            result = vk_.vkBindBufferMemory(device_, made.buffer, made.memory, 0);
        }
        if (result == VK_SUCCESS && !device_local)
            result = vk_.vkMapMemory(device_, made.memory, 0, VK_WHOLE_SIZE, 0, &made.mapped);
        if (result != VK_SUCCESS) {
            note_failure(what, result);
            destroy(made);
            return false;
        }
        out = made;
        return true;
    }

    void destroy(TableBuffer& buffer) const {
        if (buffer.buffer) vk_.vkDestroyBuffer(device_, buffer.buffer, nullptr);
        if (buffer.memory) vk_.vkFreeMemory(device_, buffer.memory, nullptr);  // unmaps it
        buffer = {};
    }

    void destroy(TableBuffers& buffers) const {
        destroy(buffers.staging);
        destroy(buffers.table);
    }

    void destroy(Generation& generation) const {
        destroy(generation.buffers);
        if (generation.pool) vk_.vkDestroyDescriptorPool(device_, generation.pool, nullptr);  // frees the set
        generation.pool = VK_NULL_HANDLE;
        generation.set = VK_NULL_HANDLE;
    }

    // A generation for a table of `bytes`: a spare when one is large enough (its pool reset for a fresh set), else
    // new buffers and a new pool. Spares too small for the table are freed: the table never shrinks.
    bool make_generation(std::size_t bytes, TableBuffers& buffers, VkDescriptorPool& pool, VkDescriptorSet& set) {
        std::size_t kept = 0;
        bool found = false;
        for (Generation& spare : spares_) {
            if (spare.buffers.table.size < bytes || spare.buffers.staging.size < bytes) {
                destroy(spare);
                continue;
            }
            if (!found && vk_.vkResetDescriptorPool(device_, spare.pool, 0) == VK_SUCCESS &&
                allocate_set(spare.pool, set)) {
                buffers = spare.buffers;
                pool = spare.pool;
                found = true;
                ++spares_reused_;
                continue;
            }
            spares_[kept++] = spare;
        }
        spares_.resize(kept);
        if (found) return true;
        TableBuffers made;
        if (!create(made.table, bytes, true) || !create(made.staging, bytes, false) || !make_set(pool, set)) {
            destroy(made);
            return false;
        }
        buffers = made;
        return true;
    }

    bool create_table_buffers() {
        const std::size_t bytes = table_.size_bytes();
        TableBuffers made;
        if (!create(made.table, bytes, true) || !create(made.staging, bytes, false)) {
            destroy(made);
            return false;
        }
        table_buffers_ = made;
        table_device_local_ = made.table.device_local;
        table_uploaded_ = false;  // the first flush() uploads it, in the command buffer that will use it
        table_on_gpu_ = false;
        table_dirty_ = existing_entries_changed_ = false;
        return true;
    }

    // Uploads what changed since the last call: the whole table when it grew (the mask moved) or was never uploaded,
    // the changed slots otherwise. The copy and its barrier go into `command_buffer`, before the draws.
    bool upload_table(VkCommandBuffer command_buffer, std::uint64_t retire_ticket, bool& replaced) {
        if (!command_buffer) return false;
        const std::size_t bytes = table_.size_bytes();
        // A table that grew, or whose live entries change, goes to a new generation; the old one may still be read.
        const bool replace = bytes > table_buffers_.table.size ||
                             (table_uploaded_ && (existing_entries_changed_ || force_full_upload_));
        bool full = force_full_upload_ || !table_uploaded_ || replace;
        if (replace) {
            TableBuffers larger;
            VkDescriptorPool pool = VK_NULL_HANDLE;
            VkDescriptorSet set = VK_NULL_HANDLE;
            if (!make_generation(bytes, larger, pool, set)) return false;
            table_device_local_ = larger.table.device_local;
            retired_.push_back({table_buffers_, pool_, set_, retire_ticket});
            table_buffers_ = larger;
            table_on_gpu_ = false;  // until this generation's upload is submitted (table_upload_result)
            pool_ = pool;
            set_ = set;
            write_table_binding(table_buffers_.table.buffer);
            replaced = true;
            ++generations_made_;
        }
        DescriptorTable::Changes changes = table_.take_changes();
        full = full || changes.all;
        bool recorded = false;
        // The staging buffer is written only for buffers that were just created (first upload or growth): a later
        // flush then either updates slots with vkCmdUpdateBuffer or rebuilds from a fresh pair, so no copy still in
        // flight can be reading bytes the CPU rewrites.
        const bool fresh = replace || !table_uploaded_;
        const auto upload_all = [&]() {
            if (fresh) {
                table_.write_to_buffer(table_buffers_.staging.mapped);
                copy_table_range(command_buffer, 0, bytes);
            } else {
                const std::vector<std::uint32_t> words = table_.build_gpu_buffer();
                update_table_range(command_buffer, 0, words.data(), bytes);
            }
            recorded = true;
        };
        if (full) {
            upload_all();
        } else if (!changes.slots.empty()) {
            std::vector<std::uint32_t> slots = std::move(changes.slots);
            std::sort(slots.begin(), slots.end());
            slots.erase(std::unique(slots.begin(), slots.end()), slots.end());
            std::vector<std::uint32_t> words;
            bool staged = true;
            for (std::size_t first = 0; staged && first < slots.size();) {
                std::size_t last = first;
                while (last + 1 < slots.size() && slots[last + 1] == slots[last] + 1) ++last;
                const std::uint32_t count = std::uint32_t(last - first + 1);
                words.resize(std::size_t(count) * 10);
                staged = table_.read_slots(slots[first], count, words.data());
                if (staged) {
                    update_table_range(command_buffer, entry_offset(slots[first]), words.data(),
                                       VkDeviceSize(words.size() * sizeof(std::uint32_t)));
                    recorded = true;
                }
                first = last + 1;
            }
            if (!staged) upload_all();  // an inconsistent slot list: the whole table is still correct
        }
        if (!recorded) return true;  // nothing was copied: keep the obligation for the next flush
        record_table_barrier(command_buffer);
        upload_in_flight_ = true;
        table_dirty_ = existing_entries_changed_ = false;
        table_uploaded_ = true;
        force_full_upload_ = false;
        return true;
    }

    // The whole table into the current generation's buffers, for a flush whose new generation could not be made,
    // once every submission that could read the current table is done (flush()). False when the table outgrew them.
    bool rewrite_table_in_place(VkCommandBuffer command_buffer) {
        const std::size_t bytes = table_.size_bytes();
        if (bytes > table_buffers_.table.size || bytes > table_buffers_.staging.size || !table_buffers_.staging.mapped)
            return false;
        table_.take_changes();  // all of it goes
        // Nothing still runs that copies from the staging buffer or reads the table.
        table_.write_to_buffer(table_buffers_.staging.mapped);
        copy_table_range(command_buffer, 0, bytes);
        record_table_barrier(command_buffer);
        upload_in_flight_ = true;
        table_dirty_ = existing_entries_changed_ = false;
        table_uploaded_ = true;
        force_full_upload_ = false;
        ++in_place_rewrites_;
        return true;
    }

    // Copies the data into the command buffer while recording (64 KiB per call, the Vulkan limit), so no staging
    // buffer is read at execution: a later flush can rewrite its own staging without racing a pending copy.
    void update_table_range(VkCommandBuffer command_buffer, VkDeviceSize offset, const void* data,
                            VkDeviceSize size) const {
        constexpr VkDeviceSize max_chunk = 65536;
        const auto* bytes = static_cast<const std::uint8_t*>(data);
        for (VkDeviceSize done = 0; done < size; done += max_chunk) {
            const VkDeviceSize chunk = std::min(max_chunk, size - done);
            vk_.vkCmdUpdateBuffer(command_buffer, table_buffers_.table.buffer, offset + done, chunk, bytes + done);
        }
    }

    void copy_table_range(VkCommandBuffer command_buffer, VkDeviceSize offset, VkDeviceSize size) const {
        const VkBufferCopy region{offset, offset, size};
        vk_.vkCmdCopyBuffer(command_buffer, table_buffers_.staging.buffer, table_buffers_.table.buffer, 1, &region);
    }

    void record_table_barrier(VkCommandBuffer command_buffer) const {
        VkBufferMemoryBarrier barrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.buffer = table_buffers_.table.buffer;
        barrier.offset = 0;
        barrier.size = VK_WHOLE_SIZE;
        vk_.vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0,
                                 nullptr, 1, &barrier, 0, nullptr);
    }

    // The entry's first dword after the mask word; each entry is 10 dwords (40 bytes).
    static constexpr VkDeviceSize entry_bytes = 10 * sizeof(std::uint32_t);
    static VkDeviceSize entry_offset(std::uint32_t slot) {
        return sizeof(std::uint32_t) + VkDeviceSize(slot) * entry_bytes;
    }

    void write_table_binding(VkBuffer buffer) const {
        const VkDescriptorBufferInfo info{buffer, 0, VK_WHOLE_SIZE};
        VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        write.dstSet = set_;
        write.dstBinding = bindless::table_binding;
        write.descriptorCount = 1;
        write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        write.pBufferInfo = &info;
        vk_.vkUpdateDescriptorSets(device_, 1, &write, 0, nullptr);
    }

    VkDescriptorImageInfo fallback_info(std::uint32_t binding) const {
        if (binding == bindless::sampler_binding) return {fallback_sampler_, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED};
        return {VK_NULL_HANDLE, fallback_by_binding_[binding], VK_IMAGE_LAYOUT_GENERAL};
    }

    // The table at binding 2 and the fallback at index 0 of every array, or in every element without partially_bound.
    bool fill() {
        write_table_binding(table_buffers_.table.buffer);
        std::vector<std::vector<VkDescriptorImageInfo>> infos;
        std::vector<VkWriteDescriptorSet> writes;
        infos.reserve(bindless::bindings.size());
        for (std::uint32_t b = 0; b < bindless::bindings.size(); ++b) {
            if (b == bindless::table_binding || !bindless::binding_enabled(b, features_)) continue;
            const std::uint32_t count = features_.partially_bound ? 1 : features_.counts[b];
            infos.emplace_back(count, fallback_info(b));
            std::fill_n(shadow_[b].begin(), count, infos.back()[0]);
            std::fill_n(shadow_written_[b].begin(), count, std::uint8_t(1));
            VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            write.dstSet = set_;
            write.dstBinding = b;
            write.descriptorCount = count;
            write.descriptorType = bindless::bindings[b].type;
            write.pImageInfo = infos.back().data();
            writes.push_back(write);
        }
        vk_.vkUpdateDescriptorSets(device_, std::uint32_t(writes.size()), writes.data(), 0, nullptr);
        return true;
    }

    // A slot has one pending write: a later one takes its place. So a view or sampler that is removed before the
    // next flush(), and may be destroyed by then, is never written.
    void queue(std::uint32_t binding, std::uint32_t index, const VkDescriptorImageInfo& info) {
        const std::uint64_t slot = (std::uint64_t(binding) << 32) | index;
        const auto [at, added] = pending_slots_.try_emplace(slot, pending_.size());
        if (added)
            pending_.push_back({binding, index, info});
        else
            pending_[at->second].info = info;
    }

    void write_sampler(std::uint32_t index, VkSampler sampler) {
        queue(bindless::sampler_binding, index, {sampler, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED});
    }

    // Queues every binding of the dimension at `index`: the given views, or the fallback where one is null.
    void write_image(ImageDimension dimension, std::uint32_t index, VkImageView sampled, VkImageLayout sampled_layout,
                     const StorageViews& storage, std::uint32_t sampled_class) {
        if (bindless::multisampled(dimension)) {
            for (std::uint32_t c = 0; c < classes; ++c) {
                if (!supports_ms_class(c)) continue;
                const auto binding = bindless::sampled_binding(dimension, c);
                queue(binding, index, c == sampled_class && sampled
                                          ? VkDescriptorImageInfo{VK_NULL_HANDLE, sampled, sampled_layout}
                                          : fallback_info(binding));
            }
            return;
        }
        const std::uint32_t sampled_at = bindless::sampled_binding(dimension);
        if (sampled_at != bindless::no_binding)
            queue(sampled_at, index,
                  sampled ? VkDescriptorImageInfo{VK_NULL_HANDLE, sampled, sampled_layout} : fallback_info(sampled_at));
        const auto store = [&](std::uint32_t binding, VkImageView view) {
            if (binding == bindless::no_binding || index >= features_.counts[binding]) return;
            queue(binding, index,
                  view ? VkDescriptorImageInfo{VK_NULL_HANDLE, view, VK_IMAGE_LAYOUT_GENERAL} : fallback_info(binding));
        };
        store(bindless::storage_binding(dimension, 0), storage.as_float);
        store(bindless::storage_binding(dimension, 1), storage.as_uint);
        store(bindless::storage_binding(dimension, 2), storage.as_sint);
        store(bindless::r32ui_binding(dimension), storage.r32ui);
    }

    VkDevice device_;
    VkPhysicalDeviceMemoryProperties memory_;
    BindlessFeatures features_;
    std::vector<std::uint32_t> families_;
    Functions vk_;
    bool ok_ = false;
    VkDescriptorSetLayout layout_ = VK_NULL_HANDLE;
    VkDescriptorPool pool_ = VK_NULL_HANDLE;
    VkDescriptorSet set_ = VK_NULL_HANDLE;
    std::array<FallbackImage, 4 * classes> fallback_images_{};  // sampled: [2D, 3D, 1D, 2D MS][class]
    // Storage, without null_descriptor: [2D, 3D, 1D][class]. Never the image of a sampled fallback.
    std::array<FallbackImage, 3 * classes> storage_fallback_images_{};
    std::vector<VkImageView> fallback_views_;
    std::array<VkImageView, bindless::bindings.size()> fallback_by_binding_{};
    VkSampler fallback_sampler_ = VK_NULL_HANDLE;
    bool fallbacks_ready_ = false;
    std::array<std::vector<std::uint8_t>, dimensions> image_used_;
    // Which of those slots also hold storage views (add_image's `storage`), and which 2D slots hold the mirror of an
    // image in another dimension (add_image's MirrorViews): removing that image frees the 2D slot too. insert_key
    // reads both to say which arrays an index is bound in. mirror_owner_ says whose mirror each of those 2D slots is
    // (dimension + 1, 0 for none) and mirror_storage_used_ whether it has storage views: the 2D slot numbers line up
    // with another dimension's indices, so the slot alone does not identify its owner.
    std::array<std::vector<std::uint8_t>, dimensions> storage_used_;
    std::vector<std::uint8_t> mirror_owner_;
    std::vector<std::uint8_t> mirror_storage_used_;
    std::vector<std::uint8_t> sampler_used_;
    std::vector<std::pair<ImageDimension, std::uint32_t>> held_images_;  // removed, not yet free (remove_image)
    std::vector<std::uint32_t> held_samplers_;
    std::vector<Pending> pending_;
    std::unordered_map<std::uint64_t, std::size_t> pending_slots_;  // binding and index -> place in pending_
    DescriptorTable table_;  // 64 slots, and the device's hash kind, from the constructor
    TableBuffers table_buffers_;
    bool table_uploaded_ = false;  // the table's current contents are in table_buffers_.table
    bool table_on_gpu_ = false;    // an upload into the current table buffers reached the queue (degrading is safe)
    bool table_device_local_ = false;
    bool table_dirty_ = false;    // the table differs from the buffer
    bool existing_entries_changed_ = false;  // ... and not only by entries added
    bool upload_in_flight_ = false;   // a copy is recorded, not yet known to have reached the queue
    bool force_full_upload_ = false;  // the last submit failed: rebuild the whole table on the next flush
    std::vector<Generation> retired_;
    std::vector<Generation> spares_;  // released generations kept for reuse (release_retired)
    std::uint64_t generations_made_ = 0;
    bool last_flush_replaced_ = false;
    bool last_flush_degraded_ = false;
    std::uint64_t spares_reused_ = 0;
    std::uint64_t degraded_flushes_ = 0;
    std::uint64_t in_place_rewrites_ = 0;
    std::uint64_t table_host_fallbacks_ = 0;
    mutable const char* last_failure_ = nullptr;
    mutable VkResult last_failure_result_ = VK_SUCCESS;
    std::array<VkDescriptorPoolSize, bindless::bindings.size()> pool_sizes_{};
    std::uint32_t pool_size_count_ = 0;
    // Every descriptor written to the set (the table's binding aside), to fill the next generation's set.
    std::array<std::vector<VkDescriptorImageInfo>, bindless::bindings.size()> shadow_;
    std::array<std::vector<std::uint8_t>, bindless::bindings.size()> shadow_written_;
};

} // namespace leonida
