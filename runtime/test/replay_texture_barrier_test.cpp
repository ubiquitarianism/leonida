// emit of an enhanced texture barrier (TextureBarrier): one VkImageMemoryBarrier2 for each layout piece, in one
// vkCmdPipelineBarrier2 call.
//   - A range in one layout is one piece: emit makes no heap allocation for it.
//   - A range whose mips are in different layouts is one piece for each mip: all of them are in the same call,
//     each with its own old layout.
//   - A barrier into UNDEFINED from a defined layout keeps each piece's layout (Vulkan has no transition into
//     UNDEFINED) and the tracker's; a piece that was never used records nothing, and the next barrier, from
//     UNDEFINED, discards as before.
// vkCmdPipelineBarrier2 is a function that records its barriers. No Vulkan loader, GPU or game data.
#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
#include "../src/vulkan_replay.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#ifdef _WIN32
#include <malloc.h>
#endif
#include <new>
#include <vector>

static std::size_t g_allocations = 0;

void* operator new(std::size_t size) {
    ++g_allocations;
    if (void* p = std::malloc(size ? size : 1)) return p;
    throw std::bad_alloc();
}
void* operator new[](std::size_t size) { return operator new(size); }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }
// The aligned forms, so that an allocation of an over-aligned type is counted also.
void* operator new(std::size_t size, std::align_val_t alignment) {
    ++g_allocations;
    const std::size_t align = static_cast<std::size_t>(alignment);
#ifdef _WIN32
    if (void* p = _aligned_malloc(size ? size : 1, align)) return p;
#else
    if (void* p = std::aligned_alloc(align, (size + align - 1) / align * align + (size ? 0 : align))) return p;
#endif
    throw std::bad_alloc();
}
void* operator new[](std::size_t size, std::align_val_t alignment) { return operator new(size, alignment); }
void operator delete(void* p, std::align_val_t) noexcept {
#ifdef _WIN32
    _aligned_free(p);
#else
    std::free(p);
#endif
}
void operator delete[](void* p, std::align_val_t alignment) noexcept { operator delete(p, alignment); }
void operator delete(void* p, std::size_t, std::align_val_t alignment) noexcept { operator delete(p, alignment); }
void operator delete[](void* p, std::size_t, std::align_val_t alignment) noexcept { operator delete(p, alignment); }

namespace {

using namespace leonida;

// The barriers of every vkCmdPipelineBarrier2 call, and the number of calls. The storage is reserved before the
// measured emit, so that the record itself does not allocate there.
std::vector<VkImageMemoryBarrier2> g_barriers;
std::uint32_t g_calls = 0;
std::uint32_t g_memory = 0;

void VKAPI_CALL record_barrier2(VkCommandBuffer, const VkDependencyInfo* info) {
    ++g_calls;
    g_memory += info->memoryBarrierCount;
    for (std::uint32_t i = 0; i < info->imageMemoryBarrierCount; ++i)
        g_barriers.push_back(info->pImageMemoryBarriers[i]);
}

// What zero-filling an image that was never used calls: its own layout changes and the clear. Counted, not recorded.
std::uint32_t g_fill_calls = 0;

void VKAPI_CALL count_barrier(VkCommandBuffer, VkPipelineStageFlags, VkPipelineStageFlags, VkDependencyFlags,
                              std::uint32_t, const VkMemoryBarrier*, std::uint32_t, const VkBufferMemoryBarrier*,
                              std::uint32_t, const VkImageMemoryBarrier*) {
    ++g_fill_calls;
}

void VKAPI_CALL count_clear(VkCommandBuffer, VkImage, VkImageLayout, const VkClearColorValue*, std::uint32_t,
                            const VkImageSubresourceRange*) {
    ++g_fill_calls;
}

int failures = 0;
void expect(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}

// Run a stencil-only round trip with coupled depth/stencil layouts. Vulkan transitions both aspects
// together on this device, but the data transfer must never read or write the sibling depth plane.
void depth_uav_barriers(VkCommandBuffer cb) {
    namespace y = d3d12_sync;
    namespace a = d3d12_access;
    namespace l = d3d12_layout;
    std::array<ResourceBinding, 4> resources{};
    auto& parent = resources[1];
    parent.aspects = depth_stencil_aspects;
    parent.format = VK_FORMAT_D32_SFLOAT_S8_UINT;
    parent.depth_uav = 2;
    parent.stencil_uav = 3;
    resources[2].format = VK_FORMAT_R32_SFLOAT;
    resources[3].format = VK_FORMAT_R8_UINT;
    LayoutTracker tracker;
    for (ResourceId i = 1; i <= 3; ++i) {
        auto& r = resources[i];
        r.image = reinterpret_cast<VkImage>(std::uintptr_t(i));
        r.width = 5;
        r.height = 7;
        tracker.track(i, r, VK_IMAGE_LAYOUT_GENERAL);
    }
    const auto resolve = [&](ResourceId id) -> const ResourceBinding* {
        return id && id < resources.size() ? &resources[id] : nullptr;
    };
    struct Access { VkImage image; VkImageAspectFlags aspect; bool read; std::uint32_t layer, layers; };
    static std::vector<Access> accesses;
    accesses.clear();
    ReplayFunctions vk{};
    vk.pipeline_barrier2 = record_barrier2;
    vk.pipeline_barrier = [](VkCommandBuffer, VkPipelineStageFlags, VkPipelineStageFlags, VkDependencyFlags,
                             std::uint32_t, const VkMemoryBarrier*, std::uint32_t, const VkBufferMemoryBarrier*,
                             std::uint32_t, const VkImageMemoryBarrier*) {};
    vk.copy_image_to_buffer = [](VkCommandBuffer, VkImage image, VkImageLayout, VkBuffer, std::uint32_t,
                                 const VkBufferImageCopy* region) {
        const auto& sub = region->imageSubresource;
        accesses.push_back({image, sub.aspectMask, true, sub.baseArrayLayer, sub.layerCount});
    };
    vk.copy_buffer_to_image = [](VkCommandBuffer, VkBuffer, VkImage image, VkImageLayout, std::uint32_t,
                                 const VkBufferImageCopy* region) {
        const auto& sub = region->imageSubresource;
        accesses.push_back({image, sub.aspectMask, false, sub.baseArrayLayer, sub.layerCount});
    };
    const BarrierSubresourceRange stencil{1, 0, 0, 0, 0, 0};
    const BarrierSubresourceRange all_planes{all_subresources, 0, 0, 0, 0, 0};
    g_barriers.clear();
    g_memory = 0;
    CommandList list;
    expect(list.barrier(TextureBarrier{y::depth_stencil, y::all, a::depth_stencil_write, a::common,
                                       l::undefined, l::undefined, 1, all_planes, 0}) &&
           list.barrier(TextureBarrier{y::depth_stencil, y::compute_shading, a::depth_stencil_write, a::unordered_access,
                                       l::depth_stencil_write, l::unordered_access, 1, stencil, 0}) &&
           list.barrier(TextureBarrier{y::compute_shading, y::all, a::unordered_access, a::shader_resource,
                                       l::unordered_access, l::generic_read, 1, stencil, 0}) &&
           list.barrier(TextureBarrier{y::all, y::copy, a::shader_resource, a::copy_source,
                                       l::generic_read, l::common, 1, stencil, 0}) && list.close(),
           "record DSV to stencil UAV then copy source");
    std::uint32_t allocations = 0;
    vk.copy_scratch = [&](VkDeviceSize size, VkBuffer& out, VkDeviceSize& offset) {
        expect(size == parent.width * parent.height, "stencil scratch holds one byte per texel");
        ++allocations;
        out = reinterpret_cast<VkBuffer>(std::uintptr_t(0x10));
        offset = 0;
        return true;
    };
    PreparedList prepared;
    vk.graphics_family = false;
    expect(prepare(list, resolve, vk, prepared) == ReplayResult::unsupported && !allocations,
           "reject depth/stencil transfers on a non-graphics family before scratch allocation");
    vk.graphics_family = true;
    // A global memory barrier in a queue-specific COMMON layout can switch accesses from a shader write
    // on the child to a copy or sample on the parent without a texture layout transition.
    for (const auto layout : {l::direct_queue_common, l::compute_queue_common}) {
        CommandList implicit;
        expect(implicit.barrier(TextureBarrier{y::copy, y::compute_shading, a::copy_dest, a::unordered_access,
                                               l::copy_dest, layout, 1, all_planes, 0}) &&
               implicit.barrier(GlobalBarrier{y::compute_shading, y::copy, a::unordered_access, a::copy_source}) &&
               implicit.close(), "record queue-specific COMMON and a global dependency");
        expect(prepare(implicit, resolve, vk, prepared) == ReplayResult::unsupported && !allocations,
               "reject ambiguous queue-specific COMMON at prepare");
    }
    for (bool before : {false, true}) {
        CommandList invalid;
        expect(invalid.barrier(TextureBarrier{y::all, y::all,
                   before ? a::unordered_access : a::copy_source, before ? a::copy_source : a::unordered_access,
                   before ? l::common : l::copy_source, before ? l::copy_source : l::common,
                   1, all_planes, 0}) && invalid.close(), "record explicit UAV access in generic COMMON");
        expect(prepare(invalid, resolve, vk, prepared) == ReplayResult::unsupported && !allocations,
               "generic COMMON disallows explicit UAV access on either side of a barrier");
    }
    if (prepare(list, resolve, vk, prepared) != ReplayResult::success) {
        expect(false, "prepare stencil transfers");
        return;
    }
    emit(prepared, cb, vk, &tracker);
    expect(g_memory == 1 && !g_barriers.empty() && g_barriers.front().oldLayout == VK_IMAGE_LAYOUT_GENERAL &&
           std::none_of(g_barriers.begin(), g_barriers.end(), [](const auto& b) { return b.newLayout == VK_IMAGE_LAYOUT_UNDEFINED; }),
           "UNDEFINED to UNDEFINED only orders memory and preserves both data and tracked layouts");
    expect(allocations == 1 && accesses.size() == 4, "reuse one staging range across both stencil transitions");
    if (accesses.size() == 4) {
        expect(accesses[0].read && accesses[0].image == parent.image && accesses[0].aspect == VK_IMAGE_ASPECT_STENCIL_BIT &&
               !accesses[1].read && accesses[1].image == resources[3].image && accesses[1].aspect == VK_IMAGE_ASPECT_COLOR_BIT &&
               accesses[2].read && accesses[2].image == resources[3].image && accesses[2].aspect == VK_IMAGE_ASPECT_COLOR_BIT &&
               !accesses[3].read && accesses[3].image == parent.image && accesses[3].aspect == VK_IMAGE_ASPECT_STENCIL_BIT,
               "stencil only moves through the R8 storage image in both directions");
    }
    expect(tracker.layout(1, 0) == VK_IMAGE_LAYOUT_GENERAL &&
           tracker.layout(1, 1) == VK_IMAGE_LAYOUT_GENERAL &&
           tracker.layout(3, 0) == VK_IMAGE_LAYOUT_GENERAL, "coupled parent and independently tracked storage layouts");
    vk.copy_scratch = [](VkDeviceSize, VkBuffer&, VkDeviceSize&) { return false; };
    expect(prepare(list, resolve, vk, prepared) == ReplayResult::unsupported, "scratch failure refuses the list");

    // Select layers 1 and 2 (out of 4). The depth-only array has no stencil child. Check the staging size,
    // image copy ranges, and the independent initial layouts of the two selected storage layers.
    parent.aspects = VK_IMAGE_ASPECT_DEPTH_BIT;
    parent.format = VK_FORMAT_D16_UNORM;
    parent.stencil_uav = 0;
    resources[2].format = VK_FORMAT_R16_UINT;
    LayoutTracker arrays;
    for (ResourceId id : {1u, 2u}) {
        resources[id].array_layers = 4;
        arrays.track(id, resources[id], VK_IMAGE_LAYOUT_GENERAL);
    }
    arrays.set(2, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    arrays.set(2, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 2, 1}, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    const BarrierSubresourceRange two_layers{0, 1, 1, 2, 0, 1};
    CommandList compute;
    expect(compute.barrier(TextureBarrier{y::all, y::compute_shading, a::common, a::unordered_access,
                                          l::common, l::unordered_access, 1, two_layers, 0}) &&
           compute.barrier(TextureBarrier{y::compute_shading, y::all, a::unordered_access, a::common,
                                          l::unordered_access, l::common, 1, two_layers, 0}) && compute.close(),
           "record D16 array compute handoff");
    vk.graphics_family = false;
    vk.queue_stages = detail::queue_stages_compute;
    vk.copy_scratch = [&](VkDeviceSize size, VkBuffer& buffer, VkDeviceSize& offset) {
        expect(size == 2 * 2 * parent.width * parent.height, "D16 scratch holds two bytes per texel and two layers");
        buffer = reinterpret_cast<VkBuffer>(std::uintptr_t(0x10));
        offset = 0;
        return true;
    };
    vk.compute_copy_aspects = [](VkFormat) -> VkImageAspectFlags { return VK_IMAGE_ASPECT_STENCIL_BIT; };
    expect(prepare(compute, resolve, vk, prepared) == ReplayResult::unsupported,
           "stencil-only compute copy capability does not authorize a depth transfer");
    vk.compute_copy_aspects = [](VkFormat fmt) -> VkImageAspectFlags {
        return fmt == VK_FORMAT_D16_UNORM ? VK_IMAGE_ASPECT_DEPTH_BIT : 0;
    };
    if (prepare(compute, resolve, vk, prepared) != ReplayResult::success) {
        expect(false, "prepare supported D16 compute transfers");
        return;
    }
    accesses.clear();
    emit(prepared, cb, vk, &arrays);
    expect(accesses.size() == 4 && std::all_of(accesses.begin(), accesses.end(), [](const Access& access) {
               return access.layer == 1 && access.layers == 2;
           }), "both images are read and written only at the requested layers");
    expect(arrays.layout(2, 0) == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL &&
           arrays.layout(2, 1) == VK_IMAGE_LAYOUT_GENERAL && arrays.layout(2, 2) == VK_IMAGE_LAYOUT_GENERAL,
           "selected storage layers end in their descriptor layout and the untouched layer keeps its layout");
    // vkCmdClearDepthStencilImage requires graphics even with maintenance10. A committed depth array first
    // seen on compute is initialized from the zero page through a buffer-to-image transfer.
    vk.zero_buffer = reinterpret_cast<VkBuffer>(std::uintptr_t(0x11));
    vk.zero_size = 1024;
    const VkImageSubresourceRange fresh{VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 1, 2};
    arrays.set(1, fresh, VK_IMAGE_LAYOUT_UNDEFINED);
    auto pieces = detail::current(&arrays, 1, fresh, VK_IMAGE_LAYOUT_GENERAL);
    accesses.clear();
    detail::initialize(vk, cb, parent, 1, pieces, VK_IMAGE_LAYOUT_GENERAL, &arrays);
    expect(accesses.size() == 1 && !accesses[0].read && accesses[0].image == parent.image &&
           accesses[0].aspect == VK_IMAGE_ASPECT_DEPTH_BIT && accesses[0].layer == 1 && accesses[0].layers == 2,
           "first use on compute copies zeros into the selected native depth layers");
}

}  // namespace

int main() {
    namespace y = d3d12_sync;
    namespace a = d3d12_access;
    namespace l = d3d12_layout;
    constexpr ResourceId id = 1;
    constexpr std::uint32_t mips = 3;
    constexpr VkImageLayout shader_read = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    constexpr VkImageLayout transfer_src = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    constexpr VkImageLayout transfer_dst = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    constexpr VkImageLayout attachment = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    ResourceBinding image{};
    image.image = reinterpret_cast<VkImage>(std::uintptr_t(0x1000));
    image.format = VK_FORMAT_R8G8B8A8_UNORM;
    image.aspects = VK_IMAGE_ASPECT_COLOR_BIT;
    image.width = image.height = 64;
    image.mip_levels = mips;
    // A second image that nothing has used yet: the tracker has it in UNDEFINED.
    constexpr ResourceId unused_id = 2;
    ResourceBinding unused = image;
    unused.image = reinterpret_cast<VkImage>(std::uintptr_t(0x2000));
    const auto resolve = [&](ResourceId r) { return r == id ? &image : r == unused_id ? &unused : nullptr; };

    ReplayFunctions vk{};
    vk.pipeline_barrier2 = record_barrier2;
    LayoutTracker tracker;
    tracker.track(id, image, shader_read);
    const VkCommandBuffer cb = reinterpret_cast<VkCommandBuffer>(std::uintptr_t(0x10));
    const BarrierSubresourceRange all{all_subresources, 0, 0, 0, 0, 0};
    g_barriers.reserve(16);

    // 1. All mips in one layout: one barrier, and emit allocates nothing.
    {
        CommandList list;
        expect(list.barrier(TextureBarrier{y::pixel_shading, y::copy, a::shader_resource, a::copy_dest,
                                           l::shader_resource, l::copy_dest, id, all, 0}) &&
                   list.close(),
               "record a barrier over one layout");
        PreparedList prepared;
        expect(prepare(list, resolve, vk, prepared) == ReplayResult::success, "prepare a barrier over one layout");
        const std::size_t before = g_allocations;
        emit(prepared, cb, vk, &tracker);
        const std::size_t allocations = g_allocations - before;
        if (allocations)
            std::fprintf(stderr, "emit made %zu heap allocations for a barrier with one piece\n", allocations);
        expect(allocations == 0, "emit allocates nothing for a barrier with one piece");
        expect(g_calls == 1 && g_barriers.size() == 1, "one call with one image barrier");
        if (g_barriers.size() == 1) {
            const VkImageMemoryBarrier2& b = g_barriers[0];
            expect(b.sType == VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 && b.pNext == nullptr, "barrier header");
            expect(b.image == image.image && b.oldLayout == shader_read && b.newLayout == transfer_dst,
                   "image and layouts of the one barrier");
            expect(b.subresourceRange.baseMipLevel == 0 && b.subresourceRange.aspectMask == VK_IMAGE_ASPECT_COLOR_BIT,
                   "range of the one barrier");
            expect(b.srcQueueFamilyIndex == VK_QUEUE_FAMILY_IGNORED && b.dstQueueFamilyIndex == VK_QUEUE_FAMILY_IGNORED,
                   "no queue family transfer");
            expect(b.srcStageMask != 0 && b.dstStageMask != 0 && b.srcAccessMask != 0 && b.dstAccessMask != 0,
                   "stages and access of the one barrier");
        }
        for (std::uint32_t mip = 0; mip < mips; ++mip)
            expect(tracker.layout(id, mip) == transfer_dst, "the tracker has the new layout");
    }

    // 2. Three mips in three layouts: three barriers in one call, each from the layout of its mip.
    {
        const VkImageLayout old_layouts[mips] = {shader_read, transfer_src, transfer_dst};
        for (std::uint32_t mip = 0; mip < mips; ++mip)
            tracker.set(id, {VK_IMAGE_ASPECT_COLOR_BIT, mip, 1, 0, 1}, old_layouts[mip]);
        g_barriers.clear();
        g_calls = 0;
        CommandList list;
        expect(list.barrier(TextureBarrier{y::pixel_shading, y::render_target, a::shader_resource, a::render_target,
                                           l::shader_resource, l::render_target, id, all, 0}) &&
                   list.close(),
               "record a barrier over three layouts");
        PreparedList prepared;
        expect(prepare(list, resolve, vk, prepared) == ReplayResult::success, "prepare a barrier over three layouts");
        emit(prepared, cb, vk, &tracker);
        expect(g_calls == 1 && g_barriers.size() == mips, "one call with three image barriers");
        for (std::uint32_t mip = 0; mip < mips && g_barriers.size() == mips; ++mip) {
            const VkImageMemoryBarrier2& b = g_barriers[mip];
            expect(b.sType == VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 && b.pNext == nullptr &&
                       b.image == image.image,
                   "header and image of each barrier");
            expect(b.oldLayout == old_layouts[mip] && b.newLayout == attachment, "layouts of each barrier");
            expect(b.subresourceRange.baseMipLevel == mip && b.subresourceRange.levelCount == 1 &&
                       b.subresourceRange.baseArrayLayer == 0 && b.subresourceRange.layerCount == 1,
                   "range of each barrier");
            expect(b.srcStageMask == g_barriers[0].srcStageMask && b.dstStageMask == g_barriers[0].dstStageMask &&
                       b.srcAccessMask == g_barriers[0].srcAccessMask &&
                       b.dstAccessMask == g_barriers[0].dstAccessMask && b.srcStageMask != 0 && b.dstStageMask != 0,
                   "stages and access of each barrier");
            expect(tracker.layout(id, mip) == attachment, "the tracker has the new layout for each mip");
        }
    }

    // 3. Into UNDEFINED from a defined layout (the retail title ends a depth buffer's use this way before its memory
    //    is aliased, #984): Vulkan has no transition into UNDEFINED, so each mip keeps the layout it is in and the
    //    barrier only orders the accesses. The tracker keeps those layouts.
    const auto kept = [&](std::uint32_t mip) { return mip == 1 ? transfer_src : attachment; };
    {
        for (std::uint32_t mip = 0; mip < mips; ++mip)
            tracker.set(id, {VK_IMAGE_ASPECT_COLOR_BIT, mip, 1, 0, 1}, kept(mip));
        g_barriers.clear();
        g_calls = 0;
        CommandList list;
        expect(list.barrier(TextureBarrier{y::render_target, y::none, a::render_target, a::no_access,
                                           l::render_target, l::undefined, id, all, 0}) &&
                   list.close(),
               "record a barrier into UNDEFINED");
        PreparedList prepared;
        expect(prepare(list, resolve, vk, prepared) == ReplayResult::success, "prepare a barrier into UNDEFINED");
        emit(prepared, cb, vk, &tracker);
        expect(g_calls == 1 && g_barriers.size() == mips, "into UNDEFINED: one call with a barrier for each mip");
        for (const VkImageMemoryBarrier2& b : g_barriers) {
            expect(b.newLayout != VK_IMAGE_LAYOUT_UNDEFINED, "no transition into VK_IMAGE_LAYOUT_UNDEFINED");
            expect(b.oldLayout == b.newLayout && b.oldLayout == kept(b.subresourceRange.baseMipLevel),
                   "into UNDEFINED: each mip keeps its own layout");
            expect(b.srcStageMask != 0 && b.srcAccessMask != 0, "into UNDEFINED: the earlier accesses are ordered");
        }
        for (std::uint32_t mip = 0; mip < mips; ++mip)
            expect(tracker.layout(id, mip) == kept(mip), "into UNDEFINED: the tracker keeps the layout");
    }

    // 4. The discard flag changes nothing there: a release has nothing to discard into.
    {
        g_barriers.clear();
        g_calls = 0;
        CommandList list;
        expect(list.barrier(TextureBarrier{y::render_target, y::none, a::render_target, a::no_access,
                                           l::render_target, l::undefined, id, all, texture_barrier_discard}) &&
                   list.close(),
               "record a barrier into UNDEFINED with the discard flag");
        PreparedList prepared;
        expect(prepare(list, resolve, vk, prepared) == ReplayResult::success,
               "prepare a barrier into UNDEFINED with the discard flag");
        emit(prepared, cb, vk, &tracker);
        expect(g_calls == 1 && g_barriers.size() == mips, "into UNDEFINED with discard: a barrier for each mip");
        for (const VkImageMemoryBarrier2& b : g_barriers)
            expect(b.oldLayout == b.newLayout && b.oldLayout == kept(b.subresourceRange.baseMipLevel),
                   "into UNDEFINED with discard: each mip keeps its own layout");
        for (std::uint32_t mip = 0; mip < mips; ++mip)
            expect(tracker.layout(id, mip) == kept(mip), "into UNDEFINED with discard: the tracker keeps the layout");
    }

    // 5. Without a tracker the layout kept is the one the barrier names.
    {
        g_barriers.clear();
        g_calls = 0;
        CommandList list;
        expect(list.barrier(TextureBarrier{y::render_target, y::none, a::render_target, a::no_access,
                                           l::render_target, l::undefined, id, all, 0}) &&
                   list.close(),
               "record a barrier into UNDEFINED for an untracked replay");
        PreparedList prepared;
        expect(prepare(list, resolve, vk, prepared) == ReplayResult::success,
               "prepare a barrier into UNDEFINED for an untracked replay");
        emit(prepared, cb, vk, nullptr);
        expect(g_calls == 1 && g_barriers.size() == 1 && g_barriers[0].oldLayout == attachment &&
                   g_barriers[0].newLayout == attachment,
               "into UNDEFINED without a tracker: the named layout is kept");
    }

    // 6. An image nothing has used is released without a Vulkan call: there is no access to order, and it is not
    //    zero-filled first, as it would be on its way into any other layout. It stays in UNDEFINED.
    {
        tracker.track(unused_id, unused, VK_IMAGE_LAYOUT_UNDEFINED);
        vk.pipeline_barrier = count_barrier;
        vk.clear_color_image = count_clear;
        g_barriers.clear();
        g_calls = 0;
        CommandList list;
        expect(list.barrier(TextureBarrier{y::render_target, y::none, a::render_target, a::no_access,
                                           l::render_target, l::undefined, unused_id, all, 0}) &&
                   list.close(),
               "record a barrier into UNDEFINED for an unused image");
        PreparedList prepared;
        expect(prepare(list, resolve, vk, prepared) == ReplayResult::success,
               "prepare a barrier into UNDEFINED for an unused image");
        emit(prepared, cb, vk, &tracker);
        expect(g_calls == 0 && g_barriers.empty(), "an unused image is released without a barrier");
        expect(g_fill_calls == 0, "an unused image is not zero-filled to be released");
        for (std::uint32_t mip = 0; mip < mips; ++mip)
            expect(tracker.layout(unused_id, mip) == VK_IMAGE_LAYOUT_UNDEFINED, "an unused image stays in UNDEFINED");
        vk.pipeline_barrier = nullptr;
        vk.clear_color_image = nullptr;
    }

    // 7. After a release D3D12 has the next barrier start from UNDEFINED: one barrier that discards from whatever
    //    layouts the mips are in (case 3 left them in two), and the tracker has the new layout.
    {
        g_barriers.clear();
        g_calls = 0;
        CommandList list;
        expect(list.barrier(TextureBarrier{y::none, y::render_target, a::no_access, a::render_target, l::undefined,
                                           l::render_target, id, all, 0}) &&
                   list.close(),
               "record the barrier that follows a release");
        PreparedList prepared;
        expect(prepare(list, resolve, vk, prepared) == ReplayResult::success,
               "prepare the barrier that follows a release");
        emit(prepared, cb, vk, &tracker);
        expect(g_calls == 1 && g_barriers.size() == 1 && g_barriers[0].oldLayout == VK_IMAGE_LAYOUT_UNDEFINED &&
                   g_barriers[0].newLayout == attachment,
               "after a release: one barrier from UNDEFINED into the new layout");
        for (std::uint32_t mip = 0; mip < mips; ++mip)
            expect(tracker.layout(id, mip) == attachment, "after a release: the tracker has the new layout");
    }

    depth_uav_barriers(cb);
    if (failures) return 1;
    std::printf("replay_texture_barrier_test: OK\n");
    return 0;
}
