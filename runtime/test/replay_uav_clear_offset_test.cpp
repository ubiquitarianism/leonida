// A buffer UAV clear on a resource that is a range of a larger VkBuffer (ResourceBinding::buffer_offset, as the GPU
// address arena places every buffer) writes that range, not the start of the VkBuffer. UavView::offset is relative to
// the resource, as copies and atomic copies already treat it. No Vulkan loader, GPU or game data: the transfer
// functions are fakes that record the ranges they are given.
#include "../src/vulkan_replay.h"
#include <array>
#include <bit>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {
struct Write {
    char kind;  // 'f' fill, 'u' update, 'c' copy (dst; src in from)
    VkDeviceSize offset, size, from;
    VkBuffer buffer = VK_NULL_HANDLE, source = VK_NULL_HANDLE;
    std::uint32_t word = 0;
};
std::vector<Write> writes;

unsigned image_clears = 0, attachment_clears = 0;
VkFormat requested_view_format = VK_FORMAT_UNDEFINED;
VkImageSubresourceRange requested_view_range{};
VkClearColorValue captured_color{};
VkClearRect captured_rect{};
VkRect2D captured_render_area{};
std::uint32_t captured_render_layers = 0;

void reset_texture_capture() {
    image_clears = attachment_clears = 0;
    requested_view_format = VK_FORMAT_UNDEFINED;
    requested_view_range = {};
    captured_color = {};
    captured_rect = {};
    captured_render_area = {};
    captured_render_layers = 0;
}

VKAPI_ATTR void VKAPI_CALL barrier(VkCommandBuffer, VkPipelineStageFlags, VkPipelineStageFlags, VkDependencyFlags,
                                   std::uint32_t, const VkMemoryBarrier*, std::uint32_t, const VkBufferMemoryBarrier*,
                                   std::uint32_t, const VkImageMemoryBarrier*) {}
VKAPI_ATTR void VKAPI_CALL fill(VkCommandBuffer, VkBuffer buffer, VkDeviceSize offset, VkDeviceSize size, std::uint32_t word) {
    writes.push_back({'f', offset, size, 0, buffer, VK_NULL_HANDLE, word});
}
VKAPI_ATTR void VKAPI_CALL update(VkCommandBuffer, VkBuffer, VkDeviceSize offset, VkDeviceSize size, const void*) {
    writes.push_back({'u', offset, size, 0});
}
VKAPI_ATTR void VKAPI_CALL copy(VkCommandBuffer, VkBuffer source, VkBuffer buffer, std::uint32_t count, const VkBufferCopy* r) {
    for (std::uint32_t i = 0; i < count; ++i) writes.push_back({'c', r[i].dstOffset, r[i].size, r[i].srcOffset, buffer, source});
}
VKAPI_ATTR void VKAPI_CALL clear_image(VkCommandBuffer, VkImage, VkImageLayout, const VkClearColorValue* value,
                                       std::uint32_t, const VkImageSubresourceRange*) {
    captured_color = *value;
    ++image_clears;
}
VKAPI_ATTR void VKAPI_CALL begin_rendering(VkCommandBuffer, const VkRenderingInfo* info) {
    captured_render_area = info->renderArea;
    captured_render_layers = info->layerCount;
}
VKAPI_ATTR void VKAPI_CALL end_rendering(VkCommandBuffer) {}
VKAPI_ATTR void VKAPI_CALL clear_attachments(VkCommandBuffer, std::uint32_t count, const VkClearAttachment* values,
                                             std::uint32_t rect_count, const VkClearRect* rects) {
    if (count == 1 && rect_count >= 1) {
        captured_color = values[0].clearValue.color;
        captured_rect = rects[0];
        ++attachment_clears;
    }
}

int failures = 0;
void expect(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}

// Every byte written (and read by a copy) lies in [begin, end) of the VkBuffer.
bool inside(VkDeviceSize begin, VkDeviceSize end) {
    if (writes.empty()) return false;
    for (const Write& w : writes) {
        if (w.offset < begin || w.offset + w.size > end) return false;
        if (w.kind == 'c' && (w.from < begin || w.from + w.size > end)) return false;
    }
    return true;
}

// UAV clears of B5G6R5, B5G5R5A1 and B4G4R4A4 textures. The packed words are what ClearUnorderedAccessViewFloat/Uint
// leave on WARP for the same values (no channel sits at a rounding midpoint).
void packed_16bit_texture_clears() {
    using namespace leonida;
    struct Case {
        const char* name;
        VkFormat format;
        std::array<std::uint32_t, 4> max;  // of R, G, B, A; 0: no such channel
        std::uint32_t red, green, blue, alpha, mixed, uint_word;
    };
    constexpr Case cases[] = {
        {"B5G6R5", VK_FORMAT_R5G6B5_UNORM_PACK16, {31, 63, 31, 0}, 0xf800, 0x07e0, 0x001f, 0x0000, 0x45ec, 0x0843},
        {"B5G5R5A1", VK_FORMAT_A1R5G5B5_UNORM_PACK16, {31, 31, 31, 1}, 0x7c00, 0x03e0, 0x001f, 0x8000, 0xa2ec, 0x8443},
        {"B4G4R4A4", VK_FORMAT_A4R4G4B4_UNORM_PACK16, {15, 15, 15, 15}, 0x0f00, 0x00f0, 0x000f, 0xf000, 0x94b6, 0xf123}};
    ReplayFunctions vk{};
    vk.pipeline_barrier = barrier;
    vk.clear_color_image = clear_image;
    vk.begin_rendering = begin_rendering;
    vk.end_rendering = end_rendering;
    vk.clear_attachments = clear_attachments;
    vk.image_view = [](const ResourceBinding& b, const VkImageSubresourceRange&) {
        requested_view_format = b.format;
        return reinterpret_cast<VkImageView>(std::uintptr_t{0x2000});
    };
    const auto run = [&](VkFormat backing, bool attachment, VkFormat view, bool is_float,
                         const std::array<std::uint32_t, 4>& values) {
        reset_texture_capture();
        ResourceBinding tex{};
        tex.image = reinterpret_cast<VkImage>(std::uintptr_t{0x3000});
        tex.format = backing;
        tex.aspects = VK_IMAGE_ASPECT_COLOR_BIT;
        tex.width = tex.height = 8;
        tex.mip_levels = tex.array_layers = 1;
        tex.color_attachment = attachment;
        CommandList list;
        list.clear_unordered_access({2, false, std::uint32_t(view), 0, 0, 0, 0, 0, 1}, is_float, values, {});
        list.close();
        return replay(list, VK_NULL_HANDLE, [&](ResourceId id) { return id == 2 ? &tex : nullptr; }, vk) ==
               ReplayResult::success;
    };
    const auto check = [](bool ok, const Case& c, const char* what) {
        char text[128];
        std::snprintf(text, sizeof text, "%s UAV clear: %s", c.name, what);
        expect(ok, text);
    };
    const auto floats = [](float r, float g, float b, float a) {
        return std::array<std::uint32_t, 4>{std::bit_cast<std::uint32_t>(r), std::bit_cast<std::uint32_t>(g),
                                            std::bit_cast<std::uint32_t>(b), std::bit_cast<std::uint32_t>(a)};
    };
    for (const Case& c : cases) {
        const struct {
            const char* what;
            bool is_float;
            std::array<std::uint32_t, 4> values;
            std::uint32_t word;
        } clears[] = {{"Float red", true, floats(1, 0, 0, 0), c.red},
                      {"Float green", true, floats(0, 1, 0, 0), c.green},
                      {"Float blue", true, floats(0, 0, 1, 0), c.blue},
                      {"Float alpha", true, floats(0, 0, 0, 1), c.alpha},
                      {"Float mixed", true, floats(0.25f, 0.75f, 0.4f, 0.6f), c.mixed},
                      {"Uint", false, {1, 2, 3, 0xffffffffu}, c.uint_word}};
        for (const auto& clear : clears)  // an R16_UINT image takes the packed word itself
            check(run(VK_FORMAT_R16_UINT, false, c.format, clear.is_float, clear.values) && image_clears == 1 &&
                      attachment_clears == 0 && captured_color.uint32[0] == clear.word,
                  c, clear.what);

        // An image of the format itself takes the channels the word holds.
        const std::array<std::uint32_t, 4> raw{1, 2, 3, 0xffffffffu};
        bool channels = run(c.format, true, c.format, false, raw) && image_clears == 1 && attachment_clears == 0;
        for (unsigned i = 0; i < 4; ++i)
            channels &= c.max[i] == 0 || captured_color.float32[i] == float(raw[i] & c.max[i]) / float(c.max[i]);
        check(channels, c, "image of the same format");

        // Every packed word survives decode() and encode().
        bool round_trip = true;
        for (std::uint32_t word = 0; word < 0x10000 && round_trip; ++word) {
            uav_clear::Element element{}, again{};
            element.size = 2;
            element.bytes[0] = std::uint8_t(word);
            element.bytes[1] = std::uint8_t(word >> 8);
            VkClearColorValue color{};
            round_trip = uav_clear::decode(c.format, element, color) &&
                         uav_clear::encode(c.format, true, std::bit_cast<std::array<std::uint32_t, 4>>(color.float32), again) &&
                         again.size == 2 && again.bytes == element.bytes;
        }
        check(round_trip, c, "decode/encode round trip");
    }
    // The render-target + UAV B5G5R5A1 texture the device backs with R16_UINT clears through its A1R5G5B5 view.
    check(run(VK_FORMAT_R16_UINT, true, VK_FORMAT_A1R5G5B5_UNORM_PACK16, true, floats(1, 0, 0, 1)) &&
              attachment_clears == 1 && image_clears == 0 &&
              requested_view_format == VK_FORMAT_A1R5G5B5_UNORM_PACK16 && captured_color.float32[0] == 1.0f &&
              captured_color.float32[1] == 0.0f && captured_color.float32[2] == 0.0f && captured_color.float32[3] == 1.0f,
          cases[1], "R16_UINT-backed render target");
}
}  // namespace

int main() {
    using namespace leonida;
    packed_16bit_texture_clears();
    constexpr VkDeviceSize at = 65536, bytes = 256;
    ResourceBinding range{};
    range.buffer = reinterpret_cast<VkBuffer>(std::uintptr_t{0x1000});
    range.buffer_offset = at;  // a range at 64 KiB of a shared buffer
    range.size = bytes;
    const auto resolve = [&](ResourceId id) -> const ResourceBinding* { return id == 1 ? &range : nullptr; };
    ReplayFunctions vk{};
    vk.pipeline_barrier = barrier;
    vk.fill_buffer = fill;
    vk.update_buffer = update;
    vk.copy_buffer = copy;

    const auto run = [&](const UavView& view, std::span<const Rect> rects = {}) {
        writes.clear();
        CommandList list;
        list.clear_unordered_access(view, false, {7, 7, 7, 7}, rects);
        list.close();
        return replay(list, VK_NULL_HANDLE, resolve, vk);
    };

    // A raw view of the whole resource: one fill of [at, at + 256).
    expect(run({1, true, 0, 0, bytes, 4, 0, 0, 1}) == ReplayResult::success, "raw whole-view clear replays");
    expect(writes.size() == 1 && writes[0].kind == 'f' && writes[0].offset == at && writes[0].size == bytes,
           "raw whole-view clear fills the resource's range of the buffer");
    // A view starting 16 bytes into the resource.
    expect(run({1, true, 0, 16, 64, 4, 0, 0, 1}) == ReplayResult::success, "raw view at 16 replays");
    expect(inside(at + 16, at + 80), "raw view at 16 stays in [at + 16, at + 80)");
    // R8_UINT elements from byte 1: unaligned ends are copied from the aligned middle, all inside the view.
    expect(run({1, true, VK_FORMAT_R8_UINT, 1, 37, 1, 0, 0, 1}) == ReplayResult::success, "R8 view at 1 replays");
    expect(inside(at + 1, at + 38), "R8 view at 1 stays in [at + 1, at + 38)");
    // R32G32B32A32_UINT (16-byte elements) is written with vkCmdUpdateBuffer and doubled with copies.
    expect(run({1, true, VK_FORMAT_R32G32B32A32_UINT, 32, 128, 16, 0, 0, 1}) == ReplayResult::success,
           "RGBA32 view at 32 replays");
    expect(inside(at + 32, at + 160), "RGBA32 view at 32 stays in [at + 32, at + 160)");
    // Rectangles count elements from the view's first byte.
    const Rect rect{2, 0, 5, 1};
    expect(run({1, true, 0, 8, 64, 4, 0, 0, 1}, std::span<const Rect>(&rect, 1)) == ReplayResult::success,
           "rectangle clear replays");
    expect(inside(at + 16, at + 28), "elements 2..5 of a view at 8 are [at + 16, at + 28)");

    const auto scratch = reinterpret_cast<VkBuffer>(std::uintptr_t{0x3000});
    constexpr VkDeviceSize scratch_at = 512;
    unsigned allocations = 0;
    expect(run({1, true, VK_FORMAT_R8_UINT, 1, 1, 1, 0, 0, 1}) == ReplayResult::unsupported && writes.empty(),
           "short clear without scratch is rejected before any command is emitted");
    vk.copy_scratch = [&](VkDeviceSize size, VkBuffer& buffer, VkDeviceSize& offset) {
        expect(size == 8, "short clear requests eight scratch bytes");
        ++allocations;
        buffer = scratch;
        offset = scratch_at;
        return true;
    };
    for (VkDeviceSize offset = 0; offset < 8; ++offset) {
        for (VkDeviceSize size = 1; size <= 8; ++size) {
            allocations = 0;
            expect(run({1, true, VK_FORMAT_R8_UINT, offset, size, 1, 0, 0, 1}) == ReplayResult::success,
                   "small R8 range replays at every word phase");
            const bool short_range = size < (4 - offset % 4) % 4 + 4;
            expect(allocations == unsigned(short_range), "only ranges lacking an aligned word allocate scratch");
            if (short_range) {
                expect(writes.size() == 2 && writes[0].kind == 'f' && writes[0].buffer == scratch &&
                       writes[0].offset == scratch_at && writes[0].size == 8 &&
                       writes[1].kind == 'c' && writes[1].source == scratch && writes[1].buffer == range.buffer &&
                       writes[1].from == scratch_at && writes[1].offset == at + offset && writes[1].size == size,
                       "short clear writes exactly the requested bytes from scratch, not neighbouring resource bytes");
            } else expect(inside(at + offset, at + offset + size), "fast path stays within the range");
        }
    }
    allocations = 0;
    const std::array<Rect, 3> small_rects{{{0, 0, 1, 1}, {0, 0, 2, 1}, {3, 0, 4, 1}}};
    expect(run({1, true, VK_FORMAT_R16_UINT, 2, 8, 2, 0, 0, 1}, small_rects) == ReplayResult::success && allocations == 1,
           "overlapping short R16 rectangles share one scratch allocation per clear");
    vk.copy_scratch = [](VkDeviceSize, VkBuffer&, VkDeviceSize&) { return false; };
    expect(run({1, true, VK_FORMAT_R8_UINT, 255, 1, 1, 0, 0, 1}) == ReplayResult::unsupported && writes.empty(),
           "scratch allocation failure rejects the list without emitting commands");
    vk.copy_scratch = [&](VkDeviceSize, VkBuffer& buffer, VkDeviceSize& offset) {
        buffer = scratch; offset = 1; return true;
    };
    expect(run({1, true, VK_FORMAT_R8_UINT, 255, 1, 1, 0, 0, 1}) == ReplayResult::unsupported && writes.empty(),
           "misaligned scratch is rejected before vkCmdFillBuffer");
    const std::array<Rect, 2> empty_rects{{{-3, 0, 0, 1}, {0, 1, 1, 2}}};
    expect(run({1, true, VK_FORMAT_R8_UINT, 255, 1, 1, 0, 0, 1}, empty_rects) == ReplayResult::success && writes.empty(),
           "empty clipped rectangles need no scratch");
    range.buffer_offset = UINT64_MAX - 2;
    expect(run({1, true, VK_FORMAT_R8_UINT, 1, 2, 1, 0, 0, 1}) == ReplayResult::invalid_clear && writes.empty(),
           "backing range overflow is rejected before scratch allocation or emission");
    // Misaligned backing buffer_offset must be rejected for multi-byte elements (e.g., e > 4 violates
    // VUID-vkCmdUpdateBuffer-dstOffset-00032; e == 2/4 would byte-swap/rotate elements in vkCmdFillBuffer).
    range.buffer_offset = at + 2;
    expect(run({1, true, VK_FORMAT_R32G32B32A32_UINT, 0, 32, 16, 0, 0, 1}) == ReplayResult::unsupported && writes.empty(),
           "misaligned backing buffer_offset for 16-byte element clear is rejected before vkCmdUpdateBuffer");
    expect(run({1, true, VK_FORMAT_R32_UINT, 0, 16, 4, 0, 0, 1}) == ReplayResult::unsupported && writes.empty(),
           "misaligned backing buffer_offset for 4-byte element clear is rejected before vkCmdFillBuffer");
    range.buffer_offset = at + 1;
    expect(run({1, true, VK_FORMAT_R16_UINT, 0, 16, 2, 0, 0, 1}) == ReplayResult::unsupported && writes.empty(),
           "odd backing buffer_offset for 2-byte element clear is rejected before vkCmdFillBuffer");
    expect(run({1, true, VK_FORMAT_R8_UINT, 0, 16, 1, 0, 0, 1}) == ReplayResult::success,
           "odd backing buffer_offset for 1-byte R8 element clear still replays");
    range.buffer_offset = at;
    vk.copy_scratch = {};

    // Typeless / mutable texture UAV clears where view_format != binding.format must clear through an attachment view
    // of view_format with clear_color decoded in view_format, avoiding SFLOAT NaN round-trips (issue #1902).
    ReplayFunctions tex_vk = vk;
    tex_vk.clear_color_image = clear_image;
    tex_vk.begin_rendering = begin_rendering;
    tex_vk.end_rendering = end_rendering;
    tex_vk.clear_attachments = clear_attachments;
    tex_vk.image_view = [](const ResourceBinding& b, const VkImageSubresourceRange& r) {
        requested_view_format = b.format;
        requested_view_range = r;
        return reinterpret_cast<VkImageView>(std::uintptr_t{0x2000});
    };

    const auto run_tex = [&](const ResourceBinding& binding, const UavView& view, bool is_float,
                             std::array<std::uint32_t, 4> values, std::span<const Rect> rects = {},
                             const ReplayFunctions* custom_vk = nullptr) {
        reset_texture_capture();
        CommandList list;
        list.clear_unordered_access(view, is_float, values, rects);
        list.close();
        return replay(list, VK_NULL_HANDLE, [&](ResourceId id) { return id == 2 ? &binding : nullptr; },
                      custom_vk ? *custom_vk : tex_vk);
    };

    ResourceBinding tex{};
    tex.image = reinterpret_cast<VkImage>(std::uintptr_t{0x3000});
    tex.format = VK_FORMAT_R16G16_SFLOAT;
    tex.aspects = VK_IMAGE_ASPECT_COLOR_BIT;
    tex.width = 8;
    tex.height = 6;
    tex.mip_levels = 2;
    tex.array_layers = 4;
    tex.color_attachment = true;

    // A dedicated compute family has no rendering commands. A rectangular clear fills transfer scratch and
    // copies the clipped rectangle to the requested array slices without disturbing adjacent pixels.
    {
        auto image = tex;
        image.format = VK_FORMAT_R8_UINT;
        auto compute = tex_vk;
        compute.graphics_family = false;
        compute.queue_stages = detail::queue_stages_compute;
        compute.begin_rendering = nullptr;
        compute.image_view = {};
        VkDeviceSize requested_bytes = 0;
        compute.copy_scratch = [&](VkDeviceSize size, VkBuffer& buffer, VkDeviceSize& offset) {
            requested_bytes = size;
            buffer = scratch;
            offset = 256;
            return true;
        };
        static std::vector<VkBufferImageCopy> copies;
        compute.copy_buffer_to_image = [](VkCommandBuffer, VkBuffer, VkImage, VkImageLayout layout,
                                          std::uint32_t count, const VkBufferImageCopy* regions) {
            expect(layout == VK_IMAGE_LAYOUT_GENERAL && count == 1, "one compute clear transfer in UAV layout");
            copies.push_back(regions[0]);
        };
        const std::array<Rect, 2> rects{{{-1, 2, 3, 4}, {1, 3, 2, 4}}};
        writes.clear();
        expect(run_tex(image, {2, false, VK_FORMAT_R8_UINT, 0, 0, 0, 0, 1, 2}, false, {0x39,0,0,0},
                       rects, &compute) == ReplayResult::success, "rectangle UAV clear on compute-only family");
        expect(requested_bytes == 12 && writes.size() == 1 && writes[0].kind == 'f' && writes[0].offset == 256 &&
               writes[0].size == 12 && copies.size() == 2 && attachment_clears == 0, "one fill and two rectangle copies");
        if (copies.size() == 2) {
            const auto& r = copies[0];
            expect(r.bufferOffset == 256 && r.imageOffset.x == 0 && r.imageOffset.y == 2 &&
                   r.imageExtent.width == 3 && r.imageExtent.height == 2 &&
                   r.imageSubresource.baseArrayLayer == 1 && r.imageSubresource.layerCount == 2 &&
                   copies[1].imageOffset.x == 1 && copies[1].imageOffset.y == 3, "clipped pixels and exactly two layers");
        }
    }

    const auto fbits = [](float f) { return std::bit_cast<std::uint32_t>(f); };
    // R16G16_TYPELESS (SFLOAT) cleared as R16G16_UNORM Float(1.0, 1.0): uses R16G16_UNORM view and float32 {1, 1}.
    expect(run_tex(tex, {2, false, VK_FORMAT_R16G16_UNORM, 0, 0, 0, 0, 1, 2}, true, {fbits(1.0f), fbits(1.0f), 0, 0}) ==
               ReplayResult::success,
           "R16G16_TYPELESS -> R16G16_UNORM whole-view clear replays");
    expect(image_clears == 0 && attachment_clears == 1,
           "R16G16_TYPELESS -> R16G16_UNORM uses attachment clear, not vkCmdClearColorImage");
    expect(requested_view_format == VK_FORMAT_R16G16_UNORM,
           "R16G16_TYPELESS -> R16G16_UNORM creates image view with view_format");
    expect(requested_view_range.baseMipLevel == 0 && requested_view_range.baseArrayLayer == 1 &&
               requested_view_range.layerCount == 2 && captured_render_layers == 2,
           "reinterpreted UAV clear preserves mip and layer range");
    expect(captured_color.float32[0] == 1.0f && captured_color.float32[1] == 1.0f,
           "R16G16_UNORM clear value is decoded in view_format (not half-float NaN)");
    expect(captured_rect.rect.offset.x == 0 && captured_rect.rect.offset.y == 0 &&
               captured_rect.rect.extent.width == 8 && captured_rect.rect.extent.height == 6 &&
               captured_rect.layerCount == 2,
           "reinterpreted whole-view clear covers full mip extent and layers");

    // R16G16_TYPELESS (SFLOAT) cleared as R16G16_UINT with half-float NaN bit patterns 0xffff and 0x7fff.
    const Rect sub_rect{1, 1, 4, 3};
    expect(run_tex(tex, {2, false, VK_FORMAT_R16G16_UINT, 0, 0, 0, 1, 0, 1}, false, {0xffff, 0x7fff, 0, 0},
                   std::span<const Rect>(&sub_rect, 1)) == ReplayResult::success,
           "R16G16_TYPELESS -> R16G16_UINT rect clear replays");
    expect(image_clears == 0 && attachment_clears == 1 && requested_view_format == VK_FORMAT_R16G16_UINT,
           "R16G16_UINT rect clear uses R16G16_UINT view");
    expect(captured_color.uint32[0] == 0xffff && captured_color.uint32[1] == 0x7fff,
           "R16G16_UINT clear passes exact uint32 values");
    expect(captured_render_area.extent.width == 4 && captured_render_area.extent.height == 3 &&
               captured_rect.rect.extent.width == 3 && captured_rect.rect.extent.height == 2,
           "R16G16_UINT rect clear clips to mip 1 extent");

    // Same-format whole-view unpredicated clear stays on vkCmdClearColorImage.
    expect(run_tex(tex, {2, false, VK_FORMAT_R16G16_SFLOAT, 0, 0, 0, 0, 0, 1}, true, {fbits(0.5f), fbits(-2.0f), 0, 0}) ==
               ReplayResult::success,
           "same-format R16G16_SFLOAT whole-view clear replays");
    expect(image_clears == 1 && attachment_clears == 0, "same-format whole-view clear uses vkCmdClearColorImage");

    // R32_TYPELESS (SFLOAT) cleared as R32_UINT with sNaN bit pattern 0x7f800001.
    ResourceBinding r32 = tex;
    r32.format = VK_FORMAT_R32_SFLOAT;
    expect(run_tex(r32, {2, false, VK_FORMAT_R32_UINT, 0, 0, 0, 0, 0, 1}, false, {0x7f800001u, 0, 0, 0}) ==
               ReplayResult::success,
           "R32_TYPELESS -> R32_UINT sNaN clear replays");
    expect(image_clears == 0 && attachment_clears == 1 && requested_view_format == VK_FORMAT_R32_UINT &&
               captured_color.uint32[0] == 0x7f800001u,
           "R32_UINT clear uses R32_UINT attachment view with exact sNaN bits");

    // R8G8B8A8_TYPELESS (UNORM) cleared as R8G8B8A8_UINT and R8G8B8A8_SINT uses view_format directly.
    ResourceBinding rgba8 = tex;
    rgba8.format = VK_FORMAT_R8G8B8A8_UNORM;
    expect(run_tex(rgba8, {2, false, VK_FORMAT_R8G8B8A8_UINT, 0, 0, 0, 0, 0, 1}, false, {0xff, 0x80, 0x7f, 1}) ==
               ReplayResult::success,
           "R8G8B8A8_TYPELESS -> R8G8B8A8_UINT clear replays");
    expect(image_clears == 0 && attachment_clears == 1 && requested_view_format == VK_FORMAT_R8G8B8A8_UINT &&
               captured_color.uint32[0] == 0xff && captured_color.uint32[1] == 0x80 &&
               captured_color.uint32[2] == 0x7f && captured_color.uint32[3] == 1,
           "R8G8B8A8_UINT clear uses R8G8B8A8_UINT attachment view");

    // Emulated R9G9B9E5_SHAREDEXP (backed by R32_UINT) must NOT request an E5B9G9R9 color attachment view.
    ResourceBinding r9e5 = tex;
    r9e5.format = VK_FORMAT_R32_UINT;
    expect(run_tex(r9e5, {2, false, VK_FORMAT_E5B9G9R9_UFLOAT_PACK32, 0, 0, 0, 0, 0, 1}, true,
                   {fbits(1.0f), fbits(2.0f), fbits(4.0f), 0}) == ReplayResult::success,
           "emulated R9G9B9E5 whole-view clear replays");
    expect(image_clears == 1 && attachment_clears == 0 && captured_color.uint32[0] == 0x94010040u,
           "emulated R9G9B9E5 whole-view clear stays on R32_UINT vkCmdClearColorImage");
    expect(run_tex(r9e5, {2, false, VK_FORMAT_E5B9G9R9_UFLOAT_PACK32, 0, 0, 0, 0, 0, 1}, true,
                   {fbits(1.0f), fbits(2.0f), fbits(4.0f), 0}, std::span<const Rect>(&sub_rect, 1)) ==
               ReplayResult::success,
           "emulated R9G9B9E5 rect clear replays");
    expect(image_clears == 0 && attachment_clears == 1 && requested_view_format == VK_FORMAT_R32_UINT &&
               captured_color.uint32[0] == 0x94010040u,
           "emulated R9G9B9E5 rect clear uses R32_UINT attachment view");

    // When dynamic rendering is unavailable, reinterpretation refuses instead of corrupting via vkCmdClearColorImage.
    ReplayFunctions no_render = tex_vk;
    no_render.begin_rendering = nullptr;
    expect(run_tex(tex, {2, false, VK_FORMAT_R16G16_UNORM, 0, 0, 0, 0, 0, 1}, true, {fbits(1.0f), fbits(1.0f), 0, 0},
                   {}, &no_render) == ReplayResult::unsupported,
           "reinterpreted UAV clear without dynamic rendering returns unsupported");

    // When color_attachment_format reports a view format unsupported as an attachment, keep the backing format path.
    ReplayFunctions no_unorm_rt = tex_vk;
    no_unorm_rt.color_attachment_format = [](VkFormat f) { return f != VK_FORMAT_R16G16_UNORM; };
    expect(run_tex(tex, {2, false, VK_FORMAT_R16G16_UNORM, 0, 0, 0, 0, 0, 1}, true, {fbits(0.0f), fbits(0.0f), 0, 0},
                   {}, &no_unorm_rt) == ReplayResult::success,
           "non-attachable view format falls back to backing format");
    expect(image_clears == 1 && attachment_clears == 0,
           "non-attachable view format whole-view clear uses vkCmdClearColorImage");

    // ---- 3D volume partial-W and XY rectangle UAV clears (direct and compute queues) ----
    {
        ResourceBinding vol{};
        vol.image = reinterpret_cast<VkImage>(std::uintptr_t{0x4000});
        vol.format = VK_FORMAT_R8G8B8A8_UNORM;
        vol.aspects = VK_IMAGE_ASPECT_COLOR_BIT;
        vol.width = 8;
        vol.height = 6;
        vol.depth = 4;
        vol.mip_levels = 2;
        vol.array_layers = 1;
        vol.volume = true;
        vol.color_attachment = false;

        static std::vector<VkBufferImageCopy> vol_copies;
        VkDeviceSize last_scratch_req = 0;
        unsigned scratch_allocs = 0;
        unsigned prepared_predicates = 0;
        bool resolved_volume = false;
        bool scratch_ok = true;
        VkDeviceSize scratch_offset_out = 256;

        auto make_vol_vk = [&](bool compute_queue) {
            ReplayFunctions v = tex_vk;
            v.graphics_family = !compute_queue;
            v.queue_stages = compute_queue ? detail::queue_stages_compute : 0;
            if (compute_queue) {
                v.begin_rendering = nullptr;
                v.image_view = {};
            }
            v.begin_conditional_rendering = [](VkCommandBuffer, const VkConditionalRenderingBeginInfoEXT*) {};
            v.end_conditional_rendering = [](VkCommandBuffer) {};
            v.prepare_predicate = [&](VkBuffer source, VkDeviceSize offset, PredicateCapture& out) {
                ++prepared_predicates;
                out = {source, offset, scratch, 0, VK_NULL_HANDLE};
                return true;
            };
            v.capture_predicate = [](VkCommandBuffer, const PredicateCapture&) {};
            v.copy_scratch = [&](VkDeviceSize size, VkBuffer& buffer, VkDeviceSize& offset) {
                ++scratch_allocs;
                last_scratch_req = size;
                if (!scratch_ok) return false;
                buffer = scratch;
                offset = scratch_offset_out;
                return true;
            };
            v.copy_buffer_to_image = [](VkCommandBuffer, VkBuffer, VkImage, VkImageLayout layout,
                                        std::uint32_t count, const VkBufferImageCopy* regions) {
                expect(layout == VK_IMAGE_LAYOUT_GENERAL && count == 1, "3D clear copy uses GENERAL and count 1");
                vol_copies.push_back(regions[0]);
            };
            return v;
        };

        auto run_vol = [&](const ResourceBinding& b, const UavView& view, bool is_float,
                           std::array<std::uint32_t, 4> values, std::span<const Rect> rects,
                           const ReplayFunctions& v, bool predicate = false) {
            writes.clear();
            vol_copies.clear();
            reset_texture_capture();
            last_scratch_req = 0;
            scratch_allocs = 0;
            prepared_predicates = 0;
            resolved_volume = false;
            CommandList list;
            ResourceBinding pred_buf{};
            pred_buf.buffer = reinterpret_cast<VkBuffer>(std::uintptr_t{0x5000});
            pred_buf.size = 64;
            if (predicate) expect(list.set_predication({9, 0, d3d12_predication::equal_zero}), "record predicate");
            expect(list.clear_unordered_access(view, is_float, values, rects), "record volume clear");
            expect(list.close(), "close volume clear list");
            return replay(list, VK_NULL_HANDLE,
                          [&](ResourceId id) -> const ResourceBinding* {
                              if (id == 2) { resolved_volume = true; return &b; }
                              if (id == 9) return &pred_buf;
                              return nullptr;
                          },
                          v);
        };

        const auto direct_vol = make_vol_vk(false);
        const auto compute_vol = make_vol_vk(true);

        // Whole-mip, non-rect 3D clear stays on vkCmdClearColorImage with zero scratch.
        expect(run_vol(vol, {2, false, VK_FORMAT_R8G8B8A8_UNORM, 0, 0, 0, 0, 0, 4, true}, false, {1, 2, 3, 4}, {},
                       direct_vol) == ReplayResult::success &&
                   image_clears == 1 && scratch_allocs == 0 && vol_copies.empty(),
               "whole-mip 3D clear keeps vkCmdClearColorImage fast path");

        // Partial W slice on mip 0 (FirstWSlice=1, WSize=2) on both direct and compute queues.
        for (bool comp : {false, true}) {
            const auto& v = comp ? compute_vol : direct_vol;
            expect(run_vol(vol, {2, false, VK_FORMAT_R8G8B8A8_UNORM, 0, 0, 0, 0, 1, 2, true}, false, {1, 2, 3, 4}, {},
                           v) == ReplayResult::success,
                   comp ? "partial 3D clear replays on compute queue" : "partial 3D clear replays on direct queue");
            expect(image_clears == 0 && scratch_allocs == 1 && last_scratch_req == 8 * 6 * 4 &&
                       writes.size() == 1 && writes[0].kind == 'f' && vol_copies.size() == 2,
                   "partial 3D clear fills one 2D slice and emits 2 depth-1 copies");
            if (vol_copies.size() == 2) {
                expect(vol_copies[0].imageSubresource.mipLevel == 0 &&
                           vol_copies[0].imageSubresource.baseArrayLayer == 0 &&
                           vol_copies[0].imageSubresource.layerCount == 1 &&
                           vol_copies[0].imageOffset.z == 1 && vol_copies[0].imageExtent.depth == 1 &&
                           vol_copies[1].imageOffset.z == 2 && vol_copies[1].imageExtent.depth == 1 &&
                           vol_copies[0].bufferOffset == 256 && vol_copies[1].bufferOffset == 256,
                       "partial 3D copies target Z=1 and Z=2 on array layer 0");
            }
        }

        // Non-zero mip (mip 1 -> 4x3x2), partial W (FirstWSlice=1, WSize=1) with clipped & overlapping rects.
        const std::array<Rect, 2> overlap{{{-2, 1, 3, 4}, {1, 0, 5, 2}}};
        expect(run_vol(vol, {2, false, VK_FORMAT_R8G8B8A8_UNORM, 0, 0, 0, 1, 1, 1, true}, false, {9, 8, 7, 6},
                       overlap, direct_vol) == ReplayResult::success &&
                   vol_copies.size() == 2 && last_scratch_req == 3 * 2 * 4,
               "mip-1 partial 3D overlapping rects clip to 4x3 and emit separate copies");
        if (vol_copies.size() == 2) {
            expect(vol_copies[0].imageSubresource.mipLevel == 1 && vol_copies[0].imageOffset.x == 0 &&
                       vol_copies[0].imageOffset.y == 1 && vol_copies[0].imageOffset.z == 1 &&
                       vol_copies[0].imageExtent.width == 3 && vol_copies[0].imageExtent.height == 2 &&
                       vol_copies[1].imageOffset.x == 1 && vol_copies[1].imageOffset.y == 0 &&
                       vol_copies[1].imageOffset.z == 1 && vol_copies[1].imageExtent.width == 3 &&
                       vol_copies[1].imageExtent.height == 2,
                   "mip-1 clipped rect offsets and extents match");
        }

        // Empty clipped rect skips without allocating scratch or emitting commands.
        const Rect offscreen{10, 10, 12, 12};
        expect(run_vol(vol, {2, false, VK_FORMAT_R8G8B8A8_UNORM, 0, 0, 0, 0, 1, 2, true}, false, {1, 2, 3, 4},
                       std::span<const Rect>(&offscreen, 1), direct_vol) == ReplayResult::success &&
                   scratch_allocs == 0 && writes.empty() && vol_copies.empty(),
               "empty clipped 3D rect skips without scratch");

        // Tiny 1..3-byte payloads pad fill to 4 bytes and copy exact texel extent.
        ResourceBinding vol_r8 = vol;
        vol_r8.format = VK_FORMAT_R8_UINT;
        const Rect one_px{2, 2, 3, 3};
        for (int width : {1, 2, 3}) {
            const Rect tiny{2, 2, 2 + width, 3};
            expect(run_vol(vol_r8, {2, false, VK_FORMAT_R8_UINT, 0, 0, 0, 0, 2, 1, true}, false, {0x5a, 0, 0, 0},
                           std::span<const Rect>(&tiny, 1), direct_vol) == ReplayResult::success &&
                       last_scratch_req == 4 && writes.size() == 1 && writes[0].kind == 'f' &&
                       writes[0].word == 0x5a5a5a5au && writes[0].size == 4 && vol_copies.size() == 1 &&
                       vol_copies[0].imageExtent.width == std::uint32_t(width) &&
                       vol_copies[0].imageExtent.height == 1 && vol_copies[0].imageOffset.z == 2,
                   "1/2/3-byte R8 volume clear fills one private word and copies the exact extent");
        }
        auto vol_r16 = vol;
        vol_r16.format = VK_FORMAT_R16_UINT;
        expect(run_vol(vol_r16, {2, false, VK_FORMAT_R16_UINT, 0, 0, 0, 0, 2, 1, true}, false, {0xcdef, 0, 0, 0},
                       std::span<const Rect>(&one_px, 1), direct_vol) == ReplayResult::success &&
                   writes.size() == 1 && writes[0].word == 0xcdefcdefu && writes[0].size == 4 &&
                   vol_copies.size() == 1 && vol_copies[0].imageExtent.width == 1,
               "one R16 texel uses padded scratch without widening the copy");

        // Wide 16-byte format exceeding 64 KiB per slice tiles cleanly with single vkCmdUpdateBuffer <= 65536 B.
        ResourceBinding big_vol = vol;
        big_vol.format = VK_FORMAT_R32G32B32A32_UINT;
        big_vol.width = 128;
        big_vol.height = 64;
        big_vol.depth = 3;
        big_vol.mip_levels = 1;
        expect(run_vol(big_vol, {2, false, VK_FORMAT_R32G32B32A32_UINT, 0, 0, 0, 0, 1, 2, true}, false,
                       {1, 2, 3, 4}, {}, direct_vol) == ReplayResult::success &&
                   scratch_allocs == 1 && last_scratch_req == 65536 && writes.size() == 1 &&
                   writes[0].kind == 'u' && writes[0].size == 65536 && vol_copies.size() == 4,
               "128x64x2 RGBA32_UINT partial 3D clear tiles into 64 KiB update_buffer chunks");
        if (vol_copies.size() == 4) {
            expect(vol_copies[0].imageOffset.y == 0 && vol_copies[0].imageExtent.height == 32 &&
                       vol_copies[0].imageOffset.z == 1 && vol_copies[1].imageOffset.y == 32 &&
                       vol_copies[1].imageExtent.height == 32 && vol_copies[1].imageOffset.z == 1 &&
                       vol_copies[2].imageOffset.y == 0 && vol_copies[2].imageOffset.z == 2 &&
                       vol_copies[3].imageOffset.y == 32 && vol_copies[3].imageOffset.z == 2,
                   "tiled 3D copies cover full 128x64 across both slices");
        }

        // Emulated R9G9B9E5_SHAREDEXP on R32_UINT 3D backing replays partial W slice.
        ResourceBinding vol_e5 = vol;
        vol_e5.format = VK_FORMAT_R32_UINT;
        expect(run_vol(vol_e5, {2, false, VK_FORMAT_E5B9G9R9_UFLOAT_PACK32, 0, 0, 0, 0, 1, 2, true}, true,
                       {fbits(1.0f), fbits(2.0f), fbits(4.0f), 0}, {}, direct_vol) == ReplayResult::success &&
                   vol_copies.size() == 2 && writes.size() == 1 && writes[0].kind == 'f' && writes[0].word == 0x94010040u,
               "emulated RGB9E5 volume float clear packs RGB 1, 2, 4 into one shared-exponent word");
        expect(run_vol(vol_e5, {2, false, VK_FORMAT_E5B9G9R9_UFLOAT_PACK32, 0, 0, 0, 0, 1, 2, true}, false,
                       {1, 2, 4, 8}, {}, direct_vol) == ReplayResult::success &&
                   vol_copies.size() == 2 && writes.size() == 1 && writes[0].word == 0,
               "emulated RGB9E5 volume uint clear writes zero even for nonzero inputs");

        // Error and safety contracts: zero commands emitted on any rejection.
        // Out-of-range W slices.
        expect(run_vol(vol, {2, false, VK_FORMAT_R8G8B8A8_UNORM, 0, 0, 0, 0, 2, 3, true}, false, {1, 2, 3, 4}, {},
                       direct_vol) == ReplayResult::invalid_subresource &&
                   writes.empty() && vol_copies.empty(),
               "out-of-range 3D W slice range is rejected as invalid_subresource");
        expect(run_vol(vol, {2, false, VK_FORMAT_R8G8B8A8_UNORM, 0, 0, 0, 1, 2, 1, true}, false, {1, 2, 3, 4}, {},
                       direct_vol) == ReplayResult::invalid_subresource &&
                   writes.empty() && vol_copies.empty(),
               "out-of-range mip-1 3D FirstWSlice is rejected as invalid_subresource");
        CommandList invalid;
        expect(!invalid.clear_unordered_access({2, false, VK_FORMAT_R8G8B8A8_UNORM, 0, 0, 0, 0, 0, 0, true}, false,
                                               {1, 2, 3, 4}) && invalid.commands().empty(),
               "zero 3D WSize is rejected while recording");

        // Typeless 3D format mismatch refused (both exact_clear_format true and false on staged path).
        ResourceBinding vol_f32 = vol;
        vol_f32.format = VK_FORMAT_R32_SFLOAT;
        expect(run_vol(vol_f32, {2, false, VK_FORMAT_R32_UINT, 0, 0, 0, 0, 1, 2, true}, false, {1, 0, 0, 0}, {},
                       direct_vol) == ReplayResult::unsupported &&
                   writes.empty() && vol_copies.empty(),
               "typeless 3D format mismatch is refused");
        expect(run_vol(vol_f32, {2, false, VK_FORMAT_R32_UINT, 0, 0, 0, 0, 1, 2, false}, false, {1, 0, 0, 0}, {},
                       direct_vol) == ReplayResult::unsupported &&
                   writes.empty() && vol_copies.empty(),
               "staged 3D format mismatch without exact_clear_format is also refused");

        // Predicated partial 3D / 3D rect clear refused on BOTH direct and compute lists.
        expect(run_vol(vol, {2, false, VK_FORMAT_R8G8B8A8_UNORM, 0, 0, 0, 0, 1, 2, true}, false, {1, 2, 3, 4}, {},
                       direct_vol, true) == ReplayResult::unsupported &&
                   prepared_predicates == 1 && resolved_volume && scratch_allocs == 0 && writes.empty() && vol_copies.empty(),
               "predicated partial 3D clear is refused on direct list");
        expect(run_vol(vol, {2, false, VK_FORMAT_R8G8B8A8_UNORM, 0, 0, 0, 0, 0, 4, true}, false, {1, 2, 3, 4},
                       std::span<const Rect>(&one_px, 1), compute_vol, true) == ReplayResult::unsupported &&
                   prepared_predicates == 1 && resolved_volume && scratch_allocs == 0 && writes.empty() && vol_copies.empty(),
               "predicated 3D rect clear is refused on compute list");

        // Failed scratch allocation, misaligned offset for 16-byte / 4-byte formats, and offset overflow.
        scratch_ok = false;
        expect(run_vol(vol, {2, false, VK_FORMAT_R8G8B8A8_UNORM, 0, 0, 0, 0, 1, 2, true}, false, {1, 2, 3, 4}, {},
                       direct_vol) == ReplayResult::unsupported &&
                   writes.empty() && vol_copies.empty(),
               "3D staged clear rejects failed scratch allocation");
        scratch_ok = true;
        scratch_offset_out = 8;  // 8-byte aligned, NOT 16-byte aligned for RGBA32_UINT
        expect(run_vol(big_vol, {2, false, VK_FORMAT_R32G32B32A32_UINT, 0, 0, 0, 0, 1, 1, true}, false, {1, 2, 3, 4},
                       std::span<const Rect>(&one_px, 1), direct_vol) == ReplayResult::unsupported &&
                   writes.empty() && vol_copies.empty(),
               "16-byte 3D clear rejects 8-byte-only aligned scratch offset");
        scratch_offset_out = 2;  // not 4-byte aligned
        expect(run_vol(vol, {2, false, VK_FORMAT_R8G8B8A8_UNORM, 0, 0, 0, 0, 1, 1, true}, false, {1, 2, 3, 4}, {},
                       direct_vol) == ReplayResult::unsupported &&
                   writes.empty() && vol_copies.empty(),
               "4-byte 3D clear rejects 2-byte aligned scratch offset");
        scratch_offset_out = UINT64_MAX - 15;  // aligned to 16, but the exclusive end is unrepresentable
        expect(run_vol(big_vol, {2, false, VK_FORMAT_R32G32B32A32_UINT, 0, 0, 0, 0, 1, 1, true}, false, {1, 2, 3, 4},
                       std::span<const Rect>(&one_px, 1), direct_vol) == ReplayResult::unsupported &&
                   writes.empty() && vol_copies.empty(),
               "3D clear rejects scratch offset + fill_bytes overflow");
        scratch_offset_out = 256;

        // Missing update_buffer callback for 16-byte 3D staged clear.
        ReplayFunctions no_update = direct_vol;
        no_update.update_buffer = nullptr;
        expect(run_vol(big_vol, {2, false, VK_FORMAT_R32G32B32A32_UINT, 0, 0, 0, 0, 1, 1, true}, false, {1, 2, 3, 4},
                       std::span<const Rect>(&one_px, 1), no_update) == ReplayResult::unsupported &&
                   writes.empty() && vol_copies.empty(),
               "16-byte 3D clear without vkCmdUpdateBuffer is rejected");
    }

    if (failures) return 1;
    std::printf("replay_uav_clear_offset: buffer offsets, typeless texture view-format clears, and 3D sliced/rect clears all passed\n");
    return 0;
}
