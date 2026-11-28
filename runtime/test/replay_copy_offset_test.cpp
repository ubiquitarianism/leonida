// CopyResource (buffer-to-buffer) and CopyTexture (buffer <-> image, including depth24_as_float) apply
// ResourceBinding::buffer_offset so suballocated arena buffers read and write their own range of the backing VkBuffer.
// No Vulkan loader, GPU or game data: fake Vulkan callbacks capture the emitted copy parameters.
#include "../src/vulkan_replay.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

namespace {

std::vector<VkBufferCopy> buffer_copies;
std::vector<VkBufferImageCopy> b2i_copies;
std::vector<VkBufferImageCopy> i2b_copies;
std::vector<leonida::Depth24Job> depth24_jobs;

// Whether a barrier since the last buffer copy ordered earlier transfer writes before transfer reads and writes, and
// that answer for each vkCmdCopyBuffer in turn: D3D12 orders back-to-back copies, Vulkan needs the barrier.
bool transfer_writes_ordered = false;
std::vector<bool> buffer_copy_ordered;

VKAPI_ATTR void VKAPI_CALL rec_barrier(VkCommandBuffer, VkPipelineStageFlags src_stages, VkPipelineStageFlags dst_stages,
                                       VkDependencyFlags, std::uint32_t memory_count, const VkMemoryBarrier* memory,
                                       std::uint32_t, const VkBufferMemoryBarrier*, std::uint32_t,
                                       const VkImageMemoryBarrier*) {
    const VkAccessFlags after = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    for (std::uint32_t i = 0; i < memory_count; ++i)
        if ((src_stages & VK_PIPELINE_STAGE_TRANSFER_BIT) && (dst_stages & VK_PIPELINE_STAGE_TRANSFER_BIT) &&
            (memory[i].srcAccessMask & VK_ACCESS_TRANSFER_WRITE_BIT) && (memory[i].dstAccessMask & after) == after)
            transfer_writes_ordered = true;
}

VKAPI_ATTR void VKAPI_CALL rec_copy_buffer(VkCommandBuffer, VkBuffer, VkBuffer, std::uint32_t count,
                                           const VkBufferCopy* regions) {
    for (std::uint32_t i = 0; i < count; ++i) buffer_copies.push_back(regions[i]);
    buffer_copy_ordered.push_back(transfer_writes_ordered);
    transfer_writes_ordered = false;
}

VKAPI_ATTR void VKAPI_CALL rec_copy_b2i(VkCommandBuffer, VkBuffer, VkImage, VkImageLayout, std::uint32_t count,
                                        const VkBufferImageCopy* regions) {
    for (std::uint32_t i = 0; i < count; ++i) b2i_copies.push_back(regions[i]);
}

VKAPI_ATTR void VKAPI_CALL rec_copy_i2b(VkCommandBuffer, VkImage, VkImageLayout, VkBuffer, std::uint32_t count,
                                        const VkBufferImageCopy* regions) {
    for (std::uint32_t i = 0; i < count; ++i) i2b_copies.push_back(regions[i]);
}

std::vector<VkImageCopy> image_copies;

VKAPI_ATTR void VKAPI_CALL noop_copy_image(VkCommandBuffer, VkImage, VkImageLayout, VkImage, VkImageLayout,
                                           std::uint32_t count, const VkImageCopy* regions) {
    for (std::uint32_t i = 0; i < count; ++i) image_copies.push_back(regions[i]);
}

template<typename T>
T handle(std::uintptr_t n) {
    return reinterpret_cast<T>(n);
}

int failures = 0;
void expect(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}

}  // namespace

int main(int argc, char** argv) {
    using namespace leonida;
    const bool expect_ordered = !(argc == 2 && std::strcmp(argv[1], "--expect-unordered") == 0);

    ReplayFunctions vk{};
    vk.pipeline_barrier = rec_barrier;
    vk.copy_buffer = rec_copy_buffer;
    vk.copy_buffer_to_image = rec_copy_b2i;
    vk.copy_image_to_buffer = rec_copy_i2b;
    vk.copy_image = noop_copy_image;
    vk.depth24_prepare = [](Depth24Job& job, std::vector<Depth24Job>& parts) {
        depth24_jobs.push_back(job);
        const std::uint64_t aw = std::uint64_t(job.width) * job.app_texel_words();
        const std::uint64_t span =
            ((std::uint64_t(job.slices) - 1) * job.app_slice_words +
             (std::uint64_t(job.height) - 1) * job.app_row_words + aw) * 4;
        if (job.app_offset % 4 || job.app_offset > job.app_size || span > job.app_size - job.app_offset) return false;
        job.scratch = handle<VkBuffer>(0x9000);
        job.scratch_offset = 0;
        parts.assign(1, job);
        return true;
    };
    vk.depth24_run = [](VkCommandBuffer, const Depth24Job&) {};

    ResourceBinding src_buf{};
    src_buf.buffer = handle<VkBuffer>(1);
    src_buf.size = 4096;
    src_buf.buffer_offset = (UINT64_C(1) << 33) + 512;

    ResourceBinding dst_buf{};
    dst_buf.buffer = handle<VkBuffer>(2);
    dst_buf.size = 4096;
    dst_buf.buffer_offset = (UINT64_C(1) << 34) + 1024;
    dst_buf.host_readable = true;

    ResourceBinding rgba_img{};
    rgba_img.image = handle<VkImage>(3);
    rgba_img.format = VK_FORMAT_R8G8B8A8_UNORM;
    rgba_img.aspects = VK_IMAGE_ASPECT_COLOR_BIT;
    rgba_img.width = 16;
    rgba_img.height = 16;
    rgba_img.mip_levels = 1;
    rgba_img.array_layers = 1;
    rgba_img.depth = 1;
    rgba_img.samples = 1;

    ResourceBinding d24_img{};
    d24_img.image = handle<VkImage>(4);
    d24_img.format = VK_FORMAT_D32_SFLOAT_S8_UINT;
    d24_img.aspects = VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
    d24_img.width = 8;
    d24_img.height = 4;
    d24_img.mip_levels = 1;
    d24_img.array_layers = 1;
    d24_img.depth = 1;
    d24_img.samples = 1;
    d24_img.depth24 = true;
    d24_img.depth24_as_float = true;

    ResourceBinding rgb96_img{};
    rgb96_img.image = handle<VkImage>(5);
    rgb96_img.format = VK_FORMAT_R32G32B32A32_UINT;
    rgb96_img.aspects = VK_IMAGE_ASPECT_COLOR_BIT;
    rgb96_img.width = 17;
    rgb96_img.height = 3;
    rgb96_img.mip_levels = 1;
    rgb96_img.array_layers = 1;
    rgb96_img.depth = 1;
    rgb96_img.samples = 1;
    rgb96_img.rgb96 = true;

    // A depth-stencil texture and a depth-only one of the same size, both 32-bit float depth (D32_FLOAT_S8X24 and
    // D32_FLOAT in D3D12).
    ResourceBinding d32s8_img{};
    d32s8_img.image = handle<VkImage>(6);
    d32s8_img.format = VK_FORMAT_D32_SFLOAT_S8_UINT;
    d32s8_img.aspects = VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
    d32s8_img.width = 64;
    d32s8_img.height = 32;
    d32s8_img.mip_levels = 1;
    d32s8_img.array_layers = 1;
    d32s8_img.depth = 1;
    d32s8_img.samples = 1;
    ResourceBinding d32_img = d32s8_img;
    d32_img.image = handle<VkImage>(7);
    d32_img.format = VK_FORMAT_D32_SFLOAT;
    d32_img.aspects = VK_IMAGE_ASPECT_DEPTH_BIT;
    ResourceBinding d32_img2 = d32_img;
    d32_img2.image = handle<VkImage>(8);

    const auto resolve = [&](ResourceId id) -> const ResourceBinding* {
        switch (id) {
        case 1: return &src_buf;
        case 2: return &dst_buf;
        case 3: return &rgba_img;
        case 4: return &d24_img;
        case 5: return &rgb96_img;
        case 6: return &d32s8_img;
        case 7: return &d32_img;
        case 8: return &d32_img2;
        default: return nullptr;
        }
    };

    const auto run = [&](const CommandList& list) {
        buffer_copies.clear();
        buffer_copy_ordered.clear();
        transfer_writes_ordered = false;
        b2i_copies.clear();
        i2b_copies.clear();
        image_copies.clear();
        depth24_jobs.clear();
        return replay(list, VK_NULL_HANDLE, resolve, vk);
    };

    // 1. CopyResource between distinct backing VkBuffers with 64-bit offsets.
    {
        CommandList list;
        expect(list.copy_resource({2, 1}) && list.close(), "record CopyResource");
        expect(run(list) == ReplayResult::success, "CopyResource distinct buffers replays");
        expect(buffer_copies.size() == 1 && buffer_copies[0].srcOffset == src_buf.buffer_offset &&
                   buffer_copies[0].dstOffset == dst_buf.buffer_offset && buffer_copies[0].size == src_buf.size,
               "CopyResource applies src and dst buffer_offset");
        expect(buffer_copy_ordered == std::vector<bool>{expect_ordered},
               "a buffer copy is ordered after earlier transfer writes (an earlier list on the queue may have written)");
    }

    // 1b. Back-to-back copies into the same buffer, with no barrier from the application: D3D12 runs them in order,
    // so each vkCmdCopyBuffer needs its own barrier after the transfer write before it (one barrier ahead of the
    // first copy would leave the second unordered against the first).
    {
        CommandList list;
        expect(list.copy_buffer({2, 0, 1, 0, 256}) && list.copy_buffer({2, 128, 1, 512, 256}) && list.close(),
               "record two overlapping CopyBufferRegion calls");
        expect(run(list) == ReplayResult::success, "back-to-back buffer copies replay");
        expect(buffer_copies.size() == 2, "both buffer copies are emitted");
        expect(buffer_copy_ordered == (std::vector<bool>{expect_ordered, expect_ordered}),
               "each back-to-back buffer copy is ordered after the transfer write before it");
    }

    // 2. CopyResource within a single shared backing VkBuffer (arena): disjoint & adjacent succeed, overlap fails.
    {
        const auto saved_src = src_buf, saved_dst = dst_buf;
        dst_buf.buffer = src_buf.buffer;
        src_buf.size = dst_buf.size = 256;
        src_buf.buffer_offset = 512;
        dst_buf.buffer_offset = 1024;
        CommandList list;
        expect(list.copy_resource({2, 1}) && list.close(), "record arena CopyResource");
        expect(run(list) == ReplayResult::success, "CopyResource disjoint arena slices replays");
        expect(buffer_copies.size() == 1 && buffer_copies[0].srcOffset == 512 && buffer_copies[0].dstOffset == 1024 &&
                   buffer_copies[0].size == 256,
               "CopyResource arena uses slice offsets");

        dst_buf.buffer_offset = 768;  // immediately adjacent [512, 768) and [768, 1024)
        expect(run(list) == ReplayResult::success, "CopyResource adjacent arena slices replays");
        expect(buffer_copies.size() == 1 && buffer_copies[0].srcOffset == 512 && buffer_copies[0].dstOffset == 768,
               "CopyResource adjacent arena slices keep exact offsets");

        dst_buf.buffer_offset = 640;  // overlaps [512, 768)
        expect(run(list) == ReplayResult::invalid_copy, "CopyResource rejects overlapping arena slices");
        dst_buf.buffer_offset = 512;  // identical offset in same VkBuffer
        expect(run(list) == ReplayResult::invalid_copy, "CopyResource rejects aliased arena slices");

        // Zero-size buffer copy and buffer_offset + size overflow must be rejected cleanly.
        src_buf.size = dst_buf.size = 0;
        dst_buf.buffer = handle<VkBuffer>(2);
        expect(run(list) == ReplayResult::invalid_copy, "CopyResource rejects zero-size buffers");

        src_buf.size = dst_buf.size = 64;
        src_buf.buffer_offset = std::numeric_limits<VkDeviceSize>::max() - 32;
        dst_buf.buffer_offset = 0;
        expect(run(list) == ReplayResult::invalid_copy, "CopyResource rejects src buffer_offset overflow");

        src_buf.buffer_offset = 0;
        dst_buf.buffer_offset = std::numeric_limits<VkDeviceSize>::max() - 32;
        expect(run(list) == ReplayResult::invalid_copy, "CopyResource rejects dst buffer_offset overflow");
        src_buf = saved_src;
        dst_buf = saved_dst;
    }

    // 3. CopyTexture (ordinary RGBA8 upload and readback) adds buffer_offset to region.bufferOffset.
    {
        CommandList up;
        const CopyLocation fp_up{1, true, 0, 512, 8, 4, 1, 256};
        const CopyLocation tex{3, false, 0, 0, 0, 0, 0, 0};
        const CopyBox box{2, 1, 0, 6, 3, 1};
        expect(up.copy_texture({tex, 1, 2, 0, fp_up, true, box}) && up.close(), "record RGBA8 upload");
        expect(run(up) == ReplayResult::success, "RGBA8 upload with buffer_offset replays");
        const VkDeviceSize expected_up = src_buf.buffer_offset + 512 + 1 * 256 + 2 * 4;
        expect(b2i_copies.size() == 1 && b2i_copies[0].bufferOffset == expected_up,
               "RGBA8 upload applies src buffer_offset");

        CommandList rb;
        const CopyLocation fp_rb{2, true, 0, 1024, 8, 4, 1, 256};
        const CopyBox rb_box{0, 0, 0, 4, 2, 1};
        expect(rb.copy_texture({fp_rb, 1, 2, 0, tex, true, rb_box}) && rb.close(), "record RGBA8 readback");
        expect(run(rb) == ReplayResult::success, "RGBA8 readback with buffer_offset replays");
        const VkDeviceSize expected_rb = dst_buf.buffer_offset + 1024 + 2 * 256 + 1 * 4;
        expect(i2b_copies.size() == 1 && i2b_copies[0].bufferOffset == expected_rb,
               "RGBA8 readback applies dst buffer_offset");

        // Misaligned backing buffer_offset (2 % 4 != 0) and buffer_offset + size overflow must fail.
        const auto saved_src = src_buf;
        src_buf.buffer_offset = 2;
        CommandList misaligned_backing;
        const CopyLocation fp_zero{1, true, 0, 0, 4, 2, 1, 256};
        expect(misaligned_backing.copy_texture({tex, 0, 0, 0, fp_zero, false, {}}) && misaligned_backing.close(),
               "record misaligned backing upload");
        expect(run(misaligned_backing) == ReplayResult::unsupported,
               "CopyTexture rejects misaligned backing buffer_offset");

        // Resource-relative stencil offset misaligned (1 % 4 != 0) even when buffer_offset + offset is 4-byte aligned.
        src_buf.buffer_offset = 3;
        CommandList misaligned_rel;
        const CopyLocation stencil_tex{4, false, 1, 0, 0, 0, 0, 0};  // plane 1 = stencil (1 byte/texel, align 4)
        const CopyLocation fp_stencil{1, true, 0, 0, 8, 2, 1, 256};
        const CopyBox stencil_box{1, 0, 0, 5, 2, 1};
        expect(misaligned_rel.copy_texture({stencil_tex, 0, 0, 0, fp_stencil, true, stencil_box}) &&
                   misaligned_rel.close(),
               "record misaligned relative stencil upload");
        expect(run(misaligned_rel) == ReplayResult::unsupported,
               "CopyTexture rejects misaligned relative footprint offset");

        src_buf.buffer_offset = std::numeric_limits<VkDeviceSize>::max() - 128;
        expect(run(misaligned_backing) == ReplayResult::invalid_copy,
               "CopyTexture rejects buffer_offset + size overflow");
        src_buf = saved_src;
    }

    // A +4 host prefix is legal for buffer transfers but not an 8-byte RGBA16 texel.
    // Stage both directions, preserving each footprint's row padding and the host prefix.
    {
        const auto saved_src = src_buf, saved_dst = dst_buf, saved_image = rgba_img;
        src_buf.buffer_offset = dst_buf.buffer_offset = 4;
        src_buf.size = dst_buf.size = 512;
        rgba_img.format = VK_FORMAT_R16G16B16A16_UNORM;
        rgba_img.width = 1;
        rgba_img.height = 2;
        vk.copy_scratch = [](VkDeviceSize bytes, VkBuffer& buffer, VkDeviceSize& offset) {
            expect(bytes == 16, "scratch contains only two occupied texels");
            buffer = handle<VkBuffer>(0x8000);
            offset = 64;
            return true;
        };
        const CopyLocation tex{3, false, 0, 0, 0, 0, 0, 0};
        const CopyLocation source{1, true, 0, 0, 1, 2, 1, 256};
        const CopyLocation target{2, true, 0, 0, 1, 2, 1, 256};
        CommandList up, down;
        expect(up.copy_texture({tex, 0, 0, 0, source, false, {}}) && up.close(), "record +4 RGBA16 upload");
        expect(down.copy_texture({target, 0, 0, 0, tex, false, {}}) && down.close(), "record +4 RGBA16 readback");
        expect(run(up) == ReplayResult::success, "+4 RGBA16 upload stages");
        expect(buffer_copies.size() == 2 && buffer_copies[0].srcOffset == 4 &&
                   buffer_copies[1].srcOffset == 260 && buffer_copies[0].dstOffset == 64 &&
                   buffer_copies[1].dstOffset == 72 && buffer_copies[0].size == 8 && buffer_copies[1].size == 8,
               "upload stages only texels, excluding prefix and row padding");
        expect(b2i_copies.size() == 1 && b2i_copies[0].bufferOffset == 64 &&
                   b2i_copies[0].bufferRowLength == 0, "image upload uses aligned packed scratch");
        expect(run(down) == ReplayResult::success, "+4 RGBA16 readback stages");
        expect(buffer_copies.size() == 2 && buffer_copies[0].dstOffset == 4 &&
                   buffer_copies[1].dstOffset == 260 && buffer_copies[0].srcOffset == 64 &&
                   buffer_copies[1].srcOffset == 72 && buffer_copies[0].size == 8 && buffer_copies[1].size == 8,
               "readback leaves host prefix, padding and tail untouched");
        expect(i2b_copies.size() == 1 && i2b_copies[0].bufferOffset == 64,
               "image readback uses aligned scratch");
        vk.copy_scratch = {};
        expect(run(up) == ReplayResult::unsupported, "unaligned import fails safely without scratch");
        rgba_img.format = VK_FORMAT_R8_UNORM;
        src_buf.buffer_offset = 1;
        expect(run(up) == ReplayResult::success, "byte-aligned R8 import retains the direct path");
        src_buf = saved_src;
        dst_buf = saved_dst;
        rgba_img = saved_image;
    }

    // 4. CopyTexture on depth24_as_float passes buffer_offset + offset and buffer_offset + size to Depth24Converter.
    {
        const auto saved_src = src_buf, saved_dst = dst_buf;
        src_buf.buffer_offset = 65536;
        src_buf.size = 1536;
        dst_buf.buffer_offset = 131072 + 4;  // 4-byte aligned, non-256-aligned
        dst_buf.size = 1536;

        const CopyLocation d24_tex{4, false, 0, 0, 0, 0, 0, 0};
        const CopyLocation fp_up{1, true, 0, 512, 8, 2, 1, 256};
        CommandList up;
        expect(up.copy_texture({d24_tex, 0, 0, 0, fp_up, false, {}}) && up.close(), "record depth24 upload");
        expect(run(up) == ReplayResult::success, "depth24 upload with buffer_offset replays");
        expect(depth24_jobs.size() == 1 && depth24_jobs[0].to_float &&
                   depth24_jobs[0].app_offset == src_buf.buffer_offset + 512 &&
                   depth24_jobs[0].app_size == src_buf.buffer_offset + src_buf.size,
               "depth24 upload passes backing offset and backing end size");

        const CopyLocation fp_rb{2, true, 0, 512, 8, 2, 1, 256};
        const CopyBox rb_box{0, 0, 0, 8, 2, 1};
        CommandList rb;
        expect(rb.copy_texture({fp_rb, 0, 0, 0, d24_tex, true, rb_box}) && rb.close(), "record depth24 readback");
        expect(run(rb) == ReplayResult::success, "depth24 readback with buffer_offset replays");
        expect(depth24_jobs.size() == 1 && !depth24_jobs[0].to_float &&
                   depth24_jobs[0].app_offset == dst_buf.buffer_offset + 512 &&
                   depth24_jobs[0].app_size == dst_buf.buffer_offset + dst_buf.size,
               "depth24 readback passes backing offset and backing end size");

        src_buf.buffer_offset = 2;  // violates 4-byte depth alignment
        expect(run(up) == ReplayResult::unsupported, "depth24 upload rejects misaligned backing buffer_offset");
        src_buf = saved_src;
        dst_buf = saved_dst;
    }

    // 5. CopyTexture on rgb96 passes buffer_offset + offset and buffer_offset + size to Depth24Converter.
    {
        const auto saved_src = src_buf, saved_dst = dst_buf;
        src_buf.buffer_offset = 2048;
        src_buf.size = 1536;
        dst_buf.buffer_offset = 4096 + 4;  // 4-byte aligned, non-256-aligned
        dst_buf.size = 1536;

        const CopyLocation rgb96_tex{5, false, 0, 0, 0, 0, 0, 0};
        const CopyLocation fp_up{1, true, 0, 512, 17, 3, 1, 256};
        CommandList up;
        expect(up.copy_texture({rgb96_tex, 0, 0, 0, fp_up, false, {}}) && up.close(), "record rgb96 upload");
        expect(run(up) == ReplayResult::success, "rgb96 upload with buffer_offset replays");
        expect(depth24_jobs.size() == 1 && depth24_jobs[0].rgb96 && depth24_jobs[0].to_float &&
                   depth24_jobs[0].app_offset == src_buf.buffer_offset + 512 &&
                   depth24_jobs[0].app_size == src_buf.buffer_offset + src_buf.size &&
                   depth24_jobs[0].app_row_words == 64,
               "rgb96 upload passes backing offset and backing end size");

        const CopyLocation fp_rb{2, true, 0, 512, 17, 3, 1, 256};
        CommandList rb;
        expect(rb.copy_texture({fp_rb, 0, 0, 0, rgb96_tex, false, {}}) && rb.close(), "record rgb96 readback");
        expect(run(rb) == ReplayResult::success, "rgb96 readback with buffer_offset replays");
        expect(depth24_jobs.size() == 1 && depth24_jobs[0].rgb96 && !depth24_jobs[0].to_float &&
                   depth24_jobs[0].app_offset == dst_buf.buffer_offset + 512 &&
                   depth24_jobs[0].app_size == dst_buf.buffer_offset + dst_buf.size &&
                   depth24_jobs[0].app_row_words == 64,
               "rgb96 readback passes backing offset and backing end size");

        src_buf.buffer_offset = 2;  // violates 4-byte rgb96 alignment
        expect(run(up) == ReplayResult::unsupported, "rgb96 upload rejects misaligned backing buffer_offset");
        src_buf = saved_src;
        dst_buf = saved_dst;
    }

    // 6. tile_footprint unaligned staging honors format alignment (8B, 16B, BC1) and 3D slice layout.
    {
        const auto saved_src = src_buf, saved_dst = dst_buf;
        VkDeviceSize scratch_req = 0;
        vk.copy_scratch = [&](std::uint64_t bytes, VkBuffer& buf, VkDeviceSize& off) {
            scratch_req = bytes;
            buf = handle<VkBuffer>(0xA000);
            off = 256;
            return true;
        };

        ResourceBinding rgba16f_img{};
        rgba16f_img.image = handle<VkImage>(6);
        rgba16f_img.format = VK_FORMAT_R16G16B16A16_SFLOAT;  // 8 bytes/texel
        rgba16f_img.aspects = VK_IMAGE_ASPECT_COLOR_BIT;
        rgba16f_img.width = 8;
        rgba16f_img.height = 4;
        rgba16f_img.mip_levels = 1;
        rgba16f_img.array_layers = 1;
        rgba16f_img.depth = 2;
        rgba16f_img.volume = true;
        rgba16f_img.samples = 1;

        ResourceBinding bc1_img{};
        bc1_img.image = handle<VkImage>(7);
        bc1_img.format = VK_FORMAT_BC1_RGBA_UNORM_BLOCK;  // 8 bytes per 4x4 block
        bc1_img.aspects = VK_IMAGE_ASPECT_COLOR_BIT;
        bc1_img.width = 12;
        bc1_img.height = 8;
        bc1_img.mip_levels = 1;
        bc1_img.array_layers = 1;
        bc1_img.depth = 1;
        bc1_img.samples = 1;

        const auto resolve_tile = [&](ResourceId id) -> const ResourceBinding* {
            if (id == 6) return &rgba16f_img;
            if (id == 7) return &bc1_img;
            return resolve(id);
        };

        // 6a. 8-byte 3D texture with bufferOffset = 4 (4-byte aligned, NOT 8-byte aligned): must stage both slices.
        src_buf.buffer_offset = 0;
        src_buf.size = 4096;
        {
            const CopyLocation tex{6, false, 0, 0, 0, 0, 0, 0};
            // footprint: width=8, height=4, depth=2, row_pitch=256 (32 texels/row, slice_pitch=1024), offset=4
            const CopyLocation fp{1, true, 0, 4, 8, 4, 2, 256};
            CopyTexture copy{tex, 0, 0, 0, fp, true, {0, 0, 0, 4, 2, 2}};  // 4x2x2 texels -> 32B/row, 2 rows, 2 slices = 128B
            copy.tile_footprint = true;
            CommandList list;
            expect(list.copy_texture(copy) && list.close(), "record 8B 3D tile_footprint upload");
            buffer_copies.clear();
            b2i_copies.clear();
            scratch_req = 0;
            expect(replay(list, VK_NULL_HANDLE, resolve_tile, vk) == ReplayResult::success,
                   "8B 3D tile_footprint upload with offset=4 replays");
            expect(scratch_req == 128, "8B 3D tile_footprint allocates 4*8 * 2 * 2 = 128 bytes of scratch");
            expect(buffer_copies.size() == 4, "8B 3D tile_footprint stages 2 rows x 2 slices");
            expect(buffer_copies.size() == 4 &&
                       buffer_copies[0].srcOffset == 4 && buffer_copies[0].dstOffset == 256 && buffer_copies[0].size == 32 &&
                       buffer_copies[1].srcOffset == 260 && buffer_copies[1].dstOffset == 288 && buffer_copies[1].size == 32 &&
                       buffer_copies[2].srcOffset == 1028 && buffer_copies[2].dstOffset == 320 && buffer_copies[2].size == 32 &&
                       buffer_copies[3].srcOffset == 1284 && buffer_copies[3].dstOffset == 352 && buffer_copies[3].size == 32,
                   "8B 3D tile_footprint row/slice strides match 256B row_pitch and 1024B slice_pitch");
            expect(b2i_copies.size() == 1 && b2i_copies[0].bufferOffset == 256 &&
                       b2i_copies[0].bufferRowLength == 0 && b2i_copies[0].bufferImageHeight == 0,
                   "8B 3D tile_footprint emits aligned scratch offset and clears row/image height");
        }

        // 6b. BC1 (8B per 4x4 block) with bufferOffset = 4: must stage 2x2 blocks = 32B, not texels*4.
        {
            const CopyLocation tex{7, false, 0, 0, 0, 0, 0, 0};
            // footprint: width=12 (3 blocks), height=8 (2 blocks), row_pitch=256 (32*8B), offset=4
            const CopyLocation fp{1, true, 0, 4, 12, 8, 1, 256};
            CopyTexture copy{tex, 0, 0, 0, fp, true, {0, 0, 0, 8, 8, 1}};  // 2x2 blocks: 16B/row, 2 rows = 32B
            copy.tile_footprint = true;
            CommandList list;
            expect(list.copy_texture(copy) && list.close(), "record BC1 tile_footprint upload");
            buffer_copies.clear();
            b2i_copies.clear();
            scratch_req = 0;
            expect(replay(list, VK_NULL_HANDLE, resolve_tile, vk) == ReplayResult::success,
                   "BC1 tile_footprint upload with offset=4 replays");
            expect(scratch_req == 32, "BC1 tile_footprint allocates 2*8 * 2 = 32 bytes of scratch");
            expect(buffer_copies.size() == 2 &&
                       buffer_copies[0].srcOffset == 4 && buffer_copies[0].dstOffset == 256 && buffer_copies[0].size == 16 &&
                       buffer_copies[1].srcOffset == 260 && buffer_copies[1].dstOffset == 272 && buffer_copies[1].size == 16,
                   "BC1 tile_footprint copies 16B block rows at 256B pitch");
            expect(b2i_copies.size() == 1 && b2i_copies[0].bufferOffset == 256 && b2i_copies[0].bufferRowLength == 0,
                   "BC1 tile_footprint emits aligned scratch offset");
        }

        vk.copy_scratch = {};
        src_buf = saved_src;
        dst_buf = saved_dst;
    }

    // 7. CopyTextureRegion of the depth plane between two depth formats with the same encoding but not the same
    // VkFormat (D32_SFLOAT_S8_UINT and D32_SFLOAT): vkCmdCopyImage requires depth/stencil formats to match exactly
    // (VUID-vkCmdCopyImage-srcImage-01548), and an NVIDIA driver handed the D32S8 -> D32 pair wrote past the end of the
    // D32 image (the world-start WRITE_INVALID). The copy goes image -> transfer scratch -> image, depth aspect on
    // both sides, 4 bytes a texel. The same format on both sides keeps vkCmdCopyImage.
    {
        VkDeviceSize scratch_bytes = 0;
        vk.copy_scratch = [&](VkDeviceSize bytes, VkBuffer& out, VkDeviceSize& base) {
            scratch_bytes = bytes;
            out = handle<VkBuffer>(0x9100);
            base = 256;
            return true;
        };
        const auto staged_depth = [&](ResourceId dst, ResourceId src, const char* what) {
            const CopyLocation to{dst, false, 0, 0, 0, 0, 0, 0}, from{src, false, 0, 0, 0, 0, 0, 0};
            CommandList list;
            expect(list.copy_texture({to, 0, 0, 0, from, false, {}}) && list.close(), what);
            expect(run(list) == ReplayResult::success, what);
            expect(image_copies.empty(), "a depth copy between different depth formats does not use vkCmdCopyImage");
            expect(i2b_copies.size() == 1 && b2i_copies.size() == 1, "the depth plane is staged through a buffer");
            if (i2b_copies.size() == 1 && b2i_copies.size() == 1) {
                const VkBufferImageCopy &read = i2b_copies[0], &write = b2i_copies[0];
                expect(read.imageSubresource.aspectMask == VK_IMAGE_ASPECT_DEPTH_BIT &&
                           write.imageSubresource.aspectMask == VK_IMAGE_ASPECT_DEPTH_BIT,
                       "both staged halves name the depth aspect");
                expect(read.imageExtent.width == 64 && read.imageExtent.height == 32 && read.imageExtent.depth == 1 &&
                           write.imageExtent.width == 64 && write.imageExtent.height == 32,
                       "the staged depth copy covers the whole 64x32 plane");
                expect(read.bufferOffset == 256 && write.bufferOffset == 256,
                       "the read and the write share one scratch range");
            }
            expect(scratch_bytes >= VkDeviceSize(64) * 32 * 4, "the scratch holds 4 bytes a depth texel");
        };
        staged_depth(7, 6, "D32_SFLOAT_S8_UINT depth plane into D32_SFLOAT");
        staged_depth(6, 7, "D32_SFLOAT into the depth plane of D32_SFLOAT_S8_UINT");

        const CopyLocation to{8, false, 0, 0, 0, 0, 0, 0}, from{7, false, 0, 0, 0, 0, 0, 0};
        CommandList same;
        expect(same.copy_texture({to, 0, 0, 0, from, false, {}}) && same.close(), "record D32_SFLOAT copy");
        expect(run(same) == ReplayResult::success, "D32_SFLOAT into D32_SFLOAT replays");
        expect(image_copies.size() == 1 && i2b_copies.empty() && b2i_copies.empty(),
               "the same depth format on both sides keeps vkCmdCopyImage");
        vk.copy_scratch = {};
    }

    if (failures) return 1;
    std::printf("replay_copy_offset: CopyResource and CopyTexture apply ResourceBinding::buffer_offset\n");
    return 0;
}
