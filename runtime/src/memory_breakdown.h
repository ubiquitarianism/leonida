#pragma once

// The mem probe's breakdown line (src/mem_probe.h, LEONIDA_MEM_PROBE=<seconds>): the runtime's own vkAllocateMemory
// bytes split by the consumer that asked for them, so the parts add up to the total the probe's `mem-probe t=` line
// prints per memory type. Diagnostic only: nothing here changes what is allocated or where.
//
// A consumer is named in one of two ways:
//   - by the caller: an Owner scope around the call that allocates (the arena, the geometry copies' and title
//     windows' 4 MiB blocks, upload heaps, staging, the texture blocks). The probe's vkAllocateMemory wrapper reads
//     the scope of the thread it runs on;
//   - by what is bound to it: memory nobody named that a dedicated image is bound to is a render target or depth
//     buffer when the image is an attachment, else a sampled texture (image_consumer()).
// Memory named neither way is "other", so the parts always sum to the total.
//
// Pure standard library (the usage bits are passed in as plain integers), so the tally and the text are tested
// without a GPU or Windows (memory_breakdown_test).

#include <array>
#include <cstdint>
#include <cstdio>
#include <string>

namespace leonida::memory_breakdown {

enum class Consumer : std::uint8_t {
    other,          // nobody named it and no image is bound to it
    arena,          // the GPU address arena's memory (docs/d3d12_vulkan.md section 3.3)
    geometry,       // geometry copies: GeometryViews' 4 MiB blocks (shims/d3d12_geometry_views.h)
    windows,        // title windows outside the arena and slot-133 shadow buffers with their own memory
    upload,         // upload and readback heap buffers with their own memory
    staging,        // texture staging and copy scratch
    image_blocks,   // blocks that hold many textures (ImageMemory, TitleImagePool)
    sampled,        // dedicated memory of a texture that is not an attachment
    render_target,  // dedicated memory of a colour or depth/stencil attachment
    count
};

constexpr std::size_t kConsumers = std::size_t(Consumer::count);

inline const char* consumer_name(Consumer c) {
    switch (c) {
        case Consumer::other: return "other";
        case Consumer::arena: return "arena";
        case Consumer::geometry: return "geometry copies";
        case Consumer::windows: return "title windows and shadows";
        case Consumer::upload: return "upload heaps";
        case Consumer::staging: return "staging";
        case Consumer::image_blocks: return "texture blocks";
        case Consumer::sampled: return "sampled textures";
        case Consumer::render_target: return "render targets and depth";
        default: return "?";
    }
}

// VkImageUsageFlagBits values, as plain integers so this header needs no Vulkan headers.
constexpr std::uint32_t kColorAttachment = 0x10;         // VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT
constexpr std::uint32_t kDepthStencilAttachment = 0x20;  // VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT

// The consumer of memory an image of `usage` is bound to, when nobody named the memory.
inline Consumer image_consumer(std::uint32_t usage) {
    return usage & (kColorAttachment | kDepthStencilAttachment) ? Consumer::render_target : Consumer::sampled;
}

// The consumer the calling thread allocates for. Set by an Owner scope; Consumer::other outside one.
inline Consumer& current_owner() {
    thread_local Consumer owner = Consumer::other;
    return owner;
}

// Names the consumer of the allocations this thread makes while the scope lives. Scopes nest: the inner one wins
// and the outer one is back when it ends.
class Owner {
public:
    explicit Owner(Consumer consumer) : previous_(current_owner()) { current_owner() = consumer; }
    ~Owner() { current_owner() = previous_; }
    Owner(const Owner&) = delete;
    Owner& operator=(const Owner&) = delete;

private:
    Consumer previous_;
};

// Live allocations per consumer. Imported memory (host pointers, the title's own pages) is system memory the runtime
// did not allocate, so it is kept apart, as the probe's total keeps it apart.
struct Tally {
    std::array<std::uint64_t, kConsumers> bytes{}, count{};
    std::uint64_t imported_bytes = 0, imported_count = 0;

    void add(Consumer consumer, std::uint64_t size, bool imported) {
        if (imported) {
            imported_bytes += size;
            ++imported_count;
            return;
        }
        const std::size_t i = std::size_t(consumer) < kConsumers ? std::size_t(consumer) : 0;
        bytes[i] += size;
        ++count[i];
    }
    std::uint64_t total() const {
        std::uint64_t sum = 0;
        for (const std::uint64_t b : bytes) sum += b;
        return sum;
    }
};

// What the device knows about the arena and the tables outside the probe's wrapped allocations.
struct DeviceShare {
    bool arena = false;
    std::uint64_t arena_reserved = 0;  // the arena's size: the address range shaders may name
    std::uint64_t arena_bound = 0;     // bytes of memory behind it
    std::uint64_t arena_in_use = 0;    // bytes allocations hold
    // arena_in_use by kind, as DeviceState::ArenaUse: buffers, descriptor heaps, shader code, heaps
    std::array<std::uint64_t, 4> arena_use{};
    std::uint64_t bindless_table_bytes = 0;  // allocated by BindlessSet through its own function table
};

inline double mib(std::uint64_t bytes) { return double(bytes) / 1048576.0; }

// The breakdown line's text after "[rt] mem-probe breakdown t=<s>: ". Every consumer is printed, zero or not, so a
// reader can line up two ticks; the figures are whole MiB.
inline std::string format(const Tally& tally, const DeviceShare& device, std::uint64_t module_bytes) {
    std::string out;
    char item[256];
    std::snprintf(item, sizeof item, "allocated %.0f MiB =", mib(tally.total()));
    out += item;
    bool first = true;
    for (std::size_t i = 1; i <= kConsumers; ++i) {
        const std::size_t c = i % kConsumers;  // "other" last
        std::snprintf(item, sizeof item, "%s %s %.0f (%llu)", first ? "" : " +", consumer_name(Consumer(c)),
                      mib(tally.bytes[c]), (unsigned long long)tally.count[c]);
        out += item;
        first = false;
        if (Consumer(c) == Consumer::arena && device.arena) {
            std::snprintf(item, sizeof item,
                          " [reserved %.0f, bound %.0f, in use %.0f: buffers %.0f, descriptor heaps %.0f, shader code "
                          "%.0f, heaps %.0f]",
                          mib(device.arena_reserved), mib(device.arena_bound), mib(device.arena_in_use),
                          mib(device.arena_use[0]), mib(device.arena_use[1]), mib(device.arena_use[2]),
                          mib(device.arena_use[3]));
            out += item;
        }
    }
    std::snprintf(item, sizeof item,
                  "; imported %.0f MiB (%llu); not in the total: bindless tables %.0f MiB (own function table), shader "
                  "modules %.0f MiB SPIR-V, pipelines and swapchain images (the driver's)",
                  mib(tally.imported_bytes), (unsigned long long)tally.imported_count,
                  mib(device.bindless_table_bytes), mib(module_bytes));
    out += item;
    return out;
}

}  // namespace leonida::memory_breakdown
