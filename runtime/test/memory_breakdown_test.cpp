// leonida::memory_breakdown (src/memory_breakdown.h): the mem probe's breakdown line. The tally of synthetic
// allocations by consumer adds up to the allocated total, an Owner scope names the allocations made under it and
// nests, an image's usage picks its kind, and the text names every part. No GPU, no device.
#include "../src/memory_breakdown.h"

#include <cstdint>
#include <cstdio>
#include <string>
#include <thread>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

namespace {

namespace mb = leonida::memory_breakdown;
constexpr std::uint64_t kMiB = 1 << 20;

bool contains(const std::string& text, const char* part) { return text.find(part) != std::string::npos; }

// The parts add up to the total, imported memory stays out of it, and an unknown consumer counts as other.
int tally() {
    mb::Tally t;
    t.add(mb::Consumer::arena, 4095 * kMiB, false);
    t.add(mb::Consumer::geometry, 4 * kMiB, false);
    t.add(mb::Consumer::geometry, 4 * kMiB, false);
    t.add(mb::Consumer::sampled, 3 * kMiB, false);
    t.add(mb::Consumer::render_target, 8 * kMiB, false);
    t.add(mb::Consumer::other, 1 * kMiB, false);
    t.add(mb::Consumer::count, 2 * kMiB, false);  // out of range: other
    t.add(mb::Consumer::windows, 6 * kMiB, true);  // imported: not allocated by us
    EXPECT(t.total() == (4095 + 8 + 3 + 8 + 1 + 2) * kMiB);
    EXPECT(t.bytes[std::size_t(mb::Consumer::geometry)] == 8 * kMiB);
    EXPECT(t.count[std::size_t(mb::Consumer::geometry)] == 2);
    EXPECT(t.bytes[std::size_t(mb::Consumer::other)] == 3 * kMiB);
    EXPECT(t.count[std::size_t(mb::Consumer::other)] == 2);
    EXPECT(t.bytes[std::size_t(mb::Consumer::windows)] == 0);
    EXPECT(t.imported_bytes == 6 * kMiB && t.imported_count == 1);
    std::uint64_t sum = 0;
    for (std::size_t i = 0; i < mb::kConsumers; ++i) sum += t.bytes[i];
    EXPECT(sum == t.total());
    return 0;
}

// An Owner scope names this thread's allocations, nests, restores the outer name, and does not leak to other threads.
int owner_scopes() {
    EXPECT(mb::current_owner() == mb::Consumer::other);
    {
        mb::Owner outer(mb::Consumer::geometry);
        EXPECT(mb::current_owner() == mb::Consumer::geometry);
        {
            mb::Owner inner(mb::Consumer::staging);
            EXPECT(mb::current_owner() == mb::Consumer::staging);
            mb::Consumer seen = mb::Consumer::count;
            std::thread([&] { seen = mb::current_owner(); }).join();
            EXPECT(seen == mb::Consumer::other);
        }
        EXPECT(mb::current_owner() == mb::Consumer::geometry);
    }
    EXPECT(mb::current_owner() == mb::Consumer::other);
    return 0;
}

// Colour and depth/stencil attachments are render targets; anything else bound alone is a sampled texture.
int image_kinds() {
    const std::uint32_t sampled = 0x4, transfer_dst = 0x2, storage = 0x8;
    EXPECT(mb::image_consumer(sampled | transfer_dst) == mb::Consumer::sampled);
    EXPECT(mb::image_consumer(storage) == mb::Consumer::sampled);
    EXPECT(mb::image_consumer(sampled | mb::kColorAttachment) == mb::Consumer::render_target);
    EXPECT(mb::image_consumer(mb::kDepthStencilAttachment) == mb::Consumer::render_target);
    return 0;
}

// The text: the total first, every consumer with its MiB and count, the arena's reserved/bound/in-use bytes, other
// last, and what lies outside the total.
int text() {
    mb::Tally t;
    t.add(mb::Consumer::arena, 4095 * kMiB, false);
    t.add(mb::Consumer::geometry, 3867 * kMiB, false);
    t.add(mb::Consumer::render_target, 1180 * kMiB, false);
    t.add(mb::Consumer::other, 819 * kMiB, false);
    t.add(mb::Consumer::other, 206 * kMiB, true);
    mb::DeviceShare share;
    share.arena = true;
    share.arena_reserved = share.arena_bound = 4095 * kMiB;
    share.arena_in_use = 757 * kMiB;
    share.arena_use = {500 * kMiB, 171 * kMiB, 80 * kMiB, 6 * kMiB};
    share.bindless_table_bytes = 2 * kMiB;
    const std::string line = mb::format(t, share, 12 * kMiB);
    std::fprintf(stderr, "%s\n", line.c_str());
    EXPECT(line.rfind("allocated 9961 MiB = arena 4095 (1) [reserved 4095, bound 4095, in use 757: buffers 500, "
                      "descriptor heaps 171, shader code 80, heaps 6]",
                      0) == 0);
    EXPECT(contains(line, " + geometry copies 3867 (1)"));
    EXPECT(contains(line, " + title windows and shadows 0 (0)"));
    EXPECT(contains(line, " + upload heaps 0 (0) + staging 0 (0) + texture blocks 0 (0) + sampled textures 0 (0)"));
    EXPECT(contains(line, " + render targets and depth 1180 (1) + other 819 (1);"));
    EXPECT(contains(line, "imported 206 MiB (1)"));
    EXPECT(contains(line, "bindless tables 2 MiB"));
    EXPECT(contains(line, "shader modules 12 MiB SPIR-V"));
    // Without an arena the bracket is left out.
    const std::string bare = mb::format(t, mb::DeviceShare{}, 0);
    EXPECT(!contains(bare, "reserved"));
    EXPECT(contains(bare, "arena 4095 (1) + geometry copies"));
    return 0;
}

}  // namespace

int main() {
    if (int r = tally()) return r;
    if (int r = owner_scopes()) return r;
    if (int r = image_kinds()) return r;
    if (int r = text()) return r;
    std::fprintf(stderr, "memory_breakdown: ok\n");
    return 0;
}
