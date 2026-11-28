// rt::d3d12::GeometryViews (runtime/shims/d3d12_geometry_views.h): device-local copies of the title's geometry, kept
// up to date from 64 KiB tiles of the title's pages that are watched for writes. A fake Vulkan device gives the copies
// host memory. No game data, no GPU.
//   1. a new view is copied whole;
//   2. a write into one tile of a 1 MiB view (16 tiles) copies that tile's 64 KiB again, not the view;
//   3. a write into a tile two views share is read once and copied into both;
//   4. two written tiles of one view are two tile copies (LEONIDA_GEOMETRY_WHOLE_VIEWS=1, whole views, fails 2-4);
//   5. a small cap refuses past its budget without re-trying while nothing changed;
//   6. retiring a range defers freeing until its GPU ticket completes, then returns budget and lets refused ranges retry;
//   7. releasing every view in a sub-4 MiB block via geometry_writes::note_released destroys the block and frees its budget;
//   8. note_released ignores ranges outside the watched tile bounds and queues ranges inside them;
//   9. overflowing the release queue (65537 disjoint ranges) collapses to one covering range and retires covered views;
//  10. reaping the last view in a standard 4 MiB block keeps it as a spare and the next view reuses it;
//  11. under a cap, an unfitting 4 MiB spare is destroyed so a larger request can succeed (or fail if still over cap);
//  12. one note_released reaches every GeometryViews instance, and a destroyed instance unsubscribes cleanly;
//  13. a submission's sequence on an idle GPU (completed == submitted): a view retired by the drain in before_execute_
//      (kAfterExecutes) is freed only after every replay begun before the retirement recorded its submissions and those
//      completed (a ticket read in the drain reaps it at once); its answers made without an alias are invalidated
//      (forget_copy); its destroyed block leaves the lock-free block list and is reported to the device;
//  14. LEONIDA_GEOMETRY_HOST picks a system-memory type (write-back for 1, write-combined for 2), never the BAR;
//  15. a retired view is copied once more from its pages when they are readable, not when they are not;
//  16. a full release queue joins touching ranges before it collapses to one covering range.
#include <windows.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <memory>
#include <vector>

#include "../shims/d3d12_geometry_views.h"
#include "../src/title_va_alias.h"

namespace {

int g_checks = 0;
int g_failures = 0;

void check(bool condition, const char* what) {
    ++g_checks;
    std::printf("%s %s\n", condition ? "PASS" : "FAIL", what);
    if (!condition) ++g_failures;
}

// The Vulkan calls TitleBufferWindows::allocate makes, over host memory.
int g_live_buffers = 0;
int g_gone_blocks = 0;  // FakeDevice::title_block_gone calls
int g_live_allocs = 0;
int g_total_allocs = 0;
struct FakeBuffer {
    VkDeviceSize size = 0;
    void* memory = nullptr;
};
VKAPI_ATTR VkResult VKAPI_CALL create_buffer(VkDevice, const VkBufferCreateInfo* info, const VkAllocationCallbacks*,
                                             VkBuffer* out) {
    auto* b = new FakeBuffer{info->size};
    ++g_live_buffers;
    *out = reinterpret_cast<VkBuffer>(b);
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL requirements(VkDevice, VkBuffer buffer, VkMemoryRequirements* out) {
    *out = {};
    out->size = reinterpret_cast<FakeBuffer*>(buffer)->size;
    out->alignment = 65536;
    out->memoryTypeBits = 0xf;  // the four types of test 14's device
}
VKAPI_ATTR VkResult VKAPI_CALL bind(VkDevice, VkBuffer buffer, VkDeviceMemory memory, VkDeviceSize) {
    reinterpret_cast<FakeBuffer*>(buffer)->memory = memory;
    return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL map(VkDevice, VkDeviceMemory memory, VkDeviceSize, VkDeviceSize, VkMemoryMapFlags,
                                   void** out) {
    *out = memory;
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL unmap(VkDevice, VkDeviceMemory) {}
VKAPI_ATTR VkDeviceAddress VKAPI_CALL address(VkDevice, const VkBufferDeviceAddressInfo* info) {
    return VkDeviceAddress(reinterpret_cast<std::uintptr_t>(reinterpret_cast<FakeBuffer*>(info->buffer)->memory));
}
VKAPI_ATTR void VKAPI_CALL destroy_buffer(VkDevice, VkBuffer buffer, const VkAllocationCallbacks*) {
    --g_live_buffers;
    delete reinterpret_cast<FakeBuffer*>(buffer);
}
VKAPI_ATTR void VKAPI_CALL free_memory(VkDevice, VkDeviceMemory memory, const VkAllocationCallbacks*) {
    --g_live_allocs;
    VirtualFree(memory, 0, MEM_RELEASE);
}

struct FakeDevice {
    struct {
        PFN_vkCreateBuffer vkCreateBuffer = create_buffer;
        PFN_vkGetBufferMemoryRequirements vkGetBufferMemoryRequirements = requirements;
        PFN_vkBindBufferMemory vkBindBufferMemory = bind;
        PFN_vkMapMemory vkMapMemory = map;
        PFN_vkUnmapMemory vkUnmapMemory = unmap;
        PFN_vkGetBufferDeviceAddress vkGetBufferDeviceAddress = address;
        PFN_vkDestroyBuffer vkDestroyBuffer = destroy_buffer;
        PFN_vkFreeMemory vkFreeMemory = free_memory;
    } vk_;
    struct {
        decltype(vk_)& vk;
    } context_{vk_};
    decltype(context_)* context = &context_;
    struct {
        VkDevice device = VK_NULL_HANDLE;
    } info;
    struct {
        VkDeviceAddress address = 0;
    } arena;
    void share(VkBufferCreateInfo&) {}
    void title_block_gone(VkDeviceAddress, VkDeviceSize) { ++g_gone_blocks; }
    VkDeviceMemory allocate(const VkMemoryRequirements& r, VkMemoryPropertyFlags required,
                            VkMemoryPropertyFlags preferred, VkMemoryPropertyFlags* chosen, bool) {
        if (chosen) *chosen = required | preferred;
        void* mem = VirtualAlloc(nullptr, r.size, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
        if (mem) {
            ++g_live_allocs;
            ++g_total_allocs;
        }
        return reinterpret_cast<VkDeviceMemory>(mem);
    }
};

// What the views copy, through the same readability check the device makes.
std::vector<std::size_t> g_copies;
rt::d3d12::GeometryCopy copy(std::uint8_t* to, const std::uint8_t* from, std::size_t size) {
    g_copies.push_back(size);
    std::memcpy(to, from, size);
    return rt::d3d12::GeometryCopy::copied;
}

// The title writes into a watched tile: reopened (as the write-fault handler does), then written.
void title_write(std::uintptr_t at, std::uint8_t value) {
    namespace watch = leonida::geometry_writes;
    watch::notify(at, at + 1, watch::Event::fault);
    *reinterpret_cast<std::uint8_t*>(at) = value;
}

}  // namespace

int main() {
    _putenv_s("LEONIDA_GEOMETRY_MB", "64");
    constexpr std::size_t kTile = 65536, kView = 16 * kTile;
    auto* title = static_cast<std::uint8_t*>(VirtualAlloc(nullptr, 2 * kView, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE));
    if (!title) return 1;
    for (std::size_t i = 0; i < 2 * kView; ++i) title[i] = std::uint8_t(i * 7);
    const auto base = reinterpret_cast<std::uintptr_t>(title);

    FakeDevice device;
    const auto views_owner = std::make_unique<rt::d3d12::GeometryViews>();
    auto& views = *views_owner;
    g_copies.clear();
    const std::uint64_t va = views.resolve(device, base, kView, copy);
    check(va != 0 && g_copies.size() == 1 && g_copies[0] == kView, "a new 1 MiB view is copied whole");
    std::uint64_t host = 0, bytes = 0;
    bool populated = false;
    check(views.lookup(device.arena.address + va, host, bytes, nullptr, &populated) && populated && bytes == kView,
          "the copy is found by its device address");
    auto* copy_bytes = reinterpret_cast<std::uint8_t*>(host);
    check(!std::memcmp(copy_bytes, title, kView), "the copy holds the title's bytes");

    // 2. One write in tile 5.
    g_copies.clear();
    views.refresh(copy);
    check(g_copies.empty(), "a refresh with nothing written copies nothing");
    title_write(base + 5 * kTile + 123, 0xab);
    views.refresh(copy);
    check(g_copies.size() == 1 && g_copies[0] == kTile, "one write copies its 64 KiB tile again, not the 1 MiB view");
    check(copy_bytes[5 * kTile + 123] == 0xab && !std::memcmp(copy_bytes, title, kView),
          "the copy has the new byte and the rest unchanged");

    // 3. A second view over tiles 8..23 shares tiles 8..15 with the first: a write in tile 10 reaches both.
    const std::uint64_t va2 = views.resolve(device, base + 8 * kTile, kView, copy);
    check(va2 != 0, "a second view over half of the first one's tiles");
    std::uint64_t host2 = 0;
    views.lookup(device.arena.address + va2, host2, bytes);
    g_copies.clear();
    title_write(base + 10 * kTile + 9, 0xcd);
    views.refresh(copy);
    check(g_copies.size() == 1 && g_copies[0] == kTile,
          "a write in a shared tile reads that tile from the title's pages once for both views");
    check(copy_bytes[10 * kTile + 9] == 0xcd && reinterpret_cast<std::uint8_t*>(host2)[2 * kTile + 9] == 0xcd,
          "both copies see it");

    // 4. Writes in two tiles of one view: two tile copies.
    g_copies.clear();
    title_write(base + 1 * kTile, 1);
    title_write(base + 3 * kTile + kTile - 1, 2);
    views.refresh(copy);
    check(g_copies.size() == 2 && g_copies[0] == kTile && g_copies[1] == kTile, "two written tiles, two tile copies");
    check(!std::memcmp(copy_bytes, title, kView), "the copy matches the title's pages");

    views.release(device);
    VirtualFree(title, 0, MEM_RELEASE);

    // 5-7. Small cap (2 MiB): fill, cap refusal (cached while nothing freed), deferred retirement for in-flight GPU
    // work, retry after reap, and full block reclamation via geometry_writes::note_released.
    auto* small_title =
        static_cast<std::uint8_t*>(VirtualAlloc(nullptr, 3 * kView, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE));
    if (!small_title) return 1;
    for (std::size_t i = 0; i < 3 * kView; ++i) small_title[i] = std::uint8_t((i * 13) ^ 0x5a);
    const auto sbase = reinterpret_cast<std::uintptr_t>(small_title);
    const auto range_a = sbase;
    const auto range_b = sbase + kView;
    const auto range_c = sbase + 2 * kView;

    const auto capped_owner = std::make_unique<rt::d3d12::GeometryViews>(2 * kView);

    auto& capped = *capped_owner;
    const std::uint64_t va_a = capped.resolve(device, range_a, kView, copy);
    const std::uint64_t va_b = capped.resolve(device, range_b, kView, copy);
    check(va_a != 0 && va_b != 0 && capped.bytes() == 2 * kView && capped.reserved() == 2 * kView &&
              capped.live_views() == 2,
          "small cap fills to its 2 MiB budget");

    // 5. Third 1 MiB view is refused; a repeat resolve without freeing short-circuits without re-counting a refusal.
    const std::uint64_t va_c1 = capped.resolve(device, range_c, kView, copy);
    check(va_c1 == 0 && capped.cap_refusals() == 1, "view past the 2 MiB cap is refused");
    const std::uint64_t va_c2 = capped.resolve(device, range_c, kView, copy);
    check(va_c2 == 0 && capped.cap_refusals() == 1,
          "repeated resolve of a refused range short-circuits while nothing has been freed");

    // 6. Retire A while GPU ticket 5 is still in flight: A disappears from lookup/find immediately, but its storage
    // and budget are held until completed_ticket reaches 5.
    std::uint64_t alias_removed_source = 0;
    VkDeviceSize alias_removed_va = 0;
    const std::size_t retired_a = capped.retire_range(
        range_a, range_a + kView, 5, [&](std::uintptr_t src, std::size_t, VkDeviceSize v) {
            alias_removed_source = src;
            alias_removed_va = v;
        });
    check(retired_a == 1 && alias_removed_source == range_a && alias_removed_va == va_a,
          "retiring A's range reports A's view and invokes the alias-removal callback");
    std::uint64_t lookup_host = 0, lookup_bytes = 0;
    check(capped.find(range_a, kView) == 0 && !capped.lookup(device.arena.address + va_a, lookup_host, lookup_bytes) &&
              capped.live_views() == 1 && capped.retired_count() == 1,
          "retired view is immediately unlinked from find and lookup");

    check(capped.reap(device, 4) == 0 && capped.retired_count() == 1 && capped.bytes() == 2 * kView &&
              capped.free_generation() == 0 && g_live_buffers == 1 && g_live_allocs == 1,
          "reap before GPU ticket completes does not free an in-use view");
    check(capped.resolve(device, range_c, kView, copy) == 0 && capped.cap_refusals() == 1,
          "refused range stays refused while retired view is still in flight");

    // GPU completes ticket 5: A's sub-range is reclaimed and free_generation bumps, letting C retry and succeed.
    check(capped.reap(device, 5) == 1 && capped.retired_count() == 0 && capped.bytes() == kView &&
              capped.free_generation() == 1 && capped.block_count() == 1,
          "reap at GPU ticket 5 reclaims A's budget while keeping B's block live");
    const std::uint64_t va_c3 = capped.resolve(device, range_c, kView, copy);
    check(va_c3 != 0 && capped.bytes() == 2 * kView && capped.live_views() == 2,
          "previously refused range retries and succeeds after budget is freed");
    check(capped.lookup(device.arena.address + va_c3, lookup_host, lookup_bytes) && lookup_bytes == kView &&
              !std::memcmp(reinterpret_cast<const void*>(lookup_host), small_title + 2 * kView, kView),
          "retried view holds C's title bytes");

    // 7. Release B and C via the xmem notification bridge (geometry_writes::note_released + drain_released):
    // once their GPU ticket completes, the block becomes empty and its Vulkan buffer and memory are destroyed.
    leonida::geometry_writes::note_released(range_b, range_c + kView);
    check(capped.has_released(), "note_released queues released title ranges");
    capped.drain_released(device, 6, 5);
    check(!capped.has_released() && capped.live_views() == 0 && capped.retired_count() == 2 &&
              g_live_buffers == 1 && g_live_allocs == 1,
          "drain_released retires B and C at ticket 6 and defers freeing while completed ticket is 5");
    check(capped.reap(device, 6) == 2 && capped.retired_count() == 0 && capped.bytes() == 0 &&
              capped.reserved() == 0 && capped.block_count() == 0 && g_live_buffers == 0 && g_live_allocs == 0,
          "reaping the last views in a block destroys its Vulkan buffer and returns all reserved budget");

    // A fresh 2 MiB view now fits in the fully reclaimed cap.
    const std::uint64_t va_full = capped.resolve(device, range_a, 2 * kView, copy);
    check(va_full != 0 && capped.bytes() == 2 * kView && capped.reserved() == 2 * kView && capped.live_views() == 1,
          "a fresh 2 MiB view succeeds after full reclamation");

    capped.release(device);
    check(g_live_buffers == 0 && g_live_allocs == 0, "teardown frees all remaining Vulkan buffers and memory");
    VirtualFree(small_title, 0, MEM_RELEASE);

    // 8-11. Watched-bounds filtering, release-queue overflow collapse, 4 MiB spare block reuse and cap eviction.
    auto* pool_title =
        static_cast<std::uint8_t*>(VirtualAlloc(nullptr, 6 * kView, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE));
    if (!pool_title) return 1;
    for (std::size_t i = 0; i < 6 * kView; ++i) pool_title[i] = std::uint8_t((i * 17) ^ 0x3c);
    const auto pbase = reinterpret_cast<std::uintptr_t>(pool_title);

    // 8. note_released outside [watched_lo, watched_hi) is ignored; inside is queued.
    const auto watched_owner = std::make_unique<rt::d3d12::GeometryViews>(6 * kView);
    auto& watched = *watched_owner;
    const std::uint64_t va_w1 = watched.resolve(device, pbase, kView, copy);
    const std::uint64_t va_w2 = watched.resolve(device, pbase + kView, kView, copy);
    check(va_w1 != 0 && va_w2 != 0 && watched.live_views() == 2 && watched.block_count() == 1 &&
              watched.empty_blocks() == 0,
          "two 1 MiB views share one 4 MiB block");
    const auto watched_lo = leonida::geometry_writes::state().watched_lo.load(std::memory_order_acquire);
    const auto watched_hi = leonida::geometry_writes::state().watched_hi.load(std::memory_order_acquire);
    check(watched_lo <= pbase && watched_hi >= pbase + 2 * kView, "watched bounds cover resolved views");
    leonida::geometry_writes::note_released(watched_lo - 4096, watched_lo);
    leonida::geometry_writes::note_released(watched_hi, watched_hi + 4096);
    check(!watched.has_released(), "note_released outside watched bounds is not queued");
    leonida::geometry_writes::note_released(pbase, pbase + 4096);
    check(watched.has_released(), "note_released inside watched bounds is queued");
    std::vector<std::pair<std::uintptr_t, std::uintptr_t>> drained;
    leonida::geometry_writes::take_released(watched.release_queue(), drained);
    check(drained.size() == 1 && drained[0].first == pbase && drained[0].second == pbase + 4096 &&
              !watched.has_released(),
          "take_released returns the queued in-bounds range");
    // A GPU-only twin freed by its own address lies outside the watched (CPU) bounds but still counts: its CPU view
    // holds the views. Only the address arithmetic of the pair is used; nothing is mapped at the GPU address.
    const std::uintptr_t gpu_twin = std::uintptr_t(0x600000000000ull);
    leonida::title_gpu_views::note_reservation(pbase, 2 * kView, false);
    check(leonida::title_gpu_views::note_reservation(gpu_twin, 2 * kView, true), "the GPU-only twin pairs");
    leonida::geometry_writes::note_released(gpu_twin + 2 * kView, gpu_twin + 2 * kView + 4096);
    check(!watched.has_released(), "a range past the twin is not queued");
    leonida::geometry_writes::note_released(gpu_twin + kView, gpu_twin + kView + 4096);
    check(watched.has_released(), "a range of the GPU-only twin over a watched view is queued");
    watched.drain_released(device, 3, 0);
    check(watched.live_views() == 1 && watched.retired_count() == 1 && watched.find(pbase + kView, kView) == 0 &&
              watched.find(pbase, kView) == va_w1,
          "draining the twin's range retires the view at its CPU address only");
    const std::uint64_t va_w2b = watched.resolve(device, pbase + kView, kView, copy);
    check(va_w2b != 0, "the retired view is made again on the next resolve");

    // 9. Queue overflow: 65537 disjoint ranges collapse into one covering range; drain retires every covered view.
    constexpr std::size_t kOverflowCount = 65537;
    for (std::size_t i = 0; i < kOverflowCount; ++i)
        leonida::geometry_writes::note_released(pbase + i * 16, pbase + i * 16 + 8);
    leonida::geometry_writes::take_released(watched.release_queue(), drained);
    check(drained.size() == 1 && drained[0].first == pbase &&
              drained[0].second == pbase + (kOverflowCount - 1) * 16 + 8 && !watched.has_released(),
          "65537 disjoint ranges collapse into a single covering range");
    for (std::size_t i = 0; i < kOverflowCount; ++i)
        leonida::geometry_writes::note_released(pbase + i * 16, pbase + i * 16 + 8);
    watched.drain_released(device, 10, 9);
    check(!watched.has_released() && watched.live_views() == 0 && watched.retired_count() == 2,
          "drain_released after overflow retires every view inside the collapsed range");
    watched.release(device);
    check(g_live_buffers == 0 && g_live_allocs == 0, "watched teardown frees its block");

    // 10. Spare 4 MiB block: retiring and reaping the sole view keeps the block empty and the next view reuses it.
    const auto spare_owner = std::make_unique<rt::d3d12::GeometryViews>(6 * kView);
    auto& spare = *spare_owner;
    const std::uint64_t va_s1 = spare.resolve(device, pbase, kView, copy);
    const int allocs_after_first = g_total_allocs;
    check(va_s1 != 0 && spare.block_count() == 1 && spare.empty_blocks() == 0 && spare.reserved() == 4 * kView &&
              g_live_buffers == 1 && g_live_allocs == 1,
          "first view under a 6 MiB cap allocates one standard 4 MiB block");
    check(spare.retire_range(pbase + kView - 16, pbase + kView, 1) == 1 && spare.reap(device, 1) == 1,
          "retiring a tail subrange and reaping frees the view");
    check(spare.live_views() == 0 && spare.block_count() == 1 && spare.empty_blocks() == 1 &&
              spare.reserved() == 4 * kView && g_live_buffers == 1 && g_live_allocs == 1,
          "reaping the last view keeps the empty 4 MiB block as a spare");
    const std::uint64_t va_s2 = spare.resolve(device, pbase + kView, kView, copy);
    check(va_s2 != 0 && spare.block_count() == 1 && spare.empty_blocks() == 0 && g_live_buffers == 1 &&
              g_live_allocs == 1 && g_total_allocs == allocs_after_first,
          "a new view reuses the spare 4 MiB block with no new allocation");

    // 11. Spare given up under the cap: destroy the unfitting 4 MiB spare so a larger request succeeds, and still
    // refuse a request that remains over the cap after the spare is destroyed.
    check(spare.retire_range(pbase + kView, pbase + 2 * kView, 2) == 1 && spare.reap(device, 2) == 1 &&
              spare.block_count() == 1 && spare.empty_blocks() == 1 && spare.reserved() == 4 * kView,
          "reaping the reused view returns the 4 MiB block to spare");
    const std::uint64_t va_big = spare.resolve(device, pbase, 5 * kView, copy);
    check(va_big != 0 && spare.block_count() == 1 && spare.empty_blocks() == 0 && spare.reserved() == 5 * kView &&
              g_live_buffers == 1 && g_live_allocs == 1,
          "a 5 MiB request under a 6 MiB cap destroys the 4 MiB spare and succeeds");
    spare.release(device);
    check(g_live_buffers == 0 && g_live_allocs == 0, "spare teardown frees its block");

    const auto tight_owner = std::make_unique<rt::d3d12::GeometryViews>(9 * kView);

    auto& tight = *tight_owner;
    const std::uint64_t va_t1 = tight.resolve(device, pbase, 3 * kView, copy);
    const std::uint64_t va_t2 = tight.resolve(device, pbase + 3 * kView, 2 * kView, copy);
    check(va_t1 != 0 && va_t2 != 0 && tight.block_count() == 2 && tight.reserved() == 8 * kView,
          "two views that do not share one 4 MiB block reserve 8 MiB");
    check(tight.retire_range(pbase + 3 * kView, pbase + 5 * kView, 1) == 1 && tight.reap(device, 1) == 1 &&
              tight.block_count() == 2 && tight.empty_blocks() == 1 && tight.reserved() == 8 * kView,
          "reaping the second block leaves one 4 MiB spare alongside one live 4 MiB block");
    const std::uint64_t va_over = tight.resolve(device, pbase, 6 * kView, copy);
    check(va_over == 0 && tight.cap_refusals() == 1 && tight.block_count() == 1 && tight.empty_blocks() == 0 &&
              tight.reserved() == 4 * kView && g_live_buffers == 1 && g_live_allocs == 1,
          "a 6 MiB request drops the 4 MiB spare and is refused because only 5 MiB of cap remain");
    const std::uint64_t va_fit = tight.resolve(device, pbase, 5 * kView, copy);
    check(va_fit != 0 && tight.block_count() == 2 && tight.reserved() == 9 * kView && g_live_buffers == 2 &&
              g_live_allocs == 2,
          "a 5 MiB request succeeds in the cap freed by dropping the spare");
    tight.release(device);
    check(g_live_buffers == 0 && g_live_allocs == 0, "tight cap teardown frees all blocks");

    VirtualFree(pool_title, 0, MEM_RELEASE);

    // 12. Two GeometryViews instances over the same title extent both receive one note_released; a destroyed instance
    // unsubscribes so later releases do not touch it and surviving instances still receive them.
    auto* shared_title =
        static_cast<std::uint8_t*>(VirtualAlloc(nullptr, 2 * kView, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE));
    if (!shared_title) return 1;
    for (std::size_t i = 0; i < 2 * kView; ++i) shared_title[i] = std::uint8_t((i * 29) ^ 0xa5);
    const auto shbase = reinterpret_cast<std::uintptr_t>(shared_title);

    {
        const auto dev_a_owner = std::make_unique<rt::d3d12::GeometryViews>(2 * kView);
        auto& dev_a = *dev_a_owner;
        const auto dev_b_owner = std::make_unique<rt::d3d12::GeometryViews>(2 * kView);
        auto& dev_b = *dev_b_owner;
        const std::uint64_t va_a1 = dev_a.resolve(device, shbase, kView, copy);
        const std::uint64_t va_b1 = dev_b.resolve(device, shbase, kView, copy);
        check(va_a1 != 0 && va_b1 != 0 && dev_a.live_views() == 1 && dev_b.live_views() == 1 &&
                  dev_a.bytes() == kView && dev_b.bytes() == kView,
              "two GeometryViews instances resolve the same title range");

        leonida::geometry_writes::note_released(shbase, shbase + kView);
        check(dev_a.has_released() && dev_b.has_released(),
              "one note_released queues the range in both instances");

        dev_a.drain_released(device, 1, 1);
        check(!dev_a.has_released() && dev_a.live_views() == 0 && dev_a.retired_count() == 0 && dev_a.bytes() == 0 &&
                  dev_b.has_released() && dev_b.live_views() == 1 && dev_b.bytes() == kView,
              "draining the first instance retires and reaps its view without emptying the second instance's queue");

        dev_b.drain_released(device, 1, 1);
        check(!dev_b.has_released() && dev_b.live_views() == 0 && dev_b.retired_count() == 0 && dev_b.bytes() == 0,
              "draining the second instance also retires and reaps its view");

        const std::uint64_t va_keep = dev_a.resolve(device, shbase + kView, kView, copy);
        check(va_keep != 0 && dev_a.live_views() == 1 && dev_a.bytes() == kView,
              "surviving instance resolves a second range");
        {
            const auto ephemeral_owner = std::make_unique<rt::d3d12::GeometryViews>(2 * kView);
            auto& ephemeral = *ephemeral_owner;
            const std::uint64_t va_eph = ephemeral.resolve(device, shbase + kView, kView, copy);
            check(va_eph != 0 && ephemeral.live_views() == 1, "scoped instance resolves the same range");
            ephemeral.release(device);
        }
        leonida::geometry_writes::note_released(shbase + kView, shbase + 2 * kView);
        check(dev_a.has_released(), "note_released after another instance is destroyed reaches the surviving instance");
        dev_a.drain_released(device, 2, 2);
        check(!dev_a.has_released() && dev_a.live_views() == 0 && dev_a.retired_count() == 0 && dev_a.bytes() == 0,
              "surviving instance retires and reaps after the destroyed instance unsubscribed");

        dev_a.release(device);
        dev_b.release(device);
    }
    check(g_live_buffers == 0 && g_live_allocs == 0, "multi-instance teardown frees all blocks");
    VirtualFree(shared_title, 0, MEM_RELEASE);

    // 13. DeviceState's order for one submission: execute_begin, drain_geometry_releases (kAfterExecutes) and the
    // console heaps in before_execute_, the replay records its submission, execute_end. Another queue's replay began
    // before the release (its heap slots may already name the view); one more begins after the retirement.
    auto* fly_title = static_cast<std::uint8_t*>(VirtualAlloc(nullptr, kView, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE));
    if (!fly_title) return 1;
    for (std::size_t i = 0; i < kView; ++i) fly_title[i] = std::uint8_t(i * 13 + 1);
    const auto flybase = reinterpret_cast<std::uintptr_t>(fly_title);
    {
        const auto fly_owner = std::make_unique<rt::d3d12::GeometryViews>(2 * kView);
        auto& fly = *fly_owner;
        const auto named_blocks = [&] {
            std::size_t n = 0;
            fly.pool().for_each_block([&](VkDeviceAddress address, VkDeviceSize) { n += address != 0; });
            return n;
        };
        const std::uint64_t va_f = fly.resolve(device, flybase, kView, copy);
        check(va_f != 0 && fly.live_views() == 1 && named_blocks() == 1, "a view in its own block, published");
        const std::uint64_t layout = fly.pool().layout_generation();
        const int gone = g_gone_blocks;

        std::uint64_t submitted = 7;  // completed == submitted below: the GPU is idle
        const std::uint64_t other = fly.execute_begin();
        const std::uint64_t own = fly.execute_begin();
        leonida::TitleVaAliases aliases;
        aliases.record_removals(true);
        // What the console heaps' slot records are told (TitleVaAliases::add_listener).
        struct Heard {
            bool removed;
            std::uint64_t title_va, bytes, shader_va;
        };
        std::vector<Heard> heard;
        aliases.add_listener(&heard, [](void* to, leonida::TitleVaAliases::Change change, std::uint64_t va,
                                        std::uint64_t bytes, std::uint64_t shader_va) {
            static_cast<std::vector<Heard>*>(to)->push_back(
                {change == leonida::TitleVaAliases::Change::removed, va, bytes, shader_va});
        });
        leonida::geometry_writes::note_released(flybase, flybase + kView);
        fly.drain_released(device, rt::d3d12::GeometryViews::kAfterExecutes, submitted,
                           [&](std::uintptr_t source, std::size_t size, VkDeviceSize va) {
                               aliases.forget_copy(source, size, va);
                           });
        check(fly.live_views() == 0 && fly.find(flybase, kView) == 0 && fly.retired_count() == 1 &&
                  fly.bytes() == kView && g_gone_blocks == gone,
              "a view retired in before_execute_ is not reaped there, though the GPU is idle");

        std::vector<std::pair<std::uint64_t, std::uint64_t>> removed;
        aliases.take_removed(removed);
        aliases.remove_listener(&heard);
        check(heard.size() == 1 && heard[0].removed && heard[0].title_va == flybase && heard[0].bytes == kView &&
                  heard[0].shader_va == va_f && removed.size() == 1 && removed[0].first == va_f &&
                  removed[0].second == kView,
              "the retirement invalidates the heap slots and key answers made through the view without an alias");

        submitted = 8;  // this replay's submission
        fly.execute_end(own, [&] { return submitted; });
        check(fly.reap(device, 8) == 0 && fly.retired_count() == 1 && fly.bytes() == kView,
              "not freed while a replay begun before the retirement is still in progress");

        const std::uint64_t later = fly.execute_begin();
        submitted = 9;  // the other queue's submission
        fly.execute_end(other, [&] { return submitted; });
        check(fly.reap(device, 8) == 0 && fly.retired_count() == 1,
              "once those replays recorded theirs, the view waits for the last of their tickets");
        check(fly.reap(device, 9) == 1 && fly.retired_count() == 0 && fly.bytes() == 0,
              "freed once that ticket completed; a replay begun after the retirement does not hold it");
        fly.execute_end(later, [&] { return submitted; });

        check(named_blocks() == 0 && fly.pool().layout_generation() != layout && g_gone_blocks == gone + 1 &&
                  g_live_buffers == 0,
              "the destroyed block leaves the lock-free block list and the device is told");
        fly.release(device);
    }
    VirtualFree(fly_title, 0, MEM_RELEASE);

    // 14. LEONIDA_GEOMETRY_HOST: the copies' memory type. A discrete GPU's types: 0 VRAM, 1 VRAM through the BAR,
    // 2 system memory write-combined, 3 system memory write-back. 1 takes type 3 (write-back preferred), 2 type 2,
    // unset any host-visible type with device-local preferred (the BAR); a device whose host-visible types are all
    // device-local (an integrated GPU) keeps them all.
    {
        struct TypedDevice : FakeDevice {
            VkPhysicalDeviceMemoryProperties memory{};
            std::uint32_t bits = 0;
            VkMemoryPropertyFlags preferred = 0;
            VkDeviceMemory allocate(const VkMemoryRequirements& r, VkMemoryPropertyFlags required,
                                    VkMemoryPropertyFlags wanted, VkMemoryPropertyFlags* chosen, bool address) {
                bits = r.memoryTypeBits;
                preferred = wanted;
                return FakeDevice::allocate(r, required, wanted, chosen, address);
            }
        };
        constexpr VkMemoryPropertyFlags kLocal = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                        kHost = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                        kCached = VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
        TypedDevice discrete;
        discrete.memory.memoryTypeCount = 4;
        const VkMemoryPropertyFlags discrete_types[4] = {kLocal, kLocal | kHost, kHost, kHost | kCached};
        for (int i = 0; i < 4; ++i) discrete.memory.memoryTypes[i].propertyFlags = discrete_types[i];
        TypedDevice integrated;
        integrated.memory.memoryTypeCount = 2;
        integrated.memory.memoryTypes[0].propertyFlags = kLocal;
        integrated.memory.memoryTypes[1].propertyFlags = kLocal | kHost | kCached;
        auto* host_title =
            static_cast<std::uint8_t*>(VirtualAlloc(nullptr, kView, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE));
        if (!host_title) return 1;
        const auto hbase = reinterpret_cast<std::uintptr_t>(host_title);
        const auto made_with = [&](TypedDevice& dev, const char* setting) {
            _putenv_s("LEONIDA_GEOMETRY_HOST", setting);
            dev.bits = ~0u;
            dev.preferred = 0;
            rt::d3d12::GeometryViews v;
            const bool made = v.resolve(dev, hbase, kView, copy) != 0;
            const int kind = v.system_memory();
            v.release(dev);
            return made ? kind : -1;
        };
        check(made_with(discrete, "1") == rt::d3d12::TitleBufferWindows::kSystemCached && discrete.bits == 1u << 3 &&
                  discrete.preferred == kCached,
              "LEONIDA_GEOMETRY_HOST=1 puts the copies in write-back system memory, never the BAR");
        check(made_with(discrete, "2") == rt::d3d12::TitleBufferWindows::kSystemUncached && discrete.bits == 1u << 2 &&
                  discrete.preferred == 0,
              "LEONIDA_GEOMETRY_HOST=2 puts them in write-combined system memory");
        check(made_with(discrete, "") == rt::d3d12::TitleBufferWindows::kDeviceLocal && discrete.bits == 0xfu &&
                  discrete.preferred == kLocal,
              "unset, every type the buffer allows is tried, device-local first");
        check(made_with(integrated, "1") == rt::d3d12::TitleBufferWindows::kSystemCached && integrated.bits == 0xfu,
              "on a device with no host type outside device-local memory, the types are left as they are");
        _putenv_s("LEONIDA_GEOMETRY_HOST", "");
        check(g_live_buffers == 0, "the memory-type views free their blocks");

        // 15. A retired view is copied once more from the title's pages when they are readable (they hold new bytes),
        // and keeps what it held when they are not (decommitted or unmapped).
        rt::d3d12::GeometryViews last;
        std::memset(host_title, 0x11, kView);
        const std::uint64_t va_l = last.resolve(device, hbase, kView, copy);
        std::uint64_t host_l = 0, bytes_l = 0;
        check(va_l && last.lookup(device.arena.address + va_l, host_l, bytes_l), "a view to retire");
        auto* held = reinterpret_cast<const std::uint8_t*>(host_l);
        // The pages' new bytes: the watched tiles are opened first (no fault handler here).
        leonida::geometry_writes::notify(hbase, hbase + kView, leonida::geometry_writes::Event::open);
        std::memset(host_title, 0x22, kView);
        g_copies.clear();
        check(last.retire_range(hbase, hbase + kView, 1, [](std::uintptr_t, std::size_t, VkDeviceSize) {}, copy) == 1 &&
                  held[0] == 0x22 && held[kView - 1] == 0x22 && last.final_copies() == 1 && g_copies.size() == 1,
              "a retired view whose pages are readable is copied from them once more");
        const std::uint64_t va_l2 = last.resolve(device, hbase, kView, copy);
        std::uint64_t host_l2 = 0;
        check(va_l2 && last.lookup(device.arena.address + va_l2, host_l2, bytes_l), "and made again");
        auto* held2 = reinterpret_cast<const std::uint8_t*>(host_l2);
        leonida::geometry_writes::notify(hbase, hbase + kView, leonida::geometry_writes::Event::open);
        std::memset(host_title, 0x33, kView);
        const auto unreadable = [](std::uint8_t*, const std::uint8_t*, std::size_t) {
            return rt::d3d12::GeometryCopy::unreadable;
        };
        check(last.retire_range(hbase, hbase + kView, 2, [](std::uintptr_t, std::size_t, VkDeviceSize) {},
                                unreadable) == 1 &&
                  held2[0] == 0x22 && last.final_copies() == 1,
              "one whose pages are not readable keeps what it held");
        check(last.reap(device, 2) == 2 && last.bytes() == 0, "both are freed at their tickets");
        last.release(device);

        // 16. A full release queue first joins ranges that touch (unmaps report page runs): 65536 neighbouring ranges
        // and one apart from them are two ranges, not one covering everything between them.
        rt::d3d12::GeometryViews coalesce;
        check(coalesce.resolve(device, hbase, kView, copy) != 0, "a view to watch the pages");
        for (std::size_t i = 0; i < 65536; ++i)
            leonida::geometry_writes::note_released(hbase + i * 8, hbase + i * 8 + 8);
        leonida::geometry_writes::note_released(hbase + kView - 4096, hbase + kView);
        std::vector<std::pair<std::uintptr_t, std::uintptr_t>> joined;
        leonida::geometry_writes::take_released(coalesce.release_queue(), joined);
        check(joined.size() == 2 && joined[0] == std::make_pair(hbase, hbase + 65536 * 8) &&
                  joined[1] == std::make_pair(hbase + kView - 4096, hbase + kView),
              "65536 touching ranges become one, and a range apart from them stays apart");
        coalesce.release(device);
        VirtualFree(host_title, 0, MEM_RELEASE);
    }

    std::printf("%d checks, %d failures\n", g_checks, g_failures);
    return g_failures ? 1 : 0;
}
