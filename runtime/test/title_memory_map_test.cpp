// leonida::title_memory_map (runtime/src/title_page_watch.h): the journal of changes to the title's address space, by
// range, and what the D3D12 layer's readers build on it: the title-texture pass probes again only the textures a
// change overlaps (TextureReadability, src/title_texture_memory.h), and a slot-133 shadow queries again only its
// pages a change overlaps. No game data, no GPU.
//   1. a reader is told the changes since its generation that overlap its range, clipped and in whole pages;
//   2. a change elsewhere moves the generation but tells the reader nothing;
//   3. a reader further behind than the journal holds is told so (it then looks at everything);
//   4. a change with no range overlaps every reader;
//   5. TextureReadability keeps a positive answer, when asked to, until a retry;
//   6. a pass over 20,000 waiting textures probes only those under the one mapped block;
//   7. between passes, sync_map retries only when a map change overlaps the texture's footprint.
#include <windows.h>

#include <cstdint>
#include <cstdio>
#include <utility>
#include <vector>

#include "../src/title_page_watch.h"
#include "../src/title_texture_memory.h"
#include "../src/title_texture_refresh.h"

namespace {

namespace map = leonida::title_memory_map;

int g_checks = 0;
int g_failures = 0;

void check(bool condition, const char* what) {
    ++g_checks;
    std::printf("%s %s\n", condition ? "PASS" : "FAIL", what);
    if (!condition) ++g_failures;
}

using Ranges = std::vector<std::pair<std::uintptr_t, std::uintptr_t>>;

bool since(std::uint64_t from, std::uint64_t& now, std::uintptr_t lo, std::uintptr_t hi, Ranges& out) {
    out.clear();
    return map::changes_since(from, now, lo, hi, [&](std::uintptr_t a, std::uintptr_t b) { out.emplace_back(a, b); });
}

}  // namespace

int main() {
    constexpr std::uintptr_t kBase = 0x7000000000;  // no memory is touched: the journal only records addresses
    Ranges got;
    std::uint64_t now = 0;

    // 1. Overlapping changes only, clipped to the reader's range, rounded out to whole 4 KiB pages.
    const std::uint64_t start = map::generation();
    map::changed(kBase + 0x10010, kBase + 0x10020);   // inside, not page-aligned
    map::changed(kBase + 0x90000, kBase + 0xa0000);   // outside
    map::changed(kBase + 0x1f000, kBase + 0x31000);   // straddles the reader's end
    check(since(start, now, kBase + 0x10000, kBase + 0x20000, got), "the journal holds the reader's changes");
    check(now == start + 3, "the reader is brought to the latest generation");
    check(got.size() == 2 && got[0] == std::make_pair(kBase + 0x10000, kBase + 0x11000) &&
              got[1] == std::make_pair(kBase + 0x1f000, kBase + 0x20000),
          "two overlapping changes, in whole pages, clipped to the reader's range");

    // 2. Caught up: a change elsewhere tells nothing, but the generation moves.
    const std::uint64_t caught_up = now;
    map::changed(reinterpret_cast<const void*>(kBase + 0x400000), 0x10000);
    check(since(caught_up, now, kBase + 0x10000, kBase + 0x20000, got) && got.empty() && now == caught_up + 1,
          "a change outside the range tells the reader nothing");

    // 3. A reader more than kJournal changes behind is told it missed some.
    const std::uint64_t behind = map::generation();
    for (std::size_t i = 0; i <= map::kJournal; ++i) map::changed(kBase + 0x800000, kBase + 0x801000);
    check(!since(behind, now, kBase, kBase + 0x1000, got) && got.empty(),
          "a reader further behind than the journal is told to look at everything");
    const std::uint64_t latest = map::generation();
    check(since(latest - 10, now, kBase + 0x800000, kBase + 0x801000, got) && got.size() == 10,
          "a reader within the journal is told each of its changes");

    // 4. No range: everyone's.
    const std::uint64_t before_all = map::generation();
    map::changed();
    check(since(before_all, now, kBase + 0x123000, kBase + 0x124000, got) && got.size() == 1 &&
              got[0] == std::make_pair(kBase + 0x123000, kBase + 0x124000),
          "a change with no range overlaps every reader");

    // 5. TextureReadability: positive answers kept until a retry when asked to, else probed every time as before.
    {
        leonida::TextureReadability r;
        unsigned probes = 0;
        const auto readable = [&] { ++probes; return true; };
        for (int i = 0; i < 100; ++i) r.check(0, true, readable, true);
        check(probes == 1 && !r.pending(), "a readable texture is probed once in 100 checks with cache_success");
        r.retry();
        check(r.check(0, true, readable, true) && probes == 2, "a retry probes it again");
        for (int i = 0; i < 10; ++i) r.check(0, true, readable);
        check(probes == 12, "without cache_success every check probes (unchanged behaviour)");
        r.defer();  // a page went between the probe and the read
        const auto unreadable = [&] { ++probes; return false; };
        check(!r.check(0, true, unreadable, true) && probes == 12, "a deferred texture waits for a retry");
        r.retry();
        check(!r.check(0, true, unreadable, true) && probes == 13 && r.pending(), "the retry finds it unreadable");
    }

    // 6. The pass: 20,000 waiting textures of 16 KiB each; one 64 KiB block is mapped and reported. Only the four
    //    textures in it are probed again; the next pass, with no change, probes none.
    {
        struct Texture {
            std::uintptr_t cpu;
            std::uint64_t size;
            leonida::TextureReadability memory;
        };
        std::vector<Texture> textures(20000);
        unsigned probes = 0;
        std::uintptr_t mapped_from = 0, mapped_to = 0;
        const auto probe = [&](const Texture& t) {
            ++probes;
            return t.cpu >= mapped_from && t.cpu + t.size <= mapped_to;
        };
        for (std::size_t i = 0; i < textures.size(); ++i) {
            textures[i].cpu = kBase + 0x10000000 + i * 0x4000;
            textures[i].size = 0x4000;
            textures[i].memory.check(0, true, [&] { return probe(textures[i]); }, true);
        }
        std::uint64_t pass_generation = map::generation();
        const auto pass = [&] {
            leonida::TitleWriteRanges changes;
            std::uint64_t upto = 0;
            const bool all = !map::changes_since(pass_generation, upto, 0, UINTPTR_MAX,
                                                 [&](std::uintptr_t a, std::uintptr_t b) { changes.emplace_back(a, b); });
            pass_generation = upto;
            leonida::merge_title_writes(changes);
            unsigned readable = 0;
            for (Texture& t : textures) {
                if (all || leonida::title_write_overlaps(changes, t.cpu, t.cpu + t.size)) t.memory.retry();
                readable += t.memory.check(0, true, [&] { return probe(t); }, true);
            }
            return readable;
        };
        probes = 0;
        mapped_from = kBase + 0x10000000 + 0x10000 * 100;
        mapped_to = mapped_from + 0x10000;
        map::changed(mapped_from, mapped_to);
        const unsigned readable = pass();
        check(probes == 4 && readable == 4, "a mapped block probes and finds only the four textures in it");
        probes = 0;
        check(pass() == 4 && probes == 0, "a pass with no change probes nothing");
    }

    // 7. Synchronous view checks between passes (sync_map): unrelated map changes cost no probe, an overlapping unmap
    //    defers without reading, and an overlapping map wakes a waiting texture before the next pass.
    {
        constexpr std::uintptr_t cpu = kBase + 0x20000000;
        constexpr std::uint64_t size = 512 * 1024;
        const std::uint64_t pass_generation = map::generation();
        bool mapped = true;
        unsigned probes = 0;
        leonida::TextureReadability r;
        const auto probe = [&] {
            ++probes;
            return mapped;
        };
        check(r.check(0, true, probe, true) && probes == 1 && !r.pending(),
              "sync_map: initial probe caches a readable answer");
        for (std::uintptr_t i = 0; i < 32; ++i)
            map::changed(cpu + size + i * 0x10000, cpu + size + (i + 1) * 0x10000);
        r.sync_map(pass_generation, cpu, size);
        check(r.check(0, true, probe, true) && probes == 1,
              "sync_map: map changes outside the footprint keep the cached answer");
        r.sync_map(pass_generation, cpu, size);
        check(r.check(0, true, probe, true) && probes == 1,
              "sync_map: a second check at the same generation does nothing");

        mapped = false;
        map::changed(cpu + 0x10000, cpu + 0x20000);
        r.sync_map(pass_generation, cpu, size);
        check(!r.check(0, true, probe, true) && r.pending() && probes == 2,
              "sync_map: an overlapping unmap between passes reprobes and defers");

        map::changed(cpu - 0x10000, cpu);
        r.sync_map(pass_generation, cpu, size);
        check(!r.check(0, true, probe, true) && r.pending() && probes == 2,
              "sync_map: an adjacent non-overlapping change leaves a pending texture waiting");

        mapped = true;
        map::changed(cpu, cpu + size);
        r.sync_map(pass_generation, cpu, size);
        check(r.check(0, true, probe, true) && !r.pending() && probes == 3,
              "sync_map: an overlapping map between passes wakes a pending texture at once");
    }

    std::printf("%d checks, %d failures\n", g_checks, g_failures);
    return g_failures ? 1 : 0;
}
