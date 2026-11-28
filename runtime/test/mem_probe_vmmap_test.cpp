// leonida::mem_probe's vmmap line (src/mem_probe.h, LEONIDA_MEM_PROBE=<seconds>): on native Windows there is no
// /proc/<pid>/smaps to read, so the probe walks the address space with VirtualQuery and sums the committed bytes by the
// smaps line's kinds. The classifier cuts synthetic regions at the title and vk_mapped ranges and names the rest by
// their type; a walk of this process sees 64 MiB it commits as PE private memory, or as title memory inside a title
// range, and a mapped view as a PE section, or as memory the Vulkan driver mapped inside a vk_mapped range. A walk
// over a 4 GiB reservation with 65 536 committed pages prints how long it takes. No GPU, no device.
#include "../src/mem_probe.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

namespace {

namespace mp = leonida::mem_probe;
constexpr std::uint64_t kMiB = 1 << 20;
constexpr std::uint64_t kKiB = 1 << 10;

bool contains(const std::string& text, const char* part) { return text.find(part) != std::string::npos; }

std::uint64_t walk(mp::VmTally& tally) {
    std::uint64_t reserved = 0;
    return mp::walk_address_space(tally, reserved);
}

// Synthetic regions: the part inside a title range is title memory, the part inside a vk_mapped range is the driver's,
// the rest goes by the region's type, and a region cut by a range counts once in each kind it reaches. The title ranges
// come unsorted and overlapping, as a caller may hand them.
int classifier() {
    const mp::Ranges title = {{0x30000000, 0x30100000}, {0x10000000, 0x18000000}, {0x17000000, 0x20000000}};
    const mp::Ranges vk_mapped = {{0x40000000, 0x40200000}};
    mp::VmTally t(title, vk_mapped);
    t.add({0x10000000, 64 * kMiB, 0x10000000, MEM_PRIVATE});   // all title
    t.add({0x1FF00000, 2 * kMiB, 0x1FF00000, MEM_PRIVATE});    // 1 MiB title, then 1 MiB private
    t.add({0x2FF00000, 3 * kMiB, 0x2FF00000, MEM_MAPPED});     // 1 MiB section, 1 MiB title, 1 MiB section
    t.add({0x40000000, 2 * kMiB, 0x40000000, MEM_MAPPED});     // all vk_mapped
    t.add({0x401F0000, 128 * kKiB, 0x401F0000, MEM_PRIVATE});  // 64 KiB vk_mapped, then 64 KiB private
    t.add({0x140000000, 3 * kMiB, 0x140000000, MEM_IMAGE});
    t.add({0x50000000, 5 * kMiB, 0x50000000, MEM_MAPPED});
    t.add({0x60000000, 1 * kMiB, 0x60000000, MEM_PRIVATE});    // one allocation in two regions
    t.add({0x60100000, 7 * kMiB, 0x60000000, MEM_PRIVATE});
    EXPECT(t.committed[mp::b_title] == 66 * kMiB && t.regions[mp::b_title] == 3);
    EXPECT(t.committed[mp::b_pe_private] == 9 * kMiB + 64 * kKiB && t.regions[mp::b_pe_private] == 4);
    EXPECT(t.committed[mp::b_pe_mapped] == 7 * kMiB && t.regions[mp::b_pe_mapped] == 2);
    EXPECT(t.committed[mp::b_vk_mapped] == 2 * kMiB + 64 * kKiB && t.regions[mp::b_vk_mapped] == 2);
    EXPECT(t.committed[mp::b_pe_image] == 3 * kMiB && t.regions[mp::b_pe_image] == 1);
    std::uint64_t sum = 0;
    for (int b = 0; b < mp::b_count; ++b) sum += t.committed[b];
    EXPECT(sum == 87 * kMiB + 128 * kKiB);  // every byte of every region, once
    EXPECT(t.total() == sum);
    const auto largest = t.largest_private(2);
    EXPECT(largest.size() == 2);
    EXPECT(largest[0].first == 8 * kMiB && largest[0].second == 0x60000000);
    EXPECT(largest[1].first == 1 * kMiB && largest[1].second == 0x1FF00000);
    // Without ranges every byte goes by its type.
    mp::VmTally bare({}, {});
    bare.add({0x10000000, 64 * kMiB, 0x10000000, MEM_PRIVATE});
    bare.add({0x40000000, 2 * kMiB, 0x40000000, MEM_MAPPED});
    EXPECT(bare.committed[mp::b_pe_private] == 64 * kMiB && bare.committed[mp::b_pe_mapped] == 2 * kMiB);
    EXPECT(bare.committed[mp::b_title] == 0 && bare.committed[mp::b_vk_mapped] == 0);
    return 0;
}

// The text: the walk's time and size, the process counters, the committed MiB of every kind with its regions, the
// reserved bytes and the largest private allocations; "n/a" where the system did not report the private working set.
int text() {
    mp::VmTally t({{0x10000000, 0x20000000}}, {{0x40000000, 0x40200000}});
    t.add({0x10000000, 64 * kMiB, 0x10000000, MEM_PRIVATE});
    t.add({0x40000000, 2 * kMiB, 0x40000000, MEM_MAPPED});
    t.add({0x60000000, 8 * kMiB, 0x60000000, MEM_PRIVATE});
    t.add({0x140000000, 3 * kMiB, 0x140000000, MEM_IMAGE});
    t.add({0x50000000, 5 * kMiB, 0x50000000, MEM_MAPPED});
    mp::ProcessCounters counters;
    counters.read = true;
    counters.private_commit = 2048 * kMiB;
    counters.peak_commit = 4096 * kMiB;
    counters.working_set = 1024 * kMiB;
    counters.peak_working_set = 1536 * kMiB;
    counters.private_working_set = 512 * kMiB;
    const std::string line = mp::vmmap_text(12.0, 3.0, 100, counters, t, 640 * kMiB);
    std::fprintf(stderr, "%s\n", line.c_str());
    EXPECT(line.rfind("[rt] mem-probe vmmap t=12s (3 ms, 100 regions): private commit 2048 MiB (peak 4096), working "
                      "set 1024 MiB (private 512, peak 1536); committed 82 MiB by kind: title 64 (1 regions), "
                      "pe-private 8 (1 regions), pe-image 3 (1 regions), pe-section 5 (1 regions), vk-mapped 2 (1 "
                      "regions); reserved 640 MiB; largest private allocations MiB: 0x60000000 8",
                      0) == 0);
    counters.private_working_set = 0;
    EXPECT(contains(mp::vmmap_text(12.0, 3.0, 100, counters, t, 0), "(private n/a, peak 1536)"));
    EXPECT(contains(mp::vmmap_text(12.0, 3.0, 100, mp::ProcessCounters{}, t, 0),
                    "(3 ms, 100 regions): process counters n/a; committed 82 MiB"));
    return 0;
}

// This process: 64 MiB it commits shows up as PE private memory and in the commit charge, as title memory when a
// title range covers it, and a 16 MiB view of a pagefile-backed section as a PE section, or as vk-mapped memory when a
// vk_mapped range covers it.
int live() {
    const bool wine = GetProcAddress(GetModuleHandleW(L"ntdll.dll"), "wine_get_version") != nullptr;
    const mp::ProcessCounters counters_before = mp::process_counters();
    mp::VmTally before({}, {});
    EXPECT(walk(before) > 0);
    EXPECT(before.committed[mp::b_pe_image] > 0);  // this executable and its DLLs

    auto* block = static_cast<std::uint8_t*>(VirtualAlloc(nullptr, 64 * kMiB, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
    EXPECT(block);
    std::memset(block, 1, 64 * kMiB);  // in the working set too
    const std::uint64_t base = reinterpret_cast<std::uintptr_t>(block);
    mp::VmTally after({}, {});
    walk(after);
    const mp::ProcessCounters counters_after = mp::process_counters();
    std::fprintf(stderr, "pe-private %.0f -> %.0f MiB, private commit %.0f -> %.0f MiB, private working set %.0f MiB\n",
                 mp::mib(before.committed[mp::b_pe_private]), mp::mib(after.committed[mp::b_pe_private]),
                 mp::mib(counters_before.private_commit), mp::mib(counters_after.private_commit),
                 mp::mib(counters_after.private_working_set));
    EXPECT(after.committed[mp::b_pe_private] >= before.committed[mp::b_pe_private] + 64 * kMiB);
    EXPECT(counters_before.read && counters_after.read);
    // Wine's commit charge does not count what the process commits (MEASURED under Wine 11.0 on macOS: it stays put
    // after the 64 MiB), so like the private working set below it is checked on Windows only.
    if (!wine) EXPECT(counters_after.private_commit >= counters_before.private_commit + 64 * kMiB);
    EXPECT(counters_after.working_set > 0 && counters_after.peak_working_set >= counters_after.working_set);
    if (!wine && counters_after.private_working_set_reported) EXPECT(counters_after.private_working_set > 0);
    bool listed = false;
    for (const auto& [bytes, allocation] : after.largest_private(4))
        if (allocation == base && bytes == 64 * kMiB) listed = true;
    EXPECT(listed);

    mp::VmTally titled({{base, base + 64 * kMiB}}, {});
    walk(titled);
    EXPECT(titled.committed[mp::b_title] == 64 * kMiB && titled.regions[mp::b_title] == 1);

    HANDLE section = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE, 0, DWORD(16 * kMiB), nullptr);
    EXPECT(section);
    void* view = MapViewOfFile(section, FILE_MAP_ALL_ACCESS, 0, 0, 0);
    EXPECT(view);
    const std::uint64_t view_base = reinterpret_cast<std::uintptr_t>(view);
    mp::VmTally mapped({}, {});
    walk(mapped);
    EXPECT(mapped.committed[mp::b_pe_mapped] >= after.committed[mp::b_pe_mapped] + 16 * kMiB);
    mp::VmTally driver({}, {{view_base, view_base + 16 * kMiB}});
    walk(driver);
    EXPECT(driver.committed[mp::b_vk_mapped] == 16 * kMiB && driver.regions[mp::b_vk_mapped] == 1);

    UnmapViewOfFile(view);
    CloseHandle(section);
    VirtualFree(block, 0, MEM_RELEASE);
    return 0;
}

// A big, fragmented address space: 4 GiB reserved with one page committed every 64 KiB gives 65 536 committed regions
// between as many reserved ones. The walk counts them all and prints how long it took, then the whole line.
int big_address_space() {
    const std::uint64_t span = 4ull << 30, step = 64 * kKiB, page = 4 * kKiB;
    auto* base = static_cast<std::uint8_t*>(VirtualAlloc(nullptr, span, MEM_RESERVE, PAGE_READWRITE));
    EXPECT(base);
    for (std::uint64_t offset = 0; offset < span; offset += step)
        EXPECT(VirtualAlloc(base + offset, page, MEM_COMMIT, PAGE_READWRITE));
    mp::VmTally tally({}, {});
    std::uint64_t reserved = 0;
    const auto t0 = std::chrono::steady_clock::now();
    const std::uint64_t walked = mp::walk_address_space(tally, reserved);
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    std::fprintf(stderr, "walk: %llu regions in %.1f ms, %llu private regions, %.0f MiB reserved\n",
                 (unsigned long long)walked, ms, (unsigned long long)tally.regions[mp::b_pe_private], mp::mib(reserved));
    EXPECT(walked >= 2 * (span / step));
    EXPECT(tally.regions[mp::b_pe_private] >= span / step);
    EXPECT(reserved >= span - (span / step) * page);
    const auto largest = tally.largest_private(1);
    EXPECT(!largest.empty() && largest[0].second == reinterpret_cast<std::uintptr_t>(base) &&
           largest[0].first == (span / step) * page);
    std::fprintf(stderr, "%s\n", mp::sample_vmmap(0.0, {}, {}).c_str());
    VirtualFree(base, 0, MEM_RELEASE);
    return 0;
}

}  // namespace

int main() {
    if (int r = classifier()) return r;
    if (int r = text()) return r;
    if (int r = live()) return r;
    if (int r = big_address_space()) return r;
    std::fprintf(stderr, "mem_probe_vmmap: ok\n");
    return 0;
}
