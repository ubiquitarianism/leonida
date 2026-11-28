// Physical pages alias and start zeroed on both hosts. Under Wine their first touch and explicit commits
// must leave the memory-map generation alone. Time the same 1,000 queries in each section mode.
#include <windows.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

#include "shims.h"
#include "xmem.h"
#include "../src/title_page_watch.h"

namespace {
constexpr size_t page = 64 * 1024;
constexpr size_t pool = size_t{1} << 30;
constexpr size_t stride = 16 * page;
int checks = 0;
int failures = 0;

bool check(bool ok, const char* name) {
    ++checks;
    if (!ok) {
        ++failures;
        std::printf("FAIL %s\n", name);
    }
    return ok;
}

double query_median(uint8_t* base) {
    using Clock = std::chrono::steady_clock;
    std::mt19937 random(12345);
    std::array<double, 1000> times{};
    for (double& time : times) {
        const auto address = base + random() % pool;
        MEMORY_BASIC_INFORMATION info{};
        const auto start = Clock::now();
        const auto bytes = VirtualQuery(address, &info, sizeof(info));
        const auto end = Clock::now();
        if (bytes != sizeof(info) || info.Type != MEM_MAPPED) return -1;
        time = std::chrono::duration<double, std::micro>(end - start).count();
    }
    std::sort(times.begin(), times.end());
    return (times[499] + times[500]) / 2;
}

void measure_reserved() {
    HANDLE section = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE | SEC_RESERVE, 0, pool, nullptr);
    if (!section) return;
    auto* base = static_cast<uint8_t*>(MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, pool));
    if (base) {
        bool ready = true;
        for (size_t at = 0; at < pool; at += stride) {
            if (!VirtualAlloc(base + at, page, MEM_COMMIT, PAGE_READWRITE)) {
                ready = false;
                break;
            }
            base[at] = 1;
        }
        const double median = ready ? query_median(base) : -1;
        if (median >= 0) std::printf("SEC_RESERVE (one raw view) VirtualQuery median %.3f us\n", median);
        UnmapViewOfFile(base);
    }
    CloseHandle(section);
}

void run() {
    const bool wine = GetProcAddress(GetModuleHandleW(L"ntdll.dll"), "wine_get_version") != nullptr;
    const size_t bytes = wine ? pool : 2 * page;
    using Reserve = void*(WINAPI*)(void*, SIZE_T, DWORD, ULONGLONG, DWORD);
    using Allocate = BOOL(WINAPI*)(uint64_t, uint64_t*, uint64_t*);
    using Map = void*(WINAPI*)(void*, uint64_t, const uint64_t*);
    using Free = BOOL(WINAPI*)(uint64_t, const uint64_t*);
    const auto reserve = reinterpret_cast<Reserve>(rt::shims::find("xmem.dll", "XMemVirtualAlloc"));
    const auto allocate = reinterpret_cast<Allocate>(rt::shims::find("xmem.dll", "XMemAllocatePhysicalPages"));
    const auto map = reinterpret_cast<Map>(rt::shims::find("xmem.dll", "XMemMapPhysicalPages"));
    const auto free_pages = reinterpret_cast<Free>(rt::shims::find("xmem.dll", "XMemFreePhysicalPages"));
    if (!check(reserve && allocate && map && free_pages, "registered physical-page API")) return;

    constexpr DWORD window = MEM_RESERVE | 0x20400000;  // XMemVirtualAlloc 64 KiB pages
    auto* base = static_cast<uint8_t*>(reserve(nullptr, bytes, window, 0x23, PAGE_READWRITE));
    auto* alias = static_cast<uint8_t*>(reserve(nullptr, 2 * page, window, 0x23, PAGE_READWRITE));
    std::vector<uint64_t> pages(bytes / page);
    uint64_t count = pages.size();
    if (!check(base && alias && base != alias, "reserve distinct windows") ||
        !check(allocate(0x280, &count, pages.data()) && count == pages.size(), "allocate physical pages") ||
        !check(map(base, count, pages.data()) == base && map(alias, 2, pages.data()) == alias, "map both windows")) return;

    namespace memory = leonida::title_memory_map;
    if (wine) {
        MEMORY_BASIC_INFORMATION info{};
        check(VirtualQuery(base, &info, sizeof(info)) == sizeof(info) &&
              info.Type == MEM_MAPPED && info.State == MEM_COMMIT, "untouched view starts committed");
    }
    const auto before = memory::generation();
    volatile uint8_t* a = base;
    volatile uint8_t* b = alias;
    check(a[0] == 0, "untouched page reads zero");
    a[page] = 0x5a;
    check(b[page] == 0x5a, "alias sees first write");
    if (wine) {
        check(memory::generation() == before, "first read and write preserve generation");
        check(rt::xmem::commit_physical_range(base, bytes) && rt::xmem::commit_physical_pages(pages.data(), count)
              && memory::generation() == before, "explicit commits preserve generation");
        for (size_t at = 0; at < bytes; at += stride) a[at] = 1;
        const double median = query_median(base);
        std::printf("SEC_COMMIT (xmem pool) VirtualQuery median %.3f us\n", median);
        check(median >= 0 && median < 20, "1,000 successful mapped queries with median below 20 us");
    }
    check(map(alias, 2, nullptr) == alias && map(base, count, nullptr) == base, "unmap both windows");
    check(free_pages(count, pages.data()), "free physical pages");
    if (wine) measure_reserved();
}
}  // namespace

int main() {
    run();
    std::printf("xmem_wine_commit_test: %d checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
