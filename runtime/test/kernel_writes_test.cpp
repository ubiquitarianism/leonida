// leonida::kernel_writes::read_file (runtime/src/title_page_watch.h) into memory that is reserved until its first
// touch, as xmem maps the title's physical pages (SEC_RESERVE section views committed by a fault handler): the kernel's
// write does not fault such a page in, so the read fails with ERROR_NOACCESS unless the committer runs first. Then
// kernel_writes::hold, the pin a host import keeps over its pages: recorded with no opener set (no tracker yet), seen
// by for_each_pin until release(), untouched by a file read's pin ending over the same pages. No game data.
// the pins and the guards of the geometry watch (src/title_geometry_watch.h): a pin made with an OVERLAPPED that does
// not say pending yet lasts until its call returns; a host thread's write (kernel_writes::write) into a geometry tile
// the watch made read-only lands and counts as a write; and a tile armed while another guard changes one of its pages
// gives every page it protected back. No game data.
#include <windows.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

#include "../src/title_geometry_watch.h"
#include "../src/title_page_watch.h"

namespace {

int g_checks = 0;
int g_failures = 0;

void check(bool condition, const char* what) {
    ++g_checks;
    std::printf("%s %s\n", condition ? "PASS" : "FAIL", what);
    if (!condition) ++g_failures;
}

constexpr DWORD kViewBytes = 1 << 20;
constexpr DWORD kFileBytes = 256 * 1024;

std::uint8_t* g_view = nullptr;
int g_commits = 0;
int g_opens = 0;
std::uintptr_t g_opened_begin = 0, g_opened_end = 0;

void count_open(std::uintptr_t begin, std::uintptr_t end) {
    ++g_opens;
    g_opened_begin = begin;
    g_opened_end = end;
}

// The bytes of [begin, end) the pins cover.
std::size_t pinned_bytes(std::uintptr_t begin, std::uintptr_t end) {
    std::size_t bytes = 0;
    leonida::kernel_writes::for_each_pin(begin, end, [&](std::uintptr_t from, std::uintptr_t to) { bytes += to - from; });
    return bytes;
}

void check_holds() {
    namespace kw = leonida::kernel_writes;
    static std::uint8_t pages[3 * 4096];
    const auto begin = reinterpret_cast<std::uintptr_t>(pages);
    kw::set_opener(nullptr);
    check(pinned_bytes(begin, begin + sizeof(pages)) == 0, "no pin before the hold");
    check(kw::hold(pages, 0) == 0 && kw::hold(nullptr, 4096) == 0, "an empty hold pins nothing");
    // No opener: begin_write records nothing, a hold does (a tracker made later must still pass over the pages).
    check(kw::begin_write(pages, 4096, nullptr) == 0, "begin_write records nothing without an opener");
    const std::uint64_t first = kw::hold(pages, 2 * 4096);
    check(first != 0 && pinned_bytes(begin, begin + sizeof(pages)) == 2 * 4096,
          "a hold without an opener pins its range");
    check(pinned_bytes(begin + 2 * 4096, begin + sizeof(pages)) == 0, "and nothing past it");
    // With an opener: the range is opened once the pin is in.
    kw::set_opener(&count_open);
    const std::uint64_t second = kw::hold(pages + 4096, 2 * 4096);
    check(second != 0 && second != first && g_opens == 1 && g_opened_begin == begin + 4096 &&
              g_opened_end == begin + 3 * 4096,
          "a hold with an opener opens exactly its range");
    // A file read's pin over the same pages ends without ending the holds.
    const std::uint64_t read = kw::begin_write(pages, sizeof(pages), nullptr);
    check(read != 0 && pinned_bytes(begin, begin + sizeof(pages)) == 2 * 4096 + 2 * 4096 + sizeof(pages),
          "a file read's pin adds to the holds");
    kw::end_write(read, false);
    check(pinned_bytes(begin, begin + sizeof(pages)) == 4 * 4096, "its end leaves the holds");
    kw::end_write(first, true);  // as a pending overlapped read would: a hold has no OVERLAPPED, so it ends too
    check(pinned_bytes(begin, begin + 4096) == 0 && pinned_bytes(begin + 4096, begin + sizeof(pages)) == 2 * 4096,
          "ending one hold leaves the other");
    kw::release(second);
    kw::release(0);
    check(pinned_bytes(begin, begin + sizeof(pages)) == 0, "released, no pin is left");
    kw::set_opener(nullptr);
}

// Stands in for rt::xmem::commit_physical_range: commits the part of the test view the range covers.
bool commit_view(const void* address, std::size_t bytes) {
    ++g_commits;
    const auto* begin = static_cast<const std::uint8_t*>(address);
    if (begin < g_view || begin + bytes > g_view + kViewBytes) return true;  // not ours
    return VirtualAlloc(const_cast<std::uint8_t*>(begin), bytes, MEM_COMMIT, PAGE_READWRITE) != nullptr;
}

DWORD protection_of(const void* address) {
    MEMORY_BASIC_INFORMATION info{};
    return VirtualQuery(address, &info, sizeof(info)) && info.State == MEM_COMMIT ? info.Protect : 0;
}

// The D3D12 layer's opener as far as the geometry watch goes (open_shadow_pages): its guards over the range lift.
int g_opened = 0;
void open_geometry(std::uintptr_t from, std::uintptr_t to) {
    ++g_opened;
    leonida::geometry_writes::notify(from, to, leonida::geometry_writes::Event::open);
}

// A pin made with an OVERLAPPED whose Internal is not STATUS_PENDING yet (ReadFile stores that when it starts) is kept
// until the call returns; once it returned pending, it lasts until the OVERLAPPED says it completed.
void check_overlapped_pin() {
    namespace kw = leonida::kernel_writes;
    std::vector<std::uint8_t> bytes(0x1000);
    const auto begin = reinterpret_cast<std::uintptr_t>(bytes.data());
    const auto pinned = [&] {
        int n = 0;
        kw::for_each_pin(begin, begin + bytes.size(), [&](std::uintptr_t, std::uintptr_t) { ++n; });
        return n;
    };
    kw::set_opener([](std::uintptr_t, std::uintptr_t) {});
    OVERLAPPED at{};  // Internal 0: reads as completed
    const std::uint64_t pin = kw::begin_write(bytes.data(), bytes.size(), &at);
    check(pin != 0 && pinned() == 1, "an overlapped pin is kept while its call has not returned (Internal still 0)");
    at.Internal = static_cast<ULONG_PTR>(STATUS_PENDING);
    kw::end_write(pin, true);
    check(pinned() == 1, "returned pending: kept while the OVERLAPPED says pending");
    at.Internal = 0;
    check(pinned() == 0 && kw::state().count.load() == 0, "and dropped once it says completed");
    const std::uint64_t sync = kw::begin_write(bytes.data(), bytes.size(), &at);
    kw::end_write(sync, false);
    check(kw::state().count.load() == 0, "an overlapped pin whose call completed ends with the call");
    kw::set_opener(nullptr);
}

// A geometry tile over 64 KiB of the title's memory, armed: read-only. A host thread copies into it through
// kernel_writes::write (as a DirectStorage queue publishes a request): the opener lifts the watch, the copy lands, and
// the tile's version moved (the geometry copy is refilled at the next submission). The tile is not armed again while
// a write into it is pinned.
void check_geometry_write() {
    namespace gw = leonida::geometry_writes;
    constexpr std::size_t tile = gw::tile_size;
    auto* pages = static_cast<std::uint8_t*>(VirtualAlloc(nullptr, tile, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
    check(pages != nullptr, "a tile of memory");
    if (!pages) return;
    const auto va = reinterpret_cast<std::uintptr_t>(pages);
    gw::Cell* cell = gw::acquire(va);
    check(gw::arm(va, *cell) && (protection_of(pages + 0x5000) & 0xff) == PAGE_READONLY, "the armed tile is read-only");
    leonida::kernel_writes::set_opener(&open_geometry);
    const auto version = cell->version.load();
    std::vector<std::uint8_t> source(0x3000);
    for (std::size_t i = 0; i < source.size(); ++i) source[i] = static_cast<std::uint8_t>(i * 13 + 5);
    bool ok = false;
    std::thread host([&] { ok = leonida::kernel_writes::write(pages + 0x4800, source.data(), source.size()); });
    host.join();
    check(ok && std::memcmp(pages + 0x4800, source.data(), source.size()) == 0,
          "a host thread's write into the armed tile lands");
    check(g_opened == 1 && cell->version.load() != version, "the watch was opened first and the tile counts as written");
    check(gw::arm(va, *cell) && (protection_of(pages) & 0xff) == PAGE_READONLY, "armed again after the write");
    const std::uint64_t pin = leonida::kernel_writes::begin_write(pages + 0x100, 0x100, nullptr);
    check(!gw::arm(va, *cell), "not armed while a write into the tile is pinned");
    leonida::kernel_writes::end_write(pin, false);
    leonida::kernel_writes::set_opener(nullptr);
    gw::release(va);
    check((protection_of(pages) & 0xff) == PAGE_READWRITE && (protection_of(pages + tile - 1) & 0xff) == PAGE_READWRITE,
          "released: the tile is writable again");
    VirtualFree(pages, 0, MEM_RELEASE);
}

// The geometry watch arms a tile while another guard (a slot-133 shadow, under a lock of its own) makes the tile's
// first page read-only and writable again. VirtualProtect over the tile returns the first page's protection only:
// taken as every page's, a read-only answer left the others read-only for good once the watch reopened the tile. Every
// page but the first must be writable after each reopen.
void check_arm_race() {
    namespace gw = leonida::geometry_writes;
    constexpr std::size_t tile = gw::tile_size;
    auto* pages = static_cast<std::uint8_t*>(VirtualAlloc(nullptr, tile, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
    if (!pages) return;
    const auto va = reinterpret_cast<std::uintptr_t>(pages);
    gw::Cell* cell = gw::acquire(va);
    std::atomic<bool> stop{false};
    std::thread other([&] {
        DWORD old = 0;
        while (!stop.load(std::memory_order_relaxed)) {
            VirtualProtect(pages, 0x1000, PAGE_READONLY, &old);
            VirtualProtect(pages, 0x1000, PAGE_READWRITE, &old);
        }
    });
    int stuck = 0, armed = 0;
    const auto until = GetTickCount64() + 1500;
    while (GetTickCount64() < until && !stuck) {
        armed += gw::arm(va, *cell) ? 1 : 0;
        gw::notify(va, va + tile, gw::Event::open);
        for (std::size_t at = 0x1000; at < tile; at += 0x1000)
            if ((protection_of(pages + at) & 0xff) != PAGE_READWRITE) ++stuck;
    }
    stop = true;
    other.join();
    std::printf("   %d arms while another guard toggled the first page, %d pages left read-only\n", armed, stuck);
    check(armed > 0 && stuck == 0, "reopening an armed tile gives back every page it protected");
    gw::release(va);
    VirtualFree(pages, 0, MEM_RELEASE);
}

}  // namespace

int main() {
    check_overlapped_pin();
    check_geometry_write();
    check_arm_race();
    wchar_t dir[MAX_PATH], path[MAX_PATH];
    GetTempPathW(MAX_PATH, dir);
    GetTempFileNameW(dir, L"kwt", 0, path);
    std::vector<std::uint8_t> bytes(kFileBytes);
    for (DWORD i = 0; i < kFileBytes; ++i) bytes[i] = static_cast<std::uint8_t>(i * 7 + (i >> 9));
    HANDLE file = CreateFileW(path, GENERIC_READ | GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS,
                              FILE_ATTRIBUTE_TEMPORARY | FILE_FLAG_DELETE_ON_CLOSE, nullptr);
    DWORD written = 0;
    check(file != INVALID_HANDLE_VALUE && WriteFile(file, bytes.data(), kFileBytes, &written, nullptr) &&
              written == kFileBytes, "temporary file written");

    HANDLE section = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE | SEC_RESERVE, 0, kViewBytes,
                                        nullptr);
    g_view = section ? static_cast<std::uint8_t*>(MapViewOfFile(section, FILE_MAP_ALL_ACCESS, 0, 0, kViewBytes))
                     : nullptr;
    MEMORY_BASIC_INFORMATION info{};
    check(g_view && VirtualQuery(g_view, &info, sizeof(info)) && info.State == MEM_RESERVE,
          "the destination view is reserved, not committed");
    std::uint8_t* destination = g_view + 0x10000;

    // Without a committer: what the title saw (Windows answers ERROR_NOACCESS; another host may fault the page in).
    OVERLAPPED at{};
    DWORD got = 0;
    const BOOL plain = leonida::kernel_writes::read_file(file, destination, kFileBytes, &got, &at);
    std::printf("   without a committer: %s, error %lu\n", plain ? "read" : "failed", plain ? 0ul : GetLastError());

    leonida::kernel_writes::set_committer(&commit_view);
    at = {};
    got = 0;
    const BOOL ok = leonida::kernel_writes::read_file(file, destination, kFileBytes, &got, &at);
    check(ok && got == kFileBytes, "with the committer the read completes");
    check(ok && std::memcmp(destination, bytes.data(), kFileBytes) == 0, "and the destination holds the file's bytes");
    check(plain || g_commits == 1, "the committer ran once, for the failed read's retry");

    // A read that succeeds first time does not call the committer.
    const int before = g_commits;
    at = {};
    check(leonida::kernel_writes::read_file(file, destination, kFileBytes, &got, &at) && g_commits == before,
          "a read into committed memory does not call the committer");

    leonida::kernel_writes::set_committer(nullptr);
    check_holds();
    UnmapViewOfFile(g_view);
    CloseHandle(section);
    CloseHandle(file);
    std::printf("%d checks, %d failures\n", g_checks, g_failures);
    return g_failures ? 1 : 0;
}
