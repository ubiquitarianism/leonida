// leonida::title_write_watch (runtime/src/title_page_watch.h): the record of the title's write-watched reservations, and
// the journal that shares the kernel's answers between the trackers over the same pages (an asking clears the kernel's
// bits for everyone). No game data, no GPU.
//   1. two readers over one page both see a write, though the first one's asking cleared the kernel's bit;
//   2. a reader isn't told again what it saw, and a write after its asking is in its next one;
//   3. a kernel write (a plain ReadFile, no file shim) into a write-watched page is in the answer, and doesn't fail;
//   4. memory the kernel doesn't write-watch (a section view) is refused;
//   5. a reader that joins another's pages (merge_reader) sees what the other had not seen yet;
//   6. the record: add, contains, for_each_in split by reservation, remove.
#include <windows.h>

#include <cstdint>
#include <cstdio>
#include <set>
#include <utility>
#include <vector>

#include "../src/title_page_watch.h"

namespace {

namespace ww = leonida::title_write_watch;
using Runs = std::vector<std::pair<std::uintptr_t, std::uintptr_t>>;

int g_checks = 0;
int g_failures = 0;

void check(bool condition, const char* what) {
    ++g_checks;
    std::printf("%s %s\n", condition ? "PASS" : "FAIL", what);
    if (!condition) ++g_failures;
}

// One asking for `reader`: the pages it is told were written, and whether a run was refused.
struct Answer {
    std::set<std::uintptr_t> pages;
    bool refused = false;
};
Answer ask(std::uint64_t reader, const Runs& runs) {
    Answer answer;
    ww::ask(
        reader, runs,
        [&](std::uintptr_t from, std::uintptr_t to) {
            for (std::uintptr_t p = from; p < to; p += ww::kPage) answer.pages.insert(p);
        },
        [&](std::uintptr_t, std::uintptr_t) { answer.refused = true; });
    return answer;
}

}  // namespace

int main() {
    constexpr SIZE_T kBytes = 16 * ww::kPage;
    auto* pages = static_cast<std::uint8_t*>(
        VirtualAlloc(nullptr, kBytes, MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH, PAGE_READWRITE));
    check(pages != nullptr, "a write-watched reservation");
    if (!pages) return 1;
    const auto base = reinterpret_cast<std::uintptr_t>(pages);
    ww::add(base, kBytes);
    const Runs all{{base, base + kBytes}};
    const std::uint64_t a = ww::open_reader(), b = ww::open_reader();

    // 1. Both readers see one write.
    pages[3 * ww::kPage + 5] = 1;
    Answer first = ask(a, all);
    check(!first.refused && first.pages == std::set<std::uintptr_t>{base + 3 * ww::kPage},
          "the first reader is told the written page");
    Answer second = ask(b, all);
    check(!second.refused && second.pages == std::set<std::uintptr_t>{base + 3 * ww::kPage},
          "the second reader is told it too, though the first asking cleared the kernel's bit");

    // 2. Nothing new for either; a later write reaches both, and only that one.
    check(ask(a, all).pages.empty() && ask(b, all).pages.empty(), "a reader isn't told again what it saw");
    pages[7 * ww::kPage] = 2;
    pages[8 * ww::kPage] = 2;
    first = ask(a, Runs{{base + 4 * ww::kPage, base + kBytes}});
    check(first.pages == std::set<std::uintptr_t>{base + 7 * ww::kPage, base + 8 * ww::kPage},
          "a later write is in the next asking (asked over part of the reservation)");
    check(ask(b, all).pages == first.pages, "and in the other reader's next one");

    // 3. A kernel write: a plain buffered ReadFile into the watched pages succeeds and is in the answer.
    wchar_t directory[MAX_PATH]{}, path[MAX_PATH]{};
    GetTempPathW(MAX_PATH, directory);
    GetTempFileNameW(directory, L"lww", 0, path);
    HANDLE file = CreateFileW(path, GENERIC_READ | GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS,
                              FILE_ATTRIBUTE_TEMPORARY | FILE_FLAG_DELETE_ON_CLOSE, nullptr);
    std::vector<std::uint8_t> bytes(2 * ww::kPage);
    for (std::size_t i = 0; i < bytes.size(); ++i) bytes[i] = std::uint8_t(i * 11 + 3);
    DWORD done = 0;
    check(file != INVALID_HANDLE_VALUE && WriteFile(file, bytes.data(), DWORD(bytes.size()), &done, nullptr) &&
              done == bytes.size(),
          "temporary file written");
    SetFilePointer(file, 0, nullptr, FILE_BEGIN);
    done = 0;
    const BOOL read = ::ReadFile(file, pages + 10 * ww::kPage + 0x800, DWORD(bytes.size()), &done, nullptr);
    CloseHandle(file);
    check(read && done == bytes.size(), "a plain ReadFile into write-watched pages succeeds (they stay writable)");
    const std::set<std::uintptr_t> kernel_pages{base + 10 * ww::kPage, base + 11 * ww::kPage, base + 12 * ww::kPage};
    check(ask(a, all).pages == kernel_pages, "the kernel's write is in the answer, all three pages it touched");
    check(ask(b, all).pages == kernel_pages, "for the other reader too");

    // 4. A section view is not write-watched: refused, and nothing reported.
    HANDLE section = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE, 0, 0x10000, nullptr);
    auto* view = section ? static_cast<std::uint8_t*>(MapViewOfFile(section, FILE_MAP_ALL_ACCESS, 0, 0, 0x10000))
                         : nullptr;
    check(view != nullptr, "a section view");
    if (view) {
        view[0] = 1;
        const auto at = reinterpret_cast<std::uintptr_t>(view);
        const Answer refused = ask(a, Runs{{at, at + 0x10000}});
        check(refused.refused && refused.pages.empty(), "the kernel refuses a section view");
        UnmapViewOfFile(view);
    }
    if (section) CloseHandle(section);

    // 5. A reader that takes over another's pages asks from the older of the two lastings.
    pages[0] = 3;
    check(ask(a, all).pages == std::set<std::uintptr_t>{base}, "reader a sees a write reader b hasn't asked about");
    const std::uint64_t c = ww::open_reader();
    check(ask(c, all).pages.empty(), "a new reader is told nothing written before it");
    ww::merge_reader(c, b);
    check(ask(c, all).pages == std::set<std::uintptr_t>{base}, "having joined b's pages, it is told what b missed");
    ww::close_reader(b);
    ww::close_reader(c);

    // 6. The record of reservations.
    const auto other = reinterpret_cast<std::uintptr_t>(
        VirtualAlloc(nullptr, 4 * ww::kPage, MEM_RESERVE | MEM_WRITE_WATCH, PAGE_READWRITE));
    ww::add(other, 4 * ww::kPage);
    const std::uintptr_t lo = (std::min)(base, other), hi = (std::max)(base + kBytes, other + 4 * ww::kPage);
    Runs split;
    ww::for_each_in(lo, hi, [&](std::uintptr_t from, std::uintptr_t to) { split.emplace_back(from, to); });
    check(split.size() == 2, "for_each_in gives one run per reservation");
    check(ww::contains(base + kBytes - 1) && !ww::contains(base + kBytes + 0x10000 * 64),
          "contains: inside a reservation, and not far past it");
    const auto generation = ww::generation();
    ww::remove(other + ww::kPage);  // any address inside it
    check(!ww::contains(other) && ww::contains(base) && ww::generation() != generation,
          "remove forgets the reservation holding the address, and only it");
    ww::remove(base);
    ww::close_reader(a);
    VirtualFree(reinterpret_cast<void*>(other), 0, MEM_RELEASE);
    VirtualFree(pages, 0, MEM_RELEASE);

    std::printf("title_write_watch_test: %s (%d checks, %d failures)\n", g_failures ? "FAILED" : "OK", g_checks,
                g_failures);
    return g_failures ? 1 : 0;
}
