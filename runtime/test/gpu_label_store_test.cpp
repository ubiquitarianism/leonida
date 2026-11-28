// leonida::label::store_value (runtime/src/vulkan_gpu_labels.h): where a completed GPU label may be stored in the
// title's memory, and what it does with a page the D3D12 layer keeps read-only while it watches for the title's writes
// (runtime/src/title_page_watch.h). The layer is stood in for by an opener that makes its "watched" pages writable
// again, as DeviceState's opener does. CPU only: no Vulkan device, no GPU. No game data.
#include "../src/vulkan_gpu_labels.h"

#include <cstdint>
#include <cstdio>

namespace {

int g_checks = 0;
int g_failures = 0;

void check(bool condition, const char* what) {
    ++g_checks;
    std::printf("%s %s\n", condition ? "PASS" : "FAIL", what);
    if (!condition) ++g_failures;
}

constexpr std::size_t kPage = 4096;

std::uintptr_t g_watched_begin = 0, g_watched_end = 0;  // the pages the stand-in layer made read-only
int g_opened = 0;
std::uintptr_t g_asked_begin = 0, g_asked_end = 0;

// Stands in for the layer's opener (open_shadow_pages): only pages it made read-only itself become writable.
void open_watched(std::uintptr_t begin, std::uintptr_t end) {
    ++g_opened;
    g_asked_begin = begin;
    g_asked_end = end;
    for (std::uintptr_t page = begin & ~(kPage - 1); page < end; page += kPage) {
        if (page < g_watched_begin || page >= g_watched_end) continue;
        DWORD old = 0;
        VirtualProtect(reinterpret_cast<void*>(page), kPage, PAGE_READWRITE, &old);
    }
}

DWORD protection(const void* p) {
    MEMORY_BASIC_INFORMATION mbi{};
    return VirtualQuery(p, &mbi, sizeof(mbi)) ? mbi.Protect : 0;
}

bool protect(void* p, DWORD how) {
    DWORD old = 0;
    return VirtualProtect(p, kPage, how, &old) != FALSE;
}

}  // namespace

int main() {
    namespace label = leonida::label;
    namespace writes = leonida::kernel_writes;
    auto* const base = static_cast<std::uint8_t*>(VirtualAlloc(nullptr, 4 * kPage, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
    if (!base) return 1;
    const auto address = reinterpret_cast<std::uintptr_t>(base);
    label::memory().track(address, 4 * kPage, address, false);
    auto* const wide = reinterpret_cast<std::uint64_t*>(base + 0x40);
    auto* const narrow = reinterpret_cast<std::uint32_t*>(base + kPage + 0x10);
    const label::Target wide_target = label::memory().capture(reinterpret_cast<std::uintptr_t>(wide), 8);
    const label::Target narrow_target = label::memory().capture(reinterpret_cast<std::uintptr_t>(narrow), 4);
    check(wide_target.bytes == 8 && narrow_target.bytes == 4, "labels are captured in a tracked allocation");

    // 1. A writable page: stored, with no opener asked.
    writes::set_opener(&open_watched);
    check(label::store_value(wide_target, 0x1122334455667788ull) && *wide == 0x1122334455667788ull && g_opened == 0,
          "an 8-byte label is stored into a read-write page");
    check(label::store_value(narrow_target, 0xA1B2C3D4u) && *narrow == 0xA1B2C3D4u && narrow[1] == 0 && g_opened == 0,
          "a 4-byte label is stored and leaves the next word alone");

    // 2. A page the layer watches (read-only until written): it is opened like a kernel write and the label lands.
    g_watched_begin = address;
    g_watched_end = address + kPage;
    check(protect(base, PAGE_READONLY), "the first page is made read-only, as a watched page is");
    check(label::store_value(wide_target, 0x0102030405060708ull), "a label for a watched read-only page is stored");
    check(*wide == 0x0102030405060708ull, "the value is in the page");
    check(g_opened == 1 && g_asked_begin == reinterpret_cast<std::uintptr_t>(wide) && g_asked_end == g_asked_begin + 8,
          "the layer was asked to open exactly the label's bytes, once");
    check(protection(base) == PAGE_READWRITE, "the page is writable again, so the layer copies it at its next refresh");
    check(writes::state().count.load() == 0, "the write's pin has ended");

    // 3. A page that stays read-only when the layer is asked (the title's own protection): dropped, no fault.
    check(protect(base + kPage, PAGE_READONLY), "the second page is made read-only by the title");
    const int opened = g_opened;
    check(!label::store_value(narrow_target, 0x55555555u) && *narrow == 0xA1B2C3D4u,
          "a label for the title's own read-only page is dropped and the old value stays");
    check(g_opened == opened + 1 && writes::state().count.load() == 0, "the layer was asked and the pin has ended");
    check(protection(base + kPage) == PAGE_READONLY, "the title's protection is untouched");

    // 4. No layer at all: a read-only page is dropped without asking anyone.
    writes::set_opener(nullptr);
    check(protect(base, PAGE_READONLY), "the first page is read-only again");
    check(!label::store_value(wide_target, 0x9999999999999999ull) && *wide == 0x0102030405060708ull,
          "with no opener a read-only page drops the label");
    writes::set_opener(&open_watched);
    check(label::store_value(wide_target, 0x9999999999999999ull) && *wide == 0x9999999999999999ull,
          "with the opener back the same label is stored");

    // 5. Guard pages, no-access pages and decommitted pages are never written, opener or not.
    check(protect(base + 2 * kPage, PAGE_READWRITE | PAGE_GUARD), "the third page is a guard page");
    const label::Target guarded = label::memory().capture(address + 2 * kPage + 8, 8);
    check(!label::store_value(guarded, 1) && (protection(base + 2 * kPage) & PAGE_GUARD),
          "a label for a guard page is dropped and the guard is still armed");
    check(protect(base + 3 * kPage, PAGE_NOACCESS), "the fourth page is no-access");
    const label::Target hidden = label::memory().capture(address + 3 * kPage + 8, 8);
    check(!label::store_value(hidden, 1), "a label for a no-access page is dropped");
    check(VirtualFree(base + 3 * kPage, kPage, MEM_DECOMMIT) != FALSE, "the fourth page is decommitted");
    check(!label::store_value(hidden, 1), "a label for a decommitted page is dropped");

    // 6. A released allocation: the captured generation is gone, so nothing is stored whatever is mapped there now.
    check(label::memory().release_pages(address, 0, true, [&] { return VirtualFree(base, 0, MEM_RELEASE) != FALSE; }),
          "the allocation is released");
    check(!label::store_value(wide_target, 2), "a label captured before the release is dropped");
    writes::set_opener(nullptr);

    std::printf("gpu_label_store_test: %s (%d checks, %d failures)\n", g_failures ? "FAILED" : "ok", g_checks,
                g_failures);
    return g_failures ? 1 : 0;
}
