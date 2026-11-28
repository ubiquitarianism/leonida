// xmem.dll: the GDK's tracked allocator. Mapped onto the CRT's aligned heap and VirtualQueryEx.
//
// The measured ABI (docs from the clean-room spec) shows four imports:
//   XMemAlloc(size, attributes)
//   XMemFree(ptr, attributes)
//   XMemVirtualQuery(process, address, out64)
//   XMemGetWorkingSetStatistics(kind, out)   (kind 0x20000001 at all seven FSR call sites)
//
// ASSUMPTION: the attribute word is not decoded by the measured call sites (one passes 0x5e50000). Until a call
// site shows the alignment/type fields, every block is 64-byte aligned (covers cache lines and SIMD) and a size of
// zero is rounded up to one byte so the result is a stable, non-null pointer that XMemFree can release.
//
// The default allocators, the physical-page functions and XMemIsToolsMemAvailable follow argument shapes measured
// locally at a title's call sites (each function says which). The page functions are built on the Windows placeholder
// APIs (VirtualAlloc2, MapViewOfFile3), whose flags the title's own XMemVirtualAlloc calls already use.
#include <windows.h>
#include <winioctl.h>
#include <filesystem>
#include <intrin.h>
#include <malloc.h>
#include <psapi.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "shims.h"
#include "startup_trace.h"
#include "xmem.h"
#include "placed_image_map_note.h"
#include "../src/dstorage_trace.h"
#include "../src/gpu_label_memory.h"
#include "../src/title_page_watch.h"
#include "../src/title_geometry_watch.h"
#include "../src/title_va_alias.h"
#include "../src/value_write_sink.h"

namespace {

constexpr size_t kAlign = 64;
constexpr size_t kNoSize = static_cast<size_t>(-1);  // XMemFree has no size field

std::string caller_field(const void* caller) {
    return ",\"caller\":" + rt::startup_trace::quote(rt::startup_trace::where(reinterpret_cast<uint64_t>(caller)));
}

// gta6rt --trace FILE: the first XMemAlloc, XMemFree, XMemAllocDefault, XMemFreeDefault and XMemVirtualAlloc call for
// each distinct raw attribute word go to the startup trace as an "xmem_attributes" event (api, the word, the size for an
// allocation, the caller as module+RVA). The bits of the attribute word are not known yet; this records them from a
// real run without logging every allocation, which would fill the file. Off: one relaxed load per call.
void note_attributes(const char* api, uint64_t attributes, size_t size, const void* caller) {
    if (!rt::startup_trace::enabled()) return;
    static std::mutex mutex;
    static std::set<std::pair<std::string, uint64_t>> seen;
    {
        const std::lock_guard lock(mutex);
        if (!seen.insert({api, attributes}).second) return;
    }
    std::string fields = std::string("\"api\":\"") + api +
                         "\",\"attributes\":" + rt::startup_trace::quote(rt::startup_trace::hex(attributes));
    if (size != kNoSize) fields += ",\"size\":" + std::to_string(size);
    fields += caller_field(caller);
    rt::startup_trace::event("xmem_attributes", fields);
}

// The title's XMemAlloc blocks come from a heap of their own instead of the CRT heap (LEONIDA_TITLE_HEAP=0: not). Slot-133
// imports pin title memory page by page for the GPU, and on the CRT heap a pinned page also holds the runtime's own
// objects (g_pages' nodes among them): a GPU store past a buffer's end, or into a block freed while its import still
// lived, then corrupted the runtime (speed-30/41/49-52: g_pages found corrupt under g_phys_mutex). Each block keeps its
// size and the heap allocation it sits in just below it.
struct TitleHeapHeader {
    void* raw;
    size_t size;
};
HANDLE title_heap() {
    static const HANDLE heap = [] {
        // Set LEONIDA_TITLE_HEAP=0 to use the CRT heap.
        const char* v = std::getenv("LEONIDA_TITLE_HEAP");
        if (v && std::strcmp(v, "0") == 0) return HANDLE(nullptr);
        const HANDLE h = HeapCreate(0, 0, 0);
        std::fprintf(stderr, "[rt] xmem: XMemAlloc blocks come from a heap of their own (%p)\n", h);
        return h;
    }();
    return heap;
}
void* allocate_block(size_t size) {
    size = size ? size : 1;
    const HANDLE heap = title_heap();
    if (!heap) return _aligned_malloc(size, kAlign);
    const size_t extra = kAlign + sizeof(TitleHeapHeader);
    if (size > SIZE_MAX - extra) return nullptr;
    void* raw = HeapAlloc(heap, 0, size + extra);
    if (!raw) return nullptr;
    const uintptr_t user = (reinterpret_cast<uintptr_t>(raw) + sizeof(TitleHeapHeader) + kAlign - 1) & ~(kAlign - 1);
    auto* header = reinterpret_cast<TitleHeapHeader*>(user) - 1;
    header->raw = raw;
    header->size = size;
    return reinterpret_cast<void*>(user);
}
size_t block_size(void* p) {
    if (!p) return 0;
    if (!title_heap()) return _aligned_msize(p, kAlign, 0);
    return (reinterpret_cast<TitleHeapHeader*>(p) - 1)->size;
}
void free_block(void* p) {
    if (!p) return;
    const HANDLE heap = title_heap();
    if (!heap) return _aligned_free(p);
    HeapFree(heap, 0, (reinterpret_cast<TitleHeapHeader*>(p) - 1)->raw);
}

// Every block and range the title gets here is recorded as the title's memory (src/title_va_alias.h): a shader
// address into it can be given a GPU copy, and an address anywhere else cannot.
void track_label_data(void* p,std::size_t bytes,bool heap) {
    if(!p)return; MEMORY_BASIC_INFORMATION info{};
    if(VirtualQuery(p,&info,sizeof(info)) && info.Type==MEM_PRIVATE)
        leonida::label::memory().track(reinterpret_cast<std::uintptr_t>(p),bytes,reinterpret_cast<std::uintptr_t>(info.AllocationBase),heap);
}

// A new block is no change to the title memory map: the heap's pages under it were committed and readable already, or
// were uncommitted (which the readers query again at every look) after a free that the map recorded.
void* title_block(size_t size) {
    void* p = allocate_block(size);
    leonida::title_va_aliases().add_title_memory(reinterpret_cast<uintptr_t>(p), size ? size : 1);
    track_label_data(p,size?size:1,true);
    return p;
}

void free_title_block(void* p) {
    if(p)leonida::label::memory().release_heap(reinterpret_cast<std::uintptr_t>(p));
    if (p) leonida::title_va_aliases().remove_title_memory(reinterpret_cast<uintptr_t>(p));
    const size_t size = block_size(p);
    free_block(p);
    if (p) {
        const auto from = reinterpret_cast<uintptr_t>(p);
        const size_t bytes = size ? size : 1;
        leonida::title_memory_map::changed(p, bytes);
        leonida::geometry_writes::note_released(from, from + bytes);
    }
}

void* __stdcall XMemAlloc(size_t size, uint64_t attributes) {
    note_attributes("XMemAlloc", attributes, size, _ReturnAddress());
    return title_block(size);
}

void __stdcall XMemFree(void* p, uint64_t attributes) {
    note_attributes("XMemFree", attributes, kNoSize, _ReturnAddress());
    free_title_block(p);
}

// XMemAllocDefault / XMemFreeDefault: the GDK's own allocator behind XMemAlloc and XMemFree, which a title that hooks
// XMemAlloc forwards to. The measured title reaches each through one wrapper that is a bare tail jump to the import,
// so they take exactly XMemAlloc's and XMemFree's arguments. They share the heap: a block from either allocator may be
// released by either free.
void* __stdcall XMemAllocDefault(size_t size, uint64_t attributes) {
    note_attributes("XMemAllocDefault", attributes, size, _ReturnAddress());
    return title_block(size);
}

void __stdcall XMemFreeDefault(void* p, uint64_t attributes) {
    note_attributes("XMemFreeDefault", attributes, kNoSize, _ReturnAddress());
    free_title_block(p);
}

// BOOL XMemIsToolsMemAvailable(void). The measured title calls it from one helper, which six out-of-memory paths reach
// only after an allocation has failed; the helper breaks into the debugger when the result is zero. Tools memory is
// the extra memory of a development kit. A PC has none, so the answer is FALSE, and the title's own breakpoint marks
// where it ran out of memory.
BOOL WINAPI XMemIsToolsMemAvailable() {
    static std::atomic<bool> warned{false};
    if (!warned.exchange(true, std::memory_order_relaxed)) {
        std::fprintf(stderr, "[rt] xmem: XMemIsToolsMemAvailable -> FALSE (no devkit tools memory; logged once)\n");
    }
    return FALSE;
}

// Measured call sites pass a zeroed 64-byte structure. Only the MEMORY_BASIC_INFORMATION-compatible prefix is
// known: callers read at least the base and size fields. Fill that prefix from VirtualQueryEx and zero the rest.
constexpr size_t kVirtualQueryOutSize = 64;
static_assert(sizeof(MEMORY_BASIC_INFORMATION) <= kVirtualQueryOutSize,
              "the 64-byte XMemVirtualQuery output must hold MEMORY_BASIC_INFORMATION");

// Returns the number of bytes written (the VirtualQuery convention) or 0 on failure. The exact return type and the
// meaning of every output field are not fully known; callers in the package only test for success.
SIZE_T __stdcall XMemVirtualQuery(HANDLE process, const void* address, void* out) {
    if (!out) return 0;
    std::memset(out, 0, kVirtualQueryOutSize);
    if (!address) return 0;

    static std::atomic<bool> warned{false};
    if (!warned.exchange(true, std::memory_order_relaxed)) {
        std::fprintf(stderr,
                     "[rt] xmem: XMemVirtualQuery: fields after MEMORY_BASIC_INFORMATION are unknown; "
                     "the 64-byte output is zero-filled there\n");
    }

    // A null handle means the current process; GetCurrentProcess() is the (HANDLE)-1 pseudo handle.
    HANDLE target = process ? process : GetCurrentProcess();
    MEMORY_BASIC_INFORMATION info{};
    SIZE_T written = VirtualQueryEx(target, address, &info, sizeof(info));
    if (!written) return 0;

    std::memcpy(out, &info, written < kVirtualQueryOutSize ? written : kVirtualQueryOutSize);
    return written;
}

// BOOL XMemGetWorkingSetStatistics(UINT32 kind, XMEM_WORKING_SET_STATISTICS* stats). Measured at all seven call sites
// in amd_fidelityfx_upscaler_xbox.dll (1.0.2.1; calls at RVAs 0x26b3, 0x2ea1, 0x39a9, 0x7452, 0x74d2, 0x7568,
// 0x75b1): ECX is the constant 0x20000001 (its meaning is not known) and RDX points at a stack buffer (0x80 bytes
// where the frame shows it). The caller reads a 64-bit size at +8 and treats a zero return as "no data". Taking the
// pointer from the first argument wrote to address 0x20000001.
// The measured FSR sites establish argument shape, kind and the 64-bit working-set read at +8; +0's title-total memory
// limit interpretation is inferred from other Series implementations (xcl, xbox-emulator) and title engine telemetry
// sizing memory pools (RT-220). Note that if GlobalMemoryStatusEx reports 0 physical memory, XMem falls back to the
// 12 GiB title limit, whereas XSystemGetAppMemoryUsageLimit remains 0.
BOOL __stdcall XMemGetWorkingSetStatistics(uint32_t /*kind*/, void* stats) {
    if (!stats) {
        static std::atomic<bool> warned{false};
        if (!warned.exchange(true, std::memory_order_relaxed)) {
            std::fprintf(stderr, "[rt] xmem: XMemGetWorkingSetStatistics(nullptr) -> FALSE\n");
        }
        return FALSE;
    }

    PROCESS_MEMORY_COUNTERS pmc{};
    if (!GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) return FALSE;

    MEMORYSTATUSEX ms{sizeof(ms)};
    uint64_t total_memory = 12ULL * 1024 * 1024 * 1024;
    if (GlobalMemoryStatusEx(&ms) && ms.ullTotalPhys != 0) {
        total_memory = ms.ullTotalPhys;
    }
    // Optional override (matching XSystemGetAppMemoryUsageLimit) so memory-constrained hosts can cap title pools.
    char env[32]{};
    const DWORD env_n = GetEnvironmentVariableA("LEONIDA_TITLE_MEMORY_MB", env, sizeof(env));
    if (env_n > 0 && env_n < sizeof(env) && env[0] != '-' && env[0] != '+') {
        errno = 0;
        char* end = nullptr;
        unsigned long long mib = std::strtoull(env, &end, 10);
        if (errno == 0 && end != env && *end == '\0' && mib > 0) {
            if (mib < 2048ULL) mib = 2048ULL;
            if (mib > 65536ULL) mib = 65536ULL;
            total_memory = static_cast<uint64_t>(mib) * 1024ULL * 1024ULL;
        }
    }

    uint8_t* bytes = static_cast<uint8_t*>(stats);
    std::memset(bytes, 0, 16);
    std::memcpy(bytes + 0, &total_memory, sizeof(total_memory));
    uint64_t working_set = pmc.WorkingSetSize;
    std::memcpy(bytes + 8, &working_set, sizeof(working_set));
    return TRUE;
}

// The placeholder APIs (Windows 10 1803 and later) come from kernelbase.dll at run time, so neither gta6rt nor the
// tests link onecore.lib. A host without them fails the calls that need them with ERROR_NOT_SUPPORTED.
using VirtualAlloc2Fn = PVOID(WINAPI*)(HANDLE, PVOID, SIZE_T, ULONG, ULONG, MEM_EXTENDED_PARAMETER*, ULONG);
using MapViewOfFile3Fn = PVOID(WINAPI*)(HANDLE, HANDLE, PVOID, ULONG64, SIZE_T, ULONG, ULONG, MEM_EXTENDED_PARAMETER*,
                                        ULONG);
using UnmapViewOfFile2Fn = BOOL(WINAPI*)(HANDLE, PVOID, ULONG);

struct PlaceholderApi {
    VirtualAlloc2Fn virtual_alloc2 = nullptr;
    MapViewOfFile3Fn map_view3 = nullptr;
    UnmapViewOfFile2Fn unmap_view2 = nullptr;
    bool available() const { return virtual_alloc2 && map_view3 && unmap_view2; }
};

const PlaceholderApi& placeholder_api() {
    static const PlaceholderApi api = [] {
        PlaceholderApi resolved;
        if (const HMODULE kernelbase = GetModuleHandleW(L"kernelbase.dll")) {
            resolved.virtual_alloc2 = reinterpret_cast<VirtualAlloc2Fn>(GetProcAddress(kernelbase, "VirtualAlloc2"));
            resolved.map_view3 = reinterpret_cast<MapViewOfFile3Fn>(GetProcAddress(kernelbase, "MapViewOfFile3"));
            resolved.unmap_view2 =
                reinterpret_cast<UnmapViewOfFile2Fn>(GetProcAddress(kernelbase, "UnmapViewOfFile2"));
        }
        return resolved;
    }();
    return api;
}

// XMem allocation-type page bits select the guest page granularity and must not be forwarded as Win32 flags.
constexpr DWORD kXmemLargePages = 0x20000000;  // 2 MiB pages
constexpr DWORD kXmem64kPages = 0x20400000;    // 64 KiB pages
constexpr DWORD kXmemPageModeMask = kXmem64kPages;
constexpr SIZE_T kXmemPageSize64k = 0x10000;
constexpr SIZE_T kXmemPageSize2m = 0x200000;
constexpr DWORD kXmemHostAllocationTypeMask = MEM_COMMIT | MEM_RESERVE | MEM_REPLACE_PLACEHOLDER |
                                              MEM_RESERVE_PLACEHOLDER | MEM_RESET | MEM_TOP_DOWN | MEM_WRITE_WATCH |
                                              MEM_RESET_UNDO;

bool round_up_size(SIZE_T value, SIZE_T alignment, SIZE_T& rounded) {
    const SIZE_T max = (std::numeric_limits<SIZE_T>::max)();
    if (alignment == 0 || (alignment & (alignment - 1)) != 0 || value > max - (alignment - 1)) return false;
    rounded = (value + alignment - 1) & ~(alignment - 1);
    return true;
}

// The console's PAGE_GRAPHICS_* protection bits (winnt.h: PAGE_GRAPHICS_NOACCESS 0x800 .. PAGE_GRAPHICS_NOCACHE
// 0x40000). The host refuses them with ERROR_INVALID_PARAMETER, and the D3D12 layer reads guest memory through the
// CPU mapping, so only the CPU protection is applied. The retail executable passes 0x801 and 0x22004.
constexpr DWORD kPageGraphicsMask = 0x0007F800;

DWORD host_protect(DWORD p) {
    const DWORD cpu = p & ~kPageGraphicsMask;
    return cpu ? cpu : PAGE_NOACCESS;
}

// Physical pages and the windows they are mapped into. One mutex covers both tables and every map and unmap.
//
// A physical-page window is a reserve-only XMemVirtualAlloc in a guest page mode, or a reserve-only replacement of
// part of a placeholder. On the host it is a placeholder, the only kind of reservation a section view can later
// replace, and its protection is kept here: the map calls take none, so a mapped page gets its window's protection, as
// AWE's MapUserPhysicalPages does.
//
// ASSUMPTION: a title does not commit memory inside such a window with kernel32!VirtualAlloc. A placeholder refuses a
// plain commit; the measured windows are only ever passed to the map calls.
struct Window {
    SIZE_T size;
    DWORD protect;
};

// A sparse temporary file gives large physical pools real shared pages without reserving their
// total size against the process's pagefile commitment. Views retain the section until the last alias dies.
HANDLE file_backed_section(std::uint64_t bytes) {
    wchar_t dir[32768]{};
    const auto len = GetEnvironmentVariableW(L"LEONIDA_XMEM_BACKING_DIR", dir, DWORD(std::size(dir)));
    if (!len || len >= std::size(dir)) return nullptr;
    const std::filesystem::path parent(dir);
    if (!parent.is_absolute()) return nullptr;
    static std::atomic<unsigned> sequence{};
    const auto name = L"physical-" + std::to_wstring(GetCurrentProcessId()) + L"-" +
                      std::to_wstring(GetTickCount64()) + L"-" + std::to_wstring(++sequence) + L".tmp";
    const HANDLE file = CreateFileW((parent / name).c_str(), GENERIC_READ | GENERIC_WRITE | DELETE,
                                   FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, CREATE_NEW,
                                   FILE_FLAG_DELETE_ON_CLOSE, nullptr);
    if (file == INVALID_HANDLE_VALUE) {
        return nullptr;
    }
    DWORD ignored{};
    LARGE_INTEGER size{};
    size.QuadPart = static_cast<LONGLONG>(bytes);
    HANDLE section = nullptr;
    if (DeviceIoControl(file, FSCTL_SET_SPARSE, nullptr, 0, nullptr, 0, &ignored, nullptr) &&
        SetFilePointerEx(file, size, nullptr, FILE_BEGIN) && SetEndOfFile(file))
        section = CreateFileMappingW(file, nullptr, PAGE_READWRITE, size.HighPart, size.LowPart, nullptr);
    CloseHandle(file);
    return section;
}

// A section and its scratch view: the whole section mapped once at an address the host picks, through which xmem
// commits, fills and copies the pages. The scratch view is recorded (g_scratch_views) while it is mapped.
struct Section {
    HANDLE handle;
    void* scratch;
    bool file_backed;
    std::vector<uint32_t, leonida::host_heap::Allocator<uint32_t>> frames;
    Section(HANDLE h, bool use_file = false);
    Section(const Section&) = delete;
    Section& operator=(const Section&) = delete;
    ~Section();
};

struct PhysicalPage {
    std::shared_ptr<Section> section;  // a mapped view holds its own reference to the section
    uint64_t offset;
};

constexpr uint64_t kPhysicalPageSize = 0x10000;
// Page numbers are 4 KiB frame numbers of 64 KiB pages: the measured title stores them packed and assumes each is a
// non-zero multiple of 16 below 2^25, with 0 for an entry it unmaps. So a page number is slot * 16 with slot in
// [1, 2^21), and freed slots are reused so a long run never runs out.
constexpr uint64_t kFramesPerPage = kPhysicalPageSize / 0x1000;
constexpr uint64_t kPageSlotLimit = uint64_t{1} << 21;

constexpr uintptr_t kTitleImage = 0x140000000;
// Only the reservations consumed by gfx_stream_pool_init_cpu_and_gpu_alias.
// Both locks protect writes; queue readers hold the physical lock, handlers the fault lock.
uintptr_t stream_cpu, stream_gpu, stream_size;
bool mirror_stream;
leonida::host_heap::map<uintptr_t, Window> stream_offsets;
std::mutex g_phys_mutex;
leonida::host_heap::map<uintptr_t, Window> g_windows;  // base -> window (host_heap: see g_pages)
// The same table for the fault handler, which may not wait for g_phys_mutex: a holder of that mutex allocates
// (g_mapped, g_windows) and the faulting title thread can hold the process heap's lock (see g_fault_lock). Host heap,
// read and written under g_fault_lock.
leonida::host_heap::map<uintptr_t, Window> g_fault_windows;
// The title's own MEM_RESERVE_PLACEHOLDER reservations (XMemVirtualAlloc), base -> extent, under g_fault_lock and kept
// until the title releases them (TitleVirtualFree). With the windows these are the title's reserved memory: the only
// place the fault handler commits a mapped section page it has no view of (commit_title_section_page).
leonida::host_heap::map<uintptr_t, Window> g_title_placeholders;
// Keep page table nodes on the runtime heap alongside the window table.
std::unordered_map<uint64_t, PhysicalPage, std::hash<uint64_t>, std::equal_to<uint64_t>,
                   leonida::host_heap::Allocator<std::pair<const uint64_t, PhysicalPage>>>
    g_pages;  // page number -> backing
std::vector<uint64_t> g_free_slots;
uint64_t g_next_slot = 1;
// A mapped run: `count` pages of one section at consecutive offsets, mapped as one view at its base. One view per run
// rather than per page: under Wine each view costs a placeholder split and a section map whose time grows with the
// number of views, and the retail title maps about 100,000 pages in one call.
enum class MapOwner { physical, scatter, slot31, mirror };
struct View {
    uint64_t count;
    std::shared_ptr<Section> section;  // kept for remapping part of the run after its pages are freed
    uint64_t offset;                   // section offset of the first page
    DWORD protect;
    // The title changed the protection of some of it (VirtualProtect): reopen_unowned leaves it as it is.
    bool title_protected = false;
    // A zero page the fault handler planted in a window placeholder the title never mapped (map_zero_page); a mapping
    // that replaces it returns its page to the zero section's free list.
    bool zero = false;
    MapOwner owner = MapOwner::physical;
    uint64_t tick = GetTickCount64();
};
leonida::host_heap::map<uintptr_t, View> g_views;  // view base -> view; host_heap: see g_fault_lock

// The zero page a planted view holds, freed when a mapping replaces or unmaps it (ZeroPages, below).
void zero_page_release(const View& view);

// Logs what the fault handler noted since the last call (FaultNote, below).
void drain_fault_notes();

// The fault race seam (rt::xmem::set_fault_race_hook), defined with the handler.
void fault_race_point(int stage);

// The views (g_views) as the fault handler reads them, which is all it reads: physical_page_fault takes it shared at
// every access violation in the process, xmem's own or not, and so does commit_physical_range. A title thread can
// take an access violation while it holds the process heap's lock (XMemAlloc is _aligned_malloc, so HeapAlloc and
// HeapFree write block headers on the title's pages: a page waiting for its first touch, or one a D3D12 shadow
// guards), and then waits here. So a thread that holds it exclusively must never wait for the process heap: it would
// wait for the faulting thread, which waits for it. Everything allocated or freed under it comes from
// leonida::host_heap (g_views, g_mapped, g_page_addresses, the sections, title_page_aliases), nothing under it logs,
// and nothing under it reads title memory, which could fault into the handler on the same thread.
SRWLOCK g_fault_lock = SRWLOCK_INIT;

// This thread's hold on g_fault_lock. The lock is not recursive: a fault this thread takes while it holds it (shared
// or exclusive) is left to the next handler rather than waited for, which would wait for itself; and a Section dropped
// under the exclusive lock unregisters its scratch view without taking it again.
thread_local int t_fault_lock_held = 0;
thread_local bool t_fault_lock_exclusive = false;

struct FaultLockExclusive {
    FaultLockExclusive() {
        AcquireSRWLockExclusive(&g_fault_lock);
        ++t_fault_lock_held;
        t_fault_lock_exclusive = true;
    }
    ~FaultLockExclusive() {
        t_fault_lock_exclusive = false;
        --t_fault_lock_held;
        ReleaseSRWLockExclusive(&g_fault_lock);
    }
    FaultLockExclusive(const FaultLockExclusive&) = delete;
    FaultLockExclusive& operator=(const FaultLockExclusive&) = delete;
};

struct FaultLockShared {
    FaultLockShared() {
        AcquireSRWLockShared(&g_fault_lock);
        ++t_fault_lock_held;
    }
    ~FaultLockShared() {
        --t_fault_lock_held;
        ReleaseSRWLockShared(&g_fault_lock);
    }
    FaultLockShared(const FaultLockShared&) = delete;
    FaultLockShared& operator=(const FaultLockShared&) = delete;
};

// The scratch views of xmem's sections (Section::scratch: each physical-page allocation's whole section, and the zero
// pool's), by base, under g_fault_lock. Host memory, not the title's: the fault handler never commits a page of one
// (commit_title_section_page), so a wild write into a section's pages through its mirror still faults.
// Never destroyed: the sections the tables declared before it hold (g_pages, g_views) drop their scratch views from it
// during the static destruction at exit, after a table declared here would be gone.
using AddressTable = std::set<uintptr_t, std::less<uintptr_t>, leonida::host_heap::Allocator<uintptr_t>>;
alignas(AddressTable) unsigned char g_scratch_views_storage[sizeof(AddressTable)];
AddressTable& g_scratch_views = *new (g_scratch_views_storage) AddressTable();

// With g_fault_lock held exclusively.
void note_scratch_locked(void* scratch, bool mapped) {
    if (!scratch) return;
    if (mapped)
        g_scratch_views.insert(reinterpret_cast<uintptr_t>(scratch));
    else
        g_scratch_views.erase(reinterpret_cast<uintptr_t>(scratch));
}

// From any thread: under the exclusive lock already held (a zero section made in the fault handler, a section whose
// last view an unmap drops), or by taking it. A thread holding it shared never makes or drops a section.
void note_scratch(void* scratch, bool mapped) {
    if (t_fault_lock_exclusive) {
        note_scratch_locked(scratch, mapped);
    } else if (!t_fault_lock_held) {
        const FaultLockExclusive lock;
        note_scratch_locked(scratch, mapped);
    }
}

Section::Section(HANDLE h, bool use_file)
    : handle(h), scratch(MapViewOfFile(h, FILE_MAP_WRITE, 0, 0, 0)), file_backed(use_file) {
    note_scratch(scratch, true);
}

Section::~Section() {
    if (scratch) {
        note_scratch(scratch, false);  // before the address can be handed out again
        UnmapViewOfFile(scratch);
    }
    CloseHandle(handle);
}

// With g_phys_mutex held; changed under g_fault_lock too (unmap_range), so they allocate from host_heap.
template <class K, class V>
using HostUnorderedMap =
    std::unordered_map<K, V, std::hash<K>, std::equal_to<K>, leonida::host_heap::Allocator<std::pair<const K, V>>>;
using AddressSet = std::set<uintptr_t, std::less<uintptr_t>, leonida::host_heap::Allocator<uintptr_t>>;
HostUnorderedMap<uintptr_t, uint64_t> g_mapped;            // mapped 64 KiB address -> page number
HostUnorderedMap<uint64_t, AddressSet> g_page_addresses;  // page number -> the addresses it is mapped at
// The same aliases, for readers that must not wait behind g_phys_mutex (geometry_aliases, from the D3D12 layer's
// submission thread: speed-16 and re-1 had it waiting there behind the streaming workers' scatter batches and the
// render thread's queue slot 31, which hold g_phys_mutex through every MapViewOfFile3). Only addresses whose page is
// mapped at another address too are in it, each with those other addresses. Written by note_mapping (g_phys_mutex
// held) under g_alias_lock exclusive, read under it shared.
std::shared_mutex g_alias_lock;
using AddressList = std::vector<uintptr_t, leonida::host_heap::Allocator<uintptr_t>>;
HostUnorderedMap<uintptr_t, AddressList> g_alias_peers;

// With g_phys_mutex held. Keeps leonida::title_page_aliases in step: a page mapped at two or more addresses is
// aliased at each of them (the D3D12 layer's slot-133 shadows guard those pages at every address, since a write
// through one address doesn't fault a guard on another).
void note_mapping(uint64_t page, uintptr_t address, bool mapped) {
    auto& addresses = g_page_addresses[page];
    if (mapped) {
        addresses.insert(address);
    } else {
        addresses.erase(address);
        leonida::title_page_aliases::set(address, false);
    }
    const bool aliased = addresses.size() > 1;
    for (const uintptr_t at : addresses) leonida::title_page_aliases::set(at, aliased);
    const std::unique_lock aliases(g_alias_lock);
    if (!mapped) g_alias_peers.erase(address);
    for (const uintptr_t at : addresses) {
        if (!aliased) {
            g_alias_peers.erase(at);
            continue;
        }
        AddressList& peers = g_alias_peers[at];
        peers.clear();
        for (const uintptr_t other : addresses)
            if (other != at) peers.push_back(other);
    }
    // After the peers: a cell's watcher that sees the new mapping asks geometry_aliases for the addresses again.
    using namespace leonida::geometry_writes;
    if (state().count.load(std::memory_order_acquire)) {
        notify(address, address + kPhysicalPageSize, Event::mapping);
        for (auto other : addresses)
            if (other != address) notify(other, other + kPhysicalPageSize, Event::mapping);
    }
    if (addresses.empty()) g_page_addresses.erase(page);
}

// Appends `value` in hex (with 0x) or decimal at `at`, which has room for it: no CRT, so no lock and no allocation.
char* append_number(char* at, unsigned long long value, bool hex) {
    char digits[24];
    int count = 0;
    do {
        const unsigned digit = static_cast<unsigned>(value % (hex ? 16 : 10));
        digits[count++] = static_cast<char>(digit < 10 ? '0' + digit : 'a' + digit - 10);
        value /= hex ? 16 : 10;
    } while (value);
    if (hex) {
        *at++ = '0';
        *at++ = 'x';
    }
    while (count) *at++ = digits[--count];
    return at;
}

char* append_text(char* at, const char* text) {
    while (*text) *at++ = *text++;
    return at;
}

// Fixed labels and at most three ranges per line fit in the caller's 512-byte buffer.
char* append_window(char* out, uintptr_t base, const Window& window) {
    out = append_number(append_text(out, "["), base, true);
    out = append_number(append_text(out, ","), base + window.size, true);
    return append_number(append_text(out, ") prot="), window.protect, true);
}

void write_window_note(char* line, char* end) {
    *end++ = '\n';
    DWORD written;
    WriteFile(GetStdHandle(STD_ERROR_HANDLE), line, static_cast<DWORD>(end - line), &written, nullptr);
}

void note_window_change(uintptr_t old_base, const Window& old_window, uintptr_t start, SIZE_T size,
                        DWORD protect, const void* caller, bool release = false) {
    char line[512];
    char* out = append_window(append_text(line, "[rt] xmem: old window "), old_base, old_window);
    out = append_window(append_text(out, release ? " release " : " new window "), start, Window{size, protect});
    out = append_number(append_text(out, " caller="), reinterpret_cast<uintptr_t>(caller), true);
    write_window_note(line, out);
}

// With g_phys_mutex held: the window holding [address, address + bytes), if one does.
const Window* find_window(uintptr_t address, uint64_t bytes) {
    auto it = g_windows.upper_bound(address);
    if (it == g_windows.begin()) return nullptr;
    --it;
    const uint64_t offset = address - it->first;
    if (offset >= it->second.size || bytes > it->second.size - offset) return nullptr;
    return &it->second;
}

// The entry of a base -> Window table that holds `address`, if one does.
template <class Table>
const Window* table_entry(const Table& table, uintptr_t address) {
    auto it = table.upper_bound(address);
    if (it == table.begin()) return nullptr;
    --it;
    return address - it->first < it->second.size ? &it->second : nullptr;
}

// Does an entry of `table` overlap [start, end)?
template <class Table>
bool table_overlaps(const Table& table, uintptr_t start, uintptr_t end) {
    auto it = table.lower_bound(end);
    if (it == table.begin()) return false;
    --it;
    return it->first + it->second.size > start;
}

// Removes [start, end) from `table`: an entry inside it goes, one reaching past either end keeps the part outside.
template <class Table>
void forget_range(Table& table, uintptr_t start, uintptr_t end, bool keep_outside = true) {
    auto it = table.upper_bound(start);
    if (it != table.begin()) {
        const auto prev = std::prev(it);
        if (prev->first + prev->second.size > start) it = prev;
    }
    while (it != table.end() && it->first < end) {
        const uintptr_t base = it->first;
        const Window entry = it->second;
        const uintptr_t entry_end = base + entry.size;
        it = table.erase(it);
        if (keep_outside && base < start) table.emplace(base, Window{static_cast<SIZE_T>(start - base), entry.protect});
        if (keep_outside && entry_end > end) {
            table.emplace(end, Window{static_cast<SIZE_T>(entry_end - end), entry.protect});
            break;
        }
    }
}

// A reserve-only replacement carves a live placeholder. Its neighbours still have their original protection.
// A fresh host reservation can instead reuse an address from a stale table entry.
template <class Table>
void replace_window(Table& table, uintptr_t start, SIZE_T size, DWORD protect, bool carve) {
    forget_range(table, start, start + size, carve);
    table.emplace(start, Window{size, protect});
}

// With g_phys_mutex held. The fault table is changed under exclusive ownership across the host split.
void record_window(void* base, SIZE_T size, DWORD protect, const void* caller, bool carve) {
    const uintptr_t start = reinterpret_cast<uintptr_t>(base);
    auto it = g_windows.lower_bound(start);
    if (it != g_windows.begin()) --it;
    for (; it != g_windows.end() && it->first < start + size; ++it)
        if (it->first + it->second.size > start)
            note_window_change(it->first, it->second, start, size, protect, caller);
    replace_window(g_windows, start, size, protect, carve);
}

// The window table the fault handler reads, kept in step with g_windows (declared with it above).
// With g_fault_lock held: the window holding `address`, if one does.
const Window* fault_window(uintptr_t address) { return table_entry(g_fault_windows, address); }

// With g_fault_lock held: is `address` inside a window?
bool in_window(uintptr_t address) { return fault_window(address) != nullptr; }

// With g_fault_lock held: is `address` in memory the title reserved, a window or a placeholder of its own?
bool in_title_reservation(uintptr_t address) {
    return in_window(address) || table_entry(g_title_placeholders, address) != nullptr;
}

// [address, address + size) as one placeholder of its own: split it out of the placeholder that holds it. Fails when
// the range is not inside one placeholder.
bool isolate_placeholder(void* address, SIZE_T size) {
    MEMORY_BASIC_INFORMATION info{};
    if (!VirtualQuery(address, &info, sizeof(info)) || info.State != MEM_RESERVE) {
        SetLastError(ERROR_INVALID_ADDRESS);
        return false;
    }
    if (info.AllocationBase == address && info.RegionSize == size) return true;
    if (!VirtualFree(address, size, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER)) return false;
    leonida::title_memory_map::changed(address, size);
    return true;
}

PVOID reserve_placeholder(void* address, SIZE_T size, SIZE_T alignment) {
    MEM_ADDRESS_REQUIREMENTS requirements{};
    MEM_EXTENDED_PARAMETER parameter{};
    ULONG parameter_count = 0;
    if (!address && alignment > kXmemPageSize64k) {
        requirements.Alignment = alignment;
        parameter.Type = MemExtendedParameterAddressRequirements;
        parameter.Pointer = &requirements;
        parameter_count = 1;
    }
    return placeholder_api().virtual_alloc2(GetCurrentProcess(), address, size, MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
                                            PAGE_NOACCESS, parameter_count ? &parameter : nullptr, parameter_count);
}

// A new ordinary reservation of the title's (MEM_RESERVE, not a placeholder, not physical, not large pages) is made
// with MEM_WRITE_WATCH and recorded (leonida::title_write_watch): the D3D12 layer's shadows of these pages then ask the
// kernel which pages were written instead of making them read-only and taking a fault at the title's first write.
// `allocate(type)` makes the call. A host that refuses the flag gets the call without it, and the pages are guarded
// as before; so are they with LEONIDA_WRITE_WATCH=0. A reservation the title asked to be write-watched itself is not
// recorded: its bits are the title's (XMemGetWriteWatch).
template <class Allocate>
void* reserve_watched(DWORD type, const void* address, SIZE_T size, Allocate&& allocate) {
    constexpr DWORD kNotWatchable = MEM_WRITE_WATCH | MEM_PHYSICAL | MEM_LARGE_PAGES | MEM_RESERVE_PLACEHOLDER |
                                    MEM_REPLACE_PLACEHOLDER | MEM_RESET | MEM_RESET_UNDO;
    // A commit at no given address reserves too.
    const bool reserves = (type & MEM_RESERVE) || (!address && (type & MEM_COMMIT));
    const bool watch = reserves && !(type & kNotWatchable) && leonida::title_write_watch::enabled();
    void* result = allocate(watch ? type | MEM_RESERVE | MEM_WRITE_WATCH : type);
    if (!watch) return result;
    if (!result) return allocate(type);
    // The reservation is [result, end of the requested range): VirtualAlloc rounds a given address down to 64 KiB.
    const uintptr_t base = reinterpret_cast<uintptr_t>(result);
    const uintptr_t end = (address ? reinterpret_cast<uintptr_t>(address) : base) + size;
    leonida::title_write_watch::add(base, end - base);
    return result;
}

// After a successful call that changes pages of an existing reservation without a write the kernel would see (a
// commit, which may bring back decommitted pages as zeros; MEM_RESET; a decommit with `decommit`): if it is
// write-watched, the shadows take [address, address + size) as written (size 0: to the reservation's end).
void note_unwatched_change(const void* address, SIZE_T size, DWORD type, bool decommit) {
    if (!address || !(decommit || (type & (MEM_COMMIT | MEM_RESET | MEM_RESET_UNDO))) || (type & MEM_RESERVE)) return;
    const auto from = reinterpret_cast<uintptr_t>(address);
    const uintptr_t to = size ? from + size : leonida::title_write_watch::end_of(from);
    if (to > from) {
        leonida::title_write_watch::mark(from, to);
        leonida::geometry_writes::notify(from, to, decommit ? leonida::geometry_writes::Event::replaced
                                                            : leonida::geometry_writes::Event::mapping);
    }
}

// PVOID XMemVirtualAlloc(address, size, allocationType, attributes, protect). All 23 calls of the measured title pass a
// nonzero attribute word; its bits are not decoded and nothing on the host depends on them, so it is accepted and only
// recorded (xmem_attributes). Its types cover committed 2 MiB and 64 KiB pages, reserve-only 64 KiB windows for physical pages, one
// MEM_RESERVE_PLACEHOLDER reservation and five MEM_REPLACE_PLACEHOLDER windows carved out of it, each then filled with
// XMemMapPhysicalPages. The guest page size sets the alignment and rounding of the region; the host's own pages stay
// 4 KiB (host large pages need SeLockMemoryPrivilege). A 2 MiB region the host places is reserved 2 MiB-aligned through
// VirtualAlloc2.
void* xmem_virtual_alloc(void* address, SIZE_T size, DWORD allocation_type, ULONGLONG attributes, DWORD protect, const void* caller);

void* WINAPI XMemVirtualAlloc(void* address, SIZE_T size, DWORD allocation_type, ULONGLONG attributes, DWORD protect) {
    drain_fault_notes();
    note_attributes("XMemVirtualAlloc", attributes, size, _ReturnAddress());
    void* result = xmem_virtual_alloc(address, size, allocation_type, attributes, protect, _ReturnAddress());
    // Diagnostic: the title's large regions (windows and placeholders log their own), so a fault left to the title can
    // be matched to the region it is in.
    static std::atomic<int> shown{0};
    const bool logs_itself = (allocation_type & MEM_RESERVE_PLACEHOLDER) ||
                             ((allocation_type & kXmemPageModeMask) && !(allocation_type & MEM_COMMIT));
    if (result && size >= (64u << 20) && !logs_itself && shown.fetch_add(1, std::memory_order_relaxed) < 64)
        std::fprintf(stderr, "[rt] xmem: XMemVirtualAlloc %p size %llu type 0x%lx protect 0x%lx\n", result,
                     static_cast<unsigned long long>(size), static_cast<unsigned long>(allocation_type),
                     static_cast<unsigned long>(protect));
    // The title's memory (src/title_va_alias.h). Never forgotten: a released range is no longer committed, which
    // the copy also checks.
    if (result) {
        leonida::title_memory_map::changed(result, size);
        leonida::title_va_aliases().add_title_memory(reinterpret_cast<uintptr_t>(result), size);
        note_unwatched_change(address, size, allocation_type, false);
        if (allocation_type & MEM_RESERVE) {
            const auto from = reinterpret_cast<uintptr_t>(result);
            leonida::geometry_writes::notify(from, from + size, leonida::geometry_writes::Event::replaced);
        }
    }
    if (result && allocation_type == 0x20402000)
        rt::xmem::note_stream_reservation(reinterpret_cast<uintptr_t>(result), size,
                                           reinterpret_cast<uintptr_t>(_ReturnAddress()) - kTitleImage);
    // A reservation: a GPU-only one (no CPU access, PAGE_GRAPHICS_* bits) is paired with the CPU-mappable one of the
    // same size made just before it (src/title_page_watch.h, title_gpu_views).
    const DWORD host_type = allocation_type & kXmemHostAllocationTypeMask;
    if (result && host_type && !(host_type & MEM_COMMIT)) {
        const bool gpu_only = (protect & kPageGraphicsMask) && host_protect(protect) == PAGE_NOACCESS;
        const auto base = reinterpret_cast<uintptr_t>(result);
        const bool stream_queue_view = allocation_type == 0x20402000 && gpu_only &&
            reinterpret_cast<uintptr_t>(_ReturnAddress()) - kTitleImage == 0x1e1d03e &&
            rt::xmem::slot31_runs() && rt::xmem::queue_page_mappings_enabled();
        if (leonida::title_gpu_views::note_reservation(base, size, gpu_only, stream_queue_view))
            std::fprintf(stderr, "[rt] xmem: GPU-only view %p (%llu MiB, protect 0x%lx) paired with CPU view %p (%s)\n",
                         result, static_cast<unsigned long long>(size >> 20), static_cast<unsigned long>(protect),
                         reinterpret_cast<void*>(leonida::title_gpu_views::other_view(base)),
                         stream_queue_view ? "separate queue mappings" : "CPU backing");
        else if (gpu_only && size >= (64u << 20))
            std::fprintf(stderr, "[rt] xmem: GPU-only reservation %p (%llu MiB, protect 0x%lx) has no CPU view\n", result,
                         static_cast<unsigned long long>(size >> 20), static_cast<unsigned long>(protect));
    }
    if(result)track_label_data(result,size,false);
    return result;
}

void* xmem_virtual_alloc(void* address, SIZE_T size, DWORD allocation_type, ULONGLONG attributes, DWORD protect, const void* caller) {
    if (size == 0) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return nullptr;
    }

    const DWORD page_mode = allocation_type & kXmemPageModeMask;
    if (page_mode != 0 && page_mode != kXmem64kPages && page_mode != kXmemLargePages) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return nullptr;
    }

    const DWORD unsupported = allocation_type & ~(kXmemHostAllocationTypeMask | kXmemPageModeMask);
    if (unsupported != 0) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return nullptr;
    }

    DWORD host_type = allocation_type & kXmemHostAllocationTypeMask;
    if (host_type == 0) host_type = MEM_RESERVE | MEM_COMMIT;
    const DWORD host_prot = host_protect(protect);

    const bool commit = (host_type & MEM_COMMIT) != 0;
    const bool new_placeholder = (host_type & MEM_RESERVE_PLACEHOLDER) != 0;
    const bool replace = (host_type & MEM_REPLACE_PLACEHOLDER) != 0;
    const bool window = page_mode != 0 && !commit && !new_placeholder;
    if ((new_placeholder || replace || window || page_mode == kXmemLargePages) && !placeholder_api().available()) {
        SetLastError(ERROR_NOT_SUPPORTED);
        return nullptr;
    }

    SYSTEM_INFO system_info{};
    GetSystemInfo(&system_info);
    SIZE_T page_granularity = system_info.dwPageSize;
    SIZE_T base_granularity = (host_type & MEM_RESERVE) ? system_info.dwAllocationGranularity : system_info.dwPageSize;
    if (page_mode == kXmem64kPages) page_granularity = base_granularity = kXmemPageSize64k;
    if (page_mode == kXmemLargePages) page_granularity = base_granularity = kXmemPageSize2m;
    if (new_placeholder || replace) {
        // Placeholders are split and replaced on allocation-granularity boundaries only.
        if (page_granularity < system_info.dwAllocationGranularity) page_granularity = system_info.dwAllocationGranularity;
        if (base_granularity < system_info.dwAllocationGranularity) base_granularity = system_info.dwAllocationGranularity;
    }
    if (page_granularity == 0 || (page_granularity & (page_granularity - 1)) != 0 || base_granularity == 0 ||
        (base_granularity & (base_granularity - 1)) != 0) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return nullptr;
    }

    if (address != nullptr) {
        const uintptr_t requested = reinterpret_cast<uintptr_t>(address);
        const uintptr_t max = (std::numeric_limits<uintptr_t>::max)();
        if (size > max - requested) {
            SetLastError(ERROR_INVALID_PARAMETER);
            return nullptr;
        }
        const uintptr_t requested_end = requested + size;
        const uintptr_t base = requested & ~(static_cast<uintptr_t>(base_granularity) - 1);
        if (requested_end > max - (static_cast<uintptr_t>(page_granularity) - 1)) {
            SetLastError(ERROR_INVALID_PARAMETER);
            return nullptr;
        }
        const uintptr_t end = (requested_end + static_cast<uintptr_t>(page_granularity) - 1) &
                              ~(static_cast<uintptr_t>(page_granularity) - 1);
        address = reinterpret_cast<void*>(base);
        size = static_cast<SIZE_T>(end - base);
    } else if (replace) {
        SetLastError(ERROR_INVALID_PARAMETER);  // a replacement names the placeholder it replaces
        return nullptr;
    } else if (page_granularity > system_info.dwPageSize && !round_up_size(size, page_granularity, size)) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return nullptr;
    }

    if (replace) {
        const std::lock_guard lock(g_phys_mutex);
        void* replaced = nullptr;
        {
            // The split, and the replacement, change pages of a placeholder of the title's: under the exclusive lock,
            // as every change of xmem's to the title's reservations is, so the fault handler never decides on a page
            // the host is part way through changing.
            const FaultLockExclusive view_lock;
            if (!isolate_placeholder(address, size)) return nullptr;
            if (window)
                replace_window(g_fault_windows, reinterpret_cast<uintptr_t>(address), size, host_prot, true);
            else
                replaced = placeholder_api().virtual_alloc2(GetCurrentProcess(), address, size, host_type | MEM_RESERVE,
                                                            host_prot, nullptr, 0);
        }
        if (!window) return replaced;
        // A reserve-only replacement stays a placeholder: the physical pages are mapped into it.
        record_window(address, size, host_prot, caller, true);
        return address;
    }
    if (new_placeholder || window) {
        void* const result = reserve_placeholder(address, size, base_granularity);
        if (result && window) {
            const std::lock_guard lock(g_phys_mutex);
            record_window(result, size, host_prot, caller, false);
            {
                const FaultLockExclusive faults;
                replace_window(g_fault_windows, reinterpret_cast<uintptr_t>(result), size, host_prot, false);
            }
            // Diagnostic (temporary): the windows a fault can be inside.
            std::fprintf(stderr, "[rt] xmem: window reserve %p size %llu protect 0x%lx\n", result,
                         static_cast<unsigned long long>(size), static_cast<unsigned long>(host_prot));
        }
        if (result && new_placeholder) {
            const uintptr_t base = reinterpret_cast<uintptr_t>(result);
            {
                const FaultLockExclusive lock;
                forget_range(g_title_placeholders, base, base + size);  // a release the table missed
                g_title_placeholders[base] = Window{size, host_prot};
            }
            std::fprintf(stderr, "[rt] xmem: placeholder reserve %p size %llu\n", result,
                         static_cast<unsigned long long>(size));
        }
        return result;
    }
    if (page_mode == kXmemLargePages && address == nullptr) {
        MEM_ADDRESS_REQUIREMENTS requirements{};
        requirements.Alignment = kXmemPageSize2m;
        MEM_EXTENDED_PARAMETER parameter{};
        parameter.Type = MemExtendedParameterAddressRequirements;
        parameter.Pointer = &requirements;
        return reserve_watched(host_type | MEM_RESERVE, nullptr, size, [&](DWORD type) {
            return placeholder_api().virtual_alloc2(GetCurrentProcess(), nullptr, size, type, host_prot, &parameter, 1);
        });
    }
    return reserve_watched(host_type, address, size,
                           [&](DWORD type) { return VirtualAlloc(address, size, type, host_prot); });
}

// With g_phys_mutex held. Maps `view` at the 64 KiB-aligned `address`, which holds no mapped page. MEM_REPLACE_PLACEHOLDER
// needs the view to fill one placeholder exactly, and earlier partial unmaps can leave several host placeholders in one
// run, so the run is mapped as one view per placeholder it spans. Views are capped at kMaxViewPages so that unmapping a
// page inside a long run only remaps the rest of one capped view, not the whole run.
constexpr uint64_t kMaxViewPages = 1024;  // 64 MiB

// Caller checked this is a private zero plant and holds the exclusive fault lock.
bool remove_zero_view(decltype(g_views)::iterator planted) {
    const uintptr_t base = planted->first;
    if (!placeholder_api().unmap_view2(GetCurrentProcess(), reinterpret_cast<void*>(base), MEM_PRESERVE_PLACEHOLDER))
        return false;
    zero_page_release(planted->second);
    g_views.erase(planted);
    leonida::title_memory_map::changed(base, base + kPhysicalPageSize);
    leonida::geometry_writes::notify(base, base + kPhysicalPageSize, leonida::geometry_writes::Event::replaced);
    return true;
}

// `done`, when given, is set to the pages of the run mapped so far, which stay mapped when a later view fails.
bool map_view(uintptr_t address, const View& view, uint64_t* done = nullptr) {
    if (done) *done = 0;
    for (uint64_t mapped = 0; mapped < view.count;) {
        const uintptr_t base = address + mapped * kPhysicalPageSize;
        void* const target = reinterpret_cast<void*>(base);
        MEMORY_BASIC_INFORMATION info{};
        if (!VirtualQuery(target, &info, sizeof(info)) || info.RegionSize < kPhysicalPageSize) {
            SetLastError(ERROR_INVALID_ADDRESS);
            return false;
        }
        if (info.State != MEM_RESERVE) {
            // A zero page a fault planted here. Only a planted one: the caller unmaps a physical-page view of this run
            // first, so a mapping never displaces a real one. This mapping replaces it, and its section page goes back
            // to the zero pool.
            const auto planted = g_views.find(base);
            if (planted == g_views.end() || planted->first != base || !planted->second.zero) {
                SetLastError(ERROR_INVALID_ADDRESS);
                return false;
            }
            if (!remove_zero_view(planted)) return false;
            if (!VirtualQuery(target, &info, sizeof(info)) || info.State != MEM_RESERVE ||
                info.RegionSize < kPhysicalPageSize) {
                SetLastError(ERROR_INVALID_ADDRESS);
                return false;
            }
        }
        const uint64_t available = info.RegionSize / kPhysicalPageSize;
        const uint64_t remaining = view.count - mapped;
        const uint64_t count = (std::min)({available, remaining, kMaxViewPages});
        const SIZE_T size = static_cast<SIZE_T>(count * kPhysicalPageSize);
        const uint64_t offset = view.offset + mapped * kPhysicalPageSize;
        if (!isolate_placeholder(target, size)) return false;
        if (!placeholder_api().map_view3(view.section->handle, GetCurrentProcess(), target, offset, size,
                                         MEM_REPLACE_PLACEHOLDER, view.protect, nullptr, 0)) {
            return false;
        }
        leonida::title_memory_map::changed(target, size);
        View part = view;
        part.count = count;
        part.offset = offset;
        g_views.emplace(base, std::move(part));
        mapped += count;
        if (done) *done = mapped;
    }
    return true;
}

// The last unmaps (diagnostic): which range, when, and on whose behalf, for the read-fault report in
// physical_page_fault. Protected by g_fault_lock.
struct UnmapRecord {
    uintptr_t start, end;
    ULONGLONG tick;
    const char* cause;
};
constexpr size_t kUnmapRecords = 1024;
UnmapRecord g_unmaps[kUnmapRecords];
uint64_t g_unmap_count = 0;
thread_local const char* t_unmap_cause = "unmap";

struct UnmapCause {
    const char* was;
    explicit UnmapCause(const char* cause) : was(std::exchange(t_unmap_cause, cause)) {}
    ~UnmapCause() { t_unmap_cause = was; }
};

// With g_fault_lock held exclusively. Unmaps every page in [start, end); the addresses become placeholders again. A
// view that reaches outside the range is unmapped whole, and its pages outside the range are mapped again, so
// neighbours keep their pages.
// There is no rollback: if mapping a remainder again fails, those neighbouring pages stay unmapped and the caller only
// sees the failure.
bool unmap_range_locked(uintptr_t start, uintptr_t end) {
    g_unmaps[g_unmap_count++ % kUnmapRecords] = {start, end, GetTickCount64(), t_unmap_cause};
    auto it = g_views.upper_bound(start);
    if (it != g_views.begin()) {
        const auto prev = std::prev(it);
        if (prev->first + prev->second.count * kPhysicalPageSize > start) it = prev;
    }
    while (it != g_views.end() && it->first < end) {
        const uintptr_t base = it->first;
        const View view = it->second;
        const uintptr_t view_end = base + view.count * kPhysicalPageSize;
        if (!placeholder_api().unmap_view2(GetCurrentProcess(), reinterpret_cast<void*>(base),
                                           MEM_PRESERVE_PLACEHOLDER)) {
            return false;
        }
        leonida::title_memory_map::changed(base, view_end);
        leonida::geometry_writes::notify(base, view_end, leonida::geometry_writes::Event::replaced);
        it = g_views.erase(it);
        // The pages of [start, end) this view held are unmapped now; the ones outside it are mapped again below.
        for (uintptr_t at = (std::max)(base, start); at < (std::min)(view_end, end); at += kPhysicalPageSize) {
            if (const auto mapped = g_mapped.find(at); mapped != g_mapped.end()) {
                note_mapping(mapped->second, at, false);
                g_mapped.erase(mapped);
            }
        }
        // A planted zero page the range unmaps whole is free again; the section page is reused, zeroed, for the next
        // plant. A partial unmap keeps it: the remainder still maps it.
        if (base >= start && view_end <= end) zero_page_release(view);
        if (base < start) {
            View left = view;
            left.count = (start - base) / kPhysicalPageSize;
            if (!map_view(base, left)) return false;
        }
        if (view_end > end) {
            const uint64_t skipped = (end - base) / kPhysicalPageSize;
            View right = view;
            right.count = view.count - skipped;
            right.offset += skipped * kPhysicalPageSize;
            return map_view(end, right);  // the last view the range touches
        }
    }
    return true;
}

bool unmap_range(uintptr_t start, uintptr_t end) {
    const FaultLockExclusive lock;  // faults on neighbouring pages wait through the split and remap
    return unmap_range_locked(start, end);
}

bool unmap_page_locked(uintptr_t address) { return unmap_range_locked(address, address + kPhysicalPageSize); }

bool unmap_page(uintptr_t address) { return unmap_range(address, address + kPhysicalPageSize); }

// With g_phys_mutex held. Maps physical page `page` (0: unmap) at the 64 KiB-aligned `address`, replacing what was
// mapped there. The caller has found the window.
bool map_page(uintptr_t address, uint64_t page, DWORD protect, MapOwner owner) {
    const auto backing = page ? g_pages.find(page) : g_pages.end();
    if (page && backing == g_pages.end()) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return false;
    }
    {
        // One exclusive section across the unmap and the map: a write fault on this page waits here and then sees the
        // mapping, instead of landing in the gap where the page is a placeholder and planting a zero page the map
        // would then discard.
        const FaultLockExclusive lock;
        if (!unmap_page_locked(address)) return false;
        if (!page) return true;
        fault_race_point(rt::xmem::kRemapBetween);
        if (!map_view(address, View{1, backing->second.section, backing->second.offset, protect, false, false, owner})) return false;
    }
    g_mapped[address] = page;
    note_mapping(page, address, true);
    return true;
}

bool on_wine() {
    static const bool wine = GetProcAddress(GetModuleHandleW(L"ntdll.dll"), "wine_get_version") != nullptr;
    return wine;
}

// Physical-page sections committed whole (under Wine, unless LEONIDA_XMEM_SEC_COMMIT=0), or reserved and committed on
// first touch (Windows, and Wine with the switch off).
bool sections_committed() {
    static const bool committed = [] {
        char v[4] = {};
        const DWORD n = GetEnvironmentVariableA("LEONIDA_XMEM_SEC_COMMIT", v, sizeof(v));
        return on_wine() && !(n == 1 && v[0] == '0');
    }();
    return committed;
}

enum class PageCommit { not_ours, committed_before, committed_now, failed };

// Two of the measured windows are reserved PAGE_NOACCESS for the CPU and the GPU alike; pages mapped there are still
// GPU memory, which the D3D12 layer reads on the CPU.
DWORD view_protect(const Window& window) {
    // A view takes no cache modifier. Three of the measured windows carry PAGE_WRITECOMBINE (0x400) next to
    // PAGE_READWRITE: Wine's MapViewOfFile3 refuses it, and PAGE_NOCACHE, with ERROR_INVALID_PARAMETER, and on Windows
    // the view comes out write-combined, which an import of those pages as a GPU buffer refuses (#7218). PAGE_GUARD
    // is an access alarm, not a cache modifier, so it is left as the window has it.
    const DWORD protect = window.protect & ~static_cast<DWORD>(PAGE_WRITECOMBINE | PAGE_NOCACHE);
    return protect == PAGE_NOACCESS ? PAGE_READWRITE : protect;
}

// A page of a window the title reserved and has not mapped yet: the console's window pages are zeroed memory it may
// write whenever it likes (measured: a title clears a 288 KiB grid at a story-load step into a window granule that it
// never passed to XMemMapPhysicalPages, and on the host the placeholder there is PAGE_NOACCESS). So the page the title
// reaches is backed by one page of a demand-zero section, mapped into the placeholder and committed. A page a mapping
// replaces goes back to the free list and is zeroed again before it is handed out, so a long run of plant-and-replace
// costs no more section pages than the plants alive at once.
//
// When a section's pages are all handed out the pool moves on to a new one; the old one lives on in the plants that
// still use it, and a page of it a mapping replaces is not reused (its offset means nothing in the new section).
struct ZeroPages {
    std::shared_ptr<Section> section;             // pagefile-backed SEC_RESERVE, kZeroPagesPages pages, and its view
    uint64_t next = 0;                            // next page of the section never handed out
    leonida::host_heap::vector<uint64_t> freed;   // pages of `section` a mapping replaced, zeroed before reuse
};
constexpr uint64_t kZeroPagesPages = 1024;  // 64 MiB per section

// At namespace scope, made before main: the fault handler that first uses it must not construct a static (MSVC
// registers its destructor with atexit, which takes the CRT's lock and the process heap).
ZeroPages g_zero_pages;

// With g_fault_lock held exclusively, and `freed`/`next` under it: a fresh or reused zeroed page of the zero section.
// False when a new section is needed but cannot be made.
bool zero_page_take(uint64_t& offset) {
    ZeroPages& pages = g_zero_pages;
    if (!pages.section || (pages.freed.empty() && pages.next >= kZeroPagesPages)) {
        const HANDLE handle = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE | SEC_RESERVE,
                                                 static_cast<DWORD>(kZeroPagesPages * kPhysicalPageSize >> 32),
                                                 static_cast<DWORD>(kZeroPagesPages * kPhysicalPageSize), nullptr);
        if (!handle) return false;
        auto section = std::allocate_shared<Section>(leonida::host_heap::Allocator<Section>{}, handle);
        if (!section->scratch) return false;  // the Section closes the handle
        pages.section = std::move(section);
        pages.next = 0;
        pages.freed.clear();  // the old section stays alive in the views that still use it
    }
    if (!pages.freed.empty()) {
        offset = pages.freed.back();
        pages.freed.pop_back();
        // A replaced plant kept whatever was written into it: the next title to reach this page expects zeroes.
        std::memset(static_cast<uint8_t*>(pages.section->scratch) + offset, 0, kPhysicalPageSize);
    } else {
        offset = pages.next * kPhysicalPageSize;
        ++pages.next;
    }
    return VirtualAlloc(static_cast<uint8_t*>(pages.section->scratch) + offset, kPhysicalPageSize, MEM_COMMIT,
                        PAGE_READWRITE) != nullptr;
}

// With g_fault_lock held exclusively: the page of the zero section a replaced plant used is free again, if that
// section is still the pool's.
void zero_page_release(const View& view) {
    if (view.zero && view.section == g_zero_pages.section) g_zero_pages.freed.push_back(view.offset);
}

// With g_fault_lock held exclusively. `address` is the 64 KiB-aligned page of a physical-page window that holds no
// mapping. Maps a zeroed page there, with the window's protection; false when the address is not such a page or the
// map fails. Only a title-code read takes g_phys_mutex before the fault lock.
bool map_zero_page(uintptr_t address, bool read_only) {
    MEMORY_BASIC_INFORMATION info{};
    if (!VirtualQuery(reinterpret_cast<const void*>(address), &info, sizeof(info)) || info.State != MEM_RESERVE ||
        info.Type != MEM_PRIVATE || info.RegionSize < kPhysicalPageSize)
        return false;
    const Window* const window = fault_window(address);
    if (!window) return false;
    const DWORD protect = read_only ? PAGE_READONLY : view_protect(*window);
    uint64_t offset = 0;
    if (!zero_page_take(offset)) return false;
    View planted{1, g_zero_pages.section, offset, protect};
    planted.zero = true;
    if (!isolate_placeholder(reinterpret_cast<void*>(address), kPhysicalPageSize) ||
        !placeholder_api().map_view3(planted.section->handle, GetCurrentProcess(), reinterpret_cast<void*>(address),
                                     offset, kPhysicalPageSize, MEM_REPLACE_PLACEHOLDER, protect, nullptr, 0)) {
        zero_page_release(planted);
        return false;
    }
    g_views[address] = planted;
    leonida::geometry_writes::notify(address, address + kPhysicalPageSize, leonida::geometry_writes::Event::mapping);
    leonida::title_memory_map::changed(address, address + kPhysicalPageSize);
    return true;
}

PageCommit commit_mapped_page(uintptr_t address, uintptr_t accessed, DWORD& protect) {
    auto it = g_views.upper_bound(address);
    if (it == g_views.begin()) return PageCommit::not_ours;
    --it;
    const uintptr_t within = address - it->first;
    if (within / kPhysicalPageSize >= it->second.count || !it->second.section->scratch) return PageCommit::not_ours;
    MEMORY_BASIC_INFORMATION info{};
    if (!VirtualQuery(reinterpret_cast<void*>(accessed), &info, sizeof(info)) || info.Type != MEM_MAPPED) {
        return PageCommit::not_ours;
    }
    // The fault handler needs the current protection of the accessed host page to decide whether to retry.
    if (sections_committed() || info.State == MEM_COMMIT) {
        protect = info.Protect;
        return PageCommit::committed_before;
    }
    void* const backing = static_cast<uint8_t*>(it->second.section->scratch) + it->second.offset + within;
    if (!VirtualAlloc(backing, kPhysicalPageSize, MEM_COMMIT, PAGE_READWRITE)) return PageCommit::failed;
    leonida::geometry_writes::notify(address, address + kPhysicalPageSize, leonida::geometry_writes::Event::mapping);
    leonida::title_memory_map::changed(address, address + kPhysicalPageSize);
    return PageCommit::committed_now;
}

bool access_allowed(DWORD protect, ULONG_PTR access) {
    if (protect & PAGE_GUARD) return false;
    switch (protect & 0xff) {
    case PAGE_EXECUTE_READWRITE:
    case PAGE_EXECUTE_WRITECOPY:
        return true;
    case PAGE_READWRITE:
    case PAGE_WRITECOPY:
        return access != 8;
    case PAGE_EXECUTE_READ:
        return access != 1;
    case PAGE_READONLY:
        return access == 0;
    case PAGE_EXECUTE:
        return access == 8;
    default:
        return false;
    }
}

// What the fault handler did with a fault in the title's reserved memory, or with a write it left to the title in any
// memory that is only reserved, kept for drain_fault_notes to log: the handler itself may not log (the faulting thread
// can hold the CRT's or the process heap's lock). Lock-free and bounded: each kind keeps its first few, so routine
// faults cannot use up the room the rare ones need.
struct FaultNote {
    uintptr_t accessed;
    uint32_t kind;  // FaultKind
    uint32_t access;
    int32_t result;  // PageCommit
    uint32_t where;  // FaultWhere
    uint32_t state, protect, type, allocation_protect;
    uintptr_t allocation_base, region_end;  // who reserved the page, and where its run ends
    bool written;  // written out by the handler already (write_refusal_now): drain_fault_notes skips it
};
enum FaultKind : uint32_t { kPlanted, kCommitted, kRefused, kFaultKinds };
enum FaultWhere : uint32_t { kNoReservation, kInWindow, kInPlaceholder };
constexpr uint32_t kFaultNoteCaps[kFaultKinds] = {20, 20, 100};
constexpr uint32_t kFaultNoteSlots = 20 + 20 + 100;
FaultNote g_fault_notes[kFaultNoteSlots];
std::atomic<bool> g_fault_note_ready[kFaultNoteSlots];
std::atomic<uint32_t> g_fault_notes_taken{0};
std::atomic<uint32_t> g_fault_notes_shown{0};
std::atomic<uint64_t> g_fault_kind_counts[kFaultKinds];

// Faults left to the title past the first kFaultNoteCaps[kRefused] are counted, not noted: drain_fault_notes says how
// many there were when the count reaches this (doubled each time it is said).
std::atomic<uint64_t> g_refused_report_at{kFaultNoteCaps[kRefused] + 1};


bool g_pool_zero_pages;

bool title_code(uintptr_t pc) {
    if (pc < kTitleImage) return false;
    __try {
        const auto* dos = reinterpret_cast<const IMAGE_DOS_HEADER*>(kTitleImage);
        if (dos->e_magic != IMAGE_DOS_SIGNATURE) return false;
        const auto* nt = reinterpret_cast<const IMAGE_NT_HEADERS64*>(kTitleImage + dos->e_lfanew);
        return nt->Signature == IMAGE_NT_SIGNATURE && pc - kTitleImage < nt->OptionalHeader.SizeOfImage;
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        return false;
    }
}

// Under the fault lock, including the mapping identity and time across view splits.
const View* covering_view(uintptr_t address, uintptr_t& base) {
    auto it = g_views.upper_bound(address);
    if (it == g_views.begin()) return nullptr;
    --it;
    base = it->first;
    return address - base < it->second.count * kPhysicalPageSize ? &it->second : nullptr;
}

char* describe_twin(char* out, uintptr_t at) {
    if (!stream_gpu || at < stream_cpu || at - stream_cpu >= stream_size)
        return append_text(out, ", outside stream CPU window");
    const auto offset = (at - stream_cpu) & ~(kPhysicalPageSize - 1);
    const auto gpu = stream_gpu + offset;
    out = append_number(append_text(out, ", GPU twin "), gpu, true);
    if (!table_overlaps(stream_offsets, offset, offset + kPhysicalPageSize)) return append_text(out, " released");
    uintptr_t base = 0;
    const auto* view = covering_view(gpu, base);
    if (!view || view->zero) return append_text(out, view ? " zero plant, no frame" : " unmapped");
    const auto index = (view->offset + gpu - base) / kPhysicalPageSize;
    out = append_text(out, view->owner == MapOwner::slot31 ? " slot31" :
                           view->owner == MapOwner::scatter ? " scatter" : " physical");
    out = append_number(append_text(out, " frame "), view->section->frames[index], true);
    return append_number(append_text(out, " mapped tick_ms "), view->tick, false);
}

// Snapshot the history under the shared lock. Write the stack buffer after releasing it.
DWORD describe_read(char* line, uintptr_t pc, uintptr_t at, const MEMORY_BASIC_INFORMATION& seen) {

    char* out = append_number(append_text(line, "[rt] xmem: title read fault at t+"), pc - kTitleImage, true);
    out = append_number(append_text(out, " on "), at, true);
    out = append_number(append_text(out, " state "), seen.State, true);
    out = append_number(append_text(out, " protect "), seen.Protect, true);
    out = append_number(append_text(out, " type "), seen.Type, true);
    out = append_number(append_text(out, " allocation "), reinterpret_cast<uintptr_t>(seen.AllocationBase), true);
    out = describe_twin(out, at);
    const auto n = g_unmap_count;
    out = append_number(append_text(out, ", unmaps "), n, false);
    unsigned hits = 0;
    for (uint64_t k = 0; k < n && k < kUnmapRecords && hits < 4; ++k) {
        const auto& r = g_unmaps[(n - 1 - k) % kUnmapRecords];
        if (at < r.start || at >= r.end) continue;
        ++hits;
        out = append_text(append_text(out, "\n  "), r.cause);
        out = append_number(append_text(out, " ["), r.start, true);
        out = append_number(append_text(out, ", "), r.end, true);
        out = append_number(append_text(out, ") ms ago "), GetTickCount64() - r.tick, false);
    }
    if (!hits) out = append_text(out, ", no recent matching unmap");
    *out++ = '\n';
    return static_cast<DWORD>(out - line);
}

// A fault in the title's reserved memory left to the title is usually the last thing the title does: its own
// handler, or none, ends the process before the next XMem call would log the note. So the handler writes that note
// out itself, outside the fault lock, straight to the standard error handle: no CRT (whose stream lock the faulting
// thread may hold), no allocation, one WriteFile of a line built on the stack.
void write_refusal_now(const FaultNote& n) {
    const HANDLE out = GetStdHandle(STD_ERROR_HANDLE);
    if (!out || out == INVALID_HANDLE_VALUE) return;
    char line[384];
    char* at = line;
    at = append_text(at, "[rt] xmem: fault at ");
    at = append_number(at, n.accessed, true);
    at = append_text(at, " (access ");
    at = append_number(at, n.access, false);
    at = append_text(at, ") left to the title: commit ");
    at = append_number(at, static_cast<unsigned>(n.result), false);
    at = append_text(at, n.where == kInWindow ? ", in a window" : ", in a title placeholder");
    at = append_text(at, ", host state ");
    at = append_number(at, n.state, true);
    at = append_text(at, " protect ");
    at = append_number(at, n.protect, true);
    at = append_text(at, " type ");
    at = append_number(at, n.type, true);
    at = append_text(at, ", allocation base ");
    at = append_number(at, n.allocation_base, true);
    at = append_text(at, ", run ends ");
    at = append_number(at, n.region_end, true);
    at = append_text(at, "\n");
    DWORD wrote = 0;
    WriteFile(out, line, static_cast<DWORD>(at - line), &wrote, nullptr);
}

// `seen` is the page as the handler's decision saw it (its last query of `accessed`), not as it is once the handler is
// done with it: after a plant, or another thread's map, that would say why a different fault was left alone.
void note_fault(FaultKind kind, uintptr_t accessed, ULONG_PTR access, PageCommit result, FaultWhere where,
                const MEMORY_BASIC_INFORMATION& seen) {
    if (g_fault_kind_counts[kind].fetch_add(1, std::memory_order_relaxed) >= kFaultNoteCaps[kind]) return;
    const uint32_t slot = g_fault_notes_taken.fetch_add(1, std::memory_order_relaxed);
    if (slot >= kFaultNoteSlots) return;
    const MEMORY_BASIC_INFORMATION& q = seen;
    FaultNote& n = g_fault_notes[slot];
    n.accessed = accessed;
    n.kind = kind;
    n.access = static_cast<uint32_t>(access);
    n.result = static_cast<int32_t>(result);
    n.where = where;
    n.state = q.State;
    n.protect = q.Protect;
    n.type = q.Type;
    n.allocation_protect = q.AllocationProtect;
    n.allocation_base = reinterpret_cast<uintptr_t>(q.AllocationBase);
    n.region_end = reinterpret_cast<uintptr_t>(q.BaseAddress) + q.RegionSize;
    n.written = kind == kRefused && where != kNoReservation;
    if (n.written) write_refusal_now(n);
    g_fault_note_ready[slot].store(true, std::memory_order_release);
}

// Logs the faults the handler noted since the last call. Called on the way into the XMem calls, never under a lock.
void drain_fault_notes() {
    if (g_fault_notes_shown.load(std::memory_order_relaxed) == g_fault_notes_taken.load(std::memory_order_relaxed) &&
        g_fault_kind_counts[kRefused].load(std::memory_order_relaxed) <
            g_refused_report_at.load(std::memory_order_relaxed))
        return;
    static std::mutex mutex;
    const std::lock_guard lock(mutex);
    if (const uint64_t refused = g_fault_kind_counts[kRefused].load(std::memory_order_relaxed);
        refused >= g_refused_report_at.load(std::memory_order_relaxed)) {
        std::fprintf(stderr,
                     "[rt] xmem: %llu faults left to the title so far; the first %u are noted below, the rest only "
                     "counted\n",
                     static_cast<unsigned long long>(refused), kFaultNoteCaps[kRefused]);
        g_refused_report_at.store(refused * 2, std::memory_order_relaxed);
    }
    uint32_t shown = g_fault_notes_shown.load(std::memory_order_relaxed);
    for (; shown < kFaultNoteSlots && g_fault_note_ready[shown].load(std::memory_order_acquire); ++shown) {
        const FaultNote& n = g_fault_notes[shown];
        const unsigned long long page = n.accessed & ~static_cast<uintptr_t>(kPhysicalPageSize - 1);
        if (n.written) {
            continue;  // the handler wrote it out at the fault
        } else if (n.kind == kPlanted) {
            std::fprintf(stderr, "[rt] xmem: window page 0x%llx now has a zero page: the title never mapped it\n", page);
        } else if (n.kind == kCommitted) {
            std::fprintf(stderr,
                         "[rt] xmem: mapped page 0x%llx committed at the fault: a section page in the title's "
                         "reservation with no view of xmem's\n",
                         page);
        } else {
            const char* const where = n.where == kInWindow        ? "a window"
                                      : n.where == kInPlaceholder ? "a title placeholder"
                                                                  : "no title reservation";
            std::fprintf(stderr,
                         "[rt] xmem: fault at 0x%llx (page 0x%llx, access %u) left to the title: commit %d, in %s, "
                         "host state 0x%lx protect 0x%lx type 0x%lx, allocation base 0x%llx (protect 0x%lx), run "
                         "ends 0x%llx\n",
                         static_cast<unsigned long long>(n.accessed), page, n.access, n.result, where,
                         static_cast<unsigned long>(n.state), static_cast<unsigned long>(n.protect),
                         static_cast<unsigned long>(n.type), static_cast<unsigned long long>(n.allocation_base),
                         static_cast<unsigned long>(n.allocation_protect),
                         static_cast<unsigned long long>(n.region_end));
        }
    }
    g_fault_notes_shown.store(shown, std::memory_order_relaxed);
}

// With g_fault_lock held: does a view of g_views cover `address`?
bool view_covers(uintptr_t address) {
    auto it = g_views.upper_bound(address);
    if (it == g_views.begin()) return false;
    --it;
    return address - it->first < it->second.count * kPhysicalPageSize;
}

// With g_fault_lock held exclusively. The host page at `accessed` is a reserved page of a section view xmem has no
// view of (g_views), in the title's reserved memory (a window or a placeholder of its own): commit it where the title
// reached. Not a scratch view of xmem's sections, which are host memory, nor anything outside the title's
// reservations (a driver's or another library's SEC_RESERVE view): a wild write there faults as it would without xmem.
bool commit_title_section_page(uintptr_t accessed) {
    MEMORY_BASIC_INFORMATION q{};
    if (!VirtualQuery(reinterpret_cast<const void*>(accessed), &q, sizeof(q)) || q.State != MEM_RESERVE ||
        q.Type != MEM_MAPPED || !in_title_reservation(accessed) ||
        g_scratch_views.count(reinterpret_cast<uintptr_t>(q.AllocationBase)))
        return false;
    const uintptr_t at = accessed & ~uintptr_t(0xfff);
    if (!VirtualAlloc(reinterpret_cast<void*>(at), 0x1000, MEM_COMMIT, PAGE_READWRITE)) return false;
    leonida::geometry_writes::notify(at, at + 0x1000, leonida::geometry_writes::Event::mapping);
    leonida::title_memory_map::changed(at, at + 0x1000);
    return true;
}

// Test seam (rt::xmem::set_fault_race_hook): called where another thread's map, unmap or plant of the faulting page can
// interleave with the handler (rt::xmem::FaultRaceStage).
std::atomic<rt::xmem::FaultRaceHook> g_fault_race_hook{nullptr};

void fault_race_point(int stage) {
    if (const auto hook = g_fault_race_hook.load(std::memory_order_acquire)) hook(stage);
}

// Test seam (rt::xmem::fault_exits): one count per exit of the handler, so a test can match the faults that reached
// the next handler against the ones the handler left to the title (and noted).
std::atomic<uint64_t> g_fault_exits[static_cast<int>(rt::xmem::FaultExit::count)];

LONG fault_exit(rt::xmem::FaultExit exit, LONG verdict) {
    g_fault_exits[static_cast<int>(exit)].fetch_add(1, std::memory_order_relaxed);
    return verdict;
}

bool commit_answers(PageCommit result, DWORD protect, ULONG_PTR access) {
    return result == PageCommit::committed_now || (result == PageCommit::committed_before && access_allowed(protect, access));
}

// Only the handler's read-only plant can be replaced on a write. Respect window and title-set protections.
bool writable_read_plant(uintptr_t address) {
    const auto view = g_views.find(address);
    if (view == g_views.end() || !view->second.zero || view->second.title_protected ||
        view->second.protect != PAGE_READONLY) return false;
    const auto* window = fault_window(address);
    return window && access_allowed(view_protect(*window), 1);
}

// A page of the title's reservation with no view of xmem's, found committed with a protection that allows the access:
// another thread committed it between the fault and the look (commit_title_section_page). The access runs again and
// goes through. Bounded: a thread that keeps faulting on one page this way (the query and the page tables disagreeing,
// which no Windows build is known to do) is left to the title after kMeanwhileRetries back-to-back retries, so it is
// noted instead of spinning silently.
constexpr unsigned kMeanwhileRetries = 8;
thread_local uintptr_t t_meanwhile_page = 0;
thread_local unsigned t_meanwhile_retries = 0;
bool committed_meanwhile(const MEMORY_BASIC_INFORMATION& page, ULONG_PTR access, uintptr_t accessed) {
    if (page.State != MEM_COMMIT || !access_allowed(page.Protect, access)) return false;
    const uintptr_t at = accessed & ~uintptr_t(0xfff);
    if (at != t_meanwhile_page) {
        t_meanwhile_page = at;
        t_meanwhile_retries = 0;
    }
    return ++t_meanwhile_retries <= kMeanwhileRetries;
}

// This thread is in physical_page_fault: a fault it takes there is passed on, not answered.
thread_local bool t_in_fault_handler = false;

// Every access violation in the process comes here first. The common ones (a first touch of a mapped physical page, a
// guard of the D3D12 layer's on a mapped page) are answered or passed on under the shared lock; only a write into a
// page of the title's reserved memory that is still a reservation on the host (State MEM_RESERVE, which no routine
// guard fault is) takes the exclusive lock.
//
// Every decision is made on what was read under the lock. Each map, unmap and plant of xmem's holds the lock
// exclusively from before it changes a page until the page is in its final state, and on Windows a page read during
// MapViewOfFile3 or UnmapViewOfFile2 can read as MEM_FREE, or as the mapping it is about to lose: a query made with no
// lock held could see a mapping that is gone by the time the handler acts on it, or a free page that is a mapped one
// a moment later, and leave a write into a mapped window page to the title.
LONG CALLBACK physical_page_fault(EXCEPTION_POINTERS* info) {
    using rt::xmem::FaultExit;
    const EXCEPTION_RECORD* const record = info ? info->ExceptionRecord : nullptr;
    // Before anything of this thread's is touched: most exceptions in the process are no access violation.
    if (!record || record->ExceptionCode != EXCEPTION_ACCESS_VIOLATION) return EXCEPTION_CONTINUE_SEARCH;
    if (record->NumberParameters < 2) return fault_exit(FaultExit::passed_foreign, EXCEPTION_CONTINUE_SEARCH);
    // A fault this thread takes while it holds g_fault_lock is not xmem's to answer: waiting for the lock again would
    // wait for itself.
    if (t_in_fault_handler || t_fault_lock_held) return fault_exit(FaultExit::nested, EXCEPTION_CONTINUE_SEARCH);
    t_in_fault_handler = true;
    const uintptr_t accessed = static_cast<uintptr_t>(record->ExceptionInformation[1]);
    const uintptr_t address = accessed & ~static_cast<uintptr_t>(kPhysicalPageSize - 1);
    const ULONG_PTR access = record->ExceptionInformation[0];
    DWORD protect = 0;
    PageCommit result;
    FaultWhere where = kNoReservation;
    MEMORY_BASIC_INFORMATION seen{};  // the page as the decision saw it
    char read_note[1024];
    DWORD read_note_size = 0;
    bool plant_read = false, promote = false, from_title = false;
    bool queried = false;
    {
        const FaultLockShared lock;
        result = commit_mapped_page(address, accessed, protect);
        promote = access == 1 && result == PageCommit::committed_before && protect == PAGE_READONLY &&
                  writable_read_plant(address);
        if (result == PageCommit::not_ours || result == PageCommit::failed || promote) {
            if (in_window(address))
                where = kInWindow;
            else if (in_title_reservation(accessed))
                where = kInPlaceholder;
            // Read under the lock, so it stays true until the lock is released: no page of a window or a placeholder of
            // the title's changes while it is held shared (each change of xmem's holds it exclusively, and a view of
            // xmem's at this page would have been found above). A committed mapped page with no view here is someone
            // else's view, not a mapping of xmem's in flight.
            if (where != kNoReservation || access == 1)
                queried = VirtualQuery(reinterpret_cast<const void*>(accessed), &seen, sizeof(seen)) != 0;
            const auto pc = reinterpret_cast<uintptr_t>(record->ExceptionAddress);
            from_title = access == 0 && result == PageCommit::not_ours && title_code(pc);
            if (from_title) {
                if (!queried) queried = VirtualQuery(reinterpret_cast<const void*>(accessed), &seen, sizeof(seen)) != 0;
                read_note_size = describe_read(read_note, pc, accessed, seen);
                plant_read = where == kInWindow && g_pool_zero_pages;
            }
            if (where != kNoReservation) fault_race_point(rt::xmem::kFaultDecided);
        }
    }
    if (read_note_size) {
        DWORD written;
        WriteFile(GetStdHandle(STD_ERROR_HANDLE), read_note, read_note_size, &written, nullptr);
    }
    const bool title_page = where != kNoReservation;
    // A write into memory that is only reserved: never a routine fault (a guard is committed), so noted below.
    const bool reserved_write = queried && access == 1 && seen.State == MEM_RESERVE;
    bool answered = commit_answers(result, protect, access);
    char plant_note[256];
    DWORD plant_note_size = 0;
    bool planted = false, committed = false, escalated = false;
    if (!answered && title_page && queried) {
        if (access == 1) fault_race_point(rt::xmem::kFaultAfterLook);
        if (committed_meanwhile(seen, access, accessed)) {
            answered = true;
        } else if ((access == 1 || plant_read) && (seen.State == MEM_RESERVE || seen.State == MEM_FREE || promote)) {
            // Only reserved: a window page the title never mapped, or a section page of the title's own not committed
            // yet. Free, under the shared lock, in memory the tables still name as the title's: not a change of xmem's
            // (none can be in flight), but decided again under the exclusive lock rather than passed on.
            fault_race_point(rt::xmem::kFaultBeforeExclusive);
            // Only title-code pool reads can wait for the whole physical mapping batch to publish its frames.
            std::unique_lock physical(g_phys_mutex, std::defer_lock);
            if (plant_read) physical.lock();
            const FaultLockExclusive lock;
            escalated = true;
            // Between the look and this lock another thread may have mapped, planted, committed or unmapped the page.
            queried = VirtualQuery(reinterpret_cast<const void*>(accessed), &seen, sizeof(seen)) != 0;
            if (view_covers(address)) {
                // Mapped, or planted by another thread's fault, since the look: answered as that view is. One that
                // still refuses is a protection the title set, not a page it never mapped.
                result = commit_mapped_page(address, accessed, protect);
                answered = commit_answers(result, protect, access);
                if (!answered && access == 1 && protect == PAGE_READONLY && writable_read_plant(address)) {
                    planted = remove_zero_view(g_views.find(address)) && map_zero_page(address, false);
                    answered = planted;
                }
            } else if (queried && seen.State == MEM_RESERVE) {
                // A write into a window page the title never mapped: back it with a zeroed page (map_zero_page). No
                // remap of this page is in flight while the lock is held.
                if (!plant_read || g_mapped.find(address) == g_mapped.end())
                    planted = map_zero_page(address, plant_read);
                committed = access == 1 && !planted && commit_title_section_page(accessed);
                answered = planted || committed;
            } else if (queried && committed_meanwhile(seen, access, accessed)) {
                answered = true;
            }
            if (planted) {
                auto* out = append_number(append_text(plant_note, "[rt] xmem: zero plant "), address, true);
                out = describe_twin(out, address);
                *out++ = '\n';
                plant_note_size = static_cast<DWORD>(out - plant_note);
            }
            // Otherwise free with no change of xmem's in flight (released behind the tables), or committed with a
            // protection that refuses the write: the title's.
        }
    }
    const LONG verdict = answered ? EXCEPTION_CONTINUE_EXECUTION : EXCEPTION_CONTINUE_SEARCH;
    FaultExit exit;
    if (planted) {
        DWORD written;
        WriteFile(GetStdHandle(STD_ERROR_HANDLE), plant_note, plant_note_size, &written, nullptr);
        note_fault(kPlanted, accessed, access, result, where, seen);
        exit = FaultExit::planted;
    } else if (committed) {
        note_fault(kCommitted, accessed, access, result, where, seen);
        exit = FaultExit::committed;
    } else if (answered) {
        exit = FaultExit::answered;
    } else if ((result == PageCommit::committed_before && !escalated) ||
               (title_page && access == 0 && result == PageCommit::not_ours && !from_title)) {
        exit = FaultExit::passed_guarded;  // a guard's (the D3D12 layer's), routine: not noted
    } else if (title_page || reserved_write) {
        note_fault(kRefused, accessed, access, result, where, seen);
        exit = FaultExit::left_to_title;
    } else {
        exit = FaultExit::passed_foreign;
    }
    t_in_fault_handler = false;
    return fault_exit(exit, verdict);
}

bool install_physical_page_fault_handler() {
    static const bool installed = [] {
        // The function-local statics the handler reaches are made here, before it can run: a first construction in the
        // handler, under the exclusive lock, would take the CRT's lock for the static's guard, allocate, and register
        // the destructor with atexit, all of which a faulting thread can be inside.
        placeholder_api();
        leonida::host_heap::handle();
        leonida::geometry_writes::state();
        char flag[4]{};
        g_pool_zero_pages = GetEnvironmentVariableA("LEONIDA_POOL_ZERO_PAGES", flag, sizeof(flag)) == 1 && flag[0] == '1';
        return AddVectoredExceptionHandler(1, &physical_page_fault) != nullptr;
    }();
    return installed;
}


bool aligned_page_address(uintptr_t address) { return (address & (kPhysicalPageSize - 1)) == 0; }

// gta6rt --trace FILE: the first XMemAllocatePhysicalPages for each distinct attribute word goes to the startup trace
// as an "xmem_physical_pages" event with the page count and the caller.
void note_physical_attributes(uint64_t attributes, uint64_t count, const void* caller) {
    if (!rt::startup_trace::enabled()) return;
    static std::mutex mutex;
    static std::set<uint64_t> seen;
    {
        const std::lock_guard lock(mutex);
        if (!seen.insert(attributes).second) return;
    }
    rt::startup_trace::event("xmem_physical_pages",
                             "\"api\":\"XMemAllocatePhysicalPages\",\"attributes\":" +
                                 rt::startup_trace::quote(rt::startup_trace::hex(attributes)) +
                                 ",\"pages\":" + std::to_string(count) + caller_field(caller));
}

// BOOL XMemAllocatePhysicalPages(attributes, UINT64* pageCount, UINT64* pageArray). Measured at four call sites: RCX is
// a constant attribute word, RDX points at the number of 64 KiB pages (every caller computes it as the byte size
// rounded up and shifted right by 16), R8 at an array that receives one page number per page. The result is tested as
// a 32-bit BOOL. After success every caller compares the count with the one it asked for and frees the pages if they
// differ, so the count is in/out.
//
// Each call is backed by one pagefile-backed section of count * 64 KiB: on Windows reserved and committed (charged) on
// touch, under Wine committed whole (a sparse temp file), given physical memory as it is touched, zero-filled. Pages keep their contents while unmapped or mapped elsewhere, and one page can
// be mapped at several addresses. ASSUMPTION: the attribute word (memory type, perhaps the page size) is not decoded;
// every page is 64 KiB, which all four callers' arithmetic agrees with.
BOOL WINAPI XMemAllocatePhysicalPages(uint64_t attributes, uint64_t* page_count, uint64_t* page_array) {
    drain_fault_notes();
    note_physical_attributes(attributes, page_count ? *page_count : 0, _ReturnAddress());
    if (!page_count || (*page_count && !page_array)) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return FALSE;
    }
    const uint64_t count = *page_count;
    if (count == 0) return TRUE;

    const std::lock_guard lock(g_phys_mutex);
    if (count > g_free_slots.size() + (kPageSlotLimit - g_next_slot)) {
        *page_count = 0;
        SetLastError(ERROR_NOT_ENOUGH_MEMORY);
        return FALSE;
    }
    if (!install_physical_page_fault_handler()) {
        *page_count = 0;
        SetLastError(ERROR_NOT_SUPPORTED);
        return FALSE;
    }
    const uint64_t bytes = count * kPhysicalPageSize;  // below 2^37: count is below 2^21
    HANDLE handle = file_backed_section(bytes);
    const bool use_file = handle != nullptr;
    // As in VIctory's xmem.rs, commit the section on Wine: its sparse temp file costs no RAM for untouched pages.
    // Querying or protecting reserved views takes a wineserver round trip under the process-wide virtual memory lock.
    // Windows charges committed sections against the commit limit (errors 1455/1450), so reserve them there.
    const DWORD flags = PAGE_READWRITE | (sections_committed() ? SEC_COMMIT : SEC_RESERVE);
    if (!handle) handle = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, flags,
                                             static_cast<DWORD>(bytes >> 32), static_cast<DWORD>(bytes), nullptr);
    if (!handle) {
        *page_count = 0;
        return FALSE;
    }
    // From host_heap: the unmap that drops the last reference frees it under g_fault_lock.
    const auto section = std::allocate_shared<Section>(leonida::host_heap::Allocator<Section>{}, handle, use_file);
    if (!section->scratch) {
        *page_count = 0;
        return FALSE;
    }
    // Diagnostic: where the host put each large section's scratch view, host memory a title write must never reach.
    static std::atomic<int> scratch_shown{0};
    if (count >= 256 && scratch_shown.fetch_add(1, std::memory_order_relaxed) < 64)
        std::fprintf(stderr, "[rt] xmem: %llu physical pages, scratch view %p size %llu\n",
                     static_cast<unsigned long long>(count), section->scratch, static_cast<unsigned long long>(bytes));
    section->frames.resize(count);
    for (uint64_t i = 0; i < count; ++i) {
        uint64_t slot;
        if (!g_free_slots.empty()) {
            slot = g_free_slots.back();
            g_free_slots.pop_back();
        } else {
            slot = g_next_slot++;
        }
        const uint64_t page = slot * kFramesPerPage;
        g_pages[page] = PhysicalPage{section, i * kPhysicalPageSize};
        section->frames[i] = static_cast<uint32_t>(page);
        page_array[i] = page;
    }
    return TRUE;
}

// BOOL XMemFreePhysicalPages(UINT64 pageCount, UINT64* pageArray). Measured at three call sites: RCX is the count (the
// one XMemAllocatePhysicalPages returned, or the sum of several allocations), RDX the array. The callers unmap the
// pages first; a page that is still mapped keeps its memory until it is unmapped. FALSE, with the entries before it
// freed, for a number that is not a live page.
BOOL WINAPI XMemFreePhysicalPages(uint64_t page_count, const uint64_t* page_array) {
    if (page_count && !page_array) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return FALSE;
    }
    const std::lock_guard lock(g_phys_mutex);
    for (uint64_t i = 0; i < page_count; ++i) {
        const auto it = g_pages.find(page_array[i]);
        if (it == g_pages.end()) {
            SetLastError(ERROR_INVALID_PARAMETER);
            return FALSE;
        }
        g_pages.erase(it);
        g_free_slots.push_back(page_array[i] / kFramesPerPage);
    }
    // No change to what any address shows (title_memory_map): the pages were unmapped first, and one still mapped
    // keeps its memory until it is.
    return TRUE;
}

// PVOID XMemMapPhysicalPages(PVOID address, UINT64 pageCount, UINT64* pageArray). Measured at twelve call sites: RCX
// is an address in a window XMemVirtualAlloc reserved, RDX a 64-bit count, R8 the pages to map there one after another,
// or null to unmap that many (four calls at shutdown). Callers test the 64-bit result for zero, so it is the address on
// success. A page number of 0 in the array unmaps that one page.
//
// Consecutive pages of one allocation are mapped as one view (MapViewOfFile3 into a placeholder of their size); a
// page remapped or unmapped later splits its view, and its neighbours stay mapped. The range is unmapped first, so a
// page number that is not a live page fails with the pages before it mapped (and recorded as mapped) and the rest of the
// range unmapped.
void* WINAPI XMemMapPhysicalPages(void* address, uint64_t page_count, const uint64_t* page_array) {
    const uintptr_t start = reinterpret_cast<uintptr_t>(address);
    if (!aligned_page_address(start) || page_count > (std::numeric_limits<uintptr_t>::max)() / kPhysicalPageSize ||
        start > (std::numeric_limits<uintptr_t>::max)() - page_count * kPhysicalPageSize) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return nullptr;
    }
    if (!placeholder_api().available()) {
        SetLastError(ERROR_NOT_SUPPORTED);
        return nullptr;
    }
    drain_fault_notes();
    const std::lock_guard lock(g_phys_mutex);
    const uintptr_t end = start + page_count * kPhysicalPageSize;
    const UnmapCause cause(page_array ? "XMemMapPhysicalPages (remap)" : "XMemMapPhysicalPages (unmap)");
    if (!page_array) {
        bool ok = false;
        {
            const FaultLockExclusive view_lock;
            ok = unmap_range_locked(start, end);
        }
        if (ok) leonida::geometry_writes::note_released(start, end);
        return ok ? address : nullptr;
    }
    const Window* const window = find_window(start, page_count * kPhysicalPageSize);
    if (!window) {
        std::fprintf(stderr, "[rt] xmem: map physical pages %p REFUSED: the range is not in one window\n", address);
        SetLastError(ERROR_INVALID_ADDRESS);
        return nullptr;
    }
    const DWORD protect = view_protect(*window);
    // The page numbers are the title's memory: copied once the window bounds their count, and before the fault lock,
    // because a read of them that faults would wait for that lock on this very thread. (The fault handler never takes
    // g_phys_mutex.)
    leonida::host_heap::vector<uint64_t> pages(page_array, page_array + page_count);
    // One exclusive section across the whole range: unmapped first, then mapped run by run, so a write fault on any of
    // these pages waits here and then sees its mapping instead of landing on a placeholder. Not split into chunks: a
    // fault in a part not mapped yet would plant a zero page that the map then discards, with the write in it. The
    // retail title's largest call (about 100,000 pages, once at load) holds it for about 100 capped views.
    uint64_t failed_i = 0, failed_done = 0;
    bool failed = false;
    DWORD failed_error = ERROR_SUCCESS;
    {
        const FaultLockExclusive view_lock;
        if (!unmap_range_locked(start, end)) return nullptr;
        fault_race_point(rt::xmem::kRemapBetween);
        for (uint64_t i = 0; i < page_count;) {
            if (pages[i] == 0) {
                ++i;
                continue;
            }
            const auto it = g_pages.find(pages[i]);
            if (it == g_pages.end()) {
                failed = true;
                failed_i = i;
                failed_error = ERROR_INVALID_PARAMETER;
                break;
            }
            View view{1, it->second.section, it->second.offset, protect};
            while (i + view.count < page_count) {
                const auto next = g_pages.find(pages[i + view.count]);
                if (next == g_pages.end() || next->second.section != view.section ||
                    next->second.offset != view.offset + view.count * kPhysicalPageSize) {
                    break;
                }
                ++view.count;
            }
            uint64_t done = 0;
            if (!map_view(start + i * kPhysicalPageSize, view, &done)) {
                failed = true;
                failed_i = i;
                failed_done = done;
                failed_error = GetLastError();
                if (failed_error == ERROR_SUCCESS) failed_error = ERROR_INVALID_ADDRESS;
                break;
            }
            i += view.count;
        }
    }
    // g_mapped and the aliases: host_heap, outside the fault lock. On a failure, for every page that did get mapped (the
    // ones before the failing page, and the part of its run that was): they stay mapped, and copy_page_mappings and
    // transfer_physical find them only here.
    const uint64_t mapped_end = failed ? failed_i + failed_done : page_count;
    for (uint64_t i = 0; i < mapped_end; ++i) {
        if (pages[i] == 0) continue;
        const uintptr_t at = start + i * kPhysicalPageSize;
        g_mapped[at] = pages[i];
        note_mapping(pages[i], at, true);
    }
    if (failed) {
        std::fprintf(stderr,
                     "[rt] xmem: map physical pages %p REFUSED: page %llu at %p (error %lu)\n", address,
                     static_cast<unsigned long long>(pages[failed_i]),
                     reinterpret_cast<void*>(start + failed_i * kPhysicalPageSize), failed_error);
        SetLastError(failed_error);
        return nullptr;
    }
    return address;
}

enum class ScatterFailure { missing_addresses, unsupported, unaligned, no_window, unmap, unknown_page, map };

// All callers hold g_phys_mutex. Preserve the API error even if the diagnostic write fails.
BOOL scatter_refused(ScatterFailure reason, uintptr_t address, DWORD error) {
    static constexpr const char* names[] = {"missing addresses", "unsupported", "unaligned", "no window",
                                            "unmap failed", "unknown page", "map failed"};
    static unsigned shown[std::size(names)]{};
    const auto index = static_cast<unsigned>(reason);
    if (shown[index] < 16) {
        ++shown[index];
        char line[512];
        char* out = append_text(append_text(line, "[rt] xmem: scatter refused: "), names[index]);
        out = append_number(append_text(out, " address="), address, true);
        out = append_number(append_text(out, " error="), error, false);
        auto next = g_windows.upper_bound(address);
        out = append_text(out, " window at/below=");
        if (next == g_windows.begin()) out = append_text(out, "none");
        else {
            const auto prev = std::prev(next);
            out = append_window(out, prev->first, prev->second);
        }
        out = append_text(out, " next=");
        out = next == g_windows.end() ? append_text(out, "none") : append_window(out, next->first, next->second);
        write_window_note(line, out);
    }
    SetLastError(error);
    return FALSE;
}

// BOOL XMemMapPhysicalPagesScatter(PVOID* addresses, UINT32 pageCount, UINT64* pageArray). Measured at two call sites:
// RCX is an array of addresses, EDX a 32-bit count, R8 the page for each address (an entry of 0, or a null array,
// unmaps). The result is tested as a 32-bit BOOL. Like MapUserPhysicalPagesScatter: entry i maps pageArray[i] at
// addresses[i].
BOOL WINAPI XMemMapPhysicalPagesScatter(void* const* addresses, uint32_t page_count, const uint64_t* page_array) {
    drain_fault_notes();
    const std::lock_guard lock(g_phys_mutex);
    if (page_count && !addresses) {
        return scatter_refused(ScatterFailure::missing_addresses, 0, ERROR_INVALID_PARAMETER);
    }
    if (!placeholder_api().available()) {
        return scatter_refused(ScatterFailure::unsupported, page_count ? reinterpret_cast<uintptr_t>(addresses[0]) : 0, ERROR_NOT_SUPPORTED);
    }
    for (uint32_t i = 0; i < page_count; ++i) {
        const uintptr_t address = reinterpret_cast<uintptr_t>(addresses[i]);
        if (!aligned_page_address(address)) {
            return scatter_refused(ScatterFailure::unaligned, address, ERROR_INVALID_PARAMETER);
        }
        const uint64_t page = page_array ? page_array[i] : 0;
        const UnmapCause cause(page ? "XMemMapPhysicalPagesScatter (remap)" : "XMemMapPhysicalPagesScatter (unmap)");
        if (page == 0) {
            if (!unmap_page(address)) return scatter_refused(ScatterFailure::unmap, address, GetLastError());
            continue;
        }
        const Window* const window = find_window(address, kPhysicalPageSize);
        if (!window) {
            return scatter_refused(ScatterFailure::no_window, address, ERROR_INVALID_ADDRESS);
        }
        if (!map_page(address, page, view_protect(*window), MapOwner::scatter)) {
            const DWORD error = GetLastError();
            return scatter_refused(g_pages.count(page) ? ScatterFailure::map : ScatterFailure::unknown_page,
                                   address, error);
        }
    }
    return TRUE;
}

// rt::xmem::copy_page_mappings_fn (xmem.h), for the console queue's CopyPageMappingsBatchX: each destination page gets
// a view of the physical page mapped at its source page, so both addresses are one memory, as two XMemMapPhysicalPages
// calls with the same page make them. Every source page is looked up before any destination page changes, so a source
// range that overlaps the destination maps what was there before the call. A destination in a GPU-only range is mapped
// at its paired CPU range too (title_gpu_views), and a source page missing at a GPU-only address is looked up there.
// A page of that CPU range the title mapped itself is the title's (title_mapped_twin): the copy leaves it as it is.

// With g_phys_mutex held. True when the paired CPU page at `address` holds a mapping of the title's own: its
// XMemMapPhysicalPages or XMemMapPhysicalPagesScatter call, or a slot-31 copy onto that address. The console's
// CopyPageMappingsBatchX changes only the GPU-only range's page table, so such a page keeps what the title put
// there; only a page the copy mirrored there itself (MapOwner::mirror), a zero plant, or a page with no mapping
// follows the GPU-only range. Each page kept is counted, and the first ones logged.
bool title_mapped_twin(uintptr_t address, uintptr_t gpu, bool unmap) {
    MapOwner owner;
    {
        const FaultLockShared faults;
        uintptr_t base = 0;
        const View* const view = covering_view(address, base);
        if (!view || view->zero || view->owner == MapOwner::mirror) return false;
        owner = view->owner;
    }
    static std::atomic<uint64_t> kept{0};
    const uint64_t n = kept.fetch_add(1, std::memory_order_relaxed) + 1;
    if (n <= 8 || (n & (n - 1)) == 0)
        std::fprintf(stderr,
                     "[rt] xmem: slot-31 %s at 0x%llx left the title's %s mapping at its paired CPU page 0x%llx "
                     "(%llu such pages so far)\n",
                     unmap ? "unmap" : "copy", static_cast<unsigned long long>(gpu),
                     owner == MapOwner::scatter ? "scatter" : owner == MapOwner::slot31 ? "slot-31" : "physical",
                     static_cast<unsigned long long>(address), static_cast<unsigned long long>(n));
    return true;
}

bool copy_page_mappings(void* destination, const void* source, uint64_t pages, bool repeat_first) {
    const uintptr_t to = reinterpret_cast<uintptr_t>(destination);
    const uintptr_t from = reinterpret_cast<uintptr_t>(source);
    constexpr uintptr_t kMax = (std::numeric_limits<uintptr_t>::max)();
    if (!aligned_page_address(to) || !aligned_page_address(from) || pages > kMax / kPhysicalPageSize ||
        to > kMax - pages * kPhysicalPageSize || (!repeat_first && from > kMax - pages * kPhysicalPageSize)) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return false;
    }
    if (!placeholder_api().available()) {
        SetLastError(ERROR_NOT_SUPPORTED);
        return false;
    }
    const std::lock_guard lock(g_phys_mutex);
    // The destination is usually a GPU-only reservation, which the D3D12 layer reads at its paired CPU view
    // (title_gpu_views::to_cpu); that view gets the same pages, or it would stay reserved and never readable.
    const uintptr_t cpu_to = leonida::title_gpu_views::to_cpu(to, pages * kPhysicalPageSize);
    const Window* const cpu_window = cpu_to != to ? find_window(cpu_to, pages * kPhysicalPageSize) : nullptr;
    if (!source) {
        for (uint64_t i = 0; i < pages; ++i) {
            const uintptr_t gpu = to + i * kPhysicalPageSize, cpu = cpu_to + i * kPhysicalPageSize;
            {
                const UnmapCause cause("CopyPageMappingsBatchX (unmap)");
                if (!unmap_page(gpu)) return false;
            }
            if (!cpu_window || title_mapped_twin(cpu, gpu, true)) continue;
            const UnmapCause cause("CopyPageMappingsBatchX (unmap paired CPU page)");
            if (!unmap_page(cpu)) return false;
        }
        return true;
    }
    const Window* const window = find_window(to, pages * kPhysicalPageSize);
    if (!window) {
        SetLastError(ERROR_INVALID_ADDRESS);
        return false;
    }
    std::vector<uint64_t> found(static_cast<size_t>(pages), 0);
    for (uint64_t i = 0; i < pages; ++i) {
        const uintptr_t src_va = from + (repeat_first ? 0 : i) * kPhysicalPageSize;
        auto it = g_mapped.find(src_va);
        if (it == g_mapped.end()) {  // a GPU-only source whose pages were mapped at its CPU view
            const uintptr_t cpu_src = leonida::title_gpu_views::to_cpu(src_va, kPhysicalPageSize);
            if (cpu_src != src_va) it = g_mapped.find(cpu_src);
        }
        if (it != g_mapped.end()) found[static_cast<size_t>(i)] = it->second;
    }
    const DWORD protect = view_protect(*window);
    const DWORD cpu_protect = cpu_window ? view_protect(*cpu_window) : 0;
    for (uint64_t i = 0; i < pages; ++i) {
        const uint64_t page = found[static_cast<size_t>(i)];
        const uintptr_t gpu = to + i * kPhysicalPageSize, cpu = cpu_to + i * kPhysicalPageSize;
        const UnmapCause cause(page ? "CopyPageMappingsBatchX (remap)" : "CopyPageMappingsBatchX (hole)");
        if (!map_page(gpu, page, protect, MapOwner::slot31)) return false;
        if (cpu_window && !title_mapped_twin(cpu, gpu, false) && !map_page(cpu, page, cpu_protect, MapOwner::mirror))
            return false;
    }
    return true;
}
[[maybe_unused]] const bool g_copy_page_mappings_set =(rt::xmem::copy_page_mappings_fn = &copy_page_mappings, true);

// The title's write watching, on reservations it made with MEM_WRITE_WATCH. One xmem write-watched for the shadows
// (reserve_watched) is refused as the host refuses a region that is not write-watched ((UINT)-1,
// ERROR_INVALID_PARAMETER), as the console would: it isn't write-watched there, and a reset here would clear the bits
// the shadows ask about.
bool shadows_write_watch(const void* base_address) {
    if (!leonida::title_write_watch::contains(reinterpret_cast<uintptr_t>(base_address))) return false;
    SetLastError(ERROR_INVALID_PARAMETER);
    return true;
}

UINT WINAPI XMemGetWriteWatch(DWORD flags, void* base_address, SIZE_T region_size, void** addresses, ULONG_PTR* count,
                              ULONG* granularity) {
    if (shadows_write_watch(base_address)) return UINT(-1);
    return GetWriteWatch(flags, base_address, region_size, addresses, count, granularity);
}

UINT WINAPI XMemResetWriteWatch(void* base_address, SIZE_T region_size) {
    if (shadows_write_watch(base_address)) return UINT(-1);
    return ResetWriteWatch(base_address, region_size);
}

// BOOL XMemTransferMemory(XMEM_TRANSFER_TYPE type, SIZE_T size). Its reference page and the XMEM_TRANSFER_TYPE enum
// page (gaming/gdk/docs/reference/system/xmem/{functions/xmemtransfermemory,enums/xmem_transfer_type}) are NDA
// topics on Microsoft Learn, so the shape comes from public code built against the GDK's xmem.h: Microsoft's
// XBOX-MonoGame-NetRumble sample (console/NetRumble.Bootstrap/Main.cpp) and FNA-XNA/NativeAOT-Xbox (Bootstrap/Main.cpp)
// both call `if (!XMemTransferMemory(XMemTransferTitleToSystem, additionalKernelPoolMemory))` with a `const size_t`,
// commented "Transfer additional memory to kernel pool" and "Must be multiples of 2MB sizes". So: two arguments, the
// enum (32 bits) in ECX and the size in RDX, and a zero return is failure. The public XLaunchNewGame page adds that
// "Memory transferred to the system using XMemTransferMemory (NDA topic) is persisted", and the game-save debugging
// page suggests it for E_OUTOFMEMORY "to transfer some memory to the system partition".
//
// On the console it moves budget between the title's partition and the system's. A PC has one pool and no budget
// split, so there is nothing to move: every direction and size returns TRUE and nothing is allocated or reserved. Not
// known (the pages are NDA): the numeric XMEM_TRANSFER_TYPE values, and whether the console refuses or rounds a size
// that is not a multiple of 2 MiB. Refusing would stop a title at startup (both samples return -1), while succeeding
// changes nothing here. TRUE is 1 in EAX, so it also reads as true if the real return type is a one-byte bool.
//
// gta6rt --trace FILE: the first call for each distinct (type, size) pair goes to the startup trace as an
// "xmem_transfer" event with the caller as module+RVA, so a run with the executable records the values it passes.
BOOL WINAPI XMemTransferMemory(uint32_t type, SIZE_T size) {
    static std::atomic<bool> warned{false};
    if (!warned.exchange(true, std::memory_order_relaxed)) {
        std::fprintf(stderr,
                     "[rt] xmem: XMemTransferMemory(type %u, %zu bytes): a PC has no title/system memory partition; "
                     "returning TRUE without moving anything (logged once)\n",
                     type, static_cast<size_t>(size));
    }
    if (rt::startup_trace::enabled()) {
        static std::mutex mutex;
        static std::set<std::pair<uint32_t, uint64_t>> seen;
        bool first = false;
        {
            const std::lock_guard lock(mutex);
            first = seen.insert({type, static_cast<uint64_t>(size)}).second;
        }
        if (first) {
            rt::startup_trace::event(
                "xmem_transfer",
                "\"api\":\"XMemTransferMemory\",\"type\":" + std::to_string(type) +
                    ",\"size\":" + std::to_string(static_cast<uint64_t>(size)) + ",\"caller\":" +
                    rt::startup_trace::quote(
                        rt::startup_trace::where(reinterpret_cast<uint64_t>(_ReturnAddress()))));
        }
    }
    return TRUE;
}

// kernel32!VirtualAlloc as a title calls it. The console's allocation type has two bits of its own, MEM_GRAPHICS
// (0x10000000, memory the GPU reads) and MEM_TITLE (0x40000000, the title's own heap). On a PC the host rejects both
// with ERROR_INVALID_PARAMETER (measured on Windows 11 26200: VirtualAlloc(NULL, 64 KiB, MEM_COMMIT | MEM_RESERVE |
// 0x10000000, PAGE_READWRITE) returns NULL; so does 0x40000000), so an engine allocating its heaps this way gets
// NULL at boot. Both are hints about where the memory lives; the host has no such split, so they are dropped and
// the rest is passed through. The PAGE_GRAPHICS_* protection bits are dropped (host_protect). Not done: whether
// graphics memory has to come from an arena the D3D12 layer can see (#313).
constexpr DWORD kMemGraphics = 0x10000000;
constexpr DWORD kMemTitle = 0x40000000;

// A new reservation is write-watched for the D3D12 layer's shadows (reserve_watched).
LPVOID WINAPI TitleVirtualAlloc(LPVOID address, SIZE_T size, DWORD allocation_type, DWORD protect) {
    const DWORD type = allocation_type & ~(kMemGraphics | kMemTitle);
    const DWORD host_prot = host_protect(protect);
    LPVOID result =
        reserve_watched(type, address, size, [&](DWORD t) { return VirtualAlloc(address, size, t, host_prot); });
    if (result) {
        leonida::title_memory_map::changed(result, size);
        leonida::title_va_aliases().add_title_memory(reinterpret_cast<uintptr_t>(result), size);
        note_unwatched_change(address, size, type, false);
        if (type & MEM_RESERVE) {
            const auto from = reinterpret_cast<uintptr_t>(result);
            leonida::geometry_writes::notify(from, from + size, leonida::geometry_writes::Event::replaced);
        }
    }
    if(result)track_label_data(result,size,false);
    // Diagnostic: the title's large reservations, so a fault the handler leaves to the title can be matched to the
    // reservation it is in (its allocation base).
    static std::atomic<int> shown{0};
    if (result && (type & MEM_RESERVE) && size >= (64u << 20) && shown.fetch_add(1, std::memory_order_relaxed) < 64)
        std::fprintf(stderr, "[rt] xmem: title VirtualAlloc reserve %p size %llu type 0x%lx protect 0x%lx\n", result,
                     static_cast<unsigned long long>(size), static_cast<unsigned long>(allocation_type),
                     static_cast<unsigned long>(protect));
    return result;
}

// The allocation a MEM_RELEASE at `address` frees: [address, end) over every region with that allocation base. False
// when `address` is not an allocation's base (VirtualFree then fails anyway).
bool allocation_extent(const void* address, uintptr_t& end) {
    MEMORY_BASIC_INFORMATION info{};
    if (!VirtualQuery(address, &info, sizeof(info)) || info.State == MEM_FREE || info.AllocationBase != address)
        return false;
    end = reinterpret_cast<uintptr_t>(info.BaseAddress) + info.RegionSize;
    for (int regions = 0; regions < 4096; ++regions) {
        MEMORY_BASIC_INFORMATION next{};
        if (!VirtualQuery(reinterpret_cast<const void*>(end), &next, sizeof(next)) || next.State == MEM_FREE ||
            next.AllocationBase != address)
            break;
        end = reinterpret_cast<uintptr_t>(next.BaseAddress) + next.RegionSize;
    }
    return true;
}

// A release of memory the title reserved as a window or a placeholder: the address range is the host's again, so it
// leaves the window and placeholder tables (a scratch view or a driver's view the host puts there later is not the
// title's). Under g_phys_mutex and the exclusive fault lock across the free and the tables: no section of xmem's (made
// under one of them) can land in the range while the tables still name it. A split (MEM_PRESERVE_PLACEHOLDER) or a
// merge (MEM_COALESCE_PLACEHOLDERS) keeps the reservation and is passed through.
BOOL title_release(void* address, SIZE_T size, DWORD free_type, const void* caller) {
    uintptr_t end = 0;
    const uintptr_t start = reinterpret_cast<uintptr_t>(address);
    bool tracked = false;
    if (allocation_extent(address, end)) {
        const FaultLockShared lock;
        tracked = table_overlaps(g_fault_windows, start, end) || table_overlaps(g_title_placeholders, start, end);
    }
    if (!tracked) return VirtualFree(address, size, free_type);
    BOOL result;
    {
        const std::lock_guard lock(g_phys_mutex);
        {
            const FaultLockExclusive view_lock;
            result = VirtualFree(address, size, free_type);
            if (result) {
                if (stream_gpu) {
                    for (const auto base : {stream_cpu, stream_gpu}) {
                        if (end > base && start < base + stream_size)
                            forget_range(stream_offsets, (std::max)(start, base) - base,
                                         (std::min)(end - base, stream_size));
                    }
                }
                forget_range(g_fault_windows, start, end);
                forget_range(g_title_placeholders, start, end);
            }
        }
        if (result) {
            auto it = g_windows.lower_bound(start);
            if (it != g_windows.begin()) --it;
            for (; it != g_windows.end() && it->first < end; ++it)
                if (it->first + it->second.size > start)
                    note_window_change(it->first, it->second, start, end - start, 0, caller, true);
            forget_range(g_windows, start, end);
        }
    }
    static std::atomic<int> shown{0};
    if (result && shown.fetch_add(1, std::memory_order_relaxed) < 64)
        std::fprintf(stderr, "[rt] xmem: the title released %p size %llu of its windows/placeholders\n", address,
                     static_cast<unsigned long long>(end - start));
    return result;
}

// The pages are freed with the GPU label regions locked (src/gpu_label_memory.h), so a completed label is never
// stored into pages that are going away.
BOOL title_free_impl(LPVOID address, SIZE_T size, DWORD free_type) {
    drain_fault_notes();
    const bool releases = (free_type & MEM_RELEASE) && !(free_type & (MEM_PRESERVE_PLACEHOLDER | MEM_COALESCE_PLACEHOLDERS));
    const uintptr_t start = reinterpret_cast<uintptr_t>(address);
    uintptr_t release_end = 0;
    if (releases && address) allocation_extent(address, release_end);
    uintptr_t decommit_end = 0;
    if ((free_type & MEM_DECOMMIT) && address) {
        decommit_end = size ? start + size : leonida::title_write_watch::end_of(start);
        if (decommit_end <= start) allocation_extent(address, decommit_end);
    }
    BOOL result;
    if (releases) {
        result = title_release(address, size, free_type, _ReturnAddress());
    } else if (free_type & (MEM_PRESERVE_PLACEHOLDER | MEM_COALESCE_PLACEHOLDERS)) {
        // A split or a merge of placeholders, which may be the title's windows: the pages it touches are part way
        // through changing until it returns, so the fault handler waits for it as it waits for a remap of xmem's.
        const FaultLockExclusive lock;
        result = VirtualFree(address, size, free_type);
    } else {
        result = VirtualFree(address, size, free_type);
    }
    // A release of size 0 frees the whole reservation, whose extent is not known here: a change anywhere.
    if (result && size) leonida::title_memory_map::changed(address, size);
    else if (result) leonida::title_memory_map::changed();
    if (result && (free_type & MEM_DECOMMIT)) note_unwatched_change(address, size, 0, true);
    if (result && (free_type & MEM_RELEASE) && !(free_type & MEM_PRESERVE_PLACEHOLDER))
        leonida::title_write_watch::remove(reinterpret_cast<uintptr_t>(address));
    if (result && releases && release_end > start)
        leonida::geometry_writes::note_released(start, release_end);
    if (result && (free_type & MEM_DECOMMIT) && decommit_end > start)
        leonida::geometry_writes::note_released(start, decommit_end);
    return result;
}

BOOL WINAPI TitleVirtualFree(LPVOID address, SIZE_T size, DWORD free_type) {
    DWORD saved = ERROR_SUCCESS;
    const bool ok = leonida::label::memory().release_pages(
        reinterpret_cast<std::uintptr_t>(address), size, (free_type & MEM_RELEASE) != 0, [&] {
            const BOOL result = title_free_impl(address, size, free_type);
            saved = GetLastError();
            return !!result;
        });
    SetLastError(saved);
    return ok ? TRUE : FALSE;
}
BOOL WINAPI TitleVirtualFreeEx(HANDLE process,LPVOID address,SIZE_T size,DWORD flags) {
    if(GetProcessId(process)!=GetCurrentProcessId())return ::VirtualFreeEx(process,address,size,flags);
    DWORD saved = ERROR_SUCCESS;
    const bool ok = leonida::label::memory().release_pages(
        reinterpret_cast<std::uintptr_t>(address), size, (flags & MEM_RELEASE) != 0, [&] {
            const BOOL result = title_free_impl(address, size, flags);
            saved = GetLastError();
            return !!result;
        });
    SetLastError(saved);return ok?TRUE:FALSE;
}

// The title makes pages of physical-page views in [start, start + size) read-only (or no-access) itself: those views
// are left as they are by reopen_unowned, which otherwise takes a read-only page there for a guard nobody reopened.
// Kept per view (a run of pages mapped together) and never cleared: coarse, but it only ever turns that safety net
// off for the view, never the title's protection.
void note_title_protect(uintptr_t start, SIZE_T size, DWORD protect) {
    switch (protect & 0xff) {
    case PAGE_READWRITE:
    case PAGE_EXECUTE_READWRITE:
    case PAGE_WRITECOPY:
    case PAGE_EXECUTE_WRITECOPY: return;  // writable: nothing for reopen_unowned to keep
    default: break;
    }
    const uintptr_t end = start + size;
    const FaultLockExclusive lock;  // the field only: no allocation, no title memory
    auto it = g_views.upper_bound(start);
    if (it != g_views.begin() && std::prev(it)->first + std::prev(it)->second.count * kPhysicalPageSize > start) --it;
    for (; it != g_views.end() && it->first < end; ++it) it->second.title_protected = true;
}

BOOL WINAPI TitleVirtualProtect(LPVOID address, SIZE_T size, DWORD protect, PDWORD old) {
    const auto start = reinterpret_cast<uintptr_t>(address);
    // Before the call: a fault on another thread in between must not take the title's protection for a lost guard.
    note_title_protect(start, size, protect);
    const BOOL result = VirtualProtect(address, size, protect, old);
    if (result) {
        leonida::title_memory_map::changed(start, start + size);
        leonida::geometry_writes::notify(start, start + size, leonida::geometry_writes::Event::replaced);
    }
    return result;
}

}  // namespace

// The host bytes of the 64 KiB page at `address` (the 64 KiB-aligned address of the page), with the section kept
// alive so that the bytes stay mapped after the lock is released. False when the page is not one of xmem's
// physical pages. The section's own mapping of a page is the host's copy of those bytes, which the CPU may write
// even where the title's own view of the page is PAGE_NOACCESS.
namespace {
struct HostPage {
    std::shared_ptr<Section> section;
    std::uint64_t offset = 0;
};

bool host_page(std::uintptr_t address, HostPage& out) {
    const std::lock_guard lock(g_phys_mutex);
    const auto mapped = g_mapped.find(address);
    if (mapped == g_mapped.end()) return false;
    const auto page = g_pages.find(mapped->second);
    if (page == g_pages.end() || !page->second.section || !page->second.section->scratch) return false;
    out.section = page->second.section;
    out.offset = page->second.offset;
    return true;
}
}  // namespace

namespace rt::xmem {

void note_stream_reservation(uintptr_t base, size_t size, uintptr_t caller_rva) {
    if (caller_rva != 0x1e1d00f && caller_rva != 0x1e1d03e) return;
    const std::lock_guard physical(g_phys_mutex);
    const FaultLockExclusive faults;
    if (caller_rva == 0x1e1d00f) {
        stream_cpu = base;
        stream_size = size;
        stream_gpu = 0;
        stream_offsets.clear();
    } else if (stream_cpu && size == stream_size && !stream_gpu) {
        stream_gpu = base;
        stream_offsets.emplace(0, Window{size, 0});
        char value[2]{};
        mirror_stream = GetEnvironmentVariableA("LEONIDA_SLOT31_CPU_TWIN", value, sizeof(value)) == 1 && value[0] == '1';
    }
    rt::placed_image_map_note::publish_pair(stream_gpu, stream_cpu, stream_size);
}

bool queue_page_mappings_enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("LEONIDA_QUEUE_PAGE_MAPPINGS");
        return !value || std::strcmp(value, "0") != 0;
    }();
    return enabled;
}

// Snapshot all source frame numbers under the physical-map lock before replacing a destination.
// This also permits the source pool and destinations to overlap. Unmap/skip ranges never read start.
bool slot31_runs() {
    static const bool on = [] {
        const char* v = std::getenv("LEONIDA_SLOT31_RUNS");
        return v && std::strcmp(v, "1") == 0;
    }();
    return on;
}

bool copy_page_mappings(std::uint32_t count, const PageBatch* batches, std::uintptr_t pool,
                        std::uint32_t pool_pages, std::uint32_t flags) {
    // Each refusal says why (the first 16): the queue call above it reports only E_FAIL.
    uint32_t at_batch = 0, at_range = 0;
    const auto refuse = [&](DWORD error, int site, uintptr_t address, uint32_t kind) {
        static std::atomic<int> shown{0};
        if (shown.fetch_add(1, std::memory_order_relaxed) < 16) {
            MEMORY_BASIC_INFORMATION q{};
            VirtualQuery(reinterpret_cast<const void*>(address), &q, sizeof(q));
            std::fprintf(stderr,
                         "[rt] xmem: copy page mappings refused (check %d): batch %u range %u kind %u at 0x%llx, "
                         "error %lu; host state 0x%lx type 0x%lx allocation base %p, %s window, %s view\n",
                         site, at_batch, at_range, kind, static_cast<unsigned long long>(address), error,
                         static_cast<unsigned long>(q.State), static_cast<unsigned long>(q.Type), q.AllocationBase,
                         find_window(address, kPhysicalPageSize) ? "in a" : "no", [&] {
                             const FaultLockShared lock;
                             return view_covers(address);
                         }() ? "a" : "no");
        }
        SetLastError(error);
        return false;
    };
    const auto fail = [](DWORD error) { SetLastError(error); return false; };
    if (flags || (count && !batches)) return fail(ERROR_INVALID_PARAMETER);
    if (!count) return true;
    if (!placeholder_api().available()) return fail(ERROR_NOT_SUPPORTED);
    struct Range {
        uintptr_t at;
        PageRange spec;
        size_t first_frame;
        DWORD protect;
    };
    std::vector<Range> plan;
    std::vector<uint64_t> frames;
    drain_fault_notes();
    const std::lock_guard lock(g_phys_mutex);
    try {
        for (uint32_t b = 0; b < count; ++b) {
            const auto& batch = batches[b];
            auto at = batch.destination;
            at_batch = b;
            at_range = 0;
            if (!aligned_page_address(at) || (batch.count && !batch.ranges))
                return refuse(ERROR_INVALID_PARAMETER, 1, at, 0);
            for (uint32_t i = 0; i < batch.count; ++i) {
                const auto r = batch.ranges[i];
                const auto bytes = uint64_t(r.count) * kPhysicalPageSize;
                at_range = i;
                if (r.kind > 3 || at > UINTPTR_MAX - bytes) return refuse(ERROR_INVALID_PARAMETER, 2, at, r.kind);
                if (r.count && r.kind != 3) {
                    // A single queue range can span both sides of a carved window. Validate all fragments
                    // before changing any mappings, and give each run its own window protection.
                    for (uint32_t offset = 0; offset < r.count;) {
                        const auto dest = at + uint64_t(offset) * kPhysicalPageSize;
                        auto it = g_windows.upper_bound(dest);
                        if (it == g_windows.begin()) return refuse(ERROR_INVALID_ADDRESS, 3, dest, r.kind);
                        --it;
                        const auto end = it->first + it->second.size;
                        if (dest >= end || end - dest < kPhysicalPageSize)
                            return refuse(ERROR_INVALID_ADDRESS, 3, dest, r.kind);
                        const auto pages = static_cast<uint32_t>((std::min)(uint64_t(r.count - offset),
                                                                          (end - dest) / kPhysicalPageSize));
                        plan.push_back({dest, {r.kind, r.start, pages}, frames.size() + (r.kind == 0 ? offset : 0),
                                        view_protect(it->second)});
                        offset += pages;
                    }
                    if (r.kind < 2) {
                        const uint32_t n = r.kind == 1 ? 1 : r.count;
                        if (!pool || !aligned_page_address(pool) || r.start > pool_pages || n > pool_pages - r.start ||
                            pool > UINTPTR_MAX - (uint64_t(r.start) + n) * kPhysicalPageSize)
                            return refuse(ERROR_INVALID_PARAMETER, 4, pool + uint64_t(r.start) * kPhysicalPageSize,
                                          r.kind);
                        const auto source = pool + uint64_t(r.start) * kPhysicalPageSize;
                        for (uint32_t j = 0; j < n; ++j) {
                            const auto src = source + uint64_t(j) * kPhysicalPageSize;
                            const auto it = g_mapped.find(src);
                            if (it == g_mapped.end()) {
                                // A hole in a valid source window unmaps the destination (CLHelper, RT-QUEUE31).
                                if (!find_window(src, kPhysicalPageSize))
                                    return refuse(ERROR_INVALID_ADDRESS, 5, src, r.kind);
                                frames.push_back(0);
                            } else {
                                if (!g_pages.contains(it->second)) return refuse(ERROR_INVALID_ADDRESS, 6, src, r.kind);
                                frames.push_back(it->second);
                            }
                        }
                    }
                }
                at += bytes;
            }
        }
        at_batch = at_range = UINT32_MAX;  // a map refusal names its address only
        // The mirror reads the published GPU frames, including any prefix left by a host failure.
        const auto apply = [&](const Range& r, uintptr_t gpu_source) {
            const bool mirror = gpu_source != 0;
            const auto bytes = uint64_t(r.spec.count) * kPhysicalPageSize;
            bool ok = true;
            bool changed_mirror = false;
            uintptr_t failed_at = r.at;
            if (!mirror && r.spec.kind == 2) {
                const UnmapCause cause("CopyPageMappingsBatchX (unmap range)");
                ok = unmap_range(r.at, r.at + bytes);
                if (ok) leonida::geometry_writes::note_released(r.at, r.at + bytes);
            } else if (!mirror && !slot31_runs()) {
                for (uint32_t j = 0; j < r.spec.count; ++j) {
                    const auto at = r.at + uint64_t(j) * kPhysicalPageSize;
                    const auto frame = frames[r.first_frame + (r.spec.kind == 0 ? j : 0)];
                    const auto existing = g_mapped.find(at);
                    if (existing != g_mapped.end() && existing->second == frame) continue;
                    const UnmapCause cause(frame ? "CopyPageMappingsBatchX (remap)" : "CopyPageMappingsBatchX (hole)");
                    if (!map_page(at, frame, r.protect, MapOwner::slot31)) {
                        ok = false;
                        break;
                    }
                }
            } else {
                // LEONIDA_SLOT31_RUNS=1 (off by default: speed-19 spent 1.6-6.3 s a call mapping with it, and its
                // load had not ended by 340 s): runs of pages whose mapping changes are unmapped once and mapped as
                // one view for each stretch of
                // consecutive pages of one section, not page by page: unmapping one page out of a larger view
                // unmaps that view and maps its remainder again, so a page-by-page remap of a run cost several
                // MapViewOfFile3 and UnmapViewOfFile2 calls a page (each a wineserver round trip under Wine), all under
                // g_phys_mutex, which the streaming workers and the D3D12 layer wait for.
                const auto frame_at = [&](uint32_t j) -> uint64_t {
                    if (!mirror) return frames[r.first_frame + (r.spec.kind == 0 ? j : 0)];
                    const auto it = g_mapped.find(gpu_source + uint64_t(j) * kPhysicalPageSize);
                    return it == g_mapped.end() ? 0 : it->second;
                };
                const auto changes = [&](uint32_t j) {
                    const auto existing = g_mapped.find(r.at + uint64_t(j) * kPhysicalPageSize);
                    if (mirror) {
                        const auto frame = frame_at(j);
                        if (existing == g_mapped.end()) return frame != 0;
                        if (existing->second == frame) return false;
                        const FaultLockShared faults;
                        uintptr_t base = 0;
                        const auto* v = covering_view(existing->first, base);
                        if (!v || v->owner != MapOwner::mirror) return false;
                    }
                    // Even without a frame number the fault handler may have planted a zero page.
                    return existing == g_mapped.end() || existing->second != frame_at(j);
                };
                for (uint32_t j = 0; j < r.spec.count && ok;) {
                    if (!changes(j)) {
                        ++j;
                        continue;
                    }
                    uint32_t end = j + 1;
                    while (end < r.spec.count && changes(end)) ++end;
                    const uintptr_t from = r.at + uint64_t(j) * kPhysicalPageSize;
                    failed_at = from;
                    changed_mirror = mirror;
                    const FaultLockExclusive faults;
                    {
                        const UnmapCause cause("CopyPageMappingsBatchX (remap run)");
                        ok = unmap_range_locked(from, r.at + uint64_t(end) * kPhysicalPageSize);
                    }
                    for (uint32_t k = j; k < end && ok;) {
                        const uint64_t frame = frame_at(k);
                        if (!frame) {  // a hole: left unmapped
                            ++k;
                            continue;
                        }
                        const auto first = g_pages.find(frame);
                        if (first == g_pages.end()) {
                            SetLastError(ERROR_INVALID_PARAMETER);
                            ok = false;
                            break;
                        }
                        uint32_t n = 1;
                        while (k + n < end && n < kMaxViewPages) {
                            const auto next = g_pages.find(frame_at(k + n));
                            if (next == g_pages.end() || next->second.section != first->second.section ||
                                next->second.offset != first->second.offset + uint64_t(n) * kPhysicalPageSize)
                                break;
                            ++n;
                        }
                        const uintptr_t at = r.at + uint64_t(k) * kPhysicalPageSize;
                        uint64_t done = 0;
                        ok = map_view(at, View{n, first->second.section, first->second.offset, r.protect, false, false,
                                               mirror ? MapOwner::mirror : MapOwner::slot31}, &done);
                        const DWORD map_error = ok ? ERROR_SUCCESS : GetLastError();
                        for (uint32_t i = 0; i < done; ++i) {
                            g_mapped[at + uint64_t(i) * kPhysicalPageSize] = frame_at(k + i);
                            note_mapping(frame_at(k + i), at + uint64_t(i) * kPhysicalPageSize, true);
                        }
                        if (!ok) {
                            failed_at = at + done * kPhysicalPageSize;
                            SetLastError(map_error);
                            break;
                        }
                        k += n;
                    }
                    j = end;
                }
            }
            const DWORD error = ok ? ERROR_SUCCESS : GetLastError();
            // Publish after changing the mapping, including partial host failures.
            if (!mirror || changed_mirror) leonida::title_texel_writes::note(reinterpret_cast<void*>(r.at), bytes);
            if (!ok) return refuse(error, 7, failed_at, r.spec.kind);
            return true;
        };
        for (const auto& r : plan) {
            const bool ok = apply(r, 0);
            const auto error = ok ? ERROR_SUCCESS : GetLastError();
            const auto end = r.at + uint64_t(r.spec.count) * kPhysicalPageSize;
            if (mirror_stream && stream_gpu && end > stream_gpu && r.at < stream_gpu + stream_size) {
                for (const auto& [offset, extent] : stream_offsets) {
                    auto gpu = (std::max)(r.at, stream_gpu + offset);
                    const auto stop = (std::min)(end, stream_gpu + offset + extent.size);
                    while (gpu < stop) {
                        const auto cpu = stream_cpu + (gpu - stream_gpu);
                        auto it = g_windows.upper_bound(cpu);
                        if (it == g_windows.begin()) return refuse(ERROR_INVALID_ADDRESS, 8, cpu, r.spec.kind);
                        --it;
                        const auto cpu_end = it->first + it->second.size;
                        if (cpu >= cpu_end) return refuse(ERROR_INVALID_ADDRESS, 8, cpu, r.spec.kind);
                        const auto bytes = (std::min)(stop - gpu, cpu_end - cpu);
                        if (bytes < kPhysicalPageSize) return refuse(ERROR_INVALID_ADDRESS, 8, cpu, r.spec.kind);
                        Range twin{cpu, {0, 0, static_cast<uint32_t>(bytes / kPhysicalPageSize)}, 0, view_protect(it->second)};
                        if (!apply(twin, gpu)) return false;
                        gpu += bytes;
                    }
                }
            }
            if (!ok) return fail(error);
        }
    } catch (const std::bad_alloc&) {
        return fail(ERROR_NOT_ENOUGH_MEMORY);
    }
    return true;
}

void geometry_aliases(std::uintptr_t va, std::vector<std::uintptr_t>& addresses) {
    AddressList snapshot;
    {
        const std::shared_lock aliases(g_alias_lock);
        const auto peers = g_alias_peers.find(va);
        if (peers == g_alias_peers.end()) return;
        snapshot = peers->second;
    }
    // The caller's vector may allocate on the process heap. Fault-side writers wait for the alias lock.
    addresses.insert(addresses.end(), snapshot.begin(), snapshot.end());
}

std::uint64_t zero_pages_taken() { return g_zero_pages.next; }

void set_fault_race_hook(FaultRaceHook hook) { g_fault_race_hook.store(hook, std::memory_order_release); }

std::uint64_t fault_exits(FaultExit exit) {
    const int at = static_cast<int>(exit);
    return at >= 0 && at < static_cast<int>(FaultExit::count) ? g_fault_exits[at].load(std::memory_order_relaxed) : 0;
}

std::uintptr_t physical_page_scratch(std::uint64_t page) {
    const std::lock_guard lock(g_phys_mutex);
    const auto it = g_pages.find(page);
    if (it == g_pages.end() || !it->second.section->scratch) return 0;
    return reinterpret_cast<std::uintptr_t>(it->second.section->scratch) + it->second.offset;
}

bool commit_physical_range(const void* address, size_t size) {
    if (size == 0) return true;
    const uintptr_t start = reinterpret_cast<uintptr_t>(address);
    if (start > (std::numeric_limits<uintptr_t>::max)() - (size - 1)) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return false;
    }
    if (sections_committed()) return true;
    const uintptr_t mask = ~static_cast<uintptr_t>(kPhysicalPageSize - 1);
    const uintptr_t first = start & mask;
    const uintptr_t pages = (((start + (size - 1)) & mask) - first) / kPhysicalPageSize + 1;
    const FaultLockShared lock;
    for (uintptr_t i = 0; i < pages; ++i) {
        DWORD protect = 0;
        const uintptr_t page = first + i * kPhysicalPageSize;
        if (commit_mapped_page(page, page, protect) == PageCommit::failed) return false;
    }
    return true;
}

std::shared_ptr<std::uint8_t> pin_mapped_data(const void* address, std::size_t size) {
    const auto start = reinterpret_cast<std::uintptr_t>(address);
    if (!size || !start || size > UINTPTR_MAX - start) return {};
    const auto end = start + size;
    const auto first = start & ~std::uintptr_t(kPhysicalPageSize - 1);
    const FaultLockShared lock;
    // The view that holds a page: a run of pages is one view, or several after a partial unmap.
    const auto view_of = [](std::uintptr_t page) {
        auto it = g_views.upper_bound(page);
        if (it == g_views.begin()) return g_views.end();
        --it;
        return (page - it->first) / kPhysicalPageSize < it->second.count ? it : g_views.end();
    };
    const auto base = view_of(first);
    if (base == g_views.end() || !base->second.section || !base->second.section->scratch) return {};
    const auto section = base->second.section;
    const auto offset = base->second.offset + (first - base->first);
    // A view the title mapped writable can be read-only for a while: the D3D12 layer guards the pages its shadows
    // copy and opens them again at the title's next write.
    bool writable = true;
    for (auto page=first; page<end; page+=kPhysicalPageSize) {
        const auto it = view_of(page);
        if (it==g_views.end() || it->second.section != section || it->second.offset+(page-it->first) != offset+page-first) return {};
        if ((it->second.protect & 255)!=PAGE_READWRITE && (it->second.protect & 255)!=PAGE_WRITECOPY) writable = false;
        DWORD previous{};
        const auto ready = commit_mapped_page(page, std::max(page,start), previous);
        if (ready!=PageCommit::committed_before && ready!=PageCommit::committed_now) return {};
    }
    for (auto at=start; at<end;) {
        MEMORY_BASIC_INFORMATION info{};
        if (!VirtualQuery(reinterpret_cast<const void*>(at), &info, sizeof(info)) ||
            info.Type!=MEM_MAPPED || info.State!=MEM_COMMIT || (info.Protect & (PAGE_NOACCESS|PAGE_GUARD)) ||
            ((info.Protect & 255)!=PAGE_READWRITE && (info.Protect & 255)!=PAGE_WRITECOPY &&
             !(writable && (info.Protect & 255)==PAGE_READONLY))) return {};
        at=reinterpret_cast<std::uintptr_t>(info.BaseAddress)+info.RegionSize;
    }
    return {section,static_cast<std::uint8_t*>(section->scratch)+offset+(start-first)};
}

// Writes `size` bytes at `address` in the title's memory, through the host pages xmem manages: the section's own
// mapping of the physical page, which the CPU may write even where the title's own view of the page is
// PAGE_NOACCESS (the GPU memory windows are). False when the range is not one of xmem's pages, or when the pages
// cannot be committed.
bool write_title_bytes(const void* address, const void* bytes, std::size_t size) {
    if (!size) return true;
    if (!address || !bytes) return false;
    if (size > 2 * kPhysicalPageSize) return false;  // a value write is four or eight bytes
    if (!commit_physical_range(address, size)) return false;
    const auto start = reinterpret_cast<std::uintptr_t>(address);
    const std::uintptr_t mask = ~static_cast<std::uintptr_t>(kPhysicalPageSize - 1);
    // Every page of the range is resolved before anything is written, so a range that is not entirely xmem's leaves
    // no half-written value behind. The section is kept alive by the handle, so the bytes stay mapped while they
    // are written.
    HostPage pages[2];
    std::size_t count = 0;
    for (std::size_t done = 0; done < size;) {
        const std::uintptr_t at = start + done;
        const std::uintptr_t page = at & mask;
        if (count == 2 || !host_page(page, pages[count])) return false;
        ++count;
        done += kPhysicalPageSize - static_cast<std::size_t>(at - page);
    }
    std::size_t done = 0;
    for (std::size_t index = 0; done < size; ++index) {
        const std::uintptr_t at = start + done;
        const std::uintptr_t page = at & mask;
        const std::size_t offset = static_cast<std::size_t>(at - page);
        std::size_t chunk = kPhysicalPageSize - offset;
        if (chunk > size - done) chunk = size - done;
        std::uint8_t* const host = static_cast<std::uint8_t*>(pages[index].section->scratch) + pages[index].offset;
        // An aligned value in one store, as kernel_writes::store makes it: a title thread polling the same physical
        // page through another view never reads half of it.
        if (chunk == size && size == 4 && !(offset & 3))
            std::atomic_ref(*reinterpret_cast<std::uint32_t*>(host + offset))
                .store(*static_cast<const std::uint32_t*>(bytes), std::memory_order_relaxed);
        else if (chunk == size && size == 8 && !(offset & 7))
            std::atomic_ref(*reinterpret_cast<std::uint64_t*>(host + offset))
                .store(*static_cast<const std::uint64_t*>(bytes), std::memory_order_relaxed);
        else
            std::memcpy(host + offset, static_cast<const std::uint8_t*>(bytes) + done, chunk);
        done += chunk;
    }
    // Through the section, not the title's views: no fault tells a guard of them (a shadow's cells) of the write.
    leonida::geometry_writes::notify(start, start + size, leonida::geometry_writes::Event::write);
    return true;
}

// The D3D12 layer calls this through leonida::value_sink (src/value_write_sink.h), so that the D3D12 library does not
// have to link the xmem shim into every consumer of that library. A process that links the shim gets the write; one
// that does not drops a value write that its host store could not land.
[[maybe_unused]] const bool g_title_write_set = (leonida::value_sink::set_title_pages(&write_title_bytes), true);

// The label target of an address in the pages the title maps: those pages are the title's memory, and the
// console's value writes name them (src/gpu_label_memory.h). The region is tracked on demand, one per page, so
// the label path finds a page by its address. The page tables are read under g_phys_mutex, and the region is
// tracked after that lock is released: the label path takes the two locks the other way round.
bool label_target(std::uintptr_t address, unsigned bytes, leonida::label::Target& out) {
    if (!bytes || (bytes != 4 && bytes != 8) || (address & (bytes - 1))) return false;
    const auto page = address & ~static_cast<std::uintptr_t>(kPhysicalPageSize - 1);
    {
        const std::lock_guard lock(g_phys_mutex);
        const auto mapped = g_mapped.find(page);
        if (mapped == g_mapped.end()) return false;
        const auto entry = g_pages.find(mapped->second);
        if (entry == g_pages.end() || !entry->second.section || !entry->second.section->scratch) return false;
    }
    // The region is tracked on demand, one per page. A region that is already there is kept: a target that is
    // still in flight keeps the region it was captured from, and a replaced region would drop it.
    if (!leonida::label::memory().capture(page, 4).owner.lock())
        leonida::label::memory().track(page, kPhysicalPageSize, page, false);
    out = leonida::label::memory().capture(address, bytes);
    return static_cast<bool>(out.owner.lock());
}

[[maybe_unused]] const bool g_label_target_set = (rt::xmem::label_target_fn = &label_target, true);

static bool commit_pages_locked(const uint64_t* page_array, size_t count) {
    if (count == 0) return true;
    if (!page_array) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return false;
    }
    size_t first = 0;
    while (first < count) {
        const auto it = g_pages.find(page_array[first]);
        if (it == g_pages.end() || !it->second.section->scratch) {
            SetLastError(ERROR_INVALID_PARAMETER);
            return false;
        }
        if (sections_committed()) {
            ++first;  // Already committed, but transfer_physical still needs every page number validated.
            continue;
        }
        size_t last = first + 1;
        while (last < count) {
            const auto next = g_pages.find(page_array[last]);
            if (next == g_pages.end() || next->second.section != it->second.section ||
                next->second.offset != it->second.offset + (last - first) * kPhysicalPageSize) {
                break;
            }
            ++last;
        }
        void* const backing = static_cast<uint8_t*>(it->second.section->scratch) + it->second.offset;
        if (!it->second.section->file_backed && !VirtualAlloc(backing, (last - first) * kPhysicalPageSize, MEM_COMMIT, PAGE_READWRITE)) return false;
        // Committing the scratch backing affects every title alias. Their addresses are unknown here.
        leonida::title_memory_map::changed();
        first = last;
    }
    return true;
}

bool commit_physical_pages(const uint64_t* page_array, size_t count) {
    const std::lock_guard lock(g_phys_mutex);
    return commit_pages_locked(page_array, count);
}

bool transfer_physical(std::span<const uint64_t> pages, uint16_t offset, std::span<uint8_t> staging, bool write) {
    if (staging.empty()) return true;
    if (pages.size() != 1 + (staging.size() - 1 + offset) / kPhysicalPageSize) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return false;
    }
    // A write goes through the section, which no guard sees: a slot-133 shadow that keeps one of the page's addresses
    // read-only, and so takes it for unchanged, would miss it. So every address the pages are mapped at is pinned and
    // opened first, as for a kernel write (the shadows copy it until the pin ends, then guard it again). Without
    // g_phys_mutex: opening waits for the shadows' locks, which a refresh holds for its walk, and XMem calls would
    // wait as long. An address mapped in between is a new mapping, which the shadows and the geometry watch copy anyway.
    std::vector<uint64_t> pins;
    if (write) {
        std::vector<std::pair<uintptr_t, size_t>> ranges;
        {
            const std::lock_guard lock(g_phys_mutex);
            size_t left = staging.size();
            uint16_t at = offset;
            for (const auto page : pages) {
                const size_t bytes = (std::min)(left, size_t(kPhysicalPageSize - at));
                if (auto mapped = g_page_addresses.find(page); mapped != g_page_addresses.end())
                    for (uintptr_t address : mapped->second) ranges.emplace_back(address + at, bytes);
                left -= bytes;
                at = 0;
            }
        }
        for (const auto& [address, bytes] : ranges)
            if (const uint64_t pin = leonida::kernel_writes::begin_write(reinterpret_cast<const void*>(address), bytes,
                                                                         nullptr))
                pins.push_back(pin);
    }
    const auto end_pins = [&] {
        for (const uint64_t pin : pins) leonida::kernel_writes::end_write(pin, false);
    };
    const std::lock_guard lock(g_phys_mutex);
    if (!commit_pages_locked(pages.data(), pages.size())) {  // also: every page is still allocated
        const DWORD error = GetLastError();
        end_pins();
        SetLastError(error);
        return false;
    }
    for (const auto page : pages) {
        const auto& backing = g_pages.at(page);
        auto* ptr = static_cast<uint8_t*>(backing.section->scratch) + backing.offset + offset;
        const size_t bytes = (std::min)(staging.size(), size_t(kPhysicalPageSize - offset));
        if (write) {
            std::memcpy(ptr, staging.data(), bytes);
            if (auto mapped = g_page_addresses.find(page); mapped != g_page_addresses.end()) {
                for (uintptr_t address : mapped->second) {
                    leonida::geometry_writes::notify(address + offset, address + offset + bytes, leonida::geometry_writes::Event::write);
                    leonida::title_texel_writes::note(reinterpret_cast<void*>(address + offset), bytes);
                    if (leonida::dstorage::trace_enabled())
                        std::fprintf(stderr, "[rt] ds-scatter cpu=%llx bytes=%zu page=%llu\n",
                                     static_cast<unsigned long long>(address + offset), bytes,
                                     static_cast<unsigned long long>(page));
                }
            }
        } else {
            std::memcpy(staging.data(), ptr, bytes);
        }
        staging = staging.subspan(bytes);
        offset = 0;
    }
    end_pins();
    return true;
}

// kernel_writes' unguarder (src/title_page_watch.h): the pages of [begin, end) in physical-page views whose window is
// writable but that are read-only now are made writable again. A page there is read-only only while a guard of the
// D3D12 layer (a slot-133 shadow, a geometry tile) keeps it so, and that guard reopens it at a write fault or before a
// kernel write; one still read-only after those had no guard left to answer for it, and would fail every write into it
// for good. Views the title protected itself (note_title_protect) are left alone. The shadows walk their pages again
// and the geometry tiles over the range count as written. True when a page of a writable view in the range is writable
// now. Called from the write-fault handler: under g_fault_lock shared, no allocation and nothing logged.
bool reopen_unowned(std::uintptr_t begin, std::uintptr_t end) {
    if (begin >= end) return false;
    constexpr uintptr_t kHostPage = 0x1000;
    bool writable = false, opened = false;
    {
        const FaultLockShared lock;
        auto it = g_views.upper_bound(begin);
        if (it != g_views.begin() && std::prev(it)->first + std::prev(it)->second.count * kPhysicalPageSize > begin)
            --it;
        for (; it != g_views.end() && it->first < end; ++it) {
            const View& view = it->second;
            if (view.title_protected || view.protect != PAGE_READWRITE && view.protect != PAGE_EXECUTE_READWRITE)
                continue;  // the title's protection, or a view no write may reach
            const DWORD read_only = view.protect == PAGE_READWRITE ? PAGE_READONLY : PAGE_EXECUTE_READ;
            const uintptr_t view_end = it->first + view.count * kPhysicalPageSize;
            const uintptr_t to = (std::min)(view_end, (end + kHostPage - 1) & ~(kHostPage - 1));
            for (uintptr_t at = (std::max)(it->first, begin & ~(kHostPage - 1)); at < to;) {
                MEMORY_BASIC_INFORMATION info{};
                if (!VirtualQuery(reinterpret_cast<const void*>(at), &info, sizeof(info))) break;
                const uintptr_t next = (std::min)(to, reinterpret_cast<uintptr_t>(info.BaseAddress) + info.RegionSize);
                if (next <= at) break;
                if (info.State == MEM_COMMIT && info.Type == MEM_MAPPED) {
                    DWORD old = 0;
                    if (info.Protect == read_only &&
                        VirtualProtect(reinterpret_cast<void*>(at), next - at, view.protect, &old)) {
                        opened = writable = true;
                    } else if (info.Protect == view.protect) {
                        writable = true;  // opened meanwhile (by its guard, or by another thread's call)
                    }
                }
                at = next;
            }
        }
    }
    if (opened) {
        // A shadow that kept such a page read-only and clean walks it again; a geometry tile over it is refilled.
        leonida::title_memory_map::changed(begin, end);
        leonida::geometry_writes::notify(begin, end, leonida::geometry_writes::Event::write);
    }
    return writable;
}

}

#define TITLE_VIRTUAL_MEMORY(dll)                                                               \
    {dll, "VirtualAlloc", reinterpret_cast<void*>(&TitleVirtualAlloc)},                          \
    {dll, "VirtualFree", reinterpret_cast<void*>(&TitleVirtualFree)},                            \
    {dll, "VirtualFreeEx", reinterpret_cast<void*>(&TitleVirtualFreeEx)},                        \
    {dll, "VirtualProtect", reinterpret_cast<void*>(&TitleVirtualProtect)}

// The classic libraries and the memory api sets a title's imports name.
SHIM_TABLE(title_memory,
    TITLE_VIRTUAL_MEMORY("kernel32.dll"),
    TITLE_VIRTUAL_MEMORY("kernelbase.dll"),
    TITLE_VIRTUAL_MEMORY("api-ms-win-core-memory-l1-1-0.dll"),
    TITLE_VIRTUAL_MEMORY("api-ms-win-core-memory-l1-1-1.dll"),
    TITLE_VIRTUAL_MEMORY("api-ms-win-core-memory-l1-1-2.dll"),
    TITLE_VIRTUAL_MEMORY("api-ms-win-core-memory-l1-1-3.dll"),
    TITLE_VIRTUAL_MEMORY("api-ms-win-core-memory-l1-1-4.dll"),
    TITLE_VIRTUAL_MEMORY("api-ms-win-core-memory-l1-1-5.dll"),
    TITLE_VIRTUAL_MEMORY("api-ms-win-core-memory-l1-1-6.dll"),
);

SHIM_TABLE(xmem,
    {"xmem.dll", "XMemAlloc", reinterpret_cast<void*>(&XMemAlloc)},
    {"xmem.dll", "XMemFree", reinterpret_cast<void*>(&XMemFree)},
    {"xmem.dll", "XMemVirtualQuery", reinterpret_cast<void*>(&XMemVirtualQuery)},
    {"xmem.dll", "XMemGetWorkingSetStatistics", reinterpret_cast<void*>(&XMemGetWorkingSetStatistics)},
    {"xmem.dll", "XMemVirtualAlloc", reinterpret_cast<void*>(&XMemVirtualAlloc)},
    {"xmem.dll", "XMemGetWriteWatch", reinterpret_cast<void*>(&XMemGetWriteWatch)},
    {"xmem.dll", "XMemResetWriteWatch", reinterpret_cast<void*>(&XMemResetWriteWatch)},
    {"xmem.dll", "XMemTransferMemory", reinterpret_cast<void*>(&XMemTransferMemory)},
    {"xmem.dll", "XMemAllocDefault", reinterpret_cast<void*>(&XMemAllocDefault)},
    {"xmem.dll", "XMemFreeDefault", reinterpret_cast<void*>(&XMemFreeDefault)},
    {"xmem.dll", "XMemIsToolsMemAvailable", reinterpret_cast<void*>(&XMemIsToolsMemAvailable)},
    {"xmem.dll", "XMemAllocatePhysicalPages", reinterpret_cast<void*>(&XMemAllocatePhysicalPages)},
    {"xmem.dll", "XMemFreePhysicalPages", reinterpret_cast<void*>(&XMemFreePhysicalPages)},
    {"xmem.dll", "XMemMapPhysicalPages", reinterpret_cast<void*>(&XMemMapPhysicalPages)},
    {"xmem.dll", "XMemMapPhysicalPagesScatter", reinterpret_cast<void*>(&XMemMapPhysicalPagesScatter)},
);
