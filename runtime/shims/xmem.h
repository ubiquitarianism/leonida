#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include "../src/gpu_label_memory.h"  // leonida::label::Target: the console's value writes (list slots 71 and 72)

namespace rt::xmem {

// CopyPageMappingsBatchX, Xbox queue slot 31. Measured x64 layout (also documented by VIctory).
struct PageRange {
    std::uint32_t kind, start, count;
};
struct PageBatch {
    std::uintptr_t destination;
    std::uint32_t count;
    const PageRange* ranges;
};
static_assert(sizeof(PageRange) == 12 && sizeof(PageBatch) == 24 && offsetof(PageBatch, ranges) == 16);

// Record the two measured stream-pool reservation return sites; other callers are ignored.
void note_stream_reservation(std::uintptr_t base, std::size_t size, std::uintptr_t caller_rva);
bool queue_page_mappings_enabled();
bool slot31_runs();
bool copy_page_mappings(std::uint32_t count, const PageBatch* batches, std::uintptr_t pool,
                        std::uint32_t pool_pages, std::uint32_t flags);

void geometry_aliases(std::uintptr_t va, std::vector<std::uintptr_t>& addresses);
bool commit_physical_range(const void* address, std::size_t size);
// A lifetime-pinned alias of a contiguous range of the title's writable mapped physical pages.
std::shared_ptr<std::uint8_t> pin_mapped_data(const void* address, std::size_t size);
// Test seam: section pages the zero-page pool has handed out. A replaced plant returns its page, so a
// plant-and-replace loop does not grow this.
std::uint64_t zero_pages_taken();
// Test seam: the address of physical page `page` in its section's scratch view (xmem's own mirror of the section), or
// 0 for a page that is not live. A write there must fault: it is not the title's memory.
std::uintptr_t physical_page_scratch(std::uint64_t page);
// Test seam: called at the points of a fault where another thread's map, unmap or plant of the faulting page can
// interleave with it (FaultRaceStage). Null: none. The hook runs on the thread named by the stage, with the locks that
// stage says are held: it may wait on an event, but must not allocate, log or fault.
enum FaultRaceStage : int {
    kFaultAfterLook = 0,       // the handler, after the shared look, with its lock released
    kFaultBeforeExclusive = 1, // the handler, just before the exclusive lock (a write the look left to it)
    kFaultDecided = 2,         // the handler, inside the shared look after the query it decides on: the lock still held
    kRemapBetween = 3,         // XMemMapPhysicalPages, between the unmap and the map, the exclusive lock held
};
using FaultRaceHook = void (*)(int stage);
void set_fault_race_hook(FaultRaceHook hook);
// Test seam: how the fault handler left the access violations it saw, one count per exit (all since the start).
enum class FaultExit : int {
    nested,          // taken while this thread held the fault lock, or inside the handler: passed on unanswered
    answered,        // a mapped page committed, or found committed and allowed (another thread's map or commit)
    planted,         // a window page the title never mapped, backed by a zero page
    committed,       // a section page in the title's reservation with no view of xmem's, committed
    passed_guarded,  // a committed page whose protection refuses the access, or a host guarded read of an unmapped page
    passed_foreign,  // memory that is not the title's reservation and not only reserved: someone else's fault
    left_to_title,   // the title's reservation, or a write into memory that is only reserved: noted (FaultNote)
    count
};
std::uint64_t fault_exits(FaultExit exit);
bool commit_physical_pages(const std::uint64_t* page_array, std::size_t count);
// leonida::kernel_writes' unguarder (src/title_page_watch.h): pages of physical-page views in [begin, end) left
// read-only with no guard to reopen them are made writable again. True when such a page there is writable now.
bool reopen_unowned(std::uintptr_t begin, std::uintptr_t end);

// Copy a validated DirectStorage staging buffer through the page table. The page list has
// one frame number per 64 KiB page. All pages are resolved before a write starts.
bool transfer_physical(std::span<const std::uint64_t> pages, std::uint16_t offset,
                       std::span<std::uint8_t> staging, bool write);

// Console queue slot 31, CopyPageMappingsBatchX (d3d12_xs_device.cpp): maps at `destination` the physical pages that
// are mapped at `source` now, `pages` 64 KiB pages one after another, or the one page at `source` every time with
// `repeat_first`. A source page with nothing mapped leaves its destination page unmapped; a null `source` unmaps them
// all. False, with the last error set, for an unaligned address or a destination outside a physical-page window.
// Set by xmem.cpp when it is linked: the command-object proxy is also compiled into tests without it.
using CopyPageMappingsFn = bool (*)(void* destination, const void* source, std::uint64_t pages, bool repeat_first);
inline CopyPageMappingsFn copy_page_mappings_fn = nullptr;

// Writes `size` bytes at `address` in the title's memory, through the host pages xmem manages: its own mapping of the
// physical page, which the CPU may write where the title's view of the page is PAGE_NOACCESS. False when the range is
// not xmem's pages. Registered with leonida::value_sink (src/value_write_sink.h) as the fallback of the console's
// value writes (list slots 71 and 72), so that the D3D12 layer reaches it without linking this shim.
bool write_title_bytes(const void* address, const void* bytes, std::size_t size);

// The label target of an address in the pages the title maps: those pages are the title's memory too, and the
// console's value writes name them (runtime/src/vulkan_gpu_labels.h). The shim owns the page tables, so it answers
// the question. False when the address is not in one of the title's mapped pages.
using LabelTargetFn = bool (*)(std::uintptr_t address, unsigned bytes, leonida::label::Target& out);
inline LabelTargetFn label_target_fn = nullptr;

}
