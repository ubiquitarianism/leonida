// Xbox device slot 133 (CreatePlacedResourceX) on xmem physical pages that were mapped but never touched. xmem
// commits physical pages on first touch, so such a range is still reserved when the title places a buffer on it, and
// the Vulkan layer's import (rt::d3d12::create_buffer_at_host_address) refuses reserved pages and answers with its
// fallback buffer: a shadow that copies only committed pages, so a reserved page reads as zeros and a write to it
// faults. With rt::xmem::commit_physical_range registered as the slot's
// pages hook (rt::d3d12_xs::set_placed_at_va_pages_hook, as gta6rt does), the pages are committed first: the buffer's
// GPU VA is the window itself and a GPU copy reads what the CPU writes there afterwards.
// Needs VK_EXT_external_memory_host (skipped otherwise). No shaders or game data.
#include "xmem_section_state.h"
#include "vk_test_device.h"
#include "d3d12_test_shaders.h"
#include "../shims/d3d12_com.h"
#include "../shims/d3d12_xs_device.h"
#include "../shims/shims.h"
#include "../shims/xmem.h"

#include "../src/title_page_watch.h"
#include <array>
#include <chrono>
#include <cstring>
#include <future>
#include <thread>
#include <vector>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "%s line %d: %s\n", who, __LINE__, #condition); return 1; } \
} while (0)

namespace {

const char* const who = "xmem slot-133 buffer";

template <class T>
struct Ref {
    T* p = nullptr;
    ~Ref() { if (p) p->Release(); }
    T* operator->() const { return p; }
};

using XMemVirtualAllocFn = void*(WINAPI*)(void*, SIZE_T, DWORD, ULONGLONG, DWORD);
using XMemAllocatePhysicalPagesFn = BOOL(WINAPI*)(std::uint64_t, std::uint64_t*, std::uint64_t*);
using XMemFreePhysicalPagesFn = BOOL(WINAPI*)(std::uint64_t, const std::uint64_t*);
using XMemMapPhysicalPagesFn = void*(WINAPI*)(void*, std::uint64_t, const std::uint64_t*);
using SlotFn = std::uint64_t(WINAPI*)(void*, std::uint64_t, std::uint64_t, std::uint64_t, std::uint64_t,
                                      std::uint64_t, std::uint64_t);
constexpr DWORD kXmem64kPages = 0x20400000;
constexpr SIZE_T kPage64k = 0x10000;
constexpr std::uint64_t kPages = 4;

bool state_is(const void* address, DWORD state) {
    MEMORY_BASIC_INFORMATION info{};
    return VirtualQuery(address, &info, sizeof(info)) && info.State == state;
}

// Slot 133 as the title calls it: (va, desc, layout, clear, riid, out).
HRESULT place(void* device, void* va, const D3D12_RESOURCE_DESC& desc, ID3D12Resource** out) {
    void* const* vtable = *static_cast<void* const* const*>(device);
    const GUID iid = __uuidof(ID3D12Resource);
    const std::uint64_t r = reinterpret_cast<SlotFn>(vtable[133])(
        device, reinterpret_cast<std::uint64_t>(va), reinterpret_cast<std::uint64_t>(&desc), 0xFFFFFFFFull, 0,
        reinterpret_cast<std::uint64_t>(&iid), reinterpret_cast<std::uint64_t>(out));
    return static_cast<HRESULT>(static_cast<std::uint32_t>(r));
}

int gpu_reads_cpu_writes(ID3D12Device* real, ID3D12Resource* buffer, const void* pages, UINT64 size) {
    D3D12_RESOURCE_DESC desc = buffer->GetDesc();
    D3D12_HEAP_PROPERTIES readback_heap{D3D12_HEAP_TYPE_READBACK};
    Ref<ID3D12Resource> readback;
    EXPECT(SUCCEEDED(real->CreateCommittedResource(&readback_heap, D3D12_HEAP_FLAG_NONE, &desc,
                                                   D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&readback.p))));
    D3D12_COMMAND_QUEUE_DESC queue_desc{D3D12_COMMAND_LIST_TYPE_DIRECT};
    Ref<ID3D12CommandQueue> queue;
    Ref<ID3D12CommandAllocator> allocator;
    Ref<ID3D12GraphicsCommandList> list;
    Ref<ID3D12Fence> fence;
    EXPECT(SUCCEEDED(real->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue.p))));
    EXPECT(SUCCEEDED(real->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p))));
    EXPECT(SUCCEEDED(real->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, nullptr,
                                             IID_PPV_ARGS(&list.p))));
    list->CopyBufferRegion(readback.p, 0, buffer, 0, size);
    EXPECT(SUCCEEDED(list->Close()));
    ID3D12CommandList* lists[] = {list.p};
    queue->ExecuteCommandLists(1, lists);
    EXPECT(SUCCEEDED(real->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))));
    EXPECT(SUCCEEDED(queue->Signal(fence.p, 1)));
    HANDLE done = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    EXPECT(SUCCEEDED(fence->SetEventOnCompletion(1, done)));
    EXPECT(WaitForSingleObject(done, 10000) == WAIT_OBJECT_0);
    CloseHandle(done);
    void* mapped = nullptr;
    EXPECT(SUCCEEDED(readback->Map(0, nullptr, &mapped)));
    const bool same = std::memcmp(mapped, pages, static_cast<size_t>(size)) == 0;
    readback->Unmap(0, nullptr);
    EXPECT(same);
    return 0;
}

int queue_remap(ID3D12Device* real, ID3D12Device* proxy) {
    using rt::xmem::PageRange;
    using rt::xmem::PageBatch;
    auto reserve = reinterpret_cast<XMemVirtualAllocFn>(rt::shims::find("xmem.dll", "XMemVirtualAlloc"));
    auto allocate = reinterpret_cast<XMemAllocatePhysicalPagesFn>(rt::shims::find("xmem.dll", "XMemAllocatePhysicalPages"));
    auto map = reinterpret_cast<XMemMapPhysicalPagesFn>(rt::shims::find("xmem.dll", "XMemMapPhysicalPages"));
    auto free_pages = reinterpret_cast<XMemFreePhysicalPagesFn>(rt::shims::find("xmem.dll", "XMemFreePhysicalPages"));
    constexpr unsigned n = 6;
    auto window = [&](DWORD protect) {
        return static_cast<std::uint8_t*>(reserve(nullptr, n * kPage64k, MEM_RESERVE | kXmem64kPages, 0, protect));
    };
    auto* pool = window(PAGE_READWRITE);
    auto* cpu = window(PAGE_READWRITE);
    auto* dst = window(0x801);
    EXPECT(pool && cpu && dst && leonida::title_gpu_views::other_view(uintptr_t(dst)) == uintptr_t(cpu));
    // xmem pairs dst with cpu as the legacy kind (read at the CPU view); this window is the kind slot 31 maps by itself.
    EXPECT(leonida::title_gpu_views::to_cpu(uintptr_t(dst)) == uintptr_t(cpu));
    EXPECT(leonida::title_gpu_views::set_independent_pages(uintptr_t(dst)));
    EXPECT(leonida::title_gpu_views::to_cpu(uintptr_t(dst)) == uintptr_t(dst));
    std::uint64_t pages[n]{}, count = n;
    EXPECT(allocate(0x280, &count, pages) && count == n);
    EXPECT(map(pool, n, pages) && map(cpu, n, pages) && map(dst, n, pages));
    for (unsigned i = 0; i < n; ++i) pool[i * kPage64k] = static_cast<std::uint8_t>(10 + i);

    D3D12_COMMAND_QUEUE_DESC desc{D3D12_COMMAND_LIST_TYPE_COPY};
    Ref<ID3D12CommandQueue> queue;
    Ref<ID3D12Fence> gate;
    EXPECT(SUCCEEDED(proxy->CreateCommandQueue(&desc, IID_PPV_ARGS(&queue.p))));
    EXPECT(SUCCEEDED(real->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&gate.p))));
    auto* real_queue = rt::d3d12_xs::queue_proxy_real_queue(queue.p);
    EXPECT(real_queue && SUCCEEDED(real_queue->Wait(gate.p, 1)));
    rt::d3d12_xs::set_copy_page_mappings_hook(&rt::xmem::copy_page_mappings);
    using Copy = HRESULT(WINAPI*)(void*, UINT, const PageBatch*, std::uintptr_t, UINT, UINT);
    auto call = reinterpret_cast<Copy>((*reinterpret_cast<void***>(queue.p))[31]);
    const PageRange first[] = {{0, 1, 2}, {1, 4, 2}};
    const PageRange second[] = {{2, UINT32_MAX, 1}, {3, UINT32_MAX, 1}};
    const PageBatch batches[] = {{uintptr_t(dst), 2, first}, {uintptr_t(dst + 4 * kPage64k), 2, second}};
    std::promise<void> started;
    auto done = std::async(std::launch::async, [&] {
        started.set_value();
        return call(queue.p, 2, batches, uintptr_t(pool), n, 0);
    });
    started.get_future().wait();
    const bool waited = done.wait_for(std::chrono::milliseconds(100)) == std::future_status::timeout;
    const auto before = dst[0];
    EXPECT(SUCCEEDED(gate->Signal(1)));
    EXPECT(SUCCEEDED(done.get()) && before == 10);
    // With list values the call can return after queuing the remap. Its gate must still order later queue work.
    const char* list_values = std::getenv("LEONIDA_LIST_VALUES");
    if (list_values && !std::strcmp(list_values, "0")) EXPECT(waited);
    EXPECT(SUCCEEDED(real_queue->Signal(gate.p, 2)));
    HANDLE remapped = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    EXPECT(remapped && SUCCEEDED(gate->SetEventOnCompletion(2, remapped)));
    const auto completion = WaitForSingleObject(remapped, 10000);
    CloseHandle(remapped);
    EXPECT(completion == WAIT_OBJECT_0);
    EXPECT(dst[0] == 11 && dst[kPage64k] == 12 && dst[2 * kPage64k] == 14 && dst[3 * kPage64k] == 14);
    EXPECT(state_is(dst + 4 * kPage64k, MEM_RESERVE) && dst[5 * kPage64k] == 15);
    dst[2 * kPage64k] = 93;
    EXPECT(pool[4 * kPage64k] == 93 && dst[3 * kPage64k] == 93 && cpu[0] == 10 && cpu[4 * kPage64k] == 93);

    // Snapshot overlapping source/destination before changing either. Reject a late invalid range atomically.
    const PageRange overlap[] = {{0, 0, 1}, {0, 1, 1}};
    const PageBatch same_pool{uintptr_t(dst + kPage64k), 2, overlap};
    EXPECT(SUCCEEDED(call(queue.p, 1, &same_pool, uintptr_t(dst), n, 0)));
    EXPECT(dst[kPage64k] == 11 && dst[2 * kPage64k] == 12);
    const PageRange invalid[] = {{2, 0, 1}, {0, n, 1}};
    const PageBatch bad{uintptr_t(dst), 2, invalid};
    EXPECT(FAILED(call(queue.p, 1, &bad, uintptr_t(pool), n, 0)) && dst[0] == 11);
    bool overflow = false;
    auto writes = leonida::title_texel_writes::take(overflow);
    EXPECT(!overflow && !writes.empty());
    EXPECT(map(dst, n, nullptr) && map(cpu, n, nullptr) && map(pool, n, nullptr));
    EXPECT(free_pages(n, pages));
    rt::d3d12_xs::set_copy_page_mappings_hook(nullptr);
    std::puts("queue slot 31: ordered two batches, four range kinds, shared physical data, independent CPU view, overlap and invalid input passed");
    return 0;
}

// Queue slot 31 as a retail run takes it: no batch hook, so the queue's default adapter sends each range to xmem's
// range callback (copy_page_mappings_fn), which mirrors a legacy-paired GPU-only destination at its CPU view. The
// source is a GPU-only pool backed only through its CPU alias, and the title maps a page of its own at one CPU twin
// while the remap waits behind queue work (deferred with list values, waited for without them). When the remap is
// applied, the GPU-only pages change and the title's CPU page keeps its bytes, through the copy and the unmap after it.
int queue_remap_default_route(ID3D12Device* real, ID3D12Device* proxy) {
    using rt::xmem::PageBatch;
    using rt::xmem::PageRange;
    auto reserve = reinterpret_cast<XMemVirtualAllocFn>(rt::shims::find("xmem.dll", "XMemVirtualAlloc"));
    auto allocate = reinterpret_cast<XMemAllocatePhysicalPagesFn>(rt::shims::find("xmem.dll", "XMemAllocatePhysicalPages"));
    auto map = reinterpret_cast<XMemMapPhysicalPagesFn>(rt::shims::find("xmem.dll", "XMemMapPhysicalPages"));
    auto free_pages = reinterpret_cast<XMemFreePhysicalPagesFn>(rt::shims::find("xmem.dll", "XMemFreePhysicalPages"));
    constexpr unsigned n = 2;
    auto window = [&](DWORD protect) {
        return static_cast<std::uint8_t*>(reserve(nullptr, n * kPage64k, MEM_RESERVE | kXmem64kPages, 0, protect));
    };
    // Two pairs, each a CPU range and then the GPU-only range of its size: the pool's and the destination's.
    auto* pool_cpu = window(PAGE_READWRITE);
    auto* pool = window(0x801);
    auto* cpu = window(PAGE_READWRITE);
    auto* dst = window(0x801);
    EXPECT(pool_cpu && pool && cpu && dst);
    EXPECT(leonida::title_gpu_views::to_cpu(uintptr_t(pool)) == uintptr_t(pool_cpu));
    EXPECT(leonida::title_gpu_views::to_cpu(uintptr_t(dst)) == uintptr_t(cpu));
    std::uint64_t pages[n + 1]{}, count = n + 1;
    EXPECT(allocate(0x280, &count, pages) && count == n + 1);
    EXPECT(map(pool_cpu, n, pages));  // the pool's pages are mapped at its CPU alias only
    for (unsigned i = 0; i < n; ++i) pool_cpu[i * kPage64k] = static_cast<std::uint8_t>(40 + i);

    D3D12_COMMAND_QUEUE_DESC desc{D3D12_COMMAND_LIST_TYPE_COPY};
    Ref<ID3D12CommandQueue> queue;
    Ref<ID3D12Fence> gate;
    EXPECT(SUCCEEDED(proxy->CreateCommandQueue(&desc, IID_PPV_ARGS(&queue.p))));
    EXPECT(SUCCEEDED(real->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&gate.p))));
    auto* real_queue = rt::d3d12_xs::queue_proxy_real_queue(queue.p);
    EXPECT(real_queue && SUCCEEDED(real_queue->Wait(gate.p, 1)));
    rt::d3d12_xs::set_copy_page_mappings_hook(nullptr);  // the retail route: the default adapter
    using Copy = HRESULT(WINAPI*)(void*, UINT, const PageBatch*, std::uintptr_t, UINT, UINT);
    auto call = reinterpret_cast<Copy>((*reinterpret_cast<void***>(queue.p))[31]);
    const PageRange copy_run[] = {{0, 0, n}};
    const PageBatch copy_batch{uintptr_t(dst), 1, copy_run};
    auto done = std::async(std::launch::async, [&] { return call(queue.p, 1, &copy_batch, uintptr_t(pool), n, 0); });
    // While the remap waits behind the held work, the title maps a page of its own at the second CPU twin.
    EXPECT(map(cpu + kPage64k, 1, &pages[n]));
    cpu[kPage64k] = 77;
    EXPECT(SUCCEEDED(gate->Signal(1)));
    EXPECT(SUCCEEDED(done.get()));
    EXPECT(SUCCEEDED(real_queue->Signal(gate.p, 2)));
    HANDLE remapped = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    EXPECT(remapped && SUCCEEDED(gate->SetEventOnCompletion(2, remapped)));
    const auto completion = WaitForSingleObject(remapped, 10000);
    CloseHandle(remapped);
    EXPECT(completion == WAIT_OBJECT_0);
    EXPECT(dst[0] == 40 && dst[kPage64k] == 41);  // the GPU-only pages, found through the pool's CPU alias
    EXPECT(cpu[0] == 40);                         // a CPU twin with no mapping of the title's follows the copy
    EXPECT(cpu[kPage64k] == 77);                  // the title's own page is kept

    // The unmap: the GPU-only pages and the mirrored twin are unmapped, the title's page stays.
    const PageRange hole[] = {{2, 0, n}};
    const PageBatch unmap_batch{uintptr_t(dst), 1, hole};
    EXPECT(SUCCEEDED(call(queue.p, 1, &unmap_batch, uintptr_t(pool), n, 0)));
    EXPECT(SUCCEEDED(real_queue->Signal(gate.p, 3)));
    remapped = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    EXPECT(remapped && SUCCEEDED(gate->SetEventOnCompletion(3, remapped)));
    EXPECT(WaitForSingleObject(remapped, 10000) == WAIT_OBJECT_0);
    CloseHandle(remapped);
    EXPECT(state_is(dst, MEM_RESERVE) && state_is(dst + kPage64k, MEM_RESERVE) && state_is(cpu, MEM_RESERVE));
    EXPECT(cpu[kPage64k] == 77);

    EXPECT(map(cpu + kPage64k, 1, nullptr) && map(pool_cpu, n, nullptr));
    EXPECT(free_pages(n + 1, pages));
    std::puts("queue slot 31 default route: a source through its CPU alias, a deferred copy and an unmap keep the "
              "title's own CPU twin page");
    return 0;
}

// What a GPU copy of `buffer` reads now (a submission: the shadows are refreshed first).
int read_back(ID3D12Device* real, ID3D12Resource* buffer, UINT64 size, std::vector<std::uint8_t>& out) {
    D3D12_RESOURCE_DESC desc = buffer->GetDesc();
    D3D12_HEAP_PROPERTIES readback_heap{D3D12_HEAP_TYPE_READBACK};
    Ref<ID3D12Resource> readback;
    EXPECT(SUCCEEDED(real->CreateCommittedResource(&readback_heap, D3D12_HEAP_FLAG_NONE, &desc,
                                                   D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&readback.p))));
    D3D12_COMMAND_QUEUE_DESC queue_desc{D3D12_COMMAND_LIST_TYPE_DIRECT};
    Ref<ID3D12CommandQueue> queue;
    Ref<ID3D12CommandAllocator> allocator;
    Ref<ID3D12GraphicsCommandList> list;
    Ref<ID3D12Fence> fence;
    EXPECT(SUCCEEDED(real->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue.p))));
    EXPECT(SUCCEEDED(real->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p))));
    EXPECT(SUCCEEDED(real->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, nullptr,
                                             IID_PPV_ARGS(&list.p))));
    list->CopyBufferRegion(readback.p, 0, buffer, 0, size);
    EXPECT(SUCCEEDED(list->Close()));
    ID3D12CommandList* lists[] = {list.p};
    queue->ExecuteCommandLists(1, lists);
    EXPECT(SUCCEEDED(real->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))));
    EXPECT(SUCCEEDED(queue->Signal(fence.p, 1)));
    HANDLE done = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    EXPECT(SUCCEEDED(fence->SetEventOnCompletion(1, done)));
    EXPECT(WaitForSingleObject(done, 10000) == WAIT_OBJECT_0);
    CloseHandle(done);
    void* mapped = nullptr;
    EXPECT(SUCCEEDED(readback->Map(0, nullptr, &mapped)));
    out.assign(static_cast<const std::uint8_t*>(mapped), static_cast<const std::uint8_t*>(mapped) + size);
    readback->Unmap(0, nullptr);
    return 0;
}

DWORD protection_of(const void* address) {
    MEMORY_BASIC_INFORMATION info{};
    return VirtualQuery(address, &info, sizeof(info)) && info.State == MEM_COMMIT ? info.Protect : 0;
}

// A write from a thread of its own (the title's, or a DirectStorage queue's), waited for.
template <class F>
bool on_thread(F f) {
    std::thread t(f);
    t.join();
    return true;
}

// The guards over a stream pool's physical pages, mapped at the CPU view and at its GPU-only twin (protect 0x801, a
// window slot 31 maps by itself), with a slot-133 buffer (a shadow: the twin's pages are mapped independently) over
// the twin. Checked by what a GPU copy reads after each step:
//   1. after a submission the twin's pages are read-only (guarded);
//   2. a pin taken with an OVERLAPPED that does not say pending yet (ReadFile stores STATUS_PENDING once it starts)
//      survives a submission: the pinned page stays writable, as the kernel's write into it needs;
//   3. a host thread's copy into a guarded page (kernel_writes::write, as a DirectStorage queue publishes) lands, and
//      so does a DirectStorage page-list write through the section (rt::xmem::transfer_physical) into another;
//   4. the pages are mapped at the CPU view too: a title thread's write through the CPU view (no fault) and one through
//      the guarded twin (a fault its shadow answers) both reach the GPU;
//   5. a CPU-view page left read-only with no guard to reopen it: a title thread's write there faults, xmem's
//      unguarder makes it writable, and the write lands; a kernel write (kernel_writes::read_file) into another
//      such page succeeds too. Both reach the GPU through the twin.
int paired_views(ID3D12Device* real) {
    namespace kw = leonida::kernel_writes;
    auto reserve = reinterpret_cast<XMemVirtualAllocFn>(rt::shims::find("xmem.dll", "XMemVirtualAlloc"));
    auto allocate = reinterpret_cast<XMemAllocatePhysicalPagesFn>(rt::shims::find("xmem.dll", "XMemAllocatePhysicalPages"));
    auto map = reinterpret_cast<XMemMapPhysicalPagesFn>(rt::shims::find("xmem.dll", "XMemMapPhysicalPages"));
    auto free_pages = reinterpret_cast<XMemFreePhysicalPagesFn>(rt::shims::find("xmem.dll", "XMemFreePhysicalPages"));
    constexpr unsigned n = 4;
    constexpr UINT64 size = n * kPage64k;
    auto* cpu = static_cast<std::uint8_t*>(reserve(nullptr, size, MEM_RESERVE | kXmem64kPages, 0, PAGE_READWRITE));
    auto* gpu = static_cast<std::uint8_t*>(reserve(nullptr, size, MEM_RESERVE | kXmem64kPages, 0, 0x801));
    EXPECT(cpu && gpu && leonida::title_gpu_views::other_view(uintptr_t(gpu)) == uintptr_t(cpu));
    // The twin is the kind slot 31 maps by itself (xmem pairs it as the legacy kind, read at the CPU view).
    EXPECT(!leonida::title_gpu_views::independently_mapped(uintptr_t(gpu), size));
    EXPECT(leonida::title_gpu_views::set_independent_pages(uintptr_t(gpu)));
    EXPECT(leonida::title_gpu_views::independently_mapped(uintptr_t(gpu), size));
    std::uint64_t pages[n]{}, count = n;
    EXPECT(allocate(0x280, &count, pages) && count == n && map(gpu, n, pages) == gpu);
    std::vector<std::uint8_t> want(size), got;
    for (UINT64 i = 0; i < size; ++i) want[i] = gpu[i] = static_cast<std::uint8_t>(i * 37u + 11u);
    kw::set_unguarder(&rt::xmem::reopen_unowned);
    int failures = 0;
    {
        D3D12_RESOURCE_DESC desc{};
        desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        desc.Width = size;
        desc.Height = desc.DepthOrArraySize = desc.MipLevels = 1;
        desc.SampleDesc.Count = 1;
        desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        Ref<ID3D12Resource> buffer;
        EXPECT(SUCCEEDED(rt::d3d12::create_buffer_at_host_address(real, gpu, desc, 0xFFFFFFFFu,
                                                                  IID_PPV_ARGS(&buffer.p))));
        EXPECT(read_back(real, buffer.p, size, got) == 0 && got == want);
        // 1.
        const DWORD guarded = protection_of(gpu + 0x8000);
        EXPECT((guarded & 0xff) == PAGE_READONLY);

        // 2. The pin of a read whose call has started but whose OVERLAPPED is still all zeros.
        OVERLAPPED starting{};
        const std::uint64_t pin = kw::begin_write(gpu + 0x10000, 0x1000, &starting);
        EXPECT(pin && (protection_of(gpu + 0x10000) & 0xff) == PAGE_READWRITE);
        EXPECT(read_back(real, buffer.p, size, got) == 0 && got == want);
        EXPECT(kw::state().count.load() == 1 && (protection_of(gpu + 0x10000) & 0xff) == PAGE_READWRITE);
        kw::end_write(pin, false);
        EXPECT(kw::state().count.load() == 0);

        // 3. A host thread's copy into a guarded page.
        EXPECT(read_back(real, buffer.p, size, got) == 0 && (protection_of(gpu + 0x20000) & 0xff) == PAGE_READONLY);
        std::vector<std::uint8_t> source(0x2800);
        for (std::size_t i = 0; i < source.size(); ++i) source[i] = static_cast<std::uint8_t>(i * 5u + 200u);
        bool wrote = false;
        on_thread([&] { wrote = kw::write(gpu + 0x20400, source.data(), source.size()); });
        EXPECT(wrote);
        std::memcpy(want.data() + 0x20400, source.data(), source.size());
        EXPECT(read_back(real, buffer.p, size, got) == 0 && got == want);
        // And a DirectStorage page-list write (through the section) into a guarded page: pinned, so it is copied.
        EXPECT((protection_of(gpu + 0x30000) & 0xff) == PAGE_READONLY);
        std::vector<std::uint8_t> listed(0x600, 0x6b);
        EXPECT(rt::xmem::transfer_physical(std::span<const std::uint64_t>(&pages[3], 1), 0x800, listed, true));
        std::fill(want.begin() + 0x30800, want.begin() + 0x30e00, std::uint8_t(0x6b));
        EXPECT(read_back(real, buffer.p, size, got) == 0 && got == want);

        // 4. Paired: the same pages at the CPU view too.
        EXPECT(map(cpu, n, pages) == cpu && cpu[0x20400] == source[0]);
        EXPECT(read_back(real, buffer.p, size, got) == 0 && got == want);
        on_thread([&] { for (int i = 0; i < 256; ++i) want[0x30000 + i] = cpu[0x30000 + i] = std::uint8_t(0xa0 + i); });
        EXPECT(read_back(real, buffer.p, size, got) == 0 && got == want);
        const bool twin_guarded = (protection_of(gpu + 0x8000) & 0xff) == PAGE_READONLY;
        EXPECT(twin_guarded);  // still the shadow's: its write below faults
        on_thread([&] { for (int i = 0; i < 64; ++i) want[0x8010 + i] = gpu[0x8010 + i] = std::uint8_t(0x30 + i); });
        EXPECT(read_back(real, buffer.p, size, got) == 0 && got == want);

        // 5. CPU-view pages read-only with no guard to answer for them.
        DWORD old = 0;
        EXPECT(VirtualProtect(cpu + 0x11000, 0x1000, PAGE_READONLY, &old));
        on_thread([&] { for (int i = 0; i < 64; ++i) want[0x11100 + i] = cpu[0x11100 + i] = std::uint8_t(0x70 + i); });
        EXPECT((protection_of(cpu + 0x11000) & 0xff) == PAGE_READWRITE);
        EXPECT(read_back(real, buffer.p, size, got) == 0 && got == want);
        wchar_t dir[MAX_PATH]{}, path[MAX_PATH]{};
        EXPECT(GetTempPathW(MAX_PATH, dir) && GetTempFileNameW(dir, L"xpv", 0, path));
        HANDLE file = CreateFileW(path, GENERIC_READ | GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS,
                                  FILE_ATTRIBUTE_TEMPORARY | FILE_FLAG_DELETE_ON_CLOSE, nullptr);
        DWORD moved = 0;
        EXPECT(file != INVALID_HANDLE_VALUE && WriteFile(file, source.data(), 0x1000, &moved, nullptr) && moved == 0x1000);
        EXPECT(VirtualProtect(cpu + 0x32000, 0x1000, PAGE_READONLY, &old));
        OVERLAPPED at{};
        bool read = false;
        on_thread([&] { read = kw::read_file(file, cpu + 0x32000, 0x1000, &moved, &at) && moved == 0x1000; });
        CloseHandle(file);
        EXPECT(read);
        std::memcpy(want.data() + 0x32000, source.data(), 0x1000);
        EXPECT(read_back(real, buffer.p, size, got) == 0 && got == want);
        std::printf("%s: paired views: guarded twin 0x%lx (0x%lx with the CPU view mapped), a pin whose OVERLAPPED "
                    "said nothing yet kept its page writable across a submission, host and title writes through "
                    "either view and into unowned read-only pages reached the GPU\n",
                    who, guarded, twin_guarded ? guarded : protection_of(gpu + 0x8000));
    }
    kw::set_unguarder(nullptr);
    EXPECT(map(cpu, n, nullptr) && map(gpu, n, nullptr) && free_pages(n, pages));
    return failures;
}

int exposure_map_ranges(void* device) {
    auto* pages = static_cast<char*>(VirtualAlloc(nullptr, 2 * kPage64k, MEM_RESERVE, PAGE_READWRITE));
    EXPECT(pages && VirtualAlloc(pages, kPage64k, MEM_COMMIT, PAGE_READWRITE));
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = desc.Height = desc.DepthOrArraySize = desc.MipLevels = 1;
    desc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
    desc.SampleDesc.Count = 1;
    desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    {
        Ref<ID3D12Resource> image;
        EXPECT(SUCCEEDED(place(device, pages, desc, &image.p)));
        for (SIZE_T offset : {SIZE_T(0), kPage64k}) {
            const D3D12_RANGE read{offset, offset + 256}, written{};
            void* raw = nullptr;
            EXPECT(SUCCEEDED(image->Map(0, &read, &raw)) && raw == pages);
            image->Unmap(0, &written);
        }
        EXPECT(state_is(pages + kPage64k, MEM_RESERVE));  // logging must not fault in the page
    }
    EXPECT(VirtualFree(pages, 0, MEM_RELEASE));
    return 0;
}

int run(ID3D12Device* real, void* device, bool eager) {
    const auto virtual_alloc = reinterpret_cast<XMemVirtualAllocFn>(rt::shims::find("xmem.dll", "XMemVirtualAlloc"));
    const auto allocate =
        reinterpret_cast<XMemAllocatePhysicalPagesFn>(rt::shims::find("xmem.dll", "XMemAllocatePhysicalPages"));
    const auto free_pages =
        reinterpret_cast<XMemFreePhysicalPagesFn>(rt::shims::find("xmem.dll", "XMemFreePhysicalPages"));
    const auto map = reinterpret_cast<XMemMapPhysicalPagesFn>(rt::shims::find("xmem.dll", "XMemMapPhysicalPages"));
    EXPECT(virtual_alloc && allocate && free_pages && map);

    constexpr UINT64 size = kPages * kPage64k;
    auto* const window =
        static_cast<std::uint8_t*>(virtual_alloc(nullptr, size, MEM_RESERVE | kXmem64kPages, 0x23, PAGE_READWRITE));
    EXPECT(window != nullptr);
    std::uint64_t pages[kPages]{};
    std::uint64_t count = kPages;
    EXPECT(allocate(0x280, &count, pages) == TRUE && count == kPages);
    EXPECT(map(window, kPages, pages) == window);
    const DWORD initial_state = expected_xmem_section_state();
    EXPECT(state_is(window, initial_state));

    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = size;
    desc.Height = desc.DepthOrArraySize = desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout = static_cast<D3D12_TEXTURE_LAYOUT>(0x1);  // the Xbox row-major word
    const auto window_va = reinterpret_cast<D3D12_GPU_VIRTUAL_ADDRESS>(window);

    // Without the hook the reserved pages are refused and the fallback answers. Its GPU VA is the title's address
    // either way: the imported buffer is bound to the window (create_host_buffer: va_ = start), and the shadow keeps
    // that VA too so the descriptors and views the title builds from it still find the buffer (89261c7ed,
    // d3d12_existing_heap_test.cpp). The reserved state is what shows the import refused these pages.
    rt::d3d12_xs::set_placed_at_va_pages_hook(nullptr);
    {
        Ref<ID3D12Resource> fallback;
        EXPECT(place(device, window, desc, &fallback.p) == S_OK && fallback.p);
        if (eager) {
            EXPECT(fallback->GetGPUVirtualAddress() == window_va && state_is(window, MEM_COMMIT));
            for (UINT64 i = 0; i < size; ++i) window[i] = static_cast<std::uint8_t>(i * 17u + 29u);
            if (gpu_reads_cpu_writes(real, fallback.p, window, size)) return 1;
        } else {
            EXPECT(fallback->GetGPUVirtualAddress() == window_va);
            EXPECT(state_is(window, initial_state));
        }
    }
    std::printf("%s: physical pages without the hook: %s\n", who, eager ? "already committed and GPU-visible" : "still reserved");

    // With it the range is committed first and imported.
    rt::d3d12_xs::set_placed_at_va_pages_hook(&rt::xmem::commit_physical_range);
    {
        Ref<ID3D12Resource> buffer;
        EXPECT(place(device, window, desc, &buffer.p) == S_OK && buffer.p);
        EXPECT(state_is(window, MEM_COMMIT) && state_is(window + size - 1, MEM_COMMIT));
        EXPECT(buffer->GetGPUVirtualAddress() == window_va);
        for (UINT64 i = 0; i < size; ++i) window[i] = static_cast<std::uint8_t>(i * 131u + 7u);
        if (gpu_reads_cpu_writes(real, buffer.p, window, size)) return 1;
    }
    rt::d3d12_xs::set_placed_at_va_pages_hook(nullptr);
    std::printf("%s: with commit_physical_range: %llu bytes committed, GPU VA is the window, a GPU copy reads the "
                "CPU's writes\n", who, static_cast<unsigned long long>(size));

    EXPECT(map(window, kPages, nullptr) == window);
    EXPECT(free_pages(count, pages) == TRUE);
    return 0;
}

int linear_texture_mapping(ID3D12Device* dev, ID3D12Device* xbox, bool private_backing) {
    const auto reserve = reinterpret_cast<XMemVirtualAllocFn>(rt::shims::find("xmem.dll","XMemVirtualAlloc"));
    const auto alloc = reinterpret_cast<XMemAllocatePhysicalPagesFn>(rt::shims::find("xmem.dll","XMemAllocatePhysicalPages"));
    const auto map = reinterpret_cast<XMemMapPhysicalPagesFn>(rt::shims::find("xmem.dll","XMemMapPhysicalPages"));
    const auto release = reinterpret_cast<XMemFreePhysicalPagesFn>(rt::shims::find("xmem.dll","XMemFreePhysicalPages"));
    EXPECT(reserve && alloc && map && release);
    std::uint64_t count=1,page{};
    void* window = private_backing ? VirtualAlloc(nullptr,kPage64k,MEM_RESERVE|MEM_COMMIT,PAGE_READWRITE)
                                   : reserve(nullptr,kPage64k,MEM_RESERVE|kXmem64kPages,0x23,PAGE_READWRITE);
    EXPECT(window);
    if (!private_backing) EXPECT(alloc(0x280,&count,&page) && map(window,1,&page)==window);
    rt::d3d12::set_device_outer(dev,xbox);
    rt::d3d12_xs::set_texture_page_provider(&rt::xmem::pin_mapped_data);
    constexpr UINT w=9,h=8;
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension=D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width=w;desc.Height=h;desc.MipLevels=desc.DepthOrArraySize=1;desc.SampleDesc.Count=1;
    desc.Format=DXGI_FORMAT_R32G32B32A32_FLOAT;desc.Flags=D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    desc.Layout=static_cast<D3D12_TEXTURE_LAYOUT>(0x100);
    Ref<ID3D12Resource> tex,rb;
    EXPECT(place(xbox,window,desc,&tex.p)==S_OK && tex->GetGPUVirtualAddress()==reinterpret_cast<UINT64>(window));
    constexpr UINT pitch=256;
    D3D12_RANGE empty{0,0};void* cpu=nullptr;
    EXPECT(tex->Map(0,&empty,&cpu)==S_OK && cpu && ((cpu!=window)==private_backing));
    std::array<float,4> pixel{};
    for(UINT y=0;y<h;++y) for(UINT x=0;x<w;++x) {
        for(UINT c=0;c<4;++c) pixel[c]=float(y*w*4+x*4+c)/512.0f;
        std::memcpy(static_cast<std::uint8_t*>(cpu)+y*pitch+x*16,pixel.data(),16);
    }
    D3D12_HEAP_PROPERTIES read_heap{D3D12_HEAP_TYPE_READBACK};
    const auto buf=d3d12test::buffer_desc(h*pitch);
    EXPECT(SUCCEEDED(dev->CreateCommittedResource(&read_heap,D3D12_HEAP_FLAG_NONE,&buf,D3D12_RESOURCE_STATE_COPY_DEST,nullptr,IID_PPV_ARGS(&rb.p))));
    D3D12_COMMAND_QUEUE_DESC q{};
    Ref<ID3D12CommandQueue> queue;Ref<ID3D12CommandAllocator> commands;Ref<ID3D12GraphicsCommandList> list;Ref<ID3D12Fence> fence;
    EXPECT(SUCCEEDED(dev->CreateCommandQueue(&q,IID_PPV_ARGS(&queue.p))));
    EXPECT(SUCCEEDED(dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,IID_PPV_ARGS(&commands.p))));
    EXPECT(SUCCEEDED(dev->CreateCommandList(0,D3D12_COMMAND_LIST_TYPE_DIRECT,commands.p,nullptr,IID_PPV_ARGS(&list.p))));
    EXPECT(SUCCEEDED(dev->CreateFence(0,D3D12_FENCE_FLAG_NONE,IID_PPV_ARGS(&fence.p))));
    const auto submit=[&](UINT64 value) {
        if(FAILED(list->Close())) return false;
        ID3D12CommandList* batch[]{list.p};queue->ExecuteCommandLists(1,batch);
        if(FAILED(queue->Signal(fence.p,value))) return false;
        const auto end=GetTickCount64()+6000;
        while(fence->GetCompletedValue()<value && GetTickCount64()<end) Sleep(1);
        return fence->GetCompletedValue()==value;
    };
    auto barrier=d3d12test::transition(tex.p,D3D12_RESOURCE_STATE_COMMON,D3D12_RESOURCE_STATE_COPY_SOURCE);
    list->ResourceBarrier(1,&barrier);
    D3D12_TEXTURE_COPY_LOCATION from{},to{};from.pResource=tex.p;to.pResource=rb.p;
    to.Type=D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;to.PlacedFootprint={0,{desc.Format,w,h,1,pitch}};
    list->CopyTextureRegion(&to,0,0,0,&from,nullptr);
    EXPECT(submit(1));
    void* bytes=nullptr;EXPECT(SUCCEEDED(rb->Map(0,nullptr,&bytes)));
    for(UINT y=0;y<h;++y) EXPECT(std::memcmp(static_cast<const std::uint8_t*>(bytes)+y*pitch,
                                             static_cast<const std::uint8_t*>(cpu)+y*pitch,w*16)==0);
    rb->Unmap(0,nullptr);
    tex->Unmap(0,&empty);
    Ref<ID3D12DescriptorHeap> rt;
    D3D12_DESCRIPTOR_HEAP_DESC hd{D3D12_DESCRIPTOR_HEAP_TYPE_RTV,1};
    EXPECT(SUCCEEDED(dev->CreateDescriptorHeap(&hd,IID_PPV_ARGS(&rt.p))));
    const auto handle=rt->GetCPUDescriptorHandleForHeapStart();
    dev->CreateRenderTargetView(tex.p,nullptr,handle);
    EXPECT(SUCCEEDED(commands->Reset()) && SUCCEEDED(list->Reset(commands.p,nullptr)));
    barrier=d3d12test::transition(tex.p,D3D12_RESOURCE_STATE_COPY_SOURCE,D3D12_RESOURCE_STATE_RENDER_TARGET);
    list->ResourceBarrier(1,&barrier);
    const float rgba[]{.25f,.5f,.75f,1.0f};
    list->ClearRenderTargetView(handle,rgba,0,nullptr);
    EXPECT(submit(2));
    D3D12_RANGE read{0,h*pitch};cpu=nullptr;
    EXPECT(tex->Map(0,&read,&cpu)==S_OK && cpu && ((cpu!=window)==private_backing));
    for(UINT y=0;y<h;++y) for(UINT x=0;x<w;++x)
        EXPECT(std::memcmp(static_cast<const std::uint8_t*>(cpu)+y*pitch+x*16,rgba,16)==0);
    tex->Unmap(0,&empty);
    if (!private_backing) {
        EXPECT(map(window,1,nullptr)==window && release(1,&page));
        cpu=reinterpret_cast<void*>(1);
        EXPECT(FAILED(tex->Map(0,&empty,&cpu)) && !cpu);
    } else {
        const auto* original=static_cast<const std::uint8_t*>(window);
        for(UINT i=0;i<kPage64k;++i) EXPECT(original[i]==0);
        EXPECT(VirtualFree(window,0,MEM_RELEASE));
    }
    std::printf("linear placed texture: %u GPU texels match writes during Map, %u mapped texels match the GPU clear, released placement refused\n",w*h,w*h);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    const bool eager = argc == 2 && std::strcmp(argv[1], "--eager-mapping") == 0;
    vktest::Device vk;
    const auto status = vk.initialize();
    if (status == vktest::DeviceStatus::unavailable) return 77;
    if (status != vktest::DeviceStatus::ready) return 1;
    if (!vk.host_pointer_alignment) {
        std::printf("%s: skipped, no VK_EXT_external_memory_host\n", who);
        return 77;
    }
    int failures = 0;
    {
        rt::d3d12::VulkanDeviceInfo info{};
        info.get_instance_proc = vk.instance_proc;
        info.instance = vk.instance;
        info.physical = vk.physical;
        info.device = vk.device;
        info.queue = vk.queue;
        info.queue_family = vk.family;
        info.host_pointer_alignment = vk.host_pointer_alignment;
        Ref<ID3D12Device> real;
        if (FAILED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&real.p)))) return 1;
        real->AddRef();  // the proxy takes one reference
        Ref<ID3D12Device> device;
        if (FAILED(rt::d3d12_xs::create_device_proxy(real.p, __uuidof(ID3D12Device),
                                                     reinterpret_cast<void**>(&device.p)))) {
            real->Release();
            return 1;
        }
        if (argc==2 && (std::strcmp(argv[1],"--linear-map")==0 || std::strcmp(argv[1],"--linear-staging")==0))
            failures+=linear_texture_mapping(real.p,device.p,std::strcmp(argv[1],"--linear-staging")==0);
        else {
            failures += run(real.p, device.p, eager);
            failures += exposure_map_ranges(device.p);
            failures += queue_remap(real.p, device.p);
            failures += queue_remap_default_route(real.p, device.p);
            failures += paired_views(real.p);
        }
    }
    if (vk.validation_errors()) {
        std::fprintf(stderr, "%u validation errors\n", vk.validation_errors());
        return 1;
    }
    return failures ? 1 : 0;
}
