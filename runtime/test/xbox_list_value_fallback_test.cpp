// The list values' write fallbacks (runtime/src/value_write_sink.h), on a Vulkan device and the queue and list proxies,
// with xmem linked in as in gta6rt. No game data.
// - A write whose address is a GPU address arena offset (a default-heap buffer) lands in the arena, read back by a GPU
//   copy; one in the arena's reserved start is dropped, never stored at that number in host memory.
// - A write to a page of xmem's that the title made PAGE_NOACCESS (its VirtualProtect, which xmem's fault handler and
//   kernel_writes' reopen leave as the title set it), where the host store faults, lands through xmem's own mapping
//   of the physical page.
// - A write to ordinary title memory lands as before, through neither fallback.
// LEONIDA_GPU_LABELS=1 throughout: the arena and xmem's pages are label targets too, and the lists here are tracked by
// the list values, so no write may be handed to the label path (one writer per list).
#include "vk_test_device.h"
#include "d3d12_test_shaders.h"
#include "../shims/d3d12_com.h"
#include "../shims/d3d12_xs_device.h"
#include "../shims/shims.h"
#include "../shims/xmem.h"
#include "../src/title_page_watch.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define CHECK(x)                                                                                                       \
    do {                                                                                                               \
        if (!(x)) {                                                                                                    \
            std::fprintf(stderr, "line %d: %s\n", __LINE__, #x);                                                       \
            return 1;                                                                                                  \
        }                                                                                                              \
    } while (0)

using namespace d3d12test;
namespace xs = rt::d3d12_xs;

namespace {
using Create = HRESULT(WINAPI *)(void *, const void *, ID3D12CommandAllocator *, ID3D12PipelineState *, const IID *,
                                 void **);
using Put = void(WINAPI *)(void *, std::uintptr_t, std::uint64_t, std::uint32_t, std::uint32_t);
using XMemVirtualAllocFn = void *(WINAPI *)(void *, SIZE_T, DWORD, ULONGLONG, DWORD);
using XMemAllocatePhysicalPagesFn = BOOL(WINAPI *)(std::uint64_t, std::uint64_t *, std::uint64_t *);
using XMemFreePhysicalPagesFn = BOOL(WINAPI *)(std::uint64_t, const std::uint64_t *);
using XMemMapPhysicalPagesFn = void *(WINAPI *)(void *, std::uint64_t, const std::uint64_t *);
constexpr DWORD kXmem64kPages = 0x20400000;
constexpr SIZE_T kPage64k = 0x10000;

struct Writes {
    std::uintptr_t address;
    std::uint64_t value;
    bool wide;
};

// One tracked list with the writes, submitted through the queue proxy; true once every write has landed or failed.
bool submit(ID3D12Device *device, ID3D12CommandQueue *queue, ID3D12CommandAllocator *allocator,
            std::initializer_list<Writes> writes) {
    std::uint64_t desc[4]{};
    Ref<ID3D12GraphicsCommandList> list;
    if (FAILED(reinterpret_cast<Create>(slot(device, 73))(device, desc, allocator, nullptr,
                                                          &__uuidof(ID3D12GraphicsCommandList),
                                                          reinterpret_cast<void **>(&list.p))) ||
        !xs::is_list_proxy(list.p))
        return false;
    for (const Writes &w : writes)
        reinterpret_cast<Put>(slot(list.p, w.wide ? 72 : 71))(list.p, w.address, w.value, 0, 0);
    if (FAILED(list->Close())) return false;
    ID3D12CommandList *submitted[1] = {list.p};
    queue->ExecuteCommandLists(1, submitted);
    Ref<ID3D12Fence> done;
    return SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&done.p))) &&
           SUCCEEDED(queue->Signal(done.p, 1)) && SUCCEEDED(done->SetEventOnCompletion(1, nullptr)) &&
           xs::list_values_idle(4000);
}

// The first `bytes` of a default-heap buffer, through a GPU copy into a readback buffer.
bool read_back(ID3D12Device *real, ID3D12Resource *buffer, void *out, UINT64 bytes) {
    D3D12_RESOURCE_DESC desc = buffer->GetDesc();
    D3D12_HEAP_PROPERTIES readback_heap{D3D12_HEAP_TYPE_READBACK};
    Ref<ID3D12Resource> readback;
    D3D12_COMMAND_QUEUE_DESC queue_desc{D3D12_COMMAND_LIST_TYPE_DIRECT};
    Ref<ID3D12CommandQueue> queue;
    Ref<ID3D12CommandAllocator> allocator;
    Ref<ID3D12GraphicsCommandList> list;
    Ref<ID3D12Fence> fence;
    if (FAILED(real->CreateCommittedResource(&readback_heap, D3D12_HEAP_FLAG_NONE, &desc,
                                             D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&readback.p))) ||
        FAILED(real->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue.p))) ||
        FAILED(real->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p))) ||
        FAILED(real->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, nullptr,
                                       IID_PPV_ARGS(&list.p))) ||
        FAILED(real->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))))
        return false;
    list->CopyBufferRegion(readback.p, 0, buffer, 0, bytes);
    if (FAILED(list->Close())) return false;
    ID3D12CommandList *lists[1] = {list.p};
    queue->ExecuteCommandLists(1, lists);
    if (FAILED(queue->Signal(fence.p, 1)) || FAILED(fence->SetEventOnCompletion(1, nullptr))) return false;
    void *mapped = nullptr;
    D3D12_RANGE range{0, static_cast<SIZE_T>(bytes)};
    if (FAILED(readback->Map(0, &range, &mapped)) || !mapped) return false;
    std::memcpy(out, mapped, static_cast<std::size_t>(bytes));
    readback->Unmap(0, nullptr);
    return true;
}
}  // namespace

int main() {
    _putenv_s("LEONIDA_GPU_LABELS", "1");
    xs::set_list_values_enabled(true);
    vktest::Device gpu;
    auto status = gpu.initialize();
    if (status == vktest::DeviceStatus::unavailable) return 77;
    CHECK(status == vktest::DeviceStatus::ready);
    rt::d3d12::VulkanDeviceInfo info{};
    info.get_instance_proc = gpu.instance_proc;
    info.instance = gpu.instance;
    info.physical = gpu.physical;
    info.device = gpu.device;
    info.queue = gpu.queue;
    info.queue_family = gpu.family;
    info.buffer_device_address = true;
    ID3D12Device *inner = nullptr;
    CHECK(SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&inner))));
    Ref<ID3D12Device> device;
    CHECK(
        SUCCEEDED(xs::create_device_proxy(inner, __uuidof(ID3D12Device), reinterpret_cast<void **>(&device.p), true)));
    rt::d3d12::set_device_outer(inner, device.p);
    D3D12_COMMAND_QUEUE_DESC queue_desc{};
    Ref<ID3D12CommandQueue> queue;
    CHECK(SUCCEEDED(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue.p))));
    Ref<ID3D12CommandAllocator> allocator;
    CHECK(SUCCEEDED(inner->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p))));

    // 1. Ordinary title memory: the host store, as before; neither fallback.
    auto *plain = static_cast<volatile LONG64 *>(VirtualAlloc(nullptr, 4096, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
    CHECK(plain);
    // The title's memory, as xmem registers it in gta6rt: an address below the arena's size is the title's there.
    leonida::title_va_aliases().add_title_memory(reinterpret_cast<std::uint64_t>(plain), 4096);
    xs::reset_list_value_stats();
    CHECK(submit(device.p, queue.p, allocator.p,
                 {{reinterpret_cast<std::uintptr_t>(plain), 11, false},
                  {reinterpret_cast<std::uintptr_t>(plain + 1), 0x500000006ull, true}}));
    auto stats = xs::list_value_stats();
    CHECK(stats.writes_landed == 2 && stats.writes_failed == 0);
    CHECK(stats.writes_arena == 0 && stats.writes_title_pages == 0 && stats.labels_handed == 0);
    CHECK(InterlockedCompareExchange64(plain, 0, 0) == 11 && InterlockedCompareExchange64(plain + 1, 0, 0) ==
                                                                 0x500000006ll);
    VirtualFree(const_cast<LONG64 *>(plain), 0, MEM_RELEASE);
    std::puts("a write to the title's own memory lands through the host store");

    // 2. An arena offset: a default-heap buffer's VA. The value lands in the arena, where the GPU reads it.
    D3D12_HEAP_PROPERTIES default_heap{D3D12_HEAP_TYPE_DEFAULT};
    D3D12_RESOURCE_DESC buffer_desc{};
    buffer_desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    buffer_desc.Width = kPage64k;
    buffer_desc.Height = 1;
    buffer_desc.DepthOrArraySize = 1;
    buffer_desc.MipLevels = 1;
    buffer_desc.SampleDesc.Count = 1;
    buffer_desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    Ref<ID3D12Resource> buffer;
    CHECK(SUCCEEDED(inner->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &buffer_desc,
                                                   D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&buffer.p))));
    const std::uint64_t va = buffer->GetGPUVirtualAddress();
    CHECK(va && va < (std::uint64_t{1} << 32));  // an arena offset, not a host address
    xs::reset_list_value_stats();
    CHECK(submit(device.p, queue.p, allocator.p, {{va, 41, false}, {va + 8, 0x300000004ull, true}}));
    stats = xs::list_value_stats();
    CHECK(stats.labels_handed == 0);
    CHECK(stats.writes_landed == 2 && stats.writes_arena == 2 && stats.writes_failed == 0);
    std::uint64_t words[2]{};
    CHECK(read_back(inner, buffer.p, words, sizeof(words)));
    CHECK(static_cast<std::uint32_t>(words[0]) == 41 && words[1] == 0x300000004ull);
    // The arena's reserved start (null region, zero page, quad indices) is the arena's, and shaders read it: a write
    // there is dropped, and never stored at that number in host memory.
    xs::reset_list_value_stats();
    CHECK(submit(device.p, queue.p, allocator.p, {{0x100, 99, false}}));
    stats = xs::list_value_stats();
    CHECK(stats.writes_landed == 0 && stats.writes_failed == 1 && stats.writes_arena == 0);
    std::puts("a write to an arena offset lands in the arena; one in its reserved start is dropped");

    // 3. A page of xmem's that the title made PAGE_NOACCESS through its VirtualProtect: the host store faults there
    // (xmem's fault handler and reopen_unowned leave a page the title protected), and the value lands through xmem's
    // own mapping of the page.
    auto reserve = reinterpret_cast<XMemVirtualAllocFn>(rt::shims::find("xmem.dll", "XMemVirtualAlloc"));
    auto allocate =
        reinterpret_cast<XMemAllocatePhysicalPagesFn>(rt::shims::find("xmem.dll", "XMemAllocatePhysicalPages"));
    auto map = reinterpret_cast<XMemMapPhysicalPagesFn>(rt::shims::find("xmem.dll", "XMemMapPhysicalPages"));
    auto free_pages = reinterpret_cast<XMemFreePhysicalPagesFn>(rt::shims::find("xmem.dll", "XMemFreePhysicalPages"));
    using ProtectFn = BOOL(WINAPI *)(void *, SIZE_T, DWORD, PDWORD);
    auto title_protect = reinterpret_cast<ProtectFn>(rt::shims::find("kernel32.dll", "VirtualProtect"));
    CHECK(reserve && allocate && map && free_pages && title_protect);
    auto *window =
        static_cast<std::uint8_t *>(reserve(nullptr, kPage64k, MEM_RESERVE | kXmem64kPages, 0, PAGE_READWRITE));
    CHECK(window);
    std::uint64_t page = 0, count = 1;
    CHECK(allocate(0x280, &count, &page) && count == 1);
    CHECK(map(window, 1, &page) == window);
    window[0] = 1;  // committed, as a page the title used
    DWORD old = 0;
    CHECK(title_protect(window, kPage64k, PAGE_NOACCESS, &old));
    // What the list values did before the fallback: the host store into the title's view, which refuses it.
    CHECK(!leonida::kernel_writes::store(window + 0x80, 1, 4));
    xs::reset_list_value_stats();
    CHECK(submit(device.p, queue.p, allocator.p,
                 {{reinterpret_cast<std::uintptr_t>(window) + 0x40, 0x5151, false},
                  {reinterpret_cast<std::uintptr_t>(window) + 0x48, 0x700000008ull, true}}));
    stats = xs::list_value_stats();
    CHECK(stats.labels_handed == 0);
    CHECK(stats.writes_landed == 2 && stats.writes_title_pages == 2 && stats.writes_failed == 0);
    const std::uintptr_t host = rt::xmem::physical_page_scratch(page);
    CHECK(host);
    std::uint32_t low = 0;
    std::uint64_t wide = 0;
    std::memcpy(&low, reinterpret_cast<const void *>(host + 0x40), sizeof(low));
    std::memcpy(&wide, reinterpret_cast<const void *>(host + 0x48), sizeof(wide));
    CHECK(low == 0x5151 && wide == 0x700000008ull);
    map(window, 1, nullptr);
    CHECK(free_pages(1, &page));
    std::puts("a write to a PAGE_NOACCESS page of xmem's lands through xmem's own mapping");

    CHECK(gpu.validation_errors() == 0);
    return 0;
}
