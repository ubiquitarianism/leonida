#include "vk_test_device.h"
#include "d3d12_test_shaders.h"
#include "../shims/d3d12_com.h"
#include "../shims/d3d12_xs_device.h"
#include "../src/gpu_label_memory.h"
#include <atomic>
#include <cstdio>
#include <cstdlib>

#define CHECK(x)                                                                                                       \
    do {                                                                                                               \
        if (!(x)) {                                                                                                    \
            std::fprintf(stderr, "line %d: %s\n", __LINE__, #x);                                                       \
            return 1;                                                                                                  \
        }                                                                                                              \
    } while (0)
using namespace d3d12test;
namespace xs = rt::d3d12_xs;
struct Data {
    void *addr = VirtualAlloc(nullptr, 4096, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    void track() {
        auto a = reinterpret_cast<std::uintptr_t>(addr);
        leonida::label::memory().track(a, 4096, a, true);
    }
    ~Data() {
        if (addr) {
            leonida::label::memory().release_heap(reinterpret_cast<std::uintptr_t>(addr));
            VirtualFree(addr, 0, MEM_RELEASE);
        }
    }
};
using Create = HRESULT(WINAPI *)(void *, const void *, ID3D12CommandAllocator *, ID3D12PipelineState *, const IID *,
                                 void **);
using Label = void(WINAPI *)(void *, std::uintptr_t, std::uint64_t, std::uint32_t, std::uint32_t);
int main() {
    _putenv_s("LEONIDA_GPU_LABELS", "1");
    // First the label path alone: these lists are not tracked by the list values, which own slots 71 to 74 of the
    // lists they track (the last section). The real list is submitted straight to the queue below.
    xs::set_list_values_enabled(false);
    xs::reset_list_value_stats();
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
    Ref<ID3D12CommandQueue> queue;
    D3D12_COMMAND_QUEUE_DESC queue_desc{};
    CHECK(SUCCEEDED(inner->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue.p))));
    Ref<ID3D12CommandAllocator> allocator;
    CHECK(SUCCEEDED(inner->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p))));
    std::uint64_t desc[4]{};
    Ref<ID3D12GraphicsCommandList> list;
    CHECK(SUCCEEDED(reinterpret_cast<Create>(slot(device.p, 73))(device.p, desc, allocator.p, nullptr,
                                                                 &__uuidof(ID3D12GraphicsCommandList),
                                                                 reinterpret_cast<void **>(&list.p))));
    CHECK(xs::is_list_proxy(list.p));
    auto *real = xs::list_proxy_real_list(list.p);
    CHECK(real);
    Data live, stale;
    CHECK(live.addr && stale.addr);
    live.track();
    stale.track();
    const auto base = reinterpret_cast<std::uintptr_t>(live.addr), old = reinterpret_cast<std::uintptr_t>(stale.addr);
    auto *values = static_cast<volatile LONG64 *>(live.addr);
    auto *retired = static_cast<volatile LONG *>(stale.addr);
    const auto write32 = reinterpret_cast<Label>(slot(list.p, 71)), write64 = reinterpret_cast<Label>(slot(list.p, 72));
    write32(list.p, base, 23, 0, 0);
    write64(list.p, base + 8, 0x100000002ull, 0, 0);
    write64(list.p, base + 16, 0, 0, 1);
    write64(list.p, base + 24, 0, 0, 1);
    write32(list.p, old, 99, 0, 0);
    CHECK(SUCCEEDED(list->Close()));
    CHECK(values[0] == 0 && values[1] == 0 && values[2] == 0 && values[3] == 0);
    leonida::label::memory().release_heap(old);
    stale.track();
    *retired = 7;
    Ref<ID3D12Fence> fence;
    CHECK(SUCCEEDED(inner->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))));
    for (unsigned pass = 1; pass <= 2; ++pass) {
        InterlockedExchange64(values, 0);
        InterlockedExchange64(values + 1, 0);
        InterlockedExchange64(values + 2, 0);
        InterlockedExchange64(values + 3, 0);
        ID3D12CommandList *submitted[1] = {real};
        queue->ExecuteCommandLists(1, submitted);
        CHECK(SUCCEEDED(queue->Signal(fence.p, pass)));
        const auto deadline = GetTickCount64() + 4000;
        while (GetTickCount64() < deadline &&
               (fence->GetCompletedValue() < pass || !InterlockedCompareExchange64(values + 3, 0, 0)))
            Sleep(1);
        CHECK(fence->GetCompletedValue() >= pass);
        CHECK(InterlockedCompareExchange64(values, 0, 0) == 23 &&
              InterlockedCompareExchange64(values + 1, 0, 0) == 0x100000002ll);
        const auto t0 = InterlockedCompareExchange64(values + 2, 0, 0),
                   t1 = InterlockedCompareExchange64(values + 3, 0, 0);
        CHECK(t0 > 0 && t1 >= t0);
        CHECK(*retired == 7);
    }
    CHECK(xs::list_value_stats().labels_handed == 5 && xs::list_value_stats().writes_recorded == 0);
    std::puts("GPU completion values, two real timestamp samples, resubmission and stale allocation guard passed");

    // With LEONIDA_GPU_LABELS=1 and the list values on (their default), a tracked list's slot 71/72 writes have one
    // owner: the list values land each exactly once, and nothing goes to the label path, which would store the same
    // address again from its own completion and clock.
    xs::set_list_values_enabled(true);
    xs::reset_list_value_stats();
    {
        Ref<ID3D12CommandQueue> proxy_queue;
        CHECK(SUCCEEDED(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&proxy_queue.p))));
        Ref<ID3D12GraphicsCommandList> tracked;
        CHECK(SUCCEEDED(reinterpret_cast<Create>(slot(device.p, 73))(device.p, desc, allocator.p, nullptr,
                                                                     &__uuidof(ID3D12GraphicsCommandList),
                                                                     reinterpret_cast<void **>(&tracked.p))));
        CHECK(xs::is_list_proxy(tracked.p));
        const auto put32 = reinterpret_cast<Label>(slot(tracked.p, 71)),
                   put64 = reinterpret_cast<Label>(slot(tracked.p, 72));
        InterlockedExchange64(values + 4, 0);
        InterlockedExchange64(values + 5, 0);
        put32(tracked.p, base + 32, 41, 0, 0);
        put64(tracked.p, base + 40, 0x300000004ull, 0, 0);
        CHECK(SUCCEEDED(tracked->Close()));
        auto stats = xs::list_value_stats();
        CHECK(stats.labels_handed == 0 && stats.writes_recorded == 2);
        ID3D12CommandList *submitted[1] = {tracked.p};
        proxy_queue->ExecuteCommandLists(1, submitted);
        Ref<ID3D12Fence> done;
        CHECK(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&done.p))));
        CHECK(SUCCEEDED(proxy_queue->Signal(done.p, 1)) && SUCCEEDED(done->SetEventOnCompletion(1, nullptr)));
        CHECK(xs::list_values_idle(4000));
        stats = xs::list_value_stats();
        CHECK(stats.labels_handed == 0 && stats.writes_landed == 2 && stats.writes_failed == 0);
        CHECK(InterlockedCompareExchange64(values + 4, 0, 0) == 41 &&
              InterlockedCompareExchange64(values + 5, 0, 0) == 0x300000004ll);
    }
    CHECK(gpu.validation_errors() == 0);
    std::puts("with GPU labels and list values both on, slot 71/72 writes landed once, through the list values");
    return 0;
}
