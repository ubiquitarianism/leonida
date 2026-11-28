// The spare real lists a list-value cut leaves behind (d3d12_xs_device.cpp restart_recording), on a Vulkan device
// and the queue and list proxies. No game data.
// A recording cut three times leaves three spare real lists when the title resets its list; later recordings cut at
// most once reuse only the last of them, and the other two stay closed and idle until the list is released. A title
// window the layer retires is freed only once every command list whose recording began before the retirement is gone
// (src/command_list_epochs.h), and a list holds the root signatures and pipeline states its recording set until its
// next Reset. The idle spares must therefore drop their recordings when they become spares: they can never be
// executed again before a cut resets them. Without that, the oldest live recording stays the first one's, and the
// root signature it set stays referenced by the two idle spares.
#include "vk_test_device.h"
#include "d3d12_test_shaders.h"
#include "../shims/d3d12_com.h"
#include "../shims/d3d12_xs_device.h"
#include "../src/command_list_epochs.h"

#include <cstdint>
#include <cstdio>

#define CHECK(x)                                                                                                       \
    do {                                                                                                               \
        if (!(x)) {                                                                                                    \
            std::fprintf(stderr, "line %d: %s\n", __LINE__, #x);                                                       \
            return 1;                                                                                                  \
        }                                                                                                              \
    } while (0)

using namespace d3d12test;
namespace xs = rt::d3d12_xs;
namespace epochs = leonida::command_list_epochs;

namespace {
using Create = HRESULT(WINAPI *)(void *, const void *, ID3D12CommandAllocator *, ID3D12PipelineState *, const IID *,
                                 void **);
// Console list slots on the newer layout (an undetected one reads the same): wait64(address, comparison, reference,
// flags) and SetComputeRootSignature (PC slot 29).
using WaitFn = std::uint64_t(WINAPI *)(void *, std::uint64_t, std::uint64_t, std::uint64_t, std::uint64_t);
using RootSignatureFn = std::uint64_t(WINAPI *)(void *, std::uint64_t);
constexpr unsigned kConsoleWait64 = 74, kConsoleSetComputeRootSignature = 33;
constexpr std::uint64_t kGreaterEqual = 7;  // memory >= reference, in D3D12_COMPARISON_FUNC order

ULONG references(IUnknown *object) {
    object->AddRef();
    return object->Release();
}
}  // namespace

int main() {
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
    {
        D3D12_COMMAND_QUEUE_DESC queue_desc{};
        Ref<ID3D12CommandQueue> queue;
        CHECK(SUCCEEDED(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue.p))));
        Ref<ID3D12CommandAllocator> allocator;
        CHECK(SUCCEEDED(inner->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p))));
        Ref<ID3D12Fence> fence;
        CHECK(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))));
        // The work each wait follows: a copy between two buffers.
        const D3D12_HEAP_PROPERTIES default_heap{D3D12_HEAP_TYPE_DEFAULT};
        const auto desc = buffer_desc(4096);
        Ref<ID3D12Resource> source, destination;
        CHECK(SUCCEEDED(inner->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &desc,
                                                       D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                       IID_PPV_ARGS(&source.p))));
        CHECK(SUCCEEDED(inner->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &desc,
                                                       D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                       IID_PPV_ARGS(&destination.p))));
        // An empty root signature: the first recording sets it before its first cut, so every cut sets it again on
        // the spare that records the next part, which holds it until its next Reset.
        D3D12_VERSIONED_ROOT_SIGNATURE_DESC root_desc{};
        root_desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_0;
        const auto blob = serialize_root_signature(root_desc);
        CHECK(!blob.empty());
        Ref<ID3D12RootSignature> root;
        CHECK(SUCCEEDED(inner->CreateRootSignature(0, blob.data(), blob.size(), IID_PPV_ARGS(&root.p))));
        // What every wait reads: already met, so no submission is held at a gate.
        auto *cell = static_cast<volatile LONG64 *>(VirtualAlloc(nullptr, 4096, MEM_RESERVE | MEM_COMMIT,
                                                                PAGE_READWRITE));
        CHECK(cell);
        *cell = 1;

        // No other command list is alive in this process: the oldest recording is one of the lists below.
        CHECK(epochs::oldest() == UINT64_MAX);
        std::uint64_t create_desc[4]{};
        Ref<ID3D12GraphicsCommandList> list;
        CHECK(SUCCEEDED(reinterpret_cast<Create>(slot(device.p, 73))(device.p, create_desc, allocator.p, nullptr,
                                                                     &__uuidof(ID3D12GraphicsCommandList),
                                                                     reinterpret_cast<void **>(&list.p))));
        CHECK(xs::is_list_proxy(list.p));
        CHECK(SUCCEEDED(list->Close()));
        const auto wait = reinterpret_cast<WaitFn>(slot(list.p, kConsoleWait64));
        const auto set_root = reinterpret_cast<RootSignatureFn>(slot(list.p, kConsoleSetComputeRootSignature));
        UINT64 submitted = 0;
        // One recording, executed and waited for: `cuts` waits for a memory value, each after a copy, and a copy
        // after the last.
        const auto record = [&](unsigned cuts, bool with_root) {
            if (FAILED(allocator->Reset()) || FAILED(list->Reset(allocator.p, nullptr))) return false;
            if (with_root) set_root(list.p, reinterpret_cast<std::uint64_t>(root.p));
            for (unsigned i = 0; i < cuts; ++i) {
                list->CopyBufferRegion(destination.p, 0, source.p, 0, 256);
                wait(list.p, reinterpret_cast<std::uint64_t>(cell), kGreaterEqual, 1, 0);
            }
            list->CopyBufferRegion(destination.p, 0, source.p, 0, 256);
            if (FAILED(list->Close()) || xs::list_proxy_part_count(list.p) != cuts + 1) return false;
            ID3D12CommandList *lists[1] = {list.p};
            queue->ExecuteCommandLists(1, lists);
            return SUCCEEDED(queue->Signal(fence.p, ++submitted)) &&
                   SUCCEEDED(fence->SetEventOnCompletion(submitted, nullptr));
        };

        xs::reset_list_value_stats();
        // 1. Three cuts: the list has three spare real lists from the next Reset on, each of which set the root
        // signature again (the test's reference, the primary list's and the three spares').
        CHECK(record(3, true));
        const ULONG held_by_cuts = references(root.p);
        // 2. A title window retired now is freed only once no list recorded before this point is alive.
        const std::uint64_t retirement = epochs::advance();
        // 3. Recordings cut at most once: only the last spare records again; the other two stay idle.
        constexpr unsigned kRecordings = 8;
        for (unsigned i = 0; i < kRecordings; ++i) CHECK(record(i % 2, false));
        const std::uint64_t oldest = epochs::oldest();
        const ULONG held_after = references(root.p);
        const auto stats = xs::list_value_stats();
        std::printf("list cut spares: %llu cuts; after %u recordings cut at most once, the oldest live recording "
                    "began in epoch %llu (the retirement: %llu); the root signature the three-cut recording set has "
                    "%lu references (%lu after that recording)\n",
                    static_cast<unsigned long long>(stats.cuts), kRecordings,
                    static_cast<unsigned long long>(oldest), static_cast<unsigned long long>(retirement), held_after,
                    held_by_cuts);
        CHECK(stats.cuts == 3 + kRecordings / 2 && stats.cut_failures == 0);
        CHECK(held_by_cuts == 5);
        // Every recording alive began after the retirement: a window retired there would get its ticket now.
        CHECK(oldest != UINT64_MAX && oldest >= retirement);
        // Nothing but the test holds the root signature: the idle spares let go of what their recordings used.
        CHECK(held_after == 1);
        VirtualFree(const_cast<LONG64 *>(cell), 0, MEM_RELEASE);
    }
    // The list and its spares are gone: no recording is alive.
    CHECK(epochs::oldest() == UINT64_MAX);
    CHECK(gpu.validation_errors() == 0);
    std::puts("list cut spares: idle spare lists hold back no retired title window and hold no root signature");
    return 0;
}
