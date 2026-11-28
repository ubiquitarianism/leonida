// LEONIDA_SHADOW_GPU=1 without LEONIDA_CONSOLE_HEAP_GPU (RT-SHADOW-GPU-RELEASED-RANGES): the device makes the
// console heaps' GPU pass object for the shadows' copies alone, and the pass itself never runs, so nothing may keep
// the arena ranges released and the alias ranges removed for it: only the pass takes those lists, at the start of a
// submission that runs it. Before, every arena release added two entries (backing() ran twice for it), a slot-133
// import added one for its backing and one for its alias range, and both lists grew for as long as the process ran.
// Checked on the Vulkan-backed device: committed buffers and slot-133 buffers made and released, with submissions
// between, leave nothing waiting. With --console-heap-gpu (LEONIDA_CONSOLE_HEAP_GPU=1) the pass still gets one entry
// per arena release, and the next submission takes them all. Both release the device at the end, which owns its
// VkDevice here: the pass's pipeline and buffers were destroyed after it before, a fault in ~ConsoleHeapGpu. No game
// data. LEONIDA_VK_VALIDATION=1 adds VK_LAYER_KHRONOS_validation.
#include "../shims/d3d12_com.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {
template<class T>
struct Ref {
    T* p = nullptr;
    ~Ref() { reset(); }
    T* operator->() const { return p; }
    void reset() {
        if (p) p->Release();
        p = nullptr;
    }
};

D3D12_RESOURCE_DESC buffer(UINT64 bytes) {
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = bytes;
    desc.Height = desc.DepthOrArraySize = desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    return desc;
}

// An empty command list, executed and waited for: what runs before every submission runs (refresh_console_heaps).
struct Submitter {
    Ref<ID3D12CommandQueue> queue;
    Ref<ID3D12CommandAllocator> allocator;
    Ref<ID3D12GraphicsCommandList> list;
    Ref<ID3D12Fence> fence;
    HANDLE done = nullptr;
    UINT64 value = 0;
    ~Submitter() {
        if (done) CloseHandle(done);
    }
    bool init(ID3D12Device* device) {
        D3D12_COMMAND_QUEUE_DESC desc{D3D12_COMMAND_LIST_TYPE_DIRECT};
        done = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        return done && SUCCEEDED(device->CreateCommandQueue(&desc, IID_PPV_ARGS(&queue.p))) &&
               SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p))) &&
               SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, nullptr,
                                                   IID_PPV_ARGS(&list.p))) &&
               SUCCEEDED(list->Close()) &&
               SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p)));
    }
    bool submit() {
        ID3D12CommandList* lists[]{list.p};
        queue->ExecuteCommandLists(1, lists);
        return SUCCEEDED(queue->Signal(fence.p, ++value)) && SUCCEEDED(fence->SetEventOnCompletion(value, done)) &&
               WaitForSingleObject(done, 30000) == WAIT_OBJECT_0;
    }
};

// `count` committed buffers made and released: as many arena releases.
bool churn_committed(ID3D12Device* device, unsigned count) {
    const D3D12_HEAP_PROPERTIES heap{D3D12_HEAP_TYPE_DEFAULT};
    const D3D12_RESOURCE_DESC desc = buffer(64 << 10);
    for (unsigned i = 0; i < count; ++i) {
        Ref<ID3D12Resource> made;
        if (FAILED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON,
                                                   nullptr, IID_PPV_ARGS(&made.p))))
            return false;
    }
    return true;
}
}  // namespace

int main(int argc, char** argv) {
    const bool heap_pass = argc == 2 && std::strcmp(argv[1], "--console-heap-gpu") == 0;
    if (argc != 1 && !heap_pass) return 2;
    // Both are read once, when the device is made. An empty value removes the variable.
    _putenv_s("LEONIDA_SHADOW_GPU", "1");
    _putenv_s("LEONIDA_CONSOLE_HEAP_GPU", heap_pass ? "1" : "");
    Ref<ID3D12Device> device;
    if (FAILED(rt::d3d12::create_vulkan_device_standalone(IID_PPV_ARGS(&device.p))) || !device.p) {
        std::printf("SKIP: no Vulkan device\n");
        return 77;  // ctest skips
    }
    if (!rt::d3d12::device_arena_size(device.p) || !rt::d3d12::console_heap_gpu_pending(device.p).made) {
        std::printf("SKIP: no GPU address arena, or the console heap pass could not be made on this device\n");
        return 77;
    }
    Submitter gpu;
    if (!gpu.init(device.p) || !gpu.submit()) {
        std::printf("FAIL: a queue and an empty command list\n");
        return 1;
    }
    constexpr unsigned kRounds = 16, kBuffers = 64;
    int failures = 0;

    if (heap_pass) {
        // The pass runs at every submission: one entry per release (not two), all taken by the next submission.
        const auto before = rt::d3d12::console_heap_gpu_pending(device.p);
        if (!churn_committed(device.p, kBuffers)) {
            std::printf("FAIL: CreateCommittedResource\n");
            return 1;
        }
        const auto released = rt::d3d12::console_heap_gpu_pending(device.p);
        std::printf("LEONIDA_CONSOLE_HEAP_GPU=1: %u buffers released: %llu ranges waiting (%llu before)\n", kBuffers,
                    static_cast<unsigned long long>(released.released),
                    static_cast<unsigned long long>(before.released));
        if (released.released - before.released != kBuffers) {
            std::printf("FAIL: an arena release is not one range for the pass\n");
            ++failures;
        }
        if (!gpu.submit()) {
            std::printf("FAIL: a submission\n");
            return 1;
        }
        const auto after = rt::d3d12::console_heap_gpu_pending(device.p);
        std::printf("after a submission: %llu released ranges, %llu removed alias ranges waiting\n",
                    static_cast<unsigned long long>(after.released), static_cast<unsigned long long>(after.removed));
        if (after.released || after.removed) {
            std::printf("FAIL: the pass did not take what was released\n");
            ++failures;
        }
        if (!failures) std::printf("console heap pass: the released ranges are taken at the next submission\n");
        return failures ? 1 : 0;
    }

    // Slot-133 buffers on pages of our own (an import and its alias range each, where the driver imports them).
    constexpr SIZE_T kPages = 64 << 10;
    void* pages = VirtualAlloc(nullptr, kPages, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    if (!pages) return 1;
    unsigned placed = 0;
    for (unsigned round = 0; round < kRounds && !failures; ++round) {
        if (!churn_committed(device.p, kBuffers)) {
            std::printf("FAIL: CreateCommittedResource\n");
            ++failures;
        }
        Ref<ID3D12Resource> title;
        if (SUCCEEDED(rt::d3d12::create_buffer_at_host_address(device.p, pages, buffer(kPages), 0xFFFFFFFFu,
                                                               IID_PPV_ARGS(&title.p))))
            ++placed;
        title.reset();
        if (!gpu.submit()) {
            std::printf("FAIL: a submission\n");
            ++failures;
        }
    }
    VirtualFree(pages, 0, MEM_RELEASE);
    const auto pending = rt::d3d12::console_heap_gpu_pending(device.p);
    std::printf("LEONIDA_SHADOW_GPU=1 alone: %u committed and %u slot-133 buffers released over %u submissions: %llu "
                "released ranges and %llu removed alias ranges waiting\n",
                kRounds * kBuffers, placed, kRounds, static_cast<unsigned long long>(pending.released),
                static_cast<unsigned long long>(pending.removed));
    if (pending.released || pending.removed) {
        std::printf("FAIL: ranges are kept for a console heap pass that never runs\n");
        ++failures;
    }
    if (!failures) std::printf("shadow GPU copies: nothing is kept for the console heap pass\n");
    return failures ? 1 : 0;
}
