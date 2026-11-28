// Console Xbox CreateDescriptorHeap (issue #313 comment 7107265058) passes a 16-byte PC
// D3D12_DESCRIPTOR_HEAP_DESC followed by address at +0x10 and size at +0x18. The title asserts
// GetCPUDescriptorHandleForHeapStart() == address. This tests the explicit Leonida console-heap
// creation path used by the Xbox device proxy slot 14. It does not cover title-memory aliasing /
// section 3.3 backing.
#include "vk_test_device.h"
#include "../shims/d3d12_com.h"
#include "../shims/d3d12_xs_device.h"
#include "../src/title_page_watch.h"

#include <cstdint>
#include <cstdio>

namespace {

// The pages a test write watch made read-only, and its opener: the D3D12 layer's slot-133 shadows protect the title's
// pages the same way and give them back through leonida::kernel_writes.
std::uintptr_t g_watched_begin = 0, g_watched_end = 0;
void open_watched(std::uintptr_t from, std::uintptr_t to) {
    DWORD old = 0;
    if (from < g_watched_end && to > g_watched_begin)
        VirtualProtect(reinterpret_cast<void*>(g_watched_begin), g_watched_end - g_watched_begin, PAGE_READWRITE, &old);
}

template <class T>
struct Ref {
    T* p = nullptr;
    ~Ref() { if (p) p->Release(); }
    T** put() {
        if (p) {
            p->Release();
            p = nullptr;
        }
        return &p;
    }
    T* get() const { return p; }
    T* operator->() const { return p; }
};

bool check_console_addressed_heap(ID3D12Device* device) {
    D3D12_DESCRIPTOR_HEAP_DESC desc{};
    desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    desc.NumDescriptors = 8;
    desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    desc.NodeMask = 0;
    const D3D12_GPU_VIRTUAL_ADDRESS address = 0x112233440000ull;
    const std::uint64_t size = 8ull * 32ull;

    Ref<ID3D12DescriptorHeap> heap;
    const HRESULT hr = rt::d3d12::create_console_descriptor_heap(
        device, &desc, address, size, IID_PPV_ARGS(heap.put()));
    if (FAILED(hr) || !heap.get()) {
        std::fprintf(stderr, "create_console_descriptor_heap failed: 0x%08lx\n", static_cast<unsigned long>(hr));
        return false;
    }
    const auto cpu = heap->GetCPUDescriptorHandleForHeapStart();
    if (cpu.ptr != static_cast<SIZE_T>(address)) {
        std::fprintf(stderr, "CPU heap start mismatch: got 0x%llx want 0x%llx\n",
                     static_cast<unsigned long long>(cpu.ptr), static_cast<unsigned long long>(address));
        return false;
    }
    const auto gpu = heap->GetGPUDescriptorHandleForHeapStart();
    if (gpu.ptr != address) {
        std::fprintf(stderr, "GPU heap start mismatch: got 0x%llx want 0x%llx\n",
                     static_cast<unsigned long long>(gpu.ptr), static_cast<unsigned long long>(address));
        return false;
    }
    const D3D12_DESCRIPTOR_HEAP_DESC got = heap->GetDesc();
    if (got.Type != desc.Type || got.NumDescriptors != desc.NumDescriptors || got.Flags != desc.Flags ||
        got.NodeMask != 1) {
        std::fprintf(stderr, "GetDesc mismatch\n");
        return false;
    }
    return true;
}

bool check_xs_proxy_slot14(ID3D12Device* real_device, bool expect_console) {
    void* proxy_raw = nullptr;
    // The proxy takes one reference to the device on success; the caller's Ref keeps its own.
    real_device->AddRef();
    if (FAILED(rt::d3d12_xs::create_device_proxy(real_device, __uuidof(ID3D12Device), &proxy_raw)) || !proxy_raw) {
        real_device->Release();
        std::fprintf(stderr, "create_device_proxy failed\n");
        return false;
    }
    Ref<ID3D12Device> proxy;
    proxy.p = static_cast<ID3D12Device*>(proxy_raw);

    struct ConsoleDescriptorHeapDesc {
        D3D12_DESCRIPTOR_HEAP_DESC desc{};
        std::uint64_t address = 0;
        std::uint64_t size = 0;
    } console{};
    console.desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    console.desc.NumDescriptors = 4;
    console.desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    alignas(64) unsigned char storage[4096]{};
    console.address = reinterpret_cast<std::uintptr_t>(storage);
    console.size = console.desc.NumDescriptors * real_device->GetDescriptorHandleIncrementSize(console.desc.Type);
    if (console.size > sizeof(storage)) return false;
    D3D12_DESCRIPTOR_HEAP_DESC ordinary = console.desc;
    Ref<ID3D12DescriptorHeap> ordinary_heap;
    if (FAILED(proxy->CreateDescriptorHeap(&ordinary, IID_PPV_ARGS(ordinary_heap.put())))) return false;

    auto check_start = [&](const char* label, bool uses_console_address) {
        Ref<ID3D12DescriptorHeap> heap;
        const HRESULT hr = proxy->CreateDescriptorHeap(&console.desc, IID_PPV_ARGS(heap.put()));
        if (FAILED(hr) || !heap.get()) {
            std::fprintf(stderr, "%s: CreateDescriptorHeap failed: 0x%08lx\n", label, static_cast<unsigned long>(hr));
            return false;
        }
        // Xbox heap slot 9 returns the handle in RAX rather than through a hidden PC return pointer.
        using GetCpuHandleFn = D3D12_CPU_DESCRIPTOR_HANDLE(WINAPI*)(ID3D12DescriptorHeap*);
        void** const vtbl = *reinterpret_cast<void***>(heap.get());
        const auto cpu = reinterpret_cast<GetCpuHandleFn>(vtbl[9])(heap.get());
        if (!cpu.ptr || (cpu.ptr == console.address) != uses_console_address) {
            std::fprintf(stderr, "%s: expected %s handle, got 0x%llx (console address 0x%llx)\n",
                         label, uses_console_address ? "console" : "ordinary",
                         static_cast<unsigned long long>(cpu.ptr), static_cast<unsigned long long>(console.address));
            return false;
        }
        return true;
    };
    if (!expect_console) return check_start("host device", false);
    if (!check_start("writable backing", true)) return false;
    ++console.size;
    if (!check_start("size mismatch", false)) return false;
    --console.size;

    void* read_only = VirtualAlloc(nullptr, console.size, MEM_RESERVE | MEM_COMMIT, PAGE_READONLY);
    if (!read_only) return false;
    console.address = reinterpret_cast<std::uintptr_t>(read_only);
    const bool ok = check_start("read-only backing", false);
    VirtualFree(read_only, 0, MEM_RELEASE);
    if (!ok) return false;

    // Writable memory a write watch has made read-only (the retail title places a slot-133 buffer over the memory it
    // carves its heap from, and that buffer's shadow watches it) is still the title's writable memory.
    void* watched = VirtualAlloc(nullptr, console.size, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    DWORD old = 0;
    if (!watched || !VirtualProtect(watched, console.size, PAGE_READONLY, &old)) return false;
    g_watched_begin = reinterpret_cast<std::uintptr_t>(watched);
    g_watched_end = g_watched_begin + console.size;
    leonida::kernel_writes::set_opener(&open_watched);
    console.address = g_watched_begin;
    const bool watched_ok = check_start("watched backing", true);
    leonida::kernel_writes::set_opener(nullptr);
    VirtualFree(watched, 0, MEM_RELEASE);
    return watched_ok;
}

bool check_normal_heap_unchanged(ID3D12Device* device) {
    D3D12_DESCRIPTOR_HEAP_DESC desc{};
    desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    desc.NumDescriptors = 4;
    desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    Ref<ID3D12DescriptorHeap> heap;
    const HRESULT hr = device->CreateDescriptorHeap(&desc, IID_PPV_ARGS(heap.put()));
    if (FAILED(hr) || !heap.get()) {
        std::fprintf(stderr, "CreateDescriptorHeap(normal) failed: 0x%08lx\n", static_cast<unsigned long>(hr));
        return false;
    }
    const auto cpu = heap->GetCPUDescriptorHandleForHeapStart();
    if (cpu.ptr == 0) {
        std::fprintf(stderr, "normal heap CPU start is null\n");
        return false;
    }
    return true;
}

}  // namespace

int main() {
    vktest::Device vk;
    const auto status = vk.initialize();
    if (status == vktest::DeviceStatus::unavailable) return 77;
    if (status != vktest::DeviceStatus::ready) return 1;

    Ref<ID3D12Device> device;
    rt::d3d12::VulkanDeviceInfo info{};
    info.get_instance_proc = vk.instance_proc;
    info.instance = vk.instance;
    info.physical = vk.physical;
    info.device = vk.device;
    info.queue = vk.queue;
    info.queue_family = vk.family;
    if (FAILED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(device.put()))) || !device.get()) {
        std::fprintf(stderr, "create_vulkan_device failed\n");
        return 1;
    }

    if (!check_normal_heap_unchanged(device.get())) return 1;
    if (!check_console_addressed_heap(device.get())) return 1;
    Ref<ID3D12Device> host;
    if (FAILED(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(host.put())))) return 1;
    if (!check_xs_proxy_slot14(host.get(), false)) return 1;
    if (!check_xs_proxy_slot14(device.get(), true)) return 1;
    std::fprintf(stderr, "ok\n");
    return 0;
}
