// Device teardown with background pipeline builds queued or running (PipelineCompiler, shims/d3d12_pipeline_compile.h).
// Every compute PSO queues a warm-up build at creation (lazy compute, LEONIDA_COMPUTE_WARMUP). The test's VkDevice
// outlives each D3D12 device made over it, as a title's would not: a build left running after the last Release
// would reach a device the test goes on to destroy, which is how d3d12_subgroup_test crashed with warm-up on.
// A job probe (PipelineCompiler::set_job_probe) parks a PSO's real warm-up build on the worker, after it counts as
// running and before it calls the driver, so "in flight" is a state the test holds, not a race. Cases:
//   cache         a build in the background, then release: the device writes the session cache, the worker joins.
//   queued        the worker held by a gate while PSOs are created and released: every queued build is dropped.
//   prompt        a PSO released while its build is in flight returns at once; the build then runs on the released
//                 PSO and the worker deletes it.
//   last device   the PSO holding the device's last reference released while its build is in flight: the device's
//                 teardown waits for the build, then the worker is joined.
//   on worker     the PSO released on the worker from inside its own build, once with the device still held and once
//                 holding its last reference (the device is then torn down on the worker, which lets itself go).
//   two devices   one torn down while the other's builds are queued behind it: the other's still run.
//   racing        many PSOs released at once with builds queued and one usually running, repeated.
//   restart       a device made after the worker stopped gets a new one, and its builds run.
//   exit          stop_pipeline_compiles with a device alive: the queue goes, later builds are refused, a warm-up
//                 is skipped and a dispatch (LEONIDA_ASYNC_PIPELINES on) builds its pipeline in place and runs;
//                 save_pipeline_caches_at_exit then writes the session cache with the device alive, once.
// After each case the compiler's thread is gone and the process's thread count is back where it was.
// --seed instead runs the session file beside an offline seed (seeded_session): a session file written beside the
// seed there now, as pipeline_cache.session.seed records, is loaded alone, the seed not read and nothing merged, and
// its pipelines, the seed's among them, are cache hits; a touched seed, no seed, no record,
// LEONIDA_PIPELINE_CACHE_BYTES=0 and LEONIDA_SESSION_CACHE_DIRECT_SEED=0 behave as before.
// No game data: the shader is test/glsl/cs_user_data.comp, the programs are made-up FSSM bytes.
#include "shader_fixture_path.h"
#include "test_temp_dir.h"
#include "vk_test_device.h"
#include "../shims/d3d12_com.h"
#include "../shims/d3d12_compile_exit.h"
#include "../shims/d3d12_pipeline_compile.h"
#include "../shims/d3d12_xs_device.h"
#include "../src/pipeline_cache_store.h"
#include "../src/shader_library.h"

#include <tlhelp32.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#ifndef LEONIDA_TEST_CS_SPV
#error "LEONIDA_TEST_CS_SPV must name the compiled test shader"
#endif

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

namespace {

using XboxComputePipelineFn = HRESULT(WINAPI*)(void* device, const D3D12_COMPUTE_PIPELINE_STATE_DESC*, std::uint64_t,
                                               std::uint64_t, const IID*, void**);
using XboxHeapHandleFn = std::uint64_t(WINAPI*)(void* heap);  // heap slots 9/10: handle in RAX
using SerializeFn = HRESULT(WINAPI*)(const D3D12_VERSIONED_ROOT_SIGNATURE_DESC*, ID3DBlob**, ID3DBlob**);
using Clock = std::chrono::steady_clock;

constexpr unsigned kPrograms = 48;  // PSOs per device: enough that the worker is still busy at the release
constexpr unsigned kRacingRounds = 12;

void* slot(void* object, unsigned index) { return (*reinterpret_cast<void***>(object))[index]; }

void put32(std::vector<std::uint8_t>& out, std::uint32_t v) {
    for (int i = 0; i < 4; ++i) out.push_back(std::uint8_t(v >> (8 * i)));
}

std::vector<std::uint8_t> zlib_stored(const std::vector<std::uint8_t>& data) {
    std::vector<std::uint8_t> out = {0x78, 0x01, 0x01};
    const auto length = std::uint16_t(data.size());
    out.insert(out.end(), {std::uint8_t(length), std::uint8_t(length >> 8), std::uint8_t(~length),
                           std::uint8_t(~length >> 8)});
    out.insert(out.end(), data.begin(), data.end());
    std::uint32_t a = 1, b = 0;
    for (const std::uint8_t x : data) {
        a = (a + x) % 65521;
        b = (b + a) % 65521;
    }
    const std::uint32_t adler = (b << 16) | a;
    out.insert(out.end(), {std::uint8_t(adler >> 24), std::uint8_t(adler >> 16), std::uint8_t(adler >> 8),
                           std::uint8_t(adler)});
    return out;
}

// DXBC { DXIL { stub, XDAT { 0x0D: XBCH + zlib(fssm) } } }, as in d3d12_compute_test.
std::vector<std::uint8_t> shader_container(const std::vector<std::uint8_t>& fssm) {
    const auto stream = zlib_stored(fssm);
    std::vector<std::uint8_t> xbch = {'X', 'B', 'C', 'H'};
    for (const std::uint32_t v : {1u, std::uint32_t(stream.size()), 0u, std::uint32_t(fssm.size()), 0u}) put32(xbch, v);
    xbch.insert(xbch.end(), stream.begin(), stream.end());
    std::vector<std::uint8_t> dxil(16, 0);
    dxil.insert(dxil.end(), {'X', 'D', 'A', 'T'});
    put32(dxil, 1);
    put32(dxil, 12);
    put32(dxil, 0x0d);
    put32(dxil, 16 + 12 + 12);
    put32(dxil, std::uint32_t(xbch.size()));
    dxil.insert(dxil.end(), xbch.begin(), xbch.end());
    std::vector<std::uint8_t> out = {'D', 'X', 'B', 'C'};
    out.resize(28, 0);
    put32(out, 1);
    put32(out, 36);
    out.insert(out.end(), {'D', 'X', 'I', 'L'});
    put32(out, std::uint32_t(dxil.size()));
    out.insert(out.end(), dxil.begin(), dxil.end());
    return out;
}

// Program i: a made-up FSSM blob whose code word (dword 1, which the shader reads through its code base) differs,
// so each has its own hash and its own index entry.
std::uint32_t code_word(unsigned i) { return 7000 + i; }
std::vector<std::uint8_t> program(unsigned i) {
    std::vector<std::uint8_t> fssm = {'F', 'S', 'S', 'M'};
    put32(fssm, code_word(i));
    for (std::uint32_t w = 0; w < 14; ++w) put32(fssm, 0xbf810000u + w);
    return fssm;
}

std::vector<std::uint8_t> serialize(const D3D12_VERSIONED_ROOT_SIGNATURE_DESC& desc) {
    const HMODULE d3d12 = LoadLibraryExA("d3d12.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    const auto fn = d3d12 ? reinterpret_cast<SerializeFn>(
                                reinterpret_cast<void*>(GetProcAddress(d3d12, "D3D12SerializeVersionedRootSignature")))
                          : nullptr;
    std::vector<std::uint8_t> out;
    ID3DBlob* blob = nullptr;
    if (fn && SUCCEEDED(fn(&desc, &blob, nullptr)) && blob) {
        const auto* p = static_cast<const std::uint8_t*>(blob->GetBufferPointer());
        out.assign(p, p + blob->GetBufferSize());
        blob->Release();
    }
    return out;
}

D3D12_RESOURCE_DESC buffer_desc(UINT64 bytes, D3D12_RESOURCE_FLAGS flags) {
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = bytes;
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    desc.Flags = flags;
    return desc;
}

unsigned thread_count() {
    const HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
    if (snapshot == INVALID_HANDLE_VALUE) return 0;
    unsigned count = 0;
    THREADENTRY32 entry{sizeof(entry)};
    for (BOOL more = Thread32First(snapshot, &entry); more; more = Thread32Next(snapshot, &entry))
        if (entry.th32OwnerProcessID == GetCurrentProcessId()) ++count;
    CloseHandle(snapshot);
    return count;
}

// Waits up to `ms` for `done`; true when it held.
template<class Done> bool eventually(Done done, unsigned ms = 10000) {
    const auto until = Clock::now() + std::chrono::milliseconds(ms);
    while (!done()) {
        if (Clock::now() > until) return false;
        Sleep(5);
    }
    return true;
}

struct Event {
    HANDLE handle = CreateEventW(nullptr, TRUE, FALSE, nullptr);
    Event() = default;
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
    ~Event() { CloseHandle(handle); }
    void set() const { SetEvent(handle); }
    bool wait(DWORD ms = 30000) const { return WaitForSingleObject(handle, ms) == WAIT_OBJECT_0; }
};

bool built(ID3D12PipelineState* pso) {
    const auto* requests = rt::d3d12::pipeline_subgroup_requests(pso);
    return requests && !requests->empty();  // recorded when the Vulkan pipeline is made (build_compute)
}

// --seed: what the device does with its pipeline caches, seen through its vkGetDeviceProcAddr: the merges (the
// session file is the only cache the runtime merges), the size of every blob a cache is created from, and how many
// compute pipelines the driver found in the cache they were created against (pipeline creation feedback, core in
// Vulkan 1.3).
struct CacheCalls {
    std::atomic<unsigned> merges{0}, builds{0}, hits{0};
    std::mutex mutex;
    std::vector<std::size_t> loads;  // initialDataSize of each vkCreatePipelineCache with data, in order
    void reset() {
        merges = builds = hits = 0;
        std::lock_guard lock(mutex);
        loads.clear();
    }
};
CacheCalls cache_calls;
PFN_vkGetInstanceProcAddr real_instance_proc = nullptr;
PFN_vkGetDeviceProcAddr real_device_proc = nullptr;
PFN_vkMergePipelineCaches real_merge = nullptr;
PFN_vkCreatePipelineCache real_create_cache = nullptr;
PFN_vkCreateComputePipelines real_create_compute = nullptr;

VKAPI_ATTR VkResult VKAPI_CALL counted_merge(VkDevice device, VkPipelineCache into, std::uint32_t count,
                                             const VkPipelineCache* from) {
    ++cache_calls.merges;
    return real_merge(device, into, count, from);
}

VKAPI_ATTR VkResult VKAPI_CALL counted_create_cache(VkDevice device, const VkPipelineCacheCreateInfo* info,
                                                    const VkAllocationCallbacks* allocator, VkPipelineCache* cache) {
    if (info && info->initialDataSize) {
        std::lock_guard lock(cache_calls.mutex);
        cache_calls.loads.push_back(info->initialDataSize);
    }
    return real_create_cache(device, info, allocator, cache);
}

VKAPI_ATTR VkResult VKAPI_CALL counted_create_compute(VkDevice device, VkPipelineCache cache, std::uint32_t count,
                                                      const VkComputePipelineCreateInfo* infos,
                                                      const VkAllocationCallbacks* allocator, VkPipeline* pipelines) {
    // The runtime's own feedback when it chains one (a second in the chain would be invalid), ours otherwise.
    std::vector<VkComputePipelineCreateInfo> copies(infos, infos + count);
    std::vector<VkPipelineCreationFeedback> ours(count);
    std::vector<VkPipelineCreationFeedbackCreateInfo> chained(count);
    std::vector<const VkPipelineCreationFeedback*> read(count);
    for (std::uint32_t i = 0; i < count; ++i) {
        for (auto* p = static_cast<const VkBaseInStructure*>(copies[i].pNext); p; p = p->pNext)
            if (p->sType == VK_STRUCTURE_TYPE_PIPELINE_CREATION_FEEDBACK_CREATE_INFO)
                read[i] = reinterpret_cast<const VkPipelineCreationFeedbackCreateInfo*>(p)->pPipelineCreationFeedback;
        if (read[i]) continue;
        chained[i] = {VK_STRUCTURE_TYPE_PIPELINE_CREATION_FEEDBACK_CREATE_INFO, copies[i].pNext, &ours[i], 0, nullptr};
        copies[i].pNext = &chained[i];
        read[i] = &ours[i];
    }
    const VkResult result = real_create_compute(device, cache, count, copies.data(), allocator, pipelines);
    if (result == VK_SUCCESS)
        for (std::uint32_t i = 0; i < count; ++i) {
            ++cache_calls.builds;
            constexpr auto hit = VK_PIPELINE_CREATION_FEEDBACK_VALID_BIT |
                                 VK_PIPELINE_CREATION_FEEDBACK_APPLICATION_PIPELINE_CACHE_HIT_BIT;
            if ((read[i]->flags & hit) == hit) ++cache_calls.hits;
        }
    return result;
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL counted_device_proc(VkDevice device, const char* name) {
    const PFN_vkVoidFunction real = real_device_proc(device, name);
    if (!real) return real;
    if (!std::strcmp(name, "vkMergePipelineCaches")) {
        real_merge = reinterpret_cast<PFN_vkMergePipelineCaches>(real);
        return reinterpret_cast<PFN_vkVoidFunction>(counted_merge);
    }
    if (!std::strcmp(name, "vkCreatePipelineCache")) {
        real_create_cache = reinterpret_cast<PFN_vkCreatePipelineCache>(real);
        return reinterpret_cast<PFN_vkVoidFunction>(counted_create_cache);
    }
    if (!std::strcmp(name, "vkCreateComputePipelines")) {
        real_create_compute = reinterpret_cast<PFN_vkCreateComputePipelines>(real);
        return reinterpret_cast<PFN_vkVoidFunction>(counted_create_compute);
    }
    return real;
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL counted_instance_proc(VkInstance instance, const char* name) {
    if (!std::strcmp(name, "vkGetDeviceProcAddr")) {
        real_device_proc = reinterpret_cast<PFN_vkGetDeviceProcAddr>(real_instance_proc(instance, name));
        return reinterpret_cast<PFN_vkVoidFunction>(counted_device_proc);
    }
    return real_instance_proc(instance, name);
}

// One D3D12 device over the test's VkDevice, the compute test's root signature (two constants, a root UAV, a CBV
// table) and PSOs. The title's references, released in any order with release().
struct Session {
    ID3D12Device* device = nullptr;
    ID3D12RootSignature* root = nullptr;
    std::vector<ID3D12PipelineState*> psos;
    bool count_cache_calls = false;  // the device gets its functions through counted_instance_proc

    int open(vktest::Device& d, const std::filesystem::path& library, unsigned count) {
        rt::d3d12::VulkanDeviceInfo info{};
        real_instance_proc = d.instance_proc;
        info.get_instance_proc = count_cache_calls ? counted_instance_proc : d.instance_proc;
        info.instance = d.instance;
        info.physical = d.physical;
        info.device = d.device;
        info.queue = d.queue;
        info.queue_family = d.family;
        info.buffer_device_address = true;
        info.subgroup = d.subgroup;
        info.shader_dir = library.string();
        ID3D12Device* inner = nullptr;
        EXPECT(SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&inner))));
        const HRESULT proxied = rt::d3d12_xs::create_device_proxy(inner, __uuidof(ID3D12Device),
                                                                  reinterpret_cast<void**>(&device), false);
        if (FAILED(proxied)) inner->Release();
        EXPECT(SUCCEEDED(proxied));
        rt::d3d12::set_device_outer(inner, device);

        D3D12_DESCRIPTOR_RANGE range{D3D12_DESCRIPTOR_RANGE_TYPE_CBV, 1, 1, 0, 0};
        D3D12_ROOT_PARAMETER params[3] = {};
        params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        params[0].Constants = {0, 0, 2};
        params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
        params[1].Descriptor = {0, 0};
        params[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        params[2].DescriptorTable = {1, &range};
        D3D12_VERSIONED_ROOT_SIGNATURE_DESC root_desc{};
        root_desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_0;
        root_desc.Desc_1_0 = {3, params, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
        const auto blob = serialize(root_desc);
        EXPECT(!blob.empty());
        EXPECT(SUCCEEDED(device->CreateRootSignature(0, blob.data(), blob.size(), IID_PPV_ARGS(&root))));
        for (unsigned i = 0; i < count; ++i) EXPECT(add(i) == 0);
        return 0;
    }

    int add(unsigned program_index) {
        const auto container = shader_container(program(program_index));
        D3D12_COMPUTE_PIPELINE_STATE_DESC desc{};
        desc.pRootSignature = root;
        desc.CS = {container.data(), container.size()};
        const IID iid = __uuidof(ID3D12PipelineState);
        ID3D12PipelineState* pso = nullptr;
        EXPECT(SUCCEEDED(reinterpret_cast<XboxComputePipelineFn>(slot(device, 62))(
            device, &desc, 0, 0, &iid, reinterpret_cast<void**>(&pso))));
        psos.push_back(pso);
        return 0;
    }

    // One dispatch of `pso` (program `program_index`): out[i] = 7 + 3 i + table[i % 4] + code word, read back.
    int dispatch(ID3D12PipelineState* pso, unsigned program_index) {
        constexpr UINT elements = 64, bytes = elements * 4;
        const std::uint32_t table_values[4] = {1000, 2000, 3000, 4000};
        const D3D12_HEAP_PROPERTIES default_heap{D3D12_HEAP_TYPE_DEFAULT};
        const D3D12_HEAP_PROPERTIES upload_heap{D3D12_HEAP_TYPE_UPLOAD};
        const D3D12_HEAP_PROPERTIES readback_heap{D3D12_HEAP_TYPE_READBACK};
        const auto uav = buffer_desc(bytes, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        const auto constants_desc = buffer_desc(256, D3D12_RESOURCE_FLAG_NONE);
        const auto readback_desc = buffer_desc(bytes, D3D12_RESOURCE_FLAG_NONE);
        ID3D12Resource *out = nullptr, *constants = nullptr, *readback = nullptr;
        ID3D12DescriptorHeap* heap = nullptr;
        ID3D12CommandQueue* queue = nullptr;
        ID3D12CommandAllocator* allocator = nullptr;
        ID3D12GraphicsCommandList* list = nullptr;
        ID3D12Fence* fence = nullptr;
        struct Releaser {
            std::vector<IUnknown**> all;
            ~Releaser() {
                for (IUnknown** p : all)
                    if (*p) (*p)->Release();
            }
        } releaser{{reinterpret_cast<IUnknown**>(&out), reinterpret_cast<IUnknown**>(&constants),
                    reinterpret_cast<IUnknown**>(&readback), reinterpret_cast<IUnknown**>(&heap),
                    reinterpret_cast<IUnknown**>(&queue), reinterpret_cast<IUnknown**>(&allocator),
                    reinterpret_cast<IUnknown**>(&list), reinterpret_cast<IUnknown**>(&fence)}};
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &uav,
                                                         D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
                                                         IID_PPV_ARGS(&out))));
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&upload_heap, D3D12_HEAP_FLAG_NONE, &constants_desc,
                                                         D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                                         IID_PPV_ARGS(&constants))));
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&readback_heap, D3D12_HEAP_FLAG_NONE, &readback_desc,
                                                         D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                         IID_PPV_ARGS(&readback))));
        void* mapped = nullptr;
        EXPECT(SUCCEEDED(constants->Map(0, nullptr, &mapped)));
        std::memcpy(mapped, table_values, sizeof(table_values));
        const D3D12_DESCRIPTOR_HEAP_DESC heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 8,
                                                   D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
        EXPECT(SUCCEEDED(device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&heap))));
        const std::uint64_t cpu = reinterpret_cast<XboxHeapHandleFn>(slot(heap, 9))(heap);
        const std::uint64_t gpu = reinterpret_cast<XboxHeapHandleFn>(slot(heap, 10))(heap);
        EXPECT(cpu && gpu >= 0x10000 && gpu < (1ull << 32));
        const std::uint64_t target = constants->GetGPUVirtualAddress();
        std::memcpy(reinterpret_cast<void*>(cpu), &target, sizeof(target));

        D3D12_COMMAND_QUEUE_DESC queue_desc{};
        EXPECT(SUCCEEDED(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue))));
        EXPECT(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator))));
        EXPECT(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator, pso,
                                                   IID_PPV_ARGS(&list))));
        EXPECT(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence))));
        list->SetDescriptorHeaps(1, &heap);
        list->SetComputeRootSignature(root);
        list->SetComputeRoot32BitConstant(0, 7, 0);
        list->SetComputeRoot32BitConstant(0, 3, 1);
        list->SetComputeRootUnorderedAccessView(1, out->GetGPUVirtualAddress());
        list->SetComputeRootDescriptorTable(2, D3D12_GPU_DESCRIPTOR_HANDLE{gpu});
        list->Dispatch(1, 1, 1);
        D3D12_RESOURCE_BARRIER barrier{};
        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier.Transition = {out, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                              D3D12_RESOURCE_STATE_COPY_SOURCE};
        list->ResourceBarrier(1, &barrier);
        list->CopyBufferRegion(readback, 0, out, 0, bytes);
        EXPECT(list->Close() == S_OK);
        ID3D12CommandList* lists[] = {list};
        queue->ExecuteCommandLists(1, lists);
        EXPECT(rt::d3d12::last_execute_result(queue) == leonida::ReplayResult::success);
        EXPECT(SUCCEEDED(queue->Signal(fence, 1)));
        EXPECT(SUCCEEDED(fence->SetEventOnCompletion(1, nullptr)));
        void* results = nullptr;
        EXPECT(SUCCEEDED(readback->Map(0, nullptr, &results)));
        std::vector<std::uint32_t> words(elements);
        std::memcpy(words.data(), results, bytes);
        for (std::uint32_t i = 0; i < elements; ++i)
            EXPECT(words[i] == 7 + 3 * i + table_values[i % 4] + code_word(program_index));
        return 0;
    }

    // The device first, as a title that lets its PSOs go last: the last PSO's Release destroys the device. In
    // creation order a PSO's build is usually the one running (the worker takes equal sizes in submission order);
    // newest first, the ones still queued are dropped.
    void release(bool newest_first = false) {
        if (device) device->Release();
        device = nullptr;
        if (root) root->Release();
        root = nullptr;
        if (newest_first)
            for (auto it = psos.rbegin(); it != psos.rend(); ++it) (*it)->Release();
        else
            for (ID3D12PipelineState* pso : psos) pso->Release();
        psos.clear();
    }
    ~Session() { release(); }
};

using rt::d3d12::PipelineCompiler;

// Waits until the worker has run everything queued so far: a last-priority marker job.
bool drain(PipelineCompiler& compiler) {
    Event done;
    static const int marker = 0;
    if (!compiler.submit(&marker, nullptr, SIZE_MAX, [&done] { done.set(); })) return false;
    return done.wait(120000);
}

// Holds the worker in a job of no device until open() (or the end of the scope).
struct Gate {
    Event entered, opened, left;
    explicit Gate(PipelineCompiler& compiler) {
        static const int gate = 0;
        compiler.submit(&gate, nullptr, 0, [this] {
            entered.set();
            opened.wait(INFINITE);
            left.set();
        });
        entered.wait();
    }
    void open() { opened.set(); }
    ~Gate() {
        opened.set();
        left.wait();  // the job is out of this object before its events close
    }
};

// Parks the build of one PSO on the worker once it counts as running, before its work (the real build) starts.
struct Park {
    PipelineCompiler& compiler;
    std::atomic<const void*> target{nullptr};
    Event parked, proceed;
    explicit Park(PipelineCompiler& c) : compiler(c) {
        compiler.set_job_probe([this](const void* owner) {
            if (owner != target.load()) return;
            parked.set();
            proceed.wait(INFINITE);
        });
    }
    ~Park() {
        proceed.set();
        compiler.set_job_probe(nullptr);
    }
};

std::vector<char> read_bytes(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}

bool write_bytes(const std::filesystem::path& path, const std::vector<char>& bytes) {
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    file.write(bytes.data(), std::streamsize(bytes.size()));
    return bool(file);
}

// A file's size and last-write time, as the runtime records a seed's (pipeline_cache_store::SeedStamp); zero when
// there is no such file.
leonida::pipeline_cache_store::SeedStamp stamp_of(const std::filesystem::path& path) {
    WIN32_FILE_ATTRIBUTE_DATA about{};
    if (!GetFileAttributesExW(path.c_str(), GetFileExInfoStandard, &about)) return {};
    return {(std::uint64_t(about.nFileSizeHigh) << 32) | about.nFileSizeLow,
            (std::uint64_t(about.ftLastWriteTime.dwHighDateTime) << 32) | about.ftLastWriteTime.dwLowDateTime};
}

// --seed: program 1's pipeline told apart from program 0's, which all programs share otherwise (the same SPIR-V, the
// same layout): the same SPIR-V with a 32-wide workgroup instead of 64, which no driver can take for the other.
bool narrow_workgroup(const std::filesystem::path& path) {
    std::vector<char> bytes = read_bytes(path);
    if (bytes.size() < 20 || bytes.size() % 4) return false;
    std::vector<std::uint32_t> words(bytes.size() / 4);
    std::memcpy(words.data(), bytes.data(), bytes.size());
    if (words[0] != 0x07230203u) return false;
    bool narrowed = false;
    for (std::size_t i = 5; i < words.size();) {
        const std::uint32_t count = words[i] >> 16, opcode = words[i] & 0xffffu;
        if (!count || count > words.size() - i) return false;
        // OpExecutionMode %main LocalSize 64 1 1
        if (opcode == 16 && count == 6 && words[i + 2] == 17 && words[i + 3] == 64) {
            words[i + 3] = 32;
            narrowed = true;
        }
        i += count;
    }
    if (!narrowed) return false;
    std::memcpy(bytes.data(), words.data(), bytes.size());
    return write_bytes(path, bytes);
}

// --seed: pipeline_cache.session.bin beside an offline pipeline_cache.bin, and the record of which seed it was
// written beside (pipeline_cache.session.seed). The journal is off, so a round's compute builds go into the shared
// cache and the session file is their only copy. The seed holds a real pipeline, program 0's, made by a first round
// with no seed; program 1 (a 32-wide workgroup, narrow_workgroup) is in no seed. Each round opens a device, builds
// the PSOs, tears the device down, which writes the session file, and reports the device's merges, the blobs it
// created caches from and how many builds were cache hits (pipeline creation feedback). A session file loaded alone
// that lacked the seed's pipeline would miss program 0.
int seeded_session(vktest::Device& d, const std::filesystem::path& library, const std::filesystem::path& cache_dir) {
    namespace store = leonida::pipeline_cache_store;
    const auto seed_file = cache_dir / "pipeline_cache.bin";
    const auto session_file = cache_dir / "pipeline_cache.session.bin";
    const auto record_file = cache_dir / "pipeline_cache.session.seed";
    std::error_code ec;
    std::filesystem::create_directories(cache_dir, ec);
    for (const auto& file : {seed_file, session_file, record_file}) std::filesystem::remove(file, ec);
    using Sizes = std::vector<std::size_t>;

    struct Result {
        unsigned merges = 0, builds = 0, hits = 0;
        Sizes loads;                               // initialDataSize of each cache the device created from a blob
        std::vector<char> session;                 // the session file after teardown; empty when there is none
        bool recorded = false;                     // a pipeline_cache.session.seed after teardown
        std::optional<store::SessionSeed> record;  // and what it says
        unsigned misses() const { return builds - hits; }
    };
    const auto round = [&](const char* name, unsigned programs, Result& result) -> int {
        cache_calls.reset();
        Session s;
        s.count_cache_calls = true;
        if (s.open(d, library, programs)) return 1;
        EXPECT(drain(PipelineCompiler::instance()));
        for (ID3D12PipelineState* pso : s.psos) EXPECT(built(pso));
        s.release();
        result.merges = cache_calls.merges.load();
        result.builds = cache_calls.builds.load();
        result.hits = cache_calls.hits.load();
        {
            std::lock_guard lock(cache_calls.mutex);
            result.loads = cache_calls.loads;
        }
        result.session = std::filesystem::exists(session_file) ? read_bytes(session_file) : std::vector<char>{};
        result.recorded = std::filesystem::exists(record_file);
        if (result.recorded) {
            const auto text = read_bytes(record_file);
            result.record = store::parse_session_seed(std::string_view(text.data(), text.size()));
        }
        std::string loads;
        for (const std::size_t bytes : result.loads) loads += (loads.empty() ? "" : ", ") + std::to_string(bytes);
        std::printf("seed: %s: %u merge(s), caches created from [%s] bytes, %u of %u compute build(s) a cache hit; "
                    "then a %zu-byte session file, %s\n", name, result.merges, loads.c_str(), result.hits,
                    result.builds, result.session.size(),
                    !result.recorded ? "no record" : result.record ? "a record" : "an UNREADABLE record");
        EXPECT(result.builds >= programs);
        return 0;
    };
    const auto records = [&](const Result& result) {
        return result.record && result.record->seed == stamp_of(seed_file) &&
               result.record->session_bytes == result.session.size();
    };

    // 0. The seed: program 0 built with no seed and no session file, and the session file that teardown writes moved
    // to pipeline_cache.bin.
    Result making;
    if (round("making the seed (no seed, no session file, program 0)", 1, making)) return 1;
    EXPECT(making.merges == 0 && making.loads.empty() && !making.recorded);
    EXPECT(making.session.size() > store::header_bytes);
    std::filesystem::rename(session_file, seed_file, ec);
    EXPECT(!ec);
    const std::vector<char> seed = read_bytes(seed_file);
    const std::size_t seed_bytes = seed.size();
    EXPECT(seed_bytes == making.session.size());
    std::printf("seed: the seed is %zu bytes, program 0's pipeline\n", seed_bytes);

    // 1. The seed alone: loaded; program 0 a hit from it, program 1 built; the session file written with a record of
    // the seed. A driver that never reports a hit to pipeline creation feedback here gets the size checks instead.
    Result first;
    if (round("seed, no session file", 2, first)) return 1;
    EXPECT(first.merges == 0);
    EXPECT(first.loads == Sizes{seed_bytes});
    const bool feedback = first.hits >= 1;
    if (!feedback)
        std::printf("seed: the driver reported no cache hit for the seed's pipeline; hits not checked, session file "
                    "sizes instead\n");
    EXPECT(!feedback || first.misses() == 1);  // program 1 only
    EXPECT(first.session.size() > seed_bytes);

    // 2. Relaunch: the session file, which holds the seed, is the cache. The seed is not read, nothing is merged, and
    // both pipelines are cache hits: program 0's, which came from the seed, and program 1's from round 1.
    Result direct;
    if (round("relaunch", 2, direct)) return 1;
    EXPECT(direct.merges == 0);  // before pipeline_cache.session.seed: the session file merged into the seed, 1
    EXPECT(records(first));
    EXPECT(direct.loads == Sizes{first.session.size()});
    EXPECT(feedback ? direct.misses() == 0 : direct.session.size() >= first.session.size());
    EXPECT(records(direct));

    // 3. The seed touched (a new corpus build): the seed is loaded and the session file merged, as before; the record
    // then names the new seed, and the next launch loads the session file alone again.
    std::filesystem::last_write_time(seed_file, std::filesystem::last_write_time(seed_file) + std::chrono::hours(1), ec);
    EXPECT(!ec && stamp_of(seed_file) != direct.record->seed);
    Result touched;
    if (round("the seed touched", 2, touched)) return 1;
    EXPECT(touched.merges == 1);
    EXPECT(touched.loads == (Sizes{seed_bytes, direct.session.size()}));
    EXPECT(!feedback || touched.misses() == 0);
    EXPECT(records(touched));
    Result again;
    if (round("relaunch after the touch", 2, again)) return 1;
    EXPECT(again.merges == 0 && again.loads == Sizes{touched.session.size()});
    EXPECT(!feedback || again.misses() == 0);
    EXPECT(records(again));

    // 4. LEONIDA_PIPELINE_CACHE_BYTES=0 with a record that matches: the seed alone (program 1, which only the session
    // file has, built again), the session file neither read nor written, and the record left as it was.
    const auto session_before = stamp_of(session_file);
    const auto record_before = stamp_of(record_file);
    const auto record_text = read_bytes(record_file);
    _putenv_s("LEONIDA_PIPELINE_CACHE_BYTES", "0");
    Result off;
    int failed = round("LEONIDA_PIPELINE_CACHE_BYTES=0", 2, off);
    _putenv_s("LEONIDA_PIPELINE_CACHE_BYTES", "67108864");
    if (failed) return 1;
    EXPECT(off.merges == 0 && off.loads == Sizes{seed_bytes});
    EXPECT(!feedback || off.misses() == 1);
    EXPECT(off.session == again.session && stamp_of(session_file) == session_before);
    EXPECT(read_bytes(record_file) == record_text && stamp_of(record_file) == record_before);

    // 5. LEONIDA_SESSION_CACHE_DIRECT_SEED=0 with a record that matches: the seed loaded and the session file merged,
    // as before, and the record written.
    _putenv_s("LEONIDA_SESSION_CACHE_DIRECT_SEED", "0");
    Result switched;
    failed = round("LEONIDA_SESSION_CACHE_DIRECT_SEED=0", 2, switched);
    _putenv_s("LEONIDA_SESSION_CACHE_DIRECT_SEED", "");
    if (failed) return 1;
    EXPECT(switched.merges == 1 && switched.loads == (Sizes{seed_bytes, again.session.size()}));
    EXPECT(!feedback || switched.misses() == 0);
    EXPECT(records(switched));

    // 6. No seed: the session file initializes the cache directly, as before, and the session file written without a
    // seed removes the record.
    std::filesystem::remove(seed_file, ec);
    Result unseeded;
    if (round("no seed", 2, unseeded)) return 1;
    EXPECT(unseeded.merges == 0 && unseeded.loads == Sizes{switched.session.size()});
    EXPECT(!feedback || unseeded.misses() == 0);
    EXPECT(!unseeded.session.empty() && !unseeded.recorded);

    // 7. The seed back, beside a session file with no record: the seed loaded and the session file merged, as before,
    // and the record written.
    EXPECT(write_bytes(seed_file, seed));
    Result unrecorded;
    if (round("seed and session file, no record", 2, unrecorded)) return 1;
    EXPECT(unrecorded.merges == 1 && unrecorded.loads == (Sizes{seed_bytes, unseeded.session.size()}));
    EXPECT(!feedback || unrecorded.misses() == 0);
    EXPECT(records(unrecorded));
    std::printf("ok\n");
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    const bool seed_mode = argc > 1 && !std::strcmp(argv[1], "--seed");
    // --seed: no journal (a build goes into the shared cache), a session file read directly when there is no seed, no
    // compile helper, no startup checkpoints and the new launch path on (the defaults but the journal, set so a
    // caller's environment cannot change which caches the rounds go through).
    if (seed_mode) {
        _putenv_s("LEONIDA_PIPELINE_JOURNAL", "0");
        _putenv_s("LEONIDA_DIRECT_SESSION_CACHE", "1");
        _putenv_s("LEONIDA_COMPILE_HELPER", "0");
        _putenv_s("LEONIDA_STARTUP_CACHE_CHECKPOINTS", "0");
        _putenv_s("LEONIDA_SESSION_CACHE_DIRECT_SEED", "");
    }
    // Lazy compute with the warm-up thread on (the defaults, set so a caller's environment cannot turn them off), and
    // asynchronous first-use builds, so the exit case's dispatch goes through queue_build.
    _putenv_s("LEONIDA_LAZY_COMPUTE", "1");
    _putenv_s("LEONIDA_COMPUTE_WARMUP", "1");
    _putenv_s("LEONIDA_ASYNC_PIPELINES", "1");
    _putenv_s("LEONIDA_STRICT_REPLAY", "0");
    const testfiles::ScratchDir scratch("leonida_compile_teardown_test");
    const auto library = scratch.path() / "spirv";
    const auto cache_dir = scratch.path() / "cache";
    std::filesystem::create_directories(library);
    _putenv_s("LEONIDA_PIPELINE_CACHE_DIR", cache_dir.string().c_str());
    _putenv_s("LEONIDA_PIPELINE_CACHE_BYTES", "67108864");

    vktest::Device d;
    const auto device_status = d.initialize();
    if (device_status == vktest::DeviceStatus::unavailable) {
        std::fprintf(stderr, "SKIP: no usable Vulkan 1.3 device\n");
        return 77;
    }
    EXPECT(device_status == vktest::DeviceStatus::ready);

    {
        std::ofstream index(library / "index.json", std::ios::binary);
        index << "{";
        for (unsigned i = 0; i < kPrograms; ++i) {
            const std::string name = "cs_teardown_" + std::to_string(i);
            EXPECT(testfiles::copy_file(vktest::shader_fixture_path(LEONIDA_TEST_CS_SPV), library / (name + ".spv")));
            const bool narrow = seed_mode && i == 1;  // a pipeline of its own (seeded_session)
            if (narrow) EXPECT(narrow_workgroup(library / (name + ".spv")));
            index << (i ? ",\n" : "") << "\"" << name << "\": {\"stage\": \"cs\", \"fssm_sha256\": \""
                  << leonida::sha256_hex(program(i)) << "\", \"bindings\": [], \"push_constant_bytes\": 144, "
                  << "\"local_size\": [" << (narrow ? 32 : 64) << ", 1, 1], \"uses_code_base\": true, "
                  << "\"wave64\": false}";
        }
        index << "}";
    }

    auto& compiler = PipelineCompiler::instance();
    EXPECT(!compiler.worker_running());
    // llvmpipe keeps no pipelines in a VkPipelineCache (measured: vkGetPipelineCacheData returns the 32-byte header
    // alone after the build), and the runtime writes no file of a bare header (save_session_pipeline_cache).
    VkPhysicalDeviceDriverProperties driver{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES};
    VkPhysicalDeviceProperties2 properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &driver};
    reinterpret_cast<PFN_vkGetPhysicalDeviceProperties2>(
        d.instance_proc(d.instance, "vkGetPhysicalDeviceProperties2"))(d.physical, &properties);
    const bool caches = driver.driverID != VK_DRIVER_ID_MESA_LLVMPIPE;
    const auto session_file = cache_dir / "pipeline_cache.session.bin";
    if (seed_mode) {
        if (!caches) {
            std::printf("SKIP: the driver keeps no pipelines in its cache\n");
            return 77;
        }
        return seeded_session(d, library, cache_dir);
    }

    // --- cache: one PSO built by the warm-up thread; the device writes the session cache at teardown -------------
    {
        Session s;
        if (s.open(d, library, 1)) return 1;
        EXPECT(compiler.worker_running());
        EXPECT(drain(compiler));  // the PSO's warm-up build has run: the device has a pipeline cache
        EXPECT(built(s.psos[0]));
        s.release();
        EXPECT(!compiler.worker_running());
        EXPECT(std::filesystem::exists(session_file) == caches);
        std::printf("cache: built in the background, session cache %s at teardown, worker joined\n",
                    caches ? "written" : "not written (the driver keeps no pipelines in its cache)");
    }
    // The driver may start threads of its own at the first pipeline build; count from here.
    const unsigned baseline = thread_count();
    EXPECT(baseline > 0);
    constexpr unsigned slack = 2;  // driver threads that come and go
    const auto threads_back = [&] { return eventually([&] { return thread_count() <= baseline + slack; }); };

    // --- queued: every warm-up build still queued behind a gate when the PSOs are released ----------------------
    {
        Gate gate(compiler);
        const auto before = compiler.stats();
        Session s;
        if (s.open(d, library, kPrograms)) return 1;
        EXPECT(compiler.stats().submitted - before.submitted == kPrograms);
        // Everything but the last PSO: each drops its own queued build.
        ID3D12PipelineState* last = s.psos.back();
        s.psos.pop_back();
        s.release();
        const auto dropped = compiler.stats().dropped - before.dropped;
        std::printf("queued: %llu of %u warm-up builds dropped at release, none run\n",
                    static_cast<unsigned long long>(dropped), kPrograms - 1);
        EXPECT(dropped == kPrograms - 1);
        // The device goes with the last PSO. The gate is a job of no device, so it is let go first: the last
        // device's release joins the worker, which must not be parked in it.
        gate.open();
        last->Release();
        EXPECT(!compiler.worker_running());
        EXPECT(threads_back());
    }

    // --- prompt: a PSO released while its real build is in flight; the device is still held ---------------------
    {
        Park park(compiler);
        Session s;
        {
            Gate gate(compiler);  // so the PSO's build cannot start before the probe knows it
            if (s.open(d, library, 1)) return 1;
            park.target = s.psos[0];
        }
        EXPECT(park.parked.wait());
        const auto before = compiler.stats();
        const long live = rt::d3d12::live_pipeline_states();
        ID3D12PipelineState* pso = s.psos[0];
        s.psos.clear();
        const auto start = Clock::now();
        pso->Release();  // the title's last reference
        const auto took = std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - start).count();
        const auto after = compiler.stats();
        Sleep(50);
        const long live_while_parked = rt::d3d12::live_pipeline_states();  // the build has not run: not deleted
        park.proceed.set();  // the build now runs on the released PSO, which the worker deletes after it
        EXPECT(drain(compiler));
        const long live_after_build = rt::d3d12::live_pipeline_states();
        std::printf("prompt: a PSO released with its build in flight returned in %lld ms; PSOs alive %ld before, %ld "
                    "while its build was parked, %ld after the build\n",
                    static_cast<long long>(took), live, live_while_parked, live_after_build);
        EXPECT(took < 250);
        EXPECT(after.deferred == before.deferred + 1 && after.waited == before.waited);
        EXPECT(live == 1 && live_while_parked == live && live_after_build == live - 1);
        s.release();
        EXPECT(!compiler.worker_running());
        EXPECT(threads_back());
    }

    // --- last device: the release that tears the device down waits for the PSO's build ---------------------------
    {
        Park park(compiler);
        Session s;
        {
            Gate gate(compiler);
            if (s.open(d, library, 1)) return 1;
            park.target = s.psos[0];
        }
        EXPECT(park.parked.wait());
        const auto before = compiler.stats();
        s.device->Release();
        s.device = nullptr;
        s.root->Release();
        s.root = nullptr;
        ID3D12PipelineState* pso = s.psos[0];
        s.psos.clear();
        std::atomic<bool> returned{false};
        std::thread title([&] {
            pso->Release();  // the PSO held the device's last reference
            returned = true;
        });
        Sleep(300);
        const bool early = returned.load();
        park.proceed.set();
        title.join();
        std::printf("last device: the device's teardown %s for the PSO's build in flight\n",
                    early ? "did NOT wait" : "waited");
        EXPECT(!early);
        EXPECT(compiler.stats().waited == before.waited + 1);
        EXPECT(!compiler.worker_running());
        EXPECT(rt::d3d12::live_pipeline_states() == 0);  // deleted after its build, before the device went
        EXPECT(threads_back());
    }

    // --- on worker: the PSO's last Release comes from inside its own build --------------------------------------
    for (const bool device_held : {true, false}) {
        Session s;
        Event released;
        std::atomic<const void*> target{nullptr};
        compiler.set_job_probe([&](const void* owner) {
            if (owner != target.load()) return;
            static_cast<ID3D12PipelineState*>(const_cast<void*>(owner))->Release();
            released.set();
        });
        const auto before = compiler.stats();
        {
            Gate gate(compiler);
            if (s.open(d, library, 1)) return 1;
            target = s.psos[0];
            s.psos.clear();  // the probe releases it
            if (!device_held) {
                s.device->Release();  // the PSO now holds the device's last reference
                s.device = nullptr;
                s.root->Release();
                s.root = nullptr;
            }
        }
        EXPECT(released.wait());
        if (device_held) {
            EXPECT(drain(compiler));  // the build ran on, then the PSO was deleted
            s.release();
        }
        // Without the device held, it is torn down on the worker after the build, and the worker lets itself go.
        EXPECT(eventually([&] { return !compiler.worker_running(); }));
        compiler.set_job_probe(nullptr);
        EXPECT(compiler.stats().deferred == before.deferred + 1);
        EXPECT(rt::d3d12::live_pipeline_states() == 0);
        EXPECT(threads_back());
        std::printf("on worker (%s): released inside its own build, deleted after it\n",
                    device_held ? "device held" : "device's last reference");
    }

    // --- two devices: one torn down while the other's builds are queued -----------------------------------------
    {
        Session a, b;
        const auto before = compiler.stats();
        {
            Gate gate(compiler);
            if (a.open(d, library, kPrograms / 2)) return 1;
            if (b.open(d, library, kPrograms / 2)) return 1;
            a.release();
            EXPECT(compiler.stats().dropped - before.dropped == kPrograms / 2);
            EXPECT(compiler.worker_running());  // b still holds it
            for (unsigned i = kPrograms / 2; i < kPrograms; ++i) EXPECT(b.add(i) == 0);
        }
        EXPECT(drain(compiler));
        for (ID3D12PipelineState* pso : b.psos) EXPECT(built(pso));
        b.release();
        EXPECT(!compiler.worker_running());
        EXPECT(threads_back());
        std::printf("two devices: %u builds of the released one dropped, all %zu of the other built\n",
                    kPrograms / 2, size_t(kPrograms / 2 + kPrograms / 2));
    }

    // --- racing: release at once, builds queued and usually one running ---------------------------------------
    {
        const auto before = compiler.stats();
        unsigned peak = 0;
        for (unsigned round = 0; round < kRacingRounds; ++round) {
            Session s;
            if (s.open(d, library, kPrograms)) return 1;
            if (round % 3 == 1) Sleep(round);  // let some rounds get a build or more under way first
            s.release(round & 1);
            EXPECT(!compiler.worker_running());
            EXPECT(rt::d3d12::live_pipeline_states() == 0);  // every PSO deleted, deferred ones included
            const unsigned now = thread_count();
            if (now > peak) peak = now;
            EXPECT(threads_back());
        }
        const auto after = compiler.stats();
        std::printf("racing: %u rounds of %u PSOs; %llu builds queued, %llu dropped, %llu PSO releases left a running "
                    "build behind, %llu device teardowns waited for one; threads %u at the start, at most %u\n",
                    kRacingRounds, kPrograms, static_cast<unsigned long long>(after.submitted - before.submitted),
                    static_cast<unsigned long long>(after.dropped - before.dropped),
                    static_cast<unsigned long long>(after.deferred - before.deferred),
                    static_cast<unsigned long long>(after.waited - before.waited), baseline, peak);
        // How many builds were dropped or left running depends on timing (tiny programs, a warm cache): reported,
        // not asserted. The queued, prompt and last device cases hold those states deterministically.
        EXPECT(after.submitted - before.submitted == kRacingRounds * kPrograms);
    }

    // --- restart: a device made after the worker stopped gets a new worker, and its builds run ------------------
    {
        EXPECT(!compiler.worker_running());
        Session s;
        if (s.open(d, library, kPrograms)) return 1;
        EXPECT(compiler.worker_running());
        EXPECT(drain(compiler));
        for (ID3D12PipelineState* pso : s.psos) EXPECT(built(pso));
        s.release();
        EXPECT(!compiler.worker_running());
        std::printf("restart: a new device after a stop got a new worker; all %u builds ran\n", kPrograms);
    }

    // --- exit: stop_pipeline_compiles with a device alive --------------------------------------------------------
    {
        Session s;
        if (s.open(d, library, kPrograms)) return 1;
        EXPECT(rt::d3d12::stop_pipeline_compiles(60000));
        EXPECT(!compiler.worker_running());
        const auto before = compiler.stats();
        static const int late = 0;
        EXPECT(!compiler.submit(&late, nullptr, 0, [] {}));
        EXPECT(compiler.stats().refused == before.refused + 1);
        // A PSO made now queues no warm-up, and its first dispatch builds the pipeline in place and runs.
        const unsigned fresh = 0;
        EXPECT(s.add(fresh) == 0);
        ID3D12PipelineState* pso = s.psos.back();
        EXPECT(compiler.stats().submitted == before.submitted && !built(pso));
        if (s.dispatch(pso, fresh)) return 1;
        EXPECT(built(pso));
        EXPECT(compiler.stats().submitted == before.submitted);
        // The process ends with the device alive (a title never releases it): the exit save writes the session
        // cache now, not at a teardown that will not come, and only once.
        std::error_code ec;
        std::filesystem::remove(session_file, ec);
        EXPECT(!std::filesystem::exists(session_file));
        EXPECT(rt::d3d12::save_pipeline_caches_at_exit(1000));
        EXPECT(std::filesystem::exists(session_file) == caches);
        EXPECT(!rt::d3d12::save_pipeline_caches_at_exit(1000));
        s.release();  // the device's teardown after the exit stop: nothing left to cancel or join
        EXPECT(!compiler.worker_running());
        EXPECT(rt::d3d12::live_pipeline_states() == 0);  // no PSO of any case leaked
        std::printf("exit: queue dropped, later builds refused, a dispatch built its pipeline in place and ran, "
                    "session cache %s with the device alive, once\n", caches ? "written" : "not written (llvmpipe)");
    }
    std::printf("ok\n");
    return 0;
}
