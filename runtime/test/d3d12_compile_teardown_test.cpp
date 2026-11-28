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
// No game data: the shader is test/glsl/cs_user_data.comp, the programs are made-up FSSM bytes.
#include "shader_fixture_path.h"
#include "test_temp_dir.h"
#include "vk_test_device.h"
#include "../shims/d3d12_com.h"
#include "../shims/d3d12_compile_exit.h"
#include "../shims/d3d12_pipeline_compile.h"
#include "../shims/d3d12_xs_device.h"
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

// One D3D12 device over the test's VkDevice, the compute test's root signature (two constants, a root UAV, a CBV
// table) and PSOs. The title's references, released in any order with release().
struct Session {
    ID3D12Device* device = nullptr;
    ID3D12RootSignature* root = nullptr;
    std::vector<ID3D12PipelineState*> psos;

    int open(vktest::Device& d, const std::filesystem::path& library, unsigned count) {
        rt::d3d12::VulkanDeviceInfo info{};
        info.get_instance_proc = d.instance_proc;
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

}  // namespace

int main() {
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
            index << (i ? ",\n" : "") << "\"" << name << "\": {\"stage\": \"cs\", \"fssm_sha256\": \""
                  << leonida::sha256_hex(program(i)) << "\", \"bindings\": [], \"push_constant_bytes\": 144, "
                  << "\"local_size\": [64, 1, 1], \"uses_code_base\": true, \"wave64\": false}";
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
