// Semaphore lifetimes across fence and queue teardown (src/vulkan_backend.h, src/vulkan_device_fault.h). No shaders,
// no game data.
//   1. A fence released right after an unheld Signal: the queue keeps the fence, and so its timeline semaphore,
//      until the GPU has run the signal (destroying a semaphore a pending submission signals is undefined,
//      VUID-vkDestroySemaphore-semaphore-05149). The Signal is kept pending on the GPU by a wait on a semaphore the
//      host signals later.
//   2. The queue whose submission uploaded the bindless table is destroyed, then another queue submits: that
//      submission must not wait on the destroyed queue's progress semaphore.
//   3. With the device-fault diagnostics on, the submit ring forgets both semaphores before they are destroyed: its
//      progress watch (once a second) and a dump after the teardown never read the counter of a destroyed one.
// The driver calls are intercepted, so 2 and 3 are checked with or without the validation layer; set
// LEONIDA_VK_VALIDATION=1 to also run under VK_LAYER_KHRONOS_validation (any validation error fails the test).
#include "vk_test_device.h"
#include "../src/vulkan_backend.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <set>
#include <thread>
#include <utility>
#include <vector>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

namespace {

using namespace leonida;

namespace driver {
PFN_vkGetDeviceProcAddr real_get = nullptr;
PFN_vkQueueSubmit real_submit = nullptr;
PFN_vkCreateSemaphore real_create = nullptr;
PFN_vkDestroySemaphore real_destroy = nullptr;
PFN_vkGetSemaphoreCounterValue real_counter = nullptr;

std::mutex mutex;
std::set<VkSemaphore> destroyed;           // destroyed and not made again
std::vector<VkSemaphore> last_signals;     // of the last vkQueueSubmit
std::vector<VkSemaphore> waits_seen;       // every wait semaphore submitted since the last reset
std::atomic<unsigned> dead_waits{0};       // submissions that waited on a destroyed semaphore
std::atomic<unsigned> dead_reads{0};       // counter reads of a destroyed semaphore
std::atomic<unsigned> live_reads{0};

VKAPI_ATTR VkResult VKAPI_CALL submit(VkQueue queue, std::uint32_t count, const VkSubmitInfo* submits, VkFence fence) {
    {
        std::lock_guard lock(mutex);
        last_signals.clear();
        for (std::uint32_t i = 0; i < count; ++i) {
            for (std::uint32_t w = 0; w < submits[i].waitSemaphoreCount; ++w) {
                const VkSemaphore s = submits[i].pWaitSemaphores[w];
                waits_seen.push_back(s);
                if (destroyed.count(s)) dead_waits.fetch_add(1);
            }
            for (std::uint32_t s = 0; s < submits[i].signalSemaphoreCount; ++s)
                last_signals.push_back(submits[i].pSignalSemaphores[s]);
        }
    }
    return real_submit(queue, count, submits, fence);
}

VKAPI_ATTR VkResult VKAPI_CALL create(VkDevice device, const VkSemaphoreCreateInfo* info,
                                      const VkAllocationCallbacks* alloc, VkSemaphore* out) {
    const VkResult result = real_create(device, info, alloc, out);
    if (result == VK_SUCCESS) {
        std::lock_guard lock(mutex);
        destroyed.erase(*out);  // a handle value the driver hands out again is alive again
    }
    return result;
}

VKAPI_ATTR void VKAPI_CALL destroy(VkDevice device, VkSemaphore semaphore, const VkAllocationCallbacks* alloc) {
    if (semaphore) {
        std::lock_guard lock(mutex);
        destroyed.insert(semaphore);
    }
    real_destroy(device, semaphore, alloc);
}

VKAPI_ATTR VkResult VKAPI_CALL counter(VkDevice device, VkSemaphore semaphore, std::uint64_t* value) {
    {
        std::lock_guard lock(mutex);
        if (destroyed.count(semaphore)) {
            dead_reads.fetch_add(1);
            *value = 0;
            return VK_ERROR_UNKNOWN;  // never reaches the driver with a dead handle
        }
    }
    live_reads.fetch_add(1);
    return real_counter(device, semaphore, value);
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL get(VkDevice device, const char* name) {
    if (!std::strcmp(name, "vkQueueSubmit")) return reinterpret_cast<PFN_vkVoidFunction>(submit);
    if (!std::strcmp(name, "vkCreateSemaphore")) return reinterpret_cast<PFN_vkVoidFunction>(create);
    if (!std::strcmp(name, "vkDestroySemaphore")) return reinterpret_cast<PFN_vkVoidFunction>(destroy);
    if (!std::strcmp(name, "vkGetSemaphoreCounterValue") || !std::strcmp(name, "vkGetSemaphoreCounterValueKHR"))
        return reinterpret_cast<PFN_vkVoidFunction>(counter);
    return real_get(device, name);
}
}  // namespace driver

// A timeline semaphore the test signals from the host, unknown to the queues' holds (so a wait on it is not held on
// the CPU: it is submitted, and the GPU waits). Like the runtime's own owners (~VulkanFence, ~VulkanQueue) it tells
// the submit ring before it goes.
struct HostGate {
    const vktest::Device& d;
    DeviceFaultDiagnostics* ring;
    VkSemaphore semaphore = VK_NULL_HANDLE;
    HostGate(const vktest::Device& device, DeviceFaultDiagnostics* diagnostics) : d(device), ring(diagnostics) {
        VkSemaphoreTypeCreateInfo type{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
        type.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
        VkSemaphoreCreateInfo info{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, &type};
        driver::create(d.device, &info, nullptr, &semaphore);
    }
    ~HostGate() {
        if (!semaphore) return;
        if (ring) ring->forget(semaphore);
        driver::destroy(d.device, semaphore, nullptr);
    }
    bool open(std::uint64_t value) const {
        VkSemaphoreSignalInfo signal{VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO};
        signal.semaphore = semaphore;
        signal.value = value;
        return d.fn<PFN_vkSignalSemaphore>("vkSignalSemaphore")(d.device, &signal) == VK_SUCCESS;
    }
};

int run(vktest::Device& d) {
    driver::real_get = d.device_proc;
    driver::real_submit = d.fn<PFN_vkQueueSubmit>("vkQueueSubmit");
    driver::real_create = d.fn<PFN_vkCreateSemaphore>("vkCreateSemaphore");
    driver::real_destroy = d.fn<PFN_vkDestroySemaphore>("vkDestroySemaphore");
    driver::real_counter = d.fn<PFN_vkGetSemaphoreCounterValue>("vkGetSemaphoreCounterValue");
    EXPECT(driver::real_submit && driver::real_create && driver::real_destroy && driver::real_counter);

    VulkanContext context(driver::get, d.device, true);
    EXPECT(context.loaded());
    context.configure_device_fault(driver::get, DeviceFaultExtensions{});  // the submit ring and its progress watch
    EXPECT(context.device_fault());
    bool arm_upload = false;
    context.set_bindless_prepare([&](VkCommandBuffer) {
        if (std::exchange(arm_upload, false)) context.note_bindless_upload();  // as a replacing bindless flush does
        return true;
    });

    void* map = nullptr;
    constexpr VkDeviceSize kSize = 4096;
    const VkBufferUsageFlags usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VkBuffer a = d.buffer(kSize, usage, &map), b = d.buffer(kSize, usage, &map);
    EXPECT(a && b);
    ResourceBinding binding;
    binding.size = kSize;
    binding.buffer = a;
    context.add_resource(1, binding);
    binding.buffer = b;
    context.add_resource(2, binding);
    CommandList copy;
    EXPECT(copy.copy_buffer({2, 0, 1, 0, kSize}));
    EXPECT(copy.close());
    const CommandList* lists[] = {&copy};

    auto stay = std::make_unique<VulkanQueue>(context, d.queue, d.family);
    EXPECT(stay->ok());

    // 1. Release a fence right after an unheld Signal that the GPU cannot run yet.
    {
        HostGate gate(d, context.device_fault());
        EXPECT(gate.semaphore);
        auto fence = std::make_shared<VulkanFence>(context, 0);
        const VkSemaphore fence_semaphore = fence->semaphore();
        EXPECT(fence_semaphore);
        std::weak_ptr<VulkanFence> watch = fence;
        stay->wait(gate.semaphore, 1);  // the next submission waits for the host on the GPU
        EXPECT(stay->signal(fence_semaphore, 1, [&fence] { return std::shared_ptr<void>(fence); }));
        fence.reset();  // the title's last Release
        {
            std::lock_guard lock(driver::mutex);
            EXPECT(!driver::destroyed.count(fence_semaphore));
        }
        EXPECT(!watch.expired());  // the queue keeps it while the Signal is pending
        EXPECT(gate.open(1));
        // Each execute drains the pins whose submissions are done.
        bool released = false;
        for (int i = 0; i < 200 && !released; ++i) {
            context.wait_retire_ticket(context.submitted_retire_ticket());
            EXPECT(stay->execute(lists) == ReplayResult::success);
            released = watch.expired();
            if (!released) std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        EXPECT(released);  // and lets it go once the GPU has run it
        std::lock_guard lock(driver::mutex);
        EXPECT(driver::destroyed.count(fence_semaphore));
        std::printf("fence released after an unheld Signal: kept until the signal ran, then destroyed\n");
    }

    // 2. The queue that uploaded the bindless table goes; another queue submits next.
    VkSemaphore uploader_progress = VK_NULL_HANDLE;
    {
        auto uploader = std::make_unique<VulkanQueue>(context, d.queue, d.family);
        EXPECT(uploader->ok());
        arm_upload = true;
        EXPECT(uploader->execute(lists) == ReplayResult::success);
        EXPECT(!arm_upload);
        {
            std::lock_guard lock(driver::mutex);
            EXPECT(!driver::last_signals.empty());
            uploader_progress = driver::last_signals[0];  // a queue's submission signals its progress timeline first
        }
        uploader.reset();
    }
    {
        std::lock_guard lock(driver::mutex);
        EXPECT(driver::destroyed.count(uploader_progress));
        driver::waits_seen.clear();
    }
    EXPECT(stay->execute(lists) == ReplayResult::success);
    {
        std::lock_guard lock(driver::mutex);
        EXPECT(std::find(driver::waits_seen.begin(), driver::waits_seen.end(), uploader_progress) ==
               driver::waits_seen.end());
    }
    EXPECT(driver::dead_waits.load() == 0);
    std::printf("bindless upload queue destroyed: the next submission on another queue does not wait on it\n");

    // 3. The ring: the progress watch reads counters once a second; a dump now reads every semaphore it still names.
    std::this_thread::sleep_for(std::chrono::milliseconds(2200));
    const unsigned reads_before = driver::live_reads.load();
    context.device_fault()->dump_submits();
    EXPECT(driver::live_reads.load() > reads_before);  // it did read the live ones
    EXPECT(driver::dead_reads.load() == 0);
    std::printf("submit ring: no counter read of a destroyed semaphore (%u live reads)\n", driver::live_reads.load());

    stay.reset();
    context.remove_resource(1);
    context.remove_resource(2);
    return 0;
}

}  // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    vktest::Device d;
    const auto status = d.initialize();
    if (status == vktest::DeviceStatus::unavailable) {
        std::printf("SKIP: no usable Vulkan device\n");
        return 77;
    }
    if (status != vktest::DeviceStatus::ready) return 1;
    const int rc = run(d);
    const unsigned errors = d.validation_errors();
    std::printf("Vulkan validation errors: %u%s\n", errors, d.validation ? "" : " (validation off)");
    return rc ? rc : errors ? 1 : 0;
}
