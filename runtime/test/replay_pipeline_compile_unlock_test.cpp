// VulkanQueue::execute_now() releases resource_mutex_ and tracker_mutex_ while compiling unbuilt pipelines under
// pipeline_mutex_ alone, so add_resource()/remove_resource() and other queues do not stall behind shader compiles,
// remove_pipeline() still waits until compilation finishes, failed non-strict dispatches build only once, and
// pass-1 re-preparation observes resource updates made while resource_mutex_ was unlocked. No GPU or game data.
#include "../src/vulkan_backend.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <mutex>
#include <string_view>
#include <thread>
#include <vector>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

namespace {

using namespace leonida;

template<class T>
T handle(std::uintptr_t n) {
    return reinterpret_cast<T>(n);
}

VulkanDeviceFunctions driver{};
std::atomic<std::uintptr_t> next_handle{10};
std::mutex record_mutex;
std::vector<std::pair<VkBuffer, VkBuffer>> recorded_copies;

PFN_vkVoidFunction VKAPI_CALL mock_get_device_proc(VkDevice, const char* name) {
#define MOCK_DEVICE_FUNCTION(n) \
    if (std::string_view(name) == #n) return reinterpret_cast<PFN_vkVoidFunction>(driver.n);
    LEONIDA_VK_DEVICE_FUNCTIONS(MOCK_DEVICE_FUNCTION)
#undef MOCK_DEVICE_FUNCTION
    return nullptr;
}

void init_mock_driver() {
    driver = {};
    driver.vkCreateCommandPool = +[](VkDevice, const VkCommandPoolCreateInfo*, const VkAllocationCallbacks*,
                                     VkCommandPool* out) {
        *out = handle<VkCommandPool>(next_handle.fetch_add(1));
        return VK_SUCCESS;
    };
    driver.vkDestroyCommandPool = +[](VkDevice, VkCommandPool, const VkAllocationCallbacks*) {};
    driver.vkCreateSemaphore = +[](VkDevice, const VkSemaphoreCreateInfo*, const VkAllocationCallbacks*,
                                   VkSemaphore* out) {
        *out = handle<VkSemaphore>(next_handle.fetch_add(1));
        return VK_SUCCESS;
    };
    driver.vkDestroySemaphore = +[](VkDevice, VkSemaphore, const VkAllocationCallbacks*) {};
    driver.vkGetSemaphoreCounterValue = +[](VkDevice, VkSemaphore, std::uint64_t* value) {
        *value = UINT64_MAX;
        return VK_SUCCESS;
    };
    driver.vkQueueWaitIdle = +[](VkQueue) { return VK_SUCCESS; };
    driver.vkAllocateCommandBuffers = +[](VkDevice, const VkCommandBufferAllocateInfo* info, VkCommandBuffer* out) {
        for (std::uint32_t i = 0; i < info->commandBufferCount; ++i)
            out[i] = handle<VkCommandBuffer>(next_handle.fetch_add(1));
        return VK_SUCCESS;
    };
    driver.vkResetCommandBuffer = +[](VkCommandBuffer, VkCommandBufferResetFlags) { return VK_SUCCESS; };
    driver.vkBeginCommandBuffer = +[](VkCommandBuffer, const VkCommandBufferBeginInfo*) { return VK_SUCCESS; };
    driver.vkEndCommandBuffer = +[](VkCommandBuffer) { return VK_SUCCESS; };
    driver.vkQueueSubmit = +[](VkQueue, std::uint32_t, const VkSubmitInfo*, VkFence) { return VK_SUCCESS; };
    driver.vkCmdPipelineBarrier = +[](VkCommandBuffer, VkPipelineStageFlags, VkPipelineStageFlags, VkDependencyFlags,
                                      std::uint32_t, const VkMemoryBarrier*, std::uint32_t,
                                      const VkBufferMemoryBarrier*, std::uint32_t, const VkImageMemoryBarrier*) {};
    driver.vkCmdBindPipeline = +[](VkCommandBuffer, VkPipelineBindPoint, VkPipeline) {};
    driver.vkCmdPushConstants = +[](VkCommandBuffer, VkPipelineLayout, VkShaderStageFlags, std::uint32_t,
                                    std::uint32_t, const void*) {};
    driver.vkCmdDispatch = +[](VkCommandBuffer, std::uint32_t, std::uint32_t, std::uint32_t) {};
    driver.vkCmdCopyBuffer = +[](VkCommandBuffer, VkBuffer src, VkBuffer dst, std::uint32_t, const VkBufferCopy*) {
        std::lock_guard lock(record_mutex);
        recorded_copies.emplace_back(src, dst);
    };
    driver.vkCmdCopyBufferToImage = +[](VkCommandBuffer, VkBuffer, VkImage, VkImageLayout, std::uint32_t,
                                        const VkBufferImageCopy*) {};
    driver.vkCmdCopyImageToBuffer = +[](VkCommandBuffer, VkImage, VkImageLayout, VkBuffer, std::uint32_t,
                                        const VkBufferImageCopy*) {};
    driver.vkCmdCopyImage = +[](VkCommandBuffer, VkImage, VkImageLayout, VkImage, VkImageLayout, std::uint32_t,
                                const VkImageCopy*) {};
}

struct SlowPipeline {
    std::mutex mutex;
    std::condition_variable cv;
    bool entered = false;
    bool allow_finish = false;
    bool fail = false;
    std::atomic<int> builds{0};
    VkPipeline handle_out = VK_NULL_HANDLE;

    VkPipeline build() {
        builds.fetch_add(1, std::memory_order_relaxed);
        {
            std::unique_lock lock(mutex);
            entered = true;
            cv.notify_all();
            cv.wait(lock, [this] { return allow_finish; });
            if (!fail) handle_out = handle<VkPipeline>(0x7001);
            return handle_out;
        }
    }

    VkPipeline find() {
        std::unique_lock lock(mutex, std::try_to_lock);
        return lock.owns_lock() ? handle_out : VK_NULL_HANDLE;
    }

    void release() {
        std::lock_guard lock(mutex);
        allow_finish = true;
        cv.notify_all();
    }
};

// Exercises concurrent add_resource/remove_resource, a second queue submission, resource replacement observed on
// pass-1 re-prepare, and remove_pipeline() waiting on pipeline_mutex_ while a slow compile runs.
int test_unlocks_resources_during_compile(bool with_workers) {
    {
        std::lock_guard lock(record_mutex);
        recorded_copies.clear();
    }
    VulkanContext context(mock_get_device_proc, handle<VkDevice>(1), false);
    if (with_workers) context.set_pipeline_workers(std::make_unique<PipelineWorkers>(2));

    ResourceBinding src{};
    src.buffer = handle<VkBuffer>(101);
    src.size = 64;
    ResourceBinding dst_old{};
    dst_old.buffer = handle<VkBuffer>(102);
    dst_old.size = 64;
    ResourceBinding dst_new{};
    dst_new.buffer = handle<VkBuffer>(202);
    dst_new.size = 64;
    context.add_resource(1, src);
    context.add_resource(2, dst_old);

    SlowPipeline slow;
    constexpr std::uintptr_t slow_id = 0x100;
    constexpr std::uintptr_t fast_id = 0x200;

    ComputePipeline slow_pso{};
    slow_pso.layout = handle<VkPipelineLayout>(5);
    slow_pso.build_context = &slow;
    slow_pso.build = [](void* ctx) { return static_cast<SlowPipeline*>(ctx)->build(); };
    slow_pso.built = [](void* ctx) { return static_cast<SlowPipeline*>(ctx)->find(); };
    context.add_pipeline(slow_id, slow_pso);

    ComputePipeline fast_pso{};
    fast_pso.layout = handle<VkPipelineLayout>(5);
    fast_pso.pipeline = handle<VkPipeline>(0x8001);
    context.add_pipeline(fast_id, fast_pso);

    VulkanQueue queue_a(context, handle<VkQueue>(1), 0, VulkanQueue::QueueKind::compute);
    VulkanQueue queue_b(context, handle<VkQueue>(2), 0, VulkanQueue::QueueKind::compute);
    EXPECT(queue_a.ok() && queue_b.ok());

    CommandList list_a;
    EXPECT(list_a.dispatch({slow_id, {}, 1, 1, 1}));
    EXPECT(list_a.copy_buffer({2, 0, 1, 0, 64}));
    EXPECT(list_a.close());

    CommandList list_b;
    EXPECT(list_b.dispatch({fast_id, {}, 1, 1, 1}));
    EXPECT(list_b.close());

    ReplayResult result_a = ReplayResult::unsupported;
    std::thread thread_a([&] {
        const CommandList* ptr = &list_a;
        result_a = queue_a.execute({&ptr, 1});
    });

    // Wait for thread_a (or a PipelineWorkers thread) to enter slow.build().
    {
        std::unique_lock lock(slow.mutex);
        const bool entered = slow.cv.wait_for(lock, std::chrono::seconds(5), [&] { return slow.entered; });
        if (!entered) {
            slow.allow_finish = true;
            slow.cv.notify_all();
            lock.unlock();
            thread_a.join();
            EXPECT(false);
        }
    }

    // While compilation is paused:
    // 1. add_resource() / remove_resource() and a second queue submission must complete without blocking.
    std::atomic<bool> resources_unblocked{false};
    ReplayResult result_b = ReplayResult::unsupported;
    std::thread resource_thread([&] {
        ResourceBinding temp{};
        temp.buffer = handle<VkBuffer>(303);
        temp.size = 64;
        context.add_resource(3, temp);
        context.remove_resource(3);
        // Replace resource 2 while resource_mutex_ is unlocked so pass-1 re-prepare must bind dst_new.
        context.remove_resource(2);
        context.add_resource(2, dst_new);
        const CommandList* ptr = &list_b;
        result_b = queue_b.execute({&ptr, 1});
        resources_unblocked.store(true, std::memory_order_release);
    });

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (!resources_unblocked.load(std::memory_order_acquire) && std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(5));

    if (!resources_unblocked.load(std::memory_order_acquire)) {
        slow.release();
        resource_thread.join();
        thread_a.join();
        EXPECT(false);
    }
    resource_thread.join();
    EXPECT(result_b == ReplayResult::success);

    // 2. remove_pipeline() must wait on pipeline_mutex_ until compilation finishes (protecting ComputePipeline*).
    std::atomic<bool> remove_started{false};
    std::atomic<bool> remove_finished{false};
    std::thread remove_thread([&] {
        remove_started.store(true, std::memory_order_release);
        context.remove_pipeline(slow_id);
        remove_finished.store(true, std::memory_order_release);
    });
    while (!remove_started.load(std::memory_order_acquire))
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    const bool removed_early = remove_finished.load(std::memory_order_acquire);

    slow.release();
    thread_a.join();
    remove_thread.join();

    EXPECT(!removed_early);
    EXPECT(remove_finished.load(std::memory_order_acquire));
    EXPECT(result_a == ReplayResult::success);
    EXPECT(slow.builds.load() == 1);
    {
        std::lock_guard lock(record_mutex);
        EXPECT(recorded_copies.size() == 1);
        EXPECT(recorded_copies[0].first == src.buffer && recorded_copies[0].second == dst_new.buffer);
    }
    return 0;
}

// A non-strict failed compute dispatch must not invoke build() a second time on pass-1 re-prepare.
int test_failed_compute_dispatch_built_once() {
    VulkanContext context(mock_get_device_proc, handle<VkDevice>(1), false);
    SlowPipeline failing;
    failing.fail = true;
    failing.allow_finish = true;

    constexpr std::uintptr_t fail_id = 0x300;
    ComputePipeline pso{};
    pso.layout = handle<VkPipelineLayout>(5);
    pso.build_context = &failing;
    pso.build = [](void* ctx) { return static_cast<SlowPipeline*>(ctx)->build(); };
    pso.built = [](void* ctx) { return static_cast<SlowPipeline*>(ctx)->find(); };
    context.add_pipeline(fail_id, pso);

    VulkanQueue queue(context, handle<VkQueue>(3), 0, VulkanQueue::QueueKind::compute);
    EXPECT(queue.ok());

    CommandList list;
    EXPECT(list.dispatch({fail_id, {}, 1, 1, 1}));
    EXPECT(list.close());
    const CommandList* ptr = &list;
    const ReplayResult r = queue.execute({&ptr, 1});
    EXPECT(r == (strict_replay() ? ReplayResult::unresolved_resource : ReplayResult::success));
    EXPECT(failing.builds.load() == 1);
    return 0;
}

}  // namespace

int main() {
    init_mock_driver();
    if (test_unlocks_resources_during_compile(true) != 0) return 1;
    if (test_unlocks_resources_during_compile(false) != 0) return 1;
    if (test_failed_compute_dispatch_built_once() != 0) return 1;
    std::printf("replay_pipeline_compile_unlock_test: passed\n");
    return 0;
}
