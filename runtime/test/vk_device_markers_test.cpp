// The AMD breadcrumbs of the device-fault report (src/vulkan_device_markers.h, LEONIDA_DEVICE_MARKERS=1) on a real
// device: dispatches recorded through CheckpointRecorder with the markers on, submitted and waited for, leave their
// queue's started and finished markers at the last dispatch's seq, and the report calls none of them hung. Two
// submissions, so the second continues the first one's numbering. The shader is an empty hand-written compute
// program; nothing is meant to fault or hang. SKIP (77) without a Vulkan device or without VK_AMD_buffer_marker.
// LEONIDA_VK_VALIDATION=1 runs it under VK_LAYER_KHRONOS_validation (any validation error fails the test).
#include "vk_test_device.h"
#include "../src/vulkan_device_fault.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <vector>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

namespace {

using namespace leonida;

// OpCapability Shader; OpMemoryModel Logical GLSL450; OpEntryPoint GLCompute %1 "main"; LocalSize 1 1 1;
// void main() {}
constexpr std::uint32_t empty_compute[] = {
    0x07230203u, 0x00010000u, 0u, 5u, 0u,
    (2u << 16) | 17u, 1u,                                   // OpCapability Shader
    (3u << 16) | 14u, 0u, 1u,                               // OpMemoryModel Logical GLSL450
    (5u << 16) | 15u, 5u, 1u, 0x6e69616du, 0u,              // OpEntryPoint GLCompute %1 "main"
    (6u << 16) | 16u, 1u, 17u, 1u, 1u, 1u,                  // OpExecutionMode %1 LocalSize 1 1 1
    (2u << 16) | 19u, 2u,                                   // %2 = OpTypeVoid
    (3u << 16) | 33u, 3u, 2u,                               // %3 = OpTypeFunction %2
    (5u << 16) | 54u, 2u, 1u, 0u, 3u,                       // %1 = OpFunction %2 None %3
    (2u << 16) | 248u, 4u,                                  // %4 = OpLabel
    (1u << 16) | 253u,                                      // OpReturn
    (1u << 16) | 56u,                                       // OpFunctionEnd
};

struct Pipeline {
    const vktest::Device& d;
    VkShaderModule module = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;

    explicit Pipeline(const vktest::Device& device) : d(device) {
        VkShaderModuleCreateInfo code{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        code.codeSize = sizeof(empty_compute);
        code.pCode = empty_compute;
        if (d.fn<PFN_vkCreateShaderModule>("vkCreateShaderModule")(d.device, &code, nullptr, &module) != VK_SUCCESS)
            return;
        VkPipelineLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        if (d.fn<PFN_vkCreatePipelineLayout>("vkCreatePipelineLayout")(d.device, &layout_info, nullptr, &layout) !=
            VK_SUCCESS)
            return;
        VkComputePipelineCreateInfo info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        info.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        info.stage.module = module;
        info.stage.pName = "main";
        info.layout = layout;
        d.fn<PFN_vkCreateComputePipelines>("vkCreateComputePipelines")(d.device, VK_NULL_HANDLE, 1, &info, nullptr,
                                                                       &pipeline);
    }
    ~Pipeline() {
        if (pipeline) d.fn<PFN_vkDestroyPipeline>("vkDestroyPipeline")(d.device, pipeline, nullptr);
        if (layout) d.fn<PFN_vkDestroyPipelineLayout>("vkDestroyPipelineLayout")(d.device, layout, nullptr);
        if (module) d.fn<PFN_vkDestroyShaderModule>("vkDestroyShaderModule")(d.device, module, nullptr);
    }
};

int run(vktest::Device& d) {
    VkPhysicalDeviceMemoryProperties memory{};
    d.instance_fn<PFN_vkGetPhysicalDeviceMemoryProperties>("vkGetPhysicalDeviceMemoryProperties")(d.physical, &memory);
    DeviceFaultExtensions enabled;
    enabled.markers = true;
    // The runtime's devices always have synchronization2 (vulkan13_features.h), and so does this one: the marker
    // writes take the vkCmdWriteBufferMarker2AMD path, as in a game run.
    DeviceFaultDiagnostics diagnostics(d.device_proc, d.device, enabled, &memory, true);
    EXPECT(diagnostics.markers_on());
    diagnostics.add_queue(d.queue);
    EXPECT(diagnostics.marker_slot(d.queue).started == 0 && diagnostics.marker_slot(d.queue).finished == 0);

    Pipeline pipeline(d);
    EXPECT(pipeline.pipeline);
    const auto bind = d.fn<PFN_vkCmdBindPipeline>("vkCmdBindPipeline");
    const auto dispatch = d.fn<PFN_vkCmdDispatch>("vkCmdDispatch");
    const auto pipeline_id = reinterpret_cast<std::uintptr_t>(&pipeline);
    diagnostics.pipeline_names[pipeline_id] = {"cs_empty_test"};
    const std::array<std::uint32_t, 32> roots{};

    std::uint64_t last_seq = 0;
    for (unsigned round = 0; round < 2; ++round) {
        const unsigned dispatches = round == 0 ? 5 : 3;
        auto submission = diagnostics.begin_submission(diagnostics.next_execute_call(), d.queue, "direct", 1);
        EXPECT(submission.markers);
        const CheckpointRecorder recorder{nullptr, submission, diagnostics.pipeline_names, 0};
        const bool ran = d.run([&](VkCommandBuffer cb) {
            for (unsigned i = 0; i < dispatches; ++i) {
                recorder.dispatch(cb, i, pipeline.pipeline, pipeline_id, roots, {1 + i, 1, 1});
                bind(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline);
                dispatch(cb, 1 + i, 1, 1);
                recorder.dispatched(cb);
            }
            return true;
        });
        EXPECT(ran);
        EXPECT(submission.dispatches.size() == dispatches);
        EXPECT(submission.dispatches.front().seq == last_seq + 1);
        last_seq = submission.dispatches.back().seq;
        EXPECT(last_seq == (round == 0 ? 5u : 8u));
        diagnostics.publish(std::move(submission));
        // vkQueueWaitIdle has returned: every marker the GPU wrote is in the coherent buffer.
        const MarkerSlot slot = diagnostics.marker_slot(d.queue);
        std::printf("round %u: started=%u finished=%u, last seq %llu\n", round, slot.started, slot.finished,
                    static_cast<unsigned long long>(last_seq));
        EXPECT(slot.started == last_seq && slot.finished == last_seq);
        EXPECT(diagnostics.report_markers(false).empty());
    }
    return 0;
}

}  // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    vktest::Device d;
    d.request_buffer_marker = true;
    const auto status = d.initialize();
    if (status == vktest::DeviceStatus::unavailable) {
        std::printf("SKIP: no usable Vulkan device\n");
        return 77;
    }
    if (status != vktest::DeviceStatus::ready) return 1;
    if (!d.buffer_marker) {
        std::printf("SKIP: the device has no VK_AMD_buffer_marker\n");
        return 77;
    }
    const int rc = run(d);
    const unsigned errors = d.validation_errors();
    std::printf("Vulkan validation errors: %u%s\n", errors, d.validation ? "" : " (validation off)");
    return rc ? rc : errors ? 1 : 0;
}
