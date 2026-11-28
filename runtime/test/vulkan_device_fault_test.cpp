#include "../src/vulkan_device_fault.h"

using namespace leonida;

namespace {
int feature_queries = 0;
int fault_queries = 0;
int checkpoint_queries = 0;
bool supported = true;
bool fault_data_written = false;
void* last_marker = nullptr;

VKAPI_ATTR void VKAPI_CALL query_features(VkPhysicalDevice, VkPhysicalDeviceFeatures2* features) {
    ++feature_queries;
    for (auto* next = static_cast<VkBaseOutStructure*>(features->pNext); next; next = next->pNext) {
        if (next->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FAULT_FEATURES_EXT) {
            auto* fault = reinterpret_cast<VkPhysicalDeviceFaultFeaturesEXT*>(next);
            fault->deviceFault = supported;
            fault->deviceFaultVendorBinary = supported;
        } else if (next->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DIAGNOSTICS_CONFIG_FEATURES_NV) {
            reinterpret_cast<VkPhysicalDeviceDiagnosticsConfigFeaturesNV*>(next)->diagnosticsConfig = supported;
        }
    }
}

VKAPI_ATTR VkResult VKAPI_CALL get_fault(VkDevice, VkDeviceFaultCountsEXT* counts, VkDeviceFaultInfoEXT* info) {
    ++fault_queries;
    if (!info) {
        counts->addressInfoCount = 1;
        counts->vendorInfoCount = 1;
        counts->vendorBinarySize = 8;
        return VK_SUCCESS;
    }
    if (counts->addressInfoCount != 1 || counts->vendorInfoCount != 1 || counts->vendorBinarySize != 8 ||
        !info->pAddressInfos || !info->pVendorInfos || !info->pVendorBinaryData)
        return VK_ERROR_UNKNOWN;
    info->pAddressInfos[0] = {VK_DEVICE_FAULT_ADDRESS_TYPE_READ_INVALID_EXT, 0x1234, 4};
    info->pVendorInfos[0].vendorFaultCode = 1;
    fault_data_written = true;
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL set_checkpoint(VkCommandBuffer, const void* marker) {
    last_marker = const_cast<void*>(marker);
}

VKAPI_ATTR void VKAPI_CALL get_checkpoints(VkQueue, std::uint32_t* count, VkCheckpointDataNV* checkpoints) {
    ++checkpoint_queries;
    *count = 1;
    if (checkpoints) {
        checkpoints[0].stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        checkpoints[0].pCheckpointMarker = last_marker;
    }
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL get_proc(VkDevice, const char* name) {
    if (std::strcmp(name, "vkGetDeviceFaultInfoEXT") == 0) return reinterpret_cast<PFN_vkVoidFunction>(get_fault);
    if (std::strcmp(name, "vkCmdSetCheckpointNV") == 0) return reinterpret_cast<PFN_vkVoidFunction>(set_checkpoint);
    if (std::strcmp(name, "vkGetQueueCheckpointDataNV") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(get_checkpoints);
    return nullptr;
}
} // namespace

int main() {
    const VkExtensionProperties advertised[] = {{VK_EXT_DEVICE_FAULT_EXTENSION_NAME, 2},
                                                {VK_NV_DEVICE_DIAGNOSTIC_CHECKPOINTS_EXTENSION_NAME, 2},
                                                {VK_NV_DEVICE_DIAGNOSTICS_CONFIG_EXTENSION_NAME, 2}};
    const VkPhysicalDeviceVulkan12Features existing{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    VkDeviceCreateInfo device{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &existing};
    DeviceFaultFeatures features;
    std::vector<const char*> extensions;
    const auto absent = features.enable(VK_NULL_HANDLE, query_features, {}, extensions, device);
    if (absent.fault || absent.checkpoints || absent.diagnostics_config || feature_queries || !extensions.empty() ||
        device.pNext != &existing)
        return 1;
    supported = false;
    const auto disabled = features.enable(VK_NULL_HANDLE, query_features, advertised, extensions, device);
    if (disabled.fault || !disabled.checkpoints || disabled.diagnostics_config || extensions.size() != 1 ||
        device.pNext != &existing)
        return 2;
    extensions.clear();
    supported = true;
    const auto enabled = features.enable(VK_NULL_HANDLE, query_features, advertised, extensions, device);
    const VkDeviceDiagnosticsConfigFlagsNV flags = VK_DEVICE_DIAGNOSTICS_CONFIG_ENABLE_SHADER_DEBUG_INFO_BIT_NV |
                                                   VK_DEVICE_DIAGNOSTICS_CONFIG_ENABLE_RESOURCE_TRACKING_BIT_NV |
                                                   VK_DEVICE_DIAGNOSTICS_CONFIG_ENABLE_AUTOMATIC_CHECKPOINTS_BIT_NV;
    if (!enabled.fault || !enabled.checkpoints || !enabled.diagnostics_config || extensions.size() != 3 ||
        features.fault.deviceFaultVendorBinary != VK_TRUE || device.pNext != &features.config ||
        features.config.flags != flags || features.config.pNext != &features.diagnostics ||
        features.diagnostics.pNext != &features.fault || features.fault.pNext != &existing)
        return 3;

    if constexpr (sizeof(void*) >= 8) {
        const auto marker = CheckpointMarker::decode(CheckpointMarker::encode(
            UINT32_MAX, CheckpointMarker::max_list, CheckpointMarker::max_command, CheckpointKind::resolve));
        if (marker.submission != UINT32_MAX || marker.list != CheckpointMarker::max_list ||
            marker.command != CheckpointMarker::max_command || marker.kind != CheckpointKind::resolve ||
            CheckpointMarker::encode(1, 0, CheckpointMarker::max_command + 1, CheckpointKind::clear) ||
            CheckpointMarker::decode(0).submission)
            return 4;
    }

    DeviceFaultDiagnostics diagnostics(get_proc, VK_NULL_HANDLE, enabled);
    const auto queue = reinterpret_cast<VkQueue>(std::uintptr_t{1});
    diagnostics.add_queue(queue);
    diagnostics.add_queue(queue);
    auto submission = diagnostics.begin_submission(diagnostics.next_execute_call(), queue, "direct", 2);
    diagnostics.pipeline_names[4] = {"test_shader"};
    if (const auto set = diagnostics.checkpoint_command()) {
        const CheckpointRecorder recorder{set, submission, diagnostics.pipeline_names, 1};
        recorder.record(VK_NULL_HANDLE, 7, CheckpointKind::draw, VK_NULL_HANDLE, 4);
        recorder.record(VK_NULL_HANDLE, 9, CheckpointKind::copy);
        const auto marker = CheckpointMarker::decode(reinterpret_cast<std::uintptr_t>(last_marker));
        if (marker.submission != submission.sequence || marker.list != 1 || marker.command != 9 ||
            marker.kind != CheckpointKind::copy || submission.lists[1].shaders.size() != 1)
            return 5;
    }
    // The CPU history must survive missing driver checkpoints and a list longer than the ring.
    const CheckpointRecorder cpu{nullptr, submission, diagnostics.pipeline_names, 1};
    std::array<std::uint32_t, 32> roots{};
    roots[5] = 42;
    for (unsigned i = 0; i < 70; ++i)
        cpu.dispatch(VK_NULL_HANDLE, i, VK_NULL_HANDLE, 4, roots, {i, 2, 3});
    cpu.dispatch(VK_NULL_HANDLE, 70, VK_NULL_HANDLE, 4, roots, {}, true, 17, 32, 19, 64, 4, 12);
    const auto& oldest = submission.dispatches[submission.dispatch_count % submission.dispatch_limit];
    const auto& latest = submission.dispatches[(submission.dispatch_count - 1) % submission.dispatch_limit];
    if (submission.dispatches.size() != 64 || oldest.command != 7 || oldest.groups[0] != 7 ||
        oldest.roots[5] != 42 || std::strcmp(oldest.shader, "test_shader") ||
        latest.list != 1 || !latest.indirect || latest.argument_buffer != 17 || latest.count_offset != 64 ||
        latest.stride != 12 || latest.queue != queue) return 7;
    diagnostics.publish(std::move(submission));
    // A stopped timeline emits the CPU history once. Idle time and new completions reset the clock.
    using Watch = SubmissionProgressWatch;
    using namespace std::chrono_literals;
    Watch watch;
    const Watch::Clock::time_point start{};
    if (watch.sample(Watch::State::idle, 3, start) ||
        watch.sample(Watch::State::pending, 3, start + 30s) ||
        watch.sample(Watch::State::pending, 3, start + 44s) ||
        !watch.sample(Watch::State::pending, 3, start + 45s) ||
        watch.sample(Watch::State::pending, 3, start + 70s) ||
        watch.sample(Watch::State::pending, 4, start + 71s) ||
        watch.sample(Watch::State::submit_lock_busy, 4, start + 86s) ||
        !watch.sample(Watch::State::submit_lock_busy, 4, start + 101s)) return 8;
    diagnostics.report_stall("synthetic pending work", 5, 4);
    if (fault_queries || checkpoint_queries) return 9;
    if (DeviceFaultDiagnostics::observe(&diagnostics, VK_SUCCESS, "test success") != VK_SUCCESS || fault_queries ||
        DeviceFaultDiagnostics::observe(&diagnostics, VK_ERROR_DEVICE_LOST, "test loss") != VK_ERROR_DEVICE_LOST ||
        DeviceFaultDiagnostics::observe(&diagnostics, VK_ERROR_DEVICE_LOST, "test repeat") != VK_ERROR_DEVICE_LOST ||
        fault_queries != 2 || !fault_data_written || checkpoint_queries != 2)
        return 6;
}
