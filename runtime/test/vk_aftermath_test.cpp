// Nsight Aftermath GPU crash dumps (src/vulkan_aftermath.h) on a real device: a compute shader writes through a
// bogus buffer device address, the device is lost, and the dump must appear and its decoded summary must name the
// faulting shader (its faulted warp's pc) by the name it was created under. Built only with -DLEONIDA_AFTERMATH_SDK.
//
// It loses the device on purpose, so it runs only with LEONIDA_AFTERMATH_FAULT_TEST=1; otherwise, without the
// Aftermath DLL, or without an NVIDIA GPU it prints SKIP and exits 77. A page fault only: no hang, no TDR.
// LEONIDA_AFTERMATH_DIR picks the dump folder (default %TEMP%\leonida_aftermath_test).
#include "vulkan_loader.h"

#include "vulkan_aftermath.h"
#include "vulkan_device_fault.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {

constexpr int skip = 77;
constexpr const char* shader_name = "aftermath_fault_cs";
// Far from anything the driver maps for this small process, inside the GPU's virtual address range.
constexpr std::uint64_t bogus_address = 0x0000123400000000ull;

#define FAIL(...)                                    \
    do {                                             \
        std::fprintf(stderr, "FAIL: " __VA_ARGS__);  \
        std::fputc('\n', stderr);                    \
        return 1;                                    \
    } while (0)

std::vector<std::uint32_t> read_spirv(const char* path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) return {};
    const auto size = std::size_t(file.tellg());
    std::vector<std::uint32_t> words(size / 4);
    file.seekg(0);
    file.read(reinterpret_cast<char*>(words.data()), std::streamsize(words.size() * 4));
    return file ? words : std::vector<std::uint32_t>{};
}

}  // namespace

int main() {
    const char* armed = std::getenv("LEONIDA_AFTERMATH_FAULT_TEST");
    if (!armed || std::strcmp(armed, "1") != 0) {
        std::puts("SKIP: set LEONIDA_AFTERMATH_FAULT_TEST=1 to run (it loses the GPU device on purpose)");
        return skip;
    }
    if (!leonida::aftermath::available()) {
        std::puts("SKIP: built without LEONIDA_AFTERMATH_SDK");
        return skip;
    }
    SetEnvironmentVariableA("LEONIDA_AFTERMATH", "1");
    SetEnvironmentVariableA("LEONIDA_AFTERMATH_KEEP_SPIRV", "1");  // the decoder maps the pc into our module
    if (!std::getenv("LEONIDA_AFTERMATH_DIR")) {
        const auto dir = std::filesystem::temp_directory_path() / "leonida_aftermath_test";
        SetEnvironmentVariableW(L"LEONIDA_AFTERMATH_DIR", dir.c_str());
    }
    if (!leonida::aftermath::enable()) {
        std::puts("SKIP: the Aftermath library did not load or enable");
        return skip;
    }

    const auto gipa = load_vulkan();
    if (!gipa) {
        std::puts("SKIP: no Vulkan loader");
        return skip;
    }
#define INSTANCE_FN(name) const auto name = reinterpret_cast<PFN_##name>(gipa(instance, #name))
    VkInstance instance = VK_NULL_HANDLE;
    const auto vkCreateInstance = reinterpret_cast<PFN_vkCreateInstance>(gipa(nullptr, "vkCreateInstance"));
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "vk_aftermath_test";
    app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo instance_info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    instance_info.pApplicationInfo = &app;
    if (!vkCreateInstance || vkCreateInstance(&instance_info, nullptr, &instance) != VK_SUCCESS) {
        std::puts("SKIP: no Vulkan 1.3 instance");
        return skip;
    }
    INSTANCE_FN(vkEnumeratePhysicalDevices);
    INSTANCE_FN(vkGetPhysicalDeviceProperties);
    INSTANCE_FN(vkGetPhysicalDeviceFeatures2);
    INSTANCE_FN(vkGetPhysicalDeviceQueueFamilyProperties);
    INSTANCE_FN(vkEnumerateDeviceExtensionProperties);
    INSTANCE_FN(vkCreateDevice);
    INSTANCE_FN(vkGetDeviceProcAddr);
    std::uint32_t count = 0;
    vkEnumeratePhysicalDevices(instance, &count, nullptr);
    std::vector<VkPhysicalDevice> physicals(count);
    vkEnumeratePhysicalDevices(instance, &count, physicals.data());
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    VkPhysicalDeviceProperties props{};
    for (VkPhysicalDevice candidate : physicals) {
        vkGetPhysicalDeviceProperties(candidate, &props);
        if (props.vendorID == 0x10de && props.apiVersion >= VK_API_VERSION_1_3) {
            physical = candidate;
            break;
        }
    }
    if (!physical) {
        std::puts("SKIP: no NVIDIA Vulkan 1.3 device");
        return skip;
    }
    std::printf("device: %s\n", props.deviceName);

    vkGetPhysicalDeviceQueueFamilyProperties(physical, &count, nullptr);
    std::vector<VkQueueFamilyProperties> families(count);
    vkGetPhysicalDeviceQueueFamilyProperties(physical, &count, families.data());
    std::uint32_t family = UINT32_MAX;
    for (std::uint32_t i = 0; i < count && family == UINT32_MAX; ++i)
        if (families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) family = i;
    if (family == UINT32_MAX) FAIL("no compute queue family");

    vkEnumerateDeviceExtensionProperties(physical, nullptr, &count, nullptr);
    std::vector<VkExtensionProperties> advertised(count);
    vkEnumerateDeviceExtensionProperties(physical, nullptr, &count, advertised.data());

    const float priority = 1.0f;
    VkDeviceQueueCreateInfo queue_info{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    queue_info.queueFamilyIndex = family;
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = &priority;
    VkPhysicalDeviceVulkan12Features vk12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    vk12.bufferDeviceAddress = VK_TRUE;
    VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &vk12};
    features.features.shaderInt64 = VK_TRUE;
    VkDeviceCreateInfo device_info{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &features};
    device_info.queueCreateInfoCount = 1;
    device_info.pQueueCreateInfos = &queue_info;
    // The runtime's path for LEONIDA_AFTERMATH=1 without LEONIDA_DEVICE_FAULT: the NVIDIA diagnostics config only.
    leonida::DeviceFaultFeatures fault_features;
    std::vector<const char*> extensions;
    const auto diagnostics = fault_features.enable(physical, vkGetPhysicalDeviceFeatures2, advertised, extensions,
                                                   device_info, {false, true, leonida::aftermath::extra_config_flags()});
    std::printf("VK_NV_device_diagnostics_config: %s\n", diagnostics.diagnostics_config ? "on" : "unavailable");
    device_info.enabledExtensionCount = std::uint32_t(extensions.size());
    device_info.ppEnabledExtensionNames = extensions.empty() ? nullptr : extensions.data();
    VkDevice device = VK_NULL_HANDLE;
    if (vkCreateDevice(physical, &device_info, nullptr, &device) != VK_SUCCESS) FAIL("vkCreateDevice");

#define DEVICE_FN(name) const auto name = reinterpret_cast<PFN_##name>(vkGetDeviceProcAddr(device, #name))
    DEVICE_FN(vkGetDeviceQueue);
    DEVICE_FN(vkCreateShaderModule);
    DEVICE_FN(vkCreatePipelineLayout);
    DEVICE_FN(vkCreateComputePipelines);
    DEVICE_FN(vkCreateCommandPool);
    DEVICE_FN(vkAllocateCommandBuffers);
    DEVICE_FN(vkBeginCommandBuffer);
    DEVICE_FN(vkCmdBindPipeline);
    DEVICE_FN(vkCmdPushConstants);
    DEVICE_FN(vkCmdDispatch);
    DEVICE_FN(vkEndCommandBuffer);
    DEVICE_FN(vkCreateFence);
    DEVICE_FN(vkQueueSubmit);
    DEVICE_FN(vkWaitForFences);
    DEVICE_FN(vkDeviceWaitIdle);

    const std::vector<std::uint32_t> words = read_spirv(LEONIDA_AFTERMATH_FAULT_SPV);
    if (words.empty()) FAIL("cannot read %s", LEONIDA_AFTERMATH_FAULT_SPV);
    // Through the wrapper the runtime installs in its function table, under the name the summary must report.
    const PFN_vkCreateShaderModule create_module = leonida::aftermath::wrap_create_shader_module(vkCreateShaderModule);
    if (create_module == vkCreateShaderModule) FAIL("wrap_create_shader_module did not wrap");
    VkShaderModuleCreateInfo module_info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    module_info.codeSize = words.size() * 4;
    module_info.pCode = words.data();
    VkShaderModule module = VK_NULL_HANDLE;
    {
        const leonida::aftermath::ShaderName named(shader_name);
        if (create_module(device, &module_info, nullptr, &module) != VK_SUCCESS) FAIL("vkCreateShaderModule");
    }

    VkPushConstantRange range{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(std::uint64_t)};
    VkPipelineLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    layout_info.pushConstantRangeCount = 1;
    layout_info.pPushConstantRanges = &range;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    if (vkCreatePipelineLayout(device, &layout_info, nullptr, &layout) != VK_SUCCESS) FAIL("vkCreatePipelineLayout");
    VkComputePipelineCreateInfo pipeline_info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    pipeline_info.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    pipeline_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipeline_info.stage.module = module;
    pipeline_info.stage.pName = "main";
    pipeline_info.layout = layout;
    VkPipeline pipeline = VK_NULL_HANDLE;
    if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pipeline) != VK_SUCCESS)
        FAIL("vkCreateComputePipelines");

    VkCommandPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pool_info.queueFamilyIndex = family;
    VkCommandPool pool = VK_NULL_HANDLE;
    if (vkCreateCommandPool(device, &pool_info, nullptr, &pool) != VK_SUCCESS) FAIL("vkCreateCommandPool");
    VkCommandBufferAllocateInfo allocate{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    allocate.commandPool = pool;
    allocate.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocate.commandBufferCount = 1;
    VkCommandBuffer commands = VK_NULL_HANDLE;
    if (vkAllocateCommandBuffers(device, &allocate, &commands) != VK_SUCCESS) FAIL("vkAllocateCommandBuffers");
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    vkBeginCommandBuffer(commands, &begin);
    vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdPushConstants(commands, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof bogus_address, &bogus_address);
    vkCmdDispatch(commands, 4, 1, 1);
    if (vkEndCommandBuffer(commands) != VK_SUCCESS) FAIL("vkEndCommandBuffer");

    VkQueue queue = VK_NULL_HANDLE;
    vkGetDeviceQueue(device, family, 0, &queue);
    VkFenceCreateInfo fence_info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VkFence fence = VK_NULL_HANDLE;
    if (vkCreateFence(device, &fence_info, nullptr, &fence) != VK_SUCCESS) FAIL("vkCreateFence");
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &commands;
    std::printf("dispatching a store to 0x%016llx\n", static_cast<unsigned long long>(bogus_address));
    VkResult result = vkQueueSubmit(queue, 1, &submit, fence);
    if (result == VK_SUCCESS) result = vkWaitForFences(device, 1, &fence, VK_TRUE, 10'000'000'000ull);
    if (result == VK_SUCCESS) result = vkDeviceWaitIdle(device);
    std::printf("result: %d\n", int(result));
    if (result != VK_ERROR_DEVICE_LOST) FAIL("the bogus store did not lose the device (VkResult %d)", int(result));

    // The runtime's device-loss path: observe() runs the hook vulkan_aftermath installed, which waits for the dump.
    leonida::DeviceFaultDiagnostics::observe(nullptr, result, "vk_aftermath_test");

    const std::string dump = leonida::aftermath::last_dump_path();
    if (dump.empty()) FAIL("no GPU crash dump arrived");
    std::error_code ec;
    const auto size = std::filesystem::file_size(std::filesystem::path(std::u8string(dump.begin(), dump.end())), ec);
    if (ec || !size) FAIL("dump %s is missing or empty", dump.c_str());
    const std::string summary = leonida::aftermath::last_summary();
    if (summary.empty()) FAIL("the dump was not decoded");
    // The faulted warp's pc, mapped through the dump's shader table to the name the module was created under.
    if (summary.find(std::string("faulting shader ") + shader_name + " ") == std::string::npos)
        FAIL("the summary does not name %s as the faulting shader", shader_name);
    std::printf("PASS: %s (%llu bytes), summary names %s\n", dump.c_str(), static_cast<unsigned long long>(size),
                shader_name);
    // The lost device is left to process exit: its objects need no destruction for the test.
    return 0;
}
