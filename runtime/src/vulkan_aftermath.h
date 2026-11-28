#pragma once

// Optional NVIDIA Nsight Aftermath GPU crash dumps (LEONIDA_AFTERMATH=1). The SDK is never part of the build unless
// CMake is given -DLEONIDA_AFTERMATH_SDK=<dir>; then vulkan_aftermath.cpp is compiled with LEONIDA_HAVE_AFTERMATH and
// loads GFSDK_Aftermath_Lib.x64.dll at run time (LEONIDA_AFTERMATH_DLL, else the SDK's lib/x64, else the DLL search
// path). Without the SDK, or without the DLL or an NVIDIA driver, every function here does nothing.
//
// Environment:
//   LEONIDA_AFTERMATH=1             enable GPU crash dumps for the Vulkan device (before vkCreateDevice)
//   LEONIDA_AFTERMATH_DIR=<dir>     where dumps go (default leonida-aftermath under the system temp directory): <stamp>.nv-gpudmp, the decoded
//                                   <stamp>.json, shader-<id>.nvdbg debug info, and <hash>.spv for the active shaders
//   LEONIDA_AFTERMATH_DLL=<file>    the Aftermath library to load
//   LEONIDA_AFTERMATH_SHADER_ERRORS=1  also set VK_DEVICE_DIAGNOSTICS_CONFIG_ENABLE_SHADER_ERROR_REPORTING_BIT_NV
//                                   (reports more shader errors, so it can fault where a run does not)
//   LEONIDA_AFTERMATH_KEEP_SPIRV=1  keep a copy of every SPIR-V module created, for the decoder (it needs the binary
//                                   to map a pc to a SPIR-V instruction). Off by default, as the game creates GBs of
//                                   them: the decoder then reads <LEONIDA_SHADERS>/<name>.spv back when its hash
//                                   matches the module's (true for unpatched library shaders)

#include <vulkan/vulkan.h>

#include <cstddef>
#include <cstdint>
#include <string>

namespace leonida::aftermath {

// True when this build has the Aftermath integration (LEONIDA_HAVE_AFTERMATH).
bool available() noexcept;

// LEONIDA_AFTERMATH=1: loads the library and calls GFSDK_Aftermath_EnableGpuCrashDumps for Vulkan. Must run before
// vkCreateDevice. The first call decides; returns whether crash dumps are enabled. Also installs the device-lost hook
// of vulkan_device_fault.h.
bool enable() noexcept;
bool enabled() noexcept;

// VK_NV_device_diagnostics_config flags to add to the device-fault defaults (shader debug info, resource tracking,
// automatic checkpoints), 0 when Aftermath is off.
VkDeviceDiagnosticsConfigFlagsNV extra_config_flags() noexcept;

// Returns a vkCreateShaderModule that also records the module's Aftermath shader hash under the name in
// ShaderName, or `real` itself when Aftermath is off. Process-wide: the first `real` wins.
PFN_vkCreateShaderModule wrap_create_shader_module(PFN_vkCreateShaderModule real) noexcept;

// Returns a vkGetDeviceProcAddr that hands out the wrapper above for "vkCreateShaderModule", so every function table
// loaded through it (the runtime's helpers load their own) records its modules; `real` itself when Aftermath is off.
PFN_vkGetDeviceProcAddr wrap_get_device_proc_addr(PFN_vkGetDeviceProcAddr real) noexcept;

// Records a SPIR-V module created outside the wrapper.
void register_spirv(const void* code, std::size_t bytes, const char* name) noexcept;

// Blocks until Aftermath has written and decoded the GPU crash dump, at most ~10 s. Any thread, any number of times:
// the first device loss waits, later calls return at once (or wait for the first one to finish).
void wait_for_dump() noexcept;

// For tests: the last dump written and the summary printed for it (empty before a dump).
std::string last_dump_path();
std::string last_summary();

namespace detail {
inline thread_local const char* shader_name = nullptr;
}

// Names the shader modules this thread creates while it is in scope (the wrapper above reads it).
class ShaderName {
public:
    explicit ShaderName(const char* name) noexcept : previous_(detail::shader_name) { detail::shader_name = name; }
    explicit ShaderName(const std::string& name) noexcept : ShaderName(name.c_str()) {}
    ~ShaderName() { detail::shader_name = previous_; }
    ShaderName(const ShaderName&) = delete;
    ShaderName& operator=(const ShaderName&) = delete;

private:
    const char* previous_;
};

}  // namespace leonida::aftermath
