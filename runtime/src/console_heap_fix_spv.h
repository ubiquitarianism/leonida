#pragma once
#include <cstdint>

namespace leonida {

// console_heap_fix.comp compiled with: glslc --target-env=vulkan1.2 -O -mfmt=c console_heap_fix.comp -o console_heap_fix.inc
inline constexpr std::uint32_t console_heap_fix_spirv[] =
#include "console_heap_fix.inc"
;

} // namespace leonida
