// The process is ending with the Vulkan device still alive (gta6rt's ExitProcess after the title's run, a console
// Ctrl+C, Ctrl+Break or close). Without this, ExitProcess ends the pipeline compiler's thread wherever it is,
// possibly inside the driver's compiler with its locks held. A device released normally needs none of this: ~Device
// cancels and joins its background builds (PipelineCompiler in d3d12_pipeline_compile.h).
#pragma once

namespace rt::d3d12 {

// Drops the queued background pipeline builds, refuses new ones and waits up to `wait_ms` for the running one.
// False when one was still running at the deadline. Defined with the Vulkan device (d3d12_vulkan_device.cpp).
bool stop_pipeline_compiles(unsigned wait_ms);

// Writes each live device's session pipeline cache (pipeline_cache.session.bin), as ~Device does, for a process that
// ends with its devices alive: after stop_pipeline_compiles, so the compiler's own builds are done. A build still
// running on another thread is given `wait_ms`, then the cache is written without it (the pipelines finished so far);
// the write itself is given 3 s before the process goes on ending. Runs once per process; true when a cache write
// finished.
bool save_pipeline_caches_at_exit(unsigned wait_ms);

}  // namespace rt::d3d12
