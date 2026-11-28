// gta6rt --precompile-pipelines: builds the compute pipelines of the whole translated shader library ahead of play.
//
// A first build of a large compute module takes the driver tens of seconds to minutes (3-4.6 MB of SPIR-V: 33-107 s
// on NVIDIA 630.20), and a title that dispatches one stalls its submissions behind it. Run with no game, this creates
// the Vulkan device exactly as `gta6rt --run` does (the same environment: LEONIDA_SHADERS, LEONIDA_PIPELINE_CACHE_DIR,
// LEONIDA_DEVICE_FAULT/LEONIDA_AFTERMATH, which change the pipelines) and builds every compute entry through the
// same path as a dispatch (build_compute_pipeline), so each lands in the pipeline journal
// (src/pipeline_cache_journal.h) under the name the game looks it up by, and in the driver's own disk cache, which
// NVIDIA keys by the executable's name. Entries whose journal file is there are skipped, so a run can be stopped at
// any time and started again.
//
// LEONIDA_SKIP_COMPUTE (the play-time list of compute programs whose dispatches are dropped) does not narrow the
// build: a player runs this with the game's environment, and a program kept out of the journal here would stay out
// after the list is shortened. --skip-listed leaves the listed entries out on purpose. The last line counts what was
// built, what was journaled already and what was left out, and why.
#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace rt::d3d12 {

struct PrecompileOptions {
    unsigned threads = 4;              // worker threads, below normal priority
    unsigned big_threads = 0;          // of those, how many may build a module over big_bytes at once (0: threads)
    std::size_t big_bytes = std::size_t{2} << 20;
    bool rebuild = false;              // --all: build entries whose journal file exists too
    bool smallest_first = false;       // default: the largest modules first (the ones that stall a title longest)
    std::size_t limit = 0;             // at most this many entries (0: all)
    std::size_t min_bytes = 0, max_bytes = 0;  // SPIR-V size bounds (0: none)
    std::vector<std::string> filters;  // only entries whose name contains one of these
    bool skip_listed = false;          // --skip-listed: leave out the entries LEONIDA_SKIP_COMPUTE names
    bool dry_run = false;              // --list: print the plan, build nothing
    bool via_helper = false;           // --via-helper: as a title does with LEONIDA_COMPILE_HELPER=1 (a test)
};

// gta6rt --compile-server (d3d12_compile_helper.h).
struct CompileServerOptions {
    unsigned long parent_pid = 0;  // --parent: ends when that process does
    unsigned threads = 2;          // --threads
    unsigned idle_threads = 1;     // --idle-threads: of `threads`, how many fill the library when nothing is asked
    unsigned idle_exit_s = 0;      // --idle-exit: ends after this long with nothing asked (0: never)
};
int compile_server(const CompileServerOptions& options);

// Returns 0 when every selected entry was built or already journaled, 2 when some could not be built, 1 when the
// device or the library could not be had.
int precompile_compute_pipelines(const PrecompileOptions& options);

}  // namespace rt::d3d12
