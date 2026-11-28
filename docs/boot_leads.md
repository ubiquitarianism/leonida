# Boot leads: open findings and how to implement them

Leads that may move the retail boot further, each with where it lands in this repository and how to check it.
Every lead started as a **hypothesis**. Its **Status** line says whether a merged pull request has answered it, an
open one is working on it, or nothing has yet. What is not marked done there has not been measured against this
package or run in this runtime: confirm it on your own copy or with a synthetic test before you build on it, and
record the result in the document named under **Record it in**.

| Lead | Status |
|---|---|
| BL-1 TLS from module index 0 | Done on `main` with a synthetic test; the retail confirmation is pending |
| BL-2 Feature level at device creation | Done: #913 |
| BL-3 Replay the console command list | In progress: #1050 (open) |
| BL-4 Descriptor words as 32-bit addresses | Done: #1166 (no bare 32-bit addresses in the census) |
| BL-5 Shader corpus count | Done: #882 (a counting difference; the carve misses nothing) |
| BL-6 Frame pacing drift reporting | Done: #891; real numbers still need a retail run |

## Rules

- **No game data, no keys, no protection workarounds.** If a format turns out to be encrypted, write that down in
  [design.md](design.md) and stop.
- **Nothing online**: no Xbox Live or Rockstar services.
- **Claim first.** Put the lead's ID (BL-1 …) in the coordination issue before you open a pull request (the search
  link is in [CONTRIBUTING.md](../CONTRIBUTING.md)), and check the open pull requests for the same item.
- A translator rule needs a row in the contract table in [design.md](design.md) with how it was confirmed, and a
  hand-written `.asm` fixture in `tests/semantic/`.
- In the pull request, say what you tested and on which GPU and driver.

## BL-1: the title's code reads its TLS from module index 0

- **Finding:** code built with `/GA` reads thread-local storage straight from `ThreadLocalStoragePointer[0]` and
  assumes the executable's own TLS block sits at index 0. Static TLS has to be set up per mapped module, not once.
- **Where:** a retail run on `main` at `867d922f` ended in CRT startup with an access violation and 0 D3D12 calls, "the
  stop-4 shape (TLS from module index 0)" ([retail_boot_intel_host.md §1](retail_boot_intel_host.md)). The loader keeps
  one TLS host slot per mapped module (`runtime/src/pe_loader.cpp`, `runtime/src/tls_host.cpp`). The integration tree
  clears stop 4 with #4313 ([retail_boot_chain.md](retail_boot_chain.md), row 4).
- **Status:** answered on `main` by `b7a3fed28` and `4710549af`, which the measurement above predates; the retail
  confirmation is still pending. `load()` rewrites the executable's `mov r64, gs:[58h]` reads so that they reach a
  per-thread view with the image's own block at `[0]`, and each other mapped module with a TLS directory has an index of
  its own. `loader_tls` checks it with a `/GA` executable beside a DLL with TLS of its own, on the loading thread and on
  a new one, and shows the stop with `LEONIDA_TLS_GA_PATCH=0`; `run_tiny_game_ga_tls` runs a `/GA` build of the stand-in
  game through `gta6rt --run`. #4313, which #321 and #64 carry, is an alternative design, not a missing fix: the
  runtime's own EXE gives up index 0 at process start, so an index-0 read of any shape works without a rewrite. What is
  left of this lead is the retail `--run` on current `main` and, if it still stops here, the read shape the rewrite does
  not know.
- **Implement:**
  1. Find the stop-4 fix on the integration tree and compare it with `Loader::initialize_tls` on `main`.
  2. Make the executable's slot resolve at index 0 of the thread's TLS array, and give every other mapped module that
     has a TLS directory its own index.
  3. Keep the per-thread copy and restore behaviour in `tls_host.cpp` (new threads get a copy; the helper's own block
     is put back before the thread exits).
- **Test:** extend `runtime/test/loader_tls_test.cpp` and `runtime/test/fixtures/tiny_game.cpp` with a synthetic image
  whose code reads `gs:[58h]` index 0 and a second module with its own TLS variable. No game files needed. Then run
  the retail `--run` to confirm the stop is cleared.
- **Record it in:** [boot.md](boot.md) (the `tls_callback`/`dllmain` section) and the stop table in
  [retail_boot_chain.md](retail_boot_chain.md).
- **Needs:** none for the synthetic test; game files for the retail confirmation.

## BL-2: feature level at device creation

- **Finding:** device creation can fail with `E_INVALIDARG` (`0x80070057`) when asked for feature level 11.0 and
  succeed when asked for 12.0.
- **Where:** `CheckFeatureSupport` reports levels up to 12_1 (`runtime/shims/d3d12_vulkan_device.cpp`, the
  `D3D12_FEATURE_FEATURE_LEVELS` case); the Xbox entry point is in `runtime/shims/d3d12_xs_device.cpp`.
- **Status:** done by #913 (merged in `531aa4f5`). The minimum was the problem, not the reported level: the host path
  of `D3D12XboxCreateDevice` asked the PC `D3D12CreateDevice` for 12_0 as its minimum, which refuses every adapter
  below it. `rt::d3d12_xs::create_host_device` now asks for `kHostDeviceFloor` = 11_0, and the device reports its
  own highest level. `xs_proxy_mock_test` covers mock hosts that top out at 11_0 through 12_2. The write-up is in
  [d3d12_xs_surface.md](d3d12_xs_surface.md) ("Feature level at device creation (BL-2)"). Still open: reproducing the
  original failure on the Wine or Proton host it came from.
- **Implement:** read the level the title passes at creation and make the layer accept the full range it can
  satisfy, returning the highest supported level through the feature-level query rather than failing creation on a
  lower request. Report 12_2 only where mesh shaders and ray tracing are present
  ([d3d12_vulkan.md](d3d12_vulkan.md), feature-level notes).
- **Test:** a mock-device case in `runtime/test/xs_proxy_mock_test.cpp` creating at 11_0, 12_0, 12_1 and 12_2 and
  asserting the result and the reported level for each.
- **Record it in:** [d3d12_xs_surface.md](d3d12_xs_surface.md).
- **Needs:** none for the mock test; a Wine or Proton host to reproduce the original failure.

## BL-3: replay the console command list at Close or Execute

- **Finding:** the console's command-list interface may be easier to serve by replaying the recorded commands when the
  list is closed or executed than by translating each call as it is made.
- **Where:** the "console list tail" listed as open in the README's D3D12 → Vulkan row and in
  [d3d12_vulkan.md](d3d12_vulkan.md); call orders are in [d3d12_command_objects.md](d3d12_command_objects.md).
- **Status:** in progress in #1050 (open: "replay the console list's inline records as their vtable calls"). Read it
  and its review before starting work of your own here.
- **Implement:** record the console-only calls into a per-list buffer, then translate them to Vulkan in order at
  `Close`/`Execute`. Keep calls that must take effect immediately (those whose results the title reads back) out of
  the buffer.
- **Test:** a recorded-sequence unit test: the same call order through the immediate path and the replay path gives
  the same Vulkan command stream. Add a GPU case only once the stream matches.
- **Record it in:** [d3d12_vulkan.md](d3d12_vulkan.md), with the measured call order.
- **Needs:** game files (call traces) to find which calls read results back; a GPU for the final check.

## BL-4: descriptor words are 32-bit GPU addresses

- **Finding:** a console descriptor word may be a 32-bit GPU address rather than a heap offset.
- **Where:** the "descriptor-table record model" listed as open in [d3d12_vulkan.md](d3d12_vulkan.md), next to the
  descriptor-heap start-handle convention in §7 of that page.
- **Result:** measured by retail descriptor heap census (1,245,184 slots across submits 50, 500 and 5000; RTX 5070, Windows 11).
  The heap holds: 910,244 empty slots; 327,543 free-list links (`value = next slot index`, terminating at `0xFFFFFFFF`);
  261 T# with layer-assigned addresses above 4 GiB; 5,840 V# (3,024 with a null base, 2,816 with a sub-4-GiB base);
  1,296 slots of interleaved constant data; and **0 slots that are bare 32-bit GPU addresses**.
- **Conclusion:** BL-4 hypothesis is not supported on this stack. Descriptors pack 48-bit addresses into V# / T# fields rather
  than exposing bare 32-bit pointer words. Characterized by model-level reference contract suite `tests/test_descriptor_address_translation.py` (53 tests modeling the documented arena bitfield partitioning and address extraction contracts).
- **Status:** done by #1166 (merged in `95eacfa7c`): the census found no bare 32-bit addresses, so the
  hypothesis is not supported on this stack.
- **Record it in:** [d3d12_vulkan.md](d3d12_vulkan.md) §3.4 with the census breakdown.
- **Needs:** none (census completed and verified).

## BL-5: a possible gap in the shader corpus count

- **Finding:** a count of 26,698 shader programs has been reported against this repository's 26,503. The difference,
  195, equals the geometry-shader count here, which may mean a different way of counting, or programs the carve
  misses.
- **Where:** the carve and library tools (`make_game.py`, `tools/shader_carve.py`, `tools/shader_library.py`) and
  [shader_library.md](shader_library.md).
- **Status:** done by #882 (merged in `71f99e22`): a counting difference, not a gap. 26,503 counts carved `DXBC`
  containers and 26,698 counts the `FSSM` payloads in them; the 195 extra are vertex-shader containers that also hold
  the export-shader build of the same shader for a legacy geometry-shader pipeline. `tools/fssm_unpack.py` prints
  both counts and a line per stage; the table is in [shader_library.md](shader_library.md) ("Containers and
  programs"). Still open: which payload a geometry-shader pipeline selects ([fssm_crosscheck.md](fssm_crosscheck.md)
  §6); the 195 export-stage variants are not translated yet.
- **Implement:** list how programs are enumerated (per file, per container, per variant) and print a breakdown by
  stage. If the 195 are a counting difference, document it. If they are real, find where the carve skips them and fix
  that, not the count.
- **Test:** a synthetic container with a geometry-stage program, so the count and the breakdown are checked without
  game files.
- **Record it in:** [shader_library.md](shader_library.md).
- **Needs:** game files for the real corpus.

## BL-6: frame pacing drift reporting

- **Finding:** frame pacing can drift on a long-running screen, and per-frame percentile and input-to-present
  figures make that visible.
- **Where:** the frame pacer in `runtime/src/frame_pacer.h` and its `PresentX` wiring in
  `runtime/shims/xbox_present.cpp`.
- **Status:** done by #891 (merged in `3ba069eb`). `FramePacer` keeps ring buffers of frame times and
  origin-to-present latencies (`frame_pacing_window` = 1024), `XboxPresentation` logs a pacing summary at teardown,
  and `tools/boot_trace.py` reports it (p50/p95/p99, mean and drift against target);
  `runtime/test/frame_pacer_test.cpp` checks it with a fake clock. Recorded in [d3d12_vulkan.md](d3d12_vulkan.md)
  ("Pacing drift reporting (BL-6)") and [boot.md](boot.md). Still open: numbers from a retail run.
- **Implement:** record frame times in a ring buffer and report p50, p95 and p99 plus the drift of the mean against
  the target interval, in the run summary beside the existing D3D12 counts ([boot.md](boot.md)).
- **Test:** extend `runtime/test/frame_pacer_test.cpp` with a fake clock feeding known intervals, asserting the
  percentiles and the drift.
- **Record it in:** [d3d12_vulkan.md](d3d12_vulkan.md) (presentation) and [boot.md](boot.md) (summary fields).
- **Needs:** none for the reporting; game files and a GPU to see real numbers.
