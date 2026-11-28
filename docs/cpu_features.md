# CPU instructions the executable needs

The Xbox Series CPU is an AMD Zen 2, and the runtime runs the game's code natively
([design.md](design.md), "Why a loader before a recompiler"). Code built for one known CPU can use every instruction
that CPU has without asking CPUID first. An instruction the host lacks raises `#UD`
(`STATUS_ILLEGAL_INSTRUCTION`) where it is reached, and a privileged one faults in user mode. This page records
which such instructions `game_scarlett_master.exe` contains, measured with `tools/cpu_feature_scan.py`. Counts only;
nothing from the executable is quoted.

## Reproduce

```powershell
python -I tools\cpu_feature_scan.py <game>\game_scarlett_master.exe            # summary
python -I tools\cpu_feature_scan.py <game>\game_scarlett_master.exe --sites    # plus the RVA of each flagged site
```

It streams MSVC `dumpbin /disasm:nobytes` (a few minutes for this executable, with no listing written to disk; a saved
listing works too, with `--image <exe>` for `.pdata`). Each instruction is mapped to the CPUID feature or privilege class it
needs, and each count is split by whether the site lies inside a `.pdata` function range. A linear disassembly also
decodes data in `.text` (jump tables, padding), so a hit outside every `.pdata` range is often not code; a leaf
function that does not touch the stack has no `.pdata` entry either, so those sites need a look by hand.

## Measured result (1.0.2.1)

26,674,075 instructions decoded in `.text`, 22,435,554 of them inside the 299,257 `.pdata` ranges.

| Feature | Class | Sites | Inside `.pdata` | What a host without it does |
|---|---|---:|---:|---|
| SSE4a (`insertq`) | AMD only | 19 | 17 | `#UD` on every Intel CPU |
| CLZERO | AMD only | 10 | 0 | `#UD` on every Intel CPU |
| MONITORX (`monitorx`, `mwaitx`) | AMD only | 4 | 4 | `#UD` on every Intel CPU |
| CLWB | newer Intel only | 8 | 6 | `#UD` on Intel client CPUs before Ice Lake / Tiger Lake (Skylake to Comet Lake) |
| Port I/O, `cli`/`hlt` | privileged | 8 | 2 | fault in user mode |
| `int n` (not 3/29h/2Ch/2Dh) | kernel call | 1 | 1 | a software interrupt Windows does not serve |
| CPUID / XGETBV | host query | 9 / 2 | 9 / 2 | run everywhere; answer with the host's values |

Each flagged site was read by hand in a private copy (not quoted here):

- All 41 SSE4a, CLZERO, MONITORX and CLWB sites are real code: each sits among instructions of the same kind of
  code (vector arithmetic around the `insertq` sites, cache-line loops around CLZERO and CLWB, a wait on a memory
  address around the two `monitorx`/`mwaitx` pairs). The 10 CLZERO sites are one leaf routine, a cache-line zeroing
  loop with no `.pdata` entry, which is why none of them is inside a `.pdata` range.
- The 8 privileged and 1 `int n` hits are data: tables of 32-bit offsets placed in `.text` (some inside a `.pdata`
  range) that the linear sweep decodes as instructions. They are listed for completeness, not as code to replace.

No AVX-512, VAES, VPCLMULQDQ, SHA, RDPID, RDPRU, XOP, FMA4, TBM, SYSCALL or SYSENTER instruction was found. The
256-bit integer, FMA3, F16C, BMI1/2, ADX, MOVBE, AES-NI and PCLMULQDQ instructions it uses are on every Intel CPU
since Haswell/Broadwell and every AMD CPU since Zen, so they are counted but not flagged. No RDTSC or RDTSCP is
in the image.

## What it means for the runtime

- On an AMD Zen 2 or later host every flagged user-mode instruction runs as on the console.
- On an Intel host the SSE4a, CLZERO and MONITORX sites, and on Intel client CPUs before Ice Lake the CLWB sites,
  raise `STATUS_ILLEGAL_INSTRUCTION` when reached. `gta6rt` emulates them (`runtime/src/cpu_emulation.h`, see
  [runtime/README.md](../runtime/README.md), "PE loader"): CLZERO zeroes the 64-byte line, CLWB is a write-back hint
  with nothing to do on coherent memory, MWAITX returns at once (an early return is one the manual allows), and
  `insertq`/`extrq` are bit-field operations on the low quadword. `cpu_emulation_test` checks each form against the
  CPU on a host that has it and against the handler on one that does not (an i7-10700K: all four lacking, all checks
  pass).
- The CLZERO routine is too hot for the #UD path: one exception and two `VirtualQuery` calls per 64-byte line. On an
  i9-13900K the routine held one thread at ~99% for the whole boot (sampled: `VirtualQuery` under
  `cpu_emulation::apply`) and the title presented ~3 frames a second. On a CPU without CLZERO the loader therefore
  rewrites the routine's loops before any image code runs (`runtime/src/clzero_patch.h`): each of the two loop shapes
  (6 + 4 sites) jumps to a stub within 2 GiB of the image that zeroes the line with eight 8-byte stores and keeps the
  loop's registers and flags. Same boot afterwards: 164,946 PresentX in 300 s against 869 in 420 s, no CLZERO left on
  the #UD path. `LEONIDA_CLZERO_PATCH=0` keeps the emulation; `clzero_patch_test` runs both shapes on any CPU.
- The CPUID sites see the host's vendor, family and features, not the console's. What the game does with the
  answers is not measured here.

Not measured: which of these sites a boot reaches, and in which order.

### Hosts without AVX

Measured on one host: Intel Xeon X5675 (Westmere, SSE4.2; no AVX, AVX2, FMA, BMI2 or XSAVE), Fedora 44,
`wine-11.21 (Staging)`, Intel SDE 10.13.1, and a local `gta6rt.exe` build from 2026-11-24.

- **What works without AVX:** building the native Linux runtime and Vulkan test tools (`ctest`), running the Vulkan GPU
  suite (`scripts/run_gpu_tests.py`), shader translation and asset export (`make_game.py` stages).
- **Running `gta6rt` directly:** `--init` and `--run` exit 4 at the AVX preflight. With `LEONIDA_SKIP_CPU_CHECK=1`,
  the preflight is skipped and the guest faults on its first AVX instruction.
- **Under Intel SDE (`sde64 -hsw`):** without `-chip_check_disable 1`, SDE aborts on `clzero` instead of raising
  `SIGILL` for `gta6rt`'s handler. With `-chip_check_disable 1`, the guest passes its first AVX instruction and reaches
  `InitializeApiImplEx2` (offline) after ~110 s, then stops before any window or D3D12 call: SDE traps an `int 3`
  without `-raise-int3`, and with `-raise-int3` the process terminates with `SIGSEGV` (`LEONIDA_SKIP_INT3=1` with
  `--trace` did not change this, recording `0` `int3_skipped` events). Adding `-ptr-raise` causes Pin to abort with an
  internal instrumentation error under Wine's exception handling.
- **Outcome on this host:** no useful path to boot `gta6rt --run` was found under SDE + Wine.

Not measured: other emulators (such as QEMU TCG with AVX2) or Rosetta 2.
