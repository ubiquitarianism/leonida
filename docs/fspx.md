# FSPX: the FidelityFX DLL's native compute programs

The package's FSR upscaler, `amd_fidelityfx_upscaler_xbox.dll`, carries 29 compute shaders as `DXBC` containers with
`SFI0`, `ISG1`, `OSG1`, `PSV0`, `RTS0`, `HASH` and `ACCC` parts and no `DXIL` ([d3d12_command_objects.md](d3d12_command_objects.md)
§3.5a). The `ACCC` part is an `FSPX` payload: a small directory around a precompiled RDNA2 program. It is not the
`XBCH`/`FSSM` format of the game's own shaders ([design.md](design.md)), but it holds the same kind of program, and
`tools/fspx_unpack.py` lays it out so `tools/rdna2spirv.py` translates it like a game compute shader.

Everything below was measured on the 1.0.2.1 package's DLL (SHA-256 `faff02f2…a75fa`). "Read from the DLL" means
Ghidra 12.2 output (through REA) for the named function; "measured" means checked on all 29 programs unless a count is
given; anything else is marked as inferred.

## Layout

```
u32  'FSPX'
u32  version          1 in all 29; the DLL does not read it
u32  count            4 in all 29
u32  total size       == the ACCC part size in all 29
count x { u32 kind, u32 offset, u32 size, u32 0 }    offset from the start of the payload
```

| Kind | Size | What it is |
|---|---|---|
| 0 | 28 bytes in all 29 | Dispatch state, see below. Required by the DLL. |
| 1 | 20 bytes in all 29 | Unknown. Optional for the DLL. |
| 6 | 768 to 41,536 bytes | The RDNA2 program, at offset 0x100 in all 29. Required by the DLL. |
| 7 | 152 to 696 bytes | Resource and hash words, see below. Optional for the DLL. |

### How the DLL reads it

The DLL's reader (RVA 0x1a20, read from the DLL) clamps the total to the buffer, then returns:

- 1 for a null or shorter-than-16-byte buffer, 2 when the magic is not `FSPX`,
- 4 when `16 + 16 * count` does not fit the total,
- 5 when kind 0 or kind 6 is missing or empty, else 0.

It skips an entry whose range starts inside the directory or ends past the total, and keeps the last entry of each
kind 0, 1, 6 and 7; other kinds are ignored. `tools/fspx_unpack.py`'s `parse_fspx` follows the same rules and codes.

The pipeline function (RVA 0x4aa0, read from the DLL) accepts either a bare `FSPX` payload or a `DXBC` container whose
`ACCC` part is one. It copies kind 6 into an `XMemAlloc` block (size rounded up to 256 bytes, attributes
`0xa8004000`) and calls device slot 67 with:

| Offset | Field |
|---|---|
| +0x00 | root signature |
| +0x08, +0x10 | kind 0 pointer, size |
| +0x18, +0x20 | kind 1 pointer, size |
| +0x28, +0x30 | the copied kind 6, its size |
| +0x38, +0x40 | kind 7 pointer, size |

The size at `+0x30` is kind 6's own size, not the rounded allocation: the DLL rounds it into a separate register for
`XMemAlloc` and stores the original, which is also the copy's length (read from the DLL's code with `objdump`, `0x5649` to
`0x56d5`). So
the bytes at `+0x28` with that size are exactly kind 6, and their SHA-256 is the key below.

**How the table was pinned** (read from the DLL's code with `objdump`; checked by running the DLL's reader on all 29
payloads):
- **The reader (`0x1a20`).** It zeroes four (pointer, size) pairs, then stores kind 0 in pair 0, kind 1 in pair 1,
  kind 6 in pair 2 and kind 7 in pair 3 (`0x1bcb` to `0x1bfb`). Each pointer points into the payload itself.
- **The container branch (`0x5510` to `0x55ea`).** For a `DXBC` container the pipeline function runs the same loop
  inline over the `ACCC` part and fills the same four places. Unlike the reader, it does not zero the kind-1 and
  kind-7 places first; all 29 payloads have both kinds.
- **The copy into the description (`0x567d` to `0x56d5`).** Pair 0 goes to `+0x08`, pair 1 to `+0x18`, the kind-6
  copy to `+0x28` with pair 2's size at `+0x30`, and pair 3 to `+0x38`.
- **Code and state come from one payload.** Kind 0 and kind 6 of one call therefore always belong to the same
  program, so the run-time lookup by code key and state key (below) has the pairing the build gives it.
- **Run.** A harness mapped the DLL without running it and called its reader on each of the 29 payloads. It built the
  description that way and called slot 67 on the Vulkan device with the 26 programs translated on `main` at
  `aa2988cc`:
  - all 26 resolve to their own entry with their own state, and the other 3 are refused with `E_INVALIDARG`;
  - the same on llvmpipe, the Ryzen 9 7950X3D's integrated GPU and the RX 7900 XTX (Mesa 26.3.1, under Wine).

`d3d12_fspx_compute_test` builds its slot-67 descriptions the same way, from a container's own directory.

**What happens when a program is not translated.** The DLL creates its pipelines one after another as a dispatch
needs them. When one of them has no translation, slot 67 fails, and so does the slot-62 fallback with the same
container. The DLL then records the null pipeline and faults at `0x65ff`, as on a PC device. The proxy logs every
failed slot-67 call with its number (`slot 67 call N failed`), because only the first call of a slot is announced.

The DLL's own messages name that call `DeserializeComputePipelineStateX` and the path "ACO". When the call fails,
the message says it falls back to "NewBE", and the DLL sets the environment variable `FFX_FSR4_XBOX_ACO_ACTIVE` to
`0`. Its next pipeline call is slot 62 (measured in #864), so "NewBE" is inferred to be that path, the one a PC
device refuses (§3.5a).

## Kind 0: dispatch state

Seven u32 words.

| Word | Meaning | How far it is checked |
|---|---|---|
| 0, 1 | Unknown | The same two values in all 29. |
| 2 | 0 | All 29. |
| 3 | `NumThreadsX` (bits 15..0), `NumThreadsY` (31..16) | Measured: equal to the container's `PSV0` thread-group size in all 29. |
| 4 | `NumThreadsZ` (15..0); bit 20 set = wave32 | Z measured against `PSV0` in all 29. Wave size: bit 20 is set in 24 programs and clear in 5, and nothing measured contradicts it (below the table). The translator takes it as the wave size. |
| 5 | `COMPUTE_PGM_RSRC1` | Inferred. `FLOAT_MODE` 0xC0, `DX10_CLAMP` and `MEM_ORDERED` in all 29. With a VGPR granule of 8 for wave32 and 4 for wave64, the highest VGPR the code uses is inside the allocation in all 21 programs that decode with no invalid word, at its last register in several. |
| 6 | `COMPUTE_PGM_RSRC2` | Inferred. `LDS_SIZE` covers the largest `ds_*` offset in all 29. `USER_SGPR` plus the enabled `TGID` SGPRs cover the SGPRs the code reads before writing them in 20 of those 21 (8 or 14 user SGPRs here). In the 21st, a wave64 program, the only SGPRs past them are the carry-out destination of a `v_add_co_ci_u32_e64`, which the check counted as a read. |

`rdna2spirv.py` reads words 3, 4 and 6 from `fspx/<name>.json` where it reads the FSSM header for a game compute
shader: the thread-group size, `RSRC2`, and the wave size from word 4 bit 20 (the `wave32` field).

**Word 4 bit 20 is the wave size.** Over the 29 programs of 1.0.2.1 (`tests/semantic/test_fspx_wave_size.py` has
the rule):

- **Strict markers (measured).** A strict wave64 marker (`rdna2spirv`'s `WAVE64_RE`: `exec_hi`, `v_mbcnt_hi`,
  `*_saveexec_b64` and the 64-bit lane-mask SOP ops) is in 4 of the 5 programs with the bit clear and in none of the
  24 with it set. A wave32 marker (`*_saveexec_b32`, `*_wrexec_b32`) is in 4 programs with the bit set and in none
  with it clear.
- **VGPR granule (measured).** In the 21 programs that decode with no invalid word, the highest VGPR is inside the
  `RSRC1` allocation with the granule the bit gives (8 for wave32, 4 for wave64) in all 21. All 18 of those with the
  bit set use a VGPR past what the wave64 granule would allocate. The 3 with the bit clear would fit either granule.
- **`PSV0` (measured).** Its expected wave lane counts are stated in 5 programs and agree with the bit in all 5 (64 in
  3 programs with it clear, 32 in 2 with it set); the other 24 leave them open (0 and the maximum).
- **Thread groups (measured; the reading is a hypothesis).** The programs with the bit clear have 64 or 256 threads
  per group; 17 with it set have 32, which as one wave64 would leave half its lanes idle.
- **`_b64` ops on `vcc` are not markers (measured counts; the reading is a hypothesis).** The three wave32 programs
  the old heuristic called wave64 write `vcc` from an SGPR pair with `s_mov_b64` (2 to 5 times each) and otherwise use
  `vcc_lo` and `vcc_hi`, including `vcc_lo` as the lane mask of `v_cmp`/`v_cndmask`: `vcc` held as an ordinary pair.
- **The one program with no evidence beyond the bit.** It has the bit clear and no marker either way, and its highest
  VGPR is the last register of the wave64 allocation, so the bit alone decides.

A strict marker against the bit makes the translation refuse the program, as a contradicting FSSM header does. A
program whose kind 0 is too short for word 4 keeps the code heuristic.

## Kind 6: the program

- **Not compressed.** Each program holds exactly one `s_endpgm`, followed by padding (mostly `s_nop 0`).
- **A 36-byte prologue** comes first. It is three 12-byte VOP3 words with opcode 0x180 (bits 31..16 = `0xD580`), plus
  literals.
  - They carry the container's shader hash. Measured: the low halves of words 0 and 3 and the literals in words 2 and
    5 equal the 16 bytes after kind 7's `0x43` word, which are the digest in the container's `HASH` part, in all 29.
  - `llvm-mc` rejects them. Measured on RDNA2 hardware (`gfx1031`, AMD Radeon RX 6700 XT, driver 26.11.2, through
    `tools/fspx_prologue_probe.py` / `tools/hw_oracle`): in wave32 and wave64 across 5 synthetic 36-byte prologues
    (50 positive checks and 8 negative controls), executing the raw words leaves VGPRs (`v0`, `v1`, `v8`..`v15`),
    SGPRs (`s[0:1]`, `s4`..`s15`), `M0`, `VCC`, `EXEC` (including inactive lanes across 64 threads) and `SCC` (both 0
    and 1) unchanged, and execution resumes cleanly at `+36` bytes. So they are hardware no-ops on RDNA2.
  - `tools/fspx_unpack.py` disassembles from the end of the prologue.
  - **A 24-byte variant**: some programs of the pipeline library in `shaders.rpf` (`tools/shader_library.py`) carry
    only the first two of those words, at words 0 and 3, and their code starts at word 6 (`s_inst_prefetch` in all of
    them). `code_range` takes that form only when both words are literal-bearing VOP3 opcode-0x180 words of the shape
    `tools/fspx_prologue_probe.py` checks (second dword `& 0xFFFE0100 == 0x03FE0100`); a program whose words 0, 3 and
    6 are all prologue words keeps the 36-byte form. Counted on the 1.0.2.1 library with an opcode-only check (#583):
    50 of 2,684 programs; read from word 0 they never decode (the prologue words are invalid encodings and the rest
    desyncs), and with the 24-byte start all 50 disassemble with 0 invalid words and 49 translate and pass
    `spirv-val` (the other stops at `s_sethalt`). `tests/test_fspx_unpack.py` (`test_short_prologue`) covers the form
    and the stricter check.
- **Target.** The programs use `v_dot4_i32_i8` (1,458 times) and `v_dot4c_i32_i8` (25,813 times), which fits the
  provider's name (`FSR4_Int8`). `llvm-mc` (LLVM 23.1.5) decodes them for `gfx1011`, `gfx1012`, `gfx1030`, `gfx1031`,
  `gfx1033`, `gfx1034` and `gfx1036`, but not for `gfx1010`, `gfx1013` (the target `shader_disasm.py` uses for the
  game) or `gfx1100`. Measured on one 10,432-byte program, prologue included: 13 invalid words with the first group,
  1,276 with `gfx1010` and `gfx1013`, 1,290 with `gfx1100`.
  - `fspx_unpack.py --disasm` defaults to `gfx1030`.
- **What llvm-mc still rejects.** With `gfx1030`, 21 of the 29 decode with no invalid word. The other 89 words, in 8
  programs, are all VOP3P opcode 0x16 with `op_sel` 2 and `op_sel_hi` 5. That is the opcode of the 1,458
  `v_dot4_i32_i8` that `llvm-mc` does decode, all of which have `op_sel` 0.
  - Measured on RDNA2 hardware (gfx1036, through `tools/hw_oracle`, the instruction given as two raw words): with
    `op_sel` 2 and `op_sel_hi` 5 the result is the dot product with the two 16-bit halves of the second source
    exchanged, in all 5 operand sets (4 of them give another word under the plain rule). So `op_sel` selects halves
    here as it does for the packed 16-bit ops. The words are in `tests/semantic/test_alu_dot4_i8.py`.
  - `tools/shader_disasm.py` (so `fspx_unpack.py --disasm` and `make_game.py`) prints such a word as
    `v_dot4_i32_i8 vD, a, b, c op_sel:[0,1,0] op_sel_hi:[1,0,1]` with its real encoding. A dword pair that llvm-mc
    rejected, with exactly opcode 0x16, `op_sel` 2, `op_sel_hi` 5, no `clamp` and no `neg`, is decoded with the two
    fields at their defaults and then given the fields and the encoding back, so the translator reads it and its
    encoding check against the blob passes. Any other `op_sel`/`op_sel_hi` value, `clamp` or `neg` is still
    undecodable (only this one was measured). A pair llvm-mc accepted, such as a literal followed by an instruction,
    is never touched. `tests/semantic/test_fspx_dot4_op_sel_disasm.py` checks it with llvm-mc (the 5 measured
    operand sets, and the forms that stay refused); the 89 words of the real programs are not re-run here, because
    that needs the package's DLL.
    When a listing lacks that pass, the translator decodes the same exact form from the original bytes on a gap
    in the listing (`rdna2spirv.skipped_form`); other undecodable words keep the reachable-code check. LLVM can
    also print the default high-half selector for an inline integer second source. If that selector is the only
    changed bit, the translator preserves the stored low-half replication (`rdna2spirv.restore_dot4_inline_sel`).
    Any other `v_dot4_i32_i8` that llvm-mc decodes but prints with different `op_sel`/`op_sel_hi` fields (for
    example `op_sel_hi` 5 with a VGPR second source) is not measured and stays refused as a dropped field
    (docs/design.md, "Signed-byte dot products").

## Kind 7: records

- **Format.** It starts with the word `0x403` in all 29, and the words `0x41`, `0x42` and `0x43` appear in all 29.
  That they tag records is inferred from the layout around them.
- **Records `0x41` and `0x42`.** The word 3 after each `0x41` or `0x42` is a register number: the 121 such numbers
  in the 29 programs are all among the register numbers the container's `RTS0` declares (measured; the check ignores
  the register class, so it is weaker than it looks). That the records list the resources the program uses is
  inferred: in one program, checked by hand, a declared SRV does not appear.
- **Record `0x43`.** The 16 bytes after it are the container's shader hash (its `HASH` part), the value embedded in
  the program's prologue.
- **Not read by the translator,** which does not read the game's FSSM binding table either. The programs load their
  descriptors with `s_load` and `s_buffer_load`, as the game's shaders do; that this is all they need is inferred.

## Build and run-time lookup

**Build.** `make_game.py`'s `shaders` stage, after the game's own shaders, runs `tools/fspx_unpack.py --disasm` on
the package's `amd_fidelityfx_upscaler_xbox.dll` into `<work>/shaders/fspx/` and `tools/rdna2spirv.py` on that
disassembly into `<work>/shaders/spirv/fspx/`, with the same `--wave64-emulate` and fallback flags. Each index entry
of a program without an FSSM blob gets two keys (`tools/shader_meta.py`):

- `fspx_code_sha256`, the SHA-256 of `fspx/<name>.code`: the whole kind-6 entry, prologue and padding included;
- `fspx_state_sha256`, the SHA-256 of `fspx/<name>.state`: the kind-0 entry (dispatch state) as it is.

The same kind-6 bytes can come with different kind-0 entries, for example two thread-group sizes, and then translate
to different modules. Such a pair is allowed: the run time tells them apart by the state key. Two entries with the
same code key and the same state key must have identical modules; when they do not, the step fails and names both
programs, and leaves no `index.json` for the runtime to load. In the 1.0.2.1 DLL the 29 kind-6 programs are 29
different byte strings, so every code key has one entry (measured).

The step is redone as a whole when the DLL, the unpacking or translating tools, `llvm-mc`, the wave64 mode or the
flags change (`fspx_build.json`, written last); it never makes the game's shaders translate again. A package without
the DLL skips it.

The executable calls slot 67 itself, for the compute pipelines of the pipeline library in `shaders.rpf`
([shader_library.md](shader_library.md)): a compute record's state blob, blob, program and layout blob have the sizes
and shapes of kinds 0, 1, 6 and 7. The next step of the same stage, `library_shaders`, unpacks those programs with
`tools/shader_library.py --stages cs` (which writes each one's `.code` and its record's state as `.state`) and
translates them into `<work>/shaders/spirv/library/`, with the same keys, the same clash check and its own
`library_build.json`.

**Run time.** `ShaderLibrary::load(dir)` (`runtime/src/shader_library.h`) reads `<dir>/index.json` and then, when they
are there, `<dir>/fspx/index.json`, `<dir>/bvh/index.json` ([embedded_shaders.md](embedded_shaders.md)), and
`<dir>/library/index.json`, whose entries are named `fspx/<name>`, `bvh/<name>`, and `library/<name>`. A missing or
unreadable one leaves the others loaded. Entries are found by `fssm_sha256` or by `fspx_code_sha256`, in two separate
tables.
`compute_program()` decides what a compute PSO's bytecode is looked up by:

| Bytecode | Key | Who passes it |
|---|---|---|
| `FSSM` blob, bare or in a `DXIL` part | `fssm_sha256` of the blob | the game (unchanged) |
| `DXBC` container whose `ACCC` part is an FSPX payload | `fspx_code_sha256` of its kind-6 bytes, then only the entries whose `fspx_state_sha256` is that of its kind-0 entry | the DLL, slot 62 |
| bare FSPX payload | the same two keys | the DLL's pipeline function accepts one (none measured at a slot) |
| bare kind-6 program: 4-byte multiple, at least 36 bytes, the prologue's words 0, 3 and 6 with bits 31..16 = `0xD580` | `fspx_code_sha256` of the bytes as they are; kind 0 is not passed | the slot-67 description at `+0x28`, size at `+0x30` |
| anything else | none: `E_INVALIDARG`, as before | |

With kind 0 (the first two rows), an input whose state key matches no entry for its code key has no program and gets
`E_INVALIDARG`, like any untranslated shader. Without it (a bare kind-6 program), the entries for the code key are
used when they all have the same module; when their modules differ, the program is ambiguous without its kind-0
state, so the call is refused with `E_INVALIDARG` and a log line saying so, instead of taking one of them.

The payload is parsed with the DLL reader's rules above (sizes and offsets checked in 64-bit arithmetic against the
clamped total), so a malformed directory, or one without kind 0 or kind 6, gives no program and `E_INVALIDARG`. The
pipeline's code base, which a translated shader reads its embedded constants through, is the kind-6 bytes: the
translator's code offsets are into `fspx/<name>.code`.

**Slot 67.** The proxy maps slot 67's description onto `CreateComputePipelineState` (#1714). When kind 0 is in the
description, which the DLL always passes (its reader refuses a payload without one), the proxy builds one FSPX
payload from the four pointer and size pairs (`leonida::fspx_payload_from_parts`) and passes that as `CS`, the
bare-payload row above. Two programs with the same code and different dispatch state therefore stay apart at slot 67
too. Each pair is checked before it is read: a pointer without a size or a size without a pointer, a size over
16 MiB, or a missing kind 6 refuses the call with `E_INVALIDARG`. Only a description without kind 0 still passes the
kind-6 bytes alone (the bare-program row).

**Slot 11.** After either path the DLL calls the pipeline's console slot 11. The Vulkan device's pipeline answers
with a blob holding the dword 0, so the DLL creates no compute scratch buffer; why 0 is justified is in
[d3d12_command_objects.md](d3d12_command_objects.md) §3.5a.

Not done here: using kind 0 at run time beyond what the translated module already carries (its thread-group size
and `RSRC2`).

**Slot 57 (`DeserializeGraphicsPipelineStateX`, graphics FSPX).** At one call site (measured locally), the retail
title calls device slot 57 (`+0x1c8`) with `IID_ID3D12PipelineState` (`{765a30f3-f624-4c6f-a828-ace948622445}`) and a
104-byte description (`+0x00..+0x60`) populated from the zlib-compressed pipeline library in `shaders.rpf`
([shader_library.md](shader_library.md): 25,471 graphics records + 2,693 compute records):

| Offset | Field | Measured layout |
|---|---|---|
| `+0x00` | `ID3D12RootSignature*` | Root signature pointer |
| `+0x08`, `+0x10` | `state_blob`, `state_blob_size` | 380..404 B (388 B in 24,797 records; 380 B in depth-only boot PSO 3) |
| `+0x18`, `+0x20` | `pipeline_blob`, `pipeline_blob_size` | 80..224 B (188 B = `0xbc` in boot PSOs 1..3); holds the `D3D12_INPUT_ELEMENT_DESC` array |
| `+0x28`, `+0x30` | `prog_a`, `prog_a_size` | First native RDNA2 kind-6 program (448 / 768 B `ps` in boot PSOs 1..2; `0/0` in boot PSO 3) |
| `+0x38`, `+0x40` | `stage_a`, `stage_a_size` | Stage blob (kind 2, 208 B in boot PSOs 1..2; `0/0` in boot PSO 3) |
| `+0x48`, `+0x50` | `prog_b`, `prog_b_size` | Second native RDNA2 kind-6 program (384 / 320 B `vs` in boot PSOs 1..3) |
| `+0x58`, `+0x60` | `stage_b`, `stage_b_size` | Stage blob (kind 2, 752 B in boot PSOs 1..3) |

- **First three boot graphics PSOs (matched in `shaders.rpf` pipeline library):**
  1. Record 8916 (`2f04ef59bc062cf8`): state 388 B (`0x4005d808`), pipeline blob 188 B (`0xea6f6e56`), `ps` prog 842 (448 B, SHA-256 `9e5ffd2c6c521c03...`), `vs` prog 843 (384 B, SHA-256 `160fbe4368fb1a63...`).
  2. Record 8917 (`f7808a66fb6b97c1`): state 388 B (`0x4005d808`), pipeline blob 188 B (`0x542385d2`), `ps` prog 844 (768 B, SHA-256 `947014e8b0771be5...`), `vs` prog 845 (320 B, SHA-256 `a2dce0214f82c742...`).
  3. Record 12511 (`e750e10eacb97b9c`): state 380 B (`0x4005d808`), pipeline blob 188 B (`0xfb724939`), no `ps`, `vs` prog 5636 (320 B, SHA-256 `f6bfddbad6aef3cd...`).
  In the library, the 448/768 B program at `+0x28` is the pixel shader (`ps`, paired with the 208 B stage blob) and the
  384/320 B program at `+0x48` is the vertex shader (`vs`, paired with the 752 B stage blob). Programs are looked up by
  `sha256(program bytes)` in `<shaders>/fspx_gfx/index.json` (`ShaderLibrary::fspx_gfx_size()`, separate from
  `fspx_size()`), and their `stage` (`"vs"` / `"ps"` / `"ms"`) is taken from the index entry rather than the blob slot.
- **Measured `pipeline blob` layout (188 B = `0xbc`):**
  - `+0x00`: u32 per-PSO checksum (`0xea6f6e56` / `0x542385d2` / `0xfb724939`)
  - `+0x04`: u32 `0x1000` (constant across 2,642 blobs), `+0x08`: `0`, `+0x0c`: `1`
  - `+0x10`: u32 per-element field-block size (`0x1c` = 28 B = 7 u32 in PSOs 1..2; `0` in PSO 3 where the stride remains 28 B)
  - `+0x1c`: u32 `0xffffffff`, `+0x24`: `0x00130100` (PSOs 1..2) / `0x00130037` (PSO 3), `+0x28`: `1`
  - `+0x38`: u32 `0x70` (or `0x6d`) = `4 + n * 28 + sum(len(name) + 1)`
  - `+0x3c`: u32 `0x0c` (constant across 2,642 blobs)
  - `+0x4c`: u32 element count (`3` across the library)
  - `+0x50`: u32 maximum name stride including NUL (`9` or `6`)
  - `+0x54..`: variable-length element array. Each element is a NUL-terminated ASCII semantic name followed by 7 u32s
    (28 B): `+0x00` `InputSlot` (`0`), `+0x04` `DXGI_FORMAT` (`6` = `R32G32B32_FLOAT` for `POSITION` at `0x54`/`0x5d`,
    `28` = `R8G8B8A8_UNORM` for `COLOR` at `0x79`/`0x7f`, `16` = `R32G32_FLOAT` for `TEXCOORD` at `0x9b`/`0xa4`),
    `+0x08` `SemanticIndex` (`0`), `+0x0c` `AlignedByteOffset` (`0`, `12`, `16`). The stored blob truncates the unused
    tail of the final 28-byte block by 1..4 bytes (`0xa4 + 28 = 0xc0 > 0xbc` in PSO 1), so the parser bounds-checks
    every 4-byte read against `blob_size`.
- **Measured `state blob` (380..388 B) and `stage blob` (kind 2):**
  - `state blob`: `+0x000` version tag (`0x4005d800` in 15,632 records, `0x4005d808` in 9,159, `0x4005d810` in 6),
    `+0x004` family (`4` in 12,735 or `6` in 12,062), `+0x02c = 2`, `+0x07c` (`0xcc0010` / `0` / `0xcc0011`),
    `+0x0f4 = 5`, `+0x0f8 = 1`, `+0x0fc = 0x33333333` (`0.1f`), `+0x100..+0x11c` (`0x00060006` x8), `+0x150 = 0x40`,
    `+0x158 = 0xffffffff`, `+0x17c = 0x10000a00`. The colour-target, depth, stencil and raster registers are decoded
    (next paragraphs); depth bias is not.
  - `stage blob`: `+0x00 = 2` (graphics stage blob), `+0x04` payload size (`196` in 208 B blobs), `+0x08 = 0x66`,
    `+0x0c = 0x40 | flags`, followed by records `0x41` (32 B input/output signature), `0x42` (32 B output signature)
    and `0x43` (20 B: tag + 16 B opaque digest).

**State blob A: the colour-target registers (RT-SLOT57-BLEND-STATE).** Blob A is the console driver's serialized
pipeline state, and its shared block holds context register values in the RDNA2 layout (Mesa's gfx10 register
headers). Measured on 1.0.2.1 over all 5,680 distinct blobs, which the 25,471 graphics records reference:

- **Shape:** 380 + 8k bytes (k = 0..5) for vertex-shader pipelines, and 464 + 8k (k = 0..7) for mesh pipelines. The
  mesh pipelines' stage block is 84 bytes longer, so every offset after `+0x28` moves by 84 there; the offsets below
  are the vertex ones. Word 0 is `0x4005d800` with bits 3 and 4 free in all 25,471 records.
- **`+0x78` `CB_TARGET_MASK`:** four component bits (R, G, B, A, as in `D3D12_COLOR_WRITE_ENABLE`) per target:
  `0xf` in 16,650 records, `0xff` in 2,805, `0xffffff` in 2,561. In 17,626 records the targets it enables are exactly
  `0..rt_count-1` of blob B. In 1,937 it also enables target 1 of a one-target pipeline: those are the dual-source
  pipelines below. The others leave a target with a format unwritten.
- **`+0x7c` `CB_COLOR_CONTROL`:** ROP3 `0xcc` (copy) in every record, so no logic op.
- **`+0xd0..+0xec` `CB_BLEND0..7_CONTROL`:** one word per target. There are 34 distinct values, every one an
  in-range RDNA2 blend: factors 0 to 20 with neither `BOTH_*` value, `COMB_FCN` 0 to 4, `SEPARATE_ALPHA_BLEND` set
  wherever `ENABLE` is, and `DISABLE_ROP3` set in all. By target 0: `0x80000000` (blending off) in 15,373 records,
  `0xe5010504` (colour `SRC_ALPHA`/`INV_SRC_ALPHA`, alpha `ONE`/`INV_SRC_ALPHA`) in 3,456, `0xf1010f01` (colour
  `ONE`/`SRC1_COLOR`, alpha `ONE`/`SRC1_ALPHA`) in 1,931, and `0xe5040504` in 1,387. 10,135 records blend on at least
  one of their targets.
- **`+0x100..+0x11c`:** eight more per-target words that follow `CB_BLENDn` (`0x06000600` where blending is off).
  They fit `SX_MRTn_BLEND_OPT`, whose `COMB_FCN` 6 is "blend disabled". They aren't read.

`runtime/src/native_pso.h` (`decode_state_a_blend`, `d3d12_blend`) maps each target's `CB_BLENDn_CONTROL` and
`CB_TARGET_MASK` nibble to a `D3D12_RENDER_TARGET_BLEND_DESC`, and `fill_blend_state` in
`runtime/shims/d3d12_xs_native_pso.h` uses it for every native PSO:

- **Before:** every target had blending off and wrote every component, so the first-run pages after #939 drew glyph
  quads solid, artwork hard-edged and composited backgrounds black.
- **With the decoded state:** the pages, the legal screens and the loading slideshow draw as on console (RTX 4070
  Laptop).
- **Integer targets:** a target with a `UINT` or `SINT` format never blends, as the colour block bypasses blending
  there and D3D12 refuses a PSO that asks for it. One library pipeline enables `MAX` on an `R8_UINT` target beside
  two others.
- **Stand-in targets:** where `B4G4R4A4_UNORM` is not renderable, the Vulkan layer draws it into an RGBA8 stand-in
  and refuses blending there (`E_NOTIMPL`, the output is quantized before it). The factory then retries the pipeline
  with blending off on those targets only (`shims/d3d12_xs_native_pso_fallback.h`), instead of leaving it to the PC
  description path. On an RTX 4070 Laptop that is the same library pipeline (the first one the title creates). Only a
  pipeline that blends a `B4G4R4A4` target is retried: `E_NOTIMPL` also stands for refusals unrelated to blending (a
  rect list without its geometry stage, mesh limits, `FullyCovered`, a dual-source output that does not fit
  `Location 0, Index 1`), and those keep the blend the title asked for.
- **Dual-source:** the 1,937 records that blend with `SRC1` factors get them as `D3D12_BLEND_SRC1_*`. The Vulkan
  layer then moves the pixel shader's `SV_Target1` from `Location 1` to `Location 0, Index 1`
  (`runtime/src/dual_source_fragment.h`, RT-DUAL-SOURCE-BLEND).
- **Kept on the defaults:** a blob of another shape, or a target with a factor or function this path does not take,
  keeps the whole pipeline on the defaults and is logged.

**State blob A: the depth, stencil and raster registers (RT-SLOT57-DEPTH-RASTER).** The same shared block, at the
same vertex-pipeline offsets (+84 in mesh pipelines). The register layouts and their checks against blob B are below,
measured over all 25,471 graphics records:

- **`+0x9c` `PA_CL_CLIP_CNTL`:** `DX_CLIP_SPACE_DEF` set in every record. Both z clips are off in 960 records and on
  in the rest, so the near and far planes always agree.
- **`+0xa0` `PA_SU_SC_MODE_CNTL`:** no culling in 13,534 records, back faces in 11,873 and front faces in 62.
  `FACE` (clockwise is front) is clear in all but 2, `POLY_MODE` is 0 in all, and the polygon-offset enables are set
  in all.
- **`+0xb8` `DB_DEPTH_CONTROL`:**
  - with no DSV in blob B, `Z_ENABLE` is clear in 7,040 of 7,079 records;
  - `STENCIL_ENABLE` is set only with formats that have stencil;
  - `DEPTH_BOUNDS_ENABLE` and bits 30-31 are clear in all;
  - `BACKFACE_ENABLE` is set in all;
  - the test is off (`0x007007f0`) in 9,062 records, otherwise mostly `GEQUAL` (a reversed depth range), with `LESS`,
    `LEQUAL`, `EQUAL` and `NOTEQUAL` in some.
- **`+0xbc` `DB_STENCIL_CONTROL`:** nonzero in 5,512 of the 6,213 records with stencil on, and in 2 without. The ops
  used are `KEEP`, `REPLACE_TEST`, `ADD_WRAP`, `ZERO` and `INVERT`.
- **`+0xc0`/`+0xc4` `DB_STENCILREFMASK(_BF)`:** the two faces have the same read and write masks in every stencil
  record, and `STENCILOPVAL` is 1 in all. `STENCILTESTVAL` (bits 7:0, the reference) is 0 in all 6,213 stencil records, front and
  back, so the reference always comes from the title (`OMSetStencilRef`) and the PSO does not need it. 245 records use a back-face function of their own.

`runtime/src/native_pso.h` (`decode_state_a_depth_raster`, `d3d12_depth_raster`) maps them to
`D3D12_DEPTH_STENCIL_DESC` and the cull, front-face and depth-clip fields of `D3D12_RASTERIZER_DESC`, and
`fill_depth_raster_state` in `runtime/shims/d3d12_xs_native_pso.h` uses them for every native PSO:

- **Before:** depth `ALWAYS` with writes wherever there was a DSV, no stencil, no culling and depth clip on.
- **With the decoded state:** every one of the 25,471 records maps. A pipeline without a DSV keeps depth and stencil
  off.
- **Not decoded:** depth bias (`PA_SU_POLY_OFFSET_*`). Its D3D12 units depend on the depth format, so it stays 0.
  The polygon-offset enables in `PA_SU_SC_MODE_CNTL` are set in all 25,471 records. The values look like the floats at
  `+0xac`/`+0xb0` (scale and offset), which are both 0 in 23,987 records and nonzero in about 1,480.
- **Front face:** `FACE` is the only field whose effect can't be seen before a world frame. It maps to
  `FrontCounterClockwise = !FACE`, as RADV and PAL program it.

## Translation status

In wave64, VALU compares and carry outputs write both predicate words. Conditional selects and carry inputs
read the corresponding word for each half-wave. Earlier LLVM listings used wave32 register spellings for these
operands. The translator uses the dispatch wave size to recover their width in cached listings.

All 29 distinct program and dispatch-state pairs in the measured DLL resolve through the compute pipeline lookup.
The eight recovered programs pass Vulkan 1.2 SPIR-V validation. The original container and the split pipeline
parts select the same module. Changing one bit of the supplied state rejects the candidate in all 29 cases.

On an RTX 4070, passes 1, 2, 7, 9, 10, 11 and 12 matched every byte of the 32 MiB reference tensor for one
workgroup. Five references came from the TuxBox native gfx1036 tensor measurements. The other two came from
an independent ISA interpreter. The postpass matched all 81,920 reference image components with a maximum
finite absolute error of 5.96e-8. These checks use deterministic inputs.

### Earlier measurements

With `tools/fspx_unpack.py --disasm` and `tools/rdna2spirv.py` on the 29 programs (`main` at `aa2988cc` with the
`v_fma_mix*` literal repair in [design.md](design.md), "Printed encodings"):
**Wave size.** On `main` at `aa2988cc`, `make_game.py`'s FSPX step translates 26 of the 29, and each now gets the
wave size of its dispatch state. Before, the code heuristic gave 4 of the 26 the other size: one wave64 program
was translated as wave32, and three wave32 programs as wave64, because they write `vcc` with `s_mov_b64`. The
translated and `spirv-val`-valid count is the same (26) either way. The breakdown below was measured when 17 translated.

With `tools/fspx_unpack.py --disasm` and `tools/rdna2spirv.py` on the 29 programs:

- **27 translate and pass `spirv-val`**, 11 of the 13 programs that use dot4 among them (design.md, "Signed-byte
  dot products"). Without the repair, 26: the `v_fma_mix_f32` it is for is refused.
- **The other 2** (both use dot4) stop at a reachable `v_dot4_i32_i8` VOP3P word with `op_sel` 2 that `llvm-mc` does
  not decode (the VOP3P words above).

These counts predate SHADER-813. A program whose reachable code falls through into those undecodable VOP3P words used
to translate with the path cut at the gap, so its later stores were missing from a module that passed `spirv-val`;
it is now refused (`fall-through into undecodable words`): programs 04, 05, 06, 09, 12 and 15 of package 1.0.2.1
(found by @aslan2142). Recount by @CuteBooty with LLVM 23.1.5 (gfx1030 listings, `spirv-val --target-env vulkan1.2`):
26 of 29 translated and passed before the change, **20 of 29** after it; the 20 kept give byte-identical SPIR-V, and
8 of the 9 refused stop at a fall-through into undecodable words.

## Unknown

- Kind 0 words 0 and 1.
- Kind 1.
- The exact layout of the kind-7 records.
- What the prologue does on the GPU.
- Whether the game itself ships FSPX programs. Its shaders measured so far are `FSSM`.
