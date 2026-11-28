# XDXR build descriptions: measured layouts

This answers open question 8 of [d3d12_vulkan.md §8.2](d3d12_vulkan.md): the layouts of
`D3D12XBOX_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_BATCHED_DESC2`, `D3D12XBOX_XDXR_STREAM_DESC` and the offline
statistics postbuild info. They are recovered from how the package's own `XboxBvhBuilder.dll` (1.0.2.1, the
884,224-byte file in `game/xbox-1.0.2.1.json`) reads them, not guessed and not taken from NDA headers. The C++
definitions are in [`runtime/src/xdxr_descs.h`](../runtime/src/xdxr_descs.h), checked by `xdxr_descs_test`.

**Method.** Static analysis of the DLL with Ghidra 12.2 (through REA 6.10.1): the field accesses, element strides,
copies and validation branches of the functions below. Addresses are RVAs (image base `0x180000000`). Nothing from
the DLL is reproduced here except type names that its own exported symbols carry; check a claim by opening the
function at its RVA in your own copy.

**Errors are thrown, not returned.** Every rejection below in the build and prebuild-info exports stores
`E_INVALIDARG` and throws it as an MSVC C++ exception through the `_CxxThrowException` thunk at `0x2302e`, and those
exports return `void`: the `Dest` alignment check in `0x216a0` (branch at `0x217a3`, throw at `0x21876`), the
`Inputs.Type` check in `0xcc00` (branch at `0xce0c`, throw at `0xd578`) and the postbuild-type check in `0xe6a0` (§4).
Only `ValidateRehydrateAccelerationStructureBuildDesc` (`0x21a00`) returns `E_INVALIDARG` as an `HRESULT`. Checked
with `llvm-objdump` on the 1.0.2.1 file (RT-155).

| RVA | Function | What it shows |
|---|---|---|
| `0x216a0` | `BuildRaytracingAccelerationStructureBatched2X` (export) | Element stride `0x50`; `Dest` alignment check; postbuild loop |
| `0x21610` | `BuildRaytracingAccelerationStructure` (export) | Wraps one standard build into a `BATCHED_DESC2` on the stack |
| `0x21880` | `GetRaytracingAccelerationStructurePrebuildInfo` (export) | Builds a `BATCHED_DESC2` from `INPUTS` + `XDXR_STREAM_DESC` |
| `0xc750` | internal accessor | Element `i` is `i * 0x50` into the `BATCHED_DESC2` array, or `i * 0x40` into an internal array without the stream desc |
| `0xcc00` | internal build-info setup | Reads `Inputs.Type`, `Inputs.Flags`, `Scratch`, the stream desc and the stream records |
| `0xe6a0` | internal batched build | Rejects the offline-statistics postbuild type |

## 1. `D3D12XBOX_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_BATCHED_DESC2` (`0x50` bytes)

The first `0x30` bytes are the public `D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC`: the standard-build
export at `0x21610` copies the caller's public desc there unchanged (six 8-byte words), puts its postbuild count
and pointer at `+0x30`/`+0x38`, zeroes `+0x40..+0x50` and calls the batched export with a count of 1.

| Offset | Size | Field | Evidence |
|---|---|---|---|
| `0x00` | 8 | `DestAccelerationStructureData` (GPU VA) | `0x216a0`: when postbuild info is requested, a nonzero low byte throws `E_INVALIDARG` (not 256-byte aligned) |
| `0x08` | 24 | `Inputs` (public `D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS`) | `0x21880` copies a caller's `INPUTS` here (3 words) |
| `0x08` | 4 | `Inputs.Type` | `0xcc00`: 0 and 1 accepted, anything else throws `E_INVALIDARG` |
| `0x0c` | 4 | `Inputs.Flags` | `0xcc00` tests bits 0 (`ALLOW_UPDATE`), 2 (`PREFER_FAST_TRACE`), 4 (`MINIMIZE_MEMORY`), 5 (`PERFORM_UPDATE`) |
| `0x10` | 16 | `Inputs.NumDescs`, `DescsLayout`, `InstanceDescs`/`pGeometryDescs` | passed as `&desc + 8` to the per-type counters |
| `0x20` | 8 | `SourceAccelerationStructureData` | copied from the public desc by `0x21610` |
| `0x28` | 8 | `ScratchAccelerationStructureData` | read by `0xcc00` |
| `0x30` | 4 | `NumPostbuildInfoDescs` | loop bound in `0x216a0` and `0xe6a0` |
| `0x34` | 4 | padding | `0x21610` writes 0; no read found |
| `0x38` | 8 | `pPostbuildInfoDescs` (public 16-byte `POSTBUILD_INFO_DESC` array) | `0x216a0` reads `InfoType` at entry `+8`, stride 16 |
| `0x40` | 16 | `D3D12XBOX_XDXR_STREAM_DESC` (§2) | `0x21880` copies a caller's stream desc here (16 bytes) |

The internal `0x40`-byte variant that the accessor also accepts has the same fields up to `0x40` and no stream desc.

## 2. `D3D12XBOX_XDXR_STREAM_DESC` (`0x10` bytes)

| Offset | Size | Field | Evidence |
|---|---|---|---|
| `0x00` | 8 | `SizeInBytes` | `0xcc00`: 64-bit load; the walk ends at `pStream + SizeInBytes` |
| `0x08` | 8 | `pStream`: an array of 16-byte records (§3) | `0xcc00`: walked in 16-byte steps |

The shape is that of `D3D12_PIPELINE_STATE_STREAM_DESC`. A zero size or a null pointer means every record's default.
`GetRaytracingAccelerationStructurePrebuildInfo` takes the same struct as its second argument.

## 3. Stream records (16 bytes each)

Every record starts with a 32-bit type. To read a setting, `0xcc00` walks the stream from the start and uses the
**first** record of the wanted type. It steps over types 0, 1 and 2 and **stops at any other type**, so a record of an
unknown type hides everything after it. The record names and the meaning of their values are not in the DLL; the
offsets, types, defaults and the conditions under which they are read are.

| Type | Offset | Type of value | Default | Read when | Observed use |
|---|---|---|---|---|---|
| 0 | `0x04` | `float` | -1.0 | always | copied into the per-build info |
| 0 | `0x08` | `float` | -1.0 | always | negative means a default from `Inputs.Flags` (below); above 10.0 selects a different bottom-level path when `MINIMIZE_MEMORY` is set |
| 0 | `0x0c` | 4 × `u8` | 0 | always | copied into the per-build info byte by byte |
| 1 | `0x0d` | `u8` | a global | bottom level (`Type` 1) only | 0 keeps the default; otherwise clamped to 15 (N), reserves `N * N * 0x90` extra bytes and is stored as a 4-bit field |
| 1 | `0x0e` | `u8` (bool) | 0 | top level (`Type` 0) only | nonzero clears one bit of the internal build flags |
| 2 | `0x08` | `float`, used as an integer | 0.2 | always | truncated to an integer; -1 means the default, otherwise the integer is used |
| 2 | `0x0c` | `float` | 0.75 | always | 0.0 means the default |

Bytes `0x04..0x0c` and `0x0f` of type 1 and `0x04..0x08` of type 2 are not read by this DLL.

Default for type 0's `0x08` value when it is negative (or when there is no type-0 record):

| `PREFER_FAST_TRACE` | `MINIMIZE_MEMORY` | Value |
|---|---|---|
| set | clear | 0.0 |
| set | set | 1.005 |
| clear | clear | 1.005 |
| clear | set | 1.01 |

## 4. Postbuild info types

| `InfoType` | In `XboxBvhBuilder.dll` |
|---|---|
| 0 `COMPACTED_SIZE`, 3 `CURRENT_SIZE` | written by one handler |
| 2 `SERIALIZATION` | written by another handler |
| 1 `TOOLS_VISUALIZATION` and other values | ignored, no error |
| `0x10` `D3D12XBOX_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_OFFLINE_STATISTICS` | thrown as `E_INVALIDARG` by `0xe6a0` before any work, for every build: the DLL only supports it for offline BVH builds |

**The offline statistics layout is not recoverable from this DLL**: it never writes them. It would have to come from
the offline builder that produces dehydrated BVHs, which is not in the package. A runtime that implements the builder
should reject `0x10` the same way.

RT-155 re-checked this on the same file (SHA-256 as in the manifest) with `llvm-objdump` on Linux. Every use of the
type is one of two checks, and neither reads past the `InfoType` field:

- `0xe6a0`, which the batched export calls first (`0x216dc`), loops over every element's postbuild descs (count at
  `+0x30`, array at `+0x38`, stride 16) and compares `InfoType` (`+8`) with `0x10`. A match branches to the error
  block at `0x106ba`, whose third message names the type, and the build throws `E_INVALIDARG` as a C++ exception
  (`0x10702`) before any work; the export returns `void`.
- The batched export's own postbuild loop (`0x2177b`..`0x217e9`) skips `0x10` explicitly, then sends 0 and 3 to
  `0x1d540` and 2 to `0x1d7c0`. Every other value falls through to the next desc.

None of the 15 other DLLs in the package names the type or an offline builder.

## 4a. The runtime serves the DLL itself

`runtime/shims/xbox_bvh_builder.cpp` serves the seven exports the executable imports with the layer's own XDXR
builder (`runtime/src/xdxr_build.*`), so the bundled DLL is not loaded and its 52 console programs are never created.
Their pipeline compile (31.2 s for one of them on an RTX 4080, 59.5 s for all 52 on an RX 9070 XT, #7231/#7049) runs
on the title's thread, longer than its ~30 s startup watchdog, so a cold first boot ends at the builder (boot stop 18).
`LEONIDA_BVH_BUILDER=console` turns the shim off.

The executable's imports and call sites, measured statically on 1.0.2.1. The counts are `call [rip+disp32]`
through the import table. For `Batched2X`, IDA finds 8 direct calls: 4 in each of two sibling functions with no
static reference, reached through a table (#7516), so 2 of them are calls of another form.

| Export | Call sites | The shim |
|---|---:|---|
| `XboxBvhBuilder(ID3D12Device11*, bool)` | 1 | keeps the device in a table keyed by `this`; writes nothing into the title's object |
| `~XboxBvhBuilder()` | 1 | releases that entry |
| `Set64BitMortonCodes(bool)` | 8 | recorded; the layer's builder has no such option |
| `GetRaytracingAccelerationStructurePrebuildInfo(INPUTS*, XDXR_STREAM_DESC*, PREBUILD_INFO*)` | 3 | the layer's sizes |
| `BuildRaytracingAccelerationStructureBatched2X(GCL6*, UINT, BATCHED_DESC2*)` | 6 (8) | one layer build per element (§1), with its postbuild descriptions |
| `CopyRaytracingAccelerationStructure(GCL4*, dest, source, mode)` | 2 | the layer's copy |
| `RehydrateRaytracingAccelerationStructureBatched2X(GCL8*, UINT, BUILD_DESC*, UINT64*)` | 4 | rebuilds each structure from source geometry; refuses source-only stream without geometry |

**Where it differs from the DLL:**
- **Stream records (§3):** they tune the console builder's heuristics, so they are ignored (logged once). The result
  is still a valid BVH of the same geometry, in the node format the traversal shaders read.
- **Validation errors:** the DLL throws them as C++ exceptions. A shim must not throw into the title, so an element
  the DLL would refuse is skipped with a log line. The offline-statistics postbuild type (`0x10`, §4) is one of
  those.
- **Rehydration (`RehydrateRaytracingAccelerationStructureBatched2X`):** Rebuilds each structure on the GPU from
  the source geometry its description carries ([bvh.md](bvh.md) §4a). The leaf frame `compressionParams` gives is
  **not applied yet**: the leaves are built from the geometry as the description has it.
  What the 4th parameter (`const UINT64* source_data`, mangling `PEB_K`) conveys is a **hypothesis**: it points to
  the array of source buffer addresses for offline dehydrated console streams (e.g. rehydration inputs).
  When geometry is provided (`NumDescs > 0`), the shim issues `BuildRaytracingAccelerationStructure` to construct
  traversable GPU BVH nodes. That is a fresh build: the title sets `SourceAccelerationStructureData` to the
  dehydrated stream, so the shim clears it and the `PERFORM_UPDATE` flag (logged once), which would otherwise make
  the layer refit from the stream's bytes. The title passes `ScratchAccelerationStructureData = 0` (measured on the retail
  title), so the shim supplies scratch, or the layer would reject the build: one new buffer per call, in which each
  element has its own 256-byte-aligned region sized via `GetRaytracingAccelerationStructurePrebuildInfo`. The
  layer's XDXR builder leaves the dependency between builds to the caller's barrier and the title puts none between
  the elements of a batch, so no two builds, in a batch, on one list or on lists recorded in parallel, share scratch
  bytes. The layer's command list holds the buffer until it is Reset or released, and the layer destroys it after
  the submission that used it has retired. When `source_data` is supplied without geometry, the shim logs once and
  refuses with `E_NOTIMPL`: offline console dehydrated streams cannot be directly deserialized by PC Vulkan/D3D12
  drivers ([bvh.md](bvh.md) §4/§4a). Empty structures (`NumDescs == 0` without source) are logged and skipped.

`xbox_bvh_builder_test` calls every export through the shim registry, the way the title's imports bind, and
`xdxr_bvh.py check-build` checks the BLAS, TLAS and compacted BLAS it builds (ctest `xbox_bvh_builder_check`).

## 5. What is still open

- The names of the stream record types and of their values.
- The offline statistics layout (§4). It is not in the package: the one DLL that knows the type only rejects or
  skips it (RT-155). A runtime doesn't need it, because the shipped builder refuses the request.
- Whether the game passes a non-empty stream desc at all. That needs the executable's call sites (RT-5).
