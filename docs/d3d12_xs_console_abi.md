# The D3D12.X console ABI: vtable slots, object fields and record encodings

Status: measured-facts note, sourced. This page collects the Xbox Series (`d3d12_xs`) ABI facts a
replacement runtime must reproduce — vtable slot numbers, object field layouts and the inline
record encodings the console SDK writes into a command-list object — for the layers that are still
missing: command lists, root arguments, descriptor heaps and present. It complements
[d3d12_vulkan.md](d3d12_vulkan.md) (the Vulkan layer plan) and
[d3d12_command_objects.md](d3d12_command_objects.md) (what this package's own DLLs measure).

Provenance rules for this page, stated once:

- **Sibling layer** — measured by a sibling open-source layer on retail Series titles and
  published as findings. Not independently reproduced here; treat every such claim as unverified
  on this package.
- **Package** — measured on this package's shipped DLLs (`XboxBvhBuilder.dll`,
  `amd_fidelityfx_upscaler_xbox.dll`); see
  [d3d12_command_objects.md](d3d12_command_objects.md) §§3.3–3.5a.
- **SDK** — the public Windows SDK `d3d12.h` order, for contrast.
- **Unconfirmed** — consistent with the surrounding facts but not yet measured anywhere we can
  check.

All slot numbers below are absolute vtable indices including `IUnknown` (0–2) and the
`ID3D12Object`/`ID3D12DeviceChild`/`ID3D12Pageable` base slots, as in
[d3d12_command_objects.md](d3d12_command_objects.md) §2.

## 1. Device (`ID3D12DeviceX` over the PC `ID3D12Device` layout)

The console device keeps the PC slot order where measured, with extra console methods placed in
slots the PC chain uses for newer interfaces, plus a tail of console-only methods. Slots marked
**Package** were called by the shipped upscaler or BVH-builder DLLs with the stated shapes; the
rest are **Sibling layer** measurements.

| Slot | Method | Evidence |
|---:|---|---|
| 8, 48 | `CreateCommandQueue` variants | Sibling layer |
| 9 | `CreateCommandAllocator` | Sibling layer |
| 10, 60 | `CreateGraphicsPipelineState` variants | Sibling layer |
| 12 | `CreateCommandList` | Sibling layer |
| 14 | `CreateDescriptorHeap` | Package + sibling layer |
| 15 | `GetDescriptorHandleIncrementSize` | Package + sibling layer |
| 17, 18, 19 | `CreateConstantBufferView` / `CreateShaderResourceView` / `CreateUnorderedAccessView` | Package + sibling layer |
| 22 | `CreateSampler` | Sibling layer |
| 23, 24 | `CopyDescriptors` / `CopyDescriptorsSimple` | Package + sibling layer |
| 25 | `GetResourceAllocationInfo` | Sibling layer |
| 29, 44 | `CreatePlacedResource` variants | Sibling layer |
| 38 | `GetCopyableFootprints` | Package + sibling layer |
| 41 | `CreateCommandSignature` | Sibling layer |
| 52 | `SetDebugFlags` | Sibling layer |
| 55 | derived pipeline state | Sibling layer |
| 56, 57, 66, 67 | serialize / deserialize pipeline state | Sibling layer; the package's upscaler calls 67 as `DeserializeComputePipelineStateX` (console desc, `IID_ID3D12PipelineState`) — Package |
| 61 | `CreateSamplerX` | Sibling layer |
| 62 | `CreateComputePipelineState` with two extra args: `(desc, NULL, NULL, riid, ppv)`; the PC interface presents `CreateStateObject` at this slot | **Confirmed here**: Package ([d3d12_vulkan.md](d3d12_vulkan.md) §7) + sibling layer |
| 73 | `CreateCommandListX` | Sibling layer |
| 74, 75 | register / unregister page pool | Sibling layer |
| 76 | `CreateCommittedResource`, same signature family as PC slot 27 | **Confirmed here**: Package ([d3d12_vulkan.md](d3d12_vulkan.md) §7) + sibling layer |
| 78–81 | frame pacing: `SetFrameIntervalX`, `ScheduleFrameEventX`, `WaitFrameEventX`, `GetFrameStatisticsX` | Sibling layer |
| 86 | `CreatePipelineStateX` | Sibling layer |

The slot-76 console resource descriptor is **larger than 52 bytes** (sibling layer); the package's
upscaler DLL writes a 3 at offset +52 (Package). The `Layout` field sits at +44; value `0x101`
denotes a 2D texture ([d3d12_vulkan.md](d3d12_vulkan.md) §7; a sibling layer measured the same on
retail titles).

### 1.1 Two runtime layouts

Two console runtime layouts exist: an older game-OS layout and a newer one in which the queue and
list slots shift **+2** (sibling layer). Detection rule (sibling layer): if a fence object is
passed to queue slot 16/17, the layout is newer — on the older layout those slots are
`GetTimestampFrequency`/`GetClockCalibration`. Which layout this package's titles were built
against is **unverified here**.

## 2. Descriptor heap

| Slot | Fact | Evidence |
|---:|---|---|
| 9, 10 | `GetCPUDescriptorHandleForHeapStart` / `GetGPUDescriptorHandleForHeapStart` return the 8-byte handle in `rax` on console; the PC convention is a hidden out-pointer | **Confirmed here**: Package + sibling layer |

Descriptor contents (sibling layer): CBV/SRV/UAV heaps use **32-byte slots** in unified memory — a
T# is 8 dwords, a V# 4 dwords, inside the 32-byte slot; sampler heaps use **16-byte slots**. The
CPU handle, GPU handle and the address in title memory are the same value.

## 3. Resource

| Slot | Fact | Evidence |
|---:|---|---|
| 11 | `GetGPUVirtualAddress` returns the title-memory address | Sibling layer; consistent with the package measurement that a CPU pointer from list slot 148 is passed unchanged as a root SRV GPU address ([d3d12_command_objects.md](d3d12_command_objects.md) §3.3a) |

## 4. Command queue

Measured queue slots (sibling layer); the queue follows the two-layout rule of §1.1:

| Method | Older layout | Newer layout |
|---|---:|---:|
| `Signal` | 14 | 16 |
| `Wait` | 15 | 17 |
| `GetTimestampFrequency` | 16 | 18 |
| `GetClockCalibration` | 17 | — (shifted) |
| `GetDesc` | 18 | — (shifted) |
| `SignalX` | 23 | 25 |
| `PresentX` | 30 | 32 |

Console-only queue methods the layer does not model should return `S_OK` and log the call:
title frame pacing flows through some of them (sibling layer).

## 5. Command list object

Measured by a sibling layer with a fake list — a 600-slot vtable of logging thunks — that let the
console SDK's inline writers record into the object (sibling layer; the package's BVH builder and
upscaler independently measured `+0x18`, `+0x28`, `+0x30`, `+0x140`, `+0x141`, `+0x344` and
`+0x544` — Package, [d3d12_command_objects.md](d3d12_command_objects.md) §§3.4–3.5a).

### 5.1 Object fields

| Offset | Field | Evidence |
|---:|---|---|
| +0x00 | vtable pointer | Package + sibling layer |
| +0x18 | inline write pointer | Package + sibling layer |
| +0x28 | dispatch-record limit | Package + sibling layer |
| +0x30 | buffer end | Package + sibling layer |
| +0x140 | dirty flag byte, set when record header bits 31:30 are 0 | Package + sibling layer |
| +0x141 | pipeline flag byte, taken from bit 20 of the pipeline's record word at +0x20 | Package + sibling layer |
| +0x143 | flag byte; read by titles, written only by the runtime | Package (read 31+1 sites, never written) + sibling layer |
| +0x344 + 4·i | header word of root parameter *i* | Package + sibling layer |
| +0x544 | current pipeline record, 16 bytes | Package + sibling layer |

### 5.2 Inline record encodings

The console SDK's inline writers emit a record only while the write pointer is below the buffer
end; on a full buffer they call the vtable method the record stands for (Package + sibling layer).

| Record | Encoding | Evidence |
|---|---|---|
| Pipeline | 28 B: `[PSO ptr][D3D12Core ptr][8 B flags][4 B]` | Package + sibling layer |
| Root parameter *i* | header at +0x344+4·i = `0xC0000000 \| i<<10` + const-offset·4 (bits 9:2) + `(count-1)<<16` | Sibling layer; header table itself confirmed Package |
| Root descriptor | header + 64-bit address = 12 B | Package + sibling layer |
| Root constants | one constant = 8 B; *n* constants = 4+4·*n* B | Package + sibling layer |
| Descriptor table | header + one 32-bit word = the 32-bit GPU address of the first descriptor | Sibling layer |
| `Dispatch` | `0x4002C400` (or `(0x1000BA00 \| flag) << 2`) followed by x, y, z = 16 B | Package + sibling layer |
| Indirect dispatch | `0x4003CC00`, 20 B total | Package (five-word form with W `0x8020`/`0x20`) + sibling layer |
| `IASetPrimitiveTopology` | `0xC000B600 \| topology` | Sibling layer |

### 5.3 Method slots

| Slot | Method | Evidence |
|---:|---|---|
| 9 | `Close` | Sibling layer |
| 10 | `Reset` | Sibling layer |
| 12 | `DrawInstanced` | Sibling layer |
| 13 | `DrawIndexedInstanced` | Sibling layer |
| 14 | `Dispatch` | Package + sibling layer |
| 15 | `CopyBufferRegion` | Package + sibling layer |
| 16 | `CopyTextureRegion` | Sibling layer |
| 17 | `CopyResource` | Sibling layer |
| 19 | `ResolveSubresource` | Sibling layer |
| 20 | `IASetPrimitiveTopology` | Sibling layer |
| 25 | `SetPipelineState` | Package + sibling layer |
| 26 | `ResourceBarrier` | Package + sibling layer |
| 32 | `SetDescriptorHeaps` | Package + sibling layer |
| 33 | `SetComputeRootSignature` | Package + sibling layer |
| 37, 39 | `Set*RootConstant(s)` | Package + sibling layer |
| 41, 43, 45 | `Set*RootDescriptor*` / table | Package + sibling layer |
| 48 | `IASetVertexBuffers` | Sibling layer |
| 51 | `ClearDepthStencilView` | Sibling layer |
| 71-74 | memory writes and waits (newer layout; 69-72 older) | **Measured here**, §5.5 |
| 80 | `ExecuteIndirect` | Sibling layer (note: the package measured slot 80 as a console-only call, a `Dispatch`-with-extra-word shape; slot assignments past 61 differ between the two sources — unconfirmed) |
| 83 | `CopyTextureRegionX` (older layout) | Sibling layer; the package measured a `CopyTextureRegion`-shaped call at slot 85 — consistent with the newer layout (+2) |
| 108 | `ResolveSubresourceRegion` | Sibling layer |
| 139 | `Reset(allocator, pso)` → PC slot 10 | **Measured here**: retail 1.0.2.1 (Pipodi) — the title creates a list, `Close`s at slot 9, then calls 139 with `(allocator proxy, 0)` before recording; the proxy unwraps the allocator and forwards to PC `Reset` |
| 146 | enhanced `Barrier` | **Confirmed here**: Package (console slot 146 takes one `D3D12_BARRIER_GROUP`; PC equivalent is GCL7 slot 80) |
| 148 | `AllocateUploadMemory` `(size, alignment)`; the returned CPU pointer is the GPU address in unified memory | Package + sibling layer |

Layout shift on the list tail (sibling layer): from `CopyTextureRegionX` on, methods sit two slots
later in the newer layout — newer→older pairs 85→83, 86→84, 87→85, 90→88, 91→89, 92→90, 95→93,
96→94.

The shim hands out console-shaped command lists by default: `GetCommandList`-shaped queries build
the proxy with the console vtable above. `LEONIDA_D3D12_LISTS=pc` makes it hand out PC-shaped lists
instead, for guests compiled against the PC headers (the `tiny_frame` test fixture calls through PC
vtable indices; without the variable its `ClearRenderTargetView` at PC slot 48 lands on the
console's `IASetVertexBuffers` and `Close` never runs). The device, queue, heap and fence stay
console-shaped either way.

### 5.4 DMA command list

| Slot | Method | Evidence |
|---:|---|---|
| 9 | `Close` | Sibling layer |
| 10 | `Reset` | Sibling layer |
| 11 | `CopyBufferRegion` | Sibling layer |
| 12 | `CopyTextureRegion` | Sibling layer |
| 13 | `CopyResource` | Sibling layer |

### 5.5 Memory writes and waits

The list has bottom-of-pipe writes of a 32- or 64-bit value into title memory and waits that stop the queue until
a 32- or 64-bit value in memory compares true against a reference. Neither has a D3D12 or Vulkan command; the
runtime models both (`runtime/shims/d3d12_xs_list_values.h`, the list cut and `execute_with_values` in
`d3d12_xs_device.cpp`).

| Older slot | Newer slot | Method | Evidence |
|---:|---:|---|---|
| 68 | 70 | depth bounds `(min, max)` as floats | Sibling layer; the retail title calls 70 (unmodelled, a no-op) |
| 69 | 71 | `write32(address, value, flush, flags)` | Sibling layer; **measured here** on retail 1.0.2.1 |
| 70 | 72 | `write64(address, value, flush, flags)`: flags 1 stores the GPU timestamp, 2 the system one (QPC) | Sibling layer; **measured here** (flags 1 and 0 seen) |
| 71 | 73 | `wait32(address, comparison, reference, word)` | Sibling layer; not called by the retail title in the boots measured |
| 72 | 74 | `wait64(address, comparison, reference, word)` | Sibling layer; **measured here** |
| 73 | 75 | pixel-shader depth order (state) | Sibling layer; the retail title calls 75 (a no-op) |

Measured here (retail 1.0.2.1, boots to the story load; temporary per-call logging, not committed):

- The lists record before the queue proxy has detected the layout (§1.1): the first value calls arrive with the
  layout still unknown. The runtime takes an undetected layout as the newer one, the title's, as it already does for
  `ExecuteIndirect` at 61.
- Up to the story load (~150 s): about 12 000 `write32` (small counters and a constant tag in a ring of 72-byte
  records), 16 000 `write64` (all with flags 1, GPU timestamps, in begin/end pairs) and 4 100 `wait64`, one per
  frame, `>=` a reference that rises by one each frame, at the start of a direct list, always met when the list is
  submitted (the CPU writes that value).
- From the story load on: job chains across the queues. Compute lists (the title uses five compute queues) and
  direct lists `wait64(address, >=, tag)`, mostly as their first calls and sometimes after two to five calls of
  work, for addresses that other lists `write64(address, tag)` with flags 0: direct lists write tags as their first
  calls and wait for compute results; compute lists wait for those tags and write theirs near their end. Before
  this runtime modelled the writes, every such wait was unmet at submission and ran on.
- Comparison code 7 is the only one seen; the codes are taken as `D3D12_COMPARISON_FUNC` (memory OP reference, as
  the sibling layer does). The wait's fourth word had 1 in its low 32 bits each time; it is not used.

The runtime's model:

- A write lands after the list part it was recorded in has run: the queue proxy signals a fence of its own behind
  the part and a pump thread stores the value with the CPU once the fence is reached
  (`leonida::kernel_writes::store`: one locked access for an aligned value, so a title thread polling it never
  reads half of it, and pinned like any host write, so the slot-133 shadows see it). Later than the console, never
  earlier. Writes land in submission order across queues: the pump reads the fences newest write first, so a write
  whose work waited for an older one is never seen done before it.
- Two targets that store cannot reach (`runtime/src/value_write_sink.h`). An address in the GPU address arena's
  range is an arena offset, a default-heap buffer, not a host address: it is asked before any host store and lands
  in the arena's own mapping (one in the arena's reserved start or a sparse buffer is dropped). A page of xmem's that
  the title's view keeps `PAGE_NOACCESS` (a GPU-only window), where the store faults, lands through xmem's own
  mapping of the physical page. Counted as `writes_arena` and `writes_title_pages`. In the boots to the story load no
  write failed the host store (0 of 63 219), so both are a safety net.
- `write64` flags 1 and 2 take the GPU's timestamp at the write's place in the list: an `EndQuery(TIMESTAMP)` on
  the real list (the Vulkan layer writes it at the bottom of the pipe) and a `ResolveQueryData` of that one slot into
  a readback ring of the device's, read when the write lands. Flag 1 lands those ticks, which count at the queue's
  `GetTimestampFrequency` (what the queue proxy answers the title's frequency query with, newer layout slot 18);
  flag 2 lands the `QueryPerformanceCounter` value at which the GPU took them, through the queue's clock
  calibration (renewed every second), never later than the landing. A write that lands before its part has run
  (recorded before a wait whose gate holds the part) takes the CPU's clock at landing instead, as before. The
  title's GPU timing reads these values back, in begin/end pairs.
- A wait recorded before any work of the part stands at the part's start; writes recorded before that first wait
  land before the queue stands at it (behind the work submitted before the part), as the title's frame lists and
  job-tag writers need. A wait recorded after work, or after a write that follows a wait, cuts the list:
  the real list so far is closed as a part, recording goes on in a spare real list on the same allocator, and the
  state set so far (heaps, root signatures and arguments, pipeline, viewports, scissors, targets, vertex and index
  buffers, blend factor, stencil reference, predication; what an executed bundle set, and nothing from before a
  `ClearState`) is set on it again. A cut that would split an open query waits for the query's `EndQuery`.
  Bundles record neither method (`ExecuteBundle` names one real list, so a bundle cannot be cut).
- At submission each part's waits are checked: met ones cost nothing; one that a write still in flight will meet
  becomes a queue `Wait` on that write's fence (the GPU orders the two queues); any other makes the queue `Wait` on
  a gate fence that is signalled once memory compares true. The Vulkan layer holds the queue's later work behind
  that wait without blocking the submitting thread (`src/queue_holds.h`). The pump only decides; a second thread,
  the releaser, signals the gates, because that Signal runs the work the layer held behind them (the held
  submission, the shadow refresh, a deferred Present), and the pump keeps landing writes and timing gates out
  meanwhile.
- A gate opens after `LEONIDA_VALUE_WAIT_TIMEOUT_MS` (default 200) whatever memory holds, counted from when the
  gates before it on its queue opened, and logged once per address. An address that timed out three times in a row
  (the last 256 addresses are counted) gets 4 ms. Once four gates in a row timed out, so does every wait for an
  address no wait was ever met for, until a gate cut short is met within its 4 ms; a wait cut short that memory
  meets within the rest of the full timeout gives its address the full timeout from then on.
  `LEONIDA_LIST_VALUES=0` drops both methods, as before. After `stop_list_values` (the process is ending) nothing
  new is landed or opened.
- Queue slot 31 (`CopyPageMappingsBatchX`, section 4) waits at most 16 ms for the queue's earlier work. Work held
  behind a gate takes longer: the remap is then deferred to the releaser, which runs it once the queue's write
  fence passes the call's point and only then opens a gate the queue's later work waits on, so the GPU sees the
  order of the call and the title's thread does not wait (the console's CPU does not either).

## 6. Root arguments

Sibling-layer measurements on retail titles:

- **Compute.** The root arguments are a packed argument array in title memory; `io[0:1]` of the
  dispatch record carries its address.
- **Graphics.** VS and PS receive root arguments in user SGPRs `s8..s31` — `s0..s7` are
  driver-reserved; `s[4:5]` of a VS holds the fetch shader — 24 user SGPRs in total. Arguments
  past those go through a table whose address is in `s[0:1]`.
- **Descriptor tables** take 4 bytes in the packed array: the record is the parameter header plus
  one 32-bit word, the 32-bit GPU address of the first descriptor.
- **Root CBV.** The shader builds its own V#; a measured example sets `DST_SEL xyzw` plus the
  format word from `s11`.

## 7. Resource barriers (list slot 26)

Sibling layer: console aliasing barriers behave as "any resource" barriers — both resource
pointers are null — where the PC API only allows those between placed or reserved resources.
Console-only state bits must be dropped before forwarding, or `Close` fails `E_INVALIDARG`.

## 8. Unified memory

Titles expect graphics memory where the CPU address equals the GPU address (sibling layer;
consistent with the package's slot-148 measurement in §3).

- The sibling layer backs this with a fully-committed sub-4 GiB region at a 4 GiB-aligned virtual
  address, opened via `ID3D12Device3::OpenExistingHeapFromAddress`. Measured limits: a partial
  commit is rejected; a full 4 GiB is rejected; 4095 MiB works.
- For reservations above 4 GiB the measured technique is: reserve with `VirtualAlloc2`
  `MEM_RESERVE_PLACEHOLDER`, split chunks with `VirtualFree` `MEM_PRESERVE_PLACEHOLDER`, back each
  chunk with a `SEC_COMMIT` section mapped via `MapViewOfFile3` `MEM_REPLACE_PLACEHOLDER`. Each
  chunk opens as its own heap with GPU/CPU coherence at the title's address. `SEC_RESERVE`
  sections are refused even when fully committed.
- Buffers in unified memory are placed resources at 64 KiB alignment. A buffer at a
  non-64 KiB-aligned console address is placed from the 64 KiB boundary below it and the copy
  paths add the delta.
- `Map` of a unified-memory resource returns its console address.
- GPU-owned textures (resources with RT/DS/UAV flags) go into "arenas" — default heaps standing
  for ranges of title addresses — because titles alias several textures at one address.

## 9. `PresentX`

Sibling layer: one frame is presented per console queue method call (`PresentX`, §4). The
presented back buffer is detected by tracking render-target binds (RTV creation and
`OMSetRenderTargets`) and `ExecuteCommandLists`.

## 10. Shader-interface facts

Measured on real console shaders by a sibling layer; where the claims overlap this repository's
translator docs they are mostly confirmed here.

- An FSSM stage descriptor at +0x110 looks like `RSRC2`: `USER_SGPR` sits in bits 5:1 (e.g.
  `0x18` → 12 SGPRs). A PS header carries an export-variant patch table from about +0x25C.
- User SGPRs (a compute example with 14 per `RSRC2`): `s4` = descriptor table 0, `s5` = table 1,
  `s[6:7]` and `s[8:9]` = root CBVs; `s14`/`s15` = thread-group id x/y.
- Mesh-shader interface: `s3[27:24]` = wave index in the group, `v5` = group·128 + thread,
  `s16..s24` = user data.
- Console GS runs on-chip: the ES writes vertices to LDS and `s_setpc`s to the GS. The GS writes
  vertices to LDS at 64 B per vertex, base = `s2 + ([s8 + 0xB8] << 8)`, and uses `s_sendmsg`
  emit/cut/done. No copy shader is present in the pack.
- `IASetPrimitiveTopology` value 17 is rectangle lists, drawn as two triangles on PC.
- Derivatives: console code computes cross-quad derivatives manually (DPP `quad_perm`,
  `ds_swizzle`) and reads helper lanes. On PC, reads across quads need quad-aware lane reads:
  `WaveReadLaneAt` leaves helper lanes out before SM 6.7, and enabling
  `WaveOpsIncludeHelperLanes` caused a device hang in the sibling layer — whole-wave ballots then
  never let helper lanes exit.
- A pixel shader doing `v_readfirstlane` after `s_wqm` can read a helper lane's value; the read
  must take the value with a quad read inside its own quad plus a wave read from a live lane of
  that quad.
- Memory model: a shared "mem" buffer used for flat accesses must be coherent
  (`globallycoherent` on D3D12). Bottom-up BVH builds with atomics produce wrong child pointers
  otherwise; console code relies on GLC/cache-invalidation semantics.
- ISG1 inputs a VS never reads get no fetch-shader VGPRs.

## 11. Unverified here

The sibling-layer measurements above were made on retail Series titles and that project's own
host runtime, not on this package or this layer. Still to be measured on this package:

- **which runtime layout this title uses** — the older/newer detection rule of §1.1 needs a queue
  call trace;
- **draw-record encodings for graphics** — only compute dispatches and copies were measured; the
  graphics draw records and their sizes are unknown;
- the `PresentX` signature and plane struct (no signature is recovered anywhere;
  [d3d12_command_objects.md](d3d12_command_objects.md) §4);
- queue/fence/allocator slots and the device creation slots 8/9/12/36 for the layout this title
  uses ([d3d12_command_objects.md](d3d12_command_objects.md) §5.3);
- the exact newer→older slot pairs on the command-list tail beyond the measured set.
