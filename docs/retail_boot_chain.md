# The retail boot chain: what stops `gta6rt --run`, in order

**Status: measured on one machine, 2026-11-23. A measurement for RT-5, not a claim; RT-5 stays with its owners
on the coordination issue.** Each row is the first failure with every fix above it in place. Nothing here quotes
the executable: no addresses, offsets, IID values or bytes; only export names, public SDK types, vtable slot
numbers, call shapes and counts. [boot.md](boot.md) has the runbook that produced the traces; #6715 maps the same
executable under `--dry-run`.

Setup: `main` `1cc6e3c12` with #6709, #4531, #6582, #4313, #4348 and #2490 merged locally, plus the local
stand-ins named below (not pull requests yet). Own 1.0.2.1 copy, Windows 11, RTX 4090 (driver 617.42),
`LEONIDA_D3D12=vulkan`, MSVC 2022 `RelWithDebInfo` build.

## 1. The chain

| # | Stop | What clears it |
|---|---|---|
| 1 | The first `XMemVirtualAlloc` calls: two 64 KiB reserve forms, then 2 MiB pages | #6709, #4531 |
| 2 | The 2 MiB reserve+commit calls carry attribute words the shim refuses, and `PAGE_GRAPHICS_*` bits (0x800..0x40000) on top of `PAGE_READWRITE` | Local: accept the attribute word, drop the graphics bits and keep the CPU protection. All seven startup allocations then succeed. #6843 measures the attribute words; no PR changes the check yet |
| 3 | `xgameruntime.lib`'s thunk treats a missing `XErrorReport` as fatal | #6582 |
| 4 | Access violation in the title's allocator: its `/GA` code reads its TLS from module index 0 | #4313 in the measured tree: the runtime's own EXE gives up index 0 at process start. `main` has a different fix since `b7a3fed28` (with `4710549af` for threads that were already running): the loader rewrites the executable's index-0 reads so that they reach the image's own block ([boot.md](boot.md) §3, step 3). Checked on `main` with a synthetic `/GA` executable beside a DLL with TLS of its own (`loader_tls`) and a `/GA` build of the stand-in game under `gta6rt --run` (`run_tiny_game_ga_tls`); the retail run was not repeated for this note, it needs game files |
| 4a | Wine only, not reached on Windows: `GameInput.dll` ordinal 1, the executable's `GameInputCreate` import (by ordinal, not by name), which the loader looks up as `gameinput.dll!#1`. When `gta6rt` doesn't serve that name, the import binds to the host's `GameInput.dll`. On Windows the installed one answers; under Wine it is Wine's builtin stub, and the title reads through a null pointer right after the call. Measured by hcristosm after stop 4 with stop 5's fix (old upstream PR 4348) merged, so its order against stop 5 is not measured. #127 reproduced it on Wine 11.21 staging: the title's own `int3` 0.65 s after start, before any D3D12 call | #127 (the `#1` alias in the shim table, `runtime/shims/gameinput.cpp`), but only in a `gta6rt` built with the GameInput shim, that is with SDL3: an installed SDL3 3.2 or newer that CMake finds, `-DLEONIDA_SDL3_FETCH=ON` (`scripts/build_runtime.ps1 -FetchSdl3`) or `-DFETCHCONTENT_SOURCE_DIR_SDL3`. A build without SDL3, one with `-DLEONIDA_SDL3=OFF`, and `build_runtime.ps1`'s `cl.exe` fallback have no `GameInputCreate` at all ([gameinput.md](gameinput.md#building)), so they still stop here under Wine. With the shim and the alias, the Wine run went on to stop 6 in hcristosm's measurement |
| 5 | Fatal assert when `DStorageGetFactory` returns null | #4348 |
| 6 | `XMemAllocatePhysicalPages` unimplemented; the title asserts on FALSE | Local stand-in, §2 |
| 7 | `D3D12XboxCreateDevice` succeeds; device slot 26 is called with its arguments swapped | Local, §3.1 |
| 8 | Device slot 48 is forwarded to the PC slot 48 and fails | #6954 |
| 9 | Device slot 73 is forwarded to the PC slot 73; the title then reads through a null list | #2490's list proxy plus local routing, §3.3 |
| 10 | An `XMemVirtualAlloc` reserve of about 185 MiB with type `0x42000`, attribute 7 and `PAGE_NOACCESS` is refused with `ERROR_INVALID_PARAMETER`. `0x40000` is `MEM_RESERVE_PLACEHOLDER` on the PC, which `VirtualAlloc` refuses and only `VirtualAlloc2` accepts; stop 11 supports the same reading on the console | Local: reserve it with `VirtualAlloc2(MEM_RESERVE \| MEM_RESERVE_PLACEHOLDER)` |
| 11 | The title carves five sub-ranges out of that placeholder with `MEM_RESERVE \| MEM_REPLACE_PLACEHOLDER \| MEM_PHYSICAL \| MEM_LARGE_PAGES` (`0x20406000`) at addresses inside it, then allocates a second batch of physical pages | Local: a physical reserve inside a known placeholder stays a placeholder (the page maps split it); other `MEM_REPLACE_PLACEHOLDER` calls split the range and allocate it in place |
| 12 | Device slot 133 is called; the proxy table had 128 slots, so the call went through a null pointer | Local: a 192-slot table, and a slot 133 stand-in, §3.4 |
| 13 | **Current stop**: right after the first `CreateDescriptorHeap`, the title calls the heap's slot 9 (`GetCPUDescriptorHandleForHeapStart` on the PC), takes the handle from `rax` and asserts that it equals a value it already holds. Hypothesis: a console descriptor heap starts at an address the title chose or can predict, which the PC handle doesn't match | Open: a question for the descriptor model ([boot.md](boot.md) Q4) |

Row 4a is the one row measured under Wine (by hcristosm on Wine 9.0, also reproduced on Proton Experimental; by
#127 on Wine 11.21 staging), not on the setup above. It is numbered 4a so that the Windows stops keep their numbers.

At stop 13 the D3D12 trace holds 1,185 calls: the device, 5 command queues, 5 fences, 7 command allocators,
581 command lists created and closed, a shader cache session, `XboxIDXGIDevice::GetAdapter`, one failed
`ID3D12Device10::CreateReservedResource2`, the first `CreateSampler2`, one committed resource (the slot 133
stand-in) and the first descriptor heap.

Without a debugger, the unhandled `int3` of a fatal assert ends the process with `C0000374` (heap corruption) or
`C0000005` during teardown, not `80000003`. An exit code of `C0000374` from `--run` can be such an assert, not a
heap bug.

## 2. Physical pages (`xmem.dll`)

The shapes, from the call sites and the checks that follow them:

| Export | Shape |
|---|---|
| `XMemAllocatePhysicalPages` | `BOOL (ULONG64 attributes, ULONG_PTR* page_count /*in/out, 64 KiB pages*/, ULONG_PTR* page_array)`; the title checks both TRUE and the count written back. Attributes seen: `0x280`, `0x380` |
| `XMemVirtualAlloc` | the range for those pages is reserved with `MEM_RESERVE \| MEM_PHYSICAL \| MEM_LARGE_PAGES` (`0x20402000`) and no `MEM_COMMIT` |
| `XMemMapPhysicalPages` | `(va, page_count, page_array)`; a null `page_array` unmaps; non-null return on success |
| `XMemMapPhysicalPagesScatter` | `(va_array, page_count, page_array)`, the same with one VA per page |
| `XMemFreePhysicalPages` | `(page_count, page_array)` |

The local stand-in that gets past stop 6 is CPU-only: one pagefile section per 64 KiB page
(`CreateFileMappingW(INVALID_HANDLE_VALUE, …, SEC_COMMIT)`), page numbers handed out as indexes into that table,
the physical reserve made with `VirtualAlloc2(MEM_RESERVE | MEM_RESERVE_PLACEHOLDER)`, each map a
`VirtualFree(MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER)` split followed by `MapViewOfFile3(MEM_REPLACE_PLACEHOLDER)`,
and each unmap `UnmapViewOfFile2(MEM_PRESERVE_PLACEHOLDER)`. #5381 has the CPU half of physical pages with dirty
tracking and #5378 a Vulkan sparse-binding mapper; neither is wired to these exports.

## 3. Device calls whose Xbox shape differs from the PC

### 3.1 Struct-returning methods take the return buffer first

Device slot 26 (`GetCustomHeapProperties`, which returns a `D3D12_HEAP_PROPERTIES`) is called as
`(return buffer, this, node_mask, heap_type)`: the first register holds the return buffer, the second the device
proxy. The PC MSVC method expects `(this, return buffer, …)`. A proxy thunk can tell the two apart: in the swapped
call the `this` register does not point at the proxy, and the second argument does. Calling the real method with
the two exchanged and returning the buffer pointer gets past it. Other struct-returning slots probably follow the
same rule; only slot 26 was reached.

### 3.2 Device slot 48 creates a command queue

Xbox device slot 48 is `(desc, riid, out)` with `riid = IID_ID3D12CommandQueue`. `desc` starts with a PC
`D3D12_COMMAND_QUEUE_DESC` (Type, Priority, Flags, NodeMask) followed by Xbox words. On the PC, slot 48 is
`OpenExistingHeapFromAddress`, so forwarding by number fails. Creating the queue from the 16-byte PC prefix, with
`Flags` limited to `D3D12_COMMAND_QUEUE_FLAG_DISABLE_GPU_TIMEOUT`, through the existing queue proxy works: four
queues come from this slot, the first of them a compute queue. The first queue comes from slot 8, as on the PC.

### 3.3 Device slot 73 creates a command list

Xbox device slot 73 is `(desc, allocator, pso, riid, out)`. `desc` is 16 bytes and its first word reads as the
`D3D12_COMMAND_LIST_TYPE`. `riid` is a console command-list IID that is not in the PC SDK (#3520 records that the
package doesn't export it). A retail run logged it as `{c3827890-4024-474c-98cf-645b51dc4874}` (#445). Creating an
`ID3D12GraphicsCommandList` of that type and handing out #2490's console list
proxy gets all 581 lists created and closed.

### 3.4 Device slot 133 places a resource at a virtual address

Xbox device slot 133 is `(va, const D3D12_RESOURCE_DESC*, state, clear, riid, out)` with `riid = IID_ID3D12Resource`
and `state = 0xFFFFFFFF`, which isn't a valid PC state. At the first call, `va` is the start of a physical-page
range the title had just reserved, and `desc` is a buffer of that range's size with Xbox bits in `Flags` above
the PC flags. The shape matches the console `CreatePlacedResourceX`. A buffer on those pages is imported with
`VK_EXT_external_memory_host` where the driver allows it, so the title's writes are the GPU's reads; a driver that
refuses falls back to a committed buffer. A texture gets a committed texture whose `GetGPUVirtualAddress` is the
title's address (the console behavior: a title clears a texture through that pointer), but its pixels do not alias
the title's pages yet. Both paths limit `Flags` to the PC bits; the buffer path also makes the Xbox layout word
`ROW_MAJOR`.

The proxy's slot table must also be longer than the title's highest slot. A shorter table leaves a null pointer,
not an `E_NOTIMPL`.

## 4. What this does not show

These are one run's first failures, not a list of what is missing. Every stand-in above is local and untested
beyond "the title goes on to the next call". Nothing has reached a draw or a present yet.

## 5. The device slots the executable calls, from the file (RT-566)

The chain above finds the next slot by running the title; `tools/d3d12_device_calls.py` reads the file alone,
without running anything, and reports the statically resolved `ID3D12Device` vtable call sites for the device paths
it follows:

```sh
python -I tools/d3d12_device_calls.py <game>/game_scarlett_master.exe --out <work>/device_calls
```

It writes `<exe>.device_calls.txt` (slot numbers and counts only) and `.json` (the same, plus each site's RVA for
local follow-up; don't paste that part) into the `--out` folder. It reads the executable only, so a slot called only
from a bundled DLL is absent from it: slot 76, for one, isn't in the census, yet `docs/d3d12_vulkan.md` (the slot 76
row of its contract table, and §4.2) records the FidelityFX upscaler DLL creating its resources through it. A slot
missing from the census is not a slot the title never uses. The method, in its docstring: it finds the
`D3D12XboxCreateDevice` call sites, follows the out-pointer in `r9` to the global or stack slot it fills and the
globals the device is copied to, and counts per slot the `mov vtable,[device]` → `call [vtable+disp]` (or
`call reg`, or a call to a dispatch routine with the method in `rax`) sites, with `disp / 8` as the slot. A
dispatch routine counts only when its code reaches `jmp rax` or `call rax` without writing `rax`, calling or
returning first; a call to any other target with a method in `rax` is reported as not resolved. Per slot it shows
calls and tail jumps, whether the device is in `rcx` or in `rdx` (the return-buffer form of §3.1), and an
argument-count estimate. Slots 0-43 carry the PC method names, which the console keeps where measured (8, 9, 14, 26,
36); past 43 only the slots `runtime/shims/d3d12_xs_device.h` and §3 name are named.

The walk follows forward branches within a function (a `jmp rel32` there is a branch, not a tail call) and keeps
a fact at a join only when both paths agree. What it doesn't see, and says so: the device reached through an object
member, a function parameter or a getter's return value; a path it loses, such as a loop's back edge; an indexed
method load. The report counts the vtable loads with no matched call, the calls with a method in `rax`
to a target that isn't a verified dispatch routine, the calls that are handed the device (not followed into), and
the stores of the device to non-stack memory. A slot can be called more often than the report says, or on a
path it doesn't follow, so a slot with no row (12 and 51, for example) isn't shown to be uncalled.

On the 1.0.2.1 executable it finds one `D3D12XboxCreateDevice` call site writing a global and 95 sites across
42 slots: 94 indirect vtable calls and one indirect vtable tail jump (at slot 2), with no dispatch call resolved;
the `rdx` form at slots 25, 26 and 108 only; and 12 sites it reports as not resolved. Every device slot
the chain above reached (8, 26, 48, 73, 133) is among them; slots like 49, 55, 57, 61, 70, 71, 74, 81, 86, 101,
108, 119, 127 and 132 are called but have no name in this repository yet.

For some of the unnamed slots there is a call shape from an independent static read with a different scanner, by
TreXito (in the review of old upstream PR 7998), with the same site counts for 49, 50, 55, 57, 71 and 81. These
are their reads, not this tool's:

| Slot | What the call is (TreXito's read) |
|---:|---|
| 49 | Only behind an OS-version gate; no run reaches it today |
| 50 | Only on a fatal path that ends in `int3` |
| 55 | `CreateDerivedPipelineState(base, n, records, riid, out)`, on a PSO-cache hit (old upstream PR 7759) |
| 57 | `(desc, IID_ID3D12PipelineState, out)`, the native-record graphics pipeline |
| 81 | `(object, a constant flags argument, &count, out)`; the result is used only on success |
