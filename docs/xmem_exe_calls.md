# XMem: how the retail executable calls it (RT-5)

`game_scarlett_master.exe` imports 12 functions from `xmem.dll` (`docs/runtime_imports.md`, "The retail executable").
This page records, from the file alone, how the executable calls them: how many call sites each has and the constant
arguments those sites pass. Nothing was run: `tools/import_call_sites.py` finds the call sites, and the instructions
that load the argument registers before each call were read by hand. No code or data of the executable is quoted, only
the constants a call passes and the counts. The call sites were measured at `main` `1cc6e3c1`, SHA-256
`e95020b2fa4a721db3df59fdecf75621730ac559d8835252f6c3b658d5a08993`; the shim column below is the current state, and
it has moved since then: `042c5f378` (#6919) gave every one of the 12 imports a shim (seven had none at `1cc6e3c1`).

```powershell
python -I tools/import_call_sites.py game_scarlett_master.exe --dll xmem.dll --dll d3d12_xs.dll --dll xg_xs.dll
```

The tool finds each import's thunk (`jmp [rip+disp32]` through its address-table slot) and the `call`/`jmp rel32`
instructions that reach it. Its counts cover direct calls only, and a stray `E8`/`E9` byte could add one, so they are
bounds, not a proof that every site is reached.

| Import | Call sites | Shim at `main` (since #6919) |
|---|---:|---|
| `XMemVirtualAlloc` | 23 | yes, all 23 types served (below) |
| `XMemMapPhysicalPages` | 12 | yes (pagefile-backed 64 KiB pool) |
| `XMemAllocatePhysicalPages` | 4 | yes |
| `XMemFreePhysicalPages` | 3 | yes |
| `XMemMapPhysicalPagesScatter` | 2 | yes |
| `XMemAlloc` | 3 | yes |
| `XMemFree` | 3 (2 calls, 1 tail jump) | yes |
| `XMemGetWorkingSetStatistics` | 3 | yes |
| `XMemTransferMemory` | 1 | yes |
| `XMemIsToolsMemAvailable` | 1 | yes (returns `FALSE`) |
| `XMemAllocDefault`, `XMemFreeDefault` | 1 tail jump each, no call | yes |

The executable also exports one symbol, `__XMemAllocationHooks__`.

## `XMemVirtualAlloc`: every site passes an attribute word

The 23 sites pass these constants (`allocation_type`, `attributes`, `protect`; the shim's parameter order is
`address, size, allocation_type, attributes, protect`):

| `allocation_type` | `attributes` | `protect` | Sites |
|---|---|---|---:|
| `0x20403000` | `0x181` | `0x4` | 3 |
| `0x20402000` | `0x21` | `0x4` | 3 |
| `0x20003000` | `0x183` | `0x22004` | 3 |
| `0x20402000` | `0x3` | `0x801` | 2 |
| `0x20402000` | `0x23` | `0x2004` | 2 |
| `0x20402000` | `0x23` | `0x2404` | 1 |
| `0x20003000` | `0x283` | `0x2004` | 1 |
| `0x20003000` | `0x283` | `0x2404` | 1 |
| `0x20401000` | `0x283` | `0x42404` | 1 |
| `0x20406000` | `0x27` | `0x1004`, `0x1404`, `0x8404`, `0x2404`, `0x2004` | 5 (one each) |
| `0x42000` | `0x7` | `0x1` | 1 |

Every `attributes` value is nonzero. Since #6919 `runtime/shims/xmem.cpp` does not decode them: `XMemVirtualAlloc`
records the word (`xmem_attributes` in the startup trace) and accepts it. All 23 allocation types are served as well.
The page-mode mask is `kXmem64kPages` `0x20400000`, which *is* `MEM_LARGE_PAGES | MEM_PHYSICAL`, so `0x20406000`
(`0x20400000 | MEM_RESERVE | MEM_REPLACE_PLACEHOLDER`) selects the 64 KiB page mode and replaces part of the
placeholder — `test_placeholders` in `runtime/test/xmem_test.cpp` allocates that exact shape and maps physical pages
into the carved window. The 23 sites divide as 17 in the 64 KiB mode (3 committed, 8 reserve-only windows, 1 plain
commit, 5 placeholder replacements), five in the 2 MiB mode `0x20000000` (committed 2 MiB-aligned through
`VirtualAlloc2`), and one `0x42000` = `MEM_RESERVE | MEM_RESERVE_PLACEHOLDER`, the ~185 MiB reservation the five
windows are carved out of (`docs/retail_boot_chain.md`, stops 10-11). That agrees with `docs/runtime_imports.md`
("The retail executable", `xmem.dll` notes), which records the same 2 MiB and placeholder paths.

The `protect` values are not Win32 protections alone. The Windows SDK's `winnt.h` defines the graphics page protections
the high bits match: `PAGE_GRAPHICS_NOACCESS` `0x800`, `_READONLY` `0x1000`, `_READWRITE` `0x2000`, `_EXECUTE_READ`
`0x8000`, `_COHERENT` `0x20000`, `_NOCACHE` `0x40000`, beside `PAGE_WRITECOMBINE` `0x400`. Read that way, `0x2004` is
`PAGE_READWRITE` plus `PAGE_GRAPHICS_READWRITE`, and `0x801` is `PAGE_NOACCESS` plus `PAGE_GRAPHICS_NOACCESS`. That is
an inference from the constants matching the SDK's names, not something the executable states. What the `attributes`
bits mean (`0x1`, `0x2`, `0x4`, `0x20`, `0x80`, `0x100`, `0x200` appear) is **unmeasured**.

## The physical-page family

Four imports share one shape, and it is the Win32 AWE family's (`AllocateUserPhysicalPages`, `MapUserPhysicalPages`,
`FreeUserPhysicalPages`, `MapUserPhysicalPagesScatter`) with the process handle replaced or dropped. These signatures
are **inferred** from the registers loaded before each call; no documentation of these functions is public:

| Function | Inferred signature | Evidence |
|---|---|---|
| `XMemAllocatePhysicalPages` | `BOOL (ULONG_PTR flags, ULONG_PTR* pages, ULONG_PTR* pageArray)` | 4 sites. `pages` points to a stack slot holding the wanted count; `pageArray` is a buffer of `count * 8` bytes. One site compares the count read back with the wanted one, so a partial allocation is possible. `flags` is `0x380`, `0x280` (2 sites) and `0x180` |
| `XMemMapPhysicalPages` | `BOOL (void* virtualAddress, ULONG_PTR pages, const ULONG_PTR* pageArray)` | 12 sites. Four pass a null `pageArray` (an unmap); one maps one page through a one-element array; the result is tested at the sites that read it |
| `XMemFreePhysicalPages` | `? (ULONG_PTR pages, ULONG_PTR* pageArray)` | 3 sites. The count is passed by value (one site frees the sum of four counts it had just unmapped), the array by pointer. Only two of the three sites were read to the argument, so this one is the least certain |
| `XMemMapPhysicalPagesScatter` | `BOOL (void** virtualAddresses, ULONG_PTR pages, const ULONG_PTR* pageArray)` | 2 sites. Shaped like the Win32 call; the result was not followed |

The page unit looks like **64 KiB**: the code that unmaps shifts byte sizes right by 16 before it passes page counts,
and the reservations that precede the map calls use the 64 KiB page mode (`0x20402000`, above). That is inferred from
those two facts. The meaning of `flags`, and whether the page numbers in `pageArray` carry anything beyond an index,
are **unmeasured**.

The sequence in the first function that calls them is: `XMemAllocatePhysicalPages`, then `XMemVirtualAlloc` of a range
with `0x20402000` and attributes `0x23`, then `XMemMapPhysicalPages` of those pages at that range. That path no longer
stops at `XMemVirtualAlloc`: since #6919 it accepts that attribute word and the reserve-only 64 KiB window.

The host's own AWE functions need `SeLockMemoryPrivilege`, which an ordinary user does not hold, so the shim emulates
the pool rather than forwarding (#6919): a pagefile-backed section in 64 KiB pages, committed at once on Wine (and on demand on native Windows) and given
physical memory as it is touched, with views placed at the reserved range. Pages keep their contents while unmapped or
mapped elsewhere, and one page can be mapped at several addresses. The attribute word is recorded, still not decoded,
and every page is 64 KiB, which all four callers' arithmetic agrees with (`runtime/shims/xmem.cpp`).

## The other imports

Observed constants, where the window before the call loads one:

| Import | Sites | Constants seen |
|---|---:|---|
| `XMemAlloc` | 3 | attributes `0xf0814000` (2 sites); one size `0x60000`; one size `0x80` |
| `XMemFree` | 3 | attributes `0x3` and `0xf0814000`; one attribute not a constant |
| `XMemTransferMemory` | 1 | direction `1`, size `0x1000000` (16 MiB) |
| `XMemGetWorkingSetStatistics` | 3 | kind `0x2000000a` (2 sites), `0x20000001` |
| `XMemIsToolsMemAvailable` | 1 | arguments not read |

`runtime/shims/xmem.cpp` says one call site "passes `0x5e50000`" as the attribute word. The two `XMemAlloc` sites here
whose attribute is a constant pass `0xf0814000`, not that value, so it comes from another image.

## What this does not show

Which of these calls the boot reaches, and in what order, is not measured: nothing ran. A call site is code that can
call, and the boot may reach only some of them. The first run's trace (`--trace`, `xmem_attributes` events) is the
signal for the order.
