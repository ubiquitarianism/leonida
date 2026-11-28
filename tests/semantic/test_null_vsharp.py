"""Semantic regression tests for a null V# (all four words zero): it reads as a zero buffer, not memory at address 0.

The D3D12 layer leaves a slot all zeros for a null SRV/UAV/CBV, an empty heap slot, or a buffer view it refuses
(runtime/shims/d3d12_vulkan_device.cpp, CreateShaderResourceView / CreateUnorderedAccessView /
CreateConstantBufferView), and a title's own empty slot holds zeros too. A shader that loads that slot as a V# gets
base 0, NUM_RECORDS 0, OOB_SELECT 0 (structured with offset), FORMAT 0 (INVALID) and DST_SEL 0 everywhere. Every
access is then out of range (docs/buffer_bounds.md: mode 0 with N = 0, s_buffer_load with a limit of 0 bytes): loads
and typed loads give 0 without touching memory, and stores and atomics leave memory alone, which is a null
descriptor's behaviour on the console. The translator needs no rule of its own for this; these tests pin it, so a
later change to the bounds checks cannot turn a null V# into a read or write at address 0.
"""
from __future__ import annotations

import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

sys.dont_write_bytecode = True
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import support  # noqa: E402
import test_value_semantics as tv  # noqa: E402

OUT = 0x2000
OUT_V = 0x31016FAC  # RAW output V#, 0x100 bytes at OUT
SENTINEL = 0x5A5A5A5A
SEED = 0x12345678

# 8 user SGPRs (s[0:3] the output V#, s[4:7] the null V#), no TGID SGPRs, 64x1x1 threads.
CS_8_USER_FSSM = tv._fssm(8 << 1, (64, 1, 1))

PROGRAM = [
    "v_cmpx_eq_u32_e32 0, v0",
    "v_mov_b32 v1, 0",
    f"v_mov_b32 v2, {SEED:#x}",
    f"v_mov_b32 v3, {SEED:#x}",
    f"v_mov_b32 v4, {SEED:#x}",
    f"v_mov_b32 v5, {SEED:#x}",
    f"v_mov_b32 v6, {SEED:#x}",
    f"v_mov_b32 v7, {SEED:#x}",
    f"v_mov_b32 v8, {SEED:#x}",
    f"v_mov_b32 v9, {SEED:#x}",
    f"v_mov_b32 v10, {SEED:#x}",
    "v_mov_b32 v11, 0xdead",
    "v_mov_b32 v12, 3",
    # Raw and structured loads, a typed load and a typed tbuffer load, all through the null V#.
    "buffer_load_dword v2, v1, s[4:7], 0 offen",
    "buffer_load_dwordx2 v[3:4], v1, s[4:7], 0 idxen",
    "buffer_load_format_xyzw v[5:8], v1, s[4:7], 0 idxen",
    "s_waitcnt vmcnt(0)",
    "buffer_store_dword v2, off, s[0:3], 0",
    "buffer_store_dwordx2 v[3:4], off, s[0:3], 0 offset:4",
    "buffer_store_dwordx4 v[5:8], off, s[0:3], 0 offset:12",
    # Stores and an atomic through the null V#: memory at its base 0 must keep the sentinel.
    "buffer_store_dword v11, v1, s[4:7], 0 offen",
    "buffer_store_dword v11, off, s[4:7], 0 offset:8",
    "buffer_atomic_add v12, v1, s[4:7], 0 offen glc",
    "s_waitcnt vmcnt(0)",
    "buffer_store_dword v12, off, s[0:3], 0 offset:28",
    # A scalar constant-buffer load (a null CBV).
    "s_buffer_load_dwordx4 s[16:19], s[4:7], 0x0",
    "s_waitcnt lgkmcnt(0)",
    "v_mov_b32 v13, s16",
    "v_mov_b32 v14, s17",
    "v_mov_b32 v15, s18",
    "v_mov_b32 v16, s19",
    "buffer_store_dwordx4 v[13:16], off, s[0:3], 0 offset:32",
    "s_endpgm",
]


class NullVsharpTests(unittest.TestCase):
    def test_null_vsharp_reads_zero_and_drops_stores(self):
        spv = support.translate_lines("cs_null_vsharp", PROGRAM, fssm=CS_8_USER_FSSM)
        low = {4 * i: SENTINEL for i in range(8)}  # what a read or write at the null V#'s base 0 would touch
        mem = tv.ReadLog(dict(low))
        support.run_fragment(
            spv,
            inputs={"%wg_id": (0, 0, 0), "%local_id": (0, 0, 0)},
            push_constants=[OUT, 0, 0x100, OUT_V, 0, 0, 0, 0] + [0] * 24,
            memory=mem,
        )
        # Nothing was read: not the null V#'s base, nor anything else.
        self.assertEqual(mem.reads, [])
        # The loads gave zeros: raw, structured, typed (FORMAT 0 and DST_SEL 0 select 0), and the scalar load.
        self.assertEqual([dict.get(mem, OUT + 4 * i) for i in range(7)], [0] * 7)
        self.assertEqual([dict.get(mem, OUT + 32 + 4 * i) for i in range(4)], [0] * 4)
        # The atomic did not happen; what it returns (out of range) is the hardware's 0.
        self.assertEqual(dict.get(mem, OUT + 28), 0)
        # Memory at address 0 is untouched, and nothing outside the output was written.
        self.assertEqual({a: v for a, v in mem.items() if a < OUT}, low)
        self.assertTrue(all(OUT <= a < OUT + 0x100 for a in mem if a not in low))


if __name__ == "__main__":
    unittest.main()
