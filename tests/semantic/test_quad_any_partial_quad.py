"""Semantic tests for s_wqm_b32 / s_wqm_b64 (Gen.quad_any) on partial quads.

In non-fragment stages (CS, VS, GS, MS), a workgroup or draw size need not be a
multiple of 4. Both OpGroupNonUniformQuadSwap (previously used in CS) and
OpGroupNonUniformShuffle over base+0..3 (previously used in VS/GS/MS) read
absent subgroup invocations on partial quads, which is undefined behavior in
SPIR-V/Vulkan and poisoned by spirv_wave_interp.

Using OpGroupNonUniformBallot + OpGroupNonUniformBallotBitExtract keeps absent
invocations at 0, so partial quads evaluate cleanly without UB.
"""

from __future__ import annotations

import os
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
HERE = os.path.dirname(os.path.abspath(__file__))
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import lds_run
import spirv_asm
import spirv_wave_interp as wi
from support import translate_lines


def _expected_wqm(mask_const: int, nlanes: int) -> tuple[int, int]:
    active_mask = mask_const & ((1 << nlanes) - 1)
    wqm = 0
    for q in range(0, 32, 4):
        if active_mask & (0xF << q):
            wqm |= 0xF << q
    wqm &= (1 << nlanes) - 1
    scc = 1 if wqm else 0
    return wqm, scc


def _run_partial_wave(spv: str, nlanes: int, builtins: dict | None = None):
    w = wi._Wave(wi._Module(spv), nlanes, None, 2_000_000, builtins=builtins)
    return w.run()


def _translate_vs_wqm32(mask_const: int) -> str:
    return translate_lines(
        "vs_partial_wqm32",
        [
            "v_and_b32_e32 v1, 31, v0",
            "v_lshlrev_b32_e32 v2, v1, 1",
            f"v_and_b32_e32 v2, {mask_const:#x}, v2",
            "v_cmp_ne_u32_e32 vcc_lo, 0, v2",
            "s_wqm_b32 s0, vcc_lo",
            "s_cselect_b32 s1, 1, 0",
            "v_cndmask_b32_e64 v3, 0, 1, s0",
            "v_mov_b32 v4, s1",
            "exp pos0 v3, v4, v1, v1 done",
            "s_endpgm",
        ],
    )


class QuadAnyPartialQuadTests(unittest.TestCase):
    def test_vs_partial_quads_execute_without_absent_lane_ub(self) -> None:
        cases = [
            (1, 0x0),
            (1, 0x1),
            (2, 0x2),
            (3, 0x0),
            (3, 0x4),
            (5, 0x10),
            (6, 0x1),
            (7, 0x20),
            (17, 0x10000),
            (32, 0x80000001),
        ]
        for nlanes, mask_const in cases:
            with self.subTest(stage="vs", nlanes=nlanes, mask=hex(mask_const)):
                spv = _translate_vs_wqm32(mask_const)
                spirv_asm.assemble(spv)
                self.assertNotIn("OpGroupNonUniformShuffle", spv)
                self.assertIn("OpGroupNonUniformBallotBitExtract", spv)
                outputs = _run_partial_wave(
                    spv,
                    nlanes,
                    builtins={"VertexIndex": lambda lane: lane, "InstanceIndex": lambda lane: 0},
                )
                want_wqm, want_scc = _expected_wqm(mask_const, nlanes)
                self.assertEqual(len(outputs), nlanes)
                for lane, out in enumerate(outputs):
                    active_word, scc_word, lane_word, _ = out[next(iter(out))]
                    want_bit = 1 if (want_wqm >> lane) & 1 else 0
                    self.assertEqual(lane_word, lane)
                    self.assertEqual(active_word, want_bit, f"lane {lane} wqm bit")
                    self.assertEqual(scc_word, want_scc, f"lane {lane} scc")

    def test_cs_partial_quads_use_ballot_and_avoid_quad_swap_ub(self) -> None:
        cases = [
            (1, 0x0),
            (1, 0x1),
            (3, 0x0),
            (3, 0x2),
            (5, 0x10),
            (7, 0x40),
            (17, 0x10001),
            (32, 0x40000002),
        ]
        for nlanes, mask_const in cases:
            with self.subTest(stage="cs", nlanes=nlanes, mask=hex(mask_const)):
                body = [
                    "v_lshlrev_b32_e32 v1, v0, 1",
                    f"v_and_b32_e32 v1, {mask_const:#x}, v1",
                    "v_cmp_ne_u32_e32 vcc_lo, 0, v1",
                    "s_wqm_b32 s8, vcc_lo",
                    "s_cselect_b32 s9, 1, 0",
                    "v_cndmask_b32_e64 v2, 0, 1, s8",
                    "v_mov_b32 v3, s9",
                    *lds_run.store("v2", 0),
                    *lds_run.store("v3", 1),
                ]
                spv = lds_run.translate(lds_run.prologue() + body, size=(nlanes, 1, 1), name="cs_partial_wqm32")
                spirv_asm.assemble(spv)
                self.assertNotIn("OpGroupNonUniformQuadSwap", spv)
                self.assertNotIn("OpCapability GroupNonUniformQuad", spv)
                self.assertIn("OpGroupNonUniformBallotBitExtract", spv)
                mem = lds_run.run(body, lanes=nlanes, size=(nlanes, 1, 1))
                want_wqm, want_scc = _expected_wqm(mask_const, nlanes)
                for lane in range(nlanes):
                    want_bit = 1 if (want_wqm >> lane) & 1 else 0
                    self.assertEqual(lds_run.out(mem, lane, 0), want_bit, f"lane {lane} wqm bit")
                    self.assertEqual(lds_run.out(mem, lane, 1), want_scc, f"lane {lane} scc")

    def test_ps_still_uses_quad_swap(self) -> None:
        spv = translate_lines(
            "ps_quad_swap_check",
            [
                "v_mbcnt_lo_u32_b32 v0, -1, 0",
                "v_cmp_eq_u32_e32 vcc_lo, 0, v0",
                "s_wqm_b32 s0, vcc_lo",
                "v_cndmask_b32_e64 v1, 0, 1, s0",
                "exp mrt0 v1, v1, v1, v1 done vm",
                "s_endpgm",
            ],
        )
        spirv_asm.assemble(spv)
        self.assertGreaterEqual(spv.count("OpGroupNonUniformQuadSwap %uint"), 2)


if __name__ == "__main__":
    unittest.main()
