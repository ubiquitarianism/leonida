"""SHADER-W64-WQM-UVOTE: a vote on a compare of scalars under whole-quad EXEC is no lane vote (tools/wave64_halves.py).

A pixel shader that needs derivatives starts with `s_wqm_b64 exec, exec`: EXEC becomes the whole quads of the launch
mask, helper lanes included. LLVM then branches on wave-uniform conditions inside that region the way it does
anywhere else: `v_cmp_* sN, <scalars>` / `s_and_b64 vcc, exec, s[N:N+1]` / `s_cbranch_vccnz`. SHADER-10 accepted that
form only while EXEC was the launch mask itself and called it `unsafe-vote-branch` here.

It is as exact under the whole quads. The compare reads immediates and scalars only, so it is the same in every
lane: its mask is every lane of EXEC or none. A quad is four consecutive lanes and never straddles the two 32-lane
halves, and a half that runs has a launch lane, so EXEC has a lane in every half that runs: each half votes as the
wave does, and either side of the branch may define scalars that are read after the join.

What stays tier C, and why (`NearMisses`): a VGPR source, even one SHADER-10 calls uniform, because it was written
under the launch mask and the helper lanes never got it; whole quads of a narrowed EXEC, or a narrowing after the
`s_wqm_b64`, because a half can then be empty; a scalar that was read out of such a VGPR with v_readlane or, under
whole-quad EXEC, with v_readfirstlane, because the lane read may be one the launch mask did not write; `s_wqm_b64`
into a register that is copied to EXEC afterwards (not modelled).

Hand-written programs, no game data. Classification in both printed forms of the compare's destination (the game's
listings name one register, SHADER-W64-SDST-PAIR); execution on the strict wave interpreter (spirv_wave_interp): a
vertex shader and a pixel shader as one wave of 64 lanes and as two of 32, equal and equal to the words derived here.
The GPU twin is runtime/test/shaders/ps_w64_a_wqm_uniform_vote.asm (w64_graphics.py).

    python -I tests/semantic/test_wave64_halves_wqm_uvote.py
"""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import support  # noqa: E402,F401  (sets the tools path)
import spirv_wave_interp as wi  # noqa: E402
from test_wave64_halves import UNIFORM, meta_of, out_param, ps, run_cut, vs  # noqa: E402

WQM = ["s_wqm_b64 exec, exec"]
# if 5 < s4: v6 = 7, s20 = 1, else v6 = 5, s20 = 2; s20 is read after the join
SIDES = ["v_mov_b32_e32 v6, 5", "s_mov_b32 s20, 2", "s_branch 2", "v_mov_b32_e32 v6, 7", "s_mov_b32 s20, 1"]
ONE_REGISTER = ["v_cmp_lt_u32_e64 s14, 5, s4", "s_and_b64 vcc, exec, s[14:15]", "s_cbranch_vccnz 3"] + SIDES
PAIR = ["v_cmp_lt_u32_e64 s[14:15], 5, s4"] + ONE_REGISTER[1:]
VCC = ["v_cmp_lt_u32_e64 vcc, 5, s4", "s_cbranch_vccnz 3"] + SIDES          # the compare straight into VCC
READ = ["v_add_nc_u32_e32 v7, s20, v0"]


class Classification(unittest.TestCase):
    def tier(self, name, lines, stage=ps):
        (tier, reasons, idioms), _ = meta_of(name, stage(lines))
        return tier, reasons, idioms

    def test_a_scalar_compare_under_whole_quads(self):
        for name, vote in (("one_register", ONE_REGISTER), ("pair", PAIR), ("vcc", VCC)):
            with self.subTest(name):
                self.assertEqual(self.tier("ps_wqm_uv_" + name, WQM + vote + READ), ("A", [], []))

    def test_a_exec_restored_after_the_vote(self):
        # the usual shape: derivatives region, then EXEC back to the launch mask saved at the top (PRO saves it)
        self.assertEqual(self.tier("ps_wqm_uv_restore", WQM + ONE_REGISTER + ["s_and_b64 exec, exec, s[10:11]"] + READ),
                         ("A", [], []))

    def test_a_uniform_loop_back_edge(self):
        # s22 counts the trips in an SGPR across the back edge: every half runs s4 of them, as the wave does
        self.assertEqual(self.tier("ps_wqm_uv_loop", WQM + [
            "s_mov_b32 s22, 0", "s_add_u32 s22, s22, 1", "v_add_nc_u32_e32 v6, 3, v6",
            "v_cmp_gt_u32_e64 vcc, s4, s22", "s_cbranch_vccnz -5", "v_add_nc_u32_e32 v7, s22, v0"]), ("A", [], []))

    def test_a_vertex_shader(self):
        self.assertEqual(self.tier("vs_wqm_uv", WQM + ONE_REGISTER + READ, vs), ("A", [], []))

    def test_launch_mask_form_is_unchanged(self):
        self.assertEqual(self.tier("ps_uv_launch", ONE_REGISTER + READ), ("A", [], []))


class NearMisses(unittest.TestCase):
    """Each program differs from an accepted one in one respect and must stay tier C."""

    def refused(self, name, lines, idioms=()):
        (tier, reasons, got_idioms), _ = meta_of(name, ps(lines))
        self.assertEqual(tier, "C", f"{name}: {reasons} {got_idioms}")
        self.assertIn("unsafe-vote-branch", reasons, name)
        for idiom in idioms:
            self.assertIn(idiom, got_idioms, name)

    def test_c_vgpr_written_under_the_launch_mask(self):
        # v1 = s4 in the launch lanes only; the helper lanes compare what they held before
        self.refused("ps_wqm_c_vgpr", ["v_mov_b32_e32 v1, s4"] + WQM + ["v_cmp_lt_u32_e64 s14, 5, v1"]
                     + ONE_REGISTER[1:] + READ)

    def test_c_vgpr_written_under_the_whole_quads(self):
        # uniform over EXEC in fact, but st["ua"] also holds VGPRs of the launch mask: not told apart, refused
        self.refused("ps_wqm_c_vgpr_inside", WQM + ["v_mov_b32_e32 v1, s4", "v_cmp_lt_u32_e64 s14, 5, v1"]
                     + ONE_REGISTER[1:] + READ)

    def test_c_whole_quads_of_a_narrowed_exec(self):
        # EXEC narrowed to x < 32 first: a half without such a pixel has no lane, votes no and keeps its own s20
        self.refused("ps_wqm_c_narrowed_before", ["v_cmp_gt_u32_e32 vcc, 32, v0", "s_and_b64 exec, exec, vcc"]
                     + WQM + ONE_REGISTER + READ)

    def test_c_narrowed_after_the_whole_quads(self):
        self.refused("ps_wqm_c_narrowed_after", WQM + ["v_cmp_gt_u32_e32 vcc, 32, v0", "s_and_b64 exec, exec, vcc"]
                     + ONE_REGISTER + READ)

    def test_c_saved_and_restored_whole_quads(self):
        # EXEC is the whole quads again after the restore, but only the write by s_wqm_b64 itself is modelled
        self.refused("ps_wqm_c_restored", WQM + ["s_mov_b64 s[12:13], exec", "v_cmp_gt_u32_e32 vcc, 32, v0",
                                                 "s_and_b64 exec, exec, vcc", "s_mov_b64 exec, s[12:13]"]
                     + ONE_REGISTER + READ)

    def test_c_whole_quads_through_a_register(self):
        self.refused("ps_wqm_c_register", ["s_wqm_b64 s[12:13], exec", "s_mov_b64 exec, s[12:13]"]
                     + ONE_REGISTER + READ)

    def test_c_scalar_read_with_readlane_from_a_uniform_vgpr(self):
        # lane 40 may be a lane the launch mask did not write; SHADER-10 calls s4 uniform, this rule does not use it
        self.refused("ps_wqm_c_readlane", ["v_mov_b32_e32 v1, s4", "v_readlane_b32 s4, v1, 40"] + WQM
                     + ONE_REGISTER + READ, idioms=["readlane-uniform"])

    def test_c_scalar_read_with_readfirstlane_under_whole_quads(self):
        # the first lane of EXEC can be a helper lane, which v1 = s4 never reached
        self.refused("ps_wqm_c_readfirstlane", ["v_mov_b32_e32 v1, s4"] + WQM + ["v_readfirstlane_b32 s4, v1"]
                     + ONE_REGISTER + READ, idioms=["readfirstlane-uniform"])

    def test_c_such_a_scalar_stays_refused_in_a_second_region(self):
        self.refused("ps_wqm_c_second_region", ["v_mov_b32_e32 v1, s4"] + WQM + [
            "v_readfirstlane_b32 s4, v1", "s_mov_b64 exec, s[10:11]"] + WQM + ONE_REGISTER + READ)

    def test_c_tainted_scalar(self):
        (tier, reasons, _), _ = meta_of("ps_wqm_c_tainted", ps(WQM + [
            "v_readfirstlane_b32 s4, v0"] + ONE_REGISTER + READ))
        self.assertEqual((tier, reasons), ("C", ["waterfall-scalar"]))

    def test_readfirstlane_under_the_launch_mask_does_not_spoil_it(self):
        # read while EXEC is the launch mask: the first lane is a launch lane, s21 is the uniform value
        (tier, reasons, idioms), _ = meta_of("ps_wqm_b_rfl_launch", ps(
            ["v_mov_b32_e32 v1, s4", "v_readfirstlane_b32 s21, v1"] + WQM
            + ["v_cmp_lt_u32_e64 s14, 5, s21"] + ONE_REGISTER[1:] + READ))
        self.assertEqual((tier, reasons, idioms), ("B", [], ["readfirstlane-uniform"]))


class VertexExecution(unittest.TestCase):
    """One wave of 64 lanes against two of 32 on the strict wave interpreter (vertex index = global lane)."""

    def test_words_equal_and_as_derived(self):
        for name, vote in (("one_register", ONE_REGISTER), ("pair", PAIR), ("vcc", VCC)):
            for push in UNIFORM:
                with self.subTest(name, s4=push[4]):
                    (tier, reasons, _), asm = meta_of("vs_wqm_uv_" + name, vs(WQM + vote + READ))
                    self.assertEqual((tier, reasons), ("A", []))
                    whole, halves = out_param(run_cut(asm, 64, push)), out_param(run_cut(asm, 32, push))
                    self.assertEqual(whole, halves)
                    want = (lambda k: (7, 1 + k)) if push[4] > 5 else (lambda k: (5, 2 + k))
                    self.assertEqual([w[:2] for w in whole], [want(k) for k in range(64)])


# a pixel program: v2 = the pixel's own number (a flat attribute), exported next to what the vote chose
PIXEL = ["v_interp_mov_f32_e32 v2, p0, attr0.x", "v_cvt_u32_f32_e32 v2, v2"] + WQM + ONE_REGISTER + [
    "v_add_nc_u32_e32 v7, s20, v2", "s_and_b64 exec, exec, s[10:11]"]


def run_pixels(asm, wave, s4):
    """mrt0 of pixels 0..63 run as 64 / wave waves of `wave` lanes, pixel p in lane p % wave of wave p // wave."""
    words = []
    for first in range(0, 64, wave):
        res = wi.run_wave(asm, wave=wave, push_constants=[0, 0, 0, 0, s4],
                          inputs={0: lambda lane, first=first: (float(first + lane), 0.0, 0.0, 0.0)})
        words += [tuple(next(v for name, v in out.items() if name.endswith("mrt0"))) for out in res["outputs"]]
    return words


class PixelExecution(unittest.TestCase):
    """One wave of 64 pixels against two of 32 on the strict wave interpreter, which runs the quad swaps that
    `s_wqm_b64` is translated to. Every lane is covered here (the interpreters do not model a helper lane that
    takes part in a quad swap), so this checks the translated vote and scalars, not partial coverage: that part is
    the argument in the module docstring."""

    def test_pixels_equal_and_as_derived(self):
        (tier, reasons, _), asm = meta_of("ps_wqm_uv_pixels", ps(PIXEL))
        self.assertEqual((tier, reasons), ("A", []))
        self.assertIn("OpGroupNonUniformQuadSwap", asm)
        for s4 in (9, 1):
            with self.subTest(s4=s4):
                whole, halves = run_pixels(asm, 64, s4), run_pixels(asm, 32, s4)
                self.assertEqual(whole, halves)
                want = (lambda p: (7, 1 + p, 0, 0)) if s4 > 5 else (lambda p: (5, 2 + p, 0, 0))
                self.assertEqual(whole, [want(p) for p in range(64)])


if __name__ == "__main__":
    unittest.main()
