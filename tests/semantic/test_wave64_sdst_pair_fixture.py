"""TEST-W64-SDST-PAIR-GPU: the one-register form of a wave64 lane-mask destination, on the fixture the GPU runs.

SHADER-W64-SDST-PAIR (docs/design.md, "Wave64 VALU mask operands"; tests/semantic/test_wave64_single_sdst.py) makes
the SGPR a wave64 VALU instruction names for a lane mask the pair it is on hardware: llvm-mc prints `v_cmp_gt_u32_e64
s4, ...` for the s[4:5] the compare writes, because the listings are made without `+wavefrontsize64`. This file adds
what that rule's own test does not have:

  * runtime/test/shaders/cs_w64x_cmp_sdst.asm, the GPU fixture (scripts/run_gpu_tests.py, check.py): six compares
    and a carry-out in the one-register form, each read back across both halves by another 64-bit scalar op
    (s_and_b64, s_or_b64, s_xor_b64, s_andn2_b64, s_and_saveexec_b64, s_mov_b64 exec, s_bcnt1 / s_ff1, SCC), with
    the high register holding another mask, 0 or nothing beforehand. Its eight words are derived from the ISA in its
    header; here it runs on a 64-wide subgroup and under --wave64-emulate, and the same program with every destination
    written as a pair translates to the same SPIR-V, text for text;
  * wave32 code through translate(): `v_cmp_* s5` there writes one register and s6 keeps its value;
  * tools/wave64_halves.py (SHADER-10) on the one-register form, which it used to read as "a lane mask combined
    with a register that is not one": the LLVM form of a uniform branch (`v_cmp_* sN` of scalars, `s_and_b64 vcc,
    exec, s[N:N+1]`, `s_cbranch_vccnz`) is tier A and exports the same words as one wave of 64 lanes and as two of
    32, a per-lane compare that narrows EXEC through the pair is a plain skip, and the mask read as data is still
    refused.

All programs are hand-written (no game data).

    python -I tests/semantic/test_wave64_sdst_pair_fixture.py
"""
import os
import re
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import support  # noqa: E402  (sets the tools path)
import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402
from test_wave64_emulate_opt import run_emulated_workgroup  # noqa: E402
from test_wave64_halves import UNIFORM, meta_of, out_param, ps, run_cut, vs  # noqa: E402

FIXTURE = "cs_w64x_cmp_sdst"
EXPECTED = [10240, 6144, 2053, 8224, 7714, 6184, 4144, 5]      # derivation in cs_w64x_cmp_sdst.asm
VSHARP = [0, 0, 0x1000, 0x31016FAC]
# a VALU lane-mask destination printed as one register: a compare's first operand, a carry-out's second
ONE_REGISTER = re.compile(r"^(\s*(?:v_cmp_\w+ |v_(?:add|sub|subrev)_co_u32\w* v\d+, ))s(\d+),")


def fssm(threads):
    h = bytearray(0x148)
    struct.pack_into("<4I", h, 0x138, 0x8088, threads, 1, 1)
    return bytes(h)


def fixture_lines():
    with open(os.path.join(support.FIXTURE_DIR, FIXTURE + ".s"), encoding="utf-8") as f:
        return f.read().splitlines()


def as_pairs(lines):
    """The listing with every one-register VALU mask destination written as the pair, as llvm-mc prints it with
    +wavefrontsize64 (and tools/rdna2_decode.py for a wave64 shader)."""
    return [ONE_REGISTER.sub(lambda m: f"{m.group(1)}s[{m.group(2)}:{int(m.group(2)) + 1}],", ln) for ln in lines]


def run_native(spv, threads=64, subgroup=64):
    return si.run_workgroup(
        spv, threads, subgroup,
        inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i % subgroup},
        push_constants=VSHARP + [0] * 28, memory={})


class Translation(unittest.TestCase):
    def test_fixture_on_a_64_wide_subgroup(self):
        spv = support.translate_fixture(FIXTURE, fssm=fssm(64))
        self.assertTrue(rdna2spirv.translate.last_gen.wave64, "no mattr line: wave64 from the code markers")
        memory = run_native(spv)
        self.assertEqual(sorted(memory), list(range(0, 32, 4)), "lane 0 stores out[0..7] only")
        self.assertEqual([memory[4 * k] for k in range(8)], EXPECTED)

    def test_fixture_under_wave64_emulation(self):
        prev = rdna2spirv.WAVE64_EMULATE
        rdna2spirv.WAVE64_EMULATE = True
        try:
            spv = support.translate_fixture(FIXTURE, fssm=fssm(64))
        finally:
            rdna2spirv.WAVE64_EMULATE = prev
        memory, _ = run_emulated_workgroup(spv, (64, 1, 1), push_constants=VSHARP + [0] * 28)
        self.assertEqual([memory.get(4 * k, 0) for k in range(8)], EXPECTED)

    def test_one_register_and_pair_forms_translate_alike(self):
        lines = fixture_lines()
        pairs = as_pairs(lines)
        self.assertEqual(sum(a != b for a, b in zip(lines, pairs)), 7, "six compares into SGPRs and one carry-out")
        one = support.translate_listing(FIXTURE, "\n".join(lines).encode() + b"\n", fssm=fssm(64))
        two = support.translate_listing(FIXTURE, "\n".join(pairs).encode() + b"\n", fssm=fssm(64))
        self.assertEqual(one, two)

    def test_wave32_code_keeps_the_single_register(self):
        # no wave64 marker: s5 is a 32-bit lane mask of its own (odd registers are common in the wave32 corpus) and
        # s6, which a wave64 compare into s5 would not even be aligned for, keeps its value
        lines = ["v_mbcnt_lo_u32_b32 v1, -1, 0", "s_mov_b32 s6, 77", "v_cmp_gt_u32_e64 s5, 12, v1",
                 "s_bcnt1_i32_b32 s7, s5", "v_cmpx_eq_u32_e32 0, v1",
                 "v_mov_b32_e32 v20, s6", "v_mov_b32_e32 v21, 0", "buffer_store_dword v20, v21, s[0:3], 0 offen",
                 "v_mov_b32_e32 v20, s7", "v_mov_b32_e32 v21, 4", "buffer_store_dword v20, v21, s[0:3], 0 offen",
                 "s_endpgm"]
        spv = support.translate_lines("cs_w32_cmp_sdst", lines, fssm=fssm(32))
        self.assertFalse(rdna2spirv.translate.last_gen.wave64)
        memory = run_native(spv, threads=32, subgroup=32)
        self.assertEqual((memory[0], memory[4]), (77, 12))


class Halves(unittest.TestCase):
    """What tools/wave64_halves.py makes of the one-register form (it read the pair as "a lane mask combined with a
    register that is not one": tier C `tainted-branch`, `tainted-exec` or `cross-lane`)."""

    def test_a_uniform_branch_in_the_llvm_form(self):
        # s4 is a user SGPR: 5 < s4 is the same in every lane, so the mask is every launch lane or none and each
        # half votes as the wave does; both sides set a VGPR and an SGPR that are read after the join
        body = ["v_cmp_lt_u32_e64 s14, 5, s4", "s_and_b64 vcc, exec, s[14:15]", "s_cbranch_vccnz 3",
                "v_mov_b32_e32 v6, 5", "s_mov_b32 s20, 2", "s_branch 2", "v_mov_b32_e32 v6, 7", "s_mov_b32 s20, 1",
                "v_add_nc_u32_e32 v7, s20, v0"]
        for push in UNIFORM:
            with self.subTest(s4=push[4]):
                (tier, reasons, idioms), asm = meta_of("vs_uv_one_register", vs(body))
                self.assertEqual((tier, reasons, idioms), ("A", [], []))
                whole, halves = out_param(run_cut(asm, 64, push)), out_param(run_cut(asm, 32, push))
                self.assertEqual(whole, halves)
                want = (lambda k: (7, 1 + k)) if push[4] > 5 else (lambda k: (5, 2 + k))
                self.assertEqual([w[:2] for w in whole], [want(k) for k in range(64)])

    def test_a_per_lane_compare_read_back_in_both_halves(self):
        # 40 > vertex index: lanes 0..39 of the wave, every lane of the first half and lanes 0..7 of the second
        body = ["v_cmp_gt_u32_e64 s14, 40, v0", "s_and_b64 s[16:17], exec, s[14:15]", "v_mov_b32_e32 v6, 1",
                "v_cndmask_b32_e64 v6, v6, 9, s16", "v_mov_b32_e32 v7, 3", "s_and_saveexec_b64 s[12:13], s[14:15]",
                "s_cbranch_execz 1", "v_add_nc_u32_e32 v7, v7, v0", "s_or_b64 exec, exec, s[12:13]"]
        (tier, reasons, idioms), asm = meta_of("vs_cmp_one_register", vs(body))
        self.assertEqual((tier, reasons, idioms), ("A", [], []))
        whole, halves = out_param(run_cut(asm, 64)), out_param(run_cut(asm, 32))
        self.assertEqual(whole, halves)
        self.assertEqual([w[:2] for w in whole], [(9, 3 + k) if k < 40 else (1, 3) for k in range(64)])

    def test_a_pixel_shader_skip_through_the_pair(self):
        (tier, reasons, _), _ = meta_of("ps_cmp_one_register", ps([
            "v_and_b32_e32 v4, 3, v0", "v_cmp_eq_u32_e64 s14, 1, v4", "s_and_saveexec_b64 s[12:13], s[14:15]",
            "s_cbranch_execz 1", "v_mul_u32_u24_e32 v6, 7, v0", "s_or_b64 exec, exec, s[12:13]"]))
        self.assertEqual((tier, reasons), ("A", []))

    def test_c_the_mask_itself_as_data_is_still_refused(self):
        # the 64 lane bits on hardware, the bits of the half here: widening the destination must not hide that
        (tier, reasons, _), _ = meta_of("vs_cmp_mask_as_data", vs([
            "v_cmp_gt_u32_e64 s14, 40, v0", "v_mov_b32_e32 v6, s14"]))
        self.assertEqual(tier, "C")
        self.assertIn("cross-lane", reasons)


if __name__ == "__main__":
    unittest.main()
