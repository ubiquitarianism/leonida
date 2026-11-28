"""Value tests for the SDWA forms of v_add_co_ci_u32, v_sub_co_ci_u32 and v_subrev_co_ci_u32.

These are the only SDWA instructions that print a carry-out between the destination and the sources
(`v_add_co_ci_u32_sdwa v4, vcc_lo, v2, v3, vcc_lo ...`), so src0_sel and src1_sel belong to operands 2 and 3, not 1
and 2. Each case runs one whole wave32 or wave64 subgroup in spirv_mini_interp.py and compares the result and the
carry-out (read back with v_cndmask_b32) with plain Python. Standard library only; no GPU, no LLVM, no game files.
"""
import itertools
import os
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
sys.path.insert(0, HERE)

import spirv_mini_interp as si  # noqa: E402
import support  # noqa: E402

M32 = 0xFFFFFFFF
X, Y, OLD = 0x8234F678, 0x9ABCDEF0, 0xA5A5A5A5
SELS = {"BYTE_0": (0, 8), "BYTE_1": (8, 8), "BYTE_2": (16, 8), "BYTE_3": (24, 8), "WORD_0": (0, 16), "WORD_1": (16, 16),
        "DWORD": (0, 32)}


def select(value, sel, sext=False):
    off, width = SELS[sel]
    value = (value >> off) & ((1 << width) - 1)
    if sext and value >> (width - 1):
        value |= M32 & ~((1 << width) - 1)
    return value


def reference(op, x, y, carry):
    if op == "v_add_co_ci_u32":
        total = x + y + carry
        return total & M32, total >> 32
    if op == "v_subrev_co_ci_u32":
        x, y = y, x
    return (x - y - carry) & M32, int(y + carry > x)


def run(line, wave, carry_lanes):
    """{lane: (v4, carry-out)} for `line`, with VCC set on the lanes below carry_lanes."""
    fssm = bytearray(0x148)
    fssm[:4] = b"FSSM"
    struct.pack_into("<I", fssm, 0xD4, 0x20 if wave == 32 else 0)
    struct.pack_into("<4I", fssm, 0x138, 0x8088, wave, 1, 1)
    vcc = "vcc_lo" if wave == 32 else "vcc"
    spv = support.translate_lines("cs_sdwa_carry", [
        f"v_mov_b32_e32 v2, {X:#x}", f"v_mov_b32_e32 v3, {Y:#x}", f"v_mov_b32_e32 v4, {OLD:#x}",
        f"v_cmp_gt_u32_e32 {vcc}, {carry_lanes}, v0",
        line.replace("VCC", vcc),
        f"v_cndmask_b32_e64 v8, 0, 1, {vcc}",
        "v_lshlrev_b32_e32 v9, 4, v0",
        "buffer_store_dword v4, v9, s[0:3], 0 offen",
        "buffer_store_dword v8, v9, s[0:3], 0 offen offset:4",
        "s_endpgm"], fssm=bytes(fssm))
    memory = si.run_workgroup(spv, wave, wave,
                              inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i},
                              push_constants=[0, 0, 0x1000, 0x31016FAC] + [0] * 28, memory={})
    return {lane: (memory[16 * lane], memory[16 * lane + 4]) for lane in range(wave)}


class SdwaCarryTests(unittest.TestCase):
    OPS = ("v_add_co_ci_u32", "v_sub_co_ci_u32", "v_subrev_co_ci_u32")

    def check(self, op, sel0, sel1, wave, sext=(False, False)):
        src = [f"sext({r})" if s else r for r, s in zip(("v2", "v3"), sext)]
        got = run(f"{op}_sdwa v4, VCC, {src[0]}, {src[1]}, VCC dst_sel:DWORD dst_unused:UNUSED_PAD "
                  f"src0_sel:{sel0} src1_sel:{sel1}", wave, carry_lanes=wave // 2 + 3)
        x, y = select(X, sel0, sext[0]), select(Y, sel1, sext[1])
        for lane, words in got.items():
            self.assertEqual(words, reference(op, x, y, int(lane < wave // 2 + 3)), f"lane {lane}")

    def test_every_source_selector_wave32(self):
        for op, sel0, sel1 in itertools.product(self.OPS, SELS, SELS):
            with self.subTest(op=op, src0_sel=sel0, src1_sel=sel1):
                self.check(op, sel0, sel1, 32)

    def test_source_selectors_wave64(self):
        for op, (sel0, sel1) in itertools.product(self.OPS, (("WORD_1", "DWORD"), ("DWORD", "BYTE_2"),
                                                              ("BYTE_1", "WORD_1"), ("DWORD", "DWORD"))):
            with self.subTest(op=op, src0_sel=sel0, src1_sel=sel1):
                self.check(op, sel0, sel1, 64)

    def test_sext(self):
        for op, sext in itertools.product(self.OPS, ((True, False), (False, True), (True, True))):
            with self.subTest(op=op, sext=sext):
                self.check(op, "BYTE_1", "WORD_1", 32, sext)

    def test_destination_selector(self):
        # dst_sel places the low half of the sum; the carry-out is still the whole 32-bit add's.
        total, carry = reference("v_add_co_ci_u32", select(X, "WORD_1"), select(Y, "WORD_0"), 1)
        for unused, rest in (("UNUSED_PAD", 0), ("UNUSED_PRESERVE", OLD & 0xFFFF)):
            with self.subTest(dst_unused=unused):
                got = run(f"v_add_co_ci_u32_sdwa v4, VCC, v2, v3, VCC dst_sel:WORD_1 dst_unused:{unused} "
                          "src0_sel:WORD_1 src1_sel:WORD_0", 32, carry_lanes=32)
                self.assertEqual(set(got.values()), {(((total & 0xFFFF) << 16) | rest, carry)})

    def test_short_form(self):
        # llvm-mc prints the carry operands; a hand-written listing without them still has its sources at 1 and 2.
        got = run("v_add_co_ci_u32_sdwa v4, v2, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:BYTE_0",
                  32, carry_lanes=0)
        self.assertEqual(set(got.values()), {reference("v_add_co_ci_u32", select(X, "WORD_1"), select(Y, "BYTE_0"), 0)})


if __name__ == "__main__":
    unittest.main()
