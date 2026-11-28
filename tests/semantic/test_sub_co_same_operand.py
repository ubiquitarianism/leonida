"""Carry-out subtracts of a register from itself: v_sub_co_u32 / v_subrev_co_u32 (and the _ci_ forms) vD, sdst, vX, vX.

x - x is 0 with no borrow, so D = 0 - carry in and the borrow out is the carry in. The translator writes that out
instead of OpISubBorrow %x %x: AMD's Windows driver (amdvlk64.dll 32.0.21037.1004 and 32.0.21045.29003, gfx1036)
dies with an access violation creating a native wave64 compute pipeline in which the borrow of such an
OpISubBorrow predicates the value of a subgroup shuffle (crosslane_fuzz.py --seed 880317 --wave 64 --w64-native
--no-maskdata --dpp --ext, programs 1 and 181; docs/design.md, "Carry-out subtract of a register from itself").

Checked here, in wave32 and native wave64 (run in spirv_mini_interp):
  - D and the borrow mask for every lane, with and without a carry in, under a partial EXEC (an inactive lane keeps
    D and reads 0 in the mask), and the wave-wide count of the mask;
  - the clamp bit (D saturates to 0 on a borrow, the borrow is still reported);
  - no OpISubBorrow is emitted for the same-operand forms, and two different registers holding equal values still
    go through OpISubBorrow and give the same answers.
With --wave64-emulate only the emitted form is checked (the last point).
"""
import gc
import os
import re
import struct
import sys
import tempfile
import unittest
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

sys.dont_write_bytecode = True
if HERE not in sys.path:
    sys.path.insert(0, HERE)
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402

STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)
VSHARP = [0, 0, 0xFFFFFFFF, 0x31016FAC]  # OOB_SELECT 3 (raw): stores are in range up to num_records
SENTINEL = 0xDEADBEEF

# (op, has carry in, reversed operands)
OPS = [
    ("v_sub_co_u32", False),
    ("v_subrev_co_u32", False),
    ("v_sub_co_ci_u32", True),
    ("v_subrev_co_ci_u32", True),
]


def _translate_and_run(name, body_lines, wave_size, emulate):
    """(SPIR-V assembly text, the output buffer after one workgroup of wave_size lanes). With emulate the module
    is only translated (the buffer is None): spirv_mini_interp does not run the --wave64-emulate round loop, as in
    test_carry_exec_mask.py; crosslane_fuzz.py runs the emulated form on a GPU."""
    prev_emu = rdna2spirv.WAVE64_EMULATE
    try:
        rdna2spirv.WAVE64_EMULATE = emulate
        with tempfile.TemporaryDirectory() as tmp:
            os.mkdir(os.path.join(tmp, "disasm"))
            os.mkdir(os.path.join(tmp, "fssm"))
            s_path = os.path.join(tmp, "disasm", name + ".s")
            with open(s_path, "w", encoding="utf-8", newline="\n") as f:
                f.write(f"; {name}\n")
                for ln in body_lines:
                    f.write(f"\t{ln}\n")
                f.write("\ts_endpgm\n")
            with open(os.path.join(tmp, name + ".dxbc"), "wb") as f:
                f.write(STUB_DXBC)
            # COMPUTE_PGM_RSRC2 0x8088: 4 user SGPRs (s[0:3] = the output V#), TGID_X.
            with open(os.path.join(tmp, "fssm", name + ".fssm"), "wb") as f:
                f.write(bytes(0x138) + struct.pack("<4I", 0x8088, wave_size, 1, 1))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ResourceWarning)
                spv = rdna2spirv.translate(s_path)
                gc.collect()
            assert rdna2spirv.translate.last_gen.wave64 == (wave_size == 64), name
    finally:
        rdna2spirv.WAVE64_EMULATE = prev_emu
    if emulate:
        return spv, None
    sg_size = wave_size
    mem = si.run_workgroup(
        spv, wave_size, sg_size,
        inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i % sg_size},
        push_constants=VSHARP + [0] * 28, memory={},
    )
    return spv, mem


class _SameOperandMixin:
    wave_size = 64
    emulate = False

    def _names(self):
        if self.wave_size == 64:
            return {"exec": "exec", "mov": "s_mov_b64", "or": "s_or_b64", "bcnt": "s_bcnt1_i32_b64",
                    "act": "s[4:5]", "tmp": "s[6:7]", "sdst": "s[10:11]", "cin": "s[12:13]", "active": [2, 5, 35, 40]}
        return {"exec": "exec_lo", "mov": "s_mov_b32", "or": "s_or_b32", "bcnt": "s_bcnt1_i32_b32",
                "act": "s4", "tmp": "s6", "sdst": "s10", "cin": "s12", "active": [2, 5, 19, 24]}

    def _program(self, op, with_ci, src_a, src_b, clamp=False):
        """The op under a partial EXEC, with lane-varying sources (v2 = v3 = 3 * lane + 7) and, for the _ci_ forms,
        a carry in on the odd lanes. Stores per lane: D, the lane's bit of the borrow mask, and the mask's count."""
        n = self._names()
        lines = [f"{n['mov']} {n['exec']}, -1"]
        lines.append(f"v_cmp_eq_u32_e64 {n['act']}, v0, {n['active'][0]}")
        for lane in n["active"][1:]:
            lines.append(f"v_cmp_eq_u32_e64 {n['tmp']}, v0, {lane}")
            lines.append(f"{n['or']} {n['act']}, {n['act']}, {n['tmp']}")
        lines += [
            "v_mul_lo_u32 v2, v0, 3",
            "v_add_nc_u32 v2, 7, v2",
            "v_mov_b32 v3, v2",
            f"v_mov_b32 v6, {SENTINEL:#x}",
            f"{n['mov']} {n['sdst']}, -1",          # stale bits must be overwritten, also on inactive lanes
        ]
        if with_ci:
            lines += ["v_and_b32 v7, 1, v0", f"v_cmp_ne_u32_e64 {n['cin']}, 0, v7"]
        inst = f"{op}_e64 v6, {n['sdst']}, {src_a}, {src_b}" + (f", {n['cin']}" if with_ci else "")
        if clamp:
            inst += " clamp"
        lines += [
            f"{n['mov']} {n['exec']}, {n['act']}",
            inst,
            f"{n['mov']} {n['exec']}, -1",
            f"{n['bcnt']} s14, {n['sdst']}",
            f"v_cndmask_b32_e64 v9, 0, 1, {n['sdst']}",
            "v_mov_b32 v10, s14",
            "v_mul_lo_u32 v11, v0, 12",
            "buffer_store_dword v6, v11, s[0:3], 0 offen",
            "v_add_nc_u32 v12, 4, v11",
            "buffer_store_dword v9, v12, s[0:3], 0 offen",
            "v_add_nc_u32 v12, 8, v11",
            "buffer_store_dword v10, v12, s[0:3], 0 offen",
        ]
        return lines

    def _check(self, mem, with_ci, clamp, what):
        if mem is None:             # --wave64-emulate: translated only
            return
        active = set(self._names()["active"])
        borrowing = {lane for lane in active if with_ci and lane & 1}
        for lane in range(self.wave_size):
            d, bit, count = mem[lane * 12], mem[lane * 12 + 4], mem[lane * 12 + 8]
            self.assertEqual(count, len(borrowing), f"{what}: lane {lane} mask count")
            if lane not in active:
                self.assertEqual(d, SENTINEL, f"{what}: inactive lane {lane} D")
                self.assertEqual(bit, 0, f"{what}: inactive lane {lane} borrow bit")
            elif lane in borrowing:
                self.assertEqual(d, 0 if clamp else 0xFFFFFFFF, f"{what}: lane {lane} D with a carry in")
                self.assertEqual(bit, 1, f"{what}: lane {lane} borrow bit")
            else:
                self.assertEqual(d, 0, f"{what}: lane {lane} D")
                self.assertEqual(bit, 0, f"{what}: lane {lane} borrow bit")

    def test_same_register_values_and_borrow(self):
        for op, with_ci in OPS:
            for clamp in (False, True):
                with self.subTest(op=op, clamp=clamp, wave=self.wave_size, emulate=self.emulate):
                    name = f"cs_{op}_same{'_clamp' if clamp else ''}_w{self.wave_size}"
                    spv, mem = _translate_and_run(name, self._program(op, with_ci, "v2", "v2", clamp),
                                                  self.wave_size, self.emulate)
                    self.assertNotIn("OpISubBorrow", spv, f"{op}: x - x is written out, not left to OpISubBorrow")
                    self._check(mem, with_ci, clamp, f"{op} v2, v2")

    def test_equal_values_in_two_registers_still_use_the_borrow_op(self):
        """The same answers when the two sources are different registers that happen to hold equal values: that
        path is not changed, so it still goes through OpISubBorrow (and never with one id twice)."""
        for op, with_ci in OPS:
            with self.subTest(op=op, wave=self.wave_size, emulate=self.emulate):
                spv, mem = _translate_and_run(f"cs_{op}_two_w{self.wave_size}",
                                              self._program(op, with_ci, "v2", "v3"), self.wave_size, self.emulate)
                borrows = re.findall(r"OpISubBorrow %umul (\S+) (\S+)", spv)
                self.assertTrue(borrows, f"{op}: two registers still use OpISubBorrow")
                self.assertFalse([b for b in borrows if b[0] == b[1]], f"{op}: OpISubBorrow of one id with itself")
                self._check(mem, with_ci, False, f"{op} v2, v3")


class SameOperandWave32(_SameOperandMixin, unittest.TestCase):
    wave_size = 32


class SameOperandWave64Native(_SameOperandMixin, unittest.TestCase):
    wave_size = 64


class SameOperandWave64Emulate(_SameOperandMixin, unittest.TestCase):
    wave_size = 64
    emulate = True


if __name__ == "__main__":
    unittest.main()
