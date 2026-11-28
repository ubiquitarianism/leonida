"""A 32-bit literal is sign-extended by the signed 64-bit vector ops (SHADER-VALU-LITERAL-SEXT).

LLVM AMDGPU operand syntax ("Conversion of Integer Values"): a 32-bit literal used as a 64-bit integer
operand is sign-extended for i64 ({-1, num.i32}) and zero-extended for u64/b64 ({0, num.u32}). On gfx1030
`-100` and `0xffffff9c` encode to the same bytes (one literal dword) and llvm-mc prints the latter, so the
printed hex has to be widened by the operand's signedness. The signed sources are v_cmp(x)_*_i64,
v_ashrrev_i64 S1 and v_mad_i64_i32 S2; v_cmp_*_u64, v_lshrrev_b64 and v_mad_u64_u32 stay zero-extended.
This is the vector half of SHADER-SALU-LITERAL-SEXT (#978). Each case runs the hand-written listing through
rdna2spirv and spirv_mini_interp and compares with the explicit 64-bit pair, which never depended on the
literal rule.
"""
from __future__ import annotations

import gc
import os
import struct
import sys
import tempfile
import unittest
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")
sys.dont_write_bytecode = True
for _p in (TOOLS_DIR, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402

STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)
M32 = 0xFFFFFFFF
NEG100 = 0xFFFFFFFFFFFFFF9C

# v5..v8 carry the 16-bit halves of v{a} and v{b} (the work registers are v10 and up).
EXPORT = [
    "v_and_b32 v5, 0xffff, v{a}",
    "v_lshrrev_b32 v6, 16, v{a}",
    "v_and_b32 v7, 0xffff, v{b}",
    "v_lshrrev_b32 v8, 16, v{b}",
    "v_or_b32 v5, 0x40000000, v5",
    "v_or_b32 v6, 0x40000000, v6",
    "v_or_b32 v7, 0x40000000, v7",
    "v_or_b32 v8, 0x40000000, v8",
]

_tmp = None


def tearDownModule() -> None:
    global _tmp
    if _tmp is not None:
        gc.collect()
        try:
            _tmp.cleanup()
        except Exception:
            pass
        _tmp = None


def run(lines: list[str], a: int = 10, b: int = 11) -> tuple[int, int]:
    """Run the listing, return the 32-bit values of VGPR a and VGPR b."""
    global _tmp
    if _tmp is None:
        _tmp = tempfile.TemporaryDirectory()
        os.mkdir(os.path.join(_tmp.name, "disasm"))
        with open(os.path.join(_tmp.name, "ps_vsext.dxbc"), "wb") as f:
            f.write(STUB_DXBC)
    path = os.path.join(_tmp.name, "disasm", "ps_vsext.s")
    body = list(lines) + [ln.format(a=a, b=b) for ln in EXPORT]
    asm = "; test_valu_signed_literal64\n" + "".join(f"\t{ln}\n" for ln in body)
    asm += "\texp mrt0 v5, v6, v7, v8 done vm\n\ts_endpgm\n"
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spv = rdna2spirv.translate(path)
    w = si.run_fragment(spv)["%out_mrt0"]
    assert all((x >> 16) == 0x4000 for x in w), [hex(x) for x in w]
    return (w[0] & 0xFFFF) | ((w[1] & 0xFFFF) << 16), (w[2] & 0xFFFF) | ((w[3] & 0xFFFF) << 16)


def pair(v: int, lo: str = "s0", hi: str = "s1") -> list[str]:
    op = "s_mov_b32" if lo.startswith("s") else "v_mov_b32"
    return [f"{op} {lo}, {v & M32:#x}", f"{op} {hi}, {(v >> 32) & M32:#x}"]


def cmp(line: str, pre: list[str] = ()) -> int:
    """The lane's bit of a compare that writes vcc_lo, as 0 or 1."""
    return run(list(pre) + [line, "v_cndmask_b32_e64 v10, 0, 1, vcc_lo"], 10, 10)[0]


class SignedLiteral(unittest.TestCase):
    def spellings(self, make):
        """The same operand as a decimal, a printed literal and an explicit SGPR pair."""
        return {
            "decimal": make("-100", []),
            "literal": make("0xffffff9c", []),
            "pair": make("s[0:1]", pair(NEG100)),
        }

    def check(self, make, want):
        got = self.spellings(make)
        self.assertEqual(got["pair"], want)
        self.assertEqual(got, {k: want for k in got})

    def test_cmp_lt_i64(self):
        # 5 < -100 is false; -100 < 5 is true (zero extension reads 0xffffff9c as 4294967196).
        self.check(lambda src, pre: cmp(f"v_cmp_lt_i64_e64 vcc_lo, 5, {src}", pre), 0)
        self.check(lambda src, pre: cmp(f"v_cmp_lt_i64_e64 vcc_lo, {src}, 5", pre), 1)

    def test_cmp_e32_literal_src0(self):
        # VOPC e32: the literal is src0, the VGPR pair src1.
        pre = pair(5, "v0", "v1")
        self.check(lambda src, p: cmp(f"v_cmp_gt_i64_e32 vcc_lo, {src}, v[0:1]", pre + p), 0)
        self.check(lambda src, p: cmp(f"v_cmp_le_i64_e32 vcc_lo, {src}, v[0:1]", pre + p), 1)
        self.check(lambda src, p: cmp(f"v_cmp_ge_i64_e32 vcc_lo, {src}, v[0:1]", pre + p), 0)

    def test_cmp_eq_ne_i64(self):
        # A VGPR pair holding -100 equals the literal only when the literal is sign-extended.
        pre = pair(NEG100, "v0", "v1")
        self.check(lambda src, p: cmp(f"v_cmp_eq_i64_e64 vcc_lo, v[0:1], {src}", pre + p), 1)
        self.check(lambda src, p: cmp(f"v_cmp_ne_i64_e64 vcc_lo, {src}, v[0:1]", pre + p), 0)

    def test_ashrrev_i64(self):
        self.check(lambda src, pre: run(pre + [f"v_ashrrev_i64 v[10:11], 4, {src}"]), (0xFFFFFFF9, M32))
        self.check(lambda src, pre: run(pre + [f"v_ashrrev_i64 v[10:11], 32, {src}"]), (M32, M32))

    def test_mad_i64_i32_src2(self):
        # 1 * 1 + (-100) = -99.
        self.check(lambda src, pre: run(pre + [f"v_mad_i64_i32 v[10:11], vcc_lo, 1, 1, {src}"]), (0xFFFFFF9D, M32))

    def test_positive_literal_stays_positive(self):
        self.assertEqual(run(["v_ashrrev_i64 v[10:11], 4, 0x7fffffff"]), (0x07FFFFFF, 0))
        self.assertEqual(run(["v_mad_i64_i32 v[10:11], vcc_lo, 1, 1, 0x7fffffff"]), (0x80000000, 0))
        self.assertEqual(cmp("v_cmp_lt_i64_e64 vcc_lo, 5, 0x7fffffff"), 1)

    def test_inline_constants_unchanged(self):
        # -16..-1 are inline constants, already 64-bit; they never went through the literal rule.
        self.assertEqual(run(["v_ashrrev_i64 v[10:11], 4, -16"]), (M32, M32))
        self.assertEqual(cmp("v_cmp_lt_i64_e64 vcc_lo, -16, 5"), 1)
        self.assertEqual(run(["v_mad_i64_i32 v[10:11], vcc_lo, 1, 1, -16"]), (0xFFFFFFF1, M32))

    def test_unsigned_ops_zero_extend(self):
        self.assertEqual(cmp("v_cmp_lt_u64_e64 vcc_lo, 5, 0xffffff9c"), 1)
        self.assertEqual(cmp("v_cmp_gt_u64_e64 vcc_lo, 0xffffff9c, 5"), 1)
        self.assertEqual(cmp("v_cmp_eq_u64_e64 vcc_lo, v[0:1], 0xffffff9c", pair(0xFFFFFF9C, "v0", "v1")), 1)
        self.assertEqual(run(["v_mad_u64_u32 v[10:11], vcc_lo, 1, 1, 0xffffff9c"]), (0xFFFFFF9D, 0))
        self.assertEqual(run(["v_lshrrev_b64 v[10:11], 4, 0xffffff9c"]), (0x0FFFFFF9, 0))
        self.assertEqual(run(["v_lshlrev_b64 v[10:11], 4, 0xffffff9c"]), (0xFFFFF9C0, 0xF))


if __name__ == "__main__":
    unittest.main()
