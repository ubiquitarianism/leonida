"""A 32-bit SALU literal is sign-extended by the signed 64-bit scalar ops (SHADER-SALU-LITERAL-SEXT).

On gfx1030 `-100` and `0xffffff9c` encode to the same bytes (one literal dword) and llvm-mc prints the
latter. The signed 64-bit ops (s_flbit_i32_i64, s_bfe_i64, s_ashr_i64) take that dword sign-extended
(LLVM #70274), the unsigned ones (s_flbit_i32_b64, s_bfe_u64, s_lshr_b64) zero-extended. Each case runs
the hand-written listing through rdna2spirv and spirv_mini_interp and compares with the explicit
64-bit pair, which never depended on the literal rule.
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

# v5..v8 carry the 16-bit halves of s4 and s5; s8 is exported through the same path in a second run.
EXPORT = [
    "v_mov_b32 v0, s{a}",
    "v_mov_b32 v1, s{b}",
    "v_and_b32 v5, 0xffff, v0",
    "v_lshrrev_b32 v6, 16, v0",
    "v_and_b32 v7, 0xffff, v1",
    "v_lshrrev_b32 v8, 16, v1",
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


def run(lines: list[str], a: int, b: int) -> tuple[int, int]:
    """Run the listing, return the 32-bit values of SGPR a and SGPR b."""
    global _tmp
    if _tmp is None:
        _tmp = tempfile.TemporaryDirectory()
        os.mkdir(os.path.join(_tmp.name, "disasm"))
        with open(os.path.join(_tmp.name, "ps_sext.dxbc"), "wb") as f:
            f.write(STUB_DXBC)
    path = os.path.join(_tmp.name, "disasm", "ps_sext.s")
    body = list(lines) + [ln.format(a=a, b=b) for ln in EXPORT]
    asm = "; test_salu_signed_literal64\n" + "".join(f"\t{ln}\n" for ln in body)
    asm += "\texp mrt0 v5, v6, v7, v8 done vm\n\ts_endpgm\n"
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spv = rdna2spirv.translate(path)
    w = si.run_fragment(spv)["%out_mrt0"]
    assert all((x >> 16) == 0x4000 for x in w), [hex(x) for x in w]
    return (w[0] & 0xFFFF) | ((w[1] & 0xFFFF) << 16), (w[2] & 0xFFFF) | ((w[3] & 0xFFFF) << 16)


def pair(v: int) -> list[str]:
    return [f"s_mov_b32 s0, {v & M32:#x}", f"s_mov_b32 s1, {(v >> 32) & M32:#x}"]


class SignedLiteral(unittest.TestCase):
    CTL = ["s_mov_b32 s6, 0x200020"]            # offset 32, width 32

    def spellings(self, make):
        """The same operand as a decimal, a printed literal and an explicit pair."""
        return {
            "decimal": make("-100", []),
            "literal": make("0xffffff9c", []),
            "pair": make("s[0:1]", pair(0xFFFFFFFFFFFFFF9C)),
        }

    def test_ashr_i64(self):
        def make(src, pre):
            return run(pre + [f"s_ashr_i64 s[4:5], {src}, 32"], 4, 5)
        got = self.spellings(make)
        self.assertEqual(got["pair"], (M32, M32))
        self.assertEqual(got, {k: (M32, M32) for k in got})

    def test_bfe_i64(self):
        def make(src, pre):
            return run(pre + self.CTL + [f"s_bfe_i64 s[4:5], {src}, s6"], 4, 5)
        got = self.spellings(make)
        self.assertEqual(got["pair"], (M32, M32))
        self.assertEqual(got, {k: (M32, M32) for k in got})

    def test_flbit_i32_i64(self):
        def make(src, pre):
            return run(pre + [f"s_flbit_i32_i64 s8, {src}"], 8, 8)
        got = self.spellings(make)
        self.assertEqual(got["pair"], (57, 57))
        self.assertEqual(got, {k: (57, 57) for k in got})

    def test_positive_literal_stays_positive(self):
        self.assertEqual(run(["s_flbit_i32_i64 s8, 0x7fffffff"], 8, 8), (33, 33))
        self.assertEqual(run(["s_ashr_i64 s[4:5], 0x7fffffff, 8"], 4, 5), (0x7FFFFF, 0))

    def test_unsigned_ops_zero_extend(self):
        self.assertEqual(run(["s_flbit_i32_b64 s8, 0xffffff9c"], 8, 8), (32, 32))
        self.assertEqual(run(["s_lshr_b64 s[4:5], 0xffffff9c, 32"], 4, 5), (0, 0))
        self.assertEqual(run(self.CTL + ["s_bfe_u64 s[4:5], 0xffffff9c, s6"], 4, 5), (0, 0))


if __name__ == "__main__":
    unittest.main()
