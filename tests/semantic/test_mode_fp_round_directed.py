"""Semantic tests for all four IEEE-754 MODE.FP_ROUND[3:2] rounding modes
(0=RNE, 1=RTP/+inf, 2=RTN/-inf, 3=RTZ) and dynamic SGPR s_setreg_b32 writes.

Verifies bit-exact f16 and f64 results against an exact Fraction oracle through
tools/rdna2spirv.py -> spirv_mini_interp.py + spirv_asm.assemble().
Standard library only; no game data.
"""

from __future__ import annotations

from fractions import Fraction
import math
import os
import pathlib
import struct
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
HERE = os.path.dirname(os.path.abspath(__file__))
TESTS_DIR = os.path.dirname(HERE)
for p in (str(ROOT / "tools"), HERE, TESTS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import spirv_asm
import spirv_interp
import spirv_mini_interp as si
from support import translate_lines

RNE, RTP, RTN, RTZ = 0, 1, 2, 3


def f32_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def f32_from_bits(u: int) -> float:
    return struct.unpack("<f", struct.pack("<I", u & 0xFFFFFFFF))[0]


def f64_words(x: float) -> tuple[int, int]:
    lo, hi = struct.unpack("<II", struct.pack("<d", x))
    return lo, hi


def f16_to_fraction(h: int) -> Fraction:
    sign = -1 if (h >> 15) & 1 else 1
    exp = (h >> 10) & 0x1F
    mant = h & 0x3FF
    if exp == 0:
        return sign * Fraction(mant, 1 << 24)
    return sign * Fraction(1024 + mant) * (Fraction(2) ** (exp - 25))


def ref_f32_to_f16(u32: int, mode: int) -> int:
    """Bit-exact IEEE-754 f32 -> f16 oracle (with f32 denormals flushed to +-0)."""
    sign = (u32 >> 31) & 1
    sg = sign << 15
    exp = (u32 >> 23) & 0xFF
    mant = u32 & 0x7FFFFF
    if exp == 0xFF:
        if mant == 0:
            return sg | 0x7C00
        return sg | 0x7E00 | ((mant >> 13) & 0x3FF)
    if exp == 0:
        return sg  # +-0 or flushed f32 denormal

    exact = (-1 if sign else 1) * Fraction((1 << 23) + mant) * (Fraction(2) ** (exp - 150))
    return ref_fraction_to_f16(exact, sign, mode)


def ref_fraction_to_f16(val: Fraction, sign_bit: int, mode: int) -> int:
    sg = sign_bit << 15
    if val == 0:
        return sg
    mag = abs(val)
    is_neg = val < 0
    sg = 0x8000 if is_neg else 0
    is_trunc = (mode == RTZ) or (mode == RTP and is_neg) or (mode == RTN and not is_neg)
    is_away = (mode == RTP and not is_neg) or (mode == RTN and is_neg)

    max_f16 = Fraction(65504)
    if mag > max_f16:
        if is_trunc:
            return sg | 0x7BFF
        if is_away or mag >= Fraction(65520):
            return sg | 0x7C00
        return sg | 0x7BFF

    # Find bracketing f16 magnitude indices in [0..0x7BFF] via binary search.
    lo, hi = 0, 0x7BFF
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if f16_to_fraction(mid) <= mag:
            lo = mid
        else:
            hi = mid - 1
    low_val = f16_to_fraction(lo)
    if low_val == mag:
        return sg | lo
    if is_trunc:
        return sg | lo
    if is_away:
        return sg | (lo + 1)
    # RNE
    high_val = f16_to_fraction(lo + 1)
    d_lo = mag - low_val
    d_hi = high_val - mag
    if d_lo < d_hi:
        return sg | lo
    if d_hi < d_lo:
        return sg | (lo + 1)
    return sg | (lo if (lo & 1) == 0 else lo + 1)


def _run_ps_mrt0(name: str, body: list[str]) -> tuple[int, int, int, int]:
    spv = translate_lines(name, body + ["s_endpgm"])
    spirv_asm.assemble(spv)
    return tuple(si.run_fragment(spv)["%out_mrt0"])


class DirectedF16ConvertTests(unittest.TestCase):
    def test_v_cvt_f16_f32_all_four_modes_exact(self) -> None:
        # Covers normals, ties, subnormals, subnormal->normal carry, tiny underflow,
        # +-0, max finite, overflow boundaries, +-inf, and NaN.
        test_u32s = [
            0x00000000,  # +0.0
            0x80000000,  # -0.0
            0x0D800000,  # +2^-100 (tiny positive -> +0x0001 in RTP, +0 in RNE/RTN/RTZ)
            0x8D800000,  # -2^-100 (tiny negative -> 0x8001 in RTN, -0 in RNE/RTP/RTZ)
            0x33000000,  # +2^-25 (exact halfway to min subnormal)
            0xB3000000,  # -2^-25
            0x33800000,  # +2^-24 (exact min subnormal 0x0001)
            0x33800001,  # +2^-24 + eps
            0xB3800001,  # -(2^-24 + eps)
            0x387FE000,  # largest exact f16 subnormal (0x03FF)
            0x387FFFFF,  # just below 2^-14 (carries into normal 0x0400 under RNE/RTP)
            0xB87FFFFF,  # negative just above -2^-14
            0x3F800000,  # +1.0 (exact)
            0x3F800001,  # +1.0 + 1 f32 ulp
            0xBF800001,  # -(1.0 + 1 f32 ulp)
            0x3F801000,  # +1.0 + 0.5 f16 ulp (tie to even -> 0x3C00 in RNE)
            0x3F803000,  # +1.0009765625 + 0.5 f16 ulp (tie to even -> 0x3C02 in RNE)
            0x477FE000,  # +65504.0 (exact max finite f16)
            0x477FE001,  # +65504.004 (overflows to +inf only in RTP)
            0xC77FE001,  # -65504.004 (overflows to -inf only in RTN)
            0x477FF000,  # +65520.0 (overflows in RNE & RTP, clamps in RTN & RTZ)
            0xC77FF000,  # -65520.0 (overflows in RNE & RTN, clamps in RTP & RTZ)
            0x47800000,  # +65536.0
            0xC7800000,  # -65536.0
            0x7F800000,  # +inf
            0xFF800000,  # -inf
            0x7FC02000,  # qNaN with payload
        ]
        for u32 in test_u32s:
            with self.subTest(u32=hex(u32)):
                got = _run_ps_mrt0(
                    "ps_cvt_f16_4modes",
                    [
                        f"v_mov_b32 v0, {u32:#x}",
                        "s_setreg_imm32_b32 hwreg(HW_REG_MODE, 2, 2), 0",
                        "v_cvt_f16_f32 v1, v0",
                        "s_setreg_imm32_b32 hwreg(HW_REG_MODE, 2, 2), 1",
                        "v_cvt_f16_f32 v2, v0",
                        "s_setreg_imm32_b32 hwreg(HW_REG_MODE, 2, 2), 2",
                        "v_cvt_f16_f32 v3, v0",
                        "s_setreg_imm32_b32 hwreg(HW_REG_MODE, 2, 2), 3",
                        "v_cvt_f16_f32 v4, v0",
                        "exp mrt0 v1, v2, v3, v4 done vm",
                    ],
                )
                want = tuple(ref_f32_to_f16(u32, m) for m in (RNE, RTP, RTN, RTZ))
                self.assertEqual(got, want)

    def test_s_round_mode_and_dynamic_s_setreg_b32(self) -> None:
        # +1.0 + 1 f32 ulp (0x3F800001) and -(1.0 + 1 f32 ulp) (0xBF800001)
        # s_round_mode bits [3:2] select FP16/FP64 mode (4=RTP, 8=RTN)
        got = _run_ps_mrt0(
            "ps_s_round_mode_directed",
            [
                "v_mov_b32 v0, 0x3f800001",
                "v_mov_b32 v1, 0xbf800001",
                "s_round_mode 4",  # RTP (+inf)
                "v_cvt_f16_f32 v2, v0",
                "v_cvt_f16_f32 v3, v1",
                "s_round_mode 8",  # RTN (-inf)
                "v_cvt_f16_f32 v4, v0",
                "v_cvt_f16_f32 v5, v1",
                "exp mrt0 v2, v3, v4, v5 done vm",
            ],
        )
        self.assertEqual(got, (0x3C01, 0xBC00, 0x3C00, 0xBC01))

        # Dynamic s_setreg_b32 from SGPR (with high bits masked via & 3)
        got_dyn = _run_ps_mrt0(
            "ps_dynamic_setreg_b32",
            [
                "v_mov_b32 v0, 0x3f800001",
                "v_mov_b32 v1, 0xbf800001",
                "s_mov_b32 s2, 0x101",  # & 3 == 1 (RTP)
                "s_setreg_b32 hwreg(HW_REG_MODE, 2, 2), s2",
                "v_cvt_f16_f32 v2, v0",
                "v_cvt_f16_f32 v3, v1",
                "s_mov_b32 s3, 0x202",  # & 3 == 2 (RTN)
                "s_setreg_b32 hwreg(HW_REG_MODE, 2, 2), s3",
                "v_cvt_f16_f32 v4, v0",
                "v_cvt_f16_f32 v5, v1",
                "exp mrt0 v2, v3, v4, v5 done vm",
            ],
        )
        self.assertEqual(got_dyn, (0x3C01, 0xBC00, 0x3C00, 0xBC01))

    def test_f16_omod_overflow_under_directed_modes(self) -> None:
        # 40960.0 (0x47200000 -> f16 0x7900) * 2 overflows f16:
        #   +40960 mul:2 -> +inf (0x7C00) in RNE/RTP, +65504 (0x7BFF) in RTN/RTZ
        #   -40960 mul:2 -> -inf (0xFC00) in RNE/RTN, -65504 (0xFBFF) in RTP/RTZ
        pos = _run_ps_mrt0(
            "ps_omod_pos",
            [
                "v_mov_b32 v0, 0x47200000",
                "s_setreg_imm32_b32 hwreg(HW_REG_MODE, 2, 2), 0",
                "v_cvt_f16_f32_e64 v1, v0 mul:2",
                "s_setreg_imm32_b32 hwreg(HW_REG_MODE, 2, 2), 1",
                "v_cvt_f16_f32_e64 v2, v0 mul:2",
                "s_setreg_imm32_b32 hwreg(HW_REG_MODE, 2, 2), 2",
                "v_cvt_f16_f32_e64 v3, v0 mul:2",
                "s_setreg_imm32_b32 hwreg(HW_REG_MODE, 2, 2), 3",
                "v_cvt_f16_f32_e64 v4, v0 mul:2",
                "exp mrt0 v1, v2, v3, v4 done vm",
            ],
        )
        self.assertEqual(pos, (0x7C00, 0x7C00, 0x7BFF, 0x7BFF))

        neg = _run_ps_mrt0(
            "ps_omod_neg",
            [
                "v_mov_b32 v0, 0xc7200000",
                "s_setreg_imm32_b32 hwreg(HW_REG_MODE, 2, 2), 0",
                "v_cvt_f16_f32_e64 v1, v0 mul:2",
                "s_setreg_imm32_b32 hwreg(HW_REG_MODE, 2, 2), 1",
                "v_cvt_f16_f32_e64 v2, v0 mul:2",
                "s_setreg_imm32_b32 hwreg(HW_REG_MODE, 2, 2), 2",
                "v_cvt_f16_f32_e64 v3, v0 mul:2",
                "s_setreg_imm32_b32 hwreg(HW_REG_MODE, 2, 2), 3",
                "v_cvt_f16_f32_e64 v4, v0 mul:2",
                "exp mrt0 v1, v2, v3, v4 done vm",
            ],
        )
        self.assertEqual(neg, (0xFC00, 0xFBFF, 0xFC00, 0xFBFF))


class DirectedF16ArithTests(unittest.TestCase):
    def test_v_add_f16_directed_and_cancellation_sign(self) -> None:
        # 2.0 (0x4000) + 2^-24 (0x0001): in f32 rounds to 2.0 (e = +2^-24 > 0).
        # Under RTP it must step up to 0x4001; under RNE/RTN/RTZ it stays 0x4000.
        got_pos = _run_ps_mrt0(
            "ps_add_f16_pos_tiny",
            [
                "v_mov_b32 v0, 0x4000",
                "v_mov_b32 v1, 0x0001",
                "s_setreg_imm32_b32 hwreg(HW_REG_MODE, 2, 2), 0",
                "v_add_f16 v2, v0, v1",
                "s_setreg_imm32_b32 hwreg(HW_REG_MODE, 2, 2), 1",
                "v_add_f16 v3, v0, v1",
                "s_setreg_imm32_b32 hwreg(HW_REG_MODE, 2, 2), 2",
                "v_add_f16 v4, v0, v1",
                "s_setreg_imm32_b32 hwreg(HW_REG_MODE, 2, 2), 3",
                "v_add_f16 v5, v0, v1",
                "exp mrt0 v2, v3, v4, v5 done vm",
            ],
        )
        self.assertEqual(got_pos, (0x4000, 0x4001, 0x4000, 0x4000))

        # -2.0 (0xC000) + (-2^-24) (0x8001): under RTN must step away to 0xC001.
        got_neg = _run_ps_mrt0(
            "ps_add_f16_neg_tiny",
            [
                "v_mov_b32 v0, 0xc000",
                "v_mov_b32 v1, 0x8001",
                "s_setreg_imm32_b32 hwreg(HW_REG_MODE, 2, 2), 0",
                "v_add_f16 v2, v0, v1",
                "s_setreg_imm32_b32 hwreg(HW_REG_MODE, 2, 2), 1",
                "v_add_f16 v3, v0, v1",
                "s_setreg_imm32_b32 hwreg(HW_REG_MODE, 2, 2), 2",
                "v_add_f16 v4, v0, v1",
                "s_setreg_imm32_b32 hwreg(HW_REG_MODE, 2, 2), 3",
                "v_add_f16 v5, v0, v1",
                "exp mrt0 v2, v3, v4, v5 done vm",
            ],
        )
        self.assertEqual(got_neg, (0xC000, 0xC000, 0xC001, 0xC000))

        # Exact cancellation (+1.0) + (-1.0): -0.0 (0x8000) only in RTN, +0.0 (0x0000) otherwise.
        got_cancel = _run_ps_mrt0(
            "ps_add_f16_cancel",
            [
                "v_mov_b32 v0, 0x3c00",
                "v_mov_b32 v1, 0xbc00",
                "s_setreg_imm32_b32 hwreg(HW_REG_MODE, 2, 2), 0",
                "v_add_f16 v2, v0, v1",
                "s_setreg_imm32_b32 hwreg(HW_REG_MODE, 2, 2), 1",
                "v_add_f16 v3, v0, v1",
                "s_setreg_imm32_b32 hwreg(HW_REG_MODE, 2, 2), 2",
                "v_add_f16 v4, v0, v1",
                "s_setreg_imm32_b32 hwreg(HW_REG_MODE, 2, 2), 3",
                "v_add_f16 v5, v0, v1",
                "exp mrt0 v2, v3, v4, v5 done vm",
            ],
        )
        self.assertEqual(got_cancel, (0x0000, 0x0000, 0x8000, 0x0000))


def _run_ps_f64_pair(name: str, body: list[str]) -> tuple[int, int]:
    spv = translate_lines(name, body + ["s_endpgm"])
    spirv_asm.assemble(spv)
    res = spirv_interp.run(spv)
    return res.vgpr[4] & 0xFFFFFFFF, res.vgpr[5] & 0xFFFFFFFF


class DirectedF64ArithTests(unittest.TestCase):
    def test_v_add_f64_directed_modes_and_overflow(self) -> None:
        # 1.0 + 2^-60: RN = 1.0 (0x3FF00000_00000000).
        # Under RTP it must step up by 1 ulp to 0x3FF00000_00000001; under RTN/RTZ/RNE it stays 1.0.
        lo1, hi1 = f64_words(1.0)
        lo2, hi2 = f64_words(math.ldexp(1.0, -60))
        for mode, want_lo in ((RNE, 0), (RTP, 1), (RTN, 0), (RTZ, 0)):
            with self.subTest(case="1+2^-60", mode=mode):
                got = _run_ps_f64_pair(
                    f"ps_f64_add_m{mode}",
                    [
                        f"v_mov_b32 v0, {lo1:#x}",
                        f"v_mov_b32 v1, {hi1:#x}",
                        f"v_mov_b32 v2, {lo2:#x}",
                        f"v_mov_b32 v3, {hi2:#x}",
                        f"s_setreg_imm32_b32 hwreg(HW_REG_MODE, 2, 2), {mode}",
                        "v_add_f64 v[4:5], v[0:1], v[2:3]",
                        "exp mrt0 v4, v5, v4, v5 done vm",
                    ],
                )
                self.assertEqual(got, (want_lo, hi1))

        # DBL_MAX + 1.0 -> +inf (0x7FF00000_00000000) under RTP, DBL_MAX under RNE/RTN/RTZ.
        dmax_lo, dmax_hi = 0xFFFFFFFF, 0x7FEFFFFF
        for mode, want_pair in (
            (RNE, (dmax_lo, dmax_hi)),
            (RTP, (0x00000000, 0x7FF00000)),
            (RTN, (dmax_lo, dmax_hi)),
            (RTZ, (dmax_lo, dmax_hi)),
        ):
            with self.subTest(case="dbl_max+1", mode=mode):
                got = _run_ps_f64_pair(
                    f"ps_f64_dmax_m{mode}",
                    [
                        f"v_mov_b32 v0, {dmax_lo:#x}",
                        f"v_mov_b32 v1, {dmax_hi:#x}",
                        f"v_mov_b32 v2, {lo1:#x}",
                        f"v_mov_b32 v3, {hi1:#x}",
                        f"s_setreg_imm32_b32 hwreg(HW_REG_MODE, 2, 2), {mode}",
                        "v_add_f64 v[4:5], v[0:1], v[2:3]",
                        "exp mrt0 v4, v5, v4, v5 done vm",
                    ],
                )
                self.assertEqual(got, want_pair)

        # Exact cancellation (+1.0) + (-1.0) -> -0.0 (0x80000000_00000000) only in RTN.
        nlo1, nhi1 = f64_words(-1.0)
        for mode, want_hi in ((RNE, 0x00000000), (RTP, 0x00000000), (RTN, 0x80000000), (RTZ, 0x00000000)):
            with self.subTest(case="f64_cancel", mode=mode):
                got = _run_ps_f64_pair(
                    f"ps_f64_cancel_m{mode}",
                    [
                        f"v_mov_b32 v0, {lo1:#x}",
                        f"v_mov_b32 v1, {hi1:#x}",
                        f"v_mov_b32 v2, {nlo1:#x}",
                        f"v_mov_b32 v3, {nhi1:#x}",
                        f"s_setreg_imm32_b32 hwreg(HW_REG_MODE, 2, 2), {mode}",
                        "v_add_f64 v[4:5], v[0:1], v[2:3]",
                        "exp mrt0 v4, v5, v4, v5 done vm",
                    ],
                )
                self.assertEqual(got, (0, want_hi))


if __name__ == "__main__":
    unittest.main()
