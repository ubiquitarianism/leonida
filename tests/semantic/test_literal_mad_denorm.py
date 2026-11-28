"""Literal MAD flushes inputs, the rounded product and the sum, as Mesa ACO documents.

Every case translates synthetic instructions and executes the emitted SPIR-V.
Expected words are hand-derived from binary32 with signed flush-to-zero; no
GPU, LLVM, game files or driver-specific denormal mode is needed.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import support


# (description, multiplicand, multiplier, addend, expected result)
CASES = (
    ("subnormal src0", 0x00000001, 0x7E800000, 0x00000000, 0x00000000),
    ("negative subnormal src0", 0x80000001, 0x7E800000, 0x80000000, 0x80000000),
    ("subnormal multiplier", 0x7E800000, 0x007FFFFF, 0x00000000, 0x00000000),
    ("negative subnormal multiplier", 0x7E800000, 0x807FFFFF, 0x80000000, 0x80000000),
    ("subnormal addend", 0x00000000, 0x3F800000, 0x00000001, 0x00000000),
    ("negative subnormal addend", 0x80000000, 0x3F800000, 0x80000001, 0x80000000),
    ("subnormal product", 0x00800000, 0x3F000000, 0x00000000, 0x00000000),
    ("negative subnormal product", 0x80800000, 0x3F000000, 0x80000000, 0x80000000),
    ("subnormal sum", 0x00800001, 0x3F800000, 0x80800000, 0x00000000),
    ("negative subnormal sum", 0x80800001, 0x3F800000, 0x00800000, 0x80000000),
    ("minimum normal", 0x00800000, 0x3F800000, 0x00000000, 0x00800000),
    ("negative minimum normal", 0x80800000, 0x3F800000, 0x80000000, 0x80800000),
    ("ordinary asymmetric operands", 0x40000000, 0x40400000, 0x40A00000, 0x41300000),
    ("separate product rounding", 0x3F800008, 0x3F7FFFF0, 0xBF800000, 0x00000000),
    ("positive infinity", 0x7F800000, 0x3F800000, 0x00000000, 0x7F800000),
    ("negative infinity", 0xFF800000, 0x3F800000, 0x00000000, 0xFF800000),
)


def program(op, x, y, z):
    # LLVM prints both instructions as x * y + z; K occupies different slots.
    operands = f"v1, {y:#x}, v3" if "madmk" in op else f"v1, v2, {z:#x}"
    return [
        f"v_mov_b32 v1, {x:#x}",
        f"v_mov_b32 v2, {y:#x}",
        f"v_mov_b32 v3, {z:#x}",
        f"{op} v0, {operands}",
        "exp mrt0 v0, v0, v0, v0 done vm",
        "s_endpgm",
    ]


class LiteralMadDenormalTests(unittest.TestCase):
    def test_boundary_values_and_controls(self):
        for op in ("v_madak_f32", "v_madmk_f32", "v_madak_f32_e32", "v_madmk_f32_e32"):
            for label, x, y, z, expected in CASES:
                with self.subTest(op=op, case=label):
                    spv = support.translate_lines("ps_literal_mad_denorm", program(op, x, y, z))
                    self.assertEqual(support.mrt0_words(spv), [expected] * 4)

    def test_nan_still_propagates(self):
        for op in ("v_madak_f32", "v_madmk_f32"):
            for x, y, z in ((0x7FC12345, 0x3F800000, 0), (0, 0x7F800000, 0)):
                with self.subTest(op=op, x=hex(x), y=hex(y)):
                    spv = support.translate_lines("ps_literal_mad_nan", program(op, x, y, z))
                    for word in support.mrt0_words(spv):
                        self.assertEqual(word & 0x7F800000, 0x7F800000)
                        self.assertNotEqual(word & 0x007FFFFF, 0)

    def test_hand_written_fixture(self):
        path = os.path.join(support.HERE, "fixtures", "literal_mad_denorm.asm")
        with open(path, encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip() and not line.lstrip().startswith(";")]
        spv = support.translate_lines("ps_literal_mad_denorm_fixture", lines)
        self.assertEqual(support.mrt0_words(spv), [0] * 4)


if __name__ == "__main__":
    unittest.main()
