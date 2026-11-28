"""Unit tests for tools/hw_oracle/asm.py."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools" / "hw_oracle"))

import asm  # noqa: E402


class TestHwOracleAsm(unittest.TestCase):
    def test_family_classification_and_invalid_arch(self) -> None:
        self.assertEqual(asm.family("gfx1010"), "gfx10")
        self.assertEqual(asm.family("gfx1150"), "gfx11")
        self.assertEqual(asm.family("gfx1201"), "gfx12")
        for bad in ("gfx900", "gfx1300", "invalid", "gfx11"):
            with self.assertRaises(ValueError):
                asm.family(bad)

    def test_literals_inline_vs_distinct_32bit(self) -> None:
        self.assertEqual(asm.literals("s_mov_b32 s4, 64"), set())
        self.assertEqual(asm.literals("s_mov_b32 s4, -16"), set())
        self.assertEqual(asm.literals("s_movk_i32 s4, 0x1234"), set())
        self.assertEqual(asm.literals("s_cmpk_eq_i32 s4, 100"), set())
        self.assertEqual(asm.literals("s_add_i32 s4, -1, 0xffffffff"), {0xFFFFFFFF})
        self.assertEqual(asm.literals("s_add_i32 s4, 0x12345678, 0x87654321"), {0x12345678, 0x87654321})

    def test_skip_reason_per_family_and_two_literals(self) -> None:
        self.assertIsNone(asm.skip_reason(["s_ff0_i32_b32 s4, s5"], "gfx1010"))
        self.assertIn("GFX11", asm.skip_reason(["s_ff0_i32_b32 s4, s5"], "gfx1150"))
        self.assertIsNone(asm.skip_reason(["s_cmpk_eq_i32 s4, 100"], "gfx1150"))
        self.assertIn("GFX12", asm.skip_reason(["s_cmpk_eq_i32 s4, 100"], "gfx1201"))
        self.assertIn(
            "two different literals",
            asm.skip_reason(["s_add_i32 s4, 0x12345678, 0x87654321"], "gfx1010"),
        )


if __name__ == "__main__":
    unittest.main()
