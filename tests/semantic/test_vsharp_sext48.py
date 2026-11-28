"""Imported buffer V#s carry a signed 48-bit offset from the Vulkan heap base."""
from pathlib import Path
import re
import unittest

import support


class VSharpSignExtensionTest(unittest.TestCase):
    def test_negative_base_and_carry_out_of_bit_47(self):
        name = "ps_vsharp_sext48"
        fixture = Path(support.HERE) / "fixtures" / f"{name}.asm"
        spv = support.translate_listing(name, fixture.read_bytes())

        # Both effective VAs must be sign-extended by shifting 16 bits in each direction before rebasing.
        shift = re.search(r"^(%\w+) = OpConstant %uint 16$", spv, re.M)
        self.assertIsNotNone(shift)
        extended = re.findall(
            rf"(%\w+) = OpShiftLeftLogical %ulong %\w+ {shift[1]}\s+"
            rf"(%\w+) = OpShiftRightArithmetic %ulong \1 {shift[1]}\s+"
            r"%\w+ = OpIAdd %ulong %heap_base \2\b",
            spv,
        )
        self.assertEqual(len(extended), 2)

        # The second base+offset sum carries out of the low 48 bits and lands on the heap base itself.
        heap = 0x100002000
        below, at_base = 0x12345678, 0x9abcdef0
        words = support.mrt0_words(
            spv, push_constants=[0] * 32 + [heap & 0xffffffff, heap >> 32],
            memory={heap - 48: below, heap: at_base},
        )
        self.assertEqual(words, [below, at_base, below, at_base])

        translator = support.rdna2spirv
        metadata = translator.shader_meta(translator.translate.last_gen, spv)
        self.assertIs(metadata.get("vsharp_base_sign_extended"), True)


if __name__ == "__main__":
    unittest.main()
