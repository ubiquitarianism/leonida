"""Synthetic legacy high-byte and REX register clobber regressions."""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "tools"), os.path.join(ROOT, "tests")]
import gdk_runtime_calls as g
import import_fail_fast as ff
from test_import_call_sites import build_image, TEXT_RVA


class HighByteWritesTest(unittest.TestCase):
    def test_legacy_byte_immediates_invalidate_parent(self):
        for reg in range(8):
            with self.subTest(reg=reg):
                insn = g.decode(bytes((0xB0 + reg, 0x55)), 0, 0)
                self.assertIsNotNone(insn)
                self.assertEqual(g.writes(insn), [reg % 4])

    def test_rex_uses_low_byte_registers(self):
        for rex in (0x40, 0x41, 0x48, 0x49):
            for reg in range(8):
                with self.subTest(rex=rex, reg=reg):
                    insn = g.decode(bytes((rex, 0xB0 + reg, 0x55)), 0, 0)
                    self.assertEqual(g.writes(insn), [reg + (8 if rex & 1 else 0)])

    def test_wider_immediates_keep_register_number(self):
        for reg in range(8):
            insn = g.decode(bytes((0xB8 + reg, 0, 0, 0, 0)), 0, 0)
            self.assertEqual(g.writes(insn), [reg])

    def test_ah_write_invalidates_import_result_but_spl_does_not(self):
        for prefix, expected in ((b"", None), (b"\x40", "eax == 0")):
            with self.subTest(prefix=prefix):
                data = bytearray(build_image([]))
                img = ff.pe_inspect.PeImage(data)
                rva = TEXT_RVA + 0x100
                at = img.offset(rva)
                # mov ah/spl, 0; test eax,eax; jne past int3; int3; ret
                code = prefix + b"\xb4\x00\x85\xc0\x75\x01\xcc\xc3"
                data[at:at + len(code)] = code
                self.assertEqual(ff.classify(ff.pe_inspect.PeImage(bytes(data)), rva), expected)


if __name__ == "__main__":
    unittest.main()
