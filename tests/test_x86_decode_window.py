"""Synthetic bounds checks for the shared x86 decoder; standard library only."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "tools"))
from gdk_runtime_calls import decode


class ObservedBytes(bytes):
    def __new__(cls, data):
        obj = super().__new__(cls, data)
        obj.reads = []
        return obj

    def __getitem__(self, key):
        if isinstance(key, int):
            self.reads.append(key)
        return super().__getitem__(key)


class DecodeWindowTest(unittest.TestCase):
    def test_long_prefix_run_reads_only_instruction_window(self):
        for offset in (0, 7):
            with self.subTest(offset=offset):
                data = ObservedBytes(b"\x90" * offset + b"\x66" * 4096 + b"\x90")
                self.assertIsNone(decode(data, offset, 0x1000))
                self.assertLessEqual(max(data.reads), offset + 14)
                self.assertLessEqual(len(data.reads), 45)

    def test_fifteen_byte_instruction_and_rva(self):
        data = b"\xcc" * 7 + b"\x66" * 14 + b"\x90"
        insn = decode(data, 7, 0x1000)
        self.assertIsNotNone(insn)
        self.assertEqual((insn.rva, insn.size, insn.op), (0x1007, 15, 0x90))

    def test_operands_cannot_extend_past_window(self):
        # MOV rax, imm64: ten bytes, plus five/six address-size prefixes.
        mov = b"\x48\xb8" + bytes(8)
        self.assertEqual(decode(b"\x67" * 5 + mov, 0, 0).size, 15)
        self.assertIsNone(decode(b"\x67" * 6 + mov, 0, 0))
        self.assertIsNone(decode(mov[:-1], 0, 0))

    def test_invalid_offsets_do_not_read_buffer(self):
        data = ObservedBytes(b"\x90")
        for offset in (-2, -1, 1, 2):
            self.assertIsNone(decode(data, offset, 0))
        self.assertEqual(data.reads, [])
        self.assertIsNone(decode(b"", 0, 0))


if __name__ == "__main__":
    unittest.main()
