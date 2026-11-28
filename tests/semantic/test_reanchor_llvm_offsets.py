"""Re-anchoring uses llvm-mc's own offsets for the words it rejected.

When a listing reports undecodable words, translate() puts every printed instruction back at its offset in the
blob. It used to search forward for the printed bytes, so a data word that llvm-mc printed as a copy of a later
instruction (its printed bytes are not the bytes it read) took that later instruction's place, and every
instruction after it lost its place. shader_ir.parse_file now reads the byte offset of each rejected word from the
llvm-mc messages (tools/shader_disasm.py feeds one line of "0xNN " bytes, five characters a byte), and the
translator skips exactly those words. Hand-built encodings, no game data.

    python -I tests/semantic/test_reanchor_llvm_offsets.py -v
"""
import os
import struct
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from support import mrt0_words, rdna2spirv, translate_listing  # noqa: E402

import shader_ir  # noqa: E402  (support puts tools/ on sys.path)

NAME = "ps_reanchor_llvm_offsets"
CODE_START = 0x100

MOV_V2 = ("v_mov_b32_e32 v2, 2.0", "f4 02 04 7e")
BRANCH = ("s_branch 2", "02 00 82 bf")          # to 0x108 + 2 * 4 = 0x110, over the two data words
BAD = "ff ff ff ff"                            # 0x108: rejected by llvm-mc, not printed
DATA = "12 34 56 78"                           # 0x10c: data that llvm-mc printed as the next instruction
MOV_V1 = ("v_mov_b32_e32 v1, v2", "02 03 02 7e")  # 0x110
EXP = ("exp mrt0 v1, v1, v1, v1 done vm", "0f 18 00 f8 01 01 01 01")
END = ("s_endpgm", "00 00 81 bf")


def code(data=DATA):
    return b"".join(bytes.fromhex(x) for x in (MOV_V2[1], BRANCH[1], BAD, data, MOV_V1[1], EXP[1], END[1]))


def listing(bad_offset=8, messages=True):
    printed = [MOV_V2, BRANCH, MOV_V1, MOV_V1, EXP, END]   # the data word printed with MOV_V1's text and bytes
    size = len(code())
    lines = [f"; {NAME}  code 0x{CODE_START:x}..0x{CODE_START + size:x} (scan), gfx1030, 1 invalid words"]
    for text, enc in printed:
        lines.append(f"\t{text:<40}; encoding: [{','.join(f'0x{b:02x}' for b in bytes.fromhex(enc))}]")
    if messages:
        lines += ["", "; --- llvm-mc messages ---",
                  f"; <stdin>:1:{bad_offset * 5 + 1}: warning: invalid instruction encoding"]
    return ("\n".join(lines) + "\n").encode("utf-8")


def blob(data=DATA):
    return bytes(CODE_START) + code(data)


def f2u(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]


class ReanchorFromLlvmOffsets(unittest.TestCase):
    def test_parse_file_reads_the_offsets(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, NAME + ".s")
            with open(path, "wb") as f:
                f.write(listing())
            info = shader_ir.parse_file(path)
        self.assertEqual(info["invalid"], 1)
        self.assertEqual(info["invalid_at"], [CODE_START + 8])

    def test_a_data_word_printed_as_a_later_instruction_keeps_its_place(self):
        out = mrt0_words(translate_listing(NAME, listing(), fssm=blob()))
        self.assertEqual(out[0], f2u(2.0))

    def test_messages_that_do_not_match_the_blob_fall_back(self):
        # The offset names a word whose bytes are the instruction itself: it is not skipped, so a listing whose
        # messages are wrong (or hand-written) still anchors as before.
        with self.assertRaises(rdna2spirv.Unsupported):
            translate_listing(NAME, listing(bad_offset=0), fssm=blob())


if __name__ == "__main__":
    unittest.main()
