"""A signed-byte dot with an inline integer keeps its stored op_sel_hi when the listing re-anchors exactly.

llvm-mc prints `v_dot4_i32_i8` with an inline integer S1 and a stored op_sel_hi of 5 as the default op_sel_hi (7),
so its printed bytes differ from the blob in bit 60. restore_dot4_inline_sel repairs that in the path for listings
without undecodable words and in the forward search. A listing that reports undecodable words with llvm-mc's own
offsets re-anchors exactly (test_reanchor_llvm_offsets.py), and that path did not try the repair: the instruction
was marked normalized and the whole program refused ("reachable instruction with normalized encoding"). One of the
FidelityFX DLL's programs has this form and a rejected word, so the title got no pipeline for it. The same
instruction must give the same result through both paths. Hand-built encodings, no game data.

    python -I tests/semantic/test_reanchor_dot4_inline_sel.py -v
"""
import os
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from support import mrt0_words, rdna2spirv, translate_listing  # noqa: E402

NAME = "ps_reanchor_dot4_inline_sel"
CODE_START = 0x100


def le(*words):
    return " ".join(f"{b:02x}" for w in words for b in struct.pack("<I", w))


# v_dot4_i32_i8 v1, v2, 10, v1: VOP3P opcode 0x16, op_sel_hi[2] (bit 14) set; S0 v2, S1 inline 10 (138), S2 v1.
DOT_W0 = 0xCC164001
DOT_W1 = 258 | (138 << 9) | (257 << 18) | (1 << 27)      # op_sel_hi[0] set, op_sel_hi[1] clear: 5
DOT_RAW = le(DOT_W0, DOT_W1)
DOT_PRINTED = le(DOT_W0, DOT_W1 | (1 << 28))              # llvm-mc prints the default op_sel_hi, 7
DOT_TEXT = "v_dot4_i32_i8 v1, v2, 10, v1"

MOV_V2 = ("v_mov_b32_e32 v2, 0x1020304", "ff 02 04 7e 04 03 02 01")
MOV_V1 = ("v_mov_b32_e32 v1, 0", "80 02 02 7e")
BRANCH = ("s_branch 1", "01 00 82 bf")          # over the one rejected word
BAD = "ff ff ff ff"                             # rejected by llvm-mc, not printed
EXP = ("exp mrt0 v1, v1, v1, v1 done vm", "0f 18 00 f8 01 01 01 01")
END = ("s_endpgm", "00 00 81 bf")


def code(with_bad):
    parts = [MOV_V2[1], MOV_V1[1]] + ([BRANCH[1], BAD] if with_bad else []) + [DOT_RAW, EXP[1], END[1]]
    return b"".join(bytes.fromhex(x) for x in parts)


def listing(with_bad):
    printed = [MOV_V2, MOV_V1] + ([BRANCH] if with_bad else []) + [(DOT_TEXT, DOT_PRINTED), EXP, END]
    size = len(code(with_bad))
    invalid = 1 if with_bad else 0
    lines = [f"; {NAME}  code 0x{CODE_START:x}..0x{CODE_START + size:x} (scan), gfx1030, {invalid} invalid words"]
    for text, enc in printed:
        lines.append(f"\t{text:<40}; encoding: [{','.join(f'0x{b:02x}' for b in bytes.fromhex(enc))}]")
    if with_bad:
        bad_offset = 8 + 4 + 4                   # after the two moves and the branch
        lines += ["", "; --- llvm-mc messages ---",
                  f"; <stdin>:1:{bad_offset * 5 + 1}: warning: invalid instruction encoding"]
    return ("\n".join(lines) + "\n").encode("utf-8")


def run(with_bad):
    return mrt0_words(translate_listing(NAME, listing(with_bad), fssm=bytes(CODE_START) + code(with_bad)))


class Dot4InlineSelExactAnchor(unittest.TestCase):
    def test_exact_anchor_translates_the_stored_op_sel_hi(self):
        # Before: Unsupported("reachable instruction with normalized encoding").
        out = run(with_bad=True)
        self.assertEqual(out[0], run(with_bad=False)[0])

    def test_the_repair_is_the_stored_form_not_the_printed_one(self):
        # S1 is the inline 10 and S0 the bytes 4, 3, 2, 1: the default op_sel_hi (the printed form) would give
        # 4 * 10 = 40; the stored op_sel_hi takes the low half of S1 for bytes 2..3 as well.
        stored = run(with_bad=True)[0]
        printed_only = mrt0_words(rdna2spirv_translate_text(
            "\n".join(["v_mov_b32_e32 v2, 0x1020304", "v_mov_b32_e32 v1, 0",
                       DOT_TEXT + " op_sel_hi:[1,0,1]"])))[0]
        self.assertEqual(stored, printed_only)

    def test_other_mismatches_still_refuse(self):
        # A dot whose S1 is a register is not the inline form: the exact path still refuses a printed mismatch.
        global DOT_RAW, DOT_PRINTED
        saved = DOT_RAW, DOT_PRINTED
        w1 = 258 | (259 << 9) | (257 << 18) | (1 << 27)
        DOT_RAW, DOT_PRINTED = le(DOT_W0, w1), le(DOT_W0, w1 | (1 << 28))
        try:
            with self.assertRaises(rdna2spirv.Unsupported):
                run(with_bad=True)
        finally:
            DOT_RAW, DOT_PRINTED = saved


def rdna2spirv_translate_text(body):
    """A pixel shader from instruction text alone, exported through mrt0 like the listings above."""
    from support import translate_lines  # noqa: E402
    return translate_lines(NAME + "_text", body.split("\n") + ["exp mrt0 v1, v1, v1, v1 done vm", "s_endpgm"])


if __name__ == "__main__":
    unittest.main()
