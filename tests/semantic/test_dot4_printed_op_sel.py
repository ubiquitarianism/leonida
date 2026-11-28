"""A v_dot4_i32_i8 whose op_sel/op_sel_hi llvm-mc printed as the defaults: only the measured forms translate.

llvm-mc decodes a VOP3P v_dot4_i32_i8 with op_sel_hi other than [1,1,1] (op_sel bit 1 clear) but prints and
re-encodes it with op_sel 0 and op_sel_hi 7, so the listing names halves the instruction does not read. The
translator takes the stored field back for one form only: op_sel_hi 5 with an inline integer second source
(rdna2spirv.restore_dot4_inline_sel, counted over the FidelityFX programs; docs/design.md, "Signed-byte dot
products"). The one form measured on hardware, op_sel 2 / op_sel_hi 5, is a word llvm-mc rejects, which
tools/shader_disasm.py prints (test_fspx_dot4_op_sel_disasm.py). Every other form, such as op_sel_hi 5 with a VGPR
second source, is not measured and stays refused. Hand-built listing and blob, the encodings assembled from the VOP3P
field layout; no game data.

    python -I tests/semantic/test_dot4_printed_op_sel.py -v
"""
import os
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from support import mrt0_words, rdna2spirv, translate_listing  # noqa: E402

NAME = "ps_dot4_printed_op_sel"
CODE_START = 0x100
A, B, C = 0x80FF7F01, 0x12F0FE7F, 0x00001000
INLINE = 5                     # inline integer constant 5: source field 128 + 5


def src_field(src):
    """VOP3P source field: a VGPR number, or ("int", n) for an inline integer constant 0..64."""
    return 128 + src[1] if isinstance(src, tuple) else 256 + src


def dot4(op_sel=0, op_sel_hi=7, dst=0, srcs=(1, 2, 3)):
    """VOP3P v_dot4_i32_i8 (opcode 0x16) vdst, srcs with the given op_sel and op_sel_hi."""
    w0 = 0xCC000000 | (0x16 << 16) | ((op_sel_hi >> 2) & 1) << 14 | (op_sel & 7) << 11 | dst
    w1 = ((op_sel_hi & 3) << 27 | src_field(srcs[2]) << 18 | src_field(srcs[1]) << 9 | src_field(srcs[0]))
    return struct.pack("<2I", w0, w1).hex(" ")


def v_mov(reg, value):
    return (f"v_mov_b32_e32 v{reg}, 0x{value:x}", f"ff {2 * reg:02x} 02 7e " + struct.pack("<I", value).hex(" "))


SETUP = [v_mov(1, A), v_mov(2, B), v_mov(3, C)]
TAIL = [("exp mrt0 v0, v0, v0, v0 done vm", "0f 18 00 f8 00 00 00 00"), ("s_endpgm", "00 00 81 bf")]
PRINTED_VGPR = ("v_dot4_i32_i8 v0, v1, v2, v3", dot4())
PRINTED_INLINE = (f"v_dot4_i32_i8 v0, v1, {INLINE}, v3", dot4(srcs=(1, ("int", INLINE), 3)))


def listing(program, end):
    lines = [f"; {NAME}  code 0x{CODE_START:x}..0x{end:x} (scan), gfx1030, 0 invalid words"]
    for text, enc in program:
        lines.append(f"\t{text:<40}; encoding: [{','.join(f'0x{b:02x}' for b in bytes.fromhex(enc))}]")
    return ("\n".join(lines) + "\n").encode("utf-8")


def ref(a, b, addend, op_sel=0, op_sel_hi=7):
    """Byte pairs 0..1 read the halves op_sel names, pairs 2..3 the halves op_sel_hi names (S0 bit 0, S1 bit 1)."""
    def byte(x, off):
        v = (x >> off) & 0xFF
        return v - 256 if v & 0x80 else v
    total = addend
    for k in range(4):
        half = (op_sel if k < 2 else op_sel_hi)
        xo = 16 * ((half >> 0) & 1) + 8 * (k & 1)
        yo = 16 * ((half >> 1) & 1) + 8 * (k & 1)
        total += byte(a, xo) * byte(b, yo)
    return total & 0xFFFFFFFF


def translate(stored_dot4, printed=PRINTED_VGPR, reanchor=False):
    listed = SETUP + [printed] + TAIL
    stored = SETUP + [("", stored_dot4)] + TAIL
    code = b"".join(bytes.fromhex(enc) for _, enc in stored)
    data = listing(listed, CODE_START + len(code))
    blob = bytes(CODE_START) + code
    if reanchor:   # a listing with an undecodable word after the code goes through the re-anchoring path
        data += b"; warning: invalid instruction encoding\n"
        blob += b"\xff" * 4
    return translate_listing(NAME, data, fssm=blob)


class Dot4PrintedOpSelTest(unittest.TestCase):
    def test_reference_is_the_plain_dot_product_for_the_defaults(self):
        sb = [lambda x, k=k: (((x >> 8 * k) & 0xFF) ^ 0x80) - 0x80 for k in range(4)]
        self.assertEqual(ref(A, B, C), (sum(f(A) * f(B) for f in sb) + C) & 0xFFFFFFFF)

    def test_default_fields_translate_as_printed(self):
        self.assertEqual(mrt0_words(translate(dot4())), [ref(A, B, C)] * 4)

    def test_inline_integer_s1_keeps_its_stored_op_sel_hi(self):
        # op_sel_hi [1,0,1] with an inline integer S1: byte pairs 2..3 of S1 come from its low half, the halves the
        # measured op_sel 2 / op_sel_hi 5 form selects. llvm-mc prints [1,1,1]; restore_dot4_inline_sel takes it back.
        stored = dot4(op_sel_hi=5, srcs=(1, ("int", INLINE), 3))
        for reanchor in (False, True):
            with self.subTest(reanchor=reanchor):
                got = mrt0_words(translate(stored, PRINTED_INLINE, reanchor))
                self.assertEqual(got, [ref(A, INLINE, C, op_sel_hi=5)] * 4)
        self.assertNotEqual(ref(A, INLINE, C, op_sel_hi=5), ref(A, INLINE, C))

    def test_unmeasured_forms_stay_refused(self):
        # Not measured on hardware and not the inline-integer form: refused, in a clean listing (a dropped field)
        # and when re-anchoring (the stored word never matches the printed one).
        for name, stored in (("op_sel_hi 5, VGPR S1", dot4(op_sel_hi=5)),
                             ("op_sel 1 / op_sel_hi 6", dot4(op_sel=1, op_sel_hi=6)),
                             ("op_sel_hi 4", dot4(op_sel_hi=4)),
                             ("non-default S2 half", dot4(op_sel_hi=3))):
            for reanchor in (False, True):
                with self.subTest(form=name, reanchor=reanchor):
                    with self.assertRaises(rdna2spirv.Unsupported) as cm:
                        translate(stored, reanchor=reanchor)
                    if not reanchor:
                        self.assertIn("dropped", str(cm.exception))

    def test_inline_form_with_another_difference_is_refused(self):
        # The stored word names v4 as S2, the printed one v3: not just the op_sel_hi field.
        with self.assertRaises(rdna2spirv.Unsupported) as cm:
            translate(dot4(op_sel_hi=5, srcs=(1, ("int", INLINE), 4)), PRINTED_INLINE)
        self.assertIn("dropped", str(cm.exception))

    def test_the_general_rule_is_not_there(self):
        self.assertFalse(hasattr(rdna2spirv, "dot4_op_sel_fields"))
        self.assertTrue(callable(rdna2spirv.restore_dot4_inline_sel))


if __name__ == "__main__":
    unittest.main()
