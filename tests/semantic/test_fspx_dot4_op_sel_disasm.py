"""The FidelityFX upscaler's programs (docs/fspx.md) hold 89 v_dot4_i32_i8 words with op_sel 2 and op_sel_hi 5, which
llvm-mc does not decode: it rejects the first dword, reads the second as another instruction, and the translator then
refuses the program ("fall-through into undecodable words"), so 9 of the 29 programs had no pipeline and the DLL's
next pipeline call faulted. tools/shader_disasm.py now prints such a word as the instruction it is, with its
op_sel/op_sel_hi fields and its real encoding, so the translator reads it and checks it against the blob.

Each case assembles a small program with llvm-mc, turns one v_dot4_i32_i8 into the op_sel 2 / op_sel_hi 5 form by
setting those fields in its two dwords, disassembles the bytes with shader_disasm, and translates the listing in the
FSPX layout (disasm/, fspx/<name>.code and .json). The expected words are the ones measured on gfx1036
(MEASURED_OP_SEL in test_alu_dot4_i8.py). They need llvm-mc with the AMDGPU target (LLVM_MC, or llvm-mc on PATH) and
skip without it. No GPU and no game files.

    python -I tests/semantic/test_fspx_dot4_op_sel_disasm.py
"""
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.dont_write_bytecode = True
sys.path.insert(0, os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools"))
sys.path.insert(0, HERE)

import rdna2spirv  # noqa: E402
import shader_disasm  # noqa: E402
import spirv_mini_interp as si  # noqa: E402
from test_alu_dot4_i8 import MEASURED_OP_SEL, ref  # noqa: E402

MCPU = "gfx1030"      # fspx_unpack.py's default: LLVM decodes v_dot4_i32_i8 for it and not for gfx1013
STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)


def assemble(line):
    """The bytes llvm-mc assembles one instruction line to."""
    p = subprocess.run(shader_disasm.MC + ["--assemble", "-triple=amdgcn--amdpal", f"-mcpu={MCPU}", "-show-encoding"],
                       input="\t" + line + "\n", capture_output=True, text=True, errors="replace")
    match = shader_disasm.ENCODING.search(p.stdout)
    if p.returncode or not match:
        raise AssertionError(f"llvm-mc cannot assemble {line!r}: {p.stderr.strip()}")
    return bytes(int(b, 16) for b in match.group(1).split(","))


def llvm_decodes_v_dot4():
    try:
        return shader_disasm.run_mc(assemble("v_dot4_i32_i8 v8, v1, v2, v3"), MCPU)[1] == 0
    except (OSError, AssertionError):
        return False


def with_op_sel(word_pair, op_sel, op_sel_hi):
    """The two dwords of a plain v_dot4_i32_i8 (op_sel 0, op_sel_hi 7) with these fields: op_sel in bits 13:11 and
    op_sel_hi[2] in bit 14 of the first dword, op_sel_hi[1:0] in bits 28:27 of the second."""
    w0, w1 = struct.unpack("<2I", word_pair)
    w0 = w0 & ~(7 << 11) & ~(1 << 14) | op_sel << 11 | (op_sel_hi >> 2) << 14
    w1 = w1 & ~(3 << 27) | (op_sel_hi & 3) << 27
    return struct.pack("<2I", w0, w1)


def program(a, b, addend, middle):
    """Code bytes of: v8 = the v_dot4_i32_i8 given by `middle` (its two dwords), v9 = the plain v_dot4_i32_i8 on the
    same registers, both written to mrt0. Returns (code, byte offset of the first dot4)."""
    head = b"".join(assemble(line) for line in (
        f"v_mov_b32_e32 v1, 0x{a:08x}", f"v_mov_b32_e32 v2, 0x{b:08x}", f"v_mov_b32_e32 v3, 0x{addend:08x}"))
    tail = b"".join(assemble(line) for line in (
        "v_dot4_i32_i8 v9, v1, v2, v3", "exp mrt0 v8, v9, v8, v9 done vm", "s_endpgm"))
    return head + middle + tail, len(head)


def translate_fspx(name, listing, code):
    """SPIR-V assembly for a listing in the FSPX layout, with the code bytes it was disassembled from beside it."""
    with tempfile.TemporaryDirectory() as tmp:
        for d in ("disasm", "fspx"):
            os.mkdir(os.path.join(tmp, d))
        with open(os.path.join(tmp, name + ".dxbc"), "wb") as f:
            f.write(STUB_DXBC)
        with open(os.path.join(tmp, "fspx", name + ".code"), "wb") as f:
            f.write(code)
        with open(os.path.join(tmp, "fspx", name + ".json"), "w", encoding="utf-8", newline="\n") as f:
            json.dump({}, f)
        src = os.path.join(tmp, "disasm", name + ".s")
        with open(src, "w", encoding="utf-8", newline="\n") as f:
            f.write(listing)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            return rdna2spirv.translate(src)


def listing_of(name, code):
    asm, bad, err = shader_disasm.disasm(code, MCPU)
    return f"; {name}  code 0x0..0x{len(code):x} (fspx), {MCPU}, {bad} invalid words\n" + asm, bad


@unittest.skipUnless(llvm_decodes_v_dot4(), "needs llvm-mc with the AMDGPU target that decodes v_dot4_i32_i8")
class Dot4OpSelListing(unittest.TestCase):
    def plain_pair(self):
        return assemble("v_dot4_i32_i8 v8, v1, v2, v3")

    def test_the_listing_carries_the_op_sel_word_and_the_program_runs(self):
        for a, b, addend, measured in MEASURED_OP_SEL:
            with self.subTest(a=f"{a:#010x}", b=f"{b:#010x}", addend=f"{addend:#010x}"):
                middle = with_op_sel(self.plain_pair(), 2, 5)
                code, at = program(a, b, addend, middle)
                listing, bad = listing_of("ps_fspx_dot4_op_sel", code)
                self.assertEqual(bad, 0, listing)
                lines = [ln for ln in listing.splitlines() if "v_dot4_i32_i8" in ln]
                self.assertEqual(len(lines), 2, listing)
                self.assertIn("v_dot4_i32_i8 v8, v1, v2, v3 op_sel:[0,1,0] op_sel_hi:[1,0,1]", lines[0])
                self.assertIn("encoding: [" + ",".join(f"0x{x:02x}" for x in middle) + "]", lines[0])
                self.assertNotIn("op_sel", lines[1])        # the plain instruction next to it stays plain
                words = support_words(translate_fspx("ps_fspx_dot4_op_sel", listing, code))
                self.assertEqual(words, [measured, ref(a, b, addend), measured, ref(a, b, addend)])

    def test_other_op_sel_values_stay_undecodable(self):
        # Only op_sel 2 / op_sel_hi 5 was measured on hardware; anything else is still rejected, never guessed.
        for op_sel, op_sel_hi in ((1, 5), (4, 5), (2, 7), (0, 5), (2, 4)):
            with self.subTest(op_sel=op_sel, op_sel_hi=op_sel_hi):
                code, _ = program(1, 1, 0, with_op_sel(self.plain_pair(), op_sel, op_sel_hi))
                listing, bad = listing_of("ps_fspx_dot4_other", code)
                self.assertNotIn("op_sel:[0,1,0] op_sel_hi:[1,0,1]", listing)
                with self.assertRaises(rdna2spirv.Unsupported):   # undecodable, or printed without a dropped field
                    translate_fspx("ps_fspx_dot4_other", listing, code)

    def test_a_clamp_or_neg_bit_on_the_word_stays_undecodable(self):
        for name, shift, dword in (("clamp", 15, 0), ("neg_hi", 8, 0), ("neg_lo", 29, 1)):
            with self.subTest(field=name):
                middle = list(struct.unpack("<2I", with_op_sel(self.plain_pair(), 2, 5)))
                middle[dword] |= 1 << shift
                code, _ = program(1, 1, 0, struct.pack("<2I", *middle))
                listing, bad = listing_of("ps_fspx_dot4_mods", code)
                self.assertGreaterEqual(bad, 1)
                self.assertNotIn("op_sel:[0,1,0]", listing)

    def test_data_that_looks_like_the_word_is_left_alone(self):
        # The same two dwords as a 32-bit literal and the instruction after it decode with no invalid word, so they
        # are not touched: only a word llvm-mc rejected is read as v_dot4_i32_i8.
        pair = struct.unpack("<2I", with_op_sel(self.plain_pair(), 2, 5))
        code = (assemble("v_mov_b32_e32 v4, 0x%08x" % pair[0]) + assemble("v_fmac_legacy_f32 v7, v1, v2") +
                assemble("s_endpgm"))
        self.assertEqual(struct.unpack_from("<2I", code, 4), pair)
        listing, bad = listing_of("ps_fspx_dot4_literal", code)
        self.assertEqual(bad, 0, listing)
        self.assertNotIn("op_sel", listing)


def support_words(spv):
    out = si.run_fragment(spv)
    return list(out["%out_mrt0"])


if __name__ == "__main__":
    unittest.main()
