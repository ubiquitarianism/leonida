"""A scalar tuple that runs into vcc is translated from the blob when llvm-mc cannot decode it.

llvm-mc rejects s_buffer_load_dwordx8 into s[100:107], a buffer resource in s[104:107] and an image resource in
s[100:107] (SGPRs 106 and 107 are vcc). It skips the first word as invalid and leaves the second word as junk, so
translate() refused every such shader ("reachable instruction with normalized encoding", or "fall-through into
undecodable words"; docs/shader_library.md section 5), or, when the junk matched the blob, translated a program
with the load missing. The words are now decoded from the blob as the VCCTuple lists the translator takes.

Hand-built listings and blobs of synthetic instructions (encodings from llvm-mc 19, the three tuple words patched in
their register field), no game data.

    python -I tests/semantic/test_vcc_tuple_fallback.py -v
"""
import os
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from support import rdna2spirv, translate_listing  # noqa: E402
import fssm_header  # noqa: E402
import spirv_mini_interp as si  # noqa: E402

NAME = "cs_vcc_tuple"
CODE_START = 0x148
BASE, OUT = 0x1000, 0x3000
INVALID = "; warning: invalid instruction encoding"

SET_SGPR_V = [  # s4..s7 = a V# over BASE (32 bytes), as tests/semantic/test_fspx_rules.py builds it
    ("s_mov_b32 s4, 0x1000", "ff 03 84 be 00 10 00 00"),
    ("s_mov_b32 s5, 0", "80 03 85 be"),
    ("s_mov_b32 s6, 32", "a0 03 86 be"),
    ("s_mov_b32 s7, 0x30000000", "ff 03 87 be 00 00 00 30"),
]
WAIT = ("s_waitcnt lgkmcnt(0)", "7f c0 8c bf")
WAIT_V = ("s_waitcnt vmcnt(0)", "70 3f 8c bf")
STORE_X4 = [
    ("v_mov_b32_e32 v6, 0x3000", "ff 02 0c 7e 00 30 00 00"),
    ("global_store_dwordx4 v6, v[10:13], s[0:1]", "00 80 78 dc 06 0a 00 00"),
    ("s_endpgm", "00 00 81 bf"),
]
# the words llvm-mc cannot decode (each is two dwords): the llvm-mc 19 encoding of the same instruction one tuple
# lower, with its register field patched into vcc
LOAD_X8_VCC = "02 19 2c f4 00 00 00 fa"        # s_buffer_load_dwordx8 [s100..s105,vcc_lo,vcc_hi], s[4:7], 0x0
LOAD_X4_VCC = "02 1a 28 f4 00 00 00 fa"        # s_buffer_load_dwordx4 [s104,s105,vcc_lo,vcc_hi], s[4:7], 0x0
LOAD_DW_FROM_VCC = "b4 02 20 f4 00 00 00 fa"   # s_buffer_load_dword s10, [s104,s105,vcc_lo,vcc_hi], 0x0
BUFFER_LOAD_VCC = "00 10 30 e0 01 0b 1a 80"    # buffer_load_dword v11, v1, [s104,s105,vcc_lo,vcc_hi], 0 offen
IMAGE_LOAD_VCC = "08 1f 00 f0 01 04 19 00"     # image_load v[4:7], v[1:2], [s100..s105,vcc_lo,vcc_hi] 2D unorm
PROGRAM_VARIANTS = ("invalid", "junk", "normalized")


def encoding(raw):
    return "; encoding: [" + ",".join(f"0x{b:02x}" for b in bytes.fromhex(raw)) + "]"


def listing_and_blob(program, variant="invalid"):
    """program: (text, hex) pairs; a pair whose text is None is a word llvm-mc rejects, listed as the variant says."""
    code = b"".join(bytes.fromhex(raw) for _, raw in program)
    lines = []
    for text, raw in program:
        if text is not None:
            lines.append(f"\t{text:<40}{encoding(raw)}")
            continue
        raw_bytes = bytes.fromhex(raw)
        lines.append(INVALID)
        if len(raw_bytes) == 4:
            continue
        second = raw_bytes[4:8]
        if variant == "invalid":
            lines.append(INVALID)
        elif variant == "junk":         # the second word decodes as an instruction of its own, bytes as stored
            lines.append(f"\ts_nop 0{'':<33}" + encoding(second.hex()))
        else:                           # printed with another encoding than the blob holds
            lines.append("\texp mrt0 v0, v0, v0, v0 done vm " + encoding("0f 18 00 f8 00 00 00 00"))
    head = f"; {NAME}  code 0x{CODE_START:x}..0x{CODE_START + len(code):x} (scan), gfx1013, 2 invalid words"
    header = bytearray(CODE_START)
    struct.pack_into("<4I", header, 0x138, 0x88, 32, 1, 1)
    struct.pack_into("<I", header, fssm_header.WAVE_FLAGS_OFFSET, 0x20)       # wave32
    return ((head + "\n" + "\n".join(lines) + "\n").encode(), bytes(header) + code)


def translate(program, variant="invalid"):
    data, blob = listing_and_blob(program, variant)
    return translate_listing(NAME, data, fssm=blob)


def run(spv, memory):
    mem = dict(memory)
    si.run_workgroup(spv, 1, 32, inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i},
                     push_constants=[0] * 36, memory=mem)
    return mem


class VccTupleFallbackTest(unittest.TestCase):
    def test_load_into_a_tuple_ending_in_vcc(self):
        program = SET_SGPR_V + [
            (None, LOAD_X8_VCC), WAIT,
            ("v_mov_b32_e32 v10, s100", "64 02 14 7e"), ("v_mov_b32_e32 v11, s105", "69 02 16 7e"),
            ("v_mov_b32_e32 v12, vcc_lo", "6a 02 18 7e"), ("v_mov_b32_e32 v13, vcc_hi", "6b 02 1a 7e"),
        ] + STORE_X4
        words = [0x11110000 + i for i in range(8)]
        for variant in PROGRAM_VARIANTS:
            with self.subTest(printed_as=variant):
                mem = run(translate(program, variant), {BASE + 4 * i: w for i, w in enumerate(words)})
                self.assertEqual([mem.get(OUT + 4 * i) for i in range(4)], [words[0], words[5], words[6], words[7]])

    def test_buffer_resource_in_vcc(self):
        # a V# loaded into [s104,s105,vcc_lo,vcc_hi], then read by s_buffer_load and by buffer_load
        program = SET_SGPR_V + [
            (None, LOAD_X4_VCC), WAIT,
            (None, LOAD_DW_FROM_VCC), WAIT,
            ("v_mov_b32_e32 v1, 4", "84 02 02 7e"),
            (None, BUFFER_LOAD_VCC), WAIT_V,
            ("v_mov_b32_e32 v10, s10", "0a 02 14 7e"),
            ("v_mov_b32_e32 v6, 0x3000", "ff 02 0c 7e 00 30 00 00"),
            ("global_store_dwordx2 v6, v[10:11], s[0:1]", "00 80 74 dc 06 0a 00 00"),
            ("s_endpgm", "00 00 81 bf"),
        ]
        memory = {BASE + 4 * i: w for i, w in enumerate([BASE + 0x100, 0, 16, 0x30000000])}
        memory[BASE + 0x100], memory[BASE + 0x104] = 0xCAFE1234, 0xBEEF5678
        for variant in PROGRAM_VARIANTS:
            with self.subTest(printed_as=variant):
                mem = run(translate(program, variant), memory)
                self.assertEqual([mem.get(OUT), mem.get(OUT + 4)], [0xCAFE1234, 0xBEEF5678])

    def test_image_resource_in_vcc(self):
        program = SET_SGPR_V + [
            (None, LOAD_X8_VCC), WAIT,
            ("v_mov_b32_e32 v1, 0", "80 02 02 7e"), ("v_mov_b32_e32 v2, 0", "80 02 04 7e"),
            (None, IMAGE_LOAD_VCC), WAIT_V,
            ("v_mov_b32_e32 v6, 0x3000", "ff 02 0c 7e 00 30 00 00"),
            ("global_store_dword v6, v4, s[0:1]", "00 80 70 dc 06 04 00 00"),
            ("s_endpgm", "00 00 81 bf"),
        ]
        for variant in PROGRAM_VARIANTS:
            with self.subTest(printed_as=variant):
                self.assertRegex(translate(program, variant), r"OpFunctionCall %\w+ %desc_lookup_fn")

    def test_load_with_register_offset_into_vcc(self):
        # The widened scalar load with an SGPR soffset and an immediate offset (the form luizwidmer/palmetto's
        # docs/boot_frontier.md measured in an FSPX program, built here from the field layout, not copied):
        # s_buffer_load_dwordx8 [s100..s105,vcc_lo,vcc_hi], s[4:7], s12 offset:0x40 reads BASE + 0x40 + s12.
        load = "02 19 2c f4 40 00 00 18"
        program = [SET_SGPR_V[0], SET_SGPR_V[1], ("s_mov_b32 s6, 0x100", "ff 03 86 be 00 01 00 00"), SET_SGPR_V[3],
                   ("s_mov_b32 s12, 32", "a0 03 8c be"),
                   (None, load), WAIT,
                   ("v_mov_b32_e32 v10, s100", "64 02 14 7e"), ("v_mov_b32_e32 v11, s105", "69 02 16 7e"),
                   ("v_mov_b32_e32 v12, vcc_lo", "6a 02 18 7e"), ("v_mov_b32_e32 v13, vcc_hi", "6b 02 1a 7e"),
                   ] + STORE_X4
        words = [0x22220000 + i for i in range(8)]
        for variant in PROGRAM_VARIANTS:
            with self.subTest(printed_as=variant):
                mem = run(translate(program, variant), {BASE + 0x60 + 4 * i: w for i, w in enumerate(words)})
                self.assertEqual([mem.get(OUT + 4 * i) for i in range(4)], [words[0], words[5], words[6], words[7]])

    def test_buffer_store_with_resource_in_vcc(self):
        # buffer_store_dword v11, v1, [s104,s105,vcc_lo,vcc_hi], 0 offen: a store through the widened resource too.
        store = "00 10 70 e0 01 0b 1a 80"
        program = SET_SGPR_V + [
            (None, LOAD_X4_VCC), WAIT,
            ("v_mov_b32_e32 v1, 4", "84 02 02 7e"),
            ("v_mov_b32_e32 v11, 0x5a5a1234", "ff 02 16 7e 34 12 5a 5a"),
            (None, store), WAIT_V,
            ("s_endpgm", "00 00 81 bf"),
        ]
        memory = {BASE + 4 * i: w for i, w in enumerate([OUT, 0, 16, 0x30000000])}
        for variant in PROGRAM_VARIANTS:
            with self.subTest(printed_as=variant):
                mem = run(translate(program, variant), memory)
                self.assertEqual(mem.get(OUT + 4), 0x5A5A1234)

    def test_junk_that_swallows_the_next_instructions(self):
        # The junk at the tuple's second word is longer than that word, so it covers the s_mov_b32 after the tuple
        # (and the first word of the 8-byte one after that, whose literal then prints as junk too). None of them is
        # printed at its own place; all of them are decoded from the blob.
        literal = ("s_mov_b32 s11, 0x12345678", "ff 03 8b be 78 56 34 12")
        swallowed = ("s_mov_b32 s10, s4", "04 03 8a be")
        tail = [WAIT, ("v_mov_b32_e32 v10, s100", "64 02 14 7e"), ("v_mov_b32_e32 v11, vcc_hi", "6b 02 16 7e"),
                ("v_mov_b32_e32 v12, s10", "0a 02 18 7e"), ("v_mov_b32_e32 v13, s11", "0b 02 1a 7e")] + STORE_X4
        program = SET_SGPR_V + [(None, LOAD_X8_VCC), swallowed, literal] + tail
        hidden = bytes.fromhex(LOAD_X8_VCC)[4:] + bytes.fromhex(swallowed[1]) + bytes.fromhex(literal[1])
        # an exp whose ignored bits are set is printed with other bytes than the blob holds
        normalized = bytes.fromhex("60 0d 00 f8 00 00 00 00")
        for name, junk in (("8 bytes", [hidden[:8]]), ("12 bytes, then 4", [hidden[:12], hidden[12:]]),
                           ("8 bytes, normalized", [normalized])):
            with self.subTest(junk=name):
                data, blob = listing_and_blob(program)
                lines = data.decode().splitlines()
                first = next(i for i, line in enumerate(lines) if line == INVALID)
                after = [] if sum(map(len, junk)) > 8 else lines[first + 3:first + 4]
                lines[first:first + 4] = [INVALID] + [f"\ts_nop 0{'':<33}" + encoding(j.hex()) for j in junk] + after
                spv = translate_listing(NAME, ("\n".join(lines) + "\n").encode(), fssm=blob)
                mem = run(spv, {BASE + 4 * i: 0x11110000 + i for i in range(8)})
                self.assertEqual([mem.get(OUT + 4 * i) for i in range(4)],
                                 [0x11110000, 0x11110007, 0x1000, 0x12345678])

    def test_swallowed_compare_in_wave64_writes_a_pair(self):
        # The junk exp covers the tuple's second word and the compare's first; the compare's second prints on its own.
        program = SET_SGPR_V + [
            ("s_mov_b32 s12, 0", "80 03 8c be"), ("s_mov_b32 s13, 0", "80 03 8d be"),
            (None, LOAD_X8_VCC),
            ("v_cmp_gt_u32_e64 s[12:13], 40, v0", "0c 00 c4 d4 a8 00 02 00"),
            WAIT,
            ("s_bcnt1_i32_b64 s10, s[12:13]", "0c 10 8a be"),
            ("v_mov_b32_e32 v10, s10", "0a 02 14 7e"),
            ("v_lshlrev_b32_e32 v6, 2, v0", "82 00 0c 34"),
            ("v_add_nc_u32_e32 v6, 0x3000, v6", "ff 0c 0c 4a 00 30 00 00"),
            ("global_store_dword v6, v10, s[0:1]", "00 80 70 dc 06 0a 00 00"),
            ("s_endpgm", "00 00 81 bf"),
        ]
        data, blob = listing_and_blob(program)
        blob = bytearray(blob)
        struct.pack_into("<4I", blob, 0x138, 0x88, 64, 1, 1)
        struct.pack_into("<I", blob, fssm_header.WAVE_FLAGS_OFFSET, 0)       # wave64
        lines = data.decode().splitlines()
        first = lines.index(INVALID)
        lines[first:first + 3] = [INVALID, "\texp mrt0 v0, v0, v0, v0 done vm " + encoding("0f 18 00 f8 00 00 00 00"),
                                  f"\ts_nop 0{'':<33}" + encoding("a8 00 02 00")]
        spv = translate_listing(NAME, ("\n".join(lines) + "\n").encode(), fssm=bytes(blob))
        mem = {BASE + 4 * i: 0 for i in range(8)}
        si.run_workgroup(spv, 64, 64, inputs_for=lambda i: {
            "%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i}, push_constants=[0] * 36, memory=mem)
        self.assertEqual([mem.get(OUT + 4 * i) for i in range(64)], [0x28] * 64)

    def test_rejected_word_after_a_tuple_is_still_refused(self):
        # inside no printed instruction: llvm-mc rejected the word itself, so it is not taken for swallowed code
        program = SET_SGPR_V + [(None, LOAD_X8_VCC), (None, "04 03 8a be 04 03 8a be"), WAIT] + STORE_X4
        with self.assertRaisesRegex(rdna2spirv.Unsupported, "fall-through into undecodable words"):
            translate(program)

    def test_other_undecodable_words_are_still_refused(self):
        # SHADER-813: a rejected word that is not a tuple into vcc (sdata 112, not a register) stays undecodable,
        # and the code that runs into it is refused
        word = "02 1c 2c f4 00 00 00 fa"
        program = SET_SGPR_V + [(None, word)] + STORE_X4
        with self.assertRaisesRegex(rdna2spirv.Unsupported, "fall-through into undecodable words"):
            translate(program)

    def test_nsa_mimg_with_vcc_tuple_does_not_read_past_code_end(self):
        # A 12-byte NSA1 image_load whose 3rd dword lies past code_end must not be restored across stop.
        mimg_nsa1 = bytes.fromhex("0a 1f 00 f0 01 04 19 00 02 00 00 00")
        self.assertEqual(rdna2spirv.restore_vcc_tuples([], mimg_nsa1, 0, 8), [])
        restored = rdna2spirv.restore_vcc_tuples([], mimg_nsa1, 0, 12)
        self.assertEqual(len(restored), 1)
        self.assertEqual((restored[0]["pc"], len(restored[0]["enc"]) // 2), (0, 12))

    def test_skipped_data_word_before_branch_target_is_not_restored(self):
        # An unreachable data word skipped by s_branch whose bytes equal the first dword of a vcc-tuple load must
        # not be restored across the branch target instruction that immediately follows it.
        program = SET_SGPR_V + [
            ("s_branch 1", "01 00 82 bf"),
            (None, "02 19 2c f4"),
            ("v_mov_b32_e32 v10, 7", "87 02 14 7e"),
            ("v_mov_b32_e32 v6, 0x3000", "ff 02 0c 7e 00 30 00 00"),
            ("global_store_dword v6, v10, s[0:1]", "00 80 70 dc 06 0a 00 00"),
            ("s_endpgm", "00 00 81 bf"),
        ]
        mem = run(translate(program), {})
        self.assertEqual(mem.get(OUT), 7)

    def test_skipped_data_word_before_branch_target_vcc_load(self):
        # Same skipped data word right before a real vcc-tuple load at the branch target: the data word is skipped
        # and the real vcc-tuple load at the target is restored cleanly across all three listing variants.
        program = SET_SGPR_V + [
            ("s_branch 1", "01 00 82 bf"),
            (None, "02 19 2c f4"),
            (None, LOAD_X8_VCC), WAIT,
            ("v_mov_b32_e32 v10, s100", "64 02 14 7e"), ("v_mov_b32_e32 v11, s105", "69 02 16 7e"),
            ("v_mov_b32_e32 v12, vcc_lo", "6a 02 18 7e"), ("v_mov_b32_e32 v13, vcc_hi", "6b 02 1a 7e"),
        ] + STORE_X4
        words = [0x22220000 + i for i in range(8)]
        for variant in PROGRAM_VARIANTS:
            with self.subTest(printed_as=variant):
                mem = run(translate(program, variant), {BASE + 4 * i: w for i, w in enumerate(words)})
                self.assertEqual([mem.get(OUT + 4 * i) for i in range(4)], [words[0], words[5], words[6], words[7]])

    def test_data_word_after_endpgm_before_branch_target_is_not_restored(self):
        # An unreachable data word after s_endpgm right before a branch target is likewise ignored.
        program = SET_SGPR_V + [
            ("s_cmp_eq_i32 s4, s4", "04 04 00 bf"),
            ("s_cbranch_scc1 2", "02 00 85 bf"),
            ("s_endpgm", "00 00 81 bf"),
            (None, "02 19 2c f4"),
            ("v_mov_b32_e32 v10, 7", "87 02 14 7e"),
            ("v_mov_b32_e32 v6, 0x3000", "ff 02 0c 7e 00 30 00 00"),
            ("global_store_dword v6, v10, s[0:1]", "00 80 70 dc 06 0a 00 00"),
            ("s_endpgm", "00 00 81 bf"),
        ]
        mem = run(translate(program), {})
        self.assertEqual(mem.get(OUT), 7)

    def test_fallthrough_into_data_word_before_branch_target_refused(self):
        # When fall-through reaches a data word equal to the first dword of a vcc-tuple load right before a branch
        # target, restore_vcc_tuples must not swallow the branch target; reachable_code refuses the fall-through.
        program = SET_SGPR_V + [
            ("s_cmp_eq_i32 s4, s4", "04 04 00 bf"),
            ("s_cbranch_scc1 1", "01 00 85 bf"),
            (None, "02 19 2c f4"),
            ("v_mov_b32_e32 v10, 7", "87 02 14 7e"),
            ("v_mov_b32_e32 v6, 0x3000", "ff 02 0c 7e 00 30 00 00"),
            ("global_store_dword v6, v10, s[0:1]", "00 80 70 dc 06 0a 00 00"),
            ("s_endpgm", "00 00 81 bf"),
        ]
        with self.assertRaisesRegex(rdna2spirv.Unsupported, "fall-through into undecodable words"):
            translate(program)


class ResyncTest(unittest.TestCase):
    """restore_vcc_tuples on hand-built instruction lists: where the words after a restored tuple are decoded, and
    where that stops."""
    LITERAL = "ff 03 8b be 78 56 34 12"             # s_mov_b32 s11, 0x12345678
    MOV = "04 03 8a be"                             # s_mov_b32 s10, s4
    END = "00 00 81 bf"                             # s_endpgm
    BAD = "ff ff ff ff"                             # no instruction

    def restore(self, words, printed, code_start=0, normalized=()):
        """printed, normalized: (pc, bytes) of the instructions llvm-mc listed; returns [(pc, size)] in order."""
        blob = bytes.fromhex("".join(words))
        insns = [{"pc": pc, "enc": blob[pc:pc + n].hex(), "op": "x", "args": []} for pc, n in printed]
        insns += [{"pc": pc, "enc": "00" * n, "op": "x", "args": [], "normalized": True} for pc, n in normalized]
        insns.sort(key=lambda i: i["pc"])
        out = rdna2spirv.restore_vcc_tuples(insns, blob, code_start, len(blob))
        return [(i["pc"], len(i["enc"]) // 2) for i in out]

    def test_stops_at_a_printed_instruction(self):
        # the tuple's junk covers words 1 and 2; the s_mov_b32 at 12 is printed, with its own operands, and stays
        words = [LOAD_X8_VCC, self.MOV, self.MOV, self.END]
        self.assertEqual(self.restore(words, [(4, 8), (12, 4), (16, 4)]), [(0, 8), (8, 4), (12, 4), (16, 4)])
        blob = bytes.fromhex("".join(words))
        insns = [{"pc": 4, "enc": blob[4:12].hex(), "op": "x", "args": []},
                 {"pc": 12, "enc": blob[12:16].hex(), "op": "printed", "args": []}]
        self.assertIn("printed", [i["op"] for i in rdna2spirv.restore_vcc_tuples(insns, blob, 0, len(blob))])

    def test_crosses_a_start_that_is_a_literal_printed_on_its_own(self):
        # The 8-byte instruction at 12 runs across a printed start (16): its own literal, which llvm-mc printed as an
        # instruction because the junk before it ended there. So a resync may not stop at every printed start.
        words = [LOAD_X8_VCC, self.MOV, self.LITERAL, self.END]
        self.assertEqual(self.restore(words, [(4, 12), (16, 4), (20, 4)]), [(0, 8), (8, 4), (12, 8), (20, 4)])

    def test_wave_size_reaches_the_decoder(self):
        blob = bytes.fromhex(LOAD_X8_VCC + "0c 00 c4 d4 a8 00 02 00" + self.END)   # v_cmp_gt_u32_e64 s12 / s[12:13]
        for wave32, n in ((True, 1), (False, 2)):
            with self.subTest(wave32=wave32):
                insns = [{"pc": 4, "enc": blob[4:16].hex(), "op": "x", "args": []},
                         {"pc": 16, "enc": self.END.replace(" ", ""), "op": "s_endpgm", "args": []}]
                out = rdna2spirv.restore_vcc_tuples(insns, blob, 0, len(blob), wave32=wave32)
                self.assertEqual((out[1]["pc"], out[1]["args"][0].get("n", 1)), (8, n))

    def branch(self, op, dwords):
        return {"s_branch": "%02x %02x 82 bf", "s_cbranch_scc1": "%02x %02x 85 bf"}[op] % (dwords & 255, dwords >> 8 & 255)

    def restore_ops(self, words, printed):
        blob = bytes.fromhex("".join(words))
        insns = [{"pc": pc, "enc": blob[pc:pc + n].hex(), "op": op, "args": args} for pc, n, op, args in printed]
        return [i["pc"] for i in rdna2spirv.restore_vcc_tuples(insns, blob, 0, len(blob))]

    def test_a_swallowed_branch_is_followed(self):
        # the s_branch at 8, in the first tuple's junk, is the only way to the second tuple at 16
        words = [LOAD_X8_VCC, self.branch("s_branch", 1), self.BAD, LOAD_X8_VCC, self.END]
        self.assertEqual(self.restore_ops(words, [(4, 8, "x", []), (24, 4, "s_endpgm", [])]), [0, 8, 16, 24])

    def test_a_branch_to_a_swallowed_instruction_does_not_hide_it(self):
        # 0: scc1 -> 16 (swallowed), then the jump to 24, from where the tuple at 8 is reached last
        imm = [{"k": "imm", "v": 3}]
        words = [self.branch("s_cbranch_scc1", 3), self.branch("s_branch", 4), LOAD_X8_VCC, self.MOV, self.END,
                 self.branch("s_cbranch_scc1", -6), self.END]
        printed = [(0, 4, "s_cbranch_scc1", imm), (4, 4, "s_branch", [{"k": "imm", "v": 4}]),
                   (12, 8, "x", []), (20, 4, "s_endpgm", []),
                   (24, 4, "s_cbranch_scc1", [{"k": "imm", "v": -5}]), (28, 4, "s_endpgm", [])]
        self.assertEqual(self.restore_ops(words, printed), [0, 4, 8, 16, 20, 24, 28])
        # the same with the branches to the two taken in the other order
        printed[0], printed[4] = (0, 4, "s_cbranch_scc1", [{"k": "imm", "v": 1}]), (24, 4, "s_cbranch_scc1",
                                                                                   [{"k": "imm", "v": -3}])
        self.assertEqual(self.restore_ops(words, printed), [0, 4, 8, 16, 20, 24, 28])

    def test_what_follows_a_branched_into_tuple_is_dropped_with_it(self):
        # a branch lands on the tuple's second word, so neither the tuple nor the words after it are taken for code
        words = [self.branch("s_cbranch_scc1", 1), LOAD_X8_VCC, self.MOV, self.END]
        printed = [(0, 4, "s_cbranch_scc1", [{"k": "imm", "v": 1}]), (8, 8, "x", []), (16, 4, "s_endpgm", [])]
        self.assertEqual(self.restore_ops(words, printed), [0, 8, 16])

    def test_drops_out_when_a_decode_fails(self):
        # a word no decoder takes inside the junk: the s_mov_b32 after it is not picked up again
        words = [LOAD_X8_VCC, self.BAD, self.MOV, self.END]
        self.assertEqual(self.restore(words, [(4, 12), (16, 4)]), [(0, 8), (16, 4)])

    def test_two_junk_regions_are_separate(self):
        # Two regions of normalized junk with matched instructions between them: the first ends at the match, so the
        # rejected word at 20, between the regions, is in neither and is not decoded after the second tuple.
        words = [LOAD_X8_VCC, self.MOV, LOAD_X8_VCC, self.MOV, self.MOV, self.END]
        self.assertEqual(self.restore(words, [(8, 4), (28, 4)], normalized=[(4, 4), (24, 4)]),
                         [(0, 8), (8, 4), (12, 8), (24, 4), (28, 4)])

    def test_nothing_is_decoded_before_a_tuple(self):
        # the first words are inside junk of something else: no tuple, so they stay as they are
        words = [self.MOV, self.MOV, self.END]
        self.assertEqual(self.restore(words, [], normalized=[(0, 8)]), [(0, 8)])

    def test_resync_to_the_last_word(self):
        self.assertEqual(self.restore([LOAD_X8_VCC, self.MOV, self.MOV], [(4, 12)]), [(0, 8), (8, 4), (12, 4)])

    def test_a_tuple_at_the_very_end_of_the_code(self):
        self.assertEqual(self.restore([self.MOV, LOAD_X8_VCC], [(0, 4)]), [(0, 4), (4, 8)])


if __name__ == "__main__":
    unittest.main()
