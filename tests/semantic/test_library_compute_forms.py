"""Hand-written pipeline-library cases. No game, GPU or LLVM installation needed."""
import struct
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from test_fspx_rules import run_cs
from unittest.mock import patch
import support
from test_wave64_emulate_opt import run_emulated_workgroup
from test_fspx_wave_size import fssm_header, translate as translate_fspx
import spirv_mini_interp
import fspx_unpack
import rdna2_decode
import rdna2spirv
import shader_library


def fixture(name):
    return [s for line in (HERE / "fixtures" / (name + ".asm")).read_text(encoding="utf-8").splitlines()
            if (s := line.partition(";")[0].strip())]


def raw_words(lines):
    return b"".join(struct.pack("<I", int(word, 0)) for line in lines if line.startswith(".long ")
                    for word in line[6:].split(","))


class LibraryComputeForms(unittest.TestCase):
    def test_short_prologue_has_two_literals(self):
        lines = fixture("fspx_short_prologue")
        body = struct.pack("<II", 0xbf800000, 0xbf810000)
        self.assertEqual(lines[-2:], ["s_nop 0", "s_endpgm"])
        code = raw_words(lines) + body
        self.assertEqual(fspx_unpack.code_range(code), (24, 32))
        self.assertEqual(shader_library.code_range(code), (24, 32))
        ordinary = bytearray(code)
        struct.pack_into("<I", ordinary, 16, 0)  # ordinary VOP3, no literal in its third source
        self.assertEqual(fspx_unpack.code_range(ordinary)[0], 0)
        self.assertEqual(fspx_unpack.code_range(code[:20])[0], 0)

    def test_recovered_vcc_tuple(self):
        code = raw_words(fixture("vcc_tuple_word"))
        self.assertEqual(rdna2spirv.restore_vcc_tuples([], code, 0, len(code))[0]["args"][0]["r"],
                         [104, 105, "vcc_lo", "vcc_hi"])

    def run_wave32(self, name, memory):
        spv, wave64 = translate_fspx(fixture(name)[:-1], wave32=True)
        self.assertFalse(wave64)
        spirv_mini_interp.run_workgroup(spv, 32, 32, inputs_for=lambda i: {
            "%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%local_index": i, "%sg_lane": i,
        }, push_constants=[0] * 36, memory=memory)
        return memory

    def test_mask_bits_form_a_wave32_pointer(self):
        result = self.run_wave32("smem_vcc_mask_pointer", {0x1000: 0x654321})
        self.assertEqual(result.get(0x3000), 0x654321)

    def test_pair_copy_xor_pointer_and_mask_uses(self):
        result = self.run_wave32("smem_vcc_pair", {0x300001000: 0xaabbccdd})
        for lane in range(32):
            addr = 0x3000 + 16 * lane
            self.assertEqual([result.get(addr + 4 * i) for i in range(4)],
                             [0x1040, 1, 0xaabbccdd, (0x1040 >> lane) & 1])

    def test_wave64_scalar_mask_bits(self):
        for subgroup_size in (32, 64):
            with self.subTest(subgroup=subgroup_size), patch.object(rdna2spirv, "WAVE64_EMULATE", subgroup_size == 32):
                spv = support.translate_lines("cs_scalar_mask_bits", fixture("wave64_scalar_mask_bits"),
                                              fssm=fssm_header(False))
                pcs = [0, 0, 0xffffffff, 0x31016fac] + [0] * 32
                if subgroup_size == 32:
                    memory, _ = run_emulated_workgroup(spv, (64, 1, 1), push_constants=pcs)
                else:
                    memory = {}
                    spirv_mini_interp.run_workgroup(spv, 64, 64, inputs_for=lambda i: {
                        "%local_id": (i, 0, 0), "%local_index": i, "%sg_lane": i,
                        "%wg_id": (0, 0, 0)}, push_constants=pcs, memory=memory)
                for lane in range(64):
                    self.assertEqual([memory[lane * 16 + i * 4] for i in range(4)],
                                     [0xffffffff, 54, 23, 2 + ((0x80000021 >> (lane // 2)) & 1)])

    def test_vopc_dpp_encoded_fields_and_recovery(self):
        # Hand-encode VOPC gt.f32 with v12 as src1, v7 as src0. Reverse the four/eight lanes.
        msb = (0x3e << 25) | (4 << 17) | (12 << 9)
        dpp8 = sum((7 - i) << (8 + 3 * i) for i in range(8))
        for tag, ext, control in ((250, 0xff080000 | (0x1b << 8), "quad_perm:[3,2,1,0]"),
                                  (233, dpp8, "dpp8:[7,6,5,4,3,2,1,0]")):
            blob = struct.pack("<II", msb | tag, ext | 7)
            for wave32 in (False, True):
                row, = rdna2_decode.decode_bytes(blob, wave32=wave32)["insns"]
                self.assertEqual((row["op"], row["args"][0]["t"], row["args"][1]["r"], row["args"][2]["r"]),
                                 ("v_cmp_gt_f32_dpp", "vcc_lo" if wave32 else "vcc", 7, 12))
                self.assertIn(control, row["mods"])
            # a listing entry as llvm-mc prints the second word: restore_vcc_tuples reads its op, as it does a real one
            junk = {"pc": 4, "enc": blob[4:].hex(), "op": "s_nop", "args": [{"k": "imm", "v": 0}], "mods": []}
            recovered, = rdna2spirv.restore_vcc_tuples([junk], blob, 0, len(blob), wave32=wave32)
            self.assertEqual((recovered["pc"], recovered["enc"]), (0, blob.hex()))
            self.assertFalse(rdna2_decode.decode_bytes(blob[:4])["insns"])

    def test_vopcx_dpp_recovery(self):
        # SHADER-VOPCX-DPP: the same pass recovers v_cmpx in DPP form (opcode 0x14, v_cmpx_gt_f32), which prints no
        # destination: the game's form, quad_perm:[2,3,0,1] bound_ctrl:1 fi:1 with src1 negated (fa 02 28 7c 01 4e 4c ff).
        blob = bytes.fromhex("fa02287c014e4cff")
        row, = rdna2_decode.decode_bytes(blob, wave32=True)["insns"]
        self.assertEqual((row["op"], len(row["args"]), row["args"][0]["r"]), ("v_cmpx_gt_f32_dpp", 2, 1))
        junk = {"pc": 4, "enc": blob[4:].hex(), "op": "s_nop", "args": [{"k": "imm", "v": 0}], "mods": []}
        recovered, = rdna2spirv.restore_vcc_tuples([junk], blob, 0, len(blob), wave32=True)
        self.assertEqual((recovered["pc"], recovered["enc"], recovered["op"]), (0, blob.hex(), "v_cmpx_gt_f32_dpp"))

    def test_vopc_dpp_reads_the_selected_lane(self):
        spv, _ = translate_fspx(fixture("vopc_dpp_compare")[:-1], wave32=False)
        memory = {}
        spirv_mini_interp.run_workgroup(spv, 64, 64, inputs_for=lambda i: {
            "%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i},
            push_constants=[0, 0, 0xffffffff, 0x31016fac] + [0] * 32, memory=memory)
        for lane in range(64):
            self.assertEqual((memory[8 * lane], memory[8 * lane + 4]), (int(lane % 4 < 2), int(lane % 8 < 4)))

    def test_vcc_predicate_users_keep_the_lane_mask(self):
        import shader_ir
        import wave64_emulate_opt

        for predicate in ("v_cndmask_b32_e32 v1, 0, 1, vcc", "s_mov_b32 exec_lo, vcc_lo",
                          "s_cbranch_vccz 1"):
            with self.subTest(predicate=predicate):
                block = [shader_ir.parse_line(line) for line in (
                    "s_mov_b64 vcc, s[8:9]", "s_add_u32 s10, vcc_hi, 1", predicate)]
                wave64_emulate_opt.mark_vcc_data_moves([block], [[]])
                self.assertNotIn("vcc_data_pair", block[0])

        blocks = [[shader_ir.parse_line(line) for line in lines] for lines in (
            ["s_mov_b64 vcc, s[8:9]", "s_add_u32 s10, vcc_hi, 1"], ["s_cbranch_vccz 1"])]
        wave64_emulate_opt.mark_vcc_data_moves(blocks, [[1], []])
        self.assertNotIn("vcc_data_pair", blocks[0][0])

    def test_copying_mask_words_to_a_tuple_requires_wave_exchange(self):
        import shader_ir
        import wave64_emulate_opt as opt
        ins = shader_ir.parse_line("s_mov_b64 vcc, s[18:19]")
        ins["vcc_data_pair"] = True
        masks = {id(ins): {18: "mask_lo", 19: "mask_hi"}}
        self.assertTrue(opt._insn_may_barrier(ins, masks, set()))
        self.assertFalse(opt._classify_block_body([ins], masks, set(), set())[0])
        self.assertFalse(opt._insn_may_barrier(ins, {}, set()))

    def test_wave64_predicates_in_wave32_style_listing(self):
        body = fixture("wave64_valu_mask_width")
        push = [0, 0, 0xffffffff, 0x31016fac] + [0] * 32
        with patch.object(rdna2spirv, "WAVE64_EMULATE", True):
            asm = support.translate_lines("cs_valu_mask_width", body, fssm=fssm_header(False))
        memory, _ = run_emulated_workgroup(asm, (64, 1, 1), push_constants=push)
        for lane in range(64):
            self.assertEqual([memory[16 * lane + 4 * i] for i in range(3)],
                             [int(lane < 36), int(lane >= 33), 7 if lane == 36 else 0])

    def test_numeric_vcc_moves_in_buffer_tuple(self):
        body = fixture("vcc_scalar_tuple_moves")
        push = [0, 0, 0xffffffff, 0x31016fac] + [0] * 32
        seed = [0x12345678, 0x25ac87d2, 0xace750bd, 0xfedcba98]
        for wave in (32, 64):
            with self.subTest(wave=wave):
                memory = {0x4000 + 4 * i: value for i, value in enumerate(seed)}
                if wave == 32:
                    asm, _ = translate_fspx(body[:-1], wave32=True)
                    spirv_mini_interp.run_workgroup(asm, 32, 32, inputs_for=lambda lane: {
                        "%local_id": (lane, 0, 0), "%local_index": lane, "%sg_lane": lane,
                        "%wg_id": (0, 0, 0)}, push_constants=push, memory=memory)
                else:
                    with patch.object(rdna2spirv, "WAVE64_EMULATE", True):
                        asm = support.translate_lines("cs_vcc_word_moves", body, fssm=fssm_header(False))
                    memory, _ = run_emulated_workgroup(asm, (64, 1, 1), push_constants=push, memory=memory)
                for lane in range(wave):
                    self.assertEqual([memory.get(16 * lane + 4 * j) for j in range(4)],
                                     [seed[0], seed[1], seed[1] + 3, seed[3]])

    def test_scalar_pointer_in_vcc(self):
        body = fixture("smem_vcc_pointer")
        words = {0x1004: 0x12345678, 0x1008: 0x89123456}
        result = run_cs(body[:-1], words)
        self.assertEqual([result.get(0x3000), result.get(0x3004)], list(words.values()))


if __name__ == "__main__":
    unittest.main()
