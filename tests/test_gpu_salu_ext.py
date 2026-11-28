"""GPU-SALU-EXT: the scalar fixture's expected words against the ISA reference, its readback and the runner verdicts,
without needing a GPU."""
import contextlib
import io
from pathlib import Path
import re
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tools"))
import run_gpu_tests as rgt
import check
import salu_diff

FIXTURE = ROOT / "runtime" / "test" / "shaders" / "cs_w32_salu_ext.asm"
HEAP = 80
# The ops the fixture is for: none of them is run by another fixture in runtime/test/shaders.
OPS = ("s_sub_i32", "s_andn2_b32", "s_orn2_b32", "s_xor_b32", "s_ashr_i32", "s_mul_hi_u32", "s_lshl3_add_u32",
       "s_lshl4_add_u32", "s_pack_ll_b32_b16", "s_pack_hh_b32_b16", "s_abs_i32", "s_bcnt0_i32_b32", "s_bitset0_b32",
       "s_bitset1_b32", "s_brev_b32", "s_flbit_i32", "s_flbit_i32_b32", "s_sext_i32_i8", "s_cmp_eq_i32",
       "s_cmp_lg_i32", "s_cmp_gt_i32", "s_cmp_ge_i32", "s_cmp_lt_i32", "s_cmp_le_i32", "s_cmp_gt_u32", "s_cmp_le_u32",
       "s_bitcmp0_b32", "s_bitcmp1_b32", "s_addk_i32", "s_mulk_i32", "s_cmovk_i32", "s_ashr_i64", "s_lshl_b64",
       "s_lshr_b64", "s_bcnt0_i32_b64", "s_bitset0_b64", "s_cmov_b64", "s_ff0_i32_b64")
# What the fixture uses besides them: copies, the SCC it sets before an op that must keep it, and the SCC capture.
HELPERS = ("s_mov_b32", "s_cmp_eq_u32", "s_cmp_lg_u32", "s_cselect_b32", "s_or_b32")


def fixture():
    """(header lines, the scalar program, the SGPRs stored in order, the offsets they are stored at)."""
    lines = [ln.strip() for ln in FIXTURE.read_text(encoding="utf-8").splitlines()]
    header = [ln for ln in lines if ln.startswith(";")]
    code = [ln for ln in lines if ln and not ln.startswith(";")]
    first_store = code.index("v_mov_b32_e32 v21, 0")
    registers, offsets = [], []
    stores = code[first_store + 1:-1]
    for move, store in zip(stores[0::2], stores[1::2]):
        registers.append(int(re.fullmatch(r"v_mov_b32_e32 v20, s(\d+)", move).group(1)))
        offset = re.fullmatch(r"buffer_store_dword v20, v21, s\[0:3\], 0 offen(?: offset:(\d+))?", store).group(1)
        offsets.append(int(offset or 0))
    return header, code[:first_store], registers, offsets, code[-1]


def reference_words(program, registers, seeds=None):
    """The words tools/salu_diff.py's reference gives for the stored SGPRs: the program runs after `s_mov_b32` of
    the sixteen operand words into s4..s19, which is how the reference takes what the GPU run gets from VKC_SGPRS."""
    seeds = check.SALU_EXT_SGPRS if seeds is None else seeds
    start = [f"s_mov_b32 s{4 + n}, {word:#x}" for n, word in enumerate(seeds)]
    words = []
    for at in range(0, len(registers), 4):   # the reference exports four words a run, v8..v11
        chunk = registers[at:at + 4]
        watch = [f"v_mov_b32 v{8 + n}, s{register}" for n, register in enumerate(chunk)]
        words += salu_diff.reference(start + program + watch)[:len(chunk)]
    return words


class SaluExtOracle(unittest.TestCase):
    def test_expected_words_are_the_reference_s(self):
        _, program, registers, _, _ = fixture()
        self.assertEqual(len(registers), 73)
        self.assertEqual(len(set(registers)), 73)
        self.assertEqual(reference_words(program, registers), check.SALU_EXT_WORDS)

    def test_the_words_depend_on_the_operands(self):
        """With other operand words the reference gives other results, so the literals are not a property of the
        program alone: a run that lost VKC_SGPRS cannot pass."""
        _, program, registers, _, _ = fixture()
        zeros = reference_words(program, registers, [0] * 16)
        self.assertNotEqual(zeros, check.SALU_EXT_WORDS)
        self.assertGreater(sum(a != b for a, b in zip(zeros, check.SALU_EXT_WORDS)), 40)

    def test_every_op_is_run_and_nothing_unmodelled(self):
        _, program, _, _, _ = fixture()
        used = [ln.split(" ", 1)[0] for ln in program]
        self.assertEqual(sorted(set(used) - set(HELPERS)), sorted(OPS))
        self.assertEqual(len(OPS), 38)
        for op in set(used):
            self.assertIn(op, salu_diff.OPS, op)

    def test_both_values_of_scc_are_seen_for_every_op_that_writes_it(self):
        """Each op that writes SCC is run on operands that make it 1 and on operands that make it 0, and each
        compare both ways; the bit captured right after the op is what is checked."""
        _, program, _, _, _ = fixture()
        start = [f"s_mov_b32 s{4 + n}, {word:#x}" for n, word in enumerate(check.SALU_EXT_SGPRS)]
        seen = {}
        for at, line in enumerate(program):
            op = line.split(" ", 1)[0]
            if op not in OPS or not salu_diff.writes_scc(op):
                continue
            self.assertRegex(program[at + 1], r"s_cselect_b32 s104, 0x[0-9a-f]+, 0", line)
            scc = salu_diff.reference(start + program[:at + 1] + ["s_cselect_b32 s30, 1, 0", "v_mov_b32 v8, s30"])[0]
            seen.setdefault(op, set()).add(scc)
        writers = [op for op in OPS if salu_diff.writes_scc(op)]
        self.assertEqual(len(writers), 24)
        self.assertEqual({op: seen.get(op) for op in writers}, {op: {0, 1} for op in writers})

    def test_scc_is_set_both_ways_before_the_ops_that_keep_it(self):
        _, program, _, _, _ = fixture()
        keepers = [op for op in OPS if not salu_diff.writes_scc(op)]
        self.assertEqual(len(keepers), 14)
        before = {}
        for at, line in enumerate(program):
            op = line.split(" ", 1)[0]
            if op in keepers:
                self.assertIn(program[at - 1], ("s_cmp_eq_u32 s4, s4", "s_cmp_lg_u32 s4, s4"), line)
                before.setdefault(op, []).append(program[at - 1])
        self.assertEqual(sorted(before), sorted(keepers))
        # The two that read SCC run once with it set and once with it clear.
        for op in ("s_cmovk_i32", "s_cmov_b64"):
            self.assertEqual(sorted(before[op]), ["s_cmp_eq_u32 s4, s4", "s_cmp_lg_u32 s4, s4"])

    def test_values_worked_out_by_hand(self):
        """A word of each kind, by the SGPR it is stored from, so the check does not depend on the store order."""
        _, _, registers, _, _ = fixture()
        word = dict(zip(registers, check.SALU_EXT_WORDS))
        pair = lambda low: word[low] | word[low + 1] << 32     # noqa: E731
        self.assertEqual(word[32], 0xFFFFFFFF)                  # 0x7fffffff - 0x80000000, wrapped
        self.assertEqual(word[40], 0xFC000000)                  # 0x80000000 >> (37 & 31), arithmetic
        self.assertEqual((word[42], word[44]), (0, (0x123456780 + 0xDEADBEEF) & 0xFFFFFFFF))
        self.assertEqual((word[46], word[47], word[48], word[26]), (0x80000000, 1, 0, 0x12345678))   # s_abs_i32
        self.assertEqual((word[51], word[52], word[53]), (0x80000000, 0x12345678 - 0x8000, 0x7FFFFFFF))
        self.assertEqual(word[54], (0xDEADBEEF * 0x12345678) >> 32)
        self.assertEqual((word[55], word[56]), (0xBEEF5678, 0xDEAD1234))
        self.assertEqual(word[57], int(f"{0x12345678:032b}"[::-1], 2))
        self.assertEqual((word[58], word[59], word[60]), (1, 0xFFFFFFFF, 31))        # s_flbit_i32
        self.assertEqual((word[61], word[62], word[27]), (16, 0xFFFFFFFF, 0))        # s_flbit_i32_b32
        self.assertEqual(word[67], (0x12345678 * -3) & 0xFFFFFFFF)
        # The two ops that read SCC: the run with SCC set takes its source, the other keeps what was there.
        _, program, _, _, _ = fixture()
        took = {line: program[program.index(line) - 1] == "s_cmp_eq_u32 s4, s4"
                for line in ("s_cmovk_i32 s68, 0x8000", "s_cmovk_i32 s69, 0x8000",
                             "s_cmov_b64 s[70:71], s[14:15]", "s_cmov_b64 s[72:73], s[14:15]")}
        self.assertEqual(sorted(took.values()), [False, False, True, True])
        self.assertNotEqual(took["s_cmovk_i32 s68, 0x8000"], took["s_cmovk_i32 s69, 0x8000"])
        self.assertNotEqual(took["s_cmov_b64 s[70:71], s[14:15]"], took["s_cmov_b64 s[72:73], s[14:15]"])
        for register in (68, 69):
            self.assertEqual(word[register],
                             0xFFFF8000 if took[f"s_cmovk_i32 s{register}, 0x8000"] else 0x12345678)
        for low in (70, 72):
            self.assertEqual(pair(low), 0x8000000100000001
                             if took[f"s_cmov_b64 s[{low}:{low + 1}], s[14:15]"] else 0x40)
        self.assertEqual(pair(74), 0xFFFFFFFFFFFF0000 & ~(1 << 37))
        self.assertEqual(pair(28), 0xFFFFFFFFFFFFFFFF & ~(1 << (71 & 63)))
        self.assertEqual((word[76], word[77], word[82]), (0, 32, 0xFFFFFFFF))        # s_ff0_i32_b64
        value, mask = 0x8000000100000001, (1 << 64) - 1
        self.assertEqual(pair(84), mask)                                             # a negative pair >> 37
        self.assertEqual(pair(86), ((value - (1 << 64)) >> (71 & 63)) & mask)
        self.assertEqual((pair(88), pair(94), pair(98)), (0, 0, 0))                  # the three zero results
        self.assertEqual((pair(90), pair(92)), ((value << 37) & mask, (value << 7) & mask))
        self.assertEqual(pair(96), 0xFFFFFFFFFFFF0000 >> 37)
        self.assertEqual((word[24], word[25]), (16, 0))                              # zeros of the two pairs
        # SCC words: 23 results of which 13 set it, 21 compares, 24 kept values in turn from 1, ten 64-bit results.
        self.assertEqual([bin(word[r]).count("1") for r in (100, 101, 102, 103)], [13, 11, 12, 6])
        self.assertEqual(word[102], 0x555555)


class SaluExtFixture(unittest.TestCase):
    def test_header_and_layout(self):
        header, program, registers, offsets, last = fixture()
        self.assertIn("; group: 1,1,1", header)
        self.assertIn("; rsrc2: 0x80a8", header)            # twenty user SGPRs: the V# and s4..s19
        self.assertEqual(offsets, [4 * n for n in range(73)])
        self.assertEqual(last, "s_endpgm")
        self.assertEqual(registers[-4:], [100, 101, 102, 103])
        self.assertEqual(program[:4], [f"s_mov_b32 s{n}, 0" for n in (100, 101, 102, 103)])
        # No operand is a literal: every source of the 38 ops is an SGPR, apart from the immediate of the three SOPK
        # forms, whose other operand is the register they change.
        for line in program:
            op, _, rest = line.partition(" ")
            if op not in OPS:
                continue
            sources = rest.split(", ")[1:] if not op.startswith(("s_cmp_", "s_bitcmp")) else rest.split(", ")
            if op in ("s_addk_i32", "s_mulk_i32", "s_cmovk_i32"):
                self.assertRegex(sources[0], r"0x[0-9a-f]+", line)
                continue
            for source in sources:
                self.assertRegex(source, r"s\d+|s\[\d+:\d+\]", line)
        # The header's table names every stored word with the value check.py expects.
        table = "\n".join(header)
        for register in (32, 40, 54, 67, 82, 25):
            index = registers.index(register)
            self.assertRegex(table, rf"out\[{index}\] +s_\w+ s{register}, .*0x{check.SALU_EXT_WORDS[index]:08x}")
        for low in (70, 84, 98):
            index = registers.index(low)
            self.assertRegex(table, rf"out\[{index},{index + 1}\] +s_\w+ s\[{low}:{low + 1}\], .*"
                                    rf"0x{check.SALU_EXT_WORDS[index]:08x} 0x{check.SALU_EXT_WORDS[index + 1]:08x}")
        for remark, line in (("takes", "s_cmp_eq_u32 s4, s4"), ("keeps", "s_cmp_lg_u32 s4, s4")):
            for op in ("s_cmovk_i32 s68, 0x8000", "s_cmovk_i32 s69, 0x8000"):
                said = next(ln for ln in header if op in ln)
                self.assertEqual(remark in said, program[program.index(op) - 1] == line, said)
        for word in check.SALU_EXT_WORDS[-4:]:
            self.assertIn(f"0x{word:08x}", table)


class SaluExtReadback(unittest.TestCase):
    def run_readback(self, mutate=None, code=0, output="", missing=False, lose_sgprs=False):
        seen = []

        def launch(*args, env):
            seen.append(env)
            self.assertEqual(args[1:], (str(Path("unused") / "cs_w32_salu_ext.spv"), "1", "dump"))
            self.assertEqual(env["VKC_SGPRS"], ",".join(f"{w:#x}" for w in check.SALU_EXT_SGPRS))
            self.assertEqual(env["VKC_SUBGROUP_SIZE"], "32")
            self.assertEqual(env["VKC_OUT_WORDS"], str(HEAP))
            self.assertEqual(env["LEONIDA_VK_DEVICE"], "synthetic device")
            words = list(check.SALU_EXT_WORDS) + [0xEEEEEEEE] * (HEAP - len(check.SALU_EXT_WORDS))
            if lose_sgprs:
                _, program, registers, _, _ = fixture()
                words[:len(registers)] = reference_words(program, registers, [0] * 16)
            if mutate:
                mutate(words)
            if not missing:
                Path(env["VKC_DUMP"]).write_bytes(struct.pack(f"<{len(words)}I", *words))
            return code, output

        with patch.object(check, "run", side_effect=launch):
            result = check.salu_ext_test("unused", "unused", {"LEONIDA_VK_DEVICE": "synthetic device"})
        self.assertEqual(len(seen), 1)
        return result

    def test_complete_readback_passes(self):
        self.assertEqual(self.run_readback(), "")

    def test_one_wrong_word_fails(self):
        for index in (0, 37, 69, 72, 73, HEAP - 1):   # results, SCC words, and the fill after them
            with self.subTest(word=index):
                output = self.run_readback(lambda data: data.__setitem__(index, data[index] ^ 4))
                self.assertIn(f"word {index}:", output)

    def test_a_run_without_the_operands_fails(self):
        self.assertIn("word 0: expected 0xffffffff, got 0x00000000", self.run_readback(lose_sgprs=True))

    def test_truncated_and_missing_dump_fail(self):
        self.assertIn(f"expected {4 * HEAP} dump bytes, got {4 * HEAP - 4}", self.run_readback(lambda data: data.pop()))
        self.assertIn("no VKC_DUMP", self.run_readback(missing=True))

    def test_unsupported_module_skips_but_nonzero_exit_fails(self):
        self.assertIsInstance(self.run_readback(code=77, output="unsupported: no Int64\n"), check.Skip)
        self.assertIn("exit 1", self.run_readback(code=1, output="dispatch failed"))


class SaluExtRunner(unittest.TestCase):
    def test_filtered_runner_counts_pass_failure_and_skip(self):
        for detail, expected, verdict in (("", 0, "1/1 passed"), ("word 69: mismatch", 1, "0/1 passed"),
                                          (check.Skip("SKIP (no Int64)"), 0, "0/1 passed, 1 skipped")):
            with self.subTest(detail=detail), tempfile.TemporaryDirectory() as tmp:
                args = SimpleNamespace(filter="cs_w32_salu_ext", build=tmp, llvm_mc="unused")

                def translate(_cmd, **_kwargs):
                    (Path(tmp) / "spirv" / "cs_w32_salu_ext.spv").touch()
                    return 0, ""

                output = io.StringIO()
                with (patch.object(rgt, "assemble_to_s", side_effect=lambda _mc, _src, dst, _name: Path(dst).touch()),
                      patch.object(rgt, "spirv_as"), patch.object(rgt, "run_cmd", side_effect=translate),
                      patch.object(rgt, "salu_ext_test", return_value=detail) as readback,
                      contextlib.redirect_stdout(output)):
                    self.assertEqual(rgt.run_suite(args, tmp), expected)
                readback.assert_called_once()
                self.assertIn(verdict, output.getvalue())
                self.assertIn("cs_w32_salu_ext", output.getvalue())


if __name__ == "__main__":
    unittest.main()
