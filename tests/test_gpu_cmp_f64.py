"""GPU-CMP-F64: the binary64 compare oracle, the readback and the runner verdicts, without needing a GPU."""
import contextlib
import io
import math
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
import run_gpu_tests as rgt
import check

WORDS = 1536        # 256 records of two result words, then 256 records of four input words
RESULTS = 512
FIXTURE = ROOT / "runtime" / "test" / "shaders" / "cs_w32_cmp_f64.asm"


def as_float(bits):
    return struct.unpack("<d", struct.pack("<Q", bits))[0]


class CmpF64Oracle(unittest.TestCase):
    def test_values_are_ascending_then_two_nans(self):
        values = [as_float(bits) for bits in check.CMP_F64_VALUES]
        self.assertEqual(len(set(check.CMP_F64_VALUES)), 16)
        self.assertEqual(values[:14], sorted(values[:14]))
        self.assertTrue(math.isnan(values[14]) and math.isnan(values[15]))
        # The first NaN differs from +inf in the low dword only, and the two zeros in the sign only.
        self.assertEqual(check.CMP_F64_VALUES[14] ^ check.CMP_F64_VALUES[13], 1)
        self.assertEqual(check.CMP_F64_VALUES[6] ^ check.CMP_F64_VALUES[7], check.CMP_F64_SIGN)

    def test_predicates_follow_the_trichotomy(self):
        """An independent restatement: an ordered pair sets exactly one of lt, eq and gt, an unordered pair none,
        and each predicate is the union its name says, negated for the n* forms."""
        unions = {"f": (), "lt": ("lt",), "eq": ("eq",), "le": ("lt", "eq"), "gt": ("gt",), "lg": ("lt", "gt"),
                  "ge": ("gt", "eq"), "o": ("lt", "eq", "gt"), "u": ("un",), "nge": ("lt", "un"),
                  "nlg": ("eq", "un"), "ngt": ("lt", "eq", "un"), "nle": ("gt", "un"), "neq": ("lt", "gt", "un"),
                  "nlt": ("eq", "gt", "un"), "tru": ("lt", "eq", "gt", "un")}
        self.assertEqual(tuple(unions), check.CMP_F64_PREDICATES)
        seen = set()
        for a_bits in check.CMP_F64_VALUES:
            for b_bits in check.CMP_F64_VALUES:
                a, b = as_float(a_bits), as_float(b_bits)
                relation = ("un" if math.isnan(a) or math.isnan(b) else
                            "lt" if a < b else "gt" if a > b else "eq")
                seen.add(relation)
                want = sum(1 << bit for bit, name in enumerate(check.CMP_F64_PREDICATES) if relation in unions[name])
                self.assertEqual(check.cmp_f64_predicates(a_bits, b_bits), want, (hex(a_bits), hex(b_bits)))
        self.assertEqual(seen, {"lt", "eq", "gt", "un"})

    def test_known_words(self):
        initial, expected = check.cmp_f64_words()
        self.assertEqual((len(initial), len(expected)), (WORDS, WORDS))
        self.assertEqual(initial[:RESULTS], [0xEEEEEEEE] * RESULTS)
        self.assertEqual(initial[RESULTS:], expected[RESULTS:])
        # Each record's v_cmp_ word and v_cmpx_ word are the same.
        self.assertEqual(expected[0:RESULTS:2], expected[1:RESULTS:2])
        self.assertTrue(all(word >> 19 == 0 for word in expected[:RESULTS]))
        # Record 0, -inf with -inf on an even thread: eq le ge o nlg ngt nlt tru, bit 16, and neither modifier bit.
        self.assertEqual(expected[0], 0x1CCCC)
        # Record 1, -inf with -max on an odd thread: lt le lg o nge ngt neq tru and no bit 16.
        self.assertEqual(expected[2], 0xAAAA)
        # Record 14, -inf with the low-dword NaN on an even thread: u, the six n* predicates and tru, bit 16, and
        # bit 18, since -(-inf) >= |NaN| is not true.
        self.assertEqual(expected[28], 0x5FF00)
        # Record 103, -0 with +0 (odd thread): equal. Record 120, +0 with the smallest subnormal: less.
        self.assertEqual(expected[2 * 103], 0xCCCC)
        self.assertEqual(expected[2 * 120] & 0xFFFF, 0xAAAA)
        # Record 35, -1 - 1 ulp with -1: less, by the low dword alone.
        self.assertEqual(expected[2 * 35] & 0xFFFF, 0xAAAA)
        # Record 163, 1 with -1 (odd thread): gt ge lg o nle neq nlt tru; |1| < 1 is false; !(-1 >= 1) is true.
        self.assertEqual(expected[2 * 163], 0x4F0F0)
        # The inputs: record 163 is {1.lo, 1.hi, -1.lo, -1.hi}.
        self.assertEqual(expected[RESULTS + 4 * 163:RESULTS + 4 * 164], [0, 0x3FF00000, 0, 0xBFF00000])

    def test_fixture_gives_each_predicate_the_oracle_s_bit(self):
        """In the assembly, each compare is followed by the OR of its result bit. For both records a thread handles,
        the v_cmp_ and the v_cmpx_ predicates come in opcode order with bit n for predicate n, and the two compares
        behind bits 16..18 are the ones the oracle models."""
        lines = [ln.strip() for ln in FIXTURE.read_text(encoding="utf-8").splitlines()]
        code = [ln for ln in lines if ln and not ln.startswith(";")]
        self.assertIn("; group: 128,1,1", lines)

        def bits(compare, accumulate):
            """(compare operands and predicate, bit) of every compare whose next instruction ORs a bit in."""
            found = []
            for ln, after in zip(code, code[1:]):
                m, bit = re.fullmatch(compare, ln), re.fullmatch(accumulate, after)
                if m and bit:
                    found.append((m.group(1), m.group(2), int(bit.group(1), 16)))
            return found

        plain = [("v[2:3], v[4:5]", name, 1 << bit) for bit, name in enumerate(check.CMP_F64_PREDICATES)]
        modified = [("|v[2:3]|, -v[4:5]", "lt", 1 << 17), ("-v[2:3], |v[4:5]|", "nge", 1 << 18)]
        # v_cmp_: the odd-thread tru is followed by the EXEC restore, not by its OR, so it is not in this list.
        got = bits(r"v_cmp_(\w+?)_f64_e(?:32|64) vcc_lo, (.+)", r"v_or_b32_e32 v8, (0x[0-9a-f]+), v6")
        self.assertEqual([(operands, name, bit) for name, operands, bit in got], (plain + modified) * 2)
        got = bits(r"v_cmpx_(\w+?)_f64_e(?:32|64) (.+)", r"v_or_b32_e32 v7, (0x[0-9a-f]+), v7")
        odd_off = [("v[2:3], v[4:5]", "tru", 1 << 16)]
        self.assertEqual([(operands, name, bit) for name, operands, bit in got], (plain + odd_off + modified) * 2)
        # Bit 16 of the v_cmp_ word: the odd threads go off, tru runs, EXEC comes back, then the bit is taken from VCC.
        at = [i for i, ln in enumerate(code) if ln == "v_cmpx_eq_u32_e32 0, v9"]
        self.assertEqual(len(at), 4)
        for i in at[0::2]:
            self.assertEqual(code[i + 1:i + 5], ["v_cmp_tru_f64_e32 vcc_lo, v[2:3], v[4:5]", "s_mov_b32 exec_lo, -1",
                                                 "v_or_b32_e32 v8, 0x10000, v6",
                                                 "v_cndmask_b32_e32 v6, v6, v8, vcc_lo"])
        for i in at[1::2]:
            self.assertEqual(code[i + 1], "v_cmpx_tru_f64_e32 v[2:3], v[4:5]")
        self.assertIn("v_and_b32_e32 v9, 1, v0", code)
        # The records are read from byte 0x800 on and the two words of a record go to byte 8 * record.
        self.assertEqual([ln for ln in code if ln.startswith("buffer_")],
                         ["buffer_load_dwordx4 v[2:5], v1, s[0:3], 0 offen offset:2048",
                          "buffer_store_dwordx2 v[6:7], v10, s[0:3], 0 offen",
                          "buffer_load_dwordx4 v[2:5], v11, s[0:3], 0 offen offset:2048",
                          "buffer_store_dwordx2 v[6:7], v10, s[0:3], 0 offen offset:1024"])
        self.assertEqual(code[:4], ["v_lshlrev_b32_e32 v1, 4, v0", "v_add_nc_u32_e32 v11, 0x800, v1",
                                    "v_lshlrev_b32_e32 v10, 3, v0", "v_and_b32_e32 v9, 1, v0"])
        self.assertEqual(code[-1], "s_endpgm")


class CmpF64Readback(unittest.TestCase):
    def run_readback(self, mutate=None, code=0, output="", missing=False):
        paths = []

        def launch(*args, env):
            paths.append(Path(env["VKC_HEAP_INIT"]))
            initial, expected = check.cmp_f64_words()
            self.assertEqual(args[1:], (str(Path("unused") / "cs_w32_cmp_f64.spv"), "1", "dump"))
            self.assertEqual(paths[-1].read_bytes(), struct.pack(f"<{WORDS}I", *initial))
            self.assertEqual(env["VKC_SUBGROUP_SIZE"], "32")
            self.assertEqual(env["VKC_OUT_WORDS"], str(WORDS))
            self.assertEqual(env["LEONIDA_VK_DEVICE"], "synthetic device")
            if mutate:
                mutate(expected)
            if not missing:
                Path(env["VKC_DUMP"]).write_bytes(struct.pack(f"<{len(expected)}I", *expected))
            return code, output

        with patch.object(check, "run", side_effect=launch):
            result = check.cmp_f64_test("unused", "unused", {"LEONIDA_VK_DEVICE": "synthetic device"})
        self.assertEqual(len(paths), 1)
        self.assertFalse(paths[0].exists(), "private heap input must be cleaned up")
        return result

    def test_complete_readback_passes(self):
        self.assertEqual(self.run_readback(), "")

    def test_one_wrong_result_bit_or_a_changed_input_fails(self):
        for word in (0, 1, RESULTS - 1, RESULTS, WORDS - 1):
            with self.subTest(word=word):
                result = self.run_readback(lambda data: data.__setitem__(word, data[word] ^ 0x8000))
                self.assertIn(f"word {word}:", result)

    def test_truncated_and_missing_dump_fail(self):
        self.assertIn(f"expected {4 * WORDS} dump bytes, got {4 * WORDS - 4}",
                      self.run_readback(lambda data: data.pop()))
        self.assertIn("no VKC_DUMP", self.run_readback(missing=True))

    def test_unsupported_module_skips_but_nonzero_exit_fails(self):
        self.assertIsInstance(self.run_readback(code=77, output="unsupported: no Int64\n"), check.Skip)
        self.assertIn("exit 1", self.run_readback(code=1, output="dispatch failed"))


class CmpF64Runner(unittest.TestCase):
    def test_filtered_runner_counts_pass_failure_and_skip(self):
        for detail, expected, verdict in (("", 0, "1/1 passed"), ("word 28: mismatch", 1, "0/1 passed"),
                                          (check.Skip("SKIP (no Int64)"), 0, "0/1 passed, 1 skipped")):
            with self.subTest(detail=detail), tempfile.TemporaryDirectory() as tmp:
                args = SimpleNamespace(filter="cs_w32_cmp_f64", build=tmp, llvm_mc="unused")

                def translate(_cmd, **_kwargs):
                    (Path(tmp) / "spirv" / "cs_w32_cmp_f64.spv").touch()
                    return 0, ""

                output = io.StringIO()
                with (patch.object(rgt, "assemble_to_s", side_effect=lambda _mc, _src, dst, _name: Path(dst).touch()),
                      patch.object(rgt, "spirv_as"), patch.object(rgt, "run_cmd", side_effect=translate),
                      patch.object(rgt, "cmp_f64_test", return_value=detail) as readback,
                      contextlib.redirect_stdout(output)):
                    self.assertEqual(rgt.run_suite(args, tmp), expected)
                readback.assert_called_once()
                self.assertIn(verdict, output.getvalue())
                self.assertIn("cs_w32_cmp_f64", output.getvalue())


if __name__ == "__main__":
    unittest.main()
