"""GPU-ROUND-F64: the binary64 rounding oracle, its judge, the readback and the runner verdicts, without a GPU."""
import contextlib
from fractions import Fraction
import io
import math
from pathlib import Path
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

WORDS = 1152        # 64 threads of eight two-word results, then 64 two-word inputs
RESULTS = 1024
FIXTURE = ROOT / "runtime" / "test" / "shaders" / "cs_w32_round_f64.asm"
OPS = ("trunc", "ceil", "rndne", "floor")


def bits_of(value):
    return struct.unpack("<Q", struct.pack("<d", value))[0]


def as_float(bits):
    return struct.unpack("<d", struct.pack("<Q", bits))[0]


def word(thread, slot):
    """The index of the low word of a thread's result."""
    return 16 * thread + 2 * slot


class RoundF64Oracle(unittest.TestCase):
    def test_inputs_cover_the_classes_the_fixture_names(self):
        values = check.ROUND_F64_VALUES
        self.assertEqual((len(values), len(set(values))), (64, 64))
        floats = [as_float(bits) for bits in values]
        self.assertEqual(sum(math.isnan(x) for x in floats), 5)
        self.assertEqual(sum(0 < abs(x) < 2.0 ** -1022 for x in floats), 2)
        self.assertEqual(sum(math.isinf(x) for x in floats), 2)
        for x in (0.0, -0.0, 0.5, -0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 0.49999999999999994, 2.0 ** 31 + 0.5,
                  2.0 ** 51 + 0.5, 2.0 ** 52 - 0.5, 2.0 ** 52, 2.0 ** 52 + 1, 2.0 ** 53 - 1, 2.0 ** 63, 1e300):
            self.assertIn(bits_of(x), values, x)

    def test_oracle_agrees_with_exact_rational_rounding(self):
        """An independent restatement: round the exact rational value of every finite input, then give the result
        the source's sign. Python rounds a Fraction's tie to the even integer."""
        exact = {"trunc": math.trunc, "ceil": math.ceil, "rndne": round, "floor": math.floor}
        for bits in check.ROUND_F64_VALUES:
            x = as_float(bits)
            for op in OPS:
                if math.isnan(x) or math.isinf(x):
                    want = bits
                else:
                    whole = exact[op](Fraction(x))
                    want = bits_of(float(whole)) | bits & check.ROUND_F64_SIGN
                    self.assertEqual(Fraction(as_float(want)), whole)       # the integer is representable
                self.assertEqual(check.round_f64(op, bits), want, (op, hex(bits)))

    def test_values_the_contract_row_names(self):
        sub, one = 1, bits_of(1.0)
        sign = check.ROUND_F64_SIGN
        expected = {("ceil", sub): one, ("ceil", sub | sign): sign, ("floor", sub): 0, ("floor", sub | sign): one | sign,
                    ("trunc", sub): 0, ("trunc", sub | sign): sign,
                    ("rndne", bits_of(0.5)): 0, ("rndne", bits_of(-0.5)): sign,
                    ("rndne", bits_of(1.5)): bits_of(2.0), ("rndne", bits_of(-1.5)): bits_of(-2.0),
                    ("rndne", bits_of(2.5)): bits_of(2.0), ("rndne", bits_of(-2.5)): bits_of(-2.0),
                    ("rndne", bits_of(0.49999999999999994)): 0, ("ceil", bits_of(-0.25)): sign,
                    ("floor", bits_of(0.25)): 0, ("rndne", bits_of(2.0 ** 51 + 0.5)): bits_of(2.0 ** 51),
                    ("rndne", bits_of(2.0 ** 52 - 0.5)): bits_of(2.0 ** 52),
                    ("floor", bits_of(2.0 ** 52 - 0.5)): bits_of(2.0 ** 52 - 1)}
        for (op, source), want in expected.items():
            self.assertEqual(check.round_f64(op, source), want, (op, hex(source)))
        for unchanged in (bits_of(2.0 ** 52 + 1), bits_of(2.0 ** 63), bits_of(1e300), bits_of(math.inf),
                          bits_of(-math.inf), 0x7FF0000000000001, 0xFFF0000080000000, 0x7FF8DEADBEEF1234):
            for op in OPS:
                self.assertEqual(check.round_f64(op, unchanged), unchanged, (op, hex(unchanged)))

    def test_layout(self):
        initial, expected = check.round_f64_words()
        self.assertEqual((len(initial), len(expected)), (WORDS, WORDS))
        self.assertEqual(initial[:RESULTS], [0xEEEEEEEE] * RESULTS)
        self.assertEqual(initial[RESULTS:], expected[RESULTS:])
        # Thread 9 holds -0.5: its eight results, then (an odd thread) the pair its inactive lane keeps.
        self.assertEqual(expected[RESULTS + 18:RESULTS + 20], [0, 0xBFE00000])
        minus_zero, plus_one, minus_one = [0, 0x80000000], [0, 0x3FF00000], [0, 0xBFF00000]
        self.assertEqual(expected[word(9, 0):word(10, 0)],
                         minus_zero + minus_zero + minus_zero + minus_one       # trunc, ceil, rndne, floor of -0.5
                         + [0, 0] + plus_one + minus_one                        # trunc of 0.5, ceil of 0.5, floor
                         + [0xDEADBEEF, 0xCAFEF00D])
        # Thread 8 holds 0.5 and is even: its last result is rndne, +0.
        self.assertEqual(expected[word(8, 7):word(9, 0)], [0, 0])


LLVMPIPE = "device: llvmpipe (LLVM 20.1.2, 256 bits) (subgroup 8..8)\n"
NVIDIA = "device: NVIDIA GeForce RTX 4070 Ti (subgroup 32..32)\n"


class RoundF64Judge(unittest.TestCase):
    def judge(self, change, out=""):
        _, expected = check.round_f64_words()
        change(expected)
        return check.round_f64_problem(tuple(expected), out)

    def set_result(self, thread, slot, bits):
        def change(data):
            data[word(thread, slot)], data[word(thread, slot) + 1] = bits & 0xFFFFFFFF, bits >> 32
        return change

    def test_the_hardware_s_words_pass(self):
        self.assertEqual(self.judge(lambda data: None), "")

    def test_a_nan_comes_back_with_every_bit_it_had(self):
        signalling = check.ROUND_F64_VALUES.index(0x7FF0000000000001)
        self.assertIn("trunc of 0x7ff0000000000001 gave 0x7ff8000000000001",              # quieted by a driver
                      self.judge(self.set_result(signalling, 0, 0x7FF8000000000001)))
        self.assertIn("trunc of 0xfff0000000000001", self.judge(self.set_result(signalling, 4, 0x7FF0000000000001)))
        payload = check.ROUND_F64_VALUES.index(0x7FF8DEADBEEF1234)
        self.assertIn("floor of 0x7ff8deadbeef1234", self.judge(self.set_result(payload, 3, 0x7FF8000000000000)))
        # A number does not come back as a NaN.
        self.assertIn("ceil of", self.judge(self.set_result(16, 1, 0x7FF8000000000000)))

    def test_a_subnormal_is_not_read_as_zero(self):
        sign = check.ROUND_F64_SIGN
        self.assertIn("ceil of 0x0000000000000001 gave 0x0000000000000000, expected 0x3ff0000000000000",
                      self.judge(self.set_result(2, 1, 0)))                  # a driver that flushes it
        self.assertIn("floor of 0x8000000000000001 gave 0x8000000000000000, expected 0xbff0000000000000",
                      self.judge(self.set_result(3, 3, sign)))
        self.assertIn("trunc of 0x0000000000000001", self.judge(self.set_result(2, 0, bits_of(1.0))))

    def test_everything_else_is_exact(self):
        tie = check.ROUND_F64_VALUES.index(bits_of(2.5))
        for thread, slot, wrong, what in ((tie, 2, bits_of(3.0), "rndne of 0x4004000000000000"),
                                          (1, 2, 0, "rndne of 0x8000000000000000"),      # -0 came back as +0
                                          (5, 1, 0, "ceil of 0xbfd0000000000000"),       # ceil of -0.25 is -0
                                          (16, 5, bits_of(2.0), "ceil of 0x3ff0000000000000"),
                                          (9, 7, 0, "the pair an inactive lane keeps")):
            with self.subTest(what=what):
                result = self.judge(self.set_result(thread, slot, wrong))
                self.assertIn(f"word {word(thread, slot)} (thread {thread}): {what}", result)
        self.assertIn(f"word {RESULTS}: an input changed", self.judge(lambda data: data.__setitem__(RESULTS, 1)))
        self.assertIn(f"word {WORDS - 1}: an input changed",
                      self.judge(lambda data: data.__setitem__(WORDS - 1, 0)))

    def test_llvmpipe_s_float_path_passes_for_trunc_and_rndne_and_on_no_other_device(self):
        index = check.ROUND_F64_VALUES.index
        above_half, big = index(0x3FE0000000000001), index(bits_of(123456789.0625))
        half_tie = index(0x4320000000000001)                      # 2^51 + 0.5
        # What llvmpipe gave on a CPU without AVX-512: (thread, slot, the word it returned, what the judge calls it).
        measured = ((above_half, 2, bits_of(0.0), "rndne of 0x3fe0000000000001"),
                    (index(bits_of(1.5)), 2, bits_of(1.0), "rndne of 0x3ff8000000000000"),
                    (index(0xC004000000000001), 2, bits_of(-2.0), "rndne of 0xc004000000000001"),
                    (index(bits_of(5.5)), 7, bits_of(5.0), "rndne of 0x4016000000000000"),
                    (big, 0, bits_of(123456789.0625), "trunc of 0x419d6f3454400000"),
                    (big, 2, bits_of(123456789.0625), "rndne of 0x419d6f3454400000"),
                    (half_tie, 4, 0xC320000000000001, "trunc of 0xc320000000000001"))   # the neg source
        for thread, slot, bits, what in measured:
            with self.subTest(what=what):
                change = self.set_result(thread, slot, bits)
                self.assertEqual(self.judge(change, LLVMPIPE), "")
                for out in ("", NVIDIA):
                    self.assertIn(f"word {word(thread, slot)} (thread {thread}): {what}", self.judge(change, out))

    def test_llvmpipe_s_float_path_is_no_wider_than_measured(self):
        index = check.ROUND_F64_VALUES.index
        big, tie = index(bits_of(123456789.0625)), index(bits_of(2.5))
        for thread, slot, wrong, what in (
                (tie, 2, bits_of(4.0), "rndne of"),                        # not a neighbouring integer
                (tie, 2, bits_of(-2.0), "rndne of"),                       # the sign of the source is lost
                (tie, 0, bits_of(2.5), "trunc of"),                        # unchanged, but not above 2^24
                (big, 0, bits_of(123456790.0), "trunc of"),                # above 2^24 only the source itself
                (index(bits_of(1.5)), 1, bits_of(1.0), "ceil of"),         # Ceil is exact on that path
                (big, 3, bits_of(123456789.0625), "floor of"),             # and so is Floor
                (index(0x7FF0000000000000), 2, bits_of(1.0), "rndne of"),  # an infinity stays
                (2, 2, bits_of(1.0), "rndne of"),                          # the smallest subnormal: below 0.5
                (index(bits_of(0.25)), 0, bits_of(1.0), "trunc of"),       # and so is 0.25
                (9, 7, 0, "the pair an inactive lane keeps")):
            with self.subTest(thread=thread, slot=slot, wrong=hex(wrong)):
                result = self.judge(self.set_result(thread, slot, wrong), LLVMPIPE)
                self.assertIn(f"word {word(thread, slot)} (thread {thread}): {what}", result)

    def test_the_float_path_allowance_covers_the_48_results_measured(self):
        """The readback llvmpipe gives without AVX-512, as measured: RoundEven truncates where the binary32 of its
        source is a tie or the source is just past one, and Trunc and RoundEven return a magnitude above 2^24
        unchanged. Rebuilt here from those two rules for the inputs they change."""
        _, data = check.round_f64_words()
        changed = 0
        truncating = {0x3FE0000000000001, 0xBFE0000000000001, bits_of(1.5), bits_of(-1.5), bits_of(3.5),
                      bits_of(-3.5), 0xC004000000000001, bits_of(5.5), bits_of(-5.5)}
        for thread, results in enumerate(check.round_f64_results()):
            for slot, (op, source) in enumerate(results):
                value = None
                if op in ("trunc", "rndne") and math.isfinite(as_float(source)) and abs(as_float(source)) > 1 << 24:
                    value = source
                elif op == "rndne" and source in truncating:
                    value = check.round_f64("trunc", source)
                if value is not None and value != check.round_f64(op, source):
                    data[word(thread, slot)], data[word(thread, slot) + 1] = value & 0xFFFFFFFF, value >> 32
                    changed += 1
        self.assertEqual(changed, 48)
        self.assertEqual(check.round_f64_problem(tuple(data), LLVMPIPE), "")
        self.assertIn("word 164 (thread 10): rndne of 0x3fe0000000000001 gave 0x0000000000000000",
                      check.round_f64_problem(tuple(data)))


class RoundF64Fixture(unittest.TestCase):
    def test_assembly_stores_the_eight_results_the_oracle_lists(self):
        lines = [ln.strip() for ln in FIXTURE.read_text(encoding="utf-8").splitlines()]
        code = [ln for ln in lines if ln and not ln.startswith(";")]
        self.assertIn("; group: 64,1,1", lines)
        self.assertEqual(code, [
            "v_lshlrev_b32_e32 v1, 3, v0", "v_add_nc_u32_e32 v1, 0x1000, v1", "v_lshlrev_b32_e32 v10, 6, v0",
            "v_and_b32_e32 v9, 1, v0",
            "buffer_load_dwordx2 v[2:3], v1, s[0:3], 0 offen", "s_waitcnt vmcnt(0)",
            "v_trunc_f64_e32 v[4:5], v[2:3]", "v_ceil_f64_e32 v[6:7], v[2:3]",
            "v_rndne_f64_e32 v[12:13], v[2:3]", "v_floor_f64_e32 v[14:15], v[2:3]",
            "v_trunc_f64_e64 v[16:17], -v[2:3]", "v_ceil_f64_e64 v[18:19], |v[2:3]|",
            "v_mov_b32_e32 v20, v2", "v_mov_b32_e32 v21, v3", "v_floor_f64_e32 v[21:22], v[20:21]",
            "v_mov_b32_e32 v24, 0xdeadbeef", "v_mov_b32_e32 v25, 0xcafef00d",
            "v_cmpx_eq_u32_e32 0, v9", "v_rndne_f64_e32 v[24:25], v[2:3]", "s_mov_b32 exec_lo, -1",
            "buffer_store_dwordx2 v[4:5], v10, s[0:3], 0 offen",
            "buffer_store_dwordx2 v[6:7], v10, s[0:3], 0 offen offset:8",
            "buffer_store_dwordx2 v[12:13], v10, s[0:3], 0 offen offset:16",
            "buffer_store_dwordx2 v[14:15], v10, s[0:3], 0 offen offset:24",
            "buffer_store_dwordx2 v[16:17], v10, s[0:3], 0 offen offset:32",
            "buffer_store_dwordx2 v[18:19], v10, s[0:3], 0 offen offset:40",
            "buffer_store_dwordx2 v[21:22], v10, s[0:3], 0 offen offset:48",
            "buffer_store_dwordx2 v[24:25], v10, s[0:3], 0 offen offset:56",
            "s_endpgm"])
        # The order of the stores is the order of round_f64_results, and the pair an odd thread keeps is the one
        # the two moves put there.
        self.assertEqual([op for op, _ in check.round_f64_results()[0]],
                         ["trunc", "ceil", "rndne", "floor", "trunc", "ceil", "floor", "rndne"])
        self.assertEqual(check.round_f64_results()[1][7], (None, 0xCAFEF00DDEADBEEF))
        x = check.ROUND_F64_VALUES[5]
        self.assertEqual(check.round_f64_results()[5][4:6],
                         [("trunc", x ^ check.ROUND_F64_SIGN), ("ceil", x & ~check.ROUND_F64_SIGN)])


class RoundF64Readback(unittest.TestCase):
    def run_readback(self, mutate=None, code=0, output="", missing=False):
        paths = []

        def launch(*args, env):
            paths.append(Path(env["VKC_HEAP_INIT"]))
            initial, expected = check.round_f64_words()
            self.assertEqual(args[1:], (str(Path("unused") / "cs_w32_round_f64.spv"), "1", "dump"))
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
            result = check.round_f64_test("unused", "unused", {"LEONIDA_VK_DEVICE": "synthetic device"})
        self.assertEqual(len(paths), 1)
        self.assertFalse(paths[0].exists(), "private heap input must be cleaned up")
        return result

    def test_complete_readback_passes(self):
        self.assertEqual(self.run_readback(), "")

    def test_a_wrong_result_or_a_changed_input_fails(self):
        for index in (0, RESULTS - 1, RESULTS, WORDS - 1):
            with self.subTest(word=index):
                result = self.run_readback(lambda data: data.__setitem__(index, data[index] ^ 0x00100000))
                self.assertIn(f"word {index - index % 2 if index < RESULTS else index}", result)

    def test_truncated_and_missing_dump_fail(self):
        self.assertIn(f"expected {4 * WORDS} dump bytes, got {4 * WORDS - 4}",
                      self.run_readback(lambda data: data.pop()))
        self.assertIn("no VKC_DUMP", self.run_readback(missing=True))

    def test_the_device_line_reaches_the_judge(self):
        thread = check.ROUND_F64_VALUES.index(bits_of(1.5))

        def truncated_tie(data):
            data[word(thread, 2)], data[word(thread, 2) + 1] = 0, bits_of(1.0) >> 32

        self.assertEqual(self.run_readback(truncated_tie, output=LLVMPIPE), "")
        self.assertIn("rndne of 0x3ff8000000000000", self.run_readback(truncated_tie, output=NVIDIA))

    def test_unsupported_float64_skips_but_nonzero_exit_fails(self):
        self.assertIsInstance(self.run_readback(code=77, output="unsupported: no Float64\n"), check.Skip)
        self.assertIn("exit 1", self.run_readback(code=1, output="dispatch failed"))


class RoundF64Runner(unittest.TestCase):
    def test_filtered_runner_counts_pass_failure_and_skip(self):
        for detail, expected, verdict in (("", 0, "1/1 passed"), ("word 132: mismatch", 1, "0/1 passed"),
                                          (check.Skip("SKIP (no Float64)"), 0, "0/1 passed, 1 skipped")):
            with self.subTest(detail=detail), tempfile.TemporaryDirectory() as tmp:
                args = SimpleNamespace(filter="cs_w32_round_f64", build=tmp, llvm_mc="unused")

                def translate(_cmd, **_kwargs):
                    (Path(tmp) / "spirv" / "cs_w32_round_f64.spv").touch()
                    return 0, ""

                output = io.StringIO()
                with (patch.object(rgt, "assemble_to_s", side_effect=lambda _mc, _src, dst, _name: Path(dst).touch()),
                      patch.object(rgt, "spirv_as"), patch.object(rgt, "run_cmd", side_effect=translate),
                      patch.object(rgt, "round_f64_test", return_value=detail) as readback,
                      contextlib.redirect_stdout(output)):
                    self.assertEqual(rgt.run_suite(args, tmp), expected)
                readback.assert_called_once()
                self.assertIn(verdict, output.getvalue())
                self.assertIn("cs_w32_round_f64", output.getvalue())


if __name__ == "__main__":
    unittest.main()
