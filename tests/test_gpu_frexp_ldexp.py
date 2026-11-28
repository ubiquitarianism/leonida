"""GPU-FREXP-LDEXP: the frexp and ldexp oracle, the readback and the runner verdicts, without needing a GPU."""
import contextlib
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

WORDS = 768         # 64 threads of eight result words, then 64 records of four input words
RESULTS = 512
FIXTURE = ROOT / "runtime" / "test" / "shaders" / "cs_w32_frexp_ldexp.asm"


def f32(bits):
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def f32_bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def f64(bits):
    return struct.unpack("<d", struct.pack("<Q", bits))[0]


def f64_bits(value):
    return struct.unpack("<Q", struct.pack("<d", value))[0]


def signed(word):
    return word - (1 << 32) if word & 0x80000000 else word


def kind32(bits):
    exponent = bits >> 23 & 0xFF
    if exponent == 0:
        return "denormal" if bits & 0x007FFFFF else "zero"
    if exponent == 255:
        return "nan" if bits & 0x007FFFFF else "inf"
    return "normal"


class FrexpLdexpOracle(unittest.TestCase):
    def test_records_cover_the_classes_the_fixture_names(self):
        self.assertEqual((len(check.FREXP_LDEXP_F32), len(set(check.FREXP_LDEXP_F32))), (64, 64))
        self.assertEqual((len(check.FREXP_F64), len(set(check.FREXP_F64))), (64, 64))
        kinds = [kind32(x) for x, _ in check.FREXP_LDEXP_F32]
        self.assertEqual({k: kinds.count(k) for k in sorted(set(kinds))},
                         {"denormal": 4, "inf": 4, "nan": 4, "normal": 48, "zero": 4})
        exponents = {signed(n) for _, n in check.FREXP_LDEXP_F32}
        self.assertTrue({512, -512, 513, -513, 0x7FFFFFFF, -0x80000000, 0, 1, -1} <= exponents)
        doubles = check.FREXP_F64
        subnormal = [d for d in doubles if d >> 52 & 0x7FF == 0 and d & ((1 << 52) - 1)]
        self.assertEqual(len(subnormal), 24)
        # Single fraction bits on both sides of the dword boundary, and patterns with bits in both dwords.
        self.assertTrue({1 << 31, 1 << 32, 1, 1 << 51} <= set(subnormal))
        self.assertTrue(any(d & 0xFFFFFFFF and d >> 32 & 0xFFFFF for d in subnormal))
        self.assertEqual(sum(math.isnan(f64(d)) for d in doubles), 4)
        self.assertEqual(sum(math.isinf(f64(d)) for d in doubles), 2)

    def test_binary32_normals_agree_with_python(self):
        """math.frexp and math.ldexp work on the double value of a binary32 normal, exactly. They agree with the
        hardware wherever its result is a normal; outside that the hardware saturates to an infinity or a zero."""
        compared = 0
        for x, n in check.FREXP_LDEXP_F32:
            if kind32(x) != "normal":
                continue
            mantissa, exponent = math.frexp(f32(x))
            self.assertEqual(check.frexp_mant_f32(x), f32_bits(mantissa), hex(x))
            self.assertEqual(signed(check.frexp_exp_i32_f32(x)), exponent, hex(x))
            field = (x >> 23 & 0xFF) + max(-512, min(512, signed(n)))
            got = check.ldexp_f32(x, n)
            if field > 254:
                self.assertEqual(got, x & 0x80000000 | 0x7F800000, (hex(x), n))
            elif field <= 0:
                self.assertEqual(got, x & 0x80000000, (hex(x), n))
            else:
                self.assertEqual(got, f32_bits(math.ldexp(f32(x), signed(n))), (hex(x), n))
                compared += 1
        self.assertGreaterEqual(compared, 16)

    def test_binary64_finite_values_agree_with_python(self):
        """math.frexp normalizes a subnormal as the hardware does, so every finite non-zero pattern is compared."""
        compared = 0
        for d in check.FREXP_F64:
            value = f64(d)
            if value == 0 or math.isinf(value) or math.isnan(value):
                continue
            mantissa, exponent = math.frexp(value)
            self.assertEqual(check.frexp_f64(d), (f64_bits(mantissa), exponent & 0xFFFFFFFF), hex(d))
            self.assertEqual(math.ldexp(f64(check.frexp_f64(d)[0]), signed(check.frexp_f64(d)[1])), value)
            compared += 1
        self.assertEqual(compared, 56)

    def test_values_the_contract_rows_name(self):
        # Zero and denormals give a signed zero and exponent 0; infinities and NaNs the input and exponent 0.
        for x in (0x00000000, 0x80000000, 0x00000001, 0x807FFFFF):
            self.assertEqual((check.frexp_mant_f32(x), check.frexp_exp_i32_f32(x)), (x & 0x80000000, 0))
            self.assertEqual(check.ldexp_f32(x, 300), x & 0x80000000)
        for x in (0x7F800000, 0xFF800000, 0x7FC00000, 0x7F800001, 0xFFC12345):
            self.assertEqual((check.frexp_mant_f32(x), check.frexp_exp_i32_f32(x)), (x, 0))
            self.assertEqual(check.ldexp_f32(x, 0xFFFFFFFB), x)
        one, two = f32_bits(1.0), f32_bits(2.0)
        self.assertEqual((check.frexp_mant_f32(one), check.frexp_exp_i32_f32(one)), (f32_bits(0.5), 1))
        self.assertEqual(check.ldexp_f32(one, 127), f32_bits(2.0 ** 127))
        self.assertEqual(check.ldexp_f32(one, 128), 0x7F800000)                 # above the exponent field 254
        self.assertEqual(check.ldexp_f32(one, 0xFFFFFF82), 0x00800000)          # 2^-126, the smallest normal
        self.assertEqual(check.ldexp_f32(one, 0xFFFFFF81), 0)                   # 2^-127: no denormal result
        self.assertEqual(check.ldexp_f32(two, 513), check.ldexp_f32(two, 512))  # the clamp
        self.assertEqual(check.ldexp_f32(0x00800000, 0x7FFFFFFF), 0x7F800000)
        self.assertEqual(check.ldexp_f32(0x7F7FFFFF, 0x80000000), 0)
        self.assertEqual(check.ldexp_f32(0x7F7FFFFF, 0xFFFFFF03), 0x00FFFFFF)   # field 254 - 253 = 1
        # The two subnormals the binary64 row gives.
        self.assertEqual(check.frexp_f64(0x0000000000000001), (0x3FE0000000000000, (-1073) & 0xFFFFFFFF))
        self.assertEqual(check.frexp_f64(0x000FFFFFFFFFFFFF), (0x3FEFFFFFFFFFFFFE, (-1022) & 0xFFFFFFFF))
        for d in (0x0000000000000000, 0x8000000000000000, 0x7FF0000000000000, 0xFFF0000000000000,
                  0x7FF0000000000001, 0xFFF8000012345678):
            self.assertEqual(check.frexp_f64(d), (d, 0))

    def test_layout(self):
        initial, expected = check.frexp_ldexp_words()
        self.assertEqual((len(initial), len(expected)), (WORDS, WORDS))
        self.assertEqual(initial[:RESULTS], [0xEEEEEEEE] * RESULTS)
        self.assertEqual(initial[RESULTS:], expected[RESULTS:])
        # Thread 0: x = 1, n = 0, d = 1. Mantissa 0.5, exponent 1, ldexp 1, mantissa of -1, the binary64 pair, its
        # exponent, and (an even thread) ldexp of |1|.
        self.assertEqual(expected[:8], [0x3F000000, 1, 0x3F800000, 0xBF000000, 0, 0x3FE00000, 1, 0x3F800000])
        self.assertEqual(expected[RESULTS:RESULTS + 4], [0x3F800000, 0, 0, 0x3FF00000])
        # Thread 53 holds the signalling NaN 0x7f800001 and is odd: mantissa unchanged, exponent 0, ldexp unchanged,
        # the mantissa of its negation, and the word its inactive lane keeps.
        self.assertEqual(check.FREXP_LDEXP_F32[53], (0x7F800001, 0xFFFFFFFD))
        self.assertEqual(expected[8 * 53:8 * 53 + 4], [0x7F800001, 0, 0x7F800001, 0xFF800001])
        self.assertEqual(expected[8 * 53 + 7], 0xDEADBEEF)
        # Thread 54 is even: ldexp of |0xffc12345| is the NaN without its sign.
        self.assertEqual(expected[8 * 54 + 7], 0x7FC12345)


class FrexpLdexpFixture(unittest.TestCase):
    def test_assembly_stores_the_eight_words_the_oracle_lists(self):
        lines = [ln.strip() for ln in FIXTURE.read_text(encoding="utf-8").splitlines()]
        code = [ln for ln in lines if ln and not ln.startswith(";")]
        self.assertIn("; group: 64,1,1", lines)
        self.assertEqual(code, [
            "v_lshlrev_b32_e32 v1, 4, v0", "v_lshlrev_b32_e32 v10, 5, v0", "v_and_b32_e32 v9, 1, v0",
            "buffer_load_dwordx4 v[2:5], v1, s[0:3], 0 offen offset:2048", "s_waitcnt vmcnt(0)",
            "v_frexp_mant_f32_e32 v12, v2", "v_frexp_exp_i32_f32_e32 v13, v2", "v_ldexp_f32 v14, v2, v3",
            "v_frexp_mant_f32_e64 v15, -v2",
            "v_frexp_mant_f64_e32 v[16:17], v[4:5]", "v_frexp_exp_i32_f64_e32 v18, v[4:5]",
            "v_mov_b32_e32 v19, 0xdeadbeef", "v_cmpx_eq_u32_e32 0, v9", "v_ldexp_f32 v19, |v2|, v3",
            "s_mov_b32 exec_lo, -1",
            "buffer_store_dwordx4 v[12:15], v10, s[0:3], 0 offen",
            "buffer_store_dwordx4 v[16:19], v10, s[0:3], 0 offen offset:16",
            "s_endpgm"])
        self.assertEqual(check.FREXP_LDEXP_KEPT, 0xDEADBEEF)


class FrexpLdexpReadback(unittest.TestCase):
    def run_readback(self, mutate=None, code=0, output="", missing=False):
        paths = []

        def launch(*args, env):
            paths.append(Path(env["VKC_HEAP_INIT"]))
            initial, expected = check.frexp_ldexp_words()
            self.assertEqual(args[1:], (str(Path("unused") / "cs_w32_frexp_ldexp.spv"), "1", "dump"))
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
            result = check.frexp_ldexp_test("unused", "unused", {"LEONIDA_VK_DEVICE": "synthetic device"})
        self.assertEqual(len(paths), 1)
        self.assertFalse(paths[0].exists(), "private heap input must be cleaned up")
        return result

    def test_complete_readback_passes(self):
        self.assertEqual(self.run_readback(), "")

    def test_one_wrong_word_or_a_changed_input_fails(self):
        # Word 419 is the one an RTX 4070 Ti got wrong before the sign-bit modifiers: thread 52, the mantissa of a
        # negated NaN.
        for index in (0, 419, RESULTS - 1, RESULTS, WORDS - 1):
            with self.subTest(word=index):
                result = self.run_readback(lambda data: data.__setitem__(index, data[index] ^ 0x00400000))
                self.assertIn(f"word {index}:", result)
        self.assertIn("word 419: expected 0xffc00000, got 0x7fffffff",
                      self.run_readback(lambda data: data.__setitem__(419, 0x7FFFFFFF)))

    def test_truncated_and_missing_dump_fail(self):
        self.assertIn(f"expected {4 * WORDS} dump bytes, got {4 * WORDS - 4}",
                      self.run_readback(lambda data: data.pop()))
        self.assertIn("no VKC_DUMP", self.run_readback(missing=True))

    def test_unsupported_module_skips_but_nonzero_exit_fails(self):
        self.assertIsInstance(self.run_readback(code=77, output="unsupported: no Int64\n"), check.Skip)
        self.assertIn("exit 1", self.run_readback(code=1, output="dispatch failed"))


class FrexpLdexpRunner(unittest.TestCase):
    def test_filtered_runner_counts_pass_failure_and_skip(self):
        for detail, expected, verdict in (("", 0, "1/1 passed"), ("word 419: mismatch", 1, "0/1 passed"),
                                          (check.Skip("SKIP (no Int64)"), 0, "0/1 passed, 1 skipped")):
            with self.subTest(detail=detail), tempfile.TemporaryDirectory() as tmp:
                args = SimpleNamespace(filter="cs_w32_frexp_ldexp", build=tmp, llvm_mc="unused")

                def translate(_cmd, **_kwargs):
                    (Path(tmp) / "spirv" / "cs_w32_frexp_ldexp.spv").touch()
                    return 0, ""

                output = io.StringIO()
                with (patch.object(rgt, "assemble_to_s", side_effect=lambda _mc, _src, dst, _name: Path(dst).touch()),
                      patch.object(rgt, "spirv_as"), patch.object(rgt, "run_cmd", side_effect=translate),
                      patch.object(rgt, "frexp_ldexp_test", return_value=detail) as readback,
                      contextlib.redirect_stdout(output)):
                    self.assertEqual(rgt.run_suite(args, tmp), expected)
                readback.assert_called_once()
                self.assertIn(verdict, output.getvalue())
                self.assertIn("cs_w32_frexp_ldexp", output.getvalue())


if __name__ == "__main__":
    unittest.main()
