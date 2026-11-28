"""A buffer offset that does not fit 32 bits is out of range in every OOB_SELECT mode.

The hardware adds index * stride (a full product) and soffset to the 48-bit V# base without wrapping at 32 bits, so
an index of 0xFFFFFFFF with a 20-byte stride lands far outside the buffer and the access has no effect. A translator
that keeps the offset in 32 bits turns the same access into base - 20, which is inside a view that covers a whole
pool: OOB_SELECT 3 does not check the index and 2 checks nothing. (Rule and hardware observation: five-star-recomp,
commit b9b57e2. The expected memory below is written out by hand.)

    python -I tests/semantic/test_buffer_offset_wrap.py -v
"""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

sys.dont_write_bytecode = True
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from spirv_mini_interp import run_workgroup
from test_loop_ballot_barrier import _translate_cs

BASE, OUTPUT = 0x4000, 0x8000
STRIDE, BYTES = 20, 4096
LANES = 32
RAW, DISABLED, STRUCTURED_OFFSET = 0x31016FAC, 0x21016FAC, 0x01016FAC
SENTINEL, VALUE = 0xA5A5A5A5, 77


def indexed(index, literal_word3=None):
    """Store VALUE at `index` of the view in s[0:3], load it back and write what was read to the output view."""
    lines = []
    view = "s[0:3]"
    if literal_word3 is not None:
        # The same descriptor from immediates, which the translator folds: the mode is known when translating.
        lines += [f"s_mov_b32 s8, {BASE:#x}", f"s_mov_b32 s9, {STRIDE << 16:#x}", f"s_mov_b32 s10, {BYTES:#x}",
                  f"s_mov_b32 s11, {literal_word3:#x}"]
        view = "s[8:11]"
    return lines + [
        f"v_mov_b32_e32 v1, {index}",
        f"v_mov_b32_e32 v2, {VALUE}",
        f"buffer_store_dword v2, v1, {view}, 0 idxen",
        f"buffer_load_dword v3, v1, {view}, 0 idxen",
        "v_lshlrev_b32_e32 v4, 2, v0",
        "buffer_store_dword v3, v4, s[4:7], 0 offen",
        "s_endpgm"]


SOFFSET = [
    # off = 0xFFFFFFF0 (32-bit, as the hardware forms it), then soffset 0x20 is added in 64 bits: 0x1_0000_0010.
    "v_mov_b32_e32 v1, -16",
    f"v_mov_b32_e32 v2, {VALUE}",
    "s_mov_b32 s12, 32",
    "buffer_store_dword v2, v1, s[0:3], s12 offen",
    "buffer_load_dword v3, v1, s[0:3], s12 offen",
    "v_lshlrev_b32_e32 v4, 2, v0",
    "buffer_store_dword v3, v4, s[4:7], 0 offen",
    "s_endpgm"]


class BufferOffsetWrapTests(unittest.TestCase):
    def run_case(self, lines, word3, stride=STRIDE, records=BYTES):
        module, _ = _translate_cs("cs_buffer_offset_wrap", lines, threads=LANES)
        memory = {address: SENTINEL for address in range(BASE - 64, BASE + 128, 4)}
        before = dict(memory)
        result = run_workgroup(
            module, LANES, LANES,
            inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i, "%local_index": i},
            push_constants=[BASE, stride << 16, records, word3, OUTPUT, 0, LANES * 4, RAW] + [0] * 24,
            memory=memory)
        view = {address: result[address] for address in before}
        return view, before, [result[OUTPUT + 4 * lane] for lane in range(LANES)]

    def test_index_inside_the_buffer_is_stored_and_read(self):
        for name, word3, literal in (("raw", RAW, None), ("raw literal", RAW, RAW), ("disabled", DISABLED, None)):
            with self.subTest(mode=name):
                view, before, read = self.run_case(indexed(2, literal), word3)
                before[BASE + 2 * STRIDE] = VALUE
                self.assertEqual(view, before)
                self.assertEqual(read, [VALUE] * LANES)

    def test_index_whose_product_with_the_stride_passes_32_bits_has_no_effect(self):
        # 0xFFFFFFFF * 20 = 0x13_FFFF_FFEC; modulo 2^32 it would be the 20 bytes below the base.
        for name, word3, literal in (("raw", RAW, None), ("raw literal", RAW, RAW), ("disabled", DISABLED, None),
                                     ("disabled literal", DISABLED, DISABLED)):
            with self.subTest(mode=name):
                view, before, read = self.run_case(indexed(-1, literal), word3)
                self.assertEqual(view, before)
                self.assertEqual(read, [0] * LANES)

    def test_structured_modes_already_refuse_the_index(self):
        view, before, read = self.run_case(indexed(-1), STRUCTURED_OFFSET, records=8)
        self.assertEqual(view, before)
        self.assertEqual(read, [0] * LANES)

    def test_soffset_is_added_without_wrapping(self):
        # DISABLED checks only that the buffer is not empty; modulo 2^32 the access would land at base + 0x10.
        view, before, read = self.run_case(SOFFSET, DISABLED, stride=0)
        self.assertEqual(view, before)
        self.assertEqual(read, [0] * LANES)


if __name__ == "__main__":
    unittest.main()
