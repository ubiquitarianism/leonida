"""buffer_load_short_d16_hi with inactive lanes: the load writes the high half of vdata in active lanes only.

Standard library only; no GPU, no LLVM, no game files.

tests/semantic/fixtures/cs_buffer_load_d16_hi_exec.asm (derivations in its header) gives every lane of one wave32 a
value of its own in v5, cuts EXEC to a mixed mask and loads a 16-bit word per lane into the high half of v5, then
stores v5 from every lane. An active lane must get the loaded 16 bits in the high half and keep its low half; an
inactive lane must keep both halves (docs/design.md, the "8/16-bit MUBUF d16 loads and stores" row: "Inactive lanes
neither load nor store"). test_buffer_sub_dword.py checks an inactive lane only for a store.
"""
import os
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)  # python -I leaves the script's folder off sys.path

import support  # noqa: E402  (support puts tools/ on sys.path, honouring LEONIDA_TOOLS_DIR)
import spirv_mini_interp as si  # noqa: E402

FIXTURE = os.path.join(HERE, "fixtures", "cs_buffer_load_d16_hi_exec.asm")
FSSM = bytearray(0x148)
struct.pack_into("<4I", FSSM, 0x138, 0x88, 32, 1, 1)
FSSM = bytes(FSSM)
LANES, EXEC, SRC, OUT = 32, 0x5AC3A50F, 0x100, 0x200
SHORTS = [((0x9C00 + 0x35 * i) ^ (i << 12)) & 0xFFFF for i in range(LANES)]   # the word lane i loads, all distinct


def initial(lane):
    return (lane * 0x10001) ^ 0xC0DE1234


def expected(lane):
    if EXEC >> lane & 1:
        return SHORTS[lane] << 16 | initial(lane) & 0xFFFF
    return initial(lane)


def run(spv):
    mem = {SRC + 4 * k: SHORTS[2 * k] | SHORTS[2 * k + 1] << 16 for k in range(LANES // 2)}
    si.run_workgroup(spv, LANES, 32, inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0),
                                                           "%sg_lane": i},
                     push_constants=[0, 0, 0xFFFFFFFF, 0x31016FAC] + [0] * 28, memory=mem)
    return [mem.get(OUT + 4 * i) for i in range(LANES)]


class BufferLoadD16HiExec(unittest.TestCase):
    def test_reference(self):
        self.assertEqual(len(set(SHORTS)), LANES)
        self.assertTrue(0 < EXEC < 0xFFFFFFFF)
        # lane 0 is active, lane 4 is not; written out by hand from the .asm header
        self.assertEqual(SHORTS[0], 0x9C00)
        self.assertEqual(expected(0), 0x9C001234)
        self.assertEqual(expected(4), 0xC0DA1230)
        # every lane's high half changes when it loads, so a lane that loaded by mistake is seen
        for lane in range(LANES):
            self.assertNotEqual(SHORTS[lane], initial(lane) >> 16)

    def test_fixture_matches_this_file(self):
        with open(FIXTURE, encoding="ascii") as f:
            text = f.read()
        self.assertIn(f"s_mov_b32 exec_lo, {EXEC:#x}", text)
        self.assertIn(f"buffer_load_short_d16_hi v5, v1, s[0:3], 0 offen offset:{SRC}", text)
        self.assertIn(f"buffer_store_dword v5, v6, s[0:3], 0 offen offset:{OUT}", text)

    def test_translated_shader(self):
        with open(FIXTURE, encoding="ascii") as f:
            lines = [ln.strip() for ln in f if ln.strip() and not ln.lstrip().startswith(";")]
        got = run(support.translate_lines("cs_buffer_load_d16_hi_exec", lines, fssm=FSSM))
        for lane in range(LANES):
            with self.subTest(lane=lane, active=bool(EXEC >> lane & 1)):
                self.assertEqual(got[lane], expected(lane), f"{got[lane]:#010x} != {expected(lane):#010x}")


if __name__ == "__main__":
    unittest.main()
