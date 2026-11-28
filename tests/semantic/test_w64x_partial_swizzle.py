"""--wave64-emulate: a cross-lane read from a lane that doesn't exist in a partial last wave gives 0.

The exchange slots of the missing lanes are zeroed in the prologue, but the read must not depend on that: RADV
(Mesa 26.3.1, gfx1036 and RX 7900 XTX) turned the store, barrier and load of an emulated ds_swizzle into a subgroup
swizzle that fetched the missing lane's register, so lanes 32-39 of a 40-thread group read garbage (crosslane_fuzz
seed 48315, 7 programs; runtime/test/shaders/cs_w64x_partial_swizzle.asm). The translator now selects 0 by the
source lane's place in the workgroup. Here the prologue's zero stores are replaced by a poison word before the
workgroup interpreter runs the module, so a read that still took the slot shows the poison.

Fixture: fixtures/w64x_partial_swizzle.asm (hand-written; expected words from its header, the RDNA2 ISA rule for a
source lane that isn't running, and the gfx1036 hardware oracle).
"""
import re
import struct
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from support import rdna2spirv, translate_lines  # noqa: E402
from test_wave64_emulate_opt import VSHARP, run_emulated_workgroup  # noqa: E402

POISON = 0xDEADBEEF
FIXTURE = Path(__file__).with_name("fixtures") / "w64x_partial_swizzle.asm"


def translate(threads, emulate=True):
    lines = [ln for ln in FIXTURE.read_text(encoding="ascii").splitlines() if ln and not ln.startswith(";")]
    fssm = bytearray(0x148)                               # header word 0xD4 = 0: wave64
    struct.pack_into("<4I", fssm, 0x138, 0x8, threads, 1, 1)   # four user SGPRs (the output V#)
    with patch.object(rdna2spirv, "WAVE64_EMULATE", emulate):
        return translate_lines(f"cs_w64x_partial_swizzle_{threads}", lines, fssm=bytes(fssm))


def poison_missing_slots(spv):
    """The prologue's zero stores to the missing lanes' slots store POISON instead."""
    spv, n = re.subn(r"(OpAtomicStore %x_yp\d+_\d+ %\w+ %\w+ )%\w+", rf"\1%x_poison", spv)
    head = re.search(r"^%uint = OpTypeInt 32 0\n", spv, re.MULTILINE)
    return spv[:head.end()] + f"%x_poison = OpConstant %uint {POISON}\n" + spv[head.end():], n


def expected(threads):
    words = {}
    for t in range(threads):
        lane, base = t % 64, t - t % 64

        def read(src):
            return 500 + src if base + src < threads else 0
        words[t] = (read(lane ^ 8), read(lane ^ 8), read((lane & ~15) | 15))
    return words


class PartialWaveSwizzle(unittest.TestCase):
    def check(self, threads):
        spv, stores = poison_missing_slots(translate(threads))
        self.assertGreater(stores, 0, "the prologue zeroes the missing lanes' slots")
        mem, _ = run_emulated_workgroup(spv, (threads, 1, 1), push_constants=VSHARP + [0] * 28)
        for t, want in expected(threads).items():
            got = tuple(mem.get(12 * t + 4 * k, 0) for k in range(3))
            self.assertEqual(got, want, f"thread {t} of {threads}")

    def test_one_partial_wave(self):
        self.check(40)          # lanes 32-39 read lanes 40-47 (SWAP 8, bpermute) and 47 (BROADCAST 16,15)

    def test_partial_second_wave(self):
        self.check(104)         # wave 1 has lanes 0-39; wave 0 is full and reads its own lanes

    def test_full_waves_unchanged(self):
        # No select is added where every lane exists: the module is the one the plain slot read gives.
        with patch.object(rdna2spirv.Gen, "x_lane_word", lambda self, idx, lane: self.x_word(idx)):
            plain = translate(64)
        self.assertEqual(translate(64), plain)


if __name__ == "__main__":
    unittest.main()
