"""Wave64: a lane mask the listing names by its low SGPR only is the pair.

The shader listings come from llvm-mc without +wavefrontsize64, which prints the wave32 form of a VALU instruction's
lane-mask operands: `v_cmp_gt_u32_e64 s8, ...` for the s[8:9] a wave64 writes, `v_cndmask_b32_e64 ..., s10` for the
s[10:11] it reads. Scalar 64-bit instructions print pairs, so the same mask is read as s[8:9] afterwards; before the
high register was stale and lanes 32..63 took their EXEC bits from it. A pipeline-library compute shader of the
retail title looped for ever on a 32-wide GPU because of it.

Synthetic programs, no game data.

    python -I tests/semantic/test_wave64_single_sdst.py -v
"""
import os
import re
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import test_wave64_emulate_opt as emulated  # noqa: E402

rdna2spirv = emulated.rdna2spirv

STALE = [("s_mov_b32 s9, 0x100001", 8), ("s_mov_b32 s11, 0x40", 8)]     # what the high registers held before
STORE = [
    ("s_mov_b64 exec, -1", 4),
    ("v_lshlrev_b32_e32 v6, 2, v0", 4),
    ("buffer_store_dword v5, v6, s[0:3], 0 offen", 8),
    ("s_endpgm", 4),
]
MASK_MOVED_TO_EXEC = STALE + [
    ("v_mov_b32_e32 v5, 0", 4),
    ("v_cmp_gt_u32_e64 s8, 48, v0", 8),             # lanes 0..47
    ("s_mov_b64 exec, s[8:9]", 4),
    ("v_mov_b32_e32 v5, 1", 4),
] + STORE
CARRY_AND_SELECT = STALE + [
    ("v_add_co_u32_e64 v1, s8, -40, v0", 8),        # carry out of lane + 0xffffffd8: lanes 40..63
    ("s_mov_b64 s[10:11], s[8:9]", 4),
    ("v_cndmask_b32_e64 v5, 2, 3, s10", 8),
    ("s_mov_b64 exec, s[10:11]", 4),
    ("v_add_co_ci_u32_e64 v5, s8, v5, 4, s10", 8),  # + 4 + the carry in, which those lanes have
] + STORE
PAIRS = re.compile(r"\b(v_cmp_gt_u32_e64 )s8,|(v_add_co(?:_ci)?_u32_e64 v\d, )s8,|(, )s10$")


def printed_as_wave64(program):
    """The program as llvm-mc prints it with +wavefrontsize64."""
    def pair(m):
        return f"{m.group(1)}s[8:9]," if m.group(1) else f"{m.group(2)}s[8:9]," if m.group(2) else ", s[10:11]"
    return [(PAIRS.sub(pair, text) if text.startswith("v_") else text, size) for text, size in program]


class SingleSdstTest(unittest.TestCase):
    def run_program(self, name, program):
        for emulate in (True, False):
            spv, _ = emulated._translate_encoded(name, program, (64, 1, 1), emulate=emulate)
            wide, _ = emulated._translate_encoded(name, printed_as_wave64(program), (64, 1, 1), emulate=emulate)
            self.assertEqual(spv, wide, "the two forms of the listing are the same program")
        spv, _ = emulated._translate_encoded(name, program, (64, 1, 1), emulate=True)
        memory, _ = emulated.run_emulated_workgroup(spv, (64, 1, 1), push_constants=emulated.VSHARP + [0] * 28)
        return [memory[4 * lane] for lane in range(64)]

    def test_compare_destination_read_as_a_pair(self):
        self.assertNotEqual(MASK_MOVED_TO_EXEC, printed_as_wave64(MASK_MOVED_TO_EXEC))
        self.assertEqual(self.run_program("cs_w64_cmp_single", MASK_MOVED_TO_EXEC), [1] * 48 + [0] * 16)

    def test_carry_out_carry_in_and_select_mask(self):
        self.assertEqual(self.run_program("cs_w64_carry_single", CARRY_AND_SELECT), [2] * 40 + [8] * 24)

    def test_wave32_keeps_single_registers(self):
        insns = [{"op": "v_cmp_gt_u32_e64", "args": [{"k": "sgpr", "r": 8}, {"k": "imm", "v": 1}, {"k": "vgpr", "r": 0}]}]
        rdna2spirv.widen_wave64_masks(insns)
        self.assertEqual(insns[0]["args"][0], {"k": "sgpr", "r": 8, "n": 2})
        # translate() calls it for wave64 programs only; an e32 compare and VCC are left as they are
        other = [{"op": "v_cmp_gt_u32_e32", "args": [{"k": "special", "t": "vcc_lo"}, {"k": "imm", "v": 1}, {"k": "vgpr", "r": 0}]},
                 {"op": "v_readlane_b32", "args": [{"k": "sgpr", "r": 8}, {"k": "vgpr", "r": 0}, {"k": "sgpr", "r": 2}]},
                 {"op": "v_cmpx_eq_u32_e64", "args": [{"k": "sgpr", "r": 4}, {"k": "vgpr", "r": 0}]}]
        before = [dict(i, args=list(i["args"])) for i in other]
        rdna2spirv.widen_wave64_masks(other)
        self.assertEqual(other, before)


if __name__ == "__main__":
    unittest.main()
