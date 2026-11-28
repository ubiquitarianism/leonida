"""In DPP with bound_ctrl:1, src0 float modifiers apply after the invalid-lane zero.

Hardware measurement on AMD RDNA3 (RX 7800 XT, HW-DPP-NEG-INVALID-LANE, Issue #1608):
An invalid source lane under bound_ctrl:1 produces 0, and the ALU input modifier
stage applies after:
- `-src0` yields `-0.0f` (0x80000000)
- `|src0|` yields `+0.0f` (0x00000000)
- `-|src0|` yields `-0.0f` (0x80000000)

Standard library only; no GPU, LLVM or retail game files.
"""
import os
import re
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import support  # noqa: E402
import spirv_wave_interp as wi  # noqa: E402


def f2u(f):
    return struct.unpack("<I", struct.pack("<f", f))[0]


class DppBoundCtrlModifiersTest(unittest.TestCase):
    def test_dpp_neg_invalid_lane_yields_negative_zero(self):
        # row_shr:1 with bound_ctrl:1 means lane 0 of each 16-lane row has no source in the row.
        # It reads an invalid source. With -v1, lane 0 should read -0.0 (0x80000000).
        lines = [
            "v_mbcnt_lo_u32_b32 v0, -1, 0",      # v0 = lane id
            "v_cvt_f32_u32 v1, v0",              # v1 = float(lane id)
            "v_mov_b32_dpp v2, -v1 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1",
            "v_mov_b32 v5, 0",
            "exp pos0 v5, v5, v5, v5 done",
            "exp param0 v2, v5, v5, v5",
            "s_endpgm",
        ]
        spv = support.translate_lines("vs_dpp_neg_zero", lines)
        res = wi.run_wave(spv, wave=32, builtins={"VertexIndex": lambda lane: lane, "InstanceIndex": lambda lane: 0})
        out_raw = [o["%out_param0"][0] for o in res["outputs"]]

        # Lane 0 and lane 16 are the first lanes in each 16-lane row; they have no predecessor.
        # They must read -0.0f (0x80000000).
        self.assertEqual(out_raw[0], 0x80000000, "Lane 0 invalid source with -v1 must yield -0.0f (0x80000000)")
        self.assertEqual(out_raw[16], 0x80000000, "Lane 16 invalid source with -v1 must yield -0.0f (0x80000000)")

        # Lane 1's source in the row is lane 0 (which was float(0.0)). Negated, it is also -0.0f.
        self.assertEqual(out_raw[1], 0x80000000)

        # Lane 2's source in the row is lane 1 (float(1.0)). Negated, it is -1.0f.
        self.assertEqual(out_raw[2], f2u(-1.0))
        # Lane 15's source is lane 14 (float(14.0)). Negated, it is -14.0f.
        self.assertEqual(out_raw[15], f2u(-14.0))

    def test_dpp_modifier_spirv_structure(self):
        # Verify the generated SPIR-V structure for abs, neg, and neg-abs on DPP operands.
        # OpSelect %uint %src_ok %shuffled %c0 must be followed by FAbs / OpFNegate.
        lines_abs = [
            "v_mov_b32_dpp v2, |v1| row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1",
            "s_endpgm",
        ]
        spv_abs = support.translate_lines("vs_dpp_abs", lines_abs)
        matches = re.findall(r"(%\w+) = OpSelect %uint (%\w+) (%\w+) %c0", spv_abs)
        self.assertTrue(len(matches) >= 1, "OpSelect against %c0 must be emitted")
        sel_res = matches[-1][0]  # The src0 select is the last one against %c0

        # Bitcast to float then FAbs
        m_cast = re.search(rf"(%\w+) = OpBitcast %float {re.escape(sel_res)}", spv_abs)
        self.assertIsNotNone(m_cast, f"OpBitcast %float {sel_res} must follow select")
        cast_res = m_cast.group(1)
        self.assertIn(f"OpExtInst %float %glsl FAbs {cast_res}", spv_abs, "FAbs must operate on the select result")

        lines_neg_abs = [
            "v_mov_b32_dpp v2, -|v1| row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1",
            "s_endpgm",
        ]
        spv_neg_abs = support.translate_lines("vs_dpp_neg_abs", lines_neg_abs)
        self.assertIn("FAbs", spv_neg_abs)
        self.assertIn("OpFNegate", spv_neg_abs)


if __name__ == "__main__":
    unittest.main()
