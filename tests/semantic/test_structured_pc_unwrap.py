"""Semantic regression tests for structured %pc guard unwrapping (SHADER-STRUCTURED-PC-UNWRAP).

Pure Python stdlib only (no llvm-mc, spirv-val, GPU, or game data). Verifies end-to-end translation
and execution of synthetic hand-written RDNA2 compute shaders where the last basic block ends in a
conditional branch with no trailing s_endpgm (fallthrough writes %pc = len(blocks) = END):

  1. Loop whose back-edge is a conditional branch at the very end of the shader (succ = [[2, 1], [], [0]],
     out_pcs[2] = {0, END}): exiting the loop leaves %pc in {1, END}, so guard 1 must stay and a
     not-taken back-edge must terminate without executing block 1.
  2. Non-loop conditional branch at the very end of the shader whose single explicit target is
     immediately dominated by it (succ = [[2], [], [1]], out_pcs[2] = {1, END}): both native
     structured control flow and --wave64-emulate structured control flow must keep guard 1.
  3. Single-arm if-then join (0 -> {2, 1}, 1 -> {2}, 2 -> []) with s_barrier in block 2: block 2
     unwraps to top level so wave32 compute does not falsely fall back to barrier rounds.
"""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

sys.dont_write_bytecode = True
for _p in (HERE, TOOLS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import rdna2spirv  # noqa: E402
from test_loop_ballot_barrier import VSHARP, _translate_cs, run_compute_workgroup  # noqa: E402


class StructuredPcUnwrapSemanticTest(unittest.TestCase):
    def test_loop_last_block_cbranch_falloff_keeps_exit_guard(self):
        """Loop back-edge in the last block with no s_endpgm after it: falling off the end must NOT run block 1."""
        prog = [
            "head:",
            "s_cmp_eq_u32 s4, 0",
            "s_cbranch_scc1 tail",
            "v_lshlrev_b32_e32 v2, 2, v0",
            "v_mov_b32_e32 v1, 0x1111",
            "buffer_store_dword v1, v2, s[0:3], 0 offen",
            "s_endpgm",
            "tail:",
            "s_mov_b32 s4, 1",
            "s_cmp_eq_u32 s5, 1",
            "s_cbranch_scc1 head",
        ]
        spv_struct, gen_struct = _translate_cs("cs_cbranch_falloff_loop", prog, threads=32, structured=True)
        spv_disp, gen_disp = _translate_cs("cs_cbranch_falloff_loop_disp", prog, threads=32, structured=False)
        self.assertEqual(gen_struct.control_flow, "structured")
        self.assertEqual(gen_disp.control_flow, "dispatch loop")
        self.assertEqual(gen_struct.succ, [[2, 1], [], [0]])
        self.assertEqual(gen_struct.out_pcs[2], frozenset({0, -1}))
        tree, _ = rdna2spirv.capped_tree(gen_struct.succ, gen_struct.out_pcs)
        self.assertEqual(tree, [
            ("loop", 0, [("block", 0), ("guard", 2, [("block", 2)])]),
            ("guard", 1, [("block", 1)]),
        ])

        cases = [
            (1, 0, [0x1111] * 32),   # B0 falls into B1 -> stores 0x1111
            (0, 1, [0x1111] * 32),   # B0 -> B2 (s4=1, back-edge taken) -> B0 -> B1 -> stores 0x1111
            (0, 0, [0] * 32),        # B0 -> B2 (back-edge not taken -> falls off end) -> no store
        ]
        for s4, s5, want in cases:
            with self.subTest(s4=s4, s5=s5):
                pc = VSHARP + [s4, s5]
                mem_s = run_compute_workgroup(spv_struct, 32, 32, push_constants=pc)
                mem_d = run_compute_workgroup(spv_disp, 32, 32, push_constants=pc)
                got_s = [mem_s.get(4 * t, 0) for t in range(32)]
                got_d = [mem_d.get(4 * t, 0) for t in range(32)]
                self.assertEqual(got_d, want)
                self.assertEqual(got_s, want)

    def test_nonloop_last_block_cbranch_falloff_keeps_dominated_target_guard(self):
        """Last block ending in s_cbranch_scc1 to its single dominated target (no s_endpgm) must keep its guard."""
        prog = [
            "s_branch tail",
            "do_store:",
            "v_lshlrev_b32_e32 v2, 2, v0",
            "v_mov_b32_e32 v1, 0x2222",
            "buffer_store_dword v1, v2, s[0:3], 0 offen",
            "s_endpgm",
            "tail:",
            "s_cmp_eq_u32 s4, 1",
            "s_cbranch_scc1 do_store",
        ]
        for mode, threads, sg_sz, emulate in (
            ("w32_structured", 32, 32, False),
            ("w64_emulate", 64, 32, True),
        ):
            # Include a 64-bit exec op in the wave64 case so rdna2spirv classifies the shader as wave64
            # and exercises wave64_emulate_opt.plan_structured_emulation().
            mode_prog = (["s_mov_b64 s[10:11], exec"] + prog) if emulate else prog
            spv, gen = _translate_cs(f"cs_cbranch_falloff_{mode}", mode_prog, threads=threads, emulate=emulate)
            if emulate:
                self.assertEqual(gen.control_flow, "emulated wave64 structured")
            else:
                self.assertEqual(gen.control_flow, "structured")
                tree, _ = rdna2spirv.capped_tree(gen.succ, gen.out_pcs)
                self.assertEqual(tree, [("block", 0), ("block", 2), ("guard", 1, [("block", 1)])])
            for s4, want_val in ((1, 0x2222), (0, 0)):
                with self.subTest(mode=mode, s4=s4):
                    mem = run_compute_workgroup(spv, threads, sg_sz, push_constants=VSHARP + [s4])
                    self.assertEqual([mem.get(4 * t, 0) for t in range(threads)], [want_val] * threads)

    def test_postdominating_join_barrier_unwraps_to_top_level(self):
        """A post-dominating merge block after a single-arm branch unwraps to top level (no false barrier rounds)."""
        prog = [
            "v_lshlrev_b32_e32 v2, 2, v0",
            "v_mov_b32_e32 v1, 1",
            "s_cmp_eq_u32 s4, 0",
            "s_cbranch_scc1 join",
            "v_mov_b32_e32 v1, 2",
            "join:",
            "ds_write_b32 v2, v1",
            "s_waitcnt lgkmcnt(0)",
            "s_barrier",
            "ds_read_b32 v3, v2",
            "s_waitcnt lgkmcnt(0)",
            "buffer_store_dword v3, v2, s[0:3], 0 offen",
            "s_endpgm",
        ]
        spv, gen = _translate_cs("cs_join_barrier_unwrap", prog, threads=64, lds_bytes=256, emulate=False)
        self.assertFalse(gen.bar_rounds)
        self.assertEqual(gen.control_flow, "structured")
        tree, _ = rdna2spirv.capped_tree(gen.succ, gen.out_pcs)
        self.assertEqual(tree, [("block", 0), ("guard", 1, [("block", 1)]), ("block", 2)])
        for s4, want_val in ((0, 1), (1, 2)):
            with self.subTest(s4=s4):
                mem = run_compute_workgroup(spv, 64, 32, push_constants=VSHARP + [s4])
                self.assertEqual([mem.get(4 * t, 0) for t in range(64)], [want_val] * 64)

    def test_wave64_emulate_exec_fold_refuses_last_block_cbranch_falloff(self):
        """A wave64 s_cbranch_execz region containing a trailing conditional branch (no s_endpgm) must NOT be
        folded into a region that unconditionally continues at `skip`: falling off the end must terminate."""
        prog_to_x = [
            "s_mov_b64 s[10:11], exec",
            "s_cbranch_execz skip",
            "s_branch last",
            "X:",
            "v_lshlrev_b32_e32 v2, 2, v0",
            "v_mov_b32_e32 v1, 0x3333",
            "buffer_store_dword v1, v2, s[0:3], 0 offen",
            "s_branch skip",
            "skip:",
            "v_lshlrev_b32_e32 v2, 2, v0",
            "v_mov_b32_e32 v1, 0x4444",
            "buffer_store_dword v1, v2, s[0:3], 0 offen offset:256",
            "s_endpgm",
            "last:",
            "s_cmp_eq_u32 s4, 1",
            "s_cbranch_scc1 X",
        ]
        prog_to_skip = [
            "s_mov_b64 s[10:11], exec",
            "s_cbranch_execz skip",
            "s_branch last",
            "skip:",
            "v_lshlrev_b32_e32 v2, 2, v0",
            "v_mov_b32_e32 v1, 0x4444",
            "buffer_store_dword v1, v2, s[0:3], 0 offen offset:256",
            "s_endpgm",
            "last:",
            "s_cmp_eq_u32 s4, 1",
            "s_cbranch_scc1 skip",
        ]
        for name, prog, want_taken_x, want_taken_skip in (
            ("branch_to_x", prog_to_x, 0x3333, 0x4444),
            ("branch_to_skip", prog_to_skip, 0, 0x4444),
        ):
            for structured in (True, False):
                spv, _ = _translate_cs(
                    f"cs_fold_falloff_{name}_{int(structured)}",
                    prog, threads=64, emulate=True, structured=structured,
                )
                for s4, exp_x, exp_skip in ((0, 0, 0), (1, want_taken_x, want_taken_skip)):
                    with self.subTest(case=name, structured=structured, s4=s4):
                        mem = run_compute_workgroup(spv, 64, 32, push_constants=VSHARP + [s4])
                        self.assertEqual([mem.get(4 * t, 0) for t in range(64)], [exp_x] * 64)
                        self.assertEqual([mem.get(256 + 4 * t, 0) for t in range(64)], [exp_skip] * 64)


if __name__ == "__main__":
    unittest.main()
