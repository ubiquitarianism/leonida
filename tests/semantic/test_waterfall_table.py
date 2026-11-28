"""Check first-active-lane selection with actual memory results for every invocation."""
from pathlib import Path
import unittest

from spirv_mini_interp import run_workgroup
from test_loop_ballot_barrier import _translate_cs

FIXTURE = Path(__file__).with_name("fixtures") / "waterfall_table_offset.asm"
TABLE, OUTPUT, DATA = 0x1000, 0x2000, 0x3000
RAW = 0x31016FAC


def program(lanes, grouped):
    asm = FIXTURE.read_text(encoding="utf-8")
    if not grouped:
        # A single first-lane read outside a retry loop selects one table entry for the wave.
        asm = "\n".join(line for line in asm.splitlines() if not line.startswith(
            ("v_cmpx_", "s_andn2_", "s_cbranch_")))
    if lanes == 64:
        asm = asm.replace("s_mov_b32 s12, exec_lo", "s_mov_b64 s[10:11], exec")
        asm = asm.replace("s_mov_b32 s13, exec_lo", "s_mov_b64 s[12:13], exec")
        asm = asm.replace("s_andn2_wrexec_b32 s13, s13", "s_andn2_wrexec_b64 s[12:13], s[12:13]")
        asm = asm.replace("s_mov_b32 exec_lo, s12", "s_mov_b64 exec, s[10:11]")
    return asm.splitlines()


class WaterfallTableTests(unittest.TestCase):
    def test_selected_lanes_use_their_table_entry(self):
        for lanes, subgroup, emulate in ((32, 32, False), (64, 64, False), (64, 32, True)):
            for grouped in (True, False):
                with self.subTest(lanes=lanes, subgroup=subgroup, grouped=grouped):
                    module, gen = _translate_cs("cs_waterfall_table", program(lanes, grouped),
                                                threads=lanes, emulate=emulate)
                    self.assertEqual(gen.wave64, lanes == 64)
                    memory = {}
                    for group in range(4):
                        address = DATA + group * 4
                        for i, word in enumerate((address, 0, 4, RAW)):
                            memory[TABLE + group * 16 + i * 4] = word
                        memory[address] = 37 + 11 * group
                    result = run_workgroup(
                        module, lanes, subgroup,
                        inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0),
                                              "%sg_lane": i % subgroup, "%local_index": i},
                        push_constants=[OUTPUT, 0, lanes * 4, RAW, TABLE, 0] + [0] * 26, memory=memory)
                    self.assertEqual([result[OUTPUT + i * 4] for i in range(lanes)],
                                     [37 + 11 * ((i // 8) % 4) if grouped else 37 for i in range(lanes)])
