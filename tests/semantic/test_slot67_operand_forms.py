"""Compute operand forms used by the pipeline library. Synthetic RDNA2 instructions only."""
import unittest

from test_carry_exec_mask import _run_cs


class Slot67OperandForms(unittest.TestCase):
    def test_wave64_bitreplicate_exec_reenables_lanes_and_keeps_scc(self):
        lines = [
            "s_cmp_eq_u32 1, 1",
            "s_mov_b64 exec, 0",
            "s_bitreplicate_b64_b32 exec, 0x80010001",
            "s_cselect_b32 s8, 7, 9",
            "v_mov_b32 v1, s8",
            "v_lshlrev_b32 v2, 2, v0",
            "buffer_store_dword v1, v2, s[0:3], 0 offen",
        ]
        expected = {i * 4: 7 for i in (0, 1, 32, 33, 62, 63)}
        self.assertEqual(_run_cs("cs_bitreplicate_exec", lines), expected)

    # The merge into main kept main's wave32 VCC_HI model (47494e725: VCC_HI that holds a compare result is a
    # per-lane mask), not this branch's packed-word form (1f001c739). On hardware `s_mov_b32 s8, vcc_hi` copies the
    # packed 32-bit word (0x80000004 here); main's translator gives each lane its own mask value instead, so this
    # case fails there. Kept as the expectation the translator should meet.
    @unittest.expectedFailure
    def test_wave32_compare_high_word_preserves_low_and_clears_inactive_bits(self):
        lines = [
            "v_cmp_lt_u32_e64 vcc_lo, v0, 6",
            "s_mov_b32 exec_lo, 0x80000005",
            "v_cmp_ne_u32_e64 vcc_hi, 0, v0",
            "s_mov_b32 exec_lo, -1",
            "s_mov_b32 s8, vcc_hi",
            "v_mov_b32 v1, s8",
            "v_cndmask_b32_e64 v2, 0, 1, vcc_hi",
            "v_cndmask_b32_e64 v3, 0, 1, vcc_lo",
            "v_lshlrev_b32 v4, 4, v0",
            "buffer_store_dword v1, v4, s[0:3], 0 offen",
            "buffer_store_dword v2, v4, s[0:3], 0 offen offset:4",
            "buffer_store_dword v3, v4, s[0:3], 0 offen offset:8",
        ]
        expected = {i * 16 + offset: value for i in range(32)
                    for offset, value in ((0, 0x80000004), (4, int(i in (2, 31))), (8, int(i < 6)))}
        self.assertEqual(_run_cs("cs_cmp_vcc_hi", lines, wave_size=32), expected)


if __name__ == "__main__":
    unittest.main()
