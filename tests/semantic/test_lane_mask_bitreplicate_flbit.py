"""s_bitreplicate_b64_b32 into VCC and s_flbit_i32_b64 of a lane mask, run on every lane of a wave64 in
spirv_mini_interp.run_workgroup.

RDNA2 ISA, SOP1:
  - s_bitreplicate_b64_b32: D[2i+1] = D[2i] = S0[i], i in 0..31; SCC is not written. Into VCC, lane L's bit is S0 bit
    L // 2.
    Of a 32-bit lane-mask source (exec_lo/exec_hi, vcc_lo/vcc_hi) into scalar data, S0 is that half of the wave's
    mask, read as data (Gen.wave_mask), then replicated as any scalar.
  - s_flbit_i32_b64: the zeros above the first 1 from bit 63, -1 when S0 is 0. Of EXEC that is 63 minus the wave's
    last active lane.
Synthetic programs only.
"""
import unittest

import saveexec_run as h
import support
import spirv_mini_interp as si

OUT_AT = 256


def run(lines):
    spv = support.translate_lines("cs_lane_mask_bitreplicate_flbit", lines, fssm=h.fssm())
    return si.run_workgroup(spv, 64, 64, inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i},
                            push_constants=h.VSHARP + [0] * 28, memory={})


def lane_index():
    return ["v_mbcnt_lo_u32_b32 v1, -1, 0", "v_mbcnt_hi_u32_b32 v1, -1, v1", "v_lshlrev_b32_e32 v21, 2, v1"]


def flbit_model(mask):
    return 63 - (mask.bit_length() - 1) if mask else 0xFFFFFFFF


class LaneMaskBitreplicateFlbit(unittest.TestCase):
    def test_bitreplicate_into_vcc(self):
        for k in (0x0F0F0F0F, 0xFF00FF00, 0, 0xFFFFFFFF, 0x80000001, 0x12345678):
            with self.subTest(k=hex(k)):
                mem = run(lane_index() + [f"s_bitreplicate_b64_b32 vcc, {k:#x}", "v_cndmask_b32_e64 v2, 0, 1, vcc",
                                          "buffer_store_dword v2, v21, s[0:3], 0 offen", "s_endpgm"])
                got = [mem[4 * lane] for lane in range(64)]
                self.assertEqual(got, [(k >> (lane // 2)) & 1 for lane in range(64)])

    def test_flbit_of_exec(self):
        for mask in (h.ALL64, 1, 0x8000000000000000, 0x00000000FFFFFFFF, 0x0000F00000000000, 0x10):
            with self.subTest(exec=hex(mask)):
                mem = run(lane_index() + [f"s_mov_b32 exec_lo, {mask & 0xFFFFFFFF:#x}", f"s_mov_b32 exec_hi, {mask >> 32:#x}",
                                          "s_flbit_i32_b64 s10, exec", "s_mov_b64 exec, -1",
                                          "v_cmpx_eq_u32_e32 0, v1", "v_mov_b32_e32 v20, s10",
                                          f"v_mov_b32_e32 v21, {OUT_AT}", "buffer_store_dword v20, v21, s[0:3], 0 offen",
                                          "s_endpgm"])
                self.assertEqual(mem[OUT_AT], flbit_model(mask))

    def test_flbit_of_vcc(self):
        # VCC from a compare: the lanes l with l < n.
        for n in (0, 1, 31, 33, 64):
            with self.subTest(n=n):
                mem = run(lane_index() + [f"v_cmp_gt_u32_e32 vcc, {n}, v1", "s_flbit_i32_b64 s10, vcc",
                                          "v_cmpx_eq_u32_e32 0, v1", "v_mov_b32_e32 v20, s10",
                                          f"v_mov_b32_e32 v21, {OUT_AT}", "buffer_store_dword v20, v21, s[0:3], 0 offen",
                                          "s_endpgm"])
                self.assertEqual(mem[OUT_AT], flbit_model((1 << n) - 1))

    def bitreplicate_of(self, setup, src):
        mem = run(lane_index() + setup + [f"s_bitreplicate_b64_b32 s[10:11], {src}", "s_mov_b64 exec, -1",
                                          "v_cmpx_eq_u32_e32 0, v1", "v_mov_b32_e32 v20, s10",
                                          "v_mov_b32_e32 v22, s11", f"v_mov_b32_e32 v21, {OUT_AT}",
                                          "buffer_store_dword v20, v21, s[0:3], 0 offen",
                                          "buffer_store_dword v22, v21, s[0:3], 0 offen offset:4", "s_endpgm"])
        return mem[OUT_AT] | mem[OUT_AT + 4] << 32

    def test_bitreplicate_of_exec_half_into_data(self):
        # Every lane that runs the instruction reads the same 32 mask bits; each becomes two bits of s[10:11].
        for mask in (h.ALL64, 0x00000000FFFFFFFF, 0xFFFFFFFF00000000, 0x8000000180000001, 0x0F0F0F0F12345678):
            for half, src in ((0, "exec_lo"), (1, "exec_hi")):
                with self.subTest(exec=hex(mask), src=src):
                    word = (mask >> (32 * half)) & 0xFFFFFFFF
                    got = self.bitreplicate_of([f"s_mov_b32 exec_lo, {mask & 0xFFFFFFFF:#x}",
                                                f"s_mov_b32 exec_hi, {mask >> 32:#x}"], src)
                    self.assertEqual(got, bitreplicate_model(word))

    def test_bitreplicate_of_a_compare_into_data(self):
        # VCC from a compare: lanes l < n. Its low and high halves replicated as data.
        for n in (0, 1, 17, 32, 45, 64):
            for half, src in ((0, "vcc_lo"), (1, "vcc_hi")):
                with self.subTest(n=n, src=src):
                    got = self.bitreplicate_of([f"v_cmp_gt_u32_e32 vcc, {n}, v1"], src)
                    self.assertEqual(got, bitreplicate_model((((1 << n) - 1) >> (32 * half)) & 0xFFFFFFFF))


def bitreplicate_model(word):
    return sum(3 << (2 * i) for i in range(32) if (word >> i) & 1)


if __name__ == "__main__":
    unittest.main()
