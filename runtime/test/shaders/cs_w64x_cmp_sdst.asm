; regen_fixtures: this source has a translator fixture in tests/semantic/fixtures/disasm
; TEST-W64-SDST-PAIR-GPU: the GPU test of SHADER-W64-SDST-PAIR (docs/design.md, "Wave64 VALU mask operands"): in
; wave64 code a VALU instruction that writes a lane mask to an SGPR writes the pair.
; A wave64 lane mask is 64 bits: v_cmp_*_e64 with SDST 4 writes s[4:5], and so does the carry-out of v_add_co_u32.
; llvm-mc names that destination by the wave size it is told, and the game's listings are disassembled without
; +wavefrontsize64, so they read "s4" while the 64-bit scalar ops after them read "s[4:5]". This file is written
; the same way on purpose: it has NO "mattr:" line, so llvm-mc assembles and prints it as it prints the game's
; shaders (one register), and the translator knows the wave size from the code (v_mbcnt_hi_u32_b32, s_*_b64 on
; EXEC). Every other wave64 fixture here names the pair, which is why none of them saw the stale high register.
; One 64-lane wave, v1 = lane 0..63, all lanes on. A mask is reported as bcnt << 8 | ff1 (population count, lowest
; lane), read with s_bcnt1_i32_b64 / s_ff1_i32_b64. Each compare below has lanes on both sides of lane 32, and the
; high register of its pair holds something else before it (another mask, 0, or nothing).
; 0) s[4:5] = EXEC (every lane), then v_cmp_gt_u32_e64 s4, 40, v1: lanes 0..39.
;    s_and_b64 s[6:7], exec, s[4:5] = lanes 0..39: out[0] = 40 << 8 | 0 = 10240. SCC 1.
;    (the old s5 for lanes 32..63: every lane, 16384.)
; 1) s[10:11] = 0, then v_cmp_le_u32_e64 s10, 48, v1: lanes 48..63. v_cmp_gt_u32_e64 s14, 8, v1: lanes 0..7
;    (s15 never written). s_or_b64 s[12:13], s[14:15], s[10:11] = lanes 0..7 and 48..63:
;    out[1] = 24 << 8 | 0 = 6144. (the old s11 and s15: lanes 0..7 only, 2048.)
; 2) v2 = lane & 7; v_cmp_eq_u32_e64 s16, 5, v2: lanes 5, 13, .., 61 (8 lanes).
;    s_and_saveexec_b64 s[18:19], s[16:17]; v3 = 1 under it; EXEC restored; v_cmp_ne_u32_e64 s20, 0, v3 and
;    s_bcnt1 / s_ff1 of s[20:21]: out[2] = 8 << 8 | 5 = 2053.
; 3) v_add_co_u32_e64 v4, s22, v1, v5 with v5 = 0xffffffe0: the carry-out is lanes 32..63 (lane + 2^32 - 32 wraps).
;    s_and_b64 s[24:25], exec, s[22:23]: out[3] = 32 << 8 | 32 = 8224.
; 4) v_cmp_gt_u32_e64 s26, 34, v1: lanes 0..33. s_xor_b64 s[28:29], exec, s[26:27] = lanes 34..63:
;    out[4] = 30 << 8 | 34 = 7714.
; 5) s_andn2_b64 s[32:33], exec, s[4:5] (the mask of case 0) = lanes 40..63: out[5] = 24 << 8 | 40 = 6184.
; 6) s_mov_b64 exec, s[10:11] (lanes 48..63, all in the upper half); v7 = 1 under it; EXEC = every lane again;
;    v_cmp_ne_u32_e32 vcc_lo, 0, v7 (VCC is printed vcc_lo the same way) and s_bcnt1 / s_ff1 of vcc:
;    out[6] = 16 << 8 | 48 = 4144.
; 7) SCC of three 64-bit ANDs: case 0 (1); s_and_b64 s[38:39], s[10:11], s[14:15] = no lane (0);
;    s_and_b64 s[40:41], s[10:11], s[22:23] = lanes 48..63, upper half only (1): out[7] = 1 + 0 * 2 + 1 * 4 = 5.
; Lane 0 stores the eight words.
; Run: vk_compute_test cs_w64x_cmp_sdst.spv 1 dump --require-wave64
;   -> out[0..7] = 10240 6144 2053 8224 7714 6184 4144 5
v_mbcnt_lo_u32_b32 v1, -1, 0
v_mbcnt_hi_u32_b32 v1, -1, v1
; 0) the high register holds another mask
s_mov_b64 s[4:5], exec
v_cmp_gt_u32_e64 s4, 40, v1
s_and_b64 s[6:7], exec, s[4:5]
s_cselect_b32 s50, 1, 0
s_bcnt1_i32_b64 s8, s[6:7]
s_ff1_i32_b64 s9, s[6:7]
s_lshl_b32 s8, s8, 8
s_or_b32 s60, s8, s9
; 1) the high register holds 0, or nothing
s_mov_b64 s[10:11], 0
v_cmp_le_u32_e64 s10, 48, v1
v_cmp_gt_u32_e64 s14, 8, v1
s_or_b64 s[12:13], s[14:15], s[10:11]
s_bcnt1_i32_b64 s8, s[12:13]
s_ff1_i32_b64 s9, s[12:13]
s_lshl_b32 s8, s8, 8
s_or_b32 s61, s8, s9
; 2) s_and_saveexec_b64 from the pair, and a count of a pair
v_and_b32_e32 v2, 7, v1
v_mov_b32_e32 v3, 0
v_cmp_eq_u32_e64 s16, 5, v2
s_and_saveexec_b64 s[18:19], s[16:17]
v_mov_b32_e32 v3, 1
s_mov_b64 exec, s[18:19]
v_cmp_ne_u32_e64 s20, 0, v3
s_bcnt1_i32_b64 s8, s[20:21]
s_ff1_i32_b64 s9, s[20:21]
s_lshl_b32 s8, s8, 8
s_or_b32 s62, s8, s9
; 3) a carry-out
v_mov_b32_e32 v5, 0xffffffe0
v_add_co_u32_e64 v4, s22, v1, v5
s_and_b64 s[24:25], exec, s[22:23]
s_bcnt1_i32_b64 s8, s[24:25]
s_ff1_i32_b64 s9, s[24:25]
s_lshl_b32 s8, s8, 8
s_or_b32 s63, s8, s9
; 4) s_xor_b64
v_cmp_gt_u32_e64 s26, 34, v1
s_xor_b64 s[28:29], exec, s[26:27]
s_bcnt1_i32_b64 s8, s[28:29]
s_ff1_i32_b64 s9, s[28:29]
s_lshl_b32 s8, s8, 8
s_or_b32 s64, s8, s9
; 5) s_andn2_b64
s_andn2_b64 s[32:33], exec, s[4:5]
s_bcnt1_i32_b64 s8, s[32:33]
s_ff1_i32_b64 s9, s[32:33]
s_lshl_b32 s8, s8, 8
s_or_b32 s65, s8, s9
; 6) the pair as EXEC
v_mov_b32_e32 v7, 0
s_mov_b64 exec, s[10:11]
v_mov_b32_e32 v7, 1
s_mov_b64 exec, -1
v_cmp_ne_u32_e32 vcc_lo, 0, v7
s_bcnt1_i32_b64 s8, vcc
s_ff1_i32_b64 s9, vcc
s_lshl_b32 s8, s8, 8
s_or_b32 s66, s8, s9
; 7) SCC
s_and_b64 s[38:39], s[10:11], s[14:15]
s_cselect_b32 s51, 2, 0
s_and_b64 s[40:41], s[10:11], s[22:23]
s_cselect_b32 s52, 4, 0
s_or_b32 s67, s50, s51
s_or_b32 s67, s67, s52
; lane 0 stores the words
v_cmpx_eq_u32_e32 0, v1
v_mov_b32_e32 v20, s60
v_mov_b32_e32 v21, 0
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, s61
v_mov_b32_e32 v21, 4
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, s62
v_mov_b32_e32 v21, 8
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, s63
v_mov_b32_e32 v21, 12
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, s64
v_mov_b32_e32 v21, 16
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, s65
v_mov_b32_e32 v21, 20
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, s66
v_mov_b32_e32 v21, 24
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, s67
v_mov_b32_e32 v21, 28
buffer_store_dword v20, v21, s[0:3], 0 offen
s_endpgm
