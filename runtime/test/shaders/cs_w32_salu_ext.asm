; GPU-SALU-EXT: 38 scalar ALU ops on operands that arrive at run time.
; group: 1,1,1
; rsrc2: 0x80a8
; mattr: +wavefrontsize32,-wavefrontsize64
; Twenty user SGPRs: s[0:3] are the output V#, and s4..s19 are sixteen operand words that vk_compute_test seeds
; from VKC_SGPRS (check.py's SALU_EXT_SGPRS), so the translator cannot fold a scalar op as it folds one whose
; operands are literals:
;   s4 0x7fffffff  s5 0x80000000  s6 0xffffffff  s7 1           s8 0x12345678  s9 0xdeadbeef  s10 0x0000ff80
;   s11 37 (a count)              s[12:13] 0x00000000_00000040   s[14:15] 0x80000001_00000001
;   s[16:17] 0xffffffff_ffff0000  s18 0       s19 71 (a count)
; One thread runs every op and stores 73 words, out[i] at byte 4 * i. A result that is a pair takes two words, low
; first. The values are the RDNA2 ISA's (docs/design.md); tests/test_gpu_salu_ext.py recomputes them from the
; instructions below with the reference of tools/salu_diff.py.
;   out[0]      s_sub_i32 s32, s4, s5                                      0xffffffff   signed overflow
;   out[1]      s_sub_i32 s33, s8, s7                                      0x12345677
;   out[2]      s_andn2_b32 s34, s9, s8                                    0xcc89a887
;   out[3]      s_andn2_b32 s35, s8, s6                                    0x00000000   zero
;   out[4]      s_orn2_b32 s36, s8, s9                                     0x33765778
;   out[5]      s_orn2_b32 s37, s18, s6                                    0x00000000   zero
;   out[6]      s_xor_b32 s38, s8, s9                                      0xcc99e897
;   out[7]      s_xor_b32 s39, s9, s9                                      0x00000000   zero
;   out[8]      s_ashr_i32 s40, s5, s11                                    0xfc000000   the count 37 is taken modulo 32
;   out[9]      s_ashr_i32 s41, s7, s11                                    0x00000000   zero
;   out[10]     s_lshl3_add_u32 s42, s5, s18                               0x00000000   zero, with SCC from the bit shifted out
;   out[11]     s_lshl3_add_u32 s43, s7, s8                                0x12345680
;   out[12]     s_lshl4_add_u32 s44, s8, s9                                0x01f3266f   the 64-bit sum passes 2^32
;   out[13]     s_lshl4_add_u32 s45, s7, s7                                0x00000011
;   out[14]     s_abs_i32 s46, s5                                          0x80000000   -2^31 stays
;   out[15]     s_abs_i32 s47, s6                                          0x00000001
;   out[16]     s_abs_i32 s48, s18                                         0x00000000   zero
;   out[17]     s_abs_i32 s26, s8                                          0x12345678   a positive word stays
;   out[18]     s_bcnt0_i32_b32 s49, s9                                    0x00000008
;   out[19]     s_bcnt0_i32_b32 s50, s6                                    0x00000000   zero
;   out[20]     s_addk_i32 s51, 0x1 on a copy of s4                        0x80000000   signed overflow
;   out[21]     s_addk_i32 s52, 0x8000 on a copy of s8                     0x1233d678   the immediate is -32768
;   out[22]     s_addk_i32 s53, 0xffff on a copy of s5                     0x7fffffff   the immediate is -1: signed overflow
;   out[23]     s_mul_hi_u32 s54, s9, s8                                   0x0fd5bdee
;   out[24]     s_pack_ll_b32_b16 s55, s8, s9                              0xbeef5678
;   out[25]     s_pack_hh_b32_b16 s56, s8, s9                              0xdead1234
;   out[26]     s_brev_b32 s57, s8                                         0x1e6a2c48
;   out[27]     s_flbit_i32 s58, s5                                        0x00000001   one bit like the sign
;   out[28]     s_flbit_i32 s59, s6                                        0xffffffff   -1: every bit is the sign
;   out[29]     s_flbit_i32 s60, s7                                        0x0000001f
;   out[30]     s_flbit_i32_b32 s61, s10                                   0x00000010
;   out[31]     s_flbit_i32_b32 s62, s18                                   0xffffffff   -1 for zero
;   out[32]     s_flbit_i32_b32 s27, s5                                    0x00000000   the top bit is set
;   out[33]     s_sext_i32_i8 s63, s10                                     0xffffff80
;   out[34]     s_sext_i32_i8 s64, s8                                      0x00000078
;   out[35]     s_bitset0_b32 s65, s11 on s65 = s6                         0xffffffdf   bit 37 modulo 32
;   out[36]     s_bitset1_b32 s66, s11 on s66 = s18                        0x00000020
;   out[37]     s_mulk_i32 s67, 0xfffd on s67 = s8                         0xc962fc98   times -3
;   out[38]     s_cmovk_i32 s68, 0x8000 on s68 = s8                        0x12345678   SCC 0: keeps
;   out[39]     s_cmovk_i32 s69, 0x8000 on s69 = s8                        0xffff8000   SCC 1: takes -32768
;   out[40,41]  s_cmov_b64 s[70:71], s[14:15] on s70 = s12, s71 = s13      0x00000040 0x00000000   SCC 0: keeps
;   out[42,43]  s_cmov_b64 s[72:73], s[14:15] on s72 = s12, s73 = s13      0x00000001 0x80000001   SCC 1: takes
;   out[44,45]  s_bitset0_b64 s[74:75], s11 on s74 = s16, s75 = s17        0xffff0000 0xffffffdf   bit 37
;   out[46,47]  s_bitset0_b64 s[28:29], s19 on s28 = s6, s29 = s6          0xffffff7f 0xffffffff   bit 71 modulo 64
;   out[48]     s_ff0_i32_b64 s76, s[16:17]                                0x00000000
;   out[49]     s_ff0_i32_b64 s77, s[78:79] on s78 = s6, s79 = s10         0x00000020   the first 0 is in the high dword
;   out[50]     s_ff0_i32_b64 s82, s[80:81] on s80 = s6, s81 = s6          0xffffffff   -1: no 0 bit
;   out[51,52]  s_ashr_i64 s[84:85], s[16:17], s11                         0xffffffff 0xffffffff
;   out[53,54]  s_ashr_i64 s[86:87], s[14:15], s19                         0x02000000 0xff000000   the count 71 is taken modulo 64
;   out[55,56]  s_ashr_i64 s[88:89], s[12:13], s11                         0x00000000 0x00000000   zero
;   out[57,58]  s_lshl_b64 s[90:91], s[14:15], s11                         0x00000000 0x00000020
;   out[59,60]  s_lshl_b64 s[92:93], s[14:15], s19                         0x00000080 0x00000080   the count 71 is taken modulo 64
;   out[61,62]  s_lshl_b64 s[94:95], s[22:23], s11 on s22 = s18, s23 = s5  0x00000000 0x00000000   zero: its one bit leaves
;   out[63,64]  s_lshr_b64 s[96:97], s[16:17], s11                         0x07ffffff 0x00000000
;   out[65,66]  s_lshr_b64 s[98:99], s[12:13], s19                         0x00000000 0x00000000   zero
;   out[67]     s_bcnt0_i32_b64 s24, s[16:17]                              0x00000010
;   out[68]     s_bcnt0_i32_b64 s25, s[80:81]                              0x00000000   zero
;   out[69]     SCC after each of the 23 ops of out[0..22], op n in bit n: 0x0056d555 (set by a signed overflow,
;               by a non-zero result, and by the sum of s_lshlN_add_u32 passing 2^32 in 64 bits)
;   out[70]     SCC of 21 compares and bit tests, in program order from bit 0: 0x000d5555. The eight s_cmp_ forms,
;               each on a pair that makes it true and then on one that makes it false: the signed ones tell
;               0x7fffffff from 0x80000000 and -1 from 1 as signed words, s_cmp_gt_u32 and s_cmp_le_u32 the same
;               words as unsigned ones, and s_cmp_le_u32 is true once more on equal words. Then s_bitcmp0_b32 on
;               bit 37 modulo 32 (set: false) and on bit 1 (clear: true), and s_bitcmp1_b32 on the same two bits.
;   out[71]     SCC after each of the 24 ops of out[23..50], none of which writes it: 0x00555555. It is set to 1
;               before the first, to 0 before the second and so on in turn, by s_cmp_eq_u32 / s_cmp_lg_u32 of s4
;               with itself; s_cmovk_i32 and s_cmov_b64 read it.
;   out[72]     SCC after each of the 10 ops of out[51..68], op n in bit n: 0x0000015b (a non-zero result)
; Words 73..79 of the 80-word heap must keep the harness's 0xeeeeeeee.
; Run: VKC_SGPRS=<the sixteen words> VKC_OUT_WORDS=80 vk_compute_test cs_w32_salu_ext.spv 1 dump, as check.py's
; salu_ext_test does; all 80 words are compared.
s_mov_b32 s100, 0
s_mov_b32 s101, 0
s_mov_b32 s102, 0
s_mov_b32 s103, 0
s_sub_i32 s32, s4, s5
s_cselect_b32 s104, 0x1, 0
s_or_b32 s100, s100, s104
s_sub_i32 s33, s8, s7
s_cselect_b32 s104, 0x2, 0
s_or_b32 s100, s100, s104
s_andn2_b32 s34, s9, s8
s_cselect_b32 s104, 0x4, 0
s_or_b32 s100, s100, s104
s_andn2_b32 s35, s8, s6
s_cselect_b32 s104, 0x8, 0
s_or_b32 s100, s100, s104
s_orn2_b32 s36, s8, s9
s_cselect_b32 s104, 0x10, 0
s_or_b32 s100, s100, s104
s_orn2_b32 s37, s18, s6
s_cselect_b32 s104, 0x20, 0
s_or_b32 s100, s100, s104
s_xor_b32 s38, s8, s9
s_cselect_b32 s104, 0x40, 0
s_or_b32 s100, s100, s104
s_xor_b32 s39, s9, s9
s_cselect_b32 s104, 0x80, 0
s_or_b32 s100, s100, s104
s_ashr_i32 s40, s5, s11
s_cselect_b32 s104, 0x100, 0
s_or_b32 s100, s100, s104
s_ashr_i32 s41, s7, s11
s_cselect_b32 s104, 0x200, 0
s_or_b32 s100, s100, s104
s_lshl3_add_u32 s42, s5, s18
s_cselect_b32 s104, 0x400, 0
s_or_b32 s100, s100, s104
s_lshl3_add_u32 s43, s7, s8
s_cselect_b32 s104, 0x800, 0
s_or_b32 s100, s100, s104
s_lshl4_add_u32 s44, s8, s9
s_cselect_b32 s104, 0x1000, 0
s_or_b32 s100, s100, s104
s_lshl4_add_u32 s45, s7, s7
s_cselect_b32 s104, 0x2000, 0
s_or_b32 s100, s100, s104
s_abs_i32 s46, s5
s_cselect_b32 s104, 0x4000, 0
s_or_b32 s100, s100, s104
s_abs_i32 s47, s6
s_cselect_b32 s104, 0x8000, 0
s_or_b32 s100, s100, s104
s_abs_i32 s48, s18
s_cselect_b32 s104, 0x10000, 0
s_or_b32 s100, s100, s104
s_abs_i32 s26, s8
s_cselect_b32 s104, 0x20000, 0
s_or_b32 s100, s100, s104
s_bcnt0_i32_b32 s49, s9
s_cselect_b32 s104, 0x40000, 0
s_or_b32 s100, s100, s104
s_bcnt0_i32_b32 s50, s6
s_cselect_b32 s104, 0x80000, 0
s_or_b32 s100, s100, s104
s_mov_b32 s51, s4
s_addk_i32 s51, 0x1
s_cselect_b32 s104, 0x100000, 0
s_or_b32 s100, s100, s104
s_mov_b32 s52, s8
s_addk_i32 s52, 0x8000
s_cselect_b32 s104, 0x200000, 0
s_or_b32 s100, s100, s104
s_mov_b32 s53, s5
s_addk_i32 s53, 0xffff
s_cselect_b32 s104, 0x400000, 0
s_or_b32 s100, s100, s104
s_cmp_eq_i32 s5, s5
s_cselect_b32 s104, 0x1, 0
s_or_b32 s101, s101, s104
s_cmp_eq_i32 s4, s5
s_cselect_b32 s104, 0x2, 0
s_or_b32 s101, s101, s104
s_cmp_lg_i32 s4, s5
s_cselect_b32 s104, 0x4, 0
s_or_b32 s101, s101, s104
s_cmp_lg_i32 s6, s6
s_cselect_b32 s104, 0x8, 0
s_or_b32 s101, s101, s104
s_cmp_gt_i32 s4, s5
s_cselect_b32 s104, 0x10, 0
s_or_b32 s101, s101, s104
s_cmp_gt_i32 s6, s7
s_cselect_b32 s104, 0x20, 0
s_or_b32 s101, s101, s104
s_cmp_ge_i32 s5, s5
s_cselect_b32 s104, 0x40, 0
s_or_b32 s101, s101, s104
s_cmp_ge_i32 s6, s7
s_cselect_b32 s104, 0x80, 0
s_or_b32 s101, s101, s104
s_cmp_lt_i32 s6, s7
s_cselect_b32 s104, 0x100, 0
s_or_b32 s101, s101, s104
s_cmp_lt_i32 s4, s5
s_cselect_b32 s104, 0x200, 0
s_or_b32 s101, s101, s104
s_cmp_le_i32 s5, s4
s_cselect_b32 s104, 0x400, 0
s_or_b32 s101, s101, s104
s_cmp_le_i32 s7, s6
s_cselect_b32 s104, 0x800, 0
s_or_b32 s101, s101, s104
s_cmp_gt_u32 s6, s7
s_cselect_b32 s104, 0x1000, 0
s_or_b32 s101, s101, s104
s_cmp_gt_u32 s4, s5
s_cselect_b32 s104, 0x2000, 0
s_or_b32 s101, s101, s104
s_cmp_le_u32 s7, s6
s_cselect_b32 s104, 0x4000, 0
s_or_b32 s101, s101, s104
s_cmp_le_u32 s5, s4
s_cselect_b32 s104, 0x8000, 0
s_or_b32 s101, s101, s104
s_cmp_le_u32 s8, s8
s_cselect_b32 s104, 0x10000, 0
s_or_b32 s101, s101, s104
s_bitcmp0_b32 s8, s11
s_cselect_b32 s104, 0x20000, 0
s_or_b32 s101, s101, s104
s_bitcmp0_b32 s8, s7
s_cselect_b32 s104, 0x40000, 0
s_or_b32 s101, s101, s104
s_bitcmp1_b32 s8, s11
s_cselect_b32 s104, 0x80000, 0
s_or_b32 s101, s101, s104
s_bitcmp1_b32 s8, s7
s_cselect_b32 s104, 0x100000, 0
s_or_b32 s101, s101, s104
s_cmp_eq_u32 s4, s4
s_mul_hi_u32 s54, s9, s8
s_cselect_b32 s104, 0x1, 0
s_or_b32 s102, s102, s104
s_cmp_lg_u32 s4, s4
s_pack_ll_b32_b16 s55, s8, s9
s_cselect_b32 s104, 0x2, 0
s_or_b32 s102, s102, s104
s_cmp_eq_u32 s4, s4
s_pack_hh_b32_b16 s56, s8, s9
s_cselect_b32 s104, 0x4, 0
s_or_b32 s102, s102, s104
s_cmp_lg_u32 s4, s4
s_brev_b32 s57, s8
s_cselect_b32 s104, 0x8, 0
s_or_b32 s102, s102, s104
s_cmp_eq_u32 s4, s4
s_flbit_i32 s58, s5
s_cselect_b32 s104, 0x10, 0
s_or_b32 s102, s102, s104
s_cmp_lg_u32 s4, s4
s_flbit_i32 s59, s6
s_cselect_b32 s104, 0x20, 0
s_or_b32 s102, s102, s104
s_cmp_eq_u32 s4, s4
s_flbit_i32 s60, s7
s_cselect_b32 s104, 0x40, 0
s_or_b32 s102, s102, s104
s_cmp_lg_u32 s4, s4
s_flbit_i32_b32 s61, s10
s_cselect_b32 s104, 0x80, 0
s_or_b32 s102, s102, s104
s_cmp_eq_u32 s4, s4
s_flbit_i32_b32 s62, s18
s_cselect_b32 s104, 0x100, 0
s_or_b32 s102, s102, s104
s_cmp_lg_u32 s4, s4
s_flbit_i32_b32 s27, s5
s_cselect_b32 s104, 0x200, 0
s_or_b32 s102, s102, s104
s_cmp_eq_u32 s4, s4
s_sext_i32_i8 s63, s10
s_cselect_b32 s104, 0x400, 0
s_or_b32 s102, s102, s104
s_cmp_lg_u32 s4, s4
s_sext_i32_i8 s64, s8
s_cselect_b32 s104, 0x800, 0
s_or_b32 s102, s102, s104
s_mov_b32 s65, s6
s_cmp_eq_u32 s4, s4
s_bitset0_b32 s65, s11
s_cselect_b32 s104, 0x1000, 0
s_or_b32 s102, s102, s104
s_mov_b32 s66, s18
s_cmp_lg_u32 s4, s4
s_bitset1_b32 s66, s11
s_cselect_b32 s104, 0x2000, 0
s_or_b32 s102, s102, s104
s_mov_b32 s67, s8
s_cmp_eq_u32 s4, s4
s_mulk_i32 s67, 0xfffd
s_cselect_b32 s104, 0x4000, 0
s_or_b32 s102, s102, s104
s_mov_b32 s68, s8
s_cmp_lg_u32 s4, s4
s_cmovk_i32 s68, 0x8000
s_cselect_b32 s104, 0x8000, 0
s_or_b32 s102, s102, s104
s_mov_b32 s69, s8
s_cmp_eq_u32 s4, s4
s_cmovk_i32 s69, 0x8000
s_cselect_b32 s104, 0x10000, 0
s_or_b32 s102, s102, s104
s_mov_b32 s70, s12
s_mov_b32 s71, s13
s_cmp_lg_u32 s4, s4
s_cmov_b64 s[70:71], s[14:15]
s_cselect_b32 s104, 0x20000, 0
s_or_b32 s102, s102, s104
s_mov_b32 s72, s12
s_mov_b32 s73, s13
s_cmp_eq_u32 s4, s4
s_cmov_b64 s[72:73], s[14:15]
s_cselect_b32 s104, 0x40000, 0
s_or_b32 s102, s102, s104
s_mov_b32 s74, s16
s_mov_b32 s75, s17
s_cmp_lg_u32 s4, s4
s_bitset0_b64 s[74:75], s11
s_cselect_b32 s104, 0x80000, 0
s_or_b32 s102, s102, s104
s_mov_b32 s28, s6
s_mov_b32 s29, s6
s_cmp_eq_u32 s4, s4
s_bitset0_b64 s[28:29], s19
s_cselect_b32 s104, 0x100000, 0
s_or_b32 s102, s102, s104
s_cmp_lg_u32 s4, s4
s_ff0_i32_b64 s76, s[16:17]
s_cselect_b32 s104, 0x200000, 0
s_or_b32 s102, s102, s104
s_mov_b32 s78, s6
s_mov_b32 s79, s10
s_cmp_eq_u32 s4, s4
s_ff0_i32_b64 s77, s[78:79]
s_cselect_b32 s104, 0x400000, 0
s_or_b32 s102, s102, s104
s_mov_b32 s80, s6
s_mov_b32 s81, s6
s_cmp_lg_u32 s4, s4
s_ff0_i32_b64 s82, s[80:81]
s_cselect_b32 s104, 0x800000, 0
s_or_b32 s102, s102, s104
s_ashr_i64 s[84:85], s[16:17], s11
s_cselect_b32 s104, 0x1, 0
s_or_b32 s103, s103, s104
s_ashr_i64 s[86:87], s[14:15], s19
s_cselect_b32 s104, 0x2, 0
s_or_b32 s103, s103, s104
s_ashr_i64 s[88:89], s[12:13], s11
s_cselect_b32 s104, 0x4, 0
s_or_b32 s103, s103, s104
s_lshl_b64 s[90:91], s[14:15], s11
s_cselect_b32 s104, 0x8, 0
s_or_b32 s103, s103, s104
s_lshl_b64 s[92:93], s[14:15], s19
s_cselect_b32 s104, 0x10, 0
s_or_b32 s103, s103, s104
s_mov_b32 s22, s18
s_mov_b32 s23, s5
s_lshl_b64 s[94:95], s[22:23], s11
s_cselect_b32 s104, 0x20, 0
s_or_b32 s103, s103, s104
s_lshr_b64 s[96:97], s[16:17], s11
s_cselect_b32 s104, 0x40, 0
s_or_b32 s103, s103, s104
s_lshr_b64 s[98:99], s[12:13], s19
s_cselect_b32 s104, 0x80, 0
s_or_b32 s103, s103, s104
s_bcnt0_i32_b64 s24, s[16:17]
s_cselect_b32 s104, 0x100, 0
s_or_b32 s103, s103, s104
s_bcnt0_i32_b64 s25, s[80:81]
s_cselect_b32 s104, 0x200, 0
s_or_b32 s103, s103, s104
v_mov_b32_e32 v21, 0
v_mov_b32_e32 v20, s32
buffer_store_dword v20, v21, s[0:3], 0 offen
v_mov_b32_e32 v20, s33
buffer_store_dword v20, v21, s[0:3], 0 offen offset:4
v_mov_b32_e32 v20, s34
buffer_store_dword v20, v21, s[0:3], 0 offen offset:8
v_mov_b32_e32 v20, s35
buffer_store_dword v20, v21, s[0:3], 0 offen offset:12
v_mov_b32_e32 v20, s36
buffer_store_dword v20, v21, s[0:3], 0 offen offset:16
v_mov_b32_e32 v20, s37
buffer_store_dword v20, v21, s[0:3], 0 offen offset:20
v_mov_b32_e32 v20, s38
buffer_store_dword v20, v21, s[0:3], 0 offen offset:24
v_mov_b32_e32 v20, s39
buffer_store_dword v20, v21, s[0:3], 0 offen offset:28
v_mov_b32_e32 v20, s40
buffer_store_dword v20, v21, s[0:3], 0 offen offset:32
v_mov_b32_e32 v20, s41
buffer_store_dword v20, v21, s[0:3], 0 offen offset:36
v_mov_b32_e32 v20, s42
buffer_store_dword v20, v21, s[0:3], 0 offen offset:40
v_mov_b32_e32 v20, s43
buffer_store_dword v20, v21, s[0:3], 0 offen offset:44
v_mov_b32_e32 v20, s44
buffer_store_dword v20, v21, s[0:3], 0 offen offset:48
v_mov_b32_e32 v20, s45
buffer_store_dword v20, v21, s[0:3], 0 offen offset:52
v_mov_b32_e32 v20, s46
buffer_store_dword v20, v21, s[0:3], 0 offen offset:56
v_mov_b32_e32 v20, s47
buffer_store_dword v20, v21, s[0:3], 0 offen offset:60
v_mov_b32_e32 v20, s48
buffer_store_dword v20, v21, s[0:3], 0 offen offset:64
v_mov_b32_e32 v20, s26
buffer_store_dword v20, v21, s[0:3], 0 offen offset:68
v_mov_b32_e32 v20, s49
buffer_store_dword v20, v21, s[0:3], 0 offen offset:72
v_mov_b32_e32 v20, s50
buffer_store_dword v20, v21, s[0:3], 0 offen offset:76
v_mov_b32_e32 v20, s51
buffer_store_dword v20, v21, s[0:3], 0 offen offset:80
v_mov_b32_e32 v20, s52
buffer_store_dword v20, v21, s[0:3], 0 offen offset:84
v_mov_b32_e32 v20, s53
buffer_store_dword v20, v21, s[0:3], 0 offen offset:88
v_mov_b32_e32 v20, s54
buffer_store_dword v20, v21, s[0:3], 0 offen offset:92
v_mov_b32_e32 v20, s55
buffer_store_dword v20, v21, s[0:3], 0 offen offset:96
v_mov_b32_e32 v20, s56
buffer_store_dword v20, v21, s[0:3], 0 offen offset:100
v_mov_b32_e32 v20, s57
buffer_store_dword v20, v21, s[0:3], 0 offen offset:104
v_mov_b32_e32 v20, s58
buffer_store_dword v20, v21, s[0:3], 0 offen offset:108
v_mov_b32_e32 v20, s59
buffer_store_dword v20, v21, s[0:3], 0 offen offset:112
v_mov_b32_e32 v20, s60
buffer_store_dword v20, v21, s[0:3], 0 offen offset:116
v_mov_b32_e32 v20, s61
buffer_store_dword v20, v21, s[0:3], 0 offen offset:120
v_mov_b32_e32 v20, s62
buffer_store_dword v20, v21, s[0:3], 0 offen offset:124
v_mov_b32_e32 v20, s27
buffer_store_dword v20, v21, s[0:3], 0 offen offset:128
v_mov_b32_e32 v20, s63
buffer_store_dword v20, v21, s[0:3], 0 offen offset:132
v_mov_b32_e32 v20, s64
buffer_store_dword v20, v21, s[0:3], 0 offen offset:136
v_mov_b32_e32 v20, s65
buffer_store_dword v20, v21, s[0:3], 0 offen offset:140
v_mov_b32_e32 v20, s66
buffer_store_dword v20, v21, s[0:3], 0 offen offset:144
v_mov_b32_e32 v20, s67
buffer_store_dword v20, v21, s[0:3], 0 offen offset:148
v_mov_b32_e32 v20, s68
buffer_store_dword v20, v21, s[0:3], 0 offen offset:152
v_mov_b32_e32 v20, s69
buffer_store_dword v20, v21, s[0:3], 0 offen offset:156
v_mov_b32_e32 v20, s70
buffer_store_dword v20, v21, s[0:3], 0 offen offset:160
v_mov_b32_e32 v20, s71
buffer_store_dword v20, v21, s[0:3], 0 offen offset:164
v_mov_b32_e32 v20, s72
buffer_store_dword v20, v21, s[0:3], 0 offen offset:168
v_mov_b32_e32 v20, s73
buffer_store_dword v20, v21, s[0:3], 0 offen offset:172
v_mov_b32_e32 v20, s74
buffer_store_dword v20, v21, s[0:3], 0 offen offset:176
v_mov_b32_e32 v20, s75
buffer_store_dword v20, v21, s[0:3], 0 offen offset:180
v_mov_b32_e32 v20, s28
buffer_store_dword v20, v21, s[0:3], 0 offen offset:184
v_mov_b32_e32 v20, s29
buffer_store_dword v20, v21, s[0:3], 0 offen offset:188
v_mov_b32_e32 v20, s76
buffer_store_dword v20, v21, s[0:3], 0 offen offset:192
v_mov_b32_e32 v20, s77
buffer_store_dword v20, v21, s[0:3], 0 offen offset:196
v_mov_b32_e32 v20, s82
buffer_store_dword v20, v21, s[0:3], 0 offen offset:200
v_mov_b32_e32 v20, s84
buffer_store_dword v20, v21, s[0:3], 0 offen offset:204
v_mov_b32_e32 v20, s85
buffer_store_dword v20, v21, s[0:3], 0 offen offset:208
v_mov_b32_e32 v20, s86
buffer_store_dword v20, v21, s[0:3], 0 offen offset:212
v_mov_b32_e32 v20, s87
buffer_store_dword v20, v21, s[0:3], 0 offen offset:216
v_mov_b32_e32 v20, s88
buffer_store_dword v20, v21, s[0:3], 0 offen offset:220
v_mov_b32_e32 v20, s89
buffer_store_dword v20, v21, s[0:3], 0 offen offset:224
v_mov_b32_e32 v20, s90
buffer_store_dword v20, v21, s[0:3], 0 offen offset:228
v_mov_b32_e32 v20, s91
buffer_store_dword v20, v21, s[0:3], 0 offen offset:232
v_mov_b32_e32 v20, s92
buffer_store_dword v20, v21, s[0:3], 0 offen offset:236
v_mov_b32_e32 v20, s93
buffer_store_dword v20, v21, s[0:3], 0 offen offset:240
v_mov_b32_e32 v20, s94
buffer_store_dword v20, v21, s[0:3], 0 offen offset:244
v_mov_b32_e32 v20, s95
buffer_store_dword v20, v21, s[0:3], 0 offen offset:248
v_mov_b32_e32 v20, s96
buffer_store_dword v20, v21, s[0:3], 0 offen offset:252
v_mov_b32_e32 v20, s97
buffer_store_dword v20, v21, s[0:3], 0 offen offset:256
v_mov_b32_e32 v20, s98
buffer_store_dword v20, v21, s[0:3], 0 offen offset:260
v_mov_b32_e32 v20, s99
buffer_store_dword v20, v21, s[0:3], 0 offen offset:264
v_mov_b32_e32 v20, s24
buffer_store_dword v20, v21, s[0:3], 0 offen offset:268
v_mov_b32_e32 v20, s25
buffer_store_dword v20, v21, s[0:3], 0 offen offset:272
v_mov_b32_e32 v20, s100
buffer_store_dword v20, v21, s[0:3], 0 offen offset:276
v_mov_b32_e32 v20, s101
buffer_store_dword v20, v21, s[0:3], 0 offen offset:280
v_mov_b32_e32 v20, s102
buffer_store_dword v20, v21, s[0:3], 0 offen offset:284
v_mov_b32_e32 v20, s103
buffer_store_dword v20, v21, s[0:3], 0 offen offset:288
s_endpgm
