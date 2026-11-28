; VCC contains a scalar byte address. An unbounded SMEM load reads its two words.
s_mov_b32 vcc_lo, 0x1000
s_mov_b32 vcc_hi, 0
s_load_dwordx2 s[8:9], vcc, 4
s_waitcnt lgkmcnt(0)
v_mov_b32_e32 v0, 0x3000
v_mov_b32_e32 v1, s8
v_mov_b32_e32 v2, s9
global_store_dwordx2 v0, v[1:2], s[0:1]
s_endpgm
