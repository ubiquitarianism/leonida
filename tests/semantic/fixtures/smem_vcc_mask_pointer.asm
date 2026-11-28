; Lane 12 compares equal. Its carry bit makes the low address 4096 on wave32.
s_mov_b32 vcc_hi, 0
v_cmp_eq_u32_e32 vcc_lo, 12, v0
s_load_dword s8, vcc, 0
s_waitcnt lgkmcnt(0)
v_mov_b32_e32 v6, 0x3000
v_mov_b32_e32 v7, s8
global_store_dword v6, v7, s[0:1]
s_endpgm
