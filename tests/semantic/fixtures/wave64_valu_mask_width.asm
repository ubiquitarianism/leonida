; Hand-written wave32-style predicates in a wave64 listing. s9 starts as all ones.
s_mov_b32 s9, -1
v_cmp_lt_u32_e64 s8, v0, 36
v_cndmask_b32_e64 v1, 0, 1, s[8:9]
v_add_co_u32_e64 v2, s10, -33, v0
v_add_co_ci_u32_e64 v2, vcc_lo, 0, 0, s[10:11]
s_mov_b32 s14, 0
s_mov_b32 s15, 16
v_cndmask_b32_e64 v3, 0, 7, s14
v_lshlrev_b32_e32 v4, 4, v0
buffer_store_dwordx3 v[1:3], v4, s[0:3], 0 offen
s_endpgm
