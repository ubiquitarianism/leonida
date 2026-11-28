; s4 is the requested stopping value. Each invocation reports every visit.
; An even increment cannot meet an odd bound, including across u32 wrap.
v_lshlrev_b32_e32 v2, 2, v0
s_mov_b32 s8, 0
again:
s_add_u32 s8, s8, 2
v_mov_b32_e32 v1, s8
buffer_store_dword v1, v2, s[0:3], 0 offen
s_cmp_eq_u32 s8, s4
s_cbranch_scc0 again
s_endpgm
