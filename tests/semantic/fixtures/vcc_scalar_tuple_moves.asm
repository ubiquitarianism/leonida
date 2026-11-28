; Build a buffer resource in SGPRs 104..107. Read both scalar words back after a tuple load.
s_mov_b32 s104, 0x4000
s_mov_b32 s105, 0
s_mov_b32 s8, 8
s_mov_b32 s9, 0x31016fac
s_mov_b64 vcc, s[8:9]
s_buffer_load_dwordx2 s[10:11], [s104,s105,vcc_lo,vcc_hi], 0
s_waitcnt lgkmcnt(0)
s_mov_b64 vcc, s[10:11]
s_mov_b64 s[12:13], vcc
s_add_u32 s14, vcc_hi, 3
s_buffer_load_dwordx4 [s104,s105,vcc_lo,vcc_hi], s[0:3], 0x4000
s_waitcnt lgkmcnt(0)
s_mov_b64 s[20:21], vcc
v_mov_b32_e32 v4, s12
v_mov_b32_e32 v5, s13
v_mov_b32_e32 v6, s14
v_mov_b32_e32 v7, s21
v_lshlrev_b32_e32 v1, 4, v0
buffer_store_dwordx4 v[4:7], v1, s[0:3], 0 offen
s_endpgm
