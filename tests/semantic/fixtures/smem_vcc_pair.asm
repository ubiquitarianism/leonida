; A wave32 carries a >4 GiB scalar pointer in VCC. Copy it both ways and XOR both halves.
s_mov_b32 s4, 0x1040
s_mov_b32 s5, 1
s_mov_b64 vcc, s[4:5]
s_mov_b64 s[14:15], vcc
v_cndmask_b32_e32 v7, 0, 1, vcc_lo
s_mov_b32 s6, 0x40
s_mov_b32 s7, 2
s_xor_b64 vcc, s[6:7], vcc
s_load_dword s8, vcc, 0
s_waitcnt lgkmcnt(0)
v_mov_b32_e32 v4, s14
v_mov_b32_e32 v5, s15
v_mov_b32_e32 v6, s8
v_lshlrev_b32_e32 v1, 4, v0
v_add_nc_u32_e32 v1, 0x3000, v1
global_store_dwordx4 v1, v[4:7], s[0:1]
s_endpgm
