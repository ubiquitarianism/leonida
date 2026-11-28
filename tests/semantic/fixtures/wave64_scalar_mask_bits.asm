; Read EXEC with bits in either half and when empty. Then replicate scalar bits into VCC.
s_cmp_eq_u32 1, 1
s_mov_b64 exec, 0
s_flbit_i32_b64 s6, exec
s_mov_b32 exec_lo, 0x200
s_flbit_i32_b64 s7, exec
s_mov_b32 exec_hi, 0x100
s_flbit_i32_b64 s8, exec
s_mov_b64 exec, -1
s_bitreplicate_b64_b32 vcc, 0x80000021
s_cselect_b32 s9, 2, 0
v_cndmask_b32 v4, 0, 1, vcc
v_add_nc_u32 v4, s9, v4
v_mov_b32 v1, s6
v_mov_b32 v2, s7
v_mov_b32 v3, s8
v_lshlrev_b32 v5, 4, v0
buffer_store_dwordx4 v[1:4], v5, s[0:3], 0 offen
s_endpgm

