; Synthetic descriptor table. Each group of eight lanes uses one of four V#s.
; s[0:3] describes the output. s[4:5] points to four input descriptors.
v_lshrrev_b32_e32 v1, 3, v0
v_and_b32_e32 v1, 3, v1
v_lshlrev_b32_e32 v1, 4, v1
v_lshlrev_b32_e32 v2, 2, v0
s_mov_b32 s12, exec_lo
s_mov_b32 s13, exec_lo
again:
v_readfirstlane_b32 s14, v1
v_cmpx_eq_u32_e32 s14, v1
s_load_dwordx4 s[16:19], s[4:5], s14
s_waitcnt lgkmcnt(0)
buffer_load_dword v3, off, s[16:19], 0
s_waitcnt vmcnt(0)
s_andn2_wrexec_b32 s13, s13
s_cbranch_execnz again
s_mov_b32 exec_lo, s12
buffer_store_dword v3, v2, s[0:3], 0 offen
s_endpgm
