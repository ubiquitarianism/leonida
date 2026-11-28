; One wave64 workgroup: a uniform loop and a uniform branch, each with a read of another wave32 half's lane, which
; the emulation exchanges through %x_buf behind a workgroup barrier. s4 is a push constant: 0 skips the branch.
v_lshlrev_b32_e32 v2, 2, v0
v_mov_b32_e32 v1, v0
s_mov_b32 s8, 0
s_mov_b32 s9, 0
again:
v_add_nc_u32_e32 v1, 1, v1
v_readlane_b32 s10, v1, 40
s_add_u32 s9, s9, s10
s_add_u32 s8, s8, 1
s_cmp_eq_u32 s8, 3
s_cbranch_scc0 again
s_cmp_eq_u32 s4, 0
s_cbranch_scc1 done
v_add_nc_u32_e32 v1, 100, v1
v_readlane_b32 s11, v1, 41
s_add_u32 s9, s9, s11
done:
v_mov_b32_e32 v3, s9
buffer_store_dword v3, v2, s[0:3], 0 offen
s_endpgm
