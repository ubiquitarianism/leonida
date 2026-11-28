; Each lane compares a permuted neighbour's float against its own lane index.
v_cvt_f32_u32 v2, v0
v_cmp_gt_f32_dpp vcc, v2, v2 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1
v_cndmask_b32 v3, 0, 1, vcc
v_cmp_gt_f32_dpp vcc, v2, v2 dpp8:[7,6,5,4,3,2,1,0]
v_cndmask_b32 v4, 0, 1, vcc
v_lshlrev_b32 v5, 3, v0
buffer_store_dwordx2 v[3:4], v5, s[0:3], 0 offen
s_endpgm

