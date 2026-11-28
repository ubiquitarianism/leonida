; group: 1,1,1
; rsrc2: 0x392
; s[0:7] = UAV T#, s8 = case (0: 1D, 1: 1D array, 2: 3D UINT, 3: 2D, 4: 3D FLOAT, 5: 1D array FLOAT, 6: 3D atomics).
; s[9:11] = workgroup XYZ. Add 1000 + x + 16*y + 256*z to each existing texel (z is the array
; layer for 1D arrays). d3d12_texture_uav_test pushes production heap descriptors and compares
; every texel against the CPU expectation and, for non-atomic UINT, against Windows WARP.
; The float volume adds the texel twice at even x (LOD 0), once at odd x (LOD 1 reads zero).
; The atomic volume stores the coordinate increment + (initial << 16) + ((initial + 5) << 8) + 7 + 99.
v_mov_b32_e32 v0, s9
v_mov_b32_e32 v1, s10
v_mov_b32_e32 v2, s11
v_lshlrev_b32_e32 v3, 8, v2
v_lshl_add_u32 v3, v1, 4, v3
v_add_nc_u32_e32 v3, v0, v3
v_add_nc_u32_e32 v3, 1000, v3
s_cmp_eq_u32 s8, 0
s_cbranch_scc0 11
image_load v4, v0, s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_1D unorm
; The sampled and formatless coherent reads must agree; a mismatch changes the result.
image_load v5, v0, s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_1D unorm glc dlc
s_waitcnt vmcnt(0)
v_xor_b32_e32 v5, v4, v5
v_add_nc_u32_e32 v3, v3, v5
v_add_nc_u32_e32 v3, v3, v4
image_store v3, v0, s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_1D unorm
s_endpgm
s_cmp_eq_u32 s8, 1
s_cbranch_scc0 7
v_mov_b32_e32 v1, v2
image_load v4, v[0:1], s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_1D_ARRAY unorm
v_add_nc_u32_e32 v3, v3, v4
image_store v3, v[0:1], s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_1D_ARRAY unorm
s_endpgm
s_cmp_eq_u32 s8, 2
s_cbranch_scc0 11
image_load v4, v[0:2], s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_3D unorm
; A sliced UAV must see the same texel via the rebased sampled alias and the sliced storage view.
image_load v5, v[0:2], s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_3D unorm glc dlc
s_waitcnt vmcnt(0)
v_xor_b32_e32 v5, v4, v5
v_add_nc_u32_e32 v3, v3, v5
v_add_nc_u32_e32 v3, v3, v4
image_store v3, v[0:2], s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_3D unorm
s_endpgm
s_cmp_eq_u32 s8, 3
s_cbranch_scc0 6
image_load v4, v[0:1], s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm
v_add_nc_u32_e32 v3, v3, v4
image_store v3, v[0:1], s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm
s_endpgm
s_cmp_eq_u32 s8, 5
s_cbranch_scc0 13
v_mov_b32_e32 v1, v2
image_load v4, v[0:1], s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_1D_ARRAY unorm
v_cvt_f32_u32_e32 v3, v3
v_add_f32_e32 v3, v3, v4
; Float array view covers two layers. Verify that the next-layer read does not contribute.
v_add_nc_u32_e32 v1, 2, v1
image_load v4, v[0:1], s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_1D_ARRAY unorm
v_add_f32_e32 v3, v3, v4
v_mov_b32_e32 v1, v2
image_store v3, v[0:1], s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_1D_ARRAY unorm
s_endpgm
s_cmp_eq_u32 s8, 6
s_cbranch_scc0 31
; Use a separate T# SGPR group so the non-atomic UINT volume case exercises the sampled alias.
s_mov_b64 s[12:13], s[0:1]
s_mov_b64 s[14:15], s[2:3]
s_mov_b64 s[16:17], s[4:5]
s_mov_b64 s[18:19], s[6:7]
; Inactive add and CAS must neither write nor change their glc destinations. The active add returns
; the initial texel, the active CAS returns that texel + 5, and the following load reads 99.
v_mov_b32_e32 v4, 7
v_mov_b32_e32 v5, 0
s_mov_b32 exec_lo, 0
image_atomic_add v4, v[0:2], s[12:19] dmask:0x1 dim:SQ_RSRC_IMG_3D unorm glc
image_atomic_cmpswap v[4:5], v[0:2], s[12:19] dmask:0x3 dim:SQ_RSRC_IMG_3D unorm glc
s_mov_b32 exec_lo, 1
v_mov_b32_e32 v6, 5
image_atomic_add v6, v[0:2], s[12:19] dmask:0x1 dim:SQ_RSRC_IMG_3D unorm glc
v_mov_b32_e32 v8, 99
v_add_nc_u32_e32 v9, 5, v6
image_atomic_cmpswap v[8:9], v[0:2], s[12:19] dmask:0x3 dim:SQ_RSRC_IMG_3D unorm glc
image_load v7, v[0:2], s[12:19] dmask:0x1 dim:SQ_RSRC_IMG_3D unorm
v_lshl_add_u32 v3, v6, 16, v3
v_lshl_add_u32 v3, v8, 8, v3
v_add_nc_u32_e32 v3, v4, v3
v_add_nc_u32_e32 v3, v7, v3
image_store v3, v[0:2], s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_3D unorm
s_endpgm
image_load v4, v[0:2], s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_3D unorm
v_cvt_f32_u32_e32 v3, v3
v_add_f32_e32 v3, v3, v4
; Relative LOD 0 reuses the physical mip selected by the view; LOD 1 is outside the single-mip view.
v_mov_b32_e32 v6, v3
v_and_b32_e32 v3, 1, v0
image_load_mip v5, v[0:3], s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_3D unorm
v_add_f32_e32 v3, v6, v5
; Float test uses WSize=2: an outside-Z fetch must be zero even if still inside the image.
v_add_nc_u32_e32 v2, 2, v2
image_load v4, v[0:2], s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_3D unorm
v_add_f32_e32 v3, v3, v4
v_mov_b32_e32 v2, s11
image_store v3, v[0:2], s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_3D unorm
s_endpgm
