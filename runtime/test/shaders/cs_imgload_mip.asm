; vk_store_test load-mip: x/y are a texel in the 8x8 mip 1. The 16x16 mip 0 has decoy data and receives
; the results. Fetch each mip-1 format twice: full-width and D16 (RG for UNORM, RA for R32 UINT/SINT).
; Store the float RG bits, and check both packed halves (including integer saturation) in the integer outputs.
v_and_b32_e32 v1, 7, v0
v_lshrrev_b32_e32 v2, 3, v0
v_mov_b32_e32 v3, 1
s_mov_b32 s8, 0x1000
s_mov_b32 s9, 0x3800000
s_mov_b32 s10, 0x11
s_mov_b32 s11, 0x11
s_mov_b32 s12, 0x11
s_mov_b32 s13, 0x11
s_mov_b32 s14, 0x11
s_mov_b32 s15, 0x11
image_load_mip v[4:5], v[1:3], s[8:15] dmask:0x3 dim:SQ_RSRC_IMG_2D unorm
image_load_mip v6, v[1:3], s[8:15] dmask:0x3 dim:SQ_RSRC_IMG_2D unorm d16
s_mov_b32 s16, 0x2000
s_mov_b32 s17, 0x1400000
s_mov_b32 s18, 0x22
s_mov_b32 s19, 0x22
s_mov_b32 s20, 0x22
s_mov_b32 s21, 0x22
s_mov_b32 s22, 0x22
s_mov_b32 s23, 0x22
image_load_mip v7, v[1:3], s[16:23] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm
image_load_mip v8, v[1:3], s[16:23] dmask:0x9 dim:SQ_RSRC_IMG_2D unorm d16
s_mov_b32 s24, 0x3000
s_mov_b32 s25, 0x1500000
s_mov_b32 s26, 0x33
s_mov_b32 s27, 0x33
s_mov_b32 s28, 0x33
s_mov_b32 s29, 0x33
s_mov_b32 s30, 0x33
s_mov_b32 s31, 0x33
image_load_mip v9, v[1:3], s[24:31] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm
image_load_mip v10, v[1:3], s[24:31] dmask:0x9 dim:SQ_RSRC_IMG_2D unorm d16
s_waitcnt vmcnt(0)
v_add_nc_u32_e32 v7, v7, v8
v_add_nc_u32_e32 v7, v7, v6
v_add_nc_u32_e32 v9, v9, v10
image_store v[4:5], v[1:2], s[8:15] dmask:0x3 dim:SQ_RSRC_IMG_2D unorm
image_store v7, v[1:2], s[16:23] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm
image_store v9, v[1:2], s[24:31] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm
s_endpgm
