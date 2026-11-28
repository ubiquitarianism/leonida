; SHADER-1404: image_sample, image_sample_l and image_sample_lz of a UINT texture outside 2D.
; Run: vk_texture_test volume-uint -> "volume-uint: 4096 exact pixels".
; The texture is a 128x128x4 volume with a 64x64x2 mip 1, bound at binding 6 index 1
; as R8G8B8A8_UINT, the format the T# names.
; Texel (x, y) of slice l at mip m (side S) holds bytes (32 * (8x / S), 32 * (8y / S), 20l + m, 255 - 2m), so a
; pixel reads the same 8x8-pixel block at both mips. image_sample reads w = 0.625 at mip 1
; (its implicit LOD is 1: slice 1), image_sample_l w = 0.375 at LOD 0 (slice 1), image_sample_lz w = 0.875 (slice 3).
; The first two are d16: the integer view saturates each channel to 16 bits, while the float view (which the
; implicit form read outside 2D before SHADER-1404) narrows the channel's bits as a float, to 0 here.
; A wrong result spills into the next byte of the packed word.
; mrt0.xyz = R | G << 8 | B << 16 of the three samples, mrt0.w = their alphas packed the same way, each
; converted to float (exact below 2^24).
v_interp_p1_f32_e32 v2, v0, attr0.x
v_interp_p2_f32_e32 v2, v1, attr0.x
v_interp_p1_f32_e32 v3, v0, attr0.y
v_interp_p2_f32_e32 v3, v1, attr0.y
s_load_dwordx8 s[8:15], s[0:1], 0x0
s_mov_b32 s16, 0x92
s_mov_b32 s17, 0xc00000
s_mov_b32 s18, 0x14500000
s_mov_b32 s19, 0
v_mov_b32_e32 v4, v2
v_mov_b32_e32 v5, v3
v_mov_b32_e32 v6, 0x3f200000
v_mov_b32_e32 v18, v2
v_mov_b32_e32 v19, v3
v_mov_b32_e32 v20, 0x3ec00000
v_mov_b32_e32 v21, 0
v_mov_b32_e32 v26, v2
v_mov_b32_e32 v27, v3
v_mov_b32_e32 v28, 0x3f600000
s_waitcnt lgkmcnt(0)
image_sample v[10:11], v[4:6], s[8:15], s[16:19] dmask:0xf dim:SQ_RSRC_IMG_3D d16
image_sample_l v[14:15], v[18:21], s[8:15], s[16:19] dmask:0xf dim:SQ_RSRC_IMG_3D d16
image_sample_lz v[22:25], v[26:28], s[8:15], s[16:19] dmask:0xf dim:SQ_RSRC_IMG_3D
s_waitcnt vmcnt(0)
v_bfe_u32 v13, v11, 16, 16
v_bfe_u32 v12, v11, 0, 16
v_bfe_u32 v11, v10, 16, 16
v_bfe_u32 v10, v10, 0, 16
v_bfe_u32 v17, v15, 16, 16
v_bfe_u32 v16, v15, 0, 16
v_bfe_u32 v15, v14, 16, 16
v_bfe_u32 v14, v14, 0, 16
v_lshl_or_b32 v30, v11, 8, v10
v_lshl_or_b32 v30, v12, 16, v30
v_lshl_or_b32 v31, v15, 8, v14
v_lshl_or_b32 v31, v16, 16, v31
v_lshl_or_b32 v32, v23, 8, v22
v_lshl_or_b32 v32, v24, 16, v32
v_lshl_or_b32 v33, v17, 8, v13
v_lshl_or_b32 v33, v25, 16, v33
v_cvt_f32_u32_e32 v30, v30
v_cvt_f32_u32_e32 v31, v31
v_cvt_f32_u32_e32 v32, v32
v_cvt_f32_u32_e32 v33, v33
exp mrt0 v30, v31, v32, v33 done vm
s_endpgm
