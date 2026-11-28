; M2 sampler control: ps_m2.asm with 1.0 added to U immediately before image_sample. With U in range a CLAMP, a WRAP
; and the fallback sampler give the same pixels; with U + 1 the CLAMP sampler returns the last column and the WRAP one
; the same texel as without the offset, so the pixels tell which sampler the draw used. Idea and measurements by
; @Vonsek (#1489). Run: d3d12_m2_textured_test.
; The same unbounded SRV table stays bound across both draws. User data as in ps_m2.asm (the SRV table at s8, the
; sampler table at s9), then root constants s10 and s11 hold the descriptor index and byte stride, respectively; the
; first six instructions move them to s[0:1], s[2:3] (high dwords 0), s4 and s5.
; fssm-at: 0x10
; fssm-data: 0x324d5350 0x4f465520
s_mov_b32 s0, s8
s_mov_b32 s1, 0
s_mov_b32 s2, s9
s_mov_b32 s3, 0
s_mov_b32 s4, s10
s_mov_b32 s5, s11
s_mul_i32 s20, s4, s5
s_load_dwordx8 s[8:15], s[0:1], s20
s_load_dwordx4 s[16:19], s[2:3], 0x0
s_waitcnt lgkmcnt(0)
v_interp_p1_f32_e32 v2, v0, attr0.x
v_interp_p2_f32_e32 v2, v1, attr0.x
v_interp_p1_f32_e32 v3, v0, attr0.y
v_interp_p2_f32_e32 v3, v1, attr0.y
v_add_f32_e32 v2, 1.0, v2
image_sample v[4:7], v[2:3], s[8:15], s[16:19] dmask:0xf dim:SQ_RSRC_IMG_2D
s_waitcnt vmcnt(0)
exp mrt0 v4, v5, v6, v7 done vm
s_endpgm
