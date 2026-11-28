; Dual-source blending (RT-DUAL-SOURCE-BLEND): ps_m15.asm with a second colour export. mrt0 is ps_m15's colour;
; mrt1 is the constant 0.5 in every channel, the second blend source (SV_Target1). d3d12_translated_draw_test
; --dual-source blends with ONE and SRC1_COLOR / SRC1_ALPHA over the clear colour (0.5, 0.125, 0.25, 0.5), so each
; covered pixel is ps_m15's colour + (0.25, 0.0625, 0.125, 0.25), exact in a float target. The translator declares
; mrt1 at Location 1; the layer moves it to Location 0, Index 1 for the SRC1 factors.
; fssm-at: 0x100
; The fifth word is unused by the code; it gives this program an FSSM of its own (the layer finds a PSO's shader by its
; FSSM hash, and ps_m15's would be the same otherwise).
; fssm-data: 0x3f000000 0x40000000 0x3f800000 0x3f800000 0x00000002
s_getpc_b64 s[6:7]
s_add_u32 s6, s6, 0xfc
s_addc_u32 s7, s7, 0
s_mov_b32 s4, s11
s_load_dwordx4 s[8:11], s[6:7], 0x0
v_interp_p1_f32_e32 v2, v0, attr0.x
v_interp_p2_f32_e32 v2, v1, attr0.x
v_interp_p1_f32_e32 v3, v0, attr0.y
v_interp_p2_f32_e32 v3, v1, attr0.y
v_interp_p1_f32_e32 v4, v0, attr0.z
v_interp_p2_f32_e32 v4, v1, attr0.z
v_interp_p1_f32_e32 v5, v0, attr0.w
v_interp_p2_f32_e32 v5, v1, attr0.w
s_waitcnt lgkmcnt(0)
v_fma_f32 v2, v2, s8, s4
v_fma_f32 v3, v3, s9, s4
v_fma_f32 v4, v4, s10, s4
v_mul_f32_e32 v5, s11, v5
v_mov_b32_e32 v6, 0.5
exp mrt1 v6, v6, v6, v6 vm
exp mrt0 v2, v3, v4, v5 done vm
s_endpgm
