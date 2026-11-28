; Pixel shader for the indexed quad-list test: param0 (a colour) read only with v_interp_mov, so the input is Flat
; and every pixel of a primitive gets the provoking vertex's colour, unscaled. Run: d3d12_quad_index_test, with
; vs_quad_inst.
;   mrt0 = param0 of the primitive's provoking vertex
; A quad list takes it from the quad's first vertex. The test's quads have a different colour at each corner, so
; a quad drawn from any other vertex shows that vertex's colour instead; the colours are 0 and 1, exact in the float
; render target.
; fssm-at: 0x10
; fssm-data: 0x44415551 0x20535020
v_interp_mov_f32_e32 v2, p0, attr0.x
v_interp_mov_f32_e32 v3, p0, attr0.y
v_interp_mov_f32_e32 v4, p0, attr0.z
v_interp_mov_f32_e32 v5, p0, attr0.w
exp mrt0 v2, v3, v4, v5 done vm
s_endpgm
