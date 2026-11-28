; Vertex shader for the indexed quad-list test: vs_m15's vertex fetch through a V#, plus the instance id. Run:
; d3d12_quad_index_test (through a D3D12 graphics PSO found by the SHA-256 of its FSSM program), with ps_quad_flat.
;
; User data, from the root signature [0] descriptor table, [1] four 32-bit constants (docs/d3d12_vulkan.md 3.5):
;   s[0:1] = console VA of the table's first slot: the 16 bytes of a V# (base VA, stride 32)
;   s2, s3 = the clip-space offset (x, y) of each instance after the first, as floats
; v0 = the vertex id and v3 = the instance id, both zero-based as on the hardware (vs_base_ids.asm): in an indexed
; draw v0 is the index the application wrote, without the draw's base vertex.
; Vertex record, 32 bytes: position xyzw at 0, colour rgba at 16 (both float).
;   pos0   = (position.x + instance * s2, position.y + instance * s3, position.z, position.w)
;   param0 = colour for instance 0, colour.grba (red and green swapped) for every later instance
; The test draws red quads with two instances: instance 0 stays red, instance 1 is green and one offset further on.
; fssm-at: 0x10
; fssm-data: 0x44415551 0x20535620
s_load_dwordx4 s[8:11], s[0:1], 0x0
s_waitcnt lgkmcnt(0)
buffer_load_dwordx4 v[4:7], v0, s[8:11], 0 idxen
buffer_load_dwordx4 v[8:11], v0, s[8:11], 0 idxen offset:16
s_waitcnt vmcnt(0)
v_cvt_f32_u32_e32 v12, v3
v_fma_f32 v4, v12, s2, v4
v_fma_f32 v5, v12, s3, v5
v_cmp_ne_u32_e32 vcc_lo, 0, v3
v_cndmask_b32_e32 v13, v8, v9, vcc_lo
v_cndmask_b32_e32 v14, v9, v8, vcc_lo
exp pos0 v4, v5, v6, v7 done
exp param0 v13, v14, v10, v11
s_endpgm
