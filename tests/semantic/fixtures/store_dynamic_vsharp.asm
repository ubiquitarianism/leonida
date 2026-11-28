; Store four distinct f32 words through the V# supplied by the caller.
v_mov_b32_e32 v4, 1.0
v_mov_b32_e32 v5, 2.0
v_mov_b32_e32 v6, 0x40400000
v_mov_b32_e32 v7, 4.0
buffer_store_format_xyzw v[4:7], off, s[0:3], 0
s_endpgm
