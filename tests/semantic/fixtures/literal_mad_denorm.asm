; Synthetic literal MAD denormal boundaries (no game data).
; Expected mrt0 words: 00000000 00000000 00000000 00000000.
; Respectively: subnormal input, addend, product, and final sum.
v_mov_b32 v1, 0x00000001
v_mov_b32 v2, 0x7e800000
v_madak_f32 v4, v1, v2, 0x00000000
v_mov_b32 v1, 0x00000000
v_mov_b32 v2, 0x00000001
v_madmk_f32 v5, v1, 0x3f800000, v2
v_mov_b32 v1, 0x00800000
v_mov_b32 v2, 0x3f000000
v_madak_f32 v6, v1, v2, 0x00000000
v_mov_b32 v1, 0x00800001
v_mov_b32 v2, 0x80800000
v_madmk_f32 v7, v1, 0x3f800000, v2
exp mrt0 v4, v5, v6, v7 done vm
s_endpgm
