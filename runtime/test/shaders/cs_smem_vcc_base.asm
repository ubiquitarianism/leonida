; group: 1,1,1
; Seed eight consecutive words at byte 128. Read them back with VCC as an ordinary scalar pointer pair.
; Expected out[0..7]: 0x7210 0x7211 0x7212 0x7213 0x7214 0x7215 0x7216 0x7217.
v_mov_b32_e32 v1, 0x7210
v_mov_b32_e32 v2, 0x7211
v_mov_b32_e32 v3, 0x7212
v_mov_b32_e32 v4, 0x7213
buffer_store_dwordx4 v[1:4], off, s[0:3], 0 offset:128
v_mov_b32_e32 v1, 0x7214
v_mov_b32_e32 v2, 0x7215
v_mov_b32_e32 v3, 0x7216
v_mov_b32_e32 v4, 0x7217
buffer_store_dwordx4 v[1:4], off, s[0:3], 0 offset:144
s_waitcnt vmcnt(0)
s_mov_b32 s4, 128
s_mov_b32 s5, 0
s_mov_b64 vcc, s[4:5]
s_load_dwordx8 s[8:15], vcc, 0
s_waitcnt lgkmcnt(0)
v_mov_b32 v0, 0
v_mov_b32 v1, s8
v_mov_b32 v2, s9
v_mov_b32 v3, s10
v_mov_b32 v4, s11
buffer_store_dwordx4 v[1:4], v0, s[0:3], 0 offen
v_mov_b32 v0, 16
v_mov_b32 v1, s12
v_mov_b32 v2, s13
v_mov_b32 v3, s14
v_mov_b32 v4, s15
buffer_store_dwordx4 v[1:4], v0, s[0:3], 0 offen
s_endpgm
