; GPU test for ds_wrxchg2st64_rtn_b32 (docs/design.md "ds_wrxchg2_rtn_b32, ds_wrxchg2st64_rtn_b32"
; row). Neither this form nor the row's st64 step is covered by a GPU fixture on main: cs_w32_new_ops
; runs ds_wrxchg2_rtn_b32 only, and cs_lds_xchg64_ext runs the _b64 pair forms. The row and the gfx1100
; measurement both say the st64 unit is 64 dwords = 256 bytes, i.e. offset0:1 and offset1:2 address
; vaddr + 256 and vaddr + 512; this fixture pins that on a GPU.
;
; LDS 4 KiB (RSRC2 0x40088), group 32x1x1, only lane 0 runs. Base 0x80, so the two elements are at
; 0x180 and 0x280 and neither is where a non-st64 unit would have put them (0x84 and 0x88) - those two
; words are seeded too and must come back untouched, which is what proves the step size.
; Every word is stored raw, in this order:
;
;   out[0]  vdst[0] = the old word at 0x180                                  -> 0x33333333
;   out[1]  vdst[1] = the old word at 0x280                                  -> 0x44444444
;   out[2]  M[0x180] after: data0                                            -> 0x55550005
;   out[3]  M[0x280] after: data1                                            -> 0x66660006
;   out[4]  M[0x84], where a 4-byte step would have written data0: untouched -> 0xB0B00084
;   out[5]  M[0x88], where a 4-byte step would have written data1: untouched -> 0xB0B00088
;   out[6]  M[0x140], one 256-byte step short of 0x180: untouched            -> 0xB0B00140
;   out[7]  M[0x2c0], one 256-byte step past 0x280: untouched                -> 0xB0B002C0
;
; Run: vk_compute_test cs_w32_wrxchg2st64.spv 1 dump
;   -> out[0..7] = 0x33333333 0x44444444 0x55550005 0x66660006
;                  0xB0B00084 0xB0B00088 0xB0B00140 0xB0B002C0
; rsrc2: 0x40088
; group: 32,1,1
; mattr: +wavefrontsize32,-wavefrontsize64
v_cmpx_eq_u32_e32 0, v0

; --- seed the two real elements and the four words the exchange must not touch ---
v_mov_b32_e32 v1, 0x180
v_mov_b32_e32 v4, 0x33333333
ds_write_b32 v1, v4
v_mov_b32_e32 v1, 0x280
v_mov_b32_e32 v4, 0x44444444
ds_write_b32 v1, v4
v_mov_b32_e32 v1, 0x84
v_mov_b32_e32 v4, 0xB0B00084
ds_write_b32 v1, v4
v_mov_b32_e32 v1, 0x88
v_mov_b32_e32 v4, 0xB0B00088
ds_write_b32 v1, v4
v_mov_b32_e32 v1, 0x140
v_mov_b32_e32 v4, 0xB0B00140
ds_write_b32 v1, v4
v_mov_b32_e32 v1, 0x2C0
v_mov_b32_e32 v4, 0xB0B002C0
ds_write_b32 v1, v4
s_waitcnt lgkmcnt(0)

; --- the st64 pair exchange: base 0x80, offset0:1 offset1:2 ---
v_mov_b32_e32 v4, 0x55550005
v_mov_b32_e32 v5, 0x66660006
v_mov_b32_e32 v1, 0x80
ds_wrxchg2st64_rtn_b32 v[12:13], v1, v4, v5 offset0:1 offset1:2
s_waitcnt lgkmcnt(0)

v_mov_b32_e32 v21, 0
buffer_store_dword v12, v21, s[0:3], 0 offen
v_mov_b32_e32 v21, 4
buffer_store_dword v13, v21, s[0:3], 0 offen

; --- read back the two elements and the four words that must be untouched ---
v_mov_b32_e32 v1, 0x180
ds_read_b32 v10, v1
v_mov_b32_e32 v2, 0x280
ds_read_b32 v11, v2
s_waitcnt lgkmcnt(0)
v_mov_b32_e32 v21, 8
buffer_store_dword v10, v21, s[0:3], 0 offen
v_mov_b32_e32 v21, 12
buffer_store_dword v11, v21, s[0:3], 0 offen

v_mov_b32_e32 v1, 0x84
ds_read_b32 v3, v1
v_mov_b32_e32 v1, 0x88
ds_read_b32 v6, v1
s_waitcnt lgkmcnt(0)
v_mov_b32_e32 v21, 16
buffer_store_dword v3, v21, s[0:3], 0 offen
v_mov_b32_e32 v21, 20
buffer_store_dword v6, v21, s[0:3], 0 offen

v_mov_b32_e32 v1, 0x140
ds_read_b32 v3, v1
v_mov_b32_e32 v1, 0x2C0
ds_read_b32 v6, v1
s_waitcnt lgkmcnt(0)
v_mov_b32_e32 v21, 24
buffer_store_dword v3, v21, s[0:3], 0 offen
v_mov_b32_e32 v21, 28
buffer_store_dword v6, v21, s[0:3], 0 offen

s_endpgm
