; GPU test for ds_wrxchg2_rtn_b32, part two: the four cases cs_w32_new_ops.asm leaves out
; (docs/design.md "ds_wrxchg2_rtn_b32, ds_wrxchg2st64_rtn_b32" row). cs_w32_new_ops covers the first
; returned word and data1 in LDS with offset0:2 offset1:5; this fixture covers:
;
;   - the SECOND returned word of the pair, which is data0 when both offsets name one dword
;   - both offsets on one dword (the second exchange reads back what the first wrote)
;   - vdst aliasing the data VGPRs (the data must be read before vdst is written)
;   - that an exchange at one base leaves a neighbouring LDS word alone
;
; LDS 4 KiB (RSRC2 0x40088), group 32x1x1, only lane 0 runs (32 lanes on one LDS address would race:
; which lane's exchange runs first is up to the hardware). LDS is seeded by ds_write_b32 before each case.
; Every word is stored raw, in this order:
;
;   out[0]  both offsets on one dword: ds_wrxchg2_rtn_b32 at base 0x200, offset0:1 offset1:1
;           (both = element 0x204). Seeded M[0x204]=0x77777777, data0=0x88880008, data1=0x99990009.
;           vdst[0] = the word the first exchange took out of M                 -> 0x77777777
;   out[1]  vdst[1] = the second exchange's return, which is data0, not the
;           word the first exchange wrote                                     -> 0x88880008
;   out[2]  M[0x204] after both: data1 landed                                 -> 0x99990009
;   out[3]  M[0x700], seeded 0xDEADBEEF before the exchange and read after it:
;           the exchange touched only 0x204                                -> 0xDEADBEEF
;   out[4]  vdst aliases the data VGPRs: ds_wrxchg2_rtn_b32 v[4:5], ... with v4=0x12341234 v5=0x56785678
;           at base 0x300, offset0:0 offset1:2, M[0x300]=0xAAAAAAAA, M[0x308]=0xBBBBBBBB.
;           vdst[0] = the old word, so the data was read before the write      -> 0xAAAAAAAA
;   out[5]  vdst[1] = the old word at 0x308                                    -> 0xBBBBBBBB
;   out[6]  M[0x300] after: data0                                             -> 0x12341234
;   out[7]  M[0x308] after: data1                                             -> 0x56785678
;
; Run: vk_compute_test cs_w32_wrxchg2_ext.spv 1 dump
;   -> out[0..7] = 0x77777777 0x88880008 0x99990009 0xDEADBEEF
;                  0xAAAAAAAA 0xBBBBBBBB 0x12341234 0x56785678
; rsrc2: 0x40088
; group: 32,1,1
; mattr: +wavefrontsize32,-wavefrontsize64
v_cmpx_eq_u32_e32 0, v0

; --- both offsets on one dword (element 0x204) ---
v_mov_b32_e32 v1, 0x204
v_mov_b32_e32 v4, 0x77777777
ds_write_b32 v1, v4
v_mov_b32_e32 v2, 0x700
v_mov_b32_e32 v5, 0xDEADBEEF
ds_write_b32 v2, v5
s_waitcnt lgkmcnt(0)
v_mov_b32_e32 v1, 0x200
v_mov_b32_e32 v4, 0x88880008
v_mov_b32_e32 v5, 0x99990009
ds_wrxchg2_rtn_b32 v[12:13], v1, v4, v5 offset0:1 offset1:1
s_waitcnt lgkmcnt(0)
v_mov_b32_e32 v21, 0
buffer_store_dword v12, v21, s[0:3], 0 offen
v_mov_b32_e32 v21, 4
buffer_store_dword v13, v21, s[0:3], 0 offen
v_mov_b32_e32 v1, 0x204
ds_read_b32 v10, v1
v_mov_b32_e32 v2, 0x700
ds_read_b32 v11, v2
s_waitcnt lgkmcnt(0)
v_mov_b32_e32 v21, 8
buffer_store_dword v10, v21, s[0:3], 0 offen
v_mov_b32_e32 v21, 12
buffer_store_dword v11, v21, s[0:3], 0 offen

; --- vdst aliases the data VGPRs ---
v_mov_b32_e32 v1, 0x300
v_mov_b32_e32 v4, 0xAAAAAAAA
ds_write_b32 v1, v4
v_mov_b32_e32 v2, 0x308
v_mov_b32_e32 v5, 0xBBBBBBBB
ds_write_b32 v2, v5
s_waitcnt lgkmcnt(0)
v_mov_b32_e32 v1, 0x300
v_mov_b32_e32 v4, 0x12341234
v_mov_b32_e32 v5, 0x56785678
ds_wrxchg2_rtn_b32 v[4:5], v1, v4, v5 offset0:0 offset1:2
s_waitcnt lgkmcnt(0)
v_mov_b32_e32 v21, 16
buffer_store_dword v4, v21, s[0:3], 0 offen
v_mov_b32_e32 v21, 20
buffer_store_dword v5, v21, s[0:3], 0 offen
ds_read_b32 v10, v1
s_waitcnt lgkmcnt(0)
v_mov_b32_e32 v21, 24
buffer_store_dword v10, v21, s[0:3], 0 offen
v_mov_b32_e32 v1, 0x308
ds_read_b32 v11, v1
s_waitcnt lgkmcnt(0)
v_mov_b32_e32 v21, 28
buffer_store_dword v11, v21, s[0:3], 0 offen

s_endpgm
