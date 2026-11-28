; regen_fixtures: this source has a translator fixture in tests/semantic/fixtures/disasm
; Partial wave64, ds_swizzle_b32 / ds_bpermute_b32 from a lane that doesn't exist: a 40x1x1 group is one wave with
; lanes 0-39 running and 40-63 missing. A source lane that doesn't exist reads 0 (RDNA2 ISA, ds_swizzle/ds_bpermute of
; a lane that isn't running; on gfx1036 through tools/hw_oracle, 40 threads, native wave64: lanes 32-39 of SWAP 8 and
; of BROADCAST 16,15 read 0). The exchange sits after a vote and an s_cbranch_execz, as in the crosslane_fuzz
; programs that found it (seed 48315): with --wave64-emulate, RADV (Mesa 26.3.1) turned the store, barrier and load of
; the exchange slot into a subgroup swizzle that fetched the missing lane's register, so lanes 32-39 read garbage
; that changed between runs although the slot was zeroed.
; v1 = lane, v2 = 500 + lane, s8 = s_bcnt1 of EXEC (40, an exchange before the branch).
;   v3 = ds_swizzle SWAP 8 of v2 (source lane ^ 8): lane 0 -> 8 -> 508, lane 31 -> 23 -> 523, lane 33 -> 41 -> 0,
;        lane 39 -> 47 -> 0.
;   v4 = ds_bpermute with addr = (lane ^ 8) * 4 (each half its own 32 lanes): lane 20 -> 28 -> 528, lane 33 -> 41 -> 0.
;   v5 = ds_swizzle BROADCAST 16,15 (lane 15 of each group of 16): lane 20 -> 31 -> 531, lane 35 -> 47 -> 0.
; Stores: out[0] lane 0 v3, out[1] lane 31 v3, out[2] lane 33 v3, out[3] lane 39 v3, out[4] lane 20 v4,
;   out[5] lane 33 v4, out[6] lane 20 v5, out[7] lane 35 v5.
; As for cs_w64x_partial, 40 threads can't have full subgroups: check.py asks for subgroup size 64 (32 when emulated).
; Run: VKC_SUBGROUP_SIZE=64 vk_compute_test cs_w64x_partial_swizzle.spv 1 dump
;   -> out[0..7] = 508 523 0 0 528 0 531 0
; mattr: +wavefrontsize64,-wavefrontsize32
; group: 40,1,1
v_mbcnt_lo_u32_b32 v1, -1, 0
v_mbcnt_hi_u32_b32 v1, -1, v1
v_add_nc_u32_e32 v2, 500, v1
v_mov_b32_e32 v3, 99
v_mov_b32_e32 v4, 99
v_mov_b32_e32 v5, 99
s_mov_b64 s[6:7], exec
s_bcnt1_i32_b64 s8, exec
v_cmp_gt_u32_e64 s[4:5], 48, v1
s_and_saveexec_b64 s[10:11], s[4:5]
s_cbranch_execz 9
ds_swizzle_b32 v3, v2 offset:swizzle(SWAP,8)
ds_swizzle_b32 v5, v2 offset:swizzle(BROADCAST,16,15)
v_xor_b32_e32 v6, 8, v1
v_lshlrev_b32_e32 v6, 2, v6
ds_bpermute_b32 v4, v6, v2
s_waitcnt lgkmcnt(0)
s_mov_b64 exec, s[6:7]
v_cmpx_eq_u32_e32 0, v1
v_mov_b32_e32 v20, v3
v_mov_b32_e32 v21, 0
buffer_store_dword v20, v21, s[0:3], 0 offen
s_mov_b64 exec, s[6:7]
v_cmpx_eq_u32_e32 31, v1
v_mov_b32_e32 v20, v3
v_mov_b32_e32 v21, 4
buffer_store_dword v20, v21, s[0:3], 0 offen
s_mov_b64 exec, s[6:7]
v_cmpx_eq_u32_e32 33, v1
v_mov_b32_e32 v20, v3
v_mov_b32_e32 v21, 8
buffer_store_dword v20, v21, s[0:3], 0 offen
s_mov_b64 exec, s[6:7]
v_cmpx_eq_u32_e32 39, v1
v_mov_b32_e32 v20, v3
v_mov_b32_e32 v21, 12
buffer_store_dword v20, v21, s[0:3], 0 offen
s_mov_b64 exec, s[6:7]
v_cmpx_eq_u32_e32 20, v1
v_mov_b32_e32 v20, v4
v_mov_b32_e32 v21, 16
buffer_store_dword v20, v21, s[0:3], 0 offen
s_mov_b64 exec, s[6:7]
v_cmpx_eq_u32_e32 33, v1
v_mov_b32_e32 v20, v4
v_mov_b32_e32 v21, 20
buffer_store_dword v20, v21, s[0:3], 0 offen
s_mov_b64 exec, s[6:7]
v_cmpx_eq_u32_e32 20, v1
v_mov_b32_e32 v20, v5
v_mov_b32_e32 v21, 24
buffer_store_dword v20, v21, s[0:3], 0 offen
s_mov_b64 exec, s[6:7]
v_cmpx_eq_u32_e32 35, v1
v_mov_b32_e32 v20, v5
v_mov_b32_e32 v21, 28
buffer_store_dword v20, v21, s[0:3], 0 offen
s_endpgm
