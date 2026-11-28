; Wave64 cross-lane reads from lanes that don't exist in a partial last wave (straight-line; the GPU test
; runtime/test/shaders/cs_w64x_partial_swizzle.asm has the same three reads behind a vote and an EXECZ branch).
; v1 = wave lane, v2 = 500 + lane. Every lane stores v3, v4, v5 at 12 * thread:
;   v3 = ds_swizzle SWAP 8 (source lane ^ 8), v4 = ds_bpermute with addr (lane ^ 8) * 4,
;   v5 = ds_swizzle BROADCAST 16,15 (lane 15 of each group of 16).
; A source lane that doesn't exist reads 0 (measured on gfx1036 through tools/hw_oracle, 40 threads, wave64).
v_mbcnt_lo_u32_b32 v1, -1, 0
v_mbcnt_hi_u32_b32 v1, -1, v1
v_add_nc_u32_e32 v2, 500, v1
ds_swizzle_b32 v3, v2 offset:swizzle(SWAP,8)
ds_swizzle_b32 v5, v2 offset:swizzle(BROADCAST,16,15)
v_xor_b32_e32 v6, 8, v1
v_lshlrev_b32_e32 v6, 2, v6
ds_bpermute_b32 v4, v6, v2
s_waitcnt lgkmcnt(0)
v_mul_u32_u24_e32 v22, 12, v0
buffer_store_dword v3, v22, s[0:3], 0 offen
buffer_store_dword v4, v22, s[0:3], 0 offen offset:4
buffer_store_dword v5, v22, s[0:3], 0 offen offset:8
s_endpgm
