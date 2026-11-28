; GPU-ROUND-F64: round binary64 to an integral value, v_trunc_f64, v_ceil_f64, v_rndne_f64 and v_floor_f64.
; group: 64,1,1
; mattr: +wavefrontsize32,-wavefrontsize64
; check.py seeds 64 binary64 inputs at byte 0x1000, {lo, hi} each: signed zeros, the smallest subnormals and
; normals, values around 0.5 and 1 (the largest value below 0.5 among them), ties at 0.5, 1.5, 2.5, 3.5, 4.5 and 5.5,
; ties at 2^31 + 0.5, 2^51 + 0.5 and 2^52 - 0.5, the integers 2^52, 2^52 + 1, 2^53 - 1, 2^53 and 2^63, 1e300, the
; largest finite value, the infinities, quiet NaNs with and without a payload and two signalling NaNs. Thread i
; reads input i from memory, so neither the translator nor the driver can fold a rounding, and writes eight results
; of two words (low first) at byte 64 * i:
;   +0   v_trunc_f64 x                toward zero
;   +8   v_ceil_f64 x                 toward +infinity
;   +16  v_rndne_f64 x                to nearest, a tie to the even integer
;   +24  v_floor_f64 x                toward -infinity
;   +32  v_trunc_f64 -x               VOP3 neg: the sign bit is flipped first
;   +40  v_ceil_f64 |x|               VOP3 abs: the sign bit is cleared first
;   +48  v_floor_f64 x                the destination pair v[21:22] overlaps the source pair v[20:21]
;   +56  v_rndne_f64 x                with the odd threads turned off first: they keep the pair 0xdeadbeef,
;                                     0xcafef00d that was in the destination
; RDNA2 ISA VOP1 opcodes 23..26. A result that is zero keeps the sign of its source (ceil of -0.25 is -0), a value
; of magnitude 2^52 or more is already integral, an infinity stays, and a NaN and a subnormal are rounded like any
; other value: a NaN comes back unchanged, signalling ones included, ceil of the smallest subnormal is 1 and floor
; of its negative is -1 (docs/design.md). The translator gives the four ops to the driver's double-precision Trunc,
; Ceil, RoundEven and Floor, except for a NaN and a subnormal, which it rounds with integer ops: Vulkan lets a driver
; return any NaN for a NaN (a signalling one came back quiet on an RTX 4070 Ti and on llvmpipe) and, with no
; DenormPreserve for 64 bits, read a subnormal as zero (llvmpipe did).
; Run with the heap initialized by check.py's round_f64_test; all 1152 words are compared, and the input records
; must come back unchanged.
v_lshlrev_b32_e32 v1, 3, v0
v_add_nc_u32_e32 v1, 0x1000, v1
v_lshlrev_b32_e32 v10, 6, v0
v_and_b32_e32 v9, 1, v0
buffer_load_dwordx2 v[2:3], v1, s[0:3], 0 offen
s_waitcnt vmcnt(0)
v_trunc_f64_e32 v[4:5], v[2:3]
v_ceil_f64_e32 v[6:7], v[2:3]
v_rndne_f64_e32 v[12:13], v[2:3]
v_floor_f64_e32 v[14:15], v[2:3]
v_trunc_f64_e64 v[16:17], -v[2:3]
v_ceil_f64_e64 v[18:19], |v[2:3]|
v_mov_b32_e32 v20, v2
v_mov_b32_e32 v21, v3
v_floor_f64_e32 v[21:22], v[20:21]
v_mov_b32_e32 v24, 0xdeadbeef
v_mov_b32_e32 v25, 0xcafef00d
v_cmpx_eq_u32_e32 0, v9
v_rndne_f64_e32 v[24:25], v[2:3]
s_mov_b32 exec_lo, -1
buffer_store_dwordx2 v[4:5], v10, s[0:3], 0 offen
buffer_store_dwordx2 v[6:7], v10, s[0:3], 0 offen offset:8
buffer_store_dwordx2 v[12:13], v10, s[0:3], 0 offen offset:16
buffer_store_dwordx2 v[14:15], v10, s[0:3], 0 offen offset:24
buffer_store_dwordx2 v[16:17], v10, s[0:3], 0 offen offset:32
buffer_store_dwordx2 v[18:19], v10, s[0:3], 0 offen offset:40
buffer_store_dwordx2 v[21:22], v10, s[0:3], 0 offen offset:48
buffer_store_dwordx2 v[24:25], v10, s[0:3], 0 offen offset:56
s_endpgm
