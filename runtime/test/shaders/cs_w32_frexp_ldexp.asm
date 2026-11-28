; GPU-FREXP-LDEXP: v_frexp_mant_f32, v_frexp_exp_i32_f32, v_ldexp_f32, v_frexp_mant_f64 and v_frexp_exp_i32_f64.
; group: 64,1,1
; mattr: +wavefrontsize32,-wavefrontsize64
; check.py seeds 64 records at byte 0x800: {x, n, d.lo, d.hi}, a binary32 value, an ldexp exponent and a binary64
; value. Thread i reads record i from memory, so neither the translator nor the driver can fold an op, and writes
; eight words at byte 32 * i:
;   +0   v_frexp_mant_f32 x           a normal keeps sign and fraction and gets the exponent field 126 (a mantissa
;                                     in [0.5, 1)); a zero or a denormal, read as zero, gives a zero of its sign;
;                                     an infinity or a NaN comes back unchanged
;   +4   v_frexp_exp_i32_f32 x        the exponent field minus 126 for a normal, 0 for everything else
;   +8   v_ldexp_f32 x, n             n is clamped to [-512, 512] and added to the exponent field: above 254 the
;                                     result is an infinity of x's sign, at 0 or below a zero of its sign (there
;                                     are no denormal results); a zero, a denormal, an infinity and a NaN ignore n
;   +12  v_frexp_mant_f32 -x          VOP3 neg: the sign bit is flipped first
;   +16  v_frexp_mant_f64 d           two words, low first: a normal gets the exponent field 1022; a subnormal is
;                                     normalized into [0.5, 1) too; a zero keeps its sign; an infinity or a NaN
;                                     comes back unchanged
;   +24  v_frexp_exp_i32_f64 d        the exponent field minus 1022 for a normal, the position of the top fraction
;                                     bit minus 1073 for a subnormal (-1073..-1022), 0 for everything else
;   +28  v_ldexp_f32 |x|, n           VOP3 abs, with the odd threads turned off first: they keep the 0xdeadbeef
;                                     that was in the destination
; The records cover normals, the clamp of n at +-512 and beyond, results that overflow and that fall below the
; smallest normal, binary32 denormals, binary64 subnormals with single bits on both sides of the dword boundary and
; with several bits, zeros, infinities and NaNs. The rules are the rows of docs/design.md for these five ops.
; The input records must come back unchanged.
; Run with the heap initialized by check.py's frexp_ldexp_test; compare all 768 words.
v_lshlrev_b32_e32 v1, 4, v0
v_lshlrev_b32_e32 v10, 5, v0
v_and_b32_e32 v9, 1, v0
buffer_load_dwordx4 v[2:5], v1, s[0:3], 0 offen offset:2048
s_waitcnt vmcnt(0)
v_frexp_mant_f32_e32 v12, v2
v_frexp_exp_i32_f32_e32 v13, v2
v_ldexp_f32 v14, v2, v3
v_frexp_mant_f32_e64 v15, -v2
v_frexp_mant_f64_e32 v[16:17], v[4:5]
v_frexp_exp_i32_f64_e32 v18, v[4:5]
v_mov_b32_e32 v19, 0xdeadbeef
v_cmpx_eq_u32_e32 0, v9
v_ldexp_f32 v19, |v2|, v3
s_mov_b32 exec_lo, -1
buffer_store_dwordx4 v[12:15], v10, s[0:3], 0 offen
buffer_store_dwordx4 v[16:19], v10, s[0:3], 0 offen offset:16
s_endpgm
