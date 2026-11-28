; GPU-CMP-F64: the 16 binary64 vector compares, v_cmp_*_f64 into VCC and v_cmpx_*_f64 on EXEC.
; group: 128,1,1
; mattr: +wavefrontsize32,-wavefrontsize64
; check.py seeds 256 records at byte 0x800: {a.lo, a.hi, b.lo, b.hi}, the Cartesian product of 16 binary64 values
; (record 16 * ia + ib holds values ia and ib), in ascending order and then two NaNs: -inf, -max, -1 - 1 ulp, -1,
; the value above -1 (high dword smaller, low dword all ones), the negative smallest subnormal, -0, +0, the smallest
; subnormal, the smallest normal, 1, 1 + 1 ulp, +max, +inf, a NaN whose payload is only in the low dword (its high
; dword is +inf's) and a negative quiet NaN.
; Thread i reads records i and i + 128 from memory, so neither the translator nor the driver can fold a compare,
; and writes two words per record at byte 8 * record: the v_cmp_ word, then the v_cmpx_ word.
; Bits 0..15 of both words are the 16 predicates in opcode order (RDNA2 ISA, VOPC compare operations):
;   f lt eq le gt lg ge o u nge nlg ngt nle neq nlt tru
; lt, eq and gt are IEEE-754 (false when either operand is a NaN, -0 == +0, no denormal flush), o is "neither is a
; NaN", u its negation, and le/lg/ge/nge/nlg/ngt/nle/neq/nlt are the unions and negations their names say, so every
; n* predicate is true for a NaN. v_cmp_ sets the lane's VCC bit; v_cmpx_ leaves the lane active only when true.
;   bit 16: v_cmp_tru / v_cmpx_tru with the odd threads turned off first: set on even threads only. A lane EXEC has
;           off gets VCC 0 from v_cmp_ and stays off through v_cmpx_.
;   bit 17: lt with VOP3 modifiers, |a| < -b.     bit 18: nge with VOP3 modifiers, !(-a >= |b|).
; Bits 19..31 are 0. EXEC is put back with a constant after each v_cmpx_, so no step needs a particular subgroup
; width. The input records must come back unchanged.
; Run with the heap initialized by check.py's cmp_f64_test; compare all 1536 words.
v_lshlrev_b32_e32 v1, 4, v0
v_add_nc_u32_e32 v11, 0x800, v1
v_lshlrev_b32_e32 v10, 3, v0
v_and_b32_e32 v9, 1, v0
buffer_load_dwordx4 v[2:5], v1, s[0:3], 0 offen offset:2048
s_waitcnt vmcnt(0)
v_mov_b32_e32 v6, 0
v_mov_b32_e32 v7, 0
v_cmp_f_f64_e32 vcc_lo, v[2:3], v[4:5]
v_or_b32_e32 v8, 0x1, v6
v_cndmask_b32_e32 v6, v6, v8, vcc_lo
v_cmp_lt_f64_e32 vcc_lo, v[2:3], v[4:5]
v_or_b32_e32 v8, 0x2, v6
v_cndmask_b32_e32 v6, v6, v8, vcc_lo
v_cmp_eq_f64_e32 vcc_lo, v[2:3], v[4:5]
v_or_b32_e32 v8, 0x4, v6
v_cndmask_b32_e32 v6, v6, v8, vcc_lo
v_cmp_le_f64_e32 vcc_lo, v[2:3], v[4:5]
v_or_b32_e32 v8, 0x8, v6
v_cndmask_b32_e32 v6, v6, v8, vcc_lo
v_cmp_gt_f64_e32 vcc_lo, v[2:3], v[4:5]
v_or_b32_e32 v8, 0x10, v6
v_cndmask_b32_e32 v6, v6, v8, vcc_lo
v_cmp_lg_f64_e32 vcc_lo, v[2:3], v[4:5]
v_or_b32_e32 v8, 0x20, v6
v_cndmask_b32_e32 v6, v6, v8, vcc_lo
v_cmp_ge_f64_e32 vcc_lo, v[2:3], v[4:5]
v_or_b32_e32 v8, 0x40, v6
v_cndmask_b32_e32 v6, v6, v8, vcc_lo
v_cmp_o_f64_e32 vcc_lo, v[2:3], v[4:5]
v_or_b32_e32 v8, 0x80, v6
v_cndmask_b32_e32 v6, v6, v8, vcc_lo
v_cmp_u_f64_e32 vcc_lo, v[2:3], v[4:5]
v_or_b32_e32 v8, 0x100, v6
v_cndmask_b32_e32 v6, v6, v8, vcc_lo
v_cmp_nge_f64_e32 vcc_lo, v[2:3], v[4:5]
v_or_b32_e32 v8, 0x200, v6
v_cndmask_b32_e32 v6, v6, v8, vcc_lo
v_cmp_nlg_f64_e32 vcc_lo, v[2:3], v[4:5]
v_or_b32_e32 v8, 0x400, v6
v_cndmask_b32_e32 v6, v6, v8, vcc_lo
v_cmp_ngt_f64_e32 vcc_lo, v[2:3], v[4:5]
v_or_b32_e32 v8, 0x800, v6
v_cndmask_b32_e32 v6, v6, v8, vcc_lo
v_cmp_nle_f64_e32 vcc_lo, v[2:3], v[4:5]
v_or_b32_e32 v8, 0x1000, v6
v_cndmask_b32_e32 v6, v6, v8, vcc_lo
v_cmp_neq_f64_e32 vcc_lo, v[2:3], v[4:5]
v_or_b32_e32 v8, 0x2000, v6
v_cndmask_b32_e32 v6, v6, v8, vcc_lo
v_cmp_nlt_f64_e32 vcc_lo, v[2:3], v[4:5]
v_or_b32_e32 v8, 0x4000, v6
v_cndmask_b32_e32 v6, v6, v8, vcc_lo
v_cmp_tru_f64_e32 vcc_lo, v[2:3], v[4:5]
v_or_b32_e32 v8, 0x8000, v6
v_cndmask_b32_e32 v6, v6, v8, vcc_lo
v_cmpx_eq_u32_e32 0, v9
v_cmp_tru_f64_e32 vcc_lo, v[2:3], v[4:5]
s_mov_b32 exec_lo, -1
v_or_b32_e32 v8, 0x10000, v6
v_cndmask_b32_e32 v6, v6, v8, vcc_lo
v_cmp_lt_f64_e64 vcc_lo, |v[2:3]|, -v[4:5]
v_or_b32_e32 v8, 0x20000, v6
v_cndmask_b32_e32 v6, v6, v8, vcc_lo
v_cmp_nge_f64_e64 vcc_lo, -v[2:3], |v[4:5]|
v_or_b32_e32 v8, 0x40000, v6
v_cndmask_b32_e32 v6, v6, v8, vcc_lo
v_cmpx_f_f64_e32 v[2:3], v[4:5]
v_or_b32_e32 v7, 0x1, v7
s_mov_b32 exec_lo, -1
v_cmpx_lt_f64_e32 v[2:3], v[4:5]
v_or_b32_e32 v7, 0x2, v7
s_mov_b32 exec_lo, -1
v_cmpx_eq_f64_e32 v[2:3], v[4:5]
v_or_b32_e32 v7, 0x4, v7
s_mov_b32 exec_lo, -1
v_cmpx_le_f64_e32 v[2:3], v[4:5]
v_or_b32_e32 v7, 0x8, v7
s_mov_b32 exec_lo, -1
v_cmpx_gt_f64_e32 v[2:3], v[4:5]
v_or_b32_e32 v7, 0x10, v7
s_mov_b32 exec_lo, -1
v_cmpx_lg_f64_e32 v[2:3], v[4:5]
v_or_b32_e32 v7, 0x20, v7
s_mov_b32 exec_lo, -1
v_cmpx_ge_f64_e32 v[2:3], v[4:5]
v_or_b32_e32 v7, 0x40, v7
s_mov_b32 exec_lo, -1
v_cmpx_o_f64_e32 v[2:3], v[4:5]
v_or_b32_e32 v7, 0x80, v7
s_mov_b32 exec_lo, -1
v_cmpx_u_f64_e32 v[2:3], v[4:5]
v_or_b32_e32 v7, 0x100, v7
s_mov_b32 exec_lo, -1
v_cmpx_nge_f64_e32 v[2:3], v[4:5]
v_or_b32_e32 v7, 0x200, v7
s_mov_b32 exec_lo, -1
v_cmpx_nlg_f64_e32 v[2:3], v[4:5]
v_or_b32_e32 v7, 0x400, v7
s_mov_b32 exec_lo, -1
v_cmpx_ngt_f64_e32 v[2:3], v[4:5]
v_or_b32_e32 v7, 0x800, v7
s_mov_b32 exec_lo, -1
v_cmpx_nle_f64_e32 v[2:3], v[4:5]
v_or_b32_e32 v7, 0x1000, v7
s_mov_b32 exec_lo, -1
v_cmpx_neq_f64_e32 v[2:3], v[4:5]
v_or_b32_e32 v7, 0x2000, v7
s_mov_b32 exec_lo, -1
v_cmpx_nlt_f64_e32 v[2:3], v[4:5]
v_or_b32_e32 v7, 0x4000, v7
s_mov_b32 exec_lo, -1
v_cmpx_tru_f64_e32 v[2:3], v[4:5]
v_or_b32_e32 v7, 0x8000, v7
s_mov_b32 exec_lo, -1
v_cmpx_eq_u32_e32 0, v9
v_cmpx_tru_f64_e32 v[2:3], v[4:5]
v_or_b32_e32 v7, 0x10000, v7
s_mov_b32 exec_lo, -1
v_cmpx_lt_f64_e64 |v[2:3]|, -v[4:5]
v_or_b32_e32 v7, 0x20000, v7
s_mov_b32 exec_lo, -1
v_cmpx_nge_f64_e64 -v[2:3], |v[4:5]|
v_or_b32_e32 v7, 0x40000, v7
s_mov_b32 exec_lo, -1
buffer_store_dwordx2 v[6:7], v10, s[0:3], 0 offen
buffer_load_dwordx4 v[2:5], v11, s[0:3], 0 offen offset:2048
s_waitcnt vmcnt(0)
v_mov_b32_e32 v6, 0
v_mov_b32_e32 v7, 0
v_cmp_f_f64_e32 vcc_lo, v[2:3], v[4:5]
v_or_b32_e32 v8, 0x1, v6
v_cndmask_b32_e32 v6, v6, v8, vcc_lo
v_cmp_lt_f64_e32 vcc_lo, v[2:3], v[4:5]
v_or_b32_e32 v8, 0x2, v6
v_cndmask_b32_e32 v6, v6, v8, vcc_lo
v_cmp_eq_f64_e32 vcc_lo, v[2:3], v[4:5]
v_or_b32_e32 v8, 0x4, v6
v_cndmask_b32_e32 v6, v6, v8, vcc_lo
v_cmp_le_f64_e32 vcc_lo, v[2:3], v[4:5]
v_or_b32_e32 v8, 0x8, v6
v_cndmask_b32_e32 v6, v6, v8, vcc_lo
v_cmp_gt_f64_e32 vcc_lo, v[2:3], v[4:5]
v_or_b32_e32 v8, 0x10, v6
v_cndmask_b32_e32 v6, v6, v8, vcc_lo
v_cmp_lg_f64_e32 vcc_lo, v[2:3], v[4:5]
v_or_b32_e32 v8, 0x20, v6
v_cndmask_b32_e32 v6, v6, v8, vcc_lo
v_cmp_ge_f64_e32 vcc_lo, v[2:3], v[4:5]
v_or_b32_e32 v8, 0x40, v6
v_cndmask_b32_e32 v6, v6, v8, vcc_lo
v_cmp_o_f64_e32 vcc_lo, v[2:3], v[4:5]
v_or_b32_e32 v8, 0x80, v6
v_cndmask_b32_e32 v6, v6, v8, vcc_lo
v_cmp_u_f64_e32 vcc_lo, v[2:3], v[4:5]
v_or_b32_e32 v8, 0x100, v6
v_cndmask_b32_e32 v6, v6, v8, vcc_lo
v_cmp_nge_f64_e32 vcc_lo, v[2:3], v[4:5]
v_or_b32_e32 v8, 0x200, v6
v_cndmask_b32_e32 v6, v6, v8, vcc_lo
v_cmp_nlg_f64_e32 vcc_lo, v[2:3], v[4:5]
v_or_b32_e32 v8, 0x400, v6
v_cndmask_b32_e32 v6, v6, v8, vcc_lo
v_cmp_ngt_f64_e32 vcc_lo, v[2:3], v[4:5]
v_or_b32_e32 v8, 0x800, v6
v_cndmask_b32_e32 v6, v6, v8, vcc_lo
v_cmp_nle_f64_e32 vcc_lo, v[2:3], v[4:5]
v_or_b32_e32 v8, 0x1000, v6
v_cndmask_b32_e32 v6, v6, v8, vcc_lo
v_cmp_neq_f64_e32 vcc_lo, v[2:3], v[4:5]
v_or_b32_e32 v8, 0x2000, v6
v_cndmask_b32_e32 v6, v6, v8, vcc_lo
v_cmp_nlt_f64_e32 vcc_lo, v[2:3], v[4:5]
v_or_b32_e32 v8, 0x4000, v6
v_cndmask_b32_e32 v6, v6, v8, vcc_lo
v_cmp_tru_f64_e32 vcc_lo, v[2:3], v[4:5]
v_or_b32_e32 v8, 0x8000, v6
v_cndmask_b32_e32 v6, v6, v8, vcc_lo
v_cmpx_eq_u32_e32 0, v9
v_cmp_tru_f64_e32 vcc_lo, v[2:3], v[4:5]
s_mov_b32 exec_lo, -1
v_or_b32_e32 v8, 0x10000, v6
v_cndmask_b32_e32 v6, v6, v8, vcc_lo
v_cmp_lt_f64_e64 vcc_lo, |v[2:3]|, -v[4:5]
v_or_b32_e32 v8, 0x20000, v6
v_cndmask_b32_e32 v6, v6, v8, vcc_lo
v_cmp_nge_f64_e64 vcc_lo, -v[2:3], |v[4:5]|
v_or_b32_e32 v8, 0x40000, v6
v_cndmask_b32_e32 v6, v6, v8, vcc_lo
v_cmpx_f_f64_e32 v[2:3], v[4:5]
v_or_b32_e32 v7, 0x1, v7
s_mov_b32 exec_lo, -1
v_cmpx_lt_f64_e32 v[2:3], v[4:5]
v_or_b32_e32 v7, 0x2, v7
s_mov_b32 exec_lo, -1
v_cmpx_eq_f64_e32 v[2:3], v[4:5]
v_or_b32_e32 v7, 0x4, v7
s_mov_b32 exec_lo, -1
v_cmpx_le_f64_e32 v[2:3], v[4:5]
v_or_b32_e32 v7, 0x8, v7
s_mov_b32 exec_lo, -1
v_cmpx_gt_f64_e32 v[2:3], v[4:5]
v_or_b32_e32 v7, 0x10, v7
s_mov_b32 exec_lo, -1
v_cmpx_lg_f64_e32 v[2:3], v[4:5]
v_or_b32_e32 v7, 0x20, v7
s_mov_b32 exec_lo, -1
v_cmpx_ge_f64_e32 v[2:3], v[4:5]
v_or_b32_e32 v7, 0x40, v7
s_mov_b32 exec_lo, -1
v_cmpx_o_f64_e32 v[2:3], v[4:5]
v_or_b32_e32 v7, 0x80, v7
s_mov_b32 exec_lo, -1
v_cmpx_u_f64_e32 v[2:3], v[4:5]
v_or_b32_e32 v7, 0x100, v7
s_mov_b32 exec_lo, -1
v_cmpx_nge_f64_e32 v[2:3], v[4:5]
v_or_b32_e32 v7, 0x200, v7
s_mov_b32 exec_lo, -1
v_cmpx_nlg_f64_e32 v[2:3], v[4:5]
v_or_b32_e32 v7, 0x400, v7
s_mov_b32 exec_lo, -1
v_cmpx_ngt_f64_e32 v[2:3], v[4:5]
v_or_b32_e32 v7, 0x800, v7
s_mov_b32 exec_lo, -1
v_cmpx_nle_f64_e32 v[2:3], v[4:5]
v_or_b32_e32 v7, 0x1000, v7
s_mov_b32 exec_lo, -1
v_cmpx_neq_f64_e32 v[2:3], v[4:5]
v_or_b32_e32 v7, 0x2000, v7
s_mov_b32 exec_lo, -1
v_cmpx_nlt_f64_e32 v[2:3], v[4:5]
v_or_b32_e32 v7, 0x4000, v7
s_mov_b32 exec_lo, -1
v_cmpx_tru_f64_e32 v[2:3], v[4:5]
v_or_b32_e32 v7, 0x8000, v7
s_mov_b32 exec_lo, -1
v_cmpx_eq_u32_e32 0, v9
v_cmpx_tru_f64_e32 v[2:3], v[4:5]
v_or_b32_e32 v7, 0x10000, v7
s_mov_b32 exec_lo, -1
v_cmpx_lt_f64_e64 |v[2:3]|, -v[4:5]
v_or_b32_e32 v7, 0x20000, v7
s_mov_b32 exec_lo, -1
v_cmpx_nge_f64_e64 -v[2:3], |v[4:5]|
v_or_b32_e32 v7, 0x40000, v7
s_mov_b32 exec_lo, -1
buffer_store_dwordx2 v[6:7], v10, s[0:3], 0 offen offset:1024
s_endpgm
