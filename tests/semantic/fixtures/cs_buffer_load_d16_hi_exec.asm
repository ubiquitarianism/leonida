; buffer_load_short_d16_hi with lanes switched off in EXEC (test_buffer_load_d16_hi_exec.py). Hand-written, wave32.
; s[0:3] is the push-constant V# (base 0, num_records 0xFFFFFFFF). Every lane first puts a value of its own in v5,
; (lane * 0x10001) ^ 0xC0DE1234: high half 0xC0DE ^ lane, low half 0x1234 ^ lane. EXEC is then cut to 0x5AC3A50F
; and each lane loads the 16-bit word at byte 0x100 + 2 * lane into the high half of v5. With EXEC back to all
; lanes, lane i stores v5 to byte 0x200 + 4 * i:
;   active lane:   (loaded short << 16) | (0x1234 ^ lane)   the high half replaced, the low half kept
;   inactive lane: (lane * 0x10001) ^ 0xC0DE1234            both halves kept
v_lshlrev_b32_e32 v1, 1, v0
v_lshlrev_b32_e32 v6, 2, v0
v_mul_u32_u24_e32 v5, 0x10001, v0
v_xor_b32_e32 v5, 0xc0de1234, v5
s_mov_b32 exec_lo, 0x5ac3a50f
buffer_load_short_d16_hi v5, v1, s[0:3], 0 offen offset:256
s_waitcnt vmcnt(0)
s_mov_b32 exec_lo, -1
buffer_store_dword v5, v6, s[0:3], 0 offen offset:512
s_endpgm
