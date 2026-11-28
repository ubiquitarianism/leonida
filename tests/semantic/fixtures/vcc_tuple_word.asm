; Synthetic s_buffer_load_dwordx4 from s[4:7] into s104, s105, vcc_lo, vcc_hi.
; A following scalar wait is outside the eight bytes the decoder must restore.
.long 0xf4281a02, 0xfa000000
s_waitcnt lgkmcnt(0)
