; V# base = 0xffffffffffc0 (-64), stride 0, RAW bounds. Read at heap_base - 48 and heap_base.
	s_mov_b32 s4, 0xffffffc0
	s_mov_b32 s5, 0x0000ffff
	s_mov_b32 s6, 0x44
	s_mov_b32 s7, 0x31016fac
	buffer_load_dword v0, off, s[4:7], 0 offset:16
	buffer_load_dword v1, off, s[4:7], 0 offset:64
	s_waitcnt vmcnt(0)
	exp mrt0 v0, v1, v0, v1 done vm
	s_endpgm
