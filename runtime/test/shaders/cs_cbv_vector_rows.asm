; group: 1,1,1
; Read four 16-byte register rows through the actual Xbox constant-buffer descriptor staged at byte 64.
s_mov_b32 s4, 64
s_mov_b32 s5, 0
s_load_dwordx4 s[8:11], s[4:5], 0
s_waitcnt lgkmcnt(0)
v_mov_b32 v20, 0
v_mov_b32 v0, 0
buffer_load_dwordx4 v[4:7], v0, s[8:11], 0 idxen
s_waitcnt vmcnt(0)
buffer_store_dword v4, v20, s[0:3], 0 offen offset:0
v_mov_b32 v0, 1
buffer_load_dwordx4 v[4:7], v0, s[8:11], 0 idxen
s_waitcnt vmcnt(0)
buffer_store_dword v4, v20, s[0:3], 0 offen offset:4
v_mov_b32 v0, 2
buffer_load_dwordx4 v[4:7], v0, s[8:11], 0 idxen
s_waitcnt vmcnt(0)
buffer_store_dword v4, v20, s[0:3], 0 offen offset:8
v_mov_b32 v0, 3
buffer_load_dwordx4 v[4:7], v0, s[8:11], 0 idxen
s_waitcnt vmcnt(0)
buffer_store_dword v4, v20, s[0:3], 0 offen offset:12
s_buffer_load_dwordx2 s[12:13], s[8:11], 0x30
s_waitcnt lgkmcnt(0)
v_mov_b32 v4, s12
v_mov_b32 v5, s13
buffer_store_dword v4, v20, s[0:3], 0 offen offset:16
buffer_store_dword v5, v20, s[0:3], 0 offen offset:20
v_mov_b32 v0, 16
buffer_load_dword v4, v0, s[8:11], 0 idxen
s_waitcnt vmcnt(0)
buffer_store_dword v4, v20, s[0:3], 0 offen offset:24
v_mov_b32 v0, 15
buffer_load_dword v4, v0, s[8:11], 0 idxen offset:12
s_waitcnt vmcnt(0)
buffer_store_dword v4, v20, s[0:3], 0 offen offset:28
s_buffer_load_dword s14, s[8:11], 0x100
s_waitcnt lgkmcnt(0)
v_mov_b32 v4, s14
buffer_store_dword v4, v20, s[0:3], 0 offen offset:32
s_endpgm
