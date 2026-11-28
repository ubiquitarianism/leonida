; group: 1,1,1
; rsrc2: 0x8090
; regen_fixtures: this source has a translator fixture in tests/semantic/fixtures/disasm
; The address guard (rdna2spirv.py --address-guard; docs/d3d12_vulkan.md, "Address guard"). Translated with the guard,
; run by scripts/run_address_guard_gpu_test.py on a 2 MiB heap whose page table (built by that script) backs the
; arena (offsets below 0x80000, the header's limit) and the one 4 KiB page at 0x90000, and nothing else. The heap's
; real memory continues past both, so every address below is inside the buffer: with the guard off the same program
; reads and writes it, with the guard on the page table refuses the accesses outside 0x90000..0x90fff. Seeds:
;   0x90000 0xa11ce000  0x90004 0xa11ce004  0x90ffc 0xf00df00d  0x91000 0xf00df00e
;   0xa0000 0xbadbad00  0xa0004 0xbadbad04  0xa0008 0xbadbad08
; Words stored to out[0..7] with the guard on (off: the second value):
;   out[0] global_load_dword at 0x90000, a backed page outside the arena     = 0xa11ce000 (0xa11ce000)
;   out[1] global_load_dword at 0xa0000, unbacked                            = 0          (0xbadbad00)
;   out[2] global_atomic_add glc at 0xa0008 with 1, unbacked: the old value  = 0          (0xbadbad08)
;   out[3] global_load_dwordx2 at 0x90ffc: its last byte is on an unbacked page, so all of it is refused
;                                                                            = 0          (0xf00df00d)
;   out[4] its second dword                                                  = 0          (0xf00df00e)
;   out[5] buffer_load_dword through a V# built in SGPRs, base 0xa0000       = 0          (0xbadbad00)
;   out[6] s_load_dword at 0x90004, backed                                   = 0xa11ce004 (0xa11ce004)
;   out[7] s_load_dword at 0xa0000, unbacked                                 = 0          (0xbadbad00)
; global_store_dword 0x5707 at 0xa0004 (unbacked) goes to the dump area (0x78004) instead, and the atomic to 0x78008;
; the six refused accesses count themselves at header + 24 (0x70018) and the last one's offset is at 0x70020.
; Run: VKC_HEAP_INIT=<heap> VKC_OUT_WORDS=524288 VKC_DUMP=<file> vk_compute_test cs_address_guard.spv 1 dump
v_mov_b32_e32 v0, 0x90000
v_mov_b32_e32 v1, 0
global_load_dword v2, v[0:1], off
v_mov_b32_e32 v0, 0xa0000
global_load_dword v3, v[0:1], off
v_mov_b32_e32 v4, 0x5707
global_store_dword v[0:1], v4, off offset:4
v_mov_b32_e32 v5, 1
global_atomic_add v5, v[0:1], v5, off offset:8 glc
v_mov_b32_e32 v0, 0x90ffc
global_load_dwordx2 v[6:7], v[0:1], off
s_mov_b32 s4, 0xa0000
s_mov_b32 s5, 0
s_mov_b32 s6, 0x100
s_mov_b32 s7, 0x31016fac
v_mov_b32_e32 v8, 0
buffer_load_dword v8, v8, s[4:7], 0 offen
s_mov_b32 s8, 0x90000
s_mov_b32 s9, 0
s_load_dword s10, s[8:9], 0x4
s_load_dword s11, s[4:5], 0x0
s_waitcnt vmcnt(0) lgkmcnt(0)
v_mov_b32_e32 v9, 0
buffer_store_dword v2, v9, s[0:3], 0 offen
buffer_store_dword v3, v9, s[0:3], 0 offen offset:4
buffer_store_dword v5, v9, s[0:3], 0 offen offset:8
buffer_store_dword v6, v9, s[0:3], 0 offen offset:12
buffer_store_dword v7, v9, s[0:3], 0 offen offset:16
buffer_store_dword v8, v9, s[0:3], 0 offen offset:20
v_mov_b32_e32 v10, s10
buffer_store_dword v10, v9, s[0:3], 0 offen offset:24
v_mov_b32_e32 v11, s11
buffer_store_dword v11, v9, s[0:3], 0 offen offset:28
s_endpgm
