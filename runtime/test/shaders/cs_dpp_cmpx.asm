; v_cmpx in DPP form (SHADER-VOPCX-DPP): src0 is read from the permuted lane, src1 from the lane itself, and the result
; goes to EXEC. v1 = float(lane - 4). Both compares take src0 from the other half of each quad (quad_perm [2,3,0,1],
; with bound_ctrl:1 and fi:1, the form the game's pixel shaders use). The first keeps a lane where
; "neighbour > self": lanes 0, 1, 4, 5 of each 8. The second negates src1 ("neighbour > -self"): lanes 4-7 of the first
; 8 and every lane after. out = (first keeps the lane ? 1 : 0) + (second keeps it ? 2 : 0), EXEC restored in between.
; Run: vk_compute_test cs_dpp_cmpx.spv 1 dump -> out[0..7] = 1 1 0 0 3 3 2 2
; llvm-mc assembles neither compare for gfx1013 ("dpp variant of this instruction is not supported"), so the GPU
; test takes this file's text as its listing; tools/shader_disasm.py prints the same lines for the encoded words.
; run_gpu_tests: text only, llvm-mc has no gfx1013 encoding for an instruction here
; regen_fixtures: this source has a translator fixture in tests/semantic/fixtures/disasm
v_mbcnt_lo_u32_b32 v0, -1, 0
v_add_nc_u32 v7, -4, v0
v_cvt_f32_i32 v1, v7
v_mov_b32 v6, 0
s_mov_b32 s10, exec_lo
v_cmpx_gt_f32_dpp v1, v1 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1 fi:1
v_mov_b32 v6, 1
s_mov_b32 exec_lo, s10
v_cmpx_gt_f32_dpp v1, -v1 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1 fi:1
v_add_nc_u32 v6, 2, v6
s_mov_b32 exec_lo, s10
v_lshlrev_b32 v2, 2, v0
buffer_store_dword v6, v2, s[0:3], 0 offen
s_endpgm
