// s[0:1] = destination. Assembled for gfx1030: LLVM rejects v_dot2c_f32_f16 for gfx1013,
// this checks whether BC-250 hardware runs it anyway (KytyPS5 decodes it for PS5 shaders).
// 10.0 + 1.5 * 3.0 + 2.0 * 4.0 = 22.5 (0x41b40000) in every lane.
  v_mov_b32 v1, 0x40003e00
  v_mov_b32 v2, 0x44004200
  v_mov_b32 v3, 0x41200000
  v_dot2c_f32_f16 v3, v1, v2
  v_lshlrev_b32 v4, 2, v0
  global_store_dword v4, v3, s[0:1]
  s_endpgm
