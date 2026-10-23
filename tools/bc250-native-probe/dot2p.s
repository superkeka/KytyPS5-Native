// s[0:1] = destination. Input for the native_gpu shader patcher hardware check (assembled for
// gfx1030, then patched): per lane 4 dwords, expected after patching
// 0x41b40000 (VGPR src0), 0x41b40000 (SGPR src0), 0x41200000 (zero inputs), 0xabcd0123.
  v_lshlrev_b32 v9, 4, v0
  v_mov_b32 v1, 0x40003e00
  v_mov_b32 v2, 0x44004200
  v_mov_b32 v3, 0x41200000
  v_dot2c_f32_f16 v3, v1, v2
  s_mov_b32 s4, 0x40003e00
  v_mov_b32 v4, 0x41200000
  v_dot2c_f32_f16 v4, s4, v2
  v_mov_b32 v5, 0x41200000
  v_mov_b32 v6, 0
  v_dot2c_f32_f16 v5, v6, v6
  v_mov_b32 v6, 0xabcd0123
  global_store_dwordx4 v9, v[3:6], s[0:1]
  s_endpgm
