// s[0:1] = destination. Hardware check for the new v_dot2c_f32_f16 forms the patcher now covers
// (literal src0, DPP src0, vdst aliasing). Assembled for gfx1030, then patched with
//   patch_blob dot2p2.bin dot2p2.patched.bin 12   (12 VGPRs used -> scratch is v12)
// and run with   bc250-native-probe --blob dot2p2.patched.bin --vgprs 13
// Expected per lane, 4 dwords:
//   +0  literal src0  10.0 + (1.5,2.0).(3.0,4.0) = 22.5   0x41b40000  (all lanes)
//   +4  DPP src0      quad_perm:[1,0,3,2] swaps lane pairs; lane0 -> 6.0 0x40c00000,
//                     lane1 -> 4.0 0x40800000
//   +8  vdst aliases  v5 = 8.0 (f16 lo 0.0, hi 2.5): 8.0 + 2.5*2.5 = 14.25  0x41640000
//   +12 control                                                    0xabcd0123
  v_lshlrev_b32 v9, 4, v0

  // literal src0: v_dot2c v3, 0x40003e00, v2
  v_mov_b32 v2, 0x44004200
  v_mov_b32 v3, 0x41200000
  v_dot2c_f32_f16 v3, 0x40003e00, v2

  // DPP src0: per-lane v1 = (lane&1) ? (2.0,2.0) : (1.0,1.0); v2 = (1.0,1.0); acc v4 = 2.0
  v_mov_b32 v2, 0x3c003c00
  v_mov_b32 v11, 0x40004000
  v_mov_b32 v1, 0x3c003c00
  v_and_b32 v10, 1, v0
  v_cmp_ne_u32_e32 vcc, 0, v10
  v_cndmask_b32 v1, v1, v11, vcc
  v_mov_b32 v4, 0x40000000
  v_dot2c_f32_f16_dpp v4, v1, v2 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf

  // vdst aliases both inputs: v_dot2c v5, v5, v5 with v5 = 8.0 (f16 lo 0.0, hi 2.5)
  v_mov_b32 v5, 0x41000000
  v_dot2c_f32_f16 v5, v5, v5

  v_mov_b32 v6, 0xabcd0123
  global_store_dwordx4 v9, v[3:6], s[0:1]
  s_endpgm
