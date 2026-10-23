// s[0:1] = destination. Probes what BC-250 executes for VOP2 opcode 0x02 (v_dot2c_f32_f16 on
// gfx10.3), emitted as raw words because LLVM rejects it for gfx1013. Per lane, 4 dwords:
// +0 accumulator 10.0, inputs (1.5,2.0).(3.0,4.0); +4 accumulator 0x12345678, same inputs;
// +8 accumulator 10.0, inputs zero; +12 control value 0xabcd0123 written without the opcode.
  v_lshlrev_b32 v9, 4, v0
  v_mov_b32 v1, 0x40003e00
  v_mov_b32 v2, 0x44004200
  v_mov_b32 v3, 0x41200000
  .long 0x04060501            // v_dot2c_f32_f16 v3, v1, v2
  v_mov_b32 v4, 0x12345678
  .long 0x04080501            // v_dot2c_f32_f16 v4, v1, v2
  v_mov_b32 v5, 0x41200000
  v_mov_b32 v6, 0
  .long 0x040a0d06            // v_dot2c_f32_f16 v5, v6, v6
  v_mov_b32 v7, 0xabcd0123
  global_store_dwordx4 v9, v[3:6], s[0:1]
  global_store_dword v9, v7, s[0:1] offset:12
  s_endpgm
