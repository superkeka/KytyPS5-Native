// s[0:1] = source, s[2:3] = destination, s4 = 16 KiB steps per group, s5 = workgroup id
// 256 threads per group, each thread reads 4 x 16 bytes per step.
  s_mul_i32 s6, s5, s4
  s_lshl_b32 s6, s6, 14
  s_add_u32 s0, s0, s6
  s_addc_u32 s1, s1, 0
  v_lshlrev_b32 v1, 4, v0
  v_add_nc_u32 v10, 0x1000, v1
  v_add_nc_u32 v11, 0x2000, v1
  v_add_nc_u32 v12, 0x3000, v1
  v_mov_b32 v8, 0
  s_mov_b32 s7, s4
loop:
  global_load_dwordx4 v[16:19], v1, s[0:1]
  global_load_dwordx4 v[20:23], v10, s[0:1]
  global_load_dwordx4 v[24:27], v11, s[0:1]
  global_load_dwordx4 v[28:31], v12, s[0:1]
  s_add_u32 s0, s0, 0x4000
  s_addc_u32 s1, s1, 0
  s_sub_u32 s7, s7, 1
  s_waitcnt vmcnt(0)
  v_xor3_b32 v8, v8, v16, v20
  v_xor3_b32 v8, v8, v24, v28
  s_cmp_lg_u32 s7, 0
  s_cbranch_scc1 loop
  s_lshl_b32 s8, s5, 10
  v_lshlrev_b32 v9, 2, v0
  v_add_nc_u32 v9, s8, v9
  global_store_dword v9, v8, s[2:3]
  s_endpgm
