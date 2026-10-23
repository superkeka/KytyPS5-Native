// s[0:1] = inputs (per lane: packed f16x2 a, packed f16x2 b, f32 acc), s[2:3] = results.
// The replacement a native backend would branch to for v_dot2c_f32_f16 acc, a, b:
// two mixed-precision FMAs, low halves then high halves, accumulating in place.
  v_lshlrev_b32 v1, 2, v0
  v_mul_u32_u24 v2, 12, v0
  global_load_dword v3, v2, s[0:1]
  global_load_dword v4, v2, s[0:1] offset:4
  global_load_dword v5, v2, s[0:1] offset:8
  s_waitcnt vmcnt(0)
  v_fma_mix_f32 v5, v3, v4, v5 op_sel_hi:[1,1,0]
  v_fma_mix_f32 v5, v3, v4, v5 op_sel:[1,1,0] op_sel_hi:[1,1,0]
  global_store_dword v1, v5, s[2:3]
  s_endpgm
