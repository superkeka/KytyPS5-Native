// s[0:1] = destination, s2 = base value, s3 = workgroup id (64 threads per group)
  v_lshlrev_b32 v1, 2, v0
  s_lshl_b32 s4, s3, 8
  v_add_nc_u32 v1, s4, v1
  s_lshl_b32 s5, s3, 6
  s_add_u32 s5, s5, s2
  v_add_nc_u32 v2, s5, v0
  global_store_dword v1, v2, s[0:1]
  s_endpgm
