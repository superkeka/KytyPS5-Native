// s[0:1] = destination. Every lane atomically adds 1 to GDS dword 0 and stores the value it
// got back, so a working GDS yields each of 0..63 exactly once.
  s_mov_b32 m0, 0x1000
  v_mov_b32 v1, 0
  v_mov_b32 v2, 1
  ds_add_rtn_u32 v3, v1, v2 gds
  s_waitcnt lgkmcnt(0)
  v_lshlrev_b32 v4, 2, v0
  global_store_dword v4, v3, s[0:1]
  s_endpgm
