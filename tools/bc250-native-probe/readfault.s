// s[0:1] = destination. Each lane loads the dword at VA 0, which has no mapping (the base of a null
// descriptor), and stores it with three markers at s[0:1] + 16 * lane.
  v_lshlrev_b32 v1, 4, v0
  v_mov_b32 v4, 0
  v_mov_b32 v5, 0
  global_load_dword v6, v[4:5], off
  s_waitcnt vmcnt(0)
  v_mov_b32 v7, 0x11111111
  v_mov_b32 v8, 0x22222222
  v_mov_b32 v9, 0x33333333
  global_store_dwordx4 v1, v[6:9], s[0:1]
  s_endpgm
