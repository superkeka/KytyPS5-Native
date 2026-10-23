// s[0:1] = flag address, s2 = expected value, s3 = max iterations, s[4:5] = result address
// Lane 0 of a single wave polls the flag and stores the iteration count (or ~0 on timeout).
  v_mov_b32 v1, 0
  s_mov_b32 s6, 0
loop:
  global_load_dword v2, v1, s[0:1] glc dlc
  s_waitcnt vmcnt(0)
  v_readfirstlane_b32 s7, v2
  s_cmp_eq_u32 s7, s2
  s_cbranch_scc1 done
  s_sleep 1
  s_add_u32 s6, s6, 1
  s_cmp_lt_u32 s6, s3
  s_cbranch_scc1 loop
  s_mov_b32 s6, -1
done:
  v_mov_b32 v3, s6
  global_store_dword v1, v3, s[4:5]
  s_endpgm
