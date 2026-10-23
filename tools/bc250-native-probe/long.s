// s[0:1] = progress address, s2 = run time in REFCLK ticks (s_memrealtime, 100 MHz on BC-250).
// One wave spins until the time is up, publishing the elapsed ticks at +0 on every
// iteration and writing 1 to +8 when it finishes on its own.
  s_memrealtime s[4:5]
  s_waitcnt lgkmcnt(0)
  v_mov_b32 v1, 0
loop:
  s_sleep 127
  s_memrealtime s[6:7]
  s_waitcnt lgkmcnt(0)
  s_sub_u32 s8, s6, s4
  s_subb_u32 s9, s7, s5
  v_mov_b32 v2, s8
  v_mov_b32 v3, s9
  global_store_dwordx2 v1, v[2:3], s[0:1] glc
  s_cmp_lt_u32 s8, s2
  s_cbranch_scc1 loop
  v_mov_b32 v4, 1
  global_store_dword v1, v4, s[0:1] offset:8 glc
  s_endpgm
