// s[0:1] = destination, s2 = value. Every lane reads GDS dwords 0 and 1 and stores them at
// dst + lane * 8; then every lane writes s2 to GDS dword 2. GDS addresses are relative to the job's
// GDS partition (M0 window: base 0, size 4 KiB), so this is the reference the CP DMA is compared to.
  s_mov_b32 m0, 0x1000
  v_mov_b32 v1, 0
  ds_read_b32 v2, v1 gds
  ds_read_b32 v3, v1 offset:4 gds
  s_waitcnt lgkmcnt(0)
  v_mov_b32 v4, s2
  ds_write_b32 v1, v4 offset:8 gds
  s_waitcnt lgkmcnt(0)
  v_lshlrev_b32 v5, 3, v0
  global_store_dwordx2 v5, v[2:3], s[0:1]
  s_endpgm
