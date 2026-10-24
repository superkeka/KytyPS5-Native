// Reads one image through its T# with the texture unit (so DCC is decoded exactly as a sampler sees
// it) and writes the texels as linear 16-byte RGBA. wave64, 8x8 threads per group.
// s[0:7] T#, s[8:9] destination, s10/s11 width/height of the region read, s12 slice (array and 3D
// variants), s13/s14 the region's first texel, s16/s17 group id x/y; v0/v1 thread id x/y.
  s_lshl_b32 s18, s16, 3
  s_lshl_b32 s19, s17, 3
  v_add_nc_u32 v2, s18, v0
  v_add_nc_u32 v3, s19, v1
  v_cmp_gt_u32_e64 s[20:21], s10, v2
  v_cmp_gt_u32_e64 s[22:23], s11, v3
  s_and_b64 s[20:21], s[20:21], s[22:23]
  s_and_saveexec_b64 s[22:23], s[20:21]
  s_cbranch_execz done
  v_add_nc_u32 v12, s13, v2
  v_add_nc_u32 v13, s14, v3
  v_mov_b32 v14, s12
  IMAGE_LOAD
  v_mul_lo_u32 v5, v3, s10
  v_add_nc_u32 v5, v5, v2
  v_lshlrev_b32 v5, 4, v5
  s_waitcnt vmcnt(0)
  global_store_dwordx4 v5, v[8:11], s[8:9]
done:
  s_endpgm
