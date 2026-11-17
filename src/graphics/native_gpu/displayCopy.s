// The display copy (Pm4Translator::SetDisplayCopies): each thread copies one texel of a 2D image to the same place in
// another. Both are reached through texture descriptors, so the texture unit decodes what the source holds (DCC) and
// the destination gets plain texels. gfx1013, wave64, 8 x 8 threads a group.
//
// In: s[0:1] = two descriptors of 8 dwords each (the source, then the destination), s2 and s3 = the source
// descriptor's last two dwords (its DCC, which follows how the title last drew the buffer; 0 and 0 for none), s4 and
// s5 = the group's x and y, v0 and v1 = the thread's x and y within its group.
//
// displayCopyShader.inc is this file assembled: gen_display_copy.sh.
  s_load_dwordx8 s[8:15], s[0:1], 0
  s_load_dwordx8 s[16:23], s[0:1], 32
  s_lshl_b32 s4, s4, 3
  s_lshl_b32 s5, s5, 3
  v_add_nc_u32 v0, s4, v0
  v_add_nc_u32 v1, s5, v1
  s_waitcnt lgkmcnt(0)
  s_mov_b32 s14, s2
  s_mov_b32 s15, s3
  image_load v[2:5], v[0:1], s[8:15] dmask:0xf dim:SQ_RSRC_IMG_2D unorm
  s_waitcnt vmcnt(0)
  image_store v[2:5], v[0:1], s[16:23] dmask:0xf dim:SQ_RSRC_IMG_2D unorm
  s_endpgm
