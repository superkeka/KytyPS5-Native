// Legacy VS of --danger multiinst (gfx10, wave64, VGPR_COMP_CNT 3): v0 = VertexID (the index fetched), v3 = InstanceID
// (gfx10 ES/VS VGPRs: VertexID, UserVGPR1, UserVGPR2 or VSPrimID, UserVGPR3 or InstanceID), s[0:1] = output.
// Every invocation takes a slot from the counter at out + 0 and writes {InstanceID, VertexID, 0xc0de, slot} at
// out + 64 + 16 * min(slot, 59), so repeated or unexpected invocations show up as records of their own. The position
// export (pos0, done) is what makes the VS legal; rasterization is killed, so it goes nowhere.
  v_mov_b32 v5, 0
  v_mov_b32 v6, 1
  global_atomic_add v4, v5, v6, s[0:1] glc
  s_waitcnt vmcnt(0)
  v_min_u32 v8, 59, v4
  v_lshlrev_b32 v10, 4, v8
  v_mov_b32 v7, 0xc0de
  global_store_dword v10, v3, s[0:1] offset:64
  global_store_dword v10, v0, s[0:1] offset:68
  global_store_dword v10, v7, s[0:1] offset:72
  global_store_dword v10, v4, s[0:1] offset:76
  v_mov_b32 v9, 1.0
  exp pos0 v5, v5, v5, v9 done
  s_endpgm
