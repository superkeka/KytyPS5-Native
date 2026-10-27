# PS4 games on the PS5 GPU ("legacy mode") and what of it applies to the BC-250

How the PS5 runs PS4 (GCN, gfx7) titles on its RDNA GPU, and whether the BC-250 (the same Oberon
silicon, `1002:13fe`, gfx1013) can do the same for Kyty's native backend. Public sources and AMD's
own headers, plus conclusions from studying the PS5 kernel (no Sony code or data here). Guesses are
marked **guess**.

## Summary

- The PS5 does not recompile PS4 shaders. Compatibility is a hardware mode organised as GPU
  **generations**: gen0 = PS4, gen1 = PS4 Pro, gen2 = PS5 (gen3 = PS5 Pro). To switch, the kernel stops
  and resets the CP, writes the generation into GRBM registers absent from Linux's headers, and loads
  **a separate CP microcode (PFP, ME, CE, MEC) for the older generation**. The system compositor's pipe
  stays native while the game's pipe runs the old generation. Clocks ("Boost mode") come from SMU
  power profiles, not from the GPU block. The kernel writes no CU masks on a switch.
- AMD's public gfx10.1 headers keep traces of it, gone in gfx10.3: `SQ_WAVE_HW_ID2.COMPAT_LEVEL`
  (bits 30:29, a per-wave compatibility level), `SQ_WAVE_HW_ID_LEGACY` (a GCN-style HW_ID layout) and
  `CC_GC_SHADER_ARRAY_CONFIG_GEN0.GEN0_INACTIVE_CU` (a separate inactive-CU mask for gen0). **Guess**: in
  gen0 the hardware itself halves the CUs from that fuse mask, and SQ decodes GCN ISA.
- A gfx7 binary does not run on the gfx10 decoder as is: many encodings match, but 13 VOP2 opcodes
  mean other instructions (silent corruption, not a fault), scalar memory is encoded differently
  (SMRD vs SMEM), descriptors are laid out differently, and stores need the separate `vscnt` wait.
- For the BC-250 the PS5 way needs the old-generation CP microcode, which only Sony's kernel has: that
  rules it out for boards without Sony files, besides the risk of poking the CP. Whether the legacy
  logic survives in BC-250 silicon is unknown; read-only checks below would tell.

## Public facts

- Mark Cerny, "The Road to PS5" (2020): the logic of the previous consoles is built into the custom
  chip; it has a native mode and PS4 / PS4 Pro legacy modes; boosted clocks need per-title testing.
- The 2019 Oberon test-data leak lists Gen2 (native, 36 CUs), Gen1 (36 CUs, 911 MHz: PS4 Pro) and Gen0
  (half the CUs and ROPs, 800 MHz: PS4).
- Patents US10235219B2 (disabling or slowing new hardware features for legacy applications, mostly
  CPU) and US10275239B2 (per-title clock and parameter selection).
- Linux `gc_10_1_0_sh_mask.h` / `gc_10_1_0_offset.h`: the fields above, and legacy/generation bits such
  as `PA_SC_ENHANCE_1.DISABLE_SC_PIPELINE_RESET_LEGACY_MODE_TRANSITION`,
  `PA_SC_BINNER_EVENT_CNTL_3.ENABLE_LEGACY_PIPELINE`, `PA_SC_ENHANCE_2.ENABLE_BLOCKING_WRITES_OF_GEN1/GEN2_REG`,
  `TD_CNTL.PRECISION_COMPATIBILITY`, `GE_PRIV_CONTROL.DISCARD_LEGACY` (their link to backward
  compatibility is a **guess**).
- PS4 emulators (shadPS4, Kyty's PS4 side) do not use a hardware mode: they recompile GCN to SPIR-V.

## gfx7 binaries on gfx10 (LLVM `GCNFeatures.td`, `VOP*Instructions.td`)

- gfx1013 = gfx10.1 plus BVH and MSAA loads. gfx10.1 still has some GCN-era instructions that 10.3
  dropped (`ds_*_src2`, `v_mad/mac_f32`, `v_mac/mul_legacy_f32`, VINTRP); wave64 is native.
- Shared encodings are many (LLVM's `_gfx6_gfx7_gfx10` multiclasses), but VOP2 reuses 13 opcodes, e.g.
  0x25 `v_add_i32` (carry out) is `v_add_nc_u32` (no carry), 0x2b `v_ldexp_f32` is `v_fmac_f32`, 0x01/0x02
  `v_readlane/v_writelane` are `v_cndmask`/`v_dot2c`.
- Removed in gfx10: `v_*_clamp`, `v_rcp/rsq_legacy`, `v_min/max_legacy_f32`, non-rev shifts,
  `v_cmps*`, `v_mul_lo_i32`, `s_cbranch_join/g_fork/i_fork`, MUBUF `addr64`, `buffer_wbinvl1`,
  `buffer_*_lds`. MIMG's DA bit became DIM, NSA was added.
- Outside the ISA: `s_waitcnt vmcnt(0)` no longer waits for stores (`vscnt`), a GL1 cache and glc/dlc
  bits, different V#/T# layouts (PS4 titles build them on the CPU through Gnm), different hazards.

## What it means for Kyty on the BC-250

- Hardware path (as the PS5): write the generation, load legacy CP microcode, rely on SQ/TA running
  gfx7 ISA and descriptors. Needs Sony's microcode (excluded for boards without Sony files), unknown
  fuse state, and touching the CP with `gpu_recovery=0` likely ends in a hard reset. High research
  interest, low practical value.
- Software path: shadPS4 (GCN to SPIR-V on RADV) already runs PS4 titles on the BC-250 with nothing
  from Sony. For the native backend: static gfx7 to gfx10 translation (SMRD to SMEM, the 13 VOP2
  remaps, expanding removed instructions, `s_waitcnt_vscnt`, descriptor rewriting at bind) plus Gnm
  PM4 to gfx10. Feasible but a project of its own, and PS4 titles are light for this board anyway.

## Read-only checks on the BC-250 (`tools/bc250-lodhw/kyty-hw.py`)

1. `kyty-hw.py read 0x203a 2` and `kyty-hw.py read 0x2048 8`: with the legacy logic present, bits
   [1:0] should read 2 (the native generation) after reset; zeros or identical noise suggest the
   registers are absent or disabled.
2. `kyty-hw.py read 0x226b` (`CC_GC_SHADER_ARRAY_CONFIG_GEN0`, bits 25:16) next to `read 0x226f`
   (`CC_GC_SHADER_ARRAY_CONFIG`): a sensible gen0 mask (half the CUs of an SA off) would mean gen0 is
   fused in. Only the current SE/SA is read without a `GRBM_GFX_INDEX` selection.
3. `kyty-hw.py read 0xf81e 5`: CP microcode versions (PFP, CE, ME, MEC1, MEC2); they should match
   amdgpu's `cyan_skillfish2_*` firmware versions.
4. A compute shader through the native backend reading `s_getreg_b32 HW_REG_HW_ID2` (id 24): the
   native wave's `COMPAT_LEVEL` [30:29]; and a raw `s_getreg` with id 4 (GCN-style HW_ID).

Writing `0x203a`/`0x2048..` is deliberately not listed: without legacy CP microcode it means nothing,
and with it needs Sony files and risks a hard reset.

## Sources

- The Road to PS5: https://www.youtube.com/watch?v=ph8LyNIT9sg ;
  https://blog.playstation.com/2020/03/18/unveiling-new-details-of-playstation-5-hardware-technical-specs/
- Oberon Gen0/1/2 leak: https://www.notebookcheck.net/Remarkable-seemingly-accurate-PS5-Xbox-Series-X-specs-leak.448890.0.html
- Patents: https://patents.google.com/patent/US10235219B2/en , https://patents.google.com/patent/US10275239B2/en
- LLVM AMDGPU: https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/AMDGPU/GCNFeatures.td
- Linux: https://github.com/torvalds/linux/blob/master/drivers/gpu/drm/amd/include/asic_reg/gc/gc_10_1_0_sh_mask.h
- shadPS4: https://github.com/shadps4-emu/shadPS4
