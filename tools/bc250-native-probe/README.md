# bc250-native-probe

Checks what the stock amdgpu kernel driver and firmware let a user-mode process do on the
BC-250, as groundwork for a native GPU backend: guest PM4 and guest shader ISA submitted
directly, guest memory mapped at the same GPU and CPU addresses. Raw ioctls, no libdrm.

Only safe cases are implemented. None of them is expected to fault, hang or reset the GPU.
Cases that deliberately fault, time out or write privileged registers are not here yet and
must not be added or run without the owner's explicit approval: on this board a GPU reset
can turn into a hard reset.

## Build and run (on the BC-250)

The SteamOS root has no kernel UAPI headers, so build inside the `bc250-mesa-dev` distrobox:

```sh
distrobox enter bc250-mesa-dev -- make
./bc250-native-probe                 # all probes
./bc250-native-probe info va dmabuf  # no GPU execution
```

`eop_irq_probe` (`make eop_irq_probe`) checks that a `RELEASE_MEM` with `INT_SEL 2` on the gfx ring
and on amdgpu compute ring 0 wakes the KFD event its context id names, with the patched amdgpu
(`bc250_gfx_eop_events=1`, `tools/bc250-amdgpu`). It also checks that `INT_SEL 0` and another event's
id do not wake it, and that 32 interrupts in one IB leave nothing hanging. It prints submit-to-wake
against submit-to-label times. With the EOP log it checks that a title's context ids from the gfx ring
and compute ring 0 land in the log in order, with their ring. 2026-10-27, 7.2.7-valve2: gfx wakes 44 us after submit (label at 25 us),
compute 73 us (label at 48 us), none lost in 300. The stock module wakes none.

`scanout_probe` (`scanout-probe.sh [vram|gtt] [seconds]`, inside the gamescope session) shows two
3840x2160 buffers laid out as the PS5's render targets (Prospero kRenderTarget, SW_64KB_R_X) through
X11 DRI3/Present with the AMD modifier the display planes list for that layout (GFX10 64K_R_X,
PIPE_XOR_BITS 4, `0x0200000000801b02`), one flip per vblank. 2026-10-28: the picture is right (the PS5
layout is the display engine's 64K_R_X), Present completes every frame as FLIP (Xwayland passes the
buffer to gamescope without a copy), 16.7 ms apart. gamescope composites it (the plane shows its own
three framebuffers) rather than scanning it out directly; why is still open.

`ring_share_probe` checks whether the patched amdgpu's two gfx rings (`bc250_gfx_queues=2`, both on
pipe 0) run past each other's waits. 2026-10-28: they do not. A label written on ring 0 lands only
after ring 1's WAIT_REG_MEM is released, so a GPU wait in a title's command buffer holds up every
other gfx-ring user, Kyty's Vulkan presenter and gamescope's composition included (RADV gives this
board no compute queue family).

`shaders.h` is generated from the `.s` files by `gen_shaders.sh` (LLVM `llvm-mc`, gfx1013).

`--blob FILE [--vgprs N]` runs a raw shader binary as one wave64 group of 64 lanes (destination
address in `s[0:1]`) and prints the first two lanes' 4 dwords; `--vgprs` (default 16) sets the
dispatch's VGPR count, which must cover any scratch register the patched stubs use. `patch_blob
IN.bin OUT.bin NUM_VGPRS` applies the native_gpu shader patcher
(`src/graphics/native_gpu/shaderPatcher.*`) to such a binary; `NUM_VGPRS` is the VGPR count the
shader allocates (the stubs use `v[NUM_VGPRS]` as scratch for the literal/DPP/aliasing forms) and
it prints the minimum `--vgprs` the patched blob needs.

## Probes

| Probe | What it checks |
|---|---|
| `info` | topology, CU bitmap, VA ranges, timestamp frequency, rings, firmware |
| `va` | mapping at KytyPS5 guest addresses; MTYPE flags accepted |
| `alloc` | largest single GTT and userptr buffers (`--max-alloc-gib`, default 4) |
| `dispatch` | hand-written PM4 + gfx1013 ISA writing into VRAM/GTT/USWC/userptr, wave32/64; code fetched from userptr |
| `identity` | CPU pointer == GPU VA for GEM (`MAP_FIXED`) and userptr memory |
| `bandwidth` | GPU streaming read per memory kind and MTYPE |
| `coherency` | a running wave polls a flag the CPU writes mid-job |
| `dmabuf` | dma-buf export per memory kind |
| `submit` | CS ioctl cost and round trip |
| `ib2` | cost of `INDIRECT_BUFFER` calls from a user IB on the gfx ring; multi-dword NOPs inline and in an IB2 |
| `dot2emu` | `v_dot2c_f32_f16` replacement (two `v_fma_mix_f32`) against a CPU reference |

## Results (2026-10-23, SteamOS kernel 6.18.50-valve2, firmware ME 99 / PFP 148 / MEC 144)

- Topology seen by the kernel: 2 SE x 2 SA, `cu_active 24`, bitmap `0x3f` per SA (the live
  40-CU routing is below the driver). 1 gfx ring, 4 compute rings, 2 SDMA rings.
  `gpu_counter_freq` 100 MHz.
- GPU VA: low range `[0x10000, 0x800000000000)`. Every KytyPS5 guest address maps except
  page zero, including `0x200000000`, the PRT aperture, the user range up to `0xfbffe00000`
  and `kExtendedMemoryBase` (`0x80000000000`). All MTYPE flags are accepted on GTT.
- Raw PM4 + raw ISA work from user mode on the compute ring into every memory kind, in wave32
  and wave64, and code executes straight from userptr memory.
- Identity mapping works: CPU pointer and GPU VA equal for both GTT (`MAP_FIXED`) and userptr.
- Single buffers up to 4 GiB GTT and 4 GiB userptr allocate and map (4 GiB userptr: ~1 s).
- Streaming read (this shader's ceiling, not the bus peak): VRAM 233 GB/s, GTT 232 GB/s in
  every MTYPE, USWC 232 GB/s, userptr 200 GB/s default / 226 GB/s UC. GTT costs nothing
  over VRAM; snooped userptr costs ~14%.
- CPU writes become visible to a running wave without any invalidation in every case:
  immediately with `glc dlc` loads on GTT, within a few ms with cached loads.
- dma-buf export: VRAM and GTT ok, userptr `EPERM`. Presentable surfaces must be GEM buffers.
- Submission: 8.3 us per CS ioctl, 31 us round trip for an empty IB.
- No amdgpu messages in the kernel log and no context reset after any probe.

## Dangerous cases

Run one at a time with `--danger CASE`, only with the owner's approval.

| Case | What it does |
|---|---|
| `vmfault` | one wave64 stores 256 bytes to the unmapped VA `0x900000000`, then a normal store checks the GPU |
| `vmfault-read` | one wave64 loads a dword at the unmapped VA `0` (a null descriptor's base) and stores it, then a normal store checks the GPU |
| `long` | one valid wave64 spins 12 s, publishing elapsed GPU time |
| `sqkill` | a wave that never ends on its own (`SQKILL_SHADER=long`, spinning 30 s, killed after `SQKILL_AFTER_MS`; or `readfault`, halted by a page fault at VA `0`, killed as soon as the fault query shows it), killed through the kyty-lodhw helper's op 17 (SQ_CMD KILL for the probe's own VMID) on `SQKILL_RING=compute` or `gfx`; passes when the job's fence signals within 0.5 s with no amdgpu reset |
| `dot2c` | `v_dot2c_f32_f16` (gfx10.3 encoding) with known inputs |
| `dot2x` | same opcode as raw words with several accumulators/inputs, plus a control store |
| `gds` | GDS/GWS/OA allocation, then 64 lanes of GDS `ds_add_rtn_u32` |
| `crossqueue` | `WAIT_REG_MEM` on compute ring 1 released by `WRITE_DATA` on ring 0 |
| `uconfig` | **took the board down**, see below; do not rerun as is |
| `clock` | GPU clock reads on the gfx ring: `COPY_DATA` SRC_SEL 9, `RELEASE_MEM` DATA_SEL 3 (two events) |
| `tag342` | libSceAgc's internal-data tags (`SET_UCONFIG_REG` of 0x342) on the gfx and a compute ring |
| `condib` | libSceAgc's conditional branch (`COND_INDIRECT_BUFFER`, 14 dwords) on the gfx ring: then, else and none taken, with and without VALID (`PROBE_CONDIB_VALID=0`) in the sizes |
| `atomicmem` | libSceAgc's `ATOMIC_MEM` around a context push or pop (atomic 0x67, command 2) on the gfx ring: the flags swap, a `WRITE_DATA` after it lands |
| `atomicops` | every no-return integer `ATOMIC_MEM` TC_OP the translator passes with `KYTY_NATIVE_GPU_ATOMIC_OPS=1` (SWAP, CMPSWAP, ADD, SUB, SMIN, UMIN, SMAX, UMAX, AND, OR, XOR, INC, DEC; 32 then 64 bits) in libSceAgc's form, one op per job on a word with a known old value, its neighbours checked, a `WRITE_DATA` marker after it; `ATOMIC_OP=<hex>[,..]`, `ATOMIC_CMD=0\|1\|2` (default 2; 1 only runs CMPSWAP), `ATOMIC_ENGINE=0\|1` (default 1 as libSceAgc's DCB form). Gfx ring only: the translator never sends `ATOMIC_MEM` to MEC. 2.11: all 26 ops right with engine 0 and 1 (COMMAND 2); CMPSWAP with `ATOMIC_CMD=1` stopped the gfx ring (amdgpu reset it, board fine) |
| `copydata` | `COPY_DATA` as libSceAgc writes it, one job per step with a marker after it: `mem` (SRC_SEL 2 to DST_SEL 2, 32 bits), `mem64`, `memobs` (SRC_SEL 1 to DST_SEL 5, Mesa's selects), `imm`, `imm64`, `clock` (SRC_SEL 9 twice, non-zero and ordered); `COPYDATA_STEP=<list>\|all`, `COPYDATA_RING=gfx\|compute` (compute ring 1; a stopped MEC can take the board down, so one compute step per run, last), `COPYDATA_DST=2\|5` (or `1` on gfx) for the immediate and clock steps, `COPYDATA_ENGINE=0\|1` (gfx: 1 is the PFP, as libSceAgc's DCB form may set it). Never a register select. `clock` runs only with `--awake` (`COPYDATA_CLOCK_IDLE=1` overrides). 2.11: every step right on gfx (engines 0 and 1, DST_SEL 2 and 5) and on compute ring 1 |
| `memsem` | `MEM_SEMAPHORE` 0x39 (libSceAgc's form, mailbox 0) on the gfx ring. `MEMSEM_STEP=signal` (default): two signals, then a wait in a second job; the 64-bit counter reads 2, then 1 (`MEMSEM_SIGNAL_TYPE=1`: 1, then 0). `MEMSEM_STEP=waitfirst`: a wait on 0 must block 1 s, then finish within 100 ms of the CPU writing 1 (else a compute-ring `WRITE_DATA` writes it through GL2). **May stop the gfx ring** (an illegal opcode or an endless wait: amdgpu resets it after its lockup timeout); run `signal` first, check dmesg. 2.11: the first `signal` took the **whole board** down at once (no kernel line, power cycle); the case now needs `MEMSEM_I_KNOW=1`. Never run it again without a reason |
| `mecforms` | libSceAgc's ACB packets without the shader-type bit on the compute ring: store dispatch, `EVENT_WRITE` + `ACQUIRE_MEM`, `DMA_DATA`, `WAIT_REG_MEM_64`, `RELEASE_MEM` |
| `condexec` | `COND_EXEC` on an abort word (gfx ring, compute ring 1): word 1 and 0, an IB2 call guarded, EXEC_COUNT 20000 and 70000, the word cleared while an ME wait (with and without `PFP_SYNC_ME`) or a PFP wait ahead of it waits; one step per run with `CONDEXEC_STEP` |
| `condexec2` | a title's `COND_EXEC` (AgcDcbCondExec's form) whose EXEC_COUNT covers a `WRITE_DATA` and a store dispatch, on the gfx ring, with the word 1 (both run) and 0 (neither runs), a `WRITE_DATA` after the range landing either way; `CONDEXEC2_WORD=1\|0` runs one. 2.11: as expected with 1 and 0 |
| `exitguard` | Kyty's exit guard: `COND_EXEC` over an `INDIRECT_BUFFER` chain to an empty IB; `GUARD_STEP` 0-3 on the gfx ring. **Steps 4-5 (compute ring) took the board down; do not run them** |
| `waitforms` | the guest's `WAIT_REG_MEM` / `WAIT_REG_MEM_64` forms (ME and PFP, EQUAL and GREATER_EQUAL, met and released by the CPU) on the gfx ring |
| `regindex-sh`, `-shc`, `-ctx`, `-uc` | 0x63 / 0x9F / 0x64 (AMD's `LOAD_*_REG_INDEX`) with one pair, read back with `COPY_DATA`; `shc` loads a compute register from a graphics-type header |
| `regindex-uc342`, `-sh-nr`, `-ctx-nr` | the same packets without register reads, a `WRITE_DATA` after them as the proof |
| `labelvis` | a `RELEASE_MEM` label without cache write-back seen by the CPU, then a `WAIT_REG_MEM` on a dword the CPU writes, per MTYPE (`LABELVIS_KIND=nc\|uc\|cc\|wc\|default`, one per run; `LABELVIS_DST=1` sends the label to L2, `LABELVIS_EVENT` sets the event dword) |
| `shadow` | CP register shadowing: `CONTEXT_CONTROL` shadow enables and `LOAD_*_REG` addresses, then 16 registers written by SET, by register table, by SET in an IB2 and by register table in an IB2; where each value lands in the shadow buffer (`SHADOW_CLASSES`, `SHADOW_LOAD=0`) |
| `shadow-restore` | `shadow`, then the SH registers loaded back from the shadow in later jobs and read: with every load on, and twice with amdgpu's load enables (a second DRM context's job in between) |
| `shadow-pushpop` | libSceAgc's context push and pop (`CONTEXT_CONTROL` + `CLEAR_STATE` CMD 1 / 2) inside a set-up shadow: markers in the shadow before each step, the same `CONTEXT_CONTROL`s without `CLEAR_STATE`, and an SH register changed between a push and a pop and read back |
| `prt-map` | P-0, no GPU work: a fresh VM gets Kyty's PRT net windows (`[64 KiB, 1 TiB)`, `[8 TiB, +512 GiB)`) by REPLACE PRT; ioctl and page-table update times, VRAM and fdinfo before and after, against 8192 PRT pages of 4 KiB; then 2000 always-valid binds of 16-64 KiB into the low window and back to PRT, without and with the pre-split |
| `prt-split` | P-6a, no GPU work, no CS: a 64 KiB buffer bound in the middle of a 1 GiB PRT mapping without and with a PRT pre-split; prints the addresses to decode with umr and holds the VM (`PRT_HOLD_SEC`, default 120) |
| `prt-read-compute` | P-1: global, flat, scalar, buffer (raw V#) and image (linear 1D T#) loads of PRT in a 1 GiB entry, a 2 MiB one, a 4 KiB page, the rest of a 1 GiB mapping a buffer was bound into (pre-split; `PRT_PRESPLIT=0` without), and the buffer; all 0 with no fault (`PRT_LOADS` picks the loads) |
| `prt-write-compute` | P-2: `global_store`, `buffer_store` and `global_atomic_add` with return into PRT, a load back (0); then a pattern buffer replaces the PRT there and must read intact |
| `prt-kfd-tlb` | P-5: a wave on a KFD queue logs reads of X while the render node switches X buffer / PRT and bumps an epoch after each page-table update: stale reads and their duration with no flush, a KFD map ioctl flush, or `PRT_TLB_FLUSH_CMD` (a shell command, e.g. the kyty-lodhw op-14 flush) |
| `prt-replace-race` | P-7: four wave64 groups read 64 KiB round after round while the CPU switches it always-valid buffer / PRT `PRT_RACE_SWITCHES` times (default 10000, 4 s at most): a fault is the REPLACE window (compute ring only) |
| `prt-cp-compute` | P-3, compute ring, one job a step (`PRT_CP_STEPS`, default `copy,dma,write,release,wait,indirect`): `COPY_DATA` and `DMA_DATA` from PRT (0), `WRITE_DATA` and `RELEASE_MEM` into it, `WAIT_REG_MEM == 0` on it, `DISPATCH_INDIRECT` with its arguments in it (0 groups); a marker after each |
| `prt-gfx-copy`, `-dma`, `-write`, `-wait`, `-dispatch` | P-4, **gfx ring, one step per run, last**: the same packets, and `dispatch` a compute wave loading PRT from the gfx ring. Draws (CB, DB, index and vertex fetch, texture sampling on PRT) are not covered: the probe has no draw harness |

`--awake SEC` keeps one wave spinning on the compute ring while the case runs, so the graphics block
never idles into GFXOFF. Cases that stop the ring now `_exit` at once instead of freeing buffers
(which waits on the stopped ring).

`shadow`, `shadow-restore` and `shadow-pushpop` check whether the CP shadows register writes into memory itself, as PAL
and Mesa use it for mid command buffer preemption and the PS5's CP for libSceAgc's context push and pop. The first job
turns shadowing on (`CONTEXT_CONTROL` 0x81018002 / 0x81018002), points the CP at a zeroed buffer S with `LOAD_UCONFIG_REG`,
`LOAD_CONTEXT_REG` and `LOAD_SH_REG` (Mesa's layout: SH space at +0, context at +0x1000, UCONFIG at +0x9000; one range per
test register), writes 16 harmless context, SH and UCONFIG registers by SET, by a one-pair 0x9F / 0x63 / 0x64 table as
libSceAgc writes them, by a SET in an IB2 and by such a table in an IB2 (as direct runs execute titles' tables), and turns shadowing off again (as every job of the three does before it
ends). The case then prints where each value landed in S against `S + region + (register - space start)`, what the
LOADs left in the SH registers (the CPU's sentinels in S: loaded) and anything else the CP wrote. SH registers are read
back with `COPY_DATA`; context and UCONFIG registers never are (those reads stop the ring). `SHADOW_CLASSES=ctx,sh,cs,uc`
picks the shadow enables; `SHADOW_LOAD=0` adds LOADs from a second region set 1 MiB in with loading off (does such a
LOAD still move the address?). `shadow-restore` then clobbers the SH registers and loads them back from S in a second
job, then twice more with the load enables left as amdgpu set them (on only after another context ran on the ring): right
after our own job and after a second DRM context's job. `shadow-pushpop` writes markers into S before libSceAgc's push
(`CONTEXT_CONTROL` 0 / 0x80018001, `CLEAR_STATE` 1), before its pop (0 / 0x80018003, `CLEAR_STATE` 2) and before a plain
`CLEAR_STATE`, and copies S out after each step and after a SET between them; then the two `CONTEXT_CONTROL`s alone; then
an SH register set with shadowing on, a push that also turns gfx SH shadowing off (0x80008001), a new value, the pop,
the register read back, and the same without `CLEAR_STATE`. Whether a pop restores context registers takes a draw and
is not checked. Run each alone, e.g. `./bc250-native-probe --danger shadow --awake 5`; a microcode that writes the
shadow somewhere else may fault the ring.

### PRT cases (for Kyty's `KYTY_NATIVE_GPU_PRT_NET`, not run yet)

`GEM_VA` with `AMDGPU_VM_PAGE_PRT` and no buffer maps a range whose gfx10 page-table entries have PRT set and VALID
clear: a client that honours it reads zeros and drops writes, one that does not faults as at an unmapped address
(never worse than today's unmapped VA). Kyty's PRT net maps the guest's GPU windows that way so a stray GPU access reads
zeros instead of halting the ring; these cases check each client before the net's default goes on. Run them one at a
time in this order, with `dmesg | tail` and a ping after each and `bc250-power cycle` at hand; stop at the first that
misbehaves:

```sh
./bc250-native-probe --danger prt-map
./bc250-native-probe --danger prt-split          # then umr in a second shell while it holds (PRT_HOLD_SEC=120)
./bc250-native-probe --danger prt-read-compute
./bc250-native-probe --danger prt-write-compute
./bc250-native-probe --danger prt-kfd-tlb        # PRT_TLB_FLUSH_CMD='...' for the op-14 flush in step 4
./bc250-native-probe --danger prt-replace-race
./bc250-native-probe --danger prt-cp-compute
./bc250-native-probe --danger prt-gfx-copy --awake 5     # gfx ring: one per run, last, each may take the board down
./bc250-native-probe --danger prt-gfx-write --awake 5
./bc250-native-probe --danger prt-gfx-wait --awake 5
./bc250-native-probe --danger prt-gfx-dma --awake 5
./bc250-native-probe --danger prt-gfx-dispatch --awake 5
```

`prt-split` creates no CS, so its VM has no VMID; umr has to walk it from its page directory (the held process's
pasid is printed). Without the pre-split, the rest of the 2 MiB and of the 1 GiB around the buffer should read as
PTE 0 until a CS or a PRT op rewrites them; with it, as PRT.

### Register shadowing results (2026-10-31, stock amdgpu, kernel 7.2)

- `shadow`: the CP shadows every one of the 16 writes at `S + region + (register - space start)`, Mesa's layout, for
  SET, for the 0x9F / 0x63 / 0x64 tables, for a SET in an IB2 and for the tables in an IB2. `SQ_THREAD_TRACE_USERDATA_2`,
  on PAL's never-shadowed list, is shadowed too. The LOADs loaded the CPU's sentinels into the SH registers; nothing
  written with shadowing off reached S, and the CP wrote nothing else into S.
- `SHADOW_CLASSES=sh` / `=cs`: the class follows the packet's shader type, not the register. A compute register written
  by a graphics-type 0x63 (as libSceAgc's DCB dispatches) is shadowed under gfx SH (bit 16) and not under CS SH (bit
  24); SETs with the compute-type header are shadowed under CS SH only. The place in S is the register's either way.
- `SHADOW_LOAD=0`: LOADs with loading off neither load nor move the shadow address (the writes still land in the first
  region set). The CP takes a class's address only from a LOAD that loads.
- `shadow-restore`: a later job with every load on loads all 6 SH registers back with the values the CP shadowed; so
  does one with amdgpu's load enables after another DRM context's job. One with amdgpu's load enables right after our
  own job leaves no trace at all, neither its register reads nor its marker, though its fence signals (twice in two
  runs; not understood). A preamble sets its load enables itself.
- `shadow-pushpop`: the push, the pop and a plain `CLEAR_STATE` leave S untouched (every marker survives). Context
  shadowing goes off at the push and on again at the pop, and the two `CONTEXT_CONTROL`s alone do the same, so it is
  their shadow enables, not `CLEAR_STATE`. The pop does not load an SH register back from S: it keeps the value set
  between push and pop, as without `CLEAR_STATE`. Through the shadow, libSceAgc's push and pop save and restore nothing
  on this microcode; whether the CP keeps context registers of its own between them takes a draw to see. A translator
  can do it with the shadow itself: copy the context region at a push, load it back at a pop.

### Direct-execution probes (2026-10-27, two gfx rings, `sched_policy=2`)

- **GPU clock reads stop an idle gfx ring.** `COPY_DATA` SRC_SEL 9 and `RELEASE_MEM` DATA_SEL 3
  stopped the ring each time from the probe, while Kyty's frames (GPU always busy) read the clock
  hundreds of thousands of times. With `--awake` all three forms work. GFXOFF is on
  (`ppfeaturemask` 0xfff7bfff has bit 0x8000). Earlier "this packet hangs" results from an idle
  board are suspect.
- **A `COPY_DATA` read of a UCONFIG register stops the ring** (VGT_OBJECT_ID, TEXTURE_GRADIENT_FACTORS,
  awake or not), and so did a read of PA_SC_SCREEN_SCISSOR_TL (context). SH registers read fine.
- **0x63, 0x64 and 0x9F all run** on the BC-250's microcode (`-nr` cases: the `WRITE_DATA` after them
  lands, the ring stays healthy). 0x63 is verified by value: SPI_SHADER_USER_DATA_PS_0 took the
  table's value and was restored. The older "0x63 stops the ring" result was not the packet.
- **`ib2`:** an IB2 call costs the CP about 0.3 us (0.285 us at 16 dwords, 0.37 at 64 and 256), the same
  from cached and write-combined GTT; a 64-dword NOP stream runs as fast inline as in one IB2.
  Multi-dword NOPs (Kyty never sent any) are skipped as expected.
- **`labelvis` (P5):** the CPU sees a GPU label written without write-back with every mapping (the label
  itself goes past the caches). A GPU `WAIT_REG_MEM` on a dword the CPU writes later **never sees it
  with the default mapping** (NC: the first poll caches the line in GL2; the ring stopped) and sees it
  with UC or CC. CC streams as fast as NC (257 vs 232 GB/s), but ASTRO BOT with all guest memory CC
  costs the GPU ~25% (19-20 ms instead of 15-17); UC everywhere doubles My First Gran Turismo's GPU time.
- **`tag342`:** the gfx ring runs the DCB form (header 0xc0017904, bit 2 set); amdgpu's compute ring
  stops on `SET_UCONFIG_REG` of 0x342.

### `vmfault` result (2026-10-23, run once with approval)

- The write faults (`GCVM_L2_PROTECTION_FAULT_STATUS 0x00541050`, client TCP, `RW=1`) and the
  wave does **not** retire: on this setup a write to an unmapped page hangs the job instead of
  being dropped.
- After ~10 s amdgpu times the job out (`ring comp_1.1.0 timeout`), writes a devcoredump and
  resets only that compute queue (`Ring comp_1.1.0 reset succeeded`, "device wedged, but
  recovered through reset"). No full GPU reset, no board reset; the display and other
  processes were unaffected.
- `AMDGPU_INFO_GPUVM_FAULT` reports the fault to user space (`addr 0x900000000`), so a native
  backend can name the bad guest address itself.
- The follow-up store on the same context and a full `dispatch` + `submit` re-run passed.

For the backend this means a single bad guest address costs a ~10 s freeze of that queue, not
the machine: every guest range must be mapped before submission, and the fault query plus the
devcoredump are the debugging path.

### `vmfault` and `vmfault-read` with and without retry (2026-10-31)

- The same for the store and the load, with `amdgpu.noretry=1` (retry off) and with the default `-1`
  (retry on for GC 10.1): one fault (client TCP; the load: `RW=0`, status `0x00701030`), the wave does
  not retire, the job times out after ~10 s and only its compute queue is reset.
- So `noretry` does not change what a fault costs: the faulting wave halts either way. On the gfx ring
  it is the whole board: Demon's Souls froze it right after this same fault (a TCP read of VA 0 from a
  gfx VMID). Every address a guest shader can reach has to be mapped.

### `sqkill` result (2026-11-02)

- Op 17 ends the probe's waves at once on both rings, with no queue reset and a clean context afterwards:
  spinning wave on compute 1.4 ms, on gfx 1.5 ms after the kill; a wave halted by a page fault at VA 0 on
  compute 0.7 ms, on gfx 1.2 ms after the fault query showed it (the fault itself is seen 4-5 ms after the
  submit). The follow-up store on the same context passed each time.
- So the halted wave above is not the end of the board: SQ_CMD KILL for the VMID retires it, the draw or
  dispatch behind it finishes and the job ends. gfx10's amdgpu has no soft recovery (gfx9's
  `gfx_v9_0_ring_soft_recovery` is this write), so nothing but the helper does it on the BC-250.

### `long` result

- A valid wave that runs 12 s is killed at 10.16 s of GPU time (`ring comp_1.1.0 timeout`,
  queue reset succeeded). The effective compute job limit is ~10 s although `lockup_timeout`
  reads 5000.
- After the queue reset the context reports `AMDGPU_CTX_QUERY2_FLAGS_RESET` and further
  submissions on it fail; a new context (new process) works normally. The backend must split
  long guest work or raise the timeout, and must recreate its context after a reset.

### `dot2c` / `dot2x` result

- BC-250 executes VOP2 opcode `0x02` (`v_dot2c_f32_f16` on gfx10.3, decoded by KytyPS5 for PS5
  shaders) without any fault, but always writes `0` to the destination, whatever the inputs or
  accumulator. LLVM's gfx1013 model, which lists no dot instructions, matches the hardware.
- So the PS5 dialect is not 1:1 with this GPU for this opcode: a native backend has to patch
  guest shaders, e.g. branch to a stub that does the same math with two `v_fma_mix_f32`
  (available on gfx1013). Every other VOP1/VOP2/VOPC/VOP3/VOP3P opcode that KytyPS5 decodes is
  valid for gfx1013 in LLVM.

### `gds` result

- GDS (4 KiB), GWS and OA allocate from user mode. 64 lanes of `ds_add_rtn_u32 ... gds` return
  64 consecutive values, so GDS atomics work; GDS is not zeroed on allocation.

### `crossqueue` result

- A `WAIT_REG_MEM` on compute ring 1 is released by a `WRITE_DATA` on compute ring 0 100 ms
  later (103 ms total), no reset: compute queues can wait on each other through memory.

### `uconfig`: took the board down (2026-10-23 13:28)

- Reads (`COPY_DATA` from register) and flipped-bit writes (`SET_UCONFIG_REG`, restored in the
  same IB) of UCONFIG registers that PS5 command buffers write and RADV does not. Output was
  block-buffered, so it is not known which register or which step was reached; the first one
  is `TA_CS_BC_BASE_ADDR` read via `COPY_DATA` on the compute ring.
- The board stopped answering ping within seconds and did not come back on its own.
- Do not rerun as is. Next attempt: line-buffered output and one register/step per process,
  starting with register reads on the gfx ring (as Mesa does) instead of the compute ring.
- After the power cycle, the crashed boot's journal ends at 13:28:31 (the build just before the
  run) with no amdgpu line at all: no timeout, reset or page fault. The whole system froze
  instantly instead of going through a GPU reset, so the kernel never logged anything.

### `dot2emu` result (safe probe)

- Two `v_fma_mix_f32` (low halves, then high halves, accumulating in place) reproduce
  `fma(a.hi, b.hi, fma(a.lo, b.lo, acc))` bit-exactly on 64 random lanes. They differ from a
  single-rounded dot by up to 52 ulp in cancellation-heavy lanes; how PS5 hardware rounds
  `v_dot2c_f32_f16` is unknown.

### Shader patcher on hardware (2026-10-23)

`dot2p.s` (three `v_dot2c_f32_f16`: VGPR src0, SGPR src0, zero inputs) assembled for gfx1030 and
run with `--blob`: unpatched every lane reads `0 0 0 0xabcd0123`; after `patch_blob` (3 sites
patched, 22 -> 48 dwords) every lane reads `0x41b40000 0x41b40000 0x41200000 0xabcd0123`, the
correct results. Unit tests: `native_gpu_shader_patcher_tests` (ctest `native_gpu_shader_patcher`).

The patcher also covers the forms a plain VOP2 rewrite cannot reach. `v_fma_mix_f32` is VOP3P, so
it takes no 32-bit literal and gfx10 VOP3P has no DPP; the literal (8-byte), DPP16/DPP8 and
vdst-aliasing forms therefore materialise the effective src0 — or a preserved copy of vdst — into
one scratch VGPR at the top of the stub before the two FMAs. The scratch register is `v[num_vgprs]`
(the first VGPR past the shader's own allocation, passed to `PatchDot2c`), so a patched site raises
the shader's VGPR count by one and the caller must widen `COMPUTE_PGM_RSRC1` VGPRS to match. The
stub shapes are:

- **literal src0**: `v_mov_b32 scratch, <literal>`, then the FMAs read `scratch` (the literal's
  packed f16 halves selected by op_sel, same as a register src0).
- **DPP16/DPP8 src0**: `v_mov_b32_dpp scratch, v<src0> <dpp>` with the original DPP control word
  copied verbatim (the control word is opcode-independent), so `scratch` holds the swizzled src0;
  vsrc1 is read straight, matching `v_dot2c` where only src0 is swizzled. DPP8 has no masks or
  modifiers and is always reproducible. A DPP16 word is only reproducible, and only then patched,
  when: row_mask and bank_mask (bits 31:28 / 27:24) are both `0xf` (otherwise masked-out lanes keep
  vdst in the original but the unmasked FMAs here would overwrite them from stale scratch); src0 and
  src1 neg/abs (bits 23:20) are clear (`v_mov_b32` applies neg/abs to the whole f32, not per f16
  half, and the FMAs read vsrc1 unmodified); and the lane select is a quad_perm (`dpp_ctrl <= 0xff`,
  always in range) or `bound_ctrl` (bit 19) is set (so a shift/rotate/broadcast reading an invalid
  lane substitutes 0 and writes, rather than disabling the lane and leaving vdst). Every other DPP16
  word is counted unsupported.
- **vdst aliasing** (`v_dot2c vN, vN, …` / `… , vN`): `v_mov_b32 scratch, vN` preserves the input
  that aliases the accumulator before the first FMA overwrites `vN`; one copy serves both operands
  of `v_dot2c vN, vN, vN`.

Still counted as unsupported: SDWA src0 (its dst_sel / sub-dword semantics cannot be reproduced by
a full-dword VOP3P write), a non-reducible DPP16 word (masks, neg/abs, or an out-of-range lane
select without bound_ctrl, as above), a literal or DPP src0 whose vsrc1 also aliases vdst (would
need a second scratch), and inline-constant src0 (unchanged).

`dot2p2.s` exercises all three on hardware (gfx1030, `patch_blob dot2p2.bin dot2p2.patched.bin 12`,
run with `--vgprs 13` — 3 sites patched, 27 -> 48 dwords, one scratch VGPR). Unpatched every lane
reads `0 0 0 0xabcd0123`; patched:

| dword | form | lane 0 | lane 1 | expected |
|---|---|---|---|---|
| +0 | literal src0 `10.0 + (1.5,2.0)·(3.0,4.0)` | `0x41b40000` | `0x41b40000` | 22.5 |
| +4 | DPP src0 `quad_perm:[1,0,3,2]` (swaps lane pairs) | `0x40c00000` | `0x40800000` | 6.0 / 4.0 |
| +8 | vdst aliasing `v5=8.0 (f16 0.0,2.5)`, `8.0 + 2.5²` | `0x41640000` | `0x41640000` | 14.25 |
| +12 | control | `0xabcd0123` | `0xabcd0123` | — |

The DPP lanes differ (6.0 vs 4.0) exactly as the quad swap predicts, so the swizzle is preserved,
not elided. Job signaled in 0.1 ms, context state `0x0`, no reset; CU bitmap unchanged after.

## Dead Cells PM4 census (2026-10-23, Vulkan-path baseline)

Dead Cells (PPSA15552 01.005.000, the owner's dump) on the `bc250-barycentric` branch boots to
the main menu at a steady 60 FPS on the BC-250 (about 170 draws per frame, 4 shaders, 2.1 GB of
GPU memory). 25 s with `--command-buffer-dump` gave 1266 DCBs (one per frame), all in guest
direct memory (`0x2_01c00000`), with this packet mix:

| Packet | Count | Native path |
|---|---:|---|
| `SET_SH_REG` 0x76 | 329075 | as is |
| `0x9F` (Kyty: SET_CONTEXT_REG_INDIRECT) | 164621 | Mesa: `LOAD_CONTEXT_REG_INDEX` (GFX8+), offset/value pairs in guest memory |
| `0x64` (Kyty: SET_UCONFIG_REG_INDIRECT) | 162225 | not in Mesa; same layout as 0x63/0x9F. Only loads `VGT_PRIMITIVE_TYPE`, `GE_CNTL`, `GE_USER_VGPR_EN` (all RADV-written); translate to plain `SET_UCONFIG_REG` on the CPU instead of trusting the opcode |
| `SET_UCONFIG_REG_INDEX` 0x7A | 162002 | `VGT_INDEX_TYPE` with index 2, standard |
| `DRAW_INDEX_2` 0x27 | 162002 | as is |
| `0x63` (Kyty: SET_SH_REG_INDIRECT) | 4640 | Mesa: `LOAD_SH_REG_INDEX` (GFX8+) |
| `ACQUIRE_MEM` 0x58 | 1269 | as is |
| `DISPATCH_DIRECT`, `EVENT_WRITE`, `CLEAR_STATE` | 6 / 4 / 1 | as is |
| Kyty markers in `IT_NOP`: `R_FLIP`, `R_RELEASE_MEM`, `R_WAIT_FLIP_DONE` | 1 each per frame | the CP skips NOPs; the backend must turn them into present, a real `RELEASE_MEM` and a CPU-side wait |

No other UCONFIG writes appear, so this title never touches the registers suspected in the
`uconfig` crash.

## gem_view_probe

`gem_view_probe.c` is a standalone check that guest direct memory can be real GTT buffers
(`src/graphics/native_gpu/gemBacking.h`) instead of userptr slices of a memfd. It covers:

- dma-buf mmap at an inner offset, mapped twice plus a whole-buffer view, with CPU aliasing
  between them;
- mprotect on a view;
- a GPU WRITE_DATA through the VM mapping, seen by every view;
- CS cost with the buffer listed.

```sh
distrobox enter kyty-dev -- sh -c "cc -O2 -o /tmp/gem_view_probe gem_view_probe.c && /tmp/gem_view_probe"
```

On 2026-10-23 every check passed. A CS listing a 256 MiB GTT buffer took 0.008 ms. The same
buffer as userptr costs a page-table walk on every submit, about 31 ms per frame for Dead Cells'
13.5 GiB backing.

## svm_probe

`svm_probe` checks KFD's shared virtual memory (`AMDKFD_IOC_SVM`) as a PS5-like memory model: a range
of the process's own memory (anonymous or a memfd) given to the GPU at the same addresses, used by
render-node submissions on the VM KFD acquired. Result (28.10.2026): the gfx and compute rings read
and write it and the CPU sees their writes, but the GPU loses write access wherever the CPU
write-protects a page: a GPU write there faults and stops the gfx ring. The ioctl's size must cover
the attributes (`AMDKFD_IOC_SVM + (n * 8 << _IOC_SIZESHIFT)`, as libhsakmt does), else it is EINVAL.

```
distrobox enter bc250-mesa-dev -- make svm_probe && ./svm_probe
```


### `condexec`, `exitguard`, `kfd_probe condexec|stopwait` (2026-10-31, stock amdgpu 7.2.7)

For Kyty's exit (Pm4Translator::SetExitGuard): how to stop a title's GPU work where it stands.

- `COND_EXEC` runs in a user IB on the gfx ring, on amdgpu's compute ring and on a KFD queue; it skips
  an IB2 call. EXEC_COUNT 20005 skips right, 70005 does not (the field is narrower).
- The PFP decides a `COND_EXEC` before an ME `WAIT_REG_MEM` ahead of it is released: the word cleared
  while the ME waits comes too late, the guarded write still lands. With `PFP_SYNC_ME` after the wait,
  or with a PFP wait, it is skipped.
- The gfx guard (`COND_EXEC` over a chain to 8 filler NOPs) leaves the IB and the job ends, also when
  the word is set while an ME (+`PFP_SYNC_ME`) or PFP wait waits.
- The same chain on amdgpu's compute ring (MEC) froze the whole board at once, with nothing in the
  kernel log. On MEC only `COND_EXEC` with a count.
- `kfd_probe stopwait destroy`: `DESTROY_QUEUE` of a KFD queue waiting on a label nobody writes takes
  0.8 ms and the queue never runs again; `stopwait exit`: the process exits with the queue waiting, KFD
  tears it down, the next process's queue works.
