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

`shaders.h` is generated from the `.s` files by `gen_shaders.sh` (LLVM `llvm-mc`, gfx1013).

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
| `long` | one valid wave64 spins 12 s, publishing elapsed GPU time |
| `dot2c` | `v_dot2c_f32_f16` (gfx10.3 encoding) with known inputs |
| `dot2x` | same opcode as raw words with several accumulators/inputs, plus a control store |
| `gds` | GDS/GWS/OA allocation, then 64 lanes of GDS `ds_add_rtn_u32` |
| `crossqueue` | `WAIT_REG_MEM` on compute ring 1 released by `WRITE_DATA` on ring 0 |
| `uconfig` | **took the board down**, see below; do not rerun as is |

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
