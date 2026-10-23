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
