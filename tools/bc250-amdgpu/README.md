# bc250-amdgpu

Kernel setup for Kyty's native GPU backend (`src/graphics/native_gpu`) on the BC-250, and a patched
amdgpu module that sets the GPU up the way the PS5 kernel sets up the same silicon (gfx1013, "Oberon").

## Modes

| Mode | Settings | Result |
|---|---|---|
| Patched amdgpu | GTT 14 GiB, this module and its options | Best. ACBs on KFD queues, two gfx rings, EOP interrupts, C_RW_S shared memory. |
| Stock amdgpu, options | GTT 14 GiB, `options amdgpu sched_policy=2 vm_update_mode=0 cwsr_enable=0` | Same games, same speed. ACBs on KFD queues, one gfx ring. |
| Stock amdgpu, GTT only | GTT 14 GiB | Works, poor. ACBs on amdgpu compute rings, cross-queue waits held on the CPU, no interrupts. Games heavy on async compute (ASTRO BOT) are unplayable. |
| Stock, nothing set | | Games over 8 GiB of guest memory (Gran Turismo 7, ASTRO BOT, My First Gran Turismo) stall loading in TTM swap-out. |

Kyty picks the path itself: KFD queues when `sched_policy=2`, compute rings otherwise.

GTT 14 GiB:

```sh
echo 'GRUB_CMDLINE_LINUX="${GRUB_CMDLINE_LINUX} ttm.pages_limit=3670016"' | sudo tee /etc/default/grub.d/bc250-kyty.cfg
sudo steamos-readonly disable && sudo update-grub; sudo steamos-readonly enable
```

Stock options: `/etc/modprobe.d/bc250-stock.conf` with the line above. amdgpu loads from the root file
system, so a reboot applies both.

## Module options

```
options amdgpu sched_policy=2 bc250_tlb_flush_type=0 bc250_gfx_queues=2 vm_update_mode=0 bc250_flush_pasid_kiq=0 bc250_flush_by_runlist=3 gpu_recovery=0
```

- `sched_policy=2`: KFD without the firmware scheduler (HWS). Queues stay on their hardware queues, as
  on the PS5. Under HWS the first KFD queues hang the GPU (`timeout waiting for kiq fence`).
- `vm_update_mode=0`: compute VMs' page tables updated by SDMA. With CPU updates a guest GEM_VA waited
  for a gfx job that waited on the GPU for an ACB whose submission waited for the GEM_VA.
- `cwsr_enable=0` (stock only): the stock wave save area holds the 24 CUs amdgpu counts at init, not
  the 40 `bc250-cu-live-manager` enables. With CWSR on, Kyty keeps ACBs on 24 CUs.
- `gpu_recovery=0` (optional): a hung ring stays hung and readable (`docs/native-gpu-hang-debugging.md`)
  instead of a reset.

Outside Kyty only KFD users (ROCm, HIP) are affected: without HWS KFD has a fixed number of hardware
queues. Steam games use Vulkan on gfx ring 0 and the kernel compute rings as before.

## What the patch changes

| Change | Where | Why |
|---|---|---|
| PASID TLB flush through MMIO (`bc250_flush_pasid_kiq=0`) | `gmc_v10_0.c` | The KIQ path hung under gfx load (ROCm#6313). |
| Runlist rebuild on KFD map/unmap (`bc250_flush_by_runlist`) | `kfd_*` | Under HWS the PASID flush finds no VMID. Inactive without HWS. |
| KFD VMID flushed by VMID without HWS | `kfd_priv.h` | With `sched_policy=2` a process keeps one static VMID that neither flush reached. |
| `bc250_tlb_flush_type` (0 = legacy) | `gmc_v10_0.c` | A type 2 invalidation of a KFD VMID never got its ACK while a KFD queue flushed GL2, and the KIQ stopped the GPU. |
| Wave save area for every CU | `kfd_queue.c` | Sized for 40 CUs, not the 24 enabled at init. |
| Two gfx rings (`bc250_gfx_queues=2`) | `gfx_v10_0.c`, `amdgpu_gfx.c` | The PS5's main and "AGR" rings (pipe 0 queues 0 and 1). An EOP on pipe 0 checks the fences of both. |
| Gfx ring index = gfx ring | `amdgpu_ctx.c` | Ring 1 only when asked for. Every Vulkan program stays on ring 0. |
| `AMDGPU_VM_MTYPE_RW` maps C_RW_S (`bc250_mtype_rw`) | `gmc_v10_0.c` | As the PS5 kernel maps a title's shared memory (type 12). amdgpu mapped it NC. |
| EOP interrupts of render-node submissions reach KFD events (`bc250_gfx_eop_events`) | `kfd_int_process_v10.c` | Stock KFD drops EOP interrupts outside KFD's VMIDs. A context id with bit 27 set signals that KFD event; other ids go to the process's EOP log (`AMDKFD_IOC_BC250_EOP_LOG`) with VMID and ring. `eop_irq_probe` checks it. |
| Out-of-tree build fixes | `Makefile`, `amdgpu_trace.h` | |

On the stock amdgpu Kyty raises its tracker event from the ACBs' own KFD queues and from a relay KFD
queue that waits for the gfx ring's points. A title's interrupts become tracker points.

## Build, install, roll back (on the board)

```sh
# WORKDIR/src.orig/drivers/gpu/drm/amd: the driver from the SteamOS kernel source of `uname -r`
tools/bc250-amdgpu/build-amdgpu-bc250.sh WORKDIR
sudo tools/bc250-amdgpu/install-amdgpu-bc250.sh     # module to updates/, /etc/modprobe.d/bc250-amdgpu.conf
sudo systemctl reboot
cat /sys/module/amdgpu/parameters/{sched_policy,bc250_tlb_flush_type,bc250_gfx_queues,vm_update_mode}   # 2 0 2 0
sudo tools/bc250-amdgpu/uninstall-amdgpu-bc250.sh   # back to the stock module and options
```

The installer takes `/home/deck/dev/amdgpu-bc250/amdgpu.ko.zst` (`AMDGPU_BC250_MODULE` overrides it).
A SteamOS update that changes the kernel drops the module until it is rebuilt.

## Trying a risky option

amdgpu cannot be reloaded on a running BC-250 (`modprobe -r` and `modprobe` hang the board), and
SteamOS's grub ignores `next_entry`. Options for one boot:

```sh
sudo tools/bc250-amdgpu/oneshot-modprobe.sh arm "options amdgpu bc250_gfx_queues=2"
sudo systemctl reboot   # a hang costs a power cycle, which boots without them
```

`oneshot-modprobe.sh uninstall` removes its unit.

## Companion pieces

- `tools/bc250-lodhw/kyty-lodhw`: the root helper. Op 14 invalidates the KFD VMIDs' TLBs (invalidation
  engine 16) after Kyty's GEM_VA changes, as the PS5 kernel does.
- `tools/bc250-native-probe/kfd_probe.c avremap` with `KFD_PROBE_KFDFLUSH=map`: 200 remaps under gfx load.
