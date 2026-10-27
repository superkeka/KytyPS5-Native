# bc250-amdgpu

A patched amdgpu kernel module for the BC-250 that sets the GPU up the way the PS5 kernel sets up the
same silicon (gfx1013, "Oberon"): a title's compute queues on hardware queues of their own, two
graphics rings, TLB flushes that reach those queues. Kyty's native GPU backend
(`src/graphics/native_gpu`) runs ACBs on KFD user-mode queues and DCBs on the gfx rings; without the
patch the KFD queues hang or read stale memory.

It is built from the SteamOS kernel's own source of the running kernel (`linux-neptune`) and
installed next to the stock module in `/usr/lib/modules/<kernel>/updates/`. A SteamOS update that
changes the kernel drops it (the board is stock again until it is rebuilt).

## What the patch changes

| Change | Where | Why |
|---|---|---|
| PASID TLB flush through MMIO (`bc250_flush_pasid_kiq=0`) | `gmc_v10_0.c` | The KIQ path of the PASID flush hung under gfx load (ROCm#6313). |
| Runlist rebuild on KFD map/unmap (`bc250_flush_by_runlist`) | `kfd_*` | Under the firmware scheduler (HWS) the PASID flush finds no VMID; a runlist rebuild flushes it. Inactive without HWS. |
| KFD VMID flushed by VMID without HWS | `kfd_priv.h` | With `sched_policy=2` a process keeps one static VMID; the PASID lookup and the runlist workaround both skip it, so its TLB was never flushed (a second queue could not fetch its ring). |
| `bc250_tlb_flush_type` (0 = legacy) | `gmc_v10_0.c` | A heavyweight (type 2) invalidation of a KFD VMID waited for the VMID's outstanding requests and never got its ACK while a KFD queue flushed GL2; the KIQ waiting for it stopped the whole GPU. |
| Wave save area for every CU the chip has | `kfd_queue.c` | amdgpu counts the 24 CUs enabled at init; `bc250-cu-live-manager` enables 40. A preempted queue with waves on all 40 overflowed the save area. Sized from `array_count x cu_per_simd_array` on the BC-250 only. |
| Two gfx rings (`bc250_gfx_queues=2`) | `gfx_v10_0.c`, `amdgpu_gfx.c` | The PS5 gives a title gfx pipe 0 queue 0 (main ring) and queue 1 ("AGR" ring); amdgpu gives Navi1x one. The second ring is mapped the same way as the first (MQD through the KIQ), as `gfx_0.0.1`. Its EOP interrupt does not arrive as its own, so an EOP on pipe 0 checks the fences of both rings. |
| Gfx ring index = gfx ring | `amdgpu_ctx.c` | With two rings of one priority amdgpu balanced a context's gfx work over both, and anything could land on ring 1. Ring index N is now ring N; everything that asks for ring 0 (every Vulkan program) stays on ring 0, and ring 1 is used only when asked for (Kyty, for a title's second graphics queue). |
| Out-of-tree build fixes | `Makefile`, `amdgpu_trace.h` | The trace header is found next to its `.c` file. |

## Module options (written by the installer)

```
options amdgpu sched_policy=2 bc250_tlb_flush_type=0 bc250_gfx_queues=2 vm_update_mode=0 bc250_flush_pasid_kiq=0 bc250_flush_by_runlist=3 gpu_recovery=0
```

- `sched_policy=2`: KFD without the firmware scheduler. Queues sit on their hardware queues for good,
  as on the PS5; HWS preempted busy queues to rebuild its runlist, and a busy queue did not preempt on
  the BC-250.
- `vm_update_mode=0`: compute VMs' page tables updated by SDMA (the dGPU default). On this board amdgpu
  otherwise updates them with the CPU, which first waits for every fence of the VM: a guest GEM_VA
  then sat in the kernel behind a gfx job that waited on the GPU for an ACB, and the ACB's submission
  waited for the GEM_VA. Graphics programs (non-compute VMs) are not affected.
- `gpu_recovery=0`: a hung ring stays hung and readable instead of a reset that can turn into a hard
  reset on this board (`docs/native-gpu-hang-debugging.md`).

What it affects outside Kyty: only KFD users (ROCm/HIP: llama.cpp and the like). Without HWS KFD has a
fixed number of hardware queues, so many ROCm processes at once can run out of them. Steam games use
Vulkan on gfx ring 0 and the kernel compute rings, as before.

## Build, install, roll back (on the board)

```sh
# WORKDIR/src.orig/drivers/gpu/drm/amd: the driver from the SteamOS kernel source of `uname -r`
tools/bc250-amdgpu/build-amdgpu-bc250.sh WORKDIR
sudo tools/bc250-amdgpu/install-amdgpu-bc250.sh     # module to updates/, /etc/modprobe.d/bc250-amdgpu.conf, initramfs
sudo systemctl reboot
cat /sys/module/amdgpu/parameters/{sched_policy,bc250_tlb_flush_type,bc250_gfx_queues,vm_update_mode}   # 2 0 2 0
sudo tools/bc250-amdgpu/uninstall-amdgpu-bc250.sh   # back to the stock module and options
```

The installer takes the module from `/home/deck/dev/amdgpu-bc250/amdgpu.ko.zst`
(`AMDGPU_BC250_MODULE` overrides it).

## Trying a risky option

amdgpu cannot be reloaded on a running BC-250: `modprobe -r amdgpu` followed by `modprobe amdgpu`
hangs the board, even with stock options (`try-gfx-queues.sh` shows that). SteamOS's grub ignores
`next_entry`, so a one-boot menu entry (`oneshot-boot.sh`) does not work either. Options for one boot:

```sh
sudo tools/bc250-amdgpu/oneshot-modprobe.sh arm "options amdgpu bc250_gfx_queues=2"
sudo systemctl reboot   # this boot only; a hang costs a power cycle, which boots without them
```

A unit ordered before udev's coldplug moves `/etc/bc250-oneshot.conf` to `/run/modprobe.d` (amdgpu is
loaded by udev from the root file system, after systemd starts), so the file is gone from the disk
before amdgpu loads. `oneshot-modprobe.sh uninstall` removes the unit.

## Companion pieces

- `tools/bc250-lodhw/kyty-lodhw`: the root helper. Op 14 invalidates the KFD VMIDs' TLBs through
  invalidation engine 16 (amdgpu uses 0-14 and 17) after Kyty's GEM_VA changes, the way the PS5
  kernel invalidates a title's VMID itself; a KFD ioctl would wait for every fence of the VM.
- `tools/bc250-native-probe/kfd_probe.c avremap` with `KFD_PROBE_KFDFLUSH=map`: 200 remaps under gfx
  load, every KFD write in the buffer mapped now.
