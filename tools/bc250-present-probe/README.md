# bc250-present-probe

Prototype of the **present path** for the BC-250 native GPU backend: taking a colour surface
that the guest rendered natively into its own (tiled) memory and getting it onto the screen
through Vulkan, without a CPU copy of the guest pages.

Standalone — it does not touch Kyty. It is the piece needed once native frames complete:
Kyty marks a flip with its own `R_FLIP` NOP marker in the DCB, and on the native path that
marker has to become "show this guest surface" (see **How this plugs into Kyty** below).

## What it does

For each test surface:

1. Builds a known linear image (a deterministic full-range 32-bit pattern) and tiles it on the
   CPU into the **guest surface** using Kyty's exact tile layout — this is the ground truth.
   Two layouts: `kLinear` (0x00) and `kRenderTarget` / `SW_64KB_R_X` (0x1b), 32bpp.
2. Makes the guest pages readable by Vulkan **with no copy** via `VK_EXT_external_memory_host`:
   the host pages are imported as `VkDeviceMemory` and bound to a storage `VkBuffer`. This is
   exactly how the guest's memfd / userptr pages are handed to the GPU on the native path.
3. Detiles/converts the surface into a tightly-packed linear image with a compute shader
   (`detile_rt.comp` for `SW_64KB_R_X`, `copy_linear.comp` for linear).
4. Reads the result back and checks it **bit-exactly** against the CPU reference, and reports
   the GPU time of the detile dispatch for a 1920×1080 frame.

The detiled linear image is the thing that, in Kyty, would be the source of a blit into the
swapchain image. The readback here stands in for that blit so the tool is self-checking.

## Build and run (on the BC-250)

Vulkan headers and glslang live in the `shadps4-dev` distrobox:

```sh
distrobox enter shadps4-dev -- make
./bc250-present-probe                 # all correctness cases + 1080p timing
./bc250-present-probe rt 1920 1080    # one custom case: <linear|rt> <w> <h>
```

Use a build directory **outside** the shared `KytyPS5-bc250` checkout
(e.g. `/home/deck/dev/present-check`). The Makefile generates `shaders.h` from the `.comp`
files with `glslangValidator --vn` and links only against the Vulkan loader (`-lvulkan`) — no
libdrm, no Kyty objects.

## Result on hardware (AMD BC-250, RADV GFX1013)

```
device: AMD BC-250 (RADV GFX1013)  (timestampPeriod=10.000 ns, minImportedHostPointerAlignment=4096)
correctness:
  linear small             17x13    lin   tiled=   0.00 MiB  gpu=  0.004 ms  PASS
  linear odd              257x129   lin   tiled=   0.17 MiB  gpu=  0.005 ms  PASS
  r_x small                64x64    r_x   tiled=   0.06 MiB  gpu=  0.005 ms  PASS
  r_x one block           128x128   r_x   tiled=   0.06 MiB  gpu=  0.007 ms  PASS
  r_x odd                 300x200   r_x   tiled=   0.38 MiB  gpu=  0.015 ms  PASS
  r_x 1080p              1920x1080  r_x   tiled=   8.44 MiB  gpu=  0.122 ms  PASS
  linear 1080p           1920x1080  lin   tiled=   7.98 MiB  gpu=  0.082 ms  PASS
timing (1920x1080, best of 20):
  r_x  detile 1920x1080 best GPU time: 0.086 ms
  lin  detile 1920x1080 best GPU time: 0.072 ms
ALL PASS
```

So the no-copy host-pointer import works on the board, and a full `SW_64KB_R_X` 1080p frame
detiles in well under 0.1 ms of GPU time — negligible against a 16.6 ms (60 Hz) frame.

The address math in both shaders is copied **verbatim** from Kyty's production GPU detiler
(`src/graphics/host_gpu/shaders/gpu_tiler_render_target.comp`, `ELEMENT_BYTES==4`, plus the
block-grid loop in `gpu_tiler_common.inc`) and from `Gen5RenderTargetOffsetInBlock<uint32_t>`
in `src/graphics/guest_gpu/tile.cpp`. Keeping it byte-identical is what makes the bit-exact
check meaningful: it proves this tool reproduces *Kyty's* layout, not merely that it is
self-consistent. The edge-block and odd-size cases confirm the block-grid math, and the linear
case exercises row-pitch removal.

## How this plugs into Kyty later (replacing what `R_FLIP` does on the native path)

On the translated-Vulkan path, Kyty renders into its own host images and, at the guest's flip,
presents a *host* image it already owns. On the native path there is no host image: the guest
rendered straight into its own memory, so the flip marker has to turn that guest surface into a
swapchain frame.

Concretely, at the `R_FLIP` marker in the DCB (the native submit path already recognises and
counts this NOP; it currently skips it):

1. Read the flip's guest surface address, format, width/height/pitch and tile mode from the
   guest flip parameters (the same fields `sceVideoOutSubmitFlip` / the display buffer
   registration carry, which Kyty already parses on the Vulkan path).
2. Look up the GEM chunk(s) backing that address (`GemBacking`, the userptr/memfd backing added
   on `native-gpu-probe`) and import the covering host range as a storage buffer via
   `VK_EXT_external_memory_host` — the step this tool prototypes. The import can be cached and
   keyed by (chunk, offset) so a steady flip chain re-imports nothing.
3. Dispatch the detile compute shader into a swapchain-sized linear image (or detile directly
   while sampling during a blit), then `vkQueuePresentKHR`. Kyty's existing `TileManager`
   already has the parametric per-family detile shaders and the present/blit plumbing; the only
   new input is "the source is imported guest memory" instead of a Kyty-owned buffer, so this
   should reuse `TileManager::Detile()` with an externally-imported source `VkBuffer`.

This keeps the native path's present entirely on the existing Vulkan device/queue, so it is
safe to bring up before native PM4 submission is reliable (it needs only Vulkan, never native
PM4).

## Open issue to resolve before it matches real native output: the pipe/bank XOR

Kyty's tiler — and therefore this tool's ground truth — uses the **within-block** `SW_64KB_R_X`
offset (`rt_offset`) and a plain block grid, with **no pipe/bank XOR**. That is correct for
Kyty's own tiled buffers, because Kyty both tiles and detiles them with the same convention and
the guest, on the Vulkan path, never sees true hardware tiling.

Real hardware is different. `SW_*_X` modes XOR pipe/bank bits (derived from the block
coordinate and the surface base address; BC-250 has `PIPE_XOR_BITS=4`, 16 pipes, 2 SE, RB+)
into the block address. A surface the guest GPU rendered *natively* is laid out with that XOR,
so detiling it with the non-XOR layout will scramble whole 64 KiB blocks (pixels within a block
stay correct; blocks land in the wrong place). `tile.cpp` already has the per-block term —
`TileGetBlockXor` (the `RenderTarget64KB` / `Depth64KB` branches) — but the renderer and the
GPU detiler do not apply it today.

Before wiring the native present in, the detile shader must add the base-address-dependent pipe
XOR to `block_index` (fold `TileGetBlockXor(block_x, block_y)` and the base-address pipe bits
into the block offset). This tool is structured so that becomes a one-function change in
`detile_rt.comp` plus feeding the surface base address through the push constants; the test can
then tile its ground truth with the XOR enabled and must stay bit-exact. Validating that
against a surface the hardware actually rendered (via the native PM4 path, once it completes) is
the real acceptance test.

## Alternatives considered and rejected

- **dma-buf import with AMD DRM format modifiers.** The clean path would be to export the guest
  surface as a dma-buf with the right `DRM_FORMAT_MOD_AMD_GFX10_*` modifier and import it as an
  optimal-tiled `VkImage`, letting RADV detile it for free on sample. Rejected because the guest
  backing is a **userptr** GEM object (memfd `MAP_SHARED` pages registered with `GEM_USERPTR`,
  per `gemBacking.h`), and userptr BOs **cannot be exported as dma-buf** on this driver
  (confirmed in the probe). It would also require the guest's exact modifier, which Kyty does
  not currently derive. `VK_EXT_external_memory_host` sidesteps both: it imports the raw pages
  as a plain buffer, and the compute shader owns the layout.

- **Copy the guest surface into a real GEM buffer, then import/blit that.** A staging copy of
  up to ~8 MiB per flip is pure overhead and reintroduces exactly the CPU/extra-copy cost the
  native path exists to avoid. The host-pointer import is zero-copy.

- **Detile on the CPU and upload a linear image.** Simple, but it serialises every frame on the
  CPU and uploads a full frame each flip. The compute detile measured here is < 0.1 ms of GPU
  time with no CPU per-pixel work and no upload.

- **Bind the guest pages as an optimal-tiled `VkImage` directly** (no compute shader). RADV's
  image tiling is its own swizzle, unrelated to the guest's `SW_64KB_R_X`, so this cannot match
  the guest layout; an explicit detile is required either way.
```
