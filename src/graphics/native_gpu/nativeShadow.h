#ifndef EMULATOR_SRC_GRAPHICS_NATIVE_GPU_NATIVESHADOW_H_
#define EMULATOR_SRC_GRAPHICS_NATIVE_GPU_NATIVESHADOW_H_

#include <cstdint>
#include <functional>

// Shadow-mode native GPU submission for the BC-250. Called from Kyty's DCB submit path, it reads
// the guest draw command buffer, translates it to a native amdgpu IB and submits it to the gfx
// ring alongside Kyty's normal Vulkan path, purely to measure whether native frames complete.
//
// It is a no-op unless BOTH env switches are set:
//   KYTY_NATIVE_GPU_GEM=1    guest memory is backed by amdgpu GEM buffers and bound into the GPU VM
//   KYTY_NATIVE_GPU_SHADOW=1 translate and submit each DCB natively
//
// Diagnostics: KYTY_NATIVE_GPU_STATS=1 logs pipeline statistics (vertices, primitives, pixel shader
// invocations) of the first frames with draws; KYTY_NATIVE_GPU_DUMP_REGS=<frame> writes the
// shadowed register state of that frame to KYTY_NATIVE_GPU_DUMP_DIR.
//
// It never changes what the Vulkan path does, so default behaviour (either switch unset) is
// byte-for-byte upstream.

namespace Libs::Graphics::NativeGpu {

// Translates and natively submits one guest DCB in shadow mode. Does nothing unless enabled. On a
// GPU timeout or context reset it disables itself for the rest of the run.
void ShadowSubmitDcb(const uint32_t* dcb, uint32_t size_in_dwords);

// While shadow submission is active, each guest DCB has already run natively before Kyty records
// it, so at a flip the guest's display surface in guest memory holds the native result. When this
// returns true the presenter re-reads that surface from guest memory instead of showing Kyty's own
// Vulkan copy, i.e. the screen shows the native frame. F9 toggles it; KYTY_NATIVE_GPU_SHOW=1 starts
// with it on. Always false when shadow submission is off or has disabled itself.
bool ShadowShowsNativeFrames();
void ShadowToggleNativeFrames();

// Primary mode (KYTY_NATIVE_GPU_PRIMARY=1, implies shadow submission and showing native frames):
// the native path renders and Kyty's command processor still runs every DCB for synchronization,
// labels and flips, but skips its own Vulkan draws and DCB dispatches. Returns false whenever the
// native path is not active, e.g. after it disabled itself, so Kyty falls back to rendering.
bool PrimaryRendering();

// Called at every video-out flip with the guest surface about to be presented. For the flips
// listed in KYTY_NATIVE_GPU_DUMP_FLIPS ("300,600" or a range "300-2000/10"), writes to
// KYTY_NATIVE_GPU_DUMP_DIR (default "."):
//   flip_<n>.native.raw  the guest bytes as the native path rendered them,
//   flip_<n>.vulkan.raw  the same surface after `download_vulkan` wrote Kyty's Vulkan copy back
//                        (shadow mode only; skipped in primary mode, where Kyty draws nothing),
//   flip_<n>.txt         the layout.
// tools/bc250-frame-diff compares the pairs.
void ShadowOnFlip(uint64_t address, uint64_t size, uint32_t width, uint32_t height,
                  uint32_t bytes_per_pixel, uint32_t pixel_format, uint32_t tile_mode,
                  const std::function<uint32_t()>& download_vulkan);

} // namespace Libs::Graphics::NativeGpu

#endif // EMULATOR_SRC_GRAPHICS_NATIVE_GPU_NATIVESHADOW_H_
