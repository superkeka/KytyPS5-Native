#ifndef EMULATOR_SRC_GRAPHICS_NATIVE_GPU_NATIVESHADOW_H_
#define EMULATOR_SRC_GRAPHICS_NATIVE_GPU_NATIVESHADOW_H_

#include <cstdint>

// Shadow-mode native GPU submission for the BC-250. Called from Kyty's DCB submit path, it reads
// the guest draw command buffer, translates it to a native amdgpu IB and submits it to the gfx
// ring alongside Kyty's normal Vulkan path, purely to measure whether native frames complete.
//
// It is a no-op unless BOTH env switches are set:
//   KYTY_NATIVE_GPU_GEM=1    guest memory is backed by amdgpu GEM buffers and bound into the GPU VM
//   KYTY_NATIVE_GPU_SHADOW=1 translate and submit each DCB natively
//
// It never changes what the Vulkan path does, so default behaviour (either switch unset) is
// byte-for-byte upstream.

namespace Libs::Graphics::NativeGpu {

// Translates and natively submits one guest DCB in shadow mode. Does nothing unless enabled. On a
// GPU timeout or context reset it disables itself for the rest of the run.
void ShadowSubmitDcb(const uint32_t* dcb, uint32_t size_in_dwords);

} // namespace Libs::Graphics::NativeGpu

#endif // EMULATOR_SRC_GRAPHICS_NATIVE_GPU_NATIVESHADOW_H_
