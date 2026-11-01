#ifndef EMULATOR_SRC_GRAPHICS_NATIVE_GPU_NATIVESHADOW_H_
#define EMULATOR_SRC_GRAPHICS_NATIVE_GPU_NATIVESHADOW_H_

#include <bitset>
#include <cstdint>
#include <functional>
#include <span>
#include <vector>

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

// On the PS5's APU (BC-250, PS5 under Linux) turns native mode on unless the switches are set by hand
// or KYTY_NATIVE_GPU=0 (nativeDefaults.cpp). Called first in main.
void ApplyNativeDefaults();

// Translates and natively submits one guest DCB in shadow mode. Does nothing unless enabled. On a
// GPU timeout or context reset it disables itself for the rest of the run.
void ShadowSubmitDcb(const uint32_t* dcb, uint32_t size_in_dwords);

// The same for an asynchronous compute buffer (KYTY_NATIVE_GPU_ACB=1, or "dry" to only translate
// and report). ACBs run on the gfx ring in guest submission order with the DCBs, unless
// KYTY_NATIVE_GPU_ACB_RINGS=1 (with native labels): then each guest compute queue is a compute
// ring and waits between the rings are real GPU waits.
void ShadowSubmitAcb(const uint32_t* acb, uint32_t size_in_dwords);

// Guest queues: 0x20 + n (7 pipes x 8 queues) are a title's compute queues, any other number a graphics
// queue. kGuestAgrQueue is the title's second graphics ring, the one the PS5 kernel gives it next to the
// main ring ("AGR", gfx pipe 0 queue 1, its own registers): sceAgcDriverAgrSubmitDcb's buffers. No queue
// context carries that number.
constexpr uint32_t kGuestAgrQueue = 0x100;
constexpr bool     IsGuestComputeQueue(uint32_t queue) { return queue >= 0x20u && queue < 0x58u; }

// KYTY_NATIVE_GPU_THREAD=1: submissions go through a pipeline (ShadowEnqueue) instead of running on the
// caller's thread: `then` (handing the buffer to Kyty's command processor) runs once the native GPU
// work has completed, in submission order within each guest queue (one queue's buffer held on a guest
// wait lets the others' finish first). `queue` is 0 for a DCB (and for Kyty's own submissions, with a
// null buffer, which only keep their place in that order), kGuestAgrQueue for an AGR DCB, the guest
// compute queue for an ACB.
bool ShadowThreaded();
// True on the pipeline thread that hands finished buffers to Kyty.
bool ShadowOnPipelineThread();
void ShadowEnqueue(const uint32_t* buffer, uint32_t size_in_dwords, uint32_t queue,
                   std::function<void(std::span<const uint32_t>)> then);

// Native labels (KYTY_NATIVE_GPU_LABELS, on by default in primary mode with the pipeline; =0 brings
// back Kyty writing them): the GPU writes the guest's labels, EOP interrupts come from a tracker that
// watches the GPU pass points in the IB, and `then` gets only the packets Kyty's command processor
// still has to run (flips, waits for a flip), or nothing at all, instead of the whole guest buffer,
// which Kyty then never reads again. Buffers the native path refuses still go to Kyty whole.
bool ShadowNativeLabels();
// What native labels signal on the host: a queue's EOP interrupt (event id: 0 for the graphics queues,
// else the guest compute queue) and the reference clock written for timestamp labels.
void ShadowSetHost(std::function<void(int event_id, uint32_t context_id)> interrupt,
                   std::function<uint64_t()> reference_clock);
// Requests a DCB's flip (AgcDcbSetFlip) once the GPU has passed it, from the tracker thread.
void ShadowSetFlipHost(std::function<void(int handle, int index, int flip_mode, int64_t flip_arg)> flip);
// The EOP flip order (VideoOutDriver::EopFlipOrder: how many SetFlip releases have run), which the tracker puts into
// WaitUntilSafeForRendering's padding before Kyty's command processor runs it, from the tracker thread.
void ShadowSetEopFlipOrderHost(std::function<uint64_t()> order);

// Guest memory rendered natively, as pieces of exported GPU buffers (dma-buf fd, buffer size, offset,
// length), so Kyty's presenter can copy a surface on the GPU instead of reading it on the CPU. False
// when not in primary mode, when any byte is not resident, or with KYTY_NATIVE_GPU_NO_ZERO_COPY=1.
struct NativeDmabufPiece {
	int      fd;
	uint64_t bo_size;
	uint64_t bo_offset;
	uint64_t len;
};
bool ShadowDmabufPieces(uint64_t vaddr, uint64_t size, std::vector<NativeDmabufPiece>& out);

// A video-out buffer registered with DCC: in native primary mode (KYTY_NATIVE_GPU_GEM and
// KYTY_NATIVE_GPU_PRIMARY, unless KYTY_NATIVE_GPU_DISPLAY_DCC=1) the translator draws it without DCC,
// so the presenter can take it as uncompressed; true then.
bool ShadowDisplayWithoutDcc(uint64_t addr, uint64_t size);

// Direct scan-out flips (KYTY_DIRECT_PRESENT=1; off unless set, for now). On the PS5 a flip does no GPU
// work: the title's release has written the frame, and the kernel queues the buffer itself to the display
// for the next vblank. With this on, display buffers become VRAM scan-out buffers at registration and
// flips hand them to the display (Xwayland, gamescope, a DCN plane) without Kyty's copies. Wanted only
// with GEM backing in primary mode (KYTY_NATIVE_GPU_GEM, KYTY_NATIVE_GPU_PRIMARY, not KYTY_NATIVE_GPU_NULL)
// and without flip dumps (KYTY_NATIVE_GPU_DUMP_FLIPS, a pending "dump N" of KYTY_NATIVE_GPU_CTL), whose
// frames go through Vulkan. The switches are read once; the rest is checked at each call: false once the
// native path has failed to start or turned itself off, since Kyty then renders through Vulkan into its
// own images and the display buffers no longer receive frames. Any thread, lock-free.
bool DirectPresentWanted();

struct NativeDisplayBuffer {
	int      dmabuf_fd = -1; // borrowed: valid while the process lives; dup() to keep
	uint64_t bo_size   = 0;
	uint64_t bo_offset = 0;  // offset of the buffer's first byte in the buffer (DRI3 offset0)
};
// sceVideoOutRegisterBuffers2 (title thread, no VideoOut lock held): the display buffer [addr, addr + size)
// becomes one VRAM scan-out buffer at the same guest addresses (GemBacking::CarveGuestRange). Idempotent:
// the same range again returns the same buffer. False, with the buffer left as it was (presented through
// Vulkan), when DirectPresentWanted() is false or the carve is refused. Never takes the native path's lock
// and never waits for the GPU; takes tens of milliseconds.
bool ShadowCarveDisplayBuffer(uint64_t addr, uint64_t size, NativeDisplayBuffer* out);
// sceVideoOutUnregisterBuffers / close: [addr, addr + size) is no longer a registered display buffer. Its
// carve stays (and keeps the data), to be reused if the range is registered again. Any thread.
void ShadowReleaseDisplayBuffer(uint64_t addr, uint64_t size);
// The display completed a direct flip (the DirectPresenter's event thread): what a Vulkan flip does here
// through ShadowOnFlip and ShadowFlipStats, less making the buffer resident (a carve is) and the dumps
// (they force Vulkan flips): the PS5 per-VMID registers, KYTY_NATIVE_GPU_FLIP_STATS / FLIP_TRACE with no
// presenter time, KYTY_NATIVE_GPU_AUDIT's flip count.
void ShadowOnDirectFlip();

// sceAgcDriverSetTFRing / sceAgcDriverSetHsOffchipParam: the title's tessellation rings, which native
// IBs program as the PS5 driver would (VGT_TF_*, VGT_HS_OFFCHIP_PARAM).
void ShadowSetTessellationRings(uint64_t tf_ring_base, uint32_t tf_ring_bytes, uint64_t offchip_granularity,
                                uint64_t offchip_buffering);

// KYTY_NATIVE_GPU_FLIP_STATS=1: called at each flip with the time the presenter spent preparing it;
// logs per-flip averages (flip interval, native submission time, presenter time).
void ShadowFlipStats(double present_seconds);
// KYTY_NATIVE_GPU_FLIP_TRACE: the title has just submitted a DCB (agc), before the pipeline takes it.
void ShadowNoteDcbSubmit();

// Stops the title's native GPU work where it stands before the process goes (the game menu's exit,
// exec, a fatal error; SIGTERM, SIGINT and SIGHUP do it themselves), as the PS5 kernel stops a title's
// queues: left on the rings, a job waiting for a label nobody will write times out, and the GPU reset
// that follows takes the BC-250 down. Once; `why` goes to the log.
void ShadowStopGpu(const char* why);

// True in primary mode with native async compute: Kyty then skips its own dispatches of ACBs too.
bool NativeAsyncCompute();

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
// KYTY_NATIVE_GPU_NULL=1 makes it true with nothing rendering at all, for CPU-side measurements.
bool PrimaryRendering();
// KYTY_NATIVE_GPU_LOD_STATS=1: the native path runs GET_LOD_STATS, so Kyty must not fill the buffer.
bool NativeLodStats();
// KYTY_LOD_STATS_MODE=usage (the default) or =banks, or KYTY_LOD_STATS_LOG=N: the translator collects
// the counter banks of the T#s each draw and dispatch reaches (MIP_STATS_CNT_EN, word6[7:0]).
bool LodBankScanOn();
// The banks seen before the GET_LOD_STATS packet that writes `addr`, once per packet; false when the
// scan is off or no report was published for it. The pipeline thread publishes; this takes.
bool TakeLodStatsBanks(uint64_t addr, std::bitset<256>& banks);
// KYTY_LOD_STATS_TRAP: makes the pages of [addr, addr + size) inaccessible; LodStatsTrapFault
// (called from the host fault handler with the faulting thread's ucontext_t) logs each CPU access with
// its PC and single-steps it before closing the pages again.
void LodStatsTrapArm(uint64_t addr, uint64_t size);
bool LodStatsTrapFault(uint64_t vaddr, uint64_t pc, bool write, void* native_context);

// Called at every video-out flip with the guest surface about to be presented. For the flips
// listed in KYTY_NATIVE_GPU_DUMP_FLIPS ("300,600" or a range "300-2000/10"), writes to
// KYTY_NATIVE_GPU_DUMP_DIR (default "."):
//   flip_<n>.native.raw  the guest bytes as the native path rendered them,
//   flip_<n>.vulkan.raw  the same surface after `download_region` wrote Kyty's Vulkan copy back
//                        (shadow mode only; skipped in primary mode, where Kyty draws nothing),
//   flip_<n>.txt         the layout,
//   flip_<n>.rts.txt     every color target bound since the previous flip, native vs Vulkan
//                        (differing bytes), with flip_<n>.rt_<addr>.{native,vulkan}.raw for those
//                        that differ (shadow mode only).
// tools/bc250-frame-diff compares the pairs. `download_region(addr, size)` writes Kyty's Vulkan copy
// of guest range [addr, addr + size) back into guest memory and returns how many images it wrote.
void ShadowOnFlip(uint64_t address, uint64_t size, uint32_t width, uint32_t height,
                  uint32_t bytes_per_pixel, uint32_t pixel_format, uint32_t tile_mode,
                  const std::function<uint32_t(uint64_t, uint64_t)>& download_region);

} // namespace Libs::Graphics::NativeGpu

#endif // EMULATOR_SRC_GRAPHICS_NATIVE_GPU_NATIVESHADOW_H_
