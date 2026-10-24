#ifndef EMULATOR_SRC_GRAPHICS_NATIVE_GPU_NATIVEPM4_H_
#define EMULATOR_SRC_GRAPHICS_NATIVE_GPU_NATIVEPM4_H_

#include <cstdint>
#include <cstdio>
#include <map>
#include <set>
#include <span>
#include <utility>
#include <vector>

// PM4 translator for the BC-250 native GPU backend. Turns one guest DCB (a PS5 draw command
// buffer, as Kyty's AGC driver submits it) into a self-contained indirect buffer (IB) for the
// amdgpu gfx ring, so it can run next to Kyty's Vulkan path in shadow mode.
//
// The translator is pure data manipulation with no hardware or platform dependency, so it builds
// and is unit-tested everywhere. The submitter (nativeSubmit.h) and the hook (nativeShadow.*) use
// it on the board.
//
// Guest VA == host VA in Kyty, so register-load packets that point at offset/value pairs in guest
// memory are read through the pointer directly; the DCB and any indirect buffers are ordinary host
// memory from the translator's point of view.

namespace Libs::Graphics::NativeGpu {

struct Pm4TranslateStats {
	uint32_t packets           = 0; // type-3 packets seen in the DCB
	uint32_t nops              = 0; // IT_NOP packets skipped (markers included)
	uint32_t marker_flip       = 0; // IT_NOP R_FLIP
	uint32_t marker_release    = 0; // IT_NOP R_RELEASE_MEM
	uint32_t marker_wait_flip  = 0; // IT_NOP R_WAIT_FLIP_DONE
	uint32_t set_context       = 0; // IT_SET_CONTEXT_REG passed through
	uint32_t set_sh            = 0; // IT_SET_SH_REG passed through
	uint32_t set_uconfig       = 0; // IT_SET_UCONFIG_REG passed through
	uint32_t set_uconfig_index = 0; // IT_SET_UCONFIG_REG_INDEX passed through
	uint32_t load_context      = 0; // IT_SET_CONTEXT_REG_INDIRECT (0x9F) expanded
	uint32_t load_sh           = 0; // IT_SET_SH_REG_INDIRECT (0x63) expanded
	uint32_t load_uconfig      = 0; // IT_SET_UCONFIG_REG_INDIRECT (0x64) expanded
	uint32_t loaded_cx_regs    = 0; // offset/value pairs expanded from 0x9F
	uint32_t loaded_sh_regs    = 0; // offset/value pairs expanded from 0x63
	uint32_t loaded_uc_regs    = 0; // offset/value pairs expanded from 0x64
	uint32_t draws             = 0; // IT_DRAW_INDEX_2
	uint32_t dispatches        = 0; // IT_DISPATCH_DIRECT / IT_DISPATCH_INDIRECT
	uint32_t num_instances     = 0; // IT_NUM_INSTANCES passed through
	uint32_t index_state       = 0; // IT_INDEX_BASE / IT_INDEX_BUFFER_SIZE passed through
	uint32_t draws_skipped     = 0; // draws left out (SetSkippedPixelShaders)
	uint32_t fmask_fixed       = 0; // CB_COLORn_FMASK set to the target base (was 0) before a draw
	uint32_t lod_stats_dropped = 0; // GET_LOD_STATS packets left out
	uint32_t context_ops       = 0; // Agc context-state push/pop/clear markers applied
	uint32_t wait_barriers     = 0; // guest waits replaced by an end-of-pipe fence and a wait on it
	uint32_t gds_releases      = 0; // RELEASE_MEM markers that send GDS data, run natively
	uint32_t packets_skipped   = 0; // packets left out (SetSkippedOpcodes)
	uint32_t set_base          = 0; // IT_SET_BASE passed through (indirect argument base)
	uint32_t pfp_sync_me       = 0; // IT_PFP_SYNC_ME passed through
	uint32_t acquire_mem       = 0; // IT_ACQUIRE_MEM
	uint32_t event_write       = 0; // IT_EVENT_WRITE
	uint32_t write_data_dropped = 0; // IT_WRITE_DATA to memory, left to Kyty (see Translate)
	uint32_t dma_data          = 0; // IT_DMA_DATA memory/immediate -> memory passed through
	uint32_t waits_dropped     = 0; // WAIT_REG_MEM(_64) on memory, dropped in shadow mode
	uint32_t waits_unmet       = 0; // of those, condition not yet true when the DCB was translated
	uint32_t pfp_syncs_emitted = 0; // PFP_SYNC_ME emitted in place of dropped PFP waits
	uint32_t clear_state       = 0; // IT_CLEAR_STATE
	uint32_t uconfig_skipped   = 0; // non-whitelisted UCONFIG writes dropped for board safety
	uint32_t preamble_cx       = 0; // context registers emitted in the IB preamble
	uint32_t preamble_sh       = 0; // SH registers emitted in the IB preamble
	uint32_t preamble_uc       = 0; // UCONFIG registers emitted in the IB preamble
};

// A guest memory range the GPU may touch while running a translated DCB, for the backend's lazy
// residency (gemBacking.h): only guest memory the GPU can reach has to be a GPU buffer. The
// translator reports candidates generously; the backend keeps the ones that fall into guest
// direct memory. A `pointer` candidate is a 64-bit address taken from shader user data, i.e.
// possibly a resource table that holds further descriptors; the backend reads it and follows the
// addresses it finds.
struct GpuRef {
	uint64_t addr    = 0;
	uint64_t size    = 0;
	bool     pointer = false;
	bool     index   = false; // an index buffer the next draws fetch from (must be GPU-mapped)

	bool operator<(const GpuRef& o) const {
		return addr != o.addr ? addr < o.addr
		                      : (size != o.size ? size < o.size : (pointer != o.pointer ? pointer < o.pointer : index < o.index));
	}
	bool operator==(const GpuRef& o) const {
		return addr == o.addr && size == o.size && pointer == o.pointer && index == o.index;
	}
};

// A color render target bound for a draw: where it is and the raw CB registers that describe it.
struct RenderTargetInfo {
	uint64_t addr    = 0;
	uint64_t size    = 0; // upper bound (16 bytes per pixel, all slices, mips)
	uint32_t width   = 0;
	uint32_t height  = 0;
	uint32_t info    = 0; // CB_COLORn_INFO (format, number type, ...)
	uint32_t attrib  = 0; // CB_COLORn_ATTRIB
	uint32_t attrib3 = 0; // CB_COLORn_ATTRIB3 (swizzle mode, resource type, ...)
	uint32_t draws   = 0; // draws in the DCB with this target bound
};

// User data of a draw whose pixel shader is under inspection (SetInspectedPixelShader).
struct DrawSnapshot {
	uint64_t ps = 0;
	uint64_t es = 0;
	uint32_t ps_user_data[32] = {};
	uint32_t gs_user_data[32] = {}; // SPI_SHADER_USER_DATA_GS_*: the NGG (merged ES/GS) stage on gfx10
};

// A draw or dispatch whose resources point into the range given to SetFindRange: directly from a
// user-data register, or through a table it points at (depth 1) or a table in that table (depth 2).
struct FindHit {
	bool     dispatch = false;
	uint32_t index    = 0; // draw or dispatch number within the DCB
	uint64_t shader   = 0; // CS for a dispatch, PS for a draw
	uint64_t es       = 0; // the NGG (ES/GS) shader of a draw
	uint32_t user_reg = 0; // SH offset of the user-data register the path starts from
	uint32_t depth    = 0;
	uint64_t table    = 0; // the table holding the reference (0 at depth 0)
	uint32_t word     = 0; // dword index of the reference in that table
	uint64_t target   = 0; // the address found
	bool     image    = false; // a T# base (address >> 8) rather than a 48-bit V#/pointer
};

struct Pm4TranslateResult {
	bool                  ok         = true;  // false when an unknown packet forced a refusal
	uint32_t              unknown_op = 0;     // the opcode that could not be translated
	uint32_t              unknown_dw = 0;     // its dword offset in the DCB
	std::set<uint32_t>    unknown_ops;        // every untranslatable opcode in the DCB
	std::vector<uint32_t> unknown_detail;     // leading body dwords of a refused known packet
	std::vector<uint32_t> ib;                 // the self-contained IB (empty when !ok)
	Pm4TranslateStats     stats;
	std::vector<uint64_t> shader_addrs;       // shader base VAs from PGM_LO writes this DCB
	std::set<uint32_t>    skipped_uconfig;    // non-whitelisted UCONFIG offsets dropped this DCB
	std::vector<GpuRef>   refs;               // guest memory the DCB's draws/dispatches may touch
	std::vector<uint32_t> ref_buckets;        // compact mode: the 16 MiB buckets (addr >> 24) of those refs
	std::vector<RenderTargetInfo> render_targets; // color targets bound for its draws, by address
	std::vector<DrawSnapshot>     inspected;      // draws with the inspected pixel shader
	std::vector<FindHit>          find_hits;      // references into the SetFindRange range
	struct UnmetWait {
		uint64_t addr;
		uint64_t ref;
		uint64_t value;
		uint32_t function;
		uint32_t dw;       // dword offset of the wait in the guest buffer
		uint32_t draws;    // draws before it in this buffer
		uint32_t dispatches;
	};
	std::vector<UnmetWait>        unmet_waits;    // WAIT_REG_MEM conditions not met at translation time
};

// Translates guest DCBs into self-contained IBs, keeping a CPU shadow of every register the guest
// sets so each IB can restore the full state in a preamble (the gfx ring is shared between
// submissions, so nothing on the hardware is assumed to survive from the previous DCB).
class Pm4Translator {
public:
	Pm4Translator()  = default;
	~Pm4Translator() = default;

	Pm4Translator(const Pm4Translator&)            = delete;
	Pm4Translator& operator=(const Pm4Translator&) = delete;

	// Translates one DCB. On an unknown packet the result's `ok` is false, `ib` is empty and the
	// shadow is left untouched past the point of failure (the caller skips this DCB natively). The
	// shadow persists across calls.
	Pm4TranslateResult Translate(std::span<const uint32_t> dcb);

	// Drops the shadowed state (e.g. after a context reset forces a fresh context).
	void ResetShadow();

	// Seeds the context shadow with the values the PS5 system driver programs before a title's
	// first command buffer, for registers games rely on but never write themselves. CLEAR_STATE on
	// amdgpu leaves them at different defaults (PA_CL_VTE_CNTL = 0 disables the viewport transform,
	// so every primitive collapses into the corner and no pixel shader ever runs). Guest writes
	// override these like any other shadowed value.
	void SeedPs5Defaults();

	// Seeds every context register in `defaults` that the shadow does not hold yet (so it never
	// overrides SeedPs5Defaults or a guest write), e.g. the PS5 library's register defaults.
	void SeedContextDefaults(std::span<const std::pair<uint32_t, uint32_t>> defaults);

	// Writes the shadowed registers as "cx|sh|uc <offset> <value>" lines (dword offsets within each
	// register space), for comparing against the state Kyty sees.
	void DumpShadow(std::FILE* f) const;

	// Depth-unit experiments (KYTY_NATIVE_GPU_DEPTH): HizOff forces hierarchical Z/stencil off in
	// DB_RENDER_OVERRIDE; NoHtile drops TILE_SURFACE_ENABLE so depth runs without HTILE at all.
	enum class DepthDebugMode { None, HizOff, NoHtile };
	void SetDepthDebugMode(DepthDebugMode mode) { m_depth_mode = mode; }

	// Experiment: render targets without DCC compression (CB_COLORn_INFO.DCC_ENABLE cleared).
	void SetNoDcc(bool no_dcc) { m_no_dcc = no_dcc; }

	// Experiment: targets without DCC lose CB_COLORn_INFO.FAST_CLEAR (no CMASK fast clear).
	void SetNoCmaskClear(bool on) { m_no_cmask_clear = on; }

	// Experiment: CB_TARGET_MASK is ANDed with `mask`, e.g. 0xf keeps only MRT0 writable.
	void SetTargetMask(uint32_t mask) { m_target_mask = mask; }

	// Guest WAIT_REG_MEMs become a full GPU barrier on a fence of ours at `va` (4 bytes in a buffer
	// that every submission maps); 0 drops them as before.
	void SetFenceAddress(uint64_t va) { m_fence_va = va; }

	// The buffers come from a compute queue (ACBs), whose packets differ from the gfx ring's
	// (DISPATCH_INDIRECT carries its argument address); they are rewritten for the gfx ring.
	void SetComputeQueue(bool compute_queue) { m_compute_queue = compute_queue; }

	// Lets DMA_DATA read or write GDS: only when every submission owns a GDS partition
	// (NativeSubmitter::HasGds), otherwise such packets are refused.
	void SetGdsAvailable(bool available) { m_gds = available; }

	// Draws whose pixel shader (SPI_SHADER_PGM_LO/HI_PS) is in `shaders` are left out of the IB, for
	// bisecting a hang down to one shader; everything else in the DCB still runs natively.
	void SetSkippedPixelShaders(std::set<uint64_t> shaders) { m_skip_ps = std::move(shaders); }

	// Packets of these opcodes are left out of the IB (bisecting a hang down to a packet type).
	void SetSkippedOpcodes(std::set<uint32_t> ops) { m_skip_ops = std::move(ops); }

	// Compact references (default off): instead of every GpuRef, only the 16 MiB buckets they touch
	// (ref_buckets); refs then keeps just the index buffers. What residency needs, without sorting a
	// list that grows with every draw.
	void SetCompactRefs(bool compact) { m_compact_refs = compact; }

	// Diagnostics: report every draw/dispatch whose resources reach [addr, addr + size) in
	// Pm4TranslateResult::find_hits (user data, then tables two levels deep). 0 size: off.
	void SetFindRange(uint64_t addr, uint64_t size) {
		m_find_lo = addr;
		m_find_hi = addr + size;
	}

	// Draws with this pixel shader are reported in Pm4TranslateResult::inspected (0: none).
	void SetInspectedPixelShader(uint64_t ps) { m_inspect_ps = ps; }

	// UCONFIG registers the backend is allowed to write. Everything else is dropped, because one
	// experiment writing other UCONFIG registers froze the whole board (see the probe README).
	static bool UconfigAllowed(uint32_t normalized_offset);

private:
	void     EmitPreamble(std::vector<uint32_t>& ib, Pm4TranslateStats& stats) const;
	uint32_t PatchContext(uint32_t offset, uint32_t value) const; // applies the depth experiment

	std::map<uint32_t, uint32_t> m_cx; // context registers (SET_CONTEXT_REG space)
	std::map<uint32_t, uint32_t> m_cx_defaults; // what a context clear restores (the seeded defaults)
	std::map<uint32_t, uint32_t> m_saved_cx;    // context pushed by an Agc context-state push
	bool                         m_cx_pushed = false;
	std::map<uint32_t, uint32_t> m_sh; // SH registers (SET_SH_REG space)
	std::map<uint32_t, uint32_t> m_uc; // whitelisted UCONFIG registers
	bool                         m_gds = false;
	DepthDebugMode               m_depth_mode = DepthDebugMode::None;
	bool                         m_no_dcc     = false;
	uint32_t                     m_target_mask = 0xffffffffu;
	bool                         m_no_cmask_clear = false;
	bool                         m_compute_queue  = false;
	uint64_t                     m_fence_va       = 0;
	uint32_t                     m_fence_seq      = 0;
	std::set<uint64_t>           m_skip_ps;
	std::set<uint32_t>           m_skip_ops;
	uint64_t                     m_inspect_ps = 0;
	bool                         m_compact_refs = false;
	std::vector<uint64_t>        m_bucket_bits;      // compact refs: one bit per 16 MiB bucket below 1 TiB
	uint64_t                     m_find_lo    = 0;
	uint64_t                     m_find_hi    = 0;
};

} // namespace Libs::Graphics::NativeGpu

#endif // EMULATOR_SRC_GRAPHICS_NATIVE_GPU_NATIVEPM4_H_
