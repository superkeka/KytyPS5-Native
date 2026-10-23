#ifndef EMULATOR_SRC_GRAPHICS_NATIVE_GPU_NATIVEPM4_H_
#define EMULATOR_SRC_GRAPHICS_NATIVE_GPU_NATIVEPM4_H_

#include <cstdint>
#include <cstdio>
#include <map>
#include <set>
#include <span>
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
	uint32_t set_base          = 0; // IT_SET_BASE passed through (indirect argument base)
	uint32_t pfp_sync_me       = 0; // IT_PFP_SYNC_ME passed through
	uint32_t acquire_mem       = 0; // IT_ACQUIRE_MEM
	uint32_t event_write       = 0; // IT_EVENT_WRITE
	uint32_t write_data_dropped = 0; // IT_WRITE_DATA to memory, left to Kyty (see Translate)
	uint32_t dma_data          = 0; // IT_DMA_DATA memory/immediate -> memory passed through
	uint32_t waits_dropped     = 0; // WAIT_REG_MEM(_64) on memory, dropped in shadow mode
	uint32_t waits_unmet       = 0; // of those, condition not yet true when the DCB was translated
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

	bool operator<(const GpuRef& o) const {
		return addr != o.addr ? addr < o.addr : (size != o.size ? size < o.size : pointer < o.pointer);
	}
	bool operator==(const GpuRef& o) const { return addr == o.addr && size == o.size && pointer == o.pointer; }
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

	// Writes the shadowed registers as "cx|sh|uc <offset> <value>" lines (dword offsets within each
	// register space), for comparing against the state Kyty sees.
	void DumpShadow(std::FILE* f) const;

	// Lets DMA_DATA read or write GDS: only when every submission owns a GDS partition
	// (NativeSubmitter::HasGds), otherwise such packets are refused.
	void SetGdsAvailable(bool available) { m_gds = available; }

	// UCONFIG registers the backend is allowed to write. Everything else is dropped, because one
	// experiment writing other UCONFIG registers froze the whole board (see the probe README).
	static bool UconfigAllowed(uint32_t normalized_offset);

private:
	void EmitPreamble(std::vector<uint32_t>& ib, Pm4TranslateStats& stats) const;

	std::map<uint32_t, uint32_t> m_cx; // context registers (SET_CONTEXT_REG space)
	std::map<uint32_t, uint32_t> m_sh; // SH registers (SET_SH_REG space)
	std::map<uint32_t, uint32_t> m_uc; // whitelisted UCONFIG registers
	bool                         m_gds = false;
};

} // namespace Libs::Graphics::NativeGpu

#endif // EMULATOR_SRC_GRAPHICS_NATIVE_GPU_NATIVEPM4_H_
