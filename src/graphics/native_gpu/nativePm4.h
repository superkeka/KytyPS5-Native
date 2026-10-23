#ifndef EMULATOR_SRC_GRAPHICS_NATIVE_GPU_NATIVEPM4_H_
#define EMULATOR_SRC_GRAPHICS_NATIVE_GPU_NATIVEPM4_H_

#include <cstdint>
#include <cstdio>
#include <map>
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
	uint32_t dispatches        = 0; // IT_DISPATCH_DIRECT
	uint32_t acquire_mem       = 0; // IT_ACQUIRE_MEM
	uint32_t event_write       = 0; // IT_EVENT_WRITE
	uint32_t write_data        = 0; // IT_WRITE_DATA to memory passed through
	uint32_t clear_state       = 0; // IT_CLEAR_STATE
	uint32_t uconfig_skipped   = 0; // non-whitelisted UCONFIG writes dropped for board safety
	uint32_t preamble_cx       = 0; // context registers emitted in the IB preamble
	uint32_t preamble_sh       = 0; // SH registers emitted in the IB preamble
	uint32_t preamble_uc       = 0; // UCONFIG registers emitted in the IB preamble
};

struct Pm4TranslateResult {
	bool                  ok         = true;  // false when an unknown packet forced a refusal
	uint32_t              unknown_op = 0;     // the opcode that could not be translated
	uint32_t              unknown_dw = 0;     // its dword offset in the DCB
	std::vector<uint32_t> ib;                 // the self-contained IB (empty when !ok)
	Pm4TranslateStats     stats;
	std::vector<uint64_t> shader_addrs;       // shader base VAs from PGM_LO writes this DCB
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

	// UCONFIG registers the backend is allowed to write. Everything else is dropped, because one
	// experiment writing other UCONFIG registers froze the whole board (see the probe README).
	static bool UconfigAllowed(uint32_t normalized_offset);

private:
	void EmitPreamble(std::vector<uint32_t>& ib, Pm4TranslateStats& stats) const;

	std::map<uint32_t, uint32_t> m_cx; // context registers (SET_CONTEXT_REG space)
	std::map<uint32_t, uint32_t> m_sh; // SH registers (SET_SH_REG space)
	std::map<uint32_t, uint32_t> m_uc; // whitelisted UCONFIG registers
};

} // namespace Libs::Graphics::NativeGpu

#endif // EMULATOR_SRC_GRAPHICS_NATIVE_GPU_NATIVEPM4_H_
