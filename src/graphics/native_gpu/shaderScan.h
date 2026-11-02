#ifndef EMULATOR_SRC_GRAPHICS_NATIVE_GPU_SHADERSCAN_H_
#define EMULATOR_SRC_GRAPHICS_NATIVE_GPU_SHADERSCAN_H_

#include "graphics/native_gpu/shaderPatcher.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <span>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// Guest shader scanner for the BC-250 native GPU backend: reports what in a PS5 shader may run differently on
// the BC-250 than on the PS5. Both GPUs are gfx1013, so a shader runs as written; the scanner only flags
// what the BC-250 is known or suspected to do differently (an opcode outside the gfx1013 set, v_dot2c_f32_f16
// which the BC-250 executes as 0, hwregs and messages whose value or effect depends on the kernel and the
// harvest), and what makes a static view of the code incomplete (indirect branches, code past s_endpgm).
//
// The walk follows control flow from dword 0 instead of sweeping linearly: s_branch, the conditional branches,
// s_call_b64 and s_subvector_loop_* targets are decoded, data after s_endpgm is not. Only words on reachable
// paths are read, each within the window (or the registered size) and, without a registered size, only on
// pages the caller says are readable. The scanner never changes the code: patching v_dot2c_f32_f16
// (PatchDot2c) is report-only for now, since 9,996 shaders of six titles hold none.

namespace Libs::Graphics::NativeGpu {

// The order is the table order in shaderIsaGfx1013.inc (tools/bc250-shader-isa/gen_gfx1013_isa.py).
enum class ScanEnc : uint8_t {
	SOP1,
	SOP2,
	SOPK,
	SOPC,
	SOPP,
	SMEM,
	VOP1,
	VOP2,
	VOPC,
	VOP3,
	VOP3P,
	VINTRP,
	DS,
	MUBUF,
	MTBUF,
	MIMG,
	FLAT,
	SCRATCH,
	GLOBAL,
	EXP,
	Count,
	Invalid = Count,
};

enum class ScanKind : uint8_t {
	Dot2c,            // VOP2 0x02 v_dot2c_f32_f16: the BC-250 writes 0 (PatchDot2c can rewrite it)
	DotOther,         // VOP2 0x0d, VOP3P 0x13..0x19: not gfx1013 opcodes, behaviour on the BC-250 not probed
	UnknownOpcode,    // (enc, op) not in shaderIsaGfx1013.inc, the gfx10.3-only csub/addtid ops included
	Hwreg,            // s_getreg/s_setreg/s_setreg_imm32 of an id outside the wave-local set; detail = hwreg id
	InvalidDpp,       // DPP16 dpp_ctrl outside the gfx10 set; detail = dpp_ctrl
	InvalidSdwa,      // SDWA sel > 6 or dst_unused 3; detail = 1 dst_sel, 2 dst_unused, 3 src0_sel, 4 src1_sel
	Bvh,              // image_bvh(64)_intersect_ray: gfx1013 has it, bit-exactness against the PS5 unverified
	MsaaLoad,         // image_msaa_load
	SendMsg,          // s_sendmsg(halt) with a message id outside GS, GS_DONE, ORDERED_PS_DONE, EARLY_PRIM_DEALLOC,
	                  // GS_ALLOC_REQ; detail = message id
	Trap,             // s_trap, s_sethalt, s_ttracedata(_imm), s_cbranch_cdbg*
	ApertureLiteral,  // a literal 0x80000000 / 0x70000000 (the PS5 SH_MEM_BASES high dwords) in a shader with FLAT ops
	Gds,              // ds_* with GDS, ds_ordered_count, ds_gws_* (the GDS policy acts on these)
	Timer,            // s_memtime / s_memrealtime
	IndirectBranch,   // s_setpc_b64, s_swappc_b64, s_rfe_b64: the target is not followed
	BranchOutOfRange, // a target before dword 0, at or past the limit, or on a page the caller says is unreadable
	OverlapDecode,    // a branch target inside an instruction decoded from another start, or the reverse
	CodeEndReached,   // s_code_end on a reachable path
	PastFirstEnd,     // a reachable instruction after the first s_endpgm (out-of-line blocks)
	PcRelOutside,     // s_getpc_b64 + s_add_u32 literal that points outside the registered image
	DecodeError,      // an undecodable word on a reachable path
	Count,
};

enum class ScanSeverity : uint8_t { Info, Warn, Error };

struct ScanFinding {
	ScanKind     kind   = ScanKind::DecodeError;
	ScanSeverity sev    = ScanSeverity::Info;
	ScanEnc      enc    = ScanEnc::Invalid;
	uint16_t     op     = 0;
	uint16_t     detail = 0; // kind-specific: hwreg id, message id, dpp_ctrl, literal high half
	uint32_t     dw     = 0; // first (lowest) dword offset seen
	uint32_t     word   = 0; // the instruction's first word there
	uint32_t     count  = 0; // instructions with this (kind, enc, op, detail, sev)
};

struct ShaderScanOptions {
	uint32_t max_dwords    = 1u << 16u; // window without a registered size (as CountDot2c)
	uint32_t known_dwords  = 0;         // the AGC shader_size / 4 when the shader is registered: bounds every read
	uint32_t safe_dwords   = 0;         // words known readable without asking (what CountDot2c already swept)
	bool     ps5_apertures = true;      // SH_MEM_BASES 0x80007000 is set as on the PS5: ApertureLiteral is Info, else Warn
	// Without known_dwords: asked once per 4 KiB page (of the host address of the dword) before a word at or past
	// safe_dwords on it is read; false stops the path there (BranchOutOfRange). Empty: the whole window is readable.
	std::function<bool(uint32_t dw)> readable;
};

struct ShaderScanResult {
	uint32_t image_dwords   = 0; // known_dwords, else the end of the furthest reachable instruction
	uint32_t instructions   = 0; // reachable instructions decoded
	uint32_t max_vgpr_field = 0; // largest VGPR number in a VOP vdst field: diagnostics only
	uint32_t first_end_dw   = UINT32_MAX; // lowest dword of a reachable s_endpgm (any form)
	bool     ended          = false; // a path ended in s_endpgm
	bool     truncated      = false; // a path ran into the limit without ending
	bool     decode_error   = false;
	int64_t  pcrel_min      = INT64_MAX; // byte offsets from the image start that s_getpc_b64 + s_add_u32 form
	int64_t  pcrel_max      = INT64_MIN;
	std::vector<uint8_t>     insn_start; // 1 bit per dword of the window: reachable instruction starts
	Dot2cScan                legacy;     // CountDot2c's counters over the reachable instructions
	std::vector<ScanFinding> findings;   // one per (kind, enc, op, detail, sev), count aggregated
	std::array<uint32_t, 20> by_enc {};  // reachable instructions per ScanEnc

	[[nodiscard]] bool IsStart(uint32_t dw) const {
		return dw / 8u < insn_start.size() && ((insn_start[dw / 8u] >> (dw % 8u)) & 1u) != 0;
	}
	[[nodiscard]] uint32_t Count(ScanKind k) const;                  // instructions of that kind
	[[nodiscard]] uint32_t Count(ScanKind k, ScanSeverity s) const;  // of that kind and severity
	[[nodiscard]] ScanSeverity Worst() const;                        // Info when there are no findings
};

ShaderScanResult ScanShader(std::span<const uint32_t> mem, const ShaderScanOptions& opt);

// The instruction's encoding and opcode as shaderIsaGfx1013.inc numbers them (Invalid when the top bits match
// no encoding). `code` must hold the whole instruction (InstructionDwords).
ScanEnc ScanDecodeOpcode(std::span<const uint32_t> code, uint16_t* op);
bool        Gfx1013HasOpcode(ScanEnc enc, uint16_t op);
const char* Gfx1013Mnemonic(ScanEnc enc, uint16_t op); // nullptr when not a gfx1013 opcode
const char* ScanEncName(ScanEnc enc);
const char* ScanKindName(ScanKind kind);
const char* ScanSeverityName(ScanSeverity sev);

// Once-per-code reporting of ScanShader results (KYTY_NATIVE_GPU_SHADER_SCAN). Not thread-safe: the caller
// serialises (DetectShaders runs under the shadow's mutex).
class ShaderScanReporter {
public:
	using Hasher = uint64_t (*)(const void* data, size_t bytes);

	struct Registered {
		uint32_t size_bytes = 0; // AGC shader_size
		uint64_t hash       = 0; // XXH3_64 of those bytes, as shader.cpp's ShaderMap keeps it
		int      type       = -1; // Prospero::ShaderBinaryType, -1 unknown
	};

	// `level` 1 logs Warn and Error findings, 2 also Info findings and keeps an opcode histogram. `hasher`
	// hashes an unregistered shader's reachable image.
	ShaderScanReporter(int level, Hasher hasher, std::FILE* out = stderr);

	// Scans the shader at `va` once per code hash: `window` is the guest memory from va (the registered size or
	// the 64 Ki dword window), `reg` the registration when shader.cpp knows the address, `safe_dwords` and
	// `readable` as in ShaderScanOptions. Returns true when this code had not been scanned before.
	bool Detect(uint64_t va, std::span<const uint32_t> window, const Registered* reg, uint32_t safe_dwords,
	            const std::function<bool(uint32_t dw)>& readable, const char* dump_dir);

	void PrintSummary(const char* why);
	void WriteHistogram(const char* dir) const; // shader_hist.txt (level 2)

	[[nodiscard]] uint32_t Shaders() const { return m_shaders; }
	[[nodiscard]] uint32_t Duplicates() const { return m_duplicates; }
	[[nodiscard]] uint32_t Flagged() const { return m_flagged; }
	[[nodiscard]] uint32_t Lines() const { return m_lines; }
	[[nodiscard]] uint32_t KindInstructions(ScanKind k) const { return m_kind_insns[static_cast<size_t>(k)]; }
	[[nodiscard]] uint32_t KindShaders(ScanKind k) const { return m_kind_shaders[static_cast<size_t>(k)]; }

private:
	int                          m_level;
	Hasher                       m_hasher;
	std::FILE*                   m_out;
	std::unordered_set<uint64_t> m_hashes;
	uint32_t                     m_shaders    = 0;
	uint32_t                     m_registered = 0;
	uint32_t                     m_duplicates = 0;
	uint32_t                     m_flagged    = 0;
	uint32_t                     m_flagged_registered = 0;
	uint32_t                     m_lines      = 0;
	uint64_t                     m_scan_ns    = 0;
	std::array<uint32_t, static_cast<size_t>(ScanKind::Count)> m_kind_insns {};
	std::array<uint32_t, static_cast<size_t>(ScanKind::Count)> m_kind_shaders {};
	// Level 2: (enc << 16 | op) -> {instructions, shaders}.
	std::unordered_map<uint32_t, std::pair<uint64_t, uint32_t>> m_hist;
};

// The emulator's hook (shaderScanNative.cpp, not part of the unit tests): KYTY_NATIVE_GPU_SHADER_SCAN, read
// once (0 off, 1 report Warn/Error once per code hash with a summary, default; 2 also Info and shader_hist.txt
// in KYTY_NATIVE_GPU_DUMP_DIR). For a shader shader.cpp does not know the size of, the first `swept_dwords`
// (CountDot2c's sweep) are known readable and `readable(addr)` says whether a guest page past them is mapped.
void ShaderScanDetect(uint64_t va, uint32_t swept_dwords, const std::function<bool(uint64_t addr)>& readable);

} // namespace Libs::Graphics::NativeGpu

#endif // EMULATOR_SRC_GRAPHICS_NATIVE_GPU_SHADERSCAN_H_
