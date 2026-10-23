#ifndef EMULATOR_SRC_GRAPHICS_NATIVE_GPU_SHADERPATCHER_H_
#define EMULATOR_SRC_GRAPHICS_NATIVE_GPU_SHADERPATCHER_H_

#include <cstdint>
#include <span>
#include <vector>

// Guest shader ISA patching for the BC-250 native GPU backend. Guest (PS5) shaders run
// unmodified on the host GPU except for opcodes the BC-250 executes differently:
//
//   v_dot2c_f32_f16 (VOP2 0x02): BC-250 executes it without a fault but writes 0.
//
// A supported site is replaced in place by an s_branch to a stub appended after the code; the
// stub does the same math with two v_fma_mix_f32 and branches back. Instruction offsets inside
// the original code never move, so branches, s_getpc-relative data and jump tables stay valid.
//
// v_fma_mix_f32 is VOP3P: it cannot carry a 32-bit literal and gfx10 VOP3P has no DPP, so the
// literal (8-byte), DPP16/DPP8 and vdst-aliasing forms are reproduced by first materialising the
// effective src0 (or a preserved copy of vdst) into one scratch VGPR at the top of the stub. The
// scratch register is v[num_vgprs] (the first VGPR past the shader's own allocation); when any
// patched site needs it the caller must raise COMPUTE_PGM_RSRC1 VGPRS to cover num_vgprs + 1.

namespace Libs::Graphics::NativeGpu {

// Length in dwords of the gfx10 instruction at code[0], including literals, DPP/SDWA words and
// MIMG NSA address words. 0 when the encoding is unknown or code is too short.
uint32_t InstructionDwords(std::span<const uint32_t> code);

struct ShaderPatchResult {
	std::vector<uint32_t> code;                 // original words with sites rewritten, then stubs
	uint32_t              patched_sites     = 0;
	uint32_t              unsupported_sites = 0; // left untouched: SDWA and some aliasing combos
	uint32_t              scratch_vgprs     = 0; // extra VGPRs the stubs use past num_vgprs (0 or 1)
	bool                  decode_error      = false;
	uint32_t              decode_error_dw   = 0; // dword offset of the undecodable instruction
};

// `code` must cover exactly the instruction stream (a linear sweep decodes every word), not data
// appended after the last instruction. `num_vgprs` is the VGPR count the shader allocates
// (COMPUTE_PGM_RSRC1 VGPRS decoded to a register count); the stubs use v[num_vgprs] as scratch
// when needed. On a decode error or an out-of-range branch nothing is patched and `code` is
// returned unchanged with scratch_vgprs == 0.
ShaderPatchResult PatchDot2c(std::span<const uint32_t> code, uint32_t num_vgprs);

struct Dot2cScan {
	uint32_t sites        = 0;     // v_dot2c_f32_f16 instructions found before s_endpgm
	uint32_t instructions = 0;     // instructions decoded
	bool     ended        = false; // true if an s_endpgm terminated the sweep
	bool     decode_error = false; // true if an instruction could not be decoded
};

// Linear-sweeps a shader whose exact length is unknown, counting v_dot2c_f32_f16 sites. The sweep
// stops at the first s_endpgm, a decode error, or after `max_dwords` dwords. Used to detect (not
// patch) the opcode the BC-250 mis-executes in shaders referenced by a command buffer.
Dot2cScan CountDot2c(std::span<const uint32_t> code, uint32_t max_dwords = 1u << 16u);

} // namespace Libs::Graphics::NativeGpu

#endif // EMULATOR_SRC_GRAPHICS_NATIVE_GPU_SHADERPATCHER_H_
