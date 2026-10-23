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

namespace Libs::Graphics::NativeGpu {

// Length in dwords of the gfx10 instruction at code[0], including literals, DPP/SDWA words and
// MIMG NSA address words. 0 when the encoding is unknown or code is too short.
uint32_t InstructionDwords(std::span<const uint32_t> code);

struct ShaderPatchResult {
	std::vector<uint32_t> code;                 // original words with sites rewritten, then stubs
	uint32_t              patched_sites     = 0;
	uint32_t              unsupported_sites = 0; // left untouched: literal/DPP/SDWA/aliasing forms
	bool                  decode_error      = false;
	uint32_t              decode_error_dw   = 0; // dword offset of the undecodable instruction
};

// `code` must cover exactly the instruction stream (a linear sweep decodes every word), not data
// appended after the last instruction. On a decode error or an out-of-range branch nothing is
// patched and `code` is returned unchanged.
ShaderPatchResult PatchDot2c(std::span<const uint32_t> code);

} // namespace Libs::Graphics::NativeGpu

#endif // EMULATOR_SRC_GRAPHICS_NATIVE_GPU_SHADERPATCHER_H_
