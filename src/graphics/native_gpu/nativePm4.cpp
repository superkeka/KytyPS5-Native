#include "graphics/native_gpu/nativePm4.h"

#include <algorithm>

namespace Libs::Graphics::NativeGpu {

namespace {

// Type-3 PM4 opcodes, mirroring src/graphics/guest_gpu/pm4.h (kept local so the translator has no
// build dependency on the rest of Kyty and stays unit-testable off the board).
constexpr uint32_t IT_NOP                      = 0x10;
constexpr uint32_t IT_CLEAR_STATE              = 0x12;
constexpr uint32_t IT_SET_BASE                 = 0x11;
constexpr uint32_t IT_INDEX_BUFFER_SIZE        = 0x13;
constexpr uint32_t IT_DISPATCH_DIRECT          = 0x15;
constexpr uint32_t IT_DISPATCH_INDIRECT        = 0x16;
constexpr uint32_t IT_INDEX_BASE               = 0x26;
constexpr uint32_t IT_DRAW_INDEX_2             = 0x27;
constexpr uint32_t IT_DRAW_INDEX_AUTO          = 0x2D;
constexpr uint32_t IT_NUM_INSTANCES            = 0x2F;
constexpr uint32_t IT_DRAW_INDEX_OFFSET_2      = 0x35;
constexpr uint32_t IT_CONTEXT_CONTROL          = 0x28;
constexpr uint32_t IT_WAIT_REG_MEM             = 0x3C;
constexpr uint32_t IT_WAIT_REG_MEM_64          = 0x93;
constexpr uint32_t IT_DMA_DATA                 = 0x50;
constexpr uint32_t IT_WRITE_DATA               = 0x37;
constexpr uint32_t IT_PFP_SYNC_ME              = 0x42;
constexpr uint32_t IT_EVENT_WRITE              = 0x46;
constexpr uint32_t IT_ACQUIRE_MEM             = 0x58;
constexpr uint32_t IT_SET_SH_REG_INDIRECT      = 0x63;
constexpr uint32_t IT_SET_UCONFIG_REG_INDIRECT = 0x64;
constexpr uint32_t IT_SET_CONTEXT_REG          = 0x69;
constexpr uint32_t IT_SET_SH_REG               = 0x76;
constexpr uint32_t IT_SET_UCONFIG_REG          = 0x79;
constexpr uint32_t IT_SET_UCONFIG_REG_INDEX    = 0x7A;
constexpr uint32_t IT_SET_CONTEXT_REG_INDIRECT = 0x9F;

// IT_NOP marker codes carried in the register field (Kyty custom codes in pm4.h).
constexpr uint32_t R_WAIT_FLIP_DONE = 0x06;
constexpr uint32_t R_FLIP           = 0x17;
constexpr uint32_t R_RELEASE_MEM    = 0x18;

// High bits of a register offset select an indexed SET variant; the register itself is the rest.
constexpr uint32_t kRegisterSelectorMask = 0x70000000u;

// Sentinels the guest uses inside register lists (pm4.h / pm4Handlers.cpp).
constexpr uint32_t kSentinelOffset = 0xffffffffu;
constexpr uint32_t kShNop          = 0x280u;
constexpr uint32_t kUcNop          = 0xA2u;

// SH registers at this offset and above live in the compute SH space; the shader-type bit of their
// SET_SH_REG packet must be set. Below it they are graphics SH registers.
constexpr uint32_t kShComputeBase = 0x200u;

constexpr uint32_t Normalize(uint32_t raw) { return raw & ~kRegisterSelectorMask; }

// SH offsets of the per-stage shader program base-address low dword (pm4.h SPI_SHADER_PGM_LO_* and
// COMPUTE_PGM_LO). The high dword is the next register; the base VA is (lo << 8) | (hi << 40).
// Render target and depth registers (context offsets) whose values locate guest memory the GPU
// writes. CB_COLORn registers repeat every 15 dwords; the *_EXT registers hold address bits 47:40.
constexpr uint32_t kCbColor0Base     = 0x318;
constexpr uint32_t kCbColorStride    = 0xF;
constexpr uint32_t kCbColor0View     = 0x31B;
constexpr uint32_t kCbColor0Cmask    = 0x31F;
constexpr uint32_t kCbColor0Fmask    = 0x321;
constexpr uint32_t kCbColor0DccBase  = 0x325;
constexpr uint32_t kCbColor0BaseExt  = 0x390;
constexpr uint32_t kCbColor0CmaskExt = 0x398;
constexpr uint32_t kCbColor0FmaskExt = 0x3A0;
constexpr uint32_t kCbColor0DccExt   = 0x3A8;
constexpr uint32_t kCbColor0Attrib2  = 0x3B0;
constexpr uint32_t kDbDepthView      = 0x002;
constexpr uint32_t kDbHtileDataBase  = 0x005;
constexpr uint32_t kDbDepthSizeXy    = 0x007;
constexpr uint32_t kDbZReadBase      = 0x012; // Z read, stencil read, Z write, stencil write
constexpr uint32_t kDbZReadBaseHi    = 0x01A; // their bits 47:40, then the HTILE base's
constexpr uint32_t kDbHtileBaseHi    = 0x01E;

bool IsRenderTargetRegister(uint32_t off) {
	return (off >= kCbColor0Base && off < kCbColor0Base + 8 * kCbColorStride) ||
	       (off >= kCbColor0BaseExt && off < kCbColor0Attrib2 + 8) || off == kDbDepthView ||
	       off == kDbHtileDataBase || off == kDbDepthSizeXy || (off >= kDbZReadBase && off <= kDbHtileBaseHi);
}

// Sizes for references whose extent the translator cannot know: a texture-like base found in
// shader user data, a metadata surface (CMASK, FMASK, DCC, HTILE) and a render target without
// dimensions. Generous on purpose: an overestimate only makes more guest memory resident.
constexpr uint64_t kTextureSpan  = 32ull << 20u;
constexpr uint64_t kMetadataSpan = 8ull << 20u;
constexpr uint64_t kSurfaceCap   = 512ull << 20u;

// SH registers that hold shader user data (where resource pointers and inline descriptors live):
// each graphics stage has a 0x40-dword block (PS 0x000, VS 0x040, GS 0x080, ES 0x0C0, HS 0x100,
// LS 0x140) with SPI_SHADER_USER_DATA_*_0..31 at +0x0C and USER_DATA_ADDR_LO/HI at +0x02;
// COMPUTE_USER_DATA_0..15 are at 0x240.
bool IsUserDataRegister(uint32_t off) {
	if (off >= 0x240u && off < 0x250u) {
		return true;
	}
	if (off >= 0x180u) {
		return false;
	}
	const uint32_t in_stage = off & 0x3fu;
	return (in_stage >= 0x0Cu && in_stage <= 0x2Bu) || in_stage == 0x02u || in_stage == 0x03u;
}

bool IsShaderPgmLo(uint32_t sh_offset) {
	switch (sh_offset) {
		case 0x8:   // SPI_SHADER_PGM_LO_PS
		case 0x88:  // SPI_SHADER_PGM_LO_GS
		case 0xC8:  // SPI_SHADER_PGM_LO_ES
		case 0x108: // SPI_SHADER_PGM_LO_HS
		case 0x148: // SPI_SHADER_PGM_LO_LS
		case 0x20C: // COMPUTE_PGM_LO
			return true;
		default: return false;
	}
}

// PacketSizeDw from pm4.h: the filler NOP (count 0x3fff, op NOP) is a single dword; otherwise the
// 14-bit count plus two gives the total dwords including the header.
uint32_t PacketSizeDw(uint32_t header) {
	return (header & 0x3fffff00u) == 0x3fff1000u ? 1u : (((header >> 16u) & 0x3fffu) + 2u);
}

// Type-3 header. body_dw is the dword count after the header; compute sets the shader-type bit for
// CS SH-space writes and dispatches, matching Mesa's PKT3 predicate/shader-type encoding.
uint32_t Pkt3(uint32_t op, uint32_t body_dw, bool compute) {
	return (3u << 30u) | (((body_dw - 1u) & 0x3fffu) << 16u) | ((op & 0xffu) << 8u) |
	       (compute ? (1u << 1u) : 0u);
}

void EmitSet(std::vector<uint32_t>& ib, uint32_t op, uint32_t offset, uint32_t value, bool compute) {
	ib.push_back(Pkt3(op, 2, compute));
	ib.push_back(offset);
	ib.push_back(value);
}

// UCONFIG writes. On GFX9+ the CP keeps its own copy of VGT_PRIMITIVE_TYPE and VGT_INDEX_TYPE and
// only updates it from SET_UCONFIG_REG_INDEX with index 1 / 2 (RADV: radeon_set_uconfig_reg_idx);
// a plain SET_UCONFIG_REG leaves the draw engine with a stale primitive or index type.
void EmitSetUconfig(std::vector<uint32_t>& ib, uint32_t offset, uint32_t value) {
	switch (offset) {
		case 0x242: EmitSet(ib, IT_SET_UCONFIG_REG_INDEX, (1u << 28u) | offset, value, false); break;
		case 0x243: EmitSet(ib, IT_SET_UCONFIG_REG_INDEX, (2u << 28u) | offset, value, false); break;
		default: EmitSet(ib, IT_SET_UCONFIG_REG, offset, value, false); break;
	}
}

// WAIT_REG_MEM compare function (CONTROL bits 2:0), as Kyty's command processor evaluates it.
bool WaitConditionMet(uint32_t function, uint64_t value, uint64_t reference) {
	switch (function) {
		case 0: return true;
		case 1: return value < reference;
		case 2: return value <= reference;
		case 3: return value == reference;
		case 4: return value != reference;
		case 5: return value >= reference;
		case 6: return value > reference;
		default: return false;
	}
}

// DMA_DATA between guest memory (or an immediate source) only: DST_SEL 0/3 (memory), SRC_SEL 0/2/3
// (memory or immediate data), and both address-space bits (SAS bit 26, DAS bit 27 of the command
// dword) clear. GDS and register transfers are refused.
bool DmaDataAllowed(uint32_t control, uint32_t command, bool gds) {
	const uint32_t dst_sel = (control >> 20u) & 0x3u;
	const uint32_t src_sel = (control >> 29u) & 0x3u;
	const bool     dst_ok  = dst_sel == 0u || dst_sel == 3u || (gds && dst_sel == 1u);
	const bool     src_ok  = src_sel == 0u || src_sel == 2u || src_sel == 3u || (gds && src_sel == 1u);
	return dst_ok && src_ok && (command & (3u << 26u)) == 0u;
}

// WRITE_DATA DST_SEL values that target memory (1 = memory via GRBM, 2 = TC/L2, 5 = memory).
// DST_SEL 0 writes a register and is refused, like any non-whitelisted register write.
bool WriteDataToMemory(uint32_t control) {
	const uint32_t dst_sel = (control >> 8u) & 0xfu;
	return dst_sel == 1u || dst_sel == 2u || dst_sel == 5u;
}

} // namespace

bool Pm4Translator::UconfigAllowed(uint32_t normalized_offset) {
	switch (normalized_offset) {
		case 0x242: // VGT_PRIMITIVE_TYPE
		case 0x243: // VGT_INDEX_TYPE
		case 0x24A: // GE_INDX_OFFSET
		case 0x24B: // GE_MULTI_PRIM_IB_RESET_EN (primitive restart)
		case 0x25B: // GE_CNTL
		case 0x25F: // GE_STEREO_CNTL
		case 0x262: // GE_USER_VGPR_EN
			// Per-draw geometry-engine state; radeonsi writes all of these on gfx10 itself.
			return true;
		default: return false;
	}
}

void Pm4Translator::DumpShadow(std::FILE* f) const {
	for (const auto& [offset, value]: m_cx) {
		std::fprintf(f, "cx 0x%03x 0x%08x\n", offset, value);
	}
	for (const auto& [offset, value]: m_sh) {
		std::fprintf(f, "sh 0x%03x 0x%08x\n", offset, value);
	}
	for (const auto& [offset, value]: m_uc) {
		std::fprintf(f, "uc 0x%03x 0x%08x\n", offset, value);
	}
}

void Pm4Translator::SeedPs5Defaults() {
	struct Default {
		uint32_t offset;
		uint32_t value;
	};
	static constexpr Default kDefaults[] = {
	    // PA_CL_VTE_CNTL: X/Y/Z scale and offset enabled, VTX_W0_FMT (Kyty's PS5 default, 1087).
	    {0x206, 0x0000043Fu},
	    // PA_SC_MODE_CNTL_0: VPORT_SCISSOR_ENABLE (Kyty assumes the viewport scissor is on).
	    {0x292, 0x00000002u},
	    // PA_SC_AA_MASK_X0Y0_X1Y0 / X0Y1_X1Y1: every sample enabled.
	    {0x30E, 0xFFFFFFFFu},
	    {0x30F, 0xFFFFFFFFu},
	    // CB_COLOR_CONTROL: MODE = CB_NORMAL, ROP3 = copy.
	    {0x202, 0x00CC0010u},
	};
	for (const auto& d: kDefaults) {
		m_cx.emplace(d.offset, d.value);
	}
}

void Pm4Translator::ResetShadow() {
	m_cx.clear();
	m_sh.clear();
	m_uc.clear();
}

void Pm4Translator::EmitPreamble(std::vector<uint32_t>& ib, Pm4TranslateStats& stats) const {
	// RADV-style preamble: take back context ownership, clear to defaults, then replay the whole
	// shadowed state. Other processes share the gfx ring between our submissions, so nothing from
	// the previous DCB is assumed to survive on the hardware.
	ib.push_back(Pkt3(IT_CONTEXT_CONTROL, 2, false));
	ib.push_back(0x80000000u); // update load enables, load nothing (we set everything below)
	ib.push_back(0x80000000u); // update shadow enables, shadow nothing

	ib.push_back(Pkt3(IT_CLEAR_STATE, 1, false));
	ib.push_back(0u);

	for (const auto& [offset, value]: m_cx) {
		EmitSet(ib, IT_SET_CONTEXT_REG, offset, value, false);
		stats.preamble_cx++;
	}
	for (const auto& [offset, value]: m_sh) {
		EmitSet(ib, IT_SET_SH_REG, offset, value, offset >= kShComputeBase);
		stats.preamble_sh++;
	}
	for (const auto& [offset, value]: m_uc) {
		EmitSetUconfig(ib, offset, value);
		stats.preamble_uc++;
	}
}

Pm4TranslateResult Pm4Translator::Translate(std::span<const uint32_t> dcb) {
	Pm4TranslateResult r;

	// First pass: walk the DCB, refuse on anything unknown, and fold register writes into the
	// shadow while collecting the body packets. Nothing is emitted until the pass succeeds, so a
	// refused DCB leaves the shadow and the IB in a usable state for the next one.
	std::vector<uint32_t> body;
	body.reserve(dcb.size());
	Pm4TranslateStats stats;

	// A scratch copy of the shadow so a refusal mid-DCB does not leave it half-updated.
	// (Unknown packets are collected in r.unknown_ops and refuse the DCB once the walk is done.)
	auto cx = m_cx;
	auto sh = m_sh;
	auto uc = m_uc;

	std::vector<uint64_t> shaders;

	// Called after each SH write. The base VA needs both the PGM_LO and PGM_HI dwords; they are set
	// one after another, so resolve the shader whether this write was the LO or the following HI.
	auto note_shader = [&](uint32_t off) {
		uint32_t lo_off;
		if (IsShaderPgmLo(off)) {
			lo_off = off;
		} else if (off > 0 && IsShaderPgmLo(off - 1)) {
			lo_off = off - 1;
		} else {
			return;
		}
		auto lo = sh.find(lo_off);
		auto hi = sh.find(lo_off + 1);
		if (lo == sh.end() || hi == sh.end()) {
			return;
		}
		uint64_t va = (static_cast<uint64_t>(lo->second) << 8u) | (static_cast<uint64_t>(hi->second) << 40u);
		if (va != 0) {
			shaders.push_back(va);
		}
	};

	// Guest memory the GPU may touch (Pm4TranslateResult::refs). Collected at every draw and
	// dispatch from the registers that changed since the previous one: the whole shadow is live at
	// the first one, since state carries over from earlier DCBs.
	std::vector<GpuRef>   refs;
	std::vector<uint32_t> dirty_sh;
	uint64_t              index_base  = 0; // from INDEX_BASE, for draws that use the bound buffer
	uint64_t              index_bytes = 0; // from INDEX_BUFFER_SIZE (indices, at most 4 bytes each)
	bool                  all_sh_dirty = true;
	bool                  rt_dirty     = true;
	auto mark_cx = [&](uint32_t off) {
		if (IsRenderTargetRegister(off)) {
			rt_dirty = true;
		}
	};
	auto reg = [](const std::map<uint32_t, uint32_t>& regs, uint32_t off) -> uint32_t {
		auto it = regs.find(off);
		return it == regs.end() ? 0u : it->second;
	};
	auto add_ref = [&](uint64_t addr, uint64_t size, bool pointer) {
		if (addr != 0) {
			refs.push_back(GpuRef {addr, size, pointer});
		}
	};
	// A user-data dword pair can be a 48-bit pointer (V# and table pointers) or the 40-bit
	// 256-byte-aligned base of a T# (dword 0 = address >> 8, low byte of dword 1 = address >> 40);
	// a single dword can be address >> 8 (shader programs, inline image bases).
	auto add_pair = [&](uint32_t lo, uint32_t hi) {
		add_ref(static_cast<uint64_t>(lo) | (static_cast<uint64_t>(hi & 0xffffu) << 32u), 0, true);
		add_ref((static_cast<uint64_t>(lo) | (static_cast<uint64_t>(hi & 0xffu) << 32u)) << 8u, kTextureSpan, false);
	};
	auto collect_refs = [&]() {
		if (all_sh_dirty) {
			for (const auto& [off, value]: sh) {
				dirty_sh.push_back(off);
			}
			all_sh_dirty = false;
		}
		std::sort(dirty_sh.begin(), dirty_sh.end());
		dirty_sh.erase(std::unique(dirty_sh.begin(), dirty_sh.end()), dirty_sh.end());
		for (uint32_t off: dirty_sh) {
			if (!IsUserDataRegister(off)) {
				continue; // shader programs come through shader_addrs, the rest is configuration
			}
			const uint32_t value = reg(sh, off);
			add_ref(static_cast<uint64_t>(value) << 8u, kTextureSpan, false);
			if (IsUserDataRegister(off + 1) && sh.count(off + 1) != 0) {
				add_pair(value, reg(sh, off + 1));
			}
			if (off > 0 && IsUserDataRegister(off - 1) && sh.count(off - 1) != 0) {
				add_pair(reg(sh, off - 1), value);
			}
		}
		dirty_sh.clear();
		if (!rt_dirty) {
			return;
		}
		rt_dirty = false;
		auto base40 = [&](uint32_t lo_reg, uint32_t ext_reg) {
			return (static_cast<uint64_t>(reg(cx, lo_reg)) << 8u) |
			       (static_cast<uint64_t>(reg(cx, ext_reg) & 0xffu) << 40u);
		};
		for (uint32_t i = 0; i < 8; i++) {
			const uint32_t step    = i * kCbColorStride;
			const uint32_t attrib2 = reg(cx, kCbColor0Attrib2 + i);
			const uint32_t view    = reg(cx, kCbColor0View + step);
			const uint64_t width   = ((attrib2 >> 14u) & 0x3fffu) + 1u;
			const uint64_t height  = (attrib2 & 0x3fffu) + 1u;
			const uint64_t slices  = ((view >> 13u) & 0x1fffu) + 1u;
			// 16 bytes per pixel is the widest color format; mips add at most a third.
			const uint64_t size = std::min(width * height * 16u * slices * 4u / 3u, kSurfaceCap);
			add_ref(base40(kCbColor0Base + step, kCbColor0BaseExt + i), size, false);
			add_ref(base40(kCbColor0Cmask + step, kCbColor0CmaskExt + i), kMetadataSpan, false);
			add_ref(base40(kCbColor0Fmask + step, kCbColor0FmaskExt + i), kMetadataSpan, false);
			add_ref(base40(kCbColor0DccBase + step, kCbColor0DccExt + i), kMetadataSpan, false);
		}
		const uint32_t size_xy = reg(cx, kDbDepthSizeXy);
		const uint64_t dslices = ((reg(cx, kDbDepthView) >> 13u) & 0x7ffu) + 1u;
		const uint64_t dsize   = std::min((static_cast<uint64_t>(size_xy & 0x3fffu) + 1u) *
		                                      (((size_xy >> 16u) & 0x3fffu) + 1u) * 8u * dslices * 4u / 3u,
		                                  kSurfaceCap);
		for (uint32_t i = 0; i < 4; i++) {
			add_ref(base40(kDbZReadBase + i, kDbZReadBaseHi + i), dsize, false);
		}
		add_ref(base40(kDbHtileDataBase, kDbHtileBaseHi), kMetadataSpan, false);
	};

	auto fail = [&](uint32_t op, uint32_t dw) {
		r.ok         = false;
		r.unknown_op = op;
		r.unknown_dw = dw;
		r.ib.clear();
		r.stats = stats;
		return r;
	};

	uint32_t dw = 0;
	while (dw < dcb.size()) {
		const uint32_t header = dcb[dw];
		if ((header >> 30u) != 3u) {
			return fail(0xffffffffu, dw); // not a type-3 packet
		}
		const uint32_t size = PacketSizeDw(header);
		if (size == 0 || dw + size > dcb.size()) {
			return fail((header >> 8u) & 0xffu, dw);
		}
		const uint32_t op          = (header >> 8u) & 0xffu;
		const uint32_t body_count  = size - 1u;
		const uint32_t* const bptr = dcb.data() + dw + 1;
		stats.packets++;

		switch (op) {
			case IT_NOP: {
				stats.nops++;
				const uint32_t marker = (header >> 2u) & 0x3fu;
				if (marker == R_FLIP) {
					stats.marker_flip++;
				} else if (marker == R_RELEASE_MEM) {
					stats.marker_release++;
				} else if (marker == R_WAIT_FLIP_DONE) {
					stats.marker_wait_flip++;
				}
				break; // the CP ignores NOPs; present/sync markers are handled elsewhere later
			}

			case IT_SET_CONTEXT_REG: {
				if (body_count >= 1) {
					const uint32_t base = Normalize(bptr[0]);
					for (uint32_t i = 1; i < body_count; i++) {
						cx[base + (i - 1)] = bptr[i];
						mark_cx(base + (i - 1));
					}
				}
				body.insert(body.end(), dcb.begin() + dw, dcb.begin() + dw + size);
				stats.set_context++;
				break;
			}

			case IT_SET_SH_REG: {
				if (body_count >= 1) {
					const uint32_t base = bptr[0];
					for (uint32_t i = 1; i < body_count; i++) {
						sh[base + (i - 1)] = bptr[i];
						dirty_sh.push_back(base + (i - 1));
						note_shader(base + (i - 1));
					}
				}
				body.insert(body.end(), dcb.begin() + dw, dcb.begin() + dw + size);
				stats.set_sh++;
				break;
			}

			case IT_SET_UCONFIG_REG:
			case IT_SET_UCONFIG_REG_INDEX: {
				// Pass through only whitelisted UCONFIG registers; drop the rest for board safety.
				bool allowed = body_count >= 1;
				if (allowed) {
					const uint32_t base = (op == IT_SET_UCONFIG_REG_INDEX) ? (bptr[0] & 0x0fffffffu)
					                                                        : Normalize(bptr[0]);
					for (uint32_t i = 1; i < body_count && allowed; i++) {
						if (!UconfigAllowed(base + (i - 1))) {
							allowed = false;
						}
					}
					if (allowed) {
						for (uint32_t i = 1; i < body_count; i++) {
							uc[base + (i - 1)] = bptr[i];
						}
					}
				}
				if (allowed) {
					const uint32_t base = (op == IT_SET_UCONFIG_REG_INDEX) ? (bptr[0] & 0x0fffffffu)
					                                                        : Normalize(bptr[0]);
					for (uint32_t i = 1; i < body_count; i++) {
						EmitSetUconfig(body, base + (i - 1), bptr[i]);
					}
					if (op == IT_SET_UCONFIG_REG_INDEX) {
						stats.set_uconfig_index++;
					} else {
						stats.set_uconfig++;
					}
				} else {
					stats.uconfig_skipped++;
					if (body_count >= 1) {
						r.skipped_uconfig.insert((op == IT_SET_UCONFIG_REG_INDEX) ? (bptr[0] & 0x0fffffffu)
						                                                          : Normalize(bptr[0]));
					}
				}
				break;
			}

			case IT_SET_CONTEXT_REG_INDIRECT:
			case IT_SET_SH_REG_INDIRECT:
			case IT_SET_UCONFIG_REG_INDIRECT: {
				if (body_count < 4) {
					return fail(op, dw);
				}
				const uint64_t addr = (static_cast<uint64_t>(bptr[0]) & 0xfffffffcull) |
				                      (static_cast<uint64_t>(bptr[1]) << 32u);
				uint32_t num = bptr[3] & 0x3fffu;
				if (op == IT_SET_CONTEXT_REG_INDIRECT) {
					stats.load_context++;
				} else if (op == IT_SET_SH_REG_INDIRECT) {
					stats.load_sh++;
				} else {
					stats.load_uconfig++;
				}
				if (num == 0 || addr == 0) {
					break;
				}
				const auto* pairs = reinterpret_cast<const uint32_t*>(addr);
				for (uint32_t i = 0; i < num; i++) {
					const uint32_t raw   = pairs[2 * i];
					const uint32_t value = pairs[2 * i + 1];
					if (raw == kSentinelOffset) {
						continue;
					}
					const uint32_t off = Normalize(raw);
					if (op == IT_SET_CONTEXT_REG_INDIRECT) {
						cx[off] = value;
						mark_cx(off);
						EmitSet(body, IT_SET_CONTEXT_REG, off, value, false);
						stats.loaded_cx_regs++;
					} else if (op == IT_SET_SH_REG_INDIRECT) {
						if (off == kShNop) {
							continue;
						}
						sh[off] = value;
						dirty_sh.push_back(off);
						note_shader(off);
						EmitSet(body, IT_SET_SH_REG, off, value, off >= kShComputeBase);
						stats.loaded_sh_regs++;
					} else {
						if (off == kUcNop) {
							continue;
						}
						if (!UconfigAllowed(off)) {
							stats.uconfig_skipped++;
							r.skipped_uconfig.insert(off);
							continue;
						}
						uc[off] = value;
						EmitSetUconfig(body, off, value);
						stats.loaded_uc_regs++;
					}
				}
				break;
			}

			case IT_WAIT_REG_MEM:
			case IT_WAIT_REG_MEM_64: {
				// Shadow mode drops guest waits: native IBs run one at a time and to completion before
				// Kyty records the same DCB, and Kyty's Vulkan path does the real synchronization. Passing
				// a wait through would hang the gfx ring on a label that Kyty only writes later. The
				// condition is still evaluated against guest memory now, so the stats show which waits a
				// primary (non-shadow) native path will have to honour.
				const bool     wide     = op == IT_WAIT_REG_MEM_64;
				const uint32_t need     = wide ? 8u : 6u;
				if (body_count < need || (bptr[0] & 0x10u) == 0) {
					return fail(op, dw); // register waits are refused
				}
				const uint64_t addr = (static_cast<uint64_t>(bptr[1]) & (wide ? ~0x7ull : ~0x3ull)) |
				                      (static_cast<uint64_t>(bptr[2] & 0x3ffffu) << 32u);
				uint64_t ref  = bptr[3];
				uint64_t mask = wide ? bptr[5] : bptr[4];
				if (wide) {
					ref |= static_cast<uint64_t>(bptr[4]) << 32u;
					mask |= static_cast<uint64_t>(bptr[6]) << 32u;
				}
				bool met = false;
				if (addr != 0) {
					const uint64_t value = wide ? *reinterpret_cast<const volatile uint64_t*>(addr)
					                            : *reinterpret_cast<const volatile uint32_t*>(addr);
					met = WaitConditionMet(bptr[0] & 0x7u, value & mask, ref);
				}
				stats.waits_dropped++;
				if (!met) {
					stats.waits_unmet++;
				}
				break;
			}
			case IT_DMA_DATA:
				if (body_count < 6 || !DmaDataAllowed(bptr[0], bptr[5], m_gds)) {
					r.unknown_detail = {bptr[0], body_count >= 6 ? bptr[5] : 0u};
					return fail(op, dw);
				}
				body.insert(body.end(), dcb.begin() + dw, dcb.begin() + dw + size);
				stats.dma_data++;
				{
					const uint64_t bytes = bptr[5] & 0x1fffffu;
					const uint32_t src_sel = (bptr[0] >> 29u) & 0x3u; // 1 GDS, 2 immediate data
					const uint32_t dst_sel = (bptr[0] >> 20u) & 0x3u; // 1 GDS
					if (src_sel != 1u && src_sel != 2u) {
						add_ref(static_cast<uint64_t>(bptr[1]) | (static_cast<uint64_t>(bptr[2]) << 32u), bytes, false);
					}
					if (dst_sel != 1u) {
						add_ref(static_cast<uint64_t>(bptr[3]) | (static_cast<uint64_t>(bptr[4]) << 32u), bytes, false);
					}
				}
				break;
			case IT_WRITE_DATA:
				// Games write their completion labels with WRITE_DATA. The native IB completes before
				// Kyty has even parsed this DCB, so a native label tells the game the GPU is done while
				// Kyty is still reading the command buffer, and the game starts rewriting it (Dead
				// Cells: a truncated packet in Kyty's parser on entering gameplay). Kyty performs the
				// write itself, in order, so the native IB leaves it out.
				if (body_count < 3 || !WriteDataToMemory(bptr[0])) {
					return fail(op, dw);
				}
				stats.write_data_dropped++;
				break;
			case IT_DRAW_INDEX_2:
				body.insert(body.end(), dcb.begin() + dw, dcb.begin() + dw + size);
				stats.draws++;
				collect_refs();
				if (body_count >= 3) { // MAX_SIZE (indices), INDEX_BASE_LO, INDEX_BASE_HI, ...
					add_ref(static_cast<uint64_t>(bptr[1]) | (static_cast<uint64_t>(bptr[2]) << 32u),
					        static_cast<uint64_t>(bptr[0]) * 4u, false);
				}
				break;
			case IT_INDEX_BASE: // the index buffer for draws that do not carry their own
				if (body_count < 2) {
					return fail(op, dw);
				}
				body.insert(body.end(), dcb.begin() + dw, dcb.begin() + dw + size);
				index_base = static_cast<uint64_t>(bptr[0]) | (static_cast<uint64_t>(bptr[1] & 0xffffu) << 32u);
				add_ref(index_base, index_bytes != 0 ? index_bytes : kTextureSpan, false);
				stats.index_state++;
				break;
			case IT_INDEX_BUFFER_SIZE:
				body.insert(body.end(), dcb.begin() + dw, dcb.begin() + dw + size);
				index_bytes = static_cast<uint64_t>(bptr[0]) * 4u;
				add_ref(index_base, index_bytes, false);
				stats.index_state++;
				break;
			case IT_DRAW_INDEX_OFFSET_2: // indexed draw into the INDEX_BASE buffer
				body.insert(body.end(), dcb.begin() + dw, dcb.begin() + dw + size);
				stats.draws++;
				collect_refs();
				add_ref(index_base, index_bytes != 0 ? index_bytes : kTextureSpan, false);
				break;
			case IT_DRAW_INDEX_AUTO: // a non-indexed draw: no index buffer to reference
				body.insert(body.end(), dcb.begin() + dw, dcb.begin() + dw + size);
				stats.draws++;
				collect_refs();
				break;
			case IT_SET_BASE: // BASE_INDEX, address: where indirect draw/dispatch arguments live
				if (body_count < 3) {
					return fail(op, dw);
				}
				body.insert(body.end(), dcb.begin() + dw, dcb.begin() + dw + size);
				add_ref(static_cast<uint64_t>(bptr[1]) | (static_cast<uint64_t>(bptr[2] & 0xffffu) << 32u),
				        kMetadataSpan, false);
				stats.set_base++;
				break;
			case IT_DISPATCH_INDIRECT: // arguments at the SET_BASE address + DATA_OFFSET
				body.insert(body.end(), dcb.begin() + dw, dcb.begin() + dw + size);
				stats.dispatches++;
				collect_refs();
				break;
			case IT_PFP_SYNC_ME:
				body.insert(body.end(), dcb.begin() + dw, dcb.begin() + dw + size);
				stats.pfp_sync_me++;
				break;
			case IT_NUM_INSTANCES: // instance count for the following draws
				body.insert(body.end(), dcb.begin() + dw, dcb.begin() + dw + size);
				stats.num_instances++;
				break;
			case IT_DISPATCH_DIRECT:
				body.insert(body.end(), dcb.begin() + dw, dcb.begin() + dw + size);
				stats.dispatches++;
				collect_refs();
				break;
			case IT_ACQUIRE_MEM:
				body.insert(body.end(), dcb.begin() + dw, dcb.begin() + dw + size);
				stats.acquire_mem++;
				break;
			case IT_EVENT_WRITE:
				body.insert(body.end(), dcb.begin() + dw, dcb.begin() + dw + size);
				stats.event_write++;
				break;
			case IT_CLEAR_STATE:
				body.insert(body.end(), dcb.begin() + dw, dcb.begin() + dw + size);
				stats.clear_state++;
				break;

			default:
				// Keep walking so one refused DCB names every packet the translator lacks.
				if (r.unknown_ops.empty()) {
					r.unknown_op = op;
					r.unknown_dw = dw;
				}
				r.unknown_ops.insert(op);
				break;
		}

		dw += size;
	}
	if (!r.unknown_ops.empty()) {
		const uint32_t first_op = r.unknown_op;
		const uint32_t first_dw = r.unknown_dw;
		auto           unknown  = std::move(r.unknown_ops);
		fail(first_op, first_dw);
		r.unknown_ops = std::move(unknown);
		return r;
	}

	// Pass succeeded: commit the shadow, build the IB (preamble from the committed shadow, then the
	// translated body), and return.
	m_cx = std::move(cx);
	m_sh = std::move(sh);
	m_uc = std::move(uc);

	r.ib.reserve(body.size() + (m_cx.size() + m_sh.size() + m_uc.size()) * 3 + 8);
	EmitPreamble(r.ib, stats);
	r.ib.insert(r.ib.end(), body.begin(), body.end());

	// The CP fetches IBs in 32-byte (8-dword) units; pad with filler NOPs.
	while ((r.ib.size() & 7u) != 0) {
		r.ib.push_back(0xffff1000u);
	}

	r.stats        = stats;
	r.shader_addrs = std::move(shaders);
	std::sort(refs.begin(), refs.end());
	refs.erase(std::unique(refs.begin(), refs.end()), refs.end());
	r.refs = std::move(refs);
	return r;
}

} // namespace Libs::Graphics::NativeGpu
