#include "graphics/native_gpu/nativePm4.h"

namespace Libs::Graphics::NativeGpu {

namespace {

// Type-3 PM4 opcodes, mirroring src/graphics/guest_gpu/pm4.h (kept local so the translator has no
// build dependency on the rest of Kyty and stays unit-testable off the board).
constexpr uint32_t IT_NOP                      = 0x10;
constexpr uint32_t IT_CLEAR_STATE              = 0x12;
constexpr uint32_t IT_DISPATCH_DIRECT          = 0x15;
constexpr uint32_t IT_DRAW_INDEX_2             = 0x27;
constexpr uint32_t IT_CONTEXT_CONTROL          = 0x28;
constexpr uint32_t IT_WAIT_REG_MEM             = 0x3C;
constexpr uint32_t IT_WAIT_REG_MEM_64          = 0x93;
constexpr uint32_t IT_WRITE_DATA               = 0x37;
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
		case 0x25B: // GE_CNTL
		case 0x262: // GE_USER_VGPR_EN
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
						EmitSet(body, IT_SET_CONTEXT_REG, off, value, false);
						stats.loaded_cx_regs++;
					} else if (op == IT_SET_SH_REG_INDIRECT) {
						if (off == kShNop) {
							continue;
						}
						sh[off] = value;
						note_shader(off);
						EmitSet(body, IT_SET_SH_REG, off, value, off >= kShComputeBase);
						stats.loaded_sh_regs++;
					} else {
						if (off == kUcNop) {
							continue;
						}
						if (!UconfigAllowed(off)) {
							stats.uconfig_skipped++;
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
			case IT_WRITE_DATA:
				if (body_count < 3 || !WriteDataToMemory(bptr[0])) {
					return fail(op, dw);
				}
				body.insert(body.end(), dcb.begin() + dw, dcb.begin() + dw + size);
				stats.write_data++;
				break;
			case IT_DRAW_INDEX_2:
				body.insert(body.end(), dcb.begin() + dw, dcb.begin() + dw + size);
				stats.draws++;
				break;
			case IT_DISPATCH_DIRECT:
				body.insert(body.end(), dcb.begin() + dw, dcb.begin() + dw + size);
				stats.dispatches++;
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
				return fail(op, dw);
		}

		dw += size;
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
	return r;
}

} // namespace Libs::Graphics::NativeGpu
