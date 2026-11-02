#include "graphics/native_gpu/nativePm4.h"

#include <algorithm>
#include <cstring>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <tuple>
#include <type_traits>
#if defined(__linux__)
#include <sys/uio.h>
#include <unistd.h>
#endif

namespace Libs::Graphics::NativeGpu {

namespace {

// Type-3 PM4 opcodes, mirroring src/graphics/guest_gpu/pm4.h (kept local so the translator has no
// build dependency on the rest of Kyty and stays unit-testable off the board).
constexpr uint32_t IT_NOP                      = 0x10;
constexpr uint32_t IT_INDIRECT_BUFFER          = 0x3F;
constexpr uint32_t IT_COPY_DATA                = 0x40;
constexpr uint32_t IT_CLEAR_STATE              = 0x12;
constexpr uint32_t IT_ATOMIC_MEM               = 0x1E;
constexpr uint32_t IT_COND_EXEC                = 0x22;
constexpr uint32_t IT_SET_BASE                 = 0x11;
constexpr uint32_t IT_SET_PREDICATION          = 0x20;
constexpr uint32_t IT_INDEX_BUFFER_SIZE        = 0x13;
constexpr uint32_t IT_DISPATCH_DIRECT          = 0x15;
constexpr uint32_t IT_DISPATCH_INDIRECT        = 0x16;
constexpr uint32_t IT_DRAW_INDIRECT            = 0x24;
constexpr uint32_t IT_RELEASE_MEM              = 0x49;
constexpr uint32_t IT_DRAW_INDEX_INDIRECT      = 0x25;
constexpr uint32_t IT_DRAW_INDIRECT_MULTI      = 0x2C;
constexpr uint32_t IT_DRAW_INDEX_INDIRECT_MULTI = 0x38;
constexpr uint32_t IT_GET_LOD_STATS            = 0x8E;
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
constexpr uint32_t IT_COND_WRITE               = 0x45;
constexpr uint32_t IT_MEM_SEMAPHORE            = 0x39;
constexpr uint32_t IT_DRAW_INDEX_MULTI_INST    = 0x3A; // Sony's (AgcDcbDrawIndexMultiInstanced)
constexpr uint32_t IT_EVENT_WRITE              = 0x46;
constexpr uint32_t IT_ACQUIRE_MEM             = 0x58;
constexpr uint32_t IT_SET_SH_REG_INDIRECT      = 0x63;
constexpr uint32_t IT_SET_UCONFIG_REG_INDIRECT = 0x64;
constexpr uint32_t IT_SET_CONTEXT_REG          = 0x69;
constexpr uint32_t IT_SET_SH_REG               = 0x76;
constexpr uint32_t IT_SET_UCONFIG_REG          = 0x79;
constexpr uint32_t IT_SET_UCONFIG_REG_INDEX    = 0x7A;
constexpr uint32_t IT_SET_CONTEXT_REG_INDIRECT = 0x9F;
// CP register shadowing (SetCpShadow): the registers of each range back from the shadow buffer, which also
// sets the class's shadow base (only with that class's load enable on).
constexpr uint32_t IT_LOAD_UCONFIG_REG         = 0x5E;
constexpr uint32_t IT_LOAD_SH_REG              = 0x5F;
constexpr uint32_t IT_LOAD_CONTEXT_REG         = 0x61;

// IT_NOP marker codes carried in the register field (Kyty custom codes in pm4.h).
constexpr uint32_t R_ZERO           = 0x00;
constexpr uint32_t R_WAIT_FLIP_DONE = 0x06;
constexpr uint32_t R_FLIP           = 0x17;
constexpr uint32_t R_RELEASE_MEM    = 0x18;
constexpr uint32_t R_CONTEXT_STATE  = 0x1A; // Kyty's marker for Agc context-state push/pop/clear
constexpr uint32_t R_DISPATCH_RESET = 0x09;
constexpr uint32_t R_PUSH_MARKER    = 0x0B;
constexpr uint32_t R_POP_MARKER     = 0x0C;

// kFlipPadMagic and kWaitFlipPadMagic (the padding NOPs of libSceAgc's SetFlip and WaitUntilSafeForRendering) are in
// nativePm4.h.

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
constexpr uint32_t kCbColor0Info     = 0x31C;
constexpr uint32_t kCbDccEnable      = 1u << 28u; // CB_COLORn_INFO.DCC_ENABLE
constexpr uint32_t kCbColor0Attrib   = 0x31D;
constexpr uint32_t kCbColor0Attrib3  = 0x3B8;
constexpr uint32_t kCbColor0Cmask    = 0x31F;
constexpr uint32_t kCbColor0Fmask    = 0x321;
constexpr uint32_t kCbColor0DccBase  = 0x325;
constexpr uint32_t kCbColor0BaseExt  = 0x390;
constexpr uint32_t kCbColor0CmaskExt = 0x398;
constexpr uint32_t kCbColor0FmaskExt = 0x3A0;
constexpr uint32_t kCbColor0DccExt   = 0x3A8;
constexpr uint32_t kCbColor0Attrib2  = 0x3B0;
constexpr uint32_t kCbTargetMask     = 0x08E;
constexpr uint32_t kDbDepthView      = 0x002;
constexpr uint32_t kDbRenderOverride = 0x003;
constexpr uint32_t kPaScShaderControl = 0x310;
constexpr uint32_t kDbZInfo          = 0x010;
constexpr uint32_t kDbStencilInfo    = 0x011;
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
std::atomic<uint64_t> g_tf_ring_base {0};
std::atomic<uint32_t> g_tf_ring_bytes {0};
std::atomic<uint32_t> g_hs_offchip_param {0};
// SetCpShadowValidation: the shadow buffer LOAD_*_REG may name (0: none).
std::atomic<uint64_t> g_cp_shadow_va {0};

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

// Native labels emit RELEASE_MEM and WAIT_REG_MEM only in forms ValidateNativeIb knows (an ACB's guest
// packets, with the compute shader-type bit, stopped the gfx ring with "Illegal opcode in command
// stream"). The barrier's form: RELEASE_MEM with CACHE_FLUSH_AND_INV_TS_EVENT, EVENT_INDEX 5 (end of
// pipe), GCR_CNTL GLM_WB | GLM_INV | GLV_INV | GL1_INV | GL2_INV | GL2_WB, 32-bit data to memory and no
// interrupt; labels in the guest's event take LabelEventCntl's form instead. WAIT_REG_MEM is EQUAL on
// memory by the micro engine, poll interval 4.
constexpr uint32_t kFlushEventCntl = 0x14u | (5u << 8u) | (1u << 12u) | (1u << 13u) | (1u << 14u) | (1u << 15u) |
                                     (1u << 20u) | (1u << 21u);
constexpr uint32_t kData32Cntl     = 1u << 29u;
constexpr uint32_t kIntSel2        = 2u << 24u; // RELEASE_MEM: interrupt once the data write is confirmed
// DATA_SEL 3: the 64-bit GPU clock counter. amdgpu's gfx_v10_0_ring_emit_fence stores a
// sequence with DATA_SEL 1 or 2 in this same packet (CACHE_FLUSH_AND_INV_TS_EVENT, EVENT_INDEX
// 5); Mesa's write-timestamp uses DATA_SEL 3 and ignores the data dwords. The event dword stays
// the one this board already runs. The data dwords are zero: the clock replaces them.
constexpr uint32_t kDataTimestampCntl = 3u << 29u;
constexpr uint32_t kWaitEqualMemMe = 3u | (1u << 4u);
constexpr uint32_t kWaitPoll       = 4u;
constexpr uint32_t kFiller         = 0xffff1000u; // the one-dword NOP IBs are padded with

// A guest label release in the guest's own event (end-of-pipe events the PS5 driver uses: the CB/DB
// flush-and-invalidate timestamps, bottom of pipe, CS and PS done; EVENT_INDEX 5 or 6) with the guest's
// cache actions, plus GLM and GL2 write-back so the CPU sees the label in memory. ASTRO BOT's gfx ring
// asks no cache action at all for ~95% of its labels; the barrier's form invalidated every cache at
// each of them, about a thousand times a frame. Anything else, and KYTY_NATIVE_GPU_LABEL_FLUSH=full,
// gets the barrier's form.
constexpr uint32_t kGcrGlmWb        = 1u << 12u;
constexpr uint32_t kGcrGlmInv       = 1u << 13u;
constexpr uint32_t kGcrGl2Wb        = 1u << 21u;
constexpr uint32_t kReleaseGcrMask  = 0x1fffu << 12u; // GCR_CNTL [24:12]
constexpr uint32_t kReleasePolicy   = 0x3u << 25u;    // CACHE_POLICY
bool LabelEventOk(uint32_t event) {
	const uint32_t type  = event & 0x3fu;
	const uint32_t index = (event >> 8u) & 0xfu;
	// 0x04 CACHE_FLUSH_TS: libSceAgc's SetFlip release.
	const bool known = type == 0x04u || type == 0x14u || type == 0x28u || type == 0x2bu || type == 0x2du || type == 0x2fu || type == 0x30u;
	return known && (index == 5u || index == 6u) && (event & ~(0x3fu | 0xf00u | kReleaseGcrMask | kReleasePolicy)) == 0u;
}
// The label itself goes to memory past the caches (DST_SEL 0) either way, and GPU waits read it
// through GL2; the added write-backs make the GPU's other results visible to the CPU by the time it
// sees the label. Without them Gran Turismo 7 froze for up to a second at a time: the title read the
// label and waited on data still in GL2. They came at every label, though (GT7: ~150 a frame, each
// followed by a wait of the same DCB that waited on the write-back too, ~1 ms of GPU a frame), and a
// label the same buffer waits on is a GPU sync: its write-back is taken off again once that wait is
// seen (StripLabelWriteBack), and the data reaches memory with the next label that keeps one or the
// segment's tracker store. KYTY_NATIVE_GPU_LABEL_WB: 1 every label (as before), 0 none (experiment:
// freezes), unset: all but the GPU-waited ones.
int LabelWriteBackMode() {
	static const int mode = [] {
		const char* v = std::getenv("KYTY_NATIVE_GPU_LABEL_WB");
		return v == nullptr || v[0] == '\0' ? 2 : (v[0] == '0' ? 0 : 1);
	}();
	return mode;
}
bool LabelWriteBack() { return LabelWriteBackMode() != 0; }
// The write-back bits LabelEventCntl added to `guest_event` that a GPU-waited label can drop: GL2_WB,
// and GLM_WB unless the guest's own GL2_WB or GLM_INV still needs it (the metadata cache is written
// back whenever GL2 is, or dropped).
uint32_t StrippableWriteBack(uint32_t guest_event, uint32_t event) {
	if (LabelWriteBackMode() != 2 || event == kFlushEventCntl) {
		return 0u;
	}
	uint32_t strip = event & ~guest_event & kGcrGl2Wb;
	if ((event & ~guest_event & kGcrGlmWb) != 0u && (guest_event & (kGcrGl2Wb | kGcrGlmInv)) == 0u) {
		strip |= kGcrGlmWb;
	}
	return strip;
}
// Releases as the title wrote them (1.11): whether the CPU sees a label while the GPU job still runs is up to the
// release's CACHE_POLICY, not its DST_SEL. With STREAM, NOA or BYPASS the label written to L2 (DST_SEL 1, as
// libSceAgc writes them) reaches the CPU within ~30 us, 32- or 64-bit, on the gfx ring and on compute; with LRU it
// stays in GL2 until a write-back (probe labelvis with the job kept running: LABELVIS_GO_UC). The PS5 is no
// different: the labels its CPU reads carry BYPASS, the LRU ones are GPU-to-GPU syncs, which see each other in
// GL2. The old belief that a DST_SEL 1 label never reaches the CPU came from probes whose job ended right after
// the label (the end-of-job fence writes GL2 back) and from runs mixing LRU labels with Kyty's CPU-side reads.
// KYTY_NATIVE_GPU_RELEASES_AS_WRITTEN=0 goes back to rewriting labels (DST_SEL 0, write-backs added, 64-bit
// data as two halves).
bool ReleasesAsWritten() {
	static const bool on = [] {
		const char* v = std::getenv("KYTY_NATIVE_GPU_RELEASES_AS_WRITTEN");
		return v == nullptr || v[0] != '0';
	}();
	return on;
}
// KYTY_NATIVE_GPU_RELEASE_GLM_WB=1: a release with GL2_WB or GLM_INV but no GLM_WB gets GLM_WB added. On 27.10 such
// releases as written gave black frames, but those runs also had LRU labels the CPU did not see; as written with
// labels as written (1.11) ASTRO BOT draws right, so they go as the title wrote them.
bool ReleaseGlmWriteBack() {
	static const bool on = [] {
		const char* v = std::getenv("KYTY_NATIVE_GPU_RELEASE_GLM_WB");
		return v != nullptr && v[0] == '1';
	}();
	return on;
}
bool AcbReleasesAsWritten() {
	static const bool on = [] {
		const char* v = std::getenv("KYTY_NATIVE_GPU_ACB_RELEASES_AS_WRITTEN");
		return v == nullptr || v[0] != '0';
	}();
	return on;
}
// KYTY_NATIVE_GPU_LABELS_VERBATIM: which label releases go as the title wrote them, by DATA_SEL: 1
// 32-bit data, 2 64-bit, 4 the GPU clock, 8 none (cache actions alone); 0 rewrites every one (data to
// memory, write-backs added, 64-bit data as two halves). All by default; with the old rewriting
// (KYTY_NATIVE_GPU_RELEASES_AS_WRITTEN=0) 64-bit labels are left out, as before.
bool LabelsVerbatim(uint32_t data_sel) {
	static const uint32_t mask = [] {
		const char* v = std::getenv("KYTY_NATIVE_GPU_LABELS_VERBATIM");
		return v != nullptr ? static_cast<uint32_t>(std::strtoul(v, nullptr, 0)) : (ReleasesAsWritten() ? 0xfu : 0xdu);
	}();
	return data_sel <= 3 && (mask & (data_sel == 0 ? 8u : 1u << (data_sel - 1u))) != 0;
}
// The tracker's own stores (segment ends, points before a GPU wait): bottom of pipe, GLM and GL2
// written back so the CPU sees the GPU's results by the time the tracker acts, nothing invalidated.
// The barrier's form invalidated every cache at each of them, hundreds of times a frame.
constexpr uint32_t kTrackerWbEventCntl = 0x28u | (5u << 8u) | kGcrGlmWb | kGcrGl2Wb;
bool FullLabelFlush() {
	static const bool full = [] {
		const char* v = std::getenv("KYTY_NATIVE_GPU_LABEL_FLUSH");
		return v != nullptr && std::strcmp(v, "full") == 0;
	}();
	return full;
}
// KYTY_NATIVE_GPU_LABEL_FLUSH=full: the barrier's form here too.
uint32_t TrackerEventCntl() { return FullLabelFlush() ? kFlushEventCntl : kTrackerWbEventCntl; }

// A RELEASE_MEM of CS_DONE or PS_DONE (EVENT_INDEX 6, end of shader) that writes no data timed out on the BC-250's gfx
// ring by itself (BC5, experiment 0016 run 40); PAL sends these events only with data, and on MEC (ASTRO BOT's ACBs,
// 140k a run) they complete without. On the PS5 the CP completes them either way. KYTY_NATIVE_GPU_EOS_NODATA picks the
// gfx form: `data` (the default) keeps the title's event, cache actions and policy and adds a 32-bit store of 0 into
// Kyty's scratch dword, PAL's form, which changes nothing the title sees; `bop` makes it BOTTOM_OF_PIPE_TS (EVENT_INDEX
// 5) with the same cache actions, as BC5 does; `asis` sends it as written.
enum class EosNoData { Data, Bop, AsIs };
EosNoData EosNoDataMode() {
	static const EosNoData mode = [] {
		const char* v = std::getenv("KYTY_NATIVE_GPU_EOS_NODATA");
		if (v != nullptr && std::strcmp(v, "bop") == 0) {
			return EosNoData::Bop;
		}
		if (v != nullptr && std::strcmp(v, "asis") == 0) {
			return EosNoData::AsIs;
		}
		return EosNoData::Data;
	}();
	return mode;
}
bool EndOfShaderEvent(uint32_t event) {
	const uint32_t type = event & 0x3fu;
	return (type == 0x2fu || type == 0x30u) && ((event >> 8u) & 0xfu) == 6u;
}

uint32_t LabelEventCntl(uint32_t guest_event) {
	const uint32_t event =
	    (guest_event & (0x3fu | 0xf00u | kReleaseGcrMask | kReleasePolicy)) | (LabelWriteBack() ? kGcrGlmWb | kGcrGl2Wb : 0u);
	return !FullLabelFlush() && LabelEventOk(event) ? event : kFlushEventCntl;
}

// KYTY_NATIVE_GPU_RELEASE_FORMS=1: each distinct guest RELEASE_MEM form (event dword, the select
// dword without its data bits, queue kind) once, with how often it came, every 4096 releases.
void NoteReleaseForm(uint32_t event, uint32_t control, bool compute) {
	static const bool on = [] {
		const char* v = std::getenv("KYTY_NATIVE_GPU_RELEASE_FORMS");
		return v != nullptr && v[0] == '1';
	}();
	if (!on) {
		return;
	}
	static std::mutex                             mutex;
	static std::map<std::tuple<uint32_t, uint32_t, bool>, uint64_t> forms;
	static uint64_t                               total = 0;
	std::lock_guard<std::mutex>                   lock(mutex);
	forms[{event, control & ~(0x7u << 29u), compute}]++;
	if ((++total & 4095u) == 0) {
		for (const auto& [k, n] : forms) {
			std::fprintf(stderr, "release form: %s event 0x%08x control 0x%08x: %llu\n", std::get<2>(k) ? "acb" : "dcb", std::get<0>(k),
			             std::get<1>(k), static_cast<unsigned long long>(n));
		}
	}
}

// KYTY_NATIVE_GPU_AUDIT (SetAudit): a sync packet's fields as the title wrote them. `odd`: its bits outside the fields
// the gfx10 layout names.
void AuditCommon(Pm4SyncAudit& s, AuditSync form, uint32_t policy, uint32_t engine, uint32_t odd) {
	const auto f = static_cast<size_t>(form);
	s.packets[f]++;
	s.policy[f][policy & 3u]++;
	s.engine[f][engine & 3u]++;
	s.odd[f] += odd != 0u ? 1u : 0u;
	s.odd_bits[f] |= odd;
}
// WAIT_REG_MEM(_64) dword 1: FUNCTION [2:0], MEM_SPACE [5:4], OPERATION [7:6], ENGINE_SEL [9:8], CACHE_POLICY [26:25].
void AuditWait(Pm4SyncAudit& s, bool wide, uint32_t control) {
	const int k = wide ? 1 : 0;
	AuditCommon(s, wide ? AuditSync::Wait64 : AuditSync::Wait, control >> 25u, control >> 8u, control & ~0x060003f7u);
	s.wait_function[k][control & 7u]++;
	s.wait_mem_space[k][(control >> 4u) & 3u]++;
	s.wait_operation[k][(control >> 6u) & 3u]++;
}
// RELEASE_MEM dword 1: EVENT_TYPE [5:0], EVENT_INDEX [11:8], GCR_CNTL [24:12], CACHE_POLICY [26:25], EXECUTE [28]; dword 2:
// DST_SEL [17:16], INT_SEL [26:24], DATA_SEL [31:29].
void AuditRelease(Pm4SyncAudit& s, bool marker, uint32_t event, uint32_t select) {
	const int       k       = marker ? 1 : 0;
	const AuditSync form    = marker ? AuditSync::ReleaseMarker : AuditSync::Release;
	const uint32_t  ev_odd  = event & ~0x17ffff3fu;
	const uint32_t  sel_odd = select & ~0xe7030000u;
	AuditCommon(s, form, event >> 25u, 0u, ev_odd);
	s.odd[static_cast<size_t>(form)] += ev_odd == 0u && sel_odd != 0u ? 1u : 0u; // odd in either dword
	s.release_select_odd[k] |= sel_odd;
	s.release_event[k][event & 0x3fu]++;
	for (uint32_t g = (event >> 12u) & 0x1fffu; g != 0; g &= g - 1u) {
		s.release_gcr[k][std::countr_zero(g)]++;
	}
	s.release_dst_sel[k][(select >> 16u) & 3u]++;
	s.release_int_sel[k][(select >> 24u) & 7u]++;
	s.release_data_sel[k][select >> 29u]++;
}
// WRITE_DATA dword 1: DST_SEL [11:8], WR_ONE_ADDR [16], WR_CONFIRM [20], CACHE_POLICY [26:25], ENGINE_SEL [31:30].
// libSceAgc's debug-marker strings are only counted, or they would swamp DST_SEL 0.
void AuditWriteData(Pm4SyncAudit& s, uint32_t control, bool marker) {
	if (marker) {
		s.write_markers++;
		return;
	}
	AuditCommon(s, AuditSync::WriteData, control >> 25u, control >> 30u, control & ~0xc6110f00u);
	s.write_dst_sel[(control >> 8u) & 0xfu]++;
	s.write_confirm += (control >> 20u) & 1u;
}
// ACQUIRE_MEM: ENGINE in dword 1 bit 31 (with COHER_CNTL), GCR_CNTL [18:0] in body dword 6.
void AuditAcquire(Pm4SyncAudit& s, std::span<const uint32_t> body) {
	const uint32_t control = !body.empty() ? body[0] : 0u;
	const uint32_t gcr     = body.size() >= 7 ? body[6] : 0u;
	AuditCommon(s, AuditSync::Acquire, 0u, control >> 31u, gcr & ~0x7ffffu);
	for (uint32_t g = gcr & 0x7ffffu; g != 0; g &= g - 1u) {
		s.acquire_gcr[std::countr_zero(g)]++;
	}
}

void EmitRelease32(std::vector<uint32_t>& ib, uint64_t addr, uint32_t value, bool compute, uint32_t event = kFlushEventCntl) {
	ib.push_back(Pkt3(IT_RELEASE_MEM, 7, compute));
	ib.push_back(event);
	ib.push_back(kData32Cntl);
	ib.push_back(static_cast<uint32_t>(addr));
	ib.push_back(static_cast<uint32_t>(addr >> 32u));
	ib.push_back(value);
	ib.push_back(0u);
	ib.push_back(0u);
}

// End-of-pipe GPU clock at `addr` (8-byte aligned). Same event as EmitRelease32, DATA_SEL 3.
void EmitTimestampRelease(std::vector<uint32_t>& ib, uint64_t addr, bool compute) {
	ib.push_back(Pkt3(IT_RELEASE_MEM, 7, compute));
	ib.push_back(kFlushEventCntl);
	ib.push_back(kDataTimestampCntl);
	ib.push_back(static_cast<uint32_t>(addr));
	ib.push_back(static_cast<uint32_t>(addr >> 32u));
	ib.push_back(0u);
	ib.push_back(0u);
	ib.push_back(0u);
}

// GPU timing (SetGpuTimes): the GPU clock when the micro engine gets here (COPY_DATA SRC_SEL 9, 64 bit,
// write confirmed), and when everything before has finished (RELEASE_MEM BOTTOM_OF_PIPE_TS, DATA_SEL 3,
// no cache actions). Both only on a busy GPU: an idle gfx ring stopped on a clock read (probe clock).
constexpr uint32_t kClockCopyCntl = 9u | (5u << 8u) | (1u << 16u) | (1u << 20u);
constexpr uint32_t kClockEopCntl  = 0x28u | (5u << 8u);
void EmitClockCopy(std::vector<uint32_t>& ib, uint64_t addr) {
	ib.push_back(Pkt3(IT_COPY_DATA, 5, false));
	ib.push_back(kClockCopyCntl);
	ib.push_back(0u);
	ib.push_back(0u);
	ib.push_back(static_cast<uint32_t>(addr));
	ib.push_back(static_cast<uint32_t>(addr >> 32u));
}
void EmitClockEop(std::vector<uint32_t>& ib, uint64_t addr) {
	ib.push_back(Pkt3(IT_RELEASE_MEM, 7, false));
	ib.push_back(kClockEopCntl);
	ib.push_back(kDataTimestampCntl);
	ib.push_back(static_cast<uint32_t>(addr));
	ib.push_back(static_cast<uint32_t>(addr >> 32u));
	ib.push_back(0u);
	ib.push_back(0u);
	ib.push_back(0u);
}

void EmitWaitEqual(std::vector<uint32_t>& ib, uint64_t addr, uint32_t ref, uint32_t mask, bool compute) {
	ib.push_back(Pkt3(IT_WAIT_REG_MEM, 6, compute));
	ib.push_back(kWaitEqualMemMe);
	ib.push_back(static_cast<uint32_t>(addr));
	ib.push_back(static_cast<uint32_t>(addr >> 32u));
	ib.push_back(ref);
	ib.push_back(mask);
	ib.push_back(kWaitPoll);
}

// A guest wait's CONTROL the gfx CP runs as the PS5's (probe waitforms): on memory, wait only (OPERATION
// 0), the micro engine or the prefetch parser, any compare function but the reserved 7.
// libSceAgc's debug markers write the rest of their string with WRITE_DATA to SQ_THREAD_TRACE_USERDATA_3
// (register 0xc343, DST_SEL 0, micro engine; the thread-trace user data registers 0xc340..0xc347).
bool TraceUserDataWrite(uint32_t control, uint32_t addr_lo, uint32_t addr_hi) {
	return ((control >> 8u) & 0xfu) == 0u && (control >> 30u) == 0u && addr_hi == 0u && addr_lo >= 0xc340u && addr_lo <= 0xc347u;
}

bool GuestWaitControlOk(uint32_t control) {
	return (control & 0x7u) != 7u && ((control >> 4u) & 0x3u) == 1u && ((control >> 6u) & 0x3u) == 0u &&
	       ((control >> 8u) & 0x3u) <= 1u;
}

} // namespace

const char* RefuseWhyName(RefuseWhy why) {
	static const char* const kNames[static_cast<size_t>(RefuseWhy::Count)] = {
	    "collected",
	    "not a type-3 header",
	    "past the end",
	    "unknown opcode",
	    "graphics opcode on MEC",
	    "geometry UCONFIG on MEC",
	    "body too short",
	    "RELEASE_MEM DATA_SEL",
	    "GDS release without a partition",
	    "register wait",
	    "DMA_DATA form",
	    "WRITE_DATA to a register",
	    "CONTEXT_CONTROL on MEC",
	    "COND_EXEC form",
	    "ATOMIC_MEM form",
	    "COPY_DATA form",
	    "MEM_SEMAPHORE form",
	    "run in place with ours around it",
	    "packet check",
	    "packet check of the resume",
	    "GDS shader",
	    "GDS shader skip learning",
	};
	return why < RefuseWhy::Count ? kNames[static_cast<size_t>(why)] : "?";
}

namespace {
template <typename T>
bool CpuAtomic(T* p, uint32_t op, T src, T cmp, bool loop) {
	using S = std::make_signed_t<T>;
	auto rmw = [&](auto next) {
		T old = __atomic_load_n(p, __ATOMIC_SEQ_CST);
		while (!__atomic_compare_exchange_n(p, &old, next(old), false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST)) {
		}
		return true;
	};
	switch (op) {
		case 0x07u: __atomic_store_n(p, src, __ATOMIC_SEQ_CST); return true;
		case 0x08u:
			// The CP retries a looping compare-swap until it succeeds; the tracker's thread gives up after a bound.
			for (uint32_t i = 0; i < (loop ? (1u << 20u) : 1u); i++) {
				T expected = cmp;
				if (__atomic_compare_exchange_n(p, &expected, src, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST)) {
					return true;
				}
			}
			return !loop;
		case 0x0fu: __atomic_fetch_add(p, src, __ATOMIC_SEQ_CST); return true;
		case 0x10u: __atomic_fetch_sub(p, src, __ATOMIC_SEQ_CST); return true;
		case 0x11u: return rmw([&](T v) { return static_cast<S>(src) < static_cast<S>(v) ? src : v; });
		case 0x12u: return rmw([&](T v) { return src < v ? src : v; });
		case 0x13u: return rmw([&](T v) { return static_cast<S>(src) > static_cast<S>(v) ? src : v; });
		case 0x14u: return rmw([&](T v) { return src > v ? src : v; });
		case 0x15u: __atomic_fetch_and(p, src, __ATOMIC_SEQ_CST); return true;
		case 0x16u: __atomic_fetch_or(p, src, __ATOMIC_SEQ_CST); return true;
		case 0x17u: __atomic_fetch_xor(p, src, __ATOMIC_SEQ_CST); return true;
		case 0x18u: return rmw([&](T v) { return v >= src ? T(0) : T(v + 1); });          // wraps to 0 past src
		case 0x19u: return rmw([&](T v) { return v == 0 || v > src ? src : T(v - 1); }); // wraps to src below 0
		default: return false;
	}
}
} // namespace

bool CpuAtomicMem(void* addr, uint32_t control, uint64_t src, uint64_t cmp) {
	if (addr == nullptr) {
		return false;
	}
	const uint32_t op   = control & 0x1fu;
	const bool     loop = ((control >> 8u) & 0xfu) == 1u;
	if ((control & 0x20u) != 0) {
		return CpuAtomic(static_cast<uint64_t*>(addr), op, src, cmp, loop);
	}
	return CpuAtomic(static_cast<uint32_t*>(addr), op, static_cast<uint32_t>(src), static_cast<uint32_t>(cmp), loop);
}

uint64_t SatisfyingValue(const WaitCond& w, uint64_t current) {
	const uint64_t ref = w.ref & w.mask;
	uint64_t       want = ref;
	switch (w.function) {
		case 1: want = ref != 0 ? ref - 1 : 0; break;                // <
		case 4: want = (~ref) & w.mask; break;                        // !=
		case 6: want = (ref + 1) & w.mask; break;                     // >
		default: break;                                               // ==, <=, >=, always
	}
	return (current & ~w.mask) | (want & w.mask);
}

namespace {

// Emits a guest wait natively, in the proven form: only EQUAL waits, a 64-bit one as the 32-bit wait
// on its low dword when the high halves of mask and reference are 0 (the same condition), a PFP one as
// the micro engine's wait and PFP_SYNC_ME. Returns false when that is not possible.
bool EmitNativeWait(std::vector<uint32_t>& ib, const WaitCond& w, bool compute) {
	if (w.function != 3) {
		return false;
	}
	EmitWaitEqual(ib, w.addr, static_cast<uint32_t>(w.ref), static_cast<uint32_t>(w.mask), compute);
	// A 64-bit equality wait (Agc's WaitOnAddress, mask ~0) as one wait per half: the label is written
	// once before it is waited on, so both halves holding their part is the 64-bit condition.
	if (w.wide && (w.mask >> 32u) != 0) {
		EmitWaitEqual(ib, w.addr + 4u, static_cast<uint32_t>(w.ref >> 32u), static_cast<uint32_t>(w.mask >> 32u), compute);
	} else if (w.wide && (w.ref >> 32u) != 0) {
		return false; // the high half must match a value the mask leaves out: never true
	}
	// MEC has no prefetch parser. PFP_SYNC_ME there is an illegal opcode; the ME wait is the whole sync.
	if (w.pfp && !compute) {
		ib.push_back((3u << 30u) | (IT_PFP_SYNC_ME << 8u));
		ib.push_back(0);
	}
	return true;
}

// The gfx ring's exit guard (Pm4Translator::SetExitGuard): COND_EXEC on the abort word over a chain to
// the empty IB. 9 dwords.
void EmitGfxExitGuard(std::vector<uint32_t>& ib, uint64_t abort_va, uint64_t stub_va) {
	ib.push_back(Pkt3(IT_COND_EXEC, 4, false));
	ib.push_back(static_cast<uint32_t>(abort_va));
	ib.push_back(static_cast<uint32_t>(abort_va >> 32u));
	ib.push_back(0u);
	ib.push_back(4u);
	ib.push_back(Pkt3(IT_INDIRECT_BUFFER, 3, false));
	ib.push_back(static_cast<uint32_t>(stub_va));
	ib.push_back(static_cast<uint32_t>(stub_va >> 32u) & 0xffffu);
	ib.push_back(8u | (1u << 20u) | (1u << 23u)); // 8 dwords, CHAIN, VALID
}

// A compute ring's exit guard: COND_EXEC on the run word; its EXEC_COUNT (the last dword, returned as an
// offset into `ib`) is set once the IB is assembled.
uint32_t EmitComputeExitGuard(std::vector<uint32_t>& ib, uint64_t run_va) {
	ib.push_back(Pkt3(IT_COND_EXEC, 4, true));
	ib.push_back(static_cast<uint32_t>(run_va));
	ib.push_back(static_cast<uint32_t>(run_va >> 32u));
	ib.push_back(0u);
	ib.push_back(0u);
	return static_cast<uint32_t>(ib.size()) - 1u;
}

// R_ZERO NOP payloads Kyty's command processor runs as flips (pm4Handlers.cpp CpOpMarker): a flip, a
// flip that writes a label first, and one that also raises the EOP interrupt.
bool IsFlipMarker(uint32_t tag) {
	return tag == 0x68750777u || tag == 0x68750778u || tag == 0x68750781u;
}

// WaitUntilSafeForRendering as libSceAgc writes it (agc.cpp): its WAIT_REG_MEM_64 on the display buffer's label is
// followed by libSceAgc's end marker (SET_UCONFIG 0x342, tag 0xCB000020 | index; header bit 2 set in a DCB, clear in
// an ACB) and the padding NOP that carries Kyty's kWaitFlipPadMagic. `at` is the first dword after the wait.
bool WaitFlipPadFollows(std::span<const uint32_t> dcb, size_t at) {
	if (at + 5u > dcb.size() || (dcb[at] & ~4u) != 0xc0017900u || dcb[at + 1] != 0x342u || (dcb[at + 2] & ~0x1fu) != 0xcb000020u) {
		return false;
	}
	const uint32_t nop = dcb[at + 3];
	return (nop >> 30u) == 3u && ((nop >> 8u) & 0xffu) == IT_NOP && ((nop >> 2u) & 0x3fu) == R_ZERO && PacketSizeDw(nop) >= 2u &&
	       at + 3u + PacketSizeDw(nop) <= dcb.size() && dcb[at + 4] == kWaitFlipPadMagic;
}

// Ties a WaitUntilSafeForRendering padding the translator copies into a CPU tail to its FlipMark point (never 0).
std::atomic<uint32_t> g_wait_flip_token {0};
uint32_t NextWaitFlipToken() {
	for (;;) {
		const uint32_t token = g_wait_flip_token.fetch_add(1, std::memory_order_relaxed) + 1u;
		if (token != 0) {
			return token;
		}
	}
}

// ATOMIC_MEM in the forms libSceAgc writes: the flag swap around a context push or pop and SetWorkloadsActive's
// 64-bit swap (TC_OP 0x67), SetWorkloadComplete's 64-bit AND (0x75).
bool AgcAtomicOp(uint32_t control) {
	return (control & 0x7fu) == 0x67u || (control & 0x7fu) == 0x75u;
}

// Every integer ATOMIC_MEM op (SetAtomicOps), by gfx10's TC_OP (Linux navi10_enum.h): bit 6 no return, bit 5 64-bit,
// the low five bits SWAP 0x07, CMPSWAP 0x08, ADD 0x0f, SUB 0x10, SMIN 0x11, UMIN 0x12, SMAX 0x13, UMAX 0x14, AND 0x15,
// OR 0x16, XOR 0x17, INC 0x18, DEC 0x19; the float ops refuse. A returning form (bit 6 clear) leaves memory as its
// no-return twin does: its old value only goes to the CP's internal register, which COMMAND 1 compares and only a
// COPY_DATA SRC_SEL 6 (refused) would read. COMMAND (bits 11:8) 0 once or 2 wait for the write confirm; 1 repeats the
// atomic until the returned old value equals CMP_DATA, the same code in Sony's ME and the stock one (13.60 microcode),
// so it passes on a returning compare-swap, the one op the tracker's CPU form (CpuAtomicMem) repeats the same way. A
// no-return op never refreshes that value and loops forever on both (probe atomicops 2.11: 0x48 with COMMAND 1 stopped
// the gfx ring, 0x08 and 0x28 finished at once). COMMAND 3 (send and continue) refuses. ENGINE_SEL the ME or the PFP
// (libSceAgc's DCB form), never the CE. All 26 no-return ops passed with COMMAND 2 on the ME and the PFP.
bool IntegerAtomicOp(uint32_t control) {
	const uint32_t tc_op   = control & 0x7fu;
	const uint32_t base    = tc_op & 0x1fu;
	const uint32_t command = (control >> 8u) & 0xfu;
	const uint32_t engine  = control >> 30u;
	const bool     integer = base == 0x07u || base == 0x08u || (base >= 0x0fu && base <= 0x19u);
	return integer && engine <= 1u && (command == 0u || command == 2u || (command == 1u && base == 0x08u && (tc_op & 0x40u) == 0));
}

// Packets that make the GPU do something (as opposed to setting state), for dropping an empty first
// segment. Labels, waits and LOD statistics say so themselves.
bool IsWork(uint32_t op) {
	switch (op) {
		case IT_DRAW_INDEX_2:
		case IT_DRAW_INDEX_OFFSET_2:
		case IT_DRAW_INDIRECT:
		case IT_DRAW_INDEX_INDIRECT:
		case IT_DRAW_INDIRECT_MULTI:
		case IT_DRAW_INDEX_INDIRECT_MULTI:
		case IT_DRAW_INDEX_AUTO:
		case IT_DISPATCH_DIRECT:
		case IT_DISPATCH_INDIRECT:
		case IT_DMA_DATA:
		case IT_EVENT_WRITE:
		case IT_ACQUIRE_MEM:
		case IT_PFP_SYNC_ME:
		case IT_GET_LOD_STATS: return true;
		default: return false;
	}
}

// amdgpu's clear state for gfx10 (clearstate_gfx10.h, gfx10_cs_data, MIT): the context registers a CLEAR_STATE resets,
// as [first, end) offsets from 0xa000, and the ones it sets to something other than 0. S holds these for every register
// nothing has written, so LOADs of them give what a CLEAR_STATE and the LOADs give (EmitShadowPreamble).
constexpr std::pair<uint32_t, uint32_t> kClearStateExtents[] = {{0x000u, 0x0d7u}, {0x0d8u, 0x1e8u}, {0x1f5u, 0x1f9u},
                                                                {0x1ffu, 0x29du}, {0x2a0u, 0x2a2u}, {0x2a3u, 0x2a4u},
                                                                {0x2a5u, 0x2e7u}, {0x2f5u, 0x3c0u}};
constexpr std::pair<uint32_t, uint32_t> kClearStateValues[] = {
    {0x00du, 0x40004000u}, // PA_SC_SCREEN_SCISSOR_BR
    {0x01fu, 0x00150055u}, // DB_RMI_L2_CACHE_CONTROL
    {0x081u, 0x80000000u}, // PA_SC_WINDOW_SCISSOR_TL
    {0x082u, 0x40004000u}, // PA_SC_WINDOW_SCISSOR_BR
    {0x083u, 0x0000ffffu}, // PA_SC_CLIPRECT_RULE
    {0x085u, 0x40004000u}, // PA_SC_CLIPRECT_0_BR
    {0x087u, 0x40004000u}, // PA_SC_CLIPRECT_1_BR
    {0x089u, 0x40004000u}, // PA_SC_CLIPRECT_2_BR
    {0x08bu, 0x40004000u}, // PA_SC_CLIPRECT_3_BR
    {0x08cu, 0xaa99aaaau}, // PA_SC_EDGERULE
    {0x08eu, 0xffffffffu}, // CB_TARGET_MASK
    {0x08fu, 0xffffffffu}, // CB_SHADER_MASK
    {0x090u, 0x80000000u}, // PA_SC_GENERIC_SCISSOR_TL
    {0x091u, 0x40004000u}, // PA_SC_GENERIC_SCISSOR_BR
    {0x094u, 0x80000000u}, // PA_SC_VPORT_SCISSOR_0_TL
    {0x095u, 0x40004000u}, // PA_SC_VPORT_SCISSOR_0_BR
    {0x096u, 0x80000000u}, // PA_SC_VPORT_SCISSOR_1_TL
    {0x097u, 0x40004000u}, // PA_SC_VPORT_SCISSOR_1_BR
    {0x098u, 0x80000000u}, // PA_SC_VPORT_SCISSOR_2_TL
    {0x099u, 0x40004000u}, // PA_SC_VPORT_SCISSOR_2_BR
    {0x09au, 0x80000000u}, // PA_SC_VPORT_SCISSOR_3_TL
    {0x09bu, 0x40004000u}, // PA_SC_VPORT_SCISSOR_3_BR
    {0x09cu, 0x80000000u}, // PA_SC_VPORT_SCISSOR_4_TL
    {0x09du, 0x40004000u}, // PA_SC_VPORT_SCISSOR_4_BR
    {0x09eu, 0x80000000u}, // PA_SC_VPORT_SCISSOR_5_TL
    {0x09fu, 0x40004000u}, // PA_SC_VPORT_SCISSOR_5_BR
    {0x0a0u, 0x80000000u}, // PA_SC_VPORT_SCISSOR_6_TL
    {0x0a1u, 0x40004000u}, // PA_SC_VPORT_SCISSOR_6_BR
    {0x0a2u, 0x80000000u}, // PA_SC_VPORT_SCISSOR_7_TL
    {0x0a3u, 0x40004000u}, // PA_SC_VPORT_SCISSOR_7_BR
    {0x0a4u, 0x80000000u}, // PA_SC_VPORT_SCISSOR_8_TL
    {0x0a5u, 0x40004000u}, // PA_SC_VPORT_SCISSOR_8_BR
    {0x0a6u, 0x80000000u}, // PA_SC_VPORT_SCISSOR_9_TL
    {0x0a7u, 0x40004000u}, // PA_SC_VPORT_SCISSOR_9_BR
    {0x0a8u, 0x80000000u}, // PA_SC_VPORT_SCISSOR_10_TL
    {0x0a9u, 0x40004000u}, // PA_SC_VPORT_SCISSOR_10_BR
    {0x0aau, 0x80000000u}, // PA_SC_VPORT_SCISSOR_11_TL
    {0x0abu, 0x40004000u}, // PA_SC_VPORT_SCISSOR_11_BR
    {0x0acu, 0x80000000u}, // PA_SC_VPORT_SCISSOR_12_TL
    {0x0adu, 0x40004000u}, // PA_SC_VPORT_SCISSOR_12_BR
    {0x0aeu, 0x80000000u}, // PA_SC_VPORT_SCISSOR_13_TL
    {0x0afu, 0x40004000u}, // PA_SC_VPORT_SCISSOR_13_BR
    {0x0b0u, 0x80000000u}, // PA_SC_VPORT_SCISSOR_14_TL
    {0x0b1u, 0x40004000u}, // PA_SC_VPORT_SCISSOR_14_BR
    {0x0b2u, 0x80000000u}, // PA_SC_VPORT_SCISSOR_15_TL
    {0x0b3u, 0x40004000u}, // PA_SC_VPORT_SCISSOR_15_BR
    {0x0b5u, 0x3f800000u}, // PA_SC_VPORT_ZMAX_0
    {0x0b7u, 0x3f800000u}, // PA_SC_VPORT_ZMAX_1
    {0x0b9u, 0x3f800000u}, // PA_SC_VPORT_ZMAX_2
    {0x0bbu, 0x3f800000u}, // PA_SC_VPORT_ZMAX_3
    {0x0bdu, 0x3f800000u}, // PA_SC_VPORT_ZMAX_4
    {0x0bfu, 0x3f800000u}, // PA_SC_VPORT_ZMAX_5
    {0x0c1u, 0x3f800000u}, // PA_SC_VPORT_ZMAX_6
    {0x0c3u, 0x3f800000u}, // PA_SC_VPORT_ZMAX_7
    {0x0c5u, 0x3f800000u}, // PA_SC_VPORT_ZMAX_8
    {0x0c7u, 0x3f800000u}, // PA_SC_VPORT_ZMAX_9
    {0x0c9u, 0x3f800000u}, // PA_SC_VPORT_ZMAX_10
    {0x0cbu, 0x3f800000u}, // PA_SC_VPORT_ZMAX_11
    {0x0cdu, 0x3f800000u}, // PA_SC_VPORT_ZMAX_12
    {0x0cfu, 0x3f800000u}, // PA_SC_VPORT_ZMAX_13
    {0x0d1u, 0x3f800000u}, // PA_SC_VPORT_ZMAX_14
    {0x0d3u, 0x3f800000u}, // PA_SC_VPORT_ZMAX_15
    {0x100u, 0xffffffffu}, // VGT_MAX_VTX_INDX
    {0x104u, 0x00550055u}, // CB_RMI_GL2_CACHE_CONTROL
    {0x10cu, 0x01000000u}, // DB_STENCILREFMASK
    {0x10du, 0x01000000u}, // DB_STENCILREFMASK_BF
    {0x1b6u, 0x00000002u}, // SPI_PS_IN_CONTROL
    {0x204u, 0x00090000u}, // PA_CL_CLIP_CNTL
    {0x205u, 0x00000004u}, // PA_SU_SC_MODE_CNTL
    {0x295u, 0x00000100u}, // VGT_GS_PER_ES
    {0x296u, 0x00000080u}, // VGT_ES_PER_GS
    {0x297u, 0x00000002u}, // VGT_GS_PER_VS
    {0x2aau, 0x000000ffu}, // IA_MULTI_VGT_PARAM
    {0x2f7u, 0x00001000u}, // PA_SC_LINE_CNTL
    {0x2f9u, 0x00000005u}, // PA_SU_VTX_CNTL
    {0x2fau, 0x3f800000u}, // PA_CL_GB_VERT_CLIP_ADJ
    {0x2fbu, 0x3f800000u}, // PA_CL_GB_VERT_DISC_ADJ
    {0x2fcu, 0x3f800000u}, // PA_CL_GB_HORZ_CLIP_ADJ
    {0x2fdu, 0x3f800000u}, // PA_CL_GB_HORZ_DISC_ADJ
    {0x30eu, 0xffffffffu}, // PA_SC_AA_MASK_X0Y0_X1Y0
    {0x30fu, 0xffffffffu}, // PA_SC_AA_MASK_X0Y1_X1Y1
    {0x311u, 0x00000003u}, // PA_SC_BINNER_CNTL_0
    {0x313u, 0x00100000u}, // PA_SC_CONSERVATIVE_RASTERIZATION_CNTL
    {0x316u, 0x0000001eu}, // VGT_VERTEX_REUSE_BLOCK_CNTL
    {0x317u, 0x00000020u}, // VGT_OUT_DEALLOC_CNTL
};

bool InClearState(uint32_t off) {
	return std::any_of(std::begin(kClearStateExtents), std::end(kClearStateExtents),
	                   [&](const auto& e) { return off >= e.first && off < e.second; });
}
uint32_t ClearStateValue(uint32_t off) {
	const auto* it = std::find_if(std::begin(kClearStateValues), std::end(kClearStateValues), [&](const auto& v) { return v.first == off; });
	return it != std::end(kClearStateValues) ? it->second : 0u;
}

// What a shadow preamble restores with LOAD_*_REG (Pm4Translator::SetCpShadow); the rest of the copy is SET.
// PA_SC_SHADER_CONTROL takes the PS5's RCIU CAM override, which may not apply to a LOAD's writes.
bool CpShadowLoadsContext(uint32_t off) {
	return off < RegFile::kDense && off != kPaScShaderControl;
}
// PAL never shadows RSRC4/RSRC3 of PS, VS, GS and HS nor COMPUTE_STATIC_THREAD_MGMT_SE0..3 (unverified here: the
// checker shows it), and SH 0x280 is the guest's NOP.
bool CpShadowLoadsSh(uint32_t off) {
	switch (off) {
		case 0x001:
		case 0x007:
		case 0x041:
		case 0x046:
		case 0x081:
		case 0x087:
		case 0x101:
		case 0x107:
		case 0x216:
		case 0x217:
		case 0x219:
		case 0x21A:
		case kShNop: return false;
		default: return off < RegFile::kDense;
	}
}
// The CP keeps its own VGT_PRIMITIVE_TYPE and VGT_INDEX_TYPE, which only SET_UCONFIG_REG_INDEX updates.
bool CpShadowLoadsUconfig(uint32_t off) {
	return off < RegFile::kDense && off != 0x242u && off != 0x243u;
}

// The MRT slots whose display-DCC decision reads context register `off`: CB_COLORn_BASE, BASE_EXT, INFO.
uint32_t DccFixSlots(uint32_t off) {
	if (off >= kCbColor0Base && off < kCbColor0Base + 8 * kCbColorStride) {
		const uint32_t rel = off - kCbColor0Base;
		return rel % kCbColorStride == 0 || rel % kCbColorStride == kCbColor0Info - kCbColor0Base ? 1u << (rel / kCbColorStride) : 0u;
	}
	if (off >= kCbColor0BaseExt && off < kCbColor0BaseExt + 8) {
		return 1u << (off - kCbColor0BaseExt);
	}
	return 0u;
}

// The MRT slots whose FMASK fix reads context register `off` (CB_TARGET_MASK: all of them).
uint32_t FmaskFixSlots(uint32_t off) {
	if (off == kCbTargetMask) {
		return 0xffu;
	}
	if (off >= kCbColor0Base && off < kCbColor0Base + 8 * kCbColorStride) {
		const uint32_t rel = off - kCbColor0Base;
		return rel % kCbColorStride == 0 || rel % kCbColorStride == kCbColor0Fmask - kCbColor0Base ? 1u << (rel / kCbColorStride) : 0u;
	}
	if (off >= kCbColor0BaseExt && off < kCbColor0BaseExt + 8) {
		return 1u << (off - kCbColor0BaseExt);
	}
	if (off >= kCbColor0FmaskExt && off < kCbColor0FmaskExt + 8) {
		return 1u << (off - kCbColor0FmaskExt);
	}
	return 0u;
}

// LOAD_*_REG from S's region at `base` of the keys of `regs` in [lo, hi) that `loads` takes, in runs. A class
// with none LOADs `anchor` (value 0 unless the GPU wrote it): the CP takes a class's base only from a LOAD that
// loads, and a shadowed write through a base this job never set goes near VA 0 (a VM fault).
// `also`: registers loaded though the copy has no value for them (S holds one: the context's clear state).
void EmitCpShadowLoad(std::vector<uint32_t>& ib, Pm4TranslateStats& stats, uint32_t op, bool compute, uint64_t base, const RegFile& regs,
                      uint32_t lo, uint32_t hi, bool (*loads)(uint32_t), uint32_t anchor, bool (*also)(uint32_t) = nullptr) {
	const size_t at = ib.size();
	ib.push_back(0u); // the header, once the length is known
	ib.push_back(static_cast<uint32_t>(base));
	ib.push_back(static_cast<uint32_t>(base >> 32u));
	uint32_t run_off = 0, run_len = 0;
	auto     flush = [&]() {
        if (run_len != 0) {
            ib.push_back(run_off);
            ib.push_back(run_len);
            stats.cp_shadow_load_regs += run_len;
            run_len = 0;
        }
	};
	for (uint32_t off = lo; off < hi && off < RegFile::kDense; off++) {
		if (!(regs.Find(off) != nullptr || (also != nullptr && also(off))) || !loads(off)) {
			continue;
		}
		if (run_len != 0 && run_off + run_len == off) {
			run_len++;
			continue;
		}
		flush();
		run_off = off;
		run_len = 1;
	}
	flush();
	if (ib.size() == at + 3) {
		ib.push_back(anchor);
		ib.push_back(1u);
		stats.cp_shadow_load_regs++;
		stats.cp_shadow_anchors++;
	}
	ib[at] = Pkt3(op, static_cast<uint32_t>(ib.size() - at - 1), compute);
	stats.cp_shadow_load_dw += static_cast<uint32_t>(ib.size() - at);
}

} // namespace

// First time each guest side effect is handled. A board run then shows which forms the title uses.
void NoteOnce(uint32_t id, const char* msg) {
	static std::atomic<uint64_t> lo {0};
	static std::atomic<uint64_t> hi {0};
	if (id >= 128u) {
		return;
	}
	auto&          word = id < 64u ? lo : hi;
	const uint64_t bit  = 1ull << (id & 63u);
	if ((word.fetch_or(bit, std::memory_order_relaxed) & bit) != 0) {
		return;
	}
	std::fprintf(stderr, "native labels: %s\n", msg);
}

void NoteEvent(uint32_t event_type) {
	static std::atomic<uint64_t> seen {0};
	if (event_type >= 64u) {
		return;
	}
	const uint64_t bit = 1ull << event_type;
	if ((seen.fetch_or(bit, std::memory_order_relaxed) & bit) != 0) {
		return;
	}
	std::fprintf(stderr, "native labels: EVENT_WRITE type 0x%02x\n", event_type);
}

bool WaitCond::Met(uint64_t value) const {
	return WaitConditionMet(function, value & mask, ref);
}

int LabelWrite::Covers(uint64_t a, bool wide_wait, uint64_t* out, uint64_t wait_mask) const {
	const bool low_only = wide_wait && (wait_mask >> 32u) == 0;
	if (span_dw > 2) {
		const uint64_t need = wide_wait && !low_only ? 8u : 4u;
		const uint64_t end  = addr + 4ull * span_dw;
		if (a + need <= addr || a >= end) {
			return 0;
		}
		if (data == nullptr || !known || (a & 3u) != 0 || a < addr || a + need > end) {
			return 2;
		}
		const uint64_t i = (a - addr) / 4u;
		*out             = need == 8u ? (data[i] | (static_cast<uint64_t>(data[i + 1]) << 32u)) : data[i];
		return 1;
	}
	if (a == addr) {
		if (wide_wait && !wide && !low_only) {
			return 2; // half of what the wait reads
		}
		*out = wide_wait ? value : (value & 0xffffffffull);
		return known ? 1 : 2;
	}
	if (wide && !wide_wait && a == addr + 4) {
		*out = value >> 32u;
		return known ? 1 : 2;
	}
	if (wide_wait && !wide && a + 4 == addr) {
		return low_only ? 0 : 2; // the high half, which the mask may leave out
	}
	return 0;
}

void NopOut(std::span<uint32_t> ib, uint32_t dw, uint32_t len) {
	if (dw + len > ib.size()) {
		return;
	}
	for (uint32_t i = 0; i < len; i++) {
		ib[dw + i] = kFiller;
	}
}

uint32_t WaitFlipGroupDw(std::span<const uint32_t> dcb, size_t at) {
	if (at >= dcb.size()) {
		return 0;
	}
	const uint32_t header = dcb[at];
	if ((header >> 30u) != 3u || ((header >> 8u) & 0xffu) != IT_WAIT_REG_MEM_64) {
		return 0;
	}
	const size_t wait = PacketSizeDw(header);
	if (!WaitFlipPadFollows(dcb, at + wait)) {
		return 0;
	}
	return static_cast<uint32_t>(wait + 3u + PacketSizeDw(dcb[at + wait + 3u]));
}

uint32_t SetWaitFlipMarks(std::span<uint32_t> tail, const std::function<uint64_t(uint32_t token)>& mark) {
	uint32_t set = 0;
	for (size_t dw = 0; dw < tail.size();) {
		const uint32_t header = tail[dw];
		const size_t   size   = (header >> 30u) == 3u ? PacketSizeDw(header) : 1u;
		if (dw + size > tail.size()) {
			break;
		}
		const size_t body_count = size - 1u;
		uint32_t*    body       = tail.data() + dw + 1;
		if ((header >> 30u) == 3u && ((header >> 8u) & 0xffu) == IT_NOP && ((header >> 2u) & 0x3fu) == R_ZERO &&
		    body_count > kWaitFlipPadEopMark + 1u && body[0] == kWaitFlipPadMagic && body[kWaitFlipPadEopMark] == 0xffffffffu &&
		    body[kWaitFlipPadEopMark + 1u] == 0xffffffffu) {
			const uint64_t value           = mark(body_count > kWaitFlipPadToken ? body[kWaitFlipPadToken] : 0u);
			body[kWaitFlipPadEopMark]      = static_cast<uint32_t>(value);
			body[kWaitFlipPadEopMark + 1u] = static_cast<uint32_t>(value >> 32u);
			set++;
		}
		dw += size;
	}
	return set;
}

uint32_t AppendKernelReleases(std::span<const uint32_t> dcb, std::vector<uint32_t>* out) {
	uint32_t n = 0;
	for (size_t dw = 0; dw < dcb.size();) {
		const uint32_t header = dcb[dw];
		const size_t   size   = (header >> 30u) == 3u ? PacketSizeDw(header) : 1u;
		if (dw + size > dcb.size()) {
			break;
		}
		const uint32_t op = (header >> 8u) & 0xffu;
		// libSceAgc's RELEASE_MEM, or Kyty's NOP marker with the same body: [1] DATA_SEL / INT_SEL, [6] the context id.
		const bool release = (header >> 30u) == 3u && (op == IT_RELEASE_MEM || (op == IT_NOP && ((header >> 2u) & 0x3fu) == R_RELEASE_MEM));
		if (release && size >= 8u) {
			const uint32_t* bptr    = dcb.data() + dw + 1;
			const uint32_t  int_sel = (bptr[1] >> 24u) & 0x7u;
			if (int_sel != 0 && (bptr[6] & 0x08000000u) != 0) {
				out->insert(out->end(), dcb.begin() + dw, dcb.begin() + dw + size);
				n++;
			}
		}
		dw += size;
	}
	return n;
}

// Opcodes an ACB passes to MEC with the header as the title wrote it, without the shader-type bit the
// translator set before: libSceAgc does not set it, and the PS5's MEC runs them so. The BC-250's MEC does too
// for SET_SH_REG, DISPATCH_DIRECT/INDIRECT, ACQUIRE_MEM, DMA_DATA and EVENT_WRITE (probe --danger mecforms;
// ASTRO BOT plays).
// KYTY_NATIVE_GPU_ACB_PLAIN=<opcodes, comma-separated> replaces the list; =0 sets the bit on every packet.
static bool MecPlainOp(uint32_t op) {
	static const std::array<bool, 256> plain = [] {
		std::array<bool, 256> a {};
		const char*           v = std::getenv("KYTY_NATIVE_GPU_ACB_PLAIN");
		if (v == nullptr) {
			// RELEASE_MEM: amdgpu's own compute-ring fences are plain RELEASE_MEMs, and libSceAgc writes its ACB ones
			// plain too.
			for (uint32_t o: {IT_SET_SH_REG, IT_DISPATCH_DIRECT, IT_DISPATCH_INDIRECT, IT_ACQUIRE_MEM, IT_DMA_DATA, IT_EVENT_WRITE,
			                  IT_RELEASE_MEM}) {
				a[o] = true;
			}
		}
		while (v != nullptr && *v != '\0') {
			char*               end = nullptr;
			const unsigned long n   = std::strtoul(v, &end, 0);
			if (end == v) {
				break;
			}
			if (n < 256) {
				a[n] = true;
			}
			v = *end == ',' ? end + 1 : end;
		}
		return a;
	}();
	return plain[op & 0xffu];
}

// An ACB's WAIT_REG_MEM / WAIT_REG_MEM_64 go to MEC as written, the micro engine's (MEC has no prefetch parser;
// ASTRO BOT plays). KYTY_NATIVE_GPU_ACB_WAITS_VERBATIM=0 emits our form.
static bool AcbWaitsVerbatim() {
	static const bool on = [] {
		const char* v = std::getenv("KYTY_NATIVE_GPU_ACB_WAITS_VERBATIM");
		return v == nullptr || v[0] != '0';
	}();
	return on;
}

static bool MecGuestWaitOk(uint32_t control) {
	return AcbWaitsVerbatim() && GuestWaitControlOk(control) && ((control >> 8u) & 0x3u) == 0u;
}

namespace {

std::atomic<int> g_validate_strict {-1}; // -1: KYTY_NATIVE_GPU_VALIDATE_STRICT decides

bool ValidateStrict() {
	int v = g_validate_strict.load(std::memory_order_relaxed);
	if (v < 0) {
		const char* e = std::getenv("KYTY_NATIVE_GPU_VALIDATE_STRICT");
		v             = e != nullptr && e[0] == '1' ? 1 : 0;
		g_validate_strict.store(v, std::memory_order_relaxed);
	}
	return v == 1;
}

// Packets of the kernel and of queue management (KIQ), which a title never writes and which reprogram the CP, the VM
// or the queues: ME_INITIALIZE, PREAMBLE_CNTL, INTERRUPT, GEN_PDEPTE, INDIRECT_BUFFER_PASID, PRIME_UTCL2,
// LOAD_CONFIG_REG, SET_CONFIG_REG, SET_QUEUE_REG, INVALIDATE_TLBS, INDIRECT_BUFFER_PRIV, REG_RMW, and 0xA0-0xAF
// (SET_RESOURCES, MAP_PROCESS, MAP_QUEUES, UNMAP_QUEUES, QUERY_STATUS, RUN_LIST ...).
bool PrivilegedOp(uint32_t op) {
	switch (op) {
		case 0x21u:
		case 0x32u:
		case 0x44u:
		case 0x4Au:
		case 0x5Au:
		case 0x5Bu:
		case 0x5Cu:
		case 0x5Du:
		case 0x60u:
		case 0x68u:
		case 0x78u:
		case 0x98u: return true;
		default: return op >= 0xA0u && op <= 0xAFu;
	}
}

// What a compute ring (MEC) cannot run: an illegal opcode there stops the ring.
bool GraphicsOnlyOp(uint32_t op) {
	switch (op) {
		case IT_CLEAR_STATE:
		case IT_SET_PREDICATION:
		case IT_INDEX_BUFFER_SIZE:
		case IT_DRAW_INDIRECT:
		case IT_DRAW_INDEX_INDIRECT:
		case IT_DRAW_INDIRECT_MULTI:
		case IT_DRAW_INDEX_INDIRECT_MULTI:
		case IT_INDEX_BASE:
		case IT_DRAW_INDEX_2:
		case IT_DRAW_INDEX_OFFSET_2:
		case IT_DRAW_INDEX_AUTO:
		case IT_NUM_INSTANCES:
		case IT_PFP_SYNC_ME:
		case IT_CONTEXT_CONTROL:
		case IT_SET_CONTEXT_REG:
		case IT_SET_CONTEXT_REG_INDIRECT:
		case IT_GET_LOD_STATS:
		case IT_LOAD_UCONFIG_REG:
		case IT_LOAD_CONTEXT_REG:
		case IT_LOAD_SH_REG:
		case IT_INDIRECT_BUFFER: // a chain on MEC took the board down (probe condexec)
			return true;
		default: return false;
	}
}

bool ValidateNativeIbStrict(std::span<const uint32_t> ib, bool lod_stats, uint32_t* bad_dw, NativeIbKind kind);

// The blacklist (ValidateNativeIb).
bool ValidateNativeIbBlacklist(std::span<const uint32_t> ib, bool lod_stats, uint32_t* bad_dw, NativeIbKind kind) {
	const bool compute = kind == NativeIbKind::Compute;
	const uint64_t s   = g_cp_shadow_va.load(std::memory_order_relaxed);
	uint32_t dw = 0;
	auto     bad = [&]() {
		if (bad_dw != nullptr) {
			*bad_dw = dw;
		}
		return false;
	};
	// The translator's own forms for the packets that touch CP shadowing (and any other packet of the IB at `at`, size
	// `size`): what the strict check says of that one packet.
	auto strict_one = [&](uint32_t at, uint32_t size) {
		uint32_t unused = 0;
		return ValidateNativeIbStrict(ib.subspan(at, size), lod_stats, &unused, kind);
	};
	auto in_s = [&](uint64_t a, uint64_t bytes) { return s != 0 && a < s + kCpShadowReserveBytes && a + bytes > s; };
	while (dw < ib.size()) {
		const uint32_t header = ib[dw];
		if ((header >> 30u) != 3u) {
			return bad();
		}
		const uint32_t size = PacketSizeDw(header);
		if (dw + size > ib.size()) {
			return bad();
		}
		const uint32_t  op = (header >> 8u) & 0xffu;
		const uint32_t* b  = ib.data() + dw + 1;
		const uint32_t  n  = size - 1u; // body dwords
		if (PrivilegedOp(op) || (compute && GraphicsOnlyOp(op))) {
			return bad();
		}
		// On MEC these need the shader-type bit unless MEC runs their plain form (KYTY_NATIVE_GPU_MEC_PLAIN_OPS); a title's
		// waits go as written without it (ASTRO BOT plays so).
		switch (op) {
			case IT_SET_BASE:
			case IT_DISPATCH_DIRECT:
			case IT_DISPATCH_INDIRECT:
			case IT_EVENT_WRITE:
			case IT_ACQUIRE_MEM:
			case IT_SET_SH_REG:
			case IT_DMA_DATA:
			case IT_WRITE_DATA:
			case IT_RELEASE_MEM:
				if (compute && (header & 2u) == 0 && !MecPlainOp(op)) {
					return bad();
				}
				break;
			default: break;
		}
		switch (op) {
			// CP shadowing: only the translator's forms (the enables while S is registered, LOADs from S).
			case IT_CONTEXT_CONTROL:
			case IT_LOAD_UCONFIG_REG:
			case IT_LOAD_CONTEXT_REG:
			case IT_LOAD_SH_REG:
				if (!strict_one(dw, size)) {
					return bad();
				}
				break;
			// The FMASK fix's COND_EXEC over S's tables is the translator's; any other skips no further than the IB.
			case IT_COND_EXEC:
				if (n < 4 || (s != 0 && in_s(b[0] | (static_cast<uint64_t>(b[1]) << 32u), 4) && !strict_one(dw, std::min<uint32_t>(10u, static_cast<uint32_t>(ib.size()) - dw)))) {
					return bad();
				}
				break;
			// S, P and the fix tables only through the translator's DMA_DATA and ATOMIC_MEM; no register space.
			case IT_DMA_DATA: {
				if (n < 6 || (b[5] & (3u << 26u)) != 0u) { // SAS / DAS: register addresses
					return bad();
				}
				const uint32_t src_sel = (b[0] >> 29u) & 0x3u;
				const uint32_t dst_sel = (b[0] >> 20u) & 0x3u;
				const uint64_t bytes   = std::max<uint64_t>(b[5] & 0x3ffffffu, 1u);
				const uint64_t src     = b[1] | (static_cast<uint64_t>(b[2]) << 32u);
				const uint64_t dst     = b[3] | (static_cast<uint64_t>(b[4]) << 32u);
				if ((((src_sel == 0u || src_sel == 3u) && in_s(src, bytes)) || ((dst_sel == 0u || dst_sel == 3u) && in_s(dst, bytes))) &&
				    !strict_one(dw, size)) {
					return bad();
				}
				break;
			}
			case IT_ATOMIC_MEM:
				if (n < 3 || (in_s(b[1] | (static_cast<uint64_t>(b[2]) << 32u), 8) && !strict_one(dw, size))) {
					return bad();
				}
				break;
			// Register space: a write is a privileged register as likely as not, and a read of a context or UCONFIG
			// register stopped the ring (probe); the translator's GPU clock copy is the one read.
			case IT_WRITE_DATA:
				// libSceAgc's debug-marker strings go to SQ_THREAD_TRACE_USERDATA_2/3, the one register write passed on.
				if (n < 3 || (((b[0] >> 8u) & 0xfu) == 0u && !(!compute && TraceUserDataWrite(b[0], b[1], b[2])))) {
					return bad();
				}
				break;
			case IT_COPY_DATA:
				if (n < 5 || ((b[0] & 0xfu) == 0u && b[0] != kClockCopyCntl) || ((b[0] >> 8u) & 0xfu) == 0u) {
					return bad();
				}
				break;
			case IT_COND_WRITE: // POLL_SPACE 0 or WRITE_SPACE 0: a register
				if (n < 1 || (b[0] & (1u << 4u)) == 0u || ((b[0] >> 8u) & 0x3u) == 0u) {
					return bad();
				}
				break;
			case IT_WAIT_REG_MEM:
			case IT_WAIT_REG_MEM_64:
				if (n < 1 || (b[0] & (1u << 4u)) == 0u) { // MEM_SPACE 0: a register
					return bad();
				}
				break;
			// An IB at VA 0 or of no length.
			case IT_INDIRECT_BUFFER:
				if (n < 3 || ((b[0] & ~3u) == 0u && (b[1] & 0xffffu) == 0u) || (b[2] & 0xfffffu) == 0u) {
					return bad();
				}
				break;
			case IT_GET_LOD_STATS:
				if (!lod_stats) {
					return bad();
				}
				break;
			default: break;
		}
		dw += size;
	}
	return true;
}

} // namespace

void SetNativeIbValidation(bool strict) {
	g_validate_strict.store(strict ? 1 : 0, std::memory_order_relaxed);
}

bool ValidateNativeIb(std::span<const uint32_t> ib, bool lod_stats, uint32_t* bad_dw, NativeIbKind kind) {
	return ValidateStrict() ? ValidateNativeIbStrict(ib, lod_stats, bad_dw, kind) : ValidateNativeIbBlacklist(ib, lod_stats, bad_dw, kind);
}

namespace {

bool ValidateNativeIbStrict(std::span<const uint32_t> ib, bool lod_stats, uint32_t* bad_dw, NativeIbKind kind) {
	const bool compute = kind == NativeIbKind::Compute;
	uint32_t dw = 0;
	auto     bad = [&]() {
		if (bad_dw != nullptr) {
			*bad_dw = dw;
		}
		return false;
	};
	while (dw < ib.size()) {
		const uint32_t header = ib[dw];
		if ((header >> 30u) != 3u) {
			return bad();
		}
		const uint32_t size = PacketSizeDw(header);
		if (dw + size > ib.size()) {
			return bad();
		}
		const uint32_t* b = ib.data() + dw + 1;
		switch ((header >> 8u) & 0xffu) {
			case IT_NOP:
				// A title's NOP on the gfx ring (AgcCbNop) is skipped by the CP; its length is checked above.
				if (header != kFiller && compute) {
					return bad();
				}
				break;
			case IT_RELEASE_MEM:
				// The barrier's form (any address and data), a GPU-clock timestamp (DATA_SEL 3, data
				// dwords zero, 8-byte address), or a GDS release as the translator has always run it
				// (DATA_SEL 5, no interrupt). Compute rings use the same bodies with the shader bit.
				// A tracker release may raise the EOP interrupt (INT_SEL 2) with the tracker's KFD event as
				// its context id (the pipeline sets both when it numbers the point), and a title's interrupt
				// keeps its context id: INT_SEL 2 after a label or into the scratch dword, or the title's
				// interrupt alone as written (INT_SEL 1, no data, gfx ring).
				if ((header != Pkt3(IT_RELEASE_MEM, 7, compute) && !(compute && MecPlainOp(IT_RELEASE_MEM) && header == Pkt3(IT_RELEASE_MEM, 7, false))) ||
				    (b[6] != 0u && ((b[1] >> 24u) & 0x7u) == 0u)) {
					return bad();
				}
				if ((b[0] == kFlushEventCntl || LabelEventOk(b[0])) && (b[1] == kData32Cntl || b[1] == (kData32Cntl | kIntSel2))) {
					break; // the barrier's form, or a label in the guest's event (LabelEventCntl)
				}
				if (!compute && LabelEventOk(b[0]) && (b[1] & ~(0x3u << 16u)) == (1u << 24u)) {
					break; // a title's interrupt alone, as written
				}
				if (LabelEventOk(b[0]) && ((b[1] >> 24u) & 0x7u) == 0u && ((b[1] >> 16u) & 0x3u) <= 1u &&
				    (b[1] & ~((0x7u << 29u) | (0x3u << 16u))) == 0u &&
				    (((b[1] >> 29u) & 0x7u) <= 2u || (((b[1] >> 29u) & 0x7u) == 3u && b[4] == 0u && b[5] == 0u && (b[2] & 7u) == 0u))) {
					break; // the title's label, GPU-clock timestamp or cache actions as written (ReleasesAsWritten)
				}
				if ((b[0] == kFlushEventCntl || (b[0] == kClockEopCntl && !compute)) && b[1] == kDataTimestampCntl &&
				    b[4] == 0u && b[5] == 0u && (b[2] & 7u) == 0u) {
					break;
				}
				if (((b[1] >> 29u) & 0x7u) == 5u && ((b[1] >> 24u) & 0x7u) == 0u) {
					break;
				}
				return bad();
			case IT_WAIT_REG_MEM:
				if (header == Pkt3(IT_WAIT_REG_MEM, 6, compute) && b[0] == kWaitEqualMemMe && b[5] == kWaitPoll) {
					break; // the translator's own form, on either ring
				}
				if ((compute && !MecGuestWaitOk(b[0])) || size != 7 || (header & 2u) != 0 || !GuestWaitControlOk(b[0])) {
					return bad();
				}
				break; // a guest wait as written
			case IT_WAIT_REG_MEM_64:
				if ((compute && !MecGuestWaitOk(b[0])) || size != 9 || (header & 2u) != 0 || !GuestWaitControlOk(b[0])) {
					return bad();
				}
				break;
			case IT_GET_LOD_STATS:
				if (compute || !lod_stats) {
					return bad();
				}
				break;
			// GPU timing (SetGpuTimes): the GPU clock to memory, on the gfx ring. A title's COPY_DATA as written
			// (SetCopyData): memory, an immediate or the GPU clock into memory, the shader-type bit on MEC.
			case IT_COPY_DATA: {
				if (size == 6 && !compute && b[0] == kClockCopyCntl && (b[3] & 7u) == 0u) {
					break;
				}
				const uint32_t src_sel = b[0] & 0xfu;
				const uint32_t dst_sel = (b[0] >> 8u) & 0xfu;
				if (size != 6 || (compute && (header & 2u) == 0 && !MecPlainOp(IT_COPY_DATA)) || !(src_sel == 1u || src_sel == 2u || src_sel == 5u || src_sel == 9u) ||
				    !(dst_sel == 1u || dst_sel == 2u || dst_sel == 5u) || (b[3] & 3u) != 0u || (b[3] == 0u && b[4] == 0u) ||
				    (b[0] >> 30u) > (compute ? 0u : 1u)) { // ENGINE_SEL: the micro engine or the prefetch parser
					return bad();
				}
				break;
			}
			// The translator's IB2 into the guest's buffer (SetDirectRuns): a call, not a chain, on the gfx
			// ring. Its packets are the ones the translator passed on as written. A chain only as an exit
			// guard's, which the COND_EXEC case takes with it.
			case IT_INDIRECT_BUFFER:
				if (compute || size != 4 || (b[2] & (1u << 20u)) != 0 || (b[2] & (1u << 23u)) == 0) {
					return bad();
				}
				break;
			// An exit guard (Pm4Translator::SetExitGuard). On the gfx ring over its chain to the empty IB, 8
			// dwords; on MEC skipping what follows, no further than kExitGuardMaxSkip.
			case IT_COND_EXEC:
				if (size != 5 || b[2] != 0u || header != Pkt3(IT_COND_EXEC, 4, compute)) {
					return bad();
				}
				// CP shadowing's FMASK fix (Pm4Translator::SetCpShadow, load mode): on a copied value in the fix tables,
				// over one of their tables, as the translator writes it.
				if (const uint64_t s = g_cp_shadow_va.load(std::memory_order_relaxed); !compute && s != 0 && b[3] == 5u) {
					const uint64_t a   = b[0] | (static_cast<uint64_t>(b[1]) << 32u);
					const uint64_t fix = s + kCpShadowFixOffset;
					const uint64_t end = s + kCpShadowDccFixOffset; // the FMASK fix's tables
					if (a >= fix && a < end && (a - fix) % 8u == 4u && dw + 10u <= ib.size() &&
					    ib[dw + 5] == Pkt3(IT_SET_CONTEXT_REG_INDIRECT, 4, false)) {
						const uint64_t t = ib[dw + 6] | (static_cast<uint64_t>(ib[dw + 7]) << 32u);
						if (t >= fix && t + 16u <= end && (t - fix) % 16u == 0 && ib[dw + 8] == 0x80000000u &&
						    ib[dw + 9] == 2u) {
							break;
						}
					}
					return bad();
				}
				if (!compute) {
					const uint32_t* c = b + 4;
					if (b[3] != 4u || dw + 9u > ib.size() || c[0] != Pkt3(IT_INDIRECT_BUFFER, 3, false) ||
					    c[3] != (8u | (1u << 20u) | (1u << 23u))) {
						// A title's COND_EXEC (SetCondExec) after the translator's PFP_SYNC_ME, its count within the IB.
						if (dw < 2u || ib[dw - 2u] != ((3u << 30u) | (IT_PFP_SYNC_ME << 8u)) || b[3] > 0x3fffu || dw + size + b[3] > ib.size()) {
							return bad();
						}
						break;
					}
					dw += 4u; // the chain
				} else if (b[3] > kExitGuardMaxSkip || dw + size + b[3] > ib.size()) {
					return bad();
				}
				break;
			// A title's MEM_SEMAPHORE as libSceAgc writes it (SetMemSemaphore): signal or wait, the gfx ring only.
			case IT_MEM_SEMAPHORE:
				if (compute || size != 4 || header != Pkt3(IT_MEM_SEMAPHORE, 3, false) || ((b[2] >> 29u) != 6u && (b[2] >> 29u) != 7u) ||
				    (b[2] & ~((7u << 29u) | (1u << 20u) | (1u << 16u))) != 0) {
					return bad();
				}
				break;
			// A guest WRITE_DATA to memory, write confirmed; on MEC from the micro engine.
			case IT_WRITE_DATA:
				if (!compute && size >= 5 && TraceUserDataWrite(b[0], b[1], b[2])) {
					break; // a debug marker's string, on the gfx ring as written
				}
				if (size < 5 || !WriteDataToMemory(b[0]) || (b[0] & (1u << 20u)) == 0 ||
				    (compute && (((header & 2u) == 0 && !MecPlainOp(IT_WRITE_DATA)) || (b[0] >> 30u) != 0))) {
					return bad();
				}
				break;
			// A DMA_DATA that reads or writes CP shadowing's S or its save area P only as the GPU push or pop
			// (Pm4Translator::SetCpShadowPushPop) or the GPU's FMASK fix: the gfx ring, S's context region to P or
			// back, or one context register of S (or the zero word) into a value dword of the fix tables, exactly as
			// the translator writes it. Any other one passes as before.
			case IT_DMA_DATA: {
				const uint64_t s = g_cp_shadow_va.load(std::memory_order_relaxed);
				if (s != 0 && size >= 7) {
					const uint32_t src_sel = (b[0] >> 29u) & 0x3u;
					const uint32_t dst_sel = (b[0] >> 20u) & 0x3u;
					const uint64_t bytes   = std::max<uint64_t>(b[5] & 0x3ffffffu, 1u);
					const uint64_t src     = b[1] | (static_cast<uint64_t>(b[2]) << 32u);
					const uint64_t dst     = b[3] | (static_cast<uint64_t>(b[4]) << 32u);
					auto           in_s    = [&](uint64_t a) { return a < s + kCpShadowReserveBytes && a + bytes > s; };
					const bool     touches = ((src_sel == 0u || src_sel == 3u) && in_s(src)) || ((dst_sel == 0u || dst_sel == 3u) && in_s(dst));
					if (touches) {
						const uint64_t cx   = s + 4ull * kCpShadowContextDw;
						const uint64_t p    = s + kCpShadowSaveOffset;
						const uint64_t fix  = s + kCpShadowFixOffset;
						const bool     ours = header == Pkt3(IT_DMA_DATA, 6, false) && b[0] == kCpShadowDmaControl;
						const bool     word = b[5] == kCpShadowDmaWordCommand &&
						                  (((src & 3u) == 0 && src >= cx && src < cx + 4ull * RegFile::kDense) || src == s + kCpShadowZeroOffset) &&
						                  dst >= fix && dst < fix + kCpShadowFixBytes && (dst - fix) % 8u == 4u; // the FMASK fix
						const bool form = ours && ((b[5] == kCpShadowDmaCommand && ((src == cx && dst == p) || (src == p && dst == cx))) || word);
						if (compute || !form) {
							return bad();
						}
						break;
					}
				}
				if (compute && (header & 2u) == 0 && !MecPlainOp(IT_DMA_DATA)) {
					return bad();
				}
				break;
			}
			// Packets both rings run. On MEC the shader-type bit is set.
			case IT_SET_BASE:
			case IT_DISPATCH_DIRECT:
			case IT_DISPATCH_INDIRECT:
			case IT_EVENT_WRITE:
			case IT_ACQUIRE_MEM:
			case IT_SET_SH_REG:
				if (compute && (header & 2u) == 0 && !MecPlainOp((header >> 8u) & 0xffu)) {
					return bad();
				}
				break;
			// libSceAgc's flag swap around a context push or pop (AgcDcbContextStateOp) and its workload swap and AND
			// (SetWorkloadsActive, SetWorkloadComplete), on the gfx ring.
			case IT_ATOMIC_MEM:
				if (compute || size != 9) {
					return bad();
				}
				// CP shadowing's display-DCC fix: an AND that clears DCC_ENABLE in the first table of a slot, as written.
				if (const uint64_t s = g_cp_shadow_va.load(std::memory_order_relaxed); s != 0 && b[0] == kCpShadowDccAnd) {
					const uint64_t a   = b[1] | (static_cast<uint64_t>(b[2]) << 32u);
					const uint64_t dcc = s + kCpShadowDccFixOffset;
					if (a >= dcc && a < dcc + 8u * 16u && (a - dcc) % 16u == 4u && b[3] == ~kCbDccEnable && b[4] == 0u && b[5] == 0u &&
					    b[6] == 0u) {
						break;
					}
					return bad();
				}
				if (!AgcAtomicOp(b[0]) && !IntegerAtomicOp(b[0])) { // libSceAgc's flag swap, workload swap or AND; the rest
					return bad();                                     // of the no-return ops with SetAtomicOps
				}
				break;
			// AgcDcbDraw[Index]IndirectMulti as written: the gfx ring only, ten dwords.
			case IT_DRAW_INDIRECT_MULTI:
			case IT_DRAW_INDEX_INDIRECT_MULTI:
				if (compute || size != 10) {
					return bad();
				}
				break;
			// CONTEXT_CONTROL only in the translator's forms: shadowing and loading off (every preamble, every job's
			// end), and while a shadow buffer is registered CP shadowing's enables and shadowing off around a GPU wait
			// with the load control untouched (kCpShadowWaitOff). The title's is dropped.
			case IT_CONTEXT_CONTROL: {
				const bool off    = b[0] == 0x80000000u && b[1] == 0x80000000u;
				const bool on_s   = g_cp_shadow_va.load(std::memory_order_relaxed) != 0;
				// The shadow preamble's enables, with every load on or (gated, and after a wait) the load control left as
				// amdgpu or the job set it.
				const bool shadow = on_s && (b[0] == kCpShadowEnables || b[0] == 0u) && b[1] == kCpShadowEnables;
				const bool wait   = on_s && b[0] == kCpShadowWaitOff[1] && b[1] == kCpShadowWaitOff[2];
				if (compute || header != Pkt3(IT_CONTEXT_CONTROL, 2, false) || (!off && !shadow && !wait)) {
					return bad();
				}
				break;
			}
			// CP shadowing's LOAD_*_REG (Pm4Translator::SetCpShadow): the gfx ring only, from the registered shadow
			// buffer's region of the class, runs of registers below 0x400, LOAD_SH_REG with the compute header only
			// for compute SH registers (0x200 up) and with the graphics one only below.
			case IT_LOAD_UCONFIG_REG:
			case IT_LOAD_CONTEXT_REG:
			case IT_LOAD_SH_REG: {
				const uint32_t op     = (header >> 8u) & 0xffu;
				const uint64_t s      = g_cp_shadow_va.load(std::memory_order_relaxed);
				const bool     cs     = (header & 2u) != 0;
				const uint32_t region = op == IT_LOAD_UCONFIG_REG ? kCpShadowUconfigDw
				                        : op == IT_LOAD_CONTEXT_REG ? kCpShadowContextDw
				                                                    : kCpShadowShDw;
				const uint64_t base   = s + 4ull * region;
				if (compute || s == 0 || size < 5 || (size & 1u) == 0 || (header & ~2u) != Pkt3(op, size - 1u, false) ||
				    (cs && op != IT_LOAD_SH_REG) || b[0] != static_cast<uint32_t>(base) || b[1] != static_cast<uint32_t>(base >> 32u)) {
					return bad();
				}
				for (uint32_t i = 2; i + 1 < size - 1u; i += 2) {
					const uint32_t off = b[i], count = b[i + 1];
					const uint32_t lo  = op != IT_LOAD_SH_REG ? 0u : (cs ? kShComputeBase : 0u);
					const uint32_t hi  = op != IT_LOAD_SH_REG ? RegFile::kDense : (cs ? RegFile::kDense : kShComputeBase);
					if (count == 0 || off < lo || off >= hi || count > hi - off) {
						return bad();
					}
				}
				break;
			}
			// Graphics-only. An illegal opcode on a compute ring.
			case IT_CLEAR_STATE:
			case IT_SET_PREDICATION:
			case IT_INDEX_BUFFER_SIZE:
			case IT_DRAW_INDIRECT:
			case IT_DRAW_INDEX_INDIRECT:
			case IT_INDEX_BASE:
			case IT_DRAW_INDEX_2:
			case IT_DRAW_INDEX_OFFSET_2: // as written with SetDrawIndexOffsetVerbatim
			case IT_DRAW_INDEX_AUTO:
			case IT_NUM_INSTANCES:
			case IT_PFP_SYNC_ME:
			case IT_SET_CONTEXT_REG:
			case IT_SET_UCONFIG_REG:
			case IT_SET_UCONFIG_REG_INDEX:
			case IT_SET_CONTEXT_REG_INDIRECT: // the guest's register tables, checked by the translator
			case IT_SET_SH_REG_INDIRECT:
			case IT_SET_UCONFIG_REG_INDIRECT:
				if (compute) {
					return bad();
				}
				break;
			default: return bad();
		}
		dw += size;
	}
	return true;
}

} // namespace

RingWaitRelease PlanRingWait(int cover, bool value_satisfies, bool gpu_write, bool same_ring, bool writer_submitted,
                             bool label_in_memory, bool native_wait, bool gfx_writer, bool shared_ring) {
	if (cover != 0) {
		// A write already emitted still owns the address. Matching memory is stale until it lands,
		// and an unknown value cannot be proved to satisfy the wait, so the segment stays held.
		// Across rings the GPU wait itself reads that stale memory: a label still holding the value
		// from its previous use passes it before the writer's ring has even run (ASTRO BOT: the gfx
		// ring then reset the label its compute ring was about to wait for, and the compute ring
		// waited forever). Such a wait goes to the GPU only while memory does not satisfy it yet;
		// otherwise it is held until the writer lands and becomes a barrier.
		const bool gpu_wait = cover == 1 && value_satisfies && gpu_write && native_wait &&
		                      (same_ring || (!shared_ring && (writer_submitted || gfx_writer) && !label_in_memory));
		return gpu_wait ? RingWaitRelease::GpuWait : RingWaitRelease::Hold;
	}
	if (label_in_memory) {
		return RingWaitRelease::Barrier;
	}
	return RingWaitRelease::Hold;
}

namespace {
// 1: inlined something, 0: nothing to inline, -1: cannot.
int FlattenInto(std::span<const uint32_t> in, std::vector<uint32_t>* out, int depth, std::string* why,
                std::vector<SourcePiece>* pieces, std::vector<uint32_t>* marks) {
	auto stop = [&](const char* what, size_t dw) {
		if (why != nullptr && why->empty()) {
			char text[160];
			std::snprintf(text, sizeof(text), "%s at dw %zu of %zu, depth %d, buffer %p", what, dw, in.size(), depth,
			              static_cast<const void*>(in.data()));
			*why = text;
		}
		return -1;
	};
	constexpr uint32_t kRewind = 0x59, kIndirect = 0x3f;
	constexpr size_t   kMaxDwords = 4u << 20u;
	int                changed    = 0;
	uint32_t           hops       = 0; // chains taken: a chain replaces the buffer, it does not nest
	// A COND_EXEC skips EXEC_COUNT dwords of the buffer it is in, as the CP fetches them. An inlined call inside the
	// range brings the target's dwords instead of its 4 (or 14), a dropped REWIND takes 2 out, so the count in `out`
	// moves by as much. A chain inside a range, or a range past the end of a called buffer, has no such count: stop.
	struct CondRange {
		size_t count_at; // the EXEC_COUNT dword in `out`
		size_t end;      // the range's end, a dword of `in`
	};
	std::vector<CondRange> conds;
	// `len`: the dwords of the call or REWIND at `dw`. A range that ends inside one skips part of the packet on the PS5,
	// and the CP parses the rest of it as packets; no count over the inlined dwords gives that, so it stops.
	auto cond_delta = [&](size_t dw, uint32_t len, int64_t delta) {
		for (const CondRange& c: conds) {
			if (dw < c.end && c.end < dw + len) {
				return false;
			}
			if (dw < c.end) {
				const int64_t count = static_cast<int64_t>((*out)[c.count_at] & 0x3fffu) + delta;
				if (count < 0 || count > 0x3fff || ((*out)[c.count_at] & ~0x3fffu) != 0) { // see the translator's COND_EXEC
					return false;
				}
				(*out)[c.count_at] = ((*out)[c.count_at] & ~0x3fffu) | static_cast<uint32_t>(count);
			}
		}
		return true;
	};
	// Packets go to `out` a run at a time, up to the next REWIND or INDIRECT_BUFFER; a top-level buffer
	// without either is never copied (nothing to inline).
	size_t run   = 0;
	auto   flush = [&](size_t to) {
		if (to == run) {
			return true;
		}
		const auto len = static_cast<uint32_t>(to - run);
		if (out->size() + len > kMaxDwords) {
			return false;
		}
		if (pieces != nullptr) {
			const uint64_t va = reinterpret_cast<uint64_t>(in.data() + run);
			const auto     at = static_cast<uint32_t>(out->size());
			if (!pieces->empty() && pieces->back().dw + pieces->back().len == at &&
			    pieces->back().va + 4ull * pieces->back().len == va) {
				pieces->back().len += len;
			} else {
				pieces->push_back({at, len, va});
			}
		}
		out->insert(out->end(), in.begin() + static_cast<std::ptrdiff_t>(run), in.begin() + static_cast<std::ptrdiff_t>(to));
		run = to;
		return true;
	};
	for (size_t dw = 0; dw < in.size();) {
		const uint32_t header = in[dw];
		const uint32_t len    = (header >> 30u) == 3u ? PacketSizeDw(header) : 1u;
		if (dw + len > in.size()) {
			return stop("packet past the end", dw);
		}
		const uint32_t op = (header >> 8u) & 0xffu;
		conds.erase(std::remove_if(conds.begin(), conds.end(), [&](const CondRange& c) { return dw >= c.end; }), conds.end());
		if ((header >> 30u) == 3u && op == IT_COND_EXEC && len == 5) {
			conds.push_back({out->size() + (dw - run) + 4u, dw + len + (in[dw + 4] & 0x3fffu)});
		}
		if ((header >> 30u) == 3u && op == kRewind && len == 2) {
			if ((in[dw + 1] & 0x80000000u) == 0) {
				return stop("REWIND not valid yet", dw);
			}
			if (!flush(dw)) {
				return stop("too large", dw);
			}
			if (!cond_delta(dw, len, -2)) {
				return stop("COND_EXEC count out of range or its range ending inside the packet", dw);
			}
			changed = 1;
			dw += len;
			run = dw;
			continue;
		}
		if ((header >> 30u) == 3u && op == kIndirect) {
			// libSceAgc's branch (AgcCbBranch, 14 dwords: COND_INDIRECT_BUFFER) with compare function 0 is
			// always taken: a call of its first target, the way Stray calls nested buffers. The BC-250's CP
			// takes the branch but does not come back from it (probe --danger condib), so it is inlined like
			// a call. A branch on memory still refuses the buffer.
			const bool always_branch = len == 14 && ((in[dw + 1] >> 8u) & 0x7u) == 0u && (in[dw + 1] & 0x3u) != 0u;
			if (len != 4 && !always_branch) {
				return stop("INDIRECT_BUFFER of another form", dw);
			}
			if (!flush(dw)) {
				return stop("too large", dw);
			}
			if (marks != nullptr && depth == 0) {
				marks->push_back(static_cast<uint32_t>(out->size()));
			}
			const uint32_t  base_dw = always_branch ? dw + 7u : dw; // the then target: address at dwords 8-9, size at 10
			const uint32_t  control = in[base_dw + 3];
			const auto*     target  = reinterpret_cast<const uint32_t*>((static_cast<uint64_t>(in[base_dw + 2]) << 32u) | (in[base_dw + 1] & ~3u));
			const uint32_t  size    = control & 0xfffffu;
			const bool      chain   = !always_branch && (control & (1u << 20u)) != 0;
			changed                = 1;
			if (size == 0) {
				if (chain && !conds.empty()) {
					return stop("chain inside a COND_EXEC", dw);
				}
				if (chain) {
					return changed;
				}
				if (!cond_delta(dw, len, -static_cast<int64_t>(len))) { // an empty call is left out
					return stop("COND_EXEC count out of range or its range ending inside the packet", dw);
				}
				dw += len;
				run = dw;
				continue;
			}
			if (target == nullptr || out->size() + size > kMaxDwords) {
				return stop("IB target null or too large", dw);
			}
			if (chain) {
				if (++hops > 65536u) {
					return stop("chain too long", dw);
				}
				if (!conds.empty()) {
					return stop("chain inside a COND_EXEC", dw);
				}
				in  = {target, size};
				dw  = 0;
				run = 0;
				continue;
			}
			if (depth >= 16) {
				return stop("calls nested too deep", dw);
			}
			const size_t before = out->size();
			if (FlattenInto({target, size}, out, depth + 1, why, pieces, nullptr) < 0) {
				return -1;
			}
			if (!cond_delta(dw, len, static_cast<int64_t>(out->size() - before) - static_cast<int64_t>(len))) {
				return stop("COND_EXEC count out of range or its range ending inside the packet", dw);
			}
			if (marks != nullptr && depth == 0) {
				marks->push_back(static_cast<uint32_t>(out->size()));
			}
			dw += len;
			run = dw;
			continue;
		}
		dw += len;
	}
	if ((changed != 0 || depth > 0) && !flush(in.size())) {
		return stop("too large", in.size());
	}
	for (const CondRange& c: conds) {
		if (depth > 0 && c.end > in.size()) {
			return stop("COND_EXEC past the end of a called buffer", in.size());
		}
	}
	return changed;
}
} // namespace

bool FlattenIndirectBuffers(std::span<const uint32_t> in, std::vector<uint32_t>* out, std::string* why,
                            std::vector<SourcePiece>* pieces, std::vector<uint32_t>* marks) {
	out->clear();
	if (pieces != nullptr) {
		pieces->clear();
	}
	if (marks != nullptr) {
		marks->clear();
	}
	if (FlattenInto(in, out, 0, why, pieces, marks) <= 0) {
		out->clear();
		if (pieces != nullptr) {
			pieces->clear();
		}
		return false;
	}
	return true;
}

void SetTessellationRings(uint64_t tf_ring_base, uint32_t tf_ring_bytes, uint32_t hs_offchip_param) {
	g_tf_ring_base.store(tf_ring_base, std::memory_order_relaxed);
	g_hs_offchip_param.store(hs_offchip_param, std::memory_order_relaxed);
	g_tf_ring_bytes.store(tf_ring_bytes, std::memory_order_release);
}

void SetCpShadowValidation(uint64_t va) {
	g_cp_shadow_va.store(va, std::memory_order_relaxed);
}

void EmitCpShadowOff(std::vector<uint32_t>& ib) {
	ib.insert(ib.end(), kCpShadowOff.begin(), kCpShadowOff.end());
}

bool CpShadowTailOk(std::span<const uint32_t> ib) {
	size_t n = ib.size();
	while (n > 0 && ib[n - 1] == kFiller) {
		n--;
	}
	if (n >= 3 && ib[n - 3] == kCpShadowOff[0] && ib[n - 2] == kCpShadowOff[1] && ib[n - 1] == kCpShadowOff[2]) {
		return true;
	}
	// A chain: the CP leaves this IB for good. An exit guard's chain runs only once a stop is armed, so behind
	// its COND_EXEC the IB would go on.
	return n >= 4 && ib[n - 4] == Pkt3(IT_INDIRECT_BUFFER, 3, false) && (ib[n - 1] & (1u << 20u)) != 0 &&
	       !(n >= 9 && ib[n - 9] == Pkt3(IT_COND_EXEC, 4, false));
}

void CpShadowBatch::Append(std::vector<uint32_t>& batch, std::span<const uint32_t> ib, const CpShadowPiece& piece) {
	if (!piece.own) {
		// At the front the batch may carry on a stream with shadowing on: turn it off there too.
		if (!m_any || m_on) {
			EmitCpShadowOff(batch);
			guards++;
		}
		m_on = false;
	} else if (!piece.head) {
		if (!m_any) {
			m_front.assign(piece.resume.begin(), piece.resume.end());
		} else if (!m_on) {
			batch.insert(batch.end(), piece.resume.begin(), piece.resume.end());
			resumes++;
		}
		m_on = true;
	} else {
		m_on = true; // its preamble turns shadowing off, then on with every base set
	}
	m_any = true;
	batch.insert(batch.end(), ib.begin(), ib.end());
}

size_t CpShadowBatch::Extra(const CpShadowPiece& piece) const {
	const size_t kept = m_front.size();
	if (!piece.own) {
		return kept + kCpShadowOff.size();
	}
	return kept + (piece.head ? 0u : piece.resume.size());
}

std::span<const uint32_t> CpShadowBatch::Front(bool stream_part, bool stream_on) const {
	if (!m_any || m_front.empty() || (stream_part && stream_on)) {
		return {};
	}
	return m_front;
}

void CpShadowBatch::Reset() {
	m_any = false;
	m_on  = false;
	m_front.clear();
}

std::vector<std::pair<size_t, size_t>> SplitIbAtPackets(std::span<const uint32_t> ib, size_t max_dw) {
	std::vector<std::pair<size_t, size_t>> pieces;
	size_t                                 start = 0;
	size_t                                 dw    = 0;
	while (dw < ib.size()) {
		const uint32_t h = ib[dw];
		// A type-3 header with count 0x3fff is the one-dword filler NOP; type-2 packets are one dword.
		size_t len = (h >> 30u) == 3u && h != 0xffff1000u ? ((h >> 16u) & 0x3fffu) + 2u : 1u;
		// A COND_EXEC with the dwords it may skip is one piece: the skip must not reach past an IB's end.
		if (len == 5u && ((h >> 8u) & 0xffu) == IT_COND_EXEC && dw + 4u < ib.size()) {
			len += ib[dw + 4u];
		}
		if (len > max_dw) {
			return {};
		}
		if (dw + len - start > max_dw) {
			pieces.emplace_back(start, dw - start);
			start = dw;
		}
		dw += len;
	}
	if (dw > start) {
		pieces.emplace_back(start, std::min(dw, ib.size()) - start);
	}
	return pieces;
}

uint32_t ComputeRingForQueue(uint32_t guest_queue, uint32_t available_rings) {
	uint32_t rings[32];
	uint32_t n = 0;
	for (uint32_t bit = 0; bit < 32; bit++) {
		if ((available_rings & (1u << bit)) != 0) {
			rings[n++] = bit;
		}
	}
	if (n == 0) {
		return 0;
	}
	const uint32_t index = guest_queue >= 0x20u ? guest_queue - 0x20u : guest_queue;
	return rings[index % n];
}

uint32_t Pm4Translator::PatchContext(uint32_t offset, uint32_t value) const {
	// CB_COLORn_INFO.DCC_ENABLE (bit 28) off, for the no-DCC experiment.
	if (m_no_dcc && offset >= kCbColor0Info && offset < kCbColor0Info + 8 * kCbColorStride &&
	    (offset - kCbColor0Info) % kCbColorStride == 0) {
		value &= ~(1u << 28u);
	}
	// CB_COLORn_INFO.FAST_CLEAR (bit 13) off on targets without DCC: CMASK-only fast clear, for the
	// no-CMASK-clear experiment.
	if (m_no_cmask_clear && offset >= kCbColor0Info && offset < kCbColor0Info + 8 * kCbColorStride &&
	    (offset - kCbColor0Info) % kCbColorStride == 0 && (value & (1u << 28u)) == 0 &&
	    ((m_no_cmask_clear_slots >> ((offset - kCbColor0Info) / kCbColorStride)) & 1u) != 0) {
		value &= ~(1u << 13u);
	}
	if (offset == kCbTargetMask) {
		value &= m_target_mask;
	}
	// The PS5 kernel programs the CP's register-override CAM (RCIU) so PA_SC_SHADER_CONTROL always has
	// LOAD_COLLISION_WAVEID (bit 2) set, whatever the game writes.
	if (m_ps5_reg_overrides && offset == kPaScShaderControl) {
		value |= 1u << 2u;
	}
	switch (m_depth_mode) {
		case DepthDebugMode::None: return value;
		case DepthDebugMode::HizOff:
			// DB_RENDER_OVERRIDE FORCE_HIZ_ENABLE, FORCE_HIS_ENABLE0/1 = FORCE_DISABLE (2).
			return offset == kDbRenderOverride ? ((value & ~0x3fu) | 0x2au) : value;
		case DepthDebugMode::NoHtile:
			// DB_Z_INFO / DB_STENCIL_INFO TILE_SURFACE_ENABLE (bit 29) off: depth without HTILE.
			return (offset == kDbZInfo || offset == kDbStencilInfo) ? (value & ~(1u << 29u)) : value;
	}
	return value;
}

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
		case 0x248: // VGT_OBJECT_ID
		case 0x383: // TEXTURE_GRADIENT_FACTORS
		case 0x384: // TEXTURE_GRADIENT_CONTROL
		{
			// What AgcCreatePrimState tables carry besides the GE registers (My First GT: in nearly every
			// one); left out before, so the GPU never had them. KYTY_NATIVE_GPU_TABLES_PRIM=0 leaves them out.
			static const bool extra = [] {
				const char* v = std::getenv("KYTY_NATIVE_GPU_TABLES_PRIM");
				return v == nullptr || v[0] != '0';
			}();
			return extra;
		}
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

void Pm4Translator::SeedContextDefaults(std::span<const std::pair<uint32_t, uint32_t>> defaults) {
	for (const auto& [offset, value]: defaults) {
		m_cx.emplace(offset, value);
		m_cx_defaults.emplace(offset, value);
	}
}

uint32_t Pm4Translator::LoadStartState(std::span<const std::pair<uint32_t, uint32_t>> context,
                                       std::span<const std::pair<uint32_t, uint32_t>> sh,
                                       std::span<const std::pair<uint32_t, uint32_t>> uconfig) {
	for (const auto& [offset, value]: context) {
		m_cx[offset]          = value;
		m_cx_defaults[offset] = value;
	}
	for (const auto& [offset, value]: sh) {
		m_sh[offset] = value;
	}
	uint32_t skipped = 0;
	for (const auto& [offset, value]: uconfig) {
		if (UconfigAllowed(offset)) {
			m_uc[offset] = value;
		} else {
			skipped++;
		}
	}
	return skipped;
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
		m_cx_defaults.emplace(d.offset, d.value);
	}
}

void Pm4Translator::ResetShadow() {
	m_cx.clear();
	m_sh.clear();
	m_uc.clear();
	m_cp = {};
}

void Pm4Translator::EmitPreamble(std::vector<uint32_t>& ib, Pm4TranslateStats& stats, const RegFile& cx,
                                 const RegFile& sh, const RegFile& uc, const Pm4CpDrawState& cp) const {
	if (m_compute_ring) {
		// MEC has no graphics context. CONTEXT_CONTROL, CLEAR_STATE, context and UCONFIG writes are
		// illegal opcodes on a compute ring. SH state is replayed: other jobs share the ring.
		for (const auto& [offset, value]: sh) {
			EmitSet(ib, IT_SET_SH_REG, offset, PatchSh(offset, value), true);
			stats.preamble_sh++;
		}
		return;
	}
	// RADV-style preamble: take back context ownership, clear to defaults, then replay the whole
	// shadowed state. Other processes share the gfx ring between our submissions, so nothing from
	// the previous DCB is assumed to survive on the hardware.
	ib.push_back(Pkt3(IT_CONTEXT_CONTROL, 2, false));
	ib.push_back(0x80000000u); // update load enables, load nothing (we set everything below)
	ib.push_back(0x80000000u); // update shadow enables, shadow nothing

	ib.push_back(Pkt3(IT_CLEAR_STATE, 1, false));
	ib.push_back(0u);

	EmitPreambleState(ib, uc, cp);
	EmitRegisterSets(ib, stats, cx, sh, uc, kCpShadowClasses);
}

void Pm4Translator::EmitShadowPreamble(std::vector<uint32_t>& ib, Pm4TranslateStats& stats, const RegFile& cx,
                                       const RegFile& sh, const RegFile& uc, const Pm4CpDrawState& cp) const {
	// As every preamble: shadowing and loading off (so the clear and nothing before the LOADs reaches S), the
	// context back to amdgpu's clear state. Then, as Mesa's shadowing preamble (ac_create_shadowing_ib_preamble):
	// PFP_SYNC_ME so the shadow writes before it have landed when the LOADs read S, the enables, and the LOADs of
	// all four classes back to back before any SET, so every class has its base in this job.
	//
	// Gated (SetCpShadowGatedLoads, load mode): as the PS5 reloads a queue's registers only when the CP switches to it from
	// another one, the LOADs load only after another client's job ran on the ring. amdgpu's CONTEXT_CONTROL in front of
	// every job turns the loads on after another context's job and leaves them off after our own
	// (gfx_v10_0_ring_emit_cntxcntl, AMDGPU_HAVE_CTX_SWITCH), when the CP still holds our state, S's. So no CLEAR_STATE,
	// the load control stays amdgpu's (bit 31 clear: not updated) and only shadowing goes on; S holds the clear state of
	// every register a CLEAR_STATE resets, and the LOADs take all of those, which gives what the clear and the LOADs gave.
	// Before 20 such preambles a frame (Demon's Souls) took a third of its frame rate.
	const bool gated = m_cp_shadow_mode == CpShadowMode::Load && m_cp_shadow_gated && !m_cp_shadow_force_load;
	if (gated) {
		ib.push_back(Pkt3(IT_PFP_SYNC_ME, 1, false));
		ib.push_back(0u);
		ib.push_back(Pkt3(IT_CONTEXT_CONTROL, 2, false));
		ib.push_back(0u);
		ib.push_back(kCpShadowEnables);
		stats.cp_shadow_gated++;
	} else {
		ib.push_back(Pkt3(IT_CONTEXT_CONTROL, 2, false));
		ib.push_back(0x80000000u);
		ib.push_back(0x80000000u);
		ib.push_back(Pkt3(IT_CLEAR_STATE, 1, false));
		ib.push_back(0u);
		ib.push_back(Pkt3(IT_PFP_SYNC_ME, 1, false));
		ib.push_back(0u);
		ib.push_back(Pkt3(IT_CONTEXT_CONTROL, 2, false));
		ib.push_back(kCpShadowEnables);
		ib.push_back(kCpShadowEnables);
	}
	const uint64_t s = m_cp_shadow_va;
	// The probe's order: UCONFIG, context, graphics SH, compute SH (the compute header, into the SH region).
	// Anchors: GE_INDX_OFFSET, PA_SC_WINDOW_OFFSET, SPI_SHADER_USER_DATA_PS_0, COMPUTE_USER_DATA_0.
	EmitCpShadowLoad(ib, stats, IT_LOAD_UCONFIG_REG, false, s + 4ull * kCpShadowUconfigDw, uc, 0, RegFile::kDense, CpShadowLoadsUconfig,
	                 0x24Au);
	EmitCpShadowLoad(ib, stats, IT_LOAD_CONTEXT_REG, false, s + 4ull * kCpShadowContextDw, cx, 0, RegFile::kDense, CpShadowLoadsContext,
	                 0x080u, gated ? InClearState : nullptr);
	EmitCpShadowLoad(ib, stats, IT_LOAD_SH_REG, false, s + 4ull * kCpShadowShDw, sh, 0, kShComputeBase, CpShadowLoadsSh, 0x00Cu);
	EmitCpShadowLoad(ib, stats, IT_LOAD_SH_REG, true, s + 4ull * kCpShadowShDw, sh, kShComputeBase, RegFile::kDense, CpShadowLoadsSh,
	                 0x240u);
	stats.cp_shadow_preambles++;
	EmitPreambleState(ib, uc, cp);
	// Record: the copy's replay as without shadowing (now shadowed into S as well). Load: what S does not restore,
	// and the classes left out of the LOAD set.
	EmitRegisterSets(ib, stats, cx, sh, uc,
	                 m_cp_shadow_mode == CpShadowMode::Record ? kCpShadowClasses : (~m_cp_shadow_classes & kCpShadowClasses));
}

std::array<uint32_t, kCpShadowFixBytes / 4u> CpShadowFixTables() {
	std::array<uint32_t, kCpShadowFixBytes / 4u> t {};
	for (uint32_t n = 0; n < 8; n++) {
		uint32_t* e = t.data() + 8u * n;
		e[0] = e[4] = kCbColor0Fmask + n * kCbColorStride;
		e[2] = e[6] = kCbColor0FmaskExt + n;
		uint32_t* d = t.data() + (kCpShadowDccFixOffset - kCpShadowFixOffset) / 4u + 4u * n;
		d[0] = d[2] = kCbColor0Info + n * kCbColorStride;
	}
	return t;
}

// The display-DCC fix on the GPU (load mode): where the copy has CB_COLORn_INFO from a verbatim table the CP may hold
// another INFO, so the CPU cannot write one without DCC; the GPU clears DCC_ENABLE in its own. The CP's INFO from S
// into the value dwords of the slot's two tables once every shadow write before it has landed (PFP_SYNC_ME; DMA_DATA by
// the ME with CP_SYNC, as the FMASK fix), ATOMIC_MEM AND on the first, a PFP wait until that word has lost DCC_ENABLE,
// then the first table sets INFO. The CP shadows that write into S. The second keeps the INFO it had.
void Pm4Translator::EmitCpShadowDccOff(std::vector<uint32_t>& ib, uint32_t slot) const {
	const uint64_t from = m_cp_shadow_va + 4ull * (kCpShadowContextDw + kCbColor0Info + slot * kCbColorStride);
	const uint64_t t    = m_cp_shadow_va + kCpShadowDccFixOffset + 16ull * slot;
	ib.push_back(Pkt3(IT_PFP_SYNC_ME, 1, false));
	ib.push_back(0u);
	for (const uint64_t to: {t + 4u, t + 12u}) {
		ib.insert(ib.end(), {Pkt3(IT_DMA_DATA, 6, false), kCpShadowDmaControl, static_cast<uint32_t>(from), static_cast<uint32_t>(from >> 32u),
		                     static_cast<uint32_t>(to), static_cast<uint32_t>(to >> 32u), kCpShadowDmaWordCommand});
	}
	ib.insert(ib.end(), {Pkt3(IT_ATOMIC_MEM, 8, false), kCpShadowDccAnd, static_cast<uint32_t>(t + 4u), static_cast<uint32_t>((t + 4u) >> 32u),
	                     ~kCbDccEnable, 0u, 0u, 0u, 6u});
	// The ME sends the atomic and goes on: the PFP, which reads the table, waits until the word has lost DCC_ENABLE
	// (with PFP_SYNC_ME alone it read the INFO from before the AND, and GT7's video-out buffers kept their DCC).
	ib.insert(ib.end(), {Pkt3(IT_WAIT_REG_MEM, 6, false), 3u | (1u << 4u) | (1u << 8u), static_cast<uint32_t>(t + 4u),
	                     static_cast<uint32_t>((t + 4u) >> 32u), 0u, kCbDccEnable, 4u});
	ib.insert(ib.end(), {Pkt3(IT_SET_CONTEXT_REG_INDIRECT, 4, false), static_cast<uint32_t>(t), static_cast<uint32_t>(t >> 32u), 0x80000000u, 1u});
}

// The slot left the video-out buffer with the INFO the GPU cleared DCC in and not written since: the INFO the CP had.
void Pm4Translator::EmitCpShadowDccRestore(std::vector<uint32_t>& ib, uint32_t slot) const {
	const uint64_t t = m_cp_shadow_va + kCpShadowDccFixOffset + 16ull * slot + 8u;
	ib.insert(ib.end(), {Pkt3(IT_SET_CONTEXT_REG_INDIRECT, 4, false), static_cast<uint32_t>(t), static_cast<uint32_t>(t >> 32u), 0x80000000u, 1u});
}

// fix_fmask on the GPU (load mode), as the CPU does it on the copy (FMASK = BASE when the target has none) but on the
// values the CP holds: its CB_COLORn_BASE, BASE_EXT, FMASK and FMASK_EXT from S into the value dwords of the slot's
// two one-register-pair tables, once every shadow write before them has landed (PFP_SYNC_ME; DMA_DATA by the ME
// with CP_SYNC and RAW_WAIT, as the push), then FMASK = BASE from the first table, and, with `check`, when the CP's
// FMASK was not 0, the second table puts FMASK and FMASK_EXT back (COND_EXEC on its copied FMASK). The PFP reads
// the COND_EXEC's word, hence the PFP_SYNC_ME after the copies. A register S has no value for is 0 on the GPU
// (amdgpu's clear state): the zero word is copied in its place. The CP shadows the tables' writes into S.
void Pm4Translator::EmitCpShadowFmaskFix(std::vector<uint32_t>& ib, uint32_t slot, bool check,
                                         const std::bitset<RegFile::kDense>& valid) const {
	const uint64_t cx   = m_cp_shadow_va + 4ull * kCpShadowContextDw;
	const uint64_t zero = m_cp_shadow_va + kCpShadowZeroOffset;
	const uint64_t t    = m_cp_shadow_va + kCpShadowFixOffset + 32ull * slot;
	auto           copy = [&](uint32_t reg, uint64_t to) {
		const uint64_t from = valid.test(reg) ? cx + 4ull * reg : zero;
		ib.insert(ib.end(), {Pkt3(IT_DMA_DATA, 6, false), kCpShadowDmaControl, static_cast<uint32_t>(from), static_cast<uint32_t>(from >> 32u),
		                     static_cast<uint32_t>(to), static_cast<uint32_t>(to >> 32u), kCpShadowDmaWordCommand});
	};
	auto table = [&](uint64_t at) {
		ib.insert(ib.end(), {Pkt3(IT_SET_CONTEXT_REG_INDIRECT, 4, false), static_cast<uint32_t>(at), static_cast<uint32_t>(at >> 32u),
		                     0x80000000u, 2u});
	};
	ib.push_back(Pkt3(IT_PFP_SYNC_ME, 1, false));
	ib.push_back(0u);
	copy(kCbColor0Base + slot * kCbColorStride, t + 4u);
	copy(kCbColor0BaseExt + slot, t + 12u);
	if (check) {
		copy(kCbColor0Fmask + slot * kCbColorStride, t + 20u);
		copy(kCbColor0FmaskExt + slot, t + 28u);
	}
	ib.push_back(Pkt3(IT_PFP_SYNC_ME, 1, false));
	ib.push_back(0u);
	table(t);
	if (check) {
		ib.insert(ib.end(), {Pkt3(IT_COND_EXEC, 4, false), static_cast<uint32_t>(t + 20u), static_cast<uint32_t>((t + 20u) >> 32u), 0u, 5u});
		table(t + 16u);
	}
}

// libSceAgc's context push on the GPU (SetCpShadowPushPop): S's context region to P once every shadow write before
// it has landed (PFP_SYNC_ME, and RAW_WAIT for the DMA's own reads), CP_SYNC so nothing after it runs before the copy
// is done. Shadowing stays on: S goes on following the inner state.
void Pm4Translator::EmitCpShadowPush(std::vector<uint32_t>& ib) const {
	const uint64_t cx = m_cp_shadow_va + 4ull * kCpShadowContextDw;
	const uint64_t p  = m_cp_shadow_va + kCpShadowSaveOffset;
	ib.push_back(Pkt3(IT_PFP_SYNC_ME, 1, false));
	ib.push_back(0u);
	ib.insert(ib.end(), {Pkt3(IT_DMA_DATA, 6, false), kCpShadowDmaControl, static_cast<uint32_t>(cx), static_cast<uint32_t>(cx >> 32u),
	                     static_cast<uint32_t>(p), static_cast<uint32_t>(p >> 32u), kCpShadowDmaCommand});
}

// The pop: P back into S's context region, PFP_SYNC_ME so the PFP's LOAD reads it only after the ME has finished
// the copy (CP_SYNC), the enables (loading context on), LOAD_CONTEXT_REG of every context register `valid` in S
// (all of 0x000-0x3FF with the whole range), then what a LOAD does not restore, from `target` as a shadow
// preamble SETs it (PA_SC_SHADER_CONTROL, DB_RENDER_OVERRIDE for HizOff, offsets from 0x400), and 0 for such
// registers `current` holds and `target` does not (the CPU pop's rule).
void Pm4Translator::EmitCpShadowPop(std::vector<uint32_t>& ib, const RegFile& target, const RegFile& current,
                                    const std::bitset<RegFile::kDense>& valid) const {
	const uint64_t cx = m_cp_shadow_va + 4ull * kCpShadowContextDw;
	const uint64_t p  = m_cp_shadow_va + kCpShadowSaveOffset;
	ib.push_back(Pkt3(IT_PFP_SYNC_ME, 1, false));
	ib.push_back(0u);
	ib.insert(ib.end(), {Pkt3(IT_DMA_DATA, 6, false), kCpShadowDmaControl, static_cast<uint32_t>(p), static_cast<uint32_t>(p >> 32u),
	                     static_cast<uint32_t>(cx), static_cast<uint32_t>(cx >> 32u), kCpShadowDmaCommand});
	ib.push_back(Pkt3(IT_PFP_SYNC_ME, 1, false));
	ib.push_back(0u);
	ib.push_back(Pkt3(IT_CONTEXT_CONTROL, 2, false));
	ib.push_back(kCpShadowEnables);
	ib.push_back(kCpShadowEnables);
	const size_t at = ib.size();
	ib.push_back(0u); // the header, once the length is known
	ib.push_back(static_cast<uint32_t>(cx));
	ib.push_back(static_cast<uint32_t>(cx >> 32u));
	if (m_cp_shadow_pop_all) {
		ib.push_back(0u);
		ib.push_back(RegFile::kDense);
	} else {
		for (uint32_t off = 0; off < RegFile::kDense;) {
			if (!valid.test(off) || !CpShadowLoadsContext(off)) {
				off++;
				continue;
			}
			uint32_t end = off + 1;
			while (end < RegFile::kDense && valid.test(end) && CpShadowLoadsContext(end)) {
				end++;
			}
			ib.push_back(off);
			ib.push_back(end - off);
			off = end;
		}
		if (ib.size() == at + 3) {
			ib.push_back(0x080u); // as the preamble's anchor: a LOAD needs a range
			ib.push_back(1u);
		}
	}
	ib[at] = Pkt3(IT_LOAD_CONTEXT_REG, static_cast<uint32_t>(ib.size() - at - 1), false);
	for (const auto& [off, value]: target) {
		if (!CpShadowLoadsContext(off)) {
			EmitSet(ib, IT_SET_CONTEXT_REG, off, PatchContext(off, value), false);
		}
	}
	for (const auto& [off, value]: current) {
		if (!CpShadowLoadsContext(off) && target.count(off) == 0 && value != 0) {
			EmitSet(ib, IT_SET_CONTEXT_REG, off, PatchContext(off, 0), false);
		}
	}
	if (m_ps5_reg_overrides && target.count(kPaScShaderControl) == 0) {
		EmitSet(ib, IT_SET_CONTEXT_REG, kPaScShaderControl, PatchContext(kPaScShaderControl, 0), false);
	}
	if (m_depth_mode == DepthDebugMode::HizOff && target.count(kDbRenderOverride) == 0) {
		EmitSet(ib, IT_SET_CONTEXT_REG, kDbRenderOverride, PatchContext(kDbRenderOverride, 0), false);
	}
}

void Pm4Translator::EmitPreambleState(std::vector<uint32_t>& ib, const RegFile& uc, const Pm4CpDrawState& cp) const {
	// The title's tessellation rings (SetTessellationRings), in the *_UMD registers user mode drivers own.
	if (const uint32_t tf_bytes = g_tf_ring_bytes.load(std::memory_order_acquire); tf_bytes != 0) {
		const uint64_t tf_base = g_tf_ring_base.load(std::memory_order_relaxed);
		EmitSetUconfig(ib, 0x24e, (tf_bytes / 4u) & 0xffffu);                           // VGT_TF_RING_SIZE_UMD
		EmitSetUconfig(ib, 0x24f, g_hs_offchip_param.load(std::memory_order_relaxed)); // VGT_HS_OFFCHIP_PARAM_UMD
		EmitSetUconfig(ib, 0x250, static_cast<uint32_t>(tf_base >> 8u));               // VGT_TF_MEMORY_BASE_UMD
		EmitSetUconfig(ib, 0x261, static_cast<uint32_t>(tf_base >> 40u) & 0xffu);      // VGT_TF_MEMORY_BASE_HI_UMD
	}

	// libSceAgc puts PIXEL_PIPE_STAT_CONTROL (event 0x38, EVENT_INDEX 1) into the context start state
	// once; the title then only sends PIXEL_PIPE_STAT_DUMP (0x39). The control word enables all 16 RB
	// instances, and without it the DB writes nothing for a dump. KYTY_NATIVE_GPU_PIXEL_PIPE_STAT=0 skips it.
	if (static const bool pixel_pipe_stat = [] {
		    const char* e = std::getenv("KYTY_NATIVE_GPU_PIXEL_PIPE_STAT");
		    return e == nullptr || e[0] != '0';
	    }();
	    pixel_pipe_stat) {
		ib.push_back(Pkt3(IT_EVENT_WRITE, 3, false));
		ib.push_back(0x38u | (1u << 8u));
		ib.push_back(0x07fffc00u);
		ib.push_back(0u);
	}
	// CP draw state as the title left it (libSceAgc's start values before its first DCB). Other
	// processes draw on this ring between our submissions and leave theirs behind.
	// KYTY_NATIVE_GPU_CP_STATE: mask of what is replayed, 1 index type, 2 NUM_INSTANCES, 4 INDEX_BASE and
	// INDEX_BUFFER_SIZE once the title has sent them, 0x10 also libSceAgc's start values of those two
	// (base 0, size 0xFFFFFFFF), 8 SET_BASE (0 leaves it to the title). 0x10 is off by default: Dead Cells
	// never binds an index buffer (it draws with DRAW_INDEX_2), and with base 0 / size 0xFFFFFFFF at the
	// start of every IB the gfx ring hung at its level load in 4 of 6 runs (0 of 7 without).
	static const uint32_t cp_state = [] {
		const char* e = std::getenv("KYTY_NATIVE_GPU_CP_STATE");
		return e == nullptr ? 0xfu : static_cast<uint32_t>(std::strtoul(e, nullptr, 0));
	}();
	if ((cp_state & 1u) != 0 && uc.count(0x243) == 0) {
		EmitSetUconfig(ib, 0x243, 0x480u); // VGT_INDEX_TYPE: 16-bit, cache policy 2
	}
	if ((cp_state & 2u) != 0) {
		ib.push_back(Pkt3(IT_NUM_INSTANCES, 1, false));
		ib.push_back(cp.num_instances);
	}
	if ((cp_state & 4u) != 0 && (cp.index_set || (cp_state & 0x10u) != 0)) {
		ib.push_back(Pkt3(IT_INDEX_BASE, 2, false));
		ib.push_back(static_cast<uint32_t>(cp.index_base));
		ib.push_back(static_cast<uint32_t>(cp.index_base >> 32u));
		ib.push_back(Pkt3(IT_INDEX_BUFFER_SIZE, 1, false));
		ib.push_back(cp.index_size);
	}
	if ((cp_state & 8u) != 0) {
		ib.push_back(Pkt3(IT_SET_BASE, 3, false));
		ib.push_back(1u);
		ib.push_back(static_cast<uint32_t>(cp.draw_base));
		ib.push_back(static_cast<uint32_t>(cp.draw_base >> 32u));
	}
}

void Pm4Translator::EmitRegisterSets(std::vector<uint32_t>& ib, Pm4TranslateStats& stats, const RegFile& cx, const RegFile& sh,
                                     const RegFile& uc, uint32_t replay) const {
	for (const auto& [offset, value]: cx) {
		if ((replay & kCpShadowContext) != 0 || !CpShadowLoadsContext(offset)) {
			EmitSet(ib, IT_SET_CONTEXT_REG, offset, PatchContext(offset, value), false);
			stats.preamble_cx++;
		}
	}
	if (m_ps5_reg_overrides && cx.count(kPaScShaderControl) == 0) {
		EmitSet(ib, IT_SET_CONTEXT_REG, kPaScShaderControl, PatchContext(kPaScShaderControl, 0), false);
	}
	if (m_depth_mode == DepthDebugMode::HizOff && cx.count(kDbRenderOverride) == 0) {
		EmitSet(ib, IT_SET_CONTEXT_REG, kDbRenderOverride, PatchContext(kDbRenderOverride, 0), false);
	}
	for (const auto& [offset, value]: sh) {
		const uint32_t cls = offset >= kShComputeBase ? kCpShadowCsSh : kCpShadowGfxSh;
		if ((replay & cls) != 0 || !CpShadowLoadsSh(offset)) {
			EmitSet(ib, IT_SET_SH_REG, offset, PatchSh(offset, value), offset >= kShComputeBase);
			stats.preamble_sh++;
		}
	}
	for (const auto& [offset, value]: uc) {
		if ((replay & kCpShadowUconfig) != 0 || !CpShadowLoadsUconfig(offset)) {
			EmitSetUconfig(ib, offset, value);
			stats.preamble_uc++;
		}
	}
}

void Pm4Translator::WriteCpShadowImage(std::span<uint32_t> s, std::span<uint32_t> save) {
	if (s.size() < kCpShadowDwords) {
		return;
	}
	// amdgpu's clear state under the copy, for the registers a preamble LOADs that nothing has written.
	for (const auto& [first, end]: kClearStateExtents) {
		for (uint32_t offset = first; offset < end; offset++) {
			s[kCpShadowContextDw + offset] = 0u;
			m_cp_valid_cx.set(offset);
		}
	}
	for (const auto& [offset, value]: kClearStateValues) {
		s[kCpShadowContextDw + offset] = value;
	}
	for (const auto& [offset, value]: m_cx) {
		if (offset < RegFile::kDense) {
			s[kCpShadowContextDw + offset] = PatchContext(offset, value);
			m_cp_valid_cx.set(offset);
		}
	}
	// P: what a pop restores, the pushed context while one is pushed (a push translated before shadowing came on).
	if (save.size() >= kCpShadowSaveBytes / 4u) {
		for (const auto& [offset, value]: m_cx_pushed ? m_saved_cx : m_cx) {
			if (offset < RegFile::kDense) {
				save[offset] = PatchContext(offset, value);
				m_saved_valid_cx.set(offset); // what a GPU pop LOADs from it
			}
		}
	}
	for (const auto& [offset, value]: m_sh) {
		if (offset < RegFile::kDense) {
			s[kCpShadowShDw + offset] = PatchSh(offset, value);
		}
	}
	for (const auto& [offset, value]: m_uc) {
		if (offset < RegFile::kDense) {
			s[kCpShadowUconfigDw + offset] = value;
		}
	}
}

Pm4Translator::CpShadowCheck Pm4Translator::CompareCpShadow(std::span<const uint32_t> s, std::FILE* report) const {
	CpShadowCheck c;
	if (s.size() < kCpShadowDwords) {
		return c;
	}
	static const char* const kWriter[] = {"nothing since the seed", "a SET", "a verbatim table", "an expanded table",
	                                      "an insert of the translator's", "the GPU's FMASK fix"};
	auto key = [&](const char* cls, uint32_t region, uint32_t off, uint32_t raw, uint32_t want, uint8_t writer) {
		c.keys++;
		const uint32_t got = s[region + off];
		if (got == want) {
			c.equal++;
			return;
		}
		const auto     w     = static_cast<RegWriter>(writer);
		const bool     later = w == RegWriter::Table || w == RegWriter::Gpu; // the GPU decided after the copy
		const char*    kind  = w == RegWriter::Gpu ? "set on the GPU" : later ? "late table" : w == RegWriter::Seed ? "unseen write" : "not shadowed";
		(later ? c.late : w == RegWriter::Seed ? c.unseen : c.unshadowed)++;
		if (report != nullptr) {
			std::fprintf(report, "%s 0x%03x: S 0x%08x, copy 0x%08x", cls, off, got, raw);
			if (want != raw) {
				std::fprintf(report, " (0x%08x as the IB writes it)", want);
			}
			std::fprintf(report, ", last written by %s: %s\n", kWriter[writer < 6 ? writer : 0], kind);
		}
	};
	for (const auto& [off, value]: m_cx) {
		if (off < RegFile::kDense) {
			key("cx", kCpShadowContextDw, off, value, PatchContext(off, value), m_writer_cx[off]);
		}
	}
	for (const auto& [off, value]: m_sh) {
		if (off < RegFile::kDense) {
			key("sh", kCpShadowShDw, off, value, PatchSh(off, value), m_writer_sh[off]);
		}
	}
	for (const auto& [off, value]: m_uc) {
		if (off < RegFile::kDense) {
			key("uc", kCpShadowUconfigDw, off, value, value, m_writer_uc[off]);
		}
	}
	// Registers the copy holds no key for. The preamble's own writes are expected there: PA_SC_SHADER_CONTROL and
	// DB_RENDER_OVERRIDE when absent, the anchors, the tess rings, VGT_INDEX_TYPE's start value, and the title's
	// SQ_THREAD_TRACE_USERDATA tags (passed as written, never kept). Anything else only a late table wrote.
	auto ours = [&](uint32_t region, uint32_t off) {
		if (region == kCpShadowContextDw) {
			return off == kPaScShaderControl || off == kDbRenderOverride || off == 0x080u;
		}
		if (region == kCpShadowShDw) {
			return off == 0x00Cu || off == 0x240u;
		}
		return off == 0x24Au || off == 0x24Eu || off == 0x24Fu || off == 0x250u || off == 0x261u || off == 0x243u ||
		       (off >= 0x340u && off <= 0x347u);
	};
	auto outside = [&](const char* cls, uint32_t region, const RegFile& regs) {
		for (uint32_t off = 0; off < RegFile::kDense; off++) {
			const uint32_t got = s[region + off];
			// The seed's clear state of a context register nothing wrote is expected too.
			if (got == 0 || regs.count(off) != 0 || (region == kCpShadowContextDw && InClearState(off) && got == ClearStateValue(off))) {
				continue;
			}
			const bool expected = ours(region, off);
			c.outside += expected ? 0u : 1u;
			if (report != nullptr) {
				std::fprintf(report, "%s 0x%03x: S 0x%08x, not in the copy%s\n", cls, off, got,
				             expected ? " (the preamble writes it, or a tag)" : ": only a late table wrote it");
			}
		}
	};
	outside("cx", kCpShadowContextDw, m_cx);
	outside("sh", kCpShadowShDw, m_sh);
	outside("uc", kCpShadowUconfigDw, m_uc);
	return c;
}

void LodBankScan::Merge(const LodBankScan& o) {
	banks |= o.banks;
	tsharps += o.tsharps;
	counting += o.counting;
	inline_ud += o.inline_ud;
	depth1 += o.depth1;
	depth2 += o.depth2;
	tables += o.tables;
	max_word = std::max(max_word, o.max_word);
	alt_counting += o.alt_counting;
	alt_banks |= o.alt_banks;
	for (const auto& [warn, n]: o.min_lod_warn) {
		min_lod_warn[warn] += n;
	}
	for (const auto& t: o.samples) {
		if (samples.size() < 8) {
			samples.push_back(t);
		}
	}
}

LodStatsMode ParseLodStatsMode(const char* mode) {
	if (mode == nullptr || mode[0] == '\0' || std::strcmp(mode, "usage") == 0 || std::strcmp(mode, "hw") == 0) {
		return LodStatsMode::Usage;
	}
	if (std::strcmp(mode, "banks") == 0) {
		return LodStatsMode::Banks;
	}
	if (std::strcmp(mode, "even") == 0) {
		return LodStatsMode::Even;
	}
	if (std::strcmp(mode, "lo") == 0) {
		return LodStatsMode::Lo;
	}
	if (std::strcmp(mode, "hi") == 0) {
		return LodStatsMode::Hi;
	}
	return LodStatsMode::All; // "all", "fill", or unrecognised: the whole-buffer fill
}

uint32_t WriteLodUsageReport(void* dst, uint32_t size_bytes, const std::bitset<256>& banks) {
	if (dst == nullptr || size_bytes == 0) {
		return 0;
	}
	auto*          bytes  = static_cast<uint8_t*>(dst);
	const uint32_t header = std::min(size_bytes, kLodStatsHeader);
	std::memset(bytes, 0, header);
	uint32_t records = 0;
	if (size_bytes > kLodStatsHeader) {
		records = (size_bytes - kLodStatsHeader) / 8u;
		if (records > 256u) {
			records = 256u;
		}
	}
	uint32_t used = 0;
	for (uint32_t bank = 0; bank < records; bank++) {
		const uint64_t rec = banks.test(bank) ? kLodStatsUsed : kLodStatsNoData;
		used += banks.test(bank) ? 1u : 0u;
		std::memcpy(bytes + kLodStatsHeader + bank * 8u, &rec, sizeof(rec));
	}
	const uint32_t tail = kLodStatsHeader + records * 8u;
	if (size_bytes > tail) {
		std::memset(bytes + tail, 0, size_bytes - tail);
	}
	if (size_bytes >= sizeof(uint32_t)) {
		*static_cast<uint32_t*>(dst) = 1;
	}
	return used;
}

void FormatLodBankRanges(const std::bitset<256>& banks, char* dst, uint32_t dst_size) {
	if (dst == nullptr || dst_size == 0) {
		return;
	}
	uint32_t used = 0;
	dst[0]        = '\0';
	auto append   = [&](const char* s) {
		while (*s != '\0' && used + 1 < dst_size) {
			dst[used++] = *s++;
		}
		dst[used] = '\0';
	};
	bool any = false;
	for (uint32_t i = 0; i < 256; i++) {
		if (!banks.test(i)) {
			continue;
		}
		uint32_t j = i;
		while (j + 1 < 256 && banks.test(j + 1)) {
			j++;
		}
		char piece[32];
		std::snprintf(piece, sizeof(piece), "%s%u%s", any ? "," : "", i, j != i ? "" : "");
		append(piece);
		if (j != i) {
			std::snprintf(piece, sizeof(piece), "-%u", j);
			append(piece);
		}
		any = true;
		i   = j;
	}
	if (!any) {
		append("none");
	}
}

void LodStatsHandoff::Push(uint64_t addr, const std::bitset<256>& banks) {
	std::lock_guard<std::mutex> lock(m_mutex);
	if (m_queue.size() >= kCap) {
		m_queue.pop_front();
		if (m_dropped++ == 0) {
			std::fprintf(stderr, "LOD stats: the report queue is full; the oldest report was dropped\n");
		}
	}
	m_queue.emplace_back(addr, banks);
}

bool LodStatsHandoff::Take(uint64_t addr, std::bitset<256>& banks) {
	std::lock_guard<std::mutex> lock(m_mutex);
	for (auto it = m_queue.begin(); it != m_queue.end(); ++it) {
		if (it->first == addr) {
			banks = it->second;
			m_queue.erase(it);
			return true;
		}
	}
	return false;
}

LodStatsHandoff& LodStatsReports() {
	static LodStatsHandoff handoff;
	return handoff;
}

Pm4TranslateResult Pm4Translator::Translate(std::span<const uint32_t> dcb) {
	Pm4TranslateResult             r;
	const std::vector<SourcePiece> source = std::move(m_source); // SetSource is for this buffer only
	m_source.clear();

	// The register tables of SET_*_REG_INDIRECT sit in guest memory the title has just written, usually
	// from another core: read one by one in the walk, each is a chain of cache misses (in GT7 the walk
	// took twice as long as the same walk on warm caches). Asking for all of them up front lets those
	// misses overlap. KYTY_NATIVE_GPU_TABLE_PREFETCH=0 turns it off.
	static const bool table_prefetch = [] {
		const char* v = std::getenv("KYTY_NATIVE_GPU_TABLE_PREFETCH");
		return v == nullptr || v[0] != '0';
	}();
	for (size_t dw = 0; table_prefetch && dw < dcb.size();) {
		const uint32_t header = dcb[dw];
		if ((header >> 30u) != 3u) {
			dw++;
			continue;
		}
		const uint32_t size = PacketSizeDw(header);
		const uint32_t op   = (header >> 8u) & 0xffu;
		if ((op == IT_SET_CONTEXT_REG_INDIRECT || op == IT_SET_SH_REG_INDIRECT || op == IT_SET_UCONFIG_REG_INDIRECT) && size >= 5 &&
		    dw + 4 < dcb.size()) {
			const uint64_t addr  = (static_cast<uint64_t>(dcb[dw + 1]) & 0xfffffffcull) | (static_cast<uint64_t>(dcb[dw + 2]) << 32u);
			const uint64_t bytes = 8ull * (dcb[dw + 4] & 0x3fffu);
			for (uint64_t line = addr & ~63ull; addr != 0 && line < addr + bytes; line += 64) {
				__builtin_prefetch(reinterpret_cast<const void*>(line));
			}
		}
		dw += size;
	}

	// First pass: walk the DCB, refuse on anything unknown, and fold register writes into the
	// shadow while collecting the body packets. Nothing is emitted until the pass succeeds, so a
	// refused DCB leaves the shadow and the IB in a usable state for the next one.
	std::vector<uint32_t> body;
	body.reserve(dcb.size());
	// SetGpuTimes: the GPU clock when the micro engine starts the buffer (its end at the last segment's
	// end). A first segment that does nothing is dropped with this clock: that pair stays without its
	// begin.
	const bool     span_timed = m_time_pairs != 0 && !m_compute_ring;
	const uint64_t span_pair  = m_time_span_va + 16ull * (m_time_spans % (m_time_pairs != 0 ? m_time_pairs : 1u));
	if (span_timed) {
		EmitClockCopy(body, span_pair);
	}
	Pm4TranslateStats stats;

	// A scratch copy of the shadow so a refusal mid-DCB does not leave it half-updated.
	// (Unknown packets are collected in r.unknown_ops and refuse the DCB once the walk is done.)
	auto cx        = m_cx;
	auto saved_cx  = m_saved_cx;
	bool cx_pushed = m_cx_pushed;
	auto sh = m_sh;
	auto uc = m_uc;
	auto cp = m_cp;
	// CP shadowing (SetCpShadow): the preambles turn it on and LOAD from S. Who last wrote each register since
	// S was seeded is kept for CompareCpShadow, committed with the copy (nothing is tracked with it off).
	const bool cp_shadow = CpShadow() != CpShadowMode::Off;
	auto       wcx       = m_writer_cx;
	auto       wsh       = m_writer_sh;
	auto       wuc       = m_writer_uc;
	auto       saved_wcx = m_saved_writer_cx;
	auto       vcx       = m_cp_valid_cx; // context registers S has a value for
	auto       saved_vcx = m_saved_valid_cx; // and had at the GPU push
	// After a shadow preamble (CLEAR_STATE, then LOADs of the keys) S holds the GPU's value only for the keys.
	if (cp_shadow) {
		vcx.reset();
		for (const auto& [off, value]: cx) {
			if (off < RegFile::kDense) {
				vcx.set(off);
			}
		}
	}
	uint8_t    fmask_done = 0; // MRT slots the GPU fixed since their registers changed or a preamble ran
	uint8_t    dcc_done   = 0; // the same for the display-DCC fix
	auto       writer    = [this, cp_shadow, &wcx, &vcx, &fmask_done, &dcc_done](std::array<uint8_t, RegFile::kDense>& w, uint32_t off,
	                                                                          RegWriter kind) {
        if (cp_shadow && off < RegFile::kDense) {
            w[off] = static_cast<uint8_t>(kind);
            if (&w == &wcx) {
                vcx.set(off);
                fmask_done &= static_cast<uint8_t>(~FmaskFixSlots(off));
                dcc_done &= static_cast<uint8_t>(~DccFixSlots(off));
                const bool info = off >= kCbColor0Base && off < kCbColor0Base + 8 * kCbColorStride &&
                                  (off - kCbColor0Base) % kCbColorStride == kCbColor0Info - kCbColor0Base;
                if (info && kind != RegWriter::Gpu) {
                    m_display_dcc_gpu &= ~(1u << ((off - kCbColor0Base) / kCbColorStride)); // the title's INFO again: nothing to give back
                }
            }
        }
	};
	const bool gpu_pushpop = CpShadowGpuPushPop();
	// Load mode: the GPU's context registers come from S, so where the copy took a register from a verbatim table
	// (the CP reads the table when it runs it, which may be after a GPU write the translator never sees) the GPU may
	// hold another value; a GPU pop gives it back as S had it at the push, and a preamble LOADs it from S. The same
	// for the FMASK the GPU's own fix set. Decisions the CPU makes on such registers are made on the GPU (the FMASK
	// fix), left out (display DCC on such an INFO), or not skipped as unchanged (a clear, a CPU pop).
	const bool shadow_load = CpShadow() == CpShadowMode::Load;
	auto       stale       = [&](uint32_t off) {
		return shadow_load && off < RegFile::kDense &&
		       (wcx[off] == static_cast<uint8_t>(RegWriter::Table) || wcx[off] == static_cast<uint8_t>(RegWriter::Gpu));
	};
	// Memory this DCB writes with WRITE_DATA / DMA_DATA before the GPU gets further. The CP reads an
	// indirect register table when it runs the packet; the translator reads it now, so a table written
	// earlier in the same DCB would be read before it is written.
	// A table the CP runs as written (TABLES_VERBATIM) is read when it runs, so only the expanded ones
	// are checked, within the bounds of everything written and against the last 256 writes (a full
	// scan per load was most of the translation in ASTRO BOT's heavy views: thousands of loads against
	// thousands of writes).
	std::vector<std::pair<uint64_t, uint64_t>> gpu_written;
	uint64_t                                   gpu_written_lo = ~0ull;
	uint64_t                                   gpu_written_hi = 0;
	auto note_gpu_written = [&](uint64_t lo, uint64_t hi) {
		gpu_written.emplace_back(lo, hi);
		gpu_written_lo = std::min(gpu_written_lo, lo);
		gpu_written_hi = std::max(gpu_written_hi, hi);
	};

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
		const uint32_t* lo = sh.Find(lo_off);
		const uint32_t* hi = sh.Find(lo_off + 1);
		if (lo == nullptr || hi == nullptr) {
			return;
		}
		uint64_t va = (static_cast<uint64_t>(*lo) << 8u) | (static_cast<uint64_t>(*hi) << 40u);
		if (va != 0 && (shaders.empty() || shaders.back() != va)) {
			shaders.push_back(va);
		}
	};

	// Guest memory the GPU may touch (Pm4TranslateResult::refs). Collected at every draw and
	// dispatch from the registers that changed since the previous one: the whole shadow is live at
	// the first one, since state carries over from earlier DCBs.
	std::vector<GpuRef>   refs;
	std::map<uint64_t, RenderTargetInfo> targets; // by address, for Pm4TranslateResult::render_targets
	std::vector<RenderTargetInfo*> bound_targets; // bound at the last state collection (map nodes stay put)
	// The user-data registers written since collect_refs last ran, each once (a bit per register), and
	// only while references are collected: draws ran a sort of every SH write before (ASTRO BOT: most
	// of the reference work, which was a quarter of the translation thread).
	std::vector<uint32_t>                       dirty_sh;
	std::array<uint64_t, RegFile::kDense / 64> dirty_sh_bits {};
	auto mark_sh = [&](uint32_t off) {
		if (!m_collect_refs || off >= RegFile::kDense || !IsUserDataRegister(off)) {
			return;
		}
		const uint64_t bit = 1ull << (off % 64);
		if ((dirty_sh_bits[off / 64] & bit) == 0) {
			dirty_sh_bits[off / 64] |= bit;
			dirty_sh.push_back(off);
		}
	};
	// The value each user-data register had when collect_refs last took its references, within
	// this DCB. Titles rewrite the same user data before every draw; an unchanged value names the
	// same buckets, and a changed neighbour takes the pair itself.
	std::array<uint32_t, RegFile::kDense> ud_seen_value {};
	std::array<uint64_t, RegFile::kDense / 64> ud_seen {};
	// The bound index buffer carries over from the previous DCB, as the CP state does on PS5.
	uint64_t index_base  = m_cp.index_base & 0xffffffffffffull; // from INDEX_BASE, for draws that use the bound buffer
	uint64_t index_bytes = m_cp.index_size != 0xffffffffu ? static_cast<uint64_t>(m_cp.index_size) * 4u
	                                                      : 0; // from INDEX_BUFFER_SIZE (indices, at most 4 bytes each)
	bool                  all_sh_dirty = true;
	bool                  rt_dirty     = true;
	auto mark_cx = [&](uint32_t off) {
		if (IsRenderTargetRegister(off)) {
			rt_dirty = true;
		}
		// The guest wrote CB_COLORn_INFO itself: its value is the target's own, so fix_display_dcc must
		// not give it back the DCC it took from a video-out buffer drawn before. A target with the
		// display's format and no DCC (My First Gran Turismo's replay image) has exactly the cleared
		// value and was turned into a DCC target with no metadata: black, and CB faults at address 0.
		if (off >= kCbColor0Info && off < kCbColor0Info + 8 * kCbColorStride && (off - kCbColor0Info) % kCbColorStride == 0) {
			m_display_dcc_off &= ~(1u << ((off - kCbColor0Info) / kCbColorStride));
		}
	};
	auto reg = [](const RegFile& regs, uint32_t off) -> uint32_t {
		const uint32_t* v = regs.Find(off);
		return v == nullptr ? 0u : *v;
	};
	// Compact refs: one bit per 16 MiB bucket of the two windows guest memory lives in, the low
	// 1 TiB and Kyty's extended memory (512 GiB at 8 TiB). A "reference" anywhere else is a user-data
	// value that is no address at all.
	constexpr uint32_t kBucketShift = 24;
	constexpr uint64_t kLowBuckets  = 1ull << (40 - kBucketShift);
	constexpr uint64_t kExtBase     = 0x080000000000ull;
	constexpr uint64_t kExtBuckets  = (512ull << 30u) >> kBucketShift;
	if (m_compact_refs) {
		m_bucket_bits.assign((kLowBuckets + kExtBuckets) / 64, 0);
	}
	auto bucket_index = [&](uint64_t bucket) -> uint64_t {
		if (bucket < kLowBuckets) {
			return bucket;
		}
		const uint64_t ext = bucket - (kExtBase >> kBucketShift);
		return bucket >= (kExtBase >> kBucketShift) && ext < kExtBuckets ? kLowBuckets + ext : ~0ull;
	};
	auto add_ref = [&](uint64_t addr, uint64_t size, bool pointer, bool index = false) {
		if (addr == 0 || !m_collect_refs) {
			return;
		}
		if (!m_compact_refs || index) {
			refs.push_back(GpuRef {addr, size, pointer, index});
			return;
		}
		const uint64_t first = addr >> kBucketShift;
		const uint64_t last  = (addr + (size != 0 ? size : 1) - 1) >> kBucketShift;
		for (uint64_t b = first; b <= last; b++) {
			const uint64_t i = bucket_index(b);
			if (i != ~0ull) {
				m_bucket_bits[i / 64] |= 1ull << (i % 64);
			}
		}
	};
	// A label the GPU can actually touch. GpuWritable is a mapping (a GPU view, or a writable
	// external range). A view's block stays the dummy page until MakeResident promotes it, and
	// that only happens for addresses this buffer references, so a reachable label is added
	// here. No mapping means it cannot be made resident: the wait stays on the CPU and the
	// write stays on the tracker. A GPU wait or release on the dummy never sees the guest
	// label, and the ring does not finish.
	auto label_gpu = [&](uint64_t addr, uint64_t bytes) -> bool {
		if (addr == 0 || bytes == 0 || m_labels == nullptr || !m_labels->GpuWritable(addr) ||
		    !m_labels->GpuWritable(addr + bytes - 1)) {
			return false;
		}
		add_ref(addr, bytes, false);
		return true;
	};
	// A user-data dword pair can be a 48-bit pointer (V# and table pointers) or the 40-bit
	// 256-byte-aligned base of a T# (dword 0 = address >> 8, low byte of dword 1 = address >> 40);
	// a single dword can be address >> 8 (shader programs, inline image bases).
	auto add_pair = [&](uint32_t lo, uint32_t hi) {
		add_ref(static_cast<uint64_t>(lo) | (static_cast<uint64_t>(hi & 0xffffu) << 32u), 0, true);
		add_ref((static_cast<uint64_t>(lo) | (static_cast<uint64_t>(hi & 0xffu) << 32u)) << 8u, m_guess_span, false);
	};
	// SetFindRange diagnostics. Guest VA == host VA, but a stale pointer must not crash the emulator,
	// so tables are read with process_vm_readv on ourselves (EFAULT instead of SIGSEGV).
	uint32_t cur_op = 0;
	auto read_safe = [&](uint64_t addr, uint32_t* dst, size_t dwords) -> size_t {
		if (m_lod_read != nullptr) {
			return m_lod_read(addr, dst, dwords, m_lod_read_user);
		}
#if defined(__linux__)
		struct iovec local  = {dst, dwords * 4};
		struct iovec remote = {reinterpret_cast<void*>(addr), dwords * 4};
		const ssize_t n     = ::process_vm_readv(::getpid(), &local, 1, &remote, 1, 0);
		return n > 0 ? static_cast<size_t>(n) / 4 : 0;
#else
		(void)addr;
		(void)dst;
		(void)dwords;
		return 0;
#endif
	};
	auto find_refs = [&]() {
		const bool dispatch = cur_op == IT_DISPATCH_DIRECT || cur_op == IT_DISPATCH_INDIRECT;
		auto in_range = [&](uint64_t a) { return a >= m_find_lo && a < m_find_hi; };
		auto pgm = [&](uint32_t lo_off) {
			return (static_cast<uint64_t>(reg(sh, lo_off)) << 8u) | (static_cast<uint64_t>(reg(sh, lo_off + 1)) << 40u);
		};
		FindHit base;
		base.dispatch = dispatch;
		base.index    = dispatch ? stats.dispatches : stats.draws;
		base.shader   = dispatch ? pgm(0x20C) : pgm(0x008);
		base.es       = dispatch ? 0 : pgm(0x0C8);
		auto check = [&](uint32_t lo, uint32_t hi, FindHit at) {
			const uint64_t image = (static_cast<uint64_t>(lo) | (static_cast<uint64_t>(hi & 0xffu) << 32u)) << 8u;
			const uint64_t ptr   = static_cast<uint64_t>(lo) | (static_cast<uint64_t>(hi & 0xffffu) << 32u);
			if (in_range(image)) {
				at.target = image;
				at.image  = true;
				r.find_hits.push_back(at);
			} else if (in_range(ptr)) {
				at.target = ptr;
				r.find_hits.push_back(at);
			}
		};
		auto plausible = [](uint64_t p) { return p >= 0x100000000ull && p < 0x10000000000ull && (p & 3u) == 0; };
		std::vector<uint32_t> regs;
		if (dispatch) {
			for (uint32_t o = 0x240; o < 0x250; o++) regs.push_back(o);
		} else {
			for (uint32_t o = 0x00C; o < 0x02C; o++) regs.push_back(o);
			for (uint32_t o = 0x08C; o < 0x0AC; o++) regs.push_back(o);
		}
		if (!dispatch) {
			// Color targets, whatever CB_TARGET_MASK says (depth 100 + MRT slot).
			for (uint32_t i = 0; i < 8; i++) {
				const uint64_t cb = (static_cast<uint64_t>(reg(cx, kCbColor0Base + i * kCbColorStride)) << 8u) |
				                    (static_cast<uint64_t>(reg(cx, kCbColor0BaseExt + i) & 0xffu) << 40u);
				if (in_range(cb)) {
					FindHit at = base;
					at.depth   = 100 + i;
					at.word    = reg(cx, kCbTargetMask);
					at.target  = cb;
					at.image   = true;
					r.find_hits.push_back(at);
				}
			}
		}
		std::set<uint64_t> visited;
		for (uint32_t o: regs) {
			if (sh.count(o) == 0 || sh.count(o + 1) == 0) {
				continue;
			}
			FindHit at  = base;
			at.user_reg = o;
			check(reg(sh, o), reg(sh, o + 1), at);
			std::vector<uint64_t> tables {static_cast<uint64_t>(reg(sh, o)) | (static_cast<uint64_t>(reg(sh, o + 1) & 0xffffu) << 32u)};
			for (uint32_t depth = 1; depth <= 2 && !tables.empty(); depth++) {
				std::vector<uint64_t> next;
				for (uint64_t t: tables) {
					if (!plausible(t) || visited.size() > 4096 || !visited.insert(t).second) {
						continue;
					}
					// The first level may be a bindless descriptor heap: read 64 KiB of it.
					static thread_local std::vector<uint32_t> d(16384);
					const size_t n = read_safe(t, d.data(), depth == 1 ? 16384 : 128);
					for (size_t j = 0; j + 1 < n; j++) {
						at.depth = depth;
						at.table = t;
						at.word  = static_cast<uint32_t>(j);
						check(d[j], d[j + 1], at);
						const uint64_t p = static_cast<uint64_t>(d[j]) | (static_cast<uint64_t>(d[j + 1] & 0xffffu) << 32u);
						if (depth < 2 && plausible(p) && next.size() < 64) {
							next.push_back(p);
						}
					}
				}
				tables = std::move(next);
			}
		}
	};
	// SetLodBankScan: the counter banks of the T#s this draw or dispatch can sample. A T# is 8 dwords
	// inline in user data or in a descriptor table (read like find_refs, two levels deep); tables are
	// read once per GET_LOD_STATS interval.
	auto scan_lod_banks = [&]() {
		const bool dispatch = cur_op == IT_DISPATCH_DIRECT || cur_op == IT_DISPATCH_INDIRECT;
		LodBankScan& s = m_lod_cur;
		auto guest = [](uint64_t a) {
			return (a >= 0x100000000ull && a < 0x10000000000ull) || (a >= 0x080000000000ull && a < 0x088000000000ull);
		};
		// where: 0 user data, 1/2 table depth.
		auto tsharp = [&](const uint32_t* t, uint32_t where, uint32_t word) {
			const uint64_t base = (static_cast<uint64_t>(t[0]) | (static_cast<uint64_t>(t[1] & 0xffu) << 32u)) << 8u;
			if ((t[3] >> 28u) < 8u || ((t[1] >> 20u) & 0x1ffu) == 0 || ((t[3] >> 12u) & 0xfu) > ((t[3] >> 16u) & 0xfu) ||
			    !guest(base)) {
				return;
			}
			s.tsharps++;
			if (((t[6] >> 20u) & 1u) != 0) {
				s.alt_counting++;
				s.alt_banks.set((t[6] >> 12u) & 0xffu);
			}
			if (((t[5] >> 25u) & 1u) == 0) {
				return;
			}
			s.counting++;
			s.banks.set(t[6] & 0xffu);
			s.min_lod_warn[(t[5] >> 8u) & 0xfffu]++;
			(where == 0 ? s.inline_ud : where == 1 ? s.depth1 : s.depth2)++;
			if (where != 0) {
				s.max_word = std::max(s.max_word, word);
			}
			if (s.samples.size() < 8) {
				std::array<uint32_t, 8> a {};
				std::copy(t, t + 8, a.begin());
				s.samples.push_back(a);
			}
		};
		auto plausible = [&](uint64_t p) { return guest(p) && (p & 3u) == 0; };
		auto scan_table = [&](uint64_t table, uint32_t depth, std::vector<uint64_t>* next) {
			if (m_lod_tables.size() >= 16384 || !m_lod_tables.insert(table).second) {
				return;
			}
			static thread_local std::vector<uint32_t> d;
			d.resize(depth == 1 ? m_lod_scan_dwords : 256);
			const size_t n = read_safe(table, d.data(), d.size());
			s.tables++;
			for (size_t k = 0; k + 8 <= n; k += 4) {
				tsharp(&d[k], depth, static_cast<uint32_t>(k));
			}
			for (size_t j = 0; next != nullptr && j + 1 < n && next->size() < 64; j++) {
				const uint64_t p = static_cast<uint64_t>(d[j]) | (static_cast<uint64_t>(d[j + 1] & 0xffffu) << 32u);
				if (plausible(p)) {
					next->push_back(p);
				}
			}
		};
		auto stage = [&](uint32_t first, uint32_t count) {
			uint32_t ud[32] = {};
			bool     have[32] = {};
			for (uint32_t i = 0; i < count; i++) {
				if (const uint32_t* v = sh.Find(first + i); v != nullptr) {
					ud[i]   = *v;
					have[i] = true;
				}
			}
			// Draws mostly repeat a stage's user data; the same block reaches the same T#s and
			// tables, which were read for this window already.
			uint64_t h = 0xcbf29ce484222325ull ^ first;
			for (uint32_t i = 0; i < count; i++) {
				h = (h ^ (have[i] ? ud[i] : 0x9e3779b9u) ^ (static_cast<uint64_t>(have[i]) << 32u)) * 0x100000001b3ull;
				h ^= h >> 29u;
			}
			if (m_lod_stages.size() >= 65536) {
				m_lod_stages.clear();
			}
			if (!m_lod_stages.insert(h).second) {
				return;
			}
			for (uint32_t i = 0; i + 8 <= count; i++) {
				if (have[i] && have[i + 7]) {
					tsharp(&ud[i], 0, i);
				}
			}
			std::vector<uint64_t> next;
			for (uint32_t i = 0; i + 1 < count; i++) {
				const uint64_t p = static_cast<uint64_t>(ud[i]) | (static_cast<uint64_t>(ud[i + 1] & 0xffffu) << 32u);
				if (have[i] && have[i + 1] && plausible(p)) {
					scan_table(p, 1, &next);
				}
			}
			for (uint64_t p: next) {
				scan_table(p, 2, nullptr);
			}
		};
		// Every user-data SGPR a draw or dispatch can reach, the same blocks IsUserDataRegister
		// covers. Graphics: PS, VS, GS, ES, HS, LS, each with 32 user-data dwords at +0x0C and the
		// spill table at USER_DATA_ADDR (+0x02). Compute: COMPUTE_USER_DATA_0..15. stage() follows
		// descriptor tables two levels deep.
		if (dispatch) {
			stage(0x240, 16);
		} else {
			static constexpr uint32_t kStageBase[] = {0x000, 0x040, 0x080, 0x0C0, 0x100, 0x140};
			for (uint32_t base: kStageBase) {
				stage(base + 0x0Cu, 32);
				stage(base + 0x02u, 2);
			}
		}
	};
	auto collect_refs = [&]() {
		// VGT_SHADER_STAGES_EN.HS_EN: the title tessellates. The preamble programs its tess factor ring
		// (SetTessellationRings), which the GE writes, so it must be resident.
		if (static std::atomic<bool> tess_logged {false};
		    !m_compute_ring && (reg(cx, 0x2d5u) & 0x4u) != 0 && !tess_logged.exchange(true)) {
			std::fprintf(stderr, "native shadow: the title draws with tessellation (VGT_SHADER_STAGES_EN 0x%08x)\n",
			             reg(cx, 0x2d5u));
		}
		if (!m_compute_ring && (reg(cx, 0x2d5u) & 0x4u) != 0) {
			add_ref(g_tf_ring_base.load(std::memory_order_relaxed), g_tf_ring_bytes.load(std::memory_order_relaxed), false);
		}
		if (m_lod_scan) {
			scan_lod_banks();
		}
		if (m_find_hi > m_find_lo) {
			find_refs();
		}
		if (all_sh_dirty) {
			for (const auto& [off, value]: sh) {
				mark_sh(off);
			}
			all_sh_dirty = false;
		}
		// User data only (shader programs come through shader_addrs, the rest is configuration). First
		// the registers whose value changed since their references were taken (their bit stays set), then
		// their references: a pair of neighbours is added once, by the lower one when it changed.
		auto fresh = [&](uint32_t off) { return off < RegFile::kDense && (dirty_sh_bits[off / 64] >> (off % 64) & 1u) != 0; };
		for (uint32_t off: dirty_sh) {
			const uint32_t value = reg(sh, off);
			const uint64_t bit   = 1ull << (off % 64);
			if ((ud_seen[off / 64] & bit) != 0 && ud_seen_value[off] == value) {
				dirty_sh_bits[off / 64] &= ~bit;
				continue;
			}
			ud_seen[off / 64] |= bit;
			ud_seen_value[off] = value;
		}
		for (uint32_t off: dirty_sh) {
			if (!fresh(off)) {
				continue;
			}
			const uint32_t value = reg(sh, off);
			add_ref(static_cast<uint64_t>(value) << 8u, m_guess_span, false);
			if (IsUserDataRegister(off + 1) && sh.count(off + 1) != 0) {
				add_pair(value, reg(sh, off + 1));
			}
			if (off > 0 && !fresh(off - 1) && IsUserDataRegister(off - 1) && sh.count(off - 1) != 0) {
				add_pair(reg(sh, off - 1), value);
			}
		}
		for (uint32_t off: dirty_sh) {
			dirty_sh_bits[off / 64] &= ~(1ull << (off % 64));
		}
		dirty_sh.clear();
		if (!rt_dirty) {
			for (RenderTargetInfo* t: bound_targets) {
				t->draws++;
			}
			return;
		}
		rt_dirty = false;
		bound_targets.clear();
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
			const uint64_t base = base40(kCbColor0Base + step, kCbColor0BaseExt + i);
			add_ref(base, size, false);
			// INFO.FORMAT (bits 6:2) == 0 means the slot is unbound.
			if (base != 0 && ((reg(cx, kCbColor0Info + step) >> 2u) & 0x1fu) != 0) {
				auto& t   = targets[base];
				t.addr    = base;
				t.size    = size;
				t.width   = static_cast<uint32_t>(width);
				t.height  = static_cast<uint32_t>(height);
				t.info    = reg(cx, kCbColor0Info + step);
				t.attrib  = reg(cx, kCbColor0Attrib + step);
				t.attrib3 = reg(cx, kCbColor0Attrib3 + i);
				bound_targets.push_back(&t);
			}
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
		for (RenderTargetInfo* t: bound_targets) {
			t->draws++;
		}
		// A surface the draw uses at GPU address 0: the CB or DB writes page 0 (CB/DB faults in the
		// journal). Logged once per kind with the registers that locate it.
		if (!m_compute_ring) {
			static std::mutex           zero_mutex;
			static std::set<std::string> zero_seen;
			auto note_zero = [&](const std::string& what) {
				std::lock_guard<std::mutex> lock(zero_mutex);
				if (zero_seen.size() < 32 && zero_seen.insert(what).second) {
					std::fprintf(stderr, "native shadow: draw uses %s at GPU address 0 (CB_TARGET_MASK 0x%08x, DB_Z_INFO 0x%08x, "
					             "DB_DEPTH_CONTROL 0x%08x)\n",
					             what.c_str(), reg(cx, kCbTargetMask), reg(cx, kDbZInfo), reg(cx, 0x200u));
				}
			};
			const uint32_t mask = reg(cx, kCbTargetMask);
			for (uint32_t i = 0; i < 8; i++) {
				const uint32_t step = i * kCbColorStride;
				const uint32_t info = reg(cx, kCbColor0Info + step);
				if (((info >> 2u) & 0x1fu) == 0 || ((mask >> (4u * i)) & 0xfu) == 0) {
					continue;
				}
				const std::string slot = "MRT" + std::to_string(i);
				if (base40(kCbColor0Base + step, kCbColor0BaseExt + i) == 0) {
					note_zero(slot + " color (INFO " + std::to_string(info) + ")");
				}
				if ((info & (1u << 28u)) != 0 && base40(kCbColor0DccBase + step, kCbColor0DccExt + i) == 0) {
					char what[160];
					std::snprintf(what, sizeof(what), "%s DCC (target 0x%llx, INFO 0x%08x, PS 0x%llx)", slot.c_str(),
					              static_cast<unsigned long long>(base40(kCbColor0Base + step, kCbColor0BaseExt + i)), info,
					              static_cast<unsigned long long>((static_cast<uint64_t>(reg(sh, 0x008u)) << 8u) |
					                                              (static_cast<uint64_t>(reg(sh, 0x009u) & 0xffu) << 40u)));
					note_zero(what);
				}
				if ((info & (1u << 13u)) != 0 && base40(kCbColor0Cmask + step, kCbColor0CmaskExt + i) == 0) {
					note_zero(slot + " CMASK");
				}
			}
			const uint32_t zinfo = reg(cx, kDbZInfo);
			const uint32_t dctl  = reg(cx, 0x200u);
			if ((zinfo & 3u) != 0 && (dctl & 6u) != 0) {
				if (base40(kDbZReadBase, kDbZReadBaseHi) == 0 || base40(kDbZReadBase + 2, kDbZReadBaseHi + 2) == 0) {
					note_zero("depth");
				}
				if ((zinfo & (1u << 29u)) != 0 && base40(kDbHtileDataBase, kDbHtileBaseHi) == 0) {
					note_zero("HTILE");
				}
			}
			if ((reg(cx, kDbStencilInfo) & 1u) != 0 && (dctl & 1u) != 0 &&
			    (base40(kDbZReadBase + 1, kDbZReadBaseHi + 1) == 0 || base40(kDbZReadBase + 3, kDbZReadBaseHi + 3) == 0)) {
				note_zero("stencil");
			}
		}
	};

	// A bound color target without FMASK pointed at its own base (SetFmaskFix, KYTY_NATIVE_GPU_FMASK_FIX=1), as radeonsi
	// and PAL program it. Up to 28.10 ASTRO BOT's MRT4 with FMASK 0 and a CMASK fast clear hung the gfx ring here
	// without it ("Illegal register access" in 0x9F, TCP reads of address 0). Since AgcDcbResetQueue in Sony's form,
	// 0x35 as written and direct memory resident at allocation it runs 524 s in load mode without the fix (1.11), so
	// the title's FMASK 0 runs as written, as on the PS5.
	auto fix_fmask = [&]() {
		if (!m_fmask_fix) {
			return;
		}
		const uint32_t mask = reg(cx, kCbTargetMask) & m_target_mask;
		for (uint32_t i = 0; i < 8; i++) {
			const uint32_t step = i * kCbColorStride;
			// Load mode, a slot some register of which the copy took from a verbatim table (or the GPU's own fix
			// changed): the GPU may hold other values, so the CP decides on its own, once until they change again. What
			// the copy knows still rules out a slot: a mask, BASE or FMASK the GPU holds as the copy does.
			if (stale(kCbTargetMask) || stale(kCbColor0Base + step) || stale(kCbColor0BaseExt + i) || stale(kCbColor0Fmask + step) ||
			    stale(kCbColor0FmaskExt + i)) {
				const uint32_t gpu_mask = stale(kCbTargetMask) ? m_target_mask : mask;
				// A BASE S has no value for is amdgpu's clear state (0) on the GPU: nothing to fix, as on the CPU. An
				// FMASK S has none for is 0 on the GPU too, as is one the copy holds as 0: set without the check.
				const bool known_base  = !stale(kCbColor0Base + step);
				const bool known_fmask = !stale(kCbColor0Fmask + step);
				if (((gpu_mask >> (4u * i)) & 0xfu) == 0 || (fmask_done & (1u << i)) != 0 || !vcx.test(kCbColor0Base + step) ||
				    (known_base && reg(cx, kCbColor0Base + step) == 0) || (known_fmask && reg(cx, kCbColor0Fmask + step) != 0)) {
					continue;
				}
				EmitCpShadowFmaskFix(body, i, !known_fmask && vcx.test(kCbColor0Fmask + step), vcx);
				// The copy keeps its FMASK values, which the GPU may no longer hold: a switch of the context SETs
				// them again, and a later fix of the slot is the GPU's too.
				for (const uint32_t off: {kCbColor0Fmask + step, kCbColor0FmaskExt + i}) {
					wcx[off] = static_cast<uint8_t>(RegWriter::Gpu);
					vcx.set(off);
				}
				fmask_done |= static_cast<uint8_t>(1u << i);
				stats.cp_shadow_fmask_gpu++;
				continue;
			}
			const uint32_t base = reg(cx, kCbColor0Base + step);
			if (((mask >> (4u * i)) & 0xfu) == 0 || base == 0 || reg(cx, kCbColor0Fmask + step) != 0) {
				continue;
			}
			const uint32_t ext = reg(cx, kCbColor0BaseExt + i);
			cx[kCbColor0Fmask + step]  = base;
			cx[kCbColor0FmaskExt + i] = ext;
			writer(wcx, kCbColor0Fmask + step, RegWriter::Insert);
			writer(wcx, kCbColor0FmaskExt + i, RegWriter::Insert);
			EmitSet(body, IT_SET_CONTEXT_REG, kCbColor0Fmask + step, base, false);
			EmitSet(body, IT_SET_CONTEXT_REG, kCbColor0FmaskExt + i, ext, false);
			stats.fmask_fixed++;
		}
	};

	// Switches the context registers to `next` (a pop, a clear): the registers that change are set
	// again, those `next` lacks go back to 0.
	auto switch_context = [&](const RegFile& next) {
		for (const auto& [off, value]: next) {
			const uint32_t* cur = cx.Find(off);
			if (cur == nullptr || *cur != value || stale(off)) {
				EmitSet(body, IT_SET_CONTEXT_REG, off, PatchContext(off, value), false);
				mark_cx(off);
				writer(wcx, off, RegWriter::Insert);
			}
		}
		for (const auto& [off, value]: cx) {
			if (next.count(off) == 0 && (value != 0 || stale(off))) {
				EmitSet(body, IT_SET_CONTEXT_REG, off, PatchContext(off, 0), false);
				mark_cx(off);
				writer(wcx, off, RegWriter::Insert);
			}
		}
		cx = next;
	};
	// Agc context-state operations (AgcDcbContextStateOp, emulated by Kyty's command processor):
	// 0 clear to the defaults, 1 push, 2 pop, 3 push then clear. The PS5 packets behind them save and
	// restore the context registers; here the shadow does it and the differences are emitted.
	// With CP shadowing in load mode (SetCpShadowPushPop) the push and the pop run on the GPU through S: a push
	// copies S's context region to P (the first of a window only), the pop copies it back and LOADs it, so the GPU
	// gets back what it had at the push, late tables included. The copy follows: saved_cx is the pushed state,
	// the pop makes it cx again (the registers that change are marked, as a CPU pop's SETs would mark them).
	auto gpu_push = [&]() {
		if (cx_pushed) {
			return; // the PS5 restores the first pushed state
		}
		EmitCpShadowPush(body);
		saved_cx  = cx;
		saved_wcx = wcx;
		saved_vcx = vcx;
		m_saved_display_dcc_gpu = m_display_dcc_gpu; // P's INFO is S's, DCC cleared or not
		cx_pushed = true;
		stats.cp_shadow_pushes++;
	};
	auto gpu_pop = [&]() {
		// P holds S as it was at the push: the registers it had a value for then are LOADed. One first written
		// inside the window has no pushed value in P; it gets the CPU pop's 0 when the copy holds it.
		EmitCpShadowPop(body, saved_cx, cx, saved_vcx);
		for (const auto& [off, value]: cx) {
			if (off < RegFile::kDense && !saved_vcx.test(off) && CpShadowLoadsContext(off) && saved_cx.count(off) == 0 &&
			    (value != 0 || stale(off))) {
				EmitSet(body, IT_SET_CONTEXT_REG, off, PatchContext(off, 0), false);
				saved_vcx.set(off);
			}
		}
		vcx = saved_vcx; // S's context region is P's again
		for (const auto& [off, value]: saved_cx) {
			const uint32_t* cur = cx.Find(off);
			if (cur == nullptr || *cur != value) {
				mark_cx(off);
			}
		}
		for (const auto& [off, value]: cx) {
			if (saved_cx.count(off) == 0) {
				mark_cx(off);
			}
		}
		cx  = saved_cx;
		wcx = saved_wcx;
		fmask_done = 0; // the CB registers are the pushed ones again
		dcc_done   = 0;
		// An INFO cleared inside the window is the pushed one again: giving that one's DCC back would put it on
		// whatever target the pushed INFO belongs to.
		m_display_dcc_gpu = m_saved_display_dcc_gpu;
		stats.cp_shadow_pops++;
	};
	auto context_state_op = [&](uint32_t operation) {
		stats.context_ops++;
		switch (operation) {
			case 0: switch_context(m_cx_defaults); break;
			case 1:
				if (gpu_pushpop) {
					gpu_push();
					break;
				}
				saved_cx  = cx;
				cx_pushed = true;
				break;
			case 2:
				if (cx_pushed) {
					if (gpu_pushpop) {
						gpu_pop();
					} else {
						switch_context(saved_cx);
					}
					saved_cx.clear();
					cx_pushed = false;
				}
				break;
			case 3:
				if (gpu_pushpop) {
					gpu_push();
				} else {
					saved_cx  = cx;
					cx_pushed = true;
				}
				switch_context(m_cx_defaults);
				break;
			default: break;
		}
	};

	// True when the bound pixel shader is one SetSkippedPixelShaders named; the draw is then left out.
	uint32_t log_instances = 1;
	uint64_t log_set_base  = 0;
	uint32_t log_draws     = 0;
	auto     log_draw      = [&](uint32_t op, uint32_t count, uint64_t indirect, bool dispatch) {
		if (!m_draw_log) {
			return;
		}
		auto fnv = [&](uint32_t first) {
			uint32_t h = 2166136261u;
			for (uint32_t i = 0; i < (dispatch ? 16u : 32u); i++) {
				h = (h ^ reg(sh, first + i)) * 16777619u;
			}
			return h;
		};
		DrawRecord d;
		d.op       = op;
		d.index    = log_draws++;
		d.count    = count;
		d.indirect = indirect;
		if (dispatch) {
			d.ps      = (static_cast<uint64_t>(reg(sh, 0x20C)) << 8u) | (static_cast<uint64_t>(reg(sh, 0x20D)) << 40u);
			d.ud_hash = fnv(0x240);
		} else {
			d.instances = log_instances;
			d.ps        = (static_cast<uint64_t>(reg(sh, 0x008)) << 8u) | (static_cast<uint64_t>(reg(sh, 0x009)) << 40u);
			d.gs        = (static_cast<uint64_t>(reg(sh, 0x0C8)) << 8u) | (static_cast<uint64_t>(reg(sh, 0x0C9)) << 40u);
			d.cb0       = (static_cast<uint64_t>(reg(cx, kCbColor0Base)) << 8u) | (static_cast<uint64_t>(reg(cx, kCbColor0BaseExt) & 0xffu) << 40u);
			d.zbase     = (static_cast<uint64_t>(reg(cx, kDbZReadBase)) << 8u) | (static_cast<uint64_t>(reg(cx, kDbZReadBaseHi) & 0xffu) << 40u);
			d.ud_hash   = fnv(0x00C);
			d.gs_hash   = fnv(0x08C);
		}
		r.draw_log.push_back(d);
	};
	// Color targets in a video-out buffer are drawn without DCC (AddDisplayRange). The shadow holds
	// the value sent, so preambles agree; a slot that moves elsewhere gets the guest's value back.
	auto fix_display_dcc = [&]() {
		if (m_display_ranges.empty()) {
			return;
		}
		for (uint32_t i = 0; i < 8; i++) {
			const uint32_t step = i * kCbColorStride;
			const uint32_t off  = kCbColor0Info + step;
			const uint32_t info = reg(cx, off);
			const uint64_t addr = (static_cast<uint64_t>(reg(cx, kCbColor0Base + step)) << 8u) |
			                      (static_cast<uint64_t>(reg(cx, kCbColor0BaseExt + i) & 0xffu) << 40u);
			const bool display = addr != 0 && std::any_of(m_display_ranges.begin(), m_display_ranges.end(), [&](const auto& d) {
				return addr >= d.first && addr < d.first + d.second;
			});
			// Load mode, an INFO the copy took from a verbatim table: the CP may hold another (it reads the table when it
			// runs it), so the GPU clears DCC in its own (EmitCpShadowDccOff), once until the slot's INFO or BASE change, and
			// gives it back when the slot leaves the video-out buffer with that INFO. The copy's INFO would replace the CP's,
			// format and all (GT7 then faulted); left out, the video-out buffer went to the display compressed (GT7's menus:
			// noise over the UI and the water). The copy's BASE decides. An INFO S has no value for is 0 on the GPU.
			if (stale(off)) {
				if (display && (dcc_done & (1u << i)) == 0 && vcx.test(off)) {
					EmitCpShadowDccOff(body, i);
					wcx[off] = static_cast<uint8_t>(RegWriter::Gpu);
					dcc_done |= static_cast<uint8_t>(1u << i);
					m_display_dcc_gpu |= 1u << i;
					stats.cp_shadow_dcc_gpu++;
					NoteOnce(17, "render target in a video-out buffer drawn without DCC");
				} else if (!display && ((m_display_dcc_gpu >> i) & 1u) != 0) {
					EmitCpShadowDccRestore(body, i);
					m_display_dcc_gpu &= ~(1u << i);
					stats.cp_shadow_dcc_back++;
				}
				continue;
			}
			if (display && (info & kCbDccEnable) != 0) {
				m_display_dcc_info[i] = info;
				m_display_dcc_off |= 1u << i;
				cx[off] = info & ~kCbDccEnable;
				EmitSet(body, IT_SET_CONTEXT_REG, off, PatchContext(off, cx[off]), false);
				writer(wcx, off, RegWriter::Insert);
				NoteOnce(17, "render target in a video-out buffer drawn without DCC");
			} else if (!display && ((m_display_dcc_off >> i) & 1u) != 0) {
				if (info == (m_display_dcc_info[i] & ~kCbDccEnable)) {
					cx[off] = m_display_dcc_info[i];
					EmitSet(body, IT_SET_CONTEXT_REG, off, PatchContext(off, cx[off]), false);
					writer(wcx, off, RegWriter::Insert);
				}
				m_display_dcc_off &= ~(1u << i);
			}
		}
	};
	auto skip_draw = [&]() {
		fix_fmask();
		fix_display_dcc();
		const uint64_t ps = (static_cast<uint64_t>(reg(sh, 0x008)) << 8u) | (static_cast<uint64_t>(reg(sh, 0x009)) << 40u);
		if (!m_inspect.empty() && m_inspect.count(ps) != 0 && r.inspected.size() < 4) {
			DrawSnapshot snap;
			snap.ps = ps;
			snap.es = (static_cast<uint64_t>(reg(sh, 0x0C8)) << 8u) | (static_cast<uint64_t>(reg(sh, 0x0C9)) << 40u);
			for (uint32_t i = 0; i < 32; i++) {
				snap.ps_user_data[i] = reg(sh, 0x00C + i);
				snap.gs_user_data[i] = reg(sh, 0x08C + i);
			}
			r.inspected.push_back(snap);
		}
		return !m_skip_ps.empty() && m_skip_ps.count(ps) != 0;
	};

	// Indirect arguments are often written by the CP itself just before (DMA_DATA copying a GDS counter,
	// WRITE_DATA), and on the gfx ring the PFP can fetch them ahead of the ME. libSceAgc's indirect draws
	// carry no sync of their own, and without one ASTRO BOT, My First GT and Dead Cells draw right: the
	// title orders its writes itself. KYTY_NATIVE_GPU_PFP_SYNC_INDIRECT=1 puts the RADV-style sync back.
	// A compute queue's buffer always gets it on the gfx ring: the PS5 runs it on MEC, which has no
	// prefetch parser, so the title's stream has nothing to order the PFP with (BC5 F91).
	auto pfp_sync_me = [&]() {
		// A compute ring has one engine; the PFP race this closes exists only on the gfx ring, and
		// PFP_SYNC_ME is illegal on MEC.
		if ((m_pfp_sync_indirect || m_compute_queue) && !m_compute_ring) {
			body.push_back((3u << 30u) | (IT_PFP_SYNC_ME << 8u));
			body.push_back(0);
			stats.pfp_syncs_emitted++;
		}
	};

	// CP DMA runs beside the CP and outlives the IB it was issued in: a DMA_DATA without CP_SYNC may still be
	// copying when the job's end-of-pipe fence or a tracker release says the job is done (BC5 F41: the next IB
	// hung). The PS5's contract is that a finished job's writes have landed, so a job end after such a DMA waits
	// for it with a zero-byte DMA_DATA with CP_SYNC, Mesa's si_cp_dma_wait_for_idle: the DMA engine finds no
	// work and holds the ME until the copies before it are done. KYTY_NATIVE_GPU_CP_DMA_IDLE=0 leaves it out.
	bool cp_dma_busy = false;
	auto cp_dma_idle = [&]() {
		if (!cp_dma_busy || !m_cp_dma_idle) {
			return;
		}
		// CP_SYNC, DST_SEL 3 and SRC_SEL 3 (TC_L2) on the micro engine; no addresses, BYTE_COUNT 0.
		body.insert(body.end(), {Pkt3(IT_DMA_DATA, 6, m_compute_ring), (1u << 31u) | (3u << 20u) | (3u << 29u), 0u, 0u, 0u, 0u, 0u});
		cp_dma_busy = false;
		stats.cp_dma_idle++;
	};

	// A guest wait turned into a full barrier on our fence: an end-of-pipe RELEASE_MEM that flushes and
	// invalidates the caches (Mesa's gfx10 wait-for-idle) and a wait on that fence. A wait the prefetch
	// parser does (ENGINE_SEL = PFP) on a label the micro engine has just written is how the guest keeps
	// the PFP from fetching ahead of the ME past a cache flush; PFP_SYNC_ME is that synchronization
	// without memory.
	//
	// SetCpShadowWaitOff: in a shadowing job every GPU wait runs with shadowing off, so a job that hangs there leaves
	// the CP outside a shadow region when amdgpu resets the queue (kCpShadowWaitOff). wait_on's `me_unsynced`: the
	// last wait was the ME's and nothing after it synced the PFP, so PFP_SYNC_ME first, or the PFP would turn
	// shadowing on again while the ME still waits.
	const bool wait_wrap = CpShadowWaitWrap();
	auto       wait_off  = [&]() {
		body.insert(body.end(), kCpShadowWaitOff.begin(), kCpShadowWaitOff.end());
		stats.cp_shadow_wait_wraps++;
	};
	auto wait_on = [&](bool me_unsynced) {
		if (me_unsynced) {
			body.push_back((3u << 30u) | (IT_PFP_SYNC_ME << 8u));
			body.push_back(0);
		}
		body.insert(body.end(), kCpShadowWaitOn.begin(), kCpShadowWaitOn.end());
	};
	auto emit_barrier_raw = [&](bool pfp) {
		if (m_fence_va != 0) {
			const uint32_t seq = ++m_fence_seq;
			body.push_back(Pkt3(IT_RELEASE_MEM, 7, m_compute_ring));
			// EVENT_TYPE CACHE_FLUSH_AND_INV_TS_EVENT (0x14), EVENT_INDEX 5 (end of pipe), GCR_CNTL:
			// GLM_WB | GLM_INV | GLV_INV | GL1_INV | GL2_INV | GL2_WB.
			body.push_back(0x14u | (5u << 8u) | (1u << 12u) | (1u << 13u) | (1u << 14u) | (1u << 15u) |
			               (1u << 20u) | (1u << 21u));
			body.push_back(1u << 29u); // DATA_SEL 1: 32-bit data; DST_SEL 0 memory; no interrupt
			body.push_back(static_cast<uint32_t>(m_fence_va));
			body.push_back(static_cast<uint32_t>(m_fence_va >> 32u));
			body.push_back(seq);
			body.push_back(0u);
			body.push_back(0u);
			body.push_back(Pkt3(IT_WAIT_REG_MEM, 6, m_compute_ring));
			body.push_back(3u | (1u << 4u)); // function EQUAL, memory space, ME engine
			body.push_back(static_cast<uint32_t>(m_fence_va));
			body.push_back(static_cast<uint32_t>(m_fence_va >> 32u));
			body.push_back(seq);
			body.push_back(0xffffffffu);
			body.push_back(4u); // poll interval
			stats.wait_barriers++;
		}
		if (pfp && !m_compute_ring) {
			body.push_back((3u << 30u) | (IT_PFP_SYNC_ME << 8u)); // PKT3 PFP_SYNC_ME, one body dword
			body.push_back(0);
			stats.pfp_syncs_emitted++;
		}
	};
	// The barrier as a whole between the wrap's two packets; the ME's wait on our fence gets its PFP_SYNC_ME before
	// the "on" whatever the guest's engine was (a barrier has drained the pipe already, so it costs little).
	auto emit_barrier = [&](bool pfp) {
		if (!wait_wrap || m_fence_va == 0) {
			emit_barrier_raw(pfp);
			return;
		}
		wait_off();
		emit_barrier_raw(pfp);
		wait_on(!pfp);
	};

	// Native labels: the segments of the IB (body offsets until it is assembled), where each starts in
	// the body, the preamble of each one after the first (the state where it starts) and whether it
	// does anything (an empty first segment is left out), plus every label write so far in order.
	const bool                         labels = m_labels != nullptr;
	std::vector<Pm4Segment>            segs(1);
	std::vector<uint32_t>              seg_body(1, 0);
	std::vector<std::vector<uint32_t>> seg_pre(1);
	std::vector<bool>                  seg_keep(1, true);
	bool                               seg_effects = false;
	std::vector<LabelWrite>            local_writes;
	// Label releases of this buffer whose added write-backs a GPU wait of the same buffer takes off
	// again (StrippableWriteBack): the label, the event dword's index in `body`, the bits.
	struct LabelRelease {
		size_t   write    = 0; // its entry in local_writes
		size_t   event_dw = 0;
		uint32_t strip    = 0;
	};
	std::vector<LabelRelease> label_releases;
	// The label's note_write follows, as local_writes' next entry.
	auto emit_label = [&](uint64_t addr, uint32_t value, uint32_t guest_event) {
		const uint32_t event = LabelEventCntl(guest_event);
		EmitRelease32(body, addr, value, m_compute_ring, event);
		if (const uint32_t strip = StrippableWriteBack(guest_event, event); strip != 0u) {
			label_releases.push_back({local_writes.size(), body.size() - 7u, strip});
		}
	};
	// CPU-side points of this segment (interrupts the guest asked for, labels the GPU cannot write,
	// the end). They share one sequence packet, emitted when the segment closes, instead of a cache
	// flush per point. A guest interrupt still runs, at the end of the segment, which is where the
	// continuation runs too.
	std::vector<TrackerPoint> queued_points;
	auto                      tracker_point = [&](TrackerPoint p) {
		queued_points.push_back(std::move(p));
		seg_effects = true;
	};
	// A label the GPU has just been given to write. A guest that waits on the address sleeps in
	// sync-on-address, a futex on that word, and a GPU store does not wake it (Dead Cells: the title
	// never asks for an interrupt, and the level load sleeps until the label changes). So the tracker
	// wakes the address once the GPU passes the end of this segment, whose release has written back
	// GL2 by then, so the woken guest reads the GPU's value. Nothing is stored on the CPU: on PS5 the
	// GPU alone writes the label, in stream order, and a CPU copy landing at the end of the segment
	// comes after writes that follow it (the next use's reset on another ring, a guest store) and can
	// put back a value they replaced.
	//
	// SetTrackerWrites keeps the old mirror: the tracker writes the same bytes when the GPU reaches
	// the end of this segment. `prior` is the dword at translate time, so a label the guest has
	// already replaced is left alone, and a store the GPU made visible is not written again. That
	// test cannot see a label that went back to its translate-time value in between: the mirror
	// then overwrites the newer write.
	auto mirror_store = [&](uint64_t addr, std::span<const uint32_t> words, bool one_addr) {
		if (words.empty() || addr == 0) {
			return;
		}
		if (!m_tracker_writes) {
			TrackerPoint wake;
			wake.kind = TrackerPoint::Kind::Wake;
			wake.addr = addr;
			tracker_point(std::move(wake));
			return;
		}
		TrackerPoint cpu;
		cpu.kind     = TrackerPoint::Kind::CpuWrite;
		cpu.mirror   = true;
		cpu.addr     = addr;
		cpu.one_addr = one_addr;
		cpu.data.assign(words.begin(), words.end());
		const uint32_t samples = one_addr ? 1u : static_cast<uint32_t>(cpu.data.size());
		cpu.prior.resize(samples);
		bool sampled = m_labels != nullptr;
		for (uint32_t i = 0; sampled && i < samples; i++) {
			uint32_t v = 0;
			if (!m_labels->SampleDword(addr + 4ull * i, &v)) {
				sampled = false;
			} else {
				cpu.prior[i] = v;
			}
		}
		if (!sampled) {
			cpu.prior.clear();
		}
		tracker_point(std::move(cpu));
		stats.labels_cpu++;
	};
	auto note_write = [&](const LabelWrite& w) {
		segs.back().writes.push_back(w);
		segs.back().writes.back().data = nullptr; // the buffer's memory is the title's again after this
		local_writes.push_back(w);
	};
	// The points queued so far (EOP interrupts, wakes, labels the CPU writes) as a tracker store here,
	// before a GPU wait: on the PS5 an interrupt fires where its release runs, and a title may submit
	// the writer of what it waits for only once it has seen that interrupt (My First Gran Turismo
	// stopped when its interrupts waited for the end of a segment its own wait held up).
	bool irq_pending = false; // an interrupt point is queued: store the points before the next GPU work
	auto flush_points = [&]() {
		irq_pending = false;
		if (queued_points.empty()) {
			return;
		}
		const uint32_t dw = static_cast<uint32_t>(body.size()) + 5u;
		EmitRelease32(body, m_tracker_va, 0u, m_compute_ring, TrackerEventCntl());
		for (auto& p: queued_points) {
			p.dw = dw;
			segs.back().points.push_back(std::move(p));
		}
		queued_points.clear();
	};
	// Ends the current segment with its End point; a first segment that does nothing is dropped (the
	// next segment's preamble holds its register writes).
	auto close_segment = [&](bool last) {
		if (seg_effects || last || segs.size() > 1 || !queued_points.empty()) {
			cp_dma_idle();
			TrackerPoint end;
			end.kind = TrackerPoint::Kind::End;
			queued_points.push_back(std::move(end));
			const uint32_t dw = static_cast<uint32_t>(body.size()) + 5u;
			EmitRelease32(body, m_tracker_va, 0u, m_compute_ring, TrackerEventCntl());
			for (auto& p: queued_points) {
				p.dw = dw;
				segs.back().points.push_back(std::move(p));
			}
			queued_points.clear();
		} else {
			seg_keep.back() = false;
		}
		seg_effects = false;
	};
	// Exit guards (SetExitGuard) after the guest waits this buffer gives the GPU. On a compute ring the
	// body offsets of their EXEC_COUNT dwords, set at assembly, and where the stretch the next one ends began.
	const bool            gfx_guards = m_exit_abort_va != 0 && m_exit_stub_va != 0 && !m_compute_ring;
	const bool            cs_guards  = m_exit_run_va != 0 && m_compute_ring;
	std::vector<uint32_t> body_guards;
	size_t                guard_from = 0;
	// `pfp_ahead`: the wait leaves the PFP parsing past it (an ME wait).
	auto guard_after_wait = [&](bool pfp_ahead) {
		if (gfx_guards) {
			if (pfp_ahead) {
				body.push_back((3u << 30u) | (IT_PFP_SYNC_ME << 8u));
				body.push_back(0);
			}
			EmitGfxExitGuard(body, m_exit_abort_va, m_exit_stub_va);
		} else if (cs_guards) {
			body_guards.push_back(EmitComputeExitGuard(body, m_exit_run_va));
			guard_from = body.size();
		}
	};
	// Cuts the IB at a guest wait nobody known satisfies yet: the rest starts held, with its own
	// preamble, a barrier block and the guest wait (the backend NOPs one of the two).
	auto split_at_wait = [&](const WaitCond& w) {
		close_segment(false);
		segs.emplace_back();
		seg_body.push_back(static_cast<uint32_t>(body.size()));
		seg_pre.emplace_back();
		seg_keep.push_back(true);
		// The keys at the cut; with CP shadowing their values come from S when the segment runs.
		if (cp_shadow) {
			EmitShadowPreamble(seg_pre.back(), stats, cx, sh, uc, cp);
		} else {
			EmitPreamble(seg_pre.back(), stats, cx, sh, uc, cp);
		}
		auto& s = segs.back();
		s.held  = true;
		s.cp_shadow_head = cp_shadow;
		if (cp_shadow) { // its preamble: S holds the GPU's value for the keys at the cut only
			vcx.reset();
			for (const auto& [off, value]: cx) {
				if (off < RegFile::kDense) {
					vcx.set(off);
				}
			}
			fmask_done = 0;
			dcc_done   = 0;
		}
		s.wait  = w;
		// No native form (not EQUAL, or a 64-bit wait on both halves), or an address the GPU cannot
		// reach: only seeing the label in memory releases it. A WAIT_REG_MEM there reads the dummy.
		// The native form polls one dword. That dword has to be resident; the rest of a 64-bit
		// wait is not what this packet reads. Built aside: a form that fails halfway leaves nothing behind.
		std::vector<uint32_t> native;
		const bool            has_native = label_gpu(w.addr, 4) && EmitNativeWait(native, w, m_compute_ring);
		// SetCpShadowWaitOff: the wrap's "off" before the barrier and its "on" after the guard, so neither patch range
		// (barrier_dw/len, wait_dw/len) holds them: whichever of the two LabelPipeline NOPs, the pair stays.
		const bool wrap = wait_wrap && (m_fence_va != 0 || has_native);
		if (wrap) {
			wait_off();
		}
		s.barrier_dw = static_cast<uint32_t>(body.size());
		emit_barrier_raw(w.pfp);
		s.barrier_len = static_cast<uint32_t>(body.size()) - s.barrier_dw;
		s.wait_dw     = static_cast<uint32_t>(body.size());
		if (has_native) {
			body.insert(body.end(), native.begin(), native.end());
		}
		s.wait_len = has_native ? static_cast<uint32_t>(native.size()) : 0u;
		guard_from = body.size();
		if (s.wait_len != 0) {
			guard_after_wait(!w.pfp); // a PFP wait's native form ends with PFP_SYNC_ME
		}
		if (wrap) {
			// A PFP wait's barrier and native form end with PFP_SYNC_ME, an ME wait's guard has one before it.
			wait_on(!w.pfp && !(gfx_guards && s.wait_len != 0));
		}
		seg_effects = true;
		stats.waits_split++;
	};

	// SetAudit: what refuses the buffer, each opcode and reason once with how often, and the first one's dwords.
	auto note_refusal = [&](uint32_t op, uint32_t at, RefuseWhy why) {
		if (!m_audit || why == RefuseWhy::Collected) {
			return;
		}
		const uint32_t key = op > 0xffu ? 0x100u : op;
		for (auto& f: r.refusals) {
			if (f.op == key && f.why == why) {
				f.count += f.last != at ? 1u : 0u;
				f.last = at;
				return;
			}
		}
		if (r.refusals.size() < 8 && at < dcb.size()) {
			Pm4Refusal f;
			f.op    = key;
			f.why   = why;
			f.count = 1;
			f.dw    = at;
			f.last  = at;
			f.nd    = static_cast<uint32_t>(std::min<size_t>(f.d.size(), dcb.size() - at));
			std::copy_n(dcb.begin() + at, f.nd, f.d.begin());
			r.refusals.push_back(f);
		}
	};
	auto fail = [&](uint32_t op, uint32_t dw, RefuseWhy why = RefuseWhy::Unknown) {
		note_refusal(op, dw, why);
		r.ok         = false;
		r.unknown_op = op;
		r.unknown_dw = dw;
		r.ib.clear();
		r.stats = stats;
		return r;
	};
	// A packet with no translation (yet): the walk goes on, so one refused buffer names every such packet, and the
	// buffer is refused at its end.
	auto soft_unknown = [&](uint32_t op, uint32_t dw) {
		if (r.unknown_ops.empty()) {
			r.unknown_op = op;
			r.unknown_dw = dw;
		}
		r.unknown_ops.insert(op);
		note_refusal(op, dw, RefuseWhy::Unknown);
	};

	uint32_t dw = 0;
	uint32_t verbatim_run = 0;
	auto     end_run      = [&]() {
        if (verbatim_run != 0) {
            stats.verbatim_runs++;
            stats.run_hist[verbatim_run < 16 ? 0 : verbatim_run < 64 ? 1 : verbatim_run < 256 ? 2 : 3]++;
            verbatim_run = 0;
        }
	};
	// SetDirectRuns: stretches of packets passed on as written, as (body offset, buffer dword, dwords).
	// The long ones run from the guest's buffer as IB2s when the IB is assembled below; their guest
	// memory is referenced so it is resident.
	// A run's packets run in place (in_place) are not in body; it is kept whatever its length, and a
	// short one is copied from the buffer when the IB is assembled.
	struct DirectRun {
		uint32_t body     = 0;
		uint32_t body_len = 0;
		uint32_t dw       = 0;
		uint32_t len      = 0;
		bool     in_place = false;
		bool     ib2      = false; // runs from the title's buffer (else copied from the translated one)
	};
	std::vector<DirectRun>         direct_runs;
	DirectRun                      direct_cur;
	const bool direct_base = m_direct_min_dw != 0 && !m_compute_ring && !source.empty();
	bool       direct      = direct_base; // off inside a title's COND_EXEC range (SetCondExec): its count is over our dwords
	// The GPU VA of buffer dword `at`, and how many dwords from there sit contiguously; 0 when unknown.
	auto source_va = [&](uint32_t at, uint32_t* left) -> uint64_t {
		auto it = std::upper_bound(source.begin(), source.end(), at,
		                           [](uint32_t v, const SourcePiece& piece) { return v < piece.dw; });
		if (it == source.begin()) {
			return 0;
		}
		--it;
		if (at >= it->dw + it->len) {
			return 0;
		}
		*left = it->dw + it->len - at;
		return it->va + 4ull * (at - it->dw);
	};
	// Whether every dword of buffer [dw, dw + len) has a known guest address.
	auto source_covers = [&](uint32_t dw, uint32_t len) {
		for (uint32_t done = 0; done < len;) {
			uint32_t left = 0;
			if (source_va(dw + done, &left) == 0) {
				return false;
			}
			done += std::min(left, len - done);
		}
		return true;
	};
	auto end_direct = [&]() {
		if (direct_cur.len >= m_direct_min_dw && source_covers(direct_cur.dw, direct_cur.len)) {
			for (uint32_t done = 0; done < direct_cur.len;) {
				uint32_t       left = 0;
				const uint64_t va   = source_va(direct_cur.dw + done, &left);
				const uint32_t take = std::min(left, direct_cur.len - done);
				add_ref(va, 4ull * take, false);
				done += take;
			}
			direct_cur.ib2 = true;
			direct_runs.push_back(direct_cur);
		} else if (direct_cur.in_place) {
			direct_runs.push_back(direct_cur);
		}
		direct_cur = {};
	};
	uint32_t cut_run = 0;
	auto     end_cut_run = [&]() {
		if (cut_run != 0) {
			stats.cut_runs++;
			cut_run = 0;
		}
	};
	stats.total_dw = static_cast<uint32_t>(dcb.size());
	// A title's COND_EXEC in any form (SetCondExec), gfx ring: if the dword at its address is 0 the CP skips EXEC_COUNT
	// dwords, which on the PS5 are the title's own. The translated range differs in length (patched packets, inserted
	// waits and stores), so the COND_EXEC goes as written with its count set to the translated range once the walk
	// reaches the range's end. Inside the range nothing may happen outside the GPU's stream: no direct runs (the IB2s
	// change the length again), no segment cut, no tracker store but a wake, no CPU-tail packet, no barrier on our fence, and no
	// register write (the shadow copy would take it as done when the CP may skip it). Label writes in it are recorded as
	// unknown once the range ends: the CP may have skipped them.
	struct CondExecRange {
		bool     active        = false;
		uint32_t dw            = 0; // the COND_EXEC in the buffer
		uint32_t end           = 0; // the buffer dword its range ends at
		size_t   count_at      = 0; // its EXEC_COUNT dword in body
		size_t   segs          = 0;
		size_t   seg_points    = 0;
		size_t   tail          = 0;
		size_t   writes        = 0; // local_writes when it began
		size_t   seg_writes    = 0; // segs.back().writes when it began
		uint32_t fence_seq     = 0;
	} cond;
	// False when something left the GPU's stream inside the range (see above). A wake that comes anyway is harmless, also
	// as a tracker store inside the range (the points a GPU wait there flushes first): the tracker passes a point once
	// its sequence or a later one is written (Reached), so a store the CP skips with the range is done at the next one.
	// The COND_EXEC flushes the points queued before it, so everything here was queued inside the range.
	auto cond_clean = [&]() {
		if (segs.size() != cond.segs || segs.back().points.size() < cond.seg_points || r.cpu_tail.size() != cond.tail ||
		    m_fence_seq != cond.fence_seq) {
			return false;
		}
		const auto wake = [](const TrackerPoint& p) { return p.kind == TrackerPoint::Kind::Wake; };
		return std::all_of(segs.back().points.begin() + static_cast<std::ptrdiff_t>(cond.seg_points), segs.back().points.end(), wake) &&
		       std::all_of(queued_points.begin(), queued_points.end(), wake);
	};
	auto cond_close = [&]() {
		const size_t count = body.size() - (cond.count_at + 1u);
		if (count > 0x3fffu) {
			return false;
		}
		body[cond.count_at] = (body[cond.count_at] & ~0x3fffu) | static_cast<uint32_t>(count);
		for (size_t i = cond.writes; i < local_writes.size(); i++) {
			local_writes[i].known = false;
		}
		for (size_t i = cond.seg_writes; i < segs.back().writes.size(); i++) {
			segs.back().writes[i].known = false;
		}
		cond.active = false;
		direct      = direct_base;
		return true;
	};
	// SetInjectWait (KYTY_NATIVE_GPU_INJECT_HANG): a wait nobody satisfies where the body starts, after the preamble, so
	// in load mode shadowing is on there, as at a title's own wait. The guarded form is what a guest wait the GPU gets
	// looks like (wrapped, a guard after it, in gpu_waits for the stop to release); the bare one is an unknown stuck point.
	if (m_inject_wait_va != 0 && labels) {
		WaitCond w;
		w.addr     = m_inject_wait_va;
		w.ref      = 1;
		w.mask     = 0xffffffffu;
		w.function = 3;
		if (m_inject_wait_bare && cp_shadow) {
			NoteOnce(40, "injected bare wait refused: CP shadowing is on (a shadowed hang without the wrap is fatal)");
		} else if (m_inject_wait_bare) {
			EmitNativeWait(body, w, m_compute_ring);
			r.injected  = true;
			seg_effects = true;
		} else {
			if (wait_wrap) {
				wait_off();
			}
			EmitNativeWait(body, w, m_compute_ring);
			segs.back().gpu_waits.push_back(w);
			guard_after_wait(true);
			if (wait_wrap) {
				wait_on(!gfx_guards);
			}
			r.injected  = true;
			seg_effects = true;
		}
	}
	// SetInjectShader: the dispatch goes where the injected wait would, with the compute state it needs; the guest's
	// compute registers it overwrites do not matter, since the title is stopped once the wave is found.
	if (m_inject_shader_va != 0 && labels) {
		auto set_sh = [&](uint32_t reg, std::initializer_list<uint32_t> v) {
			body.push_back(Pkt3(IT_SET_SH_REG, 1u + static_cast<uint32_t>(v.size()), true));
			body.push_back(reg);
			body.insert(body.end(), v.begin(), v.end());
		};
		set_sh(0x20c, {static_cast<uint32_t>(m_inject_shader_va >> 8u), static_cast<uint32_t>(m_inject_shader_va >> 40u)}); // PGM_LO/HI
		set_sh(0x212, {1u | 0xc0u << 12u | 1u << 21u | 1u << 30u, 0u}); // RSRC1: 8 VGPRs, denorms, DX10_CLAMP, MEM_ORDERED; RSRC2
		set_sh(0x228, {0u});                                            // RSRC3
		set_sh(0x204, {0u, 0u, 0u, 64u, 1u, 1u});                       // START_X..Z, NUM_THREAD_X..Z
		set_sh(0x215, {0u, 0xffffffffu, 0xffffffffu, 0u, 0xffffffffu, 0xffffffffu}); // RESOURCE_LIMITS, SE0/1, TMPRING, SE2/3
		body.push_back(Pkt3(IT_DISPATCH_DIRECT, 4, true));
		body.insert(body.end(), {1u, 1u, 1u, 1u | 1u << 2u | 1u << 6u}); // COMPUTE_SHADER_EN, FORCE_START_AT_0, ORDER_MODE
		r.injected  = true;
		seg_effects = true;
	}
	while (dw < dcb.size()) {
		if (cond.active && (!cond_clean() || (dw >= cond.end && (dw != cond.end || !cond_close())))) {
			return fail(IT_COND_EXEC, cond.dw, RefuseWhy::CondExecForm);
		}
		// A compute ring's guards skip a few thousand dwords at most: one more before the next packet.
		if (cs_guards && body.size() - guard_from > 3000u) {
			body_guards.push_back(EmitComputeExitGuard(body, m_exit_run_va));
			guard_from = body.size();
		}
		const size_t   body_before = body.size();
		const size_t   segs_before = segs.size();
		bool           in_place    = false; // passed on as written without a copy in body (direct runs)
		bool           reg_write   = false; // it writes registers: with CP shadowing its segment is not dropped
		const uint32_t header = dcb[dw];
		if ((header >> 30u) != 3u) {
			return fail(0xffffffffu, dw, RefuseWhy::NotType3); // not a type-3 packet
		}
		const uint32_t size = PacketSizeDw(header);
		if (size == 0 || dw + size > dcb.size()) {
			return fail((header >> 8u) & 0xffu, dw, RefuseWhy::Truncated);
		}
		const uint32_t op          = (header >> 8u) & 0xffu;
		cur_op                     = op;
		const uint32_t body_count  = size - 1u;
		const uint32_t* const bptr = dcb.data() + dw + 1;
		stats.packets++;
		// KYTY_NATIVE_GPU_WATCH_ADDR=0xADDR: log every packet whose body holds that 64-bit address
		// (any 8-aligned split into a low and a high dword), to find who writes or waits on a label.
		static const uint64_t watch_addr = [] {
			const char* v = std::getenv("KYTY_NATIVE_GPU_WATCH_ADDR");
			return v != nullptr ? std::strtoull(v, nullptr, 0) : 0ull;
		}();
		if (watch_addr != 0) {
			static std::atomic<uint32_t> watch_logged {0};
			for (uint32_t k = 0; k + 1 < body_count; k++) {
				if ((bptr[k] & ~7u) == static_cast<uint32_t>(watch_addr & ~7ull) && (bptr[k + 1] & 0xffffu) == (watch_addr >> 32u) &&
				    watch_logged.fetch_add(1) < 64) {
					std::fprintf(stderr, "watch addr: %s queue packet op 0x%02x marker 0x%02x len %u at dw %u:", m_compute_queue ? "compute" : "gfx",
					             op, (header >> 2u) & 0x3fu, size, dw);
					for (uint32_t i = 0; i < std::min(body_count, 10u); i++) {
						std::fprintf(stderr, " %08x", bptr[i]);
					}
					std::fprintf(stderr, "\n");
					break;
				}
			}
		}

		if (m_skip_ops.count(op) != 0) {
			stats.packets_skipped++;
			dw += size;
			continue;
		}
		if (!m_force_refuse.empty() && m_force_refuse.count(op) != 0) {
			soft_unknown(op, dw);
			dw += size;
			continue;
		}
		if (m_compute_ring) {
			// MEC has no graphics engine. These opcodes hang a compute ring ("Illegal opcode").
			// SET_UCONFIG_REG / INDEX / INDIRECT are handled below: profiler UCONFIG is dropped,
			// geometry-engine UCONFIG still refuses the buffer.
			switch (op) {
				case IT_DRAW_INDIRECT:
				case IT_DRAW_INDEX_INDIRECT:
				case IT_DRAW_INDIRECT_MULTI:
				case IT_DRAW_INDEX_INDIRECT_MULTI:
				case IT_DRAW_INDEX_2:
				case IT_DRAW_INDEX_OFFSET_2:
				case IT_DRAW_INDEX_AUTO:
				case IT_INDEX_BASE:
				case IT_INDEX_BUFFER_SIZE:
				case IT_SET_PREDICATION:
				case IT_CONTEXT_CONTROL:
				case IT_CLEAR_STATE:
				case IT_NUM_INSTANCES:
				case IT_PFP_SYNC_ME:
				case IT_SET_CONTEXT_REG:
				case IT_SET_CONTEXT_REG_INDIRECT:
				case IT_GET_LOD_STATS:
					if (r.unknown_ops.empty()) {
						r.unknown_op = op;
						r.unknown_dw = dw;
					}
					r.unknown_ops.insert(op);
					note_refusal(op, dw, RefuseWhy::GraphicsOnMec);
					dw += size;
					continue;
				default: break;
			}
		}
		// A title's EOP interrupt fires where its release runs, as on the PS5, not at the next GPU wait or the
		// segment's end: the points queued since are stored before the next GPU work (releases and labels right
		// after it still share the store).
		if (irq_pending) {
			switch (op) {
				case IT_DRAW_INDIRECT:
				case IT_DRAW_INDEX_INDIRECT:
				case IT_DRAW_INDIRECT_MULTI:
				case IT_DRAW_INDEX_INDIRECT_MULTI:
				case IT_DRAW_INDEX_2:
				case IT_DRAW_INDEX_OFFSET_2:
				case IT_DRAW_INDEX_AUTO:
				case IT_DISPATCH_DIRECT:
				case IT_DISPATCH_INDIRECT:
				case IT_DMA_DATA:
					flush_points();
					irq_pending = false;
					break;
				default: break;
			}
		}
		switch (op) {
			case IT_RELEASE_MEM: // libSceAgc's, as agc.cpp writes it: the same body as Kyty's NOP marker
			case IT_NOP: {
				stats.nops += op == IT_NOP ? 1 : 0;
				const uint32_t marker = op == IT_RELEASE_MEM ? R_RELEASE_MEM : (header >> 2u) & 0x3fu;
				if (m_audit && op == IT_NOP && marker != R_ZERO) {
					r.sync.marker_nops[marker]++;
				}
				if (m_audit && marker == R_RELEASE_MEM && body_count >= 2) {
					AuditRelease(r.sync, op == IT_NOP, bptr[0], bptr[1]);
				}
				if (marker == R_FLIP) {
					stats.marker_flip++;
					if (labels && m_eop_flips && body_count >= 5) {
						TrackerPoint flip;
						flip.kind = TrackerPoint::Kind::Flip;
						flip.data.assign(bptr, bptr + 5);
						tracker_point(std::move(flip));
						NoteOnce(22, "flip requested at the GPU's end of pipe (tracker point)");
					} else if (labels) {
						r.cpu_tail.insert(r.cpu_tail.end(), dcb.begin() + dw, dcb.begin() + dw + size);
					}
				} else if (marker == R_ZERO && labels && body_count >= 1 && IsFlipMarker(bptr[0])) {
					stats.marker_flip++;
					r.cpu_tail.insert(r.cpu_tail.end(), dcb.begin() + dw, dcb.begin() + dw + size);
					if (bptr[0] != 0x68750777u && body_count >= 4) { // writes a label first, on the CPU
						note_write({static_cast<uint64_t>(bptr[1]) | (static_cast<uint64_t>(bptr[2]) << 32u), bptr[3], true,
						            false, false});
					}
				} else if (marker == R_ZERO && body_count >= 1 && bptr[0] == kWaitFlipPadMagic) {
					// WaitUntilSafeForRendering's padding: Kyty's command processor waits there for the display buffer, as
					// for R_WAIT_FLIP_DONE, with the flip watermark the padding carries. Its WAIT_REG_MEM_64 stays out of
					// the IB (the IT_WAIT_REG_MEM_64 case).
					stats.marker_wait_flip++;
					if (labels) {
						const size_t at = r.cpu_tail.size();
						r.cpu_tail.insert(r.cpu_tail.end(), dcb.begin() + dw, dcb.begin() + dw + size);
						// On the PS5 the wait reads the label where it stands in the GPU's order: only the flips whose
						// release ran before it have written 1. A FlipMark point here notes which have; its token in
						// the copy lets the tracker put that into the padding (SetWaitFlipMarks) before Kyty runs it.
						if (body_count > kWaitFlipPadToken) {
							TrackerPoint mark;
							mark.kind = TrackerPoint::Kind::FlipMark;
							mark.ctx  = NextWaitFlipToken();
							r.cpu_tail[at + 1u + kWaitFlipPadToken] = mark.ctx;
							// The copy's mark is this run's: whatever the guest dwords still hold is not.
							r.cpu_tail[at + 1u + kWaitFlipPadEopMark]      = 0xffffffffu;
							r.cpu_tail[at + 1u + kWaitFlipPadEopMark + 1u] = 0xffffffffu;
							tracker_point(std::move(mark));
						}
					}
				} else if (marker == R_ZERO && body_count >= 1 && bptr[0] == kFlipPadMagic) {
					// SetFlip's padding is only padding: its RELEASE_MEM's interrupt (a kernel context id) does the flip.
				} else if (marker == R_RELEASE_MEM && labels) {
					stats.marker_release++;
					if (body_count < 7) {
						return fail(op, dw, RefuseWhy::ShortBody);
					}
					const uint64_t dst = static_cast<uint64_t>(bptr[2]) | (static_cast<uint64_t>(bptr[3]) << 32u);
					const uint64_t value = static_cast<uint64_t>(bptr[4]) | (static_cast<uint64_t>(bptr[5]) << 32u);
					const uint32_t data_sel = (bptr[1] >> 29u) & 0x7u;
					const uint32_t int_sel  = (bptr[1] >> 24u) & 0x7u;
					NoteReleaseForm(bptr[0], bptr[1], m_compute_queue);
					r.label_writes.emplace_back(dst, value);
					// What Kyty's command processor does with it (CpOpReleaseMem, WriteAtEndOfPipe): no data
					// for INT_SEL 4, nor for INT_SEL 1 on the graphics queue (the interrupt alone); 32-bit
					// (1) or 64-bit (2) data; the reference clock for a timestamp (3); GDS data (5).
					uint32_t sel = data_sel;
					if (dst == 0 || int_sel == 4 || (int_sel == 1 && !m_compute_queue && data_sel != 5)) {
						sel = 0;
					}
					if (sel == 5 && !m_gds) {
						if (r.unknown_ops.empty()) { // the refusal's log line names it
							r.unknown_op = op;
							r.unknown_dw = dw;
						}
						r.unknown_ops.insert(op); // GDS data without a GDS partition: leave the buffer to Kyty
						note_refusal(op, dw, RefuseWhy::ReleaseGds);
						break;
					}
					if (sel != 0 && sel != 1 && sel != 2 && sel != 3 && sel != 5) {
						r.unknown_detail = {bptr[0], bptr[1]};
						return fail(op, dw, RefuseWhy::ReleaseDataSel);
					}
					// The guest's packet as written: a DCB release in an end-of-pipe event the ring runs
					// (LabelEventOk). A data label goes as written only where the GPU may write it (label_gpu
					// below). With KYTY_NATIVE_GPU_RELEASES_AS_WRITTEN=0 (the form before 1.11) only one without
					// an interrupt whose data goes to memory (DST_SEL 0): libSceAgc's labels to L2 (DST_SEL 1) were
					// thought to stay there for good, which ReleasesAsWritten's probe disproves.
					// On this board GL2_WB does not write back the metadata cache, and GLM_INV without GLM_WB
					// drops it: ASTRO BOT's frames stayed black with its end-of-pipe GL2 write-backs and full
					// invalidates as written. Mesa never invalidates GLM without writing it back. Such a
					// release takes the rewritten form, which adds GLM_WB.
					//
					// With ReleasesAsWritten (the default) the DCB's release goes as the title wrote it, DST_SEL 1 and
					// 64-bit data included, with one change: the interrupt is cleared from it, raised by the tracker
					// instead (stock amdgpu
					// takes no guest interrupt from the gfx ring), which also keeps a kernel context (bit 27) away
					// from amdgpu. A release that writes no data (INT_SEL 1 or 4: the CP writes none then, probe
					// labelvis LABELVIS_INTSEL; or no address) goes without its DATA_SEL too: its event and cache
					// actions as written (libSceAgc's CACHE_FLUSH_AND_INV_TS interrupts flush the CB and DB).
					const uint32_t dst_sel    = (bptr[1] >> 16u) & 0x3u;
					const bool     glm_ok     = (bptr[0] & (kGcrGl2Wb | kGcrGlmInv)) == 0u || (bptr[0] & kGcrGlmWb) != 0u;
					const bool     interrupt  = int_sel == 1 || int_sel == 2 || int_sel == 4;
					const bool     whole      = ReleasesAsWritten();
					const uint32_t event_run  = whole && !glm_ok && ReleaseGlmWriteBack() ? bptr[0] | kGcrGlmWb : bptr[0];
					// An ACB's release goes as written too (KYTY_NATIVE_GPU_ACB_RELEASES_AS_WRITTEN=0: rewritten as before),
					// its header too on MEC (an ACB on the gfx ring loses the compute shader-type bit): MEC runs the events
					// libSceAgc uses there, and
					// DST_SEL 1 with BYPASS reaches the CPU from compute as from gfx (probe labelvis on compute).
					const bool     as_written = LabelsVerbatim(data_sel) && op == IT_RELEASE_MEM && size == 8u &&
					                        (whole ? (!m_compute_queue || AcbReleasesAsWritten()) && dst_sel <= 1u &&
					                                     (!interrupt || !m_guest_irq) && LabelEventOk(event_run)
					                               : !m_compute_queue && int_sel == 0 && glm_ok && LabelEventOk(bptr[0]) &&
					                                     (data_sel == 0 || (dst != 0 && dst_sel == 0)));
					// `with_interrupt`: the GPU raises the title's interrupt itself (the EOP log), as written.
					const bool data_kept = sel == data_sel;
					auto write_as_written = [&](bool with_interrupt = false) {
						const size_t at = body.size();
						body.insert(body.end(), dcb.begin() + dw, dcb.begin() + dw + size);
						if (!m_compute_ring) {
							body[at] = Pkt3(IT_RELEASE_MEM, 7, false); // an ACB on the gfx ring: no compute shader-type bit
						} else if (!MecPlainOp(IT_RELEASE_MEM)) {
							body[at] |= 2u; // KYTY_NATIVE_GPU_ACB_PLAIN without 0x49: the compute shader-type bit
						}
						if (data_kept && data_sel == 3) {
							body[at + 5] = 0u; // the GPU clock replaces the data dwords
							body[at + 6] = 0u;
						}
						body[at + 1] = event_run;
						if (!data_kept) {
							body[at + 2] &= ~(0x7u << 29u); // no data, as the CP writes none
						}
						if (!with_interrupt) {
							body[at + 2] &= ~(0x7u << 24u); // the interrupt is the tracker's (as_written holds none otherwise)
							body[at + 7] = 0u;              // and so is its context id
						}
						// CS_DONE/PS_DONE without data on the gfx ring (EosNoDataMode): the scratch store, or BOTTOM_OF_PIPE_TS.
						if (!m_compute_ring && ((body[at + 2] >> 29u) & 0x7u) == 0u && EndOfShaderEvent(body[at + 1]) &&
						    EosNoDataMode() != EosNoData::AsIs) {
							if (EosNoDataMode() == EosNoData::Data) {
								const uint64_t scratch = m_tracker_va + 8;
								body[at + 2] = (body[at + 2] & ~(0x3u << 16u)) | kData32Cntl; // DST_SEL 0 (memory), DATA_SEL 1
								if (((body[at + 2] >> 24u) & 0x7u) == 1u) {
									body[at + 2] = (body[at + 2] & ~(0x7u << 24u)) | kIntSel2; // the interrupt once the store is done
								}
								body[at + 3] = static_cast<uint32_t>(scratch);
								body[at + 4] = static_cast<uint32_t>(scratch >> 32u);
								body[at + 5] = 0u;
								body[at + 6] = 0u;
							} else {
								body[at + 1] = (body[at + 1] & ~0xf3fu) | 0x28u | (5u << 8u);
							}
							stats.eos_nodata_fixed++;
						}
					};
					TrackerPoint cpu;
					bool         cpu_point = false;
					if (sel == 3) {
						// Kyty's CPU timestamp is Sync::ReadReferenceClock(): the host TSC scaled to
						// 100 MHz. DATA_SEL 3 writes the GPU clock counter, also 100 MHz on this board
						// (gfx_v10_0's counter, the same kGpuClockHz the scratch samples use), but it
						// is not the host TSC: the zeros differ, deltas match. A game that compares the
						// value to a later ReadReferenceClock() needs the CPU clock, which
						// KYTY_NATIVE_GPU_CPU_TIMESTAMPS=1 keeps on the tracker.
						const bool gpu_clock = !m_cpu_timestamps && (dst & 7ull) == 0 && label_gpu(dst, 8);
						if (gpu_clock) {
							if (as_written) {
								write_as_written();
							} else {
								EmitTimestampRelease(body, dst, m_compute_ring);
							}
							note_write({dst, 0, false, true, true});
							stats.timestamps_native++;
							stats.labels_native++;
							seg_effects = true;
							NoteOnce(2, "RELEASE_MEM timestamp written by the GPU clock");
						} else {
							stats.timestamps++;
							cpu.kind  = TrackerPoint::Kind::Timestamp;
							cpu.addr  = dst;
							cpu.size  = 8;
							cpu_point = true;
							NoteOnce(3, "RELEASE_MEM timestamp written by the tracker (CPU clock)");
						}
						sel = 0;
					} else if ((sel == 1 || sel == 2) && !label_gpu(dst, sel == 1 ? 4ull : 8ull)) {
						cpu.kind  = TrackerPoint::Kind::CpuWrite;
						cpu.addr  = dst;
						cpu.data  = sel == 1 ? std::vector<uint32_t> {bptr[4]} : std::vector<uint32_t> {bptr[4], bptr[5]};
						cpu_point = true;
						sel       = 0;
					}
					// Otherwise not the guest's packet (an ACB's carried the compute shader-type bit, which stopped the
					// gfx ring with "Illegal opcode in command stream"): the label is written with the guest's
					// event and cache actions plus write-backs (LabelEventCntl), 32-bit data to memory; a
					// 64-bit one as two 32-bit writes, the high dword first. GDS data goes as the translator
					// always ran it. A release that only
					// flushes caches becomes a flush into our scratch dword, unless a tracker point (also a
					// flush) follows anyway.
					if ((sel == 1 || sel == 2) && as_written) {
						write_as_written();
						// The PS5 maps the memory its CPU shares with the GPU coherent (type 12, C_RW_S); stock amdgpu
						// maps it NC here, where the GPU's results stay in GL2. So a label the CPU may read gets the
						// write-backs LabelEventCntl adds, and they come off again where a wait of the same buffer
						// shows it is a GPU sync (StrippableWriteBack): Gran Turismo 7 froze without them.
						if (whole) {
							const size_t   ev    = body.size() - size + 1u;
							const uint32_t event = body[ev] | (LabelWriteBack() ? kGcrGlmWb | kGcrGl2Wb : 0u);
							body[ev]             = event;
							if (const uint32_t strip = StrippableWriteBack(bptr[0], event); strip != 0u) {
								label_releases.push_back({local_writes.size(), ev, strip});
							}
						}
						mirror_store(dst, std::span<const uint32_t>(bptr + 4, sel), false);
						NoteOnce(12, "RELEASE_MEM label as the title wrote it");
					} else if (sel == 1) {
						emit_label(dst, bptr[4], bptr[0]);
						mirror_store(dst, std::span<const uint32_t>(bptr + 4, 1), false);
						NoteOnce(0, m_tracker_writes ? "RELEASE_MEM 32-bit label (GPU store, mirrored to the CPU)"
						                             : "RELEASE_MEM 32-bit label (GPU store only)");
					} else if (sel == 2) {
						emit_label(dst + 4, bptr[5], bptr[0]);
						emit_label(dst, bptr[4], bptr[0]);
						mirror_store(dst, std::span<const uint32_t>(bptr + 4, 2), false);
						NoteOnce(1, m_tracker_writes ? "RELEASE_MEM 64-bit label (GPU store, mirrored to the CPU)"
						                             : "RELEASE_MEM 64-bit label (GPU store only, two halves)");
					} else if (sel == 5) {
						body.push_back(Pkt3(IT_RELEASE_MEM, 7, m_compute_ring));
						body.push_back(bptr[0]);
						body.push_back(bptr[1] & ~(0x7u << 24u)); // INT_SEL 0
						for (uint32_t i = 2; i < 7; i++) {
							body.push_back(i == 6 ? 0u : bptr[i]);
						}
					} else if (!cpu_point && (data_sel == 0 || !data_kept) && sel != 5 && as_written) {
						write_as_written(); // cache actions and the event alone (an interrupt is the tracker's)
					} else if (!interrupt && !cpu_point && ((bptr[0] >> 12u) & 0x1fffu) != 0) {
						EmitRelease32(body, m_tracker_va + 8, 0u, m_compute_ring, LabelEventCntl(bptr[0]));
					}
					seg_effects = true;
					if (sel != 0) {
						if (sel == 5) {
							add_ref(dst, 8, false); // GDS data; a memory label is referenced in label_gpu
						}
						note_write({dst, sel == 1 ? (value & 0xffffffffull) : value, sel != 5, true, sel == 2});
						stats.labels_native++;
						stats.gds_releases += sel == 5 ? 1u : 0u;
						if (sel == 5) {
							NoteOnce(4, "RELEASE_MEM GDS data (GPU store, no CPU mirror)");
						}
					} else if (data_sel == 0 && !interrupt && !cpu_point) {
						NoteOnce(5, "RELEASE_MEM with no data and no interrupt (cache action only)");
					}
					if (cpu_point) {
						note_write({dst, cpu.data.size() == 1 ? (value & 0xffffffffull) : value, cpu.kind != TrackerPoint::Kind::Timestamp,
						            false, cpu.data.size() != 1});
						tracker_point(std::move(cpu));
						stats.labels_cpu++;
					}
					// A title's interrupt with a context id the GPU can raise itself (SetGuestInterruptsOnGpu): as
					// written when it carries no data, else after the label, into the scratch dword.
					const uint32_t guest_ctx  = bptr[6];
					const bool     irq_on_gpu = interrupt && m_guest_irq && op == IT_RELEASE_MEM && !cpu_point &&
					                        (guest_ctx != 0 || m_guest_irq_ctx0) && guest_ctx < (1u << 27u);
					if (irq_on_gpu) {
						if (!m_compute_queue && size == 8u && int_sel == 1 && data_sel == 0 && glm_ok && LabelEventOk(bptr[0])) {
							write_as_written(true);
						} else {
							EmitRelease32(body, m_tracker_va + 8, 0u, m_compute_ring, LabelEventCntl(bptr[0]));
							body[body.size() - 6] |= kIntSel2;
							body.back() = guest_ctx;
						}
						stats.interrupts_gpu++;
						NoteOnce(21, "RELEASE_MEM interrupt raised by the GPU (EOP log)");
					} else if (interrupt) {
						// All 28 bits of the context id, as the PS5's CP hands them to the kernel: bit 27 marks a kernel
						// context (SetFlip's, from sceVideoOutSubmitEopFlip), which the host's interrupt dispatch takes
						// for itself instead of the title's event queues.
						TrackerPoint irq;
						irq.kind = TrackerPoint::Kind::Interrupt;
						irq.ctx  = bptr[6] & 0x0fffffffu;
						tracker_point(irq);
						irq_pending = m_irq_at_release;
						stats.interrupts++;
						NoteOnce(16u + int_sel, int_sel == 1 ? "RELEASE_MEM INT_SEL 1 (EOP interrupt)"
						                                     : (int_sel == 2 ? "RELEASE_MEM INT_SEL 2 (EOP interrupt)"
						                                                     : "RELEASE_MEM INT_SEL 4 (interrupt only)"));
					}
					// SetAudit: a CACHE_POLICY no GPU packet carries (approximately): a tracker point stands for the
					// release (a CPU write, a CPU timestamp, a tracker interrupt), a timestamp in our form, or
					// LabelEventCntl's fallback to the barrier's event. GDS data keeps the guest's event dword.
					if (m_audit && (bptr[0] & kReleasePolicy) != 0u && !as_written && sel != 5 &&
					    (cpu_point || data_sel == 3 || (interrupt && !irq_on_gpu) ||
					     (LabelEventCntl(bptr[0]) & kReleasePolicy) != (bptr[0] & kReleasePolicy))) {
						r.sync.policy_lost[static_cast<size_t>(op == IT_NOP ? AuditSync::ReleaseMarker : AuditSync::Release)]++;
					}
				} else if (marker == R_RELEASE_MEM) {
					stats.marker_release++;
					if (body_count >= 4) {
						r.label_writes.emplace_back(static_cast<uint64_t>(bptr[2]) | (static_cast<uint64_t>(bptr[3]) << 32u),
						                            static_cast<uint64_t>(bptr[4]) | (static_cast<uint64_t>(bptr[5]) << 32u));
					}
					// Kyty's marker carries a hardware RELEASE_MEM. One that sends GDS data (DATA_SEL 5:
					// counters the guest reads back) must run here, where the GDS partition lives: Kyty
					// would copy its own, never written, GDS buffer instead. Without the interrupt, which
					// Kyty still raises.
					if (body_count >= 7 && ((bptr[1] >> 29u) & 0x7u) == 5u && m_gds) {
						body.push_back(Pkt3(IT_RELEASE_MEM, 7, m_compute_ring));
						body.push_back(bptr[0]);
						body.push_back(bptr[1] & ~(0x7u << 24u)); // INT_SEL 0
						for (uint32_t i = 2; i < 7; i++) {
							body.push_back(i == 6 ? 0u : bptr[i]);
						}
						add_ref(static_cast<uint64_t>(bptr[2]) | (static_cast<uint64_t>(bptr[3]) << 32u), 4, false);
						stats.gds_releases++;
					}
				} else if (marker == R_WAIT_FLIP_DONE) {
					stats.marker_wait_flip++;
					if (labels) {
						r.cpu_tail.insert(r.cpu_tail.end(), dcb.begin() + dw, dcb.begin() + dw + size);
					}
				} else if (marker == R_CONTEXT_STATE && body_count >= 1) {
					context_state_op(bptr[0]);
					reg_write = true;
				}
				// A plain NOP (AgcCbNop, workload markers) runs in place with direct runs, as the PS5's CP skips
				// it: dropping it cut the title's run in two (ASTRO BOT: 44k -> 63k dwords a flip run from the
				// title's buffer). Kyty's debug markers and dispatch reset carry marker bits in header bits 7:2;
				// in place they hung ASTRO BOT's gfx ring before its first frame, so they stay dropped
				// (KYTY_NATIVE_GPU_NOPS_IN_PLACE=2 runs them too). =0 drops every NOP as before.
				static const int nops_in_place = [] {
					const char* v = std::getenv("KYTY_NATIVE_GPU_NOPS_IN_PLACE");
					return v == nullptr ? 1 : std::atoi(v);
				}();
				if (op == IT_NOP && nops_in_place != 0 && direct && !m_compute_ring && body.size() == body_before && segs.size() == segs_before &&
				    ((marker == R_ZERO && (body_count == 0 || !IsFlipMarker(bptr[0]))) ||
				     (nops_in_place >= 2 && (marker == R_PUSH_MARKER || marker == R_POP_MARKER || marker == R_DISPATCH_RESET)))) {
					in_place = true;
				}
				break; // the CP ignores NOPs; present/sync markers are handled elsewhere later
			}

			case IT_SET_CONTEXT_REG: {
				// With direct runs a packet no value of which is patched runs in place, without a copy.
				bool           unchanged = true;
				const uint32_t base      = body_count >= 1 ? Normalize(bptr[0]) : 0u;
				for (uint32_t i = 1; i < body_count; i++) {
					cx[base + (i - 1)] = bptr[i];
					mark_cx(base + (i - 1));
					writer(wcx, base + (i - 1), RegWriter::Set);
					unchanged = unchanged && PatchContext(base + (i - 1), bptr[i]) == bptr[i];
				}
				reg_write = body_count >= 2;
				if (direct && unchanged) {
					in_place = true;
				} else {
					const size_t at = body.size();
					body.insert(body.end(), dcb.begin() + dw, dcb.begin() + dw + size);
					for (uint32_t i = 1; i < body_count; i++) {
						body[at + 1 + i] = PatchContext(base + (i - 1), bptr[i]);
					}
				}
				stats.set_context++;
				break;
			}

			case IT_SET_SH_REG: {
				// With direct runs a packet no value of which is patched runs in place, without a copy.
				bool unchanged = !m_compute_ring;
				if (body_count >= 1) {
					const uint32_t base = bptr[0];
					const uint32_t last = base + body_count - 2;
					for (uint32_t i = 1; i < body_count; i++) {
						sh[base + (i - 1)] = bptr[i];
						mark_sh(base + (i - 1));
						writer(wsh, base + (i - 1), RegWriter::Set);
						unchanged = unchanged && PatchSh(base + (i - 1), bptr[i]) == bptr[i];
					}
					reg_write = body_count >= 2;
					// A shader address needs its PGM_LO and PGM_HI: once per range that touches either.
					for (uint32_t lo: {0x8u, 0x88u, 0xC8u, 0x108u, 0x148u, 0x20Cu}) {
						if (lo + 1 >= base && lo <= last) {
							note_shader(lo);
						}
					}
				}
				if (direct && unchanged) {
					in_place = true;
				} else {
					body.insert(body.end(), dcb.begin() + dw, dcb.begin() + dw + size);
					if (m_compute_ring) {
						body[body.size() - size] |= 2u;
					}
					if (body_count >= 1) {
						const size_t at = body.size() - size;
						for (uint32_t i = 1; i < body_count; i++) {
							body[at + 1 + i] = PatchSh(bptr[0] + (i - 1), bptr[i]);
						}
					}
				}
				stats.set_sh++;
				break;
			}

			case IT_SET_UCONFIG_REG:
			case IT_SET_UCONFIG_REG_INDEX: {
				// Pass through only whitelisted UCONFIG registers; drop the rest for board safety.
				// On a compute ring the whitelist is geometry-engine state (no GE on MEC), so it
				// refuses the buffer. SQ_THREAD_TRACE_USERDATA_0..7 (0x340..0x347) and every other
				// non-whitelisted offset are dropped (a compute ring stops on 0x342, probe --danger tag342).
				// Refusing the opcode refused every ACB: they write 0x342 and 0x343 as profiler markers.
				if (m_compute_ring && body_count >= 1) {
					const uint32_t base = (op == IT_SET_UCONFIG_REG_INDEX) ? (bptr[0] & 0x0fffffffu)
					                                                        : Normalize(bptr[0]);
					bool graphics = false;
					for (uint32_t i = 1; i < body_count; i++) {
						if (UconfigAllowed(base + (i - 1))) {
							graphics = true;
						}
					}
					if (graphics) {
						if (r.unknown_ops.empty()) {
							r.unknown_op = op;
							r.unknown_dw = dw;
						}
						r.unknown_ops.insert(op);
						note_refusal(op, dw, RefuseWhy::UconfigOnMec);
						break;
					}
				}
				// libSceAgc's internal-data tags (SQ_THREAD_TRACE_USERDATA_2, around waits and multi draws; a
				// 64-bit wait's begin tag goes on into USERDATA_3) go to the gfx CP as written: a user IB writes
				// them there (probe --danger tag342), while a compute ring stops on the same packet.
				if (!m_compute_ring && op == IT_SET_UCONFIG_REG && (body_count == 2 || body_count == 3) &&
				    Normalize(bptr[0]) == 0x342u) {
					body.insert(body.end(), dcb.begin() + dw, dcb.begin() + dw + size);
					stats.set_uconfig++;
					break;
				}
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
							writer(wuc, base + (i - 1), RegWriter::Set);
						}
						reg_write = body_count >= 2;
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
					return fail(op, dw, RefuseWhy::ShortBody);
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
				reg_write = true;
				const auto* pairs = reinterpret_cast<const uint32_t*>(addr);
				// The gfx CP runs these packets itself (AMD's LOAD_CONTEXT/SH/UCONFIG_REG_INDEX, bit for bit;
				// probe --danger regindex-*: the table is loaded). The packet goes as written when every pair
				// is one the translator would emit unchanged: no stub entries (the CP would write them),
				// no patched values, UCONFIG only whitelisted and not VGT_PRIMITIVE_TYPE/INDEX_TYPE (those
				// need SET_UCONFIG_REG_INDEX). SH tables may hold compute registers under a graphics-type
				// header, as libSceAgc's own SET_SH_REG does for DCB dispatches (My First GT; probe --danger
				// regindex-shc-nr runs it). The shadow copy is kept from the same pairs.
				// KYTY_NATIVE_GPU_TABLES_VERBATIM=0 expands every table as before.
				static const bool tables_verbatim = [] {
					const char* v = std::getenv("KYTY_NATIVE_GPU_TABLES_VERBATIM");
					return v == nullptr || v[0] != '0';
				}();
				// VGT_PRIMITIVE_TYPE/INDEX_TYPE in a UCONFIG table go as written too: the CP takes them from the
				// table as the PS5's does (AgcCreatePrimState puts the primitive type in every such table; My First
				// GT, ASTRO BOT and Dead Cells draw right). KYTY_NATIVE_GPU_TABLES_PRIM=0 expands those tables.
				static const bool tables_prim = [] {
					const char* v = std::getenv("KYTY_NATIVE_GPU_TABLES_PRIM");
					return v == nullptr || v[0] != '0';
				}();
				bool verbatim = tables_verbatim && !m_compute_ring && bptr[2] == 0x80000000u && (bptr[0] & 3u) == 0;
				for (uint32_t i = 0; i < num && verbatim; i++) {
					const uint32_t raw   = pairs[2 * i];
					const uint32_t value = pairs[2 * i + 1];
					const uint32_t off   = Normalize(raw);
					if (raw != off) {
						verbatim = false; // also the 0xffffffff stub
					} else if (op == IT_SET_CONTEXT_REG_INDIRECT) {
						verbatim = PatchContext(off, value) == value;
					} else if (op == IT_SET_SH_REG_INDIRECT) {
						verbatim = off != kShNop && PatchSh(off, value) == value;
					} else {
						verbatim = off != kUcNop && (tables_prim || (off != 0x242u && off != 0x243u)) && UconfigAllowed(off);
					}
				}
				if (verbatim) {
					body.insert(body.end(), dcb.begin() + dw, dcb.begin() + dw + size);
					add_ref(addr, 8ull * num, false); // the CP reads the table
					stats.loads_verbatim++;
				}
				const bool check_written = !verbatim && addr < gpu_written_hi && gpu_written_lo < addr + 8ull * num;
				for (size_t k = gpu_written.size(); check_written && k > 0 && gpu_written.size() - k < 256; k--) {
					const auto [lo, hi] = gpu_written[k - 1];
					if (addr < hi && lo < addr + 8ull * num) {
						stats.load_gpu_written++;
						if (static std::atomic<uint32_t> logged {0}; logged++ < 8) {
							std::fprintf(stderr,
							             "native shadow: indirect register load 0x%02x reads 0x%llx..0x%llx, which this DCB writes "
							             "earlier (0x%llx..0x%llx); the table is read before the GPU writes it\n",
							             op, static_cast<unsigned long long>(addr), static_cast<unsigned long long>(addr + 8ull * num),
							             static_cast<unsigned long long>(lo), static_cast<unsigned long long>(hi));
						}
						break;
					}
				}
				const RegWriter table_writer = verbatim ? RegWriter::Table : RegWriter::Expanded;
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
						writer(wcx, off, table_writer);
						if (!verbatim) {
							EmitSet(body, IT_SET_CONTEXT_REG, off, PatchContext(off, value), false);
						}
						stats.loaded_cx_regs++;
					} else if (op == IT_SET_SH_REG_INDIRECT) {
						if (off == kShNop) {
							continue;
						}
						sh[off] = value;
						mark_sh(off);
						note_shader(off);
						writer(wsh, off, table_writer);
						if (!verbatim) {
							EmitSet(body, IT_SET_SH_REG, off, PatchSh(off, value), off >= kShComputeBase);
						}
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
						if (m_compute_ring) {
							if (r.unknown_ops.empty()) {
								r.unknown_op = op;
								r.unknown_dw = dw;
							}
							r.unknown_ops.insert(op);
							note_refusal(op, dw, RefuseWhy::UconfigOnMec);
							continue;
						}
						uc[off] = value;
						writer(wuc, off, table_writer);
						if (!verbatim) {
							EmitSetUconfig(body, off, value);
						}
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
				// WaitUntilSafeForRendering: the PS5's GPU waits here until the display buffer's label is 0 (off
				// screen). Kyty's flip is a Vulkan copy on this same gfx pipe, so a GPU wait could hold up the very flip
				// that frees the buffer. Kyty's command processor waits instead, at the padding NOP that follows (the
				// IT_NOP case puts it in the CPU tail), as for R_WAIT_FLIP_DONE; the wait stays out of the IB.
				if (wide && WaitFlipPadFollows(dcb, static_cast<size_t>(dw) + size)) {
					stats.waits_flip++;
					NoteOnce(23, "WaitUntilSafeForRendering's label wait left to Kyty's command processor (CPU tail)");
					break;
				}
				if (m_audit && body_count >= 1) {
					AuditWait(r.sync, wide, bptr[0]);
				}
				if (body_count < need) {
					return fail(op, dw, RefuseWhy::ShortBody);
				}
				if ((bptr[0] & 0x10u) == 0) {
					return fail(op, dw, RefuseWhy::RegisterWait); // register waits are refused
				}
				const uint64_t addr = (static_cast<uint64_t>(bptr[1]) & (wide ? ~0x7ull : ~0x3ull)) |
				                      (static_cast<uint64_t>(bptr[2] & 0x3ffffu) << 32u);
				uint64_t ref  = bptr[3];
				uint64_t mask = wide ? bptr[5] : bptr[4];
				if (wide) {
					ref |= static_cast<uint64_t>(bptr[4]) << 32u;
					mask |= static_cast<uint64_t>(bptr[6]) << 32u;
				}
				bool     met   = false;
				uint64_t value = 0;
				if (addr != 0) {
					value = wide ? *reinterpret_cast<const volatile uint64_t*>(addr)
					             : *reinterpret_cast<const volatile uint32_t*>(addr);
					met   = WaitConditionMet(bptr[0] & 0x7u, value & mask, ref);
				}
				stats.waits_dropped++;
				if (!met) {
					stats.waits_unmet++;
					r.unmet_waits.push_back({addr, ref, value, bptr[0] & 0x7u, dw, stats.draws, stats.dispatches});
				}
				const bool pfp = (bptr[0] & (1u << 8u)) != 0;
				if (labels) {
					// The latest label write of this buffer decides; otherwise the backend does, from what is
					// ahead in the ring and from guest memory.
					WaitCond w {addr, ref, mask, bptr[0] & 0x7u, wide, pfp};
					LabelOracle::Plan plan    = LabelOracle::Plan::Split;
					bool              decided = false;
					size_t            decider = 0; // the local write that decides the wait
					for (auto it = local_writes.rbegin(); it != local_writes.rend() && !decided; ++it) {
						uint64_t  v = 0;
						const int c = it->Covers(addr, wide, &v, mask);
						if (c != 0) {
							decided = true;
							decider = static_cast<size_t>(local_writes.rend() - it) - 1u;
							plan    = c == 1 && w.Met(v) && it->gpu ? LabelOracle::Plan::Native : LabelOracle::Plan::Split;
						}
					}
					const bool own_label = decided && plan == LabelOracle::Plan::Native;
					if (!decided) {
						plan = m_labels->PlanWait(w);
					}
					if (plan == LabelOracle::Plan::Native && !m_native_waits) {
						plan = LabelOracle::Plan::Met;
					}
					// A native wait reads whatever the GPU is mapped at. If that is not this label,
					// hold the segment: the tracker (or a later CPU write) puts the value in guest
					// memory, and the barrier is the only GPU wait.
					if (plan == LabelOracle::Plan::Native && !label_gpu(addr, 4)) {
						plan = LabelOracle::Plan::Split;
					}
					if (plan == LabelOracle::Plan::Native) {
						flush_points();
					}
					const bool timed = plan == LabelOracle::Plan::Native && m_time_pairs != 0 && !m_compute_ring;
					const uint64_t wait_pair = timed ? m_time_wait_va + 16ull * (m_time_waits % m_time_pairs) : 0u;
					// The guest's packet as written where the gfx CP runs it (SetWaitsVerbatim), else our form; built aside, so
					// a form that fails halfway leaves nothing behind and the wrap goes only around a wait that is there.
					const bool as_written = plan == LabelOracle::Plan::Native && (dcb[dw] & 2u) == 0 &&
					                        (m_compute_ring ? MecGuestWaitOk(bptr[0]) : m_waits_verbatim && GuestWaitControlOk(bptr[0]));
					std::vector<uint32_t> wait_pk;
					if (as_written) {
						wait_pk.assign(dcb.begin() + dw, dcb.begin() + dw + size);
					}
					// SetAudit: the title's CACHE_POLICY, lost unless the GPU runs its own packet (our wait, a barrier, a hold).
					if (m_audit && ((bptr[0] >> 25u) & 3u) != 0u && !as_written) {
						r.sync.policy_lost[static_cast<size_t>(wide ? AuditSync::Wait64 : AuditSync::Wait)]++;
					}
					const bool emitted = plan == LabelOracle::Plan::Native && (as_written || EmitNativeWait(wait_pk, w, m_compute_ring));
					// SetCpShadowWaitOff: the clock copies and the guard inside the wrap, nothing that SETs or LOADs.
					const bool wrap = emitted && wait_wrap;
					if (wrap) {
						wait_off();
					}
					if (timed) {
						EmitClockCopy(body, wait_pair);
					}
					if (emitted) {
						body.insert(body.end(), wait_pk.begin(), wait_pk.end());
						segs.back().gpu_waits.push_back(w);
						guard_after_wait(!pfp);
					}
					if (timed) {
						EmitClockCopy(body, wait_pair + 8);
						m_time_wait_info[m_time_waits % m_time_pairs] = {addr, ref, stats.draws, met};
						m_time_waits++;
					}
					if (wrap) {
						wait_on(!pfp && !gfx_guards); // an ME wait's guard has PFP_SYNC_ME before it
					}
					if (plan == LabelOracle::Plan::Native && !emitted) {
						plan = LabelOracle::Plan::Split;
					}
					if (plan == LabelOracle::Plan::Native && own_label) {
						// A GPU sync within this buffer: the release that wrote the label it waits for need not
						// write GL2 back for the CPU.
						for (auto& rel: label_releases) {
							if (rel.strip != 0u && rel.write == decider) {
								body[rel.event_dw] &= ~rel.strip;
								rel.strip = 0u;
								stats.label_wb_stripped++;
							}
						}
					}
					// SetAudit: a wait the GPU runs, or a barrier, with the index of the first point after it (flush_points
					// has run for a GPU wait). A held one is the next segment's `wait`.
					if (m_audit && plan != LabelOracle::Plan::Split) {
						segs.back().audit_waits.push_back(
						    {w, static_cast<uint32_t>(segs.back().points.size()), plan == LabelOracle::Plan::Met, own_label, met});
					}
					switch (plan) {
						case LabelOracle::Plan::Native:
							stats.waits_native++;
							seg_effects = true;
							NoteOnce(8, "guest WAIT_REG_MEM passed through (a GPU write satisfies it)");
							break;
						case LabelOracle::Plan::Met:
							emit_barrier(pfp);
							seg_effects = true;
							NoteOnce(10, "guest WAIT_REG_MEM already met (barrier)");
							break;
						case LabelOracle::Plan::Split:
							split_at_wait(w);
							NoteOnce(9, "guest WAIT_REG_MEM held on the CPU (no known writer yet)");
							break;
					}
					break;
				}
				// The guest waits for its own end-of-pipe label: everything before has finished and its
				// writes are visible. The label itself is written by Kyty later, so give the GPU the same
				// guarantee with a fence of ours.
				if (m_audit && ((bptr[0] >> 25u) & 3u) != 0u) {
					r.sync.policy_lost[static_cast<size_t>(wide ? AuditSync::Wait64 : AuditSync::Wait)]++;
				}
				emit_barrier(pfp);
				break;
			}
			case IT_DMA_DATA:
				if (m_find_hi > m_find_lo && body_count >= 6) {
					const uint64_t dst = static_cast<uint64_t>(bptr[3]) | (static_cast<uint64_t>(bptr[4]) << 32u);
					if (dst >= m_find_lo && dst < m_find_hi) {
						FindHit h;
						h.index  = stats.dma_data;
						h.target = dst;
						h.depth  = 99; // a DMA_DATA destination
						r.find_hits.push_back(h);
					}
				}
				if (body_count < 6 || !DmaDataAllowed(bptr[0], bptr[5], m_gds)) {
					r.unknown_detail = {bptr[0], body_count >= 6 ? bptr[5] : 0u};
					return fail(op, dw, RefuseWhy::DmaForm);
				}
				{
					const uint32_t src_sel = (bptr[0] >> 29u) & 0x3u;
					const uint32_t dst_sel = (bptr[0] >> 20u) & 0x3u;
					if (m_gds_dma_sync && src_sel == 1u) {
						body.push_back((3u << 30u) | (IT_EVENT_WRITE << 8u) | (m_compute_ring ? 2u : 0u)); // CS_PARTIAL_FLUSH
						body.push_back((4u << 8u) | 7u);
					}
					body.insert(body.end(), dcb.begin() + dw, dcb.begin() + dw + size);
					if (m_compute_ring) {
						body[body.size() - size] |= 2u;
					}
					if (m_gds_dma_sync && (src_sel == 1u || dst_sel == 1u)) {
						body[body.size() - size + 1] |= 1u << 31u; // CP_SYNC
						stats.gds_dma_synced++;
					}
					cp_dma_busy = (body[body.size() - size + 1] & (1u << 31u)) == 0;
				}
				stats.dma_data++;
				{
					const uint64_t bytes = bptr[5] & 0x1fffffu;
					const uint32_t src_sel = (bptr[0] >> 29u) & 0x3u; // 1 GDS, 2 immediate data
					const uint32_t dst_sel = (bptr[0] >> 20u) & 0x3u; // 1 GDS
					stats.dma_gds_dst += dst_sel == 1u ? 1u : 0u;
					stats.dma_gds_src += src_sel == 1u ? 1u : 0u;
					if (dst_sel == 1u) {
						r.gds_dma.emplace_back(bptr[3] | 0x80000000u, static_cast<uint32_t>(bytes));
					}
					if (src_sel == 1u) {
						r.gds_dma.emplace_back(bptr[1], static_cast<uint32_t>(bytes));
					}
					if (src_sel != 1u && src_sel != 2u) {
						add_ref(static_cast<uint64_t>(bptr[1]) | (static_cast<uint64_t>(bptr[2]) << 32u), bytes, false);
					}
					if (dst_sel != 1u) {
						const uint64_t dst = static_cast<uint64_t>(bptr[3]) | (static_cast<uint64_t>(bptr[4]) << 32u);
						add_ref(dst, bytes, false);
						note_gpu_written(dst, dst + bytes);
					}
				}
				break;
			case IT_WRITE_DATA:
				if (m_audit && body_count >= 1) {
					AuditWriteData(r.sync, bptr[0], body_count >= 3 && TraceUserDataWrite(bptr[0], bptr[1], bptr[2]));
				}
				// A debug marker's string (libSceAgc, to SQ_THREAD_TRACE_USERDATA_3): as written on the gfx ring, as
				// its 0x342 tag; MEC stops on thread-trace registers, so a compute ring drops it.
				if (body_count >= 3 && TraceUserDataWrite(bptr[0], bptr[1], bptr[2])) {
					if (!m_compute_ring && direct) {
						in_place = true;
					} else if (!m_compute_ring) {
						body.insert(body.end(), dcb.begin() + dw, dcb.begin() + dw + size);
					}
					break;
				}
				// Games write their completion labels with WRITE_DATA. The native IB completes before
				// Kyty has even parsed this DCB, so a native label tells the game the GPU is done while
				// Kyty is still reading the command buffer, and the game starts rewriting it (Dead
				// Cells: a truncated packet in Kyty's parser on entering gameplay). Kyty performs the
				// write itself, in order, so the native IB leaves it out.
				if (body_count < 3) {
					return fail(op, dw, RefuseWhy::ShortBody);
				}
				if (!WriteDataToMemory(bptr[0])) {
					return fail(op, dw, RefuseWhy::WriteDataRegister);
				}
				{
					const uint64_t dst = static_cast<uint64_t>(bptr[1]) | (static_cast<uint64_t>(bptr[2]) << 32u);
					const uint64_t n   = ((bptr[0] >> 16u) & 1u) != 0 ? 1u : body_count - 3u;
					note_gpu_written(dst, dst + 4u * n);
				}
				if (labels) {
					// Native labels: Kyty no longer reads the buffer once it is translated, so the GPU writes
					// the label where the guest put it. Memory the GPU cannot write is written by the
					// tracker once the GPU gets there.
					const uint64_t dst      = static_cast<uint64_t>(bptr[1]) | (static_cast<uint64_t>(bptr[2]) << 32u);
					const uint32_t n        = body_count - 3u;
					const bool     one_addr = ((bptr[0] >> 16u) & 1u) != 0;
					const uint32_t engine   = bptr[0] >> 30u;
					const uint64_t bytes    = one_addr ? 4u : 4u * static_cast<uint64_t>(n);
					if (n == 0 || dst == 0) {
						break;
					}
					// As the guest's WRITE_DATA, in order: the micro engine writes it when it gets there, and a
					// dispatch right after reads it (My First Gran Turismo builds a dispatch's constants with
					// WRITE_DATA, then dispatches). As end-of-pipe releases the data landed after that dispatch
					// had read it, and longer writes went to the tracker, after the whole segment. The write is
					// confirmed before the engine moves on; a compute ring has only the micro engine.
					// A constant-engine write (ENGINE_SEL 2) is the micro engine's here: the native IB is
					// not a CE IB, and the ME writes it at its place in the stream, where the tracker would
					// only write it after the whole segment. KYTY_NATIVE_GPU_TRACKER_WRITES=1 leaves it to
					// the tracker as before.
					const bool gpu = (engine <= 1 || !m_tracker_writes) && label_gpu(dst, bytes);
					if (gpu && direct && (bptr[0] & (1u << 20u)) != 0 && engine <= 1) {
						// The title's own packet is what the GPU gets: with direct runs it is not copied (half
						// of ASTRO BOT's DCB dwords are such blocks), the IB runs it from the title's buffer.
						in_place = true;
					} else if (gpu) {
						const size_t at = body.size();
						body.insert(body.end(), dcb.begin() + dw, dcb.begin() + dw + size);
						body[at + 1] |= 1u << 20u; // WR_CONFIRM
						if (m_compute_ring) {
							body[at] |= 2u;                 // SHADER_TYPE compute
						}
						if (m_compute_ring || engine > 1) {
							body[at + 1] &= ~(3u << 30u);   // ENGINE_SEL micro engine
						}
					}
					if (gpu) {
						mirror_store(dst, std::span<const uint32_t>(bptr + 3, n), one_addr);
						stats.labels_native++;
						stats.write_data_dw += n;
						seg_effects = true;
						NoteOnce(6, m_tracker_writes ? "WRITE_DATA label (native WRITE_DATA, mirrored to the CPU)"
						                             : "WRITE_DATA label (native WRITE_DATA only)");
						if (engine > 1) {
							NoteOnce(11, "WRITE_DATA by the constant engine (run on the micro engine)");
						}
					}
					// Two dwords at an 8-aligned address are one 64-bit label as far as a later
					// WAIT_REG_MEM_64 is concerned (Agc's WaitOnAddress after a 64-bit WriteData: Dead
					// Cells). Recorded as two 32-bit writes, the wait reads them as unknown and is held on
					// the CPU although the write ahead of it in the ring satisfies it.
					if (!one_addr && n > 2) {
						LabelWrite block {dst, 0, true, gpu, false};
						block.span_dw = n;
						block.data    = bptr + 3;
						note_write(block);
					}
					for (uint32_t i = 0; (one_addr || n <= 2) && i < std::min<uint32_t>(n, 64);) {
						const uint64_t at = one_addr ? dst : dst + 4u * i;
						if (!one_addr && i + 1 < n && (at & 7u) == 0) {
							note_write({at, bptr[3 + i] | (static_cast<uint64_t>(bptr[4 + i]) << 32u), true, gpu, true});
							i += 2;
							continue;
						}
						note_write({at, bptr[3 + i], true, gpu, false});
						i++;
					}
					if (!gpu) {
						TrackerPoint cpu;
						cpu.kind     = TrackerPoint::Kind::CpuWrite;
						cpu.addr     = dst;
						cpu.one_addr = one_addr;
						cpu.data.assign(bptr + 3, bptr + 3 + n);
						tracker_point(std::move(cpu));
						stats.labels_cpu++;
						NoteOnce(7, "WRITE_DATA label written by the tracker (GPU cannot reach it)");
					}
					r.label_writes.emplace_back(dst, bptr[3]);
					break;
				}
				stats.write_data_dropped++;
				r.label_writes.emplace_back(static_cast<uint64_t>(bptr[1]) | (static_cast<uint64_t>(bptr[2]) << 32u), bptr[3]);
				if (body_count > 4) {
					r.data_writes.emplace_back(static_cast<uint64_t>(bptr[1]) | (static_cast<uint64_t>(bptr[2]) << 32u),
					                           body_count - 3);
				}
				if (m_write_data_native == 2 || (m_write_data_native == 1 && body_count > 4)) {
					body.insert(body.end(), dcb.begin() + dw, dcb.begin() + dw + size);
					stats.write_data_native++;
				}
				break;
			case IT_MEM_SEMAPHORE: {
				// libSceAgc's AgcDcbMemSemaphore / AgcAcbMemSemaphore: a signal (SEM_SEL 6) increments the mailbox counter at
				// the address, a wait (7) blocks until it is not 0 and decrements it. Whether the BC-250's stock microcode
				// runs 0x39 at all is not known (Mesa sends it only on SI/CIK), and a wait whose signal comes from another
				// ring or the CPU is invisible to the label pipeline; there is no exactly equivalent GPU form (a wait for
				// >= 1 and an ATOMIC_MEM SUB are not atomic against a second waiter). Probe memsem (2.11) answered the first
				// question: the stock ME took the whole board down on a signal, with no kernel line. So it refuses the
				// buffer; KYTY_NATIVE_GPU_MEM_SEMAPHORE=1 passes it as written for research only.
				if (!m_mem_semaphore || body_count < 3) {
					static std::atomic<uint32_t> told {0};
					if (body_count >= 3 && told.fetch_add(1, std::memory_order_relaxed) < 8) {
						std::fprintf(stderr, "native shadow: MEM_SEMAPHORE %08x %08x %08x refused (KYTY_NATIVE_GPU_MEM_SEMAPHORE)\n", bptr[0],
						             bptr[1], bptr[2]);
					}
					soft_unknown(op, dw);
					break;
				}
				// libSceAgc's form only (agc_mem_semaphore): three dwords, ADDR_LO bits 31:3, ADDR_HI, then SEM_SEL (31:29) 6 or
				// 7, SIGNAL_TYPE (20) and USE_MAILBOX (16), CLIENT_CODE (25:24) 0. Any other select or bit is not a packet
				// the PS5's libSceAgc writes, and what the stock microcode makes of it is not known.
				const uint32_t sem_sel = bptr[2] >> 29u;
				const uint64_t addr    = (bptr[0] & ~7u) | (static_cast<uint64_t>(bptr[1]) << 32u);
				if (body_count != 3 || (sem_sel != 6u && sem_sel != 7u) || (bptr[2] & ~((7u << 29u) | (1u << 20u) | (1u << 16u))) != 0 ||
				    m_compute_ring || !labels || !label_gpu(addr, 8)) {
					r.unknown_detail = {bptr[0], bptr[1], bptr[2]};
					return fail(op, dw, RefuseWhy::SemaphoreForm);
				}
				if (direct) {
					in_place = true;
				} else {
					body.insert(body.end(), dcb.begin() + dw, dcb.begin() + dw + size);
				}
				note_gpu_written(addr, addr + 8);
				note_write({addr, 0, false, true, true}); // the counter, both a signal and a wait change it
				seg_effects = true;
				NoteOnce(33, "MEM_SEMAPHORE as the title wrote it");
				break;
			}
			case IT_DRAW_INDEX_MULTI_INST: {
				// AgcDcbDrawIndexMultiInstanced (0xC0073A00: initiator | 0x80, an object-id array) is Sony microcode: the
				// stock ME has no 0x3A it runs, and the per-instance DRAW_INDEX_2 equivalent needs the shader's object-id
				// user-data slot, still research. Refused; its arguments are logged for when a title calls it.
				static std::atomic<uint32_t> told {0};
				if (told.fetch_add(1, std::memory_order_relaxed) < 8) {
					std::fprintf(stderr, "native shadow: DrawIndexMultiInstanced (0x3A) refused:");
					for (uint32_t i = 0; i < std::min<uint32_t>(body_count, 8u); i++) {
						std::fprintf(stderr, " %08x", bptr[i]);
					}
					std::fprintf(stderr, "\n");
				}
				soft_unknown(op, dw);
				break;
			}
			case IT_COPY_DATA: {
				// libSceAgc's AgcDcbCopyData / AgcAcbCopyData: memory (SRC_SEL 1, 2), an immediate (5) or the GPU clock (9)
				// into memory (DST_SEL 1, 2, 5), 32 or 64 bits (COUNT_SEL). As written where the GPU may write the
				// destination, as the PS5's CP runs it; an immediate into memory it cannot write is the tracker's CPU store,
				// the clock the tracker's CPU timestamp as for a RELEASE_MEM (KYTY_NATIVE_GPU_CPU_TIMESTAMPS=1 too). Register,
				// GDS and perf-counter selects and the atomic return data refuse: the first stop the ring from a user IB
				// (probe: COPY_DATA of UCONFIG/context), the last needs the CP's internal register. Only with
				// KYTY_NATIVE_GPU_COPY_DATA=1 (MEC: KYTY_NATIVE_GPU_MEC_COPY_DATA=1) until probe copydata has run it.
				if (!labels || (m_compute_ring ? !m_mec_copy_data : !m_copy_data)) {
					soft_unknown(op, dw);
					break;
				}
				if (body_count < 5) {
					return fail(op, dw, RefuseWhy::ShortBody);
				}
				const uint32_t src_sel = bptr[0] & 0xfu;
				const uint32_t dst_sel = (bptr[0] >> 8u) & 0xfu;
				const bool     wide    = ((bptr[0] >> 16u) & 1u) != 0;
				const uint64_t bytes   = wide ? 8u : 4u;
				const bool     src_mem = src_sel == 1u || src_sel == 2u;
				const bool     imm     = src_sel == 5u;
				const bool     clock   = src_sel == 9u;
				const uint64_t src     = static_cast<uint64_t>(bptr[1]) | (static_cast<uint64_t>(bptr[2]) << 32u);
				const uint64_t dst     = static_cast<uint64_t>(bptr[3]) | (static_cast<uint64_t>(bptr[4]) << 32u);
				// ENGINE_SEL (bits 31:30, gfx ring): libSceAgc's DCB form picks the micro engine or the prefetch parser (bit 30,
				// AgcDcbCopyData), both of which run it as written; the constant engine (2) is not in a DCB's stream, and MEC
				// has no engine field (AgcAcbCopyData leaves the bits 0). A 64-bit copy's addresses hold bits 31:3 only
				// (DST_64B_ADDR_LO, SRC_64B_ADDR_LO): the CP would write elsewhere than the label bookkeeping and the tracker's
				// fallbacks assume, so an address not aligned to the copy's size refuses.
				const uint32_t engine = bptr[0] >> 30u;
				if (!(src_mem || imm || clock) || !(dst_sel == 1u || dst_sel == 2u || dst_sel == 5u) || dst == 0 || (dst & (bytes - 1u)) != 0 ||
				    (src_mem && (src == 0 || (src & (bytes - 1u)) != 0)) || engine > (m_compute_ring ? 0u : 1u)) {
					r.unknown_detail = {bptr[0], bptr[1], bptr[2], bptr[3], bptr[4]};
					return fail(op, dw, RefuseWhy::CopyDataForm);
				}
				const uint64_t value = imm ? (wide ? src : (src & 0xffffffffull)) : 0u;
				if (clock && (m_cpu_timestamps || (dst & 7u) != 0 || !label_gpu(dst, 8))) {
					TrackerPoint cpu;
					cpu.kind = TrackerPoint::Kind::Timestamp;
					cpu.addr = dst;
					cpu.size = static_cast<uint32_t>(bytes);
					tracker_point(std::move(cpu));
					note_write({dst, 0, false, false, wide});
					r.label_writes.emplace_back(dst, 0u); // this buffer writes the label, as a WRITE_DATA the tracker stores
					stats.timestamps++;
					NoteOnce(30, "COPY_DATA of the GPU clock written by the tracker (CPU clock)");
					break;
				}
				const bool gpu = label_gpu(dst, bytes);
				if (!gpu && imm) {
					TrackerPoint cpu;
					cpu.kind = TrackerPoint::Kind::CpuWrite;
					cpu.addr = dst;
					cpu.data = wide ? std::vector<uint32_t> {bptr[1], bptr[2]} : std::vector<uint32_t> {bptr[1]};
					tracker_point(std::move(cpu));
					note_write({dst, value, true, false, wide});
					r.label_writes.emplace_back(dst, value);
					stats.labels_cpu++;
					NoteOnce(31, "COPY_DATA immediate written by the tracker (GPU cannot reach it)");
					break;
				}
				if (!gpu) {
					// Memory to memory the GPU cannot write: the tracker would read the source before the GPU work ahead
					// of the copy has written it.
					r.unknown_detail = {bptr[0], bptr[1], bptr[2], bptr[3], bptr[4]};
					return fail(op, dw, RefuseWhy::CopyDataForm);
				}
				if (direct && !m_compute_ring) {
					in_place = true;
				} else {
					const size_t at = body.size();
					body.insert(body.end(), dcb.begin() + dw, dcb.begin() + dw + size);
					if (m_compute_ring) {
						body[at] |= 2u; // SHADER_TYPE compute
					}
				}
				if (src_mem) {
					add_ref(src, bytes, false);
				}
				note_gpu_written(dst, dst + bytes);
				note_write({dst, value, imm, true, wide});
				if (imm) {
					mirror_store(dst, std::span<const uint32_t>(bptr + 1, wide ? 2u : 1u), false);
				} else {
					TrackerPoint wake; // a guest asleep on the label is woken once the GPU has written it
					wake.kind = TrackerPoint::Kind::Wake;
					wake.addr = dst;
					tracker_point(std::move(wake));
				}
				if (clock) {
					stats.timestamps_native++;
				}
				stats.labels_native++;
				seg_effects = true;
				r.label_writes.emplace_back(dst, value);
				NoteOnce(32, "COPY_DATA as the title wrote it");
				break;
			}
			case IT_DRAW_INDEX_2:
				log_draw(op, body_count >= 4 ? bptr[3] : 0, 0, false);
				if (skip_draw()) {
					stats.draws_skipped++;
				} else {
					body.insert(body.end(), dcb.begin() + dw, dcb.begin() + dw + size);
				}
				stats.draws++;
				collect_refs();
				if (body_count >= 3) { // MAX_SIZE (indices), INDEX_BASE_LO, INDEX_BASE_HI, ...
					add_ref(static_cast<uint64_t>(bptr[1]) | (static_cast<uint64_t>(bptr[2]) << 32u),
					        static_cast<uint64_t>(bptr[0]) * 4u, false, true);
				}
				break;
			case IT_SET_PREDICATION: {
				// Draws that follow run or not by the result of an occlusion or primitive-count query at
				// the address. libSceAgc writes the gfx10 form (flags, address low, address high) or the
				// older one (address low, flags with address bits 39:32), as Kyty's CpOpSetPredication
				// tells them apart; the native IB always gets the gfx10 form.
				if (body_count < 2) {
					return fail(op, dw, RefuseWhy::ShortBody);
				}
				constexpr uint32_t kFlagsMask = 0x00071100u; // PRED_BOOL, HINT, PRED_OP
				uint32_t           flags      = 0;
				uint64_t           addr       = 0;
				if (body_count >= 3 && (bptr[0] & ~kFlagsMask) == 0 && bptr[2] <= 0xffffu) {
					flags = bptr[0];
					addr  = (static_cast<uint64_t>(bptr[2]) << 32u) | (bptr[1] & 0xfffffff0u);
				} else {
					flags = bptr[1] & kFlagsMask;
					addr  = (bptr[0] & 0xfffffff0u) | (static_cast<uint64_t>(bptr[1] & 0xffu) << 32u);
				}
				body.push_back(Pkt3(IT_SET_PREDICATION, 3, false));
				body.push_back(flags);
				body.push_back(static_cast<uint32_t>(addr));
				body.push_back(static_cast<uint32_t>(addr >> 32u));
				add_ref(addr, 64u, false);
				NoteOnce(18, "SET_PREDICATION (draws predicated on a query result)");
				break;
			}
			case IT_INDEX_BASE: // the index buffer for draws that do not carry their own
				if (body_count < 2) {
					return fail(op, dw, RefuseWhy::ShortBody);
				}
				body.insert(body.end(), dcb.begin() + dw, dcb.begin() + dw + size);
				index_base = static_cast<uint64_t>(bptr[0]) | (static_cast<uint64_t>(bptr[1] & 0xffffu) << 32u);
				cp.index_base = static_cast<uint64_t>(bptr[0]) | (static_cast<uint64_t>(bptr[1]) << 32u);
				cp.index_set  = true;
				add_ref(index_base, index_bytes != 0 ? index_bytes : kTextureSpan, false, true);
				stats.index_state++;
				break;
			case IT_INDEX_BUFFER_SIZE:
				body.insert(body.end(), dcb.begin() + dw, dcb.begin() + dw + size);
				index_bytes = static_cast<uint64_t>(bptr[0]) * 4u;
				if (body_count >= 1) {
					cp.index_size = bptr[0];
					cp.index_set  = true;
				}
				add_ref(index_base, index_bytes, false);
				stats.index_state++;
				break;
			case IT_DRAW_INDEX_OFFSET_2: { // MAX_SIZE, INDEX_OFFSET, INDEX_COUNT, DRAW_INITIATOR
				// As written (SetDrawIndexOffsetVerbatim, the native default): the BC-250's CP draws it once the
				// queue resets set the index buffer up as libSceAgc's do (AgcDcbResetQueue op 0x10: INDEX_BASE 0,
				// INDEX_BUFFER_SIZE 0xffffffff). With Kyty's former reset (a CLEAR_STATE) ASTRO BOT was black
				// with the packet as written, which looked like a firmware gap. Otherwise DRAW_INDEX_2 at the
				// offset's address, as Kyty's own command processor does.
				if (body_count < 4) {
					return fail(op, dw, RefuseWhy::ShortBody);
				}
				log_draw(op, bptr[2], 0, false);
				if (skip_draw()) {
					stats.draws_skipped++;
				} else if (m_draw_offset_verbatim) {
					body.insert(body.end(), dcb.begin() + dw, dcb.begin() + dw + size);
				} else {
					const uint32_t index_type  = reg(uc, 0x243u) & 0x3u; // VGT_INDEX_TYPE: 0 16-bit, 1 32-bit, 2 8-bit
					const uint64_t index_bytes = index_type == 1u ? 4u : (index_type == 2u ? 1u : 2u);
					const uint64_t addr        = index_base + static_cast<uint64_t>(bptr[1]) * index_bytes;
					body.push_back(Pkt3(IT_DRAW_INDEX_2, 5, false));
					body.push_back(bptr[0]);
					body.push_back(static_cast<uint32_t>(addr));
					body.push_back(static_cast<uint32_t>(addr >> 32u));
					body.push_back(bptr[2]);
					body.push_back(bptr[3]);
				}
				stats.draws++;
				collect_refs();
				add_ref(index_base, index_bytes != 0 ? index_bytes : kTextureSpan, false, true);
				break;
			}
			case IT_DRAW_INDIRECT:       // arguments at the SET_BASE address + DATA_OFFSET; same on the gfx ring
			case IT_DRAW_INDEX_INDIRECT: // the same, indexed through INDEX_BASE / INDEX_BUFFER_SIZE
				log_draw(op, 0, log_set_base + (body_count >= 1 ? bptr[0] : 0u), false);
				if (skip_draw()) {
					stats.draws_skipped++;
				} else {
					pfp_sync_me();
					body.insert(body.end(), dcb.begin() + dw, dcb.begin() + dw + size);
				}
				stats.draws++;
				collect_refs();
				if (op == IT_DRAW_INDEX_INDIRECT) {
					add_ref(index_base, index_bytes != 0 ? index_bytes : kTextureSpan, false, true);
				}
				break;
			// AgcDcbDraw[Index]IndirectMulti: up to MAX_COUNT argument records STRIDE bytes apart from the SET_BASE
			// base + DATA_OFFSET, and with COUNT_INDIRECT_ENABLE (bit 30 of the fourth dword) the count read
			// from COUNT_ADDR. The BC-250's CP runs the packet as written, as it does the single forms.
			case IT_DRAW_INDIRECT_MULTI:
			case IT_DRAW_INDEX_INDIRECT_MULTI: {
				if (body_count < 9) {
					return fail(op, dw, RefuseWhy::ShortBody);
				}
				const uint64_t args = log_set_base + bptr[0];
				log_draw(op, 0, args, false);
				if (skip_draw()) {
					stats.draws_skipped++;
				} else {
					pfp_sync_me();
					body.insert(body.end(), dcb.begin() + dw, dcb.begin() + dw + size);
				}
				stats.draws++;
				collect_refs();
				add_ref(args, std::min<uint64_t>(std::max<uint64_t>(static_cast<uint64_t>(bptr[4]) * bptr[7], bptr[7]), kMetadataSpan),
				        false);
				if ((bptr[3] & (1u << 30u)) != 0) {
					add_ref((static_cast<uint64_t>(bptr[5]) & ~3ull) | (static_cast<uint64_t>(bptr[6] & 0xffffu) << 32u), 4, false);
				}
				if (op == IT_DRAW_INDEX_INDIRECT_MULTI) {
					add_ref(index_base, index_bytes != 0 ? index_bytes : kTextureSpan, false, true);
				}
				break;
			}
			case IT_GET_LOD_STATS: // texture LOD statistics, which the game's texture streaming reads
				// BU_SIZE == 0 is the per-frame disable: no buffer the streamer reads, so it must not
				// close the window (those samples belong to the next real report).
				if (m_lod_scan && body_count >= 3 && bptr[0] != 0) {
					LodBankScan scan = TakeLodScan();
					scan.addr        = (static_cast<uint64_t>(bptr[1]) & ~0x3full) | (static_cast<uint64_t>(bptr[2]) << 32u);
					r.lod_scans.push_back(std::move(scan));
				}
				if (m_lod_stats && body_count >= 4) {
					// A gfx10 packet (PAL: GET_LOD_STATS): BU_SIZE, BASE_LO (64-byte aligned), BASE_HI, and
					// the interval / reset control word.
					body.insert(body.end(), dcb.begin() + dw, dcb.begin() + dw + size);
					add_ref((static_cast<uint64_t>(bptr[1]) & ~0x3full) | (static_cast<uint64_t>(bptr[2] & 0xffffu) << 32u),
					        bptr[0] != 0 ? bptr[0] : 64u, false);
					stats.lod_stats_issued++;
				} else {
					stats.lod_stats_dropped++;
					// Native labels: Kyty's command processor never sees the packet, so the tracker fills the
					// statistics as Kyty would (its first dword reads as a label: 1).
					const uint64_t dst = (static_cast<uint64_t>(bptr[1]) & ~0x3full) | (static_cast<uint64_t>(bptr[2]) << 32u);
					if (labels && body_count >= 3 && dst != 0 && bptr[0] != 0) {
						TrackerPoint fill;
						fill.kind = TrackerPoint::Kind::LodStats;
						fill.addr = dst;
						fill.size = bptr[0];
						tracker_point(std::move(fill));
						note_write({dst, 1u, true, false, false});
					}
				}
				break;
			case IT_DRAW_INDEX_AUTO: // a non-indexed draw: no index buffer to reference
				log_draw(op, body_count >= 1 ? bptr[0] : 0, 0, false);
				if (skip_draw()) {
					stats.draws_skipped++;
				} else {
					body.insert(body.end(), dcb.begin() + dw, dcb.begin() + dw + size);
				}
				stats.draws++;
				collect_refs();
				break;
			case IT_SET_BASE: // BASE_INDEX, address: where indirect draw/dispatch arguments live
				if (body_count < 3) {
					return fail(op, dw, RefuseWhy::ShortBody);
				}
				body.insert(body.end(), dcb.begin() + dw, dcb.begin() + dw + size);
				if (m_compute_ring) {
					body[body.size() - size] |= 2u;
				}
				add_ref(static_cast<uint64_t>(bptr[1]) | (static_cast<uint64_t>(bptr[2] & 0xffffu) << 32u),
				        kMetadataSpan, false);
				log_set_base = static_cast<uint64_t>(bptr[1]) | (static_cast<uint64_t>(bptr[2] & 0xffffu) << 32u);
				if (!m_compute_ring && (bptr[0] & 0xfu) == 1u) {
					cp.draw_base = static_cast<uint64_t>(bptr[1]) | (static_cast<uint64_t>(bptr[2]) << 32u);
				}
				stats.set_base++;
				break;
			case IT_DISPATCH_INDIRECT: // arguments at the SET_BASE address + DATA_OFFSET
				log_draw(op, 0,
				         m_compute_queue && body_count >= 2 ? (static_cast<uint64_t>(bptr[0]) | (static_cast<uint64_t>(bptr[1] & 0xffffu) << 32u))
				                                           : log_set_base + (body_count >= 1 ? bptr[0] : 0u),
				         true);
				if (m_compute_ring) {
					// The guest packet already carries ADDR_LO, ADDR_HI, DISPATCH_INITIATOR. MEC reads
					// that form; the gfx-ring SET_BASE rewrite would be the wrong engine.
					if (body_count < 3) {
						return fail(op, dw, RefuseWhy::ShortBody);
					}
					body.push_back(dcb[dw] | 2u);
					body.insert(body.end(), dcb.begin() + dw + 1, dcb.begin() + dw + size);
					add_ref(static_cast<uint64_t>(bptr[0]) | (static_cast<uint64_t>(bptr[1] & 0xffffu) << 32u), kMetadataSpan,
					        false);
				} else {
					pfp_sync_me();
					if (m_compute_queue) {
						// A compute-queue buffer carries the argument address itself (ADDR_LO, ADDR_HI,
						// DISPATCH_INITIATOR); the gfx ring reads DATA_OFFSET from a SET_BASE base instead,
						// so set the base to that address and dispatch at offset 0, as RADV does.
						if (body_count < 3) {
							return fail(op, dw, RefuseWhy::ShortBody);
						}
						const uint64_t args = static_cast<uint64_t>(bptr[0]) | (static_cast<uint64_t>(bptr[1] & 0xffffu) << 32u);
						body.push_back(Pkt3(IT_SET_BASE, 3, true));
						body.push_back(1u); // BASE_INDEX: DISPATCH_INDIRECT / DRAW_INDIRECT base
						body.push_back(static_cast<uint32_t>(args));
						body.push_back(static_cast<uint32_t>(args >> 32u));
						body.push_back(Pkt3(IT_DISPATCH_INDIRECT, 2, true));
						body.push_back(0u);
						body.push_back(bptr[2]);
						add_ref(args, kMetadataSpan, false);
					} else {
						body.insert(body.end(), dcb.begin() + dw, dcb.begin() + dw + size);
					}
				}
				stats.dispatches++;
				collect_refs();
				break;
			case IT_PFP_SYNC_ME:
				body.insert(body.end(), dcb.begin() + dw, dcb.begin() + dw + size);
				stats.pfp_sync_me++;
				break;
			case IT_NUM_INSTANCES: // instance count for the following draws
				body.insert(body.end(), dcb.begin() + dw, dcb.begin() + dw + size);
				log_instances = body_count >= 1 ? bptr[0] : 1u;
				cp.num_instances = log_instances;
				stats.num_instances++;
				break;
			case IT_DISPATCH_DIRECT:
				log_draw(op, body_count >= 1 ? bptr[0] : 0, 0, true);
				// KYTY_NATIVE_GPU_INSPECT_PS names a compute shader too: its user data, as for a draw.
				if (const uint64_t cs = (static_cast<uint64_t>(reg(sh, 0x20Cu)) << 8u) |
				                        (static_cast<uint64_t>(reg(sh, 0x20Du) & 0xffu) << 40u);
				    !m_inspect.empty() && m_inspect.count(cs) != 0 && r.inspected.size() < 4) {
					DrawSnapshot snap;
					snap.ps = cs;
					for (uint32_t i = 0; i < 16; i++) {
						snap.ps_user_data[i] = reg(sh, 0x240u + i);
					}
					r.inspected.push_back(snap);
				}
				body.insert(body.end(), dcb.begin() + dw, dcb.begin() + dw + size);
				if (m_compute_ring) {
					body[body.size() - size] |= 2u;
				}
				stats.dispatches++;
				collect_refs();
				break;
			case IT_ACQUIRE_MEM:
				if (m_audit) {
					AuditAcquire(r.sync, std::span<const uint32_t>(bptr, body_count));
				}
				if (m_acquire_full && !m_compute_queue) {
					body.push_back((3u << 30u) | (IT_EVENT_WRITE << 8u)); // FLUSH_AND_INV_CB_META, index 0
					body.push_back(0x2eu);
					body.push_back((3u << 30u) | (IT_EVENT_WRITE << 8u)); // FLUSH_AND_INV_DB_META, index 0
					body.push_back(0x2cu);
				}
				body.insert(body.end(), dcb.begin() + dw, dcb.begin() + dw + size);
				if (m_acquire_full && body_count >= 7) {
					uint32_t* a = &body[body.size() - size + 1];
					a[1] = 0xffffffffu; // COHER_SIZE: everything
					a[2] = 0xffu;       // COHER_SIZE_HI
					a[3] = 0u;          // COHER_BASE
					a[4] = 0u;          // COHER_BASE_HI
					// GCR_CNTL: GLI_INV=1, GLM_WB, GLM_INV, GLK_WB, GLK_INV, GLV_INV, GL1_INV, GL2_INV, GL2_WB.
					a[6] |= 0x1u | 0x10u | 0x20u | 0x40u | 0x80u | 0x100u | 0x200u | 0x4000u | 0x8000u;
					a[6] &= ~((0x3u << 2u) | (0x3u << 11u)); // GL1_RANGE, GL2_RANGE = all
				}
				if (m_compute_ring) {
					body[body.size() - size] |= 2u;
				}
				stats.acquire_mem++;
				break;
			case IT_EVENT_WRITE:
				body.insert(body.end(), dcb.begin() + dw, dcb.begin() + dw + size);
				if (m_compute_ring) {
					body[body.size() - size] |= 2u;
				}
				// KYTY_NATIVE_GPU_ZPASS_SYNC=1 (experiment): an occlusion dump (PIXEL_PIPE_STAT_DUMP 0x39,
				// ZPASS_DONE 0x15) lands at the end of the pixel pipe; wait for it before what follows reads the
				// counters (a dispatch that turns them into draw arguments).
				if (static const bool zpass_sync = std::getenv("KYTY_NATIVE_GPU_ZPASS_SYNC") != nullptr;
				    zpass_sync && !m_compute_ring && body_count >= 1 && ((bptr[0] & 0x3fu) == 0x39u || (bptr[0] & 0x3fu) == 0x15u)) {
					emit_barrier(true);
				}
				// PIXEL_PIPE_STAT_DUMP (0x39) goes as the title wrote it: after PIXEL_PIPE_STAT_CONTROL (the
				// preamble) the BC-250's DB writes one 64-bit counter per RB, 16 bytes apart (STRIDE 2), bit 63
				// set, as the PS5's. KYTY_NATIVE_GPU_ZPASS_DUMP=1 sends ZPASS_DONE (0x15) instead, which writes the
				// same counters.
				if (static const bool zpass_dump = [] {
					    const char* e = std::getenv("KYTY_NATIVE_GPU_ZPASS_DUMP");
					    return e != nullptr && e[0] == '1';
				    }();
				    zpass_dump && !m_compute_ring && body_count >= 3 && (bptr[0] & 0x3fu) == 0x39u) {
					body[body.size() - size + 1] = (bptr[0] & ~0x3fu) | 0x15u;
				}
				// The draw log keeps occlusion dumps with their address, so a flip dump can show the counters.
				if (body_count >= 3 && ((bptr[0] & 0x3fu) == 0x39u || (bptr[0] & 0x3fu) == 0x15u)) {
					log_draw(op, bptr[0] & 0x3fu, (static_cast<uint64_t>(bptr[1]) & ~7ull) | (static_cast<uint64_t>(bptr[2] & 0xffffu) << 32u), false);
				}
				// Events with an address (ZPASS_DONE occlusion counters, 16 DBs x begin/end) write there. For
				// PIXEL_PIPE_STAT_CONTROL (0x38, libSceAgc's DcbResetQueue op 0x200 and the preamble) dwords 1-2 are the
				// control word (COUNTER_ID, STRIDE, INSTANCE_EN), and PIXEL_PIPE_STAT_RESET (0x3a) writes nothing either.
				if (body_count >= 3 && (bptr[0] & 0x3fu) != 0x38u && (bptr[0] & 0x3fu) != 0x3au) {
					add_ref((static_cast<uint64_t>(bptr[1]) & ~7ull) | (static_cast<uint64_t>(bptr[2] & 0xffffu) << 32u), 256, false);
				}
				// The CP only forwards EVENT_TYPE to VGT_EVENT_INITIATOR; CB/DB do the actual CMASK/FMASK/
				// HTILE flush. Record which the title issues so the squares work can see any CB/DB meta flush.
				if (body_count >= 1) {
					stats.event_type[bptr[0] & 0x3fu]++;
					stats.event_index[(bptr[0] >> 8u) & 0xfu]++;
				}
				stats.event_write++;
				if (body_count >= 1) {
					NoteEvent(bptr[0] & 0x3fu);
				}
				break;
			case IT_CONTEXT_CONTROL:
				// libSceAgc's around a context push or pop turns on register shadowing, which needs the kernel's
				// shadow memory; the translator keeps the context in its shadow and sets its own preamble.
				if (m_compute_ring) {
					return fail(op, dw, RefuseWhy::ContextControlOnMec);
				}
				break;
			case IT_COND_EXEC: {
				// libSceAgc's check around a context push or pop: a RELEASE_MEM that tells the kernel of a second
				// push or pop in a row (INT_SEL 4, context id with bit 27 set), run when its flag is set. That is
				// the PS5 kernel's; both are left out. Any other COND_EXEC refuses the buffer.
				const uint32_t next = dw + size;
				// AgcDcbResetQueue op 0x400: on the push flag, a pop (CLEAR_STATE 2) and the flag's swap back to {0, 1}. The
				// shadow pops only while a push is outstanding, and without one the swap writes what the flag holds, so
				// both go on as written and the COND_EXEC is left out.
				if (!m_compute_ring && body_count >= 4 && (bptr[3] & 0x3fffu) == 11u && next + 11u <= dcb.size() &&
				    dcb[next] == 0xc0001200u && dcb[next + 1] == 2u && dcb[next + 2] == 0xc0071e00u) {
					stats.context_checks++;
					break;
				}
				if (m_compute_ring || body_count < 4 || (bptr[3] & 0x3fffu) != 8u || next + 8u > dcb.size() ||
				    dcb[next] != 0xc0064900u || ((dcb[next + 2] >> 24u) & 0x7u) != 4u || (dcb[next + 7] & 0x08000000u) == 0) {
					// Any other COND_EXEC (SetCondExec, KYTY_NATIVE_GPU_COND_EXEC=1): as written with its count recomputed,
					// gfx ring only (a compute IB is cut into several for KFD, and MEC has not run a title's). The PFP decides
					// a COND_EXEC ahead of the ME, so a PFP_SYNC_ME first lets it see the ME's writes before it, as the PS5's
					// microcode does.
					const uint64_t addr = (bptr[0] & ~3u) | (static_cast<uint64_t>(bptr[1] & 0xffffu) << 32u);
					// libSceAgc masks EXEC_COUNT to 14 bits (AgcDcbCondExec); the CP reads more of the dword (probe condexec:
					// a 20005-dword skip on MEC), so a count with higher bits is not libSceAgc's and its range is not known.
					// A flag the tracker writes in this segment (a store, atomic or timestamp the GPU cannot do) lands on the
					// CPU after the GPU has passed the COND_EXEC, where the PS5's CP reads the title's value: refused too.
					const bool cpu_flag = std::any_of(segs.back().writes.begin(), segs.back().writes.end(), [&](const LabelWrite& w) {
						uint64_t v = 0;
						return !w.gpu && w.Covers(addr, false, &v) != 0;
					});
					if (!m_cond_exec || m_compute_ring || cond.active || body_count != 4 || addr == 0 || (bptr[3] & ~0x3fffu) != 0 ||
					    next + bptr[3] > dcb.size() || cpu_flag) {
						return fail(op, dw, RefuseWhy::CondExecForm);
					}
					// The points queued before the range (an interrupt, a store the GPU cannot do) go before it, not inside
					// where the CP may skip their store.
					flush_points();
					body.push_back((3u << 30u) | (IT_PFP_SYNC_ME << 8u));
					body.push_back(0);
					stats.pfp_syncs_emitted++;
					cond.count_at = body.size() + 4u;
					body.insert(body.end(), dcb.begin() + dw, dcb.begin() + dw + size);
					add_ref(addr, 4, false);
					cond.active     = true;
					cond.dw         = dw;
					cond.end        = next + (bptr[3] & 0x3fffu);
					cond.segs       = segs.size();
					cond.seg_points = segs.back().points.size();
					cond.tail       = r.cpu_tail.size();
					cond.writes     = local_writes.size();
					cond.seg_writes = segs.back().writes.size();
					cond.fence_seq  = m_fence_seq;
					direct          = false;
					stats.cond_execs++;
					break;
				}
				stats.context_checks++;
				dw += 8u; // the RELEASE_MEM
				break;
			}
			case IT_ATOMIC_MEM: {
				// libSceAgc's flag swap around a context push or pop, and the workload swap and AND of SetWorkloadsActive and
				// SetWorkloadComplete (into the driver's workload words): as written on the gfx ring, where the GPU may
				// write the address. MEC has not run an ATOMIC_MEM on this board, and memory the GPU cannot write must not
				// get one: with native labels the tracker performs it on the CPU when the GPU gets there. Without native
				// labels Kyty's command processor runs the whole buffer and performs it, so a compute ring leaves it out.
				// With KYTY_NATIVE_GPU_ATOMIC_OPS (on by default) every integer op goes the same way (IntegerAtomicOp), the
				// tracker running it on the CPU with the same TC_OP (CpuAtomicMem).
				if (body_count != 8 || !(AgcAtomicOp(bptr[0]) || (m_atomic_ops && IntegerAtomicOp(bptr[0])))) {
					return fail(op, dw, RefuseWhy::AtomicForm);
				}
				const uint64_t addr  = static_cast<uint64_t>(bptr[1]) | (static_cast<uint64_t>(bptr[2]) << 32u);
				const uint64_t bytes = (bptr[0] & 0x20u) != 0 ? 8u : 4u;
				if (labels && (m_compute_ring || !label_gpu(addr, bytes))) {
					TrackerPoint cpu_atomic;
					cpu_atomic.kind = TrackerPoint::Kind::Atomic;
					cpu_atomic.addr = addr;
					cpu_atomic.data = {bptr[0], bptr[3], bptr[4], bptr[5], bptr[6]};
					tracker_point(std::move(cpu_atomic));
					note_write({addr, 0, false, false, bytes == 8u});
					stats.atomics_cpu++;
					NoteOnce(24, "ATOMIC_MEM performed by the tracker (compute ring, or memory the GPU cannot write)");
					break;
				}
				if (m_compute_ring) {
					break;
				}
				if (direct) {
					in_place = true;
				} else {
					body.insert(body.end(), dcb.begin() + dw, dcb.begin() + dw + size);
				}
				if (!labels) {
					add_ref(addr, bytes, false); // label_gpu has added it otherwise
				}
				note_gpu_written(addr, addr + bytes);
				if (labels && !AgcAtomicOp(bptr[0])) {
					note_write({addr, 0, false, true, bytes == 8u}); // a value the GPU computes
				}
				break;
			}
			case IT_CLEAR_STATE:
				// CMD 1-3 (libSceAgc's AgcDcbContextStateOp): PUSH_STATE, POP_STATE, push then clear, which the
				// PS5's CP does with register shadowing the kernel sets up. The shadow does them here.
				reg_write = true;
				if (body_count >= 1 && (bptr[0] & 0xfu) != 0) {
					context_state_op(bptr[0] & 0xfu);
					break;
				}
				// The hardware resets the context to amdgpu's clear state, the PS5 CP to its own
				// defaults (Kyty: the same as a context-state clear). Follow it with the defaults, every
				// one of them, since the hardware no longer matches the shadow.
				body.insert(body.end(), dcb.begin() + dw, dcb.begin() + dw + size);
				stats.clear_state++;
				// CP shadowing: a plain CLEAR_STATE leaves S alone and the bases survive it (probe shadow-pushpop);
				// whether the enables do is not probed. KYTY_NATIVE_GPU_CP_SHADOW_REARM=1 turns them on again here.
				if (cp_shadow && m_cp_shadow_rearm) {
					body.push_back(Pkt3(IT_CONTEXT_CONTROL, 2, false));
					body.push_back(kCpShadowEnables);
					body.push_back(kCpShadowEnables);
				}
				// The clear sets the context to amdgpu's clear state, which S does not hold: only the defaults'
				// SETs below give S its values again.
				if (cp_shadow) {
					vcx.reset();
					fmask_done = 0;
					dcc_done   = 0;
				}
				for (const auto& [off, value]: m_cx_defaults) {
					EmitSet(body, IT_SET_CONTEXT_REG, off, PatchContext(off, value), false);
					mark_cx(off);
					writer(wcx, off, RegWriter::Insert);
				}
				cx = m_cx_defaults;
				break;

			default:
				// A raw RELEASE_MEM (0x49) has no case and refuses the DCB below; the title sends its EOP
				// sync as NOP markers instead. Still record its event control dword if one ever appears.
				if (op == IT_RELEASE_MEM && body_count >= 1) {
					stats.event_type[bptr[0] & 0x3fu]++;
					stats.event_index[(bptr[0] >> 8u) & 0xfu]++;
				}
				// Keep walking so one refused DCB names every packet the translator lacks.
				if (r.unknown_ops.empty()) {
					r.unknown_op = op;
					r.unknown_dw = dw;
					if (op == 0x59u && body_count >= 1) { // REWIND: its state dword, what follows
						r.unknown_detail = {bptr[0], static_cast<uint32_t>(dcb.size() - dw)};
						for (uint32_t i = 2; i < 8 && dw + i < dcb.size(); i++) {
							r.unknown_detail.push_back(dcb[dw + i]);
						}
					}
				}
				r.unknown_ops.insert(op);
				note_refusal(op, dw, RefuseWhy::Unknown);
				break;
		}
		// MecPlainOp: the title's header (no shader-type bit) where the packet is passed on.
		if (m_compute_ring && MecPlainOp(op) && body.size() > body_before && (body[body_before] & ~2u) == (header & ~2u)) {
			body[body_before] = header;
		}
		if (IsWork(op)) {
			seg_effects = true;
		}
		// A first segment that only writes registers is dropped without CP shadowing: the next segment's preamble
		// replays its writes from the copy. A LOAD from S would not see them, so with shadowing it stays.
		if (cp_shadow && reg_write) {
			seg_effects = true;
		}
		if (cond.active && dw != cond.dw && reg_write) {
			return fail(IT_COND_EXEC, cond.dw, RefuseWhy::CondExecForm); // a register write the CP may skip
		}
		const bool same_seg = segs.size() == segs_before;
		if (in_place && (!same_seg || body.size() != body_before)) {
			return fail(op, dw, RefuseWhy::InPlace); // a packet run in place has nothing of ours around it
		}
		const bool same = in_place || (same_seg && body.size() - body_before == size &&
		                               std::memcmp(body.data() + body_before, dcb.data() + dw, size * 4u) == 0);
		stats.ops[op]++;
		if (same) {
			stats.verbatim_dw += size;
			verbatim_run += size;
		} else {
			end_run();
			stats.changed_ops[op]++;
		}
		const uint32_t copied = in_place ? 0u : size;
		if (direct && same && direct_cur.len != 0 && direct_cur.body + direct_cur.body_len == body_before &&
		    direct_cur.dw + direct_cur.len == dw) {
			direct_cur.len += size;
			direct_cur.body_len += copied;
			direct_cur.in_place |= in_place;
		} else if (direct) {
			end_direct();
			if (same) {
				direct_cur = {static_cast<uint32_t>(body_before), copied, dw, size, in_place};
			}
		}
		if (same || (op == IT_NOP && same_seg && body.size() == body_before)) {
			cut_run += size;
		} else {
			end_cut_run();
			const bool tail = same_seg && body.size() - body_before > size &&
			                  std::memcmp(body.data() + body.size() - size, dcb.data() + dw, size * 4u) == 0;
			if (tail) {
				stats.inserted_ops[op]++;
			} else {
				stats.cut_ops[op]++;
			}
		}

		dw += size;
	}
	if (cond.active && (!cond_clean() || dw != cond.end || !cond_close())) {
		return fail(IT_COND_EXEC, cond.dw, RefuseWhy::CondExecForm); // the range runs past the buffer
	}
	end_run();
	end_cut_run();
	if (direct) {
		end_direct();
	}
	if (!labels) {
		cp_dma_idle(); // with native labels the last segment's close puts it before the tracker's release
	}
	if (!r.unknown_ops.empty()) {
		const uint32_t first_op = r.unknown_op;
		const uint32_t first_dw = r.unknown_dw;
		auto           unknown  = std::move(r.unknown_ops);
		fail(first_op, first_dw, RefuseWhy::Collected);
		r.unknown_ops = std::move(unknown);
		return r;
	}

	// Pass succeeded: commit the shadow, build the IB (preamble from the committed shadow, then the
	// translated body), and return. SetPreambleFromStart keeps the state where the buffer starts for it.
	// CP shadowing: the first segment's shadow preamble. In load mode it restores the state where the buffer
	// starts (the keys there, the values S holds when it runs), as on the PS5; in record mode, like the replay,
	// from the state SetPreambleFromStart picks. With SetNoPreamble it is the resume the backend may put in front.
	std::vector<uint32_t> cp_head;
	Pm4TranslateStats     resume_stats;
	Pm4TranslateStats&    head_stats      = m_no_preamble && labels ? resume_stats : stats;
	const bool            head_from_start = CpShadow() == CpShadowMode::Load || m_preamble_from_start;
	if (cp_shadow && head_from_start) {
		EmitShadowPreamble(cp_head, head_stats, m_cx, m_sh, m_uc, m_cp);
	}
	RegFile start_cx, start_sh, start_uc;
	if (m_preamble_from_start && !cp_shadow) {
		start_cx = m_cx;
		start_sh = m_sh;
		start_uc = m_uc;
	}
	m_cx        = std::move(cx);
	m_saved_cx  = std::move(saved_cx);
	m_cx_pushed = cx_pushed;
	m_sh = std::move(sh);
	m_uc = std::move(uc);
	m_writer_cx       = wcx;
	m_writer_sh       = wsh;
	m_writer_uc       = wuc;
	m_saved_writer_cx = saved_wcx;
	m_cp_valid_cx     = vcx;
	m_saved_valid_cx  = saved_vcx;
	// The IB starts where the previous DCB left the CP, not where this one leaves it.
	const Pm4CpDrawState cp_start = m_cp;
	m_cp                          = cp;
	if (cp_shadow && !head_from_start) {
		EmitShadowPreamble(cp_head, head_stats, m_cx, m_sh, m_uc, cp_start);
	}
	if (cp_shadow) {
		m_cp_shadow_force_load = false;
	}
	r.cp_shadow = cp_shadow;
	if (r.injected) {
		m_inject_wait_va   = 0; // once: a refused buffer leaves it armed for the next one
		m_inject_shader_va = 0;
	}

	// Copies body[from, to) to the IB, each direct run in it as IB2s into the guest's buffer. Returns,
	// per run, (body offset where it ends, dwords removed up to there) to move later body offsets.
	size_t direct_next = 0;
	auto   append_body = [&](uint32_t from, uint32_t to, bool last) {
		std::vector<std::pair<uint32_t, int64_t>> removed;
		int64_t                                   gone = 0;
		uint32_t                                  at   = from;
		while (direct_next < direct_runs.size() && direct_runs[direct_next].body < from) {
			direct_next++; // in a dropped segment
		}
		for (; direct_next < direct_runs.size() && (direct_runs[direct_next].body < to || last); direct_next++) {
			const DirectRun& run = direct_runs[direct_next];
			r.ib.insert(r.ib.end(), body.begin() + at, body.begin() + run.body);
			uint32_t emitted = 0;
			if (!run.ib2) { // short: its packets from the buffer, those run in place included
				r.ib.insert(r.ib.end(), dcb.begin() + run.dw, dcb.begin() + run.dw + run.len);
				emitted = run.len;
			}
			for (uint32_t done = 0; run.ib2 && done < run.len;) {
				uint32_t       left = 0;
				const uint64_t va   = source_va(run.dw + done, &left);
				const uint32_t take = std::min({left, run.len - done, 0xfffffu});
				r.ib.push_back(Pkt3(IT_INDIRECT_BUFFER, 3, false));
				r.ib.push_back(static_cast<uint32_t>(va));
				r.ib.push_back(static_cast<uint32_t>(va >> 32u));
				r.ib.push_back(take | (1u << 23u)); // IB_SIZE, VALID
				emitted += 4;
				done += take;
				stats.direct_ib2++;
			}
			stats.direct_dw += run.ib2 ? run.len : 0u;
			gone += static_cast<int64_t>(run.body_len) - emitted;
			removed.emplace_back(run.body + run.body_len, gone);
			at = run.body + run.body_len;
		}
		r.ib.insert(r.ib.end(), body.begin() + at, body.begin() + to);
		return removed;
	};

	if (labels) {
		if (span_timed) {
			EmitClockEop(body, span_pair + 8);
			m_time_spans++;
		}
		close_segment(true);
		// Each kept segment: its preamble (the first one's from the committed shadow, as without native
		// labels), its body, padding; tracker points and patch sites move from body to IB offsets.
		r.ib.reserve(body.size() + (m_cx.size() + m_sh.size() + m_uc.size()) * 3 * segs.size() + 8 * segs.size());
		for (size_t k = 0; k < segs.size(); k++) {
			if (!seg_keep[k]) {
				continue;
			}
			Pm4Segment& s = segs[k];
			s.begin       = static_cast<uint32_t>(r.ib.size());
			if (gfx_guards) {
				EmitGfxExitGuard(r.ib, m_exit_abort_va, m_exit_stub_va);
			} else if (cs_guards) {
				s.guards.push_back(EmitComputeExitGuard(r.ib, m_exit_run_va));
			}
			if (k == 0) {
				if (m_no_preamble) {
					if (cp_shadow) {
						// Whole 8-dword units, as every piece it is put in front of.
						r.cp_shadow_resume = cp_head;
						while ((r.cp_shadow_resume.size() & 7u) != 0) {
							r.cp_shadow_resume.push_back(kFiller);
						}
					}
				} else if (cp_shadow) {
					r.ib.insert(r.ib.end(), cp_head.begin(), cp_head.end());
					s.cp_shadow_head = true;
				} else if (m_preamble_from_start) {
					EmitPreamble(r.ib, stats, start_cx, start_sh, start_uc, cp_start);
				} else {
					EmitPreamble(r.ib, stats, m_cx, m_sh, m_uc, cp_start);
				}
			} else {
				r.ib.insert(r.ib.end(), seg_pre[k].begin(), seg_pre[k].end());
			}
			const uint32_t from  = seg_body[k];
			const uint32_t to    = k + 1 < segs.size() ? seg_body[k + 1] : static_cast<uint32_t>(body.size());
			const int64_t  shift   = static_cast<int64_t>(r.ib.size()) - static_cast<int64_t>(from);
			const auto     removed = append_body(from, to, k + 1 == segs.size());
			while ((r.ib.size() & 7u) != 0) {
				r.ib.push_back(0xffff1000u);
			}
			s.end = static_cast<uint32_t>(r.ib.size());
			// Offsets are never inside a direct run: points, barriers and waits are packets of ours.
			auto move = [&](uint32_t off) {
				int64_t gone = 0;
				for (const auto& [end, g]: removed) {
					if (end > off) {
						break;
					}
					gone = g;
				}
				return static_cast<uint32_t>(off + shift - gone);
			};
			// Points come in IB order: one pass over the runs.
			size_t  next_run = 0;
			int64_t gone     = 0;
			for (auto& p: s.points) {
				for (; next_run < removed.size() && removed[next_run].first <= p.dw; next_run++) {
					gone = removed[next_run].second;
				}
				p.dw = static_cast<uint32_t>(p.dw + shift - gone);
			}
			if (s.held) {
				s.barrier_dw = move(s.barrier_dw);
				s.wait_dw    = move(s.wait_dw);
			}
			if (cs_guards) {
				// Each guard skips to the next one (a compute IB has no direct runs: body offsets only shift),
				// the last to the end. A stretch too long for EXEC_COUNT keeps a guard that skips nothing.
				for (uint32_t g: body_guards) {
					if (g >= from && g < to) {
						s.guards.push_back(move(g));
					}
				}
				for (size_t i = 0; i < s.guards.size(); i++) {
					const uint32_t until = i + 1 < s.guards.size() ? s.guards[i + 1] - 4u : s.end;
					const uint32_t skip  = until - (s.guards[i] + 1u);
					r.ib[s.guards[i]]    = skip <= kExitGuardMaxSkip ? skip : 0u;
				}
			}
			r.segments.push_back(std::move(s));
		}
	} else {
		r.ib.reserve(body.size() + (m_cx.size() + m_sh.size() + m_uc.size()) * 3 + 8);
		if (cp_shadow) {
			r.ib.insert(r.ib.end(), cp_head.begin(), cp_head.end());
		} else {
			EmitPreamble(r.ib, stats, m_cx, m_sh, m_uc, cp_start);
		}
		append_body(0, static_cast<uint32_t>(body.size()), true);

		// The CP fetches IBs in 32-byte (8-dword) units; pad with filler NOPs.
		while ((r.ib.size() & 7u) != 0) {
			r.ib.push_back(0xffff1000u);
		}
	}

	r.stats        = stats;
	std::sort(shaders.begin(), shaders.end());
	shaders.erase(std::unique(shaders.begin(), shaders.end()), shaders.end());
	r.shader_addrs = std::move(shaders);
	std::sort(refs.begin(), refs.end());
	refs.erase(std::unique(refs.begin(), refs.end()), refs.end());
	r.refs = std::move(refs);
	if (m_compact_refs) {
		for (size_t w = 0; w < m_bucket_bits.size(); w++) {
			for (uint64_t bits = m_bucket_bits[w]; bits != 0; bits &= bits - 1) {
				const uint64_t i = w * 64 + static_cast<uint64_t>(__builtin_ctzll(bits));
				r.ref_buckets.push_back(static_cast<uint32_t>(i < kLowBuckets ? i : (kExtBase >> kBucketShift) + i - kLowBuckets));
			}
		}
	}
	for (auto& [addr, t]: targets) {
		r.render_targets.push_back(t);
	}
	return r;
}

} // namespace Libs::Graphics::NativeGpu
