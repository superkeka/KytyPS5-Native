#include "graphics/native_gpu/nativePm4.h"

#include <algorithm>
#include <cstring>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <tuple>
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
constexpr uint32_t IT_SET_BASE                 = 0x11;
constexpr uint32_t IT_SET_PREDICATION          = 0x20;
constexpr uint32_t IT_INDEX_BUFFER_SIZE        = 0x13;
constexpr uint32_t IT_DISPATCH_DIRECT          = 0x15;
constexpr uint32_t IT_DISPATCH_INDIRECT        = 0x16;
constexpr uint32_t IT_DRAW_INDIRECT            = 0x24;
constexpr uint32_t IT_RELEASE_MEM              = 0x49;
constexpr uint32_t IT_DRAW_INDEX_INDIRECT      = 0x25;
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
constexpr uint32_t R_ZERO           = 0x00;
constexpr uint32_t R_WAIT_FLIP_DONE = 0x06;
constexpr uint32_t R_FLIP           = 0x17;
constexpr uint32_t R_RELEASE_MEM    = 0x18;
constexpr uint32_t R_CONTEXT_STATE  = 0x1A; // Kyty's marker for Agc context-state push/pop/clear

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
constexpr uint32_t kGcrGl2Wb        = 1u << 21u;
constexpr uint32_t kReleaseGcrMask  = 0x1fffu << 12u; // GCR_CNTL [24:12]
constexpr uint32_t kReleasePolicy   = 0x3u << 25u;    // CACHE_POLICY
bool LabelEventOk(uint32_t event) {
	const uint32_t type  = event & 0x3fu;
	const uint32_t index = (event >> 8u) & 0xfu;
	const bool     known = type == 0x14u || type == 0x28u || type == 0x2bu || type == 0x2du || type == 0x2fu || type == 0x30u;
	return known && (index == 5u || index == 6u) && (event & ~(0x3fu | 0xf00u | kReleaseGcrMask | kReleasePolicy)) == 0u;
}
// KYTY_NATIVE_GPU_LABEL_WB=0 (experiment): a label release keeps exactly the guest's cache actions. The
// label itself goes to memory past the caches (DST_SEL 0) either way, and GPU waits read it through
// GL2; the added write-backs only make the GPU's other results visible to the CPU, and they come at
// every label (ASTRO BOT: ~900 a frame, ~5400 where it draws much).
bool LabelWriteBack() {
	static const bool wb = [] {
		const char* v = std::getenv("KYTY_NATIVE_GPU_LABEL_WB");
		return v == nullptr || v[0] != '0';
	}();
	return wb;
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

// Emits a guest wait natively, in the proven form: only EQUAL waits, a 64-bit one as the 32-bit wait
// on its low dword when the high halves of mask and reference are 0 (the same condition), a PFP one as
// the micro engine's wait and PFP_SYNC_ME. Returns false when that is not possible.
struct NotedWait {
	std::atomic<uint64_t> addr {0}; // written last: 0 while the entry is being filled
	WaitCond              wait;
};
std::array<NotedWait, 512> g_noted_waits;
std::atomic<uint32_t>      g_noted_next {0};

void NoteGpuWait(const WaitCond& w) {
	auto& e = g_noted_waits[g_noted_next.fetch_add(1, std::memory_order_relaxed) % g_noted_waits.size()];
	e.addr.store(0, std::memory_order_relaxed);
	e.wait = w;
	e.addr.store(w.addr, std::memory_order_release);
}
} // namespace

std::vector<WaitCond> RecentGpuWaits() {
	std::vector<WaitCond> out;
	for (const auto& e: g_noted_waits) {
		const uint64_t addr = e.addr.load(std::memory_order_acquire);
		if (addr != 0 && e.wait.addr == addr) {
			out.push_back(e.wait);
		}
	}
	return out;
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
	NoteGpuWait(w);
	return true;
}

// R_ZERO NOP payloads Kyty's command processor runs as flips (pm4Handlers.cpp CpOpMarker): a flip, a
// flip that writes a label first, and one that also raises the EOP interrupt.
bool IsFlipMarker(uint32_t tag) {
	return tag == 0x68750777u || tag == 0x68750778u || tag == 0x68750781u;
}

// Packets that make the GPU do something (as opposed to setting state), for dropping an empty first
// segment. Labels, waits and LOD statistics say so themselves.
bool IsWork(uint32_t op) {
	switch (op) {
		case IT_DRAW_INDEX_2:
		case IT_DRAW_INDEX_OFFSET_2:
		case IT_DRAW_INDIRECT:
		case IT_DRAW_INDEX_INDIRECT:
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

bool ValidateNativeIb(std::span<const uint32_t> ib, bool lod_stats, uint32_t* bad_dw, NativeIbKind kind) {
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
				if (header != kFiller) {
					return bad();
				}
				break;
			case IT_RELEASE_MEM:
				// The barrier's form (any address and data), a GPU-clock timestamp (DATA_SEL 3, data
				// dwords zero, 8-byte address), or a GDS release as the translator has always run it
				// (DATA_SEL 5, no interrupt). Compute rings use the same bodies with the shader bit.
				if (header != Pkt3(IT_RELEASE_MEM, 7, compute) || b[6] != 0u) {
					return bad();
				}
				if ((b[0] == kFlushEventCntl || LabelEventOk(b[0])) && b[1] == kData32Cntl) {
					break; // the barrier's form, or a label in the guest's event (LabelEventCntl)
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
				if (header != Pkt3(IT_WAIT_REG_MEM, 6, compute) || b[0] != kWaitEqualMemMe || b[5] != kWaitPoll) {
					return bad();
				}
				break;
			case IT_GET_LOD_STATS:
				if (compute || !lod_stats) {
					return bad();
				}
				break;
			// GPU timing (SetGpuTimes): the GPU clock to memory, on the gfx ring.
			case IT_COPY_DATA:
				if (compute || size != 6 || b[0] != kClockCopyCntl || (b[3] & 7u) != 0u) {
					return bad();
				}
				break;
			// The translator's IB2 into the guest's buffer (SetDirectRuns): a call, not a chain, on the gfx
			// ring. Its packets are the ones the translator passed on as written.
			case IT_INDIRECT_BUFFER:
				if (compute || size != 4 || (b[2] & (1u << 20u)) != 0 || (b[2] & (1u << 23u)) == 0) {
					return bad();
				}
				break;
			// A guest WRITE_DATA to memory, write confirmed; on MEC from the micro engine.
			case IT_WRITE_DATA:
				if (size < 5 || !WriteDataToMemory(b[0]) || (b[0] & (1u << 20u)) == 0 ||
				    (compute && ((header & 2u) == 0 || (b[0] >> 30u) != 0))) {
					return bad();
				}
				break;
			// Packets both rings run. On MEC the shader-type bit is set.
			case IT_SET_BASE:
			case IT_DISPATCH_DIRECT:
			case IT_DISPATCH_INDIRECT:
			case IT_EVENT_WRITE:
			case IT_DMA_DATA:
			case IT_ACQUIRE_MEM:
			case IT_SET_SH_REG:
				if (compute && (header & 2u) == 0) {
					return bad();
				}
				break;
			// Graphics-only. An illegal opcode on a compute ring.
			case IT_CLEAR_STATE:
			case IT_SET_PREDICATION:
			case IT_INDEX_BUFFER_SIZE:
			case IT_DRAW_INDIRECT:
			case IT_DRAW_INDEX_INDIRECT:
			case IT_INDEX_BASE:
			case IT_DRAW_INDEX_2:
			case IT_CONTEXT_CONTROL:
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

RingWaitRelease PlanRingWait(int cover, bool value_satisfies, bool gpu_write, bool same_ring, bool writer_submitted,
                             bool label_in_memory, bool native_wait, bool gfx_writer) {
	if (cover != 0) {
		// A write already emitted still owns the address. Matching memory is stale until it lands,
		// and an unknown value cannot be proved to satisfy the wait, so the segment stays held.
		// Across rings the GPU wait itself reads that stale memory: a label still holding the value
		// from its previous use passes it before the writer's ring has even run (ASTRO BOT: the gfx
		// ring then reset the label its compute ring was about to wait for, and the compute ring
		// waited forever). Such a wait goes to the GPU only while memory does not satisfy it yet;
		// otherwise it is held until the writer lands and becomes a barrier.
		const bool gpu_wait = cover == 1 && value_satisfies && gpu_write && native_wait &&
		                      (same_ring || ((writer_submitted || gfx_writer) && !label_in_memory));
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
                std::vector<SourcePiece>* pieces) {
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
		const uint32_t len    = (header >> 30u) == 3u ? ((header >> 16u) & 0x3fffu) + 2u : 1u;
		if (dw + len > in.size()) {
			return stop("packet past the end", dw);
		}
		const uint32_t op = (header >> 8u) & 0xffu;
		if ((header >> 30u) == 3u && op == kRewind && len == 2) {
			if ((in[dw + 1] & 0x80000000u) == 0) {
				return stop("REWIND not valid yet", dw);
			}
			if (!flush(dw)) {
				return stop("too large", dw);
			}
			changed = 1;
			dw += len;
			run = dw;
			continue;
		}
		if ((header >> 30u) == 3u && op == kIndirect) {
			if (len != 4) {
				return stop("INDIRECT_BUFFER of another form", dw);
			}
			if (!flush(dw)) {
				return stop("too large", dw);
			}
			const uint32_t control = in[dw + 3];
			const auto*    target  = reinterpret_cast<const uint32_t*>((static_cast<uint64_t>(in[dw + 2]) << 32u) | (in[dw + 1] & ~3u));
			const uint32_t size    = control & 0xfffffu;
			const bool     chain   = (control & (1u << 20u)) != 0;
			changed                = 1;
			if (size == 0) {
				if (chain) {
					return changed;
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
				in  = {target, size};
				dw  = 0;
				run = 0;
				continue;
			}
			if (depth >= 16) {
				return stop("calls nested too deep", dw);
			}
			if (FlattenInto({target, size}, out, depth + 1, why, pieces) < 0) {
				return -1;
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
	return changed;
}
} // namespace

bool FlattenIndirectBuffers(std::span<const uint32_t> in, std::vector<uint32_t>* out, std::string* why,
                            std::vector<SourcePiece>* pieces) {
	out->clear();
	if (pieces != nullptr) {
		pieces->clear();
	}
	if (FlattenInto(in, out, 0, why, pieces) <= 0) {
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

std::vector<std::pair<size_t, size_t>> SplitIbAtPackets(std::span<const uint32_t> ib, size_t max_dw) {
	std::vector<std::pair<size_t, size_t>> pieces;
	size_t                                 start = 0;
	size_t                                 dw    = 0;
	while (dw < ib.size()) {
		const uint32_t h = ib[dw];
		// A type-3 header with count 0x3fff is the one-dword filler NOP; type-2 packets are one dword.
		const size_t len = (h >> 30u) == 3u && h != 0xffff1000u ? ((h >> 16u) & 0x3fffu) + 2u : 1u;
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

	for (const auto& [offset, value]: cx) {
		EmitSet(ib, IT_SET_CONTEXT_REG, offset, PatchContext(offset, value), false);
		stats.preamble_cx++;
	}
	if (m_ps5_reg_overrides && cx.count(kPaScShaderControl) == 0) {
		EmitSet(ib, IT_SET_CONTEXT_REG, kPaScShaderControl, PatchContext(kPaScShaderControl, 0), false);
	}
	if (m_depth_mode == DepthDebugMode::HizOff && cx.count(kDbRenderOverride) == 0) {
		EmitSet(ib, IT_SET_CONTEXT_REG, kDbRenderOverride, PatchContext(kDbRenderOverride, 0), false);
	}
	for (const auto& [offset, value]: sh) {
		EmitSet(ib, IT_SET_SH_REG, offset, PatchSh(offset, value), offset >= kShComputeBase);
		stats.preamble_sh++;
	}
	for (const auto& [offset, value]: uc) {
		EmitSetUconfig(ib, offset, value);
		stats.preamble_uc++;
	}
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

	// A bound color target without FMASK must still point CB_COLORn_FMASK at its own base, as
	// radeonsi and PAL always program it: with a CMASK fast clear the CB reads that address. ASTRO BOT
	// binds such a target as MRT4 with FMASK 0, and the CB stalls forever on the gfx10 hardware here
	// (CB/SC busy, no waves; the same surface with FMASK == BASE renders fine as MRT0).
	auto fix_fmask = [&]() {
		const uint32_t mask = reg(cx, kCbTargetMask) & m_target_mask;
		for (uint32_t i = 0; i < 8; i++) {
			const uint32_t step = i * kCbColorStride;
			const uint32_t base = reg(cx, kCbColor0Base + step);
			if (((mask >> (4u * i)) & 0xfu) == 0 || base == 0 || reg(cx, kCbColor0Fmask + step) != 0) {
				continue;
			}
			const uint32_t ext = reg(cx, kCbColor0BaseExt + i);
			cx[kCbColor0Fmask + step]  = base;
			cx[kCbColor0FmaskExt + i] = ext;
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
			if (cur == nullptr || *cur != value) {
				EmitSet(body, IT_SET_CONTEXT_REG, off, PatchContext(off, value), false);
				mark_cx(off);
			}
		}
		for (const auto& [off, value]: cx) {
			if (next.count(off) == 0 && value != 0) {
				EmitSet(body, IT_SET_CONTEXT_REG, off, PatchContext(off, 0), false);
				mark_cx(off);
			}
		}
		cx = next;
	};
	// Agc context-state operations (AgcDcbContextStateOp, emulated by Kyty's command processor):
	// 0 clear to the defaults, 1 push, 2 pop, 3 push then clear. The PS5 packets behind them save and
	// restore the context registers; here the shadow does it and the differences are emitted.
	auto context_state_op = [&](uint32_t operation) {
		stats.context_ops++;
		switch (operation) {
			case 0: switch_context(m_cx_defaults); break;
			case 1:
				saved_cx  = cx;
				cx_pushed = true;
				break;
			case 2:
				if (cx_pushed) {
					switch_context(saved_cx);
					saved_cx.clear();
					cx_pushed = false;
				}
				break;
			case 3:
				saved_cx  = cx;
				cx_pushed = true;
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
		constexpr uint32_t kDccEnable = 1u << 28u;
		for (uint32_t i = 0; i < 8; i++) {
			const uint32_t step = i * kCbColorStride;
			const uint32_t off  = kCbColor0Info + step;
			const uint32_t info = reg(cx, off);
			const uint64_t addr = (static_cast<uint64_t>(reg(cx, kCbColor0Base + step)) << 8u) |
			                      (static_cast<uint64_t>(reg(cx, kCbColor0BaseExt + i) & 0xffu) << 40u);
			const bool display = addr != 0 && std::any_of(m_display_ranges.begin(), m_display_ranges.end(), [&](const auto& d) {
				return addr >= d.first && addr < d.first + d.second;
			});
			if (display && (info & kDccEnable) != 0) {
				m_display_dcc_info[i] = info;
				m_display_dcc_off |= 1u << i;
				cx[off] = info & ~kDccEnable;
				EmitSet(body, IT_SET_CONTEXT_REG, off, PatchContext(off, cx[off]), false);
				NoteOnce(17, "render target in a video-out buffer drawn without DCC");
			} else if (!display && ((m_display_dcc_off >> i) & 1u) != 0) {
				if (info == (m_display_dcc_info[i] & ~kDccEnable)) {
					cx[off] = m_display_dcc_info[i];
					EmitSet(body, IT_SET_CONTEXT_REG, off, PatchContext(off, cx[off]), false);
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
	// WRITE_DATA). On the gfx ring the PFP fetches them ahead of the ME that performs those writes,
	// which a compute queue (one engine) never does: sync the PFP to the ME first, as RADV does.
	auto pfp_sync_me = [&]() {
		// A compute ring has one engine; the PFP race this closes exists only on the gfx ring, and
		// PFP_SYNC_ME is illegal on MEC.
		if (m_pfp_sync_indirect && !m_compute_ring) {
			body.push_back((3u << 30u) | (IT_PFP_SYNC_ME << 8u));
			body.push_back(0);
			stats.pfp_syncs_emitted++;
		}
	};

	// A guest wait turned into a full barrier on our fence: an end-of-pipe RELEASE_MEM that flushes and
	// invalidates the caches (Mesa's gfx10 wait-for-idle) and a wait on that fence. A wait the prefetch
	// parser does (ENGINE_SEL = PFP) on a label the micro engine has just written is how the guest keeps
	// the PFP from fetching ahead of the ME past a cache flush; PFP_SYNC_ME is that synchronization
	// without memory.
	auto emit_barrier = [&](bool pfp) {
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
	auto flush_points = [&]() {
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
	// Cuts the IB at a guest wait nobody known satisfies yet: the rest starts held, with its own
	// preamble, a barrier block and the guest wait (the backend NOPs one of the two).
	auto split_at_wait = [&](const WaitCond& w) {
		close_segment(false);
		segs.emplace_back();
		seg_body.push_back(static_cast<uint32_t>(body.size()));
		seg_pre.emplace_back();
		seg_keep.push_back(true);
		EmitPreamble(seg_pre.back(), stats, cx, sh, uc, cp);
		auto& s = segs.back();
		s.held  = true;
		s.wait  = w;
		s.barrier_dw = static_cast<uint32_t>(body.size());
		emit_barrier(w.pfp);
		s.barrier_len = static_cast<uint32_t>(body.size()) - s.barrier_dw;
		s.wait_dw     = static_cast<uint32_t>(body.size());
		// No native form (not EQUAL, or a 64-bit wait on both halves), or an address the GPU cannot
		// reach: only seeing the label in memory releases it. A WAIT_REG_MEM there reads the dummy.
		s.wait_len = 0;
		// The native form polls one dword. That dword has to be resident; the rest of a 64-bit
		// wait is not what this packet reads.
		if (label_gpu(w.addr, 4)) {
			s.wait_len = EmitNativeWait(body, w, m_compute_ring) ? static_cast<uint32_t>(body.size()) - s.wait_dw : 0u;
		}
		seg_effects = true;
		stats.waits_split++;
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
	const bool direct = m_direct_min_dw != 0 && !m_compute_ring && !source.empty();
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
	while (dw < dcb.size()) {
		const size_t   body_before = body.size();
		const size_t   segs_before = segs.size();
		bool           in_place    = false; // passed on as written without a copy in body (direct runs)
		const uint32_t header = dcb[dw];
		if ((header >> 30u) != 3u) {
			return fail(0xffffffffu, dw); // not a type-3 packet
		}
		const uint32_t size = PacketSizeDw(header);
		if (size == 0 || dw + size > dcb.size()) {
			return fail((header >> 8u) & 0xffu, dw);
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
		if (m_compute_ring) {
			// MEC has no graphics engine. These opcodes hang a compute ring ("Illegal opcode").
			// SET_UCONFIG_REG / INDEX / INDIRECT are handled below: profiler UCONFIG is dropped,
			// geometry-engine UCONFIG still refuses the buffer.
			switch (op) {
				case IT_DRAW_INDIRECT:
				case IT_DRAW_INDEX_INDIRECT:
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
					dw += size;
					continue;
				default: break;
			}
		}
		switch (op) {
			case IT_NOP: {
				stats.nops++;
				const uint32_t marker = (header >> 2u) & 0x3fu;
				if (marker == R_FLIP) {
					stats.marker_flip++;
					if (labels) {
						r.cpu_tail.insert(r.cpu_tail.end(), dcb.begin() + dw, dcb.begin() + dw + size);
					}
				} else if (marker == R_ZERO && labels && body_count >= 1 && IsFlipMarker(bptr[0])) {
					stats.marker_flip++;
					r.cpu_tail.insert(r.cpu_tail.end(), dcb.begin() + dw, dcb.begin() + dw + size);
					if (bptr[0] != 0x68750777u && body_count >= 4) { // writes a label first, on the CPU
						note_write({static_cast<uint64_t>(bptr[1]) | (static_cast<uint64_t>(bptr[2]) << 32u), bptr[3], true,
						            false, false});
					}
				} else if (marker == R_RELEASE_MEM && labels) {
					stats.marker_release++;
					if (body_count < 7) {
						return fail(op, dw);
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
						r.unknown_ops.insert(op); // GDS data without a GDS partition: leave the buffer to Kyty
						break;
					}
					if (sel != 0 && sel != 1 && sel != 2 && sel != 3 && sel != 5) {
						r.unknown_detail = {bptr[0], bptr[1]};
						return fail(op, dw);
					}
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
							EmitTimestampRelease(body, dst, m_compute_ring);
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
					// Not the guest's packet (an ACB's carried the compute shader-type bit, which stopped the
					// gfx ring with "Illegal opcode in command stream"): the label is written with the guest's
					// event and cache actions plus write-backs (LabelEventCntl), 32-bit data to memory; a
					// 64-bit one as two 32-bit writes, the high dword first. GDS data goes as the translator
					// always ran it. A release that only
					// flushes caches becomes a flush into our scratch dword, unless a tracker point (also a
					// flush) follows anyway.
					const bool interrupt = int_sel == 1 || int_sel == 2 || int_sel == 4;
					if (sel == 1) {
						EmitRelease32(body, dst, bptr[4], m_compute_ring, LabelEventCntl(bptr[0]));
						mirror_store(dst, std::span<const uint32_t>(bptr + 4, 1), false);
						NoteOnce(0, m_tracker_writes ? "RELEASE_MEM 32-bit label (GPU store, mirrored to the CPU)"
						                             : "RELEASE_MEM 32-bit label (GPU store only)");
					} else if (sel == 2) {
						EmitRelease32(body, dst + 4, bptr[5], m_compute_ring, LabelEventCntl(bptr[0]));
						EmitRelease32(body, dst, bptr[4], m_compute_ring, LabelEventCntl(bptr[0]));
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
					if (interrupt) {
						TrackerPoint irq;
						irq.kind = TrackerPoint::Kind::Interrupt;
						irq.ctx  = bptr[6] & 0x07ffffffu;
						tracker_point(irq);
						stats.interrupts++;
						NoteOnce(16u + int_sel, int_sel == 1 ? "RELEASE_MEM INT_SEL 1 (EOP interrupt)"
						                                     : (int_sel == 2 ? "RELEASE_MEM INT_SEL 2 (EOP interrupt)"
						                                                     : "RELEASE_MEM INT_SEL 4 (interrupt only)"));
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
					unchanged = unchanged && PatchContext(base + (i - 1), bptr[i]) == bptr[i];
				}
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
						unchanged = unchanged && PatchSh(base + (i - 1), bptr[i]) == bptr[i];
					}
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
						break;
					}
				}
				// libSceAgc's internal-data tags (SQ_THREAD_TRACE_USERDATA_2, around waits and multi draws) go
				// to the gfx CP as written: a user IB writes them there (probe --danger tag342), while a
				// compute ring stops on the same packet.
				if (!m_compute_ring && op == IT_SET_UCONFIG_REG && body_count == 2 && Normalize(bptr[0]) == 0x342u) {
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
				// The gfx CP runs these packets itself (AMD's LOAD_CONTEXT/SH/UCONFIG_REG_INDEX, bit for bit;
				// probe --danger regindex-*: the table is loaded). The packet goes as written when every pair
				// is one the translator would emit unchanged: no stub entries (the CP would write them),
				// no patched values, UCONFIG only whitelisted and not VGT_PRIMITIVE_TYPE/INDEX_TYPE (those
				// need SET_UCONFIG_REG_INDEX), SH only of the header's shader type. The shadow copy is kept
				// from the same pairs. KYTY_NATIVE_GPU_TABLES_VERBATIM=0 expands every table as before.
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
				const bool compute_sh = (dcb[dw] & 2u) != 0;
				for (uint32_t i = 0; i < num && verbatim; i++) {
					const uint32_t raw   = pairs[2 * i];
					const uint32_t value = pairs[2 * i + 1];
					const uint32_t off   = Normalize(raw);
					if (raw != off) {
						verbatim = false; // also the 0xffffffff stub
					} else if (op == IT_SET_CONTEXT_REG_INDIRECT) {
						verbatim = PatchContext(off, value) == value;
					} else if (op == IT_SET_SH_REG_INDIRECT) {
						verbatim = off != kShNop && (off >= kShComputeBase) == compute_sh && PatchSh(off, value) == value;
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
							continue;
						}
						uc[off] = value;
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
					for (auto it = local_writes.rbegin(); it != local_writes.rend() && !decided; ++it) {
						uint64_t  v = 0;
						const int c = it->Covers(addr, wide, &v, mask);
						if (c != 0) {
							decided = true;
							plan    = c == 1 && w.Met(v) && it->gpu ? LabelOracle::Plan::Native : LabelOracle::Plan::Split;
						}
					}
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
					const uint64_t wait_pair = m_time_wait_va + 16ull * (m_time_waits % m_time_pairs);
					if (timed) {
						EmitClockCopy(body, wait_pair);
					}
					const bool emitted = plan == LabelOracle::Plan::Native && EmitNativeWait(body, w, m_compute_ring);
					if (timed) {
						EmitClockCopy(body, wait_pair + 8);
						m_time_wait_info[m_time_waits % m_time_pairs] = {addr, ref, stats.draws, met};
						m_time_waits++;
					}
					if (plan == LabelOracle::Plan::Native && !emitted) {
						plan = LabelOracle::Plan::Split;
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
					return fail(op, dw);
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
				// Games write their completion labels with WRITE_DATA. The native IB completes before
				// Kyty has even parsed this DCB, so a native label tells the game the GPU is done while
				// Kyty is still reading the command buffer, and the game starts rewriting it (Dead
				// Cells: a truncated packet in Kyty's parser on entering gameplay). Kyty performs the
				// write itself, in order, so the native IB leaves it out.
				if (body_count < 3 || !WriteDataToMemory(bptr[0])) {
					return fail(op, dw);
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
					return fail(op, dw);
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
					return fail(op, dw);
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
				// The PS5 command processor takes MAX_SIZE as the indices left from INDEX_OFFSET (games
				// send offsets past it); the BC-250's amdgpu microcode takes it as the whole buffer, and
				// such a draw hung the GPU (ASTRO BOT). Issue it as DRAW_INDEX_2 at the offset's address
				// instead, as Kyty's own command processor does.
				if (body_count < 4) {
					return fail(op, dw);
				}
				log_draw(op, bptr[2], 0, false);
				// KYTY_NATIVE_GPU_OFFSET_DRAW_VERBATIM=1 (experiment): the packet as the title wrote it, now that
				// libSceAgc (agc.cpp) writes MAX_SIZE as the end of the drawn range, which both semantics accept.
				// ASTRO BOT with it: no hang, but a black screen (index fetch differs; INDEX_BUFFER_SIZE?).
				static const bool offset_draw_verbatim = [] {
					const char* v = std::getenv("KYTY_NATIVE_GPU_OFFSET_DRAW_VERBATIM");
					return v != nullptr && v[0] == '1';
				}();
				if (skip_draw()) {
					stats.draws_skipped++;
				} else if (offset_draw_verbatim && bptr[0] >= bptr[1] + bptr[2]) {
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
					return fail(op, dw);
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
						return fail(op, dw);
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
							return fail(op, dw);
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
				// PIXEL_PIPE_STAT_DUMP (0x39) leaves memory untouched on the BC-250 DB. ZPASS_DONE (0x15) writes the
				// same thing: one 64-bit counter per RB, 16 bytes apart (STRIDE 2 of PS5's PIXEL_PIPE_STAT_CONTROL),
				// bit 63 set. KYTY_NATIVE_GPU_ZPASS_DUMP=0 sends 0x39 as the title wrote it.
				if (static const bool zpass_dump = [] {
					    const char* e = std::getenv("KYTY_NATIVE_GPU_ZPASS_DUMP");
					    return e == nullptr || e[0] != '0';
				    }();
				    zpass_dump && !m_compute_ring && body_count >= 3 && (bptr[0] & 0x3fu) == 0x39u) {
					body[body.size() - size + 1] = (bptr[0] & ~0x3fu) | 0x15u;
				}
				// The draw log keeps occlusion dumps with their address, so a flip dump can show the counters.
				if (body_count >= 3 && ((bptr[0] & 0x3fu) == 0x39u || (bptr[0] & 0x3fu) == 0x15u)) {
					log_draw(op, bptr[0] & 0x3fu, (static_cast<uint64_t>(bptr[1]) & ~7ull) | (static_cast<uint64_t>(bptr[2] & 0xffffu) << 32u), false);
				}
				// Events with an address (ZPASS_DONE occlusion counters, 16 DBs x begin/end) write there.
				if (body_count >= 3) {
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
			case IT_CLEAR_STATE:
				// The hardware resets the context to amdgpu's clear state, the PS5 CP to its own
				// defaults (Kyty: the same as a context-state clear). Follow it with the defaults, every
				// one of them, since the hardware no longer matches the shadow.
				body.insert(body.end(), dcb.begin() + dw, dcb.begin() + dw + size);
				stats.clear_state++;
				for (const auto& [off, value]: m_cx_defaults) {
					EmitSet(body, IT_SET_CONTEXT_REG, off, PatchContext(off, value), false);
					mark_cx(off);
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
				break;
		}
		if (IsWork(op)) {
			seg_effects = true;
		}
		const bool same_seg = segs.size() == segs_before;
		if (in_place && (!same_seg || body.size() != body_before)) {
			return fail(op, dw); // a packet run in place has nothing of ours around it
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
	end_run();
	end_cut_run();
	if (direct) {
		end_direct();
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
	m_cx        = std::move(cx);
	m_saved_cx  = std::move(saved_cx);
	m_cx_pushed = cx_pushed;
	m_sh = std::move(sh);
	m_uc = std::move(uc);
	// The IB starts where the previous DCB left the CP, not where this one leaves it.
	const Pm4CpDrawState cp_start = m_cp;
	m_cp                          = cp;

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
			if (k == 0) {
				EmitPreamble(r.ib, stats, m_cx, m_sh, m_uc, cp_start);
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
			r.segments.push_back(std::move(s));
		}
	} else {
		r.ib.reserve(body.size() + (m_cx.size() + m_sh.size() + m_uc.size()) * 3 + 8);
		EmitPreamble(r.ib, stats, m_cx, m_sh, m_uc, cp_start);
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
