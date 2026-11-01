#ifndef EMULATOR_SRC_GRAPHICS_NATIVE_GPU_NATIVEPM4_H_
#define EMULATOR_SRC_GRAPHICS_NATIVE_GPU_NATIVEPM4_H_

#include <array>
#include <string>
#include <bit>
#include <bitset>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <map>
#include <mutex>
#include <set>
#include <unordered_set>
#include <memory>
#include <span>
#include <type_traits>
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

// Shadow of one register space: offset -> value, iterated in ascending offset order like the
// std::map it replaces. Offsets below kDense (every context and SH register real packets write)
// live in a flat array with a presence bitmap, so lookups are O(1) and the copies the translator
// makes per DCB and per context push are a few KiB of memcpy instead of a tree; anything above
// falls back to a map.
class RegFile {
public:
	static constexpr uint32_t kDense = 0x400;

	class Iterator {
	public:
		Iterator(const RegFile* f, uint32_t off, std::map<uint32_t, uint32_t>::const_iterator it)
		    : m_f(f), m_off(off), m_it(it) {
			SkipAbsent();
		}
		std::pair<uint32_t, uint32_t> operator*() const {
			if (m_off < kDense) {
				return {m_off, m_f->m_values[m_off]};
			}
			return {m_it->first, m_it->second};
		}
		Iterator& operator++() {
			if (m_off < kDense) {
				m_off++;
				SkipAbsent();
			} else {
				++m_it;
			}
			return *this;
		}
		bool operator!=(const Iterator& o) const { return m_off != o.m_off || m_it != o.m_it; }

	private:
		void SkipAbsent() {
			while (m_off < kDense) {
				const uint64_t bits = m_f->m_present[m_off / 64] >> (m_off % 64);
				if (bits != 0) {
					m_off += static_cast<uint32_t>(std::countr_zero(bits));
					return;
				}
				m_off = (m_off / 64 + 1) * 64;
			}
		}
		const RegFile*                                m_f;
		uint32_t                                      m_off;
		std::map<uint32_t, uint32_t>::const_iterator m_it;
	};

	uint32_t& operator[](uint32_t off) {
		if (off >= kDense) {
			m_count += m_overflow.count(off) == 0 ? 1 : 0;
			return m_overflow[off];
		}
		if (!Has(off)) {
			m_present[off / 64] |= 1ull << (off % 64);
			m_values[off] = 0;
			m_count++;
		}
		return m_values[off];
	}
	// nullptr when the register was never written.
	[[nodiscard]] const uint32_t* Find(uint32_t off) const {
		if (off >= kDense) {
			auto it = m_overflow.find(off);
			return it == m_overflow.end() ? nullptr : &it->second;
		}
		return Has(off) ? &m_values[off] : nullptr;
	}
	[[nodiscard]] size_t count(uint32_t off) const { return Find(off) != nullptr ? 1 : 0; }
	void                 emplace(uint32_t off, uint32_t value) {
		if (count(off) == 0) {
			(*this)[off] = value;
		}
	}
	void clear() {
		m_present.fill(0);
		m_overflow.clear();
		m_count = 0;
	}
	[[nodiscard]] size_t   size() const { return m_count; }
	[[nodiscard]] Iterator begin() const { return {this, 0, m_overflow.begin()}; }
	[[nodiscard]] Iterator end() const { return {this, kDense, m_overflow.end()}; }

private:
	[[nodiscard]] bool Has(uint32_t off) const { return ((m_present[off / 64] >> (off % 64)) & 1u) != 0; }

	std::array<uint32_t, kDense>      m_values {};
	std::array<uint64_t, kDense / 64> m_present {};
	std::map<uint32_t, uint32_t>      m_overflow;
	size_t                            m_count = 0;
};

struct Pm4TranslateStats {
	uint32_t packets           = 0; // type-3 packets seen in the DCB
	// Direct execution: the dwords that went into the IB exactly as the title wrote them (same packet,
	// nothing inserted around it, no new segment), and the runs they form. A run is what an
	// INDIRECT_BUFFER into guest memory could execute in place.
	uint32_t verbatim_dw       = 0;
	uint32_t total_dw          = 0;
	uint32_t verbatim_runs     = 0;
	std::array<uint32_t, 4> run_hist {}; // runs under 16, 64, 256 dwords, and longer
	std::array<uint32_t, 256> changed_ops {}; // packets that did not go through verbatim, by opcode
	std::array<uint32_t, 256> ops {};         // every packet, by opcode
	// Direct execution (the GPU running the guest buffer in place): the stretches it would run between
	// cuts, if NOPs that emit nothing stayed in place and a packet only preceded by ours ran as itself;
	// the packets replaced, and the ones only preceded by an insertion of ours, by opcode.
	uint32_t                  cut_runs = 0;
	std::array<uint32_t, 256> cut_ops {};
	std::array<uint32_t, 256> inserted_ops {};
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
	uint32_t load_gpu_written  = 0; // indirect loads from memory a WRITE_DATA/DMA_DATA of the same DCB wrote
	uint32_t loaded_cx_regs    = 0; // offset/value pairs expanded from 0x9F
	uint32_t loaded_sh_regs    = 0; // offset/value pairs expanded from 0x63
	uint32_t loaded_uc_regs    = 0; // offset/value pairs expanded from 0x64
	uint32_t loads_verbatim    = 0; // 0x63/0x64/0x9F run by the CP as written (their pairs counted above too)
	uint32_t direct_ib2        = 0; // SetDirectRuns: INDIRECT_BUFFER calls into the guest's buffer
	uint32_t direct_dw         = 0; // ... and the dwords they run in place
	uint32_t draws             = 0; // IT_DRAW_INDEX_2
	uint32_t dispatches        = 0; // IT_DISPATCH_DIRECT / IT_DISPATCH_INDIRECT
	uint32_t num_instances     = 0; // IT_NUM_INSTANCES passed through
	uint32_t index_state       = 0; // IT_INDEX_BASE / IT_INDEX_BUFFER_SIZE passed through
	uint32_t draws_skipped     = 0; // draws left out (SetSkippedPixelShaders)
	uint32_t fmask_fixed       = 0; // CB_COLORn_FMASK set to the target base (was 0) before a draw
	uint32_t lod_stats_dropped = 0; // GET_LOD_STATS packets left out
	uint32_t lod_stats_issued  = 0; // GET_LOD_STATS packets run on the GPU (SetLodStats)
	uint32_t dma_gds_dst       = 0; // DMA_DATA writing GDS (counter resets)
	uint32_t dma_gds_src       = 0; // DMA_DATA reading GDS (counters copied out)
	uint32_t context_ops       = 0; // Agc context-state push/pop/clear markers applied
	uint32_t wait_barriers     = 0; // guest waits replaced by an end-of-pipe fence and a wait on it
	uint32_t gds_releases      = 0; // RELEASE_MEM markers that send GDS data, run natively
	uint32_t packets_skipped   = 0; // packets left out (SetSkippedOpcodes)
	uint32_t set_base          = 0; // IT_SET_BASE passed through (indirect argument base)
	uint32_t pfp_sync_me       = 0; // IT_PFP_SYNC_ME passed through
	uint32_t acquire_mem       = 0; // IT_ACQUIRE_MEM
	uint32_t event_write       = 0; // IT_EVENT_WRITE
	// Histograms of the event control dword (bptr[0]) of EVENT_WRITE (0x46) and any raw RELEASE_MEM
	// (0x49): which CB/DB meta/EOP events the title issues. EVENT_TYPE is bits [5:0], EVENT_INDEX [11:8].
	std::array<uint32_t, 64> event_type  {}; // count by EVENT_TYPE (0x2e CB_META, 0x2c DB_META, 0x14 TS...)
	std::array<uint32_t, 16> event_index {}; // count by EVENT_INDEX (0 generic, 5 end-of-pipe, 6 end-of-shader)
	uint32_t write_data_dropped = 0; // IT_WRITE_DATA to memory, left to Kyty (see Translate)
	uint32_t write_data_native  = 0; // IT_WRITE_DATA to memory also kept in the IB (SetWriteDataNative)
	uint32_t gds_dma_synced     = 0; // DMA_DATA with GDS given CP_SYNC (SetGdsDmaSync)
	uint32_t labels_native     = 0; // native labels: WRITE_DATA / RELEASE_MEM label writes run on the GPU
	uint32_t write_data_dw     = 0; // native labels: data dwords of the WRITE_DATA run on the GPU
	uint32_t labels_cpu        = 0; // native labels: label writes the tracker performs on the CPU (unreachable, or with
	                                // SetTrackerWrites a mirror of a GPU store)
	uint32_t timestamps        = 0; // native labels: timestamps the tracker still writes (CPU clock)
	uint32_t timestamps_native = 0; // native labels: timestamps the GPU writes (DATA_SEL 3)
	uint32_t interrupts        = 0; // native labels: RELEASE_MEM interrupts raised through the tracker
	uint32_t interrupts_gpu    = 0; // and those the GPU raises where they run (SetGuestInterruptsOnGpu)
	uint32_t waits_native      = 0; // native labels: guest waits passed through (an earlier GPU write satisfies them)
	uint32_t label_wb_stripped = 0; // label releases whose added write-backs a wait of the same buffer took off
	uint32_t waits_split       = 0; // native labels: guest waits the IB is cut at (no known writer yet)
	uint32_t dma_data          = 0; // IT_DMA_DATA memory/immediate -> memory passed through
	uint32_t cp_dma_idle       = 0; // zero-byte DMA_DATA with CP_SYNC before a job end after a DMA without one
	uint32_t waits_dropped     = 0; // WAIT_REG_MEM(_64) on memory, dropped in shadow mode
	uint32_t waits_unmet       = 0; // of those, condition not yet true when the DCB was translated
	uint32_t pfp_syncs_emitted = 0; // PFP_SYNC_ME emitted in place of dropped PFP waits
	uint32_t clear_state       = 0; // IT_CLEAR_STATE
	uint32_t context_checks    = 0; // libSceAgc's COND_EXEC + RELEASE_MEM around a context push/pop, left out
	uint32_t uconfig_skipped   = 0; // non-whitelisted UCONFIG writes dropped for board safety
	uint32_t preamble_cx       = 0; // context registers emitted in the IB preamble
	uint32_t preamble_sh       = 0; // SH registers emitted in the IB preamble
	uint32_t preamble_uc       = 0; // UCONFIG registers emitted in the IB preamble
	// CP register shadowing (Pm4Translator::SetCpShadow): preambles that turn it on, the dwords of their
	// LOAD_*_REG packets and the registers those load, and the classes that had no key to load (an anchor).
	uint32_t cp_shadow_preambles = 0;
	uint32_t cp_shadow_load_dw   = 0;
	uint32_t cp_shadow_load_regs = 0;
	uint32_t cp_shadow_anchors   = 0;
	// Context pushes and pops done on the GPU through S (SetCpShadowPushPop): DMA_DATA of S's context region to
	// the save area and back, then a LOAD_CONTEXT_REG.
	uint32_t cp_shadow_pushes = 0;
	uint32_t cp_shadow_pops   = 0;
	// Load mode: FMASK = BASE fixes done on the GPU from S (some of the slot's CB registers came from a verbatim
	// table, so the GPU may hold other values than the copy), and display-DCC fixes done on the GPU from S (the
	// copy's CB_COLORn_INFO came from one).
	uint32_t cp_shadow_fmask_gpu = 0;
	uint32_t cp_shadow_dcc_gpu   = 0;
	uint32_t cp_shadow_dcc_back  = 0; // and the DCC given back on the GPU (EmitCpShadowDccRestore)
};

// KYTY_NATIVE_GPU_AUDIT (Pm4Translator::SetAudit): the title's sync packets field by field, from its own dwords (not
// what the IB got), for the backend's run totals. T is uint32_t in a result, uint64_t in the backend's sums.
enum class AuditSync : uint8_t { Wait, Wait64, Release, ReleaseMarker, WriteData, Acquire, Count };
template <typename T>
struct SyncAuditT {
	static constexpr size_t kForms = static_cast<size_t>(AuditSync::Count);
	std::array<T, kForms>                packets {};
	std::array<std::array<T, 4>, kForms> policy {};      // CACHE_POLICY [26:25] (ACQUIRE_MEM has none: 0)
	std::array<std::array<T, 4>, kForms> engine {};      // ENGINE_SEL: waits [9:8], WRITE_DATA [31:30], ACQUIRE_MEM bit 31
	std::array<T, kForms>                odd {};         // packets with a bit set outside the fields the layout names
	std::array<T, kForms>                policy_lost {}; // CACHE_POLICY not 0, and the GPU runs a packet of the translator's
	std::array<std::array<T, 8>, 2>      wait_function {};  // [64-bit]
	std::array<std::array<T, 4>, 2>      wait_mem_space {};
	std::array<std::array<T, 4>, 2>      wait_operation {};
	std::array<std::array<T, 64>, 2>     release_event {};  // [Kyty's NOP marker] EVENT_TYPE
	std::array<std::array<T, 13>, 2>     release_gcr {};    // GCR_CNTL bit i (event dword bit 12 + i)
	std::array<std::array<T, 4>, 2>      release_dst_sel {};
	std::array<std::array<T, 8>, 2>      release_int_sel {};
	std::array<std::array<T, 8>, 2>      release_data_sel {};
	std::array<T, 16>                    write_dst_sel {};
	T                                    write_confirm = 0;
	T                                    write_markers = 0; // libSceAgc's debug-marker strings, not in the WRITE_DATA counts
	std::array<T, 19>                    acquire_gcr {};    // GCR_CNTL (body dword 6) bit i
	std::array<uint32_t, kForms>         odd_bits {};       // those bits, ORed (a release's event dword)
	std::array<uint32_t, 2>              release_select_odd {}; // and a release's select dword

	template <typename U>
	void Add(const SyncAuditT<U>& o) {
		Sum(packets, o.packets);
		Sum(policy, o.policy);
		Sum(engine, o.engine);
		Sum(odd, o.odd);
		Sum(policy_lost, o.policy_lost);
		Sum(wait_function, o.wait_function);
		Sum(wait_mem_space, o.wait_mem_space);
		Sum(wait_operation, o.wait_operation);
		Sum(release_event, o.release_event);
		Sum(release_gcr, o.release_gcr);
		Sum(release_dst_sel, o.release_dst_sel);
		Sum(release_int_sel, o.release_int_sel);
		Sum(release_data_sel, o.release_data_sel);
		Sum(write_dst_sel, o.write_dst_sel);
		Sum(write_confirm, o.write_confirm);
		Sum(write_markers, o.write_markers);
		Sum(acquire_gcr, o.acquire_gcr);
		for (size_t i = 0; i < kForms; i++) {
			odd_bits[i] |= o.odd_bits[i];
		}
		release_select_odd[0] |= o.release_select_odd[0];
		release_select_odd[1] |= o.release_select_odd[1];
	}

private:
	template <typename A, typename B>
	static void Sum(A& to, const B& from) {
		if constexpr (std::is_arithmetic_v<A>) {
			to += from;
		} else {
			for (size_t i = 0; i < to.size(); i++) {
				Sum(to[i], from[i]);
			}
		}
	}
};
using Pm4SyncAudit = SyncAuditT<uint32_t>;

// Why a buffer was refused (SetAudit: Pm4TranslateResult::refusals); the backend's own refusals after a translation
// (the packet check, GDS shaders) follow the translator's.
enum class RefuseWhy : uint8_t {
	Collected, // the end of the walk, refusing for the packets collected on the way (each recorded where it was)
	NotType3,
	Truncated,
	Unknown,
	GraphicsOnMec,
	UconfigOnMec,
	ShortBody,
	ReleaseDataSel,
	ReleaseGds,
	RegisterWait,
	DmaForm,
	WriteDataRegister,
	ContextControlOnMec,
	CondExecForm,
	AtomicForm,
	InPlace,
	Validate,
	ValidateResume,
	GdsShader,
	GdsShaderLearn,
	Count
};
const char* RefuseWhyName(RefuseWhy why);
// The first packet of one opcode and reason that refused a buffer, and how many of them did.
struct Pm4Refusal {
	uint32_t                op    = 0; // 0x100: not a type-3 header
	RefuseWhy               why   = RefuseWhy::Unknown;
	uint32_t                count = 0;
	uint32_t                dw    = 0; // the first one's dword offset
	uint32_t                last  = 0; // the latest one's (a packet refusing for each of its entries counts once)
	uint32_t                nd    = 0;
	std::array<uint32_t, 5> d {};      // its header and leading body dwords
};

// CP register shadowing (Pm4Translator::SetCpShadow, KYTY_NATIVE_GPU_CP_SHADOW). With the CONTEXT_CONTROL
// shadow enables on, the CP writes every register write it runs (SETs, 0x9F/0x63/0x64 tables, in IB2s too)
// into a shadow buffer S as well, at the class's base, set by the job's last LOAD_*_REG that loads, plus the
// register's offset in its space (probe shadow): SH registers at S + 0, context at S + 0x1000, UCONFIG at
// S + 0x9000 (Mesa's layout). The values below are dword offsets in S.
constexpr uint32_t kCpShadowShDw      = 0;
constexpr uint32_t kCpShadowContextDw = 0x400;
constexpr uint32_t kCpShadowUconfigDw = 0x2400;
// What the translator writes and compares: offsets below 0x400 of every class.
constexpr uint32_t kCpShadowDwords = 0x2800;
// Where a 16-bit register offset of any class can reach (S + 0x9000 + 4 * 0xffff < S + 0x49000).
constexpr uint64_t kCpShadowBytes = 0x49000;
// The save area P of a pushed context (SetCpShadowPushPop) right after S: S's context region, 0x000-0x3FF.
constexpr uint64_t kCpShadowSaveOffset = kCpShadowBytes;
constexpr uint32_t kCpShadowSaveBytes  = 0x1000;
// After P, the register tables of the GPU-side FMASK fix (Pm4Translator::SetCpShadow, load mode): per MRT slot n
// at kCpShadowFixOffset + 32 * n, {CB_COLORn_FMASK, BASE, CB_COLORn_FMASK_EXT, BASE_EXT} and then {.., FMASK, ..,
// FMASK_EXT}; then those of the GPU-side display-DCC fix, per slot at kCpShadowDccFixOffset + 16 * n,
// {CB_COLORn_INFO, the CP's INFO without DCC} and {CB_COLORn_INFO, the CP's INFO as it was}. The offsets are
// seeded once (CpShadowFixTables), the values copied from S by the GPU. Then a zero word, seeded 0 and never
// written, copied for a register S has no value for (amdgpu's clear state, 0, on the GPU).
constexpr uint64_t kCpShadowFixOffset    = kCpShadowSaveOffset + kCpShadowSaveBytes;
constexpr uint64_t kCpShadowDccFixOffset = kCpShadowFixOffset + 0x100;
constexpr uint32_t kCpShadowFixBytes     = 0x180;
// The display-DCC fix's ATOMIC_MEM: TC_OP_ATOMIC_AND_32, COMMAND 2 as libSceAgc's, which clears DCC_ENABLE in the copied
// INFO, with ENGINE_SEL the ME, after the ME's copy (libSceAgc's form sets bit 30, the PFP, which runs ahead of the ME:
// the AND came before the copy). Probes atomicand, atomicand-order.
constexpr uint32_t kCpShadowDccAnd = (2u << 8u) | 0x55u;
constexpr uint64_t kCpShadowZeroOffset = kCpShadowFixOffset + kCpShadowFixBytes;
std::array<uint32_t, kCpShadowFixBytes / 4u> CpShadowFixTables();
// What the backend reserves for S, P, the fix tables and the zero word, and the packet check guards.
constexpr uint64_t kCpShadowReserveBytes = kCpShadowZeroOffset + 16u;
// The push's and the pop's DMA_DATA: the micro engine, CP_SYNC (the ME waits for the copy), source and
// destination through L2 (SRC_SEL/DST_SEL 3, as Mesa's CP DMA on GFX9 and up); RAW_WAIT, 4 KiB.
constexpr uint32_t kCpShadowDmaControl = (1u << 31u) | (3u << 29u) | (3u << 20u);
constexpr uint32_t kCpShadowDmaCommand = (1u << 30u) | kCpShadowSaveBytes;
constexpr uint32_t kCpShadowDmaWordCommand = (1u << 30u) | 4u; // one register, for the FMASK fix
// CONTEXT_CONTROL load and shadow enables of a shadowing job: update (bit 31), context (1), UCONFIG (15),
// gfx SH (16) and CS SH (24); no global config, no CE RAM.
constexpr uint32_t kCpShadowEnables = 0x81018002u;
// Register classes (SetCpShadow's `load_classes`): context, graphics SH (below 0x200), compute SH (0x200 up)
// and UCONFIG.
constexpr uint32_t kCpShadowContext = 1u;
constexpr uint32_t kCpShadowGfxSh   = 2u;
constexpr uint32_t kCpShadowCsSh    = 4u;
constexpr uint32_t kCpShadowUconfig = 8u;
constexpr uint32_t kCpShadowClasses = 15u;

// Off: the preambles replay the CPU copy, as before. Record: every preamble turns shadowing on and LOADs
// the keys from S, so the CP has a base for every class and keeps S current, then replays the copy as
// before; what the GPU draws stays the same. Load: the keys come from S, and only what S cannot restore is
// SET from the copy.
enum class CpShadowMode { Off, Record, Load };

// Shadowing and loading off: CONTEXT_CONTROL 0x80000000 / 0x80000000 and five filler NOPs, 8 dwords. Every gfx
// job ends with it, whether CP shadowing is on or not: the enables are ring-global, amdgpu's CONTEXT_CONTROL
// before each job leaves them alone, and another process's SETs would be shadowed at our address in its own
// VM (a gfx VM fault that freezes the board). The empty stream trampoline and the exit guards' stub are it.
constexpr std::array<uint32_t, 8> kCpShadowOff = {0xc0012800u, 0x80000000u, 0x80000000u, 0xffff1000u,
                                                  0xffff1000u, 0xffff1000u, 0xffff1000u, 0xffff1000u};
void EmitCpShadowOff(std::vector<uint32_t>& ib);
// True when the last packet of `ib` before trailing fillers is that CONTEXT_CONTROL, or an INDIRECT_BUFFER chain
// that is not an exit guard's (a streamed DCB's head and parts chain on, and their last part ends with it).
bool CpShadowTailOk(std::span<const uint32_t> ib);
// The shadow buffer the packet check lets LOAD_*_REG name, and with it the shadowing CONTEXT_CONTROL (0: none).
void SetCpShadowValidation(uint64_t va);

// A piece of a gfx ring's batch as CP shadowing sees it (Shadow::AppendBatch).
struct CpShadowPiece {
	bool                      own  = false; // the shadowing translator made it (Pm4TranslateResult::cp_shadow)
	bool                      head = false; // it starts with a shadow preamble (Pm4Segment::cp_shadow_head)
	std::span<const uint32_t> resume;       // own without a head: what restores where it starts (cp_shadow_resume)
};

// Assembles one gfx ring's batch while CP shadowing is on. A piece another translator made must not run with
// our shadowing on (its SETs would land in S); a piece of ours without a preamble (a streamed chunk) carries
// on from the piece before it, which holds only inside a job of ours whose shadowing is on, so after a foreign
// piece it gets its resume. At the front of a batch that depends on how the batch is flushed: a CS of its own
// starts with shadowing off and no bases, a stream part carries on from the stream's previous part.
class CpShadowBatch {
public:
	// Appends `ib` to `batch`, with shadowing off before a foreign piece and the resume before a piece of ours
	// without a head that follows one; at the batch front the resume is kept for the flush.
	void Append(std::vector<uint32_t>& batch, std::span<const uint32_t> ib, const CpShadowPiece& piece);
	// The dwords Append and the flush may add for `piece` besides its own, and for what is kept so far.
	[[nodiscard]] size_t Extra(const CpShadowPiece& piece) const;
	// What goes in front of the batch when it is flushed: as a CS of its own (`stream_part` false), or as the next
	// part of a stream whose shadowing is `stream_on` where the part begins.
	[[nodiscard]] std::span<const uint32_t> Front(bool stream_part, bool stream_on) const;
	// Whether shadowing is on at the end of the batch, `start_on` if it holds no piece.
	[[nodiscard]] bool EndOn(bool start_on) const { return m_any ? m_on : start_on; }
	[[nodiscard]] bool Empty() const { return !m_any; }
	void               Reset();
	// The batch already holds pieces appended before shadowing was on: they leave it off.
	void               HoldsForeign() {
		m_any = true;
		m_on  = false;
	}
	uint64_t           guards  = 0; // shadowing turned off before a foreign piece
	uint64_t           resumes = 0; // resumes put in front of a piece of ours inside the batch

private:
	bool                  m_any = false;
	bool                  m_on  = false;
	std::vector<uint32_t> m_front; // the front piece's resume, until the flush
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

// User data of a draw or dispatch whose shader is under inspection (SetInspectedShaders).
struct DrawSnapshot {
	uint64_t ps = 0;
	uint64_t es = 0;
	uint32_t ps_user_data[32] = {};
	uint32_t gs_user_data[32] = {}; // SPI_SHADER_USER_DATA_GS_*: the NGG (merged ES/GS) stage on gfx10
};

// One draw or dispatch as the translator saw it (SetDrawLog), for comparing frames: which shaders,
// targets and counts it used. `indirect` is the argument address of an indirect draw or dispatch
// (its values are read when the frame is dumped, after the GPU wrote them).
struct DrawRecord {
	uint32_t op        = 0;
	uint32_t index     = 0; // draw or dispatch number in the buffer
	uint32_t count     = 0; // vertices / indices (direct draws), threadgroups X (direct dispatches)
	uint32_t instances = 1;
	uint64_t ps        = 0; // pixel shader (draws) or compute shader (dispatches)
	uint64_t gs        = 0; // the NGG ES/GS stage
	uint64_t cb0       = 0; // first bound color target
	uint64_t zbase     = 0; // depth read base
	uint64_t indirect  = 0;
	uint32_t ud_hash   = 0; // FNV of the PS (or compute) user data
	uint32_t gs_hash   = 0; // FNV of the GS user data
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

// Native labels (Pm4Translator::SetNativeLabels). The GPU writes the guest's completion labels
// itself, and the CPU-side effects of a buffer (EOP interrupts, labels the GPU cannot write) hang off
// tracker points: end-of-pipe RELEASE_MEMs that write a sequence number into a slot the backend
// polls. Guest waits pass through when an earlier write in ring order satisfies them; otherwise the
// IB is cut there and the rest is held back until the label is written.

// A guest WAIT_REG_MEM(_64) on memory: (value & mask) <function> ref.
struct WaitCond {
	uint64_t addr     = 0;
	uint64_t ref      = 0;
	uint64_t mask     = ~0ull;
	uint32_t function = 3;
	bool     wide     = false; // WAIT_REG_MEM_64
	bool     pfp      = false; // ENGINE_SEL = PFP

	[[nodiscard]] bool Met(uint64_t value) const;
};

// A label write a buffer performs, for deciding who satisfies later waits.
struct LabelWrite {
	uint64_t addr  = 0;
	uint64_t value = 0;
	bool     known = true;  // false: the value is not known when translating (GDS data, timestamps)
	bool     gpu   = true;  // written by the GPU (false: by the tracker or Kyty on the CPU)
	bool     wide  = false; // 64 bits
	// A WRITE_DATA of more than two dwords (titles build blocks of data with it, ASTRO BOT ~450000
	// dwords a frame) is one write of `span_dw` dwords. `data` points at them while the buffer is being
	// translated; a segment keeps the write without it, as touched but unknown.
	uint32_t        span_dw = 0;
	const uint32_t* data    = nullptr;

	// What a WAIT at `a` (64-bit when `wide_wait`) reads once this write has landed: 0 unrelated,
	// 1 the value in `out`, 2 touched but unknown.
	// `wait_mask`: the wait's compare mask. A 64-bit wait that masks off the high half reads only the
	// low one, so a 32-bit write there decides it (ASTRO BOT waits on 64-bit labels with mask
	// 0xffffffff and writes them as two 32-bit halves).
	int Covers(uint64_t a, bool wide_wait, uint64_t* out, uint64_t wait_mask = ~0ull) const;
};

struct TrackerPoint {
	enum class Kind : uint8_t {
		End,       // the end of a segment (the last one: the buffer is done)
		Interrupt, // RELEASE_MEM INT_SEL: raise the queue's EOP interrupt with `ctx`
		           // (points with nothing between them share one packet, i.e. one `dw`)
		CpuWrite,  // write `data` (dwords) at `addr`, all to one address when `one_addr`
		Timestamp, // write the reference clock at `addr` (`size` bytes)
		LodStats,  // fill `size` bytes at `addr` like Kyty's GET_LOD_STATS
		Wake,      // the GPU has written the label at `addr`: wake a guest sleeping on it, store nothing
		Flip,      // AgcDcbSetFlip: request the flip `data` holds (handle, index, mode, arg low, arg high)
	};
	Kind                  kind     = Kind::End;
	uint32_t              dw       = 0; // the point's 32-bit sequence value in the IB (patched before submission)
	uint32_t              ctx      = 0;
	uint64_t              addr     = 0;
	uint32_t              size     = 0;
	bool                  one_addr = false;
	bool                  mirror   = false; // also stored by the GPU; publish on the CPU only if the guest has not replaced it
	                                        // (SetTrackerWrites only)
	std::vector<uint32_t> data;
	std::vector<uint32_t> prior; // dwords at `addr` when the buffer was translated; empty: write unconditionally
};

// One self-contained piece of a translated buffer, IB dwords [begin, end) of Pm4TranslateResult::ib.
// Every segment after the first starts held, at a guest wait that no earlier write satisfied: its
// IB begins with the preamble, a barrier block (our fence) and the guest wait, so the backend can
// keep the wait (a GPU write ahead in the ring will satisfy it) or the barrier (the label was seen
// written) and NOP the other.
struct Pm4Segment {
	uint32_t                  begin       = 0;
	uint32_t                  end         = 0;
	bool                      held        = false;
	WaitCond                  wait;
	uint32_t                  barrier_dw  = 0;
	uint32_t                  barrier_len = 0;
	uint32_t                  wait_dw     = 0;
	uint32_t                  wait_len    = 0;
	std::vector<TrackerPoint> points;
	std::vector<LabelWrite>   writes;
	// The guest waits this segment gives the GPU, in order (the held wait not among them): what a stop
	// has to release, and only while the segment has not finished (SetExitGuard).
	std::vector<WaitCond>     gpu_waits;
	// Compute rings: the EXEC_COUNT dwords of the segment's exit guards, as IB offsets (the backend
	// needs none of them; tests read them).
	std::vector<uint32_t>     guards;
	// CP shadowing: the segment starts with a shadow preamble (its exit guard aside), so it restores its
	// state wherever it runs. A streamed chunk's first segment has none.
	bool                      cp_shadow_head = false;
	// SetAudit: the guest waits the segment runs as GPU waits or barriers, in order (the held wait not among them),
	// for the backend's measurements; nothing in the IB depends on it.
	struct AuditWait {
		WaitCond wait;
		uint32_t next_point = 0;     // the first point after it, an index in `points`
		bool     barrier    = false; // LabelOracle::Plan::Met: a barrier (else the wait itself)
		bool     local      = false; // a write earlier in the buffer decided it
		bool     met        = false; // memory met it when it was translated
	};
	std::vector<AuditWait>    audit_waits;
};

// Rewrites `len` dwords at `ib[dw]` as one-dword filler NOPs.
void NopOut(std::span<uint32_t> ib, uint32_t dw, uint32_t len);

// Cuts `ib` at packet boundaries into consecutive pieces (offset, length in dwords) of at most
// `max_dw` dwords each. Empty when a single packet is longer than that.
std::vector<std::pair<size_t, size_t>> SplitIbAtPackets(std::span<const uint32_t> ib, size_t max_dw);

// Which ring an IB is about to be submitted to. Compute allows the same RELEASE_MEM / WAIT_REG_MEM
// bodies as the gfx ring, with the shader-type bit set, and rejects graphics-only packets (a draw or
// PFP_SYNC_ME on MEC is an illegal opcode).
enum class NativeIbKind { Gfx, Compute };

// The native-labels packet check: true when every packet of `ib` is one the translator has run on the
// board before (in particular RELEASE_MEM and WAIT_REG_MEM only in the barrier's forms, NOPs only as
// fillers, GET_LOD_STATS only with `lod_stats`), so an "Illegal opcode in command stream" cannot come
// from a form never tried. On failure `bad_dw` is the offending packet.
bool ValidateNativeIb(std::span<const uint32_t> ib, bool lod_stats, uint32_t* bad_dw = nullptr,
                      NativeIbKind kind = NativeIbKind::Gfx);

// How a held guest wait may leave the CPU (KYTY_NATIVE_GPU_ACB_RINGS). The GPU is only given the
// wait once a writer that satisfies it is guaranteed to run:
//   same ring — the write is already ahead in that ring's batch, so the waiter is appended behind it;
//   another ring — only after that writer's CS ioctl has returned, so a compute ring cannot block on
// a label the gfx ring has not been asked to write (and the other way around).
// Until then the segment stays on the CPU (Hold). A label already in memory, and no pending write
// that will replace it, is a barrier: a wait whose writer will never be submitted does not block the
// GPU. `cover` is LabelWrite::Covers (0 unrelated, 1 known value, 2 touched but unknown).
// `gfx_writer`: the waiter is on a compute ring and the pending write is on the gfx ring, emitted
// into its batch but not submitted yet. That wait goes to the GPU at once: the gfx ring is another
// engine, so the write cannot be queued behind the waiter, whichever compute ring the kernel runs
// the waiter on (it balances contexts across them, so two compute rings may be one), and the batch
// is submitted whatever the compute ring does. With no write emitted yet the segment stays held:
// the label's previous writer says nothing about this use (ASTRO BOT cycles three label sets; a
// compute wait given to the GPU on the strength of the last frame's gfx writer deadlocked against a
// gfx wait on that compute ring's later write).
// `shared_ring`: the waiter's ring runs more than one guest queue (the AGR queue in queue 0's gfx ring).
// A wait there on another ring's write stays held until that write lands, then becomes a barrier: the
// writer may itself wait for the other guest queue's work queued behind this waiter, which on the PS5
// runs on a ring of its own.
enum class RingWaitRelease { Hold, GpuWait, Barrier };

RingWaitRelease PlanRingWait(int cover, bool value_satisfies, bool gpu_write, bool same_ring, bool writer_submitted,
                             bool label_in_memory, bool native_wait, bool gfx_writer = false, bool shared_ring = false);

// Guest compute queues are 0x20 + n (7 pipes x 8 queues). `available_rings` is the
// AMDGPU_INFO_HW_IP_INFO bitmask. Returns one of its set bits; queues spread across the rings the
// kernel exposes. No bit set: 0.
uint32_t ComputeRingForQueue(uint32_t guest_queue, uint32_t available_rings);

// The tessellation rings the title gave libSceAgcDriver (sceAgcDriverSetTFRing and
// sceAgcDriverSetHsOffchipParam), which the PS5 driver programs. Here every gfx IB's preamble writes
// them: the registers are global, and other processes on this ring program their own in between.
// Without them the GE writes tess factors at GPU address 0, faults, and the gfx ring hangs.
void SetTessellationRings(uint64_t tf_ring_base, uint32_t tf_ring_bytes, uint32_t hs_offchip_param);

// The longest stretch a compute exit guard's COND_EXEC skips (probe condexec: 20005 dwords skip,
// 70005 do not).
constexpr uint32_t kExitGuardMaxSkip = 20000;

// A value that satisfies `w`, given the label's `current` value (only the masked bits change).
uint64_t SatisfyingValue(const WaitCond& w, uint64_t current);

// A guest command buffer with its INDIRECT_BUFFER calls and chains inlined, as the CP would fetch it:
// a call runs the target and comes back, a chain (CHAIN bit) runs the target instead of the rest.
// REWIND packets already valid are dropped: they only make the CP refetch, and the translator copies
// the commands anyway. Titles write their command buffers in chunks linked this way. Returns false
// (`out` cleared) when there is nothing to inline, or something it cannot: a REWIND still waiting for
// the CPU, another IB form, or a chain deeper than 16 or longer than 4M dwords.
// `why`, when given, says what stopped it (empty when there was nothing to inline).
// Where a stretch of a buffer sits in guest memory: `len` dwords from dword `dw` of the buffer at GPU
// VA `va` (guest memory is mapped at the CPU address).
struct SourcePiece {
	uint32_t dw  = 0;
	uint32_t len = 0;
	uint64_t va  = 0;
};
// `pieces`, when given, gets where each stretch of `out` came from; `marks`, where in `out` each buffer the
// top level calls or chains to begins and ends (the title's own segments of the command buffer).
bool FlattenIndirectBuffers(std::span<const uint32_t> in, std::vector<uint32_t>* out, std::string* why = nullptr,
                            std::vector<SourcePiece>* pieces = nullptr, std::vector<uint32_t>* marks = nullptr);

// What the backend knows about labels beyond the buffer being translated.
class LabelOracle {
public:
	enum class Plan {
		Native, // a GPU write ahead in the ring satisfies the wait: pass it through
		Met,    // the label already holds a satisfying value: a barrier, as without native labels
		Split,  // nobody known satisfies it yet: cut the IB here
	};
	virtual ~LabelOracle() = default;
	// True when the GPU may write `addr` (GPU-mapped, writable guest memory).
	virtual bool GpuWritable(uint64_t addr) = 0;
	// The dword currently at `addr`, when it can be read without faulting. False leaves the tracker's
	// mirror of a GPU label store unconditional.
	virtual bool SampleDword(uint64_t addr, uint32_t* out) {
		(void)addr;
		(void)out;
		return false;
	}
	// Decides a wait no earlier write of the same buffer decides.
	virtual Plan PlanWait(const WaitCond& wait) = 0;
};

// Texture LOD statistics (SetLodBankScan): the PS5 T#s that count into GET_LOD_STATS, seen by the
// draws and dispatches since the previous GET_LOD_STATS packet. Gfx10 layout, as Kyty decodes it:
// word5 bit 25 MIP_STATS_CNT_EN, word5[19:8] MIN_LOD_WARN, word6[7:0] the counter bank (one 8-byte
// record per bank behind the packet's 64-byte header). The gfx9 layout (word6: MIN_LOD_WARN[11:0],
// COUNTER_BANK_ID[19:12], LOD_HDW_CNT_EN[20]) is counted next to it to tell which one the game uses.
struct LodBankScan {
	uint64_t                             addr = 0; // the GET_LOD_STATS buffer this scan ends at
	std::bitset<256>                     banks;     // banks of the counting T#s
	uint32_t                             tsharps  = 0; // plausible T#s scanned (repeats included)
	uint32_t                             counting = 0; // of those, with MIP_STATS_CNT_EN
	uint32_t                             inline_ud = 0; // counting T#s found in user data
	uint32_t                             depth1    = 0; // ... in a table a user-data pointer points at
	uint32_t                             depth2    = 0; // ... in a table such a table points at
	uint32_t                             tables    = 0; // tables read
	uint32_t                             max_word  = 0; // furthest dword a counting T# sat at in a table
	uint32_t                             alt_counting = 0; // gfx9 layout: word6 bit 20 set
	std::bitset<256>                     alt_banks;        // gfx9 layout: word6[19:12] of those
	std::map<uint32_t, uint32_t>         min_lod_warn;     // MIN_LOD_WARN of the counting T#s -> count
	std::vector<std::array<uint32_t, 8>> samples;          // the first counting T#s, whole
	void                                 Merge(const LodBankScan& o);
};

// What CpOpGetLodStats writes. Usage (the default, and KYTY_LOD_STATS_MODE=usage) is one record per
// bank from the translator's window. all / fill is the old whole-buffer fill (KYTY_LOD_STATS_FILL,
// 0xff unless overridden). banks / even / lo / hi are the earlier diagnostics.
enum class LodStatsMode { Usage, All, Banks, Even, Lo, Hi };
LodStatsMode ParseLodStatsMode(const char* mode);

// GET_LOD_STATS record, as ASTRO BOT's reader (eboot 0x900479d40) decodes it. Bits 23:0 are the
// sample count, bits 31:24 an id byte the streamer folds in (left 0), bits 59:56 the minimum mip
// actually sampled (0 finest; 0xF means this bank has no data). A used bank is reported as sampled
// at the finest mip; the exact mip comes later, from shader instrumentation.
constexpr uint64_t kLodStatsUsed   = 1ull;
constexpr uint64_t kLodStatsNoData = 0x0F00000000000000ull;
constexpr uint32_t kLodStatsHeader = 64;

// Writes a usage report: header dword 0 = 1 (after the records, so a reader that sees the label
// sees the records too), record `kLodStatsUsed` for each set bank, `kLodStatsNoData` for the rest.
// Returns how many used records were written. `size_bytes` is the packet's buffer size.
uint32_t WriteLodUsageReport(void* dst, uint32_t size_bytes, const std::bitset<256>& banks);

// "12-13,87" or "none". `dst` is NUL-terminated when `dst_size` > 0.
void FormatLodBankRanges(const std::bitset<256>& banks, char* dst, uint32_t dst_size);

// Banks published by the pipeline thread that translates DCBs and ACBs, taken by the command
// processor when it runs the matching GET_LOD_STATS. The processor runs after that translation
// (same thread when the pipeline is off; the completion thread when KYTY_NATIVE_GPU_THREAD is on),
// so Take sees every buffer submitted before the packet.
class LodStatsHandoff {
public:
	void Push(uint64_t addr, const std::bitset<256>& banks);
	bool Take(uint64_t addr, std::bitset<256>& banks);

private:
	static constexpr size_t kCap = 1024;
	std::mutex m_mutex;
	std::deque<std::pair<uint64_t, std::bitset<256>>> m_queue;
	uint32_t m_dropped = 0;
};
LodStatsHandoff& LodStatsReports();

// Optional stand-in for the guest-memory read the bank scan uses (process_vm_readv on Linux).
using LodGuestReadFn = size_t (*)(uint64_t addr, uint32_t* dst, size_t dwords, void* user);

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
	std::vector<DrawRecord>       draw_log;       // SetDrawLog: every draw and dispatch
	std::vector<FindHit>          find_hits;      // references into the SetFindRange range
	std::vector<LodBankScan>      lod_scans;      // SetLodBankScan: one per GET_LOD_STATS packet
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
	std::vector<std::pair<uint64_t, uint64_t>> label_writes; // (address, value) of this buffer's labels
	std::vector<std::pair<uint64_t, uint32_t>> data_writes;  // (address, dwords) of WRITE_DATA with more than one dword
	std::vector<std::pair<uint32_t, uint32_t>> gds_dma;      // (GDS offset | 0x80000000 if written, bytes)
	// Native labels only: the IB's segments in order (empty without native labels), and the packets
	// Kyty's command processor still has to run once the buffer is done (flips, waits for a flip),
	// copied out of the guest buffer.
	std::vector<Pm4Segment> segments;
	std::vector<uint32_t>   cpu_tail;
	// CP shadowing: translated with shadow preambles. With SetNoPreamble the first segment has none, and this
	// is the one it would have had (the state where the buffer starts): the backend puts it in front when the
	// segment does not carry on from the piece before it in the same job.
	bool                    cp_shadow = false;
	std::vector<uint32_t>   cp_shadow_resume;
	// SetAudit: the guest's sync packets field by field, and what refused the buffer (each opcode and reason once, at
	// most 8), up to where the walk stopped.
	Pm4SyncAudit            sync;
	std::vector<Pm4Refusal> refusals;
};

// Translates guest DCBs into self-contained IBs, keeping a CPU shadow of every register the guest
// sets so each IB can restore the full state in a preamble (the gfx ring is shared between
// submissions, so nothing on the hardware is assumed to survive from the previous DCB).
// Command processor draw state outside the register file. It lives on until a packet changes it, so on
// PS5 it carries from one DCB of the title to the next; the values are what libSceAgc's context start
// buffer sets once.
struct Pm4CpDrawState {
	uint32_t num_instances = 1;           // NUM_INSTANCES
	uint64_t index_base    = 0;           // INDEX_BASE
	uint32_t index_size    = 0xffffffffu; // INDEX_BUFFER_SIZE
	bool     index_set     = false;       // the title has sent INDEX_BASE or INDEX_BUFFER_SIZE
	uint64_t draw_base     = 0;           // SET_BASE, BASE_INDEX 1 (indirect draw/dispatch arguments)
};

class Pm4Translator {
public:
	Pm4Translator()  = default;
	~Pm4Translator() = default;

	Pm4Translator& operator=(const Pm4Translator&) = delete;
	// A copy with the same shadow and settings, to translate a buffer again without touching this one
	// (KYTY_NATIVE_GPU_TRANSLATE_BENCH).
	[[nodiscard]] std::unique_ptr<Pm4Translator> Clone() const { return std::unique_ptr<Pm4Translator>(new Pm4Translator(*this)); }

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
	// libSceAgc's context start buffer: loads every register of its tables (SET_*_REG_INDIRECT), whatever
	// was set before, and makes the context values what a context clear restores. UCONFIG registers
	// outside UconfigAllowed are left out; returns how many.
	uint32_t LoadStartState(std::span<const std::pair<uint32_t, uint32_t>> context,
	                        std::span<const std::pair<uint32_t, uint32_t>> sh,
	                        std::span<const std::pair<uint32_t, uint32_t>> uconfig);

	// Writes the shadowed registers as "cx|sh|uc <offset> <value>" lines (dword offsets within each
	// register space), for comparing against the state Kyty sees.
	void DumpShadow(std::FILE* f) const;

	// Depth-unit experiments (KYTY_NATIVE_GPU_DEPTH): HizOff forces hierarchical Z/stencil off in
	// DB_RENDER_OVERRIDE; NoHtile drops TILE_SURFACE_ENABLE so depth runs without HTILE at all.
	enum class DepthDebugMode { None, HizOff, NoHtile };
	void SetDepthDebugMode(DepthDebugMode mode) { m_depth_mode = mode; }

	// Experiment: render targets without DCC compression (CB_COLORn_INFO.DCC_ENABLE cleared).
	void SetNoDcc(bool no_dcc) { m_no_dcc = no_dcc; }
	// COMPUTE_STATIC_THREAD_MGMT_SE0..3 are ANDed with `masks` (0 for a word: left alone). KFD queues
	// with KYTY_NATIVE_GPU_KFD_CU_CLAMP: the kernel sizes a queue's wave save area by the CUs it found
	// at init (24 on the BC-250), and waves on the CUs enabled after it would overflow it.
	void SetComputeCuClamp(const std::array<uint32_t, 4>& masks) { m_cu_clamp = masks; }
	[[nodiscard]] uint32_t PatchSh(uint32_t offset, uint32_t value) const {
		static constexpr uint32_t kSe[4] = {0x216, 0x217, 0x219, 0x21a};
		for (uint32_t i = 0; i < 4; i++) {
			if (offset == kSe[i] && m_cu_clamp[i] != 0) {
				return value & m_cu_clamp[i];
			}
		}
		return value;
	}
	// The registered video-out buffers: color targets there are drawn without DCC (DCC_ENABLE cleared
	// before each draw, the guest's value back once the slot points elsewhere). The PS5 display engine
	// scans DCC out; Kyty's presenter copies the buffer as plain memory, so it has to hold plain pixels.
	// False when the range was already known.
	bool AddDisplayRange(uint64_t addr, uint64_t size) {
		for (const auto& [a, s]: m_display_ranges) {
			if (a == addr && s == size) {
				return false;
			}
		}
		m_display_ranges.emplace_back(addr, size);
		return true;
	}
	// Context register overrides the PS5 kernel applies through the CP's RCIU CAM (see PatchContext).
	void SetPs5RegOverrides(bool on) { m_ps5_reg_overrides = on; }

	// Experiment: targets without DCC lose CB_COLORn_INFO.FAST_CLEAR (no CMASK fast clear).
	// CMASK fast clear off on targets without DCC in the MRT slots of `slots` (bit n = slot n).
	void SetNoCmaskClear(bool on, uint32_t slots = 0xffu) {
		m_no_cmask_clear       = on;
		m_no_cmask_clear_slots = slots;
	}

	// Experiment: CB_TARGET_MASK is ANDed with `mask`, e.g. 0xf keeps only MRT0 writable.
	void SetTargetMask(uint32_t mask) { m_target_mask = mask; }

	// Guest WAIT_REG_MEMs become a full GPU barrier on a fence of ours at `va` (4 bytes in a buffer
	// that every submission maps); 0 drops them as before.
	void SetFenceAddress(uint64_t va) { m_fence_va = va; }
	[[nodiscard]] uint64_t FenceAddress() const { return m_fence_va; }
	// Exit guards: how a stop ends a title's GPU work where it stands, as the PS5 kernel stops a title's
	// queues, instead of running it on with made-up inputs. Each segment starts with a guard, and each
	// guest wait it gives the GPU is followed by one; a stop sets the words, then releases the waits.
	// On the gfx ring a guard is COND_EXEC on `abort_va` (0 while the title runs) over an INDIRECT_BUFFER
	// that chains to `stub_va`, 8 filler NOPs: the IB ends there and the CP returns to the ring. The PFP
	// decides a COND_EXEC before an ME wait ahead of it is released (probe condexec), so an ME wait gets
	// PFP_SYNC_ME before its guard. A chain on MEC took the board down (probe exitguard), so on a compute
	// ring a guard is COND_EXEC on `run_va` (1 while the title runs) skipping to the next guard or the
	// segment's end, with one at least every few thousand dwords. 0: none.
	void SetExitGuard(uint64_t abort_va, uint64_t run_va, uint64_t stub_va) {
		m_exit_abort_va = abort_va;
		m_exit_run_va   = run_va;
		m_exit_stub_va  = stub_va;
	}

	// Native labels: label writes run on the GPU (or at tracker points that write `tracker_va`),
	// waits pass through, become barriers or cut the IB as `oracle` decides; Kyty's command processor
	// only gets Pm4TranslateResult::cpu_tail. nullptr turns it off. The fence (SetFenceAddress) is
	// needed for the barriers of held segments. The dword at `tracker_va` + 8 takes flush-only releases.
	void SetNativeLabels(LabelOracle* oracle, uint64_t tracker_va) {
		m_labels     = oracle;
		m_tracker_va = tracker_va;
	}
	// With native labels: false turns the waits an earlier GPU write satisfies into barriers too.
	void SetNativeWaits(bool on) { m_native_waits = on; }
	// Timestamps stay on the tracker (Kyty's host TSC scaled to 100 MHz) instead of the GPU clock.
	// KYTY_NATIVE_GPU_CPU_TIMESTAMPS=1. The two clocks have the same unit and different zeros.
	void SetCpuTimestamps(bool on) { m_cpu_timestamps = on; }
	// With native labels (on a compute queue: one that has a KFD queue of its own, always in our form): a
	// title's RELEASE_MEM interrupt with a context id
	// (not 0, within libSceAgc's 27 bits) is raised by the GPU where the release runs, as on the PS5:
	// the packet as written when it carries no data, else after the label (INT_SEL 2 into the scratch
	// dword), and the patched amdgpu logs its context id for Kyty (KfdEopLog) instead of the tracker
	// raising it at the end of the segment. Interrupts with context id 0 stay tracker points: amdgpu's
	// own fences raise theirs with 0. SetGuestInterruptsCtx0: those too (a patched amdgpu with
	// bc250_gfx_eop_events=2 logs an id-0 interrupt raised inside an IB, where amdgpu's fences are not).
	void SetGuestInterruptsOnGpu(bool on) { m_guest_irq = on; }
	// A title's interrupt raised by the tracker fires before the GPU work after its release (on by default; off: at the
	// next GPU wait or the segment's end, sharing that store).
	void SetInterruptsAtRelease(bool on) { m_irq_at_release = on; }
	void SetGuestInterruptsCtx0(bool on) { m_guest_irq_ctx0 = on; }
	// With native labels: a DCB's flip (AgcDcbSetFlip) is requested as soon as the GPU passes it, a
	// tracker point (woken by the EOP interrupt), as the PS5 kernel queues a title's flip from the
	// end-of-pipe interrupt; not in Kyty's continuation after the whole buffer.
	void SetFlipsAtEop(bool on) { m_eop_flips = on; }
	// The old split of label writes (KYTY_NATIVE_GPU_TRACKER_WRITES=1): the tracker also stores every
	// label the GPU writes once the GPU passes the segment's end (a mirror), and a WRITE_DATA by the
	// constant engine is written by the tracker alone. Off (the default), a label the GPU can reach is
	// written by the GPU only, in stream order as on PS5, and the tracker just wakes a guest sleeping
	// on it: a mirror lands later than the GPU store, after writes that follow it in the ring or on
	// another ring, and can put back a value those writes replaced.
	void SetTrackerWrites(bool on) { m_tracker_writes = on; }

	// The buffers come from a compute queue (ACBs), whose packets differ from the gfx ring's
	// (DISPATCH_INDIRECT carries its argument address); they are rewritten for the gfx ring.
	void SetComputeQueue(bool compute_queue) { m_compute_queue = compute_queue; }
	// Submit those buffers to an amdgpu compute ring (MEC) instead of rewriting them for the gfx
	// ring: the guest DISPATCH_INDIRECT is kept, and RELEASE_MEM / WAIT_REG_MEM carry the compute
	// shader bit. Graphics-only packets are refused.
	void SetComputeRing(bool compute_ring) { m_compute_ring = compute_ring; }

	// Direct execution (KYTY_NATIVE_GPU_DIRECT_RUNS): a stretch of at least `min_dw` dwords the
	// translator passes on as written runs from the guest's buffer as an IB2 (INDIRECT_BUFFER) instead
	// of being copied; `source` says where the next buffer's dwords sit in guest memory (empty: copy as
	// before). The gfx ring only; the CP fetches the IB2 through GL2, so the IB needs a cache invalidate
	// before it (amdgpu's AMDGPU_IB_FLAG_EMIT_MEM_SYNC).
	void SetDirectRuns(uint32_t min_dw) { m_direct_min_dw = min_dw; }
	void SetSource(std::vector<SourcePiece> source) { m_source = std::move(source); }

	// GPU timing (KYTY_NATIVE_GPU_FLIP_TRACE): each gfx buffer is bracketed by GPU clocks (the micro
	// engine at its start, end of pipe at its end) in `span_va`, each GPU wait by two micro-engine
	// clocks in `wait_va`; each is a ring of `pairs` 16-byte (begin, end) pairs, used in order.
	// GpuTimesNext says how many pairs of each ring were handed out so far. 0 turns it off.
	void SetGpuTimes(uint64_t span_va, uint64_t wait_va, uint32_t pairs) {
		m_time_span_va = span_va;
		m_time_wait_va = wait_va;
		m_time_pairs   = pairs;
		m_time_wait_info.assign(pairs, {});
	}
	[[nodiscard]] uint64_t GpuSpansNext() const { return m_time_spans; }
	[[nodiscard]] uint64_t GpuWaitsNext() const { return m_time_waits; }
	// What the `i`th timed GPU wait waited for: the label, its reference, the draws of its DCB before
	// it, and whether guest memory already held a satisfying value when the DCB was translated.
	struct GpuWaitInfo {
		uint64_t addr  = 0;
		uint64_t ref   = 0;
		uint32_t draws = 0;
		bool     met   = false;
	};
	[[nodiscard]] const GpuWaitInfo& GpuWait(uint64_t i) const { return m_time_wait_info[i % m_time_pairs]; }

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
	// Record every draw and dispatch in Pm4TranslateResult::draw_log.
	void SetDrawLog(bool on) { m_draw_log = on; }
	// Residency at map time (GemBacking::ResidencyAtMap): no per-draw references are collected.
	void SetCollectRefs(bool on) { m_collect_refs = on; }
	// Translate builds the first segment's preamble from the state where the buffer starts (the previous
	// buffer's end) instead of the committed one (where it ends): a chunk of a DCB carries on where the
	// previous chunk left the registers, and its first draws are drawn with that state.
	void SetPreambleFromStart(bool on) { m_preamble_from_start = on; }
	// No preamble at all: the buffer continues the previous one within the same IB (a streamed chunk).
	void SetNoPreamble(bool on) { m_no_preamble = on; }
	// Run GET_LOD_STATS on the GPU instead of leaving it out (the hardware texture LOD statistics).
	void SetLodStats(bool on) { m_lod_stats = on; }
	// Collect the LOD statistics banks of the T#s each draw and dispatch reaches. Graphics draws
	// scan every stage's user-data SGPRs (PS, VS, GS, ES, HS, LS) and USER_DATA_ADDR; dispatches
	// scan COMPUTE_USER_DATA_0..15. Descriptor tables those pointers name are read two levels deep
	// (`table_dwords` of each first-level table). A non-zero-size GET_LOD_STATS closes the window:
	// Pm4TranslateResult::lod_scans gets one scan, and TakeLodScan returns what was seen since the
	// previous report. A zero-size packet does not close it.
	void SetLodBankScan(bool on, uint32_t table_dwords = 1024) {
		m_lod_scan        = on;
		m_lod_scan_dwords = table_dwords != 0 ? table_dwords : 1024;
	}
	void SetLodGuestRead(LodGuestReadFn fn, void* user) {
		m_lod_read      = fn;
		m_lod_read_user = user;
	}
	LodBankScan TakeLodScan() {
		LodBankScan out = std::move(m_lod_cur);
		m_lod_cur       = {};
		m_lod_tables.clear();
		m_lod_stages.clear();
		return out;
	}
	// PFP_SYNC_ME before indirect draws and dispatches (on by default).
	void SetPfpSyncIndirect(bool on) { m_pfp_sync_indirect = on; }
	// A zero-byte DMA_DATA with CP_SYNC before a job ends after a DMA_DATA without CP_SYNC (on by default).
	void SetCpDmaIdle(bool on) { m_cp_dma_idle = on; }
	// Guest WAIT_REG_MEM / WAIT_REG_MEM_64 on the gfx ring as written: the BC-250's CP runs both, 64-bit
	// compares, PFP waits and any compare function (probe waitforms). Off: our form (EQUAL on ME, a 64-bit
	// wait as 32-bit ones, a PFP wait as ME and PFP_SYNC_ME, other functions held on the CPU). Off by
	// default for now: My First GT stuttered more with it.
	void SetWaitsVerbatim(bool on) { m_waits_verbatim = on; }
	// DMA_DATA to or from GDS with CP_SYNC, and a CS_PARTIAL_FLUSH before a copy out of GDS, so the
	// append counters are final and copied before an indirect dispatch reads them (experiment).
	void SetGdsDmaSync(bool on) { m_gds_dma_sync = on; }
	// Every guest ACQUIRE_MEM becomes a full-range writeback and invalidate of all cache levels
	// (GL2, GLM metadata, GLK, GLV, GL1, GLI), preceded on the gfx ring by CB/DB metadata flushes
	// (experiment: the PS5 ME's ACQUIRE_MEM handler is several times longer than AMD's).
	void SetAcquireFull(bool on) { m_acquire_full = on; }
	// WRITE_DATA to memory in the native IB as well (Kyty still performs it): 0 none (default), 1 the
	// ones carrying more than one dword (data, not a label), 2 all.
	void SetWriteDataNative(uint32_t mode) { m_write_data_native = mode; }
	// How much memory a user-data value that may be an image base (address >> 8) makes resident.
	void SetGuessSpan(uint64_t bytes) { m_guess_span = bytes != 0 ? bytes : 1; }
	// KYTY_NATIVE_GPU_AUDIT: the result also records the guest's sync packets (Pm4TranslateResult::sync), what refused
	// the buffer (refusals) and each segment's GPU waits (Pm4Segment::audit_waits). The IB is the same.
	void SetAudit(bool on) { m_audit = on; }

	// Diagnostics: report every draw/dispatch whose resources reach [addr, addr + size) in
	// Pm4TranslateResult::find_hits (user data, then tables two levels deep). 0 size: off.
	void SetFindRange(uint64_t addr, uint64_t size) {
		m_find_lo = addr;
		m_find_hi = addr + size;
	}

	// Draws with this pixel shader are reported in Pm4TranslateResult::inspected (0: none).
	void SetInspectedShaders(std::set<uint64_t> shaders) { m_inspect = std::move(shaders); }

	// UCONFIG registers the backend is allowed to write. Everything else is dropped, because one
	// experiment writing other UCONFIG registers froze the whole board (see the probe README).
	static bool UconfigAllowed(uint32_t normalized_offset);

	// CP register shadowing (KYTY_NATIVE_GPU_CP_SHADOW) into the shadow buffer at `va` (kCpShadow* layout). Every
	// preamble becomes a shadow preamble: shadowing and loading off, CLEAR_STATE, PFP_SYNC_ME, the enables on,
	// then LOAD_*_REG of all four classes from S over the copy's keys (a class with none LOADs one anchor
	// register, so no class runs with shadowing on and no base), then SETs. Record: the copy's replay as before.
	// Load: only what S cannot restore (VGT_PRIMITIVE_TYPE/INDEX_TYPE, RSRC3/RSRC4, STATIC_THREAD_MGMT, SH 0x280,
	// PA_SC_SHADER_CONTROL, offsets from 0x400) and the classes left out of `load_classes`, which are replayed;
	// a buffer's first segment then restores the state where the buffer starts, as a chunk's does. `rearm`: a
	// CONTEXT_CONTROL with the enables right after a title's CLEAR_STATE 0, in case the clear turns them off.
	// The keys, and every decision made on the CPU, stay the copy's. A compute ring ignores it (MEC has no
	// CONTEXT_CONTROL).
	void SetCpShadow(uint64_t va, CpShadowMode mode, uint32_t load_classes = kCpShadowClasses, bool rearm = false) {
		m_cp_shadow_va      = va;
		m_cp_shadow_mode    = va != 0 ? mode : CpShadowMode::Off;
		m_cp_shadow_classes = load_classes & kCpShadowClasses;
		m_cp_shadow_rearm   = rearm;
	}
	[[nodiscard]] CpShadowMode CpShadow() const { return m_compute_ring ? CpShadowMode::Off : m_cp_shadow_mode; }
	// libSceAgc's context push and pop (CLEAR_STATE CMD 1/2/3) on the GPU through S, in load mode with the context
	// class LOADed (KYTY_NATIVE_GPU_CP_SHADOW_PUSHPOP, on unless 0). The first push of a window copies S's context
	// region to the save area P (shadowing stays on: S, and every preamble inside the window, follow the inner
	// state); a push while one is pushed does nothing, as the PS5's first pushed state is the one a pop restores.
	// The pop copies P back, LOADs the context from S and SETs what S does not restore, with no CPU diff, so a
	// value a late table set before the push comes back. The pop LOADs every context register S has a value for
	// (the seed and every register written since), or all of 0x000-0x3FF with `whole_range`
	// (KYTY_NATIVE_GPU_CP_SHADOW_POP_RANGE=all): a register never written since the seed is 0 in S, not amdgpu's
	// clear state. Without it (and in record mode) the push and pop stay on the CPU.
	void SetCpShadowPushPop(bool gpu, bool whole_range = false) {
		m_cp_shadow_pushpop = gpu;
		m_cp_shadow_pop_all = whole_range;
	}
	[[nodiscard]] bool CpShadowGpuPushPop() const {
		return CpShadow() == CpShadowMode::Load && m_cp_shadow_pushpop && (m_cp_shadow_classes & kCpShadowContext) != 0;
	}
	// S as the CP would hold it for the copy: every key below 0x400 at its place, with the values the IB writes
	// (PatchContext/PatchSh applied); `save` (kCpShadowSaveBytes / 4 dwords, when given) the context a pop
	// restores (the pushed one, or the current one when none is). Seeds S and P once, before any shadowing job
	// runs; from then on every context register of the seed counts as having its value in S.
	void WriteCpShadowImage(std::span<uint32_t> s, std::span<uint32_t> save = {});
	// The copy's keys against S as the GPU left it (an idle gfx ring), by who last wrote each register: a
	// verbatim table (the GPU read it later than the translator: a late table), a SET form or an insert (a
	// write the CP did not shadow, or a buffer that never ran), nothing since the seed (a table the translator
	// read with other offsets). Then S outside the keys. Each difference is a line in `report` (when given).
	struct CpShadowCheck {
		uint32_t keys       = 0;
		uint32_t equal      = 0;
		uint32_t late       = 0;
		uint32_t unshadowed = 0;
		uint32_t unseen     = 0;
		uint32_t outside    = 0; // non-zero outside the keys, the preamble's own writes not counted
	};
	CpShadowCheck CompareCpShadow(std::span<const uint32_t> s, std::FILE* report) const;
	// Who last wrote a register the copy holds (CompareCpShadow; the copy's offsets below 0x400).
	// Gpu: the GPU-side FMASK fix (load mode) changed it on the GPU, from values the copy may not hold.
	enum class RegWriter : uint8_t { Seed = 0, Set, Table, Expanded, Insert, Gpu };

private:
	void     EmitPreamble(std::vector<uint32_t>& ib, Pm4TranslateStats& stats, const RegFile& cx, const RegFile& sh,
	                      const RegFile& uc, const Pm4CpDrawState& cp) const;
	void     EmitShadowPreamble(std::vector<uint32_t>& ib, Pm4TranslateStats& stats, const RegFile& cx, const RegFile& sh,
	                            const RegFile& uc, const Pm4CpDrawState& cp) const;
	// The preamble's tess rings, PIXEL_PIPE_STAT_CONTROL and CP draw state.
	void     EmitPreambleState(std::vector<uint32_t>& ib, const RegFile& uc, const Pm4CpDrawState& cp) const;
	// SETs of the copy: every key of the classes in `replay`, the keys S cannot restore of the others.
	void     EmitRegisterSets(std::vector<uint32_t>& ib, Pm4TranslateStats& stats, const RegFile& cx, const RegFile& sh,
	                          const RegFile& uc, uint32_t replay) const;
	// SetCpShadowPushPop: the push's copy of S's context region to P, and the pop back to `target` from `current`
	// over the context registers `valid` in S.
	void     EmitCpShadowPush(std::vector<uint32_t>& ib) const;
	// fix_fmask on the GPU for MRT slot `slot`: FMASK = BASE (and the _EXT halves) from the values the CP holds (S),
	// not the copy's; with `check` only when the CP's FMASK is not 0. A register S has no value for (`valid`) is
	// amdgpu's clear state on the GPU, 0: the zero word is copied for it.
	void     EmitCpShadowFmaskFix(std::vector<uint32_t>& ib, uint32_t slot, bool check, const std::bitset<RegFile::kDense>& valid) const;
	// The display-DCC fix on the GPU for MRT slot `slot` (its INFO the copy took from a verbatim table): the CP's
	// CB_COLORn_INFO from S into both of the slot's tables, DCC_ENABLE cleared in the first by an ATOMIC_MEM AND, which
	// then sets the register; EmitCpShadowDccRestore sets it from the second, the INFO the CP had.
	void     EmitCpShadowDccOff(std::vector<uint32_t>& ib, uint32_t slot) const;
	void     EmitCpShadowDccRestore(std::vector<uint32_t>& ib, uint32_t slot) const;
	void     EmitCpShadowPop(std::vector<uint32_t>& ib, const RegFile& target, const RegFile& current,
	                         const std::bitset<RegFile::kDense>& valid) const;
	uint32_t PatchContext(uint32_t offset, uint32_t value) const; // applies the depth experiment

	RegFile                      m_cx; // context registers (SET_CONTEXT_REG space)
	RegFile                      m_cx_defaults; // what a context clear restores (the seeded defaults)
	RegFile                      m_saved_cx;    // context pushed by an Agc context-state push
	bool                         m_cx_pushed = false;
	RegFile                      m_sh; // SH registers (SET_SH_REG space)
	RegFile                      m_uc; // whitelisted UCONFIG registers
	Pm4CpDrawState               m_cp; // CP draw state where the next DCB starts
	bool                         m_gds = false;
	DepthDebugMode               m_depth_mode = DepthDebugMode::None;
	bool                         m_no_dcc     = false;
	std::array<uint32_t, 4>      m_cu_clamp {};
	std::vector<std::pair<uint64_t, uint64_t>> m_display_ranges; // AddDisplayRange
	uint32_t                     m_display_dcc_off = 0;           // MRT slots drawn with DCC cleared
	uint32_t                     m_display_dcc_gpu = 0;           // MRT slots whose CP INFO lost DCC on the GPU (EmitCpShadowDccOff)
	uint32_t                     m_saved_display_dcc_gpu = 0;     // m_display_dcc_gpu at the GPU push
	std::array<uint32_t, 8>      m_display_dcc_info {};           // their guest CB_COLORn_INFO
	bool                         m_ps5_reg_overrides = false;
	uint32_t                     m_target_mask = 0xffffffffu;
	bool                         m_no_cmask_clear = false;
	uint32_t                     m_no_cmask_clear_slots = 0xffu;
	bool                         m_compute_queue  = false;
	bool                         m_compute_ring   = false;
	bool                         m_preamble_from_start = false; // SetPreambleFromStart
	bool                         m_no_preamble         = false; // SetNoPreamble
	Pm4Translator(const Pm4Translator&) = default; // Clone
	uint32_t                     m_direct_min_dw  = 0;
	uint64_t                     m_time_span_va   = 0;
	uint64_t                     m_time_wait_va   = 0;
	uint32_t                     m_time_pairs     = 0;
	uint64_t                     m_time_spans     = 0;
	uint64_t                     m_time_waits     = 0;
	std::vector<GpuWaitInfo>     m_time_wait_info;
	std::vector<SourcePiece>     m_source;
	uint64_t                     m_fence_va       = 0;
	uint64_t                     m_exit_abort_va  = 0; // SetExitGuard
	uint64_t                     m_exit_run_va    = 0;
	uint64_t                     m_exit_stub_va   = 0;
	LabelOracle*                 m_labels         = nullptr;
	uint64_t                     m_tracker_va     = 0;
	bool                         m_native_waits   = true;
	bool                         m_cpu_timestamps = false;
	bool                         m_tracker_writes = false;
	bool                         m_guest_irq      = false;
	bool                         m_irq_at_release = true;
	bool                         m_guest_irq_ctx0 = false;
	bool                         m_eop_flips      = false;
	uint32_t                     m_fence_seq      = 0;
	std::set<uint64_t>           m_skip_ps;
	std::set<uint32_t>           m_skip_ops;
	std::set<uint64_t>           m_inspect; // KYTY_NATIVE_GPU_INSPECT_PS: pixel or compute shaders
	bool                         m_compact_refs = false;
	uint64_t                     m_guess_span   = 32ull << 20u;
	bool                         m_lod_stats    = false;
	bool                         m_pfp_sync_indirect = false;
	bool                         m_cp_dma_idle       = true;
	bool                         m_waits_verbatim    = false;
	bool                         m_gds_dma_sync      = false;
	bool                         m_acquire_full      = false;
	uint32_t                     m_write_data_native = 0;
	bool                         m_collect_refs      = true;
	bool                         m_draw_log          = false;
	bool                         m_audit             = false; // SetAudit
	bool                         m_lod_scan          = false;
	uint32_t                     m_lod_scan_dwords   = 1024;
	LodGuestReadFn               m_lod_read          = nullptr;
	void*                        m_lod_read_user     = nullptr;
	LodBankScan                  m_lod_cur;          // SetLodBankScan: since the last GET_LOD_STATS
	std::unordered_set<uint64_t> m_lod_tables;       // tables already read for m_lod_cur
	std::unordered_set<uint64_t> m_lod_stages;       // user-data blocks already scanned for m_lod_cur (hashes)
	std::vector<uint64_t>        m_bucket_bits;      // compact refs: one bit per 16 MiB bucket below 1 TiB
	uint64_t                     m_find_lo    = 0;
	uint64_t                     m_find_hi    = 0;
	uint64_t                     m_cp_shadow_va      = 0; // SetCpShadow
	CpShadowMode                 m_cp_shadow_mode    = CpShadowMode::Off;
	uint32_t                     m_cp_shadow_classes = kCpShadowClasses;
	bool                         m_cp_shadow_rearm   = false;
	// RegWriter of each register of m_cx, m_sh and m_uc below 0x400 (committed with them).
	std::array<uint8_t, RegFile::kDense> m_writer_cx {};
	std::array<uint8_t, RegFile::kDense> m_writer_sh {};
	std::array<uint8_t, RegFile::kDense> m_writer_uc {};
	bool                                 m_cp_shadow_pushpop = false; // SetCpShadowPushPop
	bool                                 m_cp_shadow_pop_all = false;
	std::array<uint8_t, RegFile::kDense> m_saved_writer_cx {}; // m_writer_cx at the GPU push
	// Context registers S has a value for: the seed's, and every one written since (committed with the copy).
	std::bitset<RegFile::kDense>         m_cp_valid_cx;
	std::bitset<RegFile::kDense>         m_saved_valid_cx; // m_cp_valid_cx at the GPU push
};

} // namespace Libs::Graphics::NativeGpu

#endif // EMULATOR_SRC_GRAPHICS_NATIVE_GPU_NATIVEPM4_H_
