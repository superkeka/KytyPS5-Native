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
#include <span>
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
	uint32_t waits_native      = 0; // native labels: guest waits passed through (an earlier GPU write satisfies them)
	uint32_t waits_split       = 0; // native labels: guest waits the IB is cut at (no known writer yet)
	uint32_t dma_data          = 0; // IT_DMA_DATA memory/immediate -> memory passed through
	uint32_t waits_dropped     = 0; // WAIT_REG_MEM(_64) on memory, dropped in shadow mode
	uint32_t waits_unmet       = 0; // of those, condition not yet true when the DCB was translated
	uint32_t pfp_syncs_emitted = 0; // PFP_SYNC_ME emitted in place of dropped PFP waits
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
enum class RingWaitRelease { Hold, GpuWait, Barrier };

RingWaitRelease PlanRingWait(int cover, bool value_satisfies, bool gpu_write, bool same_ring, bool writer_submitted,
                             bool label_in_memory, bool native_wait, bool gfx_writer = false);

// Guest compute queues are 0x20 + n (7 pipes x 8 queues). `available_rings` is the
// AMDGPU_INFO_HW_IP_INFO bitmask. Returns one of its set bits; queues spread across the rings the
// kernel exposes. No bit set: 0.
uint32_t ComputeRingForQueue(uint32_t guest_queue, uint32_t available_rings);

// The tessellation rings the title gave libSceAgcDriver (sceAgcDriverSetTFRing and
// sceAgcDriverSetHsOffchipParam), which the PS5 driver programs. Here every gfx IB's preamble writes
// them: the registers are global, and other processes on this ring program their own in between.
// Without them the GE writes tess factors at GPU address 0, faults, and the gfx ring hangs.
void SetTessellationRings(uint64_t tf_ring_base, uint32_t tf_ring_bytes, uint32_t hs_offchip_param);

// The native GPU waits put into IBs most recently (EmitNativeWait), for draining the rings when the
// process is told to stop: a ring stuck on a WAIT_REG_MEM whose writer will never be submitted cannot be
// preempted by amdgpu, and with GPU recovery off that hangs the board.
std::vector<WaitCond> RecentGpuWaits();
// A value that satisfies `w`, given the label's `current` value (only the masked bits change).
uint64_t SatisfyingValue(const WaitCond& w, uint64_t current);

// A guest command buffer with its INDIRECT_BUFFER calls and chains inlined, as the CP would fetch it:
// a call runs the target and comes back, a chain (CHAIN bit) runs the target instead of the rest.
// REWIND packets already valid are dropped: they only make the CP refetch, and the translator copies
// the commands anyway. Titles write their command buffers in chunks linked this way. Returns false
// (`out` cleared) when there is nothing to inline, or something it cannot: a REWIND still waiting for
// the CPU, another IB form, or a chain deeper than 16 or longer than 4M dwords.
// `why`, when given, says what stopped it (empty when there was nothing to inline).
bool FlattenIndirectBuffers(std::span<const uint32_t> in, std::vector<uint32_t>* out, std::string* why = nullptr);

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

private:
	void     EmitPreamble(std::vector<uint32_t>& ib, Pm4TranslateStats& stats, const RegFile& cx, const RegFile& sh,
	                      const RegFile& uc, const Pm4CpDrawState& cp) const;
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
	std::array<uint32_t, 8>      m_display_dcc_info {};           // their guest CB_COLORn_INFO
	bool                         m_ps5_reg_overrides = false;
	uint32_t                     m_target_mask = 0xffffffffu;
	bool                         m_no_cmask_clear = false;
	uint32_t                     m_no_cmask_clear_slots = 0xffu;
	bool                         m_compute_queue  = false;
	bool                         m_compute_ring   = false;
	uint64_t                     m_fence_va       = 0;
	LabelOracle*                 m_labels         = nullptr;
	uint64_t                     m_tracker_va     = 0;
	bool                         m_native_waits   = true;
	bool                         m_cpu_timestamps = false;
	bool                         m_tracker_writes = false;
	uint32_t                     m_fence_seq      = 0;
	std::set<uint64_t>           m_skip_ps;
	std::set<uint32_t>           m_skip_ops;
	std::set<uint64_t>           m_inspect; // KYTY_NATIVE_GPU_INSPECT_PS: pixel or compute shaders
	bool                         m_compact_refs = false;
	uint64_t                     m_guess_span   = 32ull << 20u;
	bool                         m_lod_stats    = false;
	bool                         m_pfp_sync_indirect = true;
	bool                         m_gds_dma_sync      = false;
	bool                         m_acquire_full      = false;
	uint32_t                     m_write_data_native = 0;
	bool                         m_collect_refs      = true;
	bool                         m_draw_log          = false;
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
};

} // namespace Libs::Graphics::NativeGpu

#endif // EMULATOR_SRC_GRAPHICS_NATIVE_GPU_NATIVEPM4_H_
