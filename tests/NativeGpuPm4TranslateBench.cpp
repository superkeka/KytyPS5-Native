// Translator cost on a Demon's Souls-like DCB set (3.11, standing at the level start): one flip is 146k dwords in 22 streamed
// chunks, 6891 SET_SH_REG, 5571 DRAW_INDEX_INDIRECT, 2105 DISPATCH_DIRECT, 1853 / 2202 / 1489 context / SH / UCONFIG register
// tables, 1297 INDEX_BASE, 255 WAIT_REG_MEM, 213 RELEASE_MEM, translated as the backend does (native labels, CP shadowing in
// load mode with gated loads and push/pop on the GPU, waits as written, direct runs, compact references, exit guards).
//
// Build and run (Mac or Linux):
//   clang++ -std=c++20 -O2 -g -Wall -Wextra -Isrc src/graphics/native_gpu/nativePm4.cpp tests/NativeGpuPm4TranslateBench.cpp -o t
//   ./t [flips] [--digest] [--dump FILE] [--cx N] [--sh N] [--uc N] [--ud PERCENT] [--chunks N] [--variant norefs|nodirect|noshadow|nolabels|fullrefs]
//       [--acbs N] [--views N] [--churn K] [--seen N] [--cold MIB]
// What the board shows and a warm loop over one DCB does not (all off by default, the digest is the DCB's alone):
//   --acbs N   N ACBs a flip (Demon's Souls: 18, 26k dwords), translated between the chunks by a compute-ring translator
//              as the backend's ACB translator is set up, and timed apart;
//   --views N  MakeResident with GemBacking's N views in a std::map: when the views' generation changes it copies the
//              merged ranges again, forgets which 16 MiB buckets it promoted and asks for each again (PromoteGuestRange:
//              a lock and a map lookup); --churn K: the generation changes every K chunks (default 1);
//   --seen N   DetectShaders' set holds N shaders already, as after a few minutes of play;
//   --cold MIB before each chunk and ACB, MIB of other memory are written: the title writes its DCB and tables on
//              other cores, so on the board they are not in this core's caches.
// The digest covers everything Translate returns that the backend uses (IB, resume, references, shaders, segments, points,
// CPU tail): an optimization must leave it unchanged. Tables, labels and the DCB live in an arena at a fixed address so the
// digest does not depend on where the allocator puts them.

#include "graphics/native_gpu/nativePm4.h"

#include <sys/mman.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <span>
#include <string>
#include <unordered_set>
#include <vector>

using Libs::Graphics::NativeGpu::CpShadowMode;
using Libs::Graphics::NativeGpu::FlattenIndirectBuffers;
using Libs::Graphics::NativeGpu::LabelOracle;
using Libs::Graphics::NativeGpu::NativeIbKind;
using Libs::Graphics::NativeGpu::Pm4TranslateResult;
using Libs::Graphics::NativeGpu::Pm4Translator;
using Libs::Graphics::NativeGpu::SourcePiece;
using Libs::Graphics::NativeGpu::WaitCond;

namespace {

constexpr uint32_t IT_SET_BASE                 = 0x11;
constexpr uint32_t IT_DISPATCH_DIRECT          = 0x15;
constexpr uint32_t IT_DRAW_INDEX_INDIRECT      = 0x25;
constexpr uint32_t IT_INDEX_BASE               = 0x26;
constexpr uint32_t IT_WAIT_REG_MEM             = 0x3C;
constexpr uint32_t IT_RELEASE_MEM              = 0x49;
constexpr uint32_t IT_SET_SH_REG_INDIRECT      = 0x63;
constexpr uint32_t IT_SET_UCONFIG_REG_INDIRECT = 0x64;
constexpr uint32_t IT_SET_SH_REG               = 0x76;
constexpr uint32_t IT_SET_CONTEXT_REG_INDIRECT = 0x9F;

constexpr uint64_t kTestShadowVa = 0x400003F80000ull; // where the backend puts S (NativeSubmitter::CpShadowVa)
constexpr uint64_t kTrackerVa    = 0x400000fff410ull;
constexpr uint64_t kFenceVa      = 0x400000fff400ull;
constexpr uint64_t kScratchVa    = 0x400000ffe000ull;
// Guest memory: tables, labels and the DCB in the low 1 TiB window, as the title's are.
constexpr uint64_t kArenaVa    = 0x00E000000000ull;
constexpr size_t   kArenaBytes = 64ull << 20u;

uint32_t Pkt3(uint32_t op, uint32_t body_dw, bool compute = false) {
	return (3u << 30u) | (((body_dw - 1u) & 0x3fffu) << 16u) | ((op & 0xffu) << 8u) | (compute ? 2u : 0u);
}

struct Rng {
	uint64_t s = 0x9e3779b97f4a7c15ull;
	uint32_t Next() {
		s ^= s << 13u;
		s ^= s >> 7u;
		s ^= s << 17u;
		return static_cast<uint32_t>(s >> 11u);
	}
	uint32_t Below(uint32_t n) { return Next() % n; }
	bool     Chance(uint32_t percent) { return Below(100) < percent; }
};

// Bump allocator in the arena (8-byte aligned).
struct Arena {
	uint8_t* base = nullptr;
	size_t   used = 0;
	uint32_t* Dwords(size_t n) {
		used         = (used + 7u) & ~size_t {7};
		uint32_t* p  = reinterpret_cast<uint32_t*>(base + used);
		used += n * 4u;
		if (used > kArenaBytes) {
			std::fprintf(stderr, "bench: arena full\n");
			std::abort();
		}
		return p;
	}
};

class BenchOracle: public LabelOracle {
public:
	bool GpuWritable(uint64_t /*addr*/) override { return true; }
	Plan PlanWait(const WaitCond& /*wait*/) override { return Plan::Native; }
};

struct Mix {
	uint32_t draws      = 5571;
	uint32_t dispatches = 2105;
	uint32_t set_sh     = 6891;
	uint32_t cx_tables  = 1853;
	uint32_t sh_tables  = 2202;
	uint32_t uc_tables  = 1489;
	uint32_t index_base = 1297;
	uint32_t set_base   = 400;
	uint32_t waits      = 255;
	uint32_t releases   = 213;
	uint32_t pushes     = 18; // CLEAR_STATE CMD 1, each popped (CMD 2) a few draws later: the mix's 37 CLEAR_STATE
	uint32_t ud_change  = 20; // percent of user-data dwords a SET_SH_REG changes
	uint32_t cx_pairs   = 40; // pairs per table: not measured on the board (stats.loaded_*_regs would tell)
	uint32_t sh_pairs   = 12;
	uint32_t uc_pairs   = 3;
	uint32_t chunks     = 22;
	uint32_t acbs       = 0;  // ACBs a flip (--acbs)
};

// Guest-like addresses user data and tables point at: 256 buffers spread over ~40 16 MiB buckets.
uint64_t GuestBuffer(uint32_t i) { return 0x200000000ull + static_cast<uint64_t>(i % 256u) * 0x280000ull; }
uint64_t GuestShader(uint32_t i) { return 0x300000000ull + static_cast<uint64_t>(i % 300u) * 0x4000ull; }

struct Flip {
	std::span<uint32_t>                        dcb;
	std::vector<std::pair<uint32_t, uint32_t>> chunks; // [begin, end) at packet boundaries
	std::vector<std::span<uint32_t>>           acbs;
};

Flip Generate(Arena& arena, const Mix& m) {
	Rng rng;
	// Register tables: a pool of distinct ones (titles reuse their state objects), as offset/value pairs.
	auto make_tables = [&](uint32_t count, uint32_t pairs, auto&& fill) {
		std::vector<uint32_t*> pool;
		for (uint32_t t = 0; t < count; t++) {
			uint32_t* p = arena.Dwords(2u * pairs);
			fill(t, std::span<uint32_t>(p, 2u * pairs));
			pool.push_back(p);
		}
		return pool;
	};
	// Context: render-state tables, and every third one a render-target table (CB/DB bases and formats).
	static constexpr uint32_t kCxState[] = {0x000, 0x001, 0x003, 0x08F, 0x090, 0x091, 0x0A0, 0x0A1, 0x105, 0x106, 0x107, 0x108,
	                                        0x191, 0x192, 0x193, 0x194, 0x1B3, 0x1B4, 0x1B5, 0x1B6, 0x1E0, 0x1E1, 0x1E2, 0x1E3,
	                                        0x200, 0x201, 0x202, 0x203, 0x204, 0x205, 0x206, 0x207, 0x2A1, 0x2A2, 0x2D5, 0x2E5};
	static constexpr uint32_t kCxTarget[] = {0x318, 0x31B, 0x31C, 0x31D, 0x31F, 0x321, 0x325, 0x390, 0x3B0, 0x3B8,
	                                         0x327, 0x32A, 0x32B, 0x32C, 0x391, 0x3B1, 0x005, 0x007, 0x010, 0x011,
	                                         0x012, 0x013, 0x014, 0x015, 0x01A, 0x01B, 0x01E};
	const auto cx_pool = make_tables(400, m.cx_pairs, [&](uint32_t t, std::span<uint32_t> p) {
		const bool target = t % 3u == 0u;
		for (uint32_t i = 0; i < p.size() / 2u; i++) {
			uint32_t off   = target ? kCxTarget[i % std::size(kCxTarget)] : kCxState[(t + i * 7u) % std::size(kCxState)];
			uint32_t value = rng.Next();
			if (off == 0x2D5u) {
				value &= ~0x4u; // no tessellation
			} else if (off == 0x318u || off == 0x327u || (off >= 0x012u && off <= 0x015u) || off == 0x005u || off == 0x31Fu ||
			           off == 0x321u || off == 0x325u) {
				value = static_cast<uint32_t>(GuestBuffer(t * 3u + i) >> 8u); // a surface base (address >> 8)
			} else if (off == 0x390u || off == 0x391u || off == 0x01Au || off == 0x01Bu || off == 0x01Eu) {
				value = 0; // bits 47:40
			} else if (off == 0x31Cu || off == 0x32Bu) {
				value = (0xau << 2u) | ((t & 1u) << 28u); // a format, DCC on some
			} else if (off == 0x31Bu || off == 0x32Au) {
				value = 0; // CB_COLORn_VIEW: one slice
			} else if (off == 0x3B0u || off == 0x3B1u) {
				value = (1919u << 14u) | 1079u; // 1920x1080
			} else if (off == 0x007u) {
				value = (1919u << 16u) | 1079u;
			}
			p[2u * i]      = off;
			p[2u * i + 1u] = value;
		}
	});
	// SH: a graphics stage's program, resources and user-data pointers, or a compute shader's.
	const auto sh_pool = make_tables(600, m.sh_pairs, [&](uint32_t t, std::span<uint32_t> p) {
		const bool     cs   = t % 10u < 3u;
		const uint64_t pgm  = GuestShader(t);
		const uint32_t gfx[] = {0x008, 0x009, 0x00A, 0x00B, 0x088, 0x089, 0x08A, 0x08B, 0x00C, 0x00D, 0x00E, 0x00F, 0x08C, 0x08D};
		const uint32_t cmp[] = {0x20C, 0x20D, 0x212, 0x213, 0x207, 0x208, 0x209, 0x240, 0x241, 0x242, 0x243};
		for (uint32_t i = 0; i < p.size() / 2u; i++) {
			const uint32_t off = cs ? cmp[i % std::size(cmp)] : gfx[i % std::size(gfx)];
			uint32_t       value = rng.Next() & 0x0fffffffu;
			if (off == 0x008u || off == 0x088u || off == 0x20Cu) {
				value = static_cast<uint32_t>(pgm >> 8u);
			} else if (off == 0x009u || off == 0x089u || off == 0x20Du) {
				value = static_cast<uint32_t>(pgm >> 40u);
			} else if (off == 0x00Cu || off == 0x08Cu || off == 0x00Eu || off == 0x240u || off == 0x242u) {
				value = static_cast<uint32_t>(GuestBuffer(t + i));
			} else if (off == 0x00Du || off == 0x08Du || off == 0x00Fu || off == 0x241u || off == 0x243u) {
				value = static_cast<uint32_t>(GuestBuffer(t + i - 1u) >> 32u);
			}
			p[2u * i]      = off;
			p[2u * i + 1u] = value;
		}
	});
	// UCONFIG: AgcCreatePrimState's primitive state.
	const auto uc_pool = make_tables(50, m.uc_pairs, [&](uint32_t t, std::span<uint32_t> p) {
		const uint32_t regs[] = {0x242, 0x24B, 0x262, 0x248, 0x24A};
		for (uint32_t i = 0; i < p.size() / 2u; i++) {
			p[2u * i]      = regs[i % std::size(regs)];
			p[2u * i + 1u] = regs[i % std::size(regs)] == 0x242u ? 4u + (t & 1u) : 0u;
		}
	});
	auto* labels = reinterpret_cast<uint64_t*>(arena.Dwords(2u * 64u));
	for (uint32_t i = 0; i < 64; i++) {
		labels[i] = 1;
	}

	std::vector<uint32_t> dcb;
	dcb.reserve(160000);
	auto append = [&](uint32_t op, std::initializer_list<uint32_t> body) {
		dcb.push_back(Pkt3(op, static_cast<uint32_t>(body.size())));
		dcb.insert(dcb.end(), body.begin(), body.end());
	};
	auto table = [&](uint32_t op, const uint32_t* pairs, uint32_t num) {
		const auto a = reinterpret_cast<uint64_t>(pairs);
		append(op, {static_cast<uint32_t>(a), static_cast<uint32_t>(a >> 32u), 0x80000000u, num});
	};
	// The user data each SH register held last: titles rewrite much of it unchanged before the next draw.
	std::vector<uint32_t> last(0x400, 0);
	std::vector<uint32_t> constants(512); // the values that are no address: a title's constants repeat
	for (uint32_t& c: constants) {
		c = rng.Next();
	}
	auto set_sh = [&](bool cs) {
		static constexpr uint32_t kSizes[] = {2, 4, 4, 4, 8, 8, 8, 12, 12, 16};
		uint32_t                  n        = kSizes[rng.Below(std::size(kSizes))];
		uint32_t                  base     = 0;
		if (cs) {
			n    = std::min(n, 16u);
			base = 0x240u + (rng.Below(2) * 2u) % (16u - n + 1u);
		} else if (n == 2 && rng.Chance(50)) {
			base = rng.Chance(50) ? 0x008u : 0x088u; // a program address
		} else {
			base = (rng.Chance(60) ? 0x00Cu : 0x08Cu) + 2u * rng.Below((32u - n) / 2u + 1u);
		}
		dcb.push_back(Pkt3(IT_SET_SH_REG, n + 1u, cs));
		dcb.push_back(base);
		for (uint32_t i = 0; i < n; i++) {
			const uint32_t off = base + i;
			uint32_t       v   = last[off];
			if (off == 0x008u || off == 0x088u) {
				v = static_cast<uint32_t>(GuestShader(rng.Below(300)) >> 8u);
			} else if (off == 0x009u || off == 0x089u) {
				v = 0;
			} else if (v == 0 || rng.Chance(m.ud_change)) {
				// A V# / table pointer pair (low dword, then high 16 bits and stride) or a constant.
				if ((off & 1u) == 0u) {
					v = rng.Chance(70) ? static_cast<uint32_t>(GuestBuffer(rng.Below(256))) : constants[rng.Below(512)];
				} else {
					v = rng.Chance(70) ? 0x2u | (16u << 16u) : constants[rng.Below(512)];
				}
			}
			last[off] = v;
			dcb.push_back(v);
		}
	};

	// Spread every kind evenly over the draws and dispatches (exact counts, Bresenham-style).
	const uint32_t items = m.draws + m.dispatches;
	struct Acc {
		uint32_t total, done = 0;
		bool     Due(uint32_t i, uint32_t n) {
			const uint64_t want = (static_cast<uint64_t>(total) * (i + 1u)) / n;
			if (done < want) {
				done++;
				return true;
			}
			return false;
		}
	};
	Acc   a_disp {m.dispatches}, a_sh {m.set_sh}, a_cx {m.cx_tables}, a_sht {m.sh_tables}, a_uc {m.uc_tables}, a_ib {m.index_base},
	    a_base {m.set_base}, a_wait {m.waits}, a_rel {m.releases}, a_push {m.pushes};
	uint32_t pop_at = 0; // the draw a pushed context is popped after
	uint32_t label_seq = 2;
	uint32_t draw      = 0; // draws so far: the graphics-only kinds are spread over the draws
	for (uint32_t i = 0; i < items; i++) {
		const bool cs = a_disp.Due(i, items);
		if (a_rel.Due(i, items)) {
			const uint32_t l   = label_seq % 64u;
			const uint64_t dst = reinterpret_cast<uint64_t>(&labels[l]);
			append(IT_RELEASE_MEM, {0x28u | (5u << 8u), 1u << 29u, static_cast<uint32_t>(dst), static_cast<uint32_t>(dst >> 32u),
			                        label_seq, 0u, 0u});
			labels[l] = label_seq;
			label_seq++;
		}
		if (a_wait.Due(i, items)) {
			const uint32_t l   = (label_seq - 1u) % 64u;
			const uint64_t dst = reinterpret_cast<uint64_t>(&labels[l]);
			append(IT_WAIT_REG_MEM, {3u | 0x10u, static_cast<uint32_t>(dst), static_cast<uint32_t>(dst >> 32u),
			                         static_cast<uint32_t>(labels[l]), 0xffffffffu, 4u});
		}
		if (!cs && pop_at == 0 && a_push.Due(draw, m.draws)) {
			append(0x12u, {1u}); // CLEAR_STATE CMD 1: push the context
			pop_at = draw + 3u;
		}
		if (!cs && pop_at != 0 && draw == pop_at) {
			append(0x12u, {2u}); // CMD 2: pop it
			pop_at = 0;
		}
		if (!cs && a_cx.Due(draw, m.draws)) {
			table(IT_SET_CONTEXT_REG_INDIRECT, cx_pool[rng.Below(static_cast<uint32_t>(cx_pool.size()))], m.cx_pairs);
		}
		if (a_sht.Due(i, items)) {
			const uint32_t t = cs ? 10u * rng.Below(60) + rng.Below(3) : 10u * rng.Below(60) + 3u + rng.Below(7);
			table(IT_SET_SH_REG_INDIRECT, sh_pool[t % sh_pool.size()], m.sh_pairs);
		}
		if (!cs && a_uc.Due(draw, m.draws)) {
			table(IT_SET_UCONFIG_REG_INDIRECT, uc_pool[rng.Below(static_cast<uint32_t>(uc_pool.size()))], m.uc_pairs);
		}
		while (a_sh.Due(i, items)) {
			set_sh(cs);
		}
		if (!cs && a_ib.Due(draw, m.draws)) {
			const uint64_t ib = GuestBuffer(rng.Below(256)) + 0x100000ull;
			append(IT_INDEX_BASE, {static_cast<uint32_t>(ib), static_cast<uint32_t>(ib >> 32u)});
		}
		if (!cs && a_base.Due(draw, m.draws)) {
			const uint64_t args = GuestBuffer(7) + 0x1000ull * rng.Below(64);
			append(IT_SET_BASE, {1u, static_cast<uint32_t>(args), static_cast<uint32_t>(args >> 32u)});
		}
		if (cs) {
			append(IT_DISPATCH_DIRECT, {1u + rng.Below(64), 1u + rng.Below(8), 1u, 1u});
		} else {
			append(IT_DRAW_INDEX_INDIRECT, {0x20u * rng.Below(4096), 0u, 0u, 0u});
			draw++;
		}
	}
	uint32_t* at = arena.Dwords(dcb.size());
	std::memcpy(at, dcb.data(), dcb.size() * 4u);
	Flip f;
	f.dcb = std::span<uint32_t>(at, dcb.size());
	// TranslateChunks' cuts: at packet boundaries, about the same size each.
	const uint32_t want = static_cast<uint32_t>(dcb.size() / m.chunks);
	uint32_t       begin = 0;
	for (uint32_t dw = 0; dw < dcb.size();) {
		dw += ((dcb[dw] >> 16u) & 0x3fffu) + 2u;
		if (dw - begin >= want && dw < dcb.size()) {
			f.chunks.emplace_back(begin, dw);
			begin = dw;
		}
	}
	f.chunks.emplace_back(begin, static_cast<uint32_t>(dcb.size()));
	// ACBs (after the DCB in the arena and from their own generator, so the DCB stays as it was): compute work as
	// libSceAgc records it, about 1.5k dwords each: a wait for a label, then dispatches with their program table and
	// user data, a cache flush every few, and the release of a label at the end.
	Rng acb_rng;
	acb_rng.s ^= 0x5bd1e995ull;
	for (uint32_t a = 0; a < m.acbs; a++) {
		std::vector<uint32_t> acb;
		auto put = [&](uint32_t op, std::initializer_list<uint32_t> body) {
			acb.push_back(Pkt3(op, static_cast<uint32_t>(body.size()), true));
			acb.insert(acb.end(), body.begin(), body.end());
		};
		{
			const uint32_t l   = acb_rng.Below(64);
			const uint64_t dst = reinterpret_cast<uint64_t>(&labels[l]);
			put(IT_WAIT_REG_MEM, {3u | 0x10u, static_cast<uint32_t>(dst), static_cast<uint32_t>(dst >> 32u),
			                      static_cast<uint32_t>(labels[l]), 0xffffffffu, 4u});
		}
		for (uint32_t d = 0; acb.size() < 1400u; d++) {
			const uint32_t* t   = sh_pool[(10u * acb_rng.Below(60) + acb_rng.Below(3)) % sh_pool.size()];
			const auto      tva = reinterpret_cast<uint64_t>(t);
			put(IT_SET_SH_REG_INDIRECT, {static_cast<uint32_t>(tva), static_cast<uint32_t>(tva >> 32u), 0x80000000u, m.sh_pairs});
			const uint32_t n = 2u + 2u * acb_rng.Below(6);
			acb.push_back(Pkt3(IT_SET_SH_REG, n + 1u, true));
			acb.push_back(0x240u);
			for (uint32_t i = 0; i < n; i++) {
				acb.push_back((i & 1u) == 0u ? static_cast<uint32_t>(GuestBuffer(acb_rng.Below(256))) : 0x2u | (16u << 16u));
			}
			put(IT_DISPATCH_DIRECT, {1u + acb_rng.Below(64), 1u + acb_rng.Below(8), 1u, 1u});
			if (d % 6u == 5u) {
				put(0x58u /* ACQUIRE_MEM */, {0u, 0xffffffffu, 0xffffffu, 0u, 0u, 0xau, 0x0u});
			}
		}
		{
			const uint32_t l   = label_seq % 64u;
			const uint64_t dst = reinterpret_cast<uint64_t>(&labels[l]);
			put(IT_RELEASE_MEM, {0x28u | (5u << 8u), 1u << 29u, static_cast<uint32_t>(dst), static_cast<uint32_t>(dst >> 32u), label_seq,
			                     0u, 0u});
			label_seq++;
		}
		uint32_t* p = arena.Dwords(acb.size());
		std::memcpy(p, acb.data(), acb.size() * 4u);
		f.acbs.emplace_back(p, acb.size());
	}
	return f;
}

// --variant: what a part of the translation costs, by leaving it out (the output then differs, of course).
std::string g_variant;

void Configure(Pm4Translator& t, BenchOracle& o) {
	t.SeedPs5Defaults();
	t.SetNativeLabels(&o, kTrackerVa);
	t.SetFenceAddress(kFenceVa);
	t.SetExitGuard(kScratchVa + 0x100, kScratchVa + 0x108, kScratchVa + 0x110);
	t.SetDirectRuns(256);
	t.SetCompactRefs(true);
	t.SetCollectRefs(true);
	t.SetWaitsVerbatim(true);
	t.SetCpShadow(kTestShadowVa, CpShadowMode::Load);
	t.SetCpShadowPushPop(true);
	t.SetCpShadowWaitOff(true);
	t.SetCpShadowGatedLoads(true);
	if (g_variant == "norefs") {
		t.SetCollectRefs(false);
	} else if (g_variant == "nodirect") {
		t.SetDirectRuns(0);
	} else if (g_variant == "noshadow") {
		t.SetCpShadow(0, CpShadowMode::Off);
	} else if (g_variant == "nolabels") {
		t.SetNativeLabels(nullptr, 0);
	} else if (g_variant == "fullrefs") {
		t.SetCompactRefs(false);
	}
}

// As the backend sets up its ACB translator for compute rings (Shadow: SetComputeRing, SetComputeQueue, compact references).
void ConfigureAcb(Pm4Translator& t, BenchOracle& o) {
	t.SetNativeLabels(&o, kTrackerVa + 16u);
	t.SetFenceAddress(kFenceVa + 8u);
	t.SetExitGuard(kScratchVa + 0x100, kScratchVa + 0x108, kScratchVa + 0x110);
	t.SetComputeRing(true);
	t.SetComputeQueue(true);
	t.SetCompactRefs(true);
	t.SetCollectRefs(true);
	t.SetWaitsVerbatim(true);
}

struct Digest {
	uint64_t h = 0xcbf29ce484222325ull;
	void     Add(const void* p, size_t bytes) {
		const auto* b = static_cast<const uint8_t*>(p);
		for (size_t i = 0; i < bytes; i++) {
			h = (h ^ b[i]) * 0x100000001b3ull;
		}
	}
	template <typename T>
	void Add(const std::vector<T>& v) {
		const uint64_t n = v.size();
		Add(&n, sizeof(n));
		Add(v.data(), v.size() * sizeof(T));
	}
	void AddValue(uint64_t v) { Add(&v, sizeof(v)); }
};

void DigestResult(Digest& d, const Pm4TranslateResult& r, std::FILE* dump) {
	d.AddValue(r.ok ? 1u : 0u);
	d.Add(r.ib);
	d.Add(r.cp_shadow_resume);
	d.Add(r.ref_buckets);
	d.Add(r.shader_addrs);
	d.Add(r.cpu_tail);
	for (const auto& ref: r.refs) {
		d.AddValue(ref.addr);
		d.AddValue(ref.size);
		d.AddValue((ref.pointer ? 1u : 0u) | (ref.index ? 2u : 0u));
	}
	for (const auto& t: r.render_targets) {
		d.AddValue(t.addr);
		d.AddValue(t.size);
		d.AddValue(t.draws);
	}
	for (const auto& s: r.segments) {
		d.AddValue(s.begin);
		d.AddValue(s.end);
		d.AddValue((s.held ? 1u : 0u) | (s.cp_shadow_head ? 2u : 0u));
		d.AddValue(s.barrier_dw);
		d.AddValue(s.wait_dw);
		d.AddValue(s.wait_len);
		d.Add(s.guards);
		for (const auto& p: s.points) {
			d.AddValue(p.dw);
			d.AddValue(static_cast<uint64_t>(p.kind));
			d.AddValue(p.addr);
		}
		d.AddValue(s.writes.size());
		d.AddValue(s.gpu_waits.size());
	}
	d.AddValue(r.stats.verbatim_dw);
	d.AddValue(r.stats.direct_dw);
	d.AddValue(r.stats.direct_ib2);
	if (dump != nullptr) {
		const uint64_t n = r.ib.size();
		std::fwrite(&n, sizeof(n), 1, dump);
		std::fwrite(r.ib.data(), 4, r.ib.size(), dump);
	}
}

double NowMs() {
	return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

volatile uint64_t g_sink = 0;

// DetectShaders' set of scanned shaders as nativeShadow.cpp keeps it (Shadow::SeenShaders): open addressing.
struct SeenShaders {
	std::vector<uint64_t> slots = std::vector<uint64_t>(4096);
	size_t                count = 0;
	bool                  Insert(uint64_t va) {
        if (2 * (count + 1) > slots.size()) {
            std::vector<uint64_t> old = std::move(slots);
            slots.assign(old.size() * 2, 0);
            for (uint64_t v: old) {
                if (v != 0) {
                    Place(v);
                }
            }
        }
        if (!Place(va)) {
            return false;
        }
        count++;
        return true;
	}
	bool Place(uint64_t va) {
		const size_t mask = slots.size() - 1;
		for (size_t i = static_cast<size_t>((va >> 8u) * 0x9e3779b97f4a7c15ull >> 32u) & mask;; i = (i + 1) & mask) {
			if (slots[i] == va) {
				return false;
			}
			if (slots[i] == 0) {
				slots[i] = va;
				return true;
			}
		}
	}
};

// What Shadow::Submit does around Translate for each chunk (nativeShadow.cpp), as far as it can run here: the second
// FlattenIndirectBuffers (Submit inlines the already flat chunk again), the packet check of the IB and the resume,
// MakeResident's bucket set and DetectShaders' set.
struct SubmitSide {
	std::set<uint64_t>           resident_buckets;
	std::unordered_set<uint64_t> seen_shaders; // the set DetectShaders had before 3.11
	SeenShaders                  seen_flat;
	std::vector<uint64_t>        resident_bits = std::vector<uint64_t>((1ull << 16u) / 64u); // the same, a bit per bucket below 1 TiB
	double                       flatten_ms = 0, validate_ms = 0, shaders_ms = 0, shaders_flat_ms = 0, resident_ms = 0, resident_bits_ms = 0;
	// --views: GemBacking's views (vaddr -> size, backing offset) and what MakeResident keeps of them.
	std::map<uint64_t, std::pair<uint64_t, uint64_t>> views;
	std::vector<std::pair<uint64_t, uint64_t>>        view_ranges;
	std::vector<uint64_t>                             view_buckets = std::vector<uint64_t>((1ull << 20u) / 64u); // 128 KiB, as on the board
	std::mutex                                        view_lock;
	uint32_t                                          churn = 1, submits = 0, refreshes = 0;
	uint64_t                                          promote_calls = 0;
	double                                            views_ms = 0;
	std::vector<uint8_t>                              cold; // --cold
	double                                            cold_ms = 0;

	// Views of 64 KiB to 2 MiB tiling the guest memory the flip references (and more past it), inserted in a scattered
	// order between other allocations, as a long-lived map's nodes are.
	void MakeViews(uint32_t n) {
		Rng                                       rng;
		std::vector<std::pair<uint64_t, uint64_t>> spans;
		const uint64_t                            bases[] = {0x200000000ull, 0x300000000ull, kArenaVa, 0x400000000ull};
		uint64_t                                  at[4]   = {bases[0], bases[1], bases[2], bases[3]};
		for (uint32_t i = 0; i < n; i++) {
			const uint32_t w    = i % 4u;
			const uint64_t size = (1ull + rng.Below(32)) << 16u;
			spans.emplace_back(at[w], size);
			at[w] += size;
		}
		for (size_t i = spans.size(); i > 1; i--) {
			std::swap(spans[i - 1], spans[rng.Below(static_cast<uint32_t>(i))]);
		}
		std::vector<std::unique_ptr<uint64_t[]>> noise;
		uint64_t                                 offset = 0;
		for (const auto& [va, size]: spans) {
			views[va] = {size, offset};
			offset += size;
			noise.emplace_back(new uint64_t[1u + rng.Below(16)]);
		}
	}
	void Evict() {
		if (cold.empty()) {
			return;
		}
		const double t0 = NowMs();
		for (size_t i = 0; i < cold.size(); i += 64) {
			cold[i]++;
		}
		cold_ms += NowMs() - t0;
	}
	// MakeResident's part that depends on the views (nativeShadow.cpp), without the ioctls.
	void ResidentViews(const Pm4TranslateResult& r) {
		const double t0 = NowMs();
		if (submits++ % churn == 0) {
			std::lock_guard<std::mutex> lock(view_lock);
			std::vector<std::pair<uint64_t, uint64_t>> ranges;
			for (const auto& [va, v]: views) {
				if (!ranges.empty() && ranges.back().second == va) {
					ranges.back().second += v.first;
				} else {
					ranges.emplace_back(va, va + v.first);
				}
			}
			view_ranges = std::move(ranges);
			std::fill(view_buckets.begin(), view_buckets.end(), 0ull);
			refreshes++;
		}
		auto backed = [&](uint64_t addr) {
			auto it = std::upper_bound(view_ranges.begin(), view_ranges.end(), std::make_pair(addr, ~0ull));
			return it != view_ranges.begin() && addr < std::prev(it)->second;
		};
		auto promote = [&](uint64_t addr, uint64_t size) {
			for (uint64_t b = addr >> 24u; b <= (addr + size - 1) >> 24u && b < (1ull << 20u); b++) {
				uint64_t& w = view_buckets[b / 64u];
				if ((w >> (b % 64u) & 1u) != 0) {
					continue;
				}
				w |= 1ull << (b % 64u);
				// PromoteGuestRange: the backing's lock and the views overlapping the bucket.
				std::lock_guard<std::mutex> lock(view_lock);
				auto it = views.upper_bound(b << 24u);
				if (it != views.begin()) {
					--it;
				}
				for (; it != views.end() && it->first < (b + 1u) << 24u; ++it) {
					g_sink += it->second.second;
				}
				promote_calls++;
			}
		};
		for (uint64_t addr: r.shader_addrs) {
			if (backed(addr)) {
				promote(addr, 64u * 1024u);
			}
		}
		for (uint32_t b: r.ref_buckets) {
			promote(static_cast<uint64_t>(b) << 24u, 1ull << 24u);
		}
		for (const auto& ref: r.refs) {
			if (backed(ref.addr)) {
				promote(ref.addr, ref.size != 0 ? ref.size : 4096u);
			}
		}
		views_ms += NowMs() - t0;
	}

	void Before(std::span<const uint32_t> chunk) {
		const double          t0 = NowMs();
		std::vector<uint32_t> flat;
		std::string           why;
		std::vector<SourcePiece> pieces;
		(void)FlattenIndirectBuffers(chunk, &flat, &why, &pieces);
		flatten_ms += NowMs() - t0;
	}
	void After(const Pm4TranslateResult& r) {
		if (!views.empty()) {
			ResidentViews(r);
		}
		double t0 = NowMs();
		for (uint64_t va: r.shader_addrs) {
			(void)seen_shaders.insert(va);
		}
		shaders_ms += NowMs() - t0;
		t0 = NowMs();
		for (uint64_t va: r.shader_addrs) {
			(void)seen_flat.Insert(va);
		}
		shaders_flat_ms += NowMs() - t0;
		t0 = NowMs();
		auto promote = [&](uint64_t addr, uint64_t size) {
			for (uint64_t b = addr >> 24u; b <= (addr + size - 1) >> 24u; b++) {
				(void)resident_buckets.insert(b);
			}
		};
		for (uint64_t addr: r.shader_addrs) {
			promote(addr, 64u * 1024u);
		}
		for (uint32_t b: r.ref_buckets) {
			promote(static_cast<uint64_t>(b) << 24u, 1ull << 24u);
		}
		for (const auto& ref: r.refs) {
			promote(ref.addr, ref.size != 0 ? ref.size : 4096u);
		}
		resident_ms += NowMs() - t0;
		// MakeResident with a bitset instead of std::set (buckets outside the low 1 TiB would keep the set).
		t0 = NowMs();
		uint64_t fresh = 0;
		auto     promote_bits = [&](uint64_t addr, uint64_t size) {
			for (uint64_t b = addr >> 24u; b <= (addr + size - 1) >> 24u; b++) {
				if (b < (1ull << 16u)) {
					uint64_t& w = resident_bits[b / 64u];
					fresh += (w >> (b % 64u) & 1u) == 0 ? 1u : 0u;
					w |= 1ull << (b % 64u);
				} else {
					fresh += resident_buckets.insert(b).second ? 1u : 0u;
				}
			}
		};
		for (uint64_t addr: r.shader_addrs) {
			promote_bits(addr, 64u * 1024u);
		}
		for (uint32_t b: r.ref_buckets) {
			promote_bits(static_cast<uint64_t>(b) << 24u, 1ull << 24u);
		}
		for (const auto& ref: r.refs) {
			promote_bits(ref.addr, ref.size != 0 ? ref.size : 4096u);
		}
		g_sink += fresh;
		resident_bits_ms += NowMs() - t0;
		t0 = NowMs();
		uint32_t bad = 0;
		if (!ValidateNativeIb(r.ib, false, &bad, NativeIbKind::Gfx)) {
			std::fprintf(stderr, "bench: the packet check refuses the IB at dw %u (header 0x%08x)\n", bad, r.ib[bad]);
		}
		if (!r.cp_shadow_resume.empty()) {
			(void)ValidateNativeIb(r.cp_shadow_resume, false, &bad, NativeIbKind::Gfx);
		}
		validate_ms += NowMs() - t0;
	}
};

} // namespace

int main(int argc, char** argv) {
	Mix         m;
	int         flips  = 40;
	bool        digest = false;
	const char* dump   = nullptr;
	uint32_t    views = 0, churn = 1, seen = 0, cold_mib = 0;
	for (int i = 1; i < argc; i++) {
		const std::string a = argv[i];
		if (a == "--digest") {
			digest = true;
		} else if (a == "--dump" && i + 1 < argc) {
			dump = argv[++i];
		} else if (a == "--variant" && i + 1 < argc) {
			g_variant = argv[++i];
		} else if (a == "--chunks" && i + 1 < argc) {
			m.chunks = static_cast<uint32_t>(std::max(1, std::atoi(argv[++i])));
		} else if (a == "--ud" && i + 1 < argc) {
			m.ud_change = static_cast<uint32_t>(std::atoi(argv[++i]));
		} else if (a == "--cx" && i + 1 < argc) {
			m.cx_pairs = static_cast<uint32_t>(std::atoi(argv[++i]));
		} else if (a == "--sh" && i + 1 < argc) {
			m.sh_pairs = static_cast<uint32_t>(std::atoi(argv[++i]));
		} else if (a == "--uc" && i + 1 < argc) {
			m.uc_pairs = static_cast<uint32_t>(std::atoi(argv[++i]));
		} else if (a == "--acbs" && i + 1 < argc) {
			m.acbs = static_cast<uint32_t>(std::atoi(argv[++i]));
		} else if (a == "--views" && i + 1 < argc) {
			views = static_cast<uint32_t>(std::atoi(argv[++i]));
		} else if (a == "--churn" && i + 1 < argc) {
			churn = static_cast<uint32_t>(std::max(1, std::atoi(argv[++i])));
		} else if (a == "--seen" && i + 1 < argc) {
			seen = static_cast<uint32_t>(std::atoi(argv[++i]));
		} else if (a == "--cold" && i + 1 < argc) {
			cold_mib = static_cast<uint32_t>(std::atoi(argv[++i]));
		} else {
			flips = std::max(1, std::atoi(argv[i]));
		}
	}
	void* mem = mmap(reinterpret_cast<void*>(kArenaVa), kArenaBytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
	if (mem == MAP_FAILED) {
		std::perror("bench: mmap");
		return 1;
	}
	if (reinterpret_cast<uint64_t>(mem) != kArenaVa) {
		std::fprintf(stderr, "bench: the arena is at %p, not 0x%llx: the digest is not comparable between runs\n", mem,
		             static_cast<unsigned long long>(kArenaVa));
	}
	Arena arena {static_cast<uint8_t*>(mem)};
	const Flip f = Generate(arena, m);
	{
		std::array<uint32_t, 256> ops {};
		for (uint32_t dw = 0; dw < f.dcb.size(); dw += ((f.dcb[dw] >> 16u) & 0x3fffu) + 2u) {
			ops[(f.dcb[dw] >> 8u) & 0xffu]++;
		}
		std::printf("DCB set: %zu dwords in %zu chunks; SET_SH_REG %u, DRAW_INDEX_INDIRECT %u, DISPATCH_DIRECT %u, tables cx/sh/uc %u/%u/%u "
		            "(%u/%u/%u pairs), INDEX_BASE %u, SET_BASE %u, WAIT_REG_MEM %u, RELEASE_MEM %u, CLEAR_STATE %u\n",
		            f.dcb.size(), f.chunks.size(), ops[IT_SET_SH_REG], ops[IT_DRAW_INDEX_INDIRECT], ops[IT_DISPATCH_DIRECT],
		            ops[IT_SET_CONTEXT_REG_INDIRECT], ops[IT_SET_SH_REG_INDIRECT], ops[IT_SET_UCONFIG_REG_INDIRECT], m.cx_pairs,
		            m.sh_pairs, m.uc_pairs, ops[IT_INDEX_BASE], ops[IT_SET_BASE], ops[IT_WAIT_REG_MEM], ops[IT_RELEASE_MEM], ops[0x12]);
	}
	Libs::Graphics::NativeGpu::SetCpShadowValidation(kTestShadowVa);

	BenchOracle   o;
	Pm4Translator t;
	Configure(t, o);
	Pm4Translator acb_t;
	ConfigureAcb(acb_t, o);
	SubmitSide            side;
	side.churn = churn;
	if (views != 0) {
		side.MakeViews(views);
	}
	for (uint32_t i = 0; i < seen; i++) {
		side.seen_shaders.insert(0x500000000ull + 0x100ull * i); // shaders of other scenes
		side.seen_flat.Insert(0x500000000ull + 0x100ull * i);
	}
	side.cold.resize(static_cast<size_t>(cold_mib) << 20u);
	double                acb_ms = 0;
	uint32_t              acb_refused = 0;
	std::vector<double>   per_flip;
	Digest                d;
	std::FILE*            dump_file = dump != nullptr ? std::fopen(dump, "wb") : nullptr;
	uint64_t              ib_dw = 0, verbatim = 0, direct = 0, refs = 0, buckets = 0, shaders = 0, targets = 0;
	for (int k = 0; k < flips; k++) {
		double ms = 0;
		for (size_t c = 0; c < f.chunks.size(); c++) {
			const auto [cb, ce] = f.chunks[c];
			const auto chunk    = std::span<const uint32_t>(f.dcb.data() + cb, ce - cb);
			side.Evict();
			side.Before(chunk);
			t.SetPreambleFromStart(true);
			t.SetNoPreamble(c > 0);
			t.SetSource({SourcePiece {0u, ce - cb, reinterpret_cast<uint64_t>(chunk.data())}});
			const double             t0 = NowMs();
			const Pm4TranslateResult r  = t.Translate(chunk);
			ms += NowMs() - t0;
			t.SetPreambleFromStart(false);
			t.SetNoPreamble(false);
			if (!r.ok) {
				std::fprintf(stderr, "bench: chunk %zu refused: op 0x%02x at dw %u\n", c, r.unknown_op, r.unknown_dw);
				return 1;
			}
			side.After(r);
			// The ACBs the title submits meanwhile, spread over the chunks.
			for (size_t a = c * f.acbs.size() / f.chunks.size(); a < (c + 1) * f.acbs.size() / f.chunks.size(); a++) {
				side.Evict();
				const double             a0 = NowMs();
				const Pm4TranslateResult ar = acb_t.Translate(f.acbs[a]);
				acb_ms += NowMs() - a0;
				if (!ar.ok) {
					if (acb_refused++ == 0) {
						std::fprintf(stderr, "bench: ACB %zu refused: op 0x%02x at dw %u\n", a, ar.unknown_op, ar.unknown_dw);
					}
					continue;
				}
				side.After(ar);
			}
			if (k == 0) {
				DigestResult(d, r, dump_file);
				ib_dw += r.ib.size();
				verbatim += r.stats.verbatim_dw;
				direct += r.stats.direct_dw;
				refs += r.refs.size();
				buckets += r.ref_buckets.size();
				shaders += r.shader_addrs.size();
				targets += r.render_targets.size();
			}
		}
		per_flip.push_back(ms);
	}
	if (dump_file != nullptr) {
		std::fclose(dump_file);
	}
	std::vector<double> sorted = per_flip;
	std::sort(sorted.begin(), sorted.end());
	const double n = static_cast<double>(flips);
	std::printf("first flip: IB %llu dwords, verbatim %llu, from the title's buffer %llu; digest %016llx\n",
	            static_cast<unsigned long long>(ib_dw), static_cast<unsigned long long>(verbatim),
	            static_cast<unsigned long long>(direct), static_cast<unsigned long long>(d.h));
	std::printf("first flip: refs %llu, ref buckets %llu, shader addresses %llu, render targets %llu (sums over the chunks)\n",
	            static_cast<unsigned long long>(refs), static_cast<unsigned long long>(buckets), static_cast<unsigned long long>(shaders),
	            static_cast<unsigned long long>(targets));
	std::printf("Translate per flip: min %.3f ms, median %.3f ms (%d flips)\n", sorted.front(), sorted[sorted.size() / 2], flips);
	if (!f.acbs.empty()) {
		size_t acb_dw = 0;
		for (const auto& a: f.acbs) {
			acb_dw += a.size();
		}
		std::printf("ACBs per flip: %zu of %zu dwords in all, translated in %.3f ms (%u refused)\n", f.acbs.size(), acb_dw, acb_ms / n,
		            acb_refused);
	}
	if (!side.views.empty()) {
		std::printf("MakeResident with %zu views: %.3f ms per flip (the ranges copied again %u times, %llu buckets asked for again)\n",
		            side.views.size(), side.views_ms / n, side.refreshes, static_cast<unsigned long long>(side.promote_calls));
	}
	if (!side.cold.empty()) {
		std::printf("evicting %u MiB before each buffer took %.3f ms per flip (not in the times above)\n", cold_mib, side.cold_ms / n);
	}
	std::printf("Submit around it per flip: flatten again %.3f ms, packet check %.3f ms, DetectShaders' set %.3f ms (std::unordered_set "
	            "%.3f ms), MakeResident's std::set %.3f ms (as a bitset %.3f ms)\n",
	            side.flatten_ms / n, side.validate_ms / n, side.shaders_flat_ms / n, side.shaders_ms / n, side.resident_ms / n,
	            side.resident_bits_ms / n);
	{
		// What one Translate call costs whatever the buffer holds (state copies, setup, assembly): a chunk of one draw,
		// with the shadow as the flips left it.
		const uint32_t tiny[] = {Pkt3(IT_SET_SH_REG, 3), 0x00Cu, 0x1000u, 0x2u, Pkt3(IT_DRAW_INDEX_INDIRECT, 4), 0u, 0u, 0u, 0u};
		uint32_t*      at     = arena.Dwords(std::size(tiny));
		std::memcpy(at, tiny, sizeof(tiny));
		const auto   chunk = std::span<const uint32_t>(at, std::size(tiny));
		const double t0    = NowMs();
		for (int i = 0; i < 2000; i++) {
			t.SetPreambleFromStart(true);
			t.SetNoPreamble(true);
			t.SetSource({SourcePiece {0u, static_cast<uint32_t>(chunk.size()), reinterpret_cast<uint64_t>(chunk.data())}});
			(void)t.Translate(chunk);
		}
		std::printf("one Translate of a one-draw chunk: %.1f us\n", (NowMs() - t0) * 1e3 / 2000.0);
	}
	if (digest) {
		std::printf("%016llx\n", static_cast<unsigned long long>(d.h));
	}
	return 0;
}
