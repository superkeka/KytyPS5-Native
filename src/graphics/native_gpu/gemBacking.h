#ifndef EMULATOR_SRC_GRAPHICS_NATIVE_GPU_GEMBACKING_H_
#define EMULATOR_SRC_GRAPHICS_NATIVE_GPU_GEMBACKING_H_

#include <cstdint>

// amdgpu GEM backing for guest "physical" memory on the BC-250 native GPU backend. Only real
// Linux with the amdgpu uAPI headers has it; everywhere else this compiles to an empty header so
// the opt-in hook in src/kernel/memoryAddressSpace.inc keeps upstream behaviour unchanged.
#if defined(__linux__) && !defined(__APPLE__)
#define KYTY_GEM_BACKING_SUPPORTED 1
#else
#define KYTY_GEM_BACKING_SUPPORTED 0
#endif

#if KYTY_GEM_BACKING_SUPPORTED

#include <atomic>
#include <cerrno>
#include <cstdio>
#include <algorithm>
#include <array>
#include <optional>
#include <string>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <memory>
#include <condition_variable>
#include <map>
#include <mutex>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <sched.h>
#include <linux/falloc.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <drm/amdgpu_drm.h>
#include <drm/drm.h>
#include <linux/kfd_ioctl.h>

namespace Libs::Graphics::NativeGpu {

// Backs guest direct memory so every guest direct-memory mapping is also mapped into the GPU VM
// at the same virtual address, on /dev/dri/renderD128, while only the memory the GPU actually
// needs is held in GPU buffers.
//
// The CPU side is a memfd, like the upstream backing: the guest views and the contiguous alias all
// map it, its pages are allocated on first touch and the kernel may swap them. A game can map far
// more direct memory than the BC-250 can keep resident (ASTRO BOT maps ~13 GiB of its 13.5 GiB
// budget and touches ~7 GiB), and GPU buffers cannot be swapped while submissions use them.
//
// The GPU side works in BlockSize() blocks of the backing. A block becomes resident when the CPU
// has written to it, which the memfd shows as data (PromoteTouched, via SEEK_DATA), or when a
// command buffer about to run references it (PromoteGuestRange: render targets, index buffers,
// pointers in shader user data, as the PM4 translator reports them). Promotion creates a cached GTT
// GEM buffer for the block, write-protects the block's views and alias slot, copies its data into
// the buffer, maps them onto the buffer with their protection restored through a dma-buf export
// (mmap through the DRM file only maps a buffer whole and from its start; a dma-buf mmap takes any
// page offset; an always-valid buffer has no dma-buf, see MapBoPiece), punches the block out of the memfd and rebinds the GPU mappings. A guest write
// during the move faults; the fault hook (WaitIfPromoting, from Memory::HandleGpuFault) waits for the
// move and retries it, so no write is lost and no read sees a stale copy. Blocks are never demoted. Resident blocks are TTM mappings (VM_PFNMAP | VM_IO): get_user_pages does not
// work on them, which only matters for process_vm_readv (Kyty uses it for crash-report stacks).
//
// GPU mappings of blocks that are not resident point at one shared dummy buffer, so an access the
// translator did not foresee reads garbage or loses its write instead of faulting: on this board a
// GPU page fault hangs the ring and can take the whole board down, and so do PRT mappings.
// DummyWritten() reports when the GPU wrote there.
//
// Simpler designs that do not work here (tools/bc250-native-probe): userptr buffers over the memfd
// cost a page-table walk of their whole size on every submission (~31 ms per frame for 13.5 GiB);
// udmabuf imports of the memfd are not snooped by the GPU, so CPU writes still in the cache stay
// invisible to it; and keeping all mapped memory in GTT buffers runs a big game out of RAM.
//
// Guest memory the GPU can touch also lives outside the direct-memory backing: flexible, program
// (code), runtime and stack ranges are committed anonymous host pages at guest VA == host VA, not
// slices of the chunks. BindExternal registers such a range's own host pages as a userptr GEM buffer
// and maps it into the same GPU VM at its guest VA, so shader code and other non-direct guest memory
// is resident on the GPU; UnbindExternal removes it. These buffers join CreatedHandles() so a
// submission lists them resident; as userptr buffers they still cost a page-table walk per submit,
// in proportion to their size.
//
// This class performs no command submission.
// KYTY_NATIVE_GPU_KFD_QUEUES=1: asynchronous compute buffers run on KFD user-mode queues (see
// nativeKfd.h). A KFD queue reaches KFD memory and, in this VM, the always-valid blocks of map
// residency; without those (partial residency, KYTY_NATIVE_GPU_ALWAYS_VALID=0) the blocks and the
// external ranges become KFD buffers.
// On by default in primary mode when amdgpu is the BC-250 build with the KFD TLB fixes (the stock
// one hangs flushing the KFD VMIDs' TLBs through the KIQ under gfx load); =0 turns it off, =1 on.
inline bool KfdQueuesWanted() {
	static const bool on = [] {
		if (const char* v = std::getenv("KYTY_NATIVE_GPU_KFD_QUEUES"); v != nullptr && v[0] != '\0') {
			return v[0] == '1';
		}
		const char* primary = std::getenv("KYTY_NATIVE_GPU_PRIMARY");
		if (primary == nullptr || primary[0] == '\0' || primary[0] == '0') {
			return false;
		}
		int   flush = 0;
		FILE* f     = std::fopen("/sys/module/amdgpu/parameters/bc250_flush_by_runlist", "re");
		if (f != nullptr) {
			if (std::fscanf(f, "%d", &flush) != 1) {
				flush = 0;
			}
			std::fclose(f);
		}
		return flush != 0;
	}();
	return on;
}

class GemBacking {
public:
	// Residency granularity. Every resident block is listed in every submission (a listed buffer
	// costs the CS ioctl about 2 us, measured), so smaller blocks waste less memory but make longer
	// lists. ASTRO BOT in gameplay: 16 MiB is ~600 buffers, 1.5-2 ms per CS, the GPU idle 40% of the
	// time (38 FPS); 64 MiB is ~160 buffers, 48 FPS, 0.9 GiB more resident; 128 MiB no faster and
	// swapping. KYTY_NATIVE_GPU_BLOCK_MIB: a power of two from 2 to 1024, default 64; 16 to 512 divide
	// the 13.5 GiB backing evenly.
	static uint64_t BlockSize() {
		static const uint64_t size = [] {
			const char*    v   = std::getenv("KYTY_NATIVE_GPU_BLOCK_MIB");
			const uint64_t mib = v != nullptr ? std::strtoull(v, nullptr, 0) : 64;
			return (mib >= 2 && mib <= 1024 && (mib & (mib - 1)) == 0 ? mib : 64) * 1024 * 1024;
		}();
		return size;
	}

	// Residency as the PS5 kernel does it (the default): a guest mapping of direct memory with GPU
	// access (or an mprotect that adds it) makes its blocks resident at once, the way the PS5 writes
	// the GPU page table entries at map time, and command buffers carry no residency work (no
	// per-draw references, no CPU-write scan before each submission) while every such block got
	// resident (MapComplete). That is within a budget: the PS5's game memory is its own, here the
	// system and Steam share it (ASTRO BOT maps 12.5 GiB of 15). A block that would leave less than
	// KYTY_NATIVE_GPU_MAP_RESERVE_MIB (default 1536) of available RAM stays out, and from then on
	// submissions make blocks resident as they reference them. KYTY_NATIVE_GPU_RESIDENCY=partial:
	// always that way.
	static bool ResidencyAtMap() {
		static const bool on = [] {
			const char* v = std::getenv("KYTY_NATIVE_GPU_RESIDENCY");
			return v == nullptr || std::strcmp(v, "partial") != 0;
		}();
		return on;
	}

	// True while every block with GPU access was made resident at map time.
	[[nodiscard]] bool MapComplete() const { return ResidencyAtMap() && !m_map_overflow.load(std::memory_order_acquire); }

	// Map-time residency (see ResidencyAtMap) for guest addresses [vaddr, vaddr + size): their blocks
	// become resident while the budget allows. Returns how many blocks were promoted.
	uint32_t PromoteAtMap(uint64_t vaddr, uint64_t size) {
		std::lock_guard<std::mutex> lock(m_lock);
		uint32_t                    promoted = 0;
		ForEachViewOverlap(vaddr, size, [&](uint64_t, uint64_t, uint64_t backing, uint64_t len) {
			for (uint64_t index = backing / BlockSize(); index <= (backing + len - 1) / BlockSize(); index++) {
				Block& block = m_blocks[index];
				if (block.resident || block.code) {
					continue;
				}
				if (MapBudgetAllows(block.size) && PromoteBlockUnlocked(index)) {
					promoted++;
				} else if (!block.code && !m_map_overflow.exchange(true, std::memory_order_acq_rel)) {
					std::fprintf(stderr,
					             "GemBacking: map-time residency stops at %llu MiB (less than %llu MiB of RAM would be left); "
					             "submissions make the rest resident as they reference it\n",
					             static_cast<unsigned long long>(m_resident_bytes >> 20u),
					             static_cast<unsigned long long>(MapReserve() >> 20u));
				}
			}
		});
		return promoted;
	}

	// With map residency, blocks are VM_ALWAYS_VALID buffers, the way PS5 memory is always in the GPU
	// page tables: the kernel keeps them valid for every submission of this VM, so submissions list
	// none of them (a listed buffer costs the CS ioctl ~2 us; ASTRO BOT has ~800 blocks). Such a buffer
	// has no dma-buf (the kernel refuses the export) and its DRM mmap only maps it whole from its start,
	// so CPU views map it whole and mremap the piece they need into place (MapBoPiece).
	// KYTY_NATIVE_GPU_ALWAYS_VALID=0 keeps listed, exportable blocks.
	static bool AlwaysValidWanted() {
		static const bool on = [] {
			const char* v = std::getenv("KYTY_NATIVE_GPU_ALWAYS_VALID");
			const char* kfd = std::getenv("KYTY_NATIVE_GPU_KFD_MEMORY");
			return ResidencyAtMap() && (kfd == nullptr || kfd[0] != '1') && (v == nullptr || v[0] != '0');
		}();
		return on;
	}

	// Creates the backing for `size` bytes: opens the device, creates the memfd, its contiguous
	// alias and the dummy GPU buffer. Returns nullptr on failure. No block is resident yet.
	static GemBacking* Create(uint64_t size, const char* device = "/dev/dri/renderD128") {
		if (size == 0) {
			return nullptr;
		}
		auto* self     = new GemBacking();
		self->m_size   = size;
		self->m_drm_fd = ::open(device, O_RDWR | O_CLOEXEC);
		if (self->m_drm_fd < 0) {
			std::fprintf(stderr, "GemBacking: open %s failed: %s\n", device, std::strerror(errno));
			delete self;
			return nullptr;
		}
		self->m_mem_fd = ::memfd_create("KytyDirectMemoryGem", MFD_CLOEXEC);
		if (self->m_mem_fd < 0 || ::ftruncate(self->m_mem_fd, static_cast<off_t>(size)) != 0) {
			std::fprintf(stderr, "GemBacking: memfd setup failed: %s\n", std::strerror(errno));
			delete self;
			return nullptr;
		}
		void* alias = ::mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, self->m_mem_fd, 0);
		if (alias == MAP_FAILED) {
			std::fprintf(stderr, "GemBacking: alias of %llu MiB failed: %s\n",
			             static_cast<unsigned long long>(size / (1024 * 1024)), std::strerror(errno));
			delete self;
			return nullptr;
		}
		self->m_alias = static_cast<uint8_t*>(alias);
		self->InitKfd();
		const auto count = (size + BlockSize() - 1) / BlockSize();
		self->m_blocks.resize(count);
		for (uint64_t i = 0; i < count; i++) {
			const uint64_t start    = i * BlockSize();
			self->m_blocks[i].size = (size - start < BlockSize()) ? size - start : BlockSize();
		}
		void* dummy_cpu = MAP_FAILED;
		if (self->CreateBo(BlockSize(), &self->m_dummy)) {
			dummy_cpu = ::mmap(nullptr, BlockSize(), PROT_READ | PROT_WRITE, MAP_SHARED,
			                   self->m_dummy.dmabuf_fd, 0);
		}
		if (dummy_cpu == MAP_FAILED) {
			std::fprintf(stderr, "GemBacking: dummy buffer setup failed\n");
			delete self;
			return nullptr;
		}
		self->m_dummy_cpu = static_cast<uint8_t*>(dummy_cpu);
		return self;
	}

	~GemBacking() {
		if (m_mover.joinable()) {
			{
				std::lock_guard<std::mutex> lock(m_mover_lock);
				m_mover_stop = true;
			}
			m_mover_cv.notify_all();
			m_mover.join();
		}
		for (const auto& [vaddr, ext]: m_external) {
			CloseExternal(vaddr, ext);
		}
		if (m_alias != nullptr) {
			::munmap(m_alias, m_size);
		}
		for (auto& block: m_blocks) {
			if (block.resident) {
				CloseBo(block.bo);
			}
		}
		if (m_dummy_cpu != nullptr) {
			::munmap(m_dummy_cpu, BlockSize());
		}
		CloseBo(m_dummy);
		CloseBo(m_tlb_bo);
		if (m_kfd_fd >= 0) {
			::close(m_kfd_fd);
		}
		if (m_mem_fd >= 0) {
			::close(m_mem_fd);
		}
		if (m_drm_fd >= 0) {
			::close(m_drm_fd);
		}
	}

	GemBacking(const GemBacking&)            = delete;
	GemBacking& operator=(const GemBacking&) = delete;

	[[nodiscard]] uint8_t* AliasBase() const { return m_alias; }

	// For the access-violation handler: true when a block is being moved into a GPU buffer (its
	// views are write-protected meanwhile), after waiting until the move is done; the faulting write
	// should then simply be retried. False, at once, otherwise. Async-signal-safe.
	// For diagnostics: true while a block is being moved.
	static bool WaitIfPromotingWouldWait() { return s_promoting.load(std::memory_order_acquire) != 0; }

	static bool WaitIfPromoting() {
		if (s_promoting.load(std::memory_order_acquire) == 0) {
			return false;
		}
		while (s_promoting.load(std::memory_order_acquire) != 0) {
			::sched_yield();
		}
		return true;
	}
	[[nodiscard]] uint64_t Size() const { return m_size; }

	// True when [offset, offset + size) lies in the backing. The alias covers all of it at all times,
	// so nothing has to be created before a range is zeroed or mapped.
	bool EnsureRange(uint64_t offset, uint64_t size) const {
		return size == 0 || (offset < m_size && size <= m_size - offset);
	}

	// Maps the backing bytes [offset, offset + size) at guest address `vaddr` with protection
	// `prot` (PROT_* flags): the memfd for blocks that are not resident, the block's dma-buf for
	// resident ones. On success the region is mapped at exactly `vaddr` and recorded as a view, so a
	// later promotion of its blocks can move it.
	bool MapView(uint64_t vaddr, uint64_t size, uint64_t offset, int prot) {
		if (size == 0 || !EnsureRange(offset, size)) {
			return false;
		}
		std::lock_guard<std::mutex> lock(m_lock);
		bool ok = ForEachBlockPiece(offset, size, [&](uint64_t index, uint64_t inner, uint64_t done, uint64_t part) {
			const Block& block = m_blocks[index];
			void* const  dst   = reinterpret_cast<void*>(vaddr + done);
			const bool   mapped =
			    block.resident ? MapBoPiece(block.bo, block.size, inner, part, prot, dst)
			                   : ::mmap(dst, part, prot, MAP_SHARED | MAP_FIXED, m_mem_fd, static_cast<off_t>(offset + done)) == dst;
			if (!mapped) {
				std::fprintf(stderr, "GemBacking: view mmap at 0x%llx (+0x%llx, block %llu) failed: %s\n",
				             static_cast<unsigned long long>(vaddr + done),
				             static_cast<unsigned long long>(part), static_cast<unsigned long long>(index),
				             std::strerror(errno));
				return false;
			}
			return true;
		});
		if (ok) {
			EraseViewsUnlocked(vaddr, size);
			m_views[vaddr] = View {size, offset, false};
			m_views_generation++;
		}
		return ok;
	}

	// Maps backing bytes [offset, offset + size) into this device's GPU VM at `vaddr`, the same
	// address the CPU view uses, so guest pointers are valid on the GPU unchanged: resident blocks
	// map their buffer, the others the dummy. REPLACE drops whatever was mapped there before. All
	// guest memory is mapped readable, writable and executable (shaders live in guest memory).
	bool BindView(uint64_t vaddr, uint64_t size, uint64_t offset) {
		if (size == 0 || !EnsureRange(offset, size)) {
			return false;
		}
		std::lock_guard<std::mutex> lock(m_lock);
		const bool ok = ForEachBlockPiece(offset, size, [&](uint64_t index, uint64_t inner, uint64_t done, uint64_t part) {
			const uint32_t handle = m_blocks[index].resident ? m_blocks[index].bo.handle : m_dummy.handle;
			return VaReplace(handle, vaddr + done, inner, offset + done, part) == 0;
		});
		auto it = m_views.find(vaddr);
		if (it != m_views.end() && it->second.size == size && it->second.offset == offset) {
			it->second.gpu = true;
			m_views_generation++; // GpuWritableRange answers change
		} else {
			EraseViewsUnlocked(vaddr, size);
			m_views[vaddr] = View {size, offset, true};
			m_views_generation++;
		}
		return ok;
	}

	// The guest memory type of backing bytes [offset, offset + size), as direct memory is allocated.
	// The PS5 kernel maps the title's shared memory (type 12, SCE_KERNEL_MTYPE_C_SHARED: ASTRO BOT and My
	// First Gran Turismo keep there the labels the CPU reads and writes) with MTYPE C_RW_S, and its
	// GPU-only memory (11, SCE_KERNEL_MTYPE_C) with C_RW_US, amdgpu's NC. Views of a coherent type map
	// with CoherentMtype(), the others with the default (NC). Off by default: the GPU then reads a CPU
	// store without stale GL2 lines, but nothing needs that yet, and My First Gran Turismo dropped a few
	// more frames with its shared memory C_RW_S (ASTRO BOT: no cost).
	// KYTY_NATIVE_GPU_COHERENT_TYPES: comma-separated types (12 as the PS5), none by default.
	void SetMemoryType(uint64_t offset, uint64_t size, int type) {
		if (size == 0) {
			return;
		}
		std::lock_guard<std::mutex> lock(m_lock);
		const uint64_t end = offset + size;
		auto           it  = m_coherent.lower_bound(offset);
		if (it != m_coherent.begin() && std::prev(it)->second > offset) {
			it = std::prev(it);
		}
		while (it != m_coherent.end() && it->first < end) {
			const auto [s, e] = *it;
			it                = m_coherent.erase(it);
			if (s < offset) {
				m_coherent.emplace(s, offset);
			}
			if (e > end) {
				m_coherent.emplace(end, e);
			}
		}
		if (CoherentType(type)) {
			m_coherent.emplace(offset, end);
			std::fprintf(stderr, "GemBacking: direct memory 0x%llx +0x%llx (type %d) maps coherent (MTYPE %s)\n",
			             static_cast<unsigned long long>(offset), static_cast<unsigned long long>(size), type,
			             CoherentMtype() == AMDGPU_VM_MTYPE_RW ? "C_RW_S" : "CC");
		}
	}

	// Removes every GPU mapping in [vaddr, vaddr + size) and forgets the views there (the caller has
	// already unmapped the CPU side).
	bool UnbindView(uint64_t vaddr, uint64_t size) {
		if (size == 0) {
			return true;
		}
		std::lock_guard<std::mutex> lock(m_lock);
		EraseViewsUnlocked(vaddr, size);
		m_views_generation++;
		const bool ok = VaOp(AMDGPU_VA_OP_CLEAR, 0, vaddr, 0, size) == 0;
		MarkVaStale();
		MapPrtHolesUnlocked(vaddr, size);
		return ok;
	}

	// sceKernelSetPrtAperture (index 0-2): on PS5 the GPU reads zeros from, and drops writes to, the
	// pages of a PRT aperture that nothing is mapped at. Titles map sparse resources into one piece by
	// piece and clear them whole (My First Gran Turismo's track load), which here faulted and hung the
	// gfx ring. The holes get PRT mappings, which behave the same; guest mappings replace them
	// (REPLACE), and unbinding one gives the hole back.
	void SetPrtAperture(int index, uint64_t vaddr, uint64_t size) {
		if (index < 0 || static_cast<size_t>(index) >= m_prt.size()) {
			return;
		}
		std::lock_guard<std::mutex> lock(m_lock);
		const auto old = m_prt[static_cast<size_t>(index)];
		m_prt[static_cast<size_t>(index)] = {vaddr, size};
		if (old.second != 0) {
			ForEachHoleUnlocked(old.first, old.second,
			                    [&](uint64_t start, uint64_t len) { VaOp(AMDGPU_VA_OP_CLEAR, 0, start, 0, len); });
		}
		for (const auto& [start, len]: m_prt) {
			if (len != 0) {
				MapPrtHolesUnlocked(start, len);
			}
		}
	}

	// Makes every block the CPU has written to resident. Promoted blocks are punched out of the
	// memfd, so SEEK_DATA only ever finds blocks that are not resident yet. Returns how many blocks
	// were promoted.
	uint32_t PromoteTouched() {
		std::lock_guard<std::mutex> lock(m_lock);
		uint32_t promoted = 0;
		uint64_t pos      = 0;
		while (pos < m_size) {
			const off_t data = ::lseek(m_mem_fd, static_cast<off_t>(pos), SEEK_DATA);
			if (data < 0) {
				break;
			}
			const uint64_t index = static_cast<uint64_t>(data) / BlockSize();
			if (!m_blocks[index].resident && !m_blocks[index].code && PromoteBlockUnlocked(index)) {
				promoted++;
			}
			pos = (index + 1) * BlockSize();
		}
		return promoted;
	}

	// [vaddr, vaddr + size) as pieces of resident block buffers, in guest order, for consumers that
	// import the buffers' dma-bufs (zero-copy presentation). False unless every byte is resident.
	struct DmabufPiece {
		int      fd;        // the block buffer's dma-buf
		uint64_t bo_size;   // its size
		uint64_t bo_offset; // where the piece starts in it
		uint64_t len;
	};
	bool DmabufPieces(uint64_t vaddr, uint64_t size, std::vector<DmabufPiece>& out) const {
		std::lock_guard<std::mutex> lock(m_lock);
		out.clear();
		uint64_t covered = 0;
		uint64_t next    = vaddr;
		bool     ok      = true;
		ForEachViewOverlap(vaddr, size, [&](uint64_t, uint64_t guest, uint64_t backing, uint64_t len) {
			if (!ok || guest != next) {
				ok = false;
				return;
			}
			while (len > 0) {
				const uint64_t index = backing / BlockSize();
				const uint64_t inner = backing % BlockSize();
				const Block&   block = m_blocks[index];
				const uint64_t piece = std::min(len, block.size - inner);
				if (!block.resident || block.bo.dmabuf_fd < 0) {
					ok = false;
					return;
				}
				out.push_back(DmabufPiece {block.bo.dmabuf_fd, block.size, inner, piece});
				backing += piece;
				len -= piece;
				covered += piece;
				next += piece;
			}
		});
		return ok && covered == size;
	}

	// Makes the blocks behind guest addresses [vaddr, vaddr + size) resident, through whichever views
	// cover them; addresses outside every view are ignored. Returns how many blocks were promoted.
	uint32_t PromoteGuestRange(uint64_t vaddr, uint64_t size) {
		std::lock_guard<std::mutex> lock(m_lock);
		uint32_t promoted = 0;
		ForEachViewOverlap(vaddr, size, [&](uint64_t, uint64_t, uint64_t backing, uint64_t len) {
			for (uint64_t index = backing / BlockSize(); index <= (backing + len - 1) / BlockSize(); index++) {
				if (!m_blocks[index].resident && !m_blocks[index].code && PromoteBlockUnlocked(index)) {
					promoted++;
				}
			}
		});
		return promoted;
	}

	// The presenter reads display buffers through dma-bufs, which an always-valid buffer cannot export:
	// the always-valid blocks behind guest addresses [vaddr, vaddr + size) move to ordinary buffers
	// (data copied, views and GPU mappings moved over), listed in every submission again. `idle` runs
	// first, without the lock, if any block has to move: the old buffers are freed, so it must leave
	// the GPU idle. Returns how many blocks moved.
	template <typename Idle>
	uint32_t ExportGuestRange(uint64_t vaddr, uint64_t size, Idle&& idle) {
		std::vector<uint64_t> indices;
		{
			std::lock_guard<std::mutex> lock(m_lock);
			ForEachViewOverlap(vaddr, size, [&](uint64_t, uint64_t, uint64_t backing, uint64_t len) {
				for (uint64_t index = backing / BlockSize(); index <= (backing + len - 1) / BlockSize(); index++) {
					if (m_blocks[index].resident && m_blocks[index].bo.always_valid) {
						indices.push_back(index);
					}
				}
			});
		}
		if (indices.empty()) {
			return 0;
		}
		idle();
		std::lock_guard<std::mutex> lock(m_lock);
		uint32_t                    moved = 0;
		for (const uint64_t index: indices) {
			if (m_blocks[index].resident && m_blocks[index].bo.always_valid && PromoteBlockUnlocked(index, true)) {
				moved++;
			}
		}
		return moved;
	}

	// True when `vaddr` lies in a view of this backing.
	[[nodiscard]] bool IsBacked(uint64_t vaddr) const {
		std::lock_guard<std::mutex> lock(m_lock);
		return FindViewUnlocked(vaddr) != m_views.end();
	}

	// True when `vaddr` is mapped into the GPU VM at all: a view of the backing or an external range.
	[[nodiscard]] bool IsGpuMapped(uint64_t vaddr) const {
		std::lock_guard<std::mutex> lock(m_lock);
		if (FindViewUnlocked(vaddr) != m_views.end()) {
			return true;
		}
		auto it = m_external.upper_bound(vaddr);
		return it != m_external.begin() && vaddr < std::prev(it)->second.end;
	}

	// True when the GPU may write `vaddr`: a view mapped into the GPU VM (its block must be made
	// resident before the write lands, or it goes to the dummy) or a writable external range. CPU
	// stacks and read-only ranges are not.
	[[nodiscard]] bool IsGpuWritable(uint64_t vaddr) const {
		std::lock_guard<std::mutex> lock(m_lock);
		if (auto v = FindViewUnlocked(vaddr); v != m_views.end()) {
			return v->second.gpu;
		}
		auto it = m_external.upper_bound(vaddr);
		return it != m_external.begin() && vaddr < std::prev(it)->second.end && std::prev(it)->second.writable;
	}

	// IsGpuWritable, plus the range [*lo, *hi) holding `vaddr` that the answer holds for until
	// ViewsGeneration changes (lo == hi when `vaddr` is in no range).
	[[nodiscard]] bool GpuWritableRange(uint64_t vaddr, uint64_t* lo, uint64_t* hi) const {
		std::lock_guard<std::mutex> lock(m_lock);
		*lo = *hi = 0;
		if (auto v = FindViewUnlocked(vaddr); v != m_views.end()) {
			*lo = v->first;
			*hi = v->first + v->second.size;
			return v->second.gpu;
		}
		auto it = m_external.upper_bound(vaddr);
		if (it != m_external.begin() && vaddr < std::prev(it)->second.end) {
			*lo = std::prev(it)->first;
			*hi = std::prev(it)->second.end;
			return std::prev(it)->second.writable;
		}
		return false;
	}

	// Copies up to `size` bytes of guest memory at `vaddr` (clipped to the view holding it) through
	// the alias, which is always readable whatever the guest's own protection. Returns the bytes
	// copied, 0 when `vaddr` is not backed.
	uint64_t ReadGuest(uint64_t vaddr, void* dst, uint64_t size) const {
		std::lock_guard<std::mutex> lock(m_lock);
		auto it = FindViewUnlocked(vaddr);
		if (it == m_views.end()) {
			return 0;
		}
		const uint64_t skip = vaddr - it->first;
		const uint64_t n    = std::min(size, it->second.size - skip);
		std::memcpy(dst, m_alias + it->second.offset + skip, n);
		return n;
	}

	// Copies `size` bytes to guest memory at `vaddr` (clipped to the view holding it) through the
	// alias, which is always writable whatever the guest's own protection. Returns the bytes copied.
	uint64_t WriteGuest(uint64_t vaddr, const void* src, uint64_t size) {
		std::lock_guard<std::mutex> lock(m_lock);
		auto it = FindViewUnlocked(vaddr);
		if (it == m_views.end()) {
			return 0;
		}
		const uint64_t skip = vaddr - it->first;
		const uint64_t n    = std::min(size, it->second.size - skip);
		std::memcpy(m_alias + it->second.offset + skip, src, n);
		return n;
	}

	// True (once) when the GPU has written into the dummy buffer since the last call, i.e. a
	// submission wrote to guest memory that was not resident; the dummy is cleared again. Call it
	// only while the GPU is idle.
	bool DummyWritten() {
		std::lock_guard<std::mutex> lock(m_lock);
		static const uint8_t kZero[65536] = {};
		bool written = false;
		for (uint64_t at = 0; at < BlockSize() && !written; at += sizeof(kZero)) {
			written = std::memcmp(m_dummy_cpu + at, kZero, sizeof(kZero)) != 0;
		}
		if (written) {
			std::memset(m_dummy_cpu, 0, BlockSize());
		}
		return written;
	}

	// Bumped whenever the set of views changes; a caller caching ViewRanges() refreshes on a change.
	[[nodiscard]] uint64_t ViewsGeneration() const { return m_views_generation.load(std::memory_order_acquire); }

	// The guest address ranges [start, end) covered by views, sorted and merged.
	[[nodiscard]] std::vector<std::pair<uint64_t, uint64_t>> ViewRanges() const {
		std::lock_guard<std::mutex>                lock(m_lock);
		std::vector<std::pair<uint64_t, uint64_t>> ranges;
		for (const auto& [vaddr, view]: m_views) {
			if (!ranges.empty() && ranges.back().second == vaddr) {
				ranges.back().second += view.size;
			} else {
				ranges.emplace_back(vaddr, vaddr + view.size);
			}
		}
		return ranges;
	}

	// The external (userptr) buffers: how many and how many bytes. Each submission revalidates their
	// pages, so their cost grows with the size, unlike the resident blocks.
	[[nodiscard]] std::pair<uint64_t, uint64_t> ExternalBuffers() const {
		std::lock_guard<std::mutex> lock(m_lock);
		uint64_t                    bytes = 0;
		for (const auto& [vaddr, ext]: m_external) {
			bytes += ext.end - vaddr;
		}
		return {m_external.size(), bytes};
	}

	[[nodiscard]] uint64_t ResidentBytes() const {
		std::lock_guard<std::mutex> lock(m_lock);
		return m_resident_bytes;
	}
	// Of ResidentBytes, what the CPU had written before each block moved (the rest the GPU may still
	// use: render targets and other GPU-only memory start out untouched).
	[[nodiscard]] uint64_t TouchedBytes() const {
		std::lock_guard<std::mutex> lock(m_lock);
		return m_touched_bytes;
	}

	// Binds a guest range that is NOT part of the memfd backing — flexible, program (code), runtime
	// or stack memory: committed anonymous host pages at guest VA == host VA — into this device's GPU
	// VM at its own address, by registering the range's host pages as a userptr GEM buffer and
	// mapping it there (REPLACE, so it is safe over whatever was mapped before). The caller maps the
	// host pages before the GPU can reach the range, so they are resident. `gpu_write` must match the
	// guest range's GPU protection: a range the GPU only reads — shader code is mapped r-x on the host
	// — is registered read-only, because amdgpu's userptr validation faults a write onto a read-only
	// host page and fails the whole submission (EPERM). Any overlapping prior external binding is
	// dropped first, so a remap of the same guest address just works. The CPU and Vulkan paths are
	// unaffected if this fails, so failures only log.
	bool BindExternal(uint64_t vaddr, uint64_t size, bool gpu_write) {
		if (vaddr == 0 || size == 0) {
			return false;
		}
		std::lock_guard<std::mutex> lock(m_lock);
		ClearExternalRangeUnlocked(vaddr, size);
		// A single guest range can hold sub-ranges the guest reprotected to execute-only or no-access
		// (seen in program memory: the shader is a readable slice, with an execute-only tail). A
		// userptr cannot fault in an unreadable page, which fails the whole submission, so register one
		// buffer per readable slice and skip the holes.
		// Executable slices are CPU instructions; shaders live in data. Each registered byte costs every
		// submission a page walk (ASTRO BOT's program text is most of its 267 MiB of userptr), so
		// they are left out unless KYTY_NATIVE_GPU_BIND_CODE=1.
		static const bool bind_code = [] {
			const char* v = std::getenv("KYTY_NATIVE_GPU_BIND_CODE");
			return v != nullptr && v[0] == '1';
		}();
		bool any = false;
		ForEachReadableSubrange(vaddr, size, [&](uint64_t s, uint64_t sz, bool writable, bool executable) {
			if (executable && !bind_code) {
				return;
			}
			any = RegisterExternalUnlocked(s, sz, gpu_write && writable) || any;
		});
		return any;
	}

	// Removes the external GPU binding(s) in [vaddr, vaddr + size) and closes the buffers they wholly
	// freed. A VA with no external binding (direct or pooled memory, or an already-unmapped range) is
	// left untouched, so this is a safe no-op on the chunk-backed ranges that flow through the same
	// GPU-range unmap hook.
	bool UnbindExternal(uint64_t vaddr, uint64_t size) {
		if (size == 0) {
			return true;
		}
		std::lock_guard<std::mutex> lock(m_lock);
		ClearExternalRangeUnlocked(vaddr, size);
		MapPrtHolesUnlocked(vaddr, size);
		return true;
	}

	// The DRM file whose GPU VM holds the bindings; command submission must use the same file.
	[[nodiscard]] int DrmFd() const { return m_drm_fd; }
	// The KFD process of this VM: with KYTY_NATIVE_GPU_KFD_MEMORY its blocks are KFD buffers, and KFD
	// user-mode queues (KYTY_NATIVE_GPU_KFD_QUEUES) run in it. -1 / 0 when neither is on.
	[[nodiscard]] int      KfdFd() const { return m_kfd_fd; }

	// How many times a GPU mapping the GPU may have cached was removed or replaced (FlushKfdTlb). A new
	// mapping of an address nothing mapped needs no flush, and that is most of a title's mapping;
	// counting every GEM_VA flushed (and with it preempted every KFD queue) all the time.
	[[nodiscard]] uint64_t VaChanges() const { return m_va_changes.load(std::memory_order_acquire); }
	void MarkVaStale() const { m_va_changes.fetch_add(1, std::memory_order_release); }

	// KFD queues reach the VM through the KFD VMIDs' TLBs, which a render-node GEM_VA change does not
	// flush: a CS flushes its own job's VMID, and nothing runs for a KFD queue (kfd_probe avremap: a
	// KFD write after a REPLACE, or a CLEAR and MAP, landed in the previous buffer half the time). A KFD
	// map ioctl ends in a flush of the process's VMID (the BC-250's patched amdgpu: by VMID without
	// HWS, a runlist rebuild with it), even for a buffer already mapped, so a small KFD buffer is
	// mapped again. No unmap: that waits for every fence of the VM, a CS job among them, and one
	// waiting on the GPU for the KFD work this flush comes before never finishes. False without the
	// KFD process.
	bool FlushKfdTlb() {
		std::lock_guard<std::mutex> lock(m_lock);
		if (m_kfd_fd < 0 || (m_tlb_bo.kfd_handle == 0 && !CreateKfdBo(4096, &m_tlb_bo))) {
			return false;
		}
		struct kfd_ioctl_map_memory_to_gpu_args mp {};
		mp.handle               = m_tlb_bo.kfd_handle;
		mp.device_ids_array_ptr = reinterpret_cast<uintptr_t>(&m_kfd_gpu);
		mp.n_devices            = 1;
		return KfdIoctl(AMDKFD_IOC_MAP_MEMORY_TO_GPU, &mp) == 0;
	}
	[[nodiscard]] uint32_t KfdGpuId() const { return m_kfd_gpu; }

	// A KFD buffer for someone else's use (the submitter's IB buffer with KFD queues): imported into
	// DrmFd() as `*handle`, CPU-mappable through `*dmabuf_fd`, and at `*kfd_va` in the GPU VM, where KFD
	// keeps it valid. `uncached`: uncached on the GPU, like a render-node IB buffer. False without KFD
	// memory. FreeKfdBuffer undoes it.
	bool CreateKfdBuffer(uint64_t size, bool uncached, uint32_t* handle, int* dmabuf_fd, uint64_t* kfd_va, uint64_t* kfd_handle) {
		std::lock_guard<std::mutex> lock(m_lock);
		Bo                          bo;
		if (m_kfd_fd < 0 ||
		    !CreateKfdBo(size, &bo, uncached ? KFD_IOC_ALLOC_MEM_FLAGS_UNCACHED | KFD_IOC_ALLOC_MEM_FLAGS_COHERENT : 0u, kfd_va)) {
			return false;
		}
		*handle     = bo.handle;
		*dmabuf_fd  = bo.dmabuf_fd;
		*kfd_handle = bo.kfd_handle;
		return true;
	}
	void FreeKfdBuffer(uint32_t handle, int dmabuf_fd, uint64_t kfd_handle) const {
		Bo bo;
		bo.handle     = handle;
		bo.dmabuf_fd  = dmabuf_fd;
		bo.kfd_handle = kfd_handle;
		CloseBo(bo);
	}

	// Every GEM buffer a submission must list as resident: the resident blocks (unless always valid),
	// the dummy and the external ranges. Grows as blocks are promoted.
	// While another thread holds m_lock (a GEM_VA can wait in the kernel for the GPU work that maps
	// the old range, and that work may be waiting for the very submission asking here: My First Gran
	// Turismo deadlocked so), the last list is returned. A block promoted meanwhile is listed by the
	// next submission; always-valid blocks are never listed.
	[[nodiscard]] std::vector<uint32_t> CreatedHandles() const {
		std::unique_lock<std::mutex> lock(m_lock, std::try_to_lock);
		if (!lock.owns_lock()) {
			std::lock_guard<std::mutex> last(m_handles_mutex);
			if (!m_last_handles.empty()) {
				return m_last_handles;
			}
			lock.lock();
		}
		std::vector<uint32_t> handles;
		handles.reserve(m_blocks.size() + m_external.size() + 1);
		for (const auto& block: m_blocks) {
			if (block.resident && !block.bo.always_valid) {
				handles.push_back(block.bo.handle);
			}
		}
		handles.push_back(m_dummy.handle);
		for (const auto& [vaddr, ext]: m_external) {
			if (ext.handle != 0) {
				handles.push_back(ext.handle);
			}
		}
		{
			std::lock_guard<std::mutex> last(m_handles_mutex);
			m_last_handles = handles;
		}
		return handles;
	}

	// Looks up the GEM buffer backing `offset`. Returns false if `offset` is out of range or its
	// block is not resident.
	bool Lookup(uint64_t offset, uint32_t* handle, uint64_t* offset_in_block) const {
		if (offset >= m_size) {
			return false;
		}
		std::lock_guard<std::mutex> lock(m_lock);
		const uint64_t              index = offset / BlockSize();
		if (!m_blocks[index].resident) {
			return false;
		}
		if (handle != nullptr) {
			*handle = m_blocks[index].bo.handle;
		}
		if (offset_in_block != nullptr) {
			*offset_in_block = offset - index * BlockSize();
		}
		return true;
	}

private:
	struct Bo {
		uint32_t handle       = 0;  // in m_drm_fd
		int      dmabuf_fd    = -1; // -1 for an always-valid buffer
		uint64_t kfd_handle   = 0;  // the KFD allocation behind `handle` (KFD memory), 0 otherwise
		uint64_t mmap_offset  = 0;  // an always-valid buffer's DRM mmap offset
		bool     always_valid = false;
	};
	struct Block {
		Bo       bo;
		uint64_t size     = 0;
		bool     resident = false;
		bool     code     = false; // holds executable views: stays in the memfd (see PromoteBlockUnlocked)
	};
	// A guest view: `size` bytes of the backing from `offset`, mapped at the key address; `gpu` when
	// it is also mapped into the GPU VM.
	struct View {
		uint64_t size   = 0;
		uint64_t offset = 0;
		bool     gpu    = false;
	};

	GemBacking() = default;

	int Ioctl(unsigned long request, void* arg) const {
		int r = -1;
		do {
			r = ::ioctl(m_drm_fd, request, arg);
		} while (r == -1 && (errno == EINTR || errno == EAGAIN));
		return r == -1 ? -errno : 0;
	}

	// KYTY_NATIVE_GPU_MTYPE=uc|nc|cc|wc (experiment): the GL2 cache mode of guest memory's GPU
	// mappings instead of amdgpu's default for GTT. uc makes every GPU write reach memory at once, so
	// the CPU sees a label without a cache write-back.
	static uint32_t MtypeFlags() {
		static const uint32_t flags = [] {
			const char* v = std::getenv("KYTY_NATIVE_GPU_MTYPE");
			if (v == nullptr) {
				return 0u;
			}
			const std::string m = v;
			return m == "uc" ? AMDGPU_VM_MTYPE_UC : m == "nc" ? AMDGPU_VM_MTYPE_NC : m == "cc" ? AMDGPU_VM_MTYPE_CC
			                                     : m == "wc"  ? AMDGPU_VM_MTYPE_WC
			                                                  : 0u;
		}();
		return flags;
	}
	static inline const uint32_t kRwxPageFlags =
	    AMDGPU_VM_PAGE_READABLE | AMDGPU_VM_PAGE_WRITEABLE | AMDGPU_VM_PAGE_EXECUTABLE | MtypeFlags();

	static bool CoherentType(int type) {
		static const std::vector<int> types = [] {
			std::vector<int> t;
			const char*      v = std::getenv("KYTY_NATIVE_GPU_COHERENT_TYPES");
			if (v == nullptr) {
				return t;
			}
			for (const char* p = v; *p != '\0';) {
				char*      next = nullptr;
				const long n    = std::strtol(p, &next, 0);
				if (next == p) {
					break; // "none" and anything else that is not a number
				}
				t.push_back(static_cast<int>(n));
				p = *next == ',' ? next + 1 : next;
			}
			return t;
		}();
		return std::find(types.begin(), types.end(), type) != types.end();
	}

	// The MTYPE of coherent memory types: C_RW_S (AMDGPU_VM_MTYPE_RW) as the PS5 maps its shared type,
	// where the patched amdgpu has it (tools/bc250-amdgpu, bc250_mtype_rw), else CC.
	static uint32_t CoherentMtype() {
		static const uint32_t mtype = [] {
			FILE* f = std::fopen("/sys/module/amdgpu/parameters/bc250_mtype_rw", "re");
			int   v = 0;
			if (f != nullptr) {
				if (std::fscanf(f, "%d", &v) != 1) {
					v = 0;
				}
				std::fclose(f);
			}
			return v == 1 ? AMDGPU_VM_MTYPE_RW : AMDGPU_VM_MTYPE_CC;
		}();
		return mtype;
	}

	// REPLACE of [vaddr, vaddr + size), backing bytes from `backing` on: the parts in coherent memory
	// with MTYPE CC (unless KYTY_NATIVE_GPU_MTYPE sets one for all). The caller holds m_lock.
	int VaReplace(uint32_t handle, uint64_t vaddr, uint64_t offset_in_bo, uint64_t backing, uint64_t size) const {
		if (m_coherent.empty() || MtypeFlags() != 0) {
			return VaOp(AMDGPU_VA_OP_REPLACE, handle, vaddr, offset_in_bo, size);
		}
		int      r    = 0;
		uint64_t done = 0;
		while (done < size) {
			const uint64_t at       = backing + done;
			uint64_t       end      = backing + size;
			bool           coherent = false;
			auto           next     = m_coherent.upper_bound(at);
			if (next != m_coherent.begin() && std::prev(next)->second > at) {
				coherent = true;
				end      = std::min(end, std::prev(next)->second);
			} else if (next != m_coherent.end()) {
				end = std::min(end, next->first);
			}
			const uint64_t len = end - at;
			const int      one = VaOp(AMDGPU_VA_OP_REPLACE, handle, vaddr + done, offset_in_bo + done, len,
			                         coherent ? kRwxPageFlags | CoherentMtype() : kRwxPageFlags);
			r                  = r != 0 ? r : one;
			done += len;
		}
		return r;
	}

	int VaOp(uint32_t op, uint32_t handle, uint64_t vaddr, uint64_t offset_in_bo, uint64_t size,
	         uint32_t page_flags = kRwxPageFlags) const {
		struct drm_amdgpu_gem_va v;
		std::memset(&v, 0, sizeof(v));
		v.handle       = handle;
		v.operation    = op;
		v.flags        = page_flags;
		v.va_address   = vaddr;
		v.offset_in_bo = offset_in_bo;
		v.map_size     = size;
		const int r    = Ioctl(DRM_IOCTL_AMDGPU_GEM_VA, &v);
		if (r != 0) {
			std::fprintf(stderr, "GemBacking: GEM_VA op %u at 0x%llx (+0x%llx) failed: %s\n", op,
			             static_cast<unsigned long long>(vaddr), static_cast<unsigned long long>(size),
			             std::strerror(-r));
		}
		return r;
	}

	void GemClose(uint32_t handle) const {
		struct drm_gem_close c;
		std::memset(&c, 0, sizeof(c));
		c.handle = handle;
		Ioctl(DRM_IOCTL_GEM_CLOSE, &c);
	}

	// KYTY_NATIVE_GPU_KFD_MEMORY=1: the KFD process of this VM (AMDKFD_IOC_ACQUIRE_VM on m_drm_fd), for
	// KFD buffers. KFD keeps its buffers resident without being listed, which is what lets a KFD
	// user-mode queue (no CS ioctl) reach guest memory; a render-node CS still lists them (an eviction
	// does not wait for a job that did not).
	void InitKfd() {
		const char* v = std::getenv("KYTY_NATIVE_GPU_KFD_MEMORY");
		if ((v == nullptr || v[0] != '1') && !KfdQueuesWanted()) {
			return;
		}
		struct stat st {};
		const int   minor = ::fstat(m_drm_fd, &st) == 0 ? static_cast<int>(st.st_rdev & 0xff) : -1;
		for (int n = 0; n < 16 && m_kfd_gpu == 0; n++) {
			char path[128];
			std::snprintf(path, sizeof(path), "/sys/class/kfd/kfd/topology/nodes/%d/properties", n);
			FILE* f = std::fopen(path, "re");
			if (f == nullptr) {
				continue;
			}
			char line[128];
			int  render_minor = -1;
			while (std::fgets(line, sizeof(line), f) != nullptr) {
				std::sscanf(line, "drm_render_minor %d", &render_minor);
			}
			std::fclose(f);
			if (render_minor >= 128 && (render_minor & 0xff) == (minor & 0xff)) {
				std::snprintf(path, sizeof(path), "/sys/class/kfd/kfd/topology/nodes/%d/gpu_id", n);
				if (FILE* g = std::fopen(path, "re"); g != nullptr) {
					unsigned id = 0;
					if (std::fscanf(g, "%u", &id) == 1) {
						m_kfd_gpu = id;
					}
					std::fclose(g);
				}
			}
		}
		m_kfd_fd = ::open("/dev/kfd", O_RDWR | O_CLOEXEC);
		struct kfd_ioctl_acquire_vm_args acq {};
		acq.drm_fd = static_cast<uint32_t>(m_drm_fd);
		acq.gpu_id = m_kfd_gpu;
		if (m_kfd_fd < 0 || m_kfd_gpu == 0 || ::ioctl(m_kfd_fd, AMDKFD_IOC_ACQUIRE_VM, &acq) != 0) {
			std::fprintf(stderr, "GemBacking: KFD memory unavailable (gpu id %u): %s; using render-node buffers\n", m_kfd_gpu,
			             std::strerror(errno));
			if (m_kfd_fd >= 0) {
				::close(m_kfd_fd);
			}
			m_kfd_fd  = -1;
			m_kfd_gpu = 0;
			return;
		}
		// libhsakmt's first call on a GPU: the process's memory policy, from which KFD programs its VMID's
		// SH_MEM_CONFIG and apertures. Without it the KFD VMID ran with SH_MEM_BASES and SH_MEM_CONFIG 0.
		struct kfd_ioctl_set_memory_policy_args policy {};
		policy.gpu_id           = m_kfd_gpu;
		policy.default_policy   = KFD_IOC_CACHE_POLICY_COHERENT;
		policy.alternate_policy = KFD_IOC_CACHE_POLICY_COHERENT;
		if (::ioctl(m_kfd_fd, AMDKFD_IOC_SET_MEMORY_POLICY, &policy) != 0) {
			std::fprintf(stderr, "GemBacking: KFD SET_MEMORY_POLICY failed: %s\n", std::strerror(errno));
		}
		struct kfd_process_device_apertures ap[4] {};
		struct kfd_ioctl_get_process_apertures_new_args apn {};
		apn.kfd_process_device_apertures_ptr = reinterpret_cast<uintptr_t>(ap);
		apn.num_of_nodes                     = 4;
		if (::ioctl(m_kfd_fd, AMDKFD_IOC_GET_PROCESS_APERTURES_NEW, &apn) == 0) {
			for (uint32_t i = 0; i < apn.num_of_nodes && i < 4; i++) {
				if (ap[i].gpu_id == m_kfd_gpu) {
					std::fprintf(stderr, "GemBacking: KFD apertures: LDS 0x%llx, scratch 0x%llx, GPUVM 0x%llx-0x%llx\n",
					             static_cast<unsigned long long>(ap[i].lds_base), static_cast<unsigned long long>(ap[i].scratch_base),
					             static_cast<unsigned long long>(ap[i].gpuvm_base), static_cast<unsigned long long>(ap[i].gpuvm_limit));
				}
			}
		}
		// KFD queues over always-valid blocks need the KFD process only (queues, the IB buffer).
		m_kfd_memory = (v != nullptr && v[0] == '1') || !AlwaysValidWanted();
		if (!m_kfd_memory) {
			std::fprintf(stderr, "GemBacking: KFD process acquired for queues (gpu id %u); blocks stay always-valid buffers\n", m_kfd_gpu);
			return;
		}
		// KYTY_NATIVE_GPU_KFD_USERPTR=0 keeps the external ranges render-node userptr buffers.
		const char* up = std::getenv("KYTY_NATIVE_GPU_KFD_USERPTR");
		m_kfd_userptr  = KfdQueuesWanted() && (up == nullptr || up[0] != '0');
		std::fprintf(stderr, "GemBacking: blocks%s are KFD buffers (gpu id %u)\n", m_kfd_userptr ? " and external ranges" : "",
		             m_kfd_gpu);
	}

	int KfdIoctl(unsigned long request, void* arg) const {
		int r = -1;
		do {
			r = ::ioctl(m_kfd_fd, request, arg);
		} while (r == -1 && (errno == EINTR || errno == EAGAIN));
		return r == -1 ? -errno : 0;
	}

	// A KFD GTT buffer at a private VA of its own (KFD maps every allocation somewhere), exported as a
	// dma-buf and imported into m_drm_fd, so it is used exactly like a render-node buffer.
	bool CreateKfdBo(uint64_t size, Bo* out, uint32_t extra_flags = 0, uint64_t* va = nullptr) const {
		struct kfd_ioctl_alloc_memory_of_gpu_args a {};
		a.va_addr = m_kfd_va_next;
		a.size    = size;
		a.gpu_id  = m_kfd_gpu;
		a.flags   = KFD_IOC_ALLOC_MEM_FLAGS_GTT | KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE | KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE |
		          KFD_IOC_ALLOC_MEM_FLAGS_NO_SUBSTITUTE | extra_flags;
		int r = KfdIoctl(AMDKFD_IOC_ALLOC_MEMORY_OF_GPU, &a);
		if (r != 0) {
			std::fprintf(stderr, "GemBacking: KFD allocation of %llu MiB failed: %s\n",
			             static_cast<unsigned long long>(size >> 20), std::strerror(-r));
			return false;
		}
		if (va != nullptr) {
			*va = a.va_addr;
		}
		m_kfd_va_next += (size + (2ull << 20) - 1) & ~((2ull << 20) - 1);
		Bo bo;
		bo.kfd_handle = a.handle;
		struct kfd_ioctl_map_memory_to_gpu_args mp {};
		mp.handle               = a.handle;
		mp.device_ids_array_ptr = reinterpret_cast<uintptr_t>(&m_kfd_gpu);
		mp.n_devices            = 1;
		struct kfd_ioctl_export_dmabuf_args ex {};
		ex.handle = a.handle;
		ex.flags  = O_RDWR | O_CLOEXEC;
		struct drm_prime_handle ph {};
		if ((r = KfdIoctl(AMDKFD_IOC_MAP_MEMORY_TO_GPU, &mp)) != 0 || (r = KfdIoctl(AMDKFD_IOC_EXPORT_DMABUF, &ex)) != 0) {
			std::fprintf(stderr, "GemBacking: KFD map/export failed: %s\n", std::strerror(-r));
			CloseBo(bo);
			return false;
		}
		bo.dmabuf_fd = static_cast<int>(ex.dmabuf_fd);
		ph.fd        = bo.dmabuf_fd;
		if ((r = Ioctl(DRM_IOCTL_PRIME_FD_TO_HANDLE, &ph)) != 0) {
			std::fprintf(stderr, "GemBacking: KFD buffer import failed: %s\n", std::strerror(-r));
			CloseBo(bo);
			return false;
		}
		bo.handle = ph.handle;
		*out      = bo;
		return true;
	}

	// A cached GTT buffer (TTM zeroes it) exported as a dma-buf for CPU mappings.
	bool CreateBo(uint64_t size, Bo* out) const {
		if (m_kfd_memory) {
			return CreateKfdBo(size, out);
		}
		union drm_amdgpu_gem_create c;
		std::memset(&c, 0, sizeof(c));
		c.in.bo_size   = size;
		c.in.alignment = 2ull * 1024 * 1024;
		c.in.domains   = AMDGPU_GEM_DOMAIN_GTT;
		int r          = Ioctl(DRM_IOCTL_AMDGPU_GEM_CREATE, &c);
		if (r != 0) {
			std::fprintf(stderr, "GemBacking: GEM_CREATE of %llu MiB failed: %s\n",
			             static_cast<unsigned long long>(size / (1024 * 1024)), std::strerror(-r));
			return false;
		}
		struct drm_prime_handle p;
		std::memset(&p, 0, sizeof(p));
		p.handle = c.out.handle;
		p.flags  = DRM_CLOEXEC | DRM_RDWR;
		r        = Ioctl(DRM_IOCTL_PRIME_HANDLE_TO_FD, &p);
		if (r != 0) {
			std::fprintf(stderr, "GemBacking: dma-buf export failed: %s\n", std::strerror(-r));
			GemClose(c.out.handle);
			return false;
		}
		out->handle    = c.out.handle;
		out->dmabuf_fd = p.fd;
		return true;
	}

	// A block's buffer: always valid with AlwaysValidWanted() (no dma-buf), otherwise CreateBo.
	bool CreateBlockBo(uint64_t size, Bo* out) const {
		if (!AlwaysValidWanted() || m_kfd_memory) {
			return CreateBo(size, out);
		}
		union drm_amdgpu_gem_create c;
		std::memset(&c, 0, sizeof(c));
		c.in.bo_size      = size;
		c.in.alignment    = 2ull * 1024 * 1024;
		c.in.domains      = AMDGPU_GEM_DOMAIN_GTT;
		c.in.domain_flags = AMDGPU_GEM_CREATE_VM_ALWAYS_VALID;
		// KYTY_NATIVE_GPU_BO_USWC=1 (experiment): write-combined for the CPU, so the GPU reaches the
		// block without snooping the CPU's caches, as the PS5 GPU reaches its "Garlic" memory. A cached
		// GTT buffer is snooped on every GPU access.
		static const bool uswc = [] {
			const char* v = std::getenv("KYTY_NATIVE_GPU_BO_USWC");
			return v != nullptr && v[0] == '1';
		}();
		if (uswc) {
			c.in.domain_flags |= AMDGPU_GEM_CREATE_CPU_GTT_USWC;
		}
		int r = Ioctl(DRM_IOCTL_AMDGPU_GEM_CREATE, &c);
		if (r != 0) {
			std::fprintf(stderr, "GemBacking: always-valid GEM_CREATE of %llu MiB failed: %s\n",
			             static_cast<unsigned long long>(size / (1024 * 1024)), std::strerror(-r));
			return false;
		}
		union drm_amdgpu_gem_mmap m;
		std::memset(&m, 0, sizeof(m));
		m.in.handle = c.out.handle;
		if ((r = Ioctl(DRM_IOCTL_AMDGPU_GEM_MMAP, &m)) != 0) {
			std::fprintf(stderr, "GemBacking: GEM_MMAP failed: %s\n", std::strerror(-r));
			GemClose(c.out.handle);
			return false;
		}
		out->handle       = c.out.handle;
		out->mmap_offset  = m.out.addr_ptr;
		out->always_valid = true;
		return true;
	}

	// Maps bytes [inner, inner + len) of `bo` (of `bo_size` bytes) at exactly `dst` with `prot`,
	// replacing whatever was there. A dma-buf maps from any page offset; an always-valid buffer is
	// mapped whole elsewhere, the piece moved to `dst` (one mremap, atomic for other threads) and the
	// rest unmapped. Safe on the mover thread.
	bool MapBoPiece(const Bo& bo, uint64_t bo_size, uint64_t inner, uint64_t len, int prot, void* dst) const {
		if (bo.dmabuf_fd >= 0) {
			return ::mmap(dst, len, prot, MAP_SHARED | MAP_FIXED, bo.dmabuf_fd, static_cast<off_t>(inner)) == dst;
		}
		auto* whole =
		    static_cast<uint8_t*>(::mmap(nullptr, bo_size, prot, MAP_SHARED, m_drm_fd, static_cast<off_t>(bo.mmap_offset)));
		if (whole == MAP_FAILED) {
			return false;
		}
		const bool moved = ::mremap(whole + inner, len, len, MREMAP_MAYMOVE | MREMAP_FIXED, dst) == dst;
		if (inner != 0) {
			::munmap(whole, inner);
		}
		if (inner + len < bo_size) {
			::munmap(whole + inner + len, bo_size - inner - len);
		}
		if (!moved) {
			::munmap(whole + inner, len);
		}
		return moved;
	}

	// The whole of `bo` read-write at an address of the kernel's choice, or nullptr.
	[[nodiscard]] uint8_t* MapBoWhole(const Bo& bo, uint64_t bo_size) const {
		void* p = ::mmap(nullptr, bo_size, PROT_READ | PROT_WRITE, MAP_SHARED, bo.dmabuf_fd >= 0 ? bo.dmabuf_fd : m_drm_fd,
		                 static_cast<off_t>(bo.dmabuf_fd >= 0 ? 0 : bo.mmap_offset));
		return p == MAP_FAILED ? nullptr : static_cast<uint8_t*>(p);
	}

	void CloseBo(Bo& bo) const {
		if (bo.dmabuf_fd >= 0) {
			::close(bo.dmabuf_fd);
		}
		if (bo.handle != 0) {
			GemClose(bo.handle);
		}
		if (bo.kfd_handle != 0) {
			struct kfd_ioctl_unmap_memory_from_gpu_args um {};
			um.handle               = bo.kfd_handle;
			um.device_ids_array_ptr = reinterpret_cast<uintptr_t>(&m_kfd_gpu);
			um.n_devices            = 1;
			KfdIoctl(AMDKFD_IOC_UNMAP_MEMORY_FROM_GPU, &um);
			struct kfd_ioctl_free_memory_of_gpu_args fr {};
			fr.handle = bo.kfd_handle;
			KfdIoctl(AMDKFD_IOC_FREE_MEMORY_OF_GPU, &fr);
		}
		bo = Bo {};
	}

	// Calls cb(block index, offset in block, offset from `offset`, length) for each block piece of
	// backing range [offset, offset + size); stops at the first false.
	template <typename Cb>
	bool ForEachBlockPiece(uint64_t offset, uint64_t size, Cb&& cb) const {
		uint64_t done = 0;
		while (done < size) {
			const uint64_t at    = offset + done;
			const uint64_t index = at / BlockSize();
			const uint64_t inner = at - index * BlockSize();
			const uint64_t part  = std::min(size - done, m_blocks[index].size - inner);
			if (!cb(index, inner, done, part)) {
				return false;
			}
			done += part;
		}
		return true;
	}

	[[nodiscard]] std::map<uint64_t, View>::const_iterator FindViewUnlocked(uint64_t vaddr) const {
		auto it = m_views.upper_bound(vaddr);
		if (it == m_views.begin()) {
			return m_views.end();
		}
		--it;
		return vaddr < it->first + it->second.size ? it : m_views.end();
	}

	// Calls cb(view address, guest address, backing offset, length) for each piece of a view that
	// overlaps guest range [vaddr, vaddr + size).
	template <typename Cb>
	void ForEachViewOverlap(uint64_t vaddr, uint64_t size, Cb&& cb) const {
		const uint64_t end = vaddr + size;
		auto           it  = m_views.upper_bound(vaddr);
		if (it != m_views.begin()) {
			--it;
		}
		for (; it != m_views.end() && it->first < end; ++it) {
			const uint64_t s = std::max(it->first, vaddr);
			const uint64_t e = std::min(it->first + it->second.size, end);
			if (s < e) {
				cb(it->first, s, it->second.offset + (s - it->first), e - s);
			}
		}
	}

	// Forgets views in [vaddr, vaddr + size), keeping the parts of straddling views outside it.
	void EraseViewsUnlocked(uint64_t vaddr, uint64_t size) {
		const uint64_t                            end = vaddr + size;
		std::vector<std::pair<uint64_t, View>>    keep;
		auto                                      it = m_views.upper_bound(vaddr);
		if (it != m_views.begin()) {
			--it;
		}
		while (it != m_views.end() && it->first < end) {
			const uint64_t vs = it->first;
			const uint64_t ve = vs + it->second.size;
			if (ve <= vaddr) {
				++it;
				continue;
			}
			const View view = it->second;
			it              = m_views.erase(it);
			if (vs < vaddr) {
				keep.emplace_back(vs, View {vaddr - vs, view.offset, view.gpu});
			}
			if (end < ve) {
				keep.emplace_back(end, View {ve - end, view.offset + (end - vs), view.gpu});
			}
		}
		for (const auto& [at, view]: keep) {
			m_views[at] = view;
		}
	}

	// The host mappings (/proc/self/maps) at one moment, sorted, for asking about many ranges: a
	// title maps guest memory in many pieces (My First Gran Turismo: 17,000 mappings), and reading the
	// file once per guest view made promoting a block take a second or more.
	class HostMaps {
	public:
		HostMaps() {
			const int fd = ::open("/proc/self/maps", O_RDONLY | O_CLOEXEC);
			if (fd < 0) {
				return;
			}
			std::string text;
			char        chunk[65536];
			for (ssize_t n = 0; (n = ::read(fd, chunk, sizeof(chunk))) > 0;) {
				text.append(chunk, static_cast<size_t>(n));
			}
			::close(fd);
			m_read = true;
			for (size_t pos = 0; pos < text.size();) {
				const size_t eol  = text.find('\n', pos);
				const char*  line = text.c_str() + pos;
				char*        next = nullptr;
				Entry        e;
				e.start = std::strtoull(line, &next, 16);
				if (next != nullptr && *next == '-') {
					e.end = std::strtoull(next + 1, &next, 16);
					if (next != nullptr && *next == ' ' && next[1] != '\0' && next[2] != '\0' && next[3] != '\0') {
						e.prot = (next[1] == 'r' ? PROT_READ : 0) | (next[2] == 'w' ? PROT_WRITE : 0) | (next[3] == 'x' ? PROT_EXEC : 0);
						m_entries.push_back(e);
					}
				}
				pos = eol == std::string::npos ? text.size() : eol + 1;
			}
		}
		// Calls cb(start, size, prot) for each host mapping overlapping [vaddr, vaddr + size). If maps
		// could not be read the whole range is reported read-write.
		template <typename Cb>
		void ForEach(uint64_t vaddr, uint64_t size, Cb&& cb) const {
			if (!m_read) {
				cb(vaddr, size, PROT_READ | PROT_WRITE);
				return;
			}
			const uint64_t end = vaddr + size;
			auto it = std::upper_bound(m_entries.begin(), m_entries.end(), vaddr, [](uint64_t v, const Entry& e) { return v < e.end; });
			for (; it != m_entries.end() && it->start < end; ++it) {
				const uint64_t cs = std::max<uint64_t>(it->start, vaddr);
				const uint64_t ce = std::min<uint64_t>(it->end, end);
				cb(cs, ce - cs, it->prot);
			}
		}

	private:
		struct Entry {
			uint64_t start = 0;
			uint64_t end   = 0;
			int      prot  = 0;
		};
		std::vector<Entry> m_entries; // /proc/self/maps lists them by address
		bool               m_read = false;
	};

	// Calls cb(start, size, prot) for each host mapping overlapping [vaddr, vaddr + size), from
	// /proc/self/maps, so a remap can keep the protection the guest (or Kyty's write tracking) set.
	// If maps cannot be read the whole range is reported read-write.
	template <typename Cb>
	static void ForEachHostProtection(uint64_t vaddr, uint64_t size, Cb&& cb) {
		HostMaps().ForEach(vaddr, size, std::forward<Cb>(cb));
	}

	// Calls cb(offset in block, length) for each data extent of block `index` in the memfd.
	template <typename Cb>
	void ForEachDataExtent(uint64_t index, Cb&& cb) const {
		const uint64_t start = index * BlockSize();
		const uint64_t end   = start + m_blocks[index].size;
		uint64_t       pos   = start;
		while (pos < end) {
			const off_t data = ::lseek(m_mem_fd, static_cast<off_t>(pos), SEEK_DATA);
			if (data < 0 || static_cast<uint64_t>(data) >= end) {
				break;
			}
			off_t hole = ::lseek(m_mem_fd, data, SEEK_HOLE);
			if (hole < 0 || static_cast<uint64_t>(hole) > end) {
				hole = static_cast<off_t>(end);
			}
			cb(static_cast<uint64_t>(data) - start, static_cast<uint64_t>(hole - data));
			pos = static_cast<uint64_t>(hole);
		}
	}

	struct MovePiece {
		uint64_t addr;
		uint64_t len;
		int      prot;
		uint64_t inner; // offset in the block
	};
	struct MoveJob {
		uint64_t               index  = 0;
		uint64_t               start  = 0; // backing offset of the block
		uint64_t               size   = 0;
		uint8_t*               bo_cpu   = nullptr;
		const Bo*              bo       = nullptr; // the block's new buffer
		const uint8_t*         from_cpu = nullptr; // the block's old buffer, when it had one
		std::vector<MovePiece> pieces;
		bool                   moved  = false;
		bool                   done   = false;
	};

	// The move proper: write-protect every CPU mapping of the block, copy its data extents into the
	// buffer, map everything onto the buffer with its protection back. A guest write meanwhile faults
	// and waits in WaitIfPromoting. It runs on the mover thread, whose stack is ordinary host memory:
	// a guest thread's stack can live in the very block being moved (games allocate thread stacks
	// in direct memory, and Kyty runs on guest stacks), and a thread that write-protects its own
	// stack faults on its next push and waits for itself forever.
	void MoveBlock(MoveJob* job) const {
		s_promoting.fetch_add(1, std::memory_order_acq_rel);
		for (const auto& p: job->pieces) {
			if ((p.prot & PROT_WRITE) != 0) {
				::mprotect(reinterpret_cast<void*>(p.addr), p.len, p.prot & ~PROT_WRITE);
			}
		}
		// pread rather than a mapping of the memfd, which would allocate pages for the holes; holes
		// stay zero in the buffer.
		if (job->from_cpu != nullptr) {
			std::memcpy(job->bo_cpu, job->from_cpu, job->size);
		}
		ForEachDataExtent(job->index, [&](uint64_t at, uint64_t len) {
			uint64_t done = 0;
			while (done < len) {
				const ssize_t n = ::pread(m_mem_fd, job->bo_cpu + at + done, len - done,
				                          static_cast<off_t>(job->start + at + done));
				if (n <= 0) {
					break;
				}
				done += static_cast<uint64_t>(n);
			}
		});
		bool moved = true;
		for (const auto& p: job->pieces) {
			void* const dst = reinterpret_cast<void*>(p.addr);
			moved = MapBoPiece(*job->bo, job->size, p.inner, p.len, p.prot, dst) && moved;
		}
		job->moved = moved;
		s_promoting.fetch_sub(1, std::memory_order_acq_rel);
	}

	// Runs MoveBlock(job) on the mover thread (started on first use) and waits for it. Only this
	// waiting touches the caller's stack, and a write there that faults during the move just waits.
	void RunOnMover(MoveJob* job) {
		std::unique_lock<std::mutex> lock(m_mover_lock);
		if (!m_mover.joinable()) {
			m_mover = std::thread([this] {
				std::unique_lock<std::mutex> l(m_mover_lock);
				for (;;) {
					m_mover_cv.wait(l, [this] { return m_mover_job != nullptr || m_mover_stop; });
					if (m_mover_stop) {
						return;
					}
					MoveJob* j = m_mover_job;
					l.unlock();
					MoveBlock(j);
					l.lock();
					j->done     = true;
					m_mover_job = nullptr;
					m_mover_cv.notify_all();
				}
			});
		}
		m_mover_job = job;
		m_mover_cv.notify_all();
		m_mover_cv.wait(lock, [job] { return job->done; });
	}

	static uint64_t MapReserve() {
		static const uint64_t reserve = [] {
			const char* v = std::getenv("KYTY_NATIVE_GPU_MAP_RESERVE_MIB");
			return (v != nullptr ? std::strtoull(v, nullptr, 0) : 1536ull) << 20u;
		}();
		return reserve;
	}

	// True when `bytes` more of resident memory still leave MapReserve() of MemAvailable.
	static bool MapBudgetAllows(uint64_t bytes) {
		FILE* f = std::fopen("/proc/meminfo", "re");
		if (f == nullptr) {
			return true;
		}
		char               line[128];
		unsigned long long kib = 0;
		while (std::fgets(line, sizeof(line), f) != nullptr && std::sscanf(line, "MemAvailable: %llu kB", &kib) != 1) {
		}
		std::fclose(f);
		return kib == 0 || (kib << 10u) >= bytes + MapReserve();
	}

	// Moves block `index` into a new buffer: from the memfd, or, for a resident block, from its old
	// buffer, which is freed (the caller has made sure the GPU no longer uses it). `exportable`: an
	// ordinary buffer with a dma-buf even when blocks are otherwise always valid.
	bool PromoteBlockUnlocked(uint64_t index, bool exportable = false) {
		Block&         block = m_blocks[index];
		const uint64_t start = index * BlockSize();
		Bo             bo;
		if (!(exportable ? CreateBo(block.size, &bo) : CreateBlockBo(block.size, &bo))) {
			return false;
		}
		uint8_t* bo_cpu   = MapBoWhole(bo, block.size);
		uint8_t* from_cpu = block.resident ? MapBoWhole(block.bo, block.size) : nullptr;
		if (bo_cpu == nullptr || (block.resident && from_cpu == nullptr)) {
			if (bo_cpu != nullptr) {
				::munmap(bo_cpu, block.size);
			}
			std::fprintf(stderr, "GemBacking: CPU map of block %llu failed: %s\n",
			             static_cast<unsigned long long>(index), std::strerror(errno));
			CloseBo(bo);
			return false;
		}
		// Every CPU mapping of the block: (address, length, protection, offset in block), the alias slot
		// first, then each guest view piece split at protection boundaries. Kept in a heap job: the
		// move runs on the mover thread and must not write to this thread's stack (see MoveBlock).
		uint64_t touched = 0; // bytes the CPU had written, the rest of the block is padding
		ForEachDataExtent(index, [&](uint64_t, uint64_t len) { touched += len; });
		auto job    = std::make_unique<MoveJob>();
		job->index  = index;
		job->start  = start;
		job->size   = block.size;
		job->bo_cpu = bo_cpu;
		job->bo     = &bo;
		job->from_cpu = from_cpu;
		std::vector<std::pair<uint64_t, uint64_t>> gpu_pieces; // (guest address, offset in block)
		std::vector<uint64_t>                      gpu_lengths;
		job->pieces.push_back(MovePiece {reinterpret_cast<uint64_t>(m_alias + start), block.size, PROT_READ | PROT_WRITE, 0});
		bool                  executable = false;
		std::optional<HostMaps> maps; // read once, at the first view of the block
		for (const auto& [view_addr, view]: m_views) {
			const uint64_t vs = std::max(view.offset, start);
			const uint64_t ve = std::min(view.offset + view.size, start + block.size);
			if (vs >= ve) {
				continue;
			}
			const uint64_t guest = view_addr + (vs - view.offset);
			if (!maps) {
				maps.emplace();
			}
			maps->ForEach(guest, ve - vs, [&](uint64_t s, uint64_t len, int prot) {
				job->pieces.push_back(MovePiece {s, len, prot, vs - start + (s - guest)});
				executable = executable || (prot & PROT_EXEC) != 0;
			});
			if (view.gpu) {
				gpu_pieces.emplace_back(guest, vs - start);
				gpu_lengths.push_back(ve - vs);
			}
		}
		// A dma-buf cannot be mapped executable: a block the guest runs code from stays in the memfd.
		if (executable) {
			std::fprintf(stderr, "GemBacking: block %llu holds executable views, not made resident\n",
			             static_cast<unsigned long long>(index));
			::munmap(bo_cpu, block.size);
			if (from_cpu != nullptr) {
				::munmap(from_cpu, block.size);
			}
			CloseBo(bo);
			block.code = true;
			return false;
		}
		RunOnMover(job.get());
		const bool moved = job->moved;
		if (!moved) {
			std::fprintf(stderr, "GemBacking: remapping block %llu failed: %s\n",
			             static_cast<unsigned long long>(index), std::strerror(errno));
		}
		::munmap(bo_cpu, block.size);
		// GPU mappings of the block's views now map the buffer instead of the dummy (or the old buffer).
		for (size_t i = 0; i < gpu_pieces.size(); i++) {
			VaReplace(bo.handle, gpu_pieces[i].first, gpu_pieces[i].second, start + gpu_pieces[i].second, gpu_lengths[i]);
		}
		if (from_cpu != nullptr) {
			MarkVaStale(); // the GPU mappings moved from a buffer the GPU used
			::munmap(from_cpu, block.size);
			CloseBo(block.bo);
			block.bo = bo;
			std::fprintf(stderr, "GemBacking: block %llu moved to an exportable buffer\n", static_cast<unsigned long long>(index));
			return true;
		}
		::fallocate(m_mem_fd, FALLOC_FL_PUNCH_HOLE | FALLOC_FL_KEEP_SIZE, static_cast<off_t>(start),
		            static_cast<off_t>(block.size));
		block.bo       = bo;
		block.resident = true;
		m_resident_bytes += block.size;
		m_touched_bytes += touched;
		std::fprintf(stderr, "GemBacking: block %llu resident (%llu KiB written by the CPU), %llu MiB of guest memory in GTT\n",
		             static_cast<unsigned long long>(index), static_cast<unsigned long long>(touched >> 10u),
		             static_cast<unsigned long long>(m_resident_bytes / (1024 * 1024)));
		return true;
	}

	// Calls cb(start, size, writable, executable) for each readable host VMA slice overlapping [vaddr, vaddr+size),
	// read from /proc/self/maps. Guest memory is mapped at host VA == guest VA; `writable` is the
	// slice's host write permission. Unreadable slices (no-access or execute-only) are skipped, since a
	// userptr cannot fault them in. If maps cannot be read, the whole range is offered once as writable
	// and the caller's protection choice still applies.
	template <typename Cb>
	static void ForEachReadableSubrange(uint64_t vaddr, uint64_t size, Cb&& cb) {
		const uint64_t end  = vaddr + size;
		FILE*          maps = ::fopen("/proc/self/maps", "re");
		if (maps == nullptr) {
			cb(vaddr, size, true, false);
			return;
		}
		char line[512];
		while (std::fgets(line, sizeof(line), maps) != nullptr) {
			unsigned long long s = 0, e = 0;
			char               perms[8] = {0};
			if (std::sscanf(line, "%llx-%llx %7s", &s, &e, perms) != 3) {
				continue;
			}
			if (e <= vaddr || s >= end || perms[0] != 'r') {
				continue;
			}
			const uint64_t cs = std::max<uint64_t>(s, vaddr);
			const uint64_t ce = std::min<uint64_t>(e, end);
			if (cs < ce) {
				cb(cs, ce - cs, perms[1] == 'w', perms[2] == 'x');
			}
		}
		::fclose(maps);
	}

	// Registers [vaddr, vaddr + size) host pages as a userptr GEM buffer and maps it into the GPU VM
	// at vaddr. `gpu_write` selects a writable or read-only buffer and page mapping. Caller holds
	// m_lock and has already cleared any overlapping external binding.
	bool RegisterExternalUnlocked(uint64_t vaddr, uint64_t size, bool gpu_write) {
		if (m_kfd_userptr) {
			// A KFD userptr buffer at its own guest VA: KFD keeps it valid (it stops its queues while the
			// CPU pages change), so a KFD queue reaches it and a render-node CS need not list it.
			struct kfd_ioctl_alloc_memory_of_gpu_args a {};
			a.va_addr     = vaddr;
			a.size        = size;
			a.mmap_offset = vaddr;
			a.gpu_id      = m_kfd_gpu;
			a.flags       = KFD_IOC_ALLOC_MEM_FLAGS_USERPTR | KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE | KFD_IOC_ALLOC_MEM_FLAGS_NO_SUBSTITUTE |
			          (gpu_write ? KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE : 0u);
			int r = KfdIoctl(AMDKFD_IOC_ALLOC_MEMORY_OF_GPU, &a);
			if (r == 0) {
				struct kfd_ioctl_map_memory_to_gpu_args mp {};
				mp.handle               = a.handle;
				mp.device_ids_array_ptr = reinterpret_cast<uintptr_t>(&m_kfd_gpu);
				mp.n_devices            = 1;
				if ((r = KfdIoctl(AMDKFD_IOC_MAP_MEMORY_TO_GPU, &mp)) != 0) {
					struct kfd_ioctl_free_memory_of_gpu_args fr {};
					fr.handle = a.handle;
					KfdIoctl(AMDKFD_IOC_FREE_MEMORY_OF_GPU, &fr);
				}
			}
			if (r != 0) {
				std::fprintf(stderr, "GemBacking: KFD userptr 0x%llx (0x%llx, %s) failed: %s\n", static_cast<unsigned long long>(vaddr),
				             static_cast<unsigned long long>(size), gpu_write ? "rw" : "ro", std::strerror(-r));
				return false;
			}
			m_external.emplace(vaddr, External {vaddr + size, 0, gpu_write, a.handle});
			m_views_generation++;
			return true;
		}
		struct drm_amdgpu_gem_userptr u;
		std::memset(&u, 0, sizeof(u));
		u.addr  = vaddr;
		u.size  = size;
		// REGISTER tracks the range via mmu-notifier without pinning, like the chunk path; the pages
		// are faulted in (and re-faulted after a CPU write) when a submission validates them.
		u.flags = AMDGPU_GEM_USERPTR_REGISTER | (gpu_write ? 0u : AMDGPU_GEM_USERPTR_READONLY);
		if (Ioctl(DRM_IOCTL_AMDGPU_GEM_USERPTR, &u) != 0) {
			std::fprintf(stderr, "GemBacking: external GEM_USERPTR 0x%llx (0x%llx, %s) failed: %s\n",
			             static_cast<unsigned long long>(vaddr),
			             static_cast<unsigned long long>(size), gpu_write ? "rw" : "ro",
			             std::strerror(errno));
			return false;
		}
		const uint32_t flags =
		    gpu_write ? kRwxPageFlags : (AMDGPU_VM_PAGE_READABLE | AMDGPU_VM_PAGE_EXECUTABLE);
		if (VaOp(AMDGPU_VA_OP_REPLACE, u.handle, vaddr, 0, size, flags) != 0) {
			GemClose(u.handle);
			return false;
		}
		m_external.emplace(vaddr, External {vaddr + size, u.handle, gpu_write});
		m_views_generation++;
		return true;
	}

	// Unmaps and closes every external buffer overlapping [vaddr, vaddr + size), re-registering the
	// portions of a straddling buffer that fall outside the range as their own buffers (closing a GEM
	// handle unmaps all of its VA, so a survivor needs its own handle). Chunk mappings never appear in
	// m_external, so they are never touched. Caller holds m_lock.
	// Calls fn(start, size) for each part of [vaddr, vaddr + size) with no view and no external buffer.
	template <typename Fn>
	void ForEachHoleUnlocked(uint64_t vaddr, uint64_t size, Fn&& fn) const {
		const uint64_t                             end = vaddr + size;
		std::vector<std::pair<uint64_t, uint64_t>> used;
		auto collect = [&](uint64_t start, uint64_t stop) {
			if (stop > vaddr && start < end) {
				used.emplace_back(std::max(start, vaddr), std::min(stop, end));
			}
		};
		for (auto it = m_views.lower_bound(vaddr); it != m_views.begin();) {
			--it;
			collect(it->first, it->first + it->second.size);
			break;
		}
		for (auto it = m_views.lower_bound(vaddr); it != m_views.end() && it->first < end; ++it) {
			collect(it->first, it->first + it->second.size);
		}
		for (auto it = m_external.lower_bound(vaddr); it != m_external.begin();) {
			--it;
			collect(it->first, it->second.end);
			break;
		}
		for (auto it = m_external.lower_bound(vaddr); it != m_external.end() && it->first < end; ++it) {
			collect(it->first, it->second.end);
		}
		std::sort(used.begin(), used.end());
		uint64_t at = vaddr;
		for (const auto& [start, stop]: used) {
			if (start > at) {
				fn(at, start - at);
			}
			at = std::max(at, stop);
		}
		if (at < end) {
			fn(at, end - at);
		}
	}

	// PRT mappings on the holes of [vaddr, vaddr + size) that lie in a PRT aperture.
	void MapPrtHolesUnlocked(uint64_t vaddr, uint64_t size) const {
		for (const auto& [start, len]: m_prt) {
			const uint64_t lo = std::max(start, vaddr);
			const uint64_t hi = std::min(start + len, vaddr + size);
			if (len == 0 || lo >= hi) {
				continue;
			}
			ForEachHoleUnlocked(lo, hi - lo, [&](uint64_t hole, uint64_t hole_size) {
				VaOp(AMDGPU_VA_OP_REPLACE, 0, hole, 0, hole_size, AMDGPU_VM_PAGE_PRT);
			});
		}
	}

	void ClearExternalRangeUnlocked(uint64_t vaddr, uint64_t size) {
		MarkVaStale();
		const uint64_t end = vaddr + size;

		std::vector<std::pair<uint64_t, External>> overlapping;
		auto                                       it = m_external.lower_bound(vaddr);
		if (it != m_external.begin()) {
			auto previous = std::prev(it);
			if (previous->second.end > vaddr) {
				it = previous;
			}
		}
		while (it != m_external.end() && it->first < end) {
			if (it->second.end > vaddr) {
				overlapping.emplace_back(it->first, it->second);
				it = m_external.erase(it);
				m_views_generation++;
			} else {
				++it;
			}
		}

		for (const auto& [start, ext]: overlapping) {
			CloseExternal(start, ext);
			if (start < vaddr) {
				RegisterExternalUnlocked(start, vaddr - start, ext.writable);
			}
			if (end < ext.end) {
				RegisterExternalUnlocked(end, ext.end - end, ext.writable);
			}
		}
	}

	struct External {
		uint64_t end        = 0; // one past the last byte; start is the m_external key
		uint32_t handle     = 0; // the userptr GEM buffer; 0 for a KFD userptr buffer
		bool     writable   = false;
		uint64_t kfd_handle = 0;
	};

	void CloseExternal(uint64_t vaddr, const External& ext) const {
		if (ext.kfd_handle != 0) {
			struct kfd_ioctl_unmap_memory_from_gpu_args um {};
			um.handle               = ext.kfd_handle;
			um.device_ids_array_ptr = reinterpret_cast<uintptr_t>(&m_kfd_gpu);
			um.n_devices            = 1;
			KfdIoctl(AMDKFD_IOC_UNMAP_MEMORY_FROM_GPU, &um);
			struct kfd_ioctl_free_memory_of_gpu_args fr {};
			fr.handle = ext.kfd_handle;
			KfdIoctl(AMDKFD_IOC_FREE_MEMORY_OF_GPU, &fr);
			return;
		}
		VaOp(AMDGPU_VA_OP_CLEAR, 0, vaddr, 0, ext.end - vaddr);
		GemClose(ext.handle);
	}

	int                           m_drm_fd = -1;
	int                           m_mem_fd = -1;
	int                           m_kfd_fd  = -1; // KYTY_NATIVE_GPU_KFD_MEMORY: see InitKfd
	uint32_t                      m_kfd_gpu = 0;
	bool                          m_kfd_userptr = false; // KYTY_NATIVE_GPU_KFD_QUEUES: external ranges are KFD userptr buffers
	bool                          m_kfd_memory  = false; // blocks are KFD buffers (InitKfd)
	Bo                            m_tlb_bo;              // FlushKfdTlb's buffer
	mutable std::mutex            m_handles_mutex;       // m_last_handles
	mutable std::vector<uint32_t> m_last_handles;        // CreatedHandles' last list
	mutable std::atomic<uint64_t> m_va_changes {0};      // VaOp calls (FlushKfdTlb)
	mutable uint64_t              m_kfd_va_next = 0x600000000000ull; // private VAs of the KFD allocations
	uint8_t*                      m_alias  = nullptr;
	uint64_t                      m_size   = 0;
	uint64_t                      m_resident_bytes = 0;
	uint64_t                      m_touched_bytes  = 0;
	std::vector<Block>            m_blocks;
	std::map<uint64_t, View>      m_views;     // guest VA -> view of the backing
	std::map<uint64_t, uint64_t>  m_coherent;  // backing ranges of coherent memory types: start -> end
	std::atomic<uint64_t>         m_views_generation {0}; // changed under m_lock, read without it
	Bo                            m_dummy;     // GPU target of every non-resident block
	uint8_t*                      m_dummy_cpu = nullptr;
	static inline std::atomic<uint32_t> s_promoting {0}; // blocks being moved right now
	std::thread                   m_mover;      // runs MoveBlock (see there)
	std::mutex                    m_mover_lock;
	std::condition_variable       m_mover_cv;
	MoveJob*                      m_mover_job  = nullptr;
	bool                          m_mover_stop = false;
	std::map<uint64_t, External>  m_external; // guest VA -> userptr buffer for non-direct ranges
	std::array<std::pair<uint64_t, uint64_t>, 3> m_prt {}; // PRT apertures (start, size), SetPrtAperture
	std::atomic<bool>             m_map_overflow {false}; // a block with GPU access stayed out of map-time residency
	mutable std::mutex            m_lock;     // guards blocks, views, m_external and the dummy
};

} // namespace Libs::Graphics::NativeGpu

#endif // KYTY_GEM_BACKING_SUPPORTED

#endif // EMULATOR_SRC_GRAPHICS_NATIVE_GPU_GEMBACKING_H_
