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
#include <cstdlib>
#include <algorithm>
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
#include <unistd.h>

#include <drm/amdgpu_drm.h>
#include <drm/drm.h>

namespace Libs::Graphics::NativeGpu {

// Backs guest direct memory so every guest direct-memory mapping is also mapped into the GPU VM
// at the same virtual address, on /dev/dri/renderD128.
//
// The CPU side starts as a memfd, like the upstream backing: the guest views and the contiguous alias
// all map it, and its pages are allocated on first touch. The GPU side works in blocks of the backing
// (BlockSize()). A resident block is a cached GTT GEM buffer that the block's views, its alias slot
// and its GPU mappings all map instead of the memfd. Promotion creates the buffer, write-protects the
// block's views and alias slot, copies its data into the buffer, maps them onto the buffer with their
// protection restored, punches the block out of the memfd and rebinds the GPU mappings. A guest write
// during the move faults; the fault hook (WaitIfPromoting, from Memory::HandleGpuFault) waits for the
// move and retries it, so no write is lost and no read sees a stale copy. Blocks are never demoted.
// Resident blocks are TTM mappings (VM_PFNMAP | VM_IO): get_user_pages does not work on them, which
// only matters for process_vm_readv (Kyty uses it for crash-report stacks).
//
// Two models decide when a block becomes resident (KYTY_NATIVE_GPU_MEM, see ModelFromEnv):
//
// Exact (default): a block becomes resident when the guest maps any of it with GPU access
// (MapGpuVisible, called from memory.cpp), so the GPU sees every byte the PS5 GPU could see and
// nothing has to be predicted; mappings without GPU access stay CPU-only in the memfd. Block buffers
// are VM_ALWAYS_VALID: the kernel keeps them valid with the VM and a submission lists none of them
// (listing 600 buffers costs ~1.3 ms per CS ioctl, none ~8 us). Blocks are 256 MiB so the GPU page
// tables can use 2 MiB fragments. An always-valid buffer cannot be exported as a dma-buf, and a DRM
// mmap only maps a buffer from its first page, so a CPU view at a page offset inside a block maps
// [0, offset + length) and mremaps the tail into place (MapBo). Display buffers, which Kyty's
// presenter imports as dma-bufs, get an exportable buffer of their own (ExportRange); those few are
// listed in each submission. TTM populates a GTT buffer whole when it is created and every
// always-valid buffer must stay in GTT, so this costs RAM for all GPU-visible memory, rounded to
// blocks; past an optional budget (KYTY_NATIVE_GPU_MEM_BUDGET_MIB) further GPU-visible blocks are
// deferred to lazy residency below.
//
// Lazy (KYTY_NATIVE_GPU_MEM=lazy, the earlier model): 16 MiB blocks become resident when the CPU has
// written to them, which the memfd shows as data (PromoteTouched, via SEEK_DATA), or when a command
// buffer about to run references them (PromoteGuestRange: render targets, index buffers, pointers in
// shader user data, as the PM4 translator reports them). Every resident block is listed in every
// submission and exported as a dma-buf for its CPU views (a dma-buf mmap takes any page offset).
// GPU mappings of blocks that are not resident point at one shared dummy buffer, so an access the
// translator did not foresee reads garbage or loses its write instead of faulting: on this board a GPU
// page fault hangs the ring and can take the whole board down, and so do PRT mappings.
// DummyWritten() reports when the GPU wrote there. The exact model only falls back to the dummy when
// a block buffer cannot be created.
//
// Simpler designs that do not work here (tools/bc250-native-probe): userptr buffers over the memfd
// cost a page-table walk of their whole size on every submission (~31 ms per frame for 13.5 GiB);
// udmabuf imports of the memfd are not snooped by the GPU, so CPU writes still in the cache stay
// invisible to it.
//
// Guest memory the GPU can touch also lives outside the direct-memory backing: flexible, program
// (code), runtime and stack ranges are committed anonymous host pages at guest VA == host VA, not
// slices of the chunks. BindExternal maps such a range into the same GPU VM at its guest VA, so
// shader code and other non-direct guest memory is resident on the GPU; UnbindExternal removes it.
// The lazy model registers the range's own host pages as a userptr GEM buffer, which every
// submission lists (CreatedHandles) and which costs a page-table walk per submit in proportion to
// its size. The exact model moves the range into an always-valid buffer instead.
//
// This class performs no command submission.
class GemBacking {
public:
	// Residency granularity of the lazy model. Every resident block is listed in every submission (a
	// listed buffer that is in place costs about two microseconds), so smaller blocks waste less memory
	// but make longer lists. 16 MiB divides the 13.5 GiB backing evenly. Also the dummy's size.
	static constexpr uint64_t kBlockSize = 16ull * 1024 * 1024;
	// Block size of the exact model: big enough for 2 MiB GPU page fragments, small enough that
	// rounding mapped ranges out to blocks wastes little.
	static constexpr uint64_t kExactBlockSize = 256ull * 1024 * 1024;

	enum class Model {
		Exact, // every mapped block resident, always-valid buffers, no per-submission list
		Lazy,  // blocks resident on CPU write or GPU reference, listed in every submission
	};

	// KYTY_NATIVE_GPU_MEM=lazy selects the lazy model; anything else (or unset) the exact one.
	static Model ModelFromEnv() {
		const char* m = std::getenv("KYTY_NATIVE_GPU_MEM");
		return m != nullptr && std::strcmp(m, "lazy") == 0 ? Model::Lazy : Model::Exact;
	}

	// Creates the backing for `size` bytes: opens the device, creates the memfd, its contiguous
	// alias and the dummy GPU buffer. Returns nullptr on failure. No block is resident yet.
	// `block_size` 0 picks the model's default (KYTY_NATIVE_GPU_MEM_BLOCK_MIB overrides the exact
	// model's); it must be a multiple of 64 KiB.
	static GemBacking* Create(uint64_t size, const char* device = "/dev/dri/renderD128",
	                          Model model = Model::Lazy, uint64_t block_size = 0) {
		if (size == 0) {
			return nullptr;
		}
		if (block_size == 0) {
			block_size = model == Model::Exact ? kExactBlockSize : kBlockSize;
			if (const char* b = std::getenv("KYTY_NATIVE_GPU_MEM_BLOCK_MIB"); b != nullptr && model == Model::Exact) {
				block_size = std::max<uint64_t>(std::strtoull(b, nullptr, 0), 1) << 20u;
			}
		}
		if (block_size % 0x10000 != 0) {
			std::fprintf(stderr, "GemBacking: block size 0x%llx is not a multiple of 64 KiB\n",
			             static_cast<unsigned long long>(block_size));
			return nullptr;
		}
		auto* self         = new GemBacking();
		self->m_size       = size;
		self->m_exact      = model == Model::Exact;
		self->m_block_size = block_size;
		if (const char* b = std::getenv("KYTY_NATIVE_GPU_MEM_BUDGET_MIB"); b != nullptr && model == Model::Exact) {
			self->m_budget = std::strtoull(b, nullptr, 0) << 20u;
		}
		self->m_drm_fd     = ::open(device, O_RDWR | O_CLOEXEC);
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
		const auto count = (size + block_size - 1) / block_size;
		self->m_blocks.resize(count);
		for (uint64_t i = 0; i < count; i++) {
			const uint64_t start    = i * block_size;
			self->m_blocks[i].size = (size - start < block_size) ? size - start : block_size;
		}
		void* dummy_cpu = MAP_FAILED;
		if (self->CreateBo(kBlockSize, &self->m_dummy, false)) {
			dummy_cpu = self->MapWhole(self->m_dummy);
		}
		if (dummy_cpu == MAP_FAILED) {
			std::fprintf(stderr, "GemBacking: dummy buffer setup failed\n");
			delete self;
			return nullptr;
		}
		self->m_dummy_cpu = static_cast<uint8_t*>(dummy_cpu);
		std::fprintf(stderr, "GemBacking: %s model, %llu MiB blocks over %llu MiB of guest memory\n",
		             self->m_exact ? "exact" : "lazy", static_cast<unsigned long long>(block_size >> 20u),
		             static_cast<unsigned long long>(size >> 20u));
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
			VaOp(AMDGPU_VA_OP_CLEAR, 0, vaddr, 0, ext.end - vaddr);
			if (!ext.always_valid || --m_external_refs[ext.handle] == 0) {
				GemClose(ext.handle);
			}
		}
		if (m_self_mem_fd >= 0) {
			::close(m_self_mem_fd);
		}
		if (m_alias != nullptr) {
			::munmap(m_alias, m_size);
		}
		for (auto& block: m_blocks) {
			if (block.resident) {
				CloseBo(block.bo);
			}
		}
		for (auto& [start, exp]: m_exports) {
			CloseBo(exp.bo);
		}
		if (m_dummy_cpu != nullptr) {
			::munmap(m_dummy_cpu, kBlockSize);
		}
		CloseBo(m_dummy);
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

	[[nodiscard]] bool Exact() const { return m_exact; }
	[[nodiscard]] uint64_t BlockSize() const { return m_block_size; }

	// Maps the backing bytes [offset, offset + size) at guest address `vaddr` with protection
	// `prot` (PROT_* flags): the memfd for blocks that are not resident, the block's buffer for
	// resident ones. On success the region is mapped at exactly `vaddr` and recorded as a view, so a
	// later promotion of its blocks can move it.
	bool MapView(uint64_t vaddr, uint64_t size, uint64_t offset, int prot) {
		if (size == 0 || !EnsureRange(offset, size)) {
			return false;
		}
		std::lock_guard<std::mutex> lock(m_lock);
		bool ok = ForEachPiece(offset, size, [&](const Piece& piece, uint64_t done) {
			void* const dst = reinterpret_cast<void*>(vaddr + done);
			const bool  mapped =
			    piece.bo != nullptr
			        ? MapBo(*piece.bo, piece.inner, piece.len, prot, dst)
			        : ::mmap(dst, piece.len, prot, MAP_SHARED | MAP_FIXED, m_mem_fd, static_cast<off_t>(offset + done)) == dst;
			if (!mapped) {
				std::fprintf(stderr, "GemBacking: view mmap at 0x%llx (+0x%llx, block %llu) failed: %s\n",
				             static_cast<unsigned long long>(vaddr + done), static_cast<unsigned long long>(piece.len),
				             static_cast<unsigned long long>(piece.block), std::strerror(errno));
			}
			return mapped;
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
		const bool ok = ForEachPiece(offset, size, [&](const Piece& piece, uint64_t done) {
			if (piece.bo == nullptr) {
				return MapDummyUnlocked(vaddr + done, piece.len);
			}
			return VaOp(AMDGPU_VA_OP_REPLACE, piece.bo->handle, vaddr + done, piece.inner, piece.len) == 0;
		});
		auto it = m_views.find(vaddr);
		if (it != m_views.end() && it->second.size == size && it->second.offset == offset) {
			it->second.gpu = true;
		} else {
			EraseViewsUnlocked(vaddr, size);
			m_views[vaddr] = View {size, offset, true};
			m_views_generation++;
		}
		return ok;
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
		return VaOp(AMDGPU_VA_OP_CLEAR, 0, vaddr, 0, size) == 0;
	}

	// Makes every block the CPU has written to resident. Promoted blocks are punched out of the
	// memfd, so SEEK_DATA only ever finds blocks that are not resident yet. Returns how many blocks
	// were promoted. The exact model only promotes blocks deferred past its budget (MapGpuVisible):
	// the others are resident already or CPU-only.
	uint32_t PromoteTouched() {
		std::lock_guard<std::mutex> lock(m_lock);
		if (m_exact && m_deferred_blocks == 0) {
			return 0;
		}
		uint32_t promoted = 0;
		uint64_t pos      = 0;
		while (pos < m_size) {
			const off_t data = ::lseek(m_mem_fd, static_cast<off_t>(pos), SEEK_DATA);
			if (data < 0) {
				break;
			}
			const uint64_t index = static_cast<uint64_t>(data) / m_block_size;
			if (!m_blocks[index].resident && (!m_exact || m_blocks[index].deferred) && PromoteBlockUnlocked(index)) {
				promoted++;
			}
			pos = (index + 1) * m_block_size;
		}
		return promoted;
	}

	// [vaddr, vaddr + size) as pieces of exportable buffers, in guest order, for consumers that import
	// the buffers' dma-bufs (zero-copy presentation): resident lazy-model blocks and ExportRange's
	// buffers. False unless every byte is in one.
	struct DmabufPiece {
		int      fd;        // the buffer's dma-buf
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
			ok = ForEachPiece(backing, len, [&](const Piece& piece, uint64_t) {
				if (piece.bo == nullptr || piece.bo->dmabuf_fd < 0) {
					return false;
				}
				out.push_back(DmabufPiece {piece.bo->dmabuf_fd, piece.bo->size, piece.inner, piece.len});
				return true;
			});
			covered += len;
			next += len;
		});
		return ok && covered == size;
	}

	// Gives guest range [vaddr, vaddr + size) (one view, resident) a buffer of its own that can be
	// exported as a dma-buf, for Kyty's presenter to import a display buffer: the exact model's
	// always-valid blocks cannot be exported. The range moves exactly like a block promotion: its CPU
	// mappings are write-protected, its bytes copied, then the CPU and GPU mappings point at the new
	// buffer. Its bytes in the block buffer are left unused. The caller makes sure the GPU is not
	// using the range meanwhile. A no-op in the lazy model (blocks are exportable) and for a range
	// already exported. Exported buffers are listed in every submission (CreatedHandles).
	bool ExportRange(uint64_t vaddr, uint64_t size) {
		if (!m_exact || size == 0) {
			return true;
		}
		std::lock_guard<std::mutex> lock(m_lock);
		auto view = FindViewUnlocked(vaddr);
		if (view == m_views.end() || vaddr + size > view->first + view->second.size) {
			return false;
		}
		const uint64_t backing = view->second.offset + (vaddr - view->first);
		const uint64_t start   = backing & ~0xfffull;
		const uint64_t end     = (backing + size + 0xfff) & ~0xfffull;
		if (auto it = m_exports.find(start); it != m_exports.end() && it->second.end == end) {
			return true;
		}
		const bool ok = ForEachPiece(start, end - start, [&](const Piece& piece, uint64_t) {
			return piece.bo != nullptr && !piece.exported;
		});
		if (!ok) {
			std::fprintf(stderr, "GemBacking: cannot export 0x%llx (+0x%llx): not resident or overlaps an export\n",
			             static_cast<unsigned long long>(vaddr), static_cast<unsigned long long>(size));
			return false;
		}
		Bo bo;
		if (!CreateBo(end - start, &bo, false)) {
			return false;
		}
		if (!MoveRangeUnlocked(start, end - start, bo, m_alias + start)) {
			CloseBo(bo);
			return false;
		}
		m_exports.emplace(start, Export {end, bo});
		std::fprintf(stderr, "GemBacking: guest 0x%llx (%llu KiB) has its own exportable buffer\n",
		             static_cast<unsigned long long>(vaddr), static_cast<unsigned long long>((end - start) >> 10u));
		return true;
	}

	// Exact model: the guest mapped [vaddr, vaddr + size) with GPU access (memory.cpp calls this when
	// such a mapping is made or gains GPU access), so its blocks become resident. Past the residency
	// budget (KYTY_NATIVE_GPU_MEM_BUDGET_MIB) a block is deferred instead: it becomes resident the lazy
	// way, when the CPU writes it or a command buffer references it, and the GPU sees the dummy there
	// until then. Returns how many blocks were promoted.
	uint32_t MapGpuVisible(uint64_t vaddr, uint64_t size) {
		std::lock_guard<std::mutex> lock(m_lock);
		uint32_t promoted = 0;
		ForEachViewOverlap(vaddr, size, [&](uint64_t, uint64_t, uint64_t backing, uint64_t len) {
			for (uint64_t index = backing / m_block_size; index <= (backing + len - 1) / m_block_size; index++) {
				Block& block = m_blocks[index];
				if (block.resident || block.failed || block.deferred) {
					continue;
				}
				if (m_budget != 0 && m_resident_bytes + block.size > m_budget) {
					block.deferred = true;
					if (m_deferred_blocks++ == 0) {
						std::fprintf(stderr, "GemBacking: residency budget of %llu MiB reached; further GPU-visible blocks "
						             "become resident when used\n", static_cast<unsigned long long>(m_budget >> 20u));
					}
					continue;
				}
				if (PromoteBlockUnlocked(index)) {
					promoted++;
				} else {
					// Out of GTT or RAM: the GPU keeps seeing the dummy there. Not retried.
					block.failed = true;
					std::fprintf(stderr, "GemBacking: block %llu could not be made resident; the GPU sees the dummy there\n",
					             static_cast<unsigned long long>(index));
				}
			}
		});
		return promoted;
	}

	// True when residency still has to follow CPU writes and command-buffer references
	// (PromoteTouched, PromoteGuestRange): always in the lazy model, in the exact one once blocks
	// were deferred past the budget.
	[[nodiscard]] bool LazyResidency() const {
		std::lock_guard<std::mutex> lock(m_lock);
		return !m_exact || m_deferred_blocks != 0;
	}

	// Makes the blocks behind guest addresses [vaddr, vaddr + size) resident, through whichever views
	// cover them; addresses outside every view are ignored. Returns how many blocks were promoted.
	uint32_t PromoteGuestRange(uint64_t vaddr, uint64_t size) {
		std::lock_guard<std::mutex> lock(m_lock);
		uint32_t promoted = 0;
		ForEachViewOverlap(vaddr, size, [&](uint64_t, uint64_t, uint64_t backing, uint64_t len) {
			for (uint64_t index = backing / m_block_size; index <= (backing + len - 1) / m_block_size; index++) {
				if (!m_blocks[index].resident && PromoteBlockUnlocked(index)) {
					promoted++;
				}
			}
		});
		return promoted;
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

	// True (once) when the GPU has written into the dummy buffer since the last call, i.e. a
	// submission wrote to guest memory that was not resident; the dummy is cleared again. Call it
	// only while the GPU is idle.
	bool DummyWritten() {
		std::lock_guard<std::mutex> lock(m_lock);
		static const uint8_t kZero[65536] = {};
		bool written = false;
		for (uint64_t at = 0; at < kBlockSize && !written; at += sizeof(kZero)) {
			written = std::memcmp(m_dummy_cpu + at, kZero, sizeof(kZero)) != 0;
		}
		if (written) {
			std::memset(m_dummy_cpu, 0, kBlockSize);
		}
		return written;
	}

	// Bumped whenever the set of views changes; a caller caching ViewRanges() refreshes on a change.
	[[nodiscard]] uint64_t ViewsGeneration() const {
		std::lock_guard<std::mutex> lock(m_lock);
		return m_views_generation;
	}

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

	[[nodiscard]] uint64_t ResidentBytes() const {
		std::lock_guard<std::mutex> lock(m_lock);
		return m_resident_bytes;
	}

	// Backing bytes some guest view maps (aliases counted once), each mapped range rounded out to
	// `granularity`: what backing every mapped byte would cost at that buffer size.
	[[nodiscard]] uint64_t MappedBytes(uint64_t granularity) const {
		std::lock_guard<std::mutex>                lock(m_lock);
		std::vector<std::pair<uint64_t, uint64_t>> ranges;
		ranges.reserve(m_views.size());
		for (const auto& [vaddr, view]: m_views) {
			ranges.emplace_back(view.offset / granularity * granularity,
			                    (view.offset + view.size + granularity - 1) / granularity * granularity);
		}
		std::sort(ranges.begin(), ranges.end());
		uint64_t total = 0;
		uint64_t end   = 0;
		for (const auto& [s, e]: ranges) {
			const uint64_t from = std::max(s, end);
			if (e > from) {
				total += e - from;
				end = e;
			}
		}
		return total;
	}

	// Bytes of the non-direct guest ranges bound with BindExternal.
	[[nodiscard]] uint64_t ExternalBytes() const {
		std::lock_guard<std::mutex> lock(m_lock);
		uint64_t                    total = 0;
		for (const auto& [vaddr, ext]: m_external) {
			total += ext.end - vaddr;
		}
		return total;
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
	//
	// The exact model instead moves the range into an always-valid buffer, like a block promotion: the
	// host pages are copied into it and the range's mappings replaced by views of it, so no
	// submission lists it and no protection change needs a new registration (a protection change
	// calls this again; a range already moved only gets its GPU write permission updated).
	bool BindExternal(uint64_t vaddr, uint64_t size, bool gpu_write) {
		if (vaddr == 0 || size == 0) {
			return false;
		}
		std::lock_guard<std::mutex> lock(m_lock);
		if (m_exact) {
			if (UpdateAlwaysValidExternalUnlocked(vaddr, size, gpu_write)) {
				return true;
			}
			ClearExternalRangeUnlocked(vaddr, size);
			if (MoveExternalUnlocked(vaddr, size, gpu_write)) {
				return true;
			}
			// No buffer for it (out of GTT or RAM): fall back to userptr registration.
		} else {
			ClearExternalRangeUnlocked(vaddr, size);
		}
		// A single guest range can hold sub-ranges the guest reprotected to execute-only or no-access
		// (seen in program memory: the shader is a readable slice, with an execute-only tail). A
		// userptr cannot fault in an unreadable page, which fails the whole submission, so register one
		// buffer per readable slice and skip the holes.
		bool any = false;
		ForEachReadableSubrange(vaddr, size, [&](uint64_t s, uint64_t sz, bool writable) {
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
		return true;
	}

	// The DRM file whose GPU VM holds the bindings; command submission must use the same file.
	[[nodiscard]] int DrmFd() const { return m_drm_fd; }

	// Every GEM buffer a submission must list as resident: the external ranges, the exported ranges,
	// the dummy and, in the lazy model, the resident blocks (the exact model's blocks are always valid
	// and need no listing; its dummy only once something maps it). Grows as blocks are promoted.
	[[nodiscard]] std::vector<uint32_t> CreatedHandles() const {
		std::lock_guard<std::mutex> lock(m_lock);
		std::vector<uint32_t>       handles;
		handles.reserve((m_exact ? 0 : m_blocks.size()) + m_external.size() + m_exports.size() + 1);
		for (const auto& block: m_blocks) {
			if (block.resident && !block.bo.always_valid) {
				handles.push_back(block.bo.handle);
			}
		}
		if (!m_exact || m_dummy_mapped) {
			handles.push_back(m_dummy.handle);
		}
		for (const auto& [vaddr, ext]: m_external) {
			if (!ext.always_valid) {
				handles.push_back(ext.handle);
			}
		}
		for (const auto& [start, exp]: m_exports) {
			handles.push_back(exp.bo.handle);
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
		const uint64_t              index = offset / m_block_size;
		if (!m_blocks[index].resident) {
			return false;
		}
		if (handle != nullptr) {
			*handle = m_blocks[index].bo.handle;
		}
		if (offset_in_block != nullptr) {
			*offset_in_block = offset - index * m_block_size;
		}
		return true;
	}

private:
	struct Bo {
		uint32_t handle       = 0;
		int      dmabuf_fd    = -1;    // exportable buffers: the dma-buf CPU views map
		uint64_t mmap_offset  = 0;     // always-valid buffers: the DRM mmap offset CPU views map
		uint64_t size         = 0;
		bool     always_valid = false;
	};
	struct Block {
		Bo       bo;
		uint64_t size     = 0;
		bool     resident = false;
		bool     failed   = false; // the exact model could not make it resident (logged once)
		bool     deferred = false; // exact model: GPU-visible, but past the budget (MapGpuVisible)
	};
	// A backing range [key, end) with an exportable buffer of its own (ExportRange).
	struct Export {
		uint64_t end = 0;
		Bo       bo;
	};
	// What backs one stretch of the backing: a buffer and the offset in it, or (bo == nullptr) the
	// memfd on the CPU side and the dummy on the GPU side.
	struct Piece {
		const Bo* bo       = nullptr;
		uint64_t  inner    = 0;
		uint64_t  len      = 0;
		uint64_t  block    = 0;
		bool      exported = false;
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

	static constexpr uint32_t kRwxPageFlags =
	    AMDGPU_VM_PAGE_READABLE | AMDGPU_VM_PAGE_WRITEABLE | AMDGPU_VM_PAGE_EXECUTABLE;

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

	// A cached GTT buffer (TTM zeroes it). Always-valid ones (the exact model's blocks) are mapped by
	// the CPU through the DRM file, which needs their mmap offset; the others are exported as a
	// dma-buf, which CPU views map and which Kyty's presenter can import.
	bool CreateBo(uint64_t size, Bo* out, bool always_valid) const {
		union drm_amdgpu_gem_create c;
		std::memset(&c, 0, sizeof(c));
		c.in.bo_size      = size;
		c.in.alignment    = 2ull * 1024 * 1024;
		c.in.domains      = AMDGPU_GEM_DOMAIN_GTT;
		c.in.domain_flags = always_valid ? AMDGPU_GEM_CREATE_VM_ALWAYS_VALID : 0;
		int r             = Ioctl(DRM_IOCTL_AMDGPU_GEM_CREATE, &c);
		if (r != 0) {
			std::fprintf(stderr, "GemBacking: GEM_CREATE of %llu MiB failed: %s\n",
			             static_cast<unsigned long long>(size / (1024 * 1024)), std::strerror(-r));
			return false;
		}
		out->size         = size;
		out->always_valid = always_valid;
		if (always_valid) {
			union drm_amdgpu_gem_mmap m;
			std::memset(&m, 0, sizeof(m));
			m.in.handle = c.out.handle;
			r           = Ioctl(DRM_IOCTL_AMDGPU_GEM_MMAP, &m);
			if (r != 0) {
				std::fprintf(stderr, "GemBacking: GEM_MMAP failed: %s\n", std::strerror(-r));
				GemClose(c.out.handle);
				return false;
			}
			out->handle      = c.out.handle;
			out->mmap_offset = m.out.addr_ptr;
			return true;
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

	// Maps buffer bytes [inner, inner + len) at exactly `dst` with protection `prot`. A dma-buf maps
	// at any page offset. A DRM file mapping always starts at the buffer's first page (a DRM mmap at
	// the buffer's offset + inner is EINVAL), so map [0, inner + len) anywhere and move its tail into
	// place with mremap, which keeps the page offset.
	bool MapBo(const Bo& bo, uint64_t inner, uint64_t len, int prot, void* dst) const {
		if (bo.dmabuf_fd >= 0) {
			return ::mmap(dst, len, prot, MAP_SHARED | MAP_FIXED, bo.dmabuf_fd, static_cast<off_t>(inner)) == dst;
		}
		if (inner == 0) {
			return ::mmap(dst, len, prot, MAP_SHARED | MAP_FIXED, m_drm_fd, static_cast<off_t>(bo.mmap_offset)) == dst;
		}
		auto* t = static_cast<uint8_t*>(
		    ::mmap(nullptr, inner + len, prot, MAP_SHARED, m_drm_fd, static_cast<off_t>(bo.mmap_offset)));
		if (t == MAP_FAILED) {
			return false;
		}
		const bool moved = ::mremap(t + inner, len, len, MREMAP_MAYMOVE | MREMAP_FIXED, dst) == dst;
		::munmap(t, moved ? inner : inner + len);
		return moved;
	}

	// The whole buffer, read-write, anywhere; MAP_FAILED on failure.
	void* MapWhole(const Bo& bo) const {
		return bo.dmabuf_fd >= 0
		           ? ::mmap(nullptr, bo.size, PROT_READ | PROT_WRITE, MAP_SHARED, bo.dmabuf_fd, 0)
		           : ::mmap(nullptr, bo.size, PROT_READ | PROT_WRITE, MAP_SHARED, m_drm_fd, static_cast<off_t>(bo.mmap_offset));
	}

	// Points the GPU mapping [vaddr, vaddr + size) at the dummy, kBlockSize at a time.
	bool MapDummyUnlocked(uint64_t vaddr, uint64_t size) {
		m_dummy_mapped = true;
		bool ok        = true;
		for (uint64_t done = 0; done < size;) {
			const uint64_t part = std::min(size - done, kBlockSize - (vaddr + done) % kBlockSize);
			ok   = VaOp(AMDGPU_VA_OP_REPLACE, m_dummy.handle, vaddr + done, (vaddr + done) % kBlockSize, part) == 0 && ok;
			done += part;
		}
		return ok;
	}

	void CloseBo(Bo& bo) const {
		if (bo.dmabuf_fd >= 0) {
			::close(bo.dmabuf_fd);
		}
		if (bo.handle != 0) {
			GemClose(bo.handle);
		}
		bo = Bo {};
	}

	// Calls cb(piece, offset from `offset`) for each stretch of backing range [offset, offset + size)
	// with one backing (Piece), split at block and export boundaries; stops at the first false.
	template <typename Cb>
	bool ForEachPiece(uint64_t offset, uint64_t size, Cb&& cb) const {
		uint64_t done = 0;
		while (done < size) {
			const uint64_t at    = offset + done;
			const uint64_t index = at / m_block_size;
			const Block&   block = m_blocks[index];
			Piece          piece;
			piece.block = index;
			piece.inner = at - index * m_block_size;
			piece.len   = std::min(size - done, block.size - piece.inner);
			piece.bo    = block.resident ? &block.bo : nullptr;
			auto exp    = m_exports.upper_bound(at);
			if (exp != m_exports.begin() && at < std::prev(exp)->second.end) {
				--exp;
				piece.bo       = &exp->second.bo;
				piece.inner    = at - exp->first;
				piece.len      = std::min(size - done, exp->second.end - at);
				piece.exported = true;
			} else if (exp != m_exports.end() && exp->first < at + piece.len) {
				piece.len = exp->first - at; // stop where the next export starts
			}
			if (!cb(static_cast<const Piece&>(piece), done)) {
				return false;
			}
			done += piece.len;
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

	// Calls cb(start, size, prot) for each host mapping overlapping [vaddr, vaddr + size), from
	// /proc/self/maps, so a remap can keep the protection the guest (or Kyty's write tracking) set.
	// If maps cannot be read the whole range is reported read-write.
	template <typename Cb>
	static void ForEachHostProtection(uint64_t vaddr, uint64_t size, Cb&& cb) {
		const uint64_t end  = vaddr + size;
		FILE*          maps = ::fopen("/proc/self/maps", "re");
		if (maps == nullptr) {
			cb(vaddr, size, PROT_READ | PROT_WRITE);
			return;
		}
		char line[512];
		while (std::fgets(line, sizeof(line), maps) != nullptr) {
			unsigned long long s = 0, e = 0;
			char               perms[8] = {0};
			if (std::sscanf(line, "%llx-%llx %7s", &s, &e, perms) != 3 || e <= vaddr || s >= end) {
				continue;
			}
			const int prot = (perms[0] == 'r' ? PROT_READ : 0) | (perms[1] == 'w' ? PROT_WRITE : 0) |
			                 (perms[2] == 'x' ? PROT_EXEC : 0);
			const uint64_t cs = std::max<uint64_t>(s, vaddr);
			const uint64_t ce = std::min<uint64_t>(e, end);
			cb(cs, ce - cs, prot);
		}
		::fclose(maps);
	}

	// Calls cb(offset from `start`, length) for each data extent of backing range [start, end) in the
	// memfd.
	template <typename Cb>
	void ForEachDataExtent(uint64_t start, uint64_t end, Cb&& cb) const {
		uint64_t pos = start;
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
		uint64_t inner; // offset in the range being moved
	};
	struct MoveJob {
		uint64_t               start  = 0;       // backing offset (or, for an external range, address) of the range
		uint64_t               size   = 0;
		const uint8_t*         src    = nullptr; // copy from here, or (nullptr) the memfd's data extents
		int                    src_fd = -1;      // or read from this file at `start` (/proc/self/mem)
		uint8_t*               bo_cpu = nullptr;
		Bo                     bo;               // the buffer the range moves into
		std::vector<MovePiece> pieces;
		bool                   moved  = false;
		bool                   done   = false;
	};

	// The move proper: write-protect every CPU mapping of the range, copy its data into the buffer,
	// map everything onto the buffer with its protection back. A guest write meanwhile faults and
	// waits in WaitIfPromoting. It runs on the mover thread, whose stack is ordinary host memory: a
	// guest thread's stack can live in the very block being moved (games allocate thread stacks in
	// direct memory, and Kyty runs on guest stacks), and a thread that write-protects its own stack
	// faults on its next push and waits for itself forever.
	void MoveBlock(MoveJob* job) const {
		s_promoting.fetch_add(1, std::memory_order_acq_rel);
		for (const auto& p: job->pieces) {
			if ((p.prot & PROT_WRITE) != 0) {
				::mprotect(reinterpret_cast<void*>(p.addr), p.len, p.prot & ~PROT_WRITE);
			}
		}
		if (job->src != nullptr) {
			std::memcpy(job->bo_cpu, job->src, job->size);
		} else if (job->src_fd >= 0) {
			uint64_t done = 0;
			while (done < job->size) {
				const ssize_t n = ::pread(job->src_fd, job->bo_cpu + done, job->size - done,
				                          static_cast<off_t>(job->start + done));
				if (n <= 0) {
					break;
				}
				done += static_cast<uint64_t>(n);
			}
		} else {
			// pread rather than a mapping of the memfd, which would allocate pages for the holes; holes
			// stay zero in the buffer.
			ForEachDataExtent(job->start, job->start + job->size, [&](uint64_t at, uint64_t len) {
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
		}
		bool moved = true;
		for (const auto& p: job->pieces) {
			moved = MapBo(job->bo, p.inner, p.len, p.prot, reinterpret_cast<void*>(p.addr)) && moved;
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

	// Moves backing range [start, start + size) into `bo`: its alias slot and every guest view piece
	// (split at protection boundaries) are remapped onto the buffer with their protection, after its
	// bytes are copied from `src` (nullptr: the memfd's data extents), and the GPU mappings of its
	// views are pointed at the buffer.
	bool MoveRangeUnlocked(uint64_t start, uint64_t size, const Bo& bo, const uint8_t* src) {
		auto* bo_cpu = static_cast<uint8_t*>(MapWhole(bo));
		if (bo_cpu == MAP_FAILED) {
			std::fprintf(stderr, "GemBacking: CPU map of a %llu KiB buffer failed: %s\n",
			             static_cast<unsigned long long>(size >> 10u), std::strerror(errno));
			return false;
		}
		// Every CPU mapping of the range: (address, length, protection, offset in range), the alias slot
		// first, then each guest view piece split at protection boundaries. Kept in a heap job: the
		// move runs on the mover thread and must not write to this thread's stack (see MoveBlock).
		auto job    = std::make_unique<MoveJob>();
		job->start  = start;
		job->size   = size;
		job->src    = src;
		job->bo_cpu = bo_cpu;
		job->bo     = bo;
		std::vector<std::pair<uint64_t, uint64_t>> gpu_pieces; // (guest address, offset in range)
		std::vector<uint64_t>                      gpu_lengths;
		job->pieces.push_back(MovePiece {reinterpret_cast<uint64_t>(m_alias + start), size, PROT_READ | PROT_WRITE, 0});
		for (const auto& [view_addr, view]: m_views) {
			const uint64_t vs = std::max(view.offset, start);
			const uint64_t ve = std::min(view.offset + view.size, start + size);
			if (vs >= ve) {
				continue;
			}
			const uint64_t guest = view_addr + (vs - view.offset);
			ForEachHostProtection(guest, ve - vs, [&](uint64_t s, uint64_t len, int prot) {
				job->pieces.push_back(MovePiece {s, len, prot, vs - start + (s - guest)});
			});
			if (view.gpu) {
				gpu_pieces.emplace_back(guest, vs - start);
				gpu_lengths.push_back(ve - vs);
			}
		}
		RunOnMover(job.get());
		::munmap(bo_cpu, bo.size);
		if (!job->moved) {
			std::fprintf(stderr, "GemBacking: remapping backing 0x%llx (+0x%llx) failed: %s\n",
			             static_cast<unsigned long long>(start), static_cast<unsigned long long>(size),
			             std::strerror(errno));
		}
		for (size_t i = 0; i < gpu_pieces.size(); i++) {
			VaOp(AMDGPU_VA_OP_REPLACE, bo.handle, gpu_pieces[i].first, gpu_pieces[i].second, gpu_lengths[i]);
		}
		return job->moved;
	}

	bool PromoteBlockUnlocked(uint64_t index) {
		Block&         block = m_blocks[index];
		const uint64_t start = index * m_block_size;
		Bo             bo;
		if (!CreateBo(block.size, &bo, m_exact)) {
			return false;
		}
		uint64_t touched = 0; // bytes the CPU had written, the rest of the block is padding
		ForEachDataExtent(start, start + block.size, [&](uint64_t, uint64_t len) { touched += len; });
		// The block's GPU mappings now map the buffer instead of the dummy, even if a CPU remap failed:
		// the buffer holds the data either way.
		(void)MoveRangeUnlocked(start, block.size, bo, nullptr);
		::fallocate(m_mem_fd, FALLOC_FL_PUNCH_HOLE | FALLOC_FL_KEEP_SIZE, static_cast<off_t>(start),
		            static_cast<off_t>(block.size));
		block.bo       = bo;
		block.resident = true;
		if (block.deferred) {
			block.deferred = false;
			m_deferred_blocks--;
		}
		m_resident_bytes += block.size;
		m_touched_bytes += touched;
		std::fprintf(stderr, "GemBacking: block %llu resident (%llu KiB written by the CPU), %llu MiB of guest memory in GTT\n",
		             static_cast<unsigned long long>(index), static_cast<unsigned long long>(touched >> 10u),
		             static_cast<unsigned long long>(m_resident_bytes / (1024 * 1024)));
		return true;
	}

	// Calls cb(start, size, writable) for each readable host VMA slice overlapping [vaddr, vaddr+size),
	// read from /proc/self/maps. Guest memory is mapped at host VA == guest VA; `writable` is the
	// slice's host write permission. Unreadable slices (no-access or execute-only) are skipped, since a
	// userptr cannot fault them in. If maps cannot be read, the whole range is offered once as writable
	// and the caller's protection choice still applies.
	template <typename Cb>
	static void ForEachReadableSubrange(uint64_t vaddr, uint64_t size, Cb&& cb) {
		const uint64_t end  = vaddr + size;
		FILE*          maps = ::fopen("/proc/self/maps", "re");
		if (maps == nullptr) {
			cb(vaddr, size, true);
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
				cb(cs, ce - cs, perms[1] == 'w');
			}
		}
		::fclose(maps);
	}

	// Registers [vaddr, vaddr + size) host pages as a userptr GEM buffer and maps it into the GPU VM
	// at vaddr. `gpu_write` selects a writable or read-only buffer and page mapping. Caller holds
	// m_lock and has already cleared any overlapping external binding.
	bool RegisterExternalUnlocked(uint64_t vaddr, uint64_t size, bool gpu_write) {
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
		return true;
	}

	// Unmaps and closes every external buffer overlapping [vaddr, vaddr + size), re-registering the
	// portions of a straddling buffer that fall outside the range as their own buffers (closing a GEM
	// handle unmaps all of its VA, so a survivor needs its own handle). Chunk mappings never appear in
	// m_external, so they are never touched. Caller holds m_lock.
	void ClearExternalRangeUnlocked(uint64_t vaddr, uint64_t size) {
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
			} else {
				++it;
			}
		}

		for (const auto& [start, ext]: overlapping) {
			VaOp(AMDGPU_VA_OP_CLEAR, 0, start, 0, ext.end - start);
			if (ext.always_valid) {
				// The survivors keep the buffer (their CPU views map it): map them again from it.
				auto keep = [&](uint64_t s, uint64_t e) {
					External part     = ext;
					part.end          = e;
					part.bo_offset    = ext.bo_offset + (s - start);
					VaOp(AMDGPU_VA_OP_REPLACE, ext.handle, s, part.bo_offset, e - s, ExternalPageFlags(ext.writable));
					m_external.emplace(s, part);
					m_external_refs[ext.handle]++;
				};
				if (start < vaddr) {
					keep(start, vaddr);
				}
				if (end < ext.end) {
					keep(end, ext.end);
				}
				if (--m_external_refs[ext.handle] == 0) {
					m_external_refs.erase(ext.handle);
					GemClose(ext.handle); // CPU views that are still mapped keep the pages alive
				}
				continue;
			}
			GemClose(ext.handle);
			if (start < vaddr) {
				RegisterExternalUnlocked(start, vaddr - start, ext.writable);
			}
			if (end < ext.end) {
				RegisterExternalUnlocked(end, ext.end - end, ext.writable);
			}
		}
	}

	static uint32_t ExternalPageFlags(bool gpu_write) {
		return gpu_write ? kRwxPageFlags : (AMDGPU_VM_PAGE_READABLE | AMDGPU_VM_PAGE_EXECUTABLE);
	}

	// Exact model: when [vaddr, vaddr + size) is wholly inside one always-valid external buffer,
	// sets its GPU write permission and returns true.
	bool UpdateAlwaysValidExternalUnlocked(uint64_t vaddr, uint64_t size, bool gpu_write) {
		auto it = m_external.upper_bound(vaddr);
		if (it == m_external.begin()) {
			return false;
		}
		--it;
		External& ext = it->second;
		if (!ext.always_valid || vaddr + size > ext.end) {
			return false;
		}
		if (ext.writable != gpu_write && it->first == vaddr && ext.end == vaddr + size) {
			ext.writable = gpu_write;
			VaOp(AMDGPU_VA_OP_REPLACE, ext.handle, vaddr, ext.bo_offset, size, ExternalPageFlags(gpu_write));
		}
		return true;
	}

	// Exact model: moves the host pages of [vaddr, vaddr + size) into a new always-valid buffer (the
	// range's mappings, split at protection boundaries, become views of it with their protection) and
	// maps it into the GPU VM there. The bytes are read through /proc/self/mem, which also reads the
	// no-access and execute-only slices.
	bool MoveExternalUnlocked(uint64_t vaddr, uint64_t size, bool gpu_write) {
		if (m_self_mem_fd < 0) {
			m_self_mem_fd = ::open("/proc/self/mem", O_RDONLY | O_CLOEXEC);
			if (m_self_mem_fd < 0) {
				return false;
			}
		}
		Bo bo;
		if (!CreateBo(size, &bo, true)) {
			return false;
		}
		auto* bo_cpu = static_cast<uint8_t*>(MapWhole(bo));
		if (bo_cpu == MAP_FAILED) {
			CloseBo(bo);
			return false;
		}
		auto job     = std::make_unique<MoveJob>();
		job->start   = vaddr;
		job->size    = size;
		job->src_fd  = m_self_mem_fd;
		job->bo_cpu  = bo_cpu;
		job->bo      = bo;
		ForEachHostProtection(vaddr, size, [&](uint64_t s, uint64_t len, int prot) {
			job->pieces.push_back(MovePiece {s, len, prot, s - vaddr});
		});
		RunOnMover(job.get());
		::munmap(bo_cpu, size);
		if (!job->moved) {
			// Some pieces may already map the buffer: keep it bound so whatever maps it stays coherent.
			std::fprintf(stderr, "GemBacking: moving external 0x%llx (+0x%llx) failed: %s\n",
			             static_cast<unsigned long long>(vaddr), static_cast<unsigned long long>(size), std::strerror(errno));
		}
		if (VaOp(AMDGPU_VA_OP_REPLACE, bo.handle, vaddr, 0, size, ExternalPageFlags(gpu_write)) != 0) {
			CloseBo(bo);
			return false;
		}
		m_external.emplace(vaddr, External {vaddr + size, bo.handle, gpu_write, true, 0});
		m_external_refs[bo.handle] = 1;
		return true;
	}

	struct External {
		uint64_t end          = 0; // one past the last byte; start is the m_external key
		uint32_t handle       = 0;
		bool     writable     = false;
		bool     always_valid = false; // exact model: a buffer the range was moved into, not a userptr
		uint64_t bo_offset    = 0;     // where the range starts in that buffer
	};

	int                           m_drm_fd = -1;
	int                           m_mem_fd = -1;
	uint8_t*                      m_alias  = nullptr;
	uint64_t                      m_size   = 0;
	uint64_t                      m_block_size = kBlockSize;
	bool                          m_exact      = false;
	bool                          m_dummy_mapped = false; // something's GPU mapping points at the dummy
	uint64_t                      m_budget       = 0;     // exact model: resident bytes MapGpuVisible stays within (0: none)
	uint64_t                      m_deferred_blocks = 0;  // exact model: blocks MapGpuVisible left to lazy residency
	uint64_t                      m_resident_bytes = 0;
	uint64_t                      m_touched_bytes  = 0;
	std::vector<Block>            m_blocks;
	std::map<uint64_t, View>      m_views;     // guest VA -> view of the backing
	uint64_t                      m_views_generation = 0;
	Bo                            m_dummy;     // GPU target of every non-resident block
	uint8_t*                      m_dummy_cpu = nullptr;
	static inline std::atomic<uint32_t> s_promoting {0}; // blocks being moved right now
	std::thread                   m_mover;      // runs MoveBlock (see there)
	std::mutex                    m_mover_lock;
	std::condition_variable       m_mover_cv;
	MoveJob*                      m_mover_job  = nullptr;
	bool                          m_mover_stop = false;
	std::map<uint64_t, External>  m_external; // guest VA -> userptr buffer for non-direct ranges
	std::map<uint64_t, Export>    m_exports;  // backing offset -> exportable buffer (ExportRange)
	std::map<uint32_t, uint32_t>  m_external_refs; // always-valid external buffer -> m_external entries using it
	int                           m_self_mem_fd = -1; // /proc/self/mem, read by MoveExternalUnlocked
	mutable std::mutex            m_lock;     // guards blocks, views, m_external and the dummy
};

} // namespace Libs::Graphics::NativeGpu

#endif // KYTY_GEM_BACKING_SUPPORTED

#endif // EMULATOR_SRC_GRAPHICS_NATIVE_GPU_GEMBACKING_H_
