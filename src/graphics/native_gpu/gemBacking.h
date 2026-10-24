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
// at the same virtual address, on /dev/dri/renderD128, while only the memory the GPU actually
// needs is held in GPU buffers.
//
// The CPU side is a memfd, like the upstream backing: the guest views and the contiguous alias all
// map it, its pages are allocated on first touch and the kernel may swap them. A game can map far
// more direct memory than the BC-250 can keep resident (ASTRO BOT maps ~13 GiB of its 13.5 GiB
// budget and touches ~7 GiB), and GPU buffers cannot be swapped while submissions use them.
//
// The GPU side works in kBlockSize blocks of the backing. A block becomes resident when the CPU
// has written to it, which the memfd shows as data (PromoteTouched, via SEEK_DATA), or when a
// command buffer about to run references it (PromoteGuestRange: render targets, index buffers,
// pointers in shader user data, as the PM4 translator reports them). Promotion creates a cached GTT
// GEM buffer for the block, write-protects the block's views and alias slot, copies its data into
// the buffer, maps them onto the buffer with their protection restored through a dma-buf export
// (mmap through the DRM file only maps a buffer whole and from its start; a dma-buf mmap takes any
// page offset), punches the block out of the memfd and rebinds the GPU mappings. A guest write
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
class GemBacking {
public:
	// Residency granularity. Every resident block is listed in every submission (a listed buffer
	// that is in place costs about a microsecond), so smaller blocks waste less memory but make
	// longer lists. 16 MiB divides the 13.5 GiB backing evenly.
	static constexpr uint64_t kBlockSize = 16ull * 1024 * 1024;

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
		const auto count = (size + kBlockSize - 1) / kBlockSize;
		self->m_blocks.resize(count);
		for (uint64_t i = 0; i < count; i++) {
			const uint64_t start    = i * kBlockSize;
			self->m_blocks[i].size = (size - start < kBlockSize) ? size - start : kBlockSize;
		}
		void* dummy_cpu = MAP_FAILED;
		if (self->CreateBo(kBlockSize, &self->m_dummy)) {
			dummy_cpu = ::mmap(nullptr, kBlockSize, PROT_READ | PROT_WRITE, MAP_SHARED,
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
			VaOp(AMDGPU_VA_OP_CLEAR, 0, vaddr, 0, ext.end - vaddr);
			GemClose(ext.handle);
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
			const Block&   block = m_blocks[index];
			const int      fd    = block.resident ? block.bo.dmabuf_fd : m_mem_fd;
			const uint64_t at    = block.resident ? inner : offset + done;
			void* const    dst   = reinterpret_cast<void*>(vaddr + done);
			if (::mmap(dst, part, prot, MAP_SHARED | MAP_FIXED, fd, static_cast<off_t>(at)) != dst) {
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
			return VaOp(AMDGPU_VA_OP_REPLACE, handle, vaddr + done, inner, part) == 0;
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
			const uint64_t index = static_cast<uint64_t>(data) / kBlockSize;
			if (!m_blocks[index].resident && PromoteBlockUnlocked(index)) {
				promoted++;
			}
			pos = (index + 1) * kBlockSize;
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
				const uint64_t index = backing / kBlockSize;
				const uint64_t inner = backing % kBlockSize;
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
			for (uint64_t index = backing / kBlockSize; index <= (backing + len - 1) / kBlockSize; index++) {
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

	// Every GEM buffer a submission must list as resident: the resident blocks, the dummy and the
	// external ranges. Grows as blocks are promoted.
	[[nodiscard]] std::vector<uint32_t> CreatedHandles() const {
		std::lock_guard<std::mutex> lock(m_lock);
		std::vector<uint32_t>       handles;
		handles.reserve(m_blocks.size() + m_external.size() + 1);
		for (const auto& block: m_blocks) {
			if (block.resident) {
				handles.push_back(block.bo.handle);
			}
		}
		handles.push_back(m_dummy.handle);
		for (const auto& [vaddr, ext]: m_external) {
			handles.push_back(ext.handle);
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
		const uint64_t              index = offset / kBlockSize;
		if (!m_blocks[index].resident) {
			return false;
		}
		if (handle != nullptr) {
			*handle = m_blocks[index].bo.handle;
		}
		if (offset_in_block != nullptr) {
			*offset_in_block = offset - index * kBlockSize;
		}
		return true;
	}

private:
	struct Bo {
		uint32_t handle    = 0;
		int      dmabuf_fd = -1;
	};
	struct Block {
		Bo       bo;
		uint64_t size     = 0;
		bool     resident = false;
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

	// A cached GTT buffer (TTM zeroes it) exported as a dma-buf for CPU mappings.
	bool CreateBo(uint64_t size, Bo* out) const {
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

	void CloseBo(Bo& bo) const {
		if (bo.dmabuf_fd >= 0) {
			::close(bo.dmabuf_fd);
		}
		if (bo.handle != 0) {
			GemClose(bo.handle);
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
			const uint64_t index = at / kBlockSize;
			const uint64_t inner = at - index * kBlockSize;
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

	// Calls cb(offset in block, length) for each data extent of block `index` in the memfd.
	template <typename Cb>
	void ForEachDataExtent(uint64_t index, Cb&& cb) const {
		const uint64_t start = index * kBlockSize;
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
		uint8_t*               bo_cpu = nullptr;
		int                    fd     = -1; // the block buffer's dma-buf
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
			moved = ::mmap(dst, p.len, p.prot, MAP_SHARED | MAP_FIXED, job->fd, static_cast<off_t>(p.inner)) == dst && moved;
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

	bool PromoteBlockUnlocked(uint64_t index) {
		Block&         block = m_blocks[index];
		const uint64_t start = index * kBlockSize;
		Bo             bo;
		if (!CreateBo(block.size, &bo)) {
			return false;
		}
		auto* bo_cpu = static_cast<uint8_t*>(
		    ::mmap(nullptr, block.size, PROT_READ | PROT_WRITE, MAP_SHARED, bo.dmabuf_fd, 0));
		if (bo_cpu == MAP_FAILED) {
			std::fprintf(stderr, "GemBacking: CPU map of block %llu failed: %s\n",
			             static_cast<unsigned long long>(index), std::strerror(errno));
			CloseBo(bo);
			return false;
		}
		// Every CPU mapping of the block: (address, length, protection, offset in block), the alias slot
		// first, then each guest view piece split at protection boundaries. Kept in a heap job: the
		// move runs on the mover thread and must not write to this thread's stack (see MoveBlock).
		auto job    = std::make_unique<MoveJob>();
		job->index  = index;
		job->start  = start;
		job->size   = block.size;
		job->bo_cpu = bo_cpu;
		job->fd     = bo.dmabuf_fd;
		std::vector<std::pair<uint64_t, uint64_t>> gpu_pieces; // (guest address, offset in block)
		std::vector<uint64_t>                      gpu_lengths;
		job->pieces.push_back(MovePiece {reinterpret_cast<uint64_t>(m_alias + start), block.size, PROT_READ | PROT_WRITE, 0});
		for (const auto& [view_addr, view]: m_views) {
			const uint64_t vs = std::max(view.offset, start);
			const uint64_t ve = std::min(view.offset + view.size, start + block.size);
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
		const bool moved = job->moved;
		if (!moved) {
			std::fprintf(stderr, "GemBacking: remapping block %llu failed: %s\n",
			             static_cast<unsigned long long>(index), std::strerror(errno));
		}
		::munmap(bo_cpu, block.size);
		::fallocate(m_mem_fd, FALLOC_FL_PUNCH_HOLE | FALLOC_FL_KEEP_SIZE, static_cast<off_t>(start),
		            static_cast<off_t>(block.size));
		// GPU mappings of the block's views now map the buffer instead of the dummy.
		for (size_t i = 0; i < gpu_pieces.size(); i++) {
			VaOp(AMDGPU_VA_OP_REPLACE, bo.handle, gpu_pieces[i].first, gpu_pieces[i].second, gpu_lengths[i]);
		}
		block.bo       = bo;
		block.resident = true;
		m_resident_bytes += block.size;
		std::fprintf(stderr, "GemBacking: block %llu resident, %llu MiB of guest memory in GTT\n",
		             static_cast<unsigned long long>(index),
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
			GemClose(ext.handle);
			if (start < vaddr) {
				RegisterExternalUnlocked(start, vaddr - start, ext.writable);
			}
			if (end < ext.end) {
				RegisterExternalUnlocked(end, ext.end - end, ext.writable);
			}
		}
	}

	struct External {
		uint64_t end      = 0; // one past the last byte; start is the m_external key
		uint32_t handle   = 0;
		bool     writable = false;
	};

	int                           m_drm_fd = -1;
	int                           m_mem_fd = -1;
	uint8_t*                      m_alias  = nullptr;
	uint64_t                      m_size   = 0;
	uint64_t                      m_resident_bytes = 0;
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
	mutable std::mutex            m_lock;     // guards blocks, views, m_external and the dummy
};

} // namespace Libs::Graphics::NativeGpu

#endif // KYTY_GEM_BACKING_SUPPORTED

#endif // EMULATOR_SRC_GRAPHICS_NATIVE_GPU_GEMBACKING_H_
