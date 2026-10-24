#include "graphics/native_gpu/gemBacking.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#if !KYTY_GEM_BACKING_SUPPORTED

int main() {
	std::puts("NativeGpuGemBackingTests: skipped (platform without amdgpu GEM backing)");
	return 0;
}

#else

#include <cerrno>
#include <ctime>
#include <initializer_list>

#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

using Libs::Graphics::NativeGpu::GemBacking;

namespace {

void Check(bool value, const char* message) {
	if (!value) {
		std::fprintf(stderr, "NativeGpuGemBackingTests: failed: %s\n", message);
		std::abort();
	}
}

// A guest-style placeholder the backing can replace with MAP_FIXED, mirroring the reserved
// address space the emulator hands to MapView.
void* ReserveGuest(uint64_t size) {
	void* p = ::mmap(nullptr, size, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
	Check(p != MAP_FAILED, "guest placeholder reservation failed");
	return p;
}


int Ioctl(int fd, unsigned long request, void* arg) {
	int r = -1;
	do {
		r = ::ioctl(fd, request, arg);
	} while (r == -1 && (errno == EINTR || errno == EAGAIN));
	return r == -1 ? -errno : 0;
}

// A write-combined GTT buffer mapped for the CPU and at `gpu_va` in the backing's GPU VM.
struct Buffer {
	uint32_t  handle = 0;
	uint32_t* cpu    = nullptr;
	uint64_t  va     = 0;
};

Buffer CreateBuffer(int fd, uint64_t size, uint64_t gpu_va) {
	Buffer b;
	union drm_amdgpu_gem_create c;
	std::memset(&c, 0, sizeof(c));
	c.in.bo_size      = size;
	c.in.alignment    = 4096;
	c.in.domains      = AMDGPU_GEM_DOMAIN_GTT;
	// Uncached for the GPU: the same IB is reused, and a cached one can be fetched stale from L2.
	c.in.domain_flags = AMDGPU_GEM_CREATE_CPU_GTT_USWC | AMDGPU_GEM_CREATE_UNCACHED;
	Check(Ioctl(fd, DRM_IOCTL_AMDGPU_GEM_CREATE, &c) == 0, "GEM_CREATE of a command buffer failed");
	b.handle = c.out.handle;
	union drm_amdgpu_gem_mmap m;
	std::memset(&m, 0, sizeof(m));
	m.in.handle = b.handle;
	Check(Ioctl(fd, DRM_IOCTL_AMDGPU_GEM_MMAP, &m) == 0, "GEM_MMAP failed");
	void* p = ::mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, static_cast<off_t>(m.out.addr_ptr));
	Check(p != MAP_FAILED, "mmap of a command buffer failed");
	b.cpu = static_cast<uint32_t*>(p);
	struct drm_amdgpu_gem_va v;
	std::memset(&v, 0, sizeof(v));
	v.handle     = b.handle;
	v.operation  = AMDGPU_VA_OP_MAP;
	v.flags      = AMDGPU_VM_PAGE_READABLE | AMDGPU_VM_PAGE_WRITEABLE | AMDGPU_VM_PAGE_EXECUTABLE;
	v.va_address = gpu_va;
	v.map_size   = size;
	Check(Ioctl(fd, DRM_IOCTL_AMDGPU_GEM_VA, &v) == 0, "GEM_VA of a command buffer failed");
	b.va = gpu_va;
	return b;
}

uint32_t Pkt3(uint32_t op, uint32_t body_dw) {
	return (3u << 30) | ((body_dw - 1u) << 16) | (op << 8) | (1u << 1);
}

// store.s from tools/bc250-native-probe for gfx1013 wave64: s[0:1] = destination, s2 = base,
// s3 = workgroup id; lane i of group g writes base + g * 64 + i to destination + (g * 64 + i) * 4.
constexpr uint32_t kStoreShader[] = {0x34020082u, 0x8f048803u, 0x4a020204u, 0x8f058603u, 0x80050205u,
                                     0x4a040005u, 0xdc708000u, 0x00000201u, 0xbf810000u};

// Runs kStoreShader over [dst_va, dst_va + groups * 256) on the compute ring, with `extra` GEM
// handles resident, and returns once the job signaled without a context reset.
void RunStore(int fd, uint32_t ctx, const Buffer& ib, const Buffer& code, uint64_t dst_va, uint32_t base,
              uint32_t groups, const uint32_t* extra, uint32_t extra_count) {
	uint32_t n   = 0;
	auto     set = [&](uint32_t reg, std::initializer_list<uint32_t> values) {
		ib.cpu[n++] = Pkt3(0x76u, static_cast<uint32_t>(values.size()) + 1); // SET_SH_REG
		ib.cpu[n++] = (reg - 0xb000u) / 4;
		for (auto v: values) {
			ib.cpu[n++] = v;
		}
	};
	set(0xb830u, {static_cast<uint32_t>(code.va >> 8), static_cast<uint32_t>(code.va >> 40)}); // PGM_LO/HI
	set(0xb848u, {1u | (0xc0u << 12) | (1u << 21) | (1u << 30), (3u << 1) | (1u << 7)});        // RSRC1/2
	set(0xb8a0u, {0});                                                                           // RSRC3
	set(0xb854u, {0});                                                                           // RESOURCE_LIMITS
	set(0xb860u, {0});                                                                           // TMPRING_SIZE
	set(0xb858u, {0xffffffffu, 0xffffffffu});                                                    // THREAD_MGMT_SE0/1
	set(0xb864u, {0xffffffffu, 0xffffffffu});                                                    // THREAD_MGMT_SE2/3
	set(0xb810u, {0, 0, 0});                                                                     // START_X/Y/Z
	set(0xb81cu, {64, 1, 1});                                                                    // NUM_THREAD_X/Y/Z
	set(0xb900u, {static_cast<uint32_t>(dst_va), static_cast<uint32_t>(dst_va >> 32), base});   // USER_DATA_0..2
	ib.cpu[n++] = Pkt3(0x15u, 4); // DISPATCH_DIRECT
	ib.cpu[n++] = groups;
	ib.cpu[n++] = 1;
	ib.cpu[n++] = 1;
	ib.cpu[n++] = 1u | 4u | (1u << 6); // COMPUTE_SHADER_EN | FORCE_START_AT_000 | ORDER_MODE
	while ((n & 7u) != 0) {
		ib.cpu[n++] = 0xffff1000u;
	}
	__sync_synchronize();

	struct drm_amdgpu_bo_list_entry entries[32];
	uint32_t                        count = 0;
	entries[count++]                      = {ib.handle, 0};
	if (code.handle != 0) { // 0 => the code lives in an external BO passed via `extra`
		entries[count++] = {code.handle, 0};
	}
	for (uint32_t i = 0; i < extra_count; i++) {
		entries[count++] = {extra[i], 0};
	}
	struct drm_amdgpu_bo_list_in list;
	std::memset(&list, 0, sizeof(list));
	list.operation    = ~0u;
	list.list_handle  = ~0u;
	list.bo_number    = count;
	list.bo_info_size = sizeof(entries[0]);
	list.bo_info_ptr  = reinterpret_cast<uintptr_t>(entries);
	struct drm_amdgpu_cs_chunk_ib ib_info;
	std::memset(&ib_info, 0, sizeof(ib_info));
	ib_info.va_start = ib.va;
	ib_info.ib_bytes = n * 4;
	ib_info.ip_type  = AMDGPU_HW_IP_COMPUTE;
	struct drm_amdgpu_cs_chunk chunks[2];
	chunks[0]               = {AMDGPU_CHUNK_ID_IB, sizeof(ib_info) / 4, reinterpret_cast<uintptr_t>(&ib_info)};
	chunks[1]               = {AMDGPU_CHUNK_ID_BO_HANDLES, sizeof(list) / 4, reinterpret_cast<uintptr_t>(&list)};
	uint64_t           ptrs[2] = {reinterpret_cast<uintptr_t>(&chunks[0]), reinterpret_cast<uintptr_t>(&chunks[1])};
	union drm_amdgpu_cs cs;
	std::memset(&cs, 0, sizeof(cs));
	cs.in.ctx_id     = ctx;
	cs.in.num_chunks = 2;
	cs.in.chunks     = reinterpret_cast<uintptr_t>(ptrs);
	Check(Ioctl(fd, DRM_IOCTL_AMDGPU_CS, &cs) == 0, "command submission failed");

	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	union drm_amdgpu_wait_cs w;
	std::memset(&w, 0, sizeof(w));
	w.in.handle  = cs.out.handle;
	w.in.timeout = static_cast<uint64_t>(ts.tv_sec + 5) * 1000000000ull + static_cast<uint64_t>(ts.tv_nsec);
	w.in.ip_type = AMDGPU_HW_IP_COMPUTE;
	w.in.ctx_id  = ctx;
	Check(Ioctl(fd, DRM_IOCTL_AMDGPU_WAIT_CS, &w) == 0 && w.out.status == 0, "GPU job did not signal");

	union drm_amdgpu_ctx q;
	std::memset(&q, 0, sizeof(q));
	q.in.op     = AMDGPU_CTX_OP_QUERY_STATE2;
	q.in.ctx_id = ctx;
	Check(Ioctl(fd, DRM_IOCTL_AMDGPU_CTX, &q) == 0 && q.out.state.flags == 0, "GPU context was reset");
}

} // namespace

int main() {
	const char* device = "/dev/dri/renderD128";
	int         probe  = ::open(device, O_RDWR | O_CLOEXEC);
	if (probe < 0) {
		std::printf("NativeGpuGemBackingTests: skipped (%s unavailable: %s)\n", device,
		            std::strerror(errno));
		return 0;
	}
	::close(probe);

	const uint64_t chunk = GemBacking::kBlockSize;
	// 2.5 blocks: two full blocks plus a half-size tail block, so the last block is partial.
	const uint64_t size = chunk * 2 + chunk / 2;

	GemBacking* gem = GemBacking::Create(size);
	Check(gem != nullptr, "Create returned null on a board with the device present");
	Check(gem->AliasBase() != nullptr, "alias base is null");
	Check(gem->Size() == size, "size mismatch");

	auto* alias = gem->AliasBase();

	// Case 1: a view wholly inside chunk 0. Write through the view, read through the alias.
	{
		const uint64_t off = 0x40000;  // 256 KiB in
		const uint64_t len = 0x20000;  // 128 KiB
		auto*          view = static_cast<uint32_t*>(ReserveGuest(len));
		Check(gem->MapView(reinterpret_cast<uint64_t>(view), len, off, PROT_READ | PROT_WRITE),
		      "MapView of in-chunk range failed");
		for (uint32_t i = 0; i < len / 4; i++) {
			view[i] = 0xa5000000u + i;
		}
		const auto* a = reinterpret_cast<const uint32_t*>(alias + off);
		for (uint32_t i = 0; i < len / 4; i++) {
			Check(a[i] == 0xa5000000u + i, "alias did not see view write (in-chunk)");
		}
	}

	// Case 2: a view straddling the block 0 / block 1 boundary. Writes through it make both blocks
	// touched; PromoteTouched must then move them (and the case 1 range in block 0) into two GEM
	// buffers without losing data, with view and alias still coherent. Lookup resolves across the seam.
	{
		const uint64_t len = 0x40000;        // 256 KiB
		const uint64_t off = chunk - len / 2; // half in chunk 0, half in chunk 1
		auto*          view = static_cast<uint32_t*>(ReserveGuest(len));
		Check(gem->MapView(reinterpret_cast<uint64_t>(view), len, off, PROT_READ | PROT_WRITE),
		      "MapView across a chunk boundary failed");

		for (uint32_t i = 0; i < len / 4; i++) {
			view[i] = 0x5a000000u + i;
		}
		const auto* a = reinterpret_cast<const uint32_t*>(alias + off);
		for (uint32_t i = 0; i < len / 4; i++) {
			Check(a[i] == 0x5a000000u + i, "alias did not see view write (crossing)");
		}

		Check(!gem->Lookup(off, nullptr, nullptr), "a block must not be resident before promotion");
		Check(gem->PromoteTouched() == 2, "PromoteTouched must promote exactly the two touched blocks");
		Check(gem->PromoteTouched() == 0, "promoted blocks must not be found touched again");
		for (uint32_t i = 0; i < len / 4; i++) {
			Check(view[i] == 0x5a000000u + i, "promotion lost data written before it");
		}
		Check(reinterpret_cast<const uint32_t*>(alias + 0x40000)[0] == 0xa5000000u,
		      "promotion lost the case 1 data in block 0");

		// Alias write must be visible through the view: same GEM pages, two mappings.
		reinterpret_cast<uint32_t*>(alias + off)[0] = 0xdeadbeefu;
		Check(view[0] == 0xdeadbeefu, "view did not see alias write");

		uint32_t low_handle  = 0;
		uint32_t high_handle = 0;
		uint64_t low_inner   = ~0ull;
		uint64_t high_inner  = ~0ull;
		Check(gem->Lookup(off, &low_handle, &low_inner), "lookup of low side failed");
		Check(gem->Lookup(chunk, &high_handle, &high_inner), "lookup of high side failed");
		Check(low_handle != 0 && high_handle != 0, "lookup returned a null handle");
		Check(low_handle != high_handle, "straddling offsets must resolve to different chunks");
		Check(low_inner == off, "chunk 0 inner offset wrong");
		Check(high_inner == 0, "chunk 1 inner offset must be 0 at the chunk start");
	}

	// Case 3: the half-size tail block was never touched, so it is not resident and does not resolve.
	{
		uint32_t handle = 0;
		uint64_t inner  = 0;
		Check(!gem->Lookup(chunk * 2 + 0x1000, &handle, &inner),
		      "an untouched block must not resolve (lazy residency)");
	}


	// Case 4: the GPU writes through a guest view at the view's own address. The view straddles the
	// chunk 0 / chunk 1 seam, so BindView must create one GPU mapping per chunk; the result is read
	// back through both the view and the alias. Then unbind, rebind and write again.
	{
		const uint64_t len    = 0x40000;        // 256 KiB = 1024 groups of 64 dwords
		const uint64_t off    = chunk - len / 2; // half in chunk 0, half in chunk 1
		const uint32_t groups = static_cast<uint32_t>(len / 256);
		auto*          view   = static_cast<uint32_t*>(ReserveGuest(len));
		const uint64_t va     = reinterpret_cast<uint64_t>(view);
		Check(gem->MapView(va, len, off, PROT_READ | PROT_WRITE), "MapView for the GPU case failed");
		Check(gem->BindView(va, len, off), "BindView across a chunk boundary failed");
		std::memset(view, 0, len);

		const int fd = gem->DrmFd();
		union drm_amdgpu_ctx c;
		std::memset(&c, 0, sizeof(c));
		c.in.op = AMDGPU_CTX_OP_ALLOC_CTX;
		Check(Ioctl(fd, DRM_IOCTL_AMDGPU_CTX, &c) == 0, "context allocation failed");
		const uint32_t ctx = c.out.alloc.ctx_id;

		// Command and code buffers far below any host address the guest views use.
		const Buffer ib   = CreateBuffer(fd, 64 * 1024, 0x0080000000ull);
		const Buffer code = CreateBuffer(fd, 64 * 1024, 0x0081000000ull);
		std::memcpy(code.cpu, kStoreShader, sizeof(kStoreShader));

		gem->PromoteGuestRange(va, len);
		uint32_t chunks[2] = {};
		Check(gem->Lookup(off, &chunks[0], nullptr) && gem->Lookup(off + len - 1, &chunks[1], nullptr),
		      "chunk lookup for the GPU case failed");

		for (const uint32_t base: {0x11000000u, 0x22000000u}) {
			RunStore(fd, ctx, ib, code, va, base, groups, chunks, 2);
			const auto* a = reinterpret_cast<const uint32_t*>(alias + off);
			for (uint32_t i = 0; i < len / 4; i++) {
				Check(view[i] == base + i, "GPU write through the guest VA not visible in the view");
				Check(a[i] == base + i, "GPU write through the guest VA not visible in the alias");
			}
			// Second round goes through a fresh mapping: clear the range, then bind it again.
			Check(gem->UnbindView(va, len), "UnbindView failed");
			Check(gem->BindView(va, len, off), "BindView after UnbindView failed");
		}
		std::printf("NativeGpuGemBackingTests: GPU wrote %llu KiB through guest VA 0x%llx across chunks\n",
		            static_cast<unsigned long long>(len / 1024), static_cast<unsigned long long>(va));

		// Case 4b: a GPU write to a view of the untouched tail block lands in the dummy (no fault),
		// is reported by DummyWritten, and after PromoteGuestRange the same write reaches the view.
		const uint64_t tail_off = chunk * 2 + 0x10000;
		auto*          tail     = static_cast<uint32_t*>(ReserveGuest(0x10000));
		const uint64_t tail_va  = reinterpret_cast<uint64_t>(tail);
		Check(gem->MapView(tail_va, 0x10000, tail_off, PROT_READ | PROT_WRITE), "MapView of the tail failed");
		Check(gem->BindView(tail_va, 0x10000, tail_off), "BindView of the tail failed");
		Check(!gem->DummyWritten(), "dummy must start clean");
		std::vector<uint32_t> all = gem->CreatedHandles();
		RunStore(fd, ctx, ib, code, tail_va, 0x66000000u, 0x10000 / 256, all.data(),
		         static_cast<uint32_t>(all.size()));
		Check(tail[0] == 0, "a GPU write to a non-resident block must not reach the view");
		Check(gem->DummyWritten(), "DummyWritten must report the stray GPU write");
		Check(gem->PromoteGuestRange(tail_va, 0x10000) == 1, "PromoteGuestRange must promote the tail block");
		all = gem->CreatedHandles();
		RunStore(fd, ctx, ib, code, tail_va, 0x77000000u, 0x10000 / 256, all.data(),
		         static_cast<uint32_t>(all.size()));
		for (uint32_t i = 0; i < 0x10000 / 4; i++) {
			Check(tail[i] == 0x77000000u + i, "GPU write after promotion not visible in the view");
		}
		Check(gem->ResidentBytes() == chunk * 2 + chunk / 2, "all three blocks must now be resident");
	}

	// Case 5: a guest range that is NOT part of the memfd backing — flexible/program/runtime/stack
	// memory in the emulator: committed anonymous host pages at guest VA == host VA. BindExternal
	// registers them as their own userptr buffer and maps them into the GPU VM at that VA; the GPU
	// writes through it, and a partial unbind in the middle must leave both survivors usable while
	// clearing only the hole.
	{
		const int fd = gem->DrmFd();
		union drm_amdgpu_ctx c;
		std::memset(&c, 0, sizeof(c));
		c.in.op = AMDGPU_CTX_OP_ALLOC_CTX;
		Check(Ioctl(fd, DRM_IOCTL_AMDGPU_CTX, &c) == 0, "external-case context allocation failed");
		const uint32_t ctx = c.out.alloc.ctx_id;

		const Buffer ib   = CreateBuffer(fd, 64 * 1024, 0x0082000000ull);
		const Buffer code = CreateBuffer(fd, 64 * 1024, 0x0083000000ull);
		std::memcpy(code.cpu, kStoreShader, sizeof(kStoreShader));

		// Three 64 KiB thirds: an anonymous host mapping standing in for non-direct guest memory.
		const uint64_t third = 0x10000;
		const uint64_t len   = third * 3;
		void*          host  = ::mmap(nullptr, len, PROT_READ | PROT_WRITE,
		                              MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
		Check(host != MAP_FAILED, "external host mapping failed");
		const uint64_t va = reinterpret_cast<uint64_t>(host);
		auto*          words = static_cast<uint32_t*>(host);

		const size_t handles_before = gem->CreatedHandles().size();
		Check(gem->BindExternal(va, len, true), "BindExternal of a non-backed range failed");
		Check(gem->CreatedHandles().size() == handles_before + 1,
		      "BindExternal must add exactly one resident buffer");

		std::memset(host, 0, len);
		{
			const std::vector<uint32_t> extra = gem->CreatedHandles();
			RunStore(fd, ctx, ib, code, va, 0x30000000u, static_cast<uint32_t>(len / 256), extra.data(),
			         static_cast<uint32_t>(extra.size()));
			for (uint32_t i = 0; i < len / 4; i++) {
				Check(words[i] == 0x30000000u + i, "GPU write through an external guest VA not visible");
			}
		}

		// Unbind the middle third: the hole is cleared, the two survivors get their own buffers.
		Check(gem->UnbindExternal(va + third, third), "partial UnbindExternal failed");
		Check(gem->CreatedHandles().size() == handles_before + 2,
		      "a straddling unbind must leave two survivor buffers");

		std::memset(host, 0, third); // first survivor only
		{
			const std::vector<uint32_t> extra = gem->CreatedHandles();
			RunStore(fd, ctx, ib, code, va, 0x40000000u, static_cast<uint32_t>(third / 256),
			         extra.data(), static_cast<uint32_t>(extra.size()));
			for (uint32_t i = 0; i < third / 4; i++) {
				Check(words[i] == 0x40000000u + i, "GPU write to the surviving third not visible");
			}
		}

		Check(gem->UnbindExternal(va, len), "full UnbindExternal failed");
		Check(gem->CreatedHandles().size() == handles_before,
		      "UnbindExternal must release every external buffer it added");

		// Read-only external range: shader code is mapped r-x, and amdgpu rejects a writable userptr
		// whose host pages are not writable (the submission fails with EPERM), so the read-only
		// registration path must work on its own. Put the shader in the range, drop host write
		// permission to mirror code memory, bind it read-only, and have the GPU fetch and execute it
		// to write a separate read-write output.
		{
			std::memcpy(host, kStoreShader, sizeof(kStoreShader));
			Check(::mprotect(host, third, PROT_READ | PROT_EXEC) == 0, "mprotect of code to r-x failed");
			Check(gem->BindExternal(va, third, false), "read-only BindExternal failed");

			auto*          rw    = static_cast<uint32_t*>(ReserveGuest(third));
			const uint64_t rw_va = reinterpret_cast<uint64_t>(rw);
			Check(gem->MapView(rw_va, third, 0x80000, PROT_READ | PROT_WRITE),
			      "MapView for the read-only shader's output failed");
			Check(gem->BindView(rw_va, third, 0x80000), "BindView for the output failed");
			gem->PromoteGuestRange(rw_va, third);
			std::memset(rw, 0, third);

			// The read-only code BO and the output chunk are both resident via CreatedHandles(); the
			// code is fetched from its guest VA, so RunStore takes a handle-less code buffer at `va`.
			const std::vector<uint32_t> all = gem->CreatedHandles();
			const Buffer                code_ro {0, nullptr, va};
			RunStore(fd, ctx, ib, code_ro, rw_va, 0x50000000u, static_cast<uint32_t>(third / 256),
			         all.data(), static_cast<uint32_t>(all.size()));
			for (uint32_t i = 0; i < third / 4; i++) {
				Check(rw[i] == 0x50000000u + i, "GPU did not execute code from a read-only userptr");
			}
			Check(gem->UnbindExternal(va, third), "read-only UnbindExternal failed");
			Check(gem->UnbindView(rw_va, third), "output UnbindView failed");
		}
		::munmap(host, len);
		std::printf("NativeGpuGemBackingTests: GPU wrote through an external (non-backed) guest VA "
		            "and survived a straddling unbind\n");
	}

	// Case 6: a non-backed range with an unreadable hole in the middle, as program memory has (the
	// shader is a readable slice with an execute-only tail). BindExternal must skip the hole — a
	// userptr over it cannot be faulted in and would fail the whole submission — and bind the two
	// readable slices, which the GPU then uses.
	{
		const int fd = gem->DrmFd();
		union drm_amdgpu_ctx c;
		std::memset(&c, 0, sizeof(c));
		c.in.op = AMDGPU_CTX_OP_ALLOC_CTX;
		Check(Ioctl(fd, DRM_IOCTL_AMDGPU_CTX, &c) == 0, "hole-case context allocation failed");
		const uint32_t ctx = c.out.alloc.ctx_id;

		const Buffer ib   = CreateBuffer(fd, 64 * 1024, 0x0084000000ull);
		const Buffer code = CreateBuffer(fd, 64 * 1024, 0x0085000000ull);
		std::memcpy(code.cpu, kStoreShader, sizeof(kStoreShader));

		const uint64_t page = 0x10000;
		const uint64_t len  = page * 3;
		void* host = ::mmap(nullptr, len, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
		Check(host != MAP_FAILED, "hole-case host mapping failed");
		const uint64_t va = reinterpret_cast<uint64_t>(host);
		Check(::mprotect(reinterpret_cast<void*>(va + page), page, PROT_NONE) == 0,
		      "punching the hole failed");

		const size_t before = gem->CreatedHandles().size();
		Check(gem->BindExternal(va, len, true), "BindExternal over a holed range failed");
		Check(gem->CreatedHandles().size() == before + 2,
		      "a middle hole must leave two readable slices bound");

		auto* w0 = static_cast<uint32_t*>(host); // first readable slice
		std::memset(w0, 0, page);
		const std::vector<uint32_t> extra = gem->CreatedHandles();
		RunStore(fd, ctx, ib, code, va, 0x60000000u, static_cast<uint32_t>(page / 256), extra.data(),
		         static_cast<uint32_t>(extra.size()));
		for (uint32_t i = 0; i < page / 4; i++) {
			Check(w0[i] == 0x60000000u + i, "GPU write to the first readable slice failed");
		}
		Check(gem->UnbindExternal(va, len), "hole-case UnbindExternal failed");
		Check(gem->CreatedHandles().size() == before, "unbind must release both readable slices");
		::munmap(host, len);
		std::printf("NativeGpuGemBackingTests: BindExternal split a holed range into readable slices "
		            "the GPU used\n");
	}

	delete gem;

	// Case 7: the exact model (small 1 MiB blocks here). Blocks are always-valid buffers, resident from
	// the moment a GPU-visible view maps them (memory.cpp calls MapGpuVisible for those), and the
	// GPU reaches them with none of them listed. Views at page offsets inside a block map the buffer
	// through the DRM file (mmap + mremap). A display range gets an exportable buffer of its own.
	{
		const uint64_t block = 1ull << 20;
		::setenv("KYTY_NATIVE_GPU_MEM_BUDGET_MIB", "1", 1); // one block, to see a deferred one below
		GemBacking*    ex    = GemBacking::Create(4 * block, device, GemBacking::Model::Exact, block);
		::unsetenv("KYTY_NATIVE_GPU_MEM_BUDGET_MIB");
		Check(ex != nullptr && ex->Exact() && ex->BlockSize() == block, "exact-model Create failed");
		auto* ex_alias = ex->AliasBase();

		// A CPU-only view: stays in the memfd, nothing resident.
		const uint64_t cpu_off = 0x3000, cpu_len = 0x5000;
		auto*          cpu_view = static_cast<uint32_t*>(ReserveGuest(cpu_len));
		Check(ex->MapView(reinterpret_cast<uint64_t>(cpu_view), cpu_len, cpu_off, PROT_READ | PROT_WRITE),
		      "exact: MapView of a CPU-only view failed");
		Check(ex->BindView(reinterpret_cast<uint64_t>(cpu_view), cpu_len, cpu_off), "exact: BindView of it failed");
		for (uint32_t i = 0; i < cpu_len / 4; i++) {
			cpu_view[i] = 0x31000000u + i;
		}
		Check(!ex->Lookup(cpu_off, nullptr, nullptr) && ex->ResidentBytes() == 0, "exact: a CPU-only view is not resident");
		Check(ex->PromoteTouched() == 0, "exact: CPU writes promote nothing");

		// A GPU-visible view aliasing it, starting one page earlier: its block becomes resident, the
		// CPU-only view moves onto the buffer with it, and no data is lost.
		const uint64_t off = 0x2000, len = 0x10000;
		auto*          view = static_cast<uint32_t*>(ReserveGuest(len));
		const uint64_t va   = reinterpret_cast<uint64_t>(view);
		Check(ex->MapView(va, len, off, PROT_READ | PROT_WRITE) && ex->BindView(va, len, off),
		      "exact: MapView/BindView of a GPU-visible view failed");
		Check(ex->MapGpuVisible(va, len) == 1 && ex->ResidentBytes() == block, "exact: one block made resident");
		uint32_t handle = 0;
		Check(ex->Lookup(off, &handle, nullptr) && handle != 0, "exact: the block resolves to a buffer");
		for (uint32_t i = 0; i < cpu_len / 4; i++) {
			Check(cpu_view[i] == 0x31000000u + i && view[(cpu_off - off) / 4 + i] == 0x31000000u + i,
			      "exact: promotion lost the CPU-only view's data");
		}
		view[0] = 0xfeedf00du;
		Check(reinterpret_cast<const uint32_t*>(ex_alias + off)[0] == 0xfeedf00du, "exact: alias does not see a view write");

		// Past the budget a GPU-visible block is deferred: lazy residency, here through a CPU write.
		{
			auto* late = static_cast<uint32_t*>(ReserveGuest(0x4000));
			const uint64_t late_va = reinterpret_cast<uint64_t>(late);
			Check(ex->MapView(late_va, 0x4000, 2 * block, PROT_READ | PROT_WRITE) && ex->BindView(late_va, 0x4000, 2 * block),
			      "exact: MapView past the budget failed");
			Check(!ex->LazyResidency() && ex->MapGpuVisible(late_va, 0x4000) == 0 && ex->LazyResidency(),
			      "exact: a block past the budget must be deferred");
			late[0] = 1;
			Check(ex->PromoteTouched() == 1 && !ex->LazyResidency() && late[0] == 1,
			      "exact: a deferred block becomes resident when written");
		}

				// The GPU writes the view with only the dummy listed (CPU-only views of other blocks map it).
		const int      fd = ex->DrmFd();
		union drm_amdgpu_ctx c;
		std::memset(&c, 0, sizeof(c));
		c.in.op = AMDGPU_CTX_OP_ALLOC_CTX;
		Check(Ioctl(fd, DRM_IOCTL_AMDGPU_CTX, &c) == 0, "exact: context allocation failed");
		const Buffer         ib   = CreateBuffer(fd, 64 * 1024, 0x0080000000ull);
		const Buffer         code = CreateBuffer(fd, 64 * 1024, 0x0081000000ull);
		std::memcpy(code.cpu, kStoreShader, sizeof(kStoreShader));
		std::vector<uint32_t> listed = ex->CreatedHandles();
		Check(listed.size() <= 1, "exact: resident blocks must not be listed");
		RunStore(fd, c.out.alloc.ctx_id, ib, code, va, 0x44000000u, static_cast<uint32_t>(len / 256), listed.data(),
		         static_cast<uint32_t>(listed.size()));
		for (uint32_t i = 0; i < len / 4; i++) {
			Check(view[i] == 0x44000000u + i, "exact: GPU write not visible in the view");
		}
		Check(cpu_view[0] == 0x44000000u + (cpu_off - off) / 4, "exact: GPU write not visible in the aliasing view");

		// A display range inside the view gets its own exportable buffer, with its data, and the GPU
		// and every view follow it there.
		const uint64_t disp = va + 0x4000, disp_len = 0x8000;
		Check(ex->ExportRange(disp, disp_len), "exact: ExportRange failed");
		Check(ex->ExportRange(disp, disp_len), "exact: ExportRange of an exported range must succeed");
		std::vector<GemBacking::DmabufPiece> pieces;
		Check(ex->DmabufPieces(disp, disp_len, pieces) && pieces.size() == 1 && pieces[0].len == disp_len,
		      "exact: an exported range is one dma-buf piece");
		auto* exported = static_cast<uint32_t*>(
		    ::mmap(nullptr, pieces[0].bo_size, PROT_READ | PROT_WRITE, MAP_SHARED, pieces[0].fd, 0));
		Check(exported != MAP_FAILED, "exact: mmap of the exported dma-buf failed");
		Check(exported[pieces[0].bo_offset / 4] == 0x44000000u + 0x4000 / 4, "exact: export lost the range's data");
		Check(!ex->DmabufPieces(va, len, pieces), "exact: the rest of the view is not exportable");
		listed = ex->CreatedHandles();
		RunStore(fd, c.out.alloc.ctx_id, ib, code, va, 0x55000000u, static_cast<uint32_t>(len / 256), listed.data(),
		         static_cast<uint32_t>(listed.size()));
		for (uint32_t i = 0; i < len / 4; i++) {
			Check(view[i] == 0x55000000u + i, "exact: GPU write after the export not visible in the view");
		}
		Check(exported[pieces[0].bo_offset / 4] == 0x55000000u + 0x4000 / 4, "exact: GPU write missed the exported buffer");
		::munmap(exported, pieces[0].bo_size);

		// An external (non-direct) range moves into an always-valid buffer: its data and protection
		// survive, the GPU writes it with nothing listed, a protection change keeps the buffer, and a
		// partial unbind keeps the rest bound.
		const uint64_t ext_len = 0x40000;
		auto*          host    = static_cast<uint32_t*>(
		    ::mmap(nullptr, ext_len, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
		Check(host != MAP_FAILED, "exact: external host range allocation failed");
		for (uint32_t i = 0; i < ext_len / 4; i++) {
			host[i] = 0x66000000u + i;
		}
		const uint64_t ext_va = reinterpret_cast<uint64_t>(host);
		::mprotect(host + 0x10000 / 4, 0x10000, PROT_NONE); // a no-access hole inside the range
		const size_t listed_before = ex->CreatedHandles().size();
		Check(ex->BindExternal(ext_va, ext_len, true), "exact: BindExternal failed");
		Check(ex->CreatedHandles().size() == listed_before, "exact: an external buffer must not be listed");
		::mprotect(host + 0x10000 / 4, 0x10000, PROT_READ | PROT_WRITE);
		for (uint32_t i = 0; i < ext_len / 4; i++) {
			Check(host[i] == 0x66000000u + i, "exact: moving an external range lost its data");
		}
		Check(ex->BindExternal(ext_va, ext_len, true), "exact: re-binding a moved range must succeed");
		listed = ex->CreatedHandles();
		RunStore(fd, c.out.alloc.ctx_id, ib, code, ext_va, 0x77000000u, static_cast<uint32_t>(ext_len / 256),
		         listed.data(), static_cast<uint32_t>(listed.size()));
		for (uint32_t i = 0; i < ext_len / 4; i++) {
			Check(host[i] == 0x77000000u + i, "exact: GPU write to an external range not visible");
		}
		Check(ex->UnbindExternal(ext_va, 0x10000), "exact: partial UnbindExternal failed");
		Check(ex->ExternalBytes() == ext_len - 0x10000, "exact: the rest of the range stays bound");
		RunStore(fd, c.out.alloc.ctx_id, ib, code, ext_va + 0x10000, 0x78000000u, static_cast<uint32_t>((ext_len - 0x10000) / 256),
		         listed.data(), static_cast<uint32_t>(listed.size()));
		Check(host[0x10000 / 4] == 0x78000000u && host[ext_len / 4 - 1] == 0x78000000u + (ext_len - 0x10000) / 4 - 1,
		      "exact: GPU write to the surviving part not visible");
		Check(ex->UnbindExternal(ext_va, ext_len) && ex->ExternalBytes() == 0, "exact: UnbindExternal failed");
		::munmap(host, ext_len);
		delete ex;
		std::puts("NativeGpuGemBackingTests: exact model: always-valid blocks, offset views, export, external ranges");
	}

	std::puts("NativeGpuGemBackingTests: all cases passed");
	return 0;
}

#endif // KYTY_GEM_BACKING_SUPPORTED
