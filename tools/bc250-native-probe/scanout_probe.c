// Direct scan-out probe for the BC-250: can a PS5 display buffer go to the screen as it is, without
// Kyty's Vulkan copy? It makes two 3840x2160 XRGB8888 buffers laid out as the PS5's render targets
// (Prospero kRenderTarget, SW_64KB_R_X: 128x128 blocks of 64 KiB in rows, Kyty's
// Gen5RenderTargetOffsetInBlock inside each), gives them to Xwayland through DRI3 with the DRM
// modifier the BC-250's display planes list for that layout (AMD GFX10 64K_R_X, PIPE_XOR_BITS 4,
// 0x0200000000801b02), and flips between them with Present, one per vblank. gamescope scans a
// fullscreen client's buffer out directly when it can. On the TV: a colour grid with a white square
// that jumps left and right; a scrambled picture means the layouts differ.
// Prints Present's completion mode (FLIP: the buffer itself went on, COPY: Xwayland copied it) and
// the intervals between completions.
// Usage: scanout_probe [vram|gtt] [seconds]  (DISPLAY=:1 inside gamescope; run scanout-probe.sh)
// Build: cc -O2 -o scanout_probe scanout_probe.c -lxcb -lxcb-dri3 -lxcb-present
#define _GNU_SOURCE
#include <drm/amdgpu_drm.h>
#include <drm/drm.h>
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>
#include <xcb/dri3.h>
#include <xcb/present.h>
#include <xcb/xcb.h>

enum { kWidth = 3840, kHeight = 2160, kBlock = 128 };
static const uint64_t kModifier = 0x0200000000801b02ull; // AMD GFX10, GFX9_64K_R_X, PIPE_XOR_BITS 4

static int drm = -1;

static int Ioctl(unsigned long req, void* arg) {
	int r;
	do {
		r = ioctl(drm, req, arg);
	} while (r == -1 && (errno == EINTR || errno == EAGAIN));
	return r == -1 ? -errno : 0;
}

static double Now(void) {
	struct timespec t;
	clock_gettime(CLOCK_MONOTONIC, &t);
	return t.tv_sec + t.tv_nsec * 1e-9;
}

// Kyty's Gen5RenderTargetOffsetInBlock for 4-byte elements (src/graphics/guest_gpu/tile.cpp).
static uint32_t OffsetInBlock(uint32_t x, uint32_t y) {
	uint32_t o = 0;
	o ^= (y << 4u) & 0x0070u;
	o ^= (y << 5u) & 0x0f00u;
	o ^= (y << 9u) & 0x1000u;
	o ^= (y << 8u) & 0x4000u;
	o ^= (x << 2u) & 0x000cu;
	o ^= (x << 5u) & 0x0380u;
	o ^= (x << 4u) & 0x0400u;
	o ^= (x << 6u) & 0x0800u;
	o ^= (x << 9u) & 0xa000u;
	return o;
}

static uint64_t Address(uint32_t x, uint32_t y) {
	const uint32_t per_row = kWidth / kBlock;
	return ((uint64_t)(y / kBlock) * per_row + x / kBlock) * 65536u + OffsetInBlock(x % kBlock, y % kBlock);
}

// A colour grid: hue by 240-pixel column, brightness by 240-pixel row, white lines every 120
// pixels, a white 240x240 square at `square_x`.
static uint32_t Pixel(uint32_t x, uint32_t y, uint32_t square_x) {
	if (x >= square_x && x < square_x + 240 && y >= 960 && y < 1200) {
		return 0x00ffffffu;
	}
	if (x % 120 == 0 || y % 120 == 0) {
		return 0x00c0c0c0u;
	}
	static const uint32_t hues[] = {0xff0000, 0xff8000, 0xffff00, 0x00ff00, 0x00ffff, 0x0080ff, 0x0000ff, 0x8000ff};
	const uint32_t h = hues[(x / 240) % 8];
	const uint32_t k = 64 + (y / 240) * 20;
	return ((((h >> 16) & 0xff) * k / 255) << 16) | ((((h >> 8) & 0xff) * k / 255) << 8) | ((h & 0xff) * k / 255);
}

struct Buf {
	uint32_t handle;
	uint64_t size;
	void*    cpu;
	int      fd;
};

static int MakeBuffer(struct Buf* b, int vram, uint32_t square_x) {
	const uint32_t rows = (kHeight + kBlock - 1) / kBlock;
	b->size             = (uint64_t)(kWidth / kBlock) * rows * 65536u;
	union drm_amdgpu_gem_create c;
	memset(&c, 0, sizeof(c));
	c.in.bo_size      = b->size;
	c.in.alignment    = 65536;
	c.in.domains      = vram ? AMDGPU_GEM_DOMAIN_VRAM : AMDGPU_GEM_DOMAIN_GTT;
	c.in.domain_flags = AMDGPU_GEM_CREATE_CPU_ACCESS_REQUIRED | (vram ? 0 : AMDGPU_GEM_CREATE_CPU_GTT_USWC);
	int r             = Ioctl(DRM_IOCTL_AMDGPU_GEM_CREATE, &c);
	if (r) {
		fprintf(stderr, "GEM_CREATE: %s\n", strerror(-r));
		return r;
	}
	b->handle = c.out.handle;
	union drm_amdgpu_gem_mmap m;
	memset(&m, 0, sizeof(m));
	m.in.handle = b->handle;
	if ((r = Ioctl(DRM_IOCTL_AMDGPU_GEM_MMAP, &m))) {
		return r;
	}
	b->cpu = mmap(NULL, b->size, PROT_READ | PROT_WRITE, MAP_SHARED, drm, (off_t)m.out.addr_ptr);
	if (b->cpu == MAP_FAILED) {
		return -errno;
	}
	// Built in system memory, then copied in one go (the mapping may be write-combined).
	uint8_t* tmp = calloc(1, b->size);
	for (uint32_t y = 0; y < kHeight; y++) {
		for (uint32_t x = 0; x < kWidth; x++) {
			*(uint32_t*)(tmp + Address(x, y)) = Pixel(x, y, square_x);
		}
	}
	memcpy(b->cpu, tmp, b->size);
	free(tmp);
	struct drm_prime_handle p;
	memset(&p, 0, sizeof(p));
	p.handle = b->handle;
	p.flags  = DRM_CLOEXEC | DRM_RDWR;
	if ((r = Ioctl(DRM_IOCTL_PRIME_HANDLE_TO_FD, &p))) {
		fprintf(stderr, "PRIME export: %s\n", strerror(-r));
		return r;
	}
	b->fd = p.fd;
	return 0;
}

int main(int argc, char** argv) {
	setvbuf(stdout, NULL, _IOLBF, 0);
	const int vram    = argc > 1 ? strcmp(argv[1], "gtt") != 0 : 1;
	const int seconds = argc > 2 ? atoi(argv[2]) : 20;
	drm               = open("/dev/dri/renderD128", O_RDWR | O_CLOEXEC);
	xcb_connection_t* c = xcb_connect(NULL, NULL);
	if (drm < 0 || xcb_connection_has_error(c)) {
		fprintf(stderr, "no render node or X display\n");
		return 1;
	}
	xcb_dri3_query_version_reply_t* dv = xcb_dri3_query_version_reply(c, xcb_dri3_query_version(c, 1, 2), NULL);
	xcb_present_query_version_reply_t* pv = xcb_present_query_version_reply(c, xcb_present_query_version(c, 1, 2), NULL);
	printf("DRI3 %u.%u, Present %u.%u, buffers in %s\n", dv ? dv->major_version : 0, dv ? dv->minor_version : 0,
	       pv ? pv->major_version : 0, pv ? pv->minor_version : 0, vram ? "VRAM" : "GTT (USWC)");
	if (!dv || dv->minor_version < 2) {
		fprintf(stderr, "DRI3 1.2 (modifiers) needed\n");
		return 1;
	}
	xcb_screen_t* screen = xcb_setup_roots_iterator(xcb_get_setup(c)).data;
	printf("X screen %ux%u, root depth %u\n", screen->width_in_pixels, screen->height_in_pixels, screen->root_depth);

	// The window: fullscreen, tagged as a game for gamescope (the wrapper sets the base layer).
	const xcb_window_t win  = xcb_generate_id(c);
	const uint32_t     mask = XCB_CW_BACK_PIXEL | XCB_CW_EVENT_MASK;
	const uint32_t     vals[] = {screen->black_pixel, XCB_EVENT_MASK_STRUCTURE_NOTIFY};
	xcb_create_window(c, XCB_COPY_FROM_PARENT, win, screen->root, 0, 0, kWidth, kHeight, 0, XCB_WINDOW_CLASS_INPUT_OUTPUT,
	                  screen->root_visual, mask, vals);
	xcb_intern_atom_reply_t* game = xcb_intern_atom_reply(c, xcb_intern_atom(c, 0, 10, "STEAM_GAME"), NULL);
	const uint32_t           appid = 424242;
	xcb_change_property(c, XCB_PROP_MODE_REPLACE, win, game->atom, XCB_ATOM_CARDINAL, 32, 1, &appid);
	xcb_change_property(c, XCB_PROP_MODE_REPLACE, win, XCB_ATOM_WM_NAME, XCB_ATOM_STRING, 8, 13, "scanout_probe");
	xcb_map_window(c, win);
	xcb_flush(c);

	struct Buf     bufs[2];
	xcb_pixmap_t   pix[2];
	for (int i = 0; i < 2; i++) {
		if (MakeBuffer(&bufs[i], vram, i == 0 ? 600 : 3000)) {
			return 2;
		}
		pix[i]  = xcb_generate_id(c);
		int fds[1] = {bufs[i].fd};
		xcb_void_cookie_t ck = xcb_dri3_pixmap_from_buffers_checked(c, pix[i], win, 1, kWidth, kHeight, kWidth * 4, 0, 0, 0, 0, 0, 0,
		                                                           0, 24, 32, kModifier, fds);
		xcb_generic_error_t* e = xcb_request_check(c, ck);
		if (e) {
			fprintf(stderr, "PixmapFromBuffers with modifier 0x%llx refused (error %u)\n", (unsigned long long)kModifier, e->error_code);
			return 3;
		}
	}
	printf("two %llu MiB buffers as pixmaps with modifier 0x%llx\n", (unsigned long long)(bufs[0].size >> 20),
	       (unsigned long long)kModifier);

	const uint32_t eid = xcb_generate_id(c);
	xcb_present_select_input(c, eid, win, XCB_PRESENT_EVENT_MASK_COMPLETE_NOTIFY);
	xcb_special_event_t* se = xcb_register_for_special_xge(c, &xcb_present_id, eid, NULL);
	xcb_flush(c);

	uint64_t  last_ust = 0;
	uint32_t  frames = 0, flips = 0, copies = 0, other = 0, late = 0;
	double    sum = 0;
	const double t_end = Now() + seconds;
	for (uint32_t serial = 1; Now() < t_end; serial++) {
		xcb_present_pixmap(c, win, pix[serial & 1], serial, 0, 0, 0, 0, 0, 0, 0, XCB_PRESENT_OPTION_NONE, 0, 0, 0, 0, NULL);
		xcb_flush(c);
		for (;;) {
			xcb_generic_event_t* ev = xcb_wait_for_special_event(c, se);
			if (!ev) {
				fprintf(stderr, "X connection lost\n");
				return 4;
			}
			xcb_present_generic_event_t* g = (xcb_present_generic_event_t*)ev;
			if (g->evtype == XCB_PRESENT_EVENT_COMPLETE_NOTIFY) {
				xcb_present_complete_notify_event_t* cn = (xcb_present_complete_notify_event_t*)ev;
				const int mine = cn->serial == serial;
				if (mine) {
					frames++;
					flips += cn->mode == XCB_PRESENT_COMPLETE_MODE_FLIP;
					copies += cn->mode == XCB_PRESENT_COMPLETE_MODE_COPY;
					other += cn->mode != XCB_PRESENT_COMPLETE_MODE_FLIP && cn->mode != XCB_PRESENT_COMPLETE_MODE_COPY;
					if (last_ust != 0) {
						const double d = (cn->ust - last_ust) * 1e-3;
						sum += d;
						late += d > 20.0;
					}
					last_ust = cn->ust;
					if (frames == 1 || frames % 120 == 0) {
						printf("frame %u: mode %s, msc %llu\n", frames,
						       cn->mode == XCB_PRESENT_COMPLETE_MODE_FLIP ? "FLIP" : cn->mode == XCB_PRESENT_COMPLETE_MODE_COPY ? "COPY" : "other",
						       (unsigned long long)cn->msc);
					}
				}
				free(ev);
				if (mine) {
					break;
				}
			} else {
				free(ev);
			}
		}
	}
	printf("%u frames: %u FLIP, %u COPY, %u other; mean interval %.2f ms, %u over 20 ms\n", frames, flips, copies, other,
	       frames > 1 ? sum / (frames - 1) : 0.0, late);
	return 0;
}
