# bc250-replay

Captures one frame of the BC-250 native backend and replays it on the GPU without the game, stopping
after any draw or dispatch and reading any image back through the texture unit (so DCC is decoded the
way the game's sampler decodes it), into a PNG plus its numbers.

## Capture (in the emulator)

```sh
KYTY_NATIVE_GPU_CAPTURE=3100            # flip number, as in KYTY_NATIVE_GPU_DUMP_FLIPS
KYTY_NATIVE_GPU_CAPTURE_FRAMES=2        # start that many flips earlier (default 2)
KYTY_NATIVE_GPU_CAPTURE_DIR=/home/deck/dev/kyty-runs/cap   # default KYTY_NATIVE_GPU_DUMP_DIR, then .
```

From flip `N - FRAMES` until flip `N` is presented, every CS the shadow path submits is captured into
`capture_<N>/`. Each CS then runs synchronously; right before it, with the GPU idle, every page the GPU
VM maps (resident direct-memory blocks, the dummy, external ranges, the submitter's scratch, GDS) is
compared with the previous point and the changed pages are saved (`delta_<n>.bin`); the first point
saves all memory (sparse files in `mem/`, several GiB for ASTRO BOT). The format is described in
`src/graphics/native_gpu/nativeCapture.h`. The game runs at a few seconds per frame while capturing.

## Replay (on the board, one GPU job at a time)

`bc250_replay` is built with the emulator (`_Build/linux-no-qt/bc250_replay`). Run it through
`replay.sh`, which goes through `tools/bc250-runs/try.sh` (shared lock, hang recovery) and copies the
output back:

```sh
tools/bc250-replay/replay.sh rp1 /home/deck/dev/kyty-runs/cap/capture_3100 --list
tools/bc250-replay/replay.sh rp2 /home/deck/dev/kyty-runs/cap/capture_3100 --at 400-420 --verify
```

`--list` needs no GPU (run the binary directly). Before each CS the replay maps the same GPU VAs, applies
the pages captured for that CS, restores the scratch area and GDS, invalidates the caches, then runs the
IB as it went to the ring. A step splits a CS at a draw: the head runs to the end of that packet, the
readbacks run, and the rest runs as its own submission with every state packet of the head in front of
it (only work before the split is dropped), so the state is the same as in one piece.

Options:

| Option | |
|---|---|
| `--list` | CSs, guest buffers, every draw/dispatch with its global index, shader and the targets it writes |
| `--at STEPS` | read back after these draws/dispatches: `N`, `A-B`, `A-B/S`, comma lists; `cs:N` after CS N; `end` (default) |
| `--rt SLOTS` | color targets per step: `auto` (what the draw writes), `all`, `none`, `0,2` — T# built from the CB registers |
| `--tsharp D0,...,D7` | an image through this T# |
| `--tsharp-at VA` | through the T# stored at VA (in replay memory at that step) |
| `--tsharp-ud ps:R[+OFF]` | through the T# in the table that user data R/R+1 of the stage (ps, gs, cs) point to; `ps:Ri` for a T# in the user data itself |
| `--find-tsharp VA` | the game's own T#s of the image at VA, found in memory (use one with `--tsharp`) |
| `--dump VA:SIZE` | raw replay memory at each step |
| `--slice`, `--mip`, `--crop X,Y,W,H` | what part of the image to read |
| `--scale F`, `--srgb`, `--alpha` | PNG mapping of float/integer formats; alpha is forced opaque unless `--alpha` |
| `--raw` | also the RGBA32 texels (`.rgba32`) |
| `--verify` | before each CS, how many captured changed pages the replay already reproduced (and by region) |
| `--no-deltas` | apply only the first point: later CSs see only what the replayed GPU work left |

Each readback writes `<step>_<what>_<base>.png` and `.txt` (the T#, the format, per-channel min/max,
non-finite count). Float formats (FP16 and others) go through the texture unit as floats, so `--scale`
brings HDR values into range.

## Checked on ASTRO BOT (PPSA21567)

`capture.sh cap1 3100` during the memory card animation after the second X: 3 CS for flips
3098-3100 (the pipeline batches a whole frame into one CS, 736 draws and dispatches), 6.3 GiB saved in
full plus 2.5 GiB of changed pages, 21 s of capture, 8.7 GiB on disk. The replay's display buffer after
the last CS is byte-identical to `flip_3100.native.raw` dumped by the game (also with the CS cut into
eight steps, and with `--no-deltas`). Every intermediate target of that frame is FP16 with DCC
(16_16_16_16 float, 16_16 float); their T#s built from the CB registers equal the game's own (found
with `--find-tsharp`) dword for dword, and both read back identically.

`readback_shaders.h` is generated from `readback.s` by `gen_shaders.sh` (LLVM `llvm-mc`, gfx1013).
