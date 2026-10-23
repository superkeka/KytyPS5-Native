# bc250-frame-diff

Frame-by-frame comparison of the native GPU path against Kyty's Vulkan path, for the same guest
flips. Use it to find out where native rendering goes wrong.

In shadow mode every guest DCB runs natively first, and then Kyty renders it with Vulkan. At each
flip listed in `KYTY_NATIVE_GPU_DUMP_FLIPS`, the emulator does three things:

1. Saves the guest display surface as the native path left it (`flip_<n>.native.raw`).
2. Writes Kyty's Vulkan copy of the same surface back into guest memory and saves that too
   (`flip_<n>.vulkan.raw`).
3. Records the surface layout (`flip_<n>.txt`).

Both images come from the same command stream, so any difference is a bug in one of the two
paths.

## Workflow

On the board:

```
cd /home/deck/dev/kyty-runs
/home/deck/dev/KytyPS5-bc250/tools/bc250-frame-diff/dc-framediff.sh dc-framediff-N    # ~4 min
python3 -I /home/deck/dev/KytyPS5-bc250/tools/bc250-frame-diff/frame_diff.py dc-framediff-N
```

- `frame_diff.py` writes `dc-framediff-N/diff/report.md`. For each flip it gives:
  - whether the flip is identical;
  - how many 64 KiB tiles differ;
  - the mean and max channel difference;
  - the share of pixels above the threshold;
  - the bounding box of the differences.
- It also writes quarter-resolution `flip_<n>.{native,vulkan,diff}.png`. In the diff image,
  differing pixels are red, amplified 8×, over the dimmed native frame.
- To zoom in on a region, pass the box from the report:

  ```
  frame_diff.py DIR --flips 1500 --step 1 --crop 1640,780,2080,1300
  ```

  This writes full-resolution `flip_<n>.crop.*.png`.
- For other games, set the env vars by hand:

  ```
  KYTY_NATIVE_GPU_GEM=1 KYTY_NATIVE_GPU_SHADOW=1 KYTY_NATIVE_GPU_DUMP_FLIPS=300-3000/50 KYTY_NATIVE_GPU_DUMP_DIR=...
  ```

- To pass extra variables to the script, use `EXTRA_ENV="KYTY_NATIVE_GPU_STATS=1"`.

The run log lists the native path's per-frame counters next to the flips. The `waits unmet/dropped`
counter is the first thing to correlate with a diverging frame.

Each dumped flip takes about 66 MB. Delete the `.raw` files once the report is done.

Supported surfaces: 32 bpp, linear or SW_64KB_R_X (render target).

## Findings (Dead Cells, 2026-10-23)

- **Menu flips:** identical, or under 0.03% pixels differing (single-pixel edges).
- **Gameplay flips:** about 2% of pixels differ. Two areas are affected:
  - A rectangular chunk of the floor-edge brick tiles is missing in the native frame.
  - The player sprite shows a different animation frame.

  Every gameplay frame has 17–24 WAIT_REG_MEM packets that are unmet when the native path drops
  them. The native draws run before the data those waits guard is written.
