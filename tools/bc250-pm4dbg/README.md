# bc250-pm4dbg

Observability for Kyty's native GPU backend on the BC-250: a PM4 decoder that shows the full register
state at any draw or dispatch, and a GPU timing report that says where each frame's time goes.

## Capturing

All switches are off by default and only act in the native path (`KYTY_NATIVE_GPU_PRIMARY=1` or
shadow mode). The timing trace needs the pipeline (`KYTY_NATIVE_GPU_THREAD`, on in primary mode).

| env | |
|---|---|
| `KYTY_NATIVE_GPU_CAPTURE_DIR=<dir>` | where `trace.bin` / `pm4.bin` go |
| `KYTY_NATIVE_GPU_CAPTURE_AT=<s>` | open the capture window this many seconds after the first submission (default 0) |
| `KYTY_NATIVE_GPU_PROFILE_FLIPS=<n>` | `trace.bin`: GPU timestamps of every DCB/ACB for n flips |
| `KYTY_NATIVE_GPU_PROFILE_DRAWS=1` | ... and of every draw and dispatch in them |
| `KYTY_NATIVE_GPU_PM4_DUMP_FLIPS=<n>` | `pm4.bin`: guest DCBs/ACBs, their IBs and the tables their user data points at, for n flips |
| `KYTY_NATIVE_GPU_DRAW_MARKERS=1` | before each draw/dispatch a `WRITE_DATA` of (buffer, item), after it a bottom-of-pipe `RELEASE_MEM` of the same; logged as `MARKERS` when a submission times out |

Example (ASTRO BOT gameplay, five minutes in):

```
tools/bc250-runs/try.sh prof 330 "KYTY_NATIVE_GPU_CAPTURE_DIR=/home/deck/dev/kyty-runs/prof KYTY_NATIVE_GPU_CAPTURE_AT=290 KYTY_NATIVE_GPU_PROFILE_FLIPS=600"
```

How it works:

- Every buffer in the window is bracketed in its IB by a `COPY_DATA` of the GPU clock (when the CP
  starts it) and a `RELEASE_MEM BOTTOM_OF_PIPE_TS` (when it and everything before it is done), into
  slots of a profile buffer (`NativeSubmitter::CreateProfileBuffer`, cached GTT in every BO list).
  With `PROFILE_DRAWS` the translator does the same around each draw/dispatch
  (`Pm4Translator::SetProfile`). End-of-pipe timestamps retire in order, so an item's cost is how much
  later it finished than everything before it: its share of the critical path. These packets do not
  stall the CP, but per-draw timestamps add an end-of-pipe event per draw.
- The completion thread reads the slots once a batch's fence has signalled and writes them with the
  CPU times of the pipeline (guest submit, translated, CS ioctl, fence seen, Kyty's command processor
  done). Each flip also writes a calibration pair (`AMDGPU_INFO_TIMESTAMP` between two
  `CLOCK_MONOTONIC` reads), so GPU and CPU times share one timeline.
- The PM4 dump takes the IB exactly as submitted (preamble with the full shadowed state, register
  loads expanded, timestamps included) plus the raw guest buffer, and 512 bytes of guest memory at
  every pointer in shader user data and at `SET_BASE`, then at every pointer in those, taken right
  before the GPU runs the buffer.

## Reading

```
pm4dbg.py profile trace.bin [--top 25] [--frames]
pm4dbg.py list    pm4.bin
pm4dbg.py draws   pm4.bin -b 1234
pm4dbg.py state   pm4.bin -b 1234 -i 17 [--all]
pm4dbg.py packets pm4.bin -b 1234 [--raw] [--regs]
pm4dbg.py table   pm4.bin -b 1234 0x2001234000
pm4dbg.py find    pm4.bin --shader 0x9001aec000
```

`-b` is the buffer number the shadow log uses (`frame 1234`), `acb:56` for an ACB, or `#7` for the
7th buffer in the file. Items count draws and dispatches together in IB order, the same numbering as
the trace and the markers.

`profile` splits each flip-to-flip interval into GPU busy time (DCB, ACB) and idle gaps, and assigns
each gap to what the buffer that ended it was doing on the CPU side: not yet submitted by the game,
queued or being translated, waiting for its batch, in the CS ioctl, or submitted but not yet started
by the GPU (scheduler, other processes on the gfx ring). Gaps between two buffers of one batch are
CP-side. It also lists the latencies around the pipeline, the longest gaps and buffers, and with
per-draw timestamps the costliest shaders and items, separately for DCB and ACB.

`state` replays the IB up to the item and prints the shaders of the active stages (address,
RSRC1/RSRC2 fields), their user data (each SGPR, as hex and float, pointers marked), the V#/T#
descriptors found inline and in the dumped tables, render targets and depth with their CB/DB fields,
viewport and scissors; `--all` adds every context, SH and UCONFIG register with decoded fields. The
descriptor scan is heuristic: `table` shows every 16 bytes of a table as V#, T# and S#.

Register names and fields come from `regdb.json`, generated from the amdgpu headers
(`gc_10_1_0_offset.h`, `gc_10_1_0_sh_mask.h`) with `gen_regdb.py HEADER_DIR > regdb.json`.

## Files

Both files are records of `u32 type, u32 payload bytes, payload` (little endian); the layouts are in
`src/graphics/native_gpu/nativeShadow.cpp` (`TraceType`) and `pm4dbg.py` parses them. Times are
`CLOCK_MONOTONIC` ns, GPU values are GPU clock ticks (100 MHz on the BC-250).
