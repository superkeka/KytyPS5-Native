# Streamed DCBs: the GPU starts while the translation goes on

How the native backend (`src/graphics/native_gpu`) gets a title's graphics command buffer (DCB) onto
the gfx ring before the whole of it is translated, why it matters, and what each part does. On by
default in primary mode.

## Why

On the PS5 a title hands its DCB to the GPU and the CP starts on it at once. Here every DCB is
translated first (`Pm4Translator::Translate`), and until this change the whole result went to the GPU
as one CS only after the translation ended. In Gran Turismo 7 a DCB is ~100K dwords; its translation
took 2–3.5 ms a frame, during which the gfx ring idled.

That delay is what made GT7 stutter. The flip trace (`KYTY_NATIVE_GPU_FLIP_TRACE`, columns
`dcb_latency_ms`, `game_gap_ms`, `acb_gap_ms`) and the guest wait log (`KYTY_LONG_WAITS`,
`KYTY_LOG_SUBMITS`) show GT7's frame loop:

1. the GPU finishes the DCB; the title's `PGL Event` thread gets the end-of-pipe event (filter -14);
2. it wakes `GPUex`, which flips (`Flipx`) and submits the next DCB and its ACBs;
3. `PGL Event` also runs two high-resolution timers (~8 ms apart) and follows vblank (filter -13).

When the end-of-frame event lands in the wrong window relative to vblank and those timers, the title
does not submit the next DCB at once but waits for the next vblank: a frame is lost (an interval of
~33 ms). The later the event, the more often that happens. Neither the CPU clock, the GPU clock, the
thread count (8C/16T), the presenter copy nor the tracker's interrupt path changed the rate; starting
the GPU earlier did.

GT7 race on the stock amdgpu, frames over 20 ms:

| | over 20 ms |
|---|---|
| whole DCB, one CS | 2.0–3.3 % |
| chunks, a CS each with a full preamble | 1.19 % (GPU +2 ms a frame) |
| streamed, 16K-dword chunks | 1.68 % |
| streamed, the title's segments | 1.49 % |

The streamed forms cost little GPU time. Stray, which turned the camera at ~50 fps before, holds
60 fps with them (frame 16.7 ms instead of 17.9).

## Where a DCB is cut

`LabelPipeline::TranslateChunks` inlines the DCB's INDIRECT_BUFFER calls and chains
(`FlattenIndirectBuffers`) and cuts the inlined stream:

- **The title's segments** (`KYTY_NATIVE_GPU_CHUNK=guest`, the default): `FlattenIndirectBuffers`
  returns `marks`, where each buffer the top level calls or chains to begins and ends. GT7's DCB is 25
  dwords of such calls; its command buffer is the buffers libSceAgc wrote and linked. The cut goes
  there: the title itself breaks its stream at those points, so a cut never lands inside one of its
  sequences. Segments under 1024 dwords join the next one. A DCB without calls (ASTRO BOT's are flat) is
  not cut at all.
- **A fixed size** (`KYTY_NATIVE_GPU_CHUNK_DW=N`): every N dwords at a packet boundary.

Neither cuts inside a COND_EXEC's range. A buffer that cannot be inlined yet (a REWIND the title has
not validated) is not chunked: it goes the usual way, which inlines it again itself.

`KYTY_NATIVE_GPU_CHUNK=off` turns chunking off.

Each chunk is translated as a buffer of its own; the translator's shadow carries the register state
from one to the next.

## The stream

The chunks of one DCB run as one IB (`Shadow::StreamBegin`, `StreamFlushLocked`):

```
head CS:   [chunk 0] [WAIT_REG_MEM counter >= 1 (PFP)] [CHAIN -> trampoline 1, 8 dw]
slot A:    [WRITE_DATA done = 1] [chunk 1] [WAIT counter >= 2] [CHAIN -> trampoline 2]
slot B:    [WRITE_DATA done = 2] [chunk 2]                       (last: no wait, no chain)
trampoline n (scratch page): [INDIRECT_BUFFER CHAIN -> slot of chunk n, exact size] [NOP]...
```

1. The first chunk goes in an ordinary CS that ends with a WAIT_REG_MEM on a counter in the scratch page
   (uncached, so the CPU's store reaches the CP) and a chain to trampoline 1. The wait runs on the PFP,
   which parses nothing past it until the counter is raised.
2. Each later chunk is copied to a free IB slot (`NativeSubmitter::WriteStreamIb`; it never waits for a
   busy slot: the submission still running may be the very one whose CP waits for this chunk). Its
   trampoline gets a chain to it with the exact size, then the counter is raised. The CP, already
   waiting, goes on.
3. The last chunk ends without a wait: its trampoline chains to it, the CP returns to the ring, and the
   head CS's fence follows.

The CP thus sees one continuous stream: no preamble between chunks, no context switch, no new job.
The slots stay taken until the head CS completes.

Why trampolines: an INDIRECT_BUFFER packet carries the size of its target, and the size of chunk n is
known only once it is translated. The chain at the end of chunk n-1 goes to a fixed 8-dword
trampoline, and the trampoline, written when chunk n is ready, carries the exact size.

### Rules that keep it from hanging

Each was a hang before it was a rule.

- **Preamble from the start state.** `Translate` builds a buffer's preamble from the committed shadow,
  i.e. the state where the buffer *ends*. For a whole DCB that is harmless (the title sets what it needs
  at the start of a frame); a chunk's first draws continue the previous chunk's pass and got the targets
  and formats of its last draws. GT7 hung on its first chunked frame with vertex waves stuck exporting.
  Chunks take their preamble from the start state (`SetPreambleFromStart`); streamed chunks after the
  first have none at all (`SetNoPreamble`).
- **No stale source pieces.** A buffer inlined in `TranslateChunks` and inlined again in `Submit` would
  run its direct runs (IB2s into guest memory) against the first inlining's layout. `Submit` drops a
  `source` once it inlines the buffer itself.
- **A trampoline is reused only once the CP is past it.** There are 60 trampolines; the translator runs
  far ahead of the GPU (Stray has ~20 DCBs a flip). Each streamed chunk starts with a PFP WRITE_DATA of
  its number, and `PublishStreamLocked` waits until the CP has entered the chunk that last used the
  trampoline it is about to rewrite.
- **One stream at a time.** A stream still open when the next chunked DCB of the same round begins is
  ended first. With two open chains, the first CP followed a trampoline nobody had written yet (the PFP
  then parsed zeros).
- **Never block while the CP waits for us.** A submission that would wait for a busy slot while a stream
  is open ends the stream first (`NativeSubmitter::SetBeforeBlockingWait`). A compute IB took a slot of
  the stream, waited for the stream's head to finish, and the head waited for a chunk that came only
  after that submission.
- **A full gfx batch during a stream is the stream's next part**, not a CS of its own queued behind it.
- **The chain always ends.** The worker flushes once more after its pump loop if a stream is still
  open, so the CP is never left waiting.

When something does not fit (no free slot, a chunk longer than a slot, a buffer created since the head
CS and so missing from its BO list), the chain is ended there and the rest goes as ordinary CSs.

## Switches and diagnostics

| | |
|---|---|
| `KYTY_NATIVE_GPU_CHUNK=guest\|off` | cut at the title's segments (default in primary mode) or not at all |
| `KYTY_NATIVE_GPU_CHUNK_DW=N` | cut every N dwords instead |
| `KYTY_NATIVE_GPU_CHUNK_STREAM=0` | a CS per chunk, each with a preamble, instead of the stream |
| `KYTY_NATIVE_GPU_CHUNK_COMPARE=1` | dumps the first chunked DCB translated whole and in chunks to `KYTY_NATIVE_GPU_DUMP_DIR` |
| `KYTY_NATIVE_GPU_FLIP_STATS=1` | per-flip report: `graphics buffers per flip` (guest size, chunked, queued), `streamed DCB chunks so far` (chains ended early, waits for a trampoline, chains ended for a slot wait) |
| `KYTY_NATIVE_GPU_FLIP_TRACE=1` | `flips.csv` with `dcb_latency_ms` (title's submission to the GPU's end), `game_gap_ms` (that to its next DCB), `acb_gap_ms`, `acb_latency_max_ms`, `irq_timeouts` |
| `KYTY_LOG_SUBMITS=1`, `KYTY_LONG_WAITS=<ms>` | the title's DCB/ACB submissions and blocking waits with thread names and CLOCK_MONOTONIC times |

`KYTY_NATIVE_GPU_FLIP_TRACE=gpu` reads the GPU clock with COPY_DATA around each span. With GFXOFF on,
such a read on a ring that idles stops the CP, and a stream idles while it waits for the next chunk:
use `=1` with the stream.

A hang in a stream shows in the CP registers (see `native-gpu-hang-debugging.md`; `umr -f navi10`
reads them on the BC-250): `CP_IB1_BASE` at `0x400003fef840 + 32 * (n % 60)` is trampoline n
(`kIbVa + kIbSize - kScratchSize + 2112`), and a PFP header of `0xc0053c00` followed by `0xc0023f00`
is the stream's wait and chain.

## What it does not do yet

Each chunk is a call of the translator of its own: inlining, residency, a pump. With GT7's ~15
segments a frame the translation went from 2.6 to 3.8 ms. A translator that hands its output to the
stream as it goes, in one pass over the DCB, would start the GPU after the first packets with no
per-chunk cost. The mechanism above is what it would feed.
