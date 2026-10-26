# Debugging native GPU hangs on the BC-250

How a hang of the native backend (`src/graphics/native_gpu`) is narrowed down to one packet or one
shader. The board runs with `amdgpu.gpu_recovery=0`: a hung ring stays hung, the board stays on SSH,
and its state can be read. After a capture the board needs a reboot (`sudo -n systemctl reboot`,
allowed without a password for `deck`).

Scripts referred to below are in `tools/bc250-runs/` (copied to `/home/deck/dev/kyty-runs/` on the
board) and `tools/bc250-native-probe/`.

## 1. Which ring

The emulator log says which submission stopped:

```
native shadow: ring slot 1 submission 600 TIMED OUT; disabling native path
native shadow: tracker stuck at 2999 (slot 1) for 12 s; draining
```

Slot 0 is the gfx ring; slot n > 0 is amdgpu compute ring n - 1 (ACBs with
`KYTY_NATIVE_GPU_ACB_RINGS`). At SIGTERM the backend releases the GPU waits it knows about
(`N pending GPU waits released`); `rings STILL BUSY` means the ring waits on something else, or a
shader does not finish.

## 2. Waiting or running: registers without root

`kyty-lodhw` (root helper, socket `/tmp/kyty-lodhw.sock`) reads registers for `kyty-hw.py`:

```sh
python3 kyty-hw.py busy 1     # GRBM_STATUS sampled: CP_BUSY, SPI_BUSY, CB/DB/PA ...
python3 kyty-hw.py cp         # CP_STAT, CP_ME_HEADER_DUMP, CP_PFP_HEADER_DUMP
python3 kyty-hw.py read 0x2084 3   # CP_CPC_STATUS, CP_CPC_BUSY_STAT, CP_CPC_STALLED_STAT1 (MEC)
python3 kyty-hw.py read 0x208e 2   # CP_MEC_ME1/ME2_HEADER_DUMP
```

Absolute offsets are the gc_10_1_0 offsets plus 0x1260 (BASE_IDX 0) or 0xA000 (BASE_IDX 1).
`mec-watch.sh OUT` waits for `TIMED OUT` in the log and samples these six times.

- CP busy, SPI idle, a PFP or ME header of `0xc0053c00` (WAIT_REG_MEM): the ring waits on memory.
  Find the wait (section 3).
- CP and SPI busy, CB/DB idle: a shader does not finish. Dump the waves (section 4).

## 3. Breadcrumbs: what the ring was given

```
KYTY_NATIVE_GPU_BREADCRUMBS=/home/deck/dev/kyty-runs/gt-crumbs KYTY_NATIVE_GPU_BREADCRUMB_SLOTS=32
```

Every submission writes its guest buffer (`dcb_N.bin`, `acbdcb_N.bin`) and the translated IB
(`ib_N.bin`, `acbib_N.bin`) to a ring of N slots; `last.txt` / `acblast.txt` name the newest slot.
Files are `fdatasync`ed, so they survive a hard reset; frames get several times slower, which can
move a timing-dependent hang.

- `tools/bc250-native-probe/ib_dump.py FILE` lists draws, dispatches, DMA, waits.
- For a stuck wait: list every WAIT_REG_MEM (address, function, reference) in the newest IBs of each
  ring, then search all buffers for the packet that writes that address (RELEASE_MEM, WRITE_DATA).
- Guest labels can be read from the hung process: `/proc/PID/mem` at the guest address (guest VA ==
  host VA). Scratch addresses (`0x4000_01ff_f000` page: tracker slots at 1040 + 16 n, barrier
  fences) belong to the backend.

## 4. Waves and page faults (root)

`bc250-umr-nopasswd.sh` (once, with the password) lets `deck` run `umr` without one. umr's
cyan_skillfish database lacks SQ_CMD and other GC registers; the navi10 one has the same GC 10.1
offsets:

```sh
sudo -n umr -f navi10 -O halt_waves -wa > waves.txt
```

Per wave: `SQ_WAVE_PC`, `INST_DW0` (`0xbf810000` is `s_endpgm`), `TRAPSTS` (bit 8 memory
violation, bit 28 XNACK error). Waves parked at `s_endpgm` with a memory violation point at a GPU
page fault:

```sh
journalctl -k -b | grep -A8 "page fault"
```

The UTCL2 client names the unit (GE2: the geometry engine, which reads index buffers and writes tess
factors; TCP: shader memory). Faults at addresses near 0 are a base register nobody programmed.

## 5. Isolating

- Switch one feature off and rerun the same input: `KYTY_NATIVE_GPU_ACB_RINGS=0` (ACBs on the gfx
  ring), `KYTY_NATIVE_GPU_CP_STATE`, `KYTY_NATIVE_GPU_GDS_DMA_SYNC=1`, ...
- Test one hypothesis in isolation with a probe danger case (`--danger gdsdma`: the title's exact
  DMA_DATA packets to and from GDS on a compute ring and on gfx).
- Reproduce with the same input. `gt-steam.sh TAG SECS [ENV=...]` launches the Steam shortcut,
  presses Return (Kyty's PSN dialog) and `GT_J` times J, screenshots every 15 s. Keyboard keys go
  only into the Kyty window (`press.sh`). A virtual pad holding player 1 blocks the real one, so
  scripted pads and a person playing do not mix.

## Cases found this way

**Compute rings sharing one barrier fence.** A guest wait becomes "RELEASE_MEM fence := N; WAIT
fence == N". All compute rings used one fence address; with ACB queues on two rings running at once,
one ring overwrote the other's N between its release and its wait, and that wait never ended (CP busy,
SPI idle, WAIT_REG_MEM in the header dump; breadcrumbs showed waits for 5 and 8 on the same
address). Each compute ring now has its own fence in its tracker slot.

**Tessellation factor ring not programmed.** The title gives the ring to
`sceAgcDriverSetTFRing` / `sceAgcDriverSetHsOffchipParam`, and the PS5 driver programs it. Nobody
did on the native path: the first tessellated draws made the GE write tess factors at GPU address 0
(GE2 faults at 0x0-0x1e000), the waves stopped at `s_endpgm` with memory violations, and the gfx ring
hung. Every gfx IB's preamble now writes VGT_TF_RING_SIZE, VGT_HS_OFFCHIP_PARAM and
VGT_TF_MEMORY_BASE(_HI) (the `_UMD` registers) from those calls.

## Launch pitfalls

- Steam's `reaper` is a subreaper: a distrobox container first started inside a game launch stays its
  child, and Steam then reports the game as running after it exits. `kyty-steam.sh` starts the
  container through `systemd-run --user` first.
- Kyty's "Sign in to PlayStation Network" dialog waits on a black screen; Return or Cross closes it.
