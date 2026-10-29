# KytyPS5-bc250

A fork of [KytyPS5](https://github.com/KytyPS5/KytyPS5) that runs PS5 games' GPU work natively on the
AMD BC-250. The BC-250 has the PS5's GPU silicon (gfx1013, "Oberon"), so the title's command buffers
run on the hardware after a light translation, and its shaders run with small patches. Vulkan stays for
presentation and as a fallback.

> [!IMPORTANT]
> Not affiliated with Sony Interactive Entertainment or PlayStation. No games or system software are
> distributed. Use only game files you obtained legally.

## Status

| Game | Native |
|---|---|
| Gran Turismo 7 | 60 fps in races |
| ASTRO BOT | levels playable |
| Stray | 60 fps |
| My First Gran Turismo | 60 fps |
| Dead Cells | 60 fps |

## Requirements

- BC-250 with SteamOS (tested), stock or patched amdgpu
- GTT 14 GiB (`ttm.pages_limit=3670016`). Without it games over 8 GiB of guest memory stall loading
- Recommended: `options amdgpu sched_policy=2 vm_update_mode=0 cwsr_enable=0`, or the patched amdgpu

Kernel modes and how to set them: [`tools/bc250-amdgpu`](tools/bc250-amdgpu/README.md).

## Build

On SteamOS inside a [Distrobox](https://distrobox.it/) container with Clang, CMake, Ninja,
glslang and the SDL3 development libraries:

```bash
git submodule update --init --recursive
cmake -S . -B _Build/linux-no-qt -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DKYTY_BUILD_LAUNCHER=OFF
cmake --build _Build/linux-no-qt --target kyty_emulator --parallel
```

## Run

```bash
KYTY_NATIVE_GPU_GEM=1 KYTY_NATIVE_GPU_PRIMARY=1 \
  ./_Build/linux-no-qt/kyty_emulator --game /path/to/PPSA01317 --fullscreen
```

From a Steam shortcut: [`tools/bc250-runs/kyty-steam.sh`](tools/bc250-runs/kyty-steam.sh). The root
helper [`tools/bc250-lodhw`](tools/bc250-lodhw/kyty-lodhw.c) handles the privileged registers (LOD
statistics, KFD TLB flushes). `kyty_emulator --help` lists the other options.

## Native backend

Code: [`src/graphics/native_gpu`](src/graphics/native_gpu).

- [Streamed DCBs](docs/native-gpu-streamed-dcbs.md): the GPU starts while the translation goes on
- [Debugging GPU hangs](docs/native-gpu-hang-debugging.md)
- [PS5 legacy GCN mode](docs/ps5-legacy-gcn-mode.md)
- [`tools/bc250-native-probe`](tools/bc250-native-probe/README.md): hardware probes
- [`tools/bc250-frame-diff`](tools/bc250-frame-diff/README.md): native and Vulkan frames compared

## License

[GNU General Public License version 2](LICENSE) (`GPL-2.0-only`). Based on
[Kyty](https://github.com/InoriRus/Kyty) (MIT, notice in [`LICENSES/Kyty-MIT.txt`](LICENSES/Kyty-MIT.txt)).
Third-party components keep their own licenses.

## Thanks

- [KytyPS5](https://github.com/KytyPS5/KytyPS5): the translation layer this fork builds on
- [InoriRus/Kyty](https://github.com/InoriRus/Kyty): the original Kyty
- [shadps4-emu/shadPS4](https://github.com/shadps4-emu/shadPS4): PS4 memory, GPU aliasing, cache
  coherency and AVPlayer reference
