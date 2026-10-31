# Web emulator core

`../data/cores/mupen64plus_next-wasm.data` is a rebuild of the EmulatorJS fork of
mupen64plus-libretro-nx (see `BASE_COMMIT.txt`) with `mupen64plus-libretro-nx-libdragon.patch`
applied. Stock browser N64 cores cannot run modern libdragon homebrew; the patch fixes, among other
things: 64-bit uncached CPU addresses, LLD/SCD, MI repeat mode, RSP/RDP completion and interrupt
handling, RSP DMA wraparound, 24-bit RDP command pointers, AI DMA address masking (audio), interrupt
Cause handling during nested exceptions, upstream RDRAM/RI emulation, accurate cxd4 vector
loads/stores, and a portable interpreter for the parallel-rsp RSP (its JIT cannot run in WASM).

Build (Emscripten 3.1.74, EmulatorJS RetroArch fork):

```bash
emmake make -j4 platform=emscripten INITIAL_HEAP=268435456 AUTO_MEMORY_GROWTH=1 LLE=1 HAVE_THR_AL=1 HAVE_PARALLEL_RSP_INTERP=1
```

then link with `Makefile.emulatorjs` from https://github.com/EmulatorJS/RetroArch (ASYNC=1, no threads)
and pack `mupen64plus_next_libretro.{js,wasm}` with `core.json`, `build.json`, `license.txt` into a 7z
named `mupen64plus_next-wasm.data`.

Licences: mupen64plus and the RSP/RDP plugins are GPL-2.0-or-later / MIT / LGPL as noted in their
sources; EmulatorJS (the `data/` folder) is GPL-3.0 (`../data/LICENSE-EmulatorJS.txt`). This patch is
the corresponding source for the shipped binary.
