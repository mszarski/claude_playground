# Vendored third-party code

| Directory | Project | Version | Licence | Why vendored |
| --- | --- | --- | --- | --- |
| `pocketfft/` | [mreineck/pocketfft](https://github.com/mreineck/pocketfft), branch `cpp` | commit `4d865cba24359b05b37da4e75aee24498c3b78e7` (2026-10-18) | BSD-3-Clause (`pocketfft/LICENSE.md`) | Header-only CPU FFT for the host backend. FFTW is GPL and is not used. The CUDA backend uses cuFFT. |

Files are unmodified copies. See `docs/OPEN_SOURCE.md` for the survey.

## Patches to third-party trees built outside this repository

`patches/` holds diffs of our own changes to two projects that are fetched and built
separately. Nothing of either project is vendored beyond the context lines a diff needs.

| Patch | Project and version | Licence of the project | Applied by | What it does |
| --- | --- | --- | --- | --- |
| `patches/holoscan-ultrasound-imaging-24d405e-long-window.patch` | NVIDIA holoscan-ultrasound-imaging, commit `24d405e` | Apache-2.0 for the NVIDIA source (per its README; `docs/OPEN_SOURCE.md` 3.2) | `tools/build_nvus_reference.sh` | raises the receive-window limit of the reference's sequence compiler from 0.88 m to 3.5 m; host C++ only (`docs/DECISIONS.md` D-45) |
| `patches/hsb-2.7.0-emulator-udp-socket.patch` | Holoscan Sensor Bridge 2.7.0, board emulator | Apache-2.0 (per the survey at the repository's HEAD; `docs/OPEN_SOURCE.md` 3.12) | `tools/build_hsb.sh`, only when `HSB_UDP_PATCH=1` | lets the emulator send through an ordinary UDP socket when `HSB_EMULATOR_UDP_SOCKET=1` is set at run time, for containers that refuse raw sockets; without the variable nothing changes (`docs/DECISIONS.md` D-79) |
