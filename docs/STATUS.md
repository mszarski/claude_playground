# Status

Updated at the end of each milestone. Newest facts first within each section.

## Where things stand

| Milestone | State | Evidence |
| --- | --- | --- |
| 1 Notes and survey | done | `PAPER_NOTES.md`, `HOLOSCAN_NOTES.md`, `OPEN_SOURCE.md`, `DECISIONS.md`; `tools/test.sh` builds and runs GoogleTest and pytest |
| 2 Record format and file replay | done | `RECORD_FORMAT.md`; `tests/cpp/test_record_io.cpp`, `tests/python/test_record.py`, `tests/python/test_sources.py` |
| 3 Simulator, point targets, all sonar configs | done; GPU forward model verified on a T4 | `tests/python/test_sim.py`: delay and amplitude against analytic values on every sonar, an independent frequency-domain oracle, Doppler, receiver motion, surface image, board clocks, ADC. |
| 4 Backprojection kernel, FLS mode | done; GPU kernel verified on a T4; PICMUS partly | `tests/python/test_backproject.py` (precision, independent reference), `tests/python/test_fls.py` (resolution, beam width, grating lobes), `tests/python/test_picmus.py`. Numbers in `RESULTS.md`. PICMUS: resolution measured; published reference numbers and contrast not done. |
| 5 Run-time optimisations | done; device slab, CUDA culling and hardware texture lookup verified on a T4 | `tests/python/test_culling.py` (image bit-identical with culling on, including a beam narrower than a block and a bistatic pair; CPU speedup in `RESULTS.md`), `tests/cpp/test_allocator.cpp`, `tests/python/test_position_lookup.py` (trajectory table restores focus under within-ping acceleration; 8-bit texture arithmetic emulated on the host stays inside the phase budget). |
| 6 Conditioning, pulse compression, post-processing | done on CPU; no GPU versions yet | `tests/python/test_conditioning.py`: compression gain against 10·log10(BT), replica windows, Doppler bank, calibration, deterministic TVG with hardware-ramp make-up, whitening (Eq. 9), median TVG (Eq. 10), DRC (Eq. 11–12), scan conversion. Numbers in `RESULTS.md`. The `dsp` stages run on the CPU only (pocketfft); a cuFFT backend is not written. |
| 7 SAS with known navigation | done on CPU | `tests/python/test_sas.py`: stripmap grid per the paper's Fig. 4, targets focus at their positions, along-track width within 4% of D/2 at three ranges, rail width within 2% of the aperture-limited theory. Numbers in `RESULTS.md`. |
| 8 Motion estimation | done on CPU, with gaps listed below | `tests/python/test_motion.py`: Ceres solve of Eq. 8 (Eq. 3–7 each tested), coarse and fine delays, two-pass unwrapping, end to end on a simulated collection. Numbers in `RESULTS.md`. |
| 9 3D near-field imaging | done on CPU; GeoTIFF output missing | `tests/python/test_volume.py`: Snell travel time against a least-time search, buried cylinder at the correct depth with refraction and 3.5 cm shallow without, slices and maximum intensity projections, bistatic culling unchanged image. Numbers in `RESULTS.md`. |
| 10 Flank array | done on CPU (Python/NumPy stages), with gaps listed below | `tests/python/test_flank.py`, `test_detect.py`, `test_calibration_sync.py`: far-field against backprojection, beam widths and ambiguity reports, sonar equation against simulation, MVDR, CFAR, clustering, tracking, calibration estimation, the `sync` stage. Numbers in `RESULTS.md`. |
| 11 Holoscan integration | not started | Holoscan has not been installed anywhere yet. |
| 12 Realistic propagation and reverberation | **in progress** | Done: `PHYSICS_NOTES.md` (TR 0702), the Bellhop provider as an external process, Rayleigh bottom reflection, validation against Bellhop on Pekeris and Munk cases (`tests/python/test_propagation.py`, skipped without Bellhop). Not done: both reverberation methods, surface and bottom scattering models, ambient and self-noise models, K-distributed clutter, the KRAKEN/RAM cross-check, the seamount case (range-dependent bathymetry is not supported), re-running the long-range scenarios at the `eigenray` tier. |
| 13 Performance | **partly** | Done: benchmark harness, timings on T4, L4, A10G and A100, power-law fit, real-time factors, multi-GPU tile splitting (`RESULTS.md`, `docs/benchmarks/`). Not done: any DGX Spark or Jetson measurement, measured multi-GPU scaling, verified GFLOPS values. Jetson build notes in `docs/JETSON.md` are untested. |

## What is next

1. Milestone 12: reverberation (frozen point scatterers for FLS/SAS, a scattering-function
   method for the flank array), scattering-strength and noise models, then the long-range
   scenarios at the `eigenray` tier.
2. Kernel speed on GPUs with slow float64: field-of-view test before the float64 solve.
3. Milestone 11 needs a Holoscan install (a GPU job image with the SDK would do).
2. GeoTIFF output (single and multi-layer), debug products per stage and the processing
   log: the framework items of the paper's Sec. II are still open.
3. Milestone 8 leftovers (see "Motion estimation limits").
3. A cuFFT backend for the `dsp` stages, so conditioning also runs on the device.
2. PICMUS leftovers: contrast/CNR on the cyst phantom; comparison with the published
   numbers once the paper can be read.

## Where the code has run

- **CPU:** everything, in an x86_64 cloud container (the development container has no GPU).
- **GPU:** the CUDA backend has been built and run on rented x86_64 machines through
  Hugging Face Jobs (`tools/gpu_job.sh`, image `nvidia/cuda:12.6.3-devel-ubuntu24.04`).
  On an NVIDIA T4, all ten GPU tests pass: echo synthesis against the CPU, backprojection
  against the independent reference for all three interpolators, and the hardware texture
  position lookup against the software table. Benchmarks on several GPU types are in
  `RESULTS.md`.
- **Not yet:** the brief's actual targets. Nothing has run on a DGX Spark, a Jetson, or
  any aarch64 machine, and Holoscan has not been installed anywhere.
- Motion estimation was not part of the GPU run (that job image has no Ceres); it has no
  GPU code.

## Open questions for the project owner

1. **Plan PDF.** `docs/plan/GPU-FLS-Phased-Plan.pdf` is not in the repo yet. Hardware
   numbers currently come from the brief alone.
2. **Ceres and SuiteSparse (copyleft): handled, confirm you agree.** Ubuntu's
   `libceres-dev` links SuiteSparse CHOLMOD/SPQR, which are GPL. Motion estimation is built
   against Ceres 2.2.0 compiled from source with `-DSUITESPARSE=OFF` (BSD only), and CMake
   refuses a SuiteSparse-linked Ceres unless `-DSONAR_ALLOW_SUITESPARSE=ON` is given. The
   cost is no sparse linear solver (dense QR is used), which matters only for very long
   runs. Say if you would rather allow the distro package.
3. **Holoscan versions.** The ultrasound reference at `24d405e` pins Sensor Bridge 2.7.0
   and Holoscan 4.4.0 with six local SDK patches. Sensor Bridge HEAD is 3.0.0 on Holoscan
   5.0, which is not API compatible. Proposed: build our ultrasound source against the
   reference's pinned pair, and keep the low-rate flank source on stock operators so it
   can follow either line. Confirm.
4. **Jetson receive path.** Per the Sensor Bridge source, AGX Orin is Linux-sockets only
   and AGX Thor is built without RoCE (CoE or sockets). 38.4 Gbit/s ultrasound capture
   therefore looks like DGX Spark / IGX only; the 49 Mbit/s flank source is fine anywhere.
5. **`popoto_4x8` reference channel.** Four 8-input boards give 32 inputs. A shared
   reference "wired into one input on every board" on a dedicated input would leave 28
   hydrophones, not 32. What is implemented: 32 hydrophones, with the reference summed
   into the first hydrophone input of each board as two short chirp bursts per ping at
   38–48 kHz, above the 20–35 kHz acoustic band. That needs an analogue summing node per
   board and enough ADC bandwidth (the assumed 102.4 kSPS just allows it). Confirm this
   matches what the hardware can do, or say which variant to build instead.
6. **Copyleft test oracles.** Bellhop / Acoustics Toolbox and bellhopcuda (GPL-3), k-Wave
   (LGPL-3) and PyMUST (LGPL) are proposed as separate-process oracles only, as the brief
   allows. Nothing copyleft is installed in the build or linked. Say if any of these should
   not be used even as an oracle.
7. **Reading.** TR 0702 is now fetched (`docs/refs/`, not committed) and summarised in
   `PHYSICS_NOTES.md`. TR 9407 is distribution-limited on DTIC and was not fetched; the
   open literature it draws on is used instead, as the brief allows. If you can supply
   TR 9407, put it in `docs/refs/`. PICMUS data itself is
   reachable through a Hugging Face mirror (`zeahub/picmus`, a converted layout); the
   published PICMUS reference numbers could not be fetched, so none are quoted yet.

## Discrepancies between the brief and the references

From reading the reference code (`HOLOSCAN_NOTES.md` section 1 has file and line for each):

- The reference's live imaging app is a native C++ loop around the non-operator
  `RoceReceiver`, not a Holoscan operator graph.
- The 1024-frame receive ring is the reference's own setting (`rx_pages`), settable from
  Python or C++ with no FPGA change. Stock `RoceReceiverOp` defaults to 2 pages.
- "Odd/even" mux: per the code, "odd" is elements 0–63 and "even" is 64–127. It is two
  contiguous halves, not an interleave. The mux is the TX73L64's own T/R switch select.
- Gain ramp: 12 dB and about 122 µs are the encoder's maxima, not the shipped settings
  (shipped: 12 dB, start 2 µs, ramp 10 µs). No AFE5932 attenuator register is written
  anywhere in the reference; the "unused 0–24 dB attenuator" was not found.
- A 150k-sample receive window exceeds two host-side caps (65,536 samples in the C++
  sequence compiler, 32,768 in the capture CLI). Neither is an FPGA limit;
  `Evm.init(samples=150000)` has no cap. Sustained 38.4 Gbit/s for 3 ms through the
  FPGA FIFO is untested by the reference, which streams about 60 µs per ping.
- No FPGA timestamp or ping counter reaches the application in the reference; the
  Sensor Bridge metadata has PTP times but the reference drops them.
- The reference transmits 5, 6.25 or 8.33 MHz only. Nothing in it supports sonar carrier
  frequencies; stages 3–6 need their own transmit chain.
- Docs say probe "L12-5 D"; code says "L7-3".

## Flank array limits so far

- The flank stages (`flank.py`, `detect.py`, `calibrate.py`, `sync.py`,
  `sonar_equation.py`) are Python/NumPy, not C++/CUDA, and not Holoscan operators.
- Everything was tested at the `free_field` tier in white noise. The long-range scenarios
  must be re-run at `eigenray` in milestone 12 before the detection ranges mean anything.
- The built-in scenarios "Popoto trial at 100 m, 1 km and 2 km" and "flank-array target at
  1, 5 and 10 km in noise" exist only as test code, not as scenario YAML files.
- No test runs a full 10 km ping with the 1 s pulse through detection and tracking; the
  10 km case was checked for SNR only.
- Tracking uses the built-in nearest-neighbour Kalman tracker. Stone Soup was not used.
- Clustering does not merge a target's sidelobe detections.
- The sync stage's reference is a design proposal (open question 5), not a description of
  existing hardware. Drift from the 1PPS and direct-path methods needs consecutive pings
  and is not implemented.
- Synthetic aperture with the flank array along track (the stretch goal) is not attempted.

## Not started at all

- Milestone 11: Holoscan operators and sources. Holoscan cannot be installed in the
  build container (no CUDA; wheels are for Python 3.12).
- Framework: GeoTIFF output, per-stage debug products, the processing log, and a single
  YAML-driven processing chain (today scenarios are YAML; the processing steps are
  called from Python).
- Scenes: faceted seafloor with bathymetry, spheres, plates, rocks, wreck, pool walls
  as surfaces. Built-in scenarios still missing: Phase 1 pool test, flat seafloor with
  objects, wreck with large motion, Popoto trials, flank-array targets in noise.

## Motion estimation limits so far

- **Surge by along-track estimation (ref [9]) is not implemented.** Surge comes from the
  DVL sample in each record.
- Delay estimation is NumPy, not C++; only the solve is C++.
- Tested with zero roll, pitch and yaw. Attitude and lever arms are in the residual
  (Eq. 4's Ω), but no test perturbs them yet.
- λ1 and λ2 are at their defaults; the tuning sweep has not been run.
- Known-bathymetry option: the hook exists and is tested only against a flat function.
- The scenario's ping spacing (0.24 m) leaves a quarter of the phase centres overlapping.
  At the full 0.32 m advance there is no overlap and DPC cannot work, as in real systems.
- Dense linear solver (no SuiteSparse): fine for hundreds of pings.

## Conditioning limits so far

- The band-pass is the band selection inside `to_baseband`; there is no separate filter
  stage for data that stays at passband.
- The Doppler bank picks one hypothesis per record (the brief allows "a small bank");
  per-range or per-beam selection is not done.
- The hardware gain-ramp limits of the reference rig (12 dB, at most about 122 µs on the
  TX73L64) are documented in `HOLOSCAN_NOTES.md` but not yet a named model in config;
  `apply_tvg` already makes up whatever ramp the record declares.

## Kernel limits so far

- The trajectory table holds the residual of the platform *position* only. Attitude
  within a ping is still the first-order model (exact for constant rates).
- An LFM pulse received by a platform with a closing speed shows range-Doppler coupling
  (about 1 mm and 7° for 0.17 m/s at 200 kHz with a 2 ms, 40 kHz sweep). Nothing
  compensates this yet; the Doppler bank in milestone 6 is where it belongs.

- The sound-speed-profile option (eigenray travel-time table) is not in the kernel yet;
  delays are straight-ray at one sound speed per ping. Planned with milestone 12.
- `make_inputs` copies every ping's samples into one buffer per call. Fine for tests;
  the slab allocator in milestone 5 replaces it.
- Real passband input is rejected; convert with `to_baseband` first.
- The FLS scan sequence for the reference rig, the hardware gain-ramp model with its
  digital make-up, and multi-angle compounding on our own probes are not done. Plane-wave
  compounding itself works (PICMUS).

## Simulator limits so far

- Scenes are point scatterers only. Seafloor, objects, walls as surfaces, sediment and
  moving-target tracks come with the later milestones that need them.
- Tiers `free_field` and `image_source` exist. `eigenray` (Bellhop) is milestone 12 and
  raises `NotImplementedError` until then.
- Noise is white Gaussian at a given RMS. Ambient spectra, self-noise and array coherence
  are milestone 12.
- Arbitrary (sampled) transmit replicas cannot be synthesised yet; CW, LFM and Gaussian can.
- `fls_300k` needed an analogue gain ramp to register a 300 m echo in 16 bits; the ramp in
  its config is a placeholder, flagged unconfirmed.

## Known deviations from the paper

None in implemented code yet. Planned resolutions of the paper's gaps are in
`DECISIONS.md` (D-01 to D-25), each tied to a numbered gap in `PAPER_NOTES.md` Part 2.

## Issue tracker

`bd` holds one issue per milestone (prefix `sonar-`). Its database is local to the
checkout; this file is the durable record.
