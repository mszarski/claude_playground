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
| 6 Conditioning, pulse compression, post-processing | done; baseband conversion and pulse compression also on the device, verified on T4 and A10G | `tests/python/test_conditioning.py`: compression gain against 10·log10(BT), replica windows, Doppler bank, calibration, deterministic TVG with hardware-ramp make-up, whitening (Eq. 9), median TVG (Eq. 10), DRC (Eq. 11–12), scan conversion. Numbers in `RESULTS.md`. Host FFT is pocketfft; the CUDA build uses cuFFT for baseband conversion and pulse compression (D-42). Whitening, median TVG, calibration and DRC are host only. |
| 7 SAS with known navigation | done on CPU | `tests/python/test_sas.py`: stripmap grid per the paper's Fig. 4, targets focus at their positions, along-track width within 4% of D/2 at three ranges, rail width within 2% of the aperture-limited theory. Numbers in `RESULTS.md`. |
| 8 Motion estimation | done on CPU, with gaps listed below | `tests/python/test_motion.py`: Ceres solve of Eq. 8 (Eq. 3–7 each tested), coarse and fine delays, two-pass unwrapping, end to end on a simulated collection. Numbers in `RESULTS.md`. |
| 9 3D near-field imaging | done on CPU | `tests/python/test_volume.py`: Snell travel time against a least-time search, buried cylinder at the correct depth with refraction and 3.5 cm shallow without, slices and maximum intensity projections, bistatic culling unchanged image. Numbers in `RESULTS.md`. |
| 10 Flank array | done on CPU (Python/NumPy stages), with gaps listed below | `tests/python/test_flank.py`, `test_detect.py`, `test_calibration_sync.py`: far-field against backprojection, beam widths and ambiguity reports, sonar equation against simulation, MVDR, CFAR, clustering, tracking, calibration estimation, the `sync` stage. Numbers in `RESULTS.md`. |
| 11 Holoscan integration | **partly** | Done: stages as Holoscan operators (`python/gpusonar/holoscan_ops.py`) and pipelines fed by the simulator and by file replay, including a real-time-paced replay, run on a T4 with Holoscan SDK 4.6.0 (`tests/python/test_holoscan.py`). The raw-frame adapter for the ultrasound reference, checked against the reference's decoder (`test_nvus_adapter.py`). The low-rate flank source on the host side: block format, ping assembler, UDP source and board emulator (`test_flank_stream.py`), and a Holoscan pipeline fed by the emulated board over UDP that detects its target, run on a T4 (`test_holoscan.py`). Not done: the ultrasound source built against the reference at the pinned commit, the flank source behind the Sensor Bridge receiver and its emulator, C++ operators. A first-hardware-run checklist draft is in `HOLOSCAN_NOTES.md`. |
| 12 Realistic propagation and reverberation | **in progress** | Done: `PHYSICS_NOTES.md` (TR 0702); the Bellhop provider as an external process; Rayleigh bottom reflection; validation against Bellhop (`test_propagation.py`); point-scatterer bottom reverberation with Lambert's law, level within 2 dB of the analytic prediction, Rayleigh statistics (`test_reverberation.py`); ambient spectrum, array-coherent Gaussian noise, tones, K-distributed clutter (`test_noise.py`). The scattering-function reverberation method (`sim/reverb.py`, D-37): level within 2 dB of the analytic prediction and of the point-scatterer method, array coherence checked. Flank and Popoto scenarios re-run at the `eigenray` tier with array-coherent ambient noise, all with bottom reverberation, and compared with the spherical estimate (`test_long_range.py`). Sea-surface backscatter (Chapman–Harris) and rough-surface reflection loss (`test_surface.py`, D-38). Rays against normal modes (KRAKEN) at 8 kHz: within 0.21 dB from 1 to 10 km (`test_kraken_crosscheck.py`); that check found the arrival cap of the first long-range scenarios too low, now fixed (D-39). Range-dependent bathymetry through Bellhop (a 2-D ridge; `test_propagation.py`). Not done: reverberation from sloping bathymetry, bubble absorption and the APL-UW high-frequency surface model, a Jackson-type bottom model, volume scattering, time spreading, the RAM cross-check, a 3-D seamount. |
| 13 Performance | **partly** | Done: benchmark harness; timings on T4, L4, A10G, A100 and two A10Gs; power-law fit; real-time factors; sessions that keep samples on the device (D-40), which removed the per-call overhead (A10G SAS 0.83 s → 0.44 s) and brought two-GPU scaling to 1.7×, images bit-identical; a 2 cm SAS image that is compute-bound; baseband conversion and pulse compression on the device with cuFFT (35–48× faster than four CPU threads, D-42) (`RESULTS.md`, `docs/benchmarks/`). Not done: any DGX Spark or Jetson measurement; verified GFLOPS figures; handing conditioned samples to the backprojection session without a round trip through host memory; motion estimation is CPU. Jetson build notes in `docs/JETSON.md` are untested. |

## What is next

1. Milestone 11: the ultrasound source built against the reference at its pinned commit;
   the flank assembler behind the Sensor Bridge receiver and emulator; C++ operators.
2. Milestone 13: keep conditioned samples on the device for the backprojection session;
   Spark and Jetson measurements when the hardware is available.
3. Milestone 12: a Jackson-type bottom, reverberation from sloping bathymetry, time
   spreading, a high-frequency surface model.
4. Port the remaining NumPy stages (FFT beamformer, clustering, sync, calibration) to the kernel framework.
5. Milestone 8 leftovers (see "Motion estimation limits").
6. PICMUS leftovers: contrast/CNR on the cyst phantom; comparison with the published
   numbers once the paper can be read.

## Where the code has run

- **CPU:** everything, in an x86_64 cloud container (the development container has no GPU).
- **GPU:** the CUDA backend is built and run on rented x86_64 machines through Hugging
  Face Jobs (`tools/gpu_job.sh`, image `nvidia/cuda:12.6.3-devel-ubuntu24.04`). The job
  can also build Ceres and the Bellhop/KRAKEN oracles, so the whole suite runs there. On
  an NVIDIA T4 the full suite passes (last run: 264 passed, 0 failed, 8 skipped: five need the PICMUS data, one the GeoTIFF library, one the reference checkout, one is a check for builds without CUDA). That covers
  echo synthesis, backprojection (all interpolators, culling, texture lookup, sessions),
  cuFFT conditioning, CFAR, the Holoscan pipelines, motion estimation, the long-range
  scenarios with GPU synthesis and the wreck scenario end to end. Benchmarks on T4, L4,
  A10G, A100 and two A10Gs are in `RESULTS.md`.
- **Holoscan:** SDK 4.6.0 (`pip install holoscan-cu12`) on the same rented T4; the
  operator pipelines run there, including one fed by the emulated flank board over UDP.
  It cannot be imported in the development container (its CUDA runtime is too old), so
  those tests skip locally.
- **Not yet:** the brief's actual targets. Nothing has run on a DGX Spark, a Jetson, or
  any aarch64 machine, and nothing has touched Sensor Bridge hardware or its emulator.
- Motion estimation has no GPU code; it runs on the CPU of whichever machine.

## Open questions for the project owner

1. **Plan PDF: received** (`docs/plan/GPU-FLS-Phased-Plan.pdf`). Applied: Popoto front-end
   figures, flank hydrophone sensitivity, the 28-hydrophone batching scheme, and tests that
   the sonar-equation tool reproduces the plan's link budgets. Differences that remain are
   listed under "Differences from the plan".
2. **Ceres and SuiteSparse: settled.** The owner confirmed BSD only. Motion estimation
   is built against Ceres 2.2.0 compiled with `-DSUITESPARSE=OFF`, and CMake refuses a
   SuiteSparse-linked Ceres unless `-DSONAR_ALLOW_SUITESPARSE=ON` is given. The cost is no
   sparse linear solver (dense QR), which matters only for very long runs.
3. **Holoscan versions: settled.** The owner approved building the ultrasound source
   against the reference's pinned pair (Sensor Bridge 2.7.0, Holoscan 4.4.0, six local SDK
   patches) and keeping the low-rate flank source on stock operators. Not built yet.
4. **Jetson receive path.** Per the Sensor Bridge source, AGX Orin is Linux-sockets only
   and AGX Thor is built without RoCE (CoE or sockets). 38.4 Gbit/s ultrasound capture
   therefore looks like DGX Spark / IGX only; the 49 Mbit/s flank source is fine anywhere.
5. **Board batching: settled by the plan.** `popoto_4x7` follows it: one dedicated
   reference input per board, 7 hydrophones each, 28 in all; the sync stage aligns the
   boards and the reference channels are dropped before beamforming. `popoto_4x8` (32
   hydrophones, reference summed into one input per board) stays as an alternative the
   plan does not describe.
6. **Copyleft test oracles: settled.** The owner approved Bellhop and KRAKEN (GPL-3)
   built and run as separate programs by the GPU job; nothing copyleft is linked. k-Wave
   (LGPL-3) and PyMUST (LGPL) would be used the same way and are not installed.
7. **Reading.** TR 0702 is now fetched (`docs/refs/`, not committed) and summarised in
   `PHYSICS_NOTES.md`. TR 9407 is distribution-limited on DTIC and was not fetched; the
   open literature it draws on is used instead, as the brief allows. If you can supply
   TR 9407, put it in `docs/refs/`. PICMUS data itself is
   reachable through a Hugging Face mirror (`zeahub/picmus`, a converted layout); the
   published PICMUS reference numbers could not be fetched, so none are quoted yet.

## Differences from the plan

- **Phase 3 (0.5–1 MHz, 64 elements, 20–50 m)** is not in the brief's list of named
  sonars. `fls_750k` covers it with placeholder values.
- **`sas_auv`** uses a 0.64 m receive array (16 × 4 cm) advancing 0.32 m per ping. The
  plan's example is a 0.32 m array at about 0.8 m/s. Same rule (advance half the array
  per ping), different size.
- **Popoto noise level.** The plan's Phase 0 estimate (37 dB at 1 km, 13 dB at 2 km) does
  not state its noise level; 38 dB re 1 µPa²/Hz reproduces it and is what the test uses.
- **Phase 1 range resolution.** The plan says about 0.5 mm; the `l12_5d` config assumes a
  50% bandwidth pulse and measures 0.28 mm. The probe's real bandwidth at 5 MHz is unknown.
- **Receive ring.** The plan places the 1024-frame default in `scanner_hardware.py`; at the
  pinned commit it is `HololinkTransportConfig.rx_pages` in the C++ transport header
  (`HOLOSCAN_NOTES.md`).
- The plan's AUV compute table agrees with what the Sensor Bridge source shows (RoCE on
  ConnectX hosts only).

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
  `sonar_equation.py`) are Python/NumPy, except CFAR, which is also a C++ kernel for the
  host and CUDA backends (`libs/detect`, `detect.cfar(..., backend=...)`).
- Detection, tracking, MVDR, calibration and sync were tested at the `free_field` tier in
  white noise. Only the SNR of the long-range scenarios has been checked at the `eigenray`
  tier, with bottom reverberation.
- No test runs a full 10 km ping with the 1 s pulse through detection and tracking; the
  10 km case was checked for SNR only.
- Tracking uses the built-in nearest-neighbour Kalman tracker. Stone Soup was not used.
- Clustering does not merge a target's sidelobe detections.
- The sync stage's reference is a design proposal (open question 5), not a description of
  existing hardware. Drift from the 1PPS and direct-path methods needs consecutive pings
  and is not implemented.
- Synthetic aperture with the flank array along track (the stretch goal) is not attempted.

## Scenes and scenarios

- Scene objects (`sim/scene.py`, `test_scene_objects.py`): points, polar grids, wires,
  seafloor patches (flat, sloped, rippled), plates, spheres, boxes, rocks, pool walls.
  Surface points carry a normal and scatter by Lambert's law about it; spheres and boxes
  cast shadows (a sphere on the seafloor leaves the floor behind it more than 10 dB
  darker in the forward-looking image).
- Not modelled: specular highlights (a smooth sphere or plate glint), diffraction into
  shadows, shadows of boundary-reflected paths (they are treated as direct), rotated
  boxes, meshes. Seafloor relief does not shadow itself.
- Sonar configs beyond the brief's nine: `subbottom`, `popoto_4x7` (the plan's wiring),
  `fls_750k` (the plan's Phase 3 mid-range step; every value a placeholder).
- Built-in scenarios: a point grid per sonar, `stripmap_points`, `rail_points`,
  `rail_position_errors`, `pool_wires_l12_5d` (Phase 1 pool test), `seafloor_objects_fls`,
  `wreck_sas` (large unknown motion), `buried_cylinder`, `flank_target_{1,5,10}km`,
  `popoto_trial_{100,1000,2000}m`.
- `wreck_sas` has been run end to end (simulate, motion estimation, image; `RESULTS.md`
  and `docs/figures/wreck_sas.png`). Its test needs a CUDA device; on a CPU only a
  2-ping smoke test runs. The pool scenario has no walls or surface.

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
- The reference rig's receive-attenuator ramp is a named model (`gpusonar/hardware.py`,
  `receive.gain_ramp` in `l12_5d.yaml`): it reproduces the reference's register values
  and enforces its encoder limits (12 dB, start within 10.23 µs, at most 122.4 µs). It is
  over before the sonar window opens, so range gain there is digital (`apply_tvg`).

## Kernel limits so far

- The trajectory table holds the residual of the platform *position* only. Attitude
  within a ping is still the first-order model (exact for constant rates).
- An LFM pulse received by a platform with a closing speed shows range-Doppler coupling
  (about 1 mm and 7° for 0.17 m/s at 200 kHz with a 2 ms, 40 kHz sweep). Nothing
  compensates this yet; the Doppler bank in milestone 6 is where it belongs.

- The sound-speed-profile option (eigenray travel-time table) is not in the kernel;
  delays are straight-ray at one sound speed per ping (plus the flat-interface Snell
  option). The simulator can bend rays (Bellhop); the imaging side cannot yet.
- `make_inputs` copies every ping's samples into one host buffer. A session
  (`BackprojectSession`) does that once and keeps the device copy between calls; a
  one-shot `backproject` call still does it every time.
- Real passband input is rejected; convert with `to_baseband` first.
- The FLS scan sequence for the reference rig and multi-angle compounding on our own
  probes are not done. Plane-wave
  compounding itself works (PICMUS).

## Simulator limits so far

- Everything is synthesised from point scatterers (see "Scenes and scenarios" for what
  that leaves out).
- Tiers: `free_field`, `image_source`, `sediment` and `eigenray` (Bellhop as an external
  program; without it those scenarios and tests skip).
- Arbitrary (sampled) transmit replicas cannot be synthesised yet; CW, LFM and Gaussian can.
- The scattering-function reverberation assumes a stationary platform, an omnidirectional
  projector and a flat uniform boundary (D-37).
- `fls_300k` needed an analogue gain ramp to register a 300 m echo in 16 bits; the ramp in
  its config is a placeholder, flagged unconfirmed.

## Known deviations from the paper

Each is argued in `DECISIONS.md`; the paper's own gaps and how they were filled are
D-01 to D-25, tied to `PAPER_NOTES.md` Part 2. Deviations in implemented code:

- Whitening and median TVG apply the square root of the paper's gain, read as a power
  gain (D-20); the literal reading is kept as an option and tested not to flatten.
- The motion solve uses ping positions as parameters instead of per-ping velocities; the
  minimiser is the same (D-30).
- The kernel tests each (point, ping) pair against the beam in float32 before the float64
  delay solve (D-35); images are unchanged.
- Two device slabs instead of one allocation per run: the resident collection and one per
  call (D-40).
- Not in the paper at all: the flank array stages, the long-range propagation and
  reverberation models, the capture sources.

## Issue tracker

`bd` holds one issue per milestone (prefix `sonar-`). Its database is local to the
checkout; this file is the durable record.
