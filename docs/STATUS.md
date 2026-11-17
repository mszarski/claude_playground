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
| 10 Flank array | done on CPU, in Python and in C++ (`libs/live`), with gaps listed below | `tests/python/test_live_cpp.py`, `test_doa.py`, `tests/cpp/test_live.cpp`, `tests/python/test_flank.py`, `test_detect.py`, `test_calibration_sync.py`: far-field against backprojection, beam widths and ambiguity reports, sonar equation against simulation, MVDR, CFAR, clustering, tracking, calibration estimation, the `sync` stage. Numbers in `RESULTS.md`. |
| 11 Holoscan integration | **partly** | Done: stages as Holoscan operators (`python/gpusonar/holoscan_ops.py`) and pipelines fed by the simulator and by file replay, including a real-time-paced replay, run on a T4 with Holoscan SDK 4.6.0 (`tests/python/test_holoscan.py`). The raw-frame adapter for the ultrasound reference, checked against the reference's decoder (`test_nvus_adapter.py`). The low-rate flank source on the host side: block format, ping assembler, UDP source and board emulator (`test_flank_stream.py`), and a Holoscan pipeline fed by the emulated board over UDP that detects its target, run on a T4 (`test_holoscan.py`). The ultrasound source driven through the reference's native runtime at the pinned commit, against its simulated hardware (`test_nvus_scanner.py`). The reference's compiler stopped at a 0.88 m window; the owner chose to patch it: `third_party/patches/` raises the limit to 3.5 m (host C++ only), `tools/build_nvus_reference.sh` builds the patched module on a CPU, and a 2.95 m window now runs through the reference's fast loop. The flank source behind Sensor Bridge's socket receiver and its board emulator: Sensor Bridge 2.7.0 builds against Holoscan 4.6.0, and with an opt-in patch that lets the emulator use an ordinary UDP socket (D-79) the loopback ran on T4 and L4 jobs with reports equal to the direct chain's at half real time; the loopback test's final assertions have not yet passed in a job, and real-time pace loses one burst per run (`RESULTS.md`, "Sensor Bridge loopback"). Holoscan operators in C++ (D-59). Not done: the reference's SDK build (Sensor Bridge 2.7.0, Holoscan 4.4.0) and any hardware run (21 MB frames are untried there), the RoCE receiver, the unpatched emulator on a machine that permits raw sockets. A first-hardware-run checklist draft is in `HOLOSCAN_NOTES.md`. |
| 12 Realistic propagation and reverberation | **in progress** | Done: `PHYSICS_NOTES.md` (TR 0702); the Bellhop provider as an external process; Rayleigh bottom reflection; validation against Bellhop (`test_propagation.py`); point-scatterer bottom reverberation with Lambert's law, level within 2 dB of the analytic prediction, Rayleigh statistics (`test_reverberation.py`); ambient spectrum, array-coherent Gaussian noise, tones, K-distributed clutter (`test_noise.py`). The scattering-function reverberation method (`sim/reverb.py`, D-37): level within 2 dB of the analytic prediction and of the point-scatterer method, array coherence checked. Flank and Popoto scenarios re-run at the `eigenray` tier with array-coherent ambient noise, all with bottom reverberation, and compared with the spherical estimate (`test_long_range.py`). Sea-surface backscatter (Chapman–Harris) and rough-surface reflection loss (`test_surface.py`, D-38). Rays against normal modes (KRAKEN) at 8 kHz: within 0.21 dB from 1 to 10 km (`test_kraken_crosscheck.py`); that check found the arrival cap of the first long-range scenarios too low, now fixed (D-39). Range-dependent bathymetry through Bellhop (a 2-D ridge; `test_propagation.py`). The SST-equivalence work (D-62 to D-78): time spreading with angle and frequency spreading, volume layers, a fluid-sediment bottom with Kirchhoff, small-slope and composite regimes and two named sediments, bistatic reverberation with scatterer Doppler, K statistics and relief, highlight targets and specular shapes, sampled transmit pulses, and a validation against TR 0702 section by section (`test_sst_validation.py`, `docs/figures/sst_validation.md`). It was tested file by file on the CPU; the full CPU suite and a GPU job on the merged tree are running and have not reported. A defect in the bistatic reverberation's Doppler, found by the validation, is fixed (D-80), and the two NumPy fallbacks are removed on the host (D-81; their CUDA side is unverified). Open: a bubble-loss law that is weak above 10 kHz. Not done: the APL-UW surface and bottom models as published (TR 9407 unread), a one-way source, the RAM cross-check, a 3-D seamount. |
| 13 Performance | **partly** | Done: benchmark harness; timings on T4, L4, A10G, A100 and two A10Gs; power-law fit; real-time factors; sessions that keep samples on the device (D-40), which removed the per-call overhead (A10G SAS 0.83 s → 0.44 s) and brought two-GPU scaling to 1.7×, images bit-identical; a 2 cm SAS image that is compute-bound; baseband conversion and pulse compression on the device with cuFFT (35–48× faster than four CPU threads, D-42) (`RESULTS.md`, `docs/benchmarks/`). Not done: any DGX Spark or Jetson measurement; verified GFLOPS figures; handing conditioned samples to the backprojection session without a round trip through host memory; motion estimation is CPU. Jetson build notes in `docs/JETSON.md` are untested. |

## Since the last round (see the sections it points to)

This round closed most of the gap to SST (APL-UW TR 0702) in the simulator, and ran the
Sensor Bridge loopback. Everything new in the simulator is opt-in: with the new keys
absent, outputs were checked identical to before. Numbers and commands are in
`RESULTS.md`; reasons in `DECISIONS.md` D-64 to D-82; derivations in `PHYSICS_NOTES.md`
Sec. 12 to 19; keys in `CONFIGURATION.md`.

**The full CPU suite and a GPU job on the merged tree are running; neither has reported
a result.** Until they do, what stands is this: each piece was tested on its own branch
with its own and neighbouring test files, and nothing below has run on a GPU except the
Sensor Bridge loopback. The CUDA code written this round (sampled-pulse synthesis on the
device) has not been compiled.

- **Targets and specular echoes** (`sim/targets.py`, D-64, D-65): targets in the SST
  manner (highlights on a trajectory, strength against aspect, a late-answering
  highlight), and smooth rigid spheres, cylinders, plates and discs that return specular
  glints by the Kirchhoff approximation, with a `resolved` mode for objects inside their
  own near field. Seabeds of point scatterers can follow the bottom scattering model
  instead of Lambert's law. Scenarios `rail_cylinder_glint`, `fls_glints_on_sediment`,
  `fls_highlight_target_aspect`. Bounce paths use the direct path's aspect.
- **Bottom and surface models** (D-66 to D-68): Kirchhoff, small-slope, composite and
  combined roughness models beside the perturbation default; two named sediments
  (`medium_sand`, `very_fine_silt`) read from an arXiv paper's table under the owner's
  ruling; a wind-speed bubble-loss law and two empirical backscatter laws. TR 9407 is
  still unread: the interpolation between regimes and the composite split are our own,
  and the bubble law is 2.3 to 7.5 times arlpy's in dB at 10 m/s (weak above 10 kHz, an
  upper estimate).
- **Reverberation** (D-69 to D-73): a bistatic geometry with both platforms moving and
  arrival elevation in the steering; K-distributed statistics; a boundary whose strength
  varies with place; moving scatterers (mean velocity and Doppler spread); a seabed with
  relief, with shadowing, and Bellhop run bearing by bearing. A taper against leakage
  along long records exists and stays off by default: none of the four shipped scenarios
  measured shows the leakage.
- **Sampled transmit pulses** (D-74, D-75): LFM, HFM, stepped FM, Barker, m-sequences,
  codes, inline samples or a file, carried in the record's existing replica fields (no
  schema change) and pulse-compressed by the C++ matched filter. Echoes of a sampled pulse
  are synthesised by the C++ kernel on the host (D-81; first in NumPy only); the CUDA
  side is written and waits for the GPU job.
- **Angle and frequency spreading of bounce paths** (D-76, D-77): each tap of a
  time-spread tail gets its own direction and, within a ping, its own Doppler lines;
  a facet-model decay time and convolved multi-bounce tails are offered beside the
  defaults. Scenario `flank_sst_spread`. A finding: the Doppler width goes as the sine of
  the grazing angle, not its inverse as TR 0702 p. 101 prints. By the owner's decision the
  convolved multi-bounce tail is now the default (D-82); the facet decay time stays
  opt-in. Rays with a within-ping Doppler are paired in C++ (D-81).
- **Validation against TR 0702** (D-78; `docs/figures/sst_validation.md`): a 61-row matrix
  through the report's sections, its two worked examples set up with our own parameters
  (not reproductions of its figures), and closed forms through the whole simulator:
  echo level against the sonar equation within 0.004 dB, `medium_sand` reverberation
  +0.27 dB from the integral. Scenarios `sst_pursuit`, `sst_asiaex`, `sst_echo_level`,
  `sst_sediment_reverb`.
- **A defect the validation found, now fixed** (D-80): bistatic reverberation gave one
  Doppler shift per component (azimuth sector and receive-ray elevation), which is wrong
  for pairs of different eigenrays that share a receive ray: up to 34 Hz with one bottom
  bounce in the Pursuit scene. Components are now also keyed by a bin of total Doppler
  shift, with the exact Doppler factor: 1.5 Hz in the same comparison, and the two tests
  that pinned the defect pass with their xfail markers removed. The Pursuit ping costs
  about five times more (5.8 s to 28.3 s). The committed validation matrix and figures
  were made before the fix and had not been regenerated when this was written.
- **Sensor Bridge loopback** (D-79): simulator, Sensor Bridge's emulator, its
  `LinuxReceiverOp`, our frames-to-pings operator and the chain, on T4 and L4 jobs. The
  emulator needed an opt-in patch to send through an ordinary UDP socket where raw
  sockets are refused. At half real time all 2006 frames arrive and the reports equal the
  direct chain's exactly. Two things are not settled: the loopback test's final
  assertions have not yet passed in a job (they were corrected after the last job, and
  await the GPU job now running on the merged tree), and at real-time pace every run loses one burst of datagrams, for a
  reason not established.
- **Stacked rows on the Popoto long-range array** (issue `sonar-k29`): an option for
  elevation and depth from two or three rows of hydrophones exists in code, in both chain
  engines. It is being re-evaluated: it was studied for rows on a 0.53 m hull, and the
  owner has since given the real mounting geometry, a vertical flat face 2 m high and 9 m
  long. Its study numbers, configs and conclusions are being redone for that face and are
  not quoted in this file, in `RESULTS.md` or in `DECISIONS.md`; the texts the option's
  own work added to `DEPLOYMENT.md` and `DESIGN.md` still describe the 0.53 m hull.

Earlier in the round:

- **SST equivalence, first part** (D-62, D-63; `RESULTS.md`): time spreading of
  boundary-bounce paths (`time_spread`), volume-scattering layers (`model: volume`), a
  rough fluid sediment bottom model (`model: fluid_sediment`) and a bubble-loss table;
  `flank_sst_shallow` uses all of them. The handbook SST takes its boundary models from
  (APL-UW TR 9407) is limited-distribution and was not read: the time-spreading split and
  tail are open substitutes and the bottom model is derived here.
- **The live chain in C++** (`libs/live`, `build/bin/sonar_live`; `ARCHITECTURE.md`,
  D-58): board stream to pings, navigation, synchronisation, clustering, tracking,
  classification and the report, as a library and a standalone program with no Python at
  run time. Compared stage by stage and report by report with the Python chain; the
  Python chain can run it (`engine: cpp`). One deployment file drives both.
- **Sparse Bayesian bearings** (`doa: sbl`, D-57): separates arrivals inside one beam of
  the long-range array, from one ping. Host and CUDA kernels. Passive bearing-time records
  from a recording (`--passive`) or from the untriggered board stream (`--listen`, D-60).
- **Holoscan in C++** (`libs/holoscan_ops`, `build/bin/sonar_live_holoscan`, D-59): six
  operators around the live library; built against Holoscan 4.6.0 and run on an L4, with
  reports identical to the standalone program's.
- **Ports from NumPy to C++** (D-61): far-field and MVDR beamforming, array calibration,
  path geometry, reverberation (its sum also on CUDA). NumPy versions kept as references.
- **On a GPU** (L4): the long-range chain takes 18 ms a ping (0.48 s with SBL); track
  classes on the 12 km scene were scored against the simulated truth (`RESULTS.md`).

- **Architecture** (`ARCHITECTURE.md`): one record between all stages; taps after any
  stage, simulator stages included, as container files that are also sources; tools to
  inspect and export them. A table of what runs on a GPU, tied to the code by a test.
- **Deployment** (`DEPLOYMENT.md`): `tools/sonar_live.py` runs the long-range chain one
  ping at a time from one YAML file: record, synchronise boards, condition, beamform,
  detect, track, classify, report. The same program runs on a scenario, a recording, an
  emulated board over UDP, and a directory of WAV recordings. Navigation comes in as UDP
  JSON. Nothing has run on a vehicle; the board forwarder and the navigation adapter are
  not written.
- **Simulator calibration** (`SIM_CALIBRATION.md`): fits boundary strengths and noise to an
  FLS log (absolute or relative levels) and reads seabed scatterer density and shadow depth
  from a SAS image. Checked on logs made by the simulator with hidden settings, not on
  real data: none is available here.
- **Classification** (`classify.py`): mover, static, line, multipath, uncertain, from the
  track alone.
- **Developed speckle** (`sas_minsas_speckle`): the ambiguity copies seen in the sparse SAS
  scenes are covered by a seabed with 2.6 scatterers per cell.
- **Sea surface**: waves travel at their physical speed, an irregular spectrum replaces
  two sinusoids, speckle is new each ping; the bands were measured to be fixed in the
  world, not the view.
- **Speed**: the synthesis kernel is 5.7 times faster; the earlier note that the host
  backend was single-threaded was wrong (several jobs were sharing the machine).

## What is next

1. Real data: a first FLS log or SAS image through `tools/calibrate_sim.py`, and a first
   Popoto recording through `tools/sonar_live.py`. Everything in those paths has only seen
   simulated input.
2. The two vehicle adapters: board samples to WAV-with-sidecar or UDP blocks; the
   vehicle's navigation message to the JSON the chain reads.
3. Milestone 11: the reference's SDK build with the long-window patch and a rig run. For
   the flank source behind Sensor Bridge: a job in which the corrected loopback test
   passes, the cause of the burst lost at real-time pace, and the same path with the C++
   engine, the CUDA backend and navigation (the C++ operators take blocks from a plain
   UDP socket).
4. Milestone 13: keep conditioned samples on the device for the backprojection session;
   reverberation's FFTs and random numbers on the device (its sum is); Spark and Jetson
   measurements.
5. Milestone 12: read the results of the full CPU suite and of the GPU job on the merged
   tree, both running now. The GPU job is the first compile and run of this round's CUDA
   code: the sampled-pulse tests of `test_sampled_engine.py` and `test_sim_synth`,
   `test_per_ray_doppler_on_the_cuda_kernel_matches_the_host`, the CUDA tests of
   `test_sim.py` and `test_sim_cpp.py`, `test_cuda_bistatic_reverberation_matches_the_host`,
   and the Sensor Bridge loopback test. Then: regenerate the validation matrix and figures
   (`tools/sst_validation.py`); the stacked-row option for the 2 m by 9 m face; a one-way
   source (target noise, tonals); per-element receive patterns in reverberation; ambient
   noise in C++ (the largest NumPy item left in a ping); and a better-founded bubble law
   and the published roughness interpolation if the papers can be supplied.
6. Classification beyond the track: extent and shape, so a buoy is not a rock and a wave
   face is not a boat; range correction for bottom-bounce arrivals.
7. A full survey scene with developed speckle (5.6 million scatterers: a GPU job).
8. Milestone 8 leftovers (see "Motion estimation limits"); PICMUS contrast.

## Where the code has run

- **CPU:** everything, in an x86_64 cloud container (the development container has no GPU).
- **GPU:** the CUDA backend is built and run on rented x86_64 machines through Hugging
  Face Jobs (`tools/gpu_job.sh`, image `nvidia/cuda:12.6.3-devel-ubuntu24.04`). The job
  can also build Ceres, the Bellhop/KRAKEN oracles and the ultrasound reference's native
  module, so the whole suite runs there. On
  an NVIDIA T4 the full suite passes (last run: 288 passed, 0 failed, 6 skipped: five need the PICMUS data, one is a check for builds without CUDA). That covers
  echo synthesis, backprojection (all interpolators, culling, texture lookup, sessions),
  cuFFT conditioning, CFAR, the Holoscan pipelines, motion estimation, the long-range
  scenarios with GPU synthesis, the realistic scenes of every mode, and the ultrasound
  source on the reference's native runtime (built there from the pinned commit). Benchmarks on T4, L4,
  A10G, A100 and two A10Gs are in `RESULTS.md`.
- **Holoscan:** SDK 4.6.0 (`pip install holoscan-cu12`) on the same rented T4; the
  operator pipelines run there, including one fed by the emulated flank board over UDP.
  It cannot be imported in the development container (its CUDA runtime is too old), so
  those tests skip locally.
- **Sensor Bridge:** 2.7.0 built against Holoscan 4.6.0 on rented L4 and T4 machines; its
  board emulator (with the opt-in UDP-socket patch, D-79) and socket receiver carried six
  simulated pings to the chain there. Not run: the stock emulator with a raw socket, the
  RoCE receiver, any Sensor Bridge hardware.
- **This round's simulator work (D-64 to D-82): CPU only so far, and file by file.** Each
  piece ran its own and neighbouring test files in the development container, on both
  engines where both apply, with Bellhop set for the reverberation and waveform work.
  **The full CPU suite and a GPU job on the merged tree are running; neither had reported
  when this was written** (two earlier attempts at the full suite on branches were killed
  on a loaded shared machine). The C++ changes (reverberation components, the sampled-pulse
  kernel, the pairing with range rates) were built and tested on the host. One `.cu` file
  changed (`synth_cuda.cu`, to upload the envelope table); it has never been compiled.
  The extension must be rebuilt after the merge (`cmake --build build -j3 --target
  _gpusonar`): the Python layer now calls bindings an older build does not have.
- **Not yet:** the brief's actual targets. Nothing has run on a DGX Spark, a Jetson, or
  any aarch64 machine, and nothing has touched Sensor Bridge hardware.
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
8. **Classic papers that could not be opened.** The surface bubble-loss law and the
   published interpolation between the Kirchhoff and composite regimes are in papers on
   pubs.aip.org, core.ac.uk and ieeexplore.ieee.org (Jackson, Winebrenner and Ishimaru
   1986; Mourad and Jackson 1989; Williams and Jackson 1998; Hall 1989; Hamilton's
   regressions). All three hosts refused automated access, and no arXiv paper found
   prints them. If you can open them in a browser and paste the formulas, they slot in as
   named models (`surface.BUBBLE_LOSS_MODELS`, `bottom.ROUGHNESS_MODELS`) without
   restructuring. Until then the bubble law is an upper estimate above 10 kHz and the
   composite split (`small_scale_kh = 0.5`) is our choice. Urick's table of simple
   target shapes and Dahl's spreading measurements were not opened either.
9. **The arlpy comparison.** `tools/boundary_models_study.py` prints a column from
   `arlpy.uwa.bubble_surface_loss` (BSD; its docstring says "APL model (1994)"). The
   formula was not read or copied and no value is stored. If that is too close to the
   restricted handbook, say so and the oracle block comes out (D-68).
10. **The two "confirmed" sediment rows** rest on a secondary table (arXiv:1809.08459,
    which attributes it to TR 9407) and on two assumptions of ours: water at 1000 kg/m³
    and 1500 m/s, and the reading of the volume strength's unit (D-67). Accept as they are?
11. **`multi_bounce: convolve` as the default: settled.** The owner decided for
    `convolve` (D-82): a cascade of independent bounces is the convolution of their
    responses, and `sum` was −2.3 dB right after the pulse and +3.7 dB later in the ASIAEX
    set-up. `sum` remains as an option. Earlier time-spreading numbers in `RESULTS.md`
    were produced under `sum` and are marked so; they were not re-measured.
12. **Passes of the analytic kernel's arrival-time solve.** Settled by the owner: the
    default is now (4, 4) (D-83). It was (3, 4), which left 0.007 rad at 8 kHz for 12 m/s
    closing at 400 m; (4, 4) leaves 5.5e-5. Records of moving scenes differ from earlier
    ones at the 1e-3 level or below.
13. **Time-spread coherence cells at an edge.** The realisation key is
    floor(position / coherence_length); a sonar exactly on a cell edge (x = 0) can put
    projector and array in different cells, making the out-leg and back-leg tails
    independent. Fixing it changes existing realisations (D-76). Fix it?
14. **`fls_170k_transit` reverberation is 1 to 1.5 dB off its expected power** over the
    second half of the record (bottom low, surface high), with and without the taper. Not
    leakage; cause not investigated (`RESULTS.md`, "`band_taper` on the shipped scenarios").
15. **`fls_120k_rough_surface` gives different sample hashes on the C++ and NumPy
    engines.** Seen while checking that the targets work changed nothing; the difference
    is older than that work. Whether it is only the single-precision difference D-61
    allows between the engines or something larger has not been checked.
16. **`band_taper` default.** Left off because none of the four shipped scenarios measured
    needs it. Its author
    recommends turning it on for any record whose reverberation falls by more than about
    50 dB with a short pulse (D-71). Keep it opt-in?
17. **The Pursuit and ASIAEX parameters** are ours, chosen to make the features TR 0702
    describes appear (D-78). If you have the example scripts from the SST distribution,
    the scenarios can carry the real values.

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

## The owner's designs (traceable; see `DESIGN.md`)

- Sonars: `fls_120k` (300 m), `fls_170k` (450 m, agreed down from 600 m), `fls_330k`
  (300 m), each 128 elements with a 120 degree sector and a configurable vertical field of
  view and tilt (`fov` block); `sas_minsas120` (330 kHz, 40 kHz, 1.09 m array, modelled on
  the Kraken MINSAS 120, now set for 2.1 cm along track); `popoto_array` (any number of
  hydrophones, boards of eight follow); `popoto_long_range` (32 hydrophones, 4–6 kHz,
  197 dB: the 12 km design).
- **12 km with the Popoto boards** (`tools/popoto_range_study.py`, `DESIGN.md` section 5):
  the 27.5 kHz band cannot do it; 4 to 6 kHz can, with 195 dB from 32 hydrophones (about
  270 W acoustic, omnidirectional) for 13 dB under spherical spreading. The shipped design
  has 2 dB more. In simulation it tracks a ship at 12 km in 100 m of water over sand, and at
  8 km in a winter duct; it does **not** see the same ship at 4 km under a summer
  thermocline over mud. No projector is chosen; electrical power is unknown.
- **MINSAS resolution**, measured on a simulated point target: 2.0 cm along track and
  1.9 cm in ground range unweighted; the survey figures use Hamming range weighting
  (2.6 cm, sidelobes −41 dB). Along-track ambiguities were −15 dB at the old 0.4 m per
  ping and are −36 dB at 0.544 m (half the array); the survey speed is now 2.18 m/s.
  The receive array went from 32 elements at 3.4 cm to 48 at 2.27 cm (same length): the
  coarser pitch left ghost copies of the cable and wreck 6.9 m along track (−42 dB on a
  point; now −47 to −54 dB at 10.3 m, visible beside the brightest objects on the sparse
  seabeds and covered on a seabed with developed speckle).
- Scenarios: `fls_{120,170,330}k_transit`; `fls_120k_cluttered_seabed`,
  `fls_120k_rough_surface`, `fls_170k_traffic` (boats with keel, propeller and wake;
  boulders, debris, a pipeline; waves); `sas_minsas_survey`, `sas_minsas_boulder_field`,
  `sas_minsas_rocky_ripples`; `popoto_far_mover`, `popoto_12km`, `popoto_env_*` (four
  oceans). Results, figures and animations in `SCENARIOS.md`.
- `tools/design_report.py` derives every design number from the config files, checks 30
  requirements, draws the diagrams and writes `DESIGN.md`; `test_design.py` fails if a
  requirement is unmet or the document is stale. `tools/scenario_diagrams.py` draws every
  scenario's geometry from its file. `CONFIGURATION.md` lists every key.
- Everything runs on a CPU (`README.md`, "Everything on a CPU"); CPU run times on four
  cores are in `SCENARIOS.md`.
- Found along the way, and what was done:
  - At 330 kHz one cell of sea surface at 250 m is below the noise floor, and with strong
    surface reverberation at 10 m/s wind nothing was detected, the boat included. The
    rough-sea example was moved to 120 kHz, where the waves show.
  - In the cluttered scene 65 tracks are classified 5 mover (one is the boat), 30 static,
    15 line, 15 uncertain; under the rough sea 57 tracks are 9 mover (one is the boat),
    39 uncertain. A buoy, a rock and a steady wave face still look alike.
  - In deep water the ship's echo arrives about 300 m late in range, consistent with a
    bottom path; ranges are not corrected for propagation.
  - At 330 kHz the anchored boat in `fls_330k_transit` is not tracked: it peaks at 12 dB
    at 277 m, under the 13 dB detection limit, against 22 dB for the boat at 238 m.
- What the design numbers say that needs a decision: the SAS at 215 dB keeps one cell of
  sand 13 dB above noise to about 117 m, not 200 m; a 197 dB source at 5 kHz is a large
  projector and amplifier for a small vehicle.
- Not real: source levels, sensitivities, element sizes and ADC choices are design values
  in the configs, flagged `unconfirmed`. Nothing is built.

## Scenes and scenarios

- Scene objects (`sim/scene.py`, `test_scene_objects.py`): points, polar grids, wires,
  seafloor patches (flat, sloped, rippled), plates, spheres, boxes, cylinders, rocks, pool
  walls, volume clutter, hulls, cables, boats (hull, keel, propeller, wake), sea-surface
  wave facets, debris.
  Surface points carry a normal and scatter by Lambert's law about it; spheres and boxes
  cast shadows (a sphere on the seafloor leaves the floor behind it more than 10 dB
  darker in the forward-looking image).
- Targets and specular shapes (`sim/targets.py`, `test_scene_targets.py`,
  `test_scene_sediment.py`): highlight targets with strength against aspect, and smooth
  rigid spheres, cylinders, plates and discs whose specular highlights (glints) are
  modelled by the Kirchhoff approximation (it warns below ka = 10). A `seafloor` item
  with `model: fluid_sediment` follows the bottom scattering model instead of Lambert's
  law, and a seafloor can take a `bathymetry`.
- Not modelled: specular echoes from the point-cloud objects (hulls, boxes, rocks and
  pool walls stay Lambert surfaces; only the shapes above glint), creeping waves and
  elastic resonances, reflections between shapes, diffraction into shadows, shadows of
  boundary-reflected paths (they are treated as direct), shadows cast by plates,
  occlusion of moving glint targets, rotated boxes, meshes. Seafloor relief does not
  shadow itself in the scene (it does in the scattering-function reverberation). On
  bounce paths a target or glint has the direct path's aspect.
- Sonar configs beyond the brief's nine: `subbottom`, `popoto_4x7` (the plan's wiring),
  `fls_750k` (the plan's Phase 3 mid-range step; every value a placeholder).
- Every mode has a realistic scenario run end to end, with figures and numbers in
  `docs/SCENARIOS.md` (full size on an A10G) and thinned versions as tests
  (`test_realistic_scenes.py`): `pool_fls_l12_5d`, `pool_fls_p4_2v`, `harbour_fls_750k`,
  `seafloor_objects_fls`, `wreck_sas`, `minefield_sas`, `rail_objects`, `buried_objects`,
  `flank_patrol`, `popoto_trial_*`. Other scenarios: a point grid per sonar,
  `stripmap_points`, `rail_points`, `rail_position_errors`, `pool_wires_l12_5d`,
  `buried_cylinder`, `flank_target_{1,5,10}km`. Added this round, none in the gallery:
  `rail_cylinder_glint`, `fls_glints_on_sediment`, `fls_highlight_target_aspect`,
  `flank_sst_shallow`, `flank_sst_spread`, and the four validation scenarios `sst_*`
  (`SCENARIOS.md`, "Scenarios for the SST-equivalence work").
- Gaps in those scenes: no surface or floor multipath in the pool scenes; direct paths
  only to objects in the forward-looking sea scenes; range-independent oceans in the
  long-range ones; a wake that does not grow; no breaking waves or bubble layer; large
  surfaces have fewer scatterers than resolution cells except in `sas_minsas_speckle`; no
  specular echoes in any gallery scene (the shapes that glint are used only in the three
  new target scenarios).
- In `flank_patrol` the −3 dB small vehicle is not detected, and multipath splits the
  wreck into several tracks (5 confirmed tracks for 2 objects).

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
- Sampled transmit pulses are synthesised by the C++ kernel on the host (D-81); the CUDA
  side is written and has not yet been verified on a GPU. With beams and absorption the
  sampled path is 3 times the analytic kernel (2.4 s against 0.80 s for 300 scatterers on
  128 channels); a large scene with a sampled pulse has not been timed. `table_factor: 0`
  and more than 32 `gain_knots` still run in NumPy, with a warning. CW cannot be given as
  a sampled pulse.
- `time_spread.doppler` runs on the C++ pairing (D-81). Path pairing has no CUDA backend.
- Still NumPy in a simulated ping, by cost: ambient noise (2.0 s of a 15–31 s ping; no
  C++ exists), the reverberation's random draws and K texture (NumPy on both engines by
  design, D-61), time spreading's taps, the scene modifiers, the ADC and board clocks.
  The full table is in `RESULTS.md`, "What in the simulator still runs in NumPy".
- There is no one-way source: no target-radiated noise or tonals through the propagation
  model (`sst_asiaex` makes a one-way path with a relay point).
- The scattering-function reverberation assumes a flat boundary in a horizontally
  stratified medium, incoherent eigenray pairs and one Doppler shift for the whole pulse
  of each echo (no range walk), and draws new noise every ping. By default it is also monostatic (the
  transmitter is taken at the array centre and to move with it), shifts each azimuth by
  the receiver's horizontal speed as if every path were horizontal, ignores arrival
  elevation when steering to the hydrophones, takes the boundary as uniform and the
  statistics as Gaussian; transmit and element beam patterns are applied at each path's
  elevation at the sonar (D-37, D-43). `geometry: bistatic`, `strength_map` and
  `statistics: {model: k}` lift the defaults in that sentence. Over a record whose
  reverberation falls by more than about 50 dB the default generator leaks the strong
  start into the far end; `band_taper: true` removes this (none of the four shipped
  scenarios measured is affected; `sst_pursuit` turns the taper on).
- With `geometry: bistatic`, two of the standing assumptions can also be lifted:
  `scatterer_motion` gives the scatterers a mean velocity and a Doppler spread (D-72),
  and `bathymetry` gives the boundary relief, with facet angles, shadowing and, with
  Bellhop, a run per bearing (D-73).
- Still missing or wrong in the reverberation:
  - In bistatic mode components are keyed by a bin of total Doppler shift, half the pulse
    resolution wide, at most `max_doppler_bins` (128), with the exact Doppler factor
    (D-80). Within a bin the shift is a mean (at most half a bin off), and the direction
    is a mean within an elevation bin. The cost is more sequences: about five times on
    the Pursuit ping. The default (monostatic) mode keeps its horizontal-path,
    first-order Doppler.
  - The first receive element's pattern stands for all (a warning fires if elements
    differ; no shipped sonar's do).
  - Over relief, legs with a bottom bounce before the scattering cell are left out by
    default (5 to 6 dB low beyond five water depths over a reflective level bed), or kept
    as flat-bottom images with an unbounded error. Shadowing uses straight rays. With
    Bellhop, transmitter and receiver must be over one ground point.
  - No wind-driven surface drift, orbital wave motion or Bragg phase speed; no attitude
    rates in the platform Doppler; the spread of platform Doppler inside one component is
    collapsed to its mean.
  - No ping-to-ping coherence: wrong for a still sonar over a still bottom.
  - The K texture is independent between pings and between neighbouring cells. Its
    `shape` has no default: there is no confirmed value for any seabed.
  - Near-specular reverberation is not removed when time spreading is on, and there is
    no near-specular surface scattering model.
  - The default generator's nearest-bin ripple (−1.4 to +1.7 dB between neighbouring 1 ms
    bins, right on average) is left as it is.
  - Bistatic mode has not been exercised with Bellhop on a flat bed, and its cost grows
    with sectors × rings × eigenrays of each leg.
- Boundary models: the bubble-loss law (`hall_novarini`) is an upper estimate above about
  10 kHz; the roughness interpolation and composite split are our own, not the published
  ones; elastic and layered bottoms are not modelled; two sediments are named, no rock.
- Time spreading: the default tail shape is a placeholder exponential; angle and
  frequency spreading use small-angle geometry, isotropic slopes and no shadowing, and
  have not been compared with measurements.
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
