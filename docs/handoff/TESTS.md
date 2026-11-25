# Tests handoff

How to run the tests, what each one protects, which decisions they pin, how to recognise a
failure that is not one, and what is not tested. Written at the tree of `933972b`, decisions to
D-181. Companions: [SOFTWARE.md](SOFTWARE.md) (architecture and settings),
[README.md](README.md) (the set).

**What is not known first.** No per-file run time is recorded anywhere in the repository; the
"class" column in section 2 is a judgement from what each test does. The log of the last full
GPU run (job `6b055ee4c97b8ed4fcd799db`) was not available when this was written: its 93 skips
are derived in section 1 from the test sources and from the saved log of an earlier run with
92 skips, not read from that job.

## 1. How to run

### One command

`tools/test.sh [extra ctest args]` does three steps (`tools/test.sh`):

1. `cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release ${SONAR_CMAKE_ARGS}` (the build
   directory is `SONAR_BUILD_DIR`, default `build/`);
2. `cmake --build build`, at full width;
3. `ctest --test-dir build --output-on-failure`: every GoogleTest case is one ctest entry
   (`gtest_discover_tests`), and the whole Python suite is one more entry named `pytest`
   (`python3 -m pytest -q tests/python` with `PYTHONPATH=python`; root `CMakeLists.txt`).

On the 4-core, 15 GB development container a full-width build or two test processes at once
exhaust the machine, and the whole suite takes about two hours (`docs/HANDOFF.md` section 3).
Use the parts instead:

```bash
export OMP_NUM_THREADS=2 PYTHONPATH=$PWD/python
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DSONAR_ENABLE_CUDA=OFF
cmake --build build -j2
ctest --test-dir build -E pytest --output-on-failure               # the C++ tests only
ctest --test-dir build -R 'Tracker\.' --output-on-failure           # one GoogleTest suite
python3 -m pytest tests/python/test_live_cpp.py -q -p no:cacheprovider   # one Python file
python3 -m pytest tests/python/test_detector.py -q -k range_estimate     # some tests of one file
GPUSONAR_SIM_ENGINE=numpy python3 -m pytest tests/python/test_sim.py -q  # the NumPy reference engine
```

Run one pytest process at a time. The program tests need `build/bin/sonar_live` at
`<repository root>/build/bin/` (section 4).

### Switches

| Variable | What it enables | Without it |
| --- | --- | --- |
| `GPUSONAR_BELLHOP=/path/bellhop.exe` | the `eigenray` tier: tests of the long-range, trial and range-dependent scenarios (`tools/build_oracles.sh <dir>` builds it and prints the exports) | those tests skip |
| `GPUSONAR_BELLHOP3D`, `GPUSONAR_KRAKEN_DIR` | the 3-D seamount comparison; rays against normal modes | skip |
| PyRAM 1.3.0 installed (`pyram`) | rays against a parabolic equation (`test_ram_crosscheck.py`) | skip |
| `GPUSONAR_REQUIRE_ORACLES=1` | an oracle test that would skip fails instead (`tests/python/oracle_gate.py`) | the run ends with a line counting the oracle skips |
| a Ceres without SuiteSparse at `SONAR_CERES_DIR` | motion estimation is built (`core.HAVE_MOTION`) | `test_motion.py`, `test_motion_delays.py` and three other tests skip |
| a CUDA build and a device | the host-against-device tests | they skip; two tests run only on a host-only build |
| the Holoscan SDK (Python wheel) | `test_holoscan.py`, parts of `test_hsb.py` | skip |
| `-DSONAR_BUILD_HOLOSCAN=ON` | `test_holoscan_cpp.py` (needs `build/bin/sonar_live_holoscan`) | skip |
| Sensor Bridge built (`tools/build_hsb.sh`) and a CUDA device | the loopback tests of `test_hsb.py` | skip |
| `GPUSONAR_HSB_STUDY=1` | two measurements in `test_hsb.py` that are not tests (`test_loopback_loss_study`; `test_popoto_rate_loopback_study`, with `GPUSONAR_HSB_STUDY_RUNS`, `_ROWS`, `_ENGINES`, `_CONFIGS`, `_BUDGET_S`) | skip, always, in the suite |
| `GPUSONAR_PICMUS_DIR`, `GPUSONAR_UACR_DIR`, `GPUSONAR_WATERMARK_DIR`, `GPUSONAR_SMARTOCEAN`, `GPUSONAR_SMARTOCEAN_CACHE` | tests on public data sets, never committed (`tools/fetch_picmus.py`, `tools/fetch_channels.py`) | skip |
| `GPUSONAR_NVUS_PYTHON`, `GPUSONAR_NVUS_DIR` | the NVIDIA ultrasound reference's module and checkout (`tools/build_nvus_reference.sh`) | skip |
| `arlpy` installed | independent checks of medium and noise formulas | skip |
| `GPUSONAR_SIM_ENGINE=numpy` or `cpp` | which simulator engine the tests use | the default engine (`sim/engine.py`) |

### The GPU suite on Hugging Face Jobs

`tools/gpu_job.sh` is written for a clean `nvidia/cuda:*-devel-ubuntu24.04` container. It
installs the toolchain, builds with `-DSONAR_ENABLE_CUDA=ON`, runs `ctest -E pytest`, then
`pytest -q -rs tests/python` with any arguments appended. `tools/gpu_job.sh --help` prints its
header, which is the authority for the list below.

| Variable | Effect |
| --- | --- |
| `GPUSONAR_TARBALL_REPO=<user>/<dataset>` | fetch `gpusonar.tar.gz` from that dataset first (the job then has no git checkout) |
| `SONAR_CMAKE_ARGS="..."` | extra CMake arguments |
| `GPUSONAR_ORACLES=1` | build Bellhop, Bellhop3D and KRAKEN, install PyRAM 1.3.0, and set `GPUSONAR_REQUIRE_ORACLES=1` |
| `GPUSONAR_CERES=1` | build Ceres 2.2.0 without SuiteSparse, so motion estimation is built and tested |
| `GPUSONAR_HOLOSCAN_CPP=1` | install the Holoscan C++ package and build the C++ operators |
| `GPUSONAR_HSB=1` | build Holoscan Sensor Bridge 2.7.0 (implies `GPUSONAR_HOLOSCAN_CPP=1`); where raw sockets are refused it applies the UDP-socket patch itself |
| `GPUSONAR_NVUS=1` | build the ultrasound reference's native module |

| `GPUSONAR_JOB=` | Runs instead of the tests | Reads | Uploads to `GPUSONAR_UPLOAD_PATH` (when `GPUSONAR_UPLOAD_REPO` is set) |
| --- | --- | --- | --- |
| unset or `tests` | ctest, then pytest | pytest arguments | nothing |
| `benchmark` | `tools/benchmark.py --backend cuda` | `GPUSONAR_SCALE` (2), `GPUSONAR_PIXEL` (0.05), `GPUSONAR_MODES` (`sas,fls,flank`) | nothing |
| `gallery` | `tools/gallery.py --backend cuda` | `GPUSONAR_GALLERY_ARGS`; set `GPUSONAR_ORACLES=1` and `GPUSONAR_CERES=1` | default `gallery` |
| `elevbench` | `tools/popoto_elevation_bench.py --full-rate --backend cuda` | `GPUSONAR_ELEVBENCH_ARGS` | default `elevbench` |
| `liveablate` | `tools/live_chain_ablation.py --backend cuda` | `GPUSONAR_LIVEABLATE_ARGS` | default `liveablate` |
| `benchall` | `tools/gpu_benchmark_all.py` | `GPUSONAR_SCALE` (1), `GPUSONAR_BENCHALL_ARGS`, `GPUSONAR_BENCHALL_PYTEST` | default `benchall` |
| `clseval` | `tools/classify_rules_eval.py --backend cuda` | `GPUSONAR_CLSEVAL_ARGS`; `GPUSONAR_ORACLES=1` for the Popoto scenes | default `clseval` |
| `falsealarm` | `tools/detector_search_factor.py --at-default --backend cuda` | `GPUSONAR_FA_ARGS` | default `falsealarm` |
| `motion` | `test_motion.py`, `test_motion_delays.py`, then `tools/motion_study.py timing` | needs `GPUSONAR_CERES=1` | nothing |

The full suite as it has been run since D-163:

```bash
GPUSONAR_ORACLES=1 GPUSONAR_CERES=1 GPUSONAR_HOLOSCAN_CPP=1 GPUSONAR_HSB=1 GPUSONAR_REQUIRE_ORACLES=1 tools/gpu_job.sh
```

- About 45 minutes on the `l4x1` flavour. L4 is the reference: every timing and every Sensor
  Bridge number in the documents is from an L4.
- How a job is launched (archive the tree, upload it to the private dataset, start a job that
  downloads it and runs the script) and the launcher's text: `docs/HANDOFF.md` section 4. The
  private dataset and Hugging Face Jobs are the only approved external destination for project
  content. Never print or paste a token.
- L4 capacity can be unavailable for hours: a job sits in `SCHEDULING`. `a10g-large` is an
  acceptable fallback; `a10g-small` is not (section 4). `hf jobs wait` has returned early while
  a job was still `SCHEDULING`: poll `hf jobs inspect <id> --format json` instead.
- The job runs from a tarball, so it is not a git checkout: `test_tools_help.py` skips there
  and must be run on a checkout.

### Current counts

| Run | Result | Source |
| --- | --- | --- |
| Full GPU suite, NVIDIA L4, tree of D-187 | **2155 passed, 97 skipped, 1 xfailed**, none failed (job `6b06728ad0582cfbbefdb5eb`; the xfail is one Popoto-rate loopback case with a frame damaged at the start, D-187). Before it, D-186's tree: 2150 passed, 97 skipped, 4 xfailed (job `6b06553a1460ffe1d5e9aa10`). Before it, D-183's tree: 2102 passed, 95 skipped (D-181's tree: 2078 passed, 93 skipped, which the skip groups below describe; the two further skips are in the tests added since) | `docs/DECISIONS.md` D-183, job `6b060a4b1460ffe1d5e93111` (before: `6b055ee4c97b8ed4fcd799db` |
| Previous | 2065 passed, 93 skipped (D-180); 1925 passed, 93 skipped (D-175 to D-178); 1875 passed, 92 skipped (D-171 to D-173) | the entries named |
| GoogleTest at this tree | 172 `TEST` cases in 15 binaries (counted in `tests/cpp/*.cpp`) | section 2 |
| pytest at this tree | 1375 test functions in 114 files (counted in `tests/python/test_*.py`); parametrisation makes them 2171 collected on the L4 job | D-181's counts |

No full CPU run of the tree at D-181 is recorded; `docs/STATUS.md` ("State of verification")
records the CPU and GPU runs up to D-143 and the decisions record the GPU runs after.

### What the 93 skips are

Derived, not read from the job: the skip conditions in the test sources, checked against the
saved log of an earlier full run that had 92 skips and the same job switches. The one skip
added since is a tool added in D-176 (`tools/popoto_bearing_study.py`: one more `--help` test
that skips without a git checkout).

| Group | Skips | Why | Files |
| --- | --- | --- | --- |
| Not a git checkout | 65 | the job runs from a tarball: 55 Python tools and 5 shell scripts x 2 flags cannot have their effect on the tree compared | `test_tools_help.py` |
| `arlpy` not installed | 6 | an optional independent check | `test_medium_models.py` (3), `test_bubble_layer.py`, `test_noise_statistics.py`, `test_tonal.py` |
| Public channel data not in the job | 10 | UACR (4 + 2 noise), Watermark (1), Smart Ocean (3) | `test_channel_replay.py`, `test_noise_fit.py`, `test_channel_sounding.py` |
| PICMUS data not in the job | 6 | | `test_picmus.py` |
| NVIDIA ultrasound reference not built or not checked out | 2 | `GPUSONAR_NVUS=1` is not part of the suite command | `test_nvus_scanner.py`, `test_nvus_adapter.py` |
| Apply to a host-only build | 3 | "built with CUDA" (2), "checks the refusal without a device" (1) | `test_conditioning.py`, `test_sim_cpp.py`, `test_motion_delays.py` |
| A measurement, not a test | 1 | needs `GPUSONAR_HSB_STUDY=1` | `test_hsb.py` (a second, `test_popoto_rate_loopback_study`, was added with D-186: one more skip from then on) |

The rows sum to 93. The earlier log's rows were the same with 64 in the first group. Read the
`-rs` lines at the end of the next job's log to confirm the list.

On a CPU container without the oracles many more tests skip: every CUDA comparison, every
Bellhop test, the Holoscan and Sensor Bridge tests, and motion estimation when Ceres is absent.
A plain `pytest` therefore passes with no oracle run at all; the summary line says how many
oracle tests were skipped.

## 2. Test map

Columns: tests = `def test_` functions in the file (parametrised cases are more). Needs = what
makes part or all of the file skip. Class: **fast** (closed forms on records built in the
test, seconds), **minutes** (simulated pings, subprocesses of the program, Monte Carlo),
**long** (many scenes or seeds). Engine-parity files are marked **parity**; files that hold a
frozen expected output are marked **golden**.

### C++ (GoogleTest, `tests/cpp`)

All are fast: the fifteen binaries together ran in under ten seconds in the saved log of an
earlier L4 job (not in the repository). `test_support.cpp` is a helper, not a suite. There is
no C++ test binary for `libs/motion` or `libs/holoscan_ops`: they are tested from Python.

| Binary | Suites | Tests | Protects | Component |
| --- | --- | --- | --- | --- |
| `test_geometry` | `RotationNed`, `WorldPosition`, `Directivity` | 8 | ASASIN Eq. 2 and its sign conventions | `libs/core` |
| `test_record_io` | `RecordIo`, `RoundTrip`, `Validator` | 11 | the container round trip, integer storage, schema checks, the validator | `libs/io` |
| `test_allocator` | `SlabAllocator` | 7 | the slab allocator (D-08) | `libs/core` |
| `test_cfar` | `Cfar` | 4 | the CFAR kernel | `libs/detect` |
| `test_sim_paths` | `Paths`, `Reverb` | 11 | eigenray providers and the reverberation generator on analytic cases | `libs/sim` |
| `test_sim_synth` | `Synth`, `Paths` | 6 | the sampled-pulse branch and the arrival-time solve (D-81, D-83) | `libs/sim` |
| `test_batched` | `Batched` | 8 | the device formulations run on the host through `HostExec`, against the reference host code | `libs/sim`, `libs/live` |
| `test_hardcoded` | `Backproject`, `Baseband`, `CfarWindow`, `Defaults`, `ImageSeries`, `Resampler`, `SynthLimits`, `Tvg` | 14 | values with no physical default are refused; fixed capacities raise or report (D-131) | `libs/dsp`, `libs/recon`, `libs/detect` |
| `test_live` | `Cluster`, `Tracker`, `Classify`, `Sbl`, `PingAssembler`, `StreamAssembler`, `Navigation`, `Options`, `Report`, `RangeDoppler` | 41 | the live library alone: tracker, classifier and echo grouping, the assembler's counting, navigation, the key=value options, the report line | `libs/live` |
| `test_beamform` | `Beamform`, `Mvdr`, `Calib` | 9 | far-field and MVDR beamformers, array calibration, in closed form | `libs/live` |
| `test_elevation` | `Rows`, `Elevation`, `Depth`, `PassiveElevation`, `Tracker`, `Chain` | 16 | stacked rows: rows from positions, elevation with roll, the second lobe, depth folding | `libs/live` |
| `test_stage2` | `Decimate`, `Blanking`, `Navigation`, `Ladder`, `ExtraJson` | 7 | decimate first, blanking, navigation, the sound-speed ladder (D-129) | `libs/live` |
| `test_derive` | `Derive`, `Capacity`, `Stream` | 12 | the derived settings to the last bit for the shipped arrays; capacity limits | `libs/live` |
| `test_detector` | `Statistic`, `Detector`, `DetectorKernels`, `DetectorDevice` | 12 | the multiscale detector's statistics and plan against closed forms; the device statistic | `libs/live`, `libs/detect` |
| `test_tonal` | `Goertzel`, `Demon`, `TonalAnalyzer`, `TonalOptions` | 6 | passive tonal analysis | `libs/live` |

### The live chain (Python)

| File | Tests | Protects | Needs | Class | Marks |
| --- | --- | --- | --- | --- | --- |
| `test_live_cpp.py` | 57 | the C++ library against the Python stages, both chains' reports, the program, every option's round trip, unknown keys and words | `build/bin/sonar_live` for the program tests | minutes | **parity**, **golden** |
| `test_live.py` | 8 | one object on simulated pings, a replayed recording, UDP blocks, navigation datagrams, a watched directory | CUDA for one test | minutes | |
| `test_detect.py` | 27 | MVDR, CFAR, clustering, the tracker (association, detection times, misses), the classes golden | CUDA for one test | minutes | **golden** |
| `test_detector.py` | 24 | the multiscale detector on both engines: thresholds, false alarms by Monte Carlo, range estimate, resolution | | minutes | **parity**, **golden** |
| `test_detector_device.py` | 2 | the device statistic gives the host's detections | CUDA build and device | minutes | |
| `test_classify.py` | 35 | classes from the track: fit through measured positions, outlier test, fit limit, echo grouping, bounces | | fast | |
| `test_doa.py` | 12 | sparse Bayesian bearings on active gates and passive frames | CUDA for one test | minutes | |
| `test_elevation.py` | 30 | elevation and depth for every build stage; both engines; the program | `sonar_live` for one test | minutes | **parity** |
| `test_multipath_sigma.py` | 5 | the depth sigma where boundary echoes share the range cell (D-133) | | minutes | **parity** |
| `test_multipath_depth.py` | 18 | depth from multipath timing: pattern, C++ against NumPy, truth on simulated pings | Bellhop for one test | minutes | **parity** |
| `test_refraction_rays.py` | 15 | rays through the ladder's profile in both depth estimators | Bellhop (2), `sonar_live` (1) | minutes | **parity** |
| `test_raytrace.py`, `test_raytrace_cpp.py` | 6, 5 | the layered ray code against closed forms; C++ against Python | Bellhop for one test | fast | **parity** |
| `test_profile_ladder.py` | 9 | the sound-speed ladder: rungs, the built profile, the depth sigma rule | | minutes | |
| `test_coverage.py` | 9 | the coverage picture against `lloyd.py` and Bellhop | Bellhop for one test | minutes | |
| `test_deployment_defaults.py` | 10 | what the shipped deployment files switch on; coverage refuses to run without a site (D-118 to D-120) | | minutes | |
| `test_decimate.py` | 10 | the decimate-first stage: filter, tones, streaming against whole-ping, the program | `sonar_live` for one test | minutes | **parity** |
| `test_blanking.py` | 7 | blanking and synchronisation when the direct sound clips a reference burst | | minutes | |
| `test_calibration_sync.py` | 12 | array calibration; the sync stage recovers injected offsets and drifts | | minutes | |
| `test_beamform_cpp.py` | 13 | C++ far-field and MVDR beamformers against NumPy; the chain with them | `sonar_live` for one test | minutes | **parity** |
| `test_flank.py` | 11 | far-field and backprojection agree; beam width; detection range against the sonar equation | | minutes | |
| `test_hardcoded.py` | 22 | another sonar or board is a configuration change: derived sizes, capacities, the program | `sonar_live` for some | minutes | |
| `test_dict_keys.py` | 8 | every options dictionary of the live library refuses a key it does not read (D-177) | | fast | **parity** |
| `test_nav_adapter.py` | 14 | the navigation adapter; what datagrams carry into a record; both receivers | | fast | **parity** |
| `test_nav_lever_arm.py` | 6 | lever arm and mounting rotation in both engines (D-167) | | minutes | **parity** |
| `test_tonal.py` | 19 | Goertzel lines and DEMON, Python against C++, the program | `sonar_live` for some; `arlpy` for one | minutes | **parity** |
| `test_lfm_range_doppler.py` | 3 | the LFM coupling is f0 v T / B, and the correction removes it (D-169, D-181) | | minutes | |
| `test_architecture.py` | 4 | stages drawn with a CUDA implementation take a backend; the documents' figures exist | | fast | |

### Sources, streams, Holoscan

| File | Tests | Protects | Needs | Class | Marks |
| --- | --- | --- | --- | --- | --- |
| `test_record.py` | 5 | the record container from Python; plain `h5py` readability | | fast | |
| `test_sources.py` | 9 | file replay: the container and WAV with a sidecar | | fast | |
| `test_wav_watch.py` | 3 | recordings that appear in a watched directory | | fast | |
| `test_taps.py` | 5 | taps can be inspected, exported and fed back in | | fast | |
| `test_flank_stream.py` | 5 | the capture stream from an emulated board through UDP back to records | skips if the loopback dropped samples | minutes | |
| `test_board_streams.py` | 23 | per-board streams merged into the whole-array stream (D-184): the block's board field (whole-array blocks unchanged), the whole-array receivers refusing a board block, merged pings equal to the whole-array pings, loss, restarts, a silent board, a late start, refusals by name, the options, the program and `--emulate --per-board` over UDP, the relay tool | the program for one test; loopback tests skip if data was dropped | minutes | **parity** |
| `test_board_clock.py` | 20 | a board clock's rate against true time (D-185): the stamp's drift an hour in and the pose it moves, `clock_ppm` in both engines and the program, boards at their own rates aligned by them, the estimate from arrival times and its opt-in discipline | the program for one test | fast | **parity** |
| `test_hsb_receive.py` | 24 | what the receiver and assembler count when frames go missing, repeat or arrive out of order, with no network (D-161, D-177) | | fast | **parity** |
| `test_hsb.py` | 24 (35 collected; 26 before D-186) | framing; the receive application's stop; the real-time loopback through Sensor Bridge's emulator and receiver (D-153, D-162, D-163, D-173); a Popoto build's trial at 32, 64 and 96 channels and 102.4 kHz, framed and through the loopback in both engines (D-186) | Holoscan SDK; Sensor Bridge and a CUDA device for the loopbacks; OpenMP for one | minutes | |
| `test_holoscan.py` | 5 | stages as Holoscan operators, fed by the simulator and by replay | Holoscan SDK | minutes | |
| `test_holoscan_cpp.py` | 3 | the C++ Holoscan application against the standalone program; an operator's error ends it with status 1 and the reason, not an abort (D-186) | built with `-DSONAR_BUILD_HOLOSCAN=ON` | minutes | **parity** |
| `test_nvus_adapter.py`, `test_nvus_scanner.py` | 7, 6 | the ultrasound reference's frames to records; its runtime as a source | the reference's checkout and native module | fast | |
| `test_picmus.py`, `test_fetch_picmus.py` | 9, 5 | the PICMUS benchmark through the kernel; the fetch tool without the network | PICMUS data for the first | minutes, fast | |

### Imaging, conditioning, motion

| File | Tests | Protects | Needs | Class |
| --- | --- | --- | --- | --- |
| `test_backproject.py` | 12 | the kernel: delay and phase precision (D-06), agreement with an independent reference (`reference_bp.py`), GPU against CPU | CUDA for the GPU half | minutes |
| `test_culling.py` | 6 | ray culling leaves the image unchanged (D-09, D-92) | | minutes |
| `test_position_lookup.py` | 6 | positions from a tabulated trajectory (D-07) | CUDA for one test | fast |
| `test_multigpu.py` | 4 | tiles give the same image as one call | CUDA for the device part | fast |
| `test_fls.py` | 7 | FLS range resolution and beam width within 10 % of theory (D-26) | | minutes |
| `test_sas.py` | 8 | SAS with known navigation; along-track resolution (D-29) | | minutes |
| `test_minsas_resolution.py` | 5 | the MINSAS-class sonar resolves what its design says | | minutes |
| `test_volume.py` | 7 | the buried cylinder at the correct depth with refraction (D-23) | | minutes |
| `test_conditioning.py` | 19 | whitening, TVG, DRC, pulse-compression gain (D-20 to D-22, D-91) | CUDA for the device half | fast |
| `test_motion.py` | 21 | motion estimation: sway and heave bounds, Eq. 4 to 8, DVL term, the energy gate (D-164, D-171) | Ceres (whole file) | long |
| `test_motion_delays.py` | 7 | delay estimation in C++ against NumPy; CUDA against host (D-148) | Ceres; CUDA for two | minutes |
| `test_pipeline.py` | 6 | the config-driven chain, its log, debug products, GeoTIFF | `rasterio` | minutes |
| `test_hardcoded_defaults.py` | 22 | the hard-coded audit outside the live chain: typed constants are keys or derived | Ceres for one | minutes |

### Simulator

| File | Tests | Protects | Needs | Class |
| --- | --- | --- | --- | --- |
| `test_sim.py` | 21 | point targets on every sonar: delay and amplitude against analytic values; CPU against GPU | CUDA for the GPU half | minutes |
| `test_sim_cpp.py` | 20 | C++ paths and reverberation against the NumPy versions (D-61) | the C++ engine; CUDA for some | minutes |
| `test_sampled_waveform.py`, `test_sampled_engine.py` | 17, 15 | sampled transmit pulses against TR 0702; the C++ kernel against the NumPy engine (D-74, D-75, D-81) | Bellhop for one each; CUDA for some | minutes |
| `test_batched.py` | 28 | the batched device formulations against the reference host code (D-122) | CUDA for the device half | minutes |
| `test_parallel_normal.py` | 8 | threaded normal deviates are bit for bit `rng.standard_normal` | | fast |
| `test_scene_objects.py`, `test_scene_targets.py`, `test_scene_sediment.py` | 19, 17, 5 | extended objects, highlight targets against arXiv:1805.06945, sediment scatterers (D-41, D-64, D-65) | Ceres and CUDA for one each | minutes |
| `test_moving_ownship.py` | 7 | a moving ownship over a seabed fixed in the world | | minutes |
| `test_propagation.py`, `test_propagation_defaults.py` | 11, 5 | providers against Bellhop; a bottom needs a description, arrivals kept by energy (D-104, D-105) | Bellhop for the first | minutes |
| `test_long_range.py` | 2 | long-range scenarios at the eigenray tier | Bellhop (whole file) | minutes |
| `test_ssp_range.py` | 14 | sound speed that changes with range in the Bellhop provider (D-110) | Bellhop for some | minutes |
| `test_kraken_crosscheck.py`, `test_ram_crosscheck.py`, `test_seamount_3d.py` | 2, 2, 3 | rays against normal modes, a parabolic equation and Bellhop3D (D-149) | Bellhop with KRAKEN, PyRAM, Bellhop3D | minutes |
| `test_reverberation.py`, `test_reverb_statistics.py` | 10, 13 | reverberation level within 2 dB of the sonar equation; K statistics (D-37, D-70) | | minutes |
| `test_reverb_bistatic.py`, `test_reverb_motion.py`, `test_reverb_relief.py`, `test_reverb_specular.py` | 23, 10, 16, 6 | bistatic, moving scatterers, relief, near-specular removal (D-69, D-72, D-73, D-123) | CUDA for one; Bellhop for some | minutes |
| `test_bottom.py`, `test_bottom_models.py`, `test_sediment_table.py`, `test_sediment_literature.py`, `test_sediment_fit.py` | 9, 16, 11, 9, 8 | the bottom model, its regimes, the named sediments against open literature, the fit (D-63, D-66, D-67, D-132) | Bellhop for one | fast |
| `test_surface.py`, `test_surface_models.py`, `test_bubble_layer.py` | 5, 13, 9 | surface backscatter, bubble loss (D-38, D-68, D-87) | Bellhop for one; `arlpy` for one | fast |
| `test_timespread.py`, `test_coherent_phase.py` | 34, 10 | time, angle and frequency spreading after TR 0702; the coherent arrival's wander (D-62, D-76, D-108) | CUDA for one | minutes |
| `test_noise.py`, `test_baseband_noise.py`, `test_noise_statistics.py`, `test_noise_fit.py` | 5, 7, 10, 27 | noise spectra, array coherence, K statistics, a fit to a recording (D-103, D-124) | `arlpy` for one; UACR noise for two | minutes |
| `test_direct_overload.py`, `test_source_level.py` | 19, 4 | the projector's direct sound and receiver overload (D-125); source levels are RMS (D-102) | | minutes |
| `test_oneway_sources.py` | 16 | one-way sources against the passive sonar equation (D-78) | `sonar_live` for one | minutes |
| `test_basin.py` | 20 | a basin's late reverberation (D-106) | Bellhop for one | minutes |
| `test_channel_replay.py`, `test_channel_sounding.py` | 23, 20 | measured-channel replay and sounding estimators on synthetic channels | UACR, Watermark, Smart Ocean data for seven | minutes |
| `test_sst_validation.py` | 23 | two worked examples of APL-UW TR 0702 end to end (D-78, D-144) | | long |
| `test_simcal.py` | 7 | simulator calibration finds hidden settings in a simulated log (D-56) | | minutes |
| `test_literature_models.py`, `test_medium_models.py`, `test_lloyd.py` | 11, 13, 21 | formulas against printed values; medium helpers; the surface-image factor (D-126) | `arlpy` for three; Bellhop for one | fast |

### Configuration, builds, tools, whole scenes

| File | Tests | Protects | Needs | Class | Marks |
| --- | --- | --- | --- | --- | --- |
| `test_environment.py` | 8 | one environment per site; contradictions (D-100, D-101) | | fast | |
| `test_yaml_numbers.py` | 3 | every tracked YAML's numbers load as numbers with plain PyYAML (D-180) | | fast | |
| `test_design.py` | 7 | every owner requirement is met by the config files; `docs/DESIGN.md` is up to date (D-44) | `matplotlib` | minutes | reads committed files |
| `test_buildspec.py` | 21 | build specs validate; the generator's output is what is committed; a different build runs through both engines (D-127) | `matplotlib`; Bellhop for one | long | reads committed files |
| `test_realistic_scenes.py` | 25 | every mode's scenario simulates and images end to end (`tools/gallery.py --quick`); `test_documented_full_size_classes` reads `docs/figures/gallery.json` | `matplotlib`; Bellhop for five; Ceres for one | long | **golden** (the gallery) |
| `test_classify_rules_eval.py` | 15 | the scoring of `tools/classify_rules_eval.py` on a made-up case, no simulation | | fast | |
| `test_gallery_echo_position.py`, `test_gallery_subgrid_bearing.py` | 4, 7 | the gallery's echo placement and leading edge (D-172, D-178, D-181); its opt-in sub-grid bearing (D-176) | | fast | |
| `test_gpu_benchmark_all.py` | 4 | the benchmark tool's table and JSON on whatever backend the build has | | minutes | |
| `test_study_resume.py` | 7 | the resumable studies reuse stored results and recompute with `--fresh` | `matplotlib` | fast | |
| `test_tools_help.py` | 5 | every tool answers `--help` and changes nothing in the tree | a git checkout | minutes | |
| `test_oracle_gate.py` | 8 | an oracle skip becomes a failure under `GPUSONAR_REQUIRE_ORACLES` | | fast | |

Helpers, not tests: `conftest.py` (`make_record`), `oracle_gate.py`, `reference_bp.py` (the
independent backprojection), `kraken_oracle.py`, `bellhop3d_oracle.py`.

### Goldens: frozen expected outputs

| File | Holds | Frozen by | Read by |
| --- | --- | --- | --- |
| `tests/python/data/tracker_before_d156.json` | both trackers' output on the tracker scenario at commit `1be89d4`, before detections had their own time | D-156 | `test_detect.py::test_detections_without_their_own_time_track_exactly_as_before`, `test_live_cpp.py::test_cpp_detections_without_their_own_time_track_exactly_as_before` |
| `tests/python/data/classes_d157.json` (made by `make_classes_d157.py`) | the default classes of that scenario from both engines, fitted through measured positions | D-157 | `test_detect.py::test_the_default_classes_of_the_tracker_scenario_are_held`, `test_live_cpp.py::test_cpp_default_classes_of_the_tracker_scenario_are_held` |
| `tests/python/data/centroid_d158.json` (made by `make_centroid_d158.py` on a build of `42d4539`) | both engines' report lines on two short scenarios with the centroid range | D-160 | `test_detector.py::test_range_estimate_centroid_given_is_the_output_from_before_d158` |
| `tests/python/data/crossing_boat_wake.json` | an input, not an expected output: the recorded detections near the crossing boat of `fls_170k_traffic` (seed 63, full size), its hull and wake | D-141 | `test_detect.py` (`_crossing_boat_tracks`: the association tests) |
| `docs/figures/gallery.json` | the full-size gallery's classes and numbers (regenerated at D-179) | D-174, D-179 | `test_realistic_scenes.py::test_documented_full_size_classes` |
| `docs/benchmarks/gpu_benchmark_all_l4.json` | the L4 timings `docs/ARCHITECTURE.md` quotes | | `test_architecture.py` |
| generated sonar, deployment and design files | what `tools/build_from_spec.py` and `tools/design_report.py` write | D-127, D-44 | `test_buildspec.py`, `test_design.py` |

A golden changes only with a decision entry that says what moved. A gallery regeneration can
break `test_documented_full_size_classes`: update its expectations honestly, with a comment
saying what got worse (`docs/HANDOFF.md` section 4).

## 3. Tests that pin a decision (D-155 to D-188)

Python tests are in `tests/python/`, C++ tests (written `Suite.Name`) in `tests/cpp/`.

| Decision | What it fixed | Tests |
| --- | --- | --- |
| D-155 | echo grouping: the object is the earliest established arrival | `test_classify.py::test_the_earliest_established_arrival_is_the_object_not_the_strongest`; `test_live_cpp.py::test_echo_grouping_matches_python_and_takes_the_earliest_established_arrival`, `::test_both_chains_group_echoes_alike`; `test_classify_rules_eval.py::test_an_established_track_is_scored_before_a_nearer_uncertain_one` |
| D-156 | each detection at transmit + R/c | `test_detect.py::test_detections_without_their_own_time_track_exactly_as_before` (golden), `::test_a_long_range_mover_is_where_its_echoes_show_it`, `::test_a_missed_track_is_never_moved_back_and_its_depth_variance_stays_positive`; `test_classify.py::test_positions_are_read_at_the_time_asked`; `test_live_cpp.py::test_both_trackers_place_a_long_range_mover_where_its_echoes_show_it`, `::test_both_chains_report_track_positions_at_the_ping_time` |
| D-157 | the line through measured positions; the leading edge as an option; extended objects scored from their outline | `test_detect.py::test_the_default_classes_of_the_tracker_scenario_are_held` (golden); `test_classify.py::test_a_noise_free_12km_crossing_is_reported_where_the_ship_is`, `::test_rows_of_misses_are_left_out_of_the_fit`; `test_live_cpp.py::test_both_engines_fit_the_measured_positions_of_a_noise_free_12km_crossing`; `test_detector.py::test_the_leading_edge_is_the_first_arrival_and_the_centroid_lies_behind_it`; `test_classify_rules_eval.py::test_an_extended_object_is_scored_from_its_outline` |
| D-158 | the first scale is the single sample; pings without velocity counted | `test_detector.py::test_the_single_sample_test_is_always_the_first_scale_and_the_leading_edge_holds_to_it`, `::test_a_plan_whose_first_scale_is_not_the_single_sample_is_refused_by_both_engines`; `Detector.TheSingleSampleTestIsTheFirstScaleWhateverTheLengthsAndAnotherPlanIsRefused`; `test_nav_adapter.py::test_the_chain_reports_pings_whose_navigation_had_no_velocity`; `test_live_cpp.py::test_program_counts_pings_whose_navigation_has_no_velocity_like_the_python_chain`; `Report.CarriesPingsWithoutVelocityOnlyWhenThereWereAny` |
| D-159 | bearing sigma kept; opt-in outlier test; the fit limit grows with range | `test_detect.py::test_the_12km_report_does_not_depend_on_the_trackers_bearing_sigma`; `test_classify.py::test_a_glinting_last_measurement_is_left_out_of_the_line`, `::test_the_outlier_test_drops_about_alpha_of_clean_tracks_whatever_the_error_shape`, `::test_the_fit_limit_grows_with_range_where_asked`; `test_live_cpp.py::test_both_engines_leave_the_same_glinting_measurement_out_of_the_line`, `::test_both_engines_grow_the_fit_limit_with_range_alike`, `::test_the_chain_derives_the_fit_limits_growth_from_the_beam`; `Classify.LeavesOneGlintingMeasurementOutOfTheLine` |
| D-160 | the leading edge is the chain's default range | `test_detector.py::test_the_chain_default_range_estimate_is_the_same_in_both_engines`, `::test_range_estimate_centroid_given_is_the_output_from_before_d158` (golden), `::test_first_given_with_mean_or_peak_is_refused_by_both_engines`, `::test_mean_without_the_key_gives_the_centroid_on_both_engines`; `test_elevation.py::test_the_elevation_is_searched_about_the_centroid_whatever_range_is_reported` |
| D-161 | blocks placed by their sample counter; lost pings counted | `test_hsb_receive.py` (the whole file); `test_live_cpp.py::test_cpp_assembler_matches_python`; the `PingAssembler` and `StreamAssembler` suites of `test_live.cpp` |
| D-162 | `--chain-omp-threads` | `test_hsb.py::test_chain_omp_threads_sets_openmps_thread_count_for_the_chains_thread`, `::test_loopback_through_sensor_bridge_gives_the_direct_chains_reports` (asserts the worst receive delay) |
| D-163 | receive always stops | `test_hsb.py::test_receive_application_stops_when_the_stream_ends_early`, `::test_receive_stops_when_its_pings_are_accounted_for_or_the_stream_is_quiet`, `::test_the_emulated_board_stays_up_quiet_for_hold_seconds_after_its_linger` |
| D-164 | ping 0's height is free | `test_motion.py::test_one_noisy_dvl_sample_on_the_first_ping_does_not_offset_the_track`, `::test_smoothness_term_reduces_noise_and_dvl_term_holds_altitude`, `::test_sway_and_heave_errors_are_a_small_fraction_of_a_wavelength` |
| D-165 | no velocity from echo delays alone | `test_detect.py::test_pings_at_one_transmit_time_give_no_fitted_velocity`, `::test_ping_times_change_no_fit_when_transmit_times_increase`; `test_detector.py::test_pings_without_timestamps_give_tracks_no_fitted_velocity`; `test_live_cpp.py::test_both_classifiers_refuse_a_velocity_from_echo_delays_alone` |
| D-166 | both engines refuse an unknown chain key | `test_live_cpp.py::test_cpp_chain_refuses_an_unknown_key_by_name`, `::test_cpp_chain_refuses_an_unknown_key_in_sync_and_profile`, `::test_python_chain_refuses_an_unknown_key_by_name` |
| D-167 | lever arm and mounting rotation | `test_nav_lever_arm.py` (the whole file); `test_live_cpp.py::test_program_moves_navigation_to_the_array_by_the_lever_arm`; `test_hsb.py::test_tool_arguments` (asserts `popoto_vehicle.yaml`'s nav section) |
| D-168 | the detector's constants at the default budget | `test_detector.py::test_the_default_budget_count_on_a_second_map_at_cpu_scale` (CPU scale; the count itself is `GPUSONAR_JOB=falsealarm`) |
| D-169 | the LFM coupling explained | `test_lfm_range_doppler.py` (the coupling within 5 % of 2 f0 v T / B between up and down chirps) |
| D-170 | FLS classes do not turn on the sound speed | `test_realistic_scenes.py::test_cluttered_classes_do_not_turn_on_the_sound_speed` |
| D-171 | the window energy gate; wider SAS seabeds | `test_motion.py::test_windows_beyond_the_seabed_bias_the_track_and_the_energy_test_drops_them`; `test_realistic_scenes.py::test_sas_scenes_with_unknown_motion` (its criterion was changed by D-171) |
| D-172 | echo groups within one gate; gallery echo placement | `test_classify.py::test_two_ships_at_one_bearing_and_velocity_farther_apart_than_one_gate_are_two_objects`, `::test_a_chain_in_range_reaches_no_farther_than_one_gate_from_the_object`; `test_live_cpp.py::test_two_ships_at_one_bearing_beyond_one_gate_are_two_objects_in_both_engines`, `::test_transitive_echo_grouping_matches_python`; `test_gallery_echo_position.py` |
| D-173 | Sensor Bridge end to end with `serve --scenario` and `engine: cpp`; the CPU quota | `test_hsb.py::test_loopback_of_a_scenario_the_board_simulates_gives_the_direct_chains_reports`, `::test_usable_cpus_honours_a_cgroup_quota`, `::test_receive_passes_its_options_to_the_receiver` |
| D-174, D-179 | the gallery regenerated | `test_realistic_scenes.py::test_documented_full_size_classes` (no expectation changed in either) |
| D-175 | earlier detections at the track's time; bounces never an object; pings without time | `test_detect.py::test_a_track_never_steps_back_in_time_when_it_takes_an_earlier_detection`, `::test_a_mover_on_pings_without_timestamps_has_no_clock`; `test_classify.py::test_a_bounce_is_never_an_echo_groups_object`; `test_elevation.py::test_a_bounce_is_never_an_echo_groups_object_in_either_engine`; `test_live_cpp.py::test_both_trackers_take_an_earlier_detection_at_the_tracks_own_time`, `::test_a_source_without_timestamps_is_counted_and_clocked_by_a_configured_ping_interval`, `::test_the_ping_interval_is_refused_unless_positive`, `::test_program_counts_pings_without_time_and_takes_the_ping_interval`; `Tracker.AnEarlierDetectionIsTakenAtTheTracksTimeWithItsDepthVariance`, `Classify.ABounceIsNeverAnEchoGroupsObject`, `Report.CarriesPingsWithoutTimeOnlyWhenThereWereAny` |
| D-176 | the gallery's opt-in sub-grid bearing | `test_gallery_subgrid_bearing.py`; `test_classify_rules_eval.py::test_cluster_profiles_record_each_hits_bearing_profiles` |
| D-177 | every options dictionary refuses an unknown key; index skips; failed stops | `test_dict_keys.py`; `Options.AnUnknownKeyIsRefusedWithTheNearestKnownOne`; `test_hsb_receive.py::test_an_index_skip_over_d_intervals_counts_only_with_samples_lost`, `::test_an_index_skip_with_jitter_seen_is_not_a_loss`; `test_hsb.py::test_run_receiver_keeps_only_the_applications_whose_stop_failed`. The beads hook has no test in `tests/` (D-177 says it was tested with a stub) |
| D-178 | the gallery's range is the leading edge | `test_gallery_echo_position.py::test_the_leading_edge_of_a_merged_multipath_echo_is_its_direct_arrival`; `test_classify_rules_eval.py::test_a_capture_without_a_recorded_range_estimate_is_the_centroid`, `::test_flank_patrol_tracks_are_scored_in_the_order_the_scoring_reads`; `test_realistic_scenes.py::test_flank_patrol_detects_and_tracks` |
| D-180 | YAML numbers; `sonar_live_hsb`'s navigation | `test_yaml_numbers.py`; `test_sediment_fit.py::test_the_row_writes_numbers_plain_yaml_reads_as_numbers`; `test_deployment_defaults.py::test_the_rehearsal_sync_reference_reaches_the_cpp_receivers`; `test_hsb.py::test_receive_builds_its_navigation_as_sonar_live_does` |
| D-181 | the opt-in LFM range-Doppler correction | `test_live_cpp.py::test_range_doppler_is_off_by_default_and_said_only_when_on`, `::test_range_doppler_ownship_adds_the_factor_times_the_ownships_closing_speed_to_each_range`, `::test_range_doppler_full_moves_each_track_by_its_own_closing_speed`, `::test_range_doppler_full_gives_the_same_report_in_both_engines`, `::test_range_doppler_refuses_an_unknown_word`; `test_lfm_range_doppler.py::test_ownship_under_way_couples_like_a_closing_target_and_the_correction_removes_it`; `Options.RangeDopplerIsOffUnlessGivenAndRefusesOtherWords` and the `RangeDoppler` suite; `test_gallery_echo_position.py::test_doppler_range_takes_out_the_ownships_lfm_coupling_only_when_asked`; `test_classify_rules_eval.py::test_a_capture_with_the_targets_doppler_part_is_scored_with_it` |
| D-184 | per-board streams merged into the whole-array stream (sonar-6f9) | `test_board_streams.py` (the whole file) |
| D-185 | a board clock's rate: configured, estimated, opt-in discipline (sonar-ubg) | `test_board_clock.py` (the whole file) |
| D-186 (sonar-ogg, sonar-x8v) | `UdpSourceOp` joins its threads in its destructor too; the Sensor Bridge loopback at a Popoto build's channel count and rate | `test_holoscan_cpp.py::test_an_operator_that_throws_ends_the_application_with_its_reason_not_an_abort` (passed before the fix too: Holoscan 4.6 stops the operators after an exception; no fail-before was found); `test_hsb.py::test_a_popoto_trial_is_framed_at_its_builds_rate_and_its_pings_come_back` (1, 2, 3 rows), `::test_loopback_at_a_popoto_builds_channel_count_and_rate_gives_the_direct_chains_reports` (C++ at 1, 2, 3 rows, Python at 1 and 2; **an expected failure (xfail) with the numbers when frames are lost**, which in the containers so far is nearly every run), `::test_popoto_rate_loopback_study` (`GPUSONAR_HSB_STUDY=1`) |
| D-187 (sonar-bep, sonar-us7) | the Popoto-rate loss is at the start of the run; the receiver's queue defaults to 4 s of the stream | `test_hsb_receive.py::test_the_receiver_queue_defaults_to_four_seconds_of_the_stream`; `test_hsb.py::test_the_loss_location_places_missing_frames_in_the_pings_cycle`; the Popoto loopback test passes when every missing frame is in the first 0.25 s of stream, none damaged |
| D-188 (sonar-h0e) | the receiver names its damaged frames (`damaged_frames`); a damaged frame in the first 0.25 s of stream is start-up loss, one later or of unknown number is not | `test_hsb.py::test_a_damaged_frames_number_is_reported_wherever_it_lies`, `::test_damaged_frames_keep_the_first_sixteen`, `::test_start_up_loss_is_accepted_only_where_every_index_confirms_it` |

The option round trip that every chain key must pass, whatever decision added it:
`test_live_cpp.py::test_every_chain_option_round_trips_through_binding_and_text`.

## 4. Known false failures, and how to recognise them

| Symptom | Cause | What to do |
| --- | --- | --- |
| Program tests skip with "build/bin/sonar_live not built", in a worktree built elsewhere | the tests look for `<repository root>/build/bin/sonar_live`; a worktree's build directory is somewhere else | make a temporary `build` symlink to the build directory, run, remove it |
| `test_study_resume.py::test_gallery_quick_does_not_default_to_the_committed_gallery` fails while that symlink exists | it compares an unresolved path through the symlink | not a failure of the code: remove the symlink and rerun that file |
| The repository's stop hook reports an untracked file | the same `build` symlink | remove it |
| Tests fail in the main checkout after a merge, on code that passes elsewhere | the built module `python/gpusonar/_gpusonar*.so` is older than the C++ sources | rebuild (`cmake --build build -j2`) before believing the failure |
| A build or a test dies with a write error, or a job upload fails | the container's disk is small (a few GB free) | remove build directories and uploaded tarballs after use |
| A GPU job on `a10g-small` hangs at 0 % CPU during the Ceres build | 16 GB of RAM: out of memory | use `l4x1`, or `a10g-large` |
| Sensor Bridge loopback loses frames or shows receive delays of tens of milliseconds on `a10g-small` | a 3-CPU cgroup quota the CPU affinity does not show; with too many OpenMP threads the receiver waits behind throttling (D-173) | the tests take half of min(affinity, quota); on other hosts pass `--chain-omp-threads` |
| A job never starts | L4 capacity: the job sits in `SCHEDULING`; `hf jobs wait` may return early | poll `hf jobs inspect <id> --format json`; fall back to `a10g-large` |
| `test_loopback_at_a_popoto_builds_channel_count_and_rate_gives_the_direct_chains_reports` ends XFAIL with "measured limit (D-186)" | at a Popoto build's rate the receiver's socket buffer (capped by the container's `rmem_max`) overflows; every lost datagram is counted as refused by it | not a code failure: the measured limit of the container. Since D-187 the test passes when the loss is only the first frames after the receiver connects (the first 0.25 s of stream); since D-188 (sonar-h0e) a frame that arrived in part (counted damaged) is accepted the same way when its number (the receiver's `damaged_frames`) is in that start (before it, such runs ended XFAIL: 3 of 15 runs of the 64 and 96 channel cases on an L4, job `6b0681011460ffe1d5e9f89f`, the damaged frame's number then not logged). It still ends XFAIL, its message naming the damaged frames, when a frame is missing or damaged later, when a damaged frame's number is not known (no frame number, or more than 16 damaged), when a ping was not reported, or when the worst delay after the start reached 5 ms. The `POPOTO_START` line prints the missing and damaged frame numbers. A FAIL of that test (an assertion before the xfail) is real: an uncounted loss, a malformed frame, the chain behind real time, or reports that differ |
| A test skips with "the loopback dropped N samples" | a UDP loopback on a loaded machine lost datagrams (`test_flank_stream.py`, `test_live.py`, `test_holoscan.py`, `test_live_cpp.py`, `test_tonal.py`, `test_holoscan_cpp.py`) | a skip by design; rerun alone |
| Many failures naming `GPUSONAR_BELLHOP` or `pyram` | `GPUSONAR_REQUIRE_ORACLES=1` is set and the oracle is missing | build the oracles (`tools/build_oracles.sh`) or unset the variable |
| `test_tools_help.py` skips entirely | not a git checkout (the GPU job's tarball) | run it on a checkout |
| `test_design.py` or `test_buildspec.py` fails after a config edit, saying a committed file differs from what is generated | committed generated files no longer match their generator | a real signal: regenerate with `tools/design_report.py` or `tools/build_from_spec.py` and review the diff |
| `test_documented_full_size_classes` fails after a gallery regeneration | it reads the committed `docs/figures/gallery.json` | update its expectations with a comment saying what got worse |
| An engine comparison fails by about 1e-5 on a CUDA build only | the engines hand a shared function single-precision values that differ (`docs/STATUS.md`: `test_multipath_sigma.py`, loosened to 1e-4) | check the size against the tolerance's stated reason before changing either |

Two historical ones, fixed, worth knowing the shape of: a test that depended on test order
because a warning is given once per process (`test_environment.py`, fixed in `d9cb5b7`); a
real-time loopback at the edge of a 212,992-byte socket buffer (D-160, cured by D-162).

## 5. Gaps: what is not tested, or only on simulated input

**Everything is simulation.** No test reads data from a vehicle, a Popoto board, a real
navigation unit or Sensor Bridge hardware (`docs/STATUS.md`, first paragraph).

| Gap | Detail | Source |
| --- | --- | --- |
| Real data | simulator calibration is checked only on its own simulated logs; the live chain has never run on a recording from the sea | `docs/SIM_CALIBRATION.md`; `sonar-sul` |
| Hardware | no DGX Spark or Jetson measurement (`docs/JETSON.md` is untested); no RoCE receiver; the board forwarder is not written | STATUS milestones 11, 13; item 36 |
| Sensor Bridge | loopback through the emulator only. The idle stop ran on a GPU only in runs set up for it; the pace table of D-173 was not repeated on L4 | D-163, D-173; `sonar-3wd`, `sonar-e2u` |
| Only on a GPU job | the Holoscan C++ operators cannot be built in the CPU container; the CUDA halves of 17 Python files and `DetectorDevice` skip on a CPU | D-158; section 2 |
| Only on a checkout | `test_tools_help.py` (65 cases) never runs in the GPU job | section 1 |
| Never in the suite | the tests on PICMUS, UACR, Watermark and Smart Ocean data, the `arlpy` checks, and the ultrasound reference run only where someone has the data or the build | section 1 |
| The six builds' self-tests | `tools/build_from_spec.py --self-test` is not part of either suite; run it by hand after a chain change | `docs/HANDOFF.md` section 6 |
| Receive fuzz | the cross-engine fuzz of D-161 and D-177 (1000 to 4000 streams per script) was run by reviewers and is not in `tests/`. Known miscounts remain | D-161, D-177; `sonar-thu` |
| Detector false alarms | the three-row static count stopped at 40 000 pings; no scenario has non-Rayleigh reverberation at the detector; more than three rows take the three-row constants unmeasured | D-168; STATUS item 17; `sonar-4xn` |
| LFM range-Doppler correction | the live chain is held by its tests only, not measured on the gallery scenes | D-181 |
| Leading edge | blanking and the burst filter judging by the centroid is untested (at most one cell) | D-160; `sonar-9hy` |
| Bounce grouping | not measured on gallery captures with `multipath` on | D-175; `sonar-ome` |
| The grown fit limit | false movers under it are not measured: no long-range clutter scene exists | D-159; `sonar-kal` |
| Timestamp-less source | the tracker has no clock by default; a real mover may fall out of its gate | D-165, D-175; `sonar-y7k` |
| Track association at long range | among a dozen arrivals in one gate it is decided by small likelihood differences; no test pins which arrival a track takes | D-181; `sonar-tlc` |
| Lever arm | zero offsets are tested bit for bit and one offset end to end; the surveyed values do not exist; the array's velocity is not corrected | D-167; `sonar-0ca` |
| Gallery classes | `flank_patrol`'s classes are not asserted in `test_documented_full_size_classes`; the full-size classification evaluation was not rerun at this tree | `sonar-n18`, `sonar-mgt`; D-170 |
| Offline captures | `classify_rules_eval` captures lack per-row ping times, so D-165's rule is not tested on them | `sonar-18n` |
| Motion estimation | the acceptance bounds are ours (the paper reports none); lambda 2 on few scenes; along-track surge not implemented; the Ceres solve is CPU only | D-164; `sonar-0bc`, `sonar-7k2` |
| Propagation | 3-D propagation is quantified against Bellhop3D, not modelled; APL-UW TR 9407's boundary models are unread and must stay so | D-149; STATUS milestone 12 |
| Performance | FLOP/s from hardware counters could not be measured on the jobs | STATUS milestone 13 |
| The beads hook | the worktree rule of D-177 has no test in `tests/` | D-177 |
