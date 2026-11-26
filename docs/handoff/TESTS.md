# Tests handoff

How to run the tests, what each one protects, which decisions they pin, how to recognise a
failure that is not one, and what is not tested. Written at the tree of `933972b`, decisions to
D-181; brought to D-201 and the opt-in 3-D tier on 2026-11-26 (counts per file collected then with
`pytest --collect-only`), and to D-225 (the merge `b88c458`) the same day: the counts in section 1, the
seventeen new test files in section 2, the decisions D-204 to D-224 in section 3, two pitfalls in
section 4 (the new files' "Tests" are `def test_` functions counted with grep, not collected). Companions: [SOFTWARE.md](SOFTWARE.md) (architecture and settings),
[README.md](README.md) (the set).

**What is not known first.** No per-file run time is recorded anywhere in the repository; the
"class" column in section 2 is a judgement from what each test does. The log of the last full
GPU run (job `6b07ea74d91f621b930a343d`, D-225) was not read for this file: its 102 skips are
derived in section 1 from the 93 of D-181's job and the decisions since, not read from its log, and
its four expected failures are presumably the four strict xfails of D-217 and D-218 (not checked).
Of the earlier job of D-201 (`6b078a26d91f621b93096bc7`), which Popoto-rate loopback case was its one
expected failure is not in its log.

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
| `GPUSONAR_BELLHOP3D`, `GPUSONAR_KRAKEN_DIR` | Bellhop3D (the seamount comparison, the `eigenray3d` tier's tests; without the variable, `bellhop3d.exe` beside `GPUSONAR_BELLHOP` is used); rays against normal modes | skip |
| `GPUSONAR_SLOW_ORACLES=1` | the `eigenray3d` tier's acceptance against Bellhop3D's converged 3-D field (`test_seamount_3d.py::test_eigenray3d_tier_over_the_seamount`, about 8 minutes) and four 3-D scenario checks (`test_3d_scenarios.py`, about 10 minutes) | skip, always, in the suite; the skip names no oracle variable, so `GPUSONAR_REQUIRE_ORACLES=1` does not turn it into a failure |
| PyRAM 1.3.0 installed (`pyram`) | rays against a parabolic equation (`test_ram_crosscheck.py`) | skip |
| `GPUSONAR_REQUIRE_ORACLES=1` | an oracle test that would skip fails instead (`tests/python/oracle_gate.py`) | the run ends with a line counting the oracle skips |
| a Ceres without SuiteSparse at `SONAR_CERES_DIR` | motion estimation is built (`core.HAVE_MOTION`) | `test_motion.py`, `test_motion_delays.py` and three other tests skip |
| a CUDA build and a device | the host-against-device tests | they skip; two tests run only on a host-only build |
| the Holoscan SDK (Python wheel) | `test_holoscan.py`, parts of `test_hsb.py` | skip |
| `-DSONAR_BUILD_HOLOSCAN=ON` | `test_holoscan_cpp.py` (needs `build/bin/sonar_live_holoscan`) | skip |
| Sensor Bridge built (`tools/build_hsb.sh`) and a CUDA device | the loopback tests of `test_hsb.py` | skip |
| `GPUSONAR_HSB_STUDY=1` | two measurements in `test_hsb.py` that are not tests (`test_loopback_loss_study`; `test_popoto_rate_loopback_study`, with `GPUSONAR_HSB_STUDY_RUNS`, `_ROWS`, `_ENGINES`, `_CONFIGS`, `_BUDGET_S`, `_RECEIVER_CORE`) | skip, always, in the suite |
| `GPUSONAR_HSB_TIMING=1` | a measurement in `test_hsb.py` (`test_frame_path_timing_study`: the receive path's cost a frame, D-190; `GPUSONAR_HSB_TIMING_ROWS`, `_ENGINES`) | skip, always, in the suite |
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

- About 45 to 50 minutes on the `l4x1` flavour (49 min 55 s for D-201's job). L4 is the reference: every timing and every Sensor
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
| Full GPU suite, NVIDIA L4, the merged tree of D-204 to D-224 (`b88c458`) | **2763 passed, 102 skipped, 4 xfailed**, none failed (job `6b07ea74d91f621b930a343d`; Bellhop, Bellhop3D, KRAKEN, PyRAM, Ceres, the Holoscan C++ operators, Sensor Bridge, `GPUSONAR_REQUIRE_ORACLES=1`, `GPUSONAR_SLOW_ORACLES=1`). Before it, the tree of D-202: 2301 passed, 102 skipped (job `6b079ab0d91f621b93098915`). The batch's earlier suites and their six failures are in D-225 | D-225, D-202 |
| Full GPU suite, NVIDIA L4, tree of D-201 | **2278 passed, 102 skipped, 1 xfailed**, none failed (job `6b078a26d91f621b93096bc7`; the xfail is one Popoto-rate loopback case, its reason not in the log). The 3-D work after D-201 is covered by a later job. Before it, D-200 over the tree of D-196: 2232 passed, 101 skipped (job `6b077ea0d91f621b930950a5`). Before it, D-195's tree: 2183 passed, 101 skipped (job `6b076d06d91f621b93092f13`). Before it, D-192's tree: 2183 passed, 101 skipped (job `6b07608ed91f621b930918b0`). Before it, D-191's tree: 2180 passed, 101 skipped (job `6b07440eb6a90bbe997e6027`). Before it, D-190's tree: 2180 passed, 100 skipped (job `6b06d64fd91f621b93079be0`). Before it, D-189's tree: 2171 passed, 98 skipped (job `6b06bbda1460ffe1d5ea75b5`). Before it, D-188's tree: 2159 passed, 97 skipped (job `6b0699f4d0582cfbbefe0373`). Before it, D-187's tree: 2155 passed, 97 skipped, 1 xfailed (job `6b06728ad0582cfbbefdb5eb`; the xfail was one Popoto-rate loopback case with a frame damaged at the start, accepted since D-188). Before it, D-186's tree: 2150 passed, 97 skipped, 4 xfailed (job `6b06553a1460ffe1d5e9aa10`). Before it, D-183's tree: 2102 passed, 95 skipped (job `6b060a4b1460ffe1d5e93111`). Before it, D-181's tree: 2078 passed, 93 skipped (job `6b055ee4c97b8ed4fcd799db`), which the skip groups below describe | each decision's entry in `docs/DECISIONS.md`, D-183 to D-201 |
| Previous | 2065 passed, 93 skipped (D-180); 1925 passed, 93 skipped (D-175 to D-178); 1875 passed, 92 skipped (D-171 to D-173) | the entries named |
| GoogleTest at this tree | 193 `TEST` cases in 15 binaries (counted as lines starting `TEST(` in `tests/cpp/*.cpp`, 2026-11-26 at D-225; 175 at D-203, 172 at D-181) | section 2 |
| pytest at this tree | 1775 test functions in 145 files (counted as `def test_` lines in `tests/python/test_*.py`, 2026-11-26 at D-225; the same count gives 1526 in 128 at `a64a26f`, as stated before; 1375 in 114 at D-181); parametrisation makes them 2869 outcomes on D-225's L4 job (2763 + 102 + 4) | counted |

No full CPU run of the tree is recorded since D-157 (`605a4e1`); `docs/STATUS.md` ("State of
verification") records the CPU and GPU runs up to D-157 and the decisions record the GPU runs after.

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

**From 93 to 102 (D-201's job), derived from the decisions, not read from a log:** +2 at D-183 (the
`--help` tests of D-182's two tools, `tools/handoff_hardware.py` and `tools/software_diagrams.py`,
by the count's rise; not itemised in D-183); +2 at D-186 (the opt-in `test_popoto_rate_loopback_study`
and one D-186 says was not looked up); +1 at D-189 (`tools/hsb_host_check.py --help`); +2 at D-190
(the opt-in `test_frame_path_timing_study` and `tools/hsb_frame_timing.py --help`); +1 at D-191
(`tools/live_chain_timing.py --help`); +1 between D-200's job (tree of D-196, 101) and D-201's
(likely D-197's `tools/eigenray_tl_check.py --help`; not checked). The 3-D work adds one more skip by
design (`GPUSONAR_SLOW_ORACLES`). D-225's job also reports 102 skips, with `GPUSONAR_SLOW_ORACLES=1` set and
the batch's new tests (the tests that ask for the C++ engine skip only on a build without it, D-219); not
itemised from its log.

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
| `test_live_cpp.py` | 60 (D-225 count) | the C++ library against the Python stages, both chains' reports, the program, every option's round trip, unknown keys and words | `build/bin/sonar_live` for the program tests | minutes | **parity**, **golden** |
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
| `test_calibration_sync.py` | 15 | array calibration; the sync stage recovers injected offsets and drifts | | minutes | |
| `test_beamform_cpp.py` | 13 | C++ far-field and MVDR beamformers against NumPy; the chain with them | `sonar_live` for one test | minutes | **parity** |
| `test_flank.py` | 11 | far-field and backprojection agree; beam width; detection range against the sonar equation | | minutes | |
| `test_hardcoded.py` | 22 | another sonar or board is a configuration change: derived sizes, capacities, the program | `sonar_live` for some | minutes | |
| `test_dict_keys.py` | 9 | every options dictionary of the live library refuses a key it does not read (D-177) | | fast | **parity** |
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
| `test_flank_stream.py` | 8 (13 collected) | the capture stream from an emulated board through UDP back to records; the 24-bit decoder and the window-only decode (D-190) | skips if the loopback dropped samples | minutes | |
| `test_board_streams.py` | 21 (36 collected) | per-board streams merged into the whole-array stream (D-184): the block's board field (whole-array blocks unchanged), the whole-array receivers refusing a board block, merged pings equal to the whole-array pings, loss, restarts, a silent board, a late start, refusals by name, the options, the program and `--emulate --per-board` over UDP, the relay tool; sonar-bwu: `build/bin/sonar_live --listen` and `sonar_live_hsb.py receive` (frames of one board each, `serve --per-board`'s framing) merging as the relay in front does, clean, a board's block lost (drop, zero), reordered and repeated, boards on their own rates | the program for one test; loopback tests skip if data was dropped | minutes | **parity** |
| `test_board_clock.py` | 13 (20 collected) | a board clock's rate against true time (D-185): the stamp's drift an hour in and the pose it moves, `clock_ppm` in both engines and the program, boards at their own rates aligned by them, the estimate from arrival times and its opt-in discipline | the program for one test | fast | **parity** |
| `test_hsb_receive.py` | 25 (99 collected) | what the receiver and assembler count when frames go missing, repeat or arrive out of order, with no network (D-161, D-177) | | fast | **parity** |
| `test_hsb_host_check.py` | 10 (12 collected) | `tools/hsb_host_check.py` on injected readings (a container like the ones used, a host set up as Sensor Bridge's guide asks), its stream arithmetic from `hardware_builds.json`, its exit statuses, and one run on the machine itself (sonar-02c) | | fast | |
| `test_hsb.py` | 31 (43 collected) | framing; the receive application's stop; the real-time loopback through Sensor Bridge's emulator and receiver (D-153, D-162, D-163, D-173); a Popoto build's trial at 32, 64 and 96 channels and 102.4 kHz, framed and through the loopback in both engines, start-up loss and damaged frames (D-186 to D-188, D-190); one stream per board through the loopback (D-200); the opt-in studies | Holoscan SDK; Sensor Bridge and a CUDA device for the loopbacks; OpenMP for one | minutes | |
| `test_holoscan.py` | 5 | stages as Holoscan operators, fed by the simulator and by replay | Holoscan SDK | minutes | |
| `test_holoscan_cpp.py` | 4 (7 collected) | the C++ Holoscan application against the standalone program; an operator's error ends it with status 1 and the reason, not an abort (D-186); its source operator merging per-board streams as the relay in front does, pings and `--listen` (sonar-bwu) | built with `-DSONAR_BUILD_HOLOSCAN=ON` | minutes | **parity** |
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
| `test_sim_cpp.py` | 31 (D-225 count) | C++ paths and reverberation against the NumPy versions (D-61) | the C++ engine; CUDA for some | minutes |
| `test_sampled_waveform.py`, `test_sampled_engine.py` | 17, 15 | sampled transmit pulses against TR 0702; the C++ kernel against the NumPy engine (D-74, D-75, D-81) | Bellhop for one each; CUDA for some | minutes |
| `test_batched.py` | 28 | the batched device formulations against the reference host code (D-122) | CUDA for the device half | minutes |
| `test_parallel_normal.py` | 8 | threaded normal deviates are bit for bit `rng.standard_normal` | | fast |
| `test_scene_objects.py`, `test_scene_targets.py`, `test_scene_sediment.py` | 19, 17, 5 | extended objects, highlight targets against arXiv:1805.06945, sediment scatterers (D-41, D-64, D-65) | Ceres and CUDA for one each | minutes |
| `test_moving_ownship.py` | 7 | a moving ownship over a seabed fixed in the world | | minutes |
| `test_propagation.py`, `test_propagation_defaults.py` | 11, 5 | providers against Bellhop; a bottom needs a description, arrivals kept by energy (D-104, D-105) | Bellhop for the first | minutes |
| `test_long_range.py` | 2 | long-range scenarios at the eigenray tier | Bellhop (whole file) | minutes |
| `test_ssp_range.py` | 14 | sound speed that changes with range in the Bellhop provider (D-110) | Bellhop for some | minutes |
| `test_bellhop_nx2d_agreement.py` | 6 | Bellhop 2-D against Bellhop3D's N × 2-D and 3-D modes, beam counts, arrivals against the loss run over a piecewise-linear bed (D-197; characterisations) | each test its oracle: Bellhop, Bellhop3D, KRAKEN, PyRAM (about 45 s in all) | minutes |
| `test_image_planes.py`, `test_multipath_scenes.py` | 17, 6 (D-225 counts; every test on both engines since D-205) | image sources off a tilted bed and walls as any plane against closed forms, both engines; the four multipath scenes' ghost geometry (D-201) | | fast to minutes |
| `test_gallaudet_boundaries.py` | 6 (10 collected) | the bottom and surface models against Gallaudet 2001's printed values at 68 kHz (D-193; characterisations: the bound is what we get plus a small margin) | | fast |
| `test_kraken_crosscheck.py`, `test_ram_crosscheck.py`, `test_seamount_3d.py` | 2, 2, 4 | rays against normal modes, a parabolic equation and Bellhop3D (D-149); the last file's acceptance of the `eigenray3d` tier (8 minutes) runs only with `GPUSONAR_SLOW_ORACLES=1` | Bellhop with KRAKEN, PyRAM, Bellhop3D | minutes |
| `test_cross_slope.py`, `test_bellhop3d_reader.py`, `test_bellhop3d_provider.py`, `test_eigenray3d_scenario.py` | 5, 4 (5 collected), 6, 4 | the N × 2-D cross-slope warning; Bellhop3D's arrivals read and merged; the opt-in `eigenray3d` provider; one bed for a scenario and `flank_seamount_3d` on both engines (sonar-rcq P1 to P6, `docs/tasks/PROPAGATION_3D.md`) | Bellhop3D for some (27 s and 47 s with it) | fast to minutes |
| `test_3d_scenarios.py` | 22 | the five 3-D scenes and their N × 2-D twins of `sonar-d7sw`: the bed `bathymetry.ridge` and `profile_along`; every file loads and builds its provider; each twin differs only in the tier and its bed is the 3-D bed along the target's bearing; what the cross-slope warning says of each bed; with the oracles, each scene's headline level (3-D minus Bellhop3D's N × 2-D) and arrival direction (`docs/SCENARIOS.md`, "Flank array over 3-D relief") | Bellhop3D and Bellhop for 5 (the level control, about 40 s); 4 of those also `GPUSONAR_SLOW_ORACLES=1` (2 to 3 minutes each, about 10 minutes in all) | fast; slow with the switch |
| `test_reverberation.py`, `test_reverb_statistics.py` | 10, 27 (42 collected) | reverberation level within 2 dB of the sonar equation; K statistics (D-37, D-70), the shape from a density per cell (D-196) | | minutes |
| `test_reverb_bistatic.py`, `test_reverb_motion.py`, `test_reverb_relief.py`, `test_reverb_specular.py` | 23, 10, 16, 6 | bistatic, moving scatterers, relief, near-specular removal (D-69, D-72, D-73, D-123) | CUDA for one; Bellhop for some | minutes |
| `test_bottom.py`, `test_bottom_models.py`, `test_sediment_table.py`, `test_sediment_literature.py`, `test_sediment_fit.py` | 9, 22, 17, 9, 25 (D-225 counts) | the bottom model, its regimes, the named sediments against open literature, the fit (D-63, D-66, D-67, D-132) | Bellhop for one | fast |
| `test_surface.py`, `test_surface_models.py`, `test_bubble_layer.py` | 5, 13, 9 | surface backscatter, bubble loss (D-38, D-68, D-87) | Bellhop for one; `arlpy` for one | fast |
| `test_timespread.py`, `test_coherent_phase.py` | 34, 10 | time, angle and frequency spreading after TR 0702; the coherent arrival's wander (D-62, D-76, D-108) | CUDA for one | minutes |
| `test_noise.py`, `test_baseband_noise.py`, `test_noise_statistics.py`, `test_noise_fit.py` | 5, 7, 10, 27 | noise spectra, array coherence, K statistics, a fit to a recording (D-103, D-124) | `arlpy` for one; UACR noise for two | minutes |
| `test_direct_overload.py`, `test_source_level.py` | 19, 4 | the projector's direct sound and receiver overload (D-125); source levels are RMS (D-102) | | minutes |
| `test_oneway_sources.py` | 16 | one-way sources against the passive sonar equation (D-78) | `sonar_live` for one | minutes |
| `test_basin.py` | 20 | a basin's late reverberation (D-106) | Bellhop for one | minutes |
| `test_channel_replay.py`, `test_channel_sounding.py` | 23, 20 | measured-channel replay and sounding estimators on synthetic channels | UACR, Watermark, Smart Ocean data for seven | minutes |
| `test_sst_validation.py` | 24 | two worked examples of APL-UW TR 0702 end to end (D-78, D-144) | | long |
| `test_simcal.py` | 7 | simulator calibration finds hidden settings in a simulated log (D-56) | | minutes |
| `test_literature_models.py`, `test_medium_models.py`, `test_lloyd.py` | 11, 13, 21 | formulas against printed values; medium helpers; the surface-image factor (D-126) | `arlpy` for three; Bellhop for one | fast |
| `test_jackson_2007_bottom.py`, `test_surface_papers_1997.py`, `test_grain_size_relations.py`, `test_surface_dahl_1997.py` | 14, 12, 10, 11 | the book's bottom equalities and the surface papers' equalities; the grain-size relations (the default since D-209); `extinction: dahl_1997` (the default since D-216) (D-206) | | fast |
| `test_measure.py` | 6 | measurement estimators and error bars for real data (D-207) | | fast |
| `test_sediment_penetration.py`, `test_sediment_roughness.py` | 16, 16 | the sediment tier's `penetration: fluid` and `rough` on both engines; the two buried scenes name fluid and medium sand (D-212, D-213) | the C++ engine for the parity half | fast to minutes |
| `test_elastic_bottom.py`, `test_elastic_live.py` | 13, 9 | elastic rock in the simulator, Bellhop and the live chain's models (D-214, D-215) | Bellhop for one | fast |
| `test_forward_scatter_dahl.py`, `test_asiaex_dahl_2004.py`, `test_kirchhoff_go.py` | 7, 5, 16 | the large-scale slope law, Dahl 2004's ASIAEX scenario, the opt-in exact Kirchhoff tail; four strict xfails between them (D-217, D-218) | | minutes |
| `test_sim_engine_select.py`, `test_sim_engine_record.py`, `test_sim_engine_arg.py` | 15, 15, 11 | engine selection never silent; the engine note in every record; the engine argument reaching every provider (D-219, D-220) | CUDA for one | fast to minutes |
| `test_reverb_budget.py` | 7 | the reverberation term of the link budget and the coverage picture against simulated data and the closed form (D-221) | Bellhop (about 7 minutes) | long |
| `test_mourad_jackson_1989.py` | 12 | the 1989 model against the paper's figures; our volume term against its corrected eq. (39) (D-223) | | fast |

### Configuration, builds, tools, whole scenes

| File | Tests | Protects | Needs | Class | Marks |
| --- | --- | --- | --- | --- | --- |
| `test_environment.py` | 8 | one environment per site; contradictions (D-100, D-101) | | fast | |
| `test_emulate.py` | 8 (15 collected) | `sonar_live.py --emulate` lays the receive window out at the scenario's sound speed; the program reports an error on its UDP path (D-183) | `sonar_live` for the program case | minutes | |
| `test_build_diagrams_layout.py` | 2 (8 collected) | the build diagrams' heading and row labels stay on the page (D-183) | `matplotlib` | fast | |
| `test_yaml_numbers.py` | 3 | every tracked YAML's numbers load as numbers with plain PyYAML (D-180) | | fast | |
| `test_design.py` | 7 | every owner requirement is met by the config files; `docs/DESIGN.md` is up to date (D-44) | `matplotlib` | minutes | reads committed files |
| `test_buildspec.py` | 21 | build specs validate; the generator's output is what is committed; a different build runs through both engines (D-127) | `matplotlib`; Bellhop for one | long | reads committed files |
| `test_realistic_scenes.py` | 25 (28 collected) | every mode's scenario simulates and images end to end (`tools/gallery.py --quick`); `test_documented_full_size_classes` reads `docs/figures/gallery.json`; `flank_patrol`'s small vehicle is not tracked (D-198) | `matplotlib`; Bellhop for five; Ceres for one | long | **golden** (the gallery) |
| `test_classify_rules_eval.py` | 15 | the scoring of `tools/classify_rules_eval.py` on a made-up case, no simulation | | fast | |
| `test_gallery_echo_position.py`, `test_gallery_subgrid_bearing.py` | 4, 7 | the gallery's echo placement and leading edge (D-172, D-178, D-181); its opt-in sub-grid bearing (D-176) | | fast | |
| `test_gpu_benchmark_all.py` | 4 | the benchmark tool's table and JSON on whatever backend the build has | | minutes | |
| `test_study_resume.py` | 7 | the resumable studies reuse stored results and recompute with `--fresh` | `matplotlib` | fast | |
| `test_tools_help.py` | 5 (83 collected) | every tool answers `--help` and changes nothing in the tree | a git checkout | minutes | |
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

## 3. Tests that pin a decision (D-155 to D-225)

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
| D-183 (sonar-8zj) | the receive window laid out at `source.sound_speed`; `sonar_live`'s UDP path reports an error and exits 1 | `test_emulate.py` (the whole file; five fail without the fix); `test_build_diagrams_layout.py` |
| D-184 | per-board streams merged into the whole-array stream (sonar-6f9) | `test_board_streams.py` (the whole file) |
| D-185 | a board clock's rate: configured, estimated, opt-in discipline (sonar-ubg) | `test_board_clock.py` (the whole file) |
| D-186 (sonar-ogg, sonar-x8v) | `UdpSourceOp` joins its threads in its destructor too; the Sensor Bridge loopback at a Popoto build's channel count and rate | `test_holoscan_cpp.py::test_an_operator_that_throws_ends_the_application_with_its_reason_not_an_abort` (passed before the fix too: Holoscan 4.6 stops the operators after an exception; no fail-before was found); `test_hsb.py::test_a_popoto_trial_is_framed_at_its_builds_rate_and_its_pings_come_back` (1, 2, 3 rows), `::test_loopback_at_a_popoto_builds_channel_count_and_rate_gives_the_direct_chains_reports` (C++ at 1, 2, 3 rows, Python at 1 and 2; **an expected failure (xfail) with the numbers when frames are lost**, which in the containers so far is nearly every run), `::test_popoto_rate_loopback_study` (`GPUSONAR_HSB_STUDY=1`) |
| D-187 (sonar-bep, sonar-us7) | the Popoto-rate loss is at the start of the run; the receiver's queue defaults to 4 s of the stream | `test_hsb_receive.py::test_the_receiver_queue_defaults_to_four_seconds_of_the_stream`; `test_hsb.py::test_the_loss_location_places_missing_frames_in_the_pings_cycle`; the Popoto loopback test passes when every missing frame is in the first 0.25 s of stream, none damaged |
| D-225 (the merge) | three sediment-fit tests compared at a stated tolerance, not bit for bit (weakened: FIT_REL 2e-6, FLAT_REL 2e-5; w2 on its limit no longer asserted) | `test_sediment_fit.py::test_the_old_default_named_is_the_same_as_before_the_opt_in_options_and_a10`, `::test_tool_options_for_error_bars_bounds_and_holds`, `::test_a_parameter_on_a_bound_is_in_the_result_and_called_not_measured` |
| D-224 | five named site rows; the soft rows' range [0.0008, 0.005]; the default model against each paper's figure (differences recorded) | `test_sediment_table.py` (`test_the_jackson_briggs_rows_are_table_v_as_printed`, `test_the_williams_jackson_panama_city_row_is_table_i`, `test_each_site_row_through_the_default_model_against_the_papers_figure` and the rest) |
| D-223 | Wagner shadowing in `composite` and `combined`; the Mourad-Jackson (1989) model; the volume term pinned to the corrected eq. (39); seven-digit constants | `test_mourad_jackson_1989.py`, `test_bottom_models.py` (+6), `test_grain_size_relations.py`, `test_reverberation.py` (+2), `test_reverb_bistatic.py` (+1); `test_gallaudet_boundaries.py` (the relations' check widened 6e-5 -> 8e-5, the miss pinned) |
| D-222 | four papers read; no code | none |
| D-221 (sonar-w2w3) | the reverberation term of the link budget and the coverage picture | `test_reverb_budget.py` (the whole file; Bellhop; about 7 minutes); `test_live_cpp.py::test_coverage_reverberation_is_equal_in_both_engines_and_off_is_as_before`; `CoverageReverb.*` |
| D-220 | every record names its engines; the engine argument reaches every provider | `test_sim_engine_record.py`, `test_sim_engine_arg.py` (the whole files) |
| D-219 | an `{axis, position}` wall takes axis 0 or 1; an explicit `cpp` without the C++ simulator raises; a stale module is named | `test_sim_cpp.py` (`test_each_wall_axis_runs_in_cpp_and_matches_numpy`, `test_a_wall_axis_that_is_not_0_or_1_is_refused_in_both_engines`, `test_an_axis_wall_needs_its_axis_and_a_finite_position`, `test_image_source_field_points_must_be_s_by_3`); `Paths.AnAxisWallOtherThanXOrYIsRefused`; `test_sim_engine_select.py` |
| D-218 | the opt-in exact Kirchhoff tail; Dahl 1996 eq. (13) and Table I kd* (two strict xfails) | `test_kirchhoff_go.py` |
| D-217 | the large-scale slope by default; `asiaex_dahl_2004` | `test_forward_scatter_dahl.py`, `test_asiaex_dahl_2004.py` (one strict xfail each); `test_timespread.py`; `test_sst_validation.py::test_asiaex_tails_decay_at_the_rule_of_thumb` (the tool's fit window changed); `test_channel_replay.py::test_model_surface_arrival_statistics_match_the_closed_forms` (moved to `slope_law: cox_munk_total`) |
| D-216 | `dahl_1997` by default, the attenuated facet term, the geometric-mean bistatic rule | `test_surface_dahl_1997.py` (hex pins at rtol 1e-14: weakened), `test_literature_models.py`, `test_gallaudet_boundaries.py`, `test_surface_models.py`; `test_sst_validation.py::test_pursuit_target_is_below_the_reverberation_but_separated_in_doppler` (weakened: the rise over the mean reverberation, not one draw) |
| D-215 | rock elastic in the live chain's models, both engines | `test_elastic_live.py`; `test_live_cpp.py::test_rock_bottom_is_elastic_and_equal_in_both_engines`; `Rock.*` |
| D-214 | rock elastic in the simulator and Bellhop | `test_elastic_bottom.py`; `Paths.ElasticRockCoefficient`; elastic cases in `test_sim_cpp.py` |
| D-213 | `penetration: rough`, both engines | `test_sediment_roughness.py`; `test_sim_cpp.py` (`test_rough_*`, one CUDA-gated); `Paths.SedimentRough*` |
| D-212 (sonar-zpm6) | `penetration: fluid`; the two buried scenes use it | `test_sediment_penetration.py`; `test_sim_cpp.py`; `test_environment.py::test_every_shipped_scenario_states_its_site_once` (narrowed for `fidelity: sediment`) |
| D-210, D-211 | the fit's small-slope default; the row's prior by default; sigma2 up to 0.005 | `test_sediment_fit.py` (`test_a10_*`, `test_default_prior_bounds_of_a_row_are_its_grain_sizes_scattering_priors`, `test_tool_applies_the_rows_prior_by_default_and_the_off_switch_gives_the_fit_of_before`, `test_the_sigma2_prior_reaches_the_printed_arafura_fit` and the rest) |
| D-209 | the relations by default; the soft rows' information keys; the Gallaudet site row | `test_grain_size_relations.py::test_the_default_is_the_relations`; `test_sediment_table.py`; three checks naming `table_linear` |
| D-207 (sonar-mxsa) | measurement estimators and the fit's opt-in error bars and priors | `test_measure.py`; `test_sediment_fit.py` |
| D-206 | the book's and the surface papers' equalities; the opt-in relations and `dahl_1997` | `test_jackson_2007_bottom.py`, `test_surface_papers_1997.py`, `test_grain_size_relations.py`, `test_surface_dahl_1997.py` |
| D-205 (sonar-8qn9, sonar-k9v7) | an `{axis, position}` wall path arrives mirrored in the wall; the general construction in C++ | `test_image_planes.py::test_an_axis_wall_gives_the_rays_of_the_same_wall_as_a_plane` and every test of the file on both engines; `Paths.AWallPathArrivesMirroredInTheWall`, `PlaneImages.*` |
| D-204 (sonar-d7sw) | the five 3-D scenes and their twins | `test_3d_scenarios.py`; the pinned list in `test_eigenray3d_scenario.py` |
| D-202, 3-D work, 2026-11-26 (sonar-rcq P1 to P6) | the cross-slope warning; Bellhop3D's arrivals read and merged; the `eigenray3d` provider; one top-level bed; the tier against Bellhop3D's converged field | `test_cross_slope.py`, `test_bellhop3d_reader.py`, `test_bellhop3d_provider.py`, `test_eigenray3d_scenario.py` (the whole files); `test_seamount_3d.py::test_eigenray3d_tier_over_the_seamount` (`GPUSONAR_SLOW_ORACLES=1`) |
| D-201 (sonar-0nsi) | image sources off tilted planes and walls against closed forms; the four multipath scenes' ghost geometry | `test_image_planes.py`, `test_multipath_scenes.py` (the whole files) |
| D-200 (sonar-bwu) | every receiver merges per-board streams as the relay in front does | `test_board_streams.py` (the `--listen` and Sensor Bridge cases); `test_holoscan_cpp.py::test_holoscan_merges_per_board_streams_as_the_relay_in_front_does`; `test_hsb.py::test_loopback_of_one_stream_per_board_gives_the_direct_chains_reports` |
| D-198 (sonar-80b) | `flank_patrol`'s small vehicle is not tracked (below the scene's reverberation) | `test_realistic_scenes.py::test_flank_patrol_detects_and_tracks` |
| D-199 | corrects D-197's account of the fjord beds; nothing changed | none (no code changed) |
| D-197 (sonar-6c8) | Bellhop 2-D and Bellhop3D N x 2-D agree over a level bed; beam counts; arrivals over a piecewise-linear bed (characterisations) | `test_bellhop_nx2d_agreement.py` (the whole file; oracles; first run on a GPU job with D-201's suite, passed) |
| D-196 (sonar-6fc) | the K shape from a density per cell; our shape is the paper's alpha; its worked example | `test_reverb_statistics.py` (`test_our_shape_is_the_papers_alpha`, `test_the_papers_worked_example`, `test_the_shape_follows_the_cell`, `test_density_statistics_in_a_ping` and the rest of the new cases) |
| D-195 (sonar-59u) | the decimate-first correction by board, the samples to the bit | `tests/cpp/test_stage2.cpp`: `Decimate.BoardsCorrectedTogetherGiveTheChannelByChannelSamplesToTheBit` |
| D-194 (sonar-nyn) | the TR 0702 matrix names only tests that exist and does not deny features we have | `test_sst_validation.py::test_matrix_names_only_tests_that_exist_and_does_not_deny_features_we_have` |
| D-193 (sonar-5wa) | the bottom and surface models against Gallaudet 2001's printed values at 68 kHz (characterisations) | `test_gallaudet_boundaries.py` (the whole file) |
| D-192 (sonar-fy6) | the resampler's weights from three trig calls, both engines; rows together against row by row to the bit, against the formula within a stated bound | `test_stage2.cpp`: `Sync.ResampleWeightsFromThreeTrigCallsAreTheFormulasTapByTap`; `test_calibration_sync.py` (the weights, whole-sample positions, the re-pinned resampler test); `test_live_cpp.py::test_both_engines_resample_with_the_same_weights` |
| D-191 (sonar-drc) | the C++ engine resamples a board's rows together, the samples to the bit | `tests/cpp/test_stage2.cpp`: `Sync.BoardsResampledTogetherGiveTheRowByRowSamplesToTheBit` |
| D-190 (sonar-h1f) | the frame path decodes only the rows a window takes; a board's rows resampled together; outputs to the bit | `test_flank_stream.py` (the decoder, the assembled samples, the held block); `test_calibration_sync.py` (the resampler against row by row); the (3, python) Popoto loopback case |
| D-189 (sonar-02c) | the host check for the Sensor Bridge measurement; the quota rule in one place | `test_hsb_host_check.py` (the whole file); `test_hsb.py::test_usable_cpus_honours_a_cgroup_quota` |
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
| The C++-engine tests of `test_image_planes.py`, `test_sim_cpp.py`, `test_sediment_penetration.py`, `test_sediment_roughness.py`, `test_elastic_bottom.py` or `test_elastic_live.py` fail with `StaleModuleError: … has no <binding>: it is older than the sources` (or a pybind11 `TypeError` on a keyword such as `roughness=`, or "unknown option coverage_bottom_shear_speed") | a module built before the current sources, e.g. the main checkout's, which lacks `sim_plane_image_eigenrays` (D-219) | rebuild before reading these as regressions |
| Every test that asks for the `cpp` engine skips, its reason naming the C++ simulator | a module built without the C++ simulator: since D-219 an explicit `cpp` raises `EngineUnavailable` (before, it ran NumPy against NumPy and passed), and `tests/python/conftest.py` turns that into a skip | build with `SONAR_BUILD_PYTHON=ON`; the summary line's skips say how many |
| A sediment-fit test differs by about 1e-8 relative, or w2 stops just off its 1e-9 limit | another platform's NumPy, SciPy or libm (the GPU job's image); within `least_squares`' tolerance | since D-225 the tests hold fitted values within 2e-6 relative; a larger difference is real |
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
| Sensor Bridge | loopback through the emulator only. The idle stop ran on a GPU only in runs set up for it; the pace table of D-173 was not repeated on L4. Never on a host with Sensor Bridge's own socket setup (the containers refuse it; `tools/hsb_host_check.py` and its test check a host, they do not measure the receiver). Per-board frames only at `flank_32`'s rate. Nothing enforces the 0.25 s lead before the first transmit | D-163, D-173, D-187, D-189, D-200; `sonar-3wd`, `sonar-e2u`, `sonar-02c`, `sonar-a75` |
| Per-board streams | only emulated boards on loopback; the Holoscan listen path was not compared with the program's `--listen`; the clock-rate keys are ignored, not refused, for a whole-array stream in the Holoscan application and `receive` | D-184, D-185, D-200; `sonar-9a2`, `sonar-pnyv` |
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
| Propagation | 3-D propagation in the shipped scenarios is N × 2-D at most; the opt-in `eigenray3d` tier is held against Bellhop3D only by the 8-minute test that runs with `GPUSONAR_SLOW_ORACLES=1`, and has no reverberation over relief; APL-UW TR 9407's boundary models are unread and must stay so | D-149, D-197; `docs/tasks/PROPAGATION_3D.md`; STATUS milestone 12 |
| Boundary models against data | one site, one wind (Gallaudet 2001, 68 kHz); characterisations, not bounds from the paper | D-193; `sonar-64sm` |
| Link budget | the reverberation term (D-221) is held against simulated data on `flank_patrol`, two 10 km scenes and `sst_sediment_reverb`; no shipped site states scattering, so no design budget is tested with it; the target margins are held at 5 dB a target | D-221 |
| New opt-in physics | `penetration: rough` has no imaging test (SAS or 3-D); `kirchhoff_go` and the 1989 model are checked against figure readings; the rock shear values are not measured | D-213, D-218, D-223, D-214 |
| Multipath in the chains | the FLS chain's ghost tracks measured on one seed, 4 pings, half densities; no rejection tested | D-201; `sonar-izys` |
| Performance | FLOP/s from hardware counters could not be measured on the jobs | STATUS milestone 13 |
| The beads hook | the worktree rule of D-177 has no test in `tests/` | D-177 |
