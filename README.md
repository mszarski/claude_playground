# gpusonar

One GPU software suite that turns raw per-channel sonar data into images and detections
for forward-looking sonar (FLS), synthetic aperture sonar (SAS) and a long-range flank
array. Every mode runs through the same record format, the same time-domain
backprojection kernel, the same config system and the same simulator.

**Continuing the work?** Start at [`docs/handoff/README.md`](docs/handoff/README.md): the
entry point of the handoff set (state, risks, hardware, software, tests, outputs, and a map
of every other document).

**State of the project.** Everything here is software, simulation and analysis: no sonar
described in it is built, and nothing has run on a vehicle, on the Popoto boards, on a DGX
Spark, a Jetson or Sensor Bridge hardware. Milestones 1 to 10 of 13 are done on the CPU and
the CUDA backend has been run and benchmarked on rented GPUs. The long-range design is
honest about where it stands: one long row at 197 dB is 1.1 dB short of detection at 12 km
on the chain's own detector, depth at range is limited by what is known of the sound-speed
profile, and two stacked rows at 3 km in shallow water read the surface or the sea bed
(the depth sigma now says so). The full GPU suite ran on the tree of decision D-200
(2232 passed, 101 skipped, none failed, on an NVIDIA L4); the whole CPU suite has not been
rerun on the present tree (last whole run on record: the tree of D-157, 157 of 157 ctest
entries). [`docs/STATUS.md`](docs/STATUS.md) says
what works, how each part was checked, and lists what is open or known to be wrong;
[`docs/RESULTS.md`](docs/RESULTS.md) has the numbers.

## What is in it

- **One record format** (HDF5, schema 1.0.0) between every source and every stage, with
  taps after any stage that are themselves sources.
- **Imaging**: time-domain backprojection for forward-looking sonar, stripmap and rail
  SAS, and 3D near-field imaging of buried objects; conditioning, pulse compression,
  motion estimation from the data (Ceres), ray culling, sessions that keep samples on the
  device.
- **A long-range chain for a towed or hull-mounted line array** (`libs/live`, in C++ and
  mirrored in Python): board synchronisation, beamforming (backprojection, far-field,
  MVDR, sparse Bayesian bearings), a detector with a stated false-alarm budget, tracking,
  classification, a report per ping. For two or three stacked rows: elevation and depth.
  Depth from the timing of the surface echo. Rays through the sound-speed profile the
  vehicle holds. A coverage picture (where it can and cannot detect). Passive bearings
  and tonal analysis (Goertzel lines, DEMON). The same program runs on a simulation, a
  recording, an emulated board over UDP, and as Holoscan operators.
- **A simulator** with four propagation tiers (free field, image source, sediment
  interface, eigenrays from Bellhop run as a separate program), reverberation from bottom,
  surface and volume, time, angle and frequency spreading of bounce paths, targets and
  specular shapes, one-way sources, noise with a spectrum and array coherence, the
  projector's direct sound and receiver overload, a basin's late reverberation, and sound
  speed that changes with range. Checked against APL-UW TR 0702 section by section and
  against three sets of measured channel soundings.
- **Designs traced to config files**: 120, 170 and 330 kHz forward-looking sonars, a
  MINSAS-class SAS, and a 4 to 6 kHz long-range array on Popoto boards as a staged build
  (a one-row prototype, a static frame, the vehicle's face with one to three rows), with
  link budgets, a depth planner, projector requirements, parts lists and checklists
  generated from each build's spec.

The repository also holds the older `trajectory_classifier` Python package, which is
unrelated to the sonar suite.

## Build and test

Requirements (Ubuntu 24.04 names): `cmake ninja-build g++ libhdf5-dev libeigen3-dev
libgtest-dev libomp-dev`, Python 3.10+ with `pybind11 numpy scipy h5py pyyaml pytest`.
A CUDA toolkit is optional.

Motion estimation needs Ceres Solver built **without SuiteSparse** (SuiteSparse is GPL and
the Ubuntu Ceres package links it):

```bash
git clone -b 2.2.0 https://github.com/ceres-solver/ceres-solver && cd ceres-solver
cmake -S . -B build -G Ninja -DSUITESPARSE=OFF -DCXSPARSE=OFF -DLAPACK=OFF -DGFLAGS=OFF \
      -DMINIGLOG=ON -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF -DUSE_CUDA=OFF \
      -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DCMAKE_INSTALL_PREFIX=/opt/ceres-nosuitesparse
cmake --build build && sudo cmake --install build
```

Without it the suite still builds; motion estimation is left out and its tests skip.

```bash
tools/test.sh            # configure, build, run GoogleTest and pytest (the whole suite)
```

The whole Python suite takes a long time on a small machine and needs several GB of
memory. To run parts of it:

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DSONAR_ENABLE_CUDA=OFF && cmake --build build -j3
ctest --test-dir build -E pytest --output-on-failure        # the C++ tests only
PYTHONPATH=python python3 -m pytest tests/python/test_detector.py -q      # one Python file
GPUSONAR_SIM_ENGINE=numpy PYTHONPATH=python python3 -m pytest tests/python/test_sim.py -q   # the NumPy reference engine
```

Tests that need something external skip without it. The switches:

| Variable | What it enables |
| --- | --- |
| `GPUSONAR_BELLHOP=/path/to/bellhop.exe` | the `eigenray` tier: every `flank_*`, `popoto_*`, `fjord_*` and `ssp_*` scenario that names Bellhop, and their tests (`tools/build_oracles.sh` builds it) |
| `GPUSONAR_KRAKEN_DIR` | the rays-against-normal-modes cross-check |
| `GPUSONAR_REQUIRE_ORACLES=1` | an oracle test (Bellhop, Bellhop3D, KRAKEN, PyRAM) that would skip fails instead; without it the run ends with a line counting the oracle skips (`tests/python/oracle_gate.py`; `GPUSONAR_ORACLES=1 tools/gpu_job.sh` sets it) |
| `GPUSONAR_PICMUS_DIR` | the PICMUS ultrasound benchmark (`tools/fetch_picmus.py`) |
| `GPUSONAR_UACR_DIR`, `GPUSONAR_WATERMARK_DIR`, `GPUSONAR_SMARTOCEAN` | the measured channel data sets (`tools/fetch_channels.py`; not in the repository, see `docs/OPEN_SOURCE.md` 3.13) |
| `GPUSONAR_SIM_ENGINE=numpy` or `cpp` | which simulator engine the tests use |

On a GPU machine `tools/gpu_job.sh` builds with CUDA and runs the suite and the
benchmarks (`GPUSONAR_JOB=benchall` for the table of every stage).

- The CUDA backend is built when a CUDA compiler is found. Force it with
  `SONAR_CMAKE_ARGS=-DSONAR_ENABLE_CUDA=ON` or turn it off with `=OFF`.
- With a toolkit whose nvcc needs an older host compiler, set `CUDAHOSTCXX`
  (for example `CUDAHOSTCXX=g++-12` with CUDA 12.0).
- From Python: `PYTHONPATH=python python3 -c "import gpusonar"`.

## Everything on a CPU

No GPU is needed for anything here. The CUDA backend is an optional second implementation
of the same kernels and gives the same results.

```bash
SONAR_CMAKE_ARGS=-DSONAR_ENABLE_CUDA=OFF tools/test.sh      # build and run every test
tools/build_oracles.sh ~/at                                 # Bellhop and KRAKEN (needs gfortran), prints two exports
tools/build_nvus_reference.sh ~/nvus                        # the ultrasound reference's native module, patched
tools/design_report.py                                      # docs/DESIGN.md and its diagrams
tools/scenario_diagrams.py                                  # docs/scenarios/<name>.png: geometry of every scenario
tools/popoto_range_study.py --range 12000                   # frequency and source level for a range
tools/gallery.py --backend host                             # every realistic scenario, full size: figure and animation
tools/sonar_live.py configs/deploy/popoto_vehicle.yaml --scenario popoto_far_mover --out run/   # the vehicle chain on a simulation
tools/sonar_live.py configs/deploy/popoto_vehicle.yaml --export-template template.h5          # template + command for the C++ program
build/bin/sonar_live --replay run/recording.h5 doa=sbl                                         # the same chain in C++, sparse Bayesian bearings
build/bin/sonar_live --passive run/recording.h5 --band 4000 6000                               # passive bearing-time record
build/bin/sonar_live --udp 5700 --template template.h5 --listen --band 4000 6000               # the same from the untriggered board stream
build/bin/sonar_live_holoscan --replay run/recording.h5 backend=cuda                           # as C++ Holoscan operators (-DSONAR_BUILD_HOLOSCAN=ON)
tools/live_benchmark.py run/recording.h5                                                       # per-ping time: host/CUDA, map/SBL
tools/inspect_tap.py run/taps/ping0000 --stage prepare --plot ping.png                          # look at any stage's data
tools/calibrate_sim.py fls --log dive.h5 --scenario fls_120k_transit                            # fit the simulator to a real log
tools/gallery.py --backend host --quick                     # the same, thinned, into build/gallery_quick/ (not docs/figures)
```

The long-range array's design and study tools. Most write a committed file under
`docs/design/` or `docs/figures/`, named in `docs/RESULTS.md`; not all: `tonal_range_study.py`
only prints (`--json` prints JSON), `coverage_map.py` writes only with `--json PATH` or
`--figure PATH`, `gpu_benchmark_all.py` prints its table unless given `--out` / `--json`,
`nav_adapter.py` converts a stream, and `build_from_spec.py` writes sonar and deployment files
under `configs/`:

```bash
tools/build_from_spec.py configs/builds/popoto_face_3x32.yaml --validate   # a build spec: sonar and deployment files, self-tests, checklists
tools/build_from_spec.py --check-shipped                    # the generated files in the repository match their specs
tools/parts_list.py                                         # docs/design/popoto_parts.md
tools/popoto_depth_planner.py --shipped                     # own depth against detection margin
tools/projector_requirements.py; tools/projector_candidates.py   # what the projector must do; listed candidates against it
tools/direct_sound_study.py                                 # the projector's direct sound at every hydrophone
tools/popoto_12km_study.py                                  # the 12 km design point through the chain (needs Bellhop)
tools/coverage_map.py --build popoto_face_1x32 --environment shelf_100m_sand --own-depth 50   # where a build can detect (prints; --json, --figure to write)
tools/multipath_depth_study.py; tools/refraction_depth_study.py --bellhop   # depth from the surface echo; depth against profile knowledge
tools/ssp_range_study.py; tools/fjord_rehearsal.py          # water that changes with range; a trial in fjord-type water
tools/nav_adapter.py --help                                 # a vendor's navigation stream to the chain's datagrams
tools/tonal_range_study.py                                  # passive detection range of a tonal (model figures; prints only)
tools/gpu_benchmark_all.py --backend host                   # every stage with a CUDA backend, timed (the CUDA column on a GPU; prints, --out / --json to write)
```

Simulator validation and fitting: `tools/sst_validation.py` (against TR 0702),
`tools/channel_validation.py`, `channel_watermark.py`, `channel_smartocean.py` (against
measured channels), `tools/basin_reverberation_fit.py`, `tools/fit_noise.py`,
`tools/fit_sediment.py`, `tools/boundary_models_study.py`, `tools/timespread_study.py`,
`tools/coherent_phase_study.py`, `tools/decay_time_sensitivity.py`.

The host backend uses every core (OpenMP). Per-scenario times on four cores are in
`docs/SCENARIOS.md`; all 26 gallery scenarios take about an hour in total, the 120 kHz transit 77 s and the
flank patrol eleven minutes (72 before the kernel rework). Those times were measured
before the latest changes of levels, noise and defaults. The gallery's numbers are from a
later full-size run with the CUDA backend on an NVIDIA L4 (commit `4530dc4`), which took
465 s for the 26 scenarios.

Without Bellhop the scenarios at the `eigenray` tier (most of `flank_*`, `popoto_*`,
`fjord_*`, `ssp_*`) and their tests are skipped; everything else runs.

## Layout

| Path | Contents |
| --- | --- |
| `libs/core` | ping record, geometry (ASASIN Eq. 2), directivity, slab allocator, host/CUDA `parallel_for` |
| `libs/io` | HDF5 container reader and writer, validator |
| `libs/sim` | simulator kernels: echo synthesis, path pairing, reverberation (also batched for the device) |
| `libs/dsp` | baseband conversion, pulse compression, calibration, TVG, whitening, DRC, FFT, host and device executors |
| `libs/recon` | the backprojection kernel (host and CUDA from one source) |
| `libs/motion` | Ceres solve of ASASIN Eq. 8 (sway and heave) |
| `libs/detect` | CFAR kernel (host and CUDA); the detector's statistics (tails and thresholds of sums of exponentials) |
| `libs/live` | the long-range chain: board stream, navigation, synchronisation, beamformers, detector, elevation, multipath depth, ray code, coverage, sound-speed ladder, tonal analysis, tracking, report |
| `libs/holoscan_ops` | Holoscan operators in C++ around the live library (`-DSONAR_BUILD_HOLOSCAN=ON`) |
| `apps/` | `sonar_live` (the chain as a standalone program) and `sonar_live_holoscan` |
| `third_party/` | vendored pocketfft (BSD-3); patches applied to separately built third-party trees |
| `python/gpusonar` | bindings and the Python layer: sources, configs, the environment schema, the live chain's Python engine, flank beamforming, detection, tracking, elevation, ray tracing, coverage, link budget and build specs, depth planning, calibration, sync, tonal analysis |
| `python/gpusonar/sim` | the simulator: scenes, targets, propagation providers (including Bellhop as an external process), reverberation, time spreading, boundary models, noise, one-way sources, receiver overload, basin reverberation, channel replay |
| `configs/sonars`, `configs/scenarios` | named sonars and the 76 scenarios (`docs/SCENARIOS.md` lists every one) |
| `configs/environments`, `configs/sediments.yaml` | named sites and sediments |
| `configs/builds`, `configs/deploy`, `configs/nav`, `configs/chains` | build specs of the staged Popoto array, deployment files (generated and hand-written), navigation adapter mappings, processing chains |
| `tools/` | study, design, validation and benchmark programs (listed above) |
| `tests/cpp`, `tests/python` | GoogleTest and pytest suites |
| `docs/` | documents (below); `docs/design/` and `docs/figures/` hold generated tables, JSON and figures; `docs/benchmarks/` the GPU timings |

## Realistic scenes for every mode

`docs/SCENARIOS.md` shows 26 extended scenes across the modes, each simulated and imaged
end to end, each with a geometry diagram, a result figure and an animation:

- forward-looking: pool, harbour, seafloor objects; the 120, 170 and 330 kHz designs under
  way against boats, a buoy and a pinnacle; a cluttered seabed with a pipeline; a rough sea
  with waves; boat traffic;
- synthetic aperture: a wreck and a mined seabed with unknown motion; the MINSAS-class
  survey over a wreck, cable and calibration cube; a boulder field; rippled sand with a
  rock outcrop; the rail; buried objects in 3D;
- long range: the flank patrol; the Popoto array against a ship at 6 km and at 12 km, and
  in four other oceans (winter duct over sand, summer thermocline over mud, shallow gravel,
  deep water over silt).

Reproduce them with `tools/gallery.py` (needs `matplotlib`; `--quick` thins the scenes).

## Documents

Start with the first three.

- [`docs/STATUS.md`](docs/STATUS.md): what works, how it was checked, what is open or known wrong, the owner's open questions.
- [`docs/RESULTS.md`](docs/RESULTS.md): measured numbers against theory, references and benchmarks, each with the command or file that produces it.
- [`docs/DECISIONS.md`](docs/DECISIONS.md): every choice made where the paper, the brief or a reference is silent, with its reason (A-01 onward, D-01 to D-136).
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md): how the pieces fit, what runs on a GPU, and how to capture, inspect and re-inject data at every stage.
- [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md): the long-range array on a vehicle: one program, one YAML file, what the vehicle must provide, the staged build, refraction in operation, the detector, a checklist for a trial in a fjord or harbour, what is unconfirmed.
- [`docs/REFERENCE_DESIGN.md`](docs/REFERENCE_DESIGN.md): how to go from a build spec to hardware, step by step, with every field of the spec.
- [`docs/DESIGN.md`](docs/DESIGN.md): the sonar designs, generated from the config files: requirements and where each is met, arrays and beams, timing, link budgets, the 12 km design point, diagrams.
- [`docs/CONFIGURATION.md`](docs/CONFIGURATION.md): every key of the sonar, scenario, environment and deployment files; units and conventions.
- [`docs/SCENARIOS.md`](docs/SCENARIOS.md): the realistic scenes of every mode with figures and numbers, and a list of every scenario file.
- [`docs/PHYSICS_NOTES.md`](docs/PHYSICS_NOTES.md): SST (APL-UW TR 0702) section by section, and the derivations behind the simulator's and the chain's models (Secs. 12 to 28).
- [`docs/REFERENCES.md`](docs/REFERENCES.md): every source behind a formula, table or constant, how it was reached and how far it is confirmed.
- [`docs/OPEN_SOURCE.md`](docs/OPEN_SOURCE.md): open-source projects and data sets checked or used, licences, roles.
- [`docs/ORG_ARL_SURVEY.md`](docs/ORG_ARL_SURVEY.md): what the org-arl repositories (arlpy and its Julia successors) hold, item by item.
- [`docs/HARDCODED_AUDIT.md`](docs/HARDCODED_AUDIT.md): an audit of hard-coded assumptions and what was done about each.
- [`docs/SIM_CALIBRATION.md`](docs/SIM_CALIBRATION.md): fitting the simulator to a real FLS log or a SAS image.
- [`docs/RECORD_FORMAT.md`](docs/RECORD_FORMAT.md): the ping record container and the WAV sidecar.
- [`docs/PAPER_NOTES.md`](docs/PAPER_NOTES.md): the ASASIN paper, equation by equation, with its gaps.
- [`docs/HOLOSCAN_NOTES.md`](docs/HOLOSCAN_NOTES.md): how the NVIDIA ultrasound reference acquires data.
- [`docs/JETSON.md`](docs/JETSON.md): build notes for Jetson (untested).

## Simulate and image a forward-looking sonar ping

```python
import numpy as np
from gpusonar import recon
from gpusonar.sim import run_scenario

result = run_scenario("point_grid_l12_5d")          # two pings: the two mux halves
fan = recon.fan_image(result.records,
                      ranges=np.linspace(0.8, 2.9, 1500),
                      bearings_rad=np.radians(np.linspace(-20, 20, 801)))
print(abs(fan.image).shape)                         # range x bearing
```

Scenarios live in `configs/scenarios`, sonars in `configs/sonars`. A scene is a list of
objects (`points`, `polar_grid`, `wire`, `seafloor`, `plate`, `sphere`, `box`, `rocks`,
`pool`); see `seafloor_objects_fls.yaml` for one with shadows. A new sonar is a new
YAML file; no signal-processing code changes.

The PICMUS benchmark: `tools/fetch_picmus.py <dir>`, then run the tests with
`GPUSONAR_PICMUS_DIR=<dir>`.

## Simulate and image a stripmap SAS run

```python
from gpusonar import recon
from gpusonar.sim import run_scenario

result = run_scenario("stripmap_points")                 # 90 pings past three targets
grid = recon.stripmap_grid(result.records, ground_range=(36.0, 47.0), pixel=(0.02, 0.04))
image = recon.sas_image(result.records, grid.points)     # along-track x ground range, complex
```

On a GPU machine pass `backend=gpusonar.core.Backend.Cuda`: conditioning (cuFFT) and
backprojection then run on the device. To image the same pings more than once (tiles,
depth layers, several GPUs, repeated focus trials), build a session so the samples are
uploaded once:

```python
from gpusonar import core, motion

recon.prepare(result.records, backend=core.Backend.Cuda)           # baseband + pulse compression
session = core.BackprojectSession(result.records)
image = session.backproject(grid.points.reshape(-1, 3), interp="linear", fov_factor=1.13,
                            backend=core.Backend.Cuda).reshape(grid.points.shape[:2])
```

With motion that navigation does not know, estimate it from the data first
(`estimate = motion.estimate_motion(records); motion.apply_motion(records, estimate)`);
the `wreck_sas` scenario and `docs/figures/wreck_sas.png` show the whole chain.

## Flank array: range x bearing map, detections, predicted range

```python
import numpy as np
from gpusonar import detect, flank
from gpusonar.sim import run_scenario
from gpusonar.sonar import SonarModel
from gpusonar.sonar_equation import SonarEquation

ping = run_scenario("point_grid_flank_32").records[0]
bearings = np.radians(np.arange(-60, 60.5, 0.5))
beams = flank.farfield_beamform(ping, bearings).valid()     # or flank.mvdr_beamform
power = abs(beams.data) ** 2
hits = detect.cluster(detect.cfar(power, pfa=1e-6), power, beams.ranges, bearings)

eq = SonarEquation.from_sonar(SonarModel.load("flank_32"), noise_spectrum_db=50.0)
print(eq.detection_range(threshold_db=13.0))
```

For several boards with their own clocks, run `gpusonar.sync.synchronise(record, reference)`
on the raw record first. For an array with unknown channel errors,
`gpusonar.calibrate.estimate_channel_errors` on pings of known sources.

### Long range: multipath, reverberation and noise

The `flank_target_{1,5,10}km` and `popoto_trial_*` scenarios run at the `eigenray` tier:
Bellhop computes the arrivals as a separate program (GPL-3, never linked; set
`GPUSONAR_BELLHOP=/path/to/bellhop.exe`). A scenario can add, under `reverberation:`,
boundary reverberation by the scattering-function method, and under
`receiver.ambient_noise:` noise with a spectrum and array coherence:

```yaml
propagation:
  environment: {depth: 100.0, sound_speed: 1500.0, bottom_speed: 1600.0, bottom_density: 1.8, bottom_attenuation: 0.5}   # older spellings, still accepted; see CONFIGURATION.md, "Environment"
  frequency: 8000.0
  max_range: 10500
  max_arrivals: 80          # too few drops echo energy: see docs/RESULTS.md
  wind_speed: 0.0           # m/s; above 0 adds rough-surface reflection loss per bounce
reverberation:
  - {boundary_z: 100.0, mu_db: -27.0}                               # seabed, Lambert
  - {boundary_z: 0.0, model: chapman_harris, wind_speed: 5.0}       # sea surface
receiver:
  ambient_noise: {level_db: 45.0}
```

The 10 km case has thousands of two-way paths; pass `backend=gpusonar.core.Backend.Cuda` to
`run_scenario` on a GPU machine. The tests that compare the eigenray tier with normal
modes also need KRAKEN (`GPUSONAR_KRAKEN_DIR`, the root of an Acoustics Toolbox build).
`GPUSONAR_ORACLES=1 tools/gpu_job.sh` builds both in a clean container.

### Simulator options after SST (APL-UW TR 0702)

The simulator also has, all opt-in and all listed key by key in `docs/CONFIGURATION.md`:

- targets as sets of highlights with strength against aspect, and smooth spheres,
  cylinders, plates and discs that return specular glints (`type: target`);
- seabeds and bottom reverberation from a rough fluid sediment model, with two named
  sediments (`sediment: medium_sand`), Kirchhoff, small-slope and composite regimes;
- time spreading of boundary-bounce paths, with angle and frequency spreading
  (`time_spread:`);
- reverberation with separate, moving transmitter and receiver, K statistics, moving
  scatterers and a seabed with relief (`geometry: bistatic`, `statistics:`,
  `scatterer_motion:`, `bathymetry:`); volume-scattering layers;
- sampled transmit pulses: LFM, HFM, stepped FM, Barker and m-sequence codes, or samples
  from a file (`transmit.waveform: {type: sampled, ...}`).

Added since: one-way sources and passive captures (`sources:`, `pings.transmit: false`);
one environment per site (`environment:` naming a file in `configs/environments/`);
source levels stated as RMS (`transmit.waveform.source_level_db_rms`); the projector's
direct sound and a receiver that overloads (`transmit.direct_path`, `receive.overload`);
noise fitted to a recording; a basin's late reverberation (`basin_reverberation:`); sound
speed that changes with range (`environment.profiles_by_range`).

`tools/sst_validation.py` checks the models against TR 0702 section by section
(`docs/figures/sst_validation.md`), and `tools/channel_validation.py` against measured
channels. What to know before relying on them (`docs/STATUS.md`): the boundary models are
open-literature substitutes for a handbook (APL-UW TR 9407) that was not read; no measured
data set settles the decay time of a rough bounce's tail; the bubble-loss law has no
stated range of validity; water that changes with range is rings about the sonar; and
many shipped shallow-water scenarios state far fewer bounces than their energy would
justify.

## The long-range array from a build spec

```bash
tools/build_from_spec.py configs/builds/popoto_static_3x32.yaml --validate
tools/sonar_live.py configs/deploy/popoto_static_3row.yaml --scenario popoto_static_trial_3row --out run/ --engine cpp
```

A build spec (`configs/builds/*.yaml`) states the array, the boards, the projector, the
platform and the operation; the sonar file, the deployment file, a parts list, diagrams
and checklists are generated from it, and seven self-tests run the generated chain on
simulated pings. `docs/REFERENCE_DESIGN.md` walks through it; `docs/DEPLOYMENT.md` says
what each report field means. The link budget's threshold is the chain's own detector's
(`docs/DECISIONS.md` D-117, D-134), so margins and the chain agree to about half a dB.

## Run a whole chain from one YAML file

```bash
tools/sonar_run.py configs/chains/stripmap_points.yaml
```

The config names the source, the steps and the outputs (GeoTIFF, a JSON-lines processing
log with per-stage timings, optional debug products per stage). `python/gpusonar/pipeline.py`
lists the stages. GeoTIFF output needs `pip install rasterio pyproj`.

## Live pipelines with Holoscan

`python/gpusonar/holoscan_ops.py` wraps the stages as Holoscan operators
(`PingSourceOp`, `ConditionOp`, `FanImageOp`, `DetectOp`, `DisplayPrepOp`) and
`FanPipeline` composes them. Needs the Holoscan SDK (`pip install holoscan-cu12` on a
CUDA 12 machine); see `tests/python/test_holoscan.py` for pipelines fed by the simulator
and by file replay. The C++ operators (`libs/holoscan_ops`, `build/bin/sonar_live_holoscan`)
wrap the long-range chain. Both ran on rented GPUs with Holoscan SDK 4.6.0, fed by the
simulator, by replay and by an emulated board over UDP; no capture hardware has been
connected (`docs/STATUS.md`).

## Reading recorded data

```python
from gpusonar.sources import make_source

# Our container
for ping in make_source({"type": "container", "path": "run.h5"}):
    print(ping.metadata.sequence, ping.samples.array.shape)

# Popoto or generic multichannel .wav, described by a YAML sidecar
for ping in make_source({"type": "wav", "sidecar": "trial.yaml"}):
    ...
```

## Connecting real data sources

Every source ends in the same thing: a `PingRecord` per transmit. What exists for each:

| Source | Path into the suite | State |
| --- | --- | --- |
| Our container files | `make_source({"type": "container", ...})` | works, tested |
| Popoto modem or any multichannel recorder | record to `.wav`, describe it in a YAML sidecar (geometry, transmit, ping times), `make_source({"type": "wav", ...})` | works on simulated recordings; no real Popoto file has been read |
| Several recorder boards with their own clocks | as above, then `gpusonar.sync.synchronise(record, reference)` | tested on simulated clocks; reference-signal wiring per the plan (`popoto_4x7`) |
| PICMUS plane-wave ultrasound sets | `tools/fetch_picmus.py`, `make_source({"type": "picmus", ...})` | works, tested |
| Holoscan ultrasound rig (TI TX7332 / AFE5832LP board, L12-5 or P4-2v) | `make_source({"type": "nvus_scanner", ...})`: the NVIDIA reference's native runtime programs one plane wave per multiplexer half and acquires; each observation becomes two records | runs against the reference's simulated hardware (built from the pinned commit); the reference's compiler limits the window to 0.88 m; **never run on hardware or with the SDK build** |
| Flank array capture board (32 channels, 24-bit, 64 kSPS) | blocks of samples with a small header, assembled into pings: `make_source({"type": "flank_udp", ...})`, or `PingAssembler.feed(block)` behind any transport | tested against an emulated board over UDP, including lost blocks; the board does not exist yet, so the block layout (`sources/flank_stream.py`) is provisional; not yet run through the Sensor Bridge receiver |

Ultrasound rig, step by step (all on the Holoscan host; nothing here changes FPGA code):

1. Check out `nvidia-holoscan/holoscan-ultrasound-imaging` at commit `24d405e` with the
   Sensor Bridge and Holoscan versions it pins (2.7.0 and 4.4.0). The current Sensor
   Bridge (3.0, Holoscan 5) does not build with it; see `docs/HOLOSCAN_NOTES.md`.
2. Work through the first-run checklist in `docs/HOLOSCAN_NOTES.md` section 11, receive
   only and with the high-voltage supplies off, before any transmit.
3. Build the reference (`./setup` on the rig; for a dry run without hardware,
   `cmake -S . -B build -DNVUS_BUILD_PYTHON=ON` gives the native module alone) and make its
   `nvus` package importable, or set `GPUSONAR_NVUS_PYTHON` to the directory holding it.
4. Acquire through the reference's own sequence API:

```python
from gpusonar.sources import make_source

source = make_source({"type": "nvus_scanner", "sonar": "l12_5d", "repetitions": 100,
                      "hardware": hardware})        # omit `hardware` for the simulated one
for record in source:                                # "odd" half, then "even", per repetition
    ...
```

   The window is the longest the reference's compiler accepts, 0.88 m. A 2 to 3 m window
   needs its slower Python driver or a change to its compiler (`docs/HOLOSCAN_NOTES.md` 9.1).
5. From there it is the same chain as the simulator: `recon.fan_image([odd, even], ...)`
   or the Holoscan operators above.

Values the hardware has to confirm (sampling rate, element order, start-of-capture delay,
gain steps) live in `configs/sonars/*.yaml` under `unconfirmed:`, not in code.
