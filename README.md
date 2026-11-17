# gpusonar

One GPU software suite that turns raw per-channel sonar data into images and detections
for forward-looking sonar (FLS), synthetic aperture sonar (SAS) and a long-range flank
array. Every mode runs through the same record format, the same time-domain
backprojection kernel, the same config system and the same simulator.

**State: early.** Milestones 1 to 10 of 13 are done, 12 and 13 are partly done, and the CUDA backend has
been run and benchmarked on rented GPUs (not yet on a DGX Spark or Jetson); see [`docs/STATUS.md`](docs/STATUS.md)
for what exists, what is next, and what has only been compiled and never run on a GPU.

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
tools/test.sh            # configure, build, run GoogleTest and pytest
```

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
tools/inspect_tap.py run/taps/ping0000 --stage prepare --plot ping.png                          # look at any stage's data
tools/calibrate_sim.py fls --log dive.h5 --scenario fls_120k_transit                            # fit the simulator to a real log
tools/gallery.py --backend host --quick                     # the same, thinned
```

The host backend uses every core (OpenMP). Per-scenario times on four cores are in
`docs/SCENARIOS.md`; all 26 gallery scenarios take about an hour in total, the 120 kHz transit 77 s and the
flank patrol eleven minutes (72 before the kernel rework).

Without Bellhop the
long-range scenarios (`flank_*`, `popoto_*`) and their tests are skipped; everything else runs.

## Layout

| Path | Contents |
| --- | --- |
| `libs/core` | ping record, geometry (ASASIN Eq. 2), directivity, slab allocator, host/CUDA `parallel_for` |
| `libs/io` | HDF5 container reader and writer, validator |
| `libs/sim` | simulator synthesis kernel |
| `libs/dsp` | baseband conversion, pulse compression, calibration, TVG, whitening, DRC |
| `libs/recon` | the backprojection kernel (host and CUDA from one source) |
| `libs/motion` | Ceres solve of ASASIN Eq. 8 (sway and heave) |
| `libs/{detect,holoscan_ops}` | not started |
| `third_party/` | vendored pocketfft (BSD-3) |
| `python/gpusonar` | bindings and the Python layer: sources, configs, simulator scenes, SAS and volume imaging, flank beamforming, detection, tracking, calibration, sync, sonar equation |
| `configs/` | named sonars and scenarios (YAML) |
| `tests/cpp`, `tests/python` | GoogleTest and pytest suites |
| `docs/` | notes, decisions, record format, status |

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

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md): how the pieces fit, what runs on a GPU, and how to capture, inspect and re-inject data at every stage.
- [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md): the Popoto array on a vehicle: one program, one YAML file, what the vehicle must provide, what is unconfirmed.
- [`docs/SIM_CALIBRATION.md`](docs/SIM_CALIBRATION.md): fitting the simulator to a real FLS log or a SAS image.
- [`docs/DESIGN.md`](docs/DESIGN.md): the sonar designs, generated from the config files: requirements and where each is met, arrays and beams, timing, link budgets, diagrams.
- [`docs/CONFIGURATION.md`](docs/CONFIGURATION.md): every key of the sonar and scenario files.
- [`docs/SCENARIOS.md`](docs/SCENARIOS.md): the realistic scenes of every mode, with figures, numbers and CPU run times.
- [`docs/STATUS.md`](docs/STATUS.md): progress, open questions, deviations.
- [`docs/RESULTS.md`](docs/RESULTS.md): measured numbers against theory and benchmarks.
- [`docs/PAPER_NOTES.md`](docs/PAPER_NOTES.md): the ASASIN paper, equation by equation, with its gaps.
- [`docs/DECISIONS.md`](docs/DECISIONS.md): every choice made where the paper or brief is silent.
- [`docs/HOLOSCAN_NOTES.md`](docs/HOLOSCAN_NOTES.md): how the NVIDIA ultrasound reference acquires data.
- [`docs/OPEN_SOURCE.md`](docs/OPEN_SOURCE.md): open-source projects checked, licences, proposed roles.
- [`docs/RECORD_FORMAT.md`](docs/RECORD_FORMAT.md): the ping record container and the WAV sidecar.

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
  environment: {depth: 100.0, sound_speed: 1500.0, bottom_speed: 1600.0, bottom_density: 1.8, bottom_attenuation: 0.5}
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
and by file replay. Capture from the ultrasound rig and the flank board is not connected
yet (`docs/STATUS.md`).

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
