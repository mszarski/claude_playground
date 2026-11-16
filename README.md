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

## Documents

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

Scenarios live in `configs/scenarios`, sonars in `configs/sonars`. A new sonar is a new
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

How to connect the Holoscan ultrasound rig and the flank-array capture board will be
documented here when milestone 11 lands.
