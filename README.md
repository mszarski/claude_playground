# gpusonar

One GPU software suite that turns raw per-channel sonar data into images and detections
for forward-looking sonar (FLS), synthetic aperture sonar (SAS) and a long-range flank
array. Every mode runs through the same record format, the same time-domain
backprojection kernel, the same config system and the same simulator.

**State: early.** Milestones 1 and 2 of 13 are done; see [`docs/STATUS.md`](docs/STATUS.md)
for what exists, what is next, and what has only been compiled and never run on a GPU.

The repository also holds the older `trajectory_classifier` Python package, which is
unrelated to the sonar suite.

## Build and test

Requirements (Ubuntu 24.04 names): `cmake ninja-build g++ libhdf5-dev libeigen3-dev
libgtest-dev libomp-dev`, Python 3.10+ with `pybind11 numpy scipy h5py pyyaml pytest`.
A CUDA toolkit is optional.

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
| `libs/core` | ping record, geometry (ASASIN Eq. 2), directivity, host/CUDA `parallel_for` |
| `libs/io` | HDF5 container reader and writer, validator |
| `libs/sim` | simulator synthesis kernel |
| `libs/{dsp,recon,motion,detect,holoscan_ops}` | not started |
| `python/gpusonar` | bindings and the Python layer (sources, configs, tools) |
| `configs/` | named sonars and scenarios (YAML) |
| `tests/cpp`, `tests/python` | GoogleTest and pytest suites |
| `docs/` | notes, decisions, record format, status |

## Documents

- [`docs/STATUS.md`](docs/STATUS.md): progress, open questions, deviations.
- [`docs/PAPER_NOTES.md`](docs/PAPER_NOTES.md): the ASASIN paper, equation by equation, with its gaps.
- [`docs/DECISIONS.md`](docs/DECISIONS.md): every choice made where the paper or brief is silent.
- [`docs/HOLOSCAN_NOTES.md`](docs/HOLOSCAN_NOTES.md): how the NVIDIA ultrasound reference acquires data.
- [`docs/OPEN_SOURCE.md`](docs/OPEN_SOURCE.md): open-source projects checked, licences, proposed roles.
- [`docs/RECORD_FORMAT.md`](docs/RECORD_FORMAT.md): the ping record container and the WAV sidecar.

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
