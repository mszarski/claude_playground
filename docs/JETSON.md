# Building on Jetson (Orin, Thor) and DGX Spark

**Untested.** Nothing in this repository has been built or run on aarch64. These are the
steps the x86_64 GPU build uses (`tools/gpu_job.sh`), adjusted for JetPack; expect to fix
things on the first try and please record what changed.

## Prerequisites

- JetPack with its CUDA toolkit (`nvcc` on PATH, typically `/usr/local/cuda/bin`).
- `sudo apt install cmake ninja-build g++ python3-dev python3-venv libhdf5-dev
  libeigen3-dev libgtest-dev libgmock-dev libomp-dev`
- CMake 3.24 or newer. JetPack 5 (Ubuntu 20.04) ships an older one: install a current
  CMake with `pip install cmake` inside the virtual environment.
- A C++20 host compiler. If the JetPack `nvcc` rejects the system g++ as too new, point it
  at an older one: `export CUDAHOSTCXX=g++-11` (the kernels themselves are C++17).
- Python: `python3 -m venv venv && . venv/bin/activate && pip install numpy scipy h5py
  pyyaml pytest pybind11 hdf5plugin`
- Motion estimation: Ceres built without SuiteSparse, as in the README.

## Build and test

```bash
SONAR_CMAKE_ARGS="-DSONAR_ENABLE_CUDA=ON" tools/test.sh
python3 tools/benchmark.py --backend cuda --scale 1 --out jetson.json
```

Send back the `pytest -rs` summary and the benchmark JSON; `docs/RESULTS.md` has the
x86_64 numbers to compare with.

## What to expect

- The GPU tests compare the CUDA backend with an independent reference. They passed on an
  x86_64 T4 on the first run. Nothing in the kernels depends on the CPU architecture, but
  struct layout and `long double` width differ on aarch64 and are worth watching:
  the precision tests use `numpy.longdouble`, which is 128-bit there.
- Speed: the kernel does part of its work in float64. Jetson GPUs, like the T4, run
  float64 much more slowly than float32, so expect T4-like behaviour: large gains from ray
  culling, modest raw throughput (`RESULTS.md`, "Performance").
- Unified memory: the CUDA driver copies samples host-to-device on every call. On a
  Jetson that copy is avoidable (the memory is shared) and is not yet avoided.
- Capture: per the Sensor Bridge source, AGX Orin receives over Linux sockets only and AGX
  Thor is built without RoCE (`HOLOSCAN_NOTES.md`). The 49 Mbit/s flank
  source (`flank_32`: 32 channels, 24 bits, 64 kHz) fits either; the 38.4 Gbit/s ultrasound capture
  does not. The Popoto builds send 78.6, 157 and 236 Mbit/s for one, two and three rows
  (`docs/handoff/hardware_builds.json`, `rate_total_bit_s`): 1.6 to 4.8 times the flank source and
  far below the ultrasound capture, so the rate alone does not rule either receiver out; no stream
  has been received on a Jetson at any of these rates.
