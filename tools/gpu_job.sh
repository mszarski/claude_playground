#!/usr/bin/env bash
# Build with the CUDA backend and run every test on a GPU machine.
# Written for a clean nvidia/cuda:*-devel-ubuntu24.04 container (for example a Hugging
# Face Job); on a provisioned host just run tools/test.sh.
#
# Usage: tools/gpu_job.sh            from a checkout
#        GPUSONAR_TARBALL_REPO=<user>/<dataset> tools/gpu_job.sh    fetch the source first
#        GPUSONAR_ORACLES=1 tools/gpu_job.sh    also build Bellhop and KRAKEN for the tests that need them
#        GPUSONAR_CERES=1 tools/gpu_job.sh      also build Ceres, so motion estimation is built and tested
#        GPUSONAR_HOLOSCAN_CPP=1 tools/gpu_job.sh  also install the Holoscan C++ package and build the C++ operators
#        GPUSONAR_NVUS=1 tools/gpu_job.sh       also build the ultrasound reference's native module (no SDK)
#        GPUSONAR_HSB=1 tools/gpu_job.sh        also build Holoscan Sensor Bridge 2.7.0 (emulator, Linux receiver, Python;
#                                               implies GPUSONAR_HOLOSCAN_CPP=1) for tests/python/test_hsb.py
#        GPUSONAR_JOB=gallery tools/gpu_job.sh  run tools/gallery.py (set GPUSONAR_ORACLES=1 and GPUSONAR_CERES=1 too)
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
apt-get update -qq
apt-get install -y -qq cmake ninja-build g++ git python3-dev python3-pip python3-venv \
    libhdf5-dev libeigen3-dev libgtest-dev libgmock-dev libomp-dev > /dev/null
python3 -m venv /opt/venv
. /opt/venv/bin/activate
pip install -q numpy scipy h5py pyyaml pytest pybind11 hdf5plugin huggingface_hub rasterio pyproj matplotlib
# Holoscan SDK 4.x (CUDA 12 build) for the operator tests; they skip if this fails.
pip install -q holoscan-cu12==4.6.0 || echo "holoscan not installed"
ulimit -s 32768 || true
if [ -n "${GPUSONAR_TARBALL_REPO:-}" ]; then
  hf download "$GPUSONAR_TARBALL_REPO" gpusonar.tar.gz --type dataset --local-dir /work > /dev/null
  mkdir -p /work/src && tar -xzf /work/gpusonar.tar.gz -C /work/src
  cd /work/src
fi
if [ "${GPUSONAR_CERES:-0}" = "1" ]; then
  # Ceres Solver without SuiteSparse (BSD; SuiteSparse is GPL), for motion estimation.
  git clone -q -b 2.2.0 --depth 1 https://github.com/ceres-solver/ceres-solver /opt/ceres-src
  cmake -S /opt/ceres-src -B /opt/ceres-build -G Ninja -DSUITESPARSE=OFF -DCXSPARSE=OFF -DLAPACK=OFF \
        -DGFLAGS=OFF -DMINIGLOG=ON -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF -DUSE_CUDA=OFF \
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/opt/ceres-nosuitesparse > /dev/null
  cmake --build /opt/ceres-build > /dev/null && cmake --install /opt/ceres-build > /dev/null
fi
holoscan_flag=""
[ "${GPUSONAR_HSB:-0}" = "1" ] && GPUSONAR_HOLOSCAN_CPP=1
if [ "${GPUSONAR_HOLOSCAN_CPP:-0}" = "1" ]; then
  # Headers and CMake files of the same SDK version as the Python wheel (the wheel has neither).
  apt-get install -y -qq --no-install-recommends holoscan=4.6.0-1 'holoscan-cuda-12=4.6.0*' > /dev/null
  pip install -q "cmake>=3.30.4"            # Holoscan 4.6's CMake files ask for it; Ubuntu 24.04 has 3.28
  hash -r
  holoscan_flag="-DSONAR_BUILD_HOLOSCAN=ON"
fi
nvcc --version | tail -1
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DSONAR_ENABLE_CUDA=ON \
      -DPython3_EXECUTABLE="$(which python3)" $holoscan_flag ${SONAR_CMAKE_ARGS:-}
cmake --build build
if [ "${GPUSONAR_ORACLES:-0}" = "1" ]; then
  # External GPL-3 propagation programs, run as separate processes (docs/OPEN_SOURCE.md).
  apt-get install -y -qq gfortran make > /dev/null
  eval "$(tools/build_oracles.sh /opt/at | tail -1)"
fi
if [ "${GPUSONAR_NVUS:-0}" = "1" ]; then
  # The NVIDIA ultrasound reference's native module at the pinned commit with the
  # long-window patch, without its SDK: for the nvus_scanner source's tests.
  eval "$(tools/build_nvus_reference.sh /opt/nvus | tail -1)"
fi
if [ "${GPUSONAR_HSB:-0}" = "1" ]; then
  # NVIDIA Holoscan Sensor Bridge, built natively (its own flow is a Docker build): the board
  # emulator and the Linux receiver operator for the loopback test of sources/hsb.py.
  # Its emulator sends through a raw socket; where the container does not permit one, build it with
  # the opt-in patch that lets it use an ordinary UDP socket, and switch that on.
  if ! python3 -c "import socket; socket.socket(socket.AF_INET, socket.SOCK_RAW, socket.IPPROTO_RAW)" 2> /dev/null; then
    export HSB_UDP_PATCH=1 HSB_EMULATOR_UDP_SOCKET=1
    echo "raw sockets not permitted: Sensor Bridge emulator patched to use a UDP socket"
  fi
  hsb_env=$(tools/build_hsb.sh /opt/hsb | tail -1) && eval "$hsb_env" || echo "Sensor Bridge not built: its tests will skip"
fi
if [ "${GPUSONAR_JOB:-tests}" = "benchmark" ]; then
  PYTHONPATH=python python3 tools/benchmark.py --backend cuda --scale "${GPUSONAR_SCALE:-2}" --pixel "${GPUSONAR_PIXEL:-0.05}" \
      --modes "${GPUSONAR_MODES:-sas,fls,flank}"
  exit 0
fi
if [ "${GPUSONAR_JOB:-tests}" = "gallery" ]; then
  # Every mode's realistic scenario, simulated and imaged; figures and numbers are uploaded
  # to GPUSONAR_UPLOAD_REPO (a Hugging Face dataset) when it is set.
  PYTHONPATH=python python3 tools/gallery.py --backend cuda --out /work/gallery ${GPUSONAR_GALLERY_ARGS:-}
  if [ -n "${GPUSONAR_UPLOAD_REPO:-}" ]; then
    hf upload "$GPUSONAR_UPLOAD_REPO" /work/gallery gallery --type dataset > /dev/null
  fi
  exit 0
fi
echo "=== ctest (C++ suites) ==="
ctest --test-dir build --output-on-failure -E pytest | tail -5
echo "=== pytest (all, with skip reasons) ==="
PYTHONPATH=python${PYTHONPATH:+:$PYTHONPATH} python3 -m pytest -q -rs tests/python "$@"
