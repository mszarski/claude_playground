#!/usr/bin/env bash
# Build with the CUDA backend and run every test on a GPU machine.
# Written for a clean nvidia/cuda:*-devel-ubuntu24.04 container (for example a Hugging
# Face Job); on a provisioned host just run tools/test.sh.
#
# Usage: tools/gpu_job.sh            from a checkout
#        GPUSONAR_TARBALL_REPO=<user>/<dataset> tools/gpu_job.sh    fetch the source first
#        GPUSONAR_ORACLES=1 tools/gpu_job.sh    also build Bellhop and KRAKEN for the tests that need them
#        GPUSONAR_CERES=1 tools/gpu_job.sh      also build Ceres, so motion estimation is built and tested
#        GPUSONAR_NVUS=1 tools/gpu_job.sh       also build the ultrasound reference's native module (no SDK)
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
pip install -q holoscan-cu12 || echo "holoscan not installed"
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
nvcc --version | tail -1
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DSONAR_ENABLE_CUDA=ON \
      -DPython3_EXECUTABLE="$(which python3)"
cmake --build build
if [ "${GPUSONAR_ORACLES:-0}" = "1" ]; then
  # External GPL-3 propagation programs, used as separate processes by the eigenray tier
  # and as test oracles (docs/OPEN_SOURCE.md). Built outside the source tree; never linked.
  apt-get install -y -qq gfortran make > /dev/null
  git clone -q --depth 1 https://github.com/oalib-acoustics/Acoustics-Toolbox /opt/at
  sed -i 's/-mcpu=apple-m2/-march=native/' /opt/at/Makefile
  (cd /opt/at && make -k all > /opt/at/build.log 2>&1 || true)
  ls -la /opt/at/Bellhop/bellhop.exe /opt/at/Kraken/kraken.exe /opt/at/KrakenField/field.exe
  export GPUSONAR_BELLHOP=/opt/at/Bellhop/bellhop.exe GPUSONAR_KRAKEN_DIR=/opt/at
fi
if [ "${GPUSONAR_NVUS:-0}" = "1" ]; then
  # The NVIDIA ultrasound reference's native module at the pinned commit, without its SDK:
  # enough for the simulated-hardware tests of the nvus_scanner source.
  pip install -q nanobind==3.0.1
  git clone -q https://github.com/nvidia-holoscan/holoscan-ultrasound-imaging /opt/nvus-src
  git -C /opt/nvus-src checkout -q 24d405e
  cmake -S /opt/nvus-src -B /opt/nvus-build -G Ninja -DCMAKE_BUILD_TYPE=Release -DNVUS_BUILD_PYTHON=ON \
        -DPython_EXECUTABLE="$(which python3)" > /dev/null
  cmake --build /opt/nvus-build > /dev/null
  mkdir -p /opt/nvus-pkg/nvus
  ln -sf /opt/nvus-src/python/nvus/* /opt/nvus-pkg/nvus/
  ln -sf /opt/nvus-build/_scanner*.so /opt/nvus-pkg/nvus/
  export GPUSONAR_NVUS_PYTHON=/opt/nvus-pkg GPUSONAR_NVUS_DIR=/opt/nvus-src
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
PYTHONPATH=python python3 -m pytest -q -rs tests/python "$@"
