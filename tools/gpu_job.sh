#!/usr/bin/env bash
# Build with the CUDA backend and run every test on a GPU machine.
# Written for a clean nvidia/cuda:*-devel-ubuntu24.04 container (for example a Hugging
# Face Job); on a provisioned host just run tools/test.sh.
#
# Usage: tools/gpu_job.sh            from a checkout
#        GPUSONAR_TARBALL_REPO=<user>/<dataset> tools/gpu_job.sh    fetch the source first
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
apt-get update -qq
apt-get install -y -qq cmake ninja-build g++ git python3-dev python3-pip python3-venv \
    libhdf5-dev libeigen3-dev libgtest-dev libgmock-dev libomp-dev > /dev/null
python3 -m venv /opt/venv
. /opt/venv/bin/activate
pip install -q numpy scipy h5py pyyaml pytest pybind11 hdf5plugin huggingface_hub
# Holoscan SDK 4.x (CUDA 12 build) for the operator tests; they skip if this fails.
pip install -q holoscan-cu12 || echo "holoscan not installed"
ulimit -s 32768 || true
if [ -n "${GPUSONAR_TARBALL_REPO:-}" ]; then
  hf download "$GPUSONAR_TARBALL_REPO" gpusonar.tar.gz --type dataset --local-dir /work > /dev/null
  mkdir -p /work/src && tar -xzf /work/gpusonar.tar.gz -C /work/src
  cd /work/src
fi
nvcc --version | tail -1
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DSONAR_ENABLE_CUDA=ON \
      -DPython3_EXECUTABLE="$(which python3)"
cmake --build build
if [ "${GPUSONAR_JOB:-tests}" = "benchmark" ]; then
  PYTHONPATH=python python3 tools/benchmark.py --backend cuda --scale "${GPUSONAR_SCALE:-2}"
  exit 0
fi
echo "=== ctest (C++ suites) ==="
ctest --test-dir build --output-on-failure -E pytest | tail -5
echo "=== pytest (all, with skip reasons) ==="
PYTHONPATH=python python3 -m pytest -q -rs tests/python "$@"
