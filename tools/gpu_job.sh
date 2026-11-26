#!/usr/bin/env bash
# Build with the CUDA backend and run every test on a GPU machine.
# Written for a clean nvidia/cuda:*-devel-ubuntu24.04 container (for example a Hugging
# Face Job); on a provisioned host just run tools/test.sh.
#
# Usage: tools/gpu_job.sh [pytest args]  from a checkout; the default job (GPUSONAR_JOB unset or "tests") builds,
#                                        runs ctest and then pytest on tests/python with these args appended
#        GPUSONAR_TARBALL_REPO=<user>/<dataset> tools/gpu_job.sh    fetch gpusonar.tar.gz from that dataset first
#        SONAR_CMAKE_ARGS="..." tools/gpu_job.sh    extra arguments for the CMake configure step
#        GPUSONAR_ORACLES=1 tools/gpu_job.sh    also build Bellhop, Bellhop3D and KRAKEN and install the PE oracle PyRAM
#                                               1.3.0 for the tests that need them, and set GPUSONAR_REQUIRE_ORACLES=1 so
#                                               that an oracle test that skips fails instead
#        GPUSONAR_CERES=1 tools/gpu_job.sh      also build Ceres, so motion estimation is built and tested
#        GPUSONAR_HOLOSCAN_CPP=1 tools/gpu_job.sh  also install the Holoscan C++ package and build the C++ operators
#        GPUSONAR_NVUS=1 tools/gpu_job.sh       also build the ultrasound reference's native module (no SDK)
#        GPUSONAR_SHARD=I/N tools/gpu_job.sh    run shard I of N of the test files only (tools/test_shards.py), so the
#                                               suite can run as N jobs at once; the C++ suites run in shard 1; the
#                                               suite has passed when every shard has
#        GPUSONAR_HSB=1 tools/gpu_job.sh        also build Holoscan Sensor Bridge 2.7.0 (emulator, Linux receiver, Python;
#                                               implies GPUSONAR_HOLOSCAN_CPP=1) for tests/python/test_hsb.py; where raw
#                                               sockets are refused it sets HSB_UDP_PATCH=1 HSB_EMULATOR_UDP_SOCKET=1 itself
# Other jobs (GPUSONAR_JOB=<mode>; each runs instead of the tests and exits with the tool's status):
#        GPUSONAR_JOB=benchmark tools/gpu_job.sh  run tools/benchmark.py --backend cuda (GPUSONAR_SCALE, default 2;
#                                               GPUSONAR_PIXEL, default 0.05 m; GPUSONAR_MODES, default sas,fls,flank);
#                                               output in the log only, nothing uploaded
#        GPUSONAR_JOB=gallery tools/gpu_job.sh  run tools/gallery.py --backend cuda --out /work/gallery (set GPUSONAR_ORACLES=1
#                                               and GPUSONAR_CERES=1 too; GPUSONAR_GALLERY_ARGS appended); uploaded to the
#                                               dataset folder GPUSONAR_UPLOAD_PATH, default gallery
#        GPUSONAR_JOB=elevbench tools/gpu_job.sh  run tools/popoto_elevation_bench.py --full-rate --backend cuda
#                                               (GPUSONAR_ELEVBENCH_ARGS appended); uploads docs/design/popoto_stage2.json
#                                               to GPUSONAR_UPLOAD_PATH, default elevbench
#        GPUSONAR_JOB=liveablate tools/gpu_job.sh  run tools/live_chain_ablation.py --backend cuda (issue sonar-y7a;
#                                               GPUSONAR_LIVEABLATE_ARGS appended); uploads live_chain_ablation.json to
#                                               GPUSONAR_UPLOAD_PATH, default liveablate
#        GPUSONAR_JOB=benchall tools/gpu_job.sh  run tools/gpu_benchmark_all.py: every stage with a CUDA backend, host
#                                               against CUDA (GPUSONAR_ORACLES=1 for the Bellhop scenarios; GPUSONAR_SCALE,
#                                               default 1; GPUSONAR_BENCHALL_ARGS e.g. "--only live,imaging --repeats 5";
#                                               GPUSONAR_BENCHALL_PYTEST="<-k expression>" runs those tests first;
#                                               GPUSONAR_UPLOAD_PATH: folder in the upload dataset, default benchall)
#        GPUSONAR_JOB=clseval tools/gpu_job.sh  run tools/classify_rules_eval.py --backend cuda (the opt-in classification
#                                               rules over several seeds; set GPUSONAR_ORACLES=1 for the Popoto scenes;
#                                               GPUSONAR_CLSEVAL_ARGS appended, e.g. "--seeds 10 --commit abc1234"); uploads
#                                               the JSON, the table and the captures to GPUSONAR_UPLOAD_PATH, default clseval
#        GPUSONAR_JOB=falsealarm tools/gpu_job.sh  run tools/detector_search_factor.py --at-default --backend cuda: the
#                                               detector's false detections at the default budget on noise-only pings
#                                               (issue sonar-ydi; GPUSONAR_FA_ARGS appended, e.g. "--maps long_range");
#                                               uploads at_default.json to GPUSONAR_UPLOAD_PATH, default falsealarm
#        GPUSONAR_JOB=motion tools/gpu_job.sh  (with GPUSONAR_CERES=1) the motion-estimation tests
#                                               (tests/python/test_motion*.py), then tools/motion_study.py timing
#                                               (delay estimation: NumPy, C++ host, CUDA); log only, nothing uploaded
# Uploads (gallery, elevbench, liveablate, benchall, clseval, falsealarm) happen only when GPUSONAR_UPLOAD_REPO names a Hugging Face
# dataset (the job needs a token that may write to it).
case "${1:-}" in -h|--help) sed -n '2,/^[^#]/{/^#/s/^# \{0,1\}//p}' "$0"; exit 0;; esac
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
  # PyRAM (BSD-3, docs/OPEN_SOURCE.md): a parabolic-equation test oracle, imported by tests/python only. The numpy
  # the suite was built against is held (numba bounds numpy); a failure does not stop the job: the oracle gate
  # then fails the PyRAM tests, which name `pyram`, under GPUSONAR_REQUIRE_ORACLES.
  pip install -q pyram==1.3.0 "numpy==$(python3 -c 'import numpy; print(numpy.__version__)')" || echo "pyram not installed"
  # The oracles were asked for: a test that skips for want of one fails instead (tests/python/oracle_gate.py).
  export GPUSONAR_REQUIRE_ORACLES="${GPUSONAR_REQUIRE_ORACLES:-1}"
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
    hf upload "$GPUSONAR_UPLOAD_REPO" /work/gallery "${GPUSONAR_UPLOAD_PATH:-gallery}" --type dataset > /dev/null
  fi
  exit 0
fi
if [ "${GPUSONAR_JOB:-tests}" = "elevbench" ]; then
  # One full-length ping at the boards' rate per row count through the C++ chain on the device
  # (tools/popoto_elevation_bench.py --full-rate); the updated popoto_stage2.json is uploaded when a repo is set.
  status=0
  PYTHONPATH=python python3 tools/popoto_elevation_bench.py --full-rate --backend cuda ${GPUSONAR_ELEVBENCH_ARGS:-} || status=$?
  if [ -n "${GPUSONAR_UPLOAD_REPO:-}" ]; then
    mkdir -p /work/elevbench && cp docs/design/popoto_stage2.json /work/elevbench/ && \
      hf upload "$GPUSONAR_UPLOAD_REPO" /work/elevbench "${GPUSONAR_UPLOAD_PATH:-elevbench}" --type dataset > /dev/null
  fi
  exit "$status"
fi
if [ "${GPUSONAR_JOB:-tests}" = "liveablate" ]; then
  # The live chain's time per ping on the device with the deployment files' additions switched off one at a time
  # (tools/live_chain_ablation.py, issue sonar-y7a); the JSON is uploaded when a repo is set.
  mkdir -p /work/liveablate
  status=0
  PYTHONPATH=python python3 tools/live_chain_ablation.py --backend cuda --json /work/liveablate/live_chain_ablation.json ${GPUSONAR_LIVEABLATE_ARGS:-} || status=$?
  if [ -n "${GPUSONAR_UPLOAD_REPO:-}" ] && [ -f /work/liveablate/live_chain_ablation.json ]; then
    hf upload "$GPUSONAR_UPLOAD_REPO" /work/liveablate "${GPUSONAR_UPLOAD_PATH:-liveablate}" --type dataset > /dev/null
  fi
  exit "$status"
fi
if [ "${GPUSONAR_JOB:-tests}" = "benchall" ]; then
  # Every stage with a CUDA backend timed on the host and on the device (tools/gpu_benchmark_all.py):
  # the table goes to the log, and with the JSON to GPUSONAR_UPLOAD_REPO (a Hugging Face dataset) when it is set.
  mkdir -p /work/benchall
  status=0
  if [ -n "${GPUSONAR_BENCHALL_PYTEST:-}" ]; then
    # A narrow test selection first (its prints shown), e.g. GPUSONAR_BENCHALL_PYTEST="batched or reverb".
    PYTHONPATH=python python3 -m pytest -q -s -rs tests/python -k "$GPUSONAR_BENCHALL_PYTEST" || echo "PRETEST_EXIT $?"
  fi
  PYTHONPATH=python python3 tools/gpu_benchmark_all.py --backend cuda --scale "${GPUSONAR_SCALE:-1}" --out /work/benchall/gpu_benchmark_all.md \
      --json /work/benchall/gpu_benchmark_all.json ${GPUSONAR_BENCHALL_ARGS:-} || status=$?
  [ -f /work/benchall/gpu_benchmark_all.md ] && cat /work/benchall/gpu_benchmark_all.md
  if [ -n "${GPUSONAR_UPLOAD_REPO:-}" ] && [ -f /work/benchall/gpu_benchmark_all.md ]; then
    hf upload "$GPUSONAR_UPLOAD_REPO" /work/benchall "${GPUSONAR_UPLOAD_PATH:-benchall}" --type dataset > /dev/null
  fi
  exit "$status"
fi
if [ "${GPUSONAR_JOB:-tests}" = "clseval" ]; then
  # The opt-in classification rules (D-152) over several seeds of the gallery's mover scenes
  # (tools/classify_rules_eval.py); the JSON, table and captures are uploaded when a repo is set.
  mkdir -p /work/clseval
  status=0
  PYTHONPATH=python python3 tools/classify_rules_eval.py --backend cuda --captures /work/clseval/captures \
      --json /work/clseval/classify_rules_eval.json --table /work/clseval/classify_rules_eval.md ${GPUSONAR_CLSEVAL_ARGS:-} || status=$?
  if [ -n "${GPUSONAR_UPLOAD_REPO:-}" ]; then
    hf upload "$GPUSONAR_UPLOAD_REPO" /work/clseval "${GPUSONAR_UPLOAD_PATH:-clseval}" --type dataset > /dev/null
  fi
  exit "$status"
fi
if [ "${GPUSONAR_JOB:-tests}" = "falsealarm" ]; then
  # The multiscale detector's false detections at the default budget on noise-only pings through the C++ chain on the
  # device (tools/detector_search_factor.py --at-default, issue sonar-ydi; GPUSONAR_FA_ARGS appended, e.g.
  # "--maps long_range --pings 100000"); the JSON is uploaded to GPUSONAR_UPLOAD_PATH, default falsealarm, when a repo is set.
  mkdir -p /work/falsealarm
  status=0
  PYTHONPATH=python python3 tools/detector_search_factor.py --at-default --backend cuda --json /work/falsealarm/at_default.json ${GPUSONAR_FA_ARGS:-} || status=$?
  if [ -n "${GPUSONAR_UPLOAD_REPO:-}" ] && [ -f /work/falsealarm/at_default.json ]; then
    hf upload "$GPUSONAR_UPLOAD_REPO" /work/falsealarm "${GPUSONAR_UPLOAD_PATH:-falsealarm}" --type dataset > /dev/null
  fi
  exit "$status"
fi
if [ "${GPUSONAR_JOB:-tests}" = "motion" ]; then
  status=0
  PYTHONPATH=python python3 -m pytest -q -rs tests/python/test_motion.py tests/python/test_motion_delays.py || status=$?
  PYTHONPATH=python python3 tools/motion_study.py timing || status=$?
  exit "$status"
fi
# GPUSONAR_SHARD=I/N: this job runs shard I of N of the test files (tools/test_shards.py: every file in exactly one
# shard); the C++ suites run in shard 1 only. Unset: everything, as before. The per-test seconds printed at the end
# (--durations) are what tools/test_shards.py --weights reads.
shard_files="tests/python"
if [ -n "${GPUSONAR_SHARD:-}" ]; then
  shard_files=$(python3 tools/test_shards.py "$GPUSONAR_SHARD")
  echo "=== shard $GPUSONAR_SHARD: $(echo "$shard_files" | wc -l) test files ==="
fi
if [ -z "${GPUSONAR_SHARD:-}" ] || [ "${GPUSONAR_SHARD%%/*}" = "1" ]; then
  echo "=== ctest (C++ suites) ==="
  ctest --test-dir build --output-on-failure -E pytest | tail -5
fi
echo "=== pytest (${GPUSONAR_SHARD:-all}, with skip reasons) ==="
# shellcheck disable=SC2086
PYTHONPATH=python${PYTHONPATH:+:$PYTHONPATH} python3 -m pytest -q -rs --durations=0 --durations-min=0.5 $shard_files "$@"
