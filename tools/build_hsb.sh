#!/usr/bin/env bash
# Build NVIDIA Holoscan Sensor Bridge (Apache-2.0) natively, without its Docker flow: the
# board emulator, the Linux socket receiver and the Python bindings, against the Holoscan
# C++ package in /opt/nvidia/holoscan. For gpusonar/sources/hsb.py and tools/sonar_live_hsb.py.
#
# Usage: tools/build_hsb.sh <prefix> [<checkout or URL>]
#   then: eval the last line it prints (PYTHONPATH and LD_LIBRARY_PATH).
# Needs: a CUDA 12 toolkit with nvcc, Holoscan 4.x C++ package and the matching `holoscan`
# Python wheel in the active python3 (tools/gpu_job.sh with GPUSONAR_HOLOSCAN_CPP=1 installs
# both), CMake >= 3.24, root for apt (HSB_SKIP_APT=1 when the packages are already there).
# HSB_JOBS (default 6) bounds the parallel compiles: each takes over a gigabyte, and a container
# reports the host's cores, not its own share.
# HSB_CUDA_ARCH (default: native) is passed as CMAKE_CUDA_ARCHITECTURES: set it to a number
# (75: T4, 89: L4) when building on a machine without the GPU.
#
# Tag 2.7.0 is the last one whose CMake accepts Holoscan 4 / CUDA 12. Left out: tests, tools,
# examples, RoCE (needs ibverbs and real NICs), GPUNetIO, FuSa, SIPL, Argus, PVA.
set -euo pipefail
prefix=$(mkdir -p "$1" && cd "$1" && pwd)
ref=${2:-https://github.com/nvidia-holoscan/holoscan-sensor-bridge}
tag=2.7.0
export DEBIAN_FRONTEND=noninteractive
if [ "${HSB_SKIP_APT:-0}" != "1" ]; then
  # What Sensor Bridge's docker/Dockerfile installs, less the display, RoCE and DOCA parts.
  apt-get install -y -qq --no-install-recommends git curl ca-certificates ninja-build pkgconf zlib1g-dev \
      libfmt-dev libyaml-cpp-dev libcurlpp-dev libcurl4-openssl-dev libssl-dev libnuma-dev libelf-dev > /dev/null
  if ! dpkg -s libnvcomp5-dev-cuda-12 > /dev/null 2>&1; then
    # nvCOMP (its CRC operator is not optional in the build): NVIDIA's local apt repository.
    . /etc/os-release
    deb=nvcomp-local-repo-ubuntu${VERSION_ID//./}-5.2.0_5.2.0-1_$(dpkg --print-architecture).deb
    curl -fsSL -o "/tmp/$deb" "https://developer.download.nvidia.com/compute/nvcomp/5.2.0/local_installers/$deb"
    dpkg -i "/tmp/$deb" > /dev/null
    cp /var/nvcomp-local-repo-*/nvcomp-*-keyring.gpg /usr/share/keyrings/
    apt-get update -qq
    apt-get install -y -qq libnvcomp5-cuda-12 libnvcomp5-dev-cuda-12 nvcomp-cuda-12 > /dev/null
    rm -f "/tmp/$deb"
  fi
fi
rm -rf "$prefix/src" "$prefix/build"
git clone -q -c advice.detachedHead=false --depth 1 -b "$tag" "$ref" "$prefix/src"
cmake -S "$prefix/src" -B "$prefix/build" -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF \
      -DCMAKE_CUDA_ARCHITECTURES="${HSB_CUDA_ARCH:-native}" -DPython3_ROOT_DIR="$(python3 -c 'import sys; print(sys.prefix)')" \
      -DPython3_FIND_VIRTUALENV=FIRST \
      -DHOLOLINK_BUILD_PYTHON=ON -DHOLOLINK_BUILD_EMULATOR=ON -DHOLOLINK_BUILD_SOCKET=ON \
      -DHOLOLINK_BUILD_ROCE=OFF -DHOLOLINK_BUILD_GPUNETIO_ROCE=OFF -DHOLOLINK_BUILD_TESTS=OFF -DHOLOLINK_BUILD_TOOLS=OFF \
      -DHOLOLINK_BUILD_EXAMPLES=OFF -DHOLOLINK_BUILD_FUSA=OFF -DHOLOLINK_BUILD_SIPL=OFF -DHOLOLINK_BUILD_ARGUS_ISP=OFF \
      -DHOLOLINK_BUILD_PVA=OFF -DHOLOLINK_BUILD_CPACK=OFF ${HSB_CMAKE_ARGS:-} > "$prefix/configure.log" 2>&1 \
  || { tail -40 "$prefix/configure.log" >&2; exit 1; }
cmake --build "$prefix/build" -j "${HSB_JOBS:-6}" > "$prefix/build.log" 2>&1 \
  || { { grep -B2 -A25 -m3 "FAILED:" "$prefix/build.log"; tail -5 "$prefix/build.log"; } >&2; exit 1; }
# The package as the build lays it out (python/lib/hololink); the modules find Holoscan's libraries by rpath.
py="$prefix/build/python/lib"
env="export PYTHONPATH=$py\${PYTHONPATH:+:\$PYTHONPATH} LD_LIBRARY_PATH=/opt/nvidia/holoscan/lib\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}"
(eval "$env"; python3 -c "import hololink, hololink.emulation as e; hololink.operators.LinuxReceiverOp; e.HSBEmulator; e.LinuxDataPlane") \
  || { echo "Sensor Bridge built but does not import" >&2; exit 1; }
echo "$env"
