#!/usr/bin/env bash
# Build the NVIDIA ultrasound reference's native module (CPU only, no Sensor Bridge SDK)
# from its pinned commit with this repository's long-window patch applied, and lay it out
# as an importable `nvus` package. Nothing is changed in the checkout you point it at:
# the work happens in a copy.
#
# Usage: tools/build_nvus_reference.sh <out-dir> [<reference checkout or URL>]
#   then: export GPUSONAR_NVUS_PYTHON=<out-dir>/pkg GPUSONAR_NVUS_DIR=<out-dir>/src
# Set NVUS_APPLY_PATCH=0 to build the reference unmodified (0.88 m window limit).
# On the rig, apply the same patch to the reference before running its ./setup.
set -euo pipefail
out=$(mkdir -p "$1" && cd "$1" && pwd)
ref=${2:-https://github.com/nvidia-holoscan/holoscan-ultrasound-imaging}
here=$(cd "$(dirname "$0")/.." && pwd)
commit=24d405e
rm -rf "$out/src" "$out/build" "$out/pkg"
git clone -q "$ref" "$out/src"
git -C "$out/src" -c advice.detachedHead=false checkout -q "$commit"
if [ "${NVUS_APPLY_PATCH:-1}" = "1" ]; then
  git -C "$out/src" apply "$here/third_party/patches/holoscan-ultrasound-imaging-$commit-long-window.patch"
fi
python3 -c "import nanobind" 2>/dev/null || python3 -m pip install -q nanobind==3.0.1
cmake -S "$out/src" -B "$out/build" -G Ninja -DCMAKE_BUILD_TYPE=Release -DNVUS_BUILD_PYTHON=ON \
      -DPython_EXECUTABLE="$(which python3)" > /dev/null
cmake --build "$out/build" > /dev/null
mkdir -p "$out/pkg/nvus"
ln -sf "$out"/src/python/nvus/* "$out/pkg/nvus/"
ln -sf "$out"/build/_scanner*.so "$out/pkg/nvus/"
echo "export GPUSONAR_NVUS_PYTHON=$out/pkg GPUSONAR_NVUS_DIR=$out/src"
