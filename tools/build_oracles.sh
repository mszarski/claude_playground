#!/usr/bin/env bash
# Build the external propagation programs (Acoustics Toolbox: Bellhop, Bellhop3D, KRAKEN, FIELD) for
# the eigenray tier and the cross-check tests. CPU only; needs gfortran and make.
# They are GPL-3 and are run as separate programs: nothing is linked into this suite.
#
# Usage: tools/build_oracles.sh <out-dir>
#   then: export GPUSONAR_BELLHOP=<out-dir>/Bellhop/bellhop.exe GPUSONAR_KRAKEN_DIR=<out-dir>
#   (Bellhop3D is found as bellhop3d.exe beside bellhop.exe: tests/python/bellhop3d_oracle.py)
case "${1:-}" in -h|--help) sed -n '2,/^[^#]/{/^#/s/^# \{0,1\}//p}' "$0"; exit 0;; esac
set -euo pipefail
if [ $# -lt 1 ] || [ "${1#-}" != "$1" ]; then
  echo "usage: $0 <out-dir> (an option is not a directory; --help prints the header)" >&2; exit 2
fi
out=$1
command -v gfortran > /dev/null || { echo "gfortran is not installed (apt-get install gfortran)" >&2; exit 1; }
rm -rf "$out"
git clone -q --depth 1 https://github.com/oalib-acoustics/Acoustics-Toolbox "$out"
# The Makefile at HEAD selects an Apple CPU; use the build machine's own.
sed -i 's/-mcpu=apple-m2/-march=native/' "$out/Makefile"
(cd "$out" && make -k all > "$out/build.log" 2>&1 || true)
ls "$out/Bellhop/bellhop.exe" "$out/Bellhop/bellhop3d.exe" "$out/Kraken/kraken.exe" "$out/KrakenField/field.exe" > /dev/null
echo "export GPUSONAR_BELLHOP=$out/Bellhop/bellhop.exe GPUSONAR_KRAKEN_DIR=$out"
