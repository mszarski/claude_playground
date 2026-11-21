#!/usr/bin/env bash
# Build the external propagation programs (Acoustics Toolbox: Bellhop, Bellhop3D, KRAKEN, FIELD) for
# the eigenray tier and the cross-check tests. CPU only; needs gfortran and make.
# They are GPL-3 and are run as separate programs: nothing is linked into this suite.
# The clone is pinned to commit 8b4682b (2025-10-08), the one the oracle tests' margins were measured
# against (docs/OPEN_SOURCE.md), not the moving HEAD.
#
# Usage: tools/build_oracles.sh <out-dir>
#   The last line printed is: export GPUSONAR_BELLHOP=<out-dir>/Bellhop/bellhop.exe GPUSONAR_KRAKEN_DIR=<out-dir>
#   with GPUSONAR_BELLHOP3D=<out-dir>/Bellhop/bellhop3d.exe added when Bellhop3D built. Bellhop, KRAKEN and
#   FIELD are required (the script fails without them); a missing Bellhop3D only warns (its tests then skip,
#   or fail under GPUSONAR_REQUIRE_ORACLES=1: tests/python/oracle_gate.py).
case "${1:-}" in -h|--help) sed -n '2,/^[^#]/{/^#/s/^# \{0,1\}//p}' "$0"; exit 0;; esac
set -euo pipefail
if [ $# -lt 1 ] || [ "${1#-}" != "$1" ]; then
  echo "usage: $0 <out-dir> (an option is not a directory; --help prints the header)" >&2; exit 2
fi
out=$1
command -v gfortran > /dev/null || { echo "gfortran is not installed (apt-get install gfortran)" >&2; exit 1; }
rm -rf "$out"
commit=8b4682bd719d5d17f5a1c798039d6be8235e3859
git init -q "$out"
git -C "$out" fetch -q --depth 1 https://github.com/oalib-acoustics/Acoustics-Toolbox "$commit"
git -C "$out" checkout -q FETCH_HEAD
# The Makefile at that commit selects an Apple CPU; use the build machine's own.
sed -i 's/-mcpu=apple-m2/-march=native/' "$out/Makefile"
(cd "$out" && make -k all > "$out/build.log" 2>&1 || true)
ls "$out/Bellhop/bellhop.exe" "$out/Kraken/kraken.exe" "$out/KrakenField/field.exe" > /dev/null
line="export GPUSONAR_BELLHOP=$out/Bellhop/bellhop.exe GPUSONAR_KRAKEN_DIR=$out"
if [ -x "$out/Bellhop/bellhop3d.exe" ]; then
  line="$line GPUSONAR_BELLHOP3D=$out/Bellhop/bellhop3d.exe"
else
  echo "warning: Bellhop3D did not build ($out/build.log); its oracle tests will skip" >&2
fi
echo "$line"
