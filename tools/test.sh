#!/usr/bin/env bash
# One command to configure, build and run every test (GoogleTest and pytest).
# Usage: tools/test.sh [extra ctest args]
case "${1:-}" in -h|--help) sed -n '2,/^[^#]/{/^#/s/^# \{0,1\}//p}' "$0"; exit 0;; esac
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build="${SONAR_BUILD_DIR:-$root/build}"
cmake -S "$root" -B "$build" -G Ninja -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}" ${SONAR_CMAKE_ARGS:-}
cmake --build "$build"
ctest --test-dir "$build" --output-on-failure "$@"
