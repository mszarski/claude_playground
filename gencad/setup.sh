#!/usr/bin/env bash
# Assemble a working GenCAD tree: clone upstream at a pinned commit, apply our patch, and symlink upstream's
# files next to ours. Run from anywhere; idempotent. Afterwards, run everything from this gencad/ directory.
#
# Upstream GenCAD (https://github.com/ferdous-alam/GenCAD) ships without a license, so it is fetched at setup
# time instead of being committed here. Everything tracked in this folder is our own code.
set -euo pipefail
cd "$(dirname "$0")"
UPSTREAM_URL=https://github.com/ferdous-alam/GenCAD
UPSTREAM_REV=f5484cf90e30d15b980313006a1424f8949b3496  # commit the multimodal code was written and tested against

if [[ ! -d upstream/.git ]]; then
  git clone -q "$UPSTREAM_URL" upstream
fi
git -C upstream checkout -q "$UPSTREAM_REV"
# apply our train_gencad.py CLI overrides (skip if already applied)
if git -C upstream apply --check ../patches/train_gencad.patch 2>/dev/null; then
  git -C upstream apply ../patches/train_gencad.patch
  echo "[setup] applied patches/train_gencad.patch"
fi

# link every upstream top-level entry we don't define ourselves
for src in upstream/*; do
  name=$(basename "$src")
  [[ -e "$name" && ! -L "$name" ]] && continue
  ln -sfn "$src" "$name"
done
echo "[setup] linked upstream into $(pwd); upstream at $(git -C upstream rev-parse --short HEAD)"
echo "[setup] next: python -m pytest tests/test_mm_smoke.py -q"
