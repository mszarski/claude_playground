#!/usr/bin/env python3
"""Example 3 of docs/handoff/EXAMPLES.md: an FLS fan image of a simulated scene.

    PYTHONPATH=python python3 docs/handoff/examples/fls_image.py --out /tmp/fls [--density 0.25]

The scene is configs/scenarios/seafloor_objects_fls.yaml with its scatterer densities
multiplied by --density (1.0: the scenario as shipped). The imaging call and the display
mapping are those of tools/gallery.py `run_fls`.
"""
import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import gpusonar as gs
from gpusonar import recon
from gpusonar.config import load_scenario_config
from gpusonar.sim import run_scenario

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--out", required=True)
ap.add_argument("--density", type=float, default=0.25)
args = ap.parse_args()
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)

cfg = load_scenario_config("seafloor_objects_fls")
for item in cfg["scene"]:                                   # thin the scene: the gallery's --quick does the same
    if "density" in item:
        item["density"] = float(item["density"]) * args.density
t0 = time.perf_counter()
result = run_scenario(cfg)                                  # one ping, 128 channels, complex baseband
t1 = time.perf_counter()
ranges = np.arange(3.6, 10.0, 0.0025)                       # m, about the range resolution
bearings = np.radians(np.arange(-22.0, 22.0, 0.2))
img = recon.fan_image(result.records, ranges, bearings, interp="sinc", window="hann")
t2 = time.perf_counter()
power = np.abs(img.image) ** 2 * ranges[:, None] ** 3       # range gain for display
power = power[:len(ranges) // 4 * 4].reshape(-1, 4, power.shape[1]).mean(axis=1)
r = ranges[:len(ranges) // 4 * 4].reshape(-1, 4).mean(axis=1)
shown = gs.core.drc(np.sqrt(power).astype(np.float32), target_median=0.2)[0]   # dynamic range compression

R, B = np.meshgrid(r, bearings, indexing="ij")
fig, ax = plt.subplots(figsize=(6.4, 5.0), constrained_layout=True)
ax.pcolormesh(R * np.sin(B), R * np.cos(B), shown, cmap="gray", vmin=0.0, vmax=1.0, shading="auto", rasterized=True)
ax.set_aspect("equal")
ax.set_xlabel("across (m)")
ax.set_ylabel("forward (m)")
ax.set_title(f"seafloor_objects_fls, density x{args.density:g}: {len(result.scene)} scatterers (simulated)", fontsize=9)
fig.savefig(out / "fls_seafloor_objects_quick.png", dpi=80)
print(f"scatterers {len(result.scene)}  samples {result.records[0].samples.array.shape}  "
      f"simulate {t1 - t0:.1f} s  image {img.image.shape} in {t2 - t1:.1f} s")
print(f"dynamic range {10 * np.log10(power.max() / np.median(power)):.1f} dB (peak over median)")
print(f"wrote {out / 'fls_seafloor_objects_quick.png'}")
