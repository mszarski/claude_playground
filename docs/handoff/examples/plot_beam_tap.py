#!/usr/bin/env python3
"""Example 5 of docs/handoff/EXAMPLES.md: draw the beamform tap of one ping of a live-chain run.

    PYTHONPATH=python python3 docs/handoff/examples/plot_beam_tap.py RUN/taps/ping0000 --out beam.png

RUN is the --out directory of tools/sonar_live.py. The tap holds the range x bearing power map
the detector saw (taps.read_arrays) and the ping's report; detections are drawn on the map.
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from gpusonar import taps

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("tap")
ap.add_argument("--out", required=True)
args = ap.parse_args()
beam = taps.read_arrays(args.tap, "beamform")                      # bearings (deg or rad), ranges (m), power [bearing, range]
entry = next(e for e in taps.manifest(args.tap) if e["stage"] == "report")   # the file's number depends on the stages run
report = json.loads((Path(args.tap) / entry["json"]).read_text())
power, ranges = np.asarray(beam["power"], dtype=float), np.asarray(beam["ranges"])
bearings = np.asarray(beam["bearings"], dtype=float)
bearings_deg = np.degrees(bearings) if np.abs(bearings).max() < 3.2 else bearings
step = max(1, len(ranges) // 1200)                                 # thin the range axis for the picture: maximum over each block
n = len(ranges) // step * step
shown = power[:, :n].reshape(len(bearings), -1, step).max(axis=2)
level = 10 * np.log10(np.maximum(shown / np.median(shown), 1e-12))
fig, ax = plt.subplots(figsize=(7.5, 3.8), constrained_layout=True)
m = ax.imshow(level, extent=[ranges[0], ranges[n - 1], bearings_deg[-1], bearings_deg[0]], aspect="auto", cmap="gray", vmin=0, vmax=30)
for d in report["detections"]:
    ax.plot(d["range_m"], d["bearing_deg"], "o", mfc="none", mec="#eb6834", ms=11, mew=1.6)
ax.set_xlabel("range (m)")
ax.set_ylabel("bearing from broadside (deg)")
ax.set_title(f"beamform tap, ping {report['ping']}: {len(report['detections'])} detection(s) circled", loc="left", fontsize=10)
fig.colorbar(m, ax=ax, label="dB re median")
fig.savefig(args.out, dpi=80)
print(f"map {power.shape[0]} bearings x {power.shape[1]} ranges ({ranges[0]:.0f} to {ranges[-1]:.0f} m); wrote {args.out}")
