#!/usr/bin/env python3
"""Example 12 of docs/handoff/EXAMPLES.md: the Python API in ten lines (the method of
tests/python/test_detect.py): load a scenario, simulate, beamform, detect.

    PYTHONPATH=python python3 docs/handoff/examples/api_ten_lines.py
"""
import numpy as np

from gpusonar import detect, flank
from gpusonar.config import load_scenario_config
from gpusonar.sim import run_scenario

cfg = load_scenario_config("point_grid_flank_32", {"receiver": {"ambient_noise": {"level_db": 45.0}}})   # a scenario plus one override
record = run_scenario(cfg).records[0]                         # simulate: one ping record, 32 channels
bearings = np.radians(np.arange(-60.0, 60.5, 1.0))            # from broadside
bm = flank.farfield_beamform(record, bearings, window="hann").valid()   # pulse compression and beams: [bearing, range]
power = np.abs(bm.data) ** 2
hits = detect.cfar(power, pfa=1e-9)                           # cell-averaging CFAR along range; the map has 1.9e7 cells
found = detect.cluster(hits, power, bm.ranges, bearings)      # one detection per group of cells
for d in sorted(found, key=lambda d: (round(d.range, -1), d.bearing)):
    print(f"range {d.range:8.1f} m   bearing {np.degrees(d.bearing):6.1f} deg   snr {d.snr_db:5.1f} dB")
print(f"{len(found)} detections in a map of {power.size:.1e} cells; the scenario has 20 targets, at 1, 3, 5 and 9 km on bearings -40, -20, 0, 20, 40 deg")
