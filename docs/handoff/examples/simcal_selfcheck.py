#!/usr/bin/env python3
"""Example 9 of docs/handoff/EXAMPLES.md: the simulator-calibration self-check.

    PYTHONPATH=python python3 docs/handoff/examples/simcal_selfcheck.py --out /tmp/simcal

The pattern of tests/python/test_simcal.py, through the command-line tool. A "real" log is
simulated with settings the calibration is not told (bottom 6 dB stronger, surface 4 dB weaker,
ambient noise 38 dB re 1 uPa^2/Hz, targets in the water), written to a record file, and
tools/calibrate_sim.py has to find the three numbers from the file and the unchanged scenario.
The log comes from the same simulator, so this checks the method, not the models.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

import yaml

from gpusonar.config import load_scenario_config
from gpusonar.sim import run_scenario

ROOT = Path(__file__).resolve().parents[3]
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--out", required=True)
args = ap.parse_args()
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)

# The scenario the calibration starts from: fls_120k_transit in 45 m of water (the seabed well inside
# the beam), direct paths only. A scenario file may extend another with `base:`.
start = out / "dive_start.yaml"
start.write_text("base: fls_120k_transit\nname: dive_start\nfidelity: free_field\npropagation: null\n"
                 "environment: {water_depth_m: 45.0}\n")

hidden = load_scenario_config(str(start))
hidden["reverberation"][0]["offset_db"] = 6.0          # the bottom of the "real" sea
hidden["reverberation"][1]["offset_db"] = -4.0         # its surface
hidden["receiver"]["ambient_noise"] = {"level_db": 38.0}
hidden["pings"]["count"] = 6
hidden["seed"] = 99
t0 = time.perf_counter()
run_scenario(hidden, output=str(out / "log.h5"))
print(f"hidden log: 6 pings in {time.perf_counter() - t0:.0f} s -> {out / 'log.h5'}")
print("hidden: bottom +6.0 dB, surface -4.0 dB, noise 38.0 dB re 1 uPa^2/Hz")
cmd = [sys.executable, str(ROOT / "tools" / "calibrate_sim.py"), "fls", "--log", str(out / "log.h5"), "--scenario", str(start),
       "--levels", "absolute", "--ranges", "25.0", "284.2", "0.06", "--out", str(out / "dive_cal.yaml"),
       "--plot", str(out / "simcal_fit.png")]
print("$ " + " ".join(cmd[1:]))
t0 = time.perf_counter()
subprocess.run(cmd, check=True)
print(f"calibration: {time.perf_counter() - t0:.0f} s")
print((out / "dive_cal.yaml").read_text())
yaml.safe_load((out / "dive_cal.yaml").read_text())     # the overrides are a scenario fragment
