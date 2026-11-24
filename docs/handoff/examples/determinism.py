#!/usr/bin/env python3
"""SIMULATION.md section 6: the same scenario and seed twice, another seed, and the NumPy engine.

    PYTHONPATH=python python3 docs/handoff/examples/determinism.py

Scenario: fls_120k_transit cut to two pings (boats as scatterers, seabed and surface
reverberation, ambient noise, ADC). Prints a hash of the samples of every ping.
"""
import hashlib

import numpy as np

from gpusonar.config import load_scenario_config
from gpusonar.sim import run_scenario


def run(seed=None, engine=None):
    cfg = load_scenario_config("fls_120k_transit", {"pings": {"count": 2}})
    if seed is not None:
        cfg["seed"] = seed
    records = run_scenario(cfg, engine=engine).records
    x = np.stack([r.samples.array for r in records])
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()[:16], x


a, xa = run()
b, _ = run()
c, _ = run(seed=1)
d, xd = run(engine="numpy")
print(f"seed of the file, C++ engine:        {a}")
print(f"the same again:                      {b}   identical: {a == b}")
print(f"seed 1:                              {c}   identical: {a == c}")
print(f"seed of the file, NumPy engine:      {d}   identical: {a == d}")
print(f"C++ against NumPy: largest difference {np.abs(xa - xd).max():.3g} of a largest sample {np.abs(xa).max():.3g} "
      f"(relative {np.abs(xa - xd).max() / np.abs(xa).max():.1e})")
