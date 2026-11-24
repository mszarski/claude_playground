#!/usr/bin/env python3
"""Example 11 of docs/handoff/EXAMPLES.md: the image-source provider against Bellhop.

    GPUSONAR_BELLHOP=/path/to/bellhop.exe PYTHONPATH=python \
        python3 docs/handoff/examples/bellhop_pekeris.py --out /tmp/bellhop

The case and the comparison of tests/python/test_propagation.py
`test_image_source_matches_bellhop_in_a_pekeris_waveguide`: 100 m of isovelocity water
(1500 m/s) over a fluid half-space (1600 m/s, 1.8 g/cm^3, 0.5 dB/wavelength), source at 30 m,
receiver at 50 m, 0.1 to 3 km, at 1 kHz. Bellhop (GPL-3) runs as a separate program: it is an
oracle for tests and is never linked into the suite.
"""
import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from gpusonar.sim import ImageSourceProvider, OceanEnvironment, find_bellhop, run_bellhop

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--out", required=True)
ap.add_argument("--frequency", type=float, default=1000.0)
args = ap.parse_args()
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
if find_bellhop() is None:
    raise SystemExit("Bellhop not found: set GPUSONAR_BELLHOP to bellhop.exe (tools/build_oracles.sh builds it)")

f = args.frequency
env = OceanEnvironment(depth=100.0, sound_speed=1500.0, bottom_speed=1600.0, bottom_density=1.8, bottom_attenuation=0.5)
ranges = np.arange(100.0, 3001.0, 50.0)
t0 = time.perf_counter()
arr = run_bellhop(env, f, 30.0, [50.0], ranges)                          # the oracle: arrivals at every range
t1 = time.perf_counter()
provider = ImageSourceProvider(1500.0, surface_z=0.0, bottom_z=100.0, max_order=40,
                               bottom=dict(speed=1600.0, density=1.8, attenuation=0.5))
points = np.column_stack([ranges, np.zeros_like(ranges), np.full_like(ranges, 50.0)])
es = provider.eigenrays(np.array([0.0, 0.0, 30.0]), points)              # ours: images in the surface and the bottom
t2 = time.perf_counter()

worst_delay = max(np.abs(arr.delay[0][j] - d).min() for j in (0, 18, 40, 58) for d in np.sort(es.delay[j])[:4])
tl_ours_inc = -10 * np.log10((np.abs(es.amplitude) ** 2).sum(axis=1))
tl_bh_inc = arr.transmission_loss(coherent=False)[0]
tl_ours = -20 * np.log10(np.abs((es.amplitude * np.exp(-2j * np.pi * f * es.delay)).sum(axis=1)))
tl_bh = arr.transmission_loss()[0]
print(f"Bellhop {t1 - t0:.1f} s, image source {t2 - t1:.2f} s; {len(ranges)} ranges at {f:g} Hz")
print(f"first four arrivals at 0.1, 1, 2.1 and 3 km: delays differ by at most {worst_delay * 1e6:.2f} microseconds")
print(f"incoherent transmission loss: median difference {np.median(np.abs(tl_ours_inc - tl_bh_inc)):.3f} dB, "
      f"worst {np.abs(tl_ours_inc - tl_bh_inc).max():.3f} dB")
print(f"coherent transmission loss:   median difference {np.median(np.abs(tl_ours - tl_bh)):.3f} dB, "
      f"worst {np.abs(tl_ours - tl_bh).max():.3f} dB")
print(f"at 3 km: {tl_bh_inc[-1]:.1f} dB against {20 * np.log10(3000.0):.1f} dB for spherical spreading")

fig, ax = plt.subplots(figsize=(7.5, 4.0), constrained_layout=True)
ax.plot(ranges / 1e3, tl_bh, color="#0b0b0b", lw=2.2, label="Bellhop, coherent")
ax.plot(ranges / 1e3, tl_ours, color="#eb6834", lw=1.3, ls="--", label="image source, coherent")
ax.plot(ranges / 1e3, tl_bh_inc, color="#2a78d6", lw=2.2, label="Bellhop, incoherent")
ax.plot(ranges / 1e3, tl_ours_inc, color="#eda100", lw=1.3, ls="--", label="image source, incoherent")
ax.plot(ranges / 1e3, 20 * np.log10(ranges), color="#888888", lw=1.0, ls=":", label="spherical spreading")
ax.invert_yaxis()
ax.set_xlabel("range (km)")
ax.set_ylabel("transmission loss (dB re 1 m)")
ax.grid(True, alpha=0.3)
ax.legend(frameon=False, fontsize=8)
ax.set_title(f"Pekeris waveguide, 100 m, source 30 m, receiver 50 m, {f:g} Hz", loc="left", fontsize=10)
fig.savefig(out / "bellhop_pekeris.png", dpi=90)
print(f"wrote {out / 'bellhop_pekeris.png'}")
