#!/usr/bin/env python3
"""Example 4 of docs/handoff/EXAMPLES.md: a SAS image with navigation only, with estimated
motion and with the true motion, and the sway and heave errors.

    PYTHONPATH=python python3 docs/handoff/examples/sas_motion.py --out /tmp/sas [--full]

The scene is configs/scenarios/wreck_sas.yaml. Without --full it is thinned exactly as
tools/gallery.py --quick thins it (densities x0.3, 40 pings of 80, 5 cm pixels); --full is the
scenario as shipped (80 pings, 2.5 cm pixels; minutes on a CPU). The steps are those of
tools/gallery.py `run_sas`. Motion estimation needs the build to have Ceres (core.HAVE_MOTION).
"""
import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import gpusonar as gs
from gpusonar import motion, recon
from gpusonar.config import load_scenario_config
from gpusonar.sim import run_scenario

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--out", required=True)
ap.add_argument("--full", action="store_true")
args = ap.parse_args()
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)

cfg = load_scenario_config("wreck_sas")
pixel = 0.025
if not args.full:
    for item in cfg["scene"]:
        if "density" in item:
            item["density"] = float(item["density"]) * 0.3
    cfg["pings"]["count"] = 40
    pixel = 0.05
t0 = time.perf_counter()
result = run_scenario(cfg)                 # records carry what navigation reports: the straight track
records, truth = result.records, result.true_positions
t1 = time.perf_counter()
recon.prepare(records)                     # conditioning and pulse compression, in place
x, y = np.arange(6.0, 13.0, pixel), np.arange(39.0, 46.5, pixel)
X, Y = np.meshgrid(x, y, indexing="ij")
points = np.stack([X, Y, np.zeros_like(X)], axis=-1)       # the seafloor plane, z = 0


def image(recs):
    return np.abs(recon.sas_image(recs, points, interp="linear")) ** 2


def sharpness(im):
    return float((im ** 2).sum() / im.sum() ** 2 * im.size)


panels = {"Navigation only": image(records)}
t2 = time.perf_counter()
if not gs.core.HAVE_MOTION:
    raise SystemExit("this build has no motion estimation (Ceres not found at configure time)")
estimate = motion.estimate_motion(records)                 # displaced-phase-centre delays, then the Ceres solve
t3 = time.perf_counter()
corrected = [r.copy() for r in records]
motion.apply_motion(corrected, estimate)
panels["Estimated motion"] = image(corrected)
exact = [r.copy() for r in records]
for r, pos, vel in zip(exact, truth, np.gradient(truth, float(cfg["pings"]["interval"]), axis=0)):
    geo = r.geometry
    pose = geo.rx_pose
    pose.position, pose.velocity = tuple(pos), tuple(vel)
    geo.rx_pose = geo.tx_pose = pose
    r.geometry = geo
panels["True motion"] = image(exact)

wavelength = records[0].environment.sound_speed / records[0].samples.center_frequency
sway = float(np.sqrt(np.mean((estimate.y - truth[:, 1]) ** 2)))
heave = float(np.sqrt(np.mean((estimate.z - truth[:, 2]) ** 2)))
nav_sway = float(np.sqrt(np.mean((truth[:, 1] - truth[0, 1]) ** 2)))
nav_heave = float(np.sqrt(np.mean((truth[:, 2] - truth[0, 2]) ** 2)))
print(f"pings {len(records)}  scatterers {len(result.scene)}  wavelength {wavelength * 1e3:.2f} mm")
print(f"simulate {t1 - t0:.1f} s  one image {t2 - t1:.1f} s  motion estimate {t3 - t2:.1f} s")
print(f"true motion unknown to navigation: sway {nav_sway * 1e3:.2f} mm rms ({nav_sway / wavelength:.2f} wavelengths), "
      f"heave {nav_heave * 1e3:.2f} mm rms ({nav_heave / wavelength:.2f} wavelengths)")
print(f"error of the estimate:             sway {sway * 1e3:.2f} mm rms ({sway / wavelength:.3f} wavelengths), "
      f"heave {heave * 1e3:.2f} mm rms ({heave / wavelength:.3f} wavelengths)")
for k, v in panels.items():
    print(f"sharpness, {k.lower():<17} {sharpness(v):8.2f}")

ref = np.median(panels["True motion"])
fig, axes = plt.subplots(1, 3, figsize=(10.5, 4.2), sharey=True, constrained_layout=True)
for ax, (title, im) in zip(axes, panels.items()):
    m = ax.imshow(10 * np.log10(np.maximum(im / ref, 1e-12)), extent=[y[0], y[-1], x[-1], x[0]], cmap="gray",
                  vmin=-15, vmax=30, aspect="equal")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("ground range y (m)")
axes[0].set_ylabel("along track x (m)")
fig.colorbar(m, ax=axes, label="dB re median seafloor", shrink=0.85)
name = "sas_wreck_full.png" if args.full else "sas_wreck_quick.png"
fig.savefig(out / name, dpi=70)
print(f"wrote {out / name}")
