#!/usr/bin/env python3
"""The bands in fls_120k_rough_surface: what makes them, and which frame they are fixed in.

Part 1 images one ping of the surface alone four ways: flat water, two regular wave
trains (the first version of the scene), the irregular sea with tilted facets only, and
the irregular sea as shipped (with tilt-independent small-scale roughness).

Part 2 images two pings a second apart and measures, by cross-correlation, how far the
pattern moved: in the world frame and in the sonar's frame, for the shipped (travelling)
waves and for the same waves frozen in place. Frozen waves must not move in the world
frame and must move back by the sonar's own advance in the sonar frame. Travelling waves
move through the world at the speed of the long waves in the sea.

Writes docs/figures/rough_surface_bands.png and .json.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import gpusonar as gs  # noqa: E402
from gpusonar import recon  # noqa: E402
from gpusonar.config import load_scenario_config  # noqa: E402
from gpusonar.sim import run_scenario  # noqa: E402

H = gs.core.Backend.Host
GX, GY = np.arange(70.0, 286.0, 1.0), np.arange(-120.0, 120.0, 1.0)      # world grid, 1 m
DT = 1.0                                                                # seconds between the two pings


def images(change, pings=1):
    """World-frame images [ping][x, y] of the surface alone (range gain r^3), and the sonar positions."""
    cfg = load_scenario_config("fls_120k_rough_surface")
    cfg["pings"] = {"count": pings, "interval": DT}
    surface = next(i for i in cfg["scene"] if i["type"] == "sea_surface")
    change(surface)
    cfg["scene"] = [surface]
    cfg["reverberation"] = []
    cfg["receiver"] = {"adc": False, "noise_rms": 0.0}
    res = run_scenario(cfg, backend=H)
    ranges = np.arange(40.0, 298.0, 0.06)
    ranges = ranges[:len(ranges) // 8 * 8]
    bearings = np.radians(np.arange(-60.0, 60.01, 0.4))
    r = ranges.reshape(-1, 8).mean(axis=1)
    out = []
    for rec, own in zip(res.records, res.true_positions):
        img = recon.fan_image([rec], ranges, bearings, interp="linear", window="hann", backend=H)
        power = (np.abs(img.image) ** 2).reshape(-1, 8, len(bearings)).mean(axis=1) * r[:, None] ** 3
        X, Y = np.meshgrid(GX, GY, indexing="ij")
        rr, bb = np.hypot(X - own[0], Y - own[1]), np.arctan2(Y - own[1], X - own[0])
        ir = np.clip(np.rint((rr - r[0]) / (r[1] - r[0])).astype(int), 0, len(r) - 1)
        ib = np.clip(np.rint((bb - bearings[0]) / (bearings[1] - bearings[0])).astype(int), 0, len(bearings) - 1)
        g = power[ir, ib]
        g[(np.abs(bb) > bearings[-1]) | (rr > r[-1]) | (rr < r[0])] = np.nan
        out.append(g)
    return out, res.true_positions


def smooth_db(g, n=4):
    """dB re the median, averaged over n x n metres to beat the speckle of sparse facets."""
    a = np.nan_to_num(g, nan=np.nanmedian(g))
    k = np.ones(n) / n
    a = np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), 0, a)
    a = np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), 1, a)
    return 10 * np.log10(np.maximum(a / np.median(a), 1e-6))


def displacement(a, b, limit=25):
    """(dx, dy) in metres by which pattern b is shifted from pattern a (cross-correlation peak)."""
    box = (slice(60, 190), slice(60, 180))                              # 130 m x 120 m well inside the sector
    fa, fb = a[box] - a[box].mean(), b[box] - b[box].mean()
    c = np.fft.ifft2(np.fft.fft2(fb) * np.conj(np.fft.fft2(fa))).real
    c = np.fft.fftshift(c) / np.sqrt((fa**2).sum() * (fb**2).sum())
    mid = np.array(c.shape) // 2
    win = c[mid[0] - limit:mid[0] + limit + 1, mid[1] - limit:mid[1] + limit + 1]
    i, j = np.unravel_index(win.argmax(), win.shape)
    return float(i - limit), float(j - limit), float(win.max())


def main():
    regular = [{"amplitude": 0.6, "wavelength": 40.0, "direction_deg": 15.0}, {"amplitude": 0.15, "wavelength": 6.0, "direction_deg": 50.0}]

    def flat(s):
        s.pop("spectrum", None), s.pop("isotropic_db", None)

    def two_trains(s):
        flat(s), s.update(waves=regular)

    cases = {"flat water": flat, "two regular wave trains, tilted facets only\n(the first version of the scene)": two_trains,
             "irregular sea, tilted facets only": lambda s: s.pop("isotropic_db", None),
             "irregular sea with small-scale roughness (as shipped)": lambda s: None}
    fig, axes = plt.subplots(2, 4, figsize=(22, 10.5), constrained_layout=True)
    ext = [GY[0], GY[-1], GX[0], GX[-1]]
    out = {}
    flat_img = None
    for ax, (label, change) in zip(axes[0], cases.items()):
        g = images(change)[0][0]
        if flat_img is None:
            flat_img = g
        d = smooth_db(g)
        ax.imshow(d, extent=ext, origin="lower", cmap="gray", vmin=-10, vmax=8, aspect="equal")
        ax.set_title(label, loc="left", fontsize=10)
        ax.set_xlabel("y, to starboard (m)")
        # Band depth: the image over the flat-water image (same facets), spread of its 4 m averages.
        rel = smooth_db(g / flat_img)[60:190, 60:180]
        out[label.split("\n")[0]] = {"band_depth_db_p90_minus_p10": float(np.percentile(rel, 90) - np.percentile(rel, 10))}
    axes[0, 0].set_ylabel("x, ahead (m), world frame")

    # Part 2: two pings DT apart, travelling and frozen.
    # The fall-off with range and the beam pattern are fixed to the sonar, not the world; dividing by
    # the flat-water image of the same ping removes them and leaves what the waves do. Averaging over
    # 12 m keeps the long waves, which travel far enough in a second to measure.
    flat2 = images(flat, pings=2)[0]
    rows = []
    for k, (label, frozen) in enumerate((("travelling (as shipped)", False), ("frozen", True))):
        imgs, own = images(lambda s, f=frozen: s.update(frozen=f, velocity=[0.0, 0.0, 0.0]), pings=2)
        a, b = smooth_db(imgs[0]), smooth_db(imgs[1])
        dx, dy, peak = displacement(smooth_db(imgs[0] / flat2[0], 12), smooth_db(imgs[1] / flat2[1], 12))
        advance = float(own[1][0] - own[0][0])
        rows.append((label, dx, dy, peak, advance))
        out[f"waves {label.split(' ')[0]}"] = {"world_frame_shift_m": [dx, dy], "sonar_frame_shift_m": [dx - advance, dy],
                                               "correlation": peak, "sonar_advance_m": advance, "seconds": DT}
        for ax, d, t in zip(axes[1, 2 * k:2 * k + 2], (a, b), (0.0, DT)):
            ax.imshow(d, extent=ext, origin="lower", cmap="gray", vmin=-10, vmax=8, aspect="equal")
            ax.axhline(170.0, color="#eb6834", lw=0.8)
            ax.axvline(0.0, color="#eb6834", lw=0.8)
            ax.set_title(f"waves {label}, t = {t:.0f} s, world frame"
                         + (f"\npattern moved {dx:+.0f} m ahead, {dy:+.0f} m to starboard; sonar advanced {advance:.0f} m" if t else ""),
                         loc="left", fontsize=10)
            ax.set_xlabel("y, to starboard (m)")
    axes[1, 0].set_ylabel("x, ahead (m), world frame")
    fig.suptitle("fls_120k_rough_surface, surface only. Top: what makes the bands. Bottom: two pings 1 s apart on a world-fixed grid "
                 "(orange lines are fixed in the world)")
    fig.savefig(ROOT / "docs" / "figures" / "rough_surface_bands.png", dpi=95)
    (ROOT / "docs" / "figures" / "rough_surface_bands.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
