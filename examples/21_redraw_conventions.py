"""Redraw example 21's three-convention figure from its saved per-beam stack.

The four-beam run takes ten minutes; redrawing how its beams reach the screen
should not.  ``21_long_range_300m.py`` saves the polar image of every
elevation beam to ``figures/21_beams_<FAR>m.pt``; this reads it back and
draws the summed, max-over-beams and single-beam pictures under one colour
scale, in seconds.

    python 21_redraw_conventions.py            # figures/21_beams_300m.pt
    HYDROPT_FAR=90 python 21_redraw_conventions.py
"""

from __future__ import annotations

import importlib.util
import math
import os
from pathlib import Path

import torch

from _common import FIGURE_DIR, save, setup


def _ex21():
    path = Path(__file__).resolve().parent / "21_long_range_300m.py"
    spec = importlib.util.spec_from_file_location("_ex21", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    setup()
    import matplotlib.pyplot as plt

    ex = _ex21()
    far = float(os.environ.get("HYDROPT_FAR", 300.0))
    saved = torch.load(FIGURE_DIR / f"21_beams_{far:.0f}m.pt")
    stack, tilts, k_boat = saved["stack"], saved["tilts"], saved["k_boat"]
    bearings, grid = saved["bearings"], saved["grid"]
    rng = grid * ex.C / 2.0

    # The same chain the example applies before it draws: absolute level,
    # receiver noise, median range gain, and a Cartesian resample.
    from hydropt import (add_receiver_noise, beam_noise_power, beam_power_scale,
                         calibrate, line_array_directivity_db, shading_window)
    shading = shading_window(ex.N_RX, "hamming")
    scale = beam_power_scale(shading, ex.PULSE_S)
    # As the example derives them: the array's directivity index, the pulse's
    # bandwidth, and the wind-driven ambient in one beam and cell.
    noise = float(beam_noise_power(torch.tensor([ex.FREQ_KHZ]),
                                   bandwidth_hz=1.0 / ex.PULSE_S,
                                   directivity_db=line_array_directivity_db(ex.N_RX),
                                   wind_speed=ex.WIND))
    ex15 = ex._ex15()
    x_range = (-0.03 * far, 1.02 * far)
    span_y = far * math.sin(math.radians(ex.SECTOR_DEG)) * 1.02
    pixel_m = (x_range[1] - x_range[0]) / 299.0
    to_cart = lambda img: ex15.to_cartesian(img, bearings, grid, n_x=300, n_y=300,
                                            x_range=x_range, y_range=(-span_y, span_y))

    # HYDROPT_REDRAW=beams draws every elevation beam on its own panel instead
    # -- the attribution view, where the seabed's return should sit in the
    # lower beams, the surface's in the upper, and a hull's seabed-image ghost
    # in the beam below the hull's own.
    if os.environ.get("HYDROPT_REDRAW", "conventions") == "beams":
        conventions = {f"beam {k}: {-t:+.1f} deg" + (" (the boat's)" if k == k_boat else ""):
                       stack[k] for k, t in enumerate(tilts)}
        out_name = f"21_elevation_beams_{far:.0f}m.png"
    else:
        conventions = {
            "summed over beams": stack.sum(dim=0),
            "max over beams": stack.max(dim=0).values,
            f"one beam, {-tilts[k_boat]:+.1f} deg (the boat's)": stack[k_boat],
        }
        out_name = f"21_display_conventions_{far:.0f}m.png"
    panels, extent = {}, None
    with torch.no_grad():
        for name, img in conventions.items():
            sig = calibrate(img, ex.SOURCE_LEVEL_DB, beam_scale=scale)
            noisy = add_receiver_noise(sig, noise,
                                       generator=torch.Generator().manual_seed(ex.SEED + 2))
            shown, _ = ex.display(noisy, rng, pixel_m=pixel_m)
            cart, gx, gy = to_cart(shown)
            panels[name] = 10.0 * torch.log10(cart.clamp_min(1e-30))
            extent = (float(gx[0]), float(gx[-1]), float(gy[0]), float(gy[-1]))
    top = max(float(c.max()) for c in panels.values())
    b = math.radians(ex.BOAT_BEARING_DEG)
    tx, ty = ex.BOAT_RANGE * math.cos(b), ex.BOAT_RANGE * math.sin(b)

    fig, axes = plt.subplots(1, len(panels), figsize=(5.5 * len(panels), 6.2))
    for ax, (name, cart) in zip(axes, panels.items()):
        im = ax.imshow(cart.numpy(), origin="lower", cmap="inferno", extent=extent,
                       vmin=6.0, vmax=top, aspect="equal")
        ax.add_patch(plt.Circle((tx, ty), 12.0, fill=False, color="white", lw=1.2))
        ax.set_title(name, fontsize=11)
        ax.set_xlabel("forward (m)")
    axes[0].set_ylabel("across (m)")
    fig.colorbar(im, ax=axes, shrink=0.8,
                 label="dB over the background at that range, floored at +6")
    tail = ("each on its own panel" if os.environ.get("HYDROPT_REDRAW") == "beams"
            else "three ways to put them on one screen")
    fig.suptitle(f"{len(tilts)} receive elevation beams of "
                 f"{ex.beam_3db_deg(ex.N_RX_ELEV):.2f} deg: " + tail, fontsize=12)
    save(fig, out_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
