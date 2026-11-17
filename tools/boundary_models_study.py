#!/usr/bin/env python3
"""Boundary models side by side: backscattering strength against grazing angle for each
sediment in configs/sediments.yaml (every roughness model) and for the surface laws, and the
bubble loss per surface bounce against wind speed.

    PYTHONPATH=python python3 tools/boundary_models_study.py [--frequency 30e3 100e3] [--wind 5 10]

Comparison with published curves: none is printed. No paper that could be opened gives
backscattering strength values in its text or tables for a stated parameter set, and values
are not read off figures. The one outside number set shown is the bubble loss computed by the
BSD library arlpy (`arlpy.uwa.bubble_surface_loss`, which says it implements the "APL model
(1994)"), called here as an oracle if arlpy is installed; its formula is not in this repository.
"""
from __future__ import annotations

import argparse
import warnings

import numpy as np

from gpusonar import acoustics
from gpusonar.sim import bottom, surface

ANGLES = np.array([2.0, 5.0, 10.0, 20.0, 30.0, 45.0, 60.0, 75.0, 85.0, 90.0])


def row(label: str, values) -> str:
    return f"  {label:<28s}" + "".join(f"{v:8.1f}" for v in np.asarray(values, dtype=float))


def header(title: str) -> None:
    print(f"\n{title}\n  {'grazing angle, deg':<28s}" + "".join(f"{a:8.0f}" for a in ANGLES))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frequency", type=float, nargs="+", default=[30e3, 100e3], help="Hz")
    ap.add_argument("--wind", type=float, nargs="+", default=[5.0, 10.0], help="m/s")
    ap.add_argument("--sound-speed", type=float, default=1500.0)
    args = ap.parse_args()
    g = np.radians(ANGLES)
    c = args.sound_speed
    warnings.simplefilter("ignore")                       # the range warnings are the subject of the notes below

    table = bottom.sediment_table()
    for f_hz in args.frequency:
        for name, entry in table.items():
            par = bottom.sediment(name)
            slope, kc = bottom.large_scale_slope(f_hz, c, **par)
            header(f"Bottom backscattering strength, dB: sediment '{name}' ({entry.get('status', 'unconfirmed')}), {f_hz / 1e3:g} kHz"
                   f"   [large-scale rms slope {np.degrees(np.arctan(slope)):.1f} deg, split at {2 * np.pi / kc * 100:.0f} cm]")
            for model in bottom.ROUGHNESS_MODELS:
                print(row(f"roughness: {model}", 10 * np.log10(np.maximum(
                    bottom.roughness_strength(g, g, f_hz, c, roughness_model=model, **par), 1e-30))))
            print(row("sediment volume", 10 * np.log10(np.maximum(bottom.volume_strength(g, g, f_hz, c, **par), 1e-30))))
            print(row("total (default model)", bottom.backscattering_strength_db(g, f_hz, c, **par)))
            print(row("total (combined)", bottom.backscattering_strength_db(g, f_hz, c, roughness_model="combined", **par)))
            print(row("Lambert, same level at 20", bottom.equivalent_lambert_mu_db(np.radians(20.0), f_hz, c, **par)
                      + 20 * np.log10(np.sin(g))))
        header(f"Empirical bottom law by type (SEARAY, as printed in arXiv:2210.06535), dB, {f_hz / 1e3:g} kHz")
        for name in surface.BOTTOM_TYPES:
            print(row(name, surface.searay_bottom_db(g, f_hz, name)))

    for f_hz in args.frequency:
        header(f"Surface backscattering strength, dB, {f_hz / 1e3:g} kHz (Kraus law held above 60 deg; "
               "Chapman-Harris fitted 0.4-6.4 kHz, below 40 deg)")
        for u in args.wind:
            print(row(f"Chapman-Harris, {u:g} m/s", acoustics.chapman_harris_db(g, u, f_hz)))
            print(row(f"Kraus empirical, {u:g} m/s", surface.kraus_surface_db(g, u, f_hz)))

    try:
        import arlpy.uwa as oracle
    except Exception:                                      # noqa: BLE001 - optional oracle
        oracle = None
    for f_hz in args.frequency:
        header(f"Bubble loss per surface bounce, dB, {f_hz / 1e3:g} kHz (resonant radius {surface.resonant_radius(f_hz) * 1e6:.0f} um)")
        for u in args.wind:
            print(row(f"hall_novarini, {u:g} m/s", surface.bubble_loss_db(g, u, f_hz, c)))
            print(row(f"hall_novarini_layer, {u:g} m/s", surface.bubble_layer_loss_db(g, u, f_hz, c)))
            print(row(f"hall_novarini_screened, {u:g} m/s", surface.bubble_screened_loss_db(g, u, f_hz, c)))
            print(row(f"apl_1994 (adopted), {u:g} m/s", surface.apl_1994_bubble_loss_db(g, u, f_hz, c)))
            if oracle is not None:
                # arlpy takes the angle from the vertical and returns an amplitude factor.
                print(row(f"arlpy oracle, {u:g} m/s", [-20 * np.log10(float(oracle.bubble_surface_loss(u, f_hz, np.pi / 2 - a))) for a in np.maximum(g, surface.MIN_GRAZING)]))
    if oracle is None:
        print("\n  (arlpy is not installed: no oracle column for the bubble loss)")
    print("\nNo published curve values are compared: see this script's docstring.")


if __name__ == "__main__":
    main()
