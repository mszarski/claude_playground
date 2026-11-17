#!/usr/bin/env python3
"""Boundary models side by side: backscattering strength against grazing angle for each
complete sediment in configs/sediments.yaml (every roughness model) and for the surface laws,
the bottom reflection loss of every named sediment, and the bubble loss per surface bounce
against wind speed.

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
    # Flat-interface reflection loss of every named sediment (it does not depend on frequency:
    # the loss parameter is per wavelength). The reflection-only rows appear here and nowhere else.
    header("Bottom reflection loss, dB: -20 log10 |R| of the fluid half-space (bottom.reflection)   [rho, nu, delta]")
    for name, entry in table.items():
        par = bottom.sediment_reflection(name)
        mark = "*" if entry.get("unconfirmed_fields") or entry.get("status") != "confirmed" else ""
        print(row(f"{name}{mark}", -20 * np.log10(np.abs(bottom.reflection(g, **par))))
              + f"   [{par['rho']:.3f}, {par['nu']:.4f}, {par['delta']:.5f}]")
    print("  (* an unconfirmed entry or field: see configs/sediments.yaml)")
    for f_hz in args.frequency:
        for name, entry in table.items():
            if bottom.is_reflection_only(name):
                continue                                  # no roughness or volume parameters: no scattering strength
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

    # Published measurements, as numbers in a paper's text (not read off figures). One set was
    # found in an open source: rock outcrops at 100 kHz. Olson, Lyons and Saebo, "Measurements of
    # high-frequency acoustic scattering from glacially-eroded rock outcrops", arXiv:1601.06078:
    # abraded (smooth) surfaces "fall between -33 dB and -26 dB at a reference angle of 20 deg
    # grazing", most near the lower end; plucked (very rough) surfaces between -30 and -20 dB;
    # earlier measurements on rock "-15 to -22 dB at 20 deg". Their Table III gives the abraded
    # surface's measured roughness spectrum, Phi(u) = phi2 / u^gamma2 with u in cycles/m:
    # phi2 = 4.847e-8 m^(4 - gamma2), gamma2 = 2.73. With K = 2 pi u and the same variance,
    # W(K) = Phi(K / 2 pi) / (2 pi)^2, so w2 = phi2 (2 pi)^(gamma2 - 2) (our conversion; the
    # paper's normalisation of Phi is assumed to be variance per unit area of the u plane).
    w2_rock = 4.847e-8 * (2.0 * np.pi) ** (2.73 - 2.0)
    header("Rock at 100 kHz, dB: the fluid model with the table's 'rock' and the roughness measured in arXiv:1601.06078 (abraded)")
    print(row("fluid model (no volume term)", bottom.backscattering_strength_db(
        g, 100e3, c, roughness_model="small_slope", w2=w2_rock, gamma2=2.73, sigma2=0.0, **bottom.sediment_reflection("rock"))))
    print("  measured there at 20 deg: -33 to -26 (abraded), -30 to -20 (plucked). The fluid model is below the measured\n"
          "  range: rock carries shear waves, which this model does not have (the paper compares with an elastic model),\n"
          "  and the level rests on the spectrum normalisation assumed above. Not a validation of a fluid 'rock' row.")

    try:
        import arlpy.uwa as oracle
    except Exception:                                      # noqa: BLE001 - optional oracle
        oracle = None
    for f_hz in args.frequency:
        header(f"Bubble loss per surface bounce, dB, {f_hz / 1e3:g} kHz (resonant radius {surface.resonant_radius(f_hz) * 1e6:.0f} um)")
        for u in args.wind:
            print(row(f"hall_novarini, {u:g} m/s", surface.bubble_loss_db(g, u, f_hz, c)))
            if oracle is not None:
                # arlpy takes the angle from the vertical and returns an amplitude factor.
                print(row(f"arlpy oracle, {u:g} m/s", [-20 * np.log10(float(oracle.bubble_surface_loss(u, f_hz, np.pi / 2 - a))) for a in np.maximum(g, surface.MIN_GRAZING)]))
    if oracle is None:
        print("\n  (arlpy is not installed: no oracle column for the bubble loss)")
    print("\nNo published curve values are compared: see this script's docstring.")


if __name__ == "__main__":
    main()
