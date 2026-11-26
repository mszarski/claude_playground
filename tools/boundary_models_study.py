#!/usr/bin/env python3
"""Boundary models side by side: the family table of every named sediment in
configs/sediments.yaml (backscattering strength at 10, 20 and 45 degrees for 6, 100 and 300 kHz,
with the empirical type law and the measured values found in open papers), backscattering
strength against grazing angle for chosen sediments (every roughness model) and for the surface
laws, the bottom reflection loss of every named sediment, and the bubble loss per surface
bounce against wind speed.

    PYTHONPATH=python python3 tools/boundary_models_study.py [--frequency 30e3 100e3] [--wind 5 10]
        [--sediment medium_sand very_fine_silt | --sediment all] [--family-only]

Comparison with measurements: the family table ends with the measured levels that open papers
print as numbers in their text (docs/REFERENCES.md lists them); values are not read off
figures. For the bubble loss the outside numbers are those of the BSD library arlpy
(`arlpy.uwa.bubble_surface_loss`, which says it implements the "APL model (1994)"), called
here as an oracle if arlpy is installed.
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


FAMILY_ANGLES = (10.0, 20.0, 45.0)
FAMILY_FREQUENCIES = (6e3, 100e3, 300e3)

# Measured backscattering strength printed as numbers in open papers (docs/REFERENCES.md):
# (description, grazing angle in degrees, frequency range in kHz, level or range in dB, source).
MEASURED = (
    ("medium sand", 45, "70-450", "about -25", "Fezzani et al. 2025, Sec. 3.3"),
    ("sandy mud; muddy very fine sand", 45, "100-450", "about -22", "Fezzani et al. 2025, Sec. 3.3"),
    ("rock", 45, "80-450", "about -13", "Fezzani et al. 2025, Sec. 3.3"),
    ("maerl; pebbles, boulders", 45, "200-450", "-13 to -11", "Fezzani et al. 2025, Secs. 3.3-3.4"),
    ("very fine sand; sand", 45, "150", "-25; -28", "Trzcinska et al. 2021 as quoted by Fezzani et al. 2025, Sec. 3.4"),
    ("rock, glacially abraded (smooth)", 20, "100", "-33 to -26", "Olson, Lyons and Saebo, arXiv:1601.06078"),
    ("rock, glacially plucked (rough)", 20, "100", "-30 to -20", "Olson, Lyons and Saebo, arXiv:1601.06078"),
    ("rock, earlier measurements", 20, "24-100", "-22 to -15", "as summarised in arXiv:1601.06078, Sec. I"),
    ("rock, granodiorite outcrops", 20, "200", "-35 to -30", "Gruber 2019 (NPS thesis), Ch. IV"),
    ("rock, granodiorite outcrops", 40, "200", "-30 to -27", "Gruber 2019 (NPS thesis), abstract"),
)


def family_order(table) -> list:
    """Rock, pebbles, then the sediments from coarse to fine by mean grain size."""
    names = [n for n in table if n != "placeholder_sand"]
    return sorted(names, key=lambda n: (-99.0 if n == "rock" else -98.0 if n == "pebbles"
                                        else float(table[n].get("mean_grain_size_phi", 0.0))))


def family_table(c: float = 1500.0, model: str = bottom.DEFAULT_ROUGHNESS_MODEL) -> None:
    """Every named row at three angles and three frequencies, with the type law and measurements."""
    table = bottom.sediment_table()
    g = np.radians(FAMILY_ANGLES)
    title = "/".join(f"{a:g}" for a in FAMILY_ANGLES)
    print(f"\nFamily table: backscattering strength, dB, at {title} degrees grazing (roughness model: {model})")
    print(f"  {'sediment':<22s}{'phi':>5s}" + "".join(f"{f / 1e3:>17g} kHz    " for f in FAMILY_FREQUENCIES))

    def cells(fn):
        return "".join("   " + " /".join(f"{v:6.1f}" for v in np.asarray(fn(f), dtype=float)) for f in FAMILY_FREQUENCIES)
    for name in family_order(table):
        entry = table[name]
        extra = {} if "empirical_backscatter" in entry else {"roughness_model": model}
        phi = entry.get("mean_grain_size_phi")
        tag = " (Lambert)" if "empirical_backscatter" in entry else ""
        print(f"  {name + tag:<22s}{'-' if phi is None else format(phi, 'g'):>5s}"
              + cells(lambda f: bottom.named_backscattering_strength_db(name, g, f, c, **extra)))
    for bt in surface.BOTTOM_TYPES:
        print(f"  {'type law: ' + bt:<22s}{'':>5s}" + cells(lambda f: surface.searay_bottom_db(g, f, bt)))
    print("  (type law: SEARAY, as printed in arXiv:2210.06535; no stated range. Lambert rows have no frequency dependence.)")
    print("\n  Measured, as numbers printed in open papers (not read off figures):")
    for what, angle, band, level, source in MEASURED:
        print(f"    {what:<34s}{angle:>3d} deg  {band:>8s} kHz  {level:>11s} dB   {source}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frequency", type=float, nargs="+", default=[30e3, 100e3], help="Hz")
    ap.add_argument("--wind", type=float, nargs="+", default=[5.0, 10.0], help="m/s")
    ap.add_argument("--sound-speed", type=float, default=1500.0)
    ap.add_argument("--sediment", nargs="+", default=["medium_sand", "very_fine_silt", "placeholder_sand"],
                    help="sediments for the per-model tables against angle; 'all' for every complete row")
    ap.add_argument("--family-only", action="store_true", help="print the family table and stop")
    args = ap.parse_args()
    g = np.radians(ANGLES)
    c = args.sound_speed
    warnings.simplefilter("ignore")                       # the range warnings are the subject of the notes below

    table = bottom.sediment_table()
    family_table(c)
    family_table(c, "combined")
    if args.family_only:
        return
    detailed = list(table) if args.sediment == ["all"] else args.sediment
    # Flat-interface reflection loss of every named sediment (it does not depend on frequency:
    # the loss parameter is per wavelength).
    header("Bottom reflection loss, dB: -20 log10 |R| of the fluid half-space (bottom.reflection)   [rho, nu, delta]")
    for name, entry in table.items():
        par = bottom.sediment_reflection(name)
        mark = "*" if entry.get("unconfirmed_fields") or entry.get("status") != "confirmed" else ""
        print(row(f"{name}{mark}", -20 * np.log10(np.abs(bottom.reflection(g, **par))))
              + f"   [{par['rho']:.3f}, {par['nu']:.4f}, {par['delta']:.5f}]")
    print("  (* an unconfirmed entry or field: see configs/sediments.yaml)")
    for f_hz in args.frequency:
        for name, entry in table.items():
            if bottom.is_reflection_only(name) or name not in detailed:
                continue                                  # rock, pebbles: no fluid scattering model (family table)
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
            # The published bubble-layer and facet terms (Gallaudet 2001, eqs. (8), (10)); stated range 12-70 kHz.
            print(row(f"bubble layer, dahl_1997, {u:g}", surface.bubble_layer_surface_db(g, u, f_hz, c, "dahl_1997")))
            print(row(f"bubble layer, apl_1994, {u:g}", surface.bubble_layer_surface_db(g, u, f_hz, c, "apl_1994")))
            print(row(f"bubble layer, hall_nov., {u:g}", surface.bubble_layer_surface_db(g, u, f_hz, c, "hall_novarini")))
        print(row("bubble saturation level", 10 * np.log10(surface.BUBBLE_RERADIATION_DAMPING * np.sin(np.maximum(g, surface.MIN_GRAZING))
                                                          / (8 * np.pi * surface.bubble_damping_constant(f_hz)))))

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
            print(row(f"hall_novarini_layer, {u:g} m/s", surface.bubble_layer_loss_db(g, u, f_hz, c)))
            print(row(f"hall_novarini_screened, {u:g} m/s", surface.bubble_screened_loss_db(g, u, f_hz, c)))
            print(row(f"apl_1994 (adopted), {u:g} m/s", surface.apl_1994_bubble_loss_db(g, u, f_hz, c)))
            if oracle is not None:
                # arlpy takes the angle from the vertical and returns an amplitude factor.
                print(row(f"arlpy oracle, {u:g} m/s", [-20 * np.log10(float(oracle.bubble_surface_loss(u, f_hz, np.pi / 2 - a))) for a in np.maximum(g, surface.MIN_GRAZING)]))
    if oracle is None:
        print("\n  (arlpy is not installed: no oracle column for the bubble loss)")
    print("\nMeasured values are compared in the family table only: see this script's docstring.")


if __name__ == "__main__":
    main()
