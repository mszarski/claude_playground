#!/usr/bin/env python3
"""What do two or more stacked rows on the Popoto long-range array buy, in numbers?

The array is one horizontal line of 32 hydrophones (4-6 kHz). This study works out, for rows
stacked at different heights on the vehicle:

  accuracy     the Cramer-Rao bound on elevation against the signal-to-noise ratio after one
               row's beamforming and pulse compression, checked by Monte Carlo with the
               estimator the live chain uses (steer the rows over elevation, take the peak),
               and what it is in metres of depth at 1, 6 and 12 km;
  ambiguity    the elevation sector without a second lobe against row separation, the height
               of the second lobe over the band inside the searched sector, how often noise
               makes the estimator pick it, and what a third row at an unequal spacing does;
  arrivals     the elevations that actually arrive in shallow and deep water;
  cone         the bearing error a single line makes on an arrival from above or below
               (it measures the cone angle), which a known elevation removes;
  multipath    direct, surface-bounce and bottom-bounce arrivals of one target: their
               elevations, how far apart they are in range, when the chain can separate them
               and what depth each gives;
  sensitivity  what roll error, row position error, phase mismatch and board clock error do;
  unchanged    what stacked rows do not improve.

Every number in docs/DESIGN.md Sec. 3 "stacked rows" is printed here and written to
docs/design/popoto_elevation_study.json. The hull the rows must fit on is NOT confirmed:
`--hull` (default 0.533 m, a 21 inch vehicle) only bounds the candidates.

Usage: tools/popoto_elevation_study.py [--hull 0.533] [--trials 4000] [--no-write]
CPU only, about a minute.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from gpusonar import elevation as el  # noqa: E402
from gpusonar.sim.noise import wenz_spectrum_db  # noqa: E402
from gpusonar.sonar import SonarModel  # noqa: E402
from gpusonar.sonar_equation import SonarEquation  # noqa: E402

C = 1500.0
F_LOW, F_HIGH = 4000.0, 6000.0
N_ROW = 32
SECTOR = (-30.0, 30.0, 0.25)


def spectrum(heights, outs, snr_row_db, true_deg, rng, trials, n_f=32):
    """Monte Carlo of the chain's estimator on its sufficient statistic: each row's beam at
    the detection, over the band (flat spectrum, one unknown complex amplitude, range known),
    in white noise at `snr_row_db` per row after pulse compression. Returns the estimates (deg)."""
    h, o = np.asarray(heights, dtype=float), np.asarray(outs, dtype=float)
    f = np.linspace(F_LOW, F_HIGH, n_f)
    grid = np.radians(np.arange(SECTOR[0], SECTOR[1] + 0.5 * SECTOR[2], SECTOR[2]))
    lead = (h[:, None] * np.sin(grid)[None, :] + o[:, None] * np.cos(grid)[None, :]) / C          # [K, E]
    steer = np.exp(-2j * np.pi * f[None, :, None] * lead[:, None, :])                             # [K, F, E]
    t = np.radians(true_deg)
    true_lead = (h * np.sin(t) + o * np.cos(t)) / C
    signal = np.exp(2j * np.pi * f[None, :] * true_lead[:, None])                                 # [K, F]
    # Signal power per bin 1; noise per bin so that the row's SNR summed over the band is snr_row.
    sigma = np.sqrt(n_f / 10.0 ** (snr_row_db / 10.0))
    out = np.empty(trials)
    for i in range(trials):
        y = signal + sigma * (rng.standard_normal(signal.shape) + 1j * rng.standard_normal(signal.shape)) / np.sqrt(2.0)
        p = np.abs(np.einsum("kf,kfe->e", y, steer)) ** 2
        e = int(np.argmax(p))
        frac = 0.0
        if 0 < e < len(grid) - 1:
            den = p[e - 1] - 2.0 * p[e] + p[e + 1]
            frac = 0.5 * (p[e - 1] - p[e + 1]) / den if den < 0.0 else 0.0
        out[i] = np.degrees(grid[e]) + frac * SECTOR[2]
    return out


def pattern(heights, outs, true_deg, n_f=64):
    """Noise-free band-averaged elevation response (power, 1 at the arrival) over the sector."""
    h, o = np.asarray(heights, dtype=float), np.asarray(outs, dtype=float)
    f = np.linspace(F_LOW, F_HIGH, n_f)
    grid = np.radians(np.arange(SECTOR[0], SECTOR[1] + 0.5 * SECTOR[2], SECTOR[2]))
    t = np.radians(true_deg)
    lag = (h[:, None] * (np.sin(grid)[None, :] - np.sin(t)) + o[:, None] * (np.cos(grid)[None, :] - np.cos(t))) / C
    p = np.abs(np.exp(-2j * np.pi * f[None, :, None] * lag[:, None, :]).sum(axis=(0, 1))) ** 2
    return np.degrees(grid), p / (len(h) * n_f) ** 2


def second_lobe_db(heights, outs, true_deg):
    """Highest point of the band-averaged response outside the main lobe, inside the sector, dB."""
    g, p = pattern(heights, outs, true_deg)
    e = int(np.argmax(p))
    lo = e
    while lo > 0 and p[lo - 1] < p[lo]:
        lo -= 1
    hi = e
    while hi < len(p) - 1 and p[hi + 1] < p[hi]:
        hi += 1
    rest = np.concatenate([p[:lo], p[hi + 1:]])
    return float(max(10.0 * np.log10(max(rest.max(), 1e-30)), -99.0)) if len(rest) else -99.0     # -99: none


def link_snr(ranges, target_db):
    """Output SNR (dB) of ONE row of the long-range array: the budget of docs/DESIGN.md Sec. 5."""
    sonar = SonarModel.load("popoto_long_range")
    noise = float(wenz_spectrum_db(np.array([sonar.center_frequency]), 5.0, 0.5)[0])
    eq = SonarEquation.from_sonar(sonar, noise, target_strength_db=target_db, temperature=12.0, depth=50.0,
                                  absorption_model="ainslie_mccolm")
    return [float(eq.snr(r)) for r in ranges]


def images(z_r, z_t, r, depth):
    """Isovelocity image arrivals at horizontal range r: (name, elevation deg up positive,
    one-way path m, straight-line depth the arrival points at)."""
    out = []
    for name, z_img in (("direct", z_t), ("surface", -z_t), ("bottom", 2.0 * depth - z_t),
                        ("surface-bottom", -(2.0 * depth - z_t)), ("bottom-surface", 2.0 * depth + z_t)):
        dz = z_img - z_r
        out.append((name, float(np.degrees(np.arctan2(-dz, r))), float(np.hypot(r, dz)), float(z_img)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hull", type=float, default=0.533, help="hull diameter the rows must fit on, m (unconfirmed)")
    ap.add_argument("--trials", type=int, default=4000)
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args()
    rng = np.random.default_rng(7)
    lam = C / (0.5 * (F_LOW + F_HIGH))
    radius = 0.5 * args.hull
    # Rows on the flank of a round hull between 45 degrees below and above its widest point.
    up_max = float(np.round(radius * np.sin(np.radians(45.0)), 2))
    out_of = lambda up: float(np.sqrt(radius**2 - up**2) - np.sqrt(radius**2 - up_max**2))       # noqa: E731
    R = {"hull_diameter_m": args.hull, "largest_row_offset_m": up_max, "wavelength_m": lam, "band_hz": [F_LOW, F_HIGH],
         "sector_deg": list(SECTOR)}
    print(f"Band {F_LOW / 1e3:.0f}-{F_HIGH / 1e3:.0f} kHz, wavelength {lam * 100:.0f} cm at the centre; "
          f"hull {args.hull:.3f} m (unconfirmed): rows within +-{up_max:.2f} m of the centreline.\n")

    # --- accuracy ---------------------------------------------------------------------------------
    geometries = {}
    for d in (0.075, 0.15, 0.25, 2.0 * up_max, 0.50, 0.75, 1.00):
        geometries[f"2 rows, {d:.2f} m apart"] = ([-d / 2, d / 2], [0.0, 0.0])
    mid = -0.05
    geometries[f"3 rows at {-up_max:+.2f}, {mid:+.2f}, {up_max:+.2f} m"] = ([-up_max, mid, up_max], [0.0, out_of(mid), 0.0])
    geometries[f"3 rows at {-up_max:+.2f}, +0.00, {up_max:+.2f} m (equal)"] = ([-up_max, 0.0, up_max], [0.0, out_of(0.0), 0.0])
    geometries["4 rows, 0.125 m apart"] = ([-0.1875, -0.0625, 0.0625, 0.1875], [0.0] * 4)
    snrs = (13.0, 20.0, 30.0)
    print("ACCURACY: one-sigma elevation (deg) against the SNR of one row after beamforming and pulse compression.")
    print("  bound = 1 / (k_rms sqrt(2 SNR sum_rows (h - mean h)^2)); 'MC' = Monte Carlo of the chain's estimator at 20 dB, over the bound.")
    print(f"  {'geometry':44s} " + " ".join(f"{s:>5.0f} dB" for s in snrs) + "   MC/bound  fits hull")
    R["accuracy"] = {}
    for name, (h, o) in geometries.items():
        heights = np.repeat(h, N_ROW)
        sig = [float(np.degrees(el.sin_elevation_sigma(heights, 10.0 ** (s / 10.0) / N_ROW, F_LOW, F_HIGH, C))) for s in snrs]
        est = spectrum(h, o, 20.0, 0.0, rng, args.trials)
        near = est[np.abs(est) < 4.0 * max(sig[1], 0.5)]            # the main lobe; the rest are counted under ambiguity
        ratio = float(near.std() / sig[1])
        fits = max(abs(v) for v in h) <= up_max + 1e-9
        R["accuracy"][name] = {"heights_m": list(h), "outs_m": list(o), "sigma_deg": dict(zip(map(str, snrs), sig)),
                               "monte_carlo_over_bound_at_20_db": ratio, "mean_error_deg_at_20_db": float(near.mean()),
                               "fits_hull": bool(fits)}
        print(f"  {name:44s} " + " ".join(f"{s:8.2f}" for s in sig) + f"   {ratio:7.2f}   {'yes' if fits else 'no'}")

    ranges = (1000.0, 6000.0, 12000.0)
    R["link_snr_db"] = {}
    print("\nDEPTH: one-sigma depth error = range x sigma (m), at the SNR the link budget gives one row (spherical spreading).")
    print("  SNR above 40 dB is taken as 40 dB: closer in, reverberation and calibration set the limit, not noise.")
    R["depth_sigma_m"] = {}
    two, three = f"2 rows, {2.0 * up_max:.2f} m apart", f"3 rows at {-up_max:+.2f}, +0.00, {up_max:+.2f} m (equal)"
    chosen = [two, three, "2 rows, 1.00 m apart"]
    R["chosen"] = {"two_rows": two, "three_rows": three, "wide": chosen[2]}
    for label, ts in (("ship, +25 dB", 25.0), ("small target, +5 dB", 5.0)):
        snr = link_snr(ranges, ts)
        R["link_snr_db"][label] = dict(zip(map(str, ranges), snr))
        print(f"  {label}: SNR " + ", ".join(f"{s:.0f} dB at {r / 1e3:.0f} km" for s, r in zip(snr, ranges)))
        for name in chosen:
            h, _ = geometries[name]
            row = []
            for r, s in zip(ranges, snr):
                if s < 10.0:
                    row.append(None)                        # not detected: no elevation at all
                    continue
                sig = el.sin_elevation_sigma(np.repeat(h, N_ROW), 10.0 ** (min(s, 40.0) / 10.0) / N_ROW, F_LOW, F_HIGH, C)
                row.append(float(r * sig))
            R["depth_sigma_m"][f"{label}; {name}"] = dict(zip(map(str, ranges), row))
            print(f"    {name:44s} " + "  ".join(("not detected" if v is None else f"{v:7.1f} m") + f" at {r / 1e3:.0f} km"
                                                  for v, r in zip(row, ranges)))

    # --- ambiguity --------------------------------------------------------------------------------
    print("\nAMBIGUITY: two rows d apart repeat their response every lambda / d in sin(elevation).")
    print("  Half-width of the sector with no second lobe, asin(lambda / 2d), deg:")
    R["unambiguous_half_sector_deg"] = {}
    for d in (0.125, 0.15, 0.25, 2.0 * up_max, 0.50, 0.75, 1.00):
        row = [float(np.degrees(el.unambiguous_elevation(d, f, C))) for f in (4000.0, 5000.0, 6000.0)]
        R["unambiguous_half_sector_deg"][f"{d:.3f}"] = row
        print(f"    d = {d:5.3f} m: " + ", ".join(f"{v:5.1f} at {f:.0f} kHz" for v, f in zip(row, (4, 5, 6))))
    print(f"  Second lobe inside the searched sector ({SECTOR[0]:.0f} to {SECTOR[1]:.0f} deg), band-averaged, dB below the true one,")
    print("  and how often the estimator lands more than 5 deg from the truth (Monte Carlo):")
    R["ambiguity"] = {}
    for name in chosen[:2] + [f"3 rows at {-up_max:+.2f}, {mid:+.2f}, {up_max:+.2f} m", "4 rows, 0.125 m apart"]:
        h, o = geometries[name]
        R["ambiguity"][name] = {}
        for true in (0.0, 10.0, 20.0):
            lobe = second_lobe_db(h, o, true)
            wrong = {}
            for s in (13.0, 20.0):
                est = spectrum(h, o, s, true, rng, args.trials)
                wrong[str(s)] = float((np.abs(est - true) > 5.0).mean())
            R["ambiguity"][name][str(true)] = {"second_lobe_db": lobe, "wrong_lobe_fraction": wrong}
            print(f"    {name:44s} arrival at {true:4.0f} deg: lobe " + ("  none" if lobe <= -99.0 else f"{lobe:6.1f} dB")
                  + f"; wrong {100 * wrong['13.0']:5.1f}% at 13 dB, "
                  f"{100 * wrong['20.0']:5.1f}% at 20 dB")

    # --- which elevations arrive --------------------------------------------------------------------
    print("\nARRIVALS: largest elevation (deg) of the direct path and of paths with one and two boundary bounces,")
    print("  straight rays, source and receiver anywhere in the water column: atan(depth x (bounces + 1) / range).")
    R["arrival_angles_deg"] = {}
    for depth in (30.0, 100.0, 1000.0):
        row = {}
        for r in (500.0, 1000.0, 3000.0, 6000.0, 12000.0):
            row[str(r)] = [float(np.degrees(np.arctan(depth * (b + 1) / r))) for b in (0, 1, 2)]
        R["arrival_angles_deg"][str(depth)] = row
        print(f"    water {depth:6.0f} m: " + "; ".join(f"{float(r) / 1e3:g} km: " + "/".join(f"{v:.1f}" for v in vals) for r, vals in row.items()))
    crit = float(np.degrees(np.arccos(1500.0 / 1650.0)))
    R["critical_grazing_deg_sand"] = crit
    print(f"    A bottom of sound speed 1650 m/s reflects fully only below {crit:.1f} deg grazing: steeper bounce paths die out.")

    # --- cone -------------------------------------------------------------------------------------
    print("\nCONE ANGLE: horizontal bearing minus what one line reports, deg (asin(sin(cone) / cos(elevation)) - cone),")
    print("  and the cross-range error it is at 1 km / 12 km:")
    R["cone_bias_deg"] = {}
    for e in (2.0, 5.0, 10.0, 20.0):
        row = {str(b): float(np.degrees(el.cone_bias(np.radians(b), np.radians(e)))) for b in (15.0, 30.0, 45.0, 60.0)}
        R["cone_bias_deg"][str(e)] = row
        print(f"    elevation {e:4.0f} deg: " + ", ".join(f"{v:5.2f} at {float(b):.0f} deg" for b, v in row.items())
              + f"   ({1000 * np.radians(row['45.0']):.0f} m / {12000 * np.radians(row['45.0']):.0f} m at 45 deg)")

    # --- multipath --------------------------------------------------------------------------------
    print("\nMULTIPATH: isovelocity water 100 m deep, array at 30 m. For each arrival: elevation (deg), range beyond the")
    print("  direct echo when both legs take that path (m), and the straight-line depth it points at (m); folded back")
    print("  into the water column that depth is the target's again.")
    res = 0.5 * C / (F_HIGH - F_LOW)
    R["multipath"] = {"range_resolution_m": res, "cases": {}}
    for z_t in (50.0, 4.0):
        for r in (1000.0, 3000.0, 6000.0, 12000.0):
            arr = images(30.0, z_t, r, 100.0)
            direct = arr[0][2]
            case = {a[0]: {"elevation_deg": a[1], "excess_range_m": a[2] - direct, "image_depth_m": a[3],
                           "folded_depth_m": el.fold_depth(a[3], 100.0)[0]} for a in arr}
            R["multipath"]["cases"][f"target {z_t:.0f} m, {r / 1e3:.0f} km"] = case
            print(f"    target {z_t:3.0f} m at {r / 1e3:4.0f} km: " + "; ".join(
                f"{k} {v['elevation_deg']:+.2f} deg, +{v['excess_range_m']:.2f} m, {v['image_depth_m']:+.0f} m" for k, v in list(case.items())[:3]))
    print(f"  Range resolution is {res:.3f} m (c / 2B). Arrivals closer than that in range are one detection whose elevation")
    print("  is a mix of theirs (its quality drops); arrivals further apart are separate detections, each with its own elevation.")
    merge = {f"{z_t:.0f}": float(2.0 * 30.0 * z_t / res) for z_t in (4.0, 20.0, 50.0)}
    R["multipath"]["direct_and_surface_merge_beyond_m"] = merge
    print("  Direct and surface-bounce echoes are 2 z_array z_target / range apart: one detection beyond "
          + ", ".join(f"{v / 1e3:.1f} km for a target at {k} m" for k, v in merge.items()) + ".")
    width = float(np.degrees(0.886 * lam / (2.0 * up_max)))
    R["multipath"]["elevation_beamwidth_deg"] = width
    print(f"  The rows' own beam in elevation is {width:.0f} deg wide ({2.0 * up_max:.2f} m aperture): arrivals are never separated")
    print("  by elevation, only by range; elevation is measured on each arrival once range has separated it.")

    # --- refraction -------------------------------------------------------------------------------
    print("\nREFRACTION: the chain takes the ray as straight. Depth from a ray traced through a sound-speed gradient")
    print("  (elevation.ray_depth) minus the straight-line depth, m, array at 100 m, arrival from 2 and 8 degrees below level:")
    R["refraction_error_m"] = {}
    for label, g in (("isothermal water, +0.017 /s", 0.017), ("thermocline, -0.10 /s", -0.10)):
        row = {}
        for r in (1000.0, 3000.0, 6000.0):
            row[str(r)] = [float(el.ray_depth(100.0, -np.radians(a), r, [0.0, 3000.0], [1500.0 - g * 100.0, 1500.0 + g * 2900.0])
                                 - (100.0 + r * np.sin(np.radians(a)))) for a in (2.0, 8.0)]
        R["refraction_error_m"][label] = row
        print(f"    {label}: " + "; ".join(f"{float(r) / 1e3:.0f} km: {v[0]:+.0f} / {v[1]:+.0f}" for r, v in row.items()))
    print("  Beyond a few kilometres in a gradient the straight line is wrong by more than the measurement: use the profile.")

    # --- sensitivity ------------------------------------------------------------------------------
    d = 2.0 * up_max
    k = 2.0 * np.pi / lam
    print(f"\nSENSITIVITY (rows {d:.2f} m apart):")
    roll = {str(a): [float(r * np.radians(a)) for r in ranges] for a in (0.1, 0.2, 0.5, 1.0)}
    R["sensitivity"] = {"roll_depth_error_m": roll}
    print("  Roll error is elevation error, one for one at broadside; depth error = range x roll error:")
    for a, v in roll.items():
        print(f"    {a} deg: " + ", ".join(f"{x:.1f} m at {r / 1e3:.0f} km" for x, r in zip(v, ranges)))
    out_err = {str(mm): float(np.degrees(mm * 1e-3 / d)) for mm in (1.0, 2.0, 5.0)}
    phase = {str(p): float(np.degrees(np.radians(p) / (k * d))) for p in (2.0, 5.0, 10.0)}
    clock = {str(us): float(np.degrees(C * us * 1e-6 / d)) for us in (0.1, 0.5, 1.0, 8.3)}
    scale = float(np.degrees(np.sin(np.radians(10.0)) * 0.005 / d))
    R["sensitivity"].update(out_offset_error_deg=out_err, phase_mismatch_error_deg=phase, board_clock_error_deg=clock,
                            separation_error_5mm_at_10deg_deg=scale)
    print("  A row further out (athwartships) than the config says, elevation bias: "
          + ", ".join(f"{v:.2f} deg per {m} mm" for m, v in out_err.items()))
    print(f"  A 5 mm error in the rows' separation: {scale:.2f} deg at 10 deg elevation, none at 0 (a scale error).")
    print("  Phase mismatch between the rows' hydrophones and preamplifiers: "
          + ", ".join(f"{v:.2f} deg per {p} deg of phase" for p, v in phase.items()))
    print("  Clock error between a board on one row and a board on the other (elevation = c dt / d): "
          + ", ".join(f"{v:.2f} deg per {u} us" for u, v in clock.items()))
    print("    8.3 us is the present synchronisation tolerance (a twentieth of a period at 6 kHz): too loose for elevation.")
    print("    Wiring each board to four hydrophones of each row (the same stations) removes this term for two rows.")

    # --- what does not improve --------------------------------------------------------------------
    print("\nUNCHANGED OR NEARLY:")
    from scipy import optimize, stats

    def needed_snr_db(rows, pfa=1e-5, pd=0.5):
        """SNR per row at which the sum of `rows` power samples (steady echo, complex Gaussian
        noise) crosses the threshold for false-alarm rate `pfa` half the time."""
        thr = stats.chi2.isf(pfa, 2 * rows)
        return 10.0 * np.log10(optimize.brentq(lambda s: stats.ncx2.sf(thr, 2 * rows, 2.0 * rows * s) - pd, 1e-3, 1e3))

    base = needed_snr_db(1)
    gain = {str(n): {"coherent_db": float(10 * np.log10(n)), "rows_added_as_power_db": float(base - needed_snr_db(n))} for n in (2, 3)}
    R["unchanged"] = {"detection_gain_db": gain}
    print("  Detection: the rows' power maps are added. Against one row, for the same false-alarm rate (1e-5) and a 50% chance")
    print("  of detection, that is worth " + ", ".join(f"{v['rows_added_as_power_db']:.1f} dB with {n} rows" for n, v in gain.items())
          + "; a coherent sum would give " + ", ".join(f"{v['coherent_db']:.1f} dB" for v in gain.values())
          + ",\n  but only on arrivals near the elevation it is steered at.")
    print("  Bearing resolution and accuracy: the same (the array is no longer); bearing BIAS on elevated arrivals is removed.")
    print("  Range resolution: the same. Left/right: rows in one vertical plane cannot tell port from starboard;")
    print("  the hull behind the array does that, as now.")

    # --- recommendation -----------------------------------------------------------------------------
    R["recommended"] = {"two_rows_m": [-up_max, up_max], "three_rows_m": [-up_max, 0.0, up_max],
                        "three_rows_out_m": [0.0, out_of(0.0), 0.0]}
    print(f"\nRECOMMENDED for the option, on a {args.hull:.3f} m hull (unconfirmed): two rows at {-up_max:+.2f} and {up_max:+.2f} m "
          f"(64 hydrophones, 8 boards)\n  where arrivals stay within about 10 deg of level (shallow water beyond 1 km); three rows at "
          f"{-up_max:+.2f}, +0.00, {up_max:+.2f} m\n  (96 hydrophones, 12 boards) where they do not (deep water, short range). On a hull this small "
          "equal spacing beats\n  unequal: the rows are then closer than a wavelength and no second lobe falls inside the sector.")
    if not args.no_write:
        path = ROOT / "docs" / "design" / "popoto_elevation_study.json"
        path.write_text(json.dumps(R, indent=2))
        print(f"\nwrote {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
