#!/usr/bin/env python3
"""What do two or three stacked rows on the Popoto long-range array buy, in numbers?

The array is one horizontal line of 32 hydrophones (4-6 kHz). The owner has a flat vertical
face 2 m high and 9 m long to mount on, so rows can be up to about 2 m apart: 5 to 8
wavelengths. This study works out, for two rows and for three:

  accuracy     the Cramer-Rao bound on elevation against the signal-to-noise ratio after one
               row's beamforming and pulse compression, checked by Monte Carlo with the
               estimator the live chain uses (steer the rows over elevation with true time
               delays over the whole band, take the peak);
  two rows     rows several wavelengths apart repeat their response every few degrees. How
               far the band (40% of the centre frequency) suppresses the repeats, how often
               noise makes the estimator pick one, and what else resolves them: the sector
               that water depth and range allow, and a track's own history;
  three rows   the middle row's height searched for the lowest repeat with the outer rows at
               the full height, and the wrong-pick rate against SNR;
  depth        what the angle is in metres at 1, 6 and 12 km, with the terms that do not
               come from noise: roll, and the bending of rays in a sound-speed gradient that
               nobody will have measured (no long-range profile);
  cone         the bearing error one line makes on an arrival from above or below;
  multipath    direct, surface and bottom arrivals: when range separates them, when the
               rows' own beam now does;
  sensitivity  roll, mounting, phase mismatch, board clocks;
  unchanged    what stacked rows do not improve;
  face length  what the 9 m of the face could give a row that is 4.65 m long today.

Every number in docs/DESIGN.md Sec. 3 "stacked rows" is printed here and written to
docs/design/popoto_elevation_study.json. Confirmed by the owner: the face (2 m x 9 m, flat,
vertical). Chosen here inside it and not confirmed: the margins and so the row heights.

Usage: tools/popoto_elevation_study.py [--face-height 2.0] [--margin 0.1] [--trials 3000] [--no-write]
CPU only, a few minutes.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from gpusonar import elevation as el  # noqa: E402
from gpusonar import flank  # noqa: E402
from gpusonar.sim.noise import wenz_spectrum_db  # noqa: E402
from gpusonar.sonar import SonarModel  # noqa: E402
from gpusonar.sonar_equation import SonarEquation  # noqa: E402

C = 1500.0
F_LOW, F_HIGH = 4000.0, 6000.0
FC = 0.5 * (F_LOW + F_HIGH)
N_ROW = 32
SECTOR = (-30.0, 30.0, 0.1)
WRONG_DEG = 3.0                      # further than this from the truth: another lobe was picked


def grid_of(sector):
    return np.radians(np.arange(sector[0], sector[1] + 0.5 * sector[2], sector[2]))


def estimates(heights, snr_row_db, true_deg, rng, trials, sector=SECTOR, n_f=32):
    """Monte Carlo of the chain's estimator on its sufficient statistic: each row's beam at
    the detection over the band (flat spectrum, one unknown complex amplitude, range known),
    in white noise at `snr_row_db` per row after pulse compression. Returns the estimates (deg)."""
    h = np.asarray(heights, dtype=float)
    f = np.linspace(F_LOW, F_HIGH, n_f)
    grid = grid_of(sector)
    steer = np.exp(-2j * np.pi * f[None, :, None] * (h[:, None] * np.sin(grid)[None, :] / C)[:, None, :])   # [K, F, E]
    signal = np.exp(2j * np.pi * f[None, :] * (h * np.sin(np.radians(true_deg)) / C)[:, None])                # [K, F]
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
        out[i] = np.degrees(grid[e]) + frac * sector[2]
    return out


def pattern(heights, true_deg, sector=SECTOR, n_f=64):
    """Noise-free elevation response over the band (power, 1 at the arrival)."""
    h = np.asarray(heights, dtype=float)
    f = np.linspace(F_LOW, F_HIGH, n_f)
    grid = grid_of(sector)
    lag = h[:, None] * (np.sin(grid)[None, :] - np.sin(np.radians(true_deg))) / C
    p = np.abs(np.exp(-2j * np.pi * f[None, :, None] * lag[:, None, :]).sum(axis=(0, 1))) ** 2
    return np.degrees(grid), p / (len(h) * n_f) ** 2


def second_lobe_db(heights, true_deg, sector=SECTOR):
    """Highest point of the response outside the main lobe, inside the sector, dB (-99: none)."""
    _, p = pattern(heights, true_deg, sector)
    e = int(np.argmax(p))
    lo = e
    while lo > 0 and p[lo - 1] < p[lo]:
        lo -= 1
    hi = e
    while hi < len(p) - 1 and p[hi + 1] < p[hi]:
        hi += 1
    rest = np.concatenate([p[:lo], p[hi + 1:]])
    return float(max(10.0 * np.log10(max(rest.max(), 1e-30)), -99.0)) if len(rest) else -99.0


def wrong_fraction(heights, snr_db, true_deg, rng, trials, sector=SECTOR):
    est = estimates(heights, snr_db, true_deg, rng, trials, sector)
    return float((np.abs(est - true_deg) > WRONG_DEG).mean())


def sigma_deg(heights, snr_row_db):
    return float(np.degrees(el.sin_elevation_sigma(np.repeat(heights, N_ROW), 10.0 ** (snr_row_db / 10.0) / N_ROW, F_LOW, F_HIGH, C)))


def link_snr(ranges, target_db):
    """Output SNR (dB) of ONE row of the long-range array: the budget of docs/DESIGN.md Sec. 5."""
    sonar = SonarModel.load("popoto_long_range")
    noise = float(wenz_spectrum_db(np.array([sonar.center_frequency]), 5.0, 0.5)[0])
    eq = SonarEquation.from_sonar(sonar, noise, target_strength_db=target_db, temperature=12.0, depth=50.0,
                                  absorption_model="ainslie_mccolm")
    return [float(eq.snr(r)) for r in ranges]


def images(z_r, z_t, r, depth):
    out = []
    for name, z_img in (("direct", z_t), ("surface", -z_t), ("bottom", 2.0 * depth - z_t)):
        dz = z_img - z_r
        out.append((name, float(np.degrees(np.arctan2(-dz, r))), float(np.hypot(r, dz)), float(z_img)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--face-height", type=float, default=2.0, help="height of the mounting face, m (the owner's figure)")
    ap.add_argument("--face-length", type=float, default=9.0, help="length of the mounting face, m (the owner's figure)")
    ap.add_argument("--margin", type=float, default=0.1, help="kept free at the top and bottom of the face, m (our choice)")
    ap.add_argument("--trials", type=int, default=3000)
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args()
    rng = np.random.default_rng(7)
    lam = C / FC
    up = float(np.round(0.5 * args.face_height - args.margin, 3))
    d = 2.0 * up
    snrs = (13.0, 20.0, 30.0)
    R = {"face_height_m": args.face_height, "face_length_m": args.face_length, "margin_m": args.margin, "outer_row_m": up,
         "baseline_m": d, "wavelength_m": lam, "band_hz": [F_LOW, F_HIGH], "sector_deg": list(SECTOR), "wrong_deg": WRONG_DEG}
    print(f"Band {F_LOW / 1e3:.0f}-{F_HIGH / 1e3:.0f} kHz, wavelength {C / F_HIGH * 100:.0f}-{C / F_LOW * 100:.1f} cm. Face {args.face_height:.1f} m x "
          f"{args.face_length:.1f} m (the owner's); {args.margin:.2f} m kept free top and bottom (our choice):\n"
          f"outer rows at -{up:.2f} and +{up:.2f} m, {d:.2f} m apart = {d * F_LOW / C:.1f} to {d * F_HIGH / C:.1f} wavelengths.\n")

    # --- accuracy ---------------------------------------------------------------------------------
    print("ACCURACY: one-sigma elevation (deg) against the SNR of one row after beamforming and pulse compression.")
    print("  bound = 1 / (k_rms sqrt(2 SNR sum_rows (h - mean h)^2)); 'MC' = Monte Carlo of the estimator at 20 dB over the bound,")
    print("  counting only estimates on the true lobe (the others are counted under ambiguity).")
    print(f"  {'geometry':40s} " + " ".join(f"{s:>5.0f} dB" for s in snrs) + "   MC/bound")
    R["accuracy"] = {}
    two_rows = {f"2 rows, {x:.2f} m apart": [-x / 2, x / 2] for x in (0.38, 0.50, 1.00, 1.50, d, args.face_height)}
    for name, h in two_rows.items():
        sig = [sigma_deg(h, s) for s in snrs]
        est = estimates(h, 20.0, 0.0, rng, args.trials)
        near = est[np.abs(est) < WRONG_DEG]
        R["accuracy"][name] = {"heights_m": list(h), "sigma_deg": dict(zip(map(str, snrs), sig)),
                               "monte_carlo_over_bound_at_20_db": float(near.std() / sig[1]), "fits_face": bool(h[1] <= 0.5 * args.face_height + 1e-9)}
        print(f"  {name:40s} " + " ".join(f"{s:8.3f}" for s in sig) + f"   {near.std() / sig[1]:7.2f}")
    two = f"2 rows, {d:.2f} m apart"

    # --- two rows: the repeats --------------------------------------------------------------------
    print(f"\nTWO ROWS {d:.2f} m apart: the response repeats every lambda / d in sin(elevation):")
    spacing = [float(np.degrees(np.arcsin(C / (f * d)))) for f in (F_LOW, FC, F_HIGH)]
    print("  " + ", ".join(f"{s:.1f} deg at {f / 1e3:.0f} kHz" for s, f in zip(spacing, (F_LOW, FC, F_HIGH))) + ".")
    print("  Over the band the repeats land at different angles and smear. With true time delays over 4-6 kHz the m-th repeat is left")
    print("  at sinc^2(m B / 2 fc) of the true peak, whatever the separation (B / fc = 0.4):")
    lobes = [float(20.0 * np.log10(abs(np.sinc(m * (F_HIGH - F_LOW) / (2.0 * FC))))) for m in (1, 2, 3, 4)]
    numeric = second_lobe_db([-up, up], 0.0)
    print("  " + ", ".join(f"m = {m}: {v:.1f} dB" for m, v in zip((1, 2, 3, 4), lobes)) + f"   (numeric, nearest repeat: {numeric:.1f} dB)")
    R["two_rows"] = {"repeat_spacing_deg": spacing, "repeat_level_db": lobes, "numeric_nearest_db": numeric, "wrong_pick": {}, "sector_ranges_m": {}}
    print(f"  How often the estimator lands more than {WRONG_DEG:.0f} deg from the truth, arrival at 2 deg:")
    half = spacing[1] / 2.0
    for label, sector in (("searching -30 to 30 deg", SECTOR), (f"searching -{2 * half:.1f} to {2 * half:.1f} deg", (-2 * half, 2 * half, 0.1)),
                          (f"searching -{half:.1f} to {half:.1f} deg (half a repeat each side)", (-half, half, 0.1))):
        row = {str(s): wrong_fraction([-up, up], s, 2.0, rng, args.trials, sector) for s in snrs}
        R["two_rows"]["wrong_pick"][label] = row
        print(f"    {label:58s} " + ", ".join(f"{100 * v:5.1f}% at {float(s):.0f} dB" for s, v in row.items()))
    print("  So the band does not make two rows unambiguous: the nearest repeats are 0.6 dB down and are picked a third to a half of the time.")
    print(f"  What does: knowing that nothing can arrive from outside +-{half:.1f} deg. Straight rays in water of depth H reach the array within")
    print("  atan(H (bounces + 1) / range) of level, so two rows are unambiguous beyond (direct path / one bounce / two bounces):")
    for depth in (30.0, 100.0, 300.0, 1000.0, 3000.0):
        row = [float(depth * (b + 1) / np.tan(np.radians(half))) for b in (0, 1, 2)]
        R["two_rows"]["sector_ranges_m"][str(depth)] = row
        print(f"    water {depth:6.0f} m: " + " / ".join(f"{v / 1e3:.1f} km" for v in row))
    R["two_rows"]["unambiguous_half_sector_deg"] = half
    print("  Inside those ranges two rows give a set of candidate elevations " + f"{spacing[1]:.1f} deg apart, not one. A track helps: its depth so far")
    print("  picks the candidate (the chain does this), and as the range changes the wrong candidates' depths move while the right one stays.")
    step = float(1000.0 * np.radians(spacing[1]))
    R["two_rows"]["candidate_depth_step_m_per_km"] = step
    print(f"  Candidates are {step:.0f} m of depth apart per km of range.")

    # --- three rows -------------------------------------------------------------------------------
    print(f"\nTHREE ROWS, outer rows at -{up:.2f} and +{up:.2f} m: where to put the middle one. Highest repeat inside {SECTOR[0]:.0f} to {SECTOR[1]:.0f} deg,")
    print("  over the band, for arrivals at 0, 10 and 20 deg (the worst of the three counts), against the middle row's height:")
    R["three_rows"] = {"search": {}}
    best, best_level = None, 0.0
    positions = np.round(np.arange(-up + 0.05, 0.0001, 0.05), 3)        # the pattern is the same mirrored about the centre
    for p in positions:
        level = max(second_lobe_db([-up, float(p), up], a) for a in (0.0, 10.0, 20.0))
        R["three_rows"]["search"][f"{p:.2f}"] = level
        if level < best_level:
            best, best_level = float(p), level
    line = ", ".join(f"{float(k):+.2f}: {v:.1f}" for k, v in R["three_rows"]["search"].items())
    print("    " + line.replace(", ", ",  "))
    print(f"  Lowest: middle row at {best:+.2f} m ({up + best:.2f} m above the bottom row, {up - best:.2f} m below the top): {best_level:.1f} dB. "
          f"Equal spacing (0.00): {R['three_rows']['search']['0.00']:.1f} dB.")
    R["three_rows"].update(best_middle_m=best, best_level_db=best_level, equal_level_db=R["three_rows"]["search"]["0.00"])
    three_h = [-up, best, up]
    three = f"3 rows at {-up:+.2f}, {best:+.2f}, {up:+.2f} m"
    equal = f"3 rows at {-up:+.2f}, +0.00, {up:+.2f} m (equal)"
    geometries = {two: [-up, up], three: three_h, equal: [-up, 0.0, up]}
    R["chosen"] = {"two_rows": two, "three_rows": three, "three_rows_heights_m": three_h, "equal": equal}
    print("  Wrong picks, searching -30 to 30 deg (arrival at 2 deg / at 15 deg), and one-sigma elevation on the true lobe:")
    R["three_rows"]["wrong_pick"] = {}
    for name in (three, equal, two):
        h = geometries[name]
        row = {str(s): [wrong_fraction(h, s, a, rng, args.trials) for a in (2.0, 15.0)] for s in (10.0,) + snrs}
        R["three_rows"]["wrong_pick"][name] = row
        sig = [sigma_deg(h, s) for s in snrs]
        est = estimates(h, 20.0, 0.0, rng, args.trials)
        near = est[np.abs(est) < WRONG_DEG]
        R["accuracy"][name] = {"heights_m": list(h), "sigma_deg": dict(zip(map(str, snrs), sig)),
                               "monte_carlo_over_bound_at_20_db": float(near.std() / sig[1]), "fits_face": True}
        print(f"    {name:40s} " + ", ".join(f"{100 * v[0]:4.1f}% / {100 * v[1]:4.1f}% at {float(s):.0f} dB" for s, v in row.items())
              + "; sigma " + " / ".join(f"{s:.3f}" for s in sig) + " deg at 13 / 20 / 30 dB")

    # --- depth ------------------------------------------------------------------------------------
    ranges = (1000.0, 6000.0, 12000.0)
    gradients = {"0.017": 0.017, "0.05": 0.05, "0.1": 0.1}
    print("\nDEPTH: depth error = range x elevation error. One sigma (m) from noise alone at the SNR the link budget gives one row")
    print("  (spherical spreading; SNR above 40 dB counted as 40 dB), then what does not come from noise.")
    R["link_snr_db"], R["depth_sigma_m"] = {}, {}
    for label, ts in (("ship, +25 dB", 25.0), ("small target, +5 dB", 5.0)):
        snr = link_snr(ranges, ts)
        R["link_snr_db"][label] = dict(zip(map(str, ranges), snr))
        print(f"  {label}: SNR " + ", ".join(f"{s:.0f} dB at {r / 1e3:.0f} km" for s, r in zip(snr, ranges)))
        for name in (two, three, "2 rows, 0.38 m apart"):
            h = geometries.get(name, [-0.19, 0.19])
            row = [None if s < 10.0 else float(r * np.radians(sigma_deg(h, min(s, 40.0)))) for r, s in zip(ranges, snr)]
            R["depth_sigma_m"][f"{label}; {name}"] = dict(zip(map(str, ranges), row))
            print(f"    {name:40s} " + "  ".join(("not detected" if v is None else f"{v:7.1f} m") + f" at {r / 1e3:.0f} km" for v, r in zip(row, ranges)))
    print("  Roll known to 0.2 deg: " + ", ".join(f"{r * np.radians(0.2):.1f} m at {r / 1e3:.0f} km" for r in ranges) + ".")
    print("  Ray bending. The vehicle measures the sound speed where it is, which fixes the angle at the array, but not the gradient along")
    print("  the path. A mean vertical gradient g bends the ray by range^2 g / 2c from the straight line:")
    R["refraction_m"] = {}
    for key, g in gradients.items():
        row = [float(r * r * g / (2.0 * C)) for r in ranges]
        R["refraction_m"][key] = dict(zip(map(str, ranges), row))
        print(f"    g = {g:5.3f} /s: " + ", ".join(f"{v:.0f} m at {r / 1e3:.0f} km" for v, r in zip(row, ranges))
              + ("   (isothermal water: pressure alone)" if key == "0.017" else ""))
    check = float(el.ray_depth(100.0, -np.radians(2.0), 3000.0, [0.0, 3000.0], [C - 0.017 * 100.0, C + 0.017 * 2900.0]) - (100.0 + 3000.0 * np.sin(np.radians(2.0))))
    R["refraction_check_3km_isothermal_m"] = check
    print(f"    (traced ray, isothermal, 3 km: {check:+.0f} m; the formula gives {-9e6 * 0.017 / 3000.0:+.0f} m.)")
    print("  Range beyond which the unknown gradient, not the measurement, limits depth (range = 2 c sigma_elevation / sigma_g):")
    R["refraction_crossover_m"] = {}
    for key, g in gradients.items():
        row = {str(s): float(2.0 * C * np.radians(sigma_deg([-up, up], s)) / g) for s in snrs}
        R["refraction_crossover_m"][key] = row
        print(f"    gradient known to {g:5.3f} /s: " + ", ".join(f"{v:.0f} m at {float(s):.0f} dB" for s, v in row.items()))
    print("  With rows this far apart depth is refraction-limited almost everywhere unless the gradient is known. What survives without")
    print("  a profile: the angle itself (exact at the array), whether an arrival is from above or below, which arrival of a group came")
    print("  first, and changes of a track's elevation from ping to ping.")

    # --- cone -------------------------------------------------------------------------------------
    print("\nCONE ANGLE: horizontal bearing minus what one line reports, deg (asin(sin(cone) / cos(elevation)) - cone):")
    R["cone_bias_deg"] = {}
    for e in (2.0, 5.0, 10.0, 20.0):
        row = {str(b): float(np.degrees(el.cone_bias(np.radians(b), np.radians(e)))) for b in (15.0, 30.0, 45.0, 60.0)}
        R["cone_bias_deg"][str(e)] = row
        print(f"    elevation {e:4.0f} deg: " + ", ".join(f"{v:5.2f} at {float(b):.0f} deg" for b, v in row.items())
              + f"   ({1000 * np.radians(row['45.0']):.0f} m / {12000 * np.radians(row['45.0']):.0f} m of cross-range at 45 deg, 1 km / 12 km)")

    # --- multipath --------------------------------------------------------------------------------
    res = 0.5 * C / (F_HIGH - F_LOW)
    width3 = float(np.degrees(0.886 * lam / (1.5 * d)))          # three rows fill the aperture roughly; two rows have lobes, not a beam
    g_, p_ = pattern(three_h, 0.0)
    main = g_[p_ >= 0.5]
    width3 = float(main.max() - main.min())
    print("\nMULTIPATH: isovelocity water 100 m deep, array at 30 m. Elevation (deg) and range beyond the direct echo (m):")
    R["multipath"] = {"range_resolution_m": res, "elevation_beamwidth_deg": width3, "cases": {}}
    for z_t in (50.0, 4.0):
        for r in (300.0, 1000.0, 3000.0, 6000.0, 12000.0):
            arr = images(30.0, z_t, r, 100.0)
            case = {a[0]: {"elevation_deg": a[1], "excess_range_m": a[2] - arr[0][2], "image_depth_m": a[3]} for a in arr}
            R["multipath"]["cases"][f"target {z_t:.0f} m, {r / 1e3:g} km"] = case
            print(f"    target {z_t:3.0f} m at {r / 1e3:4.1f} km: " + "; ".join(f"{k} {v['elevation_deg']:+.2f} deg, +{v['excess_range_m']:.2f} m" for k, v in case.items()))
    print(f"  Range resolution {res:.3f} m. The three rows' own beam in elevation is {width3:.1f} deg wide (3 dB), against 40 deg on a 0.38 m span.")
    sep = {f"{z:.0f}": float(2.0 * z / np.radians(width3)) for z in (10.0, 30.0, 100.0)}
    R["multipath"]["direct_surface_resolved_in_elevation_within_m"] = sep
    print("  Direct and surface arrivals are 2 x (the shallower of array and target) / range apart in angle: wider than that beam within "
          + ", ".join(f"{v:.0f} m for {k} m" for k, v in sep.items()) + ".")
    print("  So inside about a kilometre two arrivals in one range cell can now be told apart in elevation (three rows; two rows cannot,")
    print("  their repeats are in the way), and beyond it they still cannot: there range must separate them, as before.")
    merge = {f"{z_t:.0f}": float(2.0 * 30.0 * z_t / res) for z_t in (4.0, 20.0, 50.0)}
    R["multipath"]["direct_and_surface_merge_beyond_m"] = merge
    print("  Direct and surface echoes are within a range cell beyond " + ", ".join(f"{v / 1e3:.1f} km for a target at {k} m" for k, v in merge.items()) + ".")

    # --- sensitivity ------------------------------------------------------------------------------
    k = 2.0 * np.pi / lam
    print(f"\nSENSITIVITY (outer rows {d:.2f} m apart; in brackets the same for the 0.38 m of the earlier study):")
    roll = {str(a): [float(r * np.radians(a)) for r in ranges] for a in (0.1, 0.2, 0.5, 1.0)}
    R["sensitivity"] = {"roll_depth_error_m": roll}
    print("  Roll error is elevation error, one for one, whatever the rows: depth error = range x roll error.")
    for a, v in roll.items():
        print(f"    {a} deg: " + ", ".join(f"{x:.1f} m at {r / 1e3:.0f} km" for x, r in zip(v, ranges)))
    out_err = {str(mm): float(np.degrees(mm * 1e-3 / d)) for mm in (1.0, 5.0)}
    phase = {str(p): float(np.degrees(np.radians(p) / (k * d))) for p in (5.0, 10.0)}
    clock = {str(us): float(np.degrees(C * us * 1e-6 / d)) for us in (0.5, 1.0, 8.3)}
    scale = float(np.degrees(np.sin(np.radians(10.0)) * 0.005 / d))
    R["sensitivity"].update(out_offset_error_deg=out_err, phase_mismatch_error_deg=phase, board_clock_error_deg=clock,
                            separation_error_5mm_at_10deg_deg=scale)
    f = 1.8 / 0.38 * d / 1.8
    print("  One row further out from the face than the other: " + ", ".join(f"{v:.3f} deg per {m} mm" for m, v in out_err.items())
          + f"   [{out_err['1.0'] * d / 0.38:.2f} deg per mm]")
    print("  Phase mismatch between rows: " + ", ".join(f"{v:.3f} deg per {p} deg of phase" for p, v in phase.items()))
    print("  Clock error between boards on different rows (elevation = c dt / d): " + ", ".join(f"{v:.3f} deg per {u} us" for u, v in clock.items())
          + f"   [{clock['1.0'] * d / 0.38:.2f} deg per us]")
    print(f"  A 5 mm error in the rows' separation: {scale:.3f} deg at 10 deg elevation, none at 0.")
    worst = float(np.degrees(np.arcsin(lam / (2.0 * d))))
    print(f"  All of these shrink with the separation ({f:.1f} times against 0.38 m). Roll does not, and is now the first limit of the angle.")
    R["sensitivity"]["half_repeat_deg"] = worst

    # --- detection --------------------------------------------------------------------------------
    from scipy import optimize, stats

    def needed_snr_db(rows, pfa=1e-5, pd=0.5):
        thr = stats.chi2.isf(pfa, 2 * rows)
        return 10.0 * np.log10(optimize.brentq(lambda s: stats.ncx2.sf(thr, 2 * rows, 2.0 * rows * s) - pd, 1e-3, 1e3))

    base = needed_snr_db(1)
    gain = {str(n): {"coherent_db": float(10 * np.log10(n)), "rows_added_as_power_db": float(base - needed_snr_db(n))} for n in (2, 3)}
    R["unchanged"] = {"detection_gain_db": gain}
    print("\nUNCHANGED OR NEARLY: detection (rows' power maps added: " + ", ".join(f"{v['rows_added_as_power_db']:.1f} dB with {n} rows" for n, v in gain.items())
          + " against one row at the same false-alarm rate);")
    print("  bearing and range resolution; left/right (the face decides that, as the hull did).")

    # --- the face's length ------------------------------------------------------------------------
    print(f"\nTHE FACE IS {args.face_length:.0f} m LONG and a row is 32 hydrophones at 0.15 m, 4.65 m. For the owner, not a change made here:")
    R["face_length"] = {}
    n_full = int(args.face_length // 0.15)
    for label, n, pitch in (("as now", 32, 0.15), (f"{n_full} hydrophones at 0.15 m", n_full, 0.15),
                            ("the same 32 spread over 8.8 m", 32, 8.8 / 31.0), ("the same 32 at 0.20 m", 32, 0.20)):
        x = (np.arange(n) - (n - 1) / 2.0) * pitch
        rep = flank.ambiguity_report(x, F_LOW, F_HIGH, C, steer_deg=30.0)
        row = {"hydrophones": n, "pitch_m": float(pitch), "length_m": float(n * pitch), "beamwidth_deg": float(np.degrees(0.886 * lam / (n * pitch))),
               "array_gain_db": float(10 * np.log10(n)), "worst_lobe_steered_30_deg_db": rep["narrowband_worst_lobe_db"],
               "wideband_lobe_steered_30_deg_db": rep["wideband_lobe_db"], "boards_per_row": int(np.ceil(n / 8))}
        R["face_length"][label] = row
        print(f"    {label:32s}: {row['length_m']:.2f} m, beam {row['beamwidth_deg']:.2f} deg at broadside, gain {row['array_gain_db']:.1f} dB, "
              f"{row['boards_per_row']} boards a row; steered 30 deg, highest other lobe {row['worst_lobe_steered_30_deg_db']:.1f} dB at one "
              f"frequency, {row['wideband_lobe_steered_30_deg_db']:.1f} dB over the band")
    print("  Filling the face nearly halves the beam and adds 2.7 dB a row for almost twice the hydrophones and boards. Spreading the")
    print("  same 32 does not work: more than half a wavelength apart they have full-height repeats in bearing when steered.")
    print("  Rows shifted along the face against each other change nothing for bearing (rows are added as power) or elevation.")

    # --- recommendation ---------------------------------------------------------------------------
    w2, w3 = R["three_rows"]["wrong_pick"][two], R["three_rows"]["wrong_pick"][three]
    R["recommended"] = {"two_rows_m": [-up, up], "three_rows_m": three_h, "default": "three_rows"}
    print(f"\nRECOMMENDED: three rows at {-up:+.2f}, {best:+.2f}, {up:+.2f} m as the default of the option; two rows at {-up:+.2f}, {up:+.2f} m shipped beside it.")
    print(f"  Same accuracy ({sigma_deg([-up, up], 20.0):.2f} deg against {sigma_deg(three_h, 20.0):.2f} deg at 20 dB). Two rows pick a wrong repeat {100 * w2['20.0'][0]:.0f}% of the time at 20 dB "
          f"when searching 60 deg;\n  three rows {100 * w3['20.0'][0]:.1f}% ({100 * w3['13.0'][0]:.0f}% at 13 dB). Two rows are sound only where water depth and range confine the "
          f"arrivals to +-{half:.1f} deg.")
    if not args.no_write:
        path = ROOT / "docs" / "design" / "popoto_elevation_study.json"
        path.write_text(json.dumps(R, indent=2))
        print(f"\nwrote {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
