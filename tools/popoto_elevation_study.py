#!/usr/bin/env python3
"""What do two or three stacked rows on the Popoto long-range array buy, in numbers?

The array is one horizontal line of 32 hydrophones (4-6 kHz). The owner has a flat vertical
face 2 m high and 9 m long, and has since asked for "way more margin. Projector etc ... probably
try like a 7m horizontal and the vertical spacing is maybe a 1.5": the rows may take about 7 m of
the length and the outer rows are about 1.5 m apart (4 to 6 wavelengths). Both are parameters
(--array-length, --baseline); the study shows what 1.8 -> 1.5 m and 9 -> 7 m cost. It works out,
for two rows and for three:

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
  face length  what the usable length of the face could give a row that is 4.8 m long today;
  projector    where it sits on the face, and what its direct sound does at the hydrophones;
  stages       test AUV (one row), a static drop-in array (two or three rows), the full face;
  baseline     elevation accuracy against the rows' vertical separation, 0.1 to 2 m: the curve the
               static array's frame and any other mounting is chosen from;
  roll         depth error against roll accuracy, so a datasheet figure can be dropped in.

Every number in docs/DESIGN.md Sec. 3 "stacked rows" is printed here and written to
docs/design/popoto_elevation_study.json. Stated by the owner: the face (2 m x 9 m, flat,
vertical). The owner's working values ("probably try like", "maybe"): 7 m and 1.5 m. Chosen
here: the middle row's height, the projector's place, everything about the static array.

Usage: tools/popoto_elevation_study.py [--baseline 1.5] [--array-length 7.0] [--trials 3000] [--no-write]
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
from gpusonar.sonar import PEAK_OVER_RMS_DB, SonarModel  # noqa: E402
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
    ap.add_argument("--baseline", type=float, default=1.5, help="outer rows apart, m (the owner's working value: 'maybe a 1.5')")
    ap.add_argument("--array-length", type=float, default=7.0, help="length of the face the rows may take, m (the owner's working value)")
    ap.add_argument("--compare-baseline", type=float, default=1.8, help="the earlier baseline, for the sensitivity table")
    ap.add_argument("--trials", type=int, default=3000)
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args()
    rng = np.random.default_rng(7)
    lam = C / FC
    d = float(args.baseline)
    up = float(np.round(0.5 * d, 3))
    margin = 0.5 * (args.face_height - d)
    snrs = (13.0, 20.0, 30.0)
    R = {"face_height_m": args.face_height, "face_length_m": args.face_length, "array_length_m": args.array_length, "margin_m": margin, "outer_row_m": up,
         "baseline_m": d, "wavelength_m": lam, "band_hz": [F_LOW, F_HIGH], "sector_deg": list(SECTOR), "wrong_deg": WRONG_DEG}
    print(f"Band {F_LOW / 1e3:.0f}-{F_HIGH / 1e3:.0f} kHz, wavelength {C / F_HIGH * 100:.0f}-{C / F_LOW * 100:.1f} cm. Face {args.face_height:.1f} m x "
          f"{args.face_length:.1f} m (the owner's); rows within {args.array_length:.1f} m of it, outer rows {d:.2f} m apart (the owner's working values; {margin:.2f} m free top and bottom):\n"
          f"outer rows at -{up:.2f} and +{up:.2f} m, {d:.2f} m apart = {d * F_LOW / C:.1f} to {d * F_HIGH / C:.1f} wavelengths.\n")

    # --- accuracy ---------------------------------------------------------------------------------
    print("ACCURACY: one-sigma elevation (deg) against the SNR of one row after beamforming and pulse compression.")
    print("  bound = 1 / (k_rms sqrt(2 SNR sum_rows (h - mean h)^2)); 'MC' = Monte Carlo of the estimator at 20 dB over the bound,")
    print("  counting only estimates on the true lobe (the others are counted under ambiguity).")
    print(f"  {'geometry':40s} " + " ".join(f"{s:>5.0f} dB" for s in snrs) + "   MC/bound")
    R["accuracy"] = {}
    two_rows = {f"2 rows, {x:.2f} m apart": [-x / 2, x / 2] for x in sorted({0.38, 0.50, 1.00, d, args.compare_baseline, args.face_height})}
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
    print("  In clean noise the band does separate the repeats from 20 dB up (the rows' echoes must line up to 0.1 ms, a fifth of the pulse's")
    print("  resolution), and at 13 dB one pick in ten is wrong. But the margin is 0.6 dB: the range sidelobe of another arrival or a few degrees of")
    print("  phase error between rows is enough to tip it. The chain therefore never trusts a repeat within 1 dB of the peak (elevation.ambiguity_db).")
    R["two_rows"]["margin_db"] = float(-lobes[0])
    print(f"  What makes two rows sound: knowing that nothing can arrive from outside +-{half:.1f} deg. Straight rays in water of depth H reach the array within")
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
    roll = {str(a): [float(r * np.radians(a)) for r in ranges] for a in (0.02, 0.05, 0.1, 0.2, 0.5, 1.0)}
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
    f = d / 0.38
    print("  One row further out from the face than the other: " + ", ".join(f"{v:.3f} deg per {m} mm" for m, v in out_err.items())
          + f"   [{out_err['1.0'] * d / 0.38:.2f} deg per mm]")
    print("  Phase mismatch between rows: " + ", ".join(f"{v:.3f} deg per {p} deg of phase" for p, v in phase.items()))
    print("  Clock error between boards on different rows (elevation = c dt / d): " + ", ".join(f"{v:.3f} deg per {u} us" for u, v in clock.items())
          + f"   [{clock['1.0'] * d / 0.38:.2f} deg per us]")
    print(f"  A 5 mm error in the rows' separation: {scale:.3f} deg at 10 deg elevation, none at 0.")
    worst = float(np.degrees(np.arcsin(lam / (2.0 * d))))
    print(f"  All of these shrink with the separation ({f:.1f} times against 0.38 m). Roll does not, and is now the first limit of the angle.")
    R["sensitivity"]["half_repeat_deg"] = worst
    # What the board clocks must hold between rows for a given elevation error (the owner has not answered this one).
    R["sensitivity"]["clock_us_for_deg"] = {str(a): float(d * np.radians(a) / C * 1e6) for a in (0.05, 0.1, 0.2)}
    print("  Clock agreement between boards of different rows needed for an elevation error of: "
          + ", ".join(f"{float(a)} deg: {v:.2f} us" for a, v in R["sensitivity"]["clock_us_for_deg"].items()))

    # --- detection --------------------------------------------------------------------------------
    # What a detection needs is asked of the chain's own detector (gpusonar.buildspec.detection, from gpusonar.live): the SNR of
    # ONE row for the design detection probability, with the rows' maps added in power. No threshold and no row gain is typed here.
    from gpusonar import buildspec as bs

    face_window = SonarModel.load("popoto_long_range_3row").config["receive"]["window"]["range"]      # the face's receive window: the detector's thresholds follow from it

    def detector(n, pitch, rows, pulse, window=None):
        return bs.detection_for_row(n, pitch, rows, pulse, face_window if window is None else window)

    face = {r: detector(32, 0.15, r, 1.0) for r in (1, 2, 3, 4)}
    gain = {str(n): {"coherent_db": float(10 * np.log10(n)), "rows_added_as_power_db": float(face[n]["rows_gain_db"])} for n in (2, 3)}
    R["unchanged"] = {"detection_gain_db": gain}
    R["detection"] = {"probability": face[1]["probability"], "probability_high": face[1]["probability_high"], "statistic": face[1]["statistic"],
                      "one_row_snr_needed_db": {str(r): face[r]["threshold_db"] for r in face}, "one_row_snr_needed_high_db": {str(r): face[r]["threshold_high_db"] for r in face},
                      "what": "the SNR of one row the chain's detector needs on the face (32 at 0.15 m, 1 s pulse, the long-range window), rows added in power: gpusonar.buildspec.detection"}
    print("\nUNCHANGED OR NEARLY: detection (rows' power maps added: " + ", ".join(f"{v['rows_added_as_power_db']:.1f} dB with {n} rows" for n, v in gain.items())
          + f" against one row at the chain's detector, which needs {face[1]['threshold_db']:.1f} dB of one row's SNR for a detection probability of {face[1]['probability']:g});")
    print("  bearing and range resolution; left/right (the face decides that, as the hull did).")

    # --- the face's length ------------------------------------------------------------------------
    print(f"\nTHE ROWS MAY TAKE {args.array_length:.0f} m of the {args.face_length:.0f} m face and a row is 32 hydrophones at 0.15 m, 4.8 m. For the owner, not a change made here:")
    R["face_length"] = {}
    n_full = int(args.array_length // 0.15)
    spread = (args.array_length - 0.2) / 31.0
    for label, n, pitch in (("as now", 32, 0.15), (f"{n_full} hydrophones at 0.15 m", n_full, 0.15),
                            (f"the same 32 spread over {args.array_length - 0.2:.1f} m", 32, spread), ("the same 32 at 0.20 m", 32, 0.20)):
        x = (np.arange(n) - (n - 1) / 2.0) * pitch
        rep = flank.ambiguity_report(x, F_LOW, F_HIGH, C, steer_deg=30.0)
        row = {"hydrophones": n, "pitch_m": float(pitch), "length_m": float(n * pitch), "beamwidth_deg": float(np.degrees(0.886 * lam / (n * pitch))),
               "array_gain_db": float(10 * np.log10(n)), "worst_lobe_steered_30_deg_db": rep["narrowband_worst_lobe_db"],
               "wideband_lobe_steered_30_deg_db": rep["wideband_lobe_db"], "boards_per_row": int(np.ceil(n / 8))}
        R["face_length"][label] = row
        print(f"    {label:32s}: {row['length_m']:.2f} m, beam {row['beamwidth_deg']:.2f} deg at broadside, gain {row['array_gain_db']:.1f} dB, "
              f"{row['boards_per_row']} boards a row; steered 30 deg, highest other lobe {row['worst_lobe_steered_30_deg_db']:.1f} dB at one "
              f"frequency, {row['wideband_lobe_steered_30_deg_db']:.1f} dB over the band")
    print("  Filling the length narrows the beam and adds gain in proportion to the hydrophones and boards. Spreading the")
    print("  same 32 does not work: more than half a wavelength apart they have full-height repeats in bearing when steered.")
    print("  Rows shifted along the face against each other change nothing for bearing (rows are added as power) or elevation.")

    # --- stages -----------------------------------------------------------------------------------
    def directivity_db(n, pitch, f=FC):
        """Array gain of a line of n hydrophones against isotropic noise at broadside: N^2 / sum sinc(2 d_ij / lambda)."""
        x = np.arange(n) * pitch
        return float(10.0 * np.log10(n * n / np.sinc(2.0 * (x[:, None] - x[None, :]) * f / C).sum()))

    def reach(n, pitch, rows, source_level, pulse, target_db, det):
        """Noise-limited range (m) at which ONE row's SNR falls to what the detector needs (`det`, with the rows added):
        the link budget of Sec. 5 with the row's gain against isotropic noise. The rows are in the threshold, not in the gain."""
        noise = float(wenz_spectrum_db(np.array([FC]), 5.0, 0.5)[0])
        eq = SonarEquation(source_level, FC, F_HIGH - F_LOW, pulse, n, noise, target_strength_db=target_db, temperature=12.0, depth=50.0,
                           absorption_model="ainslie_mccolm", array_gain_db=directivity_db(n, pitch))
        return float(eq.detection_range(det["threshold_db"])), eq

    RATE_MBIT = 102.4e3 * 24 / 1e6                     # one channel at the boards' rate

    def describe(label, n, pitch, heights, source_level, pulse, platform, window=None, det=None):
        rows = len(heights)
        channels = n * rows
        di = directivity_db(n, pitch)
        det = det or detector(n, pitch, rows, pulse, window)
        out = {"hydrophones_per_row": n, "rows": rows, "channels": channels, "boards": int(np.ceil(channels / 8)), "pitch_m": pitch,
               "row_length_m": float(n * pitch), "pitch_wavelengths_at_6khz": float(pitch * F_HIGH / C), "heights_m": list(heights),
               "baseline_m": float(max(heights) - min(heights)), "beamwidth_deg": float(np.degrees(0.886 * lam / (n * pitch))),
               "gain_isotropic_noise_db": di, "gain_uncorrelated_noise_db": float(10 * np.log10(n)),
               "rows_detection_gain_db": det["rows_gain_db"], "detection_threshold_db": det["threshold_db"], "detection_threshold_high_db": det["threshold_high_db"],
               "detection_probability": det["probability"],
               "reverberation_margin_vs_32_at_half_wavelength_db": float(10 * np.log10(n * pitch / 4.8)),
               "source_level_db": source_level, "pulse_s": pulse, "blind_range_m": float(0.5 * C * pulse),
               "data_rate_mbit_s": float(channels * RATE_MBIT), "cables": channels + int(np.ceil(channels / 8)), "platform": platform}
        for key, ts in (("ship", 25.0), ("small", 5.0)):
            out[f"reach_{key}_m"], eq = reach(n, pitch, rows, source_level, pulse, ts, det)
            out[f"snr_{key}_db"] = {str(r): float(eq.snr(r)) for r in (500.0, 1000.0, 3000.0, 6000.0, 12000.0)}   # one row
        if rows > 1:
            out["elevation_sigma_deg_at_20_db"] = sigma_deg(list(heights), 20.0)
            sp = float(np.degrees(np.arcsin(min(C / (FC * out["baseline_m"]), 1.0))))
            out["repeat_spacing_deg"] = sp if C / (FC * out["baseline_m"]) < 1.0 else None
            out["highest_repeat_db"] = max(second_lobe_db(list(heights), a) for a in (0.0, 10.0, 20.0))
            depth = {}
            for r in (500.0, 1000.0, 3000.0, 6000.0, 12000.0):
                row_snr = out["snr_ship_db"][str(r)]
                if row_snr < 10.0:
                    depth[str(r)] = None
                    continue
                noise_part = r * np.radians(sigma_deg(list(heights), min(row_snr, 40.0)))
                depth[str(r)] = {"from_noise_m": float(noise_part), "roll_0p2_deg_m": float(r * np.radians(0.2)),
                                 "gradient_0p05_m": float(r * r * 0.05 / (2 * C)),
                                 "total_m": float(np.sqrt(noise_part**2 + (r * np.radians(0.2)) ** 2 + (r * r * 0.05 / (2 * C)) ** 2))}
            out["depth_sigma_ship"] = depth
        return out

    long_range = SonarModel.load("popoto_long_range")
    sl_full = float(long_range.source_level_db_rms)
    R["stages"] = {}
    stage_files = (("1: test AUV, 1 x 32", "popoto_testauv_1row"), ("2: static array, 2 x 32", "popoto_static_2row"),
                   ("3: static array, 3 x 32", "popoto_static_3row"),
                   ("4: full face, 1 x 32", "popoto_face_1row"),
                   ("5: full face, 2 x 32", "popoto_long_range_2row"), ("6: full face, 3 x 32", "popoto_long_range_3row"))
    print("\nSTAGES (each a shipped sonar file; SNR and reach from the link budget of Sec. 5: spherical spreading, noise only; detected where one row's SNR is what the chain's detector needs).")
    print("  Reach is where noise alone would stop detection; reverberation and the channel stop it earlier or later (the scenarios show which).")
    for label, name in stage_files:
        m = SonarModel.load(name)
        heights = sorted(set(np.round(-m.positions[:, 2], 3)))
        n = m.n_elements // len(heights)
        st = describe(label, n, float(m.config["array"]["pitch"]), heights, float(m.source_level_db_rms),
                      float(m.waveform_config["duration"]), dict(m.config.get("platform", {})), det=bs.detection(bs.describe_shipped(name)))    # the file's own array and window
        st["sonar"] = name
        R["stages"][label] = st
        line = (f"  {label:22s} {st['channels']:3d} hydrophones on {st['boards']:2d} boards, row {st['row_length_m']:.2f} m at {st['pitch_m']:.3f} m "
                f"({st['pitch_wavelengths_at_6khz']:.2f} wavelengths at 6 kHz), beam {st['beamwidth_deg']:.1f} deg, gain {st['gain_isotropic_noise_db']:.1f} dB in sea noise "
                f"({st['gain_uncorrelated_noise_db']:.1f} dB in the boards' own), pulse {st['pulse_s']:.1f} s (blind inside {st['blind_range_m']:.0f} m), "
                f"reach {st['reach_ship_m'] / 1e3:.1f} km ship / {st['reach_small_m'] / 1e3:.1f} km small target, {st['data_rate_mbit_s']:.0f} Mbit/s")
        print(line)
        if st["rows"] > 1:
            d_ = st["depth_sigma_ship"]
            print(f"      rows {st['baseline_m']:.2f} m apart: elevation {st['elevation_sigma_deg_at_20_db']:.2f} deg at 20 dB; repeats "
                  + ("none inside the sector" if st["highest_repeat_db"] <= -99 else f"{st['highest_repeat_db']:.1f} dB")
                  + "; depth sigma on the ship, from noise / with roll 0.2 deg and gradient known to 0.05 /s: "
                  + ", ".join(f"{float(r) / 1e3:g} km: " + ("-" if v is None else f"{v['from_noise_m']:.1f} / {v['total_m']:.0f} m") for r, v in d_.items()))
    # The prototype's span and separation as parameters.
    print("  The prototype's row over 2 m and over 3 m (32 hydrophones; same waveform as shipped), and the 0.15 m row for comparison:")
    R["prototype_span"] = {}
    proto = SonarModel.load("popoto_testauv_1row")
    p_sl, p_pulse = float(proto.source_level_db_rms), float(proto.waveform_config["duration"])
    p_window = proto.config["receive"]["window"]["range"]
    for label, n, pitch in (("2 m", 32, 0.0625), ("3 m", 32, 0.09), ("4.8 m (the full row)", 32, 0.15), ("3 m, 20 hydrophones at 0.15 m", 20, 0.15)):
        st = describe(label, n, pitch, [0.0], p_sl, p_pulse, {}, window=p_window)
        R["prototype_span"][label] = st
        print(f"    {label:30s} beam {st['beamwidth_deg']:.1f} deg, gain in sea noise {st['gain_isotropic_noise_db']:.1f} dB ({st['gain_uncorrelated_noise_db']:.1f} dB counted by hydrophones), "
              f"reach {st['reach_ship_m'] / 1e3:.1f} / {st['reach_small_m'] / 1e3:.1f} km")
    low = describe("3 m at 185 dB", 32, 0.09, [0.0], 185.0, p_pulse, {}, window=p_window)
    R["prototype_span"]["3 m, a 185 dB projector"] = low
    print(f"    The projector for the test AUV is not chosen. With {p_sl:.0f} dB as above; with 185 dB the 3 m row reaches {low['reach_ship_m'] / 1e3:.1f} / {low['reach_small_m'] / 1e3:.1f} km; "
          "each 12 dB less halves a noise-limited range of a few km.")
    print("    Closer than half a wavelength the hydrophones hear the same sea noise: 32 over 3 m gain what 20 at 0.15 m gain. The extra")
    print("    twelve still pay against the boards' own noise, against a dead channel, and they are the full system's count and wiring.")
    half_wave_f = C / (2.0 * 0.09)
    R["prototype_half_wavelength_frequency_hz"] = half_wave_f
    print(f"    A 0.09 m pitch is half a wavelength at {half_wave_f / 1e3:.1f} kHz: a higher band would fill the short array, but it is another projector and")
    print("    another band than the full face's; the prototype is there to prove this chain, so it keeps 4-6 kHz.")
    print("  ELEVATION AGAINST THE ROWS' VERTICAL SEPARATION (two rows; any mounting: the static array's frame is chosen from this curve):")
    R["baseline_sweep"] = {}
    for sep in (0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.75, 1.00, 1.50, 1.80, 2.00):
        h = [-sep / 2, sep / 2]
        free = float(np.degrees(el.unambiguous_elevation(sep, F_HIGH, C)))
        row = {"sigma_deg": {str(sn): sigma_deg(h, sn) for sn in snrs}, "repeat_free_half_sector_deg_at_6khz": free,
               "highest_repeat_db_in_sector": second_lobe_db(h, 0.0),
               "depth_sigma_m_at_300m_30db": float(300.0 * np.radians(sigma_deg(h, 30.0))), "depth_sigma_m_at_1km_20db": float(1000.0 * np.radians(sigma_deg(h, 20.0))),
               "clock_error_deg_per_us": float(np.degrees(C * 1e-6 / sep)), "row_out_error_deg_per_mm": float(np.degrees(1e-3 / sep))}
        R["baseline_sweep"][f"{sep:.2f}"] = row
        print(f"    {sep:.2f} m: " + " / ".join(f"{v:.2f}" for v in row["sigma_deg"].values()) + f" deg at 13 / 20 / 30 dB; no repeat within +-{free:.0f} deg; "
              f"highest repeat inside +-30 deg {row['highest_repeat_db_in_sector']:.1f} dB; depth {row['depth_sigma_m_at_300m_30db']:.1f} m at 300 m (30 dB), "
              f"{row['depth_sigma_m_at_1km_20db']:.0f} m at 1 km (20 dB); {row['clock_error_deg_per_us']:.2f} deg per us of clock error between rows")
    print("    Below about 0.15 m a second row gives the sign of the elevation and little more. Up to about half a wavelength at the top of the band")
    print("    there is no repeat at all; beyond it accuracy keeps improving in proportion and repeats appear, closer together the wider the rows.")

    # --- trade space ------------------------------------------------------------------------------
    L_ = args.array_length
    n_face = int(args.face_length // 0.15)
    k_today, k_fill, k_64, k_96, k_48 = "32 at 0.15 m (today)", f"{n_full} at 0.15 m (fills the {L_:.0f} m)", f"64 at {L_ / 64:.3f} m (fills the {L_:.0f} m)", f"96 at {L_ / 96:.3f} m (fills the {L_:.0f} m)", f"48 at {L_ / 48:.3f} m (fills the {L_:.0f} m)"
    print(f"\nTRADE SPACE on the face ({args.face_length:.0f} m x {args.face_height:.0f} m, rows within {L_:.0f} m, outer rows {d:.2f} m apart), 4-6 kHz, 197 dB, 1 s pulse. Per row along the face; rows up the face.")
    print("  What a row's length buys: beam (bearing resolution, and 10 log(length) less reverberation in the beam), gain against sea noise, reach.")
    R["trade_space"] = {"per_row": {}, "fixed_channels": {}, "rows": {}}
    print(f"    {'row':34s} length  beam   gain   reach ship/small   reverb margin   boards a row")
    for label, n, pitch in (("16 at 0.15 m", 16, 0.15), ("24 at 0.15 m", 24, 0.15), (k_today, 32, 0.15), (k_fill, n_full, 0.15),
                            (k_48, 48, L_ / 48), (k_64, 64, L_ / 64), (k_96, 96, L_ / 96),
                            (f"32 at {L_ / 32:.3f} m (sparse, fills the {L_:.0f} m)", 32, L_ / 32),
                            (f"{n_face} at 0.15 m (the whole {args.face_length:.0f} m, no margin)", n_face, 0.15)):
        st = describe(label, n, pitch, [0.0], sl_full, 1.0, {})
        rep_ = flank.ambiguity_report((np.arange(n) - (n - 1) / 2.0) * pitch, F_LOW, F_HIGH, C, steer_deg=30.0)
        st["wideband_lobe_steered_30_deg_db"] = rep_["wideband_lobe_db"]
        R["trade_space"]["per_row"][label] = st
        print(f"    {label:34s} {st['row_length_m']:5.2f} m {st['beamwidth_deg']:4.1f} deg {st['gain_isotropic_noise_db']:5.1f} dB  {st['reach_ship_m'] / 1e3:5.1f} / {st['reach_small_m'] / 1e3:4.1f} km"
              f"   {st['reverberation_margin_vs_32_at_half_wavelength_db']:+5.1f} dB        {int(np.ceil(n / 8)):2d}"
              + (f"   (repeats in bearing: {rep_['wideband_lobe_db']:.0f} dB when steered 30 deg)" if rep_["wideband_lobe_db"] > -12.5 else ""))
    t32, t64 = R["trade_space"]["per_row"][k_today], R["trade_space"]["per_row"][k_64]
    tf, t9 = R["trade_space"]["per_row"][k_fill], R["trade_space"]["per_row"][f"{n_face} at 0.15 m (the whole {args.face_length:.0f} m, no margin)"]
    R["trade_space"]["keys"] = {"today": k_today, "fills": k_fill, "row_64": k_64, "row_96": k_96, "row_48": k_48}
    R["trade_space"]["length_limit"] = {"array_length_m": L_, "face_length_m": args.face_length, "half_wavelength_count": [n_full, n_face],
                                        "gain_db": [tf["gain_isotropic_noise_db"], t9["gain_isotropic_noise_db"]],
                                        "beam_deg": [tf["beamwidth_deg"], t9["beamwidth_deg"]],
                                        "reach_ship_km": [tf["reach_ship_m"] / 1e3, t9["reach_ship_m"] / 1e3]}
    print(f"  The length limit {args.face_length:.0f} -> {L_:.0f} m: a filled row holds {n_face} -> {n_full} hydrophones at half a wavelength, gain {t9['gain_isotropic_noise_db']:.1f} -> {tf['gain_isotropic_noise_db']:.1f} dB, "
          f"beam {t9['beamwidth_deg']:.1f} -> {tf['beamwidth_deg']:.1f} deg, reach {t9['reach_ship_m'] / 1e3:.1f} -> {tf['reach_ship_m'] / 1e3:.1f} km (ship). The 32-hydrophone row (4.8 m) fits either.")
    R["trade_space"]["marginal_32_to_64_per_row"] = {"gain_db": t64["gain_isotropic_noise_db"] - t32["gain_isotropic_noise_db"],
                                                     "beam_deg": [t32["beamwidth_deg"], t64["beamwidth_deg"]],
                                                     "reach_ship_km": [t32["reach_ship_m"] / 1e3, t64["reach_ship_m"] / 1e3],
                                                     "reach_small_km": [t32["reach_small_m"] / 1e3, t64["reach_small_m"] / 1e3], "boards": 4}
    mg = R["trade_space"]["marginal_32_to_64_per_row"]
    print(f"  32 -> 64 a row: +{mg['gain_db']:.1f} dB, beam {mg['beam_deg'][0]:.1f} -> {mg['beam_deg'][1]:.1f} deg, noise-limited reach {mg['reach_ship_km'][0]:.1f} -> {mg['reach_ship_km'][1]:.1f} km (ship), "
          f"{mg['reach_small_km'][0]:.1f} -> {mg['reach_small_km'][1]:.1f} km (small), {t64['reverberation_margin_vs_32_at_half_wavelength_db']:.1f} dB less reverberation in the beam; 4 more boards a row.")
    print(f"  Inside {L_:.0f} m, 64 in a row are closer than half a wavelength and gain no more against sea noise than the {n_full} that fill it.")
    print(f"  Each doubling of a filled row is 3 dB and half the beam; in noise-limited range that is {100 * (mg['reach_ship_km'][1] / mg['reach_ship_km'][0] - 1):.0f}% at 13 km (spreading and absorption take the rest).")
    print(f"  Past the {L_:.0f} m ({n_full} at half a wavelength) more hydrophones in a row buy nothing against sea noise; sparser than half a wavelength")
    print("  the same count gives the narrower beam but full repeats in bearing when steered.")
    print("  What rows buy (32 a row, outer rows at the full height):")
    for rows, heights in ((1, [0.0]), (2, [-up, up]), (3, three_h), (4, [-up, best, -best, up])):
        st = describe(f"{rows} rows", 32, 0.15, heights, sl_full, 1.0, {})
        R["trade_space"]["rows"][str(rows)] = st
        print(f"    {rows} x 32: {st['channels']:3d} hydrophones, {st['boards']:2d} boards, reach {st['reach_ship_m'] / 1e3:.1f} / {st['reach_small_m'] / 1e3:.1f} km"
              + ("" if rows == 1 else f", elevation {st['elevation_sigma_deg_at_20_db']:.2f} deg at 20 dB, highest repeat {st['highest_repeat_db']:.1f} dB"))
    print(f"  A fourth row (here at the mirrored height) adds {face[4]['rows_gain_db'] - face[3]['rows_gain_db']:.1f} dB of detection and nothing else; accuracy is set by the outer rows.")
    print("  The same number of channels as one long row or as stacked rows of 32:")
    fixed = {32: [("1 x 32 at 0.15 m", 32, 0.15, [0.0])],
             64: [(f"1 x 64 at {L_ / 64:.3f} m", 64, L_ / 64, [0.0]), ("2 x 32 at 0.15 m", 32, 0.15, [-up, up])],
             96: [(f"1 x 96 at {L_ / 96:.3f} m", 96, L_ / 96, [0.0]), (f"2 x 48 at {L_ / 48:.3f} m", 48, L_ / 48, [-up, up]), ("3 x 32 at 0.15 m", 32, 0.15, three_h)]}
    for channels, options in fixed.items():
        for label, n, pitch, heights in options:
            st = describe(label, n, pitch, heights, sl_full, 1.0, {})
            R["trade_space"]["fixed_channels"][label] = st
            print(f"    {channels:2d} channels, {label:18s}: beam {st['beamwidth_deg']:.1f} deg, row gain {st['gain_isotropic_noise_db']:.1f} dB + {st['rows_detection_gain_db']:.1f} dB for the rows at the detector, "
                  f"reach {st['reach_ship_m'] / 1e3:.1f} / {st['reach_small_m'] / 1e3:.1f} km, "
                  + (f"elevation {st['elevation_sigma_deg_at_20_db']:.2f} deg at 20 dB (repeat {st['highest_repeat_db']:.1f} dB)" if len(heights) > 1 else "no elevation"))
    a, b = R["trade_space"]["fixed_channels"][f"1 x 64 at {L_ / 64:.3f} m"], R["trade_space"]["fixed_channels"]["2 x 32 at 0.15 m"]
    R["trade_space"]["one_by_64_vs_two_by_32"] = {"map_gain_db": [a["gain_isotropic_noise_db"], b["gain_isotropic_noise_db"] + b["rows_detection_gain_db"]],
                                                  "beam_deg": [a["beamwidth_deg"], b["beamwidth_deg"]], "reach_ship_km": [a["reach_ship_m"] / 1e3, b["reach_ship_m"] / 1e3]}
    print(f"  64 channels: one row of 64 gives {a['gain_isotropic_noise_db']:.1f} dB in the map against {b['gain_isotropic_noise_db'] + b['rows_detection_gain_db']:.1f} dB for two rows of 32, "
          f"a beam of {a['beamwidth_deg']:.1f} instead of {b['beamwidth_deg']:.1f} deg, {a['reverberation_margin_vs_32_at_half_wavelength_db']:.1f} dB less reverberation in the beam, and no elevation;\n  two rows of 32 give elevation to {b['elevation_sigma_deg_at_20_db']:.2f} deg at 20 dB (as candidates {spacing[1]:.1f} deg apart). Bearing and range against elevation: that is the trade.")
    print("  Is 64 needed? For bearing, range and tracks, no: one row of 32 is the design of Sec. 5. 64 are needed for whichever second thing is wanted:")
    print("  elevation (2 x 32; 3 x 32 = 96 to make it unambiguous), or a narrower beam (1 x 64, or better the 46 that fill the length).")
    print("  Another band moves every row of this table: pitch and row length scale with the wavelength (7 m hold 46 half-wavelengths")
    print("  at 5 kHz, 93 at 10 kHz), absorption and noise change the reach (tools/popoto_range_study.py), and the rows' repeat spacing scales too.")
    print("  Source level: each 6 dB is the same as 4 times the hydrophones in a row for noise-limited reach, and nothing against reverberation.")
    print("  Compute, data rate and storage go with the channel count (measured for 32, 64, 96: tools/popoto_elevation_bench.py); boards = channels / 8.")

    # --- what the smaller baseline costs ------------------------------------------------------------
    print(f"\nBASELINE {args.compare_baseline:.2f} -> {d:.2f} m (the owner's margin): what it costs.")
    R["baseline_comparison"] = {}
    for dd in (args.compare_baseline, d):
        u = dd / 2.0
        mids = np.round(np.arange(-u + 0.05, 0.0001, 0.05), 3)
        levels = [max(second_lobe_db([-u, float(p), u], a_) for a_ in (0.0, 10.0, 20.0)) for p in mids]
        i_best = int(np.argmin(levels))
        row = {"sigma_deg": {str(sn): sigma_deg([-u, u], sn) for sn in snrs}, "repeat_spacing_deg_at_5khz": float(np.degrees(np.arcsin(C / (FC * dd)))),
               "unambiguous_half_sector_deg": float(np.degrees(np.arcsin(C / (FC * dd)))) / 2.0,
               "two_rows_unambiguous_beyond_m_100m_water_one_bounce": float(200.0 / np.tan(np.radians(np.degrees(np.arcsin(C / (FC * dd))) / 2.0))),
               "best_middle_m": float(mids[i_best]), "best_middle_level_db": float(levels[i_best]),
               "clock_error_deg_per_us": float(np.degrees(C * 1e-6 / dd)), "row_out_error_deg_per_mm": float(np.degrees(1e-3 / dd)),
               "depth_sigma_m_at_1km_20db": float(1000.0 * np.radians(sigma_deg([-u, u], 20.0)))}
        R["baseline_comparison"][f"{dd:.2f}"] = row
        print(f"    {dd:.2f} m: elevation " + " / ".join(f"{v:.3f}" for v in row["sigma_deg"].values()) + f" deg at 13 / 20 / 30 dB; repeats every {row['repeat_spacing_deg_at_5khz']:.1f} deg; "
              f"two rows unambiguous beyond {row['two_rows_unambiguous_beyond_m_100m_water_one_bounce'] / 1e3:.1f} km in 100 m of water (one bounce); best middle row {row['best_middle_m']:+.2f} m "
              f"({row['best_middle_level_db']:.1f} dB); {row['clock_error_deg_per_us']:.3f} deg per us; {row['row_out_error_deg_per_mm']:.3f} deg per mm")
    print("  Accuracy and every mounting and clock tolerance scale with the baseline; the repeats move apart, which helps two rows a little.")

    # --- the projector on the face -------------------------------------------------------------------
    face = SonarModel.load("popoto_long_range_3row")
    tx = np.asarray(face.config["transmit"]["lever_arm"], dtype=float)
    dist = np.linalg.norm(face.positions - tx, axis=1)
    rxc = face.config["receive"]
    blast_db = sl_full - 20.0 * np.log10(dist.min())                                  # dB re 1 uPa at the nearest hydrophone
    volts = 10.0 ** ((blast_db + PEAK_OVER_RMS_DB + float(rxc["sensitivity_db"]) + float(rxc.get("gain_db", 0.0))) / 20.0)
    pulse = float(face.waveform_config["duration"])
    R["projector"] = {"lever_arm_m": [float(v) for v in tx], "nearest_hydrophone_m": float(dist.min()), "farthest_hydrophone_m": float(dist.max()),
                      "direct_level_db_at_nearest": float(blast_db), "direct_volts_at_adc": float(volts), "adc_full_scale_v": float(rxc["adc_full_scale"]),
                      "over_full_scale_db": float(20.0 * np.log10(volts / float(rxc["adc_full_scale"]))),
                      "direct_sound_ends_s": float(pulse + dist.max() / C), "blind_range_m": float(0.5 * C * pulse),
                      "far_field_of_row_m": float(2.0 * (32 * 0.15) ** 2 / lam),
                      "source_level_for_full_scale_db": float(sl_full - 20.0 * np.log10(volts / float(rxc["adc_full_scale"])))}
    pj = R["projector"]
    print(f"\nTHE PROJECTOR ON THE FACE: at {pj['lever_arm_m']} m from the array's centre (forward, out, down; a placeholder in the sonar file), {pj['nearest_hydrophone_m']:.2f} to {pj['farthest_hydrophone_m']:.2f} m from the hydrophones.")
    print(f"  Its direct sound is {pj['direct_level_db_at_nearest']:.0f} dB re 1 uPa at the nearest hydrophone: {pj['direct_volts_at_adc']:.0f} V at the ADC with the sonar file's sensitivity and gain, "
          f"{pj['over_full_scale_db']:.0f} dB over full scale ({pj['adc_full_scale_v']:.2f} V).")
    print(f"  The inputs are driven far into clipping for the whole pulse (to {pj['direct_sound_ends_s']:.3f} s); the plan clamps the preamp inputs and the software uses nothing inside {pj['blind_range_m']:.0f} m.")
    print("  How fast the inputs recover after the pulse is not known (unconfirmed); the simulator does not model the direct sound or the recovery.")
    print(f"  The chain takes each echo's path as projector -> target -> each hydrophone (the record carries the projector's lever arm), so the offset costs no accuracy;")
    print(f"  targets are in the far field of a row beyond {pj['far_field_of_row_m']:.0f} m, and the backprojection kernel is exact inside it too.")

    # --- recommendation ---------------------------------------------------------------------------
    w2, w3 = R["three_rows"]["wrong_pick"][two], R["three_rows"]["wrong_pick"][three]
    R["recommended"] = {"two_rows_m": [-up, up], "three_rows_m": three_h, "default": "three_rows"}
    print(f"\nRECOMMENDED for the face: three rows at {-up:+.2f}, {best:+.2f}, {up:+.2f} m (96 hydrophones, 12 boards); two rows at {-up:+.2f}, {up:+.2f} m (64, 8) shipped beside it.")
    print(f"  Same accuracy ({sigma_deg([-up, up], 20.0):.2f} deg at 20 dB). Searching 60 deg, two rows pick a wrong repeat {100 * w2['13.0'][0]:.0f}% of the time at 13 dB and {100 * w2['10.0'][0]:.0f}% at 10 dB "
          f"in clean noise, on a margin of {-lobes[0]:.1f} dB that a channel does not respect;\n  three rows {100 * w3['13.0'][0]:.1f}% and {100 * w3['10.0'][0]:.1f}% on a margin of {-best_level:.1f} dB. "
          f"Two rows are sound where water depth and range confine the arrivals to +-{half:.1f} deg, and as a first step: the third row can be added later.")
    if not args.no_write:
        path = ROOT / "docs" / "design" / "popoto_elevation_study.json"
        path.write_text(json.dumps(R, indent=2))
        print(f"\nwrote {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
