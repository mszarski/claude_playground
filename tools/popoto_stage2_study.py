#!/usr/bin/env python3
"""Stage 2 of the long-range array's reference design, in numbers: what each rung of the
sound-speed ladder buys for the depth of a target, what the decimate-first stage does to the
data rates, what blanking the projector's own sound costs in minimum range, and what a
navigation adapter has to supply.

  ladder     Illustrative, non-isothermal profiles DEFINED BY PARAMETERS HERE (they are not data
             from any place). For each: a vehicle and a target at given depths, at 1, 3, 6 and
             12 km. Truth is the first arrival by a ray integrated through the true profile
             (midpoint rule in range, 1 m steps, mirrored at the surface and the bottom; nothing
             shared with the chain's ray code). The arrival angle and travel time are taken as
             measured exactly, so that what is left is refraction alone. Each rung of the ladder
             (libs/live environment.hpp) then turns angle and time into a depth, with the sigma
             it reports:
               prior / local   straight line (no profile)
               family          an assumed isothermal gradient; the mixed-layer family with the right
                               parameters, and with its layer 25% too deep
               built           the profile the vehicle built itself: having sampled only 10 m about
                               its own depth, and having dived through the whole column
               supplied        the true profile
  decimation the filter and factor the stage picks for each stage's sonar (libs/live decimate.hpp),
             and the data rates and storage raw against decimated
  blanking   minimum range against the inputs' recovery time, for the 0.1 s and 1 s pulses
  navigation the fields a real unit's adapter must supply, and the figures to read from its datasheet

Writes docs/design/popoto_stage2.json (sections of other tools in that file are kept).
Usage: tools/popoto_stage2_study.py [--no-write]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

import gpusonar as gs  # noqa: E402
from gpusonar import elevation  # noqa: E402
from gpusonar.sim import make_pose  # noqa: E402
from gpusonar.sonar import SonarModel  # noqa: E402

core = gs.core
OUT = ROOT / "docs" / "design" / "popoto_stage2.json"
RANGES = (1000.0, 3000.0, 6000.0, 12000.0)
WATER = 200.0
GRADIENT_SIGMA = 0.05                 # the prior the deployment files ship (unconfirmed)
SENSOR_SIGMA = 0.1                    # m/s, illustrative: the built rung has no default for it
SUPPLIED_SIGMA = 0.1                  # m/s, illustrative: how well a supplied profile is taken to hold

# Illustrative profiles, by parameters (m, m/s, 1/s). Not measurements of any water.
PRESSURE = 0.017                      # the isothermal gradient the deployment files assume
PROFILES = {
    "isothermal": {"kind": "isothermal", "gradient_per_s": PRESSURE},
    "mixed layer 30 m over a strong thermocline (-0.15 /s for 40 m)":
        {"kind": "mixed_layer", "gradient_per_s": PRESSURE, "layer_depth_m": 30.0, "thermocline_gradient_per_s": -0.15, "thermocline_thickness_m": 40.0},
    "mixed layer 50 m over a weak thermocline (-0.05 /s for 60 m)":
        {"kind": "mixed_layer", "gradient_per_s": PRESSURE, "layer_depth_m": 50.0, "thermocline_gradient_per_s": -0.05, "thermocline_thickness_m": 60.0},
    "no mixed layer: -0.05 /s from the surface to 100 m":
        {"kind": "mixed_layer", "gradient_per_s": PRESSURE, "layer_depth_m": 1.0, "thermocline_gradient_per_s": -0.05, "thermocline_thickness_m": 99.0},
}
CASES = [(50.0, 20.0), (50.0, 120.0)]  # (vehicle depth, target depth)
C_SURFACE = 1500.0


def family_options(p):
    names = {"gradient_per_s": "family_gradient", "layer_depth_m": "layer_depth", "thermocline_gradient_per_s": "thermocline_gradient",
             "thermocline_thickness_m": "thermocline_thickness"}
    return dict({"family": p["kind"]}, **{names[k]: v for k, v in p.items() if k in names})


def true_profile(p):
    """Knots (depth, speed) of an illustrative profile with C_SURFACE at the surface."""
    depth, speed = core.live_family_profile(family_options(p), 0.0, C_SURFACE, WATER + 100.0)
    return np.array(depth), np.array(speed)


def shoot(depth, speed, z0, angles, r_max, water, dr=1.0, at=()):
    """Rays from depth z0 at elevation `angles` (rad, up positive) through c(z), linear between knots:
    (depth, travel time, bounces) of every ray at each range in `at`. Midpoint rule in range; mirrored at
    the surface and at the bottom. Independent of the chain's ray code (which steps along the path by Snell's law)."""
    grad = np.diff(speed) / np.diff(depth)
    c_of = lambda z: np.interp(z, depth, speed)                                         # noqa: E731
    g_of = lambda z: grad[np.clip(np.searchsorted(depth, z, side="right") - 1, 0, len(grad) - 1)]   # noqa: E731
    z = np.full(len(angles), float(z0))
    th = np.array(angles, dtype=float)
    t = np.zeros(len(angles))
    bounces = np.zeros(len(angles), dtype=int)
    out = {}
    marks = sorted(at)
    n = int(round(r_max / dr))
    for i in range(n):
        # z down, th up: dz/dr = -tan th; dth/dr = c'(z) / c(z); dt/dr = 1 / (c cos th)
        zm = z - 0.5 * dr * np.tan(th)
        thm = th + 0.5 * dr * g_of(z) / c_of(z)
        zm = np.clip(zm, 0.0, water)
        z = z - dr * np.tan(thm)
        th = th + dr * g_of(zm) / c_of(zm)
        t = t + dr / (c_of(zm) * np.cos(thm))
        up = z < 0.0
        z[up], th[up] = -z[up], -th[up]
        down = z > water
        z[down], th[down] = 2.0 * water - z[down], -th[down]
        bounces += up | down
        r = (i + 1) * dr
        while marks and r >= marks[0] - 1e-9:
            out[marks.pop(0)] = (z.copy(), t.copy(), bounces.copy())
    return out


def first_arrivals(depth, speed, z_own, z_target, ranges, water):
    """For each range: (one-way time, arrival elevation at the vehicle, bounces) of the earliest ray joining
    the vehicle and the target (reciprocity: shot from the vehicle), or None. A fan of rays finds where
    neighbours straddle the target's depth; each such pair is refined by a second, dense fan between
    them and kept only if a ray then passes within 0.2 m of the target (where neighbouring rays part,
    at a caustic or where one grazes a boundary, a straddle is not an arrival)."""
    angles = np.radians(np.linspace(-25.0, 25.0, 20001))
    fan = shoot(depth, speed, z_own, angles, max(ranges), water, at=ranges)
    out = {}
    for r in ranges:
        z, t, b = fan[r]
        miss = z - z_target
        pairs = np.nonzero((miss[:-1] * miss[1:] <= 0.0) & (b[:-1] == b[1:]) & (np.abs(z[1:] - z[:-1]) < 10.0))[0]
        best = None
        for i in sorted(pairs, key=lambda q: t[q])[:6]:                                 # earliest first
            fine = np.linspace(angles[i], angles[i + 1], 801)
            zf, tf, bf = shoot(depth, speed, z_own, fine, r, water, at=(r,))[r]
            k = int(np.argmin(np.abs(zf - z_target)))
            if abs(zf[k] - z_target) < 0.2:
                best = (float(tf[k]), float(fine[k]), int(bf[k]))
                break
        out[r] = best
    return out


def rung_profiles(p, depth, speed, z_own):
    """What each rung holds for one true profile: {label: (rung name, knots depth, speed, sigma) or None for a straight line}."""
    c_own = float(np.interp(z_own, depth, speed))
    out = {"prior / local (straight line)": None}

    def ladder(options, samples=(), supplied=((), ())):
        lad = core.SoundSpeedLadder(dict({"sample_sigma": SENSOR_SIGMA, "age_sigma_per_hour": 0.0, "distance_sigma_per_km": 0.0}, **options))
        for z in samples:
            lad.add(0, 0.0, 0.0, float(z), float(np.interp(z, depth, speed)))
        got = lad.evaluate(0, 0.0, 0.0, z_own, c_own, True, WATER, list(supplied[0]), list(supplied[1]), 0.0, GRADIENT_SIGMA)
        return got["rung"], got["depth"], got["speed"], got["sigma"]

    out["family: isothermal (assumed)"] = ladder({"ladder": ["family"], "family": "isothermal", "family_gradient": PRESSURE})
    if p["kind"] == "mixed_layer":
        out["family: mixed layer, right parameters"] = ladder(dict({"ladder": ["family"]}, **family_options(p)))
        wrong = dict(p, layer_depth_m=1.25 * p["layer_depth_m"])
        out["family: mixed layer, layer 25% too deep"] = ladder(dict({"ladder": ["family"]}, **family_options(wrong)))
    out["built: sampled 10 m about its own depth"] = ladder({"ladder": ["built"]}, np.arange(z_own - 5.0, z_own + 5.01, 1.0))
    out["built: dived through the whole column"] = ladder({"ladder": ["built"]}, np.arange(1.0, WATER, 1.0))
    out["supplied: the true profile"] = ladder({"ladder": ["supplied"], "supplied_sigma": SUPPLIED_SIGMA}, supplied=(depth, speed))
    return out


def ladder_study():
    R = {"note": "illustrative profiles defined by parameters in tools/popoto_stage2_study.py; not data", "water_depth_m": WATER,
         "gradient_sigma_per_s": GRADIENT_SIGMA, "sensor_sigma_m_s": SENSOR_SIGMA, "supplied_sigma_m_s": SUPPLIED_SIGMA,
         "surface_speed_m_s": C_SURFACE, "ranges_m": list(RANGES), "profiles": {}}
    for name, p in PROFILES.items():
        depth, speed = true_profile(p)
        entry = {"parameters": p, "cases": {}}
        for z_own, z_target in CASES:
            arrivals = first_arrivals(depth, speed, z_own, z_target, RANGES, WATER)
            c_own = float(np.interp(z_own, depth, speed))
            rungs = rung_profiles(p, depth, speed, z_own)
            case = {"vehicle_depth_m": z_own, "target_depth_m": z_target, "arrival": {}, "rungs": {k: {"error_m": {}, "sigma_m": {}} for k in rungs}}
            for r in RANGES:
                if arrivals[r] is None:
                    continue
                t, angle, bounces = arrivals[r]
                slant = c_own * t
                case["arrival"][str(int(r))] = {"elevation_deg": float(np.degrees(angle)), "bounces": bounces, "slant_m": slant}
                for label, prof in rungs.items():
                    if prof is None:
                        got = elevation.fold_depth(z_own - slant * np.sin(angle), WATER)[0]
                        sigma = elevation.refraction_sigma(slant, c_own, GRADIENT_SIGMA)
                    else:
                        _, d, c, s = prof
                        got = core.live_ray_depth(z_own, angle, slant, d, c, WATER)
                        sigma = core.live_profile_depth_sigma(z_own, angle, slant, d, c, s, GRADIENT_SIGMA, WATER)
                    case["rungs"][label]["error_m"][str(int(r))] = float(got - z_target)
                    case["rungs"][label]["sigma_m"][str(int(r))] = float(sigma)
            entry["cases"][f"vehicle {z_own:.0f} m, target {z_target:.0f} m"] = case
        R["profiles"][name] = entry
    # By rung, over every profile and case: how wrong the depth is, and whether the sigma reported owns up to it.
    R["summary"] = {}
    labels = []
    for entry in R["profiles"].values():
        for case in entry["cases"].values():
            labels += [k for k in case["rungs"] if k not in labels]
    for label in labels:
        row = {}
        for r in RANGES:
            key = str(int(r))
            pairs = [(abs(c["rungs"][label]["error_m"][key]), c["rungs"][label]["sigma_m"][key]) for e in R["profiles"].values()
                     for c in e["cases"].values() if label in c["rungs"] and key in c["rungs"][label]["error_m"]]
            err = np.array([p_[0] for p_ in pairs])
            row[key] = {"cases": len(pairs), "median_abs_error_m": float(np.median(err)), "worst_abs_error_m": float(err.max()),
                        "within_two_sigma": int(sum(e <= 2.0 * s_ + 0.5 for e, s_ in pairs)),
                        "median_sigma_m": float(np.median([p_[1] for p_ in pairs]))}
        R["summary"][label] = row
    return R


STAGES = {"test AUV, one row": "popoto_testauv_1row_clocks", "static array, two rows": "popoto_static_2row_clocks",
          "static array, three rows": "popoto_static_3row_clocks", "face, two rows": "popoto_long_range_2row_clocks",
          "face, three rows": "popoto_long_range_3row_clocks"}


def decimation_study(window=(500.0, 13000.0), ping_interval=17.9):
    """The plan the stage makes for each stage's sonar, and data rates raw against decimated. Bytes per
    sample: 3 on the wire and 4 in the container for raw 24-bit samples; 8 for a complex float32 pair."""
    R = {"window_m": list(window), "ping_interval_s": ping_interval, "bytes_per_sample": {"raw_wire": 3, "raw_container": 4, "decimated": 8},
         "stages": {}}
    for label, name in STAGES.items():
        sonar = SonarModel.load(name, {"receive": {"window": {"range": list(window)}}})
        rec = sonar.make_record(make_pose())
        w = rec.transmit.waveform
        s = rec.samples
        entry = {"sonar": name, "channels": int(s.n_channels), "boards": len(sonar.boards), "input_rate_hz": float(s.sample_rate),
                 "band_hz": [float(min(w.f_start, w.f_stop)), float(max(w.f_start, w.f_stop))], "samples_per_ping_raw": int(s.n_samples)}
        for key, options in (("default", {}), ("factor_25", {"output_rate": s.sample_rate / 25.0})):
            plan = core.live_decimate_plan(rec, options)
            n_out = (s.n_samples - 1) // plan["factor"] + 1
            raw_ping, dec_ping = 4 * s.n_samples * s.n_channels, 8 * n_out * s.n_channels
            pings_per_hour = 3600.0 / ping_interval
            entry[key] = {"factor": int(plan["factor"]), "output_rate_hz": plan["output_rate"], "taps": len(plan["taps"]),
                          "passband_hz": plan["passband"], "stopband_hz": plan["stopband"], "delay_ms": 1e3 * plan["delay_s"],
                          "samples_per_ping": int(n_out), "raw_mb_per_ping": raw_ping / 1e6, "decimated_mb_per_ping": dec_ping / 1e6,
                          "raw_gb_per_hour": raw_ping * pings_per_hour / 1e9, "decimated_gb_per_hour": dec_ping * pings_per_hour / 1e9,
                          "storage_ratio": raw_ping / dec_ping, "sample_ratio": s.n_samples / n_out,
                          "raw_stream_mbit_s": 24 * s.sample_rate * s.n_channels / 1e6,
                          "decimated_stream_mbit_s": 64 * plan["output_rate"] * s.n_channels / 1e6}
        R["stages"][label] = entry
    return R


def blanking_study(sound_speed=1500.0):
    """Minimum range against the inputs' recovery time: c (pulse + recovery + taper) / 2."""
    R = {"sound_speed_m_s": sound_speed, "taper_s": 0.005, "formula": "min range = c (pulse length + recovery_s + taper_s) / 2", "pulses": {}}
    for pulse in (0.1, 1.0):
        R["pulses"][str(pulse)] = {str(rec): 0.5 * sound_speed * (pulse + rec + R["taper_s"]) for rec in (0.0, 0.005, 0.02, 0.05, 0.1, 0.2, 0.5)}
    return R


NAVIGATION = {
    "datagram_fields_required": {
        "time_ns": "time of validity of the fix, ns since the Unix epoch, on the same time base as the boards' capture time",
        "roll, pitch, yaw": "rad; rotation z-y-x from north-east-down to the vehicle's forward-right-down axes",
        "x, y, z": "m north, east, down in one frame for the whole mission; z minus elevation.surface_z is the depth",
    },
    "datagram_fields_wanted": {
        "vx, vy, vz": "m/s over ground, north-east-down (the Doppler log's velocity turned by the attitude)",
        "altitude, bottom_lock": "m above the bottom from the Doppler log, and whether it has the bottom: water depth = depth + altitude",
        "sound_speed": "m/s at the vehicle: range scaling, the local rung of the ladder, and the samples the vehicle's own profile is built from",
        "roll_sigma": "rad: goes into every elevation and depth sigma in place of elevation.roll_sigma_deg when larger",
        "depth_sigma, altitude_sigma": "m: the water depth's sigma",
        "pitch_sigma, yaw_sigma, position_sigma, velocity_sigma, sound_speed_sigma, water_depth_sigma": "carried into the record; not used by the chain yet",
    },
    "from_the_datasheet_into_the_build_spec": {       # every value unconfirmed: we hold neither datasheet nor interface document
        "attitude_output_rate_hz": None, "roll_pitch_accuracy_deg": None, "heading_accuracy_deg": None,
        "time_stamp_accuracy_s": None, "output_latency_s": None, "time_base": None, "message_format": None,
        "velocity_accuracy_m_s": None, "altitude_accuracy_m": None, "altitude_range_m": [None, None],
        "depth_accuracy_m": None, "sound_speed_sensor": None, "sound_speed_accuracy_m_s": None,
        "mounting_alignment_to_the_array_deg": None,
    },
    "unconfirmed": True,
    "attitude_rate_rule": "a roll of amplitude A at frequency f, sampled every dt and interpolated linearly, is wrong by at most A (2 pi f dt)^2 / 8",
}


def attitude_rate_table():
    """Worst error of linear interpolation between attitude samples, for a few roll motions and output rates."""
    out = {}
    for amplitude_deg, period_s in ((2.0, 6.0), (5.0, 8.0), (10.0, 10.0)):
        row = {}
        for rate in (1.0, 5.0, 10.0, 20.0, 50.0, 100.0):
            row[str(rate)] = float(amplitude_deg * (2.0 * np.pi / period_s / rate) ** 2 / 8.0)
        out[f"roll +-{amplitude_deg:g} deg, period {period_s:g} s"] = row
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args()
    R = json.loads(OUT.read_text()) if OUT.exists() else {}
    print("LADDER: depth error (and the sigma reported) by rung; illustrative profiles, 200 m of water")
    R["profile_ladder"] = lad = ladder_study()
    for name, entry in lad["profiles"].items():
        print(f"  {name}")
        for case_name, case in entry["cases"].items():
            arrivals = ", ".join(f"{int(r) // 1000} km {a['elevation_deg']:+.2f} deg ({a['bounces']} bounces)" for r, a in case["arrival"].items())
            print(f"    {case_name}: first arrival {arrivals}")
            for label, rung in case["rungs"].items():
                cells = ", ".join(f"{int(r) // 1000} km {e:+.1f} ({rung['sigma_m'][r]:.1f})" for r, e in rung["error_m"].items())
                print(f"      {label:44s} {cells}")
    print("  SUMMARY over every profile and case: median |error| / worst |error| m (cases within two sigma of the reported sigma)")
    for label, row in lad["summary"].items():
        print(f"    {label:44s} " + ", ".join(f"{int(r) // 1000} km {v['median_abs_error_m']:.1f} / {v['worst_abs_error_m']:.0f} ({v['within_two_sigma']} of {v['cases']})"
                                              for r, v in row.items()))
    print("DECIMATION: the plan and the data rates, window 0.5-13 km")
    R.setdefault("decimation", {})["plan"] = dec = decimation_study()
    for label, e in dec["stages"].items():
        d, f = e["default"], e["factor_25"]
        print(f"  {label:26s} {e['channels']:3d} ch: factor {d['factor']} ({d['output_rate_hz']:.0f} Hz, {d['taps']} taps, {d['delay_ms']:.2f} ms), "
              f"{d['raw_mb_per_ping']:.0f} -> {d['decimated_mb_per_ping']:.0f} MB a ping, {d['raw_gb_per_hour']:.0f} -> {d['decimated_gb_per_hour']:.1f} GB an hour; "
              f"factor 25: {f['decimated_mb_per_ping']:.0f} MB, {f['decimated_gb_per_hour']:.1f} GB an hour")
    R["blanking"] = dict(R.get("blanking") or {}, **blanking_study())
    print("BLANKING: minimum range (m) against recovery time (s)")
    for pulse, row in R["blanking"]["pulses"].items():
        print(f"  pulse {pulse} s: " + ", ".join(f"{k}: {v:.0f}" for k, v in row.items()))
    R["navigation"] = dict(NAVIGATION, attitude_interpolation_error_deg=attitude_rate_table())
    print("NAVIGATION: worst attitude interpolation error (deg) by output rate (Hz)")
    for k, row in R["navigation"]["attitude_interpolation_error_deg"].items():
        print(f"  {k}: " + ", ".join(f"{r}: {v:.4f}" for r, v in row.items()))
    if not args.no_write:
        OUT.write_text(json.dumps(R, indent=2))
        print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
