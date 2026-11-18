#!/usr/bin/env python3
"""Depth through the profile the vehicle holds: what each rung of the sound-speed ladder buys the two depth estimators
once both trace their rays through it (libs/live raytrace.hpp), and what the profile ensemble adds where the single
tilt's sigma was too small.

  Waters   the stage-2 study's four illustrative profiles (200 m of water, vehicle at 50 m, targets at 20 and 120 m) and the
           deep one (1000 m, vehicle at 225 m, targets at 40 and 150 m). DEFINED BY PARAMETERS; not data from any place.
  Truth    an independent ray integration through the true profile (midpoint rule in range, 1 m steps, mirrored at the
           boundaries, surface and bottom reflections counted apart; nothing shared with the chain's ray code): the
           earliest path that touches no boundary and the earliest once-surface-reflected path, at 1, 3, 6 and 12 km.
  Measured taken as exact, so that what is left is the profile alone:
             multipath  the two-way delay of the echo reflected at the surface both ways behind the direct one, and the
                        direct echo's two-way time (what one row measures)
             elevation  the first arrival's angle and its travel time (what stacked rows measure)
  Rungs    prior / local (nothing but the sound speed at the vehicle), family: isothermal assumed, built from 10 m about the
           vehicle's depth, built from a dive through the column, supplied: the true profile
  For each the depth the estimator gives through that rung's profile, the sigma it reports (the rung's claim as one tilt
  about the vehicle, through the rays), whether truth is within two sigma, and, with the ensemble
  (elevation.profile.ensemble: the unsampled water continued with each of GRADIENTS for EXTENT m), whether truth is within
  the ensemble's span. Where the true water has no direct or no surface path the case is counted as such; where the rung's
  profile has none the estimator says so and gives no depth.
  Fusion   multipath and elevation combined as the chain does (libs/live multipath_fuse), their profile errors taken as one.
  Bellhop  (--bellhop, needs GPUSONAR_BELLHOP) simulated pings through the deep profile: the estimator itself on one row.

Writes docs/design/refraction_depth_study.json.
Usage: tools/refraction_depth_study.py [--bellhop] [--no-write]
"""

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

import gpusonar as gs  # noqa: E402
from gpusonar import multipath_depth as mp  # noqa: E402

core = gs.core
OUT = ROOT / "docs" / "design" / "refraction_depth_study.json"
RANGES = (1000.0, 3000.0, 6000.0, 12000.0)
PRESSURE = 0.017
GRADIENT_SIGMA = 0.05                 # the prior the deployment files ship (unconfirmed)
SENSOR_SIGMA = 0.1                    # m/s, illustrative
GRADIENTS = [-0.15, -0.05, PRESSURE]  # the ensemble's: what the unsampled water is thought able to hold here (illustrative)
EXTENT = [25.0, 50.0, 100.0]          # m: how far such a gradient may run beyond the sampled water (illustrative)
FC, BAND = 5000.0, 2000.0


def family(p, bottom):
    d, c = core.live_family_profile(p, 0.0, 1500.0, bottom)
    return np.array(d), np.array(c)


ISO = {"family": "isothermal", "family_gradient": PRESSURE}
STRONG = {"family": "mixed_layer", "family_gradient": PRESSURE, "layer_depth": 30.0, "thermocline_gradient": -0.15, "thermocline_thickness": 40.0}
WATERS = [("isothermal", ISO, 200.0, 50.0, (20.0, 120.0)),
          ("mixed layer 30 m over a strong thermocline (-0.15 /s for 40 m)", STRONG, 200.0, 50.0, (20.0, 120.0)),
          ("mixed layer 50 m over a weak thermocline (-0.05 /s for 60 m)",
           {"family": "mixed_layer", "family_gradient": PRESSURE, "layer_depth": 50.0, "thermocline_gradient": -0.05, "thermocline_thickness": 60.0}, 200.0, 50.0, (20.0, 120.0)),
          ("no mixed layer: -0.05 /s from the surface to 100 m",
           {"family": "mixed_layer", "family_gradient": PRESSURE, "layer_depth": 1.0, "thermocline_gradient": -0.05, "thermocline_thickness": 99.0}, 200.0, 50.0, (20.0, 120.0)),
          ("deep water: mixed layer 30 m, strong thermocline, pressure gradient to 1000 m", STRONG, 1000.0, 225.0, (40.0, 150.0))]


def shoot(depth, speed, z0, angles, ranges, water, dr=1.0):
    """Rays from depth z0 at elevation `angles` (rad, UP positive): {range: (depth, time, surface bounces, bottom bounces)}.
    Midpoint rule in range. Independent of the chain's ray code."""
    grad = np.diff(speed) / np.diff(depth)
    c_of = lambda z: np.interp(z, depth, speed)                                         # noqa: E731
    g_of = lambda z: grad[np.clip(np.searchsorted(depth, z, side="right") - 1, 0, len(grad) - 1)]   # noqa: E731
    z, th, t = np.full(len(angles), float(z0)), np.array(angles, dtype=float), np.zeros(len(angles))
    ns, nb = np.zeros(len(angles), dtype=int), np.zeros(len(angles), dtype=int)
    out, marks = {}, sorted(ranges)
    for i in range(int(round(max(ranges) / dr))):
        zm = np.clip(z - 0.5 * dr * np.tan(th), 0.0, water)
        thm = th + 0.5 * dr * g_of(z) / c_of(z)
        z = z - dr * np.tan(thm)
        th = th + dr * g_of(zm) / c_of(zm)
        t = t + dr / (c_of(zm) * np.cos(thm))
        up, down = z < 0.0, z > water
        z[up], th[up] = -z[up], -th[up]
        z[down], th[down] = 2.0 * water - z[down], -th[down]
        ns += up
        nb += down
        while marks and (i + 1) * dr >= marks[0] - 1e-9:
            out[marks.pop(0)] = (z.copy(), t.copy(), ns.copy(), nb.copy())
    return out


def true_paths(depth, speed, z_own, z_target, water):
    """{range: {"direct": (time, elevation) or None, "surface": (time, elevation) or None}}: the earliest path of each kind,
    by the independent integration (a coarse fan, then a dense one between the neighbours that straddle the target)."""
    angles = np.radians(np.linspace(-25.0, 25.0, 20001))
    fan = shoot(depth, speed, z_own, angles, RANGES, water)
    out = {}
    for r in RANGES:
        z, t, ns, nb = fan[r]
        miss = z - z_target
        got = {}
        for kind, want in (("direct", 0), ("surface", 1)):
            ok = (ns[:-1] == want) & (ns[1:] == want) & (nb[:-1] == 0) & (nb[1:] == 0)
            pairs = np.nonzero((miss[:-1] * miss[1:] <= 0.0) & ok & (np.abs(z[1:] - z[:-1]) < 20.0))[0]
            best = None
            for i in sorted(pairs, key=lambda q: t[q])[:4]:
                fine = np.linspace(angles[i], angles[i + 1], 801)
                zf, tf, nsf, nbf = shoot(depth, speed, z_own, fine, (r,), water)[r]
                keep = (nsf == want) & (nbf == 0)
                if not keep.any():
                    continue
                k = np.flatnonzero(keep)[np.argmin(np.abs(zf[keep] - z_target))]
                if abs(zf[k] - z_target) < 0.2:
                    # the time at the target's depth itself: along the wavefront dt/dz = -sin(elevation) / c (z down, elevation up)
                    best = (float(tf[k]), float(fine[k]))
                    break
            got[kind] = best
        out[r] = got
    return out


def rungs_of(depth, speed, z_own, water):
    """{label: the water profile each rung hands the rays (core.live_water_profile), with the ensemble's members}."""
    c_own = float(np.interp(z_own, depth, speed))
    ens = {"ensemble": True, "ensemble_gradients": GRADIENTS, "ensemble_extents": EXTENT}

    def ladder(options, samples=(), supplied=((), ())):
        o = dict({"sample_sigma": SENSOR_SIGMA, "age_sigma_per_hour": 0.0, "distance_sigma_per_km": 0.0}, **options, **ens)
        lad = core.SoundSpeedLadder(o)
        for z in samples:
            lad.add(0, 0.0, 0.0, float(z), float(np.interp(z, depth, speed)))
        got = lad.evaluate(0, 0.0, 0.0, z_own, c_own, True, water, list(supplied[0]), list(supplied[1]), 0.0, GRADIENT_SIGMA)
        return core.live_water_profile(o, got, [], [], False, 0.0, GRADIENT_SIGMA, z_own, c_own, water)

    return {"prior / local: nothing but the sound speed at the vehicle": ladder({"ladder": ["local"]}),
            "family: isothermal (assumed)": ladder({"ladder": ["family"], "family": "isothermal", "family_gradient": PRESSURE}),
            "built: sampled 10 m about its own depth": ladder({"ladder": ["built"]}, np.arange(z_own - 5.0, z_own + 5.01, 1.0)),
            "built: dived through the whole column": ladder({"ladder": ["built"]}, np.arange(1.0, min(water, 500.0), 1.0)),
            "supplied: the true profile": ladder({"ladder": ["supplied"], "supplied_sigma": SENSOR_SIGMA}, supplied=(depth, speed))}


def multipath_through(water_profile, z_own, slant, tau, water):
    """The depth whose both-ways surface delay through `water_profile` is `tau`, with the sigma and the ensemble's
    depths: (depth, sigma, [ensemble], reason). The estimator's own mapping (libs/live multipath_rays.cpp), noise-free."""
    g = {"range_m": slant, "array_depth": z_own, "projector_depth": z_own, "sound_speed": float(np.interp(z_own, water_profile["depth"], water_profile["speed"])),
         "f_centre": FC, "bandwidth": BAND, "water_depth": water, "profile": water_profile}
    o = dict(mp.cpp_options({"gradient_sigma_per_s": GRADIENT_SIGMA, "max_depth_m": min(water, 500.0)}), rays=1)
    grid = np.linspace(0.5, min(water, 500.0) - 0.5, 1200)
    info = core.live_multipath_rays(g, o, [float(z) for z in grid])
    paths = np.array([i["paths"] for i in info])
    if not (paths & 1).any():
        return None, None, [], "no_direct_path"
    if not (paths == 3).any():
        return None, None, [], "no_surface_path"
    d = np.array([i["surface_delay"] for i in info])
    ok = np.isfinite(d)
    cross = np.flatnonzero(ok[:-1] & ok[1:] & ((d[:-1] - tau) * (d[1:] - tau) <= 0.0))
    if cross.size == 0:
        return None, None, [], "no_such_delay"
    k = int(cross[0])
    z = float(grid[k] + (grid[k + 1] - grid[k]) * (tau - d[k]) / (d[k + 1] - d[k]))
    one = core.live_multipath_rays(g, o, [z])[0]
    if one["paths"] != 3:
        return z, float("inf"), [], ""
    return z, float(one["sigma"]), [float(v) for v in one["ensemble"]], ""


def study():
    R = {"note": "illustrative profiles defined by parameters in tools/refraction_depth_study.py; not data", "ranges_m": list(RANGES),
         "gradient_sigma_per_s": GRADIENT_SIGMA, "sensor_sigma_m_s": SENSOR_SIGMA, "ensemble": {"gradients_per_s": GRADIENTS, "extent_m": EXTENT}, "cases": []}
    for name, p, water, z_own, targets in WATERS:
        depth, speed = family(p, water + 100.0)
        c_own = float(np.interp(z_own, depth, speed))
        rungs = rungs_of(depth, speed, z_own, water)
        for z_t in targets:
            truth = true_paths(depth, speed, z_own, z_t, water)
            for r in RANGES:
                d_, s_ = truth[r]["direct"], truth[r]["surface"]
                case = {"water": name, "vehicle_depth_m": z_own, "target_depth_m": z_t, "range_m": r, "direct_path": d_ is not None, "surface_path": s_ is not None,
                        "rungs": {}}
                for label, w in rungs.items():
                    e = {}
                    if d_ is not None:
                        # stacked rows: the first arrival's angle and time
                        slant, angle = c_own * d_[0], d_[1]
                        z = core.ray_depth_arcs(z_own, angle, slant, w["depth"], w["speed"], water)
                        sig = core.live_profile_depth_sigma(z_own, angle, slant, w["depth"], w["speed"], w["sigma"] or [1e9] * len(w["depth"]), GRADIENT_SIGMA, water, True)
                        zs = [z] + [core.ray_depth_arcs(z_own, angle, slant, w["depth"], m, water) for m in w["ensemble"]]
                        e["elevation"] = {"depth_m": z, "error_m": z - z_t, "sigma_m": sig, "ensemble_m": [min(zs), max(zs)]}
                    if d_ is not None and s_ is not None:
                        slant, tau = c_own * d_[0], 2.0 * (s_[0] - d_[0])
                        z, sig, ens, reason = multipath_through(w, z_own, slant, tau, water)
                        if z is None:
                            e["multipath"] = {"depth_m": None, "reason": reason}
                        else:
                            zs = [z] + [v for v in ens if np.isfinite(v)]
                            e["multipath"] = {"depth_m": z, "error_m": z - z_t, "sigma_m": sig, "ensemble_m": [min(zs), max(zs)],
                                              "ensemble_complete": bool(all(np.isfinite(v) for v in ens))}
                        if z is not None and np.isfinite(sig) and "elevation" in e and sig > 0.0 and e["elevation"]["sigma_m"] > 0.0:
                            # the chain's fusion: here both sigmas are the profile's, one error in both
                            f = core.live_multipath_fuse(z, sig, e["elevation"]["depth_m"], e["elevation"]["sigma_m"], [], 3.0, sig, e["elevation"]["sigma_m"])
                            e["fused"] = {"depth_m": f["depth"], "error_m": f["depth"] - z_t, "sigma_m": f["sigma"], "consistent": bool(f["consistent"])}
                    case["rungs"][label] = e
                R["cases"].append(case)
    R["summary"] = summarise(R["cases"])
    R["stage2_rerun"] = stage2_rerun()
    return R


def stage2_rerun():
    """The stage-2 ladder study again (tools/popoto_stage2_study.py: its profiles, cases and independent first arrivals,
    bounces included), the depth now by the arcs, with what the ensemble adds: per rung and range, how often truth is
    within two sigma of the single tilt and how often within the ensemble's span."""
    sys.path.insert(0, str(ROOT / "tools"))
    import popoto_stage2_study as S2
    rows = {}
    for name, p in S2.PROFILES.items():
        depth, speed = S2.true_profile(p)
        for z_own, z_target in S2.CASES:
            arrivals = S2.first_arrivals(depth, speed, z_own, z_target, S2.RANGES, S2.WATER)
            c_own = float(np.interp(z_own, depth, speed))
            rungs = rungs_of(depth, speed, z_own, S2.WATER)
            for r in S2.RANGES:
                if arrivals[r] is None:
                    continue
                t, angle, bounces = arrivals[r]
                slant = c_own * t
                for label, w in rungs.items():
                    z = core.ray_depth_arcs(z_own, angle, slant, w["depth"], w["speed"], S2.WATER)
                    sig = core.live_profile_depth_sigma(z_own, angle, slant, w["depth"], w["speed"], w["sigma"] or [1e9] * len(w["depth"]), GRADIENT_SIGMA, S2.WATER, True)
                    zs = [z] + [core.ray_depth_arcs(z_own, angle, slant, w["depth"], m, S2.WATER) for m in w["ensemble"]]
                    e = rows.setdefault(label, {}).setdefault(str(int(r)), {"cases": 0, "within_two_sigma": 0, "within_ensemble": 0, "within_either": 0, "errors_m": [],
                                                                            "sigmas_m": [], "spans_m": []})
                    in_sigma, in_span = abs(z - z_target) <= 2.0 * sig + 0.5, min(zs) - 0.5 <= z_target <= max(zs) + 0.5
                    e["cases"] += 1
                    e["within_two_sigma"] += int(in_sigma)
                    e["within_ensemble"] += int(in_span)
                    e["within_either"] += int(in_sigma or in_span)
                    e["errors_m"].append(float(z - z_target))
                    e["sigmas_m"].append(float(sig))
                    e["spans_m"].append([float(min(zs)), float(max(zs))])
    return rows


def summarise(cases):
    """By estimator, rung and range, over every water and target: how wrong, what sigma, how often truth is inside
    two sigma and inside the ensemble's span."""
    out = {}
    labels = list(cases[0]["rungs"])
    for est in ("multipath", "elevation", "fused"):
        out[est] = {}
        for label in labels:
            row = {}
            for r in RANGES:
                items = [c["rungs"][label].get(est) for c in cases if c["range_m"] == r]
                with_path = [i for i in items if i is not None]
                given = [i for i in with_path if i.get("depth_m") is not None]
                err = np.array([abs(i["error_m"]) for i in given])
                entry = {"cases_with_the_paths": len(with_path), "depth_given": len(given), "said_no_path": len(with_path) - len(given)}
                if given:
                    sig = np.array([i["sigma_m"] for i in given])
                    entry.update(median_abs_error_m=float(np.median(err)), worst_abs_error_m=float(err.max()), median_sigma_m=float(np.median(sig)),
                                 within_two_sigma=int(np.sum(err <= 2.0 * sig + 0.5)))
                    if est != "fused":
                        entry["within_ensemble"] = int(sum(i["ensemble_m"][0] - 0.5 <= i["depth_m"] - i["error_m"] <= i["ensemble_m"][1] + 0.5 for i in given))
                row[str(int(r))] = entry
            out[est][label] = row
    out["true_water"] = {str(int(r)): {"cases": sum(c["range_m"] == r for c in cases), "direct_path": sum(c["range_m"] == r and c["direct_path"] for c in cases),
                                        "direct_and_surface": sum(c["range_m"] == r and c["direct_path"] and c["surface_path"] for c in cases)} for r in RANGES}
    return out


def bellhop_pings():
    """The estimator itself on Bellhop-simulated pings of the one-row face sonar through the deep profile: vehicle at 225 m,
    targets at 150 and 40 m, with each rung's profile."""
    sys.path.insert(0, str(ROOT / "tests" / "python"))
    import test_multipath_depth as T
    depth, speed = [0.0, 30.0, 70.0, 1000.0], [1500.0, 1500.5, 1494.5, 1510.3]
    own, water = 225.0, 1000.0
    rungs = rungs_of(np.array(depth), np.array(speed), own, water)
    rows = []
    for target in (150.0, 40.0):
        for rng, ts in ((1000.0, -10.0), (3000.0, 5.0), (6000.0, 18.0), (12000.0, 30.0)):
            cfg = T.scenario(own, target, rng, ts=ts)
            cfg["fidelity"] = "eigenray"
            cfg["propagation"] = {"environment": {"depth": water, "ssp_depth": depth, "ssp_speed": speed, "bottom_speed": 1480.0, "bottom_density": 1.2, "bottom_attenuation": 1.0},
                                  "frequency": FC, "max_range": rng + 1000.0, "range_step": 50.0, "depth_step": 5.0, "max_arrivals": 8}
            rec, origin = T.prepared(cfg)
            slant = float(np.hypot(rng, target - own)) + 1.0
            row = {"target_depth_m": target, "range_m": rng, "rungs": {}}
            flat = core.live_multipath_detection(rec, tuple(origin), 0.0, slant, mp.cpp_options({"gradient_sigma_per_s": GRADIENT_SIGMA, "water_depth": water}))
            row["rungs"]["flat boundaries, no profile (as before)"] = {"depth_m": flat["depth"] if flat["valid"] else None, "sigma_m": flat["sigma"] if flat["valid"] else None,
                                                                    "reason": flat["reason"]}
            for label, w in rungs.items():
                t0 = time.perf_counter()
                d = core.live_multipath_detection(rec, tuple(origin), 0.0, slant, dict(mp.cpp_options({"gradient_sigma_per_s": GRADIENT_SIGMA, "water_depth": water}), rays=1), profile=w)
                e = {"depth_m": d["depth"] if d["valid"] else None, "sigma_m": d["sigma"] if d["valid"] else None, "reason": d["reason"], "seconds": time.perf_counter() - t0}
                if d["valid"]:
                    zs = [d["depth"]] + [v for v in d["ensemble"] if np.isfinite(v)]
                    e.update(error_m=d["depth"] - target, ensemble_m=[min(zs), max(zs)])
                row["rungs"][label] = e
            rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bellhop", action="store_true")
    ap.add_argument("--no-write", action="store_true")
    a = ap.parse_args()
    R = study()
    if a.bellhop:
        if not os.environ.get("GPUSONAR_BELLHOP"):
            sys.exit("--bellhop needs GPUSONAR_BELLHOP")
        R["bellhop_pings"] = bellhop_pings()
    elif OUT.exists():
        old = json.loads(OUT.read_text())
        if "bellhop_pings" in old:
            R["bellhop_pings"] = old["bellhop_pings"]
    S = R["summary"]
    for est in ("multipath", "elevation", "fused"):
        print(f"\n{est}: median |error| / median sigma m (given of cases with the paths; within 2 sigma; within ensemble)")
        for label, row in S[est].items():
            cells = []
            for r in RANGES:
                e = row[str(int(r))]
                if not e["depth_given"]:
                    cells.append(f"none of {e['cases_with_the_paths']}")
                else:
                    cells.append(f"{e['median_abs_error_m']:.1f}/{e['median_sigma_m']:.1f} ({e['depth_given']} of {e['cases_with_the_paths']}; {e['within_two_sigma']}"
                                 + (f"; {e['within_ensemble']})" if "within_ensemble" in e else ")"))
            print(f"  {label:58s} " + " | ".join(cells))
    print("\ntrue water:", json.dumps(S["true_water"]))
    print("\nstage-2 rerun (first arrivals, bounces included): within 2 sigma / within the ensemble's span / either, of cases")
    for label, row in R["stage2_rerun"].items():
        print(f"  {label:58s} " + " | ".join(f"{e['within_two_sigma']}/{e['within_ensemble']}/{e['within_either']} of {e['cases']}" for e in row.values()))
    for row in R.get("bellhop_pings", []):
        print(f"Bellhop ping, target {row['target_depth_m']:.0f} m at {row['range_m'] / 1000:.0f} km: " + "; ".join(
            f"{k.split(':')[0]}: " + (f"{v['depth_m']:.1f} +- {v['sigma_m']:.1f}" if v["depth_m"] is not None else v["reason"]) for k, v in row["rungs"].items()))
    if not a.no_write:
        OUT.write_text(json.dumps(copy.deepcopy(R), indent=1) + "\n")
        print("wrote", OUT)


if __name__ == "__main__":
    main()
