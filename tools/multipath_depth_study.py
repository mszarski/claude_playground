#!/usr/bin/env python3
"""Depth from multipath timing with one row: what it gives, next to the stacked rows' elevation depth.

    tools/multipath_depth_study.py [--quick]      # writes docs/design/multipath_depth_study.json

1. Accuracy against truth. The one-row face sonar (popoto_long_range: 32 hydrophones, 4-6 kHz, 1 s) at 225 m, a point
   target at 150 m (deep) and at 40 m (shallow) at 1, 3, 6 and 12 km under a flat surface (the simulator's image-source
   paths, ambient noise), at two echo levels: the strongest arrival about 17 dB and about 26 dB over the noise at the
   matched filter's output. Per case: the share of pings that give a depth, its rms and largest error, the reported
   sigma, how often truth is inside two sigma; then a track of 6 pings at the lower level. Estimated by the C++ stage
   both chain engines run (libs/live multipath_depth.hpp).
2. The same geometry's limits in closed form: the shallowest target depth the band resolves, 3 r c / (4 B z_own), and
   one delay-resolution cell in depth, r c / (4 B z_own), for own depths 50, 110, 225 and 450 m.
3. The dependence on the sound-speed profile, with our ray code (gpusonar/raytrace.py): through the four illustrative
   stage-2 profiles (200 m of water, vehicle at 50 m, targets at 20 and 120 m; tools/popoto_stage2_study.py) and through
   one deep illustrative profile (vehicle at 225 m, targets at 40 and 150 m), the true one-way delay of the
   surface-reflected arrival behind the direct one, and the depth the flat straight-ray formula makes of it with the
   sound speed measured at the vehicle. Where a profile leaves no direct path or no surface path, it says so.
4. Beside them: the stacked rows' depth sigma from the angle alone (docs/design/popoto_elevation_study.json: the
   elevation sigma at 13 and 20 dB times the range) and the sound-speed ladder's depth errors (docs/design/popoto_stage2.json).
"""
import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

import gpusonar as gs  # noqa: E402
from gpusonar import multipath_depth as mp  # noqa: E402
from gpusonar import raytrace, recon  # noqa: E402
from gpusonar.sim import run_scenario  # noqa: E402

core = gs.core
OUT = ROOT / "docs" / "design" / "multipath_depth_study.json"
RANGES = (1000.0, 3000.0, 6000.0, 12000.0)
OWN = 225.0
TARGETS = {"deep, 150 m": 150.0, "shallow, 40 m": 40.0}
LEVELS_DB = (17.0, 26.0)
BAND, SOUND = 2000.0, 1500.0


def scenario(own, target, rng, ts, seed, surface=True):
    cfg = {"name": "multipath_depth_study", "sonar": "popoto_long_range", "fidelity": "image_source" if surface else "free_field",
           "sonar_overrides": {"receive": {"kind": "baseband", "sample_rate": 8.0e3, "window": {"range": [max(rng - 600.0, 80.0), rng + 600.0]}}},
           "environment": {"sound_speed": SOUND, "temperature": 12.0, "salinity": 35.0},
           "platform": {"position": [0.0, 0.0, float(own)], "rpy": [0, 0, 0], "velocity": [0.0, 0.0, 0.0]},
           "pings": {"count": 1, "interval": 20.0},
           "scene": [{"type": "points", "label": "target", "positions": [[0.0, float(rng), float(target)]], "target_strength_db": float(ts)}],
           "absorption": "ainslie_mccolm", "receiver": {"adc": False, "ambient_noise": {"wind_speed": 5.0, "shipping": 0.5}}, "seed": seed}
    if surface:
        cfg["propagation"] = {"surface_z": 0.0, "bottom_z": None, "max_order": 1}
    return cfg


def one(own, target, rng, ts, seed, options, stage=None, track=1):
    rec = run_scenario(scenario(own, target, rng, ts, seed), engine="cpp").records[0]
    recon.prepare([rec])
    origin = rec.rx_positions().mean(axis=0)
    slant = float(np.hypot(rng, target - own)) + 1.0
    t0 = time.perf_counter()
    d = core.live_multipath_detection(rec, tuple(origin), 0.0, slant, options)
    d["seconds"] = time.perf_counter() - t0
    if stage is not None:
        stage.add(track, d, mp.geometry_dict(mp.geometry(rec, origin, slant, mp.options({}))))
    return d


def accuracy(n: int) -> list:
    options = mp.cpp_options({})
    rows = []
    for name, target in TARGETS.items():
        for rng in RANGES:
            # the target strength that puts the strongest arrival at the wanted level: one loud ping tells
            ref = one(OWN, target, rng, 40.0, 1, options)
            for level in LEVELS_DB:
                ts = 40.0 + level - ref["snr_db"]
                got = [one(OWN, target, rng, ts, 100 + k, options) for k in range(n)]
                valid = [d for d in got if d["valid"]]
                err = np.array([d["depth"] - target for d in valid])
                sig = np.array([d["sigma"] for d in valid])
                row = {"target": name, "target_depth_m": target, "own_depth_m": OWN, "range_m": rng, "strongest_arrival_snr_db": float(np.mean([d["snr_db"] for d in got])),
                       "pings": n, "depth_given": len(valid) / n, "seconds_per_detection": float(np.median([d["seconds"] for d in got]))}
                if len(valid):
                    row.update(rms_error_m=float(np.sqrt(np.mean(err**2))), median_abs_error_m=float(np.median(np.abs(err))), worst_error_m=float(np.abs(err).max()),
                               median_sigma_m=float(np.median(sig)), within_two_sigma=float(np.mean(np.abs(err) <= 2.0 * sig)),
                               ambiguous=float(np.mean([d["ambiguous"] for d in valid])))
                if level == LEVELS_DB[0]:
                    tracks = []
                    for t in range(max(n // 6, 1)):
                        stage = core.MultipathDepth(options)
                        for k in range(6):
                            one(OWN, target, rng, ts, 5000 + 10 * t + k, options, stage)
                        tracks.append(stage.track(1))
                    tv = [t for t in tracks if t["valid"]]
                    row["track_of_6_pings"] = {"tracks": len(tracks), "depth_given": len(tv) / len(tracks)}
                    if tv:
                        te = np.array([t["depth"] - target for t in tv])
                        row["track_of_6_pings"].update(rms_error_m=float(np.sqrt(np.mean(te**2))), worst_error_m=float(np.abs(te).max()),
                                                       median_sigma_m=float(np.median([t["sigma"] for t in tv])))
                rows.append(row)
                print(json.dumps(row), flush=True)
    return rows


def limits() -> list:
    cfg = mp.options({})
    rows = []
    for own in (50.0, 110.0, 225.0, 450.0):
        for rng in RANGES:
            g = mp.Geometry(range_m=rng, array_depth=own, projector_depth=own, sound_speed=SOUND, f_centre=5000.0, bandwidth=BAND)
            grid, _, _ = mp.depth_grid(g, cfg)
            rows.append({"own_depth_m": own, "range_m": rng, "shallowest_target_m": float(grid[0]) if len(grid) else None,
                         "closed_form_m": float(cfg["min_delay_bandwidth"]) * rng * SOUND / (4.0 * BAND * own),
                         "delay_cell_in_depth_m": rng * SOUND / (4.0 * BAND * own),
                         "fringe_in_depth_m": rng * SOUND / (4.0 * 5000.0 * own)})
    return rows


def stage2_profiles():
    """The four illustrative profiles of the stage-2 study and one deep one (defined by parameters; not measurements)."""
    pressure = 0.017
    fam = {"isothermal": {"family": "isothermal", "family_gradient": pressure},
           "mixed layer 30 m over a strong thermocline (-0.15 /s for 40 m)":
               {"family": "mixed_layer", "family_gradient": pressure, "layer_depth": 30.0, "thermocline_gradient": -0.15, "thermocline_thickness": 40.0},
           "mixed layer 50 m over a weak thermocline (-0.05 /s for 60 m)":
               {"family": "mixed_layer", "family_gradient": pressure, "layer_depth": 50.0, "thermocline_gradient": -0.05, "thermocline_thickness": 60.0},
           "no mixed layer: -0.05 /s from the surface to 100 m":
               {"family": "mixed_layer", "family_gradient": pressure, "layer_depth": 1.0, "thermocline_gradient": -0.05, "thermocline_thickness": 99.0}}
    out = []
    for name, p in fam.items():
        depth, speed = core.live_family_profile(p, 0.0, 1500.0, 300.0)
        out.append((name, np.array(depth), np.array(speed), 200.0, 50.0, (20.0, 120.0)))
    depth, speed = core.live_family_profile(fam["mixed layer 30 m over a strong thermocline (-0.15 /s for 40 m)"], 0.0, 1500.0, 1100.0)
    out.append(("deep water: mixed layer 30 m, strong thermocline, pressure gradient to 1000 m", np.array(depth), np.array(speed), 1000.0, OWN, (40.0, 150.0)))
    return out


def profile_dependence() -> list:
    cfg = mp.options({})
    rows = []
    for name, depth, speed, water, own, targets in stage2_profiles():
        prof = raytrace.Profile(depth, speed, water)
        c_own = float(prof.c(own))
        for target in targets:
            for rng in RANGES:
                g = mp.Geometry(range_m=rng, array_depth=own, projector_depth=own, sound_speed=c_own, f_centre=5000.0, bandwidth=BAND, water_depth=water)
                grid, _, _ = mp.depth_grid(g, cfg)
                row = {"profile": name, "water_depth_m": water, "own_depth_m": own, "target_depth_m": target, "range_m": rng,
                       "band_resolves_it": bool(len(grid) and grid[0] <= target)}
                rays = raytrace.eigenrays(prof, own, target, rng, max_bounces=1)
                direct = [e for e in rays if e.n_surface == 0 and e.n_bottom == 0]
                surf = [e for e in rays if e.n_surface == 1 and e.n_bottom == 0]
                row["direct_paths"], row["surface_paths"] = len(direct), len(surf)
                if direct and surf:
                    tau = surf[0].time - direct[0].time
                    flat = (np.hypot(rng, target + own) - np.hypot(rng, target - own)) / c_own
                    z_flat = raytrace.flat_depth_from_delay(tau, own, rng, c_own)
                    row.update(delay_ms=1e3 * tau, flat_delay_ms=1e3 * flat, flat_formula_depth_m=z_flat, bias_m=z_flat - target, bias_fraction=(z_flat - target) / target)
                else:
                    row["note"] = "no direct path" if not direct else "no surface-reflected path"
                rows.append(row)
    return rows


def stacked_rows() -> dict:
    """What the repository already states for the rows' elevation depth."""
    out = {}
    try:
        el = json.loads((ROOT / "docs" / "design" / "popoto_elevation_study.json").read_text())
        sig = el["accuracy"]["2 rows, 1.50 m apart"]["sigma_deg"]
        out["angle_sigma_deg_by_snr_db"] = sig
        out["depth_sigma_from_angle_m"] = {f"{int(r)} m": {snr: float(np.radians(v) * r) for snr, v in sig.items()} for r in RANGES}
        out["two_row_candidate_spacing_m_per_km"] = el["two_rows"]["candidate_depth_step_m_per_km"]
        out["source"] = "docs/design/popoto_elevation_study.json (accuracy, 2 rows 1.5 m apart; the same figures for 3 rows over 1.5 m)"
    except (OSError, KeyError) as e:
        out["error"] = repr(e)
    out["ladder"] = "docs/design/popoto_stage2.json, profile_ladder.summary: median / worst depth error of each rung at these ranges (refraction, on top of the angle's sigma)"
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="12 pings a case instead of 36")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    n = 12 if a.quick else 36
    out = {"sonar": "popoto_long_range (one row of 32, 4-6 kHz, 1 s sweep), projector 1.5 m below the array", "options": mp.options({}),
           "noise": "ambient, wind 5 m/s, shipping 0.5", "surface": "flat (image source)",
           "accuracy": accuracy(n), "limits": limits(), "profile_dependence": profile_dependence(), "stacked_rows": stacked_rows(),
           "command": "PYTHONPATH=python python3 tools/multipath_depth_study.py" + (" --quick" if a.quick else "")}
    Path(a.out).write_text(json.dumps(copy.deepcopy(out), indent=1) + "\n")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
