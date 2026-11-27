#!/usr/bin/env python3
"""The 12 km design point, measured through the chain on simulated pings: one, two and three rows on the face.

    tools/popoto_12km_study.py [--quick]        # writes docs/design/popoto_12km.json

For each row count (sonars popoto_long_range, popoto_long_range_2row, popoto_long_range_3row) the ship-sized mover of
the `popoto_12km` scenario is put at 12 km in

    popoto_12km (100 m of isovelocity water over sand)    with Bellhop, and with image-source paths (runs anywhere)
                                                           (--merge adds to the file; --no-envs leaves the environments out)
    the same without reverberation                         image-source: what reverberation costs at 12 km
    each popoto_env_* environment, the ship moved to 12 km with Bellhop (needs GPUSONAR_BELLHOP)

With --classes (the target classes of configs/builds/targets and the depth planner's own depths; written under `class_cases`):

    the ship in non-ducting water (surface image only)     array at 10 m, and at the depth tools/popoto_depth_planner.py recommends
    a submerged mover at 6 and 12 km, the same water       the `large_submerged` class at speed, array at the depth that holds it and the ship
    the same mover at 12 km in 100 m over sand             Bellhop (three rows only: each case takes minutes)
    a 100 m x 10 m rigid cylinder crossing at 6 km         the simulator's aspect-dependent glint (sim/targets.py): seen only near broadside

every one with NO sound-speed profile knowledge (the deployment files as shipped: the bottom rung of the ladder) and scored for
track confirmation, class, speed, heading, and whether the true depth lies within two of the reported sigmas.

and the pings go through the C++ chain with the stage's deployment file. Reported per case: on which pings the ship is
detected and at what SNR over the local background, the ping at which a track on it is confirmed, the class of that
track, the other tracks' classes, and (with rows) the elevation, its candidates and the depth read for it.
Scenarios: configs/scenarios/popoto_12km_2row.yaml and popoto_12km_3row.yaml are the stacked-row variants of
popoto_12km; the environments are moved to 12 km by overrides here (their own files keep their ranges).
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

from gpusonar.config import list_configs, load_scenario_config, load_yaml  # noqa: E402
from gpusonar.live import LiveChain  # noqa: E402
from gpusonar.sim import run_scenario  # noqa: E402
from gpusonar.sim.bellhop import find_bellhop  # noqa: E402

SONAR = {1: "popoto_long_range", 2: "popoto_long_range_2row", 3: "popoto_long_range_3row"}
DEPLOY = {1: "popoto_vehicle", 2: "popoto_face_2row", 3: "popoto_vehicle_elevation"}
SCENARIO = {1: "popoto_12km", 2: "popoto_12km_2row", 3: "popoto_12km_3row"}


def image_source(cfg: dict) -> dict:
    """The scenario with straight-ray image-source paths (two bounces) in place of Bellhop: runs anywhere."""
    cfg = copy.deepcopy(cfg)
    env = (cfg.get("propagation") or {}).get("environment") or {}
    cfg["fidelity"] = "image_source"
    cfg["propagation"] = {"surface_z": 0.0, "bottom_z": float(env.get("depth", 100.0)), "max_order": 2,
                          "bottom": {"speed": float(env.get("bottom_speed", 1650.0)), "density": float(env.get("bottom_density", 1.9)),
                                     "attenuation": float(env.get("bottom_attenuation", 0.8))}}
    return cfg


def at_12km(env_name: str, rows: int) -> dict:
    """A popoto_env_* scenario with the array of `rows` rows and the ship moved to 12 km (window, Bellhop range and
    ping interval as popoto_12km's)."""
    cfg = load_scenario_config(env_name)
    ref = load_scenario_config("popoto_12km")
    cfg["sonar"] = SONAR[rows]
    cfg["sonar_overrides"] = copy.deepcopy(ref["sonar_overrides"])
    cfg["scene"] = copy.deepcopy(ref["scene"])
    cfg["pings"] = copy.deepcopy(ref["pings"])
    for key in ("max_range", "range_step"):
        cfg["propagation"][key] = ref["propagation"][key]
    return cfg


def water_depth(cfg: dict):
    prop = cfg.get("propagation") or {}
    if "bottom_z" in prop and prop["bottom_z"] is None:
        return None                                                              # no bottom: the chain is told no water depth
    return float((prop.get("environment") or {}).get("depth", prop.get("bottom_z", 100.0)))


MOVER_TS_DB = 33.0        # dB, the submerged mover as a whole: inside the `large_submerged` class's ASSUMED range (25 to 41.5 dB). A parameter, not a measurement
MOVER_DEPTH_M = 150.0     # m: inside the class's assumed depth range
SHIP_POINT_TS_DB = 18.0   # the popoto_12km scenario's ship: five points of +18 dB


def planned_depths() -> dict:
    """Own depths from the depth plan of the three-row face in non-ducting water: the ship's best, and the one that holds the ship and
    the large submerged class together."""
    from gpusonar import buildspec as bs
    from gpusonar import depthplan
    spec = bs.load("popoto_face_3x32")
    env = next(e for e in bs.environments_of(spec) if e["name"] == "deep_no_bottom_return")
    P = depthplan.plan(bs.derive(spec), env, bs.target_classes_of(spec), keep_grid=False)
    return {"ship": P["classes"]["ship"]["recommended"]["own_depth_m"], "both": P["together"]["ship + large_submerged"]["best_own_depth_m"], "environment": env["name"]}


def class_scenario(rows: int, what: str, range_m: float, own_depth: float, bellhop: bool = False) -> dict:
    """A class scenario built on popoto_12km's files: `what` is ship, mover or cylinder. Non-ducting water: image-source paths with the
    surface only (no bottom), surface reverberation, ambient noise at 5 m/s. `bellhop`: popoto_12km's own water (100 m over sand)."""
    cfg = load_scenario_config(SCENARIO[rows])
    cfg["name"] = f"popoto_class_{what}_{range_m / 1e3:g}km_{rows}row"
    cfg["sonar_overrides"]["receive"]["window"] = {"range": [range_m - 700.0, range_m + 700.0]}
    cfg["platform"]["position"] = [0.0, 0.0, float(own_depth)]
    cfg["pings"] = {"count": 6, "interval": round(2.0 * (range_m + 700.0) / 1500.0 + 1.2, 1)}
    cfg.pop("truth", None)
    if not bellhop:
        cfg["fidelity"] = "image_source"
        cfg["propagation"] = {"surface_z": 0.0, "bottom_z": None, "max_order": 1}
        cfg["reverberation"] = [{"boundary_z": 0.0, "model": "chapman_harris", "wind_speed": 5.0}]
    xs = [740.0, 770.0, 800.0, 830.0, 860.0]
    if what == "ship":
        cfg["scene"] = [{"type": "points", "label": "ship", "positions": [[x, range_m, 4.0] for x in xs], "target_strength_db": SHIP_POINT_TS_DB, "velocity": [-6.0, 0.0, 0.0]}]
    elif what == "mover":
        z = MOVER_DEPTH_M if not bellhop else 60.0
        per_point = MOVER_TS_DB - 10.0 * np.log10(5.0)                           # five points that add in power to the whole
        cfg["scene"] = [{"type": "points", "label": "submerged mover", "positions": [[x - 40.0 + 25.0 * i, range_m, z] for i, x in enumerate([800.0] * 5)],
                         "target_strength_db": float(per_point), "velocity": [-8.0, 0.0, 0.0]}]
    elif what == "cylinder":
        # the hull along its track, crossing the sonar's broadside: at the middle ping the perpendicular from the sonar meets the hull
        v = -8.0
        mid = 0.5 * (cfg["pings"]["count"] - 1) * cfg["pings"]["interval"]
        x0 = 1.5 * mid - v * mid                                                 # where the hull is at the first ping so that it is abeam of the sonar (moving at 1.5 m/s) mid-run
        cfg["scene"] = [{"type": "target", "label": "cylinder 100 m x 10 m", "origin": [float(x0), range_m, MOVER_DEPTH_M], "velocity": [v, 0.0, 0.0], "heading_deg": 0.0,
                         "shapes": [{"shape": "cylinder", "start": [-50.0, 0.0, 0.0], "end": [50.0, 0.0, 0.0], "radius": 5.0, "resolved": True, "spacing": 0.06}]}]
    else:
        raise ValueError(what)
    return cfg


def chain_for(rows: int, depth: float) -> dict:
    chain = load_yaml(ROOT / "configs" / "deploy" / f"{DEPLOY[rows]}.yaml")["chain"]
    chain.pop("sync", None)                                                  # these pings are at baseband with ideal clocks
    if chain.get("elevation"):
        chain["elevation"] = dict(chain["elevation"], water_depth=depth)             # None: no bottom, no depth bound for two rows
    return chain


def _span(d: dict) -> tuple:
    """The ranges a detection covers. The chain's detector (cell_statistic: multiscale) reports ONE detection for a group of
    arrivals: `range_m` is its energy-weighted mean, `first_range_m` its first arrival and `extent_m` what it spans; a detection
    without those fields is a point."""
    ext = float(d.get("extent_m") or 0.0)
    lo = float(d["first_range_m"]) if d.get("first_range_m") is not None else float(d["range_m"]) - 0.5 * ext
    return lo, max(lo + ext, float(d["range_m"]))


BACKEND, SIM_BACKEND = "host", "host"      # --backend, --sim-backend (opt-in): the chain's and the simulator's; host as every committed run


def _sim_backend():
    from gpusonar import core
    return core.Backend.Cuda if SIM_BACKEND == "cuda" else core.Backend.Host


def measure(cfg: dict, rows: int, engine: str = "cpp") -> dict:
    """Simulate, run the chain, score the ship. A detection is the ship's when the ranges it covers (`_span`) reach the ship's
    own to within 20 m: with many bounced arrivals merged into one detection its mean range lies behind the ship by a share of
    its extent. A track is the ship's within 30 m of the ship or of the ranges its detection covers on that ping."""
    t0 = time.time()
    records = run_scenario(copy.deepcopy(cfg), backend=_sim_backend()).records
    t_sim = time.time() - t0
    ship = cfg["scene"][0]
    pos0 = np.asarray(ship["positions"] if "positions" in ship else [ship["origin"]], dtype=float)
    vel = np.asarray(ship.get("velocity", [0, 0, 0]), dtype=float)
    interval = float(cfg["pings"]["interval"])
    chain = chain_for(rows, water_depth(cfg))
    with LiveChain(dict(chain, engine=engine, backend=BACKEND)) as live:
        reports = live.run(records)
    out = {"rows": rows, "pings": len(reports), "ping_interval_s": interval, "simulate_seconds": round(t_sim, 1),
           "chain_seconds_per_ping": float(np.median([r["processing_seconds"] for r in reports])), "per_ping": []}
    confirmed_at = None
    for k, rep in enumerate(reports):
        own = np.asarray(rep["ownship_xyz"])
        pos = pos0 + vel * interval * k
        rng = np.linalg.norm(pos - own, axis=1)
        brg = np.degrees(np.arctan2(pos[:, 0] - own[0], pos[:, 1] - own[1]))
        near = [d for d in rep["detections"] if d["snr_db"] is not None and _span(d)[0] <= rng.max() + 20.0 and _span(d)[1] >= rng.min() - 20.0
                and brg.min() - 1.5 <= d["bearing_deg"] <= brg.max() + 1.5]
        best = max(near, key=lambda d: d["snr_db"]) if near else None
        reach = max([_span(d)[1] for d in near], default=float(rng.max()))            # how far behind the ship its own detection extends on this ping
        tracks = [t for t in rep["tracks"] if rng.min() - 30.0 <= t["range_m"] <= max(rng.max(), reach) + 30.0]
        if tracks and confirmed_at is None:
            confirmed_at = k + 1
        row = {"detected": best is not None, "snr_db": best["snr_db"] if best else None, "detections": len(rep["detections"]),
               "range_error_m": (best["range_m"] - float(rng.mean())) if best else None, "bearing_error_deg": (best["bearing_deg"] - float(brg.mean())) if best else None,
               "first_range_error_m": (_span(best)[0] - float(rng.min())) if best else None, "extent_m": best.get("extent_m") if best else None}
        if best and rows > 1:
            row.update(elevation_deg=best.get("elevation_deg"), elevation_sigma_deg=best.get("elevation_sigma_deg"), ambiguous=best.get("elevation_ambiguous"),
                       candidates=len(best.get("elevation_alternatives_deg") or []) + (best.get("elevation_deg") is not None), depth_m=best.get("depth_m"),
                       depth_sigma_m=best.get("depth_sigma_m"), true_elevation_deg=float(np.degrees(np.arcsin((own[2] - pos[0, 2]) / rng.mean()))), true_depth_m=float(pos[0, 2]))
        out["per_ping"].append(row)
    last = reports[-1]
    own = np.asarray(last["ownship_xyz"])
    pos = pos0 + vel * interval * (len(reports) - 1)
    rng = np.linalg.norm(pos - own, axis=1)
    far = max(float(rng.max()), reach)                                                 # the last ping's: the ship's own detection ends here
    on = [t for t in last["tracks"] if rng.min() - 30.0 <= t["range_m"] <= far + 30.0]
    out["pings_detected"] = sum(1 for r in out["per_ping"] if r["detected"])
    out["median_snr_db"] = float(np.median([r["snr_db"] for r in out["per_ping"] if r["detected"]])) if out["pings_detected"] else None
    out["track_confirmed_at_ping"] = confirmed_at
    out["track_on_ship"] = [{"kind": t["kind"], "speed_m_s": t["speed_m_s"], "vx": t["vx"], "depth_m": t.get("depth_m")} for t in on]
    out["true_velocity_m_s"] = vel.tolist()
    out["own_depth_m"], out["true_depth_m"] = float(own[2]), float(pos0[0, 2])
    movers = [t for t in on if t["kind"] == "mover"]
    best_t = max(movers or on, key=lambda t: t.get("mean_snr_db") or 0.0) if on else None
    if best_t is not None:
        vx, vy = best_t["vx"], best_t.get("vy", 0.0)
        out["track"] = {"kind": best_t["kind"], "speed_error_m_s": float(np.hypot(vx, vy) - np.hypot(vel[0], vel[1])),
                        "heading_error_deg": float(np.degrees(np.angle(np.exp(1j * (np.arctan2(vy, vx) - np.arctan2(vel[1], vel[0])))))),
                        "depth_m": best_t.get("depth_m"), "depth_sigma_m": best_t.get("depth_sigma_m"), "depth_profile_source": best_t.get("depth_profile_source"),
                        "true_depth_within_two_sigma": (None if best_t.get("depth_m") is None or not best_t.get("depth_sigma_m") else
                                                        bool(abs(best_t["depth_m"] - pos0[0, 2]) <= 2.0 * best_t["depth_sigma_m"]))}
    out["pings_with_several_possible_elevations"] = sum(1 for r in out["per_ping"] if r.get("detected") and (r.get("candidates") or 0) > 1)
    out["depth_profile_sources"] = sorted({str(d.get("depth_profile_source")) for rep in reports for d in rep["detections"] if d.get("depth_profile_source") is not None})
    out["track_kinds"] = {k: sum(1 for t in last["tracks"] if t["kind"] == k) for k in sorted({t["kind"] for t in last["tracks"]})}
    later = [t for t in last["tracks"] if t["range_m"] > far + 30.0]
    out["later_arrivals"] = {"tracks": len(later), "classified_multipath": sum(1 for t in later if t["kind"] == "multipath")}
    return out


def class_cases(args, have: bool, out_path: Path) -> int:
    R = json.loads(out_path.read_text()) if out_path.exists() else {"cases": {}}
    depths = planned_depths()
    C = R.setdefault("class_cases", {})
    C.update(planned_own_depths_m=depths, mover={"target_strength_db": MOVER_TS_DB, "depth_m": MOVER_DEPTH_M, "speed_m_s": 8.0,
                                                  "note": "five points adding in power to the whole; inside the large_submerged class's ASSUMED range"},
             profile_knowledge="none: the deployment files as shipped (the bottom rung of the profile ladder)", cases=C.get("cases", {}), not_run=[])
    for rows in (int(x) for x in args.rows.split(",")):
        cases = [("ship at 12 km, non-ducting water, array at 10 m", class_scenario(rows, "ship", 12000.0, 10.0)),
                 (f"ship at 12 km, non-ducting water, array at the planned depth", class_scenario(rows, "ship", 12000.0, depths["ship"])),
                 ("submerged mover at 12 km, non-ducting water, array at the depth that holds both", class_scenario(rows, "mover", 12000.0, depths["both"])),
                 ("submerged mover at 6 km, non-ducting water, array at the depth that holds both", class_scenario(rows, "mover", 6000.0, depths["both"]))]
        if rows == 3:
            cases.append(("rigid cylinder 100 m x 10 m crossing broadside at 6 km, non-ducting water", class_scenario(rows, "cylinder", 6000.0, depths["both"])))
            if have and not args.quick:
                cases.append(("submerged mover (60 m deep) at 12 km, 100 m over sand, Bellhop, array at 30 m", class_scenario(rows, "mover", 12000.0, 30.0, bellhop=True)))
        elif rows == 2 and "bellhop, rows 1 and 2" not in " ".join(C["not_run"]):
            C["not_run"].append("the submerged mover in 100 m over sand with Bellhop, rows 1 and 2 (minutes each); the cylinder, rows 1 and 2")
        for label, cfg in cases:
            t0 = time.time()
            try:
                res = measure(cfg, rows)
            except Exception as exc:  # noqa: BLE001
                res = {"rows": rows, "error": f"{type(exc).__name__}: {exc}"}
            C["cases"].setdefault(label, {})[str(rows)] = res
            print(f"rows {rows}  {label:80s} " + (res.get("error") or f"detected {res['pings_detected']}/{res['pings']}, SNR {res['median_snr_db']}, track at ping {res['track_confirmed_at_ping']}, "
                  f"{res.get('track')}, several elevations on {res['pings_with_several_possible_elevations']} pings") + f"  ({time.time() - t0:.0f} s)", flush=True)
            if not args.no_write:
                out_path.write_text(json.dumps(R, indent=2, default=float) + "\n")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="image-source cases only")
    ap.add_argument("--rows", default="1,2,3")
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--merge", action="store_true", help="keep the cases already in the file and add or replace the ones run now")
    ap.add_argument("--no-envs", action="store_true", help="leave out the popoto_env_* environments (each takes minutes with Bellhop)")
    ap.add_argument("--classes", action="store_true", help="only the target-class cases (ship at depth, submerged mover, cylinder): written under class_cases, the rest of the file kept")
    ap.add_argument("--backend", choices=("host", "cuda", "auto"), default="host", help="the chain's backend (opt-in; default host, as every committed run)")
    ap.add_argument("--sim-backend", choices=("host", "cuda"), default="host", help="the simulator's backend (opt-in; default host)")
    ap.add_argument("--out", help="the JSON to write (and, with --merge or --classes, to start from; default docs/design/popoto_12km.json)")
    args = ap.parse_args(argv)
    global BACKEND, SIM_BACKEND
    BACKEND, SIM_BACKEND = args.backend, args.sim_backend
    have = find_bellhop() is not None
    out_path = Path(args.out) if args.out else ROOT / "docs" / "design" / "popoto_12km.json"
    if args.classes:
        return class_cases(args, have, out_path)
    R = {"bellhop": bool(have and not args.quick), "ship": "five points of +18 dB over 120 m at 4 m depth, crossing at 6 m/s (the popoto_12km scenario's)", "cases": {}}
    if args.merge and out_path.exists():
        R["cases"] = json.loads(out_path.read_text())["cases"]
        R["bellhop"] = bool(have)
    if out_path.exists() and "class_cases" in json.loads(out_path.read_text()):
        R["class_cases"] = json.loads(out_path.read_text())["class_cases"]          # the other half of the file (--classes): kept, as --classes keeps this half
    R["not_run"] = []
    envs = [] if args.no_envs else sorted(n for n in list_configs("scenarios") if n.startswith("popoto_env_"))
    if args.no_envs:
        R["not_run"].append(f"rows {args.rows}: the popoto_env_* environments (left out with --no-envs)")
    for rows in (int(x) for x in args.rows.split(",")):
        base = load_scenario_config(SCENARIO[rows])
        cases = [("popoto_12km: 100 m isovelocity over sand, image-source paths", image_source(base))]
        quiet = image_source(base)
        quiet.pop("reverberation", None)
        cases.append(("the same without reverberation (noise only)", quiet))
        if have and not args.quick:
            cases.append(("popoto_12km: 100 m isovelocity over sand, Bellhop", base))
            cases += [(f"{n} at 12 km, Bellhop", at_12km(n, rows)) for n in envs]
        for label, cfg in cases:
            t0 = time.time()
            try:
                res = measure(cfg, rows)
            except Exception as exc:  # noqa: BLE001
                res = {"rows": rows, "error": f"{type(exc).__name__}: {exc}"}
            R["cases"].setdefault(label, {})[str(rows)] = res
            print(f"rows {rows}  {label:62s} " + (res.get("error") or f"detected on {res['pings_detected']}/{res['pings']} pings, SNR {res['median_snr_db']}, "
                  f"track at ping {res['track_confirmed_at_ping']}, on ship {[t['kind'] for t in res['track_on_ship']]}, kinds {res['track_kinds']}") + f"  ({time.time() - t0:.0f} s)", flush=True)
            if not args.no_write:
                out_path.write_text(json.dumps(R, indent=2, default=float) + "\n")
    # say which row counts an environment was not run with (a --merge run of some rows leaves the others out)
    missing = {}
    for label, by in R["cases"].items():
        for r in ("1", "2", "3"):
            if r not in by:
                missing.setdefault(r, []).append(label.split(" at 12 km")[0])
    for r, labels in sorted(missing.items()):
        R["not_run"].append(f"{r} row{'s' if r != '1' else ''}: " + ", ".join(labels) + " (each Bellhop case takes 1 to 10 minutes here)")
    if not args.no_write:
        out_path.write_text(json.dumps(R, indent=2, default=float) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
