#!/usr/bin/env python3
"""The 12 km design point, measured through the chain on simulated pings: one, two and three rows on the face.

    tools/popoto_12km_study.py [--quick]        # writes docs/design/popoto_12km.json

For each row count (sonars popoto_long_range, popoto_long_range_2row, popoto_long_range_3row) the ship-sized mover of
the `popoto_12km` scenario is put at 12 km in

    popoto_12km (100 m of isovelocity water over sand)    with Bellhop, and with image-source paths (runs anywhere)
                                                           (--merge adds to the file; --no-envs leaves the environments out)
    the same without reverberation                         image-source: what reverberation costs at 12 km
    each popoto_env_* environment, the ship moved to 12 km with Bellhop (needs GPUSONAR_BELLHOP)

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


def water_depth(cfg: dict) -> float:
    prop = cfg.get("propagation") or {}
    return float((prop.get("environment") or {}).get("depth", prop.get("bottom_z", 100.0)))


def chain_for(rows: int, depth: float) -> dict:
    chain = load_yaml(ROOT / "configs" / "deploy" / f"{DEPLOY[rows]}.yaml")["chain"]
    chain.pop("sync", None)                                                  # these pings are at baseband with ideal clocks
    if chain.get("elevation"):
        chain["elevation"] = dict(chain["elevation"], water_depth=depth)
    return chain


def measure(cfg: dict, rows: int, engine: str = "cpp") -> dict:
    """Simulate, run the chain, score the ship."""
    t0 = time.time()
    records = run_scenario(copy.deepcopy(cfg)).records
    t_sim = time.time() - t0
    ship = cfg["scene"][0]
    pos0, vel = np.asarray(ship["positions"], dtype=float), np.asarray(ship.get("velocity", [0, 0, 0]), dtype=float)
    interval = float(cfg["pings"]["interval"])
    chain = chain_for(rows, water_depth(cfg))
    with LiveChain(dict(chain, engine=engine, backend="host")) as live:
        reports = live.run(records)
    out = {"rows": rows, "pings": len(reports), "ping_interval_s": interval, "simulate_seconds": round(t_sim, 1),
           "chain_seconds_per_ping": float(np.median([r["processing_seconds"] for r in reports])), "per_ping": []}
    confirmed_at = None
    for k, rep in enumerate(reports):
        own = np.asarray(rep["ownship_xyz"])
        pos = pos0 + vel * interval * k
        rng = np.linalg.norm(pos - own, axis=1)
        brg = np.degrees(np.arctan2(pos[:, 0] - own[0], pos[:, 1] - own[1]))
        near = [d for d in rep["detections"] if d["snr_db"] is not None and rng.min() - 20.0 <= d["range_m"] <= rng.max() + 20.0
                and brg.min() - 1.5 <= d["bearing_deg"] <= brg.max() + 1.5]
        best = max(near, key=lambda d: d["snr_db"]) if near else None
        tracks = [t for t in rep["tracks"] if rng.min() - 30.0 <= t["range_m"] <= rng.max() + 30.0]
        if tracks and confirmed_at is None:
            confirmed_at = k + 1
        row = {"detected": best is not None, "snr_db": best["snr_db"] if best else None, "detections": len(rep["detections"]),
               "range_error_m": (best["range_m"] - float(rng.mean())) if best else None, "bearing_error_deg": (best["bearing_deg"] - float(brg.mean())) if best else None}
        if best and rows > 1:
            row.update(elevation_deg=best.get("elevation_deg"), elevation_sigma_deg=best.get("elevation_sigma_deg"), ambiguous=best.get("elevation_ambiguous"),
                       candidates=len(best.get("elevation_alternatives_deg") or []) + (best.get("elevation_deg") is not None), depth_m=best.get("depth_m"),
                       depth_sigma_m=best.get("depth_sigma_m"), true_elevation_deg=float(np.degrees(np.arcsin((own[2] - pos[0, 2]) / rng.mean()))), true_depth_m=float(pos[0, 2]))
        out["per_ping"].append(row)
    last = reports[-1]
    own = np.asarray(last["ownship_xyz"])
    pos = pos0 + vel * interval * (len(reports) - 1)
    rng = np.linalg.norm(pos - own, axis=1)
    on = [t for t in last["tracks"] if rng.min() - 30.0 <= t["range_m"] <= rng.max() + 30.0]
    out["pings_detected"] = sum(1 for r in out["per_ping"] if r["detected"])
    out["median_snr_db"] = float(np.median([r["snr_db"] for r in out["per_ping"] if r["detected"]])) if out["pings_detected"] else None
    out["track_confirmed_at_ping"] = confirmed_at
    out["track_on_ship"] = [{"kind": t["kind"], "speed_m_s": t["speed_m_s"], "vx": t["vx"], "depth_m": t.get("depth_m")} for t in on]
    out["true_velocity_m_s"] = vel.tolist()
    out["track_kinds"] = {k: sum(1 for t in last["tracks"] if t["kind"] == k) for k in sorted({t["kind"] for t in last["tracks"]})}
    later = [t for t in last["tracks"] if t["range_m"] > rng.max() + 30.0]
    out["later_arrivals"] = {"tracks": len(later), "classified_multipath": sum(1 for t in later if t["kind"] == "multipath")}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="image-source cases only")
    ap.add_argument("--rows", default="1,2,3")
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--merge", action="store_true", help="keep the cases already in the file and add or replace the ones run now")
    ap.add_argument("--no-envs", action="store_true", help="leave out the popoto_env_* environments (each takes minutes with Bellhop)")
    args = ap.parse_args(argv)
    have = find_bellhop() is not None
    out_path = ROOT / "docs" / "design" / "popoto_12km.json"
    R = {"bellhop": bool(have and not args.quick), "ship": "five points of +18 dB over 120 m at 4 m depth, crossing at 6 m/s (the popoto_12km scenario's)", "cases": {}}
    if args.merge and out_path.exists():
        R["cases"] = json.loads(out_path.read_text())["cases"]
        R["bellhop"] = bool(have)
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
