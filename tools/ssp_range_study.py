#!/usr/bin/env python3
"""Water that changes with range against a chain that knows only the local profile.

    GPUSONAR_BELLHOP=.../bellhop.exe tools/ssp_range_study.py [--scenarios a,b] [--json docs/figures/ssp_range_study.json]
                                                               [--out docs/figures/ssp_range_study.md] [--commit HASH]

Each scenario of `SCENARIOS` (configs/scenarios/ssp_*: `environment.profiles_by_range`, ILLUSTRATIVE water) is simulated
twice with the same seed, sonar, platform, target and noise:

    changes with range     the scenario file as it is (Bellhop's 2-D sound-speed field)
    local everywhere       the same file with the profile at the sonar (the first of `profiles_by_range`) at every range

and the records of each go through the C++ chain of the build's own deployment file (configs/deploy), which is not told
anything about the water beyond the record's one sound speed (the local value at the sonar), in two passes:

    as deployed            the deployment file's chain, unchanged
    local profile, multipath depth
                           the same chain with the LOCAL profile given to the elevation stage (`chain.elevation.ssp`; builds with
                           stacked rows) and depth from multipath timing switched on with its defaults (`chain.multipath_depth: {}`)

Scored on every ping, as tools/fjord_rehearsal.py scores: the target's detection is the strongest from 30 m before to 60 m
behind where the target is, within 2 degrees. Reported: detection and SNR, range error of that detection (its strongest
sample) and of its first arrival (the first of the detection's `arrivals`), bearing error, elevation and depth from the
stacked rows, depth from multipath timing (per ping and of the track), the ping a track on the target is confirmed at and
the tracks that are somewhere else. Beside it, what the simulated sound itself does:
the eigenrays Bellhop finds from the array to the target in each water (delay as a range at the local sound speed, angle
at the array, boundary reflections), so that a change in the chain's numbers can be held against the change in the sound.

Few pings (the scenario's own count: four). One process, one run at a time; restartable (the JSON is rewritten after every
run). The chain is the one on the branch this is run from; nothing of it is changed.
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

from gpusonar.config import load_scenario_config, load_yaml  # noqa: E402

SCENARIOS = {"ssp_front_static_trial_2row": {"title": "Static array, two rows: a front 1.2 to 1.4 km out, target at 2.6 km", "deploy": "popoto_static_2row", "rows": 2},
             "ssp_thermocline_static_trial_3row": {"title": "Static array, three rows: the mixed layer deepens from 15 m to 45 m over 3 km, target at 2.6 km", "deploy": "popoto_static_3row", "rows": 3},
             "ssp_front_testauv_trial_1row": {"title": "Test AUV, one row: a front 200 to 300 m out, target at 500 m", "deploy": "popoto_testauv_1row", "rows": 1}}
WATERS = ["changes with range", "local everywhere"]
PASSES = ["as deployed", "local profile, multipath depth"]


def water(name: str, which: str) -> dict:
    """The scenario as typed, or with its local profile at every range."""
    cfg = load_scenario_config(name, expand=False)
    if which == "local everywhere":
        env = dict(cfg["environment"])
        sections = env.pop("profiles_by_range")
        env["profile"] = {"depth_m": list(sections[0]["depth_m"]), "speed_m_s": list(sections[0]["speed_m_s"])}
        cfg = dict(cfg, environment=env)
    elif which != "changes with range":
        raise ValueError(which)
    return cfg


def eigenray_table(config: dict, record) -> list:
    """What Bellhop finds from the array's reference point to the target (first ping): per arrival, strongest first,
    its one-way delay as a range at the record's sound speed, its angle at the array and its reflections."""
    from gpusonar.sim.propagation import make_provider
    c = float(record.environment.sound_speed)
    provider = make_provider("eigenray", c, config["propagation"])
    own = np.asarray(record.rx_positions().mean(axis=0), dtype=float)
    target = np.asarray(config["scene"][0]["positions"][0], dtype=float)
    rays = provider.eigenrays(own, target[None, :])
    slant = float(np.linalg.norm(target - own))
    ok = rays.valid[0]
    amp = np.abs(rays.amplitude[0][ok])
    rows = [{"delay_ms": float(1e3 * t), "range_excess_m": float(rays.sound_speed * t - slant), "level_db": float(20.0 * np.log10(a / amp.max())),
             "elevation_at_array_deg": float(-np.degrees(np.arcsin(np.clip(u[2], -1.0, 1.0)))), "surface": int(ns), "bottom": int(nb)}
            for t, a, u, ns, nb in zip(rays.delay[0][ok], amp, rays.launch[0][ok], rays.n_surface[0][ok], rays.n_bottom[0][ok])]
    return sorted(rows, key=lambda r: -r["level_db"])


def score(cfg: dict, reports: list, rows: int) -> dict:
    pos = np.asarray(cfg["scene"][0]["positions"][0], dtype=float)
    out = {"pings": len(reports), "per_ping": []}
    confirmed_at, track_multipath = None, None
    for k, rep in enumerate(reports):
        own = np.asarray(rep["ownship_xyz"])
        rng = float(np.linalg.norm(pos - own))
        brg = float(np.degrees(np.arctan2(pos[0] - own[0], pos[1] - own[1])))
        dets = [d for d in rep["detections"] if d.get("snr_db") is not None]
        near = [d for d in dets if -30.0 <= d["range_m"] - rng <= 60.0 and abs(d["bearing_deg"] - brg) <= 2.0]
        best = max(near, key=lambda d: d["snr_db"]) if near else None
        on = [t for t in rep["tracks"] if -40.0 <= t["range_m"] - rng <= 70.0 and abs(t.get("bearing_deg", brg) - brg) <= 3.0]
        if on and confirmed_at is None:
            confirmed_at = k + 1
        if on and on[0].get("depth_multipath_m") is not None:
            track_multipath = float(on[0]["depth_multipath_m"])
        off = [t for t in rep["tracks"] if t not in on]
        row = {"detected": best is not None, "true_range_m": rng, "true_bearing_deg": brg, "true_depth_m": float(pos[2]),
               "true_elevation_deg": float(np.degrees(np.arcsin((own[2] - pos[2]) / rng))), "other_detections": len(dets) - len(near)}
        if best is not None:
            first = best["arrivals"][0][0] if best.get("arrivals") else best.get("first_range_m")
            row.update(snr_db=best["snr_db"], range_error_m=best["range_m"] - rng, bearing_error_deg=best["bearing_deg"] - brg,
                       first_range_error_m=(first - rng) if first is not None else None,
                       cone_bearing_error_deg=(best["cone_bearing_deg"] - brg) if best.get("cone_bearing_deg") is not None else None)
            for key in ("elevation_deg", "elevation_sigma_deg", "elevation_ambiguous", "depth_m", "depth_sigma_m", "depth_profile_source", "path",
                        "depth_multipath_m", "depth_multipath_sigma_m", "multipath_reason", "multipath_note"):
                if key in best:
                    row[key] = best[key]
        out["per_ping"].append(row)
    out["track_confirmed_at_ping"] = confirmed_at
    out["tracks_elsewhere_last_ping"] = [{"range_m": float(t["range_m"]), "kind": t.get("kind")} for t in off] if reports else []
    out["track_multipath_depth_m"] = track_multipath
    return out


def run_one(name: str, which: str, engine: str = "cpp") -> dict:
    from gpusonar.live import LiveChain
    from gpusonar.sim import run_scenario
    info = SCENARIOS[name]
    cfg = water(name, which)
    t0 = time.time()
    result = run_scenario(copy.deepcopy(cfg))
    out = {"simulate_seconds": round(time.time() - t0, 1), "record_sound_speed_m_s": float(result.records[0].environment.sound_speed),
           "eigenrays": eigenray_table(result.config, result.records[0])}
    chain = load_yaml(ROOT / "configs" / "deploy" / f"{info['deploy']}.yaml")["chain"]
    told = copy.deepcopy(chain)
    told["multipath_depth"] = {}
    if info["rows"] > 1:
        ocean = result.config["propagation"]["environment"]                      # ssp_depth, ssp_speed: the LOCAL profile in both waters
        told["elevation"] = dict(told["elevation"], ssp={"depth": [float(v) for v in ocean["ssp_depth"]], "speed": [float(v) for v in ocean["ssp_speed"]]})
    for label, c in zip(PASSES, (chain, told)):
        with LiveChain(dict(c, engine=engine, backend="host")) as live:
            reports = live.run([r.copy() for r in result.records])
        out[label] = score(result.config, reports, info["rows"])
    return out


def run(names: list, store: Path, log=print) -> dict:
    res = json.loads(store.read_text()) if store.exists() else {}
    for name in names:
        for which in WATERS:
            if which in res.get(name, {}):
                continue
            res.setdefault(name, {})[which] = m = run_one(name, which)
            store.write_text(json.dumps(res, indent=1))
            d = m["as deployed"]["per_ping"]
            log(f"{name:36s} {which:20s} detected {sum(r['detected'] for r in d)}/{len(d)}, {m['simulate_seconds']:.0f} s to simulate")
    return res


def _med(rows, key, fmt="%.1f", signed=False):
    v = [r[key] for r in rows if r.get(key) is not None]
    if not v:
        return "-"
    return (("%+" + fmt[1:]) if signed else fmt) % float(np.median(v))


def _all(rows, key, fmt="%.1f"):
    return ", ".join("-" if r.get(key) is None else fmt % r[key] for r in rows)


def markdown(res: dict, commit: str = "") -> str:
    lines = ["# Water that changes with range against a chain that knows the local profile", "",
             f"Made by `tools/ssp_range_study.py`{(' at commit ' + commit) if commit else ''}. Each scenario is simulated twice (C++ synthesis; Bellhop's eigenrays): with its water as "
             "written (`environment.profiles_by_range`: Bellhop's 2-D sound-speed field) and with the profile at the sonar at every range. Same seed, target and noise. The records "
             "go through the C++ chain of the build's deployment file as shipped, and again with the local profile given to the elevation stage (stacked rows) and depth from "
             "multipath timing switched on with its defaults. The chain is never told that the water changes. Four pings each: a look at what happens, not statistics. "
             "The profiles are ILLUSTRATIVE: nobody measured them. Range is horizontal distance from the sonar and the section is the same on every bearing.", "",
             "Numbers are the median of the pings the target was detected on; in square brackets every ping. Range error: the detection's range minus the straight-line distance to the "
             "target, for the detection's strongest sample / for its first arrival. Elevation: positive up; `straight` is the straight line to the target, which is not where the "
             "sound arrives from in this water (see the eigenrays).", ""]
    for name, by_water in res.items():
        info = SCENARIOS[name]
        cfg = load_scenario_config(name, expand=False)
        pos, own = cfg["scene"][0]["positions"][0], cfg["platform"]["position"]
        lines += [f"## {info['title']} (`{name}`, chain of `configs/deploy/{info['deploy']}.yaml`)", "",
                  f"Target of {cfg['scene'][0]['target_strength_db']:g} dB, {pos[2]:g} m deep, {np.hypot(pos[0] - own[0], pos[1] - own[1]):.0f} m away horizontally; array at {own[2]:g} m; "
                  f"the record's sound speed is {next(iter(by_water.values()))['record_sound_speed_m_s']:g} m/s (the local profile at the array) in both waters.", "",
                  "| Water | Chain | detected, pings | SNR, dB | range error, m: strongest / first arrival | bearing error, degrees [every ping] | track on the target confirmed at ping | tracks elsewhere, last ping: range, m (kind) | other detections per ping |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for which, m in by_water.items():
            for label in PASSES:
                d = m[label]
                hit = [r for r in d["per_ping"] if r["detected"]]
                lines.append(f"| {which} | {label} | {len(hit)}/{d['pings']} | {_med(hit, 'snr_db')} | {_med(hit, 'range_error_m', signed=True)} / {_med(hit, 'first_range_error_m', signed=True)} | "
                             f"{_med(hit, 'bearing_error_deg', '%.2f', signed=True)} [{_all(hit, 'bearing_error_deg', '%+.2f')}] | {d['track_confirmed_at_ping'] or 'none'} | "
                             f"{', '.join('%.0f (%s)' % (t['range_m'], t['kind']) for t in d['tracks_elsewhere_last_ping']) or 'none'} | {np.mean([r['other_detections'] for r in d['per_ping']]):.1f} |")
        lines += ["", "Depth of the target's detection:", "",
                  "| Water | Chain | elevation, degrees (straight) [every ping] | pings flagged ambiguous | depth from the rows, m (true) [every ping] | its stated one sigma, m | depth from multipath timing, m: [every ping]; track | multipath stage's reason or note |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        for which, m in by_water.items():
            for label in PASSES:
                d = m[label]
                hit = [r for r in d["per_ping"] if r["detected"]]
                if not hit:
                    lines.append(f"| {which} | {label} | not detected | | | | | |")
                    continue
                why = ", ".join(sorted({str(r[k]) for r in hit for k in ("multipath_reason", "multipath_note") if r.get(k)})) or "-"
                track = d.get("track_multipath_depth_m")
                lines.append(f"| {which} | {label} | {_med(hit, 'elevation_deg', '%.2f', signed=True)} ({hit[0]['true_elevation_deg']:+.2f}) [{_all(hit, 'elevation_deg', '%+.2f')}] | "
                             f"{sum(1 for r in hit if r.get('elevation_ambiguous'))}/{len(hit)} | "
                             f"{_med(hit, 'depth_m')} ({hit[0]['true_depth_m']:.1f}) [{_all(hit, 'depth_m')}] | {_all(hit, 'depth_sigma_m')} | "
                             f"[{_all(hit, 'depth_multipath_m')}]; {'-' if track is None else '%.1f' % track} | {why} |")
        lines += ["", "The simulated sound: Bellhop's arrivals from the array to the target (one way, first ping), the six strongest of those kept (the scenario's `max_arrivals`). Range excess: the arrival's delay times the record's "
                  "sound speed, minus the straight-line distance.", "",
                  "| Water | level re the strongest, dB | delay, ms | range excess, m | angle at the array, degrees (up positive) | surface / bottom reflections |",
                  "| --- | --- | --- | --- | --- | --- |"]
        for which, m in by_water.items():
            for r in m["eigenrays"][:6]:
                lines.append(f"| {which} | {r['level_db']:.1f} | {r['delay_ms']:.2f} | {r['range_excess_m']:+.1f} | {r['elevation_at_array_deg']:+.2f} | {r['surface']} / {r['bottom']} |")
        lines.append("")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenarios", default=",".join(SCENARIOS))
    ap.add_argument("--json", default=str(ROOT / "docs" / "figures" / "ssp_range_study.json"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--commit", default="")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args(argv)
    store = Path(args.json)
    if args.report_only:
        res = json.loads(store.read_text())
    else:
        from gpusonar.sim.bellhop import find_bellhop
        if find_bellhop() is None:
            print("Bellhop is not installed: set GPUSONAR_BELLHOP to bellhop.exe", file=sys.stderr)
            return 2
        os.environ.setdefault("OMP_NUM_THREADS", "2")
        res = run([s for s in args.scenarios.split(",") if s], store)
    text = markdown(res, args.commit)
    if args.out:
        Path(args.out).write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
