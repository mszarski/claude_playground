#!/usr/bin/env python3
"""A trial in fjord or harbour water against the open-water rehearsal, through the live chain.

    GPUSONAR_BELLHOP=.../bellhop.exe tools/fjord_rehearsal.py [--builds testauv_1row,static_2row,static_3row] [--variants ...]
                                     [--ranges 300,1000,2000] [--json docs/figures/fjord_rehearsal.json] [--out docs/figures/fjord_rehearsal.md]

For each of the builds trialled first (the one-row prototype on the test AUV, the two- and three-row static arrays)
the shipped open-water rehearsal (`popoto_*_trial_*`) and the same sonar, platform, target and window in fjord-type
water (`fjord_*_trial_*`: a sound-speed maximum at 19 m, basins and sills, Bellhop; the basin's late reverberation;
a calm night's heavy-tailed noise) are simulated and passed through the C++ chain of the build's own deployment
file, one variant at a time:

    open water            the shipped rehearsal, unchanged
    profile, section      the fjord's water and seabed through Bellhop; no late tail; the open rehearsal's noise
    + late tail           + `basin_reverberation` behind the echoes (the measured set named in the scenario)
    + own ping, what-if   + `own_ping_db`: the basin's answer to the ping itself at the level the late field had
                          274 m from a source (-43 dB re the source's energy at 1 m). NOT measured at a source.
    fjord as shipped      the scenario file as it is: late tail, noise of a 1.9 m/s wind, K-distributed (shape 3.5)
    + machinery noise     as shipped with 63 dB re 1 uPa^2/Hz, flat (what the hydrophone by the research station read)

Scored on every ping: the target's detection (the strongest whose ranges, first arrival to the end of what it spans,
reach from 30 m before to 60 m behind where the target is, within 2 degrees: the detector reports an echo and the tail
behind it as ONE detection at their energy-weighted range, which lies tens of metres behind the target) and its SNR
over the chain's local background; the other detections, split into those behind the target on its bearing (within 3
degrees, up to 400 m later: later arrivals) and the rest; the ping at which a track on the target is confirmed and the tracks that are not on
it. For the fjord as shipped and the own-ping what-if the same records also go through the chain with settings one
would try against reverberation (`SETTINGS`): a higher detection threshold, a short CFAR window, both. With stacked rows also the elevation and depth of the target's detection, with the
deployment file's sound-speed assumption (a gradient prior about the local value) and with the profile given to the
chain (`chain.elevation.ssp`), both from the same records. `--ranges` repeats three variants with the target moved out
along its bearing (the one-row build). Restartable: the JSON is rewritten after every run. One process, one run at a time.

The chain is the one on the branch this is run from (state the commit with the numbers); nothing of it is changed.
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

BUILDS = {"testauv_1row": {"title": "Test AUV, one row of 32", "open": "popoto_testauv_trial_1row", "fjord": "fjord_testauv_trial_1row", "deploy": "popoto_testauv_1row", "rows": 1},
          "static_2row": {"title": "Static array, two rows", "open": "popoto_static_trial_2row", "fjord": "fjord_static_trial_2row", "deploy": "popoto_static_2row", "rows": 2},
          "static_3row": {"title": "Static array, three rows", "open": "popoto_static_trial_3row", "fjord": "fjord_static_trial_3row", "deploy": "popoto_static_3row", "rows": 3}}
OWN_PING_DB = -43.1                 # basin.MEASURED["smartocean_274m"]["own_ping_equivalent_db"]: a stand-in, not a measurement at the source
VARIANTS = ["open water", "profile, section", "+ late tail", "+ own ping, what-if", "fjord as shipped", "+ machinery noise"]
# Chain settings tried on the same records (changes to the deployment file's `chain`).
SETTINGS = {"min_snr_db 15": {"min_snr_db": 15.0},
            "cfar guard 30 m, train 100 m": {"cfar": {"guard_m": 30.0, "train_m": 100.0}},
            "min_snr_db 15, cfar guard 30 m, train 100 m": {"min_snr_db": 15.0, "cfar": {"guard_m": 30.0, "train_m": 100.0}},
            "min_snr_db 15, confirm on 4 of 4": {"min_snr_db": 15.0, "tracker": {"confirm_hits": 4}}}


# Three variants keep the open-water rehearsal's noise in the fjord's water, so that one thing changes at a time. That
# noise names a wind and a shipping level the fjord's environment does not have: the scenario loader refuses such a
# contradiction unless it is named with its reason (python/gpusonar/environment.py, `environment_overrides`).
OPEN_NOISE_REASON = ("the open-water rehearsal's noise is kept in this variant of the fjord scenario, so that only the water, the seabed and the late tail "
                     "change against the open-water run (tools/fjord_rehearsal.py); the fjord's own calm-night noise comes with the variant 'fjord as shipped'")


def with_open_noise(cfg: dict, build: dict) -> dict:
    """The fjord scenario with the open-water rehearsal's ambient noise in place of its own, the contradiction with
    the fjord's environment named: every value of that noise the environment also gives, and gives differently."""
    from gpusonar import environment
    noise = copy.deepcopy(load_scenario_config(build["open"])["receiver"]["ambient_noise"])
    cfg["receiver"]["ambient_noise"] = noise
    site = environment.resolve(cfg["environment"])                  # the fjord's site: `wind_speed`, `shipping`
    named = dict(cfg.get("environment_overrides") or {})
    for key in ("wind_speed", "shipping"):
        if key in noise and site.get(key) is not None and float(noise[key]) != float(site[key]):
            named[f"receiver.ambient_noise.{key}"] = OPEN_NOISE_REASON
    cfg["environment_overrides"] = named
    return cfg


def variant(build: dict, name: str) -> dict:
    if name == "open water":
        return load_scenario_config(build["open"])
    cfg = load_scenario_config(build["fjord"])
    if name == "profile, section":
        cfg.pop("basin_reverberation")
        with_open_noise(cfg, build)
    elif name == "+ late tail":
        with_open_noise(cfg, build)
    elif name == "+ own ping, what-if":
        with_open_noise(cfg, build)
        cfg["basin_reverberation"] = dict(cfg["basin_reverberation"], own_ping_db=OWN_PING_DB)
    elif name == "+ machinery noise":
        cfg["receiver"]["ambient_noise"] = {"level_db": 63.0}
    elif name != "fjord as shipped":
        raise ValueError(name)
    return cfg


def at_range(cfg: dict, range_m: float) -> dict:
    """The scenario with its target moved out along its bearing to `range_m` (horizontal), the window and the
    Bellhop grid moved with it, the section extended."""
    from gpusonar.sim import basin
    cfg = copy.deepcopy(cfg)
    own = np.asarray(cfg["platform"]["position"], dtype=float)
    pos = np.asarray(cfg["scene"][0]["positions"][0], dtype=float)
    d = pos[:2] - own[:2]
    pos[:2] = own[:2] + d * (range_m / np.linalg.norm(d))
    cfg["scene"][0]["positions"] = [pos.tolist()]
    pulse = float((cfg.get("sonar_overrides", {}).get("transmit", {}).get("waveform", {}) or {}).get("duration", 0.1))
    lo = max(range_m - 350.0, 0.5 * 1500.0 * pulse + 20.0)
    cfg.setdefault("sonar_overrides", {}).setdefault("receive", {})["window"] = {"range": [float(lo), float(range_m + 400.0)]}
    if cfg.get("fidelity") == "eigenray":
        prop = cfg["propagation"]
        prop.update(min_range=range_m - 60.0, max_range=range_m + 60.0)
        prop["environment"]["bathymetry"] = basin.basin_section(range_m + 500.0, step=100.0)
    return cfg


def _span(d: dict) -> tuple:
    """The ranges a detection covers (as tools/popoto_12km_study.py). The chain's detector (cell_statistic: multiscale)
    reports ONE detection for a group of arrivals: `range_m` is its energy-weighted mean, `first_range_m` its first arrival
    and `extent_m` what it spans; a detection without those fields is a point. With the basin's tail behind the echo the
    mean lies tens of metres behind the target (63 m for the two-row static array at 2.6 km), so a window about `range_m`
    alone scored a 44 dB detection of the target as "behind" it."""
    ext = float(d.get("extent_m") or 0.0)
    lo = float(d["first_range_m"]) if d.get("first_range_m") is not None else float(d["range_m"]) - 0.5 * ext
    return lo, max(lo + ext, float(d["range_m"]))


def score(cfg: dict, reports: list, rows: int) -> dict:
    target = cfg["scene"][0]
    pos0 = np.asarray(target["positions"][0], dtype=float)
    out = {"pings": len(reports), "per_ping": []}
    confirmed_at = None
    for k, rep in enumerate(reports):
        own = np.asarray(rep["ownship_xyz"])
        rng = float(np.linalg.norm(pos0 - own))
        brg = float(np.degrees(np.arctan2(pos0[0] - own[0], pos0[1] - own[1])))
        dets = [d for d in rep["detections"] if d.get("snr_db") is not None]
        # The target's: a detection whose ranges (first arrival to the end of what it spans) reach from 30 m before the
        # target to 60 m behind it, within 2 degrees.
        near = [d for d in dets if _span(d)[0] <= rng + 60.0 and _span(d)[1] >= rng - 30.0 and abs(d["bearing_deg"] - brg) <= 2.0]
        best = max(near, key=lambda d: d["snr_db"]) if near else None
        behind = [d for d in dets if d not in near and abs(d["bearing_deg"] - brg) <= 3.0 and rng - 30.0 < d["range_m"] <= rng + 400.0]
        other = [d for d in dets if d not in near and d not in behind]
        reach = max([_span(d)[1] for d in near], default=rng)        # how far behind the target its own detection extends on this ping
        tracks_on = [t for t in rep["tracks"] if -40.0 <= t["range_m"] - rng <= max(70.0, reach - rng + 30.0) and abs(t.get("bearing_deg", brg) - brg) <= 3.0]
        if tracks_on and confirmed_at is None:
            confirmed_at = k + 1
        row = {"detected": best is not None, "snr_db": best["snr_db"] if best else None, "range_error_m": (best["range_m"] - rng) if best else None,
               "first_range_error_m": (_span(best)[0] - rng) if best else None, "extent_m": best.get("extent_m") if best else None,
               "bearing_error_deg": (best["bearing_deg"] - brg) if best else None, "detections": len(dets), "behind_target": len(behind), "elsewhere": len(other),
               "behind_max_snr_db": max((d["snr_db"] for d in behind), default=None), "elsewhere_max_snr_db": max((d["snr_db"] for d in other), default=None),
               "tracks": len(rep["tracks"]), "tracks_off_target": len(rep["tracks"]) - len(tracks_on), "true_range_m": rng, "true_bearing_deg": brg,
               "all": [[round(d["range_m"], 1), round(d["bearing_deg"], 2), round(d["snr_db"], 1), round(_span(d)[0], 1), round(_span(d)[1], 1)] for d in dets]}
        if best is not None and rows > 1:
            row.update(elevation_deg=best.get("elevation_deg"), elevation_sigma_deg=best.get("elevation_sigma_deg"), elevation_ambiguous=best.get("elevation_ambiguous"),
                       candidates=len(best.get("elevation_alternatives_deg") or []) + (best.get("elevation_deg") is not None),
                       depth_m=best.get("depth_m"), depth_sigma_m=best.get("depth_sigma_m"), depth_profile_source=best.get("depth_profile_source"),
                       true_elevation_deg=float(np.degrees(np.arcsin((own[2] - pos0[2]) / rng))), true_depth_m=float(pos0[2]))
        out["per_ping"].append(row)
    hit = [r for r in out["per_ping"] if r["detected"]]
    out["pings_detected"] = len(hit)
    out["median_snr_db"] = float(np.median([r["snr_db"] for r in hit])) if hit else None
    out["track_confirmed_at_ping"] = confirmed_at
    out["behind_target_per_ping"] = float(np.mean([r["behind_target"] for r in out["per_ping"]]))
    out["elsewhere_per_ping"] = float(np.mean([r["elsewhere"] for r in out["per_ping"]]))
    out["tracks_off_target_last_ping"] = out["per_ping"][-1]["tracks_off_target"]
    if rows > 1 and hit:
        depth = [r["depth_m"] for r in hit if r.get("depth_m") is not None]
        elev = [r["elevation_deg"] for r in hit if r.get("elevation_deg") is not None]
        out["depth_m"] = {"median": float(np.median(depth)) if depth else None, "all": depth, "true": hit[0]["true_depth_m"],
                          "sigma": [r.get("depth_sigma_m") for r in hit], "source": sorted({str(r.get("depth_profile_source")) for r in hit})}
        out["elevation_deg"] = {"median": float(np.median(elev)) if elev else None, "all": elev, "true": hit[0]["true_elevation_deg"],
                                "ambiguous_pings": sum(1 for r in hit if r.get("elevation_ambiguous")), "candidates": [r.get("candidates") for r in hit]}
    return out


def run_one(cfg: dict, build: dict, engine: str = "cpp", settings: dict | None = None) -> dict:
    """Simulate (C++ synthesis), then the chain of the build's deployment file; with stacked rows a second pass of
    the same records with the scenario's profile given to the chain."""
    from gpusonar.live import LiveChain
    from gpusonar.sim import run_scenario
    t0 = time.time()
    result = run_scenario(copy.deepcopy(cfg))
    t_sim = time.time() - t0
    records = result.records
    chain = load_yaml(ROOT / "configs" / "deploy" / f"{build['deploy']}.yaml")["chain"]
    passes = {"as deployed": chain}
    env = (cfg.get("propagation") or {}).get("environment") or {}
    if build["rows"] > 1 and env.get("ssp_depth"):
        with_profile = copy.deepcopy(chain)
        with_profile["elevation"] = dict(with_profile["elevation"], ssp={"depth": list(env["ssp_depth"]), "speed": list(env["ssp_speed"])})
        passes["profile given"] = with_profile
    out = {"simulate_seconds": round(t_sim, 1)}
    for name, change in (settings or {}).items():
        c = copy.deepcopy(chain)
        for key, value in change.items():
            c[key] = dict(c.get(key) or {}, **value) if isinstance(value, dict) else value
        passes[name] = c
    for name, c in passes.items():
        t1 = time.time()
        with LiveChain(dict(c, engine=engine, backend="host")) as live:
            reports = live.run([r.copy() for r in records])
        out[name] = score(cfg, reports, build["rows"])
        out[name]["chain_seconds_per_ping"] = float(np.median([r["processing_seconds"] for r in reports]))
        out[name]["chain_seconds"] = round(time.time() - t1, 1)
    return out


def run(builds: list, variants: list, ranges: list, store: Path, log=print) -> dict:
    res = json.loads(store.read_text()) if store.exists() else {}
    todo = [(b, v, None) for b in builds for v in variants]
    todo += [(b, v, r) for b in builds if BUILDS[b]["rows"] == 1 for r in ranges for v in ("open water", "profile, section", "+ late tail", "+ own ping, what-if")]
    for b, v, r in todo:
        key = v if r is None else f"{v} @ {int(r)} m"
        if key in res.get(b, {}):
            continue
        cfg = variant(BUILDS[b], v)
        if r is not None:
            cfg = at_range(cfg, float(r))
        try:
            m = run_one(cfg, BUILDS[b], settings=SETTINGS if v in ("fjord as shipped", "+ own ping, what-if") else None)
        except Exception as err:                                           # noqa: BLE001  (a failed variant is reported, the rest still run)
            m = {"error": f"{type(err).__name__}: {err}"}
        res.setdefault(b, {})[key] = m
        store.write_text(json.dumps(res, indent=1))
        d = m.get("as deployed", {})
        log(f"{b:14s} {key:34s} " + (m["error"] if "error" in m else
            f"detected {d['pings_detected']}/{d['pings']} SNR {d['median_snr_db']} dB, behind {d['behind_target_per_ping']:.1f} elsewhere {d['elsewhere_per_ping']:.1f} per ping, "
            f"track at {d['track_confirmed_at_ping']}, off-target tracks {d['tracks_off_target_last_ping']} ({m['simulate_seconds']:.0f} s sim, {d['chain_seconds']:.0f} s chain)"))
    return res


def _snr(d):
    return "not detected" if d.get("median_snr_db") is None else f"{d['median_snr_db']:.1f}"


def markdown(res: dict, commit: str = "") -> str:
    lines = ["# A trial in fjord-type water against the open-water rehearsal", "",
             f"Made by `tools/fjord_rehearsal.py`{(' at commit ' + commit) if commit else ''}: each variant simulated (C++ synthesis, Bellhop for the fjord's water) and passed through the "
             "C++ chain of the build's deployment file as shipped. Four pings each. SNR: the target's detection over the chain's local background, median of the pings it was "
             "detected on (the strongest whose ranges, from its first arrival to the end of what it spans, reach from 30 m before the target to 60 m behind it: the detector reports the echo and "
             "the tail behind it as one detection at their energy-weighted range). Range error: that range / its first arrival, against the target's. Other detections per ping: behind the target "
             "on its bearing (within 3 degrees, up to 400 m later) / anywhere else. The fjord's profile, section and seabed are "
             "ILLUSTRATIVE (modelled on one site); the late tail's numbers are one fjord, one night, 4-8 kHz; the own-ping level is a what-if that nothing measured.", ""]
    for b, rows in res.items():
        build = BUILDS[b]
        cfg = load_scenario_config(build["fjord"])
        pos, own = cfg["scene"][0]["positions"][0], cfg["platform"]["position"]
        lines += [f"## {build['title']} (`{build['open']}` against `{build['fjord']}`)", "",
                  f"Target of {cfg['scene'][0]['target_strength_db']:g} dB at {np.hypot(pos[0] - own[0], pos[1] - own[1]):.0f} m, {pos[2]:g} m deep; array at {own[2]:g} m.", "",
                  "| Variant | detected, pings | SNR, dB | range error, m: detection / its first arrival | other detections per ping: behind / elsewhere | strongest behind / elsewhere, dB | track on the target confirmed at ping | tracks not on the target (last ping) |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        for key, m in rows.items():
            if "error" in m:
                lines.append(f"| {key} | {m['error']} | | | | | | |")
                continue
            d = m["as deployed"]
            err = [r["range_error_m"] for r in d["per_ping"] if r["detected"]]
            first = [r["first_range_error_m"] for r in d["per_ping"] if r["detected"] and r.get("first_range_error_m") is not None]
            mx = lambda k: max((r[k] for r in d["per_ping"] if r[k] is not None), default=None)   # noqa: E731
            f = lambda v: "-" if v is None else f"{v:.1f}"                                         # noqa: E731
            lines.append(f"| {key} | {d['pings_detected']}/{d['pings']} | {_snr(d)} | {('%+.1f' % np.median(err)) if err else '-'} / {('%+.1f' % np.median(first)) if first else '-'} | "
                         f"{d['behind_target_per_ping']:.1f} / {d['elsewhere_per_ping']:.1f} | {f(mx('behind_max_snr_db'))} / {f(mx('elsewhere_max_snr_db'))} | "
                         f"{d['track_confirmed_at_ping'] or 'none'} | {d['tracks_off_target_last_ping']} |")
        tried = [(key, name, m[name]) for key, m in rows.items() for name in SETTINGS if name in m]
        if tried:
            lines += ["", "The same records through the chain with other settings:", "",
                      "| Variant | chain setting | detected, pings | SNR, dB | other detections per ping: behind / elsewhere | track on the target confirmed at ping | tracks not on the target (last ping) |",
                      "| --- | --- | --- | --- | --- | --- | --- |"]
            for key, name, d in tried:
                lines.append(f"| {key} | {name} | {d['pings_detected']}/{d['pings']} | {_snr(d)} | {d['behind_target_per_ping']:.1f} / {d['elsewhere_per_ping']:.1f} | "
                             f"{d['track_confirmed_at_ping'] or 'none'} | {d['tracks_off_target_last_ping']} |")
        if build["rows"] > 1:
            lines += ["", "Elevation and depth of the target's detection (median of the pings; in brackets every ping):", "",
                      "| Variant | sound speed known to the chain | elevation, degrees (true) | pings with several candidates | depth, m (true) | stated one sigma, m | profile rung used |",
                      "| --- | --- | --- | --- | --- | --- | --- |"]
            for key, m in rows.items():
                for name in ("as deployed", "profile given"):
                    d = m.get(name)
                    if not d or "depth_m" not in d:
                        continue
                    e, z = d["elevation_deg"], d["depth_m"]
                    fmt = lambda v, s="%.1f": "-" if v is None else s % v                           # noqa: E731
                    lines.append(f"| {key} | {name} | {fmt(e['median'], '%+.2f')} ({e['true']:+.2f}) [{', '.join(fmt(v, '%+.2f') for v in e['all'])}] | "
                                 f"{sum(1 for c in e['candidates'] if (c or 0) > 1)}/{len(e['candidates'])} | {fmt(z['median'])} ({z['true']:.1f}) [{', '.join(fmt(v) for v in z['all'])}] | "
                                 f"{', '.join(fmt(v) for v in z['sigma'])} | {', '.join(z['source'])} |")
        lines.append("")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--builds", default=",".join(BUILDS))
    ap.add_argument("--variants", default=";".join(VARIANTS), help="semicolon-separated")
    ap.add_argument("--ranges", default="", help="comma-separated ranges (m) to repeat three variants at, for the one-row build")
    ap.add_argument("--json", default=str(ROOT / "docs" / "figures" / "fjord_rehearsal.json"))
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
        res = run([b for b in args.builds.split(",") if b], [v for v in args.variants.split(";") if v], [float(r) for r in args.ranges.split(",") if r], store)
    text = markdown(res, args.commit)
    if args.out:
        Path(args.out).write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
