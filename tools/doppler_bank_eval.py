#!/usr/bin/env python3
"""The sweep's Doppler bank (D4) end to end through the live chain, both engines: each target's margin, detection and
track with the bank off and on, the false detections on the scene's background, and the cost a ping.

    tools/doppler_bank_eval.py SCENE [--pings 6] [--engines python cpp] [--background 4] [--max-rate 3.0] [--json FILE]

What is run (ours). The scene as shipped (`configs/scenarios/SCENE.yaml`, its own seed and pings; the eigenray scenes
need GPUSONAR_BELLHOP) is simulated once; its records go through `LiveChain` with the deployment defaults (bearings -40
to 40 degrees every 0.5, backprojection, multiscale detector, sizes derived from the sonar; host backend), engine by
engine, without the bank and with `doppler: {mode: sweep_bank, max_range_rate_m_s: --max-rate}`. Per target and ping:

- the detection it gives: the strongest detection within 60 m of range and 1.5 degrees of bearing of where the target
  is at that ping (its centre, as seen from the platform at that ping: slant range from the platform, bearing from
  broadside towards the direction the array's elements run, i.e. atan2(north, east) for a platform heading north);
- its margin over the detector's threshold: snr_db (its strongest sample over the detector's background) less the
  single-sample test's threshold of the plan in use (`threshold_db` of the plan's first scale, which carries the bank's
  shift when the bank is on). Off -> on: the gross change is that of snr_db (the background of the maximum map is
  higher than one map's: part of the gain goes there); the net change is that of the margin (the threshold moved by
  the measured shift as well: the false-alarm rate held);
- whether a track is within 100 m of the target at the last ping, and its class.

The background: the scene with its targets removed (reverberation and noise as the scene has them, new draws), --background
pings, counted at the default budget and at 3 false detections a ping (cfar.false_alarms_per_ping), bank off and on.

Cost: the chain's processing_seconds a ping (the first ping left out), per engine, off and on.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from gpusonar.config import load_scenario_config  # noqa: E402
from gpusonar.live import LiveChain  # noqa: E402
from gpusonar.sim.simulator import run_scenario  # noqa: E402

GATE_RANGE_M, GATE_BEARING_DEG, TRACK_GATE_M = 60.0, 1.5, 100.0


def say(text: str) -> None:
    print(text, flush=True)


def centre(item) -> np.ndarray:
    if "positions" in item:
        return np.mean(np.asarray(item["positions"], dtype=float), axis=0)
    return np.asarray(item["origin"], dtype=float)


def truth(cfg: dict, n: int) -> dict:
    """Per target label: per ping (slant range m, bearing deg, x north, y east, range rate m/s opening) from the platform."""
    plat = cfg.get("platform", {})
    p0, vp = np.asarray(plat.get("position", [0, 0, 0]), dtype=float), np.asarray(plat.get("velocity", [0, 0, 0]), dtype=float)
    dt = float(cfg.get("pings", {}).get("interval", 1.0))
    out = {}
    for item in cfg.get("scene", []):
        label = item.get("label")
        if not label:
            continue
        c0, v = centre(item), np.asarray(item.get("velocity", [0, 0, 0]), dtype=float)
        rows = []
        for i in range(n):
            t = i * dt
            p, c = p0 + vp * t, c0 + v * t
            d = c - p
            u = d / np.linalg.norm(d)
            rows.append({"range": float(np.linalg.norm(d)), "bearing": float(np.degrees(np.arctan2(d[0], d[1]))), "x": float(c[0]), "y": float(c[1]),
                         "range_rate": float((v - vp) @ u)})
        out[label] = rows
    return out


def chain_config(engine: str, bank: bool, max_rate: float, budget: float | None = None) -> dict:
    cfg = {"engine": engine, "backend": "host"}
    if bank:
        cfg["doppler"] = {"mode": "sweep_bank", "max_range_rate_m_s": max_rate, "report": True}
    if budget is not None:
        cfg["cfar"] = {"false_alarms_per_ping": budget}
    return cfg


def run_chain(records, cfg: dict):
    reports, seconds, thresholds = [], [], []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with LiveChain(cfg) as chain:
            for r in records:
                t = time.perf_counter()
                rep = chain.process(r.copy())
                seconds.append(time.perf_counter() - t)
                reports.append(rep)
                thresholds.append(float(chain.detector["scales"][0]["threshold_db"]))
            if not cfg.get("doppler"):
                bank = None
            elif cfg["engine"] == "python":
                bank = json.loads(chain._bank["json"])
            else:
                bank = json.loads(chain._cpp.doppler())
    return reports, seconds, thresholds, bank


def per_target(reports, thresholds, truths: dict) -> dict:
    out = {}
    for label, rows in truths.items():
        pings = []
        for rep, thr, tr in zip(reports, thresholds, rows):
            near = [d for d in rep["detections"] if abs(d["range_m"] - tr["range"]) < GATE_RANGE_M and abs(d["bearing_deg"] - tr["bearing"]) < GATE_BEARING_DEG]
            d = max(near, key=lambda q: q["snr_db"]) if near else None
            pings.append({"snr_db": d["snr_db"] if d else None, "threshold_db": thr, "margin_db": d["snr_db"] - thr if d else None,
                          "range_rate_m_s": d.get("range_rate_m_s") if d else None, "true_range_rate_m_s": tr["range_rate"]})
        last, tr = reports[-1], rows[len(reports) - 1]
        tracks = [t for t in last["tracks"] if np.hypot(t["x"] - tr["x"], t["y"] - tr["y"]) < TRACK_GATE_M]
        out[label] = {"pings": pings, "detected": sum(p["snr_db"] is not None for p in pings),
                      "tracked": bool(tracks), "kind": tracks[0]["kind"] if tracks else None}
    return out


def summary(a: dict, b: dict) -> dict:
    """Off (a) -> on (b) for one target: mean over the pings both detect."""
    both = [(p, q) for p, q in zip(a["pings"], b["pings"]) if p["snr_db"] is not None and q["snr_db"] is not None]
    if not both:
        return {"pings_both": 0}
    gross = float(np.mean([q["snr_db"] - p["snr_db"] for p, q in both]))
    net = float(np.mean([q["margin_db"] - p["margin_db"] for p, q in both]))
    return {"pings_both": len(both), "margin_off_db": float(np.mean([p["margin_db"] for p, _ in both])),
            "margin_on_db": float(np.mean([q["margin_db"] for _, q in both])), "gross_db": gross, "net_db": net,
            "detected_off": a["detected"], "detected_on": b["detected"], "tracked_off": a["tracked"], "tracked_on": b["tracked"],
            "kind_off": a["kind"], "kind_on": b["kind"],
            "range_rate_on": [q["range_rate_m_s"] for _, q in both], "true_range_rate": [q["true_range_rate_m_s"] for _, q in both]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene")
    ap.add_argument("--pings", type=int)
    ap.add_argument("--engines", nargs="+", default=["python", "cpp"], choices=["python", "cpp"])
    ap.add_argument("--background", type=int, default=4)
    ap.add_argument("--max-rate", type=float, default=3.0)
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    cfg = load_scenario_config(a.scene, expand=False)
    if a.pings:
        cfg["pings"] = dict(cfg.get("pings", {}), count=int(a.pings))
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        records = run_scenario(copy.deepcopy(cfg)).records
    say(f"{a.scene}: {len(records)} pings simulated in {time.time() - t0:.0f} s, {records[0].samples.n_channels} channels")
    truths = truth(cfg, len(records))
    result = {"scene": a.scene, "pings": len(records), "channels": int(records[0].samples.n_channels), "max_range_rate_m_s": a.max_rate,
              "engines": {}, "background": {}}
    for engine in a.engines:
        res = {}
        for bank in (False, True):
            reports, seconds, thr, plan = run_chain(records, chain_config(engine, bank, a.max_rate))
            res["on" if bank else "off"] = {"targets": per_target(reports, thr, truths), "seconds": seconds,
                                             "seconds_median_after_first": float(np.median(seconds[1:])) if len(seconds) > 1 else seconds[0],
                                             "detections_per_ping": [len(r["detections"]) for r in reports],
                                             "threshold_db": thr[-1], "bank": plan}
            say(f"{a.scene} {engine} bank {'on ' if bank else 'off'}: {np.median(seconds[1:]) if len(seconds) > 1 else seconds[0]:.2f} s a ping, "
                f"threshold {thr[-1]:.2f} dB, detections a ping {[len(r['detections']) for r in reports]}")
        res["summary"] = {label: summary(res["off"]["targets"][label], res["on"]["targets"][label]) for label in truths}
        for label, s in sorted(res["summary"].items(), key=lambda kv: kv[1].get("net_db", -1e9)):
            if s["pings_both"]:
                say(f"  {label:14s} {engine}: margin {s['margin_off_db']:6.2f} -> {s['margin_on_db']:6.2f} dB (gross {s['gross_db']:+.2f}, net {s['net_db']:+.2f}); "
                    f"detected {s['detected_off']} -> {s['detected_on']} of {len(records)}; tracked {s['tracked_off']} ({s['kind_off']}) -> "
                    f"{s['tracked_on']} ({s['kind_on']}); range rate {np.round(s['range_rate_on'], 3).tolist()} (true {np.round(s['true_range_rate'], 2).tolist()})")
            else:
                say(f"  {label:14s} {engine}: not detected on any ping by both (off {res['off']['targets'][label]['detected']}, on {res['on']['targets'][label]['detected']})")
        result["engines"][engine] = res
    if a.background > 0:
        bcfg = copy.deepcopy(cfg)
        bcfg["scene"] = []
        bcfg["pings"] = dict(bcfg.get("pings", {}), count=int(a.background))
        bcfg["seed"] = int(cfg.get("seed", 0)) + 1000
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            back = run_scenario(bcfg).records
        for budget in (None, 3.0):
            row = {}
            for bank in (False, True):
                reports, _, _, _ = run_chain(back, chain_config("cpp", bank, a.max_rate, budget))
                row["on" if bank else "off"] = sum(len(r["detections"]) for r in reports)
            result["background"]["default" if budget is None else str(budget)] = dict(row, pings=len(back))
            say(f"{a.scene} background, {len(back)} pings, budget {'default' if budget is None else budget}: false detections off {row['off']}, on {row['on']}")
    if a.json:
        Path(a.json).write_text(json.dumps(result, indent=1, default=float) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
