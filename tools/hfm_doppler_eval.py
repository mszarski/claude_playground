#!/usr/bin/env python3
"""The linear and the hyperbolic sweep on a mover through the live chain, plain matched filter (every Doppler option off),
both engines: each target's detection margin and range error against its radial speed, relative to the same waveform,
pulse and energy at zero speed (docs/tasks/DOPPLER.md "HFM"; ours).

    tools/hfm_doppler_eval.py [--pulses 1 2 4 8] [--waveforms lfm hfm] [--speeds 0 2 5.6 8] [--pings 3]
                              [--ts-offset-db 0] [--bank] [--bank-engines python cpp] [--engines python cpp] [--json FILE]

The scene (ours, a reduced form of configs/scenarios/popoto_12km_large_submerged.yaml; that file is not changed): its
sonar (`popoto_long_range`, one row of 32, 4-6 kHz), receive window (11.3 to 12.7 km at 8 kHz baseband), environment,
ambient noise (Wenz) and seed, but FREE FIELD (no multipath), no reverberation and a STILL platform (so a target's radial
speed is its speed along the line of sight, and no beam has an own-ship rate). Two targets 12 km out, each closing at the
speed of the case straight along its line of sight:

- `point`: one highlight of the class's low end, +25 dB (configs/builds/targets/reference_classes.yaml, assumed), at -10 deg;
- `hull`: five highlights of +18.01 dB 25 m apart along a 100 m hull seen bow-on (+25.00 dB in all, as the 12 km scenes
  build `large_submerged`), at +10 deg.

`--ts-offset-db` adds to both (the losses of a long linear sweep are larger than a +25 dB target's margin: the relative
figures need it). Each waveform at each pulse length T is sent at the shipped source level (197 dB rms), so the energy is
197 + 10 log10(T (1 - 5 alpha / 8)) dB re 1 uPa^2 s at 1 m for both (Tukey 0.1: the same envelope energy, sonar.py
`envelope_energy_s`): the LFM is the analytic `type: lfm`, the HFM `type: sampled, source: hfm` with the same band, length
and taper. The chain: `LiveChain` with the deployment defaults (as tools/doppler_bank_eval.py), host backend.

Per target and ping: the strongest detection within --gate-range-m of the target's range where the pulse reaches it (the
HFM reads f1 T / B metres per m/s nearer: 192 m at 8 s and 8 m/s) and 1.5 degrees; margin = snr_db less the single-sample
threshold of the plan; range error = the detection's strongest arrival less the true range (the point's; the hull's BOW
highlight, the nearest). The closed forms (gpusonar.doppler `hfm_doppler`, the C++ library's `sweep_mismatch` for the LFM)
are written beside each case. `--bank`: the LFM cases also with `doppler: {mode: sweep_bank, combine: cfar,
max_range_rate_m_s: max speed}` (45 hypotheses a beam at 1 s and 8 m/s; 89 at 2 s): the engines of `--bank-engines`.
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

from gpusonar import core, doppler  # noqa: E402
from gpusonar.config import load_scenario_config  # noqa: E402
from gpusonar.live import LiveChain  # noqa: E402
from gpusonar.sim.simulator import run_scenario  # noqa: E402

BASE = "popoto_12km_large_submerged"
RANGE_M, DEPTH_M, TS_POINT_DB, TS_HIGHLIGHT_DB = 12000.0, 50.0, 25.0, 18.01
BEARINGS = {"point": -10.0, "hull": 10.0}
GATE_BEARING_DEG = 1.5
BAND = (4.0e3, 6.0e3)
ALPHA = 0.1


def say(text: str) -> None:
    print(text, flush=True)


def waveform(kind: str, pulse_s: float) -> dict:
    w = {"f_start": BAND[0], "f_stop": BAND[1], "duration": float(pulse_s), "window_alpha": ALPHA}
    return dict(w, type="lfm") if kind == "lfm" else dict(w, type="sampled", source="hfm")


def scene_config(kind: str, pulse_s: float, speed: float, pings: int, ts_offset_db: float = 0.0) -> dict:
    cfg = load_scenario_config(BASE, expand=False)
    cfg["name"] = f"hfm_eval_{kind}_{pulse_s:g}s_{speed:g}"
    cfg["fidelity"] = "free_field"
    cfg["reverberation"] = []
    cfg.pop("propagation", None)
    cfg["platform"] = dict(cfg["platform"], velocity=[0.0, 0.0, 0.0])
    cfg["pings"] = dict(cfg.get("pings", {}), count=int(pings))
    over = copy.deepcopy(cfg.get("sonar_overrides", {}))
    over.setdefault("transmit", {})["waveform"] = waveform(kind, pulse_s)
    cfg["sonar_overrides"] = over
    p0 = np.asarray(cfg["platform"]["position"], dtype=float)
    scene = []
    for label, b in BEARINGS.items():
        los = np.array([np.sin(np.radians(b)), np.cos(np.radians(b)), 0.0])
        c0 = np.array([p0[0], p0[1], DEPTH_M]) + RANGE_M * los
        vel = (-float(speed) * los).tolist()
        if label == "point":
            pos, ts = [c0.tolist()], TS_POINT_DB
        else:
            pos, ts = [(c0 + s * los).tolist() for s in (-50.0, -25.0, 0.0, 25.0, 50.0)], TS_HIGHLIGHT_DB
        scene.append({"type": "points", "label": label, "positions": pos, "target_strength_db": ts + float(ts_offset_db), "velocity": vel})
    cfg["scene"] = scene
    return cfg


def truth(cfg: dict, n: int, c: float) -> dict:
    """Per label, per ping: the range of the nearest highlight where the pulse reaches it (one-way delay from the ping-time
    range, as tools/doppler_bank_eval.py `--truth-at reflection`), its bearing and radial speed (opening positive)."""
    p0 = np.asarray(cfg["platform"]["position"], dtype=float)
    dt = float(cfg["pings"].get("interval", 1.0))
    out = {}
    for item in cfg["scene"]:
        pos, v = np.asarray(item["positions"], dtype=float), np.asarray(item["velocity"], dtype=float)
        near = pos[int(np.argmin(np.linalg.norm(pos - p0, axis=1)))]
        rows = []
        for i in range(n):
            t = i * dt
            t += float(np.linalg.norm(near + v * t - p0)) / c
            d = near + v * t - p0
            u = d / np.linalg.norm(d)
            rows.append({"range": float(np.linalg.norm(d)), "bearing": float(np.degrees(np.arctan2(d[0], d[1]))), "range_rate": float(v @ u)})
        out[item["label"]] = rows
    return out


def run_chain(records, cfg: dict):
    reports, thresholds, seconds = [], [], []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with LiveChain(cfg) as chain:
            for r in records:
                t0 = time.perf_counter()
                reports.append(chain.process(r.copy()))
                seconds.append(time.perf_counter() - t0)
                thresholds.append(float(chain.detector["scales"][0]["threshold_db"]))
    return reports, thresholds, seconds


def per_target(reports, thresholds, truths: dict, gate: float) -> dict:
    out = {}
    for label, rows in truths.items():
        pings = []
        for rep, thr, tr in zip(reports, thresholds, rows):
            near = [d for d in rep["detections"] if abs(d["range_m"] - tr["range"]) < gate and abs(d["bearing_deg"] - tr["bearing"]) < GATE_BEARING_DEG]
            d = max(near, key=lambda q: q["snr_db"]) if near else None
            peak = float(max(d["arrivals"], key=lambda a: a[1])[0]) if d and d.get("arrivals") else (d["range_m"] if d else None)
            pings.append({"snr_db": d["snr_db"] if d else None, "threshold_db": thr, "margin_db": d["snr_db"] - thr if d else None,
                          "range_error_m": peak - tr["range"] if d else None, "true_range_rate_m_s": tr["range_rate"]})
        got = [p for p in pings if p["margin_db"] is not None]
        out[label] = {"pings": pings, "detected": len(got),
                      "margin_db": float(np.mean([p["margin_db"] for p in got])) if got else None,
                      "range_error_m": float(np.mean([p["range_error_m"] for p in got])) if got else None}
    return out


def same(a, b, rel: float = 1e-6) -> bool:
    """Two engines' per-target figures equal to the C++ engine's printed digits."""
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k], rel) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(same(x, y, rel) for x, y in zip(a, b))
    if isinstance(a, float) and isinstance(b, float):
        return abs(a - b) <= max(rel * max(abs(a), abs(b)), 1e-6)
    return a == b


def closed_forms(kind: str, pulse_s: float, speed: float, c: float) -> dict:
    r = -float(speed)                                      # closing
    if kind == "hfm":
        h = doppler.hfm_doppler(r, BAND[0], BAND[1], pulse_s, c, window_alpha=ALPHA)
        rect = doppler.hfm_doppler(r, BAND[0], BAND[1], pulse_s, c)
        return {"peak_loss_db": h["loss_db"], "peak_loss_untapered_db": rect["loss_db"], "range_error_m": h["range_error_m"],
                "source": "ours: gpusonar.doppler.hfm_doppler (tests/python/test_hfm_doppler.py)"}
    phi = np.pi * (BAND[1] - BAND[0]) * pulse_s * abs(r) / c
    return {"peak_loss_db": float(-20.0 * np.log10(core.live_sweep_mismatch(phi))), "coupling_m": float(0.5 * sum(BAND) * pulse_s / (BAND[1] - BAND[0]) * r),
            "source": "ours: the C++ library's sweep_mismatch (untapered; tests/python/test_doppler_bank.py); coupling f_c T r / B"}


def fmt(v, spec: str) -> str:
    return "-" if v is None else format(v, spec)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pulses", type=float, nargs="+", default=[1.0])
    ap.add_argument("--waveforms", nargs="+", default=["lfm", "hfm"], choices=["lfm", "hfm"])
    ap.add_argument("--speeds", type=float, nargs="+", default=[0.0, 2.0, 5.6, 8.0])
    ap.add_argument("--pings", type=int, default=3)
    ap.add_argument("--ts-offset-db", type=float, default=0.0)
    ap.add_argument("--engines", nargs="+", default=["python", "cpp"], choices=["python", "cpp"])
    ap.add_argument("--bank", action="store_true", help="the LFM cases also with the sweep bank, combine cfar, to the largest speed")
    ap.add_argument("--bank-engines", nargs="+", default=["python", "cpp"], choices=["python", "cpp"])
    ap.add_argument("--gate-range-m", type=float, default=None, help="default: 100 m + the HFM's bias at the largest speed")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    if 0.0 not in a.speeds:
        ap.error("--speeds must include 0 (the reference)")
    result = {"command": "tools/hfm_doppler_eval.py " + " ".join(argv if argv is not None else sys.argv[1:]), "base_scene": BASE,
              "what": "free field, no reverberation, still platform, Wenz noise; point (+25 dB) and bow-on hull (5 x 18.01 dB) at 12 km closing "
                      "along the line of sight; plain filter unless `bank`; margin = snr_db - the plan's single-sample threshold",
              "ts_offset_db": a.ts_offset_db, "pings": a.pings, "cases": []}
    t_all = time.time()

    def store():
        if a.json:
            Path(a.json).write_text(json.dumps(result, indent=1, default=float) + "\n")

    for T in a.pulses:
        gate = a.gate_range_m or 100.0 + doppler.hfm_range_factor(BAND[0], BAND[1], T) * max(a.speeds)
        for kind in a.waveforms:
            for v in a.speeds:
                cfg = scene_config(kind, T, v, a.pings, a.ts_offset_db)
                t0 = time.time()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    records = run_scenario(copy.deepcopy(cfg)).records
                sim_s = time.time() - t0
                c = float(records[0].environment.sound_speed)
                truths = truth(cfg, len(records), c)
                case = {"waveform": kind, "pulse_s": T, "speed_m_s": v, "closed_form": closed_forms(kind, T, v, c), "sound_speed": c,
                        "simulated_s": round(sim_s, 1), "gate_range_m": gate, "energy_db": None, "plain": {}, "bank": {}}
                from gpusonar.sonar import SonarModel
                case["energy_db"] = SonarModel.load(cfg["sonar"], cfg["sonar_overrides"]).source_energy_db
                variants = [("plain", None, a.engines)]
                if a.bank and kind == "lfm":
                    variants.append(("bank", {"mode": "sweep_bank", "combine": "cfar", "max_range_rate_m_s": float(max(a.speeds)), "report": True}, a.bank_engines))
                for name, dop, engines in variants:
                    for engine in engines:
                        ccfg = {"engine": engine, "backend": "host"}
                        if dop:
                            ccfg["doppler"] = dop
                        reports, thr, secs = run_chain(records, ccfg)
                        case[name][engine] = {"targets": per_target(reports, thr, truths, gate),
                                              "seconds_a_ping": float(np.median(secs[1:]) if len(secs) > 1 else secs[0]),
                                              "detections_per_ping": [len(r["detections"]) for r in reports]}
                    if len(engines) == 2:
                        case[name]["engines_equal"] = same(case[name]["python"]["targets"], case[name]["cpp"]["targets"])
                    e = "cpp" if "cpp" in engines else engines[0]
                    say(f"T {T:g} s {kind} {v:g} m/s {name}: " + "; ".join(
                        f"{lab} margin {x['margin_db'] if x['margin_db'] is None else round(x['margin_db'], 2)} dB, range error "
                        f"{x['range_error_m'] if x['range_error_m'] is None else round(x['range_error_m'], 1)} m, {x['detected']}/{len(records)}"
                        for lab, x in case[name][e]["targets"].items()) + f"; engines equal {case[name].get('engines_equal')}; "
                        f"closed form {case['closed_form']['peak_loss_db']:.2f} dB; sim {sim_s:.0f} s, chain {case[name][e]['seconds_a_ping']:.1f} s a ping")
                result["cases"].append(case)
                store()
    # Relative to zero speed, per (pulse, waveform, variant, target), the C++ engine's (equal to the Python one's where both ran).
    rel = []
    for case in result["cases"]:
        ref = next(x for x in result["cases"] if x["waveform"] == case["waveform"] and x["pulse_s"] == case["pulse_s"] and x["speed_m_s"] == 0.0)
        for name in ("plain", "bank"):
            for engine in ("cpp", "python"):
                if engine not in case[name] or engine not in ref.get(name, {}):
                    continue
                for lab, x in case[name][engine]["targets"].items():
                    x0 = ref[name][engine]["targets"][lab]
                    rel.append({"pulse_s": case["pulse_s"], "waveform": case["waveform"], "variant": name, "speed_m_s": case["speed_m_s"], "target": lab,
                                "engine": engine, "margin_db": x["margin_db"], "margin_at_zero_db": x0["margin_db"],
                                "relative_db": None if x["margin_db"] is None or x0["margin_db"] is None else x["margin_db"] - x0["margin_db"],
                                "range_error_m": x["range_error_m"], "detected": x["detected"], "closed_form_peak_loss_db": case["closed_form"]["peak_loss_db"]})
                break
    result["relative"] = rel
    result["seconds"] = round(time.time() - t_all, 1)
    store()
    say("\n| T | waveform | variant | speed | target | margin (dB) | vs zero speed (dB) | closed-form peak loss | range error (m) | detected |")
    for x in sorted(rel, key=lambda x: (x["relative_db"] if x["relative_db"] is not None else -99)):
        if x["speed_m_s"] == 0.0:
            continue
        say(f"| {x['pulse_s']:g} s | {x['waveform']} | {x['variant']} | {x['speed_m_s']:g} | {x['target']} | {fmt(x['margin_db'], '.2f')} | "
            f"{fmt(x['relative_db'], '+.2f')} | {x['closed_form_peak_loss_db']:.2f} | {fmt(x['range_error_m'], '.1f')} | {x['detected']} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
