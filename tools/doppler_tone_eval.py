#!/usr/bin/env python3
"""The tone's Doppler filter bank (D3 part B) end to end through the live chain, both engines: each target's detection,
its path (sweep, tone, both), range rate and range error, its track, the tone's own margin, the false detections on the
scene's background, and the cost a ping.

    tools/doppler_tone_eval.py SCENE [--pings 6] [--engines python cpp] [--background 4] [--max-rate 3.0]
                               [--alternate HZ] [--cache DIR] [--json FILE]

What is run (ours). The scene as shipped (`configs/scenarios/SCENE.yaml`, its own seed and pings; the eigenray scenes
need GPUSONAR_BELLHOP) is simulated once and its records go through `LiveChain` with the deployment defaults (bearings
-40 to 40 degrees every 0.5, backprojection, multiscale detector, host backend), engine by engine, without the Doppler
path and with `doppler: {tone: bank, report: true, max_range_rate_m_s: --max-rate}` plus the tone's segment from the
scene's train (gpusonar.doppler.tone_from_sonar; nothing for a CW record).

`--alternate HZ`: alternating pings instead. The scene (a sweep scene such as flank_patrol) is simulated twice with twice
its pings: as shipped, and with its transmit replaced by a 1 s Hann tone at HZ alone carrying the sweep's whole energy
(the analytic CW, its source level set for the sweep's energy); the records are taken in turn, tone first (even pings from the
tone run, odd ones from the sweep run: the scene's clock and targets are the same in both). Each ping then has the
whole energy of one waveform and the pair takes two ping intervals.

Per target and ping (its centre, as seen from the platform at that ping):
- the detection: the strongest within 60 m and 1.5 degrees; else (a tone-only detection, coarse) within c T / 4 and
  2 degrees; its `doppler_path`, `range_rate_m_s` against the true range rate, its range error;
- the tone path's own detection (the Python engine's tone stage, the function both engines run): its snr_db over its
  background and the margin over the threshold of a cell with a full background (10 log10 of the stage's factor);
- whether a track is within 100 m (200 m when the target is seen by the tone alone) at the last ping, and its class.

The background: the scene with its targets removed (new draws, seed + 1000), --background pings, counted at the default
budget and at 3 false detections a ping (both paths' budgets), tone path off and on, by path.

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
sys.path.insert(0, str(ROOT / "tools"))

from doppler_bank_eval import truth  # noqa: E402
from gpusonar import doppler  # noqa: E402
from gpusonar.config import load_scenario_config  # noqa: E402
from gpusonar.live import LiveChain  # noqa: E402
from gpusonar.sim.simulator import run_scenario  # noqa: E402
from gpusonar.sonar import SonarModel  # noqa: E402

GATE_RANGE_M, GATE_BEARING_DEG = 60.0, 1.5


def say(text: str) -> None:
    print(text, flush=True)


CACHE = None          # --cache DIR: simulated records kept as container files, read back on the next run of the same scene


def simulate(cfg: dict, tag: str = "scene"):
    path = None
    if CACHE is not None:
        key = f"{cfg.get('name', 'scene')}_{tag}_{int(cfg.get('pings', {}).get('count', 1))}_{int(cfg.get('seed', 0))}.h5"
        path = Path(CACHE) / key
        if path.exists():
            from gpusonar.sources.container import ContainerSource
            with ContainerSource(str(path)) as src:
                return list(src)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        records = run_scenario(copy.deepcopy(cfg)).records
    if path is not None:
        from gpusonar import core
        Path(CACHE).mkdir(parents=True, exist_ok=True)
        writer = core.RecordWriter(str(path))
        for r in records:
            writer.write(r)
        writer.close()
    return records


def tone_keys(cfg: dict) -> dict:
    sonar = SonarModel.load(cfg["sonar"], cfg.get("sonar_overrides"))
    try:
        return doppler.tone_from_sonar(sonar)
    except ValueError:
        return {"tone": "bank"}                  # a CW record: the record's own tone


def chain_config(engine: str, tone: dict | None, max_rate: float, budget: float | None = None) -> dict:
    cfg = {"engine": engine, "backend": "host"}
    if tone is not None:
        cfg["doppler"] = dict(tone, report=True, max_range_rate_m_s=max_rate)
        if budget is not None:
            cfg["doppler"]["tone_false_alarms_per_ping"] = budget
    if budget is not None:
        cfg["cfar"] = {"false_alarms_per_ping": budget}
    return cfg


def run_chain(records, cfg: dict):
    reports, seconds, stages = [], [], []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with LiveChain(cfg) as chain:
            for r in records:
                t = time.perf_counter()
                rep = chain.process(r.copy())
                seconds.append(time.perf_counter() - t)
                reports.append(rep)
                stages.append(getattr(chain, "_tone", None))
    return reports, seconds, stages


def pick(rep, tr, tone_gate: float):
    close = [d for d in rep["detections"] if abs(d["range_m"] - tr["range"]) < GATE_RANGE_M and abs(d["bearing_deg"] - tr["bearing"]) < GATE_BEARING_DEG]
    if close:
        return max(close, key=lambda q: q["snr_db"])
    coarse = [d for d in rep["detections"] if d.get("doppler_path") == "tone" and abs(d["range_m"] - tr["range"]) < tone_gate
              and abs(d["bearing_deg"] - tr["bearing"]) < 2.0]
    return max(coarse, key=lambda q: q["snr_db"]) if coarse else None


def tone_own(stage, tr, tone_gate: float):
    """The tone stage's own detection of this target (snr over its background, margin over a full-background threshold)."""
    if not stage or not stage["plan"]["on"]:
        return None
    got = [d for d in stage["detections"] if abs(d["range"] - tr["range"]) < tone_gate and abs(np.degrees(d["bearing"]) - tr["bearing"]) < 2.0]
    if not got:
        return {"snr_db": None, "threshold_db": float(10 * np.log10(stage["factor"]))}
    d = max(got, key=lambda q: q["snr_db"])
    thr = float(10 * np.log10(stage["factor"]))
    return {"snr_db": float(d["snr_db"]), "threshold_db": thr, "margin_db": float(d["snr_db"]) - thr, "range_rate": float(d["range_rate"]),
            "range_error_m": float(d["range"] - tr["range"])}


def per_target(reports, stages, truths: dict, tone_gate: float) -> dict:
    out = {}
    for label, rows in truths.items():
        pings = []
        for rep, st, tr in zip(reports, stages, rows):
            d = pick(rep, tr, tone_gate)
            pings.append({"snr_db": d["snr_db"] if d else None, "path": d.get("doppler_path") if d else None,
                          "range_rate_m_s": d.get("range_rate_m_s") if d else None, "true_range_rate_m_s": tr["range_rate"],
                          "range_error_m": d["range_m"] - tr["range"] if d else None, "tone": tone_own(st, tr, tone_gate)})
        last, tr = reports[-1], rows[len(reports) - 1]
        alone = any(p["path"] == "tone" for p in pings)
        tracks = [t for t in last["tracks"] if np.hypot(t["x"] - tr["x"], t["y"] - tr["y"]) < (200.0 if alone else 100.0)]
        out[label] = {"pings": pings, "detected": sum(p["snr_db"] is not None for p in pings),
                      "paths": sorted({p["path"] for p in pings if p["path"]}), "tracked": bool(tracks),
                      "kind": tracks[0]["kind"] if tracks else None,
                      "track_error_m": float(min(np.hypot(t["x"] - tr["x"], t["y"] - tr["y"]) for t in tracks)) if tracks else None}
    return out


def alternate_records(cfg: dict, hz: float, pings: int):
    """Tone-only and sweep-only pings in turn (tone first): the scene as shipped and with a 1 s Hann tone at `hz` carrying
    the sweep's whole energy, each simulated with 2 x pings and taken in turn."""
    sonar = SonarModel.load(cfg["sonar"], cfg.get("sonar_overrides"))
    energy = sonar.source_energy_db                # the sweep's energy at 1 m, dB re 1 uPa^2 s (sonar.py)
    sweep_cfg = copy.deepcopy(cfg)
    sweep_cfg["pings"] = dict(cfg.get("pings", {}), count=2 * pings)
    tone_cfg = copy.deepcopy(sweep_cfg)
    # The analytic CW (fast to synthesise; the record a CW, the chain's tone-only ping): a Hann taper carries 0.375 of a
    # constant envelope's energy and `source_level_db_rms` scales the amplitude before the taper (DOPPLER.md D2).
    wave = {"type": "cw", "f_start": hz, "f_stop": hz, "duration": 1.0, "window_alpha": 1.0,
            "source_level_db_rms": float(energy - 10.0 * np.log10(0.375 * 1.0))}
    tone_cfg.setdefault("sonar_overrides", {}).setdefault("transmit", {})["waveform"] = wave
    tones, sweeps = simulate(tone_cfg, f"tone{hz:.0f}"), simulate(sweep_cfg, "sweep")
    return [tones[i] if i % 2 == 0 else sweeps[i] for i in range(2 * pings)], {"tone": "bank"}, energy, sweep_cfg


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene")
    ap.add_argument("--pings", type=int)
    ap.add_argument("--engines", nargs="+", default=["python", "cpp"], choices=["python", "cpp"])
    ap.add_argument("--background", type=int, default=4)
    ap.add_argument("--max-rate", type=float, default=3.0)
    ap.add_argument("--alternate", type=float, help="Hz: alternate tone-only (at this frequency) and sweep-only pings of SCENE")
    ap.add_argument("--json")
    ap.add_argument("--cache", help="a directory: keep the simulated records there and read them back on the next run")
    a = ap.parse_args(argv)
    global CACHE
    CACHE = a.cache
    cfg = load_scenario_config(a.scene, expand=False)
    t0 = time.time()
    energy = None
    if a.alternate:
        pairs = int(a.pings or cfg.get("pings", {}).get("count", 6)) // 2
        records, tone, energy, eval_cfg = alternate_records(cfg, a.alternate, pairs)
    else:
        if a.pings:
            cfg["pings"] = dict(cfg.get("pings", {}), count=int(a.pings))
        records, tone, eval_cfg = simulate(cfg), tone_keys(cfg), cfg
    say(f"{a.scene}: {len(records)} pings simulated in {time.time() - t0:.0f} s, {records[0].samples.n_channels} channels; tone {tone}")
    truths = truth(eval_cfg, len(records))
    c = float(records[0].environment.sound_speed)
    tone_gate = 0.25 * c * float(tone.get("tone_duration_s") or records[0].transmit.waveform.duration)
    result = {"scene": a.scene, "alternate_hz": a.alternate, "tone_energy_db": energy, "pings": len(records),
              "channels": int(records[0].samples.n_channels), "tone": tone, "max_range_rate_m_s": a.max_rate, "engines": {}, "background": {}}
    for engine in a.engines:
        res = {}
        for name, t in (("off", None), ("on", tone)):
            if name == "off" and a.alternate:
                # Today's chain on a stream of alternating pings is not defined for the tone pings (the C++ chain refuses a CW
                # window of fewer than three cells); off is the sweep pings alone.
                use = records[1::2]
                reports, seconds, stages = run_chain(use, chain_config(engine, t, a.max_rate))
                targets = per_target(reports, stages, {k: v[1::2] for k, v in truths.items()}, tone_gate)
            else:
                reports, seconds, stages = run_chain(records, chain_config(engine, t, a.max_rate))
                targets = per_target(reports, stages, truths, tone_gate)
            res[name] = {"targets": targets, "seconds": seconds,
                         "seconds_median_after_first": float(np.median(seconds[1:])) if len(seconds) > 1 else seconds[0],
                         "detections_per_ping": [len(r["detections"]) for r in reports], "reports": reports if engine == "cpp" else None}
            if t is not None and stages and stages[-1] is not None:
                res[name]["tone_plan"] = json.loads(stages[-1]["plan"]["json"])
                res[name]["tone_factor_db"] = float(10 * np.log10(stages[-1]["factor"])) if stages[-1]["plan"]["on"] else None
            say(f"{a.scene} {engine} tone {name:3s}: {res[name]['seconds_median_after_first']:.2f} s a ping, detections a ping "
                f"{res[name]['detections_per_ping']}")
            for label, s in targets.items():
                rr = [p["range_rate_m_s"] for p in s["pings"]]
                tm = [None if not p["tone"] or p["tone"].get("margin_db") is None else round(p["tone"]["margin_db"], 1) for p in s["pings"]]
                say(f"  {label:14s} detected {s['detected']} of {len(s['pings'])}, paths {s['paths']}, snr "
                    f"{[None if p['snr_db'] is None else round(p['snr_db'], 1) for p in s['pings']]}, rate {[None if v is None else round(v, 3) for v in rr]} "
                    f"(true {round(s['pings'][0]['true_range_rate_m_s'], 3)}), tone margin {tm}; tracked {s['tracked']} ({s['kind']})")
        result["engines"][engine] = res
    if len(a.engines) == 2:
        # The two engines' report lines, detection by detection (python's are kept only to compare here).
        same = True
        py_reports = run_chain(records, chain_config("python", tone, a.max_rate))[0]
        for w, g in zip(py_reports, result["engines"]["cpp"]["on"]["reports"]):
            if len(w["detections"]) != len(g["detections"]):
                same = False
                break
            for x, y in zip(w["detections"], g["detections"]):
                if (abs(x["range_m"] - y["range_m"]) > 0.05 or abs(x["bearing_deg"] - y["bearing_deg"]) > 0.05 or abs(x["snr_db"] - y["snr_db"]) > 0.05
                        or x.get("doppler_path") != y.get("doppler_path")):
                    same = False
        result["engines_equal_on"] = same
        say(f"{a.scene}: the engines' report lines with the tone path {'agree' if same else 'DIFFER'} (0.05 m, 0.05 deg, 0.05 dB, path)")
    for engine in a.engines:
        for name in ("off", "on"):
            result["engines"][engine][name].pop("reports", None)
    if a.background > 0 and not a.alternate:
        bcfg = copy.deepcopy(cfg)
        bcfg["scene"] = []
        bcfg["pings"] = dict(bcfg.get("pings", {}), count=int(a.background))
        bcfg["seed"] = int(cfg.get("seed", 0)) + 1000
        back = simulate(bcfg, "background")
        for budget in (None, 3.0):
            row = {}
            for name, t in (("off", None), ("on", tone)):
                reports, _, _ = run_chain(back, chain_config("cpp", t, a.max_rate, budget))
                row[name] = sum(len(r["detections"]) for r in reports)
                if t is not None:
                    row["on_tone_only"] = sum(1 for r in reports for d in r["detections"] if d.get("doppler_path") == "tone")
            result["background"]["default" if budget is None else str(budget)] = dict(row, pings=len(back))
            say(f"{a.scene} background, {len(back)} pings, budget {'default' if budget is None else budget}: false detections off {row['off']}, "
                f"on {row['on']} (tone alone {row['on_tone_only']})")
    if a.json:
        Path(a.json).write_text(json.dumps(result, indent=1, default=float) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
