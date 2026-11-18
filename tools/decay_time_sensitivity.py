#!/usr/bin/env python3
"""How much the decay-time rule of the time spreading matters for detection with the long-range array.

    tools/decay_time_sensitivity.py [--cases popoto_12km,...] [--seeds 2] [--out docs/figures/decay_time_sensitivity.md]
                                    [--json docs/figures/decay_time_sensitivity.json] [--pings N]

The long-range scenarios ship without a `time_spread` section (bounce paths are fully coherent).
Here each is run as shipped and with

    time_spread: {surface: {wind_speed: <the scenario's own>}, decay_time: facet | rule_of_thumb,
                  multi_bounce: convolve | sum, seed: s}

and the pings go through the C++ chain exactly as in tools/popoto_12km_study.py (its `measure`):
detection SNR of the ship over the local background, pings detected, the ping at which a track
on it is confirmed. The scattered taps are random, so every option is run with `--seeds` seeds
and a difference between rules counts only against the spread between seeds.

Cases: popoto_12km (Bellhop, needs GPUSONAR_BELLHOP), popoto_12km with image-source paths, and
the three shallow popoto_env_* environments at their own ranges (3, 4 and 8 km). One at a time,
restartable (the JSON is rewritten after every run).
"""
import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

from gpusonar.config import load_scenario_config  # noqa: E402

CASES = ["popoto_12km:image", "popoto_12km", "popoto_env_gravel_30m", "popoto_env_summer_mud_60m", "popoto_env_winter_sand_200m"]
OPTIONS = {"facet": {"decay_time": "facet", "multi_bounce": "convolve"},
           "rule_of_thumb": {"decay_time": "rule_of_thumb", "multi_bounce": "convolve"},
           "facet, sum": {"decay_time": "facet", "multi_bounce": "sum"}}


def scenario(case: str, pings: int | None) -> dict:
    import popoto_12km_study as study
    name, _, how = case.partition(":")
    cfg = load_scenario_config(name)
    if how == "image":
        cfg = study.image_source(cfg)
    if pings:
        cfg["pings"] = dict(cfg["pings"], count=int(pings))
    return cfg


def wind_of(cfg: dict) -> float:
    for layer in cfg.get("reverberation", []):
        if "wind_speed" in layer:
            return float(layer["wind_speed"])
    return float((cfg.get("receiver", {}).get("ambient_noise") or {}).get("wind_speed", 5.0))


def summary(m: dict) -> dict:
    snr = [r["snr_db"] for r in m["per_ping"] if r["detected"]]
    return {"pings": m["pings"], "pings_detected": m["pings_detected"], "median_snr_db": m["median_snr_db"], "snr_db": snr,
            "track_confirmed_at_ping": m["track_confirmed_at_ping"], "simulate_seconds": m["simulate_seconds"]}


def run(cases: list, seeds: int, pings: int | None, store: Path, log=print) -> dict:
    import popoto_12km_study as study
    res = json.loads(store.read_text()) if store.exists() else {}
    for case in cases:
        cfg = scenario(case, pings)
        wind = wind_of(cfg)
        todo = [("as shipped", None, None)] + [(name, opt, s) for name, opt in OPTIONS.items() for s in range(seeds)]
        for name, opt, seed in todo:
            key = name if seed is None else f"{name} / seed {seed}"
            if key in res.get(case, {}):
                continue
            c = copy.deepcopy(cfg)
            if opt is not None:
                c["time_spread"] = dict({"surface": {"wind_speed": wind}, "seed": 1000 + seed}, **opt)
            m = summary(study.measure(c, 1))
            m["wind_speed"] = wind
            res.setdefault(case, {})[key] = m
            store.write_text(json.dumps(res, indent=1))
            log(f"{case:32s} {key:28s} detected {m['pings_detected']}/{m['pings']} median SNR {m['median_snr_db']} dB, track at ping {m['track_confirmed_at_ping']} "
                f"({m['simulate_seconds']:.0f} s)")
    return res


def markdown(res: dict) -> str:
    lines = ["# Detection with the long-range array: how much the decay-time rule matters", "",
             "Made by `tools/decay_time_sensitivity.py`: each scenario as shipped (no time spreading) and with a wind-roughened surface under each rule. One row of 32 hydrophones, "
             "C++ chain, the scenario's own pings. SNR: the ship's detection over the local background, median over the pings it was detected on; "
             "in brackets each seed of the scattered taps.", "",
             "| Scenario | wind, m/s | as shipped | facet | rule of thumb | facet, multi_bounce sum | rule of thumb minus facet, dB | spread between seeds, dB |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for case, rows in res.items():
        def cell(name):
            keys = [k for k in rows if k == name or k.startswith(name + " / seed")]
            if not keys:
                return "-", []
            snr = [rows[k]["median_snr_db"] for k in keys]
            good = [s for s in snr if s is not None]
            det = "; ".join(f"{rows[k]['pings_detected']}/{rows[k]['pings']}, track {rows[k]['track_confirmed_at_ping'] or 'none'}" for k in keys)
            txt = (f"{np.mean(good):.1f} dB" if good else "not detected") + (" (" + ", ".join("-" if s is None else f"{s:.1f}" for s in snr) + ")" if len(keys) > 1 else "") + f"; {det}"
            return txt, good
        shipped, _ = cell("as shipped")
        facet, f = cell("facet")
        rule, r = cell("rule_of_thumb")
        fsum, _ = cell("facet, sum")
        diff = f"{np.mean(r) - np.mean(f):+.1f}" if f and r else "-"
        spread = f"{max(np.ptp(f) if len(f) > 1 else 0.0, np.ptp(r) if len(r) > 1 else 0.0):.1f}" if f and r else "-"
        wind = next(iter(rows.values())).get("wind_speed", float("nan"))
        lines.append(f"| {case} | {wind:.0f} | {shipped} | {facet} | {rule} | {fsum} | {diff} | {spread} |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", default=",".join(CASES))
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--pings", type=int, default=None)
    ap.add_argument("--json", default=str(ROOT / "docs" / "figures" / "decay_time_sensitivity.json"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args(argv)
    store = Path(args.json)
    res = json.loads(store.read_text()) if args.report_only else run(args.cases.split(","), args.seeds, args.pings, store)
    text = markdown(res)
    if args.out:
        Path(args.out).write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
