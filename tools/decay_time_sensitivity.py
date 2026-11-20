#!/usr/bin/env python3
"""How much the decay-time rule of the time spreading matters for detection with the long-range array.

    tools/decay_time_sensitivity.py [--cases popoto_12km,...] [--seeds 2] [--out docs/figures/decay_time_sensitivity.md]
                                    [--json docs/figures/decay_time_sensitivity.json] [--pings N] [--fresh]

The long-range scenarios ship without a `time_spread` section (bounce paths are fully coherent).
Here each is run as shipped and with

    time_spread: {surface: {wind_speed: <the scenario's own>}, decay_time: facet | rule_of_thumb,
                  multi_bounce: convolve | sum, seed: s}

and the pings go through the C++ chain exactly as in tools/popoto_12km_study.py (its `measure`):
detection SNR of the ship over the local background, pings detected, the ping at which a track
on it is confirmed. The scattered taps are random, so every option is run with `--seeds` seeds
and a difference between rules counts only against the spread between seeds.

Cases: popoto_12km (Bellhop, needs GPUSONAR_BELLHOP), popoto_12km with image-source paths, and
the three shallow popoto_env_* environments at their own ranges (3, 4 and 8 km). One at a time.

RESUMES FROM THE STORED JSON: a run already in `--json` (by default the committed
docs/figures/decay_time_sensitivity.json) is NOT recomputed; its stored numbers are reported as
they are, whatever the code now does. The JSON is rewritten after every run, so an interrupted
study continues where it stopped. A stored run is found by its case, which names the settings it
ran with (pings, Bellhop arrivals kept with --max-arrivals, taps), and by its option and seed: a
run with other --pings, --taps or --max-arrivals is a new case and is computed. `--fresh` ignores
everything stored and recomputes every run (the JSON then holds this run's results only). To
regenerate a committed file, use `--fresh` or an empty `--json` file.
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


def case_label(case: str, cfg: dict, max_arrivals: int | None = None, taps: int | None = None) -> str:
    """The case's key in the JSON: the scenario, its paths and the settings it runs with, e.g.
    `popoto_12km, Bellhop (3 pings, 12 strongest arrivals in every variant, 3 taps)`."""
    import inspect
    from gpusonar.sim.timespread import TimeSpreadProvider
    name, _, how = case.partition(":")
    fidelity = cfg.get("fidelity")
    paths = "image-source paths" if how == "image" else "Bellhop" if fidelity == "eigenray" else f"{fidelity} paths"
    parts = [f"{int(cfg['pings']['count'])} pings"]
    if max_arrivals and fidelity == "eigenray":
        parts.append(f"{int(max_arrivals)} strongest arrivals in every variant")
    parts.append(f"{int(taps) if taps else inspect.signature(TimeSpreadProvider.__init__).parameters['taps'].default} taps")
    return f"{name}, {paths} ({', '.join(parts)})"


def wind_of(cfg: dict) -> float:
    for layer in cfg.get("reverberation", []):
        if "wind_speed" in layer:
            return float(layer["wind_speed"])
    return float((cfg.get("receiver", {}).get("ambient_noise") or {}).get("wind_speed", 5.0))


def summary(m: dict) -> dict:
    snr = [r["snr_db"] for r in m["per_ping"] if r["detected"]]
    return {"pings": m["pings"], "pings_detected": m["pings_detected"], "median_snr_db": m["median_snr_db"], "snr_db": snr,
            "track_confirmed_at_ping": m["track_confirmed_at_ping"], "simulate_seconds": m["simulate_seconds"]}


def run(cases: list, seeds: int, pings: int | None, store: Path, log=print, options=None, max_arrivals: int | None = None, taps: int | None = None,
        fresh: bool = False, measure=None) -> dict:
    """Every run not yet stored in `store` (all of them with `fresh`), each written to `store` as it ends.
    measure: the study's `measure(config, seed)` (popoto_12km_study's by default)."""
    if measure is None:
        import popoto_12km_study as study
        measure = study.measure
    res = {} if fresh or not store.exists() else json.loads(store.read_text())
    for case in cases:
        cfg = scenario(case, pings)
        label = case_label(case, cfg, max_arrivals, taps)
        # Bellhop cases only. Since D-137 the scenarios no longer state `max_arrivals` (the energy rule keeps arrivals),
        # so the limit cannot wait for the key to be there: it is set whenever the case runs on Bellhop's eigenrays.
        if max_arrivals and cfg.get("fidelity") == "eigenray":
            # Fewer Bellhop arrivals (the strongest), in all variants alike: with time spreading every ray becomes
            # 1 + taps rays and the two-way cost grows with the square.
            cfg["propagation"]["max_arrivals"] = int(max_arrivals)
        wind = wind_of(cfg)
        # every seed for the two rules; the multi-bounce rule with the first seed only (it is a second question)
        todo = [("as shipped", None, None)] + [(name, opt, s) for name, opt in OPTIONS.items() if options is None or name in options for s in range(seeds if "sum" not in name else 1)]
        for name, opt, seed in todo:
            key = name if seed is None else f"{name} / seed {seed}"
            if key in res.get(label, {}):
                continue
            c = copy.deepcopy(cfg)
            if opt is not None:
                c["time_spread"] = dict({"surface": {"wind_speed": wind}, "seed": 1000 + seed}, **opt)
                if taps:
                    c["time_spread"]["taps"] = int(taps)
            m = summary(measure(c, 1))
            m["wind_speed"] = wind
            res.setdefault(label, {})[key] = m
            store.write_text(json.dumps(res, indent=1))
            log(f"{label:32s} {key:28s} detected {m['pings_detected']}/{m['pings']} median SNR {m['median_snr_db']} dB, track at ping {m['track_confirmed_at_ping']} "
                f"({m['simulate_seconds']:.0f} s)")
    return res


def echo_levels(pings: int = 24, log=print) -> dict:
    """Without noise, reverberation or a detector: the echo of one point target after pulse
    compression on one hydrophone, ping after ping, with the surface flat and rough under each
    rule (image-source paths, two bounces). Two geometries: `flank_sst_shallow`'s (flank array,
    1 km, 100 m of water, wind 6 m/s) and `popoto_12km`'s (long-range array, 12 km, 100 m, wind
    5 m/s). Reported re the flat surface: the strongest sample of the whole echo (what a
    detector on one beam sees), its ping-to-ping standard deviation, and the echo's energy."""
    from gpusonar import recon
    from gpusonar.sim import ImageSourceProvider, Scene, make_pose, simulate_ping
    from gpusonar.sim.timespread import wrap
    from gpusonar.sonar import SonarModel
    out = {}
    geometries = {"flank_sst_shallow geometry (1 km)": ("flank_32", {"receive": {"window": {"range": [900.0, 1150.0]}}}, [0.0, 1000.0, 50.0], 30.0, 6.0, 15.0),
                  "popoto_12km geometry (12 km)": ("popoto_long_range", {"receive": {"kind": "baseband", "sample_rate": 8.0e3, "window": {"range": [11800.0, 12300.0]}}},
                                                   [0.0, 12000.0, 4.0], 10.0, 5.0, 19.9)}
    for label, (sonar_name, overrides, target, own_depth, wind, interval) in geometries.items():
        sonar = SonarModel.load(sonar_name, overrides)
        inner = ImageSourceProvider(1500.0, surface_z=0.0, bottom_z=100.0, max_order=2, bottom=dict(speed=1650.0, density=1.9, attenuation=0.8))
        scene = Scene().add_points([target], target_strength_db=10.0)
        pose = make_pose(position=(0.0, 0.0, own_depth))
        cases = {"flat": inner}
        for rule in ("facet", "rule_of_thumb"):
            cases[rule] = wrap(inner, {"surface": {"wind_speed": wind}, "decay_time": rule, "seed": 11}, frequency=sonar.center_frequency)
        peaks, energy = {}, {}
        for name, provider in cases.items():
            pk, en = [], []
            for i in range(pings if name != "flat" else 2):
                rec = simulate_ping(sonar, pose, scene, provider=provider, environment={"sound_speed": 1500.0}, adc=False, noise_rms=0.0,
                                    scenario_time=interval * i, sequence=i)
                recon.prepare([rec])
                e = np.abs(rec.samples.array[rec.samples.array.shape[0] // 2]) ** 2
                pk.append(e.max())
                en.append(e.sum())
            peaks[name], energy[name] = np.array(pk), np.array(en)
        row = {"wind_speed": wind, "pings": pings}
        for rule in ("facet", "rule_of_thumb"):
            db = 10.0 * np.log10(peaks[rule] / peaks["flat"].mean())
            row[rule] = {"peak_db": float(10.0 * np.log10(peaks[rule].mean() / peaks["flat"].mean())), "peak_median_db": float(np.median(db)), "peak_std_db": float(db.std()),
                         "energy_db": float(10.0 * np.log10(energy[rule].mean() / energy["flat"].mean()))}
        row["difference_db"] = row["rule_of_thumb"]["peak_db"] - row["facet"]["peak_db"]
        row["standard_error_db"] = float(np.hypot(row["facet"]["peak_std_db"], row["rule_of_thumb"]["peak_std_db"]) / np.sqrt(pings))
        out[label] = row
        log(label, json.dumps(row))
    return out


def markdown(res: dict) -> str:
    lines = ["# Detection with the long-range array: how much the decay-time rule matters", "",
             "Made by `tools/decay_time_sensitivity.py`: each scenario without time spreading and with a wind-roughened surface under each rule. One row of 32 hydrophones, "
             "C++ chain; pings, Bellhop arrivals kept and taps as named in the first column (the same in every variant of a row). SNR: the ship's detection over the local background, median over the pings it was detected on; "
             "in brackets each seed of the scattered taps.", "",
             "| Scenario | wind, m/s | no time spreading | facet | rule of thumb | facet, multi_bounce sum | rule of thumb minus facet, dB | spread between seeds, dB |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for case, rows in res.items():
        if case.startswith("_"):
            continue

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
    if "_echo_levels" in res:
        lines += ["", "Without noise, reverberation or a detector: one point target, image-source paths with two bounces, one hydrophone after pulse compression. "
                  "The strongest sample of the echo, re a flat surface:", "",
                  "| Geometry | wind, m/s | pings | facet: mean (median), ping-to-ping sd, dB | rule of thumb | rule of thumb minus facet, dB | its standard error, dB | echo energy, facet / rule of thumb, dB |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        for label, r in res["_echo_levels"].items():
            f, t = r["facet"], r["rule_of_thumb"]
            lines.append(f"| {label} | {r['wind_speed']:.0f} | {r['pings']} | {f['peak_db']:+.2f} ({f['peak_median_db']:+.2f}), {f['peak_std_db']:.2f} | {t['peak_db']:+.2f} ({t['peak_median_db']:+.2f}), {t['peak_std_db']:.2f} | "
                         f"{r['difference_db']:+.2f} | {r['standard_error_db']:.2f} | {f['energy_db']:+.2f} / {t['energy_db']:+.2f} |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", default=",".join(CASES))
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--options", default=None, help="semicolon-separated subset of: " + "; ".join(OPTIONS))
    ap.add_argument("--pings", type=int, default=None)
    ap.add_argument("--max-arrivals", type=int, default=None, help="Bellhop arrivals kept per path (all variants); default: the scenario's")
    ap.add_argument("--taps", type=int, default=None, help="scattered taps per bounce ray (default: the model's 6)")
    ap.add_argument("--json", default=str(ROOT / "docs" / "figures" / "decay_time_sensitivity.json"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--report-only", action="store_true", help="only the report of the stored JSON; nothing is run")
    ap.add_argument("--fresh", action="store_true", help="ignore the stored JSON and recompute every run (also the echo levels); the JSON then holds this run's results only")
    ap.add_argument("--echo-pings", type=int, default=0, help="also the noise-free echo levels of `echo_levels`, over this many pings")
    args = ap.parse_args(argv)
    store = Path(args.json)
    res = json.loads(store.read_text()) if args.report_only else run([c for c in args.cases.split(",") if c], args.seeds, args.pings, store, options=args.options.split(";") if args.options else None,
                                                                         max_arrivals=args.max_arrivals, taps=args.taps, fresh=args.fresh)
    if args.echo_pings and (args.fresh or "_echo_levels" not in res):
        res["_echo_levels"] = echo_levels(args.echo_pings)
        store.write_text(json.dumps(res, indent=1))
    text = markdown(res)
    if args.out:
        Path(args.out).write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
