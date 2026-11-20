#!/usr/bin/env python3
"""Where the live chain's time per ping goes: the benchmark's live-chain case with the additions of the deployment
files switched off one at a time.

The pings and chains are those of `tools/gpu_benchmark_all.py` (group `live`): the stacked-row free-field scenario
with `popoto_long_range[_2row|_3row]`, the chain of `configs/deploy/popoto_vehicle.yaml` (one row) or
`popoto_vehicle_elevation.yaml` (two and three rows), its `sync` section left out, C++ engine, map bearings,
backprojection. Each variant runs `--repeats` times in turn (all variants, then all again), so the spread between
repeats of one variant shows the noise. Time per ping: the chain's own `processing_seconds`, median of the pings
after the first.

    tools/live_chain_ablation.py --backend cuda              # what issue sonar-y7a needs: the device
    tools/live_chain_ablation.py --backend host --rows 1     # the host; noisy against a few milliseconds

Written for issue sonar-y7a (the C++ chain on the device 6 to 9 ms a ping slower than at commit 15f1121).
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import statistics
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from gpusonar.config import load_yaml  # noqa: E402
from gpusonar.live import LiveChain  # noqa: E402

SONARS = {1: "popoto_long_range", 2: "popoto_long_range_2row", 3: "popoto_long_range_3row"}

# The chain sections of the two deployment files at commit 15f1121 (the benchmark's first run), `sync` left out: every
# size typed, no coverage, the elevation search of that time. Run on today's code they show what the files' changes cost.
OLD_CHAIN = {"bearings_deg": [-60.0, 60.0, 0.5], "look": "starboard", "range_cell_m": 2.0,
             "cfar": {"pfa": 1e-05, "guard_m": 100.0, "train_m": 300.0}, "min_snr_db": 9.0,
             "tracker": {"range_sigma": 15.0, "bearing_sigma_deg": 1.0, "confirm_hits": 3, "max_misses": 2},
             "classify": {"min_speed": 1.0, "multipath": {"max_excess_range": 800.0, "bearing_deg": 3.0, "velocity": 2.5}}}
OLD_ELEVATION = {"sector_deg": [-30.0, 30.0, 0.5], "min_quality": 0.6, "ambiguity_db": 1.5, "correct_bearing": True, "depth": True,
                 "fold": True, "classify": True, "water_depth": None, "roll_sigma_deg": 0.2}


def _bench():
    spec = importlib.util.spec_from_file_location("gpu_benchmark_all", ROOT / "tools" / "gpu_benchmark_all.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _drop(*path):
    def f(c):
        d = c
        for k in path[:-1]:
            d = d.get(k) or {}
        d.pop(path[-1], None)
        return c
    return f


def _set(path, value):
    def f(c):
        d = c
        for k in path[:-1]:
            d = d.setdefault(k, {})
        d[path[-1]] = value
        return c
    return f


def variants(rows: int) -> dict:
    v = {"as deployed": lambda c: c, "no coverage": _drop("coverage")}
    if rows > 1:
        v["no sector_from_rays"] = _drop("elevation", "sector_from_rays")
        v["elevation step 0.5 deg"] = _set(("elevation", "sector_deg"), [-30.0, 30.0, 0.5])
        v["attitude at transmit"] = _set(("elevation", "attitude_time"), "transmit")
        v["no gradient"] = _set(("elevation", "gradient_per_s"), None)

        def all_off(c):
            for f in (_drop("coverage"), _drop("elevation", "sector_from_rays"), _set(("elevation", "sector_deg"), [-30.0, 30.0, 0.5]),
                      _set(("elevation", "attitude_time"), "transmit"), _set(("elevation", "gradient_per_s"), None)):
                c = f(c)
            return c
        v["all of these off"] = all_off
    else:
        v["all of these off"] = _drop("coverage")
    v["cell statistic mean"] = _set(("cell_statistic",), "mean")

    def old(c, mean=False):
        o = copy.deepcopy(OLD_CHAIN)
        if rows > 1:
            o["elevation"] = copy.deepcopy(OLD_ELEVATION)
        if mean:
            o["cell_statistic"] = "mean"
        return o
    v["15f1121 chain section"] = old
    v["15f1121 chain section, mean"] = lambda c: old(c, True)
    return v


def time_variant(base: dict, change, records, backend: str) -> tuple[float, int]:
    cfg = change(copy.deepcopy(base))
    cfg.update(engine="cpp", doa="map", beamformer="backprojection", backend=backend)
    per, report = [], None
    with LiveChain(cfg) as live:
        for r in records:
            t0 = time.perf_counter()
            report = live.process(r.copy())
            per.append(report.get("processing_seconds") or (time.perf_counter() - t0))
    return statistics.median(per[1:]) if len(per) > 1 else per[0], len(report["detections"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", default="cuda", choices=("cuda", "host"))
    ap.add_argument("--rows", default="1,2,3")
    ap.add_argument("--pings", type=int, default=8)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)
    warnings.filterwarnings("ignore")
    bench = _bench()
    out = {"backend": args.backend, "pings": args.pings, "repeats": args.repeats, "rows": {}}
    for rows in (int(x) for x in args.rows.split(",")):
        deploy = ROOT / "configs" / "deploy" / ("popoto_vehicle.yaml" if rows == 1 else "popoto_vehicle_elevation.yaml")
        base = load_yaml(deploy).get("chain") or {}
        base.pop("sync", None)
        records = bench.live_pings(SONARS[rows], args.pings, None)
        vs = variants(rows)
        times = {name: [] for name in vs}
        dets = {}
        for _ in range(args.repeats):
            for name, change in vs.items():
                t, n = time_variant(base, change, records, args.backend)
                times[name].append(t)
                dets[name] = n
        out["rows"][str(rows)] = {name: {"seconds": t, "detections_last_ping": dets[name]} for name, t in times.items()}
        print(f"\n{rows} row(s), {args.backend}, {args.pings} pings, {args.repeats} repeats: chain time per ping, ms (each repeat; median)")
        ref = statistics.median(times["as deployed"])
        for name, t in times.items():
            m = statistics.median(t)
            print(f"  {name:28s} {' '.join(f'{1e3 * x:7.2f}' for x in t)}   median {1e3 * m:7.2f}   vs deployed {1e3 * (m - ref):+7.2f}   "
                  f"detections {dets[name]}", flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
