#!/usr/bin/env python3
"""Motion estimation studies on the simulated scenario of tests/python/test_motion.py
(`collect`: sas_auv at 100 kHz baseband, 40 pings, 0.24 m advance, 10 mm sway and 4 mm
heave RMS, DVL altitude noise 1 mm).

  timing   delay estimation: NumPy reference against the C++ port (host, and CUDA when
           available). Each backend runs three times in a row; the last run is reported
           (the ASASIN Sec. III-C protocol, as tools/benchmark.py).
  sweep    lambda1 (smoothness, Eq. 6) and lambda2 (DVL, Eq. 7) of Eq. 8 over a grid, on
           several seeds; reports sway and heave RMS error against the truth. It does not
           change MotionOptions' defaults; it reports what the grid suggests.

Usage: tools/motion_study.py timing [--pings 40] [--out timing.json]
       tools/motion_study.py sweep [--seeds 1 2 3] [--out sweep.json]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tests" / "python"))

import gpusonar as gs  # noqa: E402
from gpusonar import motion  # noqa: E402
from test_motion import collect  # noqa: E402  (the scenario the acceptance tests use)

core = gs.core
GRID = [0.0, 0.01, 0.1, 1.0, 10.0, 100.0]


def timing(args) -> dict:
    col, _ = collect(n_pings=args.pings, with_target=False, seed=1)
    backends = ["numpy", "host"] + (["cuda"] if core.cuda_available() else [])
    out = {"pings": args.pings, "cuda_device": core.cuda_device_name(0) if core.cuda_available() else None,
           "seconds": {}, "measurements": None}
    for name in backends:
        opts = motion.MotionOptions(delay_backend=name)
        runs = []
        for _ in range(3):
            t0 = time.perf_counter()
            found, rejected = motion.measure_delays(col.records, opts)
            runs.append(time.perf_counter() - t0)
        out["seconds"][name] = runs[-1]
        out["measurements"] = len(found) + rejected
        print(f"{name:6s} last of 3: {runs[-1] * 1e3:9.1f} ms   (runs: {', '.join(f'{r * 1e3:.1f}' for r in runs)} ms)")
    for name in backends[1:]:
        print(f"speedup {name} over numpy (measure_delays, footprints included): "
              f"{out['seconds']['numpy'] / out['seconds'][name]:.1f}x ({out['measurements']} windows)")
    # The correlations alone, without the per-window Python bookkeeping that both paths share.
    recs = col.records
    window, lag = motion.window_sizes(recs[0], motion.MotionOptions())
    half = window // 2
    jobs = []
    for ra, rb in zip(recs[:-1], recs[1:]):
        shift = motion.phase_centre_shift(ra, rb)
        n = ra.samples.n_channels
        centres = np.linspace(half + lag, ra.samples.n_samples - half - lag, 8).astype(int)
        jobs.append((ra.samples.array, rb.samples.array, shift, n, centres))
    fs, fc = recs[0].samples.sample_rate, recs[0].samples.center_frequency

    def numpy_kernel():
        for xa, xb, shift, n, centres in jobs:
            for k in range(n - shift):
                for c in centres:
                    motion.correlate_window(xa[k + shift], xb[k], int(c), half, lag, fs, fc)

    def cpp_kernel(backend):
        for xa, xb, shift, n, centres in jobs:
            core.estimate_delays(xa, xb, list(range(shift, n)), list(range(n - shift)), [int(c) for c in centres],
                                 half, lag, fs, fc, backend)

    kernels = {"numpy": numpy_kernel, "host": lambda: cpp_kernel(core.Backend.Host)}
    if "cuda" in backends:
        kernels["cuda"] = lambda: cpp_kernel(core.Backend.Cuda)
    out["correlation_seconds"] = {}
    for name, fn in kernels.items():
        runs = []
        for _ in range(3):
            t0 = time.perf_counter()
            fn()
            runs.append(time.perf_counter() - t0)
        out["correlation_seconds"][name] = runs[-1]
        print(f"correlation only, {name:6s} last of 3: {runs[-1] * 1e3:9.2f} ms")
    for name in list(kernels)[1:]:
        print(f"speedup {name} over numpy (correlation only): "
              f"{out['correlation_seconds']['numpy'] / out['correlation_seconds'][name]:.1f}x")
    return out


def sweep(args) -> dict:
    rows = []
    if args.rows:
        rows = json.loads(Path(args.rows).read_text())["rows"]
        args.seeds = sorted({r["seed"] for r in rows})
    for seed in ([] if args.rows else args.seeds):
        col, _ = collect(seed=seed, with_target=False)
        truth = col.true_positions[:len(col.records)]
        for l1 in GRID:
            for l2 in GRID:
                opts = motion.MotionOptions(lambda_smooth=l1, lambda_dvl=l2, delay_backend="host")
                try:
                    r = motion.estimate_motion([x.copy() for x in col.records], opts)
                    sway = float(np.sqrt(np.mean((r.y - truth[:, 1]) ** 2)))
                    heave = float(np.sqrt(np.mean((r.z - truth[:, 2]) ** 2)))
                    ok = bool(r.fine.converged)
                except RuntimeError as e:          # a solve that cannot run is a row, not an abort
                    sway = heave = float("nan")
                    ok = False
                    print(f"seed {seed} l1 {l1} l2 {l2}: {e}")
                rows.append({"seed": seed, "lambda_smooth": l1, "lambda_dvl": l2, "sway_rms_mm": sway * 1e3,
                             "heave_rms_mm": heave * 1e3, "converged": ok})
    print("mean over seeds", args.seeds, ": sway RMS / heave RMS, mm (rows lambda1, columns lambda2)")
    print("l1 \\ l2 " + "".join(f"{l2:>16g}" for l2 in GRID))
    table = {}
    for l1 in GRID:
        line = f"{l1:7g} "
        for l2 in GRID:
            sel = [r for r in rows if r["lambda_smooth"] == l1 and r["lambda_dvl"] == l2]
            s = float(np.mean([r["sway_rms_mm"] for r in sel]))
            h = float(np.mean([r["heave_rms_mm"] for r in sel]))
            table[(l1, l2)] = (s, h)
            line += f"{s:8.3f}/{h:7.3f}"
        print(line)
    default = table[(1.0, 1.0)]
    best_sway = min(table, key=lambda k: table[k][0])
    best_heave = min(table, key=lambda k: table[k][1])
    # The acceptance bounds of tests/python/test_motion.py: sway lambda/20, heave lambda/4 RMS (lambda = 7.5 mm).
    wl = 1500.0 / 200e3 * 1e3
    worst = {}
    print("worst seed, max(sway / bound, heave / bound); below 1 = every seed inside both bounds")
    print("l1 \\ l2 " + "".join(f"{l2:>10g}" for l2 in GRID))
    for l1 in GRID:
        line = f"{l1:7g} "
        for l2 in GRID:
            sel = [r for r in rows if r["lambda_smooth"] == l1 and r["lambda_dvl"] == l2]
            worst[(l1, l2)] = max(max(r["sway_rms_mm"] / (wl / 20), r["heave_rms_mm"] / (wl / 4)) for r in sel)
            line += f"{worst[(l1, l2)]:10.2f}"
        print(line)
    best_both = min(worst, key=lambda k: worst[k])
    print(f"default (1, 1): sway {default[0]:.3f} mm, heave {default[1]:.3f} mm, worst-seed ratio {worst[(1.0, 1.0)]:.2f}")
    for name, k in (("best sway", best_sway), ("best heave", best_heave), ("best worst-seed ratio", best_both)):
        print(f"{name}: lambda1 {k[0]:g}, lambda2 {k[1]:g}: sway {table[k][0]:.3f} mm, heave {table[k][1]:.3f} mm, "
              f"worst-seed ratio {worst[k]:.2f}")
    return {"grid": GRID, "seeds": list(args.seeds), "rows": rows,
            "best_sway": best_sway, "best_heave": best_heave, "best_worst_seed": best_both}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)
    t = sub.add_parser("timing")
    t.add_argument("--pings", type=int, default=40)
    t.add_argument("--out")
    s = sub.add_parser("sweep")
    s.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    s.add_argument("--out")
    s.add_argument("--rows", help="report from the rows of an earlier --out JSON instead of running the grid")
    args = ap.parse_args()
    if not core.HAVE_MOTION:
        sys.exit("gpusonar was built without motion estimation (Ceres not found)")
    result = timing(args) if args.mode == "timing" else sweep(args)
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=1, default=str))


if __name__ == "__main__":
    main()
