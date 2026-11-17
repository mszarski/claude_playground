#!/usr/bin/env python3
"""Compare saved eval results with bootstrap confidence intervals.

    python scripts/compare.py name1=eval1.json name2=eval2.json [--baseline name1]

Prints exact-match with a 95% bootstrap CI per model, and (with --baseline) the paired difference to the
baseline on the same rows with its CI. A difference whose CI excludes 0 is marked *.
"""

from __future__ import annotations

import argparse
import json
import random


def ci(vals: list[float], n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    rng = random.Random(seed)
    n = len(vals)
    means = sorted(sum(vals[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return sum(vals) / n, means[int(0.025 * n_boot)], means[int(0.975 * n_boot)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("evals", nargs="+", help="name=path pairs")
    ap.add_argument("--baseline", default=None)
    ap.add_argument("--metric", default="exact")
    ap.add_argument("--by-kind", action="store_true")
    args = ap.parse_args()
    data = {}
    for item in args.evals:
        name, path = item.split("=", 1)
        data[name] = json.load(open(path))["rows"]
    kinds = ["OVERALL"] + (sorted({r["kind"] for r in next(iter(data.values()))}) if args.by_kind else [])
    for kind in kinds:
        print(f"-- {kind}")
        base = None
        if args.baseline:
            base = [r[args.metric] for r in data[args.baseline] if kind == "OVERALL" or r["kind"] == kind]
        for name, rows in data.items():
            vals = [float(r[args.metric]) for r in rows if kind == "OVERALL" or r["kind"] == kind]
            m, lo, hi = ci(vals)
            line = f"  {name:<22} n={len(vals):<5} {m:.3f}  [{lo:.3f}, {hi:.3f}]"
            if base is not None and name != args.baseline and len(base) == len(vals):
                d, dlo, dhi = ci([a - b for a, b in zip(vals, base)])
                line += f"   vs {args.baseline}: {d:+.3f} [{dlo:+.3f}, {dhi:+.3f}]{' *' if dlo > 0 or dhi < 0 else ''}"
            print(line)


if __name__ == "__main__":
    main()
