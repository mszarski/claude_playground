"""Train and score several generator configurations x seeds (a GPU job; see deploy/generator_sweep.py).

  python -m rmr.generator.sweep --out runs/sweep --steps 5000 --seeds 0 1 2

Each run trains on the Hub libraries (12 emotions held out) and is scored at the deployed sampler settings
(8 steps, guidance 1.5) on the held-out emotions, from their true plans and from deployment-like simplified plans
(``data.simplified_plan_dict``; live prompts always go through smooth recipe plans): identification top-1 and mean
rank, generated / real fast-detail energy, head-pitch and antenna speed p95 against the real clips, and
``energy_hold``. One JSON line per run in ``<out>/results.jsonl``; ``summary.json`` aggregates per configuration.
"""
import argparse
import json
import os
import time

import numpy as np

from .. import library
from .. import plan as PL

CONFIGS = {
    "reference": {},
    "simplify0.5": {"simplify": 0.5},
    "simplify1.0": {"simplify": 1.0},
    "energy_ada": {"model_kw": {"energy_ada": True}},
    "energy_ada+simplify0.5": {"simplify": 0.5, "model_kw": {"energy_ada": True}},
}


def score(net, stats, dev, H, seeds=3, steps=8):
    from .data import simplified_plan_dict
    from .evaluate import energy_hold, speeds
    from .sample import generate

    out, rng = {}, np.random.default_rng(0)
    real_e = {n: PL.frames(PL.extract(A), len(A))[:, 7].mean() for n, A in H.real.items()}
    for kind in ("true", "simplified"):
        r, e, sp = [], [], []
        for name, A in H.real.items():
            for s in range(seeds):
                plan = PL.extract(A) if kind == "true" else simplified_plan_dict(A, rng)
                G = generate(net, stats, plan, dev, seed=s, steps=steps)
                r.append(H.rank(G, name))
                e.append(PL.frames(PL.extract(G), len(G))[:, 7].mean() / real_e[name])
                sp.append(speeds(G))
        r, g = np.array(r), np.mean(sp, 0)
        out[kind] = {"top1": float((r == 1).mean()), "rank": float(r.mean()), "energy": float(np.median(e)),
                     "pitch_p95": float(g[0]), "ear_p95": float(g[2])}
    out["energy_hold"] = energy_hold(net, stats, dev)
    out["real_speed"] = {k: float(v) for k, v in zip(["pitch_p95", "pitch_peak", "ear_p95", "ear_peak"],
                                                     np.mean([speeds(A) for A in H.real.values()], 0))}
    return out


def summarize(rows):
    by = {}
    for r in rows:
        by.setdefault(r["config"], []).append(r)
    keys = [("true", "top1"), ("true", "rank"), ("true", "energy"), ("simplified", "top1"), ("simplified", "rank"),
            ("simplified", "energy"), ("simplified", "pitch_p95"), ("simplified", "ear_p95")]
    out = {}
    for c, rs in by.items():
        out[c] = {f"{a}.{b}": [float(np.mean([r["score"][a][b] for r in rs])), float(np.std([r["score"][a][b] for r in rs]))]
                  for a, b in keys}
        out[c]["energy_hold"] = {lv: float(np.mean([r["score"]["energy_hold"][lv] for r in rs])) for lv in rs[0]["score"]["energy_hold"]}
        out[c]["held_out_loss"] = float(np.mean([r["held_out_loss"] for r in rs]))
        out[c]["n"] = len(rs)
    return out


def main():
    from .evaluate import HeldOut
    from .sample import load
    from .train import train

    ap = argparse.ArgumentParser(prog="python -m rmr.generator.sweep")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    moves = library.hub(library.EMOTIONS) + library.hub(library.DANCES)
    held = library.HELD_OUT_EMOTIONS
    H = HeldOut(moves, held)
    rows = []
    for c in a.configs:
        for seed in a.seeds:
            t0, ck = time.time(), os.path.join(a.out, f"{c}_s{seed}.pt")
            train(moves, held, out=ck, steps=a.steps, seed=seed, log=lambda s: print(f"[{c} s{seed}] {s}", flush=True),
                  **CONFIGS[c])
            hist = json.load(open(os.path.splitext(ck)[0] + ".history.json"))
            net, stats, dev = load(ck)
            row = {"config": c, "seed": seed, "held_out_loss": min(h["held_out"] for h in hist),
                   "minutes": (time.time() - t0) / 60, "score": score(net, stats, dev, H)}
            rows.append(row)
            with open(os.path.join(a.out, "results.jsonl"), "a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"[{c} s{seed}] {json.dumps(row['score']['simplified'])} hold {row['score']['energy_hold']} "
                  f"({row['minutes']:.1f} min)", flush=True)
            with open(os.path.join(a.out, "summary.json"), "w") as f:
                json.dump(summarize(rows), f, indent=1)


if __name__ == "__main__":
    main()
