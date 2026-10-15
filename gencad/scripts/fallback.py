"""Fallback combination of two predictors from per-part cq_iou.py rows: use the primary's part when its code builds a
valid solid, else the fallback's. Also reports an oracle (best of the two per part) as an upper bound.

  python scripts/cq_iou.py ... --rows results/vlm_cq_v1_test/rows.pkl
  python scripts/fallback.py --primary results/vlm_cq_v1_test/rows.pkl:vlm_cq:view4 \
      --fallback results/claude_baseline/rows.pkl:claude:view4 --out results/fallback/view4.json
"""
import argparse
import json
import pickle

import numpy as np


def load(spec):
    path, method = spec.split(":", 1)
    with open(path, "rb") as f:
        R = pickle.load(f)
    return R["uids"], np.asarray(R["dup"], bool), R["rows"][method]


def summary(rows, dup):
    out = {}
    for g, mask in {"all": np.ones(len(rows), bool), "dup": dup, "novel": ~dup}.items():
        r = [x for x, k in zip(rows, mask) if k and x is not None]
        out[g] = {"n": len(r), "IR": 100 * float(np.mean([not x["valid"] for x in r])),
                  "IoU_gt": 100 * float(np.mean([x["gt"] for x in r])),
                  "IoU_own": 100 * float(np.mean([x["own"] for x in r]))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--primary", required=True, help="rows.pkl:method")
    ap.add_argument("--fallback", required=True, help="rows.pkl:method")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ua, dup, A = load(a.primary)
    ub, _, B = load(a.fallback)
    assert ua == ub, "the two row files score different uids"
    comb = [x if x is None or x["valid"] else y for x, y in zip(A, B)]
    oracle = [x if x is None else max(x, y, key=lambda r: r["own"]) for x, y in zip(A, B)]
    n_fb = sum(x is not None and not x["valid"] for x in A)
    report = {"primary": a.primary, "fallback": a.fallback, "n_fallback_used": n_fb,
              "primary_only": summary(A, dup), "fallback_only": summary(B, dup),
              "fallback_on_invalid": summary(comb, dup), "oracle_best_own": summary(oracle, dup)}
    for k in ("primary_only", "fallback_only", "fallback_on_invalid", "oracle_best_own"):
        print(f"{k:20s} " + "  ".join(f"{g}: IoU gt/own={v['IoU_gt']:.1f}/{v['IoU_own']:.1f} IR={v['IR']:.1f}"
                                      for g, v in report[k].items()))
    print(f"fallback used on {n_fb} parts")
    with open(a.out, "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
