"""Render-agreement guard for the revision turn: keep a revision only when its render matches the input views better
than the turn-1 program's render (scripts/select_render.py "pix" score, input views only, no GT). Uses the
per-part IoU rows already computed for the revise preds file (cq_iou.py / rich_iou.py --rows) to report the guarded
result, so only rendering runs here.

  python scripts/revise_guard.py --preds results/<run>/revise.pkl --rows results/<run>/rich_rows.pkl --out ...
Reports, per mode and round, IoU of turn 1, unguarded revision, guarded revision.
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mm.sysutil import usable_cpus  # noqa: E402


def agreement(args):
    """-> mean pix agreement of `code`'s render with the input views (higher = better); None if it does not build."""
    code, stem, views, size = args
    from mm.cadquery_io import run_cadquery
    from scripts.render_views import DIRS, render, tessellate_cq
    from scripts.select_render import load_input
    shape = run_cadquery(code) if code else None
    if shape is None:
        return None
    tris, fid = tessellate_cq(shape)
    return float(np.mean([-np.abs(np.asarray(render(tris, fid, DIRS[k], size=size), np.float32) - ref).mean()
                          for k, ref in zip(views, load_input(stem, views, size))]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--data", default="data")
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=usable_cpus())
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from pebble import ProcessPool
    P = pickle.load(open(a.preds, "rb"))
    R = pickle.load(open(a.rows, "rb"))
    R = R["rows"] if "rows" in R else R
    stems = P.get("stems") or [os.path.join(a.data, "renders", u) for u in P["uids"]]
    key = "own" if "family" in P else "gt"
    base = sorted({m.split("@")[0] for m in P["preds"]})
    report = {}
    with ProcessPool(max_workers=a.workers) as pool:
        for b in base:
            views = [0, 1, 2, 3] if "view4" in b else [0]
            rounds = sorted(m for m in P["preds"] if m.startswith(b + "@"))
            cands = [b] + rounds
            futs = {m: [pool.schedule(agreement, args=((c, s, views, a.size),), timeout=120)
                        for c, s in zip(P["preds"][m], stems)] for m in cands}
            agr = {}
            for m, fs in futs.items():
                agr[m] = []
                for fu in fs:
                    try:
                        agr[m].append(fu.result())
                    except Exception:  # noqa: BLE001
                        agr[m].append(None)
            n = len(P["uids"])
            ok = [i for i in range(n) if R[b][i] is not None]
            score = lambda m, i: R[m][i][key] if R[m][i] else 0.0  # noqa: E731
            res = {b: 100 * float(np.mean([score(b, i) for i in ok]))}
            for m in rounds:
                upto, pick = rounds[:rounds.index(m) + 1], []
                for i in ok:                            # best-agreeing of turn 1 and the rounds so far
                    best = b
                    for r in upto:
                        if agr[r][i] is not None and (agr[best][i] is None or agr[r][i] > agr[best][i]):
                            best = r
                    pick.append(score(best, i))
                res[m] = 100 * float(np.mean([score(m, i) for i in ok]))
                res[m + "+guard"] = 100 * float(np.mean(pick))
            report[b] = res
            print(f"[guard] {b} ({key} IoU, n={len(ok)}): " + "  ".join(f"{k} {v:.1f}" for k, v in res.items()),
                  flush=True)
    json.dump(report, open(a.out, "w"), indent=1)
    print(f"[done] {a.out}")


if __name__ == "__main__":
    main()
