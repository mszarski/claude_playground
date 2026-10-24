"""Orientation search at inference (no training, no GT): for each predicted program, try the 24 axis-aligned rotations
of its solid, render each with the input cameras, and keep the rotation whose renders best match the input views
(scripts/select_render.py "pix" score). If it beats the program as written by more than --margin, the program gets
the rotation appended (`r = r.rotate(...)`), so the output is still a plain CadQuery program.

Motivation: on the weakest rich families most of the error is a correctly built part turned the wrong way (the
generators build in a canonical frame and finish with one `.rotate`; the model often picks the wrong one).

  python scripts/orient_fix.py --preds results/<run>/revise.pkl --methods vlm_cq:view4 --out results/<run>/orient.pkl
Writes a preds file holding only the `<method>+orient` predictions (scored by rich_iou.py / cq_iou.py).
"""
import argparse
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mm.sysutil import usable_cpus  # noqa: E402

AX = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}


def rot(axis, deg):
    """Right-handed rotation matrix about a principal axis (CadQuery's rotate convention)."""
    t = np.radians(deg)
    c, s = np.cos(t), np.sin(t)
    k = np.array(AX[axis], float)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.round(np.eye(3) + s * K + (1 - c) * K @ K, 12)


def candidates():
    """24 rotations as (rotate calls, matrix): bring +Z to one of 6 directions, then spin about Z."""
    out = []
    for first in [None, ("x", 90), ("x", 180), ("x", 270), ("y", 90), ("y", 270)]:
        for spin in [0, 90, 180, 270]:
            calls = ([first] if first else []) + ([("z", spin)] if spin else [])
            m = np.eye(3)
            for ax, deg in calls:
                m = rot(ax, deg) @ m
            out.append((calls, m))
    return out


def work(args):
    code, stem, views, size, margin = args
    from mm.cadquery_io import run_cadquery
    from scripts.render_views import DIRS, render, tessellate_cq
    from scripts.select_render import load_input
    if not code:
        return code, None
    shape = run_cadquery(code)
    if shape is None:
        return code, None
    tris, fid = tessellate_cq(shape)
    refs = load_input(stem, views, size)
    scores = []
    for calls, m in candidates():
        t = tris @ m.T
        scores.append(float(np.mean([-np.abs(np.asarray(render(t, fid, DIRS[k], size=size), np.float32) - ref).mean()
                                     for k, ref in zip(views, refs)])))
    best = int(np.argmax(scores))
    if best == 0 or scores[best] <= scores[0] + margin:
        return code, 0
    calls = candidates()[best][0]
    return code.rstrip() + "\n" + "\n".join(f"r = r.rotate((0, 0, 0), {AX[ax]}, {deg})" for ax, deg in calls), best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--data", default="data")
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--margin", type=float, default=0.5, help="grey levels the best rotation must win by")
    ap.add_argument("--workers", type=int, default=usable_cpus())
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from pebble import ProcessPool
    P = pickle.load(open(a.preds, "rb"))
    stems = P.get("stems") or [os.path.join(a.data, "renders", u) for u in P["uids"]]
    stats, out = {}, {}
    with ProcessPool(max_workers=a.workers) as pool:
        for m in a.methods:
            views = [0, 1, 2, 3] if "view4" in m else [0]
            futs = [pool.schedule(work, args=((c, s, views, a.size, a.margin),), timeout=300)
                    for c, s in zip(P["preds"][m], stems)]
            new, turned = [], 0
            for c, fu in zip(P["preds"][m], futs):
                try:
                    code, best = fu.result()
                except Exception:  # noqa: BLE001
                    code, best = c, None
                new.append(code)
                turned += bool(best)
            out[m + "+orient"] = new
            stats[m] = turned
            print(f"[orient] {m}: rotated {turned}/{len(new)}", flush=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    pickle.dump({**P, "preds": out, "orient_stats": stats}, open(a.out, "wb"))
    print(f"[done] {a.out}")


if __name__ == "__main__":
    main()
