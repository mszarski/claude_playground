"""How much of the rich-part error is orientation? For each prediction, own IoU under the best of the 24 axis-aligned
rotations (both solids fitted to their own unit box, rotation about the box centre) vs the plain own IoU.

  python scripts/rot_iou.py --preds results/<run>/revise.pkl --rows results/<run>/rich_rows.pkl --method vlm_cq:view4 \
      --families bent_sheet l_bracket --below 0.8 --out results/<run>/rot_iou.json
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mm.sysutil import usable_cpus  # noqa: E402


def rotations():
    """The 24 proper rotations of the cube as (axis, angle) sequences applied with Shape.rotate."""
    mats = []
    for perm in [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]:
        for signs in np.ndindex(2, 2, 2):
            m = np.zeros((3, 3))
            for i, (p, s) in enumerate(zip(perm, signs)):
                m[i, p] = -1 if s else 1
            if np.linalg.det(m) > 0:
                mats.append(m)
    return mats


def score(args):
    gt_code, pred = args
    import cadquery as cq
    from mm.cadquery_io import iou, run_cadquery
    from scripts.rich_iou import _fit
    gt, p = run_cadquery(gt_code), run_cadquery(pred) if pred else None
    if gt is None or p is None:
        return None
    a, b = _fit(gt), _fit(p)
    best = 0.0
    for m in rotations():
        loc = cq.Matrix([[*m[0], 0], [*m[1], 0], [*m[2], 0]])
        try:
            best = max(best, iou(a, _fit(b.transformGeometry(loc))))
        except Exception:  # noqa: BLE001
            pass
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--method", default="vlm_cq:view4")
    ap.add_argument("--families", nargs="*", default=None)
    ap.add_argument("--below", type=float, default=1.01, help="only parts whose own IoU is below this")
    ap.add_argument("--workers", type=int, default=usable_cpus())
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from pebble import ProcessPool
    P = pickle.load(open(a.preds, "rb"))
    R = pickle.load(open(a.rows, "rb"))["rows"][a.method]
    idx = [i for i, (f, r) in enumerate(zip(P["family"], R)) if r and r["valid"] and r["own"] < a.below
           and (not a.families or f in a.families)]
    print(f"[rot] {len(idx)} parts", flush=True)
    res = {}
    with ProcessPool(max_workers=a.workers) as pool:
        futs = [(i, pool.schedule(score, args=((P["gt_code"][i], P["preds"][a.method][i]),), timeout=300))
                for i in idx]
        for i, fu in futs:
            try:
                res[i] = fu.result()
            except Exception:  # noqa: BLE001
                res[i] = None
    report = {}
    for f in sorted({P["family"][i] for i in idx}):
        ii = [i for i in idx if P["family"][i] == f and res[i] is not None]
        own = np.array([R[i]["own"] for i in ii])
        rot = np.array([max(res[i], R[i]["own"]) for i in ii])
        report[f] = {"n": len(ii), "own": round(100 * own.mean(), 1), "best_rot": round(100 * rot.mean(), 1),
                     "fixed_by_rot": int(((rot >= 0.8) & (own < 0.8)).sum())}
        print(f"[rot] {f:18s} n={len(ii):3d} own {report[f]['own']:5.1f} -> best rotation {report[f]['best_rot']:5.1f}"
              f"  (>= 0.8 after rotation: {report[f]['fixed_by_rot']})", flush=True)
    json.dump(report, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
