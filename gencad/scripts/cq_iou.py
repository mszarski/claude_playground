"""Volume IoU of predicted parts against the DeepCAD ground truth, computed with CadQuery's OpenCascade, for both
prediction kinds: DeepCAD programs (our models; built with upstream cadlib) and CadQuery source (Claude, CAD-Recode
style models). One protocol for every method, so the numbers compare directly.

Two normalisations are reported:
  gt  - GT fitted to a unit bounding box (centred). DeepCAD predictions get the same transform (scripts/geom_eval.py,
        CAD-Recode / cadrille); CadQuery predictions are written in the canonical frame (centred, longest side 200)
        and are only scaled by 1/200. Position and size errors count.
  own - each solid fitted to its own unit bounding box: only shape and proportions count (a fair setting for
        zero-shot models that cannot know DeepCAD's coordinate frame).
Invalid predictions (no code, code that fails, empty solid) count as IoU 0.

  python scripts/cq_iou.py --preds results/claude_baseline/preds.pkl --data data --out results/claude_baseline/iou.json
"""
import argparse
import json
import os
import pickle
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _fit(shape, ref=None):
    bb = (ref or shape).BoundingBox()
    c = ((bb.xmin + bb.xmax) / 2, (bb.ymin + bb.ymax) / 2, (bb.zmin + bb.zmax) / 2)
    s = 1.0 / max(bb.xlen, bb.ylen, bb.zlen)
    return shape.translate(tuple(-v for v in c)).scale(s)


def score(args):
    gt_vec, pred, kind = args
    from mm.cadquery_io import deepcad_reference, iou, run_cadquery
    gt = deepcad_reference(gt_vec)
    if gt is None or gt.Volume() <= 0:
        return None                                   # GT does not build: not scored
    if pred is None:
        p = None
    elif kind == "cadquery":
        p = run_cadquery(pred)
    else:
        p = deepcad_reference(np.asarray(pred))
    if p is None or p.Volume() <= 1e-12:
        return {"valid": False, "gt": 0.0, "own": 0.0}
    out = {"valid": True}
    p_gt = p.scale(1 / 200.0) if kind == "cadquery" else _fit(p, gt)
    for name, (a, b) in {"gt": (_fit(gt), p_gt), "own": (_fit(gt), _fit(p))}.items():
        try:
            out[name] = float(min(iou(a, b), 1.0))
        except Exception:  # noqa: BLE001 - boolean failure
            out[name] = 0.0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--data", default="data")
    ap.add_argument("--methods", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--task_timeout", type=float, default=120.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from pebble import ProcessPool
    with open(a.preds, "rb") as f:
        P = pickle.load(f)
    kind = P.get("kind", "deepcad")
    uids, dup = P["uids"], np.asarray(P["dup"], bool)
    gts = []
    for u in uids:
        with h5py.File(os.path.join(a.data, "cad_vec", u + ".h5"), "r") as f:
            gts.append(f["vec"][:].astype(np.int64))
    report = {"n": len(uids), "kind": kind, "protocol": __doc__.split("\n\n")[1]}
    for m in a.methods or list(P["preds"]):
        rows = []
        with ProcessPool(max_workers=a.workers) as pool:
            futs = [pool.schedule(score, args=((g, p, kind),), timeout=a.task_timeout)
                    for g, p in zip(gts, P["preds"][m])]
            for fu in futs:
                try:
                    rows.append(fu.result())
                except Exception:  # noqa: BLE001 - timeout / crashed worker -> invalid prediction
                    rows.append({"valid": False, "gt": 0.0, "own": 0.0})
        res = {}
        for g, mask in {"all": np.ones(len(uids), bool), "dup": dup, "novel": ~dup}.items():
            r = [x for x, k in zip(rows, mask) if k and x is not None]
            res[g] = {"n": len(r), "IR": 100 * float(np.mean([not x["valid"] for x in r])),
                      "IoU_gt": 100 * float(np.mean([x["gt"] for x in r])),
                      "IoU_own": 100 * float(np.mean([x["own"] for x in r]))}
        report[m] = res
        print(f"{m:26s} " + "  ".join(f"{g}: IoU gt/own={v['IoU_gt']:.1f}/{v['IoU_own']:.1f} IR={v['IR']:.1f} "
                                      f"(n={v['n']})" for g, v in res.items()), flush=True)
        with open(a.out, "w") as f:
            json.dump(report, f, indent=2)
    print(f"[done] {a.out}")


if __name__ == "__main__":
    main()
