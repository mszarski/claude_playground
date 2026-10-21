"""Geometric scores for predicted CAD programs: IoU, Chamfer distance and invalidity ratio.

Same protocol as CAD-Recode / cadrille (arXiv 2505.22914): the GT solid is normalised to fit [-0.5, 0.5]^3 and the
prediction gets the *same* transform (so a wrong overall size is penalised); CD = mean squared nearest-neighbour
distance both ways on 8192 surface points, x1e3, reported as a median over valid predictions; IoU = vol(A and B) /
vol(A or B) from exact OCC booleans, reported as a mean with invalid predictions counted as 0 (strict) and over valid
ones; IR = share of predictions that do not build a non-empty solid.

Input: the --save_preds file of scripts/baselines.py (uids, dup mask, per-method program lists). Needs pythonocc-core.

  python scripts/geom_eval.py --preds results/baselines_geom/preds.pkl --data data --out results/baselines_geom/geom.json
"""
import argparse
import json
import os
import pickle
import sys

import h5py
import numpy as np
from pebble import ProcessPool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mm.sysutil import usable_cpus  # noqa: E402

N_POINTS = 8192


def _volume(shape):
    from OCC.Core.BRepGProp import brepgprop
    from OCC.Core.GProp import GProp_GProps
    p = GProp_GProps()
    brepgprop.VolumeProperties(shape, p)
    return abs(p.Mass())


def _bbox(shape):
    from OCC.Core.Bnd import Bnd_Box
    from OCC.Core.BRepBndLib import brepbndlib
    b = Bnd_Box()
    brepbndlib.Add(shape, b)
    return np.array(b.Get())   # xmin, ymin, zmin, xmax, ymax, zmax


def _transform(shape, center, scale):
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
    from OCC.Core.gp import gp_Pnt, gp_Trsf, gp_Vec
    t, s = gp_Trsf(), gp_Trsf()
    t.SetTranslation(gp_Vec(*(-center)))
    s.SetScale(gp_Pnt(0, 0, 0), scale)
    return BRepBuilderAPI_Transform(shape, s.Multiplied(t), True).Shape()


def _sample(shape, n, seed=0):
    """Area-weighted surface samples from a fresh tessellation."""
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
    from OCC.Core.TopAbs import TopAbs_FACE
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopLoc import TopLoc_Location
    from OCC.Core.TopoDS import topods
    BRepMesh_IncrementalMesh(shape, 2e-3, False, 0.2, True)
    tris = []
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        face = topods.Face(exp.Current())
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation(face, loc)
        if tri is not None:
            tr = loc.Transformation()
            nodes = np.array([tri.Node(i).Transformed(tr).Coord() for i in range(1, tri.NbNodes() + 1)])
            idx = np.array([tri.Triangle(i).Get() for i in range(1, tri.NbTriangles() + 1)]) - 1
            tris.append(nodes[idx])
        exp.Next()
    if not tris:
        return None
    T = np.concatenate(tris)                                          # (F, 3, 3)
    area = 0.5 * np.linalg.norm(np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]), axis=1)
    if area.sum() <= 0:
        return None
    rng = np.random.default_rng(seed)
    f = rng.choice(len(T), n, p=area / area.sum())
    u, v = rng.random(n), rng.random(n)
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    a, b, c = T[f, 0], T[f, 1], T[f, 2]
    return a + u[:, None] * (b - a) + v[:, None] * (c - a)


def _gt(vec):
    """GT solid and the transform that fits it into [-0.5, 0.5]^3 (None if the GT itself does not build)."""
    from mm.gencad_io import vec_to_shape
    shape = vec_to_shape(vec)
    if shape is None or _volume(shape) <= 0:
        return None
    bb = _bbox(shape)
    return shape, (bb[:3] + bb[3:]) / 2, 1.0 / (bb[3:] - bb[:3]).max()


def score_one(gt_vec, pred_vec):
    """-> dict(valid, iou, cd) for one prediction against its GT; valid=False also when the GT fails (skipped)."""
    from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Common
    from scipy.spatial import cKDTree
    from mm.gencad_io import vec_to_shape
    norm = _gt(gt_vec)
    if norm is None:
        return {"gt_ok": False}
    g, center, scale = norm
    g = _transform(g, center, scale)
    if pred_vec is None:
        return {"gt_ok": True, "valid": False}
    p = vec_to_shape(np.asarray(pred_vec))
    if p is None:
        return {"gt_ok": True, "valid": False}
    p = _transform(p, center, scale)
    vp, vg = _volume(p), _volume(g)
    if vp <= 1e-9:
        return {"gt_ok": True, "valid": False}
    common = BRepAlgoAPI_Common(g, p)
    vc = _volume(common.Shape()) if common.IsDone() else 0.0
    iou = vc / max(vg + vp - vc, 1e-12)
    sg, sp = _sample(g, N_POINTS, 0), _sample(p, N_POINTS, 1)
    cd = None
    if sg is not None and sp is not None:
        d1, _ = cKDTree(sg).query(sp, k=1)
        d2, _ = cKDTree(sp).query(sg, k=1)
        cd = float((np.mean(d1 ** 2) + np.mean(d2 ** 2)) * 1e3)
    return {"gt_ok": True, "valid": True, "iou": float(min(iou, 1.0)), "cd": cd}


def _job(args):
    gt_vec, pred_vec = args
    try:
        return score_one(gt_vec, pred_vec)
    except Exception as e:  # noqa: BLE001 - any OCC failure on a prediction counts as invalid
        return {"gt_ok": True, "valid": False, "err": repr(e)[:200]}


def summarise(rows):
    rows = [r for r in rows if r.get("gt_ok")]
    if not rows:
        return {"n": 0}
    valid = [r for r in rows if r.get("valid")]
    ious = np.array([r["iou"] for r in valid])
    cds = np.array([r["cd"] for r in valid if r.get("cd") is not None])
    return {"n": len(rows), "IR": 100 * (1 - len(valid) / len(rows)),
            "IoU": 100 * float(ious.sum() / len(rows)),             # invalid = 0
            "IoU_valid": 100 * float(ious.mean()) if len(ious) else 0.0,
            "CD_median": float(np.median(cds)) if len(cds) else None,
            "CD_mean": float(np.mean(cds)) if len(cds) else None,
            "iou>0.9": float((ious > 0.9).sum() / len(rows)), "iou>0.5": float((ious > 0.5).sum() / len(rows))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--data", default="data")
    ap.add_argument("--methods", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=None, help="first N uids only (smoke test)")
    ap.add_argument("--workers", type=int, default=usable_cpus())
    ap.add_argument("--task_timeout", type=float, default=120.0, help="seconds per prediction; a hang counts as invalid")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    with open(a.preds, "rb") as f:
        P = pickle.load(f)
    uids, dup = P["uids"][: a.limit], np.asarray(P["dup"], bool)[: a.limit]
    gts = []
    for u in uids:
        with h5py.File(os.path.join(a.data, "cad_vec", u + ".h5"), "r") as f:
            gts.append(f["vec"][:].astype(np.int64))
    methods = a.methods or list(P["preds"])
    report = {"n": len(uids), "protocol": __doc__.split("\n\n")[1]}
    for m in methods:
        preds = P["preds"][m][: len(uids)]
        rows = []
        # pebble kills a worker that hangs past the timeout or crashes (OCC segfault) and fails only that task
        with ProcessPool(max_workers=a.workers) as pool:
            futs = [pool.schedule(_job, args=((g, p),), timeout=a.task_timeout) for g, p in zip(gts, preds)]
            for fu in futs:
                try:
                    rows.append(fu.result())
                except Exception as e:  # noqa: BLE001 - TimeoutError / ProcessExpired -> invalid prediction
                    rows.append({"gt_ok": True, "valid": False, "err": repr(e)[:200]})
        res = {g: summarise([r for r, k in zip(rows, mask) if k])
               for g, mask in {"all": np.ones(len(uids), bool), "dup": dup, "novel": ~dup}.items()}
        report[m] = res
        print(f"{m:26s} " + "  ".join(
            f"{g}: IoU={r['IoU']:.1f} (valid {r['IoU_valid']:.1f}) CDmed={r['CD_median'] or float('nan'):.3f} "
            f"IR={r['IR']:.1f} (n={r['n']})" for g, r in res.items() if r.get("n")), flush=True)
        with open(a.out, "w") as f:
            json.dump(report, f, indent=2)
    print(f"[done] {a.out}")


if __name__ == "__main__":
    main()
