"""Do the 5 renders of a uid all show the GT program? Silhouette test.

The GenCAD renders of one uid look like the same part with different proportions (not different camera angles).
For each uid: build the GT solid, project its tessellation orthographically along each isometric diagonal, and
compare the filled silhouette with the non-background pixels of each render after cropping both to their bounding
box and resizing to 128x128 (removes position and overall scale, keeps aspect ratio and shape). Reports, per render
index, the silhouette IoU with the best-matching direction.

  python scripts/check_views.py --data data --n 60 --out results/check_views/report.json
"""
import argparse
import itertools
import json
import os
import sys

import h5py
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
R = 128


def triangles(shape):
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
    from OCC.Core.TopAbs import TopAbs_FACE
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopLoc import TopLoc_Location
    from OCC.Core.TopoDS import topods
    BRepMesh_IncrementalMesh(shape, 1e-3, False, 0.2, True)
    out = []
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation(topods.Face(exp.Current()), loc)
        if tri is not None:
            tr = loc.Transformation()
            nodes = np.array([tri.Node(i).Transformed(tr).Coord() for i in range(1, tri.NbNodes() + 1)])
            idx = np.array([tri.Triangle(i).Get() for i in range(1, tri.NbTriangles() + 1)]) - 1
            out.append(nodes[idx])
        exp.Next()
    return np.concatenate(out)


def normalise(mask):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return np.zeros((R, R), bool)
    crop = mask[ys.min(): ys.max() + 1, xs.min(): xs.max() + 1]
    return np.array(Image.fromarray(crop.astype(np.uint8) * 255).resize((R, R), Image.BILINEAR)) > 127


def project(tris, d, up=np.array([0, 0, 1.0])):
    d = d / np.linalg.norm(d)
    u = np.cross(up, d)
    u /= np.linalg.norm(u)
    v = np.cross(d, u)
    P = np.stack([tris @ u, -(tris @ v)], -1)                       # (F, 3, 2), image y down
    lo, hi = P.reshape(-1, 2).min(0), P.reshape(-1, 2).max(0)
    s = 400 / max((hi - lo).max(), 1e-9)
    P = (P - lo) * s + 4
    im = Image.new("L", (410, 410), 0)
    dr = ImageDraw.Draw(im)
    for t in P:
        dr.polygon([tuple(p) for p in t], fill=255)
    return normalise(np.array(im) > 0)


def render_mask(path):
    im = Image.open(path).convert("RGBA")
    a = np.array(im)
    fg = (a[..., 3] > 0) & (a[..., :3].min(-1) < 235)
    return normalise(fg)


def iou(a, b):
    return (a & b).sum() / max((a | b).sum(), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from mm.gencad_io import vec_to_shape
    with open(os.path.join(a.data, "filtered_data.json")) as f:
        splits = json.load(f)
    with open(os.path.join(a.data, "image_ids.json")) as f:
        img_ids = json.load(f)
    dirs = [np.array(s, float) for s in itertools.product([1, -1], repeat=3)]
    rows = []
    for split in ["train", "test"]:
        for u in [u for u in splits[split] if len(img_ids.get(u, [])) == 5][: a.n]:
            with h5py.File(os.path.join(a.data, "cad_vec", u + ".h5"), "r") as f:
                shape = vec_to_shape(f["vec"][:])
            if shape is None:
                continue
            tris = triangles(shape)
            sils = [project(tris, d) for d in dirs]
            sils += [np.fliplr(x) for x in sils]          # projection handedness is not known
            masks = {v: render_mask(os.path.join(a.data, "images", f"{u}_{v}.png")) for v in range(5)}
            best = {v: max(iou(m, s) for s in sils) for v, m in masks.items()}
            rows.append({"uid": u, "split": split, "iou": best})
            print(u, split, " ".join(f"{v}:{best[v]:.3f}" for v in range(5)), flush=True)
    M = np.array([[r["iou"][v] for v in range(5)] for r in rows])
    summary = {"n": len(rows), "mean_iou_by_index": M.mean(0).round(3).tolist(),
               "share_best_by_index": np.bincount(M.argmax(1), minlength=5).tolist(),
               "share_index_iou>0.95": (M > 0.95).mean(0).round(3).tolist(),
               "share_uids_any>0.95": float((M > 0.95).any(1).mean()),
               "share_uids_all>0.95": float((M > 0.95).all(1).mean())}
    print(json.dumps(summary, indent=1))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump({"summary": summary, "rows": rows}, f, indent=1)


if __name__ == "__main__":
    main()
