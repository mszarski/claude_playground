"""Render every CAD program from fixed cameras: data/renders/<uid>_<k>.png, k = 0..3.

The GenCAD renders of one uid are not views of its GT program (scripts/check_views.py): they show the part with
different proportions. These renders come straight from the GT program, so every pixel agrees with the label.

Cameras: orthographic, along the four tetrahedral diagonals (+1,+1,+1), (-1,-1,+1), (+1,-1,-1), (-1,+1,-1), so two
views look from above and two from below (hidden features of one side show in the other). All views of a part share
one scale (the bounding sphere of the solid fills the frame), so relative sizes are comparable across views.
Shading: grey Lambert per triangle on white, key light fixed to the camera; B-rep edges = boundaries between
different faces (and the silhouette) in the z-buffered face-ID image, drawn dark.

  python scripts/render_views.py --data data --out data/renders --workers 16 [--split test --limit 100]
"""
import argparse
import json
import os
import sys

import h5py
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DIRS = np.array([[1, 1, 1], [-1, -1, 1], [1, -1, -1], [-1, 1, -1]], float)


def tessellate(shape):
    """-> triangles (F, 3, 3) and the B-rep face index of each triangle."""
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
    from OCC.Core.TopAbs import TopAbs_FACE
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopLoc import TopLoc_Location
    from OCC.Core.TopoDS import topods
    BRepMesh_IncrementalMesh(shape, 2e-3, False, 0.25, True)
    tris, fid = [], []
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    k = 0
    while exp.More():
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation(topods.Face(exp.Current()), loc)
        if tri is not None and tri.NbTriangles():
            tr = loc.Transformation()
            nodes = np.array([tri.Node(i).Transformed(tr).Coord() for i in range(1, tri.NbNodes() + 1)])
            idx = np.array([tri.Triangle(i).Get() for i in range(1, tri.NbTriangles() + 1)]) - 1
            tris.append(nodes[idx])
            fid.append(np.full(len(idx), k))
        k += 1
        exp.Next()
    return np.concatenate(tris), np.concatenate(fid)


def rasterize(P, z, size):
    """Z-buffer: P (F, 3, 2) pixel coords, z (F, 3) depth (larger = closer). -> triangle index per pixel (-1 = none)."""
    zbuf = np.full((size, size), -np.inf)
    tid = np.full((size, size), -1, np.int64)
    lo = np.clip(np.floor(P.min(1)).astype(int), 0, size - 1)
    hi = np.clip(np.ceil(P.max(1)).astype(int), 0, size - 1)
    for t in range(len(P)):
        (x0, y0), (x1, y1) = lo[t], hi[t]
        xs, ys = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
        (ax, ay), (bx, by), (cx, cy) = P[t]
        den = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
        if abs(den) < 1e-12:
            continue
        w0 = ((by - cy) * (xs - cx) + (cx - bx) * (ys - cy)) / den
        w1 = ((cy - ay) * (xs - cx) + (ax - cx) * (ys - cy)) / den
        w2 = 1 - w0 - w1
        inside = (w0 >= -1e-6) & (w1 >= -1e-6) & (w2 >= -1e-6)
        if not inside.any():
            continue
        depth = w0 * z[t, 0] + w1 * z[t, 1] + w2 * z[t, 2]
        zb = zbuf[y0: y1 + 1, x0: x1 + 1]
        win = inside & (depth > zb)
        zb[win] = depth[win]
        tid[y0: y1 + 1, x0: x1 + 1][win] = t
    return tid


def render(tris, fid, d, size=448, margin=0.06):
    d = d / np.linalg.norm(d)
    up = np.array([0, 0, 1.0]) if abs(d[2]) < 0.99 else np.array([0, 1.0, 0])
    u = np.cross(up, d)
    u /= np.linalg.norm(u)
    v = np.cross(d, u)
    c = (tris.reshape(-1, 3).max(0) + tris.reshape(-1, 3).min(0)) / 2
    radius = np.linalg.norm(tris.reshape(-1, 3) - c, axis=1).max()
    T = tris - c
    s = size * (1 - 2 * margin) / (2 * radius)
    P = np.stack([T @ u * s + size / 2, -(T @ v) * s + size / 2], -1)            # (F, 3, 2)
    tid = rasterize(P, T @ d, size)
    n = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    n[(n @ d) < 0] *= -1                                                           # face the camera
    light = d + 0.4 * u + 0.7 * v                                                  # key light fixed to the camera
    light /= np.linalg.norm(light)
    shade = np.clip(0.45 + 0.5 * np.clip(n @ light, 0, 1), 0, 1) * 230
    a = np.where(tid >= 0, shade[tid], 255).astype(np.uint8)
    f = np.where(tid >= 0, fid[tid] + 1, 0)
    edge = np.zeros_like(f, bool)
    edge[:, 1:] |= f[:, 1:] != f[:, :-1]
    edge[1:, :] |= f[1:, :] != f[:-1, :]
    a[edge] = 40
    return Image.fromarray(a)


def render_uid(args):
    uid, data, out = args
    from mm.gencad_io import vec_to_shape
    paths = [os.path.join(out, f"{uid}_{k}.png") for k in range(len(DIRS))]
    if all(os.path.exists(p) for p in paths):
        return uid, "skip"
    try:
        with h5py.File(os.path.join(data, "cad_vec", uid + ".h5"), "r") as f:
            shape = vec_to_shape(f["vec"][:])
        if shape is None:
            return uid, "no_solid"
        tris, fid = tessellate(shape)
        os.makedirs(os.path.dirname(paths[0]), exist_ok=True)
        for d, p in zip(DIRS, paths):
            render(tris, fid, d).save(p, optimize=True)
        return uid, "ok"
    except Exception as e:  # noqa: BLE001
        return uid, f"error {repr(e)[:100]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="data/renders")
    ap.add_argument("--split", nargs="*", default=["train", "validation", "test"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--timeout", type=float, default=60.0)
    a = ap.parse_args()
    from pebble import ProcessPool
    with open(os.path.join(a.data, "filtered_data.json")) as f:
        splits = json.load(f)
    uids = [u for s in a.split for u in splits[s][: a.limit]]
    status = {}
    with ProcessPool(max_workers=a.workers) as pool:
        futs = {u: pool.schedule(render_uid, args=((u, a.data, a.out),), timeout=a.timeout) for u in uids}
        for k, (u, fu) in enumerate(futs.items()):
            try:
                status[u] = fu.result()[1]
            except Exception as e:  # noqa: BLE001 - timeout / crashed worker
                status[u] = f"error {repr(e)[:100]}"
            if k % 5000 == 0:
                print(f"[render] {k}/{len(uids)}", flush=True)
    counts = {}
    for s in status.values():
        counts[s.split()[0]] = counts.get(s.split()[0], 0) + 1
    print(f"[render] {counts}", flush=True)
    with open(os.path.join(a.out, "status.json"), "w") as f:
        json.dump(status, f)


if __name__ == "__main__":
    main()
