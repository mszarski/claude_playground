"""Every DeepCAD program -> CadQuery source in the CAD-Recode style (mm/cadquery_io.deepcad_to_cadquery), with a
fidelity check: the converted code is executed and its solid compared (volume IoU, both normalised to a unit bounding
box) with the reference solid that upstream cadlib builds from the original program.

Output parquet: uid, split, code, ok (code builds a solid), iou (vs reference; NaN if either side fails).

  python scripts/convert_deepcad.py --data data --out data/deepcad_cq.parquet --workers 32
"""
import argparse
import json
import os
import sys

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def convert(args):
    uid, path = args
    from mm.cadquery_io import deepcad_reference, deepcad_to_cadquery, iou, normalise, run_cadquery
    with h5py.File(path, "r") as f:
        vec = f["vec"][:].astype(np.int64)
    code = deepcad_to_cadquery(vec)
    if code is None:
        return uid, None, False, float("nan")
    shape = run_cadquery(code)
    if shape is None:
        return uid, code, False, float("nan")
    ref = deepcad_reference(vec)
    try:
        v = iou(normalise(ref), normalise(shape)) if ref is not None and ref.Volume() > 0 else float("nan")
        v = v if v <= 1.001 else float("nan")   # degenerate solids can report nonsense volumes
    except Exception:  # noqa: BLE001
        v = float("nan")
    return uid, code, True, float(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--task_timeout", type=float, default=60.0)
    a = ap.parse_args()
    from pebble import ProcessPool
    with open(os.path.join(a.data, "filtered_data.json")) as f:
        splits = json.load(f)
    items = [(u, s) for s in ["train", "validation", "test"] for u in splits[s][: a.limit]]
    rows = []
    with ProcessPool(max_workers=a.workers) as pool:
        futs = [(u, s, pool.schedule(convert, args=((u, os.path.join(a.data, "cad_vec", u + ".h5")),),
                                     timeout=a.task_timeout)) for u, s in items]
        for k, (u, s, fu) in enumerate(futs):
            try:
                _, code, ok, v = fu.result()
            except Exception:  # noqa: BLE001 - timeout / crashed worker
                code, ok, v = None, False, float("nan")
            rows.append((u, s, code, ok, v))
            if k % 10000 == 0:
                print(f"[convert] {k}/{len(futs)}", flush=True)
    df = pd.DataFrame(rows, columns=["uid", "split", "code", "ok", "iou"])
    for s, g in df.groupby("split"):
        print(f"[convert] {s}: n={len(g)} ok={g.ok.mean():.4f} iou mean={g.iou.mean():.4f} "
              f">0.99={(g.iou > 0.99).mean():.3f} >0.95={(g.iou > 0.95).mean():.3f}", flush=True)
    df.to_parquet(a.out, index=False)
    print(f"[convert] wrote {a.out}")


if __name__ == "__main__":
    main()
