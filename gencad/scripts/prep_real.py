"""Real-part test set from the Fusion 360 Gallery segmentation dataset (s2.0.1_extended_step.zip: STEP files of user
designs with a per-face label: 0 ExtrudeSide, 1 ExtrudeEnd, 2 CutSide, 3 CutEnd, 4 Fillet, 5 Chamfer, 6 RevolveSide,
7 RevolveEnd). Parts are grouped by their most telling feature (revolve > chamfer > fillet > plain extrude/cut),
--per_cat sampled per group, imported with CadQuery, checked (one valid solid, <= --max_faces faces) and rendered
with the input cameras. Output has the layout of scripts/gen_rich_data.py, so predict_vllm.py / revise_vllm.py --rich
and rich_iou.py work unchanged; the "GT program" imports the STEP file:

  python scripts/prep_real.py --zip <dir>/s2.0.1_extended_step.zip --out data/rich/real --per_cat 150
Writes <out>/<shard>/parts.parquet, <out>/<shard>/renders/<cat>/<name>_{0..3}.png, <out>/<shard>/step/<name>.stp.

Training split (no design shared with the test set; a design id is the name up to the first "_"):
  python scripts/prep_real.py --zip ... --out data/rich/real_train --per_cat 1000 --shard 0 --n_shards 4 \
      --exclude_designs datagen/real_test_designs.txt
"""
import argparse
import os
import random
import sys
import zipfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mm.sysutil import usable_cpus  # noqa: E402

CATS = ["revolve", "chamfer", "fillet", "plain"]


def category(labels):
    s = set(labels)
    if s & {6, 7}:
        return "revolve"
    if 5 in s:
        return "chamfer"
    if 4 in s:
        return "fillet"
    return "plain"


def work(args):
    """Import + check + render one STEP file -> (ok, info)."""
    step, stem, max_faces = args
    import cadquery as cq
    from scripts.render_views import DIRS, render, tessellate_cq
    try:
        wp = cq.importers.importStep(step)
        solids = wp.solids().vals()
        if len(solids) != 1:
            return False, f"{len(solids)} solids"
        shape = solids[0]
        if not shape.isValid() or shape.Volume() <= 1e-9:
            return False, "invalid"
        n_faces = len(shape.Faces())
        if n_faces > max_faces:
            return False, f"{n_faces} faces"
        tris, fid = tessellate_cq(shape)
        os.makedirs(os.path.dirname(stem), exist_ok=True)
        for j, d in enumerate(DIRS):
            render(tris, fid, d).save(f"{stem}_{j}.png", optimize=True)
        return True, {"faces": n_faces, "volume": float(shape.Volume())}
    except Exception as e:  # noqa: BLE001
        return False, type(e).__name__


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    ap.add_argument("--out", default="data/rich/real")
    ap.add_argument("--per_cat", type=int, default=150)
    ap.add_argument("--max_faces", type=int, default=120)
    ap.add_argument("--workers", type=int, default=usable_cpus())
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n_shards", type=int, default=1, help="designs are split across shards by a hash of their id")
    ap.add_argument("--exclude_designs", default=None, help="file of design ids to leave out (one per line)")
    a = ap.parse_args()
    import pandas as pd
    from pebble import ProcessPool
    rng = random.Random(a.seed + a.shard)
    z = zipfile.ZipFile(a.zip)
    names = sorted(n for n in z.namelist() if n.endswith(".seg"))
    import zlib
    excl = set(open(a.exclude_designs).read().split()) if a.exclude_designs else set()
    by_cat = {c: [] for c in CATS}
    for n in names:
        design = os.path.basename(n).split("_")[0]
        if design in excl or zlib.crc32(design.encode()) % a.n_shards != a.shard:
            continue
        labels = [int(x) for x in z.read(n).decode().split()]
        if 0 < len(labels) <= a.max_faces:
            by_cat[category(labels)].append(os.path.basename(n)[:-4])
    print("[real] candidates " + ", ".join(f"{c} {len(v)}" for c, v in by_cat.items()), flush=True)
    root = os.path.join(a.out, str(a.shard))
    os.makedirs(os.path.join(root, "step"), exist_ok=True)
    stp = {os.path.basename(n)[:-4]: n for n in z.namelist() if n.endswith(".stp")}
    rows = []
    with ProcessPool(max_workers=a.workers) as pool:
        for c in CATS:
            pool_c = by_cat[c][:]
            rng.shuffle(pool_c)
            kept, k = 0, 0
            while kept < a.per_cat and k < len(pool_c):
                batch = pool_c[k:k + 2 * a.workers]
                k += len(batch)
                futs = []
                for name in batch:
                    path = os.path.join(root, "step", name + ".stp")
                    with open(path, "wb") as f:
                        f.write(z.read(stp[name]))
                    futs.append((name, path, pool.schedule(
                        work, args=((path, os.path.join(root, "renders", c, name), a.max_faces),), timeout=120)))
                for name, path, fu in futs:
                    try:
                        ok, info = fu.result()
                    except Exception:  # noqa: BLE001
                        ok, info = False, "timeout"
                    if ok and kept < a.per_cat:
                        rows.append({"uid": f"{c}/{name}", "family": c, "faces": info["faces"],
                                     "volume": info["volume"], "code": f'r = cq.importers.importStep("{path}")'})
                        kept += 1
                    else:
                        os.remove(path)
            print(f"[real] {c}: kept {kept} of {k} tried", flush=True)
    pd.DataFrame(rows).to_parquet(os.path.join(root, "parts.parquet"))
    print(f"[done] {len(rows)} parts -> {root}")


if __name__ == "__main__":
    main()
