"""Side-by-side gallery of test parts: the input render, the ground truth and each method's output, all rendered with the
same isometric camera (scripts/render_views.py), with the per-part IoU (GT normalisation) from cq_iou.py --rows under each tile.

  python scripts/gallery.py --data data --mode view4 --out gallery.png \
      --method "SFT=results/vlm_cq_v1_test/preds.pkl:vlm_cq:view4" ...  (rows files are found next to the preds)
Picks --n_random random novel parts plus the parts where the first and last methods differ most (both directions).
"""
import argparse
import os
import pickle
import random
import sys

import h5py
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def rows_for(preds_path, method):
    d = os.path.dirname(preds_path)
    for fn in ("rows.pkl", "rows2.pkl"):
        p = os.path.join(d, fn)
        if os.path.exists(p):
            R = pickle.load(open(p, "rb"))
            if method in R["rows"]:
                return R["rows"][method]
    raise SystemExit(f"no rows for {method} next to {preds_path}")


def tile(shape, size):
    from scripts.render_views import DIRS, render, tessellate_cq
    if shape is None:
        im = Image.new("RGB", (size, size), (250, 235, 235))
        ImageDraw.Draw(im).text((size // 2 - 40, size // 2 - 6), "invalid code", fill=(170, 40, 40))
        return im
    tris, fid = tessellate_cq(shape)
    return Image.fromarray(np.asarray(render(tris, fid, DIRS[0], size=size))).convert("RGB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--mode", default="view4")
    ap.add_argument("--method", nargs="+", required=True, help="Label=preds.pkl:method")
    ap.add_argument("--n_random", type=int, default=8)
    ap.add_argument("--n_contrast", type=int, default=4, help="parts where last vs first method differ most (each way)")
    ap.add_argument("--size", type=int, default=220)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from mm.cadquery_io import deepcad_reference, run_cadquery

    labels, codes, rows = [], [], []
    for spec in a.method:
        lab, rest = spec.split("=", 1)
        path, m = rest.split(":", 1)
        P = pickle.load(open(path, "rb"))
        labels.append(lab)
        codes.append(P["preds"][m])
        rows.append(rows_for(path, m))
        uids, dup = P["uids"], np.asarray(P["dup"], bool)
    ok = [i for i in range(len(uids)) if all(r[i] is not None for r in rows)]
    rng = random.Random(a.seed)
    novel = [i for i in ok if not dup[i]]
    pick = rng.sample(novel, a.n_random)
    diff = sorted(ok, key=lambda i: rows[-1][i]["gt"] - rows[0][i]["gt"])
    gains = [i for i in reversed(diff) if i not in pick][: a.n_contrast]
    losses = [i for i in diff if i not in pick + gains][: a.n_contrast // 2]
    sel = [(i, "random novel") for i in pick] + [(i, f"{labels[-1]} >> {labels[0]}") for i in gains] + \
          [(i, f"{labels[-1]} << {labels[0]}") for i in losses]

    s, pad, head = a.size, 6, 34
    cols = 2 + len(labels)
    W, H = cols * (s + pad) + pad + 150, head + len(sel) * (s + 30 + pad) + pad
    sheet = Image.new("RGB", (W, H), "white")
    dr = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 15)
    except OSError:
        font = ImageFont.load_default()
    for c, t in enumerate(["input (" + a.mode + ")", "ground truth"] + labels):
        dr.text((150 + pad + c * (s + pad) + 4, 8), t, fill="black", font=font)
    from train_vlm_cad import load_views
    from mm.vlm_cad import compose_views
    for r, (i, why) in enumerate(sel):
        y = head + r * (s + 30 + pad)
        u = uids[i]
        dr.text((6, y + 6), f"{u}\n{'dup' if dup[i] else 'novel'}\n{why}", fill="black", font=font)
        stem = os.path.join(a.data, "renders", u)
        views = [0, 1, 2, 3] if a.mode.startswith("view4") else [0]
        inp = compose_views(load_views("", stem, views), grid_side=448).resize((s, s))
        with h5py.File(os.path.join(a.data, "cad_vec", u + ".h5"), "r") as f:
            gt = deepcad_reference(f["vec"][:].astype(np.int64))
        tiles = [inp, tile(gt, s)] + [tile(run_cadquery(c[i]) if c[i] else None, s) for c in codes]
        for c, im in enumerate(tiles):
            x = 150 + pad + c * (s + pad)
            sheet.paste(im, (x, y))
            dr.rectangle([x, y, x + s - 1, y + s - 1], outline=(200, 200, 200))
            if c >= 2:
                v = rows[c - 2][i]["gt"] * 100       # GT-frame IoU: "own" can collapse on a stray-geometry bbox
                col = (20, 120, 40) if v >= 90 else (180, 120, 0) if v >= 60 else (180, 30, 30)
                dr.text((x + 4, y + s + 4), f"IoU {v:.0f}", fill=col, font=font)
        print(f"[gallery] {r + 1}/{len(sel)} {u}", flush=True)
    sheet.save(a.out)
    print(f"[done] {a.out}")


if __name__ == "__main__":
    main()
