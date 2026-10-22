"""Gallery for the rich-part test set (scripts/gen_rich_data.py): one random part per family, input views, the GT and
each method's output with the same isometric camera, own IoU (scripts/rich_iou.py --rows) under each tile. Methods are
aligned by uid, so preds files may cover different subsets of the test set (only shared uids are drawn).

  python scripts/gallery_rich.py --mode view4 --out gallery_rich.png \
      --method "old=results/rich_base_grpo_fullres/rich_preds.pkl:vlm_cq:view4" \
               "new=results/revise_grpo_rich/revise.pkl:vlm_cq:view4"
Rows files are found next to the preds (rich_rows.pkl).
"""
import argparse
import os
import pickle
import random
import sys

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.gallery import tile  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="view4")
    ap.add_argument("--method", nargs="+", required=True, help="Label=preds.pkl:method")
    ap.add_argument("--families", nargs="*", default=None)
    ap.add_argument("--size", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from mm.cadquery_io import run_cadquery
    from mm.vlm_cad import compose_views
    from train_vlm_cad import load_views

    labels, by_uid, info = [], [], {}
    for spec in a.method:
        lab, rest = spec.split("=", 1)
        path, m = rest.split(":", 1)
        P = pickle.load(open(path, "rb"))
        R = pickle.load(open(os.path.join(os.path.dirname(path), "rich_rows.pkl"), "rb"))
        labels.append(lab)
        by_uid.append({u: (c, r) for u, c, r in zip(P["uids"], P["preds"][m], R["rows"][m])})
        for u, f, g, s in zip(P["uids"], P["family"], P["gt_code"], P["stems"]):
            info[u] = (f, g, s)
    shared = [u for u in info if all(u in d and d[u][1] is not None for d in by_uid)]
    fams = a.families or sorted({info[u][0] for u in shared})
    rng = random.Random(a.seed)
    sel = [rng.choice([u for u in shared if info[u][0] == f]) for f in fams if any(info[u][0] == f for u in shared)]

    s, pad, head, left = a.size, 6, 30, 140
    cols = 2 + len(labels)
    sheet = Image.new("RGB", (left + cols * (s + pad) + pad, head + len(sel) * (s + 26 + pad) + pad), "white")
    dr = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
    for c, t in enumerate([f"input ({a.mode})", "ground truth"] + labels):
        dr.text((left + pad + c * (s + pad) + 4, 8), t, fill="black", font=font)
    views = [0, 1, 2, 3] if a.mode.startswith("view4") else [0]
    for r, u in enumerate(sel):
        fam, gt_code, stem = info[u]
        y = head + r * (s + 26 + pad)
        dr.text((6, y + 6), fam, fill="black", font=font)
        inp = compose_views(load_views("", stem, views), grid_side=448).resize((s, s))
        tiles = [inp, tile(run_cadquery(gt_code), s)]
        tiles += [tile(run_cadquery(d[u][0]) if d[u][0] else None, s) for d in by_uid]
        for c, im in enumerate(tiles):
            x = left + pad + c * (s + pad)
            sheet.paste(im, (x, y))
            dr.rectangle([x, y, x + s - 1, y + s - 1], outline=(200, 200, 200))
            if c >= 2:
                v = by_uid[c - 2][u][1]["own"] * 100
                col = (20, 120, 40) if v >= 90 else (180, 120, 0) if v >= 60 else (180, 30, 30)
                dr.text((x + 4, y + s + 4), f"IoU {v:.0f}", fill=col, font=font)
        print(f"[gallery] {r + 1}/{len(sel)} {fam} {u}", flush=True)
    sheet.save(a.out)
    print(f"[done] {a.out}")


if __name__ == "__main__":
    main()
