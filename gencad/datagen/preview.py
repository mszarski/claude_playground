"""Contact sheet of generated parts for visual review: `python -m datagen.preview <family> --n 24 --out sheet.png`.
Each tile is the part rendered with our isometric camera (scripts/render_views.py); failed parts show their reason.
"""
import argparse
import importlib
import os
import random
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("family")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--size", type=int, default=200)
    ap.add_argument("--cols", type=int, default=6)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from datagen.check import build, check
    from scripts.render_views import DIRS, render, tessellate_cq
    mod = importlib.import_module(f"datagen.families.{a.family}")
    rows = (a.n + a.cols - 1) // a.cols
    sheet = Image.new("RGB", (a.cols * a.size, rows * (a.size + 16)), "white")
    dr = ImageDraw.Draw(sheet)
    for i in range(a.n):
        code = mod.generate(random.Random(a.seed + i))
        ok, info = check(code)
        x, y = (i % a.cols) * a.size, (i // a.cols) * (a.size + 16)
        if ok:
            tris, fid = tessellate_cq(build(code))
            sheet.paste(Image.fromarray(np.asarray(render(tris, fid, DIRS[0], size=a.size))).convert("RGB"), (x, y))
            dr.text((x + 4, y + a.size + 2), f"#{a.seed + i} faces {info['faces']}", fill="black")
        else:
            dr.rectangle([x + 2, y + 2, x + a.size - 3, y + a.size - 3], fill=(250, 235, 235))
            dr.text((x + 6, y + a.size // 2), info["reason"][:30], fill=(170, 30, 30))
            dr.text((x + 4, y + a.size + 2), f"#{a.seed + i} FAIL", fill=(170, 30, 30))
    sheet.save(a.out)
    print(a.out)


if __name__ == "__main__":
    main()
