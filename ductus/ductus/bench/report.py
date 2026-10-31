"""Overlay images and the docs/representability.md report."""
from __future__ import annotations

import datetime as _dt
import re
from pathlib import Path

import numpy as np

from .metrics import GlyphScore, bounds, raster

HEADER = """# Representability benchmark

How closely Ductus can represent a high-contrast display serif (reference:
Geist Serif, used for measurement only -- see SPEC section 8), and how much
hand work it took. One row per glyph per iteration; overlays live in
`docs/iterations/NN/`.

* **IoU** -- raster intersection-over-union at 400 px/em (1.0 = identical).
* **Hausdorff / mean** -- outline distance in font units (worst / average).
* **Δadv** -- advance width difference (spacing is scored separately from shape).
* **Overrides / Σ|Δ|** -- count and total magnitude of hand adjustments.

| iter | date | glyph | IoU | Hausdorff | mean | Δadv | overrides | Σ\\|Δ\\| | primitives | notes |
|---:|---|---|---:|---:|---:|---:|---:|---:|---|---|
"""


def next_iteration(docs: Path) -> int:
    it = docs / "iterations"
    nums = [int(p.name) for p in it.glob("[0-9][0-9]")] if it.exists() else []
    return max(nums, default=0) + 1


def append_rows(docs: Path, iteration: int, rows: list[dict]):
    path = docs / "representability.md"
    text = path.read_text() if path.exists() else HEADER
    today = _dt.date.today().isoformat()
    for r in rows:
        text += (f"| {iteration:02d} | {today} | `{r['glyph']}` | {r['iou']:.4f} | {r['hausdorff']:.1f} | "
                 f"{r['mean_dist']:.1f} | {r['advance_delta']:+d} | {r['overrides']} | {r['magnitude']:.0f} | "
                 f"{r['primitives']} | {r['notes']} |\n")
    path.write_text(text)
    return path


def overlay_image(score: GlyphScore, out: Path, px: int = 400, upm: int = 1000, title: str = ""):
    """Ours-only = red, reference-only = blue, both = dark grey, outline of ours on top."""
    from PIL import Image, ImageDraw
    a, b = bounds(score.ours) or (0, 0, 0, 0), bounds(score.ref) or (0, 0, 0, 0)
    box = (min(a[0], b[0], 0) - 60, min(a[1], b[1], -200) - 20, max(a[2], b[2]) + 60, max(a[3], b[3], 750) + 20)
    A = raster(score.ours, box, px, upm)
    B = raster(score.ref, box, px, upm)
    both = np.minimum(A, B)
    only_a, only_b = A - both, B - both
    img = np.ones(A.shape + (3,))
    for cov, col in ((both, (0.13, 0.13, 0.13)), (only_a, (0.86, 0.2, 0.25)), (only_b, (0.2, 0.42, 0.9))):
        img = img * (1 - cov[..., None]) + np.array(col) * cov[..., None]
    im = Image.fromarray((img * 255).astype(np.uint8))
    d = ImageDraw.Draw(im)
    s = px / upm
    H = im.height
    for y, lab in ((0, "baseline"), (530, "x"), (710, "cap")):
        yy = H - 1 - (y - box[1]) * s
        d.line([(0, yy), (im.width, yy)], fill=(200, 200, 200))
    if score.worst:
        (wx, wy), _ = score.worst
        cx, cy = (wx - box[0]) * s + 1, H - 1 - ((wy - box[1]) * s + 1)
        d.ellipse([cx - 7, cy - 7, cx + 7, cy + 7], outline=(230, 140, 0), width=2)
    label = (f"{title}{score.glyph}  IoU {score.iou:.4f}  H {score.hausdorff:.1f}u  "
             f"mean {score.mean_dist:.1f}u  dadv {score.advance_delta:+d}")
    canvas = Image.new("RGB", (max(im.width, 8 * len(label) + 20), im.height + 26), "white")
    canvas.paste(im, (0, 26))
    ImageDraw.Draw(canvas).text((8, 6), label, fill=(30, 30, 30))
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    return out
