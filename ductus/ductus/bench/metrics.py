"""Shape metrics between our glyph and a reference glyph.

Both outlines are aligned on the horizontal centre of their bounding boxes
(baselines shared), so spacing doesn't pollute the shape score; the advance
difference is reported separately.

* ``iou``: raster intersection-over-union of anti-aliased coverage
  (sum(min) / sum(max)) rendered at ``px`` pixels per em.
* ``hausdorff``: symmetric Hausdorff distance between the outlines, in
  font units; ``mean_dist``: mean of both directed nearest distances.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from fontTools.pens.boundsPen import BoundsPen
from fontTools.pens.freetypePen import FreeTypePen
from scipy.spatial import cKDTree

from ..geometry import Contour, bez

PX_PER_EM = 400


def bounds(contours: list[Contour]):
    if not contours:
        return None
    bp = BoundsPen(None)
    for c in contours:
        c.draw(bp)
    return bp.bounds


def translated(contours: list[Contour], dx: float) -> list[Contour]:
    out = []
    for c in contours:
        cc = c.copy()
        for p in cc.pts:
            p[0] += dx
        out.append(cc)
    return out


def align(ours: list[Contour], ref: list[Contour]) -> tuple[list[Contour], float]:
    """Translate ``ref`` so its bbox centre matches ours. Returns (ref', dx)."""
    a, b = bounds(ours), bounds(ref)
    if a is None or b is None:
        return ref, 0.0
    dx = (a[0] + a[2]) / 2 - (b[0] + b[2]) / 2
    return translated(ref, dx), dx


def sample_outline(contours: list[Contour], step: float = 2.0) -> np.ndarray:
    """Points along the outlines roughly every ``step`` units."""
    pts = []
    for c in contours:
        n = len(c.pts)
        i = 0
        while i < n:
            p0 = np.array(c.pts[i])
            if c.types[(i + 1) % n] == "off":
                P = np.array([p0, c.pts[(i + 1) % n], c.pts[(i + 2) % n], c.pts[(i + 3) % n]], float)
                L = np.linalg.norm(np.diff(P, axis=0), axis=1).sum()
                pts.append(bez(P, np.linspace(0, 1, max(2, int(L / step)) + 1)[:-1]))
                i += 3
            else:
                p1 = np.array(c.pts[(i + 1) % n], float)
                k = max(1, int(np.linalg.norm(p1 - p0) / step))
                pts.append(p0 + (p1 - p0) * np.linspace(0, 1, k + 1)[:-1, None])
                i += 1
    return np.concatenate(pts) if pts else np.zeros((0, 2))


def raster(contours: list[Contour], box, px: int = PX_PER_EM, upm: int = 1000) -> np.ndarray:
    s = px / upm
    x0, y0, x1, y1 = box
    w, h = int(np.ceil((x1 - x0) * s)) + 2, int(np.ceil((y1 - y0) * s)) + 2
    pen = FreeTypePen(None)
    for c in contours:
        c.draw(pen)
    return pen.array(width=w, height=h, transform=(s, 0, 0, s, -x0 * s + 1, -y0 * s + 1))


@dataclass
class GlyphScore:
    glyph: str
    iou: float
    hausdorff: float
    mean_dist: float
    advance_delta: int
    ref_dx: float
    ours: list = field(repr=False, default=None)
    ref: list = field(repr=False, default=None)        # aligned
    worst: tuple = field(repr=False, default=None)      # (point on ours or ref, which)


def score_glyph(name: str, ours: list[Contour], ref: list[Contour], our_adv: int, ref_adv: int,
                upm: int = 1000, px: int = PX_PER_EM) -> GlyphScore:
    ref_al, dx = align(ours, ref)
    a, b = bounds(ours), bounds(ref_al)
    if a is None or b is None:
        return GlyphScore(name, 1.0 if a is None and b is None else 0.0, float("nan"), float("nan"),
                          our_adv - ref_adv, dx, ours, ref_al)
    box = (min(a[0], b[0]) - 4, min(a[1], b[1]) - 4, max(a[2], b[2]) + 4, max(a[3], b[3]) + 4)
    A, B = raster(ours, box, px, upm), raster(ref_al, box, px, upm)
    iou = float(np.minimum(A, B).sum() / max(np.maximum(A, B).sum(), 1e-9))
    P, Q = sample_outline(ours), sample_outline(ref_al)
    dPQ, _ = cKDTree(Q).query(P)
    dQP, _ = cKDTree(P).query(Q)
    if dPQ.max() >= dQP.max():
        worst = (tuple(P[dPQ.argmax()]), "ours")
    else:
        worst = (tuple(Q[dQP.argmax()]), "ref")
    return GlyphScore(name, iou, float(max(dPQ.max(), dQP.max())),
                      float((dPQ.mean() + dQP.mean()) / 2), int(our_adv - ref_adv), dx,
                      ours, ref_al, worst)


def distance_field(ours: list[Contour], ref_aligned: list[Contour], step: float = 4.0) -> list[list[list[float]]]:
    """Per-contour polylines of our outline with distance-to-reference per
    point ([[x, y, d], ...]) -- the editor's error heat map."""
    if not ref_aligned:
        return []
    tree = cKDTree(sample_outline(ref_aligned, 1.0))
    out = []
    for c in ours:
        P = sample_outline([c], step)
        if not len(P):
            continue
        d, _ = tree.query(P)
        out.append([[round(float(x), 1), round(float(y), 1), round(float(dd), 1)] for (x, y), dd in zip(P, d)])
    return out
