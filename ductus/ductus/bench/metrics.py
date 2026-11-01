"""Shape metrics between our glyph and a reference glyph.

The reference is shifted horizontally (baselines shared) to the offset that
maximises the IoU, so spacing doesn't pollute the shape score; the advance
difference is reported separately. (Until iteration 08 the bounding-box
centres were aligned: with outlines on the integer grid, that offset moves
in half-unit steps, and half a unit on a glyph with tall stems is worth
~0.005 IoU -- the score jumped as parameters moved.)

* ``iou``: exact intersection-over-union of the filled outlines (areas of
  the boolean intersection and union, computed on the curves with
  skia-pathops). Until iteration 08 this was rasterised at 400 px/em, i.e.
  2.5 units per pixel, which made the score step-shaped: small changes
  often did not move it, and the fitters stalled on the steps.
* ``hausdorff``: symmetric Hausdorff distance between the outlines, in
  font units; ``mean_dist``: mean of both directed nearest distances.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pathops
from fontTools.pens.areaPen import AreaPen
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


def _path(contours: list[Contour]) -> pathops.Path:
    path = pathops.Path()
    pen = path.getPen()
    for c in contours:
        c.draw(pen)
    path.simplify(fix_winding=True)
    return path


def _area(path: pathops.Path) -> float:
    pen = AreaPen()
    path.draw(pen)
    return abs(pen.value)


def exact_iou(ours: list[Contour], ref: list[Contour]) -> float:
    """Area of the intersection over area of the union, on the outlines."""
    return _iou_shifted(_path(ours), _path(ref), 0.0)


def _intersection_area(A: pathops.Path, B: pathops.Path) -> float:
    """Area of A ∩ B. Skia's boolean ops can fail -- or, worse, silently
    return a wrong outline -- when edges nearly coincide. Each result is
    checked against an independent one (area(A) - area(A - B)); on a
    failure or a disagreement, B is nudged by a thousandth of a unit (which
    can't change the area measurably) and the pair recomputed."""
    aA = _area(A)
    for eps in (0.0, 1e-3, -1e-3, 3e-3, -3e-3, 1e-2):
        Bt = B
        if eps:
            Bt = pathops.Path()
            B.draw(_Shift(Bt.getPen(), eps))
        try:
            i1 = _area(pathops.op(A, Bt, pathops.PathOp.INTERSECTION, fix_winding=True))
            i2 = aA - _area(pathops.op(A, Bt, pathops.PathOp.DIFFERENCE, fix_winding=True))
        except pathops.PathOpsError:
            continue
        if abs(i1 - i2) <= 1e-4 * aA + 0.5:
            i = (i1 + i2) / 2
            # skia can also be wrong *consistently* (both ops): an
            # intersection can't exceed either area, and must agree with a
            # coarse raster (2.5 u pixels: well within 1%) -- a fitter
            # found an o with "IoU 1.007" before this check
            aB = _area(Bt)
            if i <= min(aA, aB) + 0.5 and abs(i - _raster_intersection(A, Bt, 400)) <= 0.01 * min(aA, aB):
                return i
    return _raster_intersection(A, B)


def _raster_intersection(A: pathops.Path, B: pathops.Path, px: int = 2000) -> float:
    """Last resort when skia can't intersect the two reliably: coverage of
    both at 0.5 u per pixel (rare; the exact area is used whenever it is
    self-consistent)."""
    pa, pb = FreeTypePen(None), FreeTypePen(None)
    A.draw(pa); B.draw(pb)
    ba, bb = BoundsPen(None), BoundsPen(None)
    A.draw(ba); B.draw(bb)
    if ba.bounds is None or bb.bounds is None:
        return 0.0
    x0, y0 = min(ba.bounds[0], bb.bounds[0]) - 2, min(ba.bounds[1], bb.bounds[1]) - 2
    x1, y1 = max(ba.bounds[2], bb.bounds[2]) + 2, max(ba.bounds[3], bb.bounds[3]) + 2
    k = px / 1000
    w, h = int((x1 - x0) * k) + 2, int((y1 - y0) * k) + 2
    tr = (k, 0, 0, k, -x0 * k, -y0 * k)
    ra, rb = pa.array(width=w, height=h, transform=tr), pb.array(width=w, height=h, transform=tr)
    return float(np.minimum(ra, rb).sum()) / (k * k)


def _iou_shifted(A: pathops.Path, B: pathops.Path, dx: float) -> float:
    Bt = B
    if dx:
        Bt = pathops.Path()
        B.draw(_Shift(Bt.getPen(), dx))
    inter = _intersection_area(A, Bt)
    union = _area(A) + _area(B) - inter        # area is translation-invariant
    return inter / union if union > 0 else 1.0


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
    # outline distance near the stroke ends (within ENDS_RADIUS of a terminal
    # shape): mean over all of them, and per terminal style. IoU counts area,
    # and a 4 u tip or a small fillet is almost none -- this is what sees them.
    ends_mean: float = float("nan")
    ends: dict = field(default_factory=dict)


def align_best(ours: list[Contour], ref: list[Contour], window: float = 10.0):
    """Shift ``ref`` horizontally to the offset maximising the exact IoU,
    searched around the bbox-centre offset. Returns (ref', dx, iou)."""
    from scipy.optimize import minimize_scalar
    _, dx0 = align(ours, ref)
    if not ours or not ref:
        return ref, dx0, None
    A, B = _path(ours), _path(ref)

    def neg_iou(dx):
        return -_iou_shifted(A, B, dx)

    r = minimize_scalar(neg_iou, bounds=(dx0 - window, dx0 + window), method="bounded",
                        options={"xatol": 0.02})
    dx, iou = (r.x, -r.fun) if -r.fun >= -neg_iou(dx0) else (dx0, -neg_iou(dx0))
    return translated(ref, dx), float(dx), float(iou)


class _Shift:
    """Pen filter translating x by ``dx``."""
    def __init__(self, pen, dx):
        self.pen, self.dx = pen, dx

    def _t(self, pts):
        return [(x + self.dx, y) for x, y in pts]

    def moveTo(self, p): self.pen.moveTo(*self._t([p]))
    def lineTo(self, p): self.pen.lineTo(*self._t([p]))
    def curveTo(self, *pts): self.pen.curveTo(*self._t(pts))
    def qCurveTo(self, *pts): self.pen.qCurveTo(*self._t(pts[:-1]), *([pts[-1]] if pts[-1] is None else self._t([pts[-1]])))
    def closePath(self): self.pen.closePath()
    def endPath(self): self.pen.endPath()


ENDS_RADIUS = 30.0


def terminal_style(c: Contour) -> str:
    """A terminal shape's style, from its point names (``n.stem.top.wedge.tip``)."""
    for nm in c.names:
        if nm:
            parts = nm.split(".")
            for p in parts:
                if p in ("serif", "wedge", "beak", "ball"):
                    return {"serif": "bracketed"}.get(p, p)
            return "cap" if "cap" in parts or "cut" in parts else parts[-1]
    return "other"


def ends_distance(P, dPQ, Q, dQP, terminals, radius: float = ENDS_RADIUS):
    """Mean outline distance (both ways) at outline points within ``radius``
    of a terminal shape's outline, overall and per style."""
    if not terminals:
        return float("nan"), {}
    by_style: dict[str, list] = {}
    for c in terminals:
        by_style.setdefault(terminal_style(c), []).append(c)
    allpts = np.concatenate([sample_outline([c], 2.0) for c in terminals])
    def near(pts, tpts):
        d, _ = cKDTree(tpts).query(pts)
        return d <= radius
    out, tot, n = {}, 0.0, 0
    for st, cs in by_style.items():
        tp = sample_outline(cs, 2.0)
        mP, mQ = near(P, tp), near(Q, tp)
        k = int(mP.sum() + mQ.sum())
        if k:
            out[st] = float((dPQ[mP].sum() + dQP[mQ].sum()) / k)
    mP, mQ = near(P, allpts), near(Q, allpts)
    k = int(mP.sum() + mQ.sum())
    return (float((dPQ[mP].sum() + dQP[mQ].sum()) / k) if k else float("nan")), out


def score_glyph(name: str, ours: list[Contour], ref: list[Contour], our_adv: int, ref_adv: int,
                upm: int = 1000, px: int = PX_PER_EM, terminals: list[Contour] | None = None) -> GlyphScore:
    ref_al, dx, iou = align_best(ours, ref)
    a, b = bounds(ours), bounds(ref_al)
    if a is None or b is None:
        return GlyphScore(name, 1.0 if a is None and b is None else 0.0, float("nan"), float("nan"),
                          our_adv - ref_adv, dx, ours, ref_al)
    if iou is None:
        iou = exact_iou(ours, ref_al)
    P, Q = sample_outline(ours), sample_outline(ref_al)
    dPQ, _ = cKDTree(Q).query(P)
    dQP, _ = cKDTree(P).query(Q)
    if dPQ.max() >= dQP.max():
        worst = (tuple(P[dPQ.argmax()]), "ours")
    else:
        worst = (tuple(Q[dQP.argmax()]), "ref")
    ends_mean, ends = ends_distance(P, dPQ, Q, dQP, terminals)
    return GlyphScore(name, iou, float(max(dPQ.max(), dQP.max())),
                      float((dPQ.mean() + dQP.mean()) / 2), int(our_adv - ref_adv), dx,
                      ours, ref_al, worst, ends_mean, ends)


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
