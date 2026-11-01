"""Contour parts: shapes defined directly as outlines (spec section 5.4).

``bowl`` builds a bowl from two contours -- outer and inner -- each made of
four named extrema (``bottom right top left``) joined by four cubic
quadrants. The construction always yields 4 on-curve + 8 off-curve points
per contour (principle 4), whatever the parameter values.

Controls
  box               outer extrema: left, right, bottom, top
  thick, thin       wall thickness at the sides / at top and bottom
                    (``thick_right`` / ``thin_bottom`` for asymmetry)
  stress            angle (degrees) of the thin axis from vertical;
                    positive = classic oblique stress (top of counter leans
                    left). Moves the inner extrema along the rotated axes.
  fullness          handle length as a fraction of each quadrant's extent:
                    0.552 = ellipse, higher = squarer (superellipse-like).
                    A number; an ``(h, v)`` pair giving the horizontal
                    handles (at top/bottom) and vertical handles (at the
                    sides) separately -- like a superellipse with independent
                    exponents; or 4 of either for the quadrants in order
                    bottom-right, top-right, top-left, bottom-left.
  inner_fullness    the same for the counter (default: ``fullness``).
  outer_stress      optional rotation of the outer extrema (default 0).
  rotate            rotate the whole bowl -- both contours, tangents
                    included -- about its centre (degrees, CCW). For tilted
                    bowls like the a's, whose top climbs into the stem.
"""
from __future__ import annotations

import math

from .params import val

QUADRANTS = ("br", "tr", "tl", "bl")


def _hv(v) -> tuple[float, float]:
    if isinstance(v, (tuple, list)):
        if len(v) != 2:
            raise ValueError("a quadrant fullness is a number or an (h, v) pair")
        return (val(v[0]), val(v[1]))
    return (val(v), val(v))


def _four(v) -> list[tuple[float, float]]:
    """-> 4 quadrants (br, tr, tl, bl) of (h, v) handle fractions."""
    if isinstance(v, (tuple, list)) and len(v) == 4:
        return [_hv(x) for x in v]
    return [_hv(v)] * 4


def extrema(left, right, bottom, top, stress_deg=0.0) -> dict[str, tuple[float, float]]:
    """The four extrema of a box, rotated about its centre by ``stress``."""
    cx, cy = (left + right) / 2, (bottom + top) / 2
    k = math.tan(math.radians(stress_deg))
    return {
        "bottom": (cx - (bottom - cy) * k, bottom),
        "right": (right, cy - (right - cx) * k),
        "top": (cx - (top - cy) * k, top),
        "left": (left, cy - (left - cx) * k),
    }


def quadrant_contour(cb, ext: dict, fullness: list[float]):
    """Emit bottom -> right -> top -> left (counter-clockwise) into a
    ContourBuilder, with horizontal tangents at bottom/top and vertical
    tangents at left/right."""
    (xb, b), (r, yr), (xt, t), (l, yl) = ext["bottom"], ext["right"], ext["top"], ext["left"]
    (h0, v0), (h1, v1), (h2, v2), (h3, v3) = fullness
    cb.move((xb, b), "bottom")
    cb.curve((xb + h0 * (r - xb), b), (r, yr - v0 * (yr - b)), (r, yr), "right")
    cb.curve((r, yr + v1 * (t - yr)), (xt + h1 * (r - xt), t), (xt, t), "top")
    cb.curve((xt - h2 * (xt - l), t), (l, yl + v2 * (t - yl)), (l, yl), "left")
    cb.curve_close((l, yl - v3 * (yl - b)), (xb - h3 * (xb - l), b))
    cb.close()
    return cb


def _rotate(cb, cx, cy, deg):
    a = math.radians(deg)
    ca, sa = math.cos(a), math.sin(a)
    for p in cb.contour.pts:
        x, y = p[0] - cx, p[1] - cy
        p[0], p[1] = cx + x * ca - y * sa, cy + x * sa + y * ca


def bowl(builder, left, right, bottom, top, thick, thin, stress=0.0, fullness=0.6,
         inner_fullness=None, thick_right=None, thin_bottom=None, outer_stress=0.0,
         rotate=0.0, prefix: str = ""):
    """Add an outer + inner contour pair to ``builder``; returns both."""
    l, r, b, t = val(left), val(right), val(bottom), val(top)
    tl_ = val(thick)
    tr_ = val(thick_right) if thick_right is not None else tl_
    th = val(thin)
    tb = val(thin_bottom) if thin_bottom is not None else th
    outer = builder.contour(prefix + "outer")
    inner = builder.contour(prefix + "inner")
    group = object()
    outer.group = inner.group = group
    outer.kind = inner.kind = "bowl"
    quadrant_contour(outer, extrema(l, r, b, t, val(outer_stress)), _four(fullness))
    quadrant_contour(inner, extrema(l + tl_, r - tr_, b + tb, t - th, val(stress)),
                     _four(inner_fullness if inner_fullness is not None else fullness))
    rot = val(rotate)
    if rot:
        for cb in (outer, inner):
            _rotate(cb, (l + r) / 2, (b + t) / 2, rot)
    return outer, inner


def lobe(builder, edge_x, side, far, bottom, top_join, bottom_join, thick, thin_bottom,
         out_mid=0.5, in_mid=0.5, bx_out=0.5, bx_in=0.5, tension=1.0, in_tension=1.0, prefix: str = ""):
    """A bowl closed by a stem (a d q / b p).

    Outer contour and counter both start and end on the stem edge, at two
    :class:`~ductus.junctions.Join` objects:

    * ``top_join``: ``at`` = height of the upper crotch (counter meets stem),
      ``thin`` = thickness of the lobe's top wall where it enters the stem,
      ``angle`` = how steeply the top wall *descends* away from the stem;
    * ``bottom_join``: ``at`` = height of the lower crotch, ``thin`` = wall
      thickness there, ``angle`` = how steeply the bottom wall *rises* into
      the stem.

    ``edge_x`` is the stem edge, ``side`` -1 = lobe left of it (a, d, q),
    +1 = right (b, p). ``far`` is the outer far extreme with wall ``thick``;
    ``bottom`` the outer bottom with wall ``thin_bottom``. ``out_mid`` /
    ``in_mid`` place the far extremes' heights (fraction from the bottom up to
    the top contact); ``bx_out`` places the outer bottom extreme (fraction from the
    far side towards the stem); ``bx_in`` the counter's (fraction from its far
    wall towards the outer bottom extreme -- it can never pass it).

    Fixed structure: each contour has 4 named nodes (``top far bottom end``)
    joined by 3 Hobby cubics, closed by straight runs along/inside the stem.
    """
    from .geometry import hobby_cubic
    from .junctions import contact
    ex, f, b = val(edge_x), val(far), val(bottom)
    t_at, t_thin, t_ang, _, _ = top_join.resolved()
    b_at, b_thin, b_ang, _, _ = bottom_join.resolved()
    tc, to, *_ = contact(top_join, ex, side, rising=True)       # crotch below, outer contact above
    bc, bo, *_ = contact(bottom_join, ex, side, rising=False)   # crotch above, outer contact below
    th, tb = val(thick), val(thin_bottom)
    away = 1 if side > 0 else -1                                  # x direction away from the stem

    # travel directions (degrees): leave the stem at the top, go down the far
    # side, along the bottom back towards the stem, arrive at the bottom join
    d_top = math.degrees(math.atan2(-math.sin(math.radians(t_ang)), away * math.cos(math.radians(t_ang))))
    d_far = 270.0
    d_bot = 0.0 if side < 0 else 180.0
    d_end = math.degrees(math.atan2(math.sin(math.radians(b_ang)), -away * math.cos(math.radians(b_ang))))
    dirs = [d_top, d_far, d_bot, d_end]

    o_pts = [to, (f, b + (to[1] - b) * val(out_mid)), (f + (ex - f) * val(bx_out), b), bo]
    fi = f - away * th
    # the counter's bottom sits between its far wall and the *outer* bottom, so
    # it always starts climbing to its (higher) crotch before the outer does --
    # otherwise the two bottom curves can cross when the bottom wall is thin
    i_pts = [tc, (fi, b + tb + (tc[1] - b - tb) * val(in_mid)), (fi + (o_pts[2][0] - fi) * val(bx_in), b + tb), bc]

    def chain(cb, pts, tens):
        names = ["top", "far", "bottom", "end"]
        cb.move(pts[0], names[0])
        for k in range(1, 4):
            C = hobby_cubic(pts[k - 1], math.radians(dirs[k - 1]), pts[k], math.radians(dirs[k]), tens, tens)
            cb.curve(tuple(C[1]), tuple(C[2]), tuple(C[3]), names[k])

    outer = builder.contour(prefix + "outer")
    inner = builder.contour(prefix + "inner")
    outer.group = inner.group = object()
    outer.kind = inner.kind = "lobe"
    chain(outer, o_pts, val(tension))
    inside = ex - away * 6                                      # a little way into the stem
    outer.line((inside, bo[1])).line((inside, to[1])).close()
    chain(inner, i_pts, val(in_tension))
    inner.close()                                               # closes straight up the stem edge
    return outer, inner
