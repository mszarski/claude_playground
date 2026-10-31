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
