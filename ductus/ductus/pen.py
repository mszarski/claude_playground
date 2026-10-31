"""The broad-nib pen and stroke expansion (spec section 6).

Width model: a nib held at ``angle`` draws its thinnest line when moving
along ``angle`` and its thickest perpendicular to it::

    w(theta) = hair + k * soft|sin(theta - angle)|

``soft|x|`` is ``sqrt(x^2 + e^2) - e`` (renormalised): a real nib has a
crease at its thinnest direction, but that kink makes the offset curve hard
to fit with a fixed number of cubics, so it is rounded off slightly.

``hair = width * (1 - contrast)``, and ``k`` is chosen so that a *vertical*
stroke is exactly ``width`` -- i.e. the declared ``stem`` is the vertical
stem weight, which is what the stem lint measures. Diagonals perpendicular
to the nib come out slightly heavier, as with a real broad nib.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .geometry import Contour, bez, fit_cubic, fit_cubics, tangents
from .params import val
from .skeleton import Stroke

# Fixed output segmentation: never adaptive (principle 4).
CURVE_SEGMENTS = 4       # output cubics per side per skeleton curve
SAMPLES = 16             # samples per output cubic used for fitting


class Pen:
    def __init__(self, angle=20, width=80, contrast=0.5, inside=0.5, name: str | None = None):
        """``inside``: for closed strokes (bowls), the share of the thick/thin
        modulation placed on the inner side. 0.5 = symmetric about the
        skeleton; higher keeps the outer contour rounder and puts the stress
        into the counter, as type designers usually draw it."""
        self.angle = angle
        self.width = width
        self.contrast = contrast
        self.inside = inside
        self.name = name

    def _consts(self):
        a = math.radians(val(self.angle))
        w = val(self.width)
        hair = w * (1 - val(self.contrast))
        k = (w - hair) / max(abs(math.cos(a)), 1e-6)
        return a, hair, k

    SOFTNESS = 0.12

    def widths(self, theta: np.ndarray) -> np.ndarray:
        a, hair, k = self._consts()
        e = self.SOFTNESS
        soft = lambda x: np.sqrt(x * x + e * e) - e
        s = np.sin(np.asarray(theta) - a)
        # renormalise so a vertical stroke (|s| = cos a) is still exactly `width`
        c = abs(math.cos(a))
        return hair + k * soft(s) * (c / soft(c) if c > 1e-6 else 1.0)

    def width_at(self, direction_deg) -> float:
        """Stroke width for travel direction (degrees). Usable in parts,
        e.g. to keep the *outer* edge of a curve on a metric line."""
        return float(self.widths(np.array([math.radians(val(direction_deg))]))[0])

    @property
    def hairline(self) -> float:
        return val(self.width) * (1 - val(self.contrast))

    def half_widths(self, direction_deg, inside=None) -> tuple[float, float]:
        """(inner, outer) offsets for a closed stroke travelling in
        ``direction_deg`` -- use the outer one to put a bowl's edge on a line."""
        b = val(self.inside if inside is None else inside)
        w = self.width_at(direction_deg)
        hair = self.hairline
        return hair / 2 + (w - hair) * b, hair / 2 + (w - hair) * (1 - b)


@dataclass
class StrokeEnd:
    name: str            # full name of the skeleton node
    pos: np.ndarray
    outward: np.ndarray  # unit vector pointing out of the stroke
    width: float
    kind: object         # the node's ``terminal`` tag
    stroke: str


@dataclass
class Expanded:
    contours: list[Contour]
    ends: list[StrokeEnd]


@dataclass
class _Side:
    """How one side of one skeleton segment is offset."""
    pen: Pen
    sign: float          # +1 left, -1 right
    frac: float          # share of (width - hairline) on this side
    dw0: float = 0.0     # width delta at segment start / end
    dw1: float = 0.0


def _offset(P: np.ndarray, t: np.ndarray, sd: "_Side", side: float | None = None) -> np.ndarray:
    if not isinstance(sd, _Side):          # (P, t, pen, side) convenience form
        sd = _Side(sd, side, 0.5)
    t = np.asarray(t, float)
    T = tangents(P, t)
    theta = np.arctan2(T[:, 1], T[:, 0])
    w = sd.pen.widths(theta)
    hair = sd.pen.hairline
    half = hair / 2 + (w - hair) * sd.frac + (sd.dw0 + (sd.dw1 - sd.dw0) * t) / 2
    N = np.stack([-T[:, 1], T[:, 0]], axis=1)  # left normal
    return bez(P, t) + sd.sign * N * half[:, None]


def _offset_tangent(P, t: float, sd, side=None, h=1e-4) -> np.ndarray:
    if not isinstance(sd, _Side):
        sd = _Side(sd, side, 0.5)
    a, b = max(t - h, 0.0), min(t + h, 1.0)
    d = _offset(P, np.array([b]), sd)[0] - _offset(P, np.array([a]), sd)[0]
    n = np.linalg.norm(d)
    if n < 1e-12:
        d = tangents(P, np.array([t]))[0]
        n = 1.0
    return d / n


def _offset_tangents(P, ts: np.ndarray, sd: "_Side", h=1e-4) -> np.ndarray:
    """Unit tangents of the offset curve at ``ts`` (one batched evaluation)."""
    a, b = np.clip(ts - h, 0, 1), np.clip(ts + h, 0, 1)
    pts = _offset(P, np.concatenate([a, b]), sd)
    d = pts[len(ts):] - pts[:len(ts)]
    n = np.linalg.norm(d, axis=1)
    bad = n < 1e-12
    if bad.any():
        d[bad] = tangents(P, ts[bad]); n[bad] = 1.0
    return d / n[:, None]


def _side_chain(cubics, stroke: Stroke, pen: Pen, side: float, suffix: str, frac: float) -> Contour:
    """Offset chain along one side, in direction of travel."""
    chain = Contour()
    nodes = stroke.nodes

    def sd(i):
        return _Side(pen, side, frac, nodes[i].dw, nodes[(i + 1) % len(nodes)].dw)

    chain.add(_offset(cubics[0][0], np.array([0.0]), sd(0))[0], "on", nodes[0].fullname + suffix)
    for i, (P, kind) in enumerate(cubics):
        end_node = nodes[(i + 1) % len(nodes)]
        name = end_node.fullname + suffix
        s_i = sd(i)
        if kind == "line":
            chain.add(_offset(P, np.array([1.0]), s_i)[0], "on", name)
            continue
        K, n = CURVE_SEGMENTS, SAMPLES
        ts = np.linspace(0.0, 1.0, K * n + 1)
        allQ = _offset(P, ts, s_i)
        allQ[0] = chain.pts[-1]  # exact continuity with the previous point
        Q = np.stack([allQ[k * n:(k + 1) * n + 1] for k in range(K)])
        T = _offset_tangents(P, np.linspace(0.0, 1.0, K + 1), s_i)
        Cs = fit_cubics(Q, T[:-1], T[1:])
        for k, C in enumerate(Cs):
            chain.add(C[1], "off")
            chain.add(C[2], "off")
            chain.add(C[3], "on", name if k == K - 1 else None)
    return chain


def expand_stroke(stroke: Stroke, default_pen: Pen) -> Expanded:
    pen = stroke.pen or default_pen
    cubics = stroke.cubics()
    fl = 0.5
    if stroke.closed:
        b = val(stroke.inside if stroke.inside is not None else pen.inside)
        fl = b if stroke.is_ccw() else 1 - b       # left is inside for CCW travel
    left = _side_chain(cubics, stroke, pen, +1.0, ".l", fl)
    right = _side_chain(cubics, stroke, pen, -1.0, ".r", 1 - fl)
    if stroke.closed:
        for ch in (left, right):  # last on-point duplicates the first
            ch.pts.pop(); ch.types.pop(); ch.names.pop()
        return Expanded([left, right], [])
    # open: left forward, flat end cap, right backward, flat start cap
    c = left.copy()
    rr = Contour(right.pts[::-1], right.types[::-1], right.names[::-1])
    c.pts += rr.pts; c.types += rr.types; c.names += rr.names
    ends = []
    for which, (P, _), node, t in (("start", cubics[0], stroke.nodes[0], 0.0),
                                   ("end", cubics[-1], stroke.nodes[-1], 1.0)):
        T = tangents(P, np.array([t]))[0]
        w = float(pen.widths(np.array([math.atan2(T[1], T[0])]))[0]) + node.dw
        ends.append(StrokeEnd(node.fullname, node.pos.copy(), -T if which == "start" else T,
                              w, node.terminal, stroke.fullname))
    return Expanded([c], ends)
