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

from .geometry import Contour, bez, fit_cubic, tangents
from .params import val
from .skeleton import Stroke

# Fixed output segmentation: never adaptive (principle 4).
CURVE_SEGMENTS = 4       # output cubics per side per skeleton curve
SAMPLES = 16             # samples per output cubic used for fitting


class Pen:
    def __init__(self, angle=20, width=80, contrast=0.5, name: str | None = None):
        self.angle = angle
        self.width = width
        self.contrast = contrast
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


def _offset(P: np.ndarray, t: np.ndarray, pen: Pen, side: float) -> np.ndarray:
    T = tangents(P, t)
    theta = np.arctan2(T[:, 1], T[:, 0])
    w = pen.widths(theta)
    N = np.stack([-T[:, 1], T[:, 0]], axis=1)  # left normal
    return bez(P, t) + side * N * (w / 2)[:, None]


def _offset_tangent(P, t: float, pen, side, h=1e-4) -> np.ndarray:
    a, b = max(t - h, 0.0), min(t + h, 1.0)
    d = _offset(P, np.array([b]), pen, side)[0] - _offset(P, np.array([a]), pen, side)[0]
    n = np.linalg.norm(d)
    if n < 1e-12:
        d = tangents(P, np.array([t]))[0]
        n = 1.0
    return d / n


def _side_chain(cubics, stroke: Stroke, pen: Pen, side: float, suffix: str) -> Contour:
    """Offset chain along one side, in direction of travel."""
    chain = Contour()
    P0 = cubics[0][0]
    chain.add(_offset(P0, np.array([0.0]), pen, side)[0], "on", stroke.nodes[0].fullname + suffix)
    for i, (P, kind) in enumerate(cubics):
        end_node = stroke.nodes[(i + 1) % len(stroke.nodes)]
        name = end_node.fullname + suffix
        if kind == "line":
            chain.add(_offset(P, np.array([1.0]), pen, side)[0], "on", name)
            continue
        for k in range(CURVE_SEGMENTS):
            ta, tb = k / CURVE_SEGMENTS, (k + 1) / CURVE_SEGMENTS
            ts = np.linspace(ta, tb, SAMPLES + 1)
            Q = _offset(P, ts, pen, side)
            Q[0] = chain.pts[-1]  # exact continuity with previous point
            C = fit_cubic(Q, _offset_tangent(P, ta, pen, side), _offset_tangent(P, tb, pen, side))
            chain.add(C[1], "off")
            chain.add(C[2], "off")
            chain.add(C[3], "on", name if k == CURVE_SEGMENTS - 1 else None)
    return chain


def expand_stroke(stroke: Stroke, default_pen: Pen) -> Expanded:
    pen = stroke.pen or default_pen
    cubics = stroke.cubics()
    left = _side_chain(cubics, stroke, pen, +1.0, ".l")
    right = _side_chain(cubics, stroke, pen, -1.0, ".r")
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
        w = float(pen.widths(np.array([math.atan2(T[1], T[0])]))[0])
        ends.append(StrokeEnd(node.fullname, node.pos.copy(), -T if which == "start" else T,
                              w, node.terminal, stroke.fullname))
    return Expanded([c], ends)
