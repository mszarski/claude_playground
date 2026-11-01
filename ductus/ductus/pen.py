"""The pen and stroke expansion (SPEC section 6).

Width at a point = direction model x width profile, split into left/right
halves about the skeleton.

Direction model: a nib held at ``angle`` draws its thinnest line when moving
along ``angle`` and its thickest perpendicular to it::

    w(theta) = hair + k * soft|sin(theta - angle)|

``hair = width * (1 - contrast)`` and ``k`` makes a vertical stroke exactly
``width``. ``soft|x|`` rounds off the crease at the thinnest direction so
the offset curve can be fitted with a fixed number of cubics.

Left/right split: the hairline part of the width is always centred; the
modulation (``w - hair``) is shared ``bias : 1 - bias`` between the left and
right of the direction of travel. Closed strokes use the pen's ``inside``
share for the inner side instead.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .geometry import Contour, bez, fit_cubics, tangents
from .params import val
from .skeleton import Stroke

# Fixed output segmentation: never adaptive (principle 4).
CURVE_SEGMENTS = 4       # output cubics per side per skeleton curve
SAMPLES = 16             # samples per output cubic used for fitting


class Pen:
    SOFTNESS = 0.12

    def __init__(self, angle=20, width=80, contrast=0.5, inside=0.5, name: str | None = None):
        """``inside``: for closed strokes (bowls), the share of the thick/thin
        modulation placed on the inner side (0.5 = symmetric)."""
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

    def widths(self, theta: np.ndarray, dangle=None) -> np.ndarray:
        """Width for travel directions ``theta`` (radians). ``dangle``: per-sample
        turn of the nib (radians) -- the pen angle varying along a stroke. The
        normalisation is per sample, so a vertical stroke stays exactly
        ``width`` whatever the nib's angle there."""
        a0, hair, _ = self._consts()
        w = val(self.width)
        e = self.SOFTNESS
        soft = lambda x: np.sqrt(x * x + e * e) - e
        theta = np.asarray(theta, float)
        a = a0 + (np.zeros_like(theta) if dangle is None else np.asarray(dangle, float))
        # a nib held near vertical can't keep a vertical stroke at `width`
        # (the normalisation divides by cos a): keep it within +-80 degrees
        a = np.clip(a, -math.radians(80), math.radians(80))
        c = np.maximum(np.abs(np.cos(a)), 1e-6)
        k = (w - hair) / c
        s = np.sin(theta - a)
        return hair + k * soft(s) * (c / np.maximum(soft(c), 1e-12))

    def width_at(self, direction_deg) -> float:
        """Stroke width for travel direction (degrees)."""
        return float(self.widths(np.array([math.radians(val(direction_deg))]))[0])

    @property
    def hairline(self) -> float:
        return val(self.width) * (1 - val(self.contrast))

    def half_widths(self, direction_deg, inside=None) -> tuple[float, float]:
        """(inner, outer) offsets for a closed stroke travelling in ``direction_deg``."""
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
    left: np.ndarray = None    # outline corner on the left of ``outward``
    right: np.ndarray = None   # ... and on the right


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
    profile: object = None
    g0: float = 0.0      # global stroke parameter at segment start / end
    g1: float = 1.0
    pa0: float = 0.0     # pen-angle delta (radians) at segment start / end
    pa1: float = 0.0
    arc: tuple = None    # (bezier t samples, arc-length fraction): position along the segment


def _offset(P: np.ndarray, t: np.ndarray, sd, side: float | None = None) -> np.ndarray:
    if not isinstance(sd, _Side):          # (P, t, pen, side) convenience form
        sd = _Side(sd, side, 0.5)
    t = np.asarray(t, float)
    T = tangents(P, t)
    theta = np.arctan2(T[:, 1], T[:, 0])
    # node deltas (pen angle, width) ease between nodes by arc length with a
    # smoothstep: their slope is 0 at every node, so the outline has no kink
    # there (linear blending made the width's slope jump at each node)
    u = np.interp(t, *sd.arc) if sd.arc is not None else t
    e = u * u * (3 - 2 * u)
    w = sd.pen.widths(theta, (sd.pa0 + (sd.pa1 - sd.pa0) * e) if (sd.pa0 or sd.pa1) else None)
    hair = sd.pen.hairline
    half = hair / 2 + (w - hair) * sd.frac
    if sd.profile is not None:
        # position along the stroke by arc length: the bezier parameter
        # doesn't move at constant speed, and using it made the profile's
        # slope jump at every node -- a visible kink in the outline (the s)
        half = half * sd.profile(sd.g0 + (sd.g1 - sd.g0) * u)
    half = half + (sd.dw0 + (sd.dw1 - sd.dw0) * e) / 2
    half = np.maximum(half, 0.25)   # never negative: the outline would turn inside out
    N = np.stack([-T[:, 1], T[:, 0]], axis=1)  # left normal
    return bez(P, t) + sd.sign * N * half[:, None]


def _offset_tangents(P, ts: np.ndarray, sd: "_Side", h=1e-4) -> np.ndarray:
    a, b = np.clip(ts - h, 0, 1), np.clip(ts + h, 0, 1)
    pts = _offset(P, np.concatenate([a, b]), sd)
    d = pts[len(ts):] - pts[:len(ts)]
    n = np.linalg.norm(d, axis=1)
    bad = n < 1e-12
    if bad.any():
        d[bad] = tangents(P, ts[bad]); n[bad] = 1.0
    return d / n[:, None]


def _offset_tangent(P, t: float, sd, side=None) -> np.ndarray:
    if not isinstance(sd, _Side):
        sd = _Side(sd, side, 0.5)
    return _offset_tangents(P, np.array([float(t)]), sd)[0]


ARC_SAMPLES = 64


def _arc_table(P: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """(t samples, arc-length fraction at each, total length) for one cubic."""
    ts = np.linspace(0.0, 1.0, ARC_SAMPLES + 1)
    pts = bez(P, ts)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    L = float(cum[-1])
    return ts, (cum / L if L > 1e-9 else ts), L


def _global_params(cubics) -> list[tuple[float, float]]:
    """Map each segment to a sub-range of the stroke's global position, by
    arc length (structure-independent: always the same sample count)."""
    L = [_arc_table(P)[2] for P, _ in cubics]
    total = sum(L) or 1.0
    out, acc = [], 0.0
    for l in L:
        out.append((acc / total, (acc + l) / total))
        acc += l
    return out


def _sides(stroke: Stroke, pen: Pen, cubics, sign: float, frac: float) -> list[_Side]:
    nodes = stroke.nodes
    g = _global_params(cubics)
    return [_Side(pen, sign, frac, nodes[i].dw, nodes[(i + 1) % len(nodes)].dw, stroke.profile, *g[i],
                  math.radians(nodes[i].dpa), math.radians(nodes[(i + 1) % len(nodes)].dpa),
                  _arc_table(cubics[i][0])[:2])
            for i in range(len(cubics))]


def _side_chain(cubics, stroke: Stroke, sides: list[_Side], suffix: str) -> Contour:
    """Offset chain along one side, in direction of travel."""
    chain = Contour()
    nodes = stroke.nodes
    chain.add(_offset(cubics[0][0], np.array([0.0]), sides[0])[0], "on", nodes[0].fullname + suffix)
    for i, (P, kind) in enumerate(cubics):
        name = nodes[(i + 1) % len(nodes)].fullname + suffix
        s_i = sides[i]
        if kind == "line" and stroke.profile is None:
            chain.add(_offset(P, np.array([1.0]), s_i)[0], "on", name)
            continue
        if kind == "line":   # a profiled line: still one segment, fitted as a cubic
            K, n = 1, SAMPLES
        else:
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


def stroke_fracs(stroke: Stroke, pen: Pen) -> tuple[float, float]:
    """(left share, right share) of the modulation."""
    if stroke.closed:
        b = val(stroke.inside if stroke.inside is not None else pen.inside)
        fl = b if stroke.is_ccw() else 1 - b       # left is inside for CCW travel
    else:
        fl = val(stroke.bias) if stroke.bias is not None else 0.5
    return fl, 1 - fl


def expand_stroke(stroke: Stroke, default_pen: Pen) -> Expanded:
    pen = stroke.pen or default_pen
    cubics = stroke.cubics()
    fl, fr = stroke_fracs(stroke, pen)
    L_sides = _sides(stroke, pen, cubics, +1.0, fl)
    R_sides = _sides(stroke, pen, cubics, -1.0, fr)
    left = _side_chain(cubics, stroke, L_sides, ".l")
    right = _side_chain(cubics, stroke, R_sides, ".r")
    if stroke.closed:
        for ch in (left, right):  # last on-point duplicates the first
            ch.pts.pop(); ch.types.pop(); ch.names.pop()
        return Expanded([left, right], [])
    # open: left forward, flat end cap, right backward, flat start cap
    c = left.copy()
    rr = Contour(right.pts[::-1], right.types[::-1], right.names[::-1])
    c.pts += rr.pts; c.types += rr.types; c.names += rr.names
    ends = []
    for which, (P, _), node in (("start", cubics[0], stroke.nodes[0]), ("end", cubics[-1], stroke.nodes[-1])):
        t = 0.0 if which == "start" else 1.0
        T = tangents(P, np.array([t]))[0]
        si = 0 if which == "start" else -1
        lp = _offset(P, np.array([t]), L_sides[si])[0]
        rp = _offset(P, np.array([t]), R_sides[si])[0]
        outward = -T if which == "start" else T
        # corners relative to *outward*: at the start, travel-left is outward-right
        oleft, oright = (rp, lp) if which == "start" else (lp, rp)
        ends.append(StrokeEnd(node.fullname, node.pos.copy(), outward, float(np.linalg.norm(lp - rp)),
                              node.terminal, stroke.fullname, oleft, oright))
    return Expanded([c], ends)
