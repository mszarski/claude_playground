"""Bezier math: evaluation, Hobby-style handles, least-squares fitting, and
the point-list contour representation shared by the whole pipeline."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

SQRT2 = math.sqrt(2.0)
SQRT5 = math.sqrt(5.0)


# --------------------------------------------------------------------------
# cubic Bezier evaluation (vectorised over t)

def bez(P: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Points on cubic ``P`` (4x2) at parameters ``t`` (n,) -> (n,2)."""
    t = np.asarray(t, dtype=float)[:, None]
    mt = 1.0 - t
    return (mt ** 3) * P[0] + 3 * (mt ** 2) * t * P[1] + 3 * mt * (t ** 2) * P[2] + (t ** 3) * P[3]


def bez_d1(P: np.ndarray, t: np.ndarray) -> np.ndarray:
    t = np.asarray(t, dtype=float)[:, None]
    mt = 1.0 - t
    return 3 * (mt ** 2) * (P[1] - P[0]) + 6 * mt * t * (P[2] - P[1]) + 3 * (t ** 2) * (P[3] - P[2])


def bez_d2(P: np.ndarray, t: np.ndarray) -> np.ndarray:
    t = np.asarray(t, dtype=float)[:, None]
    return 6 * (1 - t) * (P[2] - 2 * P[1] + P[0]) + 6 * t * (P[3] - 2 * P[2] + P[1])


def tangents(P: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Unit tangents; falls back to the second derivative / chord where the
    first derivative vanishes (collapsed handles)."""
    d = bez_d1(P, t)
    n = np.linalg.norm(d, axis=1)
    bad = n < 1e-9
    if bad.any():
        d2 = bez_d2(P, t)
        # sign: near t=0 the curve heads along +d2, near t=1 along -d2... use chord as safe default
        chord = P[3] - P[0]
        d[bad] = np.where(np.linalg.norm(d2[bad], axis=1, keepdims=True) > 1e-9,
                          d2[bad] * np.where(np.asarray(t)[bad, None] < 0.5, 1, -1), chord)
        n = np.linalg.norm(d, axis=1)
    return d / n[:, None]


def line_cubic(a, b) -> np.ndarray:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return np.array([a, a + (b - a) / 3, a + 2 * (b - a) / 3, b])


# --------------------------------------------------------------------------
# Hobby's handle lengths (as in METAFONT), for curves through nodes with
# given directions.

def _rho(theta: float, phi: float) -> float:
    st, sp, ct, cp = math.sin(theta), math.sin(phi), math.cos(theta), math.cos(phi)
    num = 2 + SQRT2 * (st - sp / 16) * (sp - st / 16) * (ct - cp)
    den = 3 * (1 + 0.5 * (SQRT5 - 1) * ct + 0.5 * (3 - SQRT5) * cp)
    return num / den


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def hobby_cubic(p0, d0: float, p3, d1: float, t0: float = 1.0, t1: float = 1.0) -> np.ndarray:
    """Cubic from ``p0`` leaving at angle ``d0`` (radians) to ``p3`` arriving
    with direction ``d1``."""
    p0, p3 = np.asarray(p0, float), np.asarray(p3, float)
    chord = p3 - p0
    L = float(np.hypot(*chord))
    if L < 1e-9:
        return np.array([p0, p0, p3, p3])
    ca = math.atan2(chord[1], chord[0])
    theta = _wrap(d0 - ca)
    phi = _wrap(ca - d1)
    a = L * _rho(theta, phi) / t0
    b = L * _rho(phi, theta) / t1
    c1 = p0 + a * np.array([math.cos(d0), math.sin(d0)])
    c2 = p3 - b * np.array([math.cos(d1), math.sin(d1)])
    return np.array([p0, c1, c2, p3])


# --------------------------------------------------------------------------
# Least-squares cubic fit with fixed endpoints and fixed end tangents.
# Only the two handle lengths are free, so the output always has exactly one
# cubic -- the caller decides how many segments to fit (principle 4).

def fit_cubic(Q: np.ndarray, t0: np.ndarray, t1: np.ndarray, iterations: int = 2) -> np.ndarray:
    """Fit one cubic to points ``Q`` (n,2). ``t0`` is the unit tangent at the
    start (direction of travel), ``t1`` the unit tangent at the end (direction
    of travel). Returns 4x2 control points."""
    return fit_cubics(Q[None], np.asarray(t0)[None], np.asarray(t1)[None], iterations)[0]


def fit_cubics(Q: np.ndarray, t0: np.ndarray, t1: np.ndarray, iterations: int = 2) -> np.ndarray:
    """Batched :func:`fit_cubic`: ``Q`` (K,n,2), ``t0``/``t1`` (K,2) -> (K,4,2)."""
    Q = np.asarray(Q, float)
    P0, P3 = Q[:, 0], Q[:, -1]
    seg = np.linalg.norm(np.diff(Q, axis=1), axis=2)            # (K, n-1)
    total = seg.sum(axis=1)                                     # (K,)
    chord = np.linalg.norm(P3 - P0, axis=1)
    safe = np.where(total < 1e-9, 1.0, total)
    u = np.concatenate([np.zeros((len(Q), 1)), np.cumsum(seg, axis=1) / safe[:, None]], axis=1)
    fallback = (total / 3)[:, None]
    for it in range(iterations + 1):
        mu = 1 - u
        B0, B1, B2, B3 = mu ** 3, 3 * mu ** 2 * u, 3 * mu * u ** 2, u ** 3
        A1 = B1[..., None] * t0[:, None, :]
        A2 = -B2[..., None] * t1[:, None, :]
        R = Q - ((B0 + B1)[..., None] * P0[:, None] + (B2 + B3)[..., None] * P3[:, None])
        c00 = (A1 * A1).sum((1, 2)); c01 = (A1 * A2).sum((1, 2)); c11 = (A2 * A2).sum((1, 2))
        x0 = (A1 * R).sum((1, 2)); x1 = (A2 * R).sum((1, 2))
        det = c00 * c11 - c01 ** 2
        ok = np.abs(det) > 1e-12
        dsafe = np.where(ok, det, 1.0)
        a1 = np.where(ok, (x0 * c11 - x1 * c01) / dsafe, total / 3)
        a2 = np.where(ok, (c00 * x1 - c01 * x0) / dsafe, total / 3)
        eps = 1e-3 * np.maximum(chord, total)
        bad = (a1 < eps) | (a2 < eps)
        a1 = np.where(bad, total / 3, a1)
        a2 = np.where(bad, total / 3, a2)
        P = np.stack([P0, P0 + a1[:, None] * t0, P3 - a2[:, None] * t1, P3], axis=1)   # (K,4,2)
        if it < iterations:
            uu = u[..., None]; mu_ = 1 - uu
            Bp = mu_ ** 3 * P[:, None, 0] + 3 * mu_ ** 2 * uu * P[:, None, 1] + 3 * mu_ * uu ** 2 * P[:, None, 2] + uu ** 3 * P[:, None, 3]
            d1 = 3 * mu_ ** 2 * (P[:, None, 1] - P[:, None, 0]) + 6 * mu_ * uu * (P[:, None, 2] - P[:, None, 1]) + 3 * uu ** 2 * (P[:, None, 3] - P[:, None, 2])
            d2 = 6 * mu_ * (P[:, None, 2] - 2 * P[:, None, 1] + P[:, None, 0]) + 6 * uu * (P[:, None, 3] - 2 * P[:, None, 2] + P[:, None, 1])
            diff = Bp - Q
            num = (diff * d1).sum(2)
            den = (d1 * d1).sum(2) + (diff * d2).sum(2)
            okd = np.abs(den) > 1e-12
            u = u - np.where(okd, num / np.where(okd, den, 1.0), 0.0)
            u = np.clip(u, 0, 1)
            u[:, 0], u[:, -1] = 0.0, 1.0
            u = np.maximum.accumulate(u, axis=1)
    degenerate = total < 1e-9
    if degenerate.any():
        P[degenerate] = np.stack([P0, P0, P3, P3], axis=1)[degenerate]
    return P


def fit_error(P: np.ndarray, Q: np.ndarray) -> float:
    """Max distance from samples Q to a dense sampling of P (approximate)."""
    dense = bez(P, np.linspace(0, 1, 200))
    d = np.linalg.norm(Q[:, None, :] - dense[None, :, :], axis=2).min(axis=1)
    return float(d.max())


# --------------------------------------------------------------------------
# Contours: closed point lists, on/off curve flags, optional names.
# Cubic segments are encoded on, off, off, on; lines as on, on.

@dataclass
class Contour:
    pts: list[list[float]] = field(default_factory=list)
    types: list[str] = field(default_factory=list)
    names: list[str | None] = field(default_factory=list)

    def add(self, p, kind: str = "on", name: str | None = None):
        self.pts.append([float(p[0]), float(p[1])])
        self.types.append(kind)
        self.names.append(name)

    def __len__(self):
        return len(self.pts)

    def copy(self) -> "Contour":
        return Contour([list(p) for p in self.pts], list(self.types), list(self.names))

    def reversed(self) -> "Contour":
        idx = [0] + list(range(len(self.pts) - 1, 0, -1))
        return Contour([list(self.pts[i]) for i in idx], [self.types[i] for i in idx],
                       [self.names[i] for i in idx])

    def draw(self, pen):
        if not self.pts:
            return
        assert self.types[0] == "on", "contour must start on-curve"
        pen.moveTo(tuple(self.pts[0]))
        offs = []
        for p, k in zip(self.pts[1:], self.types[1:]):
            if k == "off":
                offs.append(tuple(p))
            elif offs:
                pen.curveTo(*offs, tuple(p))
                offs = []
            else:
                pen.lineTo(tuple(p))
        if offs:
            pen.curveTo(*offs, tuple(self.pts[0]))
        pen.closePath()

    def area(self) -> float:
        from fontTools.pens.areaPen import AreaPen
        ap = AreaPen()
        self.draw(ap)
        return ap.value  # > 0 for counter-clockwise (y up)

    def signature(self) -> str:
        return "".join("o" if k == "on" else "c" for k in self.types)

    def to_json(self) -> dict:
        return {"pts": [[round(x, 2), round(y, 2)] for x, y in self.pts],
                "types": self.types, "names": self.names}

    @classmethod
    def from_ops(cls, ops) -> list["Contour"]:
        """Build contours from RecordingPen ``value``."""
        out: list[Contour] = []
        cur: Contour | None = None
        for op, args in ops:
            if op == "moveTo":
                cur = Contour()
                cur.add(args[0])
            elif op == "lineTo":
                cur.add(args[0])
            elif op == "curveTo":
                assert len(args) == 3, "only cubic curves supported"
                cur.add(args[0], "off")
                cur.add(args[1], "off")
                cur.add(args[2])
            elif op == "qCurveTo":
                # pathops degree-reduces cubics that are really quadratics;
                # elevate them back (exact) so the font stays all-cubic
                from fontTools.pens.basePen import decomposeQuadraticSegment
                for q, p in decomposeQuadraticSegment(args):
                    p0 = np.array(cur.pts[-1])
                    q, p = np.array(q, float), np.array(p, float)
                    cur.add(p0 + 2 / 3 * (q - p0), "off")
                    cur.add(p + 2 / 3 * (q - p), "off")
                    cur.add(p)
            elif op in ("closePath", "endPath"):
                # drop explicit closing point duplicating the start
                if len(cur.pts) > 1 and cur.types[-1] == "on" and \
                        abs(cur.pts[-1][0] - cur.pts[0][0]) < 1e-9 and abs(cur.pts[-1][1] - cur.pts[0][1]) < 1e-9:
                    cur.pts.pop(); cur.types.pop(); cur.names.pop()
                out.append(cur)
                cur = None
            else:
                raise ValueError(f"unsupported pen op {op}")
        return out


def svg_path(contours: list[Contour]) -> str:
    from fontTools.pens.svgPathPen import SVGPathPen
    sp = SVGPathPen(None, ntos=lambda v: f"{v:.1f}".rstrip("0").rstrip("."))
    for c in contours:
        c.draw(sp)
    return sp.getCommands()


def flatten(c: Contour, steps: int = 16) -> np.ndarray:
    """Polygon approximation of a contour."""
    from fontTools.pens.recordingPen import RecordingPen
    rp = RecordingPen()
    c.draw(rp)
    poly = []
    last = None
    for op, args in rp.value:
        if op == "moveTo":
            last = np.array(args[0]); poly.append(last)
        elif op == "lineTo":
            last = np.array(args[0]); poly.append(last)
        elif op == "curveTo":
            P = np.array([last, *args])
            poly.extend(bez(P, np.linspace(0, 1, steps + 1)[1:]))
            last = P[3]
    return np.array(poly)


def point_in_polygon(p, poly: np.ndarray) -> bool:
    x, y = p
    xs, ys = poly[:, 0], poly[:, 1]
    xs2, ys2 = np.roll(xs, -1), np.roll(ys, -1)
    cond = (ys > y) != (ys2 > y)
    with np.errstate(divide="ignore", invalid="ignore"):
        xint = xs + (y - ys) * (xs2 - xs) / (ys2 - ys)
    return bool(np.sum(cond & (x < xint)) % 2)
