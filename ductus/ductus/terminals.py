"""Terminal library (SPEC 5.5): shapes attached to tagged stroke ends.

Parts tag stroke ends (``terminal="foot"``, ``"head"``, ``"spur"``,
``"beak"`` ...); a font's :class:`TerminalSet` maps tags to styles, so
swapping the set restyles the whole font. A style is a callable
``style(end) -> Terminal`` returning contours to **add** and to **cut**.

Every style works in a frame at the stroke end: ``d`` points out of the
stroke (or along a world axis, ``axis="vertical"``/``"horizontal"``/an angle
in degrees, which turns the end into a flat cut along that axis), and ``e``
runs across it. The cut line passes through the end point. Serif sides are
given in *world* terms (``left``/``right`` for vertical axes, ``below``/
``above`` for horizontal ones).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .geometry import Contour
from .params import val


@dataclass
class Terminal:
    add: list[Contour] = field(default_factory=list)
    cut: list[Contour] = field(default_factory=list)


class Frame:
    def __init__(self, end, axis=None):
        u = np.asarray(end.outward, float)
        if axis is None:
            d = u
        elif axis == "vertical":
            d = np.array([0.0, 1.0 if u[1] >= 0 else -1.0])
        elif axis == "horizontal":
            d = np.array([1.0 if u[0] >= 0 else -1.0, 0.0])
        else:
            a = math.radians(val(axis))
            d = np.array([math.cos(a), math.sin(a)])
            if np.dot(d, u) < 0:
                d = -d
        # condition: a cut almost parallel to the stroke would project the
        # corners far away -- tilt it towards the stroke direction
        MIN = 0.3
        if abs(float(np.dot(u, d))) < MIN:
            perp = d - u * float(np.dot(u, d))
            perp /= max(np.linalg.norm(perp), 1e-9)
            d = u * MIN + perp * math.sqrt(1 - MIN * MIN)
        self.end, self.pos, self.d, self.u = end, np.asarray(end.pos, float), d, u
        self.e = np.array([-d[1], d[0]])
        self.aligned = axis is None or abs(np.dot(d, u)) > 1 - 1e-9
        ud = float(np.dot(u, d))

        def on_line(P):
            P = np.asarray(P, float)
            if abs(ud) < 1e-6:
                return P
            return P - u * (np.dot(P - self.pos, d) / ud)

        self.A, self.B = on_line(end.left), on_line(end.right)     # left / right of outward
        ea, eb = float(np.dot(self.A - self.pos, self.e)), float(np.dot(self.B - self.pos, self.e))
        self.e_hi, self.e_lo = max(ea, eb), min(ea, eb)            # extents across the cut line

    def w(self, ec, dc) -> np.ndarray:
        return self.pos + self.e * ec + self.d * dc

    def edge(self, sigma: int) -> float:
        return self.e_hi if sigma > 0 else self.e_lo

    def sigma(self, side: str) -> int:
        """World side -> sign along e."""
        vec = {"left": (-1, 0), "right": (1, 0), "below": (0, -1), "above": (0, 1),
               "outer": None}.get(side)
        if vec is None:
            raise ValueError(f"unknown side {side!r}")
        s = float(np.dot(self.e, vec))
        return 1 if s >= 0 else -1

    def flat(self, name: str) -> Terminal:
        """Fill the gap between the stroke's own cap and the cut line, and
        cut everything beyond the line. Nothing to do if already aligned."""
        t = Terminal()
        if self.aligned:
            return t
        c = Contour()
        for P, nm in ((self.end.left, "cap.a"), (self.end.right, "cap.b"), (self.B, "cut.b"), (self.A, "cut.a")):
            c.add(P, "on", f"{name}.{nm}")
        t.add.append(c)
        M = 3 * max(self.end.width, 40)
        t.cut.append(poly([self.w(self.e_lo - M, 0), self.w(self.e_hi + M, 0),
                           self.w(self.e_hi + M, M), self.w(self.e_lo - M, M)]))
        return t


def poly(points, names=None) -> Contour:
    c = Contour()
    for i, p in enumerate(points):
        c.add(p, "on", names[i] if names else None)
    return c


def _curve(c: Contour, c1, c2, p, name=None):
    c.add(c1, "off"); c.add(c2, "off"); c.add(p, "on", name)


# --------------------------------------------------------------------------
# styles

def none(end):
    return Terminal()


def flat(axis="vertical"):
    """A straight cut along an axis (or angle in degrees)."""
    def style(end):
        return Frame(end, axis).flat(end.name)
    style.__name__ = "flat"
    return style


def bracketed(length=40, bracket=30, tip=4, curve=(0.6, 0.6), cup=0, axis="vertical", slope=0, **sides):
    """Bracketed serif. Per-side settings override the shared ones:
    ``left=dict(length=.., bracket=.., tip=.., curve=(a, b))`` and
    ``right=...`` (or ``below``/``above`` for horizontal axes); a side set to
    ``0`` or ``None`` has no serif. ``curve`` = (handle along the stem,
    handle along the serif) as fractions; ``cup`` sags the outer edge;
    ``slope`` (degrees) tilts the serif's top edge where it meets the tip, so
    the serif can taper wedge-like to a sharp point."""
    def style(end):
        F = Frame(end, axis)
        t = F.flat(end.name)
        spec = {}
        for side, sv in sides.items():
            spec[F.sigma(side)] = sv
        c = Contour()
        pts = {}
        for sg in (1, -1):
            sv = spec.get(sg, {})
            if sv is None or sv == 0:
                sv = {"length": 0, "bracket": 0, "tip": 0}
            L = val(sv.get("length", length)); B = val(sv.get("bracket", bracket))
            T = val(sv.get("tip", tip)); cv = sv.get("curve", curve)
            ca, cb = val(cv[0]), val(cv[1])
            sl = math.radians(val(sv.get("slope", slope)))
            e0 = F.edge(sg)
            h = L * cb      # handle at the tip, pointing back towards the stem, rising by ``slope``
            pts[sg] = dict(S0=(e0, -(T + B)), c1=(e0, -(T + B) + B * ca),
                           c2=(e0 + sg * L - sg * h * math.cos(sl), -T - h * math.sin(sl)),
                           S1=(e0 + sg * L, -T), S2=(e0 + sg * L, 0.0))
        P, N = pts[1], pts[-1]
        nm = f"{end.name}.serif"
        cupv = val(cup)
        c.add(F.w(*P["S0"]), "on", f"{nm}.p.bracket")
        _curve(c, F.w(*P["c1"]), F.w(*P["c2"]), F.w(*P["S1"]), f"{nm}.p.top")
        c.add(F.w(*P["S2"]), "on", f"{nm}.p.tip")
        span = P["S2"][0] - N["S2"][0]
        _curve(c, F.w(P["S2"][0] - span / 3, -cupv), F.w(N["S2"][0] + span / 3, -cupv), F.w(*N["S2"]), f"{nm}.n.tip")
        c.add(F.w(*N["S1"]), "on", f"{nm}.n.top")
        _curve(c, F.w(*N["c2"]), F.w(*N["c1"]), F.w(*N["S0"]), f"{nm}.n.bracket")
        t.add.append(c)
        return t
    style.__name__ = "bracketed"
    return style


def wedge(side="left", length=60, rise=50, drop=40, hollow=0.0, axis="vertical"):
    """Flag/wedge serif (head of n, spur of a): the cut line slopes from the
    far corner down to a tip ``length`` beyond the ``side`` edge and ``rise``
    below the end; the underside returns to the stem ``drop`` lower.
    ``hollow`` (0..1) curves the underside inwards."""
    def style(end):
        F = Frame(end, axis)
        t = F.flat(end.name)
        sg = F.sigma(side)
        L, R, D, H = val(length), val(rise), val(drop), val(hollow)
        e_side, e_far = F.edge(sg), F.edge(-sg)
        tip = (e_side + sg * L, -R)
        under = (e_side, -R - D)
        nm = f"{end.name}.wedge"
        c = Contour()
        c.add(F.w(e_far, 0), "on", f"{nm}.far")
        c.add(F.w(*tip), "on", f"{nm}.tip")
        mid = ((tip[0] + under[0]) / 2, (tip[1] + under[1]) / 2)
        sag = (sg * H * L * 0.35, H * D * 0.35)          # pulls the underside towards the stem corner
        _curve(c, F.w(tip[0] + (mid[0] - tip[0]) * 0.66 - sag[0], tip[1] + (mid[1] - tip[1]) * 0.66 + sag[1]),
               F.w(under[0] + (mid[0] - under[0]) * 0.66 - sag[0], under[1] + (mid[1] - under[1]) * 0.66 + sag[1]),
               F.w(*under), f"{nm}.under")
        c.add(F.w(e_far, -R - D), "on")
        t.add.append(c)
        M = 3 * max(end.width, 40) + L
        t.cut.append(poly([F.w(e_far + -sg * 1, 0), F.w(*tip), F.w(tip[0] + sg * M, -R),
                           F.w(tip[0] + sg * M, M), F.w(e_far - sg * M, M), F.w(e_far - sg * M, 0)]))
        return t
    style.__name__ = "wedge"
    return style


def beak(side="above", length=30, depth=60, axis="horizontal"):
    """Flat cut plus a pointed beak continuing the cut line ``length`` past
    the ``side`` edge, joining the stroke edge ``depth`` back (s, c, e)."""
    def style(end):
        F = Frame(end, axis)
        t = F.flat(end.name)
        sg = F.sigma(side)
        e0 = F.edge(sg)
        L, D = val(length), val(depth)
        t.add.append(poly([F.w(e0 - sg * 2, 0), F.w(e0 + sg * L, 0), F.w(e0, -D), F.w(e0 - sg * 2, -D)],
                          [f"{end.name}.beak.base", f"{end.name}.beak.tip", f"{end.name}.beak.back", None]))
        return t
    style.__name__ = "beak"
    return style


def ball(radius=40, side="right", drop=0.0, inset=0.0, axis=None):
    """Round ball terminal flush with the ``side`` edge of the stroke end.
    ``drop`` moves it back along the stroke, ``inset`` across it."""
    def style(end):
        F = Frame(end, axis)
        sg = F.sigma(side) if side in ("left", "right", "above", "below") else 1
        r = val(radius)
        ec = F.edge(sg) - sg * (r + val(inset))
        dc = -r * 0.85 - val(drop)
        cx, cy = F.w(ec, dc)
        k = 0.5523 * r
        c = Contour()
        nm = f"{end.name}.ball"
        c.add((cx, cy - r), "on", f"{nm}.bottom")
        _curve(c, (cx + k, cy - r), (cx + r, cy - k), (cx + r, cy), f"{nm}.right")
        _curve(c, (cx + r, cy + k), (cx + k, cy + r), (cx, cy + r), f"{nm}.top")
        _curve(c, (cx - k, cy + r), (cx - r, cy + k), (cx - r, cy), f"{nm}.left")
        c.add((cx - r, cy - k), "off"); c.add((cx - k, cy - r), "off")
        return Terminal(add=[c])
    style.__name__ = "ball"
    return style


def slab(length=None, thickness=None, where=None):
    """Rectangular slab centred on the end (milestone 1)."""
    def style(end):
        if where is not None and end.kind not in where:
            return Terminal()
        from .parts import current_font
        pen = current_font().pen
        L = val(length) if length is not None else val(pen.width) * 0.45
        T = val(thickness) if thickness is not None else pen.hairline * 0.75
        d = end.outward
        n = np.array([-d[1], d[0]])
        half = end.width / 2 + L
        p = end.pos
        return Terminal(add=[poly([p - n * half, p + n * half, p + n * half - d * T, p - n * half - d * T],
                                  [f"{end.name}.serif.{k}" for k in "abcd"])])
    style.__name__ = "slab"
    return style


class TerminalSet:
    """Maps end tags to styles: ``TerminalSet(default, foot=..., head=...)``.
    Untagged ends (``terminal=True``) use ``default``; unknown tags fall back
    to ``default`` as well."""

    def __init__(self, default=none, **by_tag):
        self.default = default
        self.by_tag = by_tag
        self.__name__ = "set"

    def style_for(self, tag):
        if isinstance(tag, str) and tag in self.by_tag:
            return self.by_tag[tag]
        return self.default

    def __call__(self, end):
        return self.style_for(end.kind)(end)


def as_terminal(result) -> Terminal:
    if isinstance(result, Terminal):
        return result
    return Terminal(add=list(result or []))


class serifs:
    none = staticmethod(none)
    slab = staticmethod(slab)
    flat = staticmethod(flat)
    bracketed = staticmethod(bracketed)
    wedge = staticmethod(wedge)
    beak = staticmethod(beak)
    ball = staticmethod(ball)
    Set = TerminalSet
