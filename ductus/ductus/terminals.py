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
from .handles import handle
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
        # Always emit the fill and the cut, even when the cut line already
        # matches the stroke's cap (they are then degenerate): whether they
        # exist must not depend on parameter values (principle 4).
        t = Terminal()
        # the gap between the stroke's own cap and the cut line: a cap corner
        # that pokes past the line is clamped onto it, so the fill is never a
        # bow-tie (whose orientation -- and so point order -- would be arbitrary)
        def inside(P, on_line):
            return P if float(np.dot(np.asarray(P, float) - self.pos, self.d)) <= 0 else on_line
        c = Contour()
        for P, nm in ((inside(self.end.left, self.A), "cap.a"), (self.A, "cut.a"),
                      (self.B, "cut.b"), (inside(self.end.right, self.B), "cap.b")):
            c.add(P, "on", f"{name}.{nm}")
        t.add.append(c.reversed())          # frames are left-handed: reverse -> CCW in world
        # cut only what the stroke's own cap pushes past the line: a little
        # wider than the stroke, as deep as a tilted cap can protrude. A big
        # box would slice neighbouring parts (it did: the a's hood cut its bowl).
        m = 1.0
        # as deep as the tilted cap actually pokes past the line (+1): ~0 when
        # the cut is aligned with the cap, so it never reaches neighbours
        # above the end (it used to delete the i's dot)
        cosang = min(1.0, abs(float(np.dot(self.u, self.d))))
        tan = math.sqrt(max(0.0, 1 - cosang * cosang)) / max(cosang, 1e-3)
        D = 0.5 * self.end.width * tan + 1.0
        # across the line: as far as the cap's own corners reach (on a tilted
        # cut they sit wider than their projections A/B), plus a hair -- no
        # further, or it slices whatever the stroke runs into (the b's bowl)
        ec = [float(np.dot(np.asarray(P, float) - self.pos, self.e)) for P in (self.end.left, self.end.right)]
        lo, hi = min(self.e_lo, *ec), max(self.e_hi, *ec)
        t.cut.append(poly([self.w(lo - m, 0), self.w(hi + m, 0),
                           self.w(hi + m, D), self.w(lo - m, D)]))
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
            L_expr, B_expr = sv.get("length", length), sv.get("bracket", bracket)
            L = val(L_expr); B = val(B_expr)
            T = val(sv.get("tip", tip)); cv = sv.get("curve", curve)
            ca, cb = val(cv[0]), val(cv[1])
            sl = math.radians(val(sv.get("slope", slope)))
            e0 = F.edge(sg)
            h = L * cb      # handle at the tip, pointing back towards the stem, rising by ``slope``
            if L > 0 or B > 0:
                handle(L_expr, F.w(e0 + sg * L, 0), F.e * sg, label=f"serif length")
                handle(B_expr, F.w(e0, -(T + B)), -F.d, label=f"serif bracket")
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
        t.add.append(c.reversed())          # built CCW in the (left-handed) frame -> CCW in world
        return t
    style.__name__ = "bracketed"
    return style


def wedge(side="left", length=60, rise=50, drop=40, hollow=0.0, axis="vertical",
          tip=0.0, flat=0.0, bracket=0.0, corner=20.0, ca=0.7, cb=0.5, sweep=0.0):
    """Flag/wedge serif (head of n, spur of a): the cut line slopes from the
    far corner down to a tip ``length`` beyond the ``side`` edge and ``rise``
    below the end; the underside returns to the stem ``drop`` lower.
    ``hollow`` (0..1) curves the underside inwards.

    Finer shape (all degenerate at 0, so the point structure never changes):
    ``tip`` -- the tip is a short flat edge this tall, not a point;
    ``flat`` -- the top runs flat this far in from the far corner before
    sloping down; ``bracket`` (0..1) blends the underside from the
    ``hollow`` curve to a bracket: nearly straight from the tip towards the
    stem edge ``corner`` below the tip, then a fillet into the stem (``ca``,
    ``cb`` = handle lengths from tip and from stem, as fractions of the way
    to that corner); ``sweep`` -- the sloped top edge is a curve sagging
    this far below the straight line at its middle (a concave flank, like
    the t's), straight at 0.
    """
    def style(end):
        F = Frame(end, axis)
        t = F.flat(end.name)
        sg = F.sigma(side)
        L, R, D, H = val(length), val(rise), val(drop), val(hollow)
        T, FL, K = max(val(tip), 0.0), max(val(flat), 0.0), min(max(val(bracket), 0.0), 1.0)
        e_side, e_far = F.edge(sg), F.edge(-sg)
        tip_t = (e_side + sg * L, -R)
        tip_b = (tip_t[0], -R - T)
        handle(length, F.w(*tip_t), F.e * sg, label="wedge length")
        handle(rise, F.w(*tip_t), -F.d, label="wedge rise")
        under = (e_side, -R - T - D)
        nm = f"{end.name}.wedge"
        c = Contour()
        c.add(F.w(e_far, 0), "on", f"{nm}.far")
        c.add(F.w(e_far + sg * FL, 0), "on", f"{nm}.flat")
        # the sloped top edge: a cubic, straight when sweep is 0 (handles on
        # the line), sagging by `sweep` at its middle otherwise (4/3: a
        # cubic's midpoint moves 3/4 as far as its handles)
        top0 = np.array([e_far + sg * FL, 0.0]); top1 = np.array(tip_t)
        # no deeper than 40% of the rise: more and the edge sags past the
        # tip and the wedge folds over itself (its structure would change)
        sw = np.array([0.0, -max(-0.4 * R, min(val(sweep), 0.4 * R)) * 4 / 3])
        _curve(c, F.w(*(top0 + (top1 - top0) / 3 + sw)), F.w(*(top0 + (top1 - top0) * 2 / 3 + sw)),
               F.w(*tip_t), f"{nm}.tip")
        c.add(F.w(*tip_b), "on", f"{nm}.tip_b")
        # underside, hollow curve (the original)
        mid = ((tip_b[0] + under[0]) / 2, (tip_b[1] + under[1]) / 2)
        sag = (sg * H * L * 0.35, H * D * 0.35)          # pulls the underside towards the stem corner
        h1 = np.array([tip_b[0] + (mid[0] - tip_b[0]) * 0.66 - sag[0], tip_b[1] + (mid[1] - tip_b[1]) * 0.66 + sag[1]])
        h2 = np.array([under[0] + (mid[0] - under[0]) * 0.66 - sag[0], under[1] + (mid[1] - under[1]) * 0.66 + sag[1]])
        # underside as a bracket: towards the stem edge `corner` below the tip, then a fillet
        C = np.array([e_side, tip_b[1] - val(corner)])
        b1 = np.array(tip_b) + val(ca) * (C - np.array(tip_b))
        b2 = np.array(under) + val(cb) * (C - np.array(under))
        h1, h2 = h1 * (1 - K) + b1 * K, h2 * (1 - K) + b2 * K
        _curve(c, F.w(*h1), F.w(*h2), F.w(*under), f"{nm}.under")
        c.add(F.w(e_far, -R - T - D), "on")
        t.add.append(c if sg > 0 else c.reversed())   # CCW in world for either side
        m = 0.5 * max(end.width, 10)                  # bounded, like Frame.flat
        # only the stem's own corner above the sloped line goes: the cut
        # reaches just past the end (d = 0), not into whatever sits above it
        # (it used to delete the i's dot)
        Dc = 0.25 * max(end.width, 20)
        cut = Contour()
        cut.add(F.w(e_far + -sg * 1, 0), "on")
        cut.add(F.w(e_far + sg * FL, 0), "on")
        _curve(cut, F.w(*(top0 + (top1 - top0) / 3 + sw)), F.w(*(top0 + (top1 - top0) * 2 / 3 + sw)), F.w(*tip_t))
        for P in (F.w(tip_t[0] + sg * m, -R), F.w(tip_t[0] + sg * m, Dc), F.w(e_far - sg * m, Dc),
                  F.w(e_far - sg * m, 0)):
            cut.add(P, "on")
        t.cut.append(cut)
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
        handle(length, F.w(e0 + sg * L, 0), F.e * sg, label="beak length")
        b = poly([F.w(e0 - sg * 2, 0), F.w(e0 + sg * L, 0), F.w(e0, -D), F.w(e0 - sg * 2, -D)],
                 [f"{end.name}.beak.base", f"{end.name}.beak.tip", f"{end.name}.beak.back", None])
        t.add.append(b if sg > 0 else b.reversed())   # CCW in world for either side
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
        handle(radius, (cx, cy + r), (0, 1), label="ball radius")
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
