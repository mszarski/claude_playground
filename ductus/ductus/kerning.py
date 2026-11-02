"""Optical auto-kerning from glyph outlines.

For each glyph we sample its ink's horizontal extent at fixed heights (its
left and right *profiles*, in advance coordinates). For a pair L R set at
their advances, the white between them at height y is

    gap(y) = (advance_L - right_L(y)) + left_R(y)

wherever both glyphs have ink. A pair's *optical distance* is a power mean
of the gaps with a negative exponent -- dominated by the closest approach,
but with every height counting -- after capping very deep holes at
``depth``.

Spacing already sets each glyph's side bearings for ordinary neighbours, so
a pair is only kerned for what its *combination* does: the distance is
compared with what the same two glyphs give against a neutral partner --
a virtual flat edge ``|`` standing ``edge`` units off them at every height
(a real letter brings its own serifs into the comparison),

    expected(L, R) = d(L, |) + d(|, R) - d(|, |)

and the kern closes (or opens) a share ``strength`` of the difference.
Round sides cancel out (o is as round next to H as next to o); what is left
is the T's bar overhanging an x-height letter, diagonals that run parallel
(AV), punctuation tucked under an F or P, quotes over an A.
"""
from __future__ import annotations

import numpy as np

from .geometry import flatten

STEP = 5.0          # profile sampling (font units)
Y0, Y1 = -260.0, 960.0


def profiles(contours, step: float = STEP):
    """Left and right ink extents of an outline at heights Y0..Y1 (NaN: no ink)."""
    ys = np.arange(Y0, Y1 + step / 2, step)
    lo = np.full(ys.shape, np.nan)
    hi = np.full(ys.shape, np.nan)
    for c in contours:
        poly = flatten(c, 12)
        if len(poly) < 3:
            continue
        a, b = poly, np.roll(poly, -1, axis=0)
        for (x0, y0), (x1, y1) in zip(a, b):
            if y0 == y1:
                continue
            ymin, ymax = min(y0, y1), max(y0, y1)
            i0 = int(np.ceil((ymin - Y0) / step))
            i1 = int(np.floor((ymax - Y0) / step))
            if i1 < i0:
                continue
            yy = ys[i0:i1 + 1]
            xx = x0 + (yy - y0) * (x1 - x0) / (y1 - y0)
            seg_lo, seg_hi = lo[i0:i1 + 1], hi[i0:i1 + 1]
            lo[i0:i1 + 1] = np.where(np.isnan(seg_lo), xx, np.minimum(seg_lo, xx))
            hi[i0:i1 + 1] = np.where(np.isnan(seg_hi), xx, np.maximum(seg_hi, xx))
    return lo, hi


def _open(prof: np.ndarray, n: int, side: int) -> np.ndarray:
    """Morphological opening of a side profile over ``n`` samples: features
    thinner (in height) than the window are smoothed away. ``side`` -1: a
    left profile (protrusions are smaller x), +1: a right one."""
    x = np.where(np.isnan(prof), -side * np.inf, prof)
    pad = np.pad(x, n, mode="edge")
    win = np.lib.stride_tricks.sliding_window_view(pad, 2 * n + 1)
    ero = (win.max(axis=1) if side < 0 else win.min(axis=1))       # pull protrusions back
    pad = np.pad(ero, n, mode="edge")
    win = np.lib.stride_tricks.sliding_window_view(pad, 2 * n + 1)
    out = (win.min(axis=1) if side < 0 else win.max(axis=1))
    out = np.where(np.isfinite(out), out, np.nan)
    return np.where(np.isnan(prof), np.nan, out)


def optical(prof: tuple, height: float, reach: float, step: float = STEP) -> tuple:
    """Side profiles with *serifs* discounted: protrusions under ``height``
    tall and under ``reach`` deep are replaced by the opened profile (the
    stem behind them). Deep thin features -- a T's bar -- stay."""
    n = max(1, int(round(height / step / 2)))
    lo, hi = prof
    lo_o, hi_o = _open(lo, n, -1), _open(hi, n, +1)
    lo_e = np.where((lo_o - lo) < reach, lo_o, lo)
    hi_e = np.where((hi - hi_o) < reach, hi_o, hi)
    return lo_e, hi_e


def _distance(gap: np.ndarray, power: float, depth: float) -> float:
    g = gap[~np.isnan(gap)]
    if g.size == 0:
        return np.nan
    g = np.clip(g, 1.0, depth)
    return float(np.mean(g ** power) ** (1.0 / power))


class Kerner:
    """Pair distances from glyph profiles; ``kern(L, R)`` the suggested value."""

    def __init__(self, glyphs: dict, power: float = -32.0, depth: float = 500.0, strength: float = 0.5,
                 punct_strength: float | None = None, edge: float = 40.0, clearance: float = 20.0,
                 limit: tuple = (-160.0, 60.0), serif_height: float = 80.0, serif_reach: float = 100.0):
        self.adv = {n: g.advance for n, g in glyphs.items()}
        self.raw = {n: profiles(g.contours) for n, g in glyphs.items() if g.contours}
        # the optical measure reads stems, not serif tips; collisions use the raw outline
        self.prof = {n: optical(p, serif_height, serif_reach) for n, p in self.raw.items()}
        self.prof_len = len(next(iter(self.prof.values()))[0]) if self.prof else 0
        self.power, self.depth, self.strength = power, depth, strength
        # pairs with punctuation may be kerned more lightly (a style choice)
        self.punct_strength = strength if punct_strength is None else punct_strength
        import unicodedata
        self.punct = {n for n, g in glyphs.items()
                      if g.unicode is not None and unicodedata.category(chr(g.unicode))[0] in "PS"}
        self.edge, self.clearance, self.limit = edge, clearance, limit
        self._d: dict = {}

    def gap(self, left: str, right: str) -> np.ndarray:
        """White between the pair at each height ("|": the neutral edge)."""
        rl = self.edge if left == "|" else self.adv[left] - self.prof[left][1]
        lr = self.edge if right == "|" else self.prof[right][0]
        g = rl + lr
        return np.full(self.prof_len, g) if np.isscalar(g) else g

    def distance(self, left: str, right: str) -> float:
        key = (left, right)
        if key not in self._d:
            self._d[key] = _distance(self.gap(left, right), self.power, self.depth)
        return self._d[key]

    def kern(self, left: str, right: str) -> int:
        if left not in self.prof or right not in self.prof:
            return 0
        d = self.distance(left, right)
        exp = self.distance(left, "|") + self.distance("|", right) - self.distance("|", "|")
        if np.isnan(d) or np.isnan(exp):
            return 0
        st = self.punct_strength if (left in self.punct or right in self.punct) else self.strength
        k = -st * (d - exp)
        k = float(np.clip(k, *self.limit))
        # a kern never brings the outlines closer than ``clearance`` -- the
        # true (2-D) distance between the left glyph's right edge and the
        # right glyph's left edge, so a foot serif under a diagonal counts.
        # It only limits tightening: pairs that sit close anyway are not
        # pushed apart.
        k = int(5 * round(k / 5))
        if k < 0:
            base = self._closest(left, right)
            if base(0) < self.clearance:      # already that close: no tightening
                return 0
            while k < 0 and base(k) < self.clearance:
                k += 5
        return k

    def _closest(self, left, right):
        """f(k): the smallest distance between the pair's facing edges, the
        right glyph moved by k."""
        ys = np.arange(Y0, Y1 + STEP / 2, STEP)
        hi = self.raw[left][1]
        lo = self.raw[right][0]
        a = ~np.isnan(hi)
        b = ~np.isnan(lo)
        xl, yl = hi[a], ys[a]
        xr, yr = lo[b] + self.adv[left], ys[b]
        dy = (yl[:, None] - yr[None, :])
        near = np.abs(dy) <= self.clearance            # only heights that can matter
        dx0 = xr[None, :] - xl[:, None]

        def f(k):
            dx = dx0 + k
            d2 = np.where(near, dx * dx + dy * dy, np.inf)
            d2 = np.where(near & (dx < 0), -1.0, d2)   # overlapping at nearby heights: distance 0
            return float(np.sqrt(max(d2.min(), 0.0))) if np.isfinite(d2.min()) else np.inf
        return f


def autokern(glyphs: dict, names, min_abs: int = 10, **kw) -> dict:
    """Suggested kerning for every ordered pair of ``names`` (|value| >= min_abs)."""
    kr = Kerner(glyphs, **kw)
    names = [n for n in names if n in kr.prof]
    out = {}
    for a in names:
        for b in names:
            v = kr.kern(a, b)
            if abs(v) >= min_abs:
                out[(a, b)] = v
    return out
