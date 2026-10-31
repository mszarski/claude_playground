import math

import numpy as np
import pytest

from ductus import Pen
from ductus.geometry import bez, fit_cubic, hobby_cubic
from ductus.params import evaluating
from ductus.pen import CURVE_SEGMENTS, expand_stroke
from ductus.skeleton import Node, Stroke


def stroke(nodes, kinds, closed=False):
    s = Stroke("s", [Node(n, np.array(p, float), d) for n, p, d in nodes], kinds, closed)
    s.fullname = "g.s"
    for n in s.nodes:
        n.fullname = f"g.s.{n.name}"
    return s


def test_vertical_stem_is_exactly_stem_width():
    pen = Pen(angle=20, width=80, contrast=0.6)
    e = expand_stroke(stroke([("a", (0, 0), None), ("b", (0, 500), None)], ["line"]), pen)
    xs = sorted({round(p[0], 6) for p in e.contours[0].pts})
    assert xs == [-40, 40]


def test_width_range_follows_contrast():
    pen = Pen(angle=0, width=100, contrast=0.7)
    assert pen.width_at(90) == pytest.approx(100)
    assert pen.width_at(0) == pytest.approx(30)          # hairline = stem * (1 - contrast)
    mono = Pen(angle=30, width=100, contrast=0)
    assert all(mono.width_at(a) == pytest.approx(100) for a in range(0, 180, 15))


def test_hobby_quarter_circle():
    c = hobby_cubic((100, 0), math.pi / 2, (0, 100), math.pi)
    r = np.linalg.norm(bez(c, np.linspace(0, 1, 50)), axis=1)
    assert np.abs(r - 100).max() < 0.1


def test_fit_cubic_recovers_cubic():
    P = np.array([[0, 0], [30, 80], [120, 90], [150, 0]], float)
    t = np.linspace(0, 1, 30)
    Q = bez(P, t)
    d0 = (P[1] - P[0]) / np.linalg.norm(P[1] - P[0])
    d1 = (P[3] - P[2]) / np.linalg.norm(P[3] - P[2])
    F = fit_cubic(Q, d0, d1)
    true, fitted = bez(P, np.linspace(0, 1, 200)), bez(F, np.linspace(0, 1, 400))
    assert np.linalg.norm(true[:, None] - fitted[None], axis=2).min(1).max() < 0.6  # below the 1-unit output grid


def test_closed_ring_offsets_follow_width():
    pen = Pen(angle=0, width=60, contrast=0.5)   # vertical 60, horizontal 30
    s = stroke([("b", (0, -200), 0), ("r", (200, 0), 90), ("t", (0, 200), 180), ("l", (-200, 0), 270)],
               ["curve"] * 4, closed=True)
    e = expand_stroke(s, pen)
    assert len(e.contours) == 2 and not e.ends
    pts = {n: p for c in e.contours for p, n in zip(c.pts, c.names) if n}
    assert pts["g.s.r.r"][0] == pytest.approx(230)   # right of travel = outside
    assert pts["g.s.r.l"][0] == pytest.approx(170)
    assert pts["g.s.t.r"][1] == pytest.approx(215)
    assert pts["g.s.t.l"][1] == pytest.approx(185)


def test_point_count_is_fixed_per_segment():
    pen = Pen(angle=20, width=80, contrast=0.5)
    s = stroke([("a", (0, 0), 90), ("b", (100, 100), 0), ("c", (200, 0), -90), ("d", (200, -100), None)],
               ["curve", "curve", "line"])
    sig = expand_stroke(s, pen).contours[0].signature()
    per_side = 1 + 2 * 3 * CURVE_SEGMENTS + 1
    assert len(sig) == 2 * per_side
    with evaluating({}):
        heavy = Pen(angle=40, width=170, contrast=0.9)
        assert expand_stroke(s, heavy).contours[0].signature() == sig


def test_offset_fit_error_is_small(font):
    """Fitted outline stays within 3 units of the true offset curve, even at
    the heaviest, highest-contrast corner of the design space."""
    from ductus import pen as P
    from ductus.parts import GlyphBuilder, realize
    vals = {n: p.default for n, p in font.params.items()}
    vals.update(stem=180, contrast=0.9)
    worst = 0.0
    with evaluating(vals):
        for g in font.glyphs.values():
            gb = GlyphBuilder(g, font); g.fn(gb)
            for s in realize(gb).strokes:
                for C, kind in s.cubics():
                    if kind != "curve":
                        continue
                    for side in (1, -1):
                        for k in range(P.CURVE_SEGMENTS):
                            ts = np.linspace(k / P.CURVE_SEGMENTS, (k + 1) / P.CURVE_SEGMENTS, P.SAMPLES + 1)
                            Q = P._offset(C, ts, font.pen, side)
                            F = fit_cubic(Q, P._offset_tangent(C, ts[0], font.pen, side),
                                          P._offset_tangent(C, ts[-1], font.pen, side))
                            true = P._offset(C, np.linspace(ts[0], ts[-1], 120), font.pen, side)
                            fitted = bez(F, np.linspace(0, 1, 300))
                            d = np.linalg.norm(true[:, None] - fitted[None], axis=2).min(1).max()
                            worst = max(worst, d)
    assert worst < 3.0
