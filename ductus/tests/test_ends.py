"""Stroke-end detail: the ends metric, brackets on diagonals, wedge sweep,
needle removal."""
from pathlib import Path

import numpy as np
import pytest

from ductus.bench.metrics import score_glyph, terminal_style, translated
from ductus.compile import _drop_spikes, compile_font, translated_contours
from ductus.geometry import Contour, flatten
from ductus.loader import load_font

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def bench_font():
    return load_font(ROOT / "benchmarks/geist_serif/font.py")


def _terms(cg):
    return translated_contours(cg.stages.get("terminals", []), cg.shift)


def test_ends_metric_sees_a_change_at_the_stroke_end_only(bench_font):
    """Scoring a glyph against itself with only its head wedge changed: the
    ends distance moves, and it is larger than the whole-outline mean (the
    change is concentrated at the end)."""
    a = compile_font(bench_font, glyphs=["n"]).glyphs["n"]
    b = compile_font(bench_font, {"head_len": float(bench_font.params["head_len"].default) + 12},
                     glyphs=["n"]).glyphs["n"]
    same = score_glyph("n", a.contours, a.contours, a.advance, a.advance, terminals=_terms(a))
    diff = score_glyph("n", a.contours, b.contours, a.advance, b.advance, terminals=_terms(a))
    assert same.ends_mean == pytest.approx(0, abs=1e-3)
    assert diff.ends_mean > diff.mean_dist > 0
    assert "wedge" in diff.ends and diff.ends["wedge"] > 0


def test_terminal_styles_are_recognised(bench_font):
    cg = compile_font(bench_font, glyphs=["n", "a"]).glyphs
    styles = {terminal_style(c) for g in ("n", "a") for c in _terms(cg[g])}
    assert {"wedge", "bracketed"} <= styles


def test_bracket_starts_on_a_diagonal_stroke_edge(bench_font):
    """The v's top serifs: the bracket meets the arm's own (slanted) edge,
    so there's no step between serif and arm -- the merged outline's left
    side is monotonic in x going down from the serif."""
    g = compile_font(bench_font, glyphs=["v"]).glyphs["v"]
    pts = flatten(max(g.contours, key=lambda c: abs(c.area())), 32)
    top = pts[:, 1].max()
    band = pts[(pts[:, 1] < top - 25) & (pts[:, 1] > top - 90) & (pts[:, 0] < pts[:, 0].min() + 80)]
    band = band[np.argsort(-band[:, 1])]
    # going down the left side of the left arm, x never steps back left
    assert np.all(np.diff(band[:, 0]) > -1.0)


def test_wedge_sweep_keeps_structure_and_bends_the_edge(bench_font):
    base = compile_font(bench_font, {"t_wedge_sweep": 0}, glyphs=["t"]).glyphs["t"]
    bent = compile_font(bench_font, {"t_wedge_sweep": 40}, glyphs=["t"]).glyphs["t"]
    assert base.raw_signature == bent.raw_signature
    assert sum(abs(c.area()) for c in bent.contours) < sum(abs(c.area()) for c in base.contours)


def test_needle_spikes_are_dropped():
    c = Contour()
    for p in [(0, 0), (100, 0), (100, 100), (50, 100), (40, 104), (48, 101), (0, 100)]:
        c.add(p, "on")
    out = _drop_spikes(c)
    assert len(out.pts) < len(c.pts)
    assert abs(out.area()) == pytest.approx(abs(c.area()), rel=0.01)
    square = Contour()
    for p in [(0, 0), (100, 0), (100, 100), (0, 100)]:
        square.add(p, "on")
    assert len(_drop_spikes(square).pts) == 4
