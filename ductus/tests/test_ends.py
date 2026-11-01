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
    assert same.ends_mean == pytest.approx(0, abs=0.1)     # the best-offset search is good to ~0.02 u
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


# -- pen angle varying along a stroke ----------------------------------------

def test_pen_turning_keeps_vertical_stems_exact_and_moves_the_thick_part():
    from ductus.pen import Pen
    pen = Pen(angle=20, width=100, contrast=0.8)
    up = np.radians([90.0, 90.0, 90.0])
    w = pen.widths(up, np.radians([0.0, 15.0, -25.0]))
    assert np.allclose(w, 100.0)                        # a vertical stroke is always `width`
    diag = np.radians([45.0])
    assert pen.widths(diag, np.radians([30.0]))[0] != pytest.approx(pen.widths(diag)[0])
    assert np.allclose(pen.widths(np.radians([10.0, 70.0])), pen.widths(np.radians([10.0, 70.0]), [0.0, 0.0]))


def test_node_pen_angle_and_dp_override(bench_font):
    from ductus.overrides import Override
    base = compile_font(bench_font, glyphs=["s"]).glyphs["s"]
    turned = compile_font(bench_font, glyphs=["s"],
                          overrides={"s.spine.mid": Override("s.spine.mid", dp=25)}).glyphs["s"]
    assert base.raw_signature == turned.raw_signature              # structure unchanged
    assert abs(sum(c.area() for c in base.contours) - sum(c.area() for c in turned.contours)) > 50
    mid = [n for st in turned.skeleton for n in st["nodes"] if n["name"] == "s.spine.mid"][0]
    assert mid["pen_angle"] is not None


def test_dp_override_round_trips():
    import yaml
    from ductus.overrides import Override, dump_overrides, parse_overrides
    ovs = {"s.spine.mid": Override("s.spine.mid", dp=7.5)}
    assert parse_overrides(yaml.safe_load(dump_overrides(ovs))) == ovs


def test_spacing_table_and_kerning_reach_the_font(tmp_path):
    from ductus.emit import build_otf, validate_otf
    from fontTools.ttLib import TTFont
    f = load_font(ROOT / "examples/display_serif/font.py")
    base = compile_font(f, glyphs=["n", "o"])
    f.spacing["n"] = (10, -5)
    f.kern("n", "o", -17)
    cf = compile_font(f, glyphs=["n", "o"])
    assert cf.glyphs["n"].advance == base.glyphs["n"].advance + 5
    assert cf.kerning[("n", "o")] == -17
    otf = build_otf(cf, tmp_path / "k.otf")
    assert not validate_otf(otf)
    t = TTFont(otf)
    assert t["kern"].kernTables[0].kernTable[("n", "o")] == -17 and "GPOS" in t
