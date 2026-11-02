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
    l0, r0 = f.spacing.get("n", (0, 0))
    f.spacing["n"] = (l0 + 10, r0 - 5)
    f.kern("n", "o", -17)
    cf = compile_font(f, glyphs=["n", "o"])
    assert cf.glyphs["n"].advance == base.glyphs["n"].advance + 5
    assert cf.kerning[("n", "o")] == -17
    otf = build_otf(cf, tmp_path / "k.otf")
    assert not validate_otf(otf)
    t = TTFont(otf)
    assert t["kern"].kernTables[0].kernTable[("n", "o")] == -17 and "GPOS" in t


def test_spacing_table_widens_an_empty_glyph():
    f = load_font(ROOT / "examples/display_serif/font.py")
    base = compile_font(f, glyphs=["space"]).glyphs["space"].advance
    f.spacing["space"] = (5, 20)
    assert compile_font(f, glyphs=["space"]).glyphs["space"].advance == base - 44 + 25


def test_fi_ligature_compiles_emits_and_shapes(tmp_path):
    from ductus.emit import build_otf, validate_otf
    from ductus.proof import _shape
    from fontTools.ttLib import TTFont
    import freetype
    f = load_font(ROOT / "examples/display_serif/font.py")
    cf = compile_font(f)
    assert cf.ligatures == {"f_i": ("f", "i")} and cf.glyphs["f_i"].unicode is None
    otf = build_otf(cf, tmp_path / "l.otf")
    assert not validate_otf(otf)
    tt = TTFont(otf)
    ligs = [(first, e.Component, e.LigGlyph) for lk in tt["GSUB"].table.LookupList.Lookup
            for st in lk.SubTable for first, es in st.ligatures.items() for e in es]
    assert ("f", ["i"], "f_i") in ligs
    face = freetype.Face(str(otf))
    idx = _shape(otf, face, "fig")
    assert len(idx) == 2 and idx[0] == face.get_name_index(b"f_i")


def test_punctuation_and_digit_glyphs_are_mapped_and_shaped():
    from fontTools.pens.boundsPen import BoundsPen
    from ductus.emit import build_otf
    from fontTools.ttLib import TTFont
    import tempfile
    f = load_font(ROOT / "examples/display_serif/font.py")
    cf = compile_font(f)
    marks = ".,:;-–—'\"‘’“”!?()/0123456789"
    with tempfile.TemporaryDirectory() as d:
        t = TTFont(build_otf(cf, Path(d) / "p.otf"))
        cmap, gs = t.getBestCmap(), t.getGlyphSet()
        assert all(ord(c) in cmap for c in marks)

        def box(c):
            p = BoundsPen(gs)
            gs[cmap[ord(c)]].draw(p)
            return p.bounds
        assert box(",")[1] < -100 < 0 < box(".")[3]          # the comma's tail hangs below the baseline
        assert box("‘")[1] > 400 and box("’")[1] > 400        # quotes sit high
        assert box("(")[1] < 0 < 700 < box("(")[3]            # parentheses span descent to above caps
        assert gs[cmap[ord("—")]].width > gs[cmap[ord("–")]].width > gs[cmap[ord("-")]].width
        assert 650 < box("0")[3] < 760 and box("1")[1] == 0          # lining figures


def test_composites_place_marks_on_their_base():
    from ductus.compile import _bounds
    f = load_font(ROOT / "examples/display_serif/font.py")
    cf = compile_font(f, glyphs=["a", "aacute", "A", "Aacute", "ccedilla", "c"])
    a, aa, A, AA = (cf.glyphs[n] for n in ("a", "aacute", "A", "Aacute"))
    assert aa.unicode == ord("á") and aa.advance == a.advance
    ba, baa, bA, bAA = (_bounds(g.contours) for g in (a, aa, A, AA))
    assert baa[3] > ba[3] + 100                        # the acute stands above the a
    lift = f.capheight - f.xheight
    assert abs((bAA[3] - bA[3]) - (baa[3] - ba[3]) - (lift - (bA[3] - ba[3]))) < 20   # caps: lifted by cap - x height
    bc, bcc = _bounds(cf.glyphs["c"].contours), _bounds(cf.glyphs["ccedilla"].contours)
    assert bcc[1] < bc[1] - 100                        # the cedilla hangs below the c
    assert len(cf.glyphs["ccedilla"].contours) == len(cf.glyphs["c"].contours)   # merged into the c's outline


def test_slant_shears_the_finished_outline():
    from ductus.compile import _bounds
    f = load_font(ROOT / "examples/display_serif/font.py")
    up = compile_font(f, {"at_slant": 0}, glyphs=["at"]).glyphs["at"]
    sl = compile_font(f, glyphs=["at"]).glyphs["at"]
    assert up.raw_signature == sl.raw_signature
    assert _bounds(sl.contours) != _bounds(up.contours)
