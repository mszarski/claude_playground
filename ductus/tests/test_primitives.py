"""M2 primitives: width profiles, L/R bias, terminal library, cuts, kits."""
from pathlib import Path

import numpy as np
import pytest

from ductus.compile import compile_font
from ductus.lint import check_compatibility, check_direction, check_self_intersections, parse_sweep, run_lints
from ductus.loader import load_font
from ductus.profiles import Profile, swell, taper

ROOT = Path(__file__).resolve().parents[1]


def test_profile_interpolation():
    p = Profile((0, 0.2), (0.5, 1.0), (1, 0.4))
    assert p(np.array([0, 0.5, 1])) == pytest.approx([0.2, 1.0, 0.4])
    mid = p(np.array([0.25]))[0]
    assert 0.2 < mid < 1.0
    assert taper(0.1, 0.3)(np.array([0, 0.3, 0.9])) == pytest.approx([0.1, 1, 1])
    assert swell(0.5, 1.5, 0.5, 0.2)(np.array([0.5]))[0] == pytest.approx(1.5)


STROKE_FONT = """
font = Font("T"); stem = param(80, range=(40, 160)); contrast = param(0.6, range=(0, 0.9))
taper_ = param(0.2, range=(0.05, 1)); bias = param(0.5, range=(0.1, 0.9))
font.pen = Pen(angle=0, width=stem, contrast=contrast)
from ductus.profiles import Profile
from ductus.terminals import TerminalSet, bracketed, wedge, flat, beak, ball
font.terminals = TerminalSet(foot=bracketed(length=50, bracket=30, tip=5),
                             head=wedge(side="left", length=60, rise=40, drop=40),
                             cut=flat("vertical"), bk=beak(side="above", length=30, depth=40, axis=0),
                             bl=ball(radius=30, side="below"))
@glyph("n")
def n(g):
    g.line((0, 0), (0, 540), name="stem", names=("bottom", "top"), terminal=("foot", "head"))
    (g.path(name="arch", bias=bias, profile=Profile((0, taper_), (0.3, 1), (1, 1)))
        .start((20, 420), name="join", dir=45, terminal=False)
        .curve_to((200, 540), name="shoulder", dir=0)
        .curve_to((400, 300), name="out", dir=-90)
        .line_to((400, 0), name="leg", terminal="foot"))
@glyph("v")
def v(g):
    g.line((0, 530), (200, 0), name="diag", names=("top", "vertex"), terminal=("foot", "cut"))
@glyph("c")
def c(g):
    (g.path(name="c").start((380, 430), name="t1", dir=100, terminal="bk")
        .curve_to((200, 540), name="top", dir=180).curve_to((40, 270), name="left", dir=270)
        .curve_to((200, 0), name="bottom", dir=0).curve_to((380, 120), name="t2", dir=60, terminal="bl"))
@glyph("e")
def e(g):
    g.bowl(0, 450, -10, 540, thick=stem, thin=20)
    g.cut("ap").move((225, 260)).line((600, 260)).line((600, 120)).line((225, 120)).close()
"""


@pytest.fixture
def sfont(write_font):
    return write_font(STROKE_FONT)


def test_profile_thins_the_join_without_changing_structure(sfont):
    thin = compile_font(sfont, {"taper_": 0.1}).glyphs["n"]
    full = compile_font(sfont, {"taper_": 1.0}).glyphs["n"]
    gap = lambda g: np.linalg.norm(np.subtract(g.outline_points["n.arch.join.l"], g.outline_points["n.arch.join.r"]))
    assert gap(thin) < 0.3 * gap(full)
    assert thin.raw_signature == full.raw_signature


def test_bias_moves_modulation_to_one_side(sfont):
    def halves(b):
        g = compile_font(sfont, {"bias": b}).glyphs["n"]
        o, l, r = g.points["n.arch.out"], g.outline_points["n.arch.out.l"], g.outline_points["n.arch.out.r"]
        return abs(l[0] - o[0]), abs(r[0] - o[0])
    l1, r1 = halves(0.2)
    l2, r2 = halves(0.8)
    assert r1 > l1 and l2 > r2
    assert l1 + r1 == pytest.approx(l2 + r2, abs=1)      # total width unchanged


def test_terminals_add_and_cut(sfont):
    cf = compile_font(sfont)
    n = cf.glyphs["n"]
    assert "terminal:bracketed" in n.primitives and "terminal:wedge" in n.primitives
    assert n.bounds[3] <= 540.5                           # wedge cut nothing above the stem top
    stem_x = n.points["n.stem.bottom"][0]
    assert stem_x - n.bounds[0] > 40 + 50 - 5            # serif/wedge reach past the stem's left edge
    v = cf.glyphs["v"]
    assert v.bounds[1] == pytest.approx(0, abs=0.6)       # flat vertical cut at the vertex: nothing below 0
    for g in cf.glyphs.values():
        assert not check_direction(g) and not check_self_intersections(g), g.name
    assert "terminal:ball" in cf.glyphs["c"].primitives


def test_aperture_cut_opens_the_bowl(sfont):
    g = compile_font(sfont).glyphs["e"]
    assert len(g.contours) == 1                           # the counter is open now
    assert "|-" in g.raw_signature


def test_terminal_set_swap_restyles(sfont):
    from ductus.terminals import TerminalSet
    a = compile_font(sfont).glyphs["n"]
    b = compile_font(sfont, terminals=TerminalSet()).glyphs["n"]
    assert a.signature != b.signature and b.bounds[0] > a.bounds[0]


def test_primitive_sweep_compatible(sfont):
    findings = check_compatibility(sfont, {}, parse_sweep(["stem=40,160", "contrast=0,0.9", "taper_=0.05,1"]))
    assert not [f for f in findings if f.level == "error"]


@pytest.mark.parametrize("path", ["examples/display_serif/font.py", "benchmarks/geist_serif/font.py"])
def test_kit_fonts_lint_clean_and_compatible(path):
    font = load_font(ROOT / path)
    findings, cf = run_lints(font, {}, parse_sweep(["stem=60,130", "hairline=10,50"]))
    errors = [f for f in findings if f.level == "error"]
    assert not errors, errors[:5]
    assert cf.compile_ms < 1000


def test_display_font_has_the_test_word():
    cf = compile_font(load_font(ROOT / "examples/display_serif/font.py"))
    assert set("hamburgefonstivHOAGES") | {"space"} <= set(cf.glyphs)
    assert all(g.contours for n, g in cf.glyphs.items() if n != "space")


def test_bowl_rotation_keeps_structure_and_tilts(write_font):
    f = write_font("""
font = Font("T"); rot = param(0, range=(-30, 30)); font.pen = Pen(width=param(80))
@glyph("o")
def o(g):
    g.bowl(0, 400, 0, 300, thick=80, thin=20, rotate=rot)
""")
    a, b = compile_font(f).glyphs["o"], compile_font(f, {"rot": 20}).glyphs["o"]
    assert a.raw_signature == b.raw_signature
    top_a, top_b = a.points["o.outer.top"], b.points["o.outer.top"]
    assert top_b[0] < top_a[0]                  # CCW rotation moves the top extreme left
    assert not check_direction(b) and not check_self_intersections(b)


def test_serif_slope_tapers_the_serif(write_font):
    src = """
font = Font("T"); sl = param(0, range=(-10, 50)); font.pen = Pen(width=param(80))
from ductus.terminals import TerminalSet, bracketed
font.terminals = TerminalSet(foot=bracketed(length=70, bracket=40, tip=4, slope=sl))
@glyph("l")
def l(g):
    g.line((0, 0), (0, 600), name="stem", names=("bottom", "top"), terminal=("foot", False))
"""
    f = write_font(src)
    from ductus.geometry import flatten
    def area_near_tip(params):
        g = compile_font(f, params).glyphs["l"]
        poly = np.concatenate([flatten(c, 16) for c in g.contours])
        x0 = g.points["l.stem.bottom"][0]
        tipzone = poly[(poly[:, 0] < x0 - 40 - 50)]       # outer 20 units of the left serif
        return tipzone[:, 1].max() if len(tipzone) else 0
    # flat (slope 0): the serif stays tip-thin out near the tip; sloped: it
    # thickens steadily from a sharp tip -- the wedge-like serif
    assert area_near_tip({"sl": 35}) > area_near_tip({"sl": 0}) + 5


def test_stage_debug_geometry(sfont):
    j = compile_font(sfont).glyphs["n"].to_json()
    assert set(j["stages"]) >= {"strokes", "terminals"}
    assert all(v.startswith("M") for v in j["stages"].values())
    e = compile_font(sfont).glyphs["e"].to_json()
    assert "cuts" in e["stages"] and "contours" in e["stages"]
