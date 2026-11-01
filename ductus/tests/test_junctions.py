"""Join primitive (SPEC 5.3): arches and lobes meeting stems."""
import numpy as np
import pytest

from ductus.compile import compile_font
from ductus.junctions import contact, join
from ductus.lint import check_compatibility, check_direction, check_self_intersections, parse_sweep

LOBE_FONT = """
from ductus.junctions import join
font = Font("T"); font.pen = Pen(width=param(84))
top_at = param(280, range=(200, 340)); top_thin = param(35, range=(8, 90)); top_angle = param(20, range=(-20, 60))
bot_at = param(150, range=(60, 240)); bot_thin = param(60, range=(8, 120)); bot_angle = param(40, range=(0, 80))
thick = param(95, range=(40, 150)); thin_b = param(30, range=(6, 80)); side = 1
@glyph("a")
def a(g):
    g.line((420, 0), (420, 400), name="stem", names=("bottom", "top"), terminal=False)
    g.lobe(378, -1, 40, -12, join(top_at, top_thin, top_angle), join(bot_at, bot_thin, bot_angle),
           thick=thick, thin_bottom=thin_b)
@glyph("b")
def b(g):
    g.line((0, 0), (0, 700), name="stem", names=("bottom", "top"), terminal=False)
    g.lobe(42, +1, 480, -12, join(top_at, top_thin, top_angle), join(bot_at, bot_thin, bot_angle),
           thick=thick, thin_bottom=thin_b)
"""


def test_contact_geometry():
    J = join(at=300, thin=30, angle=0)
    crotch, outer, start, d, thin = contact(J, 100, +1, rising=True)
    assert crotch == (100, 300) and outer == (100, 330) and d == pytest.approx(0)
    assert start[0] < 100 and 300 < start[1] < 330          # starts inside the stem, mid-band
    J2 = join(at=300, thin=30, angle=45)
    _, outer2, _, d2, _ = contact(J2, 100, -1, rising=True)
    assert outer2[1] - 300 == pytest.approx(30 / np.cos(np.radians(45)))
    assert d2 == pytest.approx(135)                          # leaving to the left, upwards


@pytest.mark.parametrize("gname,side", [("a", -1), ("b", +1)])
def test_lobe_meets_stem_at_its_joins(write_font, gname, side):
    f = write_font(LOBE_FONT)
    g = compile_font(f).glyphs[gname]
    pts = g.points
    edge = pts[f"{gname}.inner.top"][0]
    assert pts[f"{gname}.inner.end"][0] == pytest.approx(edge)
    assert pts[f"{gname}.inner.top"][1] == pytest.approx(280)          # upper crotch at top_join.at
    assert pts[f"{gname}.inner.end"][1] == pytest.approx(150)          # lower crotch at bottom_join.at
    # top wall thickness where it enters the stem = thin / cos(angle)
    assert pts[f"{gname}.outer.top"][1] - 280 == pytest.approx(35 / np.cos(np.radians(20)), abs=0.05)
    assert len(g.contours) == 2                                        # stem+lobe, and one counter
    assert not check_direction(g) and not check_self_intersections(g)
    assert "contour:lobe" in g.primitives


def test_lobe_thin_bottom_never_crosses(write_font):
    """Regression: with a thin bottom wall the counter's bottom curve used to
    cross the outer one."""
    f = write_font(LOBE_FONT)
    for thin_b in (6, 12, 30):
        for bot_at in (60, 150, 240):
            g = compile_font(f, {"thin_b": thin_b, "bot_at": bot_at}).glyphs["a"]
            assert len(g.contours) == 2, (thin_b, bot_at)
            assert not check_self_intersections(g)


def test_lobe_and_arch_sweeps_are_compatible(write_font):
    f = write_font(LOBE_FONT)
    findings = check_compatibility(f, {}, parse_sweep(["top_thin=8,90", "bot_angle=0,80", "thick=40,150"]))
    assert not [x for x in findings if x.level == "error"]


def test_arch_join_controls_the_crotch():
    from pathlib import Path
    from ductus.loader import load_font
    f = load_font(Path(__file__).resolve().parents[1] / "examples/display_serif/font.py")
    a = compile_font(f, {"n_join_thin": 10}).glyphs["n"]
    b = compile_font(f, {"n_join_thin": 60}).glyphs["n"]
    assert a.raw_signature == b.raw_signature
    wa = np.linalg.norm(np.subtract(a.outline_points["n.arch.join.l"], a.outline_points["n.arch.join.r"]))
    wb = np.linalg.norm(np.subtract(b.outline_points["n.arch.join.l"], b.outline_points["n.arch.join.r"]))
    assert wa == pytest.approx(10, abs=1.5) and wb == pytest.approx(60, abs=3)
    # notch: a cut appears in the stage data either way (degenerate at depth 0), deeper with depth
    n_cut = compile_font(f, {"n_notch": 15}).glyphs["n"]
    assert n_cut.raw_signature == a.raw_signature


def test_v_vertex_is_clipped_by_the_thin_arm():
    """The thick arm is cut along the thin arm's outer edge near the vertex,
    leaving a narrow flat point (SPEC 5.8 diagonals)."""
    from pathlib import Path
    from ductus.loader import load_font
    from ductus.geometry import flatten
    f = load_font(Path(__file__).resolve().parents[1] / "benchmarks/geist_serif/font.py")
    g = compile_font(f, glyphs=["v"]).glyphs["v"]
    assert "cuts" in g.stages and len(g.contours) == 1
    poly = np.concatenate([flatten(c, 16) for c in g.contours])
    bottom = poly[poly[:, 1] < poly[:, 1].min() + 3]
    assert bottom[:, 0].max() - bottom[:, 0].min() < 70          # a narrow point, not two arms' worth


def test_g_band_option_adds_flat_bar():
    from pathlib import Path
    from ductus.loader import load_font
    root = Path(__file__).resolve().parents[1]
    bench = compile_font(load_font(root / "benchmarks/geist_serif/font.py"), glyphs=["g"]).glyphs["g"]
    display = compile_font(load_font(root / "examples/display_serif/font.py"), glyphs=["g"]).glyphs["g"]
    assert "g.bar.tl" in bench.points and "g.bar.tl" not in display.points
