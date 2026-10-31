import pytest

from ductus.compile import compile_font
from ductus.parts import DuctusError


def test_hierarchical_names(font):
    cf = compile_font(font)
    assert {"n.stem.bottom", "n.stem.top", "n.arch.join", "n.arch.shoulder",
            "n.arch.shoulder_out", "n.arch.leg"} <= set(cf.glyphs["n"].points)
    assert "m.arch2.shoulder_out" in cf.glyphs["m"].points
    assert "H.left.top" in cf.glyphs["H"].points and "H.bar.left" in cf.glyphs["H"].points
    assert "n.arch.shoulder.l" in cf.glyphs["n"].outline_points


def test_inherit_rescopes_and_modifies(font):
    cf = compile_font(font)
    h, n = cf.glyphs["h"], cf.glyphs["n"]
    assert h.points["h.stem.top"][1] == pytest.approx(cf.metrics["ascender"])
    assert n.points["n.stem.top"][1] == pytest.approx(cf.metrics["xheight"])
    assert h.points["h.arch.shoulder"] == n.points["n.arch.shoulder"] or \
        "n.arch.shoulder" in cf.overrides_used   # differs only by n's override


def test_shared_parts(font):
    """h and m reuse the arch from n; o and O share the bowl part."""
    cf = compile_font(font)
    names = lambda g: {s["name"] for s in cf.glyphs[g].skeleton}
    assert {"h.arch", "h.stem"} == names("h")
    assert {"m.arch1", "m.arch2", "m.stem"} == names("m")
    assert names("o") == {"o.bowl"} and names("O") == {"O.bowl"}


def test_duplicate_names_are_rejected(write_font):
    f = write_font("""
font = Font("T"); font.pen = Pen(width=param(80))
@part
def stem_(p, x): p.line((x, 0), (x, 500), name="stem")
@glyph("x")
def x(g):
    g += stem_(x=0)
    g += stem_(x=200)
""")
    with pytest.raises(DuctusError, match="duplicate name 'x.stem'"):
        compile_font(f)


def test_multi_stroke_part_and_nesting(write_font):
    f = write_font("""
font = Font("T"); font.pen = Pen(width=param(80))
@part
def pair(p, x):
    p.line((x, 0), (x, 500), name="a")
    p.line((x + 200, 0), (x + 200, 500), name="b")
@part
def outer(p):
    p += pair(x=0, name="inner")
@glyph("x")
def x(g):
    g += outer()
""")
    pts = compile_font(f).glyphs["x"].points
    assert {"x.outer.inner.a.start", "x.outer.inner.b.end"} <= set(pts)


def test_raw_contour_escape_hatch(write_font):
    f = write_font("""
font = Font("T"); font.pen = Pen(width=param(80))
@glyph("period")
def period(g):
    g.contour("dot").move((0, 0), "bl").line((100, 0)).line((100, 100), "tr").line((0, 100)).close()
""")
    cf = compile_font(f, overrides={})
    g = cf.glyphs["period"]
    assert "period.dot.tr" in g.points and g.bounds[2] - g.bounds[0] == 100
    assert g.contours[0].area() > 0
