"""Fine controls: per-node width/direction/tension, sidebearings, stress."""
import numpy as np
import pytest

from ductus.compile import compile_font
from ductus.lint import check_compatibility, parse_sweep
from ductus.overrides import Override, dump_overrides, parse_overrides
import yaml


def node(cf, glyph, name):
    for s in cf.glyphs[glyph].skeleton:
        for n in s["nodes"]:
            if n["name"] == name:
                return n
    raise KeyError(name)


def test_width_override_changes_width_only(font):
    ov = {"n.arch.shoulder": Override("n.arch.shoulder", dw=30)}
    a, b = compile_font(font), compile_font(font, overrides=ov)
    assert node(b, "n", "n.arch.shoulder")["width"] == pytest.approx(node(a, "n", "n.arch.shoulder")["width"] + 30)
    pa, pb = a.glyphs["n"].outline_points, b.glyphs["n"].outline_points
    gap = lambda p: np.subtract(p["n.arch.shoulder.l"], p["n.arch.shoulder.r"])
    assert np.linalg.norm(gap(pb)) - np.linalg.norm(gap(pa)) == pytest.approx(30, abs=0.5)
    assert a.glyphs["n"].raw_signature == b.glyphs["n"].raw_signature


def test_direction_and_tension_overrides(font):
    a = compile_font(font)
    b = compile_font(font, overrides={"n.arch.join": Override("n.arch.join", da=-15),
                                      "n.arch.shoulder": Override("n.arch.shoulder", dt=0.4)})
    assert node(b, "n", "n.arch.join")["dir"] == pytest.approx(node(a, "n", "n.arch.join")["dir"] - 15, abs=0.1)
    assert node(b, "n", "n.arch.shoulder")["tension"] == pytest.approx(1.4)
    assert a.glyphs["n"].contours[0].pts != b.glyphs["n"].contours[0].pts


def test_sidebearing_overrides(font):
    a = compile_font(font).glyphs["o"]
    b = compile_font(font, overrides={"o.lsb": Override("o.lsb", dx=10),
                                      "o.rsb": Override("o.rsb", dx=-4)}).glyphs["o"]
    assert b.advance == a.advance + 6
    assert b.bounds[0] == pytest.approx(a.bounds[0] + 10)


def test_stress_rounds_outer_contour(font):
    """As stress -> 1 the thick/thin modulation moves into the counter: the
    top of the o's outer contour moves to the centre (no more egg)."""
    from ductus.geometry import flatten
    skew = []
    for stress in (0.5, 0.72, 1.0):
        g = compile_font(font, {"stem": 160, "contrast": 0.7, "stress": stress}).glyphs["o"]
        pts = flatten(max(g.contours, key=lambda c: abs(c.area())), 64)
        skew.append(abs(pts[np.argmax(pts[:, 1]), 0] - (g.bounds[0] + g.bounds[2]) / 2))
    assert skew[0] > skew[1] > skew[2]
    assert skew[0] > 30 and skew[2] < 3


def test_new_fields_round_trip():
    ovs = {"a.b.c": Override("a.b.c", dx=1, dw=12.5, da=-3, dt=0.2),
           "a.lsb": Override("a.lsb", dx=-8)}
    assert parse_overrides(yaml.safe_load(dump_overrides(ovs))) == ovs
    with pytest.raises(ValueError):
        parse_overrides({"x": {"dz": 1}})


def test_controls_keep_compatibility(font):
    ov = {"n.arch.shoulder": Override("n.arch.shoulder", dx=12, dw=20, da=5, dt=0.3)}
    findings = check_compatibility(font, ov, parse_sweep(["stem=40,160", "stress=0.5,1"]))
    assert not [f for f in findings if f.level == "error"]
