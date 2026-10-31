"""Contour-part bowls (SPEC 5.4)."""
import numpy as np
import pytest

from ductus.compile import compile_font
from ductus.geometry import flatten
from ductus.lint import check_compatibility, check_direction, check_self_intersections, parse_sweep

BOWL_FONT = """
font = Font("T")
stem = param(90, range=(40, 160)); hair = param(20, range=(5, 60))
stress = param(0, range=(-20, 20)); fh = param(0.55, range=(0.4, 0.85)); fv = param(0.55, range=(0.4, 0.85))
font.pen = Pen(width=stem)
@part
def bowl(p, w, h):
    p.bowl(0, w, -10, h + 10, thick=stem, thin=hair, stress=stress,
           fullness=(fh, fv), inner_fullness=0.55)
@glyph("o")
def o(g):
    g += bowl(w=500, h=530)
"""


def test_names_structure_and_orientation(write_font):
    f = write_font(BOWL_FONT)
    g = compile_font(f).glyphs["o"]
    for side in ("outer", "inner"):
        for ext in ("bottom", "right", "top", "left"):
            assert f"o.bowl.{side}.{ext}" in g.points
    assert g.raw_signature == "occoccoccocc|occoccoccocc"
    assert len(g.contours) == 2                        # ring: outer + counter
    assert sorted(c.area() > 0 for c in g.contours) == [False, True]
    assert not check_direction(g) and not check_self_intersections(g)
    assert "contour:bowl" in g.primitives


def test_walls_and_extrema(write_font):
    f = write_font(BOWL_FONT)
    p = compile_font(f).glyphs["o"].points
    assert p["o.bowl.inner.left"][0] - p["o.bowl.outer.left"][0] == pytest.approx(90)
    assert p["o.bowl.outer.top"][1] - p["o.bowl.inner.top"][1] == pytest.approx(20)
    assert p["o.bowl.outer.top"][1] == pytest.approx(540)


def test_stress_rotates_counter(write_font):
    f = write_font(BOWL_FONT)
    p = compile_font(f, {"stress": 12}).glyphs["o"].points
    top, bot = p["o.bowl.inner.top"], p["o.bowl.inner.bottom"]
    left, right = p["o.bowl.inner.left"], p["o.bowl.inner.right"]
    assert top[0] < bot[0]          # classic oblique: counter top leans left
    assert left[1] > right[1]
    po = compile_font(f, {"stress": 12}).glyphs["o"].points
    assert po["o.bowl.outer.top"][0] == pytest.approx((po["o.bowl.outer.left"][0] + po["o.bowl.outer.right"][0]) / 2)


def test_hv_fullness_controls_handles_separately(write_font):
    f = write_font(BOWL_FONT)
    a = compile_font(f, {"fh": 0.45, "fv": 0.8}).glyphs["o"]
    b = compile_font(f, {"fh": 0.8, "fv": 0.45}).glyphs["o"]
    outer = lambda g: max(g.contours, key=lambda c: abs(c.area()))
    # same extrema, different shapes
    assert a.bounds == pytest.approx(b.bounds, abs=1)
    pa, pb = flatten(outer(a), 32), flatten(outer(b), 32)
    assert np.abs(np.sort(pa[:, 1]) - np.sort(pb[:, 1])).max() > 5


def test_bowl_sweep_is_compatible(write_font):
    f = write_font(BOWL_FONT)
    findings = check_compatibility(f, {}, parse_sweep(["stem=40,160", "hair=5,60", "stress=-20,20", "fh=0.4,0.85"]))
    assert not [x for x in findings if x.level != "info"]
    sigs = {compile_font(f, {"stem": s, "hair": h}).glyphs["o"].signature for s in (40, 160) for h in (5, 60)}
    assert len(sigs) == 1                              # even the merged outline is stable


def test_bowl_override_moves_point_and_handles(write_font):
    from ductus.overrides import Override
    f = write_font(BOWL_FONT)
    a = compile_font(f).glyphs["o"].points["o.bowl.inner.top"]
    b = compile_font(f, overrides={"o.bowl.inner.top": Override("o.bowl.inner.top", dx=-15)}).glyphs["o"].points["o.bowl.inner.top"]
    assert b[0] - a[0] == pytest.approx(-15, abs=1.01)


@pytest.mark.parametrize("path,sweep", [
    pytest.param("examples/display_serif/font.py", ["stem=60,140"], id="display"),
])
def test_m2_fonts_are_compatible_and_lint_clean(path, sweep):
    """SPEC M2 done-when 5: compatibility sweeps still pass with the new primitives."""
    from pathlib import Path
    from ductus.lint import run_lints
    from ductus.loader import load_font
    font = load_font(Path(__file__).resolve().parents[1] / path)
    findings, _ = run_lints(font, {}, parse_sweep(sweep))
    assert not [f for f in findings if f.level == "error"], findings
