"""Primitive handles and per-glyph parameter overrides (SPEC 7)."""
from pathlib import Path

import pytest

from ductus.compile import compile_font
from ductus.handles import param_of
from ductus.loader import load_font
from ductus.overrides import Override, dump_overrides, parse_overrides
from ductus.params import Param

ROOT = Path(__file__).resolve().parents[1]


def test_param_of():
    p = Param(10, (0, 20), name="p")
    assert param_of(p) == (p, 1.0)
    assert param_of(p * 2) == (p, 2.0) and param_of(3 * p) == (p, 3.0)
    assert param_of(p / 4) == (p, 0.25)
    assert param_of(p + 1) is None and param_of(p * p) is None and param_of(5) is None


@pytest.fixture(scope="module")
def font():
    return load_font(ROOT / "examples/display_serif/font.py")


def test_handles_are_emitted_and_point_somewhere_useful(font):
    cf = compile_font(font)
    n = cf.glyphs["n"].to_json()
    params = {h["param"] for h in n["handles"]}
    assert {"serif_len", "n_join_at", "n_join_thin", "n_taper_len", "n_notch"} <= params
    for h in n["handles"]:
        assert abs((h["ax"] ** 2 + h["ay"] ** 2) - 1) < 1e-6 and h["per_unit"] != 0
    a = cf.glyphs["a"].to_json()
    assert {"a_top_at", "a_bot_thin", "a_lobe_thick", "ball_r"} <= {h["param"] for h in a["handles"]}


@pytest.mark.parametrize("param", ["serif_len", "n_join_at", "n_join_thin", "n_taper_len"])
def test_dragging_a_handle_moves_it_by_the_drag(font, param):
    """A handle dragged by d units along its axis, converted to a parameter
    change with per_unit, lands d units further along -- the contract the
    editor relies on."""
    base = [h for h in compile_font(font).glyphs["n"].handles if h["param"] == param][0]
    d = 12.0
    dv = d / base["per_unit"]
    moved = [h for h in compile_font(font, overrides={f"n.@{param}": Override(f"n.@{param}", dv=dv)})
             .glyphs["n"].handles if h["param"] == param][0]
    travelled = (moved["x"] - base["x"]) * base["ax"] + (moved["y"] - base["y"]) * base["ay"]
    assert travelled == pytest.approx(d, abs=1.0)


def test_glyph_param_override_is_local(font):
    ov = {"n.@serif_len": Override("n.@serif_len", dv=30)}
    a, b = compile_font(font), compile_font(font, overrides=ov)
    assert b.glyphs["n"].advance > a.glyphs["n"].advance             # n's serifs grew
    assert b.glyphs["h"].advance == a.glyphs["h"].advance             # h untouched
    assert "n.@serif_len" in b.overrides_used and not b.warnings
    stale = compile_font(font, overrides={"n.@no_such": Override("n.@no_such", dv=1)})
    assert any("n.@no_such" in w for w in stale.warnings)


def test_dv_round_trips():
    import yaml
    ovs = {"n.@serif_len": Override("n.@serif_len", dv=6.5)}
    assert parse_overrides(yaml.safe_load(dump_overrides(ovs))) == ovs
