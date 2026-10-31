import pytest

from ductus.compile import compile_font
from ductus.lint import run_lints
from ductus.overrides import Override, add_delta, dump_overrides, load_overrides, parse_overrides


def test_yaml_round_trip(tmp_path):
    p = tmp_path / "overrides.yaml"
    p.write_text("n.arch.shoulder:      { dx: 12, dy: 0 }\n"
                 "m.arch2.shoulder_out: { dx: -6, dy: 4, stage: outline }\n")
    ovs = load_overrides(p)
    assert ovs["n.arch.shoulder"] == Override("n.arch.shoulder", 12, 0, "skeleton")
    assert ovs["m.arch2.shoulder_out"].stage == "outline"
    assert parse_overrides(__import__("yaml").safe_load(dump_overrides(ovs))) == ovs
    add_delta(p, "n.arch.shoulder", 3, -2)
    assert load_overrides(p)["n.arch.shoulder"].dx == 15


@pytest.mark.parametrize("stem", [40, 80, 160])
def test_shoulder_override_survives_stem_change(font, stem):
    """Acceptance 5: a delta on n.arch.shoulder lands relative to the
    generated shoulder at any stem width, and the glyph stays sane."""
    ov = {"n.arch.shoulder": Override("n.arch.shoulder", 12, -5)}
    base = compile_font(font, {"stem": stem}).glyphs["n"]
    moved = compile_font(font, {"stem": stem}, ov).glyphs["n"]
    bx, by = base.points["n.arch.shoulder"]
    mx, my = moved.points["n.arch.shoulder"]
    assert (mx - bx, my - by) == pytest.approx((12, -5), abs=1.01)  # spacing shift rounds
    # the shoulder stays between the stems and under the x-height+overshoot
    stem_x = moved.points["n.stem.top"][0]
    leg_x = moved.points["n.arch.leg"][0]
    assert stem_x < mx < leg_x
    assert moved.bounds[3] <= base.bounds[3] + 1
    assert len(moved.contours) == len(base.contours) == 1
    assert moved.raw_signature == base.raw_signature


def test_outline_override_moves_both_sides(font):
    ov = {"m.arch2.shoulder_out": Override("m.arch2.shoulder_out", -6, 4, "outline")}
    a = compile_font(font).glyphs["m"].outline_points
    b = compile_font(font, overrides=ov).glyphs["m"].outline_points
    for side in (".l", ".r"):
        k = "m.arch2.shoulder_out" + side
        assert b[k][0] - a[k][0] == pytest.approx(-6, abs=1.01)
        assert b[k][1] - a[k][1] == pytest.approx(4)


def test_stale_override_is_reported_not_dropped(font):
    ovs = {"n.arch.elbow": Override("n.arch.elbow", 5, 0)}
    cf = compile_font(font, overrides=ovs)
    assert any("n.arch.elbow" in w for w in cf.warnings)
    findings, _ = run_lints(font, ovs, sweep=[])
    assert any(f.check == "overrides" and f.level == "error" for f in findings)


def test_overrides_are_not_inherited(font):
    ov = {"n.arch.shoulder": Override("n.arch.shoulder", 30, 0)}
    a = compile_font(font).glyphs["h"].points["h.arch.shoulder"]
    b = compile_font(font, overrides=ov).glyphs["h"].points["h.arch.shoulder"]
    assert a == b
