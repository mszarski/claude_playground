import time

import pytest

from ductus.compile import compile_font
from ductus.emit import build_otf, validate_otf
from ductus.lint import check_compatibility, check_stems, parse_sweep, run_lints
from ductus.terminals import serifs


def test_sweep_stem_40_to_160_is_compatible(font):
    """Acceptance 4."""
    findings = check_compatibility(font, {}, parse_sweep(["stem=40,160"]))
    assert not [f for f in findings if f.level == "error"]


def test_incompatible_structure_is_caught(write_font):
    f = write_font("""
font = Font("T"); stem = param(80, range=(30, 180)); font.pen = Pen(width=stem)
@glyph("x")
def x(g):
    g.line((0, 0), (0, 500), name="a")
    if stem > 100:          # structure depending on a value: forbidden
        g.line((200, 0), (200, 500), name="b")
""")
    findings = check_compatibility(f, {}, parse_sweep(["stem=40,160"]))
    assert any(x.level == "error" for x in findings)


def test_params_change_every_glyph(font):
    """Acceptance 3: one change, all glyphs respond."""
    base = compile_font(font)
    for p, v in (("stem", 140), ("contrast", 0.8), ("xheight", 590)):
        other = compile_font(font, {p: v})
        changed = [g for g in base.glyphs if g != "space" and
                   base.glyphs[g].contours[0].pts != other.glyphs[g].contours[0].pts]
        expected = {"n", "h", "m", "o", "H", "O"} if p != "xheight" else {"n", "h", "m", "o"}
        assert expected <= set(changed), (p, changed)


def test_stems_match_declared_stem(font):
    for stem in (40, 100, 160):
        cf = compile_font(font, {"stem": stem})
        for g in cf.glyphs.values():
            assert check_stems(g, stem) == []


def test_lints_pass(font):
    findings, _ = run_lints(font, {}, sweep=parse_sweep(["stem=40,160"]))
    assert not [f for f in findings if f.level == "error"], findings


def test_slab_terminals(font):
    sans = compile_font(font).glyphs["n"]
    slab = compile_font(font, terminals=serifs.slab()).glyphs["n"]
    assert slab.bounds[2] - slab.bounds[0] > sans.bounds[2] - sans.bounds[0]
    findings, _ = run_lints(font, {}, sweep=[])
    assert not [f for f in findings if f.level == "error"]


def test_otf_is_valid(font, tmp_path):
    cf = compile_font(font)
    path = build_otf(cf, tmp_path / "t.otf")
    assert validate_otf(path) == []
    from fontTools.ttLib import TTFont
    f = TTFont(path)
    assert f.getBestCmap()[ord("n")] == "n" and f.getBestCmap()[32] == "space"
    assert f["hmtx"]["n"][0] == cf.glyphs["n"].advance


def test_compile_is_fast(font):
    compile_font(font)
    t = time.perf_counter()
    compile_font(font, {"stem": 120})
    assert (time.perf_counter() - t) < 0.3


def test_unknown_param_is_an_error(font):
    with pytest.raises(Exception, match="unknown parameter"):
        compile_font(font, {"stme": 3})
