"""Benchmark harness: metrics, report, fitter -- against references we
build ourselves, so the tests need no third-party font."""
from pathlib import Path

import pytest

from ductus.bench.metrics import score_glyph, translated
from ductus.bench.reference import Reference
from ductus.bench.run import fit_glyph, glyph_params, score_font
from ductus.compile import compile_font
from ductus.emit import build_otf

from .test_contours import BOWL_FONT


@pytest.fixture
def bowl_font(write_font):
    return write_font(BOWL_FONT + "\nfont.sidebearings = {'straight': param(40, range=(0, 120)), 'round': param(40, range=(0, 120))}\n")


def make_reference(font, params, path):
    cf = compile_font(font, params)
    return Reference(build_otf(cf, path)), cf


def test_identical_and_translated_glyphs_score_perfectly(bowl_font, tmp_path):
    ref, cf = make_reference(bowl_font, {}, tmp_path / "ref.otf")
    g = cf.glyphs["o"]
    s = score_glyph("o", translated(g.contours, 37), ref.glyph("o")[0], g.advance, ref.glyph("o")[1])
    assert s.iou > 0.995 and s.hausdorff < 0.75 and s.ref_dx == pytest.approx(37, abs=0.5)


def test_scores_degrade_with_shape_error(bowl_font, tmp_path):
    ref, _ = make_reference(bowl_font, {}, tmp_path / "ref.otf")
    near = score_font(compile_font(bowl_font, {"stem": 100}), ref, ["o"])["o"]
    far = score_font(compile_font(bowl_font, {"stem": 140, "fh": 0.8}), ref, ["o"])["o"]
    assert 1 > near.iou > far.iou
    assert near.hausdorff == pytest.approx(10, abs=1.5)  # wall 10 units thicker on each side
    assert far.hausdorff > near.hausdorff


def test_param_split_shape_vs_spacing(bowl_font):
    shape, spacing = glyph_params(bowl_font, "o", {})
    assert {"stem", "hair", "stress", "fh", "fv"} == set(shape)
    assert spacing == ["sidebearing_round"]


def test_fitter_recovers_known_parameters(bowl_font, tmp_path):
    target = {"stem": 120, "hair": 35, "fh": 0.68, "sidebearing_round": 70}
    ref, _ = make_reference(bowl_font, target, tmp_path / "ref.otf")
    r = fit_glyph(bowl_font, ref, "o", {}, maxiter=300, samples=64, restarts=2)
    assert r.before.iou < 0.95
    assert r.after.iou > 0.99 and r.after.advance_delta == 0
    assert r.params["stem"] == pytest.approx(120, abs=2)
    assert r.params["hair"] == pytest.approx(35, abs=2)


def test_report_and_overlay(bowl_font, tmp_path):
    from ductus.bench.report import append_rows, next_iteration, overlay_image
    ref, _ = make_reference(bowl_font, {"stem": 110}, tmp_path / "ref.otf")
    s = score_font(compile_font(bowl_font), ref, ["o"])["o"]
    docs = tmp_path / "docs"
    assert next_iteration(docs) == 1
    out = overlay_image(s, docs / "iterations" / "01" / "o.png")
    assert out.exists() and next_iteration(docs) == 2
    row = dict(glyph="o", iou=s.iou, hausdorff=s.hausdorff, mean_dist=s.mean_dist, advance_delta=s.advance_delta,
               overrides=0, magnitude=0.0, primitives="bowl", notes="test")
    text = append_rows(docs, 1, [row]).read_text()
    assert "| 01 |" in text and "`o`" in text and text.count("| iter |") == 1
    append_rows(docs, 2, [row])
    assert (docs / "representability.md").read_text().count("| iter |") == 1


def test_editor_reference_payload(bowl_font, tmp_path):
    from ductus.serve.server import Session
    ref_path = build_otf(compile_font(bowl_font, {"stem": 100}), tmp_path / "ref.otf")
    s = Session(bowl_font.source_path, tmp_path / "overrides.yaml", ref_path)
    R = s.payload()["data"]["reference"]["o"]
    assert 0.8 < R["iou"] < 1 and R["svg"].startswith("M") and R["heat"]
    assert all(len(p) == 3 for line in R["heat"] for p in line)
