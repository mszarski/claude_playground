"""tools/popoto_trade_study.py (sonar-3983): its variants are the shipped face builds with only the named fields changed, its
bound A is the depth planner's own figure, and it runs end to end on one case."""

import copy
import importlib.util
import json
import pathlib
import sys

import numpy as np

from gpusonar import buildspec as bs
from gpusonar import depthplan

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))


def tool():
    spec = importlib.util.spec_from_file_location("popoto_trade_study", ROOT / "tools" / "popoto_trade_study.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_a_variant_changes_only_what_it_names_and_no_shipped_file():
    t = tool()
    before = (ROOT / "configs" / "builds" / "popoto_face_2x32.yaml").read_bytes()
    shipped = bs.load("popoto_face_2x32")
    v = t.variant_spec(2, 64, "1k", 2.0)
    assert v["array"]["count"] == 64 and v["array"]["rows"] == 2 and v["array"]["pitch_m"] == shipped["array"]["pitch_m"]
    assert v["projector"]["band_hz"] == [4500.0, 5500.0] and v["projector"]["waveform"]["duration_s"] == 2.0
    assert v["projector"]["source_level_db_rms"] == 194.0                     # the level projector_selection.md pairs with 2 s
    for section in ("platform", "sync", "navigation", "operation", "sound_speed"):
        assert v[section] == shipped[section]
    d = bs.derive(v)
    assert d["per_row"] == [64, 64] and d["band_hz"] == [4500.0, 5500.0] and d["pulse_s"] == 2.0
    g = t.geometry(v, d)
    assert g["boards"] == 16 and not g["fits_face"] and abs(g["aperture_m"] - 9.6) < 1e-9          # 63 x 0.15 m does not fit 9 m
    assert (ROOT / "configs" / "builds" / "popoto_face_2x32.yaml").read_bytes() == before
    assert bs.load("popoto_face_2x32") == shipped


def test_the_bounds_and_credits_are_arithmetic_on_the_planners_parts():
    t = tool()
    noise, srr = np.array([[2.0, -1.0]]), np.array([[30.0, 5.0]])
    T, G = 11.8, 4.0
    a = t.margin(noise, srr, T, G, 0.0)
    b = t.margin(noise, srr, T, G, 10 * np.log10(3))
    assert np.all(b > a) and np.all(a < noise)                                  # reverberation costs; rows' gain against it gives some back
    assert np.allclose(t.margin(noise, None, T, G, 0.0), noise)                # nothing to scatter: the noise-only margin
    assert np.allclose(t.margin(noise, srr, T, G, 0.0, echo_db=1.9), t.margin(noise + 1.9, srr + 1.9, T, G, 0.0))
    # source level moves only the noise term: with the echo far over the noise the margin tends to the reverberation's ceiling
    assert abs(t.margin(noise + 60.0, srr, T, G, 0.0)[0, 1] - (srr[0, 1] - T - G)) < 1e-3


def test_quick_case_runs_and_bound_a_is_the_planners_figure(tmp_path):
    t = tool()
    out = tmp_path / "q"
    assert t.main(["--quick", "--site", "deep_no_bottom_return", "--class", "ship", "--out", str(out)]) == 0
    R = json.loads(out.with_suffix(".json").read_text())
    r = R["grid"]["results"]["3x32|2k|1s|197dB"]["deep_no_bottom_return"]["classes"]["ship"]
    spec = bs.load("popoto_face_3x32")
    env = next(e for e in bs.environments_of(spec) if e["name"] == "deep_no_bottom_return")
    ship = [c for c in bs.target_classes_of(spec) if c["name"] == "ship"]
    P = depthplan.plan(bs.derive(spec), env, ship, keep_grid=False)["classes"]["ship"]["recommended"]
    assert abs(r["A"]["margin_db"] - P["margin_db"]) < 1e-9 and r["A"]["own_depth_m"] == P["own_depth_m"] and r["A"]["reach_m"] == P["reach_m"]
    assert r["B"]["margin_db"] >= r["A"]["margin_db"] - 1e-9
    assert abs(r["A"]["credits"]["tone"]["margin_db"] - (r["noise_only"]["margin_db"] - 4.1)) < 1e-9
    assert r["A"]["limited_by"] == "noise" and r["met_12km"]["A"]
    assert R["answers"]["per_class_site"]["ship"]["deep_no_bottom_return"]["A"]["smallest"] == "3x32|2k|1s|197dB"
    assert copy.deepcopy(R["command"]).startswith("tools/popoto_trade_study.py --quick") and "--out" not in R["command"]
