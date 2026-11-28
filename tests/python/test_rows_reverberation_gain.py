"""What stacked rows gain against the site's reverberation: an opt-in of the depth planner, the performance surface
term and the coverage picture (`reverberation_rows_gain`; depthplan.rows_gain_against_reverberation). The default is
bound A, 0 dB (D-227's assumption: reverberation arrives by the echo's own multipath; ours), and every default output
is as before.

The figures checked against are the trade study's (docs/design/popoto_trade_study.json, `rows_measurement`; ours,
simulated with the bistatic reverberation generator, one site, 3 pings): today's chain gains 2.47 dB (2 rows) and
3.38 dB (3 rows) against the shelf's reverberation, and with that gain the +5 dB class's reach from the 3 x 32 face at
its recommended own depth moves from 2.56 km (bound A) to 8.02 km. The gain is an input, not physics of the code.
"""

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from gpusonar import buildspec as bs
from gpusonar import coverage, depthplan
from gpusonar import reverb_budget as rb

ROOT = Path(__file__).resolve().parents[2]
STUDY = json.loads((ROOT / "docs" / "design" / "popoto_trade_study.json").read_text())["rows_measurement"]
SHIPPED_PLANS = json.loads((ROOT / "docs" / "design" / "popoto_depth_planner.json").read_text())["builds"]
SHELF = "shelf_100m_sand"


@pytest.fixture(scope="module")
def shelf3():
    """The 3 x 32 face over the shelf: the default plan and the plan with the study's measured gain (one cache: the
    factor and the reverberation level do not depend on the rows' gain)."""
    # D-256: the shipped face sends the 2 s hyperbolic sweep since; the study's figures held here (docs/design/popoto_trade_study.json,
    # `shelf_with_measured_gain`, and the reaches below) were made at the grid's 1 s linear sweep, which is stated here
    spec = copy.deepcopy(bs.load("popoto_face_3x32"))
    spec["projector"]["waveform"] = {"type": "lfm", "duration_s": 1.0, "window_alpha": 0.1}
    d = bs.derive(spec)
    env = next(e for e in bs.environments_of(spec) if e["name"] == SHELF)
    classes = bs.target_classes_of(spec, range_m=12000.0)
    cache = {}
    g = STUDY["subsets"]["3"]["chain_gain_db"]
    plans = {"default": depthplan.plan(d, env, classes, cache=cache),
             "none": depthplan.plan(d, env, classes, cache=cache, reverberation_rows_gain="none"),
             "measured": depthplan.plan(d, env, classes, cache=cache, reverberation_rows_gain=g),
             "incoherent": depthplan.plan(d, env, classes, cache=cache, reverberation_rows_gain="incoherent")}
    return d, env, classes, plans, g


def test_settings_resolve():
    det3 = {"rows": 3, "rows_gain_db": 3.99}
    det1 = {"rows": 1, "rows_gain_db": 0.0}
    assert depthplan.rows_gain_against_reverberation(None, det3)[0] == 0.0
    assert depthplan.rows_gain_against_reverberation("none", det3)[0] == 0.0
    assert depthplan.rows_gain_against_reverberation("incoherent", det3)[0] == 3.99
    assert depthplan.rows_gain_against_reverberation(2.5, det3)[0] == 2.5
    assert depthplan.rows_gain_against_reverberation(2.5, det1)[0] == 0.0           # one row: nothing to gain
    assert depthplan.rows_gain_against_reverberation("incoherent", det1)[0] == 0.0
    for bad in ("coherent", -1.0, float("nan"), True, [3.0]):
        with pytest.raises(ValueError, match="reverberation_rows_gain"):
            depthplan.rows_gain_against_reverberation(bad, det3)


def test_incoherent_is_the_detectors_own_rows_gain():
    """`incoherent` = buildspec.detection's rows gain (the rows' reverberation as independent looks, as the noise):
    3.99 dB for three rows of 32 and 2.54 for two (the study's `chain_rows_gain_db_against_noise`), below bound B's
    10 log10(rows), and above the measured 3.38 / 2.47 dB (the rows' reverberation is partly correlated)."""
    for rows, spec in ((3, "popoto_face_3x32"), (2, "popoto_face_2x32")):
        det = bs.detection(bs.derive(bs.load(spec)))
        g = depthplan.rows_gain_against_reverberation("incoherent", det)[0]
        assert g == pytest.approx(STUDY["chain_rows_gain_db_against_noise"][str(rows)], abs=0.01)
        assert STUDY["subsets"][str(rows)]["chain_gain_db"] < g < 10.0 * np.log10(rows)


@pytest.fixture(scope="module")
def shelf3_shipped():
    """The 3 x 32 face as shipped (D-256: the 2 s hyperbolic sweep), the default plan and "none": what
    docs/design/popoto_depth_planner.json holds."""
    spec = bs.load("popoto_face_3x32")
    d = bs.derive(spec)
    env = next(e for e in bs.environments_of(spec) if e["name"] == SHELF)
    classes = bs.target_classes_of(spec, range_m=12000.0)
    cache = {}
    return {"default": depthplan.plan(d, env, classes, cache=cache), "none": depthplan.plan(d, env, classes, cache=cache, reverberation_rows_gain="none")}


def test_default_plan_is_unchanged(shelf3_shipped):
    """Bound A by default: the plan with no setting, with "none", and the shipped docs/design/popoto_depth_planner.json
    agree (the shipped file's numbers to 1e-9 relative), and the reverberation block states 0 dB as before."""
    P = shelf3_shipped
    a, n = P["default"], P["none"]
    assert a["reverberation"]["rows_gain_against_reverberation_db"] == 0.0
    assert "rows_gain_against_reverberation" not in a["reverberation"]
    shipped = SHIPPED_PLANS["popoto_face_3x32"]["plans"][SHELF]
    for name, c in a["classes"].items():
        np.testing.assert_array_equal(c["margin_worst_db"], n["classes"][name]["margin_worst_db"])
        r, s = c["recommended"], shipped["classes"][name]["recommended"]
        for k in ("own_depth_m", "margin_db", "reach_m", "margin_noise_only_db", "reach_noise_only_m", "factor_db"):
            assert r[k] == pytest.approx(s[k], rel=1e-9, abs=1e-12), (name, k)
        assert r["limited_by"] == s["limited_by"]
        np.testing.assert_allclose(c["margin_at_range_worst_db"], shipped["classes"][name]["margin_at_range_worst_db"], rtol=1e-9, atol=1e-12)


def test_measured_gain_moves_the_reach_as_the_study_reports(shelf3):
    """With the study's measured 3.38 dB the +5 dB class's reach from the 3 x 32 face over the shelf moves from bound A's
    2.56 km to 8.02 km at the same recommended own depth (65 m), and its margin at 12 km from -4.62 to -3.92 dB
    (docs/design/popoto_trade_study.json, shelf_with_measured_gain; ours). The ship, noise-limited, barely moves."""
    _, _, _, P, g = shelf3
    want = STUDY["shelf_with_measured_gain"]["3x32"]
    for bound, plan in (("A", P["default"]), ("chain_measured", P["measured"])):
        r = plan["classes"]["xl_auv"]["recommended"]
        w = want["xl_auv"][bound]
        assert r["own_depth_m"] == w["own_depth_m"] and r["reach_m"] == w["reach_m"], (bound, r["reach_m"], w)
        assert r["margin_db"] == pytest.approx(w["margin_db"], abs=0.01)
    assert P["default"]["classes"]["xl_auv"]["recommended"]["reach_m"] == pytest.approx(2560.0)
    assert P["measured"]["classes"]["xl_auv"]["recommended"]["reach_m"] == pytest.approx(8020.0)
    assert P["measured"]["reverberation"]["rows_gain_against_reverberation_db"] == g
    assert P["measured"]["reverberation"]["rows_gain_against_reverberation"]["setting"] == g
    ship_a, ship_m = (P[k]["classes"]["ship"]["recommended"]["margin_db"] for k in ("default", "measured"))
    assert 0.0 <= ship_m - ship_a <= 1.0 and ship_m == pytest.approx(want["ship"]["chain_measured"]["margin_db"], abs=0.01)
    # never above noise-only, never below bound A, at every own depth and range
    for name, c in P["measured"]["classes"].items():
        m, a = np.asarray(c["margin_worst_db"]), np.asarray(P["default"]["classes"][name]["margin_worst_db"])
        assert np.all(m >= a - 1e-12)
        r = c["recommended"]
        assert r["margin_db"] <= r["margin_noise_only_db"] + 1e-12
    inc = P["incoherent"]["classes"]["xl_auv"]["recommended"]
    assert inc["margin_db"] >= P["measured"]["classes"]["xl_auv"]["recommended"]["margin_db"]


def test_surface_term_takes_the_same_setting(shelf3):
    """buildspec.performance's surface term: the default as before; with g, one row's SNR is combine(SNR + G, SRR + g) - G
    and srr_db stays one row's."""
    d, env, classes, _, g = shelf3
    xl = [c for c in classes if c["name"] == "xl_auv"]
    a = bs.performance(d, environment=env, own_depth_m=65.0, classes=xl)["surface"]
    n = bs.performance(d, environment=env, own_depth_m=65.0, classes=xl, reverberation_rows_gain="none")["surface"]
    b = bs.performance(d, environment=env, own_depth_m=65.0, classes=xl, reverberation_rows_gain=g)["surface"]
    assert a == n and "rows_gain_against_reverberation_db" not in a and b["rows_gain_against_reverberation_db"] == g
    G = bs.detection(d)["rows_gain_db"]
    ca, cb = a["classes"]["xl_auv"], b["classes"]["xl_auv"]
    assert ca["srr_db"] == cb["srr_db"] and ca["snr_noise_only_db"] == cb["snr_noise_only_db"]
    for r, srr in ca["srr_db"].items():
        n_db = ca["snr_noise_only_db"][r]
        assert cb["snr_db"][r] == pytest.approx(float(rb.with_reverberation(n_db + G, srr + g) - G), abs=1e-9)
        assert cb["snr_db"][r] >= ca["snr_db"][r]


def test_coverage_picture_takes_the_gain():
    """The C++ coverage picture: `rows_gain_against_reverberation_db` adds g to the echo over reverberation; absent or 0
    the field is bit for bit the default; a negative or non-number value is refused."""
    sand = {"speed": 1650.0, "density": 1.9, "attenuation": 0.8}
    cfg = {"budget": {"source_level_db": 197.0, "noise_spectrum_db": 51.5, "array_gain_db": 15.0, "absorption_db_per_km": 0.32},
           "bottom": sand, "reverberation": {"items": [{"mu_db": -27.0}], "frequency_hz": 5000.0, "rows_gain_db": 3.99}}
    sonar = {"f_centre": 5000.0, "bandwidth": 2000.0, "pulse": 1.0, "beam_integral": 0.05}
    w, r = coverage.water([0.0, 100.0], [1500.0, 1500.0]), [2000.0, 6000.0, 12000.0]
    f0 = coverage.field(w, 100.0, 50.0, [40.0], r, cfg, sonar)
    cfg["reverberation"]["rows_gain_against_reverberation_db"] = 0.0
    fz = coverage.field(w, 100.0, 50.0, [40.0], r, cfg, sonar)
    np.testing.assert_array_equal(f0["snr_0db_target_db"], fz["snr_0db_target_db"])
    cfg["reverberation"]["rows_gain_against_reverberation_db"] = 3.38
    f3 = coverage.field(w, 100.0, 50.0, [40.0], r, cfg, sonar)
    n, s0 = f0["snr_noise_only_0db_target_db"][0], f0["snr_0db_target_db"][0]
    srr = -10.0 * np.log10(10.0 ** (-(s0 + 3.99) / 10.0) - 10.0 ** (-(n + 3.99) / 10.0))      # one row's, from the default
    assert np.allclose(f3["snr_0db_target_db"][0], rb.with_reverberation(n + 3.99, srr + 3.38) - 3.99, atol=1e-6)
    assert np.all(f3["snr_0db_target_db"][0] > s0) and np.all(f3["snr_0db_target_db"][0] <= n + 1e-9)
    assert coverage.cpp_options(cfg)["reverb_rows_gain_against_reverb_db"] == 3.38
    for bad in (-1.0, "incoherent"):
        cfg["reverberation"]["rows_gain_against_reverberation_db"] = bad
        with pytest.raises(ValueError, match="rows_gain_against_reverberation_db"):
            coverage.cpp_options(cfg)


def test_a_builds_coverage_block_writes_the_setting_only_when_asked():
    spec = bs.load("popoto_face_3x32")
    default = coverage.of_build(spec, {"environment": SHELF})
    assert "rows_gain_against_reverberation_db" not in default["reverberation"]
    assert coverage.of_build(spec, {"environment": SHELF, "reverberation_rows_gain": "none"}) == default
    inc = coverage.of_build(spec, {"environment": SHELF, "reverberation_rows_gain": "incoherent"})
    assert inc["reverberation"]["rows_gain_against_reverberation_db"] == pytest.approx(bs.detection(bs.derive(spec))["rows_gain_db"], abs=0.01)
    assert coverage.of_build(spec, {"environment": SHELF, "reverberation_rows_gain": 3.38})["reverberation"]["rows_gain_against_reverberation_db"] == 3.38
    one = coverage.of_build(bs.load("popoto_face_1x32"), {"environment": SHELF, "reverberation_rows_gain": 3.38})
    assert "rows_gain_against_reverberation_db" not in one["reverberation"]
    with pytest.raises(ValueError, match="reverberation_rows_gain"):
        coverage.of_build(spec, {"environment": SHELF, "reverberation_rows_gain": "coherent"})
    with pytest.raises(ValueError, match="environment"):
        coverage.of_build(spec, {"environment": SHELF, "rows_gain": 3.0})


def test_reach_limited_by_says_what_limits_the_reach(shelf3):
    """D-227's defect: `limited_by` is judged at the range of interest (12 km), so it reads "noise" for the +5 dB class over
    the shelf although reverberation cut its reach from 9.58 km (noise only) to 2.56 km. `reach_limited_by` judges the
    same rule (the term costs more than 3 dB) at the reported reach. The existing fields are as before (the test above)."""
    d, env, classes, P, _ = shelf3
    xl = P["default"]["classes"]["xl_auv"]["recommended"]
    assert xl["limited_by"] == "noise" and xl["reach_limited_by"] == "reverberation"
    assert xl["reach_m"] == 2560.0 and xl["reach_noise_only_m"] == 9580.0
    assert P["default"]["classes"]["ship"]["recommended"]["reach_limited_by"] == "noise"      # 12.7 km, the grid's end
    noise = depthplan.plan(d, env, classes, cache={}, reverberation=False)
    assert all("reach_limited_by" not in c["recommended"] and "limited_by" not in c["recommended"] for c in noise["classes"].values())
    rr = np.asarray(P["default"]["ranges_m"])
    for key in ("default", "measured"):
        for name, c in P[key]["classes"].items():
            r = c["recommended"]
            k_own = P[key]["own_depths_m"].index(r["own_depth_m"])
            k = int(np.argmin(np.abs(rr - r["reach_m"]))) if r["reach_m"] > 0 else 0
            cost = noise["classes"][name]["margin_worst_db"][k_own][k] - c["margin_worst_db"][k_own][k]
            assert r["reach_limited_by"] == ("reverberation" if cost > 3.0 else "noise"), (key, name, cost)
