"""tools/deep12_eval.py: the deep12 cases set against the planner (issue sonar-rctz). The predictions are the trade study's own
functions at the scene's geometry (nothing typed); the summary reads doppler_bank_eval's JSON as the deep12 sub-run writes it."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

import deep12_eval  # noqa: E402
import popoto_trade_study as T  # noqa: E402


def test_cases_parse_and_unknown_ones_are_refused():
    k = deep12_eval.parse_case("ss5_lfm4p200_cfar")
    assert (k["sea_state"], k["wave"], k["pulse_s"], k["source_level_db"], k["bank"]) == (5, "lfm", 4.0, 200.0, True)
    assert deep12_eval.parse_case("ss3_lfm1")["design"].startswith("baseline")
    for bad in ("ss6_lfm1", "ss3_cw1", "ss3_lfm0", "ss3_hfm1_max"):        # ss3_lfm2 is a case since DM.5 (any pulse)
        with pytest.raises(ValueError):
            deep12_eval.parse_case(bad)


def test_the_prediction_is_the_trade_studys_at_the_scenes_geometry():
    """The point (+25 dB, 50 m, range rate 0) at sea state 3 with the baseline: predict_point of the shipped three-row face at the
    deep site with the bubble law, own depth 225 m; the table's cell is the grid's own; the planner's margin at the scene's
    target depth is within 1 dB of the table's (worst over the class's depths, at the class's recommended own depth)."""
    from gpusonar import buildspec as bs, environment
    P = deep12_eval.predict("ss3_lfm1", threshold_db=9.45)
    pt = P["targets"]["point"]
    env = T.tx_sites(environment.load_list("illustrative"))["deep_no_bottom_return__ss3__apl_1994"]
    want = T.predict_point(bs.derive(T.variant_spec(3, 32, "2k", 1.0, 197.0)), env, 25.0, 50.0, 225.0, pt["range_m"])
    assert P["environment"] == env["name"] and P["own_depth_m"] == 225.0
    assert pt["planner_margin_db"] == pytest.approx(want["margin_db"], abs=1e-9)
    assert pt["predicted_chain_margin_db"] == pytest.approx(want["chain_snr_expected_db"] - 9.45, abs=1e-9)
    assert pt["closed_form_loss_db"] == pytest.approx(0.0, abs=1e-6) and pt["range_rate_m_s"] == pytest.approx(0.0, abs=1e-9)
    grid = json.loads((ROOT / "docs" / "design" / "popoto_trade_study.json").read_text())["transmit"]["grid"]["results"]
    assert pt["table_db"] == grid["deep_no_bottom_return__ss3__apl_1994"]["1 s 197 dB"]["large_submerged"]["margin_db"]
    assert abs(pt["planner_margin_db"] - pt["table_db"]) < 1.0
    # a mover pays the linear sweep's closed form; the hyperbolic sweep almost nothing; the bank nothing
    closing = P["targets"]["closing"]
    assert closing["closed_form_loss_db"] == pytest.approx(T.sweep_mismatch_loss_db(closing["range_rate_m_s"], 2000.0, 1.0, 1500.0))
    assert deep12_eval.predict("ss3_hfm1")["targets"]["closing"]["closed_form_loss_db"] < 0.3
    assert deep12_eval.predict("ss3_lfm1_cfar")["targets"]["closing"]["closed_form_loss_db"] == 0.0


def _result(margins, bank=False):
    side = {"targets": {lb: {"pings": [{"snr_db": None if m is None else m + 9.45, "threshold_db": 9.45, "margin_db": m} for m in ms],
                             "detected": sum(m is not None for m in ms), "tracked": True, "kind": "mover"} for lb, ms in margins.items()}}
    eng = {"off": side, **({"on": side} if bank else {})}
    return {"engines": {"cpp": eng, "python": eng}, "background": {"default": {"pings": 4, "off": 0, "on": 1}, "3.0": {"pings": 4, "off": 9, "on": 7}}}


def test_the_summary_reads_the_sub_runs_json_and_puts_the_worst_difference_first(tmp_path):
    labels = ("closing", "oblique", "crossing", "ship", "point")
    (tmp_path / "deep12_ss3_lfm1.json").write_text(json.dumps(_result({lb: [3.0, None, 2.0] for lb in labels})))
    (tmp_path / "deep12_ss5_lfm4p200_cfar.json").write_text(json.dumps(_result({lb: [1.0, 1.0, 1.0] for lb in labels}, bank=True)))
    S = deep12_eval.summarise(tmp_path)
    assert len(S["rows"]) == 10
    diffs = [r["difference_db"] for r in S["rows"]]
    assert diffs == sorted(diffs)
    r = next(r for r in S["rows"] if r["case"] == "ss3_lfm1" and r["target"] == "point")
    assert r["measured_detected"] == 2 and r["measured_margin_db"] == pytest.approx(2.5) and r["measured_engines_same_detections"]
    assert r["difference_db"] == pytest.approx(2.5 - r["planner_predicted_mover_db"])
    assert S["background_default_budget"] == {"ss3_lfm1": {"pings": 4, "false_detections": 0, "at_3_a_ping": 9},
                                              "ss5_lfm4p200_cfar": {"pings": 4, "false_detections": 1, "at_3_a_ping": 7}}
    text = deep12_eval.markdown(S)
    assert text.count("\n| ss") == 12 and "sea-state-5 alternate" in text


def test_verdicts_one_highlight_and_the_detector_scales():
    """PASS at half the pings or more (the design's Pd 0.5 a ping); a hull's one-highlight prediction is 10 log10(5) under its own;
    the detector tests the single sample and a 2 m cell only in this window, shorter than every resolved hull's extent."""
    P = deep12_eval.predict("ss3_hfm4p200")["targets"]
    for label in ("closing", "oblique", "ship"):
        assert P[label]["highlights"] == 5 and P[label]["predicted_one_highlight_db"] == pytest.approx(P[label]["predicted_mover_db"] - 6.9897, abs=1e-3)
        assert P[label]["highlight_range_extent_m"] > 20.0
    assert P["crossing"]["highlight_range_extent_m"] < 1.0 and P["point"]["highlights"] == 1
    scales = deep12_eval.detector_scales(1.0)
    assert [s["length_m"] for s in scales] == [0.375, 2]
    labels = ("closing", "oblique", "crossing", "ship", "point")
    S = {"rows": []}
    for n, case in ((3, "ss3_lfm1"), (2, "ss5_hfm4p200")):
        for lb in labels:
            S["rows"].append({"case": case, "target": lb, "verdict": "PASS" if 2 * n >= 6 else "FAIL", "measured_detected": n, "measured_pings": 6,
                              "measured_margin_db": 1.0, "planner_predicted_mover_db": 0.5, "planner_class": "large_submerged"})
    A = deep12_eval.acceptance(S)
    assert A["baseline"]["accepted"] and not A["alternate_hfm"]["accepted"] and "alternate" not in A


def test_the_committed_measurement_states_both_designs_verdicts():
    """docs/tasks/doppler_deep12_eval.json (the L4 run): the point agrees with the planner within 1 dB wherever it is detected on
    every ping; the resolved hulls fall 2 to 7 dB under their own prediction with the filters that take no Doppler loss."""
    S = json.loads((ROOT / "docs" / "tasks" / "doppler_deep12_eval.json").read_text())
    A = S["acceptance"]
    assert not A["baseline"]["accepted"] and not A["alternate"]["accepted"]
    assert [lb for lb, v in A["baseline"]["targets"].items() if v["verdict"] == "FAIL"] == ["closing", "oblique", "ship"]
    for r in S["rows"]:
        if r["target"] == "point" and r["measured_detected"] == 6:
            assert abs(r["difference_db"]) < 1.0, r["case"]
        if r["target"] in ("oblique", "closing") and r["measured_detected"] == 6 and (r["case"].endswith("_cfar") or "hfm" in r["case"]):
            assert -7.0 < r["difference_db"] < -2.0, (r["case"], r["target"])
    assert all(b["false_detections"] == 0 for b in S["background_default_budget"].values())


def test_any_pulse_and_level_parse_and_the_window_holds_the_pulse():
    """DM.5's cases: 2 s and 8 s, 200 and 203 dB; the old names mean what they meant; the receive window takes the pulse."""
    from gpusonar.config import load_scenario_config
    from gpusonar.sonar import SonarModel
    for name, want in (("ss3_lfm1", ("lfm", 1.0, 197.0, False)), ("ss5_hfm4p200", ("hfm", 4.0, 200.0, False)), ("ss3_hfm2", ("hfm", 2.0, 197.0, False)),
                       ("ss3_lfm2_cfar", ("lfm", 2.0, 197.0, True)), ("ss5_hfm8p203", ("hfm", 8.0, 203.0, False)), ("ss3_hfm1p200", ("hfm", 1.0, 200.0, False))):
        k = deep12_eval.parse_case(name)
        assert (k["wave"], k["pulse_s"], k["source_level_db"], k["bank"]) == want, name
    assert deep12_eval.parse_case("ss3_lfm1")["design"].startswith("baseline")
    with pytest.raises(ValueError):
        deep12_eval.parse_case("ss3_hfm2_cfar")                 # the bank takes a linear sweep only
    cfg = load_scenario_config("popoto_12km_deep_sea_state_5", expand=False)
    for T in (2.0, 8.0):
        s = SonarModel.load(cfg["sonar"], {**cfg["sonar_overrides"], "transmit": {"waveform": {"duration": T, "type": "sampled", "source": "hfm"}}})
        start, n = s.window(1500.0)
        assert start == pytest.approx(2 * 11300.0 / 1500.0) and n / s.sample_rate == pytest.approx(2 * 1400.0 / 1500.0 + T, abs=1e-3)


def test_the_predictions_were_written_before_the_run_and_are_the_tools():
    """docs/tasks/deep12_round3_predictions.json is what --predict-cases computes now (nothing typed), one entry per DM.5 case."""
    got = json.loads((ROOT / "docs" / "tasks" / "deep12_round3_predictions.json").read_text())
    assert [c["case"] for c in got["cases"]] == list(deep12_eval.ROUND3)
    for c in got["cases"]:
        now = deep12_eval.prediction(c["case"])
        assert c["point_predicted_margin_db"] == pytest.approx(now["point_predicted_margin_db"], abs=1e-6)
        assert c["bracket_verdict_resolved_hulls"] == now["bracket_verdict_resolved_hulls"]
        k = deep12_eval.parse_case(c["case"])
        assert c["blind_range_m"] == pytest.approx(750.0 * k["pulse_s"])
        assert (c["range_bias_m_per_m_s"] is None) == (k["wave"] == "lfm")
        if c["range_bias_m_per_m_s"] is not None:
            assert c["range_bias_m_per_m_s"] == pytest.approx(3.0 * k["pulse_s"])
    by = {c["case"]: c for c in got["cases"]}
    assert by["ss3_hfm2"]["single_listed_projector"] and not by["ss5_hfm8p200"]["single_listed_projector"] and not by["ss5_hfm4p203"]["single_listed_projector"]


def test_the_candidates_are_merged_with_their_predictions_beside_the_verdicts():
    """The L4 run of DM.5's cases (job 6b0a19496c5d61fff18a650c): each candidate carries the prediction written before it ran; where
    the bracket said held or not held, the measurement agrees (DM.6)."""
    S = json.loads((ROOT / "docs" / "tasks" / "doppler_deep12_eval.json").read_text())
    cand = {v["case"]: v for k, v in S["acceptance"].items() if k.startswith("candidate_")}
    assert set(cand) == set(deep12_eval.ROUND3)
    for case, v in cand.items():
        p = v["predicted_before_run"]
        if p["bracket_verdict_resolved_hulls"] == "held":
            assert p["resolved_hulls_held_measured"], case
        elif p["bracket_verdict_resolved_hulls"] == "not held":
            assert not p["resolved_hulls_held_measured"], case
    assert not cand["ss4_hfm2"]["accepted"] and cand["ss3_hfm2"]["accepted"] and not cand["ss5_hfm8p200"]["accepted"]
    rows = [r for r in S["rows"] if r["case"] in cand and r["target"] in ("closing", "oblique")]
    assert all(r["predicted_before_run"] is not None for r in rows)
