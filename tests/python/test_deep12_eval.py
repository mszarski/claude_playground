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
    for bad in ("ss6_lfm1", "ss3_lfm2", "ss3_hfm1_max"):
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
