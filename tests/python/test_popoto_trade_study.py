"""tools/popoto_trade_study.py (sonar-3983): its variants are the shipped face builds with only the named fields changed, its
bound A is the depth planner's own figure, and it runs end to end on one case."""

import copy
import importlib.util
import json

import pytest
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


# --- transmit options by sea state (--transmit; the owner, 2026-11-27) --------------------------------------------------

COMMITTED = ROOT / "docs" / "design" / "popoto_trade_study.json"


def _peak_loss_db(r, band, duration, c):
    """The sweep's peak loss by direct quadrature (independent of the library): phi = pi B T |r| / c, M = max over x0 of
    |integral exp(i 4 phi (u - x0)^2) du| (tests/python/test_doppler_bank.py's closed form, ours)."""
    phi = np.pi * band * duration * abs(r) / c
    u = np.linspace(-0.5, 0.5, 40001)
    best = max(abs(np.trapezoid(np.exp(4j * phi * (u - x0) ** 2), u)) for x0 in np.linspace(0.0, 0.8, 641))
    return float(-20.0 * np.log10(best))


def test_transmit_options_and_their_costs_are_their_formulas():
    t = tool()
    opts = t.tx_options()
    e = [o["energy_db"] for o in opts]
    assert e == sorted(e) and sum(o["shipped"] for o in opts) == 1 and sum(o["paired"] for o in opts) == 1
    assert {(o["pulse_s"], o["source_level_db"]) for o in opts} >= {(p, 197.0) for p in (1.0, 2.0, 4.0, 8.0)} | {(2.0, 194.0)}
    for o in opts:
        assert abs(o["energy_db"] - (o["source_level_db"] + 10 * np.log10(o["pulse_s"]))) < 1e-12
    for p, lv in ((1.0, 197.0), (8.0, 197.0), (2.0, 194.0)):
        d = bs.derive(t.variant_spec(3, 32, "2k", p, lv))
        k = t.tx_costs(d)
        c, B = d["sound_speed"], 2000.0
        assert abs(k["blind_range_m"] - c * p / 2) < 1e-9 and abs(k["range_resolution_m"] - c / (2 * B)) < 1e-12        # 750 m a second; 0.375 m at any T
        assert abs(k["ping_interval_s"] - (2 * 12700.0 / c + p)) < 1e-9 and abs(k["duty"] - p / k["ping_interval_s"]) < 1e-12
        assert abs(k["doppler_tolerance_m_s"] - c / (2 * B * p)) < 1e-12
        assert k["bank_hypotheses_8_m_s"] == 2 * int(np.ceil(8.0 / (c / (2 * B * p)) - 1e-9)) + 1
        for r in (0.4, 2.0, 8.0):
            assert abs(k["mismatch_loss_db"][f"{r:g}"] - _peak_loss_db(r, B, p, c)) < 0.05, (p, r)
            assert abs(k["mismatch_smear_m"][f"{r:g}"] - 2 * r * p) < 1e-12
    assert bs.derive(t.variant_spec(3, 32, "2k", 1.0, 197.0))["source_level_db"] == 197.0      # the shipped option is the shipped build's transmitter
    k1 = t.tx_costs(bs.derive(t.variant_spec(3, 32, "2k", 1.0, 197.0)))
    assert k1["projector"]["single_units"]                                                    # 197 dB at 5.6 % duty: listed units do it (projector_selection.md Sec. 3.2)
    k3 = t.tx_costs(bs.derive(t.variant_spec(3, 32, "2k", 1.0, 203.0)))
    assert not k3["projector"]["single_units"] and k3["projector"]["words"].startswith("no source identified")


def test_transmit_grid_moves_by_energy_where_noise_limits_and_agrees_with_the_planner():
    """The deep site at sea state 4 with the bubble law, three options (under a minute): the shipped column is the existing study's
    figure; noise-limited margins move by the level step and by 10 log10 of the pulse ratio (plus the detector's threshold change
    for the shorter window); two cells against depthplan.plan and buildspec.performance called directly."""
    t = tool()
    spec = bs.load("popoto_face_3x32")
    sites, classes = bs.environments_of(spec), bs.target_classes_of(spec, range_m=12000.0)
    envs = t.tx_sites(sites)
    pick = {k: envs[k] for k in ("deep_no_bottom_return__ss4__apl_1994",)}
    opts = [o for o in t.tx_options() if o["id"] in ("1 s 197 dB", "4 s 197 dB", "1 s 203 dB")]
    T = t.run_transmit_study(sites, classes, envs=pick, options=opts, log=lambda *a: None)
    res = T["grid"]["results"]
    R = json.loads(COMMITTED.read_text())
    sens = R["sensitivity"]["results"]["deep_no_bottom_return_sea_state_5"]["sea_state_4(9.8 m/s, from 12.6)"]["by_geometry"]["3x32"]
    deep = res["deep_no_bottom_return__ss4__apl_1994"]
    for c in ("large_submerged", "ship"):
        assert abs(deep["1 s 197 dB"][c]["margin_db"] - sens[c]["A"]["margin_db"]) < 2e-3          # nothing moved: the sensitivity table's sea state 4
        a, b, lv = deep["1 s 197 dB"][c], deep["4 s 197 dB"][c], deep["1 s 203 dB"][c]
        assert abs((lv["noise_only_db"] - a["noise_only_db"]) - 6.0) < 1e-6                         # the level step, exactly
        assert abs((b["noise_only_db"] - a["noise_only_db"]) - (10 * np.log10(4.0) + a["threshold_db"] - b["threshold_db"])) < 1e-6
        assert 0.0 <= a["threshold_db"] - b["threshold_db"] < 0.2                                  # the shorter window's threshold
        assert abs((lv["margin_db"] - a["margin_db"]) - 6.0) < 0.05                                 # noise-limited: the planner's margin follows
        k = T["options"][[o["id"] for o in T["options"]].index("4 s 197 dB")]["costs"]["mismatch_loss_db"]
        for r in ("2", "8"):
            assert abs(b["plain_filter_on_a_mover_db"][r] - (b["margin_db"] - k[r])) < 0.05         # noise-limited: the peak loss comes straight off
    # two cells by an independent path: the planner on its own range grid (no slicing), and buildspec.performance at that own depth
    env = pick["deep_no_bottom_return__ss4__apl_1994"]
    cl = [c for c in classes if c["name"] in ("large_submerged", "ship")]
    d4 = bs.derive(t.variant_spec(3, 32, "2k", 4.0, 197.0))
    P = depthplan.plan(d4, env, cl, keep_grid=False)
    for c in cl:
        assert abs(P["classes"][c["name"]]["recommended"]["margin_db"] - deep["4 s 197 dB"][c["name"]]["margin_db"]) < 0.05
    d3 = bs.derive(t.variant_spec(3, 32, "2k", 1.0, 203.0))
    c = cl[0]
    pf = bs.performance(d3, depthplan.wind_of(env), ranges=(12000.0,), environment=env, own_depth_m=deep["1 s 203 dB"][c["name"]]["own_depth_m"], classes=[c])
    assert abs(pf["surface"]["classes"][c["name"]]["margin_db"]["12000.0"] - deep["1 s 203 dB"][c["name"]]["margin_db"]) < 0.05


def test_committed_transmit_table_reproduces_and_leaves_the_old_report_as_it_was(tmp_path):
    """--report on the committed JSON gives the committed .md and JSON; without the `transmit` block it gives the report as it
    was before it (every old section byte-identical); the transmit table's shipped column is the grid's figure."""
    t = tool()
    out = tmp_path / "s"
    out.with_suffix(".json").write_bytes(COMMITTED.read_bytes())
    assert t.main(["--report", "--out", str(out)]) == 0
    assert out.with_suffix(".md").read_text() == COMMITTED.with_suffix(".md").read_text()
    assert out.with_suffix(".json").read_text() == COMMITTED.read_text()
    R = json.loads(COMMITTED.read_text())
    T = R["transmit"]
    md = COMMITTED.with_suffix(".md").read_text()
    new = "\n".join(t.transmit_section(R) + t.waveform_section(R)) + "\n"
    assert new in md and md.replace(new, "", 1) == t.report({k: v for k, v in R.items() if k != "transmit"})
    assert md.index("## 0. Transmit options") < md.index("## 0a. The plain filter on a mover") < md.index("## 1. The answer in tables")
    # Section 0a (LFM against HFM on a mover, derived at --report) is an addition: without `waveforms` the report is the one
    # before it, byte for byte (section 0 included), and the transmit section does not read it.
    old_T = {k: v for k, v in T.items() if k != "waveforms"}
    assert t.transmit_section(dict(R, transmit=old_T)) == t.transmit_section(R)
    assert md.replace("\n".join(t.waveform_section(R)) + "\n", "", 1) == t.report(dict(R, transmit=old_T))
    res, grid = T["grid"]["results"], R["grid"]["results"]["3x32|2k|1s|197dB"]
    for site in grid:
        for c in ("large_submerged", "ship"):
            assert abs(res[f"{site}__as_stated"]["1 s 197 dB"][c]["margin_db"] - grid[site]["classes"][c]["A"]["margin_db"]) < 2e-3, (site, c)
    law = T["bubble_law"]
    assert abs(res[f"deep_no_bottom_return__ss5__{law}"]["1 s 197 dB"]["ship"]["margin_db"] - grid["deep_no_bottom_return_sea_state_5"]["classes"]["ship"]["A"]["margin_db"]) < 2e-3
    assert all(set(v) == {o["id"] for o in T["options"]} for v in res.values())
    # every computed cell: noise-limited figures move by the level step and the pulse (with the detector's threshold), reverberation less
    by = {(o["pulse_s"], o["source_level_db"]): o for o in T["options"]}
    less, same = set(), 0
    for key, row in res.items():
        for c in ("large_submerged", "ship"):
            a = row["1 s 197 dB"][c]
            for (p, lv), o in by.items():
                b = row[o["id"]][c]
                step = (lv - 197.0) + 10 * np.log10(p) + a["threshold_db"] - b["threshold_db"]
                if b["own_depth_m"] == a["own_depth_m"]:                                                 # at one own depth: exactly (where the flat
                    assert abs((b["noise_only_db"] - a["noise_only_db"]) - step) < 3e-3, (key, c, o["id"])   # optimum moves, the planner chose better)
                    same += 1
                assert b["margin_db"] - a["margin_db"] <= step + 3e-3, (key, c, o["id"])                  # energy never buys more than itself
                if step > 0.5 and b["ceiling_db"] is not None and b["noise_only_db"] - b["margin_db"] > 0.1:   # where reverberation costs something: less
                    assert b["margin_db"] - a["margin_db"] < step - 1e-3 and b["margin_db"] <= b["ceiling_db"] + 2e-3, (key, c, o["id"])
                    less.add(key)
                for r in o["costs"]["mismatch_loss_db"]:
                    assert b["plain_filter_on_a_mover_db"][r] <= b["margin_db"] + 2e-3
    assert {"shelf_100m_sand__as_stated", "winter_200m_sand_profile__as_stated"} <= less
    assert same >= 0.95 * len(res) * 2 * len(by)
    assert len(json.dumps(T)) < 300_000


def test_the_waveform_section_is_the_grid_less_the_closed_forms():
    """Section 0a's cells: each option's grid margin less the closed-form peak loss on a mover closing at 2 and 8 m/s, the
    linear sweep's (live_sweep_mismatch, as section 0's own mover columns: equal to them) and the hyperbolic one's
    (doppler.hfm_doppler); the HFM's loss does not grow with the pulse, its range bias does (f1 T / B)."""
    from gpusonar import doppler
    R = json.loads(COMMITTED.read_text())
    T = R["transmit"]
    W = T["waveforms"]
    assert W == t_rounded(tool().tx_waveforms(T))
    res = T["grid"]["results"]
    law = T["bubble_law"]
    for o in T["options"]:
        q = W["per_option"][o["id"]]
        assert q["hfm"]["loss_db"]["8"] == pytest.approx(0.281, abs=2e-3) and q["hfm"]["loss_db"]["2"] == pytest.approx(0.070, abs=2e-3)
        assert q["hfm"]["range_m_per_m_s"] == pytest.approx(3.0 * o["pulse_s"], abs=1e-3)
        assert q["lfm"]["loss_db"]["8"] == pytest.approx(o["costs"]["mismatch_loss_db"]["8"], abs=2e-3)
        for n in (2, 6):
            for table, suffix in (("bubble_law", f"__{law}"), ("no_bubble_law", "")):
                r = res[f"deep_no_bottom_return__ss{n}{suffix}"][o["id"]]
                cell = W["deep"][table][str(n)][o["id"]]
                for c in ("large_submerged", "ship"):
                    assert cell["lfm"]["8"][c] == pytest.approx(r[c]["plain_filter_on_a_mover_db"]["8"], abs=3e-3)
                    assert cell["hfm"]["2"][c] == pytest.approx(r[c]["margin_db"] - doppler.hfm_doppler(-2.0, 4e3, 6e3, o["pulse_s"], q["sound_speed"])["loss_db"], abs=3e-3)
    # The finding the section leads with: closing at 8 m/s with no bank, the linear sweep holds both classes at sea state 2
    # only with 203 dB a ping (the grid's top level at 1 s) and at no higher state; the hyperbolic one holds them up to sea
    # state 6 at the energy that holds a still target (section 0's least energy), except one marginal cell: without a bubble
    # law at sea state 5 its 0.28 dB at 8 m/s takes `large_submerged` under 0 at 203 dB, one 3 dB step more (206 dB).
    le = W["least_energy"]["bubble_law"]
    assert le["2"]["lfm"]["8"]["options"] == ["1 s 203 dB"] and all(le[str(n)]["lfm"]["8"]["energy_db"] is None for n in (3, 4, 5, 6))
    for table in ("bubble_law", "no_bubble_law"):
        for n in (2, 3, 4, 5, 6):
            still = T["answers"]["least_energy"][table][str(n)]["energy_db"]
            assert W["least_energy"][table][str(n)]["hfm"]["2"]["energy_db"] == pytest.approx(still, abs=0.01)
            step = 3.01 if (table, n) == ("no_bubble_law", 5) else 0.0
            assert W["least_energy"][table][str(n)]["hfm"]["8"]["energy_db"] == pytest.approx(still + step, abs=0.01), (table, n)


def t_rounded(x):
    return tool()._rounded(x)


def test_the_report_states_what_was_measured_and_no_longer_says_not_run():
    """Sections 0 and 0a after the deep site's run (docs/tasks/doppler_deep12_eval.json; DOPPLER.md HFM.8 and "Deep site at 12 km"):
    the stale "Not run." text is gone; the margins are said to be for an unresolved, non-moving target; each design's measured
    acceptance is in the table, computed from the measurement's JSON; without the JSON the old text comes back."""
    t = tool()
    md = COMMITTED.with_suffix(".md").read_text()
    s0 = md.split("## 0. Transmit options", 1)[1].split("## 1. The answer in tables", 1)[0]
    assert "Not run" not in s0 and "Longer pulses through the chain are not measured" not in s0
    assert "UNRESOLVED, NON-MOVING target of the stated strength" in s0 and "Measured against this sentence" in s0
    M = json.loads((ROOT / "docs" / "tasks" / "doppler_deep12_eval.json").read_text())
    for name, a in M["acceptance"].items():
        row = next(line for line in s0.splitlines() if line.startswith(f"| {a['what']} (`{a['case']}`) |"))
        for lb, v in a["targets"].items():
            assert f"{v['verdict']}{'**' if v['verdict'] == 'FAIL' else ''} {v['detected']} of {v['pings']}" in row, (name, lb)
    assert "**FAIL** 0 of 6" in next(line for line in s0.splitlines() if "(`ss3_lfm1`)" in line)
    R = json.loads(COMMITTED.read_text())
    real = t.DEEP12
    try:
        t.DEEP12 = ROOT / "no_such_file.json"
        old = "\n".join(t.transmit_section(R) + t.waveform_section(R))
    finally:
        t.DEEP12 = real
    assert "**Not run.** No transmit option other than the shipped 1 s" in old and "UNRESOLVED" not in old
