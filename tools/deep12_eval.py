#!/usr/bin/env python3
"""The deep site at 12 km, measured against the planner (issue sonar-rctz; docs/tasks/DOPPLER.md "Deep site at 12 km").

    tools/deep12_eval.py DIR [--json OUT] [--markdown]     # DIR: the deep12_<case>.json of tools/doppler_gpu_runs.sh `deep12`
    tools/deep12_eval.py --predict ss3_baseline             # the planner's figures for one case, nothing measured
    tools/deep12_eval.py --reaccept docs/tasks/doppler_deep12_eval.json   # the acceptance block again from the stored rows

Cases: ss<3|4|5>_baseline is the SHIPPED sonar as it is (since D-256 the hyperbolic 2 s sweep at 197 dB; no waveform is
set); ss<n>_<lfm|hfm><seconds>[p<level>][_cfar] states its transmission whole (`sets`: type, source, length, level), so a
linear case stays linear whatever the shipped sonar sends. The former baseline (1 s linear, shipped until D-256) is
`ss3_lfm1`, kept as a named comparison (`former_baseline`).

For each case (sea state x transmission x filter) and each target of popoto_12km_deep_sea_state_5, beside what the chain
measured (tools/doppler_bank_eval.py: detected pings, the mean margin over the detected pings, tracked and its class at the
last ping; the false detections on the target-free pings at the default budget), the planner's prediction for the SAME site,
sea state, class, geometry and energy, from the trade study's own functions (tools/popoto_trade_study.py: `variant_spec`,
`predict_point`, `sweep_mismatch_loss_db`, `tx_sites`; nothing typed):

- `planner_margin_db`: the planner's margin for this target where it is (its own depth, the scene's 225 m, its range at the
  first ping, its total target strength): one row's SNR over what the three-row detector needs for Pd 0.5 (the trade
  study's statistic, here at the target's own depth rather than the worst of the class's depths);
- `predicted_chain_margin_db`: the same budget as the chain's margin: the planner's expected chain snr_db (one row's SNR less
  the aperture window's and the map's sampling loss: `predict_point`'s `chain_snr_expected_db`, as DOPPLER.md LS.3) less the
  single-sample threshold the chain used (from the measurement's JSON), for an echo with no radial speed;
- `predicted_mover_db`: that less the plain filter's closed-form PEAK loss at the target's mean true range rate (linear
  sweep: `sweep_mismatch_loss_db`; hyperbolic: gpusonar.doppler.hfm_doppler); for the bank, the zero-speed one;
- `table_db`: the trade study's own cell (docs/design/popoto_trade_study.json, `--transmit`: worst over the class's depths
  at its recommended own depth, low target strength) where the grid has the option, and its plain-filter mover columns;
- `difference_db`: measured less predicted (`predicted_mover_db`), on the pings detected;
- `predicted_one_highlight_db`: `predicted_mover_db` less 10 log10(highlights): what ONE of a hull's equal highlights should
  give when the hull's highlights are resolved in range and the detector tests each alone (its multiscale scales here are
  the single sample and a 2 m cell: `detector_scales`);
- `verdict`: PASS when the target is detected on at least half the pings (the design's detection probability, 0.5 on one
  ping), else FAIL.

Status of every number: ours (simulation, one seed; sites illustrative; target strengths assumed).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

SCENE = "popoto_12km_deep_sea_state_5"
CLASS_OF = {"closing": "large_submerged", "oblique": "large_submerged", "crossing": "large_submerged", "point": "large_submerged", "ship": "ship"}
WAVES = {"lfm1": ("lfm", 1.0, 197.0), "hfm1": ("hfm", 1.0, 197.0), "lfm4p200": ("lfm", 4.0, 200.0), "hfm4p200": ("hfm", 4.0, 200.0)}
SHIPPED = ("hfm", 2.0, 197.0)       # configs/sonars/popoto_long_range.yaml since D-256 (tests/python/test_hfm_baseline.py holds the two together)
WF = "sonar_overrides.transmit.waveform"
DESIGN = {"baseline": "baseline (shipped since D-256: hyperbolic 2 s, 197 dB, plain filter)",
          "lfm1": "former baseline (1 s, 197 dB, linear; shipped until D-256)", "lfm4p200": "sea-state-5 alternate (4 s, 200 dB), linear",
          "hfm4p200": "sea-state-5 alternate (4 s, 200 dB), hyperbolic", "hfm1": "hyperbolic 1 s, 197 dB"}


def parse_case(name: str) -> dict:
    """ss<3|4|5>_<lfm|hfm><seconds>[p<rms level dB>][_cfar] (tools/doppler_gpu_runs.sh `deep12` reads the same form): the pulse
    defaults to the shipped 1 s, the level to its 197 dB; the bank (`_cfar`) takes a linear sweep only."""
    b = re.fullmatch(r"(ss[345])_baseline", name)
    if b:
        wave, T, sl = SHIPPED
        return {"case": name, "sea_state": int(b.group(1)[2]), "key": "baseline", "wave": wave, "pulse_s": T, "source_level_db": sl,
                "bank": False, "design": DESIGN["baseline"], "sets": []}
    m = re.fullmatch(r"(ss[345])_((lfm|hfm)([0-9]+)(?:p([0-9]+))?)(_cfar)?", name)
    if not m or (m.group(6) and m.group(3) != "lfm"):
        raise ValueError(f"deep12 case {name!r}: ss3|ss4|ss5 _ baseline, or _ <lfm|hfm><seconds>[p<level>] [_cfar, linear only]")
    key, wave, T, sl = m.group(2), m.group(3), float(m.group(4)), float(m.group(5) or 197)
    if T <= 0.0:
        raise ValueError(f"deep12 case {name!r}: the pulse must be longer than 0 s")
    design = DESIGN.get(key) or f"{'hyperbolic' if wave == 'hfm' else 'linear'} {T:g} s, {sl:g} dB"
    # The transmission stated whole, in tools/doppler_gpu_runs.sh's order (type, source, length, level): the shipped sonar is
    # no longer the linear sweep (D-256), so a linear case says so and nulls the hyperbolic sweep's `source`.
    sets = ([f"{WF}.type=lfm", f"{WF}.source=null"] if wave == "lfm" else [f"{WF}.type=sampled", f"{WF}.source=hfm"]) + [f"{WF}.duration={T:.1f}"]
    if sl != 197.0:
        sets.append(f"{WF}.source_level_db_rms={sl:.1f}")
    return {"case": name, "sea_state": int(m.group(1)[2]), "key": key, "wave": wave, "pulse_s": T, "source_level_db": sl,
            "bank": bool(m.group(6)), "design": design, "sets": sets}


_CACHE: dict = {}


def _study():
    if "T" not in _CACHE:
        import popoto_trade_study as T
        from gpusonar import environment
        _CACHE["T"] = T
        _CACHE["envs"] = T.tx_sites(environment.load_list("illustrative"))
        _CACHE["grid"] = json.loads((ROOT / "docs" / "design" / "popoto_trade_study.json").read_text())["transmit"]["grid"]["results"]
    return _CACHE["T"], _CACHE["envs"], _CACHE["grid"]


def _geometry():
    """(highlights, range extent of the hull along the line of sight at the first ping, m) per label, from the scene file."""
    from gpusonar.config import load_scenario_config
    cfg = load_scenario_config(SCENE, expand=False)
    p0 = np.asarray(cfg["platform"]["position"], float)
    n, ext = {}, {}
    for it in cfg["scene"]:
        pos = np.asarray(it["positions"], float)
        u = np.mean(pos, axis=0) - p0
        u /= np.linalg.norm(u)
        n[it["label"]] = len(pos)
        ext[it["label"]] = float(np.ptp(pos @ u))
    return n, ext


HIGHLIGHTS, EXTENT = _geometry()


def detector_scales(pulse_s: float = 1.0) -> list:
    """The multiscale detector's scales the chain tests in this scene's window (three rows added: elevation {}): length (m),
    looks, single-test threshold (dB), from gpusonar.live.detector_plan on the sonar's template record."""
    from gpusonar import buildspec as bs, live
    from gpusonar.config import deep_merge, load_scenario_config, load_sonar_config
    cfg = load_scenario_config(SCENE, expand=False)
    sonar = deep_merge(load_sonar_config(cfg["sonar"]), {**cfg["sonar_overrides"], "transmit": {"waveform": {"duration": float(pulse_s)}}})
    plan = live.detector_plan(bs.detector_template(sonar), {"elevation": {}})
    return [{"length_m": sc["length_m"], "looks": sc["looks"], "threshold_db": sc["threshold_db"]} for sc in json.loads(plan["json"])["scales"]]


def targets() -> dict:
    """{label: (total target strength dB, depth m, horizontal range m at the first ping, mean range rate over 6 pings at the
    reflection)} of the scene, from its file and doppler_bank_eval's truth."""
    from gpusonar.config import load_scenario_config
    import doppler_bank_eval as dbe
    cfg = load_scenario_config(SCENE, expand=False)
    p0 = np.asarray(cfg["platform"]["position"], float)
    tr = dbe.truth(cfg, int(cfg["pings"]["count"]), float(load_scenario_config(SCENE)["environment"]["sound_speed"]))
    out = {}
    for it in cfg["scene"]:
        c = np.mean(np.asarray(it["positions"], float), axis=0)
        out[it["label"]] = (float(it["target_strength_db"] + 10.0 * np.log10(len(it["positions"]))), float(c[2]), float(np.hypot(*(c - p0)[:2])),
                            float(np.mean([r["range_rate"] for r in tr[it["label"]]])))
    return out


def predict(case: str, threshold_db: float | None = None) -> dict:
    """The planner's figures for every target of the scene in one case (`parse_case`); `threshold_db` the chain's single-sample
    threshold (the measurement's), else the planner's own (`sample_threshold_db`)."""
    T, envs, grid = _study()
    from gpusonar import buildspec as bs
    from gpusonar import doppler as dop
    k = parse_case(case)
    from gpusonar import environment
    law = next(e["bubble_model"] for e in environment.load_list("illustrative") if e["name"] == T.TX_BUBBLE_SITE)     # the scene's (apl_1994)
    env = envs[f"{T.TX_DEEP}__ss{k['sea_state']}__{law}"]
    d = bs.derive(T.variant_spec(3, 32, "2k", k["pulse_s"], k["source_level_db"]))
    f_lo, f_hi = d["band_hz"]
    c = float(d["sound_speed"])
    own = 225.0
    oid = f"{k['pulse_s']:g} s {k['source_level_db']:g} dB"
    cell = (grid.get(env["name"]) or {}).get(oid) or {}
    out = {"case": k, "environment": env["name"], "own_depth_m": own, "option": oid, "targets": {}}
    T_ = targets()
    for label, (ts, z, r, rate) in T_.items():
        key = (env["name"], k["pulse_s"], k["source_level_db"], ts, z, r)
        if key not in _CACHE:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _CACHE[key] = T.predict_point(d, env, ts, z, own, r)
        p = _CACHE[key]
        thr = float(p["sample_threshold_db"] if threshold_db is None else threshold_db)
        zero = float(p["chain_snr_expected_db"] - thr)
        if k["bank"]:
            loss = 0.0
        elif k["wave"] == "lfm":
            loss = T.sweep_mismatch_loss_db(rate, f_hi - f_lo, k["pulse_s"], c)
        else:
            loss = float(dop.hfm_doppler(rate, f_lo, f_hi, k["pulse_s"], c)["loss_db"])
        cl = CLASS_OF[label]
        n = HIGHLIGHTS[label]
        out["targets"][label] = {"class": cl, "ts_db": ts, "depth_m": z, "range_m": r, "range_rate_m_s": rate, "highlights": n,
                                 "highlight_range_extent_m": EXTENT[label],
                                 "predicted_one_highlight_db": zero - float(loss) - 10.0 * np.log10(n),
                                 "planner_margin_db": float(p["margin_db"]), "limited_by": p["limited_by"], "factor_db": float(p["factor_db"]),
                                 "chain_snr_expected_db": float(p["chain_snr_expected_db"]), "threshold_used_db": thr,
                                 "predicted_chain_margin_db": zero, "closed_form_loss_db": float(loss), "predicted_mover_db": zero - float(loss),
                                 "table_db": (cell.get(cl) or {}).get("margin_db"), "table_own_depth_m": (cell.get(cl) or {}).get("own_depth_m"),
                                 "table_plain_filter_on_a_mover_db": (cell.get(cl) or {}).get("plain_filter_on_a_mover_db") if k["wave"] == "lfm" else None}
    return out


ROUND3 = ("ss3_hfm2", "ss3_lfm2_cfar", "ss3_hfm1p200", "ss4_hfm2", "ss5_hfm8p200", "ss5_hfm4p203", "ss5_hfm8p203")   # DOPPLER.md DM.5
BRACKET_HELD_DB, BRACKET_NOT_DB = 5.4, 3.4          # the point's margin at which the resolved hulls were held / not held (DM.3, measured)


def prediction(case: str) -> dict:
    """What the planner and the measured bracket say of one case BEFORE it is run: the point's predicted chain margin, each hull's
    one-highlight prediction, the bracket's verdict for the resolved hulls (held where the point's predicted margin is at least
    5.4 dB, not held at 3.4 dB or less, inside the bracket between), the blind range c T / 2, the duty, the hyperbolic sweep's
    range bias per m/s, and whether a single listed projector gives the option (the trade study's projector column)."""
    from gpusonar import doppler as dop
    T, envs, grid = _study()
    k = parse_case(case)
    P = predict(case, threshold_db=9.45373)                 # the chain's single-sample threshold in this window (detector_scales)
    tx = json.loads((ROOT / "docs" / "design" / "popoto_trade_study.json").read_text())["transmit"]
    opt = next((o for o in tx["options"] if o["id"] == P["option"]), None)
    c = 1500.0
    point = P["targets"]["point"]["predicted_chain_margin_db"]
    shift = None
    if k["bank"]:                       # the bank's threshold shift, measured at 1 s and 4 s (DM.2 (b)): 0.65 and 0.86 dB, taken as 0.75 here
        shift = 0.75
        point -= shift
    verdict = ("held" if point >= BRACKET_HELD_DB else "not held" if point <= BRACKET_NOT_DB else "inside the bracket (between 3.4 and 5.4 dB)")
    two_way = 2.0 * 12700.0 / c
    out = {"case": case, "design": k["design"], "energy_db": k["source_level_db"] + 10.0 * np.log10(k["pulse_s"]),
           "point_predicted_margin_db": point, "bank_threshold_shift_db_assumed": shift, "planner_point_margin_db": P["targets"]["point"]["planner_margin_db"],
           "one_highlight_predicted_db": {lb: P["targets"][lb]["predicted_one_highlight_db"] - (shift or 0.0) for lb in ("closing", "oblique", "ship")},
           "bracket_verdict_resolved_hulls": verdict, "blind_range_m": 0.5 * c * k["pulse_s"],
           "duty": k["pulse_s"] / (two_way + k["pulse_s"]),
           "range_bias_m_per_m_s": dop.hfm_range_factor(4000.0, 6000.0, k["pulse_s"]) if k["wave"] == "hfm" else None,
           "single_listed_projector": (bool(opt["costs"]["projector"]["single_units"]) if opt else None),
           "projector": (opt["costs"]["projector"]["words"] if opt else "the option is not in the trade study's grid")}
    return out


def measured(res: dict, bank: bool) -> dict:
    """Per target (C++ engine): detected, mean margin, tracked, class; and the Python engine's margin where it ran."""
    eng = res["engines"]
    side = "on" if bank else "off"
    out = {}
    for label, t in eng["cpp"][side]["targets"].items():
        got = [p["margin_db"] for p in t["pings"] if p["margin_db"] is not None]
        row = {"detected": t["detected"], "pings": len(t["pings"]), "margin_db": float(np.mean(got)) if got else None,
               "margin_pings_db": [p["margin_db"] for p in t["pings"]], "tracked": t["tracked"], "kind": t["kind"],
               "threshold_db": t["pings"][0]["threshold_db"]}
        if "python" in eng:
            py = [p["margin_db"] for p in eng["python"][side]["targets"][label]["pings"]]
            row["python_margin_pings_db"] = py
            both = [(a, b) for a, b in zip(row["margin_pings_db"], py) if a is not None and b is not None]
            row["engines_max_difference_db"] = float(max(abs(a - b) for a, b in both)) if both else None
            row["engines_same_detections"] = [a is None for a in row["margin_pings_db"]] == [b is None for b in py]
        out[label] = row
    return out


PREDICTIONS = ROOT / "docs" / "tasks" / "deep12_round3_predictions.json"
_PREDICTED = {c["case"]: c for c in json.loads(PREDICTIONS.read_text())["cases"]} if PREDICTIONS.exists() else {}


def summarise(folder: Path) -> dict:
    rows, backgrounds = [], {}
    for path in sorted(folder.glob("deep12_*.json")):
        case = path.stem[len("deep12_"):]
        res = json.loads(path.read_text())
        k = parse_case(case)
        m = measured(res, k["bank"])
        thr = next(iter(m.values()))["threshold_db"]
        P = predict(case, thr)
        pre = _PREDICTED.get(case)
        bg = res.get("background", {}).get("default", {})
        backgrounds[case] = {"pings": bg.get("pings"), "false_detections": bg.get("on" if k["bank"] else "off"),
                             "at_3_a_ping": (res.get("background", {}).get("3.0") or {}).get("on" if k["bank"] else "off")}
        for label, x in m.items():
            p = P["targets"][label]
            diff = None if x["margin_db"] is None else x["margin_db"] - p["predicted_mover_db"]
            verdict = "PASS" if 2 * x["detected"] >= x["pings"] else "FAIL"
            before = None
            if pre is not None and label in ("closing", "oblique"):
                before = pre["bracket_verdict_resolved_hulls"]
            rows.append({"case": case, "sea_state": k["sea_state"], "design": k["design"], "filter": "bank cfar" if k["bank"] else "plain", "verdict": verdict,
                         "predicted_before_run": before,
                         "target": label, **{f"measured_{a}": b for a, b in x.items()}, **{f"planner_{a}": b for a, b in p.items()},
                         "difference_db": diff})
    rows.sort(key=lambda r: (r["difference_db"] is not None, r["difference_db"] if r["difference_db"] is not None else 0.0))
    return {"what": "tools/deep12_eval.py: the deep12 cases measured through the chain against the planner (ours)", "folder": str(folder),
            "detector_scales": {"1 s": detector_scales(1.0), "4 s": detector_scales(4.0)}, "highlights": HIGHLIGHTS, "highlight_range_extent_m": EXTENT,
            "rows": rows, "background_default_budget": backgrounds} | {"acceptance": None}


# name: (case, words[, a case measured with the same transmission, used until the named case itself is run])
ACCEPTANCE = {"baseline": ("ss3_baseline", "the baseline (shipped since D-256: hyperbolic 2 s, 197 dB, plain filter) at sea state 3", "ss3_hfm2"),
              "former_baseline": ("ss3_lfm1", "the former baseline (1 s, 197 dB, linear, plain filter; shipped until D-256) at sea state 3"),
              "alternate": ("ss5_lfm4p200", "the sea-state-5 alternate as configured (4 s, 200 dB, linear, plain filter) at sea state 5"),
              "alternate_bank": ("ss5_lfm4p200_cfar", "the alternate's energy, linear with the bank (combine: cfar) at sea state 5"),
              "alternate_hfm": ("ss5_hfm4p200", "the alternate's energy, hyperbolic, plain filter, at sea state 5"),
              # DM.5's energy candidates (opt-in cases; nothing configured): the measured answer to "what would accept the resolved hull"
              "candidate_ss3_hfm2": ("ss3_hfm2", "candidate baseline: hyperbolic 2 s, 197 dB (200 dB a ping), plain filter, at sea state 3"),
              "candidate_ss3_hfm1p200": ("ss3_hfm1p200", "candidate baseline: hyperbolic 1 s, 200 dB (200 dB a ping), plain filter, at sea state 3"),
              "candidate_ss3_lfm2_cfar": ("ss3_lfm2_cfar", "candidate baseline: linear 2 s, 197 dB (200 dB a ping) with the bank, at sea state 3"),
              "candidate_ss4_hfm2": ("ss4_hfm2", "the hyperbolic 2 s, 197 dB candidate at sea state 4"),
              "candidate_ss5_hfm4p203": ("ss5_hfm4p203", "candidate alternate: hyperbolic 4 s, 203 dB (209 dB a ping), plain filter, at sea state 5"),
              "candidate_ss5_hfm8p200": ("ss5_hfm8p200", "candidate alternate: hyperbolic 8 s, 200 dB (209 dB a ping), plain filter, at sea state 5"),
              "candidate_ss5_hfm8p203": ("ss5_hfm8p203", "upper bracket: hyperbolic 8 s, 203 dB (212 dB a ping), plain filter, at sea state 5")}


def acceptance(S: dict) -> dict:
    """Per design (ACCEPTANCE): each target's verdict, detected pings, mean margin and the planner's prediction (with the closed
    form's loss); `accepted` only when every hull and the ship pass."""
    R = {(r["case"], r["target"]): r for r in S["rows"]}
    out = {}
    for name, (case, words, *same) in ACCEPTANCE.items():
        stand_in = None
        if not any(k[0] == case for k in R):
            if not same or not any(k[0] == same[0] for k in R):
                continue
            stand_in, case = case, same[0]
        t = {lb: {"verdict": R[(case, lb)]["verdict"], "detected": R[(case, lb)]["measured_detected"], "pings": R[(case, lb)]["measured_pings"],
                  "margin_db": R[(case, lb)]["measured_margin_db"], "predicted_db": R[(case, lb)]["planner_predicted_mover_db"],
                  "class": R[(case, lb)]["planner_class"]} for lb in HIGHLIGHTS if (case, lb) in R}
        out[name] = {"case": case, "what": words, "targets": t,
                     "accepted": all(v["verdict"] == "PASS" for lb, v in t.items() if lb != "point")}
        if stand_in:
            out[name]["measured_as"] = (f"{case}: the same transmission as {stand_in} (the shipped sonar), measured as a candidate before the default "
                                        f"moved (DOPPLER.md DM.6); {stand_in} itself is not run yet")
        pre = _PREDICTED.get(case)
        if pre is not None:
            held = all(t[lb]["verdict"] == "PASS" for lb in ("closing", "oblique") if lb in t)
            out[name]["predicted_before_run"] = {"bracket_verdict_resolved_hulls": pre["bracket_verdict_resolved_hulls"],
                                                 "point_predicted_margin_db": pre["point_predicted_margin_db"],
                                                 "resolved_hulls_held_measured": held,
                                                 "single_listed_projector": pre["single_listed_projector"], "blind_range_m": pre["blind_range_m"],
                                                 "duty": pre["duty"], "range_bias_m_per_m_s": pre["range_bias_m_per_m_s"]}
    return out


def markdown(S: dict) -> str:
    L = ["| Case | Design | Filter | Target (range rate m/s) | Verdict | Detected | Measured margin dB | Predicted, zero speed | Closed-form loss | Predicted | Measured - predicted | One highlight, predicted | Planner at the scene / table's worst depth | Tracked (class) |",
         "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]

    def f(v, fmt="+.2f"):
        return "-" if v is None else format(v, fmt)
    for r in S["rows"]:
        L.append(f"| {r['case']} | {r['design']} | {r['filter']} | {r['target']} ({r['planner_range_rate_m_s']:+.2f}) | {r['verdict']} | {r['measured_detected']} of {r['measured_pings']} | "
                 f"{f(r['measured_margin_db'])} | {f(r['planner_predicted_chain_margin_db'])} | {f(r['planner_closed_form_loss_db'], '.2f')} | "
                 f"{f(r['planner_predicted_mover_db'])} | {f(r['difference_db'])} | {f(r['planner_predicted_one_highlight_db']) if r['planner_highlights'] > 1 else '-'} | "
                 f"{f(r['planner_planner_margin_db'])} / {f(r['planner_table_db'])} | "
                 f"{'yes' if r['measured_tracked'] else 'no'} ({r['measured_kind'] or '-'}) |")
    L += ["", "| Case | Target-free pings | False detections, default budget | At 3 a ping |", "| --- | --- | --- | --- |"]
    for case, b in sorted(S["background_default_budget"].items()):
        L.append(f"| {case} | {b['pings']} | {b['false_detections']} | {b['at_3_a_ping']} |")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", nargs="?")
    ap.add_argument("--json")
    ap.add_argument("--markdown", action="store_true")
    ap.add_argument("--predict", metavar="CASE")
    ap.add_argument("--source", help="where the case JSONs came from (jobs, tree, device), written into the result")
    ap.add_argument("--predict-cases", nargs="*", metavar="CASE", help="the predictions of these cases (default: DM.5's), written before they are run")
    ap.add_argument("--reaccept", metavar="JSON", help="rewrite the `acceptance` block of a stored summary from its own rows (nothing else changes)")
    a = ap.parse_args(argv)
    if a.reaccept:
        S = json.loads(Path(a.reaccept).read_text())
        S["acceptance"] = acceptance(S)
        Path(a.reaccept).write_text(json.dumps(S, indent=1, default=float) + "\n")
        return 0
    if a.predict:
        print(json.dumps(predict(a.predict), indent=1, default=float))
        return 0
    if a.predict_cases is not None:
        out = {"what": "tools/deep12_eval.py --predict-cases: written BEFORE the cases were run (DOPPLER.md DM.5); ours, the planner and the measured bracket",
               "bracket_db": {"held": BRACKET_HELD_DB, "not_held": BRACKET_NOT_DB}, "cases": [prediction(c_) for c_ in (a.predict_cases or ROUND3)]}
        text = json.dumps(out, indent=1, default=float) + "\n"
        if a.json:
            Path(a.json).write_text(text)
        else:
            print(text)
        return 0
    if not a.folder:
        ap.error("give the folder of deep12_<case>.json, or --predict CASE")
    S = summarise(Path(a.folder))
    S["acceptance"] = acceptance(S)
    S["command"] = "tools/deep12_eval.py DIR " + " ".join(x for x in (argv if argv is not None else sys.argv[1:]) if x != a.folder and x != a.source).replace("--source", "--source SOURCE")
    if a.source:
        S["source"] = a.source
    S["folder"] = Path(a.folder).name
    S["seconds"] = {}
    for path in sorted(Path(a.folder).glob("deep12_*.json")):
        res = json.loads(path.read_text())
        S["seconds"][path.stem[len("deep12_"):]] = {e: {k: v.get("seconds_median_after_first") for k, v in res["engines"][e].items() if k in ("off", "on")} for e in res["engines"]}
    if a.json:
        Path(a.json).write_text(json.dumps(S, indent=1, default=float) + "\n")
    if a.markdown or not a.json:
        print(markdown(S))
    return 0


if __name__ == "__main__":
    sys.exit(main())
