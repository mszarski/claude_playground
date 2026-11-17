#!/usr/bin/env python3
"""The projector's direct sound at every hydrophone of a sonar, how far it overloads the
receiver, and how long after the pulse each channel is usable again.

    python tools/direct_sound_study.py                       # the three Popoto stages, 0.1 s and 1 s pulses
    python tools/direct_sound_study.py popoto_long_range_3row --pulses 1.0 --sets clip_only dead_10ms
    python tools/direct_sound_study.py NAME --overrides '{"transmit": {"lever_arm": [6, 0, 0]}}'

For each sonar file it prints and writes (JSON, default docs/design/direct_sound_study.json):

  hydrophones   per hydrophone: distance to the projector (transmit.lever_arm), the direct level
                SL - 20 log10(r) in dB re 1 uPa, the volts that makes at the ADC
                (+ receive.sensitivity_db + receive.gain_db) and the overload in dB above
                receive.adc_full_scale. Closed form, and checked against the simulator's own
                direct arrival (transmit.direct_path) through its front-end model.
  projector_distance   the same against distance alone, and the distance at which the direct
                sound just reaches full scale.
  recovery      for each pulse length and each set of recovery parameters: per channel, the
                time after the transmit from which the front end is back within tolerance
                (no stage on its rail or dead, recovery gain within --gain-tolerance-db of
                nominal, offset at the ADC below --offset-tolerance-db re full scale), that
                time counted from the end of the direct pulse at that hydrophone, and the
                minimum range it implies: an echo is whole only if it starts after the
                channel is clean, range >= c t_clean / 2. Simulated: the direct arrival of
                the simulator (free field: the sonar alone) through sim/receiver.py
                `apply_front_end`; closed forms beside it where there is one (counted from
                the last clipped sample: the pulse's edge taper stops the clipping a little
                before the pulse ends, so the simulated figures are the shorter).

THE RECOVERY PARAMETER SETS ARE ILLUSTRATIVE. No board has been measured: dead times and time
constants are swept over decades to show what each would cost in range, the unequal rails and
the gain split of the coupling sets are invented to show the mechanism, and the two corner
frequencies are the two values quoted for the board in configs/sonars/popoto_8.yaml (100 Hz,
"10 Hz has also been quoted"; both unconfirmed there). The level convention is the
simulator's: `source_level_db` + sensitivity + gain is the PEAK voltage of the pulse at the
ADC (SonarModel.amplitude_at_1m). If the source level is an rms figure the pulse peaks 3 dB
higher and every overload figure here rises by 3 dB.
"""
from __future__ import annotations

import argparse
import copy
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from gpusonar.sim import Scene, make_pose, simulate_ping          # noqa: E402
from gpusonar.sim import receiver as rx                           # noqa: E402
from gpusonar.sonar import SonarModel                             # noqa: E402

DEFAULT_SONARS = ("popoto_long_range_3row", "popoto_static_3row", "popoto_testauv_1row")
DISTANCES = (0.3, 0.5, 1.0, 1.69, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0)


def recovery_sets(gain_db: float) -> dict:
    """ILLUSTRATIVE `receive.overload` blocks, by name. Not a board's values."""
    sets = {"clip_only": {}}
    for ms in (1, 10, 100):
        sets[f"dead_{ms}ms"] = {"adc": {"dead_time": ms * 1e-3}}
        sets[f"tau_{ms}ms"] = {"adc": {"recovery_tau": ms * 1e-3}}
    sets["dead_10ms_tau_10ms"] = {"adc": {"dead_time": 0.01, "recovery_tau": 0.01}}
    for corner in (100.0, 10.0):
        # An input stage with 20 dB of gain behind it whose rails are unequal (1 V and 2 V): its
        # clipped output has a mean, which charges the coupling capacitor.
        sets[f"coupling_{corner:g}hz_unequal_rails"] = {"preamp": {"gain_db": gain_db - 20.0, "limit_v": [1.0, 2.0]},
                                                        "highpass_corner_hz": corner}
    sets["gain_switch"] = None                    # filled per sonar and pulse: enough reduction for the nearest hydrophone
    return sets


def closed_form_recovery(block: dict | None, gain_tolerance_db: float, offset_tolerance: float, pulse: float, gain_db: float):
    """Seconds after the last clipped sample, where the model has a closed form; else None."""
    if block is None:
        return None
    adc = block.get("adc") or {}
    out = 0.0
    if adc.get("dead_time") or adc.get("recovery_tau"):
        tau = float(adc.get("recovery_tau", 0.0))
        out = float(adc.get("dead_time", 0.0)) + (tau * np.log(1.0 / (1.0 - 10.0 ** (-abs(gain_tolerance_db) / 20.0))) if tau else 0.0)
    if block.get("preamp") and block.get("highpass_corner_hz"):
        # Hard-clipped limit (a square wave between the rails): mean (L_p - L_n) / 2, charged for the
        # pulse, decaying to the offset tolerance at the ADC.
        lo, hi = block["preamp"]["limit_v"]
        k = 10.0 ** ((gain_db - block["preamp"]["gain_db"]) / 20.0)
        tau_c = 1.0 / (2.0 * np.pi * block["highpass_corner_hz"])
        v = 0.5 * abs(hi - lo) * k * (1.0 - np.exp(-pulse / tau_c))
        out = max(out, tau_c * np.log(max(v / offset_tolerance, 1.0)))
    return float(out)


def spread(values) -> dict:
    v = np.asarray(values, dtype=float)
    return {"worst": float(v.max()), "best": float(v.min()), "median": float(np.median(v))}


def study(name: str, pulses, set_names, gain_tolerance_db: float, offset_tolerance_db: float, overrides: dict | None, quiet=False) -> dict:
    say = (lambda *a, **k: None) if quiet else print
    model = SonarModel.load(name, overrides)
    cfg = model.config
    rxc, w = cfg["receive"], model.waveform_config
    c = model.nominal_sound_speed
    sl, sens, gain_db, full_scale = float(w["source_level_db"]), float(rxc["sensitivity_db"]), float(rxc.get("gain_db", 0.0)), float(rxc["adc_full_scale"])
    lever = np.asarray(cfg["transmit"].get("lever_arm", [0.0, 0.0, 0.0]), dtype=float)
    r = np.linalg.norm(model.positions - lever, axis=1)
    level = sl - 20.0 * np.log10(r)
    volts_db = level + sens + gain_db
    over = volts_db - 20.0 * np.log10(full_scale)
    out = {"sonar": name, "source_level_db": sl, "sensitivity_db": sens, "gain_db": gain_db, "adc_full_scale_v": full_scale,
           "projector_lever_arm": lever.tolist(), "sound_speed": c,
           "level_convention": "source_level_db + sensitivity_db + gain_db is the PEAK voltage at the ADC (the simulator's convention); "
                               "an rms source level adds 3 dB to every overload figure",
           "unconfirmed_in_sonar_file": list(cfg.get("unconfirmed", [])),
           "hydrophones": [{"index": i, "position": [round(float(v), 4) for v in model.positions[i]], "distance_m": round(float(r[i]), 4),
                            "direct_level_db_re_upa": round(float(level[i]), 2), "peak_volts_at_adc": round(float(10.0 ** (volts_db[i] / 20.0)), 3),
                            "overload_db_re_adc": round(float(over[i]), 2)} for i in range(len(r))],
           "summary": {"nearest_m": float(r.min()), "farthest_m": float(r.max()), "overload_db_nearest": float(over.max()),
                       "overload_db_farthest": float(over.min()), "hydrophones_overloaded": int(np.sum(over > 0.0)), "hydrophones": int(len(r)),
                       "gain_reduction_for_no_overload_db": float(max(over.max(), 0.0)),
                       "distance_for_no_overload_m": float(10.0 ** ((sl + sens + gain_db - 20.0 * np.log10(full_scale)) / 20.0))},
           "projector_distance": [{"distance_m": d, "direct_level_db_re_upa": round(sl - 20.0 * np.log10(d), 2),
                                   "overload_db_re_adc": round(sl - 20.0 * np.log10(d) + sens + gain_db - 20.0 * np.log10(full_scale), 2)}
                                  for d in DISTANCES]}
    say(f"\n=== {name}: SL {sl:g} dB re 1 uPa at 1 m, sensitivity {sens:g} dB re 1 V/uPa, gain {gain_db:g} dB, ADC full scale {full_scale:g} V; "
        f"projector at {lever.tolist()} ===")
    say("  hydrophone   x       y       z     distance  direct level  peak at ADC  overload re ADC")
    for h in out["hydrophones"]:
        p = h["position"]
        say(f"  {h['index']:4d}     {p[0]:7.3f} {p[1]:7.3f} {p[2]:7.3f} {h['distance_m']:7.3f} m   {h['direct_level_db_re_upa']:7.1f} dB  "
            f"{h['peak_volts_at_adc']:9.2f} V   {h['overload_db_re_adc']:+6.1f} dB")
    s = out["summary"]
    say(f"  nearest {s['nearest_m']:.2f} m: {s['overload_db_nearest']:+.1f} dB; farthest {s['farthest_m']:.2f} m: {s['overload_db_farthest']:+.1f} dB; "
        f"{s['hydrophones_overloaded']} of {s['hydrophones']} overloaded; no overload beyond {s['distance_for_no_overload_m']:.1f} m from the projector")

    sets = recovery_sets(gain_db)
    unknown = [n for n in set_names if n not in sets]
    if unknown:
        raise SystemExit(f"unknown recovery set(s) {unknown}: one of {list(sets)}")
    offset_tolerance = full_scale * 10.0 ** (offset_tolerance_db / 20.0)
    out["tolerance"] = {"gain_db": gain_tolerance_db, "offset_db_re_adc_full_scale": offset_tolerance_db}
    out["recovery_sets_are"] = "ILLUSTRATIVE: no board has been measured"
    out["recovery_note"] = ("recovery_after_pulse_s counts from the nominal end of the direct pulse at each hydrophone (pulse + distance / c); "
                            "closed_form_recovery_s counts from the last clipped sample, which the pulse's edge taper (window_alpha) puts "
                            "before the nominal end, the earlier the less a hydrophone is overloaded")
    out["recovery"] = {}
    for pulse in pulses:
        blocks = {}
        for n in set_names:
            blocks[n] = sets[n]
            if n == "gain_switch":
                blocks[n] = {"gain_switch": {"gain_db": -float(np.ceil(max(over.max(), 0.0)) + 6.0), "until": pulse + r.max() / c + 1e-3, "ramp": 1e-3}}
        tails = [closed_form_recovery(b, gain_tolerance_db, offset_tolerance, pulse, gain_db) or 0.0 for b in blocks.values()]
        duration = pulse + r.max() / c + 1.3 * max(tails) + 0.02
        base = {"transmit": {"direct_path": {"boundaries": False}, "waveform": {"duration": float(pulse)}},
                "receive": {"window": {"range": [0.0, 0.5 * c * (duration - pulse)]}}}
        merged = copy.deepcopy(overrides or {})
        for key, value in base.items():
            merged[key] = {**merged.get(key, {}), **value}
        linear_model = SonarModel.load(name, merged)
        linear = simulate_ping(linear_model, make_pose(), Scene(), environment={"sound_speed": c}, adc=False).samples.array
        pulse_end = pulse + r / c
        rows = []
        say(f"\n  pulse {pulse:g} s (direct pulse over at {pulse_end.min():.4f} to {pulse_end.max():.4f} s; blanking alone: minimum range "
            f"{0.5 * c * pulse_end.max():.1f} m). Recovery sets are ILLUSTRATIVE.")
        say("    set                               recovery after pulse, s (worst / best)   clean after transmit, s   minimum range, m   closed form from the last clip, s")
        for n, block in blocks.items():
            merged_set = copy.deepcopy(merged)
            merged_set["receive"] = {**merged_set["receive"], "overload": block}
            set_model = SonarModel.load(name, merged_set)
            record = set_model.make_record(make_pose(), environment={"sound_speed": c})
            ocfg = rx.overload_config(set_model)
            rx.report_gain_switch(record, ocfg)
            truth = {}
            rx.apply_front_end(record, linear, ocfg, truth=truth, arrays=True)
            clean = np.maximum(rx.recovery_time(truth, 0.0, gain_tolerance_db, offset_tolerance), pulse_end)
            closed = closed_form_recovery(block, gain_tolerance_db, offset_tolerance, pulse, gain_db)
            row = {"set": n, "overload": block, "illustrative": True,
                   "peak_db_re_adc": spread(truth["peak_db_re_adc_limit"]),
                   "recovery_after_pulse_s": dict(spread(clean - pulse_end), per_channel=[round(float(v), 5) for v in clean - pulse_end]),
                   "clean_after_transmit_s": spread(clean), "minimum_range_m": spread(0.5 * c * clean), "closed_form_recovery_s": closed}
            rows.append(row)
            if n == "clip_only":
                diff = truth["peak_db_re_adc_limit"] - over
                out["simulated_minus_closed_form_db"] = {"max_abs": float(np.abs(diff).max()), "pulse_s": float(pulse)}
            a, b_, m_ = row["recovery_after_pulse_s"], row["clean_after_transmit_s"], row["minimum_range_m"]
            say(f"    {n:34s} {a['worst']:9.4f} / {a['best']:7.4f}                  {b_['worst']:9.4f}             {m_['worst']:8.1f}        "
                + ("-" if closed is None else f"{closed:.4f}"))
        out["recovery"][f"{pulse:g}"] = rows
    if "simulated_minus_closed_form_db" in out:
        say(f"\n  simulator's direct level through its front end against the closed form: within {out['simulated_minus_closed_form_db']['max_abs']:.3f} dB")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sonars", nargs="*", default=list(DEFAULT_SONARS))
    ap.add_argument("--pulses", nargs="+", type=float, default=[0.1, 1.0], help="pulse lengths, s")
    ap.add_argument("--sets", nargs="+", default=None, help="recovery parameter sets (ILLUSTRATIVE); default: all")
    ap.add_argument("--gain-tolerance-db", type=float, default=1.0, help="recovered when the gain is within this of nominal (1 dB)")
    ap.add_argument("--offset-tolerance-db", type=float, default=-40.0, help="and the offset at the ADC is below this re full scale (-40 dB)")
    ap.add_argument("--overrides", default=None, help="JSON of sonar overrides (for example another transmit.lever_arm)")
    ap.add_argument("--out", default=str(ROOT / "docs" / "design" / "direct_sound_study.json"))
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    overrides = json.loads(args.overrides) if args.overrides else None
    set_names = args.sets or list(recovery_sets(0.0))
    result = {name: study(name, args.pulses, set_names, args.gain_tolerance_db, args.offset_tolerance_db, overrides, args.quiet)
              for name in args.sonars}
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)
    if not args.quiet:
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
