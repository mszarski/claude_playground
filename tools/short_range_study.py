#!/usr/bin/env python3
"""The short-range mode of the long-range sonar: the options compared with numbers (docs/design/short_range_mode.md).

Every number this prints is ours: the link budget and depth planner of the suite (gpusonar.buildspec, gpusonar.depthplan,
the same functions as tools/popoto_trade_study.py and the POP-3 check), the suite's own hyperbolic sweep
(gpusonar.sim.waveform, `source: hfm`) for the cross-talk, and arithmetic for the timing. Nothing is measured on water.

    python3 tools/short_range_study.py                 # every section, a summary on stdout
    python3 tools/short_range_study.py --json OUT.json # and every figure into OUT.json
    python3 tools/short_range_study.py --no-bellhop    # skip the three sites with a sound-speed profile

Sections:
  pulses     the long pulse (the D-256 baseline: 2 s hyperbolic, 4-6 kHz, 197 dB rms) and the short candidates at the same
             level and band: energy, blind range c T / 2, energy deficit.
  reach      the planner's margin for the owner's three classes (configs/builds/targets/reference_classes.yaml, low target
             strength) at 1.5 and 2.25 km and the reach (last range held, the planner's rule), per illustrative site, one row
             of 32 (the face's first build) and three rows: what the short pulse holds where the long one is blind.
  schedule   the ping interval of each mode (the window's two-way time plus the pulse), and the update rate each mode keeps
             in an alternating schedule (option a), against the long mode alone.
  crosstalk  two sweeps of the same band in one ping (option b): how much the other pulse's matched filter rejects an echo
             (a point: the peak) and a reverberation (diffuse: the mean power), for an up-sweep and a down-sweep; and the
             level of a near echo over a far one.
  notch      a short ping sent inside the long ping's listening time (option a, interleaved): what its transmission blanks
             and what its own reverberation adds to the long ping's background, by range, on the shelf site.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

C = 1500.0                                   # m/s: the builds' nominal sound speed
LONG = {"duration_s": 2.0, "source_level_db": 197.0}     # D-256, configs/sonars/popoto_long_range.yaml
SHORT = (0.1, 0.25)                          # s: the candidates of the brief (blind range 75 and 187.5 m)
BAND = (4000.0, 6000.0)
LONG_RANGE_M = (1500.0, 12700.0)             # the long window: from its blind range to the 12 km design range plus 700 m (popoto_12km's)
SHORT_RANGE_M = 3000.0                       # the short window's far edge: twice the long mode's blind range
HANDOVER_M = 1500.0
RANGES = (200.0, 400.0, 700.0, 1000.0, 1500.0, 2250.0, 3000.0, 4000.0, 6000.0, 8000.0, 12000.0)


def pulses() -> dict:
    e_long = LONG["source_level_db"] + 10.0 * np.log10(LONG["duration_s"])
    out = {"long": {"duration_s": LONG["duration_s"], "source_level_db_rms": LONG["source_level_db"], "energy_db": e_long,
                    "blind_range_m": 0.5 * C * LONG["duration_s"]}, "short": {}}
    for t in SHORT:
        e = LONG["source_level_db"] + 10.0 * np.log10(t)
        out["short"][f"{t:g}"] = {"duration_s": t, "source_level_db_rms": LONG["source_level_db"], "energy_db": e, "deficit_db": e_long - e,
                                  "blind_range_m": 0.5 * C * t, "time_bandwidth": t * (BAND[1] - BAND[0])}
    return out


def _spec(rows: int, duration_s: float, range_m):
    import popoto_trade_study as ts
    spec = ts.variant_spec(rows, 32, "2k", duration_s, LONG["source_level_db"], "hfm")
    spec["operation"]["range_m"] = [None, float(range_m)]
    return spec


def reach(bellhop: bool = True, rows_list=(1, 3), log=print) -> dict:
    from gpusonar import buildspec as bs
    from gpusonar import depthplan, target_classes
    from gpusonar.sim.bellhop import find_bellhop
    envs = depthplan.load_environments("illustrative")
    classes = target_classes.resolve(target_classes.load("reference_classes"), *BAND)
    out = {"ranges_m": list(RANGES), "handover_m": HANDOVER_M, "sites": {}}
    for env in envs:
        name = env["name"]
        if env.get("profile") is not None and (not bellhop or find_bellhop() is None):
            out["sites"][name] = {"skipped": "a sound-speed profile needs Bellhop (GPUSONAR_BELLHOP)"}
            continue
        site = out["sites"][name] = {}
        cache = {}
        for rows in rows_list:
            for t in (LONG["duration_s"],) + SHORT:
                far = LONG_RANGE_M[1] if t == LONG["duration_s"] else SHORT_RANGE_M
                d = bs.derive(_spec(rows, t, far))
                rr = np.asarray([r for r in RANGES if r > 0.5 * C * t], dtype=float)
                P = depthplan.plan(d, env, classes, range_of_interest=HANDOVER_M, ranges=rr, cache=cache, keep_grid=True)
                key = f"{rows}row_{t:g}s"
                site[key] = {}
                for c in classes:
                    e = P["classes"][c["name"]]
                    if "skipped" in e:
                        site[key][c["name"]] = {"skipped": e["skipped"]}
                        continue
                    k = int(np.argmin(np.abs(np.asarray(P["own_depths_m"]) - e["recommended"]["own_depth_m"])))
                    row = np.asarray(e["margin_worst_db"])[k]                    # [range] at the own depth the plan recommends at 1.5 km
                    held = row >= 0.0
                    first_gap = next((float(rr[i]) for i in range(len(rr)) if not held[i]), None)
                    site[key][c["name"]] = {"own_depth_m": e["recommended"]["own_depth_m"],
                                            "margin_by_range_db": {f"{r:g}": float(m) for r, m in zip(rr, row)},
                                            "held_from_blind_range_to_m": float(rr[np.flatnonzero(~held)[0] - 1]) if (~held).any() and np.flatnonzero(~held)[0] > 0
                                            else (float(rr[-1]) if held.all() else 0.0),
                                            "first_range_not_held_m": first_gap,
                                            "limited_by_at_1500m": e["recommended"].get("limited_by")}
                log(f"{name} {key}: " + ", ".join(f"{c} {v['margin_by_range_db'].get('1500', float('nan')):+.1f} dB at 1.5 km, held to {v['held_from_blind_range_to_m']:g} m"
                                                    for c, v in site[key].items() if "margin_by_range_db" in v))
    return out


def schedule(p: dict) -> dict:
    """Ping intervals: a window [r0, r1] closes 2 r1 / c + T after the transmission starts (gpusonar.sonar.SonarModel.window)."""
    long_i = 2.0 * LONG_RANGE_M[1] / C + LONG["duration_s"]
    out = {"long_alone_interval_s": long_i, "short": {}}
    for t in SHORT:
        short_i = 2.0 * SHORT_RANGE_M / C + t
        cyc = long_i + short_i
        out["short"][f"{t:g}"] = {
            "short_interval_s": short_i,
            "alternating_1to1": {"cycle_s": cyc, "long_update_s": cyc, "short_update_s": cyc, "long_rate_kept": long_i / cyc,
                                 "long_rate_cost_percent": 100.0 * (1.0 - long_i / cyc)},
            "one_short_every_3_long": {"cycle_s": 3 * long_i + short_i, "long_update_s": (3 * long_i + short_i) / 3.0,
                                       "short_update_s": 3 * long_i + short_i, "long_rate_kept": 3 * long_i / (3 * long_i + short_i)},
            "short_alone_interval_s": short_i}
    return out


def _hfm(duration: float, fs: float, up: bool = True) -> np.ndarray:
    from gpusonar.sim.waveform import build_waveform
    f0, f1 = (BAND if up else BAND[::-1])
    w = build_waveform({"type": "sampled", "source": "hfm", "f_start": f0, "f_stop": f1, "duration": duration, "window_alpha": 0.1})
    return np.asarray(w.baseband(fs, 0.5 * (BAND[0] + BAND[1])), dtype=complex)


def _xcorr(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    n = len(x) + len(h) - 1
    m = 1 << int(np.ceil(np.log2(n)))
    return np.fft.ifft(np.fft.fft(x, m) * np.conj(np.fft.fft(h, m)))     # every lag, negative ones at the end (m >= n: no wrap)


def crosstalk(fs: float = 8000.0) -> dict:
    """A pulse a through the matched filter of b. `point`: the largest output power over the largest of b through its own
    filter, at the same peak level (the same source level): what a point echo of the other pulse leaves at the peak. `diffuse`:
    the mean output power over all lags (a reverberation of the other pulse: many echoes at random delays) over b's own, in
    dB; by Parseval it is the overlap of the two spectra, which for one band is the energy ratio: no rejection."""
    long_up = _hfm(LONG["duration_s"], fs)
    out = {"sample_rate_hz": fs, "pairs": {}}

    def pair(a, b):
        own = _xcorr(b, b)
        x = _xcorr(a, b)
        point = 10.0 * np.log10(np.max(np.abs(x)) ** 2 / np.max(np.abs(own)) ** 2)
        diffuse = 10.0 * np.log10(np.sum(np.abs(x) ** 2) / np.sum(np.abs(own) ** 2))
        return {"point_db": float(point), "diffuse_db": float(diffuse)}

    for t in SHORT:
        for up in (True, False):
            s = _hfm(t, fs, up)
            tag = f"{t:g}s_{'up' if up else 'down'}"
            out["pairs"][tag] = {"long_echo_through_short_filter": pair(long_up, s), "short_echo_through_long_filter": pair(s, long_up),
                                 "energy_ratio_long_over_short_db": float(10.0 * np.log10(np.sum(np.abs(long_up) ** 2) / np.sum(np.abs(s) ** 2)))}
    # Level of a near echo over a far one: spherical spreading both ways and absorption (Ainslie-McColm at 5 kHz, the suite's).
    from gpusonar import acoustics
    alpha = float(acoustics.absorption_ainslie_mccolm(5000.0))
    near_far = {}
    for rn in (500.0, 1000.0, 1500.0):
        for rf in (6000.0, 12000.0):
            near_far[f"{rn:g}_vs_{rf:g}"] = float(40.0 * np.log10(rf / rn) + 2.0 * alpha * (rf - rn))
    out["near_over_far_echo_db"] = near_far
    out["absorption_db_per_m"] = alpha
    return out


def notch(t_short: float = 0.25, t_send_list=(2.0, 4.0, 8.0, 12.0), bellhop: bool = False) -> dict:
    """A short ping sent at t_s after the long one's start, inside its listening. (1) Its transmission blanks [t_s, t_s + T_s]:
    a long echo from R spans [2R/c, 2R/c + T_l], so every R with 2R/c in [t_s - T_l, t_s + T_s] loses up to T_s / T_l of its
    energy (blanked, the receiver recovering at once: unconfirmed). (2) Its reverberation from range r arrives at
    t_s + 2r/c, where the long ping hears R = c t_s / 2 + r; through the long filter (same band) it carries the energy ratio
    E_s / E_l (crosstalk `diffuse`), so it raises the long ping's background by the short ping's reverberation-to-noise
    ratio at r, less that ratio, over the long ping's own at R. Shelf site (isovelocity, Lambert -27 dB), 1 row, the planner's
    reverberation level (depthplan.reverberation_of)."""
    from gpusonar import buildspec as bs
    from gpusonar import depthplan
    env = next(e for e in depthplan.load_environments("illustrative") if e["name"] == "shelf_100m_sand")
    d = bs.derive(_spec(1, LONG["duration_s"], LONG_RANGE_M[1]))
    env_l, _ = depthplan._local_water(env)
    eq, _ = bs.budget(d, depthplan.wind_of(env_l))
    rr = np.arange(100.0, 12701.0, 50.0)
    own = np.asarray([50.0])
    rl, note = depthplan.reverberation_of(d, env_l, own, rr, eq.absorption_db_per_m(), {})
    snr0 = np.asarray(eq.snr(rr), dtype=float)                       # a 0 dB target's echo over noise (one row, the long pulse)
    two_tl = 2.0 * np.asarray(eq.transmission_loss(rr), dtype=float)
    rnr_long = snr0 + two_tl + np.asarray(rl[0], dtype=float)        # reverberation over noise of the long ping, by range
    ratio_db = 10.0 * np.log10(t_short / LONG["duration_s"])          # E_s / E_l at the same level
    out = {"site": "shelf_100m_sand", "own_depth_m": 50.0, "t_short_s": t_short, "reverberation": note,
           "long_reverberation_over_noise_db": {f"{r:g}": float(v) for r, v in zip(rr[::20], rnr_long[::20])}, "sends": {}}
    for ts_ in t_send_list:
        lo = 0.5 * C * (ts_ - LONG["duration_s"])
        hi = 0.5 * C * (ts_ + t_short)
        R_hear = 0.5 * C * ts_ + rr                                  # the long ping's range while the short one's echo from r arrives
        inside = R_hear <= LONG_RANGE_M[1]
        rise = np.full(rr.shape, np.nan)
        bg = 10.0 ** (np.interp(R_hear[inside], rr, rnr_long) / 10.0) + 1.0
        rise[inside] = 10.0 * np.log10(1.0 + 10.0 ** ((rnr_long[inside] + ratio_db) / 10.0) / bg)
        over3 = R_hear[inside][rise[inside] > 3.0]
        over1 = R_hear[inside][rise[inside] > 1.0]
        out["sends"][f"{ts_:g}"] = {"blanked_echo_ranges_m": [max(lo, LONG_RANGE_M[0]), hi], "largest_energy_lost_db": float(-10.0 * np.log10(1.0 - t_short / LONG["duration_s"])),
                                     "background_raised_3db_m": [float(over3.min()), float(over3.max())] if over3.size else None,
                                     "background_raised_1db_m": [float(over1.min()), float(over1.max())] if over1.size else None,
                                     "rise_db_by_long_range": {f"{R:g}": float(v) for R, v in zip(R_hear[inside][::10], rise[inside][::10])}}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", help="write every figure here")
    ap.add_argument("--no-bellhop", action="store_true", help="skip the sites with a sound-speed profile")
    ap.add_argument("--sections", nargs="*", default=["pulses", "reach", "schedule", "crosstalk", "notch"])
    a = ap.parse_args(argv)
    res = {"status": "ours: the suite's planner and waveform, arithmetic; nothing measured on water"}
    p = pulses()
    if "pulses" in a.sections:
        res["pulses"] = p
        print("pulses:", json.dumps(p, indent=1))
    if "schedule" in a.sections:
        res["schedule"] = schedule(p)
        print("schedule:", json.dumps(res["schedule"], indent=1))
    if "crosstalk" in a.sections:
        res["crosstalk"] = crosstalk()
        print("crosstalk:", json.dumps(res["crosstalk"], indent=1))
    if "notch" in a.sections:
        res["notch"] = notch()
        n = copy.deepcopy(res["notch"])
        for s in n["sends"].values():
            s.pop("rise_db_by_long_range")
        print("notch:", json.dumps(n, indent=1))
    if "reach" in a.sections:
        res["reach"] = reach(bellhop=not a.no_bellhop)
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
