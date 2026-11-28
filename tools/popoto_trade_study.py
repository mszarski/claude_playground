#!/usr/bin/env python3
"""Design trade study for the Popoto long-range array at 12 km (issue sonar-3983): row length, rows, band, pulse, source
level, own depth, and processing credits, per planning site and target class, with the reverberation term ON.

    tools/popoto_trade_study.py --grid          # the grid over the illustrative sites -> docs/design/popoto_trade_study.json
    tools/popoto_trade_study.py --sensitivity   # each site's Lambert +-3 dB and wind +-1 sea state (merged into the JSON)
    tools/popoto_trade_study.py --coverage      # the coverage picture (C++) at the build's site, at broadside and the sector's edge
    tools/popoto_trade_study.py --measure-rows  # simulator: how much stacked rows reject the shelf's reverberation (minutes)
    tools/popoto_trade_study.py --crosscheck    # simulator + live chain at two grid points against the study's prediction
    tools/popoto_trade_study.py --transmit      # transmit options (level x pulse) on the shipped face by sea state and site (merged into the JSON)
    tools/popoto_trade_study.py --report        # docs/design/popoto_trade_study.md from the JSON
    tools/popoto_trade_study.py --quick --out /tmp/x   # one variant, one site, one class: what the test runs

What it does. Every variant is built IN MEMORY from a shipped face build spec (configs/builds/popoto_face_{1,2,3}x32.yaml:
rows, heights, boards, deployment) with the row's hydrophone count, the band, the pulse and the source level replaced; no
shipped build, sonar, environment or deployment file is edited. Each variant is planned with the repository's own model,
gpusonar.depthplan.plan (the link budget of buildspec.budget, the build's own detector threshold of buildspec.detection,
the two-way factor of gpusonar.lloyd or Bellhop, and the site's boundary reverberation of the coverage picture's C++ term),
at the planner's recommended own depth, and reported as: margin at 12 km (dB over what the build's detector needs for a
detection probability of 0.5 on one ping), the term that limits it, the reach; beside it the noise-only figure.

Rows against reverberation. The planner takes stacked rows to gain nothing against reverberation (ours; it arrives by the
same multipath as the echo). That is one bound (A). The other (B) gives the rows their full 10 log10(rows) against it, as
a vertical beam would against reverberation arriving from outside it. Both are reported; --measure-rows measures where the
shelf's truth lies between them.

Credits applied on top, each with its source and status: the sweep's replica bank (+1.9 dB on the echo, measured on the
12 km ship in docs/tasks/doppler_baseline.json, ours); a Doppler-resolving Hann tone with a Doppler bank (removes the
reverberation for a mover outside the reverberation's ridge, at 4.1 dB of peak power, docs/tasks/DOPPLER.md Sec. 3, ours;
none for a stationary or crossing target; needs the owner's answer on the transmitter); a baffle behind the face (3 dB
less reverberation, ours: the reverberation term's own statement of what it leaves out).

Bellhop (a separately run GPL-3 program, never linked) is needed for the sites with a sound-speed profile: set
GPUSONAR_BELLHOP; without it those sites are skipped and the JSON says so. OMP_NUM_THREADS=1 is advised.
"""
from __future__ import annotations

import argparse
import copy
import datetime
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

OUT = ROOT / "docs" / "design" / "popoto_trade_study"
BASE = {1: "popoto_face_1x32", 2: "popoto_face_2x32", 3: "popoto_face_3x32"}
LENGTHS = (32, 64, 96)
ROWS = (1, 2, 3)
RANGE_M = 12000.0
# The shipped transmitter: since D-256 (the owner, 2026-11-28) a HYPERBOLIC sweep of 2 s at 197 dB rms
# (configs/sonars/popoto_long_range.yaml; tests/python/test_hfm_baseline.py holds the two together). Until then a 1 s LINEAR
# sweep at 197 dB: FORMER below, the waveform the grid of sections 1 to 6 was run at (`--grid`, `--sensitivity`, `--coverage`:
# their stored figures are of that waveform and stay reproducible from the JSON; the planner is waveform-blind, so a cell
# differs from the 2 s hyperbolic sweep's only by the energy and the detector's window: section 0's `2 s 197 dB` column).
SHIPPED = {"band_hz": (4000.0, 6000.0), "pulse_s": 2.0, "source_level_db": 197.0, "waveform": "hfm"}
FORMER = {"band_hz": (4000.0, 6000.0), "pulse_s": 1.0, "source_level_db": 197.0, "waveform": "lfm"}
FORMER_WORDS = "4-6 kHz, 1 s linear, 197 dB: the shipped waveform until D-256, at which the grid was run"
# Band options (centre kept at 5 kHz). docs/design/projector_selection.md Sec. 1.2: "A narrower sweep is the fallback if no projector
# covers the band ... A 1 kHz band (20%) needs Q <= 5"; the 4 kHz band (3-7 kHz) is inside the stated operating ranges of the Neptune
# T335 ("3-8 kHz") and Massa TR-1075 ("2.5-10 kHz") of Sec. 3.1, but no level at 3 or 7 kHz is given there (Sec. 5 asks for the TVR
# from 3 to 7 kHz): unconfirmed.
BANDS = {"2k": {"band_hz": (4000.0, 6000.0), "projector": "shipped band: T335 or TR-1075A/D by their sheets (projector_selection.md Sec. 4, option A)", "status": "via secondary (the sheets as read in projector_selection.md)"},
         "1k": {"band_hz": (4500.0, 5500.0), "projector": "any 4-6 kHz candidate; a 1 kHz band needs a mechanical Q of 5 or less (projector_selection.md Sec. 1.2)", "status": "via secondary"},
         "4k": {"band_hz": (3000.0, 7000.0), "projector": "T335 (sheet: 3-8 kHz) or TR-1075 (2.5-10 kHz) by their stated ranges; no level at 3 or 7 kHz is on record (projector_selection.md Sec. 3.1, 5): unconfirmed. "
                "The row's pitch (0.15 m, half a wavelength at 5 kHz) is 0.7 wavelength at 7 kHz: grating lobes inside the searched sector", "status": "unconfirmed"}}
# Pulse and the source level the design document pairs with it: same energy (projector_selection.md Sec. 1.8 table; Sec. 4 option B).
PULSES = {1.0: 197.0, 2.0: 194.0, 4.0: 191.0}
SL_STEPS = (-3.0, 3.0)
# Credits on top (each: value, source, status). Numbers are quoted from the files named, not computed here.
CREDITS = {
    "sweep_bank": {"db": 1.9, "applies_to": "the echo (noise and reverberation terms alike): a bank of time-scaled sweep replicas recovers the sweep's Doppler mismatch loss",
                   "source": "docs/tasks/doppler_baseline.json via docs/tasks/DOPPLER.md Sec. 3.2: popoto_12km ship at 12 km, range rate -0.50 m/s, 4-6 kHz 1 s sweep: 24.0 -> 25.9 dB (+1.9)",
                   "status": "ours (simulated)", "limits": "measured at one radial speed (-0.50 m/s) on this band; larger at higher speed (flank_32 6-10 kHz: +5.9 dB at 2 m/s), not measured here at the classes' speeds; "
                   "nothing for a target with no radial speed"},
    "tone": {"peak_power_db": 4.1, "min_radial_m_s": 0.8,
             "applies_to": "a moving target outside the reverberation's Doppler ridge: the reverberation term is removed (the bank's margin reaches its noise ceiling); taken here as the noise-only margin "
                           "less 4.1 dB (a Hann tone of the sweep's energy needs 4.1 dB more peak power: with a peak-limited projector it carries 4.1 dB less energy)",
             "source": "docs/tasks/DOPPLER.md Sec. 3.2-3.3 (flank_patrol small vehicle: 7.0 -> 39.8 dB, at its ceiling; +33 to +37 dB); 4.1 dB: Sec. 0 item 4",
             "status": "ours (simulated); the tone must lie outside the sweep's band in the same ping, or on its own pings (waveform package d2-train, ours: inside the band it gains nothing over a reverberating bottom)",
             "limits": "the 0.8 m/s radial speed is ours: the baseline's full recovery at +0.5 m/s with a 1 s tone at 8 kHz, scaled by the speed resolution c/(2 f T) to 5 kHz; nothing for a stationary or "
                       "crossing target (a 12 km crossing ship's radial speed is about 0.4 m/s: DOPPLER.md Sec. 5 q. 2); a single tone fades in multipath (18.9 dB less echo energy than the sweep on one "
                       "image-source path set, DOPPLER.md Sec. 0 item 3): not in the figure; range resolution becomes c T / 2 = 750 m"},
    "baffle": {"db": 3.0, "applies_to": "the reverberation only: a baffled face would remove about 3 dB of reverberation from behind (the model has no element pattern; omnidirectional elements hear the back half)",
               "source": "docs/PHYSICS_NOTES.md Sec. 32, 'Not in it': \"a baffled face would take about 3 dB of reverberation from behind it out\"", "status": "ours, unconfirmed (no baffle is designed or measured)"},
}
DATA = {"hydrophones_per_board": 8, "board_inputs_status": "confirmed: the owner's plan (PMM6081 ICD) as quoted in configs/builds/popoto_face_3x32.yaml and docs/design/popoto_parts.md",
        "rate_status": "unconfirmed: 102.4 kHz x 24 bits are assumptions (configs/builds/popoto_face_3x32.yaml: boards.sample_rate_hz, adc_bits; popoto_parts.md 'Capture boards')",
        "batching_quote": "the owner's plan, p. 14, as quoted in tools/parts_list.py: \"When to stop: past two boards, the custom 32-channel ADC board in Phase 6 is cheaper and removes the problem with one shared clock.\"",
        "price_quote": "the owner's plan, p. 13, as quoted in docs/design/popoto_parts.md: \"$7,825 development kit; $6,150 OEM board\""}


def _imports():
    from gpusonar import buildspec as bs
    from gpusonar import depthplan, reverb_budget
    return bs, depthplan, reverb_budget


_ARGV: list | None = None


def command_line() -> str:
    """The command as run (main's arguments), without --out (where it was written is not part of what it computed)."""
    a, keep = (sys.argv[1:] if _ARGV is None else _ARGV), []
    skip = False
    for x in a:
        if skip:
            skip = False
            continue
        if x == "--out":
            skip = True
            continue
        if not x.startswith("--out="):
            keep.append(x)
    return "tools/popoto_trade_study.py " + " ".join(keep)


def git_head() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


# --- variants ------------------------------------------------------------------------------------------------

def variant_spec(rows: int, per_row: int, band="2k", pulse_s: float = 1.0, source_level_db: float | None = None, waveform: str = "lfm") -> dict:
    """A face build spec with `rows` rows of `per_row` hydrophones at the module's pitch, the band, pulse and source level
    given; everything else is the shipped face build of that many rows. In memory only. `waveform` (the spec's
    `projector.waveform.type`): `lfm` by default, the sweep every stored grid of this study was run with; the planner is
    waveform-blind but for the detector's template record (1e-4 dB between an `lfm` and an `hfm` of the same length), so the
    stored figures stay reproducible to their printed digits and beyond since the shipped face became `hfm` (D-256)."""
    bs, _, _ = _imports()
    spec = copy.deepcopy(bs.load(BASE[rows]))
    spec["array"]["count"] = int(per_row)
    if int(per_row) != 32:
        spec["boards"]["clocks"] = {"seed": 1}          # the explicit list has one entry per shipped board; clocks do not touch the budget or the detector
    spec["projector"]["band_hz"] = [float(v) for v in BANDS[band]["band_hz"]]
    spec["projector"]["waveform"] = dict(spec["projector"]["waveform"], type=str(waveform), duration_s=float(pulse_s))
    spec["projector"]["source_level_db_rms"] = float(PULSES.get(float(pulse_s), SHIPPED["source_level_db"]) if source_level_db is None else source_level_db)
    spec["build"] = f"trade_{rows}x{per_row}_{band}_{pulse_s:g}s_{spec['projector']['source_level_db_rms']:g}dB"
    return spec


def variant_id(rows, per_row, band, pulse_s, sl) -> str:
    return f"{rows}x{per_row}|{band}|{pulse_s:g}s|{sl:g}dB"


def geometry(spec: dict, d: dict) -> dict:
    """Aperture, beam, fit on the face, boards and data rates of a variant (hardware facts quoted with their status)."""
    per_row, rows = int(spec["array"]["count"]), int(spec["array"]["rows"])
    pitch = float(spec["array"]["pitch_m"])
    plat = spec["platform"]
    aperture = per_row * pitch
    extent = (per_row - 1) * pitch
    usable, span = float(plat.get("usable_span_m") or plat["span_m"]), float(plat["span_m"])
    inputs = int(spec["boards"]["inputs"])
    boards_per_row = int(np.ceil(per_row / inputs))
    n = rows * per_row
    return {"rows": rows, "per_row": per_row, "hydrophones": n, "pitch_m": pitch, "aperture_m": aperture, "row_extent_m": extent,
            "beamwidth_deg": float(d["beamwidth_deg"]), "beamwidth_at_60deg_steer_deg": float(d["beamwidth_deg"]) / np.cos(np.radians(60.0)),
            "face_span_m": span, "face_usable_span_m": usable,
            "fits_usable_span": bool(extent <= usable + 1e-9), "fits_face": bool(extent <= span + 1e-9),
            "fit_note": ("fits the 7 m working span" if extent <= usable + 1e-9 else
                         f"does NOT fit: {extent:.2f} m of hydrophones against the face's {span:g} m (owner-stated) and the {usable:g} m working span (owner's working value)"),
            "boards": rows * boards_per_row, "boards_per_row": boards_per_row,
            "channel_rate_bit_s": float(d["rate_channel_bit_s"]), "board_rate_bit_s": float(d["rate_board_bit_s"]), "total_rate_bit_s": float(d["rate_channel_bit_s"]) * n,
            "record_bytes_per_hour_raw": float(d["record_bytes_per_hour"]), "ping_interval_s": float(d["ping_interval_s"]), "blind_range_m": float(d["blind_range_m"]),
            "band_hz": [float(v) for v in d["band_hz"]], "pulse_s": float(d["pulse_s"]), "source_level_db": float(d["source_level_db"]),
            "grating_lobe_free_to_deg": float(np.degrees(np.arcsin(min(d["sound_speed"] / d["band_hz"][1] / pitch - 1.0, 1.0)))) if d["sound_speed"] / d["band_hz"][1] / pitch > 1.0 else 0.0}


# --- the evaluation: the planner, and its parts for the other bound and the credits -----------------------------

def _class_depths(zt, env, classes, c):
    deepest = float(env["water_depth_m"]) - 1.0 if env.get("water_depth_m") is not None else None
    lo, hi = c["depth_m"]
    top = min(hi, deepest) if deepest is not None else hi
    return (zt >= lo - 1e-6) & (zt <= top + 1e-6)


def _best(m: np.ndarray, own: np.ndarray, rr: np.ndarray, i_r: int) -> dict:
    """At the own depth that maximises the margin at the range of interest: margin, depth, reach (last range held, the planner's rule)."""
    k = int(np.argmax(m[:, i_r]))
    held = m[k] >= 0.0
    return {"own_depth_m": float(own[k]), "margin_db": float(m[k, i_r]), "reach_m": float(rr[np.flatnonzero(held)[-1]]) if held.any() else 0.0, "_k": k}


def parts(d: dict, env: dict, classes: list, cache: dict, ranges=None, range_m: float = RANGE_M) -> dict:
    """depthplan.plan of a variant (the official figures: bound A) and, from the same cache, the parts it sums: the noise-only
    margin grid, the echo-over-reverberation ratio and the reverberation level, per class [own depth, range]."""
    bs, depthplan, reverb_budget = _imports()
    P = depthplan.plan(d, env, classes, range_of_interest=range_m, ranges=ranges, cache=cache, keep_grid=True)
    env_l, _ = depthplan._local_water(env)
    eq, _ = bs.budget(d, depthplan.wind_of(env_l))
    det = bs.detection(d)
    own, rr, zt = (np.asarray(P[k], dtype=float) for k in ("own_depths_m", "ranges_m", "target_depths_m"))
    key = (env_l.get("name"), float(d["projector_m"][2]), float(d["band_hz"][0]), float(d["band_hz"][1]), own.tobytes(), zt.tobytes(), rr.tobytes())
    f3 = cache[key][P["statistic"]]
    rl, _ = depthplan.reverberation_of(d, env_l, own, rr, eq.absorption_db_per_m(), cache) if P["reverberation"]["included"] else (None, None)
    T, G = float(P["threshold_db"]), float(det["rows_gain_db"])
    free = np.asarray(eq.snr(rr)) - T
    two_tl = 2.0 * np.asarray(eq.transmission_loss(rr), dtype=float)
    i_r = int(np.argmin(np.abs(rr - range_m)))
    out = {"plan": P, "own": own, "rr": rr, "i_r": i_r, "T": T, "G": G, "rows": int(det["rows"]), "classes": {}}
    for c in classes:
        sel = _class_depths(zt, env_l, classes, c)
        if not sel.any():
            out["classes"][c["name"]] = None
            continue
        worst = f3[:, sel, :].min(axis=1)
        ts = float(c["ts_db"][0])
        noise = worst + free[None, :] + ts                                   # noise-only margin [own, range]
        srr = None if rl is None else worst + ts - two_tl[None, :] - rl      # echo over reverberation, one row [own, range]
        out["classes"][c["name"]] = {"noise": noise, "srr": srr, "ts_db": ts, "speed_m_s": c.get("speed_m_s")}
        a = margin(noise, srr, T, G, 0.0)
        got = np.asarray(P["classes"][c["name"]]["margin_at_range_worst_db"])
        if not np.allclose(a[:, i_r], got, atol=1e-6):                      # the parts must sum to the planner's own figure
            raise AssertionError(f"{c['name']} in {env.get('name')}: the parts do not reproduce depthplan.plan (largest difference {np.max(np.abs(a[:, i_r] - got)):.3g} dB)")
    return out


def margin(noise, srr, T: float, G: float, g_rev: float, echo_db: float = 0.0, rev_db: float = 0.0):
    """Margin (dB over the build's threshold of one row's SNR) from the noise-only margin and the echo-over-reverberation
    ratio, as depthplan.plan sums them, with `g_rev` dB of rows' gain against reverberation (bound A: 0; bound B: 10 log10 rows),
    `echo_db` added to the echo (a credit on both terms) and `rev_db` taken off the reverberation (a credit on that term)."""
    _, _, reverb_budget = _imports()
    n = noise + echo_db
    if srr is None:
        return n
    return reverb_budget.with_reverberation(n + T + G, srr + echo_db + rev_db + g_rev) - T - G


def summarise(pp: dict, name: str, speed=None) -> dict | None:
    """One class at one site for one variant: bound A (the planner's), bound B, the noise-only planner, the reverberation-only
    ceiling (what no source level exceeds), the limiting term, and each credit."""
    c = pp["classes"].get(name)
    if c is None:
        return None
    own, rr, i_r, T, G, rows = pp["own"], pp["rr"], pp["i_r"], pp["T"], pp["G"], pp["rows"]
    g_full = 10.0 * np.log10(rows)
    noise, srr = c["noise"], c["srr"]
    out = {"noise_only": _clean(_best(noise, own, rr, i_r))}
    for bound, g in (("A", 0.0), ("B", g_full)):
        m = margin(noise, srr, T, G, g)
        b = _best(m, own, rr, i_r)
        k = b.pop("_k")
        e = {**b, "rows_gain_against_reverberation_db": float(g), "margin_noise_only_there_db": float(noise[k, i_r])}
        if srr is not None:
            ceiling = float(srr[k, i_r] + g - T - G)                         # infinite source level: the reverberation alone
            noise_term = float(noise[k, i_r] + T + G)                         # one row's SNR over noise, rows' gain added
            rev_term = float(srr[k, i_r] + g)
            e.update(reverberation_only_ceiling_db=ceiling, limited_by="reverberation" if rev_term < noise_term else "noise",
                     reverberation_over_noise_db=float(noise_term - rev_term), source_level_can_close=bool(ceiling >= 0.0),
                     best_ceiling_over_own_depth_db=float(np.max(srr[:, i_r] + g - T - G)))
        else:
            e.update(reverberation_only_ceiling_db=None, limited_by="noise", reverberation_over_noise_db=None, source_level_can_close=True,
                     best_ceiling_over_own_depth_db=None)
        cr = {}
        moving = speed is not None and float(speed[1]) > 0.0
        cr["sweep_bank"] = _clean(_best(margin(noise, srr, T, G, g, echo_db=CREDITS["sweep_bank"]["db"]), own, rr, i_r)) if moving else None
        cr["baffle"] = _clean(_best(margin(noise, srr, T, G, g, rev_db=CREDITS["baffle"]["db"]), own, rr, i_r))
        cr["bank_and_baffle"] = _clean(_best(margin(noise, srr, T, G, g, echo_db=CREDITS["sweep_bank"]["db"] if moving else 0.0, rev_db=CREDITS["baffle"]["db"]), own, rr, i_r))
        tone_ok = speed is not None and float(speed[1]) >= CREDITS["tone"]["min_radial_m_s"]
        cr["tone"] = _clean(_best(noise - CREDITS["tone"]["peak_power_db"], own, rr, i_r)) if tone_ok else None
        cr["tone_needs"] = (f"a radial speed of {CREDITS['tone']['min_radial_m_s']:g} m/s or more (the class's speeds: {speed[0]:g}-{speed[1]:g} m/s; "
                            f"{'never reached' if not tone_ok else 'not at its slowest or crossing' if float(speed[0]) < CREDITS['tone']['min_radial_m_s'] else 'reached when not crossing'})") if speed is not None else None
        e["credits"] = cr
        out[bound] = e
    out["met_12km"] = {"A": bool(out["A"]["margin_db"] >= 0.0), "B": bool(out["B"]["margin_db"] >= 0.0), "noise_only": bool(out["noise_only"]["margin_db"] >= 0.0)}
    return out


def _clean(b: dict) -> dict:
    return {k: v for k, v in b.items() if not k.startswith("_")}


# --- the grid ---------------------------------------------------------------------------------------------------

def plan_variants(full: bool, lengths=LENGTHS, rows=ROWS, bands=None, pulses=None, sl_steps=None) -> list:
    """(rows, per_row, band, pulse, source level): every geometry at the shipped waveform, and one factor at a time on top of
    each geometry: the other bands, the other pulses at their paired levels, the source level +-3 dB."""
    bands = list(BANDS) if bands is None else bands
    pulses = list(PULSES) if pulses is None else pulses
    sl_steps = list(SL_STEPS) if sl_steps is None else sl_steps
    out = []
    for r in rows:
        for n in lengths:
            out.append((r, n, "2k", 1.0, SHIPPED["source_level_db"]))
            if not full:
                continue
            out += [(r, n, b, 1.0, SHIPPED["source_level_db"]) for b in bands if b != "2k"]
            out += [(r, n, "2k", p, PULSES[p]) for p in pulses if p != 1.0]
            out += [(r, n, "2k", 1.0, SHIPPED["source_level_db"] + s) for s in sl_steps]
    return out


def ranges_for(d: dict, base_rr: np.ndarray | None):
    """Ranges for a variant whose pulse is longer than the shipped one: the shipped grid beyond its blind range (so the factor is
    the same numbers, sliced, not recomputed); None (the planner's own grid) otherwise."""
    if base_rr is None or d["pulse_s"] <= 1.0 + 1e-9:
        return None
    lo = max(1.2 * float(d["blind_range_m"]), 1000.0)
    return np.unique(np.r_[base_rr[base_rr >= lo], RANGE_M])


def seed_sliced_factor(cache: dict, env_name: str, proj_z: float, band, base_rr: np.ndarray, rr: np.ndarray):
    """Put the factor on `rr` (a subset of `base_rr`) into the planner's cache under its own key, from the base grid's entry."""
    idx = np.searchsorted(base_rr, rr)
    for key in list(cache):
        if isinstance(key, tuple) and len(key) == 7 and key[0] == env_name and key[1] == proj_z and key[2] == band[0] and key[3] == band[1] and key[6] == base_rr.tobytes():
            new = (key[0], key[1], key[2], key[3], key[4], key[5], rr.tobytes())
            cache[new] = {k: (v[..., idx] if isinstance(v, np.ndarray) else v) for k, v in cache[key].items()}


def run_grid(sites: list, classes: list, variants: list, log=print, bellhop: bool = True, range_m: float = RANGE_M) -> dict:
    bs, depthplan, _ = _imports()
    from gpusonar.sim.bellhop import find_bellhop
    res = {"variants": {}, "results": {}, "skipped": {}, "seconds": {}}
    for env in sites:
        name = env["name"]
        if env.get("profile") is not None and (not bellhop or find_bellhop() is None):
            res["skipped"][name] = "a sound-speed profile needs Bellhop (GPUSONAR_BELLHOP)"
            log(f"{name}: skipped (needs Bellhop)")
            continue
        t0 = time.time()
        cache, base_rr = {}, {}
        for v in sorted(variants, key=lambda v: (v[2], v[3], v[0], v[1], v[4])):      # band, then pulse: the factor is reused
            vid = variant_id(*v)
            spec = variant_spec(v[0], v[1], v[2], v[3], v[4])
            d = bs.derive(spec)
            if vid not in res["variants"]:
                res["variants"][vid] = {"rows": v[0], "per_row": v[1], "band": v[2], "pulse_s": v[3], "source_level_db": v[4], **geometry(spec, d),
                                        "threshold_db": bs.detection(d)["threshold_db"], "rows_gain_db": bs.detection(d)["rows_gain_db"]}
            rr = ranges_for(d, base_rr.get(v[2]))
            if rr is not None:
                seed_sliced_factor(cache, name, float(d["projector_m"][2]), tuple(float(x) for x in d["band_hz"]), base_rr[v[2]], rr)
            try:
                pp = parts(d, env, classes, cache, ranges=rr, range_m=range_m)
            except Exception as exc:  # noqa: BLE001  (Bellhop can fail on one water; the others are still planned)
                res["results"].setdefault(vid, {})[name] = {"skipped": f"{type(exc).__name__}: {exc}"}
                log(f"{vid} in {name}: skipped ({type(exc).__name__}: {exc})")
                continue
            if v[3] == 1.0 and v[2] not in base_rr:
                base_rr[v[2]] = pp["rr"]
            res["results"].setdefault(vid, {})[name] = {"own_depths_m": [float(pp["own"][0]), float(pp["own"][-1])], "range_step_m": float(np.median(np.diff(pp["rr"]))),
                                                        "ranges_m": [float(pp["rr"][0]), float(pp["rr"][-1])], "method": pp["plan"]["method"],
                                                        "reverberation": pp["plan"]["reverberation"]["note"],
                                                        "classes": {c["name"]: summarise(pp, c["name"], c.get("speed_m_s")) for c in classes}}
            a = {c: r["A"]["margin_db"] for c, r in res["results"][vid][name]["classes"].items() if r}
            log(f"{name} {vid}: " + ", ".join(f"{c} {m:+.1f}" for c, m in a.items()) + f"  ({time.time() - t0:.0f} s)")
        res["seconds"][name] = round(time.time() - t0, 1)
    return res


# --- sensitivity: the illustrative sites' values -------------------------------------------------------------------

def perturbed(env: dict) -> dict:
    """{label: environment}: the site's Lambert value +-3 dB (where it states one) and its wind one sea state either side
    (the median winds of acoustics.SEA_STATE_WIND_SPEED, via secondary: UnderwaterAcoustics.jl's table)."""
    from gpusonar import acoustics, environment
    out = {}
    bsct = env.get("bottom_scattering")
    if isinstance(bsct, dict) and bsct.get("model", "lambert") == "lambert" and bsct.get("mu_db") is not None:
        for s in (-3.0, 3.0):
            out[f"lambert{s:+g}dB"] = {**copy.deepcopy(env), "name": f"{env['name']}__lambert{s:+g}", "bottom_scattering": dict(bsct, mu_db=float(bsct["mu_db"]) + s)}
    w = float(environment.wind_speed(env, 5.0))
    table = np.asarray(acoustics.SEA_STATE_WIND_SPEED)
    ss = int(env["sea_state"]) if env.get("sea_state") is not None else int(np.argmin(np.abs(table - w)))
    for step in (-1, 1):
        s2 = ss + step
        if 0 <= s2 < len(table):
            e = {k: v for k, v in copy.deepcopy(env).items() if k != "sea_state"}
            e.update(name=f"{env['name']}__ss{s2}", wind_speed_m_s=float(table[s2]))
            out[f"sea_state_{s2}({table[s2]:g} m/s, from {w:g})"] = e
    return out


def run_sensitivity(sites, classes, lengths=LENGTHS, rows=ROWS, log=print, bellhop=True) -> dict:
    """The shipped waveform, every geometry, each site perturbed: does the smallest geometry that holds 12 km change?"""
    bs, _, _ = _imports()
    from gpusonar.sim.bellhop import find_bellhop
    out = {}
    for env in sites:
        if env.get("profile") is not None and (not bellhop or find_bellhop() is None):
            out[env["name"]] = {"skipped": "needs Bellhop"}
            continue
        base_cache = {}
        parts(bs.derive(variant_spec(3, 32)), env, classes, base_cache)          # the factor of the site's water, once
        for label, e in perturbed(env).items():
            cache = {}
            if label.startswith("lambert"):                                        # the bottom's scattering does not enter the factor: reuse it
                for key, val in base_cache.items():
                    if isinstance(key, tuple) and len(key) == 7 and key[0] == env["name"]:
                        cache[(e["name"],) + key[1:]] = val
            row = {}
            for r in rows:
                for n in lengths:
                    d = bs.derive(variant_spec(r, n))
                    try:
                        pp = parts(d, e, classes, cache)
                    except Exception as exc:  # noqa: BLE001
                        row[f"{r}x{n}"] = {"skipped": f"{type(exc).__name__}: {exc}"}
                        continue
                    row[f"{r}x{n}"] = {c["name"]: {b: {k: v for k, v in summarise(pp, c["name"], c.get("speed_m_s"))[b].items() if k in ("margin_db", "reach_m", "own_depth_m", "limited_by", "reverberation_only_ceiling_db")}
                                                   for b in ("A", "B", "noise_only")} for c in classes if pp["classes"].get(c["name"]) is not None}
            out.setdefault(env["name"], {})[label] = {"environment": {k: v for k, v in e.items() if k in ("bottom_scattering", "wind_speed_m_s", "sea_state")}, "by_geometry": row}
            log(f"sensitivity {env['name']} {label}: " + ", ".join(f"{g} " + "/".join(f"{x['A']['margin_db']:+.1f}" for x in v.values()) for g, v in row.items() if "skipped" not in v))
    return out


# --- the coverage picture at the build's site ------------------------------------------------------------------------

def run_coverage(spec_list, env: dict, own_depths: dict, log=print) -> dict:
    """tools/coverage_map.py's computation (the chain's C++ coverage picture, reverberation at the sector's edge) for variant specs."""
    import coverage_map as cm
    import gpusonar as gs
    from gpusonar import coverage
    from gpusonar.sim import make_pose
    from gpusonar.sonar import SonarModel
    bs, _, _ = _imports()
    out = {}
    for label, spec in spec_list:
        d = bs.derive(spec)
        block = cm.chain_block(spec, env, 0.5)
        record = SonarModel.from_config(bs.sonar_config(spec)).make_record(make_pose())
        table = coverage.pd_table(record, bs.chain_config(spec))
        res = {}
        for steer_deg in (0.0, float((block.get("reverberation") or {}).get("steer_deg", 0.0))):
            blk = copy.deepcopy(block)
            if "steer_deg" in (blk.get("reverberation") or {}):
                blk["reverberation"]["steer_deg"] = steer_deg
            sonar = {"f_centre": float(d["centre_hz"]), "bandwidth": float(d["band_hz"][1] - d["band_hz"][0]), "pulse": float(d["pulse_s"]),
                     "beam_integral": float(gs.core.coverage_beam_integral_of(record, np.radians(steer_deg)))}
            for cname, z in own_depths.items():
                water = cm.water_of(env, z, "environment", 0.1, 0.05)
                s = coverage.summary(water, env.get("water_depth_m"), z, blk, sonar, table, d["window_m"], float(d["projector_m"][2]))
                c = next((x for x in s.get("classes", []) if x["name"] == cname), None)
                if c is not None:
                    res.setdefault(f"steer_{steer_deg:g}deg", {})[cname] = {"own_depth_m": z, **{k: c.get(k) for k in ("max_range_m", "max_range_noise_only_m", "limited_by")}}
        out[label] = res
        log(f"coverage {label}: {json.dumps(res)[:300]}")
    return out


# --- answers --------------------------------------------------------------------------------------------------------

def answers(grid: dict, classes: list) -> dict:
    """Per class and site: shipped (3x32, shipped waveform) met or not under each bound; the smallest variant that meets it
    (fewest hydrophones, then fewest rows), with and without credits; and over all sites the smallest geometry."""
    out = {"per_class_site": {}, "smallest_geometry": {}}
    shipped = variant_id(3, 32, "2k", 1.0, SHIPPED["source_level_db"])
    res, var = grid["results"], grid["variants"]

    def order(vid):
        v = var[vid]
        return (v["hydrophones"], v["rows"], abs(v["source_level_db"] - SHIPPED["source_level_db"]), v["pulse_s"], v["band"] != "2k")

    sites = sorted({s for r in res.values() for s in r})
    for c in classes:
        cn = c["name"]
        for site in sites:
            row = {}
            got = (res.get(shipped, {}).get(site) or {}).get("classes", {}).get(cn)
            if got is None:
                continue
            row["shipped_3x32"] = {b: {"margin_db": got[b]["margin_db"], "reach_m": got[b]["reach_m"], "own_depth_m": got[b]["own_depth_m"], "limited_by": got[b].get("limited_by")} for b in ("A", "B")}
            row["shipped_3x32"]["noise_only"] = got["noise_only"]
            for bound in ("A", "B"):
                for credit in (None, "sweep_bank", "baffle", "bank_and_baffle", "tone"):
                    ok = []
                    for vid in sorted(res, key=order):
                        r = (res[vid].get(site) or {}).get("classes", {}).get(cn)
                        if not r:
                            continue
                        m = r[bound]["margin_db"] if credit is None else ((r[bound]["credits"].get(credit) or {}).get("margin_db"))
                        if m is not None and m >= 0.0:
                            ok.append((vid, m))
                    key = f"{bound}" + (f"+{credit}" if credit else "")
                    if ok:
                        vid, m = ok[0]
                        fit = [x for x in ok if var[x[0]]["fits_face"]]
                        row[key] = {"smallest": vid, "margin_db": m, "fits_face": var[vid]["fits_face"], "smallest_that_fits_face": fit[0][0] if fit else None}
                    else:
                        def got(vid):
                            r = ((res[vid].get(site) or {}).get("classes", {}).get(cn) or {}).get(bound, {})
                            return r.get("margin_db") if credit is None else (r.get("credits", {}).get(credit) or {}).get("margin_db")
                        best = max(((vid, got(vid)) for vid in res), key=lambda t: -1e9 if t[1] is None else t[1])
                        lim = ((res[best[0]].get(site) or {}).get("classes", {}).get(cn) or {}).get(bound, {})
                        row[key] = {"smallest": None, "best_in_grid": best[0], "best_margin_db": best[1], "best_limited_by": lim.get("limited_by"),
                                    "best_ceiling_db": lim.get("reverberation_only_ceiling_db")}
            out["per_class_site"].setdefault(cn, {})[site] = row
    # one geometry for every site: the smallest (rows x per_row) at the shipped waveform that holds 12 km at every site the class is held at by ANY variant
    for c in classes:
        cn = c["name"]
        for bound in ("A", "B"):
            geo = {}
            for vid, v in var.items():
                if v["band"] != "2k" or v["pulse_s"] != 1.0 or v["source_level_db"] != SHIPPED["source_level_db"]:
                    continue
                ok_sites = [s for s in sites if ((res[vid].get(s) or {}).get("classes", {}).get(cn) or {}).get(bound, {}).get("margin_db", -1e9) >= 0.0]
                geo[f"{v['rows']}x{v['per_row']}"] = {"sites_met": ok_sites, "hydrophones": v["hydrophones"], "fits_face": v["fits_face"]}
            out["smallest_geometry"].setdefault(cn, {})[bound] = geo
    return out


# --- the simulator: stacked rows against the shelf's reverberation ----------------------------------------------------

SHELF = "shelf_100m_sand"


def shelf_scenario(sonar: str, range_m: float, own_depth: float, scene: list, reverb: bool, noise: bool, pings: int, seed: int, half_window: float = 500.0,
                   overrides: dict | None = None, n_elevation: int | None = None) -> dict:
    """popoto_12km's water (100 m isovelocity over sand, Bellhop eigenrays, Lambert -27 dB bottom and Chapman-Harris surface at the
    environment's wind: the shelf_100m_sand site's scattering) with the array of `sonar` at `own_depth`, a window about `range_m`,
    the platform still, and the scene, reverberation and noise as asked."""
    from gpusonar.config import load_scenario_config
    cfg = load_scenario_config("popoto_12km", expand=False)
    cfg["name"] = f"trade_{sonar}_{range_m:g}"
    cfg["sonar"] = sonar
    cfg["sonar_overrides"] = copy.deepcopy(cfg.get("sonar_overrides") or {})
    cfg["sonar_overrides"]["receive"]["window"] = {"range": [range_m - half_window, range_m + half_window]}
    for k, v in (overrides or {}).items():
        cfg["sonar_overrides"][k] = copy.deepcopy(v)
    cfg["platform"] = {"position": [0.0, 0.0, float(own_depth)], "rpy": [0, 0, 0], "velocity": [0.0, 0.0, 0.0]}
    cfg["pings"] = {"count": int(pings), "interval": round(2.0 * (range_m + half_window) / 1500.0 + 1.5, 1)}
    cfg["propagation"]["max_range"] = float(range_m + half_window + 500.0)
    cfg["scene"] = scene
    if not reverb:
        cfg["reverberation"] = []
    elif n_elevation:
        # The default (monostatic) generator ignores the arrival elevation when it steers to the hydrophones (sim/reverb.py header):
        # every row would hear the same reverberation. The bistatic generator keeps each component's elevation (bins of 180/n degrees).
        cfg["reverberation"] = [dict(item, geometry="bistatic", n_elevation=int(n_elevation)) for item in cfg["reverberation"]]
    if not noise:
        cfg["receiver"] = {**{k: v for k, v in cfg["receiver"].items() if k != "ambient_noise"}, "noise_rms": 0.0}      # as tools/doppler_study.py turns it off
    cfg["seed"] = int(seed)
    cfg.pop("truth", None)
    return cfg


def row_beams(record, bearing_rad: float = 0.0):
    """[rows, n] matched-filter output of one Hann-weighted beam per row at `bearing_rad` (doppler_study's baseband, delays and
    replica), the rows' heights, the sample rate and the time of sample 0."""
    import doppler_study as ds
    from gpusonar import flank
    from gpusonar.recon.fls import aperture_weights
    x, fs, fc, c, t0 = ds.baseband(record)
    tau = flank.steering_delays(record, [bearing_rad])[0]
    w = aperture_weights(record, "hann")
    up = np.round(-record.rx_positions()[:, 2], 3)
    heights = sorted(set(up.tolist()))
    n = x.shape[1]
    f = np.fft.fftfreq(n, 1.0 / fs) + fc
    X = np.fft.fft(x, axis=1) * np.exp(-2j * np.pi * f[None, :] * tau[:, None])
    rep = ds.replica(record.transmit.waveform, fs, fc, 0.0, c)
    out = []
    for h in heights:
        m = up == h
        b = np.fft.ifft((w[m, None] * X[m]).sum(axis=0)) / w[m].sum()
        out.append(ds.correlate(b, rep))
    return np.asarray(out), heights, fs, t0


def rows_gain_curve() -> dict:
    """What the deployed chain's detector gains from adding 1, 2, 3 rows' maps in power (buildspec.detection, the face builds)."""
    bs, _, _ = _imports()
    return {r: float(bs.detection(bs.derive(bs.load(BASE[r])))["rows_gain_db"]) for r in ROWS}


def measure_rows(range_m: float = 2560.0, own_depth: float = 65.0, target_depth: float = 60.0, pings: int = 3, half_window: float = 150.0, n_elevation: int | None = 90,
                 log=print) -> dict:
    """Simulate the three-row face (popoto_long_range_3row) over the shelf: one target-only ping (one point, no reverberation, no
    noise) and `pings` reverberation-only pings (no scene, no noise; each a new draw). Per row, one beam on the target's bearing,
    pulse-compressed. From the target's per-row complex peak `a` and the reverberation's covariance R between rows within
    +-0.05 s of the target's delay:

        coherent vertical beam matched to the target:  G = (|a^H a|^2 / a^H R a) / mean_r(|a_r|^2 / R_rr)   (against white noise: N)
        the chain (rows' maps added in power):          N_eff = (tr R)^2 / ||R||_F^2 independent looks of the reverberation,
                                                        worth G_chain(N_eff) by the detector's own rows gain (interpolated)

    for the two outer rows (the 2x32 build's heights) and all three. Bound A is 0 dB, bound B 10 log10(rows)."""
    t_start = time.time()
    from gpusonar.sim import run_scenario
    point = [{"type": "points", "label": "xl_auv", "positions": [[0.0, float(range_m), float(target_depth)]], "target_strength_db": 5.0, "velocity": [0.0, 0.0, 0.0]}]
    t0 = time.time()
    tgt = run_scenario(shelf_scenario("popoto_long_range_3row", range_m, own_depth, point, reverb=False, noise=False, pings=1, seed=61, half_window=half_window)).records[0]
    zt, heights, fs, s0 = row_beams(tgt)
    t_tgt = time.time() - t0
    k = int(np.argmax(np.sum(np.abs(zt) ** 2, axis=0)))
    a = zt[:, k]
    t0 = time.time()
    recs = run_scenario(shelf_scenario("popoto_long_range_3row", range_m, own_depth, [], reverb=True, noise=False, pings=pings, seed=62, half_window=half_window,
                                       n_elevation=n_elevation)).records
    t_rev = time.time() - t0
    half = int(round(0.05 * fs))
    Y = []
    for r in recs:
        zr, h2, fs2, s2 = row_beams(r)
        assert h2 == heights and abs(fs2 - fs) < 1e-9
        kk = k + int(round((s0 - s2) * fs))
        Y.append(zr[:, max(kk - half, 0):kk + half + 1])
    Y = np.concatenate(Y, axis=1)
    R = (Y @ Y.conj().T) / Y.shape[1]
    G = rows_gain_curve()
    out = {"what": measure_rows.__doc__.strip().split("\n\n")[0].replace("\n", " "), "range_m": range_m, "own_depth_m": own_depth, "target_depth_m": target_depth,
           "reverberation_generator": (f"bistatic, {n_elevation} bins of arrival elevation ({180.0 / n_elevation:g} degrees each)" if n_elevation else "monostatic (the scenario's default)"),
           "heights_m": heights, "pings_reverberation": pings, "samples_used": int(Y.shape[1]), "target_peak_delay_s": float(s0 + k / fs),
           "simulate_seconds": {"target": round(t_tgt, 1), "reverberation": round(t_rev, 1)},
           "row_power_db_re_mean": (10 * np.log10(np.real(np.diag(R)) / np.mean(np.real(np.diag(R))))).tolist(),
           "correlation": np.abs(R / np.sqrt(np.outer(np.real(np.diag(R)), np.real(np.diag(R))))).round(3).tolist(),
           "target_row_phase_deg": np.degrees(np.angle(a / a[0])).round(1).tolist(), "target_row_power_db_re_first": (20 * np.log10(np.abs(a) / np.abs(a[0]))).round(2).tolist(),
           "chain_rows_gain_db_against_noise": {str(r): v for r, v in G.items()}, "subsets": {}}
    for label, idx in (("2 rows (outer: the 2x32 build's heights)", [0, len(heights) - 1]), ("3 rows", list(range(len(heights))))):
        Rs, s = R[np.ix_(idx, idx)], a[idx]
        g_coh = (np.abs(np.vdot(s, s)) ** 2 / np.real(np.vdot(s, Rs @ s))) / np.mean(np.abs(s) ** 2 / np.real(np.diag(Rs)))
        n_eff = float(np.real(np.trace(Rs)) ** 2 / np.sum(np.abs(Rs) ** 2))
        n = len(idx)
        g_chain = float(np.interp(n_eff, list(G), list(G.values())))
        out["subsets"][str(n)] = {"label": label, "rows": n, "bound_A_db": 0.0, "bound_B_db": 10 * np.log10(n),
                                  "coherent_vertical_beam_gain_db": float(10 * np.log10(g_coh)), "coherent_gain_against_white_noise_db": float(10 * np.log10(n)),
                                  "effective_independent_looks": n_eff, "chain_gain_db": g_chain, "chain_gain_against_noise_db": G[n],
                                  "chain_share_of_bound_B": float(g_chain / (10 * np.log10(n))), "coherent_share_of_bound_B": float(10 * np.log10(g_coh) / (10 * np.log10(n)))}
    out["seconds"] = round(time.time() - t_start, 1)
    log(json.dumps(out["subsets"], indent=1))
    return out


def shelf_with_measured_gain(meas: dict, classes: list, log=print) -> dict:
    """The shelf's 12 km figures for every geometry (shipped waveform) with the rows' gain against reverberation as measured for
    today's chain (`chain_gain_db`) and for a coherent vertical beam (`coherent_vertical_beam_gain_db`, not in the chain)."""
    bs, depthplan, _ = _imports()
    env = next(e for e in bs.environments_of(bs.load(BASE[3])) if e["name"] == SHELF)
    cache, out = {}, {}
    for r in ROWS:
        for n in LENGTHS:
            pp = parts(bs.derive(variant_spec(r, n)), env, classes, cache)
            row = {}
            for c in classes:
                cc = pp["classes"].get(c["name"])
                if cc is None:
                    continue
                e = {}
                for label, g in (("A", 0.0), ("chain_measured", 0.0 if r == 1 else meas["subsets"][str(r)]["chain_gain_db"]),
                                 ("coherent_measured", 0.0 if r == 1 else meas["subsets"][str(r)]["coherent_vertical_beam_gain_db"]), ("B", 10 * np.log10(r))):
                    b = _best(margin(cc["noise"], cc["srr"], pp["T"], pp["G"], g), pp["own"], pp["rr"], pp["i_r"])
                    b.pop("_k")
                    e[label] = {**b, "rows_gain_against_reverberation_db": float(g)}
                row[c["name"]] = e
            out[f"{r}x{n}"] = row
            log(f"shelf measured {r}x{n}: " + ", ".join(f"{c} " + "/".join(f"{v[b]['margin_db']:+.1f}" for b in v) for c, v in row.items()))
    return out


# --- the simulator and the chain against the study's prediction ---------------------------------------------------------

def predict_point(d: dict, env: dict, ts_db: float, target_depth: float, own_depth: float, range_m: float) -> dict:
    """The study's figures for ONE point at one depth, one own depth, one range (depthplan.plan with those grids)."""
    bs, depthplan, _ = _imports()
    cls = [{"name": "point", "ts_db": [ts_db, ts_db], "depth_m": [target_depth, target_depth], "status": "assumed"}]
    rr = np.unique(np.r_[np.linspace(max(1000.0, 1.2 * d["blind_range_m"]), range_m, 8), range_m])
    P = depthplan.plan(d, env, cls, range_of_interest=range_m, own_depths=[own_depth], ranges=rr, all_target_depths=[target_depth], cache={})
    c = P["classes"]["point"]
    r = c["recommended"]
    det = bs.detection(d)
    s_tot, s_n = r["margin_db"] + P["threshold_db"], r.get("margin_noise_only_db", r["margin_db"]) + P["threshold_db"]
    gap = 10.0 ** (-s_tot / 10.0) - 10.0 ** (-s_n / 10.0)                         # one row: 1/S = 1/S_noise + 1/SRR
    return {"margin_db": r["margin_db"], "margin_noise_only_db": r.get("margin_noise_only_db", r["margin_db"]), "limited_by": r.get("limited_by"),
            "row_snr_noise_only_db": s_n, "srr_db": (float(-10.0 * np.log10(gap)) if gap > 0 else None),
            "threshold_db": P["threshold_db"], "row_snr_db": r["margin_db"] + P["threshold_db"], "window_loss_db": det["window_loss_db"], "straddle_loss_db": det["straddle_loss_db"],
            "sample_threshold_db": det["sample_threshold_db"],
            "chain_snr_expected_db": r["margin_db"] + P["threshold_db"] - det["window_loss_db"] - det["straddle_loss_db"],
            "factor_db": r["factor_db"], "method": P["method"]}


def clairvoyant_split(c: dict, over: dict, half_window: float, pred: dict, background_pings: int = 2) -> dict:
    """Where a difference comes from: the same geometry simulated as a target-only ping, reverberation-only and noise-only pings,
    one Hann-weighted beam on the target's bearing (tools/doppler_study.py's beam and replica), the target's peak power over the
    mean background power within +-0.1 s of its delay (a clairvoyant background: no detector). Against the prediction's one-row
    SNR over noise and echo-over-reverberation ratio, each less the Hann window's 1.76 dB (its loss against a background spread
    in azimuth; ours for the reverberation)."""
    from gpusonar.sim import run_scenario
    scene = [{"type": "points", "label": "target", "positions": [[0.0, c["range"], c["z"]]], "target_strength_db": c["ts"], "velocity": [0.0, 0.0, 0.0]}]
    tgt = run_scenario(shelf_scenario("popoto_long_range", c["range"], c["own"], scene, reverb=False, noise=False, pings=1, seed=81, half_window=half_window, overrides=over)).records[0]
    zt, _, fs, s0 = row_beams(tgt)
    k = int(np.argmax(np.abs(zt[0]) ** 2))
    p_t = float(np.abs(zt[0, k]) ** 2)
    half = int(round(0.1 * fs))

    def background(reverb, noise, seed):
        recs = run_scenario(shelf_scenario("popoto_long_range", c["range"], c["own"], [], reverb=reverb, noise=noise, pings=background_pings, seed=seed,
                                           half_window=half_window, overrides=over)).records
        vals = []
        for r in recs:
            z, _, _, s2 = row_beams(r)
            kk = k + int(round((s0 - s2) * fs))
            vals.append(np.abs(z[0, max(kk - half, 0):kk + half + 1]) ** 2)
        return float(np.mean(np.concatenate(vals)))
    p_r, p_n = background(True, False, 82), background(False, True, 83)
    win = float(pred["window_loss_db"])
    out = {"target_peak_delay_s": float(s0 + k / fs),
           "snr_over_noise_db": 10 * np.log10(p_t / p_n), "snr_over_noise_predicted_db": pred["row_snr_noise_only_db"] - win,
           "echo_over_reverberation_db": 10 * np.log10(p_t / p_r), "echo_over_reverberation_predicted_db": None if pred["srr_db"] is None else pred["srr_db"] - win,
           "snr_over_both_db": 10 * np.log10(p_t / (p_r + p_n)), "snr_over_both_predicted_db": pred["row_snr_db"] - win}
    out["differences_db"] = {k2: (None if out[f"{k2}_predicted_db"] is None else out[f"{k2}_db"] - out[f"{k2}_predicted_db"]) for k2 in ("snr_over_noise", "echo_over_reverberation", "snr_over_both")}
    return out


def crosscheck(R: dict, pings: int = 6, half_window: float = 400.0, log=print) -> dict:
    """Two grid points through the simulator (Bellhop eigenrays in the shelf's water, its reverberation and noise) and the live
    chain (C++), against the study's prediction: a +25 dB point at 4 m at 12 km, array at 10 m (the popoto_12km geometry), and the
    xl_auv class's +5 dB at 60 m at the range of its predicted reach, array at the planner's depth; both with one row of 64."""
    bs, depthplan, _ = _imports()
    import popoto_12km_study as st
    env = next(e for e in bs.environments_of(bs.load(BASE[3])) if e["name"] == SHELF)
    spec = variant_spec(1, 64)
    d = bs.derive(spec)
    over = {"array": {"count": 64}}
    out = {"variant": "1x64|2k|1s|197dB", "sonar": "popoto_long_range with array.count 64 (an override in memory)", "chain": "configs/deploy/popoto_vehicle.yaml (C++ engine, host)", "cases": {}}
    vid = variant_id(1, 64, "2k", 1.0, 197.0)
    xl = ((R.get("grid") or {}).get("results", {}).get(vid, {}).get(SHELF) or {}).get("classes", {}).get("xl_auv")
    own_xl = xl["A"]["own_depth_m"] if xl else 65.0
    reach_xl = xl["A"]["reach_m"] if xl and xl["A"]["reach_m"] > 0 else 2500.0
    cases = {"ship_12km": {"ts": 25.0, "z": 4.0, "own": 10.0, "range": 12000.0, "vel": [-6.0, 0.0, 0.0]},
             "xl_auv_near_reach": {"ts": 5.0, "z": 60.0, "own": own_xl, "range": reach_xl, "vel": [-2.0, 0.0, 0.0]}}
    for name, c in cases.items():
        t0 = time.time()
        pred = predict_point(d, env, c["ts"], c["z"], c["own"], c["range"])
        scene = [{"type": "points", "label": name, "positions": [[0.0, c["range"], c["z"]]], "target_strength_db": c["ts"], "velocity": c["vel"]}]
        cfg = shelf_scenario("popoto_long_range", c["range"], c["own"], scene, reverb=True, noise=True, pings=pings, seed=71, half_window=half_window, overrides=over)
        m = st.measure(cfg, 1, engine="cpp")
        split = clairvoyant_split(c, over, half_window, pred)
        meas = {k: m.get(k) for k in ("pings_detected", "median_snr_db", "track_confirmed_at_ping", "pings", "simulate_seconds", "chain_seconds_per_ping")}
        snrs = [p["snr_db"] for p in m["per_ping"] if p["detected"]]
        meas["snr_db_each_ping"] = snrs
        diff = (m["median_snr_db"] - pred["chain_snr_expected_db"]) if m.get("median_snr_db") is not None else None
        out["cases"][name] = {"geometry": {k: c[k] for k in ("ts", "z", "own", "range")}, "prediction": pred, "measured": meas, "split": split,
                              "measured_minus_predicted_db": diff, "seconds": round(time.time() - t0, 1),
                              "what_is_compared": "the chain's median snr_db on the detected pings (peak over the local background) against the study's one-row SNR with "
                                                  "the reverberation term, less the detector's window and mean straddle losses (buildspec.detection: what the threshold holds)"}
        log(f"crosscheck {name}: predicted {pred['chain_snr_expected_db']:.1f} dB, measured {m.get('median_snr_db')} on {m['pings_detected']} of {m['pings']} pings")
    return out


# --- transmit options by sea state (--transmit; the owner, 2026-11-27: "lets have options for the transmit in different sea
# states, like a table. you can keep the default we have now but lets make sure the trade study covers options") ----------

TX_CLASSES = ("large_submerged", "ship")             # the requirement's, in its order (the owner, 2026-11-26; D-235)
TX_GEOMETRY = (3, 32)                                 # the shipped face (configs/builds/popoto_face_3x32.yaml)
TX_LEVELS = (197.0, 200.0, 203.0)                     # the shipped level and two 3 dB steps (projector_requirements.py's levels run to 203 dB)
TX_PULSES = (1.0, 2.0, 4.0, 8.0)                      # every pulse at FULL level
TX_PAIRED = (2.0, 194.0)                              # projector_selection.md Sec. 1.8 and Sec. 4 option B: the energy-limited projector's pairing
TX_SEA_STATES = (2, 3, 4, 5, 6)                       # whole states: acoustics.sea_state_wind_speed (via secondary: UnderwaterAcoustics.jl's table)
TX_DEEP = "deep_no_bottom_return"                     # the site that isolates the noise (no bottom: surface image and surface reverberation only)
TX_BUBBLE_SITE = "deep_no_bottom_return_sea_state_5"  # the site that states a bubble law (apl_1994); the deep table is run with and without it
TX_RADIAL = (0.4, 2.0, 8.0)                           # m/s radial: a crossing target at 12 km (DOPPLER.md LS.2: -0.39 m/s); the class's 2 and 8 m/s (assumed)
TX_PINGS = (2, 4, 8)
TX_EFFICIENCY = (0.5, 0.85)                           # transducer, amplifier: projector_selection.md Sec. 1.3's working values (ours; no part chosen)
TX_TEXTBOOK_PINGS = ("textbook rule for N looks added incoherently in the many-look limit: the threshold falls about 5 log10 N "
                     "(Urick, Principles of Underwater Sound, the detection-threshold chapter; status: unconfirmed, not opened in this session)")


def tx_options() -> list:
    """The transmit options, least energy per ping first (energy = source level + 10 log10 pulse, dB re 1 uPa^2 s at 1 m),
    then the shorter pulse. The 4-6 kHz band as shipped throughout."""
    opts = [(p, lv) for lv in TX_LEVELS for p in TX_PULSES] + [TX_PAIRED]
    out = []
    for p, lv in sorted(set(opts), key=lambda o: (round(o[1] + 10.0 * np.log10(o[0]), 6), o[0])):
        out.append({"id": f"{p:g} s {lv:g} dB", "variant": variant_id(TX_GEOMETRY[0], TX_GEOMETRY[1], "2k", p, lv), "pulse_s": float(p), "source_level_db": float(lv),
                    "energy_db": float(lv + 10.0 * np.log10(p)), "shipped": bool(p == SHIPPED["pulse_s"] and lv == SHIPPED["source_level_db"]),
                    "former_shipped": bool(p == FORMER["pulse_s"] and lv == FORMER["source_level_db"]),
                    "paired": bool((p, lv) == TX_PAIRED)})
    return out


def tx_sites(sites: list) -> dict:
    """{key: environment}: the deep site at each whole sea state, without a bubble law (as it states none) and with the law
    the sea-state-5 site states; and every illustrative site at its own stated conditions."""
    by = {e["name"]: e for e in sites}
    out = {}
    deep = by.get(TX_DEEP)
    law = (by.get(TX_BUBBLE_SITE) or {}).get("bubble_model")
    if deep is not None:
        for law_ in [None] + ([law] if law else []):
            for n in TX_SEA_STATES:
                e = {k: v for k, v in copy.deepcopy(deep).items() if k not in ("wind_speed_m_s", "sea_state", "bubble_model", "title")}
                e.update(sea_state=int(n), name=f"{TX_DEEP}__ss{n}" + (f"__{law_}" if law_ else ""),
                         title=f"{TX_DEEP} at sea state {n}" + (f", bubble loss by the {law_} law" if law_ else ", no bubble law (as the site states)"))
                if law_:
                    e["bubble_model"] = law_
                out[e["name"]] = e
    for e in sites:
        out[f"{e['name']}__as_stated"] = {**copy.deepcopy(e), "name": e["name"]}
    return out


def sweep_mismatch_loss_db(radial_m_s: float, band_hz: float, pulse_s: float, c: float) -> float:
    """The PEAK a linear sweep's matched filter loses on an echo of radial speed r: -20 log10 M(phi), phi = pi B T |r| / c (the
    closed form of tests/python/test_doppler_bank.py, ours; the C++ library's sweep_mismatch, which the chain's detector uses)."""
    import gpusonar as gs
    return float(-20.0 * np.log10(gs.core.live_sweep_mismatch(float(np.pi * band_hz * pulse_s * abs(radial_m_s) / c))))


def tx_projector(d: dict) -> dict:
    """Which listed projector supports the option's level across 4-6 kHz at its duty (tools/projector_candidates.py: the sheets
    as read in docs/design/projector_selection.md, via secondary), the least depth free of cavitation by the maker's own
    statement scaled with static pressure, and, where no single unit does, the fewest units in phase (20 log10 N on the
    weakest frequency, mutual coupling ignored: projector_selection.md Sec. 3.2's rule; unconfirmed)."""
    import projector_candidates as pc
    sl, duty = float(d["source_level_db"]), float(d["pulse_s"] / d["ping_interval_s"])
    single, arrays, none = [], [], []
    for c in pc.CANDIDATES:
        if pc.rated_voltage(c, duty) is None:
            none.append(f"{c['model']}: no rating on its sheet covers {100 * duty:.0f} % duty")
            continue
        r = pc.evaluate(c, d, 500.0, (5.0,))
        lv = float(min(r["level_at_rated_db"]))
        entry = {"model": c["model"], "level_at_rated_drive_db": lv, "units": int(r["units_for_level"])}
        if r["meets_level_across_band"]:
            entry["least_depth_maker_m"] = r.get("least_depth_for_level_maker_m")
            single.append(entry)
        else:
            per = sl - 20.0 * np.log10(entry["units"])
            r2 = pc.evaluate(c, {**d, "source_level_db": per}, 500.0, (5.0,))
            entry.update(level_each_db=float(per), least_depth_maker_m=r2.get("least_depth_for_level_maker_m"))
            arrays.append(entry)
    arrays.sort(key=lambda e: (e["units"], -e["level_at_rated_drive_db"]))
    if single:
        words = "; ".join(f"{e['model']} ({e['level_at_rated_drive_db']:.1f} dB at its rated drive"
                          + (f", free of cavitation below {e['least_depth_maker_m']:.0f} m by its maker" if e.get("least_depth_maker_m") is not None else ", no cavitation statement on its sheet") + ")"
                          for e in sorted(single, key=lambda e: -e["level_at_rated_drive_db"]))
        status = "via secondary (the sheets as read in projector_selection.md; tools/projector_candidates.py)"
    else:
        a = arrays[0] if arrays else None
        near = max(arrays, key=lambda e: e["level_at_rated_drive_db"]) if arrays else None
        words = ("no source identified: no listed unit reaches it across 4-6 kHz at this duty"
                 + (f" (the nearest, {near['model']}, gives {near['level_at_rated_drive_db']:.1f} dB at the rating that covers {100 * duty:.0f} % duty: {sl - near['level_at_rated_drive_db']:.1f} dB short)" if near else "") + (
            f"; it would need {a['units']} x {a['model']} in phase ({a['level_each_db']:.1f} dB each against {a['level_at_rated_drive_db']:.1f} dB at its rated drive; "
            + (f"each free of cavitation below {a['least_depth_maker_m']:.0f} m by its maker" if a.get("least_depth_maker_m") is not None else "no cavitation statement on its sheet")
            + "; 20 log10 N with mutual coupling ignored, projector_selection.md Sec. 3.2's rule)" if a else ""))
        status = "unconfirmed (no single listed source; the array figure ignores mutual coupling, which an NPS model found 2 dB off in places: projector_selection.md Sec. 3.2)"
    return {"duty": duty, "single_units": single, "arrays": arrays[:3], "no_rating": none, "words": words, "status": status}


def tx_costs(d: dict) -> dict:
    """What an option costs that the model computes: blind range, range resolution, ping interval and duty against the two-way
    time to 12 km, energy and power (projector_requirements.py's relations at projector_selection.md's working efficiencies),
    the sweep's Doppler tolerance and peak loss without the bank, the bank's size, and the projector."""
    import projector_requirements as pr
    c, (f_lo, f_hi), T, sl = float(d["sound_speed"]), d["band_hz"], float(d["pulse_s"]), float(d["source_level_db"])
    B = float(f_hi - f_lo)
    e = pr.energy(sl, T, float(d["ping_interval_s"]), TX_EFFICIENCY[0], TX_EFFICIENCY[1])
    step = c / (2.0 * B * T)
    return {"blind_range_m": float(d["blind_range_m"]), "blind_range_formula": "c T / 2", "range_resolution_m": c / (2.0 * B), "ping_interval_s": float(d["ping_interval_s"]),
            "two_way_12km_s": 2.0 * RANGE_M / c, "duty": T / float(d["ping_interval_s"]), "energy_db": sl + 10.0 * np.log10(T),
            "acoustic_w": e["acoustic_w"], "from_supply_w": e["from_supply_w"], "joule_per_ping": e["joule_per_ping"], "average_w": e["average_w"],
            "heat_in_transducer_average_w": e["heat_in_transducer_average_w"],
            "power_assumptions": f"omnidirectional; transducer {100 * TX_EFFICIENCY[0]:g} % and amplifier {100 * TX_EFFICIENCY[1]:g} % efficient (projector_selection.md Sec. 1.3's working values; ours: no part chosen)",
            "doppler_tolerance_m_s": step, "doppler_tolerance_formula": "c / (2 B T): the radial speed at which the plain matched filter's peak loses 0.97 dB (phi = pi / 2)",
            "mismatch_loss_db": {f"{r:g}": sweep_mismatch_loss_db(r, B, T, c) for r in TX_RADIAL},
            "mismatch_smear_m": {f"{r:g}": 2.0 * abs(r) * T for r in TX_RADIAL},
            "bank_hypotheses_8_m_s": int(2 * int(np.ceil(8.0 / step - 1e-9)) + 1), "projector": tx_projector(d)}


def tx_multi_ping(d: dict) -> dict:
    """N pings combined: the chain has no ping-to-ping integration (tracks confirm on 3 hits: live.DEFAULTS tracker), and the
    planner none. For scale only: what this detector would need if N pings' maps were added in power cell by cell as its rows'
    maps are (buildspec.detection with rows x N maps; ours), which a mover at 2-8 m/s defeats unless the cells are moved with it."""
    from gpusonar import buildspec as bs
    det = bs.detection(d)
    rows = int(det["rows"])
    return {"implemented": False, "what": "not implemented in the chain or the planner: the table's columns hold one ping",
            "detector_gain_if_added_in_power_db": {str(n): float(det["threshold_db"] - bs.detection(d, rows=rows * n)["threshold_db"]) for n in TX_PINGS},
            "detector_gain_status": "ours: buildspec.detection with rows x N maps added in power (the rows' own machinery); not implemented for pings",
            "motion_between_pings_m": {f"{r:g}": r * float(d["ping_interval_s"]) for r in TX_RADIAL[1:]}, "range_resolution_m": d["sound_speed"] / (2.0 * (d["band_hz"][1] - d["band_hz"][0])),
            "textbook": TX_TEXTBOOK_PINGS}


def run_transmit(envs: dict, classes: list, options: list, log=print, bellhop: bool = True) -> dict:
    """The shipped 3 x 32 face with each transmit option at each environment: the planner's margin at 12 km (bound A) and its
    parts (planned with every class of the build, as the grid is, since the planner's target-depth grid follows the classes; TX_CLASSES kept), worst over each class's depths at the low end of its strength, at the recommended own depth; and the same with the
    plain matched filter's peak loss on a mover (TX_RADIAL), at that own depth."""
    bs, depthplan, _ = _imports()
    from gpusonar.sim.bellhop import find_bellhop
    res, skipped, seconds, meta = {}, {}, {}, {}
    for key, env in envs.items():
        if env.get("profile") is not None and (not bellhop or find_bellhop() is None):
            skipped[key] = "a sound-speed profile needs Bellhop (GPUSONAR_BELLHOP)"
            log(f"{key}: skipped (needs Bellhop)")
            continue
        t0 = time.time()
        cache, base_rr = {}, None
        for o in sorted(options, key=lambda o: (o["pulse_s"], o["source_level_db"])):           # 1 s first: the longer pulses slice its factor
            d = bs.derive(variant_spec(TX_GEOMETRY[0], TX_GEOMETRY[1], "2k", o["pulse_s"], o["source_level_db"]))
            rr = ranges_for(d, base_rr)
            if rr is not None:
                seed_sliced_factor(cache, env["name"], float(d["projector_m"][2]), tuple(float(x) for x in d["band_hz"]), base_rr, rr)
            pp = parts(d, env, classes, cache, ranges=rr)
            if o["pulse_s"] == 1.0 and base_rr is None:
                base_rr = pp["rr"]
            B, c = float(d["band_hz"][1] - d["band_hz"][0]), float(d["sound_speed"])
            row = {}
            for cl in classes:
                if cl["name"] not in TX_CLASSES:
                    continue
                x = pp["classes"].get(cl["name"])
                if x is None:
                    continue
                m = margin(x["noise"], x["srr"], pp["T"], pp["G"], 0.0)
                b = _best(m, pp["own"], pp["rr"], pp["i_r"])
                k = b.pop("_k")
                s = summarise(pp, cl["name"], cl.get("speed_m_s"))
                movers = {f"{r:g}": float(margin(x["noise"], x["srr"], pp["T"], pp["G"], 0.0, echo_db=-sweep_mismatch_loss_db(r, B, float(d["pulse_s"]), c))[k, pp["i_r"]]) for r in TX_RADIAL}
                row[cl["name"]] = {"margin_db": b["margin_db"], "own_depth_m": b["own_depth_m"], "reach_m": b["reach_m"], "noise_only_db": float(x["noise"][k, pp["i_r"]]),
                                   "limited_by": s["A"]["limited_by"], "ceiling_db": s["A"]["reverberation_only_ceiling_db"], "threshold_db": float(pp["T"]),
                                   "plain_filter_on_a_mover_db": movers}
            res.setdefault(key, {})[o["id"]] = row
            if key not in meta:
                meta[key] = {"title": env.get("title", ""), "method": pp["plan"]["method"], "reverberation": pp["plan"]["reverberation"]["note"],
                             "wind_speed_m_s": float(depthplan.wind_of(env)), "bubble_model": env.get("bubble_model"), "site": env["name"]}
            log(f"transmit {key} {o['id']}: " + ", ".join(f"{c_} {v['margin_db']:+.2f}" for c_, v in row.items()) + f"  ({time.time() - t0:.0f} s)")
        seconds[key] = round(time.time() - t0, 1)
    return {"results": res, "skipped": skipped, "seconds": seconds, "environments": meta}


def tx_answers(T: dict) -> dict:
    """From the transmit grid: per table and sea state, the least energy that holds both classes and every option at that energy;
    per option, the highest sea state up to which both hold (every state below as well), with and without the bubble law; and
    per site at its own conditions, what energy buys."""
    opts, res = T["options"], T["grid"]["results"]

    def holds(key, oid):
        r = (res.get(key) or {}).get(oid) or {}
        return bool(r) and all((r.get(c) or {}).get("margin_db", -1e9) >= 0.0 for c in TX_CLASSES)
    out = {"least_energy": {}, "highest_sea_state": {}, "energy_buys": {}}
    for table, suffix in (("no_bubble_law", ""), ("bubble_law", f"__{T.get('bubble_law')}")):
        for n in TX_SEA_STATES:
            key = f"{TX_DEEP}__ss{n}{suffix}"
            if key not in res:
                continue
            ok = [o for o in opts if holds(key, o["id"])]
            if ok:
                e0 = ok[0]["energy_db"]
                out["least_energy"].setdefault(table, {})[str(n)] = {"energy_db": e0, "options": [o["id"] for o in ok if abs(o["energy_db"] - e0) < 0.1]}       # 3 dB a doubling: 3.01
            else:
                best = max(opts, key=lambda o: min(((res[key].get(o["id"]) or {}).get(c) or {}).get("margin_db", -1e9) for c in TX_CLASSES))
                out["least_energy"].setdefault(table, {})[str(n)] = {"energy_db": None, "options": [], "best": best["id"],
                                                                     "best_margins_db": {c: res[key][best["id"]][c]["margin_db"] for c in TX_CLASSES}}
        for o in opts:
            top = None
            for n in TX_SEA_STATES:
                if f"{TX_DEEP}__ss{n}{suffix}" not in res:
                    break
                if not holds(f"{TX_DEEP}__ss{n}{suffix}", o["id"]):
                    break
                top = n
            out["highest_sea_state"].setdefault(table, {})[o["id"]] = top
    lo, hi = opts[0], opts[-1]
    for key, by in res.items():
        if not key.endswith("__as_stated"):
            continue
        site = key[: -len("__as_stated")]
        e = {}
        for c in TX_CLASSES:
            a, b = (by.get(lo["id"]) or {}).get(c), (by.get(hi["id"]) or {}).get(c)
            if a and b:
                e[c] = {"from_db": a["margin_db"], "to_db": b["margin_db"], "gained_db": b["margin_db"] - a["margin_db"], "energy_added_db": hi["energy_db"] - lo["energy_db"],
                        "limited_by_at_most": b["limited_by"], "ceiling_db": b["ceiling_db"], "holds_at_most": b["margin_db"] >= 0.0,
                        "least_energy_that_holds": next((o["id"] for o in opts if ((by.get(o["id"]) or {}).get(c) or {}).get("margin_db", -1e9) >= 0.0), None)}
        both = next((o["id"] for o in opts if holds(key, o["id"])), None)
        out["energy_buys"][site] = {"classes": e, "least_energy_both": both}
    return out


def tx_requirement(T: dict) -> dict:
    """The requirement sentence the model supports for each option, derived from the table (not typed)."""
    A, opts = T["answers"], T["options"]
    sites_ok = {}
    for site, v in A["energy_buys"].items():
        sites_ok[site] = {o["id"]: all((((T["grid"]["results"].get(f"{site}__as_stated") or {}).get(o["id"]) or {}).get(c) or {}).get("margin_db", -1e9) >= 0.0 for c in TX_CLASSES)
                          for o in opts}
    out = {}
    for o in opts:
        n_plain = A["highest_sea_state"].get("no_bubble_law", {}).get(o["id"])
        n_bub = A["highest_sea_state"].get("bubble_law", {}).get(o["id"])
        held = [s for s in sites_ok if sites_ok[s][o["id"]]]
        not_held = [s for s in sites_ok if not sites_ok[s][o["id"]]]

        def ss(n):
            return f"at none of sea states {TX_SEA_STATES[0]} to {TX_SEA_STATES[-1]}" if n is None else f"up to sea state {n}"
        out[o["id"]] = (f"With {o['id']} (4-6 kHz sweep; {o['energy_db']:.1f} dB re 1 uPa^2 s at 1 m a ping) the shipped 3 x 32 face holds 12 km against "
                        f"`large_submerged` and `ship` (each at the low end of its assumed target strength, detection probability 0.5 on one ping at the chain's false-alarm budget, "
                        f"an echo with no radial speed: on a mover a linear sweep needs the sweep bank on, a hyperbolic sweep (the shipped one since D-256) loses 0.2 dB or less "
                        f"at 8 m/s and reads it 3 T m nearer per m/s closing) in non-ducting deep water (no bottom return, isovelocity) {ss(n_bub)} with the "
                        f"{T.get('bubble_law')} bubble law ({ss(n_plain)} without it); at the illustrative sites' own conditions at {', '.join(held) if held else 'none'}"
                        + (f", not at {', '.join(not_held)}" if not_held else "") + ".")
    return out


def run_transmit_study(sites: list, classes: list, log=print, bellhop: bool = True, envs: dict | None = None, options: list | None = None) -> dict:
    bs, _, _ = _imports()
    t0 = time.time()
    options = tx_options() if options is None else options
    cls = [c for c in classes if c["name"] in TX_CLASSES]
    cls.sort(key=lambda c: TX_CLASSES.index(c["name"]))
    envs = tx_sites(sites) if envs is None else envs
    T = {"command": command_line(), "date": datetime.date.today().isoformat(), "git_head_when_run": git_head(), "classes": [c["name"] for c in cls],
         "geometry": f"{TX_GEOMETRY[0]}x{TX_GEOMETRY[1]}", "band_hz": list(SHIPPED["band_hz"]), "bubble_law": next((e.get("bubble_model") for e in sites if e["name"] == TX_BUBBLE_SITE), None),
         "options": [], "radial_m_s": list(TX_RADIAL)}
    for o in options:
        d = bs.derive(variant_spec(TX_GEOMETRY[0], TX_GEOMETRY[1], "2k", o["pulse_s"], o["source_level_db"]))
        T["options"].append({**o, "costs": tx_costs(d)})
    T["multi_ping"] = tx_multi_ping(bs.derive(variant_spec(TX_GEOMETRY[0], TX_GEOMETRY[1])))
    T["grid"] = run_transmit(envs, classes, T["options"], log=log, bellhop=bellhop)     # every class planned (the target-depth grid is theirs); two kept
    T["answers"] = tx_answers(T)
    T["requirement"] = tx_requirement(T)
    T["seconds"] = round(time.time() - t0, 1)
    T["waveforms"] = tx_waveforms(T)                     # derived from the grid (recomputed by --report)
    return T


# --- the waveform on a mover with no bank (derived at --report from the transmit grid; docs/tasks/DOPPLER.md "HFM") -------

TX_WAVE_RADIAL = (2.0, 8.0)                           # m/s closing: the class's assumed 2 to 8 m/s (reference_classes.yaml)
TX_WAVE_MEASURED = ("docs/tasks/DOPPLER.md, \"HFM\" (docs/tasks/hfm_doppler_eval.json): the live chain, both engines, free field at 12 km, "
                    "point and bow-on hull, 1 to 8 s")


def tx_waveforms(T: dict) -> dict:
    """The plain matched filter (no sweep bank: the owner's baseline) on a target closing at 2 and 8 m/s, for the shipped linear
    sweep (LFM) and the hyperbolic sweep (HFM, `transmit.waveform: {type: sampled, source: hfm}`) of the same band, length and
    level, at every option and deep-water sea state of the transmit grid. Derived, not run: the planner has no waveform shape
    (an HFM of the same band, energy and taper gives the grid's zero-speed margin; its spectrum's tilt toward the low end,
    1 / f^2, is not in the planner), and a margin less the echo's loss is the margin with that loss (both of its terms take it:
    `margin(echo_db=)`, held in the grid's own LFM columns to 1e-3 dB). The losses are the closed forms for the PEAK,
    untapered: the LFM's -20 log10 M(pi B T v / c) (live_sweep_mismatch; ours, tests/python/test_doppler_bank.py) and the HFM's
    1 + (1 / eta - 1) f1 / B closing (gpusonar.doppler.hfm_doppler; ours, tests/python/test_hfm_doppler.py); the chain's detector
    integrates over cells and loses less with the LFM (measured: TX_WAVE_MEASURED)."""
    from gpusonar import doppler as dop
    bs, _, _ = _imports()
    opts, res = T["options"], T["grid"]["results"]
    f_lo, f_hi = (float(f) for f in T["band_hz"])
    per = {}
    for o in opts:
        c = float(bs.derive(variant_spec(TX_GEOMETRY[0], TX_GEOMETRY[1], "2k", o["pulse_s"], o["source_level_db"]))["sound_speed"])
        Tp = float(o["pulse_s"])
        lfm = {f"{v:g}": sweep_mismatch_loss_db(v, f_hi - f_lo, Tp, c) for v in TX_WAVE_RADIAL}
        hfm = {f"{v:g}": dop.hfm_doppler(-v, f_lo, f_hi, Tp, c)["loss_db"] for v in TX_WAVE_RADIAL}
        per[o["id"]] = {"sound_speed": c,
                        "lfm": {"loss_db": lfm, "range_m_per_m_s": 0.5 * (f_lo + f_hi) * Tp / (f_hi - f_lo),
                                "range_error_m": {f"{v:g}": -0.5 * (f_lo + f_hi) * Tp / (f_hi - f_lo) * v for v in TX_WAVE_RADIAL}},
                        "hfm": {"loss_db": hfm, "range_m_per_m_s": dop.hfm_range_factor(f_lo, f_hi, Tp),
                                "range_error_m": {f"{v:g}": dop.hfm_doppler(-v, f_lo, f_hi, Tp, c)["range_error_m"] for v in TX_WAVE_RADIAL}}}
    deep, least = {}, {}
    for table, suffix in (("bubble_law", f"__{T.get('bubble_law')}"), ("no_bubble_law", "")):
        for n in TX_SEA_STATES:
            key = f"{TX_DEEP}__ss{n}{suffix}"
            if key not in res:
                continue
            row = {}
            for o in opts:
                r = res[key].get(o["id"]) or {}
                row[o["id"]] = {w: {f"{v:g}": {cl: float(r[cl]["margin_db"] - per[o["id"]][w]["loss_db"][f"{v:g}"]) for cl in TX_CLASSES if cl in r}
                                    for v in TX_WAVE_RADIAL} for w in ("lfm", "hfm")}
            deep.setdefault(table, {})[str(n)] = row
            for w in ("lfm", "hfm"):
                for v in TX_WAVE_RADIAL:
                    ok = [o for o in opts if all(row[o["id"]][w][f"{v:g}"].get(cl, -1e9) >= 0.0 for cl in TX_CLASSES)]
                    e0 = ok[0]["energy_db"] if ok else None
                    least.setdefault(table, {}).setdefault(str(n), {}).setdefault(w, {})[f"{v:g}"] = {
                        "energy_db": e0, "options": [o["id"] for o in ok if abs(o["energy_db"] - e0) < 0.1] if ok else []}
    return {"radial_m_s": list(TX_WAVE_RADIAL), "convention": "closing along the line of sight; the HFM's loss is the larger closing (an up-sweep's echo overruns the top of the band)",
            "per_option": per, "deep": deep, "least_energy": least, "measured": TX_WAVE_MEASURED}


def waveform_section(R: dict) -> list:
    """## 0a: the plain filter on a mover, LFM against HFM, per sea state and option (from `tx_waveforms`)."""
    T = R.get("transmit") or {}
    W = T.get("waveforms")
    if not W:
        return []
    tx_flags(T)
    opts, law = T["options"], T.get("bubble_law")
    per, least = W["per_option"], W["least_energy"]
    sh = next(o for o in opts if o["shipped"])["id"]
    sp = [f"{v:g}" for v in W["radial_m_s"]]

    def le(table, n, w, v):
        x = least.get(table, {}).get(str(n), {}).get(w, {}).get(v) or {}
        return f"{x['energy_db']:.1f} dB ({', '.join(x['options'])})" if x.get("energy_db") is not None else "none in the grid"

    L = ["## 0a. The plain filter on a mover: linear against hyperbolic sweep (no sweep bank)", "",
         "The owner's baseline has every Doppler option off, so a moving target meets the plain matched filter. This section is **derived at `--report`** from the "
         "transmit grid above (`tx_waveforms`; no new planner run): each cell is the grid's margin less the closed-form PEAK loss of the waveform on a target closing "
         f"at {' / '.join(sp)} m/s (the class's assumed speeds), for the linear sweep (LFM; shipped until D-256) and the hyperbolic sweep (HFM: `transmit.waveform: {{type: sampled, "
         "source: hfm}`, the same band, length, taper and level, so the same energy). Status: **ours** (closed forms held to direct correlation in "
         f"tests/python/test_doppler_bank.py and tests/python/test_hfm_doppler.py). **Since D-256 the shipped sweep is the HFM ({sh}).**", ""]
    worst = []
    for table, label in (("bubble_law", f"with the {law} law"), ("no_bubble_law", "without a bubble law")):
        never = [str(n) for n in TX_SEA_STATES if str(n) in least.get(table, {}) and least[table][str(n)]["lfm"][sp[-1]]["energy_db"] is None]
        worst.append(f"**With the linear sweep and no bank, a target closing at {sp[-1]} m/s** ({label}), the least energy holding both classes: "
                     + "; ".join(f"sea state {n}: {le(table, n, 'lfm', sp[-1])}" for n in TX_SEA_STATES if str(n) in least.get(table, {}))
                     + ("" if not never else f" (**none in the grid at sea state {', '.join(never)}**)") + "; with the hyperbolic sweep: "
                     + "; ".join(f"sea state {n}: {le(table, n, 'hfm', sp[-1])}" for n in TX_SEA_STATES if str(n) in least.get(table, {})) + ".")
    h, l_ = per[sh]["hfm"], per[sh]["lfm"]
    worst.append(f"**What the HFM costs: a range bias it cannot remove without a rate.** The plain filter reads an HFM echo closing at v nearer by f1 T / B per m/s "
                 f"({h['range_m_per_m_s']:.1f} m per m/s at the shipped {sh.split(' s ')[0]} s, i.e. 3 T at 4-6 kHz: "
                 + ", ".join(f"{o['id'].split(' s')[0]} s: {per[o['id']]['hfm']['range_error_m'][sp[-1]]:.0f} m" for o in opts if o["source_level_db"] == 197.0)
                 + f" at {sp[-1]} m/s); the linear sweep's coupling is f_c T / B ({l_['range_m_per_m_s']:.1f} m per m/s at {sh.split(' s ')[0]} s), and its peak is smeared besides. "
                 "With no bank neither gives a range rate to correct it. Also: the HFM's energy density falls as 1 / f^2 across the band (2.6 dB from 4.25 to 5.75 kHz; "
                 "the projector's response across 4-6 kHz is not known), its compressed main lobe is 2 % wider and its first sidelobe 0.6 dB higher than the linear "
                 "sweep's (Tukey 0.1, unwindowed filter; test_hfm_doppler.py), and the sweep bank refuses it (it takes a linear sweep only).")
    worst.append("The HFM's closed-form peak loss is " + ", ".join(f"{per[sh]['hfm']['loss_db'][v]:.2f} dB at {v} m/s" for v in sp)
                 + " **at every pulse length** (it does not grow with T); the linear sweep's: "
                 + "; ".join(f"{o['id'].split(' s')[0]} s: " + " / ".join(f"{per[o['id']]['lfm']['loss_db'][v]:.1f}" for v in sp) + " dB" for o in opts if o["source_level_db"] == 197.0)
                 + f" at {' / '.join(sp)} m/s. Measured through the chain: {W['measured']}.")
    L += [f"- {w}" for w in worst] + [""]

    def cell(x):
        if not x or any(c not in x for c in TX_CLASSES):
            return "—"
        t = " / ".join(f"{x[c]:+.1f}" for c in TX_CLASSES)
        return f"**{t}**" if all(x[c] >= 0.0 for c in TX_CLASSES) else t
    head = "| Sea state | Waveform, closing speed | " + " | ".join(f"{o['id']}{_tag(o)}: {o['energy_db']:.1f} dB" for o in opts) + " |"
    sep = "| --- | --- | " + " | ".join("---" for _ in opts) + " |"
    for table, label in (("bubble_law", f"with the {law} bubble law"), ("no_bubble_law", "without a bubble law")):
        rows = W["deep"].get(table, {})
        if not rows:
            continue
        L += [f"### Deep water (`{TX_DEEP}`), {label}: margin at 12 km on a mover, plain filter", "",
              "Each cell: `large_submerged` / `ship`, dB; **bold** where both hold. The zero-speed margin is section 0's table.", "", head, sep]
        for n in TX_SEA_STATES:
            if str(n) not in rows:
                continue
            for w in ("lfm", "hfm"):
                for v in sp:
                    L.append(f"| {n} | {w.upper()}, {v} m/s | " + " | ".join(cell(rows[str(n)][o["id"]][w][v]) for o in opts) + " |")
        L.append("")
    L += ["### The least energy that holds both classes on a mover with no bank", "",
          "| Table | Sea state | " + " | ".join(f"{w.upper()} at {v} m/s" for w in ("lfm", "hfm") for v in sp) + " |", "| --- | --- | " + " | ".join("---" for _ in range(2 * len(sp))) + " |"]
    for table in ("bubble_law", "no_bubble_law"):
        for n in TX_SEA_STATES:
            if str(n) in least.get(table, {}):
                L.append(f"| {table.replace('_', ' ')} | {n} | " + " | ".join(le(table, n, w, v) for w in ("lfm", "hfm") for v in sp) + " |")
    L += ["", "| Option | LFM peak loss at " + " / ".join(sp) + " m/s | LFM range shift (m per m/s; at " + sp[-1] + " m/s) | HFM peak loss at " + " / ".join(sp)
          + " m/s | HFM range bias (m per m/s; at " + sp[-1] + " m/s) |", "| --- | --- | --- | --- | --- |"]
    for o in opts:
        q = per[o["id"]]
        L.append(f"| {o['id']} | " + " / ".join(f"{q['lfm']['loss_db'][v]:.1f}" for v in sp) + f" dB | {q['lfm']['range_m_per_m_s']:.1f}; {q['lfm']['range_error_m'][sp[-1]]:+.0f} m | "
                 + " / ".join(f"{q['hfm']['loss_db'][v]:.2f}" for v in sp) + f" dB | {q['hfm']['range_m_per_m_s']:.1f}; {q['hfm']['range_error_m'][sp[-1]]:+.0f} m |")
    M12 = deep12_measured()
    L += ["", "Not in this table: the LFM's loss as the chain's detector sees it (smaller: it integrates over cells), the sweep bank (section 0's zero-speed margins less "
          "its threshold correction), crossing targets (0.4 m/s: section 0's cost table), several pings. "
          + (deep12_doppler_text(M12) if M12 else "Not run: the HFM at 12 km through the eigenray scene (a GPU-job item: tools/doppler_gpu_runs.sh `hfm12`)."), ""]
    return L


def _ss(n):
    return "none of 2-6" if n is None else f"up to sea state {n}"


DEEP12 = ROOT / "docs" / "tasks" / "doppler_deep12_eval.json"     # tools/deep12_eval.py: the deep site through the simulator and the chain


def deep12_measured() -> dict | None:
    """The deep site measured through the simulator and both chain engines (docs/tasks/doppler_deep12_eval.json), or None."""
    return json.loads(DEEP12.read_text()) if DEEP12.exists() else None


def _m12_rows(M: dict) -> dict:
    return {(r["case"], r["target"]): r for r in M["rows"]}


def deep12_bullet(M: dict) -> str:
    """The lead bullet of section 0: what the table's margins are for, and the measured acceptance of both designs (computed)."""
    R = _m12_rows(M)
    pts = [abs(r["difference_db"]) for r in M["rows"] if r["target"] == "point" and r["measured_detected"] == r["measured_pings"] and r["difference_db"] is not None]
    hull = [r["difference_db"] for r in M["rows"] if r["target"] in ("closing", "oblique", "ship") and r["measured_detected"] == r["measured_pings"]
            and (r["filter"] != "plain" or "hfm" in r["case"]) and r["difference_db"] is not None]
    scales = " and ".join(f"{s_['length_m']:g} m" for s_ in M["detector_scales"]["1 s"])

    def fails(name):
        a = M["acceptance"].get(name)
        if not a:
            return "not run"
        bad = [f"`{lb}` {v['detected']} of {v['pings']}" for lb, v in a["targets"].items() if v["verdict"] == "FAIL"]
        good = [f"`{lb}` {v['detected']} of {v['pings']}" for lb, v in a["targets"].items() if v["verdict"] == "PASS"]
        return f"{a['what']}: fails for " + (", ".join(bad) or "none") + "; passes for " + (", ".join(good) or "none")
    return ("**MEASURED, and worse than the table: the table's margins are for an UNRESOLVED, NON-MOVING target of the stated strength.** "
            "Through the simulator and both chain engines at the deep site (docs/tasks/doppler_deep12_eval.json; docs/tasks/DOPPLER.md \"Deep site at 12 km\"; "
            f"ours, one seed, 6 pings a case), the planner's own +25 dB point is reproduced within {max(pts):.1f} dB wherever it is detected on every ping; a hull of the "
            f"class's +25 dB made of five highlights that resolve in range (the detector tests {scales} cells only) comes {-max(hull):.1f} to {-min(hull):.1f} dB under its own "
            "prediction with a filter that takes no Doppler loss (the bank, or a hyperbolic sweep), and the linear sweep with the plain filter loses more on a mover. "
            "So " + fails("baseline") + "; " + fails("alternate") + " (" + fails("alternate_hfm").split(": ", 1)[1].replace("fails for", "with a hyperbolic sweep: fails for") + "). "
            "Detected on half the pings or more counts as held (the design's 0.5 a ping)."
            + ((" **Measured energy candidates** (hyperbolic unless said; no configuration changed): every target held by "
                + ", ".join(v["what"].split(": ", 1)[-1] for k, v in M["acceptance"].items() if k.startswith("candidate_") and v["accepted"])
                + "; not by " + ", ".join(v["what"].split(": ", 1)[-1] for k, v in M["acceptance"].items() if k.startswith("candidate_") and not v["accepted"])
                + " (the table at the end of this section).") if any(k.startswith("candidate_") for k in M["acceptance"]) else ""))


def deep12_table(M: dict) -> list:
    """Section 0's measured table (replaces its "Not run." paragraph): each design's acceptance case per target, computed."""
    R = _m12_rows(M)
    L = ["**Measured at the deep site** (`popoto_12km_deep_sea_state_5`, the planning site with the bubble law, three rows at 225 m, the class's low +25 dB: "
         "three hulls of five +18 dB highlights, the ship of five, and one +25 dB point moving with the platform; `" + M["command"] + "`; " + M.get("source", "") + "). "
         "Each cell: detected of 6 pings, mean margin over the pings detected (dB over the chain's own threshold), and the planner's prediction for the same target "
         "(`predict_point` at its geometry, less the plain filter's closed-form loss at its range rate). Held: detected on half the pings or more.", "",
         "| Design (case) | " + " | ".join(f"`{lb}` ({R[(M['acceptance']['baseline']['case'], lb)]['planner_range_rate_m_s']:+.2f} m/s)" for lb in M["acceptance"]["baseline"]["targets"]) + " |",
         "| --- | " + " | ".join("---" for _ in M["acceptance"]["baseline"]["targets"]) + " |"]
    for name, a in M["acceptance"].items():
        cells = []
        for lb, v in a["targets"].items():
            m = "—" if v["margin_db"] is None else f"{v['margin_db']:+.1f}"
            cells.append(f"{'**' if v['verdict'] == 'FAIL' else ''}{v['verdict']}{'**' if v['verdict'] == 'FAIL' else ''} {v['detected']} of {v['pings']}, {m} (planner {v['predicted_db']:+.1f})")
        L.append(f"| {a['what']} (`{a['case']}`) | " + " | ".join(cells) + " |")
    L += ["", "The reverberation-limited shelf (`popoto_12km_large_submerged`, docs/tasks/DOPPLER.md HFM.8): every hull held on 6 of 6 pings with every "
          "transmission tried; four times the energy bought 1 to 2.4 dB there.", ""]
    return L


def deep12_doppler_text(M: dict) -> str:
    """Section 0a's measured sentence (replaces its "Not run" one): plain linear against hyperbolic and the bank, 4 s at 200 dB, sea state 3."""
    R = _m12_rows(M)
    out = []
    for lb in ("ship", "closing", "oblique", "crossing"):
        a, b, c = (R.get((k, lb)) for k in ("ss3_lfm4p200", "ss3_hfm4p200", "ss3_lfm4p200_cfar"))
        if not (a and b and c) or b["measured_margin_db"] is None:
            continue
        lin = "not detected" if a["measured_margin_db"] is None else f"{a['measured_margin_db']:+.1f} dB ({a['measured_detected']} of 6)"
        out.append(f"`{lb}` ({a['planner_range_rate_m_s']:+.2f} m/s): linear {lin}, hyperbolic {b['measured_margin_db']:+.1f}, bank {c['measured_margin_db']:+.1f} dB "
                   f"(closed-form linear loss {a['planner_closed_form_loss_db']:.1f} dB)")
    return ("Run through the chain: on the shelf scene (docs/tasks/DOPPLER.md HFM.8) and at the deep site (docs/tasks/doppler_deep12_eval.json; 4 s at 200 dB, sea state 3, "
            "plain filter unless said): " + "; ".join(out) + ". The hyperbolic sweep with the plain filter held what the bank holds, at the plain chain's cost; the "
            "linear sweep's loss through the chain is less than its closed form but grows with the speed.")


def tx_flags(T: dict) -> None:
    """The options' `shipped` / `former_shipped` flags from SHIPPED and FORMER (D-256 moved the shipped one after the grid ran:
    the stored flags are the grid's; the report says what is shipped now)."""
    for o in T.get("options", []):
        o["shipped"] = bool(o["pulse_s"] == SHIPPED["pulse_s"] and o["source_level_db"] == SHIPPED["source_level_db"])
        o["former_shipped"] = bool(o["pulse_s"] == FORMER["pulse_s"] and o["source_level_db"] == FORMER["source_level_db"])


def _tag(o: dict) -> str:
    return " (shipped, hyperbolic)" if o["shipped"] else " (shipped until D-256, linear)" if o.get("former_shipped") else ""


def transmit_section(R: dict) -> list:
    """## 0: the transmit options by sea state, for the owner (the table first, then what each costs and where energy does not help)."""
    T = R.get("transmit")
    if not T:
        return []
    tx_flags(T)
    opts, res, A = T["options"], T["grid"]["results"], T["answers"]
    ship_id = next(o["id"] for o in opts if o["shipped"])
    law = T.get("bubble_law")
    L = ["## 0. Transmit options by sea state (`--transmit`)", "",
         f"The owner, 2026-11-27: \"lets have options for the transmit in different sea states, like a table. you can keep the default we have now but lets make sure the trade study covers options.\" "
         f"(When the grid ran no default moved.) **Since D-256 (the owner, 2026-11-28) the shipped transmitter is {ship_id}, a hyperbolic sweep (HFM), plain matched filter; until then "
         f"{next(o['id'] for o in opts if o.get('former_shipped'))}, linear.** The planner is waveform-blind (the energy a ping and the detector's window), so every cell holds for either sweep "
         f"at zero radial speed; what a mover costs each sweep is section 0a. Grid: `{T['command']}`, {T['date']}, tree `{T['git_head_when_run']}`. "
         f"Every figure is **ours** (the repository's model: `depthplan.plan` of the shipped {T['geometry']} face at the own depth it recommends for each option, worst over the class's depths, at the LOW end of its assumed strength, bound A); "
         "sites **illustrative**, target strengths **assumed**. Each cell: margin at 12 km, dB, `large_submerged` / `ship`; **bold** where both hold. A margin is for ONE ping and an echo with no radial speed "
         "(or the sweep bank on); what a mover costs the plain matched filter is in the second table.", ""]
    # worst first
    worst = []
    for table, label in (("bubble_law", f"with the {law} bubble law"), ("no_bubble_law", "without a bubble law")):
        le = A["least_energy"].get(table, {})
        none = [n for n, v in le.items() if v["energy_db"] is None]
        if none:
            v = le[none[0]]
            worst.append(f"At sea state {', '.join(none)} ({label}) **no option in the grid holds both classes at the deep site**; the best is {v['best']} "
                         f"({', '.join(f'{c} {m:+.1f}' for c, m in v['best_margins_db'].items())} dB at sea state {none[0]}).")
    top_ship = A["highest_sea_state"].get("bubble_law", {}).get(ship_id)
    top_ship_plain = A["highest_sea_state"].get("no_bubble_law", {}).get(ship_id)
    worst.append(f"**The shipped transmitter ({ship_id}) holds both classes at the deep site up to sea state {top_ship} with the {law} law (sea state {top_ship_plain} without it)**; "
                 + "; ".join(f"at sea state {n}: " + " / ".join(f"{res[f'{TX_DEEP}__ss{n}__{law}'][ship_id][c]['margin_db']:+.1f}" for c in TX_CLASSES) + " dB with the law, "
                             + " / ".join(f"{res[f'{TX_DEEP}__ss{n}'][ship_id][c]['margin_db']:+.1f}" for c in TX_CLASSES) + " dB without"
                             for n in TX_SEA_STATES if top_ship is None or n == top_ship + 1) + " (`large_submerged` / `ship`).")
    for table, label in (("bubble_law", f"with the {law} law"), ("no_bubble_law", "without a bubble law")):
        need = []
        for n, v in A["least_energy"].get(table, {}).items():
            if v["energy_db"] is not None and not any(next(o for o in opts if o["id"] == i)["costs"]["projector"]["single_units"] for i in v["options"]):
                need.append(f"sea state {n} ({', '.join(v['options'])})")
        if need:
            worst.append(f"**The least energy that holds both classes needs a level no single listed projector gives** ({label}): " + "; ".join(need)
                         + " (the projector column says what it would take; unconfirmed).")
    fo = next(o for o in opts if o.get("former_shipped"))
    sh = next(o for o in opts if o["shipped"])
    worst.append(f"**Without the sweep bank (every Doppler option is off in the baseline) a mover pays a LINEAR sweep's plain matched filter peak loss**, and it grows with the pulse as fast as the energy does: "
                 + "; ".join(f"{o['id']}: {o['costs']['mismatch_loss_db']['2']:.1f} / {o['costs']['mismatch_loss_db']['8']:.1f} dB at 2 / 8 m/s" for o in opts if o["source_level_db"] == 197.0)
                 + f". Its tolerance c / (2 B T) is {fo['costs']['doppler_tolerance_m_s']:.3f} m/s at 1 s and {min(o['costs']['doppler_tolerance_m_s'] for o in opts):.3f} m/s at 8 s: "
                 "**at every pulse length in the grid, the former shipped 1 s included, the class's 2-8 m/s needs the bank with a linear sweep**; the shipped hyperbolic sweep (D-256) "
                 "needs none (section 0a: 0.2 dB or less at 8 m/s at every length; its cost is a range bias). "
                 "This is the closed form for the PEAK (the planner's statistic); the chain's multiscale detector sums cells and lost less on the one measured hull (docs/tasks/DOPPLER.md LS.3: "
                 "bank off, closing at -5.61 m/s with a 1 s sweep, 3.4 dB under the crossing target; the bank gave back +0.8 dB net). "
                 + ("Longer pulses through the chain: measured (section 0a's last paragraph)." if deep12_measured() else "Longer pulses through the chain are not measured."))
    eb = A["energy_buys"]
    flat = [f"{s} `{c}` ({v['from_db']:+.1f} -> {v['to_db']:+.1f} dB for +{v['energy_added_db']:.1f} dB of energy; {v['limited_by_at_most']}"
            + (f", ceiling {v['ceiling_db']:+.1f}" if v.get("ceiling_db") is not None else "") + ")"
            for s, e in eb.items() for c, v in e["classes"].items() if v["gained_db"] < v["energy_added_db"] - 1.0]
    never = [f"{s} `{c}` ({v['to_db']:+.1f} dB at the most energy)" for s, e in eb.items() for c, v in e["classes"].items() if not v["holds_at_most"]]
    if never:
        worst.append("**Where no option holds 12 km at the site's own conditions**: " + "; ".join(never) + ".")
    if flat:
        worst.append("**Where energy buys less than it costs** (reverberation sets a ceiling): " + "; ".join(flat) + ".")
    M12 = deep12_measured()
    if M12:
        worst.insert(0, deep12_bullet(M12))
    L += [f"- {w}" for w in worst] + [""]
    # the tables
    def cell(r):
        if not r or any(c not in r for c in TX_CLASSES):
            return "—"
        t = " / ".join(f"{r[c]['margin_db']:+.1f}" for c in TX_CLASSES)
        return f"**{t}**" if all(r[c]["margin_db"] >= 0.0 for c in TX_CLASSES) else t
    head = "| Sea state (wind) | " + " | ".join(f"{o['id']}{_tag(o)}{' (option B pairing)' if o['paired'] else ''}: {o['energy_db']:.1f} dB" for o in opts) + " | Least energy holding both |"
    sep = "| --- | " + " | ".join("---" for _ in opts) + " | --- |"
    from gpusonar import acoustics
    for table, suffix, label in (("bubble_law", f"__{law}", f"with the {law} bubble law (as `{TX_BUBBLE_SITE}` states it)"), ("no_bubble_law", "", "without a bubble law (as `deep_no_bottom_return` states)")):
        L += [f"### Deep water (`{TX_DEEP}`: no bottom return, isovelocity; noise-limited), {label}", "",
              "Columns: transmit option (energy per ping, dB re 1 uPa^2 s at 1 m), least energy first.", "", head, sep]
        for n in TX_SEA_STATES:
            key = f"{TX_DEEP}__ss{n}{suffix}"
            if key not in res:
                continue
            le = A["least_energy"].get(table, {}).get(str(n), {})
            least = (f"{le['energy_db']:.1f} dB: {', '.join(le['options'])}" if le.get("energy_db") is not None else "none in the grid")
            L.append(f"| {n} ({acoustics.sea_state_wind_speed(n):g} m/s) | " + " | ".join(cell(res[key].get(o["id"])) for o in opts) + f" | {least} |")
        L.append("")
    # costs
    L += ["### What each option costs", "",
          "Computed (ours): blind range c T / 2 (the receive window opens there: `operation.range_m` first range null, buildspec.window_range; the chain and the planner assume no reception "
          "while transmitting, and the simulator does not model the projector's own sound in the window); range resolution c / (2 B), unchanged; ping interval = two-way time to the window's end "
          f"(12.7 km) + the pulse, against {sh['costs']['two_way_12km_s']:.1f} s two-way to 12 km; power: {sh['costs']['power_assumptions']}; the plain matched filter's peak loss at 0.4 m/s "
          "(a crossing target at 12 km) / 2 / 8 m/s radial (the class's assumed speeds), and the range the mismatched peak smears over, 2 |r| T; the bank's hypotheses at +-8 m/s, c / (2 B T) apart "
          "(its threshold correction is measured only to 51 hypotheses: +0.78 dB at 45, DOPPLER.md LS.3); the projector by tools/projector_candidates.py's sheets.", "",
          "| Option | Energy (dB) | Blind range | Resolution | Ping interval / duty | Acoustic / from supply in the pulse (W) | Per ping (J) / mean (W) | Tolerance c/(2BT) | Plain filter loses at 0.4 / 2 / 8 m/s (smear at 8 m/s) | Bank hypotheses (+-8 m/s) | Projector (status) |",
          "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for o in opts:
        k = o["costs"]
        ml = k["mismatch_loss_db"]
        L.append(f"| {o['id']} | {o['energy_db']:.1f} | {k['blind_range_m'] / 1e3:.2f} km | {k['range_resolution_m']:.3f} m | {k['ping_interval_s']:.1f} s / {100 * k['duty']:.1f} % | "
                 f"{k['acoustic_w']:.0f} / {k['from_supply_w']:.0f} | {k['joule_per_ping']:.0f} / {k['average_w']:.0f} | {k['doppler_tolerance_m_s']:.3f} m/s | "
                 f"{ml['0.4']:.1f} / {ml['2']:.1f} / {ml['8']:.1f} dB ({k['mismatch_smear_m']['8']:.0f} m) | {k['bank_hypotheses_8_m_s']} | {k['projector']['words']} ({k['projector']['status']}) |")
    L.append("")
    # least energy per sea state with its costs
    L += ["### The least energy that holds both classes, per sea state, and what it costs", "",
          "| Table | Sea state | Option(s) | Blind range | Plain filter at 2 / 8 m/s: loss, and the margins left (`large_submerged` / `ship`) | Duty | Projector |", "| --- | --- | --- | --- | --- | --- | --- |"]
    byid = {o["id"]: o for o in opts}
    for table, suffix in (("bubble_law", f"__{law}"), ("no_bubble_law", "")):
        for n, v in A["least_energy"].get(table, {}).items():
            if v["energy_db"] is None:
                L.append(f"| {table.replace('_', ' ')} | {n} | none in the grid (best {v['best']}: " + " / ".join(f"{m:+.1f}" for m in v["best_margins_db"].values()) + " dB) | | | | |")
                continue
            for oid in v["options"]:
                o, r = byid[oid], res[f"{TX_DEEP}__ss{n}{suffix}"][oid]
                k = o["costs"]
                left = " ; ".join(f"{k['mismatch_loss_db'][s]:.1f} dB: " + " / ".join(f"{r[c]['plain_filter_on_a_mover_db'][s]:+.1f}" for c in TX_CLASSES) for s in ("2", "8"))
                L.append(f"| {table.replace('_', ' ')} | {n} | {oid} ({o['energy_db']:.1f} dB) | {k['blind_range_m'] / 1e3:.2f} km | {left} | {100 * k['duty']:.1f} % | {k['projector']['words']} |")
    L.append("")
    # the other sites
    sites = [s for s in eb]
    L += ["### Every illustrative site at its own stated conditions: where energy helps and where it does not", "",
          "| Site (conditions) | " + " | ".join(f"{o['id']}" for o in opts) + " | Least energy holding both | Energy buys (least -> most energetic option), per class |",
          "| --- | " + " | ".join("---" for _ in opts) + " | --- | --- |"]
    meta = T["grid"].get("environments", {})
    for s in sites:
        key = f"{s}__as_stated"
        e = eb[s]
        buys = "; ".join(f"`{c}` {v['from_db']:+.1f} -> {v['to_db']:+.1f} (+{v['gained_db']:.1f} of +{v['energy_added_db']:.1f} dB; {v['limited_by_at_most']}"
                         + (f", ceiling {v['ceiling_db']:+.1f}" if v.get("ceiling_db") is not None else "") + ")" for c, v in e["classes"].items())
        L.append(f"| {s} ({meta.get(key, {}).get('wind_speed_m_s', 0):g} m/s{', ' + meta[key]['bubble_model'] if meta.get(key, {}).get('bubble_model') else ''}) | "
                 + " | ".join(cell(res[key].get(o["id"])) for o in opts) + f" | {e['least_energy_both'] or 'none in the grid'} | {buys} |")
    for s, why in T["grid"].get("skipped", {}).items():
        L.append(f"| {s} | not run: {why} |" + " |" * len(opts) + " |")
    L.append("")
    L += ["Reverberation gains nothing from energy (the echo-over-reverberation ratio holds the source level and the pulse on both sides: depthplan.reverberation_of is re the source "
          "level and has no pulse in it), so a reverberation-limited margin rises less than the energy and stops at its ceiling; a propagation shadow (summer mud, `ship` near the surface) "
          "needs tens of dB more than any option. Noise-limited cells rise by the level step exactly and by 10 log10 of the pulse ratio plus the detector's threshold change for the shorter "
          "window (0.1 dB from 1 to 8 s: the window opens at the blind range).", ""]
    # multi-ping
    mp = T["multi_ping"]
    L += ["**N pings combined: not implemented** in the chain or the planner: every column holds one ping; tracks confirm on 3 hits (`live.DEFAULTS` tracker), which is not an SNR gain. For scale only, not in any column: "
          "this detector's threshold would fall by " + ", ".join(f"{v:.1f} dB for {n} pings" for n, v in mp["detector_gain_if_added_in_power_db"].items())
          + f" if their maps were added in power cell by cell as its rows' are ({mp['detector_gain_status']}); a mover at 2 / 8 m/s moves "
          + " / ".join(f"{v:.0f} m" for v in mp["motion_between_pings_m"].values()) + f" between pings against {mp['range_resolution_m']:.3f} m cells, so the cells would have to move with it. "
          + "The " + mp["textbook"] + ".", ""]
    # requirement wording
    L += ["### The requirement as the model supports it (proposed, for the owner to accept)", "",
          "Derived from the tables above (`tx_requirement`, recomputed by `--report`), not typed. POP-3's text in `docs/DESIGN.md` is not changed here. For the shipped transmitter:", "",
          f"> {T['requirement'][ship_id]}", "",
          *(["**Measured against this sentence (the measured table at the end of this section; docs/tasks/doppler_deep12_eval.json): it holds for an unresolved, non-moving target "
              "of the stated strength (the point) and for the hull seen across the line of sight; it does NOT hold for the class's hull whose highlights resolve in range "
              "(closing, oblique) nor, with the plain filter, for the moving ship: the baseline at sea state 3 fails for those three.**", ""] if deep12_measured() else []),
          "The same for each option (the highest whole sea state up to which both classes hold at the deep site, every lower state included; the sites held at their own conditions):", "",
          "| Option | Energy (dB) | Deep, with the " + str(law) + " law | Deep, no bubble law | Sites held as stated | Not held |", "| --- | --- | --- | --- | --- | --- |"]
    for o in opts:
        hs = A["highest_sea_state"]
        ok_s = [s for s in eb if all((((res.get(f"{s}__as_stated") or {}).get(o["id"]) or {}).get(c) or {}).get("margin_db", -1e9) >= 0.0 for c in TX_CLASSES)]
        L.append(f"| {o['id']}{_tag(o)} | {o['energy_db']:.1f} | {_ss(hs.get('bubble_law', {}).get(o['id']))} | {_ss(hs.get('no_bubble_law', {}).get(o['id']))} | "
                 f"{', '.join(ok_s) or 'none'} | {', '.join(s for s in eb if s not in ok_s) or 'none'} |")
    L.append("")
    M12 = deep12_measured()
    if M12:
        L += deep12_table(M12)
    else:
        L += ["**Not run.** No transmit option other than the 1 s linear sweep (shipped until D-256) was run through the simulator and the chain: a 4 s or 8 s sweep scene at 12 km is a GPU-job item "
              "(the CPU here took 27 min for six 1 s pings of `popoto_12km_large_submerged`, DOPPLER.md LS.1 item 8). The command, once the evaluation tool takes a sonar override "
              "(follow-up): `python3 tools/doppler_bank_eval.py popoto_12km_large_submerged --combine cfar --max-rate 8 --truth-at reflection --gate-range-m 100` with "
              "`sonar_overrides.transmit.waveform.duration: 4.0` (and the window opening at 3 km).", ""]
    return L


# --- report ---------------------------------------------------------------------------------------------------------

def _rounded(x, nd: int = 3):
    """The JSON's numbers to `nd` decimals (the committed file stays small; nothing in it is finer than 0.001 dB in meaning)."""
    if isinstance(x, float):
        return round(x, nd)
    if isinstance(x, dict):
        return {k: _rounded(v, nd) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_rounded(v, nd) for v in x]
    return x


def _f(v, fmt="+.1f"):
    return "—" if v is None else format(v, fmt)


REQUIREMENT = (
    "**How the 12 km requirement is stated.** `docs/DESIGN.md` Sec. 2, requirement POP-3 (written by `tools/design_report.py`): "
    "\"Frequency and source level that take the Popoto array to 12 km\". It names **no target class, target strength, aspect, detection or "
    "false-alarm probability, sea state or water**. What the design has used since (not the requirement): `docs/DESIGN.md`, \"The 12 km design "
    "point\": \"The targets are the two the study uses: a ship-sized target of +25 dB and a small target of +5 dB\"; detection probability 0.5 "
    "on one ping by the chain's own detector, with a false-alarm budget of 0.001 a ping over the window; ambient noise at 5 m/s wind; free "
    "field (no site). Since 2026-11-26 the owner has said whom it is for: 'Large submerged mostly but also Large Ship is what we care about' (D-235); `tools/design_report.py` checks POP-3 for `large_submerged` first and `ship` second, in free field and per site. The target classes "
    "(`configs/builds/targets/reference_classes.yaml`) are all **assumed** placeholders; `docs/STATUS.md` owner questions 7 and 8 and "
    "`docs/tasks/DOPPLER.md` Sec. 5 q. 4 ask for the class, its aspect, and whether a near-surface ship at 12 km in non-ducting water is a "
    "requirement at all. **So this study reports every class at every site**; which of them 12 km is for is the owner's.")

NOT_EVALUATED = [
    ("Reverberation statistics (K distribution)", "Gaussian (Rayleigh) statistics in every threshold; where a site's reverberation is K-distributed the "
     "threshold for the same false-alarm rate rises, by 2-6 dB in the figure this study's brief gives (no document in the repository states that range; unconfirmed; D-196 makes the K shape an opt-in input and no planning site states one). Only where reverberation limits: "
     "the +5 dB class at the shelf and sand sites; nothing for the noise-limited cases."),
    ("Doppler beyond the baseline", "the sweep bank's +1.9 dB is one measurement at -0.50 m/s (docs/tasks/DOPPLER.md Sec. 3.2); the tone's gain is the baseline's 33-37 dB at 2 km "
     "on a still platform in flank_patrol, applied here as 'reverberation removed, 4.1 dB of peak power lost'. The surface's own Doppler spread is not simulated (DOPPLER.md Sec. 0 item 1), "
     "so every tone figure where the surface sets the background is an upper bound."),
    ("Element patterns and elevation sidelobes", "the reverberation term has no vertical element pattern and no elevation sidelobes; a baffled face would remove about 3 dB of reverberation "
     "from behind (the credit column; ours). A vertical beam's sidelobes are measured, for the shelf only, by the rows measurement."),
    ("Range-dependent bottom", "every site is range independent (one water depth, one bottom, one profile); a slope or a change of sediment between 0 and 12 km changes both the factor "
     "and the reverberation by amounts no figure here bounds."),
    ("Extended and fluctuating echoes", "every target is a steady point at the LOW end of its class's assumed strength; a fluctuating echo needs more SNR for the same detection "
     "probability (the coverage picture's `fluctuation: swerling1` option; not used here)."),
    ("The 4 kHz band's grating lobes", "the row's pitch (0.15 m) is half a wavelength at 5 kHz and 0.7 wavelength at 7 kHz: the 3-7 kHz variant has grating lobes inside the searched "
     "sector, which the budget does not see; it would need a finer pitch, i.e. more hydrophones for the same aperture."),
]

OWNER_QUESTIONS = [
    "**Which target class is the 12 km requirement for**, at which aspect, detection probability and false-alarm rate? The answer below changes by class: "
    "the +25 dB classes need nothing more at most sites; the +5 dB class is not reached at 12 km by any variant in the grid.",
    "**Is the +5 dB `xl_auv` at 12 km a requirement?** (Answered by the owner, 2026-11-26, D-235: no; the requirement is for `large_submerged` first and `ship` second. The rest stands as information.) If it is, no row length, row count, band, pulse or source level in the grid reaches it where "
    "reverberation limits it; only processing that separates it from the reverberation (a Doppler-resolving tone, for a target moving radially) or a "
    "different sensing geometry would. The transmitter is now ours to choose (the owner, 2026-11-26): is a tone outside the sweep's band (or alternate tone pings) acceptable, and what is the slowest radial speed that must be seen?",
    "**Which waters?** Every site here is illustrative (Lambert values taken from the scenes that simulate the same water). A bottom-scattering "
    "measurement (or a survey value) at the operating site settles whether the answer is robust (see the sensitivity table).",
    "**A longer row does not fit the face** at the same pitch (64 hydrophones are 9.45 m, 96 are 14.25 m, against 9 m stated and 7 m working). Is a "
    "longer face, a second face, or a towed or static array on the table, or is 32 a row fixed?",
    "**More than two boards**: the plan says \"past two boards, the custom 32-channel ADC board in Phase 6 is cheaper\" (quoted in tools/parts_list.py). "
    "Every face build already has 4 to 12 boards. Is the custom board the route for any change in hydrophone count?",
]


def _rounded(x, nd: int = 3):
    """The JSON's numbers to `nd` decimals (the committed file stays small; nothing in it is finer than 0.001 dB in meaning)."""
    if isinstance(x, float):
        return round(x, nd)
    if isinstance(x, dict):
        return {k: _rounded(v, nd) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_rounded(v, nd) for v in x]
    return x


def _f(v, fmt="+.1f"):
    return "—" if v is None else format(v, fmt)


def _km(v):
    return "—" if v is None else f"{v / 1e3:.1f}"


def lead(R: dict) -> list:
    """The answer in a few plain statements, worst first, computed from the answers block."""
    g = R.get("grid") or {}
    res, var = g.get("results", {}), g.get("variants", {})
    A = R.get("answers", {})
    sites = [s["name"] for s in R.get("sites", []) if s["name"] not in g.get("skipped", {})]
    ship = variant_id(3, 32, "2k", 1.0, 197.0)
    bad, good = [], []
    for c in [c["name"] for c in R.get("classes", [])]:
        per = A.get("per_class_site", {}).get(c, {})
        miss = [s for s in sites if s in per and per[s]["shipped_3x32"]["A"]["margin_db"] < 0]
        held = [s for s in sites if s in per and per[s]["shipped_3x32"]["A"]["margin_db"] >= 0]
        never = [s for s in miss if not any((per[s].get(k) or {}).get("smallest") for k in ("A", "B", "A+sweep_bank", "A+baffle", "A+bank_and_baffle", "B+bank_and_baffle"))]
        tone_only = [s for s in never if (per[s].get("A+tone") or {}).get("smallest")]
        rev = [s for s in miss if (res.get(ship, {}).get(s) or {}).get("classes", {}).get(c, {}).get("A", {}).get("limited_by") == "reverberation"]
        ceil = [s for s in miss if ((res.get(ship, {}).get(s) or {}).get("classes", {}).get(c, {}).get("A", {}).get("reverberation_only_ceiling_db") or 1.0) < 0.0]
        if miss:
            t = (f"**`{c}`: 12 km is NOT met by the shipped 3 x 32 at the grid's 1 s linear sweep at {len(miss)} of {len(sites)} sites** ({', '.join(miss)}). ")
            if never:
                t += (f"At {', '.join(never)} **no variant in the grid reaches it**, under either rows bound, with the sweep bank and a baffle"
                      + (f"; only the Doppler-resolving tone (a mover with radial speed of {CREDITS['tone']['min_radial_m_s']:g} m/s or more, and a transmitter that sends a tone outside the sweep's band) does at {', '.join(tone_only)}" if tone_only else "")
                      + ". ")
                for s in never:
                    y = per[s].get("B+bank_and_baffle") or per[s].get("B") or {}
                    if y.get("best_in_grid"):
                        t += (f"{s}: best {y['best_in_grid']} with every credit and bound B, {y['best_margin_db']:+.1f} dB, {y.get('best_limited_by')}-limited"
                              + (f", and reverberation alone caps it at {y['best_ceiling_db']:+.1f} dB, so no source level closes it" if (y.get("best_ceiling_db") is not None and y["best_ceiling_db"] < 0)
                                 else f": at least {-y['best_margin_db']:.1f} dB more echo energy (source level or pulse) than that variant's would be needed") + ". ")
            fix = [s for s in miss if s not in never]
            for s in fix:
                x = per[s]
                for k in ("A", "B", "A+sweep_bank", "A+baffle", "A+bank_and_baffle", "B+bank_and_baffle"):
                    if (x.get(k) or {}).get("smallest"):
                        y = x[k]
                        t += f"At {s} the smallest that meets it is {y['smallest']} ({k}; {'fits the face' if y['fits_face'] else 'does not fit the face'}). "
                        break
            if ceil:
                t += (f"Reverberation alone caps the shipped 3 x 32's margin below 0 dB at {', '.join(ceil)}: no source level closes it there with that array "
                      "(only a narrower beam, rows that gain against reverberation, or a credit against it can). ")
            elif rev:
                t += f"Reverberation is the larger term at {', '.join(rev)}. "
            bad.append(t)
        if held:
            worst = min(held, key=lambda s: per[s]["shipped_3x32"]["A"]["margin_db"])
            good.append(f"`{c}`: met by the shipped 3 x 32 at the grid's 1 s linear sweep at {len(held)} of {len(sites)} sites; the least margin {per[worst]['shipped_3x32']['A']['margin_db']:+.1f} dB at {worst} "
                        f"({per[worst]['shipped_3x32']['A']['limited_by']}-limited).")
    out = ["**The answer, worst first.**", ""] + [f"- {t}" for t in bad] + [f"- {t}" for t in good]
    # how many hydrophones: per class, per site, the smallest geometry at the shipped waveform that holds 12 km
    geo = A.get("smallest_geometry", {})
    hyd = ["", f"**How many hydrophones** (the grid's waveform: {FORMER_WORDS}; own depth planned for each; the smallest geometry that holds 12 km, bound A / bound B):", ""]
    order = [f"{r}x{n}" for r, n in sorted(((r, n) for r in ROWS for n in LENGTHS), key=lambda t: (t[0] * t[1], t[0]))]
    for c in [c["name"] for c in R.get("classes", [])]:
        parts_ = []
        for s in sites:
            pick = []
            for bound in ("A", "B"):
                gg = (geo.get(c) or {}).get(bound) or {}
                k = next((k for k in order if s in (gg.get(k) or {}).get("sites_met", [])), None)
                pick.append(k or "none")
            parts_.append(f"{s}: {pick[0]} / {pick[1]}")
        hyd.append(f"- `{c}`: " + "; ".join(parts_) + ".")
    hyd += ["", "*none* = no row length (32, 64, 96) and no row count (1, 2, 3) at the grid's waveform holds it; the smallest change with the waveform, "
            "source level and credits as well is in the table *Where it is not met*. Only 32 a row fits the face at the same pitch.", ""]
    sens = (R.get("sensitivity") or {}).get("results") or {}
    if sens:
        flips = {}
        for s, per in sens.items():
            if "skipped" in per:
                continue
            for label, v in per.items():
                for k, by in v["by_geometry"].items():
                    if not isinstance(by, dict) or "skipped" in by:
                        continue
                    r_, n = (int(x) for x in k.split("x"))
                    for c, x in by.items():
                        b0 = ((res.get(variant_id(r_, n, "2k", 1.0, 197.0), {}).get(s) or {}).get("classes", {}).get(c) or {}).get("A", {}).get("margin_db")
                        if b0 is not None and (x["A"]["margin_db"] >= 0) != (b0 >= 0):
                            flips.setdefault(c, {}).setdefault(f"{s} ({label})", []).append(k)
        if flips:
            hyd += ["**Not robust.** Varying each illustrative site's Lambert value by +-3 dB and its wind by one sea state changes which geometries hold 12 km (bound A):", ""]
            for c, by in flips.items():
                hyd.append(f"- `{c}`: " + "; ".join(f"{k}: {', '.join(v)}" for k, v in by.items()) + ".")
            hyd += ["", "The wind (through the ambient noise, and the surface's scattering) moves the answer most; the bottom's scattering moves it where reverberation is "
                    "close to the noise. What would settle it: the operating area's ambient noise (or its wind and sea-state statistics) and a bottom-backscatter "
                    "measurement (or survey value) at 4-6 kHz.", ""]
        else:
            hyd += ["**Robust to the site values tried**: no geometry's met / not-met changes with each site's Lambert value +-3 dB or its wind +-1 sea state.", ""]
    return out + hyd


def transmitter_section(R: dict) -> list:
    """What each candidate design asks of a transmitter (the owner, 2026-11-26: the transducer, its board and the waveform are ours to
    choose, limited by what is physically obtainable). Requirements a part must meet; no part's figure is assumed."""
    g = R.get("grid") or {}
    var, A = g.get("variants", {}), R.get("answers", {})
    cands = {}
    for c, per in (A.get("per_class_site") or {}).items():
        for s, row in per.items():
            for k, x in row.items():
                if k.startswith(("A", "B")) and isinstance(x, dict) and x.get("smallest"):
                    cands.setdefault(x["smallest"], set()).add(f"{c} at {s} ({k})")
    cands.setdefault(variant_id(3, 32, "2k", 1.0, 197.0), set()).add("the design as shipped until D-256 (1 s linear)")
    L = ["## 6a. What each candidate design asks of the transmitter", "",
         "The owner (2026-11-26): \"We can [do] whatever is possible with a transducer and board and software we need to choose to go along with the popoto\". "
         "So band, level, pulse, duty and waveform are design variables; the rows below are REQUIREMENTS a projector and its drive must meet, computed from each "
         "variant (energy per ping = source level + 10 log10 pulse, dB re 1 uPa^2 s at 1 m; duty = pulse / ping interval). No product figure is assumed; the "
         "projector_selection.md candidates are named only where that document's sheets cover the requirement (via secondary).", "",
         "| Variant | Needed for | Band (sweep) | Source level (rms) | Energy per ping | Pulse | Ping interval / duty | Tone outside the sweep (if the Doppler credit is wanted) |",
         "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for vid in sorted(cands, key=lambda v: (var.get(v, {}).get("hydrophones", 0), v)):
        v = var.get(vid)
        if not v:
            continue
        tone = (f"a Hann tone of the same energy outside {v['band_hz'][0] / 1e3:g}-{v['band_hz'][1] / 1e3:g} kHz (e.g. above {v['band_hz'][1] / 1e3:g} kHz): the projector's band must "
                f"reach it, at {CREDITS['tone']['peak_power_db']:g} dB more peak power than the sweep for equal energy")
        L.append(f"| {vid} | {'; '.join(sorted(cands[vid]))[:300]} | {v['band_hz'][0] / 1e3:g}-{v['band_hz'][1] / 1e3:g} kHz | {v['source_level_db']:g} dB | "
                 f"{v['source_level_db'] + 10 * np.log10(v['pulse_s']):.1f} dB | {v['pulse_s']:g} s | {v['ping_interval_s']:.1f} s / {100 * v['pulse_s'] / v['ping_interval_s']:.1f} % | {tone} |")
    L += ["", "Where a case is noise-limited and no variant in the grid reaches it (the answer above gives the shortfall), the requirement is at least that many dB more energy "
          "per ping than the grid's best (200 dB, 1 s): a level, a longer pulse (blind range c T / 2 grows), or both; where reverberation caps the margin below 0 dB, no energy closes it.", "",
          "**Waveform: a tone outside the sweep's band in one ping, or tone and sweep on alternate pings** (the waveform package's measurement, branch d2-train, as the coordinator "
          "reported it 2026-11-26; ours, simulation, 2 pings, scatter about +-3 dB): a Hann tone INSIDE the sweep's band in the same ping gets no Doppler gain where the bottom "
          "reverberates, because the sweep's broadband reverberation fills the tone's Doppler cells (flank_patrol small vehicle: 10.9 dB; 33.9 dB with the sweep muted; 37.7 dB with "
          "the tone at 10.5 kHz above a 6-10 kHz sweep; tone alone 39.8 dB). At 12 km the scenes are noise-limited and the tone keeps its margin (33.2 dB on popoto_12km_doppler).", "",
          "| Choice | Buys | Costs |", "| --- | --- | --- |",
          "| Tone outside the sweep's band, one ping | the Doppler path and the sweep's range resolution on every ping; near the tone-alone margin (37.7 against 39.8 dB) | a projector and drive "
          "covering both bands linearly (peak-to-average rises with two signals); the energy split between them (or a longer ping: duty, blind range); a second receive band (the boards' "
          "inputs are specified for 10-100 kHz: popoto_parts.md); the tone's multipath fading |",
          "| Tone and sweep on alternate pings | each signal at full energy and in one band; no inter-signal leakage | each path updates at half the rate (a 12 km ping every 17.9 s "
          "becomes one of each every 35.9 s); the tracker must fuse two kinds of detection; the tone still fades |", ""]
    return L


def findings(R: dict) -> list:
    """Things the study ran into in the model and the files, with the numbers that show them."""
    g = R.get("grid") or {}
    res = g.get("results", {})

    def m(vid, site, c, b="A", k="margin_db"):
        return (((res.get(vid, {}).get(site) or {}).get("classes", {}).get(c) or {}).get(b) or {}).get(k)
    out = []
    ss5 = next((s for s in R.get("sites", []) if s.get("sea_state") is not None), None)
    if ss5:
        from gpusonar import acoustics
        wind = f"{acoustics.sea_state_wind_speed(ss5['sea_state']):g} m/s"
        if wind not in str(ss5.get("title")):
            out.append(f"- **The sea state's wind in the code is not the one in the words.** `{ss5['name']}` states `sea_state: {ss5['sea_state']}` and its title says "
                       f"\"{ss5.get('title')}\"; `acoustics.sea_state_wind_speed({ss5['sea_state']})` gives {wind} (via secondary: UnderwaterAcoustics.jl), which is what "
                       "the planner and this study computed with. Either the title or the state is wrong.")
        else:
            out.append(f"- `{ss5['name']}` is computed at {wind}, the table's wind for sea state {ss5['sea_state']} (via secondary: UnderwaterAcoustics.jl). Until 2026-11-26 its "
                       "title said 9.8 m/s, which is sea state 4 in that table; the owner chose to correct the title, so no number moved. The sensitivity table's "
                       "sea state 4 row shows what 9.8 m/s would give.")
    a = [m(f"3x32|{b}|1s|197dB", "shelf_100m_sand", "xl_auv") for b in ("1k", "2k", "4k")]
    if all(v is not None for v in a):
        out.append(f"- **A wider band does worse at 12 km in ducted water in this model** (shelf, xl_auv, 3 x 32: 1 kHz {a[0]:+.1f}, 2 kHz {a[1]:+.1f}, 4 kHz {a[2]:+.1f} dB, bound A): "
                   "the margin uses the compressed echo's PEAK (`lloyd` `peak_db`), and a wider band splits the bottom-bounced arrivals into separate peaks, while the detector's "
                   "threshold rises with the number of cells. A chain that integrates over the echo's extent (`cfar.integrate_cells`, D-97) would recover part of it; the reverberation "
                   "term itself falls with the band (shorter compressed pulse). The band is not a lever for 12 km here.")
    r1 = ((res.get("3x32|1k|1s|197dB", {}).get("shelf_100m_sand") or {}).get("classes", {}).get("xl_auv") or {}).get("A") or {}
    if r1 and r1.get("reach_m") == 0.0:
        out.append(f"- **The planner's own depth serves 12 km only.** For 3 x 32 at 1 kHz over the shelf it recommends {r1['own_depth_m']:g} m for the xl_auv, where the margin at 12 km is best "
                   f"({r1['margin_db']:+.1f} dB) and no range at all is held (reach 0 km): the depth that maximises the 12 km figure can lose the near field. The reaches in this study are "
                   "at that depth; a reach-maximising depth would read longer.")
    rm = R.get("rows_measurement") or {}
    if rm.get("monostatic_check"):
        out.append("- **The simulator's default reverberation is the same on every row** (correlation " + str(rm["monostatic_check"]["correlation"]) + "): its monostatic generator ignores the "
                   "arrival elevation when it steers to the hydrophones (sim/reverb.py header, a documented simplification). Every shipped multi-row scene therefore gives stacked rows "
                   "no gain against reverberation by construction, which is bound A, not a measurement; `geometry: bistatic` with `n_elevation` is needed to test it.")
    for name, v in ((R.get("crosscheck") or {}).get("cases") or {}).items():
        d, sp = v.get("measured_minus_predicted_db"), v.get("split") or {}
        if d is not None and abs(d) > 3.0 and sp:
            dd = sp["differences_db"]
            out.append(f"- **Cross-check `{name}` differs by {d:+.1f} dB (more than 3 dB).** Split: the simulated echo over noise is {dd['snr_over_noise']:+.1f} dB from the prediction and "
                       f"the echo over reverberation {_f(dd['echo_over_reverberation'])} dB, with a clairvoyant background; the rest, {d - dd['snr_over_both']:+.1f} dB, is the chain's own "
                       "measurement (its local background estimate and where the peak falls) beyond the window and mean straddle losses already taken off. The echo's level: the array at "
                       "10 m and the target at 4 m sit near the surface image's null, where the planner's rough-surface image series and the simulator's flat-surface Bellhop eigenrays "
                       "differ most (docs/DESIGN.md, 'Checked': the largest Bellhop differences are in the 10 m / 4 m null). The reverberation term agrees; the noise-limited "
                       "figure in this geometry is optimistic by about that much.")
    out.append("- **Limiting term.** The planner's `limited_by` says reverberation only when it costs more than 3 dB; this study calls the larger of the two powers at the "
               "detector the limiting term, and gives the reverberation-only ceiling beside it, because a case can be noise-dominated at 12 km and still have a ceiling below 0 dB "
               "(winter_200m_sand_profile, xl_auv): there neither term alone explains the shortfall and no source level closes it.")
    return out


def report(R: dict) -> str:
    """docs/design/popoto_trade_study.md from the JSON: the answer first."""
    g = R.get("grid") or {}
    res, var = g.get("results", {}), g.get("variants", {})
    classes = [c["name"] for c in R.get("classes", [])]
    sites = [s["name"] for s in R.get("sites", [])]
    titles = {s["name"]: s.get("title", "") for s in R.get("sites", [])}
    ship = variant_id(3, 32, "2k", 1.0, 197.0)
    A = R.get("answers", {})
    L = ["# Popoto 12 km trade study", "",
         f"**Generated by `tools/popoto_trade_study.py --report` from `docs/design/popoto_trade_study.json`** (issue sonar-3983). Grid: `{g.get('command', '')}`, "
         f"{g.get('date', '')}, tree `{R.get('git_head_when_run')}`. Nothing here has been built or measured at sea: every site value is **illustrative**, every target "
         "strength **assumed** (configs/builds/targets/reference_classes.yaml), every number is the repository's own model (status **ours**) unless another source is named.",
         "", "Margin: dB of one row's SNR over what the variant's own detector needs for a detection probability of 0.5 on one ping (buildspec.detection), at the own "
         "depth the planner recommends for that variant, worst over the class's depths, at the LOW end of its target strength, at 12 km. **A** = rows gain nothing "
         "against reverberation (the planner's assumption, ours); **B** = rows gain their full 10 log10(rows) against it. Noise-only = the same planner with the "
         "reverberation term off. *Ceiling* = the margin with infinite source level (reverberation alone): where it is below 0, no source level closes 12 km.", "",
         REQUIREMENT, ""] + lead(R) + transmit_section(R) + waveform_section(R) + ["## 1. The answer in tables", ""]
    # 1a: the shipped design per class and site
    L += ["### Is 12 km met by the shipped design (3 x 32) at the grid's waveform (4-6 kHz, 1 s linear, 197 dB: shipped until D-256)?", "",
          "The grid below was run at the former 1 s linear sweep; the shipped 2 s hyperbolic sweep (D-256) carries 3 dB more energy and opens the window at 1.5 km: "
          "its margins at every site are section 0's `2 s 197 dB` column (the planner is waveform-blind), and the depth plan (`docs/design/popoto_depth_planner.json`) is of it.", "",
          "| Site (all illustrative) | " + " | ".join(f"{c}: A / B / noise-only (limit; ceiling A)" for c in classes) + " |", "| --- | " + " | ".join("---" for _ in classes) + " |"]
    for s in sites:
        r = (res.get(ship, {}).get(s) or {})
        if "classes" not in r:
            L.append(f"| {s} | " + " | ".join(r.get("skipped", "not run") for _ in classes) + " |")
            continue
        cells = []
        for c in classes:
            x = r["classes"].get(c)
            if not x:
                cells.append("not in this water")
                continue
            ok = "**met**" if x["A"]["margin_db"] >= 0 else ("met only under B" if x["B"]["margin_db"] >= 0 else "**not met**")
            lim = x["A"]["limited_by"] + (", reverberation caps it below 0" if (x["A"]["reverberation_only_ceiling_db"] is not None and x["A"]["reverberation_only_ceiling_db"] < 0) else "")
            cells.append(f"{ok}: {_f(x['A']['margin_db'])} / {_f(x['B']['margin_db'])} / {_f(x['noise_only']['margin_db'])} dB ({lim}; {_f(x['A']['reverberation_only_ceiling_db'])}), "
                         f"reach {_km(x['A']['reach_m'])} km at {x['A']['own_depth_m']:g} m")
        L.append(f"| {s} | " + " | ".join(cells) + " |")
    L.append("")
    # 1b: smallest change where not met
    L += ["### Where it is not met: the smallest change in the grid", "",
          "Smallest = fewest hydrophones, then fewest rows, then the least change of source level, pulse and band. *Fits* = fits the face (9 m stated; 7 m working).", "",
          "| Class | Site | A: smallest variant (fits?) | B: smallest | with the sweep bank (A) | with a baffle (A) | bank and baffle (A) | Doppler tone, mover only (A) |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for c in classes:
        for s in sites:
            row = (A.get("per_class_site", {}).get(c) or {}).get(s)
            if not row:
                continue
            if row["shipped_3x32"]["A"]["margin_db"] >= 0:
                continue

            def cell(k):
                x = row.get(k) or {}
                if x.get("smallest"):
                    return (f"{x['smallest']} ({_f(x['margin_db'])} dB; " + ("fits" if x["fits_face"] else "does not fit; " + (f"smallest that fits: {x['smallest_that_fits_face']}"
                            if x.get("smallest_that_fits_face") else "none that fits")) + ")")
                return f"none in the grid (best {x.get('best_in_grid')}: {_f(x.get('best_margin_db'))} dB)"
            L.append(f"| {c} | {s} | {cell('A')} | {cell('B')} | {cell('A+sweep_bank')} | {cell('A+baffle')} | {cell('A+bank_and_baffle')} | {cell('A+tone')} |")
    L.append("")
    # 1c: hydrophones
    L += ["### How many hydrophones", "", f"At the grid's waveform ({FORMER_WORDS}), each geometry, the sites at which 12 km is met (bound A / bound B):", "",
          "| Geometry | Hydrophones | Fits the face | " + " | ".join(classes) + " |", "| --- | --- | --- | " + " | ".join("---" for _ in classes) + " |"]
    geo = A.get("smallest_geometry", {})
    for r_ in ROWS:
        for n in LENGTHS:
            k = f"{r_}x{n}"
            cells = []
            for c in classes:
                a_ = ((geo.get(c) or {}).get("A") or {}).get(k, {})
                b_ = ((geo.get(c) or {}).get("B") or {}).get(k, {})
                cells.append(f"{len(a_.get('sites_met', []))} / {len(b_.get('sites_met', []))} of {len(sites)}")
            v = var.get(variant_id(r_, n, "2k", 1.0, 197.0), {})
            L.append(f"| {k} | {v.get('hydrophones', r_ * n)} | {'yes' if v.get('fits_face') else 'no'} | " + " | ".join(cells) + " |")
    L.append("")
    sens = (R.get("sensitivity") or {}).get("results")
    if sens:
        L += ["### Sensitivity to the illustrative site values", "", f"`{R['sensitivity']['command']}`: each site's Lambert value +-3 dB and wind one sea state either side, "
              "every geometry at the grid's waveform (1 s linear, shipped until D-256). Margin at 12 km, bound A, for 1x32 / 3x32 / 3x96 (and whether any geometry's met/not-met changes against the site as shipped):", "",
              "| Site | Perturbation | " + " | ".join(classes) + " | Answer changes? |", "| --- | --- | " + " | ".join("---" for _ in classes) + " | --- |"]
        for s, per in sens.items():
            if "skipped" in per:
                L.append(f"| {s} | {per['skipped']} | " + " | ".join("" for _ in classes) + " | |")
                continue
            base = {c: {k: ((res.get(variant_id(int(k[0]), int(k[2:]), '2k', 1.0, 197.0), {}).get(s) or {}).get("classes", {}).get(c) or {}).get("A", {}).get("margin_db")
                        for k in (f"{r_}x{n}" for r_ in ROWS for n in LENGTHS)} for c in classes}
            for label, v in per.items():
                cells, changed = [], []
                for c in classes:
                    ms = []
                    for k in ("1x32", "3x32", "3x96"):
                        x = (v["by_geometry"].get(k) or {}).get(c)
                        ms.append(_f(x["A"]["margin_db"]) if x else "—")
                    cells.append(" / ".join(ms))
                    for k, x in v["by_geometry"].items():
                        b0 = base[c].get(k)
                        x = x.get(c) if isinstance(x, dict) else None
                        if x and b0 is not None and (x["A"]["margin_db"] >= 0) != (b0 >= 0):
                            changed.append(f"{c} {k}")
                L.append(f"| {s} | {label} | " + " | ".join(cells) + f" | {'yes: ' + ', '.join(changed) if changed else 'no'} |")
        L.append("")
    # 2: options with costs
    L += ["## 2. Options and what each costs", "", "Hardware facts are quoted with their status: 8 inputs a board (confirmed: the owner's plan, PMM6081 ICD); 102.4 kHz x 24 bits a channel "
          "(" + DATA["rate_status"] + "); " + DATA["batching_quote"] + "; price " + DATA["price_quote"] + ".", "",
          "| Geometry | Hydrophones | Aperture / extent (m) | Beam at broadside / at 60 deg | Fits the face (9 m stated, 7 m working) | Boards (PMM6081, 8 inputs) | Data rate, all channels | Raw recording an hour |",
          "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r_ in ROWS:
        for n in LENGTHS:
            v = var.get(variant_id(r_, n, "2k", 1.0, 197.0))
            if not v:
                continue
            L.append(f"| {r_} x {n} | {v['hydrophones']} | {v['aperture_m']:.2f} / {v['row_extent_m']:.2f} | {v['beamwidth_deg']:.2f} / {v['beamwidth_at_60deg_steer_deg']:.2f} deg | {v['fit_note']} | "
                     f"{v['boards']} ({v['boards_per_row']} a row) | {v['total_rate_bit_s'] / 1e6:.1f} Mbit/s ({v['board_rate_bit_s'] / 1e6:.1f} a board) | {v['record_bytes_per_hour_raw'] / 1e9:.0f} GB |")
    L += ["", "| Waveform option | Projector (projector_selection.md) | Blind range | Ping interval | Status |", "| --- | --- | --- | --- | --- |"]
    for vid in (variant_id(3, 32, "2k", 1.0, 197.0), variant_id(3, 32, "1k", 1.0, 197.0), variant_id(3, 32, "4k", 1.0, 197.0), variant_id(3, 32, "2k", 2.0, 194.0),
                variant_id(3, 32, "2k", 4.0, 191.0), variant_id(3, 32, "2k", 1.0, 194.0), variant_id(3, 32, "2k", 1.0, 200.0)):
        v = var.get(vid)
        if not v:
            continue
        b = BANDS[v["band"]]
        proj = b["projector"] if v["pulse_s"] == 1.0 and v["source_level_db"] == 197.0 else (
            "option B: the T335 at 237 / 168 V rms or one T170 ring (a 4 s pulse is 19 % duty against its 10 % rating)" if v["pulse_s"] > 1.0 else
            ("194 dB: the T335 at 237 V rms (Sec. 3.2)" if v["source_level_db"] < 197.0 else "200 dB: the TR-1075D (about 202 dB at 1000 W, 30 % duty) or TR-1075A at 4-5 kHz; the T335 (198.5 dB) falls short"))
        L.append(f"| {v['band_hz'][0] / 1e3:g}-{v['band_hz'][1] / 1e3:g} kHz, {v['pulse_s']:g} s, {v['source_level_db']:g} dB | {proj} | {v['blind_range_m'] / 1e3:.2f} km | {v['ping_interval_s']:.1f} s | {b['status'] if v['band'] != '2k' else 'via secondary'} |")
    L += ["", "Processing options (credits, applied on top): " + "; ".join(f"**{k}**: {v['applies_to']} ({v['source']}; {v['status']})" for k, v in CREDITS.items()) + ".", ""]
    # 3: per site tables
    L += ["## 3. The grid, site by site", "", "Each cell: A / B / noise-only margin at 12 km (dB), then the limiting term under A and the reach under A (km) where 12 km is not met. "
          "Rows: geometry at the grid's waveform (1 s linear, shipped until D-256), then one change at a time on the 3 x 32 face.", ""]
    for s in sites:
        L += [f"### {s}: {titles.get(s, '')}", ""]
        if g.get("skipped", {}).get(s):
            L += [f"Not run: {g['skipped'][s]}.", ""]
            continue
        L += ["| Variant | " + " | ".join(classes) + " |", "| --- | " + " | ".join("---" for _ in classes) + " |"]
        vids = [variant_id(r_, n, "2k", 1.0, 197.0) for r_ in ROWS for n in LENGTHS] + [
            vid for vid in var if vid.startswith("3x32|") and vid != ship]
        for vid in vids:
            r = (res.get(vid, {}).get(s) or {})
            if "classes" not in r:
                continue
            cells = []
            for c in classes:
                x = r["classes"].get(c)
                if not x:
                    cells.append("—")
                    continue
                t = f"{_f(x['A']['margin_db'])} / {_f(x['B']['margin_db'])} / {_f(x['noise_only']['margin_db'])}"
                if x["A"]["margin_db"] < 0:
                    t += f" ({x['A']['limited_by']}, {_km(x['A']['reach_m'])} km)"
                cells.append(t)
            L.append(f"| {vid} | " + " | ".join(cells) + " |")
        r0 = res.get(ship, {}).get(s) or {}
        if "classes" in r0:
            L.append("")
            L.append(f"Method: {r0['method']}; reverberation: {r0['reverberation']}; own depths {r0['own_depths_m'][0]:g}-{r0['own_depths_m'][1]:g} m; ranges {_km(r0['ranges_m'][0])}-{_km(r0['ranges_m'][1])} km "
                     f"in steps of about {r0['range_step_m']:.0f} m (the reach is read on that grid).")
        L.append("")
    cov = R.get("coverage")
    if cov:
        L += ["## 4. The coverage picture at the build's site", "", f"`{cov['command']}`: the chain's C++ coverage picture (tools/coverage_map.py's computation) at `{cov['site']}`, "
              f"reverberation on, own depth per class {cov['own_depths_m']}. Greatest range at which the class's worst depth is detected with probability 0.5 (km), with reverberation / noise only:", "",
              "| Geometry | Steering | " + " | ".join(classes) + " |", "| --- | --- | " + " | ".join("---" for _ in classes) + " |"]
        for geo_, by in cov["results"].items():
            for steer, cl in by.items():
                L.append(f"| {geo_} | {steer.replace('steer_', '').replace('deg', ' deg')} | " + " | ".join(
                    f"{_km((cl.get(c) or {}).get('max_range_m'))} / {_km((cl.get(c) or {}).get('max_range_noise_only_m'))} ({(cl.get(c) or {}).get('limited_by')})" for c in classes) + " |")
        L.append("")
    rm = R.get("rows_measurement")
    if rm:
        L += ["## 5. Stacked rows against reverberation, measured (shelf)", "", f"`{rm['command']}`. {rm['what']}", "",
              f"Shelf water (popoto_12km's: 100 m over sand, Bellhop eigenrays, Lambert -27 dB, Chapman-Harris), three-row face at {rm['own_depth_m']:g} m, a point at {rm['target_depth_m']:g} m and "
              f"{_km(rm['range_m'])} km (the +5 dB class's predicted reach); {rm['pings_reverberation']} reverberation-only pings, {rm['samples_used']} samples within +-0.05 s of the target's delay. "
              f"Reverberation generator: {rm.get('reverberation_generator')}. Correlation of the rows' reverberation (magnitude): {rm['correlation']}; the target's phase across the rows "
              f"{rm['target_row_phase_deg']} deg." + (f" **With the simulator's default (monostatic) generator the rows' reverberation is identical** (correlation {rm['monostatic_check']['correlation']}, "
              f"`{rm['monostatic_check']['command']}`): that generator ignores the arrival elevation when it steers to the hydrophones (sim/reverb.py), so no shipped scene can test the rows "
              "assumption; the bistatic generator keeps each component's elevation." if rm.get("monostatic_check") else ""), "",
              "| Rows | Bound A | Coherent vertical beam (not in the chain) | Today's chain (maps added in power): effective looks, gain | Bound B | Chain's gain against noise |", "| --- | --- | --- | --- | --- | --- |"]
        for k, v in rm["subsets"].items():
            L.append(f"| {v['label']} | 0 dB | {v['coherent_vertical_beam_gain_db']:+.2f} dB (white noise: {v['coherent_gain_against_white_noise_db']:+.2f}) | {v['effective_independent_looks']:.2f} looks: {v['chain_gain_db']:+.2f} dB | "
                     f"{v['bound_B_db']:+.2f} dB | {v['chain_gain_against_noise_db']:+.2f} dB |")
        sh = rm.get("shelf_with_measured_gain")
        if sh:
            L += ["", "The shelf's 12 km margins with the measured gains, dB (reach, km), at each one's own best depth: A / today's chain, measured gain / coherent vertical beam, measured gain / B:", "",
                  "| Geometry | " + " | ".join(classes) + " |", "| --- | " + " | ".join("---" for _ in classes) + " |"]
            for k, row in sh.items():
                L.append(f"| {k} | " + " | ".join(" / ".join(f"{_f(row[c][b]['margin_db'])} ({_km(row[c][b]['reach_m'])})" for b in ("A", "chain_measured", "coherent_measured", "B")) if c in row else "—" for c in classes) + " |")
        L.append("")
    cc = R.get("crosscheck")
    if cc:
        L += ["## 6. Two grid points through the simulator and the chain", "", f"`{cc['command']}`: variant {cc['variant']} ({cc['sonar']}); chain {cc['chain']}. "
              + next(iter(cc["cases"].values()))["what_is_compared"] + ".", "",
              "| Case | Predicted margin (A) / noise-only | Predicted chain SNR | Measured median SNR (pings detected) | Measured - predicted |", "| --- | --- | --- | --- | --- |"]
        for name, v in cc["cases"].items():
            p, m = v["prediction"], v["measured"]
            gm = v["geometry"]
            L.append(f"| {name}: {gm['ts']:+g} dB at {gm['z']:g} m, {_km(gm['range'])} km, array at {gm['own']:g} m | {_f(p['margin_db'])} / {_f(p['margin_noise_only_db'])} dB ({p['limited_by']}) | "
                     f"{p['chain_snr_expected_db']:.1f} dB | {_f(m.get('median_snr_db'), '.1f')} dB ({m['pings_detected']} of {m['pings']}) | {_f(v['measured_minus_predicted_db'])} dB |")
        L += ["", "Where a difference comes from: the same geometry simulated as a target-only ping, reverberation-only and noise-only pings, one Hann-weighted beam, a clairvoyant "
              "background, against the prediction less the Hann window (dB; measured / predicted / difference):", "", "| Case | Over noise | Echo over reverberation | Over both |", "| --- | --- | --- | --- |"]
        for name, v in cc["cases"].items():
            sp = v.get("split")
            if sp:
                L.append(f"| {name} | " + " | ".join(f"{sp[k + '_db']:.1f} / {_f(sp[k + '_predicted_db'], '.1f')} / {_f(sp['differences_db'][k])}" for k in ("snr_over_noise", "echo_over_reverberation", "snr_over_both")) + " |")
        L.append("")
    L += transmitter_section(R)
    L += ["## 7. Found on the way (reported, not fixed here)", ""] + findings(R) + [""]
    L += ["## 8. What the study could not evaluate, and how large each could be", ""] + [f"- **{a}.** {b}" for a, b in NOT_EVALUATED] + [""]
    L += ["## 9. Questions for the owner", ""] + [f"{i + 1}. {q}" for i, q in enumerate(OWNER_QUESTIONS)] + [""]
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid", action="store_true", help="the full grid over the illustrative sites and the build's classes")
    ap.add_argument("--geometry-only", action="store_true", help="with --grid: only the nine geometries at the shipped waveform")
    ap.add_argument("--sensitivity", action="store_true", help="Lambert +-3 dB and wind +-1 sea state at each site, every geometry, shipped waveform")
    ap.add_argument("--coverage", action="store_true", help="the coverage picture (C++) at the build's site for every geometry, broadside and the sector's edge")
    ap.add_argument("--measure-rows", action="store_true", help="simulator: stacked rows against the shelf's reverberation")
    ap.add_argument("--crosscheck", action="store_true", help="simulator and live chain at two grid points against the prediction")
    ap.add_argument("--monostatic-check", action="store_true", help="the rows measurement with the simulator's default (monostatic) reverberation, one ping")
    ap.add_argument("--transmit", action="store_true", help="the transmit options (level, pulse) on the shipped 3 x 32 face, the deep site at sea states 2-6 with and "
                    "without the bubble law and every site as stated, for large_submerged and ship (merged into the JSON as `transmit`)")
    ap.add_argument("--report", action="store_true", help="write the .md from the JSON")
    ap.add_argument("--site", action="append", help="one of the illustrative environments (repeatable; default all)")
    ap.add_argument("--class", dest="cls", action="append", help="a target class of the build (repeatable; default all)")
    ap.add_argument("--quick", action="store_true", help="one variant (3x32, shipped), one site (deep_no_bottom_return unless --site), one class (ship unless --class)")
    ap.add_argument("--no-bellhop", action="store_true")
    ap.add_argument("--out", default=str(OUT), help="output prefix (default docs/design/popoto_trade_study): PREFIX.json, PREFIX.md")
    args = ap.parse_args(argv)
    global _ARGV
    _ARGV = list(sys.argv[1:] if argv is None else argv)
    if not any((args.grid, args.sensitivity, args.coverage, args.measure_rows, args.monostatic_check, args.crosscheck, args.report, args.quick, args.transmit)):
        ap.error("give --grid, --sensitivity, --coverage, --measure-rows, --crosscheck, --transmit, --report or --quick")
    bs, _, _ = _imports()
    spec3 = bs.load(BASE[3])
    classes = bs.target_classes_of(spec3, range_m=RANGE_M)
    sites = bs.environments_of(spec3)
    if args.site:
        known = {e["name"]: e for e in sites}
        missing = [s for s in args.site if s not in known]
        if missing:
            ap.error(f"no site {missing}; the build has {sorted(known)}")
        sites = [known[s] for s in args.site]
    if args.cls:
        classes = [c for c in classes if c["name"] in args.cls]
    out = Path(args.out)
    path = out.with_suffix(".json")
    R = json.loads(path.read_text()) if path.exists() and not args.quick else {}
    R.setdefault("what", "Popoto 12 km trade study (sonar-3983): tools/popoto_trade_study.py")
    computes = args.grid or args.sensitivity or args.coverage or args.measure_rows or args.crosscheck or args.monostatic_check or args.quick
    if computes or not R.get("git_head_when_run"):                          # --report alone computes nothing: the stamp stays the run's
        R["git_head_when_run"] = git_head()
    if args.quick:
        sites = sites[:1] if not args.site else sites
        classes = classes[:1] if not args.cls else classes
        g = run_grid(sites, classes, [(3, 32, "2k", 1.0, SHIPPED["source_level_db"])], bellhop=not args.no_bellhop)
        R = {"command": command_line(), "grid": g, "answers": answers(g, classes)}
        out.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(R, indent=1, default=float) + "\n")
        return 0
    if args.grid:
        variants = plan_variants(full=not args.geometry_only)
        t0 = time.time()
        R["classes"] = [{k: c[k] for k in ("name", "title", "ts_db", "depth_m", "speed_m_s", "status") if k in c} for c in classes]
        R["sites"] = [{k: v for k, v in e.items()} for e in sites]
        R["credits"], R["bands"], R["pulses"], R["hardware"] = CREDITS, BANDS, {f"{k:g}": v for k, v in PULSES.items()}, DATA
        g = {"variants": {}, "results": {}, "skipped": {}, "seconds": {}}
        for site in sites:                                                       # one site at a time, saved after each
            part = run_grid([site], classes, variants, bellhop=not args.no_bellhop)
            g["variants"].update(part["variants"])
            for vid, r in part["results"].items():
                g["results"].setdefault(vid, {}).update(r)
            g["skipped"].update(part["skipped"])
            g["seconds"].update(part["seconds"])
            R["grid"] = {**g, "command": command_line(), "total_seconds": round(time.time() - t0, 1), "date": datetime.date.today().isoformat(),
                         "variants_planned": [variant_id(*v) for v in variants]}
            R["answers"] = answers(R["grid"], classes)
            path.write_text(json.dumps(R, indent=1, default=float) + "\n")
    if args.sensitivity:
        t0 = time.time()
        R["sensitivity"] = {"command": command_line(), "results": run_sensitivity(sites, classes, bellhop=not args.no_bellhop), "seconds": round(time.time() - t0, 1)}
        path.write_text(json.dumps(R, indent=1, default=float) + "\n")
    if args.coverage:
        site = next(e for e in bs.environments_of(spec3) if e["name"] == spec3["operation"]["coverage"]["environment"])
        depths = {}
        for c in classes:
            r = ((R.get("grid") or {}).get("results", {}).get(variant_id(3, 32, "2k", 1.0, 197.0), {}).get(site["name"]) or {}).get("classes", {}).get(c["name"])
            depths[c["name"]] = r["A"]["own_depth_m"] if r else 30.0
        specs = [(f"{r}x{n}", variant_spec(r, n)) for r in ROWS for n in LENGTHS]
        t0 = time.time()
        R["coverage"] = {"command": command_line(), "site": site["name"], "own_depths_m": depths, "results": run_coverage(specs, site, depths), "seconds": round(time.time() - t0, 1)}
        path.write_text(json.dumps(R, indent=1, default=float) + "\n")
    if args.measure_rows:
        meas = measure_rows()
        R["rows_measurement"] = {"command": command_line(), **meas, "shelf_with_measured_gain": shelf_with_measured_gain(meas, classes)}
        path.write_text(json.dumps(R, indent=1, default=float) + "\n")
    if args.monostatic_check:
        m = measure_rows(pings=1, n_elevation=None)
        R.setdefault("rows_measurement", {})["monostatic_check"] = {"command": command_line(), **{k: m[k] for k in ("reverberation_generator", "correlation", "row_power_db_re_mean", "subsets", "pings_reverberation")}}
        path.write_text(json.dumps(R, indent=1, default=float) + "\n")
    if args.crosscheck:
        R["crosscheck"] = {"command": command_line(), **crosscheck(R)}
        path.write_text(json.dumps(R, indent=1, default=float) + "\n")
    if args.transmit:                                                        # its own tree stamp: the grid's (the report's first line) stays the grid's
        R["transmit"] = run_transmit_study(sites, classes, bellhop=not args.no_bellhop)
        path.write_text(json.dumps(R, indent=1, default=float) + "\n")
    if args.report:
        if R.get("grid"):
            R["answers"] = answers(R["grid"], R.get("classes") or classes)          # from the grid as it stands (the rule may be newer than the run)
        if R.get("transmit"):                                                    # the same for the transmit table
            tx_flags(R["transmit"])                                              # which option is shipped now (D-256), not when the grid ran
            R["transmit"]["answers"] = tx_answers(R["transmit"])
            R["transmit"]["requirement"] = tx_requirement(R["transmit"])
            R["transmit"]["waveforms"] = tx_waveforms(R["transmit"])           # LFM against HFM on a mover, no bank (section 0a)
        R["credits"] = CREDITS
        R["rounding"] = "--report rewrites this file with every number rounded to 3 decimals (dB, m, s); the tables are made from the rounded numbers"
        R = _rounded(R)
        path.write_text(json.dumps(R, indent=1) + "\n")
        out.with_suffix(".md").write_text(report(R))
    return 0


if __name__ == "__main__":
    sys.exit(main())
