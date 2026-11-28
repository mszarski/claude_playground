#!/usr/bin/env python3
"""Depth against range on the three-row face: what the chain reports, its sigma, and whether it is honest (opt-in study).

The owner (2026-11-28): the face's rehearsal self-test reads a 60 m target wrong at 1.7 km, and the face's deployed depth
had not been measured end to end beyond about 1 km (the large submerged class is at 12 km, 30 to 300 m deep; one row's
multipath timing was measured to 12 km before: RESULTS, "Depth from multipath timing"). This tool measures it.

  physics   the closed forms, before any measurement (no simulation; seconds):
            (a) stacked rows: the rows' height in wavelengths, the angle that separates the direct and surface paths,
                2 z_t / R, against the rows' resolution lambda / L; the build's own depth-sigma budget by range
                (buildspec.performance: noise, roll, the gradient prior), with the gradient prior as deployed and as known;
            (b) multipath timing: the surface echo's two-way delay tau = 4 z_own z_t / (R c) (projector at the array's depth)
                against one range cell 1 / B and the stage's least delay min_delay_bandwidth / B, for own and target depths;
                and whether the direct echo is ahead of the direct-surface echoes by a range cell (z_own z_t / R > c / 2B),
                which decides what the rows read: the direct path, or the mean of the direct and surface elevations, which
                for a pressure-release surface is the sea surface whatever the target's depth (derivation in --physics).
  sweep     simulated pings (image source, a point target, ambient noise) through the face's deployed chain
            (buildspec.chain_config of configs/builds/popoto_face_3x32.yaml), both engines, the stacked-row depth and, opt-in
            (--multipath), the multipath-timing stage as a build would deploy it (rays, flagged depths withheld). For each
            case and ping: detected, depth and sigma, whether the truth is within 2 sigma (honest), and whether the depth tells
            "submerged at 30 m or more" from "at the surface" (the interval depth +- 2 sigma excludes the other).
            The water is isovelocity, so the chain is told so (gradient 0, sigma 0, as the build's self-test does): the sigma
            measured is the geometry's and the noise's; `sigma_deployed_m` adds the deployed gradient prior's term in
            quadrature (r^2 sigma_g / 2c, the chain's own refraction_sigma), which is what the deployed chain would state.

Every number this prints is ours (a simulation or a closed form), with the command that made it in the JSON.

Usage:
  PYTHONPATH=python python3 tools/depth_range_sweep.py --physics [--json OUT]
  PYTHONPATH=python python3 tools/depth_range_sweep.py --ranges 1700 --depths 60 --water 100 --own 30 --waveform hfm2 \
      [--pings 3] [--seeds 1,2] [--engines python,cpp] [--multipath] [--ts 25] [--json OUT] [--backend host|cuda]
  --quick: one case of a few seconds (the plumbing test, tests/python/test_depth_range_sweep.py)
On the GPU job: GPUSONAR_JOB=depthrange (tools/gpu_job.sh), which runs the full grid with --backend cuda.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from gpusonar import buildspec as bs  # noqa: E402

SPEC = "popoto_face_3x32"
SHIP_DEPTH_MAX_M = 6.0        # configs/builds/targets/reference_classes.yaml: `ship` depth_m [2, 6] (assumed)
SUBMERGED_MIN_M = 30.0        # the large submerged class's shallowest depth (the same file; assumed)
GRADIENT_SIGMA_DEPLOYED = 0.05   # 1/s: the face's deployment file (chain.elevation.gradient_sigma_per_s; unconfirmed)
ENGINE_TOLERANCE_M = 0.01    # the engines' depths agree to millimetres (their elevation maps differ by 1e-5 deg, single precision)
N_SIGMA = 2.0                 # honest: the truth within this many stated sigma (build_from_spec.DEPTH_SIGMAS)
WAVEFORMS = {
    "hfm2": None,                                                                     # the baseline (D-256): the sonar's own
    "lfm1": {"transmit": {"waveform": {"type": "lfm", "source": None, "duration": 1.0}}},   # the former 1 s linear sweep
}
DURATION = {"hfm2": 2.0, "lfm1": 1.0}


# --- the physics, before measuring ----------------------------------------------------------------------------------------

def physics(ranges=(1000.0, 1700.0, 2000.0, 3000.0, 5000.0, 8000.0, 12000.0), target_depths=(30.0, 60.0, 150.0, 300.0),
            own_depths=(5.0, 30.0, 100.0, 225.0), spec_name: str = SPEC) -> dict:
    spec = bs.load(spec_name)
    d = bs.derive(spec) if hasattr(bs, "derive") else bs.describe(spec)
    c = float(d["sound_speed"])
    f_lo, f_hi = (float(x) for x in d["band_hz"])
    band, fc = f_hi - f_lo, 0.5 * (f_lo + f_hi)
    lam = c / fc
    heights = [float(h) for h in spec["array"]["heights_m"]]
    aperture = max(heights) - min(heights)
    cell = c / (2.0 * band)
    min_db = 3.0                                                      # multipath_depth.DEFAULTS["min_delay_bandwidth"]
    out = {"command": "tools/depth_range_sweep.py --physics", "spec": spec_name, "sound_speed_m_s": c, "band_hz": [f_lo, f_hi],
           "wavelength_m": lam, "rows_heights_m": heights, "rows_aperture_m": aperture, "aperture_wavelengths": aperture / lam,
           "rows_resolution_deg": float(np.degrees(lam / aperture)), "range_cell_m": cell, "one_cell_s": 1.0 / band,
           "multipath_least_delay_s": min_db / band}
    # (a) stacked rows
    perf_dep = bs.performance(d, ranges=tuple(ranges))
    perf_known = bs.performance(d, ranges=tuple(ranges), gradient_sigma=0.0)
    rows = []
    for r in ranges:
        k = str(float(r))
        dep, known = perf_dep["depth_sigma_ship_m"].get(k), perf_known["depth_sigma_ship_m"].get(k)
        row = {"range_m": r, "row_snr_ship_db": perf_dep["row_snr_db"]["ship"][k],
               "depth_sigma_deployed_m": dep and dep["total_m"], "depth_sigma_gradient_known_m": known and known["total_m"],
               "from_noise_m": dep and dep["from_noise_m"], "from_roll_m": dep and dep["from_roll_m"], "from_gradient_m": dep and dep["from_gradient_m"],
               "aperture_resolves_direct_and_surface_below_m": float(0.5 * r * lam / aperture)}   # 2 z_t / R > lambda / L
        rows.append(row)
    out["stacked_rows"] = rows
    out["stacked_rows_assumptions"] = perf_dep["assumptions"] + "; " + json.dumps(perf_dep.get("depth_budget_inputs"))
    # (b) multipath timing, and what the rows read
    mp = []
    for zo in own_depths:
        for zt in target_depths:
            for r in ranges:
                tau = 4.0 * zo * zt / (r * c)                                # two-way, surface both ways, projector at the array's depth
                ahead = zo * zt / r                                          # m of range between the direct echo and the direct-surface pair
                mp.append({"own_depth_m": zo, "target_depth_m": zt, "range_m": r, "one_way_extra_path_m": 2.0 * zo * zt / r,
                           "two_way_delay_ms": 1e3 * tau, "delay_in_cells": tau * band,
                           "multipath_searchable": bool(tau * band >= min_db),
                           "direct_echo_ahead_m": ahead, "direct_echo_resolved": bool(ahead >= cell)})
    out["multipath"] = mp
    out["what_the_rows_read"] = (
        "Receive side: the rows see the direct path at theta_d = -(z_t - z_o) / R and the surface path at theta_s = (z_t + z_o) / R "
        "with reflection mu = -1. Unresolved by the aperture (2 z_t / R < lambda / L) they read Re[(theta_d + mu e^{i phi} theta_s) / "
        "(1 + mu e^{i phi})]; with mu = -1 and 1 / (1 - e^{i phi}) = 1/2 + (i/2) cot(phi/2) the real part is (theta_d + theta_s) / 2 = z_o / R "
        "for ANY phase phi: the straight line at that elevation reaches depth z_o - R z_o / R = 0, the sea surface. In range the "
        "direct-direct echo leads the direct-surface pair by z_o z_t / R; where that is less than a range cell c / 2B the echoes share "
        "the cell and the rows read the surface for a submerged target as for a ship. In shallow water the bottom path adds its own "
        "pair (D-133): the mean of direct and bottom reads the sea bed, and a mixture of all reads anything between.")
    out["rehearsal_echoes"] = {f"{r:g}": echoes(r) for r in (1000.0, 1700.0)}       # the face's rehearsal: own 30 m, target 60 m, 100 m of sand
    out["thresholds"] = {
        "direct_echo_resolved_needs_z_own_z_t_over_R_m": cell,
        "multipath_needs_z_own_z_t_over_R_m": min_db / band * c / 4.0,
    }
    return out


def echoes(range_m: float, own_z: float = 30.0, target_z: float = 60.0, water: float = 100.0, bottom=None, c: float = 1500.0,
           order: int = 2, within_m: float = 6.0) -> list:
    """The two-way echoes of a point in a flat isovelocity channel (images: a pressure-release surface, mu = -1, and a fluid
    bottom by Rayleigh's coefficient at each path's grazing angle), the projector at the array's depth: for each, how far
    behind the direct echo (m of range: half the extra two-way path), in range cells c / 2B, its pressure re the direct one
    (a pair and its mirror added), and the elevation it is received from. What the rows of DR.3 see."""
    b = dict(bottom or bs.C_ASSUMED_BOTTOM)
    cb, rho = float(b["speed"]), float(b["density"])
    flat = float(np.sqrt(max(range_m**2 - (target_z - own_z) ** 2, 1.0)))

    def rayleigh(grazing):
        k = np.cos(grazing) * c / cb
        z1, z2 = 1.0 / np.sin(grazing), rho * c / cb / np.sqrt(1.0 - k * k + 0j)
        return (z2 - z1) / (z2 + z1)
    paths = []
    for m in range(-order - 1, order + 2):
        for sign in (1, -1):
            image = 2.0 * m * water + sign * target_z
            if sign == 1:
                ns = nb = abs(m)
            else:
                ns, nb = (abs(m) + 1, abs(m)) if m <= 0 else (m - 1, m)
            if ns + nb > order:
                continue
            dz = image - own_z
            length = float(np.hypot(flat, dz))
            grazing = float(np.arctan2(abs(dz), flat))
            amp = complex((-1.0) ** ns * rayleigh(grazing) ** nb / length)
            paths.append({"length": length, "amp": amp, "elevation_deg": float(np.degrees(np.arctan2(-dz, flat))), "surface": ns, "bottom": nb})
    d0 = min(paths, key=lambda p: p["length"])
    cell = c / (2.0 * 2000.0)
    out = []
    for i, p in enumerate(paths):
        for q in paths[i:]:
            behind = 0.5 * (p["length"] + q["length"]) - d0["length"]
            if behind > within_m:
                continue
            a = (1.0 if p is q else 2.0) * p["amp"] * q["amp"] / (d0["amp"] ** 2)
            out.append({"behind_m": behind, "cells": behind / cell, "pressure": float(abs(a)), "sign": int(np.sign(a.real)) if abs(a.real) > 1e-12 else 0,
                        "reads_deg": 0.5 * (p["elevation_deg"] + q["elevation_deg"]) if p is not q else p["elevation_deg"],
                        "legs": f"{'d' if not (p['surface'] or p['bottom']) else 's' * p['surface'] + 'b' * p['bottom']}/{'d' if not (q['surface'] or q['bottom']) else 's' * q['surface'] + 'b' * q['bottom']}"})
    return sorted(out, key=lambda e: e["behind_m"])


# --- the sweep ------------------------------------------------------------------------------------------------------------

def scenario(spec: dict, *, range_m: float, target_z: float, own_z: float, water: float | None, waveform: str, seed: int, pings: int,
             ts_db: float, bearing_deg: float = 8.0, attitude: str = "level") -> dict:
    """One case as a scenario: the build's `shallow` trial (image source, baseband, the trial's noise and bottom) with the
    geometry, waveform, seed and pings asked for. `water` None: no bottom (deep water, the surface image alone)."""
    sc = copy.deepcopy(bs.scenario_configs(spec)["shallow"])
    sc.pop("truth", None)
    sign = 1.0 if spec["array"].get("look", "starboard") == "starboard" else -1.0
    rh = float(np.sqrt(max(range_m**2 - (target_z - own_z) ** 2, 1.0)))     # horizontal range for the slant range asked
    pos = [float(rh * np.sin(np.radians(bearing_deg))), float(sign * rh * np.cos(np.radians(bearing_deg))), float(target_z)]
    sc["name"] = f"depthrange_{waveform}_{int(range_m)}m_{int(target_z)}m"
    blind = 0.5 * float(sc["environment"]["sound_speed"]) * DURATION[waveform]
    lo = max(range_m - 200.0, blind + 1.0)
    sc["sonar_overrides"] = bs.deep_merge({"receive": {"kind": "baseband", "sample_rate": sc["sonar_overrides"]["receive"]["sample_rate"],
                                                       "window": {"range": [lo, range_m + 300.0]}}}, WAVEFORMS[waveform] or {})
    sc["scene"] = [{"type": "points", "label": "target", "positions": [pos], "target_strength_db": float(ts_db)}]
    sc["platform"] = dict(sc["platform"], position=[0.0, 0.0, float(own_z)])
    if attitude == "level":
        sc["platform"]["rpy"] = [0.0, 0.0, 0.0]
    sc["seed"], sc["pings"] = int(seed), {"count": int(pings), "interval": 5.0}
    if water is None:
        sc["environment"] = {k: v for k, v in sc["environment"].items() if k != "water_depth"}
        sc["propagation"] = {"surface_z": 0.0, "max_order": 1}
    else:
        sc["environment"]["water_depth"] = float(water)
        sc["propagation"] = dict(sc["propagation"], bottom_z=float(water))
    return sc, pos


def chain_for(spec: dict, water: float | None, multipath: bool) -> dict:
    """The build's deployed chain, told what the simulated water is (as its self-test does): isovelocity, the water depth."""
    chain = bs.chain_config(spec, sync=False)
    chain.pop("engine", None)
    chain.pop("coverage", None)                       # the coverage picture changes no detection and costs a ping's time
    chain["tracker"] = dict(chain.get("tracker") or {}, confirm_hits=2)
    chain["elevation"] = dict(chain["elevation"], gradient_per_s=None, gradient_sigma_per_s=0.0, water_depth=water)
    if water is None:
        chain["elevation"].pop("sector_from_depth", None)
    if multipath:
        mp = dict(bs.MULTIPATH_DEPTH_DEPLOYED)
        if water is not None:
            mp.update(water_depth=float(water), bottom=dict(bs.C_ASSUMED_BOTTOM))
        chain["multipath_depth"] = mp
    return chain


def judge(depth, sigma, truth: float) -> dict:
    """Honest (truth within N_SIGMA), and the operational call: submerged (depth - 2 sigma > the ship's deepest), surface
    (depth + 2 sigma < the submerged class's shallowest), or undecided; right or wrong against the truth."""
    if depth is None or sigma is None or not np.isfinite(sigma) or sigma <= 0.0:
        return {"honest": None, "call": "none", "call_right": None}
    honest = abs(depth - truth) <= N_SIGMA * sigma
    call = "submerged" if depth - N_SIGMA * sigma > SHIP_DEPTH_MAX_M else "surface" if depth + N_SIGMA * sigma < SUBMERGED_MIN_M else "undecided"
    right = None if call == "undecided" else (call == "submerged") == (truth >= SUBMERGED_MIN_M)
    return {"honest": bool(honest), "error_sigmas": float(abs(depth - truth) / sigma), "call": call, "call_right": right}


def attitude_history(record, pulse_s: float, rate_hz: float = 10.0) -> None:
    """What navigation sends (navigation.ins: error-free samples of the true attitude at `rate_hz`), from the transmit to the
    middle of the last echo's pulse: the deployed chain reads the attitude there (`attitude_time: arrival`). Without a
    history the chain charges attitude_rate_sigma x the time since the transmit (16 s at 12 km: 8 degrees), which is
    no property of the depth estimate. The simulator writes a history (sim/simulator.py _add_true_ins) only when the platform
    turns (`platform.rates`); this one is written for every record, by the same law (the pose's attitude turned by its
    rates), a pulse length past the record's end for margin."""
    from gpusonar import core
    s = record.samples
    t_end = s.start_time + s.n_samples / s.sample_rate + pulse_s
    pose = record.geometry.rx_pose
    p0, v0 = np.array(pose.position), np.array(pose.velocity)
    rpy0 = np.array(pose.rpy)
    old = list(record.navigation.ins or [])
    if old:                                            # the simulator turned the vehicle: keep its attitude law, extend it
        rate = np.array(old[-1].rates) if np.any(np.array(old[-1].rates) != 0.0) else (np.array(old[-1].rpy) - np.array(old[0].rpy)) / max(old[-1].t - old[0].t, 1e-9)
        rpy0 = np.array(old[0].rpy) - rate * old[0].t
    else:
        rate = np.array(pose.rates)
    out = []
    for t in np.arange(0.0, t_end + 1.5 / rate_hz, 1.0 / rate_hz):
        a = core.InsSample()
        a.t, a.position, a.rpy = float(t), tuple(p0 + v0 * t), tuple(rpy0 + rate * t)
        a.velocity, a.rates = tuple(v0), tuple(rate)
        out.append(a)
    nav = record.navigation
    nav.ins = out
    record.navigation = nav


def refraction_sigma(slant: float, c: float, gradient_sigma: float) -> float:
    return slant * slant * gradient_sigma / (2.0 * c)            # elevation.refraction_sigma (the chain's own rule)


def _first(det) -> float:
    return float(det["first_range_m"] if det.get("first_range_m") is not None else det["range_m"])


def nearest(report: dict, own, pos, gate_m: float = 8.0, ghost_m: float = 80.0):
    """The detection of the target, as the build's self-test takes it (tools/build_from_spec.py `_nearest`): by its first
    arrival, within `gate_m` of the true slant range, the nearest among those within 10 dB of the strongest there. And the
    other detections within `ghost_m` behind it (a surface or bottom echo the detector reports on its own: a ghost)."""
    rng = float(np.linalg.norm(np.asarray(pos) - np.asarray(own)))
    hits = [x for x in report["detections"] if x.get("snr_db") is not None]
    near = [x for x in hits if abs(_first(x) - rng) <= gate_m]
    if near:
        top = max(x["snr_db"] for x in near)
        near = [x for x in near if x["snr_db"] >= top - 10.0]
    det = min(near, key=lambda x: abs(_first(x) - rng)) if near else None
    ghosts = [x for x in hits if x is not det and gate_m < _first(x) - rng <= ghost_m]
    return det, rng, ghosts


def run_case(spec: dict, *, range_m, target_z, own_z, water, waveform, seed, pings, ts_db, engines, multipath, backend="host",
             attitude="level", gradient_sigma_deployed: float = GRADIENT_SIGMA_DEPLOYED) -> dict:
    from gpusonar.live import LiveChain
    from gpusonar.sim import run_scenario
    case = {"range_m": range_m, "target_depth_m": target_z, "own_depth_m": own_z, "water_depth_m": water, "waveform": waveform,
            "seed": seed, "pings": pings, "ts_db": ts_db, "attitude": attitude}
    if range_m <= 0.5 * 1500.0 * DURATION[waveform]:
        case["skipped"] = f"inside the {waveform} blind range ({0.5 * 1500.0 * DURATION[waveform]:.0f} m)"
        return case
    sc, pos = scenario(spec, range_m=range_m, target_z=target_z, own_z=own_z, water=water, waveform=waveform, seed=seed, pings=pings,
                       ts_db=ts_db, attitude=attitude)
    t0 = time.time()
    records = run_scenario(sc, backend=_backend(backend)).records
    case["simulate_s"] = round(time.time() - t0, 1)
    for rec in records:
        attitude_history(rec, DURATION[waveform])
    c = float(records[0].environment.sound_speed)
    chain = chain_for(spec, water, multipath)
    for engine in engines:
        t0 = time.time()
        with LiveChain(dict(chain, engine=engine, backend=backend)) as live:
            reports = live.run([r.copy() for r in records])
        per = []
        for rep in reports:
            det, rng, ghosts = nearest(rep, rep["ownship_xyz"], pos)
            p = {"detected": det is not None,
                 "ghosts": [{"behind_m": _first(g) - rng, "snr_db": g["snr_db"], "depth_m": g.get("depth_m"), "depth_sigma_m": g.get("depth_sigma_m"),
                             "ambiguous": bool(g.get("elevation_ambiguous")), "multipath_term_m": g.get("depth_unresolved_sigma_m"),
                             "rows": judge(g.get("depth_m"), g.get("depth_sigma_m"), target_z)} for g in ghosts]}
            if det is not None:
                el_depth, el_sigma = det.get("depth_m"), det.get("depth_sigma_m")
                p.update(snr_db=det["snr_db"], first_range_error_m=(det.get("first_range_m") or det["range_m"]) - rng,
                         elevation_deg=det.get("elevation_deg"), ambiguous=bool(det.get("elevation_ambiguous")),
                         alternatives_deg=det.get("elevation_alternatives_deg"), depth_m=el_depth, depth_sigma_m=el_sigma,
                         multipath_term_m=det.get("depth_unresolved_sigma_m"), multipath_limited=det.get("depth_multipath_limited"),
                         unresolved_span_m=det.get("depth_unresolved_span_m"), echoes=det.get("depth_unresolved_echoes"))
                p["rows"] = judge(el_depth, el_sigma, target_z)
                if el_sigma is not None:
                    dep = float(np.hypot(el_sigma, refraction_sigma(rng, c, gradient_sigma_deployed)))
                    p["sigma_deployed_m"] = dep
                    p["rows_deployed"] = judge(el_depth, dep, target_z)
                if multipath:
                    p.update(multipath_depth_m=det.get("depth_multipath_m"), multipath_sigma_m=det.get("depth_multipath_sigma_m"),
                             multipath_reason=det.get("multipath_reason"), multipath_note=det.get("multipath_note"),
                             multipath_method=det.get("multipath_method"), multipath_candidate_m=det.get("multipath_candidate_m"))
                    p["timing"] = judge(det.get("depth_multipath_m"), det.get("depth_multipath_sigma_m"), target_z)
            per.append(p)
        case[engine] = {"pings": per, "chain_s": round(time.time() - t0, 1)}
    return case


def _backend(name: str):
    from gpusonar import core
    return {"host": core.Backend.Host, "cuda": getattr(core.Backend, "Cuda", core.Backend.Host)}.get(name, core.Backend.Host)


def summary_line(case: dict, engine: str) -> str:
    if case.get("skipped"):
        return f"{case['waveform']:5s} R {case['range_m']:6.0f}  z {case['target_depth_m']:5.0f}  own {case['own_depth_m']:5.0f}  water {case['water_depth_m']}: {case['skipped']}"
    e = case[engine]
    parts = []
    for p in e["pings"]:
        if not p["detected"]:
            parts.append("miss")
            continue
        s = "no depth" if p.get("depth_m") is None else f"{p['depth_m']:.1f}+-{p['depth_sigma_m']:.1f}{'' if p['rows']['honest'] else '!'}[{p['rows']['call'][:3]}]"
        if "multipath_depth_m" in p:
            s += " | mp " + ("-" + str(p.get("multipath_reason")) if p.get("multipath_depth_m") is None else f"{p['multipath_depth_m']:.1f}+-{p['multipath_sigma_m']:.1f}")
        parts.append(s)
    return (f"{engine:6s} {case['waveform']:5s} R {case['range_m']:6.0f}  z {case['target_depth_m']:5.0f}  own {case['own_depth_m']:5.0f}  "
            f"water {case['water_depth_m']}  seed {case['seed']}: " + "; ".join(parts))


def table(paths, engine: str = "cpp") -> tuple[str, list]:
    """The sweeps' JSONs as one table, a row per case (seeds pooled), worst first: dishonest pings, then wrong calls, then
    pings without a call. Columns: detected, the stacked-row depth (median) and its stated sigma (median), honest pings,
    the call (submerged / surface / undecided / none: right or wrong), the call with the deployed gradient prior's sigma,
    and the multipath-timing depth the same way. Also whether the two engines agreed on every ping."""
    rows = {}
    for path in paths:
        data = json.loads(Path(path).read_text())
        for c in data["cases"]:
            if c.get("skipped"):
                continue
            key = (c["water_depth_m"], c["own_depth_m"], c["waveform"], c["range_m"], c["target_depth_m"])
            r = rows.setdefault(key, {"pings": 0, "detected": 0, "depth": [], "sigma": [], "honest": 0, "dishonest": 0, "calls": {}, "wrong": 0,
                                      "dep_calls": {}, "dep_wrong": 0, "mp": [], "mp_sigma": [], "mp_honest": 0, "mp_dishonest": 0, "mp_calls": {},
                                      "mp_reasons": {}, "engines_equal": True, "ghost_dishonest": 0, "mp_dep_calls": {}, "mp_dep_wrong": 0})
            other = [e for e in ("python", "cpp") if e != engine and e in c]
            for i, p in enumerate(c[engine]["pings"]):
                r["pings"] += 1
                if other:
                    q = c[other[0]]["pings"][i]
                    same = (p.get("depth_m") is None) == (q.get("depth_m") is None) and (p.get("depth_m") is None or abs(p["depth_m"] - q["depth_m"]) < ENGINE_TOLERANCE_M)
                    r["engines_equal"] = r["engines_equal"] and same and p["detected"] == q["detected"]
                r["ghost_dishonest"] += sum(1 for g in p.get("ghosts", []) if g["rows"]["honest"] is False)
                if not p["detected"]:
                    continue
                r["detected"] += 1
                j = p.get("rows") or {"honest": None, "call": "none", "call_right": None}
                if p.get("depth_m") is not None:
                    r["depth"].append(p["depth_m"])
                    r["sigma"].append(p["depth_sigma_m"])
                r["honest"] += j["honest"] is True
                r["dishonest"] += j["honest"] is False
                r["calls"][j["call"]] = r["calls"].get(j["call"], 0) + 1
                r["wrong"] += j["call_right"] is False
                jd = p.get("rows_deployed") or {"call": "none", "call_right": None}
                r["dep_calls"][jd["call"]] = r["dep_calls"].get(jd["call"], 0) + 1
                r["dep_wrong"] += jd["call_right"] is False
                if "timing" in p:
                    t = p["timing"]
                    if p.get("multipath_depth_m") is not None:
                        r["mp"].append(p["multipath_depth_m"])
                        r["mp_sigma"].append(p["multipath_sigma_m"])
                    else:
                        reason = str(p.get("multipath_reason"))
                        r["mp_reasons"][reason] = r["mp_reasons"].get(reason, 0) + 1
                    r["mp_wrong"] = r.get("mp_wrong", 0) + (t["call_right"] is False)
                    r["mp_honest"] += t["honest"] is True
                    r["mp_dishonest"] += t["honest"] is False
                    r["mp_calls"][t["call"]] = r["mp_calls"].get(t["call"], 0) + 1
                    # with the deployed gradient prior: the stage's own first-order rule, z_t r^2 sigma_g / (2 c (z_own + z_t))
                    # (multipath_depth.DEFAULTS, gradient_sigma_per_s), added in quadrature to the sigma measured with it at 0
                    if p.get("multipath_depth_m") is not None:
                        z, rr = float(p["multipath_depth_m"]), float(c["range_m"])
                        extra = z * rr * rr * GRADIENT_SIGMA_DEPLOYED / (2.0 * 1500.0 * (float(c["own_depth_m"]) + z))
                        td = judge(z, float(np.hypot(p["multipath_sigma_m"], extra)), float(c["target_depth_m"]))
                    else:
                        td = {"call": "none", "call_right": None}
                    r["mp_dep_calls"][td["call"]] = r["mp_dep_calls"].get(td["call"], 0) + 1
                    r["mp_dep_wrong"] += td["call_right"] is False
    med = lambda v: f"{np.median(v):.1f}" if v else "-"          # noqa: E731
    calls = lambda d: ", ".join(f"{k} {v}" for k, v in sorted(d.items())) or "-"   # noqa: E731
    order = sorted(rows.items(), key=lambda kv: (-(kv[1]["dishonest"] + kv[1]["ghost_dishonest"]), -kv[1]["wrong"], -kv[1]["calls"].get("undecided", 0) - kv[1]["calls"].get("none", 0), kv[0][3]))
    lines = ["| Water, own depth, pulse | Range (m) | Target (m) | Detected | Rows: depth +- sigma (median) | Honest / dishonest | Call (wrong) | Call, deployed sigma (wrong) | Ghosts dishonest | Timing: depth +- sigma | Timing honest / dishonest; no depth | Timing call (wrong) | Timing call, deployed sigma (wrong) | Engines equal |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    out = []
    for (water, own, wf, rng, z), r in order:
        lines.append(f"| {'deep' if water is None else f'{water:g} m'}, {own:g} m, {wf} | {rng:g} | {z:g} | {r['detected']}/{r['pings']} | {med(r['depth'])} +- {med(r['sigma'])} | "
                     f"{r['honest']} / {r['dishonest']} | {calls(r['calls'])} ({r['wrong']}) | {calls(r['dep_calls'])} ({r['dep_wrong']}) | {r['ghost_dishonest']} | "
                     f"{med(r['mp'])} +- {med(r['mp_sigma'])} | {r['mp_honest']} / {r['mp_dishonest']}; {calls(r['mp_reasons'])} | "
                     f"{calls(r['mp_calls'])} ({r['mp_wrong'] if 'mp_wrong' in r else 0}) | {calls(r['mp_dep_calls'])} ({r['mp_dep_wrong']}) | {'yes' if r['engines_equal'] else 'NO'} |")
        out.append({"water_depth_m": water, "own_depth_m": own, "waveform": wf, "range_m": rng, "target_depth_m": z, **r})
    return "\n".join(lines), out


def reach(rows: list) -> str:
    """From `table`'s rows: for each water, own depth, pulse and target depth, the ranges (km) at which every ping was told
    right ("submerged" for 30 m or more, "surface" for the ship), by the stacked rows and by timing, with the water known
    and with the deployed gradient prior; and the ranges with a dishonest ping."""
    groups: dict = {}
    for r in rows:
        groups.setdefault((r["water_depth_m"], r["own_depth_m"], r["waveform"], r["target_depth_m"]), []).append(r)
    lines = ["| Water, own depth, pulse | Target (m) | Rows tell it at (km) | Rows, deployed sigma | Timing tells it at | Timing, deployed sigma | Rows dishonest at (m) | Timing dishonest at (m) |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for (water, own, wf, z), rs in sorted(groups.items(), key=lambda kv: (str(kv[0][0]), kv[0][1], kv[0][2], kv[0][3])):
        rs = sorted(rs, key=lambda r: r["range_m"])
        want = "submerged" if z >= SUBMERGED_MIN_M else "surface"

        def told(key):
            """The ranges (km) at which every ping was told right."""
            ok = [f"{r['range_m'] / 1000.0:g}" for r in rs if r[key].get(want, 0) == r["pings"]]
            return " ".join(ok) or "none"
        dis = ", ".join(f"{r['range_m']:g}" for r in rs if r["dishonest"]) or "none"
        mdis = ", ".join(f"{r['range_m']:g}" for r in rs if r["mp_dishonest"]) or "none"
        lines.append(f"| {'deep' if water is None else f'{water:g} m'}, {own:g} m, {wf} | {z:g} | {told('calls')} | {told('dep_calls')} | {told('mp_calls')} | {told('mp_dep_calls')} | {dis} | {mdis} |")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--physics", action="store_true")
    ap.add_argument("--quick", action="store_true", help="one cheap case: the plumbing test")
    ap.add_argument("--ranges", default="1700")
    ap.add_argument("--depths", default="60", help="target depths, m; 4 is the surface ship (reference_classes `ship`)")
    ap.add_argument("--own", default="30", help="own depths, m")
    ap.add_argument("--water", default="100", help="water depth, m, or 'deep' (no bottom)")
    ap.add_argument("--waveform", default="hfm2", help="hfm2 (the baseline), lfm1 (the former 1 s linear sweep), or both comma separated")
    ap.add_argument("--seeds", default="1")
    ap.add_argument("--pings", type=int, default=3)
    ap.add_argument("--ts", type=float, default=25.0, help="target strength, dB (the large submerged class's low end, assumed)")
    ap.add_argument("--engines", default="python,cpp")
    ap.add_argument("--multipath", action="store_true", help="also the multipath-timing stage, as a build would deploy it")
    ap.add_argument("--attitude", default="level", choices=("level", "trial"), help="level, or the trial's roll (3 deg)")
    ap.add_argument("--backend", default="host")
    ap.add_argument("--json")
    ap.add_argument("--table", nargs="+", metavar="JSON", help="print the table of these sweeps' JSONs (no simulation)")
    args = ap.parse_args(argv)
    if args.table:
        text, rows = table(args.table)
        print(text + "\n\n" + reach(rows))
        if args.json:                       # the cases, seeds pooled, one line each; the sweeps' commands; the closed forms
            commands = [re.sub(r"--json \S+", "--json OUT", json.loads(Path(p).read_text()).get("command") or "") for p in args.table]
            lines = ['{"what": "tools/depth_range_sweep.py --table: the sweeps below pooled by case (docs/tasks/DEPTH_RANGE.md)",',
                     ' "commands": ' + json.dumps(commands) + ",", ' "physics": ' + json.dumps(physics(), default=float) + ",", ' "cases": [']
            lines += ["  " + json.dumps(r, default=float) + ("," if i + 1 < len(rows) else "") for i, r in enumerate(rows)]
            Path(args.json).write_text("\n".join(lines + [" ]}"]) + "\n")
        return 0
    if args.physics:
        out = physics()
        text = json.dumps(out, indent=1, default=float)
        if args.json:
            Path(args.json).write_text(text + "\n")
        else:
            print(text)
        return 0
    spec = bs.load(SPEC)
    fl = lambda s: [float(x) for x in s.split(",") if x]          # noqa: E731
    if args.quick:
        grid = [dict(range_m=1700.0, target_z=60.0, own_z=30.0, water=None, waveform="lfm1", seed=1)]
        args.pings, engines = 1, ("python", "cpp")
    else:
        waters = [None if w == "deep" else float(w) for w in args.water.split(",")]
        grid = [dict(range_m=r, target_z=z, own_z=o, water=w, waveform=wf, seed=int(s))
                for wf in args.waveform.split(",") for w in waters for o in fl(args.own) for r in fl(args.ranges) for z in fl(args.depths)
                for s in args.seeds.split(",")]
        engines = tuple(args.engines.split(","))
    cases = []
    for g in grid:
        case = run_case(spec, **g, pings=args.pings, ts_db=args.ts, engines=engines, multipath=args.multipath or args.quick,
                        backend=args.backend, attitude=args.attitude)
        cases.append(case)
        for engine in engines:
            print(summary_line(case, engine) if not case.get("skipped") else summary_line(case, engine), flush=True)
            if case.get("skipped"):
                break
    out = {"command": "tools/depth_range_sweep.py " + " ".join(argv if argv is not None else sys.argv[1:]), "spec": SPEC,
           "chain": "buildspec.chain_config(popoto_face_3x32) told the simulated water: isovelocity (gradient 0, sigma 0), its depth",
           "honest": f"truth within {N_SIGMA:g} stated sigma", "call": f"submerged if depth - {N_SIGMA:g} sigma > {SHIP_DEPTH_MAX_M:g} m; surface if depth + {N_SIGMA:g} sigma < {SUBMERGED_MIN_M:g} m",
           "cases": cases}
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1, default=float) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
