#!/usr/bin/env python3
"""What Doppler processing would buy, measured on the simulator's own records (issue sonar-u1fu; docs/tasks/DOPPLER.md).

A study tool: it changes nothing in the live chain, the simulator or any scenario file. Scenarios are loaded, edited in
memory (waveform, platform speed, a target's velocity) and run; the records are processed here with plain NumPy.

Two commands.

  reverb-spectrum   Is the simulated reverberation's Doppler spectrum physical? A moving platform over a Lambert bottom
                    (or a surface with wind) with a long CW pulse; per beam, the spectrum of the beamformed
                    reverberation against the expected own-motion shift 2 v f sin(bearing) / c (bearing from broadside
                    towards the direction the elements run, the platform moving along the array).

  margin SCENARIO   Each target's margin over the background on its own beam, (i) as a matched filter of one Doppler
                    hypothesis sees it (the chain's view: zero Doppler) and (ii) after a bank of Doppler hypotheses with
                    the background normalised per (beam, hypothesis) cell. Run with the scenario's own waveform and
                    with `--waveform cw` or `cw_hann` (a CW pulse at the band centre with the same duration and the
                    same energy, inside the sonar's band; the Hann-tapered one at 4.1 dB more peak power).

How the margin is measured (ours). Two kinds of run with the same configuration: target only (that target's scene
item alone; no reverberation, no ambient noise; one ping) and background only (no scene; reverberation and noise as
the scenario has them; `--pings` pings, each a new draw). Both records are converted to complex baseband
(recon.fls.prepare without compression), beamformed with a Hann aperture (the chain's default window) on the
target's beam (the bearing of the strongest target-only output), and correlated with the
replica of each hypothesis: the transmitted pulse time-scaled by (c - v) / (c + v) for range rate v (positive:
opening; gpusonar.processing.doppler_scale), normalised to unit energy. Margin = the target-only output's peak power
(over time and hypotheses) / the mean background power in the same beam and the same hypothesis, over +-`--window`
pulse lengths about the target's delay (LFM: +-0.1 s). The bearing search: 1 degree steps over +-80 degrees from
broadside, then 0.1 degree steps. The background is the clairvoyant mean of a separate run: a
real detector estimates it from neighbouring cells (a CFAR), which costs a little more. The peak is one sample, as a
detector's peak cell; the chain's cells (2 m) average a few samples.

A pulse train (`source: train`, sim/waveform.py; the scenes `*_doppler`) is processed segment by segment: each tone
or sweep has its own replica (the segment alone, from the sonar's segment table, time-scaled per hypothesis) and its
own margins, all from the same pings (the record holds the whole train as its replica; the segment table is
configuration). `--comb tones` processes each tone of a comb segment alone and sums their powers (incoherent
combination, option E); the default `whole` correlates with the comb as sent.

A third run, noise only (no scene, no reverberation, up to two pings), gives the same cells' margin against noise alone:
the ceiling that perfect rejection of the reverberation would reach.

What this does not model: the chain's CFAR, clustering and tracking; several pings' integration; the waveform's
transmitter feasibility (a CW at the band centre may not be what the projector can send: an owner question).

Examples (CPU, Bellhop for the eigenray scenes: GPUSONAR_BELLHOP):
  python3 tools/doppler_study.py reverb-spectrum --speed 3
  python3 tools/doppler_study.py margin flank_patrol --waveform cw --pings 6 --json out.json
  python3 tools/doppler_study.py margin flank_patrol --waveform cw --target small_vehicle --radial 0.25 0.5 1 2
  python3 tools/doppler_study.py margin flank_patrol_doppler --target small_vehicle --bearing 23.5

"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time

import numpy as np


def _imports():
    from gpusonar.config import load_scenario_config
    from gpusonar.sim.simulator import run_scenario
    from gpusonar.recon.fls import prepare, aperture_weights
    from gpusonar import flank
    return load_scenario_config, run_scenario, prepare, aperture_weights, flank


# The largest range rate (m/s) the baseband conversion keeps room for. Without it core.to_baseband keeps only the
# transmitted band, which for a CW is about 1/T wide: a 1 s CW record at 8 kHz comes out at 8.7 Hz of sample rate and
# an echo at 2 m/s (21 Hz off) is filtered out (found here; the live chain's prepare does the same).
MAX_RANGE_RATE = 5.0


def baseband(record):
    """[channels, n] complex baseband samples, sample rate, carrier, sound speed, start time."""
    _, _, prepare, _, _ = _imports()
    prepare([record], compress=False, max_range_rate=float(MAX_RANGE_RATE))
    s = record.samples
    return np.asarray(s.array), float(s.sample_rate), float(s.center_frequency), float(record.environment.sound_speed), \
        float(s.start_time)


def beam(record, x, fs, fc, bearing_rad):
    """One beam of a Hann-weighted conventional beamformer (exact delays in frequency, flank.py's convention)."""
    _, _, _, aperture_weights, flank = _imports()
    tau = flank.steering_delays(record, [bearing_rad])[0]
    w = aperture_weights(record, "hann")
    n = x.shape[1]
    f = np.fft.fftfreq(n, 1.0 / fs) + fc
    X = np.fft.fft(x, axis=1)
    return np.fft.ifft((w[:, None] * X * np.exp(-2j * np.pi * f[None, :] * tau[:, None])).sum(axis=0)) / w.sum()


def replica(waveform, fs, fc, range_rate=0.0, c=1500.0):
    """The echo of a target at range rate `range_rate` (m/s, positive opening), complex baseband about fc, unit
    energy: the transmitted pulse time-scaled by eta = (c - v) / (c + v). CW (type 0) and LFM (type 1) with a Tukey
    window, as libs/core waveform.hpp builds them."""
    eta = (c - range_rate) / (c + range_rate)
    T = float(waveform.duration)
    t = np.arange(int(np.floor(T / eta * fs)) + 1) / fs
    u = eta * t
    alpha = float(waveform.window_alpha)
    x = u / T
    w = np.ones_like(x)
    if alpha > 0.0:
        edge = alpha / 2.0
        lo, hi = x < edge, x > 1.0 - edge
        w[lo] = 0.5 * (1.0 - np.cos(np.pi * x[lo] / edge))
        w[hi] = 0.5 * (1.0 - np.cos(np.pi * (1.0 - x[hi]) / edge))
    f0, f1 = float(waveform.f_start), float(waveform.f_stop)
    kind = int(waveform.type)
    if kind == 0:
        phase = f0 * u
    elif kind == 1:
        phase = f0 * u + 0.5 * (f1 - f0) / T * u * u
    else:
        raise ValueError("replica: only CW and LFM waveforms are built here")
    r = w * np.exp(2j * np.pi * (phase - fc * t))
    return r / np.sqrt(np.sum(np.abs(r) ** 2))


class Proc:
    """What one processing path correlates with: the record's own analytic pulse, or one segment (or tone) of a train.
    `replicas(fs, fc, v, c)` gives the unit-energy replica of each part whose powers are summed (one part unless a
    comb is split into its tones); `offset` is the segment's start in the train (s)."""

    def __init__(self, name, duration, is_cw, band, centre, parts, offset=0.0):
        self.name, self.duration, self.is_cw, self.band, self.centre = name, float(duration), bool(is_cw), float(band), float(centre)
        self.parts, self.offset = parts, float(offset)

    def replicas(self, fs, fc, v, c):
        return [part(fs, fc, v, c) for part in self.parts]


def segment_replica(seg):
    """The replica of one train segment (sim/waveform.py TrainSegment) at range rate v: the segment alone, time-scaled
    by eta = (c - v) / (c + v), complex baseband about fc, unit energy."""
    def make(fs, fc, v, c):
        eta = (c - v) / (c + v)
        t = np.arange(int(np.floor(seg.duration / eta * fs)) + 1) / fs
        r = seg.signal(eta * t) * np.exp(-2j * np.pi * fc * t)
        return r / np.sqrt(np.sum(np.abs(r) ** 2))
    return make


def processing_paths(record, sonar, comb="whole", segments=None):
    """The paths to measure: the record's analytic pulse, or each sounding segment of a train."""
    wf = record.transmit.waveform
    train = getattr(getattr(sonar, "sampled_waveform", None), "segments", None)
    if not train:
        T = float(wf.duration)
        band = abs(float(wf.f_stop) - float(wf.f_start))
        return [Proc("pulse", T, int(wf.type) == 0, band if int(wf.type) != 0 else 1.0 / T,
                     0.5 * (float(wf.f_start) + float(wf.f_stop)), [lambda fs, fc, v, c: replica(wf, fs, fc, v, c)])]
    out = []
    for seg in train:
        if seg.kind == "gap" or (segments is not None and seg.index not in segments):
            continue
        is_cw = seg.kind == "cw"
        band = 1.0 / seg.duration if is_cw else abs(seg.f_stop - seg.f_start)
        centre = 0.5 * (seg.f_start + seg.f_stop)
        name = f"{seg.index}:{seg.kind}" + (f"x{len(seg.frequencies)}" if is_cw and len(seg.frequencies) > 1 else "")
        if is_cw and len(seg.frequencies) > 1 and comb == "tones":
            parts = [segment_replica(seg.tone(k)) for k in range(len(seg.frequencies))]
            name += ":tones"
        else:
            parts = [segment_replica(seg)]
        out.append(Proc(name, seg.duration, is_cw, band, centre, parts, seg.start))
    return out


def correlate(y, r):
    """Matched-filter output z[k] = sum_m y[k + m] conj(r[m]) (time of the echo's start), by FFT."""
    n = len(y) + len(r)
    nfft = 1 << int(np.ceil(np.log2(n)))
    z = np.fft.ifft(np.fft.fft(y, nfft) * np.conj(np.fft.fft(r, nfft)))
    return z[:len(y)]


def set_waveform(cfg, kind, sonar_cfg):
    """Replace the transmit waveform by a CW at the band centre with the scene's duration and pulse energy."""
    if kind == "scene":
        return cfg
    if str(sonar_cfg["transmit"]["waveform"].get("source", "")).lower() == "train":
        raise SystemExit("--waveform cw / cw_hann replace a scene's sweep; this scene sends a train: run it with "
                         "--waveform scene (its segments are measured one by one), or compare on the scene it was made from")
    wf = dict(sonar_cfg["transmit"]["waveform"])
    over = cfg.setdefault("sonar_overrides", {}).setdefault("transmit", {}).setdefault("waveform", {})
    wf.update(over)
    f0 = 0.5 * (float(wf["f_start"]) + float(wf["f_stop"]))
    alpha = {"cw": 0.05, "cw_hann": 1.0}[kind]
    # The simulator scales the pulse's amplitude by `source_level_db_rms` before the Tukey window (measured: a Hann CW
    # at the scene's level carries 0.39 of its energy, free field), so the same energy needs the windows' mean squares
    # made up: 1 - 5 alpha / 8 for a Tukey window of fraction alpha. A Hann CW then needs 4.1 dB more peak power than
    # the scene's LFM: a transmitter question, not free.
    a0 = float(wf.get("window_alpha", 0.0))
    level = float(wf["source_level_db_rms"]) + 10.0 * np.log10((1.0 - 5.0 * a0 / 8.0) / (1.0 - 5.0 * alpha / 8.0))
    cfg["sonar_overrides"]["transmit"]["waveform"] = dict(type="cw", f_start=f0, f_stop=f0, duration=float(wf["duration"]),
                                                          window_alpha=alpha, source_level_db_rms=float(level))
    return cfg


def scene_config(name, args):
    load_scenario_config, _, _, _, _ = _imports()
    from gpusonar.sonar import SonarModel
    cfg = load_scenario_config(name, expand=False)
    if args.platform_speed is not None:
        cfg.setdefault("platform", {})["velocity"] = [float(args.platform_speed), 0.0, 0.0]
    sonar = SonarModel.load(cfg["sonar"], cfg.get("sonar_overrides"))
    cfg = set_waveform(cfg, args.waveform, sonar.config)
    return cfg


def target_items(cfg, label):
    return [copy.deepcopy(it) for it in cfg.get("scene", []) if it.get("label") == label]


def item_centre(item):
    if "positions" in item:
        return np.mean(np.asarray(item["positions"], dtype=float), axis=0)
    return np.asarray(item["origin"], dtype=float)


def set_radial(item, platform_position, speed):
    """Give a target a velocity of `speed` m/s straight away from the platform (horizontal), and nothing across."""
    d = item_centre(item) - np.asarray(platform_position, dtype=float)
    d[2] = 0.0
    item["velocity"] = list(map(float, speed * d / np.linalg.norm(d)))
    return item


def radial_speed(item, platform):
    """Range rate (m/s, positive opening) of a target's centre relative to the platform at the first ping."""
    p = np.asarray(platform.get("position", [0, 0, 0]), dtype=float)
    vp = np.asarray(platform.get("velocity", [0, 0, 0]), dtype=float)
    d = item_centre(item) - p
    u = d / np.linalg.norm(d)
    vt = np.asarray(item.get("velocity", [0, 0, 0]), dtype=float)
    return float((vt - vp) @ u), float(-vp @ u)


def run(cfg, pings):
    _, run_scenario, _, _, _ = _imports()
    cfg = copy.deepcopy(cfg)
    cfg["pings"] = dict(cfg.get("pings", {}), count=int(pings))
    return run_scenario(cfg).records


def hypotheses(duration, fc, c, vmax):
    """Range rates of the bank: a grid at half the speed resolution of a CW of the pulse's length, c / (2 f T), out to
    +-vmax, for either waveform (an LFM's bank shows how little a Doppler-tolerant pulse separates)."""
    res = c / (2.0 * fc * float(duration))
    k = int(np.ceil(vmax / (0.5 * res)))
    return np.arange(-k, k + 1) * 0.5 * res, res


def margins(cfg, label, args, background, item_override=None, noise=None):
    """Margins of one target (dB): zero-Doppler matched filter, and the Doppler bank's best cell, for each processing
    path (the pulse; or each segment of a train). Returns one dict per path."""
    from gpusonar.sonar import SonarModel
    t0 = time.time()
    tcfg = copy.deepcopy(cfg)
    items = target_items(cfg, label) if item_override is None else [item_override]
    tcfg["scene"] = items
    tcfg["reverberation"] = []
    rx = dict(tcfg.get("receiver", {}))
    rx.pop("ambient_noise", None)
    rx["noise_rms"] = 0.0
    tcfg["receiver"] = rx
    rec_t = run(tcfg, 1)[0]
    x, fs, fc, c, start = baseband(rec_t)
    sonar = SonarModel.load(cfg["sonar"], cfg.get("sonar_overrides"))
    procs = processing_paths(rec_t, sonar, getattr(args, "comb", "whole"), getattr(args, "segments", None))
    # The target's beam: the strongest zero-Doppler output on a fine grid of bearings (from broadside), through the
    # sweep where there is one (its peak is sharp), or a bearing given.
    if getattr(args, "bearing", None) is not None:
        bearing = float(np.radians(args.bearing))
    else:
        search = next((p for p in procs if not p.is_cw), procs[0])
        r0 = search.replicas(fs, fc, 0.0, c)
        best, bearing = -1.0, 0.0
        for step, span in ((1.0, 80.0), (0.1, 1.0)):
            centre = np.degrees(bearing)
            for b in np.radians(np.arange(centre - span, centre + span + 1e-9, step)):
                yb = beam(rec_t, x, fs, fc, b)
                e = np.max(sum(np.abs(correlate(yb, r)) ** 2 for r in r0))
                if e > best:
                    best, bearing = e, b
    yt = beam(rec_t, x, fs, fc, bearing)
    bg_beams = [beam(rec_b, xb, fsb, fcb, bearing) for (xb, fsb, fcb, cb, sb), rec_b in zip(background, background_records(background))]
    nz_beams = ([beam(rec_b, xb, fsb, fcb, bearing) for (xb, fsb, fcb, cb, sb), rec_b in zip(noise, background_records(noise))]
                if noise else None)
    results = []
    for proc in procs:
        results.append(_proc_margins(proc, label, args, rec_t, yt, bg_beams, nz_beams, fs, fc, c, start, bearing))
    for r in results:
        r["seconds"] = time.time() - t0
    return results


def _proc_margins(proc, label, args, rec_t, yt, bg_beams, nz_beams, fs, fc, c, start, bearing):
    rates, res = hypotheses(proc.duration, proc.centre, c, args.vmax)
    reps = [proc.replicas(fs, fc, v, c) for v in rates]                    # [hyp][part]
    z_len = len(reps[list(rates).index(0.0) if 0.0 in rates else 0][0])

    def power(y, parts):
        return sum(np.abs(correlate(y, r)) ** 2 for r in parts)

    zt = np.array([power(yt, parts) for parts in reps])
    zb = np.array([[power(yb, parts) for parts in reps] for yb in bg_beams])         # [pings, hyp, n]
    zn = np.array([[power(yb, parts) for parts in reps] for yb in nz_beams]) if nz_beams else None
    T = proc.duration
    half = (args.window * T) if proc.is_cw else 0.1
    valid = len(yt) - z_len
    z0 = int(np.where(rates == 0.0)[0][0])
    i0 = int(np.argmax(zt[z0][:valid]))
    k = np.argmax(zt[:, :valid].max(axis=1))
    i1 = int(np.argmax(zt[k, :valid]))

    def bg(h, i, z=None):
        z = zb if z is None else z
        lo, hi = max(0, i - int(half * fs)), min(valid, i + int(half * fs) + 1)
        return float(z[:, h, lo:hi].mean()), (hi - lo) / fs

    b0, span = bg(z0, i0)
    b1, _ = bg(k, i1)
    # The reverberation's own spread on this beam: where the background's power, as a function of hypothesis at the
    # target's delay, is within 10 dB of its peak.
    prof = np.array([bg(h, i1)[0] for h in range(len(rates))])
    near = rates[prof >= prof.max() / 10.0]
    out = dict(label=label, segment=proc.name, bearing_deg=float(np.degrees(bearing)), waveform=args.waveform,
               duration_s=T, carrier_hz=proc.centre, range_m=float(0.5 * c * (start + i1 / fs - proc.offset)),
               zero_doppler_margin_db=float(10 * np.log10(zt[z0, i0] / b0)),
               bank_margin_db=float(10 * np.log10(zt[k, i1] / b1)),
               bank_best_range_rate=float(rates[k]), speed_resolution_m_s=res,
               range_resolution_m=(0.5 * c * T if proc.is_cw else 0.5 * c / proc.band),
               background_window_s=span, background_pings=len(bg_beams),
               reverb_within_10db_of_peak_m_s=([float(near.min()), float(near.max())] if len(rates) > 1 else None))
    out["gain_db"] = out["bank_margin_db"] - out["zero_doppler_margin_db"]
    if zn is not None:
        # The same cells against noise alone (no reverberation): the ceiling a perfect reverberation rejection reaches.
        out["noise_only_zero_doppler_margin_db"] = float(10 * np.log10(zt[z0, i0] / bg(z0, i0, zn)[0]))
        out["noise_only_bank_margin_db"] = float(10 * np.log10(zt[k, i1] / bg(k, i1, zn)[0]))
    # The echo's energy over its spread. A CW cell (c T / 2 long) holds every multipath arrival and highlight of the
    # target at once; a sweep's peak is one arrival. The spread: the shortest window about the bank's peak (within
    # +-0.5 s) holding 90 % of the echo's energy in that hypothesis; the margin: that energy over the background's mean
    # times the window's samples (a deflection: an energy detector over the window, which still has to pay for its
    # background's fluctuation over N independent cells, N = window x the band's width).
    lo, hi = max(0, i1 - int(0.5 * fs)), min(valid, i1 + int(0.5 * fs) + 1)
    e = zt[k, lo:hi]
    total = e.sum()
    order = np.argsort(e)[::-1]
    keep = np.zeros(len(e), bool)
    acc = 0.0
    for j in order:
        keep[j] = True
        acc += e[j]
        if acc >= 0.9 * total:
            break
    idx = np.where(keep)[0]
    a, b = lo + idx.min(), lo + idx.max() + 1
    width = (b - a) / fs
    out["spread_90_s"] = float(width)
    out["spread_cells"] = float(max(1.0, width * proc.band))
    out["spread_energy_margin_db"] = float(10 * np.log10(zt[k, a:b].sum() / (b1 * (b - a))))
    return out


_RECORDS = {}


def background_records(background):
    return _RECORDS[id(background)]


def run_background(cfg, pings):
    bcfg = copy.deepcopy(cfg)
    bcfg["scene"] = []
    recs = run(bcfg, pings)
    bb = [baseband(r) for r in recs]
    _RECORDS[id(bb)] = recs
    return bb



def cmd_margin(args):
    cfg = scene_config(args.scenario, args)
    plat = cfg.get("platform", {})
    labels = args.target or [it["label"] for it in cfg.get("scene", []) if "label" in it]
    t0 = time.time()
    background = run_background(cfg, args.pings)
    ncfg = copy.deepcopy(cfg)
    ncfg["reverberation"] = []
    noise = run_background(ncfg, min(args.pings, 2))
    results = dict(scenario=args.scenario, waveform=args.waveform, platform_velocity=plat.get("velocity"),
                   background_seconds=time.time() - t0, targets=[])
    for label in labels:
        item = target_items(cfg, label)[0]
        cases = [None] if not args.radial else [float(v) for v in args.radial]
        for v in cases:
            it = copy.deepcopy(item)
            if v is not None:
                set_radial(it, plat.get("position", [0, 0, 0]), v)
            for m in margins(cfg, label, args, background, it, noise):
                m["range_rate_m_s"], m["platform_part_m_s"] = radial_speed(it, plat)
                results["targets"].append(m)
                print(json.dumps({k: (round(val, 3) if isinstance(val, float) else val) for k, val in m.items()}), flush=True)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=1)
    return results


def cmd_reverb_spectrum(args):
    _, run_scenario, _, _, _ = _imports()
    f0, T = 5000.0, 1.0
    if args.boundary == "bottom":
        rev = dict(mu_db=-27.0)
    else:
        rev = dict(model="chapman_harris", scatterer_motion=dict(wind_speed=float(args.wind)))
    if args.geometry == "bistatic" or args.boundary == "surface":
        rev.update(geometry="bistatic", n_elevation=9)
    cfg = dict(name="doppler_study_reverb", sonar="popoto_long_range", fidelity="image_source",
               sonar_overrides=dict(transmit=dict(waveform=dict(type="cw", f_start=f0, f_stop=f0, duration=T,
                                                                window_alpha=0.5)),
                                    receive=dict(kind="baseband", sample_rate=8000.0, window=dict(range=[1500.0, 3500.0]))),
               environment=dict(base="coastal_12c_sand", water_depth_m=100.0, wind_speed_m_s=float(args.wind),
                                absorption="ainslie_mccolm", water_depth_known=False),
               propagation=dict(surface_z=0.0, max_order=1),
               platform=dict(position=[0, 0, 10.0], rpy=[0, 0, 0], velocity=[float(args.speed), 0, 0]),
               pings=dict(count=1, interval=10.0), scene=[], reverberation=[rev], receiver=dict(adc=False), seed=3)
    rec = run_scenario(cfg).records[0]
    x, fs, fc, c, _ = baseband(rec)
    rows = []
    for deg in np.arange(-80.0, 80.0 + 1e-9, 10.0):
        y = beam(rec, x, fs, fc, np.radians(deg))
        seg = y[int(T * fs):int(T * fs) + int(2.0 * fs)]
        spec = np.abs(np.fft.fftshift(np.fft.fft(seg * np.hanning(len(seg)), 8 * len(seg)))) ** 2
        ff = np.fft.fftshift(np.fft.fftfreq(8 * len(seg), 1.0 / fs)) + fc - f0
        sel = np.abs(ff) < 60.0
        p, fsel = spec[sel] / spec[sel].max(), ff[sel]
        above = fsel[p > 0.1]
        rows.append(dict(bearing_deg=float(deg), expected_hz=float(2.0 * args.speed * f0 / c * np.sin(np.radians(deg))),
                         peak_hz=float(fsel[np.argmax(p)]), minus10_db_from_hz=float(above.min()), minus10_db_to_hz=float(above.max())))
        print(" ".join(f"{k}={v:.2f}" for k, v in rows[-1].items()), flush=True)
    out = dict(speed=args.speed, boundary=args.boundary, geometry=args.geometry, wind=args.wind, carrier=f0, pulse_s=T,
               full_extent_hz=2.0 * args.speed * f0 / c, beams=rows)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=1)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog=__doc__.split("\n\n", 1)[1])
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("reverb-spectrum", help="the reverberation's Doppler spectrum per beam for a moving platform")
    r.add_argument("--speed", type=float, default=3.0, help="platform speed along the array, m/s (3)")
    r.add_argument("--boundary", choices=("bottom", "surface"), default="bottom")
    r.add_argument("--geometry", choices=("monostatic", "bistatic"), default="monostatic")
    r.add_argument("--wind", type=float, default=8.0, help="wind speed, m/s (8; the surface's heave)")
    r.add_argument("--json", help="write the result here")
    m = sub.add_parser("margin", help="each target's margin, zero-Doppler and after a Doppler bank")
    m.add_argument("scenario")
    m.add_argument("--waveform", choices=("scene", "cw", "cw_hann"), default="scene",
                   help="the scene's own; or a CW at the band centre, same duration and energy (Tukey 0.05, or Hann)")
    m.add_argument("--target", nargs="*", help="scene labels (default: every labelled item)")
    m.add_argument("--radial", nargs="*", help="replace the target's velocity: this speed straight away from the platform over ground, m/s (negative: closing)")
    m.add_argument("--platform-speed", type=float, help="platform speed along +x (the array axis), m/s")
    m.add_argument("--pings", type=int, default=4, help="background pings, each a new draw (4)")
    m.add_argument("--window", type=float, default=0.5, help="background window, +- this many pulse lengths (CW; 0.5)")
    m.add_argument("--vmax", type=float, default=4.0, help="the bank's largest range rate, m/s (4)")
    m.add_argument("--bearing", type=float, help="the target's beam, degrees from broadside (default: searched)")
    m.add_argument("--segments", type=int, nargs="*", help="a train's segments to measure, by index (default: every tone and sweep)")
    m.add_argument("--comb", choices=("whole", "tones"), default="whole",
                   help="a comb segment: correlate with it as sent (whole), or with each tone and sum the powers (tones)")
    m.add_argument("--json", help="write the results here")
    args = p.parse_args(argv)
    args.waveform = getattr(args, "waveform", "scene")
    args.platform_speed = getattr(args, "platform_speed", None)
    global MAX_RANGE_RATE
    MAX_RANGE_RATE = float(getattr(args, "vmax", 5.0)) + 1.0
    return {"reverb-spectrum": cmd_reverb_spectrum, "margin": cmd_margin}[args.cmd](args)


if __name__ == "__main__":
    main()
    sys.exit(0)
