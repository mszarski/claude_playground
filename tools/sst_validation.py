#!/usr/bin/env python3
"""Validation of the simulator against SST's own document, APL-UW TR 0702 (R. P. Goddard,
"The Sonar Simulation Toolset, Release 4.6: Science, Mathematics, and Algorithms", 2008).

What this does, end to end (scenario -> records -> processing), and repeatably:

  pursuit   the "Pursuit" example of TR 0702 Sec. 3 / Fig. 2 (configs/scenarios/sst_pursuit.yaml)
  asiaex    the ASIAEX forward-scatter example of Sec. 12.2-12.6 / Figs. 3-5 (sst_asiaex.yaml)
  checks    closed-form checks through the whole simulator (sst_closed_form.yaml)
  matrix    one row per checkable statement of TR 0702: what it says, what of ours it
            exercises, which test checks it, the measured agreement, or why it cannot be checked

Usage:
  tools/sst_validation.py [--out matrix.md] [--json numbers.json] [--figures docs/figures]
                          [--only pursuit,asiaex,checks] [--no-run]

TR 0702 gives no complete numerical test case (docs/PHYSICS_NOTES.md Sec. 9). For the two
worked examples the report prints figures and a few parameters; the rest are in the scripts
of the SST distribution, which we do not have. The scenario files say which values are ours
(`unconfirmed`). Nothing here is a numerical reproduction of Fig. 2 or Figs. 3-5: what is
checked is each feature the text describes, as a number measured on our output against a
closed form evaluated for OUR geometry.

Every TR 0702 formula used below is transcribed next to the function that uses it, with its
equation number, from the extracted text of the report. No value from any other source.
"""

from __future__ import annotations

import argparse
import functools
import json
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

import gpusonar as gs  # noqa: E402
from gpusonar import acoustics  # noqa: E402
from gpusonar.config import load_scenario_config  # noqa: E402
from gpusonar.sim import run_scenario  # noqa: E402

core = gs.core


# =================================================================================================
# TR 0702 closed forms, as printed
# =================================================================================================

def doppler_factor_eq63(v_receiver, s_receiver, v_source, s_source) -> float:
    """TR 0702 eq. (63) (= eq. (71), the platform factor of reverberation):

        Gamma_p = (1 - V_R . S_Rp) / (1 + V_S . S_Sp)

    S_Rp is the slowness vector at the receiver (magnitude 1/c, along the direction of
    propagation there); S_Sp the slowness vector at the source (magnitude 1/c, OPPOSITE to
    the direction of propagation there): Sec. 5.1. Received frequency = Gamma x transmitted."""
    return float((1.0 - np.dot(v_receiver, s_receiver)) / (1.0 + np.dot(v_source, s_source)))


def doppler_factor_eq68(v_receiver, s_rp, v_source, s_sq, v_target, s_tq, s_tp) -> float:
    """TR 0702 eq. (68), echo from a moving highlight:

        Gamma_pTq = [(1 - V_R . S_Rp) / (1 + V_S . S_Sq)] x [(1 - V_T . S_Tq) / (1 + V_T . S_Tp)]

    q is the transmit leg (source -> target), p the receive leg (target -> receiver). S_Tq is
    the slowness at the target as the RECEIVING end of q (along propagation); S_Tp the slowness
    at the target as the SOURCE end of p (opposite to propagation)."""
    return doppler_factor_eq63(v_receiver, s_rp, v_source, s_sq) * float(
        (1.0 - np.dot(v_target, s_tq)) / (1.0 + np.dot(v_target, s_tp)))


def slowness_pair(source, receiver, c: float):
    """(S_S, S_R) of the straight path from `source` to `receiver`: TR 0702 eq. (41) for the
    direct path, S_R = R / (c R), S_S = -R / (c R), R the vector from source to receiver."""
    r = np.asarray(receiver, dtype=float) - np.asarray(source, dtype=float)
    u = r / np.linalg.norm(r)
    return -u / c, u / c


def image_vector_eq36(r_source, r_receiver, depth: float, n_surface: int, n_bottom: int, bottom_first: bool):
    """TR 0702 eq. (36), the vector from the source to the image of the receiver:

        R_p = r_R - r_S - 2 |S_p - B_p| z_R z^  +/-  2 B_p D z^

    "+ if the reflection closest to the source is from the bottom, - if it is from the
    surface"; z is down, z = 0 the surface, D the depth. T_p = |R_p| / c is eq. (37)."""
    r_s, r_r = np.asarray(r_source, dtype=float), np.asarray(r_receiver, dtype=float)
    out = r_r - r_s
    out[2] += -2.0 * abs(n_surface - n_bottom) * r_r[2] + (2.0 if bottom_first else -2.0) * n_bottom * depth
    return out


def unfolded_vertical(z_source: float, z_receiver: float, depth: float, sequence: str) -> float:
    """Our own derivation, for checking eq. (36): the vertical distance travelled along a
    path that bounces from the boundaries in `sequence` ("s" surface, "b" bottom, in order
    from the source), by following the ray leg by leg."""
    z, total = z_source, 0.0
    for b in sequence:
        target = 0.0 if b == "s" else depth
        total += abs(target - z)
        z = target
    return total + abs(z_receiver - z)


def bounce_fraction_eq40(z_source: float, depth: float, vertical: float, b: int, surface_first: bool) -> float:
    """TR 0702 eq. (40): the fraction w_pb of the horizontal distance at which bounce b
    (0 = nearest the source) happens: (z_S + b D) / |Z_p| if bounce 0 is a surface bounce,
    (D - z_S + b D) / |Z_p| if it is a bottom bounce."""
    return ((z_source if surface_first else depth - z_source) + b * depth) / abs(vertical)


def thorp_eq42_db_per_kyd(f_khz):
    """TR 0702 eq. (42): alpha [dB/kyd] = 0.1 f^2 / (1 + f^2) + 40 f^2 / (4100 + f^2), f in kHz."""
    f2 = np.asarray(f_khz, dtype=float) ** 2
    return 0.1 * f2 / (1.0 + f2) + 40.0 * f2 / (4100.0 + f2)


# =================================================================================================
# Small signal tools
# =================================================================================================

def to_pressure(record, x=None):
    """Samples in micro-pascal: the record's volts divided by the receive chain's gain
    (the scenario's sensitivity and preamp gain: amplitude at 1 m / source pressure)."""
    wf = record.transmit.waveform
    x = np.asarray(record.samples.array if x is None else x)
    return x * (10.0 ** (wf.source_level_db / 20.0) / wf.amplitude)


def spectrogram(x, fs: float, t0: float, segment: float, hop: float, f_lo: float, f_hi: float, df: float = 1.0):
    """Power spectral density of short Hann-windowed segments of a real series (what the
    Matlab scripts behind TR 0702 Fig. 2 did: "sliced that signal into short segments,
    computed the power spectral density for each segment").

    Returns (segment centre times [n], frequencies [m], PSD [n, m], one-sided, in the units of
    x squared per Hz)."""
    x = np.asarray(x, dtype=float)
    n_seg, n_hop = int(round(segment * fs)), max(int(round(hop * fs)), 1)
    n_fft = int(round(fs / df))
    w = np.hanning(n_seg)
    f = np.fft.rfftfreq(n_fft, 1.0 / fs)
    keep = (f >= f_lo) & (f <= f_hi)
    starts = np.arange(0, x.shape[-1] - n_seg + 1, n_hop)
    out = np.empty((len(starts), int(keep.sum())))
    for k, i in enumerate(starts):
        out[k] = 2.0 * np.abs(np.fft.rfft(x[i:i + n_seg] * w, n_fft)[keep]) ** 2 / (fs * np.sum(w * w))
    return t0 + (starts + 0.5 * n_seg) / fs, f[keep], out


def envelope(x, fs: float, carrier: float, bandwidth: float):
    """Complex envelope of a real series about `carrier`: mixed down and low-passed to
    +-bandwidth / 2 with a raised-cosine edge (TR 0702 eq. (5) without the sqrt(2): the
    magnitude is the amplitude of the tone)."""
    x = np.asarray(x, dtype=float)
    n = x.shape[-1]
    pad = int(np.ceil(8.0 * fs / bandwidth))                  # zeros at the end: the filter is circular
    t = np.arange(n) / fs
    spectrum = np.fft.fft(2.0 * x * np.exp(-2j * np.pi * carrier * t), n + pad, axis=-1)
    f = np.abs(np.fft.fftfreq(n + pad, 1.0 / fs))
    edge = 0.25 * bandwidth
    gain = np.where(f <= 0.5 * bandwidth - edge, 1.0,
                    np.where(f >= 0.5 * bandwidth + edge, 0.0,
                             0.5 * (1.0 + np.cos(np.pi * (f - 0.5 * bandwidth + edge) / (2.0 * edge)))))
    return np.fft.ifft(spectrum * gain, axis=-1)[..., :n]


def tukey(u, duration: float, alpha: float):
    """The simulator's pulse envelope at emission time u: 1 over [0, T], with raised-cosine
    edges of alpha T / 2 each."""
    x = np.asarray(u, dtype=float) / duration
    inside = (x >= 0.0) & (x <= 1.0)
    edge = 0.5 * alpha
    if edge <= 0.0:
        return inside.astype(float)
    env = np.where(x < edge, 0.5 * (1.0 - np.cos(np.pi * x / edge)),
                   np.where(x > 1.0 - edge, 0.5 * (1.0 - np.cos(np.pi * (1.0 - x) / edge)), 1.0))
    return np.where(inside, env, 0.0)


def db10(x):
    return 10.0 * np.log10(np.maximum(np.asarray(x, dtype=float), 1e-300))


def samples_of(result, channel=None):
    """[pings, channels, n] (or [pings, n] for one channel) of a scenario result."""
    x = np.stack([np.asarray(r.samples.array) for r in result.records])
    return x if channel is None else x[:, channel]


def _quiet_run(name, overrides=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return run_scenario(name, overrides or {})


# =================================================================================================
# 1. The "Pursuit" example (TR 0702 Sec. 3, Fig. 2)
# =================================================================================================

PURSUIT = "sst_pursuit"
NO_NOISE = {"receiver": {"ambient_noise": None}}


def pursuit_geometry(cfg: dict | None = None) -> dict:
    """The numbers of the scenario file the closed forms need (all OUR choices)."""
    cfg = cfg or load_scenario_config(PURSUIT)
    wf = cfg["sonar_overrides"]["transmit"]["waveform"]
    target = next(item for item in cfg["scene"] if item["type"] == "target")
    return dict(
        c=float(cfg["environment"]["sound_speed"]), f0=float(wf["f_start"]), pulse=float(wf["duration"]),
        alpha=float(wf.get("window_alpha", 0.0)),
        z=float(cfg["platform"]["position"][2]), v=np.asarray(cfg["platform"]["velocity"], dtype=float),
        position=np.asarray(cfg["platform"]["position"], dtype=float),
        depth=float(cfg["propagation"]["bottom_z"]), bottom_coefficient=float(cfg["propagation"]["bottom_coefficient"]),
        target_origin=np.asarray(target["origin"], dtype=float), target_v=np.asarray(target["velocity"], dtype=float),
        highlights=[(np.asarray(h["position"], dtype=float), float(h["target_strength_db"])) for h in target["highlights"]],
        window=tuple(float(r) for r in cfg["sonar_overrides"]["receive"]["window"]["range"]),
        temperature=float(cfg["environment"].get("temperature", 10.0)), salinity=float(cfg["environment"].get("salinity", 35.0)),
        absorption=cfg.get("absorption", "francois_garrison"))


def hook_frequency(t, g: dict, azimuth: float = 0.0):
    """Received frequency of surface reverberation by the direct path at two-way time t
    from the azimuth `azimuth` off the bow: eq. (71) with both rays along the unit vector u
    from the sonar to the surface patch, S_R = -u / c (the echo travels along -u at the
    receiver) and S_S = -u / c:  Gamma = (1 + V . u / c) / (1 - V . u / c),
    V . u = v cos(elevation) cos(azimuth), sin(elevation) = depth / (c t / 2).
    To first order this is the familiar f (1 + 2 v cos(angle) / c)."""
    t = np.asarray(t, dtype=float)
    slant = 0.5 * g["c"] * t
    cos_el = np.sqrt(np.clip(1.0 - (g["z"] / np.maximum(slant, 1e-9)) ** 2, 0.0, 1.0))
    beta = np.linalg.norm(g["v"]) * cos_el * np.cos(azimuth) / g["c"]
    return np.where(slant >= g["z"], g["f0"] * (1.0 + beta) / (1.0 - beta), np.nan)


def hook_first_order(t, g: dict, azimuth: float = 0.0):
    """The same to first order in v / c, f (1 + 2 v cos(angle) / c). The reverberation
    generator used this until its components were keyed by Doppler; it now uses the factor
    of eq. (71) itself (sim/reverb_bistatic.doppler_factor), so `hook_frequency` is its
    closed form and this one is kept to state the difference."""
    t = np.asarray(t, dtype=float)
    slant = 0.5 * g["c"] * t
    cos_el = np.sqrt(np.clip(1.0 - (g["z"] / np.maximum(slant, 1e-9)) ** 2, 0.0, 1.0))
    return np.where(slant >= g["z"], g["f0"] * (1.0 + 2.0 * np.linalg.norm(g["v"]) * cos_el * np.cos(azimuth) / g["c"]), np.nan)


def bounce_streak(g: dict, r=None):
    """The surface return with ONE leg direct and the other reflected once from the bottom,
    from dead ahead, against the horizontal range r of the surface patch: two-way time,
    the frequency eq. (71) gives it, and the frequency of a return whose two legs were both
    the direct ray of the same patch (what a generator that knows one Doppler per arrival
    direction would give the half of it that arrives by the direct ray).

    Direct leg: slant sqrt(r^2 + z^2). Bottom-reflected leg: the image of the sonar in the
    bottom is at depth 2 D - z, slant sqrt(r^2 + (2 D - z)^2). Each ray leaves the sonar at
    cos(angle to the velocity) = r / slant."""
    r = np.linspace(0.0, 8000.0, 16001) if r is None else np.asarray(r, dtype=float)
    z, depth, c, v = g["z"], g["depth"], g["c"], float(np.linalg.norm(g["v"]))
    direct, bounce = np.hypot(r, z), np.hypot(r, 2.0 * depth - z)
    t = (direct + bounce) / c
    b_d, b_b = v * r / direct / c, v * r / bounce / c
    # One leg along each ray; eq. (71) with u_p on one and u_q on the other (either way round).
    f_true = g["f0"] * (1.0 + b_d) / (1.0 - b_b)
    f_direct_both = g["f0"] * (1.0 + b_d) / (1.0 - b_d)
    return t, f_true, f_direct_both


def collinear_echo(g: dict, highlight: int):
    """Exact kinematics of the echo of one highlight for sonar and target on one line, all
    velocities along it (the Pursuit geometry): a sample emitted at u from the sonar at
    v u reaches the highlight (at x0 + vt t) at t1 and the sonar at t2:

        c (t1 - u) = x0 + vt t1 - v u        c (t2 - t1) = x0 + vt t1 - v t2

    Returns (tau0, gamma, r_out, r_back): reception time of the sample emitted at u = 0, the
    time scale d u / d t2 (which is eq. (68) exactly for this geometry), and the two path
    lengths for the pulse's first sample."""
    c, v, vt = g["c"], float(g["v"][0]), float(g["target_v"][0])
    x0 = float(g["target_origin"][0] + g["highlights"][highlight][0][0] - g["position"][0])

    def reception(u):
        t1 = (x0 + (c - v) * u) / (c - vt)
        return t1, (x0 + (vt + c) * t1) / (c + v)

    t1, tau0 = reception(0.0)
    gamma = 1.0 / (reception(1.0)[1] - tau0)
    return tau0, gamma, c * t1, c * (tau0 - t1)


def pursuit_echo_model(g: dict, record, t, scale: float = 1.0):
    """The echo of the three highlights by the direct path, in the record's units, from TR
    0702's "simple target echo model", eq. (66) with one eigenray pair and eq. (68):

        y(t) = sum_T  A_T L_p L_q  x( Gamma (t - T_p - T_q) )

    with x(u) = a e(u) cos(2 pi f0 u) the transmitted tone, L = (1 / R) x absorption for each
    leg (eq. (38) without bounces), A_T = 10^(TS / 20). Sonar equation: SL - 2 TL + TS."""
    wf = record.transmit.waveform
    alpha = float(acoustics.absorption(g["f0"], g["temperature"], g["salinity"], 0.0, 8.0, g["absorption"]))   # dB / m
    y = np.zeros_like(np.asarray(t, dtype=float))
    for i, (_, ts) in enumerate(g["highlights"]):
        tau0, gamma, r_out, r_back = collinear_echo(g, i)
        u = gamma * scale * (np.asarray(t, dtype=float) - tau0)     # scale: for fitting the frequency
        loss = 10.0 ** (ts / 20.0) / (r_out * r_back) * 10.0 ** (-alpha * (r_out + r_back) / 20.0)
        y += wf.amplitude * loss * tukey(u, g["pulse"], g["alpha"]) * np.cos(2.0 * np.pi * g["f0"] * u)
    return y


@functools.lru_cache(maxsize=None)
def pursuit_runs(reverb_pings: int = 3) -> dict:
    """The scenario as written, and its parts: echoes alone, reverberation alone (several
    pings: each draws new reverberation), reverberation without the bottom."""
    pings = {"pings": {"count": int(reverb_pings)}}
    full = _quiet_run(PURSUIT)
    echo = _quiet_run(PURSUIT, {**NO_NOISE, "reverberation": []})
    reverb = _quiet_run(PURSUIT, {**NO_NOISE, **pings, "scene": []})
    no_bottom = _quiet_run(PURSUIT, {**NO_NOISE, **pings, "scene": [], "propagation": {"max_order": 0}})
    return dict(full=full, echo=echo, reverb=reverb, no_bottom=no_bottom)


def pursuit(reverb_pings: int = 3) -> dict:
    """Measure the features TR 0702 Sec. 3 describes on our Pursuit record. Returns numbers
    (and the spectrograms, under `display`, for the figure)."""
    g = pursuit_geometry()
    runs = pursuit_runs(reverb_pings)
    record = runs["full"].records[0]
    fs, t0 = record.samples.sample_rate, record.samples.start_time
    c, f0, pulse = g["c"], g["f0"], g["pulse"]
    f_max = float(hook_frequency(1e6, g))                    # the main-lobe ridge at long range
    segment, hop = pulse, 0.25 * pulse
    f_lo, f_hi = 9600.0, 10200.0                             # the frequency span of Fig. 2

    def psd(result, ping=0, channel=0):                      # "only one channel is shown"
        return spectrogram(to_pressure(record, samples_of(result)[ping, channel]), fs, t0, segment, hop, f_lo, f_hi)

    tc, f, s_full = psd(runs["full"])
    s_echo = psd(runs["echo"])[2]
    s_rev = np.mean([psd(runs["reverb"], k)[2] for k in range(reverb_pings)], axis=0)
    s_nob = np.mean([psd(runs["no_bottom"], k)[2] for k in range(reverb_pings)], axis=0)
    # Echoes, reverberation and noise add: the first reverberation ping of the reverberation-only
    # run is the full record's reverberation (same seed, same draws), so the rest is the noise.
    rest = samples_of(runs["full"])[0] - samples_of(runs["echo"])[0] - samples_of(runs["reverb"])[0]
    s_noise = spectrogram(to_pressure(record, rest[0]), fs, t0, segment, hop, f_lo, f_hi)[2]
    df = f[1] - f[0]
    out = {"parameters": {"carrier_hz": f0, "vehicle_speed": float(np.linalg.norm(g["v"])), "vehicle_depth": g["z"],
                          "water_depth": g["depth"], "pulse_s": pulse, "target_speed": float(np.linalg.norm(g["target_v"])),
                          "target_range": float(g["target_origin"][0]), "channels": int(record.samples.array.shape[0]),
                          "note": "all chosen by us (unconfirmed); TR 0702 gives none of them"}}

    # -- the reverberation hook -------------------------------------------------------------------
    # A segment centred at tc holds returns of two-way time tc - T/2 on average (the pulse is T long).
    te = tc - 0.5 * pulse
    predicted = hook_frequency(te, g)
    # The generator's closed form: eq. (71) exactly (the keys below keep the name they had
    # when the generator was first order in v / c).
    first_order = hook_frequency(te, g)
    start = 2.0 * g["z"] / c
    ok = (te > start + 0.02) & (tc < t0 + 2.0 * (g["window"][1] - g["window"][0]) / c)
    search = (f > f0 - 40.0) & (f < f_max + 40.0)
    # The hook is the surface return by the direct path: measured on the run without the
    # bottom (the returns by the bottom are the next item), and on the run with it.
    ridge = f[search][np.argmax(s_nob[:, search], axis=1)]
    ridge_all = f[search][np.argmax(s_rev[:, search], axis=1)]
    err = (ridge - first_order)[ok]
    resolution = 1.44 / segment                               # -3 dB width of a Hann window of this length
    # Early on the Doppler rises by more than a resolution cell within one pulse length: the
    # ridge of a segment must lie between the closed form one pulse earlier and now.
    lo = np.nan_to_num(hook_frequency(tc - pulse - 0.5 * segment, g), nan=f0) - 0.5 * resolution
    hi = hook_frequency(tc + 0.5 * segment, g) + 0.5 * resolution
    settled = ok & (te > 4.0 * g["z"] / c)                     # the closed form changes by < 1 cell per pulse from here
    early, late = ok & (tc < 0.7), ok & (tc > 4.0)
    out["hook"] = {
        "surface_onset_s": start,
        "ridge_minus_first_order_hz": {"median": float(np.median(err)), "median_abs": float(np.median(np.abs(err))),
                                       "p90_abs": float(np.percentile(np.abs(err), 90))},
        "settled_ridge_minus_first_order_hz": {"median": float(np.median((ridge - first_order)[settled])),
                                               "max_abs": float(np.abs(ridge - first_order)[settled].max())},
        "fraction_inside_slew_bracket": float(np.mean(((ridge >= lo) & (ridge <= hi))[ok])),
        "with_bottom_paths": {"fraction_inside_slew_bracket": float(np.mean(((ridge_all >= lo) & (ridge_all <= hi))[ok])),
                              "max_abs_settled_hz": float(np.abs(ridge_all - first_order)[settled].max())},
        "resolution_hz": resolution,
        "early_shift_hz": float(np.median(ridge[early]) - f0), "early_closed_form_hz": float(np.median(first_order[early]) - f0),
        "late_shift_hz": float(np.median(ridge[late]) - f0), "late_closed_form_hz": float(np.median(first_order[late]) - f0),
        "main_lobe_ridge_first_order_hz": float(f0 * (1.0 + 2.0 * np.linalg.norm(g["v"]) / c)),
        "main_lobe_ridge_eq71_hz": f_max,
        "first_order_minus_eq71_hz": float(f0 * (1.0 + 2.0 * np.linalg.norm(g["v"]) / c) - f_max),
        "rank_correlation_with_time": float(_spearman(tc[ok & (tc < 2.0)], ridge[ok & (tc < 2.0)])),
        "segments": int(ok.sum()), "pings_averaged": int(reverb_pings)}

    # -- the bottom-bounce streak -----------------------------------------------------------------
    t_streak, f_streak, f_wrong = bounce_streak(g)
    onset = float(t_streak[0])                                # 2 D / c
    low = (f > f0 - 30.0) & (f < f0 + 130.0)                 # where the streak starts: little Doppler
    excess = db10(s_rev[:, low].mean(axis=1) / s_nob[:, low].mean(axis=1))
    run = np.convolve((excess > 5.0).astype(float), np.ones(4), mode="full")[3:]          # this and the next three segments
    after = np.flatnonzero((run >= 4.0) & (tc > start + 0.3))
    first = float(tc[after[0]]) if len(after) else float("nan")
    before = (tc > start + 0.3) & (tc < onset - 0.5 * segment - 0.02)
    # Before the first path by the bottom can arrive, the bottom cannot change the record:
    # the largest change in any 70 Hz band below the hook's ridge (3-segment mean).
    worst, where = 0.0, (float("nan"), float("nan"))
    for a in np.arange(f0 - 30.0, f_max - 20.0, 35.0):
        sel = (f >= a) & (f < a + 70.0)
        ratio = np.convolve(db10(s_rev[:, sel].mean(axis=1) / s_nob[:, sel].mean(axis=1)), np.ones(3) / 3.0, mode="same")
        usable = before & (a + 70.0 < first_order - 20.0)
        if usable.any() and np.abs(ratio[usable]).max() > abs(worst):
            k = np.flatnonzero(usable)[np.argmax(np.abs(ratio[usable]))]
            worst, where = float(ratio[k]), (float(tc[k]), float(a + 35.0))
    track = (tc > onset + 0.15) & (tc < onset + 0.75)

    def on_track(spec, f_track, half=12.0):
        """Mean PSD within +-half Hz of a frequency track, over the segments of `track`."""
        vals = []
        for k in np.flatnonzero(track):
            fk = np.interp(te[k], t_streak, f_track)
            vals.append(spec[k, np.abs(f - fk) <= half].mean())
        return float(np.mean(vals))

    out["bottom_bounce"] = {
        "closed_form_onset_s": onset, "what": "one leg direct, one leg reflected once from the bottom: 2 D / c",
        "measured_onset_s": first, "onset_window_s": [onset - 0.5 * segment, onset + pulse + 0.5 * segment],
        "excess_before_db": float(np.nanmedian(excess[before])), "excess_after_db": float(np.nanmedian(excess[track])),
        "largest_change_before_onset_db": worst, "largest_change_at_s_hz": list(where),
        # The same measure 40 Hz above the track: a control. From dead ahead the pair has its
        # largest Doppler, so no return by the bottom belongs above its track.
        "control_40hz_above_track_db_re_no_bottom": float(db10(on_track(s_rev, f_streak + 40.0) / on_track(s_nob, f_streak + 40.0))),
        # Half of this return arrives by the direct ray and half by the bottom ray; eq. (71) puts
        # both halves at one frequency. Where does the power lie?
        "on_true_track_db_re_no_bottom": float(db10(on_track(s_rev, f_streak) / on_track(s_nob, f_streak))),
        "on_direct_both_ways_track_db_re_no_bottom": float(db10(on_track(s_rev, f_wrong) / on_track(s_nob, f_wrong))),
        "true_track_hz_at_2s": float(np.interp(2.0, t_streak, f_streak)),
        "direct_both_ways_track_hz_at_2s": float(np.interp(2.0, t_streak, f_wrong))}

    # -- the target echo --------------------------------------------------------------------------
    taus = [collinear_echo(g, i) for i in range(len(g["highlights"]))]
    centre = collinear_echo(g, 1)
    s_s, s_r = slowness_pair(g["position"], g["target_origin"], c)          # leg q: sonar -> target
    s_t, s_back = slowness_pair(g["target_origin"], g["position"], c)       # leg p: target -> sonar
    gamma68 = doppler_factor_eq68(g["v"], s_back, g["v"], s_s, g["target_v"], s_r, s_t)
    n = record.samples.n_samples
    t = t0 + np.arange(n) / fs
    y = samples_of(runs["echo"])[0]
    model = pursuit_echo_model(g, record, t)
    first_tau = min(tau[0] for tau in taus)
    # Until the pair with one surface-reflected leg arrives, the echo is the direct pair alone.
    r_direct = float(np.linalg.norm(g["target_origin"] - g["position"]))
    r_surface = float(np.hypot(g["target_origin"][0] - g["position"][0], g["target_origin"][2] + g["z"]))
    clean_until = first_tau + (r_surface - r_direct) / c - 0.002
    clean = (t >= first_tau - 0.005) & (t < clean_until)
    residual = float(np.sqrt(np.mean((y[:, clean] - model[None, clean]) ** 2) / np.mean(model[clean] ** 2)))
    fit = float(np.sum(y[0, clean] * model[clean]) / np.sum(model[clean] ** 2))
    env = np.abs(envelope(y[0], fs, f0 * gamma68, 400.0))
    one = record.transmit.waveform.amplitude * 10.0 ** (g["highlights"][0][1] / 20.0) / (taus[0][2] * taus[0][3])
    arrival = float(t[np.flatnonzero(env > 0.1 * one)[0]])
    # Frequency of the echo: the time scale of the model that fits the record best.
    f_echo, best = float("nan"), np.inf
    for shift in np.arange(-6.0, 6.01, 0.1):
        trial = pursuit_echo_model(g, record, t[clean], scale=1.0 + shift / (f0 * gamma68))
        res = np.mean((y[0, clean] - trial) ** 2)
        if res < best:
            best, f_echo = res, float(f0 * gamma68 + shift)
    k_t = int(np.argmin(np.abs(tc - (centre[0] + 0.5 * pulse))))
    k_f = np.abs(f - f0 * gamma68) <= 5.0
    near = np.abs(tc - tc[k_t]) <= 0.3
    cell_full = float(s_full[k_t, k_f].max())
    cell_reverb = float(s_rev[near][:, k_f].mean())
    band_level = lambda s: float(s[k_t].sum() * df)                         # noqa: E731
    ridge_at_echo = float(np.median(f[search][np.argmax(s_rev[near][:, search], axis=1)]))
    out["target"] = {
        "closed_form_arrival_s": float(first_tau), "measured_arrival_s": arrival,
        "eq68_frequency_hz": float(f0 * gamma68), "exact_kinematics_frequency_hz": float(f0 * centre[1]),
        "measured_frequency_hz": f_echo,
        "echo_vs_eq66_residual": residual, "echo_amplitude_over_sonar_equation": fit,
        "level_error_db": float(20.0 * np.log10(abs(fit))),
        "ridge_hz_at_echo_time": ridge_at_echo, "separation_from_ridge_hz": float(ridge_at_echo - f0 * gamma68),
        "cell_db_above_reverberation": float(db10(cell_full / cell_reverb)),
        "cell_db_above_noise": float(db10(cell_full / s_noise[near][:, k_f].mean())),
        "band_level_echo_minus_reverberation_db": float(db10(band_level(s_echo) / (s_rev[near].sum(axis=1).mean() * df))),
        "band_level_rise_from_the_echo_db": float(db10(band_level(s_full) / max(band_level(s_full) - band_level(s_echo), 1e-300)))}

    # -- does time spreading matter here? -----------------------------------------------------------
    wind = float(load_scenario_config(PURSUIT)["reverberation"][0]["wind_speed"])
    spread = _quiet_run(PURSUIT, {**NO_NOISE, "reverberation": [], "time_spread": {"surface": {"wind_speed": wind}}})
    ys = samples_of(spread)[0]
    later = (t >= clean_until + 0.004) & (t < clean_until + 0.004 + 2.0 * pulse)
    out["time_spreading"] = {
        "what": "the echo run again with `time_spread: {surface: {wind_speed}}` (not in the scenario file)",
        "direct_echo_change": float(np.sqrt(np.mean((ys[:, clean] - y[:, clean]) ** 2) / np.mean(y[:, clean] ** 2))),
        "surface_bounce_echo_db_re_direct": float(db10(np.mean(y[:, later] ** 2) / np.mean(y[:, clean] ** 2))),
        "surface_bounce_echo_change_db": float(db10(np.mean(ys[:, later] ** 2) / np.mean(y[:, later] ** 2)))}

    # -- what dominates ---------------------------------------------------------------------------
    inside = ok & (tc > 0.5)
    rev_band, noise_band = s_rev.sum(axis=1) * df, s_noise.sum(axis=1) * df
    out["levels"] = {
        "ridge_psd_over_noise_psd_db": {"median": float(np.median(db10(s_rev.max(axis=1) / s_noise.mean(axis=1))[inside])),
                                        "p10": float(np.percentile(db10(s_rev.max(axis=1) / s_noise.mean(axis=1))[inside], 10))},
        "band_reverberation_over_noise_db": {"median": float(np.median(db10(rev_band / noise_band)[inside])),
                                             "fraction_above": float(np.mean((rev_band > noise_band)[inside]))},
        "band_hz": [f_lo, f_hi],
        "bottom_paths_share_db": float(db10((s_rev[inside].sum() - s_nob[inside].sum()) / s_nob[inside].sum()))}
    from gpusonar.sim import noise as noise_models
    amb = load_scenario_config(PURSUIT)["receiver"]["ambient_noise"]
    out["noise"] = {"what": "full record minus echoes minus reverberation (the parts add)",
                    "psd_db_re_upa2_per_hz": float(db10(s_noise.mean())),
                    "scenario_spectrum_db": float(noise_models.wenz_spectrum_db(np.array([f0]), float(amb["wind_speed"]), float(amb["shipping"]))[0])}
    out["display"] = dict(t=tc, f=f, full=s_full, reverb=s_rev, echo=s_echo, noise=s_noise, hook=first_order,
                          streak=(t_streak, f_streak, f_wrong), band_full=s_full.sum(axis=1) * df,
                          band_reverb=rev_band, band_echo=s_echo.sum(axis=1) * df, band_noise=noise_band,
                          target=(centre[0] + 0.5 * pulse, f0 * gamma68))
    return out


def _spearman(a, b) -> float:
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def pursuit_doppler_of_cross_pairs(n_elevation: int = 90, bottom: bool = True) -> dict:
    """The mean Doppler shift of the surface reverberation against two-way time from dead
    ahead, for the Pursuit geometry with one bottom reflection allowed and no beam patterns:
    what the simulator's bistatic generator will produce (its components' power against time
    and their one Doppler shift each) against a direct integration over the surface in which
    every pair of paths has the shift of eq. (71).

    This pinned a defect, since fixed: a component used to be (azimuth sector, elevation of
    the RECEIVE ray) with one Doppler shift, the power-weighted mean over everything that
    arrived from there, so returns with a direct ray whose transmit leg went by the bottom
    were given the shift of the (much stronger, earlier) returns by the direct ray both
    ways. Components are now keyed by Doppler bin as well (reverb_bistatic.ComponentSum)."""
    from gpusonar.sim import reverb_bistatic
    from gpusonar.sim.propagation import ImageSourceProvider
    g = pursuit_geometry()
    c, f0, z, depth, v = g["c"], g["f0"], g["z"], g["depth"], float(np.linalg.norm(g["v"]))
    beta = g["bottom_coefficient"] if bottom else 0.0
    provider = ImageSourceProvider(c, surface_z=0.0, bottom_z=depth, bottom_coefficient=beta, max_order=1 if bottom else 0,
                                   engine="numpy")
    times = np.arange(0.5, 3.0, 0.01)
    n_az = 72
    azimuths = (np.arange(n_az) + 0.5) * 2.0 * np.pi / n_az
    mu = 10.0 ** (-27.0 / 10.0)
    found = reverb_bistatic.directional_power(provider, g["position"], g["position"], 0.0, times, azimuths, frequency=f0,
                                              mu_db=-27.0, tx_velocity=g["v"], rx_velocity=g["v"], n_elevation=n_elevation)
    power, doppler = found["power"][0], found["doppler"][0]                 # the sector next to the bow
    model = (power * doppler[:, None]).sum(axis=0) / np.maximum(power.sum(axis=0), 1e-300)
    # Direct integration: rings of the surface in that sector; paths d (direct) and b (one bottom bounce).
    r = np.arange(0.25, 4000.0, 0.5)
    slant = {"d": np.hypot(r, z), "b": np.hypot(r, 2.0 * depth - z)}
    amp2 = {"d": 1.0 / slant["d"] ** 2, "b": beta**2 / slant["b"] ** 2}
    sin_g = {"d": z / slant["d"], "b": (2.0 * depth - z) / slant["b"]}
    cos_v = {k: r / slant[k] * np.cos(azimuths[0]) for k in slant}          # cos(angle between the launch ray and the velocity)
    num, den = np.zeros(len(times)), np.zeros(len(times))
    dt = times[1] - times[0]
    for q in "db":
        for p in "db":
            t = (slant[q] + slant[p]) / c
            w = amp2[q] * amp2[p] * mu * sin_g[q] * sin_g[p] * r
            shift = f0 * ((1.0 + v * cos_v[p] / c) / (1.0 - v * cos_v[q] / c) - 1.0)      # eq. (71), as the generator has it
            k = np.rint((t - times[0]) / dt).astype(int)
            inside = (k >= 0) & (k < len(times))
            num += np.bincount(k[inside], weights=(w * shift)[inside], minlength=len(times))
            den += np.bincount(k[inside], weights=w[inside], minlength=len(times))
    expected = num / np.maximum(den, 1e-300)
    onset = 2.0 * depth / c
    direct_only = (times > 2.0 * z / c + 0.05) & (times < onset - 0.03)
    mixed = (times > onset + 0.05) & (times < 2.0 * (2.0 * depth - z) / c - 0.05)
    return {"times": times, "model_hz": model, "expected_hz": expected,
            "max_error_direct_only_hz": float(np.abs(model - expected)[direct_only].max()),
            "max_error_with_bottom_paths_hz": float(np.abs(model - expected)[mixed].max()),
            "mean_error_with_bottom_paths_hz": float((model - expected)[mixed].mean()),
            "bottom_path_onset_s": onset}


# =================================================================================================
# 2. ASIAEX forward scatter (TR 0702 Sec. 12.2-12.6, Figs. 3-5)
# =================================================================================================

ASIAEX = "sst_asiaex"


def asiaex_geometry(cfg: dict | None = None) -> dict:
    cfg = cfg or load_scenario_config(ASIAEX)
    so = cfg["sonar_overrides"]
    wf = so["transmit"]["waveform"]
    rx = np.asarray(cfg["platform"]["position"], dtype=float)
    tx = rx + np.asarray(so["transmit"]["lever_arm"], dtype=float)
    relay = np.asarray(cfg["scene"][0]["positions"][0], dtype=float)
    return dict(c=float(cfg["environment"]["sound_speed"]), f0=float(wf["f_start"]), pulse=float(wf["duration"]),
                alpha=float(wf.get("window_alpha", 0.0)), rx=rx, tx=tx, relay=relay,
                depth=float(cfg["propagation"]["bottom_z"]), bottom=dict(cfg["propagation"]["bottom"]),
                wind=float(cfg["time_spread"]["surface"]["wind_speed"]),
                bottom_rms_height=float(cfg["time_spread"]["bottom"]["rms_height"]),
                bottom_mss=float(cfg["time_spread"]["bottom"]["mean_square_slope"]),
                taps=int(cfg["time_spread"].get("taps", 6)), interval=float(cfg["pings"]["interval"]),
                temperature=float(cfg["environment"].get("temperature", 10.0)), salinity=float(cfg["environment"].get("salinity", 35.0)),
                absorption=cfg.get("absorption", "francois_garrison"))


ASIAEX_PATHS = {"direct": "", "surface": "s", "bottom": "b", "surface-bottom": "sb", "bottom-surface": "bs"}


def asiaex_eigenrays(g: dict) -> dict:
    """The five eigenrays from the relay (the source of the return leg) to the receiver by
    TR 0702 eqs. (36)-(38) and (40): delay, slant range, grazing angle, loss factor
    |L_p| = (1 / R_p) x absorption x prod |beta| (our Rayleigh bottom coefficient, |beta| = 1
    at the surface), and the travel times before and after each bounce."""
    from gpusonar.sim.propagation import rayleigh_reflection
    src, rcv, depth, c = g["relay"], g["rx"], g["depth"], g["c"]
    alpha = float(acoustics.absorption(g["f0"], g["temperature"], g["salinity"], 0.0, 8.0, g["absorption"]))
    out = {}
    for name, seq in ASIAEX_PATHS.items():
        vec = image_vector_eq36(src, rcv, depth, seq.count("s"), seq.count("b"), seq[:1] == "b")
        slant = float(np.linalg.norm(vec))
        grazing = float(np.arcsin(abs(vec[2]) / slant))
        beta = abs(rayleigh_reflection(grazing, c, **g["bottom"])) ** seq.count("b")
        fractions = [bounce_fraction_eq40(src[2], depth, vec[2], b, seq[0] == "s") for b in range(len(seq))]
        out[name] = dict(delay=slant / c, slant=slant, grazing=grazing, vertical=abs(float(vec[2])),
                         unfolded=unfolded_vertical(src[2], rcv[2], depth, seq),
                         loss=float(beta / slant * 10.0 ** (-alpha * slant / 20.0)),
                         legs=[(w * slant / c, (1.0 - w) * slant / c) for w in fractions])
    return out


# Two bounces that both scatter nearly everything, with equal decay times on the
# surface-bottom path (1.8 ms each): the case in which adding the mean delays into one
# exponential is "too high immediately after the pulse and too low later" (TR 0702 p. 104).
# Not ASIAEX values: a variant of the scenario for that one statement.
BOTH_ROUGH = {"surface": {"rms_height": 0.2, "mean_square_slope": 0.01836, "rms_vertical_speed": 0.19},
              "bottom": {"rms_height": 0.2, "mean_square_slope": 0.0127}}


@functools.lru_cache(maxsize=None)
def asiaex_runs(many: int = 200, seeds: int = 40, doppler_pings: int = 800) -> dict:
    surface_only = {"propagation": {"bottom_z": None}}
    bottom_only = {"propagation": {"surface_z": None}, "pings": {"count": 1}}
    fast = {"pings": {"count": int(doppler_pings), "interval": 0.05}}
    return dict(
        flat=_quiet_run(ASIAEX, {"time_spread": None}),
        spread=_quiet_run(ASIAEX),                                              # 20 pings, as in TR 0702
        convolve=_quiet_run(ASIAEX, {"time_spread": {"multi_bounce": "convolve"}}),
        sum_many=_quiet_run(ASIAEX, {"pings": {"count": int(many)}}),
        convolve_many=_quiet_run(ASIAEX, {"pings": {"count": int(many)}, "time_spread": {"multi_bounce": "convolve"}}),
        surface_many=_quiet_run(ASIAEX, {**surface_only, "pings": {"count": int(many)}}),
        surface_fast=_quiet_run(ASIAEX, {**surface_only, **fast}),
        surface_fast_gaussian=_quiet_run(ASIAEX, {**surface_only, **fast, "time_spread": {"doppler_spectrum": "gaussian"}}),
        rough_sum=_quiet_run(ASIAEX, {"pings": {"count": int(many)}, "time_spread": BOTH_ROUGH}),
        rough_convolve=_quiet_run(ASIAEX, {"pings": {"count": int(many)}, "time_spread": {**BOTH_ROUGH, "multi_bounce": "convolve"}}),
        bottom_seeds=[_quiet_run(ASIAEX, {**bottom_only, "time_spread": {"seed": 1000 + k}}) for k in range(int(seeds))])


def coherence_time(rho, lag_times) -> float:
    """The lag at which a correlation first falls to exp(-1/2) (linear interpolation)."""
    level = np.exp(-0.5)
    below = np.flatnonzero(rho < level)
    if len(below) == 0 or below[0] == 0:
        return float("nan")
    k = below[0]
    return float(lag_times[k - 1] + (rho[k - 1] - level) / (rho[k - 1] - rho[k]) * (lag_times[k] - lag_times[k - 1]))


def asiaex(many: int = 200, seeds: int = 40, doppler_pings: int = 800) -> dict:
    """Measure what TR 0702 Sec. 12.2-12.6 states about the ASIAEX simulation on our records."""
    g = asiaex_geometry()
    rays = asiaex_eigenrays(g)
    runs = asiaex_runs(many, seeds, doppler_pings)
    record = runs["flat"].records[0]
    fs, t0 = record.samples.sample_rate, record.samples.start_time
    c, f0, pulse = g["c"], g["f0"], g["pulse"]
    lead = float(np.linalg.norm(g["relay"] - g["tx"])) / c                 # the 0.1 m transmit leg
    first = lead + rays["direct"]["delay"]
    n = record.samples.n_samples
    ms = (t0 + np.arange(n) / fs - first) * 1e3                            # time after the first arrival, ms
    band = 4000.0

    def env(result):
        return envelope(samples_of(result, 0), fs, f0, band)

    def power(result):
        return np.abs(env(result)) ** 2

    amp1 = record.transmit.waveform.amplitude                             # at 1 m, record units
    relay_gain = 10.0 ** (-20.0 / 20.0) / float(np.linalg.norm(g["relay"] - g["tx"]))     # (1 / 0.1 m) x 0.1 = 1
    ref = (amp1 * relay_gain * rays["direct"]["loss"]) ** 2
    at = lambda lo, hi: (ms >= lo) & (ms < hi)                             # noqa: E731
    p_ms = pulse * 1e3
    arrival = {k: (v["delay"] - rays["direct"]["delay"]) * 1e3 for k, v in rays.items()}
    out = {"parameters": {"frequency_hz": f0, "pulse_ms": p_ms, "separation_m": float(np.hypot(*(g["tx"][:2] - g["rx"][:2]))),
                          "source_depth": float(g["tx"][2]), "receiver_depth": float(g["rx"][2]), "water_depth": g["depth"],
                          "wind": g["wind"], "bottom_rms_height": g["bottom_rms_height"], "bottom_mean_square_slope": g["bottom_mss"],
                          "pings": len(runs["spread"].records),
                          "note": "depths, sound speed, ping interval, sediment and wind laws chosen by us (unconfirmed)"}}

    # -- arrival times and levels against the image closed forms (no time spreading) ---------------
    flat = np.abs(env(runs["flat"]))
    rise = 0.25 * g["alpha"] * p_ms                                        # the taper is at half amplitude here
    table = {}
    for name, ray in rays.items():
        a = arrival[name]
        plateau = float(np.median(flat[0, at(a + 0.3 * p_ms, a + 0.7 * p_ms)]))
        window = np.flatnonzero(at(a - 1.0, a + 1.0))
        k = window[np.argmax(flat[0, window] > 0.5 * plateau)]
        # Linear interpolation of the half-amplitude crossing.
        y0, y1 = flat[0, k - 1], flat[0, k]
        crossing = ms[k - 1] + (0.5 * plateau - y0) / (y1 - y0) * (ms[k] - ms[k - 1])
        table[name] = {"closed_form_ms": float(a), "measured_ms": float(crossing - rise),
                       "error_us": float((crossing - rise - a) * 1e3),
                       "level_db": float(20.0 * np.log10(plateau / (amp1 * relay_gain * ray["loss"]))),
                       "eq36_vertical_m": ray["vertical"], "unfolded_vertical_m": ray["unfolded"],
                       "grazing_deg": float(np.degrees(ray["grazing"]))}
    out["arrivals"] = table
    out["arrival_time_max_error_us"] = float(max(abs(v["error_us"]) for v in table.values()))
    out["arrival_level_max_error_db"] = float(max(abs(v["level_db"]) for v in table.values()))

    # -- without time spreading: crisp, identical ----------------------------------------------------
    p_flat = power(runs["flat"])
    peak = {k: at(a + 0.3 * p_ms, a + 0.7 * p_ms) for k, a in arrival.items()}
    just_after = {k: at(a + p_ms + 0.3, a + p_ms + 0.7) for k, a in arrival.items()}
    late_in_pulse = {k: at(a + 0.5 * p_ms, a + 0.95 * p_ms) for k, a in arrival.items()}   # the first taps are in by then
    strong = p_flat.mean(axis=0) > 1e-6 * ref
    out["no_time_spread"] = {
        "ping_to_ping_relative_std": float((p_flat.std(axis=0) / np.maximum(p_flat.mean(axis=0), 1e-300))[strong].max()),
        "db_after_pulse_re_peak": {k: float(db10(p_flat[:, just_after[k]].mean() / p_flat[:, peak[k]].mean())) for k in ("direct", "surface", "bottom")},
        "rise_time_10_90_ms": {k: _rise_time(ms, np.sqrt(p_flat[0]), arrival[k]) for k in arrival if k != "surface"}}

    # -- with time spreading: tails, ping-to-ping variability ---------------------------------------
    p_spread = power(runs["spread"])

    def variability(p, sel):
        level = db10(p[:, sel].mean(axis=1))
        return float(level.std())

    out["time_spread"] = {
        "ping_to_ping_std_db": {k: variability(p_spread, late_in_pulse[k]) for k in arrival},
        "tail_ping_to_ping_std_db": {k: variability(p_spread, just_after[k]) for k in ("surface", "bottom")},
        "db_after_pulse_re_peak": {k: float(db10(p_spread[:, just_after[k]].mean() / p_spread[:, peak[k]].mean())) for k in ("direct", "surface", "bottom")},
        "peak_mean_db_re_flat": {k: float(db10(p_spread[:, peak[k]].mean() / p_flat[:, peak[k]].mean())) for k in arrival}}

    # -- decay of the tails against the rule of thumb of Sec. 12.4 ----------------------------------
    def rule(ray, mss):                                                    # 2 mss t1 t2 / (t1 + t2): TR 0702's two cases joined
        t1, t2 = ray["legs"][0]
        return 2.0 * mss * t1 * t2 / (t1 + t2)

    mss_surface = acoustics.mean_square_slope(g["wind"])
    p_surface = power(runs["surface_many"]).mean(axis=0)
    a = arrival["surface"]
    t_rule = rule(rays["surface"], mss_surface) * 1e3
    sel = at(a + p_ms + 0.4, a + p_ms + 0.4 + 2.5 * t_rule)
    t_surface = -1.0 / np.polyfit(ms[sel], np.log(p_surface[sel]), 1)[0]
    p_bottom = np.mean([power(r)[0] for r in runs["bottom_seeds"]], axis=0)
    a = arrival["bottom"]
    b_rule = rule(rays["bottom"], g["bottom_mss"]) * 1e3
    sel = at(a + p_ms + 0.4, a + p_ms + 0.4 + 2.0 * b_rule)
    t_bottom = -1.0 / np.polyfit(ms[sel], np.log(p_bottom[sel]), 1)[0]
    k = 2.0 * np.pi * f0 / c
    coherent = lambda h, ray: float(np.exp(-(2.0 * k * h * np.sin(ray["grazing"])) ** 2))       # noqa: E731
    out["tails"] = {
        "surface": {"rule_of_thumb_ms": float(t_rule), "fitted_ms": float(t_surface), "mean_square_slope": float(mss_surface),
                    "coherent_fraction": coherent(acoustics.rms_wave_height(g["wind"]), rays["surface"]),
                    "pings": len(runs["surface_many"].records)},
        "bottom": {"rule_of_thumb_ms": float(b_rule), "fitted_ms": float(t_bottom), "mean_square_slope": g["bottom_mss"],
                   "coherent_fraction": coherent(g["bottom_rms_height"], rays["bottom"]), "realisations": len(runs["bottom_seeds"])}}

    # -- Doppler spread of the surface path -------------------------------------------------------------
    v_rms = acoustics.rms_surface_vertical_speed(g["wind"])
    sin_g = np.sin(rays["surface"]["grazing"])
    width = 2.0 * f0 * v_rms * sin_g / c                                   # Hz: the Doppler of a mirror moving at v_rms
    a = arrival["surface"]
    tail = np.flatnonzero(at(a + p_ms + 0.3, a + p_ms + 0.3 + 2.0 * t_rule))[::8]
    spreads = {}
    for name, key in (("lorentzian", "surface_fast"), ("gaussian", "surface_fast_gaussian")):
        z = env(runs[key])[:, tail]
        z = z - z.mean(axis=0)
        dt = float(runs[key].times[1] - runs[key].times[0])
        lags = np.arange(0, min(len(z) // 4, int(8.0 / dt)))
        rho = np.array([np.mean(z[lag:] * np.conj(z[:len(z) - lag])) for lag in lags]).real
        t_e = coherence_time(rho / rho[0], lags * dt)
        if name == "gaussian":
            # Coherence exp(-t^2 / 2 sigma_t^2), sigma_t = 1 / (2 pi sigma_nu): TR 0702 eq. (96).
            gamma2 = (2.0 * k * acoustics.rms_wave_height(g["wind"]) * sin_g) ** 2
            closed = width / np.sqrt(1.0 - np.exp(-gamma2))
            measured = 1.0 / (2.0 * np.pi * t_e)
        else:
            # AR(1): coherence exp(-2 pi B |t|), down to exp(-1/2) at t = 1 / (4 pi B).
            closed = width
            measured = 1.0 / (4.0 * np.pi * t_e)
        spreads[name] = {"measured_hz": float(measured), "closed_form_hz": float(closed), "coherence_time_s": t_e,
                         "pings": len(z), "interval_s": dt}
    out["surface_doppler_spread"] = {**spreads, "tr0702_bound_hz": 10.0, "rms_vertical_speed": float(v_rms),
                                     "p101_rule_hz": float(f0 * v_rms / (c * sin_g))}

    # -- two bounces: `sum` against `convolve` ----------------------------------------------------------
    p_sum, p_conv = power(runs["sum_many"]).mean(axis=0), power(runs["convolve_many"]).mean(axis=0)
    a, nxt = arrival["surface-bottom"], arrival["bottom-surface"]
    early, late = at(a + p_ms + 0.3, a + p_ms + 1.8), at(nxt - 3.0, nxt - 0.3)
    after_last = at(nxt + p_ms + 0.3, nxt + p_ms + 1.8)
    sb = rays["surface-bottom"]
    c_s = coherent(acoustics.rms_wave_height(g["wind"]), sb)
    c_b = coherent(g["bottom_rms_height"], sb)
    (s1, s2), (b1, b2) = sb["legs"]
    t_s, t_b = 2.0 * mss_surface * s1 * s2 / (s1 + s2) * 1e3, 2.0 * g["bottom_mss"] * b1 * b2 / (b1 + b2) * 1e3
    tail_span = at(a + p_ms + 0.3, nxt - 0.3)
    centroid = lambda p: float(np.sum(p[tail_span] * (ms[tail_span] - a - p_ms)) / np.sum(p[tail_span]))      # noqa: E731
    out["two_bounces"] = {
        "path": "surface-bottom", "coherent_fraction_surface": c_s, "coherent_fraction_bottom": c_b,
        "decay_surface_ms": float(t_s), "decay_bottom_ms": float(t_b),
        "mean_delay_sum_ms": float(t_s + t_b),
        "mean_delay_convolve_ms": float(((1.0 - c_s) * t_s + (1.0 - c_b) * t_b) / (1.0 - c_s * c_b)),
        "early_window_ms_after_pulse": [0.3, 1.8], "late_window_ms_before_next_arrival": [3.0, 0.3],
        "sum_over_convolve_early_db": float(db10(p_sum[early].mean() / p_conv[early].mean())),
        "sum_over_convolve_late_db": float(db10(p_sum[late].mean() / p_conv[late].mean())),
        "sum_over_convolve_after_last_arrival_db": float(db10(p_sum[after_last].mean() / p_conv[after_last].mean())),
        "tail_centroid_sum_ms": centroid(p_sum), "tail_centroid_convolve_ms": centroid(p_conv),
        "tail_energy_sum_over_convolve_db": float(db10(p_sum[tail_span].sum() / p_conv[tail_span].sum())),
        "peak_sum_db_re_direct": float(db10(p_sum[peak["surface-bottom"]].mean() / ref)),
        "peak_convolve_db_re_direct": float(db10(p_conv[peak["surface-bottom"]].mean() / ref)),
        "pings": len(runs["sum_many"].records)}
    r_sum, r_conv = power(runs["rough_sum"]).mean(axis=0), power(runs["rough_convolve"]).mean(axis=0)
    onset, mid = at(a + 0.3, a + 0.5 * p_ms), at(a + p_ms + 2.0, a + p_ms + 5.0)
    out["two_bounces"]["sum_over_convolve_first_half_of_pulse_db"] = float(db10(p_sum[onset].mean() / p_conv[onset].mean()))
    out["two_rough_bounces"] = {
        "what": "variant: both boundaries rms height 0.2 m, decay times equal on the surface-bottom path (not ASIAEX values)",
        "sum_over_convolve_first_half_of_pulse_db": float(db10(r_sum[onset].mean() / r_conv[onset].mean())),
        "sum_over_convolve_2_to_5_ms_after_pulse_db": float(db10(r_sum[mid].mean() / r_conv[mid].mean())),
        "tail_energy_sum_over_convolve_db": float(db10(r_sum[tail_span].sum() / r_conv[tail_span].sum()))}
    out["display"] = dict(ms=ms, ref=ref, flat=p_flat, spread=p_spread, convolve=power(runs["convolve"]),
                          sum_mean=p_sum, convolve_mean=p_conv, arrival=arrival)
    return out


def _rise_time(ms, amplitude, arrival: float) -> float:
    """10 % to 90 % rise time (ms) of the pulse that starts at `arrival`."""
    sel = np.flatnonzero((ms >= arrival - 1.0) & (ms < arrival + 2.0))
    top = float(np.median(amplitude[(ms >= arrival + 1.0) & (ms < arrival + 2.0)]))
    base = float(np.median(amplitude[(ms >= arrival - 1.0) & (ms < arrival - 0.6)]))
    a = amplitude[sel]
    lo = sel[np.argmax(a > base + 0.1 * (top - base))]
    hi = sel[np.argmax(a > base + 0.9 * (top - base))]
    return float(ms[hi] - ms[lo])


# =================================================================================================
# 3. Closed forms through the whole simulator
# =================================================================================================

ECHO = "sst_echo_level"
SEDIMENT = "sst_sediment_reverb"


def _compressed(result):
    """Records made baseband and pulse compressed (what processing does first): the samples
    [pings, channels, n] and the time of sample 0 and the rate."""
    from gpusonar import recon
    records = recon.prepare([r.copy() for r in result.records])
    s = records[0].samples
    return np.stack([np.asarray(r.samples.array) for r in records]), s.start_time, s.sample_rate


def _peak_near(z, t0: float, fs: float, t: float, half: float):
    """Largest magnitude of a compressed series within +-half s of t, with its time
    (parabolic interpolation between samples)."""
    k0, k1 = int(np.floor((t - half - t0) * fs)), int(np.ceil((t + half - t0) * fs))
    a = np.abs(z[k0:k1 + 1])
    i = int(np.argmax(a))
    if 0 < i < len(a) - 1:
        y0, y1, y2 = a[i - 1], a[i], a[i + 1]
        d = 0.5 * (y0 - y2) / (y0 - 2.0 * y1 + y2)
        return float(y1 - 0.25 * (y0 - y2) * d), float(t0 + (k0 + i + d) / fs)
    return float(a[i]), float(t0 + (k0 + i) / fs)


def check_echo_level() -> dict:
    """Echo of a highlight of known strength, scenario -> record -> pulse compression, against
    SL - 2 TL + TS in free field, and against the six groups of eigenray pairs of the image
    model (TR 0702 eqs. 36-38 for each leg, eq. 66 for the pairs)."""
    cfg = load_scenario_config(ECHO)
    c = float(cfg["environment"]["sound_speed"])
    wf = cfg["sonar_overrides"]["transmit"]["waveform"]
    fc = 0.5 * (float(wf["f_start"]) + float(wf["f_stop"]))
    sonar = np.asarray(cfg["platform"]["position"], dtype=float)
    target = np.asarray(cfg["scene"][0]["origin"], dtype=float)
    ts = float(cfg["scene"][0]["highlights"][0]["target_strength_db"])
    depth, beta_b = float(cfg["propagation"]["bottom_z"]), float(cfg["propagation"]["bottom_coefficient"])
    alpha = float(acoustics.absorption(fc, cfg["environment"]["temperature"], cfg["environment"]["salinity"], 0.0, 8.0,
                                       cfg["absorption"]))
    legs = {}
    for name, (n_s, n_b, coefficient) in {"d": (0, 0, 1.0), "s": (1, 0, -1.0), "b": (0, 1, beta_b)}.items():
        slant = float(np.linalg.norm(image_vector_eq36(sonar, target, depth, n_s, n_b, n_b > 0)))
        legs[name] = (slant, coefficient / slant * 10.0 ** (-alpha * slant / 20.0))             # eq. (38)
    out = {}
    for tier in ("free_field", "image_source"):
        result = _quiet_run(ECHO, {"fidelity": tier, **({"propagation": None} if tier == "free_field" else {})})
        z, t0, fs = _compressed(result)
        a1 = result.records[0].transmit.waveform.amplitude
        sl = result.records[0].transmit.waveform.source_level_db
        groups = [("d", "d")] if tier == "free_field" else [("d", "d"), ("d", "s"), ("s", "s"), ("d", "b"), ("s", "b"), ("b", "b")]
        rows = {}
        for q, p in groups:
            (rq, lq), (rp, lp) = legs[q], legs[p]
            expected = a1 * 10.0 ** (ts / 20.0) * lq * lp * (1.0 if q == p else 2.0)          # q-p and p-q arrive together
            peak, t_peak = _peak_near(z[0, 0], t0, fs, (rq + rp) / c, 3.0 / fs)
            rows[q + p] = {"level_error_db": float(20.0 * np.log10(peak / abs(expected))),
                           "time_error_us": float((t_peak - (rq + rp) / c) * 1e6),
                           "echo_level_db_re_upa": float(sl + 20.0 * np.log10(abs(expected) / a1))}
        out[tier] = {"pairs": rows, "max_level_error_db": float(max(abs(r["level_error_db"]) for r in rows.values())),
                     "max_time_error_us": float(max(abs(r["time_error_us"]) for r in rows.values()))}
    r = legs["d"][0]
    out["sonar_equation"] = {"source_level_db": float(sl), "target_strength_db": ts, "one_way_range_m": r,
                             "transmission_loss_db": float(20.0 * np.log10(r) + alpha * r), "sample_interval_us": 1e6 / fs}
    return out


def check_sampled_doppler(v_radial: float = -6.0) -> dict:
    """A highlight target moving along the line of sight, a sampled (m-sequence) pulse, and
    the Doppler bank of the processing chain: the range rate it finds, and the time scale of
    TR 0702 eq. (68) for that motion against the exact two-way factor (c - v) / (c + v)."""
    from gpusonar.processing import doppler_bank, doppler_scale
    cfg = load_scenario_config(ECHO)
    c = float(cfg["environment"]["sound_speed"])
    sonar = np.asarray(cfg["platform"]["position"], dtype=float)
    target = np.asarray(cfg["scene"][0]["origin"], dtype=float)
    u = (target - sonar) / np.linalg.norm(target - sonar)
    scene = [dict(cfg["scene"][0], velocity=[float(x) for x in v_radial * u])]
    wave = {"type": "sampled", "source": "mseq", "degree": 9, "duration": 0.25, "f_start": 6000.0, "f_stop": 10000.0,
            "source_level_db": 200.0}
    over = {"fidelity": "free_field", "propagation": None, "scene": scene, "absorption": "none",
            "sonar_overrides": {"transmit": {"waveform": wave}, "receive": {"window": {"range": [900.0, 1100.0]}}},
            "platform": {"position": [0.0, 0.0, 150.0]}}
    result = _quiet_run(ECHO, over)
    record = result.records[0]
    rates = np.arange(-10.0, 10.5, 1.0)
    best, v_hat, peaks = doppler_bank(record, rates)
    fine = np.arange(v_hat - 1.0, v_hat + 1.01, 0.1)
    _, v_fine, _ = doppler_bank(record, fine)
    s_s, s_r = slowness_pair(sonar, target, c)
    s_t, s_back = slowness_pair(target, sonar, c)
    v_t = v_radial * u
    gamma68 = doppler_factor_eq68(np.zeros(3), s_back, np.zeros(3), s_s, v_t, s_r, s_t)
    r0 = float(np.linalg.norm(target - sonar))
    a1 = record.transmit.waveform.amplitude
    ts = float(cfg["scene"][0]["highlights"][0]["target_strength_db"])
    return {"true_range_rate": v_radial, "bank_range_rate": float(v_hat), "bank_range_rate_fine": float(v_fine),
            "eq68_time_scale": gamma68, "exact_time_scale": float(doppler_scale(v_radial, c)),
            "peak_without_doppler_over_best": float(peaks[list(rates).index(0.0)] / peaks.max()),
            "peak_over_sonar_equation": float(peaks.max() / (a1 * 10.0 ** (ts / 20.0) / r0**2)),
            "synthesis": json.loads(record.metadata.extra_json or "{}").get("gpusonar", {}).get("echo_synthesis")}


def _sediment_prediction(cfg, t, pulse: float):
    """Mean intensity (unit source amplitude at 1 m) of bottom reverberation at two-way time t:
    sigma(grazing) pi c T / R^3 with R = c (t - T / 2) / 2 (header of sst_sediment_reverb.yaml)."""
    from gpusonar.sim import bottom
    item = cfg["reverberation"][0]
    c = float(cfg["environment"]["sound_speed"])
    wf = cfg["sonar_overrides"]["transmit"]["waveform"]
    fc = 0.5 * (float(wf["f_start"]) + float(wf.get("f_stop", wf["f_start"])))
    height = float(item["boundary_z"]) - float(cfg["platform"]["position"][2])
    params = bottom.sediment(item["sediment"])
    slant = 0.5 * c * (np.asarray(t, dtype=float) - 0.5 * pulse)
    grazing = np.arcsin(np.clip(height / slant, 0.0, 1.0))
    sigma = 10.0 ** (bottom.backscattering_strength_db(grazing, fc, c, **params) / 10.0)
    return sigma * np.pi * c * pulse / slant**3, slant, grazing, params


def check_sediment_reverberation(scatterer_seeds=(11, 12)) -> dict:
    """`medium_sand` through run_scenario three ways: the closed-form integral of the model's
    scattering strength, the scattering-function reverberation, and the same sediment as point
    scatterers in the scene (TR 0702 Sec. 11.3 and Sec. 11.1 as two realisations of Sec. 10)."""
    cfg = load_scenario_config(SEDIMENT)
    pulse = float(cfg["sonar_overrides"]["transmit"]["waveform"]["duration"])
    c = float(cfg["environment"]["sound_speed"])
    result = _quiet_run(SEDIMENT)
    record = result.records[0]
    fs, t0, a1 = record.samples.sample_rate, record.samples.start_time, record.transmit.waveform.amplitude
    x = samples_of(result)
    t = t0 + np.arange(x.shape[-1]) / fs
    predicted, slant, grazing, params = _sediment_prediction(cfg, t, pulse)
    height = float(cfg["reverberation"][0]["boundary_z"])
    ok = np.flatnonzero((slant > 1.5 * height) & (slant < 140.0))
    edges = np.linspace(ok[0], ok[-1], 13).astype(int)
    block = lambda p: np.array([p[a:b].mean() for a, b in zip(edges[:-1], edges[1:])])          # noqa: E731
    sf = (np.abs(x) ** 2).mean(axis=(0, 1)) / a1**2
    d_sf = db10(block(sf) / block(predicted))
    # The same sediment as point scatterers on the bottom.
    item = {"type": "seafloor", "x_range": [-160.0, 160.0], "y_range": [-160.0, 160.0], "depth": height, "density": 1.5,
            "model": "fluid_sediment", **params}
    pts = []
    for seed in scatterer_seeds:
        r = _quiet_run(SEDIMENT, {"reverberation": [], "scene": [item], "pings": {"count": 1}, "seed": int(seed)})
        pts.append(samples_of(r)[0])
    ps = (np.abs(np.concatenate(pts, axis=0)) ** 2).mean(axis=0) / a1**2
    d_ps = db10(block(ps) / block(predicted))
    return {"sediment": cfg["reverberation"][0]["sediment"], "parameters": params,
            "backscattering_strength_db": {"at_10_deg": float(_bs(params, 10.0, cfg)), "at_30_deg": float(_bs(params, 30.0, cfg))},
            "grazing_range_deg": [float(np.degrees(grazing[ok[-1]])), float(np.degrees(grazing[ok[0]]))],
            "scattering_function_vs_integral_db": {"mean": float(d_sf.mean()), "min": float(d_sf.min()), "max": float(d_sf.max()),
                                                   "pings_x_channels": int(x.shape[0] * x.shape[1])},
            "point_scatterers_vs_integral_db": {"mean": float(d_ps.mean()), "min": float(d_ps.min()), "max": float(d_ps.max()),
                                                "realisations_x_channels": int(len(scatterer_seeds) * x.shape[1])},
            "point_scatterers_vs_scattering_function_db": {"mean": float((d_ps - d_sf).mean()), "min": float((d_ps - d_sf).min()),
                                                           "max": float((d_ps - d_sf).max())},
            "blocks": 12, "sound_speed": c}


def _bs(params, grazing_deg, cfg):
    from gpusonar.sim import bottom
    wf = cfg["sonar_overrides"]["transmit"]["waveform"]
    return bottom.backscattering_strength_db(np.radians(grazing_deg), float(wf["f_start"]), float(cfg["environment"]["sound_speed"]), **params)


def check_signal_to_reverberation(bands=(1000.0, 2000.0, 4000.0), duration: float = 0.02, ts_db: float = 20.0,
                                  slant: float = 80.0, pings: int = 8) -> dict:
    """A point target in bottom reverberation of `medium_sand`, LFM pulses of one length and
    three bandwidths, pulse compression: the ratio of the target's peak to the mean
    reverberation at its range against the closed form

        SRR = TS - 10 log10( sigma(grazing) pi c R / B_eff )

    (the compressed pulse resolves a ring c / (2 B_eff) wide: T of the tone replaced by
    1 / B_eff, B_eff = (int |P|^2 df)^2 / int |P|^4 df of the pulse's spectrum P), and its
    growth of 3 dB per octave of bandwidth (TR 0702 p. 70)."""
    from gpusonar.sim import reverb
    cfg = load_scenario_config(SEDIMENT)
    c = float(cfg["environment"]["sound_speed"])
    fc = float(cfg["sonar_overrides"]["transmit"]["waveform"]["f_start"])
    target = [{"type": "points", "label": "target", "positions": [[0.0, slant, 0.0]], "target_strength_db": ts_db}]
    rows = {}
    for band in bands:
        wave = {"type": "lfm", "f_start": fc - 0.5 * band, "f_stop": fc + 0.5 * band, "duration": duration, "window_alpha": 0.1}
        sonar = {"transmit": {"waveform": wave}, "receive": {"window": {"range": [slant - 30.0, slant + 30.0]}}}
        rev = _quiet_run(SEDIMENT, {"sonar_overrides": sonar, "pings": {"count": int(pings)}})
        echo = _quiet_run(SEDIMENT, {"sonar_overrides": sonar, "pings": {"count": 1}, "reverberation": [], "scene": target})
        z_rev, t0, fs = _compressed(rev)
        z_echo, _, _ = _compressed(echo)
        a1 = rev.records[0].transmit.waveform.amplitude
        t = t0 + np.arange(z_rev.shape[-1]) / fs
        t_target = 2.0 * slant / c
        near = np.abs(t - t_target) < 0.004                               # +-3 m of slant range
        reverberation = float((np.abs(z_rev[:, :, near]) ** 2).mean())
        peak = float(np.mean([_peak_near(z_echo[0, ch], t0, fs, t_target, 3.0 / fs)[0] for ch in range(z_echo.shape[1])]))
        pulse = reverb.baseband_pulse(rev.records[0].transmit.waveform, fs, fc)
        spectrum = np.abs(np.fft.fft(pulse, 8 * len(pulse))) ** 2
        b_eff = float(spectrum.sum() ** 2 / (spectrum**2).sum() * fs / len(spectrum))
        cw = dict(cfg, sonar_overrides={**cfg["sonar_overrides"], "transmit": {"waveform": {"f_start": fc, "duration": 1.0 / b_eff}}})
        sigma_ring = float(_sediment_prediction(cw, np.array([t_target + 0.5 / b_eff]), 1.0 / b_eff)[0][0])
        closed = ts_db - 40.0 * np.log10(slant) - 10.0 * np.log10(sigma_ring)
        rows[f"{int(band)}"] = {"measured_db": float(db10(peak**2 / reverberation)), "closed_form_db": float(closed),
                           "effective_bandwidth_hz": b_eff, "echo_peak_over_sonar_equation": float(peak / (a1 * 10.0 ** (ts_db / 20.0) / slant**2))}
    measured = np.array([rows[f"{int(b)}"]["measured_db"] for b in bands])
    return {"bands": rows, "gain_per_octave_db": [float(d) for d in np.diff(measured)],
            "max_error_db": float(max(abs(r["measured_db"] - r["closed_form_db"]) for r in rows.values())),
            "pings_x_channels": int(z_rev.shape[0] * z_rev.shape[1]), "target_strength_db": ts_db, "slant_range_m": slant}


def check_image_model(max_bounces: int = 3) -> dict:
    """TR 0702 eqs. (36), (37), (40), (41) as printed, against a leg-by-leg unfolding of each
    path (our own derivation) and against the simulator's image-source provider and the
    bounce positions its time-spreading wrapper uses, for every bounce sequence of up to
    `max_bounces` bounces."""
    from gpusonar.sim.propagation import ImageSourceProvider
    from gpusonar.sim.timespread import TimeSpreadProvider
    c, depth = 1500.0, 100.0
    src, rcv = np.array([0.0, 0.0, 30.0]), np.array([400.0, 300.0, 55.0])
    provider = ImageSourceProvider(c, surface_z=0.0, bottom_z=depth, max_order=max_bounces, engine="numpy")
    es = provider.eigenrays(src, rcv[None, :])
    spread = TimeSpreadProvider(provider, frequency=8000.0, surface={"rms_height": 0.1, "mean_square_slope": 0.01, "rms_vertical_speed": 0.0},
                                bottom_z=depth)
    frac, _ = spread._bounce_fractions(src[2], np.array([[rcv[2]]]), es.launch[..., 2] < 0.0, es.n_surface, es.n_bottom)
    worst = {"eq36_vs_unfolding_m": 0.0, "delay_s": 0.0, "launch": 0.0, "arrival": 0.0, "bounce_fraction": 0.0}
    sequences = [""]
    for first in "sb":
        seq = ""
        for k in range(max_bounces):
            seq += first if k % 2 == 0 else ("b" if first == "s" else "s")
            sequences.append(seq)
    for seq in sequences:
        n_s, n_b = seq.count("s"), seq.count("b")
        vec = image_vector_eq36(src, rcv, depth, n_s, n_b, seq[:1] == "b")
        worst["eq36_vs_unfolding_m"] = max(worst["eq36_vs_unfolding_m"], abs(abs(vec[2]) - unfolded_vertical(src[2], rcv[2], depth, seq)))
        slant = np.linalg.norm(vec)
        up_first = seq[:1] == "s"
        match = np.flatnonzero((es.n_surface[0] == n_s) & (es.n_bottom[0] == n_b)
                               & ((es.launch[0, :, 2] < 0.0) == up_first if seq else np.ones(es.delay.shape[1], dtype=bool)))
        assert len(match) == 1, (seq, match)
        j = int(match[0])
        worst["delay_s"] = max(worst["delay_s"], abs(es.delay[0, j] - slant / c))                    # eq. (37)
        s_s = -vec / (c * slant)                                                                     # eq. (41)
        s_r = vec / (c * slant)
        if (n_s + n_b) % 2 == 1:
            s_r[2] = -s_r[2]
        worst["launch"] = max(worst["launch"], float(np.abs(es.launch[0, j] - (-c * s_s)).max()))
        worst["arrival"] = max(worst["arrival"], float(np.abs(es.arrival[0, j] - c * s_r).max()))
        for b in range(len(seq)):
            w = bounce_fraction_eq40(src[2], depth, vec[2], b, seq[0] == "s")
            worst["bounce_fraction"] = max(worst["bounce_fraction"], abs(float(frac[0, j, b]) - w))
    return {"sequences": sequences, **{k: float(v) for k, v in worst.items()}}


def check_absorption() -> dict:
    """TR 0702 eq. (42) (Thorp, as SST has it) beside the model the scenarios use. Not a test
    of ours against theirs: the two are different fits; the numbers say how far apart."""
    out = {}
    for f_khz in (8.0, 10.0):
        thorp = float(thorp_eq42_db_per_kyd(f_khz) / 0.9144)                 # dB / km
        fg = 1000.0 * float(acoustics.absorption(1000.0 * f_khz, 4.0, 35.0, 1000.0, 8.0, "francois_garrison"))
        out[f"{f_khz:g}_khz"] = {"thorp_eq42_db_per_km": thorp, "francois_garrison_db_per_km": fg}
    return out


def checks() -> dict:
    return {"image_model": check_image_model(), "absorption": check_absorption(),
            "echo_level": check_echo_level(), "sampled_doppler": check_sampled_doppler(),
            "sediment_reverberation": check_sediment_reverberation(), "signal_to_reverberation": check_signal_to_reverberation()}


# =================================================================================================
# 4. Where TR 0702's text is inconsistent: the reading the code takes
# =================================================================================================

def readings() -> list[dict]:
    """docs/PHYSICS_NOTES.md Sec. 11, items 17-24: for each apparent inconsistency in the
    text, which reading the code uses and which test pins it."""
    t = "tests/python/"
    return [
        {"item": 17, "text": "eq. (18), the interpolator's coefficient table, is printed as d(floor(z/L + 1/2) L); "
                             "dimensionally it should be d(floor(z L + 1/2) / L)",
         "reading": "Neither: the table of coefficients is not implemented. `sim/waveform.py` evaluates the windowed sinc itself "
                    "(Kaiser by default; eq. (17), `hann_sinc_weight`, with `interp_window: hann`) once, onto a table of the "
                    "SIGNAL 8 times finer (`table_factor`), then a cubic.",
         "pinned_by": t + "test_sampled_waveform.py::test_interpolator_error_against_the_design_rule_eq19 (eq. 17, 19), "
                          "::test_vardelay_script_of_tr0702_lowers_each_tone_by_one_minus_delay_rate"},
        {"item": 18, "text": "eqs. (76), (123) give the scattering strength the slowness vectors at the source and receiver "
                             "(S_Sp, S_Rq); eq. (132) gives it those at the scatterer (S_Tp, S_Tq)",
         "reading": "At the scatterer, eq. (132): grazing angles of both legs at the boundary and the bistatic azimuth there "
                    "(`reverb_bistatic._leg`). Beam patterns take the directions at the platforms.",
         "pinned_by": t + "test_reverb_bistatic.py::test_strength_is_given_the_true_angles_and_bistatic_azimuth, "
                          "::test_beam_patterns_use_the_elevation_at_the_platform_for_bounce_paths"},
        {"item": 19, "text": "eq. (131) writes |U(f, r)|^2 with a lower-case r", "reading": "Read as R (a misprint); nothing depends on it.",
         "pinned_by": "-"},
        {"item": 20, "text": "eq. (134) has U, not |U|^2, and no 1/dOmega, while (135) multiplies by dtheta dphi; (129) has |U|^2 and 1/dOmega",
         "reading": "eq. (129) with (121): power |a_q|^2 |a_p|^2 S dA per component, as power per radian of azimuth times the "
                    "sector width; the generator multiplies sqrt(power) by a unit Gaussian. The bookkeeping is checked by level.",
         "pinned_by": t + "test_reverb_bistatic.py::test_bistatic_power_matches_an_integral_over_a_cartesian_grid, "
                          "::test_bistatic_series_level_follows_the_cartesian_integral; "
                          + t + "test_sst_validation.py::test_sediment_reverberation_follows_the_scattering_strength_integral"},
        {"item": 21, "text": "S_l is a cross-section per unit area in Sec. 6.4.2 but must be its square root in eq. (132)",
         "reading": "The models return the cross-section per unit area (linear). Power uses it as it is; a point scatterer's "
                    "amplitude is its square root times a unit Gaussian over sqrt(density).",
         "pinned_by": t + "test_scene_sediment.py::test_scatterer_field_has_the_model_backscattering_strength; "
                          + t + "test_sst_validation.py::test_sediment_reverberation_follows_the_scattering_strength_integral"},
        {"item": 22, "text": "eq. (61) writes position updates as (V . S)(t - t0)",
         "reading": "Not used. Arrival times are solved exactly; the consequence, eq. (63)/(68), is what tests compare with.",
         "pinned_by": t + "test_sampled_waveform.py::test_eq68_reduces_to_the_exact_two_way_factor_for_radial_motion; "
                          + t + "test_sst_validation.py::test_pursuit_target_echo_is_the_simple_target_echo_model"},
        {"item": 23, "text": "p. 101: Doppler spread = f x rms vertical speed / (c x sin(grazing angle))",
         "reading": "Proportional to sin(grazing), not to its inverse: 2 f v sin(g) / c (half-width of the default AR(1) "
                    "spectrum), or that over sqrt(1 - exp(-Gamma^2)) as the rms width of the Gaussian option "
                    "(docs/PHYSICS_NOTES.md Sec. 16). TR 0702's own remark that the ASIAEX surface spread is 'less than "
                    "10 Hz' does not decide between them: see the ASIAEX rows.",
         "pinned_by": t + "test_timespread.py::test_doppler_width_of_a_tone_through_one_surface_bounce; "
                          + t + "test_sst_validation.py::test_asiaex_surface_doppler_spread_is_under_10_hz"},
        {"item": 24, "text": "eq. (36): image vector with the term 2 |S_p - B_p| z_R",
         "reading": "Correct as printed: it equals a leg-by-leg unfolding for every sequence of up to three bounces, and the "
                    "provider's own images.",
         "pinned_by": t + "test_sst_validation.py::test_image_equations_as_printed_match_the_provider"},
    ]


# =================================================================================================
# 5. The validation matrix
# =================================================================================================

def _f(value, fmt: str = ".2f") -> str:
    return "n/a" if value is None else format(value, fmt)


def matrix(results: dict | None = None) -> list[dict]:
    """One row per checkable statement of TR 0702, section by section. `results` is
    {"pursuit": pursuit(), "asiaex": asiaex(), "checks": checks()} (any may be missing:
    the measured column then says so)."""
    results = results or {}
    P, A, C = results.get("pursuit"), results.get("asiaex"), results.get("checks")
    t = "tests/python/"
    mine = t + "test_sst_validation.py::"
    NOT_RUN = "(not run)"

    def m(source, text):
        """Measured text, or a note that the part was not run."""
        try:
            return text(source) if source is not None else NOT_RUN
        except (KeyError, TypeError):
            return NOT_RUN

    rows = []

    def row(where, states, ours, check, result):
        rows.append({"where": where, "states": states, "ours": ours, "check": check, "result": result})

    # ---- Sec. 2 --------------------------------------------------------------------------------
    row("Sec. 2.5, 5.1", "Source signals are pressure reduced to 1 m (uPa m); the loss factor L_p has units 1/m; received signal in uPa",
        "`source_level_db`, record amplitude at 1 m, `amplitude` of each eigenray",
        mine + "test_echo_level_is_the_sonar_equation",
        m(C, lambda c: f"compressed echo level - (SL - 2 TL + TS): {c['echo_level']['free_field']['max_level_error_db']:.3f} dB (free field)"))
    row("Sec. 2.1, 2.4", "Four components (direct sources, noise, target echoes, reverberation) summed into the ping",
        "echoes + reverberation + noise in `simulate_ping`; no one-way source (`DirectSignal`)",
        mine + "test_pursuit_surface_reverberation_dominates (the parts add)",
        m(P, lambda p: f"full - echoes - reverberation = noise at {p['noise']['psd_db_re_upa2_per_hz']:.1f} dB re uPa^2/Hz; "
                       f"the scenario's spectrum at the carrier: {p['noise']['scenario_spectrum_db']:.1f}"))
    # ---- Sec. 3 --------------------------------------------------------------------------------
    row("Sec. 3, Fig. 2", "Pursuit: monostatic forward-looking sonar on a high-speed vehicle, 5 receive channels, tone near 10 kHz, "
        "three-highlight fleeing target, display 0.5-5 s and 9.6-10.2 kHz",
        "`configs/scenarios/sst_pursuit.yaml`; speeds, depths, beams, pulse, wind, levels are ours (`unconfirmed`)",
        mine + "test_pursuit_scenario_holds_what_tr0702_states",
        "Set up as stated; NOT a numerical reproduction (parameters not given)")
    row("Sec. 3", "Reverberation hook: early return from overhead through sidelobes, little Doppler; late return from the main lobe, "
        "high Doppler", "`geometry: bistatic` reverberation components (azimuth x elevation, own Doppler), sinc beam patterns",
        mine + "test_pursuit_reverberation_hook_follows_the_closed_form",
        m(P, lambda p: f"ridge - f (1 + 2 v cos(angle) / c): median {p['hook']['settled_ridge_minus_first_order_hz']['median']:+.1f} Hz, "
                       f"max {p['hook']['settled_ridge_minus_first_order_hz']['max_abs']:.1f} Hz once settled (cell {p['hook']['resolution_hz']:.1f} Hz); "
                       f"{100 * p['hook']['fraction_inside_slew_bracket']:.0f} % of segments inside the bracket of one pulse length; "
                       f"shift {p['hook']['early_shift_hz']:.0f} Hz before 0.7 s, {p['hook']['late_shift_hz']:.0f} Hz after 4 s"))
    row("Sec. 3", "Main ridge at the Doppler of straight ahead (eq. 63 / 71)",
        "generator shifts each component by (Gamma - 1) f with the factor of eq. (71) itself",
        mine + "test_pursuit_reverberation_hook_follows_the_closed_form",
        m(P, lambda p: f"exact eq. (71) ridge {p['hook']['main_lobe_ridge_eq71_hz']:.1f} Hz (the first-order form is "
                       f"{-p['hook']['first_order_minus_eq71_hz']:.1f} Hz lower at 20 m/s; the generator used it until its components were keyed by Doppler)"))
    row("Sec. 3", "A streak starting at about 1.7 s that merges with the ridge: a path that reflects from the bottom",
        "image-source provider under the reverberation (pairs of direct and bottom-reflected legs)",
        mine + "test_pursuit_bottom_bounce_streak_starts_when_geometry_says",
        m(P, lambda p: f"onset {p['bottom_bounce']['measured_onset_s']:.3f} s, closed form 2 D / c = {p['bottom_bounce']['closed_form_onset_s']:.3f} s "
                       f"(our depth); on the track of eq. (71) the return is {p['bottom_bounce']['on_true_track_db_re_no_bottom']:.1f} dB above the run "
                       f"without a bottom ({p['bottom_bounce']['control_40hz_above_track_db_re_no_bottom']:+.1f} dB 40 Hz above it, where eq. (71) "
                       f"allows nothing)"))
    row("Sec. 3 / Sec. 11.2", "(implied) each eigenray pair has its own Doppler, eq. (71)",
        "components keyed by (azimuth, receive elevation, bin of total Doppler): pairs with different shifts are not merged "
        "(was one Doppler per direction: a defect found here, fixed in the reverberation package)",
        mine + "test_reverberation_gives_each_eigenray_pair_its_own_doppler, "
        "::test_bottom_paths_change_nothing_before_they_can_arrive",
        m(P, lambda p: f"Adding the bottom changes the record before any bottom path can arrive by at most "
                       f"{p['bottom_bounce']['largest_change_before_onset_db']:+.1f} dB near {p['bottom_bounce']['largest_change_at_s_hz'][0]:.2f} s, "
                       f"{p['bottom_bounce']['largest_change_at_s_hz'][1]:.0f} Hz"
                       + (f"; mean Doppler from dead ahead off by up to {p['cross_pairs']['max_error_with_bottom_paths_hz']:.0f} Hz "
                          f"({p['cross_pairs_no_bottom']['max_error_direct_only_hz']:.1f} Hz without a bottom)" if "cross_pairs" in p else "")))
    row("Sec. 3", "Target echo at about 2.8 s and 10.13 kHz, lower Doppler than the ridge (the target is running away)",
        "`type: target` (HighlightTarget), exact two-way kinematics in the kernel",
        mine + "test_pursuit_target_echo_is_the_simple_target_echo_model",
        m(P, lambda p: f"arrival {p['target']['measured_arrival_s']:.4f} s (closed form {p['target']['closed_form_arrival_s']:.4f}); frequency "
                       f"{p['target']['measured_frequency_hz']:.2f} Hz (eq. 68: {p['target']['eq68_frequency_hz']:.2f}); "
                       f"{p['target']['separation_from_ridge_hz']:.0f} Hz below the ridge; record vs eqs. (66)+(68): rms difference "
                       f"{100 * p['target']['echo_vs_eq66_residual']:.2f} %, level {p['target']['level_error_db']:+.3f} dB"))
    row("Sec. 3", "An easy high-Doppler detection although the echo is well below the reverberation level (not visible on the level plot)",
        "spectrogram of one channel, as in Fig. 2", mine + "test_pursuit_target_is_below_the_reverberation_but_separated_in_doppler",
        m(P, lambda p: f"echo {p['target']['band_level_echo_minus_reverberation_db']:+.1f} dB re reverberation in the displayed band (level rises "
                       f"{p['target']['band_level_rise_from_the_echo_db']:.1f} dB); its spectrogram cell is {p['target']['cell_db_above_reverberation']:.1f} dB above "
                       f"the reverberation there"))
    row("Sec. 3", "The dominant feature is the reverberation, primarily from the surface",
        "surface boundary (Chapman-Harris, extrapolated) is the only scattering boundary; Wenz noise",
        mine + "test_pursuit_surface_reverberation_dominates",
        m(P, lambda p: f"ridge PSD over noise PSD: median {p['levels']['ridge_psd_over_noise_psd_db']['median']:.1f} dB; level in the band over noise: "
                       f"median {p['levels']['band_reverberation_over_noise_db']['median']:.1f} dB, above it in "
                       f"{100 * p['levels']['band_reverberation_over_noise_db']['fraction_above']:.0f} % of segments (source level is our choice)"))
    row("Sec. 3 with Sec. 12", "(ours) does time spreading of the surface-bounce echo matter in this example?", "`time_spread` on the echo run",
        mine + "test_pursuit_target_echo_is_the_simple_target_echo_model",
        m(P, lambda p: f"No: the direct echo changes by {p['time_spreading']['direct_echo_change']:.1e}; the echo with a surface-reflected leg is "
                       f"{p['time_spreading']['surface_bounce_echo_db_re_direct']:.1f} dB re the direct one, arrives after it, and its level changes by "
                       f"{p['time_spreading']['surface_bounce_echo_change_db']:+.1f} dB (it becomes a random tail). None of the features "
                       f"checked depends on it: left out of the scenario file"))
    row("Sec. 3", "Signal includes noise produced by the fleeing target", "-", "cannot be checked",
        "Feature we lack: no one-way source (SST `DirectSignal`); ambient noise only")
    row("Sec. 3", "Five channels; inter-channel correlations are in the signal", "5 hydrophones, each with its pattern",
        mine + "test_pursuit_scenario_holds_what_tr0702_states", "5 channels generated; coherence across them is covered by "
        + t + "test_reverb_statistics.py::test_direction_spread_gives_the_coherence_length")
    row("Sec. 3", "Runs in a few seconds", "-", "-", m(P, lambda p: f"{p.get('seconds_one_ping', float('nan')):.0f} s for one ping of the scenario as written; "
                       f"{p.get('seconds', float('nan')):.0f} s for all the runs behind these rows (4 shared cores, CPU)"))
    # ---- Sec. 4 --------------------------------------------------------------------------------
    row("Sec. 4.1.2, eqs. (4)-(6)", "Complex envelope with sqrt(2), so that envelope power = signal power",
        "envelopes of unit peak, no sqrt(2) (`sim/waveform.py`); |envelope| is the tone's amplitude",
        "by design; " + t + "test_sampled_waveform.py::test_record_carries_the_pulse_in_its_existing_fields_and_survives_the_file",
        "Convention differs (documented, PHYSICS_NOTES Sec. 15); levels here are checked in uPa through the sonar equation")
    row("Sec. 4.1.3, 4.2, eqs. (7)-(13)", "Windowed frequency domain, cross power spectra, scattering functions as data types",
        "-", "not applicable", "Not SST's data-flow objects: our reverberation is generated in time from powers per direction")
    row("Sec. 4.3 (script, p. 20-21)", "VarDelay example: two tones, delay 1.00 ... 2.0 s; tones lowered as the delay grows",
        "`sim/waveform.py` variable-delay resampler", t + "test_sampled_waveform.py::test_vardelay_script_of_tr0702_lowers_each_tone_by_one_minus_delay_rate",
        "tones at n f1 (1 - dT/dt) within 0.02 Hz; output = x(t - T(t)) within 5.1e-3 (M = 6, Hann) [feature report]")
    row("Sec. 4.5.1, eqs. (14)-(17), (19)", "Hann-windowed sinc of M taps; M >= 2.5 / (1 - a); a = 0.8 needs 13; order 5 for <= 50 %",
        "`interp_window: hann`", t + "test_sampled_waveform.py::test_interpolator_error_against_the_design_rule_eq19",
        "error at the rule: -39.1 dB (a = 0.5, M = 6), -33.6 dB (0.8, 14), -29.4 dB (0.9, 26) [feature report]; TR gives no error figure")
    row("Sec. 4.5.1, eq. (18)", "Coefficients tabulated on a grid L = 512 times finer, nearest entry", "not implemented (signal tabulated instead)",
        "see readings, item 17", "Not checked: not implemented; the formula is misprinted")
    row("Sec. 4.5.2, eqs. (20)-(22)", "Slowly varying FIR filters built by the window method from H(f)",
        "gain at the instantaneous frequency (analytic pulses) or piecewise linear in frequency (`gain_knots`, sampled pulses)",
        t + "test_sampled_waveform.py::test_sampled_lfm_with_beam_patterns_and_absorption",
        "8.5e-4 / 4.8e-4 between the two (feature report). SST's filter lengths are not given: no direct comparison")
    row("Sec. 4.6.1, eqs. (23)-(26)", "Gaussian noise with a given cross-spectral matrix (Mitchell-McPherson)",
        "`sim/noise.py`: spectrum and the coherence of an isotropic field", t + "test_noise.py; " + mine + "test_pursuit_surface_reverberation_dominates",
        m(P, lambda p: f"noise PSD at the carrier {p['noise']['psd_db_re_upa2_per_hz']:.1f} dB vs {p['noise']['scenario_spectrum_db']:.1f} dB asked for"))
    row("Sec. 4.6.2, eq. (27)", "K-distributed white noise (`KNoise`)", "-", "cannot be checked",
        "Feature we lack as noise; K statistics exist for reverberation (" + t + "test_reverb_statistics.py::test_k_distributed_intensity_moments)")
    row("Sec. 4.6.3-4.6.4, eqs. (29)-(30)", "Harmonic families; modulated tones with numerically integrated phase",
        "analytic CW / LFM / Gaussian pulses; sampled pulses", t + "test_sampled_waveform.py (LFM, HFM, stepped FM, codes)",
        "Tonal sources (radiated noise) are not implemented: they need a one-way source")
    # ---- Sec. 5 --------------------------------------------------------------------------------
    row("Sec. 5.2, eqs. (32)-(33)", "Delay across an array from the slowness vector, with a spherical-wavefront correction",
        "not used: arrival time solved per hydrophone", "-", "Not checked (not used)")
    row("Sec. 5.4, eqs. (36)-(37)", "Image vector and delay of each eigenray",
        "`ImageSourceProvider`", mine + "test_image_equations_as_printed_match_the_provider, ::test_asiaex_arrival_times_and_levels_are_the_image_closed_forms",
        m(C, lambda c: f"eq. (36) vs leg-by-leg unfolding: {c['image_model']['eq36_vs_unfolding_m']:.1e} m; provider delay vs eq. (37): "
                       f"{c['image_model']['delay_s']:.1e} s ({len(c['image_model']['sequences'])} bounce sequences)")
        + m(A, lambda a: f"; ASIAEX arrivals in the record: max {a['arrival_time_max_error_us']:.2f} us"))
    row("Sec. 5.4, eq. (38)", "Loss = spreading x absorption x product of reflection coefficients",
        "eigenray amplitude, absorption along c x delay", mine + "test_asiaex_arrival_times_and_levels_are_the_image_closed_forms, ::test_echo_level_is_the_sonar_equation",
        m(A, lambda a: f"five ASIAEX arrivals: max {a['arrival_level_max_error_db']:.3f} dB")
        + m(C, lambda c: f"; six groups of pairs after pulse compression: max {c['echo_level']['image_source']['max_level_error_db']:.3f} dB, "
                         f"{c['echo_level']['image_source']['max_time_error_us']:.2f} us"))
    row("Sec. 5.4, eqs. (39)-(40)", "Where along the path each bounce happens", "`TimeSpreadProvider._bounce_fractions`",
        mine + "test_image_equations_as_printed_match_the_provider", m(C, lambda c: f"max difference {c['image_model']['bounce_fraction']:.1e}"))
    row("Sec. 5.4, eq. (41)", "Slowness vectors at both ends (vertical sign reversed for an odd number of bounces)",
        "eigenray `launch`, `arrival`", mine + "test_image_equations_as_printed_match_the_provider",
        m(C, lambda c: f"max difference {max(c['image_model']['launch'], c['image_model']['arrival']):.1e}"))
    row("Sec. 5.5-5.7", "CASS / GSM eigenrays, interpolation, ray identity heuristics", "Bellhop provider (`sim/bellhop.py`)",
        "cannot be checked against TR 0702", "No formulas in TR 0702; ours is tested against Bellhop itself (" + t + "test_propagation.py)")
    # ---- Sec. 6 --------------------------------------------------------------------------------
    row("Sec. 6.3, eq. (42)", "Thorp absorption", "Francois-Garrison (default), Ainslie-McColm", "not implemented",
        m(C, lambda c: "For scale, at 4 C and 1 km: " + "; ".join(
            f"{k.replace('_', ' ')}: Thorp {v['thorp_eq42_db_per_km']:.2f}, Francois-Garrison {v['francois_garrison_db_per_km']:.2f} dB/km"
            for k, v in c["absorption"].items())))
    row("Sec. 6.4.1", "Coherent forward amplitude for echoes, total for reverberation legs",
        "`time_spread`: coherent part exp(-Gamma^2 / 2) plus taps for echoes; flat-boundary paths for reverberation",
        t + "test_timespread.py::test_coherent_part_is_the_rayleigh_parameter_factor",
        "Structure as stated; the split's formula is in TR 9407 (not available): ours is the Rayleigh-parameter factor")
    row("Sec. 6.4.2", "Bistatic strength; monostatic models made bistatic by the geometric mean of the two backscattering strengths",
        "`fluid_sediment` at the true angles and bistatic azimuth; Chapman-Harris at the MEAN grazing angle",
        t + "test_reverb_bistatic.py::test_fluid_sediment_model_is_used_at_the_bistatic_azimuth", "The geometric-mean rule is not what Chapman-Harris "
        "gets here (difference in reading, not checked numerically)")
    row("Sec. 6.4.3, Table 6", "APLSurface, GilbertSurface, McDanielSurface, APLBottom, JacksonBottom", "open-literature substitutes (PHYSICS_NOTES Sec. 13)",
        "cannot be checked", "Needs TR 9407 and the McDaniel / Gilbert / Jackson papers; TR 0702 gives no formulas")
    row("Sec. 6.4.3", "Named bottom types (e.g. MediumSand)", "`configs/sediments.yaml` `medium_sand` (arXiv:1809.08459 Table 1)",
        mine + "test_sediment_reverberation_follows_the_scattering_strength_integral",
        m(C, lambda c: f"through run_scenario, level vs integral of the model's strength: mean {c['sediment_reverberation']['scattering_function_vs_integral_db']['mean']:+.2f} dB, "
                       f"{c['sediment_reverberation']['scattering_function_vs_integral_db']['min']:+.2f} to {c['sediment_reverberation']['scattering_function_vs_integral_db']['max']:+.2f} dB "
                       f"over 12 range blocks"))
    row("Sec. 6.5", "Volume layers collapsed onto sheets", "`model: volume`", t + "test_reverberation.py::test_volume_layer_level_follows_the_sonar_equation",
        "Covered at module and scenario level by the existing tests")
    # ---- Sec. 7-9 ------------------------------------------------------------------------------
    row("Sec. 7.2, Table 7", "Beam pattern classes", "one separable sinc pattern per element and for the projector", "cannot be checked",
        "No pattern is specified numerically in TR 0702; ours appears in the Pursuit hook (sidelobes) only")
    row("Sec. 8.1, eqs. (47)-(52)", "One-way source-to-receiver signal (`DirectSignal`)", "-",
        mine + "test_asiaex_arrival_times_and_levels_are_the_image_closed_forms",
        "Feature we lack; a one-way path is made with a relay 0.1 m from the projector (sst_asiaex.yaml)")
    row("Sec. 8.3, eq. (63)", "Doppler factor of a path from the velocities and slowness vectors",
        "exact arrival-time solve (echoes); first order (reverberation)",
        t + "test_reverb_bistatic.py::test_doppler_shift_is_the_closed_form_factor_to_first_order; " + mine + "test_pursuit_reverberation_hook_follows_the_closed_form",
        m(P, lambda p: f"hook ridge vs first-order closed form: median {p['hook']['ridge_minus_first_order_hz']['median']:+.1f} Hz over {p['hook']['segments']} segments"))
    row("Sec. 9.3, eqs. (66), (68), (70)", "Echo = sum over eigenray pairs of delayed, scaled, time-dilated copies; Doppler of a moving highlight",
        "two-way pairing + synthesis kernel", mine + "test_pursuit_target_echo_is_the_simple_target_echo_model, ::test_doppler_bank_finds_a_moving_highlight_through_a_sampled_pulse",
        m(P, lambda p: f"Pursuit echo vs eq. (66): {100 * p['target']['echo_vs_eq66_residual']:.2f} % rms")
        + m(C, lambda c: f"; sampled pulse + Doppler bank: range rate {c['sampled_doppler']['bank_range_rate_fine']:+.2f} m/s (true {c['sampled_doppler']['true_range_rate']:+.1f}), "
                         f"peak / sonar equation {c['sampled_doppler']['peak_over_sonar_equation']:.3f}"))
    row("Sec. 9.2", "PointTarget, HighlightTarget (delay, response, groups), ExternalTarget (aspect per ping)", "`sim/targets.py`",
        t + "test_scene_targets.py", "Covered by the feature's tests; group numbers not needed (one eigenray set per highlight)")
    # ---- Sec. 10 -------------------------------------------------------------------------------
    row("Sec. 10.3, eq. (76); Sec. 10.5, eqs. (77)-(80)", "Scatterers delta-correlated; a cell's amplitude is Gaussian with variance = cell size x strength",
        "point scatterers of `model: fluid_sediment`", mine + "test_point_scatterers_and_scattering_function_agree_for_medium_sand",
        m(C, lambda c: f"scatterer field vs integral: mean {c['sediment_reverberation']['point_scatterers_vs_integral_db']['mean']:+.2f} dB; vs scattering-function "
                       f"reverberation: mean {c['sediment_reverberation']['point_scatterers_vs_scattering_function_db']['mean']:+.2f} dB "
                       f"({c['sediment_reverberation']['point_scatterers_vs_scattering_function_db']['min']:+.2f} to "
                       f"{c['sediment_reverberation']['point_scatterers_vs_scattering_function_db']['max']:+.2f})"))
    row("Sec. 10.6, eqs. (89), (96), (103)", "Gaussian spread in time / Doppler / direction gives coherence in frequency / time / space",
        "reverberation generator", t + "test_reverb_statistics.py::test_time_spread_gives_the_coherence_bandwidth, ::test_doppler_spread_gives_the_coherence_time, "
        "::test_direction_spread_gives_the_coherence_length", "7.971 vs 7.958 Hz; 40.06 vs 39.79 ms; 0.2973 vs 0.2984 m [feature report]")
    row("Sec. 10.6.1, p. 70", "Replica correlation gains 3 dB of signal-to-reverberation per octave of bandwidth",
        "run_scenario -> to_baseband -> pulse_compress", mine + "test_signal_to_reverberation_after_pulse_compression",
        m(C, lambda c: "per octave: " + ", ".join(f"{g:+.2f}" for g in c["signal_to_reverberation"]["gain_per_octave_db"])
                       + f" dB; absolute ratio vs TS - 10 log10(sigma pi c R / B): max {c['signal_to_reverberation']['max_error_db']:.2f} dB"))
    row("Sec. 10.3, p. 66", "Eigenray pairs uncorrelated (no Lloyd's-mirror striations in reverberation)",
        "scattering-function reverberation: yes; scene scatterers: pairs share the scatterer's amplitude", "-", "Not checked end to end")
    row("Sec. 10.8, eqs. (108)-(109)", "Complex-envelope form with different source and output centre frequencies", "baseband records at the pulse's centre",
        "-", "Not checked separately; every baseband check above goes through it with Fx = Fy")
    # ---- Sec. 11 -------------------------------------------------------------------------------
    row("Sec. 11.1, eqs. (110)-(111)", "Point scatterers placed by marching outward in range", "uniform Poisson patches (`seafloor`)", "not implemented as printed",
        "Different placement, same statistics: see the Sec. 10.5 row")
    row("Sec. 11.2, eqs. (113)-(129)", "Directional Doppler density: one Gaussian per (time, Doppler, direction) cell; receiver applied last",
        "`reverb_bistatic.directional_power` + generator: (azimuth, elevation, Doppler bin) components",
        t + "test_reverb_bistatic.py; " + mine + "test_reverberation_gives_each_eigenray_pair_its_own_doppler",
        "Level, array structure and a Doppler per pair of eigenrays covered (bins of total Doppler no finer than half the pulse's 1 / T)")
    row("Sec. 11.3, eqs. (130)-(135)", "Scattering function convolved with the source spectrum, Cholesky factor, overlap-add",
        "-", "not implemented in that form", "Our generator convolves the pulse with a Gaussian sequence per direction (keeps the waveform); levels agree "
        "with the same integral (Sec. 6.4.3 row)")
    # ---- Sec. 12 -------------------------------------------------------------------------------
    row("Sec. 12.2, Fig. 3", "ASIAEX: 8 kHz, 3 ms, ~500 m, 20 pings; arrivals direct, surface, bottom, surface-bottom, bottom-surface; 0-35 ms",
        "`configs/scenarios/sst_asiaex.yaml`; depths, sound speed, sediment, ping interval, wind laws are ours (`unconfirmed`)",
        mine + "test_asiaex_arrival_times_and_levels_are_the_image_closed_forms",
        m(A, lambda a: "arrivals at " + ", ".join(f"{abs(round(v['measured_ms'], 2)):.2f}" for v in a["arrivals"].values()) + " ms after the first (our depths); "
                       "NOT a numerical reproduction"))
    row("Sec. 12.2", "Without time spreading: no ping-to-ping variation; leading and trailing edges equally crisp",
        "`time_spread` absent", mine + "test_asiaex_without_time_spreading_peaks_are_crisp_and_identical",
        m(A, lambda a: f"relative std across 20 pings {a['no_time_spread']['ping_to_ping_relative_std']:.0e}; 0.3-0.7 ms after a pulse: "
                       + ", ".join(f"{k} {v:.0f} dB" for k, v in a["no_time_spread"]["db_after_pulse_re_peak"].items()) + " re its peak"))
    row("Sec. 12.3", "Doppler spreading width of the surface peak is less than 10 Hz (TR 9407 eq. II-43a)",
        "`time_spread` process from ping to ping", mine + "test_asiaex_surface_doppler_spread_is_under_10_hz",
        m(A, lambda a: f"from the decorrelation of the tail over {a['surface_doppler_spread']['lorentzian']['pings']} pings: "
                       f"{a['surface_doppler_spread']['lorentzian']['measured_hz']:.2f} Hz (AR(1) half-width; closed form "
                       f"{a['surface_doppler_spread']['lorentzian']['closed_form_hz']:.2f}), {a['surface_doppler_spread']['gaussian']['measured_hz']:.2f} Hz "
                       f"(Gaussian rms; closed form {a['surface_doppler_spread']['gaussian']['closed_form_hz']:.2f}). The rule printed on p. 101 "
                       f"would give {a['surface_doppler_spread']['p101_rule_hz']:.1f} Hz: also under 10"))
    row("Sec. 12.4", "Coherent / incoherent split; stochastic FIR filter with Gaussian taps; AR(1) evolution", "`sim/timespread.py`",
        t + "test_timespread.py", "Structure as stated; the three formulas are in TR 9407 (not available), ours are substitutes")
    row("Sec. 12.4", "Near-specular reverberation is removed when time spreading is on", "-", "cannot be checked",
        "Feature we lack: reverberation takes the flat-boundary paths and is not reduced. No double counting with Lambert / "
        "Chapman-Harris (no specular lobe); there could be with a Kirchhoff bottom (not measured)")
    row("Sec. 12.4, rmsHeight", "Mostly incoherent if rmsHeight > wavelength / sin(grazing), mostly coherent if smaller",
        "coherent intensity exp(-(4 pi h sin(g) / wavelength)^2)", t + "test_timespread.py::test_coherent_part_is_the_rayleigh_parameter_factor",
        m(A, lambda a: f"consistent at both ends; ours is half coherent at h = 0.066 wavelength / sin(g). ASIAEX: surface {a['tails']['surface']['coherent_fraction']:.2f}, "
                       f"bottom {a['tails']['bottom']['coherent_fraction']:.2f} coherent"))
    row("Sec. 12.4, meanSquareSlope", "Spreading time ~ mss x half the one-way time (mid-path bounce); mss x twice the shorter leg (near one end)",
        "decay time 2 mss t1 t2 / (t1 + t2)", mine + "test_asiaex_tails_decay_at_the_rule_of_thumb",
        m(A, lambda a: f"fitted to the mean intensity after the pulse: surface {a['tails']['surface']['fitted_ms']:.2f} ms (rule {a['tails']['surface']['rule_of_thumb_ms']:.2f}), "
                       f"bottom {a['tails']['bottom']['fitted_ms']:.2f} ms (rule {a['tails']['bottom']['rule_of_thumb_ms']:.2f}; mss 0.045 as printed)"))
    row("Sec. 12.4, rmsVerticalSpeed (p. 101)", "Doppler spread = f x rms vertical speed / (c x sin(grazing))", "2 f v sin(g) / c", "see readings, item 23",
        "Reading differs from the print (sin g, not 1 / sin g)")
    row("Sec. 12.5, Figs. 4-5", "With time spreading: tails and ping-to-ping variation; direct path unchanged",
        "`time_spread` as in the scenario", mine + "test_asiaex_with_time_spreading_bounce_paths_get_tails_and_vary",
        m(A, lambda a: "std of level across 20 pings: " + ", ".join(f"{k} {v:.2f} dB" for k, v in a["time_spread"]["ping_to_ping_std_db"].items())
                       + "; 0.3-0.7 ms after the pulse: " + ", ".join(f"{k} {v:.0f} dB" for k, v in a["time_spread"]["db_after_pulse_re_peak"].items() if k != "direct")))
    row("Sec. 12.6", "Wind 3 m/s fits the tail, 4 m/s the variability (McDanielSurface)", "Pierson-Moskowitz height, Cox-Munk slope", "cannot be checked",
        "Needs the McDaniel laws and the data of Dahl (2002); only figures in TR 0702")
    row("Sec. 12.6", "Multiple-bounce peaks: intensity too high immediately after the pulse and too low later",
        "`multi_bounce: sum` (default) against `convolve`", mine + "test_asiaex_two_bounce_tail_sum_against_convolve",
        m(A, lambda a: f"ASIAEX parameters (bottom {a['two_bounces']['coherent_fraction_bottom']:.2f} coherent): `sum` is {a['two_bounces']['sum_over_convolve_early_db']:+.1f} dB "
                       f"right after the pulse and {a['two_bounces']['sum_over_convolve_late_db']:+.1f} dB later re `convolve` (tail centroid "
                       f"{a['two_bounces']['tail_centroid_sum_ms']:.1f} against {a['two_bounces']['tail_centroid_convolve_ms']:.1f} ms) - the OPPOSITE sign; "
                       f"with two fully rough bounces of equal decay: {a['two_rough_bounces']['sum_over_convolve_first_half_of_pulse_db']:+.1f} dB in the "
                       f"first half of the pulse, {a['two_rough_bounces']['sum_over_convolve_2_to_5_ms_after_pulse_db']:+.1f} dB 2-5 ms after it - "
                       f"the defect as TR 0702 describes it"))
    row("Sec. 12.2-12.5, Figs. 3-4 panels 3-5", "Scattering function at 1.5 and 0.1 ms updates; forward-scatter reverberation; total",
        "-", "cannot be checked", "No near-specular surface scattering model here (TR 0702's is McDaniel's) and no removal of it; only figures, no numbers")
    row("Sec. 12.2, Fig. 3 panel 1", "Measured ASIAEX data", "-", "cannot be checked", "Data of Dahl (2002) not used; no numbers in TR 0702")
    return rows


def markdown(rows: list[dict], results: dict | None = None) -> str:
    """The matrix, the readings and the numbers as Markdown."""
    lines = ["## Validation against APL-UW TR 0702", "",
             "Emitted by `tools/sst_validation.py`. Two worked examples in TR 0702 are run with parameters of our own where the "
             "report gives none (`unconfirmed` in the scenario files): they are not numerical reproductions of its figures. "
             "\"feature report\" marks a number measured by the test named, taken from the report of the work that added it.", "",
             "| TR 0702 | What it states | What of ours it exercises | Checked by | Measured, or why not |", "|---|---|---|---|---|"]
    esc = lambda x: str(x).replace("|", "\\|").replace("\n", " ")              # noqa: E731
    for r in rows:
        lines.append("| " + " | ".join(esc(r[k]) for k in ("where", "states", "ours", "check", "result")) + " |")
    lines += ["", "### Where the text of TR 0702 is inconsistent: the reading the code takes", "",
              "| Item (PHYSICS_NOTES Sec. 11) | The text | Reading in the code | Pinned by |", "|---|---|---|---|"]
    for r in readings():
        lines.append(f"| {r['item']} | {esc(r['text'])} | {esc(r['reading'])} | {esc(r['pinned_by'])} |")
    return "\n".join(lines) + "\n"


def _plain(obj):
    """Results without the arrays kept for the figures, as JSON-able values."""
    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in obj.items() if k != "display" and not isinstance(v, np.ndarray)}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


# =================================================================================================
# 6. Figures (the style of tools/gallery.py: grey images, one title line, constrained layout)
# =================================================================================================

def figure_pursuit(result: dict, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = result["display"]
    t, f = d["t"], d["f"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 6.5), sharey=True, constrained_layout=True, gridspec_kw={"width_ratios": [2.2, 1]})
    image = db10(d["full"])
    top = float(np.percentile(image, 99.9))
    m = axes[0].imshow(image, extent=[f[0] / 1e3, f[-1] / 1e3, t[-1], t[0]], cmap="gray", vmin=top - 60.0, vmax=top, aspect="auto")
    axes[0].plot(d["hook"] / 1e3, t, color="C1", lw=0.8, ls="--", label="surface, direct: f (1 + 2 v cos(angle) / c)")
    ts, fs_, _ = d["streak"]
    axes[0].plot(fs_ / 1e3, ts + 0.5 * result["parameters"]["pulse_s"], color="C0", lw=0.8, ls=":", label="one leg by the bottom, eq. (71)")
    axes[0].plot([d["target"][1] / 1e3], [d["target"][0]], marker="o", ms=11, mfc="none", mec="C3", ls="none", label="target echo, eq. (68)")
    axes[0].set_xlim(9.6, 10.2)
    axes[0].set_ylim(5.0, 0.5)
    axes[0].set_xlabel("Frequency (kHz)")
    axes[0].set_ylabel("Time (s)")
    axes[0].legend(loc="lower left", fontsize=8)
    axes[0].set_title("sst_pursuit: one of 5 channels, PSD of 0.1 s segments (simulated)", fontsize=9)
    fig.colorbar(m, ax=axes[0], label="dB re 1 uPa^2/Hz")
    for key, label, style in (("band_full", "total", dict(color="k", lw=1.0)), ("band_reverb", "reverberation (mean of pings)", dict(color="C1", lw=0.8)),
                              ("band_echo", "echoes", dict(color="C3", lw=0.8)), ("band_noise", "noise", dict(color="0.6", lw=0.8))):
        axes[1].plot(db10(d[key]), t, label=label, **style)
    lo = float(np.median(db10(d["band_noise"])))
    axes[1].set_xlim(lo - 15.0, float(db10(d["band_full"]).max()) + 5.0)
    axes[1].set_xlabel("Level in 9.6-10.2 kHz (dB re 1 uPa^2)")
    axes[1].legend(loc="lower right", fontsize=8)
    axes[1].set_title("Level; parameters are ours, not TR 0702's", fontsize=9)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def figure_asiaex(result: dict, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = result["display"]
    ms, ref = d["ms"], d["ref"]
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True, constrained_layout=True)
    panels = (("flat", "No time spreading"), ("spread", "With time spreading (multi_bounce: sum)"),
              ("convolve", "With time spreading (multi_bounce: convolve)"))
    for ax, (key, title) in zip(axes, panels):
        p = d[key]
        for ping in p:
            ax.plot(ms, db10(ping / ref), color="C3", lw=0.4, alpha=0.6)
        ax.plot(ms, db10(p.mean(axis=0) / ref), color="C0", lw=1.2)
        ax.set_ylim(-30.0, 10.0)
        ax.set_xlim(0.0, 35.0)
        ax.set_ylabel("Power (dB re direct)")
        ax.set_title(f"sst_asiaex: {title}; 20 pings and their mean (simulated)", fontsize=9)
    for name, a in d["arrival"].items():
        axes[0].annotate(name, (a + 0.2, 4.0), fontsize=8, rotation=20)
    axes[-1].set_xlabel("Time after the first arrival (ms); depths and sound speed are ours, not TR 0702's")
    fig.savefig(path, dpi=110)
    plt.close(fig)


# =================================================================================================

def run(only=("pursuit", "asiaex", "checks")) -> dict:
    import time
    results = {}
    if "pursuit" in only:
        start = time.time()
        _quiet_run(PURSUIT)
        one = time.time() - start
        results["pursuit"] = pursuit()
        results["pursuit"]["seconds_one_ping"] = one
        results["pursuit"]["cross_pairs"] = {k: v for k, v in pursuit_doppler_of_cross_pairs().items() if not isinstance(v, np.ndarray)}
        results["pursuit"]["cross_pairs_no_bottom"] = {k: v for k, v in pursuit_doppler_of_cross_pairs(bottom=False).items()
                                                       if not isinstance(v, np.ndarray)}
        results["pursuit"]["seconds"] = time.time() - start
    if "asiaex" in only:
        results["asiaex"] = asiaex()
    if "checks" in only:
        results["checks"] = checks()
    return results


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", help="write the Markdown matrix here (default: stdout)")
    parser.add_argument("--json", help="write every measured number here")
    parser.add_argument("--figures", help="directory for sst_pursuit.png and sst_asiaex.png")
    parser.add_argument("--only", default="pursuit,asiaex,checks", help="parts to run")
    parser.add_argument("--no-run", action="store_true", help="emit the matrix without running anything")
    args = parser.parse_args(argv)
    results = {} if args.no_run else run(tuple(x for x in args.only.split(",") if x))
    text = markdown(matrix(results), results)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    if args.json:
        Path(args.json).write_text(json.dumps(_plain(results), indent=1), encoding="utf-8")
    if args.figures:
        out = Path(args.figures)
        out.mkdir(parents=True, exist_ok=True)
        if "pursuit" in results:
            figure_pursuit(results["pursuit"], out / "sst_pursuit.png")
        if "asiaex" in results:
            figure_asiaex(results["asiaex"], out / "sst_asiaex.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
