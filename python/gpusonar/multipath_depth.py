"""Depth of a target from the timing of its own multipath: one row of hydrophones is enough.

A submerged target's echo arrives several times: out and back along the direct path, and with a reflection at the
sea surface on the way out, on the way back, or both (in shallow water also off the bottom). For flat boundaries
and straight rays (NumPy reference of libs/live multipath_depth.hpp; docs/PHYSICS_NOTES.md has the derivation):

    one way     R_d = sqrt(r^2 + (z_t - z)^2)      direct, amplitude 1 / R_d
                R_s = sqrt(r^2 + (z_t + z)^2)      surface image, amplitude mu / R_s, mu = -rho_c 10^(-L / 20)
                tau(z) = (R_s - R_d) / c  ~  2 z z_t / (r c)
    two way     projector at depth z_p, array at z_a:
                delay 0                      amplitude 1                 direct out, direct back
                tau(z_a)                     mu_rx                       direct out, surface back
                tau(z_p)                     mu_tx                       surface out, direct back
                tau(z_p) + tau(z_a)          mu_tx mu_rx                 surface both ways
                (z_p = z_a, mu = -1: the triplet +1, -2, +1 at 0, tau, 2 tau)

so the target's depth z_t follows from the delays, the range and the sonar's own depth, without the elevation angle.
The sound-speed profile does come into it: a mean gradient g bends both paths and, to first order, stretches the
surface echo's delay by 1 + r^2 g / (2 c (z_own + z_t)) (3% at 1 km, 30% at 3 km for 50 m and 120 m in isothermal
water). An assumed gradient is taken out (`gradient_per_s`) and what is not known of it goes into the sigma
(`gradient_sigma_per_s`), as for the elevation depth, which the same gradient moves by r^2 g / 2c: the multipath depth is
the less sensitive of the two by z_t / (z_own + z_t), not insensitive. The estimator does not pick arrivals: for every
candidate depth it predicts the whole pattern (delays, amplitudes, signs, and the carrier phase 2 pi fc tau of each
arrival) and correlates it with the compressed beam output around the detection,

    T(z, t0) = | sum_k conj(a_k) exp(+i 2 pi fc tau_k) y(t0 + tau_k) |^2 / sum_jk a_j conj(a_k) exp(-i 2 pi fc (tau_j - tau_k)) sinc(B (tau_j - tau_k))

(y: baseband, band B about fc, noise power sigma^2 per sample; the denominator is the pattern's energy), which is the
generalised likelihood ratio for "this pattern, unknown complex amplitude, unknown time t0 of the direct arrival" in
white in-band noise: L(z) = max_t0 T / sigma^2. The same statistic covers both regimes: arrivals further apart than
1 / B are resolved peaks ("delay"); closer than that they overlap and what is matched is their interference across
the band, the Lloyd-mirror ripple of the echo's spectrum with period 1 / tau = c r / (2 z z_t) ("ripple").
The single-arrival hypothesis is L_1 = max_t |y|^2 / sigma^2; a depth is reported when the pattern explains the data
better than one arrival by `min_gain` (in units of the noise power: a log-likelihood ratio).

L(z) has fringes: the carrier phase turns once for a depth change of c r / (2 fc (z_a + z_p)). Within a ping they are
real (a flat surface reflects with a known phase) and give the fine depth; from ping to ping the target's depth and
the surface move by more than a fringe, so along a track the fringe envelope (the maximum of L over each fringe) is
what is added up. Depth sigma is the posterior spread of exp(L) over depth, so neighbouring fringes and rival
depths widen it; a rival maximum within `ambiguity_drop` of the best sets the ambiguity flag.

When it cannot work, and what the report then says (`multipath_reason`):
    merged            target or sonar so shallow that tau B < `min_delay_bandwidth` at every depth it could be at: the
                      arrivals merge, the ripple period exceeds the band, and what is left is the surface cancellation
    no_surface_path   the surface reflects too little (`surface.loss_db`, e.g. bubbles at a high sea state, or roughness)
    (with a depth, as `multipath_note`, and the ambiguity flag set:)
    bottom_unmodelled the water depth is known, no `bottom` is configured, and a bottom-reflected path would arrive at a
                      grazing angle below `bottom_grazing_limit_deg`: the surface-only pattern may be fitting bottom echoes
    bottom_order      a path with one more boundary reflection than `bottom_order` would arrive inside the window with an
                      amplitude above `surface.min_amplitude`: raise `bottom_order`
    single_arrival    the pattern is no better than one arrival (no surface echo seen, or too little SNR for it)
    window            the window around the detection does not hold enough samples (edge of the record)
A target with several highlights along range is not multipath: its highlights are all of one sign and not at
0, tau, 2 tau; the test `tests/python/test_multipath_depth.py` measures how often it is taken for it.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np

from . import acoustics
from .lloyd import _images

DEFAULTS = {
    "max_depth_m": 500.0,            # deepest target depth searched (less when the water is shallower)
    "min_delay_bandwidth": 3.0,      # shallowest depth searched: where (tau_tx + tau_rx) x bandwidth is this
    "window_s": 0.25,                # longest two-way multipath spread searched (depths that spread more are not searched)
    "range_slack_m": 4.0,            # the detection's range may be this far from any one of the target's arrivals
    "oversample": 8,                 # samples of the beam window per 1 / bandwidth
    "fringe_points": 8,              # depths tried per carrier fringe
    "noise_cells": 128,              # 1 / bandwidth cells ahead of the earliest possible arrival that give the noise power
    "min_gain": 12.0,                # a depth needs L(z) - L_1 of at least this, per detection
    "ambiguity_drop": 3.0,           # a rival depth within this of the best log-likelihood: ambiguous
    "max_detections": 16,            # strongest detections examined per ping
    "min_snr_db": None,              # detections weaker than this are not examined (None: all)
    "surface": {"coefficient": -1.0, # flat-surface pressure reflection coefficient
                "rms_height_m": 0.0, # rms surface height: coherent reflection exp(-Gamma^2 / 2), Gamma = 2 k h sin(grazing)
                "loss_db": 0.0,      # further loss per surface bounce (bubbles: the named law of sim/surface.py, evaluated by the user)
                "min_amplitude": 0.5},    # a one-leg surface echo weaker than this (re the direct one): no_surface_path
    "bottom": None,                  # {speed, density, attenuation}: a fluid bottom's Rayleigh reflection, with the water depth
    "bottom_order": 2,               # boundary reflections per leg when a bottom is given (1: d, s, b; 2: also sb, bs)
    "bottom_grazing_limit_deg": 45.0,    # water depth known but no bottom given: a bottom path shallower than this makes the
                                     # depth ambiguous (`multipath_note: bottom_unmodelled`)
    "water_depth": None,             # m; None: the record's environment.water_depth (navigation's, else unknown)
    "surface_z": 0.0,                # z of the sea surface in the navigation's frame
    "own_depth_sigma_m": 0.0,        # the navigation's depth accuracy
    "model_sigma_fraction": 0.0,     # further relative error of the delays' model: sigma gains depth x this
    # Refraction: a mean vertical sound-speed gradient g bends both paths, and to first order the surface echo's delay
    # grows by the factor 1 + r^2 g / (2 c (z_own + z_t)) (docs/PHYSICS_NOTES.md; tools/multipath_depth_study.py).
    "gradient_per_s": None,          # assumed mean gradient (1/s; 0.017: isothermal water): taken out of the depth. None: not
    "gradient_sigma_per_s": None,    # how well it is known: sigma gains z_t r^2 x this / (2 c (z_own + z_t)). None: the prior
                                     # the elevation depth uses (elevation.gradient_sigma_per_s)
    "track": {"gate_m": 60.0,        # a detection within this of a track's position is that track's
              "step_m": 0.5,         # depth step of a track's accumulated likelihood
              "min_gain": 12.0,      # the accumulated gain a track's depth needs
              "memory_pings": None}, # None: all pings count alike; n: older ones fade with exp(-1 / n) per ping
    "fuse": {"consistency_sigma": 3.0},   # with a stacked-row depth: combined when they agree within this many sigma
}


@dataclass
class Geometry:
    """What one detection's fit needs to know."""

    range_m: float                   # slant range of the detection: half its two-way path, m
    array_depth: float               # m below the surface
    projector_depth: float           # m below the surface
    sound_speed: float
    f_centre: float
    bandwidth: float
    water_depth: float | None = None
    slack_m: float = 0.0             # how far range_m may be from any one arrival (0: the configuration's range_slack_m)


@dataclass
class Estimate:
    valid: bool = False
    depth: float = float("nan")      # m below the surface
    sigma: float = float("nan")      # m, one sigma, everything
    refraction_sigma: float = 0.0    # m: the part of sigma that is the unknown gradient (the same in every ping)
    gain: float = float("nan")       # L(depth) - L_1
    arrivals: int = 0                # arrivals of the pattern at that depth
    method: str = "none"             # "delay" (arrivals resolved), "ripple" (interference across the band), "none"
    ambiguous: bool = False
    alternatives: tuple = ()         # rival depths, m
    reason: str = ""                 # why there is no depth (module docstring)
    snr_db: float = float("nan")     # 10 log10 L_1: the strongest arrival over the noise
    # the likelihood itself, for a track to add up: fringe envelope L(z) - L_1 at `grid`
    grid: np.ndarray = field(default_factory=lambda: np.zeros(0), repr=False)
    envelope: np.ndarray = field(default_factory=lambda: np.zeros(0), repr=False)


def options(cfg: dict | None) -> dict:
    """`chain.multipath_depth` merged over DEFAULTS (unknown keys are an error)."""
    out = copy.deepcopy(DEFAULTS)
    for key, value in (cfg or {}).items():
        if key not in out:
            raise ValueError(f"multipath_depth: unknown key {key!r}; keys are {sorted(out)}")
        if isinstance(out[key], dict) and isinstance(value, dict):
            unknown = set(value) - set(out[key])
            if unknown:
                raise ValueError(f"multipath_depth.{key}: unknown keys {sorted(unknown)}; keys are {sorted(out[key])}")
            out[key].update(value)
        else:
            out[key] = value
    return out


def cpp_options(cfg: dict | None) -> dict:
    """`chain.multipath_depth` as the flat options of the C++ stage (libs/live MultipathOptions; `multipath_depth_<name>=value`
    arguments of build/bin/sonar_live). Both chain engines run that stage, so both read the configuration through here."""
    c = options(cfg)
    s, t = c["surface"], c["track"]
    o = {"max_depth": float(c["max_depth_m"]), "min_delay_bandwidth": float(c["min_delay_bandwidth"]), "window_s": float(c["window_s"]),
         "range_slack": float(c["range_slack_m"]), "oversample": int(c["oversample"]), "fringe_points": int(c["fringe_points"]),
         "noise_cells": int(c["noise_cells"]), "min_gain": float(c["min_gain"]), "ambiguity_drop": float(c["ambiguity_drop"]),
         "max_detections": int(c["max_detections"]), "surface_coefficient": float(s["coefficient"]), "surface_rms_height": float(s["rms_height_m"]),
         "surface_loss_db": float(s["loss_db"]), "surface_min_amplitude": float(s["min_amplitude"]), "bottom_order": int(c["bottom_order"]),
         "bottom_grazing_limit_deg": float(c["bottom_grazing_limit_deg"]),
         "water_depth": float(c["water_depth"] or 0.0), "surface_z": float(c["surface_z"]), "own_depth_sigma": float(c["own_depth_sigma_m"]),
         "model_sigma_fraction": float(c["model_sigma_fraction"]), "gradient": float(c["gradient_per_s"] or 0.0),
         "gradient_sigma": gradient_sigma(c), "track_gate": float(t["gate_m"]), "track_step": float(t["step_m"]),
         "track_min_gain": float(t["min_gain"]), "track_memory_pings": float(t["memory_pings"] or 0.0),
         "consistency_sigma": float(c["fuse"]["consistency_sigma"])}
    if c["min_snr_db"] is not None:
        o["min_snr_db"] = float(c["min_snr_db"])
    if c["bottom"] is not None:
        b = c["bottom"]
        o.update(bottom=1, bottom_speed=float(b["speed"]), bottom_density=float(b["density"]), bottom_attenuation=float(b.get("attenuation", 0.0)))
    return o


def gradient_sigma(cfg: dict) -> float:
    """`gradient_sigma_per_s`, or the elevation depth's prior when it is not given."""
    if cfg.get("gradient_sigma_per_s") is not None:
        return float(cfg["gradient_sigma_per_s"])
    from .elevation import DEFAULTS as EL
    return float(EL["gradient_sigma_per_s"])


def refraction(flat_depth: float, g: "Geometry", cfg: dict):
    """(depth, sigma): the depth whose surface delay in the assumed mean gradient equals the flat formula's at `flat_depth`
    (z (1 + K / (z_own + z)) = z_flat, K = r^2 g / 2c), and what the gradient's uncertainty leaves of it, m."""
    own = 0.5 * (g.array_depth + g.projector_depth)
    bend = g.range_m**2 / (2.0 * g.sound_speed)
    k = bend * float(cfg.get("gradient_per_s") or 0.0)
    b = own + k - flat_depth
    depth = flat_depth if k == 0.0 else 0.5 * (-b + np.sqrt(b * b + 4.0 * flat_depth * own))
    return float(depth), float(depth * bend * abs(gradient_sigma(cfg)) / (own + depth))


def geometry_dict(g: "Geometry") -> dict:
    return {"range_m": g.range_m, "array_depth": g.array_depth, "projector_depth": g.projector_depth, "sound_speed": g.sound_speed,
            "f_centre": g.f_centre, "bandwidth": g.bandwidth, "water_depth": g.water_depth or 0.0, "slack_m": g.slack_m}


class ChainStage:
    """The stage as the Python chain engine runs it: the same C++ class the C++ engine calls (libs/live MultipathDepth),
    fed from the engine's own detections and tracks; its JSON members go into the report's detections and tracks."""

    def __init__(self, cfg: dict | None):
        from . import _gpusonar as core
        self.cfg = options(cfg)
        self._stage = core.MultipathDepth(cpp_options(cfg))
        self.seconds = 0.0

    def annotate(self, record, report: dict, origin, hits, local, frame_origin, rows=None, surface_z: float = 0.0, water=None,
                 backend=None) -> None:
        """`hits`: the engine's detections (range, cone bearing, first_range, extent); `local`: what it gave the tracker (range,
        bearing about the frame origin, depth alternatives). Adds members to report["detections"] and report["tracks"]."""
        import json

        from .elevation import fold_depth
        mh = []
        for h, m, d in zip(hits, local, report["detections"]):
            item = {"range": float(h.range), "bearing": float(h.bearing), "x": float(m.range * np.sin(m.bearing)),
                    "y": float(m.range * np.cos(m.bearing)), "snr_db": float(h.snr_db)}
            for key in ("first_range", "extent"):
                value = getattr(h, key, None)
                if value is not None and np.isfinite(value):
                    item[key] = float(value)
            if d.get("depth_m") is not None:
                item.update(depth=float(d["depth_m"]) - surface_z, depth_sigma=float(d["depth_sigma_m"]),
                            depth_refraction_sigma=float(d.get("depth_refraction_sigma_m", 0.0)),
                            depth_alternatives=[float(fold_depth(z, water, surface_z)[0]) - surface_z for z in (m.depth_alternatives or [])])
            mh.append(item)
        mt = []
        for t in report["tracks"]:
            item = {"id": int(t["id"]), "x": float(t["y"] - frame_origin[1]), "y": float(t["x"] - frame_origin[0])}
            if t.get("depth_m") is not None:
                item.update(depth=float(t["depth_m"]) - surface_z, depth_sigma=float(t["depth_sigma_m"]))
            mt.append(item)
        kw = {} if backend is None else {"backend": backend}
        dets, tracks, self.seconds = self._stage.process(record, tuple(float(v) for v in origin), mh, mt, [list(map(int, r)) for r in (rows or [])], **kw)
        for d, text in zip(report["detections"], dets):
            if text:
                d.update(json.loads("{" + text + "}"))
        for t, text in zip(report["tracks"], tracks):
            if text:
                t.update(json.loads("{" + text + "}"))


# --- the predicted pattern ------------------------------------------------------------------------------------------------

def one_way(z_own: float, z_target: float, r_h, c: float, fc: float, surface: dict, water_depth=None, bottom=None, order: int = 1):
    """Path lengths [J, M] and amplitudes re the direct path [J, M] (complex) of the images joining a point at depth
    `z_own` and one at `z_target` a horizontal distance `r_h` [M] apart; the direct path is row 0. Spherical spreading,
    the surface's coefficient with roughness and loss, the bottom's Rayleigh coefficient at each image's grazing angle."""
    r_h = np.atleast_1d(np.asarray(r_h, dtype=float))
    depth = float(water_depth) if (bottom is not None and water_depth) else None
    lengths, amps = [], []
    r_d = np.sqrt(r_h**2 + (z_own - z_target) ** 2)
    for img, ns, nb in _images(float(z_target), depth, int(order) if depth is not None else 1):
        r_j = np.sqrt(r_h**2 + (z_own - img) ** 2)
        a = (r_d / r_j).astype(complex)
        if ns or nb:
            grazing = np.arcsin(np.clip(abs(z_own - img) / r_j, 0.0, 1.0))
            if ns:
                rho = acoustics.rough_coherent_reflection(grazing, float(surface["rms_height_m"]), fc, c)
                a = a * (float(surface["coefficient"]) * 10.0 ** (-float(surface["loss_db"]) / 20.0) * rho) ** ns
            if nb:
                from .sim.propagation import rayleigh_reflection
                a = a * rayleigh_reflection(grazing, c, float(bottom["speed"]), float(bottom["density"]), float(bottom.get("attenuation", 0.0))) ** nb
        lengths.append(r_j)
        amps.append(a)
    return np.array(lengths), np.array(amps)


def pattern(z_target: float, t0, g: Geometry, cfg: dict):
    """The two-way arrivals of a target at depth `z_target` whose direct echo has two-way time t0 [M] (s):
    delays after the direct one [K, M] (s) and amplitudes re the direct one [K, M] (complex). Arrival 0 is direct-direct."""
    t0 = np.atleast_1d(np.asarray(t0, dtype=float))
    z_mid = 0.5 * (g.array_depth + g.projector_depth)
    r_h = np.sqrt(np.maximum((0.5 * g.sound_speed * t0) ** 2 - (z_target - z_mid) ** 2, 1.0))
    args = (g.sound_speed, g.f_centre, cfg["surface"], g.water_depth, cfg["bottom"], cfg["bottom_order"])
    l_tx, a_tx = one_way(g.projector_depth, z_target, r_h, *args)
    l_rx, a_rx = one_way(g.array_depth, z_target, r_h, *args)
    delay = ((l_tx[:, None, :] - l_tx[0]) + (l_rx[None, :, :] - l_rx[0])) / g.sound_speed
    amp = a_tx[:, None, :] * a_rx[None, :, :]
    return delay.reshape(-1, t0.size), amp.reshape(-1, t0.size)


def pattern_energy(delay, amp, fc: float, band: float):
    """Energy of the pattern re one arrival of unit amplitude: sum_jk a_j conj(a_k) exp(-i 2 pi fc (tau_j - tau_k)) sinc(B (tau_j - tau_k)) [M]."""
    w = amp * np.exp(-2j * np.pi * fc * delay)
    out = np.sum(np.abs(w) ** 2, axis=0)
    for j in range(1, delay.shape[0]):
        out = out + 2.0 * np.sum((w[j] * np.conj(w[:j])).real * np.sinc(band * (delay[j] - delay[:j])), axis=0)
    return out


def surface_delay(z_target, g: Geometry):
    """tau(z_p) + tau(z_a): the delay of the echo reflected at the surface both ways behind the direct one, s (flat, straight)."""
    z = np.asarray(z_target, dtype=float)
    z_mid = 0.5 * (g.array_depth + g.projector_depth)
    r2 = np.maximum(g.range_m**2 - (z - z_mid) ** 2, 1.0)
    return sum(np.sqrt(r2 + (z + zo) ** 2) - np.sqrt(r2 + (z - zo) ** 2) for zo in (g.projector_depth, g.array_depth)) / g.sound_speed


def depth_grid(g: Geometry, cfg: dict):
    """(depths searched [Z], depths per fringe, deepest depth the window allows or None): from where the surface echo
    separates by `min_delay_bandwidth` / B down to `max_depth_m` (or the bottom), `fringe_points` per carrier fringe."""
    z_hi = float(cfg["max_depth_m"])
    if g.water_depth:
        z_hi = min(z_hi, float(g.water_depth))
    z_hi = min(z_hi, 0.5 * (g.array_depth + g.projector_depth) + 0.95 * g.range_m)
    probe = np.linspace(0.0, z_hi, 4001)
    tau = surface_delay(probe, g)
    ok = (tau * g.bandwidth >= float(cfg["min_delay_bandwidth"])) & (tau <= float(cfg["window_s"]))
    if not ok.any():
        return np.zeros(0), 1, None
    z_lo, z_top = float(probe[ok][0]), float(probe[ok][-1])
    capped = z_top if z_top < z_hi - 1e-9 else None
    # one fringe: the carrier phase of the surface-surface echo turns once
    dtau_dz = np.gradient(tau, probe)
    fringe = 1.0 / (g.f_centre * float(np.max(dtau_dz[ok])))
    n = int(cfg["fringe_points"])
    step = fringe / n
    count = int(np.ceil((z_top - z_lo) / step / n)) * n
    return z_lo + step * (np.arange(max(count, n)) + 0.5), n, capped


# --- the beam window around a detection -----------------------------------------------------------------------------------

def window_times(g: Geometry, cfg: dict, spread: float):
    """Two-way times re the detection's own of the samples the fit needs (uniform), and the number of leading ones that
    hold only noise: `noise_cells` cells, a guard, then from the earliest the direct arrival can be to the latest echo."""
    dt = 1.0 / (float(cfg["oversample"]) * g.bandwidth)
    slack = 2.0 * float(g.slack_m or cfg["range_slack_m"]) / g.sound_speed + 2.0 / g.bandwidth
    guard = 4.0 / g.bandwidth
    n_noise = int(cfg["noise_cells"]) * int(cfg["oversample"])
    n_before = int(np.ceil((spread + slack + guard) / dt))
    n_after = int(np.ceil((spread + slack + guard) / dt))
    return dt * np.arange(-(n_noise + n_before), n_after + 1), n_noise


def beam_window(record, bearing: float, range_m: float, t_rel, rows=None, backend=None):
    """Compressed output of the array steered at cone `bearing` (rad from broadside) at two-way times `t_rel` (s) about the
    two-way time of slant range `range_m`, as baseband samples (carrier removed with respect to `t_rel`: an arrival
    delayed by tau carries the phase -2 pi fc tau). These are the chain's own beam samples (`core.live_beam_samples`),
    unweighted; with `rows` (stacked rows: lists of channels) the rows' samples are added. The record is conditioned
    (`recon.prepare`). This is the one thing the estimator asks of the chain."""
    from . import _gpusonar as core
    from . import sync
    t_rel = np.asarray(t_rel, dtype=float)
    ranges = float(range_m) + 0.5 * record.environment.sound_speed * t_rel
    kw = {} if backend is None else {"backend": backend}
    if rows is None or len(rows) < 2:
        y = np.asarray(core.live_beam_samples(record, ranges, float(bearing), False, **kw)).ravel()
    else:
        y = sum(np.asarray(core.live_beam_samples(sync.select_channels(record, r), ranges, float(bearing), False, **kw)).ravel() for r in rows)
    s = record.samples
    w = record.transmit.waveform
    fc = s.center_frequency if s.is_complex() and s.center_frequency > 0.0 else 0.5 * (w.f_start + w.f_stop)
    return y * np.exp(-2j * np.pi * fc * t_rel)


# --- the fit --------------------------------------------------------------------------------------------------------------

def _interp(y, u):
    """y (complex) at fractional sample positions u, linearly; 0 outside."""
    i = np.floor(u).astype(np.int64)
    f = u - i
    ok = (i >= 0) & (i < y.size - 1)
    i = np.where(ok, i, 0)
    return np.where(ok, y[i] * (1.0 - f) + y[i + 1] * f, 0.0)


def noise_power(y, n_noise: int) -> float:
    """Mean noise power per sample from the leading samples: median |y|^2 / ln 2 (an echo in them does not move a median)."""
    return float(np.median(np.abs(y[:n_noise]) ** 2) / np.log(2.0))


ANCHOR = 0.45      # an arrival at least this strong re the pattern's strongest may be the echo's strongest sample
ANCHOR_CELLS = 1.5  # ... and is tried within this many 1 / B of it


def likelihood(y, t_rel, n_noise: int, g: Geometry, cfg: dict, depths=None, per_fringe: int | None = None):
    """L(z) = max_t0 T(z, t0) / sigma^2 at each depth of the grid (module docstring) and what goes with it:
    {depths, L, t0 (of the maximum, re the detection), L1, sigma2, per_fringe}.

    The search over t0 is anchored: the strongest sample of the echo (within the range slack of the detection) is one of
    the pattern's strong arrivals (those within ANCHOR of its strongest), so for each depth only the t0 that put such an
    arrival within ANCHOR_CELLS / B of that sample are tried. L_1 is that sample's power over the noise."""
    if depths is None:
        depths, per_fringe, _ = depth_grid(g, cfg)
    y = np.asarray(y, dtype=complex)
    t_rel = np.asarray(t_rel, dtype=float)
    dt = float(t_rel[1] - t_rel[0])
    sigma2 = noise_power(y, n_noise)
    slack = 2.0 * float(g.slack_m or cfg["range_slack_m"]) / g.sound_speed + 2.0 / g.bandwidth
    near = np.nonzero(np.abs(t_rel) <= slack + 0.5 * dt)[0]
    k_pk = int(near[np.argmax(np.abs(y[near]) ** 2)])
    t_pk = float(t_rel[k_pk])
    step = 2                                                       # t0 on every second sample: 4 per 1 / B
    half = int(np.ceil(ANCHOR_CELLS / g.bandwidth / (dt * step)))
    t_det = 2.0 * g.range_m / g.sound_speed
    out_l, out_t = np.zeros(len(depths)), np.zeros(len(depths))
    for i, z in enumerate(depths):
        delay0, amp0 = pattern(float(z), t_det + t_pk, g, cfg)
        mag = np.abs(amp0[:, 0])
        ms = set()
        for k in np.nonzero(mag >= ANCHOR * mag.max())[0]:
            t0k = t_pk - float(delay0[k, 0])
            for _ in range(3):                                     # the delay depends (weakly) on t0 itself
                t0k = t_pk - float(pattern(float(z), t_det + t0k, g, cfg)[0][k, 0])
            m0 = int(np.round(t0k / (dt * step)))
            ms.update(range(m0 - half, m0 + half + 1))
        t0 = dt * step * np.array(sorted(ms), dtype=float)
        delay, amp = pattern(float(z), t_det + t0, g, cfg)
        u = (t0[None, :] + delay - t_rel[0]) / dt
        rake = np.sum(np.conj(amp) * np.exp(2j * np.pi * g.f_centre * delay) * _interp(y, u), axis=0)
        t = np.abs(rake) ** 2 / np.maximum(pattern_energy(delay, amp, g.f_centre, g.bandwidth), 1e-12)
        m = int(np.argmax(t))
        out_l[i], out_t[i] = t[m] / sigma2, t0[m]
    return {"depths": np.asarray(depths, dtype=float), "L": out_l, "t0": out_t, "L1": float(np.abs(y[k_pk]) ** 2 / sigma2),
            "sigma2": sigma2, "per_fringe": int(per_fringe or 1)}


def fringe_envelope(depths, values, per_fringe: int):
    """The largest value in each run of `per_fringe` depths, and the depth it is at: (cell centres, maxima, depths of the maxima)."""
    n = len(depths) // per_fringe
    d = np.asarray(depths[:n * per_fringe]).reshape(n, per_fringe)
    v = np.asarray(values[:n * per_fringe]).reshape(n, per_fringe)
    k = np.argmax(v, axis=1)
    return d.mean(axis=1), v[np.arange(n), k], d[np.arange(n), k]


def posterior(grid, gain, drop: float, lobe: float):
    """(index of the best, sigma, rival depths) of a log-likelihood `gain` on `grid`: sigma is the spread of exp(gain)
    about the best; a rival is a local maximum more than `lobe` (m) from the best and within `drop` of it."""
    b = int(np.argmax(gain))
    p = np.exp(gain - gain[b])
    sigma = float(np.sqrt(np.sum(p * (grid - grid[b]) ** 2) / np.sum(p)))
    rivals = [float(grid[k]) for k in range(1, len(grid) - 1)
              if abs(grid[k] - grid[b]) > lobe and gain[k] >= gain[b] - drop and gain[k] >= gain[k - 1] and gain[k] > gain[k + 1]]
    return b, sigma, rivals


def envelope_lobe(z: float, g: Geometry) -> float:
    """Depth change that moves the surface-surface echo by one resolution cell 1 / B, m: the width of the envelope's main lobe."""
    d = max(0.01 * z, 0.05)
    slope = float(surface_delay(z + d, g) - surface_delay(z - d, g)) / (2.0 * d)
    return 1.0 / (g.bandwidth * max(slope, 1e-12))


def model_sigma(z: float, g: Geometry, cfg: dict) -> float:
    """What the fit cannot know, m: z_t ~ tau r c / (2 z_own), so own-depth error and the model's relative delay error scale it."""
    own = 0.5 * (g.array_depth + g.projector_depth)
    return float(z * np.hypot(float(cfg["own_depth_sigma_m"]) / max(own, 1e-6), float(cfg["model_sigma_fraction"])))


def estimate(y, t_rel, n_noise: int, g: Geometry, cfg: dict) -> Estimate:
    """Target depth from one beam window (`beam_window` at `window_times`)."""
    est = Estimate()
    s = cfg["surface"]
    if abs(float(s["coefficient"])) * 10.0 ** (-float(s["loss_db"]) / 20.0) < float(s["min_amplitude"]):
        est.reason = "no_surface_path"
        return est
    depths, per_fringe, _ = depth_grid(g, cfg)
    if len(depths) == 0:
        est.reason = "merged"
        return est
    if len(y) < n_noise + 8 or not np.any(np.abs(y[:n_noise]) > 0.0):
        est.reason = "window"
        return est
    fit = likelihood(y, t_rel, n_noise, g, cfg, depths, per_fringe)
    est.snr_db = float(10.0 * np.log10(max(fit["L1"], 1e-30)))
    grid, env, at = fringe_envelope(fit["depths"], fit["L"] - fit["L1"], per_fringe)
    est.grid, est.envelope = grid, env
    lobe = envelope_lobe(float(grid[int(np.argmax(env))]), g)
    b, sigma, rivals = posterior(grid, env, float(cfg["ambiguity_drop"]), lobe)
    est.gain = float(env[b])
    # surface path too weak at the depth found (roughness at its grazing angle)
    delay, amp = pattern(float(at[b]), 2.0 * g.range_m / g.sound_speed, g, cfg)
    est.arrivals = int(np.count_nonzero(np.abs(amp[:, 0]) >= 0.1))
    if est.gain < float(cfg["min_gain"]):
        est.reason = "single_arrival"
        return est
    if abs(amp[1, 0]) < float(s["min_amplitude"]):            # the one-leg surface echo (roughness at its grazing angle)
        est.reason = "no_surface_path"
        return est
    cell = float(grid[1] - grid[0]) if len(grid) > 1 else 0.0
    est.valid = True
    flat_depth = float(at[b])
    # resolved: each leg's surface echo stands 1 / B clear of the direct one
    est.method = "delay" if float(surface_delay(flat_depth, g)) * g.bandwidth >= 4.0 else "ripple"
    est.depth, est.refraction_sigma = refraction(flat_depth, g, cfg)
    est.sigma = float(np.sqrt(sigma**2 + cell**2 / 12.0 + model_sigma(est.depth, g, cfg) ** 2 + est.refraction_sigma**2))
    est.ambiguous = bool(rivals)
    est.alternatives = tuple(rivals[:4])
    # Paths the pattern leaves out but the water may hold: the depth is then one explanation among others.
    if g.water_depth and cfg["bottom"] is None:
        z_mid = 0.5 * (g.array_depth + g.projector_depth)
        if np.degrees(np.arctan2(2.0 * g.water_depth - z_mid - flat_depth, g.range_m)) < float(cfg["bottom_grazing_limit_deg"]):
            est.ambiguous, est.reason = True, "bottom_unmodelled"
    elif g.water_depth:
        more = dict(cfg, bottom_order=int(cfg["bottom_order"]) + 1)
        d2, a2 = pattern(flat_depth, 2.0 * g.range_m / g.sound_speed, g, more)
        legs, had = 1 + 2 * more["bottom_order"], 1 + 2 * int(cfg["bottom_order"])
        fresh = np.zeros(legs, dtype=bool)
        fresh[[had // 2 + 1, legs - 1]] = True                    # legs come as d, s.., b..: the last of each run is the new order
        new = (fresh[:, None] | fresh[None, :]).ravel() & (d2[:, 0] <= float(cfg["window_s"]))
        if new.any() and np.abs(a2[new, 0]).max() >= float(s["min_amplitude"]):
            est.ambiguous, est.reason = True, "bottom_order"
    return est


def geometry(record, origin, range_m: float, cfg: dict, first_range: float = float("nan"), extent: float = float("nan")) -> Geometry:
    """The fit's geometry for a detection at slant range `range_m` from `origin` (the array's centre, world). `first_range`,
    `extent`: where a merging detector says the detection's arrivals lie; the fit is then centred on that stretch."""
    from .elevation import tx_position
    w = record.transmit.waveform
    f_lo, f_hi = min(w.f_start, w.f_stop), max(w.f_start, w.f_stop)
    water = cfg.get("water_depth") or record.environment.water_depth
    water = float(water) if water and np.isfinite(water) and water > 0.0 else None
    sz = float(cfg["surface_z"])
    slack = 0.0
    if np.isfinite(first_range) and np.isfinite(extent) and extent >= 0.0:
        range_m, slack = first_range + 0.5 * extent, float(cfg["range_slack_m"]) + 0.5 * extent
    return Geometry(range_m=float(range_m), array_depth=float(origin[2]) - sz, projector_depth=float(tx_position(record)[2]) - sz,
                    sound_speed=float(record.environment.sound_speed), f_centre=0.5 * (f_lo + f_hi), bandwidth=(f_hi - f_lo) or 1.0 / w.duration,
                    water_depth=water, slack_m=slack)


def estimate_detection(record, origin, bearing: float, range_m: float, cfg: dict, first_range: float = float("nan"),
                       extent: float = float("nan"), rows=None, backend=None) -> Estimate:
    """Target depth for one detection of a conditioned record: slant range `range_m` at cone `bearing` from `origin`."""
    g = geometry(record, origin, range_m, cfg, first_range, extent)
    if g.array_depth <= 0.0 or g.projector_depth <= 0.0:
        return Estimate(reason="merged")
    depths, _, _ = depth_grid(g, cfg)
    if len(depths) == 0:
        return Estimate(reason="merged")
    t_rel, n_noise = window_times(g, cfg, float(surface_delay(depths[-1], g)))
    s = record.samples
    t_abs = 2.0 * g.range_m / g.sound_speed + t_rel
    if t_abs[0] < s.start_time or t_abs[-1] > s.start_time + (s.n_samples - 1) / s.sample_rate - record.transmit.waveform.duration:
        return Estimate(reason="window")
    return estimate(beam_window(record, bearing, g.range_m, t_rel, rows, backend), t_rel, n_noise, g, cfg)


# --- along a track --------------------------------------------------------------------------------------------------------

class TrackDepth:
    """A track's accumulated fringe envelope: the sum over pings of L(z) - L_1 on a fixed depth grid."""

    def __init__(self, cfg: dict):
        t = cfg["track"]
        self.step = float(t["step_m"])
        self.grid = self.step * (np.arange(int(np.ceil(float(cfg["max_depth_m"]) / self.step))) + 0.5)
        self.total = np.zeros(len(self.grid))
        self.pings = 0
        self.fade = 1.0 if not t["memory_pings"] else float(np.exp(-1.0 / float(t["memory_pings"])))
        self.cfg = cfg
        self.lobe = 0.0
        self.geometry = None
        self.cell = 0.0                      # the coarsest fringe cell among the pings added, m

    def add(self, est: Estimate, g: Geometry) -> None:
        if len(est.grid) < 2:
            return
        # depths this ping did not search count as "no better than one arrival"
        cell = float(est.grid[1] - est.grid[0])
        if cell > self.step:
            value = np.interp(self.grid, est.grid, est.envelope, left=0.0, right=0.0)
        else:                                    # finer than the track's grid: the largest in each of the track's cells
            k = np.floor(est.grid / self.step).astype(int)
            ok = (k >= 0) & (k < len(self.grid))
            low = np.full(len(self.grid), -np.inf)
            np.maximum.at(low, k[ok], est.envelope[ok])
            value = np.where(np.isfinite(low), low, 0.0)
        self.total = self.fade * self.total + value
        self.pings += 1
        self.cell = max(self.cell, cell)
        self.lobe = envelope_lobe(float(self.grid[int(np.argmax(self.total))]), g)
        self.geometry = g

    def result(self) -> Estimate:
        est = Estimate(arrivals=self.pings)
        if self.pings == 0:
            est.reason = "single_arrival"
            return est
        b, sigma, rivals = posterior(self.grid, self.total, float(self.cfg["ambiguity_drop"]), self.lobe)
        est.gain = float(self.total[b])
        if est.gain < float(self.cfg["track"]["min_gain"]):
            est.reason = "single_arrival"
            return est
        est.valid = True
        est.depth, est.refraction_sigma = refraction(float(self.grid[b]), self.geometry, self.cfg)
        # half a cell of the track's grid, and the pings' own fringe cell (the envelope has no finer structure); the
        # gradient's part does not average down over pings
        est.sigma = float(np.sqrt(sigma**2 + self.step**2 / 4.0 + self.cell**2 / 12.0 + model_sigma(est.depth, self.geometry, self.cfg) ** 2
                                  + est.refraction_sigma**2))
        est.ambiguous, est.alternatives = bool(rivals), tuple(rivals[:4])
        est.method = "track"
        return est


# --- with a stacked-row depth ---------------------------------------------------------------------------------------------

def fuse(depth_mp: float, sigma_mp: float, depth_el: float, sigma_el: float, alternatives=(), consistency_sigma: float = 3.0,
         refraction_mp: float = 0.0, refraction_el: float = 0.0) -> dict:
    """The multipath depth with the depth from the rows' elevation. `alternatives`: the depths of the other candidate
    elevations (two rows give several). The candidate nearest the multipath depth is taken; when it agrees with it
    within `consistency_sigma` sigma the two are combined, else both stand and `consistent` is False. `refraction_*`: the
    parts of the two sigmas that are the unknown gradient: one error in both, so only the rest is averaged down by
    inverse variance and the gradient's part is carried through with the weights.
    {depth, sigma, consistent, selected (index into [depth_el, *alternatives])}."""
    cands = [float(depth_el)] + [float(a) for a in alternatives]
    k = int(np.argmin([abs(c - depth_mp) for c in cands]))
    d = cands[k]
    ok = abs(d - depth_mp) <= consistency_sigma * float(np.hypot(sigma_mp, sigma_el))
    if not ok:
        return {"depth": float(depth_mp), "sigma": float(sigma_mp), "consistent": False, "selected": k}
    s1 = np.sqrt(max(sigma_mp**2 - refraction_mp**2, 1e-12))
    s2 = np.sqrt(max(sigma_el**2 - refraction_el**2, 1e-12))
    w1, w2 = 1.0 / s1**2, 1.0 / s2**2
    common = (w1 * refraction_mp + w2 * refraction_el) / (w1 + w2)
    return {"depth": float((w1 * depth_mp + w2 * d) / (w1 + w2)), "sigma": float(np.sqrt(1.0 / (w1 + w2) + common**2)), "consistent": True, "selected": k}


# --- a bound --------------------------------------------------------------------------------------------------------------

def depth_bound(z_target: float, g: Geometry, cfg: dict, snr_direct: float, n_f: int = 513) -> float:
    """Cramer-Rao bound on the target depth (m, one sigma) for the pattern in white in-band noise, the direct arrival's
    time and the complex amplitude unknown: `snr_direct` is the direct arrival's peak power over the noise power at the
    matched filter's output (linear). From the Fisher information of s(f) = sum_k a_k exp(-i 2 pi f tau_k(z)) exp(-i 2 pi f t0)
    over the band, with the amplitude projected out: J = 2 snr / F  Re[D^H (I - s s^H / s^H s) D], D = [ds/dz, ds/dt0]."""
    f = g.f_centre + g.bandwidth * (np.arange(n_f) / (n_f - 1) - 0.5)
    t_det = 2.0 * g.range_m / g.sound_speed

    def s_of(z, t0):
        delay, amp = pattern(z, t_det + t0, g, cfg)
        return np.sum(amp[:, 0, None] * np.exp(-2j * np.pi * f[None, :] * (delay[:, 0, None] + t0)), axis=0)

    dz, dt = 1e-3, 1e-8
    s0 = s_of(z_target, 0.0)
    d = np.stack([(s_of(z_target + dz, 0.0) - s_of(z_target - dz, 0.0)) / (2 * dz), (s_of(z_target, dt) - s_of(z_target, -dt)) / (2 * dt)], axis=1)
    d = d - np.outer(s0, np.conj(s0) @ d) / (np.conj(s0) @ s0)
    j = 2.0 * snr_direct / n_f * (np.conj(d).T @ d).real
    return float(np.sqrt(np.linalg.inv(j)[0, 0]))
