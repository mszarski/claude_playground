"""The sweep's Doppler bank in the live chain (D4 of docs/tasks/DOPPLER.md), opt-in: `chain.doppler`.

A long linear sweep loses matched-filter gain on an echo whose range changes (a quadratic phase error of pi B T |r| / c
at the ends of the pulse after the range-Doppler coupling has taken the frequency offset; ours, derived in
tests/python/test_doppler_bank.py). With `doppler: {mode: sweep_bank}` the chain compresses the record once per
range-rate hypothesis (the pulse time-scaled exactly), maps each with its beamformer, and keeps per map sample the
largest power over the hypotheses of that sample's beam; the detector's thresholds move by `threshold_db` (0 by default:
on the maximum the unchanged thresholds were measured to give fewer false detections than configured, libs/live
doppler.cpp bank_threshold_db); each detection carries the winning hypothesis as a coarse range rate.

The plan (hypotheses, which belong to each beam, the baseband's room, the threshold shift) is made by ONE function,
libs/live doppler.hpp `plan_doppler` (`core.live_doppler_plan`), for both engines; this module holds the Python engine's
part of the stage (the fold over hypotheses and the hypothesis of a detection), which libs/live chain.cpp does the same.

For D3 (a tone's filter bank and a CFAR per (beam, hypothesis)): the stage is `hypothesis_maps` (one map per hypothesis,
from a baseband record) and a combiner (`fold`, the maximum); D3 adds its own map maker (a short-time Fourier transform
per beam for a tone) and its own combiner (a detector run per hypothesis) beside them, and reuses `plan`'s per-beam
own-ship centring and the detection's range-rate fields.
"""

from __future__ import annotations

import numpy as np

from . import _gpusonar as core

MODES = ("off", "sweep_bank")
DEFAULTS = {
    "mode": "off",                  # "sweep_bank": the bank of time-scaled sweep replicas (an LFM transmit only)
    "max_range_rate_m_s": 3.0,      # target range rates (over ground, along each beam) spanned either side of the beam's own-ship rate
    "step_m_s": None,               # between hypotheses; None: c / (2 B T), the worst loss between two hypotheses 0.24 dB
    "hypotheses": None,             # an odd count per beam instead of max_range_rate_m_s
    "unknown_ownship_m_s": 2.0,     # added to the span on a ping whose record has no finite velocity (the vehicle's top speed)
    "threshold_db": None,           # dB every detector threshold (and min_snr_db) moves by; None: libs/live bank_threshold_db (0)
    "report": False,                # each detection's range_rate_m_s and range_rate_step_m_s on the report line (owner question 6)
}


def merged(cfg: dict | None) -> dict:
    """`chain.doppler` over DEFAULTS, its words and numbers checked as libs/live checks them."""
    d = dict(DEFAULTS, **(cfg or {}))
    if d["mode"] not in MODES:
        raise ValueError(f"chain.doppler.mode: {d['mode']!r} ({' or '.join(MODES)})")
    for key in ("max_range_rate_m_s", "unknown_ownship_m_s"):
        if not (np.isfinite(float(d[key])) and float(d[key]) >= 0.0):
            raise ValueError(f"chain.doppler.{key} = {d[key]}: a finite speed, 0 or more")
    if d["step_m_s"] is not None and not (np.isfinite(float(d["step_m_s"])) and float(d["step_m_s"]) > 0.0):
        raise ValueError(f"chain.doppler.step_m_s = {d['step_m_s']}: a speed above 0 (None: c / (2 B T))")
    if d["hypotheses"] is not None and (int(d["hypotheses"]) != d["hypotheses"] or int(d["hypotheses"]) < 1 or int(d["hypotheses"]) % 2 == 0):
        raise ValueError(f"chain.doppler.hypotheses = {d['hypotheses']}: an odd count (None: from max_range_rate_m_s)")
    if d["threshold_db"] is not None and not np.isfinite(float(d["threshold_db"])):
        raise ValueError(f"chain.doppler.threshold_db = {d['threshold_db']}: a finite number of dB (None: measured)")
    return d


def on(cfg: dict | None) -> bool:
    return merged(cfg)["mode"] != "off"


def cpp_options(cfg: dict | None) -> dict:
    """The flat doppler_* keys of libs/live (options.cpp, the binding): empty when the bank is off, so that every other
    line of the C++ options is the one it was."""
    d = merged(cfg)
    if d["mode"] == "off":
        return {}
    o = {"doppler_mode": d["mode"], "doppler_max_range_rate_m_s": float(d["max_range_rate_m_s"]),
         "doppler_unknown_ownship_m_s": float(d["unknown_ownship_m_s"]), "doppler_report": bool(d["report"])}
    if d["step_m_s"] is not None:
        o["doppler_step_m_s"] = float(d["step_m_s"])
    if d["hypotheses"] is not None:
        o["doppler_hypotheses"] = int(d["hypotheses"])
    if d["threshold_db"] is not None:
        o["doppler_threshold_db"] = float(d["threshold_db"])
    return o


def plan(record, bearings, cfg: dict | None) -> dict:
    """The bank for this record and the map's bearings (rad): libs/live plan_doppler, the one both engines use."""
    return core.live_doppler_plan(record, [float(b) for b in bearings], cpp_options(cfg))


def several(p: dict) -> bool:
    """More than the zero hypothesis alone: with only it the map is the chain's map without the bank."""
    return not (len(p["rates"]) == 1 and p["zero"] == 0)


def hypothesis_maps(record, base, p: dict, map_fn, backend=None):
    """(h, map_fn's result) for every hypothesis h of plan `p`: the zero one is `record` (compressed as the chain does
    without the bank), each other a copy of the baseband record `base` compressed with that hypothesis' time scale and
    read `p["shifts"][h]` s later (its own LFM coupling), so that an echo shows where the zero hypothesis shows it and the
    maximum over hypotheses does not smear it along the range-Doppler ridge (libs/live doppler.hpp, shifts)."""
    from .recon.fls import _set_state
    where = {} if backend is None else {"backend": backend}
    for h, scale in enumerate(p["scales"]):
        if h == p["zero"]:
            yield h, map_fn(record)
        else:
            trial = base.copy()
            core.pulse_compress(trial, window="none", doppler_scale=float(scale), **where)
            trial.samples.start_time = trial.samples.start_time + float(p["shifts"][h])
            _set_state(trial, compressed=True, replica_window="none")      # the beamformers must not compress it again
            yield h, map_fn(trial)


def fold(p: dict, h: int, fine: np.ndarray, best: np.ndarray | None, which: np.ndarray | None):
    """The running maximum over each beam's hypotheses (libs/live fold_hypothesis): `fine` [n_bearing, n] the map of
    hypothesis h; a beam that does not take h is left alone; a tie keeps the earlier hypothesis. Returns (best, which)."""
    fine = np.asarray(fine, dtype=np.float32)
    if best is None:
        best = np.full(fine.shape, -1.0, dtype=np.float32)
        which = np.full(fine.shape, -1, dtype=np.int32)
    first = np.asarray(p["first"])
    beams = (h >= first) & (h < first + int(p["per_beam"]))
    better = (fine > best) & beams[:, None]
    best[better] = fine[better]
    which[better] = h
    return best, which


def detection_hypothesis(best, which, fine_ranges, bearings, bearing: float, lo: float, hi: float) -> int:
    """The hypothesis of a detection: that of the strongest sample of the maximum map along its nearest bearing over
    [lo, hi] m (the whole row when nothing lies there); libs/live detection_hypothesis. -1 when there is none."""
    if not len(bearings) or best.shape[1] == 0:
        return -1
    ib = int(np.argmin(np.abs(np.asarray(bearings, dtype=float) - bearing)))
    r = np.asarray(fine_ranges, dtype=float)[:best.shape[1]]
    inside = np.nonzero((r >= lo) & (r <= hi))[0]
    row = best[ib]
    pick = int(inside[np.argmax(row[inside])]) if len(inside) else int(np.argmax(row))
    return int(which[ib, pick])
