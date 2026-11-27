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
COMBINES = ("max", "cfar")
BATCHED = ("auto", "on", "off")
DEFAULTS = {
    "mode": "off",                  # "sweep_bank": the bank of time-scaled sweep replicas (an LFM transmit only)
    "max_range_rate_m_s": 3.0,      # target range rates (over ground, along each beam) spanned either side of the beam's own-ship rate
    "step_m_s": None,               # between hypotheses; None: c / (2 B T), the worst loss between two hypotheses 0.24 dB
    "hypotheses": None,             # an odd count per beam instead of max_range_rate_m_s
    "unknown_ownship_m_s": 2.0,     # added to the span on a ping whose record has no finite velocity (the vehicle's top speed)
    "combine": "max",               # "max": the largest power over hypotheses, then the detector (D4: loses margin on every scene);
                                    # "cfar": each (beam, hypothesis) normalised by its own background first, the largest
                                    # normalised value tested at the budget / hypotheses per beam (D3; libs/live doppler.hpp)
    "threshold_db": None,           # dB every detector threshold (and min_snr_db) moves by; None: libs/live bank_threshold_db (0)
    "report": False,                # each detection's range_rate_m_s and range_rate_step_m_s on the report line (owner question 6)
    # D8 (libs/live doppler_batched.hpp): where the bank's hypotheses and the tone's filter bank are computed. "auto": the
    # batched formulation on the cuda backend (every hypothesis on the device from one upload of the baseband record to
    # the combined map), the per-hypothesis loop on the host (as before D8); "on": the batched formulation on either
    # backend (on the host the same kernels on CPU threads: how it is tested without a device); "off": the loop on either.
    "batched": "auto",
    "batch": None,                  # hypotheses one batch holds; None: as many as the device's free memory allows
    # The tone's Doppler filter bank (D3 part B; libs/live doppler.hpp tone_stage), beside the sweep path, on with mode off too.
    "tone": "off",                  # "bank": the tone path on (a CW record, or a train whose tone segment is named below)
    "tone_frequency_hz": None,      # Hz; None: the record's (a CW record); a train needs it (its segment table)
    "tone_start_s": 0.0,            # s from the start of the pulse to the tone's
    "tone_duration_s": None,        # s; None: the record's (CW)
    "tone_window_alpha": None,      # the tone's Tukey taper (1: Hann); None: the record's (CW)
    "tone_max_range_rate_m_s": None,     # m/s either side of each beam's own-ship rate; None: max_range_rate_m_s
    "tone_step_m_s": None,          # None: c / (4 f T), half the resolution
    "tone_hop": 0.25,               # lags this fraction of the tone apart
    "tone_notch_m_s": None,         # half-width about each beam's own-ship rate not tested; None: own-motion span + notch cells
    "tone_notch_cells": 3.0,        # resolution cells in the derived notch
    "tone_guard": None,             # hypotheses either side out of a cell's background; None: 2 resolution cells
    "tone_train": 32,               # hypotheses either side in a cell's background
    "tone_false_alarms_per_ping": None,  # None: cfar.false_alarms_per_ping
    "tone_gate_range_m": None,      # a tone and a sweep detection this close are one; None: c T / 4
    "tone_gate_bearing_deg": None,  # ... and this close in bearing; None: 2 degrees
    "tone_range_sigma_m": None,     # the tracker's range sigma for a tone-only detection; None: c T / 8
    "tone_sidelobe_db": None,       # None: -25 dB with the Hann aperture, -7 dB rectangular
    "tone_leak_margin_db": 10.0,    # a cell no more than this above a stronger cell's leak through the tone's spectrum is dropped
}
TONES = ("off", "bank")
TONE_KEYS = ("frequency_hz", "start_s", "duration_s", "window_alpha", "max_range_rate_m_s", "step_m_s", "hop", "notch_m_s", "notch_cells",
             "guard", "train", "false_alarms_per_ping", "gate_range_m", "gate_bearing_deg", "range_sigma_m", "sidelobe_db",
             "leak_margin_db")


def merged(cfg: dict | None) -> dict:
    """`chain.doppler` over DEFAULTS, its words and numbers checked as libs/live checks them."""
    d = dict(DEFAULTS, **(cfg or {}))
    if d["mode"] not in MODES:
        raise ValueError(f"chain.doppler.mode: {d['mode']!r} ({' or '.join(MODES)})")
    if d["combine"] not in COMBINES:
        raise ValueError(f"chain.doppler.combine: {d['combine']!r} ({' or '.join(COMBINES)})")
    for key in ("max_range_rate_m_s", "unknown_ownship_m_s"):
        if not (np.isfinite(float(d[key])) and float(d[key]) >= 0.0):
            raise ValueError(f"chain.doppler.{key} = {d[key]}: a finite speed, 0 or more")
    if d["step_m_s"] is not None and not (np.isfinite(float(d["step_m_s"])) and float(d["step_m_s"]) > 0.0):
        raise ValueError(f"chain.doppler.step_m_s = {d['step_m_s']}: a speed above 0 (None: c / (2 B T))")
    if d["hypotheses"] is not None and (int(d["hypotheses"]) != d["hypotheses"] or int(d["hypotheses"]) < 1 or int(d["hypotheses"]) % 2 == 0):
        raise ValueError(f"chain.doppler.hypotheses = {d['hypotheses']}: an odd count (None: from max_range_rate_m_s)")
    if d["batched"] not in BATCHED:
        raise ValueError(f"chain.doppler.batched: {d['batched']!r} ({' or '.join(BATCHED)})")
    if d["batch"] is not None and (int(d["batch"]) != d["batch"] or int(d["batch"]) < 1):
        raise ValueError(f"chain.doppler.batch = {d['batch']}: hypotheses, 1 or more (None: as memory allows)")
    if d["tone"] not in TONES:
        raise ValueError(f"chain.doppler.tone: {d['tone']!r} ({' or '.join(TONES)})")
    if not (0.0 < float(d["tone_hop"]) <= 1.0):
        raise ValueError(f"chain.doppler.tone_hop = {d['tone_hop']}: a fraction of the tone above 0, at most 1")
    if int(d["tone_train"]) < 1 or (d["tone_guard"] is not None and int(d["tone_guard"]) < 0):
        raise ValueError("chain.doppler.tone_train: 1 or more; tone_guard: None (derived) or 0 or more")
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
    if d["combine"] != "max":
        o["doppler_combine"] = d["combine"]
    o.update(_batched_options(d))
    return o


def _batched_options(d: dict) -> dict:
    """The flat doppler_batched / doppler_batch keys (D8), only when not the defaults: every other line as it was."""
    o = {}
    if d["batched"] != "auto":
        o["doppler_batched"] = d["batched"]
    if d["batch"] is not None:
        o["doppler_batch"] = int(d["batch"])
    return o


def batched(cfg: dict | None, backend) -> bool:
    """Whether the sweep bank runs by its batched formulation (D8): `batched: on`, or `auto` on the cuda backend."""
    b = merged(cfg)["batched"]
    return b == "on" or (b == "auto" and backend == core.Backend.Cuda)


def batched_maps(base, bearings, cfg: dict | None, hann: bool, range_cell_m: float, blank_range, guard_m: float, train_m: float,
                 pfa: float, backend) -> dict:
    """The sweep bank's combined map by its batched formulation from the baseband record `base` (before compression):
    libs/live sweep_bank_stage (core.live_sweep_bank), the function the C++ chain calls. On the device with the cuda
    backend, on CPU threads otherwise. Returns fine, which, cells, level (cfar; None with max), fine_ranges, ranges, k."""
    return core.live_sweep_bank(base, [float(b) for b in bearings], cpp_options(cfg), hann=bool(hann), range_cell_m=float(range_cell_m),
                                blank_range=float("nan") if blank_range is None else float(blank_range), guard_m=float(guard_m),
                                train_m=float(train_m), pfa=float(pfa), backend=backend)


def tone_on(cfg: dict | None) -> bool:
    return merged(cfg)["tone"] != "off"


def tone_options(cfg: dict | None) -> dict:
    """The flat doppler_tone_* keys of libs/live (options.cpp, the binding): empty when the tone path is off."""
    d = merged(cfg)
    if d["tone"] == "off":
        return {}
    o = {"doppler_tone": d["tone"], "doppler_max_range_rate_m_s": float(d["max_range_rate_m_s"]),
         "doppler_unknown_ownship_m_s": float(d["unknown_ownship_m_s"]), "doppler_report": bool(d["report"])}
    for key in TONE_KEYS:
        v = d["tone_" + key]
        if v is not None:
            o["doppler_tone_" + key] = int(v) if key in ("guard", "train") else float(v)
    o.update(_batched_options(d))
    return o


def all_options(cfg: dict | None) -> dict:
    """Every flat doppler_* key of libs/live for this configuration (the sweep bank's and the tone's)."""
    return dict(cpp_options(cfg), **tone_options(cfg))


def tone_stage(record, bearings, cfg: dict | None, hann: bool, budget: float, backend=None) -> dict:
    """The tone path on a complex baseband record before compression: libs/live tone_stage, the one both engines use."""
    where = {} if backend is None else {"backend": backend}
    return core.live_tone_stage(record, [float(b) for b in bearings], tone_options(cfg), hann=bool(hann), default_budget=float(budget), **where)


def tone_plan_allowance(record, bearings, cfg: dict | None, budget: float) -> tuple[bool, float]:
    """(the tone path runs on this record, the baseband allowance it needs): libs/live plan_tone."""
    p = core.live_tone_plan(record, [float(b) for b in bearings], tone_options(cfg), float(budget))
    return bool(p["on"]), float(p["baseband_max_range_rate"])


def merge_tone(sweep: list, stage: dict) -> list:
    """libs/live merge_tone: tone detections, strongest first, each give their range rate to the strongest sweep detection
    not yet taken within the gates (doppler_path 2); those that find none come back as detections of their own (path 1,
    the plan's range sigma). Sweep detections are changed in place."""
    from .detect import Detection
    p = stage["plan"]
    taken = [False] * len(sweep)
    alone = []
    for t in stage["detections"]:
        pick = -1
        for i, d in enumerate(sweep):
            if taken[i] or abs(d.range - t["range"]) > p["gate_range"] or abs(d.bearing - t["bearing"]) > p["gate_bearing"]:
                continue
            if pick < 0 or d.snr_db > sweep[pick].snr_db:
                pick = i
        if pick >= 0:
            taken[pick] = True
            sweep[pick].range_rate, sweep[pick].range_rate_step, sweep[pick].doppler_path = float(t["range_rate"]), float(p["step"]), 2
        else:
            alone.append(Detection(float(t["range"]), float(t["bearing"]), float(t["snr_db"]), int(t["cells"]), range_rate=float(t["range_rate"]),
                                   range_rate_step=float(p["step"]), range_sigma=float(p["range_sigma"]), doppler_path=1))
    return alone


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


class NormalisedFold:
    """combine cfar (D3): the running maximum over each beam's hypotheses of the map normalised by its own background
    (libs/live doppler.hpp NormalisedFold, fold_normalised, normalised_map; the same operations in the same order)."""

    def __init__(self, n_bearing: int, n_cells: int, cell_samples: int, skip: int):
        self.n_cells, self.k, self.skip = int(n_cells), int(cell_samples), int(min(max(skip, 0), n_cells))
        stride = self.n_cells * self.k
        self.ratio = np.full((n_bearing, stride), -1.0, dtype=np.float32)
        self.cell_ratio = np.full((n_bearing, self.n_cells), -1.0, dtype=np.float32)
        self.reference = np.zeros((n_bearing, self.n_cells))
        self.which = np.full((n_bearing, stride), -1, dtype=np.int32)

    def fold(self, p: dict, h: int, fine: np.ndarray, guard: int, train: int) -> None:
        from . import detect
        n_c, k, skip = self.n_cells, self.k, self.skip
        kept = n_c - skip
        if kept <= 0:
            return
        fine = np.asarray(fine, dtype=np.float32)[:, :n_c * k]
        samples = fine[:, skip * k:].reshape(fine.shape[0], kept, k)
        cells = samples.astype(np.float64).sum(axis=2) if k > 1 else samples[:, :, 0].astype(np.float64)
        cells = (cells / k).astype(np.float32)
        level, _ = detect.background(cells, int(guard), int(train))
        first = np.asarray(p["first"])
        take = (h >= first) & (h < first + int(p["per_beam"]))
        centre = take & (h == first + (int(p["per_beam"]) - 1) // 2)
        floor = np.maximum(level, 1e-300)
        self.reference[centre, skip:] = level[centre]
        c = (cells.astype(np.float64) / floor).astype(np.float32)
        cr = self.cell_ratio[:, skip:]
        better = (c > cr) & take[:, None]
        cr[better] = c[better]
        r = (samples.astype(np.float64) / floor[:, :, None]).astype(np.float32).reshape(fine.shape[0], kept * k)
        rr, ww = self.ratio[:, skip * k:], self.which[:, skip * k:]
        better = (r > rr) & take[:, None]
        rr[better] = r[better]
        ww[better] = h

    def maps(self):
        """(fine, cells, level): ratio x reference, cell_ratio x reference, reference; 0 in the skipped cells."""
        skip, k = self.skip, self.k
        ref = self.reference
        cells = (np.maximum(self.cell_ratio, 0.0).astype(np.float64) * ref).astype(np.float32)
        cells[:, :skip] = 0.0
        fine = (np.maximum(self.ratio, 0.0).astype(np.float64) * np.repeat(ref, k, axis=1)).astype(np.float32)
        fine[:, :skip * k] = 0.0
        return fine, cells, ref.copy()


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


def tone_from_sonar(sonar, index: int = 0) -> dict:
    """The `chain.doppler` tone keys for a sonar whose transmit is a train (sim.waveform SampledWaveform.segments, the
    segment table D2 built): the `index`-th one-frequency `cw` segment's frequency, start, length and taper. A live
    chain takes these from its configuration (the record carries the train as sent, not its table: no schema change).
    Raises ValueError when the train has no such tone (a comb is several tones at once: give its tones one by one)."""
    segs = [s for s in (getattr(getattr(sonar, "sampled_waveform", None), "segments", None) or [])
            if s.kind == "cw" and len(s.frequencies) == 1]
    if index >= len(segs):
        raise ValueError("doppler.tone_from_sonar: the sonar's transmit has no single-frequency tone segment "
                         f"{index} (segments: {[s.kind for s in getattr(sonar.sampled_waveform, 'segments', None) or []]})")
    s = segs[index]
    return {"tone": "bank", "tone_frequency_hz": float(s.frequencies[0]), "tone_start_s": float(s.start),
            "tone_duration_s": float(s.duration), "tone_window_alpha": float(s.window_alpha)}
