"""The `sync` stage: align capture boards that each run on their own sample clock.

Several 8-input boards are batched into one array. Every board timestamps its samples
with its own oscillator, so boards start at slightly different instants (offset) and
tick at slightly different rates (drift). Beamforming across boards needs them on one
timeline to a small fraction of a period.

Clock model for a board: sample n was taken at true time
    t = start_time + offset + n / (sample_rate * (1 + drift)).

Inputs, in order of preference:

1. `reference`: a shared electrical signal on one input of every board (two chirp bursts
   per ping at known instants). Gives offset and drift.
2. `direct_path`: the transmit pulse as it reaches each board's hydrophones through the
   water, against the delay the geometry predicts. Gives offset per ping.
3. `pps`: each board's recorded 1PPS edge sample index. Gives offset per ping.

For 2 and 3 the drift comes from how the offset changes between pings.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import hilbert

from . import _gpusonar as core
from .sim.receiver import reference_times, sync_chirp


class SyncError(RuntimeError):
    """The boards could not be aligned well enough to beamform across them."""


@dataclass
class ClockEstimate:
    offset: np.ndarray       # [boards] s, relative to the first board
    drift_ppm: np.ndarray    # [boards], relative to the first board
    method: str
    quality: np.ndarray      # [boards] correlation peak quality, 0..1 (1 = clean)


def alignment_tolerance(record) -> float:
    """A twentieth of a period at the top of the band: 1.4 us at 35 kHz, 5 us at 10 kHz."""
    w = record.transmit.waveform
    return 1.0 / (20.0 * max(w.f_start, w.f_stop))


def _board_rows(record) -> list[list[int]]:
    rows = [[] for _ in record.metadata.boards]
    for row, b in enumerate(record.metadata.channel_board):
        rows[b].append(row)
    return rows


def _burst_position(x: np.ndarray, replica: np.ndarray, expected: int, search: int, fs: float,
                    f_centre: float):
    """Sample position (fractional) where `replica` best matches real series `x`, near
    `expected`. Coarse position from the correlation envelope, refined with its phase."""
    lo = max(expected - search, 0)
    hi = min(expected + search + len(replica), len(x))
    seg = hilbert(x[lo:hi])
    rep = hilbert(replica)
    corr = np.correlate(seg, rep, mode="valid")
    mag = np.abs(corr)
    i = int(np.argmax(mag))
    if i == 0 or i == len(mag) - 1:
        return None, 0.0
    frac = 0.5 * (mag[i - 1] - mag[i + 1]) / (mag[i - 1] - 2 * mag[i] + mag[i + 1])
    coarse = lo + i + frac
    # The correlation at lag i has phase 2 pi f_centre * (i - true lag) / fs, to a period.
    fine = i - np.angle(corr[i]) * fs / (2.0 * np.pi * f_centre)
    period = fs / f_centre
    fine += np.round((i + frac - fine) / period) * period
    quality = mag[i] / (np.linalg.norm(seg[i:i + len(rep)]) * np.linalg.norm(rep) + 1e-300)
    return lo + fine, float(quality)


def estimate_clocks_reference(record, reference: dict, search_time: float = 2e-3) -> ClockEstimate:
    """Offset and drift of every board from the shared reference bursts."""
    s = record.samples
    if s.is_complex():
        raise SyncError("the sync stage works on raw real samples, before baseband conversion")
    fs = s.sample_rate
    x = s.array
    t_a, t_b = reference_times(record, reference)
    n_rep = int(np.ceil(reference["duration"] * fs)) + 1
    replica = sync_chirp(np.arange(n_rep) / fs, reference["f_start"], reference["f_stop"],
                         reference["duration"])
    f_centre = 0.5 * (reference["f_start"] + reference["f_stop"])
    search = int(search_time * fs)
    offsets, drifts, quality = [], [], []
    for board in record.metadata.boards:
        row = board.reference_channel
        if row < 0:
            raise SyncError("a board has no reference channel; use another sync method")
        n_a, q_a = _burst_position(x[row], replica, int((t_a - s.start_time) * fs), search, fs, f_centre)
        n_b, q_b = _burst_position(x[row], replica, int((t_b - s.start_time) * fs), search, fs, f_centre)
        if n_a is None or n_b is None:
            raise SyncError("reference burst not found on a board; widen search_time or check wiring")
        # n = (t - start - offset) * fs * (1 + drift)
        drift = (n_b - n_a) / ((t_b - t_a) * fs) - 1.0
        offset = (t_a - s.start_time) - n_a / (fs * (1.0 + drift))
        offsets.append(offset)
        drifts.append(drift * 1e6)
        quality.append(min(q_a, q_b))
    offsets, drifts = np.array(offsets), np.array(drifts)
    return ClockEstimate(offsets - offsets[0], drifts - drifts[0], "reference", np.array(quality))


def estimate_clocks_pps(record, nominal_pps_sample: float) -> ClockEstimate:
    """Offsets from each board's 1PPS edge sample index (drift is not observable in one ping).

    A board whose clock started late sees the edge at a smaller sample index.
    """
    fs = record.samples.sample_rate
    pps = np.array([b.pps_sample for b in record.metadata.boards], dtype=float)
    if np.any(np.isnan(pps)):
        raise SyncError("a board has no 1PPS time stamp")
    offsets = (nominal_pps_sample - pps) / fs
    return ClockEstimate(offsets - offsets[0], np.zeros(len(pps)), "pps", np.ones(len(pps)))


def estimate_clocks_direct_path(record, search_time: float = 2e-3) -> ClockEstimate:
    """Offsets from the transmit pulse's direct arrival at each board's hydrophones.

    Needs a receive window that contains the direct arrival, and the projector and
    hydrophone positions. Each channel's arrival is compared with the geometric delay;
    a board's offset is the median over its channels.
    """
    s = record.samples
    if s.is_complex():
        raise SyncError("the sync stage works on raw real samples, before baseband conversion")
    fs, c = s.sample_rate, record.environment.sound_speed
    w = record.transmit.waveform
    n_rep = int(np.ceil(w.duration * fs)) + 1
    u = np.arange(n_rep) / fs
    k = (w.f_stop - w.f_start) / w.duration
    replica = np.cos(2 * np.pi * (w.f_start * u + 0.5 * k * u**2))
    tx = np.array(record.tx_position())
    delay = np.linalg.norm(record.rx_positions() - tx, axis=1) / c
    x = s.array
    f_centre = w.center()
    offsets, quality = [], []
    for rows in _board_rows(record):
        est, q = [], []
        for row in rows:
            expected = int((delay[row] - s.start_time) * fs)
            n_hat, qq = _burst_position(x[row], replica, expected, int(search_time * fs), fs, f_centre)
            if n_hat is None:
                continue
            est.append((delay[row] - s.start_time) - n_hat / fs)
            q.append(qq)
        if not est:
            raise SyncError("direct arrival not found on a board; does the window start at transmit?")
        offsets.append(np.median(est))
        quality.append(np.median(q))
    offsets = np.array(offsets)
    return ClockEstimate(offsets - offsets[0], np.zeros(len(offsets)), "direct_path", np.array(quality))


def _sinc_resample(x: np.ndarray, positions: np.ndarray, taps: int = 16) -> np.ndarray:
    """Values of real series x at fractional sample positions (Hann-windowed sinc)."""
    base = np.floor(positions).astype(np.int64)
    frac = positions - base
    half = taps // 2
    out = np.zeros(len(positions), dtype=np.float64)
    for k in range(-half + 1, half + 1):
        idx = base + k
        d = k - frac
        wgt = np.sinc(d) * (0.5 + 0.5 * np.cos(np.pi * d / half))
        ok = (idx >= 0) & (idx < len(x))
        out[ok] += wgt[ok] * x[idx[ok]]
    return out


def resample_boards(record, estimate: ClockEstimate) -> None:
    """Resample every board onto the first board's timeline, in place, and record the
    applied correction in the metadata."""
    s = record.samples
    x = s.array.astype(np.float64)
    fs = s.sample_rate
    m = np.arange(s.n_samples)
    rows = _board_rows(record)
    for b, board_rows in enumerate(rows):
        offset, drift = estimate.offset[b], estimate.drift_ppm[b] * 1e-6
        if offset == 0.0 and drift == 0.0:
            continue
        # Nominal sample m is wanted at time m / fs; this board holds that instant at index n.
        positions = (m / fs - offset) * fs * (1.0 + drift)
        for row in board_rows:
            x[row] = _sinc_resample(x[row], positions)
    s.array = x.astype(np.float32)
    record.samples = s
    meta = record.metadata
    boards = meta.boards
    for b, board in enumerate(boards):
        board.clock_offset = float(estimate.offset[b])
        board.clock_drift_ppm = float(estimate.drift_ppm[b])
        board.synchronised = 1
    meta.boards = boards
    record.metadata = meta


@dataclass
class SyncReport:
    estimate: ClockEstimate
    residual: np.ndarray       # [boards] s: misalignment left after resampling
    tolerance: float           # s
    ok: bool


def synchronise(record, reference: dict | None = None, method: str = "reference",
                nominal_pps_sample: float | None = None, tolerance: float | None = None,
                enforce: bool = True) -> SyncReport:
    """Estimate board clocks, resample onto one timeline, and check what is left.

    With the `reference` method the check is direct: the bursts are measured again after
    resampling and must line up within `tolerance` (default: a twentieth of a period at
    the top of the band). If they do not, SyncError is raised and the record must not be
    beamformed across boards (`enforce=False` returns the report instead).
    """
    tolerance = alignment_tolerance(record) if tolerance is None else tolerance
    if method == "reference":
        if reference is None:
            raise SyncError("the reference method needs the reference signal description")
        est = estimate_clocks_reference(record, reference)
    elif method == "direct_path":
        est = estimate_clocks_direct_path(record)
    elif method == "pps":
        if nominal_pps_sample is None:
            raise SyncError("the pps method needs the nominal 1PPS sample index")
        est = estimate_clocks_pps(record, nominal_pps_sample)
    else:
        raise ValueError(f"unknown sync method {method!r}")
    resample_boards(record, est)
    if method == "reference":
        after = estimate_clocks_reference(record, reference)
        span = record.samples.n_samples / record.samples.sample_rate
        residual = np.abs(after.offset) + np.abs(after.drift_ppm) * 1e-6 * span
    elif method == "direct_path":
        residual = np.abs(estimate_clocks_direct_path(record).offset)
    else:
        residual = np.zeros(len(est.offset))      # nothing independent to check against
    ok = bool(np.all(residual <= tolerance))
    report = SyncReport(est, residual, tolerance, ok)
    if enforce and not ok:
        worst = residual.max()
        raise SyncError(f"boards are still {worst * 1e6:.2f} us apart after synchronisation "
                        f"(tolerance {tolerance * 1e6:.2f} us); not beamforming across boards")
    return report
