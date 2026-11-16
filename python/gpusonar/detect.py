"""Detection and tracking on range x bearing maps: normalisation, CFAR, clustering, tracks.

Inputs are power maps [n_bearing, n_range]. CFAR also exists as a C++ kernel for the host
and CUDA backends (libs/detect; `cfar(..., backend=...)`); the rest is NumPy.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage


def _window_mean(power: np.ndarray, guard: int, train: int) -> np.ndarray:
    """Mean of the `train` cells on each side of every cell along range, skipping `guard`
    cells next to it. Edges use whatever cells exist."""
    n = power.shape[-1]
    csum = np.concatenate([np.zeros(power.shape[:-1] + (1,)), np.cumsum(power, axis=-1)], axis=-1)
    idx = np.arange(n)

    def span(lo, hi):
        lo = np.clip(lo, 0, n)
        hi = np.clip(hi, 0, n)
        return csum[..., hi] - csum[..., lo], (hi - lo)

    left, n_left = span(idx - guard - train, idx - guard)
    right, n_right = span(idx + guard + 1, idx + guard + 1 + train)
    count = np.maximum(n_left + n_right, 1)
    return (left + right) / count, count


def normalise(power: np.ndarray, guard: int = 4, train: int = 32) -> np.ndarray:
    """Split-window normaliser: each cell divided by the mean of its range neighbours.
    The output is a signal-to-background ratio that is flat over range."""
    mean, _ = _window_mean(np.asarray(power, dtype=float), guard, train)
    return power / np.maximum(mean, 1e-300)


def cfar_threshold_factor(pfa: float, n_train: int) -> float:
    """Cell-averaging CFAR multiplier for exponentially distributed power (complex Gaussian
    noise): threshold = factor * mean of n_train cells gives the false-alarm rate `pfa`."""
    return n_train * (pfa ** (-1.0 / n_train) - 1.0)


def cfar(power: np.ndarray, pfa: float = 1e-4, guard: int = 4, train: int = 32, backend=None,
         device: int = 0) -> np.ndarray:
    """Cell-averaging constant-false-alarm-rate detector along range. Returns a boolean map.

    With `backend` (core.Backend.Host or .Cuda) the C++ kernel runs instead of NumPy; it
    takes the map as float32 and gives the same detections except where a cell sits
    within rounding of its threshold.
    """
    if backend is not None:
        from . import _gpusonar as core
        return core.cfar(np.ascontiguousarray(power, dtype=np.float32), pfa=pfa, guard=guard, train=train,
                         backend=backend, device=device)
    power = np.asarray(power, dtype=float)
    mean, count = _window_mean(power, guard, train)
    factor = count * (pfa ** (-1.0 / count) - 1.0)
    return power > factor * mean


@dataclass
class Detection:
    range: float
    bearing: float          # rad
    snr_db: float
    cells: int

    def xy(self) -> np.ndarray:
        """(across, forward) position in the array's horizontal frame."""
        return self.range * np.array([np.sin(self.bearing), np.cos(self.bearing)])


def cluster(detections: np.ndarray, power: np.ndarray, ranges, bearings, guard: int = 4,
            train: int = 32) -> list[Detection]:
    """Group touching detection cells and report one detection per group at its
    power-weighted centroid."""
    labels, n = ndimage.label(detections, structure=np.ones((3, 3), dtype=int))
    if n == 0:
        return []
    snr = normalise(power, guard, train)
    ranges, bearings = np.asarray(ranges), np.asarray(bearings)
    out = []
    for k in range(1, n + 1):
        ib, ir = np.nonzero(labels == k)
        w = power[ib, ir]
        out.append(Detection(float((w * ranges[ir]).sum() / w.sum()),
                             float((w * bearings[ib]).sum() / w.sum()),
                             float(10 * np.log10(snr[ib, ir].max())), int(len(ib))))
    return out


# --- tracking -------------------------------------------------------------------------------------

@dataclass
class Track:
    id: int
    state: np.ndarray        # [x, y, vx, vy]
    cov: np.ndarray
    hits: int = 1
    misses: int = 0
    confirmed: bool = False
    history: list = field(default_factory=list)   # (time, x, y)
    snr: list = field(default_factory=list)       # SNR (dB) of each detection it took


class Tracker:
    """Nearest-neighbour Kalman tracker with a constant-velocity model in the horizontal
    plane. Tracks start on any unassociated detection, are confirmed after `confirm_hits`
    updates and dropped after `max_misses` pings without one."""

    def __init__(self, process_noise: float = 0.05, range_sigma: float = 2.0,
                 bearing_sigma: float = np.radians(1.0), gate: float = 4.0, confirm_hits: int = 3,
                 max_misses: int = 3, initial_speed_sigma: float = 10.0):
        self.q = process_noise
        self.range_sigma = range_sigma
        self.bearing_sigma = bearing_sigma
        self.gate = gate
        self.confirm_hits = confirm_hits
        self.max_misses = max_misses
        self.initial_speed_sigma = initial_speed_sigma
        self.tracks: list[Track] = []
        self._next_id = 1
        self._time = None

    def _measurement_cov(self, d: Detection) -> np.ndarray:
        # Polar measurement noise rotated into Cartesian.
        s, c = np.sin(d.bearing), np.cos(d.bearing)
        j = np.array([[s, d.range * c], [c, -d.range * s]])
        return j @ np.diag([self.range_sigma**2, self.bearing_sigma**2]) @ j.T

    def update(self, time: float, detections: list[Detection]) -> list[Track]:
        dt = 0.0 if self._time is None else time - self._time
        self._time = time
        f = np.eye(4)
        f[0, 2] = f[1, 3] = dt
        g = np.array([[0.5 * dt**2, 0], [0, 0.5 * dt**2], [dt, 0], [0, dt]])
        q = g @ g.T * self.q**2
        h = np.eye(2, 4)
        for t in self.tracks:
            t.state = f @ t.state
            t.cov = f @ t.cov @ f.T + q
        unused = list(range(len(detections)))
        # Confirmed tracks choose first, each taking its nearest gated detection.
        for t in sorted(self.tracks, key=lambda tr: not tr.confirmed):
            best, best_d2 = None, self.gate**2
            for j in unused:
                d = detections[j]
                s = h @ t.cov @ h.T + self._measurement_cov(d)
                y = d.xy() - h @ t.state
                d2 = float(y @ np.linalg.solve(s, y))
                if d2 < best_d2:
                    best, best_d2 = j, d2
            if best is None:
                t.misses += 1
                continue
            d = detections[best]
            unused.remove(best)
            s = h @ t.cov @ h.T + self._measurement_cov(d)
            k = t.cov @ h.T @ np.linalg.inv(s)
            t.state = t.state + k @ (d.xy() - h @ t.state)
            t.cov = (np.eye(4) - k @ h) @ t.cov
            t.hits += 1
            t.snr.append(d.snr_db)
            t.misses = 0
            t.confirmed = t.confirmed or t.hits >= self.confirm_hits
        for t in self.tracks:
            t.history.append((time, float(t.state[0]), float(t.state[1])))
        self.tracks = [t for t in self.tracks if t.misses <= self.max_misses]
        for j in unused:
            d = detections[j]
            cov = np.zeros((4, 4))
            cov[:2, :2] = self._measurement_cov(d)
            cov[2, 2] = cov[3, 3] = self.initial_speed_sigma**2
            xy = d.xy()
            tr = Track(self._next_id, np.array([xy[0], xy[1], 0.0, 0.0]), cov)
            tr.history.append((time, float(xy[0]), float(xy[1])))
            tr.snr.append(d.snr_db)
            self.tracks.append(tr)
            self._next_id += 1
        return [t for t in self.tracks if t.confirmed]
