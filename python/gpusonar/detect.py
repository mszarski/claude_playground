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
    depth: float = float("nan")         # m, straight-line depth from elevation (stacked rows); NaN: none
    depth_sigma: float = float("nan")
    first_range: float = float("nan")   # m, the nearest cell of the cluster (its leading edge)
    depth_alternatives: list | None = None   # None: `depth` is the depth. A list: the elevation was ambiguous and these are
                                             # the other candidates' depths; a track with a depth takes the one that fits
    extent: float = float("nan")        # m in range the cluster spans (the multiscale detector); NaN: not reported

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
                             float(10 * np.log10(snr[ib, ir].max())), int(len(ib)), first_range=float(ranges[ir].min())))
    return out


# --- the detector at the range resolution (cell_statistic: multiscale) ------------------------------
#
# libs/live detector.hpp says what it is and why; the plan (integration lengths, their thresholds, the sidelobe
# envelopes) comes from there (`core.live_detector_plan`), so both engines test against the same numbers. This is
# the NumPy application of it; tests/python/test_detector.py holds it to the C++ one.

def multiscale(fine: np.ndarray, cell_mean: np.ndarray, fine_ranges, cell_ranges, bearings, guard: int, train: int,
               plan: dict) -> tuple[list[Detection], np.ndarray]:
    """Detections on a map `fine` [n_bearing, n_cells * k] (power, k samples per range cell) with `cell_mean`
    [n_bearing, n_cells] its cell means, and the background of each cell. A cell is flagged when its strongest
    sample, its mean, or the mean of several cells about it exceeds that statistic's threshold over the background
    (the mean of `train` cells either side beyond `guard`); flagged cells a stronger cell's sidelobes explain are
    dropped; touching cells are one detection at the energy-weighted position of their samples, with the strongest
    sample over the background as its snr_db and the range it spans as its extent."""
    from . import _gpusonar as core
    k = int(plan["samples_per_cell"])
    n_b, n_c = cell_mean.shape
    fine_ranges, cell_ranges, bearings = np.asarray(fine_ranges, dtype=float), np.asarray(cell_ranges, dtype=float), np.asarray(bearings, dtype=float)
    cells = np.asarray(fine)[:, :n_c * k].reshape(n_b, n_c, k)
    mean, count = _window_mean(np.asarray(cell_mean, dtype=float), guard, train)
    count = np.broadcast_to(count, (n_c,))
    floor = np.maximum(mean, 1e-300)
    peak = cells.max(axis=2).astype(float)
    scales = plan["scales"]
    looks = float(plan["train_looks_per_cell"])
    correction = {int(c): [core.training_correction(s["pfa"], float(c) * looks) for s in scales] for c in np.unique(count)}
    thr = np.array([[s["factor"] * correction[int(c)][i] for c in count] for i, s in enumerate(scales)])      # [scale, cell]
    flag = peak > thr[0][None, :] * floor
    csum = np.concatenate([np.zeros((n_b, 1)), np.cumsum(np.asarray(cell_mean, dtype=float), axis=1)], axis=1)
    j = np.arange(n_c)
    for i, s in enumerate(scales[1:], start=1):
        m = int(s["cells"])
        a = j - (m - 1) // 2
        ok = (a >= 0) & (a + m <= n_c)
        stat = (csum[:, a[ok] + m] - csum[:, a[ok]]) / m
        flag[:, ok] |= stat > thr[i][None, ok] * floor[:, ok]
    background = mean.astype(np.float32)

    # Cells that a stronger cell's sidelobes explain are dropped, strongest source first.
    fb, fj = np.nonzero(flag)
    if len(fb):
        fp = peak[fb, fj]
        alive = np.ones(len(fb), dtype=bool)
        sin_b = np.sin(bearings)
        env_b, env_r = np.asarray(plan["bearing_envelope"]), np.asarray(plan["range_envelope"])
        is_source = np.flatnonzero(fp >= plan["source_ratio"] * background[fb, fj].astype(float))
        for q in is_source[np.argsort(-fp[is_source], kind="stable")]:
            if not alive[q]:
                continue
            du = np.abs(sin_b[fb] - sin_b[fb[q]])
            e_b = env_b[np.minimum(np.floor(du / plan["u_step"] + 0.5).astype(int), len(env_b) - 1)]
            # Off the source's bearing its energy is spread in range by the array's own length: that many cells are let off.
            smear = np.ceil(0.25 * plan["aperture_m"] * du / plan["cell_m"] - 1e-9).astype(int)
            e_r = env_r[np.minimum(np.maximum(np.abs(fj - fj[q]) - smear, 0), n_c)]
            alive &= ~((fp < fp[q]) & ~((e_r >= 1.0) & (e_b >= 1.0)) & (fp <= plan["margin"] * e_r * e_b * fp[q]))
        flag = np.zeros_like(flag)
        flag[fb[alive], fj[alive]] = True

    labels, n = ndimage.label(flag, structure=np.ones((3, 3), dtype=int))
    out = []
    ratio = peak / np.maximum(background.astype(float), 1e-300)
    for label, where in enumerate(ndimage.find_objects(labels), start=1):
        ib, ir = np.nonzero(labels[where] == label)
        ib, ir = ib + where[0].start, ir + where[1].start
        w = cells[ib, ir, :].astype(float)                                   # [cell, sample]
        r = fine_ranges[(ir[:, None] * k + np.arange(k)[None, :])]
        near, far = float(cell_ranges[ir].min()), float(cell_ranges[ir].max())
        out.append(Detection(float((w * r).sum() / w.sum()), float((w.sum(axis=1) * bearings[ib]).sum() / w.sum()),
                             float(10 * np.log10(ratio[ib, ir].max())), int(len(ib)), first_range=near,
                             extent=far - near + float(plan["cell_m"])))
    return out, background


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
    depth: float = float("nan")                   # m, from the detections that carried one
    depth_var: float = float("inf")


class Tracker:
    """Nearest-neighbour Kalman tracker with a constant-velocity model in the horizontal
    plane. Tracks start on any unassociated detection, are confirmed after `confirm_hits`
    updates and dropped after `max_misses` pings without one."""

    def __init__(self, process_noise: float = 0.05, range_sigma: float = 2.0,
                 bearing_sigma: float = np.radians(1.0), gate: float = 4.0, confirm_hits: int = 3,
                 max_misses: int = 3, initial_speed_sigma: float = 10.0, depth_process_noise: float = 0.5):
        self.q = process_noise
        self.depth_q = depth_process_noise     # m per sqrt(s)
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

    @staticmethod
    def _take_depth(t: Track, d: Detection, gate: float = 3.0) -> None:
        """A scalar Kalman update of the track's depth with a detection's, when it has one. An
        ambiguous detection offers several candidates: a track that already has a depth takes
        the nearest if it is within `gate` sigma, and a track without one waits."""
        if not (np.isfinite(d.depth) and np.isfinite(d.depth_sigma) and d.depth_sigma > 0.0):
            return
        r = d.depth_sigma**2
        z = d.depth
        if d.depth_alternatives is not None:
            if not np.isfinite(t.depth):
                return
            z = min([d.depth] + list(d.depth_alternatives), key=lambda c: abs(c - t.depth))
            if abs(z - t.depth) > gate * np.sqrt(t.depth_var + r):
                return
        if not np.isfinite(t.depth):
            t.depth, t.depth_var = z, r
            return
        k = t.depth_var / (t.depth_var + r)
        t.depth += k * (z - t.depth)
        t.depth_var *= 1.0 - k

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
            if np.isfinite(t.depth_var):
                t.depth_var += self.depth_q**2 * dt
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
            self._take_depth(t, d)
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
            self._take_depth(tr, d)
            self.tracks.append(tr)
            self._next_id += 1
        return [t for t in self.tracks if t.confirmed]
