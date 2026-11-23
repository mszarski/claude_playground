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
    noise): threshold = factor * mean of n_train cells gives the false-alarm rate `pfa`.
    Source: docs/REFERENCES.md Sec. 12.1."""
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
    arrivals: list | None = None        # [(range m, snr dB)], nearest first: the local peaks along range inside the detection
                                        # (the multiscale detector; libs/live detections.hpp), at most the 8 strongest
    time: float | None = None           # s, when the detection shows the target: for a monostatic echo the ping's transmit
                                        # time + range / c (D-156). None (or NaN, any value that is not finite): the
                                        # ping's time given to Tracker.update (as before)
    centroid_range: float = float("nan")  # m, the energy-weighted mean range of its samples (the multiscale detector),
                                          # whichever `range` reports: the elevation search, blanking and the burst
                                          # filter go by it (D-158). NaN: `range` is the centroid (libs/live detections.hpp)

    def xy(self) -> np.ndarray:
        """(across, forward) position in the array's horizontal frame."""
        return self.range * np.array([np.sin(self.bearing), np.cos(self.bearing)])


def group_flagged(rows, cols, bearings, merge_cells: int, merge_u: float) -> np.ndarray:
    """Which flagged cells (bearing index `rows`, range-cell index `cols`) are one detection: the group number of each.
    Two cells belong together when they are no further apart than `merge_cells` + 1 cells in range and `merge_u` in
    sin(bearing), and groups are closed under that (libs/live detector.hpp, DECISIONS D-95: within what the tracker can
    tell apart in range and half the beam's half-power width in bearing; `merge_cells` and `merge_u` come from the
    detector's plan, `live.detector_plan` / `core.live_detector_plan`). `merge_u` is never taken below one step of the
    bearing grid, so touching cells always belong together. The one grouping rule of the multiscale detector and of
    `cluster(..., merge=...)`."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree
    rows, cols = np.asarray(rows), np.asarray(cols)
    sin_b = np.sin(np.asarray(bearings, dtype=float))
    if len(sin_b) > 1:
        merge_u = max(float(merge_u), float(np.abs(np.diff(sin_b)).max()))
    reach = int(merge_cells) + 1
    # Within 1 in the largest-coordinate distance of these: within merge_u and within `reach` cells.
    pts = np.stack([sin_b[rows] / (merge_u + 1e-12), cols / (reach + 1e-9)], axis=1)
    pairs = cKDTree(pts).query_pairs(1.0, p=np.inf, output_type="ndarray")
    graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(rows), len(rows)))
    return connected_components(graph, directed=False)[1]


def cluster(detections: np.ndarray, power: np.ndarray, ranges, bearings, guard: int = 4,
            train: int = 32, merge: tuple | None = None) -> list[Detection]:
    """Group touching detection cells and report one detection per group at its
    power-weighted centroid. `merge` = (merge_cells, merge_u): cells no further apart than that in range and in
    sin(bearing) are one group as well (`group_flagged`: the multiscale detector's rule, so that the pieces of one
    object that a tracker could not keep apart are one detection)."""
    if merge is None:
        labels, n = ndimage.label(detections, structure=np.ones((3, 3), dtype=int))
    else:
        fb, fj = np.nonzero(detections)
        labels, n = np.zeros(np.shape(detections), dtype=int), 0
        if len(fb):
            member = group_flagged(fb, fj, bearings, int(merge[0]), float(merge[1]))
            labels[fb, fj] = member + 1
            n = int(member.max()) + 1
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

def background(cell_mean: np.ndarray, guard: int, train: int) -> tuple[np.ndarray, np.ndarray]:
    """The multiscale detector's background of every cell of `cell_mean` [n_bearing, n_cells], and per range cell the
    number of independent training cells it is worth. The mean of the `train` cells either side beyond `guard`; where
    that window is not balanced (the ends of the map) the mean is moved along the background's slope, taken between the
    medians of the first (last) two training lengths of the same bearing (libs/live detector.hpp says why)."""
    power = np.asarray(cell_mean, dtype=float)
    n = power.shape[-1]
    mean, _ = _window_mean(power, guard, train)
    j = np.arange(n)
    n_lo = np.clip(j - guard, 0, n) - np.clip(j - guard - train, 0, n)
    n_hi = np.clip(j + guard + 1 + train, 0, n) - np.clip(j + guard + 1, 0, n)
    count = np.maximum(n_lo + n_hi, 1).astype(float)
    sloped = n >= 2 * train and train >= 8
    offset = (n_hi * (guard + 0.5 * (n_hi + 1.0)) - n_lo * (guard + 0.5 * (n_lo + 1.0))) / count if sloped else np.zeros(n)
    r = np.abs(offset) / train
    worth = 1.0 / (1.0 / count + (np.pi * r * r + 2.0 * r) / train)
    if sloped:
        med = [np.median(power[..., a:a + train], axis=-1) for a in (0, train, n - 2 * train, n - train)]
        with np.errstate(divide="ignore", invalid="ignore"):
            near = np.where((med[0] > 0) & (med[1] > 0), np.log(med[0] / med[1]) / train, 0.0)
            far = np.where((med[2] > 0) & (med[3] > 0), np.log(med[2] / med[3]) / train, 0.0)
        slope = np.where(offset[None, :] > 0.0, near[..., None], far[..., None])
        mean = mean * np.exp(np.clip(slope * offset[None, :], -2.3, 2.3))
    return mean, worth


def check_plan(plan: dict) -> None:
    """Refuses a plan whose first scale is not the single-sample test (`cells` 0) or whose other scales are not means of
    one cell or more: `multiscale` reads scale 0 as the single sample in its flag test, its arrivals and its leading
    edge. `live_detector_plan` always puts it first, whatever `integrate_cells` holds; a plan built by hand need not
    (libs/live detector.hpp check_plan; sonar-d99)."""
    scales = plan.get("scales") or []
    if not scales or int(scales[0]["cells"]) != 0:
        raise ValueError("detector plan: its first scale must be the single-sample test (cells 0); plan_detector puts it there")
    if any(int(s["cells"]) < 1 for s in scales[1:]):
        raise ValueError("detector plan: scales after the first are means of one cell or more")


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
    mean, count = background(cell_mean, guard, train)
    floor = np.maximum(mean, 1e-300)
    peak = cells.max(axis=2).astype(float)
    scales = plan["scales"]
    check_plan(plan)
    sample_factor = float(scales[0]["factor"])          # the single-sample test's: the arrivals and the leading edge (sonar-d99)
    looks = float(plan["train_looks_per_cell"])
    correction = {float(c): [core.training_correction(s["pfa"], float(c) * looks) for s in scales] for c in np.unique(count)}
    thr = np.array([[s["factor"] * correction[float(c)][i] for c in count] for i, s in enumerate(scales)])      # [scale, cell]
    flag = peak > thr[0][None, :] * floor
    csum = np.concatenate([np.zeros((n_b, 1)), np.cumsum(np.asarray(cell_mean, dtype=float), axis=1)], axis=1)
    j = np.arange(n_c)
    for i, s in enumerate(scales[1:], start=1):
        m = int(s["cells"])
        a = j - (m - 1) // 2
        ok = (a >= 0) & (a + m <= n_c)
        stat = (csum[:, a[ok] + m] - csum[:, a[ok]]) / m
        flag[:, ok] |= stat > thr[i][None, ok] * floor[:, ok]
    background_map = mean.astype(np.float32)

    # Cells that a stronger cell's sidelobes explain are dropped, strongest source first.
    fb, fj = np.nonzero(flag)
    if len(fb):
        fp = peak[fb, fj]
        alive = np.ones(len(fb), dtype=bool)
        # A cell inside the core of a stronger source (its half-power beam, the next range cell) is that source's own:
        # not dropped by another source's sidelobe rule (it is still a source itself: a second arrival has its own sidelobes).
        owned = np.zeros(len(fb), dtype=bool)
        sin_b = np.sin(bearings)
        env_b, env_r = np.asarray(plan["bearing_envelope"]), np.asarray(plan["range_envelope"])
        is_source = np.flatnonzero(fp >= plan["source_ratio"] * background_map[fb, fj].astype(float))
        for q in is_source[np.argsort(-fp[is_source], kind="stable")]:
            if not alive[q]:
                continue
            du = np.abs(sin_b[fb] - sin_b[fb[q]])
            e_b = env_b[np.minimum(np.floor(du / plan["u_step"] + 0.5).astype(int), len(env_b) - 1)]
            # Off the source's bearing its energy is spread in range by the array's own length: that many cells are let off.
            smear = np.floor(0.25 * plan["aperture_m"] * du / plan["cell_m"] + 0.5).astype(int)      # whole cells, to the nearest
            e_r = env_r[np.minimum(np.maximum(np.abs(fj - fj[q]) - smear, 0), n_c)]
            weaker = alive & (fp < fp[q])
            core = (e_r >= 1.0) & (e_b >= 1.0)
            owned |= weaker & core
            alive &= ~(weaker & ~core & ~owned & (fp <= plan["margin"] * e_r * e_b * fp[q]))
        flag = np.zeros_like(flag)
        flag[fb[alive], fj[alive]] = True
        fb, fj = fb[alive], fj[alive]

    # One detection per group of flagged cells that belong together: no further apart than merge_cells + 1 in range and
    # merge_u in sin(bearing) (touching cells always are). Groups in the order of their first cell, by bearing then range.
    out = []
    if len(fb):
        member = group_flagged(fb, fj, bearings, int(plan["merge_cells"]), float(plan["merge_u"]))
        ratio = peak / np.maximum(background_map.astype(float), 1e-300)
        first = {}
        for i, g in enumerate(member):
            first.setdefault(int(g), i)
        for g in sorted(first, key=first.get):
            ib, ir = fb[member == g], fj[member == g]
            w = cells[ib, ir, :].astype(float)                                   # [cell, sample]
            r = fine_ranges[(ir[:, None] * k + np.arange(k)[None, :])]
            near, far = float(cell_ranges[ir].min()), float(cell_ranges[ir].max())
            # Range: the energy-weighted mean of the group's samples. Bearing: the mean over its bearings weighted by what
            # each holds, over all its range cells, above half the strongest one's energy (the group ends at the beam's
            # half-power points on the bearing grid; a plain energy-weighted mean over a beam cut there is pulled towards
            # the grid, and weights that fall to nothing at the cut are not).
            rows_ = np.unique(ib)
            energy = np.array([w[ib == b_].sum() for b_ in rows_])
            # The level: half the strongest, or what the outermost bearings hold when that is more (a beam wider than the
            # modelled one is cut above its half-power points). Nothing above it (one or two bearings): the plain mean.
            soft = np.maximum(energy - max(0.5 * energy.max(), energy[0], energy[-1]), 0.0)
            if soft.sum() <= 0.0:
                soft = energy
            # Its arrivals: along range, the strongest sample over the group's bearings; the local peaks of that above the
            # single-sample threshold (a run of samples the group does not hold ends a peak), the 8 strongest, nearest first.
            sample = (ir[:, None] * k + np.arange(k)[None, :]).ravel()
            over = (w / np.maximum(background_map[ib, ir].astype(float), 1e-300)[:, None]).ravel()
            idx = np.unique(sample)
            line = np.full(len(idx), 0.0)
            np.maximum.at(line, np.searchsorted(idx, sample), over)
            left = np.concatenate([[False], idx[1:] == idx[:-1] + 1])
            right = np.concatenate([idx[:-1] == idx[1:] - 1, [False]])
            prev, nxt = np.concatenate([[0.0], line[:-1]]), np.concatenate([line[1:], [0.0]])
            is_peak = (line > sample_factor) & ~(left & (prev >= line)) & ~(right & (nxt > line))
            top = np.flatnonzero(is_peak)
            top = np.sort(top[np.argsort(-line[top], kind="stable")][:8])
            arrivals = [(float(fine_ranges[idx[q]]), float(10 * np.log10(line[q]))) for q in top]
            rng = centroid = float((w * r).sum() / w.sum())
            if plan.get("range_estimate", "centroid") == "first":
                # The leading edge (libs/live detector.hpp RangeEstimate::First, D-157): the nearest sample above the
                # single-sample threshold, else the nearest cell's centre.
                over_ = np.flatnonzero(line > sample_factor)
                rng = float(fine_ranges[idx[over_[0]]]) if len(over_) else near
            out.append(Detection(rng, float((soft * bearings[rows_]).sum() / soft.sum()),
                                 float(10 * np.log10(ratio[ib, ir].max())), int(len(ib)), first_range=near,
                                 extent=far - near + float(plan["cell_m"]), arrivals=arrivals, centroid_range=centroid))
    return out, background_map


# --- tracking -------------------------------------------------------------------------------------

@dataclass
class Track:
    id: int
    state: np.ndarray        # [x, y, vx, vy]
    cov: np.ndarray
    hits: int = 1
    misses: int = 0
    confirmed: bool = False
    history: list = field(default_factory=list)   # (time, x, y): the filtered position (the state) at each ping
    measured: list = field(default_factory=list)  # (x, y) per history row: the position of the detection the track took
                                                  # there, in the same frame; (NaN, NaN) on a row of a miss. Empty: none
                                                  # recorded (a track built from positions alone; classify then fits
                                                  # `history`, as before)
    pings: list = field(default_factory=list)     # s, per history row: the time of the ping it was added on (the time given
                                                  # to `Tracker.update`, the transmit time), where `history` holds the
                                                  # detection's own, transmit + R/c (D-156). classify uses it to refuse a
                                                  # velocity from echo delays alone (sonar-054). Empty: none recorded
    snr: list = field(default_factory=list)       # SNR (dB) of each detection it took
    cells: list = field(default_factory=list)     # flagged map cells of each detection it took (Detection.cells)
    depth: float = float("nan")                   # m, from the detections that carried one
    depth_var: float = float("inf")
    time: float = float("nan")                    # s, when `state` holds: its last detection's time, or the ping's after a miss


ASSOCIATIONS = ("likelihood", "in_order")


class Tracker:
    """Nearest-neighbour Kalman tracker with a constant-velocity model in the horizontal
    plane. Tracks start on any unassociated detection, are confirmed after `confirm_hits`
    updates and dropped after `max_misses` pings without one.

    `association` says how detections are shared out among tracks on each ping (D-141).
    Confirmed tracks always choose before tentative ones. "likelihood" (the default) then
    takes the (track, detection) pairs inside the gate most probable first, by the
    Gaussian negative log-likelihood d^2 + ln|S|, each track and each detection used once;
    ties go to the earlier track, then the earlier detection. "in_order" is the rule before
    D-141: each track in turn takes its nearest gated detection by d^2, so a track earlier
    in the list can take a detection that fits a later track better. That let new tracks
    climbing a boat's wake take the hull's detection from the hull's own track.
    Sources: docs/REFERENCES.md Sec. 12.3 (the association rule is ours)."""

    def __init__(self, process_noise: float = 0.05, range_sigma: float = 2.0,
                 bearing_sigma: float = np.radians(1.0), gate: float = 4.0, confirm_hits: int = 3,
                 max_misses: int = 3, initial_speed_sigma: float = 10.0, depth_process_noise: float = 0.5,
                 association: str = "likelihood"):
        if association not in ASSOCIATIONS:
            raise ValueError(f"tracker association {association!r}: one of {ASSOCIATIONS}")
        self.association = association
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
        self._took: dict = {}

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

    def _predict(self, t: Track, to: float) -> tuple[np.ndarray, np.ndarray, float]:
        """The track's state, covariance and depth variance moved by the constant-velocity model from `t.time` to `to`.
        `to` may be earlier than `t.time` when a detection within a ping is (D-156: each detection at transmit + R/c, so a
        nearer one is earlier). The process noise then still grows the uncertainty: Q = G G^T q^2 is an outer product,
        positive semi-definite for either sign of dt, and the depth's random walk grows by q_d^2 |dt| (ours: a variance
        never shrinks by predicting; the same for dt >= 0 as before)."""
        dt = to - t.time
        f = np.eye(4)
        f[0, 2] = f[1, 3] = dt
        g = np.array([[0.5 * dt**2, 0], [0, 0.5 * dt**2], [dt, 0], [0, dt]])
        q = g @ g.T * self.q**2
        depth_var = t.depth_var + self.depth_q**2 * abs(dt) if np.isfinite(t.depth_var) else t.depth_var
        return f @ t.state, f @ t.cov @ f.T + q, depth_var

    def update(self, time: float, detections: list[Detection]) -> list[Track]:
        """Advance to the ping at `time` (s) with its detections. Each detection is compared with every track predicted to
        the detection's own time (`Detection.time`, the ping's when None: a monostatic echo shows the target where it was
        when the sound reached it, transmit + range / c; D-156), a track takes its detection at that time and a new track
        is born there; a time that is not finite (NaN) counts as none, as in the C++ tracker. Tracks that take none count a
        miss and are predicted to `time`, or stay where they are if they already hold at a later time (the last detection
        they took was at transmit + R/c after this ping's transmit: a ping interval shorter than R/c), so a track never
        moves back in time. Returns the confirmed tracks."""
        h = np.eye(2, 4)
        whens = [time if d.time is None or not np.isfinite(d.time) else float(d.time) for d in detections]
        cache: dict = {}

        def predicted(ti: int, when: float):
            key = (ti, when)
            if key not in cache:
                cache[key] = self._predict(self.tracks[ti], when)
            return cache[key]

        self._took = {}                              # track id -> the position of the detection it took this ping
        if self.association == "likelihood":
            unused = self._associate_by_likelihood(detections, whens, h, predicted)
        else:
            unused = self._associate_in_order(detections, whens, h, predicted)
        for ti, t in enumerate(self.tracks):
            if t.misses and t.time < time:           # took nothing this ping: where it is at the ping, never earlier
                t.state, t.cov, t.depth_var = predicted(ti, time)
                t.time = time
            t.history.append((t.time, float(t.state[0]), float(t.state[1])))
            t.pings.append(float(time))
            z = self._took.get(t.id)
            t.measured.append((float(z[0]), float(z[1])) if z is not None else (float("nan"), float("nan")))
        self.tracks = [t for t in self.tracks if t.misses <= self.max_misses]
        for j in unused:
            d = detections[j]
            cov = np.zeros((4, 4))
            cov[:2, :2] = self._measurement_cov(d)
            cov[2, 2] = cov[3, 3] = self.initial_speed_sigma**2
            xy = d.xy()
            tr = Track(self._next_id, np.array([xy[0], xy[1], 0.0, 0.0]), cov, time=whens[j])
            tr.history.append((whens[j], float(xy[0]), float(xy[1])))
            tr.pings.append(float(time))
            tr.measured.append((float(xy[0]), float(xy[1])))
            tr.snr.append(d.snr_db)
            tr.cells.append(int(d.cells))
            self._take_depth(tr, d)
            self.tracks.append(tr)
            self._next_id += 1
        return [t for t in self.tracks if t.confirmed]

    def _take(self, t: Track, d: Detection, h: np.ndarray, prior, when: float) -> None:
        """The Kalman update of one track, predicted to the detection's time (`prior`: state, covariance, depth
        variance there), with that detection."""
        t.state, t.cov, t.depth_var = prior
        t.time = when
        s = h @ t.cov @ h.T + self._measurement_cov(d)
        k = t.cov @ h.T @ np.linalg.inv(s)
        z = d.xy()
        t.state = t.state + k @ (z - h @ t.state)
        t.cov = (np.eye(4) - k @ h) @ t.cov
        self._took[t.id] = z
        t.hits += 1
        t.snr.append(d.snr_db)
        t.cells.append(int(d.cells))
        self._take_depth(t, d)
        t.misses = 0
        t.confirmed = t.confirmed or t.hits >= self.confirm_hits

    def _associate_by_likelihood(self, detections: list[Detection], whens: list[float], h: np.ndarray, predicted) -> list[int]:
        """Most probable gated pairs first, confirmed tracks before tentative ones, each track predicted to each
        detection's time. Returns the indices of the detections no track took, in their original order."""
        pairs = []
        for ti, t in enumerate(self.tracks):
            for j, d in enumerate(detections):
                x, p, _ = predicted(ti, whens[j])
                s = h @ p @ h.T + self._measurement_cov(d)
                y = d.xy() - h @ x
                d2 = float(y @ np.linalg.solve(s, y))
                if d2 < self.gate**2:
                    pairs.append((not t.confirmed, d2 + float(np.log(np.linalg.det(s))), ti, j))
        pairs.sort()
        taken: dict[int, int] = {}
        used: set[int] = set()
        for _, _, ti, j in pairs:
            if ti in taken or j in used:
                continue
            taken[ti] = j
            used.add(j)
        for ti, t in enumerate(self.tracks):
            if ti in taken:
                j = taken[ti]
                self._take(t, detections[j], h, predicted(ti, whens[j]), whens[j])
            else:
                t.misses += 1
        return [j for j in range(len(detections)) if j not in used]

    def _associate_in_order(self, detections: list[Detection], whens: list[float], h: np.ndarray, predicted) -> list[int]:
        """The rule before D-141. Returns the indices of the detections no track took."""
        unused = list(range(len(detections)))
        # Confirmed tracks choose first, each taking its nearest gated detection.
        order = sorted(range(len(self.tracks)), key=lambda ti: not self.tracks[ti].confirmed)
        for ti in order:
            t = self.tracks[ti]
            best, best_d2 = None, self.gate**2
            for j in unused:
                d = detections[j]
                x, p, _ = predicted(ti, whens[j])
                s = h @ p @ h.T + self._measurement_cov(d)
                y = d.xy() - h @ x
                d2 = float(y @ np.linalg.solve(s, y))
                if d2 < best_d2:
                    best, best_d2 = j, d2
            if best is None:
                t.misses += 1
                continue
            unused.remove(best)
            self._take(t, detections[best], h, predicted(ti, whens[best]), whens[best])
        return unused
