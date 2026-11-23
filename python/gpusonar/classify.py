"""Sorting confirmed tracks: which move, which stand still, which are echoes of another.

Rule-based, from the track alone (no image features, no learning):

  mover       it was seen on most pings and its positions fit a straight line through time
              whose speed is well above the scatter of the fit;
  static      it was seen on most pings and its fitted speed is small;
  multipath   it is within the range, bearing and velocity gates of a group grown from an established
              track (mover, static, line): within the gates of any member, transitively, nearer or
              farther in range. Another arrival of the same object by another path (long-range sonar
              in a channel). Of such a group the object is the established track nearest in range,
              the others its multipath;
  line        it appears to move, but along a row of other tracks: a detection sliding
              along a pipeline, a cable or a wake as the aspect changes. Motion along a
              linear feature cannot be told from the feature standing still;
  uncertain   anything else: too few points, or positions that jump about.

What this cannot do: tell a moored buoy from a rock or from a wave face that holds still,
or a boat at anchor from a boulder. Those need more than a track (extent, depth, aspect).

Opt-in, off by default (`detections`): two rules that use what the detections themselves carry on every ping,
not only where the track went. Both rules and their thresholds are ours, with no published source; the thresholds
have no default, since they depend on the map's cells and on the site (docs/CONFIGURATION.md):

  extent      a mover (after the row test) whose detections are small, the median number of flagged map cells
              over the detections it took below `min_mover_cells`, is uncertain: a wave glint or a clutter peak
              that the tracker joined into a straight line is a few cells; a hull at these ranges spans many.
              A track without recorded cells is left as it is;
  row         in the corridor test (`linear`), another track counts as part of the row only if its mean SNR is no
              more than `row_snr_db` below this track's: the echoes of one pipeline or cable are of one strength,
              and a boat running along a row of seabed clutter is much stronger than the clutter.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .detect import Track


@dataclass
class TrackClass:
    id: int
    kind: str                 # "mover", "static", "line", "multipath", "uncertain"
    position: tuple           # (x, y) in the tracker's frame: at `time` given to classify_tracks, else at the track's
                              # last history point (the last ping, or its last detection's time: D-156)
    velocity: tuple           # fitted (vx, vy), m/s
    speed: float              # m/s
    speed_sigma: float        # one-sigma uncertainty of the fitted speed, m/s
    fit_rms: float            # RMS distance of the positions from the fitted line, m
    seen: float               # fraction of pings since birth with a detection
    mean_snr_db: float
    parent: int | None = None  # for multipath: the id of the track it echoes
    extent_cells: float = float("nan")   # median flagged map cells of the detections it took (NaN: none recorded)


def _given(at) -> bool:
    """A time was given: not None and finite (NaN, as the C++ ClassifyOptions::time, means none)."""
    return at is not None and bool(np.isfinite(at))


OUTLIER_ALPHA = 0.0    # the outlier test off by default (owner, review of sonar-ljh); 0.01 is the probability measured on (D-159)


def t_two_sided(t: float, nu: int) -> float:
    """P(|T| > t) for Student's t with an integer nu >= 1 degrees of freedom, by the finite sums of Abramowitz and
    Stegun 26.7.3 (nu odd) and 26.7.4 (nu even), theta = atan(t / sqrt(nu)): A(t|nu) = P(|T| <= t) =
    (2/pi)(theta + sin theta (cos theta + 2/3 cos^3 theta + ... + (2.4...(nu-3))/(1.3...(nu-2)) cos^(nu-2) theta)) for odd
    nu (the sum absent for nu = 1), sin theta (1 + 1/2 cos^2 theta + ... + (1.3...(nu-3))/(2.4...(nu-2)) cos^(nu-2) theta)
    for even nu. Written out so that libs/live classify.cpp computes the same sums (held to scipy in test_classify)."""
    if not np.isfinite(t):
        return 0.0
    th = float(np.arctan(abs(t) / np.sqrt(nu)))
    c, sn = float(np.cos(th)), float(np.sin(th))
    c2 = c * c
    if nu % 2:
        total, term = 0.0, c
        for j in range(1, (nu - 1) // 2 + 1):        # powers 1, 3, ..., nu - 2
            total += term
            term *= c2 * (2.0 * j) / (2.0 * j + 1.0)
        a = (2.0 / np.pi) * (th + sn * total)
    else:
        total, term = 0.0, 1.0
        for j in range(nu // 2):                     # powers 0, 2, ..., nu - 2
            total += term
            term *= c2 * (2.0 * j + 1.0) / (2.0 * j + 2.0)
        a = sn * total
    return max(0.0, 1.0 - a)


def _outlier(t: np.ndarray, xy: np.ndarray, alpha: float, origin=(0.0, 0.0)) -> int | None:
    """The one point of a line fit through (t, xy) that is an outlier, or None (sonar-ljh, per-axis form after review;
    the test is ours, built from the standard deleted-residual results of least squares).

    A measured position's error is not one variance in both coordinates: in range it is the range error (about 1 m), across
    the line of sight range x bearing error (tens of metres at 12 km). So the residuals are taken in range and cross-range:
    rotated onto u, the unit vector from `origin` (the sonar, in the track's frame) to the points' mean, and its normal. One
    rotation for the track, so that each axis's residuals are those of that axis's own straight-line fit (rotating each
    residual by its own point's line of sight breaks that: at four to six points it dropped 4 to 96 % of clean tracks).
    Origin at the points' mean: the frame's own axes. Where the line of sight turns along the track, by up to angle d from
    u, the other axis's error leaks into an axis by sin d and makes its variance grow toward the track's ends; so an axis is
    tested only while the other's scatter times max sin d is at most half its own (its variance at most 1.25 times larger
    at the ends; scatter sqrt(SS / (n - 2))). Without that rule the test dropped 2.4 to 8 % of clean ten- and twenty-point
    tracks with cross-range error 50 times the range error; with it every case measured is within Monte Carlo error of
    alpha (test_classify).
    For axis a and point i the externally studentized residual t_ai = r_ai / sqrt(s2_ai (1 - h_i)), with r_ai the residual of
    the fit through all points, h_i = 1/n + (t_i - mean t)^2 / sum (t - mean t)^2 its leverage and
    s2_ai = (SS_a - r_ai^2 / (1 - h_i)) / (n - 3) the variance of the line through the other points, is Student's t with
    n - 3 degrees of freedom for independent Gaussian errors of each axis's own variance (point i is independent of the line
    through the others). The point of largest |t_ai| over both axes (ties: the earliest point, range before cross-range) is
    an outlier when its two-sided P(|T| > |t_ai|) is below alpha / m, m = n x the axes tested (Bonferroni). An axis whose
    residuals are at rounding level (RMS at most 1e-9 of the largest coordinate, or 1e-9 m: a line without scatter in it)
    is not tested. Needs four points; a point whose removal leaves no time between the others (1 - h_i <= 1e-12) is no
    candidate; alpha <= 0 (or NaN): off."""
    n = len(t)
    if not alpha > 0.0 or n < 4:
        return None
    tm = t.mean()
    tc = t - tm
    stt = float((tc**2).sum())
    v = (tc[:, None] * xy).sum(axis=0) / stt
    mean = xy.mean(axis=0)
    r = xy - (mean + tc[:, None] * v)
    # One rotation for the track: range along u, from the origin to the points' mean, cross-range along its normal.
    d = mean - np.asarray(origin, dtype=float)
    dist = float(np.hypot(d[0], d[1]))
    ux, uy = (float(d[0] / dist), float(d[1] / dist)) if dist > 0.0 and np.isfinite(dist) else (1.0, 0.0)
    axes = (r[:, 0] * ux + r[:, 1] * uy, r[:, 0] * uy - r[:, 1] * ux)      # range, cross-range
    # How far each point's own line of sight turns from u: the other axis's error leaks into an axis by its sine.
    p = xy - np.asarray(origin, dtype=float)
    pd = np.hypot(p[:, 0], p[:, 1])
    turn = 0.0
    if dist > 0.0 and np.isfinite(dist):
        for i in range(n):
            if pd[i] > 0.0:
                turn = max(turn, abs(float(p[i, 0] * uy - p[i, 1] * ux)) / float(pd[i]))
    tol = 1e-9 * max(1.0, float(np.abs(xy).max()))
    one_minus_h = 1.0 - (1.0 / n + tc**2 / stt)
    nu = n - 3
    best, best_t2 = None, -1.0
    ss = [float((ra**2).sum()) for ra in axes]
    # An axis is tested when its residuals are above rounding level and the other axis's error, leaking in by the turn of
    # the line of sight, is at most half its own scatter (variance at most 1.25 times larger at the track's ends).
    sd = [np.sqrt(q / (n - 2)) for q in ss]
    live = [np.sqrt(ss[k] / n) > tol and sd[1 - k] * turn <= 0.5 * sd[k] for k in (0, 1)]
    tested = sum(live)
    if tested == 0:
        return None
    for i in range(n):
        if not one_minus_h[i] > 1e-12:
            continue
        for k, ra in enumerate(axes):
            if not live[k]:
                continue
            d2 = float(ra[i]) ** 2 / one_minus_h[i]
            rest = (ss[k] - d2) / nu
            t2 = d2 / rest if rest > 0.0 else (np.inf if d2 > 0.0 else 0.0)
            if t2 > best_t2:
                best, best_t2 = i, t2
    if best is None:
        return None
    return best if t_two_sided(float(np.sqrt(best_t2)), nu) < alpha / (n * tested) else None


def _delays_only(t: np.ndarray, pings: np.ndarray | None) -> bool:
    """The fitted points' times differ by their echo delays as much as by their pings (sonar-054, ours): each point's time is
    its ping's transmit time plus its delay R/c (D-156), and the transmit times span no more than the delays do,
    ptp(pings) <= ptp(t - pings). A line through such points measures range against R/c, a radial speed of the order of
    the sound speed: pings that all carry one transmit time (a source without timestamps) gave tens of km/s. A real
    target never trips it: its delays span ptp(R)/c <= (|dR/dt| / c) ptp(pings), less than ptp(pings) at any radial
    speed below c. No ping times recorded (None, empty, or a row without one): not tested, as before."""
    if pings is None or not len(pings) or not np.all(np.isfinite(pings)):
        return False
    return bool(np.ptp(pings) <= np.ptp(t - pings))


def _fit(history, at, velocity, measured, outlier_alpha, origin=(0.0, 0.0), pings=None):
    """fit_velocity's line, and the number of points it went through."""
    h = np.asarray(history, dtype=float)
    t, xy = h[:, 0], h[:, 1:3]
    last_t, last = t[-1], xy[-1]
    p = None
    if pings is not None and len(pings):
        p = np.asarray(pings, dtype=float).ravel()
        if len(p) != len(t):
            raise ValueError(f"fit_velocity: {len(p)} ping times for {len(t)} history rows")
    from_measured = False
    if measured is not None and len(measured):
        m = np.asarray(measured, dtype=float).reshape(-1, 2)
        if len(m) != len(t):
            raise ValueError(f"fit_velocity: {len(m)} measured rows for {len(t)} history rows")
        if not np.isnan(m).all():                # all NaN: none recorded, as an empty list (classify.cpp the same)
            ok = np.all(np.isfinite(m), axis=1)
            t, xy = t[ok], m[ok]
            p = None if p is None else p[ok]
            from_measured = True
    if len(t) < 3 or np.ptp(t) <= 0.0 or _delays_only(t, p):
        pos = last
        if _given(at) and velocity is not None and np.all(np.isfinite(velocity)):
            pos = pos + (float(at) - last_t) * np.asarray(velocity, dtype=float)
        return pos, np.zeros(2), 0.0, np.inf, len(t)
    if from_measured:
        out = _outlier(t, xy, outlier_alpha, origin)
        if out is not None:
            keep = np.arange(len(t)) != out
            t, xy = t[keep], xy[keep]
    tc = t - t.mean()
    v = (tc[:, None] * xy).sum(axis=0) / (tc**2).sum()
    fit = xy.mean(axis=0) + tc[:, None] * v
    res = xy - fit
    # Two coordinates, two fitted parameters each.
    rms = float(np.sqrt((res**2).sum() / max(2 * len(t) - 4, 1)))
    # No time given: the last history row's (the track's last ping), also when the measured rows end earlier (a track
    # that missed its last pings), as the short-track rule above.
    pos = xy.mean(axis=0) + ((float(at) if _given(at) else last_t) - t.mean()) * v
    return pos, v, rms, float(rms / np.sqrt((tc**2).sum())), len(t)


def fit_velocity(history, at: float | None = None, velocity=None, measured=None,
                 outlier_alpha: float = OUTLIER_ALPHA, origin=(0.0, 0.0), pings=None) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Least-squares straight line through (time, x, y) points.
    Returns the position on the line at `at` (s; None or NaN: at the time of the last row of `history`), velocity, RMS residual and the
    velocity's sigma. With fewer than three points, or no time between them (max - min), no fitted velocity and the
    last point, moved to `at` by `velocity` ((vx, vy), the track's Kalman state's) when both are given.

    `pings` (a track's `Track.pings`: the transmit time of each row of `history`; sonar-054): also no fitted velocity
    when the fitted points' transmit times span no more than their echo delays (row time less ping time) do
    (`_delays_only`): pings that all carry one transmit time. None or empty: not tested.

    `measured` (a track's `Track.measured`: (x, y) per row of `history`, NaN on a row of a miss): the line goes through
    the measured positions at their rows' times, the rows without one (not finite) left out (D-157: the filtered
    positions of a track born at zero speed lag behind a mover, +39 m along track at 12 km with the live chain's
    tracker; the measurements do not). With fewer than three measured rows (or no time between them) the short-track rule above, from the last row of
    `history` (the filtered position, where the state's `velocity` holds). None, empty or all NaN (none recorded): fit
    `history` itself.

    `outlier_alpha` (sonar-ljh): through measured positions, at most one point is left out of the line, the one the
    deleted-residual test (`_outlier`) finds an outlier at this false-alarm probability per track; 0, the default (opt-in,
    D-159), leaves every point in. A glinting echo (popoto_12km: the ship side-on at the last ping, 23 to 66 m ahead of its centre along
    track) otherwise pulls the line by its leverage at the end of the track. Filtered history is never tested: its
    points are not independent. `origin` ((x, y) in the track's frame): where the sonar is, which sets the range and
    cross-range axes the test takes the residuals in (`_outlier`); (0, 0), the frame's origin, by default."""
    return _fit(history, at, velocity, measured, outlier_alpha, (0.0, 0.0) if origin is None else origin, pings)[:4]


def fitted_points(track, outlier_alpha: float = OUTLIER_ALPHA, origin=(0.0, 0.0)) -> int:
    """How many points `fit_velocity` fits for a track: its measured rows (`Track.measured`, D-157), less a point the
    outlier test leaves out (`outlier_alpha`, about `origin`), or its history rows when it records none."""
    return int(_fit(track.history, None, None, getattr(track, "measured", None), outlier_alpha,
                    (0.0, 0.0) if origin is None else origin, getattr(track, "pings", None))[4])


def classify_tracks(tracks: list[Track], *, min_speed: float = 0.5, static_speed: float = 0.5,
                    significance: float = 5.0, min_points: int = 4, min_seen: float = 0.7,
                    max_fit_rms: float | None = None, multipath: dict | None = None,
                    linear: dict | None = None, detections: dict | None = None, time: float | None = None,
                    outlier_alpha: float = OUTLIER_ALPHA, max_fit_rms_bearing: float = 0.0,
                    origin=None) -> list[TrackClass]:
    """Classify confirmed tracks.

    `min_speed` (m/s) and `significance` (fitted speed over its sigma, raised for short tracks
    by Student's t: 5 becomes 6.6 at eight points and 13 at four) make a mover;
    `static_speed` and `min_seen` make a static object; `max_fit_rms` (m) rejects tracks
    whose positions scatter more than the sensor's accuracy allows. `multipath` switches on
    echo grouping: {"max_excess_range": m, "bearing": rad, "velocity": m/s, "origin": (x, y)}.
    `linear` switches on the test for movement along a row of tracks:
    {"length": m, "width": m}: another track ahead and another astern, within `width` of the line
    through this track along its velocity and no further than `length` along it.
    `detections` switches on the rules from the detections themselves (module docstring):
    {"min_mover_cells": cells or None, "row_snr_db": dB or None}; None or {} changes nothing.
    `time` (s, the tracker's clock): every position is the fitted line's at that time, the report's (D-156: a track's
    last point is its last detection's time, transmit + range / c, not the ping's); None: at each track's last history
    point (its last ping, also when it missed it), which is the ping's time wherever detections carry no time of their own; NaN as None (the C++ engine's NaN). The
    rules that compare positions (row, echo grouping) then compare them at one time. A track too short for a fit (fewer
    than three points) is moved from its last point to `time` by its Kalman state's velocity (`state[2:4]`), when it
    carries one. So is a track whose points' transmit times (`Track.pings`) span no more than their echo delays
    (sonar-054, `fit_velocity`): pings without timestamps (all at one time) give it no fitted velocity (speed 0), and it
    is never a mover.

    The fitted line, and so every position, velocity, fit_rms and speed_sigma here, goes through the positions the
    track MEASURED (`Track.measured`: the detections it took, at their times), not its filtered history (D-157); a
    track with fewer than three measured rows is a short track as above, from its last filtered point. A track that
    records no measurements (built by hand, a capture saved before D-157) is fitted through its history as before.
    `outlier_alpha` (sonar-ljh): at most one measured point that the deleted-residual test finds an outlier at this
    false-alarm probability per track is left out of the line (`fit_velocity`); 0, the default (opt-in, D-159): every
    point in. The test takes the
    residuals in range and cross-range about `origin` (below), each axis with its own variance. With no origin given and
    no multipath origin it is the frame's origin (0, 0), as libs/live ClassifyOptions::origin_x, origin_y default: the
    sonar's place in a frame that starts at the sonar. Chosen over the residuals' own principal axes, which are fitted to
    the noise and so make the test too ready to drop a point, and over one pooled variance, which drops about 10 % of
    clean tracks when the cross-range error is ten times the range error (review of sonar-ljh). A caller whose sonar is
    elsewhere passes `origin`.
    `max_fit_rms_bearing` (rad, sonar-4jm): where `max_fit_rms` is given, a track's limit is at least this times its
    fitted position's range from `origin` ((x, y); None: the multipath origin, else (0, 0)): the measured positions
    scatter across the line of sight by the range times the bearing error. 0: off (libs/live derives it from the beam).
    """
    det = check_detection_rules(detections)
    if not (0.0 <= outlier_alpha < 1.0):
        raise ValueError(f"classify outlier_alpha = {outlier_alpha}: from 0 (off) to below 1")
    if not (np.isfinite(max_fit_rms_bearing) and max_fit_rms_bearing >= 0.0):
        raise ValueError(f"classify max_fit_rms_bearing = {max_fit_rms_bearing}: a finite angle, zero (off) or more")
    out = []
    here = np.asarray(origin if origin is not None else (multipath or {}).get("origin", (0.0, 0.0)), dtype=float)
    from scipy import stats
    for tr in tracks:
        state = getattr(tr, "state", None)
        pos, v, rms, sigma, n_fit = _fit(tr.history, time, None if state is None or len(state) < 4 else state[2:4],
                                         getattr(tr, "measured", None), outlier_alpha, here, getattr(tr, "pings", None))
        speed = float(np.hypot(*v))
        n = len(tr.history)
        # The fit's scatter is itself estimated from few points (2m - 4 degrees of freedom, m the points fitted: the
        # measured rows, D-157, less an outlier left out), so a short track needs a larger ratio for the same confidence:
        # Student's t against the normal. `min_points` and `seen` still count the track's pings (its history rows).
        dof = max(2 * n_fit - 4, 1)
        need = significance * float(stats.t.ppf(0.9995, dof) / stats.norm.ppf(0.9995))
        seen = tr.hits / max(n, 1)
        snr = float(np.mean(tr.snr)) if getattr(tr, "snr", None) else float("nan")
        limit = max_fit_rms
        if max_fit_rms is not None and max_fit_rms_bearing > 0.0:
            limit = max(max_fit_rms, max_fit_rms_bearing * float(np.hypot(*(np.asarray(pos, dtype=float) - here))))
        steady = max_fit_rms is None or rms <= limit
        if n < min_points or not steady:
            kind = "uncertain"
        elif speed >= min_speed and speed >= need * sigma and seen >= min_seen:
            kind = "mover"                       # a real object is there on most pings; a glint that comes and goes is not
        elif speed < static_speed and seen >= min_seen:
            kind = "static"
        elif speed < need * sigma and seen >= min_seen and speed < 3.0 * static_speed:
            kind = "static"                      # slow and not distinguishable from standing still
        else:
            kind = "uncertain"
        cells = getattr(tr, "cells", None)
        extent = float(np.median(cells)) if cells else float("nan")
        out.append(TrackClass(tr.id, kind, (float(pos[0]), float(pos[1])), (float(v[0]), float(v[1])), speed,
                              sigma, rms, float(seen), snr, extent_cells=extent))
    if linear:
        xy = np.array([c.position for c in out]).reshape(-1, 2)
        for i, c in enumerate(out):
            if c.kind != "mover":
                continue
            # Other tracks in a narrow corridor ahead of and behind this one, along its own velocity.
            u = np.asarray(c.velocity) / c.speed
            d = xy - xy[i]
            along = d @ u
            across = np.abs(d[:, 0] * u[1] - d[:, 1] * u[0])
            in_row = (np.abs(along) <= linear["length"]) & (across <= linear["width"])
            if det["row_snr_db"] is not None:
                # Only echoes of about this one's strength make a row with it; NaN SNR never does.
                snr = np.array([o.mean_snr_db for o in out])
                in_row &= snr >= c.mean_snr_db - det["row_snr_db"]
            # Both ahead and astern: a boat trailing a wake has tracks astern only and stays a mover.
            if int((in_row & (along > 1e-6)).sum()) >= 1 and int((in_row & (along < -1e-6)).sum()) >= 1:
                c.kind = "line"
    if det["min_mover_cells"] is not None:
        # After the corridor test, so that a row of small echoes along a pipeline stays a line; before echo grouping,
        # so that a track too small for an object under way does not explain others as its echoes.
        for c in out:
            if c.kind == "mover" and c.extent_cells < det["min_mover_cells"]:
                c.kind = "uncertain"             # a glint or clutter joined into a line; NaN (no cells) leaves it
    if multipath:
        # Echo grouping (ours, D-155). An arrival by a longer path has the object's bearing and velocity and lies a
        # little farther out in range. In a shallow channel the strongest arrival need not be the direct one (popoto_12km,
        # the ten seed captures of tools/classify_rules_eval.py: the strongest track near the ship is 34 to 93 m beyond its
        # true range on every seed), so the group is gathered around a strong anchor on both sides in range and its object
        # is the EARLIEST established arrival, the one nearest the true range. The group grows transitively: a track joins
        # when it is within the gates of ANY member, until none joins (gates measured to the anchor alone left tracks that
        # match a member but not the anchor as second movers: popoto_12km seeds 3 and 9). A track still uncertain is not
        # made the object (a four-point track's fit is poor), but it can be a member. The cost: two objects at one bearing
        # and velocity within max_excess_range in either direction are one object (before: only one farther out than a
        # stronger one).
        # A group extends no farther than max_excess_range from its object in range (sonar-a5x): grown transitively, a
        # chain of arrivals linked two objects farther apart than one gate (two copies of the popoto_12km ship's arrivals
        # at one bearing and velocity were one object up to 900 m apart, the gate 800). A member farther than that from
        # the object is left out of the group and free for a group of its own; when it is the anchor, the anchor groups
        # again from what is left. Every measured echo group lies well inside (at most 193 m beyond its object over the
        # 60 Popoto captures of tools/classify_rules_eval.py), so the growth through the bearing and velocity gates stays.
        origin = np.asarray(multipath.get("origin", (0.0, 0.0)), dtype=float)
        polar = {}
        for c in out:
            d = np.asarray(c.position) - origin
            polar[c.id] = (float(np.hypot(*d)), float(np.arctan2(d[0], d[1])))
        established = ("mover", "static", "line")

        def linked(a, b):
            """Within the range, bearing and velocity gates of each other (False when anything is NaN)."""
            return bool(abs(polar[a.id][0] - polar[b.id][0]) <= multipath["max_excess_range"]
                        and abs(np.angle(np.exp(1j * (polar[a.id][1] - polar[b.id][1])))) <= multipath["bearing"]
                        and np.hypot(a.velocity[0] - b.velocity[0], a.velocity[1] - b.velocity[1]) <= multipath["velocity"])

        # Anchors strongest first (NaN SNR last); an anchor is itself an established track not yet in a group.
        order = sorted(out, key=lambda c: -(c.mean_snr_db if np.isfinite(c.mean_snr_db) else -np.inf))
        taken = set()
        for anchor in order:
            if anchor.kind not in established or not linked(anchor, anchor):
                continue                         # NaN range, bearing or velocity: nothing to group by; left as it is
            while anchor.id not in taken:        # again when the anchor itself lay beyond its group's object
                group, grew = [anchor], True
                while grew:
                    grew = False
                    for c in out:
                        if c.id not in taken and all(c is not g for g in group) and any(linked(c, g) for g in group):
                            group.append(c)
                            grew = True
                # The anchor is established, so an established member exists. Ties in range: the lowest id.
                obj = min((c for c in group if c.kind in established), key=lambda c: (polar[c.id][0], c.id))
                for c in group:
                    if abs(polar[c.id][0] - polar[obj.id][0]) > multipath["max_excess_range"]:
                        continue                 # beyond one gate of the object: not its echo (sonar-a5x)
                    taken.add(c.id)
                    if c is not obj:
                        c.kind, c.parent = "multipath", obj.id
    return out


DETECTION_RULES = ("min_mover_cells", "row_snr_db")


def check_detection_rules(detections: dict | None) -> dict:
    """The `detections` option with every rule named (None: off); refuses unknown keys and negative values."""
    detections = dict(detections or {})
    unknown = set(detections) - set(DETECTION_RULES)
    if unknown:
        raise ValueError(f"classify detections: unknown key(s) {sorted(unknown)}; known: {list(DETECTION_RULES)}")
    out = {k: (None if detections.get(k) is None else float(detections[k])) for k in DETECTION_RULES}
    for k, v in out.items():
        if v is not None and not v >= 0.0:
            raise ValueError(f"classify detections.{k} = {v}: must be zero or more (None switches it off)")
    return out


def summary(classes: list[TrackClass]) -> dict:
    """Counts by kind."""
    kinds = ("mover", "static", "line", "multipath", "uncertain")
    return {k: sum(c.kind == k for c in classes) for k in kinds}
