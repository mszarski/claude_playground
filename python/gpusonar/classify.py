"""Sorting confirmed tracks: which move, which stand still, which are echoes of another.

Rule-based, from the track alone (no image features, no learning):

  mover       it was seen on most pings and its positions fit a straight line through time
              whose speed is well above the scatter of the fit;
  static      it was seen on most pings and its fitted speed is small;
  multipath   it has the bearing and the velocity of a stronger track and lies a little
              further out in range: a later arrival of the same object by a longer path
              (long-range sonar in a channel);
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
    position: tuple           # (x, y) in the tracker's frame at the last ping
    velocity: tuple           # fitted (vx, vy), m/s
    speed: float              # m/s
    speed_sigma: float        # one-sigma uncertainty of the fitted speed, m/s
    fit_rms: float            # RMS distance of the positions from the fitted line, m
    seen: float               # fraction of pings since birth with a detection
    mean_snr_db: float
    parent: int | None = None  # for multipath: the id of the track it echoes
    extent_cells: float = float("nan")   # median flagged map cells of the detections it took (NaN: none recorded)


def fit_velocity(history) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Least-squares straight line through (time, x, y) points.
    Returns position at the last time, velocity, RMS residual and the velocity's sigma."""
    h = np.asarray(history, dtype=float)
    t, xy = h[:, 0], h[:, 1:3]
    if len(t) < 3 or np.ptp(t) <= 0.0:
        return xy[-1], np.zeros(2), 0.0, np.inf
    tc = t - t.mean()
    v = (tc[:, None] * xy).sum(axis=0) / (tc**2).sum()
    fit = xy.mean(axis=0) + tc[:, None] * v
    res = xy - fit
    # Two coordinates, two fitted parameters each.
    rms = float(np.sqrt((res**2).sum() / max(2 * len(t) - 4, 1)))
    return fit[-1], v, rms, float(rms / np.sqrt((tc**2).sum()))


def classify_tracks(tracks: list[Track], *, min_speed: float = 0.5, static_speed: float = 0.5,
                    significance: float = 5.0, min_points: int = 4, min_seen: float = 0.7,
                    max_fit_rms: float | None = None, multipath: dict | None = None,
                    linear: dict | None = None, detections: dict | None = None) -> list[TrackClass]:
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
    """
    det = check_detection_rules(detections)
    out = []
    from scipy import stats
    for tr in tracks:
        pos, v, rms, sigma = fit_velocity(tr.history)
        speed = float(np.hypot(*v))
        n = len(tr.history)
        # The fit's scatter is itself estimated from few points (2n - 4 degrees of freedom), so a
        # short track needs a larger ratio for the same confidence: Student's t against the normal.
        dof = max(2 * n - 4, 1)
        need = significance * float(stats.t.ppf(0.9995, dof) / stats.norm.ppf(0.9995))
        seen = tr.hits / max(n, 1)
        snr = float(np.mean(tr.snr)) if getattr(tr, "snr", None) else float("nan")
        steady = max_fit_rms is None or rms <= max_fit_rms
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
        origin = np.asarray(multipath.get("origin", (0.0, 0.0)), dtype=float)
        polar = {}
        for c in out:
            d = np.asarray(c.position) - origin
            polar[c.id] = (float(np.hypot(*d)), float(np.arctan2(d[0], d[1])))
        # Strongest first: an echo is the weaker, later copy.
        order = sorted(out, key=lambda c: -(c.mean_snr_db if np.isfinite(c.mean_snr_db) else -np.inf))
        for i, weak in enumerate(order):
            for strong in order[:i]:
                # Only a track that is itself classed as an object explains another as its echo: not an echo, and
                # not a track still uncertain (a weaker track beside an unproven one is not thereby explained away).
                if strong.kind not in ("mover", "static", "line"):
                    continue
                excess = polar[weak.id][0] - polar[strong.id][0]
                same_bearing = abs(np.angle(np.exp(1j * (polar[weak.id][1] - polar[strong.id][1])))) <= multipath["bearing"]
                same_velocity = np.hypot(weak.velocity[0] - strong.velocity[0],
                                         weak.velocity[1] - strong.velocity[1]) <= multipath["velocity"]
                if 0.0 < excess <= multipath["max_excess_range"] and same_bearing and same_velocity:
                    weak.kind, weak.parent = "multipath", strong.id
                    break
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
