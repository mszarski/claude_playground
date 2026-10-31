"""Width profiles: how stroke width varies *along* a stroke (SPEC 5.1).

Stroke width = direction model (the pen) x profile(t), with t in [0, 1]
along the whole stroke. A profile is a small curve through knots
``(t, multiplier)`` joined with smoothstep easing; knot values may be
parameters, so profiles are tunable and overridable by name.
"""
from __future__ import annotations

import numpy as np

from .params import val


class Profile:
    def __init__(self, *knots, name: str | None = None):
        if len(knots) == 1 and isinstance(knots[0], (list, tuple)) and knots and \
                isinstance(knots[0][0], (list, tuple)):
            knots = tuple(knots[0])
        if len(knots) < 1:
            raise ValueError("a profile needs at least one knot")
        self.knots = knots
        self.name = name

    def resolved(self) -> tuple[np.ndarray, np.ndarray]:
        ts = np.array([val(t) for t, _ in self.knots], float)
        ms = np.array([val(m) for _, m in self.knots], float)
        order = np.argsort(ts, kind="stable")
        ts, ms = np.clip(ts[order], 0, 1), ms[order]
        return ts, ms

    def __call__(self, t) -> np.ndarray:
        t = np.asarray(t, float)
        ts, ms = self.resolved()
        if len(ts) == 1:
            return np.full_like(t, ms[0])
        i = np.clip(np.searchsorted(ts, t, side="right") - 1, 0, len(ts) - 2)
        span = np.maximum(ts[i + 1] - ts[i], 1e-9)
        u = np.clip((t - ts[i]) / span, 0, 1)
        s = u * u * (3 - 2 * u)
        return ms[i] + (ms[i + 1] - ms[i]) * s

    def __repr__(self):
        return f"Profile{self.knots}"


def constant(m=1.0) -> Profile:
    return Profile((0, m), (1, m))


def taper(start=0.3, length=0.35, end=1.0, end_length=0.0) -> Profile:
    """Thin at the start (``start`` x width), full after ``length``;
    optionally thinning to ``end`` over the last ``end_length``."""
    return Profile((0, start), (length, 1.0), (1 - end_length, 1.0), (1, end))


def swell(ends=0.6, peak=1.0, at=0.5, spread=0.25) -> Profile:
    """Peaks at ``at`` (e.g. the middle of an s spine)."""
    return Profile((0, ends), (at - spread, (ends + peak) / 2), (at, peak),
                   (at + spread, (ends + peak) / 2), (1, ends))
