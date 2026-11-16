"""Image quality metrics used by tests, benchmarks and RESULTS.md."""

from __future__ import annotations

import numpy as np


def width_at_level(x, profile, level_db: float = -3.0) -> float:
    """Full width of the main lobe of `profile` (magnitudes) at `level_db` below its peak.

    Crossings are found by linear interpolation on either side of the peak sample.
    Returns NaN when the profile does not fall below the level inside the window.
    """
    x = np.asarray(x, dtype=float)
    a = np.abs(np.asarray(profile))
    i = int(np.argmax(a))
    thr = a[i] * 10.0 ** (level_db / 20.0)

    def crossing(step):
        j = i
        while 0 <= j + step < len(a) and a[j + step] >= thr:
            j += step
        k = j + step
        if not 0 <= k < len(a):
            return np.nan
        # a[j] >= thr > a[k]
        f = (a[j] - thr) / (a[j] - a[k])
        return x[j] + f * (x[k] - x[j])

    left, right = crossing(-1), crossing(+1)
    return float(abs(right - left))


def peak_sidelobe_db(profile) -> float:
    """Highest sidelobe relative to the main peak, dB (negative)."""
    a = np.abs(np.asarray(profile))
    i = int(np.argmax(a))
    # Walk down both sides of the main lobe to its first nulls.
    lo = i
    while lo > 0 and a[lo - 1] < a[lo]:
        lo -= 1
    hi = i
    while hi < len(a) - 1 and a[hi + 1] < a[hi]:
        hi += 1
    side = np.concatenate([a[:lo], a[hi + 1:]])
    if side.size == 0:
        return float("-inf")
    return float(20.0 * np.log10(side.max() / a[i]))


def db(x, floor: float = -300.0):
    """20 log10 |x| relative to the maximum, floored."""
    a = np.abs(np.asarray(x))
    peak = a.max() if a.size else 1.0
    with np.errstate(divide="ignore"):
        out = 20.0 * np.log10(a / peak)
    return np.maximum(out, floor)
