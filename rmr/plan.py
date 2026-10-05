"""Motion plan: the interface between the planner (text -> plan) and the generator (plan -> motion).

A plan is ``{"duration": s, "keys": [{"t", "earR", "earL", "pitch", "roll", "yaw", "z", "body", "energy"}]}``
with one keyframe every ``KDT`` seconds:

=========  ===================================================================
earR/earL  antenna droop, deg (0 = straight up, ~150 = fully drooped)
pitch      head pitch, deg (+ = head lowered)
roll, yaw  head roll / yaw, deg
z          head height, mm (+ = up)
body       body yaw, deg
energy     RMS (deg) of the fast (> 1 Hz) detail riding on top of the posture
=========  ===================================================================

Posture channels are low-passed below 1 Hz, so a plan says *where the body is and how lively it is*,
not the individual wiggles. ``extract`` turns any recorded clip into its plan, which is what lets the
generator train on motion that has no text at all.

Reference: ``common/plan.py`` in pham-tuan-binh/reachy-motion-generator (Apache-2.0).
"""
import numpy as np
from scipy.signal import butter, filtfilt

from .motion import FPS

KDT = 0.5
CH = ["earR", "earL", "pitch", "roll", "yaw", "z", "body", "energy"]


def lowpass(x, fc, fps=FPS, order=2):
    """Zero-phase Butterworth low-pass along axis 0; clips shorter than 16 frames pass through."""
    if len(x) < 16:
        return x.copy()
    b, a = butter(order, fc / (fps / 2))
    return filtfilt(b, a, x, axis=0, padlen=min(len(x) - 1, 9))


def posture(A):
    """``(T, 9)`` trajectory -> ``(T, 7)`` posture channels in plan units (earR earL pitch roll yaw z body)."""
    d = np.degrees
    return np.stack([-d(A[:, 6]), d(A[:, 7]), d(A[:, 4]), d(A[:, 3]), d(A[:, 5]), 1000 * A[:, 2], d(A[:, 8])], -1)


def _key(t, values):
    return {"t": round(float(t), 2), **{c: round(float(v), 1) for c, v in zip(CH, values)}}


def extract(A, fc=1.0, kdt=KDT):
    """``(T, 9)`` trajectory -> plan. Energy is the RMS of the high-passed head + antenna detail in a
    window of at least 0.5 s around each key."""
    P = posture(A)
    slow = lowpass(P, fc)
    fast = P[:, :5] - slow[:, :5]
    T = len(A)
    duration = T / FPS
    window = max(KDT, kdt)
    keys = []
    for t in np.arange(0, duration + 1e-9, kdt):
        i = min(T - 1, int(round(t * FPS)))
        lo, hi = max(0, int((t - window / 2) * FPS)), min(T, int((t + window / 2) * FPS) + 1)
        energy = float(np.sqrt((fast[lo:hi] ** 2).mean())) if hi > lo else 0.0
        keys.append(_key(t, [*slow[i], energy]))
    return {"duration": round(duration, 2), "keys": keys}


def frames(plan, T=None):
    """Plan -> ``(T, 8)`` per-frame conditioning, linearly interpolated between keys.
    A channel missing from a key holds the previous key's value (0 before the first)."""
    T = T or max(2, int(round(plan["duration"] * FPS)))
    last = dict.fromkeys(CH, 0.0)
    rows = []
    for k in sorted(plan["keys"], key=lambda k: k["t"]):
        last.update({c: float(k[c]) for c in CH if c in k})
        rows.append([k["t"]] + [last[c] for c in CH])
    R = np.asarray(rows)
    u = np.arange(T) / FPS
    return np.stack([np.interp(u, R[:, 0], R[:, 1 + j]) for j in range(len(CH))], -1)
