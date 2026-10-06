"""Training data: real moves -> (motion, its own plan) pairs, augmented, normalised, length-bucketed.

Reference: ``generator/data.py`` in pham-tuan-binh/reachy-motion-generator (Apache-2.0).
"""
import numpy as np
import torch
from scipy.signal import butter, filtfilt

from .. import plan as PL
from ..motion import FPS, mirror, move_to_traj, stretch

BUCKETS = [104, 176, 296, 496, 720]   # frames at 25 fps (4.2 s ... 28.8 s)
STRETCH = (0.8, 1.0, 1.25)


def augment(A):
    """Mirror x time-stretch = 6 variants per clip. Each variant gets its own extracted plan,
    so a stretched clip is paired with a stretched plan."""
    for mA in (A, mirror(A)):
        for f in STRETCH:
            yield stretch(mA, f)


def simplified_plan(B, P, rng):
    """``P`` with its posture replaced by a smoother, sparser extraction, like a recipe's (energy unchanged).

    Extracted plans keep traces of the fast motion in their 1 Hz posture curves, and the generator learns to read
    liveliness from them instead of from the energy channel; recipe plans are smooth, so their energy was mostly
    ignored. Training on these plans too makes the energy channel carry the detail (``train(simplify=...)``)."""
    S = PL.frames(PL.extract(B, fc=rng.uniform(0.3, 0.6), kdt=rng.choice([0.5, 0.75, 1.0])), len(B))
    S[:, 7] = P[:, 7]
    return S


TREMOR_BAND = (1.5, 4.0)                              # Hz: above the plan's 1 Hz posture, below the 4 Hz low-pass
TREMOR_SCALE = np.array([0.6, 1.0, 0.8, 2.5, 2.5])     # roll pitch yaw earR earL, relative to the amplitude


def add_tremor(A, rng, amp_deg=None):
    """``A`` with band-limited trembling on the head rotations and antennas over a random 1-4 s window.

    In Pollen's clips high energy mostly comes from sharp transitions; trembling on a still pose, which recipes
    ask for (``hold 3 E=7``), is rare, so the generator cannot learn it. The plan is re-extracted from the
    augmented motion, so its energy channel measures exactly the detail that was added."""
    T = len(A)
    amp = rng.uniform(1, 7) if amp_deg is None else amp_deg
    n = min(T, int(rng.uniform(1, 4) * FPS))
    s = int(rng.integers(0, T - n + 1))
    env = np.zeros(T)
    env[s:s + n] = np.sin(np.linspace(0, np.pi, n)) ** 0.5
    b, a = butter(2, [f / (FPS / 2) for f in TREMOR_BAND], "band")
    noise = filtfilt(b, a, rng.standard_normal((T, 5)), axis=0)
    noise /= noise[s:s + n].std(0) + 1e-9
    B = A.copy()
    B[:, 3:8] += noise * np.radians(amp) * TREMOR_SCALE * env[:, None]
    return B


def samples_from_moves(moves, simplified=False, tremor=0.0, seed=0):
    """``[(motion, plan)]``, or ``[(motion, plan, simplified plan)]`` with ``simplified``. With ``tremor``, that
    share of the augmented clips also gets a copy with synthetic trembling (``add_tremor``)."""
    rng, out = np.random.default_rng(seed), []
    for _, m in moves:
        for B in augment(move_to_traj(m)):
            B = B[:BUCKETS[-1]]
            for C in [B] + ([add_tremor(B, rng)] if tremor and rng.random() < tremor else []):
                P = PL.frames(PL.extract(C), len(C))
                out.append((C, P, simplified_plan(C, P, rng)) if simplified else (C, P))
    return out


def fit_stats(samples):
    X = np.concatenate([s[0] for s in samples])
    Q = np.concatenate([s[1] for s in samples])
    return {"MU": X.mean(0).tolist(), "SD": (X.std(0) + 1e-6).tolist(),
            "PMU": Q.mean(0).tolist(), "PSD": (Q.std(0) + 1e-6).tolist()}


def bucketize(samples, stats, device):
    """Group samples into fixed-length padded batches; ``M`` marks real (1) vs padding (0) frames, and ``Q2``
    holds the simplified plans when the samples carry them."""
    MU, SD, PMU, PSD = (np.array(stats[k]) for k in ("MU", "SD", "PMU", "PSD"))
    out = []
    for lo, hi in zip([0] + BUCKETS[:-1], BUCKETS):
        S = [s for s in samples if lo < len(s[0]) <= hi]
        if not S:
            continue
        X = np.zeros((len(S), hi, 9), np.float32)
        Q = np.zeros((len(S), hi, 8), np.float32)
        M = np.zeros((len(S), hi), np.float32)
        arrays = {"X": X, "Q": Q, "M": M}
        if len(S[0]) == 3:
            arrays["Q2"] = np.zeros((len(S), hi, 8), np.float32)
        for i, s in enumerate(S):
            n = len(s[0])
            X[i, :n], Q[i, :n], M[i, :n] = (s[0] - MU) / SD, (s[1] - PMU) / PSD, 1
            if len(s) == 3:
                arrays["Q2"][i, :n] = (s[2] - PMU) / PSD
        out.append({k: torch.tensor(v, device=device) for k, v in arrays.items()})
    return out
