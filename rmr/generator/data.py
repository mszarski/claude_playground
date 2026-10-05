"""Training data: real moves -> (motion, its own plan) pairs, augmented, normalised, length-bucketed.

Reference: ``generator/data.py`` in pham-tuan-binh/reachy-motion-generator (Apache-2.0).
"""
import numpy as np
import torch

from .. import plan as PL
from ..motion import mirror, move_to_traj, stretch

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


def samples_from_moves(moves, simplified=False, seed=0):
    """``[(motion, plan)]``, or ``[(motion, plan, simplified plan)]`` with ``simplified``."""
    rng, out = np.random.default_rng(seed), []
    for _, m in moves:
        for B in augment(move_to_traj(m)):
            B = B[:BUCKETS[-1]]
            P = PL.frames(PL.extract(B), len(B))
            out.append((B, P, simplified_plan(B, P, rng)) if simplified else (B, P))
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
