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


def samples_from_moves(moves):
    out = []
    for _, m in moves:
        for B in augment(move_to_traj(m)):
            B = B[:BUCKETS[-1]]
            out.append((B, PL.frames(PL.extract(B), len(B))))
    return out


def fit_stats(samples):
    X = np.concatenate([B for B, _ in samples])
    Q = np.concatenate([P for _, P in samples])
    return {"MU": X.mean(0).tolist(), "SD": (X.std(0) + 1e-6).tolist(),
            "PMU": Q.mean(0).tolist(), "PSD": (Q.std(0) + 1e-6).tolist()}


def bucketize(samples, stats, device):
    """Group samples into fixed-length padded batches; ``M`` marks real (1) vs padding (0) frames."""
    MU, SD, PMU, PSD = (np.array(stats[k]) for k in ("MU", "SD", "PMU", "PSD"))
    out = []
    for lo, hi in zip([0] + BUCKETS[:-1], BUCKETS):
        S = [(B, P) for B, P in samples if lo < len(B) <= hi]
        if not S:
            continue
        X = np.zeros((len(S), hi, 9), np.float32)
        Q = np.zeros((len(S), hi, 8), np.float32)
        M = np.zeros((len(S), hi), np.float32)
        for i, (B, P) in enumerate(S):
            n = len(B)
            X[i, :n], Q[i, :n], M[i, :n] = (B - MU) / SD, (P - PMU) / PSD, 1
        out.append({k: torch.tensor(v, device=device) for k, v in {"X": X, "Q": Q, "M": M}.items()})
    return out
