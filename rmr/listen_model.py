"""A learned listener: head motion while someone else talks, learned from real listeners.

Trained on Meta's Seamless Interaction dataset (CC-BY-NC 4.0; ``scripts/extract_listening.py``): for each
conversation, the partner's voice (loudness and voice activity, 25 Hz) and the listener's head rotation from the
dataset's face tracker. Axis 0 of that rotation is pitch with + = head down (checked against facial keypoints:
nose-to-shoulder distance), 1 is yaw, 2 is roll; the same convention as ``rmr.plan``'s pitch.

The model is a causal GRU. Each frame it reads the speaker features and its own previous motion and outputs, per axis,
a categorical distribution over the change of the head angle (``BINS`` bins over +-``DMAX`` deg per frame); sampling
it gives varied, human-like motion instead of the over-smoothed mean a regression would give. It learns the *fast*
part of head motion (the angle minus its 0.3 Hz low-pass: nods, tilts, small turns), not where the person happens to
be looking; the robot's posture still comes from ``rmr.listen``.

Because the weights come from non-commercial data, they are CC-BY-NC as well and stay optional: ``rmr.listen``'s
rules are the default, and ``LearnedHead`` only replaces their head pitch/roll/yaw when weights are given.
"""
import numpy as np

FPS = 25
BINS = 31
DMAX = 3.0            # deg per frame (75 deg/s): covers > 99.9% of listening frames
N_IN = 6              # speaker loudness, speaker speaking, onset, offset, + 2 slow context features
HIDDEN = 96
HP_ALPHA = 1 - np.exp(-2 * np.pi * 0.3 / FPS)     # one-pole 0.3 Hz low-pass, per frame


class SpeakerFeatures:
    """Causal speaker features, one frame at a time: loudness (dBFS) and speaking -> ``(N_IN,)``.

    Loudness is relative to the speaker's own running speech level, so it doesn't matter how loud the mic is."""

    def __init__(self):
        self.ref, self.a = None, 1 - np.exp(-1 / (FPS * 20))      # ~20 s memory of the speech level
        self.prev, self.since, self.talk = 0.0, 0.0, 0.0

    def step(self, db, vad):
        v = 1.0 if vad else 0.0
        if vad:
            self.ref = db if self.ref is None else self.ref + self.a * (db - self.ref)
        level = -60.0 if self.ref is None else self.ref
        rel = min(15.0, max(-40.0, db - level)) / 20
        onset, offset = max(0.0, v - self.prev), max(0.0, self.prev - v)
        self.prev = v
        self.since = 0.0 if (onset or offset) else self.since + 1 / FPS     # s since voice activity changed
        self.talk = self.talk + 1 / FPS if vad else 0.0                   # s spoken in the current stretch
        return np.array([rel, v, onset, offset, np.tanh(self.since / 2), np.tanh(self.talk / 5)], np.float32)


def speaker_features(db, vad):
    """Partner loudness (dBFS) and voice activity at 25 Hz -> ``(T, N_IN)`` (``SpeakerFeatures`` over a recording)."""
    sf = SpeakerFeatures()
    return np.stack([sf.step(float(d), bool(v)) for d, v in zip(db, vad)])


def fast_motion(rot, valid):
    """Head rotation (T, 3) rad -> its fast part in degrees: the angle minus a 0.3 Hz low-pass (gaps filled)."""
    from scipy.signal import butter, filtfilt

    deg = np.degrees(np.asarray(rot, np.float64))
    idx = np.arange(len(deg))
    if valid.sum() < 2:
        return np.zeros_like(deg)
    filled = np.stack([np.interp(idx, idx[valid], deg[valid, k]) for k in range(3)], -1)
    b, a = butter(2, 0.3, fs=FPS)
    return (filled - filtfilt(b, a, filled, axis=0)).astype(np.float32)


def to_bins(d):
    return np.clip(np.round((d + DMAX) / (2 * DMAX) * (BINS - 1)), 0, BINS - 1).astype(np.int64)


def from_bins(i):
    return np.asarray(i, np.float64) / (BINS - 1) * 2 * DMAX - DMAX


def make_net():
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = nn.GRU(N_IN + 6, HIDDEN, batch_first=True)
            self.head = nn.Linear(HIDDEN, 3 * BINS)

        def forward(self, x, h=None):
            y, h = self.gru(x, h)
            return self.head(y).view(*y.shape[:2], 3, BINS), h

    return Net()


class LearnedHead:
    """Streaming inference in numpy: ``step(features)`` -> sampled (pitch, yaw, roll) of the fast head motion, deg.
    ``weights``: dict of arrays (``export``'s JSON)."""

    def __init__(self, weights, temperature=0.8, seed=0):
        w = {k: np.asarray(v, np.float64) for k, v in weights.items()}
        self.Wi, self.Wh, self.bi, self.bh = w["gru.weight_ih_l0"], w["gru.weight_hh_l0"], w["gru.bias_ih_l0"], \
            w["gru.bias_hh_l0"]
        self.Wo, self.bo = w["head.weight"], w["head.bias"]
        self.h = np.zeros(HIDDEN)
        self.y = np.zeros(3)
        self.d = np.zeros(3)
        self.slow = np.zeros(3)
        self.temperature = temperature
        self.rng = np.random.default_rng(seed)

    def logits(self, f):
        x = np.concatenate([f, self.y / 10, self.d / DMAX])
        gi, gh = self.Wi @ x + self.bi, self.Wh @ self.h + self.bh
        H = HIDDEN
        r = 1 / (1 + np.exp(-(gi[:H] + gh[:H])))
        z = 1 / (1 + np.exp(-(gi[H:2 * H] + gh[H:2 * H])))
        n = np.tanh(gi[2 * H:] + r * gh[2 * H:])
        self.h = (1 - z) * n + z * self.h
        return (self.Wo @ self.h + self.bo).reshape(3, BINS)

    def step(self, f, u=None):
        """-> (pitch, yaw, roll) of the fast head motion, deg. ``u``: optional (3,) uniforms for the draws
        (tests); otherwise the generator's."""
        lg = self.logits(f) / max(self.temperature, 1e-3)
        p = np.exp(lg - lg.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        u = self.rng.random(3) if u is None else u
        k = np.array([min(int(np.searchsorted(np.cumsum(p[i]), u[i])), BINS - 1) for i in range(3)])
        self.d = from_bins(k)
        self.y = self.y + self.d
        # the model learned the angle minus its 0.3 Hz low-pass; apply that same high-pass (causally) to what it
        # samples, so small biases don't add up into a slow drift (looking away) the targets never had
        self.slow += HP_ALPHA * (self.y - self.slow)
        return self.y - self.slow


def export(net):
    return {k: v.detach().cpu().numpy().round(6).tolist() for k, v in net.state_dict().items()}
