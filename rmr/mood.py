"""Reachy's mood: a slow emotional state that carries across a conversation and fades when nothing happens.

Each turn moves the mood toward what Reachy read in the person (the responder's ``feeling``) and how they sounded
(the tone models' calibrated valence and arousal). Between turns it relaxes back to neutral with half-life
``HALF_LIFE`` seconds, so a sad conversation leaves Reachy subdued for a minute or two, not forever.

The mood is two numbers in [-1, 1]: ``valence`` (down ... up) and ``arousal`` (calm ... lively). The viewer and the
robot bias the listener with it (rmr.listen ``mood_style``): ear rest height, head height and tilt, how big the nods
and the sway are; the idle moves are picked by it too.
"""
import math
import time

HALF_LIFE = 90.0
GAIN = 0.35                     # how far one turn moves the mood toward what was read
FEELING_VA = {"neutral": (0.0, 0.0), "happy": (0.8, 0.5), "sad": (-0.8, -0.5), "angry": (-0.6, 0.7),
              "anxious": (-0.5, 0.5), "surprised": (0.2, 0.8)}


def clip(x):
    return max(-1.0, min(1.0, x))


class Mood:
    def __init__(self, now=None):
        self.v, self.a, self.t = 0.0, 0.0, time.time() if now is None else now

    def decay(self, now=None):
        now = time.time() if now is None else now
        k = 0.5 ** (max(0.0, now - self.t) / HALF_LIFE)
        self.v, self.a, self.t = self.v * k, self.a * k, now

    def update(self, feeling=None, valence=None, arousal=None, now=None):
        """``feeling``: the responder's read; ``valence`` / ``arousal``: the tone models' readings in 0..1."""
        self.decay(now)
        tv, ta = FEELING_VA.get(feeling or "neutral", (0.0, 0.0))
        if valence is not None:
            tv = 0.7 * tv + 0.3 * clip(2 * valence - 1)
        if arousal is not None:
            ta = 0.7 * ta + 0.3 * clip(2 * arousal - 1)
        self.v = clip(self.v + GAIN * (tv - self.v))
        self.a = clip(self.a + GAIN * (ta - self.a))
        return self.state(self.t)

    def state(self, now=None):
        self.decay(now)
        return {"valence": round(self.v, 3), "arousal": round(self.a, 3)}


def label(state):
    """A word for the mood, for display and for picking idle moves."""
    v, a = state["valence"], state["arousal"]
    if math.hypot(v, a) < 0.2:
        return "calm"
    if v >= 0:
        return "cheerful" if a >= 0 else "content"
    return "tense" if a >= 0 else "subdued"
