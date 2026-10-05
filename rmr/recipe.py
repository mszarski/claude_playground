"""Motion recipes: the tiny language the planner LLM writes, expanded into randomised plans.

A recipe is a sequence of segments separated by ``|``::

    go D k=v ...            cosine-ease to the target over D seconds (0.15-0.3 s reads as a snap)
    hold D [E=v]            stay in the current pose, optionally changing the energy
    osc D ch amp per [E=v]  sinusoid on channel ``ch`` (amplitude ``amp``, period ``per`` s) around the pose

Channels: ``e`` (both ears), ``eR``, ``eL``, ``p`` (pitch), ``r`` (roll), ``y`` (yaw), ``z`` (height, mm),
``b`` (body yaw); ``E`` is the energy (how much fast organic detail the generator adds). Units are those
of ``rmr.plan``. Every recipe starts from ``NEUTRAL``.

Example (sneezing)::

    go .6 e=40 p=-6 z=4 E=1 | go 1 p=-16 z=10 e=70 E=3 | hold .6 E=6 | go .12 p=18 z=-8 e=100 E=10 | go 1 p=2 z=2 e=30 E=1 | hold .6

Reference: ``planner/dsl.py`` in pham-tuan-binh/reachy-motion-generator (Apache-2.0).
"""
import numpy as np

from .motion import FPS
from .plan import CH, KDT, lowpass

# Column of each recipe key in the expanded (T, 8) frame array, which follows ``rmr.plan.CH``.
KEYMAP = {"eR": 0, "eL": 1, "p": 2, "r": 3, "y": 4, "z": 5, "b": 6, "E": 7}
NEUTRAL = np.array([15., 15., 0., 0., 0., 3., 0., 0.5])
LIMITS = {"e": (-25, 175), "eR": (-25, 175), "eL": (-25, 175), "p": (-30, 30), "r": (-25, 25),
          "y": (-50, 50), "z": (-25, 25), "b": (-60, 60), "E": (0, 12)}
OSC_CHANNELS = ("e", "eR", "eL", "p", "r", "y", "z", "b")
SCALED = ("p", "r", "y", "z", "b")   # head/body targets that follow the variant's amplitude scale
MAX_SECONDS = 30.0


class RecipeError(ValueError):
    pass


def _number(text, segment):
    try:
        return float(text)
    except ValueError:
        raise RecipeError(f"bad number {text!r} in {segment!r}") from None


def _apply_targets(tokens, target, amp, segment):
    for tok in tokens:
        if "=" not in tok:
            raise RecipeError(f"expected key=value, got {tok!r}")
        k, v = tok.split("=", 1)
        if k not in LIMITS:
            raise RecipeError(f"unknown channel {k!r} (use e eR eL p r y z b E)")
        v = _number(v, segment)
        lo, hi = LIMITS[k]
        if not lo <= v <= hi:
            raise RecipeError(f"{k}={v:g} outside [{lo}, {hi}]")
        if k in SCALED:
            v *= amp
        if k == "e":
            target[0] = target[1] = v
        else:
            target[KEYMAP[k]] = v


def expand(recipe, rng=None, amp=1.0, tempo=1.0):
    """Recipe -> ``(T, 8)`` per-frame ``[earR earL pitch roll yaw z body energy]`` at ``FPS`` (unfiltered).

    ``amp`` scales head/body targets, ``tempo`` scales durations; ``rng`` adds ±15% per-segment timing jitter
    (and ±20% oscillation amplitude). Raises ``RecipeError`` on anything malformed or out of range.
    """
    rng = rng or np.random.default_rng(0)
    cur = NEUTRAL.copy()
    out = [cur.copy()]
    segments = [s.split() for s in recipe.split("|") if s.strip()]
    if not segments:
        raise RecipeError("empty recipe")
    for tok in segments:
        seg_text = " ".join(tok)
        cmd = tok[0]
        if cmd not in ("go", "hold", "osc") or len(tok) < 2:
            raise RecipeError(f"bad segment {seg_text!r}")
        d = _number(tok[1], seg_text)
        if not 0.05 <= d <= 10:
            raise RecipeError(f"duration {d:g} outside [0.05, 10] s")
        d *= tempo * rng.uniform(.85, 1.15)
        n = max(1, int(round(d * FPS)))

        if cmd in ("go", "hold"):
            target = cur.copy()
            _apply_targets(tok[2:], target, amp, seg_text)
            if cmd == "hold":
                target[:7] = cur[:7]          # hold may only change energy
            w = 0.5 - 0.5 * np.cos(np.pi * np.arange(1, n + 1) / n)
            seg = cur + w[:, None] * (target - cur)
        else:
            if len(tok) < 5:
                raise RecipeError(f"osc needs: osc D ch amp period, got {seg_text!r}")
            ch = tok[2]
            if ch not in OSC_CHANNELS:
                raise RecipeError(f"osc on unknown channel {ch!r}")
            a, period = _number(tok[3], seg_text), _number(tok[4], seg_text)
            if period < 0.3:                  # validate what was written, before jitter
                raise RecipeError("osc period below 0.3 s: fast shaking belongs in E (energy), not osc")
            a *= amp * rng.uniform(.8, 1.2)
            period *= rng.uniform(.85, 1.15)
            target = cur.copy()
            _apply_targets(tok[5:], target, amp, seg_text)
            u = np.arange(1, n + 1) / FPS
            fade = np.minimum(1, np.minimum(u, u[-1] - u + 1 / FPS) / .3)   # 0.3 s fade in and out
            wave = np.sin(2 * np.pi * u / period) * fade
            seg = np.repeat(cur[None], n, 0)
            seg[:, 7] = np.linspace(cur[7], target[7], n)
            for j in ([0, 1] if ch == "e" else [KEYMAP[ch]]):
                seg[:, j] += a * wave
        out.extend(seg)
        cur = seg[-1].copy()

    F = np.asarray(out)
    if len(F) / FPS > MAX_SECONDS:
        raise RecipeError(f"recipe lasts {len(F) / FPS:.1f} s; keep it under {MAX_SECONDS:g} s")
    return F


def to_plan(F, rng=None, mirror=False, ear_jitter=6.0, fc=1.0, kdt=KDT):
    """Expanded frames -> plan: posture low-passed at ``fc`` Hz, one key every ``kdt`` s.

    The defaults match ``rmr.plan.extract`` (the generator's training plans). Ears get a small random
    offset because real clips are never perfectly symmetric.
    """
    rng = rng or np.random.default_rng(0)
    F = F.copy()
    F[:, 0] += rng.normal(0, ear_jitter)
    F[:, 1] += rng.normal(0, ear_jitter)
    if mirror:
        F[:, [3, 4, 6]] *= -1
        F[:, [0, 1]] = F[:, [1, 0]]
    S = lowpass(F[:, :7], fc) if fc else F[:, :7]
    T = len(F)
    keys = []
    for t in np.arange(0, T / FPS + 1e-9, kdt):
        i = min(T - 1, int(round(t * FPS)))
        keys.append({"t": round(float(t), 2), **{c: round(float(S[i, k]), 1) for k, c in enumerate(CH[:7])},
                     "energy": round(float(max(0, F[i, 7])), 1)})
    return {"duration": round(T / FPS, 2), "keys": keys}


def variants(recipe, n, seed=0, fc=1.0, kdt=KDT):
    """``n`` randomised plans from one recipe: amplitude x0.75-1.25, tempo x0.8-1.25, odd variants mirrored."""
    rng = np.random.default_rng(seed)
    plans = []
    for v in range(n):
        F = expand(recipe, rng, amp=rng.uniform(.75, 1.25), tempo=rng.uniform(.8, 1.25))
        plans.append(to_plan(F, rng, mirror=bool(v % 2), fc=fc, kdt=kdt))
    return plans


def check(recipe):
    """``None`` if the recipe is valid, else the error message."""
    try:
        expand(recipe)
    except RecipeError as e:
        return str(e)
    return None
