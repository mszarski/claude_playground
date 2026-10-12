import random

from rmr.idle import IDLES, MOODY, SLEEPY, pick
from rmr.mood import HALF_LIFE, Mood, label


def test_mood_moves_toward_what_was_read_and_fades():
    m = Mood(now=0)
    for t in range(4):
        m.update("sad", valence=0.2, arousal=0.2, now=t)
    s = m.state(now=4)
    assert s["valence"] < -0.4 and s["arousal"] < -0.2 and label(s) == "subdued"
    later = m.state(now=4 + HALF_LIFE)
    assert abs(later["valence"] - s["valence"] / 2) < 1e-3       # half-life
    assert label(m.state(now=4 + 10 * HALF_LIFE)) == "calm"
    m.update("happy", now=1000)
    assert m.state(now=1000)["valence"] > 0


def test_idles_follow_the_mood():
    all_idles = {**IDLES, **SLEEPY, **MOODY}
    rng = random.Random(0)
    sad = [pick(20, rng, {"valence": -0.7, "arousal": -0.5})[0] for _ in range(200)]
    assert sad.count("sigh") > 40 and "bounce" not in sad
    happy = [pick(20, rng, {"valence": 0.7, "arousal": 0.5})[0] for _ in range(200)]
    assert happy.count("bounce") > 40 and "sigh" not in happy
    assert all(n in all_idles for n in sad + happy)
    assert pick(20, random.Random(1)) == pick(20, random.Random(1), None)
