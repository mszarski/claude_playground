"""Idle behaviour: what Reachy does when nobody is talking to it.

A robot that freezes between conversations looks switched off. Every so often, when it has heard nothing for a while,
it does something small: looks around, stretches, tilts its head, twitches an antenna, or settles. After a long
silence it gets drowsy. Each idle is a recipe (``rmr.recipe``), so the generator turns it into organic motion and no
two look the same; ``rmr.server`` serves them at ``/api/idle`` and the viewer (hands-free mode) and ``rmr.robot`` ask
for one when the room has been quiet.
"""
import random

IDLES = {
    "look around": "go 1.2 y=25 p=-3 E=1 | hold 1.5 E=0.5 | go 1.5 y=-20 E=1 | hold 1.2 E=0.5 | go 1 y=0 E=0.5",
    "stretch": "go 1.5 z=15 p=-10 e=0 E=1 | hold .8 E=.5 | go 1.2 z=0 p=6 e=40 E=1 | go 1 z=3 p=0 e=15 E=.5",
    "curious tilt": "go .8 r=15 e=5 E=1 | hold 1.5 E=.5 | go .8 r=0 e=15 E=.5",
    "antenna twitch": "go .2 eR=-5 E=2 | go .4 eR=15 E=1 | hold .6 | go .2 eL=-5 E=2 | go .4 eL=15 E=1 | hold 1",
    "settle": "go 1.5 z=0 p=3 E=.5 | osc 4 z 3 2 E=.5 | go 1 z=3 p=0 E=.5",
    "glance up": "go .6 p=-15 e=5 E=1 | hold 1.2 E=.5 | go 1 p=0 e=15 E=.5",
}
SLEEPY = {"drowsy": "go 3 p=15 e=90 z=-5 E=.3 | hold 4 E=.2 | go 1 p=0 e=15 z=3 E=.6"}
QUIET_S = 10.0      # silence before the first idle
GAP_S = (12.0, 25.0)  # then one idle every 12 to 25 s
SLEEPY_S = 60.0     # after this much silence, drowsy idles


def pick(silence_s, rng=random):
    """-> (name, recipe) for an idle after ``silence_s`` seconds of quiet."""
    pool = SLEEPY if silence_s >= SLEEPY_S and rng.random() < 0.6 else IDLES
    name = rng.choice(sorted(pool))
    return name, pool[name]
