"""Instant reactions: a short move for the feeling Reachy has just read, while the full response is still being written.

The responder writes the feeling first (rmr.audio_responder), about half a second into its answer; the response
recipe takes another two or three seconds. Rather than sit still, Reachy starts reacting as soon as the feeling is
known: ears and head go where the feeling points (down for sad, up for happy, a calm settle for anger), and the
viewer glides from there into the full response when it arrives (visualizer/src/Player.js blends moves). Each is a
recipe, so the generator gives it organic detail and no two are the same.
"""
REACTIONS = {
    "neutral": "go .4 r=6 e=10 p=-2 E=.6 | hold .8 E=.5",
    "happy": "go .3 e=-5 p=-5 z=8 E=2 | hold .8 E=1",
    "sad": "go .6 e=70 p=6 z=0 E=.4 | hold .8 E=.3",
    "angry": "go .5 e=25 p=0 z=2 E=.3 | hold .8 E=.3",
    "anxious": "go .5 e=30 p=3 z=1 r=-4 E=.6 | hold .8 E=.5",
    "surprised": "go .2 e=-10 p=-6 z=10 E=2 | hold .8 E=1",
}


def reaction(feeling):
    """-> (name, recipe) for an instant reaction to ``feeling`` (rmr.respond FEELINGS)."""
    f = feeling if feeling in REACTIONS else "neutral"
    return f"reacting: {f}", REACTIONS[f]
