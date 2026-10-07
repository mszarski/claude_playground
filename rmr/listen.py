"""Listening behaviour: what Reachy does *while* someone is talking to it.

People listening don't stand still and they don't wait for the end of the sentence to react. They lean in,
nod at the pauses between phrases, and perk up when the speaker's voice lifts. This module does that from the
loudness of the speaker's voice alone, frame by frame, so it runs live with no model and no delay:

* **attentive pose**: while someone is speaking (or spoke in the last few seconds), lean in slightly, tilt the
  head and lift the antennas a little; relax back to neutral after a long silence.
* **nods**: at a pause (>= ``PAUSE`` s of silence) after at least ``MIN_TALK`` s of speech, nod once, or twice
  after a long stretch of speech. Never more often than every ``REFRACTORY`` s.
* **antenna perks**: when the voice rises sharply (fast loudness envelope ``RISE_DB`` above the slow one), flick
  the antennas up and lift the head a touch, then let them settle.

``Listener`` is a 25 Hz state machine fed one loudness value per frame; ``listening_move`` runs it over a whole
recording. ``visualizer/src/Listen.js`` is a line-for-line port driven by the live microphone; keep the
constants and the order of the update steps identical in both.

Output is a ``(T, 9)`` trajectory (``rmr.motion.DOF``), in SI units like everything else.
"""
import math

import numpy as np

from .motion import FPS, traj_to_move

# Detection (dB, seconds)
FLOOR_RISE = 0.004     # the noise floor creeps up by this fraction of the gap per frame, and drops instantly
VOICE_DB = 9.0         # a frame is speech when it is this far above the noise floor ...
MIN_DB = -55.0         # ... and above this absolute level (dBFS)
HANGOVER = 0.2         # speech continues through gaps shorter than this
PAUSE = 0.3            # silence that counts as a pause between phrases
MIN_TALK = 1.0         # speech needed before a pause earns a nod
DOUBLE_TALK = 3.0      # ... and a double nod
REFRACTORY = 1.2       # minimum time between two nods
PERK_GAP = 3.0         # ... and between two antenna perks, which lose their meaning when frequent
ENGAGED = 4.0          # stay attentive this long after the last speech
FAST, SLOW = 0.5, 0.05 # one-pole smoothing of the loudness envelopes
RISE_DB = 8.0          # fast envelope this far above the slow one = the voice lifted

# Gestures (plan units: deg, mm)
NOD_DEG, NOD_S = 7.0, 0.55        # head dip and its duration; the second nod of a double is 0.6x, 0.8x as long
NOD_Z = -2.0                      # the head also sinks a little on each nod
PERK_EARS, PERK_ATTACK, PERK_DECAY = 25.0, 0.12, 0.8   # antennas lift by up to 25 deg, settle with tau 0.8 s
PERK_PITCH, PERK_Z = -3.0, 3.0    # and the head lifts a touch
ATTENTIVE = {"ears": 6.0, "pitch": -2.0, "roll": 6.0, "z": 5.0}   # leaning in, head tilted, antennas up
IDLE = {"ears": 15.0, "pitch": 0.0, "roll": 0.0, "z": 3.0}        # rmr.recipe.NEUTRAL
POSE_TAU = 0.6                    # seconds to settle into / out of the attentive pose
BREATH_MM, BREATH_S = 1.2, 4.0    # slow breathing on z, so the robot never freezes


def _bump(u):
    """0 -> 1 -> 0 over u in [0, 1], quicker down than up (a nod drops, then recovers)."""
    if u <= 0 or u >= 1:
        return 0.0
    return math.sin(math.pi * u ** 0.7) ** 2


class Listener:
    """Feed ``step(db)`` once per frame (``1 / FPS`` s) with the speaker's loudness in dBFS; it returns the 9-DoF
    pose for that frame. ``events`` collects ``(t, "nod" | "nod2" | "perk")`` for inspection."""

    def __init__(self, side=1.0):
        self.dt = 1.0 / FPS
        self.t = 0.0
        self.floor = None
        self.fast = self.slow = None
        self.since_voice = 1e9     # s since the last speech frame
        self.talk = 0.0            # s of speech since the last nod
        self.last_nod = self.last_perk = -1e9
        self.nods = []             # (start, amplitude, duration)
        self.perk_t = -1e9
        self.side = side           # which way the head tilts; alternates every time the speaker starts again
        self.engaged = 0.0         # 0 = idle pose, 1 = attentive pose (smoothed)
        self.events = []

    def step(self, db):
        t, dt = self.t, self.dt
        # noise floor: drops at once, creeps up slowly, so it tracks the room, not the voice
        if self.floor is None or db < self.floor:
            self.floor = db
        else:
            self.floor += FLOOR_RISE * (db - self.floor)
        voiced = db > self.floor + VOICE_DB and db > MIN_DB
        if voiced:
            if self.since_voice > ENGAGED:
                self.side = -self.side
            self.since_voice = 0.0
            self.fast = db if self.fast is None else self.fast + FAST * (db - self.fast)
            self.slow = db if self.slow is None else self.slow + SLOW * (db - self.slow)
        else:
            self.since_voice += dt
        speaking = self.since_voice < HANGOVER
        if speaking:
            self.talk += dt

        # a pause after a phrase: nod (twice after a long stretch)
        if (not speaking and self.since_voice >= PAUSE and self.talk >= MIN_TALK
                and t - self.last_nod >= REFRACTORY):
            double = self.talk >= DOUBLE_TALK
            self.nods.append((t, 1.0, NOD_S))
            if double:
                self.nods.append((t + NOD_S * 0.9, 0.6, NOD_S * 0.8))
            self.events.append((round(t, 2), "nod2" if double else "nod"))
            self.last_nod, self.talk = t, 0.0
        # the voice lifts: perk the antennas
        if (voiced and self.fast is not None and self.fast - self.slow > RISE_DB
                and t - self.last_perk >= PERK_GAP):
            self.perk_t = self.last_perk = t
            self.events.append((round(t, 2), "perk"))

        target = 1.0 if self.since_voice < ENGAGED else 0.0
        self.engaged += (target - self.engaged) * (1 - math.exp(-dt / POSE_TAU))
        pose = {k: IDLE[k] + self.engaged * (ATTENTIVE[k] - IDLE[k]) for k in IDLE}
        pose["roll"] *= self.side

        nod = sum(a * _bump((t - s) / d) for s, a, d in self.nods)
        self.nods = [n for n in self.nods if t - n[0] < n[2]]
        u = t - self.perk_t
        perk = 0.0 if u < 0 else (u / PERK_ATTACK if u < PERK_ATTACK else math.exp(-(u - PERK_ATTACK) / PERK_DECAY))

        ears = pose["ears"] - PERK_EARS * perk
        pitch = pose["pitch"] + NOD_DEG * nod + PERK_PITCH * perk
        z = pose["z"] + NOD_Z * nod + PERK_Z * perk + BREATH_MM * math.sin(2 * math.pi * t / BREATH_S)
        self.t += dt
        r = math.radians
        # plan units -> trajectory: earR = -antenna_right, earL = +antenna_left (rmr.plan.posture)
        return [0.0, 0.0, z / 1000, r(pose["roll"]), r(pitch), 0.0, -r(ears), r(ears), 0.0]


def loudness(audio, sr, fps=FPS):
    """Mono float audio -> loudness in dBFS, one value per ``1 / fps`` s frame (RMS over the frame)."""
    hop = int(round(sr / fps))
    n = len(audio) // hop
    x = np.asarray(audio[: n * hop], np.float64).reshape(n, hop)
    return 10 * np.log10(np.mean(x ** 2, axis=1) + 1e-10)


def listening_traj(audio, sr, tail=1.5):
    """Run ``Listener`` over a recording (plus ``tail`` s of silence, so the last phrase gets its nod).
    Returns ``(trajectory (T, 9), events)``."""
    db = np.concatenate([loudness(audio, sr), np.full(int(tail * FPS), -100.0)])
    lis = Listener()
    A = np.array([lis.step(float(v)) for v in db])
    return A, lis.events


def listening_move(audio, sr, description="listening"):
    A, events = listening_traj(audio, sr)
    return traj_to_move(A, description), events


def render(audio_path, out, width=520, height=420):
    """Render the listening move for ``audio_path`` to ``out`` (.mp4, with the speech as its soundtrack)."""
    import os
    import subprocess

    import imageio_ffmpeg

    from .reach import Reach
    from .renderer.outputs import _label, _write
    from .renderer.sim import Sim
    from .voice import load_audio

    sr = 16000
    move, events = listening_move(load_audio(audio_path), sr)
    Reach().project(move)
    frames = Sim(width, height).play(move)
    shown = {int(round(t * FPS)): e for t, e in events}
    label, until = "", -1
    for i in range(len(frames)):
        if i in shown:
            label, until = {"nod": "nod", "nod2": "double nod", "perk": "antennas perk"}[shown[i]], i + FPS
        frames[i] = _label(frames[i], label if i < until else "")
    silent = out[:-4] + ".silent.mp4"
    _write(silent, frames)
    # pad the speech with silence to the video's length, so the closing nod isn't cut off
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", silent, "-i", audio_path,
                    "-c:v", "copy", "-c:a", "aac", "-af", f"apad=whole_dur={len(frames) / FPS:.2f}", out], check=True)
    os.remove(silent)
    return move, events


def main():
    import argparse
    import json

    ap = argparse.ArgumentParser(prog="python -m rmr.listen", description="Listening move (nods, antenna perks) "
                                 "for a recording of someone talking: rendered video with the speech, and/or move JSON.")
    ap.add_argument("audio")
    ap.add_argument("--video", help="output .mp4 (needs the render extra)")
    ap.add_argument("--json", help="output move .json")
    a = ap.parse_args()
    if a.video:
        move, events = render(a.audio, a.video)
    else:
        from .voice import load_audio
        move, events = listening_move(load_audio(a.audio), 16000)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(move, f)
    print(" ".join(f"{t:.1f}s {e}" for t, e in events))


if __name__ == "__main__":
    main()
