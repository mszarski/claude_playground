"""Run the voice loop on a Reachy Mini (or its simulator): listen, nod along, answer with a generated move.

  # the brain: the server with a voice model (a laptop, or the Space)
  python -m rmr.server --voice-model mszarski/reachy-voice:student/v2 --ckpt hf://mszarski/reachy-motion-generator/generator_v2.pt
  # the body: on the robot, or any machine that reaches its daemon
  python -m rmr.robot --server http://localhost:7860                       # robot microphone, hands-free
  python -m rmr.robot --server http://localhost:7860 --wav talk.wav        # a recording instead of the mic
  python -m rmr.robot --play move.json                                     # just play a move

Every 1/25 s the loop reads the microphone, steps ``rmr.listen.Listener`` and, unless it is answering, sends the
listening pose with ``ReachyMini.set_target``. When a turn ends (0.9 s of silence after 0.6 s of speech) it posts the
turn to ``/api/respond?stream=1`` and plays the first move as soon as the server sends it, frame by frame at 25 Hz,
then eases back into listening. Turns that end while it is answering are dropped, like the viewer's hands-free mode.

The robot client needs only ``reachy_mini`` and numpy; the models stay on the server. A simulated robot for testing:
``reachy-mini-daemon --sim --headless --no-media`` (MuJoCo), then ``--wav``.
"""
import io
import json
import threading
import time
import urllib.request

import numpy as np

from .listen import Listener
from .motion import FPS

SR = 16000
PREROLL = 0.3     # s of audio kept before the turn's first speech frame


def wav_bytes(x, sr=SR):
    import soundfile as sf

    b = io.BytesIO()
    sf.write(b, np.asarray(x, np.float32), sr, format="WAV", subtype="PCM_16")
    return b.getvalue()


def respond_stream(server, wav, n=1, headers=None):
    """POST a turn to the server; yield its stages (heard, motion, done)."""
    req = urllib.request.Request(f"{server.rstrip('/')}/api/respond?stream=1&n={n}&seed={int(time.time()) % 100000}",
                                 wav, {"content-type": "audio/wav", **(headers or {})})
    with urllib.request.urlopen(req, timeout=300) as r:
        for line in r:
            if line.strip():
                part = json.loads(line)
                if part.get("stage") == "error":
                    raise RuntimeError(part["detail"])
                yield part


def listening_target(state):
    """``Listener.step`` output (9-DoF) -> ``(head 4x4, antennas)`` for ``set_target``."""
    from .motion import traj_to_move

    f = traj_to_move(np.asarray([state]))["set_target_data"][0]
    return np.asarray(f["head"]), np.asarray(f["antennas"])


class Body:
    """Sends poses to the robot at 25 Hz: the listening pose, or a move when one is playing."""

    def __init__(self, mini):
        self.mini, self.move, self.i, self.lock = mini, None, 0, threading.Lock()

    def play(self, move):
        with self.lock:
            self.move, self.i = move["set_target_data"], 0

    @property
    def busy(self):
        return self.move is not None

    def tick(self, listening_state):
        with self.lock:
            if self.move is not None:
                f = self.move[self.i]
                self.i += 1
                if self.i >= len(self.move):
                    self.move = None
                self.mini.set_target(head=np.asarray(f["head"]), antennas=list(f["antennas"]),
                                     body_yaw=float(f.get("body_yaw", 0.0)))
                return
        head, ant = listening_target(listening_state)
        self.mini.set_target(head=head, antennas=list(ant), body_yaw=0.0)


def mic_frames(mini=None, wav=None, tail=4.0):
    """Yield 1/25 s frames of 16 kHz mono audio, in real time: from a file (then ``tail`` s of silence) or the robot."""
    hop = SR // FPS
    if wav is not None:
        from .voice import load_audio

        x = np.concatenate([load_audio(wav), np.zeros(int(tail * SR), np.float32)])
        t0 = time.monotonic()
        for i in range(len(x) // hop):
            time.sleep(max(0.0, t0 + i / FPS - time.monotonic()))
            yield x[i * hop:(i + 1) * hop]
        return
    from scipy.signal import resample_poly

    mini.media.start_recording()
    rate, buf = mini.media.get_input_audio_samplerate(), np.zeros(0, np.float32)
    while True:
        s = mini.media.get_audio_sample()
        if s is None:
            time.sleep(0.005)
            continue
        s = s.mean(1) if s.ndim == 2 else s
        if rate != SR:
            s = resample_poly(s, SR, rate).astype(np.float32)
        buf = np.concatenate([buf, s])
        while len(buf) >= hop:
            yield buf[:hop]
            buf = buf[hop:]


def run(mini, server, frames, n=1, headers=None, log=print):
    """The voice loop. Returns the list of ``(t, event)`` it went through (for tests and logs)."""
    lis, body, ring, events = Listener(), Body(mini), [], []

    def answer(audio, t):
        t0 = time.monotonic()
        try:
            for part in respond_stream(server, wav_bytes(audio), n, headers):
                waited = f"+{time.monotonic() - t0:.1f}s"
                if part["stage"] == "heard":
                    log(f"  {waited:>7} heard: {part['heard'].get('text')!r} (voice: {part['heard'].get('emotion')})")
                elif part["stage"] == "motion":
                    body.play(part["moves"][0])
                    events.append((round(lis.t, 2), "answer"))
                    log(f"  {waited:>7} moving: {part.get('idea') or part['recipe']}")
                elif part["stage"] == "done" and part.get("reading"):
                    log(f"  {waited:>7} reading: {part['reading']}")
        except Exception as e:
            log(f"[{t:5.1f}s] response failed: {type(e).__name__}: {e}")
        finally:
            pending.clear()

    pending = []
    for x in frames:
        ring.append(x)
        del ring[:-30 * FPS]
        state = lis.step(float(10 * np.log10(np.mean(np.asarray(x, np.float64) ** 2) + 1e-10)))
        body.tick(state)
        if lis.turn_end and not pending and not body.busy:
            start, end = lis.turn_end
            k = int(round((end - start + PREROLL) * FPS))
            audio = np.concatenate(ring[-k:])
            events.append((round(lis.t, 2), "turn"))
            log(f"[{lis.t:5.1f}s] your turn ended ({len(audio) / SR:.1f} s), thinking…")
            pending.append(threading.Thread(target=answer, args=(audio, lis.t), daemon=True))
            pending[0].start()
    while pending or body.busy:            # let the last answer finish
        body.tick(lis.step(-100.0))
        time.sleep(1 / FPS)
    events += [e for e in lis.events if e[1] in ("nod", "nod2", "perk")]
    return sorted(events)


def main():
    import argparse

    ap = argparse.ArgumentParser(prog="python -m rmr.robot", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server", default="http://localhost:7860", help="rmr.server (with --voice-model) or the Space")
    ap.add_argument("--wav", help="use a recording instead of the robot's microphone")
    ap.add_argument("--play", help="just play this move JSON and exit")
    ap.add_argument("--host", default="reachy-mini.local")
    ap.add_argument("--n", type=int, default=1)
    a = ap.parse_args()
    from reachy_mini import ReachyMini

    with ReachyMini(host=a.host, media_backend="no_media" if (a.wav or a.play) else "default") as mini:
        _main(a, mini)


def _main(a, mini):
    import os

    if a.play:
        with open(a.play) as f:
            move = json.load(f)
        body = Body(mini)
        body.play(move)
        while body.busy:
            body.tick(None)
            time.sleep(1 / FPS)
        return
    headers = {"authorization": f"Bearer {os.environ['HF_TOKEN']}"} if "hf.space" in a.server else None
    run(mini, a.server, mic_frames(mini, a.wav), a.n, headers)


if __name__ == "__main__":
    main()
