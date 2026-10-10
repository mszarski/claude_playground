import numpy as np

from rmr import robot
from rmr.motion import FPS
from test_listen import _speech


class FakeMini:
    def __init__(self):
        self.sent = []

    def set_target(self, head=None, antennas=None, body_yaw=None):
        assert head.shape == (4, 4) and len(antennas) == 2
        self.sent.append((head, list(antennas)))


def test_voice_loop_listens_then_answers(monkeypatch):
    move = {"time": [i / FPS for i in range(10)],
            "set_target_data": [{"head": np.eye(4).tolist(), "antennas": [0.5, -0.5], "body_yaw": 0.0}] * 10}
    turns = []

    def fake_server(server, wav, n=1, headers=None):
        turns.append(len(wav))
        yield {"stage": "heard", "heard": {"text": "hi", "emotion": "neutral"}}
        yield {"stage": "motion", "moves": [move], "recipe": "go 1", "idea": "attentive"}
        yield {"stage": "done", "reading": "ok"}

    monkeypatch.setattr(robot, "respond_stream", fake_server)
    x, sr = _speech()
    hop = sr // FPS
    frames = [x[i * hop:(i + 1) * hop] for i in range(len(x) // hop)] + [np.zeros(hop, np.float32)] * 50
    mini = FakeMini()
    events = robot.run(mini, "http://fake", frames, log=lambda *_: None)
    kinds = [e for _, e in events]
    assert len(turns) == 1 and kinds.count("turn") == 1 and kinds.count("answer") == 1 and kinds.count("nod") == 2
    assert 5.6 * sr * 2 < turns[0] < 7 * sr * 2 + 1000            # the whole turn (16-bit WAV), with the preroll
    played = [a for _, a in mini.sent if a == [0.5, -0.5]]
    assert len(played) == 10                                       # every frame of the move, once
    assert len(mini.sent) >= len(frames)                           # and a listening pose on every other frame


def test_idle_moves_in_a_quiet_room_and_stop_when_someone_speaks(monkeypatch):
    idle = {"time": [i / FPS for i in range(100)],
            "set_target_data": [{"head": np.eye(4).tolist(), "antennas": [0.2, -0.2], "body_yaw": 0.0}] * 100}
    asked = []
    monkeypatch.setattr(robot, "idle_move", lambda server, silence, headers=None: asked.append(silence) or idle)
    monkeypatch.setattr(robot, "respond_stream", lambda *a, **k: iter(()))
    hop = 16000 // FPS
    quiet = [np.full(hop, 1e-4, np.float32)] * (int(12.5 * FPS))
    x, sr = _speech(phrases=((0.0, 1.5),), loud_at=99, seconds=2.0)
    speech = [x[i * hop:(i + 1) * hop] for i in range(len(x) // hop)]
    mini = FakeMini()
    events = robot.run(mini, "http://fake", quiet + speech, log=lambda *_: None)
    import time
    time.sleep(0.2)
    kinds = [e for _, e in events]
    assert asked and asked[0] >= 10 and "idle" in kinds
    played = [a for _, a in mini.sent if a == [0.2, -0.2]]
    assert 0 < len(played) < 100                  # the idle started, and speech cut it short
