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
