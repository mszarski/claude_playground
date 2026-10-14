import pytest

pytest.importorskip("fastapi")
torch = pytest.importorskip("torch")
from fastapi.testclient import TestClient  # noqa: E402

from rmr import server  # noqa: E402
from rmr.generator.data import fit_stats, samples_from_moves  # noqa: E402
from rmr.generator.model import MotionGenerator  # noqa: E402
from test_generator import TINY, _moves  # noqa: E402


def _plan(prompt):
    if prompt == "fail":
        raise ValueError("the planner did not produce a valid recipe")
    return "perk up", "go .3 e=-10 p=-8 z=10 E=4 | hold 1 E=1"


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    ckpt = tmp_path_factory.mktemp("ck") / "tiny.pt"
    net = MotionGenerator(**TINY)
    torch.save({"sd": net.state_dict(), "stats": fit_stats(samples_from_moves(_moves())), "config": net.config}, ckpt)
    engine = server.Engine(str(ckpt))
    engine.plan = _plan
    return TestClient(server.create_app(engine))


def test_generate_returns_reachable_moves(client):
    assert client.get("/api/health").json()["ok"]
    r = client.post("/api/generate", json={"prompt": "a dog hearing its name", "n": 2, "seed": 1})
    assert r.status_code == 200
    d = r.json()
    assert d["recipe"].startswith("go .3") and d["idea"] == "perk up" and len(d["moves"]) == 2
    assert all(len(m["set_target_data"]) == len(m["time"]) > 25 for m in d["moves"])
    assert set(d["timing_ms"]) == {"planner", "generator", "total"}


def test_generate_rejects_bad_requests(client):
    assert client.post("/api/generate", json={"prompt": ""}).status_code == 422
    assert client.post("/api/generate", json={"prompt": "x", "n": 9}).status_code == 422
    assert client.post("/api/generate", json={"prompt": "fail"}).status_code == 422
    assert client.get("/").status_code == 200          # the viewer


class _Heard:
    def hear(self, audio):
        return {"text": "I got the job!", "emotion": "happy", "confidence": 0.8, "probs": {}}


class _Student:
    def answers(self, heard):
        r = {"feeling": "happy", "response": "sharing the joy. You perk up.", "recipe": "go .3 e=-10 p=-8 z=10 E=4 | hold 1"}
        yield True, {**r, "reading": ""}
        yield False, {**r, "reading": "They're thrilled."}


def test_respond_streams_motion_before_reading(client):
    eng = client.app.state.engine
    eng.listener, eng.voice_model, eng.student = _Heard(), "fake", _Student()
    r = client.post("/api/respond?stream=1&n=1", content=b"\0" * 2000)
    parts = [__import__("json").loads(line) for line in r.text.splitlines()]
    assert [p["stage"] for p in parts] == ["heard", "motion", "done"]
    assert parts[1]["moves"] and "reading" not in parts[1] and parts[2]["reading"] == "They're thrilled."
    whole = client.post("/api/respond", content=b"\0" * 2000).json()       # non-streaming: merged
    assert whole["reading"] == "They're thrilled." and whole["moves"] and whole["heard"]["text"] == "I got the job!"
    assert {"listen", "first_motion", "total"} <= set(whole["timing_ms"])


class _Seq:
    def __init__(self):
        self.n, self.seen = 0, []

    def hear(self, audio):
        self.n += 1
        return {"text": f"line {self.n}", "emotion": "neutral", "confidence": 0.5, "probs": {}}


class _Recorder(_Student):
    def __init__(self):
        self.contexts = []

    def answers(self, heard):
        self.contexts.append(heard.get("context"))
        yield from super().answers(heard)


def test_session_memory_gives_the_last_lines_as_context(client):
    eng = client.app.state.engine
    eng.listener, eng.voice_model, eng.student, eng.history = _Seq(), "fake", _Recorder(), {}
    for session in ("a", "a", "a", "b"):
        client.post(f"/api/respond?session={session}", content=b"\0" * 2000)
    assert eng.student.contexts == [None, ["line 1"], ["line 1", "line 2"], None]


def test_idle_endpoint_returns_a_move(client):
    r = client.get("/api/idle?silence=90&seed=3")
    assert r.status_code == 200
    d = r.json()
    from rmr.idle import IDLES, SLEEPY
    assert d["idea"] in {**IDLES, **SLEEPY} and d["moves"] and len(d["moves"][0]["time"]) > 25


class _Loud(_Seq):
    def hear(self, audio):
        return {**super().hear(audio), "arousal": 0.7, "dominance": 0.7, "valence": 0.65}


class _Valence(_Student):
    def __init__(self):
        self.valence = []

    def answers(self, heard):
        self.valence.append(heard["valence"])
        yield from super().answers(heard)


def test_calibration_per_session(client):
    eng = client.app.state.engine
    eng.listener, eng.voice_model, eng.student, eng.calibrate = _Loud(), "fake", _Valence(), True
    try:
        for _ in range(30):
            client.post("/api/respond?session=c", content=b"\0" * 2000)
        client.post("/api/respond?session=d", content=b"\0" * 2000)
    finally:
        eng.calibrate = False
    v = eng.student.valence
    assert v[0] > v[29] and v[29] < 0.5 and v[30] == v[0]       # drifts to the reference; a new session starts afresh


def test_feedback_logs_rated_turns_and_mood_follows_the_conversation(client, tmp_path):
    import json

    from rmr.feedback import FeedbackLog

    eng = client.app.state.engine
    eng.listener, eng.voice_model, eng.student, eng.moods = _Heard(), "fake", _Student(), {}
    assert client.post("/api/feedback", json={"turn": "x", "rating": 1}).status_code == 503   # logging off
    eng.feedback = FeedbackLog(str(tmp_path))
    parts = [json.loads(line) for line in client.post("/api/respond?stream=1&session=s1", content=b"\1" * 2000).text.splitlines()]
    turn = parts[0]["turn"]
    assert all(p["turn"] == turn for p in parts)
    assert parts[-1]["mood"]["valence"] > 0                      # a happy turn lifts the mood
    assert client.get("/api/mood?session=s1").json()["valence"] > 0
    assert client.get("/api/mood?session=other").json()["valence"] == 0
    assert client.post("/api/feedback", json={"turn": "nope", "rating": 1}).status_code == 404
    assert client.post("/api/feedback", json={"turn": turn, "rating": 2}).status_code == 422
    assert client.post("/api/feedback", json={"turn": turn, "rating": -1}).json()["ok"]
    assert client.post("/api/feedback", json={"turn": turn, "rating": 1, "save_audio": True}).json()["ok"]
    rows = [json.loads(x) for x in (tmp_path / "ratings.jsonl").read_text().splitlines()]
    assert [r["rating"] for r in rows] == [-1, 1]
    assert rows[0]["audio"] is None and rows[1]["audio"] == f"audio/{turn}.wav"
    assert (tmp_path / rows[1]["audio"]).read_bytes() == b"\1" * 2000
    assert rows[0]["recipe"] and rows[0]["heard"]["text"] == "I got the job!" and rows[0]["reading"] == "They're thrilled."
    assert "_audio" not in rows[0]


def test_puppeteered_clips_are_saved(client, tmp_path):
    import json

    import numpy as np

    from rmr.feedback import FeedbackLog

    eng = client.app.state.engine
    eng.feedback = FeedbackLog(str(tmp_path))
    fr = [{"head": np.eye(4).tolist(), "antennas": [-0.2, 0.2], "body_yaw": 0}] * 50
    move = {"time": [i / 25 for i in range(50)], "set_target_data": fr}
    r = client.post("/api/clips", json={"label": "sheepish", "description": "caught out", "move": move})
    assert r.status_code == 200
    saved = json.loads((tmp_path / "clips" / f"{r.json()['id']}.json").read_text())
    assert saved["label"] == "sheepish" and len(saved["move"]["set_target_data"]) == 50
    assert client.post("/api/clips", json={"label": "Bad!", "move": move}).status_code == 422
    assert client.post("/api/clips", json={"label": "ok", "move": {**move, "time": move["time"][:10]}}).status_code == 422


class _AudioStudent:
    def __init__(self):
        self.got = []

    def stages(self, heard, wav):
        self.got.append((heard["text"], len(wav)))
        r = {"feeling": "sad", "response": "gentle. You lower your ears.", "recipe": "go 1 e=90 p=8 E=1 | hold 1"}
        yield "feeling", "sad"
        yield "early", {**r, "reading": ""}
        yield "final", {**r, "reading": "They sound low."}


def test_audio_responder_hears_the_recording(client, monkeypatch):
    import json

    import numpy as np

    from rmr import voice

    eng = client.app.state.engine
    st = _AudioStudent()
    eng.listener, eng.audio_model, eng.audio_student = _Heard(), "fake-audio", st
    monkeypatch.setattr(voice, "load_audio", lambda b: np.zeros(len(b) // 2, np.float32))
    try:
        parts = [json.loads(x) for x in client.post("/api/respond?stream=1", content=b"\0" * 4000).text.splitlines()]
    finally:
        eng.audio_model, eng.audio_student = None, None
    assert [p["stage"] for p in parts] == ["heard", "react", "motion", "done"]
    assert parts[1]["feeling"] == "sad" and parts[1]["moves"] and "react" in parts[1]["timing_ms"]   # react at once
    assert parts[2]["feeling"] == "sad" and parts[2]["moves"] and parts[3]["reading"] == "They sound low."
    assert st.got == [("I got the job!", 2000)]                 # the transcript and the waveform both reach it


class _SlowHeard:
    def hear(self, audio):
        import time
        time.sleep(0.5)
        return {"text": "I got the job!", "emotion": "happy", "confidence": 0.8, "probs": {}, "valence": 0.8,
                "arousal": 0.7}


def test_transcript_free_audio_responder_starts_before_whisper_ends(client, monkeypatch):
    import json

    import numpy as np

    from rmr import voice

    eng = client.app.state.engine
    st = _AudioStudent()
    seen = []
    orig = st.stages

    def stages(heard, wav):
        seen.append(dict(heard))
        yield from orig({**heard, "text": heard.get("text", "")}, wav)
    st.stages = stages
    eng.listener, eng.audio_model, eng.audio_student, eng.audio_transcript = _SlowHeard(), "fake", st, False
    eng.history, eng.moods = {}, {}
    monkeypatch.setattr(voice, "load_audio", lambda b: np.zeros(len(b) // 2, np.float32))
    try:
        for i in range(2):
            parts = [json.loads(x) for x in client.post("/api/respond?stream=1&session=p", content=b"\0" * 4000)
                     .text.splitlines()]
            stages_seen = [p["stage"] for p in parts]
            assert stages_seen.index("react") < stages_seen.index("heard") < stages_seen.index("done")
            assert parts[stages_seen.index("react")]["timing_ms"]["react"] < 450    # before Whisper (0.5 s) ended
    finally:
        eng.audio_model, eng.audio_student, eng.audio_transcript = None, None, True
    assert seen[0] == {"context": [], "no_transcript": True}
    assert seen[1]["context"] == ["I got the job!"]           # the first turn's transcript became memory
    assert parts[-1]["stage"] == "done" and parts[-1]["mood"]["valence"] < 0.5  # mood still follows (sad answers)
