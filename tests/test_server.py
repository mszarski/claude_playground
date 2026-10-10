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
