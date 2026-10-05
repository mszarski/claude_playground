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
