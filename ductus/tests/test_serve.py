"""Preview server: session logic and the websocket protocol."""
import base64
import json
import shutil
from pathlib import Path

import pytest

from ductus.overrides import load_overrides
from ductus.serve.server import Session, create_app

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "test_font"


@pytest.fixture
def workdir(tmp_path):
    for f in ("font.py", "overrides.yaml"):
        shutil.copy(EXAMPLE / f, tmp_path / f)
    return tmp_path


def test_edit_writes_yaml_and_recompiles(workdir):
    s = Session(workdir / "font.py", workdir / "overrides.yaml")
    before = s.cf.glyphs["n"].points["n.arch.shoulder"]
    s.handle({"type": "set", "name": "n.arch.shoulder", "values": {"dx": 40, "dy": 0, "dw": 10}})
    assert load_overrides(workdir / "overrides.yaml")["n.arch.shoulder"].dw == 10
    after = s.cf.glyphs["n"].points["n.arch.shoulder"]
    assert after[0] - before[0] == pytest.approx(28, abs=1.01)   # was dx 12, now 40
    assert s.payload()["otf"]


def test_undo_redo(workdir):
    s = Session(workdir / "font.py", workdir / "overrides.yaml")
    s.handle({"type": "set", "name": "o.bowl.top", "values": {"dy": 20}})
    s.handle({"type": "set", "name": "o.bowl.top", "values": {"dy": 30}, "checkpoint": False})  # same drag
    s.handle({"type": "remove", "names": ["n.arch.shoulder"]})
    assert "n.arch.shoulder" not in s.overrides
    s.handle({"type": "undo"})
    assert s.overrides["n.arch.shoulder"].dx == 12 and s.overrides["o.bowl.top"].dy == 30
    s.handle({"type": "undo"})
    assert "o.bowl.top" not in s.overrides           # the whole drag undoes at once
    s.handle({"type": "redo"})
    assert s.overrides["o.bowl.top"].dy == 30
    assert "o.bowl.top" in (workdir / "overrides.yaml").read_text()


def test_params_and_terminals_are_session_state(workdir):
    s = Session(workdir / "font.py", workdir / "overrides.yaml")
    src = (workdir / "font.py").read_text()
    w0 = s.cf.glyphs["n"].advance
    s.handle({"type": "params", "values": {"stem": 150}})
    assert s.cf.params["stem"] == 150 and s.cf.glyphs["n"].advance > w0
    s.handle({"type": "terminals", "name": "slab"})
    assert len(s.cf.glyphs["n"].outline_points) > 0 and s.terminal_name == "slab"
    s.handle({"type": "reset_params"})
    assert s.cf.params["stem"] == 80
    assert (workdir / "font.py").read_text() == src     # never rewrites Python


def test_source_change_and_errors(workdir):
    s = Session(workdir / "font.py", workdir / "overrides.yaml")
    p = workdir / "font.py"
    good = p.read_text()
    p.write_text(good.replace("param(80,", "param(100,"))
    assert s.on_files_changed({p.resolve()})
    assert s.cf.params["stem"] == 100 and s.error is None
    p.write_text(good + "\ndef broken(:\n")
    s.on_files_changed({p.resolve()})
    assert s.error and "SyntaxError" in s.error
    assert s.cf.params["stem"] == 100                   # last good result is kept
    # an external edit to overrides.yaml is picked up; our own writes are not re-read
    (workdir / "overrides.yaml").write_text("n.arch.join: { dx: 0, dy: 15 }\n")
    p.write_text(good)
    s.on_files_changed({p.resolve(), (workdir / "overrides.yaml").resolve()})
    assert set(s.overrides) == {"n.arch.join"} and s.error is None


def test_stale_overrides_reported(workdir):
    s = Session(workdir / "font.py", workdir / "overrides.yaml")
    s.handle({"type": "set", "name": "n.arch.elbow", "values": {"dx": 3}})
    assert any("n.arch.elbow" in w for w in s.cf.warnings)


def test_websocket_protocol(workdir):
    from starlette.testclient import TestClient
    app = create_app(workdir / "font.py", watch=False)
    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        first = json.loads(ws.receive_text())
        assert first["type"] == "font" and set(first["data"]["glyphs"]) == {"n", "h", "m", "o", "H", "O", "space"}
        assert base64.b64decode(first["otf"])[:4] == b"OTTO"
        spec = first["data"]["param_specs"]["stem"]
        assert spec["range"] == [30.0, 180.0]
        ws.send_text(json.dumps({"type": "params", "values": {"stem": 120}, "seq": 7}))
        msg = json.loads(ws.receive_text())
        assert msg["seq"] == 7 and msg["data"]["params"]["stem"] == 120
        ws.send_text(json.dumps({"type": "bogus", "seq": 8}))
        assert json.loads(ws.receive_text())["type"] == "problem"
        assert client.get("/").status_code == 200
        assert client.get("/font.otf").content[:4] == b"OTTO"
