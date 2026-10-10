import io

import numpy as np
import pytest

from rmr.respond import respond, user_message
from rmr.voice import CANON, EmotionModel, load_audio


def test_load_audio_resamples_to_16k_mono():
    sf = pytest.importorskip("soundfile")
    t = np.arange(44100) / 44100
    stereo = np.stack([np.sin(2 * np.pi * 440 * t)] * 2, 1).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, stereo, 44100, format="WAV")
    x = load_audio(buf.getvalue())
    assert x.dtype == np.float32 and x.ndim == 1 and abs(len(x) - 16000) <= 1


def test_canon_folds_labels_into_the_shared_four():
    em = EmotionModel.__new__(EmotionModel)
    q = em.canon({"calm": 0.2, "joy": 0.3, "sadness": 0.1, "anger": 0.1, "fear": 0.3})
    assert list(q) == CANON and abs(sum(q.values()) - 1) < 1e-9 and q["happy"] > q["sad"]


def test_respond_builds_a_planner_prompt_and_trusts_words_in_the_prompt():
    seen = {}

    def chat(msgs, schema, name, model=None, temperature=None):
        seen["msgs"] = msgs
        return {"feeling": "sad", "reading": "They sound disappointed.",
                "response": "You lean in slowly, ears softly lowered."}

    heard = {"text": "I didn't get the job", "emotion": "sad", "confidence": 0.61, "arousal": 0.3, "valence": 0.2}
    r = respond(heard, chat=chat)
    assert r["response"].startswith("responding. You lean in")          # coerced into "word. sentence."
    assert r["feeling"] == "sad"
    assert "No reading of their voice" in user_message(heard, tone=False) and "61%" not in user_message(heard, tone=False)
    assert "trust the words" in seen["msgs"][0]["content"]
    assert "I didn't get the job" in user_message(heard) and "61%" in user_message(heard)


def test_respond_endpoint(monkeypatch):
    pytest.importorskip("fastapi")
    import os
    import tempfile

    import torch
    from fastapi.testclient import TestClient

    import rmr.respond
    from rmr import server
    from rmr.generator.data import fit_stats, samples_from_moves
    from rmr.generator.model import MotionGenerator
    from test_generator import TINY, _moves
    from test_server import _plan

    class FakeListener:
        def hear(self, audio):
            return {"text": "hello robot", "emotion": "happy", "confidence": 0.7, "probs": {}, "arousal": 0.6,
                    "valence": 0.8}

    monkeypatch.setattr(rmr.respond, "respond", lambda heard, model=None: {"feeling": "happy", "reading": "cheerful",
                                                                           "response": "greeting. You perk up."})
    d = tempfile.mkdtemp()
    net = MotionGenerator(**TINY)
    torch.save({"sd": net.state_dict(), "stats": fit_stats(samples_from_moves(_moves())), "config": net.config},
               os.path.join(d, "t.pt"))
    eng = server.Engine(os.path.join(d, "t.pt"))
    eng.plan, eng.listener = _plan, FakeListener()
    c = TestClient(server.create_app(eng))
    r = c.post("/api/respond?n=1", content=b"\0" * 5000, headers={"content-type": "audio/wav"})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["heard"]["text"] == "hello robot" and d["reading"] == "cheerful" and d["feeling"] == "happy" and d["prompt"] == "greeting. You perk up."
    assert len(d["moves"]) == 1 and "listen" in d["timing_ms"]
    assert c.post("/api/respond", content=b"\0" * 10).status_code == 422


def test_calibration_moves_a_loud_device_to_the_reference_level():
    from rmr.voice import REFERENCE, Calibration

    cal, loud = Calibration(prior_n=5), {"arousal": 0.65, "dominance": 0.65, "valence": 0.6, "text": "hi"}
    outs = [cal(dict(loud)) for _ in range(200)]
    assert outs[0]["valence"] > outs[-1]["valence"]                    # the prior holds the first readings back
    for k in REFERENCE:
        assert abs(outs[-1][k] - REFERENCE[k]) < 0.01 and outs[-1]["raw"][k] == loud[k]
    hi = cal({"arousal": 0.65, "dominance": 0.65, "valence": 0.8})   # a reading above the device's level stays above
    assert hi["valence"] > REFERENCE["valence"] + 0.15
    assert cal({"text": "no attributes"}) == {"text": "no attributes"}
    fixed = Calibration(fixed={"arousal": 0.6, "dominance": 0.6, "valence": 0.6})
    assert abs(fixed({"arousal": 0.6, "dominance": 0.6, "valence": 0.7})["valence"] - (REFERENCE["valence"] + 0.1)) < 1e-9
