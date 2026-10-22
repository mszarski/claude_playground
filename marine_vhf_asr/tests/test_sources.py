"""Tests for data-source helpers that do not need network access."""

import io
import json

import numpy as np
import pytest
from scipy.io import wavfile

pytest.importorskip("soundfile")
pytest.importorskip("datasets")

from marine_vhf_asr.sources import _AMBIGUOUS_NUMBER, load_audio, load_source


def test_ambiguous_numbers_filtered():
    for text in ["ETA 1430 UTC", "01-15.0N 104-00.0E", "we see MHM6003", "49°30'N", "speed 12.5"]:
        assert _AMBIGUOUS_NUMBER.search(text), text
    for text in ["slow to 8 knots", "channel 16, over", "buoy 4", "MERLION 31"]:
        assert not _AMBIGUOUS_NUMBER.search(text), text


def test_load_audio_resamples_and_downmixes(tmp_path):
    stereo = np.stack([np.sin(np.linspace(0, 100, 24000))] * 2, axis=1).astype(np.float32)
    path = tmp_path / "x.wav"
    wavfile.write(path, 24000, stereo)
    x = load_audio({"bytes": None, "path": str(path)})
    assert x.ndim == 1 and len(x) == 16000

    buf = io.BytesIO()
    wavfile.write(buf, 16000, stereo[:, 0])
    assert len(load_audio({"bytes": buf.getvalue(), "path": None})) == 24000


def test_local_manifest(tmp_path):
    wavfile.write(tmp_path / "a.wav", 16000, np.zeros(1600, dtype=np.float32))
    with open(tmp_path / "m.jsonl", "w") as f:
        f.write(json.dumps({"audio": "a.wav", "text": "CHANNEL ONE SIX", "slots": {"channel": "ONE SIX"}}) + "\n")
    ds = load_source(str(tmp_path / "m.jsonl"))
    assert ds.column_names == ["audio", "text", "slots", "source"]
    assert json.loads(ds[0]["slots"]) == {"channel": "ONE SIX"}
    assert len(load_audio(ds[0]["audio"])) == 1600
