"""Tests for the VHF channel simulator and synthesis pipeline."""

import json

import numpy as np
from scipy import signal

from marine_vhf_asr import synthesize
from marine_vhf_asr.vhf_channel import ChannelConfig, simulate_vhf

SR = 16000


def _speechlike(seconds=1.0):
    t = np.arange(int(SR * seconds)) / SR
    return np.sin(2 * np.pi * 200 * t) + 0.5 * np.sin(2 * np.pi * 6000 * t)


def test_output_shape_and_level():
    x = _speechlike()
    y, params = simulate_vhf(x, SR, np.random.default_rng(0))
    assert y.dtype == np.float32
    assert len(y) > len(x)  # lead silence + squelch padding
    assert np.isclose(np.max(np.abs(y)), 0.9)
    assert {"snr_db", "clip_drive", "faded", "narrowband"} <= params.keys()


def test_reproducible():
    x = _speechlike()
    y1, _ = simulate_vhf(x, SR, np.random.default_rng(42))
    y2, _ = simulate_vhf(x, SR, np.random.default_rng(42))
    np.testing.assert_array_equal(y1, y2)


def test_removes_out_of_band_energy():
    x = _speechlike()
    cfg = ChannelConfig(snr_db=(60, 60), background_db=(-80, -80), narrowband_prob=0, fade_prob=0)
    y, _ = simulate_vhf(x, SR, np.random.default_rng(0), cfg)
    f, p = signal.welch(y, SR, nperseg=2048)
    in_band = p[(f > 150) & (f < 250)].max()
    out_band = p[(f > 5900) & (f < 6100)].max()
    assert out_band < in_band * 1e-3


def test_synthesize_end_to_end(tmp_path):
    synthesize.main(["--out", str(tmp_path), "--dialogues", "3", "--tts", "sine", "--seed", "1"])
    rows = [json.loads(line) for line in open(tmp_path / "manifest.jsonl")]
    assert len(rows) >= 6
    for r in rows:
        assert (tmp_path / r["audio"]).exists()
        assert r["text"] == r["text"].upper()


def test_synthesize_text_only(tmp_path):
    synthesize.main(["--out", str(tmp_path), "--dialogues", "2", "--text-only"])
    rows = [json.loads(line) for line in open(tmp_path / "manifest.jsonl")]
    assert rows and "audio" not in rows[0]
