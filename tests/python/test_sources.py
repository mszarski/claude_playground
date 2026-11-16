"""File replay: our container, and .wav recordings with a sidecar."""

import numpy as np
import pytest
import yaml
from scipy.io import wavfile

import gpusonar as gs
from gpusonar.sources import ContainerSource, SidecarError, WavSource, make_source
from conftest import make_record

RATE = 96000


def _sidecar(tmp_path, **overrides):
    cfg = {
        "array_id": "popoto_8",
        "mode": "flank",
        "file": "rec.wav",
        "elements": {"count": 8, "pitch": 0.027, "axis": [1, 0, 0]},
        "transmit": {
            "waveform": {"type": "lfm", "f_start": 20000, "f_stop": 35000, "duration": 0.01},
            "lever_arm": [0.0, 0.0, 0.5],
        },
        "pings": {"times": [0.10, 0.35, 0.60], "window": [0.0, 0.2]},
        "recording_start_ns": 1_700_000_000_000_000_000,
        "environment": {"sound_speed": 1480.0},
    }
    cfg.update(overrides)
    path = tmp_path / "rec.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return str(path)


def _ramp(channels=8, seconds=1.0):
    """Each channel holds its own sample index pattern so misalignment is visible."""
    n = int(RATE * seconds)
    t = np.arange(n)
    return np.stack([((t + 1000 * c) % 20000) - 10000 for c in range(channels)], axis=1).astype(np.int16)


def test_multichannel_wav_is_cut_at_ping_times(tmp_path):
    audio = _ramp()
    wavfile.write(tmp_path / "rec.wav", RATE, audio)
    src = WavSource(_sidecar(tmp_path))
    assert len(src) == 3
    for i, t in enumerate([0.10, 0.35, 0.60]):
        rec = src.read(i)
        assert gs.validate(rec).ok, str(gs.validate(rec))
        i0 = int(round(t * RATE))
        expected = audio[i0 : i0 + int(0.2 * RATE)].T.astype(np.float32) / 32768.0
        np.testing.assert_array_equal(rec.samples.array, expected)
        assert rec.samples.sample_rate == RATE
        assert rec.transmit.time_ns == 1_700_000_000_000_000_000 + int(round(t * 1e9))
        assert rec.metadata.sequence == i
        assert rec.metadata.adc_bits == 16
    rec = src.read(0)
    x = np.array([e.lever_arm[0] for e in rec.geometry.rx])
    np.testing.assert_allclose(np.diff(x), 0.027)
    assert abs(x.mean()) < 1e-12
    assert rec.transmit.waveform.type == gs.WaveformType.Lfm
    assert rec.samples.center_frequency == 27500.0
    assert rec.environment.sound_speed == 1480.0


def test_per_channel_files_match_multichannel(tmp_path):
    audio = _ramp()
    wavfile.write(tmp_path / "rec.wav", RATE, audio)
    names = []
    for c in range(8):
        name = f"ch{c}.wav"
        wavfile.write(tmp_path / name, RATE, audio[:, c])
        names.append(name)
    multi = WavSource(_sidecar(tmp_path)).read(1).samples.array
    cfg_path = _sidecar(tmp_path, files=names)
    cfg = yaml.safe_load(open(cfg_path))
    del cfg["file"]
    open(cfg_path, "w").write(yaml.safe_dump(cfg))
    np.testing.assert_array_equal(WavSource(cfg_path).read(1).samples.array, multi)


def test_start_time_reports_sample_snapping(tmp_path):
    wavfile.write(tmp_path / "rec.wav", RATE, _ramp())
    t = 0.1 + 0.4 / RATE  # ping time that falls between samples
    src = WavSource(_sidecar(tmp_path, pings={"times": [t], "window": [0.001, 0.05]}))
    rec = src.read(0)
    i0 = int(round((t + 0.001) * RATE))
    assert rec.samples.start_time == pytest.approx(i0 / RATE - t, abs=1e-12)
    assert abs(rec.samples.start_time - 0.001) <= 0.5 / RATE


def test_pings_running_off_the_recording_are_dropped_and_counted(tmp_path):
    wavfile.write(tmp_path / "rec.wav", RATE, _ramp(seconds=0.5))
    # Windows are 0.2 s long: the pings at 0.35 s and 0.60 s end after the 0.5 s recording.
    src = WavSource(_sidecar(tmp_path))
    assert len(src) == 1
    assert src.dropped_pings == 2


def test_24_bit_and_float_wav(tmp_path):
    rng = np.random.default_rng(0)
    x24 = rng.integers(-(2**23), 2**23, size=(RATE // 2, 8), dtype=np.int64)
    # scipy writes int32; 24-bit recordings arrive left-justified in int32.
    wavfile.write(tmp_path / "rec.wav", RATE, (x24 << 8).astype(np.int32))
    rec = WavSource(_sidecar(tmp_path, pings={"times": [0.0], "window": [0.0, 0.1]})).read(0)
    expected = (x24[: int(0.1 * RATE)].T / 2**23).astype(np.float32)
    np.testing.assert_allclose(rec.samples.array, expected, atol=1e-7)

    xf = rng.standard_normal((RATE // 2, 8)).astype(np.float32) * 0.1
    wavfile.write(tmp_path / "rec.wav", RATE, xf)
    rec = WavSource(_sidecar(tmp_path, pings={"times": [0.0], "window": [0.0, 0.1]})).read(0)
    np.testing.assert_array_equal(rec.samples.array, xf[: int(0.1 * RATE)].T)


def test_multi_board_sidecar(tmp_path):
    wavfile.write(tmp_path / "rec.wav", RATE, _ramp())
    boards = [
        {"channels": [0, 1, 2, 3], "reference_channel": 3},
        {"channels": [4, 5, 6, 7], "reference_channel": 7},
    ]
    rec = WavSource(_sidecar(tmp_path, boards=boards)).read(0)
    assert rec.metadata.channel_board == [0, 0, 0, 0, 1, 1, 1, 1]
    assert [b.reference_channel for b in rec.metadata.boards] == [3, 7]
    assert gs.validate(rec).ok


@pytest.mark.parametrize(
    "change, message",
    [
        ({"elements": {"count": 4, "pitch": 0.027}}, "4 elements"),
        ({"sample_rate": 48000}, "disagrees with the audio"),
        ({"elements": None}, "elements"),
    ],
)
def test_sidecar_contradictions_are_errors(tmp_path, change, message):
    wavfile.write(tmp_path / "rec.wav", RATE, _ramp())
    path = _sidecar(tmp_path, **change)
    if change.get("elements", 1) is None:
        cfg = yaml.safe_load(open(path))
        del cfg["elements"]
        open(path, "w").write(yaml.safe_dump(cfg))
    with pytest.raises(SidecarError, match=message):
        WavSource(path)


def test_unequal_per_channel_lengths_are_rejected(tmp_path):
    audio = _ramp(channels=2)
    wavfile.write(tmp_path / "a.wav", RATE, audio[:, 0])
    wavfile.write(tmp_path / "b.wav", RATE, audio[:-10, 1])
    path = _sidecar(tmp_path, files=["a.wav", "b.wav"], elements={"count": 2, "pitch": 0.027})
    cfg = yaml.safe_load(open(path))
    del cfg["file"]
    open(path, "w").write(yaml.safe_dump(cfg))
    with pytest.raises(SidecarError, match="disagree on length"):
        WavSource(path)


def test_container_source_and_factory(tmp_path):
    path = str(tmp_path / "c.h5")
    with gs.RecordWriter(path) as w:
        for i in range(5):
            rec = make_record(seed=i)
            meta = rec.metadata
            meta.sequence = i
            rec.metadata = meta
            w.write(rec)
    src = make_source({"type": "container", "path": path, "start": 1, "stop": 4})
    assert isinstance(src, ContainerSource)
    assert [r.metadata.sequence for r in src] == [1, 2, 3]
    with pytest.raises(ValueError, match="unknown source type"):
        make_source({"type": "nope"})
