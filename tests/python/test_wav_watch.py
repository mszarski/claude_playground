"""sources/wav_watch.py: recordings that appear in a directory while the source watches it (audit B9).

A recording that arrives late (its sidecar first, its audio after) is taken once, when its audio is
there and has settled; a recording that cannot be read goes to `failed` with the reason and does
not stop the others."""

import threading
import time

import numpy as np
import yaml
from scipy.io import wavfile

from gpusonar.sources import WavWatchSource, make_source

RATE = 96000
PING_TIMES = [0.05, 0.25]


def _sidecar(directory, name, wav, **overrides):
    cfg = {
        "array_id": "popoto_8", "mode": "flank", "file": wav,
        "elements": {"count": 8, "pitch": 0.027, "axis": [1, 0, 0]},
        "transmit": {"waveform": {"type": "lfm", "f_start": 20000, "f_stop": 35000, "duration": 0.01},
                     "lever_arm": [0.0, 0.0, 0.5]},
        "pings": {"times": PING_TIMES, "window": [0.0, 0.15]},
        "recording_start_ns": 1_700_000_000_000_000_000,
        "environment": {"sound_speed": 1480.0},
    }
    cfg.update(overrides)
    (directory / name).write_text(yaml.safe_dump(cfg))


def _audio(seed, channels=8):
    return (np.random.default_rng(seed).standard_normal((RATE // 2, channels)) * 300).astype(np.int16)


def test_the_factory_builds_it(tmp_path):
    src = make_source({"type": "wav_watch", "directory": str(tmp_path), "timeout": 0.0})
    assert isinstance(src, WavWatchSource)
    assert list(src) == [] and src.done == [] and src.failed == {}


def test_a_late_recording_is_picked_up_once(tmp_path):
    wavfile.write(tmp_path / "rec0.wav", RATE, _audio(0))
    _sidecar(tmp_path, "rec0.yaml", "rec0.wav")
    src = WavWatchSource(tmp_path, poll=0.05, settle=0.2, timeout=2.0)
    late = _audio(1)

    def arrive():
        # The sidecar first; the audio only later, as a copy job might do it.
        time.sleep(0.4)
        _sidecar(tmp_path, "rec1.yaml", "rec1.wav", recording_start_ns=1_700_000_001_000_000_000)
        time.sleep(0.4)
        wavfile.write(tmp_path / "rec1.wav", RATE, late)

    writer = threading.Thread(target=arrive)
    writer.start()
    try:
        seen = []
        for rec in src:
            seen.append((list(src.done), rec))
    finally:
        writer.join()
    # Two pings from each recording, each recording once, oldest first; the run then ends on the timeout.
    assert src.done == ["rec0.yaml", "rec1.yaml"] and src.failed == {}
    assert len(seen) == 2 * len(PING_TIMES)
    assert [done for done, _ in seen] == [["rec0.yaml"]] * 2 + [["rec0.yaml", "rec1.yaml"]] * 2
    records = [r for _, r in seen]
    assert records[2].transmit.time_ns - records[0].transmit.time_ns == 10**9
    # The late recording's samples are its own audio (other noise), not the first one's again.
    assert not np.array_equal(records[0].samples.array, records[2].samples.array)
    assert records[2].samples.n_channels == 8 and records[2].samples.array.shape[1] > 0


def test_a_bad_recording_goes_to_failed_and_the_rest_are_read(tmp_path):
    wavfile.write(tmp_path / "bad.wav", RATE, _audio(2, channels=4))       # 4 channels, the sidecar says 8
    _sidecar(tmp_path, "bad.yaml", "bad.wav")
    time.sleep(0.05)                                                       # bad.yaml is the older of the two
    wavfile.write(tmp_path / "good.wav", RATE, _audio(3))
    _sidecar(tmp_path, "good.yaml", "good.wav")
    src = WavWatchSource(tmp_path, poll=0.05, settle=0.0, timeout=0.5)
    records = list(src)
    assert src.done == ["good.yaml"] and len(records) == len(PING_TIMES)
    assert list(src.failed) == ["bad.yaml"] and "4" in src.failed["bad.yaml"] and "8" in src.failed["bad.yaml"]
    # A failed recording is not tried again on the next poll: a second pass over the same source adds nothing.
    assert list(src) == [] and list(src.failed) == ["bad.yaml"] and src.done == ["good.yaml"]
