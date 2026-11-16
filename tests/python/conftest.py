import numpy as np
import pytest

import gpusonar as gs


def make_record(channels=4, samples=128, complex_data=True, seed=0):
    """A valid record with every table sized to the samples."""
    rng = np.random.default_rng(seed)
    r = gs.PingRecord()
    data = rng.standard_normal((channels, samples)).astype(np.float32)
    if complex_data:
        data = (data + 1j * rng.standard_normal((channels, samples))).astype(np.complex64)
        r.samples.kind = gs.SampleKind.ComplexBaseband
    r.samples.array = data
    r.samples.sample_rate = 64000.0
    r.samples.center_frequency = 8000.0
    w = r.transmit.waveform
    w.type = gs.WaveformType.Lfm
    w.f_start, w.f_stop, w.duration = 6000.0, 10000.0, 0.1
    r.transmit.waveform = w
    r.transmit.time_ns = 1_700_000_000_000_000_000
    geo = r.geometry
    elements = []
    for i in range(channels):
        e = gs.core.Element()
        e.lever_arm = (0.094 * i, 0.0, 0.0)
        elements.append(e)
    geo.rx = elements
    r.geometry = geo
    r.set_default_channel_tables()
    meta = r.metadata
    meta.array_id = "test"
    meta.mode = gs.Mode.Flank
    r.metadata = meta
    return r


@pytest.fixture
def record():
    return make_record()
