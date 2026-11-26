import os
import warnings

import numpy as np
import pytest

import gpusonar as gs
from gpusonar.sim import engine as sim_engine
from oracle_gate import pytest_runtest_makereport, pytest_terminal_summary  # noqa: F401  (hooks: oracle skips, GPUSONAR_REQUIRE_ORACLES)


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item):
    """A test that asks for the C++ simulator engine (engine="cpp", GPUSONAR_SIM_ENGINE=cpp) on a module built
    without it skips, naming why (a-engine: before, it ran NumPy and passed by comparing NumPy with itself).
    With the C++ simulator built, `sim/engine.py` never raises EngineUnavailable and this changes nothing."""
    try:
        return (yield)
    except sim_engine.EngineUnavailable as e:
        pytest.skip(f"needs the C++ simulator engine, which this extension module lacks: {e}")


def pytest_sessionstart(session):
    """Without the C++ simulator the default engine runs NumPy with one warning per process (sim/engine.py). It is
    given once here, in the run's header (`pytest_report_header`), and not inside whichever test first runs the
    simulator, where a test that turns warnings into errors would fail on it in an order-dependent way."""
    if not sim_engine.HAVE_CPP and not os.environ.get("GPUSONAR_SIM_ENGINE"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sim_engine.resolve()


def pytest_report_header(config):
    if not sim_engine.HAVE_CPP:
        return ("simulator engine: NumPy only (the extension module lacks the C++ simulator); tests that ask for "
                "the C++ engine skip")
    return None


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
        e.boresight = (0.0, 1.0, 0.0)          # no default: a record states where its elements look
        elements.append(e)
    geo.rx = elements
    tx = geo.tx
    tx.boresight = (0.0, 1.0, 0.0)
    geo.tx = tx
    r.geometry = geo
    env = r.environment
    env.sound_speed = 1500.0                   # no default either
    r.environment = env
    r.set_default_channel_tables()
    meta = r.metadata
    meta.array_id = "test"
    meta.mode = gs.Mode.Flank
    r.metadata = meta
    return r


@pytest.fixture
def record():
    return make_record()
