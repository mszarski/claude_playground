"""Milestone 11: stages as Holoscan operators, pipelines fed by the simulator and by
file replay. Skipped where the Holoscan SDK is not installed."""

import time

import numpy as np
import pytest

try:        # the SDK needs a recent CUDA runtime even to import, so any failure means "not here"
    import holoscan  # noqa: F401
except Exception as exc:  # noqa: BLE001
    pytest.skip(f"Holoscan SDK not usable here: {type(exc).__name__}", allow_module_level=True)

import gpusonar as gs  # noqa: E402
from gpusonar import holoscan_ops as hops  # noqa: E402
from gpusonar.sim import Scene, make_pose, simulate_ping  # noqa: E402
from gpusonar.sonar import SonarModel  # noqa: E402
from gpusonar.sources import ContainerSource  # noqa: E402


def fls_pings(n_images=3):
    sonar = SonarModel.load("l12_5d", {"receive": {"window": {"range": [1.4, 1.6]}}})
    scene = Scene().add_points([[1.5, 0.05, 0.0]])
    out = []
    for i in range(2 * n_images):
        out.append(simulate_ping(sonar, make_pose(), scene, environment={"sound_speed": 1480.0},
                                 absorption_model="none", sub_sequence=i, sequence=i,
                                 time_ns=1_700_000_000_000_000_000 + i * 20_000_000))
    return out


def test_fls_pipeline_images_the_target_from_simulated_pings():
    pings = fls_pings()
    ranges = np.linspace(1.45, 1.55, 201)
    bearings = np.radians(np.linspace(-5.0, 8.0, 261))
    app = hops.FanPipeline(pings, n_pings=len(pings), ranges=ranges, bearings_rad=bearings, pings_per_image=2)
    app.run()
    assert len(app.sink.items) == 3                      # two mux pings per image
    for item in app.sink.items:
        ir, ib = np.unravel_index(np.argmax(np.abs(item["image"])), item["image"].shape)
        assert ranges[ir] == pytest.approx(np.hypot(1.5, 0.05), abs=2e-3)
        assert np.degrees(bearings[ib]) == pytest.approx(np.degrees(np.arctan2(0.05, 1.5)), abs=0.2)
        assert float(np.median(item["display"])) == pytest.approx(0.2, abs=0.01)   # DRC stage ran


def test_flank_pipeline_from_file_replay_detects_the_target(tmp_path):
    sonar = SonarModel.load("flank_32", {"receive": {"window": {"range": [1850.0, 2150.0]}},
                                          "transmit": {"waveform": {"duration": 0.1}}})
    b = np.radians(12.0)
    scene = Scene().add_points([2040.0 * np.array([np.sin(b), np.cos(b), 0.0])], target_strength_db=10.0)
    path = str(tmp_path / "flank.h5")
    with gs.RecordWriter(path) as w:
        for i in range(4):
            w.write(simulate_ping(sonar, make_pose(), scene, environment={"sound_speed": 1500.0},
                                  noise_rms=2e-7, adc=False, absorption_model="none", sequence=i,
                                  rng=np.random.default_rng(i),
                                  time_ns=1_700_000_000_000_000_000 + i * 50_000_000))
    ranges = np.linspace(1900.0, 2150.0, 1500)
    bearings = np.radians(np.arange(-30.0, 30.01, 1.0))
    app = hops.FanPipeline(ContainerSource(path), n_pings=4, ranges=ranges, bearings_rad=bearings,
                           detect_targets=True)
    app.run()
    assert len(app.sink.items) == 4
    for item in app.sink.items:
        best = max(item["detections"], key=lambda d: d.snr_db)
        assert best.range == pytest.approx(2040.0, abs=1.5)
        assert np.degrees(best.bearing) == pytest.approx(12.0, abs=1.0)


def test_pipeline_keeps_up_with_a_real_time_replay():
    """Pings released at their recorded 20 ms spacing: the pipeline must finish each image
    before the next pair arrives, so latency stays bounded instead of growing."""
    pings = fls_pings(n_images=6)
    ranges = np.linspace(1.45, 1.55, 201)
    bearings = np.radians(np.linspace(-5.0, 8.0, 131))
    app = hops.FanPipeline(pings, n_pings=len(pings), ranges=ranges, bearings_rad=bearings,
                           pings_per_image=2, realtime=True)
    t = time.perf_counter()
    app.run()
    elapsed = time.perf_counter() - t
    assert len(app.sink.items) == 6
    span = (pings[-1].transmit.time_ns - pings[0].transmit.time_ns) * 1e-9      # 0.22 s of data
    latency = np.array(app.sink.latency)
    # Not falling behind: the last image is no later than the first by more than a ping period.
    assert latency[-1] < latency[0] + 0.04, latency
    assert elapsed < span + 2.0
