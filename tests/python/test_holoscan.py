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


def test_flank_pipeline_from_an_emulated_capture_board():
    """The low-rate source end to end: an emulated 32-channel, 24-bit, 64 kSPS board sends
    blocks over UDP; the source assembles pings; the Holoscan pipeline images and detects."""
    from gpusonar.sources import FlankBoardEmulator, make_source
    overrides = {"receive": {"window": {"range": [1850.0, 2150.0]}}, "transmit": {"waveform": {"duration": 0.1}}}
    sonar = SonarModel.load("flank_32", overrides)
    b = np.radians(12.0)
    scene = Scene().add_points([2040.0 * np.array([np.sin(b), np.cos(b), 0.0])], target_strength_db=10.0)
    pings = [simulate_ping(sonar, make_pose(), scene, environment={"sound_speed": 1500.0}, noise_rms=1e-6,
                           absorption_model="none", sequence=i, rng=np.random.default_rng(i))
             for i in range(3)]
    source = make_source({"type": "flank_udp", "sonar": "flank_32", "sonar_overrides": overrides,
                          "max_pings": 3, "timeout": 20.0})
    try:
        sender = FlankBoardEmulator(pings, ping_interval=3.2, block_samples=128).send(source.port, pace=0.25)
        ranges = np.linspace(1900.0, 2150.0, 1500)
        bearings = np.radians(np.arange(-30.0, 30.01, 1.0))
        app = hops.FanPipeline(source, n_pings=3, ranges=ranges, bearings_rad=bearings, detect_targets=True)
        app.run()
        sender.join(timeout=60.0)
    finally:
        source.close()
    if source.assembler.lost_samples:
        pytest.skip(f"the loopback dropped {source.assembler.lost_samples} samples")
    assert len(app.sink.items) == 3 and source.assembler.dropped_pings == 0
    for item in app.sink.items:
        best = max(item["detections"], key=lambda d: d.snr_db)
        assert best.range == pytest.approx(2040.0, abs=1.5)
        assert np.degrees(best.bearing) == pytest.approx(12.0, abs=1.0)


def test_condition_op_output_is_the_chains_prepare_and_gain():
    """STATUS open item 33: ConditionOp, run in the Holoscan runtime, gives the samples of a chain
    run (pipeline.py: prepare, calibrate, tvg) on the same records, for no gain, the chain's
    default gain step (absorption from the record, D-91), the law before D-91 by name, and the
    operator's older `tvg_db_per_decade` argument (the old law, as D-91 records it)."""
    from holoscan.conditions import CountCondition
    from gpusonar import pipeline
    from gpusonar.config import config_dir, load_yaml
    cfg = load_yaml(config_dir() / "chains" / "stripmap_points.yaml")
    cfg["source"]["overrides"] = {"pings": {"count": 2}}
    cfg["output"] = {}
    host = gs.core.Backend.Host
    raw = pipeline.run(dict(cfg, steps=[]), backend=host).records

    class ConditionApp(hops.Application):
        def __init__(self, records, **arguments):
            self.records, self.arguments = records, arguments
            super().__init__()

        def compose(self):
            src = hops.PingSourceOp(self, CountCondition(self, len(self.records)), source=self.records, name="source")
            self.cond = hops.ConditionOp(self, name="condition", **self.arguments)
            self.sink = hops.CollectOp(self, port="ping", name="collect")
            self.add_flow(src, self.cond, {("ping", "ping")})
            self.add_flow(self.cond, self.sink, {("ping", "ping")})

    cases = [({}, []), ({"tvg": True}, [{}]), ({"tvg": {"absorption": "none"}}, [{"absorption": "none"}]),
             ({"tvg_db_per_decade": 40.0}, [{"spreading_db_per_decade": 40.0, "absorption": "none"}])]
    for arguments, tvg in cases:
        app = ConditionApp([r.copy() for r in raw], **arguments)
        app.run()
        got = [m["record"].samples.array for m in app.sink.items]
        steps = [{"stage": "prepare"}, {"stage": "calibrate"}] + [dict(s, stage="tvg") for s in tvg]
        want = [r.samples.array for r in pipeline.run(dict(cfg, steps=steps), backend=host).records]
        assert len(got) == len(want) == 2, arguments
        for a, b in zip(got, want):
            np.testing.assert_array_equal(a, b, err_msg=str(arguments))
        assert len(app.cond.tvg_reports) == (2 if tvg else 0)
