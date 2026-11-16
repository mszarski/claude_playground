"""Scenario runner: turns a scenario config into ping records plus ground truth."""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from .. import _gpusonar as core
from .. import acoustics
from ..config import dump_yaml, load_scenario_config
from ..sonar import SonarModel, resolve_environment
from . import receiver
from .propagation import EigenrayProvider, make_provider, two_way_paths
from .scene import Scene, build_scene


def make_pose(position=(0.0, 0.0, 0.0), rpy=(0.0, 0.0, 0.0), velocity=(0.0, 0.0, 0.0),
              rates=(0.0, 0.0, 0.0)) -> "core.Pose":
    pose = core.Pose()
    pose.position = tuple(float(x) for x in position)
    pose.rpy = tuple(float(x) for x in rpy)
    pose.velocity = tuple(float(x) for x in velocity)
    pose.rates = tuple(float(x) for x in rates)
    return pose


def absorption_table(sonar: SonarModel, env: dict, model: str):
    """(f_min, f_max, table) covering the transmit band, for the synthesis kernel."""
    w = sonar.waveform_config
    f0, f1 = float(w["f_start"]), float(w.get("f_stop", w["f_start"]))
    lo, hi = min(f0, f1), max(f0, f1)
    if hi <= lo:
        hi = lo * 1.0001
    f = np.linspace(lo, hi, core.ABSORPTION_TABLE_SIZE)
    table = acoustics.absorption(
        f, env.get("temperature", 10.0), env.get("salinity", 35.0), env.get("depth", 0.0),
        env.get("ph", 8.0), model)
    return lo, hi, [float(x) for x in np.broadcast_to(table, f.shape)]


def simulate_ping(sonar: SonarModel, pose: "core.Pose", scene: Scene, *,
                  environment: dict | None = None, provider: EigenrayProvider | None = None,
                  sequence: int = 0, sub_sequence: int = 0, time_ns: int = 0,
                  scenario_time: float = 0.0, absorption_model: str = "francois_garrison",
                  noise_rms: float = 0.0, adc: bool = True, spreading: bool = True,
                  rng: np.random.Generator | None = None, tx_pose: "core.Pose | None" = None,
                  acceleration=(0.0, 0.0, 0.0), ins_rate: float = 0.0,
                  backend=core.Backend.Host) -> "core.PingRecord":
    """Simulate one transmit and return its record."""
    env = resolve_environment(environment, depth=pose.position[2])
    record = sonar.make_record(pose, sequence=sequence, sub_sequence=sub_sequence,
                               time_ns=time_ns, environment=env, tx_pose=tx_pose)
    c = record.environment.sound_speed
    provider = provider or make_provider("free_field", c)
    # The true board clock errors drive synthesis, then are cleared: the record a
    # processor receives does not know them (the sync stage estimates them).
    _set_board_clocks(record, sonar, truth=True)
    if len(scene) > 0:
        tx_position = np.array(record.tx_position())
        rx_reference = record.rx_positions().mean(axis=0)
        q, p, amp, vq, vp = two_way_paths(provider, tx_position, rx_reference,
                                          scene.at_time(scenario_time), scene.amplitudes,
                                          scene.velocities)
        core.synthesize_into(record, q, p, amp, vq, vp,
                             absorption=absorption_table(sonar, env, absorption_model),
                             spreading=spreading, backend=backend,
                             platform_acceleration=tuple(float(a) for a in acceleration))
    _set_board_clocks(record, sonar, truth=False)
    if ins_rate > 0.0:
        _add_true_ins(record, np.asarray(acceleration, dtype=float), ins_rate)
    samples = record.samples.array
    if noise_rms > 0.0:
        samples = receiver.add_noise(samples, noise_rms, rng or np.random.default_rng())
    samples = receiver.apply_hardware_tvg(record, samples)
    if adc:
        samples = receiver.apply_adc(record, samples)
    s = record.samples
    kind = s.kind
    s.array = samples
    s.kind = kind
    record.samples = s
    return record


def _add_true_ins(record, accel: np.ndarray, rate: float) -> None:
    """Attach error-free INS samples covering the receive window (true trajectory)."""
    s = record.samples
    t_end = s.start_time + s.n_samples / s.sample_rate
    pose = record.geometry.rx_pose
    p0, v0 = np.array(pose.position), np.array(pose.velocity)
    out = []
    for t in np.arange(0.0, t_end + 1.5 / rate, 1.0 / rate):
        a = core.InsSample()
        a.t = float(t)
        a.position = tuple(p0 + v0 * t + 0.5 * accel * t * t)
        a.rpy = tuple(np.array(pose.rpy) + np.array(pose.rates) * t)
        a.velocity = tuple(v0 + accel * t)
        a.rates = pose.rates
        out.append(a)
    nav = record.navigation
    nav.ins = out
    record.navigation = nav


def _set_board_clocks(record, sonar: SonarModel, truth: bool) -> None:
    meta = record.metadata
    boards = meta.boards
    for board, cfg in zip(boards, sonar.boards):
        board.clock_offset = float(cfg.get("clock_offset", 0.0)) if truth else 0.0
        board.clock_drift_ppm = float(cfg.get("clock_drift_ppm", 0.0)) if truth else 0.0
    meta.boards = boards
    record.metadata = meta


@dataclass
class ScenarioResult:
    records: list
    scene: Scene
    config: dict


def run_scenario(name_or_config, overrides: dict | None = None, output: str | None = None,
                 backend=core.Backend.Host) -> ScenarioResult:
    """Run a scenario (a name under configs/scenarios, a path, or a config dict).

    With `output`, records and ground truth are written to that container file.
    """
    if isinstance(name_or_config, dict):
        cfg = name_or_config
    else:
        cfg = load_scenario_config(name_or_config, overrides)
    sonar = SonarModel.load(cfg["sonar"], cfg.get("sonar_overrides"))
    env = resolve_environment(cfg.get("environment"))
    tier = cfg.get("fidelity", "free_field")
    provider = make_provider(tier, env["sound_speed"], cfg.get("propagation"))
    scene = build_scene(cfg.get("scene", []))
    rng = np.random.default_rng(int(cfg.get("seed", 0)))

    plat = cfg.get("platform", {})
    position = np.asarray(plat.get("position", [0.0, 0.0, 0.0]), dtype=float)
    velocity = np.asarray(plat.get("velocity", [0.0, 0.0, 0.0]), dtype=float)
    rpy = plat.get("rpy", [0.0, 0.0, 0.0])
    pings = cfg.get("pings", {})
    count = int(pings.get("count", 1))
    interval = float(pings.get("interval", 1.0))
    start_ns = int(pings.get("start_ns", 1_700_000_000_000_000_000))
    rx_cfg = cfg.get("receiver", {})

    records, truth_positions = [], []
    n_sub = len(sonar.mux_groups)
    # A rail moves between pings and stands still during each one.
    stop_and_go = bool(plat.get("stop_and_go", False))
    for i in range(count):
        t = i * interval
        pose = make_pose(position + velocity * t, rpy, (0.0, 0.0, 0.0) if stop_and_go else velocity)
        truth_positions.append(position + velocity * t)
        records.append(simulate_ping(
            sonar, pose, scene, environment=env, provider=provider, sequence=i,
            sub_sequence=i % n_sub, time_ns=start_ns + int(round(t * 1e9)), scenario_time=t,
            absorption_model=cfg.get("absorption", "francois_garrison"),
            noise_rms=float(rx_cfg.get("noise_rms", 0.0)), adc=bool(rx_cfg.get("adc", True)),
            rng=rng, backend=backend))

    if output:
        opts = core.WriterOptions()
        opts.config_yaml = dump_yaml(cfg)
        opts.compression_level = int(cfg.get("compression_level", 1))
        with core.RecordWriter(str(output), opts) as w:
            for r in records:
                w.write(r)
            w.write_truth("platform_position", np.asarray(truth_positions))
            if len(scene) > 0:
                w.write_truth("scatterer_position", scene.positions)
                w.write_truth("scatterer_velocity", scene.velocities)
                w.write_truth("scatterer_amplitude",
                              np.stack([scene.amplitudes.real, scene.amplitudes.imag], axis=1))
            w.write_truth_text("fidelity", tier)
            w.write_truth_text("labels", json.dumps(scene.labels))
    return ScenarioResult(records, scene, cfg)
