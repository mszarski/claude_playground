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
                  channel_errors: dict | None = None, clock_truth: dict | None = None,
                  noise_spectrum=None, reverberation: list | None = None,
                  backend=core.Backend.Host) -> "core.PingRecord":
    """Simulate one transmit and return its record."""
    env = resolve_environment(environment, depth=pose.position[2])
    record = sonar.make_record(pose, sequence=sequence, sub_sequence=sub_sequence,
                               time_ns=time_ns, environment=env, tx_pose=tx_pose)
    c = record.environment.sound_speed
    provider = provider or make_provider("free_field", c)
    # The true board clock errors drive synthesis, then are cleared: the record a
    # processor receives does not know them (the sync stage estimates them).
    clocks = _set_board_clocks(record, sonar, truth=True, rng=rng)
    if clock_truth is not None:
        clock_truth.update(clocks)
    if channel_errors is not None and channel_errors.get("position") is not None:
        # True element position errors drive synthesis; the processor is not told.
        cal = record.calibration
        cal.position_offset = [tuple(p) for p in np.asarray(channel_errors["position"], dtype=float)]
        record.calibration = cal
    if len(scene) > 0:
        tx_position = np.array(record.tx_position())
        rx_reference = record.rx_positions().mean(axis=0)
        points, amplitudes = scene.at_time(scenario_time), scene.amplitudes
        if scene.occluders:
            # Shadowing: a scatterer hidden from the projector or from the array returns nothing.
            seen = scene.visible(tx_position, points) & scene.visible(rx_reference, points)
            amplitudes = np.where(seen, amplitudes, 0.0)
        q, p, amp, vq, vp = two_way_paths(provider, tx_position, rx_reference, points, amplitudes,
                                          scene.velocities, scene.lambert,
                                          scene.normals if len(scene.normals) == len(scene) else None)
        core.synthesize_into(record, q, p, amp, vq, vp,
                             absorption=absorption_table(sonar, env, absorption_model),
                             spreading=spreading, backend=backend,
                             platform_acceleration=tuple(float(a) for a in acceleration))
    for item in reverberation or []:
        # Boundary reverberation by the scattering-function method (sim/reverb.py).
        from . import reverb
        w = sonar.waveform_config
        fc = 0.5 * (float(w["f_start"]) + float(w.get("f_stop", w["f_start"])))
        alpha = float(acoustics.absorption(fc, env.get("temperature", 10.0), env.get("salinity", 35.0),
                                           env.get("depth", 0.0), env.get("ph", 8.0), absorption_model))
        strength = None
        if item.get("model", "lambert") == "chapman_harris":
            strength = reverb.chapman_harris_strength(float(item["wind_speed"]), fc)
        elif item.get("model", "lambert") != "lambert":
            raise ValueError(f"unknown reverberation model {item['model']!r}")
        reverb.add_reverberation(record, provider, float(item["boundary_z"]), float(item.get("mu_db", -27.0)),
                                 rng or np.random.default_rng(), n_azimuth=item.get("n_azimuth"),
                                 strength=strength,
                                 absorption_db_per_m=alpha, max_range=item.get("max_range"))
    receiver.add_reference_signal(record, sonar, clocks)
    _set_board_clocks(record, sonar, truth=False)
    if channel_errors is not None:
        receiver.apply_channel_errors(record, channel_errors)
    if ins_rate > 0.0:
        _add_true_ins(record, np.asarray(acceleration, dtype=float), ins_rate)
    samples = record.samples.array
    if noise_rms > 0.0:
        samples = receiver.add_noise(samples, noise_rms, rng or np.random.default_rng())
    if noise_spectrum is not None:
        # Ambient noise with a spectrum (dB re 1 uPa^2/Hz against Hz) and the coherence of
        # an isotropic field across the array. Real passband records only.
        if np.iscomplexobj(samples):
            raise ValueError("noise_spectrum needs a real passband record (receive.kind: real)")
        from . import noise as noise_models
        rx_cfg = sonar.config["receive"]
        sens = float(rx_cfg.get("sensitivity_db") or 0.0) + float(rx_cfg.get("gain_db", 0.0))
        samples = samples + noise_models.gaussian_noise(
            samples.shape[1], record.samples.sample_rate, noise_spectrum, rng or np.random.default_rng(),
            positions=record.rx_positions(), sound_speed=c, sensitivity_db=sens)
    samples = receiver.apply_hardware_tvg(record, samples)
    if adc:
        samples = receiver.apply_adc(record, samples)
    s = record.samples
    kind = s.kind
    s.array = samples
    s.kind = kind
    record.samples = s
    return record


def _noise_spectrum(cfg):
    """Spectrum callable from a scenario's `receiver.ambient_noise` mapping:
    {level_db: 45} for a flat spectrum, or {wind_speed: 5, shipping: 0.5} for the Wenz fit."""
    if not cfg:
        return None
    from . import noise as noise_models
    if "level_db" in cfg:
        level = float(cfg["level_db"])
        return lambda f: np.full_like(np.asarray(f, dtype=float), level)
    return lambda f: noise_models.wenz_spectrum_db(f, float(cfg.get("wind_speed", 5.0)),
                                                    float(cfg.get("shipping", 0.5)))


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


def _set_board_clocks(record, sonar: SonarModel, truth: bool, rng=None) -> dict:
    """Write the boards' true clock errors into the record (truth=True) or clear them.
    Returns the truth: {"offset": [...], "drift_ppm": [...]} per board. Each ping draws
    its own jitter on top of the configured offset."""
    meta = record.metadata
    boards = meta.boards
    offsets, drifts = [], []
    for board, cfg in zip(boards, sonar.boards):
        offset = float(cfg.get("clock_offset", 0.0))
        jitter = float(cfg.get("jitter_rms", 0.0))
        if truth and jitter > 0.0 and rng is not None:
            offset += jitter * float(rng.standard_normal())
        drift = float(cfg.get("clock_drift_ppm", 0.0))
        board.clock_offset = offset if truth else 0.0
        board.clock_drift_ppm = drift if truth else 0.0
        offsets.append(offset)
        drifts.append(drift)
    meta.boards = boards
    record.metadata = meta
    return {"offset": offsets, "drift_ppm": drifts}


@dataclass
class Collection:
    """A multi-ping run with a true trajectory and what navigation reported."""

    records: list            # poses as navigation reported them (what a processor gets)
    true_positions: np.ndarray   # [n + 1, 3] true platform position at each transmit (one extra node)
    nav_positions: np.ndarray    # [n, 3]
    times: np.ndarray            # [n] transmit times, s
    scene: Scene


def smooth_disturbance(n: int, rms: float, correlation_pings: float, rng: np.random.Generator) -> np.ndarray:
    """A smooth zero-mean random sequence: white noise through a Gaussian filter, scaled to `rms`."""
    if rms <= 0.0:
        return np.zeros(n)
    pad = int(4 * correlation_pings) + 1
    white = rng.standard_normal(n + 2 * pad)
    k = np.arange(-pad, pad + 1)
    kernel = np.exp(-0.5 * (k / correlation_pings) ** 2)
    out = np.convolve(white, kernel / kernel.sum(), mode="same")[pad:pad + n]
    out -= out[0]                       # start on the nominal track
    return out * rms / max(out.std(), 1e-12)


def simulate_collection(sonar: SonarModel, scene: Scene, *, n_pings: int, interval: float,
                        start=(0.0, 0.0, -15.0), speed: float = 1.2, sway_rms: float = 0.0,
                        heave_rms: float = 0.0, correlation_pings: float = 8.0,
                        nav_knows_disturbance: bool = False, dvl_altitude_noise: float = 0.0,
                        environment: dict | None = None, rng: np.random.Generator | None = None,
                        seafloor_depth: float = 0.0, backend=core.Backend.Host,
                        stop_and_go: bool = False, **ping_kwargs) -> Collection:
    """Fly a straight line along +x with sway (y) and heave (z) disturbances.

    The true trajectory is piecewise linear: during ping i the platform moves at
    v_i = (p_{i+1} - p_i) / interval, the paper's first-order kinematic model. Echoes are
    synthesised from the truth. Each record then carries what navigation would report:
    the nominal straight track (unless `nav_knows_disturbance`), plus a DVL sample with
    the true altitude and surge speed. With `stop_and_go` (a rail) the platform stands
    still during each ping; the disturbances are then position errors of the rail.
    """
    rng = rng or np.random.default_rng(0)
    start = np.asarray(start, dtype=float)
    t = np.arange(n_pings + 1) * interval
    truth = np.tile(start, (n_pings + 1, 1))
    truth[:, 0] += speed * t
    truth[:, 1] += smooth_disturbance(n_pings + 1, sway_rms, correlation_pings, rng)
    truth[:, 2] += smooth_disturbance(n_pings + 1, heave_rms, correlation_pings, rng)
    nominal = np.tile(start, (n_pings, 1))
    nominal[:, 0] += speed * t[:-1]

    records = []
    for i in range(n_pings):
        v_true = np.zeros(3) if stop_and_go else (truth[i + 1] - truth[i]) / interval
        rec = simulate_ping(sonar, make_pose(truth[i], velocity=v_true), scene, environment=environment,
                            sequence=i, time_ns=1_700_000_000_000_000_000 + int(round(t[i] * 1e9)),
                            scenario_time=t[i], rng=rng, backend=backend, **ping_kwargs)
        nav_pos = truth[i] if nav_knows_disturbance else nominal[i]
        nav_vel = v_true if (nav_knows_disturbance or stop_and_go) else np.array([speed, 0.0, 0.0])
        geo = rec.geometry
        geo.rx_pose = make_pose(nav_pos, velocity=nav_vel)
        geo.tx_pose = geo.rx_pose
        rec.geometry = geo
        dvl = core.DvlSample()
        dvl.t = 0.0
        dvl.velocity_body = (speed, 0.0, 0.0)
        dvl.altitude = float(seafloor_depth - truth[i, 2] + dvl_altitude_noise * rng.standard_normal())
        nav = rec.navigation
        nav.dvl = [dvl]
        rec.navigation = nav
        records.append(rec)
    return Collection(records, truth, nominal if not nav_knows_disturbance else truth[:-1], t[:-1], scene)


@dataclass
class ScenarioResult:
    records: list
    scene: Scene
    config: dict
    true_positions: np.ndarray = None      # [n_pings, 3] platform position at each transmit


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
    rng = np.random.default_rng(int(cfg.get("seed", 0)))
    scene = build_scene(cfg.get("scene", []), rng)

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
    ping_kwargs = dict(
        provider=provider, absorption_model=cfg.get("absorption", "francois_garrison"),
        noise_rms=float(rx_cfg.get("noise_rms", 0.0)), adc=bool(rx_cfg.get("adc", True)),
        noise_spectrum=_noise_spectrum(rx_cfg.get("ambient_noise")),
        reverberation=cfg.get("reverberation"))
    motion = plat.get("motion")
    raster = plat.get("raster")
    if motion is not None:
        # A straight run along +x with sway and heave the navigation does not know about
        # (unless nav_knows_disturbance): the input for motion estimation.
        if np.any(np.abs(velocity[1:]) > 0) or np.any(np.asarray(rpy, dtype=float) != 0):
            raise ValueError("platform.motion needs a level platform moving along +x")
        col = simulate_collection(
            sonar, scene, n_pings=count, interval=interval, start=position, speed=float(velocity[0]),
            sway_rms=float(motion.get("sway_rms", 0.0)), heave_rms=float(motion.get("heave_rms", 0.0)),
            correlation_pings=float(motion.get("correlation_pings", 8.0)),
            nav_knows_disturbance=bool(motion.get("nav_knows_disturbance", False)),
            dvl_altitude_noise=float(motion.get("dvl_altitude_noise", 0.0)), environment=env, rng=rng,
            seafloor_depth=float(motion.get("seafloor_depth", 0.0)), backend=backend,
            stop_and_go=stop_and_go, **ping_kwargs)
        records, truth_positions = col.records, list(col.true_positions[:count])
    else:
        if raster is not None:
            # A stop-and-go raster over x and y at the platform's z: {x: [start, stop, step], y: [...]}.
            xs = np.arange(*[float(v) for v in raster["x"]])
            ys = np.arange(*[float(v) for v in raster["y"]])
            positions = [np.array([px, py, position[2]]) for px in xs for py in ys]
            stop_and_go = True
        else:
            positions = [position + velocity * (i * interval) for i in range(count)]
        for i, pos in enumerate(positions):
            t = i * interval
            pose = make_pose(pos, rpy, (0.0, 0.0, 0.0) if stop_and_go else velocity)
            truth_positions.append(pos)
            records.append(simulate_ping(
                sonar, pose, scene, environment=env, sequence=i, sub_sequence=i % n_sub,
                time_ns=start_ns + int(round(t * 1e9)), scenario_time=t, rng=rng, backend=backend,
                **ping_kwargs))

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
    return ScenarioResult(records, scene, cfg, np.asarray(truth_positions, dtype=float))
