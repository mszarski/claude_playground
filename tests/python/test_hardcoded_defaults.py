"""The fixes of docs/HARDCODED_AUDIT.md outside the live chain, seen from Python.

Each test is one audit row: a constant that was typed in code is now a key or is derived
from the record, today's value is still the default (the existing tests hold the outputs),
and a limit that used to cut silently raises or reports.
"""

import warnings

import numpy as np
import pytest

import gpusonar as gs
from gpusonar import acoustics, motion, pipeline, recon
from gpusonar.config import config_dir, load_yaml
from gpusonar.sim import Scene, make_pose, simulate_collection, simulate_ping
from gpusonar.sim import sources as src
from gpusonar.sim import waveform as wf_mod
from gpusonar.sonar import SonarModel

core = gs.core
C = 1500.0


# --- K-7 and the other constants two files must share ------------------------------------------

def test_constants_shared_by_two_engines_are_exported_and_equal():
    assert core.RESAMPLER_TAPS == wf_mod.DEFAULT_TAPS
    assert core.RESAMPLER_KAISER_BETA == wf_mod.DEFAULT_BETA
    assert core.SYNTH_ARRIVAL_PASSES == (4, 4) and core.SYNTH_SAMPLED_ARRIVAL_PASSES == 8
    assert core.BP_TRAJECTORY_ALLOWANCE == pytest.approx(0.02)
    assert (core.CFAR_DEFAULT_PFA, core.CFAR_DEFAULT_GUARD, core.CFAR_DEFAULT_TRAIN) == (1e-4, 4, 32)
    assert core.BASEBAND_CW_BAND_FACTOR == 4.0 and core.BASEBAND_BAND_MARGIN == 1.5


# --- K-2, K-3, R-2: no physical default ----------------------------------------------------------

def test_a_record_states_its_sound_speed_and_boresight_or_fails_validation():
    r = core.PingRecord()
    assert np.isnan(r.environment.sound_speed)
    assert np.isnan(core.Element().boresight[0])
    from conftest import make_record
    ok = make_record()
    assert core.validate(ok).ok
    env = ok.environment
    env.sound_speed = float("nan")
    ok.environment = env
    report = core.validate(ok)
    assert not report.ok and any(i.field == "environment.sound_speed" for i in report.issues)
    with pytest.raises(ValueError, match="sound speed"):
        core.backproject([ok], np.zeros((1, 3)))
    with pytest.raises(ValueError, match="sound speed"):
        core.apply_tvg(ok)


@pytest.mark.skipif(not core.HAVE_MOTION, reason="built without Ceres")
def test_motion_solver_has_no_default_scales():
    so = core.MotionSolveOptions()
    assert np.isnan(so.sound_speed) and np.isnan(so.residual_scale) and np.isnan(so.smooth_scale) and np.isnan(so.dvl_scale)
    pings = []
    for i in range(4):
        nav = core.PingNav()
        nav.time, nav.x, nav.surge = 0.2 * i, 0.3 * i, 1.5
        pings.append(nav)
    with pytest.raises(ValueError, match="sound_speed must be stated"):
        core.solve_motion(pings, [], so)
    so.sound_speed = C
    with pytest.raises(ValueError, match="have no default"):
        core.solve_motion(pings, [], so)
    so.residual_scale, so.smooth_scale, so.dvl_scale = (1.0 / 16.0) / 200e3, 2e-3, 5e-3
    assert core.solve_motion(pings, [], so).converged


# --- S-1, S-3: image sources ------------------------------------------------------------------

def test_image_sources_need_their_order_and_their_bottom():
    pts = np.array([[100.0, 0.0, 10.0]])
    with pytest.raises(ValueError, match="max_order"):
        core.sim_image_source_eigenrays(C, (0.0, 0.0, 5.0), pts)
    with pytest.raises(ValueError, match="bottom needs a reflection coefficient"):
        core.sim_image_source_eigenrays(C, (0.0, 0.0, 5.0), pts, bottom_z=30.0, max_order=2)
    with pytest.raises(ValueError, match="wall needs its reflection coefficient"):
        core.sim_image_source_eigenrays(C, (0.0, 0.0, 5.0), pts, max_order=1, walls=[(0, 50.0, complex("nan"))])
    e = core.sim_image_source_eigenrays(C, (0.0, 0.0, 5.0), pts, bottom_z=30.0, max_order=2, bottom_coefficient=0.5)
    assert e["delay"].shape == (1, 5)


def test_what_the_bounce_limit_leaves_out_is_measured():
    # 12 km in 100 m of water over a hard bottom: two bounces leave out a ray almost as
    # strong as the direct path, which is why the limit has no default.
    kw = dict(bottom_z=100.0, bottom_coefficient=0.9)
    left = [core.sim_image_series_truncation(C, 30.0, 60.0, 12000.0, max_order=n, **kw) for n in (2, 8, 40)]
    assert left[0] > 0.8 and left[0] > left[1] > left[2]
    # Closed form for the order after the last one kept (order 1 from max_order 0).
    direct = np.hypot(12000.0, 30.0)
    expect = max(1.0 / np.hypot(12000.0, -60.0 - 30.0), 0.9 / np.hypot(12000.0, 200.0 - 60.0 - 30.0)) * direct
    assert core.sim_image_series_truncation(C, 30.0, 60.0, 12000.0, max_order=0, **kw) == pytest.approx(expect, rel=1e-12)


# --- S-6, S-7: synthesis limits ---------------------------------------------------------------

def moving_ping(passes=None, limit=None, speed=12.0):
    sonar = SonarModel.load("flank_32", {"array": {"count": 2}, "receive": {"window": {"range": [350.0, 450.0]}},
                                        "transmit": {"waveform": {"duration": 0.01}}})
    rec = sonar.make_record(make_pose(), environment={"sound_speed": C})
    s = rec.samples
    s.array = np.zeros((s.n_channels, s.n_samples), dtype=np.float32)
    rec.samples = s
    q = np.array([[0.0, 400.0, 0.0]])
    v = np.array([[0.0, -speed, 0.0]])
    kw = {}
    if passes is not None:
        kw["arrival_passes"] = passes
    if limit is not None:
        kw["arrival_residual_limit"] = limit
    core.synthesize_into(rec, q, q, np.array([1.0 + 0j]), v, v, **kw)
    return rec


def test_arrival_passes_are_checked_and_can_be_derived():
    base = moving_ping().samples.array
    assert np.abs(base).max() > 0
    np.testing.assert_array_equal(moving_ping((4, 4)).samples.array, base)          # the default, stated
    with pytest.raises(ValueError, match="arrival-time solve may leave"):
        moving_ping((2, 4))                                                         # 2 passes: over a radian
    loose = moving_ping((2, 4), limit=0.0).samples.array                            # the check can be turned off
    auto = moving_ping("auto").samples.array
    # What two passes got wrong is visible; "auto" agrees with many passes to float rounding.
    exact = moving_ping((12, 12)).samples.array
    assert np.abs(loose - exact).max() > 100 * np.abs(auto - exact).max()
    assert np.abs(auto - exact).max() <= 2e-4 * np.abs(exact).max()
    # Left to the binding (as the simulator calls it), a case the default cannot hold is run with
    # more passes and says so; an ordinary case is silent and unchanged.
    with pytest.warns(RuntimeWarning, match="passes of the arrival-time solve could leave"):
        fast = moving_ping(speed=60.0).samples.array
    np.testing.assert_array_equal(fast, moving_ping("auto", speed=60.0).samples.array)
    with pytest.raises(ValueError, match="arrival-time solve may leave"):
        moving_ping((4, 4), speed=60.0)                                              # stated and not enough: an error
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        moving_ping()
    # The bound itself: T_end (v / c)^n 2 pi f.
    assert core.synth_arrival_residual_bound(0.6, 12.0, C, 1e4, 4) == pytest.approx(0.6 * (12.0 / C) ** 4 * 2 * np.pi * 1e4)
    n = core.synth_arrival_passes_for(0.6, 12.0, C, 1e4, 1e-4)
    assert core.synth_arrival_residual_bound(0.6, 12.0, C, 1e4, n) <= 1e-4 < core.synth_arrival_residual_bound(0.6, 12.0, C, 1e4, n - 1)


def test_absorption_table_must_cover_the_pulse():
    sonar = SonarModel.load("fls_330k", {"array": {"count": 2}})
    rec = sonar.make_record(make_pose(), environment={"sound_speed": C})
    s = rec.samples
    s.array = np.zeros((s.n_channels, s.n_samples), dtype=np.float32)
    rec.samples = s
    q = np.array([[200.0, 0.0, 0.0]])
    w = rec.transmit.waveform
    lo, hi = min(w.f_start, w.f_stop), max(w.f_start, w.f_stop)
    f = np.linspace(lo, hi, core.ABSORPTION_TABLE_SIZE)
    full = (lo, hi, [float(a) for a in acoustics.absorption(f, 12.0, 35.0, 0.0)])
    core.synthesize_into(rec, q, q, np.array([1.0 + 0j]), absorption=full)           # covers the band: fine
    # A table over the lower third of the band: the kernel would read its top end for the rest.
    f = np.linspace(lo, lo + (hi - lo) / 3.0, core.ABSORPTION_TABLE_SIZE)
    part = (float(f[0]), float(f[-1]), [float(a) for a in acoustics.absorption(f, 12.0, 35.0, 0.0)])
    with pytest.raises(ValueError, match="outside the absorption table"):
        core.synthesize_into(rec, q, q, np.array([1.0 + 0j]), absorption=part)


# --- K-4: the gain law ------------------------------------------------------------------------

def blank(sonar_name, environment, overrides=None):
    sonar = SonarModel.load(sonar_name, overrides or {})
    rec = sonar.make_record(make_pose(position=(0.0, 0.0, 5.0)), environment=environment)
    s = rec.samples
    kind = s.kind
    s.array = np.ones((1, s.n_samples), dtype=np.float32) if not s.is_complex() else np.ones((1, s.n_samples), dtype=np.complex64)
    s.kind = kind
    rec.samples = s
    geo = rec.geometry
    geo.rx = list(geo.rx)[:1]
    rec.geometry = geo
    rec.set_default_channel_tables()
    return rec


def test_absorption_from_the_record_is_the_audits_number_and_50_db_at_300_m():
    env = {"sound_speed": C, "temperature": 12.0, "salinity": 35.0}
    rec = blank("fls_330k", env, {"receive": {"tvg": None}})
    alpha = recon.record_absorption(rec, depth=0.0)
    assert alpha == pytest.approx(0.083, abs=0.002)                 # dB/m at 330 kHz, 12 deg C (audit K-4)
    assert alpha == pytest.approx(float(acoustics.absorption(330e3, 12.0, 35.0, 0.0)))
    # The law before D-91, by its name: no absorption.
    old = recon.tvg_parameters(rec, absorption="none")
    assert old == {"spreading_db_per_decade": 40.0, "absorption_db_per_m": 0.0, "reference_range": 1.0,
                   "max_gain_db": 120.0, "undo_hardware_ramp": True}
    # The default now derives it from the record (D-91); the clamp is as it was.
    default = recon.tvg_parameters(rec, depth=0.0)
    assert default == dict(old, absorption_db_per_m=alpha)
    assert recon.tvg_parameters(rec, absorption="record", depth=0.0) == default
    derived = recon.tvg_parameters(rec, max_gain_db=400.0, depth=0.0)
    short = recon.gain_law_db(300.0, **derived) - recon.gain_law_db(300.0, **old)
    assert short == pytest.approx(2 * alpha * 300.0) and 48.0 < short < 52.0

    # The kernel applies exactly that law, and reports the default clamp instead of hiding it.
    a = blank("fls_330k", env, {"receive": {"tvg": None}})
    t = a.samples.start_time + np.arange(a.samples.n_samples) / a.samples.sample_rate
    report = core.apply_tvg(a, **derived)
    np.testing.assert_allclose(20 * np.log10(np.abs(a.samples.array[0])), recon.gain_law_db(0.5 * C * t, **derived), atol=2e-3)
    assert report["clamped_samples"] == 0 and np.isnan(report["clamp_range"])
    b = blank("fls_330k", env, {"receive": {"tvg": None}})
    cut = core.apply_tvg(b, **recon.tvg_parameters(b, depth=0.0))
    assert cut["clamped_samples"] > 0 and 100.0 < cut["clamp_range"] < 200.0 and cut["shortfall_db"] > 20.0
    # `absorption: none` is the old law to the bit: the same samples as the bare C++ call.
    c, d = (blank("fls_330k", env, {"receive": {"tvg": None}}) for _ in range(2))
    core.apply_tvg(c, **old)
    core.apply_tvg(d)
    np.testing.assert_array_equal(c.samples.array, d.samples.array)


def test_tvg_derivation_refuses_to_assume_the_water_and_knows_one_way_spreading():
    rec = blank("fls_330k", {"sound_speed": C})                     # no temperature, no salinity
    with pytest.raises(ValueError, match="no temperature and salinity"):
        recon.tvg_parameters(rec, absorption="record")              # asked for by name: an error
    # By default such a record (one made before D-91) still processes: no absorption, and one
    # warning per process that names the missing fields.
    from gpusonar.recon import gain
    gain._warned_no_water = False
    with pytest.warns(RuntimeWarning, match="no `temperature` and no `salinity`"):
        assert recon.tvg_parameters(rec)["absorption_db_per_m"] == 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert recon.tvg_parameters(rec)["absorption_db_per_m"] == 0.0             # the second time is silent
        assert recon.tvg_parameters(rec, absorption="none")["absorption_db_per_m"] == 0.0
        # Water given beside the record stands in for what it lacks.
        assert recon.tvg_parameters(rec, temperature=5.0, salinity=35.0)["absorption_db_per_m"] > 0.06
    with pytest.raises(ValueError, match="apply only with"):
        recon.tvg_parameters(rec, absorption="none", temperature=5.0)
    assert recon.tvg_parameters(rec, absorption="record", temperature=5.0, salinity=35.0)["absorption_db_per_m"] > 0.06
    assert recon.tvg_parameters(rec, spreading="one_way")["spreading_db_per_decade"] == 20.0
    assert recon.tvg_parameters(rec, spreading="auto")["spreading_db_per_decade"] == 40.0
    with pytest.raises(ValueError, match="absorption"):
        core.apply_tvg(rec, absorption_db_per_m=-0.1)
    with pytest.raises(ValueError, match="apply only with"):
        recon.tvg_parameters(rec, absorption=0.05, temperature=5.0)
    # At 5 kHz the default law is right to a fraction of a dB at 10 km: the two sonar classes differ.
    low = blank("flank_32", {"sound_speed": C, "temperature": 12.0, "salinity": 35.0})
    assert 2 * recon.record_absorption(low) * 300.0 < 0.5


def test_pipeline_tvg_step_derives_and_logs(tmp_path):
    cfg = load_yaml(config_dir() / "chains" / "stripmap_points.yaml")
    cfg["source"]["overrides"] = {"pings": {"count": 2}}
    cfg["steps"] = [{"stage": "prepare"}, {"stage": "tvg"}]         # a bare step: absorption from the record (D-91)
    cfg["output"] = {}
    result = pipeline.run(cfg, backend=core.Backend.Host)
    entry = [e for e in result.log if e.get("stage") == "tvg"][0]
    alpha = float(acoustics.absorption(200e3, 12.0, 35.0, 0.0))
    assert entry["absorption_db_per_m"] == pytest.approx(alpha, rel=0.02)
    assert entry["spreading_db_per_decade"] == 40.0 and entry["clamped_samples"] == 0
    explicit = dict(cfg, steps=[{"stage": "prepare"}, {"stage": "tvg", "spreading": "two_way", "absorption": "record"}])
    named = pipeline.run(explicit, backend=core.Backend.Host)
    np.testing.assert_array_equal(named.records[0].samples.array, result.records[0].samples.array)
    # The law before D-91, pinned by its name; the older numeric key means the same.
    for step in ({"stage": "tvg", "absorption": "none"}, {"stage": "tvg", "absorption_db_per_m": 0.0}):
        old = pipeline.run(dict(cfg, steps=[{"stage": "prepare"}, step]), backend=core.Backend.Host)
        plain = [e for e in old.log if e.get("stage") == "tvg"][0]
        assert plain["absorption_db_per_m"] == 0.0
        # 52 m is the far end of this window: the old law is 2 alpha R lower there.
        last = lambda run: float(np.abs(run.records[0].samples.array[0, -1]))        # noqa: E731
        assert plain["gain_last_db"] == pytest.approx(entry["gain_last_db"] - 2 * entry["absorption_db_per_m"] * 52.0, abs=0.3)
        assert last(old) <= last(result)


# --- K-5: CFAR windows ------------------------------------------------------------------------

def test_cfar_window_in_resolution_cells_keeps_a_finely_sampled_target():
    assert core.cfar_window(0.5, 0.5) == (4, 32)                    # sampled at the resolution: the old default
    assert core.cfar_window(2.0, 2.0, guard_m=100.0, train_m=300.0) == (50, 150)
    # A map sampled 16 times finer than the resolution: a target is a 16-cell-wide lobe.
    rng = np.random.default_rng(3)
    n, over = 4096, 16
    noise = rng.exponential(1.0, (8, n)).astype(np.float32)
    lobe = np.sinc((np.arange(n) - n // 2) / over) ** 2
    power = (noise + 40.0 * lobe).astype(np.float32)
    naive = core.cfar(power, pfa=1e-6, guard=4, train=32)           # the guard is inside the target
    guard, train = core.cfar_window(1.0, float(over))
    sized = core.cfar(power, pfa=1e-6, guard=guard, train=train)
    assert (guard, train) == (4 * over, 32 * over)
    assert sized[:, n // 2].all() and not naive[:, n // 2].any()    # 16 dB over noise: found only with the sized window


# --- K-6: Doppler allowance -------------------------------------------------------------------

def test_baseband_default_is_unchanged_and_the_allowance_widens_a_tone():
    def tone():
        sonar = SonarModel.load("flank_32", {"array": {"count": 1}, "receive": {"window": {"range": [0.0, 375.0]}},
                                            "transmit": {"waveform": {"type": "cw", "f_start": 8000.0, "f_stop": 8000.0,
                                                                      "duration": 0.25}}})
        rec = sonar.make_record(make_pose(), environment={"sound_speed": C})
        s = rec.samples
        kind = s.kind
        t = np.arange(s.n_samples) / s.sample_rate
        s.array = np.cos(2 * np.pi * (8000.0 + 2 * 3.0 * 8000.0 / C) * t)[None, :].astype(np.float32)   # closing at 3 m/s
        s.kind = kind
        rec.samples = s
        return rec

    a, b, wide = tone(), tone(), tone()
    core.to_baseband(a)
    core.to_baseband(b, cw_band_factor=4.0, max_range_rate=0.0, doppler_allowance_hz=0.0)
    np.testing.assert_array_equal(a.samples.array, b.samples.array)
    core.to_baseband(wide, max_range_rate=5.0)
    # 4 / 0.25 s = 16 Hz kept by default; the echo is 32 Hz away and is filtered out.
    assert np.abs(wide.samples.array).mean() > 10 * np.abs(a.samples.array).mean()
    assert wide.samples.sample_rate > a.samples.sample_rate


# --- R-1: trajectory allowance of the culling pre-test -----------------------------------------

def test_trajectory_allowance_from_the_table_never_loses_what_the_exact_test_keeps():
    TILT = np.radians(20.0)
    target = 120.0 * np.array([0.0, np.cos(TILT), np.sin(TILT)])
    # A projector wider than the receive elements, so that the receive beam alone decides
    # the edge. (With equal beams the transmit test, exact at the transmit instant, cuts
    # the same pixels the pre-test would lose, and the 2 cm cost nothing.)
    sonar = SonarModel.load("sas_auv", {
        "receive": {"window": {"range": [115.0, 125.0]}},
        "transmit": {"beamwidth_az": 1.4, "aperture_width": None,
                     "waveform": {"type": "gaussian", "f_start": 180e3, "f_stop": 220e3, "duration": 1.0e-4,
                                  "window_alpha": 0.0}}})

    def ping(acceleration):
        rec = simulate_ping(sonar, make_pose(velocity=(1.6, 0.0, 0.0)), Scene().add_points([target]),
                            environment={"sound_speed": C}, absorption_model="none", adc=False,
                            acceleration=acceleration, ins_rate=100.0)
        recon.prepare([rec])
        return rec

    # Pixels along track across the edge of the processed beam: 1.13 x 0.166 rad / 2 is 11.3 m at 120 m.
    x = np.linspace(8.0, 15.0, 701)
    pts = np.column_stack([x, np.full_like(x, target[1]), np.full_like(x, target[2])])

    def image(rec, **kw):
        """The sum of the weights that contributed at each pixel: zero where the ping was rejected."""
        return core.backproject([rec], pts, interp="linear", fov_factor=1.13, position_lookup="software_table",
                                return_weight_sum=True, **kw)[1]

    # A gentle ping: the residual is under the fixed 2 cm, and the table-derived allowance is
    # the fixed one, bit for bit.
    gentle = ping((0.0, 1.0, 0.0))
    session = core.BackprojectSession([gentle], trajectory_upsample=8)
    assert 0.0 < session.trajectory_residual_max < core.BP_TRAJECTORY_ALLOWANCE
    np.testing.assert_array_equal(image(gentle, trajectory_allowance="table"), image(gentle))

    # A hard surge: by the 0.16 s arrival the platform is 1.3 m along track from its
    # first-order pose, so the beam's footprint has moved by that much. With an allowance
    # that covers any residual the pre-test rejects nothing the per-element test keeps.
    hard = ping((100.0, 0.0, 0.0))
    assert core.BackprojectSession([hard], trajectory_upsample=8).trajectory_residual_max > 1.0
    exact = image(hard, trajectory_allowance=50.0)
    fixed = image(hard)
    table = image(hard, trajectory_allowance="table")
    np.testing.assert_array_equal(table, exact)
    lost = (exact > 0) & (fixed == 0)
    assert lost.sum() > 20                                          # the fixed 2 cm drops pixels the exact test keeps
    assert not ((fixed > 0) & (exact == 0)).any() and (exact == 0).any()      # and the wider one adds nothing false
    with pytest.raises(ValueError, match="trajectory_allowance"):
        image(hard, trajectory_allowance="guess")


# --- R-3, R-4, R-5: motion windows, the imaging plane, one home for defaults -------------------

def test_motion_windows_in_metres_follow_the_record():
    class Rec:                                                      # only what window_sizes reads
        class samples:
            sample_rate, n_samples = 80e3, 4000

        class environment:
            sound_speed = C

    opts = motion.MotionOptions()
    assert motion.window_sizes(Rec, opts) == (256, 12)              # the defaults, in samples
    # 12 samples at 80 kHz are 0.1125 m of two-way range; stated in metres they follow the rate.
    per_m = 2 * 80e3 / C
    assert motion.window_sizes(Rec, motion.MotionOptions(window_m=256 / per_m, max_lag_m=12 / per_m)) == (256, 12)
    Rec.samples.sample_rate = 800e3
    assert motion.window_sizes(Rec, motion.MotionOptions(window_m=256 / per_m, max_lag_m=12 / per_m)) == (2560, 120)
    Rec.samples.n_samples = 100
    with pytest.raises(ValueError, match="does not fit"):
        motion.window_sizes(Rec, opts)


@pytest.fixture(scope="module")
def stripmap():
    """A short straight collection with a DVL: one point on the seabed, origin on the seabed."""
    sonar = SonarModel.load("sas_auv", {
        "receive": {"kind": "baseband", "sample_rate": 100e3, "window": {"range": [38.0, 48.0]}},
        "transmit": {"waveform": {"type": "gaussian", "f_start": 180e3, "f_stop": 220e3, "duration": 1e-4,
                                  "window_alpha": 0.0}}})
    col = simulate_collection(sonar, Scene().add_points([[4.8, 41.2, 0.0]]), n_pings=40, interval=0.2, speed=1.2,
                              environment={"sound_speed": C}, absorption_model="none", adc=False,
                              rng=np.random.default_rng(1))
    recon.prepare(col.records)
    return col.records


def lowered(records, dz):
    """The same collection in a frame whose origin is `dz` metres higher (at the surface, say)."""
    out = []
    for r in records:
        r = r.copy()
        geo = r.geometry
        for name in ("rx_pose", "tx_pose"):
            pose = getattr(geo, name)
            p = pose.position
            pose.position = (p[0], p[1], p[2] + dz)
            setattr(geo, name, pose)
        r.geometry = geo
        out.append(r)
    return out


def test_imaging_plane_from_the_records_follows_the_frame(stripmap):
    z, source = recon.seafloor_z(stripmap)
    assert z == pytest.approx(0.0, abs=0.02) and source == "dvl"    # origin on the seabed: the default plane is right
    deep = lowered(stripmap, 30.0)                                  # the same survey with its origin at the surface
    assert recon.seafloor_z(deep)[0] == pytest.approx(30.0, abs=0.02)
    here = recon.stripmap_grid(stripmap, (41.0, 41.4), 0.02)
    there = recon.stripmap_grid(deep, (41.0, 41.4), 0.02, plane_z="seafloor")
    assert here.z == 0.0 and there.z == pytest.approx(30.0, abs=0.02)
    kw = dict(interp="linear", backend=core.Backend.Host)
    focused = np.abs(recon.sas_image(stripmap, here.points, **kw)).max()
    np.testing.assert_allclose(np.abs(recon.sas_image(deep, there.points, **kw)).max(), focused, rtol=0.05)
    # The default plane on the surface-origin survey: 30 m above the seabed, nothing focuses.
    wrong = recon.stripmap_grid(deep, (41.0, 41.4), 0.02)
    assert np.abs(recon.sas_image(deep, wrong.points, **kw)).max() < 0.2 * focused
    with pytest.raises(ValueError, match="plane_z must be"):
        recon.stripmap_grid(stripmap, (41.0, 41.4), 0.02, plane_z="bottom")


def test_seafloor_from_water_depth_or_an_error():
    rec = SonarModel.load("sas_auv").make_record(make_pose(position=(0.0, 0.0, 12.0)),
                                                 environment={"sound_speed": C, "water_depth": 40.0})
    assert recon.seafloor_z([rec]) == (40.0, "water_depth")
    bare = SonarModel.load("sas_auv").make_record(make_pose(position=(0.0, 0.0, 12.0)), environment={"sound_speed": C})
    with pytest.raises(ValueError, match="neither a DVL altitude nor"):
        recon.seafloor_z([bare])
    assert motion.resolve_seafloor_z([rec], motion.MotionOptions(seafloor_z="auto")) == 40.0
    assert motion.resolve_seafloor_z([rec], motion.MotionOptions()) == 0.0


def test_pipeline_passes_only_the_keys_it_is_given(monkeypatch, stripmap):
    import inspect
    sig = inspect.signature(recon.sas_image).parameters             # the real functions, before they are replaced
    grid = inspect.signature(recon.stripmap_grid).parameters
    seen = {}

    def fake_grid(records, ground_range, pixel, **kw):
        seen["grid"] = kw
        return recon.StripmapGrid(np.zeros(2), np.zeros(2), 0.0, np.zeros((2, 2, 3)))

    def fake_image(records, points, **kw):
        seen["image"] = kw
        return np.zeros(points.shape[:2], dtype=np.complex64)

    monkeypatch.setattr(pipeline.recon, "stripmap_grid", fake_grid)
    monkeypatch.setattr(pipeline.recon, "sas_image", fake_image)
    state = pipeline.RunResult(records=stripmap[:2])
    facts = pipeline._stage("sas_image", {"ground_range": [41.0, 41.4], "pixel": 0.02}, state, core.Backend.Host)
    assert seen["grid"] == {} and set(seen["image"]) == {"stats", "backend"}       # no default typed in the pipeline
    assert facts["plane_z"] == 0.0 and facts["seafloor_z_source"] == "dvl"
    pipeline._stage("sas_image", {"ground_range": [41.0, 41.4], "pixel": 0.02, "fov_factor": 1.6, "plane_z": "seafloor",
                                  "trajectory_allowance": "table"}, state, core.Backend.Host)
    assert seen["grid"] == {"plane_z": "seafloor"} and seen["image"]["fov_factor"] == 1.6
    assert seen["image"]["trajectory_allowance"] == "table"
    # The defaults the pipeline used to type are the defaults of the functions it calls.
    assert (sig["interp"].default, sig["fov_factor"].default, sig["cull_block"].default) == ("sinc", 1.13, 16)
    assert (grid["plane_z"].default, grid["full_support"].default) == (0.0, True)


def test_volume_refraction_keys_go_together(stripmap):
    with pytest.raises(ValueError, match="go together"):
        recon.image_volume(stripmap[:1], [0.0], [41.0], [0.1], interface_z=0.0)


# --- M-9: knot spacing of a one-way source -----------------------------------------------------

def test_knot_phase_error_is_the_parabola_bound():
    h, a = 4e-3, 0.3                                                # delay = a t^2 on knots h apart
    t = np.arange(8) * h
    assert src.knot_phase_error((a * t**2)[None, :], 1e4) == pytest.approx(2 * np.pi * 1e4 * a * h * h / 4)
    assert src.knot_phase_error(np.zeros((1, 2)), 1e4) == 0.0


def test_a_fast_close_pass_warns_and_auto_refines():
    model = SonarModel.load("flank_32", {"array": {"count": 2, "pitch": 0.094},
                                        "transmit": {"waveform": {"type": "cw", "f_start": 8000.0, "f_stop": 8000.0,
                                                                  "duration": 0.004, "window_alpha": 0.0,
                                                                  "source_level_db": 190.0}},
                                        "receive": {"kind": "real", "sample_rate": 64000.0,
                                                    "window": {"range": [0.0, 75.0]}}})          # 0.1 s
    env = {"sound_speed": C, "temperature": 12.0, "salinity": 35.0}

    def listen(stream=None, speed=40.0):
        item = {"origin": [-2.0, 1.5, 0.0], "velocity": [speed, 0.0, 0.0],        # passes 1.5 m abeam at 0.05 s
                "signal": {"type": "tonals", "lines": [{"frequency": 20000.0, "level_db": 150.0, "phase_deg": 0.0}]}}
        if stream is not None:
            item["stream"] = stream
        return simulate_ping(model, make_pose(), Scene(), environment=env, absorption_model="none", adc=False,
                             sources=src.build_sources([item], np.random.default_rng(1)), transmit=False,
                             rng=np.random.default_rng(1)).samples.array

    with pytest.warns(RuntimeWarning, match="knot_interval"):
        coarse = listen()                                           # the default 4 ms spacing
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        auto = listen({"knot_interval": "auto"})
        fine = listen({"knot_interval": 2.5e-5})
        slow = listen(speed=0.5)                                    # an ordinary source: the default is fine, no warning
    assert np.abs(slow).max() > 0
    scale = np.abs(fine).max()
    assert np.abs(auto - fine).max() < 0.02 * scale < np.abs(coarse - fine).max()
    with pytest.raises(ValueError, match="unknown keys"):
        src.build_sources([{"origin": [0, 1, 0], "signal": {"type": "noise"}, "stream": {"knots": 3}}])


# --- G-10: flags ------------------------------------------------------------------------------

def test_design_values_of_the_shipped_sonars_are_flagged():
    flags = {n: SonarModel.load(n).config.get("unconfirmed", []) for n in
             ("flank_32", "fls_120k", "fls_170k", "fls_330k", "fls_300k", "fls_750k", "sas_auv", "sas_minsas120",
              "sas_rail", "subbottom")}
    assert {"array", "transmit", "receive"} <= set(flags["flank_32"])               # audit G-10
    for name, keys in flags.items():
        covered = lambda key: any(key == k or key.startswith(k + ".") for k in keys)       # noqa: E731
        assert covered("array.pitch") or name == "sas_rail", name                   # sas_rail: a real probe's pitch
        assert covered("transmit.waveform.duration"), name
        assert covered("receive.adc_full_scale") or name in ("subbottom",) or covered("receive"), name
