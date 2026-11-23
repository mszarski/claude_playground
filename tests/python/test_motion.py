"""Milestone 8: motion estimation (ASASIN Sec. II-D, Eq. 3-8).

Acceptance: with noisy simulated navigation, sway and heave errors are below a stated
fraction of a wavelength, and the image focuses close to the true-motion image.
"""

import numpy as np
import pytest

import gpusonar as gs
from gpusonar import motion, recon
from gpusonar.sim import Scene, seafloor_patch, simulate_collection
from gpusonar.sonar import SonarModel

core = gs.core
pytestmark = pytest.mark.skipif(not core.HAVE_MOTION, reason="built without Ceres")

C, FC = 1500.0, 200e3
WAVELENGTH = C / FC            # 7.5 mm


# --- the solver on exact synthetic delays ---------------------------------------------------

def synthetic_problem(n=30, noise=0.0, outliers=0, seed=0):
    rng = np.random.default_rng(seed)
    dt, speed, alt = 0.2, 1.2, 15.0
    t = np.arange(n + 1) * dt
    y_true = 0.02 * np.sin(2 * np.pi * t / 3.0)
    z_true = -alt + 0.01 * np.sin(2 * np.pi * t / 4.0 + 1.0) - 0.01 * np.sin(1.0)
    pings = []
    for i in range(n):
        p = core.PingNav()
        p.time, p.x, p.surge = t[i], speed * t[i], speed
        p.y, p.z = 0.0, -alt                          # navigation: a straight, level line
        p.dvl_altitude = -z_true[i]
        pings.append(p)
    measurements = []
    for i in range(n - 1):
        for ground in np.linspace(30.0, 50.0, 8):
            m = core.DelayMeasurement()
            m.ping = i
            m.tx_lever_1 = m.tx_lever_2 = (0.0, 0.0, 0.0)
            m.rx_lever_1, m.rx_lever_2 = (0.24, 0.0, 0.0), (-0.24, 0.0, 0.0)   # 0.24 m advance: shared phase centre
            m.footprint = (speed * t[i] + 0.12, ground, 0.0)
            m.t_ref = 2 * np.hypot(ground, alt) / C
            m.delay = 0.0
            m.delay = core.predict_delay(pings, m, y_true.tolist(), z_true.tolist(), C) + noise * rng.standard_normal()
            measurements.append(m)
    for j in rng.choice(len(measurements), outliers, replace=False):
        measurements[j].delay += 20e-6 * rng.choice([-1, 1])
    return pings, measurements, y_true, z_true


def options(**kw):
    o = core.MotionSolveOptions()
    o.sound_speed = C
    o.residual_scale = 1.0 / (16 * FC)
    o.smooth_scale, o.dvl_scale = 1e-3, 1e-2       # stated: the solver has no default scales
    for k, v in kw.items():
        setattr(o, k, v)
    return o


def test_solver_recovers_sway_and_heave_from_exact_delays():
    pings, meas, y_true, z_true = synthetic_problem()
    sol = core.solve_motion(pings, meas, options(lambda_smooth=0.0, lambda_dvl=1.0))
    assert sol.converged, sol.report
    assert np.abs(np.array(sol.y) - y_true).max() < 1e-5
    assert np.abs(np.array(sol.z) - z_true).max() < 1e-5
    assert sol.final_cost < 1e-6 * sol.initial_cost
    # Velocities are the first-order model: positions are their integral.
    np.testing.assert_allclose(np.diff(sol.y) / 0.2, sol.vy, atol=1e-12)


def test_residual_is_zero_at_the_truth_and_has_the_stated_sign():
    """Eq. 4 with delta = tau_2 - tau_1 (D-13): moving away from the footprint makes delta positive."""
    pings, meas, y_true, z_true = synthetic_problem(n=4)
    m = meas[0]
    flat = core.predict_delay(pings, m, [0.0] * 5, [-15.0] * 5, C)
    away = core.predict_delay(pings, m, [0.0, -0.01, -0.01, -0.01, -0.01], [-15.0] * 5, C)   # second ping 1 cm further
    # Same phase centre and no sway or heave, yet not exactly zero: the receiver travels
    # 5 cm forward during reception, which breaks the mirror symmetry of the two bistatic
    # pairs. The model carries that term, so a stop-and-hop DPC bias is not left in the data.
    assert 1e-8 < abs(flat) < 1e-6
    assert away > flat
    # Second ping 1 cm further: 2 cm more path along the line of sight. The first ping gets
    # there at -5 cm/s, so its own receiver has already moved by v * t_ref at reception,
    # which takes that much off the difference.
    cos_graze = np.cos(np.arctan2(15.0, 30.0))
    expected = (2 * 0.01 - (0.01 / 0.2) * m.t_ref) * cos_graze / C
    assert away - flat == pytest.approx(expected, rel=0.02)


def test_huber_loss_resists_outliers_where_squared_loss_does_not():
    pings, meas, y_true, _ = synthetic_problem(noise=0.05e-6, outliers=12, seed=3)
    errs = {}
    for huber in (True, False):
        # A tight DVL term pins heave, so what is left in sway is the outliers' doing.
        sol = core.solve_motion(pings, meas, options(huber=huber, lambda_smooth=0.0, dvl_scale=1e-4))
        errs[huber] = np.abs(np.array(sol.y) - y_true).max()
    # Twelve outliers of 20 microseconds (15 mm of path each) among 232 measurements.
    assert errs[True] < WAVELENGTH             # Huber: still inside a wavelength
    assert errs[False] > 10 * errs[True]       # squared loss: an order of magnitude worse


def test_smoothness_term_reduces_noise_and_dvl_term_holds_altitude():
    pings, meas, y_true, z_true = synthetic_problem(noise=0.3e-6, seed=5)
    rough = core.solve_motion(pings, meas, options(lambda_smooth=0.0))
    smooth = core.solve_motion(pings, meas, options(lambda_smooth=1.0, smooth_scale=5e-3))
    def rms(sol): return np.sqrt(np.mean((np.array(sol.y) - y_true) ** 2))
    assert rms(smooth) < rms(rough)
    # Without the DVL term heave is free to drift with the noise; with it, it is pinned.
    free = core.solve_motion(pings, meas, options(lambda_dvl=0.0, lambda_smooth=0.0))
    held = core.solve_motion(pings, meas, options(lambda_dvl=10.0, lambda_smooth=0.0, fix_first_height=True))
    drift_free = np.abs(np.array(free.z)[:-1] - z_true[:-1]).max()
    drift_held = np.abs(np.array(held.z)[:-1] - z_true[:-1]).max()
    assert drift_held < drift_free and drift_held < 2e-3      # 1.999 mm: ping 0 held on its (here exact) DVL sample
    # The default frees ping 0's height (sonar-jh8). With these exact DVL samples holding it was
    # an advantage; freed, the largest error is 2.115 mm, still a sixteenth of the free drift.
    default = core.solve_motion(pings, meas, options(lambda_dvl=10.0, lambda_smooth=0.0))
    drift_default = np.abs(np.array(default.z)[:-1] - z_true[:-1]).max()
    assert drift_default < drift_free / 10 and drift_default < 2.5e-3


def test_one_noisy_dvl_sample_on_the_first_ping_does_not_offset_the_track():
    """sonar-jh8. Exact delays; only the first ping's DVL altitude is wrong (3 mm, a 3-sigma draw of
    the simulated 1 mm DVL noise). Held as the anchor (fix_first_height, D-15 as first written) that
    error offsets every height and, through the line of sight at 17-27 degrees of depression, the
    sway by about tan(depression) times as much. Freed, ping 0's height is weighed against its
    DVL sample like every other ping, and the delays carry the rest."""
    pings, meas, y_true, z_true = synthetic_problem()
    pings[0].dvl_altitude += 3e-3
    pings[0].z = -pings[0].dvl_altitude                           # where estimate_motion starts it
    def rms(sol, truth, key): return np.sqrt(np.mean((np.array(getattr(sol, key))[:-1] - truth[:-1]) ** 2))
    held = core.solve_motion(pings, meas, options(lambda_smooth=0.0, fix_first_height=True))
    free = core.solve_motion(pings, meas, options(lambda_smooth=0.0))
    assert not core.MotionSolveOptions().fix_first_height          # the default frees it
    assert free.converged
    # Seen: held 1.10 mm heave and 0.91 mm sway RMS; freed 0.14 mm and 0.12 mm.
    assert rms(held, z_true, "z") > 0.75e-3 and rms(held, y_true, "y") > WAVELENGTH / 20
    assert rms(free, z_true, "z") < 0.3e-3 and rms(free, y_true, "y") < 0.2e-3
    # Without a DVL term there is nothing to tie the height to: it stays held.
    nodvl = core.solve_motion(pings, meas, options(lambda_smooth=0.0, lambda_dvl=0.0))
    assert nodvl.z[0] == pytest.approx(pings[0].z)


def test_correlation_weighting_downweights_bad_measurements():
    pings, meas, y_true, _ = synthetic_problem(seed=7)
    rng = np.random.default_rng(7)
    for m in meas[::5]:
        m.delay += 3e-6 * rng.standard_normal()    # a fifth of the measurements are poor...
        m.weight = 0.05                             # ...and their correlation coefficient says so
    weighted = core.solve_motion(pings, meas, options(huber=False, lambda_smooth=0.0, use_weights=True))
    unweighted = core.solve_motion(pings, meas, options(huber=False, lambda_smooth=0.0, use_weights=False))
    e_w = np.abs(np.array(weighted.y) - y_true).max()
    e_u = np.abs(np.array(unweighted.y) - y_true).max()
    assert e_w < 0.3 * e_u


# --- attitude and lever arms (Eq. 4's Omega) ----------------------------------------------------

def rotation_eq2(roll, pitch, yaw):
    """ASASIN Eq. 2, R = Rz(yaw) Ry(pitch) Rx(roll), written out here independently of libs/core."""
    cr, sr, cp, sp, cy, sy = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def eq4_delay(nav, m, y, z, c=C):
    """Eq. 4's model delay tau_2 - tau_1 in NumPy (instantaneous transmit, receiver moving
    with first-order kinematics until t_ref, lever arms rotated by each ping's attitude)."""
    def tof(i, tx_lever, rx_lever):
        a = nav[i]
        dt = nav[i + 1]["time"] - a["time"] if i + 1 < len(nav) else a["time"] - nav[i - 1]["time"]
        r = rotation_eq2(*a["rpy"])
        tx = np.array([a["x"], y[i], z[i]]) + r @ np.asarray(tx_lever)
        v = np.array([a["surge"], (y[i + 1] - y[i]) / dt, (z[i + 1] - z[i]) / dt])
        rx = np.array([a["x"], y[i], z[i]]) + v * m["t_ref"] + r @ np.asarray(rx_lever)
        ps = np.asarray(m["footprint"])
        return (np.linalg.norm(tx - ps) + np.linalg.norm(rx - ps)) / c
    i = m["ping"]
    return tof(i + 1, m["tx_lever_2"], m["rx_lever_2"]) - tof(i, m["tx_lever_1"], m["rx_lever_1"])


ATTITUDE_RMS = np.radians([2.0, 1.0, 1.0])          # roll, pitch, yaw of the tests below (ours: a lively AUV)


def attitude_problem(n=30, seed=11):
    """synthetic_problem with varying roll, pitch and yaw and non-zero lever arms. The delays
    come from eq4_delay (NumPy), not from the solver's own predict_delay."""
    from gpusonar.sim.simulator import smooth_disturbance
    rng = np.random.default_rng(seed)
    dt, speed, alt = 0.2, 1.2, 15.0
    t = np.arange(n + 1) * dt
    y_true = 0.02 * np.sin(2 * np.pi * t / 3.0)
    z_true = -alt + 0.01 * np.sin(2 * np.pi * t / 4.0 + 1.0) - 0.01 * np.sin(1.0)
    rpy = np.stack([smooth_disturbance(n, s, 8.0, rng) + s for s in ATTITUDE_RMS], axis=1)
    tx_lever = (0.10, 0.05, -0.20)
    rx_base = np.array([0.0, 0.08, -0.15])
    nav, pings = [], []
    for i in range(n):
        nav.append({"time": t[i], "x": speed * t[i], "surge": speed, "rpy": rpy[i]})
        p = core.PingNav()
        p.time, p.x, p.surge = t[i], speed * t[i], speed
        p.y, p.z = 0.0, -alt
        p.rpy = tuple(rpy[i])
        p.dvl_altitude = -z_true[i]
        pings.append(p)
    measurements = []
    for i in range(n - 1):
        for ground in np.linspace(30.0, 50.0, 8):
            md = {"ping": i, "tx_lever_1": tx_lever, "tx_lever_2": tx_lever,
                  "rx_lever_1": tuple(rx_base + [0.24, 0, 0]), "rx_lever_2": tuple(rx_base - [0.24, 0, 0]),
                  "footprint": (speed * t[i] + 0.12, ground, 0.0), "t_ref": 2 * np.hypot(ground, alt) / C}
            m = core.DelayMeasurement()
            for k, v in md.items():
                setattr(m, k, v)
            m.delay = eq4_delay(nav, md, y_true, z_true)
            measurements.append((m, md))
    return pings, nav, measurements, y_true, z_true


def test_eq4_attitude_and_lever_arms_match_an_independent_model():
    """The solver's Omega (rotation of Eq. 2 applied to each ping's lever arms) against NumPy."""
    np.testing.assert_allclose(rotation_eq2(0.3, -0.2, 1.1), np.asarray(core.rotation_ned(0.3, -0.2, 1.1)),
                               atol=1e-15)
    pings, nav, meas, y_true, z_true = attitude_problem(n=6)
    rng = np.random.default_rng(2)
    y = (y_true + 0.01 * rng.standard_normal(y_true.size)).tolist()
    z = (z_true + 0.01 * rng.standard_normal(z_true.size)).tolist()
    for m, md in meas:
        # Two ~50 ms paths differ by ~1e-6 s; double rounding leaves ~1e-17 s.
        assert core.predict_delay(pings, m, y, z, C) == pytest.approx(eq4_delay(nav, md, y, z), abs=1e-15)


def test_solver_recovers_sway_and_heave_under_attitude_and_lever_arms():
    pings, _, meas, y_true, z_true = attitude_problem()
    meas = [m for m, _ in meas]
    sol = core.solve_motion(pings, meas, options(lambda_smooth=0.0, lambda_dvl=1.0))
    assert sol.converged, sol.report
    assert np.abs(np.array(sol.y) - y_true).max() < 1e-5
    assert np.abs(np.array(sol.z) - z_true).max() < 1e-5
    # The same delays solved as if the platform were level: the attitude is not ignorable,
    # so the test above is not passing for want of attitude in the data.
    for p in pings:
        p.rpy = (0.0, 0.0, 0.0)
    level = core.solve_motion(pings, meas, options(lambda_smooth=0.0, lambda_dvl=1.0))
    assert np.abs(np.array(level.y) - y_true).max() > WAVELENGTH / 4


def test_solver_rejects_bad_input():
    pings, meas, _, _ = synthetic_problem(n=4)
    with pytest.raises(ValueError):
        core.solve_motion(pings[:2], meas, options())
    meas[0].ping = 99
    with pytest.raises(ValueError):
        core.solve_motion(pings, meas, options())


# --- end to end on simulated data --------------------------------------------------------------

def collect(sway_rms=0.01, heave_rms=0.004, n_pings=40, seed=1, with_target=True, attitude_rms=None,
            tx_lever=None, array_origin=None, bathymetry=None):
    """The simulated scenario. Opt-in extras: `attitude_rms` (roll, pitch, yaw, rad: smooth
    random attitude about a non-zero mean of the same size, reported exactly by the INS),
    lever arms (`tx_lever`, `array_origin`, body frame, m), and `bathymetry` z(x, y) for the
    scatterers (the DVL then reports the altitude above it under the vehicle)."""
    rng = np.random.default_rng(seed)
    overrides = {
        "receive": {"kind": "baseband", "sample_rate": 100e3, "window": {"range": [36.0, 52.0]}},
        "transmit": {"waveform": {"type": "gaussian", "f_start": 180e3, "f_stop": 220e3,
                                  "duration": 1e-4, "window_alpha": 0.0}}}
    if tx_lever is not None:
        overrides["transmit"]["lever_arm"] = list(tx_lever)
    if array_origin is not None:
        overrides["array"] = {"origin": list(array_origin)}
    sonar = SonarModel.load("sas_auv", overrides)
    track = 0.24 * n_pings
    pts, amp = seafloor_patch((-4.0, track + 4.0), (28.0, 52.0), 25.0, rng=rng)
    if bathymetry is not None:
        pts = np.array(pts, dtype=float)
        pts[:, 2] = bathymetry(pts[:, 0], pts[:, 1])
    scene = Scene().add_points(pts, amp)
    target = np.array([track / 2, 41.0, 0.0])
    if bathymetry is not None:
        target[2] = bathymetry(target[0], target[1])
    if with_target:
        scene.add_points([target], amplitude=60.0)       # a bright point to measure focus on
    attitude = None
    if attitude_rms is not None:
        from gpusonar.sim.simulator import smooth_disturbance
        arng = np.random.default_rng(seed + 1000)        # its own generator: the scene and noise stay the same
        attitude = np.stack([smooth_disturbance(n_pings, s, 8.0, arng) + s for s in attitude_rms], axis=1)
    col = simulate_collection(sonar, scene, n_pings=n_pings, interval=0.2, speed=1.2,
                              sway_rms=sway_rms, heave_rms=heave_rms, dvl_altitude_noise=1e-3,
                              environment={"sound_speed": C}, absorption_model="none", adc=False, rng=rng,
                              attitude=attitude)
    if bathymetry is not None:
        drng = np.random.default_rng(seed + 2000)
        for r, p in zip(col.records, col.true_positions):
            nav = r.navigation
            dvl = nav.dvl
            dvl[0].altitude = float(bathymetry(p[0], p[1]) - p[2] + 1e-3 * drng.standard_normal())
            nav.dvl = dvl
            r.navigation = nav
    recon.prepare(col.records)
    return col, target


@pytest.fixture(scope="module")
def run():
    col, target = collect()
    result = motion.estimate_motion(col.records)
    return col, target, result


def test_overlapping_phase_centres_are_found(run):
    col, _, _ = run
    # 0.24 m advance against 0.02 m phase-centre spacing: channel k + 12 meets channel k.
    assert motion.phase_centre_shift(col.records[0], col.records[1]) == 12


# Seeds of the acceptance scenario (scene, disturbances, DVL and receiver noise). One seed hid
# sonar-jh8: seed 1's first DVL sample is close to the truth, seed 2's is 2.5 mm off.
ACCEPTANCE_SEEDS = (1, 2, 3, 4, 5, 6, 7, 8)


@pytest.mark.parametrize("seed", ACCEPTANCE_SEEDS)
def test_sway_and_heave_errors_are_a_small_fraction_of_a_wavelength(seed):
    col, _ = collect(seed=seed)
    result = motion.estimate_motion(col.records)
    truth = col.true_positions[:len(col.records)]
    sway_err = result.y - truth[:, 1]
    heave_err = result.z - truth[:, 2]
    # What navigation alone had: the full disturbance, more than a wavelength of sway.
    assert np.abs(truth[:, 1]).max() > WAVELENGTH
    # Stated bounds, on every seed: sway within lambda / 20 RMS, heave within lambda / 4 RMS.
    # Both are ours (milestone 8); ASASIN (arXiv:2101.05888v2, Sec. II-D) reports no sway or heave
    # accuracy and no lambda1, lambda2, so there is no paper figure to hold them to.
    assert np.sqrt(np.mean(sway_err**2)) < WAVELENGTH / 20, np.sqrt(np.mean(sway_err**2))
    assert np.sqrt(np.mean(heave_err**2)) < WAVELENGTH / 4, np.sqrt(np.mean(heave_err**2))
    assert result.fine.converged and result.coarse.converged
    assert result.n_measurements > 500


def test_fine_pass_improves_on_the_coarse_pass(run):
    col, _, result = run
    truth = col.true_positions[:len(col.records) + 1]
    coarse_rms = np.sqrt(np.mean((np.array(result.coarse.y) - truth[:, 1]) ** 2))
    fine_rms = np.sqrt(np.mean((np.array(result.fine.y) - truth[:, 1]) ** 2))
    assert fine_rms < 0.5 * coarse_rms


def psf_peak(records, target):
    x = target[0] + np.arange(-0.15, 0.1501, 0.002)
    y = target[1] + np.arange(-0.15, 0.1501, 0.004)
    X, Y = np.meshgrid(x, y, indexing="ij")
    pts = np.stack([X, Y, np.zeros_like(X)], axis=-1)
    img = np.abs(recon.sas_image(records, pts, interp="linear", cull_block=0))
    return img.max(), img


def test_focus_approaches_the_true_motion_image(run):
    col, target, result = run
    n = len(col.records)
    def with_track(positions, velocities):
        recs = [r.copy() for r in col.records]
        for r, p, v in zip(recs, positions, velocities):
            geo = r.geometry
            pose = geo.rx_pose
            pose.position, pose.velocity = tuple(p), tuple(v)
            geo.rx_pose = geo.tx_pose = pose
            r.geometry = geo
        return recs
    v_true = np.diff(col.true_positions, axis=0) / 0.2
    ideal_peak, _ = psf_peak(with_track(col.true_positions[:n], v_true), target)
    nav_peak, _ = psf_peak(col.records, target)                     # navigation only
    estimated = [r.copy() for r in col.records]
    motion.apply_motion(estimated, result)
    est_peak, _ = psf_peak(estimated, target)
    loss_nav_db = 20 * np.log10(nav_peak / ideal_peak)
    loss_est_db = 20 * np.log10(est_peak / ideal_peak)
    assert loss_nav_db < -6.0            # uncorrected, the target is badly defocused
    assert loss_est_db > -1.0            # corrected, within 1 dB of the true-motion peak


def test_flat_seafloor_assumption_can_be_replaced_by_known_bathymetry():
    col, _ = collect(n_pings=12, with_target=False, seed=4)
    flat = motion.estimate_motion(col.records)
    opts = motion.MotionOptions(bathymetry=lambda x, y: 0.0)       # the same seafloor, supplied as a function
    known = motion.estimate_motion(col.records, opts)
    np.testing.assert_allclose(known.y, flat.y, atol=1e-6)


LEVERS = {"tx_lever": (0.10, 0.05, -0.20), "array_origin": (-0.30, 0.08, -0.15)}   # body frame, m (ours)


def errors(col, result):
    truth = col.true_positions[:len(col.records)]
    return result.y - truth[:, 1], result.z - truth[:, 2]


def test_attitude_and_lever_arms_leave_sway_and_heave_unbiased(run):
    """Roll, pitch and yaw (2, 1, 1 degrees RMS about means of the same size) and lever arms
    on the transmitter and the array: the INS attitude enters Eq. 4 through Omega, so the
    estimate keeps the level-platform bounds and does not pick up a bias."""
    base_col, _, base = run
    col, _ = collect(attitude_rms=ATTITUDE_RMS, **LEVERS)
    result = motion.estimate_motion(col.records)
    sway_err, heave_err = errors(col, result)
    assert result.fine.converged and result.n_measurements > 500
    # The bounds of test_sway_and_heave_errors_are_a_small_fraction_of_a_wavelength.
    assert np.sqrt(np.mean(sway_err**2)) < WAVELENGTH / 20, np.sqrt(np.mean(sway_err**2))
    assert np.sqrt(np.mean(heave_err**2)) < WAVELENGTH / 4, np.sqrt(np.mean(heave_err**2))
    # Unbiased: the mean error moves by less than lambda / 50 from the level case's (same
    # scene, disturbances and noise). The level case itself has a heave bias of about 0.3 mm.
    base_sway, base_heave = errors(base_col, base)
    assert abs(sway_err.mean()) < WAVELENGTH / 50, sway_err.mean()
    assert abs(heave_err.mean() - base_heave.mean()) < WAVELENGTH / 50, (heave_err.mean(), base_heave.mean())
    # Control: the same records with the attitude withheld from the estimator. Sway is then
    # off by several wavelengths (about 36 mm RMS here), so the bounds above are not met by accident.
    level = [r.copy() for r in col.records]
    for r in level:
        geo = r.geometry
        pose = geo.rx_pose
        pose.rpy = (0.0, 0.0, 0.0)
        geo.rx_pose = geo.tx_pose = pose
        r.geometry = geo
    ignored = motion.estimate_motion(level)
    assert np.sqrt(np.mean(errors(col, ignored)[0] ** 2)) > WAVELENGTH


# --- known bathymetry (Eq. 5: p_s,z from a known seafloor) ---------------------------------------

SLOPE = lambda x, y: 0.1 * np.asarray(y)                         # 1 in 10 down to starboard (ours)
UNDULATING = lambda x, y: (0.05 * np.asarray(y) + 0.3 * np.sin(2 * np.pi * np.asarray(x) / 4.0)
                           + 0.4 * np.sin(2 * np.pi * np.asarray(y) / 9.0))   # ours: 0.3 m under the track


def test_dvl_altitude_is_referred_to_a_seafloor_away_from_z_zero():
    """A flat seabed at z = 5 m, stated as seafloor_z: the DVL altitude is measured from it,
    so heave must come out near the truth (before, Eq. 7 took the seabed at z = 0)."""
    col, _ = collect(n_pings=24, with_target=False, seed=6, bathymetry=lambda x, y: 5.0 + 0.0 * np.asarray(y))
    result = motion.estimate_motion(col.records, motion.MotionOptions(seafloor_z=5.0))
    sway_err, heave_err = errors(col, result)
    assert np.sqrt(np.mean(heave_err**2)) < WAVELENGTH / 4, np.sqrt(np.mean(heave_err**2))
    assert np.sqrt(np.mean(sway_err**2)) < WAVELENGTH / 20, np.sqrt(np.mean(sway_err**2))


def test_seafloor_z_auto_is_a_datum_it_shifts_heave_by_a_common_offset_only():
    """seafloor_z="auto" (mean of z_nav + DVL altitude) is a datum: the ping-0 anchor and
    Eq. 7 both use it, so it may move the whole heave track by its own error but must not
    change the relative motion. True seabed at z = 0 here; the auto estimate absorbs the
    mean heave that navigation does not know (0.63 mm on this seed)."""
    col, _ = collect(n_pings=24, with_target=False, seed=6)
    opts = motion.MotionOptions(seafloor_z="auto")
    floor = motion.resolve_seafloor_z(col.records, opts)
    auto = motion.estimate_motion([r.copy() for r in col.records], opts)
    zero = motion.estimate_motion([r.copy() for r in col.records])
    # The datum moves heave by exactly its value and leaves sway alone. (Before Eq. 7 was
    # referred to the seabed, a datum away from 0 put a step between ping 0 and the rest:
    # sway 0.054 mm RMS here, but 0.33 mm for -0.63 mm and 1.37 mm for +5 mm.)
    # Within 0.1 micrometre: the footprints sit 0.63 mm lower too, which moves the geometry a little (4.5e-8 m seen).
    np.testing.assert_allclose(auto.y, zero.y, rtol=0, atol=1e-7)
    np.testing.assert_allclose(auto.z - zero.z, floor, rtol=0, atol=1e-7)
    sway_err, heave_err = errors(col, auto)
    assert np.sqrt(np.mean(sway_err**2)) < WAVELENGTH / 20
    assert np.sqrt(np.mean((heave_err - heave_err.mean()) ** 2)) < WAVELENGTH / 4
    # The common offset is the datum's error against the true seabed at z = 0 (stated: within 0.5 mm).
    assert abs(heave_err.mean() - floor) < 5e-4, (heave_err.mean(), floor)


@pytest.mark.parametrize("name,surface", [("slope", SLOPE), ("undulating", UNDULATING)])
def test_known_bathymetry_keeps_the_flat_seafloor_bounds_on_a_non_flat_seafloor(name, surface):
    """Scatterers on a non-flat seafloor; the DVL reports the altitude above it under the
    vehicle. Given the surface, the estimator meets the flat-seafloor bounds (sway lambda/20,
    heave lambda/4 RMS). The simulated DVL (collect) is a single vertical beam, the same
    model _refer_dvl assumes, so this test cannot catch a wrong DVL geometry (four slanted
    beams, a footprint average)."""
    col, _ = collect(bathymetry=surface)
    known = motion.estimate_motion([r.copy() for r in col.records], motion.MotionOptions(bathymetry=surface))
    sway_err, heave_err = errors(col, known)
    assert known.fine.converged
    assert np.sqrt(np.mean(sway_err**2)) < WAVELENGTH / 20, np.sqrt(np.mean(sway_err**2))
    assert np.sqrt(np.mean(heave_err**2)) < WAVELENGTH / 4, np.sqrt(np.mean(heave_err**2))
    # The flat assumption on the same records does worse (on the undulating seafloor it
    # misreads the DVL by the 0.3 m relief under the track).
    flat = motion.estimate_motion([r.copy() for r in col.records])
    flat_sway = np.sqrt(np.mean(errors(col, flat)[0] ** 2))
    assert flat_sway > np.sqrt(np.mean(sway_err**2))


def test_simulated_attitude_must_be_one_row_per_ping():
    sonar = SonarModel.load("sas_auv")
    with pytest.raises(ValueError, match=r"\(4, 3\)"):
        simulate_collection(sonar, Scene(), n_pings=4, interval=0.2, attitude=np.zeros(12))   # flat: refused
