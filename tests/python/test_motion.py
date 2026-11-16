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
    held = core.solve_motion(pings, meas, options(lambda_dvl=10.0, lambda_smooth=0.0))
    drift_free = np.abs(np.array(free.z)[:-1] - z_true[:-1]).max()
    drift_held = np.abs(np.array(held.z)[:-1] - z_true[:-1]).max()
    assert drift_held < drift_free and drift_held < 2e-3


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


def test_solver_rejects_bad_input():
    pings, meas, _, _ = synthetic_problem(n=4)
    with pytest.raises(ValueError):
        core.solve_motion(pings[:2], meas, options())
    meas[0].ping = 99
    with pytest.raises(ValueError):
        core.solve_motion(pings, meas, options())


# --- end to end on simulated data --------------------------------------------------------------

def collect(sway_rms=0.01, heave_rms=0.004, n_pings=40, seed=1, with_target=True):
    rng = np.random.default_rng(seed)
    sonar = SonarModel.load("sas_auv", {
        "receive": {"kind": "baseband", "sample_rate": 100e3, "window": {"range": [36.0, 52.0]}},
        "transmit": {"waveform": {"type": "gaussian", "f_start": 180e3, "f_stop": 220e3,
                                  "duration": 1e-4, "window_alpha": 0.0}}})
    track = 0.24 * n_pings
    pts, amp = seafloor_patch((-4.0, track + 4.0), (28.0, 52.0), 25.0, rng=rng)
    scene = Scene().add_points(pts, amp)
    target = np.array([track / 2, 41.0, 0.0])
    if with_target:
        scene.add_points([target], amplitude=60.0)       # a bright point to measure focus on
    col = simulate_collection(sonar, scene, n_pings=n_pings, interval=0.2, speed=1.2,
                              sway_rms=sway_rms, heave_rms=heave_rms, dvl_altitude_noise=1e-3,
                              environment={"sound_speed": C}, absorption_model="none", adc=False, rng=rng)
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


def test_sway_and_heave_errors_are_a_small_fraction_of_a_wavelength(run):
    col, _, result = run
    truth = col.true_positions[:len(col.records)]
    sway_err = result.y - truth[:, 1]
    heave_err = result.z - truth[:, 2]
    # What navigation alone had: the full disturbance, more than a wavelength of sway.
    assert np.abs(truth[:, 1]).max() > WAVELENGTH
    # Stated bounds: sway within lambda / 20 RMS, heave within lambda / 4 RMS.
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
