"""Milestone 4: the backprojection kernel.

- Delay and phase precision against extended-precision arithmetic (DECISIONS.md D-06).
- Agreement with an independent CPU reference, for every interpolation mode.
- GPU against the CPU reference (skipped without a device).
"""

import numpy as np
import pytest

import gpusonar as gs
from gpusonar.sim import Scene, make_pose, simulate_ping
from gpusonar.sonar import SonarModel

import reference_bp as ref

LD = np.longdouble


def small(name, r0, r1, **extra):
    over = {"receive": {"window": {"range": [r0, r1]}}}
    return SonarModel.load(name, gs.config.deep_merge(over, extra) if extra else over)


def phase_error_deg(record, points, stop_and_hop=False):
    """Worst kernel phase error against long double, over all (point, element) pairs.

    The phase that matters is the total carrier phase the kernel applies: the phase
    rotation plus the phase slope of the interpolation position (a sample-position
    error of ds samples shifts the baseband signal's phase by 2 pi f_bb ds / fs,
    which is second order; the rotation term carries the full carrier).
    """
    pos, cyc = gs.core.debug_delays([record], points, stop_and_hop=stop_and_hop)
    fs, t0, fc = record.samples.sample_rate, record.samples.start_time, record.samples.center_frequency
    worst_phase = 0.0
    worst_pos = 0.0
    for ip, pt in enumerate(points):
        for ch in range(record.samples.n_channels):
            tau = ref.exact_delay(record, ch, pt, stop_and_hop)
            exact_cyc = LD(fc) * tau
            exact_frac = float(exact_cyc - np.floor(exact_cyc))
            d = (cyc[ip, ch] - exact_frac + 0.5) % 1.0 - 0.5
            worst_phase = max(worst_phase, abs(d) * 360.0)
            exact_pos = float((tau - LD(t0)) * LD(fs))
            worst_pos = max(worst_pos, abs(pos[ip, ch] - exact_pos))
    return worst_phase, worst_pos


def empty_baseband_record(sonar, pose, env):
    rec = sonar.make_record(pose, environment=env)
    return rec


def test_phase_error_under_one_degree_at_10_km():
    """Flank array, 10 kHz carrier, targets out to 10 km, platform moving and yawing."""
    sonar = SonarModel.load("flank_32", {"receive": {"kind": "baseband"},
                                          "transmit": {"waveform": {"f_start": 10e3, "f_stop": 10e3,
                                                                    "type": "cw", "duration": 0.01}}})
    pose = make_pose(position=(1234.5, -987.6, 30.0), rpy=(0.02, -0.01, 0.7),
                     velocity=(1.4, 1.2, 0.05), rates=(0.01, 0.005, 0.02))
    rec = empty_baseband_record(sonar, pose, {"sound_speed": 1503.7})
    assert rec.samples.center_frequency == 10e3
    rng = np.random.default_rng(0)
    r = rng.uniform(200.0, 10000.0, 40)
    b = rng.uniform(-1.2, 1.2, 40)
    pts = np.array(pose.position) + np.column_stack([r * np.cos(b + 0.7 + np.pi / 2), r * np.sin(b + 0.7 + np.pi / 2), rng.uniform(-20, 60, 40)])
    worst_phase, worst_pos = phase_error_deg(rec, pts)
    assert worst_phase < 1.0, worst_phase
    assert worst_phase < 0.05          # the split leaves far more margin than required
    assert worst_pos < 1e-3            # samples


def test_phase_error_under_one_degree_at_5_mhz():
    """Linear probe, 5 MHz carrier, 0.75-3 m, both mux halves, slow platform motion."""
    sonar = small("l12_5d", 0.75, 2.95, receive={"kind": "baseband"})
    pose = make_pose(position=(0.3, -0.2, 0.5), rpy=(0.01, 0.02, -0.4),
                     velocity=(0.05, 0.02, 0.0), rates=(0.0, 0.0, 0.05))
    rng = np.random.default_rng(1)
    r = rng.uniform(0.75, 3.0, 30)
    b = rng.uniform(-0.4, 0.4, 30)
    pts = np.array(pose.position) + np.column_stack([r * np.cos(b - 0.4), r * np.sin(b - 0.4), rng.uniform(-0.05, 0.05, 30)])
    for sub in (0, 1):
        rec = sonar.make_record(pose, sub_sequence=sub, environment={"sound_speed": 1481.3})
        assert rec.samples.center_frequency == 5e6
        worst_phase, worst_pos = phase_error_deg(rec, pts)
        assert worst_phase < 1.0, worst_phase
        assert worst_phase < 0.05
        assert worst_pos < 1e-3


def test_plain_float32_would_fail_the_same_budget():
    """Why the split exists: a float32 range at 10 km cannot hold 1 degree at 10 kHz."""
    c, f = 1500.0, 10e3
    r = np.float32(9876.543)
    step = np.spacing(r)                      # float32 resolution at this range
    assert step * f / c * 360.0 > 1.0         # one step alone exceeds a degree two-way-free


def test_stop_and_hop_differs_from_continuous_motion_by_the_expected_amount():
    sonar = SonarModel.load("flank_32", {"receive": {"kind": "baseband"}})
    v = np.array([0.0, 2.0, 0.0])            # closing on a broadside target at 2 m/s
    rec = sonar.make_record(make_pose(velocity=v), environment={"sound_speed": 1500.0})
    pt = np.array([[0.0, 5000.0, 0.0]])
    pos_c, _ = gs.core.debug_delays([rec], pt, stop_and_hop=False)
    pos_s, _ = gs.core.debug_delays([rec], pt, stop_and_hop=True)
    # The receiver closes by v * tau during the 6.7 s flight: about 13 m of path.
    d_samples = (pos_s - pos_c)[0, 16]
    expected = (2.0 * (2 * 5000.0 / 1500.0) / 1500.0) * rec.samples.sample_rate
    assert d_samples == pytest.approx(expected, rel=2e-3)


# --- kernel against the independent reference ----------------------------------------------

def simulated_fls(name="p4_2v", n_targets=5, seed=2, velocity=(0.0, 0.0, 0.0)):
    sonar = small(name, 0.05, 0.09, receive={"kind": "baseband"})
    rng = np.random.default_rng(seed)
    pts = np.column_stack([rng.uniform(0.06, 0.08, n_targets), rng.uniform(-0.01, 0.01, n_targets),
                           np.zeros(n_targets)])
    scene = Scene().add_points(pts, amplitude=rng.uniform(0.5, 1.0, n_targets))
    rec = simulate_ping(sonar, make_pose(velocity=velocity), scene,
                        environment={"sound_speed": 1480.0}, absorption_model="none", adc=False)
    gs.core.pulse_compress(rec)      # echoes now peak at their delay
    return rec, pts


@pytest.mark.parametrize("interp", ["nearest", "linear", "sinc"])
def test_kernel_matches_reference(interp):
    rec, targets = simulated_fls()
    rng = np.random.default_rng(7)
    # Pixels on and around the targets, so the comparison covers strong and weak values.
    pix = np.vstack([targets, targets + rng.normal(0, 3e-4, targets.shape),
                     np.column_stack([rng.uniform(0.06, 0.08, 6), rng.uniform(-0.01, 0.01, 6), np.zeros(6)])])
    got = gs.core.backproject([rec], pix, interp=interp)
    want = ref.backproject([rec], pix, interp=interp)
    scale = np.abs(want).max()
    assert scale > 0
    # float32 accumulation over 64 elements, table interpolation for sinc.
    tol = {"nearest": 2e-5, "linear": 2e-5, "sinc": 2e-4}[interp]
    assert np.abs(got - want).max() / scale < tol


def test_kernel_matches_reference_with_motion_and_weights():
    rec, targets = simulated_fls(velocity=(0.5, 3.0, 0.0), seed=4)
    weights = [np.linspace(0.2, 1.0, rec.samples.n_channels).astype(np.float32).tolist()]
    got = gs.core.backproject([rec], targets, interp="sinc", weights=weights)
    want = ref.backproject([rec], targets, interp="sinc", weights=weights)
    assert np.abs(got - want).max() / np.abs(want).max() < 2e-4
    # And stop-and-hop is a genuinely different answer at this speed.
    hop = gs.core.backproject([rec], targets, interp="sinc", weights=weights, stop_and_hop=True)
    assert np.abs(hop - want).max() / np.abs(want).max() > 1e-2


def test_image_peaks_at_the_targets_with_coherent_gain():
    rec, targets = simulated_fls(n_targets=1, seed=9)
    img = gs.core.backproject([rec], targets, interp="sinc")
    # Coherent sum over N elements: every channel contributes its compressed peak, in phase.
    x = rec.samples.array
    expected = np.abs(x).max(axis=1).sum()
    assert np.abs(img[0]) == pytest.approx(expected, rel=1e-2)
    # Off the target by a quarter wavelength in range the sum has decorrelated noticeably.
    off = targets + np.array([[1480.0 / 2.72e6 / 4, 0.0, 0.0]])
    assert np.abs(gs.core.backproject([rec], off)[0]) < 0.95 * np.abs(img[0])


def test_real_passband_input_is_rejected_with_a_clear_message():
    sonar = small("p4_2v", 0.05, 0.06)
    rec = sonar.make_record(make_pose())
    with pytest.raises(ValueError, match="to_baseband"):
        gs.core.backproject([rec], np.array([[0.055, 0.0, 0.0]]))


def test_weight_sum_counts_contributing_elements():
    rec, targets = simulated_fls(n_targets=1)
    img, wsum = gs.core.backproject([rec], targets, return_weight_sum=True)
    assert wsum[0] == pytest.approx(rec.samples.n_channels)
    far = np.array([[5.0, 0.0, 0.0]])          # outside the receive window: nothing contributes
    img, wsum = gs.core.backproject([rec], far, return_weight_sum=True)
    assert wsum[0] == 0 and img[0] == 0


@pytest.mark.skipif(not gs.cuda_available(), reason="no CUDA device: GPU kernel not run")
@pytest.mark.parametrize("interp", ["nearest", "linear", "sinc"])
def test_gpu_kernel_matches_reference(interp):
    rec, targets = simulated_fls()
    got = gs.core.backproject([rec], targets, interp=interp, backend=gs.core.Backend.Cuda)
    want = ref.backproject([rec], targets, interp=interp)
    assert np.abs(got - want).max() / np.abs(want).max() < 2e-4
    host = gs.core.backproject([rec], targets, interp=interp)
    assert np.abs(got - host).max() / np.abs(want).max() < 1e-5


# --- sessions: a collection prepared once and imaged many times ------------------------------

def test_session_matches_the_one_shot_call():
    from gpusonar.sim import run_scenario
    from gpusonar import recon as grecon
    records = run_scenario("point_grid_fls_300k").records
    grecon.prepare(records)
    rng = np.random.default_rng(0)
    pts = np.column_stack([rng.uniform(-3, 3, 400), rng.uniform(2, 12, 400), rng.uniform(-0.5, 0.5, 400)])
    one_shot = gs.core.backproject(records, pts, fov_factor=1.5)
    session = gs.core.BackprojectSession(records)
    first = session.backproject(pts, fov_factor=1.5)
    second = session.backproject(pts[:100], fov_factor=1.5)
    np.testing.assert_array_equal(first, one_shot)
    np.testing.assert_array_equal(second, one_shot[:100])
    assert session.resident_bytes == 0                      # host backend keeps nothing on a device
    with pytest.raises(ValueError, match="trajectory"):
        session.backproject(pts, position_lookup="software_table")


@pytest.mark.skipif(not gs.core.cuda_available(), reason="no CUDA device")
def test_session_keeps_samples_on_the_device():
    from gpusonar.sim import run_scenario
    from gpusonar import recon as grecon
    records = run_scenario("point_grid_fls_300k").records
    grecon.prepare(records)
    rng = np.random.default_rng(1)
    pts = np.column_stack([rng.uniform(-3, 3, 2000), rng.uniform(2, 12, 2000), np.zeros(2000)])
    one_shot = gs.core.backproject(records, pts, fov_factor=1.5, backend=gs.core.Backend.Cuda)
    session = gs.core.BackprojectSession(records)
    a = session.backproject(pts, fov_factor=1.5, backend=gs.core.Backend.Cuda)
    held = session.resident_bytes
    n_samples = sum(r.samples.array.size for r in records)
    assert held >= n_samples * 8                            # complex64 samples are resident
    b = session.backproject(pts, fov_factor=1.5, backend=gs.core.Backend.Cuda)
    assert session.resident_bytes == held                   # second call uploads nothing more
    np.testing.assert_array_equal(a, one_shot)
    np.testing.assert_array_equal(b, one_shot)
