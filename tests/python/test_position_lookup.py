"""Milestone 5: sensor positions from a tabulated trajectory (ASASIN Sec. II-C, D-07).

The record's pose is first order in time. When the platform accelerates within a
ping, the true position departs from that model; the residual is tabulated from the
INS samples and looked up during backprojection, in software or (on a GPU) from
texture memory with hardware interpolation.
"""

import numpy as np
import pytest

import gpusonar as gs
from gpusonar import recon
from gpusonar.sim import Scene, make_pose, simulate_ping
from gpusonar.sonar import SonarModel

C = 1500.0
TILT = np.radians(20.0)
TARGET = 120.0 * np.array([0.0, np.cos(TILT), np.sin(TILT)])     # slant range 120 m, on boresight


def ping(acceleration, ins_rate=100.0):
    # A short Gaussian pulse instead of the config's LFM: an accelerating array has a
    # closing speed at arrival, and LFM turns that Doppler into a range and phase shift
    # of its own (about 1 mm and 7 degrees here), which would mask what is being tested.
    sonar = SonarModel.load("sas_auv", {
        "receive": {"window": {"range": [115.0, 125.0]}},
        "transmit": {"waveform": {"type": "gaussian", "f_start": 180e3, "f_stop": 220e3,
                                  "duration": 1.0e-4, "window_alpha": 0.0}}})
    rec = simulate_ping(sonar, make_pose(velocity=(1.6, 0.0, 0.0)), Scene().add_points([TARGET]),
                        environment={"sound_speed": C}, absorption_model="none", adc=False,
                        acceleration=acceleration, ins_rate=ins_rate)
    recon.prepare([rec])
    return rec


def image_line(rec, **kw):
    y = TARGET[1] + np.linspace(-0.05, 0.05, 201)
    pts = np.column_stack([np.zeros_like(y), y, y * np.tan(TILT)])
    return y, gs.core.backproject([rec], pts, interp="sinc", **kw)


def test_no_residual_means_no_change():
    """With a trajectory that is exactly first order, every lookup mode gives the same image."""
    rec = ping((0.0, 0.0, 0.0))
    _, base = image_line(rec)
    for mode in ("software_table", "texture"):
        _, img = image_line(rec, position_lookup=mode)
        np.testing.assert_array_equal(img, base)


def test_table_lookup_restores_focus_under_acceleration():
    # Sway acceleration of 1 m/s^2: by the 0.17 s arrival time the array has moved
    # 15 mm sideways, two wavelengths at 200 kHz, beyond what the pose model knows.
    accel = (0.0, 1.0, 0.0)
    rec = ping(accel)
    still = ping((0.0, 0.0, 0.0))
    y, ideal = image_line(still)
    _, analytic = image_line(rec)
    _, table = image_line(rec, position_lookup="software_table")
    i_ideal = np.argmax(np.abs(ideal))
    # Without the table the target is imaged in the wrong place...
    shift = y[np.argmax(np.abs(analytic))] - y[i_ideal]
    tau = 2 * np.linalg.norm(TARGET) / C
    # Only the receive leg is affected (the array was still in place at transmit), so the
    # apparent slant range moves by half the along-ray displacement.
    expected = -0.25 * 1.0 * tau**2 * np.cos(TILT) ** 2
    assert shift == pytest.approx(expected, rel=0.2)
    assert abs(shift) > 0.5 * C / 200e3            # most of a wavelength: a real defocus
    # ...and with it the image matches the unaccelerated one in position, level and phase.
    # (The range response is 19 mm wide, so its sampled maximum is only good to a millimetre
    # or two; the level and phase checks below are the sharp ones.)
    assert abs(y[np.argmax(np.abs(table))] - y[i_ideal]) <= 2e-3
    assert np.abs(table).max() == pytest.approx(np.abs(ideal).max(), rel=1e-3)
    err = np.abs(table - ideal).max() / np.abs(ideal).max()
    assert err < 0.02, err


def test_texture_arithmetic_stays_inside_the_phase_budget():
    """8-bit interpolation weights (hardware texture filtering, emulated on the host here)
    must not cost more than a small fraction of a degree, because only the residual is
    stored in the table."""
    rec = ping((0.0, 1.0, 0.3))
    _, soft = image_line(rec, position_lookup="software_table")
    _, tex = image_line(rec, position_lookup="texture")
    peak = np.abs(soft).max()
    i = np.argmax(np.abs(soft))
    phase_deg = np.degrees(np.angle(tex[i] * np.conj(soft[i])))
    assert abs(phase_deg) < 0.5
    assert np.abs(tex - soft).max() / peak < 0.01
    # The two are not bit-identical: the emulation really does quantise the weights.
    assert np.any(tex != soft)


def test_coarser_table_is_worse_and_upsampling_helps():
    rec = ping((0.0, 1.0, 0.0), ins_rate=10.0)       # only 10 Hz navigation
    still = ping((0.0, 0.0, 0.0))
    _, ideal = image_line(still)
    peak = np.abs(ideal).max()
    errs = {}
    for up in (1, 8):
        _, img = image_line(rec, position_lookup="software_table", trajectory_upsample=up)
        errs[up] = np.abs(img - ideal).max() / peak
    # Linear interpolation of a parabola between 10 Hz samples leaves a visible error;
    # cubic upsampling before the lookup removes most of it.
    assert errs[8] < 0.3 * errs[1]


def test_non_uniform_ins_is_rejected():
    rec = ping((0.0, 1.0, 0.0))
    nav = rec.navigation
    ins = nav.ins
    ins[3].t += 1e-3
    nav.ins = ins
    rec.navigation = nav
    with pytest.raises(ValueError, match="uniformly spaced"):
        image_line(rec, position_lookup="software_table")


@pytest.mark.skipif(not gs.cuda_available(), reason="no CUDA device: hardware texture lookup not run")
def test_hardware_texture_matches_software_table():
    rec = ping((0.0, 1.0, 0.3))
    _, soft = image_line(rec, position_lookup="software_table", backend=gs.core.Backend.Cuda)
    _, tex = image_line(rec, position_lookup="texture", backend=gs.core.Backend.Cuda)
    assert np.abs(tex - soft).max() / np.abs(soft).max() < 0.01
