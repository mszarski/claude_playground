"""Milestone 10 (part): array calibration and the multi-board sync stage.

Acceptance: calibration recovers injected channel errors; the sync stage recovers
injected clock offsets and drifts on `popoto_4x8`, and the 32-channel beam pattern then
matches a single-clock array within 1 dB.
"""

import numpy as np
import pytest

import gpusonar as gs
from gpusonar import calibrate, flank, recon, sync
from gpusonar.sim import Scene, make_pose, simulate_ping
from gpusonar.sonar import SonarModel

core = gs.core
C = 1500.0


# --- calibration ---------------------------------------------------------------------------------

def calibration_pings(errors, bearings_deg=(-40.0, -20.0, 0.0, 20.0, 40.0), range_m=1000.0):
    sonar = SonarModel.load("flank_32", {"receive": {"window": {"range": [range_m - 40, range_m + 40]}},
                                          "transmit": {"waveform": {"duration": 0.05}}})
    records, sources = [], []
    for b in np.radians(bearings_deg):
        src = range_m * np.array([np.sin(b), np.cos(b), 0.0])
        records.append(simulate_ping(sonar, make_pose(), Scene().add_points([src], target_strength_db=20.0),
                                     environment={"sound_speed": C}, absorption_model="none", adc=False,
                                     channel_errors=errors))
        sources.append(src)
    return sonar, records, sources


@pytest.fixture(scope="module")
def injected():
    rng = np.random.default_rng(11)
    n = 32
    position = np.zeros((n, 3))
    position[:, :2] = rng.normal(0.0, 5e-3, (n, 2))             # 5 mm RMS in the array plane
    return {"gain": 10 ** (rng.normal(0.0, 1.0, n) / 20),       # 1 dB RMS
            "phase": np.radians(rng.normal(0.0, 20.0, n)),      # 20 degrees RMS
            "position": position}


def test_calibration_recovers_injected_gain_phase_and_position(injected):
    sonar, records, sources = calibration_pings(injected)
    est = calibrate.estimate_channel_errors(records, sources)
    # Compare in the same gauge the estimator reports: mean gain 1, mean phase 0, mean offset 0.
    gain = injected["gain"] / injected["gain"].mean()
    phase = injected["phase"] - injected["phase"].mean()
    position = injected["position"] - injected["position"].mean(axis=0)
    assert np.abs(20 * np.log10(est.gain / gain)).max() < 0.02           # dB, against 1 dB RMS injected
    assert np.degrees(np.abs(est.phase - phase)).max() < 1.0
    assert np.abs(est.position[:, :2] - position[:, :2]).max() < 0.3e-3  # 0.3 mm
    assert est.residual_rms_deg < 0.5


def test_applying_the_calibration_restores_the_beam(injected):
    sonar, records, sources = calibration_pings(injected)
    est = calibrate.estimate_channel_errors(records, sources)
    # A fresh ping on a target that was not one of the calibration sources.
    b = np.radians(11.0)
    target = 1000.0 * np.array([np.sin(b), np.cos(b), 0.0])
    def ping(errors):
        return simulate_ping(sonar, make_pose(), Scene().add_points([target]), environment={"sound_speed": C},
                             absorption_model="none", adc=False, channel_errors=errors)
    bearings = np.radians(np.linspace(-20.0, 40.0, 241))
    def pattern(rec):
        return np.abs(recon.fan_image([rec], [1000.0], bearings).image[0])
    ideal = pattern(ping(None))
    raw = pattern(ping(injected))
    rec = ping(injected)
    recon.prepare([rec])
    calibrate.store_calibration(rec, est)
    core.apply_calibration(rec)
    fixed = pattern(rec)
    assert raw.max() < 0.985 * ideal.max()                        # errors cost coherent gain
    # Calibration gives it back, up to the one thing it cannot see: the array's mean gain.
    common = injected["gain"].mean()
    assert fixed.max() == pytest.approx(ideal.max() * common, rel=0.01)
    # Sidelobes: errors scatter energy into the far sidelobes (the first sidelobe barely
    # moves). Uncalibrated the far floor rises; calibrated most of that goes away. It does
    # not return exactly to the ideal pattern: the elements really are 5 mm out of place,
    # so the calibrated array is a slightly irregular one, correctly steered.
    far = np.abs(np.degrees(bearings) - 11.0) > 15.0
    assert raw[far].mean() > 1.5 * ideal[far].mean()
    assert fixed[far].mean() < 0.75 * raw[far].mean()
    assert fixed[far].mean() < 1.4 * ideal[far].mean() * common


def test_calibration_needs_enough_sources():
    sonar, records, sources = calibration_pings(None, bearings_deg=(0.0, 20.0))
    with pytest.raises(ValueError, match="at least 3"):
        calibrate.estimate_channel_errors(records, sources)


# --- sync ----------------------------------------------------------------------------------------

def four_board_ping(bearing_deg=0.0, single_clock=False, seed=0, range_m=600.0):
    over = {"receive": {"window": {"range": [range_m - 150.0, range_m + 150.0]}}}
    if single_clock:
        over["receive"]["boards"] = [
            {"first": 8 * i, "count": 8, "clock_offset": 0.0, "clock_drift_ppm": 0.0, "reference_channel": 8 * i}
            for i in range(4)]
    sonar = SonarModel.load("popoto_4x8", over)
    b = np.radians(bearing_deg)
    target = range_m * np.array([np.sin(b), np.cos(b), 0.0])
    truth = {}
    rec = simulate_ping(sonar, make_pose(), Scene().add_points([target], target_strength_db=10.0),
                        environment={"sound_speed": C}, absorption_model="none", adc=False,
                        rng=np.random.default_rng(seed), clock_truth=truth)
    return sonar, rec, target, truth


def test_reference_bursts_are_present_only_on_the_reference_inputs():
    sonar, rec, _, _ = four_board_ping()
    x = rec.samples.array
    early = slice(0, int(0.012 * rec.samples.sample_rate))        # before any echo arrives
    for row in range(32):
        assert (np.abs(x[row, early]).max() > 0) == (row in (0, 8, 16, 24))
    assert [b.reference_channel for b in rec.metadata.boards] == [0, 8, 16, 24]
    assert all(b.clock_offset == 0.0 for b in rec.metadata.boards)   # the processor is not told


def test_sync_recovers_injected_clock_offsets_and_drifts():
    sonar, rec, _, truth = four_board_ping()
    ref = sonar.config["receive"]["reference"]
    est = sync.estimate_clocks_reference(rec, ref)
    true_offset = np.array(truth["offset"]) - truth["offset"][0]
    true_drift = np.array(truth["drift_ppm"]) - truth["drift_ppm"][0]
    assert np.abs(true_offset).max() > 100e-6                     # the boards really are far apart
    np.testing.assert_allclose(est.offset, true_offset, atol=0.1e-6)       # 0.1 microsecond
    np.testing.assert_allclose(est.drift_ppm, true_drift, atol=0.3)        # ppm
    assert est.quality.min() > 0.9


def test_synchronised_boards_are_within_tolerance():
    sonar, rec, _, truth = four_board_ping()
    report = sync.synchronise(rec, sonar.config["receive"]["reference"])
    # A twentieth of a period at 35 kHz.
    assert report.tolerance == pytest.approx(1.0 / (20 * 35e3))
    assert report.ok and report.residual.max() < report.tolerance
    assert report.residual.max() < 1.0e-6
    assert all(b.synchronised == 1 for b in rec.metadata.boards)
    assert rec.metadata.boards[1].clock_offset == pytest.approx(truth["offset"][1] - truth["offset"][0], abs=0.1e-6)


def beam_pattern(rec, range_m, bearings):
    return np.abs(recon.fan_image([rec], [range_m], bearings).image[0])


@pytest.mark.parametrize("bearing", [0.0, 14.0])
def test_beam_pattern_after_sync_matches_a_single_clock_array(bearing):
    sonar, rec, target, _ = four_board_ping(bearing)
    _, ideal_rec, _, _ = four_board_ping(bearing, single_clock=True)
    bearings = np.radians(bearing + np.linspace(-12.0, 12.0, 481))
    unsynced = beam_pattern(rec.copy(), 600.0, bearings)
    sync.synchronise(rec, sonar.config["receive"]["reference"])
    synced = beam_pattern(rec, 600.0, bearings)
    ideal = beam_pattern(ideal_rec, 600.0, bearings)
    # Without sync the four sub-arrays do not add up: 310 microseconds is ten carrier periods.
    assert unsynced.max() < 0.6 * ideal.max()
    # With sync: peak and every lobe down to -20 dB within 1 dB of the single-clock array.
    assert 20 * np.log10(synced.max() / ideal.max()) == pytest.approx(0.0, abs=0.2)
    strong = ideal > 0.1 * ideal.max()
    diff_db = 20 * np.log10(synced[strong] / ideal[strong])
    assert np.abs(diff_db).max() < 1.0, np.abs(diff_db).max()


def test_sync_refuses_when_alignment_cannot_be_reached():
    sonar, rec, _, _ = four_board_ping()
    ref = dict(sonar.config["receive"]["reference"])
    # Sabotage: tell the stage the bursts are a different chirp, so its estimate is poor.
    bad = dict(ref, f_start=30e3, f_stop=36e3)
    with pytest.raises(sync.SyncError):
        sync.synchronise(rec, bad)
    # An impossible tolerance is also refused, with the residual reported.
    sonar, rec, _, _ = four_board_ping()
    with pytest.raises(sync.SyncError, match="not beamforming across boards"):
        sync.synchronise(rec, ref, tolerance=1e-12)
    sonar, rec, _, _ = four_board_ping()
    report = sync.synchronise(rec, ref, tolerance=1e-12, enforce=False)
    assert not report.ok and report.residual.max() > 1e-12


def test_pps_method_gives_offsets():
    sonar, rec, _, truth = four_board_ping()
    fs = rec.samples.sample_rate
    meta = rec.metadata
    boards = meta.boards
    nominal = 5000.0
    for b, off in zip(boards, truth["offset"]):
        b.pps_sample = nominal - off * fs          # a late-starting board sees the edge earlier
    meta.boards = boards
    rec.metadata = meta
    est = sync.estimate_clocks_pps(rec, nominal)
    np.testing.assert_allclose(est.offset, np.array(truth["offset"]) - truth["offset"][0], atol=1e-9)
    assert est.method == "pps" and np.all(est.drift_ppm == 0)


def test_direct_path_method_gives_offsets():
    """The transmit pulse reaching the hydrophones directly, in a window that opens at transmit."""
    over = {"receive": {"window": {"range": [0.0, 60.0]}, "reference": None},
            "transmit": {"waveform": {"duration": 0.005}, "lever_arm": [0.0, 2.0, 0.5]}}
    sonar = SonarModel.load("popoto_4x8", over)
    truth = {}
    rec = sonar.make_record(make_pose(), environment={"sound_speed": C})
    # The direct arrival, synthesised as a "scatterer" standing at the projector would be
    # a two-way path; instead place the projector's pulse into each channel directly.
    from gpusonar.sim.simulator import _set_board_clocks
    clocks = _set_board_clocks(rec, sonar, truth=True, rng=np.random.default_rng(0))
    tx = np.array(rec.tx_position())
    rx = rec.rx_positions()
    s = rec.samples
    x = s.array
    w = rec.transmit.waveform
    n = np.arange(s.n_samples)
    for row in range(32):
        b = rec.metadata.channel_board[row]
        t_true = s.start_time + clocks["offset"][b] + n / (s.sample_rate * (1 + clocks["drift_ppm"][b] * 1e-6))
        u = t_true - np.linalg.norm(rx[row] - tx) / C
        inside = (u >= 0) & (u <= w.duration)
        k = (w.f_stop - w.f_start) / w.duration
        x[row] = np.where(inside, np.cos(2 * np.pi * (w.f_start * u + 0.5 * k * u**2)), 0.0)
    s.array = x
    rec.samples = s
    _set_board_clocks(rec, sonar, truth=False)
    est = sync.estimate_clocks_direct_path(rec)
    true_offset = np.array(clocks["offset"]) - clocks["offset"][0]
    np.testing.assert_allclose(est.offset, true_offset, atol=0.5e-6)
