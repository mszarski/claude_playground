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



def test_an_unknown_reference_mode_is_refused():
    """Audit A15: a misspelt `receive.reference.mode` used to add no reference, silently."""
    for mode in ("sync-chirp", None):
        with pytest.raises(ValueError, match="receive.reference.mode"):
            SonarModel.load("popoto_4x8", {"receive": {"reference": {"mode": mode}}})
    sonar = SonarModel.load("popoto_4x8")
    sonar.config["receive"]["reference"]["mode"] = "chirp"                     # past the loader: refused where it is used
    with pytest.raises(ValueError, match="receive.reference.mode"):
        simulate_ping(sonar, make_pose(), Scene(), environment={"sound_speed": C}, absorption_model="none", adc=False,
                      rng=np.random.default_rng(0))

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


def _direct_weights(frac, taps=16):
    """The resampler's weights from the formula, a sine and a cosine a tap (sync.py until sonar-fy6): [taps, len(frac)]."""
    half = taps // 2
    d = np.arange(-half + 1, half + 1)[:, None] - np.asarray(frac, dtype=np.float64)[None, :]
    return np.sinc(d) * (0.5 + 0.5 * np.cos(np.pi * d / half))


def _resample_row_by_row(x, positions, taps=16, direct=False):
    """The board resampler's arithmetic until sonar-h1f, one channel row at a time (the reference for the tests below),
    with the weights from the formula (`direct`) or from sync._sinc_weights (three trig calls a sample, sonar-fy6)."""
    base = np.floor(positions).astype(np.int64)
    frac = positions - base
    half = taps // 2
    weights = _direct_weights(frac, taps) if direct else sync._sinc_weights(frac, taps)
    out = np.zeros(len(positions), dtype=np.float64)
    for k in range(-half + 1, half + 1):
        idx = base + k
        wgt = weights[k + half - 1]
        ok = (idx >= 0) & (idx < len(x))
        out[ok] += wgt[ok] * x[idx[ok]]
    return out


# The largest difference between a weight from three trig calls and the formula's (measured 2.5e-16 with numpy, 2.8e-16 in
# libs/live; the weights are at most 1): a weight's last bit or two.
WEIGHT_BOUND = 4.5e-16


def _chain_fractions():
    """Fractions the chain makes: (m / fs - offset) fs (1 + drift) for boards late and early, with and without drift."""
    fs, m = 102400.0, np.arange(0, 200000, 7)
    out = []
    for offset in (3.1e-4, -1.7e-4, 2.1e-3, 1.234567e-5):
        for drift in (0.0, 12e-6, -55.5e-6):
            pos = (m / fs - offset) * fs * (1.0 + drift)
            out.append(pos - np.floor(pos))
    return np.concatenate(out)


def test_sinc_weights_from_three_trig_calls_are_the_formulas_tap_by_tap():
    """sonar-fy6: sync._sinc_weights makes a position's 16 weights from sin(pi frac), cos(pi frac / 8) and
    sin(pi frac / 8), with sin(pi (k - frac)) = -(-1)^k sin(pi frac) and cos(pi (k - frac) / 8) = cos(pi k / 8)
    cos(pi frac / 8) + sin(pi k / 8) sin(pi frac / 8). Tap by tap against the formula over a dense set of fractions, the
    ones at the ends and the ones the chain makes; both against the formula in long double, where the platform has it."""
    tiny = np.finfo(np.float64).smallest_subnormal
    frac = np.concatenate([np.arange(200000) / 200000.0, _chain_fractions(),
                           [0.0, tiny, 1e-300, 1e-13, 1e-12, 1e-9, 0.5, np.nextafter(0.5, 0.0), np.nextafter(0.5, 1.0),
                            np.nextafter(1.0, 0.0), 1.0 - 1e-12, 1.0 - 1e-9, 1.0 - 1e-15]])
    assert frac.min() >= 0.0 and frac.max() < 1.0
    got, want = sync._sinc_weights(frac), _direct_weights(frac)
    assert np.isfinite(got).all()
    assert np.abs(got - want).max() <= WEIGHT_BOUND, np.abs(got - want).max()
    # On a sample the k = 0 tap is 1 and every other tap 0, exactly; next to the following sample its tap is 1 to a bit
    # or two (sin(pi frac) is taken from frac - 1 there: as it is, it would be wrong in its leading digits at 1 - 2^-53).
    on = sync._sinc_weights(np.array([0.0]))[:, 0]
    assert np.array_equal(on, (np.arange(-7, 9) == 0).astype(float))
    assert np.abs(sync._sinc_weights(np.array([np.nextafter(1.0, 0.0), 1.0 - 1e-9]))[8] - 1.0).max() <= WEIGHT_BOUND
    if np.finfo(np.longdouble).eps < np.finfo(np.float64).eps:
        L = np.longdouble
        d = np.arange(-7, 9).astype(L)[:, None] - frac.astype(L)[None, :]
        pi = L(4) * np.arctan(L(1))
        with np.errstate(divide="ignore", invalid="ignore"):
            exact = np.where(d == 0, L(1), np.sin(pi * d) / (pi * d)) * (L(0.5) + L(0.5) * np.cos(pi * d / 8))
        assert float(np.abs(got - exact).max()) <= WEIGHT_BOUND and float(np.abs(want - exact).max()) <= WEIGHT_BOUND


def test_whole_sample_positions_pass_the_samples_on_as_they_are():
    """frac = 0 (sonar-fy6): the samples themselves, to the bit, and 0 where the position is off either end."""
    x = np.random.default_rng(5).standard_normal((3, 50))
    positions = np.arange(-4, 54).astype(np.float64)
    got = sync._sinc_resample(x, positions)
    want = np.zeros((3, len(positions)))
    want[:, 4:54] = x
    assert np.array_equal(got, want)


def test_resampling_the_boards_gives_the_row_by_row_resamplers_samples_to_the_bit():
    """sonar-h1f: resample_boards computes a board's sinc weights once for all its rows and resamples the rows
    together; every sample is the one the row-by-row resampler with the same weights gives, to the bit, with offsets
    that run the taps past both ends of the window, either sign, with drift, and a board left as it is.
    sonar-fy6: the weights come from three trig calls a sample, not a sine and a cosine a tap; against the formula's
    weights every sample is within one float32 spacing of its value plus 16 x WEIGHT_BOUND x the largest input."""
    _, rec, _, _ = four_board_ping()
    est = sync.ClockEstimate(offset=np.array([0.0, 3.1e-4, -2.27e-4, 1.3e-5]), drift_ppm=np.array([0.0, 40.0, -55.5, 0.0]),
                             method="test", quality=np.ones(4))
    x = rec.samples.array.astype(np.float64)
    fs, n = rec.samples.sample_rate, rec.samples.n_samples
    m = np.arange(n)
    want, formula = x.copy(), x.copy()
    for b in range(1, 4):
        positions = (m / fs - est.offset[b]) * fs * (1.0 + est.drift_ppm[b] * 1e-6)
        assert positions.min() < 8 or positions.max() > n - 8          # the taps run past an end
        for row in range(8 * b, 8 * b + 8):
            want[row] = _resample_row_by_row(x[row], positions)
            formula[row] = _resample_row_by_row(x[row], positions, direct=True)
        np.testing.assert_array_equal(sync._sinc_resample(x[8 * b], positions), want[8 * b])
    jumbled = np.random.default_rng(3).permutation(positions)              # inside or outside the series by turns
    np.testing.assert_array_equal(sync._sinc_resample(x[9], jumbled), _resample_row_by_row(x[9], jumbled))
    sync.resample_boards(rec, est)
    got = rec.samples.array
    assert got.dtype == np.float32
    assert np.array_equal(got, want.astype(np.float32))
    near = formula.astype(np.float32)
    spacing = np.spacing(np.maximum(np.abs(got), np.abs(near)))
    apart = np.abs(got.astype(np.float64) - near)
    assert (apart <= spacing + 16 * WEIGHT_BOUND * np.abs(x).max()).all(), (apart - spacing).max()
    assert np.array_equal(got[:8], x[:8].astype(np.float32))          # the first board: not moved
    assert [b.clock_offset for b in rec.metadata.boards] == list(est.offset)


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


# --- the plan's batching scheme: a dedicated reference input per board (28 hydrophones) ------

def four_by_seven_ping(single_clock=False, seed=0, range_m=600.0, bearing_deg=0.0):
    over = {"receive": {"window": {"range": [range_m - 150.0, range_m + 150.0]}},
            "transmit": {"waveform": {"duration": 0.05}}}
    if single_clock:
        over["receive"]["boards"] = [
            {"first": 8 * i, "count": 8, "clock_offset": 0.0, "clock_drift_ppm": 0.0,
             "reference_channel": 8 * i + 7} for i in range(4)]
    sonar = SonarModel.load("popoto_4x7", over)
    b = np.radians(bearing_deg)
    target = range_m * np.array([np.sin(b), np.cos(b), 0.0])
    truth = {}
    rec = simulate_ping(sonar, make_pose(), Scene().add_points([target], target_strength_db=10.0),
                        environment={"sound_speed": C}, absorption_model="none", adc=False,
                        rng=np.random.default_rng(seed), clock_truth=truth)
    return sonar, rec, target, truth


def test_dedicated_reference_inputs_leave_28_hydrophones():
    sonar, rec, target, truth = four_by_seven_ping()
    x = rec.samples.array
    fs = rec.samples.sample_rate
    # Reference inputs carry the two bursts and nothing else: no echo on them.
    echo = slice(int(0.19 * fs), int(0.30 * fs))          # target echo, between the bursts
    assert np.abs(x[7, echo]).max() == 0 and np.abs(x[6, echo]).max() > 0
    ref = sonar.config["receive"]["reference"]
    report = sync.synchronise(rec, ref)
    true_offset = np.array(truth["offset"]) - truth["offset"][0]
    np.testing.assert_allclose(report.estimate.offset, true_offset, atol=0.1e-6)
    assert report.ok and report.residual.max() < report.tolerance
    hydro = sync.drop_reference_channels(rec)
    assert hydro.samples.n_channels == 28
    assert gs.validate(hydro).ok, str(gs.validate(hydro))
    assert hydro.metadata.channel_element == [i for i in range(32) if i % 8 != 7]
    # Beam pattern of the 28 synchronised hydrophones against the same array on one clock.
    _, ideal, _, _ = four_by_seven_ping(single_clock=True)
    ideal = sync.drop_reference_channels(ideal)
    bearings = np.radians(np.linspace(-12.0, 12.0, 481))
    synced, single = beam_pattern(hydro, 600.0, bearings), beam_pattern(ideal, 600.0, bearings)
    strong = single > 0.1 * single.max()
    assert np.abs(20 * np.log10(synced[strong] / single[strong])).max() < 1.0
