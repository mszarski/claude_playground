"""Range-Doppler coupling of the FLS LFM pulse, as the simulator makes it and the processing (pulse compression with
the nominal replica, backprojection) reads it: a target closing at radial speed v is read nearer by f0 v T / B for an
up-chirp and farther by the same for a down-chirp (issue sonar-9r2). This test holds the size, which
`fls_170k_traffic`'s closing boat (8 m/s relative, about 1.35 m) shows, the same for the array's own motion, and the
correction the live chain's opt-in `range_doppler` applies (sonar-fyj; detect.lfm_range_doppler_factor). By default
nothing compensates it.

Source of the formula: ours. The echo of a closing target is the pulse compressed in time by eta = (1 + v/c)/(1 - v/c),
so its frequency at each point of the sweep is up by f_d = 2 v f0 / c (to first order in v/c); against the nominal
replica of an LFM of sweep rate k = B / T the same frequencies come f_d / k earlier, a two-way delay of -f_d T / B,
a range of -c/2 f_d T / B = -f0 v T / B."""

import numpy as np
import pytest

from gpusonar import detect, recon
from gpusonar.sim import build_scene, make_pose, simulate_ping
from gpusonar.sonar import SonarModel

C = 1500.0


def _peak_range(f_start, f_stop, speed, ownship=0.0):
    """Range (m) of the compressed, beamformed peak of a point 200 m dead ahead at the array's depth, closing at
    `speed` (m/s), read at bearing 0 with a fine range grid and a parabola through the peak; the array under way
    towards it at `ownship` (m/s)."""
    sonar = SonarModel.load("fls_170k", {
        "receive": {"kind": "baseband", "sample_rate": 40000.0, "window": {"range": [180.0, 220.0]}},
        "transmit": {"lever_arm": [0, 0, 0], "waveform": {"f_start": f_start, "f_stop": f_stop}}})
    scene = build_scene([{"type": "points", "positions": [[200.0, 0.0, 20.0]], "velocity": [-speed, 0.0, 0.0],
                          "target_strength_db": 0.0}], rng=np.random.default_rng(0))
    rec = simulate_ping(sonar, make_pose(position=(0.0, 0.0, 20.0), velocity=(ownship, 0.0, 0.0)), scene, environment={"sound_speed": C},
                        adc=False, absorption_model="none", rng=np.random.default_rng(1))
    ranges = np.arange(195.0, 202.0, 0.005)
    img = recon.fan_image([rec], ranges, np.array([0.0]))
    p = np.abs(img.image[:, 0])
    i = int(np.argmax(p))
    y0, y1, y2 = p[i - 1], p[i], p[i + 1]
    return float(ranges[i] + 0.005 * 0.5 * (y0 - y2) / (y0 - 2.0 * y1 + y2)), rec


def test_a_closing_target_reads_nearer_by_f0_v_t_over_b_with_an_up_chirp():
    f1, f2, T, v = 160.0e3, 180.0e3, 20.0e-3, 8.0          # fls_170k's pulse (configs/sonars/fls_170k.yaml)
    shift = 0.5 * (f1 + f2) * v * T / (f2 - f1)               # 1.36 m
    (up, _), (down, _) = _peak_range(f1, f2, v), _peak_range(f2, f1, v)
    (still_up, _), (still_down, _) = _peak_range(f1, f2, 0.0), _peak_range(f2, f1, 0.0)
    assert still_up == pytest.approx(200.0, abs=0.05) and still_down == pytest.approx(200.0, abs=0.05)
    # The coupling flips with the sweep; what is common to both sweeps is the target's motion over the round trip
    # (v R / c = 1.07 m) and over half the pulse (v T / 2 = 0.08 m).
    assert up - down == pytest.approx(-2.0 * shift, rel=0.05)
    assert 0.5 * (up + down) == pytest.approx(200.0 - v * 200.0 / C - 0.5 * v * T, abs=0.1)


def test_the_factor_is_the_centre_frequency_times_the_pulse_over_the_signed_sweep():
    """detect.lfm_range_doppler_factor: f0 T / (f_stop - f_start), positive for an up-chirp, negative for a down-chirp,
    0 for a pulse without a linear sweep (CW, Gaussian, arbitrary), whose coupling this correction does not model."""
    from gpusonar import core
    w = core.Waveform()
    w.type, w.f_start, w.f_stop, w.duration = core.WaveformType.Lfm, 160.0e3, 180.0e3, 20.0e-3
    assert detect.lfm_range_doppler_factor(w) == pytest.approx(0.17)
    w.f_start, w.f_stop = 180.0e3, 160.0e3
    assert detect.lfm_range_doppler_factor(w) == pytest.approx(-0.17)
    w.f_stop = w.f_start
    assert detect.lfm_range_doppler_factor(w) == 0.0
    for kind in (core.WaveformType.Cw, core.WaveformType.Gaussian):
        w.type, w.f_start, w.f_stop = kind, 160.0e3, 180.0e3
        assert detect.lfm_range_doppler_factor(w) == 0.0


def test_ownship_under_way_couples_like_a_closing_target_and_the_correction_removes_it():
    """The coupling follows the closing speed whoever moves: the array under way at 2 m/s towards a static object reads
    it f0 v T / B nearer with the up-chirp (0.34 m, D-169) and as much farther with the down-chirp. The correction
    (sonar-fyj: range + factor x closing speed, factor = detect.lfm_range_doppler_factor of the record's pulse) brings
    the two sweeps together, for the array's motion and for the target's alike."""
    f1, f2, T = 160.0e3, 180.0e3, 20.0e-3
    for speed, ownship in ((0.0, 2.0), (8.0, 0.0), (6.0, 2.0)):
        (up, rec_up), (down, rec_down) = _peak_range(f1, f2, speed, ownship), _peak_range(f2, f1, speed, ownship)
        closing = speed + ownship
        shift = 0.5 * (f1 + f2) * closing * T / (f2 - f1)
        assert up - down == pytest.approx(-2.0 * shift, rel=0.05)
        k_up, k_down = detect.lfm_range_doppler_factor(rec_up.transmit.waveform), detect.lfm_range_doppler_factor(rec_down.transmit.waveform)
        fixed_up, fixed_down = up + k_up * closing, down + k_down * closing
        assert abs(fixed_up - fixed_down) < 0.05 * 2.0 * shift
