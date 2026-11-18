"""The projector's direct sound at the hydrophones (`transmit.direct_path`) and a front end
that overloads and recovers (`receive.overload`): sim/sources.py `direct_arrival`,
sim/receiver.py `apply_front_end`.

Closed forms used: spherical spreading from the projector to each element (level
A1m / r_i, delay r_i / c, phase -2 pi f r_i / c: the near field of the array, not a plane
wave); the image of the projector in the surface; the projector's sinc pattern; a hard
limiter; the exponential return of gain after a dead time; the charge a clipped burst with
mean m leaves on a coupling capacitor, m (1 - exp(-T / tau)), its decay exp(-t / tau), and the
time it holds the ADC on its rail, tau ln(V / full scale).

Every overload parameter in this file is ILLUSTRATIVE: chosen to make the closed forms easy
to check, not taken from any board.
"""
import json
import pathlib
import subprocess
import sys

import numpy as np
import pytest

import gpusonar as gs
from gpusonar import taps
from gpusonar.config import load_scenario_config
from gpusonar.live import LiveChain
from gpusonar.sim import Scene, make_pose, make_provider, run_scenario, simulate_ping
from gpusonar.sim import receiver as rx
from gpusonar.sonar import SonarModel

core = gs.core
ROOT = pathlib.Path(__file__).resolve().parents[2]
C = 1500.0
F = 5000.0
ENV = {"sound_speed": C, "temperature": 12.0, "salinity": 35.0}
FS = 102400.0


def sonar(samples=4096, start=0.0, count=8, direct=True, overload=None, wave=None, transmit=None, receive=None, base="popoto_long_range_3row"):
    """The three-row face (projector 4 m from the array's centre, in its plane) with a short pulse."""
    w = dict(type="cw", f_start=F, f_stop=F, duration=0.004, window_alpha=0.0)
    w.update(wave or {})
    tx = {"waveform": w, **(transmit or {})}
    if direct is not None:
        tx["direct_path"] = direct
    # The window as a range pair (what the sonar files use): it opens at `start` and holds `samples`.
    r = {"window": {"range": [0.5 * C * start, 0.5 * C * (start + (samples - 0.5) / FS - w["duration"])]}, **(receive or {})}
    if overload is not None:
        r["overload"] = overload
    return SonarModel.load(base, {"array": {"count": count}, "transmit": tx, "receive": r})


def ping(model, scene=None, *, provider=None, pose=None, truth=None, **kw):
    kw.setdefault("adc", False)
    return simulate_ping(model, pose or make_pose(position=(0.0, 0.0, 10.0)), scene or Scene(), environment=ENV,
                         provider=provider, absorption_model="none", front_end_truth=truth, **kw)


def times(rec):
    s = rec.samples
    return s.start_time + np.arange(s.n_samples) / s.sample_rate


def tone_fit(x, t, f):
    """Complex amplitude a of a real tone x = Re{a exp(2 pi i f t)}, by least squares."""
    basis = np.stack([np.cos(2 * np.pi * f * t), -np.sin(2 * np.pi * f * t)], axis=1)
    re, im = np.linalg.lstsq(basis, np.asarray(x, dtype=float), rcond=None)[0]
    return complex(re, im)


def geometry(rec):
    pos, tx = np.asarray(rec.rx_positions()), np.asarray(rec.tx_position())
    return pos, tx, np.linalg.norm(pos - tx, axis=1)


# --- the direct arrival ----------------------------------------------------------------------

def test_direct_arrival_has_each_elements_own_spreading_delay_and_phase():
    model = sonar()
    rec = ping(model)
    x, t = rec.samples.array, times(rec)
    _, _, r = geometry(rec)
    assert r.min() == pytest.approx(3.4809, abs=1e-3) and r.max() / r.min() > 1.3          # the near field: no plane wave
    a1 = model.amplitude_at_1m()                                                          # 10^((SL + sensitivity + gain) / 20)
    assert a1 == pytest.approx(10.0 ** ((197.0 - 208.0 + 43.0) / 20.0))
    fits = []
    for ch in range(x.shape[0]):
        inside = (t > r[ch] / C + 2e-4) & (t < r[ch] / C + 0.004 - 2e-4)
        fits.append(tone_fit(x[ch, inside], t[inside], F))
        assert 20 * np.log10(abs(fits[-1]) / (a1 / r[ch])) == pytest.approx(0.0, abs=0.01)   # SL - 20 log r_i
        assert np.abs(x[ch, t < r[ch] / C - 2.0 / FS]).max() < 1e-3 * a1 / r[ch]            # nothing before r_i / c
        assert np.abs(x[ch, t > r[ch] / C + 0.004 + 2.0 / FS]).max() < 1e-3 * a1 / r[ch]    # nor after the pulse
    fits = np.array(fits)
    phase = np.angle(fits / fits[0] * np.exp(2j * np.pi * F * (r - r[0]) / C))
    assert np.abs(phase).max() < 2e-3                                                    # -2 pi f r_i / c


def test_keys_absent_or_off_change_nothing(tmp_path):
    scene = Scene().add_points([[10.0, 60.0, 10.0]], target_strength_db=0.0)
    plain = SonarModel.load("popoto_long_range_3row", {"array": {"count": 8}, "receive": {"window": {"range": [0.0, 20.0]}}})
    off = SonarModel.load("popoto_long_range_3row", {"array": {"count": 8}, "transmit": {"direct_path": False},
                                                      "receive": {"window": {"range": [0.0, 20.0]}}})
    a = ping(plain, scene, adc=True, noise_rms=1e-4, rng=np.random.default_rng(3))
    with taps.sim_capture(tmp_path):
        b = ping(off, scene, adc=True, noise_rms=1e-4, rng=np.random.default_rng(3))
    assert np.array_equal(a.samples.array, b.samples.array)
    assert list(taps.read_sim_taps(tmp_path)) == list(taps.SIM_STAGES)                    # no new simulator stage either
    # An overload block with every stage switched off is the receiver as it was.
    none = SonarModel.load("popoto_long_range_3row", {"array": {"count": 8}, "receive": {
        "window": {"range": [0.0, 20.0]}, "overload": {"adc": {"clip": False}}}})
    truth = {}
    c = ping(none, scene, adc=True, noise_rms=1e-4, rng=np.random.default_rng(3), truth=truth)
    assert np.array_equal(a.samples.array, c.samples.array) and np.all(np.isnan(truth["last_overload_time"]))


def test_surface_image_of_the_projector_arrives_with_its_own_level_delay_and_sign():
    model = sonar(samples=8192, wave={"duration": 0.002})
    provider = make_provider("image_source", C, {"surface_z": 0.0, "bottom_z": None, "max_order": 1})
    rec = ping(model, provider=provider)
    x, t = rec.samples.array, times(rec)
    pos, tx, r = geometry(rec)
    image = tx * np.array([1.0, 1.0, -1.0])                                               # mirrored in z = 0
    r_image = np.linalg.norm(pos - image, axis=1)
    a1 = model.amplitude_at_1m()
    for ch in (0, 7, 23):
        direct = (t > r[ch] / C + 2e-4) & (t < r[ch] / C + 0.002 - 2e-4)
        bounce = (t > r_image[ch] / C + 2e-4) & (t < r_image[ch] / C + 0.002 - 2e-4)
        a_d, a_s = tone_fit(x[ch, direct], t[direct], F), tone_fit(x[ch, bounce], t[bounce], F)
        assert abs(a_s) == pytest.approx(a1 / r_image[ch], rel=2e-3)
        want = -np.exp(-2j * np.pi * F * (r_image[ch] - r[ch]) / C) * r[ch] / r_image[ch]   # pressure release: -1
        assert a_s / a_d == pytest.approx(want, abs=5e-3)
    # `boundaries: false` keeps the straight arrival only.
    only = ping(sonar(samples=8192, wave={"duration": 0.002}, direct={"boundaries": False}), provider=provider)
    late = t > r.max() / C + 0.002 + 1e-3
    assert np.abs(only.samples.array[:, late]).max() == 0.0 and np.abs(x[:, late]).max() > 0.5 * a1 / r_image.max()
    assert np.allclose(only.samples.array[:, ~late], x[:, ~late], atol=1e-6 * a1)


def test_projector_pattern_weights_each_element_and_each_ray_by_its_own_direction():
    bw = 0.5                                                                              # rad, elevation beamwidth of the projector
    model = sonar(samples=8192, wave={"duration": 0.002}, transmit={"boresight": [-1.0, 0.0, 0.0], "beamwidth_el": bw})
    provider = make_provider("image_source", C, {"surface_z": 0.0, "bottom_z": None, "max_order": 1})
    rec = ping(model, provider=provider)
    x, t = rec.samples.array, times(rec)
    pos, tx, r = geometry(rec)
    image_elements = pos * np.array([1.0, 1.0, -1.0])                                     # the ray to the image of the element
    r_image = np.linalg.norm(image_elements - tx, axis=1)
    a1 = model.amplitude_at_1m()
    gains = []
    for ch in range(x.shape[0]):
        direct = (t > r[ch] / C + 2e-4) & (t < r[ch] / C + 0.002 - 2e-4)
        bounce = (t > r_image[ch] / C + 2e-4) & (t < r_image[ch] / C + 0.002 - 2e-4)
        g_direct = np.sinc(0.886 * ((pos[ch, 2] - tx[2]) / r[ch]) / bw)                    # sinc(0.886 sin(elevation) / beamwidth)
        g_bounce = np.sinc(0.886 * ((image_elements[ch, 2] - tx[2]) / r_image[ch]) / bw)   # leaving upwards, towards the surface
        gains.append(g_direct)
        assert abs(tone_fit(x[ch, direct], t[direct], F)) == pytest.approx(a1 / r[ch] * g_direct, rel=2e-3)
        assert abs(tone_fit(x[ch, bounce], t[bounce], F)) == pytest.approx(a1 / r_image[ch] * abs(g_bounce), rel=5e-3, abs=2e-4 * a1 / r_image[ch])
    assert max(gains) - min(gains) > 0.03                                                 # the rows do sit at different angles


def test_projector_and_array_moving_together_hear_no_doppler():
    model = sonar(samples=4096, wave={"duration": 0.03})
    still = ping(model)
    moving = ping(model, pose=make_pose(position=(0.0, 0.0, 10.0), velocity=(3.0, 0.0, 0.0)))
    t = times(still)
    _, _, r = geometry(still)
    ch = int(np.argmax(r))
    early = (t > r[ch] / C + 1e-3) & (t < r[ch] / C + 6e-3)
    late = (t > r[ch] / C + 2.2e-2) & (t < r[ch] / C + 2.7e-2)
    drift = [np.angle(tone_fit(rec.samples.array[ch, late], t[late], F) / tone_fit(rec.samples.array[ch, early], t[early], F))
             for rec in (still, moving)]
    # A projector closing at 3 m/s on a still array would shift 5 kHz by 10 Hz: 1.3 rad over these 21 ms.
    assert abs(drift[1] - drift[0]) < 2e-3
    assert np.abs(moving.samples.array - still.samples.array).max() > 1e-3 * np.abs(still.samples.array).max()   # the delay does change


def test_sampled_pulse_direct_arrival_is_the_same_on_both_engines_and_matches_the_analytic_one():
    lfm = {"type": "lfm", "f_start": 4000.0, "f_stop": 6000.0, "duration": 0.01, "window_alpha": 0.0}
    analytic = ping(sonar(wave=lfm)).samples.array
    sampled = {"type": "sampled", "source": "lfm", "f_start": 4000.0, "f_stop": 6000.0, "duration": 0.01}
    got = {engine: ping(sonar(wave=sampled), engine=engine).samples.array for engine in ("cpp", "numpy")}
    peak = np.abs(analytic).max()
    assert np.abs(got["cpp"] - got["numpy"]).max() < 2e-4 * peak
    t = np.arange(analytic.shape[1]) / FS
    body = (t > 3.2e-3) & (t < 1.0e-2)                                                   # away from the edges the two pulses are the same sweep
    assert np.abs(got["cpp"][:, body] - analytic[:, body]).max() < 0.03 * peak


# --- the limiter, the recovery and the coupling capacitor ------------------------------------

def test_limit_stage_clips_holds_its_rail_through_the_dead_time_and_returns_gain_exponentially():
    n = 20000
    t = np.arange(n) / FS
    burst = t < 0.05
    u = np.where(burst, 6.0 * np.sin(2 * np.pi * F * t), 0.2 * np.sin(2 * np.pi * F * t))[None, :]
    lo, hi, dead, tau = 1.0, 2.0, 0.01, 0.02
    y, clipped, held, gain = rx.limit_stage(u, FS, (lo, hi), 0.0, 0.0)
    assert np.array_equal(y, np.clip(u, -lo, hi)) and np.array_equal(clipped, (u >= hi) | (u <= -lo))
    y, clipped, held, gain = rx.limit_stage(u, FS, (lo, hi), dead, tau, "rail")
    last = np.nonzero(clipped[0])[0][-1]
    since = (np.arange(n) - last) / FS
    rail = hi if u[0, last] > 0 else -lo
    in_dead = (since > 0) & (since <= dead)
    after = since > dead
    assert np.all(y[0, in_dead] == rail) and np.all(held[0, in_dead])
    want = 1.0 - np.exp(-(since[after] - dead) / tau)
    assert np.allclose(gain[0, after], want, atol=1e-12) and np.allclose(y[0, after], want * u[0, after], atol=1e-12)
    zero = rx.limit_stage(u, FS, (lo, hi), dead, tau, "zero")[0]
    assert np.all(zero[0, in_dead] == 0.0) and np.array_equal(zero[0, after], y[0, after])
    # A stage shared by a board: the channel that never clipped is dead and recovering with the one that did.
    two = np.vstack([u[0], 0.2 * np.sin(2 * np.pi * F * t)])
    own = rx.limit_stage(two, FS, (lo, hi), dead, tau, "rail", shared=False)
    both = rx.limit_stage(two, FS, (lo, hi), dead, tau, "rail", shared=True)
    assert np.array_equal(own[0][1], two[1]) and not own[2][1].any()
    assert np.all(both[0][1, in_dead] == 0.0) and np.allclose(both[0][1, after], want * two[1, after], atol=1e-12)
    assert np.array_equal(both[0][0], own[0][0])


def clipped_mean(amplitude, lo, hi):
    """Mean of a sine of the given amplitude limited to [-lo, hi] (integrated over a cycle)."""
    def half(limit):
        if amplitude <= limit:
            return 2.0 * amplitude
        theta = np.arcsin(limit / amplitude)
        return 2.0 * amplitude * (1.0 - np.cos(theta)) + limit * (np.pi - 2.0 * theta)
    return (half(hi) - half(lo)) / (2.0 * np.pi)


def front_end(x, overload, start=0.0, pre=None, receive=None):
    """The model on bare samples [C, N], with its truth."""
    model = sonar(samples=x.shape[1], start=start, count=max(x.shape[0] // 3, 1), overload=overload, receive=receive)
    rec = model.make_record(make_pose())
    cfg = rx.overload_config(model)
    rx.report_gain_switch(rec, cfg)
    truth = {}
    return rx.apply_front_end(rec, x.astype(np.float32), cfg, pre=pre, truth=truth, arrays=True), truth, rec


def test_unequal_rails_charge_the_coupling_capacitor_which_then_holds_the_adc_on_its_rail():
    # A low corner: the carrier's ripple on the capacitor is under 1 %. The burst ends on a crest, where the
    # capacitor's own (linear) response to the carrier, A / (2 pi f tau) cos(2 pi f t), passes through zero.
    n, burst_time, corner = 60000, 500.25 / F, 10.0
    t = np.arange(n) / FS
    tau = 1.0 / (2 * np.pi * corner)
    gain_db, preamp_gain_db = 43.0, 23.0                       # 20 dB of gain behind the capacitor
    k = 10.0 ** ((gain_db - preamp_gain_db) / 20.0)
    lo, hi = 0.5, 1.5                                          # V at the input stage's output
    amplitude = 20.0 * k                                       # ADC-referred volts: 20 V at the input stage's output
    x = np.tile(np.where(t < burst_time, amplitude * np.sin(2 * np.pi * F * t), 0.0), (3, 1))
    y, truth, _ = front_end(x, {"preamp": {"limit_v": [lo, hi], "gain_db": preamp_gain_db}, "highpass_corner_hz": corner,
                                "adc": {"limit_v": 2.61}})
    m = clipped_mean(20.0, lo, hi)
    assert m == pytest.approx(0.5 * (hi - lo), rel=0.05)       # nearly a square wave: half the difference of the rails
    after = t > burst_time + 2.0 / F
    want = -m * k * (1.0 - np.exp(-burst_time / tau)) * np.exp(-(t - burst_time) / tau)
    assert np.allclose(truth["offset"][0, after], want[after], atol=0.03 * m * k)
    # The offset starts above the ADC's full scale: the ADC sits on its rail for tau ln(V / full scale).
    held_for = tau * np.log(m * k / 2.61)
    released = np.nonzero((truth["flags"][0] & rx.FLAG_ADC) != 0)[0][-1] / FS - burst_time
    assert held_for > 5e-3 and released == pytest.approx(held_for, abs=4e-4)
    stuck = after & (t < burst_time + held_for - 5e-4)
    assert np.all(y[0, stuck] == np.float32(-2.61))
    free = t > burst_time + held_for + 5e-4
    assert np.allclose(y[0, free], want[free], atol=0.03 * m * k)
    assert truth["peak_db_re_preamp_limit"][0] == pytest.approx(20 * np.log10(20.0 / lo), abs=0.01)
    # Equal rails: the clipped burst has no mean and leaves (almost) nothing; the ripple is about L f_c / f.
    y, truth, _ = front_end(x, {"preamp": {"limit_v": 1.5, "gain_db": preamp_gain_db}, "highpass_corner_hz": corner, "adc": {"limit_v": 2.61}})
    assert np.abs(truth["offset"][0, after]).max() < 0.03 * 1.5 * k
    # The rail held through a dead time is a mean too: m = L for T_d.
    dead = 0.004
    y, truth, _ = front_end(x, {"preamp": {"limit_v": 1.5, "gain_db": preamp_gain_db, "dead_time": dead}, "highpass_corner_hz": corner,
                                "adc": {"clip": False}})
    last = np.nonzero((truth["flags"][0] & rx.FLAG_PREAMP) != 0)[0][-1]
    sign = np.sign(y[0, last + 5])
    later = t > t[last] + dead + 1e-4
    want = -sign * 1.5 * k * (1.0 - np.exp(-dead / tau)) * np.exp(-(t - t[last] - dead) / tau)
    assert np.allclose(y[0, later], want[later], atol=0.03 * 1.5 * k)


def test_coupling_alone_passes_the_band_and_removes_a_step_with_the_corners_time_constant():
    n, corner = 30000, 100.0
    t = np.arange(n) / FS
    tone = np.sin(2 * np.pi * F * t)[None, :]
    out = rx.coupling_highpass(tone, FS, corner)
    a_in, a_out = tone_fit(tone[0, 10000:], t[10000:], F), tone_fit(out[0, 10000:], t[10000:], F)
    # s / (s + 2 pi f_c) at 5 kHz (the bilinear transform warps 5 kHz by 0.8 %: 1.6e-4 in the quadrature part).
    assert a_out / a_in == pytest.approx(1j * F / (corner + 1j * F), abs=3e-4)
    step = rx.coupling_highpass(np.ones((1, n)), FS, corner)
    assert np.allclose(step[0], np.exp(-2 * np.pi * corner * t), atol=4e-3)


def test_state_is_carried_from_the_transmit_into_a_window_that_opens_after_it():
    """The same ping recorded from the transmit on, and from 40 ms on (after the pulse): the
    later window must hold the same samples, which needs the dead time and the recovering gain
    of an overload that happened before it opened."""
    lfm = {"type": "lfm", "f_start": 4000.0, "f_stop": 6000.0, "duration": 0.02, "window_alpha": 0.0}
    overload = {"preamp": {"limit_v": [1.0, 3.0], "gain_db": 33.0, "dead_time": 0.004, "recovery_tau": 0.03},
                "highpass_corner_hz": 10.0, "adc": {"dead_time": 0.002, "recovery_tau": 0.01, "dead_output": "zero"}}
    scene = Scene().add_points([[5.0, 45.0, 10.0]], target_strength_db=20.0)                # its echo starts at 60 ms
    n_all, n_skip = 16384, 4096
    whole_truth, late_truth = {"arrays": True}, {"arrays": True}
    whole = ping(sonar(samples=n_all, wave=lfm, overload=overload), scene, truth=whole_truth)
    late = ping(sonar(samples=n_all - n_skip, start=n_skip / FS, wave=lfm, overload=overload), scene, truth=late_truth)
    a, b = whole.samples.array[:, n_skip:], late.samples.array
    scale = np.abs(a).max()
    assert late_truth["pre_window_samples"] == n_skip and whole_truth["pre_window_samples"] == 0
    assert np.abs(a - b).max() < 2e-3 * scale
    assert np.allclose(whole_truth["gain"][:, n_skip:], late_truth["gain"], atol=1e-3)
    assert np.allclose(whole_truth["last_overload_time"], late_truth["last_overload_time"], atol=2.0 / FS)
    assert np.allclose(whole_truth["peak_db_re_adc_limit"], late_truth["peak_db_re_adc_limit"], atol=0.05)
    assert 0.1 < late_truth["gain"][:, 0].min() and late_truth["gain"][:, 0].max() < 0.6     # still recovering when it opens
    # Without the interval before the window the front end would look untouched: the echo at full gain.
    linear = ping(sonar(samples=n_all - n_skip, start=n_skip / FS, wave=lfm), scene).samples.array
    model = sonar(samples=n_all - n_skip, start=n_skip / FS, wave=lfm, overload=overload)
    cold = rx.apply_front_end(model.make_record(make_pose()), linear, rx.overload_config(model))
    assert np.abs(cold - b).max() > 0.2 * scale


def test_overload_figures_of_the_three_row_face_match_the_sonar_equation():
    model = SonarModel.load("popoto_long_range_3row", {"transmit": {"direct_path": True, "waveform": {"duration": 0.02}},
                                                       "receive": {"window": {"range": [0.0, 20.0]}, "overload": {}}})
    truth = {}
    rec = ping(model, truth=truth, pose=make_pose())
    _, _, r = geometry(rec)
    want = 197.0 - 20 * np.log10(r) - 208.0 + 43.0 - 20 * np.log10(2.61)                   # SL - 20 log r + sensitivity + gain - full scale
    assert np.allclose(truth["peak_db_re_adc_limit"], want, atol=0.05)
    assert r.min() == pytest.approx(1.687, abs=1e-3) and want.max() == pytest.approx(19.1, abs=0.05) and want.min() > 7.0
    assert np.abs(rec.samples.array).max() == np.float32(2.61) and np.all(truth["clipped_fraction"] > 0.0)
    assert np.all(np.isnan(truth["peak_db_re_preamp_limit"]))                              # no input stage was described


def test_gain_switch_keeps_the_front_end_linear_and_is_reported_in_the_records_ramp():
    lfm = {"type": "lfm", "f_start": 4000.0, "f_stop": 6000.0, "duration": 0.02, "window_alpha": 0.0}
    scene = Scene().add_points([[5.0, 45.0, 10.0]], target_strength_db=20.0)
    switch = {"gain_db": -30.0, "until": 0.03, "ramp": 0.01}
    slow = {"dead_time": 0.05, "recovery_tau": 0.05, "dead_output": "zero"}
    linear = ping(sonar(samples=16384, wave=lfm), scene)
    truth = {"arrays": True}
    rec = ping(sonar(samples=16384, wave=lfm, overload={"adc": slow, "gain_switch": switch}), scene, truth=truth)
    t = times(rec)
    assert truth["peak_db_re_adc_limit"].max() < 0.0 and not truth["flags"].any()
    want = linear.samples.array * 10.0 ** (np.interp(t, [0.03, 0.04], [-30.0, 0.0]) / 20.0)
    assert np.allclose(rec.samples.array, want, atol=1e-6 * np.abs(want).max())
    assert list(rec.metadata.tvg_time) == pytest.approx([0.03, 0.04]) and list(rec.metadata.tvg_gain_db) == [-30.0, 0.0]
    echo = t > 0.06
    core.apply_tvg(rec, spreading_db_per_decade=0.0)                                       # a processor undoing the record's ramp
    assert np.allclose(rec.samples.array[:, echo], linear.samples.array[:, echo], atol=1e-5 * np.abs(want).max())
    # Without the switch the same slow front end is dead when the echo arrives.
    lost = ping(sonar(samples=16384, wave=lfm, overload={"adc": slow}), scene)
    assert np.abs(lost.samples.array[:, (t > 0.06) & (t < 0.07)]).max() == 0.0
    # Not reported: the samples are the same, the record carries no ramp.
    quiet = ping(sonar(samples=16384, wave=lfm, overload={"adc": slow, "gain_switch": dict(switch, report=False)}), scene)
    assert np.allclose(quiet.samples.array, want, atol=1e-6 * np.abs(want).max()) and len(quiet.metadata.tvg_time) == 0
    with pytest.raises(ValueError, match="gain_ramp"):
        ping(sonar(samples=4096, wave=lfm, overload={"gain_switch": switch}, receive={"tvg": {"time": [0.0, 1.0], "gain_db": [0.0, 6.0]}}))


def test_a_stage_shared_by_a_board_takes_the_whole_board_down_and_only_that_board():
    lfm = {"type": "lfm", "f_start": 4000.0, "f_stop": 6000.0, "duration": 0.01, "window_alpha": 0.0}
    limit = 11.4                                               # V: the direct level 3.5 m from the projector, inside the nearer boards' span
    truths = {}
    for scope in ("channel", "board"):
        truths[scope] = {"arrays": True}
        rec = ping(sonar(samples=8192, count=16, wave=lfm, receive={"board_size": 8},
                         overload={"adc": {"limit_v": limit, "dead_time": 0.01, "scope": scope, "dead_output": "zero"}}), truth=truths[scope])
    boards = np.asarray(rec.metadata.channel_board)
    clips = truths["channel"]["peak_db_re_adc_limit"] > 0.0
    mixed = [b for b in np.unique(boards) if clips[boards == b].any() and not clips[boards == b].all()]
    clean = [b for b in np.unique(boards) if not clips[boards == b].any()]
    assert mixed and clean
    dead = {scope: (truths[scope]["flags"] & rx.FLAG_HELD).any(axis=1) for scope in truths}
    assert np.array_equal(dead["channel"], clips)
    for b in mixed:
        assert dead["board"][boards == b].all()
    for b in clean:
        assert not dead["board"][boards == b].any()


def test_overload_refuses_a_complex_record_and_unknown_keys():
    with pytest.raises(ValueError, match="passband"):
        ping(sonar(overload={}, receive={"kind": "baseband", "sample_rate": 8000.0}))
    with pytest.raises(ValueError, match="unknown keys"):
        rx.overload_config(sonar(overload={"preamp": {"limit": 1.0}}))
    with pytest.raises(ValueError, match="limit_v"):
        rx.overload_config(sonar(overload={"preamp": {"gain_db": 20.0}}))
    with pytest.raises(ValueError, match="gain_db"):
        rx.overload_config(sonar(overload={"gain_switch": {"until": 0.1}}))


def test_overload_intervals_and_recovery_time():
    t = np.arange(1000) / 1000.0
    flags = np.zeros(1000, dtype=np.uint8)
    flags[100:200:3] = 1
    flags[500:520] = 2
    assert rx.overload_intervals(flags, t) != rx.overload_intervals(flags, t, merge_gap=0.005)
    assert rx.overload_intervals(flags, t, merge_gap=0.005) == [(pytest.approx(0.1), pytest.approx(0.199)), (pytest.approx(0.5), pytest.approx(0.519))]
    assert rx.overload_intervals(flags, t, merge_gap=0.005, mask=2) == [(pytest.approx(0.5), pytest.approx(0.519))]
    gain = np.clip((t - 0.52) / 0.2, 0.0, 1.0)
    truth = {"time": t, "flags": flags[None, :], "gain": gain[None, :], "offset": np.zeros((1, 1000)), "sample_rate": 1000.0}
    assert rx.recovery_time(truth, 0.3, 6.0206, 1.0)[0] == pytest.approx(0.62, abs=2e-3)   # gain back to a half
    assert rx.recovery_time(truth, 0.8, 1.0, 1.0)[0] == 0.8


# --- the scenario, the live chain and the study tool -----------------------------------------

CHAIN = {"bearings_deg": [-60.0, 60.0, 1.0], "range_cell_m": 2.0, "cfar": {"pfa": 1e-6, "guard_m": 15.0, "train_m": 50.0},
         "min_snr_db": 12.0, "backend": "host", "classify": {"multipath": None}}


def detections(result):
    with LiveChain(CHAIN) as chain:
        report = chain.run([r.copy() for r in result.records])[0]
    return [d for d in report["detections"] if d["snr_db"] > 30.0]


def near(found, range_m):
    return [d for d in found if abs(d["range_m"] - range_m) < 5.0]


@pytest.fixture(scope="module")
def slow_scenario():
    return run_scenario(load_scenario_config("popoto_testauv_direct_overload", {"pings": {"count": 1}}))


def test_close_target_is_lost_inside_the_recovery_interval(slow_scenario):
    result = slow_scenario
    truth = result.front_end[0]
    assert truth["peak_db_re_adc_limit"].min() > 20.0 and truth["peak_db_re_adc_limit"].max() == pytest.approx(34.2, abs=0.2)
    # Clipping ends with the pulse (0.1 s, plus the travel and the boundary returns); the stage is dead 0.25 s more.
    assert np.all((truth["last_overload_time"] > 0.34) & (truth["last_overload_time"] < 0.37))
    rec = result.records[0]
    assert np.abs(rec.samples.array).max() == pytest.approx(2.61, rel=1e-6) and taps.stats(rec)[0]["clipped_fraction"] > 0.05
    assert rec.samples.start_time < 0.1 and len(rec.metadata.tvg_time) == 0                 # clipped samples are in the record; nothing else is
    found = detections(result)
    assert not near(found, 151.0) and near(found, 350.4)                                   # 0.2-0.3 s is inside the dead time; 0.47 s is not


@pytest.mark.parametrize("variant", ["gain_switch", "fast"])
def test_close_target_is_restored_by_gain_switching_or_a_faster_recovery(variant):
    result = run_scenario(load_scenario_config(f"popoto_testauv_direct_overload_{variant}", {"pings": {"count": 1}}))
    truth = result.front_end[0]
    found = detections(result)
    assert near(found, 151.0) and near(found, 350.4)
    if variant == "gain_switch":
        assert truth["peak_db_re_adc_limit"].max() < -5.0 and np.all(np.isnan(truth["last_overload_time"]))
        assert list(result.records[0].metadata.tvg_gain_db) == [-40.0, 0.0]
    else:
        assert truth["peak_db_re_adc_limit"].max() > 30.0 and np.nanmax(truth["last_overload_time"]) < 0.13


def test_both_engines_give_the_same_overloaded_record(slow_scenario):
    other = run_scenario(load_scenario_config("popoto_testauv_direct_overload", {"pings": {"count": 1}}), engine="numpy")
    a, b = slow_scenario.records[0].samples.array, other.records[0].samples.array
    assert np.abs(a - b).max() < 1e-4 * 2.61
    for key in ("peak_db_re_adc_limit", "last_overload_time", "clipped_fraction"):
        assert np.allclose(slow_scenario.front_end[0][key], other.front_end[0][key], atol=1e-3)


def test_direct_sound_study_tool(tmp_path):
    out = tmp_path / "study.json"
    done = subprocess.run([sys.executable, str(ROOT / "tools" / "direct_sound_study.py"), "popoto_testauv_1row", "--pulses", "0.02",
                           "--sets", "clip_only", "dead_10ms_tau_10ms", "--out", str(out)],
                          capture_output=True, text=True, env={"PYTHONPATH": str(ROOT / "python"), "PATH": "/usr/bin:/bin"})
    assert done.returncode == 0, done.stderr
    study = json.loads(out.read_text())["popoto_testauv_1row"]
    model = SonarModel.load("popoto_testauv_1row")
    r = np.linalg.norm(model.positions - np.array([0.0, 0.0, 0.3]), axis=1)
    want = 197.0 - 20 * np.log10(r) - 208.0 + 43.0 - 20 * np.log10(2.61)
    table = study["hydrophones"]
    assert [h["distance_m"] for h in table] == pytest.approx(list(r), abs=1e-3)       # tabulated to the millimetre
    assert [h["overload_db_re_adc"] for h in table] == pytest.approx(list(want), abs=0.01)
    assert study["simulated_minus_closed_form_db"]["max_abs"] < 0.1
    case = {c["set"]: c for c in study["recovery"]["0.02"]}
    pulse_end = 0.02 + r.max() / C
    # Clipping only: back as soon as the pulse has passed the farthest hydrophone.
    assert case["clip_only"]["clean_after_transmit_s"]["worst"] == pytest.approx(pulse_end, abs=5e-4)
    # Dead 10 ms, then gain within 1 dB after tau ln(1 / (1 - 10^(-1/20))) = 2.22 tau.
    want_time = 0.010 + 0.010 * np.log(1.0 / (1.0 - 10.0 ** (-1.0 / 20.0)))
    assert case["dead_10ms_tau_10ms"]["recovery_after_pulse_s"]["worst"] == pytest.approx(want_time, abs=6e-4)
    assert case["dead_10ms_tau_10ms"]["closed_form_recovery_s"] == pytest.approx(want_time, abs=1e-6)
    assert case["dead_10ms_tau_10ms"]["minimum_range_m"]["worst"] == pytest.approx(0.5 * C * (pulse_end + want_time), abs=1.0)
