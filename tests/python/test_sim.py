"""Milestone 3: simulator point targets on every sonar config.

Acceptance: echo delay and amplitude match analytic values; CPU and GPU forward
models agree (the GPU half is skipped where no device is present).
"""

import numpy as np
import pytest
from scipy.optimize import brentq
from scipy.signal import hilbert

import gpusonar as gs
from gpusonar import acoustics
from gpusonar.config import list_configs, load_sonar_config
from gpusonar.sim import (ImageSourceProvider, Scene, make_pose, run_scenario, simulate_ping)
from gpusonar.sonar import SonarModel

ALL_SONARS = ["l12_5d", "p4_2v", "fls_300k", "sas_rail", "sas_auv", "popoto_8", "popoto_8_low",
              "popoto_4x8", "flank_32"]


def small(name, r0, r1, **extra):
    """A sonar with its receive window cut down so tests stay fast."""
    over = {"receive": {"window": {"range": [r0, r1]}}}
    over = gs.config.deep_merge(over, extra) if extra else over
    return SonarModel.load(name, over)


def peak_time(x, fs, start):
    """Time of the envelope peak, by parabolic interpolation of |analytic signal|."""
    env = np.abs(x) if np.iscomplexobj(x) else np.abs(hilbert(x))
    i = int(np.argmax(env))
    a, b, c = env[i - 1], env[i], env[i + 1]
    frac = 0.5 * (a - c) / (a - 2 * b + c)
    peak = b - 0.25 * (a - c) * frac
    return start + (i + frac) / fs, peak


# --- configs -------------------------------------------------------------------

def test_every_named_sonar_is_shipped():
    shipped = set(list_configs("sonars"))
    assert set(ALL_SONARS) <= shipped                 # the nine the brief names
    # Anything beyond those is listed here on purpose, so additions are deliberate.
    assert shipped - set(ALL_SONARS) == {"subbottom", "popoto_4x7", "fls_750k",
                                         # the owner's designs (docs/DESIGN.md)
                                         "fls_120k", "fls_170k", "fls_330k", "sas_minsas120", "popoto_array",
                                         "popoto_long_range"}


@pytest.mark.parametrize("name", ALL_SONARS)
def test_sonar_config_builds_a_valid_record(name):
    # Short windows: this checks structure, not signal content.
    c = load_sonar_config(name).get("nominal_sound_speed", 1500.0)
    sonar = small(name, 1.0, 1.0 + 1e-3 * c)
    rec = sonar.make_record(make_pose())
    report = gs.validate(rec)
    assert report.ok, str(report)
    assert rec.metadata.array_id == name
    assert rec.samples.n_channels == len(sonar.mux_groups[0])


@pytest.mark.parametrize(
    "name, count, pitch, fc, fs",
    [
        ("l12_5d", 128, 0.3e-3, 5.0e6, 50e6),
        ("p4_2v", 64, 0.3e-3, 2.72e6, 50e6),
        ("fls_300k", 128, 2.5e-3, 300e3, 1.25e6),
        ("popoto_8", 8, 0.027, 27.5e3, 102400.0),
        ("popoto_4x8", 32, 0.094, 27.5e3, 102400.0),
        ("flank_32", 32, 0.094, 8e3, 64000.0),
    ],
)
def test_design_points_match_the_brief(name, count, pitch, fc, fs):
    sonar = SonarModel.load(name)
    assert sonar.n_elements == count
    d = np.linalg.norm(np.diff(sonar.positions, axis=0), axis=1)
    np.testing.assert_allclose(d, pitch, rtol=1e-12)
    assert sonar.center_frequency == pytest.approx(fc)
    assert sonar.sample_rate == fs


def test_brief_specifics():
    p4 = SonarModel.load("p4_2v")
    assert p4.bandwidth / p4.center_frequency == pytest.approx(0.74, abs=1e-6)
    flank = SonarModel.load("flank_32")
    assert flank.aperture == pytest.approx(31 * 0.094)  # "3.0 m" is 32 pitches; ends are 2.91 m apart
    assert flank.waveform_config["duration"] == 1.0
    assert flank.config["receive"]["adc_bits"] == 24
    start, n = flank.window()
    assert start + n / flank.sample_rate >= 2 * 10000.0 / 1500.0  # the 13.3 s round trip fits
    l12 = SonarModel.load("l12_5d")
    assert [len(g) for g in l12.mux_groups] == [64, 64]
    assert 140_000 <= l12.window()[1] <= 160_000  # "about 150k samples per ping"
    sparse = SonarModel.load("popoto_8", {"array": {"layout_choice": "sparse"}})
    assert sparse.aperture == pytest.approx(3.0)
    assert len(np.unique(np.round(np.diff(np.sort(sparse.positions[:, 0])), 6))) > 1
    low = SonarModel.load("popoto_8_low")
    assert (low.waveform_config["f_start"], low.waveform_config["f_stop"]) == (6e3, 10e3)
    assert len(SonarModel.load("popoto_4x8").boards) == 4


def test_mux_halves_carry_their_element_indices():
    sonar = small("l12_5d", 0.05, 0.06)
    a = sonar.make_record(make_pose(), sub_sequence=0)
    b = sonar.make_record(make_pose(), sub_sequence=1)
    assert a.metadata.channel_element == list(range(0, 64))
    assert b.metadata.channel_element == list(range(64, 128))
    assert b.metadata.sub_sequence == 1
    # Second half sits on the other side of the array centre.
    assert a.rx_positions()[:, 1].max() < 0 < b.rx_positions()[:, 1].min()


# --- delay and amplitude against analytic values ----------------------------------

@pytest.mark.parametrize("name, target", [
    ("l12_5d", (0.10, 0.004, 0.0)),
    ("p4_2v", (0.10, -0.02, 0.001)),
    ("sas_rail", (0.002, 0.094, 0.034)),
])
def test_gaussian_echo_delay_and_amplitude(name, target):
    c = 1480.0
    sonar = small(name, 0.08, 0.12)
    env = {"sound_speed": c, "temperature": 20.0, "salinity": 0.0}
    scene = Scene().add_points([target], amplitude=0.5)
    rec = simulate_ping(sonar, make_pose(), scene, environment=env, adc=False)
    x = rec.samples.array
    tx = np.array(rec.tx_position())
    rx = rec.rx_positions()
    s = np.array(target)
    r_out = np.linalg.norm(s - tx)
    r_back = np.linalg.norm(s - rx, axis=1)
    half = 0.5 * sonar.waveform_config["duration"]  # the Gaussian peaks mid-pulse
    alpha = float(acoustics.absorption(sonar.center_frequency, 20.0, 0.0))
    axes_ok = 0
    for ch in range(0, rx.shape[0], 7):
        t_peak, a_peak = peak_time(x[ch], rec.samples.sample_rate, rec.samples.start_time)
        t_true = (r_out + r_back[ch]) / c + half
        # Delay to a hundredth of a carrier period.
        assert abs(t_peak - t_true) < 0.01 / sonar.center_frequency
        # Amplitude: spreading, absorption and both directivities.
        d_tx = _directivity(rec.geometry.tx, s - tx)
        d_rx = _directivity(rec.geometry.rx[ch], s - rx[ch])
        expected = 0.5 / (r_out * r_back[ch]) * 10 ** (-alpha * (r_out + r_back[ch]) / 20) * d_tx * d_rx
        assert a_peak == pytest.approx(expected, rel=5e-3)
        axes_ok += 1
    assert axes_ok >= 9


def _directivity(element, direction):
    """Independent re-statement of the element pattern (rectangular aperture)."""
    b = np.array(element.boresight)
    ref = np.array([0.0, 0.0, 1.0])
    if abs(ref @ b) > 0.999:
        ref = np.array([1.0, 0.0, 0.0])
        az = ref - b * (ref @ b)
        az /= np.linalg.norm(az)
        el = np.cross(b, az)
    else:
        el = ref - b * (ref @ b)
        el /= np.linalg.norm(el)
        az = np.cross(el, b)
    d = direction / np.linalg.norm(direction)
    if np.isnan(element.beamwidth_az) and np.isnan(element.beamwidth_el):
        return 1.0
    if d @ b <= 0:
        return 0.0
    out = 1.0
    if not np.isnan(element.beamwidth_az):
        out *= np.sinc(0.886 * (d @ az) / element.beamwidth_az)
    if not np.isnan(element.beamwidth_el):
        out *= np.sinc(0.886 * (d @ el) / element.beamwidth_el)
    return out


def test_flank_echo_level_follows_the_sonar_equation():
    """EL = SL - 2 TL + TS, then the receive chain, for a target at 1 km."""
    sonar = small("flank_32", 900.0, 1100.0, transmit={"waveform": {"duration": 0.05}})
    env = {"sound_speed": 1500.0, "temperature": 10.0, "salinity": 35.0, "depth": 20.0}
    ts_db = 10.0
    target = np.array([0.0, 1000.0, 0.0])
    scene = Scene().add_points([target], target_strength_db=ts_db)
    rec = simulate_ping(sonar, make_pose(), scene, environment=env, adc=False)
    x = rec.samples.array
    # RMS-free estimate of the echo's peak amplitude from the analytic envelope plateau.
    env_amp = np.abs(hilbert(x[16]))
    plateau = np.sort(env_amp)[-200:].mean()
    r = np.linalg.norm(target - np.array(rec.tx_position())), np.linalg.norm(target - rec.rx_positions()[16])
    alpha = float(acoustics.absorption(8e3, 10.0, 35.0, 20.0))
    tl = sum(20 * np.log10(ri) + alpha * ri for ri in r)
    el_db = 200.0 - tl + ts_db                       # dB re 1 uPa at the hydrophone
    volts_db = el_db + (-180.0)                      # dB re 1 V
    # Absorption varies across the 6-10 kHz sweep (about +-0.5 dB two-way at 1 km here),
    # so compare with a tolerance that covers the band.
    assert 20 * np.log10(plateau) == pytest.approx(volts_db, abs=0.7)


@pytest.mark.parametrize("name", ["flank_32", "fls_300k", "sas_auv", "popoto_8"])
def test_lfm_echo_delay_by_matched_filter(name):
    c = 1500.0
    sonar_full = SonarModel.load(name)
    dur = min(sonar_full.waveform_config["duration"], 0.02)
    r0 = {"flank_32": 300.0, "fls_300k": 20.0, "sas_auv": 20.0, "popoto_8": 100.0}[name]
    sonar = small(name, r0 - 5.0, r0 + 5.0, transmit={"waveform": {"duration": dur}},
                  receive={"kind": "baseband"})
    target = sonar.boresight * r0 + np.array([0.3, 0.0, 0.0])
    scene = Scene().add_points([target])
    rec = simulate_ping(sonar, make_pose(), scene, environment={"sound_speed": c},
                        absorption_model="none", adc=False)
    fs, fc = rec.samples.sample_rate, rec.samples.center_frequency
    w = sonar.waveform_config
    n = int(round(dur * fs))
    u = np.arange(n) / fs
    k = (w["f_stop"] - w["f_start"]) / dur
    replica = np.exp(2j * np.pi * ((w["f_start"] - fc) * u + 0.5 * k * u**2))
    tx = np.array(rec.tx_position())
    rx = rec.rx_positions()
    for ch in (0, rx.shape[0] - 1):
        mf = np.correlate(rec.samples.array[ch], replica, mode="valid")
        t_peak, _ = peak_time(mf, fs, rec.samples.start_time)
        t_true = (np.linalg.norm(target - tx) + np.linalg.norm(target - rx[ch])) / c
        # Baseband mixing is referenced to t = 0, so the compressed peak sits at the delay.
        assert abs(t_peak - t_true) < 0.02 / sonar.bandwidth


# --- independent frequency-domain oracle ----------------------------------------------

@pytest.mark.parametrize("kind", ["real", "baseband"])
def test_time_domain_synthesis_matches_frequency_domain_oracle(kind):
    """Static geometry: sum of delayed pulses, built by FFT phase ramps instead of
    by evaluating the pulse at shifted times."""
    c = 1480.0
    sonar = small("p4_2v", 0.05, 0.09, receive={"kind": kind})
    rng = np.random.default_rng(3)
    pts = np.column_stack([rng.uniform(0.06, 0.08, 6), rng.uniform(-0.01, 0.01, 6), np.zeros(6)])
    amps = rng.uniform(0.5, 1.0, 6) * np.exp(1j * rng.uniform(0, 2 * np.pi, 6))
    scene = Scene().add_points(pts, amplitude=amps)
    rec = simulate_ping(sonar, make_pose(), scene, environment={"sound_speed": c},
                        absorption_model="none", adc=False)
    x = rec.samples.array
    fs, t0, n = rec.samples.sample_rate, rec.samples.start_time, rec.samples.n_samples
    w = sonar.waveform_config
    fc, bw, dur = sonar.center_frequency, sonar.bandwidth, w["duration"]
    sigma = np.sqrt(2 * np.log(2)) / (np.pi * bw)
    # Reference pulse sampled on the record grid, centred at t_ref inside the window.
    t = t0 + np.arange(n) / fs
    t_ref = t[n // 2]
    pulse = np.exp(-0.5 * ((t - t_ref) / sigma) ** 2) * np.exp(2j * np.pi * fc * (t - t_ref))
    spectrum = np.fft.fft(pulse)
    f = np.fft.fftfreq(n, 1 / fs)
    tx = np.array(rec.tx_position())
    rx = rec.rx_positions()
    for ch in (0, 31, 63):
        acc = np.zeros(n, dtype=complex)
        for p, a in zip(pts, amps):
            r1, r2 = np.linalg.norm(p - tx), np.linalg.norm(p - rx[ch])
            tau = (r1 + r2) / c + 0.5 * dur          # pulse centre arrival
            d_tx = _directivity(rec.geometry.tx, p - tx)
            d_rx = _directivity(rec.geometry.rx[ch], p - rx[ch])
            acc += a * d_tx * d_rx / (r1 * r2) * np.fft.ifft(
                spectrum * np.exp(-2j * np.pi * f * (tau - t_ref)))
        oracle = acc.real if kind == "real" else acc * np.exp(-2j * np.pi * fc * t)
        err = np.linalg.norm(x[ch] - oracle) / np.linalg.norm(oracle)
        # The kernel truncates the Gaussian at +-duration/2 (about 3.5 sigma here).
        assert err < 5e-3, err


# --- motion ------------------------------------------------------------------------------

def test_receiver_moves_during_reception():
    """The echo arrives when the *moved* receiver meets it, not where it was at transmit."""
    c, v = 1500.0, np.array([0.0, 30.0, 0.0])  # fast closing speed to make the effect large
    sonar = small("flank_32", 950.0, 1050.0, transmit={"waveform": {
        "type": "gaussian", "f_start": 7e3, "f_stop": 9e3, "duration": 4e-3}})
    target = np.array([0.0, 1000.0, 0.0])
    rec = simulate_ping(sonar, make_pose(velocity=v), Scene().add_points([target]),
                        environment={"sound_speed": c}, absorption_model="none", adc=False)
    ch = 5
    tx0 = np.array(rec.tx_position())
    rx0 = rec.rx_positions()[ch]
    u = 0.5 * sonar.waveform_config["duration"]        # emission time of the pulse centre

    def residual(t):
        return (np.linalg.norm(target - (tx0 + v * u)) + np.linalg.norm(target - (rx0 + v * t))) / c - (t - u)

    t_true = brentq(residual, 1.0, 1.5, xtol=1e-13)
    t_stop_and_hop = u + (np.linalg.norm(target - tx0) + np.linalg.norm(target - rx0)) / c
    t_peak, _ = peak_time(rec.samples.array[ch], rec.samples.sample_rate, rec.samples.start_time)
    assert abs(t_peak - t_true) < 2e-7
    assert abs(t_true - t_stop_and_hop) > 1e-2      # the two models differ by ~27 ms here


def test_doppler_shift_on_a_long_cw_pulse():
    c, speed, f0 = 1500.0, 5.0, 8000.0
    sonar = small("flank_32", 480.0, 560.0, receive={"kind": "baseband"}, transmit={"waveform": {
        "type": "cw", "f_start": f0, "f_stop": f0, "duration": 0.05, "window_alpha": 0.0}})
    target = np.array([0.0, 500.0, 0.0])
    scene = Scene().add_points([target], velocity=(0.0, -speed, 0.0))  # closing
    rec = simulate_ping(sonar, make_pose(), scene, environment={"sound_speed": c},
                        absorption_model="none", adc=False)
    x = rec.samples.array[16]
    fs = rec.samples.sample_rate
    idx = np.flatnonzero(np.abs(x) > 0.5 * np.abs(x).max())
    seg = x[idx[50]:idx[-50]]
    shift = np.angle(np.sum(seg[1:] * np.conj(seg[:-1]))) * fs / (2 * np.pi)
    expected = f0 * ((c + speed) / (c - speed) - 1.0)   # two-way, moving reflector
    assert shift == pytest.approx(expected, rel=2e-3)
    assert expected == pytest.approx(53.5, abs=0.1)


# --- multipath, clocks, ADC ---------------------------------------------------------------

def test_surface_image_arrives_inverted_at_the_image_delay():
    c = 1500.0
    # Projector co-located with the array so the two single-bounce paths coincide
    # instead of beating against each other.
    sonar = small("popoto_8", 95.0, 110.0, transmit={"lever_arm": [0, 0, 0], "waveform": {
        "type": "gaussian", "f_start": 20e3, "f_stop": 35e3, "duration": 4e-4}})
    pose = make_pose(position=(0.0, 0.0, 5.0))           # 5 m below the surface
    target = np.array([0.0, 100.0, 8.0])
    provider = ImageSourceProvider(c, surface_z=0.0, bottom_z=None, max_order=1)
    kw = dict(environment={"sound_speed": c}, absorption_model="none", adc=False)
    rec = simulate_ping(sonar, pose, Scene().add_points([target]), provider=provider, **kw)
    direct = simulate_ping(sonar, pose, Scene().add_points([target]), **kw)
    ch = 4
    x_direct = direct.samples.array[ch]
    multipath = rec.samples.array[ch] - x_direct
    tx = np.array(rec.tx_position())
    rx = rec.rx_positions()[ch]
    image = target * np.array([1, 1, -1])
    fs, t0 = rec.samples.sample_rate, rec.samples.start_time
    half = 0.5 * sonar.waveform_config["duration"]
    r_d_tx, r_d_rx = np.linalg.norm(target - tx), np.linalg.norm(target - rx)
    r_i_tx, r_i_rx = np.linalg.norm(image - tx), np.linalg.norm(image - rx)
    # Three extra arrivals: out direct / back via surface, the reverse, and both via surface.
    t_single = np.array([r_d_tx + r_i_rx, r_i_tx + r_d_rx]) / c + half
    t_double = (r_i_tx + r_i_rx) / c + half
    env = np.abs(hilbert(multipath))
    # Single-bounce echoes: two of them nearly coincide, each -1/(R_direct R_image).
    i_s = int(round((t_single.mean() - t0) * fs))
    expected_single = 1.0 / (r_d_tx * r_i_rx) + 1.0 / (r_i_tx * r_d_rx)
    assert env[i_s - 3:i_s + 4].max() == pytest.approx(expected_single * sonar.amplitude_at_1m(), rel=0.05)
    # Double bounce: (-1)^2, at the image-to-image delay.
    i_dd = int(round((t_double - t0) * fs))
    assert env[i_dd - 3:i_dd + 4].max() == pytest.approx(
        sonar.amplitude_at_1m() / (r_i_tx * r_i_rx), rel=0.05)
    # Pressure-release surface: a single bounce flips the sign. Compare carrier phase at
    # each echo's own centre, where a Gaussian pulse has zero phase.
    def phase_at(x, t):
        a = hilbert(x)
        i = (t - t0) * fs
        j = int(np.floor(i))
        v = a[j] * (1 - (i - j)) + a[j + 1] * (i - j)
        return np.angle(v)
    assert abs(phase_at(x_direct, (r_d_tx + r_d_rx) / c + half)) < 0.1
    assert abs(abs(phase_at(multipath, t_double)) - 0.0) < 0.15
    # The two single-bounce echoes are 180 degrees from the direct one.
    mid = t_single.mean()
    carrier_advance = 2 * np.pi * sonar.center_frequency * (mid - t_single)
    assert np.all(np.abs(carrier_advance) < 0.3)   # they overlap almost exactly here
    assert abs(abs(phase_at(multipath, mid)) - np.pi) < 0.3


def test_board_clock_offset_shifts_that_boards_samples():
    c = 1500.0
    base = {"transmit": {"waveform": {"type": "gaussian", "f_start": 20e3, "f_stop": 35e3,
                                      "duration": 4e-4}}}
    sonar = small("popoto_4x8", 95.0, 105.0, **base)
    target = np.array([0.0, 100.0, 0.0])
    rec = simulate_ping(sonar, make_pose(), Scene().add_points([target]),
                        environment={"sound_speed": c}, absorption_model="none", adc=False)
    # The processor-facing record does not carry the true clock errors.
    assert all(b.clock_offset == 0.0 and b.clock_drift_ppm == 0.0 for b in rec.metadata.boards)
    tx = np.array(rec.tx_position())
    rx = rec.rx_positions()
    fs, t0 = rec.samples.sample_rate, rec.samples.start_time
    half = 2e-4
    for board, cfg in enumerate(sonar.boards):
        ch = 8 * board + 3
        t_true = (np.linalg.norm(target - tx) + np.linalg.norm(target - rx[ch])) / c + half
        # A board whose clock runs `offset` late and `drift` fast timestamps the echo early.
        n_true = (t_true - t0 - cfg["clock_offset"]) * fs * (1 + cfg["clock_drift_ppm"] * 1e-6)
        t_peak, _ = peak_time(rec.samples.array[ch], fs, t0)
        assert abs((t_peak - t0) * fs - n_true) < 0.02


def test_adc_quantises_and_clips():
    sonar = small("l12_5d", 0.02, 0.05)
    scene = Scene().add_points([[0.03, 0.0, 0.0]], amplitude=5e-4)   # echo of ~0.5: within range
    rec = simulate_ping(sonar, make_pose(), scene, environment={"sound_speed": 1480.0})
    x = rec.samples.array
    step = 1.0 / 2048
    assert rec.samples.storage == gs.StorageType.Int16
    assert rec.samples.storage_scale == pytest.approx(step)
    np.testing.assert_allclose(x / step, np.rint(x / step), atol=1e-4)
    assert 0.2 < np.abs(x).max() < 1.0
    loud = Scene().add_points([[0.03, 0.0, 0.0]], amplitude=1e-2)     # ~10x full scale
    rec = simulate_ping(sonar, make_pose(), loud, environment={"sound_speed": 1480.0})
    x = rec.samples.array
    assert x.max() == pytest.approx(2047 * step) and x.min() == pytest.approx(-1.0)
    assert gs.validate(rec).warning_count > 0     # the validator notices the clipping


def test_noise_level():
    sonar = small("popoto_8", 100.0, 400.0)
    rec = simulate_ping(sonar, make_pose(), Scene(), noise_rms=3e-4, adc=False,
                        rng=np.random.default_rng(0))
    assert rec.samples.array.std() == pytest.approx(3e-4, rel=0.01)


# --- scenario runner -------------------------------------------------------------------------

def test_scenario_writes_records_and_ground_truth(tmp_path):
    cfg = {
        "name": "t", "sonar": "p4_2v", "fidelity": "free_field",
        "sonar_overrides": {"receive": {"window": {"range": [0.05, 0.08]}}},
        "environment": {"temperature": 20.0, "salinity": 0.0},
        "platform": {"position": [0, 0, 0.5], "velocity": [0.0, 0.01, 0.0]},
        "pings": {"count": 3, "interval": 0.1},
        "scene": [{"type": "polar_grid", "origin": [0, 0, 0.5], "forward": [1, 0, 0],
                   "across": [0, 1, 0], "ranges": [0.06, 0.07], "bearings_deg": [-10, 0, 10]}],
    }
    path = str(tmp_path / "s.h5")
    result = run_scenario(cfg, output=path)
    assert len(result.records) == 3 and len(result.scene) == 6
    assert gs.validate_file(path).ok, str(gs.validate_file(path))
    reader = gs.RecordReader(path)
    assert reader.read_truth("scatterer_position").shape == (6, 3)
    np.testing.assert_allclose(reader.read_truth("platform_position")[:, 1], [0.0, 0.001, 0.002])
    assert reader.read_truth_text("fidelity") == "free_field"
    assert "sonar: p4_2v" in reader.info.config_yaml
    rec = reader.read(2)
    assert rec.environment.sound_speed == pytest.approx(acoustics.sound_speed(20.0, 0.0), rel=1e-6)
    assert np.abs(rec.samples.array).max() > 0


# --- CPU against GPU ---------------------------------------------------------------------------

@pytest.mark.skipif(not gs.cuda_available(), reason="no CUDA device: GPU forward model not run")
def test_gpu_forward_model_matches_cpu():
    sonar = small("p4_2v", 0.05, 0.09)
    rng = np.random.default_rng(5)
    pts = np.column_stack([rng.uniform(0.06, 0.08, 200), rng.uniform(-0.02, 0.02, 200), np.zeros(200)])
    scene = Scene().add_points(pts, amplitude=rng.uniform(0.5, 1.0, 200))
    kw = dict(environment={"sound_speed": 1480.0}, adc=False)
    cpu = simulate_ping(sonar, make_pose(), scene, backend=gs.core.Backend.Host, **kw).samples.array
    gpu = simulate_ping(sonar, make_pose(), scene, backend=gs.core.Backend.Cuda, **kw).samples.array
    # Same arithmetic; only the summation order differs (atomic adds).
    assert np.abs(cpu - gpu).max() < 1e-5 * np.abs(cpu).max()


# --- built-in scenarios: a point-target grid for each sonar, at full size ------------------

@pytest.mark.parametrize("name", ALL_SONARS)
def test_point_grid_scenario_runs_end_to_end(name):
    from gpusonar.config import load_scenario_config
    cfg = load_scenario_config(f"point_grid_{name}")
    assert cfg["fidelity"] == "free_field"
    result = run_scenario(cfg)
    rec = result.records[0]
    assert gs.validate(rec).error_count == 0, str(gs.validate(rec))
    x = rec.samples.array
    # Every grid range produces an echo: energy appears around each expected delay.
    c = rec.environment.sound_speed
    fs, t0 = rec.samples.sample_rate, rec.samples.start_time
    ranges = cfg["scene"][0]["ranges"]
    ch = x.shape[0] // 2
    dur = rec.transmit.waveform.duration
    for r in ranges:
        i0 = int((2 * r / c - t0) * fs)
        i1 = int((2 * r / c + dur - t0) * fs) + 8
        assert np.abs(x[ch, max(i0 - 64, 0):i1 + 64]).max() > 0, f"no echo near {r} m"


def test_hardware_gain_ramp_is_applied_and_recorded():
    over = {"receive": {"tvg": {"time": [0.0, 0.04], "gain_db": [0.0, 40.0]}}}
    sonar = small("fls_300k", 5.0, 25.0, **over)
    flat = small("fls_300k", 5.0, 25.0, receive={"tvg": None})
    scene = Scene().add_points([[10.0, 0, 0], [20.0, 0, 0]])
    kw = dict(environment={"sound_speed": 1500.0}, absorption_model="none", adc=False)
    a = simulate_ping(sonar, make_pose(), scene, **kw)
    b = simulate_ping(flat, make_pose(), scene, **kw)
    assert a.metadata.tvg_time == [0.0, 0.04] and a.metadata.tvg_gain_db == [0.0, 40.0]
    assert list(b.metadata.tvg_time) == []
    t = a.samples.start_time + np.arange(a.samples.n_samples) / a.samples.sample_rate
    gain = 10 ** (np.interp(t, [0.0, 0.04], [0.0, 40.0]) / 20)
    np.testing.assert_allclose(a.samples.array, b.samples.array * gain, rtol=1e-5, atol=1e-12)


def test_phase_3_sonar_images_its_point_grid():
    """`fls_750k` (the plan's 0.5-1 MHz mid-range step): the grid scenario runs, every
    target is found within a range cell and a beamwidth, and absorption is what limits it."""
    from gpusonar import acoustics, recon
    from gpusonar.config import load_scenario_config
    cfg = load_scenario_config("point_grid_fls_750k")
    cfg["scene"][0].update(ranges=[5, 30, 45], bearings_deg=[-50, 0, 25])
    result = run_scenario(cfg)
    rec = result.records[0]
    assert gs.validate(rec).error_count == 0
    centre = rec.rx_positions().mean(axis=0)
    for target in result.scene.positions:
        offset = target - centre
        rng_m, bearing = np.linalg.norm(offset), np.arctan2(offset[1], offset[0])
        ranges = rng_m + np.arange(-0.05, 0.0501, 0.001)
        bearings = bearing + np.radians(np.arange(-3.0, 3.01, 0.1))
        img = np.abs(recon.fan_image([rec.copy()], ranges, bearings, interp="linear").image)
        ir, ib = np.unravel_index(np.argmax(img), img.shape)
        assert abs(ranges[ir] - rng_m) < 0.006                      # c / 2B = 3.75 mm
        assert abs(np.degrees(bearings[ib] - bearing)) < 1.0        # beam about 1.6 degrees at broadside
    alpha = float(acoustics.absorption(750e3, 12.0, 35.0, 5.0))
    assert 0.15 < alpha < 0.30                                      # dB/m: 15 to 30 dB two-way at 50 m
