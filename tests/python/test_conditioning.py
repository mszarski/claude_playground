"""Milestone 6: conditioning, pulse compression and post-processing.

Acceptance: whitening flattens a known transfer function; TVG flattens intensity over
range; DRC hits the target median brightness; chirp compression gain matches
10 log10(BT) within 1 dB.
"""

import numpy as np
import pytest

import gpusonar as gs
from gpusonar import processing
from gpusonar.sim import Scene, make_pose, simulate_ping
from gpusonar.sonar import SonarModel

core = gs.core


def noise_record(n_channels=64, n_samples=8192, fs=20e3, fc=8e3, seed=0, band=(6e3, 10e3), duration=0.1):
    """A baseband record of complex white noise with an LFM transmit description."""
    rng = np.random.default_rng(seed)
    r = core.PingRecord()
    s = r.samples
    s.kind = core.SampleKind.ComplexBaseband
    s.array = ((rng.standard_normal((n_channels, n_samples)) + 1j * rng.standard_normal((n_channels, n_samples)))
               / np.sqrt(2)).astype(np.complex64)
    s.sample_rate, s.center_frequency, s.start_time = fs, fc, 0.05
    r.samples = s
    w = r.transmit.waveform
    w.type, w.f_start, w.f_stop, w.duration = core.WaveformType.Lfm, band[0], band[1], duration
    t = r.transmit
    t.waveform = w
    r.transmit = t
    geo = r.geometry
    geo.rx = [core.Element() for _ in range(n_channels)]
    r.geometry = geo
    r.set_default_channel_tables()
    return r


def set_samples(rec, x):
    s = rec.samples
    kind = s.kind
    s.array = x.astype(np.complex64)
    s.kind = kind
    rec.samples = s


# --- pulse compression ----------------------------------------------------------------------

@pytest.mark.parametrize("B, T", [(4e3, 0.25), (4e3, 0.05), (15e3, 0.02)])
def test_chirp_compression_gain_is_bt(B, T):
    """SNR gain of the matched filter, measured, against 10 log10(BT)."""
    fs = 2.5 * B
    n = int(fs * (T + 0.2))
    rec = noise_record(n_channels=32, n_samples=n, fs=fs, fc=8e3, band=(8e3 - B / 2, 8e3 + B / 2), duration=T)
    noise = rec.samples.array.copy()                 # unit variance per complex sample, bandwidth fs
    # A unit-amplitude echo at delay 0.1 s.
    tau, t0 = 0.1, rec.samples.start_time
    t = t0 + np.arange(n) / fs
    u = t - tau
    k = B / T
    inside = (u >= 0) & (u <= T)
    echo = np.where(inside, np.exp(2j * np.pi * ((-B / 2) * u + 0.5 * k * u**2)) * np.exp(-2j * np.pi * 8e3 * tau), 0)
    amp = 0.2
    set_samples(rec, noise + amp * echo)
    snr_in_band = amp**2 / (1.0 * B / fs)            # signal power over the noise power inside B
    core.pulse_compress(rec)
    y = rec.samples.array
    i_peak = int(round((tau - t0) * fs))
    peak = np.abs(y[:, i_peak]).mean()
    assert peak == pytest.approx(amp, rel=0.05)      # amplitude preserved
    far = np.r_[0:i_peak - int(1.2 * T * fs) if i_peak > int(1.2 * T * fs) else 0, i_peak + int(0.05 * fs):n - int(T * fs)]
    noise_out = np.mean(np.abs(y[:, far]) ** 2)
    gain_db = 10 * np.log10((amp**2 / noise_out) / snr_in_band)
    assert gain_db == pytest.approx(10 * np.log10(B * T), abs=1.0)


def test_replica_window_trades_gain_for_sidelobes():
    from gpusonar.metrics import peak_sidelobe_db
    B, T, fs = 4e3, 0.1, 16e3
    n = int(fs * 0.4)
    base = noise_record(n_channels=1, n_samples=n, fs=fs, band=(6e3, 10e3), duration=T)
    t = base.samples.start_time + np.arange(n) / fs
    u = t - 0.15
    echo = np.where((u >= 0) & (u <= T), np.exp(2j * np.pi * (-B / 2 * u + 0.5 * B / T * u**2)), 0)[None, :]
    out = {}
    for window in ("none", "hamming"):
        rec = noise_record(n_channels=1, n_samples=n, fs=fs, band=(6e3, 10e3), duration=T)
        set_samples(rec, echo)
        core.pulse_compress(rec, window=window)
        out[window] = np.abs(rec.samples.array[0])
    assert peak_sidelobe_db(out["none"]) == pytest.approx(-13.3, abs=1.0)     # unweighted LFM
    assert peak_sidelobe_db(out["hamming"]) < -30.0
    assert out["hamming"].max() == pytest.approx(out["none"].max(), rel=0.02)  # normalised to echo amplitude


def test_doppler_bank_finds_the_range_rate_of_a_long_pulse():
    c, v_true = 1500.0, -6.0                      # closing at 6 m/s
    sonar = SonarModel.load("flank_32", {"receive": {"window": {"range": [1400.0, 1600.0]}, "kind": "baseband"}})
    target = np.array([0.0, 1500.0, 0.0])
    scene = Scene().add_points([target], velocity=(0.0, v_true, 0.0))
    rec = simulate_ping(sonar, make_pose(), scene, environment={"sound_speed": c},
                        absorption_model="none", adc=False)
    rates = np.arange(-10.0, 10.5, 1.0)
    best, v_best, peaks = processing.doppler_bank(rec, rates)
    assert v_best == pytest.approx(v_true, abs=1.0)
    # A 1 s, 4 kHz sweep at 8 kHz is not Doppler tolerant at this speed: the zero-Doppler
    # replica loses most of the peak, which is why the bank exists.
    zero = peaks[np.argmin(np.abs(rates))]
    assert zero < 0.5 * peaks.max()
    # Amplitude is restored to the echo's own level by the right hypothesis.
    r = np.linalg.norm(target)
    expected = sonar.amplitude_at_1m() / r**2
    assert peaks.max() == pytest.approx(expected, rel=0.1)


# --- calibration and deterministic TVG -----------------------------------------------------

def test_calibration_removes_injected_gain_phase_and_delay():
    rec = noise_record(n_channels=8, n_samples=4096, seed=3)
    # Band-limit the noise so a fractional delay is well defined.
    x = rec.samples.array
    X = np.fft.fft(x, axis=1)
    f = np.fft.fftfreq(x.shape[1], 1 / rec.samples.sample_rate)
    X[:, np.abs(f) > 3e3] = 0
    clean = np.fft.ifft(X, axis=1)
    rng = np.random.default_rng(1)
    gain = rng.uniform(0.5, 2.0, 8)
    phase = rng.uniform(-np.pi, np.pi, 8)
    delay = rng.uniform(-2e-4, 2e-4, 8)              # up to 4 samples either way
    fc, fs = rec.samples.center_frequency, rec.samples.sample_rate
    # A channel that is late by `delay` records x(t - delay).
    late = np.fft.ifft(np.fft.fft(clean, axis=1) * np.exp(-2j * np.pi * (f[None, :] + fc) * delay[:, None]), axis=1)
    set_samples(rec, late * (gain * np.exp(1j * phase))[:, None])
    cal = rec.calibration
    cal.gain, cal.phase, cal.delay = gain.tolist(), phase.tolist(), delay.tolist()
    rec.calibration = cal
    core.apply_calibration(rec)
    err = np.abs(rec.samples.array - clean).max() / np.abs(clean).max()
    assert err < 1e-4
    # Applied once: the tables are neutral afterwards.
    assert rec.calibration.gain == [1.0] * 8 and rec.calibration.delay == [0.0] * 8


def test_digital_tvg_completes_the_hardware_ramp():
    rec = noise_record(n_channels=2, n_samples=4000, fs=20e3)
    env = rec.environment
    env.sound_speed = 1500.0
    rec.environment = env
    t = rec.samples.start_time + np.arange(4000) / 20e3
    r = 0.5 * 1500.0 * t
    hw_db = np.interp(t, [0.05, 0.25], [0.0, 30.0])
    set_samples(rec, np.ones((2, 4000)) / r**2 * 10 ** (hw_db / 20))     # 40 log R decay, then the analogue ramp
    meta = rec.metadata
    meta.tvg_time, meta.tvg_gain_db = [0.05, 0.25], [0.0, 30.0]
    rec.metadata = meta
    core.apply_tvg(rec, spreading_db_per_decade=40.0)
    y = np.abs(rec.samples.array[0])
    np.testing.assert_allclose(y, 1.0, rtol=1e-4)       # flat: hardware plus digital equals 40 log R
    assert list(rec.metadata.tvg_time) == []            # the ramp is accounted for


def test_remove_dc():
    sonar = SonarModel.load("popoto_8", {"receive": {"window": {"range": [100.0, 130.0]}}})
    rec = simulate_ping(sonar, make_pose(), Scene(), noise_rms=1e-3, adc=False, rng=np.random.default_rng(0))
    s = rec.samples
    s.array = rec.samples.array + 0.25
    rec.samples = s
    core.remove_dc(rec)
    assert np.abs(rec.samples.array.mean(axis=1)).max() < 1e-6


# --- ASASIN Eq. 9: whitening -------------------------------------------------------------------

def shaped_noise(transfer_db, seed=0):
    """White noise through a known in-band transfer function (given in dB against f in Hz)."""
    rec = noise_record(n_channels=64, n_samples=8192, seed=seed)
    x = rec.samples.array
    f = np.fft.fftfreq(x.shape[1], 1 / rec.samples.sample_rate)
    H = 10 ** (transfer_db(f) / 20)
    set_samples(rec, np.fft.ifft(np.fft.fft(x, axis=1) * H, axis=1))
    return rec, f


def band_spectrum_db(rec, f, f_lo, f_hi, bins=16):
    P = (np.abs(np.fft.fft(rec.samples.array, axis=1)) ** 2).mean(axis=0)
    edges = np.linspace(f_lo, f_hi, bins + 1)
    return np.array([10 * np.log10(P[(f >= a) & (f < b)].mean()) for a, b in zip(edges[:-1], edges[1:])])


def test_whitening_flattens_a_known_transfer_function():
    # In-band (|f| < 2 kHz baseband): a 12 dB tilt plus a 6 dB peak-to-peak ripple.
    # Out of band: 40 dB down, as after a band-pass.
    def transfer(f):
        inband = 6.0 * f / 2e3 + 3.0 * np.sin(2 * np.pi * f / 700.0)
        return np.where(np.abs(f) < 2e3, inband, -40.0)

    rec, f = shaped_noise(transfer)
    before = band_spectrum_db(rec, f, -1.8e3, 1.8e3)
    out_of_band_before = band_spectrum_db(rec, f, 4e3, 8e3, bins=4)
    assert np.ptp(before) > 12.0
    gain = core.whiten([rec], gamma=0.01)
    after = band_spectrum_db(rec, f, -1.8e3, 1.8e3)
    assert np.ptp(after) < 1.0                    # flat to within 1 dB where there is signal
    # h: the largest gain is exactly 1 (0 dB minimum attenuation), so nothing is amplified.
    assert gain.max() == pytest.approx(1.0)
    assert gain.min() > 0.0
    # Out of band the gain saturates at that maximum: the noise floor there is passed
    # through unchanged instead of being blown up, and the band still stands above it.
    out_of_band = band_spectrum_db(rec, f, 4e3, 8e3, bins=4)
    np.testing.assert_allclose(out_of_band, out_of_band_before, atol=0.2)
    assert out_of_band.max() < after.min() - 10.0


def test_whitening_gamma_sets_how_far_weak_bands_are_raised():
    def transfer(f):
        return np.where(np.abs(f) < 2e3, -20.0 * (np.abs(f) / 2e3), -60.0)

    spans = {}
    for gamma in (0.001, 1.0):
        rec, f = shaped_noise(transfer, seed=2)
        core.whiten([rec], gamma=gamma)
        spans[gamma] = np.ptp(band_spectrum_db(rec, f, -1.8e3, 1.8e3))
    assert spans[0.001] < 2.0                    # small gamma: fully flattened
    assert spans[1.0] > 3.0                      # large gamma: weak edges left partly down
    assert spans[1.0] > 2.0 * spans[0.001]


def test_whitening_literal_amplitude_reading_does_not_flatten():
    """Why G is applied as a power gain (DECISIONS.md D-20)."""
    def transfer(f):
        return np.where(np.abs(f) < 2e3, 6.0 * f / 2e3, -40.0)

    rec, f = shaped_noise(transfer, seed=4)
    core.whiten([rec], gamma=0.01, power_gain=False)
    after = band_spectrum_db(rec, f, -1.8e3, 1.8e3)
    assert np.ptp(after) > 8.0         # the tilt comes out inverted, not removed
    assert after[0] > after[-1]        # (it rose with frequency going in)


# --- ASASIN Eq. 10: median time-varying gain ---------------------------------------------------

def reverberation(seed=0, n_channels=64, n_samples=6000):
    """Noise-like returns whose intensity falls by 30 dB across the record."""
    rec = noise_record(n_channels=n_channels, n_samples=n_samples, seed=seed)
    decay = 10 ** (-30.0 * np.arange(n_samples) / n_samples / 20)
    set_samples(rec, rec.samples.array * decay[None, :])
    return rec


def intensity_db(rec, bins=20):
    p = (np.abs(rec.samples.array) ** 2).mean(axis=0)
    return 10 * np.log10(p.reshape(bins, -1).mean(axis=1))


def test_median_tvg_flattens_intensity_over_range():
    rec = reverberation()
    assert np.ptp(intensity_db(rec)) > 25.0
    gain = core.median_tvg([rec], alpha=1e-4, smooth_samples=64)
    after = intensity_db(rec)
    assert np.ptp(after) < 1.0
    assert gain.shape == (rec.samples.n_samples,)


def test_median_tvg_ignores_bright_scatterers():
    """The paper's reason for the median: a few strong returns must not dent the gain curve."""
    plain = reverberation(seed=5)
    bright = reverberation(seed=5)
    x = bright.samples.array
    x[:6, 3000:3040] *= 1000.0                    # a 60 dB target on 6 of 64 channels
    set_samples(bright, x)
    x_plain = plain.samples.array.copy()          # before the gain is applied in place
    g_plain = core.median_tvg([plain], alpha=1e-4)
    g_bright = core.median_tvg([bright], alpha=1e-4)
    ratio_db = 10 * np.log10(g_bright[3000:3040] / g_plain[3000:3040])
    # Six outliers among 64 channels nudge the median by a rank or two: a dB or so.
    assert np.abs(ratio_db).max() < 3.0
    # A mean-based gain would have been notched by tens of dB at the same samples.
    mean_notch_db = 10 * np.log10((np.abs(x[:, 3000:3040]) ** 2).mean() /
                                  (np.abs(x_plain[:, 3000:3040]) ** 2).mean())
    assert mean_notch_db > 40.0


def test_median_tvg_batch_of_pings_shares_one_curve():
    a, b = reverberation(seed=1), reverberation(seed=2)
    gain = core.median_tvg([a, b], alpha=1e-4, smooth_samples=64)
    assert np.ptp(intensity_db(a)) < 1.5 and np.ptp(intensity_db(b)) < 1.5
    assert gain.shape == (6000,)


# --- ASASIN Eq. 11-12: dynamic range compression ----------------------------------------------

@pytest.mark.parametrize("b", [0.1, 0.2, 0.5])
def test_drc_hits_the_target_median(b):
    rng = np.random.default_rng(0)
    # Speckle with an 80 dB dynamic range: Rayleigh background plus a few specular points.
    img = rng.rayleigh(1.0, (300, 400)).astype(np.float32)
    img[rng.integers(0, 300, 20), rng.integers(0, 400, 20)] = 1e4
    out, q = core.drc(img, target_median=b, percentile=100.0)
    assert out.shape == img.shape
    assert np.median(out) == pytest.approx(b, abs=1e-3)
    assert out.min() >= 0.0 and out.max() == pytest.approx(1.0)
    # Eq. 12, evaluated independently.
    med = np.median(img / img.max())
    assert q == pytest.approx((b - b * med) / (med - b * med), rel=1e-3)
    # Eq. 11 is monotonic: ordering of pixels is preserved.
    flat_in, flat_out = img.ravel(), out.ravel()
    order = np.argsort(flat_in)[::997]
    assert np.all(np.diff(flat_out[order]) >= -1e-7)


def test_drc_makes_the_background_visible():
    rng = np.random.default_rng(1)
    img = rng.rayleigh(1.0, (200, 200)).astype(np.float32)
    img[100, 100] = 1e4                            # one specular return 80 dB up
    linear = img / img.max()
    assert np.median(linear) < 1e-3                # displayed as-is: an empty image with one dot
    out, _ = core.drc(img, target_median=0.2)
    assert np.median(out) == pytest.approx(0.2, abs=5e-3)


def test_drc_rejects_a_bad_target():
    with pytest.raises(ValueError):
        core.drc(np.ones((4, 4), dtype=np.float32), target_median=1.0)


# --- scan conversion ----------------------------------------------------------------------------

def test_scan_conversion_puts_a_fan_pixel_at_its_cartesian_position():
    ranges = np.linspace(1.0, 10.0, 181)
    bearings = np.radians(np.linspace(-30, 30, 121))
    img = np.zeros((181, 121))
    r0, b0 = 6.0, np.radians(15.0)
    R, B = np.meshgrid(ranges, bearings, indexing="ij")
    img = np.exp(-((R - r0) / 0.5) ** 2 - ((B - b0) / 0.06) ** 2)
    out, x, y = processing.scan_convert(img, ranges, bearings, nx=400, ny=400)
    iy, ix = np.unravel_index(np.argmax(out), out.shape)
    assert x[ix] == pytest.approx(r0 * np.sin(b0), abs=0.05)
    assert y[iy] == pytest.approx(r0 * np.cos(b0), abs=0.05)
    assert out.max() == pytest.approx(1.0, abs=0.02)
    # Outside the fan is filled, not extrapolated.
    assert out[np.argmin(np.abs(y - 9.0)), 0] == 0.0


# --- conditioning on the device (cuFFT) against the host ------------------------------------

needs_cuda = pytest.mark.skipif(not gs.cuda_available(), reason="no CUDA device")


def _raw_ping(name, overrides=None, target=(1.5, 0.05, 0.0)):
    sonar = SonarModel.load(name, overrides)
    return simulate_ping(sonar, make_pose(), Scene().add_points([list(target)]),
                         environment={"sound_speed": 1480.0}, absorption_model="none", noise_rms=1e-3,
                         rng=np.random.default_rng(8))


@needs_cuda
@pytest.mark.parametrize("name, overrides, target", [
    ("l12_5d", {"receive": {"window": {"range": [1.4, 1.6]}}}, (1.5, 0.05, 0.0)),           # real passband in
    ("flank_32", {"receive": {"window": {"range": [900.0, 1100.0]}}}, (0.0, 1000.0, 0.0)),  # long LFM
    ("sas_auv", {"receive": {"kind": "baseband", "sample_rate": 100e3, "window": {"range": [38.0, 52.0]}}},
     (0.0, 42.0, 15.0)),                                                                     # complex in
])
def test_device_conditioning_matches_the_host(name, overrides, target):
    host = _raw_ping(name, overrides, target)
    dev = host.copy()
    gs.core.to_baseband(host)
    gs.core.to_baseband(dev, backend=gs.core.Backend.Cuda)
    assert dev.samples.array.shape == host.samples.array.shape
    assert dev.samples.sample_rate == host.samples.sample_rate
    assert dev.samples.center_frequency == host.samples.center_frequency
    scale = np.abs(host.samples.array).max()
    assert np.abs(dev.samples.array - host.samples.array).max() < 2e-4 * scale
    gs.core.pulse_compress(host, window="hann")
    gs.core.pulse_compress(dev, window="hann", backend=gs.core.Backend.Cuda)
    scale = np.abs(host.samples.array).max()
    assert np.abs(dev.samples.array - host.samples.array).max() < 2e-4 * scale
    # Repeated calls reuse the cached plans and give the same answer.
    again = _raw_ping(name, overrides, target)
    gs.core.to_baseband(again, backend=gs.core.Backend.Cuda)
    gs.core.pulse_compress(again, window="hann", backend=gs.core.Backend.Cuda)
    np.testing.assert_array_equal(again.samples.array, dev.samples.array)


@needs_cuda
def test_image_from_device_conditioning_matches():
    from gpusonar import recon
    a = [_raw_ping("l12_5d", {"receive": {"window": {"range": [1.4, 1.6]}}})]
    b = [a[0].copy()]
    ranges = np.linspace(1.47, 1.53, 121)
    bearings = np.radians(np.linspace(-2.0, 6.0, 81))
    host = recon.fan_image(a, ranges, bearings, interp="linear").image
    dev = recon.fan_image(b, ranges, bearings, interp="linear", backend=gs.core.Backend.Cuda).image
    assert np.abs(dev - host).max() < 1e-3 * np.abs(host).max()


def test_conditioning_without_a_cuda_build_says_so():
    if gs.cuda_compiled():
        pytest.skip("built with CUDA")
    rec = _raw_ping("l12_5d", {"receive": {"window": {"range": [1.4, 1.6]}}})
    with pytest.raises(RuntimeError, match="CUDA"):
        gs.core.to_baseband(rec, backend=gs.core.Backend.Cuda)
