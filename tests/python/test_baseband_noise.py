"""Ambient noise in baseband records: the configured spectrum across the band and the coherence between hydrophones
that the real-passband path has (sim/noise.py `baseband_gaussian_noise`; D-94). Closed forms: the power 2 S fs of a
complex envelope, sin(kd)/(kd) of an isotropic field, the Cron-Sherman vertical pair, and the gain of a line array
against that noise, N^2 / sum(Gamma)."""

import numpy as np
import pytest

from gpusonar.sim import Scene, make_pose, simulate_ping
from gpusonar.sim import noise as nm
from gpusonar.sonar import SonarModel

C = 1500.0


def line(n, pitch):
    return np.column_stack([(np.arange(n) - (n - 1) / 2.0) * pitch, np.zeros(n), np.zeros(n)])


def test_power_is_twice_the_spectrum_times_the_band_and_follows_its_slope():
    rng = np.random.default_rng(1)
    fs, fc, n = 8000.0, 5000.0, 1 << 15
    flat = lambda f: np.full_like(np.asarray(f, dtype=float), 50.0)             # noqa: E731
    x = nm.baseband_gaussian_noise(n, fs, fc, flat, rng, n_channels=4, sensitivity_db=-100.0)
    assert x.dtype == np.complex64 and x.shape == (4, n)
    assert 10 * np.log10(np.mean(np.abs(x) ** 2)) == pytest.approx(10 * np.log10(2.0 * fs) + 50.0 - 100.0, abs=0.1)
    assert abs(np.corrcoef(x[0].real, x[1].real)[0, 1]) < 0.03                  # no positions: independent
    # the Wenz spectrum falls across 1 to 9 kHz: the baseband spectrum is its image about the carrier
    wenz = lambda f: nm.wenz_spectrum_db(f, 5.0, 0.5)                           # noqa: E731
    y = nm.baseband_gaussian_noise(n, fs, fc, wenz, rng, n_channels=8)
    psd = (np.abs(np.fft.fft(y, axis=1)) ** 2).mean(axis=0) / (n * fs)          # two-sided density of the envelope
    fb = np.fft.fftfreq(n, 1.0 / fs)
    for lo, hi in ((-3500.0, -2500.0), (-500.0, 500.0), (2500.0, 3500.0)):
        band = (fb >= lo) & (fb < hi)
        want = 10 * np.log10(2.0 * np.mean(10.0 ** (wenz(fc + fb[band]) / 10.0)))
        assert 10 * np.log10(psd[band].mean()) == pytest.approx(want, abs=0.25)
    assert wenz(np.array([2000.0]))[0] - wenz(np.array([8000.0]))[0] > 8.0      # the slope that a flat draw loses
    total = 2.0 * np.trapezoid(10.0 ** (wenz(np.linspace(1000.0, 9000.0, 4001)) / 10.0), np.linspace(1000.0, 9000.0, 4001))
    assert 10 * np.log10(np.mean(np.abs(y) ** 2)) == pytest.approx(10 * np.log10(total), abs=0.15)


@pytest.mark.parametrize("separation", [0.05, 0.09, 0.15, 0.4])
def test_isotropic_coherence_between_two_hydrophones(separation):
    rng = np.random.default_rng(2)
    fs, fc, n = 400.0, 5000.0, 1 << 17                                         # a narrow band: one coherence value
    flat = lambda f: np.zeros_like(np.asarray(f, dtype=float))                 # noqa: E731
    x = nm.baseband_gaussian_noise(n, fs, fc, flat, rng, positions=line(2, separation), sound_speed=C)
    got = np.mean(x[0] * np.conj(x[1])) / np.sqrt(np.mean(np.abs(x[0]) ** 2) * np.mean(np.abs(x[1]) ** 2))
    assert got.real == pytest.approx(np.sinc(2.0 * fc * separation / C), abs=0.02) and abs(got.imag) < 0.02


def test_surface_noise_coherence_of_a_vertical_pair_is_the_closed_form():
    rng = np.random.default_rng(3)
    fs, fc, n, d = 400.0, 5000.0, 1 << 17, 0.1
    flat = lambda f: np.zeros_like(np.asarray(f, dtype=float))                 # noqa: E731
    pair = np.array([[0.0, 0.0, 10.0], [0.0, 0.0, 10.0 + d]])                  # the second hydrophone deeper
    x = nm.baseband_gaussian_noise(n, fs, fc, flat, rng, positions=pair, sound_speed=C, coherence="surface")
    got = np.mean(x[0] * np.conj(x[1])) / np.sqrt(np.mean(np.abs(x[0]) ** 2) * np.mean(np.abs(x[1]) ** 2))
    k = 2 * np.pi * fc * d / C
    want = 2 * (np.sin(k) / k + (np.cos(k) - 1) / k ** 2) + 2j * (np.sin(k) / k ** 2 - np.cos(k) / k)      # Cron-Sherman, vertical pair
    assert got == pytest.approx(want, abs=0.02)
    # the real-passband generator takes the same option and gives the same coherence at the carrier
    real = nm.gaussian_noise(1 << 17, 32000.0, lambda f: np.where(np.abs(np.asarray(f) - fc) < 200.0, 0.0, -200.0), rng, positions=pair, sound_speed=C, coherence="surface")
    spec = np.fft.rfft(real, axis=1)
    band = np.abs(np.fft.rfftfreq(1 << 17, 1 / 32000.0) - fc) < 200.0
    got_real = np.sum(spec[0, band] * np.conj(spec[1, band])) / np.sqrt(np.sum(np.abs(spec[0, band]) ** 2) * np.sum(np.abs(spec[1, band]) ** 2))
    assert got_real == pytest.approx(want, abs=0.03)
    with pytest.raises(ValueError, match="coherence"):
        nm.coherence_matrix("lumpy", pair, fc, C)


@pytest.mark.parametrize("pitch,name", [(0.09, "the test AUV's short row"), (0.15, "the long row: half a wavelength")])
def test_array_gain_against_sea_noise_is_the_closed_form(pitch, name):
    """Broadside sum of a 32-hydrophone row at 5 kHz: gain against isotropic noise N^2 / sum(Gamma_ij), which is
    10 log N only at half-wavelength pitch (and above)."""
    rng = np.random.default_rng(4)
    n_el, fs, fc, n = 32, 400.0, 5000.0, 1 << 15
    flat = lambda f: np.zeros_like(np.asarray(f, dtype=float))                 # noqa: E731
    pos = line(n_el, pitch)
    x = nm.baseband_gaussian_noise(n, fs, fc, flat, rng, positions=pos, sound_speed=C)
    measured = 10 * np.log10(n_el ** 2 * np.mean(np.abs(x) ** 2) / np.mean(np.abs(x.sum(axis=0)) ** 2))
    closed = 10 * np.log10(n_el ** 2 / nm.isotropic_coherence(pos, fc, C).sum())
    assert measured == pytest.approx(closed, abs=0.35), name
    full = 10 * np.log10(n_el)
    if pitch < 0.14:
        assert closed == pytest.approx(13.0, abs=0.3) and full - closed > 1.9       # 2.1 dB less than 10 log N
    else:
        assert closed == pytest.approx(full, abs=0.2)
    independent = nm.baseband_gaussian_noise(n, fs, fc, flat, rng, positions=pos, sound_speed=C, coherence="independent")
    assert 10 * np.log10(n_el ** 2 * np.mean(np.abs(independent) ** 2) / np.mean(np.abs(independent.sum(axis=0)) ** 2)) == pytest.approx(full, abs=0.35)


BASEBAND = {"receive": {"kind": "baseband", "sample_rate": 8000.0, "window": {"range": [2000.0, 2600.0]}}}


def _noise(overrides, options=None, statistics=None, seed=5):
    sonar = SonarModel.load("popoto_testauv_1row", overrides)
    wenz = lambda f: nm.wenz_spectrum_db(f, 5.0, 0.5)                          # noqa: E731
    rec = simulate_ping(sonar, make_pose(position=(0.0, 0.0, 10.0)), Scene(), environment={"sound_speed": C}, adc=False, transmit=False,
                        noise_spectrum=wenz, noise_options=options, noise_statistics=statistics, rng=np.random.default_rng(seed))
    return sonar, rec.samples.array


def test_a_baseband_record_has_the_shared_noise_and_the_old_draw_is_selectable():
    sonar, x = _noise(BASEBAND)
    pitch = float(np.linalg.norm(sonar.positions[1] - sonar.positions[0]))
    assert pitch == pytest.approx(0.09, abs=0.005)
    rho = np.mean(x[0] * np.conj(x[1])) / np.sqrt(np.mean(np.abs(x[0]) ** 2) * np.mean(np.abs(x[1]) ** 2))
    assert rho.real > 0.3                                                       # neighbours 0.3 wavelengths apart share noise
    gain = lambda a: 10 * np.log10(a.shape[0] ** 2 * np.mean(np.abs(a) ** 2) / np.mean(np.abs(a.sum(axis=0)) ** 2))       # noqa: E731
    _, old = _noise(BASEBAND, {"coherence": "independent", "spectrum": "flat"})
    assert gain(old) == pytest.approx(10 * np.log10(32), abs=0.4) and gain(old) - gain(x) > 1.0
    assert abs(np.mean(old[0] * np.conj(old[1]))) / np.mean(np.abs(old[0]) ** 2) < 0.05
    # the old draw, sample for sample: white at the centre frequency's level, one Gaussian pair per sample
    rx = sonar.config["receive"]
    sens = rx["sensitivity_db"] + rx.get("gain_db", 0.0)
    g = np.random.default_rng(5)
    sigma = np.sqrt(2.0 * 10.0 ** (nm.wenz_spectrum_db(np.array([5000.0]), 5.0, 0.5)[0] / 10.0) * 8000.0) * 10.0 ** (sens / 20.0)
    want = (sigma / np.sqrt(2.0)) * (g.standard_normal(old.shape) + 1j * g.standard_normal(old.shape)).astype(np.complex64)
    assert np.allclose(old, want, rtol=1e-5, atol=1e-9 * sigma)              # (the record then passes the gain stage: a float32 product)
    # flat keeps the level, shaped the band's own power (more below the carrier than is lost above it)
    _, flat = _noise(BASEBAND, {"spectrum": "flat"})
    assert 10 * np.log10(np.mean(np.abs(flat) ** 2) / np.mean(np.abs(old) ** 2)) == pytest.approx(0.0, abs=0.2)
    assert np.mean(np.abs(x) ** 2) > 1.2 * np.mean(np.abs(flat) ** 2)
    with pytest.raises(ValueError, match="coherence"):
        _noise(BASEBAND, {"coherence": "lumpy"})


def test_white_statistics_keep_their_own_generator_and_k_texture_rides_on_the_shared_field():
    _, a = _noise(BASEBAND, statistics={"model": "alpha_stable", "alpha": 2.0})
    assert abs(np.mean(a[0] * np.conj(a[1]))) / np.mean(np.abs(a[0]) ** 2) < 0.05           # white and independent, as before
    _, k0 = _noise(BASEBAND, statistics={"model": "k", "shape": 2.0})
    assert abs(np.mean(k0[0] * np.conj(k0[1]))) / np.mean(np.abs(k0[0]) ** 2) < 0.06
    _, k = _noise(BASEBAND, statistics={"model": "k", "shape": 2.0, "correlation_time": 0.01})
    _, gauss = _noise(BASEBAND)
    assert 10 * np.log10(np.mean(np.abs(k) ** 2) / np.mean(np.abs(gauss) ** 2)) == pytest.approx(0.0, abs=0.6)       # the mean power is the field's
    assert (np.mean(k[0] * np.conj(k[1])) / np.mean(np.abs(k[0]) ** 2)).real > 0.3                                 # and so is the coherence
    kurt = lambda v: np.mean(np.abs(v) ** 4) / np.mean(np.abs(v) ** 2) ** 2                                     # noqa: E731
    assert kurt(k) > 1.25 * kurt(gauss)                                                                         # heavier tails


def test_real_passband_default_is_unchanged_and_takes_the_options():
    real = {"receive": {"kind": "real", "sample_rate": 32000.0, "window": {"range": [2000.0, 2100.0]}}}
    _, a = _noise(real)
    _, b = _noise(real, {"coherence": "isotropic", "spectrum": "shaped"})
    assert np.array_equal(a, b)
    _, ind = _noise(real, {"coherence": "independent"})
    # the spectrum is steep (most power at the lowest frequencies, where neighbours are fully coherent): few independent samples
    assert abs(np.corrcoef(ind[0], ind[1])[0, 1]) < 0.4 and np.corrcoef(a[0], a[1])[0, 1] > 0.8
