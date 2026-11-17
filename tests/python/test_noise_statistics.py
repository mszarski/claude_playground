"""K-distributed ambient noise (sim/noise.py `k_noise`, `gamma_texture`; scenario key
`receiver.ambient_noise.statistics`).

Closed forms: with tau a gamma variate of unit mean and shape nu, E{tau^2} = 1 + 1/nu and
E{tau^3} = (1 + 1/nu)(1 + 2/nu). A real sample x = sqrt(tau) g has m4 / m2^2 = 3 (1 + 1/nu)
(kurtosis 3 / nu, TR 0702 eq. 27); a complex envelope has intensity moments
E{I^2}/E{I}^2 = 2 (1 + 1/nu) and E{I^3}/E{I}^3 = 6 (1 + 1/nu)(1 + 2/nu).
"""
from math import gamma as gamma_fn

import numpy as np
import pytest
from scipy.signal import csd, welch

from gpusonar.sim import Scene, make_pose, run_scenario, simulate_ping
from gpusonar.sim import noise as nz
from gpusonar.sonar import SonarModel

FS = 64000.0
FLAT = lambda f: np.full_like(np.asarray(f, dtype=float), 50.0)       # noqa: E731


def kurtosis_ratio(x):
    x = np.asarray(x, dtype=float).ravel()
    return float(np.mean(x**4) / np.mean(x**2) ** 2)


def test_white_k_noise_as_in_tr0702_has_the_k_kurtosis_and_the_asked_level():
    n = 1 << 19
    for nu in (0.5, 2.0, 8.0):
        x = nz.k_noise(n, FS, FLAT, np.random.default_rng(1), {"model": "k", "shape": nu}, n_channels=4, sensitivity_db=-180.0)
        assert kurtosis_ratio(x) == pytest.approx(3.0 * (1.0 + 1.0 / nu), rel=0.06)
        from gpusonar.sonar_equation import noise_rms_volts
        assert x.std() == pytest.approx(noise_rms_volts(50.0, -180.0, FS), rel=0.02)
        # Independent between channels, and white: flat spectrum at the asked level.
        assert abs(np.corrcoef(x[0], x[1])[0, 1]) < 0.01
        f, pxx = welch(x, fs=FS, nperseg=1024, axis=1)
        band = (f > 500.0) & (f < 31000.0)
        assert np.abs(10 * np.log10(pxx.mean(axis=0)[band]) - (50.0 - 180.0)).max() < 0.7


def test_filtering_white_k_noise_brings_it_back_towards_gaussian():
    """TR 0702 Sec. 4.6.2: a linear combination of independent numbers is closer to Gaussian.
    A filter 1/16 of the band wide leaves about 1/16 of the excess kurtosis."""
    nu, n = 0.5, 1 << 19
    narrow = lambda f: np.where((np.asarray(f) > 8000.0) & (np.asarray(f) < 10000.0), 50.0, -200.0)    # noqa: E731
    x = nz.k_noise(n, FS, narrow, np.random.default_rng(2), {"model": "k", "shape": nu}, n_channels=4)
    excess = kurtosis_ratio(x) / 3.0 - 1.0
    assert 0.03 < excess < 0.25 and excess < 0.15 * (1.0 / nu)


def test_texture_has_unit_mean_gamma_moments_and_the_stated_correlation_time():
    nu, tau = 1.5, 4e-3
    t = nz.gamma_texture(1 << 21, FS, nu, tau, np.random.default_rng(3), n_channels=3, common=False) ** 2
    assert t.mean() == pytest.approx(1.0, rel=0.02)
    assert (t**2).mean() == pytest.approx(1.0 + 1.0 / nu, rel=0.05)
    cell = int(round(tau * FS))
    d = t[0] - t[0].mean()
    acov = lambda lag: float(np.mean(d[:-lag] * d[lag:]) / np.mean(d * d))   # noqa: E731
    assert acov(cell // 2) == pytest.approx(0.5, abs=0.04)             # a triangle: half at half the correlation time
    assert abs(acov(cell)) < 0.04 and abs(acov(2 * cell)) < 0.04       # zero from the correlation time on
    same = nz.gamma_texture(4096, FS, nu, tau, np.random.default_rng(3), n_channels=3, common=True)
    assert np.array_equal(same[0], same[2]) and not np.array_equal(t[0], t[1])
    fast = nz.gamma_texture(1 << 16, FS, nu, 0.0, np.random.default_rng(3)) ** 2
    assert abs(np.corrcoef(fast[0, :-1], fast[0, 1:])[0, 1]) < 0.02     # correlation time 0: a variate per sample


def test_compound_k_noise_keeps_the_spectrum_and_the_array_coherence():
    nu, tau, n = 1.0, 5e-3, 1 << 20
    sonar = SonarModel.load("flank_32")
    positions = sonar.positions[:4]
    wenz = lambda f: nz.wenz_spectrum_db(f, wind_speed=8.0)             # noqa: E731
    stats = {"model": "k", "shape": nu, "correlation_time": tau}
    x = nz.k_noise(n, FS, wenz, np.random.default_rng(5), stats, positions=positions, sensitivity_db=-180.0)
    gauss = nz.gaussian_noise(n, FS, wenz, np.random.default_rng(5), positions=positions, sensitivity_db=-180.0)
    # Intensity moments of the K distribution, for a real sample.
    assert kurtosis_ratio(gauss) == pytest.approx(3.0, rel=0.03)
    assert kurtosis_ratio(x) == pytest.approx(3.0 * (1.0 + 1.0 / nu), rel=0.12)
    # Power spectrum unchanged (resolution 15.6 Hz... coarser than 1 / tau = 200 Hz: 250 Hz bins).
    f, pxx = welch(x, fs=FS, nperseg=256, axis=1)
    want = nz.wenz_spectrum_db(np.maximum(f, 1.0), wind_speed=8.0) - 180.0
    band = (f > 1000.0) & (f < 30000.0)
    assert np.abs(10 * np.log10(pxx.mean(axis=0)[band]) - want[band]).max() < 1.0
    _, pgg = welch(gauss, fs=FS, nperseg=256, axis=1)
    assert np.abs(10 * np.log10(pxx.mean(axis=0)[band] / pgg.mean(axis=0)[band])).max() < 0.8
    # Common texture: the coherence between hydrophones is the isotropic sin(kd)/(kd).
    def coherence(y, j):
        _, p00 = welch(y[0], fs=FS, nperseg=512)
        ff, pjj = welch(y[j], fs=FS, nperseg=512)
        _, p0j = csd(y[0], y[j], fs=FS, nperseg=512)
        return ff, np.real(p0j) / np.sqrt(p00 * pjj)
    ff, measured = coherence(x, 1)
    d = np.linalg.norm(positions[1] - positions[0])
    sel = (ff > 1000.0) & (ff < 20000.0)
    assert np.abs(measured[sel] - np.sinc(2 * ff[sel] * d / 1500.0)).max() < 0.15
    # A texture per channel lowers it by E{sqrt(tau)}^2 = (Gamma(nu + 1/2) / Gamma(nu))^2 / nu.
    y = nz.k_noise(n, FS, wenz, np.random.default_rng(5), dict(stats, common=False), positions=positions, sensitivity_db=-180.0)
    _, own = coherence(y, 1)
    low = (ff > 1000.0) & (ff < 4000.0)                                 # where the coherence is large
    factor = (gamma_fn(nu + 0.5) / gamma_fn(nu)) ** 2 / nu
    assert factor == pytest.approx(np.pi / 4.0)
    assert np.mean(own[low] / np.sinc(2 * ff[low] * d / 1500.0)) == pytest.approx(factor, abs=0.06)


def test_infinite_shape_and_gaussian_model_are_bit_identical_and_draw_nothing_more():
    for stats in (None, {}, {"model": "gaussian"}, {"model": "k", "shape": float("inf")}, {"model": "k", "shape": np.inf, "correlation_time": 0.01}):
        a, b = np.random.default_rng(7), np.random.default_rng(7)
        x = nz.k_noise(8192, FS, FLAT, a, stats, n_channels=3)
        y = nz.gaussian_noise(8192, FS, FLAT, b, n_channels=3)
        assert np.array_equal(x, y) and a.integers(1 << 30) == b.integers(1 << 30)
    big = nz.k_noise(8192, FS, FLAT, np.random.default_rng(7), {"model": "k", "shape": 1e9, "correlation_time": 0.01}, n_channels=3)
    assert np.allclose(big, nz.gaussian_noise(8192, FS, FLAT, np.random.default_rng(7), n_channels=3), rtol=1e-3, atol=1e-9)
    with pytest.raises(ValueError, match="shape"):
        nz.k_noise(64, FS, FLAT, np.random.default_rng(0), {"model": "k"}, n_channels=1)
    with pytest.raises(ValueError, match="unknown noise statistics"):
        nz.k_noise(64, FS, FLAT, np.random.default_rng(0), {"model": "weibull"}, n_channels=1)
    with pytest.raises(ValueError, match="positive"):
        nz.k_noise(64, FS, FLAT, np.random.default_rng(0), {"model": "k", "shape": 0.0}, n_channels=1)


def one_ping(sonar, stats, seed=11, **kw):
    return simulate_ping(sonar, make_pose(position=(0.0, 0.0, -15.0)), Scene(), environment={"sound_speed": 1500.0},
                         adc=False, noise_spectrum=FLAT, noise_statistics=stats, rng=np.random.default_rng(seed), **kw).samples.array


def test_simulated_record_default_unchanged_and_baseband_envelope_is_k():
    raw = SonarModel.load("flank_32")
    assert np.array_equal(one_ping(raw, None), one_ping(raw, {"model": "k", "shape": np.inf}))
    assert np.array_equal(one_ping(raw, None), one_ping(raw, {"model": "gaussian"}))
    assert not np.array_equal(one_ping(raw, None), one_ping(raw, {"model": "k", "shape": 1.0}))
    base = SonarModel.load("sas_auv", {"receive": {"kind": "baseband", "sample_rate": 100e3, "window": {"range": [5.0, 245.0]}}})
    assert np.array_equal(one_ping(base, None), one_ping(base, {"model": "k", "shape": np.inf, "correlation_time": 1e-3}))
    gauss = np.abs(one_ping(base, None)) ** 2
    assert (gauss**2).mean() / gauss.mean() ** 2 == pytest.approx(2.0, rel=0.03)
    for nu, tau in ((1.0, 0.0), (3.0, 0.0), (2.0, 1e-3)):
        inten = np.abs(one_ping(base, {"model": "k", "shape": nu, "correlation_time": tau})) ** 2
        tol = 0.05 if tau == 0.0 else 0.15
        assert inten.mean() == pytest.approx(gauss.mean(), rel=tol)                      # the level is unchanged
        assert (inten**2).mean() / inten.mean() ** 2 == pytest.approx(2.0 * (1.0 + 1.0 / nu), rel=tol)
        assert (inten**3).mean() / inten.mean() ** 3 == pytest.approx(6.0 * (1.0 + 1.0 / nu) * (1.0 + 2.0 / nu), rel=3 * tol)


def test_scenario_key():
    cfg = {"sonar": "flank_32", "environment": {"sound_speed": 1500.0}, "seed": 3, "pings": {"count": 1},
           "platform": {"position": [0.0, 0.0, -20.0]}, "scene": [],
           "receiver": {"adc": False, "ambient_noise": {"level_db": 50.0}}}
    plain = run_scenario(cfg).records[0].samples.array
    spiky_cfg = dict(cfg, receiver={"adc": False, "ambient_noise": {"level_db": 50.0, "statistics": {"model": "k", "shape": 0.5}}})
    spiky = run_scenario(spiky_cfg).records[0].samples.array
    assert np.array_equal(plain, run_scenario(dict(cfg)).records[0].samples.array)
    assert kurtosis_ratio(plain) == pytest.approx(3.0, rel=0.05)
    # 3 (1 + 1/nu) = 9 for white independent channels; the mixing that gives the array its
    # isotropic coherence is a linear combination of channels and takes part of it back.
    assert 4.5 < kurtosis_ratio(spiky) < 9.5
    assert spiky.std() == pytest.approx(plain.std(), rel=0.03)
