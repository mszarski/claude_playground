"""Noise generator fitted to a multichannel recording (sim/noise_fit.py).

There is no real recording: the "recordings" here are synthetic with a known cross-spectral
matrix, built independently of the code under test: isotropic noise from the repository's own
`noise.gaussian_noise` (sin(kd)/(kd) coherence), an interferer that is a band-limited source
delayed across the array as a plane wave, and heavy tails from a texture.
"""

import os

import numpy as np
import pytest
from scipy import signal

import gpusonar as gs
from gpusonar.sim import noise as noise_models
from gpusonar.sim import noise_fit as nf

from conftest import make_record

FS, C_SOUND, N_CH, SPACING = 8000.0, 1500.0, 8, 0.25
BEARING = np.radians(30.0)
POS = np.zeros((N_CH, 3))
POS[:, 0] = np.arange(N_CH) * SPACING
SEP = np.abs(POS[:, None, 0] - POS[None, :, 0])


def iso_level_db(f):
    """One-sided spectrum level of the isotropic part: falls 6 dB/octave above 1 kHz."""
    return -36.0 - 10.0 * np.log10(1.0 + (np.asarray(f, dtype=float) / 1000.0) ** 2)


_TAPS = signal.firwin(65, [1500.0, 2500.0], pass_zero=False, fs=FS)
INTERFERER_RMS = 0.4


def interferer_psd(f):
    """One-sided PSD of the interferer at a hydrophone."""
    h = np.abs(np.fft.rfft(_TAPS, 4096))
    gain = np.interp(f, np.fft.rfftfreq(4096, 1.0 / FS), h)
    return INTERFERER_RMS**2 * gain**2 * 2.0 / FS / np.sum(_TAPS**2)


def truth_csd(f, interferer=True):
    """One-sided cross-spectral density [F, C, C] of the synthetic field."""
    f = np.asarray(f, dtype=float)
    out = (10.0 ** (iso_level_db(f) / 10.0))[:, None, None] * np.sinc(2.0 * f[:, None, None] * SEP / C_SOUND)
    if interferer:
        a = np.exp(-2j * np.pi * f[:, None] * (POS[:, 0] * np.sin(BEARING) / C_SOUND)[None])
        out = out + interferer_psd(f)[:, None, None] * a[:, :, None] * np.conj(a[:, None, :])
    return out


def make_field(n, rng, interferer=True, source_family="gaussian", source_par=0.0, hold=1):
    x = noise_models.gaussian_noise(n, FS, iso_level_db, rng, positions=POS, sound_speed=C_SOUND).astype(np.float64)
    if interferer:
        e = rng.standard_normal(n)
        if source_family != "gaussian":
            tau = np.repeat(nf.texture(source_family, source_par, n // hold + 1, rng), hold)[:n]
            e = e * np.sqrt(tau)
        s = signal.lfilter(_TAPS, 1.0, e) * INTERFERER_RMS / np.sqrt(np.sum(_TAPS**2))
        f = np.fft.rfftfreq(n, 1.0 / FS)
        sf = np.fft.rfft(s)
        delay = POS[:, 0] * np.sin(BEARING) / C_SOUND
        x = x + np.fft.irfft(sf[None, :] * np.exp(-2j * np.pi * f[None, :] * delay[:, None]), n, axis=1)
    return x


def welch_csd(x, nfft=256):
    f, p = signal.csd(x[:, None, :], x[None, :, :], fs=FS, window="hann", nperseg=nfft, noverlap=nfft // 2, axis=-1)
    return f, np.moveaxis(p, -1, 0)                       # scipy: conj(X_i) X_j -> transpose of ours


@pytest.fixture(scope="module")
def field():
    rng = np.random.default_rng(7)
    x = make_field(1 << 18, rng)
    return x, nf.fit(x, FS, nfft=256, positions=POS)


# -- the variates and the estimator statistics ---------------------------------------------------

def test_positive_stable_has_its_laplace_transform():
    rng = np.random.default_rng(0)
    for a in (0.4, 0.75, 0.95):
        s = nf.positive_stable(a, 400_000, rng)
        for t in (0.3, 1.0, 4.0):
            est = np.exp(-t * s)
            assert est.mean() == pytest.approx(np.exp(-t**a), abs=5.0 * est.std() / np.sqrt(s.size))
    assert np.all(nf.positive_stable(1.0, 10, rng) == 1.0)


def test_sas_innovations_have_the_stable_characteristic_function():
    """sqrt(tau) g is symmetric alpha-stable of scale 1/sqrt 2: E cos(t x) = exp(-|t / sqrt 2|^alpha);
    alpha = 2 is the standard normal (the convention of the UACR noise files)."""
    rng = np.random.default_rng(1)
    for alpha in (1.2, 1.7, 2.0):
        g = rng.standard_normal(400_000)
        x = np.sqrt(nf.texture("sas", alpha, g.shape, rng)) * g
        for t in (0.5, 1.0, 2.0):
            assert np.cos(t * x).mean() == pytest.approx(np.exp(-abs(t / np.sqrt(2.0)) ** alpha), abs=4e-3)


@pytest.mark.parametrize("family,par", [("gaussian", 0.0), ("k", 0.7), ("k", 3.0), ("sas", 1.3), ("sas", 1.8)])
def test_log_moments_closed_forms(family, par):
    rng = np.random.default_rng(2)
    g = rng.standard_normal(1_000_000)
    la = np.log(np.abs(np.sqrt(nf.texture(family, par, g.shape, rng)) * g))
    mean, var = nf.log_abs_moments(family, par)
    assert la.mean() == pytest.approx(mean, abs=0.006)
    assert la.var() == pytest.approx(var, rel=0.02)


def test_k_family_has_unit_variance_and_kurtosis_3_over_shape():
    rng = np.random.default_rng(3)
    g = rng.standard_normal(2_000_000)
    x = np.sqrt(nf.texture("k", 4.0, g.shape, rng)) * g
    assert x.var() == pytest.approx(1.0, rel=0.01)
    assert np.mean(x**4) / x.var() ** 2 - 3.0 == pytest.approx(3.0 / 4.0, abs=0.06)


def test_coherence_estimator_floor_and_standard_deviation():
    """The numbers the fit report quotes: floor sqrt(pi / 4 n_d) at zero coherence, standard
    deviation (1 - g^2) / sqrt(2 n_d) elsewhere, for n_d independent complex Gaussian blocks."""
    rng = np.random.default_rng(4)
    n_d, trials = 64, 4000
    z = (rng.standard_normal((trials, 3, n_d)) + 1j * rng.standard_normal((trials, 3, n_d))) / np.sqrt(2.0)
    a, b = z[:, 0], z[:, 1]
    coh = lambda u, v: np.abs((u * np.conj(v)).mean(axis=1)) / np.sqrt((np.abs(u) ** 2).mean(axis=1) * (np.abs(v) ** 2).mean(axis=1))  # noqa: E731
    assert coh(a, b).mean() == pytest.approx(nf.coherence_floor(n_d), rel=0.03)
    g = 0.6
    d = g * a + np.sqrt(1.0 - g**2) * z[:, 2]
    assert coh(a, d).std() == pytest.approx(nf.coherence_std(g, n_d), rel=0.08)


def test_effective_averages_of_overlapped_hann_blocks():
    """Variance of a Welch spectrum of white noise against the count the report uses."""
    rng = np.random.default_rng(5)
    nfft, k = 64, 40
    w = np.hanning(nfft + 1)[:nfft]
    x = rng.standard_normal((3000, nfft // 2 * (k + 1)))
    _, p = signal.welch(x, window=w, nperseg=nfft, noverlap=nfft // 2, axis=1)
    rel_var = (p[:, 5:-5].var(axis=0) / p[:, 5:-5].mean(axis=0) ** 2).mean()
    assert 1.0 / rel_var == pytest.approx(nf.effective_averages(k, w, nfft // 2), rel=0.05)
    assert nf.effective_averages(k, w, nfft // 2) == pytest.approx(k / (1.0 + 2.0 / 36.0 * (1.0 - 1.0 / k)), rel=1e-3)


# -- spectrum and coherence ------------------------------------------------------------------------

def test_fit_recovers_spectrum_and_coherence(field):
    x, model = field
    f, psd = model.psd()
    truth = truth_csd(f)
    t_psd = np.real(np.einsum("fcc->fc", truth))
    sel = (f > 200.0) & (f < 3800.0)
    assert np.abs(10.0 * np.log10(psd[sel] / t_psd[sel])).max() < 0.6
    # Coherence against spacing and frequency: the closed form of the mixture.
    coh = model.coherence()[1:model.nfft // 2]
    t_coh = truth / np.sqrt(t_psd[:, :, None] * t_psd[:, None, :])
    assert np.abs(coh[sel] - t_coh[sel]).max() < 0.08
    assert np.sqrt(np.mean(np.abs(coh[sel] - t_coh[sel]) ** 2)) < 0.02
    assert model.family == "gaussian"
    assert model.report["excluded_fraction"] == 0.0 and not model.tones


def test_isotropic_coherence_length_is_0p3_wavelengths():
    rng = np.random.default_rng(8)
    model = nf.fit(make_field(1 << 17, rng, interferer=False), FS, nfft=256, positions=POS)
    rows = [r for r in model.report["coherence_length"] if 1200.0 < r["frequency_hz"] < 3000.0]
    assert rows
    for r in rows:                                         # sin(x)/x = 1/2 at x = 1.8955: d = 0.3017 wavelengths
        assert r["wavelengths"] == pytest.approx(0.3017, abs=0.05)


@pytest.mark.parametrize("method", ["mitchell_mcpherson", "filter"])
def test_regenerated_noise_has_the_fitted_spectrum_and_coherence(field, method):
    _, model = field
    y = model.generate(1 << 18, np.random.default_rng(11), method=method)
    assert y.shape == (N_CH, 1 << 18) and y.dtype == np.float64
    f, p = welch_csd(y)
    fm, psd = model.psd()
    sel = (f > 200.0) & (f < 3800.0)
    got = np.real(np.einsum("fcc->fc", p))
    want = np.vstack([psd[:1], psd, psd[-1:]])[: len(f)]
    assert np.abs(10.0 * np.log10(got[sel] / want[sel])).max() < 0.7
    coh = np.conj(p) / np.sqrt(got[:, :, None] * got[:, None, :])
    want_coh = model.coherence()[: len(f)]
    assert np.sqrt(np.mean(np.abs(coh[sel] - want_coh[sel]) ** 2)) < 0.03
    # ... and so the truth's, through fit and regeneration.
    truth = truth_csd(f)
    t_psd = np.real(np.einsum("fcc->fc", truth))
    assert np.sqrt(np.mean(np.abs(coh[sel] - (truth / np.sqrt(t_psd[:, :, None] * t_psd[:, None, :]))[sel]) ** 2)) < 0.04


def test_mitchell_mcpherson_ensemble_average_converges_to_the_input_psd(field):
    """TR 0702 Sec. 4.6.1: "if the expectation operator is replaced by an average over independent
    realizations, that average should converge to the original PSD". The error of the average of R
    periodograms falls as 1 / sqrt(R), and the window satisfies eq. (26)."""
    _, model = field
    n = model.nfft
    w = nf._sine_window(n)
    assert np.allclose(w[: n // 2] ** 2 + w[n // 2:] ** 2, 1.0)                      # eq. (26) at 50 % overlap
    rng = np.random.default_rng(12)
    f, psd = model.psd()
    sel = slice(8, len(f) - 8)
    errs = {}
    for r in (16, 256):
        acc = np.zeros((n // 2 + 1, N_CH))
        for _ in range(r):
            y = model.generate(n, rng, method="mitchell_mcpherson")
            acc += np.abs(np.fft.rfft(y, axis=1).T) ** 2 * 2.0 / (FS * n)
        est = (acc / r)[1:n // 2]                           # the model's bins
        errs[r] = np.sqrt(np.mean((est[sel] / psd[sel] - 1.0) ** 2))
    assert errs[256] < 0.08
    assert errs[16] / errs[256] == pytest.approx(4.0, rel=0.35)                     # sqrt(256 / 16)
    # Total power is exact in expectation (the sum over frequency of the density).
    y = model.generate(1 << 17, rng, method="mitchell_mcpherson")
    assert y.var(axis=1) == pytest.approx(np.real(np.einsum("fcc->c", model.csd)) * FS / n, rel=0.03)


# -- what an array processor sees ---------------------------------------------------------------------

def _beam_power(csd_f, f, bearings):
    """Conventional beamformer output density w^H S w, w = steering / C."""
    tau = POS[:, 0][None, :] * np.sin(bearings)[:, None] / C_SOUND
    w = np.exp(-2j * np.pi * f * tau) / N_CH
    return np.real(np.einsum("ba,ac,bc->b", np.conj(w), csd_f, w))


def test_beamformed_noise_level_against_bearing(field):
    """The interferer shows at its bearing with its level, and the rest of the bearing response
    is the isotropic field's, in noise regenerated from the fit."""
    _, model = field
    y = model.generate(1 << 18, np.random.default_rng(13))
    f, p = welch_csd(y)
    bearings = np.radians(np.arange(-90.0, 91.0, 2.0))
    for f0 in (1500.0, 2000.0, 2500.0):
        k = int(np.argmin(np.abs(f - f0)))
        got = _beam_power(np.conj(p[k]), f[k], bearings)
        want = _beam_power(truth_csd(np.array([f[k]]))[0], f[k], bearings)
        assert np.abs(10.0 * np.log10(got / want)).max() < 1.0
        if f0 == 2000.0:
            assert abs(np.degrees(bearings[np.argmax(got)]) - 30.0) <= 2.0
            assert got.max() > 2.0 * np.median(got)


def test_array_gain_against_fitted_isotropic_noise_matches_closed_form():
    """Array gain of the conventional beamformer at broadside in isotropic noise:
    C^2 / sum_ij sin(k d_ij) / (k d_ij); 10 log C where the spacing is half a wavelength."""
    rng = np.random.default_rng(14)
    model = nf.fit(make_field(1 << 18, rng, interferer=False), FS, nfft=256, positions=POS)
    y = model.generate(1 << 18, rng)
    f, p = welch_csd(y)
    for f0 in (750.0, 1500.0, 3000.0):
        k = int(np.argmin(np.abs(f - f0)))
        closed = N_CH**2 / np.sinc(2.0 * f[k] * SEP / C_SOUND).sum()
        for csd_f in (model.csd[k], np.conj(p[k])):
            mean_channel = np.real(np.trace(csd_f)) / N_CH
            gain = mean_channel / _beam_power(csd_f, f[k], np.array([0.0]))[0]
            assert 10.0 * np.log10(gain / closed) == pytest.approx(0.0, abs=0.35)
    k = int(np.argmin(np.abs(f - 3000.0)))                 # 0.25 m is half a wavelength at 3 kHz
    assert N_CH**2 / np.sinc(2.0 * f[k] * SEP / C_SOUND).sum() == pytest.approx(N_CH, rel=1e-6)


# -- tails ------------------------------------------------------------------------------------------------

def test_tail_parameters_recovered_on_white_noise():
    rng = np.random.default_rng(20)
    n = 1 << 18
    g = rng.standard_normal((2, n))
    assert nf.fit(g, FS, nfft=256).family == "gaussian"
    k = nf.fit(g * np.sqrt(nf.texture("k", 1.5, (2, n), rng)), FS, nfft=256)
    assert k.family == "k" and k.tail_parameter == pytest.approx(1.5, rel=0.2)
    assert k.texture_time == 0.0 and not k.texture_common
    s = nf.fit(3.0 * g * np.sqrt(nf.texture("sas", 1.5, (2, n), rng)), FS, nfft=256, transient_threshold=None)
    assert s.family == "sas" and s.tail_parameter == pytest.approx(1.5, abs=0.08)
    # Regenerated: the same stable law, scale 3 / sqrt 2 (read off the characteristic function).
    y = s.generate(1 << 18, rng)
    for t in (0.2, 0.5):
        assert np.cos(t * y).mean() == pytest.approx(np.exp(-abs(t * 3.0 / np.sqrt(2.0)) ** 1.5), abs=0.03)


def test_texture_memory_and_commonality_recovered():
    rng = np.random.default_rng(21)
    n, hold = 1 << 18, 40
    tau = np.repeat(nf.texture("k", 2.0, (1, n // hold + 1), rng), hold, axis=1)[:, :n]
    model = nf.fit(rng.standard_normal((3, n)) * np.sqrt(tau), FS, nfft=256)
    assert model.family == "k" and model.tail_parameter == pytest.approx(2.0, rel=0.25)
    assert model.texture_common
    assert model.texture_time * FS == pytest.approx(hold, rel=0.3)
    y = model.generate(1 << 18, rng)
    # Bursts last: the power of neighbouring samples is correlated in the regenerated noise too.
    p = y[0] ** 2
    lag = np.corrcoef(p[:-10], p[10:])[0, 1]
    assert lag > 0.05
    assert np.corrcoef(y[0] ** 2, y[1] ** 2)[0, 1] > 0.05


def test_heavy_tailed_field_keeps_spectrum_coherence_and_tail():
    """The whole field modulated by a slow common K texture: the regenerated noise has the
    recording's spectrum and coherence (they do not depend on the tail) and its amplitude
    statistics."""
    rng = np.random.default_rng(22)
    n, hold = 1 << 18, 200
    tau = np.repeat(nf.texture("k", 1.5, n // hold + 1, rng), hold)[:n]
    x = make_field(n, rng) * np.sqrt(tau)
    model = nf.fit(x, FS, nfft=256, positions=POS)
    assert model.family == "k" and model.tail_parameter == pytest.approx(1.5, rel=0.35)
    y = model.generate(n, rng)
    stat = lambda a: (np.log(np.abs(a)).var(), np.mean(a**4) / np.mean(a**2) ** 2 - 3.0)  # noqa: E731
    (lv_x, k_x), (lv_y, k_y) = stat(x), stat(y)
    assert lv_y == pytest.approx(lv_x, abs=0.06)
    assert lv_x > 1.3 and k_x > 1.0                        # the recording is not Gaussian ...
    assert k_y == pytest.approx(k_x, rel=0.4)              # ... and neither is the regenerated noise
    f, p = welch_csd(y)
    got = np.real(np.einsum("fcc->fc", p))
    truth = truth_csd(f)
    t_psd = np.real(np.einsum("fcc->fc", truth))
    sel = (f > 200.0) & (f < 3800.0)
    assert np.abs(10.0 * np.log10(got[sel] / t_psd[sel]).mean(axis=1)).max() < 1.0
    coh = np.conj(p) / np.sqrt(got[:, :, None] * got[:, None, :])
    assert np.sqrt(np.mean(np.abs(coh[sel] - (truth / np.sqrt(t_psd[:, :, None] * t_psd[:, None, :]))[sel]) ** 2)) < 0.05
    # Forced Gaussian, the same spectra and no tail.
    z = model.generate(n, rng, family="gaussian")
    assert stat(z)[1] == pytest.approx(0.0, abs=0.1)


# -- guards -----------------------------------------------------------------------------------------------

def test_tones_bursts_and_transients_are_excluded_and_reported():
    rng = np.random.default_rng(30)
    n = 1 << 18
    clean = make_field(n, rng, interferer=False)
    t = np.arange(n) / FS
    x = clean.copy()
    tone_rms = 0.15
    x += np.sqrt(2.0) * tone_rms * np.cos(2.0 * np.pi * 1234.0 * t + rng.uniform(0, 2 * np.pi, (N_CH, 1)))
    burst = signal.chirp(np.arange(160) / FS, 1000.0, 0.02, 3000.0) * np.hanning(160) * 6.0     # a sync-like burst
    starts = (np.arange(0.2, t[-1] - 0.1, 0.5) * FS).astype(int)
    for s in starts:
        x[:, s:s + 160] += burst
    x[:, 100_000:108_000] += 1.5 * rng.standard_normal((N_CH, 8000))                          # a one-second event
    model = nf.fit(x, FS, nfft=256, positions=POS)
    r = model.report
    kinds = [e["kind"] for e in r["events"]]
    assert kinds.count("periodic burst") >= len(starts) - 2
    assert any(e["kind"] == "transient" and e["start_s"] < 100_000 / FS + 0.05 and e["end_s"] > 108_000 / FS - 0.05 for e in r["events"])
    period = [e["period_s"] for e in r["events"] if e["kind"] == "periodic burst"]
    assert np.median(period) == pytest.approx(0.5, abs=0.02)
    assert len(r["tones"]) == 1 and r["tones"][0]["frequency"] == pytest.approx(1234.0, abs=FS / 256)
    assert r["tones"][0]["rms"] == pytest.approx(tone_rms, rel=0.15)
    text = nf.format_report(model)
    assert "periodic burst" in text and "1234" in text or "1218" in text or "1250" in text
    # The fitted spectrum is the clean field's, at the tone too.
    f, psd = model.psd()
    want = 10.0 ** (iso_level_db(f) / 10.0)
    sel = (f > 200.0) & (f < 3800.0)
    assert np.abs(10.0 * np.log10(psd[sel].mean(axis=1) / want[sel])).max() < 0.6
    # Without the guards the same recording gives a spectrum that is wrong by many dB.
    raw = nf.fit(x, FS, nfft=256, transient_threshold=None, tone_threshold_db=np.inf, tail="gaussian")
    _, raw_psd = raw.psd()
    assert np.abs(10.0 * np.log10(raw_psd[sel].mean(axis=1) / want[sel])).max() > 6.0
    # The line can be put back.
    y = model.generate(1 << 17, rng, include_tones=True)
    fy, py = signal.welch(y, fs=FS, nperseg=1024, axis=1)
    k = int(np.argmin(np.abs(fy - 1234.0)))
    line_power = (py[:, k - 3:k + 4].sum(axis=1) * (fy[1] - fy[0])).mean() - 7 * (fy[1] - fy[0]) * 10.0 ** (iso_level_db(1234.0) / 10.0)
    assert line_power == pytest.approx(tone_rms**2, rel=0.3)


def test_impulsive_noise_is_not_mistaken_for_signals():
    rng = np.random.default_rng(31)
    n = 1 << 18
    x = rng.standard_normal((2, n)) * np.sqrt(nf.texture("sas", 1.6, (2, n), rng))
    model = nf.fit(x, FS, nfft=256)
    assert model.report["excluded_fraction"] < 0.02
    assert model.report["impulses_kept"] > 20
    assert model.family == "sas" and model.tail_parameter == pytest.approx(1.6, abs=0.1)


def test_short_recording_is_refused():
    rng = np.random.default_rng(32)
    with pytest.raises(ValueError, match="independent"):
        nf.fit(rng.standard_normal((8, 256 * 6)), FS, nfft=256)


# -- the model file and its use for another configuration ---------------------------------------------------

def test_model_file_round_trip_and_size(tmp_path, field):
    x, model = field
    path = tmp_path / "noise.h5"
    model.save(str(path))
    assert os.path.getsize(path) < x.astype(np.float32).nbytes / 20
    back = nf.NoiseModel.load(str(path))
    assert np.allclose(back.csd, model.csd, rtol=1e-5, atol=1e-12 * np.abs(model.csd).max() + 1e-12)
    assert back.family == model.family and back.sample_rate == FS and np.allclose(back.positions, POS)
    assert back.report["blocks_used"] == model.report["blocks_used"]
    a = model.generate(4096, np.random.default_rng(5))
    b = back.generate(4096, np.random.default_rng(5))
    assert np.allclose(a, b, rtol=1e-3, atol=1e-5 * np.abs(a).max())


def test_channel_and_sample_rate_rules(field):
    _, model = field
    rng = np.random.default_rng(40)
    with pytest.raises(ValueError, match="channels"):
        model.generate(1024, rng, n_channels=16)
    # A subset of the array: exact.
    y = model.generate(1 << 17, rng, channels=[0, 3, 7], n_channels=3)
    f, p = welch_csd(y)
    k = int(np.argmin(np.abs(f - 2000.0)))
    want = truth_csd(np.array([f[k]]))[0][np.ix_([0, 3, 7], [0, 3, 7])]
    got = np.conj(p[k])
    assert np.abs(got / np.sqrt(np.outer(np.diag(got).real, np.diag(got).real))
                  - want / np.sqrt(np.outer(np.diag(want).real, np.diag(want).real))).max() < 0.08
    # A higher sample rate asks for frequencies the recording never saw: refused, or zero there.
    with pytest.raises(ValueError, match="does not cover"):
        model.generate(1024, rng, sample_rate=2 * FS)
    z = model.generate(1 << 16, rng, sample_rate=2 * FS, allow_uncovered=True)
    fz, pz = signal.welch(z, fs=2 * FS, nperseg=512, axis=1)
    assert pz[:, fz > 4300.0].max() < 1e-3 * pz[:, (fz > 500) & (fz < 3500)].min()
    sel = (fz > 300.0) & (fz < 3500.0)
    assert np.abs(10.0 * np.log10(pz[:, sel].mean(axis=0) / np.real(np.einsum("fcc->f", truth_csd(fz[sel])) / N_CH))).max() < 1.0
    # A lower one: the band it covers, at the recording's density.
    lo = model.generate(1 << 16, rng, sample_rate=FS / 2)
    fl, pl = signal.welch(lo, fs=FS / 2, nperseg=128, axis=1)
    sel = (fl > 300.0) & (fl < 1700.0)
    assert np.abs(10.0 * np.log10(pl[:, sel].mean(axis=0) / np.real(np.einsum("fcc->f", truth_csd(fl[sel])) / N_CH))).max() < 1.0


def test_complex_baseband_output_has_the_band_power_doubled(field):
    _, model = field
    rng = np.random.default_rng(41)
    fc, fs_bb = 2000.0, 2000.0
    z = model.generate(1 << 16, rng, sample_rate=fs_bb, center_frequency=fc)
    assert np.iscomplexobj(z)
    f = np.linspace(fc - fs_bb / 2, fc + fs_bb / 2, 400)
    band_power = np.trapezoid(np.real(np.einsum("fcc->fc", truth_csd(f))), f, axis=0)       # one-sided density over the band
    assert (np.abs(z) ** 2).mean(axis=1) == pytest.approx(2.0 * band_power, rel=0.08)
    # The interferer's phase across the array survives the move to baseband.
    lag = (z[0] * np.conj(z[1])).mean()
    f0, p = welch_csd(model.generate(1 << 16, rng))
    sel = (f0 > fc - fs_bb / 2) & (f0 < fc + fs_bb / 2)
    assert np.angle(lag) == pytest.approx(np.angle(np.conj(p[sel, 0, 1]).sum()), abs=0.1)


# -- recordings on disk and the scenario hook ---------------------------------------------------------------

def test_load_wav_and_record_container(tmp_path):
    from scipy.io import wavfile
    rng = np.random.default_rng(50)
    x = (rng.standard_normal((40000, 3)) * 3000).astype(np.int16)
    wavfile.write(tmp_path / "a.wav", 16000, x)
    rec = nf.load_recording(str(tmp_path / "a.wav"))
    assert rec.n_channels == 3 and rec.sample_rate == 16000.0 and rec.segments[0].shape == (3, 40000)
    # Records: each ping a segment; the board's reference input is dropped.
    path = str(tmp_path / "passive.h5")
    with gs.RecordWriter(path) as w:
        for i in range(6):
            r = make_record(channels=4, samples=4096, complex_data=False, seed=i)
            meta = r.metadata
            board = gs.core.Board()
            board.reference_channel = 3
            meta.boards = [board]
            meta.sequence = i
            r.metadata = meta
            w.write(r)
    rec = nf.load_recording(path)
    assert rec.n_channels == 3 and rec.dropped_channels == [3] and len(rec.segments) == 6
    assert rec.positions.shape == (3, 3)
    model = nf.fit(rec, nfft=128)
    assert model.report["dropped_reference_channels"] == [3]
    assert model.report["blocks_total"] == 6 * (4096 // 64 - 1)                              # no block spans two records
    f, psd = model.psd()
    assert 10.0 * np.log10(psd[5:-5].mean() / (2.0 / 64000.0)) == pytest.approx(0.0, abs=0.3)   # unit white noise


def test_scenario_hook(tmp_path, field):
    _, model = field
    path = str(tmp_path / "m.h5")
    model.save(path)
    rng = np.random.default_rng(51)
    cfg = {"model": "fitted", "file": path, "gain_db": 6.0}
    y = nf.scenario_noise(cfg, 1 << 15, FS, rng, n_channels=9, hydrophone_channels=list(range(8)))
    assert y.shape == (9, 1 << 15) and not y[8].any()
    want = np.real(np.einsum("fcc->c", model.csd)) * FS / model.nfft * 10.0 ** 0.6
    assert y[:8].var(axis=1) == pytest.approx(want, rel=0.1)
    with pytest.raises(ValueError, match="channels"):
        nf.scenario_noise(cfg, 1024, FS, rng, n_channels=16)


def test_fit_noise_tool(tmp_path, capsys):
    """tools/fit_noise.py on a .wav with a heavy tail, a line and periodic bursts: the report names
    them, and the tail is fitted with the line notched out of the whitened recording."""
    import importlib.util
    import pathlib
    from scipy.io import wavfile
    spec = importlib.util.spec_from_file_location("fit_noise", pathlib.Path(__file__).resolve().parents[2] / "tools" / "fit_noise.py")
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    rng = np.random.default_rng(60)
    n, hold = 1 << 18, 100
    tau = np.repeat(nf.texture("k", 1.5, n // hold + 1, rng), hold)[:n]
    x = make_field(n, rng) * np.sqrt(tau)
    t = np.arange(n) / FS
    x += np.sqrt(2.0) * 0.15 * np.cos(2.0 * np.pi * 1234.0 * t + rng.uniform(0, 2 * np.pi, (N_CH, 1)))
    for s in (np.arange(0.2, t[-1] - 0.1, 1.0) * FS).astype(int):
        x[:, s:s + 160] += 6.0 * np.hanning(160) * np.cos(2.0 * np.pi * 2000.0 * t[:160])
    wavfile.write(tmp_path / "rec.wav", int(FS), (x / 12.0).astype(np.float32).T)
    out = tmp_path / "model.h5"
    assert tool.main([str(tmp_path / "rec.wav"), "--out", str(out), "--nfft", "256", "--check"]) == 0
    text = capsys.readouterr().out
    assert "periodic burst" in text and "Lines taken out" in text and "Round trip" in text and "K-type" in text
    model = nf.NoiseModel.load(str(out))
    assert model.family == "k" and model.tail_parameter == pytest.approx(1.5, rel=0.3)
    assert model.texture_common and model.texture_time * FS == pytest.approx(hold, rel=0.35)
    assert len(model.tones) == 1
