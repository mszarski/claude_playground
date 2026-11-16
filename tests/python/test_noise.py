"""Milestone 12 (part): noise spectra, array coherence, K-distributed clutter."""

import numpy as np
import pytest
from scipy.signal import csd, welch

from gpusonar.sim import Scene, lambert_rms_amplitude, make_pose, seafloor_patch, simulate_ping
from gpusonar.sim import noise as nz
from gpusonar.sonar import SonarModel


def test_wenz_spectrum_shape():
    f = np.array([100.0, 1e3, 8e3, 27.5e3, 300e3])
    quiet = nz.wenz_spectrum_db(f, wind_speed=2.0, shipping=0.2)
    rough = nz.wenz_spectrum_db(f, wind_speed=15.0, shipping=0.2)
    busy = nz.wenz_spectrum_db(f, wind_speed=2.0, shipping=0.9)
    assert np.all(rough >= quiet) and np.all(busy >= quiet)
    assert rough[2] - quiet[2] > 10.0            # wind sets the level at 8 kHz
    assert busy[0] - quiet[0] > 10.0             # shipping sets it at 100 Hz
    assert abs(busy[2] - quiet[2]) < 1.0         # and has no say at 8 kHz
    total, parts = nz.wenz_spectrum_db(f, components=True)
    assert parts["thermal"][-1] > parts["wind"][-1]          # thermal noise takes over by 300 kHz
    assert np.all(np.diff(nz.wenz_spectrum_db(np.array([1e3, 2e3, 4e3, 8e3, 16e3]))) < 0)   # falls with frequency here


def test_generated_noise_has_the_requested_spectrum():
    fs, n = 64000.0, 1 << 18
    rng = np.random.default_rng(0)
    x = nz.gaussian_noise(n, fs, lambda f: nz.wenz_spectrum_db(f, wind_speed=8.0), rng,
                          n_channels=4, sensitivity_db=-180.0)
    f, pxx = welch(x, fs=fs, nperseg=4096, axis=1)
    measured = 10 * np.log10(pxx.mean(axis=0))
    wanted = nz.wenz_spectrum_db(np.maximum(f, 1.0), wind_speed=8.0) - 180.0
    band = (f > 500.0) & (f < 30000.0)
    assert np.abs(measured[band] - wanted[band]).max() < 1.0


def test_white_noise_variance_is_the_spectrum_times_the_band():
    fs, n = 64000.0, 1 << 16
    x = nz.gaussian_noise(n, fs, lambda f: np.full_like(f, 50.0), np.random.default_rng(4),
                          n_channels=8, sensitivity_db=-180.0)
    # One-sided level S over 0..fs/2: variance = S * fs / 2. Same rule as
    # sonar_equation.noise_rms_volts.
    from gpusonar.sonar_equation import noise_rms_volts
    assert x.std() == pytest.approx(noise_rms_volts(50.0, -180.0, fs), rel=0.02)


def test_isotropic_noise_coherence_across_the_array():
    fs, n = 64000.0, 1 << 18
    sonar = SonarModel.load("flank_32")
    positions = sonar.positions[:8]                       # 9.4 cm pitch
    rng = np.random.default_rng(1)
    x = nz.gaussian_noise(n, fs, lambda f: np.full_like(f, 60.0), rng, positions=positions,
                          sensitivity_db=-180.0)
    f, p00 = welch(x[0], fs=fs, nperseg=1024)
    for j in (1, 3):
        _, pjj = welch(x[j], fs=fs, nperseg=1024)
        _, p0j = csd(x[0], x[j], fs=fs, nperseg=1024)
        measured = np.real(p0j) / np.sqrt(p00 * pjj)
        d = np.linalg.norm(positions[j] - positions[0])
        wanted = np.sinc(2 * f * d / 1500.0)
        band = (f > 1000.0) & (f < 20000.0)
        assert np.abs(measured[band] - wanted[band]).max() < 0.12
    # At half-wavelength spacing (8 kHz for 9.4 cm) adjacent hydrophones are uncorrelated:
    # that is what makes the array gain 10 log N.
    assert abs(np.sinc(2 * 7979.0 * 0.094 / 1500.0)) < 0.01


def test_tones_appear_as_lines():
    fs, n = 64000.0, 1 << 16
    rng = np.random.default_rng(2)
    base = nz.gaussian_noise(n, fs, lambda f: np.full_like(f, -80.0), rng, n_channels=2)
    x = nz.add_tones(base, fs, [(3000.0, 0.5)], rng)
    f, pxx = welch(x[0], fs=fs, nperseg=8192, scaling="spectrum")
    i = np.argmin(np.abs(f - 3000.0))
    assert pxx[i] == pytest.approx(0.25, rel=0.1)      # rms^2 of the line (it sits on a bin)


def clutter_scintillation(shape, seed=3):
    """Scintillation index of seafloor echo intensity with and without patch texture."""
    c, density = 1500.0, 60.0
    sonar = SonarModel.load("sas_auv", {
        "receive": {"kind": "baseband", "sample_rate": 100e3, "window": {"range": [35.0, 75.0]}},
        "transmit": {"waveform": {"type": "gaussian", "f_start": 180e3, "f_stop": 220e3,
                                  "duration": 1e-4, "window_alpha": 0.0}}})
    rng = np.random.default_rng(seed)
    pts, amp = seafloor_patch((-12.0, 12.0), (30.0, 75.0), density,
                              rms_amplitude=lambert_rms_amplitude(-27.0, density), rng=rng)
    if shape is not None:
        amp = amp * nz.k_texture(pts, shape, cell=4.0, rng=rng)
    rec = simulate_ping(sonar, make_pose(position=(0.0, 0.0, -15.0)), Scene().add_points(pts, amp, lambert=True),
                        environment={"sound_speed": c}, absorption_model="none", adc=False)
    x = rec.samples.array
    # Remove the smooth range trend (spreading, beam pattern) with a low-order fit that
    # cannot follow 4 m patches, then pool channels and ranges.
    inten = np.abs(x[:, 400:3600]) ** 2
    n = np.arange(inten.shape[1])
    coeffs = np.polyfit(n, np.log(inten.mean(axis=0) + 1e-300), 3)
    inten = inten / np.exp(np.polyval(coeffs, n))[None, :]
    inten = inten / inten.mean()
    return inten.var() / inten.mean() ** 2


def test_clutter_can_be_made_k_distributed():
    rayleigh = clutter_scintillation(None)
    k = clutter_scintillation(1.0)
    # Rayleigh speckle has scintillation index 1 (checked properly, without a beam
    # pattern in the way, in test_reverberation.py); the low-order trend removal here
    # leaves it a little above. K with shape 1 is 1 + 2/shape = 3 in the ideal case, less
    # here because a beam footprint averages over more than one 4 m patch.
    assert rayleigh < 1.4
    assert k > rayleigh + 0.6
