"""Noise models: ambient spectra, self-noise, and Gaussian noise with a given spectrum
and the right coherence across an array.

Ambient spectrum: the widely used four-term empirical fit to the Wenz curves (turbulence,
shipping, wind, thermal), as given by Coates (1990) and repeated in the underwater
communications literature. **The coefficients below are written from memory and have not
been checked against a source in this repository's build environment; verify before
relying on absolute levels.** Levels are spectrum levels in dB re 1 uPa^2/Hz.
"""

from __future__ import annotations

import numpy as np


def wenz_spectrum_db(frequency, wind_speed: float = 5.0, shipping: float = 0.5,
                     components: bool = False):
    """Ambient noise spectrum level at `frequency` (Hz).

    wind_speed in m/s; shipping activity from 0 (light) to 1 (heavy).
    """
    f = np.maximum(np.asarray(frequency, dtype=float), 1.0) / 1e3      # kHz
    turbulence = 17.0 - 30.0 * np.log10(f)
    ship = 40.0 + 20.0 * (shipping - 0.5) + 26.0 * np.log10(f) - 60.0 * np.log10(f + 0.03)
    wind = 50.0 + 7.5 * np.sqrt(wind_speed) + 20.0 * np.log10(f) - 40.0 * np.log10(f + 0.4)
    thermal = -15.0 + 20.0 * np.log10(f)
    parts = np.stack([turbulence, ship, wind, thermal])
    total = 10.0 * np.log10((10.0 ** (parts / 10.0)).sum(axis=0))
    if components:
        return total, {"turbulence": turbulence, "shipping": ship, "wind": wind, "thermal": thermal}
    return total


def self_noise_spectrum_db(frequency, broadband_db: float = -300.0, slope_db_per_octave: float = 0.0,
                           reference_frequency: float = 1e3):
    """A configurable platform/flow noise spectrum: a level at the reference frequency
    with a constant slope per octave. Tones are added separately (`add_tones`)."""
    f = np.maximum(np.asarray(frequency, dtype=float), 1.0)
    return broadband_db + slope_db_per_octave * np.log2(f / reference_frequency)


def isotropic_coherence(positions, frequency: float, sound_speed: float = 1500.0) -> np.ndarray:
    """Coherence matrix of 3-D isotropic noise between omnidirectional sensors:
    sin(k d) / (k d), with d the separation."""
    p = np.asarray(positions, dtype=float)
    d = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=2)
    return np.sinc(2.0 * frequency * d / sound_speed)


def gaussian_noise(n_samples: int, sample_rate: float, spectrum_db, rng: np.random.Generator,
                   positions=None, sound_speed: float = 1500.0, sensitivity_db: float = 0.0,
                   n_channels: int | None = None, n_bands: int = 96) -> np.ndarray:
    """Real Gaussian noise [channels, samples] with one-sided power spectral density
    `spectrum_db(f)` (dB re 1 uPa^2/Hz, converted to sample units by `sensitivity_db`).

    With `positions` [channels, 3] the channels have the coherence of isotropic noise,
    sin(kd)/(kd) at each frequency; without, they are independent. Coherence is imposed
    with a Cholesky factor per frequency band (`n_bands` bands across 0 to Nyquist).
    """
    if positions is not None:
        positions = np.asarray(positions, dtype=float)
        n_channels = positions.shape[0]
    if n_channels is None:
        raise ValueError("give positions or n_channels")
    f = np.fft.rfftfreq(n_samples, 1.0 / sample_rate)
    psd = 10.0 ** ((np.asarray(spectrum_db(np.maximum(f, 1.0)), dtype=float) + sensitivity_db) / 10.0)
    psd[0] = 0.0
    # One-sided PSD S(f): variance per rfft bin of the unnormalised FFT is S * fs * N / 2
    # (shared between real and imaginary parts).
    scale = np.sqrt(psd * sample_rate * n_samples / 4.0)
    white = rng.standard_normal((n_channels, len(f))) + 1j * rng.standard_normal((n_channels, len(f)))
    if positions is not None:
        edges = np.linspace(0, len(f), n_bands + 1).astype(int)
        for a, b in zip(edges[:-1], edges[1:]):
            if b <= a:
                continue
            gamma = isotropic_coherence(positions, f[(a + b) // 2], sound_speed)
            chol = np.linalg.cholesky(gamma + 1e-9 * np.eye(n_channels))
            white[:, a:b] = chol @ white[:, a:b]
    return np.fft.irfft(white * scale[None, :], n=n_samples, axis=1).astype(np.float32)


def add_tones(noise: np.ndarray, sample_rate: float, tones, rng: np.random.Generator) -> np.ndarray:
    """Add narrowband self-noise lines: `tones` is a list of (frequency Hz, rms in sample units).
    Each channel gets a random phase."""
    t = np.arange(noise.shape[1]) / sample_rate
    out = noise.astype(np.float64)
    for freq, rms in tones:
        phase = rng.uniform(0.0, 2.0 * np.pi, (noise.shape[0], 1))
        out += np.sqrt(2.0) * rms * np.cos(2.0 * np.pi * freq * t[None, :] + phase)
    return out.astype(np.float32)


def k_texture(positions, shape: float, cell: float, rng: np.random.Generator) -> np.ndarray:
    """Amplitude texture that turns Rayleigh speckle into K-distributed clutter.

    The seafloor is divided into square patches of side `cell` (larger than a resolution
    cell); each patch gets a gamma-distributed power with unit mean and shape parameter
    `shape`. Scatterer amplitudes multiplied by the square root of their patch's value give
    speckle whose intensity is K distributed: scintillation index 1 + 2 / shape (1 for
    pure Rayleigh, which is the limit of large shape).
    """
    p = np.asarray(positions, dtype=float)
    ix = np.floor(p[:, 0] / cell).astype(np.int64)
    iy = np.floor(p[:, 1] / cell).astype(np.int64)
    keys, inverse = np.unique(np.stack([ix, iy], axis=1), axis=0, return_inverse=True)
    power = rng.gamma(shape, 1.0 / shape, len(keys))
    return np.sqrt(power[np.ravel(inverse)])
