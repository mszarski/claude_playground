"""Noise models: ambient spectra, self-noise, and Gaussian noise with a given spectrum
and the right coherence across an array.

Ambient spectrum: the widely used four-term empirical fit to the Wenz curves (turbulence,
shipping, wind, thermal), as given by Coates (1990) and repeated in the underwater
communications literature. The coefficients were checked against two independent
copies of the formulas (the ns-3 UAN noise model and arXiv:1801.03641); the original
book was not consulted. Levels are spectrum levels in dB re 1 uPa^2/Hz.
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


def noise_statistics(cfg) -> dict | None:
    """Check a `statistics` mapping of the noise configuration and return it normalised:
    None for Gaussian noise (no mapping, `model: gaussian`, or an infinite shape), else
    {"shape", "correlation_time", "common"}."""
    if not cfg:
        return None
    model = str(cfg.get("model", "gaussian")).lower()
    if model == "gaussian":
        return None
    if model != "k":
        raise ValueError(f"unknown noise statistics {model!r}: use 'gaussian' or 'k'")
    if "shape" not in cfg:
        raise ValueError("K-distributed noise needs statistics.shape")
    shape = float(cfg["shape"])
    if not shape > 0.0:
        raise ValueError("K-distributed noise needs a positive shape")
    tau = float(cfg.get("correlation_time", 0.0) or 0.0)
    if tau < 0.0:
        raise ValueError("statistics.correlation_time must not be negative")
    if np.isinf(shape):
        return None
    return {"shape": shape, "correlation_time": tau, "common": bool(cfg.get("common", True))}


def gamma_texture(n_samples: int, sample_rate: float, shape: float, correlation_time: float,
                  rng: np.random.Generator, n_channels: int = 1, common: bool = True) -> np.ndarray:
    """Amplitude factor [channels, samples] that makes Gaussian noise K distributed: the
    square root of a gamma variate of unit mean and shape `shape`.

    The variate is constant over cells `correlation_time` long and independent from cell to
    cell (its autocovariance is a triangle that reaches zero at that lag); the first cell
    starts at a random point so the cell edges are not tied to the record. With
    `correlation_time` 0 every sample has its own variate. `common`: one texture for all
    channels (rows are the same array), else one per channel.
    """
    cell = max(int(round(float(correlation_time) * float(sample_rate))), 1)
    rows = 1 if common else int(n_channels)
    offset = int(rng.integers(0, cell)) if cell > 1 else 0
    n_cells = -(-(n_samples + offset) // cell)
    tau = rng.gamma(shape, 1.0 / shape, (rows, n_cells))
    out = np.sqrt(np.repeat(tau, cell, axis=1)[:, offset:offset + n_samples]) if cell > 1 else np.sqrt(tau[:, :n_samples])
    return np.broadcast_to(out, (int(n_channels), n_samples)) if common else out


def k_noise(n_samples: int, sample_rate: float, spectrum_db, rng: np.random.Generator, statistics: dict | None,
            positions=None, sound_speed: float = 1500.0, sensitivity_db: float = 0.0,
            n_channels: int | None = None, n_bands: int = 96) -> np.ndarray:
    """`gaussian_noise` with K statistics: `statistics` = {"model": "k", "shape": nu,
    "correlation_time": seconds, "common": bool} (see `noise_statistics`). Two cases.

    correlation_time > 0 (the compound model, as for reverberation): the Gaussian noise of
    `gaussian_noise`, spectrum and array coherence included, times `gamma_texture`. The mean
    power is unchanged; the spectrum is convolved with the texture's, 1 / correlation_time
    wide, so it is unchanged at any resolution coarser than that. With `common` (default) all
    hydrophones share the texture: a burst of noise reaches the whole array at once (travel
    time across the array is neglected against the correlation time), and the coherence
    between channels is exactly that of the Gaussian field. With `common: false` each channel
    has its own, and the coherence between two channels falls by E{sqrt(tau)}^2 =
    (Gamma(nu + 1/2) / Gamma(nu))^2 / nu.

    correlation_time = 0: TR 0702's KNoise (Sec. 4.6.2): white noise with a fresh
    K-distributed number for each sample and channel, then the filter that gives it the
    spectrum (and, with `positions`, the mixing that gives the array coherence). The filter
    brings the samples back towards Gaussian (TR 0702 says so): the kurtosis 3 / nu holds for a
    white spectrum only.

    No statistics, `model: gaussian` or an infinite shape: `gaussian_noise` itself, bit for
    bit, with nothing more drawn from `rng`.
    """
    stats = noise_statistics(statistics)
    kwargs = dict(positions=positions, sound_speed=sound_speed, sensitivity_db=sensitivity_db, n_channels=n_channels, n_bands=n_bands)
    if stats is None:
        return gaussian_noise(n_samples, sample_rate, spectrum_db, rng, **kwargs)
    if stats["correlation_time"] > 0.0:
        x = gaussian_noise(n_samples, sample_rate, spectrum_db, rng, **kwargs)
        return x * gamma_texture(n_samples, sample_rate, stats["shape"], stats["correlation_time"], rng,
                                 x.shape[0], stats["common"]).astype(np.float32)
    if positions is not None:
        positions = np.asarray(positions, dtype=float)
        n_channels = positions.shape[0]
    if n_channels is None:
        raise ValueError("give positions or n_channels")
    f = np.fft.rfftfreq(n_samples, 1.0 / sample_rate)
    psd = 10.0 ** ((np.asarray(spectrum_db(np.maximum(f, 1.0)), dtype=float) + sensitivity_db) / 10.0)
    psd[0] = 0.0
    white = rng.standard_normal((n_channels, n_samples)) * gamma_texture(n_samples, sample_rate, stats["shape"], 0.0, rng, n_channels, False)
    spectrum = np.fft.rfft(white, axis=1)                    # unit-variance white: E|X|^2 = N per bin
    if positions is not None:
        edges = np.linspace(0, len(f), n_bands + 1).astype(int)
        for a, b in zip(edges[:-1], edges[1:]):
            if b > a:
                gamma = isotropic_coherence(positions, f[(a + b) // 2], sound_speed)
                spectrum[:, a:b] = np.linalg.cholesky(gamma + 1e-9 * np.eye(n_channels)) @ spectrum[:, a:b]
    return np.fft.irfft(spectrum * np.sqrt(psd * sample_rate / 2.0)[None, :], n=n_samples, axis=1).astype(np.float32)


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
