"""Noise models: ambient spectra, self-noise, and Gaussian noise with a given spectrum
and the right coherence across an array.

Ambient spectrum: the widely used four-term empirical fit to the Wenz curves (turbulence,
shipping, wind, thermal), as given by Coates (1990) and repeated in the underwater
communications literature. The coefficients were checked against two independent
copies of the formulas (the ns-3 UAN noise model and arXiv:1801.03641); the original
book was not consulted. Levels are spectrum levels in dB re 1 uPa^2/Hz. A third open
print, read 2026-11-18: Hurt, "Effect of wind noise on undersea acoustic network performance
and design", MS thesis, Naval Postgraduate School (2005), https://hdl.handle.net/10945/1936,
eqs. (2.7)-(2.9) (wind, attributed there to Wenz 1962; thermal and shipping, attributed to
Urick 1983) and its appendix listing (turbulence 17 - 30 log10 f): the same constants, f in
kHz, wind speed in m/s, shipping density 0 to 1 (tests/python/test_literature_models.py).
Its eq. (2.7) prints "7.5 w" where its own listing has 7.5 w^0.5; the square root is right.
"""

from __future__ import annotations

import numpy as np

from ..acoustics import NOMINAL_SOUND_SPEED


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


def isotropic_coherence(positions, frequency: float, sound_speed: float = NOMINAL_SOUND_SPEED) -> np.ndarray:
    """Coherence matrix of 3-D isotropic noise between omnidirectional sensors:
    sin(k d) / (k d), with d the separation."""
    p = np.asarray(positions, dtype=float)
    d = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=2)
    return np.sinc(2.0 * frequency * d / sound_speed)


def surface_noise_coherence(positions, frequency: float, sound_speed: float = NOMINAL_SOUND_SPEED, nodes: int = 96) -> np.ndarray:
    """Coherence matrix (complex, Hermitian) of noise made at the sea surface, between
    omnidirectional sensors in deep, uniform water: the Cron-Sherman model. The noise generators
    use it with `coherence: surface` (`receiver.ambient_noise.coherence`); the default is the isotropic matrix.

    The model: sound arrives from above only, with power per unit solid angle proportional to
    cos(t), t the angle of the arrival from the vertical (a sheet of surface dipoles, no
    attenuation, no bottom, azimuthal symmetry). For two sensors a distance d apart, the line
    between them at angle g from the vertical (positions with z positive down, the second
    sensor deeper when g = 0),

        Gamma = 2 integral_0^1 u exp(i k d u cos g) J0(k d sin g sqrt(1 - u^2)) du,   u = cos t

    (our evaluation, by Gauss-Legendre quadrature). For a vertical pair it is the closed form
    printed as eq. (4) of Shajahan, Barclay and Lin, "Mapping of surface-generated noise
    coherence", Front. Mar. Sci. 9:1016702 (2022; CC BY; after Cron and Sherman 1962 and
    Buckingham 2013),

        Gamma = 2 [sin x / x + (cos x - 1) / x^2] + 2 i [sin x / x^2 - cos x / x],   x = k d,

    whose real part first crosses zero at d = 0.371 wavelengths (tan(x / 2) = x), before the
    half wavelength of the isotropic sin(x) / x (their eq. (3)), and whose imaginary part is the
    up-down asymmetry; for a horizontal pair it is 2 J1(x) / x (first zero at 0.61 wavelengths).
    The same paper shows the model failing in shallow or range-dependent water (the first zero
    crossing moves by up to 30 %), where the bottom returns sound from below."""
    from scipy.special import j0
    p = np.asarray(positions, dtype=float)
    sep = p[None, :, :] - p[:, None, :]                     # from sensor i to sensor j
    d = np.linalg.norm(sep, axis=2)
    k = 2.0 * np.pi * float(frequency) / float(sound_speed)
    along = k * sep[:, :, 2]                                 # k d cos g
    across = k * np.sqrt(np.maximum(d**2 - sep[:, :, 2] ** 2, 0.0))
    x, w = np.polynomial.legendre.leggauss(int(nodes))
    u, w = 0.5 * (x + 1.0), 0.5 * w                         # nodes on 0..1
    kernel = np.exp(1j * along[:, :, None] * u) * j0(across[:, :, None] * np.sqrt(1.0 - u * u))
    return 2.0 * np.sum(w * u * kernel, axis=2)


COHERENCE_MODELS = ("isotropic", "surface", "independent")
DEFAULT_COHERENCE = "isotropic"


def coherence_matrix(model: str, positions, frequency: float, sound_speed: float) -> np.ndarray | None:
    """Coherence between hydrophones at one frequency by a named model: `isotropic` (sin(kd)/(kd)), `surface`
    (noise made at the sea surface: `surface_noise_coherence`, complex) or `independent` (None: no mixing)."""
    if model == "independent":
        return None
    if model == "isotropic":
        return isotropic_coherence(positions, frequency, sound_speed)
    if model == "surface":
        return surface_noise_coherence(positions, frequency, sound_speed)
    raise ValueError(f"unknown noise coherence {model!r}: one of {', '.join(COHERENCE_MODELS)}")


def _mixing(gamma: np.ndarray) -> np.ndarray:
    """A factor L with L L^H = gamma (a coherence matrix): Cholesky, or the eigen-decomposition when closely
    spaced hydrophones make the matrix singular to rounding."""
    n = gamma.shape[0]
    try:
        return np.linalg.cholesky(gamma + 1e-9 * np.eye(n))
    except np.linalg.LinAlgError:
        w, v = np.linalg.eigh(gamma)
        return v * np.sqrt(np.maximum(w, 0.0))


def flat_spectrum(spectrum_db, frequency: float):
    """A spectrum held at its level at `frequency` (`spectrum: flat`)."""
    level = float(np.asarray(spectrum_db(np.array([float(frequency)])))[0])
    return lambda f: np.full_like(np.asarray(f, dtype=float), level)


def baseband_gaussian_noise(n_samples: int, sample_rate: float, center_frequency: float, spectrum_db, rng: np.random.Generator,
                            positions=None, sound_speed: float = NOMINAL_SOUND_SPEED, sensitivity_db: float = 0.0,
                            n_channels: int | None = None, n_bands: int = 96, coherence: str = DEFAULT_COHERENCE) -> np.ndarray:
    """Complex-envelope Gaussian noise [channels, samples] for a baseband record: the band `sample_rate` wide about
    `center_frequency`, with the one-sided spectrum `spectrum_db(f)` across it (dB re 1 uPa^2/Hz at the passband
    frequency f = center_frequency + baseband frequency; `sensitivity_db` converts to sample units) and, with
    `positions`, the coherence between hydrophones of `coherence` at each passband frequency (`coherence_matrix`),
    imposed per band as in `gaussian_noise`.

    The complex envelope carries twice the power of the real signal in the band: E|x|^2 = 2 x integral of the
    spectrum over the band, so a flat spectrum S gives E|x|^2 = 2 S sample_rate (what the earlier white generator
    made). The cross-spectrum of two channels at passband frequency f is 2 S(f) Gamma_ij(f), Gamma the model's
    coherence with the suite's exp(+i w t) convention; an isotropic field gives a real Gamma, noise from the surface
    a complex one (the deeper hydrophone hears it later)."""
    if positions is not None:
        positions = np.asarray(positions, dtype=float)
        n_channels = positions.shape[0]
    if n_channels is None:
        raise ValueError("give positions or n_channels")
    order = np.argsort(np.fft.fftfreq(n_samples, 1.0 / sample_rate), kind="stable")      # bins by passband frequency
    f = center_frequency + np.fft.fftfreq(n_samples, 1.0 / sample_rate)
    psd = 10.0 ** ((np.asarray(spectrum_db(np.maximum(f, 1.0)), dtype=float) + sensitivity_db) / 10.0)
    psd = np.where(f > 0.0, psd, 0.0)                         # nothing below 0 Hz (a band wider than twice the carrier)
    # x = ifft(X): E|x|^2 = sum E|X_k|^2 / N^2, and E|x|^2 = 2 sum_k S_k fs / N, so E|X_k|^2 = 2 S_k fs N.
    scale = np.sqrt(2.0 * psd * sample_rate * n_samples)
    white = (rng.standard_normal((n_channels, n_samples)) + 1j * rng.standard_normal((n_channels, n_samples))) / np.sqrt(2.0)
    if positions is not None and coherence != "independent":
        edges = np.linspace(0, n_samples, min(int(n_bands), n_samples) + 1).astype(int)
        for a, b in zip(edges[:-1], edges[1:]):
            if b <= a:
                continue
            bins = order[a:b]
            gamma = coherence_matrix(coherence, positions, max(float(f[bins[len(bins) // 2]]), 1.0), sound_speed)
            white[:, bins] = _mixing(gamma) @ white[:, bins]
    return np.fft.ifft(white * scale[None, :], axis=1).astype(np.complex64)


def gaussian_noise(n_samples: int, sample_rate: float, spectrum_db, rng: np.random.Generator,
                   positions=None, sound_speed: float = NOMINAL_SOUND_SPEED, sensitivity_db: float = 0.0,
                   n_channels: int | None = None, n_bands: int = 96, coherence: str = DEFAULT_COHERENCE) -> np.ndarray:
    """Real Gaussian noise [channels, samples] with one-sided power spectral density
    `spectrum_db(f)` (dB re 1 uPa^2/Hz, converted to sample units by `sensitivity_db`).

    With `positions` [channels, 3] the channels have the coherence of `coherence` (`coherence_matrix`): by
    default isotropic noise, sin(kd)/(kd) at each frequency; `surface`: noise made at the sea surface;
    `independent`, or without positions: none. Coherence is imposed with a Cholesky factor per frequency
    band (`n_bands` bands across 0 to Nyquist).
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
    if positions is not None and coherence != "independent":
        edges = np.linspace(0, len(f), n_bands + 1).astype(int)
        for a, b in zip(edges[:-1], edges[1:]):
            if b <= a:
                continue
            if coherence == "isotropic":                      # as it always was, bit for bit
                gamma = isotropic_coherence(positions, f[(a + b) // 2], sound_speed)
                chol = np.linalg.cholesky(gamma + 1e-9 * np.eye(n_channels))
            else:
                chol = _mixing(coherence_matrix(coherence, positions, f[(a + b) // 2], sound_speed))
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
    if model in ("alpha_stable", "sas"):
        if "alpha" not in cfg:
            raise ValueError("alpha-stable noise needs statistics.alpha (0.1 to 2; 2 is Gaussian)")
        alpha = float(cfg["alpha"])
        if not 0.1 <= alpha <= 2.0:
            raise ValueError("statistics.alpha must be in the range 0.1 to 2")
        return {"model": "alpha_stable", "alpha": alpha}
    if model != "k":
        raise ValueError(f"unknown noise statistics {model!r}: use 'gaussian', 'k' or 'alpha_stable'")
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
    return {"model": "k", "shape": shape, "correlation_time": tau, "common": bool(cfg.get("common", True))}


def alpha_stable(alpha: float, scale, size, rng: np.random.Generator) -> np.ndarray:
    """Independent symmetric alpha-stable numbers, characteristic function exp(-|scale t|^alpha),
    by the method of Chambers, Mallows and Stuck (JASA 71, 1976; not opened):

        x = scale (cos((1 - alpha) p) / w)^(1 / alpha - 1) sin(alpha p) / cos(p)^(1 / alpha)

    p uniform on (-pi/2, pi/2), w exponential with unit mean. The formula was read in arlpy
    1.9.3 (`arlpy.stable.rnd`, its symmetric branch; BSD 3-clause, Copyright (c) 2016 Mandar
    Chitre), which cites that paper; this is our own code for it. At alpha = 2 it is
    2 scale sqrt(w) sin(p) identically: a Gaussian of variance 2 scale^2 (the Box-Muller
    form). At alpha = 1 it is scale tan(p), a Cauchy variate. For alpha < 2 the variance is
    infinite (and for alpha <= 1 the mean absolute value too): `scale` is the only measure
    of level."""
    if not 0.1 <= alpha <= 2.0:
        raise ValueError("alpha must be in the range 0.1 to 2")
    p = (rng.uniform(size=size) - 0.5) * np.pi
    w = -np.log(1.0 - rng.uniform(size=size))
    return scale * (np.cos((1.0 - alpha) * p) / w) ** (1.0 / alpha - 1.0) * np.sin(alpha * p) / np.cos(p) ** (1.0 / alpha)


def alpha_stable_noise(n_samples: int, sample_rate: float, spectrum_db, rng: np.random.Generator, alpha: float,
                       sensitivity_db: float = 0.0, n_channels: int = 1) -> np.ndarray:
    """Impulsive ambient noise [channels, samples]: independent symmetric alpha-stable samples
    (snapping shrimp are the usual example), white over the record's band and independent
    between hydrophones.

    Level. A stable law with alpha < 2 has no variance, so a spectrum level cannot be its
    power density. The configured spectrum fixes the SCALE instead: scale = sigma / sqrt(2),
    sigma^2 the power the Gaussian noise of that spectrum would have in the band 0 to
    sample_rate / 2 (the integral of the spectrum). At alpha = 2 the result is then exactly
    Gaussian noise of that power; for alpha < 2 it is the stable noise with the same scale
    (the same core of the distribution, heavier tails), and any measured "level" grows with
    the length of the record. The shape of the spectrum is not reproduced (white in band):
    a colouring filter would change the distribution, and no model with memory was taken
    from a source."""
    f = np.fft.rfftfreq(n_samples, 1.0 / sample_rate)
    psd = 10.0 ** ((np.asarray(spectrum_db(np.maximum(f, 1.0)), dtype=float) + sensitivity_db) / 10.0)
    psd[0] = 0.0
    sigma = np.sqrt(np.sum(psd) * sample_rate / n_samples)
    return alpha_stable(alpha, sigma / np.sqrt(2.0), (int(n_channels), n_samples), rng).astype(np.float32)


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
            positions=None, sound_speed: float = NOMINAL_SOUND_SPEED, sensitivity_db: float = 0.0,
            n_channels: int | None = None, n_bands: int = 96, coherence: str = DEFAULT_COHERENCE) -> np.ndarray:
    """`gaussian_noise` with other statistics (`coherence`: as there; the white K samples of correlation_time 0
    are mixed with the isotropic matrix or not at all). `statistics` = {"model": "alpha_stable", "alpha"}:
    `alpha_stable_noise` (white in band, independent channels; read its note on level). Or
    {"model": "k", "shape": nu, "correlation_time": seconds, "common": bool} (see
    `noise_statistics`): K-distributed noise, in two cases.

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
        return gaussian_noise(n_samples, sample_rate, spectrum_db, rng, **kwargs, coherence=coherence)
    if stats["model"] == "alpha_stable":
        channels = n_channels if positions is None else np.asarray(positions).shape[0]
        return alpha_stable_noise(n_samples, sample_rate, spectrum_db, rng, stats["alpha"], sensitivity_db, channels)
    if stats["correlation_time"] > 0.0:
        x = gaussian_noise(n_samples, sample_rate, spectrum_db, rng, **kwargs, coherence=coherence)
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
    if positions is not None and coherence != "independent":
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
