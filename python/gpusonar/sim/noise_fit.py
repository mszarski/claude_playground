"""A noise generator fitted to a multichannel recording.

Input: a passive (no transmit) recording of the array, [channels, samples]. Output: a small
model (`NoiseModel`, saved as HDF5: the cross-spectral matrix on a frequency grid and a
handful of numbers, not the recording) that generates new noise of any length with the
recording's

- power spectrum per channel,
- cross-spectra between channels (so the coherence against spacing and frequency, and
  everything a beamformer does with it),
- correlation in time (it is the same information as the spectra),
- amplitude statistics: Gaussian, or heavy-tailed with a fitted tail parameter.

The model
---------
    x = B * eta ,     eta_c[n] = innovation_scale * sqrt(tau_c[n]) * g_c[n]

g is white unit Gaussian noise, independent between channels and samples; `*` is a
multichannel convolution with filters B whose transfer matrix satisfies
B(f) B(f)^H = S(f) fs, S the two-sided cross-spectral density matrix of the recording; tau is
a positive "texture" that carries the tail:

    gaussian   tau = 1
    k          tau ~ gamma(shape nu, mean 1): eta is K-type (kurtosis excess 3 / nu)
    sas        tau = positive stable variate of index alpha / 2: eta is symmetric alpha-stable
               (an alpha-sub-Gaussian vector when the texture is common to the channels)

tau is constant over `texture_len` samples (1: a new value every sample: impulsive clicks;
long: bursts) and either common to all channels or independent between them. Because g is
white and independent of tau, eta is white whatever tau does, so the spectra and coherences
of x are those of B for every tail setting (for `sas` in the sense of the underlying Gaussian:
an alpha-stable variate has no variance).

Where each piece comes from
---------------------------
- The factor B: per-frequency Cholesky factor of the cross-spectral matrix. This is APL-UW
  TR 0702 eq. (23) (R. P. Goddard, "The Sonar Simulation Toolset, Release 4.6", 2008, Sec.
  4.6.1: "SST uses Cholesky factorization"). For Gaussian noise the default generator is
  TR 0702's eqs. (24)-(26) as printed ("Mitchell-McPherson"): X_c(f, t_u) = sum_c'
  G_cc'(f) g_c'(f, t_u) with independent complex unit Gaussians per frequency and update,
  inverse transform, multiply by a window with sum_u |w(t - t_u)|^2 = 1 (a sine window at 50 %
  overlap is used: sin^2 + cos^2 = 1; the Mitchell-McPherson window itself is in a 1981 paper
  that was not opened), and add.
- Heavy tails in the time domain, not the frequency domain: TR 0702 Sec. 4.6.2 ("If we
  simply substituted spiky random numbers for the Gaussian ones in Eq. (24), most of the
  spikiness would be lost in the subsequent inverse Fourier transform ... generate white
  noise using a fresh, independent K-distributed random number for each sample ... [then]
  apply a filter"). That is the `eta` -> B form above.
- The same form, "mixing coefficients applied to independent innovations", is what
  NoiseModels.jl (org-arl, MIT; `src/gaussian.jl` read) and the Underwater Acoustic Channel
  Repository noise files (uwa-channels, MIT; file specification read) use:
  n_i = sum_j sum_k beta_ij(k) eta_j(n + k). Adopted from them: the form, and the convention
  that the innovations are symmetric alpha-stable of scale 1/sqrt(2) so that alpha = 2 is the
  standard normal (UACR file specification, "Noise file format"). NOT adopted: their
  estimator (NoiseModels.jl minimises the squared error of the lagged covariances over the
  mixing coefficients by BFGS with automatic differentiation; here B is obtained in closed
  form from the Welch cross-spectral matrix, which needs no optimiser and gives the
  estimator variance below).
- The sub-Gaussian construction (a Gaussian vector times the square root of a positive
  alpha/2-stable variate) is the one in `asgn.m` of the alpha-SGNm toolbox (A. Mahmood,
  BSD 3-clause; read: its first block draws `A = stabrnd(alpha/2, 1, 2 cos(pi alpha/4)^(2/alpha), 0)`
  and returns `chol(R) sqrt(A T) S`). NOT adopted: its alpha-SGN(m) sampler, in which every
  m + 1 adjacent samples are jointly sub-Gaussian; that needs tabulated densities and a
  rejection loop. Here the memory of the bursts is a texture held for `texture_len` samples.
- The positive stable variate is drawn with Kanter's formula; the tail estimators use the
  mean and variance of log|x| (formulas below). These are derived in the comments and tested
  against their defining properties (tests/python/test_noise_fit.py), not copied.

Guard against fitting signals as noise
--------------------------------------
- Reference inputs (the capture boards' synchronisation channels) are dropped by the loader.
- Transients: the recording is cut into half-overlapping blocks; a block is flagged when its
  power, relative to the channel's median block power, is above `transient_threshold` robust
  standard deviations. Runs of flagged blocks that last `event_blocks` or more, and flagged
  blocks that recur at a fixed interval (pings, sync bursts), are EXCLUDED from the fit and
  listed. Isolated flagged blocks are kept as part of the noise (impulsive noise looks like
  this) and counted.
- Tonals: frequency bins that stand out of the median-smoothed spectrum by more than
  `tone_threshold_db` are listed, removed from the cross-spectral matrix (interpolated across)
  and kept separately as lines (frequency and complex amplitude per channel), which
  `generate(include_tones=True)` adds back.

What the fit assumes, and how good it is
----------------------------------------
- Stationarity over the recording. Nothing checks it beyond the transient test; the report
  gives the spread of block power so that a drifting level shows.
- Block length `nfft / fs` much longer than the sound's transit time across the array;
  otherwise the coherence of a directional source is biased low by (1 - transit/block)^2.
- With n_d independent blocks the coherence estimate |gamma^| has, for complex Gaussian
  blocks: where the true coherence is zero, |gamma^|^2 ~ Beta(1, n_d - 1), i.e. a floor
  E|gamma^| = sqrt(pi / (4 n_d)); elsewhere a standard deviation (1 - |gamma|^2) / sqrt(2 n_d)
  (first-order). Half-overlapping Hann blocks count as n_d = K / (1 + 2 rho^2 (1 - 1/K)) with
  rho the overlap correlation of the window (1/6 for Hann): computed from the window. The
  report prints n_d, both numbers, and the record length needed for a given accuracy. The
  spectrum level has a standard deviation of 4.34 / sqrt(n_d) dB.
- The Cholesky factor needs n_d well above the number of channels (refused below 2 x channels).
- The tail is fitted to the whitened recording e = B^-1 x. The triangular factor is one of
  many; when independent impulsive sources sit on strongly coherent channels the whitened
  channels are mixtures of them and look more Gaussian than the sources.

Using the model for another configuration
-----------------------------------------
- Channel count must match the model's, or `channels=[...]` must pick a subset (exact: a
  sub-matrix of the cross-spectral matrix). Anything else is refused.
- Any sample rate: the cross-spectral matrix is a function of frequency in Hz and is
  interpolated to the new grid (same resolution in Hz) and factored again. Frequencies the
  recording did not cover get no noise; that is refused unless `allow_uncovered=True`.
- A complex-baseband output about `center_frequency` from a real recording is the band
  moved down, with twice the power (the complex envelope's), as the simulator does.
- `texture_len` is kept in seconds. The K shape is fitted per sample of the recording; at
  another rate it is used unchanged, which is approximate.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np

from ..acoustics import NOMINAL_SOUND_SPEED

FORMAT_VERSION = 1
_EULER = 0.5772156649015329
_VAR_LOG_NORMAL = np.pi**2 / 8.0          # variance of log|g|, g standard normal (= trigamma(1/2) / 4)


# =================================================================================================
# Heavy-tailed variates and their log-moments
# =================================================================================================

def positive_stable(a: float, size, rng: np.random.Generator) -> np.ndarray:
    """Positive stable variates S of index a (0 < a <= 1) with Laplace transform
    E exp(-s S) = exp(-s^a)  (Kanter's representation; S = 1 for a = 1).

        S = sin(a U) / sin(U)^(1/a) * (sin((1 - a) U) / E)^((1 - a) / a),   U ~ U(0, pi), E ~ Exp(1)
    """
    if not 0.0 < a <= 1.0:
        raise ValueError("positive_stable: index must be in (0, 1]")
    if a > 1.0 - 1e-9:
        return np.ones(size)
    u = rng.uniform(0.0, np.pi, size)
    u = np.clip(u, 1e-12, np.pi - 1e-12)
    e = rng.exponential(1.0, size)
    return np.sin(a * u) / np.sin(u) ** (1.0 / a) * (np.sin((1.0 - a) * u) / e) ** ((1.0 - a) / a)


def texture(family: str, parameter: float, size, rng: np.random.Generator) -> np.ndarray:
    """The texture tau of the module docstring. With g standard normal, sqrt(tau) g is
    `gaussian`: N(0, 1); `k`: unit variance, kurtosis excess 3 / parameter; `sas`: symmetric
    alpha-stable (alpha = parameter) of scale 1/sqrt(2), characteristic function
    exp(-|t / sqrt 2|^alpha): E exp(i t sqrt(S) g) = E exp(-S t^2 / 2) = exp(-(t^2 / 2)^(alpha/2))."""
    if family == "gaussian":
        return np.ones(size)
    if family == "k":
        return rng.gamma(parameter, 1.0 / parameter, size)
    if family == "sas":
        return positive_stable(parameter / 2.0, size, rng)
    raise ValueError(f"unknown tail family {family!r}")


def log_abs_moments(family: str, parameter: float) -> tuple[float, float]:
    """Mean and variance of log|sqrt(tau) g| for the three families (g standard normal).

    log|g|: mean -(euler + log 2) / 2, variance pi^2 / 8.
    gamma(nu, mean 1): log tau has mean digamma(nu) - log(nu), variance trigamma(nu).
    positive stable of index a: E S^p = Gamma(1 - p/a) / Gamma(1 - p) (from the Laplace
    transform), so log S has mean euler (1/a - 1) and variance (pi^2 / 6)(1/a^2 - 1).
    With a = alpha / 2 the variance of log|x| is (pi^2 / 6)(1/alpha^2 + 1/2)."""
    from scipy.special import polygamma
    m0, v0 = -(_EULER + np.log(2.0)) / 2.0, _VAR_LOG_NORMAL
    if family == "gaussian":
        return m0, v0
    if family == "k":
        return m0 + 0.5 * (float(polygamma(0, parameter)) - np.log(parameter)), v0 + 0.25 * float(polygamma(1, parameter))
    if family == "sas":
        a = parameter / 2.0
        return m0 + 0.5 * _EULER * (1.0 / a - 1.0), v0 + 0.25 * (np.pi**2 / 6.0) * (1.0 / a**2 - 1.0)
    raise ValueError(f"unknown tail family {family!r}")


def _k_shape_from_log_variance(v: float) -> float:
    from scipy.optimize import brentq
    from scipy.special import polygamma
    excess = 4.0 * (v - _VAR_LOG_NORMAL)
    if excess <= float(polygamma(1, 1e3)):
        return 1e3
    if excess >= float(polygamma(1, 0.05)):
        return 0.05
    return float(brentq(lambda nu: float(polygamma(1, nu)) - excess, 0.05, 1e3))


def _alpha_from_log_variance(v: float) -> float:
    x = 6.0 * v / np.pi**2 - 0.5
    return float(np.clip(1.0 / np.sqrt(max(x, 1e-9)), 0.3, 2.0))


def _tail_ratio(a: np.ndarray) -> float:
    """Quantile ratio q(0.9995) / q(0.5) of |a|: 5.19 for a Gaussian."""
    q = np.quantile(np.abs(a), [0.5, 0.9995])
    return float(q[1] / max(q[0], 1e-300))


# =================================================================================================
# Recording
# =================================================================================================

@dataclass
class Recording:
    """Segments [channels, samples] of one array at one sample rate. `center_frequency` > 0
    means complex baseband samples about that frequency."""
    segments: list
    sample_rate: float
    center_frequency: float = 0.0
    positions: np.ndarray | None = None
    channels: list | None = None                 # indices of the kept channels in the file
    dropped_channels: list = field(default_factory=list)
    segment_times: list | None = None            # start time of each segment, s
    source: str = ""
    notes: list = field(default_factory=list)

    @property
    def n_channels(self) -> int:
        return int(self.segments[0].shape[0])

    @property
    def duration(self) -> float:
        return sum(s.shape[1] for s in self.segments) / self.sample_rate


def load_recording(path: str, channels=None) -> Recording:
    """Read a recording: a .wav (all channels), a WAV sidecar .yaml (docs/RECORD_FORMAT.md; its
    `files`, `scale` and element positions are used, not its ping times: the whole file is taken),
    or the suite's HDF5 record container (each record a segment; the boards' reference inputs
    are dropped; records that carry a transmit waveform are noted, since a fit wants none)."""
    path = str(path)
    ext = os.path.splitext(path)[1].lower()
    notes: list = []
    positions = None
    dropped: list = []
    times = None
    fc = 0.0
    if ext == ".wav":
        from scipy.io import wavfile
        rate, data = wavfile.read(path)
        data = np.atleast_2d(np.asarray(data, dtype=np.float64).T)
        if data.shape[0] > data.shape[1]:
            data = data.T
        segments = [data]
    elif ext in (".yaml", ".yml"):
        import yaml
        from scipy.io import wavfile
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        base = os.path.dirname(os.path.abspath(path))
        files = cfg.get("files") or [cfg["file"]]
        blocks, rates = [], set()
        for name in files:
            r, d = wavfile.read(os.path.join(base, name))
            d = np.asarray(d, dtype=np.float64)
            blocks.append(d.T if d.ndim == 2 else d[None, :])
            rates.add(float(r))
        if len(rates) != 1 or len({b.shape[1] for b in blocks}) != 1:
            raise ValueError("audio files disagree on sample rate or length")
        rate = rates.pop()
        segments = [np.concatenate(blocks, axis=0) * float(cfg.get("scale", 1.0))]
        el = cfg.get("elements")
        if isinstance(el, dict) and "positions" in el:
            positions = np.asarray(el["positions"], dtype=float)
        elif isinstance(el, list):
            try:
                positions = np.asarray(el, dtype=float).reshape(-1, 3)
            except (TypeError, ValueError):
                positions = None
        if cfg.get("transmit"):
            notes.append("the sidecar describes a transmit: make sure the sonar was not pinging")
    elif ext in (".h5", ".hdf5"):
        from ..sources import ContainerSource
        src = ContainerSource(path)
        segments, times = [], []
        rate = None
        active = 0
        for rec in src:
            s = rec.samples
            x = np.asarray(s.array)
            x = x.astype(np.complex128 if np.iscomplexobj(x) else np.float64)
            if rate is None:
                rate = float(s.sample_rate)
                fc = float(s.center_frequency) if np.iscomplexobj(x) else 0.0
                positions = np.asarray(rec.rx_positions(), dtype=float)
                dropped = sorted({int(b.reference_channel) for b in rec.metadata.boards if int(b.reference_channel) >= 0})
            elif float(s.sample_rate) != rate or x.shape[0] != segments[0].shape[0]:
                raise ValueError("records disagree on sample rate or channel count")
            active += int(float(rec.transmit.waveform.duration) > 0.0)
            segments.append(x)
            times.append(float(rec.metadata.sequence))
        times = None
        if not segments:
            raise ValueError(f"{path}: no records")
        if active:
            notes.append(f"{active} of {len(segments)} records describe a transmit pulse: make sure the sonar was not pinging")
    else:
        raise ValueError(f"{path}: expected .wav, a WAV sidecar .yaml or a record container .h5")
    n_ch = segments[0].shape[0]
    keep = [c for c in range(n_ch) if c not in dropped]
    if channels is not None:
        keep = [int(c) for c in channels]
    segments = [s[keep] for s in segments]
    if positions is not None and len(positions) == n_ch:
        positions = positions[keep]
    else:
        positions = None
    return Recording(segments=segments, sample_rate=float(rate), center_frequency=fc, positions=positions,
                     channels=keep, dropped_channels=[c for c in dropped if c not in keep], segment_times=times,
                     source=os.path.basename(path), notes=notes)


# =================================================================================================
# The model
# =================================================================================================

def effective_averages(n_blocks: int, window: np.ndarray, hop: int) -> float:
    """Number of independent blocks that `n_blocks` blocks `hop` apart are worth for a spectrum
    of Gaussian noise: K / (1 + 2 rho^2 (1 - 1/K)), rho = sum w[n] w[n + hop] / sum w^2."""
    if n_blocks <= 1:
        return float(max(n_blocks, 0))
    rho = float(np.sum(window[:-hop] * window[hop:]) / np.sum(window**2)) if hop < len(window) else 0.0
    return n_blocks / (1.0 + 2.0 * rho**2 * (1.0 - 1.0 / n_blocks))


def coherence_floor(n_d: float) -> float:
    """E|gamma^| where the true coherence is zero, n_d independent complex Gaussian blocks."""
    return float(np.sqrt(np.pi / (4.0 * max(n_d, 1.0))))


def coherence_std(gamma: float, n_d: float) -> float:
    """First-order standard deviation of |gamma^| at true coherence magnitude `gamma`."""
    return float((1.0 - gamma**2) / np.sqrt(2.0 * max(n_d, 1.0)))


def _factor(s2: np.ndarray, ridge: float = 1e-9) -> np.ndarray:
    """Lower-triangular G[f] with G G^H = S[f] (TR 0702 eq. 23). Bins with no power give zeros."""
    f, c, _ = s2.shape
    out = np.zeros_like(s2)
    tr = np.real(np.trace(s2, axis1=1, axis2=2)) / c
    ok = tr > 0.0
    if ok.any():
        a = s2[ok] + (ridge * tr[ok])[:, None, None] * np.eye(c)
        a = 0.5 * (a + np.conj(np.swapaxes(a, 1, 2)))
        try:
            out[ok] = np.linalg.cholesky(a)
        except np.linalg.LinAlgError:
            fac = np.zeros_like(a)
            for i in range(len(a)):
                try:
                    fac[i] = np.linalg.cholesky(a[i])
                except np.linalg.LinAlgError:                          # not positive definite: eigen factor
                    w, v = np.linalg.eigh(a[i])
                    fac[i] = v * np.sqrt(np.maximum(w, 0.0))[None, :]
            out[ok] = fac
    return out


def _sine_window(n: int) -> np.ndarray:
    return np.sin(np.pi * (np.arange(n) + 0.5) / n)


@dataclass
class NoiseModel:
    """Fitted noise: the two-sided cross-spectral density `csd` [nfft, C, C] (sample units^2/Hz)
    on the frequencies np.fft.fftfreq(nfft, 1 / sample_rate) (+ center_frequency for a
    complex-baseband recording), the tail, and the lines that were taken out."""
    csd: np.ndarray
    sample_rate: float
    center_frequency: float = 0.0
    is_complex: bool = False
    family: str = "gaussian"
    tail_parameter: float = 0.0                  # K shape nu, or alpha
    innovation_scale: float = 1.0
    texture_time: float = 0.0                    # s the texture is held (0: one sample)
    texture_common: bool = True
    tones: list = field(default_factory=list)    # [{frequency, amplitude: [C] complex rms}]
    positions: np.ndarray | None = None
    report: dict = field(default_factory=dict)

    @property
    def n_channels(self) -> int:
        return int(self.csd.shape[1])

    @property
    def nfft(self) -> int:
        return int(self.csd.shape[0])

    @property
    def frequency(self) -> np.ndarray:
        return np.fft.fftfreq(self.nfft, 1.0 / self.sample_rate)

    # -- files ---------------------------------------------------------------------------------

    def save(self, path: str) -> None:
        import h5py
        with h5py.File(path, "w") as h:
            h.attrs["format"] = "gpusonar-noise-model"
            h.attrs["version"] = FORMAT_VERSION
            h.attrs["sample_rate"] = self.sample_rate
            h.attrs["center_frequency"] = self.center_frequency
            h.attrs["is_complex"] = bool(self.is_complex)
            h.attrs["family"] = self.family
            h.attrs["tail_parameter"] = self.tail_parameter
            h.attrs["innovation_scale"] = self.innovation_scale
            h.attrs["texture_time"] = self.texture_time
            h.attrs["texture_common"] = bool(self.texture_common)
            h.attrs["report_json"] = json.dumps(self.report, default=float)
            # A real recording's matrix at -f is the conjugate of that at f: half is stored.
            half = self.csd if self.is_complex else self.csd[: self.nfft // 2 + 1]
            h.attrs["nfft"] = self.nfft
            h.create_dataset("csd", data=half.astype(np.complex128), compression="gzip")
            if self.positions is not None:
                h.create_dataset("positions", data=np.asarray(self.positions, dtype=np.float64))
            if self.tones:
                h.create_dataset("tone_frequency", data=np.array([t["frequency"] for t in self.tones], dtype=np.float64))
                h.create_dataset("tone_amplitude", data=np.array([t["amplitude"] for t in self.tones], dtype=np.complex128))

    @classmethod
    def load(cls, path: str) -> "NoiseModel":
        import h5py
        with h5py.File(path, "r") as h:
            if h.attrs.get("format") != "gpusonar-noise-model":
                raise ValueError(f"{path}: not a fitted noise model")
            if int(h.attrs["version"]) > FORMAT_VERSION:
                raise ValueError(f"{path}: noise model version {int(h.attrs['version'])} is newer than this code")
            tones = []
            if "tone_frequency" in h:
                tones = [{"frequency": float(f), "amplitude": np.asarray(a)} for f, a in zip(h["tone_frequency"][:], h["tone_amplitude"][:])]
            csd = h["csd"][:].astype(np.complex128)
            if not bool(h.attrs["is_complex"]):
                n = int(h.attrs["nfft"])
                csd = np.concatenate([csd, np.conj(csd[1:n - n // 2][::-1])], axis=0)
            return cls(csd=csd, sample_rate=float(h.attrs["sample_rate"]),
                       center_frequency=float(h.attrs["center_frequency"]), is_complex=bool(h.attrs["is_complex"]),
                       family=str(h.attrs["family"]), tail_parameter=float(h.attrs["tail_parameter"]),
                       innovation_scale=float(h.attrs["innovation_scale"]), texture_time=float(h.attrs["texture_time"]),
                       texture_common=bool(h.attrs["texture_common"]), tones=tones,
                       positions=h["positions"][:] if "positions" in h else None,
                       report=json.loads(h.attrs["report_json"]))

    # -- the spectra ---------------------------------------------------------------------------

    def psd(self, one_sided: bool = True):
        """(frequency Hz, PSD [F, C]). One-sided (real recordings): the positive frequencies with
        twice the two-sided density, the usual spectrum level."""
        d = np.real(np.einsum("fcc->fc", self.csd))
        if one_sided and not self.is_complex:
            h = self.nfft // 2
            return self.frequency[1:h], 2.0 * d[1:h]
        order = np.argsort(self.frequency)
        return self.frequency[order] + self.center_frequency, d[order]

    def coherence(self) -> np.ndarray:
        """Complex coherence [nfft, C, C]: csd_ij / sqrt(csd_ii csd_jj)."""
        d = np.sqrt(np.maximum(np.real(np.einsum("fcc->fc", self.csd)), 1e-300))
        return self.csd / (d[:, :, None] * d[:, None, :])

    def csd_at(self, frequency) -> np.ndarray:
        """Cross-spectral density [len(frequency), C, C] at absolute frequencies (Hz), linearly
        interpolated; zero outside what the recording covered."""
        f = np.asarray(frequency, dtype=float) - self.center_frequency
        order = np.argsort(self.frequency)
        grid, s = self.frequency[order], self.csd[order]
        df = self.sample_rate / self.nfft
        pos = (f - grid[0]) / df
        i0 = np.clip(np.floor(pos).astype(int), 0, self.nfft - 2)
        w = np.clip(pos - i0, 0.0, 1.0)[:, None, None]
        out = (1.0 - w) * s[i0] + w * s[i0 + 1]
        inside = (f >= grid[0] - 1e-9) & (f <= grid[-1] + df + 1e-9)
        if not self.is_complex:
            inside &= np.abs(f) <= self.sample_rate / 2.0 + 1e-9
        return np.where(inside[:, None, None], out, 0.0)

    def covers(self, f_low: float, f_high: float) -> bool:
        if self.is_complex:
            return f_low >= self.center_frequency - self.sample_rate / 2 - 1e-6 and f_high <= self.center_frequency + self.sample_rate / 2 + 1e-6
        return f_low >= -1e-6 and f_high <= self.sample_rate / 2.0 * (1.0 + 1e-9)

    # -- generation ----------------------------------------------------------------------------

    def _target(self, sample_rate, center_frequency, channels, allow_uncovered):
        """(two-sided density on the target grid [n, C', C'], n, fs, complex?) for a target
        sample rate / baseband centre / channel subset."""
        fs = float(self.sample_rate if sample_rate is None else sample_rate)
        complex_out = self.is_complex if center_frequency is None else float(center_frequency) > 0.0
        fc = self.center_frequency if center_frequency is None else float(center_frequency)
        if self.is_complex and not complex_out:
            raise ValueError("noise model: a complex-baseband recording cannot give real passband noise")
        lo, hi = (fc - fs / 2.0, fc + fs / 2.0) if complex_out else (0.0, fs / 2.0)
        if not self.covers(lo, hi) and not allow_uncovered:
            raise ValueError(f"noise model: the recording does not cover {lo:.0f}-{hi:.0f} Hz (it covers "
                             f"{'%.0f-%.0f' % (self.center_frequency - self.sample_rate / 2, self.center_frequency + self.sample_rate / 2) if self.is_complex else '0-%.0f' % (self.sample_rate / 2)} Hz); "
                             "pass allow_uncovered=True to get no noise where it has no data")
        same = fs == self.sample_rate and complex_out == self.is_complex and fc == self.center_frequency
        if same:
            s2, n = self.csd, self.nfft
        else:
            n = max(16, 2 * int(round(self.nfft * fs / self.sample_rate / 2.0)))
            f = np.fft.fftfreq(n, 1.0 / fs) + (fc if complex_out else 0.0)
            s2 = self.csd_at(f)
            if complex_out and not self.is_complex:
                s2 = 4.0 * s2                       # complex envelope of a real band: 2 x the one-sided density
        if channels is not None:
            idx = np.asarray(channels, dtype=int)
            s2 = s2[:, idx][:, :, idx]
        return s2, n, fs, complex_out

    def generate(self, n_samples: int, rng: np.random.Generator | None = None, *, sample_rate: float | None = None,
                 center_frequency: float | None = None, n_channels: int | None = None, channels=None,
                 method: str = "auto", include_tones: bool = False, allow_uncovered: bool = False,
                 family: str | None = None) -> np.ndarray:
        """A realisation [channels, n_samples] (float64, or complex128 for a baseband target).

        method: "mitchell_mcpherson" (TR 0702 eqs. 24-26; Gaussian only), "filter" (white
        innovations through the mixing filters; any tail), or "auto" (the first for a Gaussian
        model, the second otherwise). `n_channels`, when given, must equal the model's (after
        `channels`): the rule for a sonar with a different array is to refuse."""
        rng = rng or np.random.default_rng()
        s2, n, fs, complex_out = self._target(sample_rate, center_frequency, channels, allow_uncovered)
        c = s2.shape[1]
        if n_channels is not None and int(n_channels) != c:
            raise ValueError(f"noise model has {c} channels, the sonar has {n_channels}: fit a recording of this "
                             "array, or pick the matching channels with channels=[...]")
        family = self.family if family is None else family
        if method == "auto":
            method = "mitchell_mcpherson" if family == "gaussian" else "filter"
        g = _factor(s2)
        if method == "mitchell_mcpherson":
            if family != "gaussian":
                raise ValueError("Mitchell-McPherson synthesis is Gaussian (TR 0702 Sec. 4.6.2); use method='filter'")
            x = self._mitchell_mcpherson(g, n, fs, n_samples, complex_out, rng)
        elif method == "filter":
            x = self._filtered(g, n, fs, n_samples, complex_out, family, rng)
        else:
            raise ValueError("method is 'auto', 'mitchell_mcpherson' or 'filter'")
        if include_tones and self.tones:
            t = np.arange(n_samples) / fs
            fc = (self.center_frequency if center_frequency is None else float(center_frequency)) if complex_out else 0.0
            for tone in self.tones:
                amp = np.asarray(tone["amplitude"])
                if channels is not None:
                    amp = amp[np.asarray(channels, dtype=int)]
                ph = rng.uniform(0.0, 2.0 * np.pi)
                if complex_out:
                    if abs(tone["frequency"] - fc) < fs / 2.0:
                        x = x + np.sqrt(2.0) * amp[:, None] * np.exp(1j * (2.0 * np.pi * (tone["frequency"] - fc) * t[None, :] + ph))
                elif tone["frequency"] < fs / 2.0:
                    x = x + np.sqrt(2.0) * np.abs(amp)[:, None] * np.cos(2.0 * np.pi * tone["frequency"] * t[None, :] + np.angle(amp)[:, None] + ph)
        return x

    @staticmethod
    def _mitchell_mcpherson(g, n, fs, n_samples, complex_out, rng):
        """TR 0702 eqs. (24)-(26): per update, X(f) = G(f) g(f) with independent complex unit
        Gaussians, inverse transform, sine window (sum of squares 1 at 50 % overlap), add."""
        c = g.shape[1]
        hop = n // 2
        w = _sine_window(n)
        n_upd = int(np.ceil(n_samples / hop)) + 1
        out = np.zeros((c, (n_upd + 1) * hop), dtype=np.complex128)
        scale = np.sqrt(n * fs)
        for u in range(n_upd):
            z = (rng.standard_normal((n, c)) + 1j * rng.standard_normal((n, c))) / np.sqrt(2.0)
            xf = np.einsum("fab,fb->af", g, z)
            out[:, u * hop:u * hop + n] += np.fft.ifft(xf, axis=1) * scale * w
        out = out[:, hop:hop + n_samples]          # the first half block has one window only
        return out if complex_out else np.sqrt(2.0) * out.real

    def _filtered(self, g, n, fs, n_samples, complex_out, family, rng):
        """White innovations sqrt(tau) g through the mixing filters ifft(G(f) sqrt(fs))."""
        c = g.shape[1]
        taps = mixing_filters(g, fs)                                    # [C, C, n], centre at n // 2
        total = n_samples + n
        eta = rng.standard_normal((c, total))
        if complex_out:
            eta = (eta + 1j * rng.standard_normal((c, total))) / np.sqrt(2.0)
        if family != "gaussian":
            hold = max(1, int(round(self.texture_time * fs)))
            n_tau = total // hold + 2
            tau = texture(family, self.tail_parameter, (1 if self.texture_common else c, n_tau), rng)
            tau = np.repeat(tau, hold, axis=1)[:, int(rng.integers(0, hold)):][:, :total]
            eta = eta * np.sqrt(tau)
        eta = eta * self.innovation_scale
        y = convolve_matrix(taps, eta)
        y = y[:, n // 2 + n // 2: n // 2 + n // 2 + n_samples]
        return y if complex_out else y.real


def mixing_filters(g: np.ndarray, fs: float) -> np.ndarray:
    """Time-domain mixing filters [C, C, n] of the factor G[f] (fft order): the inverse transform
    of G sqrt(fs), circularly centred at n // 2 and tapered with a Hann window so that the ends
    meet at zero (a smoothing of the spectrum by about one bin)."""
    n = g.shape[0]
    h = np.fft.ifft(g * np.sqrt(fs), axis=0)
    h = np.roll(h, n // 2, axis=0) * np.hanning(n + 1)[:n, None, None]
    return np.moveaxis(h, 0, 2)


def convolve_matrix(taps: np.ndarray, x: np.ndarray) -> np.ndarray:
    """y_a = sum_b taps[a, b] * x_b (full convolution, [A, len(x) + n - 1]) by overlap-add FFT."""
    a, b, n = taps.shape
    total = x.shape[1]
    block = 3 * n
    size = 1 << int(np.ceil(np.log2(block + n)))
    hf = np.fft.fft(taps, size, axis=2)
    out = np.zeros((a, total + n - 1 + size), dtype=np.complex128)
    for start in range(0, total, block):
        seg = x[:, start:start + block]
        yf = np.einsum("abf,bf->af", hf, np.fft.fft(seg, size, axis=1))
        out[:, start:start + size] += np.fft.ifft(yf, axis=1)
    out = out[:, :total + n - 1]
    return out if (np.iscomplexobj(taps) and np.abs(taps.imag).max() > 0) or np.iscomplexobj(x) else out.real


# =================================================================================================
# The fit
# =================================================================================================

def _frames(segments, nfft: int, hop: int):
    """(segment index, start sample) of every block."""
    out = []
    for i, s in enumerate(segments):
        for start in range(0, s.shape[1] - nfft + 1, hop):
            out.append((i, start))
    return out


def _classify_events(flag: np.ndarray, seg_of: np.ndarray, event_blocks: int):
    """Runs of flagged blocks -> (exclude mask, list of events). Long runs and runs that recur
    at a fixed interval are excluded; other runs are kept as impulsive noise."""
    exclude = np.zeros(len(flag), dtype=bool)
    events = []
    i = 0
    while i < len(flag):
        if not flag[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(flag) and flag[j + 1] and seg_of[j + 1] == seg_of[i]:
            j += 1
        events.append({"first": i, "last": j, "kind": "transient" if (j - i + 1) >= event_blocks else "impulse"})
        i = j + 1
    short = [e for e in events if e["kind"] == "impulse"]
    if len(short) >= 5:
        starts = np.array([e["first"] for e in short], dtype=float)
        gaps = np.diff(starts)
        period = float(np.median(gaps))
        if period >= 2.0:
            mult = gaps / period
            regular = np.abs(mult - np.round(mult)) * period <= 1.0
            if regular.mean() >= 0.8 and (np.round(mult) == 1).mean() >= 0.5:
                for e in short:
                    e["kind"] = "periodic burst"
                    e["period_blocks"] = period
    for e in events:
        if e["kind"] != "impulse":
            exclude[e["first"]:e["last"] + 1] = True
    return exclude, events


def fit(recording, sample_rate: float | None = None, *, nfft: int = 1024, smooth: int = 1,
        transient_threshold: float | None = 8.0, event_blocks: int = 4, tone_threshold_db: float | None = None,
        tail: str = "auto", tail_samples: int = 1 << 20, positions=None, sound_speed: float = NOMINAL_SOUND_SPEED) -> NoiseModel:
    """Fit a `NoiseModel` to a recording: a `Recording`, or an array [channels, samples] with
    `sample_rate`. See the module docstring for the method and what is excluded.

    nfft: block length (frequency resolution fs / nfft). smooth: odd number of bins of extra
    smoothing of the cross-spectral matrix. transient_threshold: robust standard deviations
    of block power above which a block is flagged (None: keep everything). tone_threshold_db:
    how far a bin must stand out of the median-smoothed spectrum to be a line (None: the
    larger of 3 dB and six standard deviations of the spectrum estimate; inf: keep lines).
    tail: "auto", "gaussian", "k" or "sas"."""
    if not isinstance(recording, Recording):
        x = np.asarray(recording)
        if sample_rate is None:
            raise ValueError("fit: give sample_rate with an array")
        recording = Recording(segments=[np.atleast_2d(x)], sample_rate=float(sample_rate), positions=positions)
    fs = recording.sample_rate
    is_complex = any(np.iscomplexobj(s) for s in recording.segments)
    segs = [np.asarray(s, dtype=np.complex128 if is_complex else np.float64) for s in recording.segments]
    segs = [s - s.mean(axis=1, keepdims=True) for s in segs if s.shape[1] >= nfft]
    if not segs:
        raise ValueError(f"fit: no segment is as long as one block ({nfft} samples)")
    c = segs[0].shape[0]
    nfft = int(nfft)
    hop = nfft // 2
    window = np.hanning(nfft + 1)[:nfft]
    frames = _frames(segs, nfft, hop)
    seg_of = np.array([f[0] for f in frames])
    k_all = len(frames)

    # -- transients -----------------------------------------------------------------------------
    power = np.array([np.mean(np.abs(segs[i][:, s:s + nfft]) ** 2, axis=1) for i, s in frames])        # [K, C]
    med = np.maximum(np.median(power, axis=0), 1e-300)
    rel = power / med
    sig = np.maximum(1.4826 * np.median(np.abs(rel - 1.0), axis=0), 1e-6)
    score = ((rel - 1.0) / sig).max(axis=1)
    score_all = (rel.mean(axis=1) - 1.0) / max(1.4826 * np.median(np.abs(rel.mean(axis=1) - np.median(rel.mean(axis=1)))), 1e-6)
    if transient_threshold is None:
        flag = np.zeros(k_all, dtype=bool)
    else:
        flag = (score > transient_threshold) | (score_all > transient_threshold)
    exclude, events = _classify_events(flag, seg_of, int(event_blocks))
    use = ~exclude
    k_used = int(use.sum())
    n_d = effective_averages(k_used, window, hop) * max(int(smooth), 1)
    if n_d < 2 * c:
        raise ValueError(f"fit: {k_used} usable blocks ({n_d:.0f} independent) for {c} channels: the cross-spectral "
                         f"matrix needs at least {2 * c}. Record longer or use a shorter nfft.")

    # -- Welch cross-spectral matrix ---------------------------------------------------------------
    s2 = np.zeros((nfft, c, c), dtype=np.complex128)
    norm = fs * np.sum(window**2)
    for (i, s), ok in zip(frames, use):
        if ok:
            xf = np.fft.fft(segs[i][:, s:s + nfft] * window, axis=1)
            s2 += np.einsum("af,bf->fab", xf, np.conj(xf))
    s2 /= k_used * norm
    s2[0] = 0.0 if not is_complex else s2[0]                               # the mean was removed

    # -- tonals -----------------------------------------------------------------------------------
    diag = np.real(np.einsum("fcc->fc", s2))
    level = (diag / np.maximum(np.median(diag, axis=0), 1e-300)).mean(axis=1)
    order = np.argsort(np.fft.fftfreq(nfft))
    lev = level[order]
    half = max(8, nfft // 64)
    pad = np.pad(lev, half, mode="edge")
    background = np.array([np.median(pad[i:i + 2 * half + 1]) for i in range(nfft)])
    thr_db = max(3.0, 10.0 * np.log10(1.0 + 6.0 / np.sqrt(max(n_d * c**0.5, 1.0)))) if tone_threshold_db is None else float(tone_threshold_db)
    hot = lev > background * 10.0 ** (thr_db / 10.0)
    hot[order == 0] = False
    tones = []
    line_bins = np.zeros(nfft, dtype=bool)                                  # fft order
    if hot.any():
        grown = hot.copy()
        for shift in (-2, -1, 1, 2):                                        # the Hann main lobe and first side lobes
            grown |= np.roll(hot, shift)
        line_bins[order] = grown
        idx_sorted = np.where(grown)[0]
        clean = np.where(~grown)[0]
        s_sorted = s2[order]
        fgrid = np.sort(np.fft.fftfreq(nfft, 1.0 / fs))
        runs = np.split(idx_sorted, np.where(np.diff(idx_sorted) > 1)[0] + 1)
        filled = s_sorted.copy()
        for run in runs:
            lo = clean[clean < run[0]]
            hi = clean[clean > run[-1]]
            a = s_sorted[lo[-1]] if len(lo) else s_sorted[hi[0]]
            b = s_sorted[hi[0]] if len(hi) else a
            for m, r in enumerate(run):
                wgt = (m + 1) / (len(run) + 1)
                filled[r] = (1.0 - wgt) * a + wgt * b
            # Power matrix of the line (a real line has half its power at -f).
            excess = (s_sorted[run] - filled[run]).sum(axis=0) * (fs / nfft) * (1.0 if is_complex else 2.0)
            w, v = np.linalg.eigh(0.5 * (excess + excess.conj().T))
            peak = run[np.argmax(lev[run])]
            if fgrid[peak] > 0 or is_complex:
                tones.append({"frequency": float(fgrid[peak] + recording.center_frequency),
                              "amplitude": v[:, -1] * np.sqrt(max(w[-1], 0.0)),
                              "level_above_background_db": float(10.0 * np.log10(lev[peak] / background[peak]))})
        s2 = np.empty_like(s2)
        s2[order] = filled
    if smooth and int(smooth) > 1:
        m = int(smooth) | 1
        kern = np.ones(m) / m
        ss = s2[order]
        ss = np.apply_along_axis(lambda v: np.convolve(np.pad(v, m // 2, mode="edge"), kern, mode="valid"), 0, ss)
        s2 = np.empty_like(ss)
        s2[order] = ss

    model = NoiseModel(csd=s2, sample_rate=fs, center_frequency=recording.center_frequency, is_complex=is_complex,
                       tones=tones, positions=None if recording.positions is None else np.asarray(recording.positions, dtype=float))

    # -- tail: whiten what was kept and look at the innovations ----------------------------------
    g = _factor(s2)
    diag = np.real(np.einsum("fcc->fc", s2))
    # Where there is something to whiten; the lines are notched out (they are not in the model).
    band = (diag.mean(axis=1) > 1e-4 * diag.mean(axis=1).max()) & ~line_bins
    winv = np.zeros_like(g)
    winv[band] = np.linalg.inv(g[band] + 1e-12 * np.eye(c)) / np.sqrt(fs)
    wt = np.moveaxis(np.roll(np.fft.ifft(winv, axis=0), nfft // 2, axis=0) * np.hanning(nfft + 1)[:nfft, None, None], 0, 2)
    pieces, budget = [], int(tail_samples)
    # Whole stretches between excluded blocks, so that the texture's memory can be seen.
    for i, seg in enumerate(segs):
        bad = np.zeros(seg.shape[1], dtype=bool)
        for (si, s), ex in zip(frames, exclude):
            if si == i and ex:
                bad[s:s + nfft] = True
        edges = np.flatnonzero(np.diff(np.concatenate([[True], bad, [True]]).astype(int)))
        for a, b in zip(edges[::2], edges[1::2]):
            if b - a < 4 * nfft or budget <= 0:
                continue
            b = min(b, a + budget)
            e = convolve_matrix(wt, seg[:, a:b])[:, nfft:b - a]            # drop the filter's ends
            budget -= e.shape[1]
            pieces.append(e)
    tail_report = {"family": "gaussian"}
    family, par, scale, t_time, common = "gaussian", 0.0, 1.0, 0.0, True
    if pieces and tail != "gaussian":
        e_all = np.concatenate(pieces, axis=1)
        mag = np.abs(e_all) / (np.sqrt(2.0) if False else 1.0)
        if is_complex:
            mag = np.abs(e_all.real) * np.sqrt(2.0)                         # one quadrature, unit variance when Gaussian
        mag = np.maximum(mag, 1e-300)
        fraction = float(band.mean())
        la = np.log(mag)
        v = float(la.var())
        chunks = np.array_split(la.ravel(), 20)
        v_se = float(np.std([ch.var() for ch in chunks]) / np.sqrt(len(chunks)))
        kurt = float(np.mean(mag**4) / np.mean(mag**2) ** 2 - 3.0)
        ratio = _tail_ratio(mag)
        nu = _k_shape_from_log_variance(v)
        alpha = _alpha_from_log_variance(v)
        probe = np.random.default_rng(12345)
        gp = probe.standard_normal(1_000_000)
        pred = {"gaussian": _tail_ratio(gp), "k": _tail_ratio(np.sqrt(texture("k", nu, gp.shape, probe)) * gp),
                "sas": _tail_ratio(np.sqrt(texture("sas", alpha, gp.shape, probe)) * gp)}
        excess = v - _VAR_LOG_NORMAL
        if tail == "auto":
            if excess < max(4.0 * v_se, 0.01):
                family = "gaussian"
            else:
                family = min(("k", "sas"), key=lambda fam: abs(np.log(pred[fam] / ratio)))
        else:
            family = tail
        par = {"gaussian": 0.0, "k": nu, "sas": alpha}[family]
        if family == "sas":
            # Scale from E log|e| = log(scale) + [mean for scale 1/sqrt 2 texture form].
            scale = float(np.exp(la.mean() - log_abs_moments("sas", alpha)[0]))
        if family != "gaussian":
            # Memory and commonality of the texture from l = log e^2: cov(l_t, l_t+k) = cov(log tau_t, log tau_t+k).
            l2 = 2.0 * la
            v_tau = max(float(l2.var()) - 4.0 * _VAR_LOG_NORMAL, 1e-9)
            lc = l2 - l2.mean(axis=1, keepdims=True)
            max_lag = min(4 * nfft, lc.shape[1] // 4)
            nf = 1 << int(np.ceil(np.log2(lc.shape[1] + max_lag)))
            spec = np.abs(np.fft.rfft(lc, nf, axis=1)) ** 2
            acov = np.fft.irfft(spec, nf, axis=1)[:, :max_lag + 1].mean(axis=0) / lc.shape[1]
            rho = acov[1:] / v_tau
            # Sum of the texture's correlation while it stays clearly positive: (L - 1) / 2 for a hold of L.
            noise = 3.0 * float(l2.var()) / np.sqrt(lc.size) / v_tau
            stop = np.argmax(rho < noise) if (rho < noise).any() else len(rho)
            hold = 1.0 + 2.0 * float(rho[:stop].sum())
            t_time = 0.0 if hold < 1.5 else hold / fs
            if c > 1:
                cc = (lc @ lc.T) / lc.shape[1]
                off = (cc.sum() - np.trace(cc)) / (c * (c - 1))
                common_frac = float(off / v_tau)
            else:
                common_frac = 1.0
            common = common_frac > 0.5
            tail_report.update({"texture_hold_samples": hold, "texture_common_fraction": common_frac})
        tail_report.update({"family": family, "log_variance": v, "log_variance_gaussian": _VAR_LOG_NORMAL, "log_variance_se": v_se,
                            "kurtosis_excess": kurt, "k_shape": nu, "alpha": alpha, "tail_ratio": ratio, "tail_ratio_predicted": pred,
                            "whitened_samples": int(e_all.size), "whitened_band_fraction": fraction})
    model.family, model.tail_parameter, model.innovation_scale = family, float(par), float(scale)
    model.texture_time, model.texture_common = float(t_time), bool(common)

    # -- report -----------------------------------------------------------------------------------
    block_s = nfft / fs
    excluded = []
    for e in events:
        i0, s0 = frames[e["first"]]
        i1, s1 = frames[e["last"]]
        excluded.append({"kind": e["kind"], "segment": int(i0), "start_s": s0 / fs, "end_s": (s1 + nfft) / fs,
                         **({"period_s": e["period_blocks"] * hop / fs} if "period_blocks" in e else {})})
    used_power = power[use]
    model.report = {
        "source": recording.source, "sample_rate": fs, "n_channels": c, "duration_s": recording.duration,
        "channels": recording.channels, "dropped_reference_channels": recording.dropped_channels, "notes": list(recording.notes),
        "nfft": nfft, "resolution_hz": fs / nfft, "block_s": block_s, "smooth": int(smooth),
        "blocks_total": k_all, "blocks_used": k_used, "independent_averages": n_d,
        "spectrum_std_db": 4.343 / np.sqrt(n_d), "coherence_floor": coherence_floor(n_d),
        "coherence_std_at_0.5": coherence_std(0.5, n_d),
        "record_length_for_coherence_std_s": {str(eps): _record_length(eps, nfft, fs, window, hop) for eps in (0.1, 0.05, 0.02)},
        "block_power_spread_db": float(10.0 * np.log10(np.quantile(used_power.mean(axis=1), 0.95) / np.quantile(used_power.mean(axis=1), 0.05))),
        "events": excluded, "excluded_fraction": float(exclude.mean()),
        "impulses_kept": int(sum(e["kind"] == "impulse" for e in events)),
        "tones": [{"frequency": t["frequency"], "level_above_background_db": t["level_above_background_db"],
                   "rms": float(np.sqrt(np.mean(np.abs(t["amplitude"]) ** 2)))} for t in tones],
        "tone_threshold_db": thr_db, "tail": tail_report,
        "spectrum": _spectrum_table(model), "coherence_length": _coherence_length(model, sound_speed),
    }
    if recording.positions is not None:
        p = np.asarray(recording.positions, dtype=float)
        transit = float(np.linalg.norm(p[:, None] - p[None], axis=2).max() / sound_speed)
        model.report["array_transit_s"] = transit
        model.report["coherence_bias_from_block"] = float((1.0 - min(transit / block_s, 1.0)) ** 2)
    return model


def _record_length(eps: float, nfft: int, fs: float, window: np.ndarray, hop: int) -> float:
    """Record length, s, for a coherence standard deviation `eps` where the coherence is small
    (std (1 - gamma^2) / sqrt(2 n_d) -> n_d = 1 / (2 eps^2))."""
    n_d = 1.0 / (2.0 * eps**2)
    rho = float(np.sum(window[:-hop] * window[hop:]) / np.sum(window**2))
    k = n_d * (1.0 + 2.0 * rho**2)
    return float((nfft + (k - 1.0) * hop) / fs)


def _spectrum_table(model: NoiseModel, rows: int = 12) -> list:
    f, p = model.psd()
    edges = np.linspace(0, len(f), rows + 1).astype(int)
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        if b > a:
            band = p[a:b].mean(axis=0)
            out.append({"frequency_hz": float(f[a:b].mean()), "level_db": float(10 * np.log10(max(band.mean(), 1e-300))),
                        "min_db": float(10 * np.log10(max(band.min(), 1e-300))), "max_db": float(10 * np.log10(max(band.max(), 1e-300)))})
    return out


def _coherence_length(model: NoiseModel, sound_speed: float, rows: int = 8) -> list:
    """Per band: the separation at which the mean coherence magnitude first falls below 0.5, in
    metres and wavelengths when positions are known, in channel steps otherwise. For isotropic
    noise, sin(kd)/(kd) = 0.5 at d = 0.302 wavelengths."""
    c = model.n_channels
    if c < 2:
        return []
    coh = np.abs(model.coherence())
    f = model.frequency
    if model.positions is not None:
        d = np.linalg.norm(model.positions[:, None] - model.positions[None], axis=2)
        unit = "m"
    else:
        d = np.abs(np.arange(c)[:, None] - np.arange(c)[None]).astype(float)
        unit = "channels"
    iu = np.triu_indices(c, 1)
    sep = np.round(d[iu], 6)
    seps = np.unique(sep)
    pos = np.where(f > 0)[0] if not model.is_complex else np.argsort(f)
    edges = np.linspace(0, len(pos), rows + 1).astype(int)
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        if b <= a:
            continue
        band = coh[pos[a:b]].mean(axis=0)[iu]
        curve = np.array([band[sep == s].mean() for s in seps])
        fm = float(np.abs(f[pos[a:b]] + model.center_frequency).mean())
        xs, ys = np.concatenate([[0.0], seps]), np.concatenate([[1.0], curve])
        below = np.where(ys < 0.5)[0]
        if len(below):
            j = below[0]
            length = float(xs[j - 1] + (ys[j - 1] - 0.5) / max(ys[j - 1] - ys[j], 1e-12) * (xs[j] - xs[j - 1]))
        else:
            length = float("inf")
        row = {"frequency_hz": fm, "length": length, "unit": unit, "coherence_at_first_spacing": float(curve[0])}
        if unit == "m" and np.isfinite(length) and fm > 0:
            row["wavelengths"] = length * fm / sound_speed
        out.append(row)
    return out


def format_report(model: NoiseModel) -> str:
    """The fit report as text."""
    r = model.report
    out = [f"Noise model fitted to {r.get('source') or 'an array'}: {r['n_channels']} channels, {r['duration_s']:.2f} s at {r['sample_rate']:.0f} Hz",
           f"  blocks of {r['nfft']} samples ({r['block_s'] * 1e3:.2f} ms, resolution {r['resolution_hz']:.2f} Hz): {r['blocks_used']} of {r['blocks_total']} used, "
           f"{r['independent_averages']:.0f} independent averages",
           f"  estimator: spectrum level +-{r['spectrum_std_db']:.2f} dB (1 sigma); coherence floor {r['coherence_floor']:.3f} where the truth is 0, "
           f"+-{r['coherence_std_at_0.5']:.3f} at 0.5",
           "  record length for a coherence standard deviation of " + ", ".join(f"{k}: {v:.1f} s" for k, v in r["record_length_for_coherence_std_s"].items()),
           f"  block power spread (5-95 %): {r['block_power_spread_db']:.2f} dB (stationarity is assumed)"]
    if r.get("dropped_reference_channels"):
        out.append(f"  reference inputs dropped: channels {r['dropped_reference_channels']}")
    if "array_transit_s" in r:
        out.append(f"  array transit time {r['array_transit_s'] * 1e3:.3f} ms: coherence of a directional source biased by x{r['coherence_bias_from_block']:.3f} at endfire")
    for n in r.get("notes", []):
        out.append(f"  NOTE: {n}")
    out.append("Spectrum level (sample units^2/Hz, mean over channels; min..max across channels)")
    for row in r["spectrum"]:
        out.append(f"  {row['frequency_hz']:10.1f} Hz  {row['level_db']:8.2f} dB  ({row['min_db']:.2f} .. {row['max_db']:.2f})")
    if r["coherence_length"]:
        out.append("Coherence length (separation where the mean |coherence| falls to 0.5)")
        for row in r["coherence_length"]:
            extra = f" = {row['wavelengths']:.2f} wavelengths" if "wavelengths" in row else ""
            out.append(f"  {row['frequency_hz']:10.1f} Hz  {row['length']:.3f} {row['unit']}{extra}   (|coherence| at the first spacing {row['coherence_at_first_spacing']:.3f})")
    t = r["tail"]
    if t.get("family", "gaussian") == "gaussian" and "log_variance" not in t:
        out.append("Amplitude statistics: Gaussian (not tested)")
    else:
        out.append(f"Amplitude statistics of the whitened recording ({t['whitened_samples']} samples): var(log|e|) = {t['log_variance']:.4f} +- {t['log_variance_se']:.4f} "
                   f"(Gaussian {t['log_variance_gaussian']:.4f}), kurtosis excess {t['kurtosis_excess']:.2f}")
        out.append(f"  tail ratio q99.95/q50 = {t['tail_ratio']:.2f}; predicted: Gaussian {t['tail_ratio_predicted']['gaussian']:.2f}, "
                   f"K (shape {t['k_shape']:.2f}) {t['tail_ratio_predicted']['k']:.2f}, alpha-stable (alpha {t['alpha']:.2f}) {t['tail_ratio_predicted']['sas']:.2f}")
        if t["family"] == "gaussian":
            out.append("  -> Gaussian")
        else:
            name = {"k": f"K-type, shape {model.tail_parameter:.2f}", "sas": f"symmetric alpha-stable, alpha {model.tail_parameter:.2f}, scale factor {model.innovation_scale:.3f}"}[t["family"]]
            out.append(f"  -> {name}; texture held {t.get('texture_hold_samples', 1.0):.1f} samples, {'common to' if model.texture_common else 'independent between'} "
                       f"channels (common fraction {t.get('texture_common_fraction', 1.0):.2f})")
    if r["tones"]:
        out.append(f"Lines taken out of the spectrum (threshold {r['tone_threshold_db']:.1f} dB above the median-smoothed spectrum)")
        for tone in r["tones"]:
            out.append(f"  {tone['frequency']:10.1f} Hz  +{tone['level_above_background_db']:.1f} dB, rms {tone['rms']:.4g}")
    else:
        out.append(f"No lines above {r['tone_threshold_db']:.1f} dB")
    ex = [e for e in r["events"] if e["kind"] != "impulse"]
    out.append(f"Excluded: {len(ex)} event(s), {100.0 * r['excluded_fraction']:.1f} % of blocks; {r['impulses_kept']} isolated impulsive block(s) kept as noise")
    bursts = [e for e in ex if e["kind"] == "periodic burst"]
    if bursts:
        out.append(f"  periodic burst  x{len(bursts)}, about every {bursts[0]['period_s']:.3f} s, from {bursts[0]['start_s']:.3f} s "
                   f"(segment {bursts[0]['segment']}) to {bursts[-1]['end_s']:.3f} s (segment {bursts[-1]['segment']}): pings or sync reference bursts?")
    rest = [e for e in ex if e["kind"] != "periodic burst"]
    for e in rest[:30]:
        out.append(f"  {e['kind']:15s} segment {e['segment']}  {e['start_s']:.4f} - {e['end_s']:.4f} s")
    if len(rest) > 30:
        out.append(f"  ... and {len(rest) - 30} more")
    return "\n".join(out)


# =================================================================================================
# For the simulator
# =================================================================================================

_CACHE: dict = {}


def scenario_noise(cfg: dict, n_samples: int, sample_rate: float, rng: np.random.Generator, *, n_channels: int,
                   center_frequency: float = 0.0, hydrophone_channels=None) -> np.ndarray:
    """Noise [n_channels, n_samples] for a scenario's `receiver.ambient_noise: {model: fitted, file: ...,
    gain_db: 0, channels: [...], include_tones: false, allow_uncovered: false}`.

    `hydrophone_channels`: the record's channels that are hydrophones (the rest, e.g. reference
    inputs, get no noise). The model's channel count must equal their number."""
    path = str(cfg["file"])
    key = (path, os.path.getmtime(path))
    if key not in _CACHE:
        _CACHE.clear()
        _CACHE[key] = NoiseModel.load(path)
    model = _CACHE[key]
    rows = list(range(n_channels)) if hydrophone_channels is None else [int(i) for i in hydrophone_channels]
    x = model.generate(n_samples, rng, sample_rate=sample_rate, center_frequency=center_frequency if center_frequency > 0 else (None if not model.is_complex else 0.0),
                       n_channels=len(rows), channels=cfg.get("channels"), include_tones=bool(cfg.get("include_tones", False)),
                       allow_uncovered=bool(cfg.get("allow_uncovered", False)))
    x = x * 10.0 ** (float(cfg.get("gain_db", 0.0)) / 20.0)
    out = np.zeros((n_channels, n_samples), dtype=x.dtype)
    out[rows] = x
    return out
