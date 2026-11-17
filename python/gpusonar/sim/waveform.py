"""Sampled (arbitrary) transmit waveforms: building them, carrying them in a record, and
synthesising their echoes.

A sampled waveform is held as a complex envelope about a carrier, on a uniform grid:

    transmitted pressure  q(u) = amplitude * Re{ b(u) exp(2 pi i F u) },   0 <= u <= duration

with b given by its samples b[m] = b(m / sample_rate) (band-limited between them).

Config (`transmit.waveform` of a sonar):

    type: sampled
    source: lfm | hfm | stepped_fm | barker | mseq | code | samples | file
    f_start, f_stop: the occupied band, Hz (carrier F = their mean unless `carrier` is given)
    duration: s (built-in sources; a file's or a list's duration is its length)
    window_alpha: Tukey taper (lfm, hfm, stepped_fm)
    steps: number of frequency steps (stepped_fm)
    code: Barker length (barker), or a list of chip values (code)
    degree, taps: shift-register length and feedback taps (mseq)
    rolloff: fraction of the band over which a phase code's spectrum is rolled off (0.25)
    samples: [...]          inline samples: real passband, or [re, im] pairs of the envelope
    file: x.wav | x.npy | x.h5, dataset: name in the HDF5 file
    sample_rate: Hz of `samples` / of an .npy or HDF5 file (a WAV file carries its own)
    carrier: Hz removed from complex samples (default: the band centre)
    normalise: peak (default: unit peak envelope) | none
    oversample: envelope sample rate / bandwidth for built-in sources (8)
    interp_taps: taps M of the windowed-sinc interpolator (16)
    interp_window: kaiser (default) | hann (the window of TR 0702 eq. 17); interp_beta: Kaiser shape (12)
    gain_knots: frequency intervals for frequency-dependent loss and beam patterns (8)
    table_factor: the synthesiser reads the envelope from a table this many times finer (8); 0: none
    replica_sample_rate: rate of the real replica stored in the record (4 x top of band)

In the record the waveform is `WaveformType.Arbitrary` with the existing fields
`replica` (real passband samples of q(u) / amplitude) and `replica_sample_rate`;
`f_start`/`f_stop` give the band, their mean is the carrier. Nothing else is needed by
processing: `SampledWaveform.from_record` recovers the envelope (analytic signal, mixed
down), and pulse compression builds its matched filter from it.

Echo synthesis (`synthesize_into`) follows the analytic kernel (libs/sim synth_kernel.hpp)
step by step, with the closed-form pulse replaced by interpolation of the envelope:

- exact arrival times of the start, middle and end of the pulse for every (path, channel)
  from the moving transmitter, scatterer and receiver; emission time u(t) is the quadratic
  through them. The echo is b(u(t)) exp(2 pi i F u(t)): a delayed, time-scaled copy
  (Doppler as time scaling of envelope and carrier, with range curvature), which is what
  SST's variable-delay resampler does (TR 0702 Sec. 4.5.1, eq. 14), with the delay solved
  exactly per sample instead of interpolated linearly from a slow track;
- the envelope is interpolated with the windowed sinc of TR 0702 eqs. 15-16 (Kaiser window by
  default, the Hann window of eq. 17 on request). For speed this is done once, onto a table
  `table_factor` times finer than the envelope's samples, which is then read with a
  four-point cubic at each output sample;
- loss that depends on frequency (absorption, both beam patterns) is applied in frequency:
  the envelope is split by triangular weights into bands about `gain_knots + 1` frequencies
  and each band takes its own gain, i.e. the gain is piecewise linear across the band.

This engine is NumPy only. The C++/CUDA synthesis kernel takes analytic pulses only.

Reference: R. P. Goddard, "The Sonar Simulation Toolset, Release 4.6: Science, Mathematics,
and Algorithms", APL-UW TR 0702 (2008), Sec. 4.1 (eqs. 3-6), Sec. 4.5.1 (eqs. 14-19).
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from .. import _gpusonar as core

DEFAULT_TAPS = 16
# Kaiser shape for the default interpolator: with 16 taps and a signal filling up to half the
# envelope's Nyquist band, the measured worst error of the response (passband and images) is
# -117 dB, against -62 dB for the Hann window of TR 0702 eq. 17 (tests/python/test_sampled_waveform.py).
DEFAULT_BETA = 12.0
DEFAULT_OVERSAMPLE = 8.0
DEFAULT_GAIN_KNOTS = 8
DEFAULT_TABLE_FACTOR = 8

# Barker codes. Checked by test (every aperiodic autocorrelation sidelobe has magnitude <= 1).
BARKER = {
    2: (1, -1), 3: (1, 1, -1), 4: (1, 1, -1, 1), 5: (1, 1, 1, -1, 1),
    7: (1, 1, 1, -1, -1, 1, -1), 11: (1, 1, 1, -1, -1, -1, 1, -1, -1, 1, -1),
    13: (1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1),
}


# --- interpolation (TR 0702 Sec. 4.5.1) ---------------------------------------------------

def hann_sinc_weight(z, taps: int) -> np.ndarray:
    """Interpolation weight d(z) of TR 0702 eq. 17: the ideal sinc weight times a Hann
    window `taps` samples long. z is the distance to the sample, in samples."""
    z = np.asarray(z, dtype=float)
    return np.where(np.abs(z) <= 0.5 * taps, np.cos(np.pi * z / taps) ** 2 * np.sinc(z), 0.0)


_KAISER_TABLES: dict = {}
_KAISER_STEPS = 1024          # table entries per sample


def kaiser_sinc_weight(z, taps: int, beta: float) -> np.ndarray:
    """As hann_sinc_weight with a Kaiser window of shape `beta` in place of the Hann window.
    The window is read from a table (1024 entries per sample, linear between them: the
    tabulation of TR 0702 eq. 18 with interpolation instead of the nearest entry)."""
    z = np.asarray(z, dtype=float)
    key = (int(taps), float(beta))
    if key not in _KAISER_TABLES:
        grid = np.linspace(0.0, 0.5 * taps, taps * _KAISER_STEPS // 2 + 1)
        _KAISER_TABLES[key] = np.i0(beta * np.sqrt(np.clip(1.0 - (2.0 * grid / taps) ** 2, 0.0, None))) / np.i0(beta)
    table = _KAISER_TABLES[key]
    x = np.minimum(np.abs(z), 0.5 * taps) * _KAISER_STEPS
    i = np.minimum(x.astype(np.int64), len(table) - 2)
    x = x - i
    return np.where(np.abs(z) <= 0.5 * taps, ((1.0 - x) * table[i] + x * table[i + 1]) * np.sinc(z), 0.0)


def interpolation_weight(z, taps: int, beta: float | None = None) -> np.ndarray:
    return hann_sinc_weight(z, taps) if beta is None else kaiser_sinc_weight(z, taps, beta)


def interpolate(x: np.ndarray, position, taps: int = DEFAULT_TAPS, beta: float | None = None) -> np.ndarray:
    """The band-limited signal through uniform samples `x`, at fractional sample indices
    `position` (TR 0702 eqs. 15-16). Samples outside the array count as zero.

    x is [N] with `position` of any shape, or [P, N] with `position` [P, L] (row by row).
    The window on the sinc is Hann (TR 0702 eq. 17) when `beta` is None, else Kaiser.
    """
    if taps < 2 or taps % 2:
        raise ValueError("interpolate: taps must be an even number, at least 2")
    x = np.asarray(x)
    position = np.asarray(position, dtype=float)
    n = x.shape[-1]
    n0 = np.floor(position)
    frac = position - n0
    n0 = n0.astype(np.int64)
    out = np.zeros(position.shape, dtype=np.result_type(x.dtype, np.float64))
    if x.ndim == 2:
        flat = x.reshape(-1)
        row = (np.arange(x.shape[0], dtype=np.int64) * n)[:, None]
    for k in range(-taps // 2, taps // 2):
        idx = n0 - k
        ok = (idx >= 0) & (idx < n)
        w = np.where(ok, interpolation_weight(frac + k, taps, beta), 0.0)
        idx = np.clip(idx, 0, n - 1)
        out += w * (flat[row + idx] if x.ndim == 2 else x[idx])
    return out


def interpolate_cubic(x: np.ndarray, position) -> np.ndarray:
    """Four-point Lagrange (cubic) interpolation of uniform samples at fractional indices;
    samples outside the array count as zero. For signals sampled far above their bandwidth
    (the synthesiser's fine table). x is [N], or [P, N] with `position` [P, L]."""
    x = np.asarray(x)
    position = np.asarray(position, dtype=float)
    n = x.shape[-1]
    i0 = np.floor(position)
    f = position - i0
    i0 = i0.astype(np.int64)
    weights = (-f * (f - 1.0) * (f - 2.0) / 6.0, (f + 1.0) * (f - 1.0) * (f - 2.0) / 2.0,
               -(f + 1.0) * f * (f - 2.0) / 2.0, (f + 1.0) * f * (f - 1.0) / 6.0)
    out = np.zeros(position.shape, dtype=np.result_type(x.dtype, np.float64))
    if x.ndim == 2:
        flat = x.reshape(-1)
        row = (np.arange(x.shape[0], dtype=np.int64) * n)[:, None]
    for k, w in zip((-1, 0, 1, 2), weights):
        idx = i0 + k
        ok = (idx >= 0) & (idx < n)
        idx = np.clip(idx, 0, n - 1)
        out += np.where(ok, w, 0.0) * (flat[row + idx] if x.ndim == 2 else x[idx])
    return out


def _cubic_inside(x: np.ndarray, position: np.ndarray) -> np.ndarray:
    """interpolate_cubic for positions known to lie in [1, N - 3]: no bounds handling."""
    i0 = position.astype(np.int64)
    f = position - i0
    if x.ndim == 2:
        i0 += (np.arange(x.shape[0], dtype=np.int64) * x.shape[1])[:, None]
        x = x.reshape(-1)
    fm, fp, f2 = f - 1.0, f + 1.0, f - 2.0
    out = np.take(x, i0 - 1) * (-f * fm * f2 / 6.0)
    out += np.take(x, i0) * (fp * fm * f2 / 2.0)
    out += np.take(x, i0 + 1) * (-fp * f * f2 / 2.0)
    out += np.take(x, i0 + 2) * (fp * f * fm / 6.0)
    return out


def interpolator_response(taps: int, frequency, beta: float | None = None) -> np.ndarray:
    """Frequency response of the interpolation weights seen as a continuous filter d(z):
    D(f) = integral d(z) exp(-2 pi i f z) dz, f in cycles per input sample. Ideal: 1 for
    |f| < 0.5 (the Nyquist band), 0 outside (where the copies of the spectrum sit)."""
    f = np.atleast_1d(np.asarray(frequency, dtype=float))
    z = np.linspace(-0.5 * taps, 0.5 * taps, 128 * taps + 1)
    d = interpolation_weight(z, taps, beta)
    dz = z[1] - z[0]
    w = np.full(z.shape, dz)
    w[0] = w[-1] = 0.5 * dz
    return (d * w) @ np.cos(2.0 * np.pi * np.outer(z, f))


def var_delay(x: np.ndarray, sample_rate: float, first_time: float, out_times, delay_times,
              delay_values, taps: int = DEFAULT_TAPS, carrier: float = 0.0,
              beta: float | None = None) -> np.ndarray:
    """y(t) = x(t - T(t)) (TR 0702 eq. 14) for uniform samples x starting at `first_time`.

    The delay T is given at `delay_times` and interpolated linearly to the output times,
    as SST's VarDelay does; the signal is interpolated with the Hann-windowed sinc. For a
    complex envelope about `carrier` the carrier's phase turns with the delay.
    """
    out_times = np.asarray(out_times, dtype=float)
    delay = np.interp(out_times, delay_times, delay_values)
    y = interpolate(x, (out_times - delay - first_time) * sample_rate, taps, beta)
    if carrier:
        y = y * np.exp(-2j * np.pi * carrier * delay)
    return y


# --- the waveform -------------------------------------------------------------------------

def _tukey(x: np.ndarray, alpha: float) -> np.ndarray:
    if alpha <= 0.0:
        return np.ones_like(x)
    edge = 0.5 * alpha
    return np.where(x < edge, 0.5 * (1 - np.cos(np.pi * x / edge)),
                    np.where(x > 1 - edge, 0.5 * (1 - np.cos(np.pi * (1 - x) / edge)), 1.0))


def _padded_fft(x: np.ndarray, pad: int):
    n_fft = int(2 ** np.ceil(np.log2(len(x) + 2 * pad)))
    buf = np.zeros(n_fft, dtype=complex)
    buf[pad:pad + len(x)] = x
    return np.fft.fft(buf), n_fft


def analytic_envelope(real_samples: np.ndarray, sample_rate: float, carrier: float) -> np.ndarray:
    """Complex envelope about `carrier` of a real passband burst: its analytic signal
    (negative frequencies removed) mixed down. TR 0702 eq. 5 without the sqrt(2), so that
    real = Re{envelope exp(2 pi i F t)}."""
    x = np.asarray(real_samples, dtype=float)
    pad = len(x) // 8 + 64
    spectrum, n_fft = _padded_fft(x, pad)
    h = np.zeros(n_fft)
    h[0] = 1.0
    h[1:n_fft // 2] = 2.0
    h[n_fft // 2] = 1.0
    a = np.fft.ifft(spectrum * h)[pad:pad + len(x)]
    return a * np.exp(-2j * np.pi * carrier * np.arange(len(x)) / sample_rate)


def upsample(x: np.ndarray, factor: int) -> np.ndarray:
    """Band-limited interpolation of complex samples to `factor` times the rate (zero-padded
    spectrum). Output sample factor * m equals input sample m."""
    if factor <= 1:
        return np.asarray(x)
    pad = len(x) // 8 + 64
    spectrum, n_fft = _padded_fft(np.asarray(x, dtype=complex), pad)
    big = np.zeros(n_fft * factor, dtype=complex)
    half = n_fft // 2
    big[:half] = spectrum[:half]
    big[-half:] = spectrum[half:]
    y = np.fft.ifft(big) * factor
    return y[pad * factor:(pad + len(x) - 1) * factor + 1]


@dataclass
class SampledWaveform:
    envelope: np.ndarray     # complex [N]; sample m is at emission time m / sample_rate
    sample_rate: float       # Hz, of the envelope
    carrier: float           # Hz
    f_start: float           # occupied band, Hz
    f_stop: float
    duration: float          # s; the pulse occupies emission times [0, duration]
    taps: int = DEFAULT_TAPS
    beta: float | None = DEFAULT_BETA     # Kaiser shape of the interpolation window; None: Hann
    gain_knots: int = DEFAULT_GAIN_KNOTS
    replica_sample_rate: float = 0.0      # of the real replica in the record; 0: 4 x top of band
    table_factor: int = DEFAULT_TABLE_FACTOR   # synthesis: fine table at this many times the envelope rate; 0: none
    _bands: np.ndarray | None = field(default=None, repr=False, compare=False)
    _fine: np.ndarray | None = field(default=None, repr=False, compare=False)
    _fine_total: np.ndarray | None = field(default=None, repr=False, compare=False)
    _replica: np.ndarray | None = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        self.envelope = np.ascontiguousarray(self.envelope, dtype=complex)
        if self.envelope.ndim != 1 or len(self.envelope) < 2:
            raise ValueError("sampled waveform: need a one-dimensional series of at least 2 samples")
        if not (self.sample_rate > 0.0 and self.duration > 0.0 and self.carrier > 0.0):
            raise ValueError("sampled waveform: sample_rate, duration and carrier must be positive")
        if not np.all(np.isfinite(self.envelope)) or not np.any(self.envelope):
            raise ValueError("sampled waveform: samples must be finite and not all zero")

    @property
    def bandwidth(self) -> float:
        return abs(self.f_stop - self.f_start)

    @property
    def band_occupancy(self) -> float:
        """Fraction of the envelope's Nyquist band the signal's band covers (alpha of TR 0702 eq. 19)."""
        return self.bandwidth / self.sample_rate

    def evaluate(self, u) -> np.ndarray:
        """Envelope b(u) at emission times u (s); zero outside [0, duration]."""
        u = np.asarray(u, dtype=float)
        inside = (u >= 0.0) & (u <= self.duration)
        return np.where(inside, interpolate(self.envelope, u * self.sample_rate, self.taps, self.beta), 0.0)

    def baseband(self, sample_rate: float, center_frequency: float, doppler_scale: float = 1.0) -> np.ndarray:
        """The pulse (or its echo time-scaled by `doppler_scale`) as complex baseband samples
        about `center_frequency` at `sample_rate`, unit amplitude: sample k is at k / sample_rate."""
        n = int(np.floor(self.duration / doppler_scale * sample_rate)) + 1
        t = np.arange(n) / sample_rate
        u = doppler_scale * t
        cyc = self.carrier * u - center_frequency * t
        return self.evaluate(u) * np.exp(2j * np.pi * (cyc - np.floor(cyc)))

    def passband(self, sample_rate: float) -> np.ndarray:
        """Real samples of Re{b(u) exp(2 pi i F u)} at u = m / sample_rate."""
        n = int(np.floor(self.duration * sample_rate)) + 1
        u = np.arange(n) / sample_rate
        cyc = self.carrier * u
        return np.real(self.evaluate(u) * np.exp(2j * np.pi * (cyc - np.floor(cyc))))

    def record_rate(self) -> float:
        return self.replica_sample_rate or 4.0 * max(self.f_start, self.f_stop)

    def to_record(self, waveform: "core.Waveform") -> None:
        """Fill a record waveform's existing fields for an arbitrary pulse (amplitude and
        source level are left to the caller)."""
        rate = self.record_rate()
        if rate <= 2.0 * max(self.f_start, self.f_stop):
            raise ValueError("sampled waveform: replica_sample_rate must exceed twice the top of the band")
        waveform.type = core.WaveformType.Arbitrary
        waveform.f_start = float(self.f_start)
        waveform.f_stop = float(self.f_stop)
        waveform.duration = float(self.duration)
        waveform.window_alpha = 0.0
        waveform.replica_sample_rate = float(rate)
        if self._replica is None:
            self._replica = self.passband(rate).astype(np.float32)
        waveform.replica = self._replica

    @classmethod
    def from_record(cls, waveform: "core.Waveform", taps: int = DEFAULT_TAPS, beta: float | None = DEFAULT_BETA,
                    gain_knots: int = DEFAULT_GAIN_KNOTS) -> "SampledWaveform":
        """The waveform a record carries (type Arbitrary): the envelope of its real replica
        about the band centre."""
        if waveform.type != core.WaveformType.Arbitrary:
            raise ValueError("the record's waveform is not a sampled one")
        replica = np.asarray(waveform.replica, dtype=float)
        rate = float(waveform.replica_sample_rate)
        if len(replica) < 2 or not rate > 0.0:
            raise ValueError("the record's arbitrary waveform has no replica")
        carrier = 0.5 * (waveform.f_start + waveform.f_stop)
        return cls(analytic_envelope(replica, rate, carrier), rate, carrier, waveform.f_start,
                   waveform.f_stop, waveform.duration, taps=taps, beta=beta, gain_knots=gain_knots,
                   replica_sample_rate=rate)

    # Frequency-dependent gain: bands about K + 1 knot frequencies across [f_lo, f_hi].
    def knot_frequencies(self) -> np.ndarray:
        lo, hi = min(self.f_start, self.f_stop), max(self.f_start, self.f_stop)
        if self.gain_knots < 1 or hi <= lo:
            return np.array([0.5 * (lo + hi)])
        return np.linspace(lo, hi, self.gain_knots + 1)

    def band_envelopes(self) -> np.ndarray:
        """[K + 1, N]: the envelope split by triangular weights in frequency about the knot
        frequencies. The rows sum to the envelope; a gain g_i on row i is a gain piecewise
        linear in frequency through (f_i, g_i), constant outside the band."""
        if self._bands is None:
            knots = self.knot_frequencies() - self.carrier
            if len(knots) == 1:
                self._bands = self.envelope[None, :]
            else:
                pad = len(self.envelope) // 8 + 64
                spectrum, n_fft = _padded_fft(self.envelope, pad)
                f = np.fft.fftfreq(n_fft, 1.0 / self.sample_rate)
                x = np.clip((f - knots[0]) / (knots[1] - knots[0]), 0.0, len(knots) - 1.0)
                rows = [np.fft.ifft(spectrum * np.clip(1.0 - np.abs(x - i), 0.0, 1.0))[pad:pad + len(self.envelope)]
                        for i in range(len(knots))]
                self._bands = np.array(rows)
        return self._bands

    _FINE_PAD = 2

    def fine_bands(self) -> np.ndarray:
        """band_envelopes() resampled by the windowed-sinc interpolator onto a grid
        `table_factor` times finer, from _FINE_PAD fine samples before emission time 0 to
        _FINE_PAD after the last sample. The synthesiser reads this table with a cubic."""
        if self._fine is None:
            # Long enough for every emission time in [0, duration] with the cubic's four points.
            n, f, pad = len(self.envelope), self.table_factor, self._FINE_PAD
            pos = (np.arange(n * f + 2 * pad + 2) - pad) / f
            self._fine = np.array([interpolate(row, pos, self.taps, self.beta) for row in self.band_envelopes()])
        return self._fine

    def synthesis_values(self, gains: np.ndarray | None, u: np.ndarray) -> np.ndarray:
        """Envelope at emission times u [P, L]; with `gains` [P, K + 1] each band takes its
        gain. Two stages when table_factor > 0: the fine table, then a cubic."""
        if self.table_factor > 0:
            table = self.fine_bands()
            # Emission times outside the pulse are clamped here and zeroed by the caller.
            pos = np.clip(u * (self.sample_rate * self.table_factor) + self._FINE_PAD, 1.0, table.shape[1] - 3.0)
            if gains is None:
                if self._fine_total is None:
                    self._fine_total = table.sum(axis=0) if len(table) > 1 else table[0]
                return _cubic_inside(self._fine_total, pos)
            return _cubic_inside(gains @ table, pos)
        pos = u * self.sample_rate
        if gains is None:
            return interpolate(self.envelope, pos, self.taps, self.beta)
        return interpolate(gains @ self.band_envelopes(), pos, self.taps, self.beta)


# --- builders -----------------------------------------------------------------------------

def mseq(degree: int, taps=None) -> np.ndarray:
    """A maximal-length sequence of +-1 chips, length 2^degree - 1, from a Fibonacci shift
    register. `taps` are the register stages (1..degree) fed back; when not given, the first
    tap set (fewest taps, then lowest) whose period is 2^degree - 1 is found by trial."""
    if not 2 <= degree <= 16:
        raise ValueError("mseq: degree must be 2..16")
    length = 2 ** degree - 1

    def run(tap_set):
        state = [1] * degree
        out = np.empty(length, dtype=int)
        start = tuple(state)
        for i in range(length):
            out[i] = state[-1]
            bit = 0
            for t in tap_set:
                bit ^= state[t - 1]
            state = [bit] + state[:-1]
            if tuple(state) == start and i < length - 1:
                return None                       # shorter period: not maximal
        return out if tuple(state) == start else None

    if taps is not None:
        bits = run([int(t) for t in taps])
        if bits is None:
            raise ValueError(f"mseq: taps {list(taps)} do not give a maximal-length sequence of degree {degree}")
    else:
        from itertools import combinations
        bits = None
        for extra in (1, 3, 5):
            for combo in combinations(range(1, degree), extra):
                bits = run(list(combo) + [degree])
                if bits is not None:
                    break
            if bits is not None:
                break
        if bits is None:
            raise ValueError(f"mseq: no feedback taps found for degree {degree}")
    return 1 - 2 * bits


def _chip_envelope(chips: np.ndarray, duration: float, rate: float, bandwidth: float, rolloff: float) -> np.ndarray:
    """Rectangular chips, band-limited to +-bandwidth/2 with a raised-cosine edge covering
    the outer `rolloff` of the band, sampled at `rate` over [0, duration]."""
    fine = 16
    n = int(np.floor(duration * rate)) + 1
    u = np.arange((n - 1) * fine + 1) / (rate * fine)
    idx = np.minimum((u / duration * len(chips)).astype(int), len(chips) - 1)
    x = np.asarray(chips, dtype=complex)[idx]
    pad = len(x) // 4 + 64 * fine
    spectrum, n_fft = _padded_fft(x, pad)
    f = np.abs(np.fft.fftfreq(n_fft, 1.0 / (rate * fine)))
    edge = 0.5 * bandwidth
    flat = edge * (1.0 - rolloff)
    h = np.where(f <= flat, 1.0, np.where(f >= edge, 0.0,
                 0.5 * (1.0 + np.cos(np.pi * (f - flat) / max(edge - flat, 1e-300)))))
    return np.fft.ifft(spectrum * h)[pad:pad + len(x)][::fine]


def _read_file(path: str, dataset: str | None, rate: float | None):
    path = os.path.expanduser(str(path))
    if not os.path.isabs(path) and not os.path.exists(path):
        from ..config import config_dir
        candidate = config_dir() / path
        if candidate.exists():
            path = str(candidate)
    ext = os.path.splitext(path)[1].lower()
    if ext == ".wav":
        from scipy.io import wavfile
        file_rate, data = wavfile.read(path)
        data = np.asarray(data)
        if data.ndim == 2:
            data = data[:, 0]
        if np.issubdtype(data.dtype, np.integer):
            data = data.astype(float) / float(np.iinfo(data.dtype).max)
        return data.astype(float), float(file_rate)
    if ext == ".npy":
        data = np.load(path, allow_pickle=False)
    elif ext in (".h5", ".hdf5"):
        import h5py
        if not dataset:
            raise ValueError("sampled waveform: an HDF5 file needs `dataset`")
        with h5py.File(path, "r") as f:
            data = np.asarray(f[dataset])
            if rate is None and "sample_rate" in f[dataset].attrs:
                rate = float(f[dataset].attrs["sample_rate"])
    else:
        raise ValueError(f"sampled waveform: unknown file type {ext!r} (use .wav, .npy, .h5)")
    if data.dtype.fields and {"r", "i"} <= set(data.dtype.fields):
        data = data["r"] + 1j * data["i"]
    return data, rate


def build_waveform(cfg: dict) -> SampledWaveform:
    """A SampledWaveform from a sonar's `transmit.waveform` mapping (type: sampled)."""
    source = str(cfg.get("source", "")).lower()
    for key in ("f_start", "f_stop"):
        if cfg.get(key) is None:
            raise ValueError(f"sampled waveform: `{key}` is required (the occupied band)")
    f0, f1 = float(cfg["f_start"]), float(cfg["f_stop"])
    band = abs(f1 - f0)
    if not band > 0.0:
        raise ValueError("sampled waveform: f_start and f_stop must differ (they give the occupied band)")
    carrier = float(cfg["carrier"]) if cfg.get("carrier") is not None else 0.5 * (f0 + f1)
    oversample = float(cfg.get("oversample", DEFAULT_OVERSAMPLE))
    alpha = float(cfg.get("window_alpha", 0.0))
    window = str(cfg.get("interp_window", "kaiser")).lower()
    if window not in ("kaiser", "hann"):
        raise ValueError("sampled waveform: interp_window must be `kaiser` or `hann`")
    common = dict(taps=int(cfg.get("interp_taps", DEFAULT_TAPS)),
                  beta=float(cfg.get("interp_beta", DEFAULT_BETA)) if window == "kaiser" else None,
                  gain_knots=int(cfg.get("gain_knots", DEFAULT_GAIN_KNOTS)),
                  table_factor=int(cfg.get("table_factor", DEFAULT_TABLE_FACTOR)),
                  replica_sample_rate=float(cfg.get("replica_sample_rate", 0.0)))

    if source in ("samples", "file"):
        rate = float(cfg["sample_rate"]) if cfg.get("sample_rate") is not None else None
        if source == "samples":
            data = np.asarray(cfg["samples"], dtype=float)
            if data.ndim == 2 and data.shape[1] == 2:
                data = data[:, 0] + 1j * data[:, 1]
        else:
            data, rate = _read_file(cfg["file"], cfg.get("dataset"), rate)
        if rate is None:
            raise ValueError("sampled waveform: `sample_rate` of the samples is required")
        data = np.asarray(data)
        if data.ndim != 1:
            raise ValueError("sampled waveform: samples must be one series (real, or [re, im] pairs)")
        duration = (len(data) - 1) / rate
        if np.iscomplexobj(data):
            if rate < band:
                raise ValueError("sampled waveform: the envelope's sample_rate is below the stated bandwidth")
            envelope = data
        else:
            if rate <= 2.0 * max(f0, f1):
                raise ValueError("sampled waveform: real samples must be sampled above twice the top of the band")
            envelope = analytic_envelope(data, rate, carrier)
        # Enough margin for the interpolator: at least `oversample` samples per cycle of bandwidth.
        factor = int(np.ceil(oversample * band / rate - 1e-9))
        if factor > 1:
            envelope, rate = upsample(envelope, factor), rate * factor
    else:
        if cfg.get("duration") is None:
            raise ValueError("sampled waveform: `duration` is required for a built-in source")
        duration = float(cfg["duration"])
        rate = oversample * band
        n = int(np.floor(duration * rate)) + 1
        if n < 4:
            raise ValueError("sampled waveform: time-bandwidth product too small to sample; raise `oversample`")
        u = np.arange(n) / rate
        if source == "lfm":
            phase = (f0 + 0.5 * (f1 - f0) / duration * u) * u
            envelope = _tukey(u / duration, alpha) * np.exp(2j * np.pi * (phase - carrier * u))
        elif source == "hfm":
            # Hyperbolic FM: the instantaneous period 1/f is linear in time from 1/f0 to 1/f1,
            # so the phase is the integral of 1 / (1/f0 + k u).
            k = (1.0 / f1 - 1.0 / f0) / duration
            phase = np.log1p(k * f0 * u) / k
            envelope = _tukey(u / duration, alpha) * np.exp(2j * np.pi * (phase - carrier * u))
        elif source == "stepped_fm":
            steps = int(cfg.get("steps", 0))
            if steps < 2:
                raise ValueError("sampled waveform: stepped_fm needs `steps` >= 2")
            dwell = duration / steps
            i = np.minimum((u / dwell).astype(int), steps - 1)
            freqs = f0 + (f1 - f0) * np.arange(steps) / (steps - 1)
            start_phase = np.concatenate([[0.0], np.cumsum(freqs[:-1] * dwell)])      # phase-continuous
            phase = start_phase[i] + freqs[i] * (u - i * dwell)
            envelope = _tukey(u / duration, alpha) * np.exp(2j * np.pi * (phase - carrier * u))
        elif source in ("barker", "mseq", "code"):
            if source == "barker":
                length = int(cfg.get("code", 13))
                if length not in BARKER:
                    raise ValueError(f"sampled waveform: no Barker code of length {length}; lengths are {sorted(BARKER)}")
                chips = np.array(BARKER[length], dtype=complex)
            elif source == "mseq":
                chips = mseq(int(cfg["degree"]), cfg.get("taps")).astype(complex)
            else:
                chips = np.asarray(cfg["code"], dtype=float)
                if chips.ndim == 2 and chips.shape[1] == 2:
                    chips = chips[:, 0] + 1j * chips[:, 1]
                chips = chips.astype(complex)
            if band < len(chips) / duration:
                raise ValueError("sampled waveform: the band is narrower than the chip rate "
                                 f"({len(chips) / duration:.6g} Hz); widen f_start..f_stop or lengthen the pulse")
            envelope = _chip_envelope(chips, duration, rate, band, float(cfg.get("rolloff", 0.25)))
            envelope = envelope * np.exp(2j * np.pi * (0.5 * (f0 + f1) - carrier) * u)
        else:
            raise ValueError(f"sampled waveform: unknown source {source!r} (lfm, hfm, stepped_fm, barker, "
                             "mseq, code, samples, file)")
    normalise = str(cfg.get("normalise", "peak")).lower()
    if normalise == "peak":
        envelope = envelope / np.abs(envelope).max()
    elif normalise != "none":
        raise ValueError("sampled waveform: normalise must be `peak` or `none`")
    return SampledWaveform(envelope, rate, carrier, f0, f1, duration, **common)


# --- echo synthesis -----------------------------------------------------------------------

def _element_axes(boresight: np.ndarray):
    """Body-frame (boresight, az, el) of an element (libs/core directivity.hpp element_axes)."""
    b = np.asarray(boresight, dtype=float)
    b = b / np.linalg.norm(b)
    ref = np.array([0.0, 0.0, 1.0])
    along = ref @ b
    if abs(along) > 0.999:
        ref = np.array([1.0, 0.0, 0.0])
        az = ref - b * (ref @ b)
        az /= np.linalg.norm(az)
        return b, az, np.cross(b, az)
    el = ref - b * along
    el /= np.linalg.norm(el)
    return b, np.cross(el, b), el


def _sinc_pi(x):
    return np.sinc(x)


def _directivity(element, rotation: np.ndarray, direction: np.ndarray, ratio: np.ndarray) -> np.ndarray:
    """Amplitude response of an element towards world unit vectors `direction` [..., 3] at
    frequency ratios `ratio` [K] (f / f_ref): [..., K]. Mirrors libs/core directivity.hpp."""
    bw_az, bw_el = element.beamwidth_az, element.beamwidth_el
    omni_az, omni_el = not bw_az == bw_az, not bw_el == bw_el
    shape = direction.shape[:-1] + (len(ratio),)
    if omni_az and omni_el:
        return np.ones(shape)
    b, az, el = (rotation @ a for a in _element_axes(element.boresight))
    d = np.ones(shape)
    if not omni_az:
        d = d * _sinc_pi(0.886 * ratio * (direction @ az)[..., None] / bw_az)
    if not omni_el:
        d = d * _sinc_pi(0.886 * ratio * (direction @ el)[..., None] / bw_el)
    return np.where((direction @ b)[..., None] > 0.0, d, 0.0)


def _absorption(table, f: np.ndarray) -> np.ndarray:
    """dB/m at frequencies f from the (f_min, f_max, values) table the kernel uses."""
    if table is None:
        return np.zeros_like(f)
    lo, hi, values = table
    values = np.asarray(values, dtype=float)
    if not hi - lo > 0.0:
        return np.full_like(f, values[0])
    return np.interp(f, np.linspace(lo, hi, len(values)), values)


def synthesize_into(record: "core.PingRecord", waveform: SampledWaveform | None, q, p, amplitude,
                    q_velocity=None, p_velocity=None, *, absorption=None, spreading: bool = True,
                    platform_acceleration=(0.0, 0.0, 0.0), block: int = 1 << 21) -> None:
    """Add the echoes of two-way paths into record.samples for a sampled transmit waveform.

    Same arguments and geometry as `core.synthesize_into` (the analytic kernel). `waveform`
    None takes the replica from the record.
    """
    if waveform is None:
        waveform = SampledWaveform.from_record(record.transmit.waveform)
    s = record.samples
    if s.kind == core.SampleKind.ComplexAnalytic:
        raise ValueError("synthesize: analytic output is not supported; use real or baseband")
    baseband = s.kind == core.SampleKind.ComplexBaseband
    q = np.atleast_2d(np.asarray(q, dtype=float))
    p = np.atleast_2d(np.asarray(p, dtype=float))
    amp_path = np.atleast_1d(np.asarray(amplitude, dtype=complex)).astype(np.complex64).astype(complex)
    n_paths = len(q)
    if n_paths == 0:
        return
    qv = np.zeros_like(q) if q_velocity is None else np.asarray(q_velocity, dtype=float).reshape(q.shape)
    pv = np.zeros_like(p) if p_velocity is None else np.asarray(p_velocity, dtype=float).reshape(p.shape)
    c = record.environment.sound_speed
    fs_nom, start_nom, n_samples = s.sample_rate, s.start_time, s.n_samples
    f_ref = s.center_frequency
    T, F = waveform.duration, waveform.carrier
    accel = np.asarray(platform_acceleration, dtype=float)
    amp0 = record.transmit.waveform.amplitude

    # Elements in the world frame with first-order motion, as the kernel sees them.
    h = 1e-4
    geo = record.geometry
    rx_p = record.rx_positions(0.0)
    rx_v = (record.rx_positions(h) - record.rx_positions(-h)) / (2.0 * h)
    tx_p = np.asarray(record.tx_position(0.0))
    tx_v = (np.asarray(record.tx_position(h)) - np.asarray(record.tx_position(-h))) / (2.0 * h)
    rot_tx = np.asarray(core.rotation_ned(*geo.tx_pose.rpy))
    rot_rx = np.asarray(core.rotation_ned(*geo.rx_pose.rpy))
    n_ch = rx_p.shape[0]
    meta = record.metadata
    clock_offset, clock_drift = np.zeros(n_ch), np.zeros(n_ch)
    for i in range(min(n_ch, len(meta.channel_board))):
        b = meta.channel_board[i]
        if b < len(meta.boards):
            clock_offset[i] = meta.boards[b].clock_offset
            clock_drift[i] = meta.boards[b].clock_drift_ppm * 1e-6

    def arrival(u, sel):
        """Arrival time [P, C] and leg lengths of sound emitted at u, for paths `sel`.
        The same fixed-point solve as the kernel's arrival_time, run to convergence (each
        pass gains a factor speed / c; the kernel stops after 3 and 4 passes)."""
        tx_pos = tx_p + tx_v * u + accel * (0.5 * u * u)
        ts = np.full(len(q[sel]), u)
        for _ in range(8):
            r_out = np.linalg.norm(q[sel] + qv[sel] * ts[:, None] - tx_pos, axis=1)
            ts = u + r_out / c
        p_pos = p[sel] + pv[sel] * ts[:, None]
        t = np.repeat(ts[:, None], n_ch, axis=1)
        for _ in range(8):
            rx_pos = rx_p[None] + rx_v[None] * t[..., None] + accel * (0.5 * t[..., None] ** 2)
            r_back = np.linalg.norm(p_pos[:, None, :] - rx_pos, axis=2)
            t = ts[:, None] + r_back / c
        return t, r_out, r_back

    knot_f = waveform.knot_frequencies()
    ratio = knot_f / f_ref if f_ref > 0.0 else np.ones_like(knot_f)
    alpha = _absorption(absorption, knot_f)
    table_len = len(waveform.envelope) * max(waveform.table_factor, 1)
    waveform.synthesis_values(None, np.zeros((1, 1)))        # build the tables before the threads start
    workers = max(1, min(n_ch, os.cpu_count() or 1))
    block = max(block // workers, 1 << 16)
    out = s.array
    acc_re = np.zeros((n_ch, n_samples))
    acc_im = np.zeros((n_ch, n_samples)) if baseband else None

    outer = max(1, min(n_paths, (1 << 18) // max(n_ch, 1)))
    for o0 in range(0, n_paths, outer):
        sel = slice(o0, min(o0 + outer, n_paths))
        t0, _, _ = arrival(0.0, sel)
        tm, rom, rbm = arrival(0.5 * T, sel)
        t1, _, _ = arrival(T, sel)
        # Gain at mid-pulse for each knot frequency: [P, C, K].
        ts_mid = 0.5 * T + rom / c
        out_dir = (q[sel] + qv[sel] * ts_mid[:, None] - (tx_p + tx_v * (0.5 * T))) / rom[:, None]
        back_dir = (p[sel] + pv[sel] * ts_mid[:, None])[:, None, :] - (rx_p[None] + rx_v[None] * tm[..., None])
        back_dir = back_dir / rbm[..., None]
        gain = np.full(rbm.shape, amp0)
        if spreading:
            gain = gain / (rom[:, None] * rbm)
        knots = gain[..., None] * _directivity(geo.tx, rot_tx, out_dir, ratio)[:, None, :]
        for ic in range(n_ch):
            knots[:, ic, :] *= _directivity(geo.rx[ic], rot_rx, back_dir[:, ic, :], ratio)
        knots = knots * 10.0 ** (-alpha * (rom[:, None] + rbm)[..., None] / 20.0)
        flat = len(knot_f) == 1 or np.all(np.abs(knots - knots[..., :1]) <= 1e-12 * np.abs(knots[..., :1]))

        def channel(ic):
            fs = fs_nom * (1.0 + clock_drift[ic])
            start = start_nom + clock_offset[ic]
            a0, am, a1 = t0[:, ic], tm[:, ic], t1[:, ic]
            n_first = np.maximum(np.ceil((a0 - start) * fs), 0.0)
            n_last = np.minimum(np.floor((a1 - start) * fs), n_samples - 1.0)
            live = np.nonzero(n_last >= n_first)[0]
            if len(live) == 0:
                return
            length = int((n_last[live] - n_first[live]).max()) + 1
            per = max(1, block // max(length, table_len if not flat else 1))
            for i0 in range(0, len(live), per):
                rows = live[i0:i0 + per]
                n = n_first[rows, None].astype(np.int64) + np.arange(length)[None, :]
                ok = n <= n_last[rows, None]
                t = start + n / fs
                # Emission time: the quadratic through (t0, 0), (tm, T/2), (t1, T).
                ua = 0.5 * T / ((am[rows] - a0[rows]) * (am[rows] - a1[rows]))
                ub = T / ((a1[rows] - a0[rows]) * (a1[rows] - am[rows]))
                u = (t - a0[rows, None]) * (ua[:, None] * (t - a1[rows, None]) + ub[:, None] * (t - am[rows, None]))
                ok &= (u >= 0.0) & (u <= T)
                if flat:
                    env = waveform.synthesis_values(None, u) * knots[rows, ic, 0][:, None]
                else:
                    env = waveform.synthesis_values(knots[rows, ic, :], u)
                cyc = F * u
                if baseband:
                    # A baseband receiver mixes on its nominal clock.
                    cyc = cyc - f_ref * (start_nom + n / fs_nom)
                value = np.where(ok, amp_path[o0 + rows, None] * env * np.exp(2j * np.pi * (cyc - np.floor(cyc))), 0.0)
                idx = np.where(ok, n, 0).ravel()
                acc_re[ic] += np.bincount(idx, weights=value.real.ravel(), minlength=n_samples)[:n_samples]
                if baseband:
                    acc_im[ic] += np.bincount(idx, weights=value.imag.ravel(), minlength=n_samples)[:n_samples]

        # One channel per task: each writes its own row, and NumPy releases the interpreter
        # lock inside the array operations.
        if workers > 1:
            with ThreadPoolExecutor(workers) as pool:
                list(pool.map(channel, range(n_ch)))
        else:
            for ic in range(n_ch):
                channel(ic)
    kind = s.kind
    s.array = (out + (acc_re + 1j * acc_im if baseband else acc_re)).astype(out.dtype)
    s.kind = kind
    record.samples = s


_warned = False


def note_numpy_fallback(record: "core.PingRecord", backend, engine) -> None:
    """Make the engine used for a sampled pulse visible: the record's extra_json says
    {"gpusonar": {"echo_synthesis": "numpy_sampled"}}, and asking for the device backend or
    the C++ engine by name warns once that this step ran in NumPy instead."""
    import json
    global _warned
    extra = json.loads(record.metadata.extra_json or "{}")
    extra.setdefault("gpusonar", {})["echo_synthesis"] = "numpy_sampled"
    meta = record.metadata
    meta.extra_json = json.dumps(extra)
    record.metadata = meta
    asked = backend == core.Backend.Cuda or (engine or os.environ.get("GPUSONAR_SIM_ENGINE") or "").lower() == "cpp"
    if asked and not _warned:
        import warnings
        _warned = True
        warnings.warn("sampled transmit waveform: echoes are synthesised by the NumPy engine on the host "
                      "(the C++/CUDA synthesis kernel takes CW, LFM and Gaussian pulses only)",
                      RuntimeWarning, stacklevel=3)


# --- pulse compression (reference implementation) -------------------------------------------

def pulse_compress_reference(record: "core.PingRecord", doppler_scale: float = 1.0) -> np.ndarray:
    """Matched filter of a complex baseband record against the record's sampled replica, in
    NumPy, with the conventions of core.pulse_compress (lag = replica start, unit echo ->
    unit peak). Returns the compressed samples; the record is not changed. A reference for
    the C++ implementation, which is the one processing uses."""
    s = record.samples
    if s.kind != core.SampleKind.ComplexBaseband:
        raise ValueError("pulse_compress_reference: samples must be complex baseband")
    wf = SampledWaveform.from_record(record.transmit.waveform)
    rep = wf.baseband(s.sample_rate, s.center_frequency, doppler_scale)
    x = s.array
    n = x.shape[1]
    n_fft = int(2 ** np.ceil(np.log2(n + len(rep))))
    hf = np.conj(np.fft.fft(rep, n_fft)) / np.sum(np.abs(rep) ** 2)
    return np.fft.ifft(np.fft.fft(x, n_fft, axis=1) * hf, axis=1)[:, :n]
