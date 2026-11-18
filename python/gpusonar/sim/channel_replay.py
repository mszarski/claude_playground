"""Measured time-varying channels: replay, analysis, and comparison with our time spreading.

A measured channel is a complex-baseband impulse response h(t, tau) per receiver: `h[n, m, k]`
at time n / fs_time, receiver m, delay k / fs_delay, about a centre frequency fc. This is the
form of the Underwater Acoustic Channel Repository (UACR, `uwa-channels`; Li, Chitre, Cuji,
Preisig, Singer, Stojanovic, van Walree, "Underwater Acoustic Channel Library",
arXiv:2609.03207, 2026; data: Zenodo doi:10.5281/zenodo.21287414, CC BY 4.0). `load_uacr` reads
its files from the published file specification (the fields `h_hat`, `theta_hat`, `phi_hat`,
`params`, `meta`); none of the project's replay code was read or copied.

Three things are done with such a channel.

1. `replay`: pass any transmit signal through it,
       y_m(t) = sum_k h_m(t, k / fs_delay) x(t - k / fs_delay) ,
   with h interpolated linearly in t between the stored snapshots, the signal taken to complex
   baseband at fs_delay and back, and the file's tracked phase put back (`theta_hat`: phase
   only; `phi_hat`: phase and the delay drift it stands for, phi = -2 pi fc dtau, UACR file
   specification). A one-way propagation tier made of a measurement instead of a model.

2. Analysis: the quantities our time-spreading model (sim/timespread.py) predicts, estimated
   from h(t, tau) per arrival: the delay spread after the arrival (mean excess delay and
   exponential decay time), the coherent share of its energy, the width and shape of its
   Doppler spectrum, its coherence time, the statistics of its level from snapshot to snapshot,
   and the delay-Doppler scattering function.

3. Comparison: `model_channel` makes h(t, tau) from our own eigenray providers (image source
   or Bellhop, wrapped by `timespread.wrap`) for the experiment's geometry, band-limited and
   sampled like the measurement and passed through the measurement's own channel estimator
   (as far as its description allows), so that the same estimators run on both.

What a channel estimate is not
------------------------------
The stored h is the output of an adaptive estimator (LMS or RLS, updated every delay sample;
arXiv:2609.03207 Sec. IV), not the channel. It follows fluctuations only up to the
estimator's bandwidth: for RLS with forgetting factor lambda the estimate is, to first order,
the channel averaged over an exponential window of 1 / (1 - lambda) updates, a one-pole
low-pass with 3 dB frequency (1 - lambda) fs_delay / (2 pi). Faster fluctuations are attenuated
and, being unmodelled, raise the estimate's noise floor. The snapshots are stored at fs_time,
so the Doppler axis ends at +-fs_time / 2. A phase-locked loop removes a common phase per
receiver before estimation; what it removes is in `theta_hat`. `estimator_filter` applies the
one-pole model of this to a model channel; it is our approximation, not the estimator.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np


# =================================================================================================
# The channel
# =================================================================================================

@dataclass
class TimeVaryingChannel:
    h: np.ndarray                         # [T, M, K] complex baseband
    fs_time: float                        # Hz, snapshots
    fs_delay: float                       # Hz, delay axis
    fc: float                             # Hz
    phase: np.ndarray | None = None       # [N, M] rad, sampled at fs_delay (theta_hat or phi_hat)
    phase_kind: str = ""                  # "", "theta" (phase only), "phi" (phase and delay drift)
    meta: dict = field(default_factory=dict)
    name: str = ""

    @property
    def n_time(self) -> int:
        return int(self.h.shape[0])

    @property
    def n_receivers(self) -> int:
        return int(self.h.shape[1])

    @property
    def n_delay(self) -> int:
        return int(self.h.shape[2])

    @property
    def duration(self) -> float:
        return self.n_time / self.fs_time

    @property
    def delay(self) -> np.ndarray:
        return np.arange(self.n_delay) / self.fs_delay

    @property
    def time(self) -> np.ndarray:
        return np.arange(self.n_time) / self.fs_time

    def estimator_bandwidth(self) -> float | None:
        """3 dB frequency (Hz) of the one-pole model of the file's RLS estimator, or None when
        the file does not say (LMS: the step size alone does not give it)."""
        lam = self.meta.get("lambda")
        if lam is None or not 0.0 < float(lam) < 1.0:
            return None
        return (1.0 - float(lam)) * self.fs_delay / (2.0 * np.pi)


def _mat_string(a) -> str:
    return "".join(chr(int(c)) for c in np.asarray(a).ravel())


def load_uacr(path: str, receivers=None) -> TimeVaryingChannel:
    """Read a UACR channel file (MATLAB -v7.3, i.e. HDF5: h5py shows `h_hat` as
    [time, receiver, delay]; or MAT v5 through scipy, where it is [delay, receiver, time]).
    `receivers`: indices to load (the files are large)."""
    import h5py
    path = str(path)
    meta: dict = {}
    if h5py.is_hdf5(path):
        with h5py.File(path, "r") as f:
            version = float(np.asarray(f["version"]).ravel()[0]) if "version" in f else 1.0
            n_rx = f["h_hat"].shape[1]
            rx = list(range(n_rx)) if receivers is None else [int(r) for r in receivers]
            raw = f["h_hat"][:, rx, :] if rx == sorted(rx) else np.stack([f["h_hat"][:, r, :] for r in rx], axis=1)
            h = raw["real"] + 1j * raw["imag"] if raw.dtype.names else raw.astype(np.complex128)
            par = {k: float(np.asarray(f["params"][k]).ravel()[0]) for k in f["params"]}
            phase, kind = None, ""
            for key, k in (("phi_hat", "phi"), ("theta_hat", "theta")):
                if key in f:
                    phase = np.asarray(f[key][:, rx] if rx == sorted(rx) else np.stack([f[key][:, r] for r in rx], axis=1), dtype=np.float64)
                    kind = k
                    break
            if "f_resamp" in f:
                meta["f_resamp"] = np.asarray(f["f_resamp"]).ravel()
            if "meta" in f:
                for k in f["meta"]:
                    d = f["meta"][k]
                    if d.dtype == np.uint16:
                        meta[k] = _mat_string(d[()])
                    elif d.size == 1:
                        meta[k] = float(np.asarray(d).ravel()[0])
    else:
        from scipy.io import loadmat
        m = loadmat(path, squeeze_me=True, struct_as_record=False)
        version = float(m.get("version", 1.0))
        h = np.moveaxis(np.atleast_3d(np.asarray(m["h_hat"], dtype=np.complex128)), (0, 1, 2), (2, 1, 0))
        rx = list(range(h.shape[1])) if receivers is None else [int(r) for r in receivers]
        h = h[:, rx, :]
        par = {k: float(getattr(m["params"], k)) for k in ("fs_delay", "fs_time", "fc")}
        phase, kind = None, ""
        for key, k in (("phi_hat", "phi"), ("theta_hat", "theta")):
            if key in m:
                phase = np.atleast_2d(np.asarray(m[key], dtype=np.float64)).T[:, rx]
                kind = k
                break
        if "meta" in m:
            for k in getattr(m["meta"], "_fieldnames", []):
                v = getattr(m["meta"], k)
                meta[k] = v if isinstance(v, str) else (float(v) if np.size(v) == 1 else v)
    if version < 1.0:
        raise ValueError(f"{path}: channel file version {version} is older than 1.0")
    meta["receivers"] = rx
    return TimeVaryingChannel(h=h, fs_time=par["fs_time"], fs_delay=par["fs_delay"], fc=par["fc"], phase=phase, phase_kind=kind,
                              meta=meta, name=str(meta.get("codename") or os.path.splitext(os.path.basename(path))[0]))


# =================================================================================================
# Replay
# =================================================================================================

def _resample(x: np.ndarray, rate_in: float, rate_out: float) -> np.ndarray:
    from scipy.signal import resample_poly
    if rate_in == rate_out:
        return x
    r = Fraction(rate_out / rate_in).limit_denominator(4096)
    if abs(float(r) * rate_in / rate_out - 1.0) > 1e-6:
        raise ValueError(f"cannot resample {rate_in} Hz to {rate_out} Hz with a small rational factor")
    return resample_poly(x, r.numerator, r.denominator, axis=-1)


def replay(channel: TimeVaryingChannel, x, sample_rate: float, *, receivers=None, start: float = 0.0,
           apply_phase: bool = True, baseband: bool = False) -> np.ndarray:
    """Received signals [receivers, samples] at `sample_rate` for the transmit signal `x`.

    x: real passband samples (default), or with `baseband=True` the complex envelope about the
    channel's centre frequency; the output is of the same kind. Only the part of x inside the
    channel's band (fs_delay wide about fc; the estimate itself covers half of it) is passed.
    start: s into the measurement at which the signal starts. The signal, with the channel's
    delay span, must fit in what remains of the measurement.
    apply_phase: put back the tracked phase (and, for `phi_hat`, its delay drift), relative to
    its value at `start`.
    Scale: as stored. UACR files are normalised so that an in-band signal of unit power gives
    output powers that sum to the number of receivers in the file."""
    x = np.asarray(x)
    fs_d, fc = channel.fs_delay, channel.fc
    n_in = x.shape[-1]
    if baseband:
        xb = _resample(x.astype(np.complex128), sample_rate, fs_d)
    else:
        t = np.arange(n_in) / sample_rate
        xb = _resample(2.0 * x.astype(np.float64) * np.exp(-2j * np.pi * fc * t), sample_rate, fs_d)
    rx = list(range(channel.n_receivers)) if receivers is None else [int(r) for r in receivers]
    k = channel.n_delay
    n = len(xb) + k - 1
    n0 = int(round(start * fs_d))
    pos = (n0 + np.arange(n)) * channel.fs_time / fs_d              # snapshot index of every output sample
    if pos[-1] > channel.n_time - 1:
        raise ValueError(f"replay: the signal ({n / fs_d:.2f} s with the delay span) runs past the measurement "
                         f"({channel.duration:.2f} s, start {start:.2f} s)")
    xpad = np.concatenate([np.zeros(k - 1, dtype=np.complex128), xb, np.zeros(k - 1, dtype=np.complex128)])
    win = np.lib.stride_tricks.sliding_window_view(xpad, k)[:, ::-1]   # row n: x[n], x[n-1], ..., x[n-k+1]
    y = np.zeros((len(rx), n), dtype=np.complex128)
    i0 = np.floor(pos).astype(int)
    edges = np.flatnonzero(np.diff(i0)) + 1
    for a, b in zip(np.concatenate([[0], edges]), np.concatenate([edges, [n]])):
        i = i0[a]
        w = (pos[a:b] - i)[:, None]
        ha = channel.h[i][rx].T                                         # [K, M]
        hb = channel.h[min(i + 1, channel.n_time - 1)][rx].T
        seg = win[a:b]
        y[:, a:b] = ((1.0 - w) * (seg @ ha) + w * (seg @ hb)).T
    if apply_phase and channel.phase is not None:
        idx = np.clip(n0 + np.arange(n), 0, channel.phase.shape[0] - 1)
        ph = channel.phase[idx][:, rx].T - channel.phase[min(n0, channel.phase.shape[0] - 1)][rx][:, None]
        y = y * np.exp(1j * ph)
        if channel.phase_kind == "phi":
            # The delay drift the phase stands for: evaluate at n Ts + phi / (2 pi fc).
            from scipy.interpolate import CubicSpline
            grid = np.arange(n)
            for m in range(len(rx)):
                at = np.clip(grid + ph[m] / (2.0 * np.pi * fc) * fs_d, 0, n - 1)
                y[m] = CubicSpline(grid, y[m])(at)
    if "f_resamp" in channel.meta and np.any(np.abs(np.asarray(channel.meta["f_resamp"]) - 1.0) > 0):
        raise NotImplementedError("replay: files with a constant resampling factor (f_resamp) are not supported")
    out = _resample(y, fs_d, sample_rate)
    if baseband:
        return out
    t = np.arange(out.shape[-1]) / sample_rate
    return np.real(out * np.exp(2j * np.pi * fc * t))


# =================================================================================================
# Analysis
# =================================================================================================

def reference_phase(channel: TimeVaryingChannel, delay_bin=None, half_width: int = 1) -> np.ndarray:
    """Unit phasors [T, M] of the reference arrival (default: each receiver's strongest delay
    bin): dividing by them removes what is common to all arrivals (source and receiver motion,
    clocks, what the estimator's phase loop left)."""
    p = (np.abs(channel.h) ** 2).mean(axis=0)                        # [M, K]
    out = np.empty((channel.n_time, channel.n_receivers), dtype=np.complex128)
    for m in range(channel.n_receivers):
        b = int(np.argmax(p[m])) if delay_bin is None else int(np.atleast_1d(delay_bin)[m if np.size(delay_bin) > 1 else 0])
        s = channel.h[:, m, max(b - half_width, 0):b + half_width + 1].sum(axis=1)
        out[:, m] = s / np.maximum(np.abs(s), 1e-300)
    return out


def delay_profile(channel: TimeVaryingChannel, reference: np.ndarray | None = None, slow: float = 0.0) -> dict:
    """Intensity against delay per receiver: `total` = <|h|^2>, `coherent` = |<h>|^2 (after
    dividing by the reference phase, if given; `slow` > 0: the mean over a sliding window of
    that many seconds instead of the whole record, which tolerates slow drift), `floor` = the
    median over delay of the total (the estimator's noise where the channel is sparse)."""
    h = channel.h if reference is None else channel.h * np.conj(reference)[:, :, None]
    total = (np.abs(h) ** 2).mean(axis=0)
    if slow > 0.0:
        n = max(2, int(round(slow * channel.fs_time)))
        c = np.cumsum(np.concatenate([np.zeros((1,) + h.shape[1:]), h], axis=0), axis=0)
        mean = (c[n:] - c[:-n]) / n
        coherent = (np.abs(mean) ** 2).mean(axis=0) - total / n       # less the share of the fluctuating part left by a finite window
    else:
        coherent = np.abs(h.mean(axis=0)) ** 2
    return {"delay": channel.delay, "total": total, "coherent": np.maximum(coherent, 0.0), "floor": np.median(total, axis=1)}


def find_arrivals(profile: np.ndarray, n: int = 6, min_separation: int = 3, threshold_db: float = -25.0) -> list:
    """Delay bins of up to `n` local maxima of an intensity profile, strongest first, at least
    `min_separation` bins apart and above `threshold_db` relative to the strongest."""
    p = np.asarray(profile, dtype=float)
    peaks = [i for i in range(1, len(p) - 1) if p[i] >= p[i - 1] and p[i] > p[i + 1]]
    peaks.sort(key=lambda i: -p[i])
    out: list = []
    for i in peaks:
        if p[i] < p.max() * 10.0 ** (threshold_db / 10.0) or len(out) >= n:
            break
        if all(abs(i - j) >= min_separation for j in out):
            out.append(i)
    return out


def doppler_spectrum(x: np.ndarray, fs_time: float, nfft: int = 512):
    """(nu Hz, S [..]) Welch spectrum over axis 0 of complex series, Hann blocks at 50 % overlap,
    summed over the other axes; normalised so that sum(S) * dnu = mean power."""
    x = np.asarray(x)
    x = x.reshape(x.shape[0], -1)
    nfft = int(min(nfft, x.shape[0]))
    w = np.hanning(nfft + 1)[:nfft]
    hop = nfft // 2
    acc = np.zeros(nfft)
    count = 0
    for s in range(0, x.shape[0] - nfft + 1, hop):
        acc += (np.abs(np.fft.fft(x[s:s + nfft] * w[:, None], axis=0)) ** 2).sum(axis=1)
        count += 1
    s = np.fft.fftshift(acc) / (count * fs_time * np.sum(w**2))
    return np.fft.fftshift(np.fft.fftfreq(nfft, 1.0 / fs_time)), s


def spectrum_widths(nu: np.ndarray, s: np.ndarray, floor: float = 0.0, line_half_width: float = 0.0) -> dict:
    """Widths of a Doppler spectrum about zero, after subtracting `floor` (a level) and leaving
    out |nu| <= line_half_width (a coherent line):

    hwhm     half width at half the maximum of the smoothed pedestal
    w50, w90 half widths holding 50 % and 90 % of the pedestal's power within the axis
    rms      second-moment width over |nu| <= 6 w50 (four standard deviations of a Gaussian),
             with the floor subtracted without clipping: the far wings, where only the floor's
             own fluctuation is left, would otherwise dominate a second moment
    shape    w90 / w50: 2.44 for a Gaussian, 6.31 for a Lorentzian on an unlimited axis
    A Gaussian of rms width s has hwhm 1.177 s, w50 0.674 s, w90 1.645 s; a Lorentzian of half
    width a has hwhm a, w50 a, w90 6.31 a (less on a limited axis)."""
    nu = np.asarray(nu, dtype=float)
    signed = np.asarray(s, dtype=float) - floor
    keep = np.abs(nu) > line_half_width
    if line_half_width > 0.0:
        # Fill the line's bins with the level just outside it, so that the pedestal is whole.
        edge = signed[keep & (np.abs(nu) <= 2.0 * line_half_width + (nu[1] - nu[0]))]
        signed = np.where(keep, signed, edge.mean() if len(edge) else 0.0)
    p = np.maximum(signed, 0.0)
    total = p.sum()
    if total <= 0.0:
        return {"hwhm": 0.0, "w50": 0.0, "w90": 0.0, "rms": 0.0, "shape": 0.0}
    order = np.argsort(np.abs(nu))
    cum = np.cumsum(p[order]) / total
    a = np.abs(nu)[order]
    w50 = float(np.interp(0.5, cum, a))
    w90 = float(np.interp(0.9, cum, a))
    core = np.abs(nu) <= max(6.0 * w50, 3.0 * (nu[1] - nu[0]))
    rms = float(np.sqrt(max((signed[core] * nu[core] ** 2).sum(), 0.0) / max(signed[core].sum(), 1e-300)))
    # Half maximum of the folded spectrum, lightly smoothed.
    half = len(nu) // 2
    fold = 0.5 * (p[half:][: half] + p[: half + 1][::-1][: half])
    sm = np.convolve(fold, np.ones(3) / 3.0, mode="same")
    sm[0] = fold[:2].mean()
    below = np.flatnonzero(sm < 0.5 * sm[0])
    dnu = nu[1] - nu[0]
    if len(below) and below[0] > 0:
        j = below[0]
        hwhm = float((j - 1 + (sm[j - 1] - 0.5 * sm[0]) / max(sm[j - 1] - sm[j], 1e-300)) * dnu)
    else:
        hwhm = float(a.max()) if not len(below) else float(dnu)
    return {"hwhm": hwhm, "w50": w50, "w90": w90, "rms": rms, "shape": w90 / max(w50, 1e-12)}


def coherence_time(x: np.ndarray, fs_time: float, level: float = np.exp(-1.0)) -> float:
    """Lag (s) at which the magnitude of the time autocorrelation of the series (axis 0, summed
    over the other axes, means removed) first falls to `level` of its value at zero lag."""
    x = np.asarray(x)
    x = x.reshape(x.shape[0], -1)
    x = x - x.mean(axis=0, keepdims=True)
    n = x.shape[0]
    nf = 1 << int(np.ceil(np.log2(2 * n)))
    r = np.fft.ifft(np.abs(np.fft.fft(x, nf, axis=0)) ** 2, axis=0)[: n // 2].sum(axis=1)
    r = np.abs(r / (n - np.arange(n // 2)))
    r = r / max(r[0], 1e-300)
    below = np.flatnonzero(r < level)
    if not len(below):
        return float("inf")
    j = below[0]
    return float((j - 1 + (r[j - 1] - level) / max(r[j - 1] - r[j], 1e-300)) / fs_time)


def arrival_statistics(channel: TimeVaryingChannel, receiver: int, delay_bin: int, *, half_width: int = 1, tail_bins: int = 12,
                       reference: np.ndarray | None = None, floor: float | None = None, line_half_width: float = 0.3,
                       slow: float = 2.0, nfft: int = 512, pulse: np.ndarray | None = None) -> dict:
    """Everything our time-spreading model predicts, for the arrival at `delay_bin` of one receiver.

    The arrival's own bins are delay_bin +- half_width; its tail is the `tail_bins` after the
    peak. `reference`: unit phasors from `reference_phase` (strongly recommended on real data).
    `floor`: the estimate's noise level per delay bin (default: the median of the profile).
    `pulse`: the intensity response of the measurement to a single path, [bins from the peak
    on], to subtract the instrument from the tail (e.g. the direct arrival's profile).

    Returned: level (mean power of the arrival's bins, floor subtracted); coherent_fraction
    (power of the mean over the whole record) and coherent_fraction_slow (of the mean over
    `slow` seconds), both of the arrival's bins; coherent_fraction_arrival, the same over the
    arrival's bins and its tail together (the number to compare with exp(-Gamma^2): at the
    peak alone the scattered part, spread in delay, counts for less); coherent_line_fraction (power within
    +-line_half_width Hz above the pedestal, from the Doppler spectrum); scintillation_index
    and level_std_db of the arrival's energy per snapshot; the Doppler widths of
    `spectrum_widths` for the pedestal (`doppler`), with the spectrum itself; coherence_time_s
    of the fluctuating part; mean_excess_delay_s and decay_time_s of the incoherent profile
    after the peak (decay_time_s: slope of a straight-line fit to its logarithm)."""
    h = channel.h[:, receiver, :]
    if reference is not None:
        h = h * np.conj(reference[:, receiver])[:, None]
    k0, k1 = max(delay_bin - half_width, 0), min(delay_bin + half_width + 1, channel.n_delay)
    own = h[:, k0:k1]
    total_profile = (np.abs(h) ** 2).mean(axis=0)
    if floor is None:
        floor = float(np.median(total_profile))
    n_own = k1 - k0
    level = float(total_profile[k0:k1].sum() - n_own * floor)
    if level <= 0.0:
        raise ValueError(f"arrival_statistics: nothing above the floor at delay bin {delay_bin}")
    mean = own.mean(axis=0)
    coh = float((np.abs(mean) ** 2).sum() / max(level, 1e-300))
    n_slow = max(2, int(round(slow * channel.fs_time)))
    c = np.cumsum(np.concatenate([np.zeros((1, n_own)), own], axis=0), axis=0)
    mslow = (c[n_slow:] - c[:-n_slow]) / n_slow
    coh_slow = float("nan")
    if len(mslow):                                                       # the record is longer than the window
        coh_slow = float(((np.abs(mslow) ** 2).mean(axis=0).sum() - total_profile[k0:k1].sum() / n_slow) / max(level, 1e-300))
    energy = (np.abs(own) ** 2).sum(axis=1)
    e_mean = energy.mean()
    # Scintillation of the arrival's energy, with the floor's share of mean and variance taken out
    # (the floor is complex Gaussian: variance of |n|^2 is floor^2 per bin, and 2 floor x signal cross term).
    sig = max(e_mean - n_own * floor, 1e-300)
    var = energy.var() - n_own * floor**2 - 2.0 * floor * sig
    si = float(max(var, 0.0) / sig**2)
    nu, s = doppler_spectrum(own, channel.fs_time, nfft)
    dnu = nu[1] - nu[0]
    s_floor = n_own * floor / channel.fs_time                        # a white floor of that power
    line = np.abs(nu) <= line_half_width
    near = (np.abs(nu) > line_half_width) & (np.abs(nu) <= 2.0 * line_half_width + dnu)
    pedestal_at_line = s[near].mean() if near.any() else 0.0
    line_power = float(np.maximum(s[line] - pedestal_at_line, 0.0).sum() * dnu)
    widths = spectrum_widths(nu, s, floor=s_floor, line_half_width=line_half_width)
    fluct = own - mean[None, :]
    tau_c = coherence_time(fluct, channel.fs_time)
    # Tail after the peak: incoherent profile less floor, and less the instrument's own response.
    t1 = min(delay_bin + tail_bins + 1, channel.n_delay)
    inc = total_profile[delay_bin:t1] - floor - np.abs(h[:, delay_bin:t1].mean(axis=0)) ** 2
    inc_raw = np.maximum(inc, 0.0)
    whole = total_profile[k0:t1].sum() - (t1 - k0) * floor
    coh_arrival = float((np.abs(h[:, k0:t1].mean(axis=0)) ** 2).sum() / whole) if whole > 0.0 else float("nan")
    if pulse is not None:
        pl = np.asarray(pulse, dtype=float)[: len(inc)]
        pl = np.concatenate([pl, np.zeros(len(inc) - len(pl))])
        inc = inc - pl / max(pl[0], 1e-300) * inc[0]
    inc = np.maximum(inc, 0.0)
    lag = np.arange(len(inc)) / channel.fs_delay
    mean_excess = float((inc_raw * lag).sum() / max(inc_raw.sum(), 1e-300))
    good = inc_raw[1:] > max(2.0 * floor / np.sqrt(max(channel.n_time / 8.0, 1.0)), 1e-4 * inc_raw[0], 1e-300)
    stop = np.argmax(~good) + 1 if (~good).any() else len(inc_raw)
    if stop >= 4:
        slope = np.polyfit(lag[1:stop], np.log(inc_raw[1:stop]), 1)[0]
        decay = float(-1.0 / slope) if slope < 0 else float("inf")
    else:
        decay = float("nan")
    return {"delay_s": delay_bin / channel.fs_delay, "level": level, "level_db_re_floor": float(10 * np.log10(level / floor)) if floor > 0.0 else float("inf"),
            "coherent_fraction": coh, "coherent_fraction_arrival": coh_arrival, "coherent_fraction_slow": coh_slow, "coherent_line_fraction": line_power / max(level, 1e-300),
            "scintillation_index": si, "level_std_db": float(np.std(10.0 * np.log10(np.maximum(energy, 1e-300)))),
            "doppler": widths, "doppler_nu": nu, "doppler_spectrum": s, "doppler_floor": s_floor,
            "coherence_time_s": tau_c, "mean_excess_delay_s": mean_excess, "decay_time_s": decay,
            "tail_profile": inc_raw, "tail_fit_bins": int(stop)}


def subband_statistics(channel: TimeVaryingChannel, receiver: int, delay_bin: int, *, before: int = 2, after: int = 8,
                        reference: np.ndarray | None = None, fraction: float = 0.7, nfft: int = 512, line_half_width: float = 0.3) -> dict:
    """The arrival at `delay_bin` in the lower and the upper half of the band: a test of how its
    spreading scales with frequency inside one measurement.

    The arrival's bins (delay_bin - before .. delay_bin + after, Hann-tapered at the ends) are
    transformed along delay to H(t, f); f covers +-fs_delay / 2 about fc, of which the estimate
    occupies the middle half. The lower band is the bins in -fraction fs_delay / 4 .. 0, the
    upper 0 .. +fraction fs_delay / 4. Returned per band: centre frequency (Hz), rms Doppler
    width and coherent share (as `arrival_statistics`), and their upper / lower ratios. For a
    path whose length changes in time the Doppler width is proportional to frequency; for a
    rough-surface bounce the logarithm of the coherent share is proportional to its square."""
    h = channel.h[:, receiver, :]
    if reference is not None:
        h = h * np.conj(reference[:, receiver])[:, None]
    k0, k1 = max(delay_bin - before, 0), min(delay_bin + after + 1, channel.n_delay)
    seg = h[:, k0:k1]
    n = seg.shape[1]
    taper = np.ones(n)
    edge = max(1, min(before, 2))
    taper[:edge] = 0.5 * (1.0 - np.cos(np.pi * (np.arange(edge) + 0.5) / edge))
    taper[-2:] = taper[:2][::-1] if n > 4 else 1.0
    size = 64
    hf = np.fft.fft(seg * taper, size, axis=1)
    f = np.fft.fftfreq(size, 1.0 / channel.fs_delay)
    edge_f = fraction * channel.fs_delay / 4.0
    out = {}
    for name, sel in (("lower", (f < 0) & (f >= -edge_f)), ("upper", (f > 0) & (f <= edge_f))):
        x = hf[:, sel]
        nu, sp = doppler_spectrum(x, channel.fs_time, nfft)
        w = spectrum_widths(nu, sp, floor=float(np.median(sp[np.abs(nu) > 0.35 * channel.fs_time])), line_half_width=line_half_width)
        power = (np.abs(x) ** 2).mean(axis=0).sum()
        out[name] = {"frequency": float(channel.fc + f[sel].mean()), "doppler_rms": w["rms"], "doppler_w50": w["w50"],
                     "coherent_fraction": float((np.abs(x.mean(axis=0)) ** 2).sum() / max(power, 1e-300))}
    out["frequency_ratio"] = out["upper"]["frequency"] / out["lower"]["frequency"]
    out["doppler_ratio"] = out["upper"]["doppler_rms"] / max(out["lower"]["doppler_rms"], 1e-12)
    return out


def scattering_function(channel: TimeVaryingChannel, receiver: int = 0, nfft: int = 256, reference: np.ndarray | None = None):
    """Delay-Doppler scattering function S[nu, tau] of one receiver: the Welch spectrum over
    time of every delay bin. Returns (nu Hz, tau s, S)."""
    h = channel.h[:, receiver, :]
    if reference is not None:
        h = h * np.conj(reference[:, receiver])[:, None]
    nfft = int(min(nfft, h.shape[0]))
    w = np.hanning(nfft + 1)[:nfft]
    acc = np.zeros((nfft, h.shape[1]))
    count = 0
    for s in range(0, h.shape[0] - nfft + 1, nfft // 2):
        acc += np.abs(np.fft.fft(h[s:s + nfft] * w[:, None], axis=0)) ** 2
        count += 1
    s = np.fft.fftshift(acc, axes=0) / (count * channel.fs_time * np.sum(w**2))
    return np.fft.fftshift(np.fft.fftfreq(nfft, 1.0 / channel.fs_time)), channel.delay, s


# =================================================================================================
# Synthetic channels (tests, and what a model looks like through the same estimators)
# =================================================================================================

def band_pulse(n_delay: int, fs_delay: float, bandwidth: float, delay: float) -> np.ndarray:
    """A path of unit amplitude at `delay` (s) as the response of a band `bandwidth` wide,
    sampled at fs_delay: a sinc of unit peak with a Hann taper over +-8 / bandwidth. Its
    energy summed over the delay bins is fs_delay / bandwidth."""
    x = bandwidth * (np.arange(n_delay) / fs_delay - delay)
    return np.sinc(x) * np.where(np.abs(x) < 8.0, 0.5 * (1.0 + np.cos(np.pi * x / 8.0)), 0.0)


def fading_process(n: int, fs_time: float, width: float, spectrum: str, rng: np.random.Generator, lines: int = 0) -> np.ndarray:
    """Unit-variance complex Gaussian series with a Doppler spectrum of the given kind:
    "gaussian" (rms width `width` Hz) or "lorentzian" (half width `width` Hz, i.e. the
    autocorrelation exp(-2 pi width |t|) of an AR(1) process). Filtered white noise."""
    if width <= 0.0:
        return np.full(n, np.exp(2j * np.pi * rng.uniform()) , dtype=np.complex128)
    nf = 1 << int(np.ceil(np.log2(4 * n)))
    nu = np.fft.fftfreq(nf, 1.0 / fs_time)
    if spectrum == "gaussian":
        shape = np.exp(-0.5 * (nu / width) ** 2)
    elif spectrum == "lorentzian":
        shape = 1.0 / (1.0 + (nu / width) ** 2)
    else:
        raise ValueError("spectrum is 'gaussian' or 'lorentzian'")
    z = (rng.standard_normal(nf) + 1j * rng.standard_normal(nf)) * np.sqrt(shape)
    x = np.fft.ifft(z)[:n]
    return x / np.sqrt(np.mean(np.abs(np.fft.ifft(z)) ** 2))


def synthetic_channel(paths: list, *, n_time: int, fs_time: float, fs_delay: float, n_delay: int, fc: float, bandwidth: float,
                      n_receivers: int = 1, noise: float = 0.0, common_phase_rms: float = 0.0, seed: int = 0,
                      taps_per_decay: int = 6) -> TimeVaryingChannel:
    """A channel with known spreads. Each path: {delay s, amplitude (coherent part, complex),
    incoherent (mean power of the scattered part), decay_time s (exponential tail; 0: none),
    doppler_width Hz, doppler_spectrum "gaussian"|"lorentzian"}. The scattered part is a set
    of closely spaced taps with independent fading and exponentially falling mean power.
    `noise`: mean power per delay bin of white estimation noise. `common_phase_rms`: a slow
    random phase common to all paths of a receiver (rad), which the reference removes."""
    rng = np.random.default_rng(seed)
    h = np.zeros((n_time, n_receivers, n_delay), dtype=np.complex128)
    for m in range(n_receivers):
        for p in paths:
            d = float(p["delay"])
            h[:, m, :] += complex(p.get("amplitude", 0.0)) * band_pulse(n_delay, fs_delay, bandwidth, d)[None, :]
            inc = float(p.get("incoherent", 0.0))
            if inc <= 0.0:
                continue
            decay = float(p.get("decay_time", 0.0))
            if decay > 0.0:
                step = min(decay / taps_per_decay, 0.5 / bandwidth)
                lags = np.arange(0.0, 6.0 * decay, step)
                weights = np.exp(-lags / decay)
            else:
                lags, weights = np.array([0.0]), np.array([1.0])
            weights = weights / weights.sum() * inc
            for lag, wgt in zip(lags, weights):
                g = fading_process(n_time, fs_time, float(p.get("doppler_width", 0.0)), str(p.get("doppler_spectrum", "gaussian")), rng)
                h[:, m, :] += np.sqrt(wgt) * g[:, None] * band_pulse(n_delay, fs_delay, bandwidth, d + lag)[None, :]
        if common_phase_rms > 0.0:
            drift = np.cumsum(rng.standard_normal(n_time))
            drift = drift / max(drift.std(), 1e-12) * common_phase_rms
            h[:, m, :] *= np.exp(1j * drift)[:, None]
    if noise > 0.0:
        h += np.sqrt(noise / 2.0) * (rng.standard_normal(h.shape) + 1j * rng.standard_normal(h.shape))
    return TimeVaryingChannel(h=h, fs_time=fs_time, fs_delay=fs_delay, fc=fc, meta={"synthetic": True}, name="synthetic")


def estimator_filter(channel: TimeVaryingChannel, bandwidth_hz: float) -> TimeVaryingChannel:
    """The channel as a tracking estimator of 3 dB bandwidth `bandwidth_hz` would report it: a
    causal one-pole low-pass along time (an exponential window). See the module docstring; this
    is a model of the estimator, not the estimator."""
    from scipy.signal import lfilter
    a = np.exp(-2.0 * np.pi * bandwidth_hz / channel.fs_time)
    h = lfilter([1.0 - a], [1.0, -a], channel.h, axis=0, zi=(channel.h[:1] * a))[0]
    return TimeVaryingChannel(h=h, fs_time=channel.fs_time, fs_delay=channel.fs_delay, fc=channel.fc, phase=channel.phase,
                              phase_kind=channel.phase_kind, meta=dict(channel.meta, estimator_bandwidth_applied=bandwidth_hz), name=channel.name)


# =================================================================================================
# Our model, in the measurement's form
# =================================================================================================

def model_channel(provider, source, receivers, *, n_time: int, fs_time: float, fs_delay: float, n_delay: int, fc: float,
                  bandwidth: float, lead: float = 0.0, absorption_db_per_km: float = 0.0, name: str = "model",
                  return_rays: bool = False):
    """h(t, tau) from an eigenray provider (image source or Bellhop, usually wrapped by
    `timespread.wrap`): for every snapshot time the provider's rays from `source` to each of
    `receivers` [M, 3] are laid on the delay axis as band-limited pulses (`band_pulse`) with
    their complex amplitudes and the carrier phase exp(-2 pi i fc delay).

    The delay axis starts `lead` seconds before the first arrival of each receiver (the
    measurement's axis has no absolute time either). Amplitudes are for unit pressure at 1 m.
    With `return_rays`, also the flat-boundary eigenrays (delay, n_surface, n_bottom, launch)."""
    source = np.asarray(source, dtype=float).reshape(3)
    receivers = np.asarray(receivers, dtype=float).reshape(-1, 3)
    m = len(receivers)
    h = np.zeros((n_time, m, n_delay), dtype=np.complex128)
    inner = getattr(provider, "inner", provider)
    base = inner.eigenrays(source, receivers)
    t0 = np.where(base.valid, base.delay, np.inf).min(axis=1) - lead          # [M]
    c = float(base.sound_speed)
    grid = np.arange(n_delay) / fs_delay
    amps, first = [], None
    for n in range(n_time):
        if hasattr(provider, "set_ping"):
            provider.set_ping(n / fs_time)
        rays = provider.eigenrays(source, receivers)
        if first is None:
            first = rays                                   # the geometry does not change: delays are those of the first snapshot
        amp = np.where(rays.valid, rays.amplitude, 0.0) * 10.0 ** (-absorption_db_per_km * c * rays.delay / 1000.0 / 20.0)
        # (With the provider's `doppler` option each line's phase is already that of this time.)
        amps.append(amp * np.exp(-2j * np.pi * fc * rays.delay))
    amps = np.array(amps)                                  # [T, M, K]
    for r in range(m):
        rel = first.delay[r] - t0[r]
        ok = first.valid[r] & (rel > -8.0 / bandwidth) & (rel < grid[-1] + 8.0 / bandwidth)
        x = bandwidth * (grid[None, :] - rel[ok][:, None])
        pulse = np.sinc(x) * np.where(np.abs(x) < 8.0, 0.5 * (1.0 + np.cos(np.pi * x / 8.0)), 0.0)   # band_pulse, all rays at once
        h[:, r, :] = amps[:, r, ok] @ pulse
    ch = TimeVaryingChannel(h=h, fs_time=fs_time, fs_delay=fs_delay, fc=fc, meta={"model": True, "lead": lead}, name=name)
    if return_rays:
        return ch, {"delay": base.delay - t0[:, None], "n_surface": base.n_surface, "n_bottom": base.n_bottom,
                    "launch": base.launch, "valid": base.valid, "amplitude": base.amplitude, "sound_speed": c}
    return ch
