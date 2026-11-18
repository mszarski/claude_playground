"""Channel sounding from raw hydrophone recordings of known probe signals.

`sim/channel_replay.py` analyses channels somebody else estimated (an adaptive tracker whose
bandwidth we do not control, PHYSICS_NOTES Sec. 21.2). Here the estimate is ours: the recording
of a known transmitted probe is correlated against the probe, snapshot by snapshot, and the
result is a `channel_replay.TimeVaryingChannel` h[n, m, k] that the estimators of that module
take unchanged. No tracker, no forgetting factor: each snapshot is made from its own stretch of
the recording and nothing else, so the estimate's response to a fluctuation of frequency nu is
the block average, sinc(pi nu T_block), and its Doppler axis ends at +-1 / (2 T_hop).

Estimators (all tested on synthetic channels with known answers, tests/python/test_channel_sounding.py):

`sound_periodic`   a probe that repeats with period P (chirp train, periodic m-sequence): each
                   period of the recording is circularly deconvolved by one period of the probe,
                   H = Y X* / (|X|^2 + eps), then shaped by a window W(f) over the probe's band.
                   Exact for a channel shorter than P and frozen over P; a channel longer than P
                   folds (delay aliasing), which shows as energy before the first arrival.
`sound_sliding`    any probe (aperiodic sequence): blocks of length T_block every T_hop, each
                   deconvolved by the same stretch of the transmitted signal (regularised, block
                   extended by the delay window; the probe's own correlation sidelobes are
                   removed to the extent the block is longer than the channel).
`align`            start time of the probe in the recording and the clock/Doppler time scale,
                   from the peak of the wideband ambiguity function over a few trial scales.
`tail_decay`       exponential decay time of the scattered tail of one arrival, with the
                   instrument's own pulse (taken from an arrival that has no tail, or the
                   window's transform) deconvolved by fitting  A p(tau) + B (p * e^{-tau/T})(tau)
                   to the incoherent profile, instead of a straight line to its logarithm
                   (which needs a tail several resolution cells long).
`coherent_split`   an arrival's power split into the part that is steady over the record (after
                   removing the reference arrival's phase) and the rest, with the estimate's
                   noise floor removed from both.

The loaders are for the SFI Smart Ocean dataset (SODAC): van Walree, Otnes, Tomasi, Henriksen,
Danre, Bergh, "SFI Smart Ocean dataset for underwater acoustic communications", IEEE DataPort
2025, doi:10.21227/3aa6-4k33 (CC BY 4.0); described in van Walree et al., "Descriptor: The SFI
Smart Ocean Dataset for Acoustic Communications (SODAC)", IEEE Data Descriptions 2 (2025)
218-223, doi:10.1109/IEEEDATA.2025.3577998 (read as hal-05793814). Taken from that article:
the file naming, 24-bit WAV at 96 kHz with the start time in the header's comment field, the
factor 94 from full scale to pascals (receivers calibrated to 20 kHz), the positions and
depths (Table II), the source levels (Table I). Probe definitions are NOT taken from the
supplementary waveform description (not available to us): the transmitted WAV files of the
dataset are used as the probes directly.

Nothing here loads a whole archive: a recording is one member of the zip (at most 56 MB),
read member by member, and reduced at once to complex baseband at a few kHz.
"""
from __future__ import annotations

import io
import os
import re
import struct
import zipfile
from fractions import Fraction

import numpy as np

from .channel_replay import TimeVaryingChannel

SODAC_PASCAL_PER_FULL_SCALE = 94.0          # Descriptor article, "Insights and notes" item 8
SODAC_POSITIONS = {                          # Descriptor article, Table II (latitude N, longitude E, depth m)
    "TX": (60.08888, 5.26785, 20.0),         # LF sender depth: Table I
    "R1": (60.09109, 5.26571, 43.0),
    "R2": (60.08478, 5.28424, 14.0),
}
SODAC_RANGE_M = {"R1": 270.0, "R2": 1020.0}  # Table II, as printed (0.27 and 1.02 km)
SODAC_SOURCE_LEVEL_DB = {"expected": 181.0, "measured": 183.0}   # Table I, LF, dB re 1 uPa^2 m^2


# =================================================================================================
# Reading
# =================================================================================================

def read_wav(f) -> tuple[np.ndarray, float, dict]:
    """(samples [N] or [N, C] float64 in +-1 full scale, sample rate, info) of a PCM (8/16/24/32
    bit) or IEEE-float WAV given as a path, bytes or binary file object. `info` holds the text
    fields of a LIST/INFO chunk (the SODAC recordings carry their start time in ICMT)."""
    if isinstance(f, (bytes, bytearray)):
        f = io.BytesIO(f)
    own = isinstance(f, (str, os.PathLike))
    fh = open(f, "rb") if own else f
    try:
        head = fh.read(12)
        if head[:4] != b"RIFF" or head[8:12] != b"WAVE":
            raise ValueError("read_wav: not a RIFF/WAVE file")
        fmt = None
        info: dict = {}
        data = None
        while True:
            h = fh.read(8)
            if len(h) < 8:
                break
            cid, n = h[:4], struct.unpack("<I", h[4:])[0]
            if cid == b"data":
                data = fh.read(n)
                break
            body = fh.read(n + (n & 1))
            if cid == b"fmt ":
                fmt = struct.unpack("<HHIIHH", body[:16])
            elif cid == b"LIST" and body[:4] == b"INFO":
                p = 4
                while p + 8 <= n:
                    k, m = body[p:p + 4], struct.unpack("<I", body[p + 4:p + 8])[0]
                    info[k.decode("ascii", "replace")] = body[p + 8:p + 8 + m].split(b"\0")[0].decode("latin-1")
                    p += 8 + m + (m & 1)
        if fmt is None or data is None:
            raise ValueError("read_wav: no fmt or data chunk")
        tag, channels, rate, _, _, bits = fmt
        if tag == 3:
            x = np.frombuffer(data, dtype="<f4" if bits == 32 else "<f8").astype(np.float64)
        elif tag in (1, 0xFFFE):
            if bits == 24:
                b = np.frombuffer(data[: len(data) // 3 * 3], dtype=np.uint8).reshape(-1, 3)
                v = b[:, 0].astype(np.int32) | (b[:, 1].astype(np.int32) << 8) | (b[:, 2].astype(np.int8).astype(np.int32) << 16)
                x = v / 8388608.0
            elif bits == 16:
                x = np.frombuffer(data, dtype="<i2") / 32768.0
            elif bits == 32:
                x = np.frombuffer(data, dtype="<i4") / 2147483648.0
            elif bits == 8:
                x = (np.frombuffer(data, dtype=np.uint8).astype(np.float64) - 128.0) / 128.0
            else:
                raise ValueError(f"read_wav: {bits}-bit PCM not supported")
        else:
            raise ValueError(f"read_wav: format tag {tag} not supported")
        if channels > 1:
            x = x[: len(x) // channels * channels].reshape(-1, channels)
        return x, float(rate), info
    finally:
        if own:
            fh.close()


def to_baseband(x: np.ndarray, fs: float, fc: float, fs_out: float, *, taps_per_phase: int = 24) -> np.ndarray:
    """Complex envelope of a real series about `fc`, sampled at `fs_out`: x(t) = Re{b(t) e^{i 2 pi fc t}}
    for a signal within +-0.46 fs_out of fc (Kaiser low-pass, 80 dB, then rational resampling).
    The envelope's sample n is at time n / fs_out of the input."""
    from scipy.signal import firwin, resample_poly
    x = np.asarray(x, dtype=np.float64)
    ratio = Fraction(fs_out / fs).limit_denominator(1000)
    up, down = ratio.numerator, ratio.denominator
    t = np.arange(len(x)) / fs
    z = 2.0 * x * np.exp(-2j * np.pi * fc * t)
    n = 2 * taps_per_phase * max(up, down) + 1
    h = firwin(n, 0.92 * min(fs_out, fs) / 2.0, window=("kaiser", 8.0), fs=fs * up)
    return resample_poly(z, up, down, window=h * 1.0)


def sodac_members(archive: str, receiver: str | None = None) -> list:
    """[(cycle, receiver, member name)] of the recordings in an RX-xx.zip, sorted."""
    out = []
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            m = re.search(r"([LMH]F)_(R[12])_cycle(\d+)\.wav$", name)
            if m and (receiver is None or m.group(2) == receiver):
                out.append((int(m.group(3)), m.group(2), name))
    return sorted(out)


def sodac_time(info: dict) -> float | None:
    """Seconds since 14-Nov-2024 00:00 UTC of a recording's start, from its header comment
    ('14-Nov-2024  16:59:54.417'), or None."""
    m = re.match(r"\s*(\d+)-(\w+)-(\d+)\s+(\d+):(\d+):([\d.]+)", info.get("ICMT", ""))
    if not m:
        return None
    return (int(m.group(1)) - 14) * 86400.0 + int(m.group(4)) * 3600.0 + int(m.group(5)) * 60.0 + float(m.group(6))


def load_sodac_recording(archive: str, member: str, fc: float, fs_out: float, *, pascal: bool = True):
    """(complex envelope about fc at fs_out, info) of one recording of an RX archive, read from
    the zip without extracting it. In pascals with `pascal` (factor 94, stated for <= 20 kHz)."""
    with zipfile.ZipFile(archive) as z:
        x, fs, info = read_wav(z.read(member))
    if x.ndim > 1:
        x = x[:, 0]
    b = to_baseband(x * (SODAC_PASCAL_PER_FULL_SCALE if pascal else 1.0), fs, fc, fs_out)
    info = dict(info, sample_rate=fs, start_s=sodac_time(info), member=member)
    return b.astype(np.complex64), info


def load_sodac_probe(archive: str, name: str, fc: float, fs_out: float) -> np.ndarray:
    """Complex envelope at fs_out of a transmitted waveform (e.g. 'LFM-25-LF') from TX-waveforms.zip,
    scaled to unit rms over its duration."""
    with zipfile.ZipFile(archive) as z:
        member = [n for n in z.namelist() if n.endswith("/" + name + ".wav")]
        if not member:
            raise KeyError(f"load_sodac_probe: no {name}.wav in {archive}")
        x, fs, _ = read_wav(z.read(member[0]))
    b = to_baseband(x, fs, fc, fs_out)
    return b / np.sqrt(np.mean(np.abs(b) ** 2))


def load_ctd(path: str, min_salinity: float = 5.0, step: float = 2.0) -> list:
    """Sound-speed profiles from the dataset's CTD.csv (columns: date, time UTC, salinity ppt,
    temperature degC, depth m, sound speed m/s — the sound speed is the file's own column, not
    recomputed). One entry per cast (a cast = one date): samples in water (salinity above
    `min_salinity`; the file starts and ends in air), split at the deepest sample into the way
    down and the way up, each averaged in `step`-metre depth bins; `speed` is the mean of the
    two and `hysteresis` their difference (up minus down: the sensor's lag). Returned per cast:
    {date, start, end, depth [m], speed, speed_down, speed_up, temperature, salinity, max_depth}."""
    import csv
    rows = []
    with open(path, encoding="latin-1") as f:
        r = csv.reader(f)
        next(r)
        for x in r:
            if len(x) >= 6:
                rows.append((x[0].strip(), x[1].strip(), float(x[2]), float(x[3]), float(x[4]), float(x[5])))
    casts = []
    for date in sorted({x[0] for x in rows}, key=lambda d: d.split("/")[::-1]):
        c = [x for x in rows if x[0] == date]
        a = np.array([x[2:] for x in c])
        wet = a[:, 0] > min_salinity
        i_deep = int(np.argmax(np.where(wet, a[:, 2], -1.0)))
        idx = np.arange(len(c))
        edges = np.arange(0.0, a[wet, 2].max() + step, step)
        mid = 0.5 * (edges[:-1] + edges[1:])

        def binned(sel, col):
            out = np.full(len(mid), np.nan)
            for j in range(len(mid)):
                s = sel & (a[:, 2] >= edges[j]) & (a[:, 2] < edges[j + 1])
                if s.any():
                    out[j] = a[s, col].mean()
            return out
        down, up = binned(wet & (idx <= i_deep), 3), binned(wet & (idx >= i_deep), 3)
        both = np.nanmean(np.stack([down, up]), axis=0)
        ok = np.isfinite(both)
        casts.append({"date": date, "start": c[0][1], "end": c[-1][1], "depth": mid[ok], "speed": both[ok],
                      "speed_down": down[ok], "speed_up": up[ok], "hysteresis": (up - down)[ok],
                      "temperature": np.nanmean(np.stack([binned(wet & (idx <= i_deep), 1), binned(wet & (idx >= i_deep), 1)]), axis=0)[ok],
                      "salinity": np.nanmean(np.stack([binned(wet & (idx <= i_deep), 0), binned(wet & (idx >= i_deep), 0)]), axis=0)[ok],
                      "max_depth": float(a[wet, 2].max())})
    return casts


# =================================================================================================
# Sounding
# =================================================================================================

def find_probe(y: np.ndarray, probe: np.ndarray, *, head: int | None = None) -> tuple[int, float]:
    """(sample of the recording at which the probe's strongest arrival starts, peak over the
    median of the correlation magnitude). `head`: correlate only that many leading samples of
    the probe (enough to find it; a long probe decorrelates over its length when the clocks
    differ)."""
    p = np.asarray(probe)[: head or len(probe)]
    n = 1 << int(np.ceil(np.log2(len(y) + len(p))))
    c = np.abs(np.fft.ifft(np.fft.fft(y, n) * np.conj(np.fft.fft(p, n))))[: len(y)]
    k = int(np.argmax(c))
    return k, float(c[k] / max(np.median(c), 1e-300))


def band_window(n: int, fs: float, band: float, kind: str = "hann") -> np.ndarray:
    """Weights over the n FFT bins of a complex envelope sampled at fs, scaled so that a single
    path keeps its amplitude at its peak: zero outside |f| <= band / 2, and inside "flat" (sinc pulse, first sidelobe -13 dB, resolution 1 / band) or "hann" (raised
    cosine over the band: first sidelobe -31.5 dB, half-power width 1.44 / band)."""
    f = np.fft.fftfreq(n, 1.0 / fs)
    inside = np.abs(f) <= band / 2.0
    if kind == "flat":
        w = inside.astype(float)
    elif kind == "hann":
        w = np.where(inside, 0.5 * (1.0 + np.cos(2.0 * np.pi * f / band)), 0.0)
    else:
        raise ValueError("band_window: kind is 'flat' or 'hann'")
    return w * (n / w.sum())                 # a single path of amplitude a peaks at a (on a delay bin)


def sound_periodic(y: np.ndarray, period_probe: np.ndarray, fs: float, fc: float, *, start: int, n_periods: int, band: float,
                   window: str = "hann", lead: float = 0.0, regularisation: float = 1e-3, name: str = "") -> TimeVaryingChannel:
    """Impulse responses from a probe of period P = len(period_probe) samples: one per period.

    y: complex envelope of the recording (one receiver [N] or several [N, M]); `start`: sample
    at which a period of the probe begins as received on the first arrival (`find_probe`);
    the delay axis starts `lead` seconds before that. Period n uses y[start + n P - lead ...]
    of length P and nothing else: H = Y X* / (|X|^2 + r max|X|^2) W, W = `band_window`.
    Delay is circular with period P / fs (a longer channel folds)."""
    y = np.asarray(y)
    if y.ndim == 1:
        y = y[:, None]
    x = np.asarray(period_probe)
    p = len(x)
    xf = np.fft.fft(x)
    px = np.abs(xf) ** 2
    g = np.conj(xf) / (px + regularisation * px.max()) * band_window(p, fs, band, window)
    s0 = int(start) - int(round(lead * fs))
    n_periods = int(min(n_periods, (len(y) - s0) // p))
    if s0 < 0 or n_periods < 1:
        raise ValueError("sound_periodic: the probe does not fit in the recording")
    seg = y[s0:s0 + n_periods * p].reshape(n_periods, p, y.shape[1])
    h = np.fft.ifft(np.fft.fft(seg, axis=1) * g[None, :, None], axis=1)
    return TimeVaryingChannel(h=np.ascontiguousarray(h.transpose(0, 2, 1)), fs_time=fs / p, fs_delay=fs, fc=fc,
                              meta={"estimator": "periodic", "period_s": p / fs, "band": band, "window": window, "lead": lead}, name=name)


def sound_sliding(y: np.ndarray, probe: np.ndarray, fs: float, fc: float, *, start: int, block: int, hop: int, n_delay: int, band: float,
                  window: str = "hann", lead: float = 0.0, regularisation: float = 1e-3, name: str = "") -> TimeVaryingChannel:
    """Impulse responses from any known probe, one per `hop` samples, each a least-squares fit
    of `n_delay` taps to `block` samples of the recording:  minimise |y[n] - sum_k h[k] x[n - k]|^2
    over n in the block, with the transmitted x known before the block too (so there is no
    edge effect; what the channel does beyond the delay window is left as noise).
    Normal equations with a ridge `regularisation` x mean diagonal (the probe has no energy
    outside its band, where the fit is otherwise undetermined), then W = `band_window` along delay.
    A snapshot is the channel averaged over block / fs seconds; needs block >= 2 n_delay or so."""
    y = np.asarray(y)
    if y.ndim == 1:
        y = y[:, None]
    x = np.asarray(probe, dtype=np.complex128)
    nlead = int(round(lead * fs))
    w = band_window(n_delay, fs, band, window)
    out = []
    n0 = n_delay                                           # first block start within the probe: all lags known
    xp = np.concatenate([np.zeros(n_delay, dtype=np.complex128), x])
    while n0 + block <= len(x):
        r0 = int(start) - nlead + n0
        if r0 + block > len(y):
            break
        # A[n, k] = x[n0 + n - k]: columns are the probe delayed by k.
        idx = (n0 + np.arange(block))[:, None] - np.arange(n_delay)[None, :] + n_delay
        a = xp[idx]
        r = a.conj().T @ a
        r[np.diag_indices(n_delay)] += regularisation * np.real(np.trace(r)) / n_delay
        hk = np.linalg.solve(r, a.conj().T @ y[r0:r0 + block])
        out.append(np.fft.ifft(np.fft.fft(hk, axis=0) * w[:, None], axis=0).T)
        n0 += hop
    if not out:
        raise ValueError("sound_sliding: the probe does not fit in the recording")
    return TimeVaryingChannel(h=np.array(out), fs_time=fs / hop, fs_delay=fs, fc=fc,
                              meta={"estimator": "sliding", "block_s": block / fs, "band": band, "window": window, "lead": lead}, name=name)


def remove_drift(channel: TimeVaryingChannel, receiver: int = 0, delay_bin: int | None = None, half_width: int = 1) -> tuple[TimeVaryingChannel, dict]:
    """Take out what a clock-rate difference (or a steady range rate) does: the reference
    arrival's phase changes linearly, and every delay slides in proportion. The rate is
    fitted to the unwrapped phase of the reference arrival (default: the strongest bin of
    `receiver`): phase = -2 pi fc tau(t), so tau(t) = tau0 - slope t / (2 pi fc). Every snapshot
    is shifted back in delay (a phase ramp across the band) and the linear phase is removed.
    Returned with {"rate": d tau / dt, "doppler_hz": slope / 2 pi, "phase_residual_rad": rms of
    what the straight line leaves}. Fluctuations about the line (source sway, the medium) are
    NOT removed here: `channel_replay.reference_phase` does that."""
    p = (np.abs(channel.h[:, receiver, :]) ** 2).mean(axis=0)
    b = int(np.argmax(p)) if delay_bin is None else int(delay_bin)
    s = channel.h[:, receiver, max(b - half_width, 0):b + half_width + 1].sum(axis=1)
    t = channel.time
    ph = np.unwrap(np.angle(s))
    slope, icpt = np.polyfit(t, ph, 1)
    rate = -slope / (2.0 * np.pi * channel.fc)            # d tau / dt
    f = np.fft.fftfreq(channel.n_delay, 1.0 / channel.fs_delay)
    shift = rate * (t - t.mean())                         # s: where the arrival is relative to mid-record
    hf = np.fft.fft(channel.h, axis=2) * np.exp(2j * np.pi * f[None, None, :] * shift[:, None, None])
    h = np.fft.ifft(hf, axis=2) * np.exp(-1j * slope * t)[:, None, None]
    out = TimeVaryingChannel(h=h, fs_time=channel.fs_time, fs_delay=channel.fs_delay, fc=channel.fc, meta=dict(channel.meta, drift_rate=rate), name=channel.name)
    return out, {"rate": float(rate), "doppler_hz": float(slope / (2.0 * np.pi)), "phase_residual_rad": float(np.std(ph - slope * t - icpt))}


# =================================================================================================
# Estimators on a sounding
# =================================================================================================

def tail_decay(profile: np.ndarray, pulse: np.ndarray, fs_delay: float, *, floor: float = 0.0, decay_grid=None,
               density=None, steady: np.ndarray | None = None, oversample: int = 8) -> dict:
    """Decay time of the scattered tail behind one arrival, with the instrument's pulse taken out.

    `density(lag s, T)`: the tail's shape for mean excess delay T in place of the exponential
    (e.g. `timespread.facet_delay_density` of a = T / (1 + sin^2 g), b = a sin^2 g); it is
    sampled `oversample` times per bin and summed. `steady`: the steady (coherent) intensity
    profile on the same bins, used in place of `pulse` for the unspread term: a modulation
    common to the arrival and to the echoes that follow it (a heaving mirror, a swaying source)
    fluctuates in proportion to the steady profile, echoes included, and is then not mistaken
    for a tail.

    profile: mean intensity against delay around the arrival (the fluctuating part: total less
    |mean|^2), [K]. pulse: the intensity response of the sounding to a single steady path on
    the same grid, [K], its peak where the arrival's specular delay is. Model:
        profile = A pulse + B (pulse * e)(tau) + floor,   e = exp(-tau / T) / T for tau >= 0 ,
    the scattered taps being independent so that their intensities add. T is searched on a
    grid; A, B >= 0 by least squares with weights 1 / max(profile, 2 % of its peak) (an
    averaged intensity has a multiplicative error). Returned: decay_time_s, the amplitudes
    (`direct` = A, `tail` = B, both as energies: the tail's share of the fluctuating energy is
    tail_share), `resolved` (False when a tail cannot be told from none: the misfit without
    one is within 10 % of the best), and the misfits."""
    y = np.maximum(np.asarray(profile, dtype=float) - floor, 0.0)
    p = np.asarray(pulse, dtype=float)
    p = p / p.sum()
    k = len(y)
    lag = np.arange(k) / fs_delay
    if decay_grid is None:
        decay_grid = np.geomspace(0.15 / fs_delay, 0.5 * k / fs_delay, 80)
    scale = max(y.max(), 1e-300)
    wgt = 1.0 / np.maximum(y, 0.02 * scale)

    def solve(cols):
        a = np.stack(cols, axis=1) * wgt[:, None]
        coef, *_ = np.linalg.lstsq(a, y * wgt, rcond=None)
        if np.any(coef < 0.0):                               # non-negative: keep the single best column
            best = (np.inf, np.zeros(len(cols)))
            for j in range(len(cols)):
                cw = cols[j] * wgt
                c1 = max(float(cw @ (y * wgt) / max(cw @ cw, 1e-300)), 0.0)
                r = float(((cw * c1 - y * wgt) ** 2).sum())
                if r < best[0]:
                    cc = np.zeros(len(cols))
                    cc[j] = c1
                    best = (r, cc)
            return best
        return float((((a @ coef) - y * wgt) ** 2).sum()), coef
    first = p if steady is None else np.asarray(steady, dtype=float) / max(float(np.sum(steady)), 1e-300)
    fine = (np.arange(k * oversample) + 0.5) / (oversample * fs_delay)
    r0, _ = solve([first])
    best = None
    for t in decay_grid:
        e = np.exp(-fine / t) if density is None else np.asarray(density(fine, t), dtype=float)
        e = e.reshape(k, oversample).sum(axis=1)
        e = e / e.sum()
        conv = np.convolve(p, e)[:k]
        r, c = solve([first, conv])
        if best is None or r < best[0]:
            best = (r, float(t), c)
    r, t, c = best
    total = max(c.sum(), 1e-300)
    return {"decay_time_s": t, "direct": float(c[0]), "tail": float(c[1]), "tail_share": float(c[1] / total),
            "misfit": r, "misfit_no_tail": r0, "resolved": bool(c[1] > 0.0 and r < 0.9 * r0)}


def arrival_split(channel: TimeVaryingChannel, receiver: int, bins: slice, *, reference: np.ndarray | None = None, floor: float = 0.0,
                  detrend: float = 0.0) -> dict:
    """Power of an arrival (delay bins `bins`) split into steady and fluctuating parts.

    The snapshots are divided by `reference` (unit phasors, `channel_replay.reference_phase`).
    `detrend` > 0: the steady part is the mean over a sliding window of that many seconds
    (slow changes of the arrival — tide, drifting moorings — are then not counted as
    scattering; fluctuations slower than about 1 / detrend Hz are lost with them). `floor`:
    the sounding's noise power per delay bin, removed from the total and, as floor / (snapshots
    in the mean), from the steady part. Returned: total, coherent, coherent_fraction, its
    standard error from four quarters of the record, and the fluctuating series [T, bins]."""
    h = channel.h[:, receiver, bins]
    if reference is not None:
        h = h * np.conj(reference[:, receiver])[:, None]
    nb = h.shape[1]

    def one(x):
        total = float((np.abs(x) ** 2).mean(axis=0).sum() - nb * floor)
        if detrend > 0.0 and len(x) > 2:
            n = max(2, min(int(round(detrend * channel.fs_time)), len(x)))
            c = np.cumsum(np.concatenate([np.zeros((1, nb)), x], axis=0), axis=0)
            mean = (c[n:] - c[:-n]) / n
            # A window of n snapshots keeps 1/n of the fluctuating power in its mean: take it out.
            raw = float((np.abs(mean) ** 2).mean(axis=0).sum() - nb * floor / n)
            coh = (raw - total / n) / (1.0 - 1.0 / n)
            before = n // 2
            pad = np.concatenate([np.repeat(mean[:1], before, axis=0), mean, np.repeat(mean[-1:], len(x) - len(mean) - before, axis=0)])
            return total, coh, x - pad
        mean = x.mean(axis=0)
        return total, float((np.abs(mean) ** 2).sum() - nb * floor / len(x)), x - mean[None, :]
    total, coh, fluct = one(h)
    q = len(h) // 4
    parts = [one(h[i * q:(i + 1) * q])[:2] for i in range(4)] if q >= 4 else []
    fr = [c / t for t, c in parts if t > 0.0]
    return {"total": total, "coherent": coh, "coherent_fraction": coh / total if total > 0.0 else float("nan"),
            "coherent_fraction_se": float(np.std(fr) / 2.0) if len(fr) == 4 else float("nan"), "fluctuating": fluct}
