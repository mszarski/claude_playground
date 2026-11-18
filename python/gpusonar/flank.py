"""Flank-array (long-range, single-ping) processing: range x bearing maps.

Two beamformers that must agree where both apply:

- backprojection through the shared kernel (exact, near or far field);
- a far-field conventional beamformer in the frequency domain (plane-wave steering),
  which is what backprojection reduces to at long range and is cheaper per ping.

Plus the adaptive (MVDR) beamformer, array pattern prediction for arbitrary element
positions, and transmit blanking.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import _gpusonar as core
from .recon.fls import array_frame, aperture_weights, fan_image, prepare


@dataclass
class BeamMap:
    data: np.ndarray        # [n_bearing, n_range_or_time] complex
    bearings: np.ndarray    # rad, from the boresight towards increasing element index
    times: np.ndarray       # two-way time of each column, s
    ranges: np.ndarray      # c * t / 2, m
    n_valid: int = 0        # columns with a fully overlapped replica; later ones are not usable

    def valid(self) -> "BeamMap":
        """The map without the tail where the matched filter only partly overlaps the data
        (the last pulse length of the record). Detection must not run on that tail."""
        n = self.n_valid or self.data.shape[1]
        return BeamMap(self.data[:, :n], self.bearings, self.times[:n], self.ranges[:n], n)


def _n_valid(record) -> int:
    s = record.samples
    return max(int(s.n_samples - np.ceil(record.transmit.waveform.duration * s.sample_rate)), 1)


def _element_offsets(record):
    """Element positions relative to the array centre, and the array frame."""
    origin, forward, across, _ = array_frame(record)
    return record.rx_positions() - origin, origin, forward, across


def steering_delays(record, bearings_rad) -> np.ndarray:
    """[n_bearing, n_elements] plane-wave delays p_n . u / c: positive where the element
    is nearer the source and hears it earlier."""
    offsets, _, forward, across = _element_offsets(record)
    b = np.asarray(bearings_rad, dtype=float)
    u = np.cos(b)[:, None] * forward + np.sin(b)[:, None] * across
    return (u @ offsets.T) / record.environment.sound_speed


def _beamform_engine(engine, name: str) -> str:
    """Which implementation of a far-field beamformer runs: "cpp" (libs/live beamform, on the
    host threads or on the CUDA device) when the extension has it, else "numpy". The NumPy
    code below is the reference the C++ one is tested against."""
    if engine not in (None, "cpp", "numpy"):
        raise ValueError(f"unknown beamformer engine {engine!r}: use 'cpp' or 'numpy'")
    built = hasattr(core, name)
    if engine == "cpp" and not built:
        raise RuntimeError(f"this build of the extension has no {name}")
    return "cpp" if (engine or ("cpp" if built else "numpy")) == "cpp" else "numpy"


def _cpp_backend(backend) -> dict:
    """Keyword arguments that hand a backend to a C++ beamformer (builds before the beamformers
    had backends take none and run on the host)."""
    if backend is None or not getattr(core, "BEAMFORM_BACKENDS", False):
        return {}
    return {"backend": backend}


def farfield_beamform(record, bearings_rad, window: str = "rect", engine: str | None = None, backend=None) -> BeamMap:
    """Conventional far-field beamformer on a prepared (baseband, compressed) record.

    B(u, f) = sum_n w_n X_n(f) exp(-j 2 pi (f + fc) p_n . u / c): each channel is delayed
    by its plane-wave lead, exactly (a phase ramp over frequency), and summed. The output
    time axis is the two-way delay to the array centre.

    `engine`: "cpp" (the default when the extension has it) runs the C++ beamformer, on the
    host threads or, with `backend=core.Backend.Cuda`, on the device; "numpy" runs the
    reference code below (host only). They agree to single-precision rounding.
    """
    if _beamform_engine(engine, "farfield_beamform_cpp") == "cpp" and window in ("rect", "hann", "hamming"):
        return BeamMap(**core.farfield_beamform_cpp(record, [float(b) for b in np.asarray(bearings_rad, dtype=float)],
                                                    window, **_cpp_backend(backend)))
    prepare([record])
    x = record.samples.array
    fs, fc = record.samples.sample_rate, record.samples.center_frequency
    n = x.shape[1]
    X = np.fft.fft(x, axis=1)
    f = np.fft.fftfreq(n, 1.0 / fs) + fc
    w = aperture_weights(record, window)
    tau = steering_delays(record, bearings_rad)                  # [B, N]
    out = np.empty((len(tau), n), dtype=np.complex64)
    for i, t_row in enumerate(tau):
        steer = np.exp(-2j * np.pi * f[None, :] * t_row[:, None])     # [N, n]
        out[i] = np.fft.ifft((w[:, None] * X * steer).sum(axis=0))
    times = record.samples.start_time + np.arange(n) / fs
    return BeamMap(out, np.asarray(bearings_rad, dtype=float), times,
                   0.5 * record.environment.sound_speed * times, _n_valid(record))


def backprojection_map(record, ranges, bearings_rad, window: str = "rect", **kw):
    """Range x bearing map by backprojection (the fan is in the array's horizontal plane)."""
    return fan_image([record], ranges, bearings_rad, window=window, **kw)


def sample_beam_map(bm: BeamMap, bearing_index: int, times) -> np.ndarray:
    """Beam output at arbitrary two-way times, by linear interpolation of the complex series."""
    t = np.asarray(times, dtype=float)
    pos = (t - bm.times[0]) / (bm.times[1] - bm.times[0])
    i = np.clip(np.floor(pos).astype(int), 0, bm.data.shape[1] - 2)
    w = pos - i
    row = bm.data[bearing_index]
    return (1 - w) * row[i] + w * row[i + 1]


def blank_transmit(record, guard: float = 0.0, taper: float = 0.0) -> None:
    """Zero the samples during the transmit pulse and for `guard` seconds after it, with an
    optional raised-cosine `taper` back to full weight. For raw (uncompressed) records the
    direct pulse occupies [0, duration]; apply before pulse compression."""
    s = record.samples
    t = s.start_time + np.arange(s.n_samples) / s.sample_rate
    end = record.transmit.waveform.duration + guard
    w = np.ones(s.n_samples)
    w[t < end] = 0.0
    if taper > 0.0:
        ramp = (t >= end) & (t < end + taper)
        w[ramp] = 0.5 - 0.5 * np.cos(np.pi * (t[ramp] - end) / taper)
    kind = s.kind
    s.array = (s.array * w[None, :]).astype(s.array.dtype)
    s.kind = kind
    record.samples = s


def mvdr_beamform(record, bearings_rad, n_subbands: int = 32, loading: float = 3e-2,
                  block: int = 256, guard_blocks: int = 1, engine: str | None = None, backend=None) -> BeamMap:
    """Adaptive (minimum-variance distortionless-response) beamformer.

    The band is split into `n_subbands` sub-bands, narrow against c / aperture so that one
    steering vector describes the array response in each. In every sub-band and every
    range block, the element covariance R is estimated from the *other* range blocks (all
    but the block itself and `guard_blocks` either side), and the weights
    w = R^-1 a / (a^H R^-1 a) pass the look direction undistorted while minimising what
    that training data contains: directional noise, jammers, reverberation that extends in
    range. Sub-band outputs are summed, which restores range resolution.

    The block's own samples are left out of R on purpose. With a strong echo in the
    training data, MVDR treats the small mismatch between the echo and its steering vector
    as interference and cancels the echo itself.

    `loading` is diagonal loading as a fraction of the mean element power in the training
    data: it limits noise gain and makes the weights tolerant of calibration errors.

    The output is scaled by the number of elements so that a target reads the same as in
    the conventional sum beamformer and in backprojection (which have a gain of N).
    Costs to know about: each sub-band has only about n_samples / n_subbands independent
    snapshots to estimate an N x N covariance from, so in plain white noise MVDR is a few
    dB worse than the conventional beamformer; more sub-bands model the array better but
    leave fewer snapshots each.

    `engine` and `backend` as in `farfield_beamform`: the C++ beamformer by default (host
    threads, or the CUDA device), the NumPy code below as the reference.
    """
    if _beamform_engine(engine, "mvdr_beamform_cpp") == "cpp":
        try:
            return BeamMap(**core.mvdr_beamform_cpp(record, [float(b) for b in np.asarray(bearings_rad, dtype=float)],
                                                    n_subbands=int(n_subbands), loading=float(loading), block=int(block),
                                                    guard_blocks=int(guard_blocks), **_cpp_backend(backend)))
        except ValueError as e:                                  # the same refusals, in the words of this function
            raise ValueError(str(e).replace("mvdr_beamform: ", "")) from None
        except RuntimeError as e:
            if "singular" not in str(e):
                raise
            raise np.linalg.LinAlgError(str(e).replace("mvdr_beamform: ", "")) from None   # what NumPy's solve raises
    prepare([record])
    x = record.samples.array
    fs, fc = record.samples.sample_rate, record.samples.center_frequency
    n_el, n = x.shape
    X = np.fft.fft(x, axis=1)
    f_bb = np.fft.fftfreq(n, 1.0 / fs)
    w = record.transmit.waveform
    lo, hi = min(w.f_start, w.f_stop) - fc, max(w.f_start, w.f_stop) - fc
    edges = np.linspace(lo, hi, n_subbands + 1)
    tau = steering_delays(record, bearings_rad)                  # [B, N]
    n_blocks = int(np.ceil(n / block))
    if n_blocks < 2 * guard_blocks + 3:
        raise ValueError("record too short for this block size: no training data would be left")
    out = np.zeros((len(tau), n), dtype=np.complex128)
    eye = np.eye(n_el)
    for k in range(n_subbands):
        sel = (f_bb >= edges[k]) & (f_bb < edges[k + 1])
        if not sel.any():
            continue
        xk = np.fft.ifft(np.where(sel[None, :], X, 0.0), axis=1)          # [N, n]
        f_abs = fc + 0.5 * (edges[k] + edges[k + 1])
        a = np.exp(2j * np.pi * f_abs * tau)                              # [B, N] steering vectors
        blocks = [xk[:, i * block:(i + 1) * block] for i in range(n_blocks)]
        outer = [b @ b.conj().T for b in blocks]
        counts = [b.shape[1] for b in blocks]
        total, total_count = sum(outer), sum(counts)
        for i in range(n_blocks):
            j0, j1 = max(i - guard_blocks, 0), min(i + guard_blocks + 1, n_blocks)
            r = (total - sum(outer[j0:j1])) / (total_count - sum(counts[j0:j1]))
            r = r + loading * np.trace(r).real / n_el * eye
            ra = np.linalg.solve(r, a.T).T                                # rows: R^-1 a
            weights = ra / np.einsum("bn,bn->b", a.conj(), ra)[:, None]
            out[:, i * block:(i + 1) * block] += n_el * (weights.conj() @ blocks[i])
    times = record.samples.start_time + np.arange(n) / fs
    return BeamMap(out.astype(np.complex64), np.asarray(bearings_rad, dtype=float), times,
                   0.5 * record.environment.sound_speed * times, _n_valid(record))


def add_jammer(record, bearing_rad: float, rms: float, rng: np.random.Generator) -> None:
    """Add a far-field broadband noise source to a raw real-passband record: the same white
    noise on every channel, each with its plane-wave lead. For tests and scenarios."""
    s = record.samples
    if s.is_complex():
        raise ValueError("add_jammer works on raw real-passband records")
    x = s.array
    n = x.shape[1]
    noise = np.fft.rfft(rng.standard_normal(n) * rms)
    f = np.fft.rfftfreq(n, 1.0 / s.sample_rate)
    lead = steering_delays(record, [bearing_rad])[0]              # [N]
    jam = np.fft.irfft(noise[None, :] * np.exp(2j * np.pi * f[None, :] * lead[:, None]), n=n, axis=1)
    s.array = (x + jam).astype(np.float32)
    record.samples = s


# --- array patterns for arbitrary layouts ------------------------------------------------------

def array_pattern(positions_along_axis, frequency: float, sin_bearings, steer_sin: float = 0.0,
                  sound_speed: float = 1500.0, weights=None) -> np.ndarray:
    """Narrowband array factor magnitude (1 at the steering direction) of a line array."""
    x = np.asarray(positions_along_axis, dtype=float)
    w = np.ones_like(x) if weights is None else np.asarray(weights, dtype=float)
    k = 2.0 * np.pi * frequency / sound_speed
    u = np.asarray(sin_bearings, dtype=float)
    af = (w[None, :] * np.exp(1j * k * x[None, :] * (u[:, None] - steer_sin))).sum(axis=1)
    return np.abs(af) / w.sum()


def beamwidth_theory(aperture_length: float, frequency: float, sound_speed: float = 1500.0) -> float:
    """-3 dB width (rad, at broadside) of a uniformly weighted filled line aperture."""
    return 0.886 * (sound_speed / frequency) / aperture_length


def ambiguity_report(positions_along_axis, f_low: float, f_high: float, sound_speed: float = 1500.0,
                     steer_deg: float = 0.0, n_freq: int = 33, n_angle: int = 7201) -> dict:
    """Ambiguity pattern of a layout over a band: where a source could be mistaken to be.

    Evaluates the array factor over sin(bearing) at `n_freq` frequencies. Reports the
    main-lobe width, and the highest lobe outside the main lobe both for a single
    frequency (worst case in the band) and for the band-averaged power pattern (what a
    wideband pulse sees: grating lobes sit at different angles for different frequencies
    and smear out).
    """
    x = np.asarray(positions_along_axis, dtype=float)
    u = np.linspace(-1.0, 1.0, n_angle)
    us = np.sin(np.radians(steer_deg))
    freqs = np.linspace(f_low, f_high, n_freq)
    patterns = np.array([array_pattern(x, f, u, us, sound_speed) for f in freqs])
    wide = np.sqrt((patterns**2).mean(axis=0))

    def main_lobe_edges(p):
        i0 = int(np.argmin(np.abs(u - us)))
        lo = i0
        while lo > 0 and p[lo - 1] < p[lo]:
            lo -= 1
        hi = i0
        while hi < len(p) - 1 and p[hi + 1] < p[hi]:
            hi += 1
        return lo, hi

    def outside_peak(p):
        lo, hi = main_lobe_edges(p)
        side = np.concatenate([p[:lo], p[hi + 1:]])
        us_side = np.concatenate([u[:lo], u[hi + 1:]])
        j = int(np.argmax(side))
        return float(20 * np.log10(max(side[j], 1e-12))), float(np.degrees(np.arcsin(us_side[j])))

    def width3(p):
        i0 = int(np.argmin(np.abs(u - us)))
        thr = p[i0] / np.sqrt(2.0)
        lo = i0
        while lo > 0 and p[lo] >= thr:
            lo -= 1
        hi = i0
        while hi < len(p) - 1 and p[hi] >= thr:
            hi += 1
        return float(np.degrees(np.arcsin(u[hi]) - np.arcsin(u[lo])))

    single = [outside_peak(p) for p in patterns]
    worst = max(single, key=lambda s: s[0])
    wide_level, wide_angle = outside_peak(wide)
    pitch = np.diff(np.sort(x))
    return {
        "aperture": float(x.max() - x.min()),
        "min_pitch": float(pitch.min()),
        "beamwidth_deg_center": width3(patterns[n_freq // 2]),
        "narrowband_worst_lobe_db": worst[0],
        "narrowband_worst_lobe_deg": worst[1],
        "wideband_lobe_db": wide_level,
        "wideband_lobe_deg": wide_angle,
        # A uniform array is free of grating lobes for all steering angles when pitch <= lambda / 2.
        "grating_free_frequency": float(sound_speed / (2.0 * pitch.min())),
        "sin_bearing": u,
        "wideband_pattern": wide,
    }
