"""Receive chain effects applied after echo synthesis: noise and the ADC."""

from __future__ import annotations

import numpy as np

from .. import _gpusonar as core


def add_noise(samples: np.ndarray, rms: float, rng: np.random.Generator) -> np.ndarray:
    """Add white Gaussian noise of the given RMS (per real component for complex data)."""
    if rms <= 0.0:
        return samples
    if np.iscomplexobj(samples):
        noise = rng.standard_normal(samples.shape) + 1j * rng.standard_normal(samples.shape)
        return (samples + rms * noise).astype(np.complex64)
    return (samples + rms * rng.standard_normal(samples.shape)).astype(np.float32)


def apply_hardware_tvg(record: "core.PingRecord", samples: np.ndarray) -> np.ndarray:
    """Apply the analogue gain ramp described by the record (piecewise linear in dB)."""
    t_knots, g_knots = record.metadata.tvg_time, record.metadata.tvg_gain_db
    if len(t_knots) == 0:
        return samples
    s = record.samples
    t = s.start_time + np.arange(s.n_samples) / s.sample_rate
    gain = 10.0 ** (np.interp(t, t_knots, g_knots) / 20.0)
    return (samples * gain).astype(samples.dtype)


def quantise(samples: np.ndarray, bits: int, full_scale: float) -> tuple[np.ndarray, float]:
    """Mid-tread ADC: round to 2^bits levels over [-full_scale, +full_scale) and clip.

    Returns the quantised samples (still in physical units) and the step size.
    """
    step = full_scale / 2.0 ** (bits - 1)
    lo, hi = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1

    def q(x):
        return np.clip(np.rint(x / step), lo, hi) * step

    if np.iscomplexobj(samples):
        return (q(samples.real) + 1j * q(samples.imag)).astype(np.complex64), step
    return q(samples).astype(np.float32), step


def apply_adc(record: "core.PingRecord", samples: np.ndarray) -> np.ndarray:
    """Quantise per the record's ADC description and set its integer storage to match."""
    bits = record.metadata.adc_bits
    full_scale = record.metadata.adc_full_scale
    if bits <= 0 or not np.isfinite(full_scale):
        return samples
    out, step = quantise(samples, bits, full_scale)
    s = record.samples
    s.storage = core.StorageType.Int16 if bits <= 16 else core.StorageType.Int32
    s.storage_scale = step
    record.samples = s
    return out


def sync_chirp(t, f_start: float, f_stop: float, duration: float) -> np.ndarray:
    """The shared reference burst at times `t` after its start: a Hann-windowed linear chirp."""
    t = np.asarray(t, dtype=float)
    inside = (t >= 0.0) & (t <= duration)
    k = (f_stop - f_start) / duration
    tt = np.where(inside, t, 0.0)
    return np.where(inside, np.sin(np.pi * tt / duration) ** 2
                    * np.cos(2.0 * np.pi * (f_start * tt + 0.5 * k * tt**2)), 0.0)


def reference_times(record: "core.PingRecord", ref: dict) -> tuple[float, float]:
    """True times (after transmit) of the two reference bursts for this record's window."""
    s = record.samples
    t_end = s.start_time + s.n_samples / s.sample_rate
    return s.start_time + float(ref["lead"]), t_end - float(ref["tail"]) - float(ref["duration"])


def add_reference_signal(record: "core.PingRecord", sonar, clocks: dict) -> None:
    """Sum the shared synchronisation reference into each board's reference input, as that
    board's own (wrong) sample clock sees it."""
    ref = sonar.config["receive"].get("reference")
    if not ref or ref.get("mode") != "sync_chirp" or record.samples.is_complex():
        return
    s = record.samples
    x = s.array
    t_a, t_b = reference_times(record, ref)
    n = np.arange(s.n_samples)
    for board, channel in enumerate(ref["channels"]):
        if channel not in record.metadata.channel_element:
            continue
        row = record.metadata.channel_element.index(channel)
        if ref.get("dedicated"):
            x[row] = 0.0                      # a reference input has no hydrophone on it
        offset = clocks["offset"][board]
        drift = clocks["drift_ppm"][board] * 1e-6
        t_true = s.start_time + offset + n / (s.sample_rate * (1.0 + drift))
        for t0 in (t_a, t_b):
            x[row] += float(ref["level"]) * sync_chirp(t_true - t0, ref["f_start"], ref["f_stop"],
                                                       ref["duration"]).astype(np.float32)
    s.array = x
    record.samples = s


def apply_channel_errors(record: "core.PingRecord", errors: dict) -> None:
    """Impose per-channel gain and phase errors on the samples (the truth a calibration has
    to find), then leave the record's calibration tables neutral: the processor starts
    without them. Position errors were already used by the synthesis."""
    s = record.samples
    x = s.array
    n_ch = x.shape[0]
    gain = np.asarray(errors.get("gain", np.ones(n_ch)), dtype=float)
    phase = np.asarray(errors.get("phase", np.zeros(n_ch)), dtype=float)
    factor = gain * np.exp(1j * phase)
    kind = s.kind
    if np.iscomplexobj(x):
        s.array = (x * factor[:, None]).astype(np.complex64)
    else:
        # A phase shift of a real signal: rotate its analytic signal and take the real part.
        spectrum = np.fft.rfft(x, axis=1)
        s.array = np.fft.irfft(spectrum * factor[:, None], n=x.shape[1], axis=1).astype(np.float32)
    s.kind = kind
    record.samples = s
    cal = record.calibration
    cal.position_offset = [(0.0, 0.0, 0.0)] * n_ch
    record.calibration = cal
