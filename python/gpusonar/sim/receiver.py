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
