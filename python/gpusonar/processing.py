"""Processing steps that sit around the kernels: Doppler hypothesis bank, scan conversion."""

from __future__ import annotations

import numpy as np

from . import _gpusonar as core


def doppler_scale(range_rate: float, sound_speed: float) -> float:
    """Echo time-scale for a two-way path whose length changes at 2 * range_rate.

    Positive range_rate means opening (the echo is stretched, scale < 1).
    """
    return (sound_speed - range_rate) / (sound_speed + range_rate)


def doppler_bank(record: "core.PingRecord", range_rates, window: str = "none", backend=None):
    """Doppler-tolerant pulse compression for long pulses on moving platforms or targets.

    Compresses a baseband record once per range-rate hypothesis (m/s, positive opening)
    and keeps, for the whole record, the hypothesis with the highest compressed peak.
    Returns (compressed record, best range rate, peak per hypothesis).
    `backend` (core.Backend) is where each compression runs; the host when not given.
    """
    where = {} if backend is None else {"backend": backend}
    c = record.environment.sound_speed
    best, best_peak, peaks = None, -1.0, []
    for v in range_rates:
        trial = record.copy()
        core.pulse_compress(trial, window=window, doppler_scale=doppler_scale(float(v), c), **where)
        peak = float(np.abs(trial.samples.array).max())
        peaks.append(peak)
        if peak > best_peak:
            best, best_peak, best_v = trial, peak, float(v)
    return best, best_v, np.asarray(peaks)


def scan_convert(image: np.ndarray, ranges, bearings_rad, nx: int = 512, ny: int = 512,
                 fill: float = 0.0):
    """Resample a range x bearing image onto a Cartesian grid for display.

    x is across (range * sin bearing), y is forward (range * cos bearing). Bilinear
    interpolation in (range, bearing); points outside the fan get `fill`.
    Returns (cartesian image [ny, nx], x axis, y axis).
    """
    ranges = np.asarray(ranges, dtype=float)
    bearings = np.asarray(bearings_rad, dtype=float)
    r_max = ranges[-1]
    x = np.linspace(r_max * np.sin(bearings[0]), r_max * np.sin(bearings[-1]), nx)
    y = np.linspace(min(0.0, ranges[0] * np.cos(np.abs(bearings).max())), r_max, ny)
    X, Y = np.meshgrid(x, y)
    R = np.hypot(X, Y)
    B = np.arctan2(X, Y)
    # Fractional indices (axes need not be uniform).
    fr = np.interp(R, ranges, np.arange(len(ranges)), left=-1, right=-1)
    fb = np.interp(B, bearings, np.arange(len(bearings)), left=-1, right=-1)
    ok = (fr >= 0) & (fb >= 0)
    i0 = np.clip(np.floor(fr).astype(int), 0, len(ranges) - 2)
    j0 = np.clip(np.floor(fb).astype(int), 0, len(bearings) - 2)
    wr = np.clip(fr - i0, 0.0, 1.0)
    wb = np.clip(fb - j0, 0.0, 1.0)
    img = np.asarray(image)
    out = ((1 - wr) * (1 - wb) * img[i0, j0] + wr * (1 - wb) * img[i0 + 1, j0]
           + (1 - wr) * wb * img[i0, j0 + 1] + wr * wb * img[i0 + 1, j0 + 1])
    out = np.where(ok, out, fill)
    return out, x, y
