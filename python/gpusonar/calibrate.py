"""Array calibration from known sources: per-channel gain, phase and position errors.

Each calibration ping holds one strong echo (or source) at a known position. Sampling
every channel at the delay the nominal geometry predicts and restoring the carrier gives
one complex number per channel and ping. Without errors those are equal across channels.
With errors,

    v[c, k] = A[k] * g[c] * exp(j (phi[c] + (2 pi / lambda) u[k] . dp[c]))

for gain g, phase phi and position error dp of channel c, and unit vector u from the array
to source k. Gains come from magnitudes; phi and dp from a linear least-squares fit over
sources at different bearings (three or more for a 2-D position error).

What cannot be seen, and is fixed by convention: the mean gain (set to 1), the mean phase
(0) and the mean position error (0) of the array. They are common to every channel and
do not affect beamforming.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import _gpusonar as core
from .recon.fls import array_frame, prepare


@dataclass
class ChannelErrors:
    gain: np.ndarray          # [channels], mean 1
    phase: np.ndarray         # [channels] rad, mean 0
    position: np.ndarray      # [channels, 3] body frame, m, mean 0
    residual_rms_deg: float   # phase left unexplained by the fit


def channel_responses(record, source_position) -> np.ndarray:
    """Complex response of each channel to a source at a known position: the channel's
    compressed samples at the nominal two-way delay, with the carrier restored."""
    prepare([record])
    x = record.samples.array
    s = record.samples
    c = record.environment.sound_speed
    tx = np.array(record.tx_position())
    rx = record.rx_positions()
    src = np.asarray(source_position, dtype=float)
    tau = (np.linalg.norm(src - tx) + np.linalg.norm(src - rx, axis=1)) / c
    # Exact band-limited value at the (fractional) delay: evaluate each channel's Fourier
    # series there. Linear interpolation would cost up to a few percent of amplitude,
    # differently per channel, and read as gain error.
    n = x.shape[1]
    f = np.fft.fftfreq(n, 1.0 / s.sample_rate)
    spectrum = np.fft.fft(x, axis=1)
    v = (spectrum * np.exp(2j * np.pi * f[None, :] * (tau - s.start_time)[:, None])).sum(axis=1) / n
    return v * np.exp(2j * np.pi * s.center_frequency * tau)


def estimate_channel_errors(records, source_positions, planar: bool = True) -> ChannelErrors:
    """Estimate gain, phase and element position errors from calibration pings.

    `planar` restricts position errors to the plane of the array axis and the boresight
    (two unknowns per channel), which is what sources in that plane can observe.
    """
    records = list(records)
    if len(records) != len(source_positions):
        raise ValueError("one source position per calibration ping is required")
    rec0 = records[0]
    origin, forward, across, normal = array_frame(rec0)
    rot = core.rotation_ned(*rec0.geometry.rx_pose.rpy)
    wavelength = rec0.environment.sound_speed / rec0.transmit.waveform.center()
    v = np.array([channel_responses(r, p) for r, p in zip(records, source_positions)])   # [K, C]
    n_src, n_ch = v.shape

    mag = np.abs(v) / np.median(np.abs(v), axis=1, keepdims=True)
    gain = np.median(mag, axis=0)
    gain /= gain.mean()

    # Phase of each channel relative to the array's coherent mean for that ping.
    psi = np.angle(v * np.conj(v.mean(axis=1, keepdims=True)))
    basis = [across, forward] if planar else [across, forward, normal]
    units = []
    for p in source_positions:
        d = np.asarray(p, dtype=float) - origin
        units.append(d / np.linalg.norm(d))
    units = np.array(units)                                              # [K, 3]
    k_wave = 2.0 * np.pi / wavelength
    # A source further along u is reached sooner by an element displaced along u: the
    # two-way path shortens by u . dp (receive leg only), so the restored phase rises.
    design = np.column_stack([np.ones(n_src)] + [k_wave * (units @ b) for b in basis])   # [K, 1 + dims]
    if n_src < design.shape[1]:
        raise ValueError(f"need at least {design.shape[1]} sources at different bearings")
    coeffs, *_ = np.linalg.lstsq(design, psi, rcond=None)               # [1 + dims, C]
    phase = coeffs[0] - coeffs[0].mean()
    world = sum(np.outer(coeffs[1 + i], b) for i, b in enumerate(basis))  # [C, 3] world frame
    world -= world.mean(axis=0)
    position = world @ rot                                               # world -> body (R^T applied to rows)
    residual = psi - design @ coeffs
    return ChannelErrors(gain, phase, position, float(np.degrees(np.sqrt(np.mean(residual**2)))))


def store_calibration(record, errors: ChannelErrors) -> None:
    """Write estimated errors into a record's calibration tables. `core.apply_calibration`
    then removes gain and phase; the kernel uses the position offsets directly."""
    cal = record.calibration
    cal.gain = [float(g) for g in errors.gain]
    cal.phase = [float(p) for p in errors.phase]
    cal.position_offset = [tuple(float(x) for x in p) for p in errors.position]
    record.calibration = cal
