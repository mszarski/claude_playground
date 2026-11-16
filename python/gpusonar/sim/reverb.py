"""Reverberation by the scattering-function method (after SST; docs/PHYSICS_NOTES.md).

For long ranges, placing individual scatterers over kilometres of boundary is too
expensive. Instead the reverberation *intensity* is computed against time and arrival
direction, and a Gaussian time series with that structure is generated for the array:

1. The boundary is divided into range rings. For each ring the eigenray provider gives
   the one-way paths (delay, amplitude, grazing angle) from the sonar.
2. Every pair of outbound and return paths contributes power at time tau_out + tau_back:
   |a_out|^2 |a_back|^2 x scattering strength x ring area. Pairs add incoherently, as in
   SST. This is the reverberation power against time, per unit azimuth.
3. For each azimuth sector an independent complex Gaussian sequence with that
   time-varying variance is drawn: one random "cell" per time sample and direction.
4. Each hydrophone receives every sector's sequence with its plane-wave lead for that
   direction (an exact delay, applied in frequency), and the result is convolved with the
   transmitted pulse.

The output is raw (uncompressed) reverberation, complex baseband or real passband to
match the record, added to the record, so it goes through pulse compression and
beamforming like any echo and keeps the right coherence across the array.

Only paths travelling towards the boundary count as incident (a receiver placed on a
Bellhop boundary also sees the reflected arrival; Lambert's law is defined on the incident
intensity).

Limits: the projector is taken as omnidirectional in azimuth, monostatic (one provider call from the array centre serves both legs), the
platform is treated as stationary (no Doppler spread), the arrival elevation is ignored
when steering to the hydrophones (fine at the low grazing angles of long ranges), and the
boundary is flat and uniform. Each ping draws new noise, so this method cannot support
synthetic aperture processing; the point-scatterer method (scene.seafloor_patch) is the
one for imaging modes.
"""

from __future__ import annotations

import numpy as np

from .. import _gpusonar as core
from .propagation import EigenrayProvider


def baseband_pulse(waveform, sample_rate: float, center_frequency: float) -> np.ndarray:
    """The transmitted pulse as complex baseband samples (unit amplitude)."""
    n = int(np.floor(waveform.duration * sample_rate)) + 1
    u = np.arange(n) / sample_rate
    t, f0, f1 = waveform.duration, waveform.f_start, waveform.f_stop
    alpha = waveform.window_alpha
    x = u / t
    env = np.ones(n)
    if alpha > 0:
        edge = 0.5 * alpha
        env = np.where(x < edge, 0.5 * (1 - np.cos(np.pi * x / edge)),
                       np.where(x > 1 - edge, 0.5 * (1 - np.cos(np.pi * (1 - x) / edge)), 1.0))
    if waveform.type == core.WaveformType.Cw:
        phase = f0 * u
    elif waveform.type == core.WaveformType.Lfm:
        phase = (f0 + 0.5 * (f1 - f0) / t * u) * u
    elif waveform.type == core.WaveformType.Gaussian:
        sigma = np.sqrt(2 * np.log(2)) / (np.pi * (f1 - f0))
        env = np.exp(-0.5 * ((u - 0.5 * t) / sigma) ** 2)
        phase = 0.5 * (f0 + f1) * (u - 0.5 * t)
    else:
        raise ValueError("unsupported waveform for reverberation synthesis")
    return env * np.exp(2j * np.pi * (phase - center_frequency * u))


def reverberation_power(provider: EigenrayProvider, source, boundary_z: float, times: np.ndarray,
                        mu_db: float, absorption_db_per_m: float = 0.0, max_range: float | None = None,
                        ring_width: float | None = None, min_range: float = 0.0,
                        strength=None) -> np.ndarray:
    """Reverberation power per radian of azimuth in each time bin, for unit source
    amplitude at 1 m, from a flat boundary at depth `boundary_z`.

    The scattering strength is Lambert's law with parameter `mu_db`, or `strength`: a
    function of the incident and scattered grazing angles (radians) returning the linear
    scattering strength per unit area.

    `times` are the centres of uniform time bins (two-way time after transmit).
    """
    source = np.asarray(source, dtype=float)
    dt = times[1] - times[0]
    c = provider.sound_speed
    r_max = max_range or 0.5 * c * (times[-1] + dt)
    dr = ring_width or 0.25 * c * dt                       # two rings per time bin at grazing incidence
    rings = np.arange(min_range + 0.5 * dr, r_max, dr)
    power = np.zeros(len(times))
    if len(rings) == 0:
        return power
    pts = np.column_stack([source[0] + rings, np.full_like(rings, source[1]), np.full_like(rings, boundary_z)])
    es = provider.eigenrays(source, pts)
    if strength is None:
        mu = 10.0 ** (mu_db / 10.0)
        strength = lambda g_in, g_out: mu * np.sin(g_in) * np.sin(g_out)     # noqa: E731
    side = 1.0 if boundary_z >= source[2] else -1.0        # incident paths travel towards the boundary
    incident = es.valid & (es.arrival[:, :, 2] * side > 0.0)
    graze = np.arcsin(np.clip(np.abs(es.arrival[:, :, 2]), 0.0, 1.0))   # grazing angle at the boundary
    a2 = np.where(incident, np.abs(es.amplitude) ** 2, 0.0)
    k = es.delay.shape[1]
    area = rings * dr                                      # ring area per radian of azimuth
    for i in range(k):
        for j in range(k):
            t = es.delay[:, i] + es.delay[:, j]
            p = a2[:, i] * a2[:, j] * strength(graze[:, i], graze[:, j]) * area
            if absorption_db_per_m:
                p = p * 10.0 ** (-absorption_db_per_m * c * t / 10.0)
            idx = np.rint((t - times[0]) / dt).astype(int)
            ok = (idx >= 0) & (idx < len(times)) & (p > 0)
            np.add.at(power, idx[ok], p[ok])
    return power


def chapman_harris_strength(wind_speed: float, frequency: float):
    """Sea-surface scattering strength function for `reverberation_power`. Chapman-Harris is
    a backscattering fit; for a path pair with different grazing angles it is evaluated at
    their mean."""
    from .. import acoustics
    return lambda g_in, g_out: 10.0 ** (acoustics.chapman_harris_db(0.5 * (g_in + g_out), wind_speed,
                                                                    frequency) / 10.0)


def default_azimuth_count(positions: np.ndarray, wavelength: float) -> int:
    """Two sectors per beamwidth (lambda / extent) of the array, and at least 72."""
    extent = np.linalg.norm(positions.max(axis=0) - positions.min(axis=0))
    return int(max(72, np.ceil(4.0 * np.pi * extent / wavelength)))


def add_reverberation(record, provider: EigenrayProvider, boundary_z: float, mu_db: float,
                      rng: np.random.Generator, n_azimuth: int | None = None,
                      absorption_db_per_m: float = 0.0, max_range: float | None = None,
                      power_step: float | None = None, strength=None) -> np.ndarray:
    """Add boundary reverberation to a raw record (complex baseband or real passband), in place.

    Returns the reverberation power per radian in each sample of the impulse response
    (for the processing log and for tests).
    """
    s = record.samples
    wf = record.transmit.waveform
    real = s.kind == core.SampleKind.RealPassband
    carrier = s.kind != core.SampleKind.ComplexBaseband
    fs, n = s.sample_rate, s.n_samples
    fc = 0.5 * (wf.f_start + wf.f_stop) if carrier else s.center_frequency
    c = record.environment.sound_speed
    pulse = baseband_pulse(wf, fs, fc) * wf.amplitude
    # Impulse-response time axis: echoes of scatterers at two-way time t occupy [t, t + T].
    times = s.start_time - wf.duration + np.arange(n + len(pulse)) / fs
    positions = record.rx_positions()
    centre = positions.mean(axis=0)
    n_azimuth = n_azimuth or default_azimuth_count(positions, c / fc)
    # Power on a coarse time grid (it varies slowly), as a density, then per sample.
    step = power_step or max(1.0 / fs, min(0.25 * wf.duration, 2e-3))
    coarse = np.arange(max(times[0], 0.0), times[-1] + step, step)
    if len(coarse) < 2:
        coarse = np.array([times[0], times[0] + step])
    min_range = 0.6 * 0.5 * c * coarse[0] if coarse[0] > 0.2 else 0.0
    density = reverberation_power(provider, centre, boundary_z, coarse, mu_db, absorption_db_per_m,
                                  max_range, min_range=min_range, strength=strength) / step
    power = np.interp(times, coarse, density, left=0.0, right=0.0) / fs
    # Reverberation occupies the pulse's band only, so it is generated at a reduced rate
    # (at least four times the bandwidth) and interpolated to the record's rate.
    band = max(abs(wf.f_stop - wf.f_start), 4.0 / wf.duration)
    decim = max(int(fs // (4.0 * band)), 1)
    fs_low = fs / decim
    n_low = int(np.ceil(len(times) / decim))
    t_low = times[0] + np.arange(n_low) / fs_low
    p_low = np.interp(t_low, coarse, density, left=0.0, right=0.0) / fs_low
    pulse_low = baseband_pulse(wf, fs_low, fc) * wf.amplitude
    dphi = 2.0 * np.pi / n_azimuth
    sigma = np.sqrt(p_low * dphi / 2.0)                    # per real component, per sector
    f = np.fft.fftfreq(n_low, 1.0 / fs_low)
    total = np.zeros((positions.shape[0], n_low), dtype=np.complex64)
    offsets = positions - centre
    w = (2.0 * np.pi * (f + fc)).astype(np.float32)
    for m in range(n_azimuth):
        phi = (m + 0.5) * dphi
        b = sigma * (rng.standard_normal(n_low) + 1j * rng.standard_normal(n_low))
        spectrum = np.fft.fft(b).astype(np.complex64)
        lead = ((offsets @ np.array([np.cos(phi), np.sin(phi), 0.0])) / c).astype(np.float32)   # [channels]
        arg = lead[:, None] * w[None, :]
        total += spectrum[None, :] * (np.cos(arg) + 1j * np.sin(arg))
    total *= np.fft.fft(pulse_low, n_low).astype(np.complex64)[None, :]
    # Band-limited interpolation to the record's rate: zero-pad the spectrum.
    n_full = n_low * decim
    if decim > 1:
        padded = np.zeros((positions.shape[0], n_full), dtype=np.complex64)
        half = n_low // 2
        padded[:, :half] = total[:, :half]
        padded[:, n_full - (n_low - half):] = total[:, half:]
        series = np.fft.ifft(padded, axis=1) * decim
    else:
        series = np.fft.ifft(total, axis=1)
    # Sample k of the convolution is at times[0] + k / fs; the record starts one pulse later.
    start = int(round((s.start_time - times[0]) * fs))
    out = series[:, start:start + n]
    if carrier:
        t_abs = s.start_time + np.arange(n) / fs
        out = out * np.exp(2j * np.pi * fc * t_abs)[None, :]
    if real:
        out = np.real(out)
    x = s.array + out.astype(s.array.dtype)
    kind = s.kind
    s.array = x
    s.kind = kind
    record.samples = s
    return power[start:start + n]
