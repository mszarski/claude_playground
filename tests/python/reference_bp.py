"""CPU reference backprojection, used only in tests.

Deliberately naive and independent of the kernel: float64 (or long double for
delays), one pixel and one element at a time, the implicit arrival time solved by
plain iteration to convergence, interpolation coefficients from their defining
formulas rather than a table. It shares no code with libs/recon.
"""

import numpy as np
from scipy.special import i0

import gpusonar as gs

LD = np.longdouble


def _positions(record, t):
    return np.asarray(record.rx_positions(float(t)), dtype=LD)


def exact_delay(record, channel, point, stop_and_hop=False):
    """Two-way delay for one element in extended precision."""
    x = np.asarray(point, dtype=LD)
    tx = np.asarray(record.tx_position(), dtype=LD)
    c = LD(record.environment.sound_speed)
    if record.transmit.kind == gs.TransmitKind.PlaneWave:
        r = gs.core.rotation_ned(*record.geometry.tx_pose.rpy)
        n = np.asarray(r @ np.asarray(record.transmit.direction), dtype=LD)
        r_tx = (x - tx) @ n
    else:
        r_tx = np.sqrt(((x - tx) ** 2).sum())
    p0 = np.asarray(record.rx_positions(0.0)[channel], dtype=LD)
    if stop_and_hop:
        return (r_tx + np.sqrt(((x - p0) ** 2).sum())) / c
    # First-order motion of the element: evaluate at two times to get its velocity exactly
    # as the record's pose model defines it (rotation included), then iterate.
    tau = (r_tx + np.sqrt(((x - p0) ** 2).sum())) / c
    for _ in range(12):
        p = np.asarray(record.rx_positions(float(tau))[channel], dtype=LD)
        # rx_positions takes a float64 time; refine the last bits with the local velocity.
        h = 1e-3
        v = (np.asarray(record.rx_positions(float(tau) + h)[channel], dtype=LD) - p) / LD(h)
        p = p + v * (tau - LD(float(tau)))
        tau = (r_tx + np.sqrt(((x - p) ** 2).sum())) / c
    return tau


def _kaiser_sinc(frac, taps=8, beta=6.0):
    half = taps // 2
    k = np.arange(taps)
    x = (k - (half - 1)) - frac
    h = np.sinc(x) * np.where(np.abs(x / half) < 1, i0(beta * np.sqrt(np.clip(1 - (x / half) ** 2, 0, 1))) / i0(beta), 0.0)
    return h / h.sum()


def interpolate(samples, pos, interp):
    n = int(np.floor(pos))
    frac = float(pos - n)
    N = len(samples)
    if interp == "nearest":
        i = n + 1 if frac >= 0.5 else n
        return samples[i] if 0 <= i < N else None
    if interp == "linear":
        if n < 0 or n + 1 >= N:
            return None
        return (1 - frac) * samples[n] + frac * samples[n + 1]
    first = n - 3
    if first < 0 or first + 8 > N:
        return None
    return complex(np.dot(_kaiser_sinc(frac), samples[first:first + 8]))


def backproject(records, points, interp="sinc", stop_and_hop=False, weights=None):
    """Coherent sum per point, float64/complex128."""
    points = np.atleast_2d(points)
    out = np.zeros(len(points), dtype=complex)
    for ir, rec in enumerate(records):
        x = rec.samples.array.astype(complex)
        fs, t0 = rec.samples.sample_rate, rec.samples.start_time
        fc = rec.samples.center_frequency if rec.samples.kind == gs.SampleKind.ComplexBaseband else 0.0
        for ip, pt in enumerate(points):
            for ch in range(x.shape[0]):
                tau = exact_delay(rec, ch, pt, stop_and_hop)
                pos = (tau - LD(t0)) * LD(fs)
                v = interpolate(x[ch], pos, interp)
                if v is None:
                    continue
                cyc = LD(fc) * tau
                phase = float(cyc - np.floor(cyc))
                w = 1.0 if weights is None else weights[ir][ch]
                out[ip] += w * v * np.exp(2j * np.pi * phase)
    return out
