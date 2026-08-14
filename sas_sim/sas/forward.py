"""Point-scatterer forward model, single- or multichannel receive array.

Geometry (2D, matching sas_min.py): x = along-track, y = cross-track range.
One transmitter per ping at (x_tx, 0); N receive elements at
(x_tx + rx_off[i], 0).

PHYSICS ASSUMPTIONS (flagged per project brief):
  * Exact bistatic two-way delay tau = (R_tx + R_rx) / c is used -- NO
    phase-center approximation (PCA). TDBP can focus with exact delays,
    so the PCA is unnecessary here; it becomes relevant only for
    beamformers that require monostatic-gridded data (roadmap rung 2).
  * Receive elements are ideal omnidirectional point hydrophones: no
    element directivity pattern, no baffle, no inter-element shading.
  * Stop-and-hop: the platform is frozen during each ping (same as
    sas_min.py). No propagation loss, no additive noise (same as anchor).
"""

import numpy as np

from .waveform import chirp


def rx_offsets_uniform(n_rx, d_rx):
    """Along-track offsets of a uniform n_rx-element array centered on the
    transmitter. Phase centers (tx/rx midpoints) then sit at offset/2,
    i.e. spaced d_rx/2 apart -- the classic SAS vernier/DPCA layout."""
    return (np.arange(n_rx) - (n_rx - 1) / 2) * d_rx


def simulate_raw(x_tx, rx_off, scat, t, fc, K, Tp, c):
    """Raw baseband I/Q for a multichannel stripmap collection.

    x_tx   : (n_pings,) transmitter along-track positions
    rx_off : (n_rx,) receive-element offsets from the transmitter
             (use np.array([0.0]) for the single-channel/monostatic case)
    scat   : (n_scat, 3) columns x, y, reflectivity
    t      : (n_fast,) fast-time axis
    returns raw : (n_pings, n_rx, n_fast) complex

    r_{n,i}(t) = sum_k A_k * chirp(t - tau) * exp(-j 2 pi fc tau),
    tau = (R_tx + R_rx) / c   (exact bistatic delay, see module docstring)
    """
    x_tx = np.asarray(x_tx, dtype=float)
    rx_off = np.asarray(rx_off, dtype=float)
    sx, sy, sa = scat[:, 0], scat[:, 1], scat[:, 2]
    n_pings, n_rx, n_fast = len(x_tx), len(rx_off), len(t)

    raw = np.zeros((n_pings, n_rx, n_fast), dtype=complex)
    for n in range(n_pings):
        R_tx = np.sqrt((x_tx[n] - sx)**2 + sy**2)          # (n_scat,)
        for i in range(n_rx):
            x_rx = x_tx[n] + rx_off[i]
            R_rx = np.sqrt((x_rx - sx)**2 + sy**2)
            tau = (R_tx + R_rx) / c
            for k in range(len(sx)):
                raw[n, i] += (sa[k] * chirp(t - tau[k], K, Tp)
                              * np.exp(-1j * 2 * np.pi * fc * tau[k]))
    return raw


def fast_time_axis(x_tx, rx_off, scat, Tp, fs, c):
    """Fast-time axis covering all bistatic echo delays with 1.5*Tp margin,
    plus the equivalent-range gate used to kill the pre-echo ramp."""
    x_tx = np.asarray(x_tx, dtype=float)
    rx_off = np.asarray(rx_off, dtype=float)
    sx, sy = scat[:, 0], scat[:, 1]
    x_rx = x_tx[:, None] + rx_off[None, :]                    # (n_pings, n_rx)
    R_tx = np.sqrt((x_tx[:, None] - sx[None, :])**2 + sy[None, :]**2)
    R_rx = np.sqrt((x_rx[:, :, None] - sx[None, None, :])**2
                   + sy[None, None, :]**2)
    tau_all = (R_tx[:, None, :] + R_rx) / c
    t0 = tau_all.min() - 1.5 * Tp
    t1 = tau_all.max() + 1.5 * Tp
    t = np.arange(t0, t1, 1 / fs)
    return t, tau_all.min()
