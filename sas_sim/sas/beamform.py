"""Time-Domain Back-Projection with exact bistatic delays per channel.

Reduces exactly to the sas_min.py beamformer when rx_off == [0.0]
(monostatic: tau = 2R/c).
"""

import numpy as np


def tdbp(rc, t, x_tx, rx_off, gx, gy, fc, c):
    """Delay-and-sum with focusing phase, per output pixel.

    rc     : (n_pings, n_rx, n_fast) range-compressed data aligned to `t`
    x_tx   : (n_pings,) transmitter along-track positions
    rx_off : (n_rx,) receive-element offsets from the transmitter
    gx, gy : pixel coordinate arrays (same shape), along-track / range [m]
    returns slc : complex image, same shape as gx
    """
    x_tx = np.asarray(x_tx, dtype=float)
    rx_off = np.asarray(rx_off, dtype=float)
    slc = np.zeros(gx.shape, dtype=complex)
    ti, tq = np.real(rc), np.imag(rc)        # interp real & imag separately

    for n in range(len(x_tx)):
        R_tx = np.sqrt((x_tx[n] - gx)**2 + gy**2)
        for i in range(len(rx_off)):
            x_rx = x_tx[n] + rx_off[i]
            R_rx = np.sqrt((x_rx - gx)**2 + gy**2)
            tau = (R_tx + R_rx) / c          # exact bistatic delay
            smp = (np.interp(tau.ravel(), t, ti[n, i]) +
                   1j * np.interp(tau.ravel(), t, tq[n, i])).reshape(gx.shape)
            slc += smp * np.exp(1j * 2 * np.pi * fc * tau)   # focusing phase
    return slc
