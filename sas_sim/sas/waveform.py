"""LFM chirp + matched-filter range compression (lifted from sas_min.py)."""

import numpy as np


def chirp(tt, K, Tp):
    """Baseband LFM chirp with rate K [Hz/s], nonzero over |tt| <= Tp/2."""
    s = np.exp(1j * np.pi * K * tt**2)
    s[np.abs(tt) > Tp / 2] = 0.0
    return s


def range_compress(raw, t, fs, K, Tp):
    """Matched-filter range compression along the last (fast-time) axis.

    Same algorithm as sas_min.py: Hanning-tapered matched filter, integer
    lag calibrated against a known synthetic echo so a scatterer's peak
    lands exactly at its true two-way delay on the fast-time axis `t`.

    raw : (..., n_fast) complex baseband I/Q
    returns rc with the same shape, aligned to `t`.
    """
    n_fast = raw.shape[-1]
    tref = np.arange(-Tp / 2, Tp / 2, 1 / fs)
    h = np.conj(chirp(tref, K, Tp)[::-1])
    h *= np.hanning(len(h))                  # taper -> lower range sidelobes
    nfft = 1 << int(np.ceil(np.log2(n_fast + len(h))))
    H = np.fft.fft(h, nfft)

    # calibrate the integer lag from a known single echo at t = t[cal]
    cal = n_fast // 2
    probe = chirp(t - t[cal], K, Tp)
    pc = np.fft.ifft(np.fft.fft(probe, nfft) * H)
    lag = np.argmax(np.abs(pc)) - cal

    rc = np.fft.ifft(np.fft.fft(raw, nfft, axis=-1) * H, axis=-1)
    return rc[..., lag:lag + n_fast]
