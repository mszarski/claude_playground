"""One hydrophone as sound (docs/tasks/VISUALIZATION.md section 2.3).

Three ways, each named in what it returns, with what it misrepresents:

- `as_is`: real passband samples played at their own rate. Exact; audible only when the band is (the flank array's
  6-10 kHz, Popoto's 4-6 kHz). Players may refuse rates above 48 or 96 kHz.
- `remodulate`: complex baseband put back on its own carrier at a rate of at least 2.5 x the top of the band
  (FFT interpolation, then x exp(j 2 pi fc t), real part). Exact up to the interpolation; the same audibility rule.
- `shift`: the band moved down so that its centre sits at `audio_center` Hz (default 2 kHz), its width kept.
  Every frequency moves by the same number of hertz: a Doppler shift keeps its size in Hz but not its ratio, and
  harmonic relations are lost. Only `audio_bandwidth` about the centre is kept (default 1.8 x `audio_center`, 3.6 kHz:
  a band-pass first, so that the band does not fold through 0 Hz): of a 20-60 kHz wide FLS or SAS band, the rest is
  dropped, and with it the sweep's range resolution.

Time stretching (playing at a lower rate) is not done here: it lowers every frequency by the same ratio and
lengthens the sound by it; it is a page option in the design, not code.
"""

from __future__ import annotations

import struct

import numpy as np


def _baseband(x: np.ndarray, fs: float, fc: float) -> np.ndarray:
    """Complex baseband of a real passband signal about fc (FFT: the negative half dropped)."""
    n = len(x)
    X = np.fft.fft(x)
    f = np.fft.fftfreq(n, 1.0 / fs)
    X[f < 0] = 0.0
    a = np.fft.ifft(2.0 * X)
    return a * np.exp(-2j * np.pi * fc * np.arange(n) / fs)


def hydrophone_audio(record, channel: int = 0, mode: str = "auto", audio_center: float = 2000.0,
                     audio_bandwidth: float | None = None, audible_top: float = 20000.0):
    """(sample rate Hz, samples float [N], what was done) of one channel of a record as sound. `auto`: `as_is` for a
    real audible band, `remodulate` for an audible baseband record, `shift` otherwise."""
    s = record.samples
    x = np.asarray(s.array)[channel]
    fs, fc = float(s.sample_rate), float(s.center_frequency)
    if mode == "auto":
        top = fc + 0.5 * fs if s.is_complex() else None
        if not s.is_complex():
            # A real record: its band is the sonar's; centre frequency is the carrier the record states.
            mode = "as_is" if fc < audible_top * 0.8 else "shift"
        else:
            mode = "remodulate" if top is not None and top <= audible_top else "shift"
    if mode == "as_is":
        if s.is_complex():
            raise ValueError("as_is needs real passband samples; use remodulate or shift")
        return fs, x.astype(float), {"mode": "as_is", "sample_rate": fs}
    a = x.astype(complex) if s.is_complex() else _baseband(x.astype(float), fs, fc)
    if mode == "shift" and audio_bandwidth is None:
        # Kept narrow enough that the band does not fold through zero once it sits at `audio_center`.
        audio_bandwidth = min(fs, 1.8 * float(audio_center))
    if audio_bandwidth is not None:
        A = np.fft.fft(a)
        A[np.abs(np.fft.fftfreq(len(a), 1.0 / fs)) > 0.5 * audio_bandwidth] = 0.0
        a = np.fft.ifft(A)
    carrier = fc if mode == "remodulate" else float(audio_center)
    half = 0.5 * (audio_bandwidth or fs)
    out_fs = float(int(np.ceil(2.5 * (carrier + half) / 1000.0)) * 1000)        # a whole number of kHz
    n = len(a)
    m = int(round(n * out_fs / fs))
    A = np.fft.fft(a)
    f = np.fft.fftfreq(n, 1.0 / fs)
    B = np.zeros(m, dtype=complex)
    keep = np.abs(f) < 0.5 * min(fs, out_fs)
    B[np.round(f[keep] * m / out_fs).astype(int) % m] = A[keep]
    b = np.fft.ifft(B) * (m / n)
    y = np.real(b * np.exp(2j * np.pi * carrier * np.arange(m) / out_fs))
    return out_fs, y, {"mode": mode, "carrier_hz": carrier, "from_center_hz": fc, "sample_rate": out_fs,
                       "kept_bandwidth_hz": audio_bandwidth or fs, "shift_hz": (fc - carrier) if mode == "shift" else 0.0}


def write_wav(path: str, fs: float, y: np.ndarray, normalise: bool = True) -> dict:
    """A mono 32-bit float WAV (format 3) as taps.export_wav writes them. Returns {sample_rate, n, scale}."""
    y = np.asarray(y, dtype=float)
    scale = 1.0 / max(float(np.abs(y).max()), 1e-30) if normalise else 1.0
    data = (y * scale).astype("<f4").tobytes()
    rate = int(round(fs))
    header = struct.pack("<4sI4s4sIHHIIHH4sI", b"RIFF", 36 + len(data), b"WAVE", b"fmt ", 16, 3, 1, rate, rate * 4, 4, 32,
                         b"data", len(data))
    with open(path, "wb") as f:
        f.write(header + data)
    return {"sample_rate": rate, "n": len(y), "scale": scale}


def read_wav(path: str):
    """(sample rate, samples) of a WAV written by `write_wav` (for the tests)."""
    with open(path, "rb") as f:
        raw = f.read()
    fields = struct.unpack("<4sI4s4sIHHIIHH4sI", raw[:44])
    return fields[7], np.frombuffer(raw[44:], dtype="<f4")
