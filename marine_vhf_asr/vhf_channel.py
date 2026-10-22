"""
Marine VHF radio channel simulation.

Turns clean (TTS or studio) speech into something that sounds like a
narrowband FM voice transmission picked up by a scanner or shore receiver:
- Engine / wind background picked up by the handset mic
- 300-3000 Hz voice band limiting
- Transmitter limiting (hard-ish clipping) and FM de-emphasis
- Receiver hiss at a random SNR, with occasional picket-fence fades
- Squelch open burst before and squelch tail after the transmission
- Optional 8 kHz scanner/recorder path

All randomness is drawn from the supplied numpy Generator so augmentation is
reproducible.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy import signal


@dataclass
class ChannelConfig:
    """Ranges sampled uniformly for each simulated transmission."""
    snr_db: Tuple[float, float] = (3.0, 30.0)
    background_db: Tuple[float, float] = (-40.0, -15.0)  # mic background relative to speech
    band_hz: Tuple[float, float] = (300.0, 3000.0)
    clip_drive: Tuple[float, float] = (1.0, 4.0)          # >1 drives the limiter harder
    squelch_open_s: Tuple[float, float] = (0.03, 0.15)
    squelch_tail_s: Tuple[float, float] = (0.08, 0.30)
    lead_silence_s: Tuple[float, float] = (0.05, 0.40)
    fade_prob: float = 0.3          # probability of picket-fence fading in a clip
    narrowband_prob: float = 0.3    # probability of the 8 kHz recorder path


def _colored_noise(n: int, rng: np.random.Generator, exponent: float) -> np.ndarray:
    """Noise with 1/f^exponent power spectrum (0=white, 1=pink, 2=brown)."""
    spec = rng.standard_normal(n // 2 + 1) + 1j * rng.standard_normal(n // 2 + 1)
    f = np.arange(len(spec), dtype=float)
    f[0] = 1.0
    spec /= f ** (exponent / 2)
    noise = np.fft.irfft(spec, n)
    return noise / (np.std(noise) + 1e-12)


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x ** 2)) + 1e-12)


def _fade_envelope(n: int, sr: int, rng: np.random.Generator) -> np.ndarray:
    """Short random dropouts, like a weak signal behind a moving ship's superstructure."""
    env = np.ones(n)
    for _ in range(rng.integers(1, 5)):
        width = int(sr * rng.uniform(0.02, 0.15))
        start = rng.integers(0, max(1, n - width))
        depth = rng.uniform(0.1, 0.7)
        env[start:start + width] *= depth
    # Smooth edges so fades do not click
    win = signal.windows.hann(max(3, int(sr * 0.01)))
    return np.convolve(env, win / win.sum(), mode="same")


def simulate_vhf(
    audio: np.ndarray,
    sr: int,
    rng: Optional[np.random.Generator] = None,
    config: Optional[ChannelConfig] = None,
) -> Tuple[np.ndarray, dict]:
    """
    Apply a random marine VHF channel to a mono speech clip.

    Args:
        audio: Mono float waveform
        sr: Sample rate (16 kHz recommended)
        rng: Random generator
        config: Parameter ranges

    Returns:
        (degraded waveform with peak 0.9, dict of sampled parameters)
    """
    rng = rng or np.random.default_rng()
    cfg = config or ChannelConfig()
    x = np.asarray(audio, dtype=np.float64)
    x = x / (np.max(np.abs(x)) + 1e-12)
    u = lambda r: float(rng.uniform(*r))  # noqa: E731
    params = {}

    # 1. Background at the microphone (engine rumble / wind): brown-ish noise
    params["background_db"] = u(cfg.background_db)
    bg = _colored_noise(len(x), rng, exponent=rng.uniform(1.0, 2.0))
    x = x + bg * _rms(x) * 10 ** (params["background_db"] / 20)

    # 2. Transmitter: voice band filter and limiter
    lo, hi = cfg.band_hz
    hi = min(hi, 0.45 * sr)
    sos = signal.butter(4, [lo, hi], btype="bandpass", fs=sr, output="sos")
    x = signal.sosfiltfilt(sos, x)
    params["clip_drive"] = u(cfg.clip_drive)
    x = np.tanh(params["clip_drive"] * x / (np.max(np.abs(x)) + 1e-12))

    # 3. Pad with squelch open burst, lead silence and squelch tail
    lead = int(sr * u(cfg.lead_silence_s))
    sq_open = int(sr * u(cfg.squelch_open_s))
    sq_tail = int(sr * u(cfg.squelch_tail_s))
    speech_rms = _rms(x)
    x = np.concatenate([np.zeros(lead + sq_open), x, np.zeros(sq_tail)])
    carrier = np.zeros_like(x)
    carrier[lead:] = 1.0  # receiver hiss is present once squelch opens

    # 4. Receiver hiss at random SNR, squelch tail is a louder burst
    params["snr_db"] = u(cfg.snr_db)
    hiss = _colored_noise(len(x), rng, exponent=0.0)
    hiss = signal.sosfiltfilt(sos, hiss)
    hiss /= _rms(hiss)
    noise_gain = speech_rms * 10 ** (-params["snr_db"] / 20)
    noise = hiss * noise_gain * carrier
    tail_start = len(x) - sq_tail
    noise[lead:lead + sq_open] *= 3.0
    noise[tail_start:] *= 4.0
    y = x + noise

    params["faded"] = bool(rng.random() < cfg.fade_prob)
    if params["faded"]:
        y *= _fade_envelope(len(y), sr, rng)

    # 5. Receiver de-emphasis (gentle 1st-order low-pass)
    b, a = signal.butter(1, min(2500.0, 0.45 * sr), btype="low", fs=sr)
    y = signal.lfilter(b, a, y)

    # 6. Optional narrowband recorder path
    params["narrowband"] = bool(rng.random() < cfg.narrowband_prob) and sr > 8000
    if params["narrowband"]:
        y = signal.resample_poly(y, 8000, sr)
        y = signal.resample_poly(y, sr, 8000)

    y = 0.9 * y / (np.max(np.abs(y)) + 1e-12)
    return y.astype(np.float32), params
