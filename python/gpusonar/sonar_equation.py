"""Sonar-equation tool: predicted signal-to-noise ratio against range for a sonar config.

    SNR_out = SL - 2 TL + TS - (NL0 - AG) + 10 log10(T)

with SL the source level (dB re 1 uPa at 1 m), TL the one-way transmission loss, TS the
target strength, NL0 the noise spectrum level (dB re 1 uPa^2/Hz) at the hydrophone, AG the
array gain, and 10 log10(T) the matched-filter gain against white noise for a pulse of
duration T (equivalently: in-band noise NL0 + 10 log10(B), compression gain 10 log10(BT)).
SNR_out is the ratio of peak signal power to mean noise power at the beamformer output.

Reverberation (sonar-w2w3, D-198): with `reverberation_db` (a function of slant range giving the reverberation level
re the source level at the target's delay, after the same pulse compression and the beam: gpusonar/reverb_budget.py,
Jackson and Richardson 2007 (G.11)-(G.13) with the compressed pulse) the SNR is over the power sum of noise and
reverberation:

    SNR_out = -10 log10(10^(-SNR_noise / 10) + 10^(-SRR / 10)),   SRR = TS - 2 TL - RL_re_SL

`signal_to_noise` and `signal_to_reverberation` give the two apart and `limited_by` which is the smaller. Without it
(the default) every figure is the noise-only one, computed exactly as before. A budget without reverberation is
optimistic wherever the boundaries' reverberation exceeds the noise: say so where its ranges are quoted.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import acoustics
from .sonar import SonarModel


@dataclass
class SonarEquation:
    source_level_db: float              # dB re 1 uPa at 1 m, RMS (the suite's one convention: sonar.source_level_db_rms)
    center_frequency: float
    bandwidth: float
    pulse_duration: float
    n_elements: int
    noise_spectrum_db: float           # dB re 1 uPa^2 / Hz
    target_strength_db: float = 0.0
    spreading: float = 20.0            # TL = spreading * log10(R) + alpha R  (20 = spherical)
    temperature: float = 10.0
    salinity: float = 35.0
    depth: float = 0.0
    absorption_model: str = "francois_garrison"
    array_gain_db: float | None = None  # default: 10 log10(N), uncorrelated noise (from_sonar: against isotropic sea noise)
    reverberation_db: object = None     # range -> reverberation level re SL at that delay, dB (reverb_budget.at_ranges); None: noise only

    @classmethod
    def from_sonar(cls, sonar: SonarModel, noise_spectrum_db: float, **kw) -> "SonarEquation":
        """The equation of a sonar file. The array gain, unless given, is the array's gain against ISOTROPIC noise at the
        centre frequency (`isotropic_noise_gain_db`: the noise the simulator draws, D-103), which is 10 log10(N) at half a
        wavelength of pitch and less for hydrophones closer than that."""
        w = sonar.waveform_config
        if sonar.source_level_db_rms is None:
            raise ValueError(f"sonar {sonar.name!r} has no source level; the chain is uncalibrated")
        if kw.get("array_gain_db") is None:
            kw["array_gain_db"] = isotropic_noise_gain_db(sonar.positions, sonar.center_frequency, sonar.nominal_sound_speed, sonar.boresight)
        return cls(float(sonar.source_level_db_rms), sonar.center_frequency, sonar.bandwidth,
                   float(w["duration"]), sonar.n_elements, noise_spectrum_db, **kw)

    def absorption_db_per_m(self) -> float:
        return float(acoustics.absorption(self.center_frequency, self.temperature, self.salinity,
                                          self.depth, model=self.absorption_model))

    def transmission_loss(self, range_m):
        r = np.asarray(range_m, dtype=float)
        return self.spreading * np.log10(r) + self.absorption_db_per_m() * r

    def array_gain(self) -> float:
        return 10.0 * np.log10(self.n_elements) if self.array_gain_db is None else self.array_gain_db

    def echo_level(self, range_m):
        """dB re 1 uPa at a hydrophone."""
        return self.source_level_db - 2.0 * self.transmission_loss(range_m) + self.target_strength_db

    def noise_level_in_band(self) -> float:
        return self.noise_spectrum_db + 10.0 * np.log10(self.bandwidth)

    def processing_gain(self) -> float:
        return 10.0 * np.log10(self.bandwidth * self.pulse_duration)

    def signal_to_noise(self, range_m):
        """Output SNR (dB) against noise alone, after pulse compression and beamforming."""
        return (self.echo_level(range_m) - self.noise_level_in_band() + self.processing_gain()
                + self.array_gain())

    def noise_equivalent_db(self) -> float:
        """The noise after processing in the echo's units (dB re 1 uPa^2): what the echo level is compared with."""
        return self.noise_level_in_band() - self.processing_gain() - self.array_gain()

    def reverberation_level(self, range_m):
        """Reverberation at the delay of a target at `range_m`, after processing, in the echo's units (dB re 1 uPa^2);
        -inf without the term."""
        if self.reverberation_db is None:
            return np.full(np.shape(range_m), -np.inf) if np.ndim(range_m) else -np.inf
        return self.source_level_db + np.asarray(self.reverberation_db(range_m), dtype=float)

    def signal_to_reverberation(self, range_m):
        """Echo over reverberation (dB; +inf without the term). Independent of the source level."""
        return self.echo_level(range_m) - self.reverberation_level(range_m)

    def snr(self, range_m):
        """Output SNR (dB) after pulse compression and beamforming: over noise alone, or with `reverberation_db` over
        the power sum of noise and reverberation."""
        if self.reverberation_db is None:
            return self.signal_to_noise(range_m)
        n, r = self.signal_to_noise(range_m), self.signal_to_reverberation(range_m)
        return -10.0 * np.log10(10.0 ** (-np.asarray(n) / 10.0) + 10.0 ** (-np.asarray(r) / 10.0))

    def limited_by(self, range_m):
        """"noise" or "reverberation": the larger of the two after processing at that range."""
        r = self.signal_to_reverberation(range_m) < self.signal_to_noise(range_m)
        return np.where(r, "reverberation", "noise") if np.ndim(r) else ("reverberation" if r else "noise")

    def detection_range(self, threshold_db: float, r_min: float = 1.0, r_max: float = 1e6) -> float:
        """Largest range at which SNR >= threshold (SNR falls monotonically with range)."""
        if self.snr(r_min) < threshold_db:
            return 0.0
        if self.snr(r_max) >= threshold_db:
            return r_max
        lo, hi = r_min, r_max
        for _ in range(80):
            mid = np.sqrt(lo * hi)
            lo, hi = (mid, hi) if self.snr(mid) >= threshold_db else (lo, mid)
        return float(lo)

    def table(self, ranges) -> list[dict]:
        rows = [{"range_m": float(r), "tl_db": float(self.transmission_loss(r)),
                 "echo_level_db": float(self.echo_level(r)), "snr_db": float(self.snr(r))} for r in ranges]
        if self.reverberation_db is not None:            # the two parts and which limits (the noise-only table is unchanged)
            for row in rows:
                r = row["range_m"]
                row.update(snr_noise_db=float(self.signal_to_noise(r)), srr_db=float(self.signal_to_reverberation(r)),
                           reverberation_level_db=float(self.reverberation_level(r)), limited_by=str(self.limited_by(r)))
        return rows


def isotropic_noise_gain_db(positions, frequency: float, sound_speed: float = acoustics.NOMINAL_SOUND_SPEED, direction=None, weights=None) -> float:
    """Gain of an array of omnidirectional hydrophones at `positions` ([N, 3] or [N] along a line, m) against isotropic
    noise, summed in phase for a plane wave from `direction` (default: no steering, which is broadside of a line or a
    plane): |sum w|^2 / sum_ij w_i w_j* sinc(2 d_ij / lambda), the closed form the simulator's noise is held to (D-103).
    `weights`: an aperture window (default: none). Above 4096 hydrophones: 10 log10(N) (the sum is N^2 terms)."""
    x = np.asarray(positions, dtype=float)
    if x.ndim == 1:
        x = np.stack([x, np.zeros_like(x), np.zeros_like(x)], axis=1)
    n = len(x)
    w = np.ones(n, dtype=complex) if weights is None else np.asarray(weights, dtype=complex)
    if n > 4096:
        return float(10.0 * np.log10(np.abs(w.sum()) ** 2 / np.sum(np.abs(w) ** 2)))
    k = 2.0 * np.pi * frequency / sound_speed
    if direction is not None:
        u = np.asarray(direction, dtype=float)
        w = w * np.exp(-1j * k * (x @ (u / np.linalg.norm(u))))            # the weights that bring a wave from u into phase
        signal = np.abs(np.sum(w * np.exp(1j * k * (x @ (u / np.linalg.norm(u)))))) ** 2
    else:
        signal = np.abs(w.sum()) ** 2
    d = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=2)
    noise = float(np.real(np.conj(w) @ (np.sinc(2.0 * d * frequency / sound_speed) @ w)))
    return float(10.0 * np.log10(signal / noise))


def noise_rms_volts(noise_spectrum_db: float, sensitivity_db: float, sample_rate: float,
                    gain_db: float = 0.0, real_samples: bool = True) -> float:
    """RMS of white noise at the ADC for a pressure spectrum level, per real sample.

    White over the Nyquist band fs / 2 for real sampling (fs for complex baseband).
    """
    band = 0.5 * sample_rate if real_samples else sample_rate
    return 10.0 ** ((noise_spectrum_db + 10.0 * np.log10(band) + sensitivity_db + gain_db) / 20.0)
