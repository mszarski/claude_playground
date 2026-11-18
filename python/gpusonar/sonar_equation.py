"""Sonar-equation tool: predicted signal-to-noise ratio against range for a sonar config.

    SNR_out = SL - 2 TL + TS - (NL0 - AG) + 10 log10(T)

with SL the source level (dB re 1 uPa at 1 m), TL the one-way transmission loss, TS the
target strength, NL0 the noise spectrum level (dB re 1 uPa^2/Hz) at the hydrophone, AG the
array gain, and 10 log10(T) the matched-filter gain against white noise for a pulse of
duration T (equivalently: in-band noise NL0 + 10 log10(B), compression gain 10 log10(BT)).
SNR_out is the ratio of peak signal power to mean noise power at the beamformer output.
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
    array_gain_db: float | None = None  # default: 10 log10(N), uncorrelated noise

    @classmethod
    def from_sonar(cls, sonar: SonarModel, noise_spectrum_db: float, **kw) -> "SonarEquation":
        w = sonar.waveform_config
        if sonar.source_level_db_rms is None:
            raise ValueError(f"sonar {sonar.name!r} has no source level; the chain is uncalibrated")
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

    def snr(self, range_m):
        """Output SNR (dB) after pulse compression and beamforming."""
        return (self.echo_level(range_m) - self.noise_level_in_band() + self.processing_gain()
                + self.array_gain())

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
        return [{"range_m": float(r), "tl_db": float(self.transmission_loss(r)),
                 "echo_level_db": float(self.echo_level(r)), "snr_db": float(self.snr(r))} for r in ranges]


def noise_rms_volts(noise_spectrum_db: float, sensitivity_db: float, sample_rate: float,
                    gain_db: float = 0.0, real_samples: bool = True) -> float:
    """RMS of white noise at the ADC for a pressure spectrum level, per real sample.

    White over the Nyquist band fs / 2 for real sampling (fs for complex baseband).
    """
    band = 0.5 * sample_rate if real_samples else sample_rate
    return 10.0 ** ((noise_spectrum_db + 10.0 * np.log10(band) + sensitivity_db + gain_db) / 20.0)
