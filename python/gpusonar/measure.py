"""Estimators for measured seafloor data: backscattering strength from a calibrated echo level,
the number of independent samples behind an average, and the statistical error of the average.

Why a module of its own, beside `sonar_equation.py` and `calibrate.py` rather than under `sim/`: these
functions turn a REAL sonar's calibrated levels into the quantities the simulator's models predict
(they are the measurement side of `sim/sediment_fit.py` and `tools/fit_sediment.py`), and nothing in
the simulator calls them. They are Python only; neither engine has a counterpart.

Sources (the formulas are from the reference notes, which give each with its equation number; the
book's pages are not in the repository):

  [JR07]  D. R. Jackson and M. D. Richardson, "High-Frequency Seafloor Acoustics" (Springer, 2007),
          App. G (measurement techniques) via docs/references/jackson_richardson_2007/
          03_wave_theories_reflection_penetration.md Sec. 2.9 and Sec. 6; Ch. 16 (statistics) via
          02_roughness_heterogeneity_statistics.md Sec. 2.6; Sect. 11.1 via note 03 Sec. 2.5.
  [DP97]  P. H. Dahl and W. J. Plant, J. Acoust. Soc. Am. 101, 2596 (1997), eqs. (5)-(7), via
          docs/references/surface_papers_mcdaniel_dahl.md Sec. 2.9.

What each function rests on is said in its docstring: "printed" (the source prints it), "corrected"
(the source prints it with a misprint the note identifies), or "ours" (derived here from a printed
statement, with the derivation).

Statistics of an average of N independent intensity samples (Gaussian scattered field, [JR07] (16.14):
each intensity sample is exponential with standard deviation equal to its mean):

  - average the intensities, then take dB: relative standard deviation 1/sqrt(N) ([JR07] G.2.3,
    printed); in dB +1 sigma = 10 log10(1 + 1/sqrt(N)) (0.97 dB at N = 16, 0.50 at 67, 0.26 at 260;
    the note's form). The average of N unit-mean exponentials is Gamma(N, 1/N), so (ours, from that
    distribution) the dB of the average has mean 10 log10(e) (psi(N) - ln N) (about -4.34 / (2 N) dB)
    and standard deviation 10 log10(e) sqrt(psi'(N)) (about 4.34 / sqrt(N) dB; 1.10 dB at N = 16);
  - take dB of each sample, then average: biased low by 10 log10(e) C_E = 2.507 dB for every N
    ([JR07] (16.15), [DP97] eq. (6) with sigma_T = 0), standard deviation 5.57 / sqrt(N) dB
    ([JR07] (16.17), [DP97] eq. (7) with sigma_T = 0; the 1 / sqrt(N) for independent samples is
    ours). Hence the book's rule for reflection and scattering data alike: average squared
    envelopes, never dB ([JR07] Sect. 11.1).
"""
from __future__ import annotations

import math
import warnings

import numpy as np
from scipy.special import digamma, polygamma

from .acoustics import NOMINAL_SOUND_SPEED

DB = 10.0 / math.log(10.0)                     # 10 log10(e) = 4.3429 dB per neper of intensity
EULER_GAMMA = 0.5772156649015329               # C_E
DB_MEAN_BIAS_DB = -DB * EULER_GAMMA            # -2.507 dB: [JR07] (16.15); [DP97] eq. (6)
SINGLE_LOOK_DB_STD = DB * math.pi / math.sqrt(6.0)   # 5.57 dB: [JR07] (16.17); [DP97] eq. (7)
MAX_RANGE_RESOLUTION_RATIO = 0.1               # c tau / 2 against r_t: "<<" of (G.9), our threshold
MAX_HORIZONTAL_BEAMWIDTH = math.radians(20.0)  # (G.24)-(G.27): "about 20 degrees or less"


# --- Statistical error -----------------------------------------------------------------------------

def sigma_db_from_samples(n):
    """[JR07] G.2.3 (printed: standard deviation of an average of N independent intensity samples =
    mean / sqrt(N), "if N = 16 ... one-fourth of the mean, or roughly 1 dB"); in dB as the note
    writes it: +1 sigma = 10 log10(1 + 1/sqrt(N)). 0.969 dB at N = 16, 0.501 at 67, 0.261 at 260.
    It is the UPPER error bar: the lower one, 10 log10(1 - 1/sqrt(N)), is larger in size (-1.25 dB
    at N = 16; `error_bar_db`), and the standard deviation of the dB value itself is
    `intensity_mean_db_std` (1.10 dB at 16). `tools/fit_sediment.py --independent-samples` uses this
    one, the book's, as the error of a dB data point."""
    n = np.asarray(n, dtype=float)
    if np.any(~np.isfinite(n)) or np.any(n < 1.0):
        raise ValueError("independent samples must be finite and at least 1")
    out = 10.0 * np.log10(1.0 + 1.0 / np.sqrt(n))
    return float(out) if out.ndim == 0 else out


def error_bar_db(n) -> tuple:
    """(lower, upper) dB error bars of an intensity average of N independent samples: 10 log10(1 -+
    1/sqrt(N)) (from [JR07] G.2.3's mean / sqrt(N); the lower one is -inf at N = 1)."""
    n = np.asarray(n, dtype=float)
    with np.errstate(divide="ignore"):
        return 10.0 * np.log10(1.0 - 1.0 / np.sqrt(n)), 10.0 * np.log10(1.0 + 1.0 / np.sqrt(n))


def samples_for_error_db(sigma_db: float) -> int:
    """The least N with `sigma_db_from_samples(N) <= sigma_db` (the inverse of G.2.3's rule):
    16 for 0.97 dB, 68 for 0.5 dB (67 gives 0.5006), 263 for 0.26 dB (260 gives 0.2613)."""
    if not sigma_db > 0.0:
        raise ValueError("sigma_db must be positive")
    n = 1.0 / (10.0 ** (sigma_db / 10.0) - 1.0) ** 2
    m = max(1, math.ceil(n - 1e-9))
    return m - 1 if m > 1 and sigma_db_from_samples(m - 1) <= sigma_db else m


def intensity_mean_db_bias(n):
    """Mean of 10 log10(average of N unit-mean exponential intensities): 10 log10(e) (psi(N) - ln N),
    about -4.34 / (2 N) dB (-0.137 dB at 16, -0.032 at 67). Ours: the average is Gamma(N, 1/N) and
    E[ln X] = psi(N) - ln N for that distribution."""
    n = np.asarray(n, dtype=float)
    out = DB * (digamma(n) - np.log(n))
    return float(out) if out.ndim == 0 else out


def intensity_mean_db_std(n):
    """Standard deviation of 10 log10(average of N independent exponential intensities):
    10 log10(e) sqrt(psi'(N)), about 4.34 / sqrt(N) dB (5.57 at N = 1, 1.10 at 16, 0.53 at 67). Ours,
    as `intensity_mean_db_bias` (Var[ln X] = psi'(N) for Gamma(N, .))."""
    n = np.asarray(n, dtype=float)
    out = DB * np.sqrt(polygamma(1, n))
    return float(out) if out.ndim == 0 else out


def db_mean_bias(n=1):
    """Mean of the average of N dB values of exponential intensities, relative to the dB of the mean
    intensity: -10 log10(e) C_E = -2.507 dB whatever N ([JR07] (16.15): <ln y> = ln<y> - 0.5772;
    [DP97] eq. (6) with sigma_T = 0)."""
    return DB_MEAN_BIAS_DB if np.ndim(n) == 0 else np.full(np.shape(n), DB_MEAN_BIAS_DB)


def db_mean_std(n):
    """Standard deviation of the average of N independent dB values of exponential intensities:
    5.57 / sqrt(N) dB ([JR07] (16.17) and [DP97] eq. (7) give 5.57 dB for one sample; the 1 / sqrt(N)
    for independent samples is ours)."""
    n = np.asarray(n, dtype=float)
    out = SINGLE_LOOK_DB_STD / np.sqrt(n)
    return float(out) if out.ndim == 0 else out


def intensity_average_db(levels_db, axis=None):
    """10 log10 of the mean INTENSITY of levels given in dB: the average the book prescribes for
    reflection and scattering data ([JR07] Sect. 11.1: average the squared reflected pressure over
    many patches; averaging envelopes or dB is biased, by -2.507 dB for Rayleigh speckle)."""
    levels_db = np.asarray(levels_db, dtype=float)
    return 10.0 * np.log10(np.mean(10.0 ** (levels_db / 10.0), axis=axis))


# --- The number of independent samples ------------------------------------------------------------

def time_resolution(pulse_length: float, bandwidth: float | None = None) -> float:
    """Spacing of independent samples within a ping: the width of the transmitted signal's
    autocorrelation, about 1 / bandwidth, and the pulse length for a CW pulse ([JR07] p. 429, via note
    02 Sec. 2.6). `bandwidth` None means CW."""
    if not pulse_length > 0.0:
        raise ValueError("pulse_length must be positive")
    if bandwidth is None:
        return float(pulse_length)
    if not bandwidth > 0.0:
        raise ValueError("bandwidth must be positive")
    return float(min(pulse_length, 1.0 / bandwidth))


def independent_samples(duration: float, pulse_length: float, bandwidth: float | None = None, *,
                        beams: int = 1, pings: int = 1) -> float:
    """N behind one averaged value: (time window / time resolution) per beam and ping, times the
    beams and pings averaged. The window part follows [JR07] p. 429 (samples spaced by the
    resolution are independent); at least one sample per beam and ping. Ours, and only true if
    (a) the beams' footprints do not overlap (beams at least a beamwidth apart), and (b) the pings
    see different seafloor or decorrelated water: [JR07] pp. 422-424 warn that a still sonar's
    ping-to-ping changes come from the water column only and need not match a moving sonar's
    ensemble (the seabed itself decorrelates over about a day for roughness, a month for volume,
    p. 203); at the sea surface [DP97] found pings independent beyond O(10) ms at 50 kHz. Count only
    pings that satisfy (b); `effective_looks` gives the exact count for correlated samples."""
    per = max(1.0, float(duration) / time_resolution(pulse_length, bandwidth))
    if int(beams) < 1 or int(pings) < 1:
        raise ValueError("beams and pings must be at least 1")
    return per * int(beams) * int(pings)


def effective_looks(sample_times, rho_s) -> float:
    """Effective number of independent samples in an average of intensity samples at `sample_times`
    (s) of one ping: N^2 / sum_ij rho_s(t_i - t_j)^2, from the intensity covariance <|P|^2>^2 rho_s^2
    ([JR07] (16.26), McDaniel 1990; (16.22): B_PP = <|P|^2> rho_s, valid when the cross section
    varies by less than about 1 dB across the band). The ratio is the standard variance of a mean of
    correlated samples (ours to write down; the detector's effective-looks count, D-93/D-168, is the
    same formula). `rho_s`: the magnitude of the normalised autocorrelation of the baseband pulse, a
    function of the lag in seconds (`rho_cw`, `rho_lfm`)."""
    t = np.asarray(sample_times, dtype=float).ravel()
    lag = t[:, None] - t[None, :]
    r = np.abs(np.asarray(rho_s(lag), dtype=float))
    return float(len(t) ** 2 / np.sum(r**2))


def rho_cw(pulse_length: float):
    """|rho_s| of a rectangular CW pulse of length T: 1 - |tau| / T, zero beyond T (ours: the
    autocorrelation of a rectangle)."""
    T = float(pulse_length)
    return lambda tau: np.clip(1.0 - np.abs(tau) / T, 0.0, None)


def rho_lfm(pulse_length: float, bandwidth: float):
    """|rho_s| of a rectangular linear FM pulse, length T, sweep B: |(1 - |tau|/T) sinc(B tau (1 -
    |tau|/T))| (ours: the standard chirp ambiguity function on the zero-Doppler cut)."""
    T, B = float(pulse_length), float(bandwidth)

    def rho(tau):
        a = np.clip(1.0 - np.abs(tau) / T, 0.0, None)
        return np.abs(a * np.sinc(B * tau * a))
    return rho


def two_way_time(height: float, grazing, sound_speed: float = NOMINAL_SOUND_SPEED):
    """Two-way time to a flat boundary `height` below (or above) the sonar at grazing angle
    `grazing` (radians), straight paths: 2 h / (c sin g) (the slant range of (G.9), r_t = c t / 2)."""
    return 2.0 * abs(height) / (sound_speed * np.sin(np.asarray(grazing, dtype=float)))


# --- Backscattering strength from a calibrated sonar ([JR07] App. G.2.1) --------------------------

def psi_3(psi_tx: float, psi_rx: float) -> float:
    """3-dB full width (radians) of the product of two Gaussian power patterns of 3-dB full widths
    `psi_tx` and `psi_rx`: 1 / sqrt(1/psi_x^2 + 1/psi_r^2) ([JR07] (G.29), printed)."""
    return 1.0 / math.sqrt(1.0 / psi_tx**2 + 1.0 / psi_rx**2)


def psi_gaussian(grazing, psi_tx: float, psi_rx: float | None = None, tilt=None):
    """The azimuthal integral Psi of (G.12) for Gaussian azimuthal patterns ([JR07] (G.28), printed):

        Psi = cos(theta - theta_0) / (2 cos theta) sqrt(pi / ln 2) Psi_3,

    theta the grazing angle, theta_0 the beam's depression (tilt; default: theta, on axis), Psi_3
    from (G.29) (`psi_rx` None: the product pattern's width is `psi_tx`). The pattern widths are
    angles across the TILTED beam: a direction at depression theta and horizontal azimuth phi is
    phi cos(theta) / cos(theta - theta_0) off the beam's vertical plane, whence the angular factor.
    Psi ~ Psi_3 drops sqrt(pi / ln 2) / 2 (0.27 dB) and the factor. Valid for horizontal beamwidths
    of about 20 degrees or less ((G.24)-(G.27)); wider is warned about."""
    width = psi_tx if psi_rx is None else psi_3(psi_tx, psi_rx)
    if width > MAX_HORIZONTAL_BEAMWIDTH:
        warnings.warn(f"psi_gaussian: a beamwidth of {math.degrees(width):.1f} degrees is beyond the about 20 degrees "
                      "for which (G.28) holds ((G.24)-(G.27))", stacklevel=2)
    g = np.asarray(grazing, dtype=float)
    t0 = g if tilt is None else np.asarray(tilt, dtype=float)
    out = np.cos(g - t0) / (2.0 * np.cos(g)) * math.sqrt(math.pi / math.log(2.0)) * width
    return float(out) if np.ndim(out) == 0 else out


def insonified_area(range_m, pulse_length: float, psi, sound_speed: float = NOMINAL_SOUND_SPEED):
    """The boundary area behind one time sample at moderate to small grazing, r_t (c tau / 2) Psi
    ([JR07] (G.9)-(G.11): the strip is c tau / (2 cos theta) wide and r cos theta long per radian, the
    cosines cancel; `sim/reverb.k_cell_area` is the same area with Psi a sharp sector)."""
    return np.asarray(range_m, dtype=float) * 0.5 * sound_speed * pulse_length * np.asarray(psi, dtype=float)


def _check_resolution(range_m, pulse_length, sound_speed):
    ratio = 0.5 * sound_speed * pulse_length / np.min(np.asarray(range_m, dtype=float))
    if ratio > MAX_RANGE_RESOLUTION_RATIO:
        warnings.warn(f"(G.13) assumes c tau / 2 << r_t ((G.9)); here c tau / (2 r) = {ratio:.2f}: the strip "
                      "approximation fails near the nadir and at short range", stacklevel=3)


def reverberation_level_db(strength_db, range_m, pulse_length: float, psi, *, source_level_db: float = 0.0,
                           receive_sensitivity_db: float = 0.0, tx_directivity_db=0.0, rx_directivity_db=0.0,
                           absorption_db_per_m: float = 0.0, sound_speed: float = NOMINAL_SOUND_SPEED,
                           reference_range: float = 1.0):
    """The monostatic seafloor reverberation level 10 log10(<|V_r|^2> / 2) of (G.13), corrected:

        L = SL + RS + S_b + D_x(theta, 0) + D_r(theta, 0) - 2 alpha_w r_t - 30 log10(r_t / r_ref)
            + 10 log10(c tau Psi / (2 r_ref))

    with alpha_w the WATER absorption in dB/m (one way). Misprint: (G.13) prints "-2 alpha_p r_t", the
    sediment's attenuation symbol; the water's is meant, consistent with exp(-4 k_w'' r_t) in (G.11)
    (note 03 Sec. 2.9 and Sec. 6). The pulse window behind r_t is r_t - c tau / 4 < r < r_t + c tau / 4
    ((G.6)-(G.7) corrected: both bounds are printed with a minus sign; the upper one is a plus, same
    sources). D_x, D_r are the transmit and receive pattern gains in dB at the grazing angle on the
    beam's azimuth; SL dB re 1 uPa at 1 m, RS dB re 1 V/uPa: then L is dB re 1 V^2."""
    _check_resolution(range_m, pulse_length, sound_speed)
    r = np.asarray(range_m, dtype=float)
    return (source_level_db + receive_sensitivity_db + np.asarray(strength_db, dtype=float) + tx_directivity_db
            + rx_directivity_db - 2.0 * absorption_db_per_m * r - 30.0 * np.log10(r / reference_range)
            + 10.0 * np.log10(sound_speed * pulse_length * np.asarray(psi, dtype=float) / (2.0 * reference_range)))


def backscattering_strength_db(level_db, range_m, pulse_length: float, psi, *, source_level_db: float = 0.0,
                               receive_sensitivity_db: float = 0.0, tx_directivity_db=0.0, rx_directivity_db=0.0,
                               absorption_db_per_m: float = 0.0, sound_speed: float = NOMINAL_SOUND_SPEED,
                               reference_range: float = 1.0):
    """S_b (dB re 1 m^2 per m^2) from a calibrated mean reverberation level `level_db` =
    10 log10(<|V_r|^2> / 2) at slant range `range_m` = c t / 2: (G.13) solved for S_b (with the
    corrections of `reverberation_level_db`). `level_db` must be a mean of INTENSITY over the
    independent samples (`intensity_average_db`), not of dB. Moderate to small grazing (c tau / 2 <<
    r_t, warned otherwise); only the main-lobe return (Fig. G.4: not the fathometer or sidelobe
    returns). `psi` from `psi_gaussian` or a numerical (G.12) of the measured patterns."""
    zero = reverberation_level_db(0.0, range_m, pulse_length, psi, source_level_db=source_level_db,
                                  receive_sensitivity_db=receive_sensitivity_db, tx_directivity_db=tx_directivity_db,
                                  rx_directivity_db=rx_directivity_db, absorption_db_per_m=absorption_db_per_m,
                                  sound_speed=sound_speed, reference_range=reference_range)
    return np.asarray(level_db, dtype=float) - zero
