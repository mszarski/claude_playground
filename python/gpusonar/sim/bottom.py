"""Scattering strength of a rough fluid sediment: interface roughness plus sediment volume.

This is the kind of bottom model SST points to (`JacksonBottom`, `APLBottom`: a fluid sediment
with a power-law roughness spectrum and volume inhomogeneities, six parameters). The APL-UW
handbook that defines those models (TR 9407) was not available, so nothing here is copied from
it: both terms are derived below from the wave equation, and tests check them against limits
with known answers. What is NOT here, and is in the published models: the Kirchhoff
(near-vertical) regime, the composite-roughness correction for large-scale slopes, and any
table of named sediments. Use below about 60 degrees of grazing and for k x rms relief (at
scales near the acoustic wavelength) below about one.

Parameters (dimensionless unless noted), as in the published models:
  rho     sediment density / water density
  nu      sediment sound speed / water sound speed
  delta   loss parameter: imaginary / real part of the sediment wavenumber
  w2      roughness spectral strength, m^(4 - gamma2):  W(K) = w2 / K^gamma2,  K in rad/m, with
          the relief zeta(R) = integral of zeta(K) exp(i K.R) d2K and <|zeta|^2> = integral of W d2K
  gamma2  roughness spectral exponent (between 2 and 4)
  sigma2  sediment volume scattering cross section per unit volume divided by the sediment's
          attenuation coefficient in dB/m

Roughness term. Water (wavenumber k, density 1) above a fluid (wavenumber k2 = k (1 + i delta) / nu,
density rho), interface at z = zeta(R). A plane wave arrives with horizontal wavenumber Ki. To
first order in zeta the boundary conditions (pressure and (1/density) x normal derivative
continuous at the displaced interface) are moved to z = 0:

    A - B                           = i g (1 - rho) zeta(Q)
    (i b1s) A + (i b2s / rho) B     = T [ b1i^2 - b2i^2 / rho - (1 - 1/rho) Q.Ki ] zeta(Q)

for the scattered wave amplitudes A (water, upward) and B (sediment, downward) at Ks = Ki + Q,
where b1 = sqrt(k^2 - K^2), b2 = sqrt(k2^2 - K^2) are vertical wavenumbers (i: incident, s:
scattered), T = 1 + V is the flat-interface pressure transmission and g = b1i (1 - V). With
A = a zeta(Q), the scattering cross section per unit area and solid angle is

    sigma_r = b1s^2 |a|^2 W(Q).

Limits: rho -> 0 gives 4 k^4 sin^2(ti) sin^2(ts) W (pressure release); rho -> infinity gives
backscatter 4 k^4 (1 + cos^2 t)^2 W (rigid); no contrast gives zero.

Volume term. Scatterers in the sediment with cross section per unit volume sigma_vol see the
transmitted field |T(ti)|^2 exp(-2 Im(b2i) z); their scattered sound returns through the
interface, which by reciprocity costs (1/rho^2) |T(ts)|^2 exp(-2 Im(b2s) z). Integrated over depth:

    sigma_v = sigma_vol |T(ti)|^2 |T(ts)|^2 / ( rho^2 * 2 (Im b2i + Im b2s) ),

with sigma_vol = sigma2 x attenuation in dB/m = sigma2 x (20 / ln 10) k delta / nu.
"""
from __future__ import annotations

import numpy as np

PLACEHOLDER_SAND = {"rho": 1.9, "nu": 1.1, "delta": 0.0147, "w2": 2.0e-5, "gamma2": 3.25, "sigma2": 0.002}
"""Illustrative values only, NOT a measured or published sediment: the density, speed and loss
are those the scenarios already use for their sand bottom (1.9, 1650 m/s, 0.8 dB per wavelength);
the roughness and volume numbers were chosen to give a backscattering strength near -30 dB at
20 degrees and 100 kHz. Replace with values for the site."""


def _vertical(k2, horizontal):
    b = np.sqrt(k2 - horizontal**2 + 0j)
    return np.where(b.imag < 0.0, -b, b)          # decays away from the interface


def reflection(grazing, rho: float, nu: float, delta: float):
    """Flat-interface pressure reflection coefficient V of the fluid sediment at a grazing angle."""
    g = np.asarray(grazing, dtype=float)
    kappa = (1.0 + 1j * delta) / nu
    p = _vertical(kappa**2, np.cos(g))
    return (rho * np.sin(g) - p) / (rho * np.sin(g) + p)


def roughness_strength(grazing_in, grazing_out, frequency: float, sound_speed: float, *, rho: float, nu: float,
                       delta: float, w2: float, gamma2: float, azimuth=np.pi, **_ignored):
    """Cross section per unit area and solid angle from interface roughness (first-order
    perturbation theory). `azimuth` is the angle between the horizontal directions of the
    incident and scattered waves: pi for backscatter, 0 for forward scatter."""
    ti, ts, phi = np.broadcast_arrays(np.asarray(grazing_in, dtype=float), np.asarray(grazing_out, dtype=float),
                                      np.asarray(azimuth, dtype=float))
    k = 2.0 * np.pi * float(frequency) / float(sound_speed)
    k2sq = (k * (1.0 + 1j * delta) / nu) ** 2
    ki, ks = k * np.cos(ti), k * np.cos(ts)                       # horizontal wavenumbers
    b1i, b1s = k * np.sin(ti), k * np.sin(ts)
    b2i, b2s = _vertical(k2sq, ki), _vertical(k2sq, ks)
    v = (rho * b1i - b2i) / (rho * b1i + b2i)
    t, g = 1.0 + v, b1i * (1.0 - v)
    q_dot_ki = ki * ks * np.cos(phi) - ki**2                      # Q = Ks - Ki
    q = np.sqrt(np.maximum(ki**2 + ks**2 - 2.0 * ki * ks * np.cos(phi), 0.0))
    rhs1 = 1j * g * (1.0 - rho)
    rhs2 = t * (b1i**2 - b2i**2 / rho - (1.0 - 1.0 / rho) * q_dot_ki)
    # A - B = rhs1;  i b1s A + (i b2s / rho) B = rhs2   ->   A
    a = (rhs2 + (1j * b2s / rho) * rhs1) / (1j * b1s + 1j * b2s / rho)
    with np.errstate(divide="ignore"):
        spectrum = np.where(q > 0.0, w2 / np.maximum(q, 1e-300) ** gamma2, 0.0)
    return b1s**2 * np.abs(a) ** 2 * spectrum


def volume_strength(grazing_in, grazing_out, frequency: float, sound_speed: float, *, rho: float, nu: float,
                    delta: float, sigma2: float, **_ignored):
    """Cross section per unit area and solid angle from scatterers inside the sediment."""
    ti, ts = np.broadcast_arrays(np.asarray(grazing_in, dtype=float), np.asarray(grazing_out, dtype=float))
    k = 2.0 * np.pi * float(frequency) / float(sound_speed)
    k2sq = (k * (1.0 + 1j * delta) / nu) ** 2
    b2i, b2s = _vertical(k2sq, k * np.cos(ti)), _vertical(k2sq, k * np.cos(ts))
    t_in = 1.0 + reflection(ti, rho, nu, delta)
    t_out = 1.0 + reflection(ts, rho, nu, delta)
    sigma_vol = sigma2 * (20.0 / np.log(10.0)) * k * delta / nu   # 1/m
    decay = 2.0 * (b2i.imag + b2s.imag)
    return np.where(decay > 0.0, sigma_vol * np.abs(t_in) ** 2 * np.abs(t_out) ** 2 / (rho**2 * np.maximum(decay, 1e-300)), 0.0)


def strength(grazing_in, grazing_out, frequency: float, sound_speed: float, azimuth=np.pi, **params):
    """Roughness plus volume."""
    return (roughness_strength(grazing_in, grazing_out, frequency, sound_speed, azimuth=azimuth, **params)
            + volume_strength(grazing_in, grazing_out, frequency, sound_speed, **params))


def backscattering_strength_db(grazing, frequency: float, sound_speed: float = 1500.0, **params):
    return 10.0 * np.log10(np.maximum(strength(grazing, grazing, frequency, sound_speed, **params), 1e-300))


def reverberation_strength(frequency: float, sound_speed: float, **params):
    """The model as the function of (grazing in, grazing out) the reverberation code takes
    (backscatter geometry: the scattered wave returns over the incident one)."""
    return lambda g_in, g_out: strength(g_in, g_out, frequency, sound_speed, **params)


def equivalent_lambert_mu_db(grazing: float, frequency: float, sound_speed: float = 1500.0, **params) -> float:
    """The Lambert parameter that gives this model's backscattering strength at one grazing
    angle: for scenes whose seabed is point scatterers with Lambert's law (set their `mu_db`
    to this at the middle of the swath)."""
    return float(backscattering_strength_db(grazing, frequency, sound_speed, **params) - 20.0 * np.log10(np.sin(grazing)))
