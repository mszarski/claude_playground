"""Medium properties: sound speed and absorption.

Implemented from the published formulas (no code copied):

- Francois & Garrison (1982), J. Acoust. Soc. Am. 72, 896-907 and 1879-1890.
- Ainslie & McColm (1998), J. Acoust. Soc. Am. 103, 1671-1672.
- Mackenzie (1981), J. Acoust. Soc. Am. 70, 807-812 (sea water sound speed).
- Marczak (1997), J. Acoust. Soc. Am. 102, 2776-2779 (pure water sound speed).

`arlpy.uwa` is the test oracle where it covers the same formula.
"""

from __future__ import annotations

import numpy as np


def sound_speed(temperature: float, salinity: float = 35.0, depth: float = 0.0) -> float:
    """Sound speed in m/s. Mackenzie for sea water, Marczak for fresh water (S < 5 PSU)."""
    t, s, d = float(temperature), float(salinity), float(depth)
    if s < 5.0:
        c = (1.402385e3 + 5.038813 * t - 5.799136e-2 * t**2 + 3.287156e-4 * t**3
             - 1.398845e-6 * t**4 + 2.787860e-9 * t**5)
        # Small corrections so brackish water and depth are not ignored outright.
        return c + 1.340 * s + 1.630e-2 * d
    return (1448.96 + 4.591 * t - 5.304e-2 * t**2 + 2.374e-4 * t**3 + 1.340 * (s - 35.0)
            + 1.630e-2 * d + 1.675e-7 * d**2 - 1.025e-2 * t * (s - 35.0) - 7.139e-13 * t * d**3)


def absorption_francois_garrison(frequency, temperature=10.0, salinity=35.0, depth=0.0, ph=8.0):
    """Absorption in dB/m at `frequency` (Hz, scalar or array). Valid 100 Hz to 1 MHz."""
    f = np.asarray(frequency, dtype=float) / 1e3  # kHz
    t, s, d = float(temperature), float(salinity), float(depth)
    c = 1412.0 + 3.21 * t + 1.19 * s + 0.0167 * d
    theta = t + 273.0
    # Boric acid
    a1 = 8.86 / c * 10.0 ** (0.78 * ph - 5.0)
    f1 = 2.8 * np.sqrt(s / 35.0) * 10.0 ** (4.0 - 1245.0 / theta)
    # Magnesium sulphate
    a2 = 21.44 * s / c * (1.0 + 0.025 * t)
    p2 = 1.0 - 1.37e-4 * d + 6.2e-9 * d**2
    f2 = 8.17 * 10.0 ** (8.0 - 1990.0 / theta) / (1.0 + 0.0018 * (s - 35.0))
    # Pure water
    if t <= 20.0:
        a3 = 4.937e-4 - 2.59e-5 * t + 9.11e-7 * t**2 - 1.50e-8 * t**3
    else:
        a3 = 3.964e-4 - 1.146e-5 * t + 1.45e-7 * t**2 - 6.5e-10 * t**3
    p3 = 1.0 - 3.83e-5 * d + 4.9e-10 * d**2
    boric = a1 * f1 * f**2 / (f1**2 + f**2) if s > 0.0 else 0.0
    mgso4 = a2 * p2 * f2 * f**2 / (f2**2 + f**2)
    water = a3 * p3 * f**2
    return (boric + mgso4 + water) / 1e3  # dB/km -> dB/m


def absorption_ainslie_mccolm(frequency, temperature=10.0, salinity=35.0, depth=0.0, ph=8.0):
    """Absorption in dB/m at `frequency` (Hz). A simpler fit to the same data."""
    f = np.asarray(frequency, dtype=float) / 1e3
    t, s, z = float(temperature), float(salinity), float(depth) / 1e3
    f1 = 0.78 * np.sqrt(s / 35.0) * np.exp(t / 26.0)
    f2 = 42.0 * np.exp(t / 17.0)
    boric = 0.106 * f1 * f**2 / (f1**2 + f**2) * np.exp((ph - 8.0) / 0.56) if s > 0.0 else 0.0
    mgso4 = 0.52 * (1.0 + t / 43.0) * (s / 35.0) * f2 * f**2 / (f2**2 + f**2) * np.exp(-z / 6.0)
    water = 0.00049 * f**2 * np.exp(-(t / 27.0 + z / 17.0))
    return (boric + mgso4 + water) / 1e3


_MODELS = {
    "francois_garrison": absorption_francois_garrison,
    "ainslie_mccolm": absorption_ainslie_mccolm,
}


def absorption(frequency, temperature=10.0, salinity=35.0, depth=0.0, ph=8.0,
               model: str = "francois_garrison"):
    """Absorption in dB/m. `model` is `francois_garrison`, `ainslie_mccolm` or `none`."""
    if model == "none":
        return np.zeros_like(np.asarray(frequency, dtype=float))
    try:
        fn = _MODELS[model]
    except KeyError as e:
        raise ValueError(f"unknown absorption model {model!r}") from e
    return fn(frequency, temperature, salinity, depth, ph)


# --- sea surface ---------------------------------------------------------------------------------

KNOTS_PER_MS = 1.943844


def chapman_harris_db(grazing, wind_speed: float, frequency: float):
    """Sea-surface backscattering strength in dB (Chapman and Harris, JASA 34, 1962).

    S = 3.3 beta log10(theta / 30) - 42.4 log10(beta) + 2.6,
    beta = 158 (v f^(1/3))^(-0.58), theta the grazing angle in degrees, v the wind speed in
    knots, f in Hz. `grazing` is in radians and `wind_speed` in m/s here. The fit was made
    to explosive-source data from 400 to 6400 Hz and grazing angles below about 40
    degrees; using it at 8 kHz and above is an extrapolation.
    """
    theta = np.degrees(np.maximum(np.asarray(grazing, dtype=float), 1e-3))
    v = max(float(wind_speed), 0.1) * KNOTS_PER_MS
    beta = 158.0 * (v * float(frequency) ** (1.0 / 3.0)) ** -0.58
    return 3.3 * beta * np.log10(theta / 30.0) - 42.4 * np.log10(beta) + 2.6


def rms_wave_height(wind_speed: float) -> float:
    """RMS surface displacement in m of a fully developed sea (Pierson-Moskowitz):
    significant height 0.21 U^2 / g, and rms displacement a quarter of that. U is strictly
    the wind at 19.5 m; the difference from the usual 10 m wind is ignored."""
    return 0.25 * 0.21 * float(wind_speed) ** 2 / 9.81


def surface_coherent_reflection(grazing, wind_speed: float, frequency: float, sound_speed: float = 1500.0):
    """Amplitude of the coherent (specular) reflection from a rough surface relative to a
    flat one: exp(-Gamma^2 / 2), Gamma = 2 k h sin(grazing) the Rayleigh parameter.

    This removes all scattered energy from the forward path, so it overstates the loss
    when Gamma is large (much of the scattered sound still travels down range).
    """
    k = 2.0 * np.pi * float(frequency) / float(sound_speed)
    gamma = 2.0 * k * rms_wave_height(wind_speed) * np.sin(np.asarray(grazing, dtype=float))
    return np.exp(-0.5 * gamma**2)
