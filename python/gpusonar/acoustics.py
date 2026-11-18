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


YARDS_PER_KM = 1000.0 / 0.9144
THORP_MAX_FREQUENCY = 100.0e3           # Hz, warn above: see `absorption_thorp`


def absorption_thorp(frequency, temperature=10.0, salinity=35.0, depth=0.0, ph=8.0):
    """Absorption in dB/m at `frequency` (Hz) by the two-term Thorp expression that SST uses:

        a [dB/kyd] = 0.1 f^2 / (1 + f^2) + 40 f^2 / (4100 + f^2),   f in kHz

    (Goddard, APL-UW TR 0702, eq. (42): "the expression in [Urick 1983] page 108", with "the
    last two terms ... omitted, to match the implementation in CASS"; our notes give its stated
    validity as 4 deg C and about 1 km depth). It has no temperature, salinity, depth or pH
    dependence: those arguments are accepted and ignored. Here for like-for-like comparison
    with SST runs; Francois-Garrison remains the default.

    A low-frequency formula: without the omitted pure-water term it levels off at 40.1 dB/kyd
    (43.9 dB/km). Against Francois-Garrison at 10 deg C it gives 0.27 vs 0.24 dB/km at 4 kHz,
    1.15 vs 0.96 at 10 kHz, 31 vs 34 at 100 kHz and 44 vs 222 at 750 kHz, so a warning is
    raised above `THORP_MAX_FREQUENCY` (our limit, not the source's)."""
    if np.max(frequency) > THORP_MAX_FREQUENCY:
        import warnings
        warnings.warn(f"absorption_thorp: above {THORP_MAX_FREQUENCY / 1e3:.0f} kHz the two-term Thorp expression lacks the "
                      "pure-water term and is far too low; use francois_garrison", stacklevel=2)
    f2 = (np.asarray(frequency, dtype=float) / 1e3) ** 2
    return (0.1 * f2 / (1.0 + f2) + 40.0 * f2 / (4100.0 + f2)) * YARDS_PER_KM / 1e6   # dB/kyd -> dB/km -> dB/m


_MODELS = {
    "francois_garrison": absorption_francois_garrison,
    "ainslie_mccolm": absorption_ainslie_mccolm,
    "thorp": absorption_thorp,
}


def absorption(frequency, temperature=10.0, salinity=35.0, depth=0.0, ph=8.0,
               model: str = "francois_garrison"):
    """Absorption in dB/m. `model` is `francois_garrison`, `ainslie_mccolm`, `thorp` or `none`."""
    if model == "none":
        return np.zeros_like(np.asarray(frequency, dtype=float))
    try:
        fn = _MODELS[model]
    except KeyError as e:
        raise ValueError(f"unknown absorption model {model!r}") from e
    return fn(frequency, temperature, salinity, depth, ph)


def water_density(temperature: float = 10.0, salinity: float = 35.0) -> float:
    """Density of sea water in kg/m3 at atmospheric pressure, from temperature (deg C) and
    salinity (PSU): the one-atmosphere polynomial

        rho = A(t) + S (B(t) + C(t) sqrt(S) + D S)

    with the coefficients printed in arlpy (BSD 3-clause), arlpy/uwa.py:101-121, and again in
    UnderwaterAcoustics.jl (MIT), src/basic.jl:111-121, both of which attribute it to
    "Fofonoff (1985 - IES 80)". Neither has a pressure term, so neither has this: it is the
    density near the surface. Their documented value: 1022.7 kg/m3 at 27 deg C and 35 ppt."""
    t, s = float(temperature), float(salinity)
    a = 999.842594 + t * (6.793952e-02 + t * (-9.095290e-03 + t * (1.001685e-04 + t * (-1.120083e-06 + t * 6.536332e-09))))
    b = 0.824493 + t * (-4.0899e-03 + t * (7.6438e-05 + t * (-8.2467e-07 + t * 5.3875e-09)))
    c = -5.72466e-03 + t * (1.0227e-04 - t * 1.6546e-06)
    return a + s * (b + c * np.sqrt(s) + 4.8314e-04 * s)


def bubbly_sound_speed(void_fraction, sound_speed: float = 1500.0, gas_sound_speed: float = 340.0,
                       density_ratio: float = 1000.0):
    """Sound speed (m/s) of water holding a volume fraction `void_fraction` of gas bubbles, at
    frequencies well below the bubbles' resonance (the mixture then has the mean density and
    the mean compressibility of its parts: Wood's equation). `density_ratio` is water density /
    gas density. Written as in arlpy (BSD 3-clause), arlpy/uwa.py:215-231 (`bubble_soundspeed`,
    "based on Wood (1964) or Buckingham (1997)"; the same expression is in UnderwaterAcoustics.jl,
    src/basic.jl:30-33):

        c_mix = c / sqrt( (g (c / c_gas)^2 m + (1 - g) / m) (g / m + (1 - g) m) ),  m = sqrt(density_ratio)

    which is 1 / sqrt(mean density x mean compressibility) with the water density divided out
    (a test checks that identity). g = 0 gives c, g = 1 gives c_gas, and in between the speed
    falls far below both: about 2 c_gas / m near g = 1/2."""
    g = np.asarray(void_fraction, dtype=float)
    c, m = float(sound_speed), np.sqrt(float(density_ratio))
    out = c / np.sqrt((g * (c / float(gas_sound_speed)) ** 2 * m + (1.0 - g) / m) * (g / m + (1.0 - g) * m))
    return float(out) if out.ndim == 0 else out


# --- sea surface ---------------------------------------------------------------------------------

SEA_STATE_WIND_SPEED = (0.8, 2.6, 4.4, 6.9, 9.8, 12.6, 19.3, 26.5, 30.6, 32.9)
"""Wind speed in m/s for sea states 0 to 9, as printed in UnderwaterAcoustics.jl (MIT),
src/stdlib.jl:411-422 (`SeaState0` ... `SeaState9`), whose comment reads "WMO sea states from
APL-UW TR 9407 (1994), II-4 Table 2 (median windspeed)". TR 9407 was not read."""


def sea_state_wind_speed(sea_state) -> float:
    """Median wind speed (m/s) of a sea state 0 to 9 (`SEA_STATE_WIND_SPEED`). Whole states only:
    the source tabulates ten values and gives no rule between them."""
    n = float(sea_state)
    if n != int(n) or not 0 <= int(n) <= 9:
        raise ValueError(f"sea_state must be a whole number from 0 to 9, not {sea_state!r}")
    return SEA_STATE_WIND_SPEED[int(n)]


def resolve_sea_state(config, _where: str = "scenario"):
    """A copy of a scenario configuration in which every mapping that has `sea_state` has
    `wind_speed` instead (`sea_state_wind_speed`), at any depth. So `sea_state` is accepted
    wherever `wind_speed` is. Giving both in one mapping is an error unless they agree."""
    if isinstance(config, list):
        return [resolve_sea_state(v, f"{_where}[{i}]") for i, v in enumerate(config)]
    if not isinstance(config, dict):
        return config
    out = {k: resolve_sea_state(v, f"{_where}.{k}") for k, v in config.items() if k != "sea_state"}
    if "sea_state" in config:
        wind = sea_state_wind_speed(config["sea_state"])
        if "wind_speed" in config and abs(float(config["wind_speed"]) - wind) > 1e-9:
            raise ValueError(f"{_where}: sea_state {config['sea_state']} means a wind speed of {wind} m/s, but "
                             f"wind_speed is {config['wind_speed']}; give one of them")
        out["wind_speed"] = wind
    return out


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


def mean_square_slope(wind_speed: float) -> float:
    """Mean-square slope of the wind-roughened sea surface (both axes together), Cox and Munk
    (1954), clean surface: 0.003 + 0.00512 U, U in m/s at 12.5 m (taken as the 10 m wind)."""
    return 0.003 + 0.00512 * max(float(wind_speed), 0.0)


def mean_square_slope_components(wind_speed: float) -> tuple[float, float]:
    """(upwind, crosswind) mean-square slopes of the clean sea surface, Cox and Munk (1954) as
    printed in Guerin, Capelle and Hartmann, "Revisiting the Cox and Munk wave-slope statistics
    using IASI observations of the sea surface", arXiv:2210.05456v2, eq. (12):

        m_u = 3.16e-3 U,    m_c = 3e-3 + 1.92e-3 U,

    U in m/s at a mast height of 12.5 m, for 1 to 14 m/s. Their sum, 0.003 + 0.00508 U, is 0.8 %
    below the separate regression of the total that `mean_square_slope` carries (0.00512 U,
    not printed in the opened source; the two published fits differ by that much). The
    surface is steeper along the wind than across it by a factor 1.2 to 1.5 over that range;
    the time-spreading model takes the slopes as isotropic."""
    u = max(float(wind_speed), 0.0)
    return 3.16e-3 * u, 3.0e-3 + 1.92e-3 * u


def rms_surface_vertical_speed(wind_speed: float) -> float:
    """RMS vertical speed in m/s of the surface of a fully developed sea: the square root of the
    second moment of the Pierson-Moskowitz frequency spectrum
    S(w) = a g^2 w^-5 exp(-b (g / (U w))^4), a = 0.0081, b = 0.74, which integrates to
    m2 = a sqrt(pi) U^2 / (4 sqrt(b)). (The zeroth moment gives `rms_wave_height`.)"""
    a, b = 0.0081, 0.74
    return float(np.sqrt(a * np.sqrt(np.pi) / (4.0 * np.sqrt(b))) * max(float(wind_speed), 0.0))


def rough_coherent_reflection(grazing, rms_height: float, frequency: float, sound_speed: float = 1500.0):
    """Amplitude of the coherent reflection from a boundary of given rms height relative to a
    flat one: exp(-Gamma^2 / 2), Gamma = 2 k h sin(grazing)."""
    k = 2.0 * np.pi * float(frequency) / float(sound_speed)
    gamma = 2.0 * k * float(rms_height) * np.sin(np.asarray(grazing, dtype=float))
    return np.exp(-0.5 * gamma**2)


def surface_coherent_reflection(grazing, wind_speed: float, frequency: float, sound_speed: float = 1500.0):
    """Amplitude of the coherent (specular) reflection from a rough surface relative to a
    flat one: exp(-Gamma^2 / 2), Gamma = 2 k h sin(grazing) the Rayleigh parameter.

    This removes all scattered energy from the forward path, so it overstates the loss
    when Gamma is large (much of the scattered sound still travels down range).
    """
    k = 2.0 * np.pi * float(frequency) / float(sound_speed)
    gamma = 2.0 * k * rms_wave_height(wind_speed) * np.sin(np.asarray(grazing, dtype=float))
    return np.exp(-0.5 * gamma**2)
