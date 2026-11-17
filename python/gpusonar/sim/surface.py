"""Sea-surface models that depend on the wind: bubble loss per bounce and backscattering strength.

Substitutes, from the open literature, for what SST takes from the limited-distribution
APL-UW handbook (TR 9407), which was not read. Two things are here.

1. Bubble loss per surface bounce (`bubble_loss_db`), from the Hall-Novarini background
   bubble population and resonant-bubble extinction.

   Population (opened: Nennouche, Khalighi, Dowhuszko and Merad, arXiv:2603.23727v2, 2026,
   eqs. (20)-(23), who take it from Keiffer, Novarini and Norton, JASA 97, 227-234, 1995): bubbles per unit volume and unit radius

       n(r, z) = 1.6e10 G(r, z) (U10 / 13)^3 exp(-z / L(U10))                  [m^-4]
       L = 0.4 m for U10 <= 7.5 m/s, else 0.4 + 0.115 (U10 - 7.5) m
       G = (r_ref / r)^4 for r_min <= r <= r_ref,  (r_ref / r)^kappa for r_ref < r <= r_max
       kappa = 4.37 + (z / 2.55)^2,   r_ref = 54.4e-6 + 1.984e-6 z  m
       r_min = 10 micrometres, r_max = 1 mm

   with r and z in metres and U10 the wind at 10 m in m/s. (The source does not print the unit
   of n; m^-4 is the unit that makes its eqs. (19) and (24), a bubble count per m^3 from
   radii in metres, consistent.)

   Extinction (our derivation; the standard resonant-bubble approximation). A bubble of radius
   a much smaller than the wavelength is a damped oscillator with resonance at the Minnaert
   frequency, f_R = sqrt(3 g P / rho) / (2 pi a), g = 1.4 the ratio of specific heats of air
   and P = P_atm + rho g_0 z the static pressure. Its scattering cross section is
   4 pi a^2 / ((f_R^2 / f^2 - 1)^2 + d^2) and its extinction cross section is that times
   d / (k a), because of the total damping d only the part k a is re-radiation. If n varies
   little across the resonance, integrating over radius gives an extinction coefficient that
   does not depend on d:

       beta(z) = 2 pi^2 a_R^2 n(a_R, z) / k    [1/m, intensity],  a_R the resonant radius.

   A surface bounce at grazing angle t crosses the layer twice along a straight path:

       loss = (10 / ln 10) * 2 / sin(t) * integral_0^inf beta(z) dz    [dB].

   Not included: bubbles off resonance, refraction by the bubbly layer (it bends rays towards
   the normal, so the loss at small grazing angles is overstated; the angle is floored at
   `MIN_GRAZING`), plumes under breaking waves, multiple scattering. Valid range: resonant
   radius inside the population's 10 micrometres to 1 mm, i.e. about 3.3 to 330 kHz; outside
   it the loss is zero and a warning is raised. The source gives no wind-speed range; the law
   is a cubic in U10 and is used here up to `MAX_WIND` with a warning above.

   How far to trust it. Keiffer et al. built the population for "the low to moderate frequency
   range" (their title). `tools/boundary_models_study.py` compares the loss with the law that
   the BSD library arlpy implements (`arlpy.uwa.bubble_surface_loss`, "APL model (1994)"; its
   formula was not read or copied here, only called as an oracle): at 10 m/s and 10 degrees
   ours is about 2.3 times the oracle's dB at 10 kHz, 4.4 times at 30 kHz and 7.5 times at
   100 kHz. So this law grows too fast with frequency; a warning is raised above
   `TRUSTED_BUBBLE_FREQUENCY`. Use a measured table where one exists.

2. Surface backscattering strength at high frequency (`kraus_surface_db`), an empirical
   formula in wind speed, frequency and grazing angle (opened: Morency and Stilwell,
   arXiv:2210.06535, 2022, Sec. II-F, quoting D. Kraus, "Fundamentals of ocean acoustics", lecture
   notes, Hochschule Bremen):

       SS = 10 log10( 10^-5.05 (1 + v)^2 (f + 0.1)^(v / 150) tan^b(t) )
       b  = 4 (v + 2) / (v + 1) + (2.5 (f + 0.1)^(-1/3) - 4) cos^(1/8)(t)

   v the wind speed in knots, f in kHz, t the grazing angle. The opened source states no
   range of validity; it uses the formula in a forward-looking sonar simulator.
   Treat it as an alternative to Chapman-Harris above about 10 kHz and check it against
   data for the site. A warning is raised below `KRAUS_MIN_FREQUENCY`, where Chapman-Harris
   is the fitted law. tan^b diverges at normal incidence, so above `KRAUS_MAX_GRAZING` the
   value at that angle is held (our guard, not part of the formula).

The same source gives an empirical bottom law by bottom type (`searay_bottom_db`), kept here
with the surface law it came with:

       SB = 10 log10( 3.03 b f^(3.2 - 0.8 bt) 10^(2.8 bt - 12) + 10^-4.42 )
       b  = g (sin t + 0.19)^(bt cos^16 t),   g = 1 + 125 exp(-2.64 (bt - 1.75)^2 - (50 / bt) cot^2 t)

   bt = 1 mud, 2 sand, 3 gravel, 4 rock; f in kHz.
"""
from __future__ import annotations

import math
import warnings

import numpy as np

from ..acoustics import KNOTS_PER_MS

R_MIN, R_MAX = 10.0e-6, 1.0e-3          # m, radii covered by the Hall-Novarini population
MIN_GRAZING = np.radians(2.0)           # floor on the grazing angle in the slant-path factor
MAX_WIND = 20.0                         # m/s, warn above (no range is given in the opened source)
KRAUS_MIN_FREQUENCY = 10.0e3            # Hz, warn below
KRAUS_MAX_GRAZING = np.radians(60.0)    # the tan^b form diverges at normal incidence: held constant above (our guard)
TRUSTED_BUBBLE_FREQUENCY = 10.0e3       # Hz, warn above: see `bubble_loss_db`
ATMOSPHERE, WATER_DENSITY, GRAVITY, AIR_GAMMA = 101325.0, 1025.0, 9.81, 1.4
BOTTOM_TYPES = {"mud": 1.0, "sand": 2.0, "gravel": 3.0, "rock": 4.0}


def hall_novarini_density(radius, depth, wind_speed: float):
    """Bubbles per m^3 per metre of radius at `depth` (m) for a 10 m wind speed (m/s)."""
    r, z = np.broadcast_arrays(np.asarray(radius, dtype=float), np.asarray(depth, dtype=float))
    u = max(float(wind_speed), 0.0)
    scale = 0.4 if u <= 7.5 else 0.4 + 0.115 * (u - 7.5)
    r_ref = 54.4e-6 + 1.984e-6 * z
    kappa = 4.37 + (z / 2.55) ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        g = np.where(r <= r_ref, (r_ref / r) ** 4, (r_ref / r) ** kappa)
    inside = (r >= R_MIN) & (r <= R_MAX) & (z >= 0.0)
    return np.where(inside, 1.6e10 * g * (u / 13.0) ** 3 * np.exp(-np.maximum(z, 0.0) / scale), 0.0)


def resonant_radius(frequency: float, depth=0.0):
    """Radius (m) of the air bubble resonant at `frequency` (Hz) at `depth` (m): Minnaert."""
    pressure = ATMOSPHERE + WATER_DENSITY * GRAVITY * np.asarray(depth, dtype=float)
    return np.sqrt(3.0 * AIR_GAMMA * pressure / WATER_DENSITY) / (2.0 * np.pi * float(frequency))


def bubble_extinction(frequency: float, depth, wind_speed: float, sound_speed: float = 1500.0):
    """Extinction coefficient of the bubble layer, 1/m of intensity, at `depth`."""
    k = 2.0 * np.pi * float(frequency) / float(sound_speed)
    a = resonant_radius(frequency, depth)
    return 2.0 * np.pi**2 * a**2 * hall_novarini_density(a, depth, wind_speed) / k


def bubble_layer_depth_integral(frequency: float, wind_speed: float, sound_speed: float = 1500.0) -> float:
    """Integral of the extinction coefficient over depth (one vertical crossing, nepers of intensity)."""
    u = max(float(wind_speed), 0.0)
    scale = 0.4 if u <= 7.5 else 0.4 + 0.115 * (u - 7.5)
    z = np.linspace(0.0, 30.0 * scale, 6001)
    return float(np.trapezoid(bubble_extinction(frequency, z, u, sound_speed), z))


def bubble_loss_db(grazing, wind_speed: float, frequency: float, sound_speed: float = 1500.0):
    """Energy lost to near-surface bubbles at one surface bounce, dB (positive), at grazing
    angles in radians. Hall-Novarini population with resonant extinction (module docstring)."""
    a0 = float(resonant_radius(frequency, 0.0))
    if not R_MIN <= a0 <= R_MAX:
        warnings.warn(f"bubble_loss_db: the bubble resonant at {frequency:.0f} Hz ({a0 * 1e6:.0f} um) is outside the "
                      f"Hall-Novarini population ({R_MIN * 1e6:.0f}-{R_MAX * 1e6:.0f} um); the loss is taken as zero", stacklevel=2)
    if float(wind_speed) > MAX_WIND:
        warnings.warn(f"bubble_loss_db: wind speed {wind_speed} m/s is above {MAX_WIND} m/s; the cubic wind law is "
                      "an extrapolation there", stacklevel=2)
    if float(frequency) > TRUSTED_BUBBLE_FREQUENCY and float(wind_speed) > 0.0:
        warnings.warn(f"bubble_loss_db: above {TRUSTED_BUBBLE_FREQUENCY / 1e3:.0f} kHz this law rests on the small-radius end of "
                      "a population fitted for low to moderate frequencies and is probably too high (several times "
                      "the dB of the handbook law that arlpy implements); treat it as an upper estimate", stacklevel=2)
    g = np.maximum(np.asarray(grazing, dtype=float), MIN_GRAZING)
    return (10.0 / math.log(10.0)) * 2.0 * bubble_layer_depth_integral(frequency, wind_speed, sound_speed) / np.sin(g)


BUBBLE_LOSS_MODELS = {"hall_novarini": bubble_loss_db}


def bubble_loss_table(model: str, wind_speed: float, frequency: float, sound_speed: float = 1500.0):
    """(grazing angles in radians, loss in dB) for a named wind-speed law, for interpolation."""
    try:
        fn = BUBBLE_LOSS_MODELS[model]
    except KeyError as e:
        raise ValueError(f"unknown bubble loss model {model!r}; one of {sorted(BUBBLE_LOSS_MODELS)}") from e
    grazing = np.radians(np.concatenate([[0.0], np.arange(2.0, 90.5, 1.0)]))
    return grazing, np.asarray(fn(grazing, wind_speed, frequency, sound_speed), dtype=float)


def kraus_surface_db(grazing, wind_speed: float, frequency: float):
    """Empirical sea-surface backscattering strength, dB, at grazing angles in radians, wind
    speed in m/s and frequency in Hz (module docstring, item 2)."""
    if float(frequency) < KRAUS_MIN_FREQUENCY:
        warnings.warn(f"kraus_surface_db: {frequency:.0f} Hz is below {KRAUS_MIN_FREQUENCY:.0f} Hz; no range of "
                      "validity is stated in the source, and Chapman-Harris is the fitted law there", stacklevel=2)
    t = np.clip(np.asarray(grazing, dtype=float), 1e-3, KRAUS_MAX_GRAZING)
    v = max(float(wind_speed), 0.0) * KNOTS_PER_MS
    f = float(frequency) / 1e3
    b = 4.0 * (v + 2.0) / (v + 1.0) + (2.5 * (f + 0.1) ** (-1.0 / 3.0) - 4.0) * np.cos(t) ** 0.125
    return 10.0 * (-5.05 + 2.0 * np.log10(1.0 + v) + (v / 150.0) * np.log10(f + 0.1) + b * np.log10(np.tan(t)))


def kraus_surface_strength(wind_speed: float, frequency: float):
    """`kraus_surface_db` as the function of (grazing in, grazing out) the reverberation code
    takes. A backscattering fit: for a path pair it is evaluated at the mean angle."""
    return lambda g_in, g_out: 10.0 ** (kraus_surface_db(0.5 * (g_in + g_out), wind_speed, frequency) / 10.0)


def searay_bottom_db(grazing, frequency: float, bottom_type):
    """Empirical bottom backscattering strength, dB, by bottom type: "mud", "sand", "gravel",
    "rock" or the number 1 to 4 (module docstring). Frequency in Hz, grazing in radians."""
    bt = float(BOTTOM_TYPES.get(bottom_type, bottom_type)) if isinstance(bottom_type, str) else float(bottom_type)
    if not 1.0 <= bt <= 4.0:
        raise ValueError("bottom type must be mud, sand, gravel, rock or a number from 1 to 4")
    t = np.clip(np.asarray(grazing, dtype=float), 1e-3, 0.5 * np.pi - 1e-6)
    f = float(frequency) / 1e3
    g = 1.0 + 125.0 * np.exp(-2.64 * (bt - 1.75) ** 2 - (50.0 / bt) / np.tan(t) ** 2)
    b = g * (np.sin(t) + 0.19) ** (bt * np.cos(t) ** 16)
    return 10.0 * np.log10(3.03 * b * f ** (3.2 - 0.8 * bt) * 10.0 ** (2.8 * bt - 12.0) + 10.0**-4.42)


def searay_bottom_strength(frequency: float, bottom_type):
    return lambda g_in, g_out: 10.0 ** (searay_bottom_db(0.5 * (g_in + g_out), frequency, bottom_type) / 10.0)
