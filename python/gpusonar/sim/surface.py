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

   The derivation was re-checked step by step (tests/python/test_bubble_layer.py; PHYSICS_NOTES
   Sec. 20) and is not where the gap comes from. The unit of n is m^-4 (the source's own count
   of 4.7e7 bubbles per m^3 at 13 m/s, its eq. 24 and Fig. 9, needs it); the population's air
   fraction at 2 m in a 10.5 m/s wind, 0.76e-7, agrees with an independent quote of 1e-7; the
   radii in play (33 to 330 micrometres for 100 to 10 kHz) are inside the 10 micrometre to
   1 mm range; and integrating the full damped response of every radius (`bubble_layer_medium`)
   gives 20 to 45 % MORE extinction than the resonant approximation, not less. What a pure
   extinction law leaves out is physical: refraction in the layer, and the sound the bubbles
   scatter back when the layer is opaque. Both are in the two further models below
   (`hall_novarini_layer`, `hall_novarini_screened`); the difference from the handbook law
   that remains (2 to 3 times its dB at 10 to 30 kHz) is a disagreement between this
   population with linear bubble acoustics and a law fitted to measurements at sea, most
   likely the patchiness of real bubble clouds, which no opened source lets us quantify.

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


# --- The bubble layer as an effective medium (models `hall_novarini_layer`, `hall_novarini_screened`).
#
# Opened: Beelen, Nijhof, de Jong, van Wijngaarden and Krug, "Bubble curtains for noise
# mitigation: one vs. two", arXiv:2410.14415v1 (2024), eqs. (6)-(9), who print the model of
# Commander and Prosperetti (JASA 85, 1989; not opened) with these constants:
#
#     c^2 / c_m^2 = 1 + 4 pi c^2 integral a n(a) / (w0^2 - w^2 + 2 i b w) da                  (6)
#     w0^2 = p0 / (rho a^2) (Re Phi - 2 s / (a p0))                                          (7)
#     b    = 2 mu / (rho a^2) + p0 / (2 rho a^2 w) Im Phi + w^2 a / (2 c)                    (8)
#     Phi  = 3 g / (1 - 3 i (g - 1) x (sqrt(i / x) coth sqrt(i / x) - 1))                    (9)
#
# p0 = p_inf + 2 s / a, s = 72.8 mN/m, mu = 1 mPa s, g = 1.4, thermal diffusivity of air
# D = 18.46e-6 m^2/s; the three terms of b are viscous, thermal and radiation damping. The
# source prints "x = D / (w a)"; x must be a pure number (it sits under a square root beside i),
# so x = D / (w a^2) is used: a misprint in the source, and the only reading with the right
# limits (Phi -> 3 g for large bubbles, adiabatic; Phi -> 3 for small ones, isothermal: tested).
WATER_VISCOSITY, SURFACE_TENSION, AIR_DIFFUSIVITY = 1.0e-3, 72.8e-3, 18.46e-6


def bubble_oscillator(radius, frequency: float, depth=0.0, sound_speed: float = 1500.0):
    """(w0^2, b): squared natural angular frequency and damping coefficient (1/s) of an air
    bubble of `radius` (m) at `depth` (m) driven at `frequency` (Hz); eqs. (7)-(9) above."""
    a = np.asarray(radius, dtype=float)
    w = 2.0 * np.pi * float(frequency)
    p0 = ATMOSPHERE + WATER_DENSITY * GRAVITY * np.asarray(depth, dtype=float) + 2.0 * SURFACE_TENSION / a
    x = AIR_DIFFUSIVITY / (w * a * a)
    root = np.sqrt(1j / x)
    phi = 3.0 * AIR_GAMMA / (1.0 - 3.0j * (AIR_GAMMA - 1.0) * x * (root / np.tanh(root) - 1.0))
    w0_sq = p0 / (WATER_DENSITY * a * a) * (phi.real - 2.0 * SURFACE_TENSION / (a * p0))
    b = (2.0 * WATER_VISCOSITY / (WATER_DENSITY * a * a) + p0 / (2.0 * WATER_DENSITY * a * a * w) * phi.imag
         + w * w * a / (2.0 * sound_speed))
    return w0_sq, b


def bubble_layer_medium(frequency: float, depth, wind_speed: float, sound_speed: float = 1500.0, n_radius: int = 1600):
    """The Hall-Novarini layer as an effective medium at `depth` (array, m): a dict with
    `extinction` and `scattering` (1/m of intensity) and `phase_speed` (m/s).

    Every bubble radius of the population is integrated with its full damped response, eq. (6);
    nothing is assumed about resonance. extinction = 2 |Im k_m|; scattering is the integral
    of 4 pi |f|^2 n with f = a w^2 / (w0^2 - w^2 + 2 i b w), the bubble's scattering amplitude
    in eq. (6) (the two are tied by the optical theorem: their ratio at one radius is the
    radiation part of b over b)."""
    z = np.atleast_1d(np.asarray(depth, dtype=float))
    w = 2.0 * np.pi * float(frequency)
    k = w / float(sound_speed)
    a = np.geomspace(R_MIN, R_MAX, int(n_radius))
    w0_sq, b = bubble_oscillator(a[None, :], frequency, z[:, None], sound_speed)
    n = hall_novarini_density(a[None, :], z[:, None], wind_speed)
    f = a[None, :] * w * w / (w0_sq - w * w + 2.0j * b * w)
    km = k * np.sqrt(1.0 + 4.0 * np.pi / (k * k) * np.trapezoid(f * n, a, axis=1))
    return {"extinction": 2.0 * np.abs(km.imag), "scattering": 4.0 * np.pi * np.trapezoid(np.abs(f) ** 2 * n, a, axis=1),
            "phase_speed": w / km.real}


def _layer_profile(frequency: float, wind_speed: float, sound_speed: float, n_depth: int = 240):
    u = max(float(wind_speed), 0.0)
    scale = 0.4 if u <= 7.5 else 0.4 + 0.115 * (u - 7.5)
    z = np.linspace(0.0, 14.0 * scale, n_depth)
    return z, bubble_layer_medium(frequency, z, u, sound_speed)


def bubble_layer_loss_db(grazing, wind_speed: float, frequency: float, sound_speed: float = 1500.0, screened: bool = False):
    """Loss in dB at one surface bounce through the Hall-Novarini layer treated as an
    effective medium (model `hall_novarini_layer`; with `screened`, `hall_novarini_screened`).

    Differences from `bubble_loss_db`, each from first principles:
      - extinction from the full damped response of every radius (`bubble_layer_medium`),
        not the resonant approximation, so the population's edges do not switch the loss off;
      - the ray is refracted in the layer: Snell's law with the layer's phase speed,
        cos t(z) = cos t0 c_m(z) / c, replaces the straight path and the 2 degree floor. Where
        bubbles below resonance dominate the layer is slow and the ray steepens (a shorter
        path); a ray that would turn below the surface (a fast layer, shallow angle) is
        integrated to its turning depth only;
      - `screened`: an optically thick layer is not a hole. Of the energy it removes from the
        path, the bubbles scatter back into the water the single-scattering albedo of a deep
        layer of isotropic scatterers, A = (w/2) (1 - m ln(1 + 1/m)), m = sin t0, w the
        layer's ratio of scattering to extinction (the single-scattering result of radiative
        transfer for a half-space, derived in PHYSICS_NOTES Sec. 20). The energy left is
        exp(-T) + A (1 - exp(-T)). This is an upper bound on what comes back near the
        specular path: A counts all directions.
    """
    z, medium = _layer_profile(frequency, wind_speed, sound_speed)
    g = np.asarray(grazing, dtype=float)
    cos0 = np.cos(g).reshape(-1, 1)
    arg = 1.0 - (cos0 * medium["phase_speed"][None, :] / float(sound_speed)) ** 2            # sin^2 of the local grazing angle
    # From below (the last depth) up to the surface, or to the first depth where the ray turns.
    alive = np.cumprod((arg > 0.0)[:, ::-1], axis=1)[:, ::-1].astype(bool)
    root = np.sqrt(np.where(alive, np.maximum(arg, 0.0), 0.0))
    beta = medium["extinction"]
    # A depth step with sin^2 linear across it: dz * 2 / (s1 + s2), finite at a turning point.
    denom = np.maximum(root[:, 1:] + root[:, :-1], 1e-12)
    steps = 0.5 * (beta[None, 1:] + beta[None, :-1]) * np.diff(z)[None, :] * 2.0 / denom
    tau = 2.0 * np.sum(np.where(alive[:, 1:], steps, 0.0), axis=1)
    kept = np.exp(-tau)
    if screened:
        total = float(np.trapezoid(beta, z))
        albedo = float(np.trapezoid(medium["scattering"], z)) / total if total > 0.0 else 0.0
        m = np.maximum(np.sin(g).ravel(), 1e-9)
        kept = kept + 0.5 * albedo * (1.0 - m * np.log1p(1.0 / m)) * (1.0 - kept)
    out = -10.0 * np.log10(np.maximum(kept, 1e-300))
    return out.reshape(g.shape) if g.ndim else float(out[0])


def bubble_screened_loss_db(grazing, wind_speed: float, frequency: float, sound_speed: float = 1500.0):
    """`bubble_layer_loss_db` with the scattered return of an opaque layer (model `hall_novarini_screened`)."""
    return bubble_layer_loss_db(grazing, wind_speed, frequency, sound_speed, screened=True)


def apl_1994_bubble_loss_db(grazing, wind_speed: float, frequency: float, sound_speed: float = 1500.0):
    """Surface bubble loss per bounce, dB, by the empirical law the owner adopted (model
    `apl_1994`, the recommended one):

        loss = 1.26e-3 / sin(t) * U^1.57 * f^0.85                       U >= 6 m/s
        loss = (the value at 6 m/s) * exp(1.2 (U - 6))                  U <  6 m/s

    t the grazing angle, U the wind speed at 10 m in m/s, f in kHz.

    Source: the formula was read in arlpy 1.9.3 (`arlpy.uwa.bubble_surface_loss`; BSD
    3-clause, Copyright (c) 2016-2020 Acoustic Research Laboratory), whose docstring says
    "computed based on APL model (1994)", and in the same authors' UnderwaterAcoustics.jl
    (`surface_reflection_coef`, MIT licence, commit 7b9c47e), which cites "APL-UW Technical
    Report 9407 II-21". TR 9407 itself was not read. This is our own code for that formula:
    arlpy takes the angle from the vertical and returns an amplitude factor; here the angle is
    grazing and the result is dB of energy. Neither source states a range of frequency, wind
    speed or angle, and neither caps the loss; 1 / sin(t) diverges at grazing incidence, so the
    angle is floored at `MIN_GRAZING` (our guard) and a warning is raised above `MAX_WIND`.
    `sound_speed` is not used.
    """
    u = max(float(wind_speed), 0.0)
    if u > MAX_WIND:
        warnings.warn(f"apl_1994 bubble loss: wind speed {wind_speed} m/s is above {MAX_WIND} m/s; the sources "
                      "state no range and the power law is an extrapolation there", stacklevel=2)
    f = float(frequency) / 1e3
    base = 1.26e-3 * max(u, 6.0) ** 1.57 * f**0.85
    if u < 6.0:
        base = 1.26e-3 * 6.0**1.57 * f**0.85 * math.exp(1.2 * (u - 6.0))
    return base / np.sin(np.maximum(np.asarray(grazing, dtype=float), MIN_GRAZING))


BUBBLE_LOSS_MODELS = {"hall_novarini": bubble_loss_db, "hall_novarini_layer": bubble_layer_loss_db,
                      "hall_novarini_screened": bubble_screened_loss_db, "apl_1994": apl_1994_bubble_loss_db}


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


# --- Surface backscattering from the bubble layer and from wave facets (model `bubble_layer`) -----
#
# Opened: Gallaudet, "Shallow water acoustic backscatter and reverberation measurements using a
# 68-kHz cylindrical array", dissertation (2001), public release, https://hdl.handle.net/10945/11015,
# Ch. IV Sec. II.C, pp. 124-125, eqs. (7)-(12), which summarises the APL-UW surface model (the
# handbook, his ref. 59, was not read) and attributes the bubble term to Crowther (1980) and
# McDaniel and Gorman (1982) as modified by McDaniel (1993). Backscattering strength is
# 10 log10(sigma_r + sigma_b), roughness plus bubbles:
#
#   sigma_b = delta_r sin(t) / (8 pi delta) (1 + 8 b exp(-2 b) - exp(-4 b)),   b = beta_I / sin(t)   (10)
#       delta = 2.55e-2 f^(1/3) the total damping constant, delta_r = 0.0136 the re-radiation
#       damping constant at resonance, t the grazing angle, beta_I "the depth-integrated
#       extinction cross section per unit volume due to scattering and absorption";
#       saturation, b >> 1:  sigma_b = delta_r sin(t) / (8 pi delta)                              (11)
#       sparse layer, b << 1: sigma_b = 1.5 delta_r beta_I / (pi delta)                           (12)
#   sigma_f = sec^4(ti) / (4 pi s^2) exp(-tan^2(ti) / s^2),  ti = 90 deg - t                       (8)
#       near-specular scattering from wave facets (the high-frequency limit of the Kirchhoff
#       approximation for Gaussian, isotropic slopes of mean square s^2), for t above about 70 deg.
#
# The source says the model "agrees well with data collected during winds of 3-15 m/s and for
# frequencies of 12-70 kHz, with an estimated uncertainty of +-5 dB for wind speeds < 8 m/s".
#
# What the source does not print, and what is used here instead (ours, stated per item):
#   - the unit of f in delta. kHz is used: it gives delta = 0.08 at 30 kHz and 0.10 at 68 kHz,
#     the size of the damping of a resonant bubble (the damping of eqs. (7)-(9) above gives
#     0.07 to 0.12 for 10 to 100 kHz: tested), where Hz would give more than 0.5.
#   - the empirical law for beta_I against wind speed and frequency. beta_I is the optical depth
#     of one vertical crossing of the layer, so a surface bounce loses 2 beta_I / sin(t) nepers
#     of intensity: exactly what the bubble loss laws of this module give. It is taken from one
#     of them (`extinction`): "apl_1994" (default; beta_I = loss in dB x sin(t) ln(10) / 20, which
#     for that law does not depend on t) or "hall_novarini" (`bubble_layer_depth_integral`).
#   - the small-scale roughness term sigma_sc = G(U10) tan^4(t) (their eq. (9); G is "an empirical
#     function" not printed) and the bubble extinction applied to the roughness terms. They are
#     left out: between about 30 and 70 degrees, where the source says sigma_sc is fitted, this
#     model is therefore a lower bound; below about 30 degrees bubbles dominate at these
#     frequencies and winds, and above about 70 degrees the facets do.
#   - delta_r = 0.0136 is k a at resonance for a bubble near the surface (2 pi f a_R / c with
#     Minnaert's a_R f = 3.26 m Hz gives 0.01366: tested), so it is not a free constant.
BUBBLE_RERADIATION_DAMPING = 0.0136


def bubble_damping_constant(frequency: float) -> float:
    """Total damping constant of a resonant bubble, 2.55e-2 f^(1/3) with f in kHz (comment above)."""
    return 2.55e-2 * (float(frequency) / 1e3) ** (1.0 / 3.0)


def bubble_layer_optical_depth(wind_speed: float, frequency: float, sound_speed: float = 1500.0,
                               extinction: str = "apl_1994") -> float:
    """beta_I: nepers of intensity lost in one vertical crossing of the bubble layer, from a
    bubble loss law of this module: "apl_1994" or "hall_novarini"."""
    if extinction == "apl_1994":
        t = np.radians(30.0)                              # the law is exactly 1 / sin(t): any angle gives the same
        return float(apl_1994_bubble_loss_db(t, wind_speed, frequency)) * math.sin(t) * math.log(10.0) / 20.0
    if extinction == "hall_novarini":
        return bubble_layer_depth_integral(frequency, wind_speed, sound_speed)
    raise ValueError(f"unknown extinction {extinction!r}; 'apl_1994' or 'hall_novarini'")


def bubble_backscatter(grazing, wind_speed: float, frequency: float, sound_speed: float = 1500.0,
                       extinction: str = "apl_1994"):
    """Backscattering cross section per unit area and solid angle of the near-surface bubble
    layer, eq. (10) of the comment above, at grazing angles in radians."""
    t = np.maximum(np.asarray(grazing, dtype=float), MIN_GRAZING)
    b = bubble_layer_optical_depth(wind_speed, frequency, sound_speed, extinction) / np.sin(t)
    level = BUBBLE_RERADIATION_DAMPING * np.sin(t) / (8.0 * np.pi * bubble_damping_constant(frequency))
    # 1 + 8 b exp(-2 b) - exp(-4 b), written so that small b keeps its leading term 12 b.
    return level * (8.0 * b * np.exp(-2.0 * b) - np.expm1(-4.0 * b))


def facet_backscatter(grazing, wind_speed: float):
    """Near-specular backscattering cross section of the wave facets, eq. (8) of the comment
    above, with the Cox-Munk mean square slope of `acoustics.mean_square_slope`."""
    from ..acoustics import mean_square_slope
    s2 = mean_square_slope(wind_speed)
    t = np.clip(np.asarray(grazing, dtype=float), 1e-6, 0.5 * np.pi)
    tan_i = 1.0 / np.tan(t)                               # tangent of the angle of incidence
    with np.errstate(over="ignore", under="ignore"):
        return (1.0 + tan_i**2) ** 2 / (4.0 * np.pi * s2) * np.exp(-(tan_i**2) / s2)


def bubble_layer_surface_db(grazing, wind_speed: float, frequency: float, sound_speed: float = 1500.0,
                            extinction: str = "apl_1994"):
    """Sea-surface backscattering strength, dB, from the bubble layer plus the wave facets
    (model `bubble_layer`; comment above): the two terms of the published high-frequency model
    whose formulas could be read. Stated range of the published model: 12 to 70 kHz, 3 to
    15 m/s; a warning is raised outside it."""
    if not 12e3 <= float(frequency) <= 70e3 or not 3.0 <= float(wind_speed) <= 15.0:
        warnings.warn(f"bubble_layer_surface_db: {frequency / 1e3:g} kHz, {wind_speed:g} m/s is outside the range the "
                      "published model was compared with data in (12-70 kHz, 3-15 m/s)", stacklevel=2)
    total = bubble_backscatter(grazing, wind_speed, frequency, sound_speed, extinction) + facet_backscatter(grazing, wind_speed)
    return 10.0 * np.log10(np.maximum(total, 1e-300))


def bubble_layer_surface_strength(wind_speed: float, frequency: float, sound_speed: float = 1500.0,
                                  extinction: str = "apl_1994"):
    """`bubble_layer_surface_db` as the function of (grazing in, grazing out) the reverberation
    code takes. A backscattering model: for a path pair it is evaluated at the mean angle."""
    return lambda g_in, g_out: 10.0 ** (bubble_layer_surface_db(0.5 * (g_in + g_out), wind_speed, frequency,
                                                               sound_speed, extinction) / 10.0)
