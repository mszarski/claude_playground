"""Scattering strength of a rough fluid sediment: interface roughness plus sediment volume.

This is the kind of bottom model SST points to (`JacksonBottom`, `APLBottom`: a fluid sediment
with a power-law roughness spectrum and volume inhomogeneities, six parameters). The APL-UW
handbook that defines those models (TR 9407) was not available, so nothing here is copied from
it: both terms are derived below from the wave equation, and tests check them against limits
with known answers. The default roughness term is the lowest-order small-slope approximation
(`roughness_model="small_slope"`, D-86): first-order perturbation theory wherever that is valid
(below about 60 degrees of grazing and for k x rms relief, at scales near the acoustic
wavelength, below about one) and the Kirchhoff result in the specular direction, where
perturbation theory with a power-law spectrum diverges. The alternatives (`roughness_model=`,
see "Other regimes" below) are first-order perturbation theory alone (`"perturbation"`, the
default before D-86), the Kirchhoff approximation near the specular direction, and the
composite-roughness (large-scale slope) average at small grazing angles. Named sediments are in `configs/sediments.yaml` (`sediment()` reads it):
the handbook's sediment classes from sandy gravel to clay, every parameter read from an open
document that attributes it to the handbook (the file's header and `provenance:` entries say
which), and the placeholder. The scattering parameters follow the mean grain size
(`grain_size_scattering`); rock and pebbles are not fluids and carry an empirical Lambert law
instead (`empirical_backscatter`).

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

Other regimes (`roughness_model="small_slope"` is the default, `DEFAULT_ROUGHNESS_MODEL`;
`"perturbation"`, the formulas above alone, was the default before D-86 and is unchanged).

Notation: q = ks - ki is the change of the (three-dimensional) wave vector, Q its horizontal
part, qz its vertical part: Q^2 = k^2 (cos^2 ti + cos^2 ts - 2 cos ti cos ts cos phi),
qz = k (sin ti + sin ts), q^2 = Q^2 + qz^2. For a pure power-law spectrum the relief has no
finite variance, but its structure function is finite for 2 < gamma2 < 4:

    D(R) = <(zeta(R0 + R) - zeta(R0))^2> = 2 integral W(K) (1 - cos K.R) d2K = Ch2 R^(2 a),
    a = gamma2 / 2 - 1,   Ch2 = 2 pi w2 Gamma(1 - a) / (4^a a Gamma(1 + a)).

"kirchhoff": tangent-plane approximation, the local reflection coefficient V taken at the
angle for which the facet reflects ki into ks (sin t_is = q / 2k). For Gaussian relief

    sigma_k = |V(t_is)|^2 / (8 pi) (q^2 / qz)^2 I,
    I = integral_0^inf J0(Q R) [exp(-qz^2 D(R) / 2)] R dR        (the "Kirchhoff integral").

This is the standard physical-optics result (our derivation; the Kirchhoff integral in this
form, with the argument qz^2, is eq. (24) of Olson, arXiv:2106.08287, taken there from
Jackson and Olson 2020). In the specular direction and for small relief it equals the
perturbation result exactly (both give |V|^2 q^4 W(Q) / 4), which a test checks. It is
accurate near the specular direction and wrong at low grazing angles away from it.

"small_slope": the lowest-order small-slope approximation (Voronovich). It keeps the
perturbation theory's angular factor and replaces the spectrum by the Kirchhoff integral:

    sigma_ss = [sigma_pert / W(Q)] I / (2 pi qz^2).

It tends to perturbation theory when qz^2 D is small at the scales that matter and to the
Kirchhoff result in the specular direction, so it is the documented interpolation between the
two regimes. (Written here from those two requirements; the source above confirms that the
small-slope and Kirchhoff approximations share the integral I with the argument qz^2.)

"composite": the perturbation result averaged over the slopes of the large-scale relief
(McDaniel and Gorman 1983 and Jackson, Winebrenner and Ishimaru 1986 introduced this for the
bottom; neither could be opened, so the recipe below is ours). The relief is split at a
wavenumber Kc chosen so the small-scale part (K > Kc) has k x rms height = `small_scale_kh`:
h_s^2 = 2 pi w2 Kc^(2 - gamma2) / (gamma2 - 2). The large-scale part tilts the surface with a
Gaussian slope of variance s^2 = pi w2 Kc^(4 - gamma2) / (4 - gamma2) per axis. With slope
components (u, v) along and across the incident horizontal direction, the local grazing
angles are ti + atan(u) and ts - atan(u cos phi + v sin phi); the cross section is the
average over (u, v) of the perturbation one with the small-scale spectrum (zero below Kc),
and zero where a local angle is negative (shadowed facet). `small_scale_kh` = 0.5 was our
choice when this was written; a published criterion has since been read and is the same
number: Fonseca, Lurton, Fezzani and Roche, "A simplified semi-empirical model for
multifrequency seafloor backscattering angular response (ESAB)", Front. Remote Sens. (2025),
doi:10.3389/frsen.2025.1619218, eqs. (6)-(8), "following Novarini and Caruthers (1994), the
cut-off spatial frequency corresponds to the condition that the Rayleigh parameter ... for
microroughness equals unity", g^2 = 4 k^2 zeta_mu^2 = 1, i.e. k x rms height = 0.5, with
zeta_mu^2 = 2 pi integral_Kc^inf W K dK and the facet slope variance 2 pi integral_0^Kc W K^3 dK
(both axes together: twice the per-axis s^2 above). Olson and Lyons (arXiv:2106.05909,
Sec. VIII, eq. (44), after McDaniel and Gorman 1983) write the composite model as this same
average of the tilted small-scale perturbation result over the large-scale slope density and
give its conditions: k x small-scale rms height below one, grazing angles above about twice
the large-scale rms slope angle, shadowing below it. The handbook's own split and its
interpolation between the Kirchhoff and composite results were not found in an open
document. The composite result describes Bragg scattering only: towards the specular
direction, where Q falls below Kc, it goes to zero and the Kirchhoff term is the right one.

Validity of the lowest-order small-slope result (`small_slope`), from a comparison with
boundary-element solutions for power-law roughness (Olson, "Validity criteria for the small
slope approximation for backscattering of sound from rough surfaces with a power-law
roughness spectrum", J. Acoust. Soc. Am. 160, 1918-1935, 2026, CC BY,
https://hdl.handle.net/10945/76181; one-dimensional surfaces, backscatter, 1 dB criterion):
for a pressure-release surface the largest valid k x rms height is of order one (0.1 to 2)
whatever the spectral exponent; for a sand-like fluid it is about 1.9 times that (0.6 to 3),
for a mud-like fluid larger still (up to about 11); the largest valid rms slope lies between
about 10 and 60 degrees depending on the exponent. At small grazing angles "PT is accurate
whenever SSA(2) is accurate", and the paper notes that the composite model "has largely
been superseded by the SSA" for the mean level. The next term of the small-slope series
helps for a pressure-release surface and not for fluid sediments.

"combined": the small-slope cross section (full spectrum) averaged over the same large-scale
slopes. This is Kirchhoff in the specular direction, perturbation theory at intermediate
angles and the composite result at small grazing angles, in one expression with no switch.
The large-scale slopes are counted twice near the specular direction (once in the Kirchhoff
integral, once in the average), which widens the specular peak by about the rms slope, a few
degrees at most. (A smooth minimum of the Kirchhoff and composite results was tried first and
rejected: for a sediment the Kirchhoff result is BELOW the perturbation one at 20-30 degrees,
because it carries the normal-incidence reflection coefficient, so a minimum pulls the
valid perturbation result down by up to 6 dB.)
"""
from __future__ import annotations

import functools
import math

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.special import hankel1

PLACEHOLDER_SAND = {"rho": 1.9, "nu": 1.1, "delta": 0.0147, "w2": 2.0e-5, "gamma2": 3.25, "sigma2": 0.002}
"""Illustrative values only, NOT a measured or published sediment: the density, speed and loss
are those the scenarios already use for their sand bottom (1.9, 1650 m/s, 0.8 dB per wavelength);
the roughness and volume numbers were chosen to give a backscattering strength near -30 dB at
20 degrees and 100 kHz. Replace with values for the site."""


SEDIMENT_KEYS = ("rho", "nu", "delta", "w2", "gamma2", "sigma2")
_DB_PER_NEPER = 20.0 / math.log(10.0)


def w2_from_cm4(strength_cm4: float, gamma2: float) -> float:
    """Spectral strength in m^(4 - gamma2) (wavenumbers in rad/m) from the form with a 1 cm
    reference length, W = beta (h K)^(-gamma2), h = 1 cm, beta in cm^4 (Bentrem et al.,
    arXiv:physics/0607146, eq. (7) with K = 2 pi f / c, who take it from Mourad and Jackson):
    W [m^4] = 1e-8 beta (0.01 K [1/m])^(-gamma2), so w2 = beta 10^(2 gamma2 - 8)."""
    return float(strength_cm4) * 10.0 ** (2.0 * float(gamma2) - 8.0)


def parameters_from_physical(*, density: float, sound_speed: float, attenuation_db_per_m: float,
                             attenuation_frequency: float, w2: float, gamma2: float, volume_strength_db: float,
                             water_density: float, water_sound_speed: float) -> dict:
    """The six model parameters from a sediment given in physical units:
    rho = density / water density;  nu = sound speed / water sound speed;
    delta from the attenuation a (dB/m) at frequency f: the amplitude decays as
    exp(-Im(k2) x) with Im(k2) = 2 pi f delta / c_sediment, so a = (20 / ln 10) 2 pi f delta / c_sediment;
    sigma2 = volume scattering cross section per unit volume (1/m, 10^(dB / 10)) / a."""
    delta = attenuation_db_per_m * sound_speed / (_DB_PER_NEPER * 2.0 * math.pi * attenuation_frequency)
    return {"rho": density / water_density, "nu": sound_speed / water_sound_speed, "delta": delta, "w2": float(w2),
            "gamma2": float(gamma2), "sigma2": 10.0 ** (volume_strength_db / 10.0) / attenuation_db_per_m}


def sediment_table(path=None) -> dict:
    """The entries of `configs/sediments.yaml` (or `path`) as written."""
    from pathlib import Path

    import yaml

    from .. import config
    file = Path(path) if path is not None else config.config_dir() / "sediments.yaml"
    with open(file) as f:
        return yaml.safe_load(f) or {}


REFLECTION_KEYS = ("rho", "nu", "delta")
DB_PER_WAVELENGTH = 40.0 * math.pi / math.log(10.0)
"""dB per wavelength for a loss parameter delta of one: a = delta 40 pi / ln 10 (the same
conversion as in `propagation.rayleigh_reflection`; UnderwaterAcoustics.jl (MIT), src/basic.jl:171-179,
prints it and attributes it to APL-UW TR 9407, IV-9 equation (4))."""


def _entry(name: str, path=None) -> dict:
    """A table entry with its ratios worked out. `physical:` gives all six parameters in
    physical units; `reflection:` gives density (kg/m3), sound_speed (m/s) and the loss
    parameter delta with the water density and sound speed the source's ratios refer to."""
    table = sediment_table(path)
    if name not in table:
        raise ValueError(f"unknown sediment {name!r}; the table has {sorted(table)}")
    entry = table[name]
    if "physical" in entry:
        entry = dict(parameters_from_physical(**{k: float(v) for k, v in entry["physical"].items()}), **entry)
    if "reflection" in entry:
        r = {k: float(v) for k, v in entry["reflection"].items()}
        entry = dict({"rho": r["density"] / r["water_density"], "nu": r["sound_speed"] / r["water_sound_speed"],
                      "delta": r["delta"]}, **entry)
    if "scattering" in entry:
        # The three scattering parameters in the units of their source: spectral strength in
        # cm^4 with a 1 cm reference length (`w2_cm4`), or already in m^(4 - gamma2) (`w2`).
        sc = {k: float(v) for k, v in entry["scattering"].items()}
        w2 = w2_from_cm4(sc["w2_cm4"], sc["gamma2"]) if "w2_cm4" in sc else sc["w2"]
        entry = dict({"w2": w2, "gamma2": sc["gamma2"], "sigma2": sc["sigma2"]}, **entry)
    return entry


def _warn_status(name: str, entry: dict, keys) -> None:
    import warnings
    # "confirmed": read from a cited source. "fitted (...)": fitted to measurements by
    # tools/fit_sediment.py; what the data left open is in `unconfirmed_fields`.
    if entry.get("status") != "confirmed" and not str(entry.get("status", "")).startswith("fitted"):
        warnings.warn(f"sediment {name!r} is {entry.get('status', 'unconfirmed')}: its parameters are not from a "
                      "measured or published source", stacklevel=3)
    doubtful = [k for k in entry.get("unconfirmed_fields", []) if k in keys]
    if doubtful:
        warnings.warn(f"sediment {name!r}: {doubtful} could not be confirmed (see the entry's note in the sediment "
                      "table)", stacklevel=3)


def is_reflection_only(name: str, path=None) -> bool:
    """True for an entry that carries density, speed and loss but no roughness or volume parameters."""
    entry = _entry(name, path)
    return any(k not in entry for k in SEDIMENT_KEYS) and all(k in entry for k in REFLECTION_KEYS)


def sediment(name: str, path=None, supplied=None) -> dict:
    """The six parameters of a named sediment from `configs/sediments.yaml` (or `path`). An
    entry gives them directly, or gives `physical:` values that `parameters_from_physical`
    converts. An entry whose `status` is not "confirmed" raises a warning: its numbers are not
    from a source.

    Entries marked `reflection_only` carry the density ratio, speed ratio and loss parameter
    only. They are enough for the reflection coefficient (`sediment_reflection`) and not for
    the scattering strength: asking for one here is an error naming the missing parameters,
    unless `supplied` (the scenario item) gives them, in which case what the table has is returned."""
    entry = _entry(name, path)
    missing = [k for k in SEDIMENT_KEYS if k not in entry and k not in (supplied or {})]
    if missing:
        what = ("is reflection-only (density, sound speed and loss; usable for bottom reflection): for a "
                "scattering strength give" if entry.get("reflection_only") else "lacks")
        hint = (" (it is not a fluid sediment: the table gives an empirical Lambert law for it, "
                "bottom.empirical_backscatter)" if "empirical_backscatter" in entry else "")
        raise ValueError(f"sediment {name!r} {what} {missing}{hint}")
    _warn_status(name, entry, SEDIMENT_KEYS)
    return {k: float(entry[k]) for k in SEDIMENT_KEYS if k in entry}


def sediment_reflection(name: str, path=None) -> dict:
    """{rho, nu, delta} of a named sediment: all that the flat-interface reflection coefficient
    needs. Works for every entry, the reflection-only ones included."""
    entry = _entry(name, path)
    missing = [k for k in REFLECTION_KEYS if k not in entry]
    if missing:
        raise ValueError(f"sediment {name!r} lacks {missing}")
    _warn_status(name, entry, REFLECTION_KEYS)
    return {k: float(entry[k]) for k in REFLECTION_KEYS}


def fluid_halfspace(name: str, water_sound_speed: float, path=None) -> dict:
    """A named sediment as the fluid half-space the eigenray providers take:
    {speed (m/s), density (relative to the water), attenuation (dB per wavelength)}, with
    speed = nu x the water sound speed given here (the table holds ratios)."""
    par = sediment_reflection(name, path)
    return {"speed": par["nu"] * float(water_sound_speed), "density": par["rho"],
            "attenuation": par["delta"] * DB_PER_WAVELENGTH}


def _vertical(k2, horizontal):
    b = np.sqrt(k2 - horizontal**2 + 0j)
    return np.where(b.imag < 0.0, -b, b)          # decays away from the interface


def reflection(grazing, rho: float, nu: float, delta: float):
    """Flat-interface pressure reflection coefficient V of the fluid sediment at a grazing angle."""
    g = np.asarray(grazing, dtype=float)
    kappa = (1.0 + 1j * delta) / nu
    p = _vertical(kappa**2, np.cos(g))
    return (rho * np.sin(g) - p) / (rho * np.sin(g) + p)


def _perturbation_factor(ti, ts, phi, k, rho, nu, delta):
    """sigma_pert / W(Q), and Q, for broadcast angle arrays."""
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
    return b1s**2 * np.abs(a) ** 2, q


def _angles(grazing_in, grazing_out, azimuth):
    return np.broadcast_arrays(np.asarray(grazing_in, dtype=float), np.asarray(grazing_out, dtype=float),
                               np.asarray(azimuth, dtype=float))


def _power_law(q, w2, gamma2):
    with np.errstate(divide="ignore"):
        return np.where(q > 0.0, w2 / np.maximum(q, 1e-300) ** gamma2, 0.0)


def perturbation_strength(grazing_in, grazing_out, frequency: float, sound_speed: float, *, rho: float, nu: float,
                          delta: float, w2: float, gamma2: float, azimuth=np.pi, **_ignored):
    """Roughness cross section by first-order perturbation theory (the module's derivation)."""
    ti, ts, phi = _angles(grazing_in, grazing_out, azimuth)
    k = 2.0 * np.pi * float(frequency) / float(sound_speed)
    factor, q = _perturbation_factor(ti, ts, phi, k, rho, nu, delta)
    return factor * _power_law(q, w2, gamma2)


def structure_constant(w2: float, gamma2: float) -> tuple[float, float]:
    """(Ch2, a) of the structure function D(R) = Ch2 R^(2 a) of a power-law relief spectrum."""
    if not 2.0 < gamma2 < 4.0:
        raise ValueError("the structure function of a power-law spectrum needs 2 < gamma2 < 4")
    a = 0.5 * gamma2 - 1.0
    return 2.0 * np.pi * w2 * math.gamma(1.0 - a) / (4.0**a * a * math.gamma(1.0 + a)), a


def _large_b(a: float) -> float:
    return 1.0e4**a


@functools.lru_cache(maxsize=16)
def _kirchhoff_table(a: float):
    """Spline of log F against log b for F(b) = integral_0^inf J0(x) x exp(-b x^(2a)) dx.

    J0 = Re H0(1), and the integrand with H0(1) is analytic in the upper right quadrant, so the
    path is turned to x = t exp(i psi): there H0(1) decays like exp(-t sin psi) and the
    exponential still decays while cos(2 a psi) > 0. No oscillating tail is left to integrate."""
    psi = min(0.5 * np.pi, 0.8 * np.pi / (4.0 * a))
    edges = np.concatenate([[0.0], np.geomspace(1e-10, 0.5, 220), np.arange(1.0, 40.0 / np.sin(psi) + 0.5, 0.5)])
    x, w = np.polynomial.legendre.leggauss(8)
    t = (0.5 * (edges[1:] + edges[:-1])[:, None] + 0.5 * np.diff(edges)[:, None] * x).ravel()
    wt = (0.5 * np.diff(edges)[:, None] * w).ravel()
    rot = np.exp(1j * psi)
    kernel = wt * hankel1(0, t * rot) * t * rot**2
    log_b = np.linspace(np.log(_SMALL_B), np.log(_large_b(a)), 500)
    f = np.array([(kernel * np.exp(-b * (t * rot) ** (2.0 * a))).sum().real for b in np.exp(log_b)])
    return CubicSpline(log_b, np.log(f))


_SMALL_B = 1e-7


def kirchhoff_integral(b, a: float):
    """F(b) = integral_0^inf J0(x) x exp(-b x^(2a)) dx for 0 < a < 1, b > 0. With
    b = qz^2 Ch2 / (2 Q^(2a)) the Kirchhoff integral is I = F / Q^2."""
    b = np.asarray(b, dtype=float)
    spline = _kirchhoff_table(float(a))
    lo, hi = _SMALL_B, _large_b(a)
    inside = np.exp(spline(np.log(np.clip(b, lo, hi))))
    # Small b: first term of the expansion in powers of b (the perturbation limit).
    small = b * 2.0 ** (2.0 * a + 1.0) * a * math.gamma(1.0 + a) / math.gamma(1.0 - a)
    # Large b: expand J0 in powers of x; three terms.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        u = np.maximum(b, lo) ** (-1.0 / a)
        large = (math.gamma(1.0 / a) * u - 0.25 * math.gamma(2.0 / a) * u**2 + math.gamma(3.0 / a) * u**3 / 64.0) / (2.0 * a)
    return np.where(b < lo, small, np.where(b > hi, large, inside))


def _kirchhoff_geometry(ti, ts, phi, k):
    q_h = k * np.sqrt(np.maximum(np.cos(ti) ** 2 + np.cos(ts) ** 2 - 2.0 * np.cos(ti) * np.cos(ts) * np.cos(phi), 0.0))
    q_z = k * (np.sin(ti) + np.sin(ts))
    return q_h, q_z


def _integral_over_q2(q_h, q_z, w2, gamma2):
    """The Kirchhoff integral I = integral J0(Q R) exp(-qz^2 D(R) / 2) R dR, finite at Q = 0."""
    ch2, a = structure_constant(w2, gamma2)
    half = 0.5 * q_z**2 * ch2
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        b = half / np.maximum(q_h, 1e-300) ** (2.0 * a)
        general = kirchhoff_integral(b, a) / np.maximum(q_h, 1e-300) ** 2
        specular = math.gamma(1.0 / a) / (2.0 * a) * np.maximum(half, 1e-300) ** (-1.0 / a)
    # Towards Q = 0 the general form is 0/0 numerically: use its limit once b is huge.
    return np.where(b > 1e12, specular, general)


def kirchhoff_strength(grazing_in, grazing_out, frequency: float, sound_speed: float, *, rho: float, nu: float,
                       delta: float, w2: float, gamma2: float, azimuth=np.pi, **_ignored):
    """Roughness cross section in the Kirchhoff (tangent-plane) approximation: near specular."""
    ti, ts, phi = _angles(grazing_in, grazing_out, azimuth)
    k = 2.0 * np.pi * float(frequency) / float(sound_speed)
    q_h, q_z = _kirchhoff_geometry(ti, ts, phi, k)
    q2 = q_h**2 + q_z**2
    v = reflection(np.arcsin(np.clip(np.sqrt(q2) / (2.0 * k), 0.0, 1.0)), rho, nu, delta)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.abs(v) ** 2 / (8.0 * np.pi) * (q2 / q_z) ** 2 * _integral_over_q2(q_h, q_z, w2, gamma2)
    return np.where(q_z > 0.0, out, 0.0)


def small_slope_strength(grazing_in, grazing_out, frequency: float, sound_speed: float, *, rho: float, nu: float,
                         delta: float, w2: float, gamma2: float, azimuth=np.pi, **_ignored):
    """Roughness cross section in the lowest-order small-slope approximation: perturbation
    theory away from the specular direction, Kirchhoff in it."""
    ti, ts, phi = _angles(grazing_in, grazing_out, azimuth)
    k = 2.0 * np.pi * float(frequency) / float(sound_speed)
    factor, _ = _perturbation_factor(ti, ts, phi, k, rho, nu, delta)
    q_h, q_z = _kirchhoff_geometry(ti, ts, phi, k)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = factor * _integral_over_q2(q_h, q_z, w2, gamma2) / (2.0 * np.pi * q_z**2)
    return np.where(q_z > 0.0, out, 0.0)


def large_scale_slope(frequency: float, sound_speed: float, *, w2: float, gamma2: float, small_scale_kh: float = 0.5,
                      **_ignored) -> tuple[float, float]:
    """(rms slope per axis of the large-scale relief, cutoff wavenumber Kc in rad/m) for the
    composite-roughness split: relief at K > Kc has k x rms height = `small_scale_kh`."""
    if not 2.0 < gamma2 < 4.0:
        raise ValueError("composite roughness needs 2 < gamma2 < 4")
    k = 2.0 * np.pi * float(frequency) / float(sound_speed)
    kc = (2.0 * np.pi * w2 * k**2 / ((gamma2 - 2.0) * small_scale_kh**2)) ** (1.0 / (gamma2 - 2.0))
    return float(np.sqrt(np.pi * w2 * kc ** (4.0 - gamma2) / (4.0 - gamma2))), float(kc)


def _slope_average(fn, ti, ts, phi, slope: float, nodes: int):
    """Average fn(local ti, local ts, phi) over a Gaussian large-scale slope (rms `slope` per
    axis); facets the incident or the scattered wave cannot see contribute nothing."""
    x, w = np.polynomial.hermite_e.hermegauss(int(nodes))             # weight exp(-x^2 / 2)
    w = w / w.sum()
    total = np.zeros(ti.shape)
    for xu, wu in zip(x, w):
        for xv, wv in zip(x, w):
            li = ti + np.arctan(slope * xu)
            ls = ts - np.arctan(slope * (xu * np.cos(phi) + xv * np.sin(phi)))
            lit = (li > 0.0) & (ls > 0.0)
            li, ls = np.clip(li, 1e-6, 0.5 * np.pi), np.clip(ls, 1e-6, 0.5 * np.pi)
            total += wu * wv * np.where(lit, fn(li, ls, phi), 0.0)
    return total


def composite_strength(grazing_in, grazing_out, frequency: float, sound_speed: float, *, rho: float, nu: float,
                       delta: float, w2: float, gamma2: float, azimuth=np.pi, small_scale_kh: float = 0.5,
                       slope_nodes: int = 12, **_ignored):
    """Perturbation cross section of the small-scale relief (K > Kc) averaged over the slopes
    of the large-scale relief (composite roughness). For low and moderate grazing angles."""
    ti, ts, phi = _angles(grazing_in, grazing_out, azimuth)
    k = 2.0 * np.pi * float(frequency) / float(sound_speed)
    slope, kc = large_scale_slope(frequency, sound_speed, w2=w2, gamma2=gamma2, small_scale_kh=small_scale_kh)

    def small_scale(li, ls, ph):
        factor, q = _perturbation_factor(li, ls, ph, k, rho, nu, delta)
        return factor * np.where(q >= kc, _power_law(q, w2, gamma2), 0.0)
    return _slope_average(small_scale, ti, ts, phi, slope, slope_nodes)


def combined_strength(grazing_in, grazing_out, frequency: float, sound_speed: float, *, rho: float, nu: float,
                      delta: float, w2: float, gamma2: float, azimuth=np.pi, small_scale_kh: float = 0.5,
                      slope_nodes: int = 12, **_ignored):
    """The small-slope cross section averaged over the large-scale slopes: Kirchhoff near the
    specular direction, perturbation theory at intermediate angles, composite roughness at
    small grazing angles."""
    ti, ts, phi = _angles(grazing_in, grazing_out, azimuth)
    slope, _ = large_scale_slope(frequency, sound_speed, w2=w2, gamma2=gamma2, small_scale_kh=small_scale_kh)
    par = dict(rho=rho, nu=nu, delta=delta, w2=w2, gamma2=gamma2)
    return _slope_average(lambda li, ls, ph: small_slope_strength(li, ls, frequency, sound_speed, azimuth=ph, **par),
                          ti, ts, phi, slope, slope_nodes)


ROUGHNESS_MODELS = ("perturbation", "kirchhoff", "small_slope", "composite", "combined")

DEFAULT_ROUGHNESS_MODEL = "small_slope"
"""The roughness model used when none is named (D-86; "perturbation" before). Reasons: the
lowest-order small-slope result is what current work uses and it is valid wherever first-order
perturbation theory is, and beyond (Olson, J. Acoust. Soc. Am. 160, 1918-1935, 2026,
https://hdl.handle.net/10945/76181: "PT is accurate whenever SSA(2) is accurate"); it equals
the perturbation result for small relief and the Kirchhoff result in the specular direction
(tests in `test_bottom_models.py`), where perturbation theory with a power-law spectrum
diverges. Evidence is the literature (PHYSICS_NOTES Sec. 23.2), not a measurement of ours."""

_warned_no_small_slope = False


def roughness_strength(grazing_in, grazing_out, frequency: float, sound_speed: float, *, rho: float, nu: float,
                       delta: float, w2: float, gamma2: float, azimuth=np.pi, roughness_model: str | None = None,
                       small_scale_kh: float = 0.5, **_ignored):
    """Cross section per unit area and solid angle from interface roughness. `azimuth` is the
    angle between the horizontal directions of the incident and scattered waves: pi for
    backscatter, 0 for forward scatter. `roughness_model`: "small_slope" (the default,
    `DEFAULT_ROUGHNESS_MODEL`), "perturbation" (first order; the default before D-86),
    "kirchhoff", "composite" or "combined" (module docstring). The small-slope result needs a
    structure function, i.e. 2 < gamma2 < 4: outside that range, when no model is named, the
    perturbation result is used and a warning says so once (naming "small_slope" raises)."""
    global _warned_no_small_slope
    if roughness_model is None:
        roughness_model = DEFAULT_ROUGHNESS_MODEL
        if not 2.0 < gamma2 < 4.0:
            if not _warned_no_small_slope:
                import warnings
                warnings.warn(f"bottom roughness: gamma2 = {gamma2:g} is outside (2, 4), where the default small-slope model "
                              "has no structure function; first-order perturbation theory is used instead "
                              "(name roughness_model to choose)", stacklevel=2)
                _warned_no_small_slope = True
            roughness_model = "perturbation"
    args = (grazing_in, grazing_out, frequency, sound_speed)
    par = dict(rho=rho, nu=nu, delta=delta, w2=w2, gamma2=gamma2, azimuth=azimuth)
    if roughness_model == "perturbation":
        return perturbation_strength(*args, **par)
    if roughness_model == "kirchhoff":
        return kirchhoff_strength(*args, **par)
    if roughness_model == "small_slope":
        return small_slope_strength(*args, **par)
    if roughness_model == "composite":
        return composite_strength(*args, small_scale_kh=small_scale_kh, **par)
    if roughness_model == "combined":
        return combined_strength(*args, small_scale_kh=small_scale_kh, **par)
    raise ValueError(f"unknown roughness_model {roughness_model!r}; one of {ROUGHNESS_MODELS}")


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


# --- Scattering parameters from the mean grain size -------------------------------------------
#
# The handbook's default roughness and volume parameters as functions of the mean grain size
# Mz (phi units, Mz = -log2 of the diameter in mm), as printed in two open papers that
# attribute them to it (their Ref. 19, APL-UW TR 9407; the handbook was not read):
#
#   Snellen, Eleftherakis, Amiri-Simkooei, Koomans and Simons, "An inter-comparison of sediment
#   classification methods based on multi-beam echo-sounder backscatter and sediment natural
#   radioactivity data", J. Acoust. Soc. Am. 134 (2), 959-970 (2013), eq. (5) and the text
#   after it (author copy: https://repository.tudelft.nl/record/uuid:118445fc-6696-4fe2-abae-53272b80db43):
#
#       w2 = 0.00207 h^2  [cm^4],   h = (2.03846 - 0.26923 Mz) / (1 + 0.076923 Mz)   -1 <= Mz < 5
#                                   h = 0.5                                           5 <= Mz <= 9
#       sigma2 = 0.002 for -1 <= Mz < 5.5,   0.001 for 5.5 <= Mz < 9
#
#     with the relief spectrum W(K) = (h0 K)^(-gamma) w2, h0 = 1 cm (their Sec. III.B), which is
#     the convention of `w2_from_cm4`.
#   Snellen, Siemes and Simons, "Model-based sediment classification using single-beam
#   echosounder signals", J. Acoust. Soc. Am. 129 (5), 2878-2888 (2011), Sec. II.A and Table II
#   (https://repository.tudelft.nl/record/uuid:b635a0a8-2ca1-44ae-8fbc-0866b3304827): "a value of
#   3.25 for the exponent of the bottom relief spectrum ... is found to work well for many types
#   of sediments", and the table of rho, nu, delta, sigma2 and w2 (cm^4) at Mz = -1, 0, ..., 9
#   that `GRAIN_SIZE_W2_TABLE_CM4` repeats (sigma2 = 0.001 at Mz = 9 is from that table).
#
# These are defaults, not measurements of a site: both papers stress that fitted values of w2
# and sigma2 "often deviate significantly" from them (2013, Sec. III.B), and the 2013 paper's
# inversions at 300 kHz found sigma2 about half the rule for Mz < 5.5.
GRAIN_SIZE_GAMMA2 = 3.25
GRAIN_SIZE_W2_TABLE_CM4 = {-1.0: 0.0129, 0.0: 0.0086, 1.0: 0.0056, 2.0: 0.0035, 3.0: 0.0021, 4.0: 0.0011,
                           5.0: 0.0005, 6.0: 0.0005, 7.0: 0.0005, 8.0: 0.0005, 9.0: 0.0005}
"""Snellen, Siemes and Simons (2011), Table II, column w2 (cm^4), typed as printed (four decimals)."""


def _check_grain_size(phi: float) -> float:
    phi = float(phi)
    if not -1.0 <= phi <= 9.0:
        raise ValueError(f"mean grain size {phi} phi is outside the range of the published relations (-1 to 9 phi)")
    return phi


def grain_size_w2_cm4(phi: float) -> float:
    """Roughness spectral strength in cm^4 (1 cm reference length) for a mean grain size in phi
    units: Snellen et al. (2013), eq. (5) (comment above)."""
    phi = _check_grain_size(phi)
    h = (2.03846 - 0.26923 * phi) / (1.0 + 0.076923 * phi) if phi < 5.0 else 0.5
    return 0.00207 * h * h


def grain_size_sigma2(phi: float) -> float:
    """Volume parameter for a mean grain size in phi units: 0.002 below 5.5 phi, 0.001 from
    5.5 phi (Snellen et al. 2013, text after eq. (5))."""
    return 0.002 if _check_grain_size(phi) < 5.5 else 0.001


def grain_size_scattering(phi: float) -> dict:
    """{w2 (m^(4 - gamma2)), gamma2, sigma2} for a mean grain size in phi units."""
    return {"w2": w2_from_cm4(grain_size_w2_cm4(phi), GRAIN_SIZE_GAMMA2), "gamma2": GRAIN_SIZE_GAMMA2,
            "sigma2": grain_size_sigma2(phi)}


GRAIN_SIZE_TABLE = (
    (-1.0, 2.492, 1.337, 0.01705), (-0.5, 2.401, 1.3067, 0.01667), (0.0, 2.314, 1.2778, 0.01630),
    (0.5, 2.231, 1.2503, 0.01638), (1.0, 2.151, 1.2241, 0.01645), (1.5, 1.845, 1.1782, 0.01624),
    (2.0, 1.615, 1.1396, 0.01610), (2.5, 1.451, 1.1073, 0.01602), (3.0, 1.339, 1.0800, 0.01728),
    (3.5, 1.268, 1.0568, 0.01875), (4.0, 1.224, 1.0364, 0.02019), (4.5, 1.195, 1.0179, 0.02158),
    (5.0, 1.169, 0.9999, 0.01261), (5.5, 1.149, 0.9885, 0.00676), (6.0, 1.149, 0.9873, 0.00386),
    (6.5, 1.148, 0.9861, 0.00306), (7.0, 1.147, 0.9849, 0.00242), (7.5, 1.147, 0.9837, 0.00194),
    (8.0, 1.146, 0.9824, 0.00163), (9.0, 1.145, 0.9800, 0.00148))
"""(bulk grain size index in phi, density ratio, sound speed ratio, loss parameter) as printed in
Cintron, "Environmental impact on mine hunting in the Yellow Sea using the CASS/GRAB model", MS
thesis, Naval Postgraduate School (2001), https://hdl.handle.net/10945/10842, Table 1 ("APL/UW
TR9407 Geo-acoustic parameters associated with bulk grain size index used by the CASS/GRAB
model ... From NAVOCEANO 1999"), rows "Sandy Gravel" (-1) to "Clay" (9). Its three coarser rows
(Rough Rock, Rock, "Gravel, Cobble or Pebble": density ratio 2.5, speed ratio 2.5, 2.5 and 1.8,
loss 0.0137, with index values -9, -7 and -3 that are labels, not grain sizes) are not fluids
and are not here."""


def sediment_from_grain_size(phi: float) -> dict:
    """The six parameters for a mean grain size in phi units (-1 to 9), for a site whose grain
    size is known and that is not one of the named classes.

    rho, nu and delta are interpolated linearly between the tabulated classes of
    `GRAIN_SIZE_TABLE` (every half phi). The interpolation is ours: the handbook's own
    regressions of these three against grain size were not found in an open document, and the
    table shows they are piecewise (the density ratio drops from 2.151 to 1.615 between 1 and
    2 phi), so between two classes the result is a guess good to the difference of its
    neighbours. w2, gamma2 and sigma2 are `grain_size_scattering`."""
    phi = _check_grain_size(phi)
    table = np.array(GRAIN_SIZE_TABLE)
    out = {key: float(np.interp(phi, table[:, 0], table[:, i])) for i, key in ((1, "rho"), (2, "nu"), (3, "delta"))}
    return dict(out, **grain_size_scattering(phi))


# --- Rock and other bottoms that are not fluid sediments: an empirical law ------------------------

LAMBERT_MAX_MU_DB = -10.0 * math.log10(math.pi)
"""The largest Lambert parameter energy conservation allows, 10 log10(1 / pi) = -4.97 dB: a
surface that scatters all the incident power with the Lambert pattern (Olson, Lyons and Saebo,
arXiv:1601.06078, introduction; the integral of mu sin(ti) sin(ts) over the hemisphere is pi mu sin(ti))."""


def lambert_backscatter_db(grazing, mu_db: float):
    """Lambert's law for backscatter, 10 log10(mu sin^2(grazing)) (written as sigma = mu sin^2(theta)
    in Olson, Lyons and Saebo, arXiv:1601.06078, where the Lambert fits are reported)."""
    return float(mu_db) + 20.0 * np.log10(np.maximum(np.sin(np.asarray(grazing, dtype=float)), 1e-300))


def empirical_backscatter(name: str, path=None) -> dict:
    """The empirical backscattering law of a table entry that is not a fluid sediment (rock,
    pebbles): {"model": "lambert", "mu_db", "mu_db_range", ...} as written in the table, whose
    `basis` field cites the measurements. Warns when the entry marks it unconfirmed."""
    entry = _entry(name, path)
    if "empirical_backscatter" not in entry:
        raise ValueError(f"sediment {name!r} has no empirical backscattering law; its fluid-sediment parameters "
                         "are bottom.sediment(name)")
    _warn_status(name, entry, ("empirical_backscatter",))
    law = dict(entry["empirical_backscatter"])
    if law.get("model") != "lambert":
        raise ValueError(f"sediment {name!r}: unknown empirical model {law.get('model')!r}")
    return law


def named_backscattering_strength_db(name: str, grazing, frequency: float, sound_speed: float = 1500.0, path=None,
                                     **model):
    """Backscattering strength in dB of a named table entry: the fluid-sediment model for a
    sediment (extra keywords such as `roughness_model` are passed on), the entry's empirical
    Lambert law for rock and pebbles (which has no frequency dependence)."""
    if "empirical_backscatter" in _entry(name, path):
        return lambert_backscatter_db(grazing, empirical_backscatter(name, path)["mu_db"])
    return backscattering_strength_db(grazing, frequency, sound_speed, **dict(sediment(name, path), **model))
