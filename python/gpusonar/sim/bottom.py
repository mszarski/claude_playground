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

Printed since (record of what agrees; nothing below was changed by it): Jackson and Richardson,
"High-Frequency Seafloor Acoustics", Springer (2007) [J&R], prints the forms derived here, and
tests/python/test_jackson_2007_bottom.py holds ours against them as retyped from
docs/references/jackson_richardson_2007/: the spectrum convention (6.5)-(6.8), the structure
constant (D.11)-(D.13), the cm conversion D.1.3, flat-interface reflection (8.49)-(8.53),
delta and its conversions (2.6), (8.25)-(8.27), perturbation (13.9), (13.21)-(13.25), Kirchhoff
(13.10), (13.12)-(13.17), small-slope (13.11), the composite cut-off and large-scale slope
(13.72)-(13.77) and the volume term (14.7)-(14.9). Not in J&R: our `combined` model. The book's
composite carries Wagner's shadowing factor (13.81)-(13.83); ours has carried it too since C1 of
docs/references/bottom_mourad_jackson_1989.md (`shadowing="wagner"`, the default inside `composite` and
`combined`; `"none"` is the model before), with pi^(-1/2) in (13.82) as Wagner (1967) (22) and Mourad and
Jackson (1989) (25) print it: the book's pi^(+1/2) is a misprint (docs/references/shadowing_wagner_1967.md).

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

with sigma_vol = sigma2 x attenuation in dB/m = sigma2 x (20 / ln 10) k delta / nu. This is
Mourad and Jackson (1989) eq. (39) with its misprint corrected (the page prints |1 - R|^2 for
|1 - R^2|^2; tests/python/test_mourad_jackson_1989.py). Their model averages it over the
large-scale slope and shadows it, (38): the opt-in `volume_model="mourad_jackson_1989"`.

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
bottom; neither could be opened, so the recipe below was written as ours; J&R (2007) (13.80) has
its structure, with the differences said below). The relief is split at a
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
(both axes together: twice the per-axis s^2 above). J&R (2007) print the same split and slope,
(13.72)-(13.77), and attribute k_w h_s = 1/2 to "D. Jackson et al. 1986a"; our
`large_scale_slope` gives the slopes printed for Fig. 13.19 (0.0227, 0.0595, 0.1432). Their
average (13.80) is over the in-plane slope only and adds Wagner's shadowing factor (13.81)-(13.83);
ours multiplies its two-axis average by the same factor (`wagner_bistatic_shadowing`, which is Wagner's
backscatter R in backscatter; its bistatic halves are ours). Olson and Lyons (arXiv:2106.05909,
Sec. VIII, eq. (44), after McDaniel and Gorman 1983) write the composite model as this same
average of the tilted small-scale perturbation result over the large-scale slope density and
give its conditions: k x small-scale rms height below one, grazing angles above about twice
the large-scale rms slope angle, shadowing below it. The handbook's own split and its
interpolation between the Kirchhoff and composite results were not found in an open
document; the 1989 model they grew from prints both (Mourad and Jackson 1989, eqs. (18) and
(33)-(37), read since: the opt-in `mourad_jackson_1989`, section comment below; its split is
k h_s = 1/sqrt(2), not our 1/2). The composite result describes Bragg scattering only: towards the specular
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

"williams_jackson_1998" (opt-in, any geometry): the PUBLISHED smooth minimum, read since. Williams and Jackson
(1998), J. Acoust. Soc. Am. 103, 169-181, eq. (10), join the Kirchhoff result (11)-(15) and first-order
perturbation theory (16)-(19), not the composite result, by sigma = (sigma_k^-2 + sigma_p^-2)^(-1/2), "the
smaller cross section takes precedence", and compare it with bistatic data at 40 kHz
(docs/references/bottom_bistatic_williams_jackson_1998.md). Its two terms are ours (`kirchhoff_strength`,
`perturbation_strength`, equal to the print to 1e-13 dB). The effect the paragraph above describes is real and
is what the published model does: for sands it lies below our default small-slope result by up to 6.1 dB at 20
degrees backscatter (Panama City, 40 kHz) and 6.8 dB at 30 degrees (medium sand, 30 kHz), out of plane by up
to 5.3 dB, and where the two terms are equal (very fine silt) 1.5 dB below both. Against the paper's own
bistatic data (one site, Table AII, 74 values) it is closer than ours (mean +3.3 against +6.1 dB), but the
fluid model is known to overpredict at that site (J&R p. 351), so this does not settle which approximation is
right; the default stays small-slope (D-86; the choice is the owner's, [WJ] 5.4).
"""
from __future__ import annotations

import functools
import math

import numpy as np

from ..acoustics import NOMINAL_SOUND_SPEED
from scipy.interpolate import CubicSpline
from scipy.special import erfcx, hankel1

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
    W [m^4] = 1e-8 beta (0.01 K [1/m])^(-gamma2), so w2 = beta 10^(2 gamma2 - 8). A second print:
    Jackson and Richardson (2007), App. D.1.3, p. 479."""
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
    """The entries of `configs/sediments.yaml` (or `path`) as written. Keys that only inform the
    reader (`sigma2_range`, `basis`, `w2_note`, `note`, `provenance`, ...) are carried as they are and no
    model reads them: `sediment` returns the six parameters only."""
    from pathlib import Path

    import yaml

    from .. import config
    file = Path(path) if path is not None else config.config_dir() / "sediments.yaml"
    with open(file) as f:
        return yaml.safe_load(f) or {}


REFLECTION_KEYS = ("rho", "nu", "delta")
SHEAR_KEYS = ("nu_t", "delta_t")
"""An elastic row's shear speed ratio to the water and shear loss parameter (J&R (9.23), q = t)."""
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
        if "shear_speed" in r:          # an elastic solid (rock): shear speed ratio and loss parameter (J&R (9.23))
            entry = dict({"nu_t": r["shear_speed"] / r["water_sound_speed"], "delta_t": r.get("shear_delta", 0.0)}, **entry)
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


def sediment_reflection(name: str, path=None, shear: bool = True) -> dict:
    """{rho, nu, delta} of a named sediment: all that the flat-interface reflection coefficient
    needs. Works for every entry, the reflection-only ones included. A row that is an elastic
    solid (its `reflection:` block has `shear_speed`; today only `rock`) adds {nu_t, delta_t},
    the shear speed ratio and shear loss parameter of `reflection`, unless `shear` is False (the
    fluid half-space of the same row, as before A8)."""
    entry = _entry(name, path)
    missing = [k for k in REFLECTION_KEYS if k not in entry]
    if missing:
        raise ValueError(f"sediment {name!r} lacks {missing}")
    keys = REFLECTION_KEYS + (SHEAR_KEYS if shear and "nu_t" in entry else ())
    _warn_status(name, entry, keys + (("shear",) if "nu_t" in keys else ()))
    return {k: float(entry[k]) for k in keys}


def fluid_halfspace(name: str, water_sound_speed: float, path=None) -> dict:
    """A named sediment as the fluid half-space the eigenray providers take:
    {speed (m/s), density (relative to the water), attenuation (dB per wavelength)}, with
    speed = nu x the water sound speed given here (the table holds ratios). Always a fluid: a
    row's shear is left out (the `sediment` tier's penetration and the live chain's depth
    models take this; `halfspace` keeps the shear)."""
    par = sediment_reflection(name, path, shear=False)
    return {"speed": par["nu"] * float(water_sound_speed), "density": par["rho"],
            "attenuation": par["delta"] * DB_PER_WAVELENGTH}


def halfspace(name: str, water_sound_speed: float, path=None) -> dict:
    """`fluid_halfspace`, plus {shear_speed (m/s), shear_attenuation (dB per wavelength)} for a
    row that is an elastic solid (`rock`): the half-space of the image-source tier and Bellhop
    (`propagation.rayleigh_reflection`). Equal to `fluid_halfspace` for every fluid row."""
    par = sediment_reflection(name, path)
    out = {"speed": par["nu"] * float(water_sound_speed), "density": par["rho"],
           "attenuation": par["delta"] * DB_PER_WAVELENGTH}
    if "nu_t" in par:
        out.update(shear_speed=par["nu_t"] * float(water_sound_speed), shear_attenuation=par["delta_t"] * DB_PER_WAVELENGTH)
    return out


def _vertical(k2, horizontal):
    b = np.sqrt(k2 - horizontal**2 + 0j)
    return np.where(b.imag < 0.0, -b, b)          # decays away from the interface


def reflection(grazing, rho: float, nu: float, delta: float, nu_t: float = 0.0, delta_t: float = 0.0):
    """Flat-interface pressure reflection coefficient V of the fluid sediment at a grazing angle
    (equal to Jackson and Richardson 2007, eqs. (8.49)-(8.53)).

    With a shear speed ratio `nu_t` > 0 (and shear loss parameter `delta_t`) the bottom is an
    elastic solid and V is J&R's (9.34)-(9.40): V = (z_e - 1) / (z_e + 1),
    z_e = z_wp cos^2 2theta_t + z_wt sin^2 2theta_t, z_wq = a_rho a_q sin theta_w / sin theta_q,
    cos theta_q = a_q cos theta_w, cos 2theta_t = 2 a_t^2 cos^2 theta_w - 1, a_q = nu_q / (1 + i delta_q)
    (9.23) ([N3] Sec. 2.3). It is evaluated multiplied through by the two vertical wavenumbers,
    s_q = sin theta_q / a_q = sqrt(1 / a_q^2 - cos^2 theta_w) on the branch that decays into the
    bottom (as the fluid's), so that no critical angle divides by zero. nu_t = 0 is the fluid
    coefficient, computed by the fluid expression itself (bit-identical to before A8); as nu_t -> 0
    the elastic one tends to it (tests/python/test_elastic_bottom.py). Twins:
    `propagation.rayleigh_reflection`, libs/sim paths.cpp `rayleigh_reflection`."""
    g = np.asarray(grazing, dtype=float)
    kappa = (1.0 + 1j * delta) / nu
    p = _vertical(kappa**2, np.cos(g))
    if nu_t == 0.0:
        return (rho * np.sin(g) - p) / (rho * np.sin(g) + p)
    a_t = nu_t / (1.0 + 1j * delta_t)
    s = _vertical(1.0 / a_t**2, np.cos(g))
    cos2t = 2.0 * a_t**2 * np.cos(g) ** 2 - 1.0
    num = rho * np.sin(g) * (cos2t**2 * s + (1.0 - cos2t**2) * p)       # z_e x p s
    return (num - p * s) / (num + p * s)


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
    """(Ch2, a) of the structure function D(R) = Ch2 R^(2 a) of a power-law relief spectrum
    (the same as Jackson and Richardson 2007, eqs. (D.11)-(D.13): Gamma(2 - a) = (1 - a) Gamma(1 - a))."""
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
    # The power is computed once, not once per b (the same values: 3 times faster to build, which
    # matters to a fit with a free exponent, where every step builds a new table).
    tp = (t * rot) ** (2.0 * a)
    f = np.array([(kernel * np.exp(-b * tp)).sum().real for b in np.exp(log_b)])
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
    composite-roughness split: relief at K > Kc has k x rms height = `small_scale_kh`
    (Jackson and Richardson 2007, eqs. (13.72)-(13.77), with `small_scale_kh` = 1/2)."""
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


# --- Shadowing of the large-scale relief: Wagner (1967) ----------------------------------------------
#
# R. J. Wagner, "Shadowing of Randomly Rough Surfaces", J. Acoust. Soc. Am. 41 (1), 138-147 (1967) [W67],
# extracted in docs/references/shadowing_wagner_1967.md (read on the page images; status words there).
# Geometric optics, one-dimensional in the plane of incidence, a Gaussian height process; diffraction
# ignored ([W67] p. 145). Used by `composite_strength` and `combined_strength` (default "wagner" inside
# those two opt-in models) and by the Mourad-Jackson (1989) opt-in below.

SHADOWING_MODELS = ("wagner", "none")


def _wagner_b(v):
    """[W67] (22): B = [exp(-v^2) - sqrt(pi) v erfc v] / (4 sqrt(pi) v) = [pi^(-1/2) exp(-v^2) - v erfc v] / (4 v)
    (confirmed; [MJ89] (25) prints the same; J&R (2007) (13.82) prints pi^(+1/2), a misprint). Evaluated as
    exp(-v^2) [1 - sqrt(pi) v erfcx(v)] / (4 sqrt(pi) v), which has no cancellation of two tiny numbers at
    large v and underflows to 0 there (no shadowing). inf for v <= 0 (a ray at or below the horizon)."""
    v = np.asarray(v, dtype=float)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore", under="ignore"):
        vv = np.maximum(v, 1e-300)
        b = np.exp(-vv * vv) * np.maximum(1.0 - math.sqrt(math.pi) * vv * erfcx(vv), 0.0) / (4.0 * math.sqrt(math.pi) * vv)
    return np.where(v > 0.0, b, np.inf)


def _wagner_r_of_b(b):
    """(1 - exp(-2B)) / (2B): 1 at B = 0, 0 at B = inf; 1 - B (its series, exact to double precision)
    below B = 1e-8, where B may be subnormal."""
    b = np.asarray(b, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        r = -np.expm1(-2.0 * b) / (2.0 * np.maximum(b, 1e-8))
    return np.where(b < 1e-8, 1.0 - b, np.where(np.isinf(b), 0.0, r))


def wagner_backscatter_shadowing(grazing, total_rms_slope: float, cutoff: float | None = None):
    """Wagner's backscatter shadow function R ([W67] eq. (49) with (20)-(22)):

        R = (1 - exp(-2B)) / (2B),   B = [pi^(-1/2) exp(-v^2) - v erfc v] / (4 v),   v = tan(grazing) / s,

    s = the TOTAL (both-axes) rms slope of the large-scale relief, i.e. sqrt(2 |psi_0''|) of [W67] (21) for
    an isotropic surface = sqrt(2) x the per-axis slope `large_scale_slope` returns (ours, [W67] Sec. 2.1).
    R, not the illuminated fraction S of [W67] (20): R is the factor for backscatter once the facets that
    face away from the source are left out ([W67] (48)-(50)), which a slope average that drops or clamps
    them has already done (S would count them twice). Mourad and Jackson (1989) (24)-(26) and J&R (2007)
    (13.81)-(13.83) use R. Limits (ours): R -> 2 sqrt(pi) v as v -> 0; R = 1 at normal incidence; 0 at and
    below grazing. `cutoff`: R = 1 where v > cutoff ([MJ89] p. 1173 sets it 1 for t > 2, an error under
    1.3e-4); None evaluates it everywhere."""
    g = np.asarray(grazing, dtype=float)
    with np.errstate(over="ignore", invalid="ignore"):
        v = np.where(g > 0.0, np.tan(np.minimum(g, 0.5 * np.pi)) / float(total_rms_slope), 0.0)
    r = _wagner_r_of_b(_wagner_b(v))
    if cutoff is not None:
        r = np.where(v > cutoff, 1.0, r)
    return r


def wagner_bistatic_shadowing(grazing_in, grazing_out, azimuth, total_rms_slope: float):
    """The shadow factor of the bistatic composite average (ours, derived from [W67] (33), (35)-(36) and
    (46)-(47) as [W67] (48) is from (47) and (20); not printed; docs/references/shadowing_wagner_1967.md
    Sec. 2.4 and C-W2):
      backward half, cos(azimuth) < 0: R(min(grazing_in, grazing_out)), the monostatic R of the lower ray
        ([W67] (35)-(36): in the backward half the shadowing is that of the lower ray);
      forward half, cos(azimuth) >= 0: (1 - exp[-2(B_in + B_out)]) / [2(B_in + B_out)] ([W67] (33)).
    [W67] treats rays in the plane of incidence only; applying the two halves out of the plane by the sign
    of cos(azimuth) is OURS. Symmetric in the two angles (reciprocity), and exactly
    `wagner_backscatter_shadowing` in backscatter (azimuth = pi, equal angles)."""
    ti, ts, phi = _angles(grazing_in, grazing_out, azimuth)
    s = float(total_rms_slope)
    backward = wagner_backscatter_shadowing(np.minimum(ti, ts), s)
    with np.errstate(over="ignore", invalid="ignore"):
        vi = np.where(ti > 0.0, np.tan(np.minimum(ti, 0.5 * np.pi)) / s, 0.0)
        vs = np.where(ts > 0.0, np.tan(np.minimum(ts, 0.5 * np.pi)) / s, 0.0)
    forward = _wagner_r_of_b(_wagner_b(vi) + _wagner_b(vs))
    return np.where(np.cos(phi) < 0.0, backward, forward)


def _check_shadowing(shadowing: str) -> str:
    if shadowing not in SHADOWING_MODELS:
        import difflib
        near = difflib.get_close_matches(str(shadowing), SHADOWING_MODELS, n=1)
        raise ValueError(f"unknown shadowing {shadowing!r}" + (f"; did you mean {near[0]!r}?" if near else "")
                         + f" (one of {SHADOWING_MODELS})")
    return shadowing


def _shadowed(average, ti, ts, phi, slope: float, shadowing: str):
    """The slope average times Wagner's factor ("wagner") or unchanged ("none": the same array)."""
    if _check_shadowing(shadowing) == "none":
        return average
    return average * wagner_bistatic_shadowing(ti, ts, phi, math.sqrt(2.0) * slope)


def composite_strength(grazing_in, grazing_out, frequency: float, sound_speed: float, *, rho: float, nu: float,
                       delta: float, w2: float, gamma2: float, azimuth=np.pi, small_scale_kh: float = 0.5,
                       slope_nodes: int = 12, shadowing: str = "wagner", **_ignored):
    """Perturbation cross section of the small-scale relief (K > Kc) averaged over the slopes
    of the large-scale relief (composite roughness). For low and moderate grazing angles.
    `shadowing`: "wagner" (the default since C1 of docs/references/bottom_mourad_jackson_1989.md)
    multiplies the average by Wagner's shadow function, `wagner_bistatic_shadowing` with the total
    slope sqrt(2) x per-axis (J&R (2007) (13.80)-(13.83) and Mourad and Jackson (1989) (23)-(26) both
    print it); "none" is the model before C1, bit for bit."""
    ti, ts, phi = _angles(grazing_in, grazing_out, azimuth)
    k = 2.0 * np.pi * float(frequency) / float(sound_speed)
    slope, kc = large_scale_slope(frequency, sound_speed, w2=w2, gamma2=gamma2, small_scale_kh=small_scale_kh)

    def small_scale(li, ls, ph):
        factor, q = _perturbation_factor(li, ls, ph, k, rho, nu, delta)
        return factor * np.where(q >= kc, _power_law(q, w2, gamma2), 0.0)
    return _shadowed(_slope_average(small_scale, ti, ts, phi, slope, slope_nodes), ti, ts, phi, slope, shadowing)


def combined_strength(grazing_in, grazing_out, frequency: float, sound_speed: float, *, rho: float, nu: float,
                      delta: float, w2: float, gamma2: float, azimuth=np.pi, small_scale_kh: float = 0.5,
                      slope_nodes: int = 12, shadowing: str = "wagner", **_ignored):
    """The small-slope cross section averaged over the large-scale slopes: Kirchhoff near the
    specular direction, perturbation theory at intermediate angles, composite roughness at
    small grazing angles. `shadowing` as for `composite_strength` (default "wagner"; "none" is the
    model before C1, bit for bit)."""
    ti, ts, phi = _angles(grazing_in, grazing_out, azimuth)
    slope, _ = large_scale_slope(frequency, sound_speed, w2=w2, gamma2=gamma2, small_scale_kh=small_scale_kh)
    par = dict(rho=rho, nu=nu, delta=delta, w2=w2, gamma2=gamma2)
    return _shadowed(_slope_average(lambda li, ls, ph: small_slope_strength(li, ls, frequency, sound_speed, azimuth=ph, **par),
                                    ti, ts, phi, slope, slope_nodes), ti, ts, phi, slope, shadowing)


# --- Williams and Jackson (1998): the published bistatic roughness model (opt-in) -----------------
#
# K. L. Williams and D. R. Jackson, "Bistatic bottom scattering: Model, experiments, and model/data comparison",
# J. Acoust. Soc. Am. 103 (1), 169-181 (1998) [WJ], extracted in docs/references/bottom_bistatic_williams_jackson_1998.md
# (read on the page images; "confirmed" there = printed). The roughness cross section, any geometry:
#   sigma_br = [sigma_kr^eta + sigma_pr^eta]^(1/eta),  eta = -2: "the smaller cross section takes precedence"  (10)
#   sigma_kr  the Kirchhoff approximation (11)-(15), R at the facet angle arcsin(Delta): `kirchhoff_strength`
#   sigma_pr  first-order perturbation theory (16)-(19): `perturbation_strength`
# Both terms are ours, unchanged: they equal the print to 2.8e-14 and 5.7e-14 dB (the note Sec. 4 items 3-4). The
# printed reason for the join: with a power-law spectrum perturbation theory overpredicts near specular (W(K)
# diverges at K = 0) and Kirchhoff overpredicts elsewhere (Thorsos 1990, via secondary). No slope averaging, no
# shadowing. Stated scope (pp. 169-170): 10-100 kHz; a lossy fluid sediment (no elasticity, no porosity), no
# gradients; isotropic Gaussian relief with a power-law spectrum, 2 <= gamma2 <= 4 ((6); the structure function
# needs 2 < gamma2 < 4, so the end points are refused); transverse isotropy (only the azimuth difference counts).
# Where the two terms are equal (smooth relief: very fine silt everywhere; Panama City at about 10 deg backscatter)
# (10) lies 10 log10(2) / 2 = 1.5 dB below both: that is the printed rule, not an error.
# Against our default (small-slope, D-86; both printed, so neither wins by the owner's rule: [WJ] 5.4 is the
# owner's decision): -6.1 dB at 20 deg backscatter (Panama City, 40 kHz), -6.8 dB at 30 deg (medium sand, 30 kHz),
# within 0.2 dB at 5 deg and at specular for sands (the note Sec. 4 item 1).

WILLIAMS_JACKSON_1998 = "williams_jackson_1998"

WJ98_FREQUENCY_RANGE = (10e3, 100e3)
"""[WJ] title and p. 169: "10-100 kHz" (confirmed)."""

_warned_wj98_range = False


def _wj98_warn_range(frequency: float) -> None:
    global _warned_wj98_range
    lo, hi = WJ98_FREQUENCY_RANGE
    if not lo <= float(frequency) <= hi and not _warned_wj98_range:
        _warned_wj98_range = True
        import warnings
        warnings.warn(f"williams_jackson_1998: {float(frequency) / 1e3:g} kHz is outside the paper's 10-100 kHz (Williams and "
                      "Jackson 1998, p. 169); the model is evaluated as printed. This warning is given once.", stacklevel=3)


def williams_jackson_1998_roughness(grazing_in, grazing_out, frequency: float, sound_speed: float, *, rho: float, nu: float,
                                    delta: float, w2: float, gamma2: float, azimuth=np.pi, **_ignored):
    """[WJ] (10) with eta = -2: (sigma_kr^-2 + sigma_pr^-2)^(-1/2), the Kirchhoff (`kirchhoff_strength`, [WJ] (11)-(15))
    and first-order perturbation (`perturbation_strength`, (16)-(19)) cross sections joined so that the smaller takes
    precedence. Any geometry: `azimuth` pi is backscatter, 0 forward ([WJ] (2); our convention). In the exact
    specular direction (horizontal wave-vector change zero) the perturbation term is infinite and (10) gives the
    Kirchhoff term; at zero grazing both are zero and so is this. Module comment above for the scope; outside
    10-100 kHz a warning (once)."""
    _wj98_warn_range(frequency)
    ti, ts, phi = _angles(grazing_in, grazing_out, azimuth)
    par = dict(rho=rho, nu=nu, delta=delta, w2=w2, gamma2=gamma2, azimuth=phi)
    sk = np.asarray(kirchhoff_strength(ti, ts, frequency, sound_speed, **par), dtype=float)
    sp = np.asarray(perturbation_strength(ti, ts, frequency, sound_speed, **par), dtype=float)
    q_h, _ = _kirchhoff_geometry(ti, ts, phi, 2.0 * np.pi * float(frequency) / float(sound_speed))
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        joined = 1.0 / np.sqrt(1.0 / sk**2 + 1.0 / sp**2)
    joined = np.where(q_h > 0.0, joined, sk)
    return np.where((sk > 0.0) & ((sp > 0.0) | (q_h == 0.0)), joined, 0.0)


ROUGHNESS_MODELS = ("perturbation", "kirchhoff", "small_slope", "composite", "combined")
"""The roughness models for any geometry (bistatic). The fit (`sediment_fit`) and its model spread use these."""

OPT_IN_ROUGHNESS_MODELS = (WILLIAMS_JACKSON_1998,)
"""Roughness models for any geometry (bistatic) that are opt-in: Williams and Jackson (1998), the published
bistatic join of Kirchhoff and perturbation theory. Offered by the fit (`sediment_fit.fit`, tools/fit_sediment.py)
but kept out of `ROUGHNESS_MODELS`, which the fit's model spread (`sediment_fit.roughness_model_spread_db`) and
tools/boundary_models_study.py iterate over: their numbers stay what they were before it existed."""

BACKSCATTER_ROUGHNESS_MODELS = ("mourad_jackson_1989",)
"""Roughness models for backscatter only (opt-in): Mourad and Jackson (1989), whose paper is monostatic."""

DEFAULT_ROUGHNESS_MODEL = "small_slope"
"""The roughness model used when none is named (D-86; "perturbation" before). Reasons: the
lowest-order small-slope result is what current work uses and it is valid wherever first-order
perturbation theory is, and beyond (Olson, J. Acoust. Soc. Am. 160, 1918-1935, 2026,
https://hdl.handle.net/10945/76181: "PT is accurate whenever SSA(2) is accurate"); it equals
the perturbation result for small relief and the Kirchhoff result in the specular direction
(tests in `test_bottom_models.py`), where perturbation theory with a power-law spectrum
diverges. Evidence is the literature (PHYSICS_NOTES Sec. 23.2), not a measurement of ours."""

_warned_no_small_slope = False

LARGE_GAMMA2 = 3.7
"""Above this relief spectral exponent `roughness_strength` warns (once per process per model) for the models in
`LARGE_GAMMA2_MODELS`: Jackson and Briggs (1992), p. 974 (confirmed, docs/references/
bottom_backscatter_jackson_briggs_1992.md Sec. 2.2): an exponent near 4 extrapolated below the measured wavenumbers
gives an unrealistic large-scale slope (Quinault along strike, gamma2 3.92: 33 deg at 35 kHz) and too strong
scattering, and past applications had exponents "generally less than 3.7". Values are not changed."""

LARGE_GAMMA2_MODELS = ("small_slope", "composite", "combined", "williams_jackson_1998")
"""The models whose result rests on the spectrum's extrapolation to low wavenumbers, through the structure function
(small-slope, the Kirchhoff part of williams_jackson_1998) or the large-scale slope (composite, combined): the list
of [JB] 5.3. First-order perturbation theory alone uses the spectrum at the Bragg wavenumber only (no warning);
`kirchhoff` alone is a near-specular model (no warning); mourad_jackson_1989 has its own stated range."""

_warned_large_gamma2: set = set()


def _warn_large_gamma2(roughness_model: str, gamma2: float) -> None:
    # From 4 on these models have no structure function and raise (structure_constant): nothing to warn about.
    if roughness_model in LARGE_GAMMA2_MODELS and LARGE_GAMMA2 < gamma2 < 4.0 and roughness_model not in _warned_large_gamma2:
        _warned_large_gamma2.add(roughness_model)
        import warnings
        warnings.warn(f"bottom roughness ({roughness_model}): gamma2 = {gamma2:g} is above 3.7. Jackson and Briggs (1992), "
                      "p. 974: an exponent near 4 extrapolated below the measured wavenumbers gives an unrealistic "
                      "large-scale slope (33 deg at 35 kHz for their gamma2 3.92) and too strong scattering; past fits had "
                      "exponents generally below 3.7. With their along-strike Quinault inputs our small-slope result is "
                      "5.6 dB (30 deg) and 10.3 dB (40 deg) above the across-strike one at 35 kHz, an anisotropy their data "
                      "do not show. Evaluated as given; this warning is given once per model.", stacklevel=3)


def roughness_strength(grazing_in, grazing_out, frequency: float, sound_speed: float, *, rho: float, nu: float,
                       delta: float, w2: float, gamma2: float, azimuth=np.pi, roughness_model: str | None = None,
                       small_scale_kh: float = 0.5, shadowing: str | None = None, **_ignored):
    """Cross section per unit area and solid angle from interface roughness. `azimuth` is the
    angle between the horizontal directions of the incident and scattered waves: pi for
    backscatter, 0 for forward scatter. `roughness_model`: "small_slope" (the default,
    `DEFAULT_ROUGHNESS_MODEL`), "perturbation" (first order; the default before D-86),
    "kirchhoff", "composite" or "combined" (module docstring), "williams_jackson_1998" (opt-in, any
    geometry: the published bistatic join of Kirchhoff and perturbation theory, `williams_jackson_1998_roughness`),
    or for backscatter only "mourad_jackson_1989" (the 1989 model's roughness term, opt-in). The small-slope result needs a
    structure function, i.e. 2 < gamma2 < 4: outside that range, when no model is named, the
    perturbation result is used and a warning says so once (naming "small_slope" raises).
    `shadowing` ("wagner" | "none"; None = the model's own default): Wagner's shadow function, carried by
    "composite" and "combined" (default "wagner" there) and by "mourad_jackson_1989"; the other models have
    none, so naming "wagner" with them is an error. Above gamma2 = 3.7 the models of `LARGE_GAMMA2_MODELS` warn once
    each (Jackson and Briggs 1992, p. 974; `LARGE_GAMMA2`); the values are unchanged."""
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
    if shadowing is not None and _check_shadowing(shadowing) != "none" and roughness_model in ("perturbation", "kirchhoff", "small_slope",
                                                                                                 WILLIAMS_JACKSON_1998):
        raise ValueError(f"shadowing {shadowing!r}: roughness_model {roughness_model!r} has no shadowing (it is carried by "
                         "'composite', 'combined' and 'mourad_jackson_1989')")
    _warn_large_gamma2(roughness_model, gamma2)
    args = (grazing_in, grazing_out, frequency, sound_speed)
    par = dict(rho=rho, nu=nu, delta=delta, w2=w2, gamma2=gamma2, azimuth=azimuth)
    if roughness_model == "perturbation":
        return perturbation_strength(*args, **par)
    if roughness_model == "kirchhoff":
        return kirchhoff_strength(*args, **par)
    if roughness_model == "small_slope":
        return small_slope_strength(*args, **par)
    shade = {} if shadowing is None else {"shadowing": shadowing}
    if roughness_model == "composite":
        return composite_strength(*args, small_scale_kh=small_scale_kh, **shade, **par)
    if roughness_model == "combined":
        return combined_strength(*args, small_scale_kh=small_scale_kh, **shade, **par)
    if roughness_model == MOURAD_JACKSON_1989:
        return mourad_jackson_1989_roughness(*args, **shade, **par)
    if roughness_model == WILLIAMS_JACKSON_1998:
        return williams_jackson_1998_roughness(*args, **par)
    known = ROUGHNESS_MODELS + OPT_IN_ROUGHNESS_MODELS + BACKSCATTER_ROUGHNESS_MODELS
    import difflib
    near = difflib.get_close_matches(str(roughness_model), known, n=1)
    raise ValueError(f"unknown roughness_model {roughness_model!r}" + (f"; did you mean {near[0]!r}?" if near else "")
                     + f"; one of {known}")


VOLUME_MODELS = ("stockhausen", "mourad_jackson_1989", "williams_jackson_1998")
"""`volume_model`: "stockhausen" (the default: the flat-interface volume term below, Stockhausen (1963) as
generalised for absorption; J&R (14.7)-(14.9); [MJ89] (39) corrected), "mourad_jackson_1989" (that term
averaged over the large-scale slope and shadowed, [MJ89] (38); backscatter only, opt-in) or
"williams_jackson_1998" (opt-in, any geometry: the same transmission through the interface with a volume cross
section from a 3-D heterogeneity spectrum that depends on the bistatic angles, [WJ] (20)-(22);
`williams_jackson_1998_volume`). The last is named after its paper, like the 1989 pair, rather than "spectral":
it is (21) with the paper's signed mu, i.e. J&R (14.10) with rho_rk = -1 and an isotropic power law (no L_c, no
Lambda), not the general (14.10) the plan's G-15 lists."""

VOLUME_SPECTRAL_KEYS = ("w3", "gamma3", "mu")
"""The parameters of `volume_model="williams_jackson_1998"` ([WJ] (7)-(9)): w3 the volume spectral strength in
m^(3 - gamma3) (`w3_from_cm3` from the paper's cm^3), gamma3 its exponent, mu the ratio of the compressibility
to the density fluctuation, SIGNED as the paper prints it (Table I: -0.69, -2.44). No class values exist."""


def w3_from_cm3(strength_cm3: float, gamma3: float) -> float:
    """Volume spectral strength in m^(3 - gamma3) (wavenumbers in rad/m) from the form with a 1 cm reference length,
    W_rr(k) = w3 (h0 k)^(-gamma3), h0 = 1 cm, w3 in cm^3 ([WJ] (7) and Table I, "at 2 pi / lambda = 1 cm^-1"):
    W [m^3] = 1e-6 w3 (0.01 k [1/m])^(-gamma3), so w3 [m^(3 - gamma3)] = w3 [cm^3] x 10^(2 gamma3 - 6) (ours, the
    argument of `w2_from_cm4`; the note Sec. 2: 1.3e-4 cm^3 -> 0.013 m^-1 at gamma3 = 4)."""
    return float(strength_cm3) * 10.0 ** (2.0 * float(gamma3) - 6.0)


def williams_jackson_1998_volume_cross_section(grazing_in, grazing_out, frequency: float, sound_speed: float, *, nu: float,
                                               delta: float, w3: float, gamma3: float, mu: float, azimuth=np.pi, **_ignored):
    """[WJ] (21)-(22): the volume scattering cross section per unit volume (1/m) of weak density and compressibility
    fluctuations with an isotropic power-law spectrum W_rr(k) = w3 / k^gamma3 (w3 in m^(3 - gamma3)), first order (Born):

        sigma_v = (pi / 2) k^4 |mu kappa^2 + cos ti cos ts cos phi - P(ti) P(ts)|^2 W_rr(Delta k),
        Delta k = k |4 Delta_t^2 + {Re[P(ti) + P(ts)]}^2|^(1/2),   kappa = (1 + i delta) / nu,  P = (kappa^2 - cos^2)^(1/2)

    (Im P > 0: not stated, ours as `_vertical`). mu signed as printed (negative: anti-correlated); this is J&R (2007)
    (14.10) with rho_rk = -1 and mu' = -mu (ours, algebra; the note Sec. 2; tests). The bracket (monopole plus dipole)
    vanishes near one azimuth: a null the isotropic term has not (Eckernfoerde 10/10 deg: phi 39.3 deg)."""
    ti, ts, phi = _angles(grazing_in, grazing_out, azimuth)
    k = 2.0 * np.pi * float(frequency) / float(sound_speed)
    kappa2 = ((1.0 + 1j * delta) / nu) ** 2
    p_i, p_s = _vertical(kappa2, np.cos(ti)), _vertical(kappa2, np.cos(ts))
    dt2 = 0.25 * np.maximum(np.cos(ti) ** 2 + np.cos(ts) ** 2 - 2.0 * np.cos(ti) * np.cos(ts) * np.cos(phi), 0.0)
    dk = k * np.sqrt(np.abs(4.0 * dt2 + (p_i + p_s).real ** 2))
    bracket = mu * kappa2 + np.cos(ti) * np.cos(ts) * np.cos(phi) - p_i * p_s
    with np.errstate(divide="ignore"):
        return 0.5 * np.pi * k**4 * np.abs(bracket) ** 2 * _power_law(dk, w3, gamma3)


def _through_interface(ti, ts, k: float, rho: float, nu: float, delta: float, sigma_vol):
    """sigma_vol |T(ti)|^2 |T(ts)|^2 / (rho^2 2 (Im b2i + Im b2s)): the flat-interface volume term (module docstring;
    [WJ] (20); J&R (14.8)) for a volume cross section per unit volume `sigma_vol` (1/m, a number or an array)."""
    k2sq = (k * (1.0 + 1j * delta) / nu) ** 2
    b2i, b2s = _vertical(k2sq, k * np.cos(ti)), _vertical(k2sq, k * np.cos(ts))
    t_in = 1.0 + reflection(ti, rho, nu, delta)
    t_out = 1.0 + reflection(ts, rho, nu, delta)
    decay = 2.0 * (b2i.imag + b2s.imag)
    return np.where(decay > 0.0, sigma_vol * np.abs(t_in) ** 2 * np.abs(t_out) ** 2 / (rho**2 * np.maximum(decay, 1e-300)), 0.0)


def williams_jackson_1998_volume(grazing_in, grazing_out, frequency: float, sound_speed: float, *, rho: float, nu: float,
                                 delta: float, w3: float | None = None, gamma3: float | None = None, mu: float | None = None,
                                 azimuth=np.pi, **_ignored):
    """[WJ] (20) with (21)-(22): `williams_jackson_1998_volume_cross_section` through the flat interface, our
    transmission form (equal to the print to 1.1e-14 dB, the note Sec. 4 item 5). Any geometry. For a sigma_v that
    does not depend on angle it is our default term with sigma2 = sigma_v / (attenuation in dB/m) (tests). Needs
    w3, gamma3, mu (`VOLUME_SPECTRAL_KEYS`); a named site row's `volume_spectral:` block gives them
    (`volume_spectral`); no class values exist. Stated scope as for the roughness term (10-100 kHz; weak
    inhomogeneities; the paper treats volume scattering as an interface cross section, slight penetration)."""
    missing = [n for n, v in (("w3", w3), ("gamma3", gamma3), ("mu", mu)) if v is None]
    if missing:
        raise ValueError(f"volume_model 'williams_jackson_1998' needs w3, gamma3 and mu (missing {missing}): the volume "
                         "spectrum of Williams and Jackson (1998) (7)-(9); no class values exist for them (the paper prints "
                         "two sites, Table I): give them, or name a site row with a `volume_spectral:` block "
                         "(williams_jackson_1998_panama_city; bottom.volume_spectral)")
    _wj98_warn_range(frequency)
    ti, ts, phi = _angles(grazing_in, grazing_out, azimuth)
    k = 2.0 * np.pi * float(frequency) / float(sound_speed)
    sigma_vol = williams_jackson_1998_volume_cross_section(ti, ts, frequency, sound_speed, nu=nu, delta=delta, w3=float(w3),
                                                           gamma3=float(gamma3), mu=float(mu), azimuth=phi)
    return _through_interface(ti, ts, k, rho, nu, delta, sigma_vol)


def volume_spectral(name: str, path=None) -> dict:
    """{w3 (m^(3 - gamma3)), gamma3, mu} of a named row's `volume_spectral: {w3_cm3, gamma3, mu}` block (configs/
    sediments.yaml; today only williams_jackson_1998_panama_city, [WJ] Table I): the parameters of
    `volume_model="williams_jackson_1998"`. A row without the block is refused: no class values exist for them."""
    entry = _entry(name, path)
    if "volume_spectral" not in entry:
        raise ValueError(f"sediment {name!r} has no `volume_spectral:` block: no class values exist for w3, gamma3 and mu "
                         "(Williams and Jackson 1998 print two sites; J&R 2007 Table 14.1 sites only), so volume_model "
                         "'williams_jackson_1998' needs them given, or a site row that has them "
                         "(williams_jackson_1998_panama_city)")
    block = {k: float(v) for k, v in entry["volume_spectral"].items()}
    return {"w3": w3_from_cm3(block["w3_cm3"], block["gamma3"]) if "w3_cm3" in block else block["w3"],
            "gamma3": block["gamma3"], "mu": block["mu"]}


def item_volume_spectral(item: dict) -> dict:
    """For a scenario item (`reverberation` `fluid_sediment`, `seafloor` `fluid_sediment`): {w3, gamma3, mu} as floats
    when its `volume_model` is "williams_jackson_1998", the item's own values completed from its named `sediment`'s
    `volume_spectral:` block (`volume_spectral`); {} for any other volume model. Refused, naming the reason, when the
    three cannot all be found (no class values exist)."""
    if item.get("volume_model") != WILLIAMS_JACKSON_1998:
        return {}
    given = {k: float(item[k]) for k in VOLUME_SPECTRAL_KEYS if k in item}
    if len(given) < len(VOLUME_SPECTRAL_KEYS):
        if "sediment" not in item:
            raise ValueError(f"volume_model 'williams_jackson_1998' needs w3, gamma3 and mu (missing "
                             f"{[k for k in VOLUME_SPECTRAL_KEYS if k not in given]}): no class values exist for them "
                             "(Williams and Jackson 1998 print two sites); give them, or name a site row with a "
                             "`volume_spectral:` block (sediment: williams_jackson_1998_panama_city)")
        given = dict(volume_spectral(item["sediment"], item.get("sediment_file")), **given)
    return given


def volume_strength(grazing_in, grazing_out, frequency: float, sound_speed: float, *, rho: float, nu: float,
                    delta: float, sigma2: float | None = None, volume_model: str | None = None, **_ignored):
    """Cross section per unit area and solid angle from scatterers inside the sediment
    (equal to Jackson and Richardson 2007, eqs. (14.7)-(14.9), with (8.26) for the attenuation).
    `volume_model` (`VOLUME_MODELS`): None or "stockhausen" is this term; "mourad_jackson_1989" is
    `mourad_jackson_1989_volume` (backscatter only; needs w2 and gamma2, which `strength` passes);
    "williams_jackson_1998" is `williams_jackson_1998_volume` (any geometry; needs w3, gamma3, mu; `sigma2` unused)."""
    if sigma2 is None and volume_model != WILLIAMS_JACKSON_1998:
        raise ValueError(f"volume_model {volume_model or 'stockhausen'!r} needs sigma2")
    if volume_model is not None and volume_model != "stockhausen":
        if volume_model == MOURAD_JACKSON_1989:
            return mourad_jackson_1989_volume(grazing_in, grazing_out, frequency, sound_speed, rho=rho, nu=nu, delta=delta,
                                              sigma2=sigma2, **_ignored)
        if volume_model == WILLIAMS_JACKSON_1998:
            return williams_jackson_1998_volume(grazing_in, grazing_out, frequency, sound_speed, rho=rho, nu=nu, delta=delta,
                                                **_ignored)
        import difflib
        near = difflib.get_close_matches(str(volume_model), VOLUME_MODELS, n=1)
        raise ValueError(f"unknown volume_model {volume_model!r}" + (f"; did you mean {near[0]!r}?" if near else "")
                         + f" (one of {VOLUME_MODELS})")
    ti, ts = np.broadcast_arrays(np.asarray(grazing_in, dtype=float), np.asarray(grazing_out, dtype=float))
    k = 2.0 * np.pi * float(frequency) / float(sound_speed)
    sigma_vol = sigma2 * (20.0 / np.log(10.0)) * k * delta / nu   # 1/m
    return _through_interface(ti, ts, k, rho, nu, delta, sigma_vol)


def strength(grazing_in, grazing_out, frequency: float, sound_speed: float, azimuth=np.pi, **params):
    """Roughness plus volume."""
    return (roughness_strength(grazing_in, grazing_out, frequency, sound_speed, azimuth=azimuth, **params)
            + volume_strength(grazing_in, grazing_out, frequency, sound_speed, azimuth=azimuth, **params))


def backscattering_strength_db(grazing, frequency: float, sound_speed: float = NOMINAL_SOUND_SPEED, **params):
    return 10.0 * np.log10(np.maximum(strength(grazing, grazing, frequency, sound_speed, **params), 1e-300))


def reverberation_strength(frequency: float, sound_speed: float, **params):
    """The model as the function of (grazing in, grazing out) the reverberation code takes
    (backscatter geometry: the scattered wave returns over the incident one)."""
    return lambda g_in, g_out: strength(g_in, g_out, frequency, sound_speed, **params)


def equivalent_lambert_mu_db(grazing: float, frequency: float, sound_speed: float = NOMINAL_SOUND_SPEED, **params) -> float:
    """The Lambert parameter that gives this model's backscattering strength at one grazing
    angle: for scenes whose seabed is point scatterers with Lambert's law (set their `mu_db`
    to this at the middle of the swath)."""
    return float(backscattering_strength_db(grazing, frequency, sound_speed, **params) - 20.0 * np.log10(np.sin(grazing)))


# --- Mourad and Jackson (1989): the published 1989 backscatter model (opt-in) --------------------
#
# P. D. Mourad and D. R. Jackson, "High Frequency Sonar Equation Models for Bottom Backscatter and
# Forward Loss", OCEANS '89 Proceedings, pp. 1168-1175 [MJ89], extracted equation by equation in
# docs/references/bottom_mourad_jackson_1989.md (read on the page images; "confirmed" there = printed).
# Selected by `roughness_model="mourad_jackson_1989"` and `volume_model="mourad_jackson_1989"` (the
# paper's model is both together); neither is a default (D-86's small-slope model stays: J&R 2007, the
# later print, recommend small-slope, p. 336). Implemented as printed:
#   roughness  sigma_r = f(x) sigma_kr + [1 - f(x)] sigma_cr                                      (33)
#              sigma_kr: the paper's closed-form Kirchhoff approximation (19)-(22), NOT our exact
#                        Kirchhoff integral (`kirchhoff_strength`; equal at gamma2 = 3 and at 90 deg)
#              sigma_cr = S(theta, s) F(theta, sigma_pr, s): Wagner's R times the slope average   (23)-(26)
#              F = pi^(-1/2) sum a_n sigma(theta - theta_n), three-point Gauss-Hermite              (27)-(29)
#              (arguments <= 0 set to 1 deg, above 90 deg to 90 deg; in-plane slope only)
#              sigma_pr: first-order perturbation with K = [4 k^2 cos^2 theta + (k/10)^2]^(1/2)  (30)-(32)
#              f(x) = 1 / (1 + e^x), x = 80 (cos theta - cos theta_kdB),
#              cos theta_kdB = (1/C4 + 4)^(-1/4), C4 = 1000^(1/(1 + alpha)) (a q_c^2)^(1/alpha) (34)-(37)
#   volume     sigma_v = S(theta, s) F(theta, sigma_pv, s)                                         (38)
#              sigma_pv = 5 delta sigma2 |1 - R^2|^2 sin^2 theta / (nu ln 10 |P|^2 Im P)          (39)
#              with |1 - R^2|^2: the page prints |1 - R|^2, a misprint (the paper's own figures
#              reproduce only with the square; J&R (14.8); tests/python/test_mourad_jackson_1989.py)
#   slope      s^2 = (2 pi w2)^(1/alpha) / (2 (1 - alpha)) (k^2 / alpha)^((1 - alpha)/alpha)       (18)
#              the TOTAL (both-axes) rms slope of the relief below the K at which the small-scale
#              relief has k h_s = 1/sqrt(2) (ours, algebra: sqrt(2) x `large_scale_slope` with
#              small_scale_kh = 2^(-1/2)); J&R and our `composite` use k h_s = 1/2.
#   S          Wagner's R with t = tan(theta) / s, set to 1 for t > 2 (p. 1173).
# Backscatter only: the paper is monostatic, so grazing_in != grazing_out or azimuth != pi is refused.
# Stated range: 10-100 kHz, sands to clays, relations for 1 <= Mz <= 9 phi (p. 1169); outside it a
# warning (the parameters themselves are what the model takes; `mourad_jackson_1989_sediment` refuses a
# grain size outside 1-9 phi).

MOURAD_JACKSON_1989 = "mourad_jackson_1989"

MJ89_GAUSS_HERMITE_PRINTED = ((0.295410, 1.224745), (1.181636, 0.0), (0.295410, -1.224745))
"""[MJ89] (28)-(29) as printed: (a_n, theta_n / arctan(s)) for n = -1, 0, 1 (confirmed). They are the
three-point Gauss-Hermite weights sqrt(pi)/6, 2 sqrt(pi)/3 and nodes -+sqrt(3/2) (ours: 0.2954090,
1.1816359, 1.2247449) rounded to six decimals; the printed 0.295410 is one unit of the sixth decimal
above sqrt(pi)/6 = 0.29540898 (the effect: 5e-6 dB). The model below uses the exact quadrature
(`numpy.polynomial.hermite.hermgauss(3)`), which is what (27) cites (Carnahan et al. 1969)."""

MJ89_FREQUENCY_RANGE = (10e3, 100e3)
"""[MJ89] abstract: 10-100 kHz (confirmed)."""


def mourad_jackson_1989_relations(phi: float) -> dict:
    """[MJ89] (1), (2), (4)-(7) at a mean grain size Mz in phi (confirmed; "defined for 1 <= Mz <= 9", p. 1169:
    outside that a ValueError): {rho, nu, sigma2, gamma2, w2_cm4, h_cm}. rho and nu with the paper's seven-digit
    constants; h the rms relief of a 100 cm track (6); w2 in cm^4 with h0 = 1 cm (7), for `w2_from_cm4`. The
    step of sigma2 (4) and of h (6) is at 5.0 phi as printed; our class rule (`grain_size_sigma2`, Snellen et
    al.) steps sigma2 at 5.5 phi and stays (the note Sec. 0 item 5). delta (3) needs Hamilton's (1972)
    attenuation, which [MJ89] does not print: `apl_grain_size_relations` has it (J&R (11.3))."""
    mz = float(phi)
    if not 1.0 <= mz <= 9.0:
        raise ValueError(f"mean grain size {mz:g} phi: the Mourad-Jackson (1989) relations are defined for 1 <= Mz <= 9 only")
    if mz <= 5.3:
        rho = -0.0165406 * mz**3 + 0.2290201 * mz**2 - 1.1069031 * mz + 3.0455234
        nu = -0.0014881 * mz**3 + 0.0213937 * mz**2 - 0.1382798 * mz + 1.3424740
    else:
        rho = -0.0012973 * mz + 1.1564755
        nu = -0.0024324 * mz + 1.0018916
    h = (2.03846 - 0.26923 * mz) / (1.0 + 0.076923 * mz) if mz < 5.0 else 0.5
    return {"rho": rho, "nu": nu, "sigma2": 0.002 if mz < 5.0 else 0.001, "gamma2": 3.25, "w2_cm4": 0.00207 * h * h,
            "h_cm": h}


_MJ89_RHO_SPAN = (mourad_jackson_1989_relations(9.0)["rho"], mourad_jackson_1989_relations(1.0)["rho"])
_MJ89_NU_SPAN = (mourad_jackson_1989_relations(9.0)["nu"], mourad_jackson_1989_relations(1.0)["nu"])


def _mj89_check_geometry(grazing_in, grazing_out, azimuth, what: str):
    ti, ts, phi = _angles(grazing_in, grazing_out, azimuth)
    if np.any(ti != ts) or np.any(np.abs(np.cos(phi) + 1.0) > 1e-12):
        raise ValueError(f"{what} 'mourad_jackson_1989' is a backscatter model (the paper is monostatic): grazing_in must "
                         "equal grazing_out and azimuth be pi; for other geometries use the default (small_slope) or "
                         "'composite' / 'combined' with the default volume model")
    return ti


def _mj89_warn_range(frequency: float, rho: float, nu: float) -> None:
    import warnings
    lo, hi = MJ89_FREQUENCY_RANGE
    if not lo <= float(frequency) <= hi:
        warnings.warn(f"mourad_jackson_1989: {float(frequency) / 1e3:g} kHz is outside the paper's 10-100 kHz", stacklevel=3)
    if not (_MJ89_RHO_SPAN[0] - 1e-9 <= rho <= _MJ89_RHO_SPAN[1] + 1e-9 and _MJ89_NU_SPAN[0] - 1e-9 <= nu <= _MJ89_NU_SPAN[1] + 1e-9):
        warnings.warn(f"mourad_jackson_1989: rho {rho:.4g}, nu {nu:.4g} lie outside what the paper's grain-size relations "
                      f"span for 1-9 phi (rho {_MJ89_RHO_SPAN[0]:.4f}-{_MJ89_RHO_SPAN[1]:.4f}, nu {_MJ89_NU_SPAN[0]:.4f}-"
                      f"{_MJ89_NU_SPAN[1]:.4f}): the model was stated for sands to clays", stacklevel=3)


def mourad_jackson_1989_slope(frequency: float, sound_speed: float, *, w2: float, gamma2: float, **_ignored) -> float:
    """[MJ89] (18): the large-scale rms slope s (total, both axes) of the 1989 model, w2 in m^(4 - gamma2)."""
    if not 2.0 < gamma2 < 4.0:
        raise ValueError("the Mourad-Jackson (1989) model needs 2 < gamma2 < 4 ((15))")
    alpha = 0.5 * gamma2 - 1.0
    k = 2.0 * np.pi * float(frequency) / float(sound_speed)
    return float(np.sqrt((2.0 * np.pi * w2) ** (1.0 / alpha) / (2.0 * (1.0 - alpha)) * (k * k / alpha) ** ((1.0 - alpha) / alpha)))


def _mj89_kirchhoff_constants(k: float, w2: float, gamma2: float):
    """(a, b, q_c, alpha) of [MJ89] (20)-(22); C_h^2 of (17) is `structure_constant`."""
    ch2, alpha = structure_constant(w2, gamma2)
    q_c = ch2 * 2.0 ** (1.0 - 2.0 * alpha) * k ** (2.0 * (1.0 - alpha))
    a = (8.0 * alpha**2 * math.gamma(0.5 / alpha + 0.5) / (math.gamma(0.5) * math.gamma(1.0 / alpha) * math.gamma(0.5 / alpha))) ** (2.0 * alpha)
    b = a ** (0.5 + 0.5 / alpha) * math.gamma(1.0 / alpha) / (2.0 * alpha)
    return a, b, q_c, alpha


def mourad_jackson_1989_kirchhoff(grazing, frequency: float, sound_speed: float, *, rho: float, nu: float, delta: float,
                                  w2: float, gamma2: float, **_ignored):
    """[MJ89] (19)-(22): the paper's closed-form approximation to the Kirchhoff backscatter,
        sigma_kr = b q_c |R(90)|^2 / (8 pi [cos^(4 alpha) theta + a q_c^2 sin^4 theta]^((1 + alpha)/(2 alpha))),
    exact at alpha = 1/2 (gamma2 = 3) and at 90 deg; elsewhere it differs from our exact `kirchhoff_strength`
    (e.g. +3.7 dB at gamma2 3.67, 60 deg: the note Sec. 4 D6)."""
    theta = np.asarray(grazing, dtype=float)
    k = 2.0 * np.pi * float(frequency) / float(sound_speed)
    a, b, q_c, alpha = _mj89_kirchhoff_constants(k, w2, gamma2)
    r90 = abs(complex(reflection(0.5 * np.pi, rho, nu, delta))) ** 2
    base = np.abs(np.cos(theta)) ** (4.0 * alpha) + a * q_c**2 * np.sin(theta) ** 4
    return b * q_c * r90 / (8.0 * np.pi * base ** ((1.0 + alpha) / (2.0 * alpha)))


def mourad_jackson_1989_kirchhoff_angle(frequency: float, sound_speed: float, *, w2: float, gamma2: float, **_ignored) -> float:
    """theta_kdB of [MJ89] (36)-(37), radians: the centre of the interpolation, where the Kirchhoff result has
    fallen about 15 dB below its peak (exactly so as theta_kdB -> 90 deg; never below 45 deg, the "+4")."""
    k = 2.0 * np.pi * float(frequency) / float(sound_speed)
    a, _, q_c, alpha = _mj89_kirchhoff_constants(k, w2, gamma2)
    c4 = 1000.0 ** (1.0 / (1.0 + alpha)) * (a * q_c**2) ** (1.0 / alpha)
    return float(np.arccos((1.0 / c4 + 4.0) ** -0.25))


def _mj89_p(theta, nu: float, delta: float):
    kappa = (1.0 + 1j * delta) / nu
    return kappa, _vertical(kappa**2, np.cos(theta))                   # (11)-(12); the root that decays into the sediment


def mourad_jackson_1989_perturbation(grazing, frequency: float, sound_speed: float, *, rho: float, nu: float, delta: float,
                                     w2: float, gamma2: float, low_wavenumber: bool = True, **_ignored):
    """[MJ89] (30)-(32): sigma_pr = 4 k^4 sin^4 theta |Y|^2 W(K_theta),
    Y = [(rho - 1)^2 cos^2 theta + rho^2 - kappa^2] / [rho sin theta + P]^2 ("a modified version of Kuo's
    expression"), K_theta = [4 k^2 cos^2 theta + (k/10)^2]^(1/2); `low_wavenumber=False` drops the (k/10)^2
    term, and then this equals our `perturbation_strength` in backscatter."""
    theta = np.asarray(grazing, dtype=float)
    k = 2.0 * np.pi * float(frequency) / float(sound_speed)
    kappa, p = _mj89_p(theta, nu, delta)
    y = ((rho - 1.0) ** 2 * np.cos(theta) ** 2 + rho**2 - kappa**2) / (rho * np.sin(theta) + p) ** 2
    big_k = np.sqrt(4.0 * k * k * np.cos(theta) ** 2 + ((0.1 * k) ** 2 if low_wavenumber else 0.0))
    return 4.0 * k**4 * np.sin(theta) ** 4 * np.abs(y) ** 2 * _power_law(big_k, w2, gamma2)


def mourad_jackson_1989_volume_unaveraged(grazing, *, rho: float, nu: float, delta: float, sigma2: float, **_ignored):
    """[MJ89] (39) with |1 - R^2|^2 (the page prints |1 - R|^2: a misprint, module comment above). Equal to our
    `volume_strength` in backscatter (to 1e-12 relative, tests/python/test_mourad_jackson_1989.py)."""
    theta = np.asarray(grazing, dtype=float)
    _, p = _mj89_p(theta, nu, delta)
    r = reflection(theta, rho, nu, delta)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = 5.0 * delta * sigma2 * np.abs(1.0 - r**2) ** 2 * np.sin(theta) ** 2 / (nu * math.log(10.0) * np.abs(p) ** 2 * p.imag)
    return np.where(p.imag > 0.0, out, 0.0)


def _mj89_average(fn, theta, s: float):
    """[MJ89] (27)-(29): F = pi^(-1/2) sum_n a_n fn(theta - theta_n), theta_n = x_n arctan(s), three-point
    Gauss-Hermite (exact weights and nodes, `MJ89_GAUSS_HERMITE_PRINTED`); an argument <= 0 is set to 1 deg,
    one above 90 deg to 90 deg (text after (29))."""
    x, w = np.polynomial.hermite.hermgauss(3)
    total = 0.0
    for xn, an in zip(x, w):
        arg = theta - xn * math.atan(s)
        arg = np.where(arg <= 0.0, np.radians(1.0), np.minimum(arg, 0.5 * np.pi))
        total = total + an * fn(arg)
    return total / math.sqrt(math.pi)


def _mj89_shadow(theta, s: float, shadowing: str):
    if _check_shadowing(shadowing) == "none":
        return 1.0
    return wagner_backscatter_shadowing(theta, s, cutoff=2.0)            # (24)-(26), S = 1 for t > 2


def mourad_jackson_1989_roughness(grazing_in, grazing_out, frequency: float, sound_speed: float, *, rho: float, nu: float,
                                  delta: float, w2: float, gamma2: float, azimuth=np.pi, shadowing: str = "wagner",
                                  **_ignored):
    """The roughness term of [MJ89], (13)-(37) (module comment above): backscatter only. `shadowing="none"`
    drops Wagner's factor S (the paper's model has it)."""
    theta = _mj89_check_geometry(grazing_in, grazing_out, azimuth, "roughness_model")
    _mj89_warn_range(frequency, rho, nu)
    par = dict(rho=rho, nu=nu, delta=delta, w2=w2, gamma2=gamma2)
    s = mourad_jackson_1989_slope(frequency, sound_speed, **par)
    composite = _mj89_shadow(theta, s, shadowing) * _mj89_average(
        lambda x: mourad_jackson_1989_perturbation(x, frequency, sound_speed, **par), theta, s)
    kirch = mourad_jackson_1989_kirchhoff(theta, frequency, sound_speed, **par)
    from scipy.special import expit
    f = expit(-80.0 * (np.cos(theta) - math.cos(mourad_jackson_1989_kirchhoff_angle(frequency, sound_speed, **par))))
    return f * kirch + (1.0 - f) * composite


def mourad_jackson_1989_volume(grazing_in, grazing_out, frequency: float, sound_speed: float, *, rho: float, nu: float,
                               delta: float, sigma2: float, w2: float, gamma2: float, azimuth=np.pi, shadowing: str = "wagner",
                               **_ignored):
    """The volume term of [MJ89], (38) with (39) corrected: the unaveraged term averaged over the large-scale slope
    (18) by (27)-(29) and multiplied by Wagner's S, "in analogy with" the roughness term. Backscatter only.
    This is the printed recipe for the slope and shadow corrections of the volume term (G-16; J&R Sect. 14.1.2
    describes it without a formula)."""
    theta = _mj89_check_geometry(grazing_in, grazing_out, azimuth, "volume_model")
    _mj89_warn_range(frequency, rho, nu)
    s = mourad_jackson_1989_slope(frequency, sound_speed, w2=w2, gamma2=gamma2)
    return _mj89_shadow(theta, s, shadowing) * _mj89_average(
        lambda x: mourad_jackson_1989_volume_unaveraged(x, rho=rho, nu=nu, delta=delta, sigma2=sigma2), theta, s)


def mourad_jackson_1989_sediment(phi: float, water_sound_speed: float | None = None) -> dict:
    """The six parameters of the paper's "surface" set for a mean grain size (1-9 phi; refused outside):
    rho, nu, sigma2, gamma2, w2 from [MJ89] (1)-(2), (4)-(7) (`mourad_jackson_1989_relations`), delta by (3) with
    Hamilton's attenuation as J&R (11.3) prints it (`apl_grain_size_relations`; [MJ89] gives c1 = 1.528 m/ms
    for (3); `water_sound_speed` None is `GRAIN_SIZE_WATER_SOUND_SPEED`, 1528 m/s). Reproduces the printed inset
    values of the paper's Figs. 3-9 (tests)."""
    water_sound_speed = GRAIN_SIZE_WATER_SOUND_SPEED if water_sound_speed is None else water_sound_speed
    rel = mourad_jackson_1989_relations(phi)
    k = apl_grain_size_relations(phi)["k_db_per_m_khz"]
    delta = k * rel["nu"] * float(water_sound_speed) * math.log(10.0) / (40000.0 * math.pi)
    return {"rho": rel["rho"], "nu": rel["nu"], "delta": delta, "w2": w2_from_cm4(rel["w2_cm4"], rel["gamma2"]),
            "gamma2": rel["gamma2"], "sigma2": rel["sigma2"]}


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


GRAIN_SIZE_WATER_SOUND_SPEED = 1528.0
"""The water sound speed (m/s) that turns the attenuation of (11.3) into the loss parameter of
`GRAIN_SIZE_TABLE`: it reproduces the table's delta (0.01624 at 1.5 phi), as in UnderwaterAcoustics.jl
(docs/references/jackson_richardson_2007/01_sediment_properties.md Sec. 3.1)."""


def apl_grain_size_relations(phi: float, water_sound_speed: float = GRAIN_SIZE_WATER_SOUND_SPEED) -> dict:
    """{rho, nu, delta, k_db_per_m_khz} for a mean grain size Mz in phi units (-1 to 9) by the TR 9407
    relations as printed in Jackson and Richardson, "High-Frequency Seafloor Acoustics" (2007), eqs.
    (11.1)-(11.3), pp. 313-314 (first given by Mourad and Jackson 1989 and expanded in APL-UW TR 9407;
    (11.3) taken from Hamilton 1972). The book calls (11.1)-(11.2) "ad hoc" expressions to improve
    model-data fits, not geoacoustic regressions (p. 314). The constants of the 1 to 9 phi pieces are
    those of the first print, Mourad and Jackson (1989) [MJ89] eqs. (1)-(2) (confirmed,
    docs/references/bottom_mourad_jackson_1989.md Sec. 2.2; "for internal consistency to four decimal
    places", p. 1170), of which J&R print four decimals (3.0455, 1.3425, 1.1565, 1.0019) (C5 of that note;
    rho moves by -2.45e-5 to +2.34e-5, nu by -2.6e-5 to 0, and the pieces meet at 5.3 phi to 4e-6 instead
    of 5.1e-5). The -1 <= Mz < 1 pieces are J&R's ([MJ89] does not print them; its 0.84 phi site follows them):

        rho = 0.007797 Mz^2 - 0.17057 Mz + 2.3139                          -1 <= Mz < 1
            = -0.0165406 Mz^3 + 0.2290201 Mz^2 - 1.1069031 Mz + 3.0455234   1 <= Mz < 5.3
            = -0.0012973 Mz + 1.1564755                                     5.3 <= Mz <= 9     (11.1)
        nu  = 0.002709 Mz^2 - 0.056452 Mz + 1.2778                          -1 <= Mz < 1
            = -0.0014881 Mz^3 + 0.0213937 Mz^2 - 0.1382798 Mz + 1.3424740   1 <= Mz < 5.3
            = -0.0024324 Mz + 1.0018916                                     5.3 <= Mz <= 9     (11.2)
        k   = 0.4556                                                        -1 <= Mz < 0
            = 0.0245 Mz + 0.4556                                            0 <= Mz < 2.6
            = 0.1245 Mz + 0.1978                                            2.6 <= Mz < 4.5
            = 0.20098 Mz^2 - 2.5228 Mz + 8.0399                             4.5 <= Mz < 6.0
            = 0.0117 Mz^2 - 0.2041 Mz + 0.9431                              6.0 <= Mz < 9.5
            = 0.0601                                                        9.5 <= Mz          (11.3)

    k in dB m^-1 kHz^-1; delta = k nu c_w ln 10 / (40 000 pi), the book's (2.6) with alpha = k f_kHz,
    c_w = `water_sound_speed` (1528 m/s, the table's water).

    Two printed constants are NOT used, as misprints: the book prints 1.0091 in the last piece of (11.2),
    which jumps by 0.0072 at 5.3 phi and misses the tabulated rows from 5.5 to 9 phi by 0.0072; 1.0019 is
    continuous with the cubic (0.98901) and gives the rows, and [MJ89] (2) prints 1.0018916 (the misprint is
    confirmed in the original). It prints 0.4566 in the first piece of (11.3); the next piece starts at
    0.4556 and the rows at -1 and -0.5 phi follow 0.4556 (still the notes' argument: [MJ89] prints no k(Mz)). (11.3) has steps as printed (+0.0022 at 2.6, -0.0009 at 4.5,
    +0.0013 at 6.0 phi), kept. Tests: tests/python/test_grain_size_relations.py."""
    mz = _check_grain_size(phi)
    if mz < 1.0:
        rho = 0.007797 * mz**2 - 0.17057 * mz + 2.3139
        nu = 0.002709 * mz**2 - 0.056452 * mz + 1.2778
    elif mz < 5.3:
        # The constants to the original's seven digits ([MJ89] (1)-(2); J&R print 3.0455 and 1.3425): C5.
        rho = -0.0165406 * mz**3 + 0.2290201 * mz**2 - 1.1069031 * mz + 3.0455234
        nu = -0.0014881 * mz**3 + 0.0213937 * mz**2 - 0.1382798 * mz + 1.3424740
    else:
        rho = -0.0012973 * mz + 1.1564755                  # J&R print 1.1565
        nu = -0.0024324 * mz + 1.0018916                   # J&R print 1.0091: a misprint (docstring); [MJ89] (2) 1.0018916
    if mz < 0.0:
        k = 0.4556                                         # printed 0.4566: a misprint (docstring)
    elif mz < 2.6:
        k = 0.0245 * mz + 0.4556
    elif mz < 4.5:
        k = 0.1245 * mz + 0.1978
    elif mz < 6.0:
        k = 0.20098 * mz**2 - 2.5228 * mz + 8.0399
    elif mz < 9.5:
        k = 0.0117 * mz**2 - 0.2041 * mz + 0.9431
    else:
        k = 0.0601
    delta = k * nu * float(water_sound_speed) * math.log(10.0) / (40000.0 * math.pi)
    return {"rho": rho, "nu": nu, "delta": delta, "k_db_per_m_khz": k}


GRAIN_SIZE_METHODS = ("table_linear", "relations")


def sediment_from_grain_size(phi: float, method: str = "relations") -> dict:
    """The six parameters for a mean grain size in phi units (-1 to 9), for a site whose grain
    size is known and that is not one of the named classes.

    `method="relations"` (the default since the conformance item A5, docs/references/CONFORMANCE.md
    Sec. 1.1: a printed reference outranks our interpolation): rho, nu and delta from the relations
    themselves, `apl_grain_size_relations` (J&R 2007 eqs. (11.1)-(11.3), printed; with the constants
    1.0019 and 0.4556 for two printed misprints, and since C5 the seven-digit constants of Mourad and
    Jackson 1989), which give the table at its points (rho 5e-4, nu 1e-4, delta 0.5 %).
    `method="table_linear"` (the default before A5; selectable): rho, nu and delta are interpolated
    linearly between the tabulated classes of `GRAIN_SIZE_TABLE` (every half phi). The interpolation is
    ours; the table is piecewise (the density ratio drops from 2.151 to 1.615 between 1 and 2 phi), so
    between two classes the result is a guess good to the difference of its neighbours.
    Between the points the two differ by at most 0.68 dB in the normal-incidence reflection
    coefficient (at 5.3 phi), under 0.1 dB outside 5-6 phi; rho by up to 0.0106 (1.25 phi), delta by up
    to 7.6 % (near 5.8 phi). At Gallaudet's 6.36 phi (his Table VI prints 1.1483, 0.9864, 0.0033) the
    relations give 1.148225, 0.986422, 0.003267 (1.148249, 0.986430 before C5: his rho follows J&R's
    rounded constant, not the original's) and the interpolation 1.14828, 0.986436, 0.003284
    (tests/python/test_gallaudet_boundaries.py). w2, gamma2 and sigma2 are `grain_size_scattering`
    either way."""
    if method not in GRAIN_SIZE_METHODS:
        import difflib
        near = difflib.get_close_matches(str(method), GRAIN_SIZE_METHODS, n=1)
        hint = f"; did you mean {near[0]!r}?" if near else ""
        raise ValueError(f"sediment_from_grain_size: unknown method {method!r}{hint} (one of {GRAIN_SIZE_METHODS})")
    phi = _check_grain_size(phi)
    if method == "relations":
        rel = apl_grain_size_relations(phi)
        out = {key: rel[key] for key in ("rho", "nu", "delta")}
    else:
        table = np.array(GRAIN_SIZE_TABLE)
        out = {key: float(np.interp(phi, table[:, 0], table[:, i])) for i, key in ((1, "rho"), (2, "nu"), (3, "delta"))}
    return dict(out, **grain_size_scattering(phi))


# --- Rock and other bottoms that are not fluid sediments: an empirical law ------------------------

LAMBERT_MAX_MU_DB = -10.0 * math.log10(math.pi)
"""The largest Lambert parameter energy conservation allows, 10 log10(1 / pi) = -4.97 dB: a
surface that scatters all the incident power with the Lambert pattern (Olson, Lyons and Saebo,
arXiv:1601.06078, introduction; the integral of mu sin(ti) sin(ts) over the hemisphere is pi mu sin(ti)).
Also Jackson and Richardson (2007), eq. (2.23), p. 26."""


DEFAULT_LAMBERT_MU_DB = -29.0
"""The Lambert parameter of a reverberation boundary that gives neither `mu_db` nor a model nor a
sediment (D-89; -27 dB before, written from memory as "the Mackenzie value"). The one open print
found: Muggleworth, "Shallow water reverberation measurement and prediction", MS thesis, Naval
Postgraduate School (1994), https://hdl.handle.net/10945/28529, p. 34: "In 1961, K. V. Mackenzie
reported success using Lambert's law with mu_dB = -29 dB to model bottom backscattering in deep
water at 530 and 1030 Hz", for 20 to 60 degrees of grazing (Mackenzie's paper itself was not
opened). It is a low-frequency, deep-water value: at sonar frequencies name a sediment
(`{model: fluid_sediment, sediment: ...}`, `{sediment: rock}`) or give the site's `mu_db`;
measured sands at 45 degrees and 100 kHz and above are near -25 dB, i.e. mu about -22 dB.
`default_lambert_mu_db` warns once when the bare default is used at 10 kHz or above."""

DEFAULT_LAMBERT_WARN_FREQUENCY = 10e3
_warned_default_lambert = False


def default_lambert_mu_db(frequency: float | None = None) -> float:
    """`DEFAULT_LAMBERT_MU_DB`, with one warning per process when `frequency` (Hz) is 10 kHz or
    above: the value is a deep-water one from about 1 kHz."""
    global _warned_default_lambert
    if frequency is not None and float(frequency) >= DEFAULT_LAMBERT_WARN_FREQUENCY and not _warned_default_lambert:
        _warned_default_lambert = True
        import warnings
        warnings.warn(f"reverberation boundary without mu_db at {float(frequency) / 1e3:g} kHz: the default Lambert parameter "
                      f"{DEFAULT_LAMBERT_MU_DB:g} dB is a deep-water value measured at 0.5 and 1 kHz (Mackenzie 1961 as quoted by "
                      "Muggleworth 1994); name a sediment ({model: fluid_sediment, sediment: ...} or {sediment: rock}) or give "
                      "the site's mu_db. This warning is given once.", stacklevel=2)
    return DEFAULT_LAMBERT_MU_DB


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


def named_lambert_mu_db(name: str, path=None) -> float:
    """The Lambert parameter, dB, of a table entry that carries an empirical Lambert law (rock,
    pebbles): what the reverberation item `{boundary_z, sediment: <name>}` uses. A fluid sediment
    has no such number (its strength is not Lambert's law): the error names the model to use."""
    entry = _entry(name, path)
    if "empirical_backscatter" not in entry:
        raise ValueError(f"sediment {name!r} is a fluid sediment and has no Lambert parameter: use "
                         f"{{model: fluid_sediment, sediment: {name}}} or give mu_db")
    return float(empirical_backscatter(name, path)["mu_db"])


def named_backscattering_strength_db(name: str, grazing, frequency: float, sound_speed: float = NOMINAL_SOUND_SPEED, path=None,
                                     **model):
    """Backscattering strength in dB of a named table entry: the fluid-sediment model for a
    sediment (extra keywords such as `roughness_model` are passed on), the entry's empirical
    Lambert law for rock and pebbles (which has no frequency dependence)."""
    if "empirical_backscatter" in _entry(name, path):
        return lambert_backscatter_db(grazing, empirical_backscatter(name, path)["mu_db"])
    if model.get("volume_model") == WILLIAMS_JACKSON_1998 and any(k not in model for k in VOLUME_SPECTRAL_KEYS):
        model = dict(volume_spectral(name, path), **model)          # the row's `volume_spectral:` block; refused without
    return backscattering_strength_db(grazing, frequency, sound_speed, **dict(sediment(name, path), **model))
