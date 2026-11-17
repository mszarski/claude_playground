"""Scattering strength of a rough fluid sediment: interface roughness plus sediment volume.

This is the kind of bottom model SST points to (`JacksonBottom`, `APLBottom`: a fluid sediment
with a power-law roughness spectrum and volume inhomogeneities, six parameters). The APL-UW
handbook that defines those models (TR 9407) was not available, so nothing here is copied from
it: both terms are derived below from the wave equation, and tests check them against limits
with known answers. The default roughness term is first-order perturbation theory: use it
below about 60 degrees of grazing and for k x rms relief (at scales near the acoustic
wavelength) below about one. Opt-in alternatives (`roughness_model=`, see "Other regimes"
below) add the Kirchhoff approximation near the specular direction, the small-slope
approximation that joins the two, and the composite-roughness (large-scale slope) average at
small grazing angles. No table of named sediments with these six parameters is shipped: none
was found outside the handbook (`configs/sediments.yaml` holds the placeholder and says
what is missing; `sediment()` reads it).

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

Other regimes (all opt-in; `roughness_model="perturbation"` is the default and is unchanged).

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
and zero where a local angle is negative (shadowed facet). `small_scale_kh` = 0.5 is our
choice, not a published value. It describes Bragg scattering only: towards the specular
direction, where Q falls below Kc, it goes to zero and the Kirchhoff term is the right one.

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


def sediment(name: str, path=None) -> dict:
    """The six parameters of a named sediment from `configs/sediments.yaml` (or `path`). An
    entry whose `status` is not "confirmed" raises a warning: its numbers are not from a
    source."""
    import warnings
    from pathlib import Path

    import yaml

    from .. import config
    file = Path(path) if path is not None else config.config_dir() / "sediments.yaml"
    with open(file) as f:
        table = yaml.safe_load(f) or {}
    if name not in table:
        raise ValueError(f"unknown sediment {name!r} in {file}; it has {sorted(table)}")
    entry = table[name]
    missing = [k for k in SEDIMENT_KEYS if k not in entry]
    if missing:
        raise ValueError(f"sediment {name!r} in {file} lacks {missing}")
    if entry.get("status") != "confirmed":
        warnings.warn(f"sediment {name!r} is {entry.get('status', 'unconfirmed')}: its parameters are not from a "
                      "measured or published source", stacklevel=2)
    return {k: float(entry[k]) for k in SEDIMENT_KEYS}


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


def roughness_strength(grazing_in, grazing_out, frequency: float, sound_speed: float, *, rho: float, nu: float,
                       delta: float, w2: float, gamma2: float, azimuth=np.pi, roughness_model: str = "perturbation",
                       small_scale_kh: float = 0.5, **_ignored):
    """Cross section per unit area and solid angle from interface roughness. `azimuth` is the
    angle between the horizontal directions of the incident and scattered waves: pi for
    backscatter, 0 for forward scatter. `roughness_model`: "perturbation" (default, first
    order), "kirchhoff", "small_slope", "composite" or "combined" (module docstring)."""
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
