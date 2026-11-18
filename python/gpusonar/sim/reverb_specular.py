"""The near-specular part of a boundary's scattering strength, and its removal from
reverberation when time spreading already carries it.

TR 0702 Sec. 12.1 and 12.4: with `doTimeSpread` on, "the effects of near-specular forward
boundary scattering are removed from the reverberation model and added to the forward sound
transformation model", "to avoid double-counting". It does not say how the near-specular
part is delimited. This is our partition, derived from the model that makes the tails
(sim/timespread.py; PHYSICS_NOTES Sec. 16): a boundary of rms height h whose facets have
Gaussian slopes of total mean-square slope s2 (s2 / 2 per axis).

Energy on a bounce. Of the energy |R|^2 a flat boundary would reflect, time spreading keeps
exp(-G^2) coherent (G = 2 k h sin g) and puts the rest, |R|^2 (1 - exp(-G^2)), in the tail.

The same energy as a scattering strength. Geometric optics on those facets (the
high-frequency limit of the Kirchhoff integral) gives, for incident grazing angle g_i,
scattered grazing angle g_s and bistatic azimuth a (0 forward, pi back),

    lobe = |R|^2 / 4 * (|q|^2 / q_z^2)^2 * p(-q_x / q_z, -q_y / q_z)
    q = (cos g_s cos a - cos g_i, cos g_s sin a, sin g_i + sin g_s)       (difference of unit vectors)
    p(x, y) = exp(-(x^2 + y^2) / s2) / (pi s2)                            (slope density)

per unit area and solid angle. Its integral over scattered directions is |R|^2 sin g_i,
all the energy that falls on the area: the facets redirect, they do not lose (checked in
tests/python/test_reverb_specular.py by quadrature). The part of it that the tail carries is

    specular_part = (1 - exp(-(k h q_z)^2)) * lobe,

k h q_z being G at the specular direction. `exclude` subtracts it from a strength, clipped at
zero: what a reverberation generator should use for a boundary whose bounce eigenrays are
time spread. If the strength has such a lobe (a Kirchhoff or small-slope model with the
same slopes) the lobe goes and the rest stays; if it has none (Lambert, Chapman-Harris) the
subtraction only clips a small region, and nothing was double counted to begin with.

When there is something to remove. Boundary reverberation here is the sonar's own pulse
scattered once by the boundary. The tail of a bounce eigenray duplicates it only if that
same pulse also reaches the receiver as a one-way signal over the bounce: a separate
transmitter modelled as a `sources:` entry (SST's DirectSignal; the ASIAEX example), with
`time_spread` on. Then set `exclude_specular: true` on the bistatic reverberation item of
that boundary. Echoes of targets over bounce paths have tails too, but those are sound
scattered by the boundary AND the target: no reverberation item holds that, and nothing is
to be removed for them. The key is therefore opt-in and never automatic.

Backscatter is untouched except near normal incidence: at azimuth pi the facet needed has
slope cot g, and the lobe is exp(-cot^2 g / s2) down (1e-7 at 60 degrees for s2 = 0.02). At
normal incidence the lobe IS the backscatter: the nadir return of a sonar looking straight
down. In the simulator nothing else carries that return (there is no eigenray from the
projector to the boundary and straight back unless a source is put at the projector), so
it must stay in the reverberation. A monostatic item (the default generator) therefore
refuses the key with a warning and removes nothing. A `geometry: bistatic` item takes it
wherever the platforms are: with the projector beside the array it removes the nadir
return, which is right only if that return is simulated as a one-way source.

Accuracy of the partition: the removed strength, integrated over the boundary with both
legs' spreading, equals the tail energy to 1 % (2 % at s2 = 0.045) when the grazing angle
is well above the rms slope (tested at 49 and 63 degrees); at 9 degrees grazing, where the
lobe is as wide as the grazing angle, it exceeds the tail by 9 / 17 / 33 / 65 % for
s2 = 0.005 / 0.01 / 0.02 / 0.045 (geometric optics puts more energy down range than the
flat-boundary energy the tail conserves). The clip at zero bounds any over-removal by what
the strength held there.
"""

from __future__ import annotations

import warnings

import numpy as np

from ..acoustics import NOMINAL_SOUND_SPEED


def facet_lobe(grazing_in, grazing_out, azimuth, mean_square_slope: float):
    """Geometric-optics scattering strength of a perfectly reflecting boundary with Gaussian
    facet slopes (module docstring), per unit area and solid angle. Also returns q_z and the
    facet's own grazing angle (half the angle between the two rays)."""
    gi, gs, az = np.broadcast_arrays(np.asarray(grazing_in, dtype=float), np.asarray(grazing_out, dtype=float),
                                     np.asarray(azimuth, dtype=float))
    qx = np.cos(gs) * np.cos(az) - np.cos(gi)
    qy = np.cos(gs) * np.sin(az)
    qz = np.maximum(np.sin(gi) + np.sin(gs), 1e-12)
    q2 = qx * qx + qy * qy + qz * qz
    s2 = float(mean_square_slope)
    lobe = 0.25 * (q2 / (qz * qz)) ** 2 * np.exp(-(qx * qx + qy * qy) / (qz * qz * s2)) / (np.pi * s2)
    return lobe, qz, np.arcsin(np.clip(0.5 * np.sqrt(q2), 0.0, 1.0))


def specular_part(mean_square_slope: float, rms_height: float, frequency: float, sound_speed: float = NOMINAL_SOUND_SPEED,
                  reflection=1.0):
    """The strength (g_in, g_out, azimuth) that time spreading carries as the incoherent tail
    of a bounce: (1 - exp(-(k h q_z)^2)) |R|^2 x `facet_lobe`. `reflection`: |R| as a number or
    a function of the facet's grazing angle."""
    if not mean_square_slope > 0.0:
        raise ValueError("exclude_specular: the boundary needs a mean-square slope (time_spread gives it)")
    k = 2.0 * np.pi * float(frequency) / float(sound_speed)

    def part(grazing_in, grazing_out, azimuth=np.pi):
        lobe, qz, local = facet_lobe(grazing_in, grazing_out, azimuth, mean_square_slope)
        r = np.abs(reflection(local)) if callable(reflection) else abs(reflection)
        return -np.expm1(-(k * float(rms_height) * qz) ** 2) * r**2 * lobe
    return part


def exclude(strength, part):
    """`strength` less `part`, not below zero, as a function of (g_in, g_out, azimuth)."""
    from .reverb_bistatic import takes_azimuth
    full = strength if takes_azimuth(strength) else (lambda g_in, g_out, azimuth=np.pi: strength(g_in, g_out))
    return lambda grazing_in, grazing_out, azimuth=np.pi: np.maximum(full(grazing_in, grazing_out, azimuth) - part(grazing_in, grazing_out, azimuth), 0.0)


def boundary_part(item: dict, provider, frequency: float, sound_speed: float):
    """The specular part for a reverberation item with `exclude_specular`: from the mapping
    given ({mean_square_slope, rms_height, reflection}), or, with `true`, from the scenario's
    time spreading at that boundary and the provider's reflection coefficient. None (with a
    warning) when it does not apply."""
    opt = item.get("exclude_specular")
    if not opt:
        return None
    if str(item.get("geometry", "monostatic")) != "bistatic":
        warnings.warn("reverberation: exclude_specular applies to `geometry: bistatic` items only; a monostatic item "
                      "holds backscatter, where the specular lobe matters only at normal incidence and is then the "
                      "nadir return itself (nothing else carries it). Nothing is removed", stacklevel=3)
        return None
    if isinstance(opt, dict):
        return specular_part(float(opt["mean_square_slope"]), float(opt["rms_height"]), frequency, sound_speed,
                             float(opt.get("reflection", 1.0)))
    inner = getattr(provider, "inner", None)
    if inner is None or not hasattr(provider, "surface"):
        raise ValueError("reverberation: exclude_specular: true needs `time_spread` in the scenario (or give "
                         "{mean_square_slope, rms_height, reflection})")
    z = float(item["boundary_z"])
    at_surface = abs(z - provider.surface_z) < 1e-6
    at_bottom = provider.bottom_z is not None and abs(z - provider.bottom_z) < 1e-6
    par = provider.surface if at_surface else provider.bottom if at_bottom else None
    if par is None or not par.get("mean_square_slope", 0.0) > 0.0:
        raise ValueError(f"reverberation: exclude_specular at z = {z}: time spreading is not on for that boundary")
    reflection = 1.0
    if at_surface:
        reflection = abs(getattr(inner, "surface_coefficient", -1.0))
    elif getattr(inner, "bottom", None):
        from .propagation import rayleigh_reflection
        bottom = dict(inner.bottom)
        reflection = lambda g: np.abs(rayleigh_reflection(g, sound_speed, **bottom))        # noqa: E731
    elif hasattr(inner, "bottom_coefficient"):
        reflection = abs(inner.bottom_coefficient)
    else:
        warnings.warn("reverberation: exclude_specular: this provider does not expose a bottom reflection coefficient; "
                      "|R| = 1 is used for the removed part", stacklevel=3)
    return specular_part(par["mean_square_slope"], par["rms_height"], getattr(provider, "frequency", frequency), sound_speed, reflection)
