r"""Synthesising environments: rough surfaces, bathymetry, and sound-speed fields.

hydropt's fields and height fields take arrays, which is the right interface and
an awkward place to start from.  This module builds those arrays: a sea surface
from a wind speed, a seabed from a roughness exponent, a range-dependent ocean
from an internal-wave displacement.  Everything returns an ordinary hydropt
object -- :class:`hydropt.boundaries.BilinearHeightField`,
:class:`hydropt.fields.GriddedField` -- so nothing downstream needs to know a
generator was involved, and every field stays learnable if you want to fit it.

Spectral synthesis, and how it is normalised
--------------------------------------------
Every random field here is made the same way: shape white noise in the
wavenumber domain by the square root of a power spectrum, transform back, and
scale.  What makes such a field *physical* is the scaling, and there are two
defensible choices:

``normalise="sample"`` (the default) scales the realisation so its **sample**
    RMS is exactly what you asked for.  A surface built for a 1.2 m significant
    wave height has one, which is usually what you want from a simulator, and it
    makes a test exact.  The cost is that the variance-of-the-variance is
    removed: an ensemble of these is slightly less variable than the real thing.
``normalise="ensemble"`` scales so the RMS is right **in expectation**, leaving
    each realisation's own variance to fluctuate as it should.  Use this when the
    spread across realisations is itself the thing being studied.

The spectrum sets the *shape* and the RMS sets the *level*, so the spectral
constants (Phillips' ``alpha``, and so on) cancel out of the result and only the
wavenumber dependence matters.  That is why the Pierson-Moskowitz surface here
takes its level from the significant wave height ``H_s = 0.22 U^2 / g`` rather
than from the spectral prefactor: the two agree, and the former is checkable.

What these are not
------------------
A spectral realisation is a *sample of a random process with the right
second-order statistics*, not a measurement.  It has the right RMS and the right
spectral slope; it does not have the crests, the breaking, the sandwaves or the
outcrops that a real surface or seabed has, and a Gaussian random field has no
skewness where a real wave field does.  And hydropt reflects **specularly** off
whatever surface you hand it, so a rough surface here bends the specular
direction but does not scatter energy out of it -- see the README's limits.

References
----------
Pierson & Moskowitz (1964) for the wave spectrum; Fox & Hayes (1985) for
power-law seabed roughness; Munk (1981) / Flatte et al. (1979) for the
displacement picture of internal-wave sound-speed fluctuations.
"""

from __future__ import annotations

import math
from typing import Callable, Literal, Sequence

import torch
from torch import Tensor

from .boundaries import BilinearHeightField, FlatHeight, HeightField
from .fields import DepthProfile, GriddedField

__all__ = [
    "gaussian_seamount",
    "spectral_field",
    "pierson_moskowitz_surface",
    "fractal_bathymetry",
    "internal_wave_perturbation",
    "wave_number_peak_pm",
    "significant_wave_height_pm",
]

G = 9.80665  # m/s^2
_PM_ALPHA = 8.1e-3
_PM_BETA = 0.74
Normalise = Literal["sample", "ensemble"]


# --------------------------------------------------------------------------- #
# Deterministic shapes
# --------------------------------------------------------------------------- #
def gaussian_seamount(
    shape: tuple[int, int],
    spacing: tuple[float, float],
    *,
    base_depth: float,
    height: float,
    width: float,
    centre: tuple[float, float] | None = None,
    origin: tuple[float, float] = (0.0, 0.0),
    learnable: bool = True,
    dtype: torch.dtype | None = None,
) -> BilinearHeightField:
    """A single Gaussian rise on a flat seabed.

    The shape the bathymetry examples build by hand, as a function.  Depth is
    positive downward, so the seamount *subtracts* from ``base_depth``.

    Args:
        shape: ``(ny, nx)`` node counts.
        spacing: ``(dx, dy)`` node spacing (m).
        base_depth: seabed depth away from the rise (m).
        height: how far the summit rises above the surrounding seabed (m).
        width: Gaussian 1/e half-width (m), i.e. ``exp(-(r/width)^2)``.
        centre: ``(x, y)`` of the summit.  Defaults to the node nearest the
            middle of the grid, *snapped to a node*, so that the summit is
            actually sampled and ``height`` means what it says.  Give a centre
            between nodes and the realised relief is lower by
            ``exp(-(dr/width)^2)`` for the offset ``dr`` -- correct behaviour,
            and a surprise if the summit silently lands between samples.
        origin: ``(x0, y0)`` of node ``[0, 0]``.
    """
    dtype = dtype or torch.get_default_dtype()
    ny, nx = shape
    dx, dy = spacing
    x = origin[0] + torch.arange(nx, dtype=dtype) * dx
    y = origin[1] + torch.arange(ny, dtype=dtype) * dy
    if centre is None:
        # Snap to a node, so the summit is sampled and `height` is the relief.
        centre = (float(x[nx // 2]), float(y[ny // 2]))
    xx, yy = torch.meshgrid(x, y, indexing="xy")
    r2 = (xx - centre[0]) ** 2 + (yy - centre[1]) ** 2
    heights = base_depth - height * torch.exp(-r2 / (width * width))
    return BilinearHeightField(heights, origin=origin, spacing=spacing,
                               learnable=learnable)


# --------------------------------------------------------------------------- #
# Spectral synthesis
# --------------------------------------------------------------------------- #
def spectral_field(
    shape: tuple[int, int],
    spacing: tuple[float, float],
    psd: Callable[[Tensor], Tensor],
    rms: float,
    *,
    normalise: Normalise = "sample",
    generator: torch.Generator | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """A real 2-D random field with a prescribed spectral shape and RMS.

    Args:
        shape: ``(ny, nx)``.
        spacing: ``(dx, dy)`` in metres.
        psd: called with the radial wavenumber ``k`` (rad/m, ``[ny, nx]``,
            ``k = 0`` at the DC node) and returning a non-negative power density
            of the same shape.  Only its *shape* in ``k`` matters; any constant
            factor is removed by the normalisation.  Return zero at ``k = 0`` or
            it will be zeroed anyway -- a random mean offset is not roughness.
        rms: target RMS of the field (m, or whatever the field measures).
        normalise: ``"sample"`` or ``"ensemble"``; see the module docstring.
        generator: RNG, for reproducibility.

    Returns:
        ``[ny, nx]`` real field with zero mean.
    """
    dtype = dtype or torch.get_default_dtype()
    ny, nx = int(shape[0]), int(shape[1])
    dx, dy = float(spacing[0]), float(spacing[1])
    if ny < 2 or nx < 2:
        raise ValueError(f"need at least 2 nodes per axis, got {(ny, nx)}")

    kx = 2.0 * math.pi * torch.fft.fftfreq(nx, d=dx, dtype=dtype)
    ky = 2.0 * math.pi * torch.fft.fftfreq(ny, d=dy, dtype=dtype)
    k = (kx.reshape(1, -1) ** 2 + ky.reshape(-1, 1) ** 2).sqrt()

    weight = psd(k)
    if weight.shape != (ny, nx):
        raise ValueError(f"psd returned {tuple(weight.shape)}, expected {(ny, nx)}")
    weight = torch.nan_to_num(weight, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    weight[0, 0] = 0.0  # zero mean: a DC offset is not roughness
    amp = weight.sqrt()

    noise = torch.randn(ny, nx, generator=generator, dtype=dtype)
    field = torch.fft.ifft2(torch.fft.fft2(noise) * amp).real

    total = float((amp * amp).sum())
    if total <= 0.0:
        raise ValueError("psd is zero everywhere; nothing to synthesise")
    if normalise == "ensemble":
        # Var(field) = (1/N) sum |amp|^2 for unit-variance white noise, so this
        # makes the RMS right in expectation and leaves the realisation's own
        # variance free to fluctuate.
        field = field * (rms / math.sqrt(total / (ny * nx)))
    elif normalise == "sample":
        sd = float(field.std(unbiased=False))
        if sd <= 0.0:
            raise ValueError("synthesised field is constant; check the psd")
        field = field * (rms / sd)
    else:
        raise ValueError(f"unknown normalise {normalise!r}")
    return field - field.mean()


# --------------------------------------------------------------------------- #
# Sea surface
# --------------------------------------------------------------------------- #
def significant_wave_height_pm(wind_speed: float) -> float:
    """Pierson-Moskowitz significant wave height, ``H_s = 0.22 U^2 / g`` (m).

    ``U`` is the wind speed at 19.5 m, the height the original fit used.  A
    fully-developed sea: a shorter fetch or a younger sea gives less.
    """
    return 0.22 * wind_speed * wind_speed / G


def wave_number_peak_pm(wind_speed: float) -> float:
    r"""Peak wavenumber of the PM *wavenumber* spectrum (rad/m).

    Note this is **not** ``omega_p^2 / g``.  Changing variables from frequency to
    wavenumber carries a Jacobian, so the two spectra peak in different places:
    the ``S(omega)`` peak at ``omega_p = 0.877 g / U`` maps to ``0.769 g / U^2``,
    while ``S(k) = (alpha/2) k^-3 exp(-beta g^2 / (U^4 k^2))`` peaks where
    ``k^2 = 2 beta g^2 / (3 U^4)``, i.e. at ``0.702 g / U^2``.  Both are correct
    statements about different functions; this returns the second, because that
    is the spectrum being sampled.
    """
    return math.sqrt(2.0 * _PM_BETA / 3.0) * G / (wind_speed * wind_speed)


def pierson_moskowitz_surface(
    shape: tuple[int, int],
    spacing: tuple[float, float],
    wind_speed: float,
    *,
    origin: tuple[float, float] = (0.0, 0.0),
    rms: float | None = None,
    k_max: float | None = None,
    normalise: Normalise = "sample",
    learnable: bool = False,
    generator: torch.Generator | None = None,
    dtype: torch.dtype | None = None,
) -> BilinearHeightField:
    r"""A fully-developed wind sea as a height field, from the wind speed alone.

    The omnidirectional Pierson-Moskowitz wavenumber spectrum, isotropic in
    direction:

    .. math::
        S(k) = \frac{\alpha}{2} k^{-3}
               \exp\!\left(-\frac{\beta g^2}{U^4 k^2}\right),
        \qquad S_{2D} = \frac{S(k)}{2\pi k}

    with ``alpha = 8.1e-3`` and ``beta = 0.74``.  The level comes from
    ``H_s = 0.22 U^2 / g`` and ``RMS = H_s / 4``, not from ``alpha``, because the
    normalisation removes any constant factor anyway and the wave height is the
    checkable quantity.

    Isotropic, so no wind direction and no directional spreading: a real wind sea
    is elongated across wind, and a swell system is much narrower still.  Pass
    your own ``psd`` to :func:`spectral_field` if the directionality matters.

    Args:
        shape: ``(ny, nx)`` node counts.  The grid must be fine enough to sample
            the waves that matter: the peak wavelength is
            ``2 pi / wave_number_peak_pm(U)``, and nodes coarser than a quarter of
            that alias the spectrum rather than resolving it.
        spacing: ``(dx, dy)`` node spacing (m).
        wind_speed: ``U`` at 19.5 m (m/s).
        rms: override the RMS elevation (m); defaults to ``H_s / 4``.
        k_max: roll the spectrum off above this wavenumber (rad/m).  Defaults to
            the grid's own Nyquist, which is the honest limit -- capillary waves
            a grid cannot represent should not be pretended into it.
        learnable: register the heights as a parameter.  Off by default: a wave
            field is usually a given, not an unknown.

    Returns:
        A :class:`hydropt.boundaries.BilinearHeightField` of surface elevation
        with **depth positive downward**, so a wave crest is a negative height.
        Use it as a scene's ``surface``.
    """
    dtype = dtype or torch.get_default_dtype()
    if wind_speed <= 0.0:
        raise ValueError("wind_speed must be positive")
    a = _PM_BETA * G * G / wind_speed**4
    nyquist = math.pi / max(float(spacing[0]), float(spacing[1]))
    cutoff = nyquist if k_max is None else float(k_max)

    def psd(k: Tensor) -> Tensor:
        kk = k.clamp_min(1e-12)
        s1d = kk.pow(-3.0) * torch.exp(-a / (kk * kk))
        s2d = s1d / (2.0 * math.pi * kk)  # omnidirectional -> per unit area of k
        return torch.where(k <= cutoff, s2d, torch.zeros_like(s2d))

    target = significant_wave_height_pm(wind_speed) / 4.0 if rms is None else float(rms)
    elevation = spectral_field(shape, spacing, psd, target, normalise=normalise,
                               generator=generator, dtype=dtype)
    # Depth is positive downward, so a crest (positive elevation) is negative depth.
    return BilinearHeightField(-elevation, origin=origin, spacing=spacing,
                               learnable=learnable)


# --------------------------------------------------------------------------- #
# Seabed
# --------------------------------------------------------------------------- #
def fractal_bathymetry(
    shape: tuple[int, int],
    spacing: tuple[float, float],
    *,
    base_depth: float,
    rms: float,
    exponent: float = 3.0,
    k_min: float | None = None,
    k_max: float | None = None,
    origin: tuple[float, float] = (0.0, 0.0),
    normalise: Normalise = "sample",
    learnable: bool = True,
    generator: torch.Generator | None = None,
    dtype: torch.dtype | None = None,
) -> BilinearHeightField:
    r"""Power-law ("fractal") seabed roughness on a flat base depth.

    Abyssal seafloor topography follows a power law over a wide band of scales,
    ``S_{2D}(k) \propto k^{-\gamma}`` with ``gamma`` typically 2.5 to 3.5 (Fox &
    Hayes 1985).  ``exponent`` is that ``gamma``: larger means smoother at short
    scales, and the realised slope is checkable, which the tests do.

    Args:
        base_depth: mean seabed depth (m).
        rms: RMS relief about that depth (m).
        exponent: spectral exponent ``gamma`` of the 2-D PSD.
        k_min: flatten the spectrum below this wavenumber (rad/m), i.e. stop the
            power law before it puts unbounded power into the longest scales.
            Defaults to the fundamental of the grid, which is the smallest
            wavenumber the grid can represent at all.
        k_max: roll off above this wavenumber; defaults to the grid's Nyquist.
        learnable: on by default -- an unknown seabed is the usual case, and this
            is a natural starting point for the bathymetry inversion.
    """
    dtype = dtype or torch.get_default_dtype()
    ny, nx = int(shape[0]), int(shape[1])
    dx, dy = float(spacing[0]), float(spacing[1])
    fundamental = 2.0 * math.pi / max(nx * dx, ny * dy)
    lo = fundamental if k_min is None else float(k_min)
    hi = (math.pi / max(dx, dy)) if k_max is None else float(k_max)

    def psd(k: Tensor) -> Tensor:
        kk = k.clamp_min(lo)  # flat below k_min, power law above
        return torch.where(k <= hi, kk.pow(-float(exponent)), torch.zeros_like(kk))

    relief = spectral_field(shape, spacing, psd, float(rms), normalise=normalise,
                            generator=generator, dtype=dtype)
    return BilinearHeightField(base_depth + relief, origin=origin, spacing=spacing,
                               learnable=learnable)


# --------------------------------------------------------------------------- #
# Range-dependent sound speed
# --------------------------------------------------------------------------- #
def internal_wave_perturbation(
    base: DepthProfile,
    shape: tuple[int, int, int],
    spacing: tuple[float, float, float],
    *,
    rms_displacement: float,
    correlation_length: float,
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    mode: int = 1,
    water_depth: float | None = None,
    normalise: Normalise = "sample",
    learnable: bool = True,
    generator: torch.Generator | None = None,
    dtype: torch.dtype | None = None,
) -> GriddedField:
    r"""A range-dependent ocean, from internal-wave vertical displacement.

    Internal waves move water up and down; a parcel carries its sound speed with
    it, so a vertical displacement ``zeta`` shows up as

    .. math:: \delta c(x, y, z) = -\frac{dc}{dz}\,\zeta(x, y, z)

    which is the displacement picture used for ocean sound-speed fluctuations
    (Flatte et al. 1979; Munk 1981).  Building the field this way rather than
    perturbing ``c`` directly means the perturbation is automatically largest
    where the background gradient is steepest -- in the thermocline -- and
    vanishes in an isothermal layer, which is what is observed and is not
    something a field of independent noise would reproduce.

    The displacement is a horizontally Gaussian-correlated random field times a
    baroclinic mode shape ``sin(m pi z / H)``, which vanishes at the surface and
    the seabed as a trapped internal wave must.

    Args:
        base: background depth-only profile; the returned field is a
            *perturbation* on it, which is much better conditioned for inversion.
        shape: ``(nz, ny, nx)`` node counts, matching
            :class:`hydropt.fields.GriddedField`.
        spacing: ``(dx, dy, dz)`` node spacing (m).
        rms_displacement: RMS vertical displacement ``zeta`` (m).  Open-ocean
            internal waves run from a few metres to a few tens of metres.
        correlation_length: horizontal correlation length of the displacement (m).
        mode: vertical mode number; 1 is the first baroclinic mode.
        water_depth: ``H`` for the mode shape; defaults to the grid's depth span.
        learnable: register the perturbation values as parameters.

    Returns:
        A :class:`hydropt.fields.GriddedField` wrapping ``base``, whose values
        are ``delta c`` in m/s.

    The result is a *frozen* field: one realisation, no time evolution and no
    dispersion, so it cannot say anything about how a channel decorrelates
    between pings.
    """
    dtype = dtype or torch.get_default_dtype()
    nz, ny, nx = (int(v) for v in shape)
    dx, dy, dz = (float(v) for v in spacing)
    if nz < 2:
        raise ValueError("need at least 2 depth nodes")
    depth_span = (nz - 1) * dz
    H = depth_span if water_depth is None else float(water_depth)
    if H <= 0.0:
        raise ValueError("water_depth must be positive")

    # Horizontal structure: Gaussian-correlated, so the PSD is Gaussian too.
    ell = float(correlation_length)
    if ell <= 0.0:
        raise ValueError("correlation_length must be positive")

    def psd(k: Tensor) -> Tensor:
        return torch.exp(-(k * ell) ** 2 / 2.0)

    horizontal = spectral_field((ny, nx), (dx, dy), psd, 1.0,
                                normalise=normalise, generator=generator,
                                dtype=dtype)  # unit RMS

    z = origin[2] + torch.arange(nz, dtype=dtype) * dz
    shape_z = torch.sin(mode * math.pi * (z - origin[2]).clamp(0.0, H) / H)
    # dc/dz of the background, by central differences on the profile itself.
    c = base.c_of_z(z)
    dcdz = torch.gradient(c, spacing=(z,))[0]

    # delta c = -(dc/dz) * zeta, zeta = rms * shape(z) * horizontal(x, y)
    zeta_scale = float(rms_displacement)
    values = (-(dcdz * shape_z * zeta_scale).reshape(nz, 1, 1)
              * horizontal.reshape(1, ny, nx))
    return GriddedField(values, origin=origin, spacing=(dx, dy, dz), base=base,
                        learnable=learnable)
