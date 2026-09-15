"""Launch-direction generators.

Elevation is measured from horizontal and is **positive downward** to match the
depth-positive-down frame, so a direction is

``d = (cos(elev) cos(azim), cos(elev) sin(azim), sin(elev))``.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

__all__ = [
    "directions_from_angles",
    "spherical_fan",
    "structured_fan",
    "fan_2d",
    "fan_angular_spacing",
    "fan_sigma_d",
    "fibonacci_sphere",
    "fibonacci_cone",
    "receiver_cone_importance",
]


def directions_from_angles(elev_rad: Tensor, azim_rad: Tensor) -> Tensor:
    """Unit directions from elevation/azimuth, broadcast to a common shape."""
    elev_rad, azim_rad = torch.broadcast_tensors(elev_rad, azim_rad)
    ce, se = torch.cos(elev_rad), torch.sin(elev_rad)
    return torch.stack((ce * torch.cos(azim_rad), ce * torch.sin(azim_rad), se), dim=-1)


def spherical_fan(
    n_elev: int,
    n_azim: int,
    elev_range_deg: tuple[float, float] = (-20.0, 20.0),
    azim_range_deg: tuple[float, float] = (0.0, 360.0),
    *,
    endpoint_azim: bool = False,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> Tensor:
    """Elevation x azimuth grid of launch directions, ``[n_elev * n_azim, 3]``.

    ``endpoint_azim=False`` (the default) omits the duplicate ray at 360 deg
    when the azimuth range wraps the full circle.
    """
    dtype = dtype or torch.get_default_dtype()
    e0, e1 = (math.radians(v) for v in elev_range_deg)
    a0, a1 = (math.radians(v) for v in azim_range_deg)
    elev = torch.linspace(e0, e1, n_elev, dtype=dtype, device=device)
    if endpoint_azim or n_azim == 1:
        azim = torch.linspace(a0, a1, n_azim, dtype=dtype, device=device)
    else:
        azim = a0 + (a1 - a0) * torch.arange(n_azim, dtype=dtype, device=device) / n_azim
    return directions_from_angles(elev[:, None], azim[None, :]).reshape(-1, 3)


def structured_fan(
    n_elev: int,
    n_azim: int,
    elev_range_deg: tuple[float, float] = (-20.0, 20.0),
    azim_range_deg: tuple[float, float] = (-20.0, 20.0),
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """A spherical fan that also reports its launch-angle grids.

    Returns ``(directions, elev, azim)`` with ``directions`` of shape
    ``[n_elev * n_azim, 3]`` laid out **elevation-major**, so ray
    ``i * n_azim + j`` was launched at ``elev[i]``, ``azim[j]``.

    Ray-tube spreading needs exactly this: the tube cross-section is built from
    differences between *neighbouring launch angles*, so it needs to know which
    rays are neighbours and how far apart in angle they are.  A Fibonacci fan
    has no such neighbour structure, which is why it cannot be used for it.

    Both endpoints are included on each axis, unlike :func:`spherical_fan`,
    because a tube derivative needs the grid spacing to be uniform and known.
    """
    dtype = dtype or torch.get_default_dtype()
    e0, e1 = (math.radians(v) for v in elev_range_deg)
    a0, a1 = (math.radians(v) for v in azim_range_deg)
    elev = torch.linspace(e0, e1, n_elev, dtype=dtype, device=device)
    azim = torch.linspace(a0, a1, n_azim, dtype=dtype, device=device)
    dirs = directions_from_angles(elev[:, None], azim[None, :]).reshape(-1, 3)
    return dirs, elev, azim


def fan_2d(
    n: int,
    elev_range_deg: tuple[float, float] = (-20.0, 20.0),
    azimuth_deg: float = 0.0,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> Tensor:
    """Single-azimuth fan, ``[n, 3]`` -- the 2-D limit used by the Snell tests."""
    return spherical_fan(
        n, 1, elev_range_deg, (azimuth_deg, azimuth_deg), dtype=dtype, device=device
    )


def fibonacci_sphere(
    n: int,
    *,
    jitter: float = 0.0,
    generator: torch.Generator | None = None,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> Tensor:
    """``n`` near-uniform directions on the full sphere via the golden-angle spiral.

    ``jitter`` (in units of one cell) randomises the spiral so that repeated
    renders decorrelate; pass an explicit ``generator`` to make that
    reproducible -- a stored seed is exactly what a path-replay backward pass
    would need to regenerate the same fan.
    """
    dtype = dtype or torch.get_default_dtype()
    i = torch.arange(n, dtype=dtype, device=device)
    if jitter:
        noise = torch.rand(n, dtype=dtype, device=device, generator=generator) - 0.5
        i = i + jitter * noise
    # cos(polar) uniformly spaced -> equal-area bands.
    cz = 1.0 - 2.0 * (i + 0.5) / n
    cz = cz.clamp(-1.0, 1.0)
    r = torch.sqrt((1.0 - cz * cz).clamp_min(0.0))
    phi = i * (math.pi * (3.0 - math.sqrt(5.0)))
    return torch.stack((r * torch.cos(phi), r * torch.sin(phi), cz), dim=-1)


def _basis_from_axis(axis: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Orthonormal frame whose third vector is ``axis`` (normalised)."""
    w = axis / axis.norm().clamp_min(1e-30)
    ref = torch.tensor([1.0, 0.0, 0.0], dtype=w.dtype, device=w.device)
    if bool((w * ref).sum().abs() > 0.9):
        ref = torch.tensor([0.0, 1.0, 0.0], dtype=w.dtype, device=w.device)
    u = torch.cross(ref, w, dim=-1)
    u = u / u.norm().clamp_min(1e-30)
    v = torch.cross(w, u, dim=-1)
    return u, v, w


def fibonacci_cone(
    n: int,
    axis: Tensor,
    half_angle_deg: float,
    *,
    jitter: float = 0.0,
    generator: torch.Generator | None = None,
) -> Tensor:
    """``n`` near-uniform directions inside a cone of the given half-angle.

    Equal-area within the spherical cap, so the sampling density is uniform per
    unit solid angle -- the weight each ray carries is the same.
    """
    dtype, device = axis.dtype, axis.device
    cos_a = math.cos(math.radians(half_angle_deg))
    i = torch.arange(n, dtype=dtype, device=device)
    if jitter:
        noise = torch.rand(n, dtype=dtype, device=device, generator=generator) - 0.5
        i = i + jitter * noise
    cz = 1.0 - (1.0 - cos_a) * ((i + 0.5) / n)
    cz = cz.clamp(-1.0, 1.0)
    r = torch.sqrt((1.0 - cz * cz).clamp_min(0.0))
    phi = i * (math.pi * (3.0 - math.sqrt(5.0)))
    u, v, w = _basis_from_axis(axis)
    return (
        r[:, None] * torch.cos(phi)[:, None] * u
        + r[:, None] * torch.sin(phi)[:, None] * v
        + cz[:, None] * w
    )


def receiver_cone_importance(
    n: int,
    source: Tensor,
    target: Tensor,
    half_angle_deg: float = 20.0,
    *,
    n_background: int = 0,
    jitter: float = 0.0,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Directions concentrated on the cone from ``source`` towards ``target``.

    Rays that never come near a receiver contribute nothing to the energy-time
    curve and nothing to its gradient, so aiming the fan is the cheapest
    available variance reduction.  The cone must be wide enough to contain the
    *refracted* arrivals, not just the straight-line bearing: in a strong
    channel the launch angle of an eigenray can sit many degrees off the direct
    bearing.  ``n_background`` adds whole-sphere samples so that paths outside
    the cone are not lost entirely.
    """
    axis = (target.reshape(3) - source.reshape(3)).to(source.dtype)
    dirs = fibonacci_cone(n, axis, half_angle_deg, jitter=jitter, generator=generator)
    if n_background > 0:
        bg = fibonacci_sphere(
            n_background, jitter=jitter, generator=generator,
            dtype=dirs.dtype, device=dirs.device,
        )
        dirs = torch.cat((dirs, bg), dim=0)
    return dirs


def fan_angular_spacing(directions: Tensor, *, chunk: int = 512) -> Tensor:
    """Angle from each ray to its nearest neighbour in the fan, ``[R]`` (rad).

    This is the resolution the fan actually has.  It is measured from the
    directions themselves rather than assumed from the generator's arguments, so
    it is right for a fan that was aimed, weighted, concatenated or hand-built,
    and it is per-ray, so it follows a fan whose density varies across it.

    Why it matters: a splat that accepts arrivals within ``sigma_d`` of a point
    (:func:`hydropt.beamform.extract_arrivals`) weights each ray by
    ``exp(-0.5 (d / sigma_d)^2)`` on its miss distance ``d``.  If the fan's ray
    spacing at the range of interest is much wider than ``sigma_d``, no ray
    passes within ``sigma_d`` except by luck, and the amplitude measures that
    luck instead of the field.  Pairing this with
    :func:`fan_sigma_d` sizes the splat to the fan.

    Computed in chunks, so a fan of tens of thousands of rays does not
    materialise an ``R x R`` matrix.  Detached: this is a property of the
    sampling, not of the scene, and should not carry gradient.
    """
    d = directions.detach().reshape(-1, 3)
    d = d / d.norm(dim=-1, keepdim=True).clamp_min(1e-30)
    n = d.shape[0]
    if n < 2:
        return torch.full((n,), math.pi, dtype=d.dtype, device=d.device)
    out = torch.empty(n, dtype=d.dtype, device=d.device)
    for i in range(0, n, chunk):
        block = d[i:i + chunk]
        cos = (block @ d.T).clamp(-1.0, 1.0)
        # Exclude each ray's own entry, which is cos = 1.
        rows = torch.arange(block.shape[0], device=d.device)
        cos[rows, rows + i] = -1.0
        out[i:i + chunk] = cos.max(dim=1).values.clamp(-1.0, 1.0).arccos()
    return out


def fan_sigma_d(directions: Tensor, arclen: Tensor, *, factor: float = 1.0,
                floor: float = 0.0, spacing: Tensor | None = None) -> Tensor:
    """Splat width matched to a fan's own ray spacing, ``[R, S+1]`` (m).

    ``sigma_d = factor * spacing * arclen``: the fan's angular resolution
    (:func:`fan_angular_spacing`) carried out to each vertex's range, which is
    the transverse distance between neighbouring rays there.  Pass the result as
    ``sigma_d`` to :func:`hydropt.beamform.extract_arrivals` or
    :func:`hydropt.receiver.splat_etc`, both of which accept a per-ray,
    per-vertex width.

    This makes the extraction *sampling-invariant*: densify the fan and the
    splat narrows in step, so the answer converges instead of drifting.  A fixed
    scalar ``sigma_d`` has no such property -- it is only correct for one fan
    density at one range.

    Args:
        directions: the fan that produced ``arclen``, ``[R, 3]``.
        arclen: path length at each vertex, ``[R, S+1]`` -- ``TraceResult.arclen``.
        factor: multiplier on the spacing.  1.0 puts neighbouring rays at
            ``1 sigma``; larger overlaps them more, which smooths the result at
            the cost of resolution.
        floor: minimum width (m), for the vertices near the source where the
            ray spacing collapses to nothing.
        spacing: a precomputed :func:`fan_angular_spacing`, ``[R]``.  The search
            is quadratic in the fan size, so pass it when calling this repeatedly
            for the same fan.

    For a physically derived width rather than a geometric one, use
    :func:`hydropt.beams.beam_sum_kwargs`, which sizes each beam from the
    frequency and the ray-tube dynamics.  This function is the cheap
    alternative: it costs one nearest-neighbour search, not six traces.
    """
    if spacing is None:
        spacing = fan_angular_spacing(directions)
    spacing = spacing.detach().reshape(-1, 1)
    s = arclen.detach().to(dtype=spacing.dtype, device=spacing.device)
    if s.shape[0] != spacing.shape[0]:
        raise ValueError(f"arclen has {s.shape[0]} rays, directions has "
                         f"{spacing.shape[0]}")
    return (factor * spacing * s).clamp_min(floor)
