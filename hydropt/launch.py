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
    "fan_2d",
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
