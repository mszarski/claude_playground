r"""Ray-tube (geometric Jacobian) spreading, and the caustic index that comes with it.

``1/s^2`` spreading is exact only in a homogeneous medium.  Everywhere else the
ocean focuses and defocuses sound, and a refracting channel does it strongly:
the convergence zones of a deep sound channel exist precisely *because* the ray
tube collapses there.  Using ``1/s^2`` gets their timing right and their level
wrong, and -- worse for an inverse problem -- wrong *smoothly*, so a fit will
happily absorb the error into whatever parameter is nearest to hand.

The geometry
------------
Energy is conserved along a ray tube.  A tube launched into solid angle
``dOmega = cos(e) de da`` has, at arclength ``s``, a cross-section
perpendicular to the ray of area ``J de da`` where

.. math::
    J = \left| \left( \frac{\partial \mathbf{r}}{\partial e} \times
                      \frac{\partial \mathbf{r}}{\partial a} \right)
        \cdot \hat{\mathbf{t}} \right|

so the intensity is ``I = cos(e) / J`` -- the ``de da`` cancels, leaving a
quantity that depends only on how the fan spreads, not on how finely it was
sampled.  In a homogeneous medium ``J = s^2 cos(e)`` exactly and this collapses
back to ``1/s^2``, which is the first thing the tests check.

How the derivatives are taken
-----------------------------
Two ways, and both work.

:func:`ray_tube` takes central differences between *neighbouring rays of a
structured fan*, the approach production ray codes use.  It is
``O(dtheta^2)`` accurate -- measured, it converges at exactly that rate -- costs
nothing beyond the trace you already ran, and stays reverse-mode differentiable.

:func:`ray_tube_jvp` instead pushes forward-mode derivatives of ray position
with respect to launch angle straight through the integrator, which is what the
brief originally proposed.  It turns out ``torch.func.jvp`` handles the whole
tracer, reflections included, agreeing with a central difference to 1e-9; see
``scripts/check_jvp.py``.  It is exact rather than second-order, and needs no
neighbour structure at all, so it works on a Fibonacci fan.  It costs three
traces instead of one.

Use the fan version by default and the jvp version to check it, or when the fan
is unstructured.

Neighbouring rays are only comparable while they share an interaction history.
Once one ray of a pair has bounced and the other has not, the difference between
them is a step across a reflection, not a derivative, and the tube is nonsense.
Those vertices are masked out.

Caustics
--------
The *signed* Jacobian changes sign wherever the tube turns inside out -- that is
a caustic, and counting the sign changes gives the KMAH index.  Each one
contributes ``-pi/2`` of phase, which :mod:`hydropt.beamform` can now apply; an
energy-only renderer has no use for it.  The intensity itself is unbounded at a
caustic, so ``min_jacobian`` floors the tube area.  That is the crude fix; the
principled one is Gaussian beams, which replace the sharp tube with a smooth
beam of finite width and never divide by zero at all.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import torch
from torch import Tensor

from .tracer import TraceResult

__all__ = ["RayTube", "ray_tube", "ray_tube_jvp", "spherical_spreading"]


class RayTube(NamedTuple):
    """Ray-tube geometry along a traced bundle.  Arrays are ``[rays, vertices]``."""

    jacobian: Tensor  # signed tube Jacobian J (m^2 per steradian-ish)
    spreading: Tensor  # intensity factor cos(e)/|J|, drop-in for 1/s^2
    caustics: Tensor  # KMAH index: caustics passed so far, as a float count
    valid: Tensor  # True where neighbouring rays still share a bounce history

    @property
    def kmah_phase(self) -> Tensor:
        """Phase advance ``-pi/2`` per caustic, in radians."""
        return -0.5 * math.pi * self.caustics


def spherical_spreading(result: TraceResult, *, min_range: float = 1.0) -> Tensor:
    """The ``1/s^2`` factor, shaped like :attr:`RayTube.spreading`.

    Provided so the two spreading laws are interchangeable at a call site.
    """
    s = result.arclen.clamp_min(min_range)
    return 1.0 / (s * s)


def _central(values: Tensor, spacing: Tensor, dim: int) -> Tensor:
    """Central difference along ``dim``, one-sided at the two edges."""
    n = values.shape[dim]
    if n < 2:
        raise ValueError("need at least two rays along each fan axis for a ray tube")
    fwd = values.narrow(dim, 1, n - 1) - values.narrow(dim, 0, n - 1)
    step = spacing
    # Interior: average of the two one-sided differences = central difference.
    interior = 0.5 * (fwd.narrow(dim, 1, n - 2) + fwd.narrow(dim, 0, n - 2)) if n > 2 else None
    first = fwd.narrow(dim, 0, 1)
    last = fwd.narrow(dim, n - 2, 1)
    parts = [first] + ([interior] if interior is not None else []) + [last]
    return torch.cat(parts, dim=dim) / step


def ray_tube(
    result: TraceResult,
    elev: Tensor,
    azim: Tensor,
    *,
    min_jacobian: float = 1e-3,
    min_range: float = 1.0,
) -> RayTube:
    """Ray-tube spreading for a bundle traced from :func:`hydropt.launch.structured_fan`.

    Args:
        result: the traced bundle.  Its rays must be the structured fan's, in
            elevation-major order.
        elev, azim: the fan's launch-angle grids, ``[n_elev]`` and ``[n_azim]``.
        min_jacobian: floor on ``|J|`` as a fraction of the local ``s^2``, so a
            caustic gives a large but finite intensity rather than infinity.
        min_range: floor on path length, keeping the near field finite.

    Returns:
        :class:`RayTube`.  ``spreading`` is a drop-in replacement for the
        ``1/s^2`` term in :func:`hydropt.receiver.splat_etc` and
        :func:`hydropt.beamform.extract_arrivals`, and is differentiable in
        every scene parameter, because it is built from the traced path.
    """
    n_elev, n_azim = int(elev.shape[0]), int(azim.shape[0])
    n_rays, n_vert = int(result.pos.shape[0]), int(result.pos.shape[1])
    if n_elev * n_azim != n_rays:
        raise ValueError(
            f"fan is {n_elev} x {n_azim} = {n_elev * n_azim} rays but the trace has "
            f"{n_rays}; ray_tube needs the structured fan it was traced from"
        )
    dtype, device = result.pos.dtype, result.pos.device
    elev = elev.to(dtype=dtype, device=device)
    azim = azim.to(dtype=dtype, device=device)

    pos = result.pos.reshape(n_elev, n_azim, n_vert, 3)

    # Tangent: the outgoing direction at each vertex.  A central difference in
    # arclength would straddle reflections and point nowhere in particular.
    fwd = pos.narrow(2, 1, n_vert - 1) - pos.narrow(2, 0, n_vert - 1)
    tangent = torch.cat((fwd, fwd.narrow(2, n_vert - 2, 1)), dim=2)
    tangent = tangent / tangent.norm(dim=-1, keepdim=True).clamp_min(1e-30)

    d_elev = _central(pos, _spacing(elev), dim=0)
    d_azim = _central(pos, _spacing(azim), dim=1)

    signed = (torch.cross(d_elev, d_azim, dim=-1) * tangent).sum(-1)  # [E, A, V]

    # A tube only means anything while its rays share an interaction history.
    bounces = (result.bounce_grazing.abs() > 0).to(dtype).cumsum(dim=1)
    bounces = bounces.reshape(n_elev, n_azim, n_vert)
    same_e = _neighbours_agree(bounces, dim=0)
    same_a = _neighbours_agree(bounces, dim=1)
    alive = result.alive.reshape(n_elev, n_azim, n_vert) > 0
    valid = same_e & same_a & alive

    # Caustics: sign flips of the signed Jacobian, counted only where valid.
    sign = torch.sign(signed.detach())
    flip = (sign[..., 1:] * sign[..., :-1] < 0) & valid[..., 1:] & valid[..., :-1]
    caustics = torch.cat((torch.zeros_like(flip[..., :1]), flip), dim=-1)
    caustics = caustics.to(dtype).cumsum(dim=-1)

    s = result.arclen.reshape(n_elev, n_azim, n_vert).clamp_min(min_range)
    cos_e = torch.cos(elev).reshape(n_elev, 1, 1)
    floor = min_jacobian * s * s * cos_e.clamp_min(1e-6)
    magnitude = torch.maximum(signed.abs(), floor)
    spreading = cos_e / magnitude

    # Where the tube is meaningless, fall back to spherical spreading rather
    # than emitting a number built from a difference across a reflection.
    spreading = torch.where(valid, spreading, 1.0 / (s * s))

    flat = (n_rays, n_vert)
    return RayTube(
        jacobian=signed.reshape(*flat),
        spreading=spreading.reshape(*flat),
        caustics=caustics.reshape(*flat),
        valid=valid.reshape(*flat),
    )


def _spacing(grid: Tensor) -> Tensor:
    """Uniform spacing of a 1-D grid, as a scalar tensor."""
    if grid.shape[0] < 2:
        raise ValueError("a fan axis needs at least two angles for a ray tube")
    return grid[1] - grid[0]


def _neighbours_agree(values: Tensor, dim: int) -> Tensor:
    """True where a cell matches both of its neighbours along ``dim``."""
    n = values.shape[dim]
    same = values.narrow(dim, 1, n - 1) == values.narrow(dim, 0, n - 1)
    lo = torch.cat((same.narrow(dim, 0, 1), same), dim=dim)
    hi = torch.cat((same, same.narrow(dim, n - 2, 1)), dim=dim)
    return lo & hi


def ray_tube_jvp(
    scene,
    elev: Tensor,
    azim: Tensor,
    *,
    min_jacobian: float = 1e-3,
    min_range: float = 1.0,
    trace_kwargs: dict | None = None,
) -> RayTube:
    """Ray tube from forward-mode derivatives, exact and fan-agnostic.

    Pushes ``d(position)/d(elevation)`` and ``d(position)/d(azimuth)`` through
    the integrator with :func:`torch.func.jvp` rather than differencing
    neighbouring rays.  That removes the ``O(dtheta^2)`` truncation of
    :func:`ray_tube` and, more usefully, removes the requirement that the fan be
    a structured grid: ``elev`` and ``azim`` are per-ray, so a Fibonacci fan or
    any other sampling works.

    It also needs no "same bounce history" mask.  Neighbour differences break
    down when one ray of a pair has reflected and the other has not, because the
    difference is then a step across a reflection rather than a derivative;
    forward-mode differentiates the ray's own piecewise-smooth trajectory and
    has no such problem.

    The cost is three traces: one for the path itself and one per tangent, since
    ``jvp`` carries a single tangent per pass.

    Args:
        scene: the scene to trace through.
        elev, azim: per-ray launch angles in radians, both ``[R]``.
        min_jacobian, min_range: as :func:`ray_tube`.
        trace_kwargs: forwarded to :func:`hydropt.tracer.trace`.

    Returns:
        :class:`RayTube`.  ``valid`` reflects ray liveness only.
    """
    from .launch import directions_from_angles
    from .tracer import trace as _trace

    kw = trace_kwargs or {}
    elev = elev.reshape(-1)
    azim = azim.reshape(-1)

    def positions(e: Tensor, a: Tensor) -> Tensor:
        return _trace(scene, directions_from_angles(e, a), **kw).pos

    _, d_elev = torch.func.jvp(lambda e: positions(e, azim), (elev,),
                               (torch.ones_like(elev),))
    _, d_azim = torch.func.jvp(lambda a: positions(elev, a), (azim,),
                               (torch.ones_like(azim),))

    result = _trace(scene, directions_from_angles(elev, azim), **kw)
    fwd = result.pos[:, 1:] - result.pos[:, :-1]
    tangent = torch.cat((fwd, fwd[:, -1:]), dim=1)
    tangent = tangent / tangent.norm(dim=-1, keepdim=True).clamp_min(1e-30)

    signed = (torch.cross(d_elev, d_azim, dim=-1) * tangent).sum(-1)
    valid = result.alive > 0

    sign = torch.sign(signed.detach())
    flip = (sign[:, 1:] * sign[:, :-1] < 0) & valid[:, 1:] & valid[:, :-1]
    caustics = torch.cat((torch.zeros_like(flip[:, :1]), flip), dim=1)
    caustics = caustics.to(signed.dtype).cumsum(dim=1)

    s = result.arclen.clamp_min(min_range)
    cos_e = torch.cos(elev).reshape(-1, 1)
    floor = min_jacobian * s * s * cos_e.abs().clamp_min(1e-6)
    spreading = cos_e / torch.maximum(signed.abs(), floor)
    spreading = torch.where(valid, spreading, 1.0 / (s * s))

    return RayTube(jacobian=signed, spreading=spreading, caustics=caustics, valid=valid)
