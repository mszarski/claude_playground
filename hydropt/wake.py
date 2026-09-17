"""The surface wake of a manoeuvring vessel, as a height field.

A wake is not a new kind of scatterer: it is the sea surface with waves on it,
and the surface is already a :class:`~hydropt.boundaries.HeightField` that the
tracer intersects and reflects off.  So a wake enters this package as another
term in the surface height, alongside the wind sea from
:func:`~hydropt.environment.pierson_moskowitz_surface`, and everything
downstream -- specular reflection off the local normal, surface reverberation,
coherence loss -- follows without changing.

**Why packets rather than the textbook integral.**  The classical Kelvin wake
comes from a stationary-phase integral that assumes a straight track at
constant speed, and a vessel that turns has neither.  The construction here is
the one that generalises: every point on the track emits wave groups, and the
Kelvin pattern is their envelope rather than a primitive.  For a source moving
at speed ``V``, stationary phase keeps the waves whose phase speed matches the
source's component in their direction,

    c_p = V cos(theta)   ->   k = g / (V cos(theta))^2

and each group then travels in a straight line at the group speed
``c_g = V cos(theta) / 2``.  Summing those packets over emission time and
direction gives the wake.  On a straight track it reproduces Kelvin exactly --
the envelope closes at ``asin(1/3) = 19.47`` degrees and the transverse waves
come out at ``2 pi V^2 / g``, both of which are tests below.  On a curved track
each packet simply leaves from where the vessel actually was, pointing where it
was actually heading, so the arms bunch on the inside of a turn and stretch on
the outside, which is what a turning wake looks like.

**Deep water only.**  The dispersion above is the deep-water one.  It holds
while the Froude number ``V / sqrt(g h)`` stays below about 0.7; past that the
wake widens from the Kelvin angle and this model is simply wrong, so it refuses
rather than extrapolating.  In 30 m of water that limit is about 23 knots.

**What is parameterised and what is measured.**  The geometry -- angle,
wavelength, how the arms curve -- is fixed by gravity and the track, and is not
adjustable.  The amplitude and its directional spread depend on the hull, and
those are the parameters to fit against a real vessel's data rather than assume.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from .boundaries import BilinearHeightField

__all__ = ["wake_packets", "wake_elevation", "kelvin_wake_surface",
           "froude_number"]

G = 9.80665


def froude_number(speed: float, water_depth: float) -> float:
    """``V / sqrt(g h)`` -- the number that decides whether deep water applies."""
    if water_depth <= 0.0:
        raise ValueError(f"water depth must be positive, got {water_depth}")
    return speed / math.sqrt(G * water_depth)


def wake_packets(track: Tensor, times: Tensor, *, n_directions: int = 64,
                 max_angle_deg: float = 80.0, decay_time: float = 60.0,
                 directional_exponent: float = 2.0,
                 observation_time: float | None = None):
    """Wave groups shed by a vessel along ``track``, at ``observation_time``.

    Args:
        track: ``[T, 2]`` vessel positions in metres.  Any shape of course --
            the model never assumes it is straight.
        times: ``[T]`` the times those positions were occupied, seconds.
        n_directions: wave directions per emission point.  The pattern is a
            superposition, so too few shows as individual groups rather than a
            wake.
        max_angle_deg: directions are kept to within this of the heading.
            ``k`` goes as ``1/cos^2``, so waves near 90 degrees are
            arbitrarily short and carry almost no energy -- including them
            costs resolution everywhere else.
        decay_time: e-folding time of a group's amplitude, seconds.  Stands in
            for breaking, viscosity and lateral spreading, and sets how far
            astern the wake is still visible.
        directional_exponent: the hull's directional spread, as ``cos^n``.
            This and the amplitude are the hull-dependent parameters; the
            geometry is not.
        observation_time: when the wake is being looked at.  Defaults to the
            last track time, i.e. the wake behind the vessel's present
            position.

    Returns ``(centre [P, 2], wavevector [P, 2], amplitude [P], width [P])``
    for ``P = T * n_directions`` groups.  Differentiable in the track.
    """
    track = track.reshape(-1, 2)
    times = times.reshape(-1)
    if track.shape[0] != times.shape[0]:
        raise ValueError(f"track has {track.shape[0]} points but times has "
                         f"{times.shape[0]}")
    if track.shape[0] < 2:
        raise ValueError("a track needs at least two points to have a heading")
    if decay_time <= 0.0:
        raise ValueError(f"decay_time must be positive, got {decay_time}")

    dtype = track.dtype
    t_obs = float(times[-1]) if observation_time is None else observation_time

    # Heading and speed from the track itself, by central differences.
    step = track[1:] - track[:-1]
    dt = (times[1:] - times[:-1]).clamp_min(1e-9)
    speed = step.norm(dim=-1) / dt                       # [T-1]
    heading = torch.atan2(step[:, 1], step[:, 0])        # [T-1]
    emit = 0.5 * (track[1:] + track[:-1])                # [T-1, 2]
    emit_t = 0.5 * (times[1:] + times[:-1])

    live = (t_obs - emit_t) >= 0.0
    emit, emit_t = emit[live], emit_t[live]
    speed, heading = speed[live], heading[live]
    if emit.shape[0] == 0:
        z = torch.zeros(0, dtype=dtype, device=track.device)
        return z.reshape(0, 2), z.reshape(0, 2), z, z

    lim = math.radians(max_angle_deg)
    theta = torch.linspace(-lim, lim, n_directions, dtype=dtype,
                           device=track.device)
    ct = theta.cos().clamp_min(1e-3)

    # Stationary phase for a moving source: c_p = V cos(theta).
    k = G / (speed.reshape(-1, 1) * ct.reshape(1, -1)) ** 2       # [E, D]
    c_g = 0.5 * speed.reshape(-1, 1) * ct.reshape(1, -1)
    age = (t_obs - emit_t).reshape(-1, 1).clamp_min(0.0)

    direction = heading.reshape(-1, 1) + theta.reshape(1, -1)
    n_hat = torch.stack([direction.cos(), direction.sin()], dim=-1)  # [E, D, 2]
    centre = emit.unsqueeze(1) + n_hat * (c_g * age).unsqueeze(-1)
    wavevector = n_hat * k.unsqueeze(-1)

    # Amplitude: the hull's directional spread, the decay, and the geometric
    # spreading of a group whose front lengthens as it travels -- energy is
    # conserved as A^2 * width, and width grows with distance, so A falls as
    # 1/sqrt of it.
    #
    # Clamped at a WAVELENGTH, not at a metre.  A group has not separated from
    # the hull until it has travelled about one, and clamping closer lets the
    # freshest groups keep an amplitude the far field cannot approach: at 6 m/s
    # a 1 m floor makes the newest group 20 times the one 400 m astern, and the
    # wake renders as a blob at the vessel with nothing behind it.
    travelled = (c_g * age).clamp_min(2.0 * math.pi / k)
    amplitude = (ct.reshape(1, -1) ** directional_exponent
                 * torch.exp(-age / decay_time)
                 / travelled.sqrt())

    # Each group stands for a cell of (emission time, direction), so it is
    # spread over the patch that cell maps to -- the same reasoning as the
    # fan-density splat widths in `hydropt.launch`, and for the same reason:
    # without it the answer counts groups instead of measuring the surface.
    # ...but never narrower than about a third of a wavelength, or the
    # envelope would cut into the wave it is carrying.
    d_theta = float(theta[1] - theta[0]) if n_directions > 1 else lim
    width = (travelled * d_theta).clamp_min(0.35 * 2.0 * math.pi / k)
    return (centre.reshape(-1, 2), wavevector.reshape(-1, 2),
            amplitude.reshape(-1), width.reshape(-1))


def wake_elevation(xy: Tensor, centre: Tensor, wavevector: Tensor,
                   amplitude: Tensor, width: Tensor, *, tile: int = 2048,
                   reach: float = 3.0) -> Tensor:
    """Surface elevation at ``xy`` from a set of wave groups, ``[N]`` metres.

    Each group is a plane wave under a Gaussian envelope: it contributes
    ``A exp(-|dx|^2 / 2 w^2) cos(k . dx)`` at offset ``dx`` from its centre.
    Summed, the groups interfere into the wake pattern -- the cusp lines are
    where they pile up, not something drawn in.

    Evaluated in tiles of points, with the groups culled against each tile.
    The pair count is otherwise the product of two large numbers -- 23,000
    groups over a 280 x 240 m grid at 1 m is 1.6e9 pairs and 4.4 GB in one
    tensor -- and almost all of those pairs are a group evaluated far outside
    its own envelope, contributing nothing.  ``reach`` is how many envelope
    widths are kept; beyond 3 the Gaussian is below 1 percent.
    """
    xy = xy.reshape(-1, 2)
    out = torch.zeros(xy.shape[0], dtype=xy.dtype, device=xy.device)
    if centre.shape[0] == 0 or xy.shape[0] == 0:
        return out
    span = reach * width
    for lo in range(0, xy.shape[0], tile):
        pts = xy[lo:lo + tile]
        mid = 0.5 * (pts.amin(0) + pts.amax(0))
        radius = (pts - mid).norm(dim=-1).max()
        near = (centre - mid).norm(dim=-1) <= radius + span
        if not bool(near.any()):
            continue
        c, k = centre[near], wavevector[near]
        a, w = amplitude[near], width[near]
        d = pts.unsqueeze(1) - c.unsqueeze(0)                 # [tile, C, 2]
        phase = (d * k.unsqueeze(0)).sum(-1)
        env = torch.exp(-0.5 * (d * d).sum(-1) / (w * w).unsqueeze(0))
        out[lo:lo + tile] = (a.unsqueeze(0) * env * torch.cos(phase)).sum(1)
    return out


def kelvin_wake_surface(track: Tensor, times: Tensor, *, amplitude: float = 0.3,
                        extent=((-200.0, 200.0), (-200.0, 200.0)),
                        spacing: float = 1.0, water_depth: float | None = None,
                        base: Tensor | None = None, learnable: bool = False,
                        **packet_kwargs) -> BilinearHeightField:
    """The wake as a height field, ready to be a scene's surface.

    Args:
        track, times: the vessel's course, as for :func:`wake_packets`.
        amplitude: peak elevation scale, metres.  Hull-dependent -- fit it.
        extent: ``((x0, x1), (y0, y1))`` the field covers, metres.
        spacing: node spacing.  Must resolve the waves: the transverse
            wavelength is ``2 pi V^2 / g``, which is 16 m at 10 knots but only
            6 m at 6 knots, and a grid coarser than a quarter of it turns the
            wake into aliasing.
        water_depth: if given, the Froude number is checked and deep water
            refused past 0.7 rather than silently extrapolated.
        base: ``[ny, nx]`` heights to add the wake to -- a wind sea, typically.
        learnable: register the resulting heights as a parameter.

    Depth is positive down, so the wake is subtracted: a crest is a smaller
    depth.
    """
    track = track.reshape(-1, 2)
    if water_depth is not None:
        step = track[1:] - track[:-1]
        dt = (times.reshape(-1)[1:] - times.reshape(-1)[:-1]).clamp_min(1e-9)
        fastest = float((step.norm(dim=-1) / dt).max())
        fr = froude_number(fastest, water_depth)
        if fr > 0.7:
            raise ValueError(
                f"Froude number {fr:.2f} in {water_depth:.0f} m of water "
                f"({fastest:.1f} m/s): past about 0.7 the wake widens from the "
                f"Kelvin angle and this deep-water model does not apply")

    (x0, x1), (y0, y1) = extent
    nx = max(int(round((x1 - x0) / spacing)) + 1, 2)
    ny = max(int(round((y1 - y0) / spacing)) + 1, 2)
    xs = torch.linspace(x0, x1, nx, dtype=track.dtype, device=track.device)
    ys = torch.linspace(y0, y1, ny, dtype=track.dtype, device=track.device)
    gx, gy = torch.meshgrid(xs, ys, indexing="xy")        # [ny, nx]
    xy = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)

    packets = wake_packets(track, times, **packet_kwargs)
    eta = wake_elevation(xy, *packets).reshape(ny, nx)
    peak = eta.detach().abs().max().clamp_min(1e-12)
    heights = -(amplitude / peak) * eta
    if base is not None:
        heights = heights + base
    return BilinearHeightField(heights, origin=(float(x0), float(y0)),
                               spacing=(float(xs[1] - xs[0]),
                                        float(ys[1] - ys[0])),
                               learnable=learnable)
