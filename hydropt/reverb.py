"""Seabed and surface reverberation.

For an active sonar, reverberation -- not noise -- usually sets the detection
limit, and a forward-looking geometry is the worst case: the bottom return
arrives at grazing angles smeared over a long range spread, right where the
targets are.

How this is computed
--------------------
A patch of seabed is a scatterer, so in principle reverberation is the target
problem of :mod:`hydropt.active` repeated over every insonified patch -- which
would be a render per patch.  The shortcut is that the ray fan has *already*
sampled the seabed: every bottom reflection in a traced bundle is one patch,
weighted by the solid angle its ray carries.  One render therefore yields the
whole reverberation series.

For a monostatic sonar -- and a forward-looking array a few hundred millimetres
across, at tens of metres range, is monostatic to well within a beamwidth --
reciprocity closes the return path for free: the echo retraces the outbound
ray, so it arrives at ``2 tau``, carries the outbound loss twice, and comes back
along the reverse of its own launch direction.  That last point is what lets
reverberation be beamformed: each patch has a bearing.

The energy a single bounce returns is

.. math::
    E = \\Delta\\Omega \; \\frac{\\sigma_b(\\theta,\\theta)}{\\sin\\theta}
        \; \\frac{L^2}{r^2}

where ``r`` is the slant range to the patch, ``L`` the one-way loss, and
``DeltaOmega r^2 / sin(theta)`` the patch area the ray's solid angle projects
onto the seabed -- the ``1/sin`` being the grazing stretch.  With Lambert
scattering, ``sigma_b = mu sin^2(theta)``, this collapses to
``DeltaOmega mu sin(theta) L^2 / r^2``, and summing it over a flat bottom
reproduces the classical ``r^-5`` reverberation decay (see the tests).

Phase
-----
Scattering off a boundary rough on the scale of a wavelength randomises phase,
so each patch is given an independent uniform random phase from a seeded
generator.

Worth being precise about how much this buys, because it is less than it first
appears: patches within one range cell already differ in path length by many
wavelengths at 100 kHz, so their *carrier* phase ``-omega tau`` is effectively
random whatever the scattering phase.  Zeroing the scattering phase therefore
does **not** make reverberation add coherently -- measured, it changes the
beamformed peak by well under an order of magnitude, and downwards at that.
What the random phase does fix is the residual case the carrier does not cover:
patches at equal range and nearby bearing, which share a travel time and would
otherwise add in amplitude rather than in power and return an unphysically
bright, target-like echo.

The practical consequence either way is that reverberation is spread over the
whole insonified range window rather than concentrated in a few cells, which is
what the tests check.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .absorption import thorp_db_per_km
from .beamform import ArrivalSet
from .mesh import segment_mesh_transmission
from .tracer import TraceResult, bounce_events


def _lit_patches(result: TraceResult, events, occluders) -> Tensor:
    """Which bounce events the occluding bodies still let the source reach.

    The outbound leg is taken as the straight segment from wherever the ray last
    left a boundary -- the source, for the first bounce -- to the patch.  Under
    refraction that is an approximation, but a shadow is cast over the last few
    tens of metres of a path, where the bending is far below the width of the
    body casting it.

    Monostatic, so one test does both legs: the return retraces the outbound
    ray, and a patch the body hides on the way out is equally hidden on the way
    back.
    """
    n = events.count
    lit = torch.ones(n, dtype=torch.bool, device=events.position.device)
    if n == 0:
        return lit
    source = result.pos[events.ray, 0].detach()
    patch = events.position.detach()
    if n > 1:
        # bounce_events comes out sorted by (ray, step), so the previous event
        # is the previous bounce whenever it belongs to the same ray.
        same = (events.ray[1:] == events.ray[:-1]).unsqueeze(-1)
        start = torch.cat([source[:1],
                           torch.where(same, patch[:-1], source[1:])], dim=0)
    else:
        start = source
    for vertices, faces in occluders:
        lit = lit & (segment_mesh_transmission(start, patch,
                                               vertices.detach(), faces) > 0.5)
    return lit

__all__ = [
    "LambertScattering",
    "cone_solid_angle",
    "reverberation_arrivals",
    "render_reverberation",
]


def cone_solid_angle(half_angle_deg: float) -> float:
    """Solid angle of a cone, ``2 pi (1 - cos a)`` steradians."""
    return 2.0 * math.pi * (1.0 - math.cos(math.radians(half_angle_deg)))


class LambertScattering(nn.Module):
    r"""Lambert's law, ``sigma_b = mu sin(theta_i) sin(theta_s)``.

    The workhorse seabed backscatter model.  ``mu`` is quoted in dB and runs
    from about -35 dB for smooth mud to -15 dB for rock; -27 dB is a common
    default for sand.  It is learnable, so a measured reverberation series can
    be inverted for the bottom type.
    """

    def __init__(self, strength_db: float = -27.0, *, learnable: bool = True) -> None:
        super().__init__()
        t = torch.as_tensor(float(strength_db))
        if learnable:
            self.strength_db = nn.Parameter(t)
        else:
            self.register_buffer("strength_db", t)

    def mu(self) -> Tensor:
        return 10.0 ** (self.strength_db / 10.0)

    def forward(self, grazing_in: Tensor, grazing_out: Tensor) -> Tensor:
        return (self.mu().to(grazing_in.dtype)
                * torch.sin(grazing_in) * torch.sin(grazing_out))

    def extra_repr(self) -> str:
        return f"strength_db={float(self.strength_db):.1f}"


def reverberation_arrivals(
    result: TraceResult,
    launch_directions: Tensor,
    freqs_khz: Tensor,
    *,
    scattering: LambertScattering,
    solid_angle_per_ray: float,
    ray_weights: Tensor | None = None,
    absorption=thorp_db_per_km,
    boundary: str = "bottom",
    surface=None,
    bottom=None,
    spread_min_range: float = 1.0,
    generator: torch.Generator | None = None,
    max_arrivals: int | None = None,
    occluders=None,
    surface_gain=None,
) -> ArrivalSet:
    """Monostatic reverberation as an :class:`~hydropt.beamform.ArrivalSet`.

    Every boundary reflection in ``result`` becomes one arrival, at twice its
    one-way time, travelling back along the reverse of its launch direction.
    The output drops straight into :func:`hydropt.beamform.beamform`, so
    reverberation and target echoes can be summed before beamforming -- which
    is the only way to see whether a target is actually detectable.

    Args:
        result: a traced bundle from the projector.
        launch_directions: the ``[R, 3]`` directions that produced it.
        scattering: backscatter model.
        solid_angle_per_ray: steradians each ray represents, e.g.
            ``cone_solid_angle(half_angle) / n_rays``.
        ray_weights: transmit directivity, applied **once**.  The outbound leg
            passes through the projector's pattern; the return does not -- it
            arrives at the receive array, whose directivity is the beamformer's
            job, not a per-ray weight.  Squaring it here (an easy mistake, since
            the path is reciprocal in *geometry*) suppresses off-axis patches
            twice over and makes reverberation look far weaker than it is.
        boundary: ``"bottom"``, ``"surface"`` or ``"both"``.
        surface, bottom: the scene's boundaries.  Pass both: without them each
            bounce is taken at the vertex *after* the reflection, which biases
            every patch range by up to one step length -- a systematic error in
            the reverberation range scale, not noise.
        generator: seeds the random scattering phase and the subsampling.
        occluders: bodies that cast shadows, as ``(vertices, faces)`` meshes in
            the **world frame**.  A patch the body hides returns nothing, which
            is how an object on the seabed paints the dark band behind itself --
            and that band, not the object's own echo, is what an operator reads
            its height from.  Without this a bottom object sits in the image
            with the seabed showing straight through behind it.
        surface_gain: ``f(xy) -> [P]`` linear multiplier on the *surface*
            patches' scattering strength, from their horizontal position.  The
            grazing-angle model is one law for the whole boundary, which is
            right for a wind sea and wrong wherever something has changed the
            surface locally -- a bubble wake being the case that matters here,
            since it is tens of dB above the ambient sea and is what a wake
            actually looks like in a sonar image.  A multiplier rather than a
            replacement, so it composes with the angle law instead of
            overriding it, and differentiable, so what produced the patch is
            still fittable through it.
        max_arrivals: cap the patch count by keeping a **random** subset and
            scaling its energy to compensate.  Keeping the *strongest* patches
            instead -- the sensible choice for target echoes -- is badly wrong
            here: reverberation falls off as ``r^-5``, so the strongest patches
            are all at the shortest ranges and the sample collapses onto the
            first few range cells instead of filling the window.  Random
            subsampling is unbiased in expectation and preserves the range
            distribution.
    """
    if boundary not in ("bottom", "surface", "both"):
        raise ValueError(f"unknown boundary {boundary!r}")
    dtype, device = result.pos.dtype, result.pos.device
    freqs_khz = freqs_khz.to(dtype=dtype, device=device)

    events = bounce_events(result, bottom_only=(boundary == "bottom"),
                           surface=surface, bottom=bottom)
    if boundary == "surface":
        keep = ~events.is_bottom
        events = type(events)(*(t[keep] for t in events))
    if occluders:
        lit = _lit_patches(result, events, occluders)
        events = type(events)(*(t[lit] for t in events))
    if events.count == 0:
        z = torch.zeros(0, dtype=dtype, device=device)
        return ArrivalSet(z, torch.zeros(0, int(freqs_khz.shape[0]), dtype=dtype,
                                         device=device),
                          torch.zeros(0, 3, dtype=dtype, device=device), z, z, z)

    r = events.arclen.clamp_min(spread_min_range)
    graze = events.grazing.clamp_min(1e-6)
    sigma_b = scattering(graze, graze)
    if surface_gain is not None:
        gain = surface_gain(events.position[..., :2]).to(dtype=dtype,
                                                         device=device)
        sigma_b = sigma_b * torch.where(events.is_bottom,
                                        torch.ones_like(gain), gain)

    # One-way loss to the patch: specular reflections already taken, plus
    # volume absorption over the outbound path.
    alpha = absorption(freqs_khz).view(1, -1)  # [1, B] dB/km
    one_way = (10.0 ** (-events.refl_db_incident / 10.0)).unsqueeze(1) * 10.0 ** (
        -(alpha * r.unsqueeze(1)) / 1.0e4)

    weight = torch.full_like(r, float(solid_angle_per_ray))
    if ray_weights is not None:
        # Outbound only: the return arrives at the receive array, not back
        # through the projector.
        weight = weight * ray_weights.to(dtype=dtype, device=device)[events.ray]

    energy = ((weight * sigma_b / torch.sin(graze) / (r * r)).unsqueeze(1)
              * one_way * one_way)

    dirs = launch_directions.to(dtype=dtype, device=device)
    dirs = dirs / dirs.norm(dim=-1, keepdim=True).clamp_min(1e-30)
    # Reciprocity: the return retraces the outbound ray, so it reaches the array
    # travelling opposite to the way it was launched.
    direction = -dirs[events.ray]

    phase = torch.rand(events.count, dtype=dtype, device=device,
                       generator=generator) * (2.0 * math.pi)

    arrivals = ArrivalSet(
        time=2.0 * events.time,
        amplitude=energy.clamp_min(0.0).sqrt(),
        direction=direction,
        phase=phase,
        distance=torch.zeros_like(r),
        path_length=2.0 * r,
    )
    if max_arrivals is not None and arrivals.n_arrivals > max_arrivals:
        n = arrivals.n_arrivals
        pick = torch.randperm(n, generator=generator, device=device)[:max_arrivals]
        pick = pick[arrivals.time.detach()[pick].argsort()]
        # Each retained patch now stands for n / max_arrivals of them, so its
        # energy is scaled up accordingly -- amplitude by the square root.
        scale = math.sqrt(n / max_arrivals)
        arrivals = ArrivalSet(
            time=arrivals.time[pick],
            amplitude=arrivals.amplitude[pick] * scale,
            direction=arrivals.direction[pick],
            phase=arrivals.phase[pick],
            distance=arrivals.distance[pick],
            path_length=arrivals.path_length[pick],
        )
    return arrivals


def render_reverberation(
    arrivals: ArrivalSet,
    time_grid: Tensor,
    *,
    sigma_t: float,
    time_gate: float = 5.0,
) -> Tensor:
    """Splat reverberation arrivals into an energy-time curve, ``[1, bands, bins]``.

    Incoherent: reverberation is a sum of independently-phased patches, so
    energies add.  Use :func:`hydropt.beamform.beamform` on the same arrivals
    when the bearing structure matters.
    """
    dtype, device = time_grid.dtype, time_grid.device
    n_time = int(time_grid.shape[0])
    if arrivals.n_arrivals == 0:
        return torch.zeros(1, arrivals.amplitude.shape[-1], n_time,
                           dtype=dtype, device=device)

    t0 = time_grid[0]
    dt = (time_grid[-1] - time_grid[0]) / (n_time - 1)
    half_w = int(math.ceil(time_gate * sigma_t / float(dt)))
    offsets = torch.arange(-half_w, half_w + 1, device=device)
    norm_t = 1.0 / (math.sqrt(2.0 * math.pi) * sigma_t)

    tau = arrivals.time
    centre = torch.round((tau.detach() - t0) / dt).long()
    bins = centre.unsqueeze(1) + offsets
    valid = (bins >= 0) & (bins < n_time)
    bins_c = bins.clamp(0, n_time - 1)
    kern = torch.exp(-0.5 * ((tau.unsqueeze(1) - (t0 + bins_c.to(dtype) * dt)) / sigma_t) ** 2)
    kern = kern * norm_t * valid.to(dtype)

    energy = arrivals.amplitude**2  # [A, B]
    n_band = int(energy.shape[1])
    vals = energy.unsqueeze(2) * kern.unsqueeze(1)  # [A, B, W]
    band = torch.arange(n_band, device=device).view(1, -1, 1)
    flat = (band * n_time + bins_c.unsqueeze(1)).reshape(-1)
    out = torch.zeros(n_band * n_time, dtype=dtype, device=device)
    return out.index_add(0, flat, vals.reshape(-1)).view(1, n_band, n_time)
