"""Eigenrays: the discrete paths from a point to a receiver, found exactly.

A ray fan does not hit a receiver.  The usual way round that is to accept every
ray passing within some ``sigma_d`` of it and weight by the miss distance -- the
splat that :func:`hydropt.beamform.extract_arrivals` does -- which works, and
has one property that is fatal at long range: ``sigma_d`` has to scale with the
fan's spacing, so it is an *acceptance aperture set by the sampling* rather than
by the receiver.

Measured on a 0.31 m array at 250 m: a 45 degree return cone of 420 rays gives
7.8 degrees of ray spacing, so ``sigma_d`` comes out at 18 m and the extraction
collects over ninety times the real aperture.  Every accepted ray keeps its own
direction, so a target's rendered angular size can never be smaller than the
fan's angular spacing: a 30 m hull subtending 4.82 degrees imaged as 12.58, its
25 m of across-bearing extent reading as 54 m.

Narrowing the cone does not fix it.  Measured, a 45 degree cone carries 80
percent of its energy in surface and bottom bounces that a 1 degree cone never
finds -- the wide cone is not over-collecting, it is the only one seeing the
multipath.  Resolving a 0.31 m aperture *and* keeping a 45 degree cone needs
0.07 degree spacing, about five million rays.  The fan cannot do both.

So this module stops sampling and solves instead.  Shoot a coarse fan only to
find out **how many** distinct paths there are and roughly where they leave,
then refine each one's launch direction until the ray lands on the receiver.
What comes back is one arrival per path, with:

  * the direction the ray actually arrives from, not a nearby ray's;
  * spreading from the ray tube's own divergence -- the same 2x2 Jacobian the
    refinement already needs, whose determinant is the tube's cross-section,
    and which reduces to ``1/L^2`` for a straight ray in a homogeneous medium;
  * no ``sigma_d`` anywhere, so nothing depends on how the fan was sampled.

**Differentiable.**  The bracketing and all but the last refinement step run
under ``no_grad``, and the final Newton step carries the exact implicit
derivative of the converged launch direction -- the same split
:func:`hydropt.boundaries.find_crossing` uses to make a bisection
differentiable.  The arrival *time* needs even less: an eigenray is a
stationary point of travel time among paths joining its endpoints, so by
Fermat the launch direction can be detached and the time still differentiates
correctly.

**What is discontinuous.**  The number of paths.  Move a target far enough and
a bounce path appears or vanishes, and the arrival count steps.  Within a
regime the gradient is exact; across one it is undefined.  That is the same
class of discontinuity the splat already has when a ray crosses its acceptance
gate, and the same the tracer has at every accept/reject.
"""

from __future__ import annotations

import math
import warnings

import torch
from torch import Tensor

from .absorption import thorp_db_per_km
from .beamform import ArrivalSet
from .boundaries import FlatHeight, HeightField
from .launch import fibonacci_cone
from .receiver import _closest_approach
from .rough import roughness_weights
from .tracer import trace

__all__ = ["eigenray_arrivals", "find_eigenrays", "mean_boundary"]


def mean_boundary(boundary: HeightField) -> tuple[HeightField, float]:
    """A boundary's mean plane and the RMS it departs from it.

    An eigenray through a ROUGH boundary is not a well-posed thing to solve
    for.  Perturb the launch direction by a hair and the ray reflects off a
    different facet of the wave field, so the miss distance jumps rather than
    varying smoothly and Newton has no derivative to work with.  Measured in a
    120 kHz scene over a Pierson-Moskowitz sea, the refinement left residuals
    of 2.8, 7.7, 15.8 and 64.6 m where the same geometry with flat boundaries
    converged every path to under 3 cm.

    The physics says the same thing.  A coherent field's specular path is
    defined on the MEAN surface; roughness does not move it, it costs it
    amplitude -- the Eckart coherence factor, which at 100 kHz over a wind sea
    is tens of orders of magnitude.  What the roughness scatters elsewhere is
    not a coherent arrival at all, it is reverberation, and
    :func:`hydropt.reverb.reverberation_arrivals` already models it.

    So solve on the mean plane and pay the coherence loss, rather than chasing
    a path through the facets.
    """
    h = getattr(boundary, "heights", None)
    if h is None:                       # already flat, or has no height grid
        return boundary, 0.0
    with torch.no_grad():
        mean = float(h.mean())
        rms = float((h - h.mean()).pow(2).mean().sqrt())
    return FlatHeight(mean), rms


class _SmoothedScene:
    """A scene view whose boundaries are their own mean planes."""

    def __init__(self, scene, source: Tensor) -> None:
        surface, self.surface_rms = mean_boundary(scene.surface)
        bottom, self.bottom_rms = mean_boundary(scene.bottom)
        object.__setattr__(self, "_scene", scene)
        object.__setattr__(self, "_source", source)
        object.__setattr__(self, "_surface", surface)
        object.__setattr__(self, "_bottom", bottom)

    def __getattr__(self, name: str):
        if name == "surface":
            return self._surface
        if name == "bottom":
            return self._bottom
        return getattr(self._scene, name)

    def source_position(self) -> Tensor:
        return self._source


def _frame(axis: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Orthonormal ``(a, e1, e2)`` with ``a`` along ``axis``."""
    a = axis.reshape(3) / axis.reshape(3).norm().clamp_min(1e-30)
    other = (torch.tensor([0.0, 0.0, 1.0], dtype=a.dtype, device=a.device)
             if abs(float(a[2])) < 0.9
             else torch.tensor([1.0, 0.0, 0.0], dtype=a.dtype, device=a.device))
    e1 = torch.linalg.cross(a, other)
    e1 = e1 / e1.norm().clamp_min(1e-30)
    return a, e1, torch.linalg.cross(a, e1)


def _aim(frame: tuple[Tensor, Tensor, Tensor], uv: Tensor) -> Tensor:
    """Unit directions from 2-D transverse offsets ``uv`` ``[P, 2]``."""
    a, e1, e2 = frame
    d = (a.reshape(1, 3) + uv[:, :1] * e1.reshape(1, 3)
         + uv[:, 1:2] * e2.reshape(1, 3))
    return d / d.norm(dim=-1, keepdim=True).clamp_min(1e-30)


def _closest(result, point: Tensor):
    """Per-ray closest approach to ``point``: miss vector and where it happened.

    Returns ``(miss [R, 3], step [R], frac [R])`` -- the vector from the point
    to the ray at closest approach, the segment it fell in and how far along it.
    """
    pos = result.pos
    p0, p1 = pos[:, :-1], pos[:, 1:]
    seg = p1 - p0
    seg_len2 = (seg * seg).sum(-1)
    tstar, dist = _closest_approach(p0, seg, seg_len2, point.reshape(1, 3))
    tstar, dist = tstar[..., 0], dist[..., 0]                    # [R, S]
    live = (result.alive[:, :-1] * result.alive[:, 1:]) > 0
    dist = torch.where(live & (seg_len2 > 0), dist,
                       torch.full_like(dist, float("inf")))
    step = dist.detach().argmin(dim=1)                           # [R]
    rows = torch.arange(pos.shape[0], device=pos.device)
    frac = tstar[rows, step]
    at = p0[rows, step] + frac.unsqueeze(-1) * seg[rows, step]
    return at - point.reshape(1, 3), step, frac


def _bounce_signature(result, step: Tensor) -> Tensor:
    """How many surface and bottom reflections precede each closest approach.

    The label that separates one path from another: a direct arrival, a surface
    bounce and a bottom bounce are different paths even when they land within a
    metre of each other, and they must not be averaged together.
    """
    g = result.bounce_grazing
    idx = torch.arange(g.shape[1], device=g.device).reshape(1, -1)
    before = idx <= step.reshape(-1, 1)
    surface = ((g < 0) & before).sum(dim=1)
    bottom = ((g > 0) & before).sum(dim=1)
    return surface * 1000 + bottom


def find_eigenrays(scene, source: Tensor, receiver: Tensor, *,
                   bracket_rays: int = 2000,
                   bracket_half_angle_deg: float = 60.0,
                   max_paths: int = 8, n_refine: int = 8,
                   tolerance: float | None = None,
                   trace_kwargs: dict | None = None):
    """Launch directions of every path from ``source`` to ``receiver``.

    Args:
        scene: the scene; its own source is ignored, ``source`` is used.
        source, receiver: ``[3]`` points.
        bracket_rays, bracket_half_angle_deg: the coarse fan, used ONLY to count
            the paths and bracket them.  It needs to be wide enough to contain
            the bounces you care about -- that is what a 45 degree cone was
            doing right -- and only coarse enough to separate them.
        max_paths: keep this many distinct paths, nearest first.
        n_refine: Newton steps.  All but the last run under ``no_grad``.
        tolerance: stop refining a path once it lands this close (m).  Defaults
            to a thousandth of the source-receiver separation.

    Returns ``(directions [P, 3], miss [P], signature [P])``.
    """
    source = source.reshape(3)
    receiver = receiver.reshape(3)
    tkw = dict(trace_kwargs or {})
    span = float((receiver - source).detach().norm())
    tol = tolerance if tolerance is not None else 1e-3 * span
    frame = _frame((receiver - source).detach())

    # --- bracket: how many paths are there, and roughly where do they leave?
    with torch.no_grad():
        fan = fibonacci_cone(bracket_rays, frame[0], bracket_half_angle_deg)
        res = trace(_SmoothedScene(scene, source.reshape(1, 3).expand(
            bracket_rays, 3)), fan, **tkw)
        miss, step, _ = _closest(res, receiver)
        d = miss.norm(dim=-1)
        sig = _bounce_signature(res, step)
        picks = []
        for s in sig.unique():
            same = (sig == s).nonzero().reshape(-1)
            best = same[d[same].argmin()]
            picks.append((float(d[best]), int(best), int(s)))
        picks.sort()
        picks = picks[:max_paths]
    if not picks:
        empty = torch.zeros(0, 3, dtype=source.dtype, device=source.device)
        z = torch.zeros(0, dtype=source.dtype, device=source.device)
        return empty, z, z.long()

    # Work in the transverse offsets of the frame: a launch direction is
    # `normalise(a + u e1 + v e2)`, so the miss is a smooth function of (u, v)
    # and a 2x2 Newton step is all that is needed.
    a, e1, e2 = frame
    dirs = fan[[i for _, i, _ in picks]]
    uv = torch.stack([(dirs * e1).sum(-1) / (dirs * a).sum(-1).clamp_min(1e-9),
                      (dirs * e2).sum(-1) / (dirs * a).sum(-1).clamp_min(1e-9)],
                     dim=-1)
    signature = torch.tensor([s for _, _, s in picks], device=source.device)
    step_uv = max(1e-6, 0.25 * tol / max(span, 1e-9))

    def probe(u: Tensor, src: Tensor):
        """Miss vectors, transverse components, for P candidate directions."""
        p = u.shape[0]
        directions = _aim(frame, u)
        r = trace(_SmoothedScene(scene, src.reshape(1, 3).expand(p, 3)),
                  directions, **tkw)
        m, st, fr = _closest(r, receiver)
        return torch.stack([(m * e1).sum(-1), (m * e2).sum(-1)], dim=-1), r, st, fr

    # Converge under no_grad, then ALWAYS take one more step with gradient
    # tracking.  Breaking out of the loop the moment the residual is small
    # skips that step and the launch direction comes back a constant -- which
    # is exactly what happened once solving on the mean planes made the
    # refinement converge in two or three iterations instead of grinding:
    # the speed-up and a dead gradient to the target's own pose were the same
    # change.  The final step is what carries the implicit derivative, so it
    # is not optional and cannot be skipped for being unnecessary numerically.
    for it in range(n_refine + 1):
        last = it == n_refine
        with torch.set_grad_enabled(last):
            src = source if last else source.detach()
            base, _, _, _ = probe(uv, src)
            if not last and float(base.norm(dim=-1).max()) < tol:
                uv = uv.detach()
                continue
            # 2x2 Jacobian by central differences on the two offsets.
            with torch.no_grad():
                jac = []
                for k in range(2):
                    bump = torch.zeros_like(uv)
                    bump[:, k] = step_uv
                    plus, _, _, _ = probe(uv.detach() + bump, source.detach())
                    minus, _, _, _ = probe(uv.detach() - bump, source.detach())
                    jac.append((plus - minus) / (2 * step_uv))
                j = torch.stack(jac, dim=-1)                     # [P, 2, 2]
                det = j[:, 0, 0] * j[:, 1, 1] - j[:, 0, 1] * j[:, 1, 0]
                ok = det.abs() > 1e-12
            inv = torch.zeros_like(j)
            safe = torch.where(ok, det, torch.ones_like(det)).reshape(-1, 1, 1)
            inv[:, 0, 0], inv[:, 1, 1] = j[:, 1, 1], j[:, 0, 0]
            inv[:, 0, 1], inv[:, 1, 0] = -j[:, 0, 1], -j[:, 1, 0]
            stepv = (inv / safe @ base.unsqueeze(-1)).squeeze(-1)
            uv = uv - torch.where(ok.reshape(-1, 1), stepv,
                                  torch.zeros_like(stepv))

    with torch.no_grad():
        final, _, _, _ = probe(uv.detach(), source.detach())
        residual = final.norm(dim=-1)
    return _aim(frame, uv), residual, signature


def eigenray_arrivals(scene, source: Tensor, receiver: Tensor,
                      freqs_khz: Tensor, *, absorption=thorp_db_per_km,
                      spread_min_range: float = 1.0,
                      accept: float | None = None,
                      coherent: bool = False,
                      **kwargs) -> ArrivalSet:
    """One arrival per path from ``source`` to ``receiver``, no splat.

    The amplitude carries the ray tube's own divergence rather than an assumed
    ``1/L^2``: the 2x2 Jacobian ``d(miss)/d(launch offsets)`` that the
    refinement needs IS the tube's cross-section per unit solid angle, so its
    determinant is the spreading.  For a straight ray in a homogeneous medium
    it comes out as ``L^2`` exactly, which is the check in the tests.

    Args:
        coherent: multiply each bounce path by its Eckart coherence factor, for
            a receiver that needs the coherent field at a point.  Off by
            default: an imaging sonar collects the bounce's energy across
            elevation whatever the surface did to its phase, and the ghost
            returns a shallow-water operator expects come from exactly these
            paths.  See the note at the calculation.
        accept: discard paths that still miss by more than this (m).  Defaults
            to a hundredth of the source-receiver separation; a path that will
            not converge is one the bracket found and the refinement could not
            close, and reporting it as an arrival would be inventing one.

    Returns an :class:`~hydropt.beamform.ArrivalSet` shaped exactly like
    :func:`hydropt.beamform.extract_arrivals`, so it drops in wherever that
    does.
    """
    source = source.reshape(3)
    receiver = receiver.reshape(3)
    dtype, device = source.dtype, source.device
    freqs_khz = freqs_khz.to(dtype=dtype, device=device)
    span = float((receiver - source).detach().norm())
    keep_within = accept if accept is not None else 1e-2 * span

    directions, residual, _ = find_eigenrays(scene, source, receiver, **kwargs)
    good = residual <= keep_within
    if int(good.sum()) == 0:
        z = torch.zeros(0, dtype=dtype, device=device)
        return ArrivalSet(z, torch.zeros(0, int(freqs_khz.shape[0]),
                                         dtype=dtype, device=device),
                          torch.zeros(0, 3, dtype=dtype, device=device),
                          z, z, z, torch.zeros(0, 3, dtype=dtype, device=device))
    directions = directions[good]
    p = directions.shape[0]

    tkw = dict(kwargs.get("trace_kwargs") or {})
    smooth = _SmoothedScene(scene, source.reshape(1, 3).expand(p, 3))
    result = trace(smooth, directions, **tkw)
    miss, step, frac = _closest(result, receiver)
    rows = torch.arange(p, device=device)

    tau_a, tau_b = result.tau[rows, step], result.tau[rows, step + 1]
    time = tau_a + frac * (tau_b - tau_a)
    s_a, s_b = result.arclen[rows, step], result.arclen[rows, step + 1]
    path_length = s_a + frac * (s_b - s_a)
    seg = result.pos[rows, step + 1] - result.pos[rows, step]
    direction = seg / seg.norm(dim=-1, keepdim=True).clamp_min(1e-30)
    first = result.pos[:, 1] - result.pos[:, 0]
    launch = first / first.norm(dim=-1, keepdim=True).clamp_min(1e-30)
    db = result.refl_db[rows, step + 1]
    phase = result.refl_phase[rows, step + 1]

    # Ray-tube divergence, from the same Jacobian the refinement used.  The
    # tube's cross-section at the receiver per unit solid angle at the source is
    # |det d(transverse position)/d(launch offsets)|, and intensity is its
    # reciprocal.  Straight ray, homogeneous medium: the offsets are angles, the
    # transverse position grows as L times the angle, so |det| = L^2 and this is
    # 1/L^2 -- the spherical spreading it has to reduce to.
    frame = _frame((receiver - source).detach())
    a, e1, e2 = frame
    with torch.no_grad():
        base_uv = torch.stack(
            [(directions * e1).sum(-1) / (directions * a).sum(-1).clamp_min(1e-9),
             (directions * e2).sum(-1) / (directions * a).sum(-1).clamp_min(1e-9)],
            dim=-1)
        h = max(1e-6, 1e-4)
        cols = []
        for k in range(2):
            bump = torch.zeros_like(base_uv)
            bump[:, k] = h
            out = []
            for sign in (1.0, -1.0):
                d2 = _aim(frame, base_uv + sign * bump)
                r2 = trace(_SmoothedScene(
                    scene, source.detach().reshape(1, 3).expand(p, 3)),
                    d2, **tkw)
                m2, _, _ = _closest(r2, receiver)
                out.append(torch.stack([(m2 * e1).sum(-1), (m2 * e2).sum(-1)],
                                       dim=-1))
            cols.append((out[0] - out[1]) / (2 * h))
        j = torch.stack(cols, dim=-1)
        area = (j[:, 0, 0] * j[:, 1, 1] - j[:, 0, 1] * j[:, 1, 0]).abs()
        # A tube far tighter than spherical is a Jacobian that failed, not a
        # caustic.  Focusing by more than this over a single leg would be a
        # remarkable piece of geometry; a finite difference that straddled a
        # discontinuity is the ordinary explanation, and letting it through
        # multiplies the energy by the reciprocal of however small it got.
        spherical = path_length.detach().clamp_min(spread_min_range) ** 2
        clamped = int((area < 1e-2 * spherical).sum())
        if clamped:
            warnings.warn(
                f"{clamped} of {int(area.numel())} ray tubes came back tighter "
                f"than a hundredth of spherical and were clamped. That is "
                f"usually a Jacobian straddling a discontinuity rather than "
                f"real focusing, and the level leans on the clamp.",
                RuntimeWarning, stacklevel=2)
        area = area.clamp_min(1e-2 * spherical)
    spread = 1.0 / area

    # Roughness costs the path amplitude, it does not move it.  The solve ran
    # on the mean planes, so every bounce now pays its Eckart coherence factor
    # at each band -- which at 100 kHz over a wind sea is tens of orders of
    # magnitude, i.e. a coherent surface bounce at these frequencies is
    # nothing.  The energy the roughness scatters elsewhere is reverberation
    # and is somebody else's job.
    # Bounce paths keep their ENERGY, deliberately.  A rough sea at 120 kHz
    # destroys the coherent reflection -- the Eckart factor is ~-100 dB -- but
    # a pressure-release surface reflects all of the energy; roughness smears
    # the bounce over a few degrees of elevation, and a horizontal line array
    # has no elevation resolution, so it collects that energy regardless.
    # Charging the coherence loss here (which this solver briefly did) deleted
    # every bounce path -- 8 per highlight down to the direct one -- and with
    # them the hull's ghost a few metres beyond it that every operator knows
    # from shallow water.  Eckart is for a coherent field at a point; an
    # imaging sonar's echo is an energy quantity across elevation.
    #
    # `coherent=True` restores the factor, evaluated `up_to_step` because the
    # path ends at the receiver and the trace does not: a ray that arrives
    # early keeps flying, and a bounce it makes out there is no part of the
    # path.  Counting it once charged the direct path a surface bounce it never
    # made, and the boat left the image.
    if coherent:
        coherence = roughness_weights(
            result, freqs_khz, surface_rms=smooth.surface_rms,
            bottom_rms=smooth.bottom_rms, surface=smooth.surface,
            bottom=smooth.bottom, up_to_step=step, sound_speed=float(scene.field(
                source.reshape(1, 3)).reshape(-1)[0]))
    else:
        coherence = torch.ones(p, int(freqs_khz.shape[0]), dtype=dtype, device=device)

    alpha = absorption(freqs_khz).view(1, -1)
    energy = ((spread * 10.0 ** (-db / 10.0)).unsqueeze(1)
              * 10.0 ** (-(alpha * path_length.unsqueeze(1)) / 1.0e4)
              * coherence)
    # A path the roughness has annihilated is not an arrival, and carrying it
    # at exactly zero is worse than dropping it: the derivative of sqrt at zero
    # is infinite, so a single underflowed path turns the whole gradient into
    # NaN.  That is not hypothetical -- the Eckart factor for a surface bounce
    # at 120 kHz over a 0.09 m sea underflows in float64, and it took the
    # gradient to the target's own pose with it.
    with torch.no_grad():
        alive = energy.detach().max(dim=1).values > 0.0
    if not bool(alive.any()):
        z = torch.zeros(0, dtype=dtype, device=device)
        return ArrivalSet(z, torch.zeros(0, int(freqs_khz.shape[0]),
                                         dtype=dtype, device=device),
                          torch.zeros(0, 3, dtype=dtype, device=device),
                          z, z, z, torch.zeros(0, 3, dtype=dtype, device=device))
    keep = alive.nonzero().reshape(-1)
    time, energy = time[keep], energy[keep]
    direction, phase = direction[keep], phase[keep]
    miss, path_length, launch = miss[keep], path_length[keep], launch[keep]

    order = time.detach().argsort()
    return ArrivalSet(time=time[order],
                      amplitude=energy.sqrt()[order],
                      direction=direction[order], phase=phase[order],
                      distance=miss.norm(dim=-1).detach()[order],
                      path_length=path_length[order],
                      launch_direction=launch[order])
