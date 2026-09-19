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
from .fields import IsoProfile
from .launch import fibonacci_cone
from .receiver import _LEN_EPS
from .rough import coherent_reflection_loss_db, roughness_weights
from .tracer import trace

__all__ = ["eigenray_arrivals", "eigenray_arrivals_batched", "find_eigenrays",
           "find_eigenrays_batched", "image_arrivals_batched", "mean_boundary"]


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


def _frames(axes: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Orthonormal ``(a, e1, e2)``, each ``[N, 3]``, with ``a`` along each row."""
    a = axes.reshape(-1, 3)
    a = a / a.norm(dim=-1, keepdim=True).clamp_min(1e-30)
    z = torch.tensor([0.0, 0.0, 1.0], dtype=a.dtype, device=a.device)
    x = torch.tensor([1.0, 0.0, 0.0], dtype=a.dtype, device=a.device)
    other = torch.where(a[:, 2:3].abs() < 0.9, z.view(1, 3), x.view(1, 3))
    e1 = torch.linalg.cross(a, other.expand_as(a), dim=-1)
    e1 = e1 / e1.norm(dim=-1, keepdim=True).clamp_min(1e-30)
    return a, e1, torch.linalg.cross(a, e1, dim=-1)


def _frame(axis: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Orthonormal ``(a, e1, e2)`` with ``a`` along ``axis``."""
    a, e1, e2 = _frames(axis.reshape(1, 3))
    return a[0], e1[0], e2[0]


def _aim(frame: tuple[Tensor, Tensor, Tensor], uv: Tensor) -> Tensor:
    """Unit directions from 2-D transverse offsets ``uv`` ``[P, 2]``.

    The frame is one ``[3]`` triple shared by every row, or ``[P, 3]`` triples,
    one per row.
    """
    a, e1, e2 = (f.reshape(-1, 3) for f in frame)
    d = a + uv[:, :1] * e1 + uv[:, 1:2] * e2
    return d / d.norm(dim=-1, keepdim=True).clamp_min(1e-30)


def _closest(result, point: Tensor):
    """Per-ray closest approach to ``point``: miss vector and where it happened.

    ``point`` is one ``[3]`` point for every ray or ``[R, 3]``, one per ray.
    Returns ``(miss [R, 3], step [R], frac [R])`` -- the vector from the point
    to the ray at closest approach, the segment it fell in and how far along it.
    """
    pos = result.pos
    p0, p1 = pos[:, :-1], pos[:, 1:]
    seg = p1 - p0
    seg_len2 = (seg * seg).sum(-1)
    target = point.reshape(-1, 1, 3)                             # [R or 1, 1, 3]
    rel = target - p0                                            # [R, S, 3]
    tstar = (rel * seg).sum(-1) / seg_len2.clamp_min(_LEN_EPS)
    tstar = tstar.clamp(0.0, 1.0)
    delta = rel - tstar.unsqueeze(-1) * seg
    # Not delta.norm(): its backward is 0/0 at zero distance, which a ray passing
    # exactly through a receiver would hit.
    dist = (delta * delta).sum(-1).clamp_min(_LEN_EPS).sqrt()   # [R, S]
    live = (result.alive[:, :-1] * result.alive[:, 1:]) > 0
    dist = torch.where(live & (seg_len2 > 0), dist,
                       torch.full_like(dist, float("inf")))
    step = dist.detach().argmin(dim=1)                           # [R]
    rows = torch.arange(pos.shape[0], device=pos.device)
    frac = tstar[rows, step]
    at = p0[rows, step] + frac.unsqueeze(-1) * seg[rows, step]
    return at - target[:, 0], step, frac


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


def _empty_arrivals(n_bands: int, dtype, device) -> ArrivalSet:
    z = torch.zeros(0, dtype=dtype, device=device)
    return ArrivalSet(z, torch.zeros(0, n_bands, dtype=dtype, device=device),
                      torch.zeros(0, 3, dtype=dtype, device=device),
                      z, z, z, torch.zeros(0, 3, dtype=dtype, device=device))


def find_eigenrays_batched(scene, sources: Tensor, receivers: Tensor, *,
                           bracket_rays: int = 2000,
                           bracket_half_angle_deg: float = 60.0,
                           max_paths: int | None = None, n_refine: int = 8,
                           tolerance: float | None = None,
                           trace_kwargs: dict | None = None):
    """:func:`find_eigenrays` for ``N`` source-receiver pairs in one go.

    Every pair's bracket fan goes into one trace and every pair's refinement
    probes into one trace per Newton iteration, so the cost of a solve is one
    Python loop over steps rather than one per pair -- which is most of it at
    these fan sizes.  Each pair converges on its own tolerance and stops
    stepping on its own; the arithmetic per path is what the single-pair
    solve does.

    Returns ``(directions [P, 3], miss [P], signature [P], pair [P])``, the
    last being which pair each path belongs to.
    """
    sources = sources.reshape(-1, 3)
    receivers = receivers.reshape(-1, 3)
    n_pairs = int(sources.shape[0])
    if receivers.shape[0] != n_pairs:
        raise ValueError("sources and receivers must pair up, one row each: got "
                         f"{n_pairs} and {int(receivers.shape[0])}")
    dtype, device = sources.dtype, sources.device
    tkw = dict(trace_kwargs or {})
    span = (receivers - sources).detach().norm(dim=-1)                  # [N]
    tol = (span * 1e-3 if tolerance is None
           else torch.full_like(span, float(tolerance)))
    a_n, e1_n, e2_n = _frames((receivers - sources).detach())

    def empty():
        z = torch.zeros(0, dtype=dtype, device=device)
        return (torch.zeros(0, 3, dtype=dtype, device=device), z, z.long(),
                z.long())

    # --- bracket: how many paths are there, and roughly where do they leave?
    with torch.no_grad():
        fan = torch.cat([fibonacci_cone(bracket_rays, a_n[n], bracket_half_angle_deg)
                         for n in range(n_pairs)], dim=0)                # [N B, 3]
        ray_src = sources.detach().repeat_interleave(bracket_rays, dim=0)
        ray_rcv = receivers.detach().repeat_interleave(bracket_rays, dim=0)
        res = trace(_SmoothedScene(scene, ray_src), fan, **tkw)
        miss, step, _ = _closest(res, ray_rcv)
        d = miss.norm(dim=-1)
        sig = _bounce_signature(res, step)
        pick_ray, pick_pair, pick_sig = [], [], []
        for n in range(n_pairs):
            lo = n * bracket_rays
            d_n, sig_n = d[lo:lo + bracket_rays], sig[lo:lo + bracket_rays]
            picks = []
            for s in sig_n.unique():
                same = (sig_n == s).nonzero().reshape(-1)
                best = same[d_n[same].argmin()]
                picks.append((float(d_n[best]), int(best), int(s)))
            picks.sort()
            for _, i, s in picks[:max_paths]:
                pick_ray.append(lo + i)
                pick_pair.append(n)
                pick_sig.append(s)
    if not pick_ray:
        return empty()

    pair = torch.tensor(pick_pair, device=device)                        # [P]
    signature = torch.tensor(pick_sig, device=device)
    a, e1, e2 = a_n[pair], e1_n[pair], e2_n[pair]                        # [P, 3]
    frame = (a, e1, e2)
    src_p, rcv_p = sources[pair], receivers[pair]
    tol_p, span_p = tol[pair], span[pair]

    # Work in the transverse offsets of the frame: a launch direction is
    # `normalise(a + u e1 + v e2)`, so the miss is a smooth function of (u, v)
    # and a 2x2 Newton step is all that is needed.
    dirs = fan[pick_ray]
    uv = torch.stack([(dirs * e1).sum(-1) / (dirs * a).sum(-1).clamp_min(1e-9),
                      (dirs * e2).sum(-1) / (dirs * a).sum(-1).clamp_min(1e-9)],
                     dim=-1)
    step_uv = (0.25 * tol_p / span_p.clamp_min(1e-9)).clamp_min(1e-6)    # [P]

    def probe(u: Tensor, src: Tensor):
        """Miss vectors, transverse components, for every candidate direction."""
        directions = _aim(frame, u)
        r = trace(_SmoothedScene(scene, src), directions, **tkw)
        m, st, fr = _closest(r, rcv_p)
        return torch.stack([(m * e1).sum(-1), (m * e2).sum(-1)], dim=-1), r, st, fr

    # Converge under no_grad, then ALWAYS take one more step with gradient
    # tracking.  Breaking out of the loop the moment the residual is small
    # skips that step and the launch direction comes back a constant -- which
    # is exactly what happened once solving on the mean planes made the
    # refinement converge in two or three iterations instead of grinding:
    # the speed-up and a dead gradient to the target's own pose were the same
    # change.  The final step is what carries the implicit derivative, so it
    # is not optional and cannot be skipped for being unnecessary numerically.
    #
    # A pair whose paths have all landed within its tolerance stops stepping
    # (its rows are masked out) while the others carry on, exactly as it would
    # have stopped in a solve of its own.
    for it in range(n_refine + 1):
        last = it == n_refine
        with torch.set_grad_enabled(last):
            src = src_p if last else src_p.detach()
            base, _, _, _ = probe(uv, src)
            if last:
                active = torch.ones_like(tol_p, dtype=torch.bool)
            else:
                worst = torch.zeros_like(tol).scatter_reduce(
                    0, pair, base.detach().norm(dim=-1), reduce="amax",
                    include_self=False)
                active = (worst >= tol)[pair]
                if not bool(active.any()):
                    uv = uv.detach()
                    continue
            # 2x2 Jacobian by central differences on the two offsets.
            with torch.no_grad():
                jac = []
                for k in range(2):
                    bump = torch.zeros_like(uv)
                    bump[:, k] = step_uv
                    plus, _, _, _ = probe(uv.detach() + bump, src_p.detach())
                    minus, _, _, _ = probe(uv.detach() - bump, src_p.detach())
                    jac.append((plus - minus) / (2 * step_uv.unsqueeze(-1)))
                j = torch.stack(jac, dim=-1)                     # [P, 2, 2]
                det = j[:, 0, 0] * j[:, 1, 1] - j[:, 0, 1] * j[:, 1, 0]
                ok = (det.abs() > 1e-12) & active
            inv = torch.zeros_like(j)
            safe = torch.where(ok, det, torch.ones_like(det)).reshape(-1, 1, 1)
            inv[:, 0, 0], inv[:, 1, 1] = j[:, 1, 1], j[:, 0, 0]
            inv[:, 0, 1], inv[:, 1, 0] = -j[:, 0, 1], -j[:, 1, 0]
            stepv = (inv / safe @ base.unsqueeze(-1)).squeeze(-1)
            uv = uv - torch.where(ok.reshape(-1, 1), stepv,
                                  torch.zeros_like(stepv))

    with torch.no_grad():
        final, _, _, _ = probe(uv.detach(), src_p.detach())
        residual = final.norm(dim=-1)
    return _aim(frame, uv), residual, signature, pair


def find_eigenrays(scene, source: Tensor, receiver: Tensor, *,
                   bracket_rays: int = 2000,
                   bracket_half_angle_deg: float = 60.0,
                   max_paths: int | None = None, n_refine: int = 8,
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
        max_paths: keep at most this many distinct paths, nearest first.
            ``None``, the default, keeps every path the bracket finds.  A cap
            is a trap: which paths lie nearest is decided by where the fan's
            rays happened to fall, not by how much energy they carry, and a
            cap of 8 in a six-bounce channel was found dropping the DIRECT
            path on five of six highlights on one leg and the surface bounce
            on the other.  The refinement is cheap once it is batched; there
            is nothing to save.
        n_refine: Newton steps.  All but the last run under ``no_grad``.
        tolerance: stop refining a path once it lands this close (m).  Defaults
            to a thousandth of the source-receiver separation.

    Returns ``(directions [P, 3], miss [P], signature [P])``.  Several pairs
    at once: :func:`find_eigenrays_batched`.
    """
    directions, residual, signature, _ = find_eigenrays_batched(
        scene, source.reshape(1, 3), receiver.reshape(1, 3),
        bracket_rays=bracket_rays, bracket_half_angle_deg=bracket_half_angle_deg,
        max_paths=max_paths, n_refine=n_refine, tolerance=tolerance,
        trace_kwargs=trace_kwargs)
    return directions, residual, signature


_BOUNCE_MODES = ("split", "specular", "coherent")


def _hashed_phase(identity: Tensor) -> Tensor:
    """A phase in [0, 2 pi) that is a fixed function of an integer identity.

    Not a random draw: the same path gets the same phase every call, however
    the endpoints move, so an image is a smooth function of a target's pose
    rather than a fresh speckle realisation at every step.
    """
    x = identity.to(torch.float64)
    frac = torch.frac(torch.sin(x * 12.9898 + 78.233) * 43758.5453)
    return (2.0 * math.pi * frac.abs())


def _split_by_coherence(energy: Tensor, coherence: Tensor, phase: Tensor,
                        identity: Tensor, bounce: str):
    """The arrivals a set of paths becomes, given each path's coherent fraction.

    ``energy`` and ``coherence`` are ``[P, B]``, the path's full energy and the
    fraction of it that is still specular after its bounces (the product of
    the Eckart factors, one for a flat boundary); ``phase`` ``[P]`` is the
    specular phase.  Returns ``(rows, amplitude, phase)``: which path each
    arrival comes from ``[Q]``, and its amplitude ``[Q, B]`` and phase ``[Q]``.

    ``bounce="specular"`` keeps every path whole at its specular phase, which
    is a flat-boundary model.  ``"coherent"`` keeps only the specular part,
    which is what a coherent field at a point needs.  ``"split"`` (the
    default) keeps both: the specular part at its phase, and the rest as a
    second arrival at a phase fixed by the path's identity -- the energy a
    rough boundary scatters out of the specular direction still reaches an
    imaging sonar, but with no phase relation to the direct path, so it has
    to add in power, not in amplitude.  Adding it in amplitude on the mean
    plane puts Lloyd's-mirror fringes on a target under a rough sea that the
    sea does not have: measured on a boat 1 m under a 0.09 m sea at 100 kHz,
    the echo swung 60x with a few metres of range and every gradient with it.
    """
    if bounce not in _BOUNCE_MODES:
        raise ValueError(f"bounce must be one of {_BOUNCE_MODES}, got {bounce!r}")
    tiny = torch.finfo(energy.dtype).tiny
    p = int(energy.shape[0])
    rows = torch.arange(p, device=energy.device)
    if bounce == "specular":
        return rows, energy.clamp_min(tiny).sqrt(), phase
    spec = energy * coherence
    if bounce == "coherent":
        return rows, spec.clamp_min(tiny).sqrt(), phase
    rest = energy * (1.0 - coherence)
    with torch.no_grad():
        has_spec = (coherence.detach().max(dim=1).values > 1e-12).nonzero().reshape(-1)
        has_rest = ((1.0 - coherence.detach()).max(dim=1).values > 1e-12).nonzero().reshape(-1)
    rows = torch.cat([rows[has_spec], rows[has_rest]])
    amplitude = torch.cat([spec[has_spec].clamp_min(tiny).sqrt(),
                           rest[has_rest].clamp_min(tiny).sqrt()], dim=0)
    scrambled = phase + _hashed_phase(identity).to(phase.dtype)
    phase = torch.cat([phase[has_spec], scrambled[has_rest]])
    return rows, amplitude, phase


def _plane_depth(boundary: HeightField, dtype, device) -> Tensor:
    """A boundary's mean depth as a differentiable scalar."""
    h = getattr(boundary, "heights", None)
    if h is not None:
        return h.to(dtype).mean()
    probe = torch.zeros(1, 2, dtype=dtype, device=device)
    return boundary.height(probe).reshape(())


def _into_channel(z: Tensor, depth: Tensor, margin: Tensor,
                  slack_surface: Tensor, slack_bottom: Tensor) -> Tensor:
    """Depths below the mean surface, moved just inside the channel when they
    lie a little outside it, and NaN when they lie further out than that."""
    above = z < margin
    below = z > depth - margin
    too_far = (z < -slack_surface) | (z > depth + slack_bottom)
    fixed = torch.where(above, margin.expand_as(z), z)
    fixed = torch.where(below, (depth - margin).expand_as(z), fixed)
    return torch.where(too_far, torch.full_like(z, float("nan")), fixed)


def image_arrivals_batched(scene, sources: Tensor, receivers: Tensor,
                           freqs_khz: Tensor, *, absorption=thorp_db_per_km,
                           spread_min_range: float = 1.0,
                           bounce: str = "split",
                           max_bounces: int | None = None) -> list[ArrivalSet]:
    """Every path between ``N`` pairs by the method of images, no tracing.

    The eigenray solve already runs on the boundaries' mean planes.  When the
    sound speed is constant as well, rays are straight lines between two
    parallel mirrors, and every path is a straight line to an image of the
    receiver: unfold the channel about its planes and the receiver appears
    at depths ``2 m H +/- r`` for every integer ``m``.  The planes the straight
    line crosses on its way to that image are the bounces, in order, at one
    grazing angle for all of them.  Launch direction, length, arrival
    direction, bounce count and grazing angle are all closed-form, and the
    spreading is exactly ``1/L^2``.

    So there is no fan, no Newton refinement and no ray-tube Jacobian: the
    whole solve is a few tensor expressions over ``N x (2 max_bounces + 1)``
    candidates, and it is differentiable in the endpoints, the sound speed
    and every boundary-loss parameter through the same loss modules the
    tracer calls at a bounce.  :func:`eigenray_arrivals_batched` picks this
    route by default whenever the profile is an :class:`IsoProfile`; a
    refracting profile needs the traced solve, which stays what it was.

    Three things differ from the traced solve, all deliberate.  Every path
    up to ``max_bounces`` (the scene's, unless given) is returned, not only
    those inside a bracket cone and a step budget.  The mean planes keep
    their gradient, so a learnable boundary sees d/d(mean depth) through the
    bounces.  And an endpoint a little outside the channel (within three RMS
    of that boundary's roughness, or one percent of the depth) is moved just
    inside it for the path geometry rather than given no paths: a waterline
    patch in a wave crest or a highlight on a seabed object where the bed
    dips below its mean is in the water locally.  Further out than that, a
    pair gets no paths at all.

    Returns a list of ``N`` :class:`~hydropt.beamform.ArrivalSet`, sorted by
    time, in the same shape :func:`eigenray_arrivals_batched` returns.
    """
    sources = sources.reshape(-1, 3)
    receivers = receivers.reshape(-1, 3)
    n_pairs = int(sources.shape[0])
    dtype, device = sources.dtype, sources.device
    freqs_khz = freqs_khz.to(dtype=dtype, device=device).reshape(-1)
    n_bands = int(freqs_khz.shape[0])
    if not isinstance(scene.field, IsoProfile):
        raise ValueError("the method of images needs a constant sound speed "
                         f"(IsoProfile); this scene's profile is "
                         f"{type(scene.field).__name__}.  Use method='trace'.")
    n_max = int(scene.max_bounces if max_bounces is None else max_bounces)

    # The mean planes, WITH their gradient: a specular path's length and
    # bounce angles depend on where the mean boundary is, so a learnable
    # height field gets d/d(mean) through every bounce -- uniform over its
    # nodes, which is the part of the boundary a coherent path can see.  (The
    # traced solve builds its planes under no_grad and so cannot pass this
    # on.)  What the roughness about the mean does to the path is a loss, not
    # a displacement, and reverberation is where its gradient lives.
    _, surface_rms = mean_boundary(scene.surface)
    _, bottom_rms = mean_boundary(scene.bottom)
    z_s = _plane_depth(scene.surface, dtype, device)
    z_b = _plane_depth(scene.bottom, dtype, device)
    depth = z_b - z_s
    if float(depth) <= 0.0:
        raise ValueError("the seabed must lie below the sea surface")
    c = scene.field(sources[:1].detach()).reshape(-1)[0].to(dtype)

    # Depths below the mean surface, and the receiver's images.  An endpoint
    # a little OUTSIDE the channel -- a waterline patch in a wave crest, a
    # highlight on a seabed object where the bed lies below its mean -- is in
    # the water locally and only outside the mean plane, so it is moved just
    # inside for the path geometry (an error of at most the roughness it sits
    # in) rather than given no paths at all.  "A little" is three RMS of that
    # boundary's roughness, or one percent of the depth for a flat one; any
    # further out and there are no paths.
    margin = 1e-3 * depth.detach()
    slack_s = torch.maximum(3.0 * surface_rms * torch.ones_like(margin), 1e-2 * depth.detach())
    slack_b = torch.maximum(3.0 * bottom_rms * torch.ones_like(margin), 1e-2 * depth.detach())
    s = _into_channel(sources[:, 2] - z_s, depth, margin, slack_s, slack_b)   # [N]
    r = _into_channel(receivers[:, 2] - z_s, depth, margin, slack_s, slack_b)
    m = torch.arange(-(n_max // 2 + 2), n_max // 2 + 3, device=device)  # [M]
    z_img = torch.stack([2.0 * m.to(dtype) * depth + r.view(-1, 1),
                         2.0 * m.to(dtype) * depth - r.view(-1, 1)],
                        dim=-1).reshape(n_pairs, -1)                     # [N, K]
    s_k = s.view(-1, 1).expand_as(z_img)

    # The planes crossed between the source depth and the image depth are the
    # bounces: plane j sits at depth j H, even j is the surface and its
    # images, odd j the seabed and its.
    with torch.no_grad():
        eps = 1e-9 * float(depth)
        lo = torch.minimum(s_k, z_img.detach()) / depth
        hi = torch.maximum(s_k, z_img.detach()) / depth
        j_lo = torch.ceil(lo + eps).long()
        j_hi = torch.floor(hi - eps).long()
        n_bounce = (j_hi - j_lo + 1).clamp_min(0)
        evens = (torch.div(j_hi, 2, rounding_mode="floor")
                 - torch.div(j_lo - 1, 2, rounding_mode="floor"))
        n_surf = torch.where(n_bounce > 0, evens, torch.zeros_like(evens))
        n_bot = n_bounce - n_surf
        inside = (torch.isfinite(s) & torch.isfinite(r)).view(-1, 1)
        keep = (n_bounce <= n_max) & inside.expand_as(n_bounce)
        pair_k = torch.arange(n_pairs, device=device).view(-1, 1).expand_as(n_bounce)
        sel = keep.reshape(-1).nonzero().reshape(-1)
    if sel.numel() == 0:
        return [_empty_arrivals(n_bands, dtype, device) for _ in range(n_pairs)]

    pair = pair_k.reshape(-1)[sel]
    n_surf = n_surf.reshape(-1)[sel].to(dtype)
    n_bot = n_bot.reshape(-1)[sel].to(dtype)
    n_bounce = n_bounce.reshape(-1)[sel]
    dz = (z_img - s_k).reshape(-1)[sel]                                  # [P]
    dxy = (receivers[:, :2] - sources[:, :2])[pair]                      # [P, 2]
    rho = dxy.norm(dim=-1)
    length = torch.sqrt(rho * rho + dz * dz)
    length_safe = length.clamp_min(1e-30)

    launch = torch.cat([dxy, dz.view(-1, 1)], dim=-1) / length_safe.view(-1, 1)
    flip = torch.where(n_bounce % 2 == 1, -torch.ones_like(dz), torch.ones_like(dz))
    direction = torch.cat([dxy, (flip * dz).view(-1, 1)], dim=-1) / length_safe.view(-1, 1)
    graze = torch.asin((dz.abs() / length_safe).clamp(0.0, 1.0))         # [P]

    db = n_surf * scene.surface_loss(graze) + n_bot * scene.bottom_loss(graze)
    phase = (n_surf * scene.surface_loss.reflection_phase(graze)
             + n_bot * scene.bottom_loss.reflection_phase(graze))
    time = length / c
    spread = 1.0 / length.clamp_min(spread_min_range) ** 2

    # The fraction of each path that is still specular after its bounces: the
    # Eckart factor per bounce, one for a flat boundary.  What the split does
    # with the rest is _split_by_coherence's business.
    if surface_rms > 0.0 or bottom_rms > 0.0:
        cs = coherent_reflection_loss_db(graze.view(-1, 1), surface_rms,
                                         freqs_khz.view(1, -1), float(c))
        cb = coherent_reflection_loss_db(graze.view(-1, 1), bottom_rms,
                                         freqs_khz.view(1, -1), float(c))
        coherence = 10.0 ** (-(n_surf.view(-1, 1) * cs + n_bot.view(-1, 1) * cb) / 10.0)
    else:
        coherence = torch.ones(int(sel.numel()), n_bands, dtype=dtype, device=device)

    alpha = absorption(freqs_khz).view(1, -1)
    energy = ((spread * 10.0 ** (-db / 10.0)).unsqueeze(1)
              * 10.0 ** (-(alpha * length.unsqueeze(1)) / 1.0e4))
    identity = pair * 4096 + sel % z_img.shape[1]          # which pair, which image
    rows, amplitude, phase = _split_by_coherence(energy, coherence, phase,
                                                 identity, bounce)
    time, direction, launch = time[rows], direction[rows], launch[rows]
    length, pair = length[rows], pair[rows]
    with torch.no_grad():
        alive = amplitude.detach().max(dim=1).values > 0.0
    distance = torch.zeros_like(length)

    out: list[ArrivalSet] = []
    for n in range(n_pairs):
        rows = ((pair == n) & alive).nonzero().reshape(-1)
        if rows.numel() == 0:
            out.append(_empty_arrivals(n_bands, dtype, device))
            continue
        rows = rows[time.detach()[rows].argsort()]
        out.append(ArrivalSet(time=time[rows], amplitude=amplitude[rows],
                              direction=direction[rows], phase=phase[rows],
                              distance=distance[rows], path_length=length[rows],
                              launch_direction=launch[rows]))
    return out


def eigenray_arrivals_batched(scene, sources: Tensor, receivers: Tensor,
                              freqs_khz: Tensor, *, absorption=thorp_db_per_km,
                              spread_min_range: float = 1.0,
                              accept: float | None = None,
                              bounce: str = "split",
                              method: str = "auto",
                              **kwargs) -> list[ArrivalSet]:
    """:func:`eigenray_arrivals` for ``N`` pairs: one solve, ``N`` arrival sets.

    ``sources`` and ``receivers`` are ``[N, 3]``, paired row by row.  Returns a
    list of ``N`` :class:`~hydropt.beamform.ArrivalSet`, each sorted by time
    and possibly empty, differentiable in its own pair's endpoints and in the
    scene.

    ``method``: ``"images"`` is :func:`image_arrivals_batched`, closed-form
    and exact for a constant sound speed; ``"trace"`` brackets with a fan and
    refines with Newton on traced rays, and is what a refracting profile
    needs; ``"auto"`` (the default) takes the images whenever the profile is
    an :class:`IsoProfile`.  Only the traced solve reads the bracket
    arguments in ``kwargs``.
    """
    if method not in ("auto", "images", "trace"):
        raise ValueError(f"method must be 'auto', 'images' or 'trace', got {method!r}")
    if method == "images" or (method == "auto" and isinstance(scene.field, IsoProfile)):
        return image_arrivals_batched(
            scene, sources, receivers, freqs_khz, absorption=absorption,
            spread_min_range=spread_min_range, bounce=bounce,
            max_bounces=kwargs.get("max_bounces"))
    sources = sources.reshape(-1, 3)
    receivers = receivers.reshape(-1, 3)
    n_pairs = int(sources.shape[0])
    dtype, device = sources.dtype, sources.device
    freqs_khz = freqs_khz.to(dtype=dtype, device=device)
    n_bands = int(freqs_khz.shape[0])
    span = (receivers - sources).detach().norm(dim=-1)                  # [N]
    keep_within = (1e-2 * span if accept is None
                   else torch.full_like(span, float(accept)))

    directions, residual, _, pair = find_eigenrays_batched(
        scene, sources, receivers,
        **{k: v for k, v in kwargs.items() if k != "max_bounces"})
    good = residual <= keep_within[pair]
    if int(good.sum()) == 0:
        return [_empty_arrivals(n_bands, dtype, device) for _ in range(n_pairs)]
    directions, pair = directions[good], pair[good]
    p = int(directions.shape[0])
    src_p, rcv_p = sources[pair], receivers[pair]

    tkw = dict(kwargs.get("trace_kwargs") or {})
    smooth = _SmoothedScene(scene, src_p)
    result = trace(smooth, directions, **tkw)
    miss, step, frac = _closest(result, rcv_p)
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

    # Ray-tube divergence.  The tube's cross-section at the receiver per unit
    # solid angle at the source is |det d(transverse position)/d(launch
    # angle)|, and intensity is its reciprocal.  Straight ray, homogeneous
    # medium: the transverse position grows as L times the angle, so |det| =
    # L^2 and this is 1/L^2 -- the spherical spreading it has to reduce to.
    #
    # In the PATH'S OWN frames, not the refinement's.  The Newton solve works
    # in offsets on the source-receiver chord's tangent plane, which is a fine
    # parametrisation to converge in and a wrong one to measure a tube in: an
    # offset there is an angle only for a path that leaves along the chord.
    # A bottom bounce leaving 30 degrees off it moves by cos^2 of that per
    # unit offset, and its miss, read in the chord's transverse plane, is
    # foreshortened by another cosine at arrival.  Measured on a flat-bottom
    # bounce with the chord-plane Jacobian: +2.7 dB over 1/L^2 one way and
    # +3.4 dB the other, on the same path.  So: bump the converged launch
    # direction by a true angle in a frame perpendicular to IT, and read the
    # miss in a frame perpendicular to the ARRIVING ray, which the miss vector
    # already lies in.  Both legs of a path then agree, and both read 1/L^2.
    with torch.no_grad():
        d0 = directions.detach()
        _, f1, f2 = _frames(d0)                       # perpendicular to launch
        _, g1, g2 = _frames(direction.detach())       # perpendicular to arrival
        h = 1e-4                                      # radians
        cols = []
        for f in (f1, f2):
            out = []
            for sign in (1.0, -1.0):
                d2 = d0 + sign * h * f
                d2 = d2 / d2.norm(dim=-1, keepdim=True)
                r2 = trace(_SmoothedScene(scene, src_p.detach()), d2, **tkw)
                m2, _, _ = _closest(r2, rcv_p)
                out.append(torch.stack([(m2 * g1).sum(-1), (m2 * g2).sum(-1)],
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

    # The fraction of each path still specular after its bounces (the Eckart
    # factors along it), evaluated `up_to_step` because the path ends at the
    # receiver and the trace does not: a ray that arrives early keeps flying,
    # and a bounce it makes out there is no part of the path.  Counting it
    # once charged the direct path a surface bounce it never made, and the
    # boat left the image.  What becomes of the non-specular remainder is
    # _split_by_coherence's business; by default it keeps its energy at a
    # phase of its own, because a rough boundary scatters it, it does not
    # absorb it -- and the ghosts a shallow-water operator expects are made of
    # exactly that energy.
    coherence = roughness_weights(
        result, freqs_khz, surface_rms=smooth.surface_rms,
        bottom_rms=smooth.bottom_rms, surface=smooth.surface,
        bottom=smooth.bottom, up_to_step=step, sound_speed=float(scene.field(
            src_p[:1].detach()).reshape(-1)[0]))

    alpha = absorption(freqs_khz).view(1, -1)
    energy = ((spread * 10.0 ** (-db / 10.0)).unsqueeze(1)
              * 10.0 ** (-(alpha * path_length.unsqueeze(1)) / 1.0e4))
    # A path's identity for the hashed phase: its pair, its bounce signature
    # and which way it left, which tells a surface-then-bottom path from a
    # bottom-then-surface one.
    sig = _bounce_signature(result, step)
    identity = pair * 4096 + sig * 2 + (launch[:, 2].detach() > 0).long()
    rows, amplitude, phase = _split_by_coherence(energy, coherence, phase,
                                                 identity, bounce)
    time, direction, launch = time[rows], direction[rows], launch[rows]
    path_length, miss, pair = path_length[rows], miss[rows], pair[rows]
    # A path the roughness has annihilated is not an arrival, and carrying it
    # at exactly zero is worse than dropping it: the derivative of sqrt at zero
    # is infinite, so a single underflowed path turns the whole gradient into
    # NaN.  That is not hypothetical -- the Eckart factor for a surface bounce
    # at 120 kHz over a 0.09 m sea underflows in float64, and it took the
    # gradient to the target's own pose with it.
    with torch.no_grad():
        alive = amplitude.detach().max(dim=1).values > 0.0
    distance = miss.norm(dim=-1).detach()

    out: list[ArrivalSet] = []
    for n in range(n_pairs):
        keep = ((pair == n) & alive).nonzero().reshape(-1)
        if keep.numel() == 0:
            out.append(_empty_arrivals(n_bands, dtype, device))
            continue
        keep = keep[time.detach()[keep].argsort()]
        out.append(ArrivalSet(time=time[keep], amplitude=amplitude[keep],
                              direction=direction[keep], phase=phase[keep],
                              distance=distance[keep],
                              path_length=path_length[keep],
                              launch_direction=launch[keep]))
    return out


def eigenray_arrivals(scene, source: Tensor, receiver: Tensor,
                      freqs_khz: Tensor, *, absorption=thorp_db_per_km,
                      spread_min_range: float = 1.0,
                      accept: float | None = None,
                      bounce: str = "split",
                      method: str = "auto",
                      **kwargs) -> ArrivalSet:
    """One arrival per path from ``source`` to ``receiver``, no splat.

    The amplitude carries the ray tube's own divergence rather than an assumed
    ``1/L^2``: the 2x2 Jacobian ``d(transverse miss)/d(launch angle)``, taken
    in frames perpendicular to the path's own launch and arrival directions,
    IS the tube's cross-section per unit solid angle, so its determinant is
    the spreading.  For any ray in a homogeneous medium, bounces included, it
    comes out as the unfolded ``L^2``, which is the check in the tests.

    Args:
        bounce: what a bounce off a rough boundary becomes.  ``"split"`` (the
            default): its specular part, weighted by the Eckart factor, plus
            the rest of its energy as a second arrival at a phase fixed by the
            path's identity, so it adds in power with the direct path as
            incoherently scattered energy does.  ``"specular"``: the whole
            path at its specular phase, a flat-boundary model.  ``"coherent"``:
            the specular part alone, for a coherent field at a point.  A flat
            boundary makes the three identical.
        accept: discard paths that still miss by more than this (m).  Defaults
            to a hundredth of the source-receiver separation; a path that will
            not converge is one the bracket found and the refinement could not
            close, and reporting it as an arrival would be inventing one.

    Returns an :class:`~hydropt.beamform.ArrivalSet` shaped exactly like
    :func:`hydropt.beamform.extract_arrivals`, so it drops in wherever that
    does.  Several pairs at once: :func:`eigenray_arrivals_batched`.
    """
    return eigenray_arrivals_batched(
        scene, source.reshape(1, 3), receiver.reshape(1, 3), freqs_khz,
        absorption=absorption, spread_min_range=spread_min_range,
        accept=accept, bounce=bounce, method=method, **kwargs)[0]
