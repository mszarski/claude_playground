r"""Gaussian beams: finite amplitude at caustics, and no arbitrary floor.

:mod:`hydropt.spreading` gives the true geometric ray tube, but a geometric tube
collapses to zero area at a caustic and at the source itself, so the intensity
there is infinite and has to be clamped.  ``min_jacobian`` is an admission, not
a model.

A Gaussian beam replaces the infinitely thin ray with a beam of finite
transverse width.  The beam parameter is complex, so it never passes through
zero, and the amplitude stays finite everywhere -- through caustics, through the
source, through everything.

Dynamic ray tracing without the dynamic ray equations
-----------------------------------------------------
The textbook route integrates a second ODE system alongside the ray,

``dq/ds = c p``,  ``dp/ds = -(c_nn / c^2) q``

which needs ``c_nn``, the second derivative of sound speed transverse to the
ray.  That is an awkward thing to ask hydropt's fields for: a piecewise-linear
profile has ``d^2c/dz^2`` equal to a train of delta functions at its knots, and
a trilinear grid has it identically zero inside every cell.  Dynamic ray tracing
through either is ill-defined.

But ``q`` and ``p`` describe how a *paraxial perturbation* of the initial
conditions evolves, and hydropt can already differentiate a traced ray with
respect to its initial conditions exactly, with forward-mode AD.  The two
fundamental solutions are simply two perturbations:

``Q1`` -- perturb the **launch angle**, leaving the source where it is.  The ray
    starts at the same point, so ``Q1(0) = 0``: this is the point-source
    solution, and ``det Q1`` is exactly the geometric ray tube.
``Q2`` -- perturb the **source position** transversally, leaving the launch
    direction alone.  ``Q2(0) = I``, ``P2(0) = 0``.

Both come out of :func:`torch.func.jvp` on the ordinary tracer, so no second
derivative of ``c`` is ever required and the result is exact rather than
second-order.  The Gaussian beam is then the complex combination

``Q = Q1 + i beta Q2``

whose determinant cannot vanish where ``det Q1`` does, because the imaginary
part is still there.  ``beta`` has units of length and is the beam's transverse
width scale at the source.

In a homogeneous medium this is checkable by hand: ``Q1 = s I`` and ``Q2 = I``,
so ``det Q = (s + i beta)^2`` and the spreading ``1/|det Q|`` is
``1/(s^2 + beta^2)`` -- indistinguishable from ``1/s^2`` once ``s >> beta``, and
finite rather than infinite at the source.  Both limits are tested.

Choosing beta
-------------
``beta`` is a physical width, not a fudge factor, but the theory does not pin it
down: Gaussian-beam codes differ in the choice, and it is the price of the
method.  Too small and the beam is a geometric ray again, with the caustic
singularity back; too large and the field is over-smoothed and arrivals merge.
A wavelength or a few is the usual starting point, and
:func:`suggest_beam_width` implements that.

Cost
----
Six traces -- one for the path, two launch-angle tangents, three source-position
tangents -- but *not* six times the wall time.  Five of them are forward-mode
dual traces, which neither fuse nor checkpoint, and measured on 120 rays x 1500
steps they come to about **100x** a plain traced bundle (112 s against 1.1 s).
That is the real price of exactness.  The geometric tube in
:func:`hydropt.spreading.ray_tube` costs one trace plus a neighbour difference
and is second-order accurate, which is the right default for most work; reach
for beams when the amplitude at a caustic is the thing you actually need.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import torch
from torch import Tensor

from .launch import directions_from_angles
from .tracer import TraceResult, trace

__all__ = ["GaussianBeams", "gaussian_beams", "suggest_beam_width",
           "beam_sum_kwargs"]


class GaussianBeams(NamedTuple):
    """Gaussian-beam geometry along a bundle.  Arrays are ``[rays, vertices]``."""

    spreading: Tensor  # 1/|det Q|, a drop-in for the 1/s^2 term
    det_q: Tensor  # complex beam determinant; never zero
    geometric: Tensor  # 1/|det Q1|, the geometric tube, for comparison
    width: Tensor  # transverse 1/e beam half-width (m)
    caustics: Tensor  # KMAH index, from sign changes of the *geometric* tube
    result: TraceResult  # the traced bundle these were built from

    @property
    def kmah_phase(self) -> Tensor:
        return -0.5 * math.pi * self.caustics


def suggest_beam_width(freq_khz: float, *, wavelengths: float = 3.0,
                       sound_speed: float = 1500.0) -> float:
    """A starting ``beta``: a few wavelengths at the given frequency.

    Not a derived optimum -- Gaussian-beam codes differ on the choice, and the
    field does depend on it.  It is a defensible default, and the width is
    reported back so the effect of changing it can be seen.
    """
    return wavelengths * sound_speed / (freq_khz * 1.0e3)


class _ShiftedSource:
    """Scene view whose source is displaced, for the source-position tangents."""

    def __init__(self, scene, offset: Tensor) -> None:
        object.__setattr__(self, "_scene", scene)
        object.__setattr__(self, "_offset", offset)

    def __getattr__(self, name: str):
        return getattr(self._scene, name)

    def source_position(self) -> Tensor:
        return self._scene.source_position() + self._offset


def _transverse_frame(tangent: Tensor) -> tuple[Tensor, Tensor]:
    """Any orthonormal pair perpendicular to ``tangent``.

    The choice does not matter.  Rotating the frame multiplies both ``Q1`` and
    ``Q2`` on the left by the same orthogonal matrix, so ``|det Q|`` -- the only
    thing used downstream -- is unchanged.
    """
    ref = torch.zeros_like(tangent)
    # Pick whichever axis is least parallel to the tangent, per vertex.
    least = tangent.abs().argmin(dim=-1, keepdim=True)
    ref.scatter_(-1, least, 1.0)
    u = torch.cross(tangent, ref, dim=-1)
    u = u / u.norm(dim=-1, keepdim=True).clamp_min(1e-30)
    v = torch.cross(tangent, u, dim=-1)
    return u, v


def gaussian_beams(
    scene,
    elev: Tensor,
    azim: Tensor,
    *,
    beam_width: float,
    freq_khz: float,
    trace_kwargs: dict | None = None,
) -> GaussianBeams:
    """Trace a bundle as Gaussian beams.

    Args:
        scene: the scene to trace through.
        elev, azim: per-ray launch angles in radians, ``[R]`` each.  Any
            sampling works -- unlike the neighbour-difference ray tube, this
            needs no structured fan.  Broadcastable shapes are accepted, so the
            axes from :func:`hydropt.launch.structured_fan` can be passed as
            ``elev[:, None], azim[None, :]`` and give the same elevation-major
            ray order as its ``directions``.
        beam_width: ``beta``, the transverse width scale at the source, in
            metres.  See :func:`suggest_beam_width`.
        freq_khz: frequency, needed only for the reported beam ``width``.  The
            spreading itself depends on frequency solely through ``beam_width``.
        trace_kwargs: forwarded to :func:`hydropt.tracer.trace`.

    Returns:
        :class:`GaussianBeams`.  ``spreading`` is a drop-in replacement for the
        ``1/s^2`` term in :func:`hydropt.receiver.splat_etc`, finite everywhere
        and needing no clamp.
    """
    kw = trace_kwargs or {}
    elev, azim = torch.broadcast_tensors(elev, azim)
    elev, azim = elev.reshape(-1), azim.reshape(-1)
    beta = float(beam_width)

    def by_angles(e: Tensor, a: Tensor) -> Tensor:
        return trace(scene, directions_from_angles(e, a), **kw).pos

    _, dr_de = torch.func.jvp(lambda e: by_angles(e, azim), (elev,),
                              (torch.ones_like(elev),))
    _, dr_da = torch.func.jvp(lambda a: by_angles(elev, a), (azim,),
                              (torch.ones_like(azim),))

    dirs = directions_from_angles(elev, azim)
    zero3 = torch.zeros(3, dtype=elev.dtype, device=elev.device)

    def by_source(offset: Tensor) -> Tensor:
        return trace(_ShiftedSource(scene, offset), dirs, **kw).pos

    source_jac = []
    for axis in range(3):
        tangent = torch.zeros_like(zero3)
        tangent[axis] = 1.0
        _, d = torch.func.jvp(by_source, (zero3,), (tangent,))
        source_jac.append(d)
    source_jac = torch.stack(source_jac, dim=-1)  # [R, V, 3(out), 3(in)]

    result = trace(scene, dirs, **kw)
    fwd = result.pos[:, 1:] - result.pos[:, :-1]
    tangent = torch.cat((fwd, fwd[:, -1:]), dim=1)
    tangent = tangent / tangent.norm(dim=-1, keepdim=True).clamp_min(1e-30)
    u, v = _transverse_frame(tangent)

    # Ray-centred normalisation: the launch-angle perturbations are rescaled to
    # unit change of *direction*, which makes the input bases of Q1 and Q2 the
    # same transverse frame at the source -- without that the two cannot be
    # added at all.  |d(dir)/d(elev)| = 1 and |d(dir)/d(azim)| = cos(elev).
    cos_e = torch.cos(elev).reshape(-1, 1, 1).clamp_min(1e-9)
    col_e, col_a = dr_de, dr_da / cos_e

    def project(col: Tensor) -> tuple[Tensor, Tensor]:
        return (col * u).sum(-1), (col * v).sum(-1)

    q1_uu, q1_vu = project(col_e)
    q1_uv, q1_vv = project(col_a)
    q1 = torch.stack((torch.stack((q1_uu, q1_uv), -1),
                      torch.stack((q1_vu, q1_vv), -1)), dim=-2)  # [R, V, 2, 2]

    # Q2 in the same frame: the source-position Jacobian projected onto the
    # transverse frame at both ends.
    u0, v0 = u[:, :1], v[:, :1]  # frame at the source, broadcast along vertices
    def project_source(out_vec: Tensor, in_vec: Tensor) -> Tensor:
        tmp = (source_jac * in_vec.unsqueeze(-2)).sum(-1)  # [R, V, 3]
        return (tmp * out_vec).sum(-1)

    q2 = torch.stack((
        torch.stack((project_source(u, u0), project_source(u, v0)), -1),
        torch.stack((project_source(v, u0), project_source(v, v0)), -1),
    ), dim=-2)

    q = torch.complex(q1, beta * q2)
    det = q[..., 0, 0] * q[..., 1, 1] - q[..., 0, 1] * q[..., 1, 0]
    det_geo = q1[..., 0, 0] * q1[..., 1, 1] - q1[..., 0, 1] * q1[..., 1, 0]

    magnitude = det.abs().clamp_min(1e-30)
    spreading = 1.0 / magnitude
    geometric = 1.0 / det_geo.abs().clamp_min(1e-30)

    # Transverse beam width.  The beam's amplitude falls as
    # exp(-n^2 / 2W^2) with W = 1 / sqrt(omega |Im(P Q^-1)|), where P is the
    # second dynamic quantity, P = (1/c) dQ/ds.  Q is known at every vertex, so
    # P comes from differencing it rather than from another ODE.
    ds = (result.arclen[:, 1:] - result.arclen[:, :-1]).clamp_min(1e-9)
    dq = (q[:, 1:] - q[:, :-1]) / ds.unsqueeze(-1).unsqueeze(-1)
    dq = torch.cat((dq, dq[:, -1:]), dim=1)
    c = scene.field(result.pos).clamp_min(1e-9)
    p_dyn = dq / c.unsqueeze(-1).unsqueeze(-1)

    # M = P Q^-1 for a 2x2, written out to avoid a batched complex solve.
    a11, a12 = q[..., 0, 0], q[..., 0, 1]
    a21, a22 = q[..., 1, 0], q[..., 1, 1]
    inv_det = 1.0 / torch.where(det.abs() > 1e-30, det, torch.full_like(det, 1e-30))
    i11, i12 = a22 * inv_det, -a12 * inv_det
    i21, i22 = -a21 * inv_det, a11 * inv_det
    b11, b12 = p_dyn[..., 0, 0], p_dyn[..., 0, 1]
    b21, b22 = p_dyn[..., 1, 0], p_dyn[..., 1, 1]
    m11 = b11 * i11 + b12 * i21
    m12 = b11 * i12 + b12 * i22
    m21 = b21 * i11 + b22 * i21
    m22 = b21 * i12 + b22 * i22
    det_im = (m11.imag * m22.imag - m12.imag * m21.imag).abs()
    omega = 2.0 * math.pi * freq_khz * 1.0e3
    width = 1.0 / (omega * det_im.clamp_min(1e-300).sqrt()).clamp_min(1e-300).sqrt()

    sign = torch.sign(det_geo.detach())
    alive = result.alive > 0
    flip = (sign[:, 1:] * sign[:, :-1] < 0) & alive[:, 1:] & alive[:, :-1]
    caustics = torch.cat((torch.zeros_like(flip[:, :1]), flip), dim=1)
    caustics = caustics.to(spreading.dtype).cumsum(dim=1)

    return GaussianBeams(spreading=spreading, det_q=det, geometric=geometric,
                         width=width, caustics=caustics, result=result)


def beam_sum_kwargs(beams: GaussianBeams) -> dict:
    r"""Render ``beams`` as a **Gaussian beam sum**, not as an arbitrary aperture.

    Returns the ``spreading`` and ``sigma_d`` to hand
    :func:`hydropt.receiver.splat_etc` or
    :func:`hydropt.beamform.extract_arrivals`::

        beams = gaussian_beams(scene, elev, azim, beam_width=beta, freq_khz=f)
        etc = splat_etc(beams.result, receivers, grid, freqs,
                        ray_weights=solid_angle, sigma_t=..., **beam_sum_kwargs(beams))

    Why this is the fix rather than a convenience.  A splat sums
    ``amplitude * exp(-n^2 / W^2)`` over rays, and for a dense fan the number of
    rays landing within ``W`` of a point falls as ``1/s^2``.  So the sum comes out
    as ``amplitude * W^2 / s^2``, and reproducing free-field spreading needs
    ``amplitude * W^2`` to be **constant**.  For Gaussian beams it is: the
    amplitude is ``1/|det Q| = 1/(s^2 + beta^2)`` in a homogeneous medium and the
    width obeys ``W^2 = c (s^2 + beta^2) / (omega beta)``, whose product is
    ``c / (omega beta)`` at every range.  A *fixed* ``sigma_d`` breaks exactly that
    cancellation, which is the ``1/R^4`` defect the README describes.

    Measured: with the beam width as the kernel width the energy exponent is
    2.000 and the calibration constant is flat to 1.001x -- and, the point of the
    whole construction, **independent of beta** from 60 m to 1000 m, because beta
    parameterises the decomposition and not the physics.

    The ``sqrt(2)`` converts between the two Gaussian conventions: the renderer
    weights by ``exp(-d^2 / 2 sigma_d^2)`` and a beam's profile is
    ``exp(-n^2 / W^2)``.

    ``ray_weights`` is still yours to supply -- the per-ray solid angle
    ``cos(e) de da`` -- because only the caller knows how the fan was built.
    """
    return {"spreading": beams.spreading,
            "sigma_d": beams.width / math.sqrt(2.0)}
