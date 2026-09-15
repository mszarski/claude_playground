"""Fixed-step RK4 ray tracer for the 3-D acoustic eikonal equations.

The ray system, parameterised by arclength ``s`` (Jensen et al.,
*Computational Ocean Acoustics*, sec. 3.2), is

.. math::
    \\frac{d\\mathbf{x}}{ds} = c\\,\\boldsymbol{\\xi}, \\qquad
    \\frac{d\\boldsymbol{\\xi}}{ds} = -\\frac{\\nabla c}{c^2}, \\qquad
    \\frac{d\\tau}{ds} = \\frac{1}{c}

with the slowness vector ``xi = (xi, eta, zeta)`` satisfying ``|xi| = 1/c``.
That constraint is a conserved quantity of the exact flow but drifts under any
finite-order integrator, so it is re-imposed after every step -- this is what
keeps long-range runs stable at practical step sizes.

Every operation is batched over rays; the only Python-level loops are over
*steps* (inherent to an ODE solve) and over *checkpoint chunks*.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, NamedTuple

import torch
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from .boundaries import HeightField, find_crossing, grazing_angle, reflect

if TYPE_CHECKING:  # pragma: no cover
    from .scene import Scene

__all__ = ["RayState", "TraceResult", "BounceEvents", "bounce_events",
           "rk4_step", "trace"]


class RayState(NamedTuple):
    """Per-ray integration state.  All tensors have leading dimension ``[rays]``."""

    pos: Tensor  # [R, 3] position (m)
    slow: Tensor  # [R, 3] slowness vector, |slow| = 1/c
    tau: Tensor  # [R] travel time (s)
    arclen: Tensor  # [R] path length (m)
    refl_db: Tensor  # [R] accumulated reflection loss (dB)
    refl_phase: Tensor  # [R] accumulated reflection phase (rad), for coherent work
    bounce_grazing: Tensor  # [R] signed grazing angle of a bounce taken this step
    alive: Tensor  # [R] 1.0 while the ray is still propagating
    n_surface: Tensor  # [R] surface bounce count (float, for batched maths)
    n_bottom: Tensor  # [R] bottom bounce count


class TraceResult(NamedTuple):
    """Sampled ray paths.  ``S = n_steps``, so vertex arrays are ``S + 1`` long."""

    pos: Tensor  # [R, S+1, 3]
    tau: Tensor  # [R, S+1]
    arclen: Tensor  # [R, S+1]
    refl_db: Tensor  # [R, S+1]
    refl_phase: Tensor  # [R, S+1]
    bounce_grazing: Tensor  # [R, S+1] see bounce_events()
    alive: Tensor  # [R, S+1]
    n_surface: Tensor  # [R]
    n_bottom: Tensor  # [R]

    @property
    def n_rays(self) -> int:
        return int(self.pos.shape[0])

    @property
    def n_vertices(self) -> int:
        return int(self.pos.shape[1])

    def bounces(self) -> Tensor:
        """Total bounce count per ray, ``[R]`` (integer dtype)."""
        return (self.n_surface + self.n_bottom).round().long()


class BounceEvents(NamedTuple):
    """Every boundary reflection in a traced bundle, as a flat sparse list.

    Bounces are rare compared with steps -- a handful per ray against thousands
    of vertices -- so they are decoded on demand rather than stored densely.
    """

    ray: Tensor  # [E] index of the ray that bounced
    step: Tensor  # [E] vertex index at which it bounced
    position: Tensor  # [E, 3]
    time: Tensor  # [E] travel time to the bounce (s)
    arclen: Tensor  # [E] path length to the bounce (m)
    refl_db: Tensor  # [E] reflection loss accumulated *including* this bounce
    refl_phase: Tensor  # [E] reflection phase accumulated including this bounce
    refl_db_incident: Tensor  # [E] loss accumulated on the way *to* this bounce
    refl_phase_incident: Tensor  # [E] phase accumulated on the way to this bounce
    grazing: Tensor  # [E] grazing angle (rad, always positive)
    is_bottom: Tensor  # [E] True for a seabed bounce, False for the surface

    @property
    def count(self) -> int:
        return int(self.ray.shape[0])


def bounce_events(
    result: TraceResult,
    *,
    surface: HeightField | None = None,
    bottom: HeightField | None = None,
    bottom_only: bool = False,
    min_grazing: float = 1e-9,
) -> BounceEvents:
    """Decode the boundary reflections recorded in a :class:`TraceResult`.

    ``bounce_grazing`` stores a signed grazing angle at the vertex following
    each reflection: positive for the seabed, negative for the surface, zero
    where nothing happened.  A grazing angle of exactly zero would be
    ambiguous, but it also means the ray ran parallel to the boundary and never
    crossed it, so ``min_grazing`` discards that degenerate case.

    The recorded vertex sits *past* the reflection, because the tracer takes the
    remainder of the step along the mirrored direction before storing anything.
    Left uncorrected that biases every bounce range by up to one step length --
    two metres at a 2 m step, which for reverberation is a systematic range
    error, not noise.  Passing ``surface`` and ``bottom`` refines each event
    back onto the boundary by walking against the outgoing direction until it
    crosses, which costs no extra storage in the traced path and reuses the same
    differentiable root-find as the tracer itself.

    Args:
        surface, bottom: boundaries to refine against.  Both must be given;
            without them the events are returned at their recorded vertices.
        bottom_only: keep only seabed bounces.
        min_grazing: reflections shallower than this are dropped.
    """
    g = result.bounce_grazing
    mask = g.abs() > min_grazing
    if bottom_only:
        mask = mask & (g > 0)
    ray, step = mask.nonzero(as_tuple=True)
    signed = g[ray, step]
    is_bottom = signed > 0

    position = result.pos[ray, step]
    time = result.tau[ray, step]
    arclen = result.arclen[ray, step]

    if surface is not None and bottom is not None and ray.numel() > 0:
        n_vert = result.pos.shape[1]
        nxt = (step + 1).clamp_max(n_vert - 1)
        forward = result.pos[ray, nxt] - position
        # At the final vertex there is no next one; fall back to the incoming
        # segment, which after reflection points the same way.
        degenerate = forward.norm(dim=-1) < 1e-12
        if bool(degenerate.any()):
            prev = (step - 1).clamp_min(0)
            forward = torch.where(degenerate.unsqueeze(-1),
                                  position - result.pos[ray, prev], forward)
        d_out = forward / forward.norm(dim=-1, keepdim=True).clamp_min(1e-30)

        # One step back along -d_out brackets the boundary we just left.
        span = (arclen - result.arclen[ray, (step - 1).clamp_min(0)]).clamp_min(1e-9)
        back = position - span.unsqueeze(-1) * d_out
        frac_s = find_crossing(position, back, surface)
        frac_b = find_crossing(position, back, bottom)
        frac = torch.where(is_bottom, frac_b, frac_s)
        walk = frac * span

        position = position - walk.unsqueeze(-1) * d_out
        arclen = arclen - walk
        # Convert the walked distance to time at the local sound speed implied
        # by this step, rather than assuming a reference speed.
        prev = (step - 1).clamp_min(0)
        dt_ds = ((time - result.tau[ray, prev])
                 / (result.arclen[ray, step] - result.arclen[ray, prev]).clamp_min(1e-9))
        time = time - walk * dt_ds

    return BounceEvents(
        ray=ray, step=step,
        position=position,
        time=time,
        arclen=arclen,
        refl_db=result.refl_db[ray, step],
        refl_phase=result.refl_phase[ray, step],
        refl_db_incident=result.refl_db[ray, (step - 1).clamp_min(0)],
        refl_phase_incident=result.refl_phase[ray, (step - 1).clamp_min(0)],
        grazing=signed.abs(),
        is_bottom=is_bottom,
    )


def _renormalise(slow: Tensor, c: Tensor) -> Tensor:
    """Rescale ``slow`` to magnitude ``1/c`` without changing its direction."""
    mag = slow.norm(dim=-1, keepdim=True).clamp_min(1e-30)
    return slow * (1.0 / (c.unsqueeze(-1) * mag))


def rk4_step(
    field,
    pos: Tensor,
    slow: Tensor,
    tau: Tensor,
    h: float | Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """One classical RK4 step of arclength ``h``.  Pure function of its inputs.

    Returns ``(pos, slow, tau)`` *before* boundary handling and before the
    slowness renormalisation, so callers can inspect the raw ODE step.
    """

    def deriv(p: Tensor, q: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        c, gc = field.c_and_grad(p)
        dp = c.unsqueeze(-1) * q
        dq = -gc / (c * c).unsqueeze(-1)
        return dp, dq, 1.0 / c

    dp1, dq1, dt1 = deriv(pos, slow)
    dp2, dq2, dt2 = deriv(pos + (0.5 * h) * dp1, slow + (0.5 * h) * dq1)
    dp3, dq3, dt3 = deriv(pos + (0.5 * h) * dp2, slow + (0.5 * h) * dq2)
    dp4, dq4, dt4 = deriv(pos + h * dp3, slow + h * dq3)

    w = h / 6.0
    pos_n = pos + w * (dp1 + 2.0 * dp2 + 2.0 * dp3 + dp4)
    slow_n = slow + w * (dq1 + 2.0 * dq2 + 2.0 * dq3 + dq4)
    tau_n = tau + w * (dt1 + 2.0 * dt2 + 2.0 * dt3 + dt4)
    return pos_n, slow_n, tau_n


def _handle_boundaries(
    field,
    surface: HeightField,
    bottom: HeightField,
    surface_loss,
    bottom_loss,
    state: RayState,
    pos_n: Tensor,
    slow_n: Tensor,
    tau_n: Tensor,
    h: float,
    *,
    n_bisect: int,
    n_newton: int,
    min_advance: float,
) -> RayState:
    """Detect, locate and apply at most one boundary reflection per step.

    The reflection is resolved on the straight chord between the step endpoints,
    which is an ``O(h^2)`` approximation to the curved ray -- the same order as
    the surrounding integrator, so it does not dominate the error.  After
    reflecting, the remaining fraction of the step is taken as a straight
    advance along the mirrored direction.

    When a step manages to cross *both* boundaries (a very thin water column, or
    an over-long step), the earlier crossing wins and the other is deferred to
    the next step.
    """
    pos_o, slow_o, tau_o = state.pos, state.slow, state.tau

    g_surf_o = pos_o[..., 2] - surface.height(pos_o[..., :2])
    g_surf_n = pos_n[..., 2] - surface.height(pos_n[..., :2])
    g_bot_o = pos_o[..., 2] - bottom.height(pos_o[..., :2])
    g_bot_n = pos_n[..., 2] - bottom.height(pos_n[..., :2])

    hit_surf = (g_surf_o >= 0) & (g_surf_n < 0)  # travelled up through the surface
    hit_bot = (g_bot_o <= 0) & (g_bot_n > 0)  # travelled down through the seabed
    active = (hit_surf | hit_bot) & (state.alive > 0)

    zero = torch.zeros_like(state.refl_db)
    if not bool(active.any()):
        return RayState(pos_n, slow_n, tau_n, state.arclen + h, state.refl_db,
                        state.refl_phase, zero,
                        state.alive, state.n_surface, state.n_bottom)

    # Everything below runs on the bouncing rays only.  Running it on the whole
    # batch costs nothing in a deep-water scene, where whole chunks of steps
    # have no reflection at all and the early return above fires -- but it is
    # ruinous in the shallow, high-bounce scenes active sonar cares about.  In a
    # 30 m channel at a 0.25 m step *some* ray hits on nearly every step, so the
    # two bisection searches would run across every ray in the fan whether or
    # not it was anywhere near a boundary: measured on 30,000 rays with roughly
    # 100 bouncing, that is 35x more work per search than the rays that need it.
    idx = active.nonzero(as_tuple=True)[0]
    po, pn = pos_o[idx], pos_n[idx]
    so, sn = slow_o[idx], slow_n[idx]
    to, tn = tau_o[idx], tau_n[idx]
    hs, hb = hit_surf[idx], hit_bot[idx]

    t_surf = find_crossing(po, pn, surface, n_bisect=n_bisect, n_newton=n_newton)
    t_bot = find_crossing(po, pn, bottom, n_bisect=n_bisect, n_newton=n_newton)
    use_surf = hs & (~hb | (t_surf <= t_bot))
    t = torch.where(use_surf, t_surf, t_bot)

    pos_hit = po + t.unsqueeze(-1) * (pn - po)
    tau_hit = to + t * (tn - to)
    c_hit = field(pos_hit)
    slow_hit = _renormalise(so + t.unsqueeze(-1) * (sn - so), c_hit)

    normal = torch.where(use_surf.unsqueeze(-1),
                         surface.normal(pos_hit[..., :2]),
                         bottom.normal(pos_hit[..., :2]))

    graze = grazing_angle(slow_hit, normal)
    loss = torch.where(use_surf, surface_loss(graze), bottom_loss(graze))
    # Pressure phase imposed by the bounce.  Energy-only rendering ignores this;
    # a beamformer cannot.
    phase = torch.where(use_surf, surface_loss.reflection_phase(graze),
                        bottom_loss.reflection_phase(graze))

    slow_refl = reflect(slow_hit, normal)
    dir_refl = slow_refl * c_hit.unsqueeze(-1)  # unit vector

    # Advance the remainder of the step along the mirrored direction.  The floor
    # on the advance nudges the ray clear of the boundary so the same crossing
    # is not re-detected on the following step.
    t_adv = (1.0 - t).clamp_min(min_advance)
    pos_after = pos_hit + (t_adv * h).unsqueeze(-1) * dir_refl
    tau_after = tau_hit + t_adv * h / c_hit
    arclen_after = state.arclen[idx] + t * h + t_adv * h
    slow_after = _renormalise(slow_refl, field(pos_after))

    ones = torch.ones_like(t)
    zeros_t = torch.zeros_like(t)
    return RayState(
        pos=pos_n.index_copy(0, idx, pos_after),
        slow=slow_n.index_copy(0, idx, slow_after),
        tau=tau_n.index_copy(0, idx, tau_after),
        arclen=(state.arclen + h).index_copy(0, idx, arclen_after),
        refl_db=state.refl_db.index_add(0, idx, loss),
        refl_phase=state.refl_phase.index_add(0, idx, phase),
        # Sign carries which boundary, magnitude the grazing angle; zero means
        # no bounce.  Bounces are sparse, so recording them in the one array
        # they can be decoded from costs far less than a per-vertex counter per
        # boundary.  Read it through bounce_events(), never directly.
        bounce_grazing=zero.index_copy(0, idx, torch.where(use_surf, -graze, graze)),
        alive=state.alive,
        n_surface=state.n_surface.index_add(
            0, idx, torch.where(use_surf, ones, zeros_t)),
        n_bottom=state.n_bottom.index_add(
            0, idx, torch.where(use_surf, zeros_t, ones)),
    )


def _step(
    field,
    surface: HeightField,
    bottom: HeightField,
    surface_loss,
    bottom_loss,
    state: RayState,
    h: float,
    *,
    max_bounces: int,
    domain: tuple[float, float, float, float] | None,
    n_bisect: int,
    n_newton: int,
    min_advance: float,
) -> RayState:
    """One full tracer step: RK4, renormalise, reflect, then retire dead rays."""
    pos_n, slow_n, tau_n = rk4_step(field, state.pos, state.slow, state.tau, h)
    slow_n = _renormalise(slow_n, field(pos_n))

    nxt = _handle_boundaries(
        field, surface, bottom, surface_loss, bottom_loss, state,
        pos_n, slow_n, tau_n, h,
        n_bisect=n_bisect, n_newton=n_newton, min_advance=min_advance,
    )

    # A single non-finite value would poison the whole backward pass, so bad
    # rays are rolled back to their previous position and retired.
    finite = torch.isfinite(nxt.pos).all(-1) & torch.isfinite(nxt.slow).all(-1) & torch.isfinite(nxt.tau)
    alive = nxt.alive * finite.to(nxt.alive.dtype)
    alive = alive * ((nxt.n_surface + nxt.n_bottom) <= max_bounces).to(alive.dtype)
    if domain is not None:
        x0, x1, y0, y1 = domain
        inside = (
            (nxt.pos[..., 0] >= x0) & (nxt.pos[..., 0] <= x1)
            & (nxt.pos[..., 1] >= y0) & (nxt.pos[..., 1] <= y1)
        )
        alive = alive * inside.to(alive.dtype)

    # Retired rays freeze in place: zero-length segments splat no energy, and
    # freezing keeps their stored path finite for plotting.
    keep = (alive > 0).unsqueeze(-1) & torch.isfinite(nxt.pos)
    keep_s = (alive > 0).unsqueeze(-1) & torch.isfinite(nxt.slow)
    keep_1 = (alive > 0) & torch.isfinite(nxt.tau)
    return RayState(
        pos=torch.where(keep, nxt.pos, state.pos),
        slow=torch.where(keep_s, nxt.slow, state.slow),
        tau=torch.where(keep_1, nxt.tau, state.tau),
        arclen=torch.where(keep_1, nxt.arclen, state.arclen),
        refl_db=torch.where(keep_1, nxt.refl_db, state.refl_db),
        refl_phase=torch.where(keep_1, nxt.refl_phase, state.refl_phase),
        bounce_grazing=torch.where(keep_1, nxt.bounce_grazing,
                                   torch.zeros_like(nxt.bounce_grazing)),
        alive=alive,
        n_surface=nxt.n_surface,
        n_bottom=nxt.n_bottom,
    )


def trace(
    scene: "Scene",
    directions: Tensor,
    *,
    n_steps: int | None = None,
    step_size: float | None = None,
    checkpoint_every: int | None = None,
    step_controller: Callable[..., Tensor] | None = None,
) -> TraceResult:
    """Trace ``directions`` (``[R, 3]`` unit vectors) through ``scene``.

    Args:
        scene: a :class:`hydropt.scene.Scene`.
        directions: launch directions, ``[R, 3]``; normalised internally.
        n_steps, step_size, checkpoint_every: override the scene defaults.
        step_controller: hook for adaptive stepping.  Not implemented -- passing
            anything other than ``None`` raises.  An adaptive scheme would
            return a per-ray ``h`` from a local error estimate (e.g. RK4 vs two
            half-steps, or the local radius of curvature ``c / |grad c|``);
            since ``h`` enters the state update differentiably, autograd would
            carry straight through it, but rays would then desynchronise in
            arclength, which the checkpointed chunking assumes they do not.

    Returns:
        :class:`TraceResult` with per-vertex positions, travel times, path
        lengths, accumulated reflection loss and liveness.

    Memory is ``O(rays x steps)`` for the stored path.  Reverse-mode autograd
    through the unrolled integrator would otherwise also store four field
    evaluations and their intermediates per step; ``checkpoint_every`` bounds
    that by recomputing each chunk in the backward pass.
    """
    if step_controller is not None:
        raise NotImplementedError(
            "adaptive stepping is a documented hook; see trace() docstring"
        )

    n_steps = int(scene.n_steps if n_steps is None else n_steps)
    h = float(scene.step_size if step_size is None else step_size)
    chunk = int(scene.checkpoint_every if checkpoint_every is None else checkpoint_every)

    dirs = directions / directions.norm(dim=-1, keepdim=True).clamp_min(1e-30)
    src = scene.source_position().to(dirs.dtype)
    pos0 = src.expand_as(dirs).clone()
    c0 = scene.field(pos0)
    slow0 = dirs / c0.unsqueeze(-1)

    R = dirs.shape[0]
    zeros = torch.zeros(R, dtype=dirs.dtype, device=dirs.device)
    state = RayState(
        pos=pos0,
        slow=slow0,
        tau=zeros.clone(),
        arclen=zeros.clone(),
        refl_db=zeros.clone(),
        refl_phase=zeros.clone(),
        bounce_grazing=zeros.clone(),
        alive=torch.ones_like(zeros),
        n_surface=zeros.clone(),
        n_bottom=zeros.clone(),
    )

    kw = dict(
        max_bounces=int(scene.max_bounces),
        domain=scene.domain,
        n_bisect=int(scene.n_bisect),
        n_newton=int(scene.n_newton),
        min_advance=float(scene.min_advance),
    )

    def run_chunk(k: int, *flat: Tensor) -> tuple[Tensor, ...]:
        """Advance ``k`` steps, returning stacked per-vertex outputs + final state."""
        st = RayState(*flat)
        pos_l, tau_l, arc_l, db_l, ph_l, bg_l, al_l = [], [], [], [], [], [], []
        for _ in range(k):
            st = _step(scene.field, scene.surface, scene.bottom,
                       scene.surface_loss, scene.bottom_loss, st, h, **kw)
            pos_l.append(st.pos)
            tau_l.append(st.tau)
            arc_l.append(st.arclen)
            db_l.append(st.refl_db)
            ph_l.append(st.refl_phase)
            bg_l.append(st.bounce_grazing)
            al_l.append(st.alive)
        return (
            torch.stack(pos_l, dim=1),
            torch.stack(tau_l, dim=1),
            torch.stack(arc_l, dim=1),
            torch.stack(db_l, dim=1),
            torch.stack(ph_l, dim=1),
            torch.stack(bg_l, dim=1),
            torch.stack(al_l, dim=1),
            *st,
        )

    pos_out = [state.pos.unsqueeze(1)]
    tau_out = [state.tau.unsqueeze(1)]
    arc_out = [state.arclen.unsqueeze(1)]
    db_out = [state.refl_db.unsqueeze(1)]
    ph_out = [state.refl_phase.unsqueeze(1)]
    bg_out = [state.bounce_grazing.unsqueeze(1)]
    al_out = [state.alive.unsqueeze(1)]

    step = n_steps if chunk <= 0 else min(chunk, n_steps)
    done = 0
    while done < n_steps:
        k = min(step, n_steps - done)
        use_ckpt = chunk > 0 and torch.is_grad_enabled() and any(
            t.requires_grad for t in state
        )
        if use_ckpt:
            out = checkpoint(run_chunk, k, *state, use_reentrant=False)
        else:
            out = run_chunk(k, *state)
        pos_out.append(out[0])
        tau_out.append(out[1])
        arc_out.append(out[2])
        db_out.append(out[3])
        ph_out.append(out[4])
        bg_out.append(out[5])
        al_out.append(out[6])
        state = RayState(*out[7:])
        done += k

    return TraceResult(
        pos=torch.cat(pos_out, dim=1),
        tau=torch.cat(tau_out, dim=1),
        arclen=torch.cat(arc_out, dim=1),
        refl_db=torch.cat(db_out, dim=1),
        refl_phase=torch.cat(ph_out, dim=1),
        bounce_grazing=torch.cat(bg_out, dim=1),
        alive=torch.cat(al_out, dim=1),
        n_surface=state.n_surface,
        n_bottom=state.n_bottom,
    )
