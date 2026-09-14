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

from .boundaries import HeightField, grazing_angle, find_crossing, reflect

if TYPE_CHECKING:  # pragma: no cover
    from .scene import Scene

__all__ = ["RayState", "TraceResult", "rk4_step", "trace"]


class RayState(NamedTuple):
    """Per-ray integration state.  All tensors have leading dimension ``[rays]``."""

    pos: Tensor  # [R, 3] position (m)
    slow: Tensor  # [R, 3] slowness vector, |slow| = 1/c
    tau: Tensor  # [R] travel time (s)
    arclen: Tensor  # [R] path length (m)
    refl_db: Tensor  # [R] accumulated reflection loss (dB)
    alive: Tensor  # [R] 1.0 while the ray is still propagating
    n_surface: Tensor  # [R] surface bounce count (float, for batched maths)
    n_bottom: Tensor  # [R] bottom bounce count


class TraceResult(NamedTuple):
    """Sampled ray paths.  ``S = n_steps``, so vertex arrays are ``S + 1`` long."""

    pos: Tensor  # [R, S+1, 3]
    tau: Tensor  # [R, S+1]
    arclen: Tensor  # [R, S+1]
    refl_db: Tensor  # [R, S+1]
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
    any_hit = hit_surf | hit_bot

    if not bool(any_hit.any()):
        return RayState(pos_n, slow_n, tau_n, state.arclen + h, state.refl_db,
                        state.alive, state.n_surface, state.n_bottom)

    t_surf = find_crossing(pos_o, pos_n, surface, n_bisect=n_bisect, n_newton=n_newton)
    t_bot = find_crossing(pos_o, pos_n, bottom, n_bisect=n_bisect, n_newton=n_newton)

    use_surf = hit_surf & (~hit_bot | (t_surf <= t_bot))
    t = torch.where(use_surf, t_surf, t_bot)
    t = torch.where(any_hit, t, torch.zeros_like(t))

    pos_hit = pos_o + t.unsqueeze(-1) * (pos_n - pos_o)
    tau_hit = tau_o + t * (tau_n - tau_o)
    slow_hit = slow_o + t.unsqueeze(-1) * (slow_n - slow_o)
    c_hit = field(pos_hit)
    slow_hit = _renormalise(slow_hit, c_hit)

    n_surf_vec = surface.normal(pos_hit[..., :2])
    n_bot_vec = bottom.normal(pos_hit[..., :2])
    normal = torch.where(use_surf.unsqueeze(-1), n_surf_vec, n_bot_vec)

    graze = grazing_angle(slow_hit, normal)
    loss = torch.where(use_surf, surface_loss(graze), bottom_loss(graze))

    slow_refl = reflect(slow_hit, normal)
    dir_refl = slow_refl * c_hit.unsqueeze(-1)  # unit vector

    # Advance the remainder of the step along the mirrored direction.  The floor
    # on the advance nudges the ray clear of the boundary so the same crossing
    # is not re-detected on the following step.
    t_adv = (1.0 - t).clamp_min(min_advance)
    pos_after = pos_hit + (t_adv * h).unsqueeze(-1) * dir_refl
    tau_after = tau_hit + t_adv * h / c_hit
    arclen_after = state.arclen + t * h + t_adv * h
    slow_after = _renormalise(slow_refl, field(pos_after))

    m = (any_hit & (state.alive > 0)).unsqueeze(-1)
    mf = (any_hit & (state.alive > 0)).to(pos_n.dtype)
    return RayState(
        pos=torch.where(m, pos_after, pos_n),
        slow=torch.where(m, slow_after, slow_n),
        tau=torch.where(mf > 0, tau_after, tau_n),
        arclen=torch.where(mf > 0, arclen_after, state.arclen + h),
        refl_db=state.refl_db + mf * loss,
        alive=state.alive,
        n_surface=state.n_surface + (hit_surf & (state.alive > 0) & use_surf).to(mf.dtype),
        n_bottom=state.n_bottom + (hit_bot & (state.alive > 0) & ~use_surf).to(mf.dtype),
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
        pos_l, tau_l, arc_l, db_l, al_l = [], [], [], [], []
        for _ in range(k):
            st = _step(scene.field, scene.surface, scene.bottom,
                       scene.surface_loss, scene.bottom_loss, st, h, **kw)
            pos_l.append(st.pos)
            tau_l.append(st.tau)
            arc_l.append(st.arclen)
            db_l.append(st.refl_db)
            al_l.append(st.alive)
        return (
            torch.stack(pos_l, dim=1),
            torch.stack(tau_l, dim=1),
            torch.stack(arc_l, dim=1),
            torch.stack(db_l, dim=1),
            torch.stack(al_l, dim=1),
            *st,
        )

    pos_out = [state.pos.unsqueeze(1)]
    tau_out = [state.tau.unsqueeze(1)]
    arc_out = [state.arclen.unsqueeze(1)]
    db_out = [state.refl_db.unsqueeze(1)]
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
        al_out.append(out[4])
        state = RayState(*out[5:])
        done += k

    return TraceResult(
        pos=torch.cat(pos_out, dim=1),
        tau=torch.cat(tau_out, dim=1),
        arclen=torch.cat(arc_out, dim=1),
        refl_db=torch.cat(db_out, dim=1),
        alive=torch.cat(al_out, dim=1),
        n_surface=state.n_surface,
        n_bottom=state.n_bottom,
    )
