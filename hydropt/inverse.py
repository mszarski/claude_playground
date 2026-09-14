"""Gradient-based inversion: fit scene parameters to a measured ETC.

The forward model is differentiable end to end, so an inverse problem is just an
optimisation over ``scene.parameters()``.  What makes these problems work in
practice is less the gradient than three choices around it, all handled here:

**Log-domain loss.**  An ETC spans decades.  A plain MSE on linear energy is
dominated by the loudest arrival and carries almost no information about the
rest, so ``"log_mse"`` (the default) compares ``log10(etc + eps)``.  That also
turns the boundary-loss problem, where energy depends on ``10^(-L/10)``, into a
nearly linear one.

**Sigma annealing.**  Kernel widths control the basin of attraction (see
:mod:`hydropt.receiver`).  Starting wide gives a far-from-correct initial guess
a usable gradient; shrinking later recovers resolution.  ``sigma_d_schedule``
and ``sigma_t_schedule`` take either a constant or a ``(start, end)`` pair
interpolated geometrically over the run.

**Parameter projection.**  Some parameters have to stay physical (positive
attenuation, monotone knot depths, a bottom below the surface).  Pass
``project`` to clamp after each step; it runs under ``no_grad``.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field as dc_field
from typing import Callable, Iterable, Literal, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

from .scene import Scene

__all__ = ["FitHistory", "blur_time", "etc_loss", "fit"]


def blur_time(etc: Tensor, sigma_bins: float) -> Tensor:
    """Convolve an ETC along its time axis with a unit-area Gaussian.

    Used to bring a measurement up to the model's current time resolution while
    the kernel is being annealed.  Because both the splat kernel and this one
    are unit-area Gaussians, blurring a measurement made at ``sigma_meas`` by
    ``sqrt(sigma^2 - sigma_meas^2)`` reproduces *exactly* what the model would
    render at ``sigma`` -- so the comparison stays like-for-like at every stage
    instead of matching a blurred model against a sharp measurement.
    """
    if sigma_bins <= 1e-3:
        return etc
    radius = max(1, int(math.ceil(4.0 * sigma_bins)))
    offsets = torch.arange(-radius, radius + 1, dtype=etc.dtype, device=etc.device)
    kernel = torch.exp(-0.5 * (offsets / sigma_bins) ** 2)
    kernel = kernel / kernel.sum()
    flat = etc.reshape(-1, 1, etc.shape[-1])
    padded = F.pad(flat, (radius, radius), mode="constant", value=0.0)
    return F.conv1d(padded, kernel.view(1, 1, -1)).reshape(etc.shape)

LossName = Literal["log_mse", "mse", "l1", "log_l1"]


@dataclass
class FitHistory:
    """Everything a fit recorded, for plotting and for assertions in examples."""

    loss: list[float] = dc_field(default_factory=list)
    params: dict[str, list[list[float]]] = dc_field(default_factory=dict)
    extra: dict[str, list[float]] = dc_field(default_factory=dict)
    seconds: float = 0.0
    skipped_steps: int = 0

    def record(self, loss: float, named: Iterable[tuple[str, Tensor]],
               extra: dict[str, float] | None = None) -> None:
        self.loss.append(float(loss))
        self.snapshot(named, extra)

    def snapshot(self, named: Iterable[tuple[str, Tensor]],
                 extra: dict[str, float] | None = None) -> None:
        """Record parameter values without a matching loss.

        Used once after the loop closes, so the last entry in ``params`` is the
        converged state rather than the state one Adam step before it.
        """
        for name, tensor in named:
            self.params.setdefault(name, []).append(
                tensor.detach().reshape(-1).tolist()
            )
        for key, value in (extra or {}).items():
            self.extra.setdefault(key, []).append(float(value))

    def final(self, name: str) -> list[float]:
        """Converged value of a parameter.

        ``params`` carries one more entry than ``loss``: entry ``i`` holds the
        parameters that *produced* ``loss[i]``, and the extra final entry is the
        state after the last step.
        """
        return self.params[name][-1]

    def __len__(self) -> int:
        return len(self.loss)


def etc_loss(pred: Tensor, target: Tensor, *, kind: LossName = "log_mse",
             eps: float | None = None) -> Tensor:
    """Discrepancy between a predicted and a measured ETC.

    ``eps`` sets the noise floor of the log compression.  It defaults to
    ``1e-6`` of the target's peak, which keeps empty time bins from dominating
    while still letting a genuinely quiet bin carry information.
    """
    if kind in ("log_mse", "log_l1"):
        if eps is None:
            eps = 1e-6 * float(target.max().clamp_min(1e-300))
        pred = torch.log10(pred.clamp_min(0.0) + eps)
        target = torch.log10(target.clamp_min(0.0) + eps)
    residual = pred - target
    if kind in ("mse", "log_mse"):
        return (residual * residual).mean()
    return residual.abs().mean()


def _schedule(spec: float | tuple[float, float], t: float) -> float:
    """Geometric interpolation for an annealed hyperparameter; ``t`` in ``[0, 1]``."""
    if isinstance(spec, (int, float)):
        return float(spec)
    start, end = (float(v) for v in spec)
    if start <= 0 or end <= 0:
        raise ValueError("annealed sigmas must be positive")
    return start * (end / start) ** t


def fit(
    scene: Scene,
    target_etc: Tensor,
    directions: Tensor,
    *,
    params_to_optimize: Sequence[Tensor] | dict[str, Tensor] | None = None,
    time_grid: Tensor | None = None,
    n_iters: int = 100,
    optimizer: torch.optim.Optimizer | None = None,
    lr: float = 0.05,
    loss_kind: LossName = "log_mse",
    sigma_d_schedule: float | tuple[float, float] = 80.0,
    sigma_t_schedule: float | tuple[float, float] = 3e-3,
    target_sigma_t: float | None = None,
    ray_chunk_size: int = 0,
    splat_kwargs: dict | None = None,
    regulariser: Callable[[], Tensor] | None = None,
    project: Callable[[], None] | None = None,
    callback: Callable[[int, float], None] | None = None,
    log_every: int = 10,
    verbose: bool = True,
    track: dict[str, Callable[[], float]] | None = None,
) -> FitHistory:
    """Fit ``scene`` parameters so its rendered ETC matches ``target_etc``.

    Args:
        scene: the scene to optimise, modified in place.
        target_etc: measured ETC, ``[Nr, B, T]``.
        directions: launch directions to render with, ``[R, 3]``.
        params_to_optimize: tensors to optimise; a dict is used for its names in
            the history.  Defaults to every parameter of ``scene`` that requires
            grad.
        time_grid: time grid matching ``target_etc``; defaults to the scene's.
        n_iters: optimiser steps.
        optimizer: pre-built optimiser; if ``None``, Adam with ``lr``.
        loss_kind: see :func:`etc_loss`.
        sigma_d_schedule, sigma_t_schedule: constant, or ``(start, end)``
            annealed geometrically over the run.
        target_sigma_t: the time-kernel width ``target_etc`` was measured at.
            When given, the target is blurred at each iteration to match the
            model's current ``sigma_t``, so an annealed model is never compared
            against a sharper measurement.  Leave it ``None`` only if
            ``sigma_t_schedule`` is constant.
        ray_chunk_size: render in ray chunks of this size to bound memory
            (0 = one shot).
        splat_kwargs: extra arguments forwarded to the splatter.
        regulariser: returns a scalar penalty added to the data misfit.  Ocean
            inverse problems are routinely underdetermined -- many profiles or
            seabeds explain the same arrivals -- and a smoothness penalty is the
            standard way to pick the least contrived one.
        project: called after each step, under ``no_grad``, to re-impose
            physical constraints.
        callback: ``(iteration, loss)`` hook.
        track: named scalars to record each iteration, e.g. a true-vs-fitted RMS
            error.  Handy for the acceptance checks in the examples.

    Returns:
        :class:`FitHistory`.
    """
    if params_to_optimize is None:
        named = [(n, p) for n, p in scene.named_parameters() if p.requires_grad]
    elif isinstance(params_to_optimize, dict):
        named = list(params_to_optimize.items())
    else:
        named = [(f"param_{i}", p) for i, p in enumerate(params_to_optimize)]
    if not named:
        raise ValueError("nothing to optimise: no parameter requires grad")

    params = [p for _, p in named]
    opt = optimizer if optimizer is not None else torch.optim.Adam(params, lr=lr)
    grid = scene.default_time_grid() if time_grid is None else time_grid
    if grid.shape[0] != target_etc.shape[-1]:
        raise ValueError(
            f"time grid has {grid.shape[0]} bins but target_etc has {target_etc.shape[-1]}"
        )

    dt_grid = (grid[-1] - grid[0]) / max(grid.shape[0] - 1, 1)
    if target_sigma_t is None and not isinstance(sigma_t_schedule, (int, float)):
        raise ValueError(
            "sigma_t_schedule anneals the model's time kernel, so target_sigma_t "
            "must say what width the measurement was made at -- otherwise a "
            "blurred prediction is being fitted to a sharp measurement."
        )

    history = FitHistory()
    extra_fns = track or {}
    n_skipped = 0
    started = time.perf_counter()

    for it in range(n_iters):
        t = it / max(n_iters - 1, 1)
        sigma_d = _schedule(sigma_d_schedule, t)
        sigma_t = _schedule(sigma_t_schedule, t)

        opt.zero_grad(set_to_none=True)
        render = scene.render_chunked if ray_chunk_size > 0 else scene.render
        kw = dict(sigma_d=sigma_d, sigma_t=sigma_t, **(splat_kwargs or {}))
        if ray_chunk_size > 0:
            kw["chunk_size"] = ray_chunk_size
        pred = render(directions, grid, **kw)

        reference = target_etc
        if target_sigma_t is not None and sigma_t > target_sigma_t:
            extra = math.sqrt(sigma_t**2 - target_sigma_t**2)
            reference = blur_time(target_etc, extra / float(dt_grid))

        data_loss = etc_loss(pred, reference, kind=loss_kind)
        loss = data_loss if regulariser is None else data_loss + regulariser()
        loss.backward()

        # A single non-finite gradient would otherwise turn every parameter into
        # NaN and silently waste the rest of the run.  Skipping the step keeps
        # the fit alive and the count is reported, so the problem is visible
        # rather than hidden.
        # Recorded after the backward pass but before the step, so that
        # loss[i] and params[i] describe the same scene.
        extras = {k: fn() for k, fn in extra_fns.items()}
        if regulariser is not None:
            extras.setdefault("data_loss", data_loss.item())
        history.record(loss.item(), named, extras)

        bad = [n for n, p in named
               if p.grad is not None and not torch.isfinite(p.grad).all()]
        if bad:
            n_skipped += 1
            opt.zero_grad(set_to_none=True)
            if verbose and n_skipped <= 3:
                print(f"  iter {it:4d}  skipped: non-finite gradient on {', '.join(bad)}")
        else:
            opt.step()
        if project is not None:
            with torch.no_grad():
                project()
        if callback is not None:
            callback(it, loss.item())
        if verbose and (it % log_every == 0 or it == n_iters - 1):
            bits = " ".join(f"{k}={v:.4g}" for k, v in extras.items())
            print(f"  iter {it:4d}  loss {loss.item():.6e}  "
                  f"sigma_d {sigma_d:7.2f}  {bits}")

    history.snapshot(named, {k: fn() for k, fn in extra_fns.items()})
    history.seconds = time.perf_counter() - started
    history.skipped_steps = n_skipped
    if n_skipped and verbose:
        print(f"  note: {n_skipped} of {n_iters} steps skipped on non-finite gradients")
    return history
