"""The :class:`Scene` -- everything the tracer and the renderer need.

Implemented as an :class:`torch.nn.Module` rather than a plain dataclass, so
that sub-module parameters (profile knots, bathymetry nodes, boundary losses)
register automatically and ``scene.parameters()`` is directly usable by a
``torch.optim`` optimiser.  It is still constructed and read like a dataclass.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor, nn

from .absorption import octave_bands
from .boundaries import BoundaryLoss, ConstantLoss, FlatHeight, HeightField
from .fields import IsoProfile, SoundSpeedField
from .receiver import make_time_grid, splat_etc
from .tracer import TraceResult, trace

__all__ = ["Scene"]


class Scene(nn.Module):
    """A 3-D ocean acoustic scene.

    Args:
        field: sound-speed field ``c(x, y, z)``.
        bottom: seabed height field ``h(x, y)`` (depth, positive down).
        surface: sea-surface height field; defaults to the flat plane ``z = 0``.
        source: source position ``(x, y, z)`` in metres.
        receivers: ``[Nr, 3]`` receiver positions.
        surface_loss, bottom_loss: reflection losses in dB per bounce.
        freqs_khz: ``[B]`` band centre frequencies.
        step_size: RK4 arclength step (m).
        n_steps: number of steps; ``step_size * n_steps`` is the maximum path
            length any ray can reach.
        max_bounces: rays are retired after this many reflections.
        domain: optional ``(x0, x1, y0, y1)`` box; rays leaving it are retired.
        checkpoint_every: chunk length for gradient checkpointing (0 disables).
        n_bisect, n_newton, min_advance: boundary-intersection settings, see
            :func:`hydropt.boundaries.find_crossing`.
        learn_source: register the source position as a parameter.
    """

    def __init__(
        self,
        field: SoundSpeedField | None = None,
        bottom: HeightField | None = None,
        surface: HeightField | None = None,
        *,
        source: Sequence[float] | Tensor = (0.0, 0.0, 100.0),
        receivers: Tensor | None = None,
        surface_loss: BoundaryLoss | None = None,
        bottom_loss: BoundaryLoss | None = None,
        freqs_khz: Tensor | None = None,
        step_size: float = 25.0,
        n_steps: int = 2000,
        max_bounces: int = 60,
        domain: tuple[float, float, float, float] | None = None,
        checkpoint_every: int = 0,
        n_bisect: int = 12,
        n_newton: int = 1,
        min_advance: float = 1e-3,
        learn_source: bool = False,
    ) -> None:
        super().__init__()
        self.field = field if field is not None else IsoProfile(1500.0)
        self.bottom = bottom if bottom is not None else FlatHeight(1000.0)
        self.surface = surface if surface is not None else FlatHeight(0.0)
        self.surface_loss = surface_loss if surface_loss is not None else ConstantLoss(0.5)
        self.bottom_loss = bottom_loss if bottom_loss is not None else ConstantLoss(3.0)

        src = torch.as_tensor(source, dtype=torch.get_default_dtype()).reshape(3)
        if learn_source:
            self.source = nn.Parameter(src)
        else:
            self.register_buffer("source", src)

        rec = receivers if receivers is not None else torch.tensor([[1000.0, 0.0, 100.0]])
        self.register_buffer(
            "receivers", torch.as_tensor(rec, dtype=src.dtype).reshape(-1, 3)
        )
        self.register_buffer(
            "freqs_khz",
            octave_bands(0.1, 4, dtype=src.dtype) if freqs_khz is None
            else torch.as_tensor(freqs_khz, dtype=src.dtype).reshape(-1),
        )

        self.step_size = float(step_size)
        self.n_steps = int(n_steps)
        self.max_bounces = int(max_bounces)
        self.domain = tuple(domain) if domain is not None else None
        self.checkpoint_every = int(checkpoint_every)
        self.n_bisect = int(n_bisect)
        self.n_newton = int(n_newton)
        self.min_advance = float(min_advance)

    # -- accessors ---------------------------------------------------------- #
    def source_position(self) -> Tensor:
        """Source position, ``[3]`` (a parameter when ``learn_source=True``)."""
        return self.source

    @property
    def max_path_length(self) -> float:
        return self.step_size * self.n_steps

    def default_time_grid(self, n_bins: int = 600, *, c_ref: float = 1500.0) -> Tensor:
        """Time grid spanning zero to the longest traceable travel time."""
        return make_time_grid(0.0, self.max_path_length / c_ref, n_bins,
                              dtype=self.source.dtype, device=self.source.device)

    # -- forward model ------------------------------------------------------ #
    def trace(self, directions: Tensor, **kwargs) -> TraceResult:
        """Trace ``directions`` through this scene (see :func:`hydropt.tracer.trace`)."""
        return trace(self, directions, **kwargs)

    def render(
        self,
        directions: Tensor,
        time_grid: Tensor | None = None,
        *,
        sigma_d: float = 50.0,
        sigma_t: float = 2.0e-3,
        ray_chunk: int = 0,
        trace_kwargs: dict | None = None,
        **splat_kwargs,
    ) -> Tensor:
        """Trace and splat in one call.  Returns the ETC, ``[Nr, B, T]``."""
        result = self.trace(directions, **(trace_kwargs or {}))
        grid = self.default_time_grid() if time_grid is None else time_grid
        return splat_etc(
            result, self.receivers, grid, self.freqs_khz,
            sigma_d=sigma_d, sigma_t=sigma_t, ray_chunk=ray_chunk, **splat_kwargs,
        )

    def render_chunked(
        self,
        directions: Tensor,
        time_grid: Tensor | None = None,
        *,
        chunk_size: int = 512,
        sigma_d: float = 50.0,
        sigma_t: float = 2.0e-3,
        trace_kwargs: dict | None = None,
        **splat_kwargs,
    ) -> Tensor:
        """Same ETC as :meth:`render`, accumulated over chunks of rays.

        Peak memory scales with ``chunk_size`` instead of the whole fan, which
        is what makes large fans practical under reverse-mode autograd.  The
        result is a plain sum over chunks, so it is identical up to
        floating-point reassociation -- this is what ``tests/test_batching.py``
        pins down.
        """
        grid = self.default_time_grid() if time_grid is None else time_grid
        total: Tensor | None = None
        for lo in range(0, int(directions.shape[0]), int(chunk_size)):
            part = self.render(
                directions[lo : lo + int(chunk_size)], grid,
                sigma_d=sigma_d, sigma_t=sigma_t,
                trace_kwargs=trace_kwargs, **splat_kwargs,
            )
            total = part if total is None else total + part
        if total is None:
            raise ValueError("no directions given")
        return total
