"""Sound-speed fields ``c(x, y, z)`` as differentiable :class:`torch.nn.Module`s.

Coordinate convention throughout hydropt: ``x``, ``y`` are horizontal (m) and
``z`` is *depth* (m, positive downward) with ``z = 0`` at the mean sea surface.

Every field exposes two batched entry points:

``forward(points)``
    ``points`` has shape ``[..., 3]``; returns ``c`` with shape ``[...]``.
``c_and_grad(points)``
    returns ``(c, grad_c)`` where ``grad_c`` has shape ``[..., 3]`` and holds
    ``(dc/dx, dc/dy, dc/dz)``.

The base class implements ``c_and_grad`` via reverse-mode autograd, so a
user-defined field only has to implement ``forward``.  All built-in fields
override it with analytic (and therefore cheaper) expressions; the analytic
gradients are themselves differentiable with respect to the field parameters,
which is what makes second-order terms in the inverse problems work.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import Tensor, nn

__all__ = [
    "SoundSpeedField",
    "DepthProfile",
    "IsoProfile",
    "MunkProfile",
    "PiecewiseLinearProfile",
    "GriddedField",
]


class SoundSpeedField(nn.Module):
    """Abstract base class for sound-speed fields."""

    def forward(self, points: Tensor) -> Tensor:  # pragma: no cover - abstract
        raise NotImplementedError

    def c_and_grad(self, points: Tensor) -> tuple[Tensor, Tensor]:
        """Value and spatial gradient of ``c`` at ``points`` (``[..., 3]``).

        Generic autograd fallback.  ``c`` is a *pointwise* function of each row
        of ``points``, so differentiating ``c.sum()`` recovers the per-point
        gradient in a single backward pass.
        """
        grad_mode = torch.is_grad_enabled()
        with torch.enable_grad():
            q = points if points.requires_grad else points.detach().requires_grad_(True)
            c = self(q)
            # allow_unused/materialize_grads keep a field whose value does not
            # depend on position (a constant profile) from raising here.
            (g,) = torch.autograd.grad(
                c.sum(), q, create_graph=grad_mode,
                allow_unused=True, materialize_grads=True,
            )
        if not grad_mode:
            c, g = c.detach(), g.detach()
        return c, g

    @property
    def dtype(self) -> torch.dtype:
        for p in self.parameters():
            return p.dtype
        for b in self.buffers():
            return b.dtype
        return torch.get_default_dtype()

    @property
    def device(self) -> torch.device:
        for p in self.parameters():
            return p.device
        for b in self.buffers():
            return b.device
        return torch.device("cpu")


class DepthProfile(SoundSpeedField):
    """Base class for depth-only profiles ``c(z)``.

    Subclasses implement :meth:`c_of_z` and :meth:`dcdz`; the horizontal
    components of the gradient are identically zero.
    """

    def c_of_z(self, z: Tensor) -> Tensor:  # pragma: no cover - abstract
        raise NotImplementedError

    def dcdz(self, z: Tensor) -> Tensor:  # pragma: no cover - abstract
        raise NotImplementedError

    def forward(self, points: Tensor) -> Tensor:
        return self.c_of_z(points[..., 2])

    def c_and_grad(self, points: Tensor) -> tuple[Tensor, Tensor]:
        z = points[..., 2]
        c = self.c_of_z(z)
        dz = self.dcdz(z)
        zero = torch.zeros_like(dz)
        return c, torch.stack((zero, zero, dz), dim=-1)


class IsoProfile(DepthProfile):
    """Constant sound speed.  Rays are straight lines.

    Args:
        c0: sound speed (m/s).
        learnable: register ``c0`` as a :class:`torch.nn.Parameter`.
    """

    def __init__(self, c0: float = 1500.0, *, learnable: bool = False) -> None:
        super().__init__()
        t = torch.as_tensor(float(c0))
        self.c0 = nn.Parameter(t) if learnable else None
        if self.c0 is None:
            self.register_buffer("_c0", t)

    def _value(self) -> Tensor:
        return self.c0 if self.c0 is not None else self._c0

    def c_of_z(self, z: Tensor) -> Tensor:
        return self._value().to(z.dtype).expand(z.shape).clone()

    def dcdz(self, z: Tensor) -> Tensor:
        return torch.zeros_like(z)


class LinearGradientProfile(DepthProfile):
    """``c(z) = c0 + g z``.  Rays are exact circular arcs -- used by the tests."""

    def __init__(self, c0: float = 1500.0, g: float = 0.016, *, learnable: bool = False) -> None:
        super().__init__()
        c0_t, g_t = torch.as_tensor(float(c0)), torch.as_tensor(float(g))
        if learnable:
            self.c0 = nn.Parameter(c0_t)
            self.g = nn.Parameter(g_t)
        else:
            self.register_buffer("c0", c0_t)
            self.register_buffer("g", g_t)

    def c_of_z(self, z: Tensor) -> Tensor:
        return self.c0.to(z.dtype) + self.g.to(z.dtype) * z

    def dcdz(self, z: Tensor) -> Tensor:
        return self.g.to(z.dtype).expand(z.shape).clone()


class MunkProfile(DepthProfile):
    r"""Canonical Munk deep-water profile.

    .. math::
        c(z) = c_1 \left[ 1 + \epsilon\,(\eta + e^{-\eta} - 1) \right],
        \qquad \eta = \frac{2 (z - z_1)}{B}

    The minimum sits at the sound-channel axis ``z = z1``.  All four parameters
    (``c1``, ``z1``, ``B``, ``eps``) are learnable by default.
    """

    def __init__(
        self,
        c1: float = 1500.0,
        z1: float = 1300.0,
        B: float = 1300.0,
        eps: float = 7.37e-3,
        *,
        learnable: bool = True,
    ) -> None:
        super().__init__()
        vals = {"c1": c1, "z1": z1, "B": B, "eps": eps}
        for name, v in vals.items():
            t = torch.as_tensor(float(v))
            if learnable:
                setattr(self, name, nn.Parameter(t))
            else:
                self.register_buffer(name, t)

    def _eta(self, z: Tensor) -> Tensor:
        return 2.0 * (z - self.z1.to(z.dtype)) / self.B.to(z.dtype)

    def c_of_z(self, z: Tensor) -> Tensor:
        eta = self._eta(z)
        return self.c1.to(z.dtype) * (1.0 + self.eps.to(z.dtype) * (eta + torch.exp(-eta) - 1.0))

    def dcdz(self, z: Tensor) -> Tensor:
        eta = self._eta(z)
        return (
            self.c1.to(z.dtype)
            * self.eps.to(z.dtype)
            * (1.0 - torch.exp(-eta))
            * (2.0 / self.B.to(z.dtype))
        )


class PiecewiseLinearProfile(DepthProfile):
    """Piecewise-linear ``c(z)`` through learnable knot values.

    Knot *depths* are fixed (a buffer) and knot *values* are the learnable
    parameters.  Holding the depths fixed keeps the knots sorted, which the
    interpolation relies on; ``learn_depths=True`` is available but then it is
    the caller's job to keep them monotone (e.g. with a projection step).

    Outside ``[depths[0], depths[-1]]`` the profile is held constant, so
    ``dc/dz`` is zero there.  Note that ``dc/dz`` is piecewise *constant* and
    therefore jumps at the knots; that is harmless for RK4 but means a
    gradcheck must not straddle a knot.
    """

    def __init__(
        self,
        depths: Sequence[float] | Tensor,
        values: Sequence[float] | Tensor,
        *,
        learnable: bool = True,
        learn_depths: bool = False,
    ) -> None:
        super().__init__()
        d = torch.as_tensor(depths, dtype=torch.get_default_dtype()).flatten()
        v = torch.as_tensor(values, dtype=torch.get_default_dtype()).flatten()
        if d.numel() != v.numel():
            raise ValueError(f"depths and values must match: {d.numel()} vs {v.numel()}")
        if d.numel() < 2:
            raise ValueError("need at least two knots")
        if not bool((d[1:] > d[:-1]).all()):
            raise ValueError("depths must be strictly increasing")
        if learnable:
            self.values = nn.Parameter(v)
        else:
            self.register_buffer("values", v)
        if learn_depths:
            self.depths = nn.Parameter(d)
        else:
            self.register_buffer("depths", d)

    def _segment(self, z: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        d = self.depths.to(z.dtype)
        v = self.values.to(z.dtype)
        n = d.numel()
        # Bin index, clamped so that both i and i+1 are valid knots.
        idx = torch.searchsorted(d.detach().contiguous(), z.detach().contiguous(), right=True) - 1
        idx = idx.clamp(0, n - 2)
        d0, d1 = d[idx], d[idx + 1]
        v0, v1 = v[idx], v[idx + 1]
        return d0, d1, v0, v1

    def c_of_z(self, z: Tensor) -> Tensor:
        d0, d1, v0, v1 = self._segment(z)
        t = ((z - d0) / (d1 - d0)).clamp(0.0, 1.0)
        return v0 + t * (v1 - v0)

    def dcdz(self, z: Tensor) -> Tensor:
        d0, d1, v0, v1 = self._segment(z)
        slope = (v1 - v0) / (d1 - d0)
        d = self.depths.to(z.dtype)
        inside = ((z >= d[0]) & (z <= d[-1])).to(z.dtype)
        return slope * inside


class GriddedField(SoundSpeedField):
    """3-D sound speed on a regular grid with trilinear interpolation.

    Use this for range-dependent oceans (fronts, eddies, internal waves).  The
    grid holds either absolute sound speed, or -- when ``base`` is given -- a
    *perturbation* added to a depth-only background profile.  The perturbation
    form is much better conditioned for inversion, because the optimiser starts
    from a physically sane ocean and only has to explain the residual.

    Args:
        values: ``[nz, ny, nx]`` tensor of sound speeds (or perturbations, m/s).
        origin: ``(x0, y0, z0)`` of grid node ``[0, 0, 0]`` in metres.
        spacing: ``(dx, dy, dz)`` node spacing in metres.
        base: optional background :class:`DepthProfile`.
        learnable: register ``values`` as a parameter.

    Outside the box the field is clamped to the boundary node values, so the
    gradient there is zero along the clamped axes.
    """

    def __init__(
        self,
        values: Tensor,
        origin: Sequence[float],
        spacing: Sequence[float],
        *,
        base: DepthProfile | None = None,
        learnable: bool = True,
    ) -> None:
        super().__init__()
        v = torch.as_tensor(values, dtype=torch.get_default_dtype())
        if v.ndim != 3:
            raise ValueError(f"values must be [nz, ny, nx], got shape {tuple(v.shape)}")
        if min(v.shape) < 2:
            raise ValueError("each grid axis needs at least two nodes")
        if learnable:
            self.values = nn.Parameter(v)
        else:
            self.register_buffer("values", v)
        self.register_buffer("origin", torch.as_tensor(origin, dtype=v.dtype))
        self.register_buffer("spacing", torch.as_tensor(spacing, dtype=v.dtype))
        self.base = base

    @property
    def shape(self) -> tuple[int, int, int]:
        nz, ny, nx = self.values.shape
        return int(nz), int(ny), int(nx)

    def _lattice(self, points: Tensor) -> tuple[Tensor, ...]:
        """Return corner indices, fractional offsets and inside-mask per axis."""
        nz, ny, nx = self.shape
        origin = self.origin.to(points.dtype)
        spacing = self.spacing.to(points.dtype)
        n = torch.tensor([nx, ny, nz], device=points.device)

        u = (points - origin) / spacing  # [..., 3] in (x, y, z) order
        inside = (u >= 0) & (u <= (n - 1).to(u.dtype))
        u = u.clamp(torch.zeros_like(u), (n - 1).to(u.dtype).expand_as(u))
        i0 = u.floor().long().clamp(torch.zeros_like(n), n - 2)
        frac = (u - i0.to(u.dtype)).clamp(0.0, 1.0)
        return i0, frac, inside

    def _gather_corners(self, i0: Tensor) -> Tensor:
        """Gather the 8 cell corners.  Returns ``[..., 2, 2, 2]`` as ``[w, v, u]``."""
        nz, ny, nx = self.shape
        ix, iy, iz = i0[..., 0], i0[..., 1], i0[..., 2]
        flat = self.values.reshape(-1)
        out = []
        for dz_ in (0, 1):
            for dy_ in (0, 1):
                for dx_ in (0, 1):
                    lin = (iz + dz_) * (ny * nx) + (iy + dy_) * nx + (ix + dx_)
                    out.append(flat[lin])
        return torch.stack(out, dim=-1).reshape(*i0.shape[:-1], 2, 2, 2)

    def forward(self, points: Tensor) -> Tensor:
        c, _ = self.c_and_grad(points)
        return c

    def c_and_grad(self, points: Tensor) -> tuple[Tensor, Tensor]:
        i0, frac, inside = self._lattice(points)
        corners = self._gather_corners(i0).to(points.dtype)  # [..., w, v, u]
        fu, fv, fw = frac[..., 0], frac[..., 1], frac[..., 2]

        wu = torch.stack((1 - fu, fu), dim=-1)  # [..., 2]
        wv = torch.stack((1 - fv, fv), dim=-1)
        ww = torch.stack((1 - fw, fw), dim=-1)
        du = torch.stack((-torch.ones_like(fu), torch.ones_like(fu)), dim=-1)

        def contract(au: Tensor, av: Tensor, aw: Tensor) -> Tensor:
            # corners is [..., w, v, u]
            t = (corners * au[..., None, None, :]).sum(-1)  # [..., w, v]
            t = (t * av[..., None, :]).sum(-1)  # [..., w]
            return (t * aw).sum(-1)

        c = contract(wu, wv, ww)
        spacing = self.spacing.to(points.dtype)
        gx = contract(du, wv, ww) / spacing[0]
        gy = contract(wu, du, ww) / spacing[1]
        gz = contract(wu, wv, du) / spacing[2]
        grad = torch.stack((gx, gy, gz), dim=-1) * inside.to(points.dtype)

        if self.base is not None:
            cb, gb = self.base.c_and_grad(points)
            c = c + cb
            grad = grad + gb
        return c, grad
