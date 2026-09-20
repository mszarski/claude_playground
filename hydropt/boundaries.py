"""Sea surface, bathymetry, ray-boundary intersection and reflection losses.

Boundaries are *height fields* ``z = h(x, y)`` in the depth-positive-downward
frame, so the sea surface is ``h ~ 0`` and the seabed is ``h ~ water depth``.
Both are the same class of object, which is what lets a learnable sea state
``eta(x, y)`` drop in later with no change to the tracer.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import Tensor, nn

__all__ = [
    "HeightField",
    "FlatHeight",
    "BilinearHeightField",
    "reflect",
    "find_crossing",
    "grazing_angle",
    "BoundaryLoss",
    "ConstantLoss",
    "RayleighBottomLoss",
]


# --------------------------------------------------------------------------- #
# Height fields
# --------------------------------------------------------------------------- #
class HeightField(nn.Module):
    """Abstract boundary ``z = h(x, y)``."""

    def height(self, xy: Tensor) -> Tensor:  # pragma: no cover - abstract
        """``xy``: ``[..., 2]`` -> ``[...]``."""
        raise NotImplementedError

    def height_and_slope(self, xy: Tensor) -> tuple[Tensor, Tensor]:  # pragma: no cover
        """Return ``(h, dh)`` with ``dh`` of shape ``[..., 2]`` = ``(dh/dx, dh/dy)``."""
        raise NotImplementedError

    def forward(self, xy: Tensor) -> Tensor:
        return self.height(xy)

    def normal(self, xy: Tensor) -> Tensor:
        """Unit normal to the surface, ``[..., 3]``.

        For ``F = z - h(x, y) = 0`` the gradient is ``(-h_x, -h_y, 1)``; the
        orientation is irrelevant to mirroring, so no sign convention is
        imposed.
        """
        _, dh = self.height_and_slope(xy)
        n = torch.cat((-dh, torch.ones_like(dh[..., :1])), dim=-1)
        return n / n.norm(dim=-1, keepdim=True).clamp_min(1e-30)


class FlatHeight(HeightField):
    """Constant-depth boundary ``h(x, y) = z0``.  ``z0=0`` is the flat sea surface."""

    def __init__(self, z0: float = 0.0, *, learnable: bool = False) -> None:
        super().__init__()
        t = torch.as_tensor(float(z0))
        if learnable:
            self.z0 = nn.Parameter(t)
        else:
            self.register_buffer("z0", t)

    def height(self, xy: Tensor) -> Tensor:
        return self.z0.to(xy.dtype).expand(xy.shape[:-1]).clone()

    def height_and_slope(self, xy: Tensor) -> tuple[Tensor, Tensor]:
        return self.height(xy), torch.zeros_like(xy)


class BilinearHeightField(HeightField):
    """Learnable height field on a regular ``x``-``y`` grid, bilinearly interpolated.

    Args:
        heights: ``[ny, nx]`` node heights in metres (depth, positive down).
        origin: ``(x0, y0)`` of node ``[0, 0]``.
        spacing: ``(dx, dy)`` node spacing.
        learnable: register ``heights`` as a parameter.

    Slopes are analytic but only piecewise constant across cell boundaries, so
    the surface normal is discontinuous at cell edges.  That is fine for energy
    bookkeeping and for gradients with respect to the node heights; it does mean
    a very coarse grid produces visibly faceted reflections.
    """

    def __init__(
        self,
        heights: Tensor,
        origin: Sequence[float] = (0.0, 0.0),
        spacing: Sequence[float] = (1.0, 1.0),
        *,
        learnable: bool = True,
    ) -> None:
        super().__init__()
        h = torch.as_tensor(heights, dtype=torch.get_default_dtype())
        if h.ndim != 2 or min(h.shape) < 2:
            raise ValueError(f"heights must be [ny, nx] with >=2 nodes per axis, got {tuple(h.shape)}")
        if learnable:
            self.heights = nn.Parameter(h)
        else:
            self.register_buffer("heights", h)
        self.register_buffer("origin", torch.as_tensor(origin, dtype=h.dtype))
        self.register_buffer("spacing", torch.as_tensor(spacing, dtype=h.dtype))

    @property
    def shape(self) -> tuple[int, int]:
        ny, nx = self.heights.shape
        return int(ny), int(nx)

    def _corners(self, xy: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        ny, nx = self.shape
        origin = self.origin.to(xy.dtype)
        spacing = self.spacing.to(xy.dtype)
        n = torch.tensor([nx, ny], device=xy.device)

        u = (xy - origin) / spacing
        inside = (u >= 0) & (u <= (n - 1).to(u.dtype))
        u = u.clamp(torch.zeros_like(u), (n - 1).to(u.dtype).expand_as(u))
        i0 = u.floor().long().clamp(torch.zeros_like(n), n - 2)
        frac = (u - i0.to(u.dtype)).clamp(0.0, 1.0)

        flat = self.heights.reshape(-1)
        ix, iy = i0[..., 0], i0[..., 1]
        corners = torch.stack(
            [flat[(iy + dy_) * nx + (ix + dx_)] for dy_ in (0, 1) for dx_ in (0, 1)],
            dim=-1,
        ).reshape(*xy.shape[:-1], 2, 2)  # [..., v, u]
        return corners.to(xy.dtype), frac, inside.to(xy.dtype)

    def height(self, xy: Tensor) -> Tensor:
        return self.height_and_slope(xy)[0]

    def height_and_slope(self, xy: Tensor) -> tuple[Tensor, Tensor]:
        corners, frac, inside = self._corners(xy)
        fu, fv = frac[..., 0], frac[..., 1]
        wu = torch.stack((1 - fu, fu), dim=-1)
        wv = torch.stack((1 - fv, fv), dim=-1)
        du = torch.stack((-torch.ones_like(fu), torch.ones_like(fu)), dim=-1)

        def contract(au: Tensor, av: Tensor) -> Tensor:
            t = (corners * au[..., None, :]).sum(-1)  # [..., v]
            return (t * av).sum(-1)

        h = contract(wu, wv)
        spacing = self.spacing.to(xy.dtype)
        dhdx = contract(du, wv) / spacing[0]
        dhdy = contract(wu, du) / spacing[1]
        dh = torch.stack((dhdx, dhdy), dim=-1) * inside
        return h, dh


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
def reflect(d: Tensor, n: Tensor) -> Tensor:
    """Mirror ``d`` about the plane with unit normal ``n``: ``d - 2 (d.n) n``.

    Works for unit direction vectors and for the slowness vector alike, since
    mirroring is linear and therefore preserves ``|(xi, eta, zeta)| = 1/c``.
    """
    return d - 2.0 * (d * n).sum(-1, keepdim=True) * n


def grazing_angle(d: Tensor, n: Tensor) -> Tensor:
    """Grazing angle (rad, measured from the boundary *plane*) of ``d`` against ``n``.

    ``d`` need not be normalised.  Zero means the ray skims the boundary,
    ``pi/2`` means normal incidence.
    """
    dh = d / d.norm(dim=-1, keepdim=True).clamp_min(1e-30)
    return torch.asin(((dh * n).sum(-1)).abs().clamp(0.0, 1.0))


def find_crossing(
    p0: Tensor,
    p1: Tensor,
    boundary: HeightField,
    *,
    n_bisect: int = 12,
    n_newton: int = 1,
) -> Tensor:
    """Fraction ``t`` in ``[0, 1]`` where the chord ``p0 -> p1`` crosses ``boundary``.

    The signed distance along the chord is ``g(t) = z(t) - h(x(t), y(t))``.  We
    bracket its root with ``n_bisect`` bisection steps **under no_grad**, then
    take ``n_newton`` Newton steps *with* gradient tracking.

    That split matters.  Plain bisection returns a dyadic rational built from
    the constants ``1/2, 1/4, ...`` and is therefore *constant* with respect to
    the boundary parameters -- its gradient is identically zero, which would
    silently break bathymetry inversion.  A single Newton step from the
    bracketed point restores the correct derivative: near the root

    ``dt/dtheta = -(dg/dtheta) / (dg/dt)``

    which is exactly the implicit-function-theorem result, obtained here for
    free by letting autograd differentiate the Newton update.
    """
    delta = p1 - p0

    def g_of(t: Tensor) -> tuple[Tensor, Tensor]:
        p = p0 + t.unsqueeze(-1) * delta
        h, dh = boundary.height_and_slope(p[..., :2])
        g = p[..., 2] - h
        dgdt = delta[..., 2] - (dh * delta[..., :2]).sum(-1)
        return g, dgdt

    with torch.no_grad():
        lo = torch.zeros(p0.shape[:-1], dtype=p0.dtype, device=p0.device)
        hi = torch.ones_like(lo)
        g_lo, _ = g_of(lo)
        s_lo = torch.sign(g_lo)
        for _ in range(n_bisect):
            mid = 0.5 * (lo + hi)
            g_mid, _ = g_of(mid)
            same = torch.sign(g_mid) == s_lo
            lo = torch.where(same, mid, lo)
            hi = torch.where(same, hi, mid)
        t = 0.5 * (lo + hi)

    for _ in range(n_newton):
        g, dgdt = g_of(t)
        # Floor the magnitude but keep the sign: substituting a fixed positive
        # epsilon would flip the step direction for a negative slope, and a
        # 1e-30 denominator makes the *gradient* through this division 1e30.
        # dg/dt is normally of order the step size; it only collapses for a ray
        # running parallel to the boundary, where the crossing is genuinely
        # ill-conditioned and the clamp below is the real guard.
        safe = dgdt.abs().clamp_min(1e-9)
        step = g / torch.where(dgdt < 0, -safe, safe)
        t = (t - step).clamp(0.0, 1.0)
    return t


# --------------------------------------------------------------------------- #
# Reflection losses
# --------------------------------------------------------------------------- #
class BoundaryLoss(nn.Module):
    """Reflection loss in dB as a function of grazing angle.

    ``forward(grazing_rad) -> loss_db`` with the same shape as the input.
    Energy is multiplied by ``10 ** (-loss_db / 10)`` at each bounce.

    Losses here are frequency *independent*.  hydropt accumulates one scalar
    reflection loss per ray, which keeps path memory at ``O(rays x steps)``
    instead of ``O(rays x steps x bands)``; absorption carries all the spectral
    dependence.  A frequency-dependent reflection coefficient is a documented
    hook -- see the README.
    """

    def forward(self, grazing_rad: Tensor) -> Tensor:  # pragma: no cover - abstract
        raise NotImplementedError

    def reflection_phase(self, grazing_rad: Tensor) -> Tensor:
        """Phase shift imposed on the *pressure* by one bounce, in radians.

        Zero by default.  This is what separates an energy model from a
        coherent one: amplitude alone cannot beamform, and the two boundaries
        behave completely differently.  The sea surface is a pressure-release
        interface, so its reflection coefficient is -1 and every surface bounce
        flips the sign -- a path with an odd number of them arrives inverted.
        A seabed's phase is angle-dependent and comes out of the reflection
        coefficient itself; see :class:`RayleighBottomLoss`.
        """
        return torch.zeros_like(grazing_rad)


class ConstantLoss(BoundaryLoss):
    """Angle-independent loss of ``L`` dB per bounce.

    ``L`` is not constrained to be non-negative; a fit that drives it below zero
    is telling you the data wants gain at that boundary, which is usually a sign
    of a mis-specified scene rather than something to hide behind a clamp.
    """

    def __init__(self, loss_db: float = 1.0, *, learnable: bool = True,
                 pressure_release: bool = False) -> None:
        super().__init__()
        t = torch.as_tensor(float(loss_db))
        if learnable:
            self.loss_db = nn.Parameter(t)
        else:
            self.register_buffer("loss_db", t)
        self.pressure_release = bool(pressure_release)

    def forward(self, grazing_rad: Tensor) -> Tensor:
        return self.loss_db.to(grazing_rad.dtype).expand(grazing_rad.shape).clone()

    def reflection_phase(self, grazing_rad: Tensor) -> Tensor:
        if not self.pressure_release:
            return torch.zeros_like(grazing_rad)
        return torch.full_like(grazing_rad, math.pi)


class RayleighBottomLoss(BoundaryLoss):
    r"""Rayleigh reflection loss for a lossy fluid sediment half-space.

    With grazing angles measured from the interface, Snell's law is
    ``cos(theta2) / c2 = cos(theta1) / c1`` and the impedances are
    ``Z_i = rho_i c_i / sin(theta_i)``, giving

    .. math:: R = \frac{Z_2 - Z_1}{Z_2 + Z_1}, \qquad BL = -10 \log_{10} |R|^2

    Sediment attenuation enters as a complex sound speed
    ``c2 / (1 + i delta)`` with ``delta = alpha_lambda ln(10) / (40 pi)`` for
    ``alpha_lambda`` in dB per wavelength; this is the standard loss-tangent
    form (Jensen et al., *Computational Ocean Acoustics*, sec. 1.4).  Below the
    critical angle ``sin(theta2)`` goes imaginary and ``|R| -> 1``, so the loss
    correctly collapses to the sediment attenuation alone.

    Learnable parameters: ``rho2`` (kg/m^3), ``c2`` (m/s), ``alpha_lambda``
    (dB/wavelength).  Water density and sound speed are buffers.
    """

    def __init__(
        self,
        rho2: float = 1800.0,
        c2: float = 1700.0,
        alpha_lambda: float = 0.5,
        *,
        rho1: float = 1024.0,
        c1: float = 1500.0,
        learnable: bool = True,
    ) -> None:
        super().__init__()
        for name, v in {"rho2": rho2, "c2": c2, "alpha_lambda": alpha_lambda}.items():
            t = torch.as_tensor(float(v))
            if learnable:
                setattr(self, name, nn.Parameter(t))
            else:
                self.register_buffer(name, t)
        self.register_buffer("rho1", torch.as_tensor(float(rho1)))
        self.register_buffer("c1", torch.as_tensor(float(c1)))

    def coefficient(self, grazing_rad: Tensor) -> Tensor:
        """Complex pressure reflection coefficient ``R``."""
        dt = grazing_rad.dtype
        cdt = torch.complex128 if dt == torch.float64 else torch.complex64

        rho1 = self.rho1.to(dt)
        c1 = self.c1.to(dt)
        rho2 = self.rho2.to(dt).clamp_min(1.0)
        c2 = self.c2.to(dt).clamp_min(1.0)
        delta = self.alpha_lambda.to(dt).clamp_min(0.0) * (math.log(10.0) / (40.0 * math.pi))

        # Complex sediment sound speed: c2 / (1 + i delta).
        c2c = torch.complex(c2, torch.zeros_like(c2)).to(cdt) / torch.complex(
            torch.ones_like(delta), delta
        ).to(cdt)

        theta1 = grazing_rad.clamp_min(1e-9)
        sin1 = torch.sin(theta1).to(cdt)
        cos1 = torch.cos(theta1).to(cdt)

        cos2 = (c2c / c1.to(cdt)) * cos1
        sin2 = torch.sqrt(1.0 - cos2 * cos2)  # principal branch -> Re >= 0

        z1 = rho1.to(cdt) * c1.to(cdt) / sin1
        z2 = rho2.to(cdt) * c2c / sin2

        return (z2 - z1) / (z2 + z1)

    def forward(self, grazing_rad: Tensor) -> Tensor:
        r = self.coefficient(grazing_rad)
        r2 = (r.real**2 + r.imag**2).clamp(1e-12, 1.0)
        return -10.0 * torch.log10(r2)

    def reflection_phase(self, grazing_rad: Tensor) -> Tensor:
        """``arg(R)``.  Below the critical angle this sweeps rapidly with angle,
        which is exactly the behaviour an energy-only model throws away."""
        return torch.angle(self.coefficient(grazing_rad))
