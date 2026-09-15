"""Differentiable energy-time-curve (ETC) splatting and receiver arrays.

There are no hard hit tests anywhere in hydropt.  A discrete "did this ray hit
the receiver" predicate has zero gradient almost everywhere and an undefined one
on a measure-zero set, which is useless for optimisation.  Instead a ray that
*passes near* a receiver deposits energy through a pair of Gaussian kernels: one
in closest-approach distance, one in arrival time.

Where the energy is deposited
-----------------------------
Each ray is a polyline.  For every segment the closest approach to a receiver is
available in closed form (a clamped projection), giving a miss distance ``d``, an
arrival time and a path length at that point.  hydropt then reduces those
per-segment values along the ray with one of three ``mode`` settings:

``"local_min"`` (default)
    Every *local minimum* of ``d`` along the polyline becomes one arrival.  A
    ray that swings past the receiver twice -- once before a bottom bounce and
    once after -- correctly produces two arrivals.
``"global_min"``
    Only the single closest approach on the whole ray.  Cheaper, and fine when
    each launch angle can only ever pass a receiver once.
``"line_integral"``
    Every segment splats, weighted by its length.  This is the smoothest
    objective of the three, but **it destroys time resolution**: segments up to
    ``space_gate * sigma_d`` away all contribute, and their closest-approach
    times span the whole passage.  At 3 km range with ``sigma_d = 40 m`` that is
    ~300 ms of smear against ~5 ms of true arrival separation.  Use it only for
    a deliberately blurred objective, never to read arrival structure off.

Reducing to local minima is what decouples the two kernels: ``sigma_d`` then
controls *amplitude acceptance only*, and the arrival time stays as sharp as the
integrator.

Kernel widths and the bias/variance trade-off
---------------------------------------------
``sigma_d`` (m) sets how far from a receiver a ray may pass and still be heard.
Large ``sigma_d`` biases levels high -- rays that physically miss still
contribute, and a geometric shadow fills in -- but widens the basin of
attraction enormously, because a scene parameter only has a gradient if *some*
ray currently passes within a few ``sigma_d`` of a receiver.  Small ``sigma_d``
approaches the true geometric answer but the gradient vanishes until the fan is
already nearly right.  The recipe used by :func:`hydropt.inverse.fit` is to
anneal: start wide enough that the initial guess produces a signal, then shrink.

``sigma_t`` (s) is the receiver's impulse response -- it is what gives an ETC
bin a finite value.  Set it to a few time-grid bins; below one bin the splat
aliases.  Unlike the line-integral mode, it does not need to exceed the
per-step travel time.

Sparsification
--------------
Contributions are gated at ``space_gate * sigma_d`` and ``time_gate * sigma_t``.
At the defaults (6 and 5 sigma) discarded terms are below ``exp(-18) ~ 1e-8`` of
the peak, so this is sparsification rather than approximation -- but it does
make the gradient *exactly* zero for rays currently far from every receiver,
which is the mechanism behind the annealing advice above.

Absolute level, and a trap in it
-------------------------------
Each ray carries unit energy scaled by ``spreading`` (``1/s^2`` by default), and
the acceptance kernel acts as an aperture of effective area ``2 pi sigma_d^2``.
An earlier version of this note said ETC magnitudes are "calibrated only up to a
scale factor".  That was wrong, and the error mattered: the aperture subtends a
solid angle that *itself* shrinks as ``1/s^2``, so a dense-fan sum applies
geometric spreading **twice** and the energy falls as ``1/R^4``.  The
mis-calibration is a factor of ``s^2``, not a constant.

So for an absolute or range-dependent level, pass ``ray_weights`` (per-ray solid
angle ``cos(e) de da``) **and** unit ``spreading``, letting the ray *count* inside
the acceptance supply the spreading -- which it does exactly.  Calibrated that
way against free space, eigenray energies match the exact image-source solution
to five or six figures; see ``tests/test_pekeris.py`` and
``scripts/validate_pekeris.py``.

Passing ``spreading=ray_tube(...).spreading`` double-counts in the same way, for
the same reason: in a dense-fan sum, refractive focusing is already carried by
where the rays land.  The ray tube is the right tool for an *eigenray* treatment,
which this renderer is not.

Ratios of two renders made the same way -- an inversion's prediction against its
synthetic measurement, or one spreading law against another -- are unaffected,
which is why this went unnoticed for so long.
"""

from __future__ import annotations

import math
from typing import Callable, Literal

import torch
from torch import Tensor

from .absorption import thorp_db_per_km
from .tracer import TraceResult

__all__ = [
    "vertical_line_array",
    "horizontal_line_array",
    "make_time_grid",
    "splat_etc",
]

SplatMode = Literal["local_min", "global_min", "line_integral"]

# Floor on squared lengths before sqrt, to keep the backward pass finite.
_LEN_EPS = 1e-12


def vertical_line_array(x: float, y: float, z_top: float, z_bottom: float, n: int,
                        *, dtype: torch.dtype | None = None,
                        device: torch.device | str | None = None) -> Tensor:
    """Vertical line array (VLA) of ``n`` elements, ``[n, 3]``."""
    dtype = dtype or torch.get_default_dtype()
    z = torch.linspace(float(z_top), float(z_bottom), n, dtype=dtype, device=device)
    return torch.stack((torch.full_like(z, float(x)), torch.full_like(z, float(y)), z), dim=-1)


def horizontal_line_array(x0: float, y0: float, x1: float, y1: float, z: float, n: int,
                          *, dtype: torch.dtype | None = None,
                          device: torch.device | str | None = None) -> Tensor:
    """Horizontal line array (HLA) of ``n`` elements at constant depth, ``[n, 3]``."""
    dtype = dtype or torch.get_default_dtype()
    t = torch.linspace(0.0, 1.0, n, dtype=dtype, device=device)
    return torch.stack(
        (float(x0) + t * (float(x1) - float(x0)),
         float(y0) + t * (float(y1) - float(y0)),
         torch.full_like(t, float(z))),
        dim=-1,
    )


def make_time_grid(t_start: float, t_end: float, n_bins: int, *,
                   dtype: torch.dtype | None = None,
                   device: torch.device | str | None = None) -> Tensor:
    """Uniform time grid, ``[n_bins]``.  Uniformity is assumed by :func:`splat_etc`."""
    dtype = dtype or torch.get_default_dtype()
    return torch.linspace(float(t_start), float(t_end), n_bins, dtype=dtype, device=device)


def _closest_approach(
    p0: Tensor, seg: Tensor, seg_len2: Tensor, receivers: Tensor
) -> tuple[Tensor, Tensor]:
    """Per-segment closest approach to each receiver.

    Returns ``(tstar, dist)``, both ``[Rc, S, Nr]``, where ``tstar`` is the
    clamped position along the segment.
    """
    n_recv = int(receivers.shape[0])
    rel = receivers.view(1, 1, n_recv, 3) - p0.unsqueeze(2)  # [Rc, S, Nr, 3]
    tstar = (rel * seg.unsqueeze(2)).sum(-1) / seg_len2.unsqueeze(2).clamp_min(_LEN_EPS)
    tstar = tstar.clamp(0.0, 1.0)
    delta = rel - tstar.unsqueeze(-1) * seg.unsqueeze(2)
    # Not delta.norm(): its backward is 0/0 at zero distance, which a ray passing
    # exactly through a receiver would hit.
    return tstar, (delta * delta).sum(-1).clamp_min(_LEN_EPS).sqrt()


def splat_etc(
    result: TraceResult,
    receivers: Tensor,
    time_grid: Tensor,
    freqs_khz: Tensor,
    *,
    sigma_d: float,
    sigma_t: float,
    mode: SplatMode = "local_min",
    absorption: Callable[[Tensor], Tensor] = thorp_db_per_km,
    ray_weights: Tensor | None = None,
    source_energy: Tensor | float = 1.0,
    spreading: Tensor | None = None,
    spread_min_range: float = 1.0,
    space_gate: float = 6.0,
    time_gate: float = 5.0,
    ray_chunk: int = 0,
) -> Tensor:
    """Splat a traced bundle onto an energy-time curve, ``[receivers, bands, time_bins]``.

    Each retained arrival contributes

    ``E(s, band) * exp(-d^2 / 2 sigma_d^2) * N(tau; t, sigma_t)``

    where ``N`` is a unit-area Gaussian on the time grid and

    ``E = 1/s^2 * 10^(-L_refl/10) * 10^(-alpha(f) s / 10)``

    combines spherical spreading, accumulated reflection loss and volume
    absorption.

    Args:
        result: output of :func:`hydropt.tracer.trace`.
        receivers: ``[Nr, 3]`` receiver positions -- a whole array is handled in
            this one batched call.
        time_grid: ``[T]`` uniform time grid (s).
        freqs_khz: ``[B]`` band centre frequencies (kHz).
        sigma_d, sigma_t: kernel widths (m, s); see the module docstring.
        mode: how per-segment closest approaches are reduced along each ray.
        absorption: ``f_khz -> dB/km``.
        ray_weights: optional per-ray weight, ``[R]`` (e.g. solid angle) or
            ``[R, B]`` for a weight that differs by band -- which is how
            frequency-dependent boundary effects get in, since
            :class:`hydropt.boundaries.BoundaryLoss` is frequency-independent by
            design.  See :func:`hydropt.rough.roughness_weights`.
        spreading: optional ``[R, S+1]`` intensity factor replacing ``1/s^2``,
            as produced by :func:`hydropt.spreading.ray_tube`.  ``1/s^2`` is
            exact only in a homogeneous medium; in a refracting channel it can
            be tens of dB wrong at a convergence zone.
        source_energy: scalar or ``[B]`` source level multiplier.
        spread_min_range: floor on ``s`` in ``1/s^2``, keeping the near field finite.
        space_gate, time_gate: sparsification cut-offs, in units of sigma.
        ray_chunk: process rays in chunks of this size to bound peak memory
            (0 = all at once).  Mathematically identical either way.

    Returns:
        ``[Nr, B, T]`` energy density (per second) on the time grid.
    """
    if mode not in ("local_min", "global_min", "line_integral"):
        raise ValueError(f"unknown mode {mode!r}")

    pos, tau, arclen = result.pos, result.tau, result.arclen
    refl_db, alive = result.refl_db, result.alive
    dtype, device = pos.dtype, pos.device

    receivers = receivers.to(dtype=dtype, device=device)
    time_grid = time_grid.to(dtype=dtype, device=device)
    freqs_khz = freqs_khz.to(dtype=dtype, device=device)

    n_recv = int(receivers.shape[0])
    n_band = int(freqs_khz.shape[0])
    n_time = int(time_grid.shape[0])
    if n_time < 2:
        raise ValueError("time_grid needs at least two bins")

    t0 = time_grid[0]
    dt = (time_grid[-1] - time_grid[0]) / (n_time - 1)
    alpha = absorption(freqs_khz)  # [B] dB/km

    half_w = int(math.ceil(time_gate * sigma_t / float(dt)))
    offsets = torch.arange(-half_w, half_w + 1, device=device)
    band_ix = torch.arange(n_band, device=device)
    norm_t = 1.0 / (math.sqrt(2.0 * math.pi) * sigma_t)

    n_rays = int(pos.shape[0])
    chunk = n_rays if ray_chunk <= 0 else int(ray_chunk)
    etc_flat = torch.zeros(n_recv * n_band * n_time, dtype=dtype, device=device)

    for lo in range(0, n_rays, chunk):
        hi = min(lo + chunk, n_rays)
        p0, p1 = pos[lo:hi, :-1], pos[lo:hi, 1:]  # [Rc, S, 3]
        seg = p1 - p0
        seg_len2 = (seg * seg).sum(-1)
        # Retired rays are frozen in place and so have zero-length segments.
        # sqrt(0) has an infinite derivative, and although those segments are
        # masked out below, autograd still evaluates sqrt's backward on them as
        # 0/0 = NaN -- which then poisons every parameter gradient.  The floor is
        # 1e-12 m^2 against real segments of tens of metres.
        seg_len = seg_len2.clamp_min(_LEN_EPS).sqrt()

        tstar, dist = _closest_approach(p0, seg, seg_len2, receivers)  # [Rc, S, Nr]
        live = (alive[lo:hi, :-1] * alive[lo:hi, 1:]) > 0  # [Rc, S]
        usable = (seg_len > 0).unsqueeze(2) & live.unsqueeze(2)
        near = (dist < space_gate * sigma_d) & usable

        if mode == "line_integral":
            keep = near
        elif mode == "global_min":
            masked = torch.where(near, dist.detach(), torch.full_like(dist, float("inf")))
            best = masked.argmin(dim=1, keepdim=True)  # [Rc, 1, Nr]
            keep = torch.zeros_like(near).scatter(1, best, True) & near
        else:  # local_min
            big = torch.full_like(dist[:, :1], float("inf"))
            d_pad = torch.where(usable, dist.detach(), torch.full_like(dist, float("inf")))
            prev = torch.cat((big, d_pad[:, :-1]), dim=1)
            nxt = torch.cat((d_pad[:, 1:], big), dim=1)
            keep = (d_pad <= prev) & (d_pad < nxt) & near

        if not bool(keep.any()):
            continue

        ri, si, qi = keep.nonzero(as_tuple=True)
        d_sel = dist[ri, si, qi]
        t_sel = tstar[ri, si, qi]
        len_sel = seg_len[ri, si]

        tau_a, tau_b = tau[lo:hi][ri, si], tau[lo:hi][ri, si + 1]
        tau_c = tau_a + t_sel * (tau_b - tau_a)
        s_c = arclen[lo:hi][ri, si] + t_sel * len_sel
        db_c = refl_db[lo:hi][ri, si + 1]

        w_space = torch.exp(-0.5 * (d_sel / sigma_d) ** 2)
        if mode == "line_integral":
            # Quadrature of the line integral, normalised so a ray passing
            # straight through the receiver contributes unit weight.
            w_space = w_space * len_sel / (math.sqrt(2.0 * math.pi) * sigma_d)
        band_weight = None
        if ray_weights is not None:
            rw = ray_weights.to(dtype=dtype, device=device)
            if rw.ndim == 1:
                w_space = w_space * rw[lo:hi][ri]
            elif rw.ndim == 2:
                if rw.shape[1] != n_band:
                    raise ValueError(
                        f"per-band ray_weights has {rw.shape[1]} bands, "
                        f"expected {n_band}")
                # Held back until the band axis exists, below.
                band_weight = rw[lo:hi][ri]
            else:
                raise ValueError(
                    f"ray_weights must be [R] or [R, B], got {tuple(rw.shape)}")

        if spreading is None:
            s_eff = s_c.clamp_min(spread_min_range)
            spread = 1.0 / (s_eff * s_eff)
        else:
            sp = spreading.to(dtype=dtype, device=device)[lo:hi]
            # Linear across one step: spreading varies by O(2h/s) over a step,
            # so this is far below the discretisation already in the path.
            spread = sp[ri, si] + t_sel * (sp[ri, si + 1] - sp[ri, si])
        refl = 10.0 ** (-db_c / 10.0)
        # alpha [dB/km] * s [m] / 1000 -> dB, then energy factor 10^(-dB/10).
        absorb = 10.0 ** (-(alpha.view(1, n_band) * s_c.unsqueeze(1)) / 1.0e4)
        amp = (w_space * spread * refl).unsqueeze(1) * absorb  # [n, B]
        if band_weight is not None:
            amp = amp * band_weight
        if not isinstance(source_energy, (int, float)) or source_energy != 1.0:
            amp = amp * torch.as_tensor(source_energy, dtype=dtype, device=device)

        # Windowed Gaussian scatter in time.
        centre = torch.round((tau_c.detach() - t0) / dt).long()
        bins = centre.unsqueeze(1) + offsets.view(1, -1)  # [n, W]
        valid = (bins >= 0) & (bins < n_time)
        bins_c = bins.clamp(0, n_time - 1)
        t_at = t0 + bins_c.to(dtype) * dt
        kern = torch.exp(-0.5 * ((tau_c.unsqueeze(1) - t_at) / sigma_t) ** 2) * norm_t
        kern = kern * valid.to(dtype)

        vals = amp.unsqueeze(2) * kern.unsqueeze(1)  # [n, B, W]
        flat_ix = (qi.view(-1, 1, 1) * n_band + band_ix.view(1, -1, 1)) * n_time + bins_c.unsqueeze(1)
        etc_flat = etc_flat + torch.zeros_like(etc_flat).index_add(
            0, flat_ix.reshape(-1), vals.reshape(-1)
        )

    return etc_flat.view(n_recv, n_band, n_time)
