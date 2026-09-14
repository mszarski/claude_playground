"""Coherent arrivals and array beamforming.

An energy response cannot be beamformed.  Conventional beamforming sums
*complex pressure* across elements, ``b(s) = sum_m w_m p_m(t - tau_m(s))``;
doing the same on energy throws away the array gain, the nulls and the
sidelobe structure -- everything a beamformer exists for.  This module is the
coherent path: it pulls an arrival list out of a traced bundle, synthesises the
complex field across an aperture, and beamforms it.

Why not splat coherently to each element
----------------------------------------
The obvious approach -- splat complex pressure to every element the way
:mod:`hydropt.receiver` splats energy -- does not survive contact with a ray
fan.  Each arrival's time carries a discretisation error because the nearest
launch direction is not exactly the eigenray: about 80 us at 3 km for a
600-ray fan, and 13 us at 40 m for a 4000-ray cone.  Against a 100 kHz
quarter-wave period of 2.5 us that is several whole cycles of phase error, so
independently-splatted elements would be mutually decorrelated and beamforming
would return noise.

What beamforming actually consumes is not absolute phase but *relative* phase
across the aperture, and that is far better conditioned: at half-wave spacing
adjacent elements are struck by essentially the same ray, so their phase
difference is set by geometry rather than by which ray happened to be nearest.

So hydropt extracts arrivals at a single **phase centre** and propagates each
one across the aperture analytically, as a plane wave travelling along the
measured arrival direction:

.. math::
    \\tau_m = \\tau_0 + \\frac{(\\mathbf{r}_m - \\mathbf{r}_0)\\cdot\\hat{k}}{c}

The differential delays are then exact to the plane-wave approximation, which
is excellent for an aperture small against the range, regardless of the
absolute timing error.  Absolute phase remains unreliable, so do not use these
outputs for anything that compares phase *between* pings without first adding
ray-tube interpolation (see the README).

Steering is applied as a true time delay, folded into each arrival before
synthesis, not as a narrowband phase shift.  That matters more than it sounds:
a 32-element half-wave array at 100 kHz is 15.5 wavelengths long, so the delay
across it is ~155 us -- many times a short pulse's envelope width.  Phase-only
steering would align the carriers while leaving the envelopes scattered, and
the beam would collapse even when pointed correctly.
"""

from __future__ import annotations

import math
from typing import Literal, NamedTuple

import torch
from torch import Tensor

from .absorption import thorp_db_per_km
from .receiver import _closest_approach
from .tracer import TraceResult

__all__ = [
    "ArrivalSet",
    "extract_arrivals",
    "shading_window",
    "azimuth_steering",
    "element_field",
    "beamform",
]

WindowKind = Literal["uniform", "hann", "hamming", "blackman"]


class ArrivalSet(NamedTuple):
    """Coherent arrivals at one point.  ``A`` arrivals, ``B`` frequency bands."""

    time: Tensor  # [A] arrival time (s)
    amplitude: Tensor  # [A, B] pressure amplitude (sqrt of energy)
    direction: Tensor  # [A, 3] unit direction of propagation at arrival
    phase: Tensor  # [A] accumulated boundary phase (rad)
    distance: Tensor  # [A] miss distance of the contributing ray (m)
    path_length: Tensor  # [A] path length at closest approach (m)

    @property
    def n_arrivals(self) -> int:
        # Deliberately not __len__: NamedTuple._make and _replace check len()
        # against the field count, so overriding it silently breaks _replace.
        return int(self.time.shape[0])


def extract_arrivals(
    result: TraceResult,
    point: Tensor,
    freqs_khz: Tensor,
    *,
    sigma_d: float,
    absorption=thorp_db_per_km,
    ray_weights: Tensor | None = None,
    spread_min_range: float = 1.0,
    space_gate: float = 6.0,
    max_arrivals: int | None = None,
) -> ArrivalSet:
    """Pull coherent arrivals at ``point`` out of a traced bundle.

    Uses the same local-minimum reduction as :func:`hydropt.receiver.splat_etc`
    -- every local minimum of the miss distance along a ray is one arrival -- but
    returns the arrivals themselves rather than splatting them, so phase and
    arrival direction survive.

    The arrival direction is taken from the ray's own segment at closest
    approach, which needs no extra storage in :class:`TraceResult`.

    Args:
        result: a traced bundle.
        point: ``[3]`` -- normally the array phase centre.
        freqs_khz: ``[B]`` band centres.
        sigma_d: acceptance width (m); sets the amplitude weight, as in the
            energy renderer.
        max_arrivals: keep only this many strongest arrivals.  Useful because
            a dense fan produces one arrival per ray that passes nearby, and
            for a coherent sum the near-duplicates add nothing but cost.

    Returns:
        :class:`ArrivalSet`, differentiable in every scene parameter.
    """
    pos, tau, arclen = result.pos, result.tau, result.arclen
    dtype, device = pos.dtype, pos.device
    point = point.reshape(1, 1, 3).to(dtype=dtype, device=device)
    freqs_khz = freqs_khz.to(dtype=dtype, device=device)

    p0, p1 = pos[:, :-1], pos[:, 1:]
    seg = p1 - p0
    seg_len2 = (seg * seg).sum(-1)
    seg_len = seg_len2.clamp_min(1e-12).sqrt()

    tstar, dist = _closest_approach(p0, seg, seg_len2, point.reshape(1, 3))
    tstar, dist = tstar[..., 0], dist[..., 0]  # single point

    live = (result.alive[:, :-1] * result.alive[:, 1:]) > 0
    usable = (seg_len > 0) & live
    big = torch.full_like(dist[:, :1], float("inf"))
    d_pad = torch.where(usable, dist.detach(), torch.full_like(dist, float("inf")))
    prev = torch.cat((big, d_pad[:, :-1]), dim=1)
    nxt = torch.cat((d_pad[:, 1:], big), dim=1)
    keep = (d_pad <= prev) & (d_pad < nxt) & (dist < space_gate * sigma_d) & usable

    ri, si = keep.nonzero(as_tuple=True)
    if ri.numel() == 0:
        z = torch.zeros(0, dtype=dtype, device=device)
        return ArrivalSet(z, torch.zeros(0, freqs_khz.shape[0], dtype=dtype, device=device),
                          torch.zeros(0, 3, dtype=dtype, device=device), z, z, z)

    t_sel = tstar[ri, si]
    d_sel = dist[ri, si]
    len_sel = seg_len[ri, si]
    tau_a, tau_b = tau[ri, si], tau[ri, si + 1]
    time = tau_a + t_sel * (tau_b - tau_a)
    s_c = arclen[ri, si] + t_sel * len_sel
    db_c = result.refl_db[ri, si + 1]
    phase = result.refl_phase[ri, si + 1]
    direction = seg[ri, si] / len_sel.unsqueeze(-1)

    weight = torch.exp(-0.5 * (d_sel / sigma_d) ** 2)
    if ray_weights is not None:
        weight = weight * ray_weights.to(dtype=dtype, device=device)[ri]
    s_eff = s_c.clamp_min(spread_min_range)
    alpha = absorption(freqs_khz).view(1, -1)
    energy = ((weight * (1.0 / (s_eff * s_eff)) * 10.0 ** (-db_c / 10.0)).unsqueeze(1)
              * 10.0 ** (-(alpha * s_c.unsqueeze(1)) / 1.0e4))
    amplitude = energy.clamp_min(0.0).sqrt()

    if max_arrivals is not None and time.shape[0] > max_arrivals:
        order = amplitude.detach().sum(1).argsort(descending=True)[:max_arrivals]
        order = order[time.detach()[order].argsort()]
        time, amplitude = time[order], amplitude[order]
        direction, phase = direction[order], phase[order]
        d_sel, s_c = d_sel[order], s_c[order]

    return ArrivalSet(time=time, amplitude=amplitude, direction=direction,
                      phase=phase, distance=d_sel, path_length=s_c)


def shading_window(n: int, kind: WindowKind = "uniform", *,
                   dtype: torch.dtype | None = None,
                   device: torch.device | str | None = None) -> Tensor:
    """Amplitude shading weights, ``[n]``, normalised to unit sum.

    Uniform shading gives the narrowest mainlobe and the worst sidelobes
    (-13.2 dB for a line array); tapering trades one for the other.
    """
    dtype = dtype or torch.get_default_dtype()
    if kind == "uniform":
        w = torch.ones(n, dtype=dtype, device=device)
    elif kind == "hann":
        w = torch.hann_window(n, periodic=False, dtype=dtype, device=device)
    elif kind == "hamming":
        w = torch.hamming_window(n, periodic=False, dtype=dtype, device=device)
    elif kind == "blackman":
        w = torch.blackman_window(n, periodic=False, dtype=dtype, device=device)
    else:
        raise ValueError(f"unknown shading window {kind!r}")
    return w / w.sum()


def azimuth_steering(n: int, half_sector_deg: float = 60.0, elevation_deg: float = 0.0,
                     *, dtype: torch.dtype | None = None,
                     device: torch.device | str | None = None) -> tuple[Tensor, Tensor]:
    """``(directions, angles_deg)`` for a horizontal beam sweep about +x."""
    dtype = dtype or torch.get_default_dtype()
    ang = torch.linspace(-half_sector_deg, half_sector_deg, n, dtype=dtype, device=device)
    a = ang * math.pi / 180.0
    e = torch.full_like(a, elevation_deg * math.pi / 180.0)
    dirs = torch.stack((torch.cos(e) * torch.cos(a), torch.cos(e) * torch.sin(a),
                        torch.sin(e)), dim=-1)
    return dirs, ang


def _synthesise(
    arrivals: ArrivalSet,
    delays: Tensor,  # [..., A] extra delay per arrival for each output channel
    weights: Tensor,  # [...] complex or real weight per channel-arrival
    freqs_khz: Tensor,
    time_grid: Tensor,
    sigma_t: float,
    time_gate: float,
) -> Tensor:
    """Common complex splatting kernel.  Returns ``[..., B, T]``."""
    dtype = time_grid.dtype
    device = time_grid.device
    n_time = int(time_grid.shape[0])
    t0 = time_grid[0]
    dt = (time_grid[-1] - time_grid[0]) / (n_time - 1)
    half_w = int(math.ceil(time_gate * sigma_t / float(dt)))
    offsets = torch.arange(-half_w, half_w + 1, device=device)
    norm_t = 1.0 / (math.sqrt(2.0 * math.pi) * sigma_t)

    tau = arrivals.time.reshape(*([1] * (delays.ndim - 1)), -1) + delays  # [..., A]
    omega = 2.0 * math.pi * freqs_khz * 1.0e3  # [B] rad/s

    centre = torch.round((tau.detach() - t0) / dt).long()
    bins = centre.unsqueeze(-1) + offsets  # [..., A, W]
    valid = (bins >= 0) & (bins < n_time)
    bins_c = bins.clamp(0, n_time - 1)
    env = torch.exp(-0.5 * ((tau.unsqueeze(-1) - (t0 + bins_c.to(dtype) * dt)) / sigma_t) ** 2)
    env = env * norm_t * valid.to(dtype)

    # Carrier phase: -omega * tau, plus the boundary phase the path accumulated.
    ph = arrivals.phase.reshape(*([1] * (delays.ndim - 1)), -1)
    arg = ph.unsqueeze(-2) - omega.view(-1, 1) * tau.unsqueeze(-2)  # [..., B, A]
    amp = arrivals.amplitude.T.reshape(*([1] * (delays.ndim - 1)), *arrivals.amplitude.T.shape)
    carrier = torch.polar(amp * weights.unsqueeze(-2), arg)  # [..., B, A]

    vals = carrier.unsqueeze(-1) * env.unsqueeze(-3).to(carrier.dtype)  # [..., B, A, W]

    lead = delays.shape[:-1]
    n_lead = int(torch.tensor(lead).prod()) if lead else 1
    n_band = int(freqs_khz.shape[0])
    flat_bins = bins_c.reshape(n_lead, 1, -1).expand(n_lead, n_band, -1)
    chan = torch.arange(n_lead, device=device).view(-1, 1, 1) * n_band
    band = torch.arange(n_band, device=device).view(1, -1, 1)
    flat_ix = ((chan + band) * n_time + flat_bins).reshape(-1)

    out = torch.zeros(n_lead * n_band * n_time, dtype=vals.dtype, device=device)
    out = out.index_add(0, flat_ix, vals.reshape(n_lead, n_band, -1).reshape(-1))
    return out.view(*lead, n_band, n_time)


def element_field(
    arrivals: ArrivalSet,
    elements: Tensor,
    freqs_khz: Tensor,
    time_grid: Tensor,
    *,
    sigma_t: float,
    sound_speed: float = 1500.0,
    phase_centre: Tensor | None = None,
    time_gate: float = 5.0,
) -> Tensor:
    """Complex pressure at each array element, ``[elements, bands, time_bins]``.

    Each arrival is propagated across the aperture as a plane wave along its own
    measured direction; see the module docstring for why this, rather than
    independently splatting to each element.
    """
    elements = elements.reshape(-1, 3).to(dtype=time_grid.dtype, device=time_grid.device)
    centre = elements.mean(0) if phase_centre is None else phase_centre.reshape(3)
    offset = elements - centre  # [M, 3]
    delays = (offset @ arrivals.direction.T) / sound_speed  # [M, A]
    weights = torch.ones(elements.shape[0], arrivals.direction.shape[0],
                         dtype=time_grid.dtype, device=time_grid.device)
    return _synthesise(arrivals, delays, weights, freqs_khz, time_grid, sigma_t, time_gate)


def beamform(
    arrivals: ArrivalSet,
    elements: Tensor,
    freqs_khz: Tensor,
    time_grid: Tensor,
    steer_directions: Tensor,
    *,
    sigma_t: float,
    sound_speed: float = 1500.0,
    shading: Tensor | None = None,
    phase_centre: Tensor | None = None,
    time_gate: float = 5.0,
    steer_chunk: int = 0,
) -> Tensor:
    """Delay-and-sum beam power, ``[steer_directions, bands, time_bins]``.

    ``steer_directions`` are **look** directions: unit vectors pointing from
    the array towards where you are listening, which is the convention a sonar
    operator expects.  A wave propagating along ``k`` came *from* ``-k``, so the
    effective delay at element ``m`` is ``(r_m - r_0) . (k + l) / c`` and
    vanishes for every element when the look direction ``l`` is ``-k`` -- which
    is what makes the beam sum coherently on target.  For a forward-looking
    sonar this means sweeping look directions about ``+x`` while the echoes
    themselves arrive travelling roughly along ``-x``.

    Steering is a true time delay folded into each arrival before synthesis, so
    both the carrier and the envelope are aligned.

    Args:
        shading: ``[elements]`` amplitude weights; uniform if omitted.
        steer_chunk: process steering directions in chunks to bound memory.

    Returns:
        Real beam power ``|b|^2``.  Differentiable in element positions,
        shading weights and every scene parameter behind ``arrivals``.
    """
    dtype, device = time_grid.dtype, time_grid.device
    elements = elements.reshape(-1, 3).to(dtype=dtype, device=device)
    steer = steer_directions.reshape(-1, 3).to(dtype=dtype, device=device)
    steer = steer / steer.norm(dim=-1, keepdim=True).clamp_min(1e-30)
    centre = elements.mean(0) if phase_centre is None else phase_centre.reshape(3)
    offset = elements - centre  # [M, 3]
    w = (shading_window(elements.shape[0], "uniform", dtype=dtype, device=device)
         if shading is None else shading.to(dtype=dtype, device=device))

    n_steer = int(steer.shape[0])
    chunk = n_steer if steer_chunk <= 0 else int(steer_chunk)
    out = []
    for lo in range(0, n_steer, chunk):
        sv = steer[lo : lo + chunk]  # [S, 3]
        # [S, M, A] effective delay, then sum the element axis coherently.
        rel = arrivals.direction.view(1, 1, -1, 3) + sv.view(-1, 1, 1, 3)
        delays = (offset.view(1, -1, 1, 3) * rel).sum(-1) / sound_speed
        weights = w.view(1, -1, 1).expand_as(delays)
        field = _synthesise(arrivals, delays, weights, freqs_khz, time_grid,
                            sigma_t, time_gate)  # [S, M, B, T]
        b = field.sum(dim=1)  # coherent sum across the aperture
        out.append(b.real**2 + b.imag**2)
    return torch.cat(out, dim=0)
