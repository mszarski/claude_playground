"""Active sonar: two-way propagation through a scattering target.

A passive scene renders one path, source to receiver.  An active sonar renders
two: projector to target, then target to receive array.  For a monostatic
system the array sits beside the projector, so the one-way response is just the
outgoing pulse -- the target is what makes the problem interesting.

Composition without a render per arrival
----------------------------------------
The obvious implementation, relaunching a fan for every incident arrival, is
quadratic and unnecessary.  For a target whose scattering does not depend on
which incident path delivered the energy, the two legs are *separable*: the
echo is the time convolution of the inbound channel response with the outbound
one, scaled by the scattering cross-section.

.. math::
    E_{\\text{echo}}(t) = \\sigma \\int E_{\\text{in}}(u)\\,E_{\\text{out}}(t-u)\\,du

So the whole two-way problem costs exactly **two** renders, whatever the
multipath complexity.  Spreading and absorption compose correctly on their own:
the inbound response already carries ``1/s_1^2`` and ``10^(-alpha s_1/10)``, the
outbound ``1/s_2^2`` and ``10^(-alpha s_2/10)``, and their product is the
two-way law over the total path.

The kernel widths compose too.  Each leg is rendered with ``sigma_t / sqrt(2)``
so that convolving the two unit-area Gaussians reproduces exactly the requested
``sigma_t`` in the echo, rather than smearing it by another factor of root two.

What this assumes
-----------------
Scattering is isotropic -- one cross-section, no aspect dependence.  An
aspect-dependent target breaks the separability that makes the convolution
valid, because the outbound amplitude would then depend on the inbound
direction; the honest treatment is a render per incident arrival, or a
factorised pattern applied to the outbound fan for a dominant incident
direction.  See :class:`PointTarget` for where that would attach.

Echoes here are *energy*, summed incoherently.  Coherent echoes -- the ones a
beamformer consumes -- are built from arrival lists instead; see
:mod:`hydropt.beamform`.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import torch
from torch import Tensor, nn

from .launch import fibonacci_cone, fibonacci_sphere
from .receiver import splat_etc
from .scene import Scene
from .tracer import trace

__all__ = ["PointTarget", "EchoResult", "return_fan", "render_echo"]


class PointTarget(nn.Module):
    """An isotropic point scatterer with a learnable position and strength.

    Args:
        position: ``(x, y, z)`` in metres.
        target_strength_db: target strength ``TS``; the scattering
            cross-section used is ``sigma = 10 ** (TS / 10)``.  Real targets
            run from about -40 dB (a small fish) to +10 dB or more (a
            broadside submarine hull).
        learnable: register both as parameters.

    Aspect dependence would attach here as a ``pattern(incident, scattered)``
    method; note that using it invalidates the convolution shortcut in
    :func:`render_echo` -- see the module docstring.
    """

    def __init__(
        self,
        position: tuple[float, float, float] | Tensor,
        target_strength_db: float = -10.0,
        *,
        learnable: bool = True,
    ) -> None:
        super().__init__()
        pos = torch.as_tensor(position, dtype=torch.get_default_dtype()).reshape(3)
        ts = torch.as_tensor(float(target_strength_db))
        if learnable:
            self.position = nn.Parameter(pos)
            self.target_strength_db = nn.Parameter(ts)
        else:
            self.register_buffer("position", pos)
            self.register_buffer("target_strength_db", ts)

    def cross_section(self) -> Tensor:
        """Scattering cross-section ``sigma = 10 ** (TS / 10)``."""
        return 10.0 ** (self.target_strength_db / 10.0)

    def extra_repr(self) -> str:
        return (f"position={self.position.tolist()}, "
                f"TS={float(self.target_strength_db):.1f} dB")


class _RelocatedScene:
    """A read-only view of a scene with the source moved elsewhere.

    :func:`hydropt.tracer.trace` only ever reads attributes off the scene, so a
    proxy is enough and avoids rebuilding an ``nn.Module`` -- which matters,
    because the target position is usually a :class:`torch.nn.Parameter` and
    rebuilding would either detach it or smuggle a grad-requiring tensor into a
    buffer.  Every sub-module is shared by reference, so gradients reach the
    field, boundaries and losses exactly as they would through the original.
    """

    def __init__(self, scene: Scene, source: Tensor) -> None:
        object.__setattr__(self, "_scene", scene)
        object.__setattr__(self, "_source", source)

    def __getattr__(self, name: str):
        return getattr(self._scene, name)

    def source_position(self) -> Tensor:
        return self._source


class EchoResult(NamedTuple):
    """Output of :func:`render_echo`."""

    etc: Tensor  # [receivers, bands, echo_bins] two-way energy response
    inbound: Tensor  # [1, bands, leg_bins] projector -> target
    outbound: Tensor  # [receivers, bands, leg_bins] target -> array
    leg_time_grid: Tensor  # [leg_bins] one-way time base of the two legs


def return_fan(
    target: PointTarget,
    receivers: Tensor,
    n_rays: int = 3000,
    *,
    half_angle_deg: float = 45.0,
    n_background: int = 0,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Launch directions from a target towards a receive array.

    The array subtends a tiny solid angle from any useful range, so an
    isotropic fan from the target wastes almost every ray.  The cone must still
    be generous: the surface- and bottom-reflected returns leave the target at
    angles well away from the direct bearing, and a tight cone silently drops
    exactly the multipath that carries the target's depth.
    """
    axis = receivers.reshape(-1, 3).mean(0) - target.position.detach().reshape(3)
    dirs = fibonacci_cone(n_rays, axis, half_angle_deg, generator=generator)
    if n_background > 0:
        dirs = torch.cat((dirs, fibonacci_sphere(
            n_background, generator=generator, dtype=dirs.dtype, device=dirs.device)), 0)
    return dirs


def render_echo(
    scene: Scene,
    target: PointTarget,
    tx_directions: Tensor,
    rx_directions: Tensor,
    echo_time_grid: Tensor,
    *,
    sigma_d: float,
    sigma_t: float,
    tx_weights: Tensor | None = None,
    rx_weights: Tensor | None = None,
    ray_chunk: int = 0,
    trace_kwargs: dict | None = None,
    **splat_kwargs,
) -> EchoResult:
    """Render the two-way echo from ``target`` onto the scene's receive array.

    Args:
        scene: the scene; its ``source`` is the projector and its ``receivers``
            the receive array.
        target: the scatterer.
        tx_directions: projector launch directions, ``[Nt, 3]``.  Apply the
            transmit beam pattern through ``tx_weights``, not by narrowing this
            fan, so that sidelobe illumination is still modelled.
        rx_directions: launch directions from the target, ``[Nr, 3]``; see
            :func:`return_fan`.
        echo_time_grid: uniform two-way time grid (s).
        sigma_d, sigma_t: kernel widths for the echo.  Each leg is rendered at
            ``sigma_t / sqrt(2)`` so the convolution lands on ``sigma_t``.
        tx_weights, rx_weights: per-ray weights, e.g. transmit directivity.
        ray_chunk, trace_kwargs, splat_kwargs: forwarded to the tracer/splatter.

    Returns:
        :class:`EchoResult`.  Differentiable in the target position and
        strength, and in every scene parameter, through both legs.
    """
    grid = echo_time_grid
    n_echo = int(grid.shape[0])
    if n_echo < 2:
        raise ValueError("echo_time_grid needs at least two bins")
    dt = (grid[-1] - grid[0]) / (n_echo - 1)
    dt_f = float(dt)
    if float(grid[0]) < 0.0:
        raise ValueError("echo_time_grid must start at a non-negative time")

    # Each leg is one-way, so it only has to span the full echo window; the
    # convolution then covers everything up to twice that.
    n_leg = int(math.ceil(float(grid[-1]) / dt_f)) + 1
    leg_grid = torch.arange(n_leg, dtype=grid.dtype, device=grid.device) * dt
    sigma_leg = sigma_t / math.sqrt(2.0)
    kw = dict(sigma_d=sigma_d, sigma_t=sigma_leg, ray_chunk=ray_chunk, **splat_kwargs)

    # Leg 1: projector -> target.  The scene is unchanged; the target simply
    # stands in as the receiver, which keeps its position differentiable.
    inbound = splat_etc(
        trace(scene, tx_directions, **(trace_kwargs or {})),
        target.position.reshape(1, 3), leg_grid, scene.freqs_khz,
        ray_weights=tx_weights, **kw,
    )

    # Leg 2: target -> array, traced from a scene view whose source is the target.
    outbound = splat_etc(
        trace(_RelocatedScene(scene, target.position), rx_directions, **(trace_kwargs or {})),
        scene.receivers, leg_grid, scene.freqs_khz,
        ray_weights=rx_weights, **kw,
    )

    # Full linear convolution along time, done by FFT so it stays one op and
    # differentiable.  Zero-padding to >= 2n-1 is what makes it linear rather
    # than circular; getting that wrong wraps late echoes onto early ones.
    n_fft = 1 << int(math.ceil(math.log2(max(2 * n_leg - 1, 2))))
    spec = torch.fft.rfft(inbound, n=n_fft) * torch.fft.rfft(outbound, n=n_fft)
    full = torch.fft.irfft(spec, n=n_fft)[..., : 2 * n_leg - 1]
    # dt turns the discrete sum into the continuous convolution integral.
    full = full * dt * target.cross_section()

    offset = int(round(float(grid[0]) / dt_f))
    if offset + n_echo > full.shape[-1]:
        pad = offset + n_echo - full.shape[-1]
        full = torch.nn.functional.pad(full, (0, pad))
    etc = full[..., offset : offset + n_echo]
    return EchoResult(etc=etc, inbound=inbound, outbound=outbound, leg_time_grid=leg_grid)
