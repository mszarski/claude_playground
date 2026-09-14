r"""Rough-surface coherence loss: what a wind sea does to a specular reflection.

hydropt reflects specularly.  A height field bends the specular direction, which
is real 3-D physics and is what `examples/10` measures, but a *specular* model
says nothing about the energy a rough boundary scatters out of that direction --
and above a few kHz that is nearly all of it.

The correction is the oldest result in rough-surface scattering.  A Gaussian
height distribution of RMS ``sigma`` spreads the phase of the reflected field,
and averaging over that spread leaves a coherent (specular) reflection
coefficient reduced by the Eckart factor

.. math::
    R_{\text{coh}} = R_0 \exp\!\left(-\tfrac{1}{2}\Gamma^2\right),
    \qquad \Gamma = 2 k \sigma \sin\theta

where ``theta`` is the grazing angle and ``Gamma`` is the Rayleigh roughness
parameter -- the round-trip phase spread in radians.  In energy, and in decibels,

.. math::
    \text{loss (dB)} = \frac{10}{\ln 10}\,\Gamma^2 \approx 4.343\,\Gamma^2

so ``Gamma = 1`` costs 4.34 dB and ``Gamma = 2`` costs 17.4 dB.  It is
quadratic in frequency, in RMS height and in the sine of the grazing angle, and
the quadratic in frequency is the whole story: the same sea that is a mild
nuisance at 500 Hz annihilates the specular path at 100 kHz.

This is a loss, and the energy goes somewhere hydropt does not put it
--------------------------------------------------------------------
Applying this removes energy from the specular path and **does not re-radiate it**.
That is exactly right for a coherent calculation -- the beamformer should not see
a surface ghost that is not there -- and it is a one-sided correction for an
energy budget.  :mod:`hydropt.reverb` models boundary *backscatter* separately,
but the two are not coupled: scattering out of the specular path here does not
feed the reverberation there.  So the total energy in a scene with roughness loss
is lower than the total energy in the real one, and the difference is the
scattered field.

Frequency, and why this is a weight rather than a BoundaryLoss
--------------------------------------------------------------
:class:`hydropt.boundaries.BoundaryLoss` is deliberately frequency-independent:
the tracer accumulates one scalar reflection loss per ray, which keeps path
memory at ``O(rays x steps)`` rather than ``O(rays x steps x bands)``.  Eckart
loss is quadratic in frequency, so it cannot go there without giving that up.

:func:`roughness_weights` therefore computes it *after* the trace, from the
bounce events the trace already recorded, as a per-ray per-band factor to hand
the renderer as ``ray_weights``.  Path memory is untouched, the frequency
dependence is exact, and the cost is one pass over the bounce list.

:class:`RoughSurfaceLoss` is the drop-in alternative for a single-band scene: a
``BoundaryLoss`` evaluated at one design frequency.  It is the right tool only
when the scene really has one frequency -- across 0.5 to 4 kHz the Rayleigh
parameter varies by 8, and so the loss by 64.

References
----------
Eckart (1953); Beckmann & Spizzichino (1963), *The Scattering of Electromagnetic
Waves from Rough Surfaces*; Jensen et al., *Computational Ocean Acoustics*, sec.
1.6 (the surface-roughness reflection coefficient); Brekhovskikh & Lysanov,
*Fundamentals of Ocean Acoustics*, ch. 9.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .boundaries import BoundaryLoss, ConstantLoss, HeightField
from .tracer import TraceResult, bounce_events

__all__ = [
    "rayleigh_roughness",
    "coherent_reflection_loss_db",
    "RoughSurfaceLoss",
    "roughness_weights",
    "wind_sea_rms_height",
]

# 10 / ln(10): converts Gamma^2 into decibels of energy loss.
_DB_PER_GAMMA2 = 10.0 / math.log(10.0)


def wind_sea_rms_height(wind_speed: float) -> float:
    """RMS surface elevation of a fully-developed wind sea (m).

    ``H_s / 4`` with the Pierson-Moskowitz ``H_s = 0.22 U^2 / g``, so this is the
    same sea :func:`hydropt.environment.pierson_moskowitz_surface` generates --
    handy when you want the *loss* without generating the surface.
    """
    from .environment import significant_wave_height_pm

    return significant_wave_height_pm(wind_speed) / 4.0


def rayleigh_roughness(grazing_rad: Tensor | float, rms_height: float,
                       freq_khz: Tensor | float,
                       sound_speed: float = 1500.0) -> Tensor:
    r"""The Rayleigh roughness parameter ``Gamma = 2 k sigma sin(theta)``.

    Dimensionless, and readable as the round-trip phase spread in radians that
    the roughness imposes.  ``Gamma << 1`` is a smooth boundary; ``Gamma >~ 2``
    has essentially no coherent reflection left.
    """
    graze = torch.as_tensor(grazing_rad)
    freq = torch.as_tensor(freq_khz, dtype=graze.dtype, device=graze.device)
    k = 2.0 * math.pi * freq * 1.0e3 / float(sound_speed)
    return 2.0 * k * float(rms_height) * torch.sin(graze).abs()


def coherent_reflection_loss_db(grazing_rad: Tensor | float, rms_height: float,
                                freq_khz: Tensor | float,
                                sound_speed: float = 1500.0) -> Tensor:
    r"""Eckart coherent-reflection loss in dB, ``(10/ln 10) Gamma^2``.

    Returns energy loss, so it adds to a ``BoundaryLoss``.  Zero at grazing
    incidence and zero for a smooth boundary, both exactly: a ray running along
    the boundary sees no height variation along its own direction of travel, and
    a flat boundary has none to see.

    At large ``Gamma`` the loss is enormous and the corresponding energy factor
    ``10^(-dB/10)`` underflows to exactly zero in double precision.  That is the
    correct answer -- there is no coherent reflection -- not a numerical failure,
    but it does mean a dB plot of it will run off the bottom of any axis.
    """
    gamma = rayleigh_roughness(grazing_rad, rms_height, freq_khz, sound_speed)
    return _DB_PER_GAMMA2 * gamma * gamma


class RoughSurfaceLoss(BoundaryLoss):
    """A smooth boundary loss plus Eckart roughness at one design frequency.

    A drop-in :class:`hydropt.boundaries.BoundaryLoss`, so it works anywhere one
    does -- at the price of a single frequency, because the base class is
    frequency-independent by design.  For a multi-band scene use
    :func:`roughness_weights` instead, which gets the frequency dependence right.

    Args:
        rms_height: RMS boundary elevation ``sigma`` (m).  Learnable, so surface
            roughness or seabed relief can be *fitted* from reflection strength.
        freq_khz: the design frequency.  Not learnable -- it is a property of the
            scene, not of the boundary.
        smooth: the underlying loss, applied on top.  Defaults to a lossless
            boundary, which is what a pressure-release sea surface is *before*
            roughness -- and after it, at any useful frequency, is not.
        sound_speed: for the wavenumber (m/s).
        pressure_release: forwarded to the default ``smooth`` loss; ignored when
            ``smooth`` is given.

    Example:
        >>> import torch
        >>> from hydropt.rough import RoughSurfaceLoss, wind_sea_rms_height
        >>> loss = RoughSurfaceLoss(wind_sea_rms_height(10.0), freq_khz=2.0,
        ...                         pressure_release=True)
        >>> float(loss(torch.tensor([0.5])))  # 0.5 rad grazing  # doctest: +SKIP
    """

    def __init__(self, rms_height: float, freq_khz: float, *,
                 smooth: BoundaryLoss | None = None,
                 sound_speed: float = 1500.0,
                 learnable: bool = True,
                 pressure_release: bool = False) -> None:
        super().__init__()
        sigma = torch.as_tensor(float(rms_height))
        if learnable:
            self.rms_height = nn.Parameter(sigma)
        else:
            self.register_buffer("rms_height", sigma)
        self.register_buffer("freq_khz", torch.as_tensor(float(freq_khz)))
        self.register_buffer("sound_speed", torch.as_tensor(float(sound_speed)))
        self.smooth = smooth if smooth is not None else ConstantLoss(
            0.0, learnable=False, pressure_release=pressure_release)

    def gamma(self, grazing_rad: Tensor) -> Tensor:
        """The Rayleigh parameter at these grazing angles."""
        k = (2.0 * math.pi * self.freq_khz.to(grazing_rad.dtype) * 1.0e3
             / self.sound_speed.to(grazing_rad.dtype))
        return 2.0 * k * self.rms_height.to(grazing_rad.dtype).abs() * torch.sin(
            grazing_rad).abs()

    def forward(self, grazing_rad: Tensor) -> Tensor:
        g = self.gamma(grazing_rad)
        return self.smooth(grazing_rad) + _DB_PER_GAMMA2 * g * g

    def reflection_phase(self, grazing_rad: Tensor) -> Tensor:
        """Unchanged by roughness.

        Eckart averaging reduces the coherent field's *amplitude* and leaves the
        mean phase where it was; the phase spread it describes is what has been
        averaged away, not a shift to be applied.
        """
        return self.smooth.reflection_phase(grazing_rad)

    def extra_repr(self) -> str:
        return (f"rms_height={float(self.rms_height):.4g} m, "
                f"freq={float(self.freq_khz):.4g} kHz")


def roughness_weights(
    result: TraceResult,
    freqs_khz: Tensor,
    *,
    surface_rms: float = 0.0,
    bottom_rms: float = 0.0,
    surface: HeightField | None = None,
    bottom: HeightField | None = None,
    sound_speed: float = 1500.0,
    min_grazing: float = 1e-9,
) -> Tensor:
    r"""Per-ray, per-band coherent-reflection factor from a bundle's bounces.

    The frequency-correct route: every bounce the trace recorded contributes
    ``exp(-Gamma^2)`` in energy at each band, and the factors multiply along the
    ray.  Returns ``[R, B]``, ready to pass as ``ray_weights`` to
    :func:`hydropt.receiver.splat_etc` or
    :func:`hydropt.beamform.extract_arrivals`, both of which accept a per-band
    weight for exactly this.

    Nothing about the trace changes, so this costs one pass over the bounce list
    and no extra path memory -- which is the reason it lives here rather than in
    a :class:`hydropt.boundaries.BoundaryLoss`.

    Args:
        result: a traced bundle.
        freqs_khz: ``[B]`` band centres.
        surface_rms, bottom_rms: RMS elevation of each boundary (m).  Zero leaves
            that boundary smooth, so a scene with a rough sea over a flat seabed
            passes ``surface_rms`` alone.
        surface, bottom: the height fields, forwarded to
            :func:`hydropt.tracer.bounce_events` so it can refine each bounce back
            onto the boundary.  Without them the grazing angles are the recorded
            ones, which are biased by up to one step.
        sound_speed: for the wavenumber (m/s).

    Returns:
        ``[R, B]`` energy factors in ``(0, 1]``, differentiable in nothing by
        default -- pass learnable RMS values through
        :func:`coherent_reflection_loss_db` yourself if you want to fit them.

    A ray with no bounces gets exactly 1, which is the identity, so this is safe
    to apply to a bundle whose direct paths carry the signal.
    """
    dtype, device = result.pos.dtype, result.pos.device
    freqs = freqs_khz.to(dtype=dtype, device=device).reshape(-1)
    n_rays, n_band = result.n_rays, int(freqs.shape[0])
    if surface_rms <= 0.0 and bottom_rms <= 0.0:
        return torch.ones(n_rays, n_band, dtype=dtype, device=device)

    events = bounce_events(result, surface=surface, bottom=bottom,
                           min_grazing=min_grazing)
    if events.count == 0:
        return torch.ones(n_rays, n_band, dtype=dtype, device=device)

    k = 2.0 * math.pi * freqs * 1.0e3 / float(sound_speed)  # [B]
    sin_theta = torch.sin(events.grazing.to(dtype)).abs()  # [E]
    rms = torch.where(events.is_bottom,
                      torch.full_like(sin_theta, float(bottom_rms)),
                      torch.full_like(sin_theta, float(surface_rms)))
    gamma = 2.0 * k.reshape(1, -1) * rms.reshape(-1, 1) * sin_theta.reshape(-1, 1)
    # Accumulated in the log, so the product along a ray is a sum: that avoids
    # multiplying many already-tiny factors, and the single exp at the end
    # underflows cleanly to zero for a very rough boundary -- which is the
    # physical answer (no coherent reflection) rather than a failure.
    log_factor = -(gamma * gamma)  # [E, B]
    total = torch.zeros(n_rays, n_band, dtype=dtype, device=device)
    total.index_add_(0, events.ray.to(device), log_factor)
    return torch.exp(total)
