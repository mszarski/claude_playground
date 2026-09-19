"""Ambient noise, and what it does to a beamformed image.

Every image in this package has so far been noise-free, which quietly decides
the one question a sonar is built to answer.  Without noise a target 60 dB
below the reverberation is still "visible" -- it is simply a small number in a
cell -- and detection range comes out as whatever the reverberation happens to
allow.  Real sonar stops at whichever of the two runs out first, and which one
that is depends on frequency, bandwidth, sea state and range.

**Levels here are absolute, in dB re 1 uPa.**  Arrival amplitudes elsewhere in
the package are relative to a unit source, so an image is in units of the
source's own pressure squared; :func:`calibrate` multiplies by the source level
and puts the image on the same scale as the noise.  Nothing can be said about
detection until both are on one scale.

The ambient spectrum follows the standard four-component fit to the Wenz
curves: turbulence at the bottom of the band, distant shipping through the
tens of hertz, wind-driven surface noise over the decade around 1 kHz, and
thermal noise rising at 20 dB/decade above about 50 kHz.  They are summed as
powers, not levels, because they are independent processes.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

__all__ = [
    "ambient_noise_db",
    "line_array_directivity_db",
    "beam_noise_power",
    "calibrate",
    "add_receiver_noise",
]


def ambient_noise_db(freqs_khz: Tensor | float, *, wind_speed: float = 5.0,
                     shipping: float = 0.5, turbulence: bool = True,
                     thermal: bool = True) -> Tensor:
    """Ambient spectrum level, dB re 1 uPa^2/Hz, at each frequency.

    Args:
        freqs_khz: frequencies, kHz.
        wind_speed: m/s at 10 m height.  The surface-noise term is what a sonar
            in a seaway actually hears through most of the band.
        shipping: traffic density 0 (remote) to 1 (busy lane).
        turbulence, thermal: include those terms.

    The four components, with ``f`` in kHz:

    * turbulence ``17 - 30 log f`` -- only matters below a few hertz;
    * shipping ``40 + 20(s - 0.5) + 26 log f - 60 log(f + 0.03)``;
    * wind ``50 + 7.5 sqrt(w) + 20 log f - 40 log(f + 0.4)``;
    * thermal ``-15 + 20 log f``.

    Summed as powers.  At 100 kHz the wind term gives 34 dB at 10 m/s against
    the thermal term's 25 dB, so a high-frequency imaging sonar in a seaway is
    still surface-noise limited, not thermal -- a distinction that decides
    whether raising the frequency buys anything.

    The surface term does not vanish at ``wind_speed=0``: ``50 + 7.5 sqrt(w)``
    keeps a flat-calm floor, worth 9.9 dB at 100 kHz.  That is the fit's own
    behaviour and it is left in, because a sea with no surface noise at all is
    not a sea -- but it means a "no wind" spectrum is not a pure thermal one
    until well above 100 kHz.
    """
    f = torch.as_tensor(freqs_khz, dtype=torch.get_default_dtype()).reshape(-1)
    if float(f.min()) <= 0.0:
        raise ValueError("frequencies must be positive")
    if wind_speed < 0.0:
        raise ValueError(f"wind speed must be >= 0, got {wind_speed}")
    if not 0.0 <= shipping <= 1.0:
        raise ValueError(f"shipping must be in [0, 1], got {shipping}")
    log_f = torch.log10(f)

    power = torch.zeros_like(f)
    if turbulence:
        power = power + 10.0 ** ((17.0 - 30.0 * log_f) / 10.0)
    power = power + 10.0 ** ((40.0 + 20.0 * (shipping - 0.5) + 26.0 * log_f
                              - 60.0 * torch.log10(f + 0.03)) / 10.0)
    power = power + 10.0 ** ((50.0 + 7.5 * math.sqrt(wind_speed) + 20.0 * log_f
                              - 40.0 * torch.log10(f + 0.4)) / 10.0)
    if thermal:
        power = power + 10.0 ** ((-15.0 + 20.0 * log_f) / 10.0)
    return 10.0 * torch.log10(power)


def line_array_directivity_db(n_elements: int,
                              spacing_wavelengths: float = 0.5) -> float:
    """Directivity index of a line array against isotropic noise, dB.

    ``10 log10(N)`` at half-wavelength spacing, which is the spacing that makes
    the element outputs independent in an isotropic field.  Wider spacing does
    not keep buying directivity against noise -- it buys grating lobes, which
    let the noise back in from the directions the beam is not looking.
    """
    if n_elements < 1:
        raise ValueError(f"need at least one element, got {n_elements}")
    if spacing_wavelengths <= 0.0:
        raise ValueError("spacing must be positive")
    d = 10.0 * math.log10(n_elements)
    if spacing_wavelengths > 0.5:
        # Elements further apart than lambda/2 sample a correlated field no
        # better, and the array's response repeats: cap at the lambda/2 value.
        d = 10.0 * math.log10(1.0 + (n_elements - 1) * 0.5 / spacing_wavelengths)
    return d


def beam_noise_power(freqs_khz: Tensor | float, *, bandwidth_hz: float,
                     directivity_db: float = 0.0, **ambient) -> Tensor:
    """Noise power in one beam and one range cell, linear uPa^2.

    Args:
        bandwidth_hz: receiver bandwidth.  For a pulse of length ``tau``
            matched-filtered, ``1 / tau``: a 0.12 ms pulse is 8.3 kHz, and
            every doubling of bandwidth costs 3 dB of noise.  Using the
            transducer's bandwidth instead of the pulse's is the classic way to
            be 10 dB pessimistic.
        directivity_db: the array's directivity index, from
            :func:`line_array_directivity_db`.
        ambient: passed to :func:`ambient_noise_db`.

    ``NL + 10 log10(B) - DI``, as a power rather than a level, so it can be
    added to an image.
    """
    if bandwidth_hz <= 0.0:
        raise ValueError(f"bandwidth must be positive, got {bandwidth_hz}")
    level = (ambient_noise_db(freqs_khz, **ambient)
             + 10.0 * math.log10(bandwidth_hz) - directivity_db)
    return 10.0 ** (level / 10.0)


def calibrate(image: Tensor, source_level_db: float, *,
              beam_scale: float = 1.0) -> Tensor:
    """Put a beamformed image onto an absolute scale, uPa^2.

    Two factors stand between an image and a pressure, and both are large:

    * the beamformer's own normalisation -- the coherent sum over the aperture
      and the unit-area pulse envelope, together 70 dB for a 64-element array
      and a 0.12 ms pulse.  Pass it as ``beam_scale``, from
      :func:`hydropt.beamform.beam_power_scale`;
    * the source level.  Arrival amplitudes are relative to a unit source, so
      an image is in units of the source's pressure squared at 1 m;
      ``source_level_db`` is that pressure in dB re 1 uPa at 1 m, 210-220 dB
      for an imaging sonar.

    A complex ``image`` is taken to be the beamformed FIELD (``beamform(...,
    complex_output=True)``) and is scaled by the square root of the same
    factor, so that ``|calibrate(b)|^2 == calibrate(|b|^2)``.

    Without both, comparing an image with a noise level compares a ratio with a
    pressure, and the error is not small enough to notice as a discrepancy --
    it is large enough to look like a different question's answer.
    """
    if beam_scale <= 0.0:
        raise ValueError(f"beam_scale must be positive, got {beam_scale}")
    factor = 10.0 ** (source_level_db / 10.0) / beam_scale
    if image.is_complex():
        return image * math.sqrt(factor)       # a field: the amplitude scales
    return image * factor


def add_receiver_noise(power: Tensor, noise_power: Tensor | float, *,
                       generator: torch.Generator | None = None) -> Tensor:
    """One realisation of ``|signal + noise|^2``, from the signal's power alone.

    The beamformer returns ``|b|^2``, having already thrown away the phase, and
    adding a noise *power* to it would drop the cross term between signal and
    noise -- which is not a detail: on a cell with equal signal and noise the
    cross term is the larger part of the variance, and a detector tuned on
    ``S + N`` alone reports a false-alarm rate that the real one will not meet.

    Writing the (unknown) signal phasor as real, ``|s + n|^2 = (sqrt(S) + x)^2
    + y^2`` with ``x, y ~ N(0, N/2)``.  That needs only the power and the noise
    level, gives exactly the right Rice statistics, and stays differentiable in
    the signal through ``sqrt(S)``.

    ``noise_power`` broadcasts against ``power``, so a per-band noise level can
    be applied to a ``[beams, bands, bins]`` image directly.

    **Given the field instead** -- a complex ``power``, from ``beamform(...,
    complex_output=True)`` -- the noise phasor is added to it and the result is
    ``|b + n|^2``: the same statistics, but smooth in the field.  That matters
    for a gradient.  Through the power alone the model passes through
    ``sqrt(S) = |b|``, which has a kink wherever the field goes through a null,
    and a coherent target's fringes put a null within a sixteenth of a
    wavelength of a quarter of its cells: measured on ``examples/22``, the
    analytic gradient of the noisy image was five times a wavelength-scale
    finite difference through the power and agreed with it through the field.
    """
    real_dtype = power.real.dtype if power.is_complex() else power.dtype
    n = torch.as_tensor(noise_power, dtype=real_dtype, device=power.device)
    if bool((n < 0).any()):
        raise ValueError("noise power must be non-negative")
    sigma = (0.5 * n).sqrt()
    shape = torch.broadcast_shapes(power.shape, n.shape)
    x = torch.randn(shape, dtype=real_dtype, device=power.device,
                    generator=generator) * sigma
    y = torch.randn(shape, dtype=real_dtype, device=power.device,
                    generator=generator) * sigma
    if power.is_complex():
        re, im = power.real + x, power.imag + y
        return re * re + im * im
    # Floored at the smallest normal number, not at zero.  The derivative of
    # sqrt at exactly zero is infinite, and a cell the beamformer left at
    # exactly zero (an empty bin, or a value that underflowed) then turns the
    # gradient of EVERY parameter into NaN, because 0 * inf is NaN and the
    # image's sum sees every cell.  It happened on a 90 m image with two such
    # cells.  At the floor the slope is large but finite, and it multiplies a
    # zero, so the cell contributes exactly nothing, as it should.
    tiny = torch.finfo(real_dtype).tiny
    return (power.clamp_min(tiny).sqrt() + x) ** 2 + y ** 2
