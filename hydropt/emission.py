"""Emission: sound a vessel radiates, heard by the array as band-limited noise.

Everything else in a picture is an echo of the sonar's own pulse, so it sits
in one range cell per path.  A propeller radiates continuously, so what the
array hears from it is in EVERY range cell of the ping at a level set by the
one-way path, and the beamformer puts it on the emitter's bearing at every
range: a spoke (``examples/24`` measures one, and shows that a glint does
not draw one).  The range-varying gain of a display then makes the spoke
brighten with range, since the reverberation it competes with falls off and
the emission does not.

:func:`emission_arrivals` builds that signal as an :class:`ArrivalSet` the
beamformer already knows how to form: a train of pulses one receiver
bandwidth wide, a pulse-width apart across the whole time grid, with
independent random phases, on every one-way path from the emitter to the
array's centre by the method of images (the sound speed must be constant;
the paths carry the boundaries' losses and the absorption).  Paths whose
first metres cross the emitter's own hull are removed (the hull shadows
its propeller bow-on), by :func:`~hydropt.mesh.segment_mesh_transmission`.

What a propeller radiates is not the same in every direction, and the
difference is what makes a spoke come and go as a vessel turns.  Cavitation
noise is generated at the stern, so the hull shields it forward (a bow
aspect reads 10-20 dB below the beam and quarters in measured ship
signatures), while dead astern the propeller is heard through its own
bubble wake, which takes a few dB off over a narrow cone.
:func:`propeller_directivity` is that pattern, an amplitude weight on each
path's launch direction: full level on the quarters, ``bow_db`` down at the
bow with a smooth ``(1 - cos)`` fall between, and a ``wake_db`` notch
``wake_half_deg`` wide astern.  Passed as ``pattern`` it multiplies the
binary shadow of the hull mesh, which is exact for a path that starts into
the hull but says nothing about the paths that clear it.

The level is set so that the received band level comes out right after the
beamformer and :func:`~hydropt.noise.calibrate`: a train of pulses with
unit-area envelope and random phase has mean power ``a^2 10^(SL/10)
sqrt(pi)`` per path, so the amplitude that puts the received level at
``(emission level - path loss)`` is the path's own pressure ratio scaled from
the sonar's source level to the emission's, over ``pi^(1/4)``.  The emission
level is given as a spectrum level (dB re 1 uPa^2/Hz at 1 m) and the
receiver's bandwidth ``1 / pulse_s`` turns it into a band level.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from .beamform import ArrivalSet
from .eigenray import image_arrivals_batched

__all__ = ["emission_arrivals", "propeller_directivity"]


def emission_arrivals(scene, emitter: Tensor, array_centre: Tensor, time_grid: Tensor, *,
                      spectrum_level_db: float, source_level_db: float, pulse_s: float,
                      rx_pattern=None, pattern=None, occluder=None, shadow_reach: float = 35.0,
                      generator: torch.Generator | None = None) -> tuple[ArrivalSet | None, int, int]:
    """A continuous emitter's arrivals at the array, as a random-phase pulse train.

    Args:
        scene: the :class:`~hydropt.scene.Scene`; its profile must be
            isovelocity (the paths are by the method of images).
        emitter: ``[3]`` world position of what radiates (a propeller).
        array_centre: ``[3]`` the receive array's centre.
        time_grid: the beamformer's ``[T]`` grid; the train spans it with a
            margin of three pulses either side.
        spectrum_level_db: radiated noise, dB re 1 uPa^2/Hz at 1 m, at the
            scene's frequency.
        source_level_db: the sonar's, as passed to :func:`calibrate`, so
            the emission lands at its own level after calibration.
        pulse_s: the sonar's pulse length: the receiver's bandwidth is its
            inverse, and the pulses are spaced by it.
        rx_pattern: the element's directivity as ``f(arrival directions)``.
        pattern: the emitter's own directivity as ``f(launch directions) ->
            amplitude weight`` (:func:`propeller_directivity`); ``None`` is
            omnidirectional.
        occluder: ``(world_vertices, faces)`` of the emitter's own hull, or
            ``None``; a path whose first ``shadow_reach`` metres cross it is
            dropped.
        generator: for the phases.

    Returns:
        ``(arrivals, n_clear, n_paths)``: the arrival set (``None`` when the
        hull shadows every path), how many paths were kept, and how many
        the channel had.
    """
    emitter = emitter.reshape(1, 3)
    paths = image_arrivals_batched(scene, emitter, array_centre.reshape(1, 3),
                                   scene.freqs_khz)[0]
    amp = paths.amplitude
    if rx_pattern is not None:
        amp = amp * rx_pattern(paths.direction).reshape(-1, 1)
    if pattern is not None:
        amp = amp * pattern(paths.launch_direction).reshape(-1, 1)
    if occluder is not None:
        from .mesh import segment_mesh_transmission
        verts, faces = occluder
        starts = emitter.expand(paths.n_arrivals, 3)
        ends = starts + float(shadow_reach) * paths.launch_direction
        amp = amp * segment_mesh_transmission(starts, ends, verts, faces).reshape(-1, 1)
    keep = (amp.detach().max(dim=1).values > 0.0).nonzero().reshape(-1)
    n_clear, n_paths = int(keep.numel()), int(paths.n_arrivals)
    if n_clear == 0:
        return None, 0, n_paths
    amp = amp[keep]
    band_db = float(spectrum_level_db) + 10.0 * math.log10(1.0 / float(pulse_s))
    level = 10.0 ** ((band_db - float(source_level_db)) / 20.0) / math.pi ** 0.25
    spacing = float(pulse_s)
    t = torch.arange(float(time_grid[0]) - 3 * spacing, float(time_grid[-1]) + 3 * spacing,
                     spacing, dtype=time_grid.dtype)
    n_t, n_p = int(t.shape[0]), n_clear
    phase = 2.0 * math.pi * torch.rand(n_t * n_p, generator=generator, dtype=time_grid.dtype)
    rep = lambda x: x[keep].repeat(n_t, *([1] * (x.ndim - 1)))
    arrivals = ArrivalSet(
        time=t.repeat_interleave(n_p),
        amplitude=(amp * level).repeat(n_t, 1),
        direction=rep(paths.direction), phase=phase,
        distance=rep(paths.distance), path_length=rep(paths.path_length),
        launch_direction=rep(paths.launch_direction))
    return arrivals, n_clear, n_paths


def propeller_directivity(heading_deg: float, *, bow_db: float = 20.0, wake_db: float = 6.0,
                          wake_half_deg: float = 15.0):
    """The amplitude pattern of a propeller's noise about its vessel's heading.

    Returns ``f(directions [..., 3]) -> [...]``, an amplitude weight on the
    direction a path LEAVES the propeller in.  With ``theta`` the horizontal
    angle between that direction and the stern (``theta = 0`` is a path going
    straight astern, ``180`` straight ahead through the hull):

    ``dB = -bow_db * (1 - cos theta) / 2 - wake_db * 2^(-(theta / wake_half_deg)^2)``

    so a quarter (``theta = 45``) is 3 dB down on ``bow_db = 20``, the beam
    10 dB down, the bow 20 dB down, and the wake's notch takes ``wake_db``
    off dead astern, half of it ``wake_half_deg`` away.  A path that leaves
    steeply up or down keeps only its horizontal aspect.
    """
    h = math.radians(float(heading_deg))
    stern = torch.tensor([-math.cos(h), -math.sin(h)])

    def pattern(directions: Tensor) -> Tensor:
        d = directions[..., :2]
        n = d.norm(dim=-1).clamp_min(1e-9)
        cos_t = (d @ stern.to(dtype=directions.dtype)) / n
        cos_t = cos_t.clamp(-1.0, 1.0)
        theta_deg = torch.rad2deg(torch.acos(cos_t))
        db = (-float(bow_db) * 0.5 * (1.0 - cos_t)
              - float(wake_db) * torch.pow(2.0, -(theta_deg / float(wake_half_deg)) ** 2))
        return torch.pow(10.0, db / 20.0)

    return pattern
