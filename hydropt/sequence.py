"""Sequences: a target moved along a trajectory, and a picture rendered per pose.

The picture pipeline of ``examples/21`` -- trace, reverberation, target
echoes, beamformer, calibration, receiver noise, display, resampling to
metres -- is linear in the arrivals up to the noise, so a moving target does
not need the scene redone: the static background's complex beams are formed
once and the target's beams are added to them per pose.  That is what
``examples/22`` does for a fit, and what a sequence needs for every frame.
This module makes it a reusable pair:

* :class:`Trajectory`: poses ``(x, y, heading)`` against time, interpolated
  between samples (headings unwrapped, so a turn through north interpolates
  the short way), or built from waypoints and a speed.
* :class:`PictureRenderer`: everything about the sonar and the scene that a
  frame needs, the background cached on first use, ``picture(targets)`` for
  any set of targets, and ``sequence(builder, trajectory, times)`` for the
  frames of a moving one.  A vessel that radiates as it moves (a spoke,
  ``examples/24``) is an ``emitter``: a callable of the pose returning
  arrivals (:func:`~hydropt.emission.emission_arrivals`) that are formed
  with the echoes, coherently, in every frame.

The other way round -- the sonar moving through a world of stationary
things -- is the same machinery with the frame turned inside out.  The
sonar stays where ``examples/21`` built it, at its frame's origin looking
along ``+x``, and each frame re-expresses the world in that frame: every
target's world pose becomes a pose relative to the sonar
(:func:`relative_pose`), and the world's sea and seabed are resampled onto
the sonar-frame grids the scene was built with
(:func:`reframe_height_field`) so that the scene can be rebuilt and
re-traced there.  ``ownship_sequence`` does this per frame: the background
is formed again for every pose (the sea scrolls under a moving sonar, and
a rough seabed's texture with it; keeping one background would freeze the
sea to the sonar), and the targets are rebuilt at their relative poses.  A
frame then costs a whole picture, not an echo; ``examples/29`` measures it.

The display (gain, floor) and the resampling to metres are the example's
own conventions and are passed in as callables, so the renderer knows only
``[beams, bands, bins]`` images until they are applied.  Receiver noise is
drawn afresh for every frame from a seed and the frame index, which is what
a sequence of pings has: the same sea, a different noise.
"""

from __future__ import annotations

import math
from typing import Callable, Iterable, Sequence

import torch
from torch import Tensor

from .active import target_arrivals
from .boundaries import BilinearHeightField
from .beamform import ArrivalSet, beamform
from .labels import Label, label_from_beams, world_geometry
from .noise import add_receiver_noise, calibrate
from .reverb import reverberation_arrivals
from .tracer import trace

__all__ = ["Trajectory", "PictureRenderer", "relative_pose", "reframe_height_field"]


class Trajectory:
    """Poses ``(x, y, heading_deg)`` against time, interpolated between samples.

    Args:
        times: ``[N]`` seconds, increasing.
        positions: ``[N, 2]`` metres, ``x`` forward and ``y`` to port.
        headings_deg: ``[N]`` degrees, the body's ``+x`` from the world's,
            positive to port (as ``mesh_target``'s ``yaw``).

    Headings are unwrapped before interpolation, so a sample at 350 followed
    by one at 10 turns 20 degrees, not 340.  ``at(t)`` clamps to the ends.
    """

    def __init__(self, times, positions, headings_deg) -> None:
        dt = torch.get_default_dtype()
        self.times = torch.as_tensor(times, dtype=dt).reshape(-1)
        self.positions = torch.as_tensor(positions, dtype=dt).reshape(-1, 2)
        h = torch.as_tensor(headings_deg, dtype=dt).reshape(-1)
        if self.times.numel() != self.positions.shape[0] or h.numel() != self.times.numel():
            raise ValueError("times, positions and headings must have one entry per sample")
        if self.times.numel() < 1:
            raise ValueError("a trajectory needs at least one sample")
        if bool((self.times[1:] <= self.times[:-1]).any()):
            raise ValueError("times must increase")
        # unwrap: every step brought into (-180, 180]
        steps = (h[1:] - h[:-1] + 180.0) % 360.0 - 180.0
        self.headings_deg = torch.cat([h[:1], h[:1] + steps.cumsum(0)])

    @classmethod
    def from_waypoints(cls, waypoints, speed: float, *, start_time: float = 0.0,
                       headings_deg=None) -> "Trajectory":
        """A track through waypoints at constant speed; headings along the track.

        ``headings_deg`` overrides the track direction (a vessel crabbing in
        a current, or holding a heading while drifting).
        """
        dt = torch.get_default_dtype()
        p = torch.as_tensor(waypoints, dtype=dt).reshape(-1, 2)
        if p.shape[0] < 2:
            raise ValueError("a track needs at least two waypoints")
        if speed <= 0.0:
            raise ValueError(f"speed must be positive, got {speed}")
        seg = p[1:] - p[:-1]
        length = seg.norm(dim=-1)
        times = start_time + torch.cat([torch.zeros(1, dtype=dt), length.cumsum(0)]) / speed
        if headings_deg is None:
            along = torch.rad2deg(torch.atan2(seg[:, 1], seg[:, 0]))
            headings = torch.cat([along[:1], along])        # the heading INTO each waypoint
        else:
            headings = torch.as_tensor(headings_deg, dtype=dt).reshape(-1)
        return cls(times, p, headings)

    @property
    def duration(self) -> float:
        return float(self.times[-1] - self.times[0])

    def at(self, t: float) -> tuple[float, float, float]:
        """``(x, y, heading_deg)`` at time ``t``, clamped to the trajectory's span."""
        times = self.times
        if times.numel() == 1 or t <= float(times[0]):
            return float(self.positions[0, 0]), float(self.positions[0, 1]), float(self.headings_deg[0])
        if t >= float(times[-1]):
            return (float(self.positions[-1, 0]), float(self.positions[-1, 1]),
                    float(self.headings_deg[-1]))
        i = int(torch.searchsorted(times, torch.tensor(t, dtype=times.dtype))) - 1
        i = max(0, min(i, times.numel() - 2))
        f = (t - float(times[i])) / float(times[i + 1] - times[i])
        p = self.positions[i] + f * (self.positions[i + 1] - self.positions[i])
        h = float(self.headings_deg[i] + f * (self.headings_deg[i + 1] - self.headings_deg[i]))
        return float(p[0]), float(p[1]), h

    def poses(self, times: Iterable[float]) -> list[tuple[float, float, float]]:
        return [self.at(float(t)) for t in times]


def relative_pose(world_pose, ownship_pose) -> tuple[float, float, float]:
    """A world pose ``(x, y, heading_deg)`` as seen from the ownship's frame.

    The ownship at ``(x_o, y_o)`` heading ``h_o`` looks along its own ``+x``:
    the world is shifted by ``-(x_o, y_o)`` and turned by ``-h_o``, and a
    target's heading is measured from the ownship's.
    """
    xw, yw, hw = (float(v) for v in world_pose)
    xo, yo, ho = (float(v) for v in ownship_pose)
    c, s = math.cos(math.radians(ho)), math.sin(math.radians(ho))
    dx, dy = xw - xo, yw - yo
    return c * dx + s * dy, -s * dx + c * dy, hw - ho


def reframe_height_field(field, *, shape, spacing, origin, x: float, y: float,
                         heading_deg: float) -> BilinearHeightField:
    """The world's height field sampled onto a sonar-frame grid at an ownship pose.

    ``shape`` (``ny, nx``), ``spacing`` and ``origin`` are the grid the scene
    was built with in the sonar's frame; each of its nodes is carried into
    the world by the ownship pose and the world ``field`` read there.  The
    result is a fixed (non-learnable) :class:`BilinearHeightField` in the
    sonar's frame, bilinear in a bilinear field: a little smoother than the
    world's between its nodes, which at eight nodes per wavelength of a
    wind sea is a few percent of its RMS height.  Nodes that fall outside
    the world field are clamped to its edge, as the tracer clamps a ray, so
    the world must cover every pose's swath.
    """
    ny, nx = (int(v) for v in shape)
    dt = torch.get_default_dtype()
    xs = float(origin[0]) + float(spacing[0]) * torch.arange(nx, dtype=dt)
    ys = float(origin[1]) + float(spacing[1]) * torch.arange(ny, dtype=dt)
    Y, X = torch.meshgrid(ys, xs, indexing="ij")
    c, s = math.cos(math.radians(heading_deg)), math.sin(math.radians(heading_deg))
    wx = float(x) + c * X - s * Y
    wy = float(y) + s * X + c * Y
    with torch.no_grad():
        h = field.height(torch.stack((wx, wy), dim=-1).reshape(-1, 2)).reshape(ny, nx)
    return BilinearHeightField(h, origin=tuple(float(v) for v in origin),
                               spacing=tuple(float(v) for v in spacing), learnable=False)


class PictureRenderer:
    """One sonar, one scene, a background formed once, a picture per set of targets.

    Args:
        scene: the :class:`~hydropt.scene.Scene` (sound speed, boundaries,
            losses, source).
        elements: ``[M, 3]`` receive element positions.
        directions: ``[R, 3]`` transmit fan directions, with ``tx_weights``
            their pattern weights and ``tx_pattern`` the same pattern as a
            function of direction (the eigenray return leg needs it).
        rx_pattern: the receive element's directivity, ``f(directions)``.
        time_grid, steer, sigma_t, shading: the beamformer's.
        source_level_db, noise_power: for :func:`calibrate` and
            :func:`add_receiver_noise` (the noise in the same units).
        scattering, solid_angle_per_ray, boundary, occluders, surface_gain,
            max_arrivals: passed to :func:`reverberation_arrivals`.
        display: ``f(noisy) -> shown``, the example's gain and floor on a
            ``[beams, bands, bins]`` image; ``None`` shows the calibrated
            image as it is.
        to_cartesian: ``f(shown) -> (cart, gx, gy)``, the resampling to
            metres; ``None`` returns the polar image.
        target_kwargs: passed to :func:`target_arrivals` (the return leg,
            ray counts, caps).
        seed: reverberation and target rays are drawn from it; the receiver
            noise from ``seed + 2 + frame``.
        steer_chunk: the beamformer's.
        beam_deg, sound_speed, label_margin_db: for the labels
            (:mod:`hydropt.labels`): the beam's half-power width (by default
            ``101.5 / n`` degrees, 1.3 times that under a shading window),
            the sound speed that turns the time grid into range, and the
            margin a target must stand over everything else in a cell to
            be in its mask.
    """

    def __init__(self, scene, *, elements: Tensor, directions: Tensor, tx_weights: Tensor,
                 tx_pattern: Callable, rx_pattern: Callable, time_grid: Tensor,
                 steer: Tensor, sigma_t: float, shading: Tensor | None,
                 source_level_db: float, noise_power: float,
                 scattering, solid_angle_per_ray: float, boundary: str = "both",
                 occluders=None, surface_gain=None, max_arrivals: int | None = None,
                 display: Callable | None = None, to_cartesian: Callable | None = None,
                 target_kwargs: dict | None = None, seed: int = 0,
                 steer_chunk: int = 8, beam_deg: float | None = None,
                 sound_speed: float = 1500.0, label_margin_db: float = 3.0) -> None:
        from .beamform import beam_power_scale, shading_window
        self.scene = scene
        self.elements = elements
        self.directions = directions
        self.tx_weights = tx_weights
        self.tx_pattern = tx_pattern
        self.rx_pattern = rx_pattern
        self.time_grid = time_grid
        self.steer = steer
        self.sigma_t = float(sigma_t)
        self.shading = shading
        self.source_level_db = float(source_level_db)
        self.noise_power = float(noise_power)
        self.scattering = scattering
        self.solid_angle_per_ray = float(solid_angle_per_ray)
        self.boundary = boundary
        self.occluders = occluders
        self.surface_gain = surface_gain
        self.max_arrivals = max_arrivals
        self.display = display
        self.to_cartesian = to_cartesian
        self.target_kwargs = dict(return_leg="eigenray", n_rx_rays=2000,
                                  rx_half_angle_deg=45.0, max_arrivals_per_leg=24)
        self.target_kwargs.update(target_kwargs or {})
        self.seed = int(seed)
        self.steer_chunk = int(steer_chunk)
        n = int(elements.reshape(-1, 3).shape[0])
        # beamform's default is UNIT-SUM uniform weights, not ones: the scale
        # must be of the weights it actually uses, or a picture without
        # shading reads 20 log10(n) dB low after calibrate()
        w = shading if shading is not None else shading_window(n, "uniform", dtype=time_grid.dtype)
        self.beam_scale = beam_power_scale(w, self.sigma_t)
        self._background: Tensor | None = None
        self.n_reverberation = 0
        self.beam_deg = float(beam_deg) if beam_deg is not None else (
            101.5 / n * (1.3 if shading is not None else 1.0))
        self.sound_speed = float(sound_speed)
        self.range_cell_m = 2.355 * self.sigma_t * self.sound_speed / 2.0   # the pulse's FWHM
        self.label_margin_db = float(label_margin_db)
        self.bearings_deg = torch.rad2deg(torch.atan2(steer[:, 1], steer[:, 0]))
        self.ranges_m = time_grid * self.sound_speed / 2.0

    def set_scene(self, scene, background: Tensor | None = None) -> None:
        """A new scene (the world re-expressed at another ownship pose).

        The cached background goes with it -- unless ``background`` is the
        complex beams of this scene's reverberation, formed earlier by
        :meth:`background` and kept by the caller (a pose visited again in
        another scenario costs its echoes and no trace).
        """
        self.scene = scene
        self._background = background

    def beams(self, arrivals: ArrivalSet) -> Tensor:
        """The complex beams of an arrival set."""
        return beamform(arrivals, self.elements, self.scene.freqs_khz, self.time_grid,
                        self.steer, sigma_t=self.sigma_t, shading=self.shading,
                        steer_chunk=self.steer_chunk, complex_output=True)

    def background(self) -> Tensor:
        """The complex beams of the reverberation: traced and formed once, then kept."""
        if self._background is None:
            with torch.no_grad():
                result = trace(self.scene, self.directions)
                rev = reverberation_arrivals(
                    result, self.directions, self.scene.freqs_khz,
                    scattering=self.scattering,
                    solid_angle_per_ray=self.solid_angle_per_ray,
                    ray_weights=self.tx_weights * self.rx_pattern(-self.directions),
                    boundary=self.boundary, surface=self.scene.surface,
                    bottom=self.scene.bottom, max_arrivals=self.max_arrivals,
                    occluders=self.occluders, surface_gain=self.surface_gain,
                    generator=torch.Generator().manual_seed(self.seed + 1))
                self.n_reverberation = rev.n_arrivals
                self._background = self.beams(rev)
        return self._background

    def echo(self, target) -> ArrivalSet:
        return target_arrivals(self.scene, target, self.directions,
                               tx_weights=self.tx_weights, tx_pattern=self.tx_pattern,
                               rx_pattern=self.rx_pattern,
                               generator=torch.Generator().manual_seed(self.seed),
                               **self.target_kwargs)

    def picture(self, targets: Sequence, *, extra_arrivals: Sequence = (),
                frame: int = 0, coherent: bool = True, labels: bool = False,
                extra_names: Sequence[str] = ()):
        """The picture with ``targets`` in the scene.

        ``extra_arrivals`` are :class:`ArrivalSet`s formed and added
        coherently whatever ``coherent`` says (an emission is a field, not a
        target with an expected intensity); ``None`` entries are skipped.
        ``coherent=False`` adds the targets' arrivals in power (their expected
        intensity, for a fit's model side); the background stays coherent.
        Returns what ``to_cartesian`` returns, or the displayed polar image.

        ``labels=True`` (coherent only) also returns a list of
        :class:`~hydropt.labels.Label`, one per target label and per extra
        arrival set, from each one's own beams against everything else's:
        the class is the target's ``label`` attribute (``extra_names`` for
        the extra arrivals; targets sharing a label are one label, their
        beams summed and their geometry pooled -- a buoy with its chain),
        its mask and boxes are where its energy stands over the rest by
        ``label_margin_db``, and its geometry box is its world extent
        dilated by the resolution (:mod:`hydropt.labels`).
        """
        b_rev = self.background()
        own_extra = []
        for arr in extra_arrivals:
            if arr is not None:
                be = self.beams(arr)
                own_extra.append(be)
                b_rev = b_rev + be
            else:
                own_extra.append(None)
        noise_gen = torch.Generator().manual_seed(self.seed + 2 + int(frame))
        if coherent:
            b = b_rev
            own = []
            for t in targets:
                bt = self.beams(self.echo(t))
                own.append(bt)
                b = b + bt
            field = calibrate(b, self.source_level_db, beam_scale=self.beam_scale)
            noisy = add_receiver_noise(field, self.noise_power, generator=noise_gen)
            if labels:
                # the noisy field itself: |b + n|^2 is what add_receiver_noise
                # returns, and its phasor is recovered up to the noise's own
                # phase by drawing the same noise again on the field
                noise_field = add_receiver_noise(field, self.noise_power,
                                                 generator=torch.Generator().manual_seed(
                                                     self.seed + 2 + int(frame)),
                                                 complex_output=True)
                labs = self._labels(targets, own, own_extra, extra_names, noise_field)
        else:
            back = calibrate(b_rev, self.source_level_db, beam_scale=self.beam_scale)
            noisy = add_receiver_noise(back, self.noise_power, generator=noise_gen)
            for t in targets:
                power = beamform(self.echo(t), self.elements, self.scene.freqs_khz,
                                 self.time_grid, self.steer, sigma_t=self.sigma_t,
                                 shading=self.shading, steer_chunk=self.steer_chunk,
                                 coherent=False, checkpoint=False)
                noisy = noisy + calibrate(power, self.source_level_db, beam_scale=self.beam_scale)
        shown = noisy if self.display is None else self.display(noisy)
        out = shown if self.to_cartesian is None else self.to_cartesian(shown)
        if labels and coherent:
            return out, labs
        return out

    def _labels(self, targets, own, own_extra, extra_names, noisy_field) -> list[Label]:
        """One label per target and per extra arrival set, from the beams."""
        scale = math.sqrt(10.0 ** (self.source_level_db / 10.0) / self.beam_scale)
        cal = lambda bb: bb * scale                      # calibrate() on a field
        total = noisy_field
        labs = []
        with torch.no_grad():
            # targets sharing a label are one thing to the picture (a buoy
            # and its chain, a hull and its fittings): their beams are
            # summed and their geometry pooled before the mask is taken
            groups: dict[str, tuple[Tensor, list]] = {}
            for t, bt in zip(targets, own):
                name = getattr(t, "label", type(t).__name__)
                pts = world_geometry(t)
                if name in groups:
                    b0, p0 = groups[name]
                    groups[name] = (b0 + bt, p0 + [pts])
                else:
                    groups[name] = (bt, [pts])
            for name, (bt, pts) in groups.items():
                bt_c = cal(bt)
                rest = (total - bt_c).abs() ** 2
                labs.append(label_from_beams(
                    name, "target", bt_c.abs() ** 2, rest,
                    margin_db=self.label_margin_db, to_cartesian=self.to_cartesian,
                    geometry_points=torch.cat(pts, dim=0), beam_deg=self.beam_deg,
                    range_m=self.range_cell_m, bearings_deg=self.bearings_deg,
                    ranges_m=self.ranges_m))
            names = list(extra_names) + ["emission"] * (len(own_extra) - len(extra_names))
            for be, name in zip(own_extra, names):
                if be is None:
                    continue
                be_c = cal(be)
                rest = (total - be_c).abs() ** 2
                labs.append(label_from_beams(
                    name, "emission", be_c.abs() ** 2, rest, margin_db=self.label_margin_db,
                    to_cartesian=self.to_cartesian, bearings_deg=self.bearings_deg,
                    ranges_m=self.ranges_m))
        return labs

    def sequence(self, builder: Callable[[float, float, float], object],
                 trajectory: Trajectory, times: Iterable[float], *,
                 extra_targets: Sequence = (), emitters: Sequence[Callable] = (),
                 coherent: bool = True, labels: bool = False):
        """Frames of a target moved along ``trajectory``: ``(t, pose, picture)`` per time.

        ``builder(x, y, heading_deg)`` returns the target at that pose (a
        ``mesh_target`` of the hull, say); ``extra_targets`` are rendered in
        every frame as they are; each of ``emitters`` is
        ``f(x, y, heading_deg, frame) -> ArrivalSet | None``, what the vessel
        radiates from that pose (its propeller, through
        :func:`~hydropt.emission.emission_arrivals`), added to the field.
        A generator, so frames can be drawn and dropped as they come.  With
        ``labels=True`` each frame is ``(t, pose, picture, labels)``, the
        emitters' labels named by their ``label`` attribute.
        """
        names = [getattr(e, "label", "emission") for e in emitters]
        for k, t in enumerate(times):
            pose = trajectory.at(float(t))
            target = builder(*pose)
            with torch.no_grad():
                extra = [e(*pose, k) for e in emitters]
                out = self.picture([target, *extra_targets], extra_arrivals=extra,
                                   frame=k, coherent=coherent, labels=labels,
                                   extra_names=names)
            yield (float(t), pose, *out) if labels else (float(t), pose, out)

    def ownship_sequence(self, world_targets: Sequence, trajectory: Trajectory,
                         times: Iterable[float], *, scene_at: Callable | None = None,
                         emitters: Sequence[Callable] = (), coherent: bool = True,
                         labels: bool = False):
        """Frames of the SONAR moved along ``trajectory`` through stationary targets.

        ``world_targets`` are ``(world_pose, builder)`` pairs: the pose
        ``(x, y, heading_deg)`` of a thing in the world, and ``builder(x, y,
        heading_deg)`` making it at a pose in the sonar's frame (as
        :meth:`sequence` takes).  ``scene_at(x, y, heading_deg)`` returns
        the scene as seen from the ownship pose (the world's sea and seabed
        through :func:`reframe_height_field`), or ``(scene, background)``
        with that scene's reverberation beams from an earlier visit (see
        :meth:`set_scene`); given, the background is re-traced for every
        frame it is not supplied for, and without ``scene_at`` the one
        background is kept, which freezes the sea to the sonar and is only
        right for a flat, featureless one.  ``emitters`` are as in :meth:`sequence` but
        are called with the OWNSHIP pose, for things that radiate on the
        sonar's own platform.  Yields ``(t, ownship_pose, picture)``, or
        ``(t, ownship_pose, picture, labels)`` with ``labels=True``.
        """
        names = [getattr(e, "label", "emission") for e in emitters]
        for k, t in enumerate(times):
            pose = trajectory.at(float(t))
            with torch.no_grad():
                if scene_at is not None:
                    seen = scene_at(*pose)
                    if isinstance(seen, tuple):
                        self.set_scene(*seen)
                    else:
                        self.set_scene(seen)
                targets = [build(*relative_pose(wp, pose)) for wp, build in world_targets]
                extra = [e(*pose, k) for e in emitters]
                out = self.picture(targets, extra_arrivals=extra, frame=k, coherent=coherent,
                                   labels=labels, extra_names=names)
            yield (float(t), pose, *out) if labels else (float(t), pose, out)
