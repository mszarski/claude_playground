"""Fitting the scene to a real picture: the sea, the seabed, the tilt, the gain.

Before a boat can be fitted to a real sonar picture (``examples/22`` fits one
to a simulated picture) the picture's *background* has to be the model's:
the level of the reverberation against range, where the seabed enters the
lobe, where the surface does, where the noise floor takes over, and the
head's gain that puts it all on the display's scale.  None of these is
known for a real ping to the precision a fit needs, and all of them are
smooth in a handful of parameters:

* the seabed's Lambert strength (``seabed_db``) and the surface's, as a
  gain on the surface patches (``surface_db``) -- what the sea state
  amounts to at these grazing angles;
* the head's tilt (``tilt_deg``): the transmit fan's array factor and the
  receive elevation envelope both steer with it, so a wrong tilt moves the
  range at which the seabed enters the picture -- and the trace does not
  depend on it (the fan's directions are fixed, only their weights move),
  so the tilt is fitted without retracing;
* a gain offset (``gain_db``): the difference between the model's absolute
  calibration and whatever the real head's processing chain applied;
* the ambient level (``noise_db``), which sets the far end.

Altitude and water depth are held: the vehicle's navigation knows them,
and they move the trace.  The wind enters only through ``surface_db`` here
(the surface's roughness bends rays; its *strength* at grazing incidence is
what the picture sees).

**The observable is the swath's level against range, not the picture.**
A real picture is one realisation of speckle; so is a rendered one, and the
two never share a realisation.  What both share is the expected level, and
the median over the beams of a bearing sector at each range is that level
to within the speckle's own median-to-mean ratio, which is the same for
both and falls into ``gain_db``.  :func:`range_profile` takes it (in dB,
per sector, smoothed over a few metres of range) and the fit minimises the
squared difference of profiles.  This is a *smooth* loss in every parameter
above: the parameters scale amplitudes or steer weights, none of them moves
a scatterer, so the coherent picture's gradient is usable here where it
was not for a pose (``examples/22``'s reason for the incoherent model).

The real picture goes in as :class:`RealPicture`: a ``[beams, bins]`` image
in dB with its bearings and ranges, from an array or a file
(:func:`load_picture`), or resampled from a Cartesian image
(:func:`cartesian_to_polar`, the inverse of ``examples/15``'s
``to_cartesian``).  ``examples/30`` runs the fit on a picture rendered with
hidden parameters and independent seeds, loaded the way a real one would
be, and recovers them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import torch
from torch import Tensor, nn

from .beamform import beamform, line_array_factor
from .noise import add_receiver_noise, calibrate
from .reverb import LambertScattering, reverberation_arrivals

__all__ = ["RealPicture", "load_picture", "cartesian_to_polar", "range_profile",
           "SceneFit", "fit_scene"]


@dataclass
class RealPicture:
    """A sonar picture as the fit takes it: ``[beams, bins]`` in dB, with its axes."""
    image_db: Tensor
    bearings_deg: Tensor      # [beams]
    ranges_m: Tensor          # [bins]
    name: str = "picture"

    def __post_init__(self) -> None:
        if self.image_db.ndim == 3:
            self.image_db = self.image_db[:, 0]
        if self.image_db.shape != (self.bearings_deg.numel(), self.ranges_m.numel()):
            raise ValueError(f"image {tuple(self.image_db.shape)} does not match "
                             f"{self.bearings_deg.numel()} bearings x {self.ranges_m.numel()} ranges")


def cartesian_to_polar(cart: Tensor, gx: Tensor, gy: Tensor, bearings_deg: Tensor,
                       ranges_m: Tensor) -> Tensor:
    """Resample a ``[n_y, n_x]`` picture in metres onto ``[beams, bins]``, bilinearly.

    The inverse of ``examples/15``'s ``to_cartesian``: ``x`` forward along
    ``gx``, ``y`` to port along ``gy``, the sonar at the origin.  Cells the
    picture does not cover come back as zero.
    """
    B, R = torch.meshgrid(torch.deg2rad(bearings_deg), ranges_m, indexing="ij")
    X, Y = R * torch.cos(B), R * torch.sin(B)
    x0, x1 = float(gx[0]), float(gx[-1])
    y0, y1 = float(gy[0]), float(gy[-1])
    sx = 2.0 * (X - x0) / (x1 - x0) - 1.0
    sy = 2.0 * (Y - y0) / (y1 - y0) - 1.0
    samples = torch.stack([sx, sy], dim=-1).unsqueeze(0).to(cart.dtype)
    src = cart.reshape(1, 1, *cart.shape)
    out = torch.nn.functional.grid_sample(src, samples, mode="bilinear",
                                          padding_mode="zeros", align_corners=True)
    return out.reshape(X.shape)


def load_picture(path, *, bearings_deg: Tensor | None = None, ranges_m: Tensor | None = None,
                 db_range: tuple[float, float] | None = None, name: str | None = None
                 ) -> RealPicture:
    """A picture from a file.

    ``.npz`` with ``image`` (dB, ``[beams, bins]``) and ``bearings``,
    ``ranges``; or with ``image`` (dB, ``[n_y, n_x]``), ``gx``, ``gy`` and
    the ``bearings_deg`` and ``ranges_m`` to resample onto; or an ``.npy``
    / a greyscale ``.png`` of the polar image whose grey levels span
    ``db_range``, with the axes given.
    """
    import numpy as np
    path = Path(path)
    dt = torch.get_default_dtype()
    name = name or path.stem
    if path.suffix == ".npz":
        z = np.load(path)
        img = torch.as_tensor(np.asarray(z["image"]), dtype=dt)
        if "bearings" in z and "ranges" in z:
            return RealPicture(img, torch.as_tensor(np.asarray(z["bearings"]), dtype=dt),
                               torch.as_tensor(np.asarray(z["ranges"]), dtype=dt), name)
        if bearings_deg is None or ranges_m is None:
            raise ValueError("a Cartesian picture needs the bearings and ranges to resample onto")
        gx = torch.as_tensor(np.asarray(z["gx"]), dtype=dt)
        gy = torch.as_tensor(np.asarray(z["gy"]), dtype=dt)
        return RealPicture(cartesian_to_polar(img, gx, gy, bearings_deg, ranges_m),
                           bearings_deg, ranges_m, name)
    if bearings_deg is None or ranges_m is None:
        raise ValueError("an image file needs the bearings and ranges of its axes")
    if path.suffix == ".npy":
        img = torch.as_tensor(np.load(path), dtype=dt)
    else:
        from matplotlib.image import imread
        a = np.asarray(imread(path), dtype=np.float64)
        if a.ndim == 3:
            a = a[..., :3].mean(-1)
        if a.max() > 1.0:
            a = a / 255.0
        if db_range is None:
            raise ValueError("a greyscale image needs db_range=(lo, hi) for its grey levels")
        img = torch.as_tensor(db_range[0] + a * (db_range[1] - db_range[0]), dtype=dt)
    return RealPicture(img, bearings_deg, ranges_m, name)


def range_profile(image_db: Tensor, bearings_deg: Tensor, ranges_m: Tensor, *,
                  sectors_deg: Sequence[tuple[float, float]] = ((-60.0, -20.0), (-20.0, 20.0),
                                                                 (20.0, 60.0)),
                  smooth_m: float = 5.0) -> Tensor:
    """The swath's level against range: ``[sectors, bins]`` dB.

    The median over the beams of each sector at each range bin, then a
    moving mean over ``smooth_m`` of range.  A median, because it is the
    level the display's own gain reads and a target or a spoke cannot move
    it; smoothed, because a bin is half a metre and the speckle's median
    over 60 beams still wanders by a decibel.
    """
    img = image_db[:, 0] if image_db.ndim == 3 else image_db
    rows = []
    for lo, hi in sectors_deg:
        sel = (bearings_deg >= lo) & (bearings_deg < hi)
        rows.append(img[sel].median(dim=0).values)
    prof = torch.stack(rows)
    dr = float(ranges_m[1] - ranges_m[0]) if ranges_m.numel() > 1 else smooth_m
    k = max(1, int(round(smooth_m / dr)))
    if k > 1:
        pad = k // 2
        w = torch.ones(1, 1, k, dtype=prof.dtype) / k
        prof = torch.nn.functional.conv1d(
            torch.nn.functional.pad(prof.unsqueeze(1), (pad, k - 1 - pad), mode="replicate"),
            w).squeeze(1)
    return prof


def rx_envelope(directions: Tensor, tilt_deg: Tensor, *, n_elements: int, n_beams: int,
                beam_deg: float) -> Tensor:
    """The receive elevation envelope: the sum of ``n_beams`` beams about ``tilt_deg``.

    ``examples/21``'s ``receive_beam`` with the tilt as a tensor, so the fit
    can move it.  ``directions`` are ARRIVAL directions (``z`` down); sound
    from ``tilt`` degrees up travels downward, so the steer is ``+sin``.
    """
    tilts = tilt_deg + (torch.arange(n_beams, dtype=tilt_deg.dtype) - (n_beams - 1) / 2.0) * beam_deg
    total = None
    for t in tilts:
        w = line_array_factor(directions[..., 2], n_elements, sin_steer=-torch.sin(torch.deg2rad(t)))
        total = w if total is None else total + w
    return total


class SceneFit(nn.Module):
    """The scene's fittable parameters and the picture they make.

    Args:
        scene: the :class:`~hydropt.scene.Scene`, already built (its trace is
            taken once, without gradient: nothing here moves a ray).
        directions: ``[R, 3]`` the transmit fan.
        elements, time_grid, steer, sigma_t, shading: the beamformer's.
        source_level_db, noise_power: the calibration and the ambient noise
            power the ``noise_db`` offset applies to.
        solid_angle_per_ray: for :func:`reverberation_arrivals`.
        n_tx, n_rx_elev, n_elev_beams, elev_beam_deg: the head's vertical
            arrays, for the transmit factor and the receive envelope.
        seabed_db, surface_db, tilt_deg, gain_db, noise_db: starting values.
        fit: which of those five are learnable.
        seed: the patch draw and the noise draw (fixed through a fit).
    """

    def __init__(self, scene, directions: Tensor, *, elements: Tensor, time_grid: Tensor,
                 steer: Tensor, sigma_t: float, shading: Tensor | None, source_level_db: float,
                 noise_power: float, solid_angle_per_ray: float, n_tx: int, n_rx_elev: int,
                 n_elev_beams: int, elev_beam_deg: float, seabed_db: float = -27.0,
                 surface_db: float = 0.0, tilt_deg: float = -5.0, gain_db: float = 0.0,
                 noise_db: float = 0.0,
                 fit: Sequence[str] = ("seabed_db", "surface_db", "tilt_deg", "gain_db", "noise_db"),
                 max_arrivals: int | None = None, seed: int = 0, steer_chunk: int = 8) -> None:
        super().__init__()
        from .beamform import beam_power_scale, shading_window
        from .tracer import trace
        self.scene = scene
        self.directions = directions
        self.elements, self.time_grid, self.steer = elements, time_grid, steer
        self.sigma_t, self.shading = float(sigma_t), shading
        self.source_level_db, self.noise_power = float(source_level_db), float(noise_power)
        self.solid = float(solid_angle_per_ray)
        self.n_tx, self.n_rx_elev, self.n_elev_beams = int(n_tx), int(n_rx_elev), int(n_elev_beams)
        self.elev_beam_deg = float(elev_beam_deg)
        self.max_arrivals, self.seed, self.steer_chunk = max_arrivals, int(seed), int(steer_chunk)
        n = int(elements.reshape(-1, 3).shape[0])
        w = shading if shading is not None else shading_window(n, "uniform", dtype=time_grid.dtype)
        self.beam_scale = beam_power_scale(w, self.sigma_t)
        dt = time_grid.dtype
        for name, value in (("seabed_db", seabed_db), ("surface_db", surface_db),
                            ("tilt_deg", tilt_deg), ("gain_db", gain_db), ("noise_db", noise_db)):
            t = torch.tensor(float(value), dtype=dt)
            if name in fit:
                setattr(self, name, nn.Parameter(t))
            else:
                self.register_buffer(name, t)
        self.seabed = LambertScattering(float(seabed_db), learnable=False)
        with torch.no_grad():
            self.result = trace(scene, directions)
        self.elev = torch.asin(directions[:, 2].clamp(-1.0, 1.0))

    def weights(self) -> Tensor:
        """The fan's two-way elevation weight at the current tilt."""
        tx = line_array_factor(torch.sin(self.elev), self.n_tx,
                               sin_steer=torch.sin(torch.deg2rad(self.tilt_deg)))
        rx = rx_envelope(-self.directions, self.tilt_deg, n_elements=self.n_rx_elev,
                         n_beams=self.n_elev_beams, beam_deg=self.elev_beam_deg)
        return tx * rx

    def picture(self) -> Tensor:
        """The calibrated, noisy reverberation picture ``[beams, bands, bins]`` at the parameters."""
        self.seabed.strength_db = self.seabed_db          # the Lambert strength, live
        gain = 10.0 ** (self.surface_db / 10.0)
        rev = reverberation_arrivals(
            self.result, self.directions, self.scene.freqs_khz, scattering=self.seabed,
            solid_angle_per_ray=self.solid, ray_weights=self.weights(), boundary="both",
            surface=self.scene.surface, bottom=self.scene.bottom, max_arrivals=self.max_arrivals,
            surface_gain=lambda xy: gain.expand(xy.shape[:-1]),
            generator=torch.Generator().manual_seed(self.seed + 1))
        b = beamform(rev, self.elements, self.scene.freqs_khz, self.time_grid, self.steer,
                     sigma_t=self.sigma_t, shading=self.shading, steer_chunk=self.steer_chunk,
                     complex_output=True)
        field = calibrate(b, self.source_level_db, beam_scale=self.beam_scale)
        field = field * 10.0 ** (self.gain_db / 20.0)
        noise = self.noise_power * 10.0 ** ((self.noise_db + self.gain_db) / 10.0)
        return add_receiver_noise(field, noise, generator=torch.Generator().manual_seed(self.seed + 2))

    def values(self) -> dict:
        return {k: float(getattr(self, k).detach()) for k in
                ("seabed_db", "surface_db", "tilt_deg", "gain_db", "noise_db")}

    def renderer(self, *, tx_pattern: Callable, display: Callable | None = None,
                 to_cartesian: Callable | None = None, target_kwargs: dict | None = None,
                 **kwargs):
        """The fitted scene as a :class:`~hydropt.sequence.PictureRenderer`.

        Its background is this fit's picture (same patches, same noise draw
        at frame 0, the gain folded into the source level and the noise),
        so a target can now be fitted or labelled on the fitted background.
        ``tx_pattern`` is the projector's pattern as a function of direction
        for the target's solved return leg (``examples/21``'s
        ``transmit_pattern``); the receive pattern is the fitted envelope.
        """
        from .sequence import PictureRenderer
        with torch.no_grad():
            tilt = self.tilt_deg.detach().clone()
            gain = 10.0 ** (float(self.surface_db.detach()) / 10.0)
            tx = line_array_factor(torch.sin(self.elev), self.n_tx,
                                   sin_steer=torch.sin(torch.deg2rad(tilt)))
        rx = lambda d: rx_envelope(d, tilt, n_elements=self.n_rx_elev,
                                   n_beams=self.n_elev_beams, beam_deg=self.elev_beam_deg)
        g = float(self.gain_db.detach())
        r = PictureRenderer(
            self.scene, elements=self.elements, directions=self.directions, tx_weights=tx,
            tx_pattern=tx_pattern, rx_pattern=rx, time_grid=self.time_grid, steer=self.steer,
            sigma_t=self.sigma_t, shading=self.shading, source_level_db=self.source_level_db + g,
            noise_power=self.noise_power * 10.0 ** ((float(self.noise_db.detach()) + g) / 10.0),
            scattering=LambertScattering(float(self.seabed_db.detach()), learnable=False),
            solid_angle_per_ray=self.solid, boundary="both",
            surface_gain=lambda xy: torch.full(xy.shape[:-1], gain, dtype=xy.dtype),
            max_arrivals=self.max_arrivals, display=display, to_cartesian=to_cartesian,
            target_kwargs=target_kwargs, seed=self.seed, steer_chunk=self.steer_chunk, **kwargs)
        return r


def fit_scene(model: SceneFit, real: RealPicture, *, steps: int = 40, lr: float = 0.5,
              sectors_deg=((-60.0, -20.0), (-20.0, 20.0), (20.0, 60.0)), smooth_m: float = 5.0,
              lr_scale: dict | None = None, log: Callable[[str], None] | None = None,
              floor_db: float = -300.0) -> list[dict]:
    """Descend the profile misfit.  Returns the history of ``(step, loss, values)``.

    Adam on the learnable parameters, each at ``lr`` times its entry in
    ``lr_scale`` (dB per step for the levels, degrees for the tilt).  The
    loss is the mean squared difference, in dB, between :func:`range_profile`
    of the real picture and of the model's, over the bins where the real
    profile is above ``floor_db``.
    """
    target = range_profile(real.image_db, real.bearings_deg, real.ranges_m,
                           sectors_deg=sectors_deg, smooth_m=smooth_m)
    keep = target > floor_db
    scale = {"seabed_db": 1.0, "surface_db": 1.0, "tilt_deg": 0.5, "gain_db": 1.0, "noise_db": 1.0}
    scale.update(lr_scale or {})
    groups = [{"params": [p], "lr": lr * scale[n]} for n, p in model.named_parameters()]
    opt = torch.optim.Adam(groups)
    history = []
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        pic = model.picture()
        prof = range_profile(10.0 * torch.log10(pic.clamp_min(1e-30)), real.bearings_deg,
                             real.ranges_m, sectors_deg=sectors_deg, smooth_m=smooth_m)
        loss = ((prof - target)[keep] ** 2).mean()
        loss.backward()
        opt.step()
        history.append(dict(step=step, loss=float(loss.detach()), **model.values()))
        if log is not None:
            log(f"  step {step:3d}  loss {float(loss.detach()):8.3f} dB^2  "
                + "  ".join(f"{k} {v:+7.2f}" for k, v in model.values().items()))
    return history
