"""Propagation: one interface, several providers.

An EigenrayProvider returns, for a source point and a set of field points, the
eigenrays joining them. Echo synthesis sums over every pair of outbound and
return eigenrays (as in SST). Providers:

- FreeFieldProvider: the direct path only. Tier `free_field`.
- ImageSourceProvider: constant sound speed, flat surface and bottom (and optional
  vertical walls), with reflection coefficients. Tier `image_source`.
- Bellhop-backed provider: milestone 12. Tier `eigenray`.

The synthesis kernel takes straight-line paths, so each eigenray is handed to it
as an *apparent position*: the point at distance c * delay from the sonar along
the ray's direction there. For the first two providers that is exact (it is the
image position). For traced rays it reproduces the delay and arrival angle at the
array, with the wavefront curvature of a point at the travel-time distance.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass

import numpy as np


@dataclass
class EigenraySet:
    """Eigenrays from one source to S field points, K rays each (invalid rays masked).

    amplitude is the complex pressure at the field point for unit pressure at 1 m,
    at the centre frequency, including spreading and boundary reflections but not
    absorption (which is applied along `delay * sound_speed` per frequency).
    """

    delay: np.ndarray       # [S, K] s
    amplitude: np.ndarray   # [S, K] complex
    launch: np.ndarray      # [S, K, 3] unit vector leaving the source
    arrival: np.ndarray     # [S, K, 3] unit vector of travel on arrival at the field point
    n_surface: np.ndarray   # [S, K] int
    n_bottom: np.ndarray    # [S, K] int
    valid: np.ndarray       # [S, K] bool
    sound_speed: float

    @property
    def n_rays(self) -> int:
        return self.delay.shape[1]

    def apparent_positions(self, source) -> np.ndarray:
        """[S, K, 3] straight-line equivalents of the field points as seen from the source."""
        return np.asarray(source, dtype=float) + (self.sound_speed * self.delay)[..., None] * self.launch

    def excess_amplitude(self) -> np.ndarray:
        """[S, K] amplitude relative to spherical spreading over the travel-time distance."""
        return self.amplitude * (self.sound_speed * self.delay)


class EigenrayProvider(abc.ABC):
    """Eigenrays between a source and field points in a given environment."""

    tier: str = ""

    @abc.abstractmethod
    def eigenrays(self, source, points) -> EigenraySet:
        """Eigenrays from `source` [3] to each of `points` [S, 3] (NED, z down)."""


def _direct(source, points, c):
    d = points - source
    r = np.linalg.norm(d, axis=1)
    return r, d / r[:, None]


class FreeFieldProvider(EigenrayProvider):
    tier = "free_field"

    def __init__(self, sound_speed: float):
        self.sound_speed = float(sound_speed)

    def eigenrays(self, source, points) -> EigenraySet:
        source = np.asarray(source, dtype=float)
        points = np.atleast_2d(np.asarray(points, dtype=float))
        r, u = _direct(source, points, self.sound_speed)
        s = points.shape[0]
        return EigenraySet(
            delay=(r / self.sound_speed)[:, None],
            amplitude=(1.0 / r)[:, None].astype(complex),
            launch=u[:, None, :],
            arrival=u[:, None, :],
            n_surface=np.zeros((s, 1), dtype=int),
            n_bottom=np.zeros((s, 1), dtype=int),
            valid=np.ones((s, 1), dtype=bool),
            sound_speed=self.sound_speed,
        )


class ImageSourceProvider(EigenrayProvider):
    """Images of the field point in a flat surface (z = surface_z) and bottom (z = bottom_z).

    `max_order` bounces in total. Reflection coefficients are constant and real here
    (surface -1 by default: pressure release); angle-dependent bottom loss arrives with
    the boundary models in milestone 12.
    """

    tier = "image_source"

    def __init__(self, sound_speed: float, surface_z: float | None = 0.0,
                 bottom_z: float | None = None, surface_coefficient: complex = -1.0,
                 bottom_coefficient: complex = 0.5, max_order: int = 2,
                 walls: list[dict] | None = None, bottom: dict | None = None):
        self.sound_speed = float(sound_speed)
        self.surface_z = surface_z
        self.bottom_z = bottom_z
        self.surface_coefficient = complex(surface_coefficient)
        self.bottom_coefficient = complex(bottom_coefficient)
        self.max_order = int(max_order)
        # Fluid sediment half-space {speed, density (g/cm^3), attenuation (dB/wavelength)}:
        # when given, each bottom bounce uses the Rayleigh reflection coefficient at the
        # ray's grazing angle instead of the constant `bottom_coefficient`.
        self.bottom = bottom
        # Vertical walls: {"axis": 0 or 1, "position": metres, "coefficient": complex}.
        # Walls are applied to first order only, on top of the vertical image series.
        self.walls = walls or []

    def _vertical_images(self, z: np.ndarray):
        """Yield (image_z, n_surface, n_bottom) for the bounce sequences up to max_order."""
        yield z, 0, 0
        zs, zb = self.surface_z, self.bottom_z
        for first in ("s", "b"):
            if (first == "s" and zs is None) or (first == "b" and zb is None):
                continue
            img = z
            ns = nb = 0
            nxt = first
            for _ in range(self.max_order):
                if nxt == "s":
                    if zs is None:
                        break
                    img = 2.0 * zs - img
                    ns += 1
                    nxt = "b"
                else:
                    if zb is None:
                        break
                    img = 2.0 * zb - img
                    nb += 1
                    nxt = "s"
                yield img, ns, nb

    def eigenrays(self, source, points) -> EigenraySet:
        source = np.asarray(source, dtype=float)
        points = np.atleast_2d(np.asarray(points, dtype=float))
        c = self.sound_speed
        rays = []  # (image positions [S, 3], coefficient, ns, nb)
        for img_z, ns, nb in self._vertical_images(points[:, 2]):
            img = points.copy()
            img[:, 2] = img_z
            coef = self.surface_coefficient**ns * self.bottom_coefficient**nb
            rays.append((img, coef, ns, nb))
            for wall in self.walls:
                wimg = img.copy()
                axis = int(wall["axis"])
                wimg[:, axis] = 2.0 * float(wall["position"]) - wimg[:, axis]
                rays.append((wimg, coef * complex(wall.get("coefficient", 0.8)), ns, nb))
        s, k = points.shape[0], len(rays)
        out = EigenraySet(
            delay=np.zeros((s, k)), amplitude=np.zeros((s, k), dtype=complex),
            launch=np.zeros((s, k, 3)), arrival=np.zeros((s, k, 3)),
            n_surface=np.zeros((s, k), dtype=int), n_bottom=np.zeros((s, k), dtype=int),
            valid=np.ones((s, k), dtype=bool), sound_speed=c)
        for j, (img, coef, ns, nb) in enumerate(rays):
            r, u = _direct(source, img, c)
            out.delay[:, j] = r / c
            if self.bottom is not None and nb > 0:
                # In an isovelocity layer every bounce of one image path has the same
                # grazing angle: that of the straight line to the image.
                grazing = np.arcsin(np.clip(np.abs(u[:, 2]), 0.0, 1.0))
                coef = (self.surface_coefficient**ns
                        * rayleigh_reflection(grazing, c, **self.bottom) ** nb)
            out.amplitude[:, j] = coef / r
            out.launch[:, j] = u
            # Direction of travel at the real point: mirror the vertical component once per bounce.
            arr = u.copy()
            if (ns + nb) % 2 == 1:
                arr[:, 2] = -arr[:, 2]
            out.arrival[:, j] = arr
            out.n_surface[:, j] = ns
            out.n_bottom[:, j] = nb
        return out


def snell_crossing(source, points, c_water: float, c_sediment: float, interface_z: float):
    """Least-time paths from a source in the water to points below a flat interface.

    Returns (travel time [S], crossing points [S, 3]). Points above the interface get the
    straight path and a crossing point equal to the point itself.
    """
    source = np.asarray(source, dtype=float)
    points = np.atleast_2d(np.asarray(points, dtype=float))
    a = interface_z - source[2]
    b = points[:, 2] - interface_z
    h = points[:, :2] - source[:2]
    dist = np.linalg.norm(h, axis=1)
    below = (b > 0) & (a > 0)
    bb = np.where(below, b, 1.0)
    u = dist * a / (a + bb)
    for _ in range(40):
        l1 = np.sqrt(u**2 + a**2)
        l2 = np.sqrt((dist - u) ** 2 + bb**2)
        f = u / (c_water * l1) - (dist - u) / (c_sediment * l2)
        df = a**2 / (c_water * l1**3) + bb**2 / (c_sediment * l2**3)
        u = np.clip(u - f / df, 0.0, dist)
    t_refr = np.sqrt(u**2 + a**2) / c_water + np.sqrt((dist - u) ** 2 + bb**2) / c_sediment
    unit = np.divide(h, dist[:, None], out=np.zeros_like(h), where=dist[:, None] > 0)
    cross = np.column_stack([source[:2] + unit * u[:, None], np.full(len(points), interface_z)])
    t_direct = np.linalg.norm(points - source, axis=1) / c_water
    return np.where(below, t_refr, t_direct), np.where(below[:, None], cross, points)


class SedimentProvider(EigenrayProvider):
    """Water over a fluid sediment half-space with a flat interface: one refracted
    eigenray per point (Snell's law). Used for buried objects.

    Amplitude is spherical spreading over the travel-time distance times a constant
    two-way-independent transmission factor; angle-dependent transmission and sediment
    attenuation come with the boundary models in milestone 12.
    """

    tier = "image_source"

    def __init__(self, sound_speed: float, sediment_speed: float, interface_z: float,
                 transmission: complex = 1.0):
        self.sound_speed = float(sound_speed)
        self.sediment_speed = float(sediment_speed)
        self.interface_z = float(interface_z)
        self.transmission = complex(transmission)

    def eigenrays(self, source, points) -> EigenraySet:
        source = np.asarray(source, dtype=float)
        points = np.atleast_2d(np.asarray(points, dtype=float))
        t, cross = snell_crossing(source, points, self.sound_speed, self.sediment_speed, self.interface_z)
        launch = cross - source
        launch /= np.linalg.norm(launch, axis=1)[:, None]
        below = points[:, 2] > self.interface_z
        arrival = points - cross
        norm = np.linalg.norm(arrival, axis=1)
        arrival = np.where(norm[:, None] > 0, arrival / np.maximum(norm, 1e-300)[:, None], launch)
        s = len(points)
        amp = np.where(below, self.transmission, 1.0) / (self.sound_speed * t)
        return EigenraySet(
            delay=t[:, None], amplitude=amp[:, None].astype(complex), launch=launch[:, None, :],
            arrival=arrival[:, None, :], n_surface=np.zeros((s, 1), dtype=int),
            n_bottom=np.zeros((s, 1), dtype=int), valid=np.ones((s, 1), dtype=bool),
            sound_speed=self.sound_speed)


def rayleigh_reflection(grazing, c_water: float, speed: float, density: float,
                        attenuation: float = 0.0, water_density: float = 1.0):
    """Plane-wave pressure reflection coefficient of a fluid half-space (Rayleigh).

    `grazing` in radians; `density` in the same units as `water_density`; `attenuation` in
    dB per wavelength, entering as an imaginary part of the sediment sound speed.
    R = (m sin(g) - sqrt(n^2 - cos^2(g))) / (m sin(g) + sqrt(n^2 - cos^2(g))),
    m = rho_b / rho_w, n = c_w / c_b.
    """
    g = np.asarray(grazing, dtype=float)
    delta = attenuation / (40.0 * np.pi * np.log10(np.e))      # loss tangent from dB/wavelength
    c_b = speed / (1.0 + 1j * delta)
    m = density / water_density
    n = c_water / c_b
    root = np.sqrt(n**2 - np.cos(g) ** 2 + 0j)
    root = np.where(root.imag < 0, -root, root)                 # decaying branch in the sediment
    return (m * np.sin(g) - root) / (m * np.sin(g) + root)


def make_provider(tier: str, sound_speed: float, options: dict | None = None) -> EigenrayProvider:
    options = options or {}
    if tier == "free_field":
        return FreeFieldProvider(sound_speed)
    if tier == "image_source":
        return ImageSourceProvider(sound_speed, **options)
    if tier == "eigenray":
        from .bellhop import BellhopProvider, OceanEnvironment
        opts = dict(options)
        env = OceanEnvironment(**opts.pop("environment"))
        return BellhopProvider(env, **opts)
    raise ValueError(f"unknown fidelity tier {tier!r}")


def two_way_paths(provider: EigenrayProvider, tx_position, rx_reference, points, amplitudes,
                  velocities=None):
    """Kernel paths for every (outbound eigenray, return eigenray) pair.

    Returns q [P, 3], p [P, 3], amplitude [P] and velocities ([P, 3] each or None).
    The return leg uses reciprocity: eigenrays from the receive reference point to the
    scatterer, travelled backwards.
    """
    out = provider.eigenrays(tx_position, points)
    back = provider.eigenrays(rx_reference, points)
    q_all = out.apparent_positions(tx_position)       # [S, Ko, 3]
    p_all = back.apparent_positions(rx_reference)     # [S, Kb, 3]
    a_out = out.excess_amplitude()
    a_back = back.excess_amplitude()
    s, ko = out.delay.shape
    kb = back.delay.shape[1]
    amplitudes = np.asarray(amplitudes, dtype=complex)
    q = np.repeat(q_all[:, :, None, :], kb, axis=2).reshape(-1, 3)
    p = np.repeat(p_all[:, None, :, :], ko, axis=1).reshape(-1, 3)
    amp = (amplitudes[:, None, None] * a_out[:, :, None] * a_back[:, None, :]).reshape(-1)
    valid = (out.valid[:, :, None] & back.valid[:, None, :]).reshape(-1)
    vq = vp = None
    if velocities is not None and np.any(velocities):
        # Images move with the scatterer horizontally; vertical image motion is mirrored
        # per bounce. Scatterer velocities are small corrections, so the horizontal part
        # is kept and the vertical part is applied to the direct path only.
        v = np.asarray(velocities, dtype=float)
        vq = np.repeat(np.repeat(v[:, None, None, :], ko, axis=1), kb, axis=2).reshape(-1, 3)
        vp = vq.copy()
        vq, vp = vq[valid], vp[valid]
    return q[valid], p[valid], amp[valid], vq, vp
