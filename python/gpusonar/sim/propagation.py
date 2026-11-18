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

The first two providers, the sediment provider and `two_way_paths` also exist in C++
(libs/sim/src/paths.cpp), which is used by default; `engine="numpy"` or
GPUSONAR_SIM_ENGINE=numpy selects the NumPy code in this file (sim/engine.py).
"""

from __future__ import annotations

import abc
from dataclasses import dataclass

import numpy as np

from .. import _gpusonar as core
from . import engine as _engine


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
    # [S, K] m/s, or None: rate of change of the ray's path length from a moving boundary
    # (sim/timespread.py, `doppler`). The ray is then shifted in frequency by -f rate / c.
    range_rate: np.ndarray | None = None

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


def _cpp_points(engine, points):
    """True when the C++ engine is selected and takes these field points ([S, 3])."""
    return _engine.resolve(engine) == "cpp" and points.ndim == 2 and points.shape[1] == 3


def _source3(source):
    return tuple(float(x) for x in source)


def _direct(source, points, c):
    d = points - source
    r = np.linalg.norm(d, axis=1)
    return r, d / r[:, None]


class FreeFieldProvider(EigenrayProvider):
    tier = "free_field"

    def __init__(self, sound_speed: float, engine: str | None = None):
        self.sound_speed = float(sound_speed)
        self.engine = engine

    def eigenrays(self, source, points) -> EigenraySet:
        source = np.asarray(source, dtype=float)
        points = np.atleast_2d(np.asarray(points, dtype=float))
        if _cpp_points(self.engine, points):
            return EigenraySet(**core.sim_free_field_eigenrays(self.sound_speed, _source3(source), points))
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

    The surface reflects with `surface_coefficient` (-1 by default: pressure release, a fact of the air above).
    The bottom has NO default: a bottom needs a description, either `bottom` (a fluid half-space {speed, density,
    attenuation} or {sediment: name}: the Rayleigh coefficient at each ray's grazing angle; a scenario's
    environment supplies it) or an explicit constant `bottom_coefficient`. A `bottom_z` with neither is an error
    (before D-104 it silently reflected 0.5 at every angle); a wall likewise needs its `coefficient`.

    Bounces: `max_order` when given. Left out, the series is cut by energy (D-105): orders are kept while some field
    point has an arrival within `arrival_floor_db` (20 dB) of its strongest, up to `max_order_cap` (8). When the cap
    cuts off an order that is still above the floor, a warning says by how much, once per provider, and
    `truncation` holds it; `order_used` is the order of the last call.
    """

    tier = "image_source"
    DEFAULT_ARRIVAL_FLOOR_DB = 20.0
    DEFAULT_MAX_ORDER_CAP = 8

    def __init__(self, sound_speed: float, surface_z: float | None = 0.0,
                 bottom_z: float | None = None, surface_coefficient: complex = -1.0,
                 bottom_coefficient: complex | None = None, max_order: int | None = None,
                 walls: list[dict] | None = None, bottom: dict | None = None,
                 engine: str | None = None, arrival_floor_db: float | None = None, max_order_cap: int | None = None):
        self.sound_speed = float(sound_speed)
        self.engine = engine
        self.surface_z = surface_z
        self.bottom_z = bottom_z
        self.surface_coefficient = complex(surface_coefficient)
        if bottom_z is not None and bottom is None and bottom_coefficient is None:
            raise ValueError(f"image_source: a bottom at z = {bottom_z} with no description. Give the scenario's environment a `bottom` (a name in "
                             "configs/sediments.yaml or {speed, density, attenuation}), or `propagation.bottom`, or an explicit "
                             "`propagation.bottom_coefficient`; no reflection coefficient is assumed")
        # With `bottom` the Rayleigh coefficient is used and this constant is not; 0 keeps arithmetic defined.
        self.bottom_coefficient = complex(0.0 if bottom_coefficient is None else bottom_coefficient)
        self.max_order = None if max_order is None else int(max_order)
        self.arrival_floor_db = float(self.DEFAULT_ARRIVAL_FLOOR_DB if arrival_floor_db is None else arrival_floor_db)
        self.max_order_cap = int(self.DEFAULT_MAX_ORDER_CAP if max_order_cap is None else max_order_cap)
        self.order_used = self.max_order
        self.truncation = None                   # {"order": cap, "left_out_db": level of the first order cut, re the strongest arrival}
        for wall in walls or []:
            if wall.get("coefficient") is None:
                raise ValueError("image_source: a wall needs its reflection `coefficient` (none is assumed)")
        # Fluid sediment half-space {speed, density (g/cm^3), attenuation (dB/wavelength)}:
        # when given, each bottom bounce uses the Rayleigh reflection coefficient at the
        # ray's grazing angle instead of the constant `bottom_coefficient`.
        # {sediment: <name in configs/sediments.yaml>[, sediment_file]} gives the three numbers
        # from the table (speed = its ratio x this provider's sound speed); keys given beside it win.
        if bottom is not None and "sediment" in bottom:
            from . import bottom as bottom_model
            rest = {k: v for k, v in bottom.items() if k not in ("sediment", "sediment_file")}
            bottom = dict(bottom_model.fluid_halfspace(bottom["sediment"], self.sound_speed, bottom.get("sediment_file")), **rest)
        self.bottom = bottom
        # Vertical walls: {"axis": 0 or 1, "position": metres, "coefficient": complex}.
        # Walls are applied to first order only, on top of the vertical image series.
        self.walls = walls or []

    def _order_for(self, source, points) -> int:
        """The bounce order of this call: `max_order` when it was given, else the energy criterion of the class
        docstring, worked out here for both engines on up to 256 of the field points (evenly taken, both ends in)."""
        if self.max_order is not None:
            return self.max_order
        if self.surface_z is None and self.bottom_z is None:
            return 0
        pts = points if len(points) <= 256 else points[np.unique(np.linspace(0, len(points) - 1, 256).astype(int))]
        probe = self._eigenrays_numpy(source, pts, self.max_order_cap + 1, walls=False)
        mag = np.abs(probe.amplitude)
        rel = mag / np.maximum(mag.max(axis=1, keepdims=True), 1e-300)
        bounces = (probe.n_surface + probe.n_bottom)[0]
        level = np.array([rel[:, bounces == n].max() if np.any(bounces == n) else 0.0 for n in range(self.max_order_cap + 2)])
        floor = 10.0 ** (-self.arrival_floor_db / 20.0)
        above = np.flatnonzero(level[:self.max_order_cap + 1] >= floor)
        order = int(above[-1]) if len(above) else 0
        if level[self.max_order_cap + 1] >= floor:
            order = self.max_order_cap
            left = float(20.0 * np.log10(level[self.max_order_cap + 1]))
            if self.truncation is None:
                import warnings
                warnings.warn(f"image_source: the bounce series was cut at the cap of {self.max_order_cap} bounces while the next order is still "
                              f"{left:.1f} dB re the strongest arrival (floor: -{self.arrival_floor_db:g} dB). Arrivals are missing: raise "
                              "`max_order_cap`, or state `max_order`", RuntimeWarning, stacklevel=4)
            self.truncation = {"order": order, "left_out_db": left}
        self.order_used = order if self.order_used is None else max(self.order_used, order)
        return self.order_used

    def _vertical_images(self, z: np.ndarray, max_order: int | None = None):
        """Yield (image_z, n_surface, n_bottom) for the bounce sequences up to max_order."""
        max_order = self.max_order if max_order is None else max_order
        yield z, 0, 0
        zs, zb = self.surface_z, self.bottom_z
        for first in ("s", "b"):
            if (first == "s" and zs is None) or (first == "b" and zb is None):
                continue
            img = z
            ns = nb = 0
            nxt = first
            for _ in range(max_order):
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
        order = self._order_for(source, points)
        if _cpp_points(self.engine, points) and all(int(w["axis"]) in (0, 1) for w in self.walls):
            walls = [(int(w["axis"]), float(w["position"]), complex(w["coefficient"])) for w in self.walls]
            return EigenraySet(**core.sim_image_source_eigenrays(
                self.sound_speed, _source3(source), points,
                surface_z=None if self.surface_z is None else float(self.surface_z),
                bottom_z=None if self.bottom_z is None else float(self.bottom_z),
                surface_coefficient=self.surface_coefficient, bottom_coefficient=self.bottom_coefficient,
                max_order=order, walls=walls,
                bottom=None if self.bottom is None else _fluid_bottom(**self.bottom)))
        return self._eigenrays_numpy(source, points, order)

    def _eigenrays_numpy(self, source, points, order: int, walls: bool = True) -> EigenraySet:
        c = self.sound_speed
        rays = []  # (image positions [S, 3], coefficient, ns, nb)
        for img_z, ns, nb in self._vertical_images(points[:, 2], order):
            img = points.copy()
            img[:, 2] = img_z
            coef = self.surface_coefficient**ns * self.bottom_coefficient**nb
            rays.append((img, coef, ns, nb, 1.0))
            for wall in self.walls if walls else []:
                wimg = img.copy()
                axis = int(wall["axis"])
                wimg[:, axis] = 2.0 * float(wall["position"]) - wimg[:, axis]
                wall_coef = complex(wall["coefficient"])
                rays.append((wimg, coef * wall_coef, ns, nb, wall_coef))
        s, k = points.shape[0], len(rays)
        out = EigenraySet(
            delay=np.zeros((s, k)), amplitude=np.zeros((s, k), dtype=complex),
            launch=np.zeros((s, k, 3)), arrival=np.zeros((s, k, 3)),
            n_surface=np.zeros((s, k), dtype=int), n_bottom=np.zeros((s, k), dtype=int),
            valid=np.ones((s, k), dtype=bool), sound_speed=c)
        for j, (img, coef, ns, nb, wall_coef) in enumerate(rays):
            r, u = _direct(source, img, c)
            out.delay[:, j] = r / c
            if self.bottom is not None and nb > 0:
                # In an isovelocity layer every bounce of one image path has the same
                # grazing angle: that of the straight line to the image.
                grazing = np.arcsin(np.clip(np.abs(u[:, 2]), 0.0, 1.0))
                coef = (self.surface_coefficient**ns
                        * rayleigh_reflection(grazing, c, **self.bottom) ** nb) * wall_coef
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


def _fluid_bottom(speed: float, density: float, attenuation: float = 0.0, water_density: float = 1.0):
    """A sediment mapping as the tuple the C++ provider takes (the arguments of `rayleigh_reflection`)."""
    return float(speed), float(density), float(attenuation), float(water_density)


def snell_crossing(source, points, c_water: float, c_sediment: float, interface_z: float,
                   engine: str | None = None):
    """Least-time paths from a source in the water to points below a flat interface.

    Returns (travel time [S], crossing points [S, 3]). Points above the interface get the
    straight path and a crossing point equal to the point itself.
    """
    source = np.asarray(source, dtype=float)
    points = np.atleast_2d(np.asarray(points, dtype=float))
    if _cpp_points(engine, points):
        return core.sim_snell_crossing(_source3(source), points, float(c_water), float(c_sediment), float(interface_z))
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
                 transmission: complex = 1.0, engine: str | None = None):
        self.sound_speed = float(sound_speed)
        self.engine = engine
        self.sediment_speed = float(sediment_speed)
        self.interface_z = float(interface_z)
        self.transmission = complex(transmission)

    def eigenrays(self, source, points) -> EigenraySet:
        source = np.asarray(source, dtype=float)
        points = np.atleast_2d(np.asarray(points, dtype=float))
        if _cpp_points(self.engine, points):
            return EigenraySet(**core.sim_sediment_eigenrays(self.sound_speed, self.sediment_speed, self.interface_z,
                                                             self.transmission, _source3(source), points))
        t, cross = snell_crossing(source, points, self.sound_speed, self.sediment_speed, self.interface_z,
                                  engine="numpy")
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
    if tier == "sediment":
        return SedimentProvider(sound_speed, **options)
    if tier == "eigenray":
        from .bellhop import BellhopProvider, OceanEnvironment
        opts = dict(options)
        env = OceanEnvironment(**opts.pop("environment"))
        return BellhopProvider(env, **opts)
    raise ValueError(f"unknown fidelity tier {tier!r}")


def two_way_paths(provider: EigenrayProvider, tx_position, rx_reference, points, amplitudes,
                  velocities=None, lambert=None, normals=None, engine: str | None = None):
    """Kernel paths for every (outbound eigenray, return eigenray) pair.

    Returns q [P, 3], p [P, 3], amplitude [P] and velocities ([P, 3] each or None).
    The return leg uses reciprocity: eigenrays from the receive reference point to the
    scatterer, travelled backwards.

    The pairing runs in C++ for any provider's eigenrays (Bellhop's too) unless the NumPy
    engine is selected.
    """
    out = provider.eigenrays(tx_position, points)
    back = provider.eigenrays(rx_reference, points)
    # Rays with a path-length rate (frequency spreading within a ping) give each end of a
    # pair its own velocity: the scatterer's plus the leg's rate along the ray.
    rates = [None if e.range_rate is None or not np.any(e.range_rate) else e.range_rate for e in (out, back)]
    doppler = rates[0] is not None or rates[1] is not None
    if _engine.resolve(engine) == "cpp":
        weighted = lambert is not None and bool(np.any(lambert))
        moving = velocities is not None and bool(np.any(velocities))
        args = (out.delay, out.amplitude, out.launch, out.arrival, out.valid, float(out.sound_speed),
                back.delay, back.amplitude, back.launch, back.arrival, back.valid, float(back.sound_speed),
                _source3(tx_position), _source3(rx_reference), np.asarray(amplitudes, dtype=complex))
        kwargs = dict(velocities=np.asarray(velocities, dtype=float) if moving else None,
                      lambert=np.asarray(lambert, dtype=bool) if weighted else None,
                      normals=np.asarray(normals, dtype=float) if weighted and normals is not None else None)
        if doppler:
            return core.sim_two_way_paths(*args, **kwargs, out_range_rate=rates[0], back_range_rate=rates[1])
        q, p, amp, vq = core.sim_two_way_paths(*args, **kwargs)
        return q, p, amp, vq, (None if vq is None else vq.copy())
    q_all = out.apparent_positions(tx_position)       # [S, Ko, 3]
    p_all = back.apparent_positions(rx_reference)     # [S, Kb, 3]
    a_out = out.excess_amplitude()
    a_back = back.excess_amplitude()
    s, ko = out.delay.shape
    kb = back.delay.shape[1]
    amplitudes = np.asarray(amplitudes, dtype=complex)
    q = np.repeat(q_all[:, :, None, :], kb, axis=2).reshape(-1, 3)
    p = np.repeat(p_all[:, None, :, :], ko, axis=1).reshape(-1, 3)
    amp = amplitudes[:, None, None] * a_out[:, :, None] * a_back[:, None, :]
    if lambert is not None and np.any(lambert):
        # Lambert's law for boundary patches: each path pair is weighted by
        # sqrt(sin(g_in) sin(g_out)), with g the grazing angle of that eigenray at the
        # (horizontal) boundary. The amplitude drawn for the patch is shared by all pairs.
        sin_in = np.abs(out.arrival[:, :, 2])
        sin_out = np.abs(back.arrival[:, :, 2])
        if normals is not None:
            # Surface points with their own normal: the cosine from the normal replaces the
            # sine of the grazing angle, and a face turned away from a path returns nothing.
            nrm = np.asarray(normals, dtype=float)
            has = np.isfinite(nrm).all(axis=1)
            n_safe = np.where(has[:, None], nrm, 0.0)
            cos_in = np.maximum(-(out.arrival * n_safe[:, None, :]).sum(axis=2), 0.0)
            cos_out = np.maximum(-(back.arrival * n_safe[:, None, :]).sum(axis=2), 0.0)
            sin_in = np.where(has[:, None], cos_in, sin_in)
            sin_out = np.where(has[:, None], cos_out, sin_out)
        factor = np.sqrt(sin_in[:, :, None] * sin_out[:, None, :])
        amp = np.where(np.asarray(lambert, dtype=bool)[:, None, None], amp * factor, amp)
    amp = amp.reshape(-1)
    valid = (out.valid[:, :, None] & back.valid[:, None, :]).reshape(-1) & (amp != 0)
    vq = vp = None
    if velocities is not None and np.any(velocities):
        # Images move with the scatterer horizontally; vertical image motion is mirrored
        # per bounce. Scatterer velocities are small corrections, so the horizontal part
        # is kept and the vertical part is applied to the direct path only.
        v = np.asarray(velocities, dtype=float)
        vq = np.repeat(np.repeat(v[:, None, None, :], ko, axis=1), kb, axis=2).reshape(-1, 3)
        vp = vq.copy()
        vq, vp = vq[valid], vp[valid]
    if doppler:
        # A ray whose path lengthens at `range_rate`: its apparent position recedes at that
        # speed along the ray, which the synthesis kernel turns into the frequency shift.
        if vq is None:
            vq, vp = np.zeros((int(valid.sum()), 3)), np.zeros((int(valid.sum()), 3))
        if rates[0] is not None:
            vq = vq + np.repeat((rates[0][..., None] * out.launch)[:, :, None, :], kb, axis=2).reshape(-1, 3)[valid]
        if rates[1] is not None:
            vp = vp + np.repeat((rates[1][..., None] * back.launch)[:, None, :, :], ko, axis=1).reshape(-1, 3)[valid]
    return q[valid], p[valid], amp[valid], vq, vp
