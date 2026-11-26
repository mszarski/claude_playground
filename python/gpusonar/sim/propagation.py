"""Propagation: one interface, several providers.

An EigenrayProvider returns, for a source point and a set of field points, the
eigenrays joining them. Echo synthesis sums over every pair of outbound and
return eigenrays (as in SST; `two_way_paths`). The providers, by fidelity tier
(`make_provider`):

- FreeFieldProvider (`free_field`): the direct path only.
- ImageSourceProvider (`image_source`): constant sound speed, flat surface and bottom
  and optional vertical walls; the bottom is a fluid half-space (Rayleigh coefficient
  at each ray's grazing angle) or an explicit constant, never assumed (D-104); bounce
  orders kept by energy with a cap that warns (D-105). Opt-in (sonar-0nsi): a bottom
  sloping in x and y (`bottom_slope`) and walls as arbitrary planes ({point, normal}),
  with every image path checked for validity (`_PlaneImages`; in C++ too since sonar-k9v7).
- SedimentProvider (`sediment`): one refracted ray per point into a fluid sediment
  under a flat interface (Snell's law), for buried objects.
- sim.bellhop.BellhopProvider (`eigenray`): arrivals from Bellhop, an external program
  run as a separate process, for a sound-speed profile, a bathymetry and water that
  changes with range; its seabed is never assumed either.
- sim.timespread.TimeSpreadProvider wraps any of them to spread bounce paths in time
  (rough boundaries, D-62).

The synthesis kernel takes straight-line paths, so each eigenray is handed to it
as an *apparent position*: the point at distance c * delay from the sonar along
the ray's direction there. For the free-field and image-source providers that is
exact (it is the image position). For refracted and traced rays it reproduces the
delay and arrival angle at the array, with the wavefront curvature of a point at
the travel-time distance.

The free-field, image-source and sediment providers and `two_way_paths` also exist
in C++ (libs/sim/src/paths.cpp), which is used by default; `engine="numpy"` or
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
    """The direct path only, in water of constant `sound_speed` (m/s) with no boundaries: one
    eigenray per field point, delay r / c, amplitude 1 / r (spherical spreading, no absorption).
    Tier `free_field`. `engine` picks the C++ or NumPy code (sim/engine.py)."""

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

    Walls, two forms. {axis: 0 | 1, position, coefficient}: a vertical plane x or y = position, applied once on top
    of every image of the surface and bottom series, with no check that the path meets the wall or stays in the
    water (the form before sonar-0nsi; since sonar-8qn9 its arrival direction is mirrored in the wall as well).
    {point: [x, y, z], normal: [nx, ny, nz], coefficient[, bounds: {x: [lo, hi], y: [..], z: [..]}]}: any plane,
    `normal` pointing into the water. Opt-in with it (or with `bottom_slope`), every plane becomes a mirror of
    one general construction (`_PlaneImages`): the surface, the bottom (tilted by `bottom_slope` = [dz/dx, dz/dy],
    the depth at x = y = 0 being `bottom_z`, the convention of a scene `seafloor`'s `slope`) and the walls; image
    paths are the bounce sequences with at most `max_order` (or the energy rule's order) surface and bottom
    bounces and at most `wall_order` (1) wall bounces, and a path is valid only where every specular point lies on
    its plane's face (inside `bounds`), the source, the field point and every specular point are on the water side
    of every unbounded plane, and no leg crosses a bounded wall. Invalid paths are masked (`valid`, amplitude 0).
    Each bounce off the bottom takes the Rayleigh coefficient at its own grazing angle to the tilted bed. Both
    engines have the construction (C++: libs/sim paths.cpp `plane_image_eigenrays`, since sonar-k9v7; it agrees with
    the NumPy one to rounding, not to the bit: NumPy's dot products go through BLAS). The energy rule's order is
    worked out in NumPy for both engines, as for the flat series.
    """

    tier = "image_source"
    DEFAULT_ARRIVAL_FLOOR_DB = 20.0
    DEFAULT_MAX_ORDER_CAP = 8

    def __init__(self, sound_speed: float, surface_z: float | None = 0.0,
                 bottom_z: float | None = None, surface_coefficient: complex = -1.0,
                 bottom_coefficient: complex | None = None, max_order: int | None = None,
                 walls: list[dict] | None = None, bottom: dict | None = None,
                 engine: str | None = None, arrival_floor_db: float | None = None, max_order_cap: int | None = None,
                 bottom_slope=None, wall_order: int | None = None):
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
        for i, wall in enumerate(self.walls):
            _check_wall(wall, i)
        # Opt-in (sonar-0nsi): tilted bottom and walls as arbitrary planes, through `_PlaneImages`.
        self.bottom_slope = None if bottom_slope is None else tuple(float(v) for v in bottom_slope)
        if self.bottom_slope is not None and len(self.bottom_slope) != 2:
            raise ValueError("image_source: bottom_slope is [dz/dx, dz/dy] (two numbers)")
        self.general = self.bottom_slope is not None or any("normal" in w for w in self.walls)
        self.wall_order = None if wall_order is None else int(wall_order)
        self._planes = None
        if self.general:
            if any("axis" in w for w in self.walls):
                raise ValueError("image_source: with bottom_slope or a wall given as {point, normal}, give every wall as "
                                 "{point, normal, coefficient}: an {axis, position} wall does not say which side is water")
            if self.bottom_slope is not None and bottom_z is None:
                raise ValueError("image_source: bottom_slope tilts the bottom, and there is no bottom (bottom_z)")
            if self.wall_order is None:
                self.wall_order = 1
            if self.wall_order < 0:
                raise ValueError("image_source: wall_order must be 0 or more")
            self._planes = _PlaneImages(self)
        elif self.wall_order is not None:
            raise ValueError("image_source: wall_order applies to walls given as {point, normal, coefficient} (or with "
                             "bottom_slope); {axis, position} walls are applied once")

    def _order_for(self, source, points) -> int:
        """The bounce order of this call: `max_order` when it was given, else the energy criterion of the class
        docstring, worked out here for both engines on up to 256 of the field points (evenly taken, both ends in)."""
        if self.max_order is not None:
            return self.max_order
        if self.surface_z is None and self.bottom_z is None:
            return 0
        pts = points if len(points) <= 256 else points[np.unique(np.linspace(0, len(points) - 1, 256).astype(int))]
        probe = (self._planes.eigenrays(source, pts, self.max_order_cap + 1, 0) if self.general
                 else self._eigenrays_numpy(source, pts, self.max_order_cap + 1, walls=False))
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
        if self.general:
            if _cpp_points(self.engine, points):
                return self._planes.eigenrays_cpp(source, points, order, self.wall_order)
            return self._planes.eigenrays(source, points, order, self.wall_order)
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
            rays.append((img, coef, ns, nb, 1.0, None))
            for wall in self.walls if walls else []:
                wimg = img.copy()
                axis = int(wall["axis"])
                wimg[:, axis] = 2.0 * float(wall["position"]) - wimg[:, axis]
                wall_coef = complex(wall["coefficient"])
                rays.append((wimg, coef * wall_coef, ns, nb, wall_coef, axis))
        s, k = points.shape[0], len(rays)
        out = EigenraySet(
            delay=np.zeros((s, k)), amplitude=np.zeros((s, k), dtype=complex),
            launch=np.zeros((s, k, 3)), arrival=np.zeros((s, k, 3)),
            n_surface=np.zeros((s, k), dtype=int), n_bottom=np.zeros((s, k), dtype=int),
            valid=np.ones((s, k), dtype=bool), sound_speed=c)
        for j, (img, coef, ns, nb, wall_coef, wall_axis) in enumerate(rays):
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
            # Direction of travel at the real point: the direction to the image mirrored in every plane met, the
            # vertical component once per surface or bottom bounce and the wall's horizontal one at its bounce
            # (before sonar-8qn9 the wall's was left out).
            arr = u.copy()
            if (ns + nb) % 2 == 1:
                arr[:, 2] = -arr[:, 2]
            if wall_axis is not None:
                arr[:, wall_axis] = -arr[:, wall_axis]
            out.arrival[:, j] = arr
            out.n_surface[:, j] = ns
            out.n_bottom[:, j] = nb
        return out


def _fluid_bottom(speed: float, density: float, attenuation: float = 0.0, water_density: float = 1.0):
    """A sediment mapping as the tuple the C++ provider takes (the arguments of `rayleigh_reflection`)."""
    return float(speed), float(density), float(attenuation), float(water_density)


_AXIS_WALL_KEYS = ("axis", "position", "coefficient")
_PLANE_WALL_KEYS = ("point", "normal", "coefficient", "bounds")


def _refuse_unknown(given: dict, known, where: str) -> None:
    """A key not in `known` is refused by name with the nearest known key (the rule of D-166/D-177)."""
    import difflib
    for key in given:
        if key not in known:
            near = difflib.get_close_matches(str(key), [str(k) for k in known], n=1)
            raise ValueError(f"{where}{key}: unknown key" + (f"; did you mean {near[0]!r}?" if near else f" (known: {', '.join(known)})"))


def _check_wall(wall: dict, i: int) -> None:
    """A wall is {axis, position, coefficient} or {point, normal, coefficient[, bounds]}; anything else is refused."""
    where = f"image_source: walls[{i}]."
    if "point" in wall or "normal" in wall:
        _refuse_unknown(wall, _PLANE_WALL_KEYS, where)
        for key in ("point", "normal"):
            if key not in wall or np.shape(wall[key]) != (3,):
                raise ValueError(f"{where}{key}: three numbers needed")
        if not np.linalg.norm(np.asarray(wall["normal"], dtype=float)) > 0.0:
            raise ValueError(f"{where}normal: a zero vector")
        bounds = wall.get("bounds")
        if bounds is not None:
            _refuse_unknown(bounds, ("x", "y", "z"), where + "bounds.")
            for key, span in bounds.items():
                if np.shape(span) != (2,) or not float(span[0]) <= float(span[1]):
                    raise ValueError(f"{where}bounds.{key}: [low, high] needed")
    else:
        _refuse_unknown(wall, _AXIS_WALL_KEYS, where)


PLANE_TOLERANCE_M = 1e-6     # a point this close to a plane is on it (scatterers laid on a boundary are)


class _Plane:
    """One mirror: points x with n . (x - point) = 0, `normal` n (unit) pointing into the water."""

    def __init__(self, kind: str, point, normal, coefficient=None, bounds=None):
        self.kind = kind                                      # "surface", "bottom" or "wall"
        self.point = np.asarray(point, dtype=float)
        n = np.asarray(normal, dtype=float)
        self.normal = n / np.linalg.norm(n)
        self.coefficient = coefficient
        self.bounds = None if not bounds else [("xyz".index(k), float(v[0]), float(v[1])) for k, v in bounds.items()]
        # A plane normal to an axis mirrors as 2 c - x on that axis: the arithmetic of the flat series, so a level
        # bottom (`bottom_slope: [0, 0]`) gives its images to the bit.
        on_axis = np.flatnonzero(self.normal != 0.0)
        self.axis = int(on_axis[0]) if len(on_axis) == 1 else None

    def distance(self, x):
        """Signed distance of points [..., 3] from the plane, positive on the water side."""
        return (x - self.point) @ self.normal

    def mirror(self, x):
        if self.axis is not None:
            y = np.array(x, dtype=float, copy=True)
            y[..., self.axis] = 2.0 * self.point[self.axis] - y[..., self.axis]
            return y
        return x - 2.0 * self.distance(x)[..., None] * self.normal

    def on_face(self, x):
        """[S] bool: points (on the plane) inside its bounds; every point for an unbounded plane."""
        ok = np.ones(x.shape[0], dtype=bool)
        for axis, lo, hi in self.bounds or []:
            ok &= (x[:, axis] >= lo - PLANE_TOLERANCE_M) & (x[:, axis] <= hi + PLANE_TOLERANCE_M)
        return ok

    def blocks(self, a, b):
        """[S] bool: the segment a -> b passes through the plane's face (strictly: both ends off the plane)."""
        da, db = self.distance(a), self.distance(b)
        cross = ((da > PLANE_TOLERANCE_M) & (db < -PLANE_TOLERANCE_M)) | ((da < -PLANE_TOLERANCE_M) & (db > PLANE_TOLERANCE_M))
        if not cross.any():
            return cross
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(cross, da / (da - db), 0.0)
        return cross & self.on_face(a + t[:, None] * (b - a))


class _PlaneImages:
    """Image sources for any set of planes in water of one sound speed (sonar-0nsi; opt-in through
    `ImageSourceProvider(bottom_slope=..)` or a wall given as {point, normal}).

    A path from the source S to the field point P meeting planes p1, ..., pn in that order has the image
    I = R1(R2(...Rn(P))) (R: the mirror in a plane), and its length is |I - S|. The sequences are grown from P
    outwards, a mirror being applied only where the image so far lies on the water side of that plane (otherwise no
    path through it can exist), with no plane twice in a row, at most `order` surface and bottom bounces and at most
    `wall_order` wall bounces. Each sequence is then traced back from S: the straight line to the current image
    meets the next plane between its ends (the specular point), on its face; the source, P and every specular point
    lie on the water side of every unbounded plane (the water they bound is convex, so each leg then stays in it);
    no leg passes through a bounded wall. A path that fails any of these is invalid for that point (`valid` False,
    amplitude 0). One reflection off one plane is exact; between non-parallel planes (a wedge, a corner) the number
    of valid images is finite and the checks find them. Not modelled: diffraction at a face's edge (a bounded wall's
    end), and a path's shadowing by scene objects (as for direct paths, `Scene.visible` uses the straight line).

    Coefficients: the surface's, a wall's constant `coefficient`, and the bottom's (the Rayleigh coefficient of its
    fluid half-space at that bounce's own grazing angle to the tilted bed, or its constant)."""

    def __init__(self, provider: "ImageSourceProvider"):
        self.provider = provider
        planes = []
        if provider.surface_z is not None:
            planes.append(_Plane("surface", (0.0, 0.0, float(provider.surface_z)), (0.0, 0.0, 1.0), provider.surface_coefficient))
        if provider.bottom_z is not None:
            sx, sy = provider.bottom_slope or (0.0, 0.0)
            planes.append(_Plane("bottom", (0.0, 0.0, float(provider.bottom_z)), (sx, sy, -1.0)))
        for wall in provider.walls:
            planes.append(_Plane("wall", wall["point"], wall["normal"], complex(wall["coefficient"]), wall.get("bounds")))
        self.planes = planes

    def _reflection(self, plane: _Plane, sin_grazing):
        p = self.provider
        if plane.kind == "bottom":
            if p.bottom is not None:
                return rayleigh_reflection(np.arcsin(sin_grazing), p.sound_speed, **p.bottom)
            return np.full(len(sin_grazing), p.bottom_coefficient)
        return np.full(len(sin_grazing), plane.coefficient, dtype=complex)

    def sequences(self, points, order: int, wall_order: int):
        """[(plane indices in the order met from the source, image [S, 3], possible [S] bool)], the direct path first."""
        planes = self.planes
        tol = PLANE_TOLERANCE_M
        water = np.ones(len(points), dtype=bool)
        for plane in planes:
            if plane.bounds is None:
                water &= plane.distance(points) >= -tol
        nodes = [((), points, water)]
        frontier = nodes[:]
        while frontier:
            grown = []
            for seq, img, ok in frontier:
                n_wall = sum(planes[k].kind == "wall" for k in seq)
                n_vert = len(seq) - n_wall
                for k, plane in enumerate(planes):
                    if seq and seq[0] == k:
                        continue
                    if (n_wall + 1 > wall_order) if plane.kind == "wall" else (n_vert + 1 > order):
                        continue
                    m = ok & (plane.distance(img) >= -tol)
                    if m.any():
                        grown.append(((k,) + seq, plane.mirror(img), m))
            nodes += grown
            frontier = grown
        return nodes

    def eigenrays_cpp(self, source, points, order: int, wall_order: int) -> EigenraySet:
        """`eigenrays` by the C++ construction (libs/sim paths.cpp `plane_image_eigenrays`, sonar-k9v7): the same
        rays in the same order, to rounding. Each plane goes with the unit normal computed here, so that both
        engines mirror in the same plane."""
        p = self.provider
        planes = [(pl.kind, _source3(pl.point), _source3(pl.normal),
                   None if pl.kind == "bottom" else complex(pl.coefficient),
                   [(int(axis), float(lo), float(hi)) for axis, lo, hi in pl.bounds or []]) for pl in self.planes]
        return EigenraySet(**core.sim_plane_image_eigenrays(
            p.sound_speed, _source3(source), np.asarray(points, dtype=float), planes, int(order), int(wall_order),
            complex(p.bottom_coefficient), None if p.bottom is None else _fluid_bottom(**p.bottom)))

    def eigenrays(self, source, points, order: int, wall_order: int) -> EigenraySet:
        source = np.asarray(source, dtype=float)
        points = np.atleast_2d(np.asarray(points, dtype=float))
        c = self.provider.sound_speed
        planes = self.planes
        tol = PLANE_TOLERANCE_M
        unbounded = [p for p in planes if p.bounds is None]
        bounded = [p for p in planes if p.bounds is not None]
        source_in_water = all(p.distance(source) >= -tol for p in unbounded)
        nodes = self.sequences(points, order, wall_order)
        s, k = len(points), len(nodes)
        out = EigenraySet(
            delay=np.zeros((s, k)), amplitude=np.zeros((s, k), dtype=complex),
            launch=np.zeros((s, k, 3)), arrival=np.zeros((s, k, 3)),
            n_surface=np.zeros((s, k), dtype=int), n_bottom=np.zeros((s, k), dtype=int),
            valid=np.zeros((s, k), dtype=bool), sound_speed=c)
        for j, (seq, image, possible) in enumerate(nodes):
            r, u = _direct(source, image, c)
            valid = possible & source_in_water
            a = np.broadcast_to(source, image.shape)
            target = image
            direction = u.copy()
            coef = np.ones(s, dtype=complex)
            vertices = [a]
            for idx in seq:
                plane = planes[idx]
                da, dt = plane.distance(a), plane.distance(target)
                denom = da - dt
                with np.errstate(divide="ignore", invalid="ignore"):
                    t = np.where(denom > 0.0, da / np.where(denom > 0.0, denom, 1.0), -1.0)
                valid &= (denom > 0.0) & (t >= -1e-9) & (t <= 1.0 + 1e-9)
                x = a + t[:, None] * (target - a)
                valid &= plane.on_face(x)
                cos_n = direction @ plane.normal                    # negative: travelling into the plane
                coef = coef * self._reflection(plane, np.clip(np.abs(cos_n), 0.0, 1.0))
                direction = direction - 2.0 * cos_n[:, None] * plane.normal
                target = plane.mirror(target)                       # the image of P for the bounces still to come
                vertices.append(x)
                a = x
            vertices.append(points)
            for v in vertices[1:-1]:
                for plane in unbounded:
                    valid &= plane.distance(v) >= -tol
            for plane in bounded:
                for v0, v1 in zip(vertices[:-1], vertices[1:]):
                    valid &= ~plane.blocks(np.broadcast_to(v0, points.shape), v1)
            out.delay[:, j] = r / c
            out.amplitude[:, j] = np.where(valid, coef / r, 0.0)
            out.launch[:, j] = u
            out.arrival[:, j] = direction
            out.n_surface[:, j] = sum(planes[i].kind == "surface" for i in seq)
            out.n_bottom[:, j] = sum(planes[i].kind == "bottom" for i in seq)
            out.valid[:, j] = valid
        return out


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
    `transmission` factor for points below the interface. Not modelled: transmission
    that depends on the angle, and attenuation in the sediment.
    """

    tier = "sediment"

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
    Sources: docs/REFERENCES.md Sec. 12.4 (the dB/wavelength conversion is ours).
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
    """The eigenray provider of a fidelity tier, with `sound_speed` in m/s and the tier's
    `options` passed as keyword arguments:

    - `free_field`: FreeFieldProvider (direct path; options are ignored);
    - `image_source`: ImageSourceProvider (surface, bottom and walls; flat, or any planes);
    - `sediment`: SedimentProvider (one refracted ray into a fluid sediment);
    - `eigenray`: sim.bellhop.BellhopProvider; `options["environment"]` is the
      OceanEnvironment's keyword arguments and the rest go to the provider (`sound_speed`
      is not used: the environment's profile gives it);
    - `eigenray3d` (opt-in): sim.bellhop3d.Bellhop3DProvider (Bellhop3D over a 3-D
      `bathymetry`), the same `environment` and the provider's keywords; an unknown key is
      refused with the nearest known one.

    Any other tier raises ValueError."""
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
    if tier == "eigenray3d":
        # Bellhop3D (opt-in; sim/bellhop3d.py). Keys are checked here: a misspelt one is refused with the nearest.
        import difflib
        import inspect
        from .bellhop import OceanEnvironment
        from .bellhop3d import Bellhop3DProvider
        opts = dict(options)
        known = [k for k in inspect.signature(Bellhop3DProvider.__init__).parameters if k not in ("self", "env")] + ["environment"]
        for key in opts:
            if key not in known:
                near = difflib.get_close_matches(key, known, 1)
                raise ValueError(f"propagation.{key}: unknown key for fidelity eigenray3d"
                                 + (f"; did you mean {near[0]!r}?" if near else f" (known: {', '.join(sorted(known))})"))
        env = OceanEnvironment(**opts.pop("environment"))
        return Bellhop3DProvider(env, **opts)
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
