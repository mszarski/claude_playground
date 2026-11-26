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
            _engine.ran("eigenrays", "cpp")
            return EigenraySet(**_engine.call("sim_free_field_eigenrays", self.sound_speed, _source3(source), points))
        _engine.ran("eigenrays", "numpy")
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
    water (the form before sonar-0nsi; since sonar-8qn9 its arrival direction is mirrored in the wall as well). The
    axis is the integer 0 or 1 and the position a finite number; anything else (2, a negative value, a float, a
    bool, a string) is refused on both engines (a-wallaxis): the horizontal planes are `surface_z` and `bottom_z`.
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
        # An elastic row (`rock`, A8) adds shear_speed and shear_attenuation: the bounce is then
        # elastic, J&R (9.34)-(9.40); {sediment: rock, shear_speed: 0} is the same row as a fluid.
        if bottom is not None and "sediment" in bottom:
            from . import bottom as bottom_model
            rest = {k: v for k, v in bottom.items() if k not in ("sediment", "sediment_file")}
            bottom = dict(bottom_model.halfspace(bottom["sediment"], self.sound_speed, bottom.get("sediment_file")), **rest)
        if bottom is not None and not all(np.isfinite(float(bottom.get(k, 0.0))) and float(bottom.get(k, 0.0)) >= 0.0
                                          for k in ("shear_speed", "shear_attenuation")):
            raise ValueError("image_source: an elastic bottom needs shear_speed (m/s, 0 = a fluid) and shear_attenuation "
                             "(dB per wavelength) >= 0")
        self.bottom = bottom
        # Vertical walls: {"axis": 0 or 1, "position": metres, "coefficient": complex} (another axis is refused).
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
        # Field points are [S, 3] (or one point [3]) on both engines. Before a-wallaxis any other shape sent the C++
        # engine to the NumPy code without a word (`_cpp_points`), which then failed with an IndexError.
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"image_source: field points [S, 3] (or one point [3]) are needed, not shape {points.shape}")
        order = self._order_for(source, points)
        if self.general:
            if _cpp_points(self.engine, points):
                _engine.ran("eigenrays", "cpp")
                return self._planes.eigenrays_cpp(source, points, order, self.wall_order)
            _engine.ran("eigenrays", "numpy")
            return self._planes.eigenrays(source, points, order, self.wall_order)
        # Every wall is axis 0 or 1 (`_check_wall`), so the C++ engine, when selected, always runs; before a-wallaxis
        # another axis sent it to NumPy without a word.
        if _cpp_points(self.engine, points):
            _engine.ran("eigenrays", "cpp")
            walls = [(int(w["axis"]), float(w["position"]), complex(w["coefficient"])) for w in self.walls]
            return EigenraySet(**_engine.call("sim_image_source_eigenrays",
                self.sound_speed, _source3(source), points,
                surface_z=None if self.surface_z is None else float(self.surface_z),
                bottom_z=None if self.bottom_z is None else float(self.bottom_z),
                surface_coefficient=self.surface_coefficient, bottom_coefficient=self.bottom_coefficient,
                max_order=order, walls=walls,
                bottom=None if self.bottom is None else _fluid_bottom(**self.bottom)))
        _engine.ran("eigenrays", "numpy")
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


def _fluid_bottom(speed: float, density: float, attenuation: float = 0.0, water_density: float = 1.0,
                  shear_speed: float = 0.0, shear_attenuation: float = 0.0):
    """A sediment mapping as the tuple the C++ provider takes (the arguments of `rayleigh_reflection`); the
    four numbers of a fluid, six with an elastic bottom's shear."""
    fluid = float(speed), float(density), float(attenuation), float(water_density)
    return fluid if shear_speed == 0.0 else fluid + (float(shear_speed), float(shear_attenuation))


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
        # The old form is a vertical plane, x = position (axis 0) or y = position (axis 1), and nothing else (a-wallaxis).
        # Axis 2 would be a horizontal plane laid once over every image of the surface and bottom series: the images
        # of a plane at the surface or bottom again under another coefficient and with no bounce counted, and no valid
        # path for a plane between them; the horizontal planes are `surface_z` and `bottom_z`, and any other plane is
        # the {point, normal} form. Before, 2 and -1 to -3 ran in NumPy on both engines as an array index (2 and -1: z,
        # -2: y, -3: x), 3 and -4 raised an IndexError, and a float, bool or string was cut by int() (0.7 -> 0, "1" -> 1).
        axis = wall.get("axis")
        if isinstance(axis, (bool, np.bool_)) or not isinstance(axis, (int, np.integer)) or int(axis) not in (0, 1):
            raise ValueError(f"{where}axis: {axis!r} is not a wall axis; give 0 (a wall x = position) or 1 (a wall y = position). "
                             "The horizontal planes are surface_z and bottom_z; any other plane is a wall given as "
                             "{point, normal, coefficient}")
        try:
            position = float(wall["position"])
        except (KeyError, TypeError, ValueError):
            position = None
        if position is None or not np.isfinite(position):
            raise ValueError(f"{where}position: {wall.get('position')!r}; a wall {{axis, position, coefficient}} needs its "
                             "position (metres) as a finite number")


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
        return EigenraySet(**_engine.call("sim_plane_image_eigenrays",
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
        return _engine.call("sim_snell_crossing", _source3(source), points, float(c_water), float(c_sediment),
                            float(interface_z))
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


_SEDIMENT_KEYS = ("sediment_speed", "interface_z", "transmission", "engine", "penetration", "bottom", "frequency",
                  "subcritical", "roughness")
_FLUID_BOTTOM_KEYS = ("speed", "density", "attenuation", "water_density", "sediment", "sediment_file")
PENETRATION_MODELS = ("constant", "fluid", "rough")
SUBCRITICAL_POLICIES = ("evanescent", "ray", "refuse")


class SedimentProvider(EigenrayProvider):
    """Water over a fluid sediment half-space with a flat interface: one refracted
    eigenray per point (Snell's law). Used for buried objects.

    `penetration` says what the amplitude of a point below the interface carries on top of
    spherical spreading over the travel-time distance (c x delay):

    - `constant` (the default; the tier as it has always been): the constant `transmission`
      (default 1). No angle dependence, no density, no attenuation in the sediment.
    - `fluid` (opt-in, G-3 of docs/tasks/SST_GAPS_JACKSON.md): the flat-interface fluid field
      above the critical angle, Jackson and Richardson (2007) (15.1) along the ray: per leg
      V_wp(theta_w) / sqrt(a_rho) x exp(-k_p'' L). V_wp = 1 + V is the pressure transmission
      coefficient (8.47), (8.55) at the ray's grazing angle in the water, V the Rayleigh
      coefficient of `bottom` (`fluid_transmission`); a_rho the density ratio; L the length of
      the ray in the sediment; k_p'' = k_w delta / nu ((8.18), (8.26)) at `frequency` (the
      sonar's band centre in a scenario), delta from the bottom's dB per wavelength ((8.27)),
      nu = sediment speed / water speed. The echo of a buried point multiplies a leg from the
      transmitter and a leg from the receiver (`two_way_paths`), so it carries
      V_wp(theta_1) V_wp(theta_2) / a_rho exp(-k_p''(L_1 + L_2)): the reciprocity factor
      1 / a_rho of the return leg (8.68), split evenly between the legs so that the echo is the
      same with transmitter and receiver swapped. A ONE-WAY use of the tier (a radiating source
      under the interface, sim/sources.py) is therefore off by sqrt(a_rho) (the one-way field
      in the water from a source in the sediment carries 1 / a_rho; in the sediment from a
      source in the water, 1): it is said in docs/PHYSICS_NOTES.md.
      `bottom` is {speed, density, attenuation[, water_density]} or {sediment: name in
      configs/sediments.yaml}, as for the image-source tier; `attenuation` (dB per wavelength)
      must be stated: none is assumed. `sediment_speed`, when given, must equal the bottom's.
      Points whose straight line from the source meets the interface below the critical angle
      arccos(c_w / c_s) (lossless, (8.56); a slow sediment has none) are reached by no real
      refracted plane wave: the flat-interface field there is evanescent ([N3] Sec. 3.5); the
      least-time ray still found for them enters near the critical angle and runs almost
      horizontally through the sediment, and its level is not the field's. `subcritical` says
      what they get: `evanescent` (the default) the zeroth-order field (15.1) of the plane wave
      at the straight line's grazing angle theta_d: per leg V_wp(theta_d) / sqrt(a_rho)
      exp(-k_w Im{sqrt(1/a_p^2 - cos^2 theta_d)} d), d the point's depth under the interface
      ([N3] Sec. 3.5 item 2; the delay and directions stay the least-time ray's); `ray` the
      formula of the points above (for comparison only); `refuse` an error. With `evanescent` or
      `ray` a warning counts them, once per provider. Penetration by roughness scattering,
      which carries the measured sub-critical field (10-30 dB below the incident field, [N3]
      Sec. 2.7), is NOT modelled (G-13): sub-critical buried points are far too quiet. The
      amplitude jumps where a point crosses the criterion (the ray's leg above, the plane
      wave's decay below). Still approximate everywhere: the spreading (exact on the vertical
      only), one frequency for the attenuation, no layering, a frequency-independent speed.
    - `rough` (opt-in, G-13): `fluid`, and the sub-critical points get, on top of the evanescent
      zeroth order, the first-order roughness-scattered field of Jackson and Richardson (2007)
      Ch. 15 ((15.2)-(15.4), (15.11)-(15.16)) as ONE seeded realisation of a frozen rough
      interface: extra eigenrays per point, one per delay bin of interface cells
      (`sim/penetration.py` `RoughPatches`, which says what that means for imaging and what is
      still wrong). `roughness` gives the relief spectrum: {w2, gamma2} or {sediment: name},
      and K0 (1/m, von Karman (D.7); 0 the power law), stated; `seed` and the numerical keys
      `patch`, `refine`, `floor_db`, `bin` have documented defaults. Points above the critical
      angle get exactly the `fluid` ray (the scattered field there is at least 20 dB under the
      zeroth order, tests/python/test_sediment_roughness.py). `subcritical` must be absent or
      `evanescent`. Both engines: C++ `sediment_eigenrays` with `penetration.cpp`, the same
      interface for a seed (an explicit SplitMix64 hash, no generator) and the same arithmetic
      (tests/python/test_sim_cpp.py: equal to rounding).
    """

    tier = "sediment"

    def __init__(self, sound_speed: float, sediment_speed: float | None = None, interface_z: float | None = None,
                 transmission: complex | None = None, engine: str | None = None, penetration: str = "constant",
                 bottom: dict | None = None, frequency: float | None = None, subcritical: str | None = None,
                 roughness: dict | None = None):
        self.sound_speed = float(sound_speed)
        self.engine = engine
        if penetration not in PENETRATION_MODELS:
            near = __import__("difflib").get_close_matches(str(penetration), PENETRATION_MODELS, n=1)
            raise ValueError(f"sediment: penetration {penetration!r} unknown: one of {', '.join(PENETRATION_MODELS)}"
                             + (f"; did you mean {near[0]!r}?" if near else ""))
        self.penetration = penetration
        if interface_z is None:
            raise ValueError("sediment: interface_z (the depth of the water-sediment interface) is needed; there is no default")
        self.interface_z = float(interface_z)
        self.bottom = None
        self.frequency = None
        self.subcritical = None
        self.rough = None
        self.roughness = None
        if roughness is not None and penetration != "rough":
            raise ValueError("sediment: roughness is used only with penetration: rough")
        if penetration == "constant":
            for key, value in (("bottom", bottom), ("frequency", frequency), ("subcritical", subcritical)):
                if value is not None:
                    raise ValueError(f"sediment: {key} is used only with penetration: fluid (the constant "
                                     "`transmission` of the default carries no sediment)")
            if sediment_speed is None:
                raise ValueError("sediment: sediment_speed (m/s) is needed; there is no default")
            self.sediment_speed = float(sediment_speed)
            self.transmission = complex(1.0 if transmission is None else transmission)
            return
        # penetration: fluid
        if transmission is not None:
            raise ValueError("sediment: transmission is the constant of penetration: constant; with penetration: fluid "
                             "the interface's own coefficient is used")
        if bottom is None:
            raise ValueError("sediment: penetration: fluid needs the sediment as `bottom` ({speed, density, attenuation} "
                             "or {sediment: name in configs/sediments.yaml}, or the environment's `bottom`); none is assumed")
        _refuse_unknown(bottom, _FLUID_BOTTOM_KEYS, "sediment: bottom.")
        if "sediment" in bottom:
            from . import bottom as bottom_model
            rest = {k: v for k, v in bottom.items() if k not in ("sediment", "sediment_file")}
            bottom = dict(bottom_model.fluid_halfspace(bottom["sediment"], self.sound_speed, bottom.get("sediment_file")), **rest)
        for key in ("speed", "density", "attenuation"):
            if bottom.get(key) is None:
                raise ValueError(f"sediment: bottom.{key} is needed for penetration: fluid"
                                 + (" (dB per wavelength; no loss is assumed: give 0 for a lossless sediment)"
                                    if key == "attenuation" else "") + "; there is no default")
        self.bottom = {"speed": float(bottom["speed"]), "density": float(bottom["density"]),
                       "attenuation": float(bottom["attenuation"]), "water_density": float(bottom.get("water_density", 1.0))}
        b = self.bottom
        if not (b["speed"] > 0 and b["density"] > 0 and b["water_density"] > 0 and b["attenuation"] >= 0
                and all(np.isfinite(list(b.values())))):
            raise ValueError(f"sediment: bottom {b}: speed, density and water_density > 0, attenuation >= 0 dB per wavelength")
        if sediment_speed is not None and abs(float(sediment_speed) - b["speed"]) > 1e-9 * b["speed"]:
            raise ValueError(f"sediment: sediment_speed {float(sediment_speed)!r} differs from the bottom's speed {b['speed']!r}; "
                             "give one (or both equal)")
        self.sediment_speed = b["speed"]
        if frequency is None or not float(frequency) > 0:
            raise ValueError("sediment: penetration: fluid needs `frequency` (Hz, the band centre; a scenario fills in "
                             "the sonar's) for the attenuation in the sediment")
        self.frequency = float(frequency)
        policy = "evanescent" if subcritical is None else subcritical
        if policy not in SUBCRITICAL_POLICIES:
            raise ValueError(f"sediment: subcritical {subcritical!r}: one of {', '.join(SUBCRITICAL_POLICIES)}")
        self.subcritical = policy
        self.transmission = complex(1.0)
        self._warned = False
        if penetration == "rough":
            # G-13: the roughness-scattered field of the sub-critical points (sim/penetration.py; C++
            # libs/sim/src/penetration.cpp).
            from . import penetration as pen
            if policy != "evanescent":
                raise ValueError(f"sediment: penetration: rough adds the roughness-scattered field to the evanescent zeroth "
                                 f"order; subcritical {policy!r} does not apply (leave it out)")
            if roughness is None:
                raise ValueError("sediment: penetration: rough needs `roughness` ({w2, gamma2, K0} or {sediment: name, K0}); "
                                 "none is assumed")
            par = pen.roughness_from_config(dict(roughness))
            self.roughness = par
            delta = b["attenuation"] / (40.0 * np.pi * np.log10(np.e))
            self.rough = pen.RoughPatches(c_w=self.sound_speed, c_s=b["speed"], interface_z=self.interface_z,
                                          a_rho=b["density"] / b["water_density"], nu=b["speed"] / self.sound_speed,
                                          delta=delta, frequency=self.frequency, **par)

    def _check_subcritical(self, source, points) -> None:
        """Count the points under the interface whose straight line from `source` meets it below the
        critical angle (lossless, (8.56)); refuse, or warn once per provider, as `subcritical` says."""
        if self.sediment_speed <= self.sound_speed or (self._warned and self.subcritical != "refuse"):
            return                                   # a slow sediment has no critical angle
        below = (points[:, 2] > self.interface_z) & (source[2] < self.interface_z)
        if not below.any():
            return
        d = points[below] - source
        sin_direct = d[:, 2] / np.linalg.norm(d, axis=1)
        n = int(np.count_nonzero(sin_direct < np.sqrt(1.0 - (self.sound_speed / self.sediment_speed) ** 2)))
        if n == 0:
            return
        text = (f"sediment: {n} of {len(points)} points below the critical angle "
                f"({np.degrees(np.arccos(self.sound_speed / self.sediment_speed)):.1f} deg grazing on the straight line from "
                "the source): no real refracted plane wave reaches them. "
                + ("They get the evanescent flat-interface field (15.1) and one realisation of the roughness-scattered "
                   "field (penetration: rough, G-13)" if self.rough is not None
                   else "They get the evanescent flat-interface field (15.1); penetration by roughness scattering is not "
                   "modelled (G-13), so they are far too quiet" if self.subcritical == "evanescent"
                   else "They keep the least-time ray's level, which is not the field's (20-30 dB too loud)")
                + ". Said once per provider")
        if self.subcritical == "refuse":
            raise ValueError(text)
        self._warned = True
        import warnings
        warnings.warn(text, RuntimeWarning, stacklevel=3)

    def eigenrays(self, source, points) -> EigenraySet:
        source = np.asarray(source, dtype=float)
        points = np.atleast_2d(np.asarray(points, dtype=float))
        fluid = self.penetration in ("fluid", "rough")
        if fluid:
            self._check_subcritical(source, points)
        if _cpp_points(self.engine, points):
            _engine.ran("eigenrays", "cpp")
            if fluid:
                return EigenraySet(**_engine.call(
                    "sim_sediment_eigenrays",
                    self.sound_speed, self.sediment_speed, self.interface_z, self.transmission, _source3(source), points,
                    bottom=_fluid_bottom(**self.bottom), frequency=self.frequency,
                    evanescent=self.subcritical == "evanescent",
                    roughness=None if self.rough is None else self.rough.cpp_args()))
            return EigenraySet(**_engine.call("sim_sediment_eigenrays", self.sound_speed, self.sediment_speed,
                                              self.interface_z, self.transmission, _source3(source), points))
        _engine.ran("eigenrays", "numpy")
        t, cross = snell_crossing(source, points, self.sound_speed, self.sediment_speed, self.interface_z,
                                  engine="numpy")
        launch = cross - source
        launch /= np.linalg.norm(launch, axis=1)[:, None]
        below = points[:, 2] > self.interface_z
        arrival = points - cross
        norm = np.linalg.norm(arrival, axis=1)
        arrival = np.where(norm[:, None] > 0, arrival / np.maximum(norm, 1e-300)[:, None], launch)
        s = len(points)
        if fluid:
            # Per leg: V_wp(theta_w) / sqrt(a_rho) exp(-k_p'' L) (class docstring). The C++ engine
            # (libs/sim/src/paths.cpp sediment_eigenrays) does the same operations in the same order.
            b = self.bottom
            grazing = np.arcsin(np.clip(np.abs(launch[:, 2]), 0.0, 1.0))
            coef = fluid_transmission(grazing, self.sound_speed, **b) / np.sqrt(b["density"] / b["water_density"])
            delta = b["attenuation"] / (40.0 * np.pi * np.log10(np.e))
            k_loss = 2.0 * np.pi * self.frequency / self.sound_speed * delta / (b["speed"] / self.sound_speed)
            factor = coef * np.exp(-k_loss * norm)
            if self.subcritical == "evanescent" and b["speed"] > self.sound_speed and source[2] < self.interface_z:
                # Below the critical angle on the straight line from the source: the plane wave's zeroth-order
                # field (15.1), decaying with depth at k_w Im{sqrt(1/a_p^2 - cos^2 theta_d)} ([N3] Sec. 3.5).
                d = points - source
                sin_d = d[:, 2] / np.linalg.norm(d, axis=1)
                sub = below & (sin_d < np.sqrt(1.0 - (self.sound_speed / b["speed"]) ** 2))
                if sub.any():
                    g_d = np.arcsin(np.clip(sin_d, 0.0, 1.0))
                    kappa = (1.0 + 1j * delta) / (b["speed"] / self.sound_speed)          # 1 / a_p
                    root = np.sqrt(kappa**2 - np.cos(g_d) ** 2 + 0j)
                    root = np.where(root.imag < 0, -root, root)
                    k_w = 2.0 * np.pi * self.frequency / self.sound_speed
                    evan = (fluid_transmission(g_d, self.sound_speed, **b) / np.sqrt(b["density"] / b["water_density"])
                            * np.exp(-k_w * root.imag * (points[:, 2] - self.interface_z)))
                    factor = np.where(sub, evan, factor)
            amp = np.where(below, factor, 1.0) / (self.sound_speed * t)
        else:
            amp = np.where(below, self.transmission, 1.0) / (self.sound_speed * t)
        if self.rough is not None:
            return self._with_scattered(source, points, t, amp, launch, arrival, below)
        return EigenraySet(
            delay=t[:, None], amplitude=amp[:, None].astype(complex), launch=launch[:, None, :],
            arrival=arrival[:, None, :], n_surface=np.zeros((s, 1), dtype=int),
            n_bottom=np.zeros((s, 1), dtype=int), valid=np.ones((s, 1), dtype=bool),
            sound_speed=self.sound_speed)

    def _with_scattered(self, source, points, t, amp, launch, arrival, below) -> EigenraySet:
        """`rough` (G-13): column 0 is the `fluid` ray; the sub-critical points get further columns, one per delay bin
        of the roughness-scattered realisation (`penetration.RoughPatches.point_rays`); the rest are invalid."""
        s = len(points)
        sub = np.zeros(s, dtype=bool)
        if self.sediment_speed > self.sound_speed and source[2] < self.interface_z:
            d = points - source
            sin_d = d[:, 2] / np.linalg.norm(d, axis=1)
            sub = below & (sin_d < np.sqrt(1.0 - (self.sound_speed / self.sediment_speed) ** 2))
        rays = {int(k): self.rough.point_rays(source, points[k]) for k in np.flatnonzero(sub)}
        # the cost cap (penetration.rough_cost_check): raise before any pairing, never truncate
        from . import penetration as pen
        pen.rough_cost_check([1 + len(r[0]) for r in rays.values()], list(rays), self.rough.max_pairs)
        extra = max((len(r[0]) for r in rays.values()), default=0)
        k = 1 + extra
        delay = np.zeros((s, k))
        amplitude = np.zeros((s, k), dtype=complex)
        lau = np.zeros((s, k, 3))
        arr = np.zeros((s, k, 3))
        valid = np.zeros((s, k), dtype=bool)
        delay[:, 0], amplitude[:, 0], lau[:, 0], arr[:, 0], valid[:, 0] = t, amp, launch, arrival, True
        lau[:, 1:], arr[:, 1:] = launch[:, None, :], arrival[:, None, :]
        for i, (t_b, a_b, l_b, r_b) in rays.items():
            n = len(t_b)
            delay[i, 1:1 + n], amplitude[i, 1:1 + n], lau[i, 1:1 + n], arr[i, 1:1 + n] = t_b, a_b, l_b, r_b
            valid[i, 1:1 + n] = True
        return EigenraySet(delay=delay, amplitude=amplitude, launch=lau, arrival=arr,
                           n_surface=np.zeros((s, k), dtype=int), n_bottom=np.zeros((s, k), dtype=int), valid=valid,
                           sound_speed=self.sound_speed)


def rayleigh_reflection(grazing, c_water: float, speed: float, density: float,
                        attenuation: float = 0.0, water_density: float = 1.0,
                        shear_speed: float = 0.0, shear_attenuation: float = 0.0):
    """Plane-wave pressure reflection coefficient of a fluid half-space (Rayleigh), or of an
    elastic one when `shear_speed` > 0.

    `grazing` in radians; `density` in the same units as `water_density`; `attenuation` in
    dB per wavelength, entering as an imaginary part of the sediment sound speed.
    R = (m sin(g) - sqrt(n^2 - cos^2(g))) / (m sin(g) + sqrt(n^2 - cos^2(g))),
    m = rho_b / rho_w, n = c_w / c_b.
    Sources: docs/REFERENCES.md Sec. 12.4 (the dB/wavelength conversion is ours).

    Elastic (A8; `shear_speed` m/s, `shear_attenuation` dB per wavelength of the shear wave):
    Jackson and Richardson (2007) (9.34)-(9.40), as `bottom.reflection` with nu_t = shear_speed /
    c_water: with n_t = c_w / c_t and the decaying roots p = sqrt(n^2 - cos^2 g),
    s = sqrt(n_t^2 - cos^2 g), and C = cos 2theta_t = 2 cos^2 g / n_t^2 - 1,
    R = (m sin g (C^2 s + (1 - C^2) p) - p s) / (m sin g (C^2 s + (1 - C^2) p) + p s).
    `shear_speed` = 0 (the default) is the fluid expression above, unchanged.
    """
    g = np.asarray(grazing, dtype=float)
    delta = attenuation / (40.0 * np.pi * np.log10(np.e))      # loss tangent from dB/wavelength
    c_b = speed / (1.0 + 1j * delta)
    m = density / water_density
    n = c_water / c_b
    root = np.sqrt(n**2 - np.cos(g) ** 2 + 0j)
    root = np.where(root.imag < 0, -root, root)                 # decaying branch in the sediment
    if shear_speed == 0.0:
        return (m * np.sin(g) - root) / (m * np.sin(g) + root)
    delta_t = shear_attenuation / (40.0 * np.pi * np.log10(np.e))
    n_t = c_water * (1.0 + 1j * delta_t) / shear_speed
    s = np.sqrt(n_t**2 - np.cos(g) ** 2 + 0j)
    s = np.where(s.imag < 0, -s, s)
    cos2t = 2.0 * np.cos(g) ** 2 / n_t**2 - 1.0
    num = m * np.sin(g) * (cos2t**2 * s + (1.0 - cos2t**2) * root)
    return (num - root * s) / (num + root * s)


def fluid_transmission(grazing, c_water: float, speed: float, density: float,
                       attenuation: float = 0.0, water_density: float = 1.0):
    """Plane-wave pressure transmission coefficient from the water into a fluid half-space:
    V_wp = 1 + V (Jackson and Richardson 2007, (8.47); = 2 z_wp / (z_wp + 1), (8.55)), with V
    `rayleigh_reflection` (equal to (8.49)-(8.53)). The pressure just below the interface per
    unit incident pressure, at a grazing angle in the water (radians). Above the critical angle
    the field it starts is refracted; below it, evanescent. A pressure ratio: lossless, the
    energy flux it carries is 1 - |V|^2 = |V_wp|^2 Re{sin theta_p / (a_rho a_p)} / sin theta_w
    (tests/python/test_sediment_penetration.py)."""
    return 1.0 + rayleigh_reflection(grazing, c_water, speed, density, attenuation, water_density)


def _with_engine(provider: EigenrayProvider, engine: str | None) -> EigenrayProvider:
    if engine is not None:
        provider.engine = engine
    return provider


def make_provider(tier: str, sound_speed: float, options: dict | None = None,
                  engine: str | None = None) -> EigenrayProvider:
    """The eigenray provider of a fidelity tier, with `sound_speed` in m/s and the tier's
    `options` passed as keyword arguments:

    - `free_field`: FreeFieldProvider (direct path; options are ignored);
    - `image_source`: ImageSourceProvider (surface, bottom and walls; flat, or any planes);
    - `sediment`: SedimentProvider (one refracted ray into a fluid sediment; an unknown key is
      refused with the nearest known one);
    - `eigenray`: sim.bellhop.BellhopProvider; `options["environment"]` is the
      OceanEnvironment's keyword arguments and the rest go to the provider (`sound_speed`
      is not used: the environment's profile gives it);
    - `eigenray3d` (opt-in): sim.bellhop3d.Bellhop3DProvider (Bellhop3D over a 3-D
      `bathymetry`), the same `environment` and the provider's keywords; an unknown key is
      refused with the nearest known one.

    `engine` ("cpp" or "numpy"): the engine of the free_field, image_source and sediment
    providers, in place of an `engine` key of `options` (a-engarg: the caller's engine choice
    reaches the providers the simulator builds for itself); None (the default) leaves it to
    the options, else to sim/engine.py's default. Bellhop has one implementation: ignored.

    Any other tier raises ValueError."""
    options = options or {}
    if tier == "free_field":
        return FreeFieldProvider(sound_speed, engine=engine)
    if tier == "image_source":
        return _with_engine(ImageSourceProvider(sound_speed, **options), engine)
    if tier == "sediment":
        _refuse_unknown(options, _SEDIMENT_KEYS, "sediment: ")
        return _with_engine(SedimentProvider(sound_speed, **options), engine)
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
        _engine.ran("pairing", "cpp")
        weighted = lambert is not None and bool(np.any(lambert))
        moving = velocities is not None and bool(np.any(velocities))
        args = (out.delay, out.amplitude, out.launch, out.arrival, out.valid, float(out.sound_speed),
                back.delay, back.amplitude, back.launch, back.arrival, back.valid, float(back.sound_speed),
                _source3(tx_position), _source3(rx_reference), np.asarray(amplitudes, dtype=complex))
        kwargs = dict(velocities=np.asarray(velocities, dtype=float) if moving else None,
                      lambert=np.asarray(lambert, dtype=bool) if weighted else None,
                      normals=np.asarray(normals, dtype=float) if weighted and normals is not None else None)
        if doppler:
            return _engine.call("sim_two_way_paths", *args, **kwargs, out_range_rate=rates[0],
                                back_range_rate=rates[1])
        q, p, amp, vq = _engine.call("sim_two_way_paths", *args, **kwargs)
        return q, p, amp, vq, (None if vq is None else vq.copy())
    _engine.ran("pairing", "numpy")
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
