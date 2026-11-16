"""Scenes: collections of point scatterers with amplitude and velocity.

Every scene object reduces to point scatterers for synthesis. Extended objects (seafloor
with relief, plates, walls, spheres, boxes, rocks) are generators that emit surface points
with a normal each, and may register an occluder.

Surface points scatter by Lambert's law about their own normal, so a face turned away
from the sonar returns nothing. Occluders (spheres and axis-aligned boxes) block the
straight line between the sonar and any scatterer behind them, which is what puts an
acoustic shadow behind an object. Both use straight lines from the sonar: with a
multipath provider, boundary-reflected paths are shadowed as if they were direct.
Specular highlights and diffraction are not modelled.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Scene:
    positions: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))  # NED, m
    amplitudes: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=complex))
    velocities: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))  # m/s
    labels: list[str] = field(default_factory=list)  # one per added object, for ground truth
    counts: list[int] = field(default_factory=list)  # scatterers each of those added
    # Per scatterer: 1 = a boundary patch whose amplitude scales with sqrt(sin(grazing_in) *
    # sin(grazing_out)) (Lambert's law); 0 = a point target with a fixed amplitude.
    lambert: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    # Per scatterer: unit surface normal (pointing into the water), or NaN for none. A
    # Lambert scatterer without a normal is a patch of a horizontal boundary.
    normals: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    # ("sphere", centre [3], radius), ("box", min [3], max [3]) and ("capsule", (end_a, end_b),
    # radius) primitives that block sound.
    occluders: list = field(default_factory=list)
    # Surfaces that change shape with time (waves): (first index, count, fn) where
    # fn(x [n], y [n], t) returns (z [n], normals [n, 3]) for those scatterers.
    dynamic: list = field(default_factory=list)
    # Scatterers whose echo does not keep its phase from ping to ping (small-scale surface
    # roughness, bubbles): (first index, count, rms amplitude, seed). Their complex
    # amplitudes are drawn afresh for every ping time.
    redrawn: list = field(default_factory=list)

    def __len__(self) -> int:
        return self.positions.shape[0]

    def add_points(self, positions, amplitude=1.0, velocity=(0.0, 0.0, 0.0), label="points",
                   target_strength_db=None, lambert: bool = False, normals=None) -> "Scene":
        """Add scatterers. `amplitude` is the pressure ratio at 1 m (10^(TS/20))."""
        pos = np.atleast_2d(np.asarray(positions, dtype=float))
        if pos.shape[1] != 3:
            raise ValueError("positions must be [n, 3]")
        n = pos.shape[0]
        if target_strength_db is not None:
            amplitude = 10.0 ** (np.asarray(target_strength_db, dtype=float) / 20.0)
        amp = np.broadcast_to(np.asarray(amplitude, dtype=complex), (n,)).copy()
        vel = np.broadcast_to(np.asarray(velocity, dtype=float), (n, 3)).copy()
        self.positions = np.vstack([self.positions, pos])
        self.amplitudes = np.concatenate([self.amplitudes, amp])
        self.velocities = np.vstack([self.velocities, vel])
        self.lambert = np.concatenate([self.lambert, np.full(n, bool(lambert))])
        nrm = np.full((n, 3), np.nan) if normals is None else np.broadcast_to(
            np.asarray(normals, dtype=float), (n, 3)).copy()
        self.normals = np.vstack([self.normals, nrm])
        self.labels.append(label)
        self.counts.append(n)
        return self

    def label_masks(self) -> dict:
        """{label: boolean mask over the scatterers}; pieces sharing a label are merged."""
        out, start = {}, 0
        for label, n in zip(self.labels, self.counts):
            out.setdefault(label, np.zeros(len(self), dtype=bool))[start:start + n] = True
            start += n
        return out

    def add_surface(self, positions, normals, mu_db: float, density: float, label: str,
                    rng: np.random.Generator | None = None, velocity=(0.0, 0.0, 0.0)) -> "Scene":
        """Add Lambert surface points: complex Gaussian amplitudes scaled so `density`
        points per square metre give the backscattering strength mu + 10 log10(cos_in cos_out),
        the cosines taken from each point's normal."""
        rng = rng or np.random.default_rng()
        n = len(positions)
        amp = lambert_rms_amplitude(mu_db, density) * (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2.0)
        return self.add_points(positions, amp, velocity, label, lambert=True, normals=normals)

    def visible(self, sensor, positions: np.ndarray | None = None) -> np.ndarray:
        """[P] bool: the straight line from `sensor` to each scatterer misses every occluder."""
        pts = self.positions if positions is None else positions
        ok = np.ones(len(pts), dtype=bool)
        s = np.asarray(sensor, dtype=float)
        d = pts - s                                           # segment s + t d, t in [0, 1]
        t_end = 1.0 - 1e-4                                    # stop just short of the scatterer itself
        for kind, a, b in self.occluders:
            if kind == "sphere":
                c, r = np.asarray(a, dtype=float), 0.999 * float(b)
                t = np.clip(((c - s) * d).sum(axis=1) / np.maximum((d * d).sum(axis=1), 1e-30), 0.0, t_end)
                ok &= np.linalg.norm(s + t[:, None] * d - c, axis=1) >= r
            elif kind == "box":
                lo = np.asarray(a, dtype=float) + 1e-4
                hi = np.asarray(b, dtype=float) - 1e-4
                with np.errstate(divide="ignore", invalid="ignore"):
                    t0 = (lo - s) / d
                    t1 = (hi - s) / d
                near = np.where(np.abs(d) > 0, np.minimum(t0, t1), np.where((s >= lo) & (s <= hi), -np.inf, np.inf))
                far = np.where(np.abs(d) > 0, np.maximum(t0, t1), np.where((s >= lo) & (s <= hi), np.inf, -np.inf))
                enter, leave = near.max(axis=1), far.min(axis=1)
                ok &= ~((enter < leave) & (leave > 0.0) & (enter < t_end))
            elif kind == "capsule":
                # A cylinder with rounded ends: closest approach of the two segments.
                p0, p1 = (np.asarray(v, dtype=float) for v in a)
                e, r = p1 - p0, 0.999 * float(b)
                w = s - p0
                aa = np.maximum((d * d).sum(axis=1), 1e-30)
                ee = float(e @ e)
                bb, cc, ff = d @ e, d @ w, float(e @ w)
                denom = aa * ee - bb * bb
                t = np.where(denom > 1e-12 * aa * ee, np.clip((bb * ff - cc * ee) / np.maximum(denom, 1e-30), 0.0, t_end), 0.0)
                u = (bb * t + ff) / ee
                t = np.where(u < 0.0, np.clip(-cc / aa, 0.0, t_end), np.where(u > 1.0, np.clip((bb - cc) / aa, 0.0, t_end), t))
                u = np.clip(u, 0.0, 1.0)
                ok &= np.linalg.norm(w + t[:, None] * d - u[:, None] * e, axis=1) >= r
            else:
                raise ValueError(f"unknown occluder {kind!r}")
        return ok

    def at_time(self, t: float) -> np.ndarray:
        """Scatterer positions at scenario time t: first-order motion, then the shape of
        any time-varying surface."""
        points = self.positions + self.velocities * t
        for first, count, fn in self.dynamic:
            points[first:first + count, 2] = fn(points[first:first + count, 0], points[first:first + count, 1], t)[0]
        return points

    def amplitudes_at(self, t: float) -> np.ndarray:
        """Scatterer amplitudes at scenario time t."""
        if not self.redrawn:
            return self.amplitudes
        amp = self.amplitudes.copy()
        for first, count, rms, seed in self.redrawn:
            rng = np.random.default_rng([int(seed), int(round(t * 1e6))])
            amp[first:first + count] = rms * (rng.standard_normal(count) + 1j * rng.standard_normal(count)) / np.sqrt(2.0)
        return amp

    def normals_at(self, t: float) -> np.ndarray:
        """Surface normals at scenario time t."""
        if not self.dynamic:
            return self.normals
        normals = self.normals.copy()
        moved = self.positions + self.velocities * t
        for first, count, fn in self.dynamic:
            keep = ~np.isnan(normals[first:first + count, 0])       # scatterers without a normal stay without
            normals[first:first + count][keep] = fn(moved[first:first + count, 0], moved[first:first + count, 1], t)[1][keep]
        return normals


def polar_grid(origin, forward, across, ranges, bearings_deg, depth_offset=0.0) -> np.ndarray:
    """Points on a range x bearing fan: origin + r (cos(b) forward + sin(b) across)."""
    origin = np.asarray(origin, dtype=float)
    f = np.asarray(forward, dtype=float)
    a = np.asarray(across, dtype=float)
    r, b = np.meshgrid(np.asarray(ranges, dtype=float), np.radians(bearings_deg), indexing="ij")
    pts = origin + r[..., None] * (np.cos(b)[..., None] * f + np.sin(b)[..., None] * a)
    pts = pts.reshape(-1, 3)
    pts[:, 2] += depth_offset
    return pts


def lambert_rms_amplitude(mu_db: float, density: float) -> float:
    """RMS amplitude of each scatterer so that a patch of `density` scatterers per square
    metre has the Lambert backscattering strength S = mu + 10 log10(sin(g_in) sin(g_out)).

    The point-scatterer model of reverberation: the variance of the summed amplitude from
    an area dA is (scattering cross-section per unit area) x dA. The angular factor is
    applied per path at synthesis time (Scene.lambert).
    """
    return float(np.sqrt(10.0 ** (mu_db / 10.0) / density))


def seafloor_patch(x_range, y_range, density: float, depth: float = 0.0, rms_amplitude: float = 1.0,
                   rng: np.random.Generator | None = None, bathymetry=None):
    """Random point scatterers on the seafloor: the point-scatterer model of reverberation.

    Positions are uniform over the patch, `density` per square metre; amplitudes are
    complex Gaussian, which gives fully developed (Rayleigh) speckle once several
    scatterers share a resolution cell. `bathymetry(x, y)` returns the seafloor depth
    offset from `depth` (positive down) when the floor is not flat.
    Returns (positions [n, 3], amplitudes [n]).
    """
    rng = rng or np.random.default_rng()
    area = (x_range[1] - x_range[0]) * (y_range[1] - y_range[0])
    n = int(round(area * density))
    x = rng.uniform(x_range[0], x_range[1], n)
    y = rng.uniform(y_range[0], y_range[1], n)
    z = np.full(n, float(depth))
    if bathymetry is not None:
        z = z + bathymetry(x, y)
    amp = rms_amplitude * (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2.0)
    return np.column_stack([x, y, z]), amp


def wire(start, end, spacing: float) -> np.ndarray:
    """Points along a thin wire or line from `start` to `end`, `spacing` apart."""
    a, b = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
    n = max(int(np.ceil(np.linalg.norm(b - a) / spacing)) + 1, 2)
    return a + np.linspace(0.0, 1.0, n)[:, None] * (b - a)


def plate_points(origin, edge_u, edge_v, density: float, rng: np.random.Generator):
    """Random points on the parallelogram origin + a u + b v (a, b in [0, 1]) and its unit
    normal u x v. Returns (positions [n, 3], normal [3])."""
    o, u, v = (np.asarray(x, dtype=float) for x in (origin, edge_u, edge_v))
    cross = np.cross(u, v)
    area = np.linalg.norm(cross)
    n = max(int(round(area * density)), 1)
    ab = rng.uniform(0.0, 1.0, (n, 2))
    return o + ab[:, :1] * u + ab[:, 1:] * v, cross / area


def sphere_points(centre, radius: float, density: float, rng: np.random.Generator):
    """Random points on a sphere and their outward normals."""
    n = max(int(round(4.0 * np.pi * radius**2 * density)), 8)
    v = rng.standard_normal((n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return np.asarray(centre, dtype=float) + radius * v, v


def cylinder_points(end_a, end_b, radius: float, density: float, rng: np.random.Generator,
                    caps: bool = True):
    """Random points on a cylinder between two end points, with outward normals."""
    a, b = np.asarray(end_a, dtype=float), np.asarray(end_b, dtype=float)
    axis = b - a
    length = np.linalg.norm(axis)
    axis = axis / length
    ref = np.array([0.0, 0.0, 1.0]) if abs(axis[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(axis, ref)
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    n = max(int(round(2.0 * np.pi * radius * length * density)), 8)
    phi, along = rng.uniform(0.0, 2.0 * np.pi, n), rng.uniform(0.0, length, n)
    radial = np.cos(phi)[:, None] * u + np.sin(phi)[:, None] * v
    pts, normals = [a + along[:, None] * axis + radius * radial], [radial]
    if caps:
        m = max(int(round(np.pi * radius**2 * density)), 1)
        for centre, sign in ((a, -1.0), (b, 1.0)):
            rr, ph = radius * np.sqrt(rng.uniform(0.0, 1.0, m)), rng.uniform(0.0, 2.0 * np.pi, m)
            pts.append(centre + rr[:, None] * (np.cos(ph)[:, None] * u + np.sin(ph)[:, None] * v))
            normals.append(np.tile(sign * axis, (m, 1)))
    return np.vstack(pts), np.vstack(normals)


def hull_points(origin, length: float, beam: float, depth: float, heading_deg: float, n: int,
                rng: np.random.Generator, keel_down: bool = True):
    """Random points on a simple ship hull, with outward normals and the area each stands for.

    The hull runs `length` from stern to bow along `heading_deg` (0 = +x, 90 = +y) with
    `origin` at the middle of its deck line. Its half-breadth tapers to both ends,
    b(u) = beam / 2 * sqrt(1 - |2u - 1|^3), and each section is half an ellipse `depth`
    deep. With `keel_down` the deck edge is at origin z and the keel `depth` below it (a
    floating boat with origin z at the waterline shows its underwater body; a wreck sitting
    upright on the seabed has origin z = seabed - depth). Returns (points, normals, areas).
    """
    u = rng.uniform(0.02, 0.98, n)
    th = rng.uniform(0.0, np.pi, n)
    sign = 1.0 if keel_down else -1.0

    def surface(u_, th_):
        half = 0.5 * beam * np.sqrt(np.maximum(1.0 - np.abs(2.0 * u_ - 1.0) ** 3, 0.0))
        return np.stack([(u_ - 0.5) * length, half * np.cos(th_), sign * depth * np.sin(th_) * (half / (0.5 * beam)) ** 0.5],
                        axis=-1)

    p = surface(u, th)
    eps = 1e-4
    du = (surface(u + eps, th) - surface(u - eps, th)) / (2 * eps)
    dth = (surface(u, th + eps) - surface(u, th - eps)) / (2 * eps)
    cross = np.cross(du, dth)
    jac = np.linalg.norm(cross, axis=1)
    normals = cross / np.maximum(jac, 1e-12)[:, None]
    outward = (normals * (p - np.array([0.0, 0.0, 0.0]))).sum(axis=1) < 0
    normals[outward] *= -1.0                              # away from the centreline
    areas = jac * (0.96 * np.pi) / n
    h = np.radians(heading_deg)
    rot = np.array([[np.cos(h), -np.sin(h), 0.0], [np.sin(h), np.cos(h), 0.0], [0.0, 0.0, 1.0]])
    return np.asarray(origin, dtype=float) + p @ rot.T, normals @ rot.T, areas


def box_faces(lo, hi):
    """The six faces of an axis-aligned box as (origin, edge_u, edge_v) with outward normals u x v."""
    lo, hi = np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)
    e = np.diag(hi - lo)
    faces = []
    for axis in range(3):
        u, v = e[(axis + 1) % 3], e[(axis + 2) % 3]
        faces.append((lo, v, u))                         # normal -axis
        faces.append((lo + e[axis], u, v))               # normal +axis
    return faces


def relief(slope=(0.0, 0.0), ripples: dict | None = None):
    """Seafloor depth offset z(x, y) (positive down) and its gradient, for a tilted plane
    plus sinusoidal ripples {amplitude, wavelength, direction_deg}."""
    sx, sy = (float(v) for v in slope)
    amp = float(ripples["amplitude"]) if ripples else 0.0
    k = 2.0 * np.pi / float(ripples["wavelength"]) if ripples else 0.0
    th = np.radians(float(ripples.get("direction_deg", 0.0))) if ripples else 0.0
    cx, cy = np.cos(th), np.sin(th)

    def height(x, y):
        return sx * x + sy * y + amp * np.sin(k * (cx * x + cy * y))

    def gradient(x, y):
        g = amp * k * np.cos(k * (cx * x + cy * y))
        return sx + g * cx, sy + g * cy

    return height, gradient


GRAVITY = 9.81


def wave_components(item: dict, rng: np.random.Generator) -> np.ndarray:
    """Wave trains of a `sea_surface` item as rows [amplitude, kx, ky, omega, phase].

    `waves: [{amplitude, wavelength, direction_deg, phase}]` gives regular trains.
    `spectrum: {wind_speed, significant_height, direction_deg, spread_deg, components}`
    gives an irregular wind sea: a Pierson-Moskowitz spectrum for that wind (m/s), sampled
    at `components` frequencies from 0.6 to 5 times the peak, each train with a random
    phase and a direction drawn about the mean with `spread_deg` standard deviation, the
    whole scaled to `significant_height` when that is given (a sea not fully developed).
    Every train travels at its deep-water speed, omega^2 = g k, unless `frozen: true`.
    """
    rows = []
    for w in item.get("waves", []):
        k = 2.0 * np.pi / float(w["wavelength"])
        th = np.radians(float(w.get("direction_deg", 0.0)))
        rows.append([float(w["amplitude"]), k * np.cos(th), k * np.sin(th), np.sqrt(GRAVITY * k), float(w.get("phase", 0.0))])
    spec = item.get("spectrum")
    if spec:
        u, n = float(spec["wind_speed"]), int(spec.get("components", 30))
        peak = 0.877 * GRAVITY / u
        edges = np.geomspace(0.6 * peak, 5.0 * peak, n + 1)
        omega, d_omega = np.sqrt(edges[:-1] * edges[1:]), np.diff(edges)
        density = 8.1e-3 * GRAVITY**2 / omega**5 * np.exp(-0.74 * (GRAVITY / (u * omega)) ** 4)
        amp = np.sqrt(2.0 * density * d_omega)
        if "significant_height" in spec:
            amp *= float(spec["significant_height"]) / (4.0 * np.sqrt(np.sum(amp**2) / 2.0))
        th = np.radians(float(spec.get("direction_deg", 0.0)) + float(spec.get("spread_deg", 20.0)) * rng.standard_normal(n))
        k = omega**2 / GRAVITY
        rows += [[a, kk * np.cos(t), kk * np.sin(t), o, ph]
                 for a, kk, t, o, ph in zip(amp, k, th, omega, rng.uniform(0.0, 2.0 * np.pi, n))]
    comp = np.array(rows, dtype=float).reshape(-1, 5)
    if item.get("frozen", False):
        comp[:, 3] = 0.0
    return comp


def wave_surface(components: np.ndarray, depth: float = 0.0):
    """fn(x, y, t) -> (z, normals) for the sum of the wave trains; normals point down into the water."""
    def fn(x, y, t):
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        z, gx, gy = np.full(x.shape, float(depth)), np.zeros(x.shape), np.zeros(x.shape)
        for a, kx, ky, omega, phase in components:
            arg = kx * x + ky * y - omega * t + phase
            z += a * np.sin(arg)
            c = a * np.cos(arg)
            gx += c * kx
            gy += c * ky
        normals = np.column_stack([-gx, -gy, np.ones(x.shape)])
        return z, normals / np.linalg.norm(normals, axis=1, keepdims=True)
    return fn


def _add_box(scene: Scene, lo, hi, item: dict, rng, vel, outward: bool, skip_top: bool = False) -> None:
    """Surface points on the faces of an axis-aligned box, normals outward or inward."""
    for i, (origin, u, v) in enumerate(box_faces(lo, hi)):
        if skip_top and i == 4:                              # the face at minimum z (NED: the top)
            continue
        pts, normal = plate_points(origin, u, v, float(item["density"]), rng)
        scene.add_surface(pts, normal if outward else -normal, float(item.get("mu_db", -20.0)),
                          float(item["density"]), item.get("label", item["type"]), rng, vel)


def build_scene(items: list[dict], rng: np.random.Generator | None = None) -> Scene:
    """Build a scene from the `scene:` list of a scenario config."""
    scene = Scene()
    rng = rng or np.random.default_rng()
    for item in items:
        kind = item["type"]
        ts = item.get("target_strength_db")
        amp = item.get("amplitude", 1.0)
        vel = item.get("velocity", (0.0, 0.0, 0.0))
        if kind == "points":
            scene.add_points(item["positions"], amp, vel, item.get("label", "points"), ts)
        elif kind == "polar_grid":
            pts = polar_grid(item.get("origin", [0, 0, 0]), item["forward"], item["across"],
                             item["ranges"], item["bearings_deg"], item.get("depth_offset", 0.0))
            scene.add_points(pts, amp, vel, item.get("label", "polar_grid"), ts)
        elif kind == "seafloor":
            mu = item.get("lambert_mu_db")
            rms = (lambert_rms_amplitude(float(mu), float(item["density"])) if mu is not None
                   else float(item.get("rms_amplitude", 1.0)))
            pts, a = seafloor_patch(item["x_range"], item["y_range"], float(item["density"]),
                                    float(item.get("depth", 0.0)), rms, rng)
            normals = None
            if item.get("slope") is not None or item.get("ripples") is not None:
                # Relief: move the points onto the surface and give each its normal (up = -z).
                if mu is None:
                    raise ValueError("a seafloor with slope or ripples needs lambert_mu_db")
                height, gradient = relief(item.get("slope", (0.0, 0.0)), item.get("ripples"))
                pts[:, 2] += height(pts[:, 0], pts[:, 1])
                gx, gy = gradient(pts[:, 0], pts[:, 1])
                normals = np.column_stack([gx, gy, -np.ones(len(pts))])
                normals /= np.linalg.norm(normals, axis=1, keepdims=True)
            scene.add_points(pts, a, vel, item.get("label", "seafloor"), lambert=mu is not None, normals=normals)
        elif kind == "wire":
            scene.add_points(wire(item["start"], item["end"], float(item.get("spacing", 0.01))), amp, vel,
                             item.get("label", "wire"), ts)
        elif kind == "plate":
            pts, normal = plate_points(item["origin"], item["edge_u"], item["edge_v"], float(item["density"]), rng)
            scene.add_surface(pts, normal, float(item.get("mu_db", -20.0)), float(item["density"]),
                              item.get("label", "plate"), rng, vel)
        elif kind == "sphere":
            pts, normals = sphere_points(item["centre"], float(item["radius"]), float(item["density"]), rng)
            scene.add_surface(pts, normals, float(item.get("mu_db", -20.0)), float(item["density"]),
                              item.get("label", "sphere"), rng, vel)
            scene.occluders.append(("sphere", list(item["centre"]), float(item["radius"])))
        elif kind == "cylinder":
            pts, normals = cylinder_points(item["start"], item["end"], float(item["radius"]),
                                           float(item["density"]), rng, bool(item.get("caps", True)))
            scene.add_surface(pts, normals, float(item.get("mu_db", -15.0)), float(item["density"]),
                              item.get("label", "cylinder"), rng, vel)
            if item.get("occlude", True):
                scene.occluders.append(("capsule", (list(item["start"]), list(item["end"])), float(item["radius"])))
        elif kind == "hull":
            # A ship hull as a Lambert surface: a floating boat (keel down from the
            # waterline) or a wreck on the seabed.
            n = int(item.get("count", 4000))
            pts, normals, areas = hull_points(item["origin"], float(item["length"]), float(item["beam"]),
                                              float(item["depth"]), float(item.get("heading_deg", 0.0)), n, rng,
                                              bool(item.get("keel_down", True)))
            a = np.sqrt(10.0 ** (float(item.get("mu_db", -10.0)) / 10.0) * areas) * (
                rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2.0)
            scene.add_points(pts, a, vel, item.get("label", "hull"), lambert=True, normals=normals)
            if item.get("occlude", False):
                # A capsule along the keel line stands in for the hull as a shadow caster
                # (static hulls only: occluders do not move with a target).
                if np.any(np.asarray(vel, dtype=float) != 0.0):
                    raise ValueError("hull: occlude is for hulls that do not move")
                length, beam, depth = float(item["length"]), float(item["beam"]), float(item["depth"])
                radius = 0.45 * min(beam, 2.0 * depth)
                h = np.radians(float(item.get("heading_deg", 0.0)))
                along = np.array([np.cos(h), np.sin(h), 0.0])
                sign = 1.0 if item.get("keel_down", True) else -1.0
                mid = np.asarray(item["origin"], dtype=float) + np.array([0.0, 0.0, sign * 0.5 * depth])
                half = max(0.5 * length - radius, 0.0) * 0.85
                scene.occluders.append(("capsule", (list(mid - half * along), list(mid + half * along)), radius))
        elif kind == "cable":
            # A cable or pipe along a polyline of waypoints, as cylinders end to end.
            way = np.asarray(item["waypoints"], dtype=float)
            for p0, p1 in zip(way[:-1], way[1:]):
                pts, normals = cylinder_points(p0, p1, float(item["radius"]), float(item["density"]), rng, caps=False)
                scene.add_surface(pts, normals, float(item.get("mu_db", -12.0)), float(item["density"]),
                                  item.get("label", "cable"), rng, vel)
                if item.get("occlude", True):
                    scene.occluders.append(("capsule", (list(p0), list(p1)), float(item["radius"])))
        elif kind == "sea_surface":
            # A patch of the sea surface with waves, seen from below: facets on a sum of wave
            # trains (`waves`, `spectrum`: see wave_components), each with its normal pointing
            # down into the water, so slopes facing the sonar return more (Lambert about the
            # facet). The facets stay where they are (plus `velocity`, a surface drift) and the
            # wave shape travels through them at each train's own speed.
            pts, a = seafloor_patch(item["x_range"], item["y_range"], float(item["density"]), float(item.get("depth", 0.0)),
                                    lambert_rms_amplitude(float(item.get("mu_db", -30.0)), float(item["density"])), rng)
            components = wave_components(item, rng)
            surface = wave_surface(components, float(item.get("depth", 0.0)))
            pts[:, 2], normals = surface(pts[:, 0], pts[:, 1], 0.0)
            moving = bool(np.any(components[:, 3] != 0.0)) if len(components) else False
            # The small scatterers on the surface lose their phase in milliseconds at these
            # frequencies, so each ping sees a new speckle on the same wave shape, unless
            # `coherent: true` keeps one realisation (useful for testing).
            fresh = not bool(item.get("coherent", False))
            seed = int(rng.integers(0, 2**31))
            if moving:
                scene.dynamic.append((len(scene), len(pts), surface))
            if fresh:
                scene.redrawn.append((len(scene), len(pts),
                                      lambert_rms_amplitude(float(item.get("mu_db", -30.0)), float(item["density"])), seed))
            scene.add_points(pts, a, vel, item.get("label", "sea_surface"), lambert=True, normals=normals)
            if "isotropic_db" in item:
                # Small-scale roughness and bubbles scatter back at any tilt. Without this part a
                # facet tilted away from the sonar returns nothing and the troughs are black; with
                # it they are only darker. Strength in dB per square metre, the same at every angle.
                n = len(pts)
                iso = lambert_rms_amplitude(float(item["isotropic_db"]), float(item["density"]))
                if moving:
                    scene.dynamic.append((len(scene), n, surface))
                if fresh:
                    scene.redrawn.append((len(scene), n, iso, seed + 1))
                scene.add_points(pts.copy(), iso * (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2.0),
                                 vel, item.get("label", "sea_surface"))
        elif kind == "boat":
            # A boat afloat: hull, keel, propeller and rudder at the stern, and optionally the
            # bubble wake it has left astern. `origin` is amidships on the waterline.
            length, beam, draft = float(item["length"]), float(item["beam"]), float(item["draft"])
            heading = float(item.get("heading_deg", 0.0))
            h = np.radians(heading)
            along, across = np.array([np.cos(h), np.sin(h), 0.0]), np.array([-np.sin(h), np.cos(h), 0.0])
            origin = np.asarray(item["origin"], dtype=float)
            label = item.get("label", "boat")
            n = int(item.get("count", 800))
            pts, normals, areas = hull_points(origin, length, beam, draft, heading, n, rng)
            mu = 10.0 ** (float(item.get("mu_db", -8.0)) / 10.0)
            a = np.sqrt(mu * areas) * (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2.0)
            scene.add_points(pts, a, vel, label, lambert=True, normals=normals)
            keel_depth = float(item.get("keel_depth", 0.0))
            if keel_depth > 0.0:
                # A fin keel under the middle third: a thin plate seen from either side.
                for sign in (1.0, -1.0):
                    kp, kn = plate_points(origin + along * (-length / 6) + np.array([0.0, 0.0, draft]),
                                          along * (length / 3), np.array([0.0, 0.0, keel_depth]),
                                          float(item.get("keel_density", 40.0)), rng)
                    scene.add_surface(kp, sign * across, float(item.get("mu_db", -8.0)), float(item.get("keel_density", 40.0)),
                                      label, rng, vel)
            # Propeller and rudder: a compact cluster of strong scatterers under the stern.
            stern = origin - along * (0.45 * length) + np.array([0.0, 0.0, 0.8 * draft])
            k = int(item.get("propeller_points", 12))
            prop = stern + rng.normal(0.0, 0.15, (k, 3))
            ts_prop = float(item.get("propeller_ts_db", -5.0))
            scene.add_points(prop, np.sqrt(10.0 ** (ts_prop / 10.0) / k) * (rng.standard_normal(k) + 1j * rng.standard_normal(k)) / np.sqrt(2.0),
                             vel, label)
            wake = item.get("wake")
            if wake:
                # Bubbles astern: a wedge of volume scatterers that stays where the boat has
                # been (it does not move with the boat) and is densest near the stern.
                m, wl = int(wake.get("count", 600)), float(wake["length"])
                s_along = wl * rng.uniform(0.0, 1.0, m) ** 1.5
                half_w = 0.5 * beam + float(wake.get("spread", 0.06)) * s_along
                wp = (origin - along * (0.5 * length) - along[None, :] * s_along[:, None]
                      + across[None, :] * (rng.uniform(-1.0, 1.0, m) * half_w)[:, None])
                wp[:, 2] = rng.uniform(0.1, float(wake.get("depth", 2.0)), m)
                ts_wake = float(wake.get("strength_db_per_m", -25.0)) + 10.0 * np.log10(wl)
                scene.add_points(wp, np.sqrt(10.0 ** (ts_wake / 10.0) / m) * (rng.standard_normal(m) + 1j * rng.standard_normal(m)) / np.sqrt(2.0),
                                 (0.0, 0.0, 0.0), label + "_wake")
        elif kind == "debris":
            # Seabed clutter: small boxes of random size and position (crates, blocks, junk).
            x, y, depth = item["x_range"], item["y_range"], float(item.get("depth", 0.0))
            lo_s, hi_s = item.get("size_range", [0.3, 1.2])
            for _ in range(int(item["count"])):
                size = rng.uniform(lo_s, hi_s, 3) * np.array([1.0, 1.0, 0.6])
                cx, cy = float(rng.uniform(*x)), float(rng.uniform(*y))
                lo = [cx - size[0] / 2, cy - size[1] / 2, depth - size[2]]
                hi = [cx + size[0] / 2, cy + size[1] / 2, depth]
                _add_box(scene, lo, hi, {**item, "type": "debris", "label": item.get("label", "debris")}, rng, vel,
                         outward=True)
                scene.occluders.append(("box", lo, hi))
        elif kind == "volume":
            # Point scatterers spread through a box (sediment inhomogeneities, bubbles, fish):
            # isotropic, complex Gaussian amplitudes of the given RMS.
            lo, hi = np.asarray(item["min"], dtype=float), np.asarray(item["max"], dtype=float)
            n = int(item["count"])
            pts = rng.uniform(lo, hi, (n, 3))
            a = float(item.get("rms_amplitude", 1.0)) * (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2.0)
            scene.add_points(pts, a, vel, item.get("label", "volume"))
        elif kind == "box":
            _add_box(scene, item["min"], item["max"], item, rng, vel, outward=True)
            scene.occluders.append(("box", list(item["min"]), list(item["max"])))
        elif kind == "pool":
            # Walls and floor facing inwards; the water surface is left to the propagation tier.
            x, y, z = item["x_range"], item["y_range"], item["z_range"]
            _add_box(scene, [x[0], y[0], z[0]], [x[1], y[1], z[1]], item, rng, vel, outward=False,
                     skip_top=True)
        elif kind == "rocks":
            x, y = item["x_range"], item["y_range"]
            r_lo, r_hi = item.get("radius_range", [0.1, 0.3])
            for _ in range(int(item["count"])):
                r = float(rng.uniform(r_lo, r_hi))
                c = [float(rng.uniform(*x)), float(rng.uniform(*y)), float(item.get("depth", 0.0)) - 0.3 * r]
                pts, normals = sphere_points(c, r, float(item["density"]), rng)
                keep = pts[:, 2] < float(item.get("depth", 0.0))           # the part above the floor
                scene.add_surface(pts[keep], normals[keep], float(item.get("mu_db", -15.0)),
                                  float(item["density"]), item.get("label", "rocks"), rng, vel)
                scene.occluders.append(("sphere", c, r))
        else:
            raise ValueError(f"unknown scene item type {kind!r}")
    return scene
