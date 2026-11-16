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
        """Scatterer positions at scenario time t (first-order motion)."""
        return self.positions + self.velocities * t


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
