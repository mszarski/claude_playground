"""Scenes: collections of point scatterers with amplitude and velocity.

Every scene object reduces to point scatterers for synthesis. Extended objects
(seafloor facets, spheres, walls) are added as generators that emit points.
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

    def __len__(self) -> int:
        return self.positions.shape[0]

    def add_points(self, positions, amplitude=1.0, velocity=(0.0, 0.0, 0.0), label="points",
                   target_strength_db=None) -> "Scene":
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
        self.labels.append(label)
        return self

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


def build_scene(items: list[dict], rng: np.random.Generator | None = None) -> Scene:
    """Build a scene from the `scene:` list of a scenario config."""
    scene = Scene()
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
            pts, a = seafloor_patch(item["x_range"], item["y_range"], float(item["density"]),
                                    float(item.get("depth", 0.0)), float(item.get("rms_amplitude", 1.0)), rng)
            scene.add_points(pts, a, vel, item.get("label", "seafloor"))
        else:
            raise ValueError(f"unknown scene item type {kind!r}")
    return scene
