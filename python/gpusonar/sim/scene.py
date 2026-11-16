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


def build_scene(items: list[dict]) -> Scene:
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
        else:
            raise ValueError(f"unknown scene item type {kind!r}")
    return scene
