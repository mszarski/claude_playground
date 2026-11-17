"""A seabed that is not flat: resolved large-scale relief (a slope, a bank, a ridge).

One description serves the scattering-function reverberation (sim/reverb_relief.py) and
the point-scatterer seafloor of a scene (sim/scene.py), so that both describe the same
bottom. Relief smaller than a resolution cell is not described here: it stays in the
scattering strength (the roughness spectrum w2, gamma2 of sim/bottom.py, or Lambert's mu).

Frame: world NED, depth z positive down. `depth(x, y)` is the seabed's z; the unit normal
of the bed pointing into the water is (D_x, D_y, -1) / sqrt(1 + D_x^2 + D_y^2), and a
facet over a horizontal area dA has area dA sqrt(1 + D_x^2 + D_y^2).
"""

from __future__ import annotations

import numpy as np


class Bathymetry:
    """Seabed depth over the horizontal plane, with its gradient."""

    def __init__(self, depth, gradient=None, step: float = 0.25, flat_depth: float | None = None):
        self._depth, self._gradient, self.step = depth, gradient, float(step)
        # The depth when the bed is exactly level everywhere (then the flat code applies).
        self.flat_depth = flat_depth

    def depth(self, x, y) -> np.ndarray:
        x, y = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        return np.broadcast_to(np.asarray(self._depth(x, y), dtype=float), x.shape)

    def gradient(self, x, y):
        """(dD/dx, dD/dy): analytic when the description has one, else central differences
        over `step` metres."""
        x, y = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        if self._gradient is not None:
            gx, gy = self._gradient(x, y)
            return np.broadcast_to(np.asarray(gx, dtype=float), x.shape), np.broadcast_to(np.asarray(gy, dtype=float), x.shape)
        h = self.step
        return ((self.depth(x + h, y) - self.depth(x - h, y)) / (2.0 * h),
                (self.depth(x, y + h) - self.depth(x, y - h)) / (2.0 * h))

    def normal(self, x, y) -> np.ndarray:
        """Unit normals of the bed pointing into the water, [..., 3]."""
        gx, gy = self.gradient(x, y)
        n = np.stack([gx, gy, -np.ones_like(gx)], axis=-1)
        return n / np.linalg.norm(n, axis=-1, keepdims=True)

    def area_factor(self, x, y) -> np.ndarray:
        """True facet area over horizontal area: 1 / cos(slope)."""
        gx, gy = self.gradient(x, y)
        return np.sqrt(1.0 + gx * gx + gy * gy)


def plane(depth: float, slope=(0.0, 0.0), point=(0.0, 0.0)) -> Bathymetry:
    """A uniformly sloping bed: `depth` at the horizontal `point`, `slope` = (dD/dx, dD/dy)
    (metres of depth per metre north and east; positive deepens in that direction)."""
    d0, (sx, sy), (px, py) = float(depth), (float(v) for v in slope), (float(v) for v in point)
    return Bathymetry(lambda x, y: d0 + sx * (x - px) + sy * (y - py),
                      lambda x, y: (np.full(np.shape(x), sx), np.full(np.shape(x), sy)),
                      flat_depth=d0 if sx == 0.0 and sy == 0.0 else None)


def _interp_axis(axis, v):
    """Indices and weight for linear interpolation on an increasing axis, held beyond its ends."""
    v = np.clip(v, axis[0], axis[-1])
    hi = np.clip(np.searchsorted(axis, v, side="right"), 1, max(len(axis) - 1, 1))
    lo = hi - 1
    if len(axis) == 1:
        return lo, lo, np.zeros_like(v)
    return lo, hi, (v - axis[lo]) / (axis[hi] - axis[lo])


def grid(x, y, depth, step: float | None = None) -> Bathymetry:
    """Depth on a rectangular grid, `depth[len(x), len(y)]`, interpolated bilinearly and
    held beyond the edges."""
    x, y, table = np.asarray(x, dtype=float), np.asarray(y, dtype=float), np.asarray(depth, dtype=float)
    if table.shape != (len(x), len(y)) or np.any(np.diff(x) <= 0.0) or np.any(np.diff(y) <= 0.0):
        raise ValueError("bathymetry grid: depth must be [len(x), len(y)] on increasing x and y")

    def fn(px, py):
        i0, i1, fx = _interp_axis(x, px)
        j0, j1, fy = _interp_axis(y, py)
        return ((table[i0, j0] * (1 - fx) + table[i1, j0] * fx) * (1 - fy)
                + (table[i0, j1] * (1 - fx) + table[i1, j1] * fx) * fy)

    spacing = min(np.diff(x).min() if len(x) > 1 else 1.0, np.diff(y).min() if len(y) > 1 else 1.0)
    return Bathymetry(fn, step=step or min(0.25 * spacing, 0.25),
                      flat_depth=float(table.flat[0]) if np.all(table == table.flat[0]) else None)


def polar(ranges, depth, azimuth_deg=None, origin=(0.0, 0.0), step: float | None = None) -> Bathymetry:
    """Depth against horizontal range, and optionally azimuth (degrees from x = north
    towards y = east), about a point fixed in the world: `depth[len(ranges)]` or
    `depth[len(azimuth_deg), len(ranges)]`, linear in both, held beyond the last range."""
    ranges, table = np.asarray(ranges, dtype=float), np.asarray(depth, dtype=float)
    if np.any(np.diff(ranges) <= 0.0):
        raise ValueError("bathymetry: range must be increasing")
    if azimuth_deg is None:
        if table.shape != ranges.shape:
            raise ValueError("bathymetry: depth must have one value per range")
        table, az = table[None, :], None
    else:
        az = np.radians(np.asarray(azimuth_deg, dtype=float)) % (2.0 * np.pi)
        if table.shape != (len(az), len(ranges)):
            raise ValueError("bathymetry: depth must have shape [azimuths, ranges]")
        order = np.argsort(az)
        az, table = az[order], table[order]
    ox, oy = float(origin[0]), float(origin[1])

    def fn(px, py):
        dx, dy = px - ox, py - oy
        lo, hi, fr = _interp_axis(ranges, np.hypot(dx, dy))
        if az is None:
            return table[0][lo] * (1 - fr) + table[0][hi] * fr
        a = np.arctan2(dy, dx) % (2.0 * np.pi)
        k1 = np.searchsorted(az, a, side="right") % len(az)
        k0 = (k1 - 1) % len(az)
        span = (az[k1] - az[k0]) % (2.0 * np.pi)
        fa = np.where(span > 0.0, ((a - az[k0]) % (2.0 * np.pi)) / np.where(span > 0.0, span, 1.0), 0.0)
        return ((table[k0, lo] * (1 - fr) + table[k0, hi] * fr) * (1 - fa)
                + (table[k1, lo] * (1 - fr) + table[k1, hi] * fr) * fa)

    spacing = np.diff(ranges).min() if len(ranges) > 1 else 1.0
    return Bathymetry(fn, step=step or min(0.25 * spacing, 0.25),
                      flat_depth=float(table.flat[0]) if np.all(table == table.flat[0]) else None)


def make_bathymetry(spec) -> Bathymetry:
    """A Bathymetry from its description: a Bathymetry, a function depth(x, y), or a mapping

        {plane: {depth, slope: [dD/dx, dD/dy], point: [x, y]}}
        {x: [...], y: [...], depth: [[...]]}                      depth[len(x)][len(y)]
        {range: [...], depth: [...], azimuth_deg: [...], origin: [x, y]}

    with an optional `step` (metres) for the numerical gradient of the two tables."""
    if isinstance(spec, Bathymetry):
        return spec
    if callable(spec):
        return Bathymetry(spec)
    if "plane" in spec:
        p = spec["plane"]
        return plane(p["depth"], p.get("slope", (0.0, 0.0)), p.get("point", (0.0, 0.0)))
    if "x" in spec and "y" in spec:
        return grid(spec["x"], spec["y"], spec["depth"], spec.get("step"))
    if "range" in spec:
        return polar(spec["range"], spec["depth"], spec.get("azimuth_deg"), spec.get("origin", (0.0, 0.0)), spec.get("step"))
    raise ValueError("bathymetry: give `plane`, `x`/`y`/`depth`, or `range`/`depth`")
