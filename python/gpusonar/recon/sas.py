"""Synthetic aperture imaging with known navigation: stripmap and rail.

The shared backprojection kernel does the work, over many pings. This module builds
the seafloor grid (ASASIN Sec. II-B, Fig. 3 and 4) and runs it with ray culling.

Units and conventions: metres, world frame north-east-down. z is positive DOWN and the
sea surface is z = 0 unless the scenario puts its origin elsewhere (many scenes put it on
the seabed, with the sonar at negative z). The imaging plane `plane_z` is a z in that
frame. Its default, 0.0, is therefore the seabed only for a scene whose origin is on the
seabed; for a survey whose origin is at the surface it is the surface, and the image is
defocused without any message (docs/HARDCODED_AUDIT.md R-4). `plane_z="seafloor"` takes
the plane from the records instead (`seafloor_z`). No level enters here: images keep the
units of the samples.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .. import _gpusonar as core
from .fls import prepare


@dataclass
class StripmapGrid:
    x: np.ndarray        # along-track axis, m
    y: np.ndarray        # cross-track (ground range) axis, m, signed like the boresight
    z: float             # depth of the imaging plane, m (NED)
    points: np.ndarray   # [nx, ny, 3]

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.x), len(self.y)


def seafloor_z(records, tolerance: float = 0.5) -> tuple[float, str]:
    """(z of the seabed under the collection, where it came from), from the records alone.

    From the DVL: each ping's z plus its altitude, averaged over the pings that have bottom
    lock ("dvl"). Without a DVL altitude: `environment.water_depth`, taken as a z, which is
    right when the surface is at z = 0 ("water_depth"). With neither the seabed is not
    known and a ValueError says so. When both are there and differ by more than
    `tolerance` metres the DVL wins and the source reads "dvl (water_depth differs by ...)":
    the scenario's origin is then not at the surface.
    """
    found = []
    for r in records:
        z = float(r.geometry.rx_pose.position[2])
        for d in r.navigation.dvl[:1]:
            if np.isfinite(d.altitude):
                found.append(z + float(d.altitude))
    depth = float(records[0].environment.water_depth) if len(records) else float("nan")
    if found:
        z_dvl = float(np.mean(found))
        if np.isfinite(depth) and abs(depth - z_dvl) > tolerance:
            return z_dvl, f"dvl (water_depth differs by {depth - z_dvl:+.2f} m)"
        return z_dvl, "dvl"
    if np.isfinite(depth):
        return depth, "water_depth"
    raise ValueError("the records carry neither a DVL altitude nor environment.water_depth: the seabed's z is not "
                     "known. Give plane_z (or seafloor_z) as a number")


def resolve_plane_z(records, plane_z) -> float:
    """A number as given; "seafloor" from the records (`seafloor_z`)."""
    if isinstance(plane_z, str):
        if plane_z != "seafloor":
            raise ValueError("plane_z must be a z in metres or 'seafloor'")
        return seafloor_z(records)[0]
    return float(plane_z)


def stripmap_grid(records, ground_range, pixel, *, plane_z: float | str = 0.0, beamwidth_az=None,
                  full_support: bool = True) -> StripmapGrid:
    """Seafloor grid for a straight-line collection.

    `plane_z` is the z (down) of the imaging plane: a number, or "seafloor" to take it from
    the records' DVL altitude or water depth. The default 0.0 is the seabed only when the
    scenario's origin is on the seabed.

    Along-track extent follows the paper (DECISIONS.md D-11): with `full_support` the
    grid starts `y_max * tan(beamwidth_az / 2)` after the first ping and ends the same
    distance before the last, so even the far-range corners are seen across the whole
    beam and resolution is constant over the image. `ground_range` is (near, far) along
    the direction the sonar looks; `pixel` is (dx, dy) or one number.
    """
    first, last = records[0], records[-1]
    p0 = first.rx_positions().mean(axis=0)
    p1 = last.rx_positions().mean(axis=0)
    track = p1 - p0
    length = float(np.linalg.norm(track[:2]))
    if length <= 0.0:
        raise ValueError("stripmap_grid needs a collection that moves along track")
    along = np.array([track[0], track[1], 0.0]) / length
    # Horizontal direction the array looks in.
    r = core.rotation_ned(*first.geometry.rx_pose.rpy)
    look = r @ np.asarray(first.geometry.rx[0].boresight)
    across = look - along * (look @ along)
    across[2] = 0.0
    across /= np.linalg.norm(across)

    near, far = float(ground_range[0]), float(ground_range[1])
    if beamwidth_az is None:
        beamwidth_az = first.geometry.rx[0].beamwidth_az
    margin = far * np.tan(0.5 * beamwidth_az) if (full_support and np.isfinite(beamwidth_az)) else 0.0
    if 2.0 * margin >= length:
        raise ValueError(
            f"collection is {length:.2f} m long but full angular support at {far:.1f} m needs more "
            f"than {2 * margin:.2f} m; lengthen the track or pass full_support=False")
    plane_z = resolve_plane_z(records, plane_z)
    dx, dy = (pixel, pixel) if np.isscalar(pixel) else pixel
    s = np.arange(margin, length - margin + 0.5 * dx, dx)
    g = np.arange(near, far + 0.5 * dy, dy)
    origin = np.array([p0[0], p0[1], plane_z])
    pts = origin + s[:, None, None] * along + g[None, :, None] * across
    return StripmapGrid(s, g, plane_z, pts)


def sas_image(records, points: np.ndarray, *, interp: str = "sinc", fov_factor: float = 1.13,
              directivity_weight: bool = False, cull_block: int = 16, compress: bool = True,
              replica_window: str = "none", position_lookup: str = "analytic",
              trajectory_allowance=None, stats: dict | None = None, backend=core.Backend.Host) -> np.ndarray:
    """Backproject many pings onto a 2-D grid of points [n0, n1, 3]. Returns [n0, n1] complex.

    `fov_factor` limits each ping to pixels inside that multiple of the FWHM beam. The
    default 1.13 (= 1 / 0.886) processes an angular extent of lambda / D for an element of
    length D, which is the synthetic aperture behind the classic D / 2 resolution. A larger
    factor (up to 2.26, the main lobe null to null) gives finer resolution and lets in
    more of the along-track ambiguities. The default belongs to this function alone: a
    caller that has no value of its own passes none (config-driven chains pass only the
    keys their config gives).

    `trajectory_allowance` matters with a trajectory table (`position_lookup` other than
    "analytic") and a field-of-view limit: it is how far the kernel's pre-test lets the
    platform stray from its first-order pose within a ping before it rejects a pixel. None
    is the kernel's fixed 0.02 m; "table" takes it from each ping's own table (at least
    0.02 m), which is what a long ping or a turning platform needs; a number is metres.
    """
    records = list(records)
    prepare(records, compress=compress, window=replica_window, backend=backend)
    shape = points.shape[:2]
    kw = {}
    if cull_block > 0 and fov_factor > 0:
        kw = dict(cull_block=cull_block, grid_shape=shape)
    if stats is not None:
        kw["stats"] = stats
    if trajectory_allowance is not None:
        kw["trajectory_allowance"] = trajectory_allowance
    img = core.backproject(records, points.reshape(-1, 3), interp=interp, fov_factor=fov_factor,
                           directivity_weight=directivity_weight, position_lookup=position_lookup,
                           backend=backend, **kw)
    return img.reshape(shape)
