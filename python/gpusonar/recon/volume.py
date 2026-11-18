"""3D near-field imaging (ASASIN Sec. III-B): a volume built as 2D layers over focus depth.

Each layer is one call of the shared backprojection kernel on a horizontal grid at one
depth, optionally with refraction at the sediment-water interface. Slices and maximum
intensity projections are the 2D views of the volume.

Units and conventions: metres, world frame north-east-down; `depths` and `interface_z` are
z values (positive down) in the scenario's frame, not depths below the sea surface unless
the frame's origin is at the surface. The sediment's sound speed has no default.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .. import _gpusonar as core
from .fls import prepare


@dataclass
class Volume:
    data: np.ndarray     # [n_depth, n_x, n_y] complex
    depths: np.ndarray   # z of each layer, m (NED, down)
    x: np.ndarray
    y: np.ndarray

    def depth_slice(self, depth: float) -> np.ndarray:
        """The layer nearest `depth`, magnitude."""
        return np.abs(self.data[int(np.argmin(np.abs(self.depths - depth)))])

    def mip(self, axis: str = "depth") -> np.ndarray:
        """Maximum intensity projection: collapse one axis keeping the brightest voxel."""
        index = {"depth": 0, "x": 1, "y": 2}[axis]
        return np.abs(self.data).max(axis=index)

    def peak(self) -> tuple[float, float, float]:
        """(x, y, depth) of the brightest voxel."""
        iz, ix, iy = np.unravel_index(np.argmax(np.abs(self.data)), self.data.shape)
        return float(self.x[ix]), float(self.y[iy]), float(self.depths[iz])


def image_volume(records, x, y, depths, *, interface_z: float | None = None,
                 sediment_speed: float | None = None, interp: str = "linear",
                 fov_factor: float = 0.0, cull_block: int = 0, compress: bool = True,
                 backend=core.Backend.Host) -> Volume:
    """Backproject onto horizontal layers at each of `depths`.

    With `interface_z` and `sediment_speed`, travel times below the interface follow the
    refracted (Snell) path. Without them the medium is treated as water throughout. The two
    go together and neither has a default (the sediment is a fact of the site): giving one
    without the other is an error, not a silent water-only image.
    Ray culling (per layer, transmitter and receiver tested separately) is on when
    `cull_block` > 0 and a field of view is set.
    """
    records = list(records)
    prepare(records, compress=compress)
    x, y, depths = (np.asarray(v, dtype=float) for v in (x, y, depths))
    X, Y = np.meshgrid(x, y, indexing="ij")
    refraction = None
    if (interface_z is None) != (sediment_speed is None):
        raise ValueError("image_volume: interface_z and sediment_speed go together (refraction needs both)")
    if interface_z is not None and sediment_speed is not None:
        refraction = (float(interface_z), float(sediment_speed))
    kw = {}
    if cull_block > 0 and fov_factor > 0:
        kw = dict(cull_block=cull_block, grid_shape=X.shape)
    # One session for all layers: on a GPU the samples are uploaded once.
    from .multigpu import make_session
    session = make_session(records, kw)
    layers = []
    for z in depths:
        pts = np.column_stack([X.ravel(), Y.ravel(), np.full(X.size, z)])
        img = session.backproject(pts, interp=interp, fov_factor=fov_factor, stop_and_hop=True,
                               refraction=refraction, backend=backend, **kw)
        layers.append(img.reshape(X.shape))
    return Volume(np.stack(layers), depths, x, y)
