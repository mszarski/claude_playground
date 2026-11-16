"""Multi-GPU backprojection by splitting the image into tiles (ASASIN Sec. III-C)."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .. import _gpusonar as core


def backproject_tiles(records, points: np.ndarray, devices=None, **kw) -> np.ndarray:
    """Backproject a grid [rows, cols, 3] with one tile of rows per CUDA device.

    Each device gets every ping and a contiguous band of image rows; pixels are
    independent, so the result is the same as a single call. Keyword arguments are
    passed to `core.backproject` (culling is applied per tile).
    """
    if devices is None:
        devices = list(range(max(core.cuda_device_count(), 1)))
    rows, cols = points.shape[:2]
    edges = np.linspace(0, rows, len(devices) + 1).astype(int)
    cull = kw.pop("cull_block", 0)

    def run(i):
        r0, r1 = edges[i], edges[i + 1]
        tile = points[r0:r1].reshape(-1, 3)
        extra = dict(cull_block=cull, grid_shape=(r1 - r0, cols)) if cull else {}
        return core.backproject(records, tile, backend=core.Backend.Cuda, device=devices[i],
                                **extra, **kw).reshape(r1 - r0, cols)

    with ThreadPoolExecutor(len(devices)) as pool:
        tiles = list(pool.map(run, range(len(devices))))
    return np.concatenate(tiles, axis=0)
