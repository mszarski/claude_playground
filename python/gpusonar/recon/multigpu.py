"""Multi-GPU backprojection by splitting the image into tiles (ASASIN Sec. III-C)."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .. import _gpusonar as core


def make_session(records, kw: dict):
    """A `core.BackprojectSession` for `records`, taking the session-level keyword
    arguments (`weights`, `trajectory_upsample`, `trajectory_allowance`) out of `kw`."""
    upsample = kw.pop("trajectory_upsample", 8)
    if kw.get("position_lookup", "analytic") == "analytic":
        upsample = 0
    allowance = kw.pop("trajectory_allowance", None)
    extra = {} if allowance is None else {"trajectory_allowance": allowance}
    return core.BackprojectSession(records, weights=kw.pop("weights", None), trajectory_upsample=upsample, **extra)


def backproject_tiles(records, points: np.ndarray, devices=None, backend=core.Backend.Cuda, **kw) -> np.ndarray:
    """Backproject a grid [rows, cols, 3] with one tile of rows per CUDA device.

    Each device gets every ping and a contiguous band of image rows; pixels are
    independent, so the result is the same as a single call (tests/python/test_multigpu.py
    holds it to `sas_image` on the host). Keyword arguments are passed to `core.backproject`
    (culling is applied per tile). The pings are prepared once and uploaded once per device;
    pass `session=` to reuse a session across calls.

    `backend` is CUDA by default. With `core.Backend.Host` the tiles run on the host, one
    thread each, and `devices` (default `[0]`) only sets how many tiles there are: the host
    path takes no device index.
    """
    if devices is None:
        devices = list(range(max(core.cuda_device_count(), 1))) if backend == core.Backend.Cuda else [0]
    devices = list(devices)
    if not devices:
        raise ValueError("backproject_tiles: devices is empty")
    rows, cols = points.shape[:2]
    edges = np.linspace(0, rows, len(devices) + 1).astype(int)
    cull = kw.pop("cull_block", 0)
    session = kw.pop("session", None) or make_session(records, kw)

    def run(i):
        r0, r1 = edges[i], edges[i + 1]
        tile = points[r0:r1].reshape(-1, 3)
        extra = dict(cull_block=cull, grid_shape=(r1 - r0, cols)) if cull else {}
        return session.backproject(tile, backend=backend, device=devices[i],
                                   **extra, **kw).reshape(r1 - r0, cols)

    with ThreadPoolExecutor(len(devices)) as pool:
        tiles = list(pool.map(run, range(len(devices))))
    return np.concatenate(tiles, axis=0)
