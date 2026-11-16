"""Benchmark evaluations against public data."""

from __future__ import annotations

import numpy as np

from . import _gpusonar as core
from .metrics import width_at_level
from .sources import PicmusSource

# Nominal point-target layout of the PICMUS simulated resolution phantom, found by
# locating the image maxima (the phantom description itself is not in the data file):
# a column at x = 0 and a row at a depth near 20 mm. (depth mm, lateral mm)
PICMUS_POINTS_MM = [(10.0, 0.0), (15.0, 0.0), (20.0, 0.0), (25.0, 0.0), (30.0, 0.0),
                    (20.0, -15.0), (20.0, -10.0), (20.0, -5.0), (20.0, 5.0), (20.0, 10.0),
                    (20.0, 15.0)]


def picmus_plane_wave_subset(n_angles: int, n_total: int = 75) -> list[int]:
    """Indices of `n_angles` plane waves spread evenly over the full set, centred on 0 degrees."""
    if n_angles == 1:
        return [n_total // 2]
    return [int(round(i)) for i in np.linspace(0, n_total - 1, n_angles)]


def f_number_fov_factor(record, f_number: float) -> float:
    """fov_factor that limits each element to the acceptance cone of a receive F-number
    (aperture = depth / F-number, i.e. half-angle atan(1 / (2 F)))."""
    half_angle = np.arctan(0.5 / f_number)
    return float(2.0 * half_angle / record.geometry.rx[0].beamwidth_az)


def picmus_resolution(path: str, n_angles: int, interp: str = "sinc",
                      f_number: float | None = None, backend=core.Backend.Host) -> dict:
    """Axial and lateral -6 dB widths (FWHM of the envelope) of every point target.

    Returns per-point widths in metres and their means, the PICMUS resolution metric.
    """
    src = PicmusSource(path)
    records = [src.read(i) for i in picmus_plane_wave_subset(n_angles, len(src))]
    wavelength = src.sound_speed / records[0].samples.center_frequency
    fov = f_number_fov_factor(records[0], f_number) if f_number else 0.0
    if fov:
        # Only the azimuth acceptance is limited; the targets are in the imaging plane.
        for r in records:
            geo = r.geometry
            elements = geo.rx
            for e in elements:
                e.beamwidth_el = float("nan")
            geo.rx = elements
            tx = geo.tx
            tx.beamwidth_az = tx.beamwidth_el = float("nan")
            geo.tx = tx
            r.geometry = geo
    half = 1.5e-3
    step = wavelength / 16
    offsets = np.arange(-half, half + step / 2, step)
    axial, lateral, found = [], [], []
    for depth_mm, lat_mm in PICMUS_POINTS_MM:
        # Coarse search for the peak near the nominal position, then cuts through it.
        zz, xx = np.meshgrid(depth_mm * 1e-3 + offsets[::4], lat_mm * 1e-3 + offsets[::4], indexing="ij")
        pts = np.column_stack([zz.ravel(), xx.ravel(), np.zeros(zz.size)])
        coarse = np.abs(core.backproject(records, pts, interp=interp, fov_factor=fov, backend=backend)).reshape(zz.shape)
        iz, ix = np.unravel_index(np.argmax(coarse), coarse.shape)
        z0, x0 = zz[iz, ix], xx[iz, ix]
        line_z = np.column_stack([z0 + offsets, np.full_like(offsets, x0), np.zeros_like(offsets)])
        line_x = np.column_stack([np.full_like(offsets, z0), x0 + offsets, np.zeros_like(offsets)])
        az = np.abs(core.backproject(records, line_z, interp=interp, fov_factor=fov, backend=backend))
        ax = np.abs(core.backproject(records, line_x, interp=interp, fov_factor=fov, backend=backend))
        axial.append(width_at_level(offsets, az, -6.0))
        lateral.append(width_at_level(offsets, ax, -6.0))
        found.append((float(z0), float(x0)))
    src.close()
    return {"n_angles": n_angles, "axial": np.array(axial), "lateral": np.array(lateral),
            "axial_mean": float(np.nanmean(axial)), "lateral_mean": float(np.nanmean(lateral)),
            "positions": found, "wavelength": wavelength}
