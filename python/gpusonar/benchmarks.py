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


# --- The challenge's own evaluation (PICMUS archive `archive_to_download.zip`, opened 2026-11-21) ---------------
# Source: the challenge's MATLAB evaluation code, code/src/@us_contrast/us_contrast.m and @us_resolution/us_resolution.m
# (free of use, cite Liebgott et al., IEEE IUS 2016: the challenge site's Copyright page), ported line by line below;
# the CNR is eq. (1) of the paper (opened, the copy on the challenge site). Geometry from the archive's
# database/simulation/*_scan.hdf5 and *_phantom.hdf5 files, all confirmed by opening them.
#
# Beamforming grid of every PICMUS image: x (lateral) from -19.05 mm in steps of a third of a wavelength, 387 columns;
# z (depth) from 5 mm in steps of a quarter wavelength, 609 rows; wavelength 1540 m/s / 5.208 MHz. (The scan files'
# axes equal this to 1e-17 m.)
PICMUS_WAVELENGTH = 1540.0 / 5.208e6
PICMUS_GRID = {"x0": -19.05e-3, "nx": 387, "dx": PICMUS_WAVELENGTH / 3.0, "z0": 5e-3, "nz": 609, "dz": PICMUS_WAVELENGTH / 4.0}
# Simulated contrast phantom: nine anechoic cysts of 8 mm diameter (lateral mm, depth mm), in the phantom file's order
# (middle column, left, right), and the phantom file's lateral resolution, the padding of the cyst masks.
PICMUS_CYSTS_SIMU_MM = [(0.0, 18.0), (0.0, 30.0), (0.0, 42.0), (-12.0, 18.0), (-12.0, 30.0), (-12.0, 42.0),
                        (12.0, 18.0), (12.0, 30.0), (12.0, 42.0)]
PICMUS_CYST_DIAMETER_SIMU = 8e-3
PICMUS_LATERAL_RESOLUTION_SIMU = 0.00062407256         # m (the file holds float32)
# Simulated resolution phantom: the phantom file's twenty point scatterers (lateral mm, depth mm).
PICMUS_SCATTERERS_SIMU_MM = ([(0.0, d) for d in (10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0)]
                             + [(x, 20.0) for x in (-15.0, -10.0, -5.0, 5.0, 10.0, 15.0)]
                             + [(x, 40.0) for x in (-15.0, -10.0, -5.0, 5.0, 10.0, 15.0)])
# What the challenge's own evaluation code printed for the challenge's reference delay-and-sum (IQ data; receive
# F-number 1.75 with a Tukey 25 % window, das_iq.m): the archive's evaluation/*_simu_evaluation_from_iq.txt, means.
# These are the only published numbers used here; the IUS 2016 paper itself prints no numbers, only images.
PICMUS_REFERENCE_DAS_SIMU = {
    1: {"cnr_db": 9.72, "axial_mm": 0.40, "lateral_mm": 0.82},
    11: {"cnr_db": 12.26, "axial_mm": 0.40, "lateral_mm": 0.54},
    75: {"cnr_db": 15.36, "axial_mm": 0.40, "lateral_mm": 0.56},
}


def picmus_grid() -> tuple[np.ndarray, np.ndarray]:
    """(x, z) axes of the challenge's beamforming grid, metres."""
    g = PICMUS_GRID
    return g["x0"] + np.arange(g["nx"]) * g["dx"], g["z0"] + np.arange(g["nz"]) * g["dz"]


def picmus_cnr(envelope: np.ndarray, cysts=None, diameter: float = PICMUS_CYST_DIAMETER_SIMU,
               lateral_resolution: float = PICMUS_LATERAL_RESOLUTION_SIMU, padding: float = 1.0) -> np.ndarray:
    """CNR (dB) of each cyst, as the challenge computes it (us_contrast.m; paper eq. 1):
    20 log10(|mu_in - mu_out| / sqrt((var_in + var_out) / 2)) over the B-mode image in dB (20 log10 of the envelope
    over its maximum). Inside: the disc of radius r - padding * lateral_resolution; outside: the ring from
    r + padding * lateral_resolution to 1.2 sqrt(r_in^2 + r_out^2). Each value rounded to 0.1 dB as the challenge does.
    `envelope` is [z, x] on `picmus_grid()`; `cysts` (lateral m, depth m), default the simulated phantom's."""
    x, z = picmus_grid()
    env = np.asarray(envelope, dtype=float)
    if env.shape != (len(z), len(x)):
        raise ValueError(f"envelope shape {env.shape}: the PICMUS grid is {(len(z), len(x))} [z, x]")
    bmode = 20.0 * np.log10(env / env.max())
    xx, zz = np.meshgrid(x, z)
    if cysts is None:
        cysts = [(a * 1e-3, b * 1e-3) for a, b in PICMUS_CYSTS_SIMU_MM]
    out = []
    for xc, zc in cysts:
        r = diameter / 2.0
        rin, rout1 = r - padding * lateral_resolution, r + padding * lateral_resolution
        rout2 = 1.2 * np.sqrt(rin**2 + rout1**2)
        d2 = (xx - xc) ** 2 + (zz - zc) ** 2
        inside, outside = bmode[d2 <= rin**2], bmode[(d2 >= rout1**2) & (d2 <= rout2**2)]
        value = 20.0 * np.log10(abs(inside.mean() - outside.mean()) / np.sqrt((inside.var(ddof=1) + outside.var(ddof=1)) / 2.0))
        out.append(round(value * 10.0) / 10.0)
    return np.array(out)


def _challenge_fwhm(axis_mm: np.ndarray, profile_db: np.ndarray) -> float:
    """us_resolution.m Compute_6dB_Resolution: linear interpolation to ten times the samples, then the distance from the
    first to the last sample within 6 dB of the maximum (not only the main lobe's crossings)."""
    xi = np.linspace(axis_mm[0], axis_mm[-1], len(axis_mm) * 10)
    yi = np.interp(xi, axis_mm, profile_db)
    ind = np.nonzero(yi >= yi.max() - 6.0)[0]
    return float(xi[ind.max()] - xi[ind.min()])


def picmus_challenge_resolution(envelope: np.ndarray, points=None, pad: float = 1.8e-3) -> np.ndarray:
    """Axial and lateral -6 dB widths (mm) of each point target, as the challenge measures them (us_resolution.m):
    in a square of +-`pad` about the scatterer's nominal position, the cuts through the brightest pixel of the B-mode
    image (dB), each width by `_challenge_fwhm`. Returns [n_points, 2] (axial, lateral)."""
    x, z = picmus_grid()
    env = np.asarray(envelope, dtype=float)
    if env.shape != (len(z), len(x)):
        raise ValueError(f"envelope shape {env.shape}: the PICMUS grid is {(len(z), len(x))} [z, x]")
    bmode = 20.0 * np.log10(env / env.max())
    xx, zz = np.meshgrid(x, z)
    if points is None:
        points = [(a * 1e-3, b * 1e-3) for a, b in PICMUS_SCATTERERS_SIMU_MM]
    out = []
    for px, pz in points:
        m = (xx > px - pad) & (xx < px + pad) & (zz > pz - pad) & (zz < pz + pad)
        iz, ix = np.nonzero(m)
        jz, jx = np.unravel_index(np.argmax(np.where(m, bmode, bmode.min())), bmode.shape)
        z0, z1, x0, x1 = iz.min(), iz.max() + 1, ix.min(), ix.max() + 1
        out.append((_challenge_fwhm(z[z0:z1] * 1e3, bmode[z0:z1, jx]), _challenge_fwhm(x[x0:x1] * 1e3, bmode[jz, x0:x1])))
    return np.array(out)


def picmus_image(path: str, n_angles: int, f_number: float | None = 1.75, interp: str = "sinc",
                 backend=core.Backend.Host) -> np.ndarray:
    """Envelope [z, x] of a PICMUS data file on the challenge's grid through our kernel: the challenge's plane-wave
    subsets (`picmus_plane_wave_subset`, the same indices as its das_iq.m script), uniform weighting, and for a
    receive F-number each element limited to its acceptance cone (`f_number_fov_factor`): a hard edge where the
    challenge's reference tapers the aperture with a Tukey 25 % window (das_iq.m). None: every element everywhere."""
    src = PicmusSource(path)
    records = [src.read(i) for i in picmus_plane_wave_subset(n_angles, len(src))]
    src.close()
    fov = _limit_receive_aperture(records, f_number) if f_number else 0.0
    x, z = picmus_grid()
    zz, xx = np.meshgrid(z, x, indexing="ij")
    pix = np.column_stack([zz.ravel(), xx.ravel(), np.zeros(zz.size)])      # depth forward, lateral across
    return np.abs(core.backproject(records, pix, interp=interp, fov_factor=fov, backend=backend)).reshape(zz.shape)


def picmus_contrast(path: str, n_angles: int, f_number: float | None = 1.75, backend=core.Backend.Host) -> dict:
    """CNR (dB) of the simulated contrast phantom's nine cysts through our kernel and the challenge's metric."""
    cnr = picmus_cnr(picmus_image(path, n_angles, f_number, backend=backend))
    return {"n_angles": n_angles, "f_number": f_number, "cnr_db": cnr, "cnr_mean_db": float(cnr.mean())}


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


def _limit_receive_aperture(records, f_number: float) -> float:
    """Set the records up for a receive F-number and return the kernel's fov_factor. Only the azimuth acceptance is
    limited; the targets are in the imaging plane."""
    fov = f_number_fov_factor(records[0], f_number)
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
    return fov


def picmus_challenge_widths(path: str, n_angles: int, f_number: float | None = 1.75, backend=core.Backend.Host) -> dict:
    """The simulated resolution phantom's twenty targets through our kernel and the challenge's own width measure
    (`picmus_challenge_resolution`), means in mm as the challenge prints them."""
    w = picmus_challenge_resolution(picmus_image(path, n_angles, f_number, backend=backend))
    return {"n_angles": n_angles, "f_number": f_number, "widths_mm": w, "axial_mm": float(w[:, 0].mean()),
            "lateral_mm": float(w[:, 1].mean())}


def picmus_resolution(path: str, n_angles: int, interp: str = "sinc",
                      f_number: float | None = None, backend=core.Backend.Host) -> dict:
    """Axial and lateral -6 dB widths (FWHM of the envelope) of every point target.

    Returns per-point widths in metres and their means, the PICMUS resolution metric.
    """
    src = PicmusSource(path)
    records = [src.read(i) for i in picmus_plane_wave_subset(n_angles, len(src))]
    wavelength = src.sound_speed / records[0].samples.center_frequency
    fov = _limit_receive_aperture(records, f_number) if f_number else 0.0
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
