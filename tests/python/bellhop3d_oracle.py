"""BELLHOP3D (Acoustics Toolbox, M. Porter; GPL-3) as an external test oracle for 3-D bathymetry.

Runs `bellhop3d.exe` as a separate process on an environment file and a 3-D bathymetry file written here, and
reads back the shade file. Nothing of it is linked into or shipped with the suite: it lives under tests/ and is
skipped unless the program is found: GPUSONAR_BELLHOP3D, else `bellhop3d.exe` beside GPUSONAR_BELLHOP (an
Acoustics Toolbox build puts both in Bellhop/; tools/build_oracles.sh builds it), else on PATH.

File formats, from the Acoustics Toolbox at HEAD 8b4682b (opened): the run-type letters and the beam fan from
doc/bellhop3d.htm (RunType(6:6) '2' = N x 2-D, '3' = 3-D; bearings in degrees, "math convention with 0 degrees
pointing along the x-axis"); the bathymetry file from Bellhop/bdry3DMod.f90, ReadBTY3D (a type letter, Nx, the
x values in km, Ny, the y values in km, then Ny rows of Nx depths in m); the shade file records as
tests/python/kraken_oracle.py reads them, with one block of receiver depths per bearing (record 10 +
i_theta * NSz * NRz + i_z for the first frequency and source).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile

import numpy as np


def find_bellhop3d() -> str | None:
    path = os.environ.get("GPUSONAR_BELLHOP3D")
    if path and os.path.exists(path):
        return path
    beside = os.path.join(os.path.dirname(os.environ.get("GPUSONAR_BELLHOP", "") or "."), "bellhop3d.exe")
    if os.environ.get("GPUSONAR_BELLHOP") and os.path.exists(beside):
        return beside
    return shutil.which("bellhop3d.exe")


def read_shade_bearings(path: str):
    """Pressure [n_theta, n_rz, n_rr] (complex), bearings (deg), receiver depths (m), ranges (m)."""
    with open(path, "rb") as f:
        raw = f.read()
    recl = 4 * int(np.frombuffer(raw, dtype="<i4", count=1)[0])
    rec = lambda i: raw[i * recl:(i + 1) * recl]                     # noqa: E731
    n_freq, n_theta, _, _, n_sz, n_rz, n_rr = (int(v) for v in np.frombuffer(rec(2), dtype="<i4", count=7))
    if n_freq != 1 or n_sz != 1:
        raise ValueError("expected one frequency and one source depth")
    theta = np.frombuffer(rec(4), dtype="<f8", count=n_theta).astype(float)
    rz = np.frombuffer(rec(8), dtype="<f4", count=n_rz).astype(float)
    rr = np.frombuffer(rec(9), dtype="<f8", count=n_rr).astype(float)
    p = np.stack([np.stack([np.frombuffer(rec(10 + it * n_rz + iz), dtype="<c8", count=n_rr) for iz in range(n_rz)])
                  for it in range(n_theta)])
    return p, theta, rz, rr


def run_bellhop3d(depth_fn, x, y, water_depth, sound_speed, bottom, frequency, source_depth, receiver_depths,
                  bearings_deg, ranges, mode, n_alpha, alpha_deg, n_beta, beta_deg, box_km, beam="B",
                  timeout=1800.0):
    """Incoherent transmission loss [n_theta, n_rz, n_rr] (dB re 1 m) of a point source at (0, 0, source_depth)
    over the bed `depth_fn(x, y)` sampled on the grid x, y (m), isovelocity water over a fluid half-space
    `bottom` = (speed, density, attenuation dB/wavelength). `mode` '2' (N x 2-D) or '3' (3-D)."""
    exe = find_bellhop3d()
    if exe is None:
        raise RuntimeError("Bellhop3D is not installed (GPUSONAR_BELLHOP3D)")
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    bearings, ranges = np.asarray(bearings_deg, dtype=float), np.asarray(ranges, dtype=float)
    rz = np.atleast_1d(np.asarray(receiver_depths, dtype=float))
    for v in (bearings, ranges):
        if len(v) > 1 and not np.allclose(np.diff(v), v[1] - v[0]):
            raise ValueError("bearings and ranges must be equally spaced (they are written as first, last /)")
    with tempfile.TemporaryDirectory(prefix="gpusonar_bellhop3d_") as tmp:
        table = depth_fn(x[None, :], y[:, None])                         # [ny, nx]: one row per y
        with open(os.path.join(tmp, "run.bty"), "w", encoding="ascii") as f:
            f.write("'R'\n" + f"{len(x)}\n" + " ".join(f"{v / 1000.0:.6f}" for v in x) + "\n"
                    + f"{len(y)}\n" + " ".join(f"{v / 1000.0:.6f}" for v in y) + "\n"
                    + "".join(" ".join(f"{d:.4f}" for d in row) + "\n" for row in table))
        speed, density, attenuation = bottom
        lines = ["'gpusonar 3-D oracle'", f"{frequency:.6f}", "1", "'CVW'", f"2  0.0  {water_depth:.4f}",
                 f"  0.0  {sound_speed:.4f}  /", f"  {water_depth:.4f}  {sound_speed:.4f}  /",
                 "'A~' 0.0", f" {water_depth:.4f}  {speed:.4f} 0.0 {density:.4f} {attenuation:.4f} /",
                 "1", "0.0 /", "1", "0.0 /", "1", f"{source_depth:.4f} /",
                 f"{len(rz)}", " ".join(f"{v:.4f}" for v in rz) + " /",
                 f"{len(ranges)}", f"{ranges[0] / 1000.0:.6f} {ranges[-1] / 1000.0:.6f} /",
                 f"{len(bearings)}", f"{bearings[0]:.6f} {bearings[-1]:.6f} /",
                 f"'I{beam}   {mode}'", f"{n_alpha}", f"{alpha_deg[0]:.3f} {alpha_deg[1]:.3f} /",
                 f"{n_beta}", f"{beta_deg[0]:.3f} {beta_deg[1]:.3f} /",
                 f"0.0  {box_km[0]:.4f} {box_km[1]:.4f} {1.05 * water_depth:.4f}"]
        with open(os.path.join(tmp, "run.env"), "w", encoding="ascii") as f:
            f.write("\n".join(lines) + "\n")
        proc = subprocess.run([exe, "run"], cwd=tmp, capture_output=True, text=True, timeout=timeout)
        shd = os.path.join(tmp, "run.shd")
        if proc.returncode != 0 or not os.path.exists(shd):
            prt = os.path.join(tmp, "run.prt")
            log = open(prt, errors="replace").read()[-800:] if os.path.exists(prt) else ""
            raise RuntimeError(f"Bellhop3D failed (exit {proc.returncode}): {proc.stderr[-300:]}\n{log}")
        p, theta, _, rr = read_shade_bearings(shd)
    if not (np.allclose(theta, bearings) and np.allclose(rr, ranges)):
        raise RuntimeError("Bellhop3D returned another receiver grid than the one asked for")
    return -20.0 * np.log10(np.maximum(np.abs(p), 1e-30))
