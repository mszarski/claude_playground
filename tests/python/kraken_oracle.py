"""KRAKEN (normal modes, Acoustics Toolbox, GPL-3) as an external test oracle.

Runs `kraken.exe` and `field.exe` as separate processes and reads the shade file they
write. Nothing here is linked into or shipped with the suite: it lives under tests/ and
is skipped unless GPUSONAR_KRAKEN_DIR points at a directory holding the two programs
(`Kraken/kraken.exe` and `KrakenField/field.exe` of an Acoustics Toolbox build, or both
side by side).
"""

from __future__ import annotations

import os
import subprocess
import tempfile

import numpy as np


def find_kraken():
    """(kraken.exe, field.exe) or None."""
    root = os.environ.get("GPUSONAR_KRAKEN_DIR")
    if not root:
        return None
    for a, b in ((os.path.join(root, "Kraken", "kraken.exe"), os.path.join(root, "KrakenField", "field.exe")),
                 (os.path.join(root, "kraken.exe"), os.path.join(root, "field.exe"))):
        if os.path.exists(a) and os.path.exists(b):
            return a, b
    return None


def read_shade(path: str):
    """Pressure [n_rz, n_rr] (complex), receiver depths (m) and ranges (m) of a shade file
    with one source and one frequency."""
    with open(path, "rb") as f:
        raw = f.read()
    recl = 4 * int(np.frombuffer(raw, dtype="<i4", count=1)[0])
    rec = lambda i: raw[i * recl:(i + 1) * recl]                     # noqa: E731
    counts = np.frombuffer(rec(2), dtype="<i4", count=7)             # Nfreq, Ntheta, NSx, NSy, NSz, NRz, NRr
    n_rz, n_rr = int(counts[5]), int(counts[6])
    if counts[0] != 1 or counts[4] != 1:
        raise ValueError("expected one frequency and one source depth")
    rz = np.frombuffer(rec(8), dtype="<f4", count=n_rz).astype(float)
    rr = np.frombuffer(rec(9), dtype="<f8", count=n_rr).astype(float)
    p = np.stack([np.frombuffer(rec(10 + i), dtype="<c8", count=n_rr) for i in range(n_rz)])
    return p, rz, rr


def run_kraken(depth: float, sound_speed: float, bottom_speed: float, bottom_density: float,
               bottom_attenuation: float, frequency: float, source_depth: float, receiver_depths,
               ranges, incoherent: bool = False, timeout: float = 900.0):
    """Transmission loss [n_rz, n_rr] in dB re 1 m for an isovelocity layer over a fluid
    half-space (attenuation in dB per wavelength), trapped modes only."""
    exes = find_kraken()
    if exes is None:
        raise RuntimeError("KRAKEN is not installed (GPUSONAR_KRAKEN_DIR)")
    rd = np.atleast_1d(np.asarray(receiver_depths, dtype=float))
    rr = np.atleast_1d(np.asarray(ranges, dtype=float))
    env = [
        "'gpusonar oracle'", f"{frequency:.6f}", "1", "'CVW'",
        f"0  0.0  {depth:.4f}", f"  0.0  {sound_speed:.4f}  /", f"  {depth:.4f}  {sound_speed:.4f}  /",
        "'A' 0.0", f" {depth:.4f}  {bottom_speed:.4f} 0.0 {bottom_density:.4f} {bottom_attenuation:.4f} /",
        f"{0.9 * sound_speed:.3f}  {bottom_speed:.3f}", "0.0",
        "1", f"{source_depth:.6f} /",
        f"{len(rd)}", " ".join(f"{v:.6f}" for v in rd) + " /",
    ]
    flp = [
        "/,", "'RA I'" if incoherent else "'RA'", "99999", "1", "0.0 /",
        f"{len(rr)}", " ".join(f"{v / 1000.0:.9f}" for v in rr) + " /",
        "1", f"{source_depth:.6f} /",
        f"{len(rd)}", " ".join(f"{v:.6f}" for v in rd) + " /",
        f"{len(rd)}", " ".join("0.0" for _ in rd) + " /",
    ]
    with tempfile.TemporaryDirectory(prefix="gpusonar_kraken_") as tmp:
        with open(os.path.join(tmp, "run.env"), "w", encoding="ascii") as f:
            f.write("\n".join(env) + "\n")
        with open(os.path.join(tmp, "run.flp"), "w", encoding="ascii") as f:
            f.write("\n".join(flp) + "\n")
        for exe in exes:
            proc = subprocess.run([exe, "run"], cwd=tmp, capture_output=True, text=True, timeout=timeout)
            if proc.returncode != 0:
                raise RuntimeError(f"{os.path.basename(exe)} failed: {proc.stderr[-400:]} {proc.stdout[-400:]}")
        p, _, _ = read_shade(os.path.join(tmp, "run.shd"))
    return -20.0 * np.log10(np.maximum(np.abs(p), 1e-30))
