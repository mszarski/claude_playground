"""Bellhop as an external eigenray provider (the `eigenray` fidelity tier).

Bellhop (Acoustics Toolbox, M. Porter; GPL-3) is run as a separate process: an
environment file is written, `bellhop.exe` is executed, and its arrivals file is read
back. Nothing of Bellhop is linked or copied into this package. The executable is found
through the GPUSONAR_BELLHOP environment variable or as `bellhop.exe` on PATH.

A call per scatterer would be far too slow, so arrivals are computed once per source
depth on a range-depth grid and looked up; within a grid cell the delay is corrected to
first order with the arrival angle (a plane wave at the field point).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field

import numpy as np

from .. import acoustics
from .propagation import EigenrayProvider, EigenraySet


class BellhopError(RuntimeError):
    pass


def find_bellhop() -> str | None:
    """Path of the Bellhop executable, or None when it is not installed."""
    path = os.environ.get("GPUSONAR_BELLHOP")
    if path and os.path.exists(path):
        return path
    return shutil.which("bellhop.exe") or shutil.which("bellhop")


@dataclass
class OceanEnvironment:
    """An ocean for Bellhop: a sound-speed profile over a flat bottom, or over a bottom
    whose depth changes with range from the source (`bathymetry`).

    With bathymetry the model is 2-D: the same depth profile is used on every bearing
    from every source, so it describes a ridge or slope seen along one line, not a
    3-D seamount.
    """

    depth: float                                   # water depth, m (the deepest point with bathymetry)
    ssp_depth: np.ndarray = None                   # m, from 0 to `depth`
    ssp_speed: np.ndarray = None                   # m/s
    bottom_speed: float = 1600.0                   # m/s, fluid half-space
    bottom_density: float = 1.8                    # g/cm^3
    bottom_attenuation: float = 0.5                # dB per wavelength
    surface: str = "vacuum"                        # pressure release
    sound_speed: float = 1500.0                    # used when no profile is given
    bathymetry: list = None                        # [[range_m, depth_m], ...] from the source, or None

    def depth_at(self, range_m):
        """Water depth at a horizontal range from the source."""
        if self.bathymetry is None:
            return np.full_like(np.asarray(range_m, dtype=float), self.depth)
        b = np.asarray(self.bathymetry, dtype=float)
        return np.interp(range_m, b[:, 0], b[:, 1])

    def profile(self):
        if self.ssp_depth is None:
            return np.array([0.0, self.depth]), np.array([self.sound_speed, self.sound_speed])
        z, c = np.asarray(self.ssp_depth, dtype=float), np.asarray(self.ssp_speed, dtype=float)
        if z[0] > 0.0:
            z, c = np.r_[0.0, z], np.r_[c[0], c]
        if z[-1] < self.depth:
            z, c = np.r_[z, self.depth], np.r_[c, c[-1]]
        return z, c

    def speed_at(self, depth: float) -> float:
        z, c = self.profile()
        return float(np.interp(depth, z, c))


def munk_profile(depth: float = 5000.0, n: int = 51, axis_depth: float = 1300.0,
                 axis_speed: float = 1500.0, epsilon: float = 0.00737):
    """The canonical Munk deep-water sound-speed profile."""
    z = np.linspace(0.0, depth, n)
    zt = 2.0 * (z - axis_depth) / 1300.0
    return z, axis_speed * (1.0 + epsilon * (zt - 1.0 + np.exp(-zt)))


def write_env(path: str, env: OceanEnvironment, frequency: float, source_depth: float,
              receiver_depths, receiver_ranges, run_type: str = "A", n_beams: int = 0,
              angles=(-80.0, 80.0), title: str = "gpusonar") -> None:
    z, c = env.profile()
    lines = [f"'{title}'", f"{frequency:.6f}", "1", "'CVW'", f"{len(z)}  0.0  {env.depth:.4f}"]
    lines += [f"  {zi:.4f}  {ci:.4f}  /" for zi, ci in zip(z, c)]
    if env.bathymetry is not None:
        b = np.asarray(env.bathymetry, dtype=float)
        if b.ndim != 2 or b.shape[1] != 2 or np.any(np.diff(b[:, 0]) <= 0) or b[:, 1].max() > env.depth + 1e-6:
            raise BellhopError("bathymetry must be [[range_m, depth_m], ...] with increasing range "
                               "and no depth below `depth`")
        r_end = 1.2 * max(float(np.max(receiver_ranges)), b[-1, 0]) + 1000.0
        rows = [(-1000.0, b[0, 1])] + [tuple(v) for v in b if v[0] > -1000.0] + [(r_end, b[-1, 1])]
        with open(os.path.splitext(path)[0] + ".bty", "w", encoding="ascii") as f:
            f.write("'L'\n" + f"{len(rows)}\n" + "".join(f"{r / 1000.0:.6f} {d:.4f}\n" for r, d in rows))
    lines += ["'A~' 0.0" if env.bathymetry is not None else "'A' 0.0",
              f" {env.depth:.4f}  {env.bottom_speed:.4f} 0.0 {env.bottom_density:.4f} {env.bottom_attenuation:.4f} /",
              "1", f"{source_depth:.6f} /"]
    rd = np.atleast_1d(np.asarray(receiver_depths, dtype=float))
    rr = np.atleast_1d(np.asarray(receiver_ranges, dtype=float))
    lines += [f"{len(rd)}", " ".join(f"{v:.6f}" for v in rd) + " /",
              f"{len(rr)}", " ".join(f"{v / 1000.0:.9f}" for v in rr) + " /",
              f"'{run_type}'", f"{n_beams}", f"{angles[0]:.3f} {angles[1]:.3f} /",
              f"0.0  {1.05 * env.depth:.3f}  {1.05 * rr.max() / 1000.0 + 0.1:.6f}"]
    with open(path, "w", encoding="ascii") as f:
        f.write("\n".join(lines) + "\n")


@dataclass
class Arrivals:
    """Arrivals on a receiver grid: lists indexed [i_depth][i_range] of arrays."""

    frequency: float
    depths: np.ndarray
    ranges: np.ndarray
    amplitude: list = field(default_factory=list)   # complex (phase folded in)
    delay: list = field(default_factory=list)       # s
    launch_angle: list = field(default_factory=list)   # rad, positive towards greater depth
    arrival_angle: list = field(default_factory=list)
    n_surface: list = field(default_factory=list)
    n_bottom: list = field(default_factory=list)

    def pressure(self, i_depth: int, i_range: int, frequency: float | None = None) -> complex:
        """Coherent sum of the arrivals: complex pressure for a unit source (1 m reference)."""
        f = self.frequency if frequency is None else frequency
        a, t = self.amplitude[i_depth][i_range], self.delay[i_depth][i_range]
        return complex(np.sum(a * np.exp(-2j * np.pi * f * t)))

    def transmission_loss(self, coherent: bool = True) -> np.ndarray:
        """[n_depth, n_range] dB re 1 m."""
        out = np.zeros((len(self.depths), len(self.ranges)))
        for i in range(len(self.depths)):
            for j in range(len(self.ranges)):
                if coherent:
                    p = abs(self.pressure(i, j))
                else:
                    p = np.sqrt(np.sum(np.abs(self.amplitude[i][j]) ** 2))
                out[i, j] = -20.0 * np.log10(max(p, 1e-30))
        return out


def read_arrivals(path: str) -> Arrivals:
    """Parse a Bellhop ASCII arrivals file (2-D, one source depth)."""
    with open(path, "r", encoding="ascii") as f:
        tokens = f.read().replace("'", " ").split()
    it = iter(tokens)
    if next(it) != "2D":
        raise BellhopError("only 2-D arrivals files are supported")
    freq = float(next(it))
    n_sz = int(next(it))
    for _ in range(n_sz):
        next(it)
    n_rz = int(next(it))
    depths = np.array([float(next(it)) for _ in range(n_rz)])
    n_rr = int(next(it))
    ranges = np.array([float(next(it)) for _ in range(n_rr)])
    arr = Arrivals(freq, depths, ranges)
    next(it)                                   # largest arrival count
    for _ in range(n_rz):
        rows = {k: [] for k in ("amp", "delay", "launch", "arrive", "ns", "nb")}
        for _ in range(n_rr):
            n = int(next(it))
            vals = np.array([float(next(it)) for _ in range(8 * n)]).reshape(n, 8) if n else np.zeros((0, 8))
            # Bellhop writes phases for the exp(-i omega t) time convention. This suite uses
            # exp(+i omega t) (an echo of delay tau carries exp(-j 2 pi f tau)), so conjugate.
            rows["amp"].append(vals[:, 0] * np.exp(-1j * np.radians(vals[:, 1])))
            rows["delay"].append(vals[:, 2])
            rows["launch"].append(np.radians(vals[:, 4]))
            rows["arrive"].append(np.radians(vals[:, 5]))
            rows["ns"].append(vals[:, 6].astype(int))
            rows["nb"].append(vals[:, 7].astype(int))
        arr.amplitude.append(rows["amp"])
        arr.delay.append(rows["delay"])
        arr.launch_angle.append(rows["launch"])
        arr.arrival_angle.append(rows["arrive"])
        arr.n_surface.append(rows["ns"])
        arr.n_bottom.append(rows["nb"])
    return arr


def run_bellhop(env: OceanEnvironment, frequency: float, source_depth: float, receiver_depths,
                receiver_ranges, n_beams: int = 0, angles=(-80.0, 80.0),
                executable: str | None = None, timeout: float = 600.0) -> Arrivals:
    """Run Bellhop for arrivals on a receiver grid and return them."""
    exe = executable or find_bellhop()
    if exe is None:
        raise BellhopError("Bellhop is not installed: set GPUSONAR_BELLHOP to bellhop.exe")
    with tempfile.TemporaryDirectory(prefix="gpusonar_bellhop_") as tmp:
        write_env(os.path.join(tmp, "run.env"), env, frequency, source_depth, receiver_depths,
                  receiver_ranges, "A", n_beams, angles)
        proc = subprocess.run([exe, "run"], cwd=tmp, capture_output=True, text=True, timeout=timeout)
        arr_path = os.path.join(tmp, "run.arr")
        if proc.returncode != 0 or not os.path.exists(arr_path) or os.path.getsize(arr_path) == 0:
            log = ""
            prt = os.path.join(tmp, "run.prt")
            if os.path.exists(prt):
                with open(prt, "r", errors="replace") as f:
                    log = f.read()[-800:]
            raise BellhopError(f"Bellhop failed (exit {proc.returncode}): {proc.stderr[-300:]}\n{log}")
        return read_arrivals(arr_path)


class BellhopProvider(EigenrayProvider):
    """Eigenrays from cached Bellhop arrivals.

    Geometry is reduced to (horizontal range, depth): the environment is range independent
    and the same in every direction. Depth is the NED z coordinate measured from the sea
    surface (z = 0 is the surface).
    """

    tier = "eigenray"

    def __init__(self, env: OceanEnvironment, frequency: float, max_range: float,
                 range_step: float | None = None, depth_step: float | None = None,
                 max_arrivals: int = 8, n_beams: int = 0, angles=(-80.0, 80.0),
                 min_range: float = 1.0, wind_speed: float = 0.0):
        self.env = env
        # Wind speed (m/s) for the rough-surface reflection loss, applied per surface bounce
        # to Bellhop's flat-surface arrivals (acoustics.surface_coherent_reflection). 0: flat.
        self.wind_speed = float(wind_speed)
        self.frequency = float(frequency)
        self.max_arrivals = int(max_arrivals)
        self.n_beams = n_beams
        self.angles = angles
        self.ranges = np.arange(min_range, max_range + (range_step or max_range / 200.0),
                                range_step or max_range / 200.0)
        dz = depth_step or env.depth / 50.0
        self.depths = np.arange(0.5 * dz, env.depth, dz)
        self._cache: dict[float, Arrivals] = {}
        self.sound_speed = env.speed_at(0.0)

    def along_profile(self, profile) -> "BellhopProvider":
        """This provider for one bearing over relief: the same ocean with the depth profile
        `profile` = [[range_m, depth_m], ...] from the source (Bellhop's own bathymetry, so
        bottom bounces and blocked rays are traced over it). Used once per azimuth sector by
        the reverberation over relief (sim/reverb_relief.py): N x 2-D."""
        import copy
        import dataclasses
        profile = [[float(r), float(d)] for r, d in profile]
        deepest = max(self.env.depth, max(d for _, d in profile))
        other = copy.copy(self)
        other.env = dataclasses.replace(self.env, depth=deepest, bathymetry=profile)
        dz = self.depths[1] - self.depths[0] if len(self.depths) > 1 else deepest / 50.0
        other.depths = np.arange(0.5 * dz, deepest, dz)
        other._cache = {}
        other.relief_aware = True
        return other

    def arrivals_for(self, source_depth: float) -> Arrivals:
        key = round(float(source_depth), 2)
        if key not in self._cache:
            self._cache[key] = run_bellhop(self.env, self.frequency, key, self.depths, self.ranges,
                                           self.n_beams, self.angles)
        return self._cache[key]

    def eigenrays(self, source, points) -> EigenraySet:
        source = np.asarray(source, dtype=float)
        points = np.atleast_2d(np.asarray(points, dtype=float))
        arr = self.arrivals_for(source[2])
        c_src = self.env.speed_at(source[2])
        c_surface = self.env.speed_at(0.0)
        s, k = len(points), self.max_arrivals
        out = EigenraySet(
            delay=np.zeros((s, k)), amplitude=np.zeros((s, k), dtype=complex),
            launch=np.zeros((s, k, 3)), arrival=np.zeros((s, k, 3)),
            n_surface=np.zeros((s, k), dtype=int), n_bottom=np.zeros((s, k), dtype=int),
            valid=np.zeros((s, k), dtype=bool), sound_speed=c_src)
        horiz = points[:, :2] - source[:2]
        rng = np.linalg.norm(horiz, axis=1)
        unit = np.divide(horiz, rng[:, None], out=np.tile([1.0, 0.0], (s, 1)), where=rng[:, None] > 0)
        ir = np.clip(np.rint((rng - self.ranges[0]) / (self.ranges[1] - self.ranges[0])).astype(int),
                     0, len(self.ranges) - 1)
        iz = np.clip(np.rint((points[:, 2] - self.depths[0]) / (self.depths[1] - self.depths[0])).astype(int),
                     0, len(self.depths) - 1)
        for i in range(s):
            amp = arr.amplitude[iz[i]][ir[i]]
            if len(amp) == 0:
                continue
            if self.wind_speed > 0.0:
                # Grazing angle at the surface from the launch angle by Snell's law.
                cos_s = np.clip(np.cos(arr.launch_angle[iz[i]][ir[i]]) * c_surface / c_src, -1.0, 1.0)
                loss = acoustics.surface_coherent_reflection(np.arccos(cos_s), self.wind_speed,
                                                             self.frequency, c_surface)
                amp = amp * loss ** arr.n_surface[iz[i]][ir[i]]
            order = np.argsort(-np.abs(amp))[:k]
            n = len(order)
            la, aa = arr.launch_angle[iz[i]][ir[i]][order], arr.arrival_angle[iz[i]][ir[i]][order]
            # First-order delay correction from the grid node to the actual point.
            c_pt = self.env.speed_at(points[i, 2])
            dr = rng[i] - self.ranges[ir[i]]
            dz = points[i, 2] - self.depths[iz[i]]
            out.delay[i, :n] = arr.delay[iz[i]][ir[i]][order] + (dr * np.cos(aa) + dz * np.sin(aa)) / c_pt
            out.amplitude[i, :n] = amp[order]
            out.launch[i, :n, :2] = np.cos(la)[:, None] * unit[i]
            out.launch[i, :n, 2] = np.sin(la)
            out.arrival[i, :n, :2] = np.cos(aa)[:, None] * unit[i]
            out.arrival[i, :n, 2] = np.sin(aa)
            out.n_surface[i, :n] = arr.n_surface[iz[i]][ir[i]][order]
            out.n_bottom[i, :n] = arr.n_bottom[iz[i]][ir[i]][order]
            out.valid[i, :n] = True
        return out
