"""Bellhop as an external eigenray provider (the `eigenray` fidelity tier).

Bellhop (Acoustics Toolbox, M. Porter; GPL-3) is run as a separate process: an
environment file is written, `bellhop.exe` is executed, and its arrivals file is read
back. Nothing of Bellhop is linked or copied into this package. The executable is found
through the GPUSONAR_BELLHOP environment variable or as `bellhop.exe` on PATH.

A call per scatterer would be far too slow, so arrivals are computed once per source
depth on a range-depth grid and looked up; within a grid cell the delay is corrected to
first order with the arrival angle (a plane wave at the field point).

The water may change with range (`OceanEnvironment.ssp_range`, `ssp_speed_by_range`): profiles on one depth
grid at several ranges from the source, handed to Bellhop as its own 2-D sound-speed field (SSP option 'Q',
a `.ssp` file beside the `.env`; format from the Acoustics Toolbox's doc/RangeDepSSPFile.htm and the reader in
Bellhop/sspMod.f90, subroutine Quad: the number of profiles, their ranges in km, then one row per depth of the
`.env` profile with one speed per range). Bellhop interpolates linearly in depth and in range between them.
Without those two fields nothing here changes: the same `.env`, no `.ssp`.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import dataclasses
from dataclasses import dataclass, field

import numpy as np

from ..acoustics import NOMINAL_SOUND_SPEED

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

    The water may change with range the same way: `ssp_range` (m from the source, the first 0, increasing) and
    `ssp_speed_by_range` ([n_range][n_depth] m/s on the one depth grid `ssp_depth`, which starts at 0 and reaches
    the water depth). The first profile is the LOCAL one, the water at the source (the sonar): `ssp_speed`, when
    given too, must equal it, and `profile()` and `speed_at(depth)` return it. Between two ranges the speed is
    linear in range (Bellhop's quadrilateral interpolation); beyond the last range the last profile holds, as a
    bathymetry's last depth does. Like the bathymetry this is one section used on every bearing from every
    source: water that changes with distance from the sonar, not a front lying across a map.
    """

    depth: float                                   # water depth, m (the deepest point with bathymetry)
    ssp_depth: np.ndarray = None                   # m, from 0 to `depth`
    ssp_speed: np.ndarray = None                   # m/s
    bottom_speed: float = 1600.0                   # m/s, fluid half-space
    bottom_density: float = 1.8                    # g/cm^3
    bottom_attenuation: float = 0.5                # dB per wavelength
    surface: str = "vacuum"                        # pressure release
    sound_speed: float = NOMINAL_SOUND_SPEED                    # used when no profile is given
    bathymetry: list = None                        # [[range_m, depth_m], ...] from the source, or None
    sediment: str = None                           # a name in configs/sediments.yaml in place of the three bottom_* numbers
    sediment_file: str = None                      # another table for `sediment`
    sediment_name: str = field(default=None, init=False)   # the name `sediment` was resolved from
    ssp_range: list = None                         # [range_m, ...] from the source, the first 0: the water changes with range
    ssp_speed_by_range: list = None                # [n_range][n_depth] m/s on `ssp_depth`; row 0 is the local profile

    def __post_init__(self):
        if self.ssp_range is not None or self.ssp_speed_by_range is not None:
            self._check_sections()
        if self.sediment is None:
            return
        # The table holds ratios: speed = ratio x the water sound speed at the bottom; Bellhop's
        # water density is 1 g/cm^3, so the density ratio is the number it takes.
        defaults = {f.name: f.default for f in dataclasses.fields(self) if f.name.startswith("bottom_")}
        if any(getattr(self, k) != v for k, v in defaults.items()):
            raise ValueError("OceanEnvironment: give `sediment` or the bottom_* numbers, not both")
        from . import bottom
        half = bottom.fluid_halfspace(self.sediment, self.speed_at(self.depth), self.sediment_file)
        self.bottom_speed, self.bottom_density, self.bottom_attenuation = half["speed"], half["density"], half["attenuation"]
        # The numbers now stand for the name, so that dataclasses.replace (along_profile) keeps them.
        self.sediment_name, self.sediment = self.sediment, None

    def _check_sections(self) -> None:
        """The range-dependent water holds together (ValueError otherwise); `ssp_speed` becomes the local profile."""
        what = "OceanEnvironment: a sound speed that changes with range"
        if self.ssp_range is None or self.ssp_speed_by_range is None or self.ssp_depth is None:
            raise ValueError(f"{what} needs `ssp_range`, `ssp_speed_by_range` and the depth grid `ssp_depth`, all three")
        r, z = np.asarray(self.ssp_range, dtype=float), np.asarray(self.ssp_depth, dtype=float)
        try:
            c = np.asarray(self.ssp_speed_by_range, dtype=float)
        except ValueError:
            c = np.zeros(0)                                   # rows of different lengths
        if r.ndim != 1 or len(r) < 2:
            raise ValueError(f"{what} needs profiles at two ranges or more (one profile is `ssp_depth`, `ssp_speed`)")
        if r[0] != 0.0 or np.any(np.diff(r) <= 0.0) or not np.all(np.isfinite(r)):
            raise ValueError(f"{what}: `ssp_range` must start at 0 (the profile at the source) and increase; it is {r.tolist()}")
        if c.ndim != 2 or c.shape != (len(r), len(z)):
            raise ValueError(f"{what}: `ssp_speed_by_range` must hold one profile per range, each with one speed per depth of `ssp_depth` "
                             f"({len(r)} x {len(z)}); every profile is on the same depth grid")
        if z.ndim != 1 or len(z) < 2 or z[0] != 0.0 or np.any(np.diff(z) <= 0.0):
            raise ValueError(f"{what}: `ssp_depth` must start at 0 (the surface) and increase")
        if z[-1] < self.depth - 1e-6:
            raise ValueError(f"{what}: the profiles end at {z[-1]:g} m and the water is {self.depth:g} m deep (the deepest point with a "
                             "bathymetry); they must reach the bottom, nothing is extended")
        if not np.all(np.isfinite(c)) or np.any(c <= 0.0):
            raise ValueError(f"{what}: every speed must be a positive number")
        if self.ssp_speed is not None and not np.array_equal(np.asarray(self.ssp_speed, dtype=float), c[0]):
            raise ValueError(f"{what}: `ssp_speed` is the local profile and must equal the first of `ssp_speed_by_range` (or be left out)")
        self.ssp_speed = c[0].copy()

    def section(self):
        """(ranges [R] m, depths [Z] m, speeds [R, Z] m/s) of the range-dependent water, the depth grid cut at the
        water depth (the speed there by the same linear interpolation Bellhop uses), or None."""
        if self.ssp_range is None:
            return None
        r, z, c = np.asarray(self.ssp_range, dtype=float), np.asarray(self.ssp_depth, dtype=float), np.asarray(self.ssp_speed_by_range, dtype=float)
        if z[-1] > self.depth + 1e-6:
            keep = z < self.depth - 1e-9
            at_bottom = np.array([np.interp(self.depth, z, row) for row in c])
            z, c = np.r_[z[keep], self.depth], np.column_stack([c[:, keep], at_bottom])
        return r, z, c

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

    def speed_at(self, depth: float, range_m: float | None = None) -> float:
        """Sound speed at a depth: of the local profile, or, with `range_m` in water that changes with range, at that
        range from the source (linear in depth within each profile, then linear in range; the end profiles hold
        outside the ranges given: what Bellhop traces through)."""
        if range_m is not None and self.ssp_range is not None:
            r, z, c = self.section()
            j = int(np.clip(np.searchsorted(r, range_m, side="right") - 1, 0, len(r) - 2))
            w = float(np.clip((range_m - r[j]) / (r[j + 1] - r[j]), 0.0, 1.0))
            return float((1.0 - w) * np.interp(depth, z, c[j]) + w * np.interp(depth, z, c[j + 1]))
        z, c = self.profile()
        return float(np.interp(depth, z, c))


def munk_profile(depth: float = 5000.0, n: int = 51, axis_depth: float = 1300.0,
                 axis_speed: float = NOMINAL_SOUND_SPEED, epsilon: float = 0.00737):
    """The canonical Munk deep-water sound-speed profile."""
    z = np.linspace(0.0, depth, n)
    zt = 2.0 * (z - axis_depth) / 1300.0
    return z, axis_speed * (1.0 + epsilon * (zt - 1.0 + np.exp(-zt)))


def write_env(path: str, env: OceanEnvironment, frequency: float, source_depth: float,
              receiver_depths, receiver_ranges, run_type: str = "A", n_beams: int = 0,
              angles=(-80.0, 80.0), title: str = "gpusonar") -> None:
    z, c = env.profile()
    ssp_option = "CVW"
    if env.ssp_range is not None:
        # Water that changes with range: Bellhop's 2-D field ('Q'). The `.env` keeps the local profile on the shared
        # depth grid (Bellhop takes the depths from it); the `.ssp` file holds one row per depth with one speed per
        # profile range (km). Bellhop stops with an error where a ray leaves the ranges given, and traces to its box
        # (the last line of the `.env`) on either side of the source, so the first profile is repeated before the
        # source and the last one beyond the box: the end profiles hold outside what was given.
        env._check_sections()
        sr, z, cm = env.section()
        c = cm[0]
        ssp_option = "QVW"
        edge = max(1.05 * float(np.max(receiver_ranges)) + 100.0, sr[-1]) + 1000.0      # beyond the box
        cols = np.column_stack([cm[0], cm.T, cm[-1]])                                    # [n_depth, n_range + 2]
        with open(os.path.splitext(path)[0] + ".ssp", "w", encoding="ascii") as f:
            f.write(f"{len(sr) + 2}\n" + " ".join(f"{v / 1000.0:.6f}" for v in np.r_[-edge, sr, edge]) + "\n"
                    + "".join(" ".join(f"{v:.4f}" for v in row) + "\n" for row in cols))
    lines = [f"'{title}'", f"{frequency:.6f}", "1", f"'{ssp_option}'", f"{len(z)}  0.0  {env.depth:.4f}"]
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

    Geometry is reduced to (horizontal range, depth): the environment is the same in every direction
    from the source of each call (the sonar: its projector, or the array's reference point for the return leg
    and for sources heard one way). It is range independent unless the ocean has a `bathymetry` or a sound
    speed by range (`ssp_range`): both are then functions of the horizontal distance from that source, on every
    bearing. Depth is the NED z coordinate measured from the sea surface (z = 0 is the surface).

    In water that changes with range, what this class works out for itself uses: the LOCAL profile (range 0) for
    the speed at the source (`EigenraySet.sound_speed`, which turns a delay into an apparent position) and for the
    surface grazing angle of the rough-surface loss (Snell's law from the launch angle, exact only where the
    speed does not change along the ray: an approximation when `wind_speed` > 0); the speed at the field point's
    own range and depth for the first-order delay correction from the grid node to the point.
    """

    tier = "eigenray"

    def __init__(self, env: OceanEnvironment, frequency: float, max_range: float,
                 range_step: float | None = None, depth_step: float | None = None,
                 max_arrivals: int | None = None, n_beams: int = 0, angles=(-80.0, 80.0),
                 min_range: float = 1.0, wind_speed: float = 0.0, arrival_floor_db: float = 20.0, max_arrivals_cap: int = 64):
        # Arrivals kept per field point: `max_arrivals` (the strongest) when given. Left out (D-105): every arrival
        # within `arrival_floor_db` of the point's strongest, up to `max_arrivals_cap`; when the cap cuts off arrivals
        # above the floor a warning says how much energy was dropped, once, and `truncation` holds it (before D-105 the
        # default was the strongest eight, silently). `dropped_energy` is the largest fraction of a point's
        # arrival energy left out so far, whichever rule cut it.
        self.arrival_floor_db = float(arrival_floor_db)
        self.max_arrivals_cap = int(max_arrivals_cap)
        self.arrivals_used = None
        self.dropped_energy = 0.0
        self.truncation = None
        self.env = env
        # Wind speed (m/s) for the rough-surface reflection loss, applied per surface bounce
        # to Bellhop's flat-surface arrivals (acoustics.surface_coherent_reflection). 0: flat.
        self.wind_speed = float(wind_speed)
        self.frequency = float(frequency)
        self.max_arrivals = None if max_arrivals is None else int(max_arrivals)
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
        horiz = points[:, :2] - source[:2]
        rng = np.linalg.norm(horiz, axis=1)
        ir = np.clip(np.rint((rng - self.ranges[0]) / (self.ranges[1] - self.ranges[0])).astype(int),
                     0, len(self.ranges) - 1)
        iz = np.clip(np.rint((points[:, 2] - self.depths[0]) / (self.depths[1] - self.depths[0])).astype(int),
                     0, len(self.depths) - 1)
        s, k = len(points), self.max_arrivals
        if k is None:
            # The energy criterion, on the grid nodes these points use (flat-surface amplitudes: a rough-surface
            # loss only lowers the later ones further).
            floor, need, over = 10.0 ** (-self.arrival_floor_db / 20.0), 1, 0.0
            for a_iz, a_ir in {(int(a), int(b)) for a, b in zip(iz, ir)}:
                mag = np.sort(np.abs(arr.amplitude[a_iz][a_ir]))[::-1]
                if len(mag) == 0:
                    continue
                n = int(np.sum(mag >= floor * mag[0]))
                if n > self.max_arrivals_cap:
                    over = max(over, float(np.sum(mag[self.max_arrivals_cap:] ** 2) / np.sum(mag ** 2)))
                need = max(need, min(n, self.max_arrivals_cap))
            k = need if self.arrivals_used is None else max(self.arrivals_used, need)
            if over > 0.0:
                if self.truncation is None:
                    import warnings
                    warnings.warn(f"eigenray: arrivals were cut at the cap of {self.max_arrivals_cap} per point while more lie within "
                                  f"{self.arrival_floor_db:g} dB of the strongest: {100.0 * over:.1f} % of a point's arrival energy is left out. "
                                  "Raise `max_arrivals_cap`, or state `max_arrivals`", RuntimeWarning, stacklevel=3)
                self.truncation = {"arrivals": self.max_arrivals_cap, "dropped_energy": max(over, (self.truncation or {}).get("dropped_energy", 0.0))}
        self.arrivals_used = k
        out = EigenraySet(
            delay=np.zeros((s, k)), amplitude=np.zeros((s, k), dtype=complex),
            launch=np.zeros((s, k, 3)), arrival=np.zeros((s, k, 3)),
            n_surface=np.zeros((s, k), dtype=int), n_bottom=np.zeros((s, k), dtype=int),
            valid=np.zeros((s, k), dtype=bool), sound_speed=c_src)
        unit = np.divide(horiz, rng[:, None], out=np.tile([1.0, 0.0], (s, 1)), where=rng[:, None] > 0)
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
            ranked = np.argsort(-np.abs(amp))
            order = ranked[:k]
            if len(ranked) > k:
                self.dropped_energy = max(self.dropped_energy, float(np.sum(np.abs(amp[ranked[k:]]) ** 2) / max(np.sum(np.abs(amp) ** 2), 1e-300)))
            n = len(order)
            la, aa = arr.launch_angle[iz[i]][ir[i]][order], arr.arrival_angle[iz[i]][ir[i]][order]
            # First-order delay correction from the grid node to the actual point.
            c_pt = self.env.speed_at(points[i, 2]) if self.env.ssp_range is None else self.env.speed_at(points[i, 2], rng[i])
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
