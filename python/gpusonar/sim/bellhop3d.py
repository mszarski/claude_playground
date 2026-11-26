"""Bellhop3D as an external program: its input writers, its arrivals reader and the merging of its beam
contributions into eigenrays (docs/tasks/PROPAGATION_3D.md, option A, packages P2 and P3).

Bellhop3D (Acoustics Toolbox, M. Porter; GPL-3) is run as a separate process, exactly as Bellhop is
(sim/bellhop.py): files are written, `bellhop3d.exe` runs in a temporary directory, its output is read back.
Nothing of it is linked or copied into this package (docs/OPEN_SOURCE.md rules 2 and 3; the owner approved it as
an opt-in provider, 2026-11-26). The executable is GPUSONAR_BELLHOP3D, else `bellhop3d.exe` beside
GPUSONAR_BELLHOP (an Acoustics Toolbox build puts both in Bellhop/; tools/build_oracles.sh builds it), else on
PATH.

File formats, from the Acoustics Toolbox at HEAD 8b4682b (opened; format only): the run-type letters and the
beam fan from doc/bellhop3d.htm (RunType(6:6) '2' = N x 2-D, '3' = 3-D; bearings in degrees, "math convention
with 0 degrees pointing along the x-axis": here x is north and y east, so a bearing turns from north towards
east as the suite's azimuths do); the bathymetry file from Bellhop/bdry3DMod.f90, ReadBTY3D (a type letter, Nx,
the x values in km, Ny, the y values in km, then Ny rows of Nx depths in m); the arrivals file from
Bellhop/ReadEnvironmentBell.f90 (its header: '3D', the frequency, the source x, y, z, the receiver depths,
ranges and bearings: positions in metres, the program having converted its km inputs, misc/SourceReceiverPositions.f90) and Bellhop/ArrMod.f90, WriteArrivalsASCII3D and
WriteArrivalsBinary3D (per bearing, depth, range: the count, then per arrival amplitude, phase (degrees), delay
(real, imaginary: s), source declination and azimuth, receiver declination and azimuth (degrees), surface and
bottom bounces). In 3-D mode the amplitudes are those of the coherent field (the arrival influence is that of
the coherent TL run, Bellhop/influence3D.f90 ApplyContribution, which ScalePressure3D leaves unscaled for these
beam types); in N x 2-D mode the writer applies 1 / sqrt(r) itself.

What Bellhop3D's arrivals are, and why they are merged here (measured, docs/tasks/PROPAGATION_3D.md section 1
item 3 and P2): one receiver gets one arrival per beam contribution that the program did not join to the one
before it. It joins only consecutive contributions (ArrMod.f90 AddArr3D: phase and omega * delay within 0.05),
and consecutive means adjacent in declination: neighbours in azimuth are traced a whole declination fan apart,
so a single ray family reaches a receiver as tens to hundreds of entries. `merge_arrivals` sums them back.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field

import numpy as np

from .propagation import EigenrayProvider


class Bellhop3DError(RuntimeError):
    pass


def find_bellhop3d() -> str | None:
    path = os.environ.get("GPUSONAR_BELLHOP3D")
    if path and os.path.exists(path):
        return path
    beside = os.path.join(os.path.dirname(os.environ.get("GPUSONAR_BELLHOP", "") or "."), "bellhop3d.exe")
    if os.environ.get("GPUSONAR_BELLHOP") and os.path.exists(beside):
        return beside
    return shutil.which("bellhop3d.exe")


def _vector(values, scale=1.0, fmt="{:.6f}") -> str:
    """A Bellhop vector: 'first last /' when equally spaced (Bellhop fills it in), else every value."""
    v = np.atleast_1d(np.asarray(values, dtype=float)) * scale
    if len(v) > 2 and np.allclose(np.diff(v), v[1] - v[0], rtol=0.0, atol=1e-9 * max(1.0, np.abs(v).max())):
        return f"{fmt.format(v[0])} {fmt.format(v[-1])} /"
    return " ".join(fmt.format(x) for x in v) + " /"


def write_bty3d(path: str, depth_fn, x, y) -> None:
    """The bed as Bellhop3D's rectilinear grid ('R'): `depth_fn(x, y)` (m, positive down) sampled on x and y (m)."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    table = np.asarray(depth_fn(x[None, :], y[:, None]), dtype=float)          # [ny, nx]: one row per y
    with open(path, "w", encoding="ascii") as f:
        f.write("'R'\n" + f"{len(x)}\n" + " ".join(f"{v / 1000.0:.6f}" for v in x) + "\n"
                + f"{len(y)}\n" + " ".join(f"{v / 1000.0:.6f}" for v in y) + "\n"
                + "".join(" ".join(f"{d:.4f}" for d in row) + "\n" for row in table))


def write_env3d(path: str, *, frequency: float, profile, depth: float, bottom, source_depth: float, receiver_depths,
                ranges, bearings_deg, run_type: str, mode: str, n_alpha: int, alpha_deg, n_beta: int, beta_deg,
                box_km, beam: str = "B", title: str = "gpusonar 3-D") -> None:
    """The `.env` of a Bellhop3D run with a source at (0, 0, `source_depth`): `profile` = (z [m], c [m/s]) of the
    water down to `depth`, `bottom` = (speed, density, attenuation dB/wavelength) of a fluid half-space under the
    bed of the `.bty` beside it ('A~'), receivers on the bearings x depths x ranges given (m, degrees),
    `run_type` 'I', 'C', 'A' or 'a', `mode` '2' or '3', `beam` Bellhop's beam letter."""
    z, c = (np.asarray(v, dtype=float) for v in profile)
    rz = np.atleast_1d(np.asarray(receiver_depths, dtype=float))
    rr = np.atleast_1d(np.asarray(ranges, dtype=float))
    th = np.atleast_1d(np.asarray(bearings_deg, dtype=float))
    speed, density, attenuation = bottom
    lines = [f"'{title}'", f"{frequency:.6f}", "1", "'CVW'", f"{len(z)}  0.0  {depth:.4f}"]
    lines += [f"  {zi:.4f}  {ci:.4f}  /" for zi, ci in zip(z, c)]
    lines += ["'A~' 0.0", f" {depth:.4f}  {speed:.4f} 0.0 {density:.4f} {attenuation:.4f} /",
              "1", "0.0 /", "1", "0.0 /", "1", f"{source_depth:.4f} /",
              f"{len(rz)}", " ".join(f"{v:.4f}" for v in rz) + " /",
              f"{len(rr)}", _vector(rr, 1e-3),
              f"{len(th)}", _vector(th),
              f"'{run_type}{beam}   {mode}'", f"{n_alpha}", f"{alpha_deg[0]:.3f} {alpha_deg[1]:.3f} /",
              f"{n_beta}", f"{beta_deg[0]:.3f} {beta_deg[1]:.3f} /",
              f"0.0  {box_km[0]:.4f} {box_km[1]:.4f} {1.05 * depth:.4f}"]
    with open(path, "w", encoding="ascii") as f:
        f.write("\n".join(lines) + "\n")


# --- the arrivals file ---------------------------------------------------------------------------------------------

@dataclass
class Arrivals3D:
    """Bellhop3D arrivals for one source: per receiver (bearing i_theta, depth i_z, range i_r) arrays of the same
    length. Amplitudes complex in the suite's exp(+i omega t) convention (an echo of delay tau carries
    exp(-j 2 pi f tau)): Bellhop writes phases for exp(-i omega t), so they are conjugated here, as sim/bellhop.py
    does. Angles in radians: declination positive towards greater depth, azimuth from x (north) towards y (east)."""

    frequency: float
    source: np.ndarray                      # (x, y, z): m, in the run's frame (the provider puts the source at 0, 0)
    depths: np.ndarray
    ranges: np.ndarray                      # m
    bearings: np.ndarray                    # degrees, as given
    amplitude: dict = field(default_factory=dict)       # (i_theta, i_z, i_r) -> complex [n]
    delay: dict = field(default_factory=dict)           # s (the real part; the imaginary part is attenuation)
    source_decl: dict = field(default_factory=dict)
    source_azim: dict = field(default_factory=dict)
    receiver_decl: dict = field(default_factory=dict)
    receiver_azim: dict = field(default_factory=dict)
    n_surface: dict = field(default_factory=dict)
    n_bottom: dict = field(default_factory=dict)

    def keys(self):
        return [(t, z, r) for t in range(len(self.bearings)) for z in range(len(self.depths)) for r in range(len(self.ranges))]

    def counts(self) -> np.ndarray:
        return np.array([len(self.amplitude[k]) for k in self.keys()])

    def energy(self, key) -> float:
        return float(np.sum(np.abs(self.amplitude[key]) ** 2))


def _store(arr: Arrivals3D, key, rows: np.ndarray) -> None:
    """rows [n, 10]: A, phase (deg), delay re, delay im, src decl, src azim, rcv decl, rcv azim (deg), ns, nb."""
    arr.amplitude[key] = rows[:, 0] * np.exp(-1j * np.radians(rows[:, 1]))
    arr.delay[key] = rows[:, 2]
    arr.source_decl[key] = np.radians(rows[:, 4])
    arr.source_azim[key] = np.radians(rows[:, 5])
    arr.receiver_decl[key] = np.radians(rows[:, 6])
    arr.receiver_azim[key] = np.radians(rows[:, 7])
    arr.n_surface[key] = rows[:, 8].astype(int)
    arr.n_bottom[key] = rows[:, 9].astype(int)


def read_arrivals3d(path: str) -> Arrivals3D:
    """A Bellhop3D arrivals file, ASCII ('A') or binary ('a'), one frequency and one source."""
    with open(path, "rb") as f:
        head = f.read(16)
    if b"3D" in head[:12] and not head.lstrip().startswith(b"'3D'"):
        return _read_binary(path)
    with open(path, "r", encoding="ascii") as f:
        it = iter(f.read().replace("'", " ").split())
    if next(it) != "3D":
        raise Bellhop3DError("not a 3-D arrivals file")
    freq = float(next(it))
    sx = [float(next(it)) for _ in range(int(next(it)))]
    sy = [float(next(it)) for _ in range(int(next(it)))]
    sz = [float(next(it)) for _ in range(int(next(it)))]
    if len(sx) != 1 or len(sy) != 1 or len(sz) != 1:
        raise Bellhop3DError("expected one source position")
    rz = np.array([float(next(it)) for _ in range(int(next(it)))])
    rr = np.array([float(next(it)) for _ in range(int(next(it)))])
    th = np.array([float(next(it)) for _ in range(int(next(it)))])
    arr = Arrivals3D(freq, np.array([sx[0], sy[0], sz[0]]), rz, rr, th)
    next(it)                                                                   # the largest count
    for key in arr.keys():
        n = int(next(it))
        rows = np.array([float(next(it)) for _ in range(10 * n)]).reshape(n, 10) if n else np.zeros((0, 10))
        _store(arr, key, rows)
    return arr


def _read_binary(path: str) -> Arrivals3D:
    """Fortran sequential unformatted records (gfortran: a 4-byte length before and after each record)."""
    raw = open(path, "rb").read()
    pos, records = 0, []
    while pos < len(raw):
        n = int(np.frombuffer(raw, "<i4", 1, pos)[0])
        records.append(raw[pos + 4:pos + 4 + n])
        pos += n + 8
    it = iter(records)
    if b"3D" not in next(it):
        raise Bellhop3DError("not a 3-D arrivals file")
    freq = float(np.frombuffer(next(it), "<f4")[0])
    src = []
    for _ in range(3):
        r = next(it)
        n = int(np.frombuffer(r, "<i4", 1)[0])
        if n != 1:
            raise Bellhop3DError("expected one source position")
        src.append(float(np.frombuffer(r, "<f8", 1, 4)[0]) if len(r) >= 12 else float(np.frombuffer(r, "<f4", 1, 4)[0]))

    def vec(r):
        n = int(np.frombuffer(r, "<i4", 1)[0])
        width = (len(r) - 4) // max(n, 1)
        return np.frombuffer(r, "<f8" if width == 8 else "<f4", n, 4).astype(float)
    rz, rr, th = vec(next(it)), vec(next(it)), vec(next(it))
    arr = Arrivals3D(freq, np.array(src), rz, rr, th)
    next(it)                                                                   # the largest count
    for key in arr.keys():
        n = int(np.frombuffer(next(it), "<i4", 1)[0])
        rows = np.zeros((n, 10))
        for k in range(n):
            r = next(it)
            # A (real*8: an 8-byte factor times a real), phase (real*4), delay (complex*8), four angles (real*4),
            # two bounce counts written as real*4: 8 + 4 + 8 + 16 + 8 = 44 bytes.
            if len(r) != 44:
                raise Bellhop3DError(f"unexpected binary arrival record of {len(r)} bytes")
            rows[k, 0] = np.frombuffer(r, "<f8", 1, 0)[0]
            rows[k, 1] = np.frombuffer(r, "<f4", 1, 8)[0]
            rows[k, 2:4] = np.frombuffer(r, "<f4", 2, 12)
            rows[k, 4:8] = np.frombuffer(r, "<f4", 4, 20)
            rows[k, 8:10] = np.frombuffer(r, "<f4", 2, 36)
        _store(arr, key, rows)
    return arr


# --- merging -------------------------------------------------------------------------------------------------------

@dataclass
class Merged:
    """Eigenrays of one receiver after merging: the same fields as Arrivals3D's per-receiver arrays."""

    amplitude: np.ndarray
    delay: np.ndarray
    source_decl: np.ndarray
    source_azim: np.ndarray
    receiver_decl: np.ndarray
    receiver_azim: np.ndarray
    n_surface: np.ndarray
    n_bottom: np.ndarray
    contributions: np.ndarray               # how many beam contributions each eigenray holds


def _circular_mean(angles, weights):
    return np.arctan2(np.sum(weights * np.sin(angles)), np.sum(weights * np.cos(angles)))


def merge_arrivals(arr: Arrivals3D, key, d_alpha: float, d_beta: float, reach: float = 1.5) -> Merged:
    """The beam contributions at one receiver merged into eigenrays (ours: docs/tasks/PROPAGATION_3D.md, P2).

    A hat beam ('G') reaches a receiver only from within one beam spacing of it, so one ray family arrives as the
    contributions of the two to four neighbouring beams (in declination and azimuth) around it, whose hat weights
    sum to one. Contributions are therefore joined when they have the same surface and bottom bounce counts and
    their SOURCE angles are neighbours: declinations within `reach` x `d_alpha` and azimuths within `reach` x
    `d_beta` (radians: the run's beam spacings), transitively. Distinct families leave the source at distinct
    angles and stay apart even where they arrive at one delay and angle (over relief they do). A group's amplitude
    has the sum of its contributions' magnitudes (the weights sum to one; Bellhop's own merge of consecutive
    contributions adds magnitudes too, ArrMod.f90 AddArr3D) and the phase of their coherent sum; its delay and angles
    are energy-weighted means (azimuths circular). Measured against Bellhop3D's own incoherent TL run on a level bed
    and over the seamount: docs/tasks/PROPAGATION_3D.md, P2. Gaussian beams overlap many more beams with weights
    that do not sum to one, and their arrivals do not hold the field (measured, same place): use hat beams."""
    amp, delay = arr.amplitude[key], arr.delay[key]
    if len(amp) == 0:
        z = np.zeros(0)
        return Merged(z.astype(complex), z, z, z, z, z, z.astype(int), z.astype(int), z.astype(int))
    ns, nb = arr.n_surface[key], arr.n_bottom[key]
    rd, ra = arr.receiver_decl[key], arr.receiver_azim[key]
    sd, sa = arr.source_decl[key], arr.source_azim[key]
    n = len(amp)
    parent = np.arange(n)

    def find(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k
    near_d = np.abs(sd[:, None] - sd[None, :]) <= reach * d_alpha
    near_a = np.abs(np.angle(np.exp(1j * (sa[:, None] - sa[None, :])))) <= reach * d_beta
    same = (ns[:, None] == ns[None, :]) & (nb[:, None] == nb[None, :]) & near_d & near_a
    for i, j in zip(*np.nonzero(np.triu(same, 1))):
        a, b = find(i), find(j)
        if a != b:
            parent[b] = a
    roots = np.array([find(k) for k in range(n)])
    groups = [np.flatnonzero(roots == r) for r in np.unique(roots)]
    out = {name: [] for name in ("a", "t", "sd", "sa", "rd", "ra", "ns", "nb", "n")}
    for g in groups:
        w = np.abs(amp[g]) ** 2 + 1e-300
        total = np.sum(amp[g])
        out["a"].append(np.sum(np.abs(amp[g])) * (total / abs(total) if abs(total) > 0.0 else 1.0))
        out["t"].append(np.average(delay[g], weights=w))
        out["sd"].append(np.average(arr.source_decl[key][g], weights=w))
        out["sa"].append(_circular_mean(arr.source_azim[key][g], w))
        out["rd"].append(np.average(rd[g], weights=w))
        out["ra"].append(_circular_mean(ra[g], w))
        out["ns"].append(int(ns[g[0]]))
        out["nb"].append(int(nb[g[0]]))
        out["n"].append(len(g))
    order = np.argsort(-np.abs(np.asarray(out["a"])))
    pick = lambda name, dtype=float: np.asarray(out[name], dtype=dtype)[order]          # noqa: E731
    return Merged(pick("a", complex), pick("t"), pick("sd"), pick("sa"), pick("rd"), pick("ra"), pick("ns", int),
                  pick("nb", int), pick("n", int))


# --- the provider ---------------------------------------------------------------------------------------------------

# Measured bytes per receiver of a binary ('a') arrivals file at the default fan (hat beams 241 x 361): 3.7 kB on a
# level 200 m bed at 1 to 5 km, 4.8 kB over the seamount of test_seamount_3d.py at 2 to 6 km (P2; 52 bytes an
# arrival, up to 186 arrivals a receiver). The size check takes 10 kB a receiver, about twice the measured.
BYTES_PER_RECEIVER = 10_000


def _run(exe: str, tmp: str, timeout: float) -> None:
    import subprocess
    proc = subprocess.run([exe, "run"], cwd=tmp, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        prt = os.path.join(tmp, "run.prt")
        log = open(prt, errors="replace").read()[-800:] if os.path.exists(prt) else ""
        raise Bellhop3DError(f"Bellhop3D failed (exit {proc.returncode}): {proc.stderr[-300:]}\n{log}")


def _read_tl(path: str) -> np.ndarray:
    """Incoherent pressure magnitude [n_theta, n_rz, n_rr] of a shade file (one frequency, one source)."""
    raw = open(path, "rb").read()
    recl = 4 * int(np.frombuffer(raw, dtype="<i4", count=1)[0])
    rec = lambda i: raw[i * recl:(i + 1) * recl]                                 # noqa: E731
    n_theta, n_rz, n_rr = (int(v) for v in np.frombuffer(rec(2), dtype="<i4", count=7)[[1, 5, 6]])
    return np.abs(np.stack([np.stack([np.frombuffer(rec(10 + it * n_rz + iz), dtype="<c8", count=n_rr)
                                      for iz in range(n_rz)]) for it in range(n_theta)]))


class Bellhop3DProvider(EigenrayProvider):
    """Eigenrays over a 3-D bed from Bellhop3D (the opt-in `eigenray3d` tier; docs/tasks/PROPAGATION_3D.md P3).

    For each source position (x, y rounded to `track_tolerance`, z to 0.01 m) Bellhop3D runs in 3-D mode with hat
    beams (`n_alpha` declinations over `angles`, `n_beta` azimuths over +-`azimuth_span` / 2 about the mean bearing of
    the points asked for) for ONLY the receivers the points need: their bearings snapped to `bearing_step` degrees,
    their ranges and depths to the `range_step` / `depth_step` grid (the nodes, as the 2-D provider looks up), as the
    product bearings x depths x ranges of the nodes not yet cached. Two runs a time: binary arrivals ('a') and the
    incoherent TL ('I') at the same receivers. The arrivals are merged into eigenrays (`merge_arrivals`) and each
    receiver's eigenrays are scaled so that their summed energy is the TL run's (over relief merged hat-beam arrivals
    miss it by up to 4 dB at single receivers; on a level bed by 0.05 dB: P2); the largest scaling so far is
    `energy_correction_db`. Then the arrival rule of D-105 (`max_arrivals`, else within `arrival_floor_db` of the
    strongest, up to `max_arrivals_cap`, with the same warning and `truncation`).

    A point takes its node's eigenrays with a first-order delay correction along the 3-D arrival direction (the
    2-D provider's, D-36). `launch` and `arrival` are true 3-D unit vectors: a ray turned out of its vertical plane
    by the bed arrives with its own azimuth.

    Disk: a run's arrivals file is about 4 to 5 kB a receiver (measured, `BYTES_PER_RECEIVER`); a request whose
    estimate exceeds `max_file_mb` is refused with the estimate, before anything is written. Files live in a
    temporary directory removed after each run. Run time: about 90 s a run over the seamount at the default fan on
    a shared 4-core machine (two runs a source position).

    The bed: `bathymetry` (a sim/bathymetry.py description or a Bathymetry; None: level at `env.depth`), written as
    Bellhop3D's grid every `grid_step` m over the box the fan can reach (12.5 m: the seamount's summit shadow needs it,
    sonar-6c8). The water: `env`'s profile (range-independent: `ssp_range` is refused), its bottom half-space.
    Surface roughness: `wind_speed` as the 2-D provider applies it, per surface bounce."""

    tier = "eigenray3d"
    relief_aware = True

    def __init__(self, env, frequency: float, max_range: float, bathymetry=None, range_step: float | None = None,
                 depth_step: float | None = None, bearing_step: float = 0.5, min_range: float = 1.0,
                 n_alpha: int = 241, angles=(-30.0, 30.0), n_beta: int = 361, azimuth_span: float = 120.0,
                 grid_step: float = 12.5, max_arrivals: int | None = None, arrival_floor_db: float = 20.0,
                 max_arrivals_cap: int = 64, wind_speed: float = 0.0, max_file_mb: float = 300.0,
                 track_tolerance: float = 1.0, timeout: float = 3600.0):
        from .bathymetry import make_bathymetry, Bathymetry
        if getattr(env, "ssp_range", None) is not None:
            raise ValueError("eigenray3d: water that changes with range (ssp_range) is not supported; the 3-D tier takes one profile")
        if getattr(env, "bathymetry", None) is not None:
            raise ValueError("eigenray3d: the bed is `bathymetry` (a 3-D description), not environment.bathymetry (a range profile of the 2-D tier)")
        if n_alpha < 241:
            raise ValueError(f"eigenray3d: n_alpha {n_alpha} is below 241, the hat-beam count found converged over +-30 degrees (sonar-6c8)")
        self.env, self.frequency, self.max_range = env, float(frequency), float(max_range)
        self.bed = (make_bathymetry({"plane": {"depth": env.depth}}) if bathymetry is None
                    else bathymetry if isinstance(bathymetry, Bathymetry) else make_bathymetry(bathymetry))
        step = range_step or self.max_range / 200.0
        self.ranges = np.arange(min_range, self.max_range + step, step)
        dz = depth_step or env.depth / 50.0
        self.depths = np.arange(0.5 * dz, env.depth, dz)
        self.bearing_step = float(bearing_step)
        self.n_alpha, self.angles, self.n_beta, self.azimuth_span = int(n_alpha), tuple(angles), int(n_beta), float(azimuth_span)
        self.grid_step, self.max_file_mb, self.track_tolerance, self.timeout = float(grid_step), float(max_file_mb), float(track_tolerance), float(timeout)
        self.max_arrivals = None if max_arrivals is None else int(max_arrivals)
        self.arrival_floor_db, self.max_arrivals_cap = float(arrival_floor_db), int(max_arrivals_cap)
        self.wind_speed = float(wind_speed)
        self.sound_speed = env.speed_at(0.0)
        self.arrivals_used, self.truncation, self.dropped_energy = None, None, 0.0
        self.energy_correction_db = 0.0
        self.runs = []                                   # per Bellhop3D run: receivers, seconds, bytes
        self._cache: dict = {}

    # -- nodes ---------------------------------------------------------------------------------------------------
    def _key(self, source):
        t = self.track_tolerance
        return (round(float(source[0]) / t) * t, round(float(source[1]) / t) * t, round(float(source[2]), 2))

    def _nodes(self, source, points):
        d = points[:, :2] - source[:2]
        rng = np.hypot(d[:, 0], d[:, 1])
        bearing = np.degrees(np.arctan2(d[:, 1], d[:, 0]))
        it = np.rint(bearing / self.bearing_step).astype(int)
        ir = np.clip(np.rint((rng - self.ranges[0]) / (self.ranges[1] - self.ranges[0])).astype(int), 0, len(self.ranges) - 1)
        iz = np.clip(np.rint((points[:, 2] - self.depths[0]) / (self.depths[1] - self.depths[0])).astype(int), 0, len(self.depths) - 1)
        return it, iz, ir

    def estimate_bytes(self, n_receivers: int) -> int:
        return int(n_receivers) * BYTES_PER_RECEIVER

    # -- runs ----------------------------------------------------------------------------------------------------
    def _trace(self, key, it, iz, ir) -> None:
        """Run Bellhop3D for the product of the bearings, depths and ranges given (indices) and cache every node."""
        import tempfile
        import time
        th, zs, rs = np.unique(it), np.unique(iz), np.unique(ir)
        n = len(th) * len(zs) * len(rs)
        size = self.estimate_bytes(n)
        if size > self.max_file_mb * 1e6:
            raise Bellhop3DError(f"eigenray3d: {n} receivers ({len(th)} bearings x {len(zs)} depths x {len(rs)} ranges) would write about "
                                 f"{size / 1e6:.0f} MB of arrivals ({BYTES_PER_RECEIVER / 1e3:g} kB a receiver), over max_file_mb = "
                                 f"{self.max_file_mb:g}: ask for fewer points, coarser steps, or raise max_file_mb")
        exe = find_bellhop3d()
        if exe is None:
            raise Bellhop3DError("Bellhop3D is not installed: set GPUSONAR_BELLHOP3D to bellhop3d.exe")
        bearings = th * self.bearing_step
        centre = float(np.degrees(np.angle(np.mean(np.exp(1j * np.radians(bearings))))))
        spread = float(np.max(np.abs(np.angle(np.exp(1j * np.radians(bearings - centre)))))) * 180.0 / np.pi if len(th) > 1 else 0.0
        half = 0.5 * self.azimuth_span + spread
        beta = (centre - half, centre + half)
        sx, sy, sz = key
        reach = 1.05 * self.ranges[rs].max() + 100.0
        # The bed on the box the fan can reach (source at the origin of the run's frame).
        a = np.radians(np.linspace(beta[0], beta[1], 64))
        xs, ys = np.r_[0.0, reach * np.cos(a)], np.r_[0.0, reach * np.sin(a)]
        lo_x, hi_x, lo_y, hi_y = xs.min() - 200.0, xs.max() + 200.0, ys.min() - 200.0, ys.max() + 200.0
        step = self.grid_step if self.bed.flat_depth is None else max(self.grid_step, 500.0)
        # Nodes on multiples of the step from the source, so that a bed symmetric about the source gives a symmetric grid.
        gx = np.arange(np.floor(lo_x / step), np.ceil(hi_x / step) + 1) * step
        gy = np.arange(np.floor(lo_y / step), np.ceil(hi_y / step) + 1) * step
        bed = lambda x, y: self.bed.depth(x + sx, y + sy)                         # noqa: E731
        deepest = float(max(self.env.depth, np.max(bed(gx[None, :], gy[:, None]))))
        z, c = self.env.profile()
        if deepest > z[-1]:
            z, c = np.r_[z, deepest], np.r_[c, c[-1]]
        box = (max(abs(lo_x), abs(hi_x)) / 1000.0, max(abs(lo_y), abs(hi_y)) / 1000.0)
        bottom = (self.env.bottom_speed, self.env.bottom_density, self.env.bottom_attenuation)
        out = {}
        for run_type in ("a", "I"):
            with tempfile.TemporaryDirectory(prefix="gpusonar_bellhop3d_") as tmp:
                write_bty3d(os.path.join(tmp, "run.bty"), bed, gx, gy)
                write_env3d(os.path.join(tmp, "run.env"), frequency=self.frequency, profile=(z, c), depth=deepest, bottom=bottom,
                            source_depth=sz, receiver_depths=self.depths[zs], ranges=self.ranges[rs], bearings_deg=bearings,
                            run_type=run_type, mode="3", n_alpha=self.n_alpha, alpha_deg=self.angles, n_beta=self.n_beta,
                            beta_deg=beta, box_km=box, beam="G", title="gpusonar eigenray3d")
                t0 = time.time()
                _run(exe, tmp, self.timeout)
                if run_type == "a":
                    path = os.path.join(tmp, "run.arr")
                    out["a"] = read_arrivals3d(path)
                    self.runs.append(dict(receivers=n, seconds=time.time() - t0, bytes=os.path.getsize(path)))
                else:
                    out["I"] = _read_tl(os.path.join(tmp, "run.shd"))
                    self.runs[-1]["seconds_tl"] = time.time() - t0
        arr, tl = out["a"], out["I"]
        d_alpha = np.radians((self.angles[1] - self.angles[0]) / (self.n_alpha - 1))
        d_beta = np.radians((beta[1] - beta[0]) / (self.n_beta - 1))
        cache = self._cache.setdefault(key, {})
        for k in arr.keys():
            m = merge_arrivals(arr, k, d_alpha, d_beta)
            e_m, e_tl = float(np.sum(np.abs(m.amplitude) ** 2)), float(tl[k]) ** 2
            if e_m > 0.0:
                scale = np.sqrt(e_tl / e_m) if e_tl > 0.0 else 0.0
                m.amplitude = m.amplitude * scale
                if scale > 0.0:
                    self.energy_correction_db = max(self.energy_correction_db, abs(20.0 * np.log10(scale)))
            cache[(int(th[k[0]]), int(zs[k[1]]), int(rs[k[2]]))] = m

    # -- eigenrays -----------------------------------------------------------------------------------------------
    def eigenrays(self, source, points):
        from .propagation import EigenraySet
        from .. import acoustics
        source = np.asarray(source, dtype=float)
        points = np.atleast_2d(np.asarray(points, dtype=float))
        key = self._key(source)
        it, iz, ir = self._nodes(np.array(key), points)
        cache = self._cache.get(key, {})
        missing = [i for i in range(len(points)) if (int(it[i]), int(iz[i]), int(ir[i])) not in cache]
        if missing:
            self._trace(key, it[missing], iz[missing], ir[missing])
            cache = self._cache[key]
        nodes = [cache[(int(a), int(b), int(c))] for a, b, c in zip(it, iz, ir)]
        k = self.max_arrivals
        if k is None:
            floor, need, over = 10.0 ** (-self.arrival_floor_db / 20.0), 1, 0.0
            for m in {id(m): m for m in nodes}.values():
                mag = np.sort(np.abs(m.amplitude))[::-1]
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
                    warnings.warn(f"eigenray3d: arrivals were cut at the cap of {self.max_arrivals_cap} per point while more lie within "
                                  f"{self.arrival_floor_db:g} dB of the strongest: {100.0 * over:.1f} % of a point's arrival energy is left out. "
                                  "Raise `max_arrivals_cap`, or state `max_arrivals`", RuntimeWarning, stacklevel=2)
                self.truncation = {"arrivals": self.max_arrivals_cap, "dropped_energy": max(over, (self.truncation or {}).get("dropped_energy", 0.0))}
        self.arrivals_used = k
        s = len(points)
        c_src = self.env.speed_at(source[2])
        c_surface = self.env.speed_at(0.0)
        out = EigenraySet(delay=np.zeros((s, k)), amplitude=np.zeros((s, k), dtype=complex), launch=np.zeros((s, k, 3)),
                          arrival=np.zeros((s, k, 3)), n_surface=np.zeros((s, k), dtype=int), n_bottom=np.zeros((s, k), dtype=int),
                          valid=np.zeros((s, k), dtype=bool), sound_speed=c_src)
        unit = lambda decl, azim: np.stack([np.cos(decl) * np.cos(azim), np.cos(decl) * np.sin(azim), np.sin(decl)], axis=-1)  # noqa: E731
        for i, m in enumerate(nodes):
            amp = m.amplitude
            if len(amp) == 0:
                continue
            if self.wind_speed > 0.0:
                cos_s = np.clip(np.cos(m.source_decl) * c_surface / c_src, -1.0, 1.0)
                amp = amp * acoustics.surface_coherent_reflection(np.arccos(cos_s), self.wind_speed, self.frequency, c_surface) ** m.n_surface
            ranked = np.argsort(-np.abs(amp))
            order = ranked[:k]
            if len(ranked) > k:
                self.dropped_energy = max(self.dropped_energy, float(np.sum(np.abs(amp[ranked[k:]]) ** 2) / max(np.sum(np.abs(amp) ** 2), 1e-300)))
            n = len(order)
            arrival = unit(m.receiver_decl[order], m.receiver_azim[order])
            # First-order delay correction from the node to the point, along the 3-D arrival direction.
            node = np.array([key[0] + self.ranges[ir[i]] * np.cos(np.radians(it[i] * self.bearing_step)),
                             key[1] + self.ranges[ir[i]] * np.sin(np.radians(it[i] * self.bearing_step)), self.depths[iz[i]]])
            c_pt = self.env.speed_at(points[i, 2])
            out.delay[i, :n] = m.delay[order] + arrival @ (points[i] - node) / c_pt
            out.amplitude[i, :n] = amp[order]
            out.launch[i, :n] = unit(m.source_decl[order], m.source_azim[order])
            out.arrival[i, :n] = arrival
            out.n_surface[i, :n] = m.n_surface[order]
            out.n_bottom[i, :n] = m.n_bottom[order]
            out.valid[i, :n] = True
        return out
