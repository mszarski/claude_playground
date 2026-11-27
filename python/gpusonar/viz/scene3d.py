"""What the 3-D page needs beyond the scene trace (docs/tasks/VISUALIZATION.md, package V4): Bellhop3D's own 3-D rays
for the `eigenray3d` tier, the bed as a surface (a 3-D bed sampled on a grid), and the targets' and the platform's
positions over the pings. Read only: nothing here changes the simulator, a provider, a default or a record.

Bellhop3D's rays. The provider (`sim/bellhop3d.py`) runs Bellhop3D for arrivals ('a') and TL ('I') only. Bellhop3D
writes a ray file only for run type 'R' (Bellhop/bellhop3D.f90: `WriteRay3D` is called when RunType(1:1) == 'R';
'E' writes nothing in 3-D), one ray per launch pair (declination alpha, azimuth beta) of its fan. So the 3-D rays are
found here by shooting, as a separate process exactly as the provider runs it (GPL-3, never linked): the same water,
half-space and bed grid (`write_env3d`, `write_bty3d` and the provider's own `grid_step`, nodes on multiples of it from
the source), a small fan about each eigenray's launch declination (the trace's `launch_deg`) and azimuth (the
eigenray's own when the trace has it, else a band about the target's bearing), refined twice about the best ray.
A ray is matched to an eigenray when its surface and bottom bounces before its closest approach to the target equal
the eigenray's; the miss distance and the travel time along the drawn ray are kept beside the provider's delay, so a
page can say how close the drawn ray is to the one the provider summed. Ours; not a provider change.

The ray file format, read from the Acoustics Toolbox source (Bellhop/ReadEnvironmentBell.f90 opens it and writes the
header: title, frequency, NSx NSy NSz, Nalpha Nbeta, top depth, bottom depth, 'xyz'; Bellhop/WriteRay.f90 WriteRay3D
writes per ray its declination in RADIANS (`Angles%alpha`, already converted), then the point count and the surface
and bottom bounce counts over the whole ray, then x y z per point in metres in the run's frame: the source at the
origin, x along bearing 0 (north here), y along bearing 90 (east)). A reflection is written as two equal consecutive
points (the step that lands on the boundary and the reflected step leaving it): `ray_bounces` counts those, and
equals Bellhop3D's own counts on every one of 697 rays over the seamount (measured 2026-11-27, ours).
"""

from __future__ import annotations

import os
import subprocess
import tempfile

import numpy as np


# --- Bellhop3D ray files --------------------------------------------------------------------------------------------

def read_rays3d(path: str) -> dict:
    """A Bellhop3D ray file ('xyz'): {title, frequency, n_alpha, n_beta, top, bottom, rays: [{alpha_deg, n_top,
    n_bottom, xyz [n, 3]}]} (format: module docstring)."""
    with open(path, "r", encoding="ascii", errors="replace") as f:
        lines = f.read().splitlines()
    if len(lines) < 7 or "xyz" not in lines[6]:
        raise ValueError(f"{path}: not a 3-D ray file (line 7 is {lines[6] if len(lines) > 6 else None!r}, not 'xyz')")
    head = {"title": lines[0].strip().strip("'").strip(), "frequency": float(lines[1].split()[0]),
            "n_alpha": int(lines[3].split()[0]), "n_beta": int(lines[3].split()[1]),
            "top": float(lines[4].split()[0]), "bottom": float(lines[5].split()[0])}
    rays, i = [], 7
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        alpha = float(lines[i].split()[0])
        n, n_top, n_bot = (int(v) for v in lines[i + 1].split()[:3])
        xyz = (np.array(" ".join(lines[i + 2:i + 2 + n]).split(), dtype=float).reshape(n, 3) if n else np.zeros((0, 3)))
        rays.append({"alpha_deg": float(np.degrees(alpha)), "n_top": n_top, "n_bottom": n_bot, "xyz": xyz})
        i += 2 + n
    head["rays"] = rays
    return head


def ray_bounces(xyz: np.ndarray, top: float = 0.0, tol: float = 1e-6) -> list[tuple[int, str]]:
    """[(index, "surface" | "bottom")] of a 3-D ray's reflections: two equal consecutive points (module docstring),
    indexed at the first; "surface" when on the top depth `top`."""
    if len(xyz) < 2:
        return []
    d = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    return [(int(i), "surface" if abs(xyz[i, 2] - top) < 1e-3 else "bottom") for i in np.flatnonzero(d < tol)]


def closest_approach(xyz: np.ndarray, point) -> tuple[float, int, np.ndarray]:
    """(distance, segment index i, the nearest point on segment i..i+1) of a polyline to `point`."""
    p = np.asarray(point, float)
    if len(xyz) < 2:
        return float(np.linalg.norm(xyz[0] - p)) if len(xyz) else np.inf, 0, (xyz[0] if len(xyz) else p)
    a, b = xyz[:-1], xyz[1:]
    ab = b - a
    ll = np.einsum("ij,ij->i", ab, ab)
    t = np.clip(np.einsum("ij,ij->i", p - a, ab) / np.where(ll > 0, ll, 1.0), 0.0, 1.0)
    q = a + t[:, None] * ab
    dist = np.linalg.norm(q - p, axis=1)
    i = int(np.argmin(dist))
    return float(dist[i]), i, q[i]


def cut_ray(xyz: np.ndarray, point):
    """(polyline up to its closest approach to `point`, miss distance, bounces on it [(index, boundary)])."""
    miss, i, q = closest_approach(xyz, point)
    line = np.vstack([xyz[:i + 1], q[None]])
    return line, miss, [(k, b) for k, b in ray_bounces(xyz) if k <= i]


def travel_time(xyz: np.ndarray, speed_at) -> float:
    """Travel time along a 3-D polyline: each segment's length over the speed at its middle depth."""
    if len(xyz) < 2:
        return 0.0
    seg = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    zm = 0.5 * (xyz[1:, 2] + xyz[:-1, 2])
    c = np.array([float(speed_at(z)) for z in zm])
    return float(np.sum(seg / c))


def _vec(values) -> str:
    return " ".join(f"{v:.5f}" for v in values) + " /"


def run_rays3d(p, origin, target, alphas, betas, executable: str | None = None, timeout: float = 600.0,
               margin: float | None = None) -> dict:
    """Bellhop3D run type 'R' from `origin` (the provider's source key) for the fan alphas x betas (degrees), with the
    provider's water, half-space and bed on its grid step, in a box that reaches just past `target`. Returns
    `read_rays3d` with each ray's xyz moved into the scene's frame (metres, NED) and its launch azimuth `beta_deg`
    (from the order Bellhop3D writes: azimuth outer, declination inner, both sorted)."""
    from ..sim import bellhop3d as b3
    exe = executable or b3.find_bellhop3d()
    if exe is None:
        raise b3.Bellhop3DError("Bellhop3D is not installed: set GPUSONAR_BELLHOP3D to bellhop3d.exe")
    key = np.asarray(p._key(np.asarray(origin, float)), float)
    rel = np.asarray(target, float)[:2] - key[:2]
    rng = float(np.hypot(*rel))
    m = margin if margin is not None else max(300.0, 0.12 * rng)
    lo_x, hi_x = min(0.0, rel[0]) - m, max(0.0, rel[0]) + m
    lo_y, hi_y = min(0.0, rel[1]) - m, max(0.0, rel[1]) + m
    step = p.grid_step if p.bed.flat_depth is None else max(p.grid_step, 500.0)
    gx = np.arange(np.floor(lo_x / step), np.ceil(hi_x / step) + 1) * step
    gy = np.arange(np.floor(lo_y / step), np.ceil(hi_y / step) + 1) * step
    bed = lambda x, y: p.bed.depth(x + key[0], y + key[1])                     # noqa: E731
    deepest = float(max(p.env.depth, np.max(bed(gx[None, :], gy[:, None]))))
    z, c = p.env.profile()
    if deepest > z[-1]:
        z, c = np.r_[z, deepest], np.r_[c, c[-1]]
    bottom = (p.env.bottom_speed, p.env.bottom_density, p.env.bottom_attenuation)
    if p.env.bottom_shear_speed:
        bottom += (p.env.bottom_shear_speed, p.env.bottom_shear_attenuation or 0.0)
    alphas = np.unique(np.round(np.clip(np.asarray(alphas, float), -89.0, 89.0), 5))
    betas = np.unique(np.round(np.asarray(betas, float), 5))
    box = (max(abs(lo_x), abs(hi_x)) / 1000.0, max(abs(lo_y), abs(hi_y)) / 1000.0)
    with tempfile.TemporaryDirectory(prefix="gpusonar_rays3d_") as tmp:
        b3.write_bty3d(os.path.join(tmp, "run.bty"), bed, gx, gy)
        b3.write_env3d(os.path.join(tmp, "run.env"), frequency=p.frequency, profile=(z, c), depth=deepest, bottom=bottom,
                       source_depth=float(key[2]), receiver_depths=[float(target[2])], ranges=[max(rng, 1.0)],
                       bearings_deg=[float(np.degrees(np.arctan2(rel[1], rel[0])))], run_type="R", mode="3",
                       n_alpha=len(alphas), alpha_deg=(alphas[0], alphas[-1]), n_beta=len(betas),
                       beta_deg=(betas[0], betas[-1]), box_km=box, beam="G", title="gpusonar rays3d")
        # The fan as explicit lists (Bellhop reads every value given and sorts them, misc angleMod ReadRay*Angles).
        path = os.path.join(tmp, "run.env")
        lines = open(path, encoding="ascii").read().splitlines()
        i = next(k for k, ln in enumerate(lines) if ln.startswith("'RG"))
        lines[i + 2], lines[i + 4] = _vec(alphas), _vec(betas)
        with open(path, "w", encoding="ascii") as f:
            f.write("\n".join(lines) + "\n")
        proc = subprocess.run([exe, "run"], cwd=tmp, capture_output=True, text=True, timeout=timeout)
        ray_path = os.path.join(tmp, "run.ray")
        if proc.returncode != 0 or not os.path.exists(ray_path):
            raise b3.Bellhop3DError(f"Bellhop3D ray run failed (exit {proc.returncode})")
        out = read_rays3d(ray_path)
    for n, r in enumerate(out["rays"]):
        r["beta_deg"] = float(betas[n // len(alphas)]) if n // len(alphas) < len(betas) else None
        r["xyz"] = r["xyz"] + np.array([key[0], key[1], 0.0])
    out["key"] = key.tolist()
    return out


def _grazing(xyz: np.ndarray, i: int) -> float:
    """Grazing angle (rad) of the reflection written at points i, i+1: the incoming and outgoing steps' normal is
    their difference (specular)."""
    if i < 1 or i + 2 >= len(xyz):
        return float("nan")
    a, b = xyz[i] - xyz[i - 1], xyz[i + 2] - xyz[i + 1]
    a, b = a / max(np.linalg.norm(a), 1e-300), b / max(np.linalg.norm(b), 1e-300)
    n = b - a
    if np.linalg.norm(n) < 1e-12:
        return 0.0
    n = n / np.linalg.norm(n)
    return float(np.arcsin(min(abs(a @ n), 1.0)))


def bellhop3d_leg(p, origin, target, rays: list[dict], frequency: float | None = None, refine: int = 2,
                  executable: str | None = None) -> dict:
    """Bellhop3D's own 3-D ray for each eigenray of one leg (`rays`: the trace's ray dicts of that leg, with
    `n_surface`, `n_bottom`, `launch_deg` and, when known, `launch_azimuth_deg`). Returns {k: {polyline, bounces,
    miss_m, travel_time_s, alpha_deg, beta_deg} | None} and the runs' count. Shooting as in the module docstring."""
    from .trace import bounce_loss_db
    origin, target = np.asarray(origin, float), np.asarray(target, float)
    rel = target[:2] - origin[:2]
    bearing = float(np.degrees(np.arctan2(rel[1], rel[0])))
    want = [r for r in rays if r.get("n_surface") is not None]
    if not want:
        return {"rays": {}, "runs": 0}
    alphas = np.concatenate([r["launch_deg"] + np.array([-0.4, -0.2, 0.0, 0.2, 0.4]) for r in want])
    az = [r.get("launch_azimuth_deg") for r in want]
    if all(a is not None for a in az):
        betas = np.concatenate([a + np.array([-0.6, -0.3, 0.0, 0.3, 0.6]) for a in az])
    else:
        betas = bearing + np.arange(-6.0, 6.001, 0.4)
    runs = 1
    first = run_rays3d(p, origin, target, alphas, betas, executable)
    best = {}

    def consider(r, cand):
        if cand["n_top"] is None:
            return
        line, miss, bnc = cut_ray(cand["xyz"], target)
        ns = sum(1 for _, b in bnc if b == "surface")
        if ns != r["n_surface"] or len(bnc) - ns != r["n_bottom"] or abs(cand["alpha_deg"] - r["launch_deg"]) > 1.5:
            return
        if r["k"] not in best or miss < best[r["k"]][0]:
            best[r["k"]] = (miss, cand, line, bnc)

    for r in want:
        for cand in first["rays"]:
            consider(r, cand)
    h_a, h_b = 0.1, 0.12
    for _ in range(refine):
        for r in want:
            if r["k"] not in best:
                continue
            c0 = best[r["k"]][1]
            if c0["beta_deg"] is None:
                continue
            res = run_rays3d(p, origin, target, c0["alpha_deg"] + h_a * np.arange(-2, 3),
                             c0["beta_deg"] + h_b * np.arange(-2, 3), executable)
            runs += 1
            for cand in res["rays"]:
                consider(r, cand)
        h_a, h_b = h_a / 4.0, h_b / 4.0
    out = {}
    env = p.env
    for r in want:
        if r["k"] not in best:
            out[r["k"]] = None
            continue
        miss, cand, line, bnc = best[r["k"]]
        bounces = []
        for i, b in bnc:
            g = _grazing(cand["xyz"], i)
            bounces.append({"xyz": cand["xyz"][i].tolist(), "boundary": b, "grazing_deg": float(np.degrees(g)),
                            "loss_db": None if not np.isfinite(g) else bounce_loss_db(p, b, g, frequency)})
        out[r["k"]] = {"polyline": line.tolist(), "bounces": bounces, "miss_m": miss,
                       "travel_time_s": travel_time(line, env.speed_at), "alpha_deg": cand["alpha_deg"],
                       "beta_deg": cand["beta_deg"], "n_points": int(len(line))}
    return {"rays": out, "runs": runs}


def add_bellhop3d_rays(trace: dict, p, max_rays: int = 8, executable: str | None = None) -> dict:
    """Put Bellhop3D's own 3-D ray under each drawn eigenray of an `eigenray3d` trace (`ray["bellhop3d"]`: polyline,
    bounces, miss_m, travel_time_s, alpha_deg, beta_deg; or None when no ray of these bounces came near the target).
    The `max_rays` strongest eigenrays of each leg (those the page draws). Returns {legs, rays, matched, runs}."""
    hyd = np.asarray(trace["sonar"]["hydrophones"], float)
    origins = {"out": np.asarray(trace["sonar"]["projector"], float), "back": hyd.mean(axis=0)}
    n = matched = runs = 0
    for leg in trace["legs"]:
        target = np.asarray(trace["targets"][leg["target"]]["position"], float)
        for which in ("out", "back"):
            rays = sorted(leg[which], key=lambda r: -r["amplitude"])[:max_rays]
            res = bellhop3d_leg(p, origins[which], target, rays, trace["sonar"]["center_frequency"],
                                executable=executable)
            runs += res["runs"]
            for r in rays:
                r["bellhop3d"] = res["rays"].get(r["k"])
                n += 1
                matched += r["bellhop3d"] is not None
    trace.setdefault("scene3d", {})["bellhop3d"] = {"rays": n, "matched": matched, "runs": runs}
    return trace["scene3d"]["bellhop3d"]


# --- the bed as a surface, motion over the pings --------------------------------------------------------------------

def extents(trace: dict, margin: float = 0.15) -> tuple[float, float, float, float]:
    """(x0, x1, y0, y1) of the scene: projector, hydrophones, targets, the platform's track, one-way sources, with a
    margin (a fraction of the larger side, at least 1 m)."""
    pts = [trace["sonar"]["projector"], *trace["sonar"]["hydrophones"], *[t["position"] for t in trace["targets"]],
           *[o["position"] for o in trace.get("oneway") or []]]
    track = (trace.get("view") or {}).get("track")
    if track:
        pts += list(track)
    a = np.asarray(pts, float)
    x0, x1, y0, y1 = a[:, 0].min(), a[:, 0].max(), a[:, 1].min(), a[:, 1].max()
    m = max(margin * max(x1 - x0, y1 - y0), 1.0)
    return float(x0 - m), float(x1 + m), float(y0 - m), float(y1 + m)


def bed_grid(p, box, n: int = 49) -> dict:
    """The 3-D bed of an `eigenray3d` provider on an n x n grid over `box` (x0, x1, y0, y1): {x, y, z [ny][nx]}."""
    x = np.linspace(box[0], box[1], n)
    y = np.linspace(box[2], box[3], n)
    z = np.asarray(p.bed.depth(x[None, :], y[:, None]), float)
    return {"x": np.round(x, 2).tolist(), "y": np.round(y, 2).tolist(), "z": np.round(z, 2).tolist()}


def motion(trace: dict, cfg: dict, scene=None, max_pings: int = 120) -> dict:
    """The targets' positions over the run's pings (first-order motion: the traced position plus the label's mean
    velocity times the time from the traced ping; a time-varying sea surface's shape is not followed) and the
    platform's track: {times [P], targets: [per trace target: [[x, y, z] per ping]], track, ping (the traced one), moving: bool}."""
    view = trace.get("view") or {}
    n_pings = int(view.get("n_pings") or 1)
    interval = float((cfg.get("pings") or {}).get("interval", 1.0))
    stride = max(1, int(np.ceil(n_pings / max_pings)))
    idx = list(range(0, n_pings, stride))
    times = np.asarray(idx, float) * interval
    t0 = float(view.get("ping_of_run") or 0) * interval
    vel = {}
    if scene is not None and len(scene):
        first = 0
        for label, count in zip(scene.labels, scene.counts):
            if count:
                vel.setdefault(label, []).append(scene.velocities[first:first + count].mean(axis=0))
            first += count
    out, moving = [], False
    for t in trace["targets"]:
        v = np.mean(vel[t["label"]], axis=0) if t["label"] in vel else np.zeros(3)
        moving |= bool(np.linalg.norm(v) > 0.0)
        pos = np.asarray(t["position"], float)[None, :] + (times - t0)[:, None] * v[None, :]
        out.append(np.round(pos, 2).tolist())
    track = view.get("track")
    return {"times": np.round(times, 4).tolist(), "ping_index": idx, "targets": out, "moving": moving,
            "track": None if track is None else np.round(np.asarray(track, float)[idx], 2).tolist(),
            "ping": int(view.get("ping_of_run") or 0), "interval": interval}


def resolved(name: str, quick: bool = True):
    """(resolved config, provider (numpy; nothing run), scene) of a scenario at the page's ping, as static_trace
    resolves it (`trace.static_trace`)."""
    from .. import acoustics
    from .. import environment as site
    from ..config import deep_merge, load_scenario_config
    from ..sim import propagation as prop
    from ..sim.scene import build_scene
    from ..sonar import SonarModel
    from . import catalogue
    raw = load_scenario_config(name, None, expand=False)
    plan = catalogue.view_plan(raw, quick=quick)
    cfg = acoustics.resolve_sea_state(deep_merge(raw, plan["overrides"]))
    sonar = SonarModel.load(cfg["sonar"], cfg.get("sonar_overrides"))
    env = site.resolve(cfg.get("environment"), strict=True)
    cfg = site.apply_to_scenario(cfg, env, sonar)
    provider = prop.make_provider(cfg.get("fidelity", "free_field"), env["sound_speed"], cfg.get("propagation"), engine="numpy")
    scene = build_scene(cfg.get("scene", []), np.random.default_rng(int(cfg.get("seed", 0))))
    return {**cfg, "environment": env}, provider, scene


def enrich(trace: dict, name: str | None = None, bellhop3d: bool = True, executable: str | None = None) -> dict:
    """Add `scene3d` to a trace (fresh or read back from a file): {box, bed_grid (eigenray3d), motion, bellhop3d
    (summary), notes}. Bellhop3D's rays are added only for `eigenray3d` and when `bellhop3d` and the program is found;
    otherwise the reason is in `notes` and the page keeps the 2-D projection, labelled as such."""
    from .trace import _inner
    name = name or trace["scenario"]
    cfg, provider, scene = resolved(name)
    p = _inner(provider)
    s3 = trace.setdefault("scene3d", {})
    s3["box"] = extents(trace)
    s3["notes"] = []
    s3["motion"] = motion(trace, cfg, scene)
    if trace["tier"] == "eigenray3d":
        s3["bed_grid"] = bed_grid(p, s3["box"])
        if not bellhop3d:
            s3["notes"].append("Bellhop3D's own rays not traced for this page (--no-bellhop3d): the rays are the 2-D projection")
        else:
            from ..sim import bellhop3d as b3
            if (executable or b3.find_bellhop3d()) is None:
                s3["notes"].append("Bellhop3D is not installed (GPUSONAR_BELLHOP3D): the rays are the 2-D projection")
            else:
                add_bellhop3d_rays(trace, p, executable=executable)
    return trace
