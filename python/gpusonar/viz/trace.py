"""The scene trace: a scenario reduced to what a picture of it needs (docs/tasks/VISUALIZATION.md section 4).

A scene trace is a NEW file kind (JSON, `kind: gpusonar.scene_trace`), not the record format
(docs/RECORD_FORMAT.md is untouched). It holds, for the first ping of a run:

- the sonar (projector, hydrophones, band, how the samples are held), the water (depth, sound-speed profile, bottom,
  surface wind), the platform pose;
- the targets (one representative point per scene object that is not a boundary patch);
- the eigenrays the simulator's own provider returns for each leg (projector to target, array reference point to
  target), with a polyline and its bounce points where the tier allows one to be drawn (free field and flat image
  sources: exact straight segments; `eigenray`: Bellhop's own ray file, run type 'E', as a separate process);
- every two-way pair with its delay at each hydrophone, as the synthesis kernel places it (apparent positions);
- the models in force (`models_in_force`), with their source and status words;
- `chain`: reserved for the live chain's taps (beam map peaks, detections, tracks), null here.

How exact each part is, is said where it is made. Nothing here changes what the simulator computes; the trace is
built from the same provider the simulator builds (`make_provider`, wrapped by time spreading when the scenario
has `time_spread`), called again.
"""

from __future__ import annotations

import os
import subprocess
import tempfile

import numpy as np

from .. import acoustics
from ..sim import propagation as prop

TRACE_KIND = "gpusonar.scene_trace"
TRACE_VERSION = 1


# --- the provider the simulator builds ----------------------------------------------------------------------------

def provider_for(result, engine: str | None = None, base=None):
    """The eigenray provider `run_scenario` built for `result` (a ScenarioResult): the same tier, sound speed and
    options, the same time-spreading wrapper. A basin's late tail (`basin_reverberation`) is not attached: it is not
    a ray. Bellhop is run again (its cache belonged to the run) unless `base`, the provider `make_provider` returned
    for the run itself (tools/scenario_view.py keeps it), is given: then that one, with its cache, is used."""
    cfg = result.config
    tier = cfg.get("fidelity", "free_field")
    provider = base if base is not None else prop.make_provider(tier, cfg["environment"]["sound_speed"],
                                                                 cfg.get("propagation"), engine=engine)
    if cfg.get("time_spread"):
        from ..sim import timespread
        from ..sonar import SonarModel
        spread = dict(cfg["time_spread"])
        spread.setdefault("seed", int(cfg.get("seed", 0)))
        sonar = SonarModel.load(cfg["sonar"], cfg.get("sonar_overrides"))
        provider = timespread.wrap(provider, spread, frequency=sonar.center_frequency)
    return provider


def _inner(provider):
    """The provider under a time-spreading wrapper (or the provider itself)."""
    return getattr(provider, "inner", None) or getattr(provider, "base", None) or provider


# --- targets --------------------------------------------------------------------------------------------------------

BOUNDARY_ITEMS = ("seafloor", "sea_surface", "volume")     # scene items that are boundaries or a layer, not targets


def scene_targets(scene, tx_position, rx_reference, time_s: float, frequency: float, sound_speed: float,
                  max_targets: int = 6, boundary_labels=()):
    """[(label, position [3], amplitude)]: the `max_targets` strongest targets at this ping. A point target (a
    scatterer that is not a Lambert patch) is a target of its own; an object made of surface patches (a hull, a
    box, rocks) is one target, at its strongest patch for this ping. Objects whose label is in `boundary_labels`
    (seafloor, sea surface, volume layer: the reverberation's, not targets) are left out."""
    points, amps, _ = scene.evaluate(time_s, np.asarray(tx_position, float), np.asarray(rx_reference, float),
                                     frequency=frequency, sound_speed=sound_speed)
    lambert = np.asarray(scene.lambert, dtype=bool) if len(scene.lambert) == len(scene) else np.zeros(len(scene), bool)
    groups, first = {}, 0
    for label, count in zip(scene.labels, scene.counts):
        idx = np.arange(first, first + count)
        first += count
        if count == 0 or label in boundary_labels:
            continue
        for i in idx[~lambert[idx]]:
            groups[(label, int(i))] = [int(i)]
        if np.any(lambert[idx]):
            groups.setdefault((label, -1), []).extend(idx[lambert[idx]].tolist())
    out = []
    for (label, _), idx in groups.items():
        i = idx[int(np.argmax(np.abs(amps[idx])))]
        out.append((label, np.asarray(points[i], float), complex(amps[i])))
    out.sort(key=lambda t: -abs(t[2]))
    return out[:max_targets]


# --- straight-line paths: free field and flat image sources ---------------------------------------------------------

def folded_path(start, direction, length: float, surface_z: float | None, bottom_z: float | None):
    """The real path of a straight image ray: from `start` along the unit `direction` for `length` m in the
    unfolded (image) space, folded back into the water between a flat surface at `surface_z` and a flat bottom at
    `bottom_z` (either may be None). Returns (polyline [n, 3], bounces [(xyz [3], "surface" | "bottom")]).
    Exact for the flat isovelocity image construction (the image of the end point is start + length x direction)."""
    start, u = np.asarray(start, float), np.asarray(direction, float)
    end = start + length * u
    z0, z1 = start[2], end[2]
    crossings = []                                            # (fraction along the unfolded line, boundary)
    if surface_z is not None and bottom_z is not None:
        a, d = float(surface_z), float(bottom_z) - float(surface_z)
        lo, hi = sorted(((z0 - a) / d, (z1 - a) / d))
        for m in range(int(np.ceil(lo)), int(np.floor(hi)) + 1):
            zc = a + m * d
            if abs(z1 - z0) < 1e-12 or not (min(z0, z1) < zc < max(z0, z1)):
                continue
            crossings.append(((zc - z0) / (z1 - z0), "surface" if m % 2 == 0 else "bottom"))
    else:
        plane, name = (surface_z, "surface") if surface_z is not None else (bottom_z, "bottom")
        if plane is not None and min(z0, z1) < float(plane) < max(z0, z1):
            crossings.append(((float(plane) - z0) / (z1 - z0), name))
    crossings.sort()

    def fold(z):
        if surface_z is not None and bottom_z is not None:
            a, d = float(surface_z), float(bottom_z) - float(surface_z)
            t = np.mod((z - a) / d, 2.0)
            return a + d * np.where(t > 1.0, 2.0 - t, t)
        if surface_z is not None:
            return float(surface_z) + np.abs(z - float(surface_z))
        if bottom_z is not None:
            return float(bottom_z) - np.abs(float(bottom_z) - z)
        return z

    fr = [0.0] + [f for f, _ in crossings] + [1.0]
    pts = np.array([start + f * (end - start) for f in fr])
    pts[:, 2] = fold(pts[:, 2])
    bounces = [(pts[i + 1].copy(), name) for i, (_, name) in enumerate(crossings)]
    return pts, bounces


# --- Bellhop's own rays (run type 'E') ------------------------------------------------------------------------------

def read_rays(path: str):
    """Parse a Bellhop 2-D ASCII ray file: [{"alpha_deg", "n_top", "n_bottom", "rz" [n, 2] (range m, depth m)}].
    Format as Bellhop writes it (header: title, frequency, source counts, beam counts, top and bottom depths, 'rz';
    then per ray its launch angle, the step count and the boundary hits over the whole ray, and one (r, z) per step);
    read from the file itself, not from a document."""
    with open(path, "r", encoding="ascii", errors="replace") as f:
        lines = f.read().splitlines()
    i = 7
    rays = []
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        alpha = float(lines[i].split()[0])
        n, ntop, nbot = (int(v) for v in lines[i + 1].split()[:3])
        rz = np.array([[float(v) for v in lines[i + 2 + k].split()[:2]] for k in range(n)]) if n else np.zeros((0, 2))
        rays.append({"alpha_deg": alpha, "n_top": ntop, "n_bottom": nbot, "rz": rz})
        i += 2 + n
    return rays


def bellhop_rays(env, frequency: float, source_depth: float, receiver_depth: float, receiver_range: float,
                 angles=(-80.0, 80.0), n_beams: int = 0, executable: str | None = None, timeout: float = 120.0):
    """Bellhop's eigenrays (run type 'E') from a source depth to one receiver, in the provider's ocean. A separate
    process (GPL-3, never linked); None when Bellhop is not installed."""
    from ..sim import bellhop
    exe = executable or bellhop.find_bellhop()
    if exe is None:
        return None
    with tempfile.TemporaryDirectory(prefix="gpusonar_rays_") as tmp:
        bellhop.write_env(os.path.join(tmp, "run.env"), env, frequency, source_depth, [receiver_depth],
                          [max(receiver_range, 1.0)], "E", n_beams, angles)
        proc = subprocess.run([exe, "run"], cwd=tmp, capture_output=True, text=True, timeout=timeout)
        path = os.path.join(tmp, "run.ray")
        if proc.returncode != 0 or not os.path.exists(path):
            raise bellhop.BellhopError(f"Bellhop ray run failed (exit {proc.returncode})")
        return read_rays(path)


def _cut_at_range(rz: np.ndarray, r_end: float) -> np.ndarray:
    """The ray up to the first time it reaches range `r_end` (interpolated there)."""
    for k in range(1, len(rz)):
        if rz[k, 0] >= r_end:
            w = (r_end - rz[k - 1, 0]) / max(rz[k, 0] - rz[k - 1, 0], 1e-12)
            return np.vstack([rz[:k], rz[k - 1] + w * (rz[k] - rz[k - 1])])
    return rz


def _bellhop_bounces(rz: np.ndarray, env):
    """[(index, "surface" | "bottom")] of the steps on a boundary. Bellhop writes the boundary point itself, and
    often twice in a row (the step that reaches it and the reflected step that leaves it): consecutive steps on
    the same boundary are one bounce, indexed at the first."""
    out = []
    for k in range(1, len(rz) - 1):
        r, z = rz[k]
        name = "surface" if z <= 1e-6 else ("bottom" if z >= float(env.depth_at(r)) - 1e-3 else None)
        if name is None:
            continue
        if out and out[-1][1] == name and np.hypot(*(rz[k] - rz[out[-1][0]])) < 1e-3 * max(1.0, abs(r)) + 1e-3:
            continue
        out.append((k, name))
    return out


def _travel_time(rz: np.ndarray, env) -> float:
    """Travel time along a polyline in (range, depth): sum of segment length over the speed at its middle."""
    if len(rz) < 2:
        return 0.0
    d = np.diff(rz, axis=0)
    mid = 0.5 * (rz[1:] + rz[:-1])
    if env.ssp_range is None:
        c = np.array([env.speed_at(z) for z in mid[:, 1]])
    else:
        c = np.array([env.speed_at(z, r) for r, z in mid])
    return float(np.sum(np.hypot(d[:, 0], d[:, 1]) / c))


# --- per-bounce loss as the model applies it ------------------------------------------------------------------------

def bounce_loss_db(provider, boundary: str, grazing_rad: float, frequency: float | None = None) -> float | None:
    """-20 log10 |R| at one bounce, as the provider's model applies it: the image tier's surface coefficient, its
    bottom's Rayleigh coefficient (or constant); at the eigenray tier the same Rayleigh coefficient of the ocean's
    half-space (Bellhop's own reflection is not reported per bounce: this is the same formula evaluated again) and
    the rough-surface coherent loss when `wind_speed` > 0. None where no per-bounce law exists (time-spread taps,
    tilted planes)."""
    p = _inner(provider)
    if isinstance(p, prop.ImageSourceProvider) and not p.general:
        if boundary == "surface":
            r = abs(p.surface_coefficient)
        elif p.bottom is not None:
            r = abs(complex(prop.rayleigh_reflection(grazing_rad, p.sound_speed, **p.bottom)))
        else:
            r = abs(p.bottom_coefficient)
    elif getattr(p, "tier", "") in ("eigenray", "eigenray3d"):
        env = p.env
        if boundary == "surface":
            r = 1.0
            if p.wind_speed > 0.0:
                r = float(acoustics.surface_coherent_reflection(grazing_rad, p.wind_speed, frequency or p.frequency,
                                                                env.speed_at(0.0)))
        else:
            r = abs(complex(prop.rayleigh_reflection(grazing_rad, env.speed_at(env.depth), env.bottom_speed,
                                                     env.bottom_density, env.bottom_attenuation,
                                                     shear_speed=env.bottom_shear_speed or 0.0,
                                                     shear_attenuation=env.bottom_shear_attenuation or 0.0)))
    else:
        return None
    return float(-20.0 * np.log10(max(r, 1e-30)))


# --- general planes (tilted bed, {point, normal} walls) -------------------------------------------------------------

def plane_nodes(p, points, n_rays: int):
    """The plane sequences (`_PlaneImages.sequences`) behind an EigenraySet of `n_rays` columns for `points`: the
    sequences depend on the bounce order the call used, so the order whose sequence count is `n_rays` is taken (the
    stated `max_order`, else the energy rule's, which `order_used` holds after the call). None when none fits."""
    pl = p._planes
    orders = [p.max_order] if p.max_order is not None else [p.order_used or 0] + list(range(p.max_order_cap + 2))
    for order in orders:
        nodes = pl.sequences(np.atleast_2d(points), int(order), int(p.wall_order))
        if len(nodes) == n_rays:
            return nodes
    return None


def plane_path(p, seq, source, point, frequency: float | None = None):
    """The real path of one image ray among general planes: (vertices [n, 3], bounces [{xyz, boundary, grazing_deg,
    loss_db}]), traced from `source` exactly as `_PlaneImages.eigenrays` traces it (the straight line to the image of
    the point for the bounces still to come meets the next plane; the image is mirrored in it). Exact for the
    construction; NumPy, ours."""
    pl = p._planes
    src, pt = np.asarray(source, float), np.asarray(point, float)
    target = pt.copy()
    for idx in reversed(seq):                                 # the image of P in the planes met, last first
        target = pl.planes[idx].mirror(target[None])[0]
    a = src.copy()
    direction = (target - a) / max(np.linalg.norm(target - a), 1e-300)
    verts, bounces = [a.copy()], []
    for idx in seq:
        plane = pl.planes[idx]
        da, dt = float(plane.distance(a[None])[0]), float(plane.distance(target[None])[0])
        t = da / (da - dt) if da != dt else 0.0
        x = a + t * (target - a)
        cos_n = float(direction @ plane.normal)
        g = float(np.arcsin(min(abs(cos_n), 1.0)))
        r = abs(complex(np.atleast_1d(pl._reflection(plane, np.array([abs(cos_n)])))[0]))
        bounces.append({"xyz": x.tolist(), "boundary": plane.kind, "grazing_deg": float(np.degrees(g)),
                        "loss_db": float(-20.0 * np.log10(max(r, 1e-30)))})
        direction = direction - 2.0 * cos_n * plane.normal
        target = plane.mirror(target[None])[0]
        verts.append(x)
        a = x
    verts.append(pt)
    return np.array(verts), bounces


# --- the sediment tier's refracted ray ------------------------------------------------------------------------------

def sediment_path(p, source, point):
    """(vertices, crossing or None): the least-time path of the sediment tier (`snell_crossing`, NumPy): straight to
    the interface crossing, refracted to the buried point; straight for a point above the interface."""
    src, pt = np.asarray(source, float), np.asarray(point, float)
    _, cross = prop.snell_crossing(src, pt[None], p.sound_speed, p.sediment_speed, p.interface_z, engine="numpy")
    cross = cross[0]
    if pt[2] > p.interface_z and src[2] < p.interface_z:
        return np.array([src, cross, pt]), cross
    return np.array([src, pt]), None


def interface_loss_db(p, grazing_rad: float) -> float | None:
    """-20 log10 of what the sediment tier's amplitude carries per leg at the interface, besides spreading: the
    constant `transmission` (penetration constant) or V_wp / sqrt(a_rho) (fluid, rough; the class docstring)."""
    if p.penetration == "constant":
        return float(-20.0 * np.log10(max(abs(p.transmission), 1e-30)))
    b = p.bottom
    v = abs(complex(prop.fluid_transmission(grazing_rad, p.sound_speed, **b))) / np.sqrt(b["density"] / b["water_density"])
    return float(-20.0 * np.log10(max(v, 1e-30)))


# --- eigenray3d seen in 2-D -----------------------------------------------------------------------------------------

def section_bed(p, origin, target, n: int = 241):
    """[[range, depth], ...] of a 3-D bed (`p.bed`) along the vertical plane from `origin` towards `target`, out to
    the target's range (and 10 % beyond)."""
    o, t = np.asarray(origin, float), np.asarray(target, float)
    d = t[:2] - o[:2]
    rng = float(np.linalg.norm(d))
    u = d / rng if rng > 0 else np.array([1.0, 0.0])
    r = np.linspace(0.0, 1.1 * max(rng, 1.0), n)
    z = np.asarray(p.bed.depth(o[0] + r * u[0], o[1] + r * u[1]), float)
    return np.column_stack([r, z]).tolist()


def section_env(p, origin, target):
    """A 2-D Bellhop ocean on the vertical section of an `eigenray3d` provider's bed from `origin` towards `target`:
    the same water and half-space, the bed sampled along the bearing. For drawing only (a projection)."""
    import dataclasses
    bed = np.asarray(section_bed(p, origin, target), float)
    return dataclasses.replace(p.env, bathymetry=bed.tolist(), depth=float(max(bed[:, 1].max(), p.env.depth)))


# --- one leg's rays -------------------------------------------------------------------------------------------------

def _leg_rays(provider, eig, s: int, origin, target, frequency: float, with_paths: bool, points=None):
    """The valid eigenrays of field point `s` of an EigenraySet `eig` from `origin`, each with a polyline and its
    bounces where the tier allows. `points`: every field point of the call that made `eig` (general planes need them
    to name each column's plane sequence)."""
    p = _inner(provider)
    tier = getattr(p, "tier", "")
    c = float(eig.sound_speed)
    out, br, nodes = [], None, None
    horiz = np.asarray(target[:2], float) - np.asarray(origin[:2], float)
    rng = float(np.linalg.norm(horiz))
    unit = horiz / rng if rng > 0 else np.array([1.0, 0.0])
    env2 = None
    if with_paths and tier == "eigenray":
        env2 = p.env
    elif with_paths and tier == "eigenray3d":
        env2 = section_env(p, origin, target)
    if env2 is not None:
        n_beams = getattr(p, "n_beams", 0) if tier == "eigenray" else 0
        br = bellhop_rays(env2, p.frequency, float(origin[2]), float(target[2]), rng, p.angles, n_beams)
    general = isinstance(p, prop.ImageSourceProvider) and p.general
    if with_paths and general:
        nodes = plane_nodes(p, points if points is not None else np.asarray(target, float)[None], eig.delay.shape[1])
    used = set()
    cands = []                                                # each Bellhop ray once: cut at the range, its bounces
    for bi, cand in enumerate(br or []):
        rz = _cut_at_range(cand["rz"], rng)
        if len(rz) < 2 or rz[-1, 0] < rng - 1e-6:
            continue
        b = _bellhop_bounces(rz, env2)
        ns = sum(1 for _, n in b if n == "surface")
        cands.append((bi, rz, b, ns, len(b) - ns))
    for k in np.flatnonzero(eig.valid[s]):
        ray = {"k": int(k), "delay_s": float(eig.delay[s, k]), "amplitude": float(abs(eig.amplitude[s, k])),
               "n_surface": int(eig.n_surface[s, k]), "n_bottom": int(eig.n_bottom[s, k]),
               "launch_deg": float(np.degrees(np.arcsin(np.clip(eig.launch[s, k, 2], -1, 1)))),
               "arrival_deg": float(np.degrees(np.arcsin(np.clip(eig.arrival[s, k, 2], -1, 1)))),
               "polyline": None, "bounces": [], "path": "not drawn"}
        if not with_paths:
            pass
        elif tier == "free_field":
            ray.update(polyline=[list(map(float, origin)), list(map(float, target))], path="straight (exact)")
        elif isinstance(p, prop.ImageSourceProvider) and not general and not p.walls:
            pts, bounces = folded_path(origin, eig.launch[s, k], c * eig.delay[s, k], p.surface_z, p.bottom_z)
            g = float(np.arcsin(min(abs(eig.launch[s, k, 2]), 1.0)))
            ray.update(polyline=pts.tolist(), path="image, folded (exact)",
                       bounces=[{"xyz": b.tolist(), "boundary": name, "grazing_deg": float(np.degrees(g)),
                                 "loss_db": bounce_loss_db(p, name, g, frequency)} for b, name in bounces])
        elif isinstance(p, prop.ImageSourceProvider) and not general:
            ray["path"] = "not drawn: {axis, position} walls are applied once over every image (no path to draw)"
        elif general:
            if nodes is None:
                ray["path"] = "not drawn: the plane sequences could not be matched to the provider's columns"
            else:
                verts, bounces = plane_path(p, nodes[k][0], origin, target, frequency)
                length = float(np.sum(np.linalg.norm(np.diff(verts, axis=0), axis=1)))
                if abs(length - c * eig.delay[s, k]) <= 1e-6 * max(1.0, length):
                    ray.update(polyline=verts.tolist(), bounces=bounces,
                               path="image among general planes, specular points (exact)",
                               walls=sum(b["boundary"] == "wall" for b in bounces))
                else:
                    ray["path"] = f"not drawn: the traced path ({length:.3f} m) is not the image's ({c * eig.delay[s, k]:.3f} m)"
        elif tier == "sediment":
            if k == 0:
                verts, cross = sediment_path(p, origin, target)
                bounces = []
                if cross is not None:
                    d = cross - np.asarray(origin, float)
                    g = float(np.arcsin(min(abs(d[2]) / max(np.linalg.norm(d), 1e-300), 1.0)))
                    bounces.append({"xyz": cross.tolist(), "boundary": "interface", "grazing_deg": float(np.degrees(g)),
                                    "loss_db": interface_loss_db(p, g)})
                ray.update(polyline=verts.tolist(), bounces=bounces,
                           path="refracted at the interface (Snell, least time; exact for the tier)" if cross is not None
                           else "straight (above the interface)")
            else:
                ray["path"] = ("not drawn: a roughness-scattered contribution (penetration: rough, G-13), a sum over "
                               "interface cells, not a ray")
        elif br is not None:
            best, score = None, np.inf
            for bi, rz, b, ns, nb in cands:
                if ns != ray["n_surface"] or nb != ray["n_bottom"]:
                    continue
                miss = abs(rz[-1, 1] - target[2]) + 0.1 * abs(br[bi]["alpha_deg"] - ray["launch_deg"]) + (50.0 if bi in used else 0.0)
                if miss < score:
                    best, score = (bi, rz, b), miss
            if best is not None:
                bi, rz, b = best
                used.add(bi)
                xyz = np.column_stack([origin[0] + rz[:, 0] * unit[0], origin[1] + rz[:, 0] * unit[1], rz[:, 1]])
                bounces = []
                for idx, name in b:
                    d = rz[idx] - rz[idx - 1]
                    slope = 0.0
                    if name == "bottom" and env2.bathymetry is not None:
                        r0, r1 = rz[idx, 0] - 1.0, rz[idx, 0] + 1.0
                        slope = float(np.arctan2(float(env2.depth_at(r1)) - float(env2.depth_at(r0)), r1 - r0))
                    g = abs(float(np.arctan2(d[1], d[0])) - slope)
                    g = min(g, np.pi - g)
                    bounces.append({"xyz": xyz[idx].tolist(), "boundary": name, "grazing_deg": float(np.degrees(g)),
                                    "loss_db": bounce_loss_db(p, name, g, frequency)})
                ray.update(polyline=xyz.tolist(), bounces=bounces, polyline_delay_s=_travel_time(rz, env2),
                           bellhop_alpha_deg=br[bi]["alpha_deg"], end_depth_miss_m=float(abs(rz[-1, 1] - target[2])),
                           path=("Bellhop ray file (E), matched by bounces" if tier == "eigenray" else
                                 "PROJECTION: 2-D Bellhop ray on this bearing's section of the 3-D bed, matched by "
                                 "bounces; not Bellhop3D's own ray (V4)"))
            elif tier == "eigenray3d":
                ray["path"] = ("not drawn: no 2-D ray on this bearing's section has these bounce counts (the 3-D ray may "
                               "leave the vertical plane; V4 draws Bellhop3D's own rays)")
            else:
                ray["path"] = "not drawn: no Bellhop ray matched these bounce counts"
        elif tier in ("eigenray", "eigenray3d"):
            ray["path"] = "not drawn: Bellhop is not installed (GPUSONAR_BELLHOP)"
        out.append(ray)
    return out


def _channel_delays(eig, s, k, origin, elements):
    """Per-hydrophone arrival delay of a one-way ray from apparent position (reciprocity: the ray from the array's
    reference point to the source, the apparent source seen from each element), as sim/sources.py places it."""
    q = np.asarray(origin, float) + eig.sound_speed * eig.delay[s, k] * eig.launch[s, k]
    return (np.linalg.norm(q[None, :] - elements, axis=1) / eig.sound_speed).tolist()


# --- the trace ------------------------------------------------------------------------------------------------------

def build_trace(result, ping: int = 0, engine: str | None = None, max_targets: int = 6, rays: bool = True,
                provider=None, base_provider=None, view: dict | None = None, eigenrays: bool = True) -> dict:
    """The scene trace of one ping of `result` (a ScenarioResult from `run_scenario`). `view`: the page's choice of
    ping (`catalogue.view_plan`: note, track, ping index of the full run). `eigenrays=False`: the provider is never
    called (no legs, pairs, one-way or direct rays): the geometry, environment, models and beams only (`static_trace`)."""
    from ..sonar import SonarModel
    from . import beams as beams_mod
    cfg = result.config
    rec = result.records[ping]
    sonar = SonarModel.load(cfg["sonar"], cfg.get("sonar_overrides"))
    provider = provider or provider_for(result, engine, base=base_provider)
    tx = np.asarray(rec.tx_position(), float)
    elements = np.asarray(rec.rx_positions(), float)
    rx_ref = elements.mean(axis=0)
    c_rec = float(rec.environment.sound_speed)
    t_ping = float(result.times[ping]) if result.times is not None else 0.0
    boundary = {it.get("label", it.get("type")) for it in cfg.get("scene") or [] if it.get("type") in BOUNDARY_ITEMS}
    targets = scene_targets(result.scene, tx, rx_ref, t_ping, sonar.center_frequency, c_rec, max_targets, boundary)
    trace = {
        "kind": TRACE_KIND, "version": TRACE_VERSION, "scenario": cfg.get("name"),
        "tier": cfg.get("fidelity", "free_field"), "ping": ping, "time_s": t_ping,
        "sonar": _sonar_part(sonar, rec, tx, elements),
        "models": models_in_force(cfg, provider, sonar),
        "scattering": scattering_curves(cfg, sonar.center_frequency),
        "targets": [], "legs": [], "pairs": [], "oneway": [], "direct": None, "chain": None, "missing": [],
    }
    trace["environment"] = _environment_part(cfg, provider, sonar, tx, [t[1] for t in targets])
    try:
        trace["beams"] = beams_mod.beam_part(cfg.get("name"), sonar, rec, c_rec)
    except Exception as exc:                                  # noqa: BLE001  (a picture never stops a run)
        trace["beams"] = None
        trace["missing"].append(f"receive beams: not drawn ({type(exc).__name__}: {exc})")
    if view is not None:
        trace["view"] = {"note": view.get("note"), "ping_of_run": view.get("ping"), "n_pings": view.get("n_pings"),
                         "track": np.asarray(view.get("track"), float).round(3).tolist() if view.get("track") is not None else None}
    inner = _inner(provider)
    trace["time_spread_taps"] = inner is not provider
    a1m = float(sonar.amplitude_at_1m())
    if not eigenrays:
        trace["targets"] = [{"label": lb, "position": pos.tolist(), "amplitude": abs(a)} for lb, pos, a in targets]
        trace["missing"].append("eigenrays: not computed (a static trace: geometry, environment, models and beams only)")
        return trace
    if targets:
        pts = np.array([t[1] for t in targets])
        out = provider.eigenrays(tx, pts)
        back = provider.eigenrays(rx_ref, pts)
        # Time spreading (sim/timespread.py) replaces each bounce eigenray by taps about it: the pairs are the wrapper's
        # (what the simulator synthesises), the drawn rays the bounce eigenrays under it (the taps are not drawn).
        geo_out, geo_back = (out, back) if inner is provider else (inner.eigenrays(tx, pts), inner.eigenrays(rx_ref, pts))
        for s, (label, pos, amp) in enumerate(targets):
            trace["targets"].append({"label": label, "position": pos.tolist(), "amplitude": abs(amp)})
            legs_out = _leg_rays(inner, geo_out, s, tx, pos, sonar.center_frequency, rays, pts)
            legs_back = _leg_rays(inner, geo_back, s, rx_ref, pos, sonar.center_frequency, rays, pts)
            trace["legs"].append({"target": s, "out": legs_out, "back": legs_back})
            pair_out = legs_out if inner is provider else _leg_rays(provider, out, s, tx, pos, sonar.center_frequency, False)
            pair_back = legs_back if inner is provider else _leg_rays(provider, back, s, rx_ref, pos, sonar.center_frequency, False)
            for ro in pair_out:
                for rb in pair_back:
                    k, j = ro["k"], rb["k"]
                    q = tx + out.sound_speed * out.delay[s, k] * out.launch[s, k]
                    p = rx_ref + back.sound_speed * back.delay[s, j] * back.launch[s, j]
                    # As the kernel places the echo: the outbound apparent point q seen from the projector, the
                    # return apparent point p seen from each hydrophone (two_way_paths, core.synthesize_into).
                    per_channel = (np.linalg.norm(q - tx) / out.sound_speed
                                   + np.linalg.norm(p[None, :] - elements, axis=1) / back.sound_speed)
                    level = a1m * abs(amp) * ro["amplitude"] * rb["amplitude"]
                    trace["pairs"].append({
                        "target": s, "out": k, "back": j, "delay_s": ro["delay_s"] + rb["delay_s"],
                        "channel_delay_s": per_channel.tolist(),
                        "n_surface": ro["n_surface"] + rb["n_surface"], "n_bottom": ro["n_bottom"] + rb["n_bottom"],
                        "level_db": float(20.0 * np.log10(max(level, 1e-30))),
                    })
    elif cfg.get("scene"):
        trace["missing"].append("targets: none of the scene's objects returns an echo at this ping (beam pattern or range)")
    # One-way sources (sim/sources.py): the eigenrays from the array's reference point to each source, travelled
    # backwards (`one_way_paths`), at this ping's scenario time.
    if cfg.get("sources"):
        from ..sim import sources as source_models
        for src in source_models.build_sources(cfg["sources"]):
            pos, _ = src.at(t_ping)
            eig = provider.eigenrays(rx_ref, pos[None])
            geo = eig if inner is provider else inner.eigenrays(rx_ref, pos[None])
            legs = _leg_rays(inner, geo, 0, rx_ref, pos, sonar.center_frequency, rays, pos[None])
            arrivals = [{"k": int(k), "delay_s": float(eig.delay[0, k]),
                         "level_db": float(20.0 * np.log10(max(abs(eig.amplitude[0, k]), 1e-30))),
                         "n_surface": int(eig.n_surface[0, k]), "n_bottom": int(eig.n_bottom[0, k]),
                         "channel_delay_s": _channel_delays(eig, 0, k, rx_ref, elements)}
                        for k in np.flatnonzero(eig.valid[0])]
            trace["oneway"].append({"label": src.label, "position": pos.tolist(), "rays": legs, "arrivals": arrivals,
                                    "signals": [str(sig.get("type")) for sig in src.signals]})
    # The projector heard directly (transmit.direct_path): the same provider from the array's reference point to the
    # projector (or the straight line alone with boundaries: false), sim/sources.py `direct_arrival`.
    from ..sim import sources as source_models
    dp = source_models.direct_path_config(sonar)
    if dp is not None:
        dprov = provider if dp["boundaries"] else prop.make_provider("free_field", c_rec, engine="numpy")
        eig = dprov.eigenrays(rx_ref, tx[None])
        legs = _leg_rays(_inner(dprov), eig, 0, rx_ref, tx, sonar.center_frequency, rays, tx[None])
        trace["direct"] = {"boundaries": dp["boundaries"], "rays": legs,
                           "arrivals": [{"k": int(k), "delay_s": float(eig.delay[0, k]),
                                         "channel_delay_s": _channel_delays(eig, 0, k, rx_ref, elements),
                                         "n_surface": int(eig.n_surface[0, k]), "n_bottom": int(eig.n_bottom[0, k])}
                                        for k in np.flatnonzero(eig.valid[0])]}
    _note_missing(trace)
    return trace


def static_trace(name: str, quick: bool = True) -> dict:
    """A trace of a scenario without simulating it or calling its provider: the page's ping as `catalogue.view_plan`
    chooses it, an empty record of the sonar at that pose, the scene, the resolved environment, the models in force,
    the beams. Seconds for any scenario (no Bellhop, no synthesis): what `test_every_scenario_yields_a_trace` runs."""
    from .. import acoustics
    from .. import environment as site
    from ..config import deep_merge, load_scenario_config
    from ..sim.scene import build_scene
    from ..sim.simulator import ScenarioResult, make_pose
    from ..sonar import SonarModel
    from . import catalogue
    raw = load_scenario_config(name, None, expand=False)
    plan = catalogue.view_plan(raw, quick=quick)
    cfg = acoustics.resolve_sea_state(deep_merge(raw, plan["overrides"]))
    sonar = SonarModel.load(cfg["sonar"], cfg.get("sonar_overrides"))
    env = site.resolve(cfg.get("environment"), strict=True)
    cfg = site.apply_to_scenario(cfg, env, sonar)
    provider = prop.make_provider(cfg.get("fidelity", "free_field"), env["sound_speed"], cfg.get("propagation"), engine="numpy")
    plat = cfg.get("platform") or {}
    rpy = plat.get("rpy", [0.0, 0.0, 0.0])
    if plat.get("legs"):
        leg = plat["legs"][0]
        pos, rpy = leg["start"], [0.0, 0.0, np.radians(float(leg["heading_deg"]))]
    elif plat.get("raster"):
        pos = [float(plat["raster"]["x"][0]), float(plat["raster"]["y"][0]), float((plat.get("position") or [0, 0, 0])[2])]
    else:
        pos = plat.get("position", [0.0, 0.0, 0.0])
    rec = sonar.make_record(make_pose(pos, rpy, plat.get("velocity", [0.0, 0.0, 0.0])), environment=env)
    scene = build_scene(cfg.get("scene", []), np.random.default_rng(int(cfg.get("seed", 0))))
    result = ScenarioResult([rec], scene, {**cfg, "environment": env}, np.asarray([pos], float), times=np.zeros(1))
    return build_trace(result, engine="numpy", base_provider=provider, view=plan, eigenrays=False)


def _note_missing(trace: dict) -> None:
    """Add to `trace["missing"]` one line per reason a ray was not drawn, with its count: a page says what it lacks."""
    reasons = {}
    for leg in trace["legs"]:
        for which in ("out", "back"):
            for r in leg[which]:
                if r["polyline"] is None:
                    reasons[r["path"]] = reasons.get(r["path"], 0) + 1
    for group in trace["oneway"] + ([trace["direct"]] if trace["direct"] else []):
        for r in group["rays"]:
            if r["polyline"] is None:
                reasons[r["path"]] = reasons.get(r["path"], 0) + 1
    for why, n in sorted(reasons.items()):
        trace["missing"].append(f"{n} eigenray(s) {why}")
    if trace["time_spread_taps"]:
        trace["missing"].append("time-spread taps: drawn as the bounce eigenrays under them, the taps' own spread is not "
                                "drawn (V3)")
    if trace["tier"] == "eigenray3d":
        trace["missing"].append("eigenray3d: the section shows a 2-D projection on each target's bearing; Bellhop3D's "
                                "own 3-D rays are V4")


def _sonar_part(sonar, rec, tx, elements) -> dict:
    s = rec.samples
    w = sonar.waveform_config
    return {"name": sonar.config.get("name"), "mode": sonar.config.get("mode"),
            "band_hz": [float(w.get("f_start", sonar.center_frequency)), float(w.get("f_stop", sonar.center_frequency))],
            "center_frequency": float(sonar.center_frequency), "sample_kind": str(s.kind).split(".")[-1],
            "sample_rate": float(s.sample_rate), "start_time_s": float(s.start_time), "n_samples": int(s.n_samples),
            "n_channels": int(s.n_channels), "projector": tx.tolist(), "hydrophones": elements.tolist(),
            "source_level_db_rms": sonar.config["transmit"]["waveform"].get("source_level_db_rms"),
            "projector_kind": sonar.config["transmit"].get("kind", "point")}


def _environment_part(cfg, provider, sonar=None, tx=None, target_points=()) -> dict:
    p = _inner(provider)
    env = cfg["environment"]
    fc = float(sonar.center_frequency) if sonar is not None else None
    out = {"sound_speed": float(env["sound_speed"]), "surface_z": 0.0, "bottom_z": None, "ssp": None,
           "bathymetry": None, "bottom": env.get("bottom"), "wind_speed": env.get("wind_speed"),
           "absorption": cfg.get("absorption", "francois_garrison"), "ssp_section": None, "interface": None,
           "surface": None, "critical_angle_deg": None, "bathymetry_note": None}
    if isinstance(p, prop.ImageSourceProvider):
        # The bottom the tier reflects from: its half-space, or its constant coefficient (never the site's row
        # when the scenario overrides it).
        bottom = None if p.bottom_z is None else (dict(p.bottom) if p.bottom is not None
                                                  else {"coefficient": abs(p.bottom_coefficient)})
        out.update(surface_z=p.surface_z, bottom_z=p.bottom_z, bottom=bottom)
        if p.general:
            out["planes"] = [{"kind": pl.kind, "point": pl.point.tolist(), "normal": pl.normal.tolist(),
                              "bounds": pl.bounds, "coefficient": None if pl.coefficient is None else abs(complex(pl.coefficient))}
                             for pl in p._planes.planes]
            out["bottom_slope"] = list(p.bottom_slope) if p.bottom_slope is not None else None
    elif getattr(p, "tier", "") in ("eigenray", "eigenray3d"):
        z, c = p.env.profile()
        out.update(bottom_z=float(p.env.depth), ssp={"z": z.tolist(), "c": c.tolist()},
                   bathymetry=p.env.bathymetry,
                   bottom={"speed": p.env.bottom_speed, "density": p.env.bottom_density,
                           "attenuation": p.env.bottom_attenuation, "shear_speed": p.env.bottom_shear_speed,
                           "sediment": getattr(p.env, "sediment_name", None)},
                   wind_speed_propagation=float(getattr(p, "wind_speed", 0.0) or 0.0))
        sec = p.env.section()
        if sec is not None:
            r, zz, cc = sec
            out["ssp_section"] = {"r": r.tolist(), "z": zz.tolist(), "c": cc.tolist()}
        if p.tier == "eigenray3d" and tx is not None and len(target_points):
            out["bathymetry"] = section_bed(p, tx, target_points[0])
            out["bottom_z"] = float(max(b[1] for b in out["bathymetry"]))
            out["bathymetry_note"] = ("the 3-D bed (`bathymetry`) cut along the first target's bearing: a section of a 3-D "
                                      "relief, which rays may leave")
        elif p.env.bathymetry is not None:
            out["bathymetry_note"] = "a bed that changes with range from the sonar: the same section on every bearing"
    elif getattr(p, "tier", "") == "sediment":
        out.update(bottom_z=float(p.interface_z))
        b = p.bottom or {}
        crit = None
        if p.sediment_speed > p.sound_speed:
            crit = float(np.degrees(np.arccos(p.sound_speed / p.sediment_speed)))
        out["interface"] = {"z": float(p.interface_z), "sediment_speed": float(p.sediment_speed),
                            "penetration": p.penetration, "subcritical": p.subcritical, "bottom": dict(b) if b else None,
                            "transmission": None if p.penetration != "constant" else abs(p.transmission),
                            "critical_deg": crit, "rough": p.roughness}
        out["bottom"] = dict(b) if b else {"speed": float(p.sediment_speed)}
    else:
        out["bottom"] = None                                      # free field: no boundary at all
    if out["bottom_z"] is None and env.get("site_water_depth"):
        out["bottom_z_site"] = float(env["site_water_depth"])     # the site's depth; the tier has no bottom
    b = out.get("bottom")
    if isinstance(b, dict) and b.get("speed") and float(b["speed"]) > out["sound_speed"] and out["interface"] is None:
        out["critical_angle_deg"] = float(np.degrees(np.arccos(out["sound_speed"] / float(b["speed"]))))
    # The surface's roughness and bubble loss where the scenario sets them (time spreading's surface section).
    ts = (cfg.get("time_spread") or {}).get("surface")
    if ts:
        from ..sim import timespread
        try:
            par = timespread.boundary_parameters(dict(ts), surface=True, frequency=fc)
        except Exception:                                     # noqa: BLE001
            par = None
        if par:
            bl = par.get("bubble_loss")
            out["surface"] = {"rms_height": float(par["rms_height"]), "mean_square_slope": float(par["mean_square_slope"]),
                              "wind_speed": ts.get("wind_speed"), "given": sorted(ts),
                              "bubble_loss": None if bl is None else {"grazing_deg": np.degrees(bl[0]).tolist(),
                                                                       "loss_db": np.asarray(bl[1]).tolist()}}
    if fc is not None:
        from .. import acoustics
        from .. import environment as site
        try:
            a = float(np.asarray(acoustics.absorption(fc, *site.water(env), out["absorption"])))
            out["absorption_db_per_km"] = 1000.0 * a
        except Exception:                                     # noqa: BLE001
            out["absorption_db_per_km"] = None
    noise = (cfg.get("receiver") or {}).get("ambient_noise")
    out["noise"] = {k: v for k, v in (noise or {}).items() if k != "file"} or None
    return out


# --- what is modelled here ------------------------------------------------------------------------------------------
# Sources and status words as docs/handoff/SIMULATION.md section 3 states them (that table is the authority; a row
# here repeats it, shortened). "default" means the scenario left the choice to the code's default.

_SOURCES = {
    "free_field": ("direct path, spherical spreading", "standard result", "ours"),
    "image_source": ("flat-boundary isovelocity images; bounce order by energy (D-105)", "TR 0702 Sec. 5.4; energy rule", "confirmed; ours"),
    "sediment": ("one refracted ray into a fluid sediment", "Snell; J&R (2007) through the notes", "confirmed"),
    "eigenray": ("Bellhop 2-D eigenrays, looked up on a range-depth grid", "Bellhop (GPL-3, separate process), not re-validated; lookup ours", "ours"),
    "eigenray3d": ("Bellhop3D 3-D eigenrays, merged and scaled to its TL run", "Bellhop3D (GPL-3, separate process); merge ours", "ours"),
    "surface_pressure_release": ("surface reflection coefficient -1", "pressure release", "ours"),
    "rayleigh_fluid": ("Rayleigh fluid half-space reflection", "standard; sediment table via secondary (REFERENCES 3.1)", "via secondary"),
    "rayleigh_elastic": ("elastic half-space reflection", "J&R (2007) (9.34)-(9.40); shear values unconfirmed", "confirmed"),
    "bottom_constant": ("constant bottom reflection coefficient", "the scenario's value", "ours"),
    "rough_surface_coherent": ("coherent loss exp(-Gamma^2/2) per surface bounce", "standard result, not checked against print", "ours"),
    "time_spread": ("time/angle/Doppler spreading of bounce paths", "TR 0702 eqs. 89-103; facet model ours; Dahl 1996 (A7) slope", "confirmed; ours"),
    "lambert": ("Lambert bottom scattering mu sin sin", "the law and -29 dB as a thesis quotes them", "via secondary"),
    "fluid_sediment": ("rough fluid sediment scattering", "perturbation/volume ours = published forms; small-slope validity confirmed", "ours"),
    "williams_jackson_1998": ("Williams-Jackson (1998) bistatic model (opt-in)", "Williams and Jackson 1998", "confirmed"),
    "mourad_jackson_1989": ("Mourad-Jackson (1989) model (opt-in)", "Mourad and Jackson 1989", "confirmed"),
    "searay_bottom": ("empirical bottom law by type", "sim/surface.py", "unconfirmed"),
    "chapman_harris": ("Chapman-Harris surface scattering", "McDaniel 1993 eq. (27)", "via secondary"),
    "kraus_surface": ("Kraus surface scattering", "via a secondary print", "via secondary"),
    "bubble_layer_surface": ("bubble-layer surface model", "a dissertation printing the model's terms; dahl_1997 extinction (Table II)", "via secondary"),
    "volume": ("volume layer as a sheet", "TR 0702 Sec. 6.5", "confirmed"),
    "reverb_generator": ("scattering-function reverberation", "TR 0702 Sec. 10-11 (idea); generator ours", "confirmed; ours"),
    "k_statistics": ("K-distributed reverberation", "Abraham and Lyons 2002 eqs. (15), (19)", "confirmed"),
    "wenz": ("ambient noise from the Wenz fit", "three prints of the curves", "via secondary"),
    "noise_level": ("ambient noise at a stated level", "the scenario's value", "ours"),
    "noise_fitted": ("ambient noise fitted to a recording", "sim/noise_fit.py", "ours"),
    "francois_garrison": ("absorption, Francois-Garrison", "within 0.6 % of 21 published values", "via secondary"),
    "ainslie_mccolm": ("absorption, Ainslie-McColm", "within 3 % of 21 published values", "via secondary"),
    "thorp": ("absorption, Thorp", "TR 0702 eq. 42", "confirmed"),
    "none": ("no absorption", "-", "ours"),
    "mackenzie": ("sound speed from temperature and salinity, Mackenzie", "held to arlpy 1.9.3; paper not opened", "unconfirmed"),
    "ssp_profile": ("sound-speed profile in depth", "the environment's profile", "ours"),
    "ssp_range": ("water that changes with range (rings about the sonar)", "Bellhop 'Q' field", "ours"),
    "bathymetry": ("a bed that changes with range (one section on every bearing)", "Bellhop bed file", "ours"),
    "bathymetry3d": ("a 3-D bed (sim/bathymetry.py), relief across and along the look", "Bellhop3D bed grid ('R'); relief illustrative", "ours"),
    "penetration": ("sediment penetration", "J&R (2007) (8.18)...(15.16) through the notes; split ours", "confirmed; ours"),
    "basin": ("a basin's late reverberation tail", "fitted to one data set", "ours"),
    "direct_overload": ("projector direct sound / front-end overload", "closed forms", "ours"),
    "oneway_sources": ("radiating sources heard one way", "sim/sources.py", "ours"),
    "bistatic": ("bistatic reverberation geometry", "TR 0702 Sec. 6.4.2 (geometric mean); geometry ours", "confirmed; ours"),
    "targets": ("point scatterers and highlight targets", "TR 0702 Sec. 9.2 (highlights); shapes ours", "confirmed; ours"),
    "general_planes": ("images among general planes: a tilted bed, {point, normal} walls (opt-in, sonar-0nsi)", "PHYSICS_NOTES section 30", "ours"),
    "rough_surface_height": ("the surface's rms height and slope (time spreading's surface)", "wind laws of sim/timespread.py; Dahl 1996 (A7) slope", "confirmed; ours"),
    "bubble_loss": ("bubble loss per surface bounce", "PN 20.4; apl_1994 and hall_novarini via secondary; a stated number ours", "via secondary; ours"),
    "direct_path": ("the projector's sound heard directly (transmit.direct_path)", "sim/sources.py, PN 20.1, 22", "ours"),
    "overload": ("front-end overload and recovery (receive.overload)", "sim/receiver.py, PN 22", "ours"),
    "noise_statistics": ("non-Gaussian ambient noise statistics", "sim/noise.py; K: Abraham and Lyons 2002", "confirmed; ours"),
    "scene_seafloor": ("a seafloor patch of point scatterers in the scene (its own scattering law)", "Lambert as a thesis quotes it; fluid sediment: published forms", "via secondary; ours"),
    "scene_sea_surface": ("a sea-surface patch of point scatterers (wave facets) in the scene", "sim/scene.py; facets ours", "ours"),
    "relief": ("relief and penetration of the reverberation bed", "sim/reverb_relief.py", "ours"),
    "specular_exclusion": ("specular exclusion in the reverberation", "sim/reverb_specular.py", "ours"),
}


def _short_value(v):
    """A bed description in a few words: its kind and its numbers (no long tables)."""
    if isinstance(v, dict):
        return {k: (_short_value(x) if isinstance(x, dict) else x) for k, x in v.items() if not isinstance(x, (list, tuple)) or len(x) <= 4}
    return v


def _entry(area, model, key, value, chosen="scenario"):
    what, source, status = _SOURCES[model]
    return {"area": area, "model": model, "what": what, "key": key, "value": value, "chosen": chosen,
            "source": source, "status": status}


def models_in_force(cfg: dict, provider=None, sonar=None) -> list[dict]:
    """Every model a scenario's run uses, as a list of {area, model, what, key, value, chosen, source, status}.
    Read from the resolved config (and the provider, for what the tier does with the bottom). A model the scenario
    does not use is never listed."""
    out = []
    tier = cfg.get("fidelity", "free_field")
    out.append(_entry("propagation", tier, "fidelity", tier, "scenario" if "fidelity" in cfg else "default"))
    prop_cfg = cfg.get("propagation") or {}
    p = _inner(provider) if provider is not None else None
    if tier in ("image_source", "eigenray", "eigenray3d") and (tier != "image_source" or prop_cfg.get("surface_z", 0.0) is not None):
        out.append(_entry("surface", "surface_pressure_release", "propagation.surface_coefficient", -1.0, "default"))
    if tier == "image_source" and (prop_cfg.get("bottom_slope") is not None or any("normal" in w for w in prop_cfg.get("walls") or [])):
        out.append(_entry("propagation", "general_planes", "propagation.bottom_slope/walls",
                          {"bottom_slope": prop_cfg.get("bottom_slope"), "walls": len(prop_cfg.get("walls") or []),
                           "wall_order": prop_cfg.get("wall_order", 1)}))
    if tier == "image_source":
        b = prop_cfg.get("bottom")
        if prop_cfg.get("bottom_z") is not None:
            if b is not None:
                el = float(b.get("shear_speed", 0.0) or 0.0) > 0.0 or (b.get("sediment") == "rock" and b.get("shear_speed", None) != 0)
                out.append(_entry("bottom", "rayleigh_elastic" if el else "rayleigh_fluid", "propagation.bottom", b))
            else:
                out.append(_entry("bottom", "bottom_constant", "propagation.bottom_coefficient", prop_cfg.get("bottom_coefficient")))
    if tier in ("eigenray", "eigenray3d"):
        e = prop_cfg.get("environment") or {}
        el = bool(p is not None and getattr(p, "env", None) is not None and (p.env.bottom_shear_speed or 0) > 0)
        out.append(_entry("bottom", "rayleigh_elastic" if el else "rayleigh_fluid", "propagation.environment.bottom",
                          {k: e.get(k) for k in ("sediment", "bottom_speed", "bottom_density", "bottom_attenuation") if e.get(k) is not None}))
        if float(prop_cfg.get("wind_speed", 0.0) or 0.0) > 0.0:
            out.append(_entry("surface", "rough_surface_coherent", "propagation.wind_speed", prop_cfg["wind_speed"]))
        if e.get("ssp_depth") is not None:
            out.append(_entry("water", "ssp_profile", "propagation.environment.ssp_depth/ssp_speed", f"{len(e['ssp_depth'])} points"))
        if e.get("ssp_range") is not None:
            out.append(_entry("water", "ssp_range", "propagation.environment.ssp_range", e["ssp_range"]))
        if tier == "eigenray3d" and (prop_cfg.get("bathymetry") is not None or cfg.get("bathymetry") is not None):
            b3 = prop_cfg.get("bathymetry") or cfg.get("bathymetry")
            out.append(_entry("bottom", "bathymetry3d", "bathymetry", _short_value(b3)))
        elif e.get("bathymetry") is not None or prop_cfg.get("bathymetry") is not None or cfg.get("bathymetry") is not None:
            out.append(_entry("bottom", "bathymetry", "propagation.environment.bathymetry", "given"))
    if tier == "sediment":
        pen = prop_cfg.get("penetration", "constant")
        val = {"penetration": pen, "interface_z": prop_cfg.get("interface_z")}
        if pen != "constant":
            val.update(bottom=prop_cfg.get("bottom"), subcritical=prop_cfg.get("subcritical", "evanescent"))
        if pen == "rough":
            val["roughness"] = prop_cfg.get("roughness")
        out.append(_entry("bottom", "penetration", "propagation.penetration", val,
                          "scenario" if "penetration" in prop_cfg else "default"))
    if cfg.get("time_spread"):
        out.append(_entry("boundaries", "time_spread", "time_spread", sorted(cfg["time_spread"])))
        surf = cfg["time_spread"].get("surface") or {}
        hkeys = {k: surf[k] for k in ("wind_speed", "rms_height", "mean_square_slope", "height_scale", "slope_law") if k in surf}
        if hkeys:
            out.append(_entry("surface", "rough_surface_height", "time_spread.surface", hkeys))
        if surf.get("bubble_loss_db") not in (None, 0, 0.0):
            out.append(_entry("surface", "bubble_loss", "time_spread.surface.bubble_loss_db", surf["bubble_loss_db"]))
    for i, item in enumerate(cfg.get("reverberation") or []):
        model = item.get("model", "lambert")
        key = f"reverberation[{i}]"
        if model == "fluid_sediment":
            rm = item.get("roughness_model", "small_slope")
            vm = item.get("volume_model", "stockhausen")
            out.append(_entry("bottom", "fluid_sediment", key, {"roughness_model": rm, "volume_model": vm,
                                                                "sediment": item.get("sediment")},
                              "scenario" if "roughness_model" in item else "default"))
            for opt in ("williams_jackson_1998", "mourad_jackson_1989"):
                if opt in (rm, vm):
                    out.append(_entry("bottom", opt, key, opt))
        elif model == "lambert":
            out.append(_entry("bottom" if float(item.get("boundary_z", 1.0) or 1.0) > 0 else "surface", "lambert", key,
                              {"mu_db": item.get("mu_db", item.get("sediment", "default"))}))
        elif model in _SOURCES:
            extra = {"wind_speed": item.get("wind_speed")}
            if model == "bubble_layer_surface":
                extra["extinction"] = item.get("extinction", "dahl_1997")
            out.append(_entry("surface" if model in ("chapman_harris", "kraus_surface", "bubble_layer_surface") else
                              ("water" if model == "volume" else "bottom"), model, key, extra))
        if item.get("relief") or item.get("bathymetry") is not None:
            out.append(_entry("reverberation", "relief", key + ".relief", item.get("relief", "bathymetry")))
        if item.get("exclude_specular"):
            out.append(_entry("reverberation", "specular_exclusion", key + ".exclude_specular", item["exclude_specular"]))
        if item.get("geometry") == "bistatic":
            out.append(_entry("reverberation", "bistatic", key + ".geometry", "bistatic"))
        stats = (item.get("statistics") or {}).get("model")
        if stats == "k":
            out.append(_entry("reverberation", "k_statistics", key + ".statistics", item["statistics"]))
    if cfg.get("reverberation"):
        out.append(_entry("reverberation", "reverb_generator", "reverberation", f"{len(cfg['reverberation'])} item(s)"))
    noise = (cfg.get("receiver") or {}).get("ambient_noise")
    if noise:
        m = noise.get("model", "level_db" if "level_db" in noise else "wenz")
        out.append(_entry("water", {"wenz": "wenz", "fitted": "noise_fitted"}.get(m, "noise_level"),
                          "receiver.ambient_noise", {k: v for k, v in noise.items() if k not in ("file", "statistics")}))
        if noise.get("statistics"):
            out.append(_entry("water", "noise_statistics", "receiver.ambient_noise.statistics", noise["statistics"]))
    absorption = cfg.get("absorption", "francois_garrison")
    out.append(_entry("water", absorption if absorption in _SOURCES else "francois_garrison", "absorption", absorption,
                      "scenario" if "absorption" in cfg else "default"))
    out.append(_entry("water", "mackenzie", "environment.temperature/salinity",
                      {k: cfg["environment"].get(k) for k in ("temperature", "salinity")}, "default"))
    if cfg.get("sources"):
        out.append(_entry("sources", "oneway_sources", "sources", f"{len(cfg['sources'])} source(s)"))
    if cfg.get("basin_reverberation"):
        out.append(_entry("reverberation", "basin", "basin_reverberation", "given"))
    if sonar is not None:
        if sonar.config["transmit"].get("direct_path"):
            out.append(_entry("sources", "direct_path", "sonar transmit.direct_path", sonar.config["transmit"]["direct_path"]))
        if sonar.config.get("receive", {}).get("overload") is not None:
            out.append(_entry("receiver", "overload", "sonar receive.overload", sorted(sonar.config["receive"]["overload"])))
    for i, item in enumerate(cfg.get("scene") or []):
        if item.get("type") == "seafloor":
            law = item.get("model", "lambert")
            out.append(_entry("bottom", "scene_seafloor", f"scene[{i}] ({item.get('label', 'seafloor')})",
                              {"model": law, **({"lambert_mu_db": item["lambert_mu_db"]} if "lambert_mu_db" in item else {}),
                               **({"slope": item["slope"]} if "slope" in item else {})}))
        elif item.get("type") == "sea_surface":
            out.append(_entry("surface", "scene_sea_surface", f"scene[{i}] ({item.get('label', 'sea_surface')})",
                              {k: item[k] for k in ("mu_db", "isotropic_db", "wind_speed", "roughness") if k in item}))
    out.append(_entry("targets", "targets", "scene", f"{len(cfg.get('scene') or [])} item(s)"))
    return out


def scattering_curves(cfg: dict, frequency: float, n: int = 90) -> list[dict]:
    """Monostatic scattering strength against grazing angle of each `reverberation` item, from the law the simulator
    itself builds (simulator.reverberation_item_strength): [{key, model, grazing_deg, strength_db}]. Lambert items
    are mu sin^2. An item whose law cannot be evaluated alone (`exclude_specular` needs the provider) is skipped."""
    from ..sim.simulator import reverberation_item_strength
    out = []
    g = np.radians(np.linspace(0.5, 90.0, n))
    for i, item in enumerate(cfg.get("reverberation") or []):
        if item.get("exclude_specular"):
            continue
        try:
            filled, strength, offset = reverberation_item_strength(dict(item), frequency, cfg["environment"])
            if strength is None:
                s = 10.0 ** ((float(filled["mu_db"]) + offset) / 10.0) * np.sin(g) ** 2
            else:
                s = np.asarray(strength(g, g), dtype=float)
        except Exception:                                   # noqa: BLE001  (a picture never stops a run)
            continue
        out.append({"key": f"reverberation[{i}]", "model": item.get("model", "lambert"),
                    "grazing_deg": np.degrees(g).round(2).tolist(),
                    "strength_db": (10.0 * np.log10(np.maximum(s, 1e-30))).round(2).tolist()})
    return out


def write_trace(trace: dict, path: str) -> None:
    import json
    with open(path, "w", encoding="utf-8") as f:
        json.dump(trace, f, separators=(",", ":"))
