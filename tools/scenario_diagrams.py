#!/usr/bin/env python3
"""A geometry diagram for every scenario, drawn from its YAML file.

For each scenario this builds the scene exactly as the simulator does and draws

  plan view   the scatterers by object, the ownship track over all pings, the receive
              window and horizontal beam at the first ping, and where moving objects go;
  side view   the same along the boresight: depth against distance, the vertical beam,
              the sea surface and the seabed where the scenario has them;
  for Bellhop scenarios, the sound-speed profile and the one-way transmission loss from
  the sonar's depth (incoherent sum of Bellhop's arrivals), with the target marked.

    tools/scenario_diagrams.py                 # all scenarios -> docs/scenarios/<name>.png
    tools/scenario_diagrams.py --only popoto_12km,fls_170k_traffic

Nothing here is typed in: change the YAML and the diagram follows.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Wedge  # noqa: E402

from gpusonar.config import list_configs, load_scenario_config  # noqa: E402
from gpusonar.sim import build_scene, find_bellhop, make_provider  # noqa: E402
from gpusonar.sonar import SonarModel  # noqa: E402

# Categorical colours in a fixed order (colour-blind checked); objects beyond the list are grey.
COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8b5fd6", "#d6457a", "#4aa3b8", "#8a6a3b"]
INK, MUTED, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#d9d8d2"
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
                     "font.size": 9})


def ownship_track(cfg):
    """Ownship position [n, 3] and heading (rad) [n] at every ping, and the ping times."""
    plat, pings = cfg.get("platform", {}), cfg.get("pings", {})
    interval = float(pings.get("interval", 1.0))
    if "legs" in plat:
        pos, head = [], []
        step = float(plat["speed"]) * interval
        for leg in plat["legs"]:
            h = np.radians(float(leg["heading_deg"]))
            n = int(leg["pings"]) if "pings" in leg else int(float(leg["length"]) / step) + 1
            for i in range(n):
                pos.append(np.asarray(leg["start"], dtype=float) + i * step * np.array([np.cos(h), np.sin(h), 0.0]))
                head.append(h)
        pos, head = np.array(pos), np.array(head)
        return pos, head, np.arange(len(pos)) * interval
    p0 = np.asarray(plat.get("position", [0.0, 0.0, 0.0]), dtype=float)
    yaw = float(plat.get("rpy", [0.0, 0.0, 0.0])[2])
    if "raster" in plat:
        xs, ys = np.arange(*plat["raster"]["x"]), np.arange(*plat["raster"]["y"])
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        pos = p0 + np.column_stack([X.ravel(), Y.ravel(), np.zeros(X.size)])
        return pos, np.full(len(pos), yaw), np.arange(len(pos)) * interval
    n = int(pings.get("count", 1))
    t = np.arange(n) * interval
    v = np.asarray(plat.get("velocity", [0.0, 0.0, 0.0]), dtype=float)
    return p0 + t[:, None] * v, np.full(n, yaw), t


def boundaries(cfg):
    """Depths (z, down) of the sea surface and seabed the scenario states, or None."""
    env, prop = cfg.get("environment", {}), cfg.get("propagation", {})
    bed = env.get("water_depth") or prop.get("environment", {}).get("depth") or prop.get("bottom_z")
    surface = 0.0 if (bed is not None or cfg.get("fidelity") == "eigenray") else None
    for b in cfg.get("reverberation", []):
        if b.get("boundary_z") is None:          # boundaries now come from the environment (water depth), as above
            continue
        z = float(b["boundary_z"])
        if z <= 0.0 and surface is None:
            surface = z
        elif z > 0.0 and bed is None:
            bed = z
    return surface, (None if bed is None else float(bed))


def transmission_loss(cfg, source_depth):
    """One-way incoherent transmission loss [depth, range] from Bellhop's arrivals, in dB."""
    provider = make_provider("eigenray", float(cfg["environment"]["sound_speed"]), cfg["propagation"])
    arr = provider.arrivals_for(source_depth)
    tl = np.full((len(provider.depths), len(provider.ranges)), np.nan)
    for iz in range(len(provider.depths)):
        for ir in range(len(provider.ranges)):
            a = arr.amplitude[iz][ir]
            if len(a):
                tl[iz, ir] = -10.0 * np.log10(max(float(np.sum(np.abs(a) ** 2)), 1e-30))
    return provider, tl


def draw(name, path):
    cfg = load_scenario_config(name)
    sonar = SonarModel.load(cfg["sonar"], cfg.get("sonar_overrides"))
    scene = build_scene(cfg["scene"], np.random.default_rng(int(cfg.get("seed", 0))))
    pos, head, times = ownship_track(cfg)
    win = sonar.config["receive"].get("window", {}).get("range")
    bore = np.asarray(sonar.boresight, dtype=float)
    az0 = head[0] + np.arctan2(bore[1], bore[0])                    # boresight azimuth in the world, first ping
    el0 = np.arcsin(np.clip(bore[2] / np.linalg.norm(bore), -1, 1))  # positive looks down
    half_az = min(0.5 * (sonar.beamwidth_az or np.pi), np.radians(75.0)) if np.isfinite(sonar.beamwidth_az or np.inf) else np.radians(75.0)
    bw_el = sonar.beamwidth_el if (sonar.beamwidth_el and np.isfinite(sonar.beamwidth_el)) else None
    pts = scene.positions
    r_far = float(win[1]) if win else float(np.linalg.norm(pts - pos[0], axis=1).max())
    r_near = float(win[0]) if win else 0.0
    surface, bed = boundaries(cfg)
    eigen = cfg.get("fidelity") == "eigenray"
    info = {}
    use_km = r_far > 2000.0
    k = 1e-3 if use_km else 1.0
    unit = "km" if use_km else "m"

    ncols = 3 if eigen else 2
    fig, axes = plt.subplots(1, ncols, figsize=(6.2 * ncols if not eigen else 17.5, 5.6), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1.0, 1.25, 0.42]} if eigen else None)
    plan, side = axes[0], axes[1]

    # --- plan view ---------------------------------------------------------------------------
    masks = scene.label_masks()
    rng = np.random.default_rng(0)
    handles = []
    vel = scene.velocities if hasattr(scene, "velocities") else np.zeros_like(pts)
    for i, (label, mask) in enumerate(masks.items()):
        idx = np.flatnonzero(mask)
        col = COLOURS[i] if i < len(COLOURS) else MUTED
        show = idx if len(idx) <= 2500 else rng.choice(idx, 2500, replace=False)
        size = 16 if len(idx) < 20 else (5 if len(idx) < 400 else 1.5)
        h = plan.scatter(pts[show, 1] * k, pts[show, 0] * k, s=size, color=col, linewidths=0, label=label, zorder=3)
        if i < 10:
            handles.append(h)
        v = vel[idx].mean(axis=0)
        if np.hypot(v[0], v[1]) > 1e-6 and times[-1] > 0:
            c = pts[idx].mean(axis=0)
            plan.annotate("", xy=((c[1] + v[1] * times[-1]) * k, (c[0] + v[0] * times[-1]) * k), xytext=(c[1] * k, c[0] * k),
                          arrowprops=dict(arrowstyle="-|>", color=col, lw=1.8), zorder=5)
            plan.annotate(f"{np.hypot(v[0], v[1]):.1f} m/s", (c[1] * k, c[0] * k), textcoords="offset points", xytext=(6, 6),
                          fontsize=8, color=INK)
    theta = 90.0 - np.degrees(az0)
    raster = "raster" in cfg.get("platform", {})
    if r_far > 0 and "legs" not in cfg.get("platform", {}) and abs(bore[2]) < 0.9 * np.linalg.norm(bore):
        plan.add_patch(Wedge((pos[0, 1] * k, pos[0, 0] * k), r_far * k, theta - np.degrees(half_az), theta + np.degrees(half_az),
                             width=(r_far - r_near) * k, fc="#2a78d6", alpha=0.07, ec="#2a78d6", lw=0.8, zorder=1))
    steps = np.linalg.norm(np.diff(pos[:, :2], axis=0), axis=1)
    breaks = np.flatnonzero(steps > 3.0 * max(np.median(steps), 1e-9)) + 1 if len(steps) and "raster" not in cfg.get("platform", {}) else []
    for leg in np.split(np.arange(len(pos)), breaks):               # separate survey legs are separate lines
        plan.plot(pos[leg, 1] * k, pos[leg, 0] * k, color=INK, lw=0 if raster else 1.6, marker="." if raster else None, ms=2, zorder=6)
        if len(leg) > 1 and len(breaks):
            plan.annotate("", xy=(pos[leg[-1], 1] * k, pos[leg[-1], 0] * k), xytext=(pos[leg[-2], 1] * k, pos[leg[-2], 0] * k),
                          arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.6), zorder=6)
            hd = head[leg[0]] + np.arctan2(bore[1], bore[0])
            g0, g1 = np.sqrt(max(r_near ** 2 - pos[leg[0], 2] ** 2, 0.0)), np.sqrt(max(r_far ** 2 - pos[leg[0], 2] ** 2, 0.0))
            look = np.array([np.cos(hd), np.sin(hd)])
            quad = [pos[leg[0], :2] + look * g0, pos[leg[-1], :2] + look * g0, pos[leg[-1], :2] + look * g1, pos[leg[0], :2] + look * g1]
            plan.add_patch(plt.Polygon([(q[1] * k, q[0] * k) for q in quad], fc="#2a78d6", alpha=0.06, ec="#2a78d6", lw=0.8, zorder=1))
    plan.plot(pos[0, 1] * k, pos[0, 0] * k, marker="^", color=INK, ms=7, ls="none", zorder=7)
    moved = float(np.linalg.norm(pos[-1] - pos[0]))
    plan.annotate((f"sonar at {len(pos)} raster positions, looking down" if raster else
                   f"ownship, {len(pos)} pings" + (f" over {moved:.3g} m" if moved > 0 else ", not moving")),
                  (pos[0, 1] * k, pos[0, 0] * k), textcoords="offset points", xytext=(8, -12), fontsize=8)
    plan.set_aspect("equal", adjustable="datalim")
    plan.set_xlabel(f"y, to starboard ({unit})")
    plan.set_ylabel(f"x, ahead ({unit})")
    span = f"{r_near * k:.4g} to {r_far * k:.4g} {unit}" if win else "no range window"
    plan.set_title(f"Plan: receive window {span}, shaded where the beam reaches", loc="left", fontsize=9)
    plan.legend(handles=handles, loc="best", fontsize=7, markerscale=2.5 if len(pts) > 400 else 1.0, framealpha=0.9)

    # --- side view along the boresight -------------------------------------------------------
    d = np.array([np.cos(az0), np.sin(az0)])
    along = (pts[:, :2] - pos[0, :2]) @ d
    cross = (pts[:, :2] - pos[0, :2]) @ np.array([-d[1], d[0]])
    near_plane = np.abs(cross) <= np.maximum(np.tan(half_az) * np.abs(along), 0.02 * max(r_far, 1.0))
    if half_az < 0.2 or "raster" in cfg.get("platform", {}):
        near_plane = np.ones(len(pts), dtype=bool)                  # a narrow beam swept along a track sees everything
    if eigen and find_bellhop() is not None:
        provider, tl = transmission_loss(cfg, float(pos[0, 2]))
        lo = np.nanpercentile(tl, 2)
        m = side.imshow(tl, extent=[provider.ranges[0] * k, provider.ranges[-1] * k, provider.depths[-1], provider.depths[0]],
                        aspect="auto", cmap="Blues_r", vmin=lo, vmax=lo + 60.0, interpolation="nearest", zorder=0)
        fig.colorbar(m, ax=side, label="one-way transmission loss (dB), incoherent sum of arrivals", shrink=0.85)
        side.grid(False)
        target = pts.mean(axis=0)
        rt = float(np.hypot(*(target[:2] - pos[0, :2])))
        it, iz = np.abs(provider.ranges - rt).argmin(), np.abs(provider.depths - target[2]).argmin()
        info = {"frequency_hz": provider.frequency, "source_depth_m": float(pos[0, 2]), "target_range_m": rt,
                "target_depth_m": float(target[2]), "one_way_loss_db": float(tl[iz, it]),
                "spherical_spreading_db": float(20 * np.log10(rt)), "water_depth_m": float(provider.env.depth)}
        side.plot(rt * k, target[2], marker="o", mfc="none", mec="#eb6834", mew=2, ms=11, ls="none", zorder=5)
        side.annotate(f"target: {tl[iz, it]:.1f} dB one way\n(spherical spreading alone: {20 * np.log10(rt):.1f} dB)",
                      (rt * k, target[2]), textcoords="offset points", xytext=(-190, 34 if target[2] > 0.5 * provider.env.depth else -46), fontsize=8,
                      bbox=dict(fc=SURFACE, ec=GRID, alpha=0.9))
        side.plot(provider.ranges[0] * k, pos[0, 2], marker="^", color="#eb6834", ms=8, ls="none", zorder=5)
        side.set_ylim(provider.env.depth, 0.0)
        side.set_xlabel(f"range ({unit})")
        side.set_title(f"Bellhop at {provider.frequency / 1e3:.3g} kHz from the sonar at {pos[0, 2]:.0f} m depth", loc="left", fontsize=9)
    else:
        for i, (label, mask) in enumerate(masks.items()):
            idx = np.flatnonzero(mask & near_plane)
            if not len(idx):
                continue
            show = idx if len(idx) <= 2500 else rng.choice(idx, 2500, replace=False)
            side.scatter(along[show] * k, pts[show, 2], s=16 if len(idx) < 20 else (5 if len(idx) < 400 else 1.5),
                         color=COLOURS[i] if i < len(COLOURS) else MUTED, linewidths=0, zorder=3)
        z0 = pos[0, 2]
        if bw_el:
            rr = np.array([0.0, r_far])
            for e in (el0 - bw_el / 2, el0 + bw_el / 2):
                side.plot(rr * np.cos(e) * k, z0 + rr * np.sin(e), color="#2a78d6", lw=1.0, zorder=2)
            side.fill_between(rr * k, z0 + rr * np.tan(el0 - bw_el / 2), z0 + rr * np.tan(el0 + bw_el / 2), color="#2a78d6",
                              alpha=0.07, zorder=1)
        side.plot(0.0, z0, marker="^", color=INK, ms=7, ls="none", zorder=6)
        zs = np.r_[pts[near_plane, 2] if near_plane.any() else pts[:, 2], z0]
        z_lo, z_hi = zs.min(), zs.max()
        if surface is not None:
            side.axhline(surface, color="#4aa3b8", lw=1.4)
            side.annotate("sea surface", (r_far * k, surface), textcoords="offset points", xytext=(-4, 4), ha="right", fontsize=8)
            z_lo = min(z_lo, surface)
        if bed is not None:
            side.axhline(bed, color="#8a6a3b", lw=1.4)
            side.annotate("seabed", (r_far * k, bed), textcoords="offset points", xytext=(-4, 4), ha="right", fontsize=8)
            z_hi = max(z_hi, bed)
        pad = 0.08 * max(z_hi - z_lo, 1e-3)
        side.set_ylim(z_hi + pad, z_lo - pad)
        side.set_xlim(min(along[near_plane].min() if near_plane.any() else 0.0, 0.0) * k - 0.02 * r_far * k, 1.03 * r_far * k)
        side.set_xlabel(f"distance along the boresight ({unit})")
        beam = f"vertical beam {np.degrees(bw_el):.3g} deg, boresight {-np.degrees(el0) + 0.0:+.3g} deg (up positive)" if bw_el else "no vertical beam limit"
        side.set_title(f"Side: {beam} (scales differ)", loc="left", fontsize=9)
    side.set_ylabel("z, depth (m); down is positive")

    # --- sound-speed profile -----------------------------------------------------------------
    if eigen:
        from gpusonar.sim.bellhop import OceanEnvironment
        env = OceanEnvironment(**cfg["propagation"]["environment"])
        z, c = env.profile()
        ax = axes[2]
        ax.plot(c, z, color="#2a78d6", lw=2)
        ax.plot(c, z, "o", color="#2a78d6", ms=4)
        ax.set_ylim(env.depth, 0.0)
        ax.margins(x=0.3)
        ax.set_xlabel("sound speed (m/s)")
        ax.set_ylabel("depth (m)")
        ax.set_title("Profile and bottom", loc="left", fontsize=9)
        ax.text(0.04, 0.03, f"bottom: {env.bottom_speed:.0f} m/s\n{env.bottom_density:.2g} g/cm3\n{env.bottom_attenuation:.2g} dB/wavelength",
                transform=ax.transAxes, fontsize=8, va="bottom", bbox=dict(fc=SURFACE, ec=GRID))
        ax.ticklabel_format(useOffset=False)

    wf = sonar.config["transmit"]["waveform"]
    f0, f1 = float(wf["f_start"]), float(wf.get("f_stop", wf["f_start"]))
    fig.suptitle(f"{name}: {cfg['sonar']} ({len(sonar.positions)} elements, {min(f0, f1) / 1e3:.4g}-{max(f0, f1) / 1e3:.4g} kHz), "
                 f"{cfg.get('fidelity', 'free_field')}, {len(pts)} scatterers", fontsize=10)
    fig.savefig(path, dpi=105)
    plt.close(fig)
    return info


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", default="")
    ap.add_argument("--out", default=str(ROOT / "docs" / "scenarios"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    names = [n for n in args.only.split(",") if n] or sorted(list_configs("scenarios"))
    import json
    loss_path = out / "transmission_loss.json"
    losses = json.loads(loss_path.read_text()) if loss_path.exists() else {}
    for name in names:
        info = draw(name, out / f"{name}.png")
        if info:
            losses[name] = info
            loss_path.write_text(json.dumps(dict(sorted(losses.items())), indent=2))
        print(f"[{name}] {out / (name + '.png')} {info or ''}")


if __name__ == "__main__":
    main()
