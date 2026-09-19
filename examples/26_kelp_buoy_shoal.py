"""Three more scenarios at ``examples/21``'s settings: kelp, a moored buoy, a shoal.

As ``examples/23``: the same sonar, environment and picture, one thing added
at a time, the bare picture beside it under the same window, and the
difference between them.

* **a kelp forest** over part of the picture.  Giant kelp stands from
  holdfasts on the bottom to the surface, and what a sonar sees of it is not
  the stipes but the gas-filled pneumatocysts along the fronds -- bladders
  of 2-3 cm, well above their resonance at 120 kHz, so each scatters as a
  rigid body of its size, about -33 dB.  A plant carries hundreds.  The
  forest here is 77 plants at 4 m spacing over a 45 x 30 m stand at 140 m
  on the port bow, each plant a column of 8 point scatterers from the
  bottom to the surface, each standing for forty bladders (-17 dB): -8 dB
  a plant, and a stand that fills the water column and reads as a cloud
  rather than a target.  A forest is not transparent, either: the fronds
  and blades in front of a point take from the sound on the way in and out,
  so each point is attenuated by 0.4 dB a metre, each way, of stand between
  it and the sonar.  The front of the stand is bright, and 45 m in the
  sound is 36 dB down: a leading edge that fades, which is how a forest
  reads.  (What the stand takes from the reverberation beyond it is not
  modelled; a hard occluder is the wrong tool for a thing that is
  translucent.)
* **a buoy moored with a chain.**  A steel buoy of 0.75 m radius at the
  surface (a rigid sphere, ``sigma = a^2/4``, -8.5 dB at every aspect), its
  chain a catenary 45 m long from the buoy to a concrete sinker 30 m along
  the bottom -- 32 mm stud link, five 20 cm links a metre of -20 dB each,
  sampled as 200 point scatterers along the curve -- and the sinker itself,
  a 0.8 m concrete block on the seabed off the mesh.  The buoy is a point;
  the chain is the line under it that gives a moored buoy away, and whether
  it shows depends on how it lies.  A range cell is 0.22 m deep and a beam
  wide: a chain running along the line of sight puts one link in a cell, a
  chain lying across it puts a beam's width of links in one, 8 m of chain
  at this range, 25 dB more.  This one lies across, as a chain in a
  cross-current does.
* **a shoal**: a second school of fish, unlike 23's.  Small pelagic fish,
  800 of them at -43 dB (12 cm), packed in a 12 x 10 x 4 m ball at 8 m
  depth, 100 m away on the port bow: -14 dB incoherently in a body a
  single beam wide, a compact bright blob where 23's loose school was a
  diffuse patch.

Every picture stays differentiable in what was put into it: the forest's
position, the buoy's, the shoal's and its fish's strength.

Acceptance criteria:
  * the kelp stand reads well above what was in its cells;
  * the buoy is a point well above its surroundings, and the chain's line
    reads above the bare picture along its length;
  * the shoal reads well above what was in its cells;
  * each picture carries gradients to its scenario's parameters.

Float32, as ``examples/23``.  21's switches carry through.
"""

from __future__ import annotations

import importlib.util
import math
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from _common import banner, check, save, setup, timed
from hydropt import (
    CurvedSurfaceScattering, ExtendedTarget, IsotropicScattering, LambertScattering,
    add_receiver_noise, azimuth_steering, beam_noise_power, beam_power_scale, beamform,
    box_mesh, calibrate, fish_school, line_array_directivity_db, make_time_grid,
    reverberation_arrivals, shading_window, target_arrivals, trace,
)
from hydropt.beamform import ArrivalSet
from hydropt.mesh import boat_hull_mesh, mesh_target

SCENARIO = os.environ.get("HYDROPT_SCENARIO", "all")
if SCENARIO not in ("all", "kelp", "buoy", "shoal"):
    raise SystemExit(f"HYDROPT_SCENARIO must be all, kelp, buoy or shoal, got {SCENARIO!r}")
SCENARIOS = ("kelp", "buoy", "shoal") if SCENARIO == "all" else (SCENARIO,)
CAPTION = {"kelp": "a kelp forest on the port bow",
           "buoy": "a buoy moored with a chain",
           "shoal": "a shoal of small fish, packed"}

# the kelp stand
KELP_RANGE, KELP_BEARING_DEG = 140.0, 32.0
KELP_STAND = (45.0, 30.0)         # along the line of sight, across it
KELP_SPACING = 4.0                # plants
KELP_POINTS = 8                   # per plant, bottom to surface
KELP_POINT_DB = -17.0             # forty -33 dB bladders each
KELP_EXTINCTION_DB_PER_M = 0.4    # each way, through the stand
# the buoy
BUOY_RANGE, BUOY_BEARING_DEG = 120.0, 8.0
BUOY_RADIUS = 0.75
CHAIN_LENGTH, CHAIN_SCOPE = 45.0, 30.0    # metres of chain, horizontal span to the sinker
CHAIN_LINK_DB, LINKS_PER_M, CHAIN_POINTS = -20.0, 5.0, 200    # 32 mm stud link
CHAIN_DIRECTION_DEG = BUOY_BEARING_DEG + 90.0   # across the line of sight: a cross-current
SINKER = 0.8                      # a concrete block, metres
# the shoal
SHOAL_RANGE, SHOAL_BEARING_DEG, SHOAL_DEPTH = 100.0, 35.0, 8.0
SHOAL_RADII = (6.0, 5.0, 2.0)
N_SHOAL, SHOAL_TS_DB = 800, -43.0
LEGS = {"kelp": 6, "buoy": 24, "shoal": 8}   # paths per leg per point: many points, fewer paths


def _ex21():
    path = Path(__file__).resolve().parent / "21_long_range_300m.py"
    spec = importlib.util.spec_from_file_location("_ex21", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def polar(r, bearing_deg, depth=0.0):
    b = math.radians(bearing_deg)
    return (r * math.cos(b), r * math.sin(b), depth)


def catenary(length: float, span: float, drop: float, n: int) -> torch.Tensor:
    """``[n, 3]`` points along a chain of ``length`` from (0, 0, 0) to (span, 0, drop).

    A hanging chain is a catenary; its parameter is found from the length and
    the chord by bisection.  ``z`` is down.
    """
    chord = math.hypot(span, drop)
    if length <= chord:
        t = torch.linspace(0.0, 1.0, n)
        return torch.stack([t * span, torch.zeros(n), t * drop], -1)
    # solve sqrt(L^2 - drop^2) = 2 a sinh(span / (2 a)) for a
    target = math.sqrt(length ** 2 - drop ** 2)
    lo, hi = 1e-3, 1e4
    for _ in range(200):
        a = 0.5 * (lo + hi)
        if 2 * a * math.sinh(span / (2 * a)) > target:
            lo = a
        else:
            hi = a
    a = 0.5 * (lo + hi)
    # the curve y = a cosh((x - x0)/a) + c through both ends
    x0 = span / 2 - a * math.asinh(drop / (2 * a * math.sinh(span / (2 * a))))
    c = -a * math.cosh(-x0 / a)
    x = torch.linspace(0.0, span, n)
    z = a * torch.cosh((x - x0) / a) + c
    return torch.stack([x, torch.zeros(n), z], -1)


def main() -> int:
    setup(double=False)
    banner("26 -- kelp, a moored buoy and a shoal, at 21's settings: " + ", ".join(SCENARIOS))
    ex = _ex21()
    ex15 = ex._ex15()
    C = ex.C
    rx = ex.horizontal_array()
    scene, bottom, sea, sediment = ex.build_scene(rx, learnable=False)
    print(f"  {ex.FREQ_KHZ:.0f} kHz, {ex.N_RX} x {ex.N_TX}, {2 * ex.SECTOR_DEG:.0f} deg "
          f"swath to {ex.FAR:.0f} m; AUV at {ex.AUV_DEPTH:.0f} m in {ex.WATER_DEPTH:.0f} m")

    # ---- 21's boat, at rest ----------------------------------------------- #
    head = polar(ex.BOAT_RANGE, ex.BOAT_BEARING_DEG)
    girth = ex.HULL_BEAM + 2.0 * ex.HULL_DRAUGHT
    verts, faces = boat_hull_mesh(ex.HULL_LENGTH, ex.HULL_BEAM, ex.HULL_DRAUGHT,
                                  n_long=int(round(110 * ex.HULL_LENGTH / 12.0)),
                                  n_around=int(round(34 * girth / (3.2 + 2.0 * 4.0))))

    def make_boat():
        return mesh_target(verts, faces, position=head, yaw=ex.BOAT_HEADING_DEG, n_patches=6,
                           sound_speed=C, diffuse_db=ex.DIFFUSE_DB, learnable=False,
                           learnable_shape=False, facet_chunk=4096, checkpoint=False)

    # ---- the kelp stand --------------------------------------------------- #
    g = torch.Generator().manual_seed(21)
    n_along = int(KELP_STAND[0] / KELP_SPACING)
    n_across = int(KELP_STAND[1] / KELP_SPACING)
    u = (torch.arange(n_along) - (n_along - 1) / 2) * KELP_SPACING
    v = (torch.arange(n_across) - (n_across - 1) / 2) * KELP_SPACING
    U, V = torch.meshgrid(u, v, indexing="ij")
    plants = torch.stack([U, V], -1).reshape(-1, 2)
    plants = plants + 0.3 * KELP_SPACING * (2 * torch.rand(plants.shape, generator=g) - 1)
    depths = torch.linspace(ex.WATER_DEPTH - 0.5, 0.5, KELP_POINTS)
    kelp_offsets = torch.cat([
        torch.cat([plants.repeat_interleave(KELP_POINTS, 0),
                   depths.repeat(plants.shape[0]).unsqueeze(-1)], -1)], 0)
    kelp_offsets[:, 2] += 0.3 * (2 * torch.rand(kelp_offsets.shape[0], generator=g) - 1)
    kelp_offsets[:, 2] -= ex.WATER_DEPTH / 2                # about the stand's mid-depth
    # extinction with depth into the stand along the line of sight (the
    # stand's own x, its yaw being its bearing), in 2 m bins of one pattern each
    into = (kelp_offsets[:, 0] + KELP_STAND[0] / 2).clamp_min(0.0)
    bins = (into / 2.0).floor().long()
    kelp_patterns = {int(b): IsotropicScattering(
        KELP_POINT_DB - 2.0 * KELP_EXTINCTION_DB_PER_M * (float(b) + 0.5) * 2.0, learnable=False)
        for b in bins.unique()}
    kelp = ExtendedTarget(kelp_offsets, [kelp_patterns[int(b)] for b in bins],
                          position=polar(KELP_RANGE, KELP_BEARING_DEG, ex.WATER_DEPTH / 2),
                          yaw=KELP_BEARING_DEG, learnable=True)
    n_plants = int(plants.shape[0])

    # ---- the buoy, its chain and its sinker ------------------------------- #
    buoy_xy = polar(BUOY_RANGE, BUOY_BEARING_DEG)
    buoy = ExtendedTarget(torch.zeros(1, 3),
                          CurvedSurfaceScattering(BUOY_RADIUS, BUOY_RADIUS, learnable=False),
                          position=(buoy_xy[0], buoy_xy[1], BUOY_RADIUS * 0.6), learnable=True)
    chain_pts = catenary(CHAIN_LENGTH, CHAIN_SCOPE, ex.WATER_DEPTH - BUOY_RADIUS * 0.6 - 0.4,
                         CHAIN_POINTS)
    link_db = CHAIN_LINK_DB + 10 * math.log10(LINKS_PER_M * CHAIN_LENGTH / CHAIN_POINTS)
    chain = ExtendedTarget(chain_pts, IsotropicScattering(link_db, learnable=False),
                           position=(buoy_xy[0], buoy_xy[1], BUOY_RADIUS * 0.6),
                           yaw=CHAIN_DIRECTION_DEG, learnable=False)
    cd = math.radians(CHAIN_DIRECTION_DEG)
    sinker_xy = (buoy_xy[0] + CHAIN_SCOPE * math.cos(cd), buoy_xy[1] + CHAIN_SCOPE * math.sin(cd))
    s_verts, s_faces = box_mesh((SINKER, SINKER, SINKER))
    sinker = mesh_target(s_verts, s_faces, position=(sinker_xy[0], sinker_xy[1],
                                                     ex.WATER_DEPTH - SINKER / 2),
                         n_patches=1, sound_speed=C, diffuse_db=-10.0, learnable=False,
                         facet_chunk=4096, checkpoint=False)

    # ---- the shoal -------------------------------------------------------- #
    shoal = fish_school(N_SHOAL, polar(SHOAL_RANGE, SHOAL_BEARING_DEG, SHOAL_DEPTH),
                        radii=SHOAL_RADII, target_strength_db=SHOAL_TS_DB, yaw=0.0,
                        learnable=True, generator=torch.Generator().manual_seed(12))

    # ---- the sonar, as 21 has it ------------------------------------------ #
    dirs, _ = ex.transmit_fan(seed=ex.SEED)
    w_tx = ex.transmit_pattern(dirs)
    tilt = ex.passes_deg()[0]
    rx_beam = lambda d: ex.receive_beam(d, tilt)
    steer, bearings = azimuth_steering(181, ex.SECTOR_DEG)
    grid = make_time_grid(2.0 * ex.NEAR / C, 2.0 * ex.FAR / C, ex.N_BINS)
    rng = grid * C / 2.0
    shading = shading_window(ex.N_RX, "hamming")
    scale = beam_power_scale(shading, ex.PULSE_S)
    noise = float(beam_noise_power(scene.freqs_khz, bandwidth_hz=1.0 / ex.PULSE_S,
                                   directivity_db=line_array_directivity_db(ex.N_RX),
                                   wind_speed=ex.WIND))
    solid = (math.radians(2 * ex.SECTOR_DEG)
             * math.radians(ex.ELEV_DEG[1] - ex.ELEV_DEG[0]) / dirs.shape[0])
    seabed = LambertScattering(-27.0, learnable=False)
    span_y = ex.FAR * math.sin(math.radians(ex.SECTOR_DEG)) * 1.02
    x_range = (-0.03 * ex.FAR, 1.02 * ex.FAR)
    pixel_m = (x_range[1] - x_range[0]) / 299.0

    def echo(target, legs=24):
        return target_arrivals(
            scene, target, dirs, return_leg="eigenray", n_rx_rays=2000,
            rx_half_angle_deg=45.0, tx_weights=w_tx, tx_pattern=ex.transmit_pattern,
            rx_pattern=rx_beam, max_arrivals_per_leg=legs,
            generator=torch.Generator().manual_seed(ex.SEED))

    with torch.no_grad(), timed("  trace + reverberation, once"):
        result = trace(scene, dirs)
        rev = reverberation_arrivals(
            result, dirs, scene.freqs_khz, scattering=seabed,
            solid_angle_per_ray=solid, ray_weights=w_tx * rx_beam(-dirs),
            boundary="both", surface=sea, bottom=scene.bottom, max_arrivals=ex.PATCHES,
            generator=torch.Generator().manual_seed(ex.SEED + 1))

    def ping(targets, legs):
        parts = [rev] + [echo(t, l) for t, l in zip(targets, legs)]
        both = ArrivalSet(*(None if any(p[i] is None for p in parts)
                            else torch.cat([p[i] for p in parts], dim=0)
                            for i in range(len(rev))))
        image = beamform(both, rx, scene.freqs_khz, grid, steer, sigma_t=ex.PULSE_S,
                         shading=shading, steer_chunk=8)
        signal = calibrate(image, ex.SOURCE_LEVEL_DB, beam_scale=scale)
        noisy = add_receiver_noise(signal, noise,
                                   generator=torch.Generator().manual_seed(ex.SEED + 2))
        shown, _ = ex.display(noisy, rng, pixel_m=pixel_m)
        cart, gx, gy = ex15.to_cartesian(shown, bearings, grid, n_x=300, n_y=300,
                                         x_range=x_range, y_range=(-span_y, span_y))
        return cart, gx, gy, sum(p.n_arrivals for p in parts[1:])

    banner("the bare picture: sea, seabed, the boat at rest")
    with torch.no_grad(), timed("  ping"):
        bare, gx, gy, _ = ping([make_boat()], [24])
    X, Y = torch.meshgrid(gx, gy, indexing="xy")
    db = lambda t: 10.0 * torch.log10(t.detach().clamp_min(1e-30))
    bare_db = db(bare)
    ext = [float(gx.min()), float(gx.max()), float(gy.min()), float(gy.max())]

    ok = True
    for name in SCENARIOS:
        banner(f"scenario: {name}")
        overlays = []
        if name == "kelp":
            print(f"  {n_plants} plants of {KELP_POINTS} points at {KELP_POINT_DB:.0f} dB over "
                  f"{KELP_STAND[0]:.0f} x {KELP_STAND[1]:.0f} m at {KELP_RANGE:.0f} m, bearing "
                  f"{KELP_BEARING_DEG:+.0f} deg: {KELP_POINT_DB + 10 * math.log10(KELP_POINTS):.0f} dB a plant")
            targets, legs = [make_boat(), kelp], [24, LEGS["kelp"]]
            live = {"stand position": kelp.position}
            kb = math.radians(KELP_BEARING_DEG)
            cx, cy = KELP_RANGE * math.cos(kb), KELP_RANGE * math.sin(kb)
            hx, hy = KELP_STAND[0] / 2, KELP_STAND[1] / 2
            cs, sn = math.cos(kb), math.sin(kb)
            corners = [(-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy), (-hx, -hy)]
            overlays.append(("w-", [cx + a * cs - b * sn for a, b in corners],
                             [cy + a * sn + b * cs for a, b in corners]))
        elif name == "buoy":
            print(f"  a {BUOY_RADIUS:.2f} m buoy at {BUOY_RANGE:.0f} m, bearing {BUOY_BEARING_DEG:+.0f} "
                  f"deg; {CHAIN_LENGTH:.0f} m of chain ({link_db:.0f} dB a point, {CHAIN_POINTS}) to a "
                  f"{SINKER:.1f} m sinker {CHAIN_SCOPE:.0f} m away on bearing {CHAIN_DIRECTION_DEG:.0f}")
            targets, legs = [make_boat(), buoy, chain, sinker], [24, 24, LEGS["buoy"], 24]
            live = {"buoy position": buoy.position}
            overlays.append(("w:", [buoy_xy[0], sinker_xy[0]], [buoy_xy[1], sinker_xy[1]]))
        else:
            print(f"  {N_SHOAL} fish of {SHOAL_TS_DB:.0f} dB in {2 * SHOAL_RADII[0]:.0f} x "
                  f"{2 * SHOAL_RADII[1]:.0f} x {2 * SHOAL_RADII[2]:.0f} m at {SHOAL_RANGE:.0f} m, "
                  f"bearing {SHOAL_BEARING_DEG:+.0f} deg, {SHOAL_DEPTH:.0f} m deep -- "
                  f"{SHOAL_TS_DB + 10 * math.log10(N_SHOAL):.0f} dB incoherently")
            targets, legs = [make_boat(), shoal], [24, LEGS["shoal"]]
            live = {"shoal position": shoal.position,
                    "fish target strength": shoal.pattern_for(0).target_strength_db}
            sb = math.radians(SHOAL_BEARING_DEG)
            tt = torch.linspace(0, 2 * math.pi, 100)
            overlays.append(("w-", (SHOAL_RANGE * math.cos(sb) + SHOAL_RADII[0] * tt.cos()).tolist(),
                             (SHOAL_RANGE * math.sin(sb) + SHOAL_RADII[1] * tt.sin()).tolist()))

        with timed("  ping"):
            cart, _, _, n_echo = ping(targets, legs)
        print(f"  {n_echo} target arrivals")
        cart_db = db(cart)

        if name == "kelp":
            rel = torch.stack([X - cx, Y - cy], -1)
            a_ = rel[..., 0] * cs + rel[..., 1] * sn
            b_ = -rel[..., 0] * sn + rel[..., 1] * cs
            inside = (a_.abs() < hx) & (b_.abs() < hy)
            gain_db = float((cart_db - bare_db)[inside].mean())
            front = float((cart_db - bare_db)[inside & (a_ < -hx + 10.0)].mean())
            back = float((cart_db - bare_db)[inside & (a_ > hx - 10.0)].mean())
            print(f"  the stand reads {gain_db:+.1f} dB over what was in its {int(inside.sum())} cells: "
                  f"{front:+.1f} dB over its front 10 m, {back:+.1f} dB over its back 10 m")
            verdict = check("the kelp stand reads well above what was in its cells",
                            gain_db > 6.0, f"{gain_db:+.1f} dB")
        elif name == "buoy":
            near = (X - buoy_xy[0]) ** 2 + (Y - buoy_xy[1]) ** 2 < 4.0 ** 2
            around = ((X - buoy_xy[0]) ** 2 + (Y - buoy_xy[1]) ** 2 < 25.0 ** 2) & ~near
            buoy_db = float(cart_db[near].max() - cart_db[around].median())
            # the chain's line in plan, from 5 m off the buoy to 3 m short of the sinker
            t = torch.linspace(5.0 / CHAIN_SCOPE, 1 - 3.0 / CHAIN_SCOPE, 40)
            lx = buoy_xy[0] + t * (sinker_xy[0] - buoy_xy[0])
            ly = buoy_xy[1] + t * (sinker_xy[1] - buoy_xy[1])
            ix = ((lx - float(gx[0])) / float(gx[1] - gx[0])).round().long().clamp(0, gx.numel() - 1)
            iy = ((ly - float(gy[0])) / float(gy[1] - gy[0])).round().long().clamp(0, gy.numel() - 1)
            along = (cart_db - bare_db)[iy, ix]
            chain_db = float(along.median())
            print(f"  the buoy peaks {buoy_db:+.1f} dB over its surroundings; the chain's line reads "
                  f"{chain_db:+.1f} dB (median) over the bare picture, {float(along.max()):+.1f} at most")
            verdict = check("the buoy is a point well above its surroundings, and the chain a line",
                            buoy_db > 10.0 and chain_db > 3.0,
                            f"buoy {buoy_db:+.1f} dB, chain {chain_db:+.1f} dB")
        else:
            sx, sy = SHOAL_RANGE * math.cos(sb), SHOAL_RANGE * math.sin(sb)
            inside = ((X - sx) / SHOAL_RADII[0]) ** 2 + ((Y - sy) / SHOAL_RADII[1]) ** 2 < 1.0
            gain_db = float((cart_db - bare_db)[inside].mean())
            print(f"  the shoal reads {gain_db:+.1f} dB over what was in its {int(inside.sum())} cells")
            verdict = check("the shoal reads well above what was in its cells",
                            gain_db > 8.0, f"{gain_db:+.1f} dB")
        ok &= verdict

        with timed("  backward"):
            cart.sum().backward()
        states = {k: (p.grad is not None and bool(torch.isfinite(p.grad).all())
                      and float(p.grad.abs().sum()) > 0) for k, p in live.items()}
        for k, v in states.items():
            print(f"  d(picture)/d({k:<20s}): {'OK' if v else 'ZERO'}")
        ok &= check(f"the {name} picture carries gradients to " + " and ".join(live),
                    all(states.values()),
                    ", ".join(f"{k}: {'OK' if v else 'ZERO'}" for k, v in states.items()))

        # ---- the figure --------------------------------------------------- #
        ref = float(bare_db.max())
        fig, axes = plt.subplots(1, 3, figsize=(19, 6.2))
        for ax, img, title in ((axes[0], bare_db, "the bare picture: sea, seabed, the boat at rest"),
                               (axes[1], cart_db, f"with {CAPTION[name]}")):
            im = ax.imshow(img.numpy(), origin="lower", extent=ext, vmin=ex.THRESHOLD_DB,
                           vmax=ref, cmap="inferno", aspect="equal")
            ax.set_title(title); ax.set_xlabel("forward (m)"); ax.set_ylabel("across (m)")
        m = axes[2].imshow((cart_db - bare_db).clamp(-15.0, 25.0).numpy(), origin="lower",
                           extent=ext, vmin=-15, vmax=25, cmap="coolwarm", aspect="equal")
        axes[2].set_title(f"{name} minus bare (dB)"); axes[2].set_xlabel("forward (m)")
        fig.colorbar(im, ax=axes[1], fraction=0.04, label="dB re the background at that range")
        fig.colorbar(m, ax=axes[2], fraction=0.04, label="dB")
        for ax in axes:
            for style, xs, ys in overlays:
                ax.plot(xs, ys, style, lw=1, alpha=0.7)
            ax.plot(head[0], head[1], "c+", ms=10, mew=1.5)
            ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
        fig.suptitle(f"{ex.FREQ_KHZ:.0f} kHz FLS, {2 * ex.SECTOR_DEG:.0f} deg to {ex.FAR:.0f} m, "
                     f"median TVG floored at +{ex.THRESHOLD_DB:.0f} dB: what {CAPTION[name]} adds to 21's picture")
        save(fig, f"26_scenario_{name}.png")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
