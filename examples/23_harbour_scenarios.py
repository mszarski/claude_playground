"""Three scenarios at ``examples/21``'s settings: a breakwater, a wake, a school.

The same sonar, environment and picture as ``examples/21`` -- a 120 kHz Mills
cross on an AUV at 12 m in 30 m of water, a light wind sea over rough sand,
the median-TVG image on a grid in metres out to 300 m -- each time with ONE
thing added, and the bare picture beside it under the same colour scale:

* **a breakwater along the right-hand edge of the field of view**, the whole
  way out: 300 m of vertical caisson face, seen along its length from a sonar
  8 m off its line in the harbour.  That is grazing incidence -- ten degrees
  down to one and a half -- where a flat face has no specular return, so what it
  shows is its roughness, a diffuse channel of -6 dB for concrete and marine
  growth, and it is bright anyway, because there is so much of it in every
  beam.  It is also an *occluder*: the water beyond it is not lit, and the
  reverberation there is gone.
* **a vessel under way, with its wake.**  The 30 m boat of ``examples/21``,
  now the head of a 90 s straight track at 6 m/s.  Its Kelvin wake is added to
  the wind sea as a height field, and -- the part a sonar actually sees, as
  ``examples/20`` measured -- the band of entrained air behind it multiplies
  the surface backscatter by 20 dB along the track, a hull's width wide and
  decaying over minutes.
* **a school of fish** in mid-water: a thousand swim-bladder fish of -40 dB
  each, in an ellipsoid 30 m by 16 m by 6 m.  Incoherently that is -10 dB of
  target strength; coherently, in an image, it is speckle about that, spread over
  the school's footprint rather than glinting from a point.

They are separate scenarios, not one scene: ``HYDROPT_SCENARIO`` picks one
(``seawall``, ``wake``, ``fish``) or, by default, runs all three in turn, each
into its own figure.  The bare picture is formed once and shared.  Every
scenario's image stays differentiable in what was put into it: the vessel's
position, the school's.

Acceptance criteria:
  * the breakwater stands well above the reverberation at its own range,
    along its whole length;
  * the wake band stands above the sea beside it, and does not in the bare
    picture;
  * the school stands above what was in its cells before;
  * the picture carries gradients to the school's position and the vessel's.
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
    LambertScattering, add_receiver_noise, azimuth_steering, beam_noise_power,
    beam_power_scale, beamform, calibrate, fish_school, line_array_directivity_db,
    make_time_grid, reverberation_arrivals, seawall_mesh, shading_window,
    target_arrivals, trace, wave_number_peak_pm,
)
from hydropt.beamform import ArrivalSet
from hydropt.mesh import boat_hull_mesh, mesh_target
from hydropt.wake import bubble_wake_gain, kelvin_wake_surface

SCENARIO = os.environ.get("HYDROPT_SCENARIO", "all")
if SCENARIO not in ("all", "seawall", "wake", "fish"):
    raise SystemExit(f"HYDROPT_SCENARIO must be all, seawall, wake or fish, got {SCENARIO!r}")
SCENARIOS = ("seawall", "wake", "fish") if SCENARIO == "all" else (SCENARIO,)
CAPTION = {"seawall": "a breakwater along the right-hand edge",
           "wake": "a vessel under way, with its wake",
           "fish": "a school of fish"}

# the breakwater: along the FOV's right-hand edge.  Its line runs at -50 deg,
# 8 m outboard of the sonar, so seen from the sonar it enters the 60 deg edge
# of the fan at the near range and converges on -51.5 deg by 300 m: inside
# the swath the whole way, 1.5 to 10 degrees of grazing.  (A line parallel to
# the edge itself is either outside the fan along its whole length or, offset
# the other way, in front of the sonar.)
WALL_BEARING_DEG = -50.0
WALL_OFFSET = 8.0                 # metres outboard of the line through the sonar
WALL_FROM, WALL_TO = 20.0, 320.0  # along that line
WALL_ABOVE_WATER = 3.0
WALL_DIFFUSE_DB = -6.0
# the vessel: 21's boat as the head of a straight track
SPEED = 6.0                       # m/s, 11.7 knots
TRACK_SECONDS, N_TRACK = 90.0, 181
WAKE_AMPLITUDE = 0.3
# the bubble band is as wide as the hull, and 20 dB is the middle of what is
# published for high-frequency backscatter from a ship's bubble wake
BUBBLE_GAIN_DB, BUBBLE_WIDTH, BUBBLE_DECAY = 20.0, 8.0, 300.0
# the school
SCHOOL_RANGE, SCHOOL_BEARING_DEG, SCHOOL_DEPTH = 150.0, 12.0, 15.0
SCHOOL_RADII = (15.0, 8.0, 3.0)
N_FISH, FISH_TS_DB = 1000, -40.0


def _ex21():
    path = Path(__file__).resolve().parent / "21_long_range_300m.py"
    spec = importlib.util.spec_from_file_location("_ex21", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def place(vertices: torch.Tensor, yaw_deg: float, position) -> torch.Tensor:
    """Body-frame vertices into the world, for the occlusion test."""
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    rot = torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=vertices.dtype)
    return vertices @ rot.T + torch.tensor(position, dtype=vertices.dtype)


def main() -> int:
    setup()          # float64: see the note on precision in the docstring
    banner("23 -- three scenarios at 21's settings: " + ", ".join(SCENARIOS))
    ex = _ex21()
    ex15 = ex._ex15()
    C = ex.C
    rx = ex.horizontal_array()
    scene, bottom, sea, sediment = ex.build_scene(rx, learnable=False)
    print(f"  {ex.FREQ_KHZ:.0f} kHz, {ex.N_RX} x {ex.N_TX}, {2 * ex.SECTOR_DEG:.0f} deg "
          f"swath to {ex.FAR:.0f} m; AUV at {ex.AUV_DEPTH:.0f} m in {ex.WATER_DEPTH:.0f} m")

    # ---- 21's boat, at rest, and the same hull as the head of a track ------ #
    b = math.radians(ex.BOAT_BEARING_DEG)
    head = torch.tensor([ex.BOAT_RANGE * math.cos(b), ex.BOAT_RANGE * math.sin(b)])
    h = math.radians(ex.BOAT_HEADING_DEG)
    girth = ex.HULL_BEAM + 2.0 * ex.HULL_DRAUGHT
    verts, faces = boat_hull_mesh(ex.HULL_LENGTH, ex.HULL_BEAM, ex.HULL_DRAUGHT,
                                  n_long=int(round(110 * ex.HULL_LENGTH / 12.0)),
                                  n_around=int(round(34 * girth / (3.2 + 2.0 * 4.0))))

    def make_boat():
        return mesh_target(verts, faces, position=(float(head[0]), float(head[1]), 0.0),
                           yaw=ex.BOAT_HEADING_DEG, n_patches=6, sound_speed=C,
                           diffuse_db=ex.DIFFUSE_DB, learnable=True,
                           learnable_shape=False, facet_chunk=256)

    times = torch.linspace(0.0, TRACK_SECONDS, N_TRACK)
    track = head + SPEED * (times - times[-1]).unsqueeze(-1) * torch.tensor([math.cos(h), math.sin(h)])
    dx = 2.0 * math.pi / wave_number_peak_pm(ex.WIND) / 8.0
    n = int(math.ceil(700.0 / dx)) + 1
    origin = (-40.0, -n * dx / 2)
    extent = ((origin[0], origin[0] + (n - 1) * dx), (origin[1], origin[1] + (n - 1) * dx))

    # ---- the breakwater --------------------------------------------------- #
    th = math.radians(WALL_BEARING_DEG)
    along = torch.tensor([math.cos(th), math.sin(th)])
    inboard = torch.tensor([-math.sin(th), math.cos(th)])      # toward the swath
    wall_mid = 0.5 * (WALL_FROM + WALL_TO) * along - WALL_OFFSET * inboard
    w_verts, w_faces = seawall_mesh(WALL_TO - WALL_FROM, ex.WATER_DEPTH + WALL_ABOVE_WATER,
                                    slope_deg=90.0, n_along=150, n_up=4)
    wall_pos = (float(wall_mid[0]), float(wall_mid[1]), ex.WATER_DEPTH)
    wall = mesh_target(w_verts, w_faces, position=wall_pos, yaw=WALL_BEARING_DEG,
                       n_patches=150, split_axis=0, sound_speed=C,   # 2 m along: under the pixel
                       diffuse_db=WALL_DIFFUSE_DB, learnable=False, facet_chunk=256)
    wall_world = place(w_verts, WALL_BEARING_DEG, wall_pos)

    # ---- the school ------------------------------------------------------- #
    sb = math.radians(SCHOOL_BEARING_DEG)
    school_xy = (SCHOOL_RANGE * math.cos(sb), SCHOOL_RANGE * math.sin(sb))
    school = fish_school(N_FISH, (school_xy[0], school_xy[1], SCHOOL_DEPTH),
                         radii=SCHOOL_RADII, target_strength_db=FISH_TS_DB, yaw=20.0,
                         learnable=True, generator=torch.Generator().manual_seed(11))

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

    def echo(target):
        return target_arrivals(
            scene, target, dirs, return_leg="eigenray", n_rx_rays=2000,
            rx_half_angle_deg=45.0, tx_weights=w_tx, tx_pattern=ex.transmit_pattern,
            rx_pattern=rx_beam, max_arrivals_per_leg=24,
            generator=torch.Generator().manual_seed(ex.SEED))

    def ping(surface, *, targets, occluders=None, gain=None):
        scene.surface = surface
        result = trace(scene, dirs)
        rev = reverberation_arrivals(
            result, dirs, scene.freqs_khz, scattering=seabed,
            solid_angle_per_ray=solid, ray_weights=w_tx * rx_beam(-dirs),
            boundary="both", surface=surface, bottom=scene.bottom,
            max_arrivals=ex.PATCHES, occluders=occluders, surface_gain=gain,
            generator=torch.Generator().manual_seed(ex.SEED + 1))
        parts = [rev] + [echo(t) for t in targets]
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
        return cart, gx, gy, rev.n_arrivals, sum(p.n_arrivals for p in parts[1:])

    banner("the bare picture: sea, seabed, the boat at rest")
    with torch.no_grad(), timed("  ping"):
        bare, gx, gy, n_rev0, n_echo0 = ping(sea, targets=[make_boat()])
    print(f"  {n_rev0} patches + {n_echo0} target arrivals")
    X, Y = torch.meshgrid(gx, gy, indexing="xy")          # [n_y, n_x], as the image
    R = torch.hypot(X, Y)
    db = lambda t: 10.0 * torch.log10(t.detach().clamp_min(1e-30))
    bare_db = db(bare)
    ext = [float(gx.min()), float(gx.max()), float(gy.min()), float(gy.max())]

    ok = True
    for name in SCENARIOS:
        banner(f"scenario: {name}")
        overlays = []
        if name == "seawall":
            print(f"  {WALL_TO - WALL_FROM:.0f} m of vertical face along bearing "
                  f"{WALL_BEARING_DEG:+.0f} deg, {WALL_OFFSET:.0f} m outboard of the sonar's "
                  f"line, seabed to {WALL_ABOVE_WATER:.0f} m above the water, diffuse "
                  f"{WALL_DIFFUSE_DB:.0f} dB; the boat at rest as in 21")
            boat = make_boat()
            targets, surface, gain, occl = [boat, wall], sea, None, [(wall_world, w_faces)]
            live = {"vessel position": boat.position}
            p0 = WALL_FROM * along - WALL_OFFSET * inboard
            p1 = WALL_TO * along - WALL_OFFSET * inboard
            overlays.append(("c--", [float(p0[0]), float(p1[0])], [float(p0[1]), float(p1[1])]))
        elif name == "wake":
            print(f"  the {ex.HULL_LENGTH:.0f} m boat under way at {SPEED:.0f} m/s, heading "
                  f"{ex.BOAT_HEADING_DEG:.0f} deg, {SPEED * TRACK_SECONDS:.0f} m of track "
                  f"behind it")
            with timed("  the Kelvin wake on the wind sea"):
                surface = kelvin_wake_surface(
                    track, times, amplitude=WAKE_AMPLITUDE, extent=extent, spacing=dx,
                    base=sea.heights.detach(), water_depth=ex.WATER_DEPTH,
                    n_directions=96, max_angle_deg=62.0, decay_time=240.0)
            eta = (surface.heights - sea.heights).detach()
            print(f"  wake {float(eta.std()):.3f} m RMS over the grid, peak "
                  f"{float(eta.abs().max()):.2f} m; bubble band +{BUBBLE_GAIN_DB:.0f} dB, "
                  f"{BUBBLE_WIDTH:.0f} m wide at the stern")
            gain = lambda xy: bubble_wake_gain(xy, track, times, gain_db=BUBBLE_GAIN_DB,
                                               width=BUBBLE_WIDTH, decay_time=BUBBLE_DECAY)
            boat = make_boat()
            targets, occl = [boat], None
            live = {"vessel position": boat.position}
            overlays.append(("w:", track[:, 0].tolist(), track[:, 1].tolist()))
        else:
            print(f"  {N_FISH} fish of {FISH_TS_DB:.0f} dB at {SCHOOL_RANGE:.0f} m, bearing "
                  f"{SCHOOL_BEARING_DEG:+.0f} deg, {SCHOOL_DEPTH:.0f} m deep -- "
                  f"{FISH_TS_DB + 10 * math.log10(N_FISH):.0f} dB incoherently; the boat "
                  f"at rest as in 21")
            boat = make_boat()
            targets, surface, gain, occl = [boat, school], sea, None, None
            live = {"school position": school.position,
                    "fish target strength": school.pattern_for(0).target_strength_db}
            cy, sy = math.cos(math.radians(20.0)), math.sin(math.radians(20.0))
            tt = torch.linspace(0, 2 * math.pi, 100)
            ex_, ey = SCHOOL_RADII[0] * tt.cos(), SCHOOL_RADII[1] * tt.sin()
            overlays.append(("w-", (school_xy[0] + ex_ * cy - ey * sy).tolist(),
                             (school_xy[1] + ex_ * sy + ey * cy).tolist()))

        with timed("  ping"):
            cart, _, _, n_rev, n_echo = ping(surface, targets=targets, occluders=occl, gain=gain)
        print(f"  {n_rev} patches + {n_echo} target arrivals")
        cart_db = db(cart)

        # ---- measured on the picture ------------------------------------- #
        if name == "seawall":
            s_along = X * along[0] + Y * along[1]
            d_off = X * inboard[0] + Y * inboard[1] + WALL_OFFSET     # 0 on the wall's line
            on_wall = (d_off.abs() < 6.0) & (s_along > 60.0) & (s_along < ex.FAR * 0.95)
            beside = (d_off > 15.0) & (d_off < 40.0) & (s_along > 60.0) & (s_along < ex.FAR * 0.95)
            over_bare = float((cart_db - bare_db)[on_wall].median())
            over_beside = float(cart_db[on_wall].median() - cart_db[beside].median())
            print(f"  along its line the wall reads {over_bare:+.1f} dB over the bare picture, "
                  f"{over_beside:+.1f} dB over the water 15-40 m inboard")
            verdict = check("the breakwater stands well above the water inboard of it, along its length",
                            over_beside > 8.0, f"{over_beside:+.1f} dB, {over_bare:+.1f} dB over bare")
        elif name == "wake":
            rel = torch.stack([X - float(head[0]), Y - float(head[1])], dim=-1)
            astern = -(rel[..., 0] * math.cos(h) + rel[..., 1] * math.sin(h))
            abeam = (-rel[..., 0] * math.sin(h) + rel[..., 1] * math.cos(h)).abs()
            inswath = (R < ex.FAR * 0.95) & (R > ex.NEAR + 10)
            band = (astern > 25.0) & (astern < 150.0) & (abeam < 5.0) & inswath
            side = (astern > 25.0) & (astern < 150.0) & (abeam > 20.0) & (abeam < 45.0) & inswath
            over = float(cart_db[band].median() - cart_db[side].median())
            over_bare = float(bare_db[band].median() - bare_db[side].median())
            print(f"  the band astern stands {over:+.1f} dB over the sea beside it "
                  f"({over_bare:+.1f} dB in the bare picture)")
            verdict = check("the wake band stands above the sea beside it, and did not before",
                            over > 4.0 and abs(over_bare) < 2.5,
                            f"{over:+.1f} dB with the wake, {over_bare:+.1f} dB without")
        else:
            u = (X - school_xy[0]) * cy + (Y - school_xy[1]) * sy
            v = -(X - school_xy[0]) * sy + (Y - school_xy[1]) * cy
            in_school = (u / SCHOOL_RADII[0]) ** 2 + (v / SCHOOL_RADII[1]) ** 2 < 1.0
            gain_db = float((cart_db - bare_db)[in_school].mean())
            print(f"  the school reads {gain_db:+.1f} dB over what was in its "
                  f"{int(in_school.sum())} cells")
            verdict = check("the school stands above what was in its cells",
                            gain_db > 5.0, f"{gain_db:+.1f} dB")
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
            # 21's own window: floored just above the background, up to the boat
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
            ax.plot(float(head[0]), float(head[1]), "c+", ms=10, mew=1.5)
            ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
        fig.suptitle(f"{ex.FREQ_KHZ:.0f} kHz FLS, {2 * ex.SECTOR_DEG:.0f} deg to {ex.FAR:.0f} m, "
                     f"median TVG floored at +{ex.THRESHOLD_DB:.0f} dB: what {CAPTION[name]} adds to 21's picture")
        save(fig, f"23_scenario_{name}.png")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
