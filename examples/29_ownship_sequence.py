"""The sonar under way: the ownship moves, the world stands still.

``examples/28`` moved a boat past a fixed sonar.  This is the other case,
the usual one for a forward-looking sonar on an AUV or a survey launch: the
sonar moves and everything in the picture is stationary -- a boat moored
across the swath, a buoy on its chain, a stand of kelp -- so that every
frame is a new view of the same world.  The ownship runs 145 m forward at
6 knots with a 30 degree turn to port, a ping every few seconds, and each
ping is rendered as ``examples/21`` renders one.

**Construction and assumptions.**  The sonar is 21's, untouched: it sits at the origin of
its own frame looking along ``+x``, with 21's fan, beams, display and grid
in metres.  What moves is the world, re-expressed in that frame per ping:

* **the sea and the seabed** are built once at world scale -- the same
  generators as 21's (``fractal_bathymetry`` at 0.9 m RMS over a 30 m
  shelf, a Pierson-Moskowitz sea for a 4 m/s wind at eight nodes per peak
  wavelength), over 1000 m instead of 700 so that every pose's swath is
  inside them -- and for each ping resampled onto 21's own grids at the
  ownship pose (``hydropt.reframe_height_field``), so the scene the tracer
  sees has 21's shape and the world's heights.  The reverberation is then
  traced afresh for every ping.  That is what makes a frame cost a whole
  picture (10 s at 300 m) rather than an echo: keeping one background
  would freeze the sea's speckle to the sonar, and the picture would say
  the sea moved with the boat;
* **the targets** are 21's boat, moored; 26's buoy with its chain and
  sinker; and 26's kelp stand (with the plants' per-bin extinction along
  the stand's own axis, fixed in the world, not along the changing line of
  sight -- a stand seen end-on from the side reads a little bright at its
  far end).  Each has a world pose, and each ping rebuilds it at its pose
  relative to the ownship (``hydropt.relative_pose``): position turned and
  shifted, heading less the ownship's;
* **the display gain** is taken from the first ping and held, as 28 does
  and as an AGC would; the receiver noise is drawn afresh per ping.

Acceptance criteria:
  * the buoy, a point, stands more than 10 dB over the 25 m disc around
    where the ownship pose says it should be, in every ping, with its
    peak within one beam width at its range plus 3 m of that place;
    carried back into the world by the ownship pose, the buoy's position
    scatters (RMS over the pings) by less than a beam width at its mean
    range, and its mean lands within a beam plus the buoy's radius of the
    truth (the peak is the sphere's near face);
  * the boat's echo (the ping's excess over the same ping without targets,
    within 40 m of its relative position) has its centroid within half a
    hull plus one beam width of the boat in every ping;
  * the kelp stand reads more than 4 dB over its cells without it in the
    median ping;
  * consecutive pings' target-free pictures correlate below 0.5 (the sea
    is rendered anew at each pose, not carried along), while a ping
    rendered twice at one pose correlates at 1;
  * a frame's cost is reported (a whole picture each, see above).

``HYDROPT_FRAMES`` sets the number of pings (12); the track and the world
scale with the head's swath (``HYDROPT_SONAR=330``).
"""

from __future__ import annotations

import importlib.util
import math
import os
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
import torch

from _common import FIGURE_DIR, banner, check, save, setup, timed
from hydropt import (
    CurvedSurfaceScattering, ExtendedTarget, IsotropicScattering, LambertScattering,
    PictureRenderer, Trajectory, azimuth_steering, beam_noise_power, box_mesh,
    fractal_bathymetry, line_array_directivity_db, make_time_grid, pierson_moskowitz_surface,
    reframe_height_field, relative_pose, shading_window, wave_number_peak_pm,
)
from hydropt.mesh import boat_hull_mesh, mesh_target

N_FRAMES = int(os.environ.get("HYDROPT_FRAMES", 12))
SPEED = 3.0                       # m/s, 6 knots: a survey speed
WORLD_M = 1000.0                  # the world's sea and seabed, a square this wide (at 300 m)

# ---- the world, laid out for the 120 kHz head's 300 m swath (scaled by S) -- #
BOAT_WORLD = (270.0, -20.0, 120.0)        # x, y, heading: moored, bow to port; inside the sector at every pose
BUOY_WORLD = (200.0, 40.0)
BUOY_RADIUS = 0.75
CHAIN_LENGTH, CHAIN_SCOPE = 45.0, 30.0    # as 26: metres of chain, horizontal span to the sinker
CHAIN_LINK_DB, LINKS_PER_M, CHAIN_POINTS = -26.0, 5.0, 200
CHAIN_DIRECTION_DEG = 90.0                # the chain runs to port in the world
SINKER = 0.8
KELP_WORLD = (230.0, 120.0, 20.0)         # the stand's centre and the axis its extinction runs along
KELP_STAND = (45.0, 30.0)                 # as 26
KELP_SPACING, KELP_POINTS, KELP_POINT_DB, KELP_EXTINCTION_DB_PER_M = 3.0, 8, -17.0, 0.4
LEGS = {"kelp": 6, "chain": 6}            # paths per leg per point for the many-point targets


def _ex21():
    path = Path(__file__).resolve().parent / "21_long_range_300m.py"
    spec = importlib.util.spec_from_file_location("_ex21", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_EX = _ex21()
S = _EX.FAR / 300.0


def track() -> Trajectory:
    """145 m forward with a 30 degree turn to port, at 6 knots."""
    f = S
    waypoints = [(0.0, 0.0), (40.0 * f, 0.0), (80.0 * f, 3.0 * f), (115.0 * f, 14.0 * f),
                 (145.0 * f, 32.0 * f)]
    return Trajectory.from_waypoints(waypoints, speed=SPEED * S)


def world_fields(ex, seed: int = 3):
    """The world's seabed and sea at world scale, by 21's generators.

    Both are a fresh realisation (a bigger grid draws a different field for
    the same seed) with 21's statistics: 0.9 m RMS fractal relief on the
    30 m shelf, a 4 m/s Pierson-Moskowitz sea at eight nodes per peak
    wavelength.  Their origin puts the ownship's start 100 m inside the
    near edge, and the track's turn to port is inside them by 300 m.
    """
    size = WORLD_M * S
    nb = int(math.ceil(size / 16.0)) + 1
    bottom = fractal_bathymetry((nb, nb), (16.0, 16.0), base_depth=ex.WATER_DEPTH, rms=0.9,
                                exponent=3.0, origin=(-100.0 * S, -size / 2), learnable=False,
                                generator=torch.Generator().manual_seed(seed))
    dx = 2.0 * math.pi / wave_number_peak_pm(ex.WIND) / 8.0
    ns = int(math.ceil(size / dx)) + 1
    surface = pierson_moskowitz_surface((ns, ns), (dx, dx), ex.WIND,
                                        origin=(-100.0 * S, -size / 2), learnable=False,
                                        generator=torch.Generator().manual_seed(seed + 1))
    return bottom, surface


def catenary(length: float, span: float, drop: float, n: int) -> torch.Tensor:
    """``[n, 3]`` points along a chain of ``length`` from (0, 0, 0) to (span, 0, drop) (as 26)."""
    chord = math.hypot(span, drop)
    if length <= chord:
        t = torch.linspace(0.0, 1.0, n)
        return torch.stack([t * span, torch.zeros(n), t * drop], -1)
    target = math.sqrt(length ** 2 - drop ** 2)
    lo, hi = 1e-3, 1e4
    for _ in range(200):
        a = 0.5 * (lo + hi)
        if 2 * a * math.sinh(span / (2 * a)) > target:
            lo = a
        else:
            hi = a
    a = 0.5 * (lo + hi)
    x0 = span / 2 - a * math.asinh(drop / (2 * a * math.sinh(span / (2 * a))))
    c = -a * math.cosh(-x0 / a)
    x = torch.linspace(0.0, span, n)
    z = a * torch.cosh((x - x0) / a) + c
    return torch.stack([x, torch.zeros(n), z], -1)


def main() -> int:
    setup(double=False)
    banner("29 -- the sonar under way through a world that stands still")
    ex = _EX
    ex15 = ex._ex15()
    C = ex.C
    rx = ex.horizontal_array()
    # 21's scene once: it fixes the grids (shape, spacing, origin) the tracer
    # will see, and every setting but the two height fields
    scene0, bottom0, surface0, sediment = ex.build_scene(rx, learnable=False)
    print(f"  {ex.FREQ_KHZ:.0f} kHz, {ex.N_RX} x {ex.N_TX}, {2 * ex.SECTOR_DEG:.0f} deg "
          f"swath to {ex.FAR:.0f} m; AUV at {ex.AUV_DEPTH:.0f} m in {ex.WATER_DEPTH:.0f} m")

    # ---- the world ---------------------------------------------------------- #
    with timed("  the world's sea and seabed"):
        world_bottom, world_surface = world_fields(ex)
    print(f"  seabed {tuple(world_bottom.shape)} nodes at 16 m, sea {tuple(world_surface.shape)} "
          f"nodes at {float(world_surface.spacing[0]):.2f} m, {WORLD_M * S:.0f} m square; "
          f"21's grids are {tuple(bottom0.shape)} and {tuple(surface0.shape)}")

    def scene_at(x, y, heading_deg):
        """21's scene with the world's heights read at the ownship pose."""
        b = reframe_height_field(world_bottom, shape=bottom0.shape, spacing=bottom0.spacing.tolist(),
                                 origin=bottom0.origin.tolist(), x=x, y=y, heading_deg=heading_deg)
        s = reframe_height_field(world_surface, shape=surface0.shape,
                                 spacing=surface0.spacing.tolist(), origin=surface0.origin.tolist(),
                                 x=x, y=y, heading_deg=heading_deg)
        scene, _, _, _ = ex.build_scene(rx, learnable=False, bottom=b, surface=s)
        return scene

    # ---- the targets, each a builder at a pose in the sonar's frame -------- #
    girth = ex.HULL_BEAM + 2.0 * ex.HULL_DRAUGHT
    verts, faces = boat_hull_mesh(ex.HULL_LENGTH, ex.HULL_BEAM, ex.HULL_DRAUGHT,
                                  n_long=int(round(110 * ex.HULL_LENGTH / 12.0)),
                                  n_around=int(round(34 * girth / (3.2 + 2.0 * 4.0))))

    def boat(x, y, heading_deg):
        return mesh_target(verts, faces, position=(x, y, 0.0), yaw=heading_deg, n_patches=6,
                           sound_speed=C, diffuse_db=ex.DIFFUSE_DB, learnable=False,
                           learnable_shape=False, facet_chunk=4096, checkpoint=False)

    buoy_depth = BUOY_RADIUS * 0.6

    def buoy(x, y, heading_deg):
        return ExtendedTarget(torch.zeros(1, 3),
                              CurvedSurfaceScattering(BUOY_RADIUS, BUOY_RADIUS, learnable=False),
                              position=(x, y, buoy_depth), learnable=False)

    chain_pts = catenary(CHAIN_LENGTH, CHAIN_SCOPE, ex.WATER_DEPTH - buoy_depth - 0.4, CHAIN_POINTS)
    link_db = CHAIN_LINK_DB + 10 * math.log10(LINKS_PER_M * CHAIN_LENGTH / CHAIN_POINTS)

    def chain(x, y, heading_deg):
        # the chain hangs from the buoy along the world's CHAIN_DIRECTION: its
        # heading in the sonar frame is that direction less the ownship's
        return ExtendedTarget(chain_pts, IsotropicScattering(link_db, learnable=False),
                              position=(x, y, buoy_depth), yaw=heading_deg, learnable=False)

    s_verts, s_faces = box_mesh((SINKER, SINKER, SINKER))

    def sinker(x, y, heading_deg):
        return mesh_target(s_verts, s_faces, position=(x, y, ex.WATER_DEPTH - SINKER / 2),
                           yaw=heading_deg, n_patches=1, sound_speed=C, diffuse_db=-10.0,
                           learnable=False, facet_chunk=4096, checkpoint=False)

    # the kelp stand, as 26 builds it: plants on a jittered grid, eight points
    # each from the bottom to the surface, and an extinction along the
    # stand's own x in 2 m bins of one pattern each
    g = torch.Generator().manual_seed(21)
    n_along, n_across = int(KELP_STAND[0] / KELP_SPACING), int(KELP_STAND[1] / KELP_SPACING)
    u = (torch.arange(n_along) - (n_along - 1) / 2) * KELP_SPACING
    v = (torch.arange(n_across) - (n_across - 1) / 2) * KELP_SPACING
    U, V = torch.meshgrid(u, v, indexing="ij")
    plants = torch.stack([U, V], -1).reshape(-1, 2)
    plants = plants + 0.3 * KELP_SPACING * (2 * torch.rand(plants.shape, generator=g) - 1)
    depths = torch.linspace(ex.WATER_DEPTH - 0.5, 0.5, KELP_POINTS)
    kelp_offsets = torch.cat([plants.repeat_interleave(KELP_POINTS, 0),
                              depths.repeat(plants.shape[0]).unsqueeze(-1)], -1)
    kelp_offsets[:, 2] += 0.3 * (2 * torch.rand(kelp_offsets.shape[0], generator=g) - 1)
    kelp_offsets[:, 2] -= ex.WATER_DEPTH / 2
    into = (kelp_offsets[:, 0] + KELP_STAND[0] / 2).clamp_min(0.0)
    bins = (into / 2.0).floor().long()
    kelp_patterns = {int(b): IsotropicScattering(
        KELP_POINT_DB - 2.0 * KELP_EXTINCTION_DB_PER_M * (float(b) + 0.5) * 2.0, learnable=False)
        for b in bins.unique()}
    kelp_list = [kelp_patterns[int(b)] for b in bins]

    def kelp(x, y, heading_deg):
        return ExtendedTarget(kelp_offsets, kelp_list, position=(x, y, ex.WATER_DEPTH / 2),
                              yaw=heading_deg, learnable=False)

    sc = lambda x, y, h=None: ((x * S, y * S) if h is None else (x * S, y * S, h))
    boat_w = sc(*BOAT_WORLD)
    buoy_w = sc(*BUOY_WORLD) + (0.0,)
    cd = math.radians(CHAIN_DIRECTION_DEG)
    chain_w = (buoy_w[0], buoy_w[1], CHAIN_DIRECTION_DEG)
    sinker_w = (buoy_w[0] + CHAIN_SCOPE * math.cos(cd), buoy_w[1] + CHAIN_SCOPE * math.sin(cd), 0.0)
    kelp_w = sc(*KELP_WORLD)
    world = [(boat_w, boat), (buoy_w, buoy), (chain_w, chain), (sinker_w, sinker), (kelp_w, kelp)]
    print(f"  the boat moored at ({boat_w[0]:.0f}, {boat_w[1]:.0f}) m heading {boat_w[2]:.0f} deg; "
          f"the buoy at ({buoy_w[0]:.0f}, {buoy_w[1]:.0f}) m, its chain to port; the kelp stand "
          f"{KELP_STAND[0]:.0f} x {KELP_STAND[1]:.0f} m at ({kelp_w[0]:.0f}, {kelp_w[1]:.0f}) m, "
          f"{int(plants.shape[0])} plants")

    # ---- the sonar, as 21 has it, in a renderer ---------------------------- #
    dirs, _ = ex.transmit_fan(seed=ex.SEED)
    w_tx = ex.transmit_pattern(dirs)
    tilt = ex.passes_deg()[0]
    rx_beam = lambda d: ex.receive_beam(d, tilt)
    steer, bearings = azimuth_steering(ex.N_BEAMS, ex.SECTOR_DEG)
    grid = make_time_grid(2.0 * ex.NEAR / C, 2.0 * ex.FAR / C, ex.N_BINS)
    rng = grid * C / 2.0
    shading = shading_window(ex.N_RX, "hamming")
    noise = float(beam_noise_power(scene0.freqs_khz, bandwidth_hz=1.0 / ex.PULSE_S,
                                   directivity_db=line_array_directivity_db(ex.N_RX),
                                   wind_speed=ex.WIND))
    solid = (math.radians(2 * ex.SECTOR_DEG)
             * math.radians(ex.ELEV_DEG[1] - ex.ELEV_DEG[0]) / dirs.shape[0])
    span_y = ex.FAR * math.sin(math.radians(ex.SECTOR_DEG)) * 1.02
    x_range = (-0.03 * ex.FAR, 1.02 * ex.FAR)
    pixel_m = (x_range[1] - x_range[0]) / 299.0
    gain = []

    def display(noisy):
        if not gain:
            gain.append(ex.display_gain(noisy, rng, pixel_m=pixel_m))
        return ex.display(noisy, rng, pixel_m=pixel_m, gain=gain[0])[0]

    def to_cartesian(shown):
        return ex15.to_cartesian(shown, bearings, grid, n_x=300, n_y=300,
                                 x_range=x_range, y_range=(-span_y, span_y))

    renderer = PictureRenderer(
        scene0, elements=rx, directions=dirs, tx_weights=w_tx, tx_pattern=ex.transmit_pattern,
        rx_pattern=rx_beam, time_grid=grid, steer=steer, sigma_t=ex.PULSE_S, shading=shading,
        source_level_db=ex.SOURCE_LEVEL_DB, noise_power=noise,
        scattering=LambertScattering(-27.0, learnable=False), solid_angle_per_ray=solid,
        boundary="both", max_arrivals=ex.PATCHES, display=display, to_cartesian=to_cartesian,
        seed=ex.SEED)

    traj = track()
    times = torch.linspace(0.0, traj.duration, N_FRAMES).tolist()
    print(f"  {N_FRAMES} pings over {traj.duration:.0f} s at {SPEED * S:.1f} m/s, one every "
          f"{times[1] - times[0]:.1f} s; the ownship from (0, 0) heading 0 to "
          f"({float(traj.positions[-1, 0]):.0f}, {float(traj.positions[-1, 1]):.0f}) m heading "
          f"{float(traj.headings_deg[-1]):.0f} deg")

    # ---- the frames -------------------------------------------------------- #
    banner("the pings")
    beam_deg = ex.beam_3db_deg(ex.N_RX, shading)
    db = lambda t: 10.0 * torch.log10(t.detach().clamp_min(1e-30))
    frames, bares, poses, costs = [], [], [], []
    buoy_err, buoy_over, buoy_world, boat_err, kelp_db = [], [], [], [], []
    X = Y = B = R = None
    last = time.perf_counter()
    for k, (t, pose, (cart, gx, gy)) in enumerate(
            renderer.ownship_sequence(world, traj, times, scene_at=scene_at)):
        # the same ping without its targets: the background is cached, so
        # this is one beamformed picture more, and the noise draw is the same
        with torch.no_grad():
            bare, _, _ = renderer.picture([], frame=k)
        now = time.perf_counter()
        costs.append(now - last)
        last = now
        if X is None:
            X, Y = torch.meshgrid(gx, gy, indexing="xy")
            R = torch.hypot(X, Y)
            B = torch.rad2deg(torch.atan2(Y, X))
        cart_db, bare_db = db(cart), db(bare)
        frames.append(cart_db); bares.append(bare_db); poses.append(pose)
        # the buoy: the brightest cell within a beam of where the pose puts
        # it, and how far that stands over the 25 m disc around it (the chain
        # and the sinker are in that disc, so the peak is looked for locally)
        bx, by, _ = relative_pose(buoy_w, pose)
        rb = math.hypot(bx, by)
        window = (X - bx) ** 2 + (Y - by) ** 2 < (math.radians(beam_deg) * rb + 3.0) ** 2
        disc = (X - bx) ** 2 + (Y - by) ** 2 < 25.0 ** 2
        j = int((cart_db * window - 1e6 * (~window)).argmax())
        px, py = float(X.reshape(-1)[j]), float(Y.reshape(-1)[j])
        buoy_err.append(math.hypot(px - bx, py - by))
        buoy_over.append(float(cart_db.reshape(-1)[j] - cart_db[disc].median()))
        # ... and carried back into the world by the ownship pose
        c, s = math.cos(math.radians(pose[2])), math.sin(math.radians(pose[2]))
        buoy_world.append((pose[0] + c * px - s * py, pose[1] + s * px + c * py))
        # the boat: the centroid of the excess over the bare ping near it
        hx, hy, hh = relative_pose(boat_w, pose)
        disc = (X - hx) ** 2 + (Y - hy) ** 2 < 40.0 ** 2
        lit = (cart_db - bare_db).clamp_min(0.0) * disc * (cart_db > ex.THRESHOLD_DB)
        w = lit.sum()
        cx, cy = (float((lit * X).sum() / w), float((lit * Y).sum() / w)) if float(w) > 0 else (1e9, 1e9)
        boat_err.append(math.hypot(cx - hx, cy - hy))
        # the kelp: its cells over the same cells bare
        kx, ky, kh = relative_pose(kelp_w, pose)
        ck, sk = math.cos(math.radians(kh)), math.sin(math.radians(kh))
        along = ck * (X - kx) + sk * (Y - ky)
        across = -sk * (X - kx) + ck * (Y - ky)
        stand = (along.abs() < KELP_STAND[0] / 2) & (across.abs() < KELP_STAND[1] / 2)
        kelp_db.append(float(cart_db[stand].mean() - bare_db[stand].mean()))
        print(f"  t = {t:5.1f} s  ownship ({pose[0]:6.1f}, {pose[1]:6.1f}) m heading {pose[2]:5.1f} deg, "
              f"{costs[-1]:5.1f} s: buoy {buoy_err[-1]:4.1f} m off its place at {rb:.0f} m, "
              f"{buoy_over[-1]:+5.1f} dB over its surroundings; "
              f"boat centroid {boat_err[-1]:4.1f} m off; kelp {kelp_db[-1]:+5.1f} dB")

    # the same pose rendered twice: identical (the sea is a function of the pose)
    with torch.no_grad():
        renderer.set_scene(scene_at(*poses[-1]))
        again, _, _ = renderer.picture([], frame=len(poses) - 1)
    # correlate over the swath's cells only, at the display's floor: the
    # cells outside the sector are zero in every ping and would correlate
    # any two pictures at one
    swath = (R > ex.NEAR + 5.0) & (R < ex.FAR - 5.0) & (B.abs() < ex.SECTOR_DEG - 2.0)

    def corr(a, b):
        a = a[swath].clamp_min(ex.THRESHOLD_DB); b = b[swath].clamp_min(ex.THRESHOLD_DB)
        return float(torch.corrcoef(torch.stack([a - a.mean(), b - b.mean()]))[0, 1])
    self_corr = corr(bares[-1], db(again))
    step_corr = [corr(bares[i], bares[i + 1]) for i in range(len(bares) - 1)]
    bw = lambda r: math.radians(beam_deg) * r
    tol_boat = [0.5 * ex.HULL_LENGTH + bw(math.hypot(*relative_pose(boat_w, p)[:2])) for p in poses]
    bwx = torch.tensor([p[0] for p in buoy_world]); bwy = torch.tensor([p[1] for p in buoy_world])
    # RMS over the pings: the buoy's blob is a beam wide, and its peak wanders
    # within it as the multipath fringes shift with the range, so one ping can
    # sit a beam off and the twelve together still say where the buoy is
    scatter = float(torch.hypot(bwx - bwx.mean(), bwy - bwy.mean()).pow(2).mean().sqrt())
    mean_range = sum(math.hypot(*relative_pose(buoy_w, p)[:2]) for p in poses) / len(poses)
    med_kelp = sorted(kelp_db)[len(kelp_db) // 2]
    print(f"  {len(frames)} pings, {sum(costs) / len(costs):.1f} s each (a whole picture: trace, "
          f"reverberation, five echoes, two beamformed pictures)")
    print(f"  the buoy carried back into the world: ({bwx.mean():.1f}, {bwy.mean():.1f}) m against "
          f"({buoy_w[0]:.1f}, {buoy_w[1]:.1f}), RMS scatter {scatter:.1f} m over the pings, a beam "
          f"{bw(mean_range):.1f} m at its mean range of {mean_range:.0f} m")
    print(f"  consecutive bare pings correlate at {min(step_corr):+.2f} to {max(step_corr):+.2f}; "
          f"the same pose twice at {self_corr:+.3f}")

    banner("acceptance")
    ok = True
    ok &= check("the buoy is where the ownship pose puts it in every ping",
                min(buoy_over) > 10.0,
                f"{min(buoy_over):+.1f} dB over its surroundings at least, within "
                f"{max(buoy_err):.1f} m of its place")
    # the mean may sit a beam plus the buoy's radius off: the peak is the
    # sphere's specular point on its near face, not its centre
    ok &= check("carried back into the world, the buoy stands still",
                scatter < bw(mean_range) and
                math.hypot(float(bwx.mean()) - buoy_w[0], float(bwy.mean()) - buoy_w[1])
                < bw(mean_range) + BUOY_RADIUS,
                f"RMS scatter {scatter:.1f} m, the mean "
                f"{math.hypot(float(bwx.mean()) - buoy_w[0], float(bwy.mean()) - buoy_w[1]):.1f} m "
                f"from the truth, a beam {bw(mean_range):.1f} m")
    ok &= check("the moored boat's echo is at the boat in every ping",
                all(e < t for e, t in zip(boat_err, tol_boat)),
                f"centroid off by {max(boat_err):.1f} m at most, tolerance {min(tol_boat):.1f} m")
    ok &= check("the kelp stand reads over its cells", med_kelp > 4.0,
                f"{med_kelp:+.1f} dB in the median ping")
    ok &= check("the sea is rendered anew at each pose",
                max(step_corr) < 0.5 and self_corr > 0.999,
                f"consecutive pings {max(step_corr):+.2f} at most, the same pose twice {self_corr:+.3f}")

    # ---- the figures ------------------------------------------------------- #
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    ref = float(max(f.max() for f in bares))
    ext = [float(gx.min()), float(gx.max()), float(gy.min()), float(gy.max())]

    def overlays(ax, pose):
        """Where the pose puts each target, in the sonar's frame."""
        bx, by, _ = relative_pose(buoy_w, pose)
        hx, hy, hh = relative_pose(boat_w, pose)
        kx, ky, kh = relative_pose(kelp_w, pose)
        ax.plot(bx, by, "co", ms=8, mfc="none", mew=1.2)
        h = math.radians(hh)
        ax.plot([hx - 15 * math.cos(h), hx + 15 * math.cos(h)],
                [hy - 15 * math.sin(h), hy + 15 * math.sin(h)], "c-", lw=1.0)
        ck, sk = math.cos(math.radians(kh)), math.sin(math.radians(kh))
        a, b = KELP_STAND[0] / 2, KELP_STAND[1] / 2
        corners = [(a, b), (-a, b), (-a, -b), (a, -b), (a, b)]
        ax.plot([kx + ck * u - sk * v for u, v in corners], [ky + sk * u + ck * v for u, v in corners],
                "c--", lw=0.8)

    fig, ax = plt.subplots(figsize=(9, 7.5))
    im = ax.imshow(frames[0].numpy(), origin="lower", extent=ext, vmin=ex.THRESHOLD_DB, vmax=ref,
                   cmap="inferno", aspect="equal")
    ax.set_xlabel("forward (m)"); ax.set_ylabel("across (m)")
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
    fig.colorbar(im, ax=ax, fraction=0.04, label="dB re the background at that range")
    fig.suptitle(f"{ex.FREQ_KHZ:.0f} kHz FLS, {2 * ex.SECTOR_DEG:.0f} deg to {ex.FAR:.0f} m: "
                 f"the sonar under way, the world still")
    marks = []

    def draw(i):
        im.set_data(frames[i].numpy())
        for m in marks:
            m.remove()
        marks.clear()
        n0 = len(ax.lines)
        overlays(ax, poses[i])
        marks.extend(ax.lines[n0:])
        x, y, h = poses[i]
        ax.set_title(f"t = {times[i]:.0f} s: ownship at ({x:.0f}, {y:.0f}) m heading {h:.0f} deg")
        return [im, *marks]

    anim = animation.FuncAnimation(fig, draw, frames=len(frames), interval=300, blit=False)
    gif = FIGURE_DIR / f"29_ownship{ex.TAG}.gif"
    with timed("  gif"):
        anim.save(gif, writer=animation.PillowWriter(fps=3))
    print(f"  wrote {gif}")
    plt.close(fig)

    pick = [int(round(i)) for i in torch.linspace(0, len(frames) - 1, 5).tolist()]
    fig, axes = plt.subplots(2, 3, figsize=(17, 9.5))
    for ax, i in zip(axes.ravel()[:5], pick):
        x, y, h = poses[i]
        ax.imshow(frames[i].numpy(), origin="lower", extent=ext, vmin=ex.THRESHOLD_DB, vmax=ref,
                  cmap="inferno", aspect="equal")
        overlays(ax, poses[i])
        ax.set_title(f"t = {times[i]:.0f} s: ownship ({x:.0f}, {y:.0f}) m, heading {h:.0f} deg")
        ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
    # the world: the track, the targets, and where the buoy was seen from each ping
    ax = axes.ravel()[5]
    ax.plot(traj.positions[:, 0].numpy(), traj.positions[:, 1].numpy(), "k-", lw=1, label="the track")
    for p in poses:
        ax.plot(p[0], p[1], "k.", ms=4)
    ax.plot(buoy_w[0], buoy_w[1], "bo", mfc="none", label="buoy"); ax.plot(bwx, bwy, "b.", ms=3)
    h = math.radians(boat_w[2])
    ax.plot([boat_w[0] - 15 * math.cos(h), boat_w[0] + 15 * math.cos(h)],
            [boat_w[1] - 15 * math.sin(h), boat_w[1] + 15 * math.sin(h)], "r-", lw=2, label="boat")
    ck, sk = math.cos(math.radians(kelp_w[2])), math.sin(math.radians(kelp_w[2]))
    a, b = KELP_STAND[0] / 2, KELP_STAND[1] / 2
    corners = [(a, b), (-a, b), (-a, -b), (a, -b), (a, b)]
    ax.plot([kelp_w[0] + ck * u - sk * v for u, v in corners],
            [kelp_w[1] + sk * u + ck * v for u, v in corners], "g--", label="kelp")
    ax.set_aspect("equal"); ax.legend(fontsize=8); ax.set_title("the world, and the buoy seen from each ping")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    fig.suptitle(f"{ex.FREQ_KHZ:.0f} kHz FLS to {ex.FAR:.0f} m: the sonar under way, "
                 f"five of {len(frames)} pings")
    save(fig, f"29_ownship{ex.TAG}.png")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
