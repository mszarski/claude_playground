"""An AUV's forward-looking sonar: boat, wind sea, seabed -- imaged in Cartesian.

The scene, which is the one you would actually be flying:

  * a 100 kHz FLS on an **AUV at 18 m** in 30 m of water, looking forward and
    slightly up, with a Mills-cross array (64 receive x 6 transmit);
  * a 12 m boat on the surface at 55 m, quartering across the bow;
  * a Pierson-Moskowitz wind sea overhead, and a rough sand seabed below;
  * the boat as a **triangle mesh**, so its aspect dependence comes from the
    geometry rather than from a chosen primitive.

The output is a **Cartesian** image -- metres east by metres north, the sonar at
the origin -- rather than the bearing-range rectangle the earlier examples plot.
That matters for more than looks: in bearing-range coordinates a straight
seabed ridge is a curve and a 12 m boat is a bearing extent that shrinks with
range, so you cannot read geometry off it directly.  Resampled onto a grid in
metres, the scene is laid out the way it really is.

Acceptance criteria:
  * the boat appears in the Cartesian image within a beamwidth of where it is;
  * its along-track extent in metres is consistent with a 12 m hull, which is
    the thing a Cartesian image is *for*;
  * the seabed and sea surface are *in* the image as reverberation, not
    scenery, and the boat's contrast against them is what decides detection;
  * a loss on the Cartesian image still carries gradients to the scene.
"""

from __future__ import annotations

import importlib.util
import math
import time
from pathlib import Path

import torch

from _common import banner, check, save, setup, timed
from hydropt import (
    azimuth_steering, beamform, line_array_factor, make_time_grid,
    shading_window, target_arrivals,
)
from hydropt.beamform import ArrivalSet
from hydropt.launch import fibonacci_cone
from hydropt.mesh import boat_hull_mesh, mesh_target
from hydropt.reverb import (
    LambertScattering, cone_solid_angle, reverberation_arrivals,
)
from hydropt.tracer import trace

C = 1500.0
FREQ_KHZ = 100.0
WATER_DEPTH = 30.0
AUV_DEPTH = 18.0            # the sonar is subsea, on the vehicle
WIND = 5.0
BOAT_RANGE = 55.0
BOAT_BEARING_DEG = 12.0     # off to starboard
BOAT_HEADING_DEG = 55.0     # quartering across our bow
BOAT_DRAUGHT = 1.0
HULL_LENGTH, HULL_BEAM = 12.0, 3.2

SECTOR_DEG = 60.0           # +/- 60 = 120 deg swath
N_RX, N_TX = 64, 6
TILT_DEG = 12.0             # the fan is tilted UP toward the surface
# Pulse length sets range resolution: 1.2e-4 s is 9 cm, which is a realistic
# short FLS pulse and a good deal coarser than the 2 cm the earlier examples
# used.  It matters for the picture as well as the physics -- a pulse far
# shorter than the display's cell size shows the patch sampling rather than
# the seabed.
PULSE_S = 1.2e-4


def _mills():
    path = Path(__file__).resolve().parent / "13_mills_cross_fls.py"
    spec = importlib.util.spec_from_file_location("_mc13", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def transmit_fan(n_elev: int = 56, n_azim: int = 330, *, seed: int = 0):
    """A wide azimuth swath, narrow in elevation, tilted up at the surface.

    Wide enough in elevation to reach BOTH the surface above and the seabed
    below: from 12 m of altitude the bottom sits 8-22 deg down at these ranges,
    so a fan that only looks up never illuminates it and the seabed is missing
    from the image -- and from its gradient -- for purely geometric reasons.

    **Jittered on purpose.**  A regular lattice in (elevation, azimuth) puts
    every bottom bounce on a regular lattice too, and the reverberation then
    images as a set of clean concentric arcs -- the sampling pattern, not the
    seabed.  Real reverberation is speckle.  Displacing each ray randomly
    within its own cell breaks the lattice while keeping the fan's density, so
    the patches fill the swath instead of ruling it.
    """
    g = torch.Generator().manual_seed(seed)
    e0, e1 = -math.radians(20.0), math.radians(26.0)
    a0, a1 = -math.radians(SECTOR_DEG), math.radians(SECTOR_DEG)
    el = torch.linspace(e0, e1, n_elev)
    az = torch.linspace(a0, a1, n_azim)
    E, A = torch.meshgrid(el, az, indexing="ij")
    E, A = E.reshape(-1), A.reshape(-1)
    de = (e1 - e0) / max(n_elev - 1, 1)
    da = (a1 - a0) / max(n_azim - 1, 1)
    E = E + (torch.rand(E.shape, generator=g, dtype=E.dtype) - 0.5) * de
    A = A + (torch.rand(A.shape, generator=g, dtype=A.dtype) - 0.5) * da
    dirs = torch.stack([E.cos() * A.cos(), E.cos() * A.sin(), E.sin()], dim=-1)
    weights = line_array_factor(torch.sin(E), N_TX,
                                sin_steer=math.sin(math.radians(-TILT_DEG)))
    return dirs, weights


def to_cartesian(image, bearings, grid, *, n_x: int = 240, n_y: int = 240,
                 x_range=(-7.0, 90.0), y_range=(-70.0, 70.0)):
    """Resample a [bearing, range] image onto a metric grid, differentiably.

    Bilinear in (bearing, range), so the result carries gradient back to the
    beamformed image and through it to the whole scene.  Returns
    ``(cart [n_y, n_x], x [n_x], y [n_y])`` with ``x`` forward (north) and ``y``
    to starboard (east), the sonar at the origin.
    """
    power = image[:, 0]                                   # [bearings, time]
    rng = grid * C / 2.0
    x = torch.linspace(*x_range, n_x, dtype=power.dtype)
    y = torch.linspace(*y_range, n_y, dtype=power.dtype)
    X, Y = torch.meshgrid(x, y, indexing="xy")            # [n_y, n_x]
    R = torch.hypot(X, Y)
    B = torch.rad2deg(torch.atan2(Y, X))

    # `grid_sample` wants normalised [-1, 1] coordinates, x = last axis.
    b0, b1 = float(bearings[0]), float(bearings[-1])
    r0, r1 = float(rng[0]), float(rng[-1])
    gb = 2.0 * (B - b0) / (b1 - b0) - 1.0
    gr = 2.0 * (R - r0) / (r1 - r0) - 1.0
    samples = torch.stack([gr, gb], dim=-1).unsqueeze(0)  # [1, n_y, n_x, 2]
    src = power.unsqueeze(0).unsqueeze(0)                 # [1, 1, bearings, time]
    out = torch.nn.functional.grid_sample(src, samples, mode="bilinear",
                                          padding_mode="zeros",
                                          align_corners=True)
    return out.reshape(X.shape), x, y


def main() -> int:
    setup()
    banner("15 -- an AUV's FLS: boat, wind sea, seabed, imaged in Cartesian")
    mc = _mills()
    fls = mc._fls

    rx = mc.horizontal_array()
    scene, bottom, surface, sediment = fls.build_scene(rx)
    # Put the sonar on the vehicle, deeper than examples/12's 12 m.
    scene.source = torch.tensor([0.0, 0.0, AUV_DEPTH])
    print(f"  {FREQ_KHZ:.0f} kHz, {N_RX} receive x {N_TX} transmit (Mills cross)")
    print(f"  AUV at {AUV_DEPTH:.0f} m in {WATER_DEPTH:.0f} m of water -- "
          f"{WATER_DEPTH - AUV_DEPTH:.0f} m of altitude")
    print(f"  {SECTOR_DEG * 2:.0f} deg swath, fan tilted {TILT_DEG:.0f} deg up")
    print(f"  wind {WIND:.0f} m/s over the surface, sand seabed")

    verts, faces = boat_hull_mesh(HULL_LENGTH, HULL_BEAM, BOAT_DRAUGHT,
                                  n_long=110, n_around=34)
    b = math.radians(BOAT_BEARING_DEG)
    boat = mesh_target(
        verts, faces,
        position=(BOAT_RANGE * math.cos(b), BOAT_RANGE * math.sin(b), BOAT_DRAUGHT),
        yaw=BOAT_HEADING_DEG, n_patches=6, sound_speed=C,
        learnable=True, learnable_shape=False, facet_chunk=256)
    print(f"  boat: {HULL_LENGTH:.0f} m hull as {faces.shape[0]} facets in "
          f"{boat.n_highlights} patches,")
    print(f"        at {BOAT_RANGE:.0f} m, bearing {BOAT_BEARING_DEG:+.0f} deg, "
          f"heading {BOAT_HEADING_DEG:.0f} deg")
    depression = math.degrees(math.atan2(AUV_DEPTH - BOAT_DRAUGHT, BOAT_RANGE))
    print(f"        we look up at it by {depression:.1f} deg")

    banner("ping")

    seabed_backscatter = LambertScattering(-27.0, learnable=True)

    def reverb(bundle, dirs, weights, n_patches, seed=11):
        solid = (math.radians(2 * SECTOR_DEG) * math.radians(46.0)
                 / dirs.shape[0])
        return reverberation_arrivals(
            bundle, dirs, scene.freqs_khz,
            scattering=seabed_backscatter,
            solid_angle_per_ray=solid, ray_weights=weights,
            boundary="both", surface=scene.surface, bottom=scene.bottom,
            max_arrivals=n_patches,
            generator=torch.Generator().manual_seed(seed))

    steer, bearings = azimuth_steering(181, SECTOR_DEG)
    grid = make_time_grid(2.0 * 8.0 / C, 2.0 * 95.0 / C, 420)

    def render(arr, st=None, gr=None):
        return beamform(arr, rx, scene.freqs_khz,
                        grid if gr is None else gr,
                        steer if st is None else st, sigma_t=PULSE_S,
                        shading=shading_window(N_RX, "hamming"), steer_chunk=8)

    def combine(a, c):
        return ArrivalSet(*(None if a[i] is None or c[i] is None
                            else torch.cat([a[i], c[i]], dim=0)
                            for i in range(len(a))))

    def ping(n_elev, n_azim, n_patches, n_rx_rays, cap, seed):
        dirs, weights = transmit_fan(n_elev, n_azim, seed=seed)
        echo = target_arrivals(scene, boat, dirs, n_rx_rays=n_rx_rays,
                               rx_half_angle_deg=45.0, tx_weights=weights,
                               max_arrivals_per_leg=cap,
                               generator=torch.Generator().manual_seed(seed))
        # The seabed and the sea surface are not scenery: every bottom and
        # surface bounce in the transmit fan is a scattering patch with its own
        # bearing and range, so reverberation beamforms exactly like an echo
        # and sums with it *before* beamforming.  Without this the boat would
        # sit on a black background, which is neither what a sonar shows nor
        # what sets detection.
        rev = reverb(trace(scene, dirs), dirs, weights, n_patches, seed=seed + 1)
        return echo, rev, dirs

    # The picture wants as many patches as it can get: each covers about a
    # beamwidth by a pulse length, so a few thousand over a 120 deg x 87 m
    # field leaves most of the image empty and the seabed reads as scattered
    # dots rather than a bottom.
    #
    # What limits it is not the patches but autograd: `beamform` keeps an
    # [arrivals, steer, time] intermediate alive for the backward pass -- that
    # is arrivals x bearings x time whatever the chunk size -- and the trace
    # retains its own graph.  Under `no_grad` neither cost is paid, so the
    # displayed ping is dense, and differentiability is demonstrated further
    # down on the same code path at a size that fits in memory.
    t0 = time.perf_counter()
    with torch.no_grad():
        with timed("  dense ping for the picture"):
            echo, rev, dirs = ping(56, 330, 40000, 420, 24, 5)
            image = render(combine(echo, rev))
            echo_img = render(echo)
    forward = time.perf_counter() - t0
    print(f"  {dirs.shape[0]} transmit rays -> {echo.n_arrivals} target "
          f"arrivals + {rev.n_arrivals} reverberation patches")
    print(f"  image {tuple(image.shape)} = {len(bearings)} bearings x "
          f"{len(grid)} range bins")

    banner("the same ping, on a grid in metres")
    with timed("  resample to Cartesian"):
        cart, gx, gy = to_cartesian(image, bearings, grid)
    print(f"  {cart.shape[1]} x {cart.shape[0]} cells over "
          f"{float(gx[0]):.0f}..{float(gx[-1]):.0f} m forward, "
          f"{float(gy[0]):.0f}..{float(gy[-1]):.0f} m across")
    beam_m = BOAT_RANGE * math.radians(2.0 * math.degrees(math.asin(2.0 / N_RX)))
    print(f"  cell {float(gx[1] - gx[0]):.2f} x {float(gy[1] - gy[0]):.2f} m, "
          f"against {beam_m:.1f} m of beamwidth and "
          f"{PULSE_S * C / 2:.2f} m of range resolution at the boat")
    print(f"  (a display much finer than the sonar's own resolution shows the")
    print(f"   patch sampling rather than the seabed)")

    # The boat's own extent has to be measured on the echo alone: with the
    # seabed lit, everything within 18 m of the boat is above any fixed
    # threshold, so the combined image measures reverberation, not the hull.
    with torch.no_grad():
        echo_cart, _, _ = to_cartesian(echo_img, bearings, grid)

    det = cart.detach()
    flat = int(echo_cart.reshape(-1).argmax())
    px = float(gx[flat % cart.shape[1]])
    py = float(gy[flat // cart.shape[1]])
    tx = BOAT_RANGE * math.cos(b)
    ty = BOAT_RANGE * math.sin(b)
    beamwidth = 2.0 * math.degrees(math.asin(1.0 / (N_RX / 2.0)))
    tol = BOAT_RANGE * math.radians(beamwidth)
    # Distance to the nearest part of the hull, not to its centre: the peak of
    # an extended target lands on whichever part is glinting, which for a 12 m
    # boat is up to 6 m from the middle.  Asking it to land on the centre would
    # be asking the wrong question, as examples/13 found.
    world = boat.world_positions().detach()
    # Distance to the hull's axis, not to the nearest patch centroid: the
    # patches are six points along a 12 m boat, so nearest-centroid overstates
    # how far the peak really is from the body.
    p0, p1 = world[0, :2], world[-1, :2]
    seg = p1 - p0
    t = (((torch.tensor([px, py], dtype=seg.dtype) - p0) * seg).sum()
         / (seg * seg).sum()).clamp(0.0, 1.0)
    axis_dist = float((p0 + t * seg
                       - torch.tensor([px, py], dtype=seg.dtype)).norm())
    # ...and then outside the hull's own footprint: the boat is HULL_BEAM wide,
    # so a point on its side is already half a beam off the axis.
    err = max(0.0, axis_dist - HULL_BEAM / 2.0)
    centre_err = math.hypot(px - tx, py - ty)
    print(f"\n  the boat's echo peaks at    ({px:+.1f}, {py:+.1f}) m")
    print(f"  the boat's centre is at    ({tx:+.1f}, {ty:+.1f}) m")
    print(f"  {centre_err:.1f} m from the centre of a {HULL_LENGTH:.0f} m hull, "
          f"{axis_dist:.1f} m from its centreline,")
    print(f"  {err:.1f} m outside the hull's {HULL_BEAM:.1f} m beam")
    print(f"  (beamwidth is {tol:.1f} m at that range)")

    # extent of the boat's own return, in metres
    near = echo_cart.clone()
    keep = (torch.hypot(torch.as_tensor(gx).reshape(1, -1) - tx,
                        torch.as_tensor(gy).reshape(-1, 1) - ty) < 18.0)
    near = torch.where(keep, near, torch.zeros_like(near))
    lit = near > near.max() * 10 ** (-1.0)          # within 10 dB of the peak
    ys, xs = torch.nonzero(lit, as_tuple=True)
    span = math.hypot(float(gx[xs.max()] - gx[xs.min()]),
                      float(gy[ys.max()] - gy[ys.min()]))
    print(f"  the boat's return spans {span:.1f} m at -10 dB "
          f"(hull is {HULL_LENGTH:.0f} m)")

    # What actually decides detection: the boat against the reverberation
    # around it, not against nothing.
    GX = torch.as_tensor(gx).reshape(1, -1).expand_as(det)
    GY = torch.as_tensor(gy).reshape(-1, 1).expand_as(det)
    rng_cell = torch.hypot(GX, GY)
    brg_cell = torch.rad2deg(torch.atan2(GY, GX))
    ring = torch.hypot(GX - tx, GY - ty)
    # Reverberation competes with a target at its OWN range: the return falls
    # as r^-5, so a background taken from a ring in the ground plane is mostly
    # sampling longer ranges and flatters the target by tens of dB.  The
    # comparison that means anything is same range, different bearing.  Cells
    # outside the sector are exact zeros from the resampling's padding, so
    # they are excluded too.
    same_range = (rng_cell - BOAT_RANGE).abs() < 4.0
    off_target = (brg_cell - BOAT_BEARING_DEG).abs() > 10.0
    inside = brg_cell.abs() < SECTOR_DEG - 4.0
    background = det[same_range & off_target & inside]
    on_target = det[ring < 6.0].max()
    # Against the MEAN, not the median: reverberation is a speckle field, so
    # its median sits far below the level a detector actually competes with.
    srr = 10 * math.log10(float(on_target / background.mean().clamp_min(1e-30)))
    brightest = det.reshape(-1).argmax()
    bx = float(gx[int(brightest) % cart.shape[1]])
    by = float(gy[int(brightest) // cart.shape[1]])
    print(f"\n  signal-to-reverberation at the boat: {srr:+.1f} dB")
    print(f"  (against the mean of {int(background.numel())} cells at the same "
          f"range; the median sits\n   "
          f"{10 * math.log10(float(background.mean() / background.median().clamp_min(1e-30))):.0f} dB "
          f"lower, which is what a speckle field does)")
    boat_is_brightest = math.hypot(bx - tx, by - ty) < 8.0
    print(f"  the brightest cell in the whole image is at ({bx:+.1f}, {by:+.1f}) m,")
    if boat_is_brightest:
        print(f"  which is the boat -- it wins here, but only by {srr:.0f} dB over")
        print(f"  the reverberation at its own range.  Drop the target strength")
        print(f"  or raise the sea state and the seabed takes the peak.")
    else:
        print(f"  which is seabed, not the boat: a target does not have to be the")
        print(f"  loudest thing in the picture, and often is not.")

    seabed_power = float(det[:, gx < 35.0].max() / det.max())
    print(f"  seabed return in the near field: "
          f"{10 * math.log10(max(seabed_power, 1e-30)):+.1f} dB re peak")

    banner("still differentiable, through the resampling")
    print("  Same code path, a coarser fan and fewer patches, gradients on.")
    print("  The size is a memory limit in `beamform` and in the trace's own")
    print("  graph, not a limit of the method.")
    t0 = time.perf_counter()
    # cap 24, not 10: with a splat sized to the fan many rays pass within
    # it along much the same path, so a tight cap spends its budget on
    # direct-path near-duplicates and drops the bottom-bounced arrivals --
    # which is where the sediment's gradient lives.  Reverberation barely
    # constrains the sediment on its own: a first-bounce patch never passes
    # through a bottom *reflection*, only the backscatter strength below.
    # Its own coarser beamformer grid as well as a coarser fan: `beamform`
    # retains [arrivals, steer, time] for the backward pass, so at the display
    # grid's 181 bearings x 420 bins this alone would want several GB per
    # intermediate on top of the 5 GB the dense ping already peaked at.
    steer_g, brg_g = azimuth_steering(91, SECTOR_DEG)
    grid_g = make_time_grid(2.0 * 8.0 / C, 2.0 * 95.0 / C, 200)
    echo_g, rev_g, dirs_g = ping(16, 90, 900, 220, 24, 21)
    cart_g, _, _ = to_cartesian(render(combine(echo_g, rev_g), steer_g, grid_g),
                                brg_g, grid_g, n_x=120, n_y=120)
    cart_g.sum().backward()
    backward = time.perf_counter() - t0
    live = {"boat position": boat.position, "boat heading": boat.orientation,
            "seabed": bottom.heights, "waves": surface.heights,
            "sediment c2": sediment.c2,
            "seabed backscatter": seabed_backscatter.strength_db}
    states = {k: (p.grad is not None and torch.isfinite(p.grad).all()
                  and float(p.grad.abs().sum()) > 0) for k, p in live.items()}
    for name, ok_g in states.items():
        print(f"  d(cartesian image)/d({name:<14s}): {'OK' if ok_g else 'ZERO'}")
    print(f"\n  {dirs_g.shape[0]} rays, {echo_g.n_arrivals + rev_g.n_arrivals} "
          f"arrivals: {backward:.1f} s for forward and backward together")
    print(f"  (the dense picture above took {forward:.1f} s with no graph kept)")
    print(f"  The resampling is bilinear and differentiable, so a loss written")
    print(f"  in metres -- where a target is, how long it is -- reaches the")
    print(f"  scene just as one written on the bearing-range image does.")

    save(_plot(cart, gx, gy, image, bearings, grid, boat, verts, faces, tx, ty,
               echo_cart), "15_auv_cartesian.png")

    banner("acceptance")
    ok = check("the boat's echo lands on the boat, within a beamwidth",
               err < tol,
               f"{err:.1f} m outside the hull against {tol:.1f} m of beamwidth")
    ok &= check("its return is the size of a boat, not a point",
                4.0 < span < 3.0 * HULL_LENGTH,
                f"{span:.1f} m at -10 dB, hull {HULL_LENGTH:.0f} m")
    ok &= check("the seabed and surface fill the image, not a black background",
                seabed_power > 1e-3,
                f"near-field reverberation "
                f"{10 * math.log10(max(seabed_power, 1e-30)):+.1f} dB re peak")
    ok &= check("the boat stands above the reverberation around it",
                srr > 3.0, f"{srr:+.1f} dB signal-to-reverberation")
    ok &= check("the Cartesian image is still differentiable end to end",
                all(states.values()),
                f"{sum(states.values())}/{len(states)} live")
    return 0 if ok else 1


def _plot(cart, gx, gy, image, bearings, grid, boat, verts, faces, tx, ty,
          echo_cart):
    import matplotlib.pyplot as plt
    import numpy as np
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=(13.5, 9.6))

    ax = fig.add_subplot(2, 2, 1, projection="3d")
    v = verts.detach().numpy()
    tris = v[faces.numpy()]
    # Shade by facet normal so the form reads: a flat colour on a 3-D surface
    # looks like a silhouette, and subsampling the facets looks like stripes.
    nrm = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-30
    shade = 0.35 + 0.65 * np.clip(nrm @ np.array([0.3, -0.5, -0.8]), 0, 1)
    ax.add_collection3d(Poly3DCollection(tris, facecolors=plt.cm.Blues(0.3 + 0.5 * shade),
                                         edgecolor="none"))
    ax.set_xlim(-6.5, 6.5); ax.set_ylim(-3.4, 3.4); ax.set_zlim(1.6, -0.6)
    ax.set_box_aspect((13, 6.8, 3.0)); ax.view_init(elev=24, azim=-62)
    ax.set_title("the hull: transom aft (left), fine entry forward", fontsize=10)
    ax.set_xlabel("x fwd (m)", fontsize=7, labelpad=-4)
    ax.set_ylabel("y (m)", fontsize=7, labelpad=-4)
    ax.set_zlabel("depth (m)", fontsize=7, labelpad=-6)
    ax.tick_params(labelsize=5.5, pad=-2)

    def db(a, floor=1e-4):
        a = np.asarray(a)
        return 10 * np.log10(np.maximum(a, a.max() * floor) / a.max())

    ax2 = fig.add_subplot(2, 2, 2)
    m2 = ax2.pcolormesh(grid.detach().numpy() * C / 2.0, bearings.numpy(),
                        db(image[:, 0].detach().numpy()), cmap="inferno",
                        vmin=-22, vmax=0, shading="auto")
    ax2.set_xlabel("range (m)", fontsize=9)
    ax2.set_ylabel("bearing (deg)", fontsize=9)
    ax2.set_title("as beamformed: bearing x range", fontsize=10)
    fig.colorbar(m2, ax=ax2, label="dB re peak")

    def cart_panel(axis, field, title, mark_boat):
        d = db(field)
        mm = axis.pcolormesh(gy.numpy(), gx.numpy(), d.T, cmap="inferno",
                             vmin=-22, vmax=0, shading="auto")
        axis.plot([0], [0], "^", color="#5ff0c0", ms=10, mec="k", mew=0.6)
        axis.annotate("AUV", (0, 0), color="#5ff0c0", fontsize=8,
                      xytext=(6, 4), textcoords="offset points")
        if mark_boat:
            w = boat.world_positions().detach().numpy()
            axis.plot(w[:, 1], w[:, 0], "-", color="#7fdfff", lw=2.0, alpha=0.9)
            axis.add_patch(plt.Circle((ty, tx), 11.0, fill=False,
                                      ec="#7fdfff", lw=1.0, ls="--"))
            axis.annotate("boat", (ty, tx), color="#7fdfff", fontsize=9,
                          xytext=(13, 9), textcoords="offset points")
        axis.set_aspect("equal")
        axis.set_xlabel("across track (m)", fontsize=9)
        axis.set_ylabel("along track (m)", fontsize=9)
        axis.set_title(title, fontsize=10)
        return mm

    ax3 = fig.add_subplot(2, 2, 3)
    m3 = cart_panel(ax3, cart.detach().numpy(),
                    "the ping in metres: boat, sea surface and seabed", True)
    fig.colorbar(m3, ax=ax3, label="dB re peak")

    ax4 = fig.add_subplot(2, 2, 4)
    m4 = cart_panel(ax4, echo_cart.numpy(),
                    "the boat's echo alone -- what the reverberation hides", True)
    fig.colorbar(m4, ax=ax4, label="dB re peak")

    fig.tight_layout()
    return fig


if __name__ == "__main__":
    raise SystemExit(main())
