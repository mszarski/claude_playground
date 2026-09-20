"""The full inverse on a picture with a boat in it: the scene, then the boat.

``examples/30`` fitted the scene to a bare picture; ``examples/22`` fitted a
boat to a picture whose background was the model's own.  A real picture is
both at once: a background the model has to fit first, and a boat on it
whose speckle, ghosts and background the model never shares.  This example
runs that pipeline on such a picture, loaded from a file, and says how far
each stage gets:

1. **the scene** (``hydropt.SceneFit``, as 30): seabed, surface, tilt, gain
   and noise from the swath's level against range.  The boat is in the
   picture while this runs, and the observable is a median over 60 beams,
   which one boat cannot move -- measured below: the fitted values are as
   good as 30's without a boat;
2. **the detection**: the real picture over its own level against range
   (the median over every beam, in dB) is an excess map; its brightest
   cell is the contact, in range and bearing, and the fitted scene as a
   :class:`~hydropt.PictureRenderer` (``SceneFit.renderer``) is where the
   boat is put;
3. **the heading**, read off the echo and then searched, never descended
   (``examples/16`` measured why: the loss against yaw ripples at a tenth
   of a degree).  The excess above 10 dB round the detection is the hull's
   echo, elongated along the hull at either head (a beam across at
   120 kHz, a row of beads at 330), so its centroid is the hull's centre --
   better than the peak, which is one bead and at a fine head an end of
   the hull -- and its principal axis is the heading to a hull's 180
   degree ambiguity.  Measured: 53.7 degrees for a hull heading 55.  Then
   the last 1.3 degrees, which matter more than they look: a hull patch
   5 m long glints through a lobe ``lambda / 5 m`` wide (0.14 degrees at
   120 kHz), so the model at 53.7 has different patches lit than the real
   hull at 55, its picture is wrong by a factor of four in the loss, and
   the position it descends to is 3 m off where the same descent at
   55.0 reaches half a metre.  ``examples/16`` measured this as the
   ripple of the loss against yaw at a tenth of a degree; here it is put
   to use.  After the coarse position stages the heading is SCANNED at
   half a lobe's step over 1.5 degrees either side of the axis, and the
   deep, narrow minimum is the heading.  Coarser searches were tried and
   are worse, all measured: a grid at 15 degree steps scored at the
   detection read 75 for 55, and 5 and 2.5 degree steps about the axis
   walked 53.7 to 64 to 71 -- steps wider than the lobe see only the
   ripple's envelope, whose minimum is elsewhere;
4. **the position**, by 22's descent on the fitted background: the model's
   *incoherent* picture (its expected intensity), the display gain frozen
   from the real picture, a blur schedule from 8 m to 0.5 m -- and, new
   here, the loss restricted to a disc round the detection (half a hull
   plus a beam plus 5 m, so the hull is inside it whichever way it lies,
   at either head), so that the real sea's speckle outside the boat does
   not count.  Inside it the real and model backgrounds still differ
   (different realisations, ~1 dB in the sector median and 5.6 dB cell by
   cell), and that is the floor the fit lands on.  The same descent with
   the true heading is run for reference, to say what the heading costs.

**The "real" picture** is 21's scene with hidden scene values (30's) and
21's hull at a hidden pose -- 230 m, bearing -22 degrees, heading 55 --
rendered with independent realisations of everything random and saved as
``figures/31_real_picture.npz``; the model starts from 21's defaults with
no boat.

Acceptance criteria:
  * the scene fit with the boat in the picture recovers the seabed
    strength and the gain within 1.5 dB (as 30 without it);
  * the detection (the peak of the excess) lands within a beam width plus
    half a hull of the boat's centre;
  * the heading, from the echo's axis and the fine scan, is within 2
    degrees of the truth.  Measured: 53.7 from the axis and 55.16 after
    the scan at 120 kHz; 55.0 from the axis and 53.5 after the scan at
    330 kHz, where the lobe is 0.05 degrees, the scan is 115 headings and
    its minimum is 1.5 degrees off the axis -- and costs nothing in
    position there (0.29 m), because at 2.8 m beams the position is set
    by the beads and not by which patch glints;
  * the position fit ends within 3 m of the truth (a quarter of the 12 m
    beam at 120 kHz; a beam is 2.8 m at 330 kHz, where the bound is the
    hull's bead spacing), closer than the detection started, on a
    background the model does not share; the same descent with the true
    heading is reported next to the check so the reader sees what the
    heading costs -- and what the scan's coarse stage buys: at 330 kHz
    that reference descent, straight through the schedule from the
    centroid, falls into a bead-spacing minimum 15 m away, where the
    scanned path (its position re-started from the 4 m stage) reaches
    0.29 m.  Measured: 0.15 m at 120 kHz (0.45 m for the reference), 0.29 m
    at 330 kHz (14.8 m for the reference).

**Construction and assumptions.**  21's sonar and scene (see 21's
docstring; ``HYDROPT_SONAR=330`` runs the other head), 30's ``HIDDEN``
scene values, ``BOAT_HIDDEN`` the boat's pose (scaled by ``S``), 22's
``SCHEDULE`` and learning rates for the position, a disc of half a hull
plus a beam plus ``BOX_MARGIN_M`` for the region.  Assumptions: the hull's shape
and diffuse level are known (identity, not shape, is the question here);
altitude and depth known; the picture's display linear in dB with an
unknown offset.  To vary: ``BOAT_HIDDEN``, ``SCHEDULE``, ``BOX_MARGIN_M``.
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
import numpy as np
import torch

from _common import FIGURE_DIR, banner, check, save, setup, timed
from hydropt import (
    SceneFit, azimuth_steering, beam_noise_power, draw_labels, fit_scene,
    line_array_directivity_db, load_picture, make_time_grid, range_profile, shading_window,
)
from hydropt.mesh import boat_hull_mesh, mesh_target

HIDDEN = dict(seabed_db=-22.0, surface_db=3.0, tilt_deg=-7.0, gain_db=4.0, noise_db=2.0)
BOAT_HIDDEN = dict(range=230.0, bearing_deg=-22.0, heading_deg=55.0)   # range scales with S
REAL_SEED = 101
STEPS = int(os.environ.get("HYDROPT_STEPS", 40))
SCHEDULE = [(8.0, 6), (4.0, 10), (2.0, 8), (0.5, 24)]   # (blur in metres, steps): 22's, with an 8 m stage first
LR = [1.5, 1.0, 0.5, 0.3]
BOX_MARGIN_M = 5.0                                 # the region: half a hull + a beam + this, round the detection
YAW_SCAN_DEG = 1.5                                 # the fine heading scan, either side of the echo's axis
SECTORS = ((-60.0, -20.0), (-20.0, 20.0), (20.0, 60.0))


def _ex21():
    path = Path(__file__).resolve().parent / "21_long_range_300m.py"
    spec = importlib.util.spec_from_file_location("_ex21", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_EX = _ex21()
S = _EX.FAR / 300.0


def main() -> int:
    setup(double=False)
    banner("31 -- the scene, then the boat, fitted to a picture with a boat in it")
    ex = _EX
    ex15 = ex._ex15()
    C = ex.C
    rx = ex.horizontal_array()
    steer, bearings = azimuth_steering(ex.N_BEAMS, ex.SECTOR_DEG)
    grid = make_time_grid(2.0 * ex.NEAR / C, 2.0 * ex.FAR / C, ex.N_BINS)
    ranges = grid * C / 2.0
    shading = shading_window(ex.N_RX, "hamming")
    elev_beam = ex.beam_3db_deg(ex.N_RX_ELEV)
    beam_deg = ex.beam_3db_deg(ex.N_RX, shading)
    girth = ex.HULL_BEAM + 2.0 * ex.HULL_DRAUGHT
    verts, faces = boat_hull_mesh(ex.HULL_LENGTH, ex.HULL_BEAM, ex.HULL_DRAUGHT,
                                  n_long=int(round(110 * ex.HULL_LENGTH / 12.0)),
                                  n_around=int(round(34 * girth / (3.2 + 2.0 * 4.0))))

    def boat(x, y, heading_deg, *, learnable=False):
        t = mesh_target(verts, faces, position=(x, y, 0.0), yaw=heading_deg, n_patches=6,
                        sound_speed=C, diffuse_db=ex.DIFFUSE_DB, learnable=learnable,
                        learnable_shape=False, facet_chunk=4096, checkpoint=False)
        t.label = "boat hull"
        return t

    def model_for(scene, dirs, seed, **values):
        solid = (math.radians(2 * ex.SECTOR_DEG)
                 * math.radians(ex.ELEV_DEG[1] - ex.ELEV_DEG[0]) / dirs.shape[0])
        noise = float(beam_noise_power(scene.freqs_khz, bandwidth_hz=1.0 / ex.PULSE_S,
                                       directivity_db=line_array_directivity_db(ex.N_RX),
                                       wind_speed=ex.WIND))
        return SceneFit(scene, dirs, elements=rx, time_grid=grid, steer=steer, sigma_t=ex.PULSE_S,
                        shading=shading, source_level_db=ex.SOURCE_LEVEL_DB, noise_power=noise,
                        solid_angle_per_ray=solid, n_tx=ex.N_TX, n_rx_elev=ex.N_RX_ELEV,
                        n_elev_beams=ex.N_ELEV_BEAMS, elev_beam_deg=elev_beam,
                        max_arrivals=ex.PATCHES, seed=seed, **values)

    span_y = ex.FAR * math.sin(math.radians(ex.SECTOR_DEG)) * 1.02
    x_range = (-0.03 * ex.FAR, 1.02 * ex.FAR)
    pixel_m = (x_range[1] - x_range[0]) / 299.0
    db = lambda t: 10.0 * torch.log10(t.clamp_min(1e-30))

    # ---- the real picture: hidden scene, hidden boat, its own seeds --------- #
    banner("the real picture")
    b = math.radians(BOAT_HIDDEN["bearing_deg"])
    true_xy = (BOAT_HIDDEN["range"] * S * math.cos(b), BOAT_HIDDEN["range"] * S * math.sin(b))
    true_h = BOAT_HIDDEN["heading_deg"]
    real_scene, *_ = ex.build_scene(rx, seed=REAL_SEED, learnable=False)
    real_dirs, _ = ex.transmit_fan(seed=REAL_SEED)
    with torch.no_grad(), timed("  trace + boat + picture"):
        truth = model_for(real_scene, real_dirs, REAL_SEED, fit=(), **HIDDEN)
        real_r = truth.renderer(tx_pattern=lambda d: ex.transmit_pattern(d, tilt_deg=HIDDEN["tilt_deg"]))
        real_noisy = real_r.picture([boat(*true_xy, true_h)], frame=0)
        real_db = db(real_noisy)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    real_path = FIGURE_DIR / f"31_real_picture{ex.TAG}.npz"
    np.savez(real_path, image=real_db[:, 0].numpy(), bearings=bearings.numpy(), ranges=ranges.numpy())
    print(f"  the boat at ({true_xy[0]:.1f}, {true_xy[1]:.1f}) m, {BOAT_HIDDEN['range'] * S:.0f} m on "
          f"bearing {BOAT_HIDDEN['bearing_deg']:+.0f} deg, heading {true_h:.0f} deg; "
          f"hidden scene: " + ", ".join(f"{k} {v:+.1f}" for k, v in HIDDEN.items()))
    del truth, real_r, real_scene

    # ---- 1. the scene, with the boat in the picture ------------------------- #
    banner("1. the scene fitted on the swath's level, the boat in the picture")
    real = load_picture(real_path)
    scene, *_ = ex.build_scene(rx, learnable=False)
    dirs, _ = ex.transmit_fan(seed=ex.SEED)
    with timed("  trace"):
        model = model_for(scene, dirs, ex.SEED, seabed_db=-27.0, surface_db=0.0,
                          tilt_deg=ex.TILT_DEG, gain_db=0.0, noise_db=0.0)
    t0 = time.perf_counter()
    hist = fit_scene(model, real, steps=STEPS, lr=0.6, sectors_deg=SECTORS)
    fitted = model.values()
    print(f"  {STEPS} steps, {(time.perf_counter() - t0) / STEPS:.1f} s each; loss "
          f"{hist[0]['loss']:.1f} -> {hist[-1]['loss']:.2f} dB^2")
    for k in HIDDEN:
        print(f"  {k:>11s} fitted {fitted[k]:+7.2f}  hidden {HIDDEN[k]:+7.2f}  error {fitted[k] - HIDDEN[k]:+.2f}")

    # ---- 2. the detection: the excess over the fitted level ----------------- #
    banner("2. the detection")
    gain_real = []

    def display(noisy):
        if not gain_real:
            gain_real.append(ex.display_gain(noisy.detach(), ranges, pixel_m=pixel_m))
        return ex.display(noisy, ranges, pixel_m=pixel_m, gain=gain_real[0])[0]

    def to_cart(shown):
        return ex15.to_cartesian(shown, bearings, grid, n_x=300, n_y=300,
                                 x_range=x_range, y_range=(-span_y, span_y))

    renderer = model.renderer(tx_pattern=lambda d: ex.transmit_pattern(d, tilt_deg=fitted["tilt_deg"]),
                              display=display, to_cartesian=to_cart)
    with torch.no_grad():
        shown_real = display(real_noisy)     # the real picture's gain, held for every model picture
        cart_real, gx, gy = to_cart(shown_real)
        # the excess: the real picture over its own level against range (the
        # median over every beam, which the boat cannot move)
        level = range_profile(real.image_db, bearings, ranges, sectors_deg=((-60.0, 60.0),),
                              smooth_m=5.0)[0]
    excess = real.image_db - level.reshape(1, -1)
    # the brightest cell of the excess, away from the near edge
    excess[:, ranges < ex.NEAR + 10.0] = -1e9
    k = int(excess.argmax())
    det_b, det_r = float(bearings[k // excess.shape[1]]), float(ranges[k % excess.shape[1]])
    det_xy = (det_r * math.cos(math.radians(det_b)), det_r * math.sin(math.radians(det_b)))
    det_err = math.hypot(det_xy[0] - true_xy[0], det_xy[1] - true_xy[1])
    peak_err = det_err
    beam_m = math.radians(beam_deg) * det_r
    print(f"  excess peaks {float(excess.max()):+.1f} dB at {det_r:.1f} m, bearing {det_b:+.1f} deg: "
          f"({det_xy[0]:.1f}, {det_xy[1]:.1f}) m, {det_err:.1f} m from the boat's centre "
          f"(a beam is {beam_m:.1f} m here, half a hull {0.5 * ex.HULL_LENGTH:.0f})")
    # the peak is one bead of the hull, and at a fine head that is an end of
    # it: the centroid of the excess above 10 dB within 25 m of the peak is
    # the other candidate for the hull's centre, and the search below tries
    # the peak, the centroid and their midpoint
    Bp, Rp = torch.meshgrid(bearings, ranges, indexing="ij")
    Xp, Yp = Rp * torch.cos(torch.deg2rad(Bp)), Rp * torch.sin(torch.deg2rad(Bp))
    near = ((Xp - det_xy[0]) ** 2 + (Yp - det_xy[1]) ** 2 < 25.0 ** 2) & (excess > 10.0)
    w = (10.0 ** (excess / 10.0)) * near
    cent_xy = (float((w * Xp).sum() / w.sum()), float((w * Yp).sum() / w.sum()))
    cent_err = math.hypot(cent_xy[0] - true_xy[0], cent_xy[1] - true_xy[1])
    # ... and its principal axis is the hull's heading, to the 180 degree
    # ambiguity of a hull: the echo is elongated along the hull at either
    # head (a beam across at 120 kHz, a row of beads at 330)
    dx, dy = Xp - cent_xy[0], Yp - cent_xy[1]
    cxx, cyy, cxy = float((w * dx * dx).sum()), float((w * dy * dy).sum()), float((w * dx * dy).sum())
    axis_deg = math.degrees(0.5 * math.atan2(2.0 * cxy, cxx - cyy)) % 180.0
    print(f"  the excess above 10 dB round it has its centroid at ({cent_xy[0]:.1f}, {cent_xy[1]:.1f}) m, "
          f"{cent_err:.1f} m from the boat's centre, and its principal axis at {axis_deg:.1f} deg "
          f"(the hull heads {true_h:.0f})")

    # ---- the region, the loss ----------------------------------------------- #
    X, Y = torch.meshgrid(gx, gy, indexing="xy")
    R = torch.hypot(X, Y); B = torch.rad2deg(torch.atan2(Y, X))
    # the region: a disc in metres that holds the hull whichever way it lies,
    # at either head (a box one beam wide clipped a 30 m hull at 330 kHz, and
    # a clipped hull has no elongation for the heading search to score)
    radius = 0.5 * ex.HULL_LENGTH + beam_m + BOX_MARGIN_M
    region = (X - det_xy[0]) ** 2 + (Y - det_xy[1]) ** 2 < radius ** 2
    floor = float(cart_real.max()) * 1e-4

    def blur(cart, sigma_m):
        if sigma_m <= 0.0:
            return cart
        sig = sigma_m / pixel_m
        half = int(math.ceil(3.0 * sig))
        t = torch.arange(-half, half + 1, dtype=cart.dtype)
        kk = torch.exp(-0.5 * (t / sig) ** 2); kk = (kk / kk.sum()).reshape(1, 1, 1, -1)
        img = cart.unsqueeze(0).unsqueeze(0)
        img = torch.nn.functional.conv2d(img, kk, padding=(0, half))
        img = torch.nn.functional.conv2d(img, kk.transpose(-1, -2), padding=(half, 0))
        return img[0, 0]

    blurred = {}

    def loss_of(cart, sigma_m=0.0):
        if sigma_m not in blurred:
            blurred[sigma_m] = blur(cart_real, sigma_m)
        ref = blurred[sigma_m]
        d = torch.log10(blur(cart, sigma_m) + floor) - torch.log10(ref + floor)
        return (d[region] ** 2).mean()

    def model_cart(target):
        cart, _, _ = renderer.picture([target], frame=0, coherent=False)
        return cart

    # ---- 3 and 4. heading by search and position by descent, interleaved --- #
    banner("3. the heading, from the echo's axis")

    def h_error(h):
        d = abs(h - true_h) % 180.0
        return min(d, 180.0 - d)              # a hull is the same turned through 180

    best_h = axis_deg
    det_xy, det_err = cent_xy, cent_err     # the hull's centre, not its brightest bead
    print(f"  coarse: the axis, {best_h:.1f} deg against {true_h:.0f} true ({h_error(best_h):.1f} off), "
          f"from the centroid ({det_err:.1f} m from the boat's centre)")

    def descend(target, stages, lrs):
        hist = []
        n = 0
        for (sigma_m, steps), lr in zip(stages, lrs):
            opt = torch.optim.Adam([target.position], lr=lr)
            for _ in range(steps):
                opt.zero_grad(set_to_none=True)
                L = loss_of(model_cart(target), sigma_m)
                L.backward()
                with torch.no_grad():
                    target.position.grad[2] = 0.0
                opt.step()
                n += 1
                with torch.no_grad():
                    p = target.position
                    hist.append((float(L.detach()),
                                 math.hypot(float(p[0]) - true_xy[0], float(p[1]) - true_xy[1])))
            print(f"    blur {sigma_m:3.1f} m: loss {hist[-1][0]:.4f}, {hist[-1][1]:.2f} m from the truth")
        return hist, n

    banner("4. the position by descent, the heading scanned finely between its stages")
    t0 = time.perf_counter()
    fit_boat = boat(*det_xy, best_h, learnable=True)
    hist_a, n_a = descend(fit_boat, SCHEDULE[:2], LR[:2])           # 8 m, then 4 m
    with torch.no_grad():
        xy1 = (float(fit_boat.position[0]), float(fit_boat.position[1]))
    # the fine scan: a hull patch of PATCH_M metres glints through a lobe
    # lambda / PATCH_M wide, so the loss against yaw has a minimum that deep
    # and that narrow at the true heading (examples/16's ripple) and the
    # scan steps at half the lobe, over +/- YAW_SCAN_DEG about the axis
    step = 0.5 * math.degrees(ex.LAMBDA / (ex.HULL_LENGTH / 6.0))
    scan = [best_h + k * step for k in range(-int(YAW_SCAN_DEG / step), int(YAW_SCAN_DEG / step) + 1)]
    with torch.no_grad(), timed(f"  {len(scan)} headings at {step:.3f} deg steps about {best_h:.1f}, 2 m blur"):
        scores = [float(loss_of(model_cart(boat(*xy1, h)), 2.0)) for h in scan]
    best_h = scan[int(np.argmin(scores))] % 180.0
    print(f"  the scan's minimum {min(scores):.4f} at {best_h:.2f} deg against {true_h:.0f} true "
          f"({h_error(best_h):.2f} off); at the axis it was {scores[len(scan) // 2]:.4f}")
    fit_boat = boat(*xy1, best_h, learnable=True)
    hist_b, n_b = descend(fit_boat, SCHEDULE[2:], LR[2:])           # 2 m, then 0.5 m
    n_steps = n_a + n_b
    per_step = (time.perf_counter() - t0) / n_steps
    history = [(0.0, det_err)] + hist_a + hist_b
    final_err = history[-1][1]
    h_err = h_error(best_h)
    print(f"  {n_steps} steps, {per_step:.1f} s each: {det_err:.1f} m (centroid) -> {final_err:.2f} m, "
          f"heading {best_h:.2f} ({h_err:.2f} deg off)")
    # what the heading costs: the same descent with the true heading, reported
    banner("the same descent with the true heading, for reference")
    oracle = boat(*det_xy, true_h, learnable=True)
    hist_o, _ = descend(oracle, SCHEDULE, LR)
    print(f"  with the heading known: {det_err:.1f} -> {hist_o[-1][1]:.2f} m")

    # ---- the figure --------------------------------------------------------- #
    with torch.no_grad():
        cart_fit, _, _ = renderer.picture([fit_boat], frame=0)
        _, labs = renderer.picture([fit_boat], frame=0, labels=True)
    ext = [float(gx.min()), float(gx.max()), float(gy.min()), float(gy.max())]
    ref = float(db(cart_real).max())
    fig, axes = plt.subplots(1, 3, figsize=(19, 6.2))
    for ax, img, title in ((axes[0], db(cart_real), "the real picture (its own sea, its own boat)"),
                           (axes[1], db(cart_fit), "the fitted scene with the fitted boat"),
                           (axes[2], (db(cart_real) - db(cart_fit)).clamp(-15, 15), "real minus model (dB)")):
        m = ax.imshow(img.numpy(), origin="lower", extent=ext, cmap="inferno" if ax is not axes[2] else "coolwarm",
                      vmin=ex.THRESHOLD_DB if ax is not axes[2] else -15, vmax=ref if ax is not axes[2] else 15,
                      aspect="equal")
        ax.plot(true_xy[0], true_xy[1], "c+", ms=12, mew=1.5)
        ax.plot(det_xy[0], det_xy[1], "wx", ms=8, mew=1.2)
        ax.set_title(title); ax.set_xlabel("forward (m)"); ax.set_ylabel("across (m)")
        ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
    draw_labels(axes[1], [l for l in labs if l.visible])
    fig.colorbar(m, ax=axes[2], fraction=0.04, label="dB")
    fig.suptitle(f"{ex.FREQ_KHZ:.0f} kHz FLS to {ex.FAR:.0f} m: scene fitted with the boat in the "
                 f"picture, boat detected ({det_err:.1f} m), heading {best_h:.0f} ({true_h:.0f}), "
                 f"position fitted to {final_err:.2f} m")
    save(fig, f"31_fit_scene_and_boat{ex.TAG}.png")

    banner("acceptance")
    ok = check("the scene fit, with the boat in the picture, recovers seabed and gain within 1.5 dB",
               abs(fitted["seabed_db"] - HIDDEN["seabed_db"]) < 1.5 and abs(fitted["gain_db"] - HIDDEN["gain_db"]) < 1.5,
               f"seabed {fitted['seabed_db'] - HIDDEN['seabed_db']:+.2f}, gain {fitted['gain_db'] - HIDDEN['gain_db']:+.2f} dB")
    ok &= check("the detection lands within a beam plus half a hull of the boat",
                peak_err < beam_m + 0.5 * ex.HULL_LENGTH, f"{peak_err:.1f} m against {beam_m + 0.5 * ex.HULL_LENGTH:.1f}")
    ok &= check("the heading, from the echo's axis and the fine scan, is within 2 degrees",
                h_err <= 2.0, f"{best_h:.2f} against {true_h:.0f} deg (the axis gave {axis_deg:.1f})")
    ok &= check("the position fit ends within 3 m of the truth, closer than it started",
                final_err < 3.0 and final_err < det_err, f"{det_err:.1f} -> {final_err:.2f} m")
    print(f"  (with the true heading the same descent ends at {hist_o[-1][1]:.2f} m: what the heading costs)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
