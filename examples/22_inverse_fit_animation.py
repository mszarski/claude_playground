"""Watching the inverse fit: a boat's position recovered from a 300 m image.

The sonar, the environment and the boat are ``examples/21``'s -- a 120 kHz
Mills cross on an AUV at 12 m in 30 m of water, a light wind sea over a rough
sand seabed, a 30 m boat at 250 m, 58 degrees off the line of sight -- and so
is the picture: the median-TVG image on a grid in metres, at the gain the
measurement set.  What is new is that the boat starts in the wrong place and
the image loss walks it home, frame by frame, into
``figures/22_inverse_fit.gif``.

The fit compares the measurement with a TEMPLATE: the same ping with the
boat's arrivals added in power rather than in field (``beamform(...,
coherent=False)``), which is the expected intensity of its image over the
phases of its patches.  The coherent image of a 30 m hull is a speckle
pattern in the hull's position -- 5 cm across track rearranges the
interference between its patches, a wavelength in range does the same
against its ghosts -- so a loss against the coherent picture is a cusp at
the truth on a rough plateau, measured here to be +-2 m wide however much
the pictures are blurred first (the residual speckle of a 6 m blur is still
a tenth of the level, and it decorrelates within half a metre).  The
expected intensity moves smoothly with the boat, and the measurement's own
speckle, which does not move, costs a floor on the loss and not a minimum in
the wrong place.  That is what a sonar template match does, and it is why.

Three things make it cheap enough to watch:

* **the return leg is solved**, by the method of images (the sound speed is
  constant), so a ping's target arrivals cost a few seconds, not a minute;
* **the reverberation is formed once.**  The beamformer is linear in the
  arrivals, so the seabed's and surface's complex beams are computed a single
  time and the boat's beams added to them on every step: ``|b_rev +
  b_boat|^2`` is exactly the image of the whole ping, and the template adds
  the boat's expected intensity to ``|b_rev + noise|^2``;
* **only the boat carries a graph.**  The scene is built non-learnable; the
  gradient has one place to go.

And one thing makes it honest: before the descent, the analytic gradient is
checked against a finite difference -- on the template's loss with the step
scaled to each axis's own resolution (a fifth of a range cell, a fifth of a
beam), as ``examples/16`` does; and on the coherent picture itself, at a
sixteenth, a thirty-second and a sixty-fourth of a wavelength, inside one
fringe of the echo, where the two have to agree in size as well and the
ratio has to converge on one as the step shrinks.  A fit that converges on a
wrong gradient converges by luck.  The fine check found two stages of
``examples/21``'s picture that a fit must not re-derive from every trial
image, and both are handled here:

* **the display gain is the measurement's**, taken once and held.  The gain
  is a median over beams, and the derivative of an order statistic is that
  of whichever beam holds it, while a finite step sees the median hop between
  beams (six times off, measured);
* **the receiver noise is added to the field**, not to the power.  Through
  the power the model passes through ``sqrt(S) = |b|``, which has a kink at
  every null of the field, and a coherent hull's fringes put a null within a
  sixteenth of a wavelength of a quarter of its cells (five times off).  The
  complex beams are already there, so the noise phasor is simply added to
  them: the same Rice statistics, smooth in the field.

Position only.  ``examples/16`` measured why: with every path solved the loss
against yaw ripples at a tenth of a degree, the scale at which a hull's ends
move by a wavelength, so heading is captured only from within a fraction of a
degree and is held at the truth here.

Acceptance criteria:
  * the template's gradient agrees with the secant (cosine over 0.85), and
    the coherent picture's is the wavelength-scale finite difference;
  * the fit closes more than 90 % of the gap between the loss where it
    started and the loss at the truth;
  * the boat ends within a metre of where it is, from 4 m in range and 6 m
    across-track away, through a coarse-to-fine schedule of blurs on the
    picture (see ``SCHEDULE``).

**Double precision, deliberately.**  A picture is fine in float32; a gradient
through it is not.  The image is a coherent sum of ninety thousand phasors
whose derivatives carry the carrier's 7.5e5 rad/s, and in single precision
the backward pass loses the cancellation between them: measured at 90 m, the
float64 gradient of the displayed image matched a 0.8 mm finite difference to
0.3 percent, and the float32 one had the wrong sign.  So this example ignores
``HYDROPT_EXAMPLE_DTYPE`` in spirit -- run it in float64, which is the
default -- and checks the derivative at that scale before trusting it.  The
other switches of ``examples/21`` (``HYDROPT_FAR``, ``HYDROPT_BOAT``, ...)
carry through, since that module is imported for its settings.
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
    LambertScattering, add_receiver_noise, azimuth_steering, beam_noise_power,
    beam_power_scale, beamform, calibrate, line_array_directivity_db,
    make_time_grid, reverberation_arrivals, shading_window, target_arrivals,
    trace,
)
from hydropt.mesh import boat_hull_mesh, mesh_target

# Inside the basin.  The hull is six patches 5 m apart along its length, so
# the image loss has a local minimum every time the model's patches land on
# the measurement's neighbours: from 8 m out in range the fit walked to 11 and
# settled there (examples/19 has the transport loss for that haul).  Across
# track the basin is the beam, 13 m at this range.
START_OFFSET = (4.0, -6.0)      # metres: range, across-track
# Coarse to fine, on the PICTURE: both are blurred by a few metres first, which
# widens the basin (the measurement's speckle is fixed, so the template's loss
# is smooth at any blur; the blur is against the speckle's contribution to the
# loss, not its roughness), and the blur then shrinks to half a pixel.  It
# has to: measured along range, the loss at 2 and 4 m of blur is LOWER 1.75 m
# beyond the truth than at it, and at 1 m of blur has a local minimum 1.25 m
# beyond.  That is not speckle but the ghost.  A hull patch's direct return
# and its surface image are 0.2 m apart in range here (2 z_s z_t / R), inside
# one range cell, and they interfere with a phase that turns through two
# radians along the hull's 16 m of range extent: an 8 m fringe over the
# measured blob that the template, which adds them in power, does not have,
# and which moves the blurred blob's centre.  At half a metre of blur the
# resolved structure pins the truth (a clean V, 15 % deeper than its
# neighbours a quarter of a metre away), so the last stage is done there.
SCALE = float(os.environ.get("HYDROPT_FIT_SCALE", 1.0))     # x the steps below
SCHEDULE = [(4.0, round(10 * SCALE)), (2.0, round(8 * SCALE)),
            (0.5, round(30 * SCALE))]                       # (blur m, steps)
LR = [1.0, 0.5, 0.3]            # metres per Adam step, per stage
CROP = 45.0                     # half-size of the window drawn about the boat, m


def _ex21():
    path = Path(__file__).resolve().parent / "21_long_range_300m.py"
    spec = importlib.util.spec_from_file_location("_ex21", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    setup()          # float64: see the note on precision in the docstring
    banner("22 -- the inverse fit, frame by frame")
    ex = _ex21()
    ex15 = ex._ex15()
    C = ex.C
    rx = ex.horizontal_array()
    scene, bottom, surface, sediment = ex.build_scene(rx, learnable=False)
    print(f"  {ex.FREQ_KHZ:.0f} kHz, {ex.N_RX} x {ex.N_TX}, swath to {ex.FAR:.0f} m; "
          f"boat {ex.HULL_LENGTH:.0f} m at {ex.BOAT_RANGE:.0f} m, "
          f"bearing {ex.BOAT_BEARING_DEG:+.0f} deg, heading {ex.BOAT_HEADING_DEG:.0f} deg")

    def build_boat(dx: float = 0.0, dy: float = 0.0, *, learnable: bool):
        b = math.radians(ex.BOAT_BEARING_DEG)
        girth = ex.HULL_BEAM + 2.0 * ex.HULL_DRAUGHT
        verts, faces = boat_hull_mesh(
            ex.HULL_LENGTH, ex.HULL_BEAM, ex.HULL_DRAUGHT,
            n_long=int(round(110 * ex.HULL_LENGTH / 12.0)),
            n_around=int(round(34 * girth / (3.2 + 2.0 * 4.0))))
        return mesh_target(
            verts, faces,
            position=(ex.BOAT_RANGE * math.cos(b) + dx,
                      ex.BOAT_RANGE * math.sin(b) + dy, 0.0),
            yaw=ex.BOAT_HEADING_DEG, n_patches=6, sound_speed=C,
            diffuse_db=ex.DIFFUSE_DB, learnable=learnable,
            learnable_shape=False, facet_chunk=256)

    truth = build_boat(learnable=False)
    true_xy = truth.position.detach()[:2].clone()

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
    di = line_array_directivity_db(ex.N_RX)
    noise = float(beam_noise_power(scene.freqs_khz, bandwidth_hz=1.0 / ex.PULSE_S,
                                   directivity_db=di, wind_speed=ex.WIND))
    solid = (math.radians(2 * ex.SECTOR_DEG)
             * math.radians(ex.ELEV_DEG[1] - ex.ELEV_DEG[0]) / dirs.shape[0])

    def beams(arrivals, sigma_t):
        return beamform(arrivals, rx, scene.freqs_khz, grid, steer,
                        sigma_t=sigma_t, shading=shading, steer_chunk=8,
                        complex_output=True)

    banner("the background, once")
    with torch.no_grad(), timed("  trace + reverberation"):
        result = trace(scene, dirs)
        seabed = LambertScattering(-27.0, learnable=False)
        rev = reverberation_arrivals(
            result, dirs, scene.freqs_khz, scattering=seabed,
            solid_angle_per_ray=solid, ray_weights=w_tx * rx_beam(-dirs),
            boundary="both", surface=scene.surface, bottom=scene.bottom,
            max_arrivals=ex.PATCHES,
            generator=torch.Generator().manual_seed(ex.SEED + 1))
        b_rev = beams(rev, ex.PULSE_S)
    print(f"  {rev.n_arrivals} patches -> complex beams {tuple(b_rev.shape)}")

    def echo(boat):
        return target_arrivals(
            scene, boat, dirs, return_leg="eigenray", n_rx_rays=2000,
            rx_half_angle_deg=45.0, tx_weights=w_tx,
            tx_pattern=ex.transmit_pattern, rx_pattern=rx_beam,
            max_arrivals_per_leg=24,
            generator=torch.Generator().manual_seed(ex.SEED))

    span_y = ex.FAR * math.sin(math.radians(ex.SECTOR_DEG)) * 1.02
    x_range = (-0.03 * ex.FAR, 1.02 * ex.FAR)
    pixel_m = (x_range[1] - x_range[0]) / 299.0

    scale = beam_power_scale(shading, ex.PULSE_S)

    def received(boat, *, template=False):
        """The calibrated, noisy [beams, bands, bins] image of a ping.

        Coherent -- the ping as the sonar would record it -- or, as the
        ``template``, with the boat's expected intensity added to the
        reverberation and noise (see the docstring).
        """
        arr = echo(boat)
        if template:
            back = calibrate(b_rev, ex.SOURCE_LEVEL_DB, beam_scale=scale)
            noisy = add_receiver_noise(back, noise,
                                       generator=torch.Generator().manual_seed(ex.SEED + 2))
            boat_power = beamform(arr, rx, scene.freqs_khz, grid, steer,
                                  sigma_t=ex.PULSE_S, shading=shading, steer_chunk=8,
                                  coherent=False)
            return noisy + calibrate(boat_power, ex.SOURCE_LEVEL_DB, beam_scale=scale)
        field = calibrate(b_rev + beams(arr, ex.PULSE_S), ex.SOURCE_LEVEL_DB,
                          beam_scale=scale)
        # noise on the FIELD, not the power: same statistics, no kink at a
        # null (see the docstring, and add_receiver_noise)
        return add_receiver_noise(field, noise,
                                  generator=torch.Generator().manual_seed(ex.SEED + 2))

    # The display gain is the MEASUREMENT's, taken once and held: the model
    # picture is shown at the gain of the picture it is fitted to, as a sonar
    # would show a template over its own AGC'd image.  Re-deriving a median
    # gain from every trial picture would also break the gradient -- the
    # derivative of an order statistic is that of one beam, while a finite
    # step sees the median hop between beams; measured, six times off.
    gain = []

    def picture(boat, *, template=False):
        """The displayed image, in metres, differentiable in the boat."""
        noisy = received(boat, template=template)
        if not gain:
            gain.append(ex.display_gain(noisy.detach(), rng, pixel_m=pixel_m))
        shown, _ = ex.display(noisy, rng, pixel_m=pixel_m, gain=gain[0])
        cart, gx, gy = ex15.to_cartesian(shown, bearings, grid, n_x=300, n_y=300,
                                         x_range=x_range, y_range=(-span_y, span_y))
        return cart, gx, gy

    with torch.no_grad(), timed("  the measurement"):
        meas, gx, gy = picture(truth)          # sets the gain
    floor = float(meas.max()) * 1e-4

    def blur(cart, sigma_m):
        """A Gaussian blur of the picture, in metres, separable and differentiable."""
        if sigma_m <= 0.0:
            return cart
        sig = sigma_m / pixel_m
        half = int(math.ceil(3.0 * sig))
        t = torch.arange(-half, half + 1, dtype=cart.dtype)
        k = torch.exp(-0.5 * (t / sig) ** 2)
        k = (k / k.sum()).reshape(1, 1, 1, -1)
        img = cart.unsqueeze(0).unsqueeze(0)
        img = torch.nn.functional.conv2d(img, k, padding=(0, half))
        img = torch.nn.functional.conv2d(img, k.transpose(-1, -2), padding=(half, 0))
        return img[0, 0]

    blurred = {}

    def loss_of(cart, sigma_m=0.0):
        if sigma_m not in blurred:
            blurred[sigma_m] = blur(meas, sigma_m)
        ref = blurred[sigma_m]
        return ((torch.log10(blur(cart, sigma_m) + floor)
                 - torch.log10(ref + floor)) ** 2).mean()

    # ---- is the gradient right? ------------------------------------------ #
    banner("the gradient, against a finite difference")
    BLUR = SCHEDULE[-1][0]      # the finest stage's loss, the one the fit ends on
    boat = build_boat(*START_OFFSET, learnable=True)
    cart, _, _ = picture(boat, template=True)
    L0 = loss_of(cart, BLUR)
    L0.backward()
    g = boat.position.grad.detach()[:2].clone()
    cell_range = ex.PULSE_S * C / 2.0
    cell_bearing = ex.BOAT_RANGE * math.radians(ex.beam_3db_deg(ex.N_RX, shading))
    fd = torch.zeros(2)
    with torch.no_grad():
        for k, h in enumerate((0.2 * cell_range, 0.2 * cell_bearing)):
            vals = []
            for sign in (1.0, -1.0):
                d = [0.0, 0.0]
                d[k] = sign * h
                trial = build_boat(START_OFFSET[0] + d[0], START_OFFSET[1] + d[1],
                                   learnable=False)
                vals.append(float(loss_of(picture(trial, template=True)[0], BLUR)))
            fd[k] = (vals[0] - vals[1]) / (2 * h)
    # compare in cell units, as 16 does, so neither axis dominates the cosine
    units = torch.tensor([cell_range, cell_bearing])
    cos = float((g * units * fd * units).sum()
                / ((g * units).norm() * (fd * units).norm()).clamp_min(1e-30))
    finite = bool(torch.isfinite(g).all()) and float(g.abs().sum()) > 0
    print(f"  template: d loss / d(range, across): analytic {g[0]:+.3e}, {g[1]:+.3e}; "
          f"secant over a fifth of a cell {fd[0]:+.3e}, {fd[1]:+.3e}")
    print(f"  cosine in cell units: {cos:+.3f}  (steps of {0.2 * cell_range:.3f} m "
          f"and {0.2 * cell_bearing:.2f} m)")
    # The secant across a fifth of a cell spans several wavelengths, and the
    # loss RIPPLES at the wavelength scale -- the hull's six patches interfere
    # -- so the local slope can be a thousand times the secant while pointing
    # the same way.  A step inside one ripple is where the two have to agree
    # in size as well as direction: that is the derivative being right, rather
    # than the trend being right.  The step is halved twice, because a central
    # difference under-reads a sinusoid by sin(kh)/kh: the loss is quadratic in
    # a two-way coherent image, so its finest ripple has a period of a quarter
    # wavelength, and a sixteenth-wavelength step reads that at 64 %.  The
    # ratio has to CONVERGE on one as the step shrinks; the check is at the
    # finest step.
    coh = build_boat(*START_OFFSET, learnable=True)
    L_coh = loss_of(picture(coh)[0], BLUR)
    L_coh.backward()
    g_coh = coh.position.grad.detach()[:2].clone()
    print(f"  the coherent picture: d loss / d(range, across) = {g_coh[0]:+.3e}, {g_coh[1]:+.3e}")
    fine = {}
    with torch.no_grad():
        for div in (16, 32, 64):
            h_fine = ex.LAMBDA / div
            fd_fine = torch.zeros(2)
            for k in range(2):
                vals = []
                for sign in (1.0, -1.0):
                    d = [0.0, 0.0]
                    d[k] = sign * h_fine
                    trial = build_boat(START_OFFSET[0] + d[0], START_OFFSET[1] + d[1],
                                       learnable=False)
                    vals.append(float(loss_of(picture(trial)[0], BLUR)))
                fd_fine[k] = (vals[0] - vals[1]) / (2 * h_fine)
            ratio = float(g_coh.norm() / fd_fine.norm().clamp_min(1e-30))
            cos_fine = float((g_coh * fd_fine).sum()
                             / (g_coh.norm() * fd_fine.norm()).clamp_min(1e-30))
            fine[div] = (h_fine, ratio, cos_fine)
            print(f"  at a wavelength / {div} ({h_fine * 1e3:.2f} mm): finite difference "
                  f"{fd_fine[0]:+.3e}, {fd_fine[1]:+.3e}; |analytic| / |fd| = {ratio:.3f}, "
                  f"cosine {cos_fine:+.3f}")
    h_fine, ratio, cos_fine = fine[64]

    # ---- the fit ---------------------------------------------------------- #
    N_STEPS = sum(n for _, n in SCHEDULE)
    banner(f"the fit: {N_STEPS} steps from {START_OFFSET[0]:+.0f} m in range, "
           f"{START_OFFSET[1]:+.0f} m across, coarse then fine")
    history = []          # (step, loss, dx, dy, |grad|, blur)
    frames = []           # the model picture about the boat, per step
    win = None
    t0 = time.perf_counter()
    step = 0
    cx, cy = float(true_xy[0]), float(true_xy[1])
    win = ((gx >= cx - CROP) & (gx <= cx + CROP), (gy >= cy - CROP) & (gy <= cy + CROP))
    for stage, (sigma_m, n_steps) in enumerate(SCHEDULE):
        opt = torch.optim.Adam([boat.position], lr=LR[stage])
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_steps, eta_min=0.1 * LR[stage])
        print(f"  stage {stage + 1}: {sigma_m:.2f} m blur, {n_steps} steps of up to {LR[stage]:.2f} m")
        last = stage == len(SCHEDULE) - 1
        for k in range(n_steps + (1 if last else 0)):
            opt.zero_grad()
            cart, _, _ = picture(boat, template=True)
            L = loss_of(cart, sigma_m)
            stepping = k < n_steps
            if stepping:
                L.backward()
                with torch.no_grad():
                    boat.position.grad[2] = 0.0
            off = (boat.position.detach()[:2] - true_xy)
            gnorm = float(boat.position.grad[:2].norm()) if boat.position.grad is not None else 0.0
            history.append((step, float(L.detach()), float(off[0]), float(off[1]), gnorm, sigma_m))
            frames.append(10.0 * torch.log10(cart.detach()[win[1]][:, win[0]].clamp_min(1e-30)))
            print(f"  step {step:3d}: loss {float(L.detach()):.5f}  error {float(off.norm()):6.2f} m "
                  f"(range {float(off[0]):+6.2f}, across {float(off[1]):+6.2f})  |grad| {gnorm:.2e}")
            if stepping:
                opt.step()
                sched.step()
            step += 1
    print(f"  {N_STEPS} steps in {time.perf_counter() - t0:.0f} s")
    final_err = math.hypot(history[-1][2], history[-1][3])

    # ---- the animation ---------------------------------------------------- #
    banner("the animation")
    meas_db = 10.0 * torch.log10(meas[win[1]][:, win[0]].clamp_min(1e-30))
    vmax = float(meas_db.max())
    vmin = vmax - 40.0          # the boat sits ~38 dB over the median: show the sea it sits in
    ext = [float(gx[win[0]].min()), float(gx[win[0]].max()),
           float(gy[win[1]].min()), float(gy[win[1]].max())]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    im = axes[0].imshow(frames[0].numpy(), origin="lower", extent=ext, vmin=vmin,
                        vmax=vmax, cmap="inferno", aspect="equal")
    axes[0].set_title("the template, where the fit thinks the boat is")
    axes[1].imshow(meas_db.numpy(), origin="lower", extent=ext, vmin=vmin, vmax=vmax,
                   cmap="inferno", aspect="equal")
    axes[1].set_title("the measurement")
    for ax in axes[:2]:
        ax.set_xlabel("forward (m)"); ax.set_ylabel("across (m)")
        ax.plot(float(true_xy[0]), float(true_xy[1]), "c+", ms=14, mew=2)
    dot, = axes[0].plot([], [], "wo", ms=7, mfc="none", mew=2)
    trail, = axes[0].plot([], [], "w-", lw=1, alpha=0.7)
    steps = [h[0] for h in history]
    losses = [h[1] for h in history]
    errs = [math.hypot(h[2], h[3]) for h in history]
    axes[2].plot(steps, losses, color="0.6", lw=1)
    edge = 0
    for sigma_m, n_steps in SCHEDULE:            # the stages, and their blurs
        axes[2].axvline(edge, color="0.8", lw=0.8, ls="--")
        axes[2].text(edge + 0.3, 0.98, f"{sigma_m:g} m blur", transform=axes[2].get_xaxis_transform(),
                     fontsize=8, color="0.4", va="top")
        edge += n_steps
    lloss, = axes[2].plot([], [], "k-", lw=2, label="loss")
    axes[2].set_xlabel("step"); axes[2].set_ylabel("log-image loss")
    ax2 = axes[2].twinx()
    ax2.plot(steps, errs, color="tab:blue", lw=1, alpha=0.4)
    lerr, = ax2.plot([], [], color="tab:blue", lw=2, label="position error (m)")
    ax2.set_ylabel("position error (m)", color="tab:blue")
    axes[2].set_title("descent")
    fig.suptitle(f"{ex.FREQ_KHZ:.0f} kHz FLS at {ex.FAR:.0f} m: recovering the boat's "
                 f"position by descent on the displayed image")

    def draw(i):
        im.set_data(frames[i].numpy())
        xs = [float(true_xy[0]) + h[2] for h in history[:i + 1]]
        ys = [float(true_xy[1]) + h[3] for h in history[:i + 1]]
        dot.set_data([xs[-1]], [ys[-1]])
        trail.set_data(xs, ys)
        lloss.set_data(steps[:i + 1], losses[:i + 1])
        lerr.set_data(steps[:i + 1], errs[:i + 1])
        axes[0].set_title(f"the template, step {i} (loss at a {history[i][5]:.2f} m blur): "
                          f"{errs[i]:.2f} m from the truth")
        return im, dot, trail, lloss, lerr

    anim = animation.FuncAnimation(fig, draw, frames=len(frames), interval=200, blit=False)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    gif = FIGURE_DIR / "22_inverse_fit.gif"
    with timed("  gif"):
        anim.save(gif, writer=animation.PillowWriter(fps=5))
    print(f"  wrote {gif}")
    draw(len(frames) - 1)
    save(fig, "22_inverse_fit.png")

    banner("acceptance")
    ok = check("the gradient is finite and points where the secant does",
               finite and cos > 0.85, f"cosine {cos:+.3f} over a fifth of a cell")
    ok &= check("and inside one ripple it IS the finite difference, in size and direction",
                cos_fine > 0.99 and 0.9 < ratio < 1.1,
                f"|analytic| / |fd| = {ratio:.3f}, cosine {cos_fine:+.3f} at {h_fine * 1e3:.2f} mm")
    with torch.no_grad():
        L_end = float(loss_of(picture(boat, template=True)[0], BLUR))
        L_start = float(loss_of(picture(build_boat(*START_OFFSET, learnable=False),
                                        template=True)[0], BLUR))
        L_true = float(loss_of(picture(truth, template=True)[0], BLUR))
    ok &= check("the fit closes more than 90 % of the gap to the loss at the truth",
                L_end - L_true < 0.1 * (L_start - L_true),
                f"{L_start:.5f} at the start, {L_end:.5f} at the end, {L_true:.5f} at the truth")
    ok &= check("the boat ends within a metre of where it is",
                final_err < 1.0,
                f"{math.hypot(*START_OFFSET):.1f} m -> {final_err:.2f} m, against a "
                f"{cell_bearing:.1f} m beam")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
