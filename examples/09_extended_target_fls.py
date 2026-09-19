"""Forward-looking sonar against an extended target: glints, not point scatterers.

A point scatterer gives an FLS the right arrival time and the wrong picture.  But
the obvious replacement -- chop the body into sections and expect it to "resolve
into highlights" -- is wrong too, and this example shows why.

At 100 kHz a 0.8 m hull section has a beamwidth of about ``lambda / 2a`` = 0.54
degrees.  A 4 m hull at 40 m subtends 5.7 degrees, so the sections at its ends
see the sonar some 2.3 degrees off their own broadside -- four beamwidths out.
They return essentially nothing.  **A smooth hull glints**: the echo comes from
the one place on it where the specular condition is met, and that place moves as
the geometry changes.  Measured below: the specular section carries about 71% of
the energy and its neighbours 7%, with the ends at 0.5%.

What *does* spread across a body's extent is a set of **discrete** scatterers --
edges, corners, fittings, a wreck's structure -- each small enough to be broad in
aspect.  That is the other target built here, and it behaves the way the
textbook picture describes.

The difference matters because it decides what an image of the target looks
like: a glinting hull gives one bright return that jumps about as the target
turns, and a structured wreck gives a spatial extent you can measure.

**Why the caps are there.** Physical optics takes the hull's cross-section to
*exactly zero* end-on, because the projected length vanishes.  That is a
modelling artefact, not physics: a real cylinder end-on returns its end cap.  The
caps are what keep the target visible at every aspect, and they are the reason an
extended target is built from parts rather than from one formula.

Acceptance criteria:
  * the smooth hull's echo concentrates on its specular section, and that
    section moves when the body slides along its own axis;
  * a body of discrete scatterers spreads across its bearing extent, where a
    point target at the same place does not;
  * the aspect pattern falls monotonically and sits on or above the direct-path
    envelope, multipath having filled its nulls in;
  * the bare hull all but vanishes end-on, and the capped body does not.
"""

from __future__ import annotations

import math

import torch

from _common import banner, check, save, setup, timed
from hydropt import (
    ConstantLoss, CylinderScattering, ExtendedTarget, FlatHeight,
    PiecewiseLinearProfile, PlateScattering, PointTarget, RayleighBottomLoss, Scene,
    azimuth_steering, beamform, compose_arrivals, extract_arrivals, make_time_grid,
    shading_window, target_arrivals, trace,
)
from hydropt.active import _RelocatedScene, return_fan
from hydropt.launch import fan_sigma_d, fibonacci_cone

C = 1500.0
FREQ_HZ = 100e3
LAM = C / FREQ_HZ
N_ELEMENTS = 32
WATER_DEPTH = 30.0
VEHICLE_DEPTH = 10.0
SECTOR_DEG = 45.0

HULL_LENGTH = 4.0
HULL_RADIUS = 0.25
N_SECTIONS = 5
TARGET_RANGE = 40.0
TARGET_DEPTH = 14.0
TX_RAYS = 5000
RX_RAYS = 6000
RX_HALF_ANGLE_DEG = 25.0


def receive_array() -> torch.Tensor:
    y = (torch.arange(N_ELEMENTS, dtype=torch.get_default_dtype())
         - (N_ELEMENTS - 1) / 2) * (LAM / 2)
    return torch.stack((torch.zeros_like(y), y,
                        torch.full_like(y, VEHICLE_DEPTH)), dim=-1)


def build_scene(receivers: torch.Tensor) -> Scene:
    return Scene(
        field=PiecewiseLinearProfile([0.0, VEHICLE_DEPTH, WATER_DEPTH],
                                     [1512.0, 1505.0, 1503.0], learnable=False),
        bottom=FlatHeight(WATER_DEPTH), surface=FlatHeight(0.0),
        source=(0.0, 0.0, VEHICLE_DEPTH), receivers=receivers,
        surface_loss=ConstantLoss(2.0, learnable=False, pressure_release=True),
        bottom_loss=RayleighBottomLoss(1900.0, 1650.0, 0.8, learnable=False),
        freqs_khz=torch.tensor([FREQ_HZ / 1e3]),
        step_size=0.2, n_steps=600, max_bounces=4,
    )


def build_target(yaw_deg: float, *, caps: bool = True) -> ExtendedTarget:
    """A hull of cylinder sections, optionally with flat end caps.

    Body ``x`` is the hull axis, so ``yaw = 0`` points the nose at the sonar
    (end-on) and ``yaw = 90`` presents the flank (broadside).
    """
    section = HULL_LENGTH / N_SECTIONS
    xs = torch.linspace(-HULL_LENGTH / 2 + section / 2,
                        HULL_LENGTH / 2 - section / 2, N_SECTIONS)
    hull = CylinderScattering(section, HULL_RADIUS, sound_speed=C, learnable=False)
    offsets = [torch.stack([xs, torch.zeros(N_SECTIONS), torch.zeros(N_SECTIONS)], -1)]
    patterns: list = [hull] * N_SECTIONS
    if caps:
        # A disc of radius r, as a square plate of the same area, normal along
        # the hull axis: this is what a real cylinder returns end-on.
        side = math.sqrt(math.pi) * HULL_RADIUS
        cap = PlateScattering(side, side, normal=(1.0, 0.0, 0.0),
                              length_axis=(0.0, 1.0, 0.0), sound_speed=C,
                              learnable=False)
        offsets.append(torch.tensor([[-HULL_LENGTH / 2, 0.0, 0.0],
                                     [HULL_LENGTH / 2, 0.0, 0.0]]))
        patterns += [cap, cap]
    return ExtendedTarget(torch.cat(offsets, 0), patterns,
                          position=(TARGET_RANGE, 0.0, TARGET_DEPTH),
                          yaw=yaw_deg, learnable=False)


def build_discrete(yaw_deg: float, ts_db: float = -12.0) -> ExtendedTarget:
    """The same layout, but as discrete scatterers rather than hull sections.

    A corner, an edge or a fitting is small against the wavelength in at least
    one dimension, so it is broad in aspect where a specular panel is not.  Taken
    as isotropic here, which is the limit of that.
    """
    xs = torch.linspace(-HULL_LENGTH / 2, HULL_LENGTH / 2, N_SECTIONS)
    offsets = torch.stack([xs, torch.zeros(N_SECTIONS), torch.zeros(N_SECTIONS)], -1)
    return ExtendedTarget(offsets, ts_db, position=(TARGET_RANGE, 0.0, TARGET_DEPTH),
                          yaw=yaw_deg, learnable=False)


def transmit_shading(directions: torch.Tensor) -> torch.Tensor:
    az = torch.atan2(directions[:, 1], directions[:, 0])
    el = torch.asin(directions[:, 2].clamp(-1.0, 1.0))
    return (torch.exp(-0.5 * (az / math.radians(14.0)) ** 2)
            * torch.exp(-0.5 * (el / math.radians(8.0)) ** 2))


def energy_by_highlight(arrivals, target) -> list[tuple[int, float, float, float]]:
    """Energy share per highlight: ``(index, world y, body offset, share)``.

    Each highlight sits at its own bearing, and the bearings are separated by
    more than the multipath spread about any one of them, so binning arrivals to
    the nearest highlight bearing attributes them correctly.

    Both the world position and the body-frame offset are reported, because the
    two say different things: the glint stays put in the *world* (at the specular
    point) while moving along the *body*, and it is the second that shows the
    glint is not attached to any particular piece of the hull.
    """
    world = target.world_positions().detach()
    body = target.highlights.detach()
    geo = torch.atan2(world[:, 1], world[:, 0]) * 180.0 / math.pi
    d = -arrivals.direction.detach()
    bearing = torch.atan2(d[:, 1], d[:, 0]) * 180.0 / math.pi
    energy = (arrivals.amplitude.detach() ** 2).sum(-1)
    total = energy.sum().clamp_min(1e-300)
    nearest = (bearing.reshape(-1, 1) - geo.reshape(1, -1)).abs().argmin(dim=1)
    return [(i, float(world[i, 1]), float(body[i, 0]),
             float(energy[nearest == i].sum() / total))
            for i in range(world.shape[0])]


def spreads(arrivals) -> tuple[float, float]:
    """Energy-weighted spread of an arrival set in range (m) and bearing (deg).

    Weighted, because the unweighted spread of *any* target in this channel is
    set by its surface and bottom multipath rather than by its size.
    """
    w = (arrivals.amplitude ** 2).sum(-1)
    w = w / w.sum().clamp_min(1e-300)
    rng = arrivals.path_length / 2.0
    # Bearing of the arrival, from the direction it reached the array travelling.
    d = -arrivals.direction
    bearing = torch.atan2(d[:, 1], d[:, 0]) * 180.0 / math.pi

    def wstd(x):
        mean = (w * x).sum()
        return float(((w * (x - mean) ** 2).sum()).clamp_min(0.0).sqrt())

    return wstd(rng), wstd(bearing)


def main() -> int:
    setup()
    banner("09 -- forward-looking sonar against an extended target")

    elements = receive_array()
    centre = elements.mean(0)
    scene = build_scene(elements)
    tx_dirs = fibonacci_cone(TX_RAYS, torch.tensor([1.0, 0.0, 0.0]), SECTOR_DEG)
    tx_w = transmit_shading(tx_dirs)
    section = HULL_LENGTH / N_SECTIONS
    print(f"  {FREQ_HZ / 1e3:.0f} kHz, lambda = {LAM * 1e3:.1f} mm; "
          f"hull {HULL_LENGTH:.1f} m = {HULL_LENGTH / LAM:.0f} wavelengths")
    print(f"  {N_SECTIONS} sections of {section:.2f} m + 2 end caps, "
          f"at {TARGET_RANGE:.0f} m, {TARGET_DEPTH:.0f} m deep")
    print(f"  broadside sigma of the whole hull: "
          f"{HULL_RADIUS * HULL_LENGTH ** 2 / (2 * LAM):.1f} m^2 "
          f"(TS {10 * math.log10(HULL_RADIUS * HULL_LENGTH ** 2 / (2 * LAM)):+.1f} dB)")

    def arrivals_for(target, seed=3):
        return target_arrivals(scene, target, tx_dirs, n_rx_rays=RX_RAYS,
                               rx_half_angle_deg=RX_HALF_ANGLE_DEG,
                               tx_weights=tx_w, max_arrivals_per_leg=8,
                               return_leg="eigenray", tx_pattern=transmit_shading,
                               generator=torch.Generator().manual_seed(seed))

    # ---- 1. a smooth hull glints; discrete scatterers spread ----------------- #
    banner("a smooth hull glints -- it does not resolve into highlights")
    section_beamwidth = math.degrees(LAM / (2 * section))
    subtended = math.degrees(HULL_LENGTH / TARGET_RANGE)
    print(f"  each {section:.2f} m section has a beamwidth of ~lambda/2a = "
          f"{section_beamwidth:.2f} deg")
    print(f"  the body subtends {subtended:.2f} deg, so its end sections see the "
          f"sonar {subtended / 2:.2f} deg")
    print(f"  off their own broadside -- {subtended / 2 / section_beamwidth:.1f} "
          f"beamwidths out.  They return almost nothing.")

    broadside = build_target(90.0)
    with timed("  smooth hull, 7 highlights"):
        ext_arr = arrivals_for(broadside)
    ext_rng, ext_brg = spreads(ext_arr)
    shares = energy_by_highlight(ext_arr, broadside)
    best_share = max(sh for _, _, _, sh in shares)
    print("\n  where the hull's energy comes from:")
    for i, y, off, share in shares:
        tag = "  <- specular section" if share == best_share else ""
        print(f"    highlight {i} at y = {y:+5.2f} m: {share * 100:6.2f}%{tag}")
    glint_share = best_share

    banner("and the glint moves: slide the body along its own axis")
    print("  the specular section is the one whose broadside points at the sonar,")
    print("  so sliding the hull along itself hands the glint to a different section.")
    print("  Note which column moves: the glint stays at the specular point in the")
    print("  world, and travels along the *body*.  It is not a feature of the hull.")
    glint_offsets, glint_worlds = [], []
    print("\n    body centre      glint, world y    glint, along body    share")
    for slide in (-1.6, 0.0, 1.6):
        t = build_target(90.0)
        with torch.no_grad():
            t.position[1] = slide
        arr = arrivals_for(t)
        _, world_y, offset, share = max(energy_by_highlight(arr, t),
                                        key=lambda x: x[3])
        glint_offsets.append(offset)
        glint_worlds.append(world_y)
        print(f"    y = {slide:+5.2f} m        {world_y:+6.2f} m          "
              f"{offset:+6.2f} m         {share * 100:4.0f}%")

    banner("discrete scatterers do spread, which is what an image of a wreck shows")
    discrete = build_discrete(90.0)
    with timed("  discrete-scatterer body"):
        dis_arr = arrivals_for(discrete)
    dis_rng, dis_brg = spreads(dis_arr)

    point = PointTarget((TARGET_RANGE, 0.0, TARGET_DEPTH), 0.0, learnable=False)
    rx_dirs = return_fan(point, elements, RX_RAYS, half_angle_deg=45.0,
                         generator=torch.Generator().manual_seed(3))
    with timed("  point target, for contrast"):
        # Each leg's splat is sized to its own fan, exactly as `target_arrivals`
        # does it above -- otherwise this contrast would be measured on a
        # different yardstick than the target it is being compared against.
        tx_res = trace(scene, tx_dirs)
        inbound = extract_arrivals(tx_res, point.position, scene.freqs_khz,
                                   sigma_d=fan_sigma_d(tx_dirs, tx_res.arclen),
                                   ray_weights=tx_w, max_arrivals=8)
        rx_res = trace(_RelocatedScene(scene, point.position), rx_dirs)
        outbound = extract_arrivals(
            rx_res, centre, scene.freqs_khz,
            sigma_d=fan_sigma_d(rx_dirs, rx_res.arclen), max_arrivals=8)
        pt_arr = compose_arrivals(inbound, outbound, point)
    pt_rng, pt_brg = spreads(pt_arr)

    # Broadside the body lies across the line of sight, so its extent is a
    # *bearing* extent.  In range it is nearly flat -- the sagitta of a 4 m chord
    # at 40 m is 5 cm -- so range spread is multipath for every target here and
    # comparing it would show nothing.
    print(f"\n  body subtends {subtended:.2f} deg; range sagitta only "
          f"{HULL_LENGTH ** 2 / (8 * TARGET_RANGE) * 100:.1f} cm")
    print(f"  smooth hull:        weighted bearing spread {ext_brg:.2f} deg")
    print(f"  discrete scatterers: weighted bearing spread {dis_brg:.2f} deg")
    print(f"  point target:        weighted bearing spread {pt_brg:.2f} deg")

    # ---- 2. the aspect pattern ---------------------------------------------- #
    banner("aspect dependence, and what multipath does to it")
    # The sinc itself oscillates with a period of lambda / 2a = 0.54 deg here, so
    # sampling it every 10 deg would compare two arbitrary samples of a fast
    # oscillation and mean nothing.  What is comparable is its *envelope*,
    # cos^2(theta) / (k a sin theta)^2, which bounds the direct-path return.
    yaws = [90.0, 80.0, 60.0, 45.0, 20.0, 0.0]
    energies, envelope = [], []
    k = 2.0 * math.pi / LAM
    for yaw in yaws:
        t = build_target(yaw, caps=False)  # hull only: no cap to fill end-on
        arr = arrivals_for(t)
        energies.append(float((arr.amplitude ** 2).sum()) if arr.n_arrivals else 0.0)
        theta = math.radians(90.0 - yaw)
        arg = k * section * math.sin(theta)
        envelope.append(math.cos(theta) ** 2 if arg == 0.0
                        else (math.cos(theta) / arg) ** 2)
    ref, eref = energies[0], envelope[0]
    print("   yaw    off-broadside    measured    direct-path envelope   fill-in")
    fills = []
    for yaw, e, v in zip(yaws, energies, envelope):
        e_db = 10 * math.log10(max(e / ref, 1e-30))
        v_db = 10 * math.log10(max(v / eref, 1e-30))
        fills.append(e_db - v_db)
        print(f"  {yaw:5.1f}   {90 - yaw:6.1f} deg     {e_db:+8.1f} dB   {v_db:+11.1f} dB"
              f"   {e_db - v_db:+7.1f} dB")
    swing = 10 * math.log10(max(energies[0], 1e-300) / max(energies[-1], 1e-300))
    print(f"  broadside over end-on: {swing:+.1f} dB")
    print(f"  the measured pattern sits {min(fills[1:]):+.1f} to {max(fills[1:]):+.1f} dB "
          f"above the direct-path envelope.")
    print("  That is multipath filling the pattern in: a surface- or bottom-bounced")
    print("  ray strikes the hull at a different aspect than the direct one, so it")
    print("  is not in the same null.  A shallow-water FLS does not see a target's")
    print("  aspect nulls as deeply as a free-field calculation predicts.")

    # ---- 3. what the end caps are for --------------------------------------- #
    banner("end-on: the caps are the difference between a target and nothing")
    bare = arrivals_for(build_target(0.0, caps=False))
    capped = arrivals_for(build_target(0.0, caps=True))
    e_bare = float((bare.amplitude ** 2).sum()) if bare.n_arrivals else 0.0
    e_capped = float((capped.amplitude ** 2).sum()) if capped.n_arrivals else 0.0
    side = math.sqrt(math.pi) * HULL_RADIUS
    print(f"  hull only:   energy {e_bare:.4e}")
    print(f"  hull + caps: energy {e_capped:.4e}  "
          f"({10 * math.log10(max(e_capped / max(e_bare, 1e-300), 1e-300)):+.1f} dB)")
    print(f"  cap sigma at normal incidence: {(side * side / LAM) ** 2:.2f} m^2 "
          f"(TS {20 * math.log10(side * side / LAM):+.1f} dB)")

    # ---- image and plots ---------------------------------------------------- #
    steer, angles = azimuth_steering(421, SECTOR_DEG)
    shading = shading_window(N_ELEMENTS, "hamming")
    img_grid = make_time_grid(2.0 * (TARGET_RANGE - 12.0) / C,
                              2.0 * (TARGET_RANGE + 12.0) / C, 1400)
    with timed("  beamformed image of the extended target"):
        image = beamform(ext_arr, elements, scene.freqs_khz, img_grid, steer,
                         sigma_t=3e-5, shading=shading, steer_chunk=70)
    with timed("  beamformed image of the discrete-scatterer body"):
        image_dis = beamform(dis_arr, elements, scene.freqs_khz, img_grid, steer,
                             sigma_t=3e-5, shading=shading, steer_chunk=70)

    save(_plot_images(img_grid, angles, image, image_dis, broadside, discrete),
         "09_image.png")
    save(_plot_aspect(yaws, energies, envelope), "09_aspect.png")

    banner("acceptance")
    ok = check("a smooth hull's echo concentrates on one specular section",
               glint_share > 0.5, f"{glint_share * 100:.0f}% from one highlight")
    ok &= check("the glint travels along the body as the body slides",
                glint_offsets[0] > glint_offsets[1] > glint_offsets[2],
                "body offset " + " -> ".join(f"{o:+.2f}" for o in glint_offsets) + " m")
    ok &= check("while staying put at the specular point in the world",
                max(abs(y) for y in glint_worlds) < 0.5 * HULL_LENGTH / N_SECTIONS,
                "world y " + " / ".join(f"{y:+.2f}" for y in glint_worlds) + " m")
    ok &= check("discrete scatterers spread across the body's bearing extent",
                dis_brg > 0.3 * subtended,
                f"{dis_brg:.2f} deg weighted, of {subtended:.2f} deg subtended")
    ok &= check("a point target at the same place does not",
                pt_brg < 0.5 * dis_brg, f"{pt_brg:.2f} deg against {dis_brg:.2f}")
    ok &= check("the aspect pattern falls monotonically off broadside",
                all(a >= b * (1 - 1e-9) for a, b in zip(energies, energies[1:])),
                " > ".join(f"{10 * math.log10(max(e / ref, 1e-30)):.0f}" for e in energies)
                + " dB")
    # Multipath can only add paths the direct-path envelope leaves out, so the
    # measurement should sit on or above it.  Asserted with a decibel of slack
    # rather than as a law: both sides are normalised to broadside, which itself
    # contains multipath.
    ok &= check("measured pattern sits on or above the direct-path envelope",
                min(fills) > -1.0, f"lowest {min(fills):+.1f} dB")
    ok &= check("the hull all but vanishes end-on", swing > 30.0,
                f"{swing:+.1f} dB broadside over end-on")
    ok &= check("end caps keep the body visible end-on", e_capped > 100.0 * e_bare,
                f"{10 * math.log10(max(e_capped / max(e_bare, 1e-300), 1e-300)):+.1f} dB")
    return 0 if ok else 1


def _plot_images(grid, angles, image_hull, image_discrete, hull, discrete):
    import matplotlib.pyplot as plt
    import numpy as np

    rng = grid.detach().numpy() * C / 2.0
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharex=True, sharey=True)
    panels = ((axes[0], image_hull, hull,
               f"Smooth hull, {HULL_LENGTH:.0f} m broadside\nglints at one specular point"),
              (axes[1], image_discrete, discrete,
               "Discrete scatterers, same layout\nspread across the body"))
    for ax, img, target, title in panels:
        a = img[:, 0].detach().numpy()
        db = 10 * np.log10(np.maximum(a, a.max() * 1e-6) / a.max())
        m = ax.pcolormesh(rng, angles.numpy(), db, cmap="inferno", vmin=-30, vmax=0,
                          shading="auto")
        world = target.world_positions().detach().numpy()
        r = np.hypot(np.hypot(world[:, 0], world[:, 1]), world[:, 2] - VEHICLE_DEPTH)
        b = np.degrees(np.arctan2(world[:, 1], world[:, 0]))
        ax.plot(r, b, "+", color="#7fdfff", ms=9, mew=1.4)
        ax.set_xlabel("range (m)")
        ax.set_title(title, fontsize=10)
        ax.set_ylim(-9, 9)
    axes[0].set_ylabel("bearing (deg)")
    fig.colorbar(m, ax=axes, label="dB re each panel's own peak")
    fig.suptitle("Crosses mark the true highlight positions", fontsize=9, y=0.02)
    return fig


def _plot_aspect(yaws, energies, envelope):
    import matplotlib.pyplot as plt
    import numpy as np

    off = np.array([90.0 - y for y in yaws])
    ref, eref = energies[0], envelope[0]
    meas = 10 * np.log10(np.maximum(np.array(energies) / ref, 1e-30))
    env = 10 * np.log10(np.maximum(np.array(envelope) / eref, 1e-30))
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.plot(off, meas, "o-", lw=1.4, label="measured, full two-way pipeline")
    ax.plot(off, env, "s--", lw=1.2,
            label=r"direct-path envelope $\cos^2\theta/(ka\sin\theta)^2$")
    ax.fill_between(off, env, meas, alpha=0.15, color="#1f6f8b",
                    label="filled in by multipath")
    ax.set_xlabel("aspect off broadside (deg)")
    ax.set_ylabel("dB re broadside")
    ax.set_ylim(max(-90, env.min() - 5), 5)
    ax.set_title("Aspect dependence of a hull of cylinder sections")
    ax.grid(alpha=0.3, lw=0.4)
    ax.legend(fontsize=8)
    return fig


if __name__ == "__main__":
    raise SystemExit(main())
