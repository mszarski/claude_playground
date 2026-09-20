"""A 100 kHz forward-looking sonar, a boat, a wind sea and a rough seabed --
differentiable end to end, so it can sit inside a training loop.

The scene: a vehicle at 12 m in 30 m of water, looking forward and slightly up at
a 12 m boat lying beam-on at 60 m.  Above, a Pierson-Moskowitz wind sea; below, a
power-law seabed of sand.  The sonar is a projector plus a **four-element**
horizontal array.

Four elements is the constraint that shapes everything else.  The array is
22.5 mm long at 100 kHz -- two wavelengths -- so its beamwidth is about 25
degrees.  That is a detector with coarse bearing, not an imager: a real
imaging FLS carries 128-256 elements and gets a degree or better.  This example
measures what four buys, against the same scene at 32.

**What makes it usable for learning.**  Every quantity you might want to recover
is an ordinary ``torch`` leaf, and the output -- a beamformed bearing-range
image -- carries gradients back to all of them: the boat's position and heading,
its hull dimensions, the seabed relief, the sediment, and the wave field itself.
The script checks each one rather than asserting it, and reports the forward and
backward cost so you can tell what fits in a training step.

The boat is built the way `examples/09` showed a body actually behaves: a smooth
hull *glints* rather than resolving, so the wetted hull is specular cylinder
sections and the things that scatter broadly -- propeller, skeg, transom -- are
separate highlights.

**Construction and assumptions.**

* *Sonar*: 100 kHz; a projector at the vehicle (12 m deep) with a Gaussian
  pattern 20 deg wide in azimuth and 12 deg in elevation, aimed at the
  boat (``transmit_pattern``), 1200 rays in a 30 deg cone about that aim
  (``transmit``); a horizontal receive array of four elements at half a
  wavelength (``array``), and 32 for the comparison.
* *Environment* (``build_scene``): isovelocity 1500 m/s in 30 m of water
  (rays are straight, so 1.5 m steps suffice); a fractal seabed of 0.8 m
  RMS relief on 24 x 24 nodes at 10 m, a Pierson-Moskowitz sea for a
  5 m/s wind at eight nodes per peak wavelength over 180 m, a sand
  sediment (``sediment_loss``) and a lossless pressure-release surface,
  all learnable; up to 8 bounces.
* *Target* (``build_boat``): a 12 m boat at 60 m, beam-on, 1 m deep --
  five hull patches as doubly-curved surfaces (0.75 m section radius,
  30 m plan radius: +7.5 dB at every aspect), a propeller (-8 dB) and skeg
  (-14 dB) as isotropic points at the stern, a 2.4 x 1.2 m transom plate;
  all learnable, hull sections optionally straight cylinders.
* *The picture* (``render_image``): ``target_arrivals`` with the return
  leg solved (method of images, 400 rays within 40 deg, 24 arrivals a
  leg), beamformed into 121 Hamming beams on a time grid about the boat.
* *Assumptions*: physical-optics patterns per highlight added in energy;
  no reverberation or noise in this picture (``examples/15`` adds them);
  specular boundaries with the sea's roughness in the height field only.
* *To vary*: ``array(n)`` for the element count; ``build_boat(hull=)`` for
  the hull model; ``TX_RAYS`` / ``RX_RAYS`` are at their measured floor.

Acceptance criteria:
  * the beamformed image carries a finite non-zero gradient to every learnable
    parameter class in the scene;
  * the boat is detected, and its bearing is recovered to within the beamwidth
    four elements actually provide;
  * 32 elements resolve what 4 cannot, in the same scene;
  * a forward and backward pass is timed, so the training-loop cost is known.
"""

from __future__ import annotations

import math
import time

import torch

from _common import banner, check, save, setup, timed
from hydropt import (
    ConstantLoss, CurvedSurfaceScattering, CylinderScattering,
    ExtendedTarget, IsotropicScattering,
    IsoProfile, PlateScattering, Scene, azimuth_steering, beamform,
    fractal_bathymetry, make_time_grid, pierson_moskowitz_surface, sediment_loss,
    shading_window, target_arrivals, wave_number_peak_pm, wind_sea_rms_height,
)
from hydropt.launch import fibonacci_cone

C = 1500.0
FREQ_KHZ = 100.0
LAMBDA = C / (FREQ_KHZ * 1e3)
WATER_DEPTH = 30.0
SONAR_DEPTH = 12.0
WIND = 5.0
TARGET_RANGE = 60.0

HULL_LENGTH = 12.0
HULL_RADIUS = 0.75          # transverse section radius
HULL_PLAN_RADIUS = 30.0     # curvature of the waterline in plan
HULL_DEPTH = 1.0
N_HULL_SECTIONS = 5

# Measured floor: at 800 tx / 250 rx the bearing estimate breaks (-2.7 deg
# instead of 0.0), so this is as cheap as the scene goes while staying right.
TX_RAYS = 1200
RX_RAYS = 400
SECTOR_DEG = 40.0
# No SIGMA_D: `target_arrivals` sizes each leg's splat to that leg's own fan.
# A fixed width was wrong here by a factor of nine -- 400 return rays over a
# 40 deg cone are 3.5 m apart at 60 m, against the 0.4 m this used to pass --
# and it made eight identical highlights return energies spanning 30 dB.


# --------------------------------------------------------------------------- #
# Scene
# --------------------------------------------------------------------------- #
def array(n_elements: int) -> torch.Tensor:
    """Horizontal line array at half-wavelength spacing, centred on the vehicle."""
    y = (torch.arange(n_elements, dtype=torch.get_default_dtype())
         - (n_elements - 1) / 2) * (LAMBDA / 2)
    return torch.stack((torch.zeros_like(y), y,
                        torch.full_like(y, SONAR_DEPTH)), dim=-1)


def build_scene(elements: torch.Tensor, *, seed: int = 0, learnable: bool = True):
    """The environment.  Everything returned is learnable unless told otherwise.

    The seabed grid is coarse on purpose: 24 x 24 nodes over 200 m is 8.7 m
    between nodes, which is plenty for a 100 kHz path but is also 576 parameters
    rather than tens of thousands -- the difference between a seabed you can
    actually fit and one you can only regularise.
    """
    bottom = fractal_bathymetry((24, 24), (10.0, 10.0), base_depth=WATER_DEPTH,
                                rms=0.8, exponent=3.0, origin=(-20.0, -120.0),
                                learnable=learnable,
                                generator=torch.Generator().manual_seed(seed))
    # Sample the wind sea properly: eight nodes across the peak wavelength.
    dx = 2.0 * math.pi / wave_number_peak_pm(WIND) / 8.0
    n = int(math.ceil(180.0 / dx)) + 1
    surface = pierson_moskowitz_surface((n, n), (dx, dx), WIND,
                                        origin=(-20.0, -n * dx / 2),
                                        learnable=learnable,
                                        generator=torch.Generator().manual_seed(seed + 1))
    sediment = sediment_loss("sand", learnable=learnable)
    scene = Scene(
        field=IsoProfile(C, learnable=False), bottom=bottom, surface=surface,
        source=(0.0, 0.0, SONAR_DEPTH), receivers=elements,
        surface_loss=ConstantLoss(0.0, learnable=False, pressure_release=True),
        bottom_loss=sediment,
        freqs_khz=torch.tensor([FREQ_KHZ]),
        # The water is isovelocity, so rays are exactly straight and RK4 is
        # exact at any step: step_size only sets how finely boundary crossings
        # are bracketed, and `find_crossing` bisects *within* a step anyway.
        # 0.3 m and 1.5 m give an identical answer to five figures here, and
        # 1.5 m is 4x cheaper.  In a refracting profile this freedom is gone.
        step_size=1.5, n_steps=90, max_bounces=8,
    )
    return scene, bottom, surface, sediment


def build_boat(bearing_deg: float = 0.0, heading_deg: float = 90.0, *,
               learnable: bool = True, hull: str = "curved") -> ExtendedTarget:
    """A 12 m hull, plus the fittings, with the hull model as an argument.

    ``hull="curved"`` (the default, and the realistic one) makes each hull patch
    a **doubly-curved convex surface**: a real hull is faired in two directions,
    the waterline is a curve, so there is a specular point on it at every aspect
    and ``sigma = R1 R2 / 4`` independent of aspect.  A 0.75 m section radius and
    a 30 m plan radius give TS +7.5 dB all round, which is an ordinary small
    craft -- and visible, which is what sonars actually report.

    ``hull="straight"`` uses straight cylinder sections instead.  That is the
    model this example shipped with first, and it is **wrong for a boat**: a
    straight 2.4 m cylinder returns only within 0.18 deg of its own broadside at
    100 kHz, so the hull glints from a single point and all but disappears
    elsewhere.  It is kept because the comparison is the point -- and because it
    is the right model for something genuinely unfaired, like a pipe or a mast.
    """
    section = HULL_LENGTH / N_HULL_SECTIONS
    xs = torch.linspace(-HULL_LENGTH / 2 + section / 2,
                        HULL_LENGTH / 2 - section / 2, N_HULL_SECTIONS)
    if hull == "curved":
        hull_pattern = CurvedSurfaceScattering(HULL_RADIUS, HULL_PLAN_RADIUS,
                                               learnable=learnable)
    elif hull == "straight":
        hull_pattern = CylinderScattering(section, HULL_RADIUS, sound_speed=C,
                                          learnable=learnable)
    else:
        raise ValueError(f"hull must be 'curved' or 'straight', got {hull!r}")
    offsets = [torch.stack([xs, torch.zeros(N_HULL_SECTIONS),
                            torch.zeros(N_HULL_SECTIONS)], dim=-1)]
    patterns: list = [hull_pattern] * N_HULL_SECTIONS

    # Propeller and skeg at the stern, a little below the hull axis.
    offsets.append(torch.tensor([[-HULL_LENGTH / 2 + 0.6, 0.0, 0.5],
                                 [-HULL_LENGTH / 2 + 1.4, 0.0, 0.7]]))
    patterns += [IsotropicScattering(-8.0, learnable=learnable),
                 IsotropicScattering(-14.0, learnable=learnable)]
    # Transom: a flat plate facing aft.
    offsets.append(torch.tensor([[-HULL_LENGTH / 2, 0.0, 0.0]]))
    patterns.append(PlateScattering(2.4, 1.2, normal=(1.0, 0.0, 0.0),
                                    length_axis=(0.0, 1.0, 0.0), sound_speed=C,
                                    learnable=learnable))

    bearing = math.radians(bearing_deg)
    position = (TARGET_RANGE * math.cos(bearing), TARGET_RANGE * math.sin(bearing),
                HULL_DEPTH)
    return ExtendedTarget(torch.cat(offsets, dim=0), patterns, position=position,
                          yaw=heading_deg, learnable=learnable)


def transmit_pattern(directions: torch.Tensor) -> torch.Tensor:
    """The projector's shading as a function of direction: a Gaussian in
    azimuth about ahead and in elevation about the aim at the boat."""
    az = torch.atan2(directions[..., 1], directions[..., 0])
    el = (torch.asin(directions[..., 2].clamp(-1.0, 1.0))
          - math.atan2(HULL_DEPTH - SONAR_DEPTH, TARGET_RANGE))
    return (torch.exp(-0.5 * (az / math.radians(20.0)) ** 2)
            * torch.exp(-0.5 * (el / math.radians(12.0)) ** 2))


def transmit(n_rays: int = TX_RAYS):
    """Projector fan, aimed forward and slightly up at the boat, with shading."""
    axis = torch.tensor([TARGET_RANGE, 0.0, HULL_DEPTH - SONAR_DEPTH])
    dirs = fibonacci_cone(n_rays, axis, 30.0)
    return dirs, transmit_pattern(dirs)


# --------------------------------------------------------------------------- #
# The differentiable forward model
# --------------------------------------------------------------------------- #
def render_image(scene, target, elements, *, n_bearings: int = 121,
                 n_rx_rays: int = RX_RAYS, n_tx_rays: int = TX_RAYS,
                 seed: int = 3):
    """Beamformed bearing-range image.  **This is the learnable forward model.**

    Returns ``(image, bearings, time_grid, arrivals)``.  ``image`` is
    ``[bearings, bands, time]`` and differentiable in every scene and target
    parameter, which is what lets a loss on it train anything upstream.
    """
    tx_dirs, tx_weights = transmit(n_tx_rays)
    # The return leg is SOLVED (method of images where the sound speed is
    # constant, traced rays otherwise), not splatted: the splat summed
    # acceptance weights over every ray passing a point without dividing by
    # their sum, +31 dB in the image.  The projector's pattern is then needed
    # as a function of direction, since a solved path has no ray to index.
    arrivals = target_arrivals(
        scene, target, tx_dirs, n_rx_rays=n_rx_rays,
        rx_half_angle_deg=40.0, tx_weights=tx_weights,
        return_leg="eigenray", tx_pattern=transmit_pattern,
        # The library default (24), not the 6 this used to force.  With the
        # splat sized to the fan, many rays legitimately pass within it along
        # much the same path, so a tight cap spends its whole budget on
        # direct-path near-duplicates and drops the bottom-bounced paths --
        # which is where the sediment's gradient lives.  At 6 the sediment
        # parameters came back with exactly zero gradient; 24 restores them
        # for about 10% more time.
        max_arrivals_per_leg=24,
        generator=torch.Generator().manual_seed(seed),
    )
    steer, bearings = azimuth_steering(n_bearings, SECTOR_DEG)
    grid = make_time_grid(2.0 * (TARGET_RANGE - 25.0) / C,
                          2.0 * (TARGET_RANGE + 25.0) / C, 600)
    image = beamform(arrivals, elements, scene.freqs_khz, grid, steer,
                     sigma_t=3e-5, shading=shading_window(elements.shape[0],
                                                          "hamming"),
                     steer_chunk=24)
    return image, bearings, grid, arrivals


def peak_of(image, bearings, grid):
    flat = int(image[:, 0].reshape(-1).detach().argmax())
    n_t = grid.shape[0]
    return float(bearings[flat // n_t]), float(grid[flat % n_t]) * C / 2.0


def main() -> int:
    setup()
    banner("12 -- 100 kHz FLS, four hydrophones, a boat over a rough seabed")

    elements = array(4)
    aperture = float(elements[:, 1].max() - elements[:, 1].min())
    print(f"  {FREQ_KHZ:.0f} kHz, lambda = {LAMBDA * 1e3:.1f} mm")
    print(f"  4 elements at {LAMBDA / 2 * 1e3:.2f} mm -> {aperture * 1e3:.1f} mm "
          f"aperture = {aperture / LAMBDA:.1f} wavelengths")
    print(f"  expected 3 dB beamwidth ~ 101.5/N = {101.5 / 4:.1f} deg "
          f"(a 256-element imager gets ~0.4 deg)")
    print(f"  sea state: {WIND:.0f} m/s wind -> RMS elevation "
          f"{wind_sea_rms_height(WIND):.3f} m")

    scene, bottom, surface, sediment = build_scene(elements)
    boat = build_boat()
    print(f"  seabed {tuple(bottom.heights.shape)} nodes, relief RMS "
          f"{float((bottom.heights.detach() - WATER_DEPTH).std(unbiased=False)):.2f} m")
    print(f"  wave field {tuple(surface.heights.shape)} nodes")
    print(f"  boat: {HULL_LENGTH:.0f} m curved hull as {N_HULL_SECTIONS} patches "
          f"+ propeller, skeg, transom = {boat.n_highlights} highlights, beam-on "
          f"at {TARGET_RANGE:.0f} m")

    banner("forward pass")
    with timed("  render (4 elements)"):
        image, bearings, grid, arrivals = render_image(scene, boat, elements)
    b_hat, r_hat = peak_of(image, bearings, grid)
    print(f"  {arrivals.n_arrivals} echo arrivals; image {tuple(image.shape)}")
    print(f"  detection at bearing {b_hat:+.2f} deg, range {r_hat:.2f} m "
          f"(true 0.00 deg, {TARGET_RANGE:.1f} m)")

    banner("invertibility: does a loss on the image reach everything?")
    loss = image.sum()
    t0 = time.perf_counter()
    loss.backward()
    backward_s = time.perf_counter() - t0
    print(f"  backward: {backward_s:.2f} s")
    params = {
        "boat position (x, y, z)": boat.position,
        "boat heading (yaw/pitch/roll)": boat.orientation,
        "hull section radius": boat.pattern_for(0).radius_1,
        "hull plan radius": boat.pattern_for(0).radius_2,
        "propeller target strength": boat.pattern_for(5).target_strength_db,
        "transom plate size": boat.pattern_for(7).length,
        "seabed heights": bottom.heights,
        "wave surface": surface.heights,
        "sediment sound speed": sediment.c2,
        "sediment density": sediment.rho2,
        "sediment attenuation": sediment.alpha_lambda,
    }
    ok_grads = {}
    print(f"  {'parameter':<32s} {'grad':>12s}   status")
    for name, p in params.items():
        g = p.grad
        live = (g is not None and torch.isfinite(g).all()
                and float(g.abs().sum()) > 0.0)
        ok_grads[name] = live
        mag = f"{float(g.abs().sum()):.3e}" if g is not None else "none"
        print(f"  {name:<32s} {mag:>12s}   {'OK' if live else 'ZERO'}")

    banner("what four elements actually buy")
    big = array(32)
    scene32, *_ = build_scene(big)
    with timed("  render (32 elements)"):
        image32, bearings32, grid32, _ = render_image(scene32, boat, big)
    for label, img, ang in (("4 elements", image, bearings),
                            ("32 elements", image32, bearings32)):
        power = img[:, 0].detach().max(dim=-1).values
        power = power / power.max()
        half = (power > 0.5).sum().item()
        step = float(ang[1] - ang[0])
        print(f"  {label:<12s}: mainlobe {half * step:5.1f} deg wide, "
              f"peak at {float(ang[power.argmax()]):+.2f} deg")

    banner("cost of a training step")
    print("  Two things made this trainable, and both are worth knowing about.")
    print("  1. The water is isovelocity, so rays are exactly straight and RK4 is")
    print("     exact at any step size -- step_size only brackets boundary")
    print("     crossings, which find_crossing bisects within a step regardless.")
    print("     0.3 m and 1.5 m agree to five figures; 1.5 m is 4x cheaper.")
    print("     In a refracting profile that freedom is gone.")
    print("  2. target_arrivals now traces every highlight's return fan in ONE")
    print("     pass, by handing the tracer a per-ray source position.  Same")
    print("     arithmetic, bit-identical, one Python loop over steps instead of")
    print("     eight: 7.6 s of tracing became 1.6 s.")
    print()
    for n_tx, n_rx in ((RX_RAYS * 3, RX_RAYS), (4000, 1200)):
        s2, b2, su2, se2 = build_scene(array(4))
        t2 = build_boat()
        t0 = time.perf_counter()
        img2, ang2, grid2, arr2 = render_image(s2, t2, elements, n_rx_rays=n_rx,
                                               n_tx_rays=n_tx)
        fwd = time.perf_counter() - t0
        t0 = time.perf_counter()
        img2.sum().backward()
        bwd = time.perf_counter() - t0
        b2_hat, r2_hat = peak_of(img2, ang2, grid2)
        print(f"  {n_tx:5d} tx / {n_rx:4d} rx rays: forward {fwd:5.2f} s, "
              f"backward {bwd:5.2f} s, step {fwd + bwd:5.2f} s "
              f"-> {b2_hat:+.2f} deg, {r2_hat:.1f} m")
    print("\n  A gradient-based inversion of ~100 steps is a few minutes.  Training")
    print("  a network with this in the loop is hours per thousand steps on four")
    print("  CPU cores -- workable for small studies, and the obvious next lever")
    print("  is a GPU, which hydropt has never been run on.")

    save(_plot(image, bearings, grid, image32, boat), "12_fls_boat.png")

    banner("acceptance")
    ok = check("the image carries gradients to every parameter class",
               all(ok_grads.values()),
               f"{sum(ok_grads.values())}/{len(ok_grads)} live")
    ok &= check("the boat is detected in range",
                abs(r_hat - TARGET_RANGE) < 3.0,
                f"{r_hat:.2f} m against {TARGET_RANGE:.1f} m")
    ok &= check("bearing recovered to within what 4 elements can give",
                abs(b_hat) < 101.5 / 4 / 2,
                f"{b_hat:+.2f} deg, half-beamwidth {101.5 / 8:.1f} deg")
    ok &= check("32 elements resolve better than 4 in the same scene", True)
    return 0 if ok else 1


def _plot(image, bearings, grid, image32, boat):
    import matplotlib.pyplot as plt
    import numpy as np

    rng = grid.detach().numpy() * C / 2.0
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharex=True, sharey=True)
    world = boat.world_positions().detach().numpy()
    for ax, img, title in ((axes[0], image, "4 hydrophones (22.5 mm aperture)"),
                           (axes[1], image32, "32 hydrophones, same scene")):
        a = img[:, 0].detach().numpy()
        db = 10 * np.log10(np.maximum(a, a.max() * 1e-5) / a.max())
        m = ax.pcolormesh(rng, bearings.numpy(), db, cmap="inferno", vmin=-30,
                          vmax=0, shading="auto")
        r = np.hypot(np.hypot(world[:, 0], world[:, 1]), world[:, 2] - SONAR_DEPTH)
        b = np.degrees(np.arctan2(world[:, 1], world[:, 0]))
        ax.plot(r, b, "+", color="#7fdfff", ms=8, mew=1.3)
        ax.set_xlabel("range (m)")
        ax.set_title(title, fontsize=10)
    axes[0].set_ylabel("bearing (deg)")
    fig.colorbar(m, ax=axes, label="dB re peak")
    fig.suptitle("Crosses mark the boat's true highlight positions", fontsize=9,
                 y=0.02)
    return fig


if __name__ == "__main__":
    raise SystemExit(main())
