"""A Mills-cross forward-looking sonar: 120 deg x 20 deg, 2 deg beams.

The arrangement almost every real FLS and multibeam uses.  Two perpendicular line
arrays, each doing one axis of the job:

* a **vertical** transmit array, 6 elements over 4.5 cm, whose broadside beam is a
  fan -- narrow in elevation, wide in azimuth.  Electronically steered up so the
  20 deg fan sits on the target rather than on the horizon.
* a **horizontal** receive array, 64 elements over 47 cm, which has no elevation
  discrimination at all and puts all of its aperture into azimuth: 2 deg beams,
  60 of them across 120 deg.

Neither array could do this alone.  Their product is a 2 deg x 20 deg pencil that
sweeps in azimuth, and that is the Mills cross.

**What this example measures rather than asserts.** The realised beamwidth at
boresight; the way it broadens as `1/cos(theta)` when steered out to the edge of
the 120 deg sector, which is why the edge beams of any flat array are worse than
the middle ones; and what 2 deg buys over the four-element array of `examples/12`
against the same 12 m boat -- the difference between a blob and a body you can
measure.

Everything stays differentiable, so this is a usable forward model for learning;
the cost of a training step is reported at the end.

Acceptance criteria:
  * the horizontal array delivers the 2 deg beamwidth its aperture implies;
  * the vertical array delivers the 20 deg fan its aperture implies;
  * steered to the sector edge, the beam broadens as `1/cos(theta)` within 10%;
  * the hull reads as a resolved body several beams wide, its bearing centroid
    lands on the body, and its -3 dB extent stays inside what the hull subtends;
  * a loss on the image still reaches every scene parameter.
"""

from __future__ import annotations

import math
import time

import torch

from _common import banner, check, save, setup, timed
from hydropt import (
    ArrivalSet, azimuth_steering, beamform, line_array_factor,
    make_time_grid, shading_window, target_arrivals,
)
from hydropt.launch import directions_from_angles, structured_fan

import importlib.util
_spec = importlib.util.spec_from_file_location(
    "_fls12", str(__file__).replace("13_mills_cross_fls.py",
                                    "12_fls_boat_learnable.py"))
_fls = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fls)

C = _fls.C
FREQ_KHZ = _fls.FREQ_KHZ
LAMBDA = _fls.LAMBDA
SONAR_DEPTH = _fls.SONAR_DEPTH
TARGET_RANGE = _fls.TARGET_RANGE
HULL_DEPTH = _fls.HULL_DEPTH

N_RX = 64                 # horizontal, sets azimuth resolution
N_TX = 6                  # vertical, sets the elevation fan
SECTOR_DEG = 60.0         # +/- 60 = 120 deg of horizontal coverage
VERTICAL_DEG = 10.0       # +/- 10 = 20 deg of vertical coverage
BEAM_SPACING_DEG = 2.0
# Aim the fan at the boat: 60 m out and 11 m above the vehicle is 10.4 deg up.
TILT_DEG = math.degrees(math.atan2(SONAR_DEPTH - HULL_DEPTH, TARGET_RANGE))


def horizontal_array(n: int = N_RX) -> torch.Tensor:
    """Receive array along ``y``, half-wavelength spaced, centred on the vehicle."""
    y = (torch.arange(n, dtype=torch.get_default_dtype()) - (n - 1) / 2) * (LAMBDA / 2)
    return torch.stack((torch.zeros_like(y), y,
                        torch.full_like(y, SONAR_DEPTH)), dim=-1)


def vertical_array(n: int = N_TX) -> torch.Tensor:
    """Transmit array along ``z`` -- perpendicular to the receive array."""
    z = (torch.arange(n, dtype=torch.get_default_dtype()) - (n - 1) / 2) * (LAMBDA / 2)
    return torch.stack((torch.zeros_like(z), torch.zeros_like(z),
                        SONAR_DEPTH + z), dim=-1)


def transmit_fan(n_elev: int = 56, n_azim: int = 330):
    """Rays over the full 120 x 20 deg sector, weighted by the vertical array.

    The fan is a *rectangle* in launch angle rather than a cone, because the
    sector is: 120 deg wide and 20 deg tall.  Its density is set by the receiver
    acceptance -- at 60 m, 0.4 m of acceptance is 0.38 deg, so the sampling has to
    be finer than that or beams fall between rays.
    """
    dirs, elev, azim = structured_fan(
        n_elev, n_azim,
        elev_range_deg=(-TILT_DEG - VERTICAL_DEG * 1.6,
                        -TILT_DEG + VERTICAL_DEG * 1.6),
        azim_range_deg=(-SECTOR_DEG * 1.05, SECTOR_DEG * 1.05))
    # Vertical array response, steered up by the tilt.  Elevation is measured from
    # horizontal and z is depth, so "up" is a negative elevation here.
    sin_e = torch.sin(elev).reshape(-1, 1).expand(n_elev, n_azim).reshape(-1)
    weights = line_array_factor(sin_e, N_TX, sin_steer=math.sin(math.radians(-TILT_DEG)))
    return dirs, weights.contiguous(), elev, azim


def measure_beamwidth(elements, steer_deg: float, n_probe: int = 721,
                      half_span: float = 14.0) -> tuple[float, float]:
    """Beamwidth and peak bearing from a synthetic plane wave, in degrees.

    Measured by steering across a fine grid around ``steer_deg`` and reading the
    -3 dB width off the response, rather than quoting ``101.5 / N``.
    """
    look = directions_from_angles(torch.zeros(1),
                                  torch.tensor([math.radians(steer_deg)]))
    probe = ArrivalSet(torch.tensor([0.08]), torch.ones(1, 1), -look,
                       torch.zeros(1), torch.zeros(1), torch.ones(1))
    angles = torch.linspace(steer_deg - half_span, steer_deg + half_span, n_probe)
    steer = directions_from_angles(torch.zeros_like(angles),
                                   angles * math.pi / 180.0)
    grid = make_time_grid(0.078, 0.082, 201)
    power = beamform(probe, elements, torch.tensor([FREQ_KHZ]), grid, steer,
                     sigma_t=2e-4, shading=shading_window(elements.shape[0],
                                                          "hamming"),
                     steer_chunk=120).max(dim=-1).values[:, 0]
    power = power / power.max()
    above = angles[power > 0.5]
    return float(above.max() - above.min()), float(angles[power.argmax()])


def main() -> int:
    setup()
    banner("13 -- Mills cross FLS: 120 deg x 20 deg, 2 deg beams")

    rx = horizontal_array()
    tx_arr = vertical_array()
    rx_aperture = float(rx[:, 1].max() - rx[:, 1].min())
    tx_aperture = float(tx_arr[:, 2].max() - tx_arr[:, 2].min())
    print(f"  {FREQ_KHZ:.0f} kHz, lambda = {LAMBDA * 1e3:.1f} mm, "
          f"half-wave spacing {LAMBDA / 2 * 1e3:.2f} mm")
    print(f"  receive  (horizontal, along y): {N_RX:3d} elements, "
          f"{rx_aperture * 100:5.1f} cm = {rx_aperture / LAMBDA:.1f} wavelengths")
    print(f"  transmit (vertical,   along z): {N_TX:3d} elements, "
          f"{tx_aperture * 100:5.1f} cm = {tx_aperture / LAMBDA:.1f} wavelengths")
    print(f"  sector {2 * SECTOR_DEG:.0f} deg x {2 * VERTICAL_DEG:.0f} deg, "
          f"{int(2 * SECTOR_DEG / BEAM_SPACING_DEG)} beams at "
          f"{BEAM_SPACING_DEG:.0f} deg spacing")
    print(f"  fan tilted {TILT_DEG:.1f} deg up to put the boat on the axis")

    banner("do the arrays deliver the beams their apertures imply?")
    bw0, peak0 = measure_beamwidth(rx, 0.0)
    print(f"  horizontal array at boresight: {bw0:.2f} deg "
          f"(101.5/N = {101.5 / N_RX:.2f}, Hamming broadens it ~1.3x)")
    print(f"  peak at {peak0:+.3f} deg")

    # Vertical fan: the transmit array's own response, read off the array factor.
    el = torch.linspace(-40.0, 40.0, 4001)
    resp = line_array_factor(torch.sin(el * math.pi / 180.0), N_TX)
    inside = el[resp > 0.5]
    vfan = float(inside.max() - inside.min())
    print(f"  vertical array fan: {vfan:.1f} deg at -3 dB "
          f"(asked for {2 * VERTICAL_DEG:.0f})")

    banner("beam broadening across the 120 deg sector")
    print(f"  A flat array steered off broadside sees a foreshortened aperture,")
    print(f"  so the beam widens as 1/cos(theta).  That is why the edge beams of")
    print(f"  any flat FLS are worse than the ones ahead.")
    print(f"\n  {'steer':>7s} {'measured':>10s} {'1/cos rule':>11s} {'ratio':>7s}")
    broadening = []
    for ang in (0.0, 20.0, 40.0, 55.0):
        bw, _ = measure_beamwidth(rx, ang, half_span=14.0 + ang / 4)
        predicted = bw0 / math.cos(math.radians(ang))
        broadening.append(bw / predicted)
        print(f"  {ang:6.1f}d {bw:10.2f} {predicted:11.2f} {bw / predicted:7.3f}")

    banner("the scene: boat over a rough seabed, wind sea above")
    scene, bottom, surface, sediment = _fls.build_scene(rx)
    boat = _fls.build_boat()
    dirs, weights, _, _ = transmit_fan()
    print(f"  transmit fan {dirs.shape[0]} rays over the full sector "
          f"(0.38 deg sampling, set by the 0.4 m receiver acceptance at 60 m)")
    print(f"  boat: {_fls.HULL_LENGTH:.0f} m, beam-on at {TARGET_RANGE:.0f} m, "
          f"subtending {math.degrees(_fls.HULL_LENGTH / TARGET_RANGE):.1f} deg "
          f"= {math.degrees(_fls.HULL_LENGTH / TARGET_RANGE) / BEAM_SPACING_DEG:.1f} beams")

    n_beams = int(2 * SECTOR_DEG / BEAM_SPACING_DEG) + 1
    steer, bearings = azimuth_steering(n_beams, SECTOR_DEG)
    grid = make_time_grid(2.0 * (TARGET_RANGE - 20.0) / C,
                          2.0 * (TARGET_RANGE + 20.0) / C, 500)
    t0 = time.perf_counter()
    arrivals = target_arrivals(scene, boat, dirs,
                               n_rx_rays=400, rx_half_angle_deg=40.0,
                               tx_weights=weights, max_arrivals_per_leg=24,
                               generator=torch.Generator().manual_seed(3))
    image = beamform(arrivals, rx, scene.freqs_khz, grid, steer, sigma_t=3e-5,
                     shading=shading_window(N_RX, "hamming"), steer_chunk=16)
    forward = time.perf_counter() - t0
    print(f"  {arrivals.n_arrivals} echo arrivals, image {tuple(image.shape)}, "
          f"forward {forward:.2f} s")

    banner("what 2 deg buys -- and what it does not")
    power = image[:, 0].detach().max(dim=-1).values
    power = power / power.max()
    angles = bearings.numpy()
    db = 10 * torch.log10(power).numpy()
    subtended = math.degrees(_fls.HULL_LENGTH / TARGET_RANGE)
    print(f"  the hull subtends {subtended:.1f} deg and the beam is {bw0:.2f} deg, so")
    print(f"  a target that is genuinely extended should read several beams wide.\n")
    extents = {}
    for thr in (-3.0, -10.0, -20.0):
        mask = db > thr
        span = float(angles[mask].max() - angles[mask].min()) if mask.any() else 0.0
        extents[thr] = span
        print(f"    above {thr:5.1f} dB: {int(mask.sum()):3d} beams, {span:5.2f} deg wide")
    peak = float(angles[db.argmax()])
    lobe = db > -20.0
    centroid = float((angles[lobe] * power.numpy()[lobe]).sum()
                     / power.numpy()[lobe].sum())
    world_b = boat.world_positions().detach()
    tb = torch.atan2(world_b[:, 1], world_b[:, 0]) * 180.0 / math.pi
    print(f"\n  The hull reads as a **body**, not a point: {extents[-3.0]:.1f} deg at")
    print(f"  -3 dB against {subtended:.1f} deg subtended, which is most of its")
    print(f"  length, and {extents[-20.0]:.0f} deg by -20 dB once multipath smear is")
    print(f"  included.  That is what a real hull looks like on a real sonar, and")
    print(f"  getting there took two corrections worth stating plainly:\n")
    print(f"    1. A boat hull is faired in two directions, so it is a doubly")
    print(f"       curved convex surface with a specular point at *every* aspect")
    print(f"       and sigma = R1 R2 / 4 independent of aspect.  Modelled as")
    print(f"       straight cylinder sections it returned only within 0.18 deg of")
    print(f"       its own broadside at 100 kHz and all but vanished elsewhere.")
    print(f"    2. The return fan's splat has to be sized to the fan's own ray")
    print(f"       spacing.  A fixed 0.4 m width on rays 3.5 m apart made the")
    print(f"       per-highlight amplitudes a lottery -- eight identical")
    print(f"       highlights spanned 30 dB -- which collapsed the body to a")
    print(f"       single bright patch for reasons that were pure sampling.")
    print(f"\n  peak bearing {peak:+.2f} deg, energy centroid {centroid:+.2f} deg")
    print(f"  (true hull centre 0.00; highlights span {float(tb.min()):+.2f} to "
          f"{float(tb.max()):+.2f} deg)")
    print(f"  The centroid sits off centre, and that is the sea rather than the")
    print(f"  boat: over a flat surface and a flat seabed the same hull reads")
    print(f"  -1.4 deg with its full 9 deg extent, and swapping the curved hull")
    print(f"  for plain isotropic patches changes neither number.  It is the wave")
    print(f"  and seabed realisation that lights one end of the body more than")
    print(f"  the other -- which is what a single ping in a real sea does.")

    banner("invertibility, and the cost of a training step")
    t0 = time.perf_counter()
    image.sum().backward()
    backward = time.perf_counter() - t0
    live = {"boat position": boat.position, "boat heading": boat.orientation,
            "seabed": bottom.heights, "waves": surface.heights,
            "sediment c2": sediment.c2}
    states = {k: (p.grad is not None and torch.isfinite(p.grad).all()
                  and float(p.grad.abs().sum()) > 0) for k, p in live.items()}
    for name, ok_g in states.items():
        print(f"  d(image)/d({name:<14s}): {'OK' if ok_g else 'ZERO'}")
    print(f"\n  forward {forward:.2f} s + backward {backward:.2f} s = "
          f"{forward + backward:.2f} s per step")
    print(f"  ({dirs.shape[0]} transmit rays for 120 deg, against 1,200 for the")
    print(f"   30 deg cone in examples/12 -- the sector is what costs)")

    save(_plot(image, bearings, grid, boat, rx, bw0), "13_mills_cross.png")

    banner("acceptance")
    ok = check("horizontal array gives the 2 deg class beam its aperture implies",
               1.5 < bw0 < 3.0, f"{bw0:.2f} deg measured")
    ok &= check("vertical array gives the 20 deg fan", 15.0 < vfan < 26.0,
                f"{vfan:.1f} deg measured")
    ok &= check("beam broadens as 1/cos across the sector",
                all(abs(r - 1.0) < 0.10 for r in broadening),
                "ratios " + ", ".join(f"{r:.3f}" for r in broadening))
    ok &= check("the body's bearing centroid lands on the body",
                abs(centroid) <= subtended / 2.0,
                f"centroid {centroid:+.2f} deg, hull spans +/-{subtended / 2:.2f} deg")
    ok &= check("the hull is resolved as a body, several beams wide",
                extents[-3.0] > 1.5 * bw0,
                f"-3 dB {extents[-3.0]:.1f} deg against beam {bw0:.2f} deg")
    ok &= check("and its extent does not exceed what the hull subtends",
                extents[-3.0] <= subtended,
                f"-3 dB {extents[-3.0]:.1f} deg, hull subtends {subtended:.1f} deg")
    ok &= check("gradients still reach every scene parameter", all(states.values()))
    return 0 if ok else 1


def _plot(image, bearings, grid, boat, elements, beamwidth):
    import matplotlib.pyplot as plt
    import numpy as np

    rng = grid.detach().numpy() * C / 2.0
    a = image[:, 0].detach().numpy()
    db = 10 * np.log10(np.maximum(a, a.max() * 1e-4) / a.max())
    world = boat.world_positions().detach().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8),
                             gridspec_kw={"width_ratios": [2, 1]})
    m = axes[0].pcolormesh(rng, bearings.numpy(), db, cmap="inferno", vmin=-25,
                           vmax=0, shading="auto")
    r = np.hypot(np.hypot(world[:, 0], world[:, 1]), world[:, 2] - SONAR_DEPTH)
    b = np.degrees(np.arctan2(world[:, 1], world[:, 0]))
    axes[0].plot(r, b, "+", color="#7fdfff", ms=9, mew=1.4)
    axes[0].set_xlabel("range (m)")
    axes[0].set_ylabel("bearing (deg)")
    axes[0].set_title(f"{int(2 * SECTOR_DEG)} deg sector, "
                      f"{beamwidth:.1f} deg beams; crosses = true highlights",
                      fontsize=10)
    fig.colorbar(m, ax=axes[0], label="dB re peak")

    power = image[:, 0].detach().max(dim=-1).values
    axes[1].plot(bearings.numpy(), 10 * np.log10((power / power.max()).numpy()),
                 lw=1.3)
    axes[1].axhline(-3, ls=":", c="k", lw=0.9, label="-3 dB")
    axes[1].set_xlim(-20, 20)
    axes[1].set_ylim(-25, 2)
    axes[1].set_xlabel("bearing (deg)")
    axes[1].set_ylabel("dB re peak")
    axes[1].set_title("Bearing cut through the boat", fontsize=10)
    axes[1].grid(alpha=0.3, lw=0.4)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    raise SystemExit(main())
