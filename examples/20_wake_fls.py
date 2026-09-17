"""Your boat's wake, seen from underneath: an AUV's FLS looking up at a wake.

The scene from ``examples/15``, with the boat now *moving*: a 12 m hull at
6 m/s (11.7 knots) on a turning track that crosses the sonar's swath, and the
wake it has been laying down for the last minute.  The AUV is 18 m down in 30 m
of water looking forward and up, so the sea surface -- and everything on it --
is in the picture from about 50 m out.

**A wake reaches a sonar by two quite different routes, and they are not the
same size.**

*The waves.*  ``hydropt.wake`` gives the Kelvin pattern as a height field, which
is added to the wind sea and reflects like any other surface.  The wake tilts
the water, the tilt changes the grazing angle, and the grazing angle changes the
backscatter.  Measured below, that is worth about a decibel.  It is real, and at
a single look it is invisible: one ping of reverberation is Rayleigh speckle
with a 5.6 dB spread, so a 1 dB modulation needs tens of looks to come out of
the noise.  This example prints the number.

*The bubbles.*  What a sonar or a radar actually sees is the other wake: the
band of entrained air and turbulence the hull and propeller leave along the
track, which scatters tens of dB above the sea around it and lasts for minutes.
That is a change in the surface's scattering strength, not in its shape, so it
enters through ``surface_gain`` rather than through the height field -- and it
is the reason a wake is usually the most conspicuous thing in the image.

Both are built from the same track, both are differentiable in it, and the last
section spends that: the image carries gradients to the vessel's **speed and
rate of turn**, which is what makes a wake a measurement of the vessel rather
than a decoration on it.

**Layover.** A surface feature images at its slant range, so the wake is drawn
farther out than it lies -- 2.8 m at 57 m range from 18 m of depth.  The masks
below allow for it; an operator reading ranges off the screen has to as well.

Acceptance criteria:
  * the wake band lands along the vessel's track, allowing for layover;
  * its contrast over the surface around it is the scattering gain we put in;
  * the wave channel on its own is small, and the example says how small and
    how many looks it would take to see;
  * the boat's own echo is still on the boat;
  * the image carries gradients to the vessel's speed and rate of turn.
"""

from __future__ import annotations

import importlib.util
import math
import time
from pathlib import Path

import torch

from _common import banner, check, save, setup, timed
from hydropt import (
    azimuth_steering, beamform, make_time_grid, shading_window, target_arrivals,
)
from hydropt.beamform import ArrivalSet
from hydropt.environment import pierson_moskowitz_surface
from hydropt.mesh import boat_hull_mesh, mesh_target
from hydropt.reverb import LambertScattering, reverberation_arrivals
from hydropt.tracer import trace
from hydropt.wake import bubble_wake_gain, froude_number, kelvin_wake_surface

C = 1500.0
WATER_DEPTH = 30.0
AUV_DEPTH = 18.0
HULL_LENGTH, HULL_BEAM, HULL_DRAUGHT = 12.0, 3.2, 1.0

# A light breeze, not the 5 m/s examples/15 runs in.  Wakes are read in calm
# water for a reason: the wind sea is the competition, and at 3 m/s it is
# 0.05 m RMS against the wake's 0.07 m rather than three times it.
WIND = 3.0
SPEED = 6.0                 # 11.7 knots
TURN_RATE = math.radians(-0.86)   # rad/s -- a slow turn to starboard
TRACK_SECONDS = 60.0
N_TRACK = 121
VESSEL_AT = (35.0, -45.0)   # where it is now: 57 m, bearing -52 deg
VESSEL_HEADING_DEG = 245.0  # and where it is pointing

WAKE_AMPLITUDE = 0.6        # peak wave height, metres -- hull-dependent
BUBBLE_GAIN_DB = 15.0       # the turbulent wake over the ambient sea
BUBBLE_WIDTH = 4.0          # half-width where it is laid down
BUBBLE_DECAY = 300.0        # bubbles rise out over minutes, not seconds

SURFACE_SPACING = 1.2       # resolves the wake's shortest waves, not just the sea
SURFACE_SPAN = 180.0
PATCHES = 40000
SEED = 5


def _ex15():
    path = Path(__file__).resolve().parent / "15_auv_scene_cartesian.py"
    spec = importlib.util.spec_from_file_location("_ex15", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def vessel_track(speed: torch.Tensor, turn_rate: torch.Tensor,
                 times: torch.Tensor) -> torch.Tensor:
    """Where the vessel has been, from its speed and rate of turn.

    Anchored at the **present** position and heading rather than at the start,
    because that is the question an image asks: the boat is there, pointing
    that way -- where did it come from, and how fast?  Changing either
    parameter then swings the past track about the vessel, which is exactly the
    wake the image sees.  Differentiable in both.
    """
    psi = turn_rate * (times - times[-1])
    v = speed * torch.stack([psi.cos(), psi.sin()], dim=-1)
    dt = times[1:] - times[:-1]
    step = 0.5 * (v[1:] + v[:-1]) * dt.unsqueeze(-1)       # trapezoid
    pos = torch.cat([torch.zeros(1, 2, dtype=v.dtype), step.cumsum(0)], dim=0)
    pos = pos - pos[-1]
    # Rotate so the final heading is the one we say it is.
    rot = math.radians(VESSEL_HEADING_DEG)
    c, s = math.cos(rot), math.sin(rot)
    spin = torch.tensor([[c, -s], [s, c]], dtype=pos.dtype)
    return pos @ spin.T + torch.tensor(VESSEL_AT, dtype=pos.dtype)


def sea_surface(track, times, *, wake: bool):
    """The wind sea, with or without the wake's waves on top of it.

    The same seeded wind sea either way, so any difference between the two
    images is the wake and not a different sea.
    """
    n = int(round(SURFACE_SPAN / SURFACE_SPACING)) + 1
    origin = (-20.0, -n * SURFACE_SPACING / 2)
    sea = pierson_moskowitz_surface(
        (n, n), (SURFACE_SPACING, SURFACE_SPACING), WIND, origin=origin,
        generator=torch.Generator().manual_seed(1))
    if not wake:
        return sea, None
    extent = ((origin[0], origin[0] + (n - 1) * SURFACE_SPACING),
              (origin[1], origin[1] + (n - 1) * SURFACE_SPACING))
    field = kelvin_wake_surface(
        track, times, amplitude=WAKE_AMPLITUDE, extent=extent,
        spacing=SURFACE_SPACING, base=sea.heights, water_depth=WATER_DEPTH,
        n_directions=96, max_angle_deg=62.0, decay_time=240.0)
    return field, (field.heights - sea.heights)


def image_position(xy: torch.Tensor, height: float = AUV_DEPTH) -> torch.Tensor:
    """Where a point on the surface is *drawn*, which is not where it is.

    The image puts every echo at its slant range along its bearing, and a patch
    of sea surface is ``height`` metres above the sonar, so it lands farther out
    than its horizontal distance by ``r - sqrt(r^2 - height^2)``.
    """
    rho = xy.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    return xy * (torch.hypot(rho, torch.full_like(rho, height)) / rho)


def main() -> int:
    setup()
    banner("20 -- a boat's wake in a forward-looking sonar")
    ex15 = _ex15()
    mc = ex15._mills()
    fls = mc._fls

    rx = mc.horizontal_array()
    rx = torch.stack([rx[:, 0], rx[:, 1],
                      torch.full_like(rx[:, 2], AUV_DEPTH)], dim=-1)
    scene, bottom, _, sediment = fls.build_scene(rx, learnable=True)
    scene.source = torch.tensor([0.0, 0.0, AUV_DEPTH])

    speed = torch.tensor(SPEED, requires_grad=True)
    turn_rate = torch.tensor(TURN_RATE, requires_grad=True)
    times = torch.linspace(0.0, TRACK_SECONDS, N_TRACK)
    track = vessel_track(speed, turn_rate, times)
    run = float((track[1:] - track[:-1]).detach().norm(dim=-1).sum())
    print(f"  vessel: {HULL_LENGTH:.0f} m hull at {SPEED:.1f} m/s "
          f"({SPEED * 1.944:.1f} kn), turning "
          f"{abs(math.degrees(TURN_RATE)):.2f} deg/s to starboard")
    print(f"          {run:.0f} m of track over the last {TRACK_SECONDS:.0f} s, "
          f"now at ({VESSEL_AT[0]:.0f}, {VESSEL_AT[1]:.0f}) m = "
          f"{math.hypot(*VESSEL_AT):.0f} m, bearing "
          f"{math.degrees(math.atan2(VESSEL_AT[1], VESSEL_AT[0])):+.0f} deg")
    print(f"  Froude {froude_number(SPEED, WATER_DEPTH):.2f} in "
          f"{WATER_DEPTH:.0f} m -- deep water holds below 0.7")
    lam = 2 * math.pi * SPEED ** 2 / 9.80665
    print(f"  transverse wavelength 2 pi V^2 / g = {lam:.1f} m, sampled at "
          f"{SURFACE_SPACING:.1f} m ({lam / SURFACE_SPACING:.0f} nodes)")

    wake_surface, eta = sea_surface(track, times, wake=True)
    plain_surface, _ = sea_surface(track, times, wake=False)
    print(f"  wind sea {float(plain_surface.heights.std()):.3f} m RMS, "
          f"wake {float(eta.detach().std()):.3f} m RMS, "
          f"peak {float(eta.detach().abs().max()):.2f} m")

    def bubbles(xy):
        return bubble_wake_gain(xy, track, times, gain_db=BUBBLE_GAIN_DB,
                                width=BUBBLE_WIDTH, decay_time=BUBBLE_DECAY)

    banner("ping")
    boat = _boat(track)
    steer, bearings = azimuth_steering(181, ex15.SECTOR_DEG)
    grid = make_time_grid(2.0 * 8.0 / C, 2.0 * 95.0 / C, 420)
    seabed = LambertScattering(-27.0, learnable=False)
    dirs, tx_weights = ex15.transmit_fan(56, 330, seed=SEED)
    solid = (math.radians(2 * ex15.SECTOR_DEG) * math.radians(46.0)
             / dirs.shape[0])

    def render(surface, gain, *, with_boat=True, seed=SEED + 1):
        scene.surface = surface
        result = trace(scene, dirs)
        rev = reverberation_arrivals(
            result, dirs, scene.freqs_khz, scattering=seabed,
            solid_angle_per_ray=solid, ray_weights=tx_weights, boundary="both",
            surface=surface, bottom=scene.bottom, max_arrivals=PATCHES,
            surface_gain=gain,
            generator=torch.Generator().manual_seed(seed))
        if with_boat:
            echo = target_arrivals(scene, boat, dirs, n_rx_rays=420,
                                   rx_half_angle_deg=45.0,
                                   tx_weights=tx_weights,
                                   max_arrivals_per_leg=24,
                                   generator=torch.Generator().manual_seed(SEED))
            rev = ArrivalSet(*(None if rev[i] is None or echo[i] is None
                               else torch.cat([rev[i], echo[i]], dim=0)
                               for i in range(len(rev))))
        return beamform(rev, rx, scene.freqs_khz, grid, steer,
                        sigma_t=ex15.PULSE_S,
                        shading=shading_window(ex15.N_RX, "hamming"),
                        steer_chunk=8)

    t0 = time.perf_counter()
    with timed("  waves + bubbles + boat"):
        full = render(wake_surface, bubbles)
    forward = time.perf_counter() - t0
    with torch.no_grad():
        with timed("  waves only, no bubbles"):
            tilt = render(wake_surface, None)
        with timed("  no wake at all"):
            bare = render(plain_surface, None)
        # A control look: the same sea, the same rays, only a fresh draw of the
        # scattering phases.  Reverberation is speckle, so two looks at one
        # scene disagree cell by cell -- and without this there is no way to
        # tell how much of the wave channel's apparent effect is the wake and
        # how much is the dice.
        with timed("  no wake, second look"):
            again = render(plain_surface, None, seed=SEED + 101)

    cart, gx, gy = ex15.to_cartesian(full, bearings, grid)
    with torch.no_grad():
        tilt_c, _, _ = ex15.to_cartesian(tilt, bearings, grid)
        bare_c, _, _ = ex15.to_cartesian(bare, bearings, grid)
        again_c, _, _ = ex15.to_cartesian(again, bearings, grid)

    banner("where the wake is in the picture")
    det = cart.detach()
    GX = torch.as_tensor(gx).reshape(1, -1).expand_as(det)
    GY = torch.as_tensor(gy).reshape(-1, 1).expand_as(det)
    cells = torch.stack([GX.reshape(-1), GY.reshape(-1)], dim=-1)
    # Undo the layover to ask the gain where each *drawn* cell really is.
    rho = cells.norm(dim=-1)
    ground = cells * (torch.sqrt((rho ** 2 - AUV_DEPTH ** 2).clamp_min(0.0))
                      / rho.clamp_min(1e-9)).unsqueeze(-1)
    with torch.no_grad():
        gain_db = (10 * torch.log10(bubbles(ground))).reshape(det.shape)
    lit = (bare_c > 0) & (det > 0) & (tilt_c > 0) & (again_c > 0)
    # The hull sits at the head of its own wake, so its echo has to be cut out
    # of the band or the band's contrast is partly the boat's target strength.
    bx, by = float(track[-1, 0]), float(track[-1, 1])
    clear_of_boat = torch.hypot(GX - bx, GY - by) > 15.0
    on_wake = lit & clear_of_boat & (gain_db > BUBBLE_GAIN_DB - 3.0)
    off_wake = lit & clear_of_boat & (gain_db < 0.5) & (rho.reshape(det.shape) > 55.0)
    band = 10 * math.log10(float(det[on_wake].mean() / det[off_wake].mean()))
    print(f"  {int(on_wake.sum())} cells inside the wake band, "
          f"{int(off_wake.sum())} on clean surface beyond 55 m")
    print(f"  (both clear of the boat, which sits at the head of its own wake)")
    print(f"  the band stands {band:+.1f} dB over the sea around it "
          f"(put in: {BUBBLE_GAIN_DB:+.0f} dB on the surface patches; the")
    print(f"  seabed under the band is unchanged and dilutes it)")

    # Where the band's centre of brightness sits, against the track it came
    # from -- drawn at slant range, so the track has to be laid over too.
    with torch.no_grad():
        w = det[on_wake]
        cx = float((GX[on_wake] * w).sum() / w.sum())
        cy = float((GY[on_wake] * w).sum() / w.sum())
    drawn = image_position(track.detach())
    d = (drawn - torch.tensor([cx, cy])).norm(dim=-1)
    print(f"  its brightest centre is at ({cx:+.1f}, {cy:+.1f}) m, "
          f"{float(d.min()):.1f} m from the laid-over track")
    print(f"  (layover pushes a surface feature out by "
          f"{float((drawn - track.detach()).norm(dim=-1).max()):.1f} m here)")

    banner("the waves on their own, without the bubbles")
    with torch.no_grad():
        sea_only = lit & (rho.reshape(det.shape) > 55.0)
        a, b, c = tilt_c[sea_only], bare_c[sea_only], again_c[sea_only]
        shift = 10 * math.log10(float(a.mean() / b.mean()))
        control_shift = 10 * math.log10(float(c.mean() / b.mean()))
        per_cell = float((10 * torch.log10(a / b)).abs().mean())
        control_cell = float((10 * torch.log10(c / b)).abs().mean())
        speckle = float((10 * torch.log10(b / b.mean())).std())
    looks = (speckle / max(abs(shift), 1e-6)) ** 2
    print(f"  over {int(sea_only.sum())} surface cells beyond 55 m:")
    print(f"    mean level, waves vs no waves:  {shift:+.2f} dB")
    print(f"    mean level, one re-deal of the speckle: {control_shift:+.2f} dB")
    print(f"    typical cell, waves vs no waves:  {per_cell:.2f} dB")
    print(f"    typical cell, one re-deal:        {control_cell:.2f} dB")
    print(f"  The per-cell figures agree, so what the wave channel does to any")
    print(f"  ONE cell is the dice, not the wake: tilting the water redistributes")
    print(f"  backscatter, it does not add any.  What is left is the mean, and a")
    print(f"  single look has a {speckle:.1f} dB spread, so resolving "
          f"{abs(shift):.2f} dB of it")
    print(f"  takes of order {looks:.0f} looks.")
    print(f"  -- which is why the wake you see in a sonar image is the bubbles,")
    print(f"     not the waves.  Both are in this picture; only one is obvious.")

    banner("the boat itself")
    tx, ty = float(track[-1, 0]), float(track[-1, 1])
    with torch.no_grad():
        near_boat = torch.hypot(GX - tx, GY - ty) < 12.0
        peak = det.clone()
        peak[~near_boat] = 0.0
        flat = int(peak.reshape(-1).argmax())
        px, py = float(gx[flat % det.shape[1]]), float(gy[flat // det.shape[1]])
    beamwidth = 2.0 * math.degrees(math.asin(1.0 / (ex15.N_RX / 2.0)))
    tol = math.hypot(tx, ty) * math.radians(beamwidth)
    err = max(0.0, math.hypot(px - tx, py - ty) - HULL_LENGTH / 2.0)
    print(f"  the hull's echo peaks at ({px:+.1f}, {py:+.1f}) m against a boat "
          f"centred on ({tx:+.1f}, {ty:+.1f})")
    print(f"  {err:.1f} m outside the hull, beamwidth is {tol:.1f} m there")

    banner("the wake is a measurement of the vessel")
    print("  A loss on the image, differentiated back through the wake to the")
    print("  speed and the rate of turn that made it -- no boat echo needed.")
    t0 = time.perf_counter()
    cart[on_wake].sum().backward()
    backward = time.perf_counter() - t0
    grads = {"vessel speed": speed, "rate of turn": turn_rate,
             "seabed": bottom.heights, "sediment c2": sediment.c2}
    live = {}
    for name, p in grads.items():
        ok_g = (p.grad is not None and bool(torch.isfinite(p.grad).all())
                and float(p.grad.abs().sum()) > 0)
        live[name] = ok_g
        size = "" if p.grad is None else f"  |grad| = {float(p.grad.abs().sum()):.3e}"
        print(f"  d(wake band)/d({name:<14s}): {'OK' if ok_g else 'ZERO'}{size}")
    print(f"\n  forward {forward:.1f} s + backward {backward:.1f} s")

    save(_plot(det, tilt_c, bare_c, gx, gy, eta.detach(), track.detach(),
               gain_db, ex15), "20_wake_fls.png")

    banner("acceptance")
    ok = check("the wake band lands on the vessel's track, allowing for layover",
               float(d.min()) < 8.0,
               f"{float(d.min()):.1f} m from the laid-over track")
    ok &= check("its contrast is the scattering gain we put in",
                abs(band - BUBBLE_GAIN_DB) < 5.0,
                f"{band:+.1f} dB measured against {BUBBLE_GAIN_DB:+.0f} dB in")
    ok &= check("what the waves do to one cell is speckle, not signal",
                abs(per_cell - control_cell) < 1.5,
                f"{per_cell:.2f} dB against {control_cell:.2f} dB for a "
                f"re-deal of the same scene")
    ok &= check("and what is left is a mean shift no single look could see",
                abs(shift) < 2.0 and looks > 50.0,
                f"{shift:+.2f} dB under {speckle:.1f} dB of speckle, "
                f"{looks:.0f} looks")
    ok &= check("the boat's own echo is still on the boat",
                err < tol, f"{err:.1f} m outside the hull against {tol:.1f} m")
    ok &= check("the image carries gradients to the vessel's speed and turn",
                live["vessel speed"] and live["rate of turn"],
                f"{sum(live.values())}/{len(live)} parameters live")
    return 0 if ok else 1


def _boat(track):
    """The hull, put where the track says the vessel is and pointing that way."""
    verts, faces = boat_hull_mesh(HULL_LENGTH, HULL_BEAM, HULL_DRAUGHT,
                                  n_long=110, n_around=34)
    end = track.detach()
    heading = end[-1] - end[-2]
    return mesh_target(
        verts, faces,
        position=(float(end[-1, 0]), float(end[-1, 1]), HULL_DRAUGHT),
        yaw=math.degrees(math.atan2(float(heading[1]), float(heading[0]))),
        n_patches=6, sound_speed=C, learnable=False, learnable_shape=False,
        facet_chunk=256)


def _plot(full, tilt, bare, gx, gy, eta, track, gain_db, ex15):
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(1, 4, figsize=(21.5, 5.6))
    n = eta.shape[0]
    origin = (-20.0, -n * SURFACE_SPACING / 2)
    extent = [origin[0], origin[0] + (n - 1) * SURFACE_SPACING,
              origin[1], origin[1] + (n - 1) * SURFACE_SPACING]
    e = eta.numpy()
    lim = float(np.abs(e).max())
    axes[0].imshow(e, origin="lower", extent=extent, cmap="RdBu_r",
                   vmin=-lim, vmax=lim)
    axes[0].plot(track[:, 0], track[:, 1], "k-", lw=1.2)
    axes[0].plot(track[-1, 0], track[-1, 1], "ko", ms=5)
    axes[0].plot(0, 0, "k^", ms=9)
    axes[0].set_title(f"the wake's waves ({lim:.2f} m peak)")

    drawn = image_position(track).numpy()
    for ax, img, title in ((axes[1], full, "sonar: waves + bubbles + boat"),
                           (axes[2], tilt, "waves only, no bubbles"),
                           (axes[3], bare, "no wake at all")):
        d = 10 * np.log10(np.maximum(img.numpy(), 1e-30))
        pk = float(np.max(10 * np.log10(np.maximum(full.numpy(), 1e-30))))
        ax.imshow(d, origin="lower", cmap="inferno", vmin=pk - 42, vmax=pk,
                  extent=[float(gx[0]), float(gx[-1]),
                          float(gy[0]), float(gy[-1])])
        ax.plot(drawn[:, 0], drawn[:, 1], color="deepskyblue", lw=0.7,
                alpha=0.8)
        ax.set_title(title)
    for ax in axes:
        ax.set_aspect("equal")
        ax.set_xlim(-10, 100)
        ax.set_ylim(-80, 80)
        ax.set_xlabel("forward (m)")
    axes[0].set_ylabel("across (m)")
    fig.suptitle("A 12 m boat at 11.7 kn, turning: the wake an AUV's FLS sees "
                 "from 18 m below it", y=1.02)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    raise SystemExit(main())
