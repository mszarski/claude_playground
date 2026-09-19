"""A moored buoy three ways: what the chain draws, and why.

``examples/26`` moored its buoy on 45 m of chain to a sinker 30 m away, lying
across the line of sight, and the chain came out as a line 30 m long and
nearly as bright as the boat.  That looked wrong, and the question of why
has two answers, both worth having.

**The chain is bright because the background is quiet.**  A 32 mm stud link
returns about -26 dB at any aspect (the specular off its bar's bend), and a
range cell 0.22 m deep holds one or two links of a chain running along the
line of sight and a beam's width of them, sixty-odd, of one lying across.
Either would sink into seabed reverberation.  But 21's sonar looks 5
degrees UP from 12 m down, and at 120 m the seabed is below the lobe: the
background is sea-surface reverberation at 4.6 degrees of grazing, a floor
some 45 dB under a square metre, and one link per cell stands 25 dB over
it.  So in this picture a chain shows however it lies -- across (26's) or
along, as the first two scenarios here confirm -- and what a downward-
tilted harbour sonar over a bright seabed would show is a fainter thing.

**The mooring was the wrong shape.**  A taut 30 m scope is a laid mooring's
geometry, not a buoy's.  A harbour buoy hangs on a heavy chain that drops
steeply -- a catenary with a small parameter, ``H/w`` of a few metres --
and lays its excess on the bottom as ground chain out to the anchor.  Seen
from the sonar, the hanging part spans a dozen metres in plan under the
buoy, and the ground chain lies on the seabed, which at these ranges is
six to nine degrees down -- in the skirt of a lobe whose half-power edge
is 5.4 degrees down and whose elevation elements are unshaded, so the
skirt is only 13 dB down -- and it reads as a fainter continuation of the
tail.  The third scenario is that mooring: a short bright tail under the
buoy and a faint trace beyond it, which is what a moored buoy looks like.

Three scenarios, each against the bare picture (21's boat at rest) and
drawn close up about the buoy:

* **across**: 26's mooring, the chain across the line of sight;
* **along**: the same chain running away from the sonar.  Its half-power
  width is the beam; its visible width is about twice that, because a
  line 25 dB over so quiet a floor shows the beam's skirts at -10 and
  -20 dB too -- the same reason the caisson of ``examples/23`` looked
  wider than a beam;
* **slack**: a catenary of parameter 5 m from the buoy to a touchdown on
  the seabed, then 40 m of ground chain on the bottom to the anchor.

Acceptance criteria:
  * across and along, the bright line under the buoy is about as long as
    the mooring's plan span, and the along one is one beam wide;
  * slack, the bright tail is no longer than the hanging part's plan span,
    and the ground chain reads fainter than the tail.

Float32, as ``examples/23``.  21's switches carry through; the buoy's range
scales with the swath, the mooring's geometry (set by the water depth) does
not.
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
    box_mesh, calibrate, line_array_directivity_db, make_time_grid,
    reverberation_arrivals, shading_window, target_arrivals, trace,
)
from hydropt.beamform import ArrivalSet
from hydropt.mesh import boat_hull_mesh, mesh_target

SCENARIO = os.environ.get("HYDROPT_SCENARIO", "all")
if SCENARIO not in ("all", "across", "along", "slack"):
    raise SystemExit(f"HYDROPT_SCENARIO must be all, across, along or slack, got {SCENARIO!r}")
SCENARIOS = ("across", "along", "slack") if SCENARIO == "all" else (SCENARIO,)
CAPTION = {"across": "a taut mooring, the chain across the line of sight (26's)",
           "along": "a taut mooring, the chain along the line of sight",
           "slack": "a slack mooring: a steep catenary, ground chain on the bottom"}

BUOY_RANGE, BUOY_BEARING_DEG = 120.0, 8.0     # the range scales with the swath
BUOY_RADIUS = 0.75
CHAIN_LENGTH, CHAIN_SCOPE = 45.0, 30.0        # the taut mooring: metres of chain, plan span
CHAIN_LINK_DB, LINKS_PER_M = -26.0, 5.0       # 32 mm stud link: pi R1 R2 off the bar's bend
SLACK_A = 5.0                                 # the slack catenary's parameter H/w, metres
GROUND_CHAIN = 40.0                           # metres on the bottom to the anchor
SINKER = 0.8
CHAIN_STEP = 0.25                             # metres of chain per point scatterer
BRIGHT_DB = 6.0                               # a cell is the chain's if this far over bare: the display's own floor


def _ex21():
    path = Path(__file__).resolve().parent / "21_long_range_300m.py"
    spec = importlib.util.spec_from_file_location("_ex21", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_EX = _ex21()
S = _EX.FAR / 300.0
BUOY_RANGE = BUOY_RANGE * S


def polar(r, bearing_deg, depth=0.0):
    b = math.radians(bearing_deg)
    return (r * math.cos(b), r * math.sin(b), depth)


def taut_catenary(length: float, span: float, drop: float, step: float) -> torch.Tensor:
    """Points along a chain of ``length`` hung between (0, 0) and (span, drop), z down."""
    target = math.sqrt(max(length ** 2 - drop ** 2, 1e-6))
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
    n = max(int(length / step), 2)
    x = torch.linspace(0.0, span, n)
    z = a * torch.cosh((x - x0) / a) + c
    return torch.stack([x, torch.zeros(n), z], -1)


def slack_mooring(a: float, drop: float, ground: float, step: float):
    """A mooring chain with parameter ``a`` from the buoy to a touchdown, then on the bottom.

    The hanging part is ``z = drop - a (cosh((x_t - x)/a) - 1)``, horizontal at
    the touchdown ``x_t = a acosh(drop/a + 1)``, of length ``a sinh(x_t/a)``;
    the ground chain runs on from there.  Returns the hanging points, the
    ground points and the touchdown span.
    """
    x_t = a * math.acosh(drop / a + 1.0)
    hang_len = a * math.sinh(x_t / a)
    n = max(int(hang_len / step), 2)
    x = torch.linspace(0.0, x_t, n)
    z = drop - a * (torch.cosh((x_t - x) / a) - 1.0)
    hanging = torch.stack([x, torch.zeros(n), z], -1)
    m = max(int(ground / step), 2)
    xg = torch.linspace(x_t, x_t + ground, m)
    on_bottom = torch.stack([xg, torch.zeros(m), torch.full((m,), drop)], -1)
    return hanging, on_bottom, x_t, hang_len


def main() -> int:
    setup(double=False)
    banner("27 -- a moored buoy three ways: " + ", ".join(SCENARIOS))
    ex = _EX
    ex15 = ex._ex15()
    C = ex.C
    rx = ex.horizontal_array()
    scene, bottom, sea, sediment = ex.build_scene(rx, learnable=False)
    print(f"  {ex.FREQ_KHZ:.0f} kHz, {ex.N_RX} x {ex.N_TX}, {2 * ex.SECTOR_DEG:.0f} deg "
          f"swath to {ex.FAR:.0f} m; AUV at {ex.AUV_DEPTH:.0f} m in {ex.WATER_DEPTH:.0f} m")

    head = polar(ex.BOAT_RANGE, ex.BOAT_BEARING_DEG)
    girth = ex.HULL_BEAM + 2.0 * ex.HULL_DRAUGHT
    verts, faces = boat_hull_mesh(ex.HULL_LENGTH, ex.HULL_BEAM, ex.HULL_DRAUGHT,
                                  n_long=int(round(110 * ex.HULL_LENGTH / 12.0)),
                                  n_around=int(round(34 * girth / (3.2 + 2.0 * 4.0))))

    def make_boat():
        return mesh_target(verts, faces, position=head, yaw=ex.BOAT_HEADING_DEG, n_patches=6,
                           sound_speed=C, diffuse_db=ex.DIFFUSE_DB, learnable=False,
                           learnable_shape=False, facet_chunk=4096, checkpoint=False)

    # ---- the buoy and its moorings ---------------------------------------- #
    bx, by, bz = polar(BUOY_RANGE, BUOY_BEARING_DEG, BUOY_RADIUS * 0.6)
    drop = ex.WATER_DEPTH - 0.4 - bz                       # buoy to the chain on the bottom
    buoy = ExtendedTarget(torch.zeros(1, 3),
                          CurvedSurfaceScattering(BUOY_RADIUS, BUOY_RADIUS, learnable=False),
                          position=(bx, by, bz), learnable=True)
    point_db = lambda pts: CHAIN_LINK_DB + 10 * math.log10(LINKS_PER_M * CHAIN_STEP)
    taut = taut_catenary(CHAIN_LENGTH, CHAIN_SCOPE, drop, CHAIN_STEP)
    hanging, ground, x_t, hang_len = slack_mooring(SLACK_A, drop, GROUND_CHAIN, CHAIN_STEP)
    print(f"  a {BUOY_RADIUS:.2f} m buoy at {BUOY_RANGE:.0f} m, bearing {BUOY_BEARING_DEG:+.0f} deg; "
          f"links of {CHAIN_LINK_DB:.0f} dB, {LINKS_PER_M:.0f} a metre")
    print(f"  taut: {CHAIN_LENGTH:.0f} m of chain over a {CHAIN_SCOPE:.0f} m span; slack: parameter "
          f"{SLACK_A:.0f} m, {hang_len:.0f} m hanging over {x_t:.1f} m of plan, then {GROUND_CHAIN:.0f} m "
          f"of ground chain")

    def chain_target(points, yaw):
        return ExtendedTarget(points, IsotropicScattering(point_db(points), learnable=False),
                              position=(bx, by, bz), yaw=yaw, learnable=False)

    def sinker_at(span, yaw):
        d = math.radians(yaw)
        return mesh_target(*box_mesh((SINKER, SINKER, SINKER)),
                           position=(bx + span * math.cos(d), by + span * math.sin(d),
                                     ex.WATER_DEPTH - SINKER / 2),
                           n_patches=1, sound_speed=C, diffuse_db=-10.0, learnable=False,
                           facet_chunk=4096, checkpoint=False)

    # ---- the sonar, as 21 has it ------------------------------------------ #
    dirs, _ = ex.transmit_fan(seed=ex.SEED)
    w_tx = ex.transmit_pattern(dirs)
    tilt = ex.passes_deg()[0]
    rx_beam = lambda d: ex.receive_beam(d, tilt)
    steer, bearings = azimuth_steering(ex.N_BEAMS, ex.SECTOR_DEG)
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
    beam_m = math.radians(ex.beam_3db_deg(ex.N_RX, shading)) * BUOY_RANGE

    def echo(target):
        return target_arrivals(
            scene, target, dirs, return_leg="eigenray", n_rx_rays=2000,
            rx_half_angle_deg=45.0, tx_weights=w_tx, tx_pattern=ex.transmit_pattern,
            rx_pattern=rx_beam, max_arrivals_per_leg=24,
            generator=torch.Generator().manual_seed(ex.SEED))

    with torch.no_grad(), timed("  trace + reverberation, once"):
        result = trace(scene, dirs)
        rev = reverberation_arrivals(
            result, dirs, scene.freqs_khz, scattering=seabed,
            solid_angle_per_ray=solid, ray_weights=w_tx * rx_beam(-dirs),
            boundary="both", surface=sea, bottom=scene.bottom, max_arrivals=ex.PATCHES,
            generator=torch.Generator().manual_seed(ex.SEED + 1))

    def ping(targets):
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
        return cart, gx, gy

    banner("the bare picture: sea, seabed, the boat at rest")
    with torch.no_grad(), timed("  ping"):
        bare, gx, gy = ping([make_boat()])
    X, Y = torch.meshgrid(gx, gy, indexing="xy")
    db = lambda t: 10.0 * torch.log10(t.detach().clamp_min(1e-30))
    bare_db = db(bare)
    ext = [float(gx.min()), float(gx.max()), float(gy.min()), float(gy.max())]
    dx_, dy_ = float(gx[1] - gx[0]), float(gy[1] - gy[0])

    def along_line(diff_db, pts_world):
        """Bright extent along a plan polyline: the arc length to the farthest bright sample."""
        arc = torch.cat([torch.zeros(1), (pts_world[1:, :2] - pts_world[:-1, :2]).norm(dim=-1).cumsum(0)])
        ix = ((pts_world[:, 0] - float(gx[0])) / dx_).round().long().clamp(0, gx.numel() - 1)
        iy = ((pts_world[:, 1] - float(gy[0])) / dy_).round().long().clamp(0, gy.numel() - 1)
        vals = diff_db[iy, ix]
        bright = vals > BRIGHT_DB
        extent = float(arc[bright].max()) if bool(bright.any()) else 0.0
        return extent, float(vals.median())

    ok = True
    for name in SCENARIOS:
        banner(f"scenario: {name}")
        yaw = BUOY_BEARING_DEG + (90.0 if name == "across" else 0.0)   # across, or away
        if name == "slack":
            chains = [chain_target(hanging, yaw), chain_target(ground, yaw)]
            span = x_t + GROUND_CHAIN
            plan = hanging
        else:
            chains = [chain_target(taut, yaw)]
            span = CHAIN_SCOPE
            plan = taut
        targets = [make_boat(), buoy] + chains + [sinker_at(span, yaw)]
        with torch.no_grad(), timed("  ping"):
            cart, _, _ = ping(targets)
        diff = db(cart) - bare_db
        c, s_ = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
        world = lambda pts: torch.stack([bx + pts[:, 0] * c, by + pts[:, 0] * s_, pts[:, 2]], -1)
        extent, med = along_line(diff, world(plan))
        near = (X - bx) ** 2 + (Y - by) ** 2 < 4.0 ** 2
        peak = float(diff[near].max())
        print(f"  the buoy peaks {peak:+.1f} dB over the bare picture; along the hanging chain the "
              f"line reads {med:+.1f} dB (median), bright out to {extent:.1f} m of its "
              f"{float(plan[-1, 0]):.1f} m of plan")
        if name == "across":
            ok &= check("across, the chain draws a line about as long as its plan span",
                        0.6 * CHAIN_SCOPE < extent < 1.4 * CHAIN_SCOPE and med > BRIGHT_DB,
                        f"{extent:.1f} m bright of a {CHAIN_SCOPE:.0f} m span, {med:+.1f} dB")
        elif name == "along":
            # one beam wide: cells bright beside the line, across it, at mid-span
            mid = world(plan[plan.shape[0] // 2:plan.shape[0] // 2 + 1])[0]
            offs = torch.linspace(-3 * beam_m, 3 * beam_m, 61)
            px = mid[0] - offs * s_
            py = mid[1] + offs * c
            ix = ((px - float(gx[0])) / dx_).round().long().clamp(0, gx.numel() - 1)
            iy = ((py - float(gy[0])) / dy_).round().long().clamp(0, gy.numel() - 1)
            prof = 10.0 ** (diff[iy, ix] / 10.0)               # power, for a half-power width
            width = float((prof > 0.5 * float(prof.max())).to(torch.float32).sum()) * float(offs[1] - offs[0])
            print(f"  across the line at mid-span it is {width:.1f} m wide at half power, "
                  f"against a {beam_m:.1f} m beam")
            ok &= check("along, the chain is a line one beam wide and about its span long",
                        0.6 * CHAIN_SCOPE < extent < 1.4 * CHAIN_SCOPE and width < 1.8 * beam_m,
                        f"{extent:.1f} m bright of {CHAIN_SCOPE:.0f}, {width:.1f} m wide against "
                        f"a {beam_m:.1f} m beam")
        else:
            g_ext, g_med = along_line(diff, world(ground))
            print(f"  the ground chain reads {g_med:+.1f} dB (median) over the bare picture along "
                  f"its {GROUND_CHAIN:.0f} m on the bottom")
            ok &= check("slack, the tail is no longer than the hanging part's plan, and the ground chain fainter",
                        extent <= 1.3 * x_t + 2.0 and g_med < med - 3.0,
                        f"tail bright out to {extent:.1f} m of {x_t:.1f} m at {med:+.1f} dB; ground chain "
                        f"{g_med:+.1f} dB, in the lobe's skirt")

        # ---- the figure, close up about the buoy --------------------------- #
        ref = float(bare_db.max())
        win = (bx - 25.0 * S - 5, bx + 55.0 * S + 5, by - 40.0 * S - 5, by + 40.0 * S + 5)
        fig, axes = plt.subplots(1, 3, figsize=(19, 6.2))
        for ax, img, title in ((axes[0], bare_db, "the bare picture, close up"),
                               (axes[1], db(cart), CAPTION[name])):
            im = ax.imshow(img.numpy(), origin="lower", extent=ext, vmin=ex.THRESHOLD_DB,
                           vmax=ref, cmap="inferno", aspect="equal")
            ax.set_title(title); ax.set_xlabel("forward (m)"); ax.set_ylabel("across (m)")
        m = axes[2].imshow(diff.clamp(-15.0, 25.0).numpy(), origin="lower", extent=ext,
                           vmin=-15, vmax=25, cmap="coolwarm", aspect="equal")
        axes[2].set_title(f"{name} minus bare (dB)"); axes[2].set_xlabel("forward (m)")
        fig.colorbar(im, ax=axes[1], fraction=0.04, label="dB re the background at that range")
        fig.colorbar(m, ax=axes[2], fraction=0.04, label="dB")
        for ax in axes:
            for pts, style in ((plan, "w:"),) + (((ground, "w--"),) if name == "slack" else ()):
                w_ = world(pts)
                ax.plot(w_[:, 0].tolist(), w_[:, 1].tolist(), style, lw=1, alpha=0.7)
            ax.plot(bx, by, "c+", ms=10, mew=1.5)
            ax.plot([0.0, 1.2 * bx], [0.0, 1.2 * by], "c:", lw=0.6, alpha=0.4)   # the line of sight
            ax.set_xlim(win[0], win[1]); ax.set_ylim(win[2], win[3])
        fig.suptitle(f"{ex.FREQ_KHZ:.0f} kHz FLS, median TVG floored at +{ex.THRESHOLD_DB:.0f} dB: "
                     f"{CAPTION[name]}")
        save(fig, f"27_mooring_{name}{ex.TAG}.png")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
