"""A mine-like object on the seabed, and the shadow that measures it.

Every example so far has put the target in mid-water or on the surface, where
it sits against reverberation and is found by being brighter.  An object lying
on the bottom is a different problem, and it is the one most survey sonar is
actually flown for.  Its echo competes with the seabed right underneath it, but
the object also *blocks* the sound on its way past, and the dark band behind it
is often the clearer signal -- and the only one that carries its **height**.

  * an FLS on an AUV at 18 m in 30 m of water, looking forward and **down**;
  * a 4 m x 1.5 m cylinder lying on the seabed at 35 m;
  * Lambert reverberation off a rough sand bottom, which is what the shadow is
    a hole in -- with no reverberation there is nothing to cast a shadow on.

The height comes out of the geometry, not out of the echo.  A body of height
``h`` at horizontal distance ``D``, lit from depth ``z_s`` over a bottom at
``z_b``, shadows the bottom out to ``D (z_b - z_s) / (z_b - h - z_s)``, so
measuring the shadow's length ``L`` inverts to ``h = L (z_b - z_s) / (D + L)``.
That is how an operator reads a contact, and it is what this example checks.

Acceptance criteria:
  * the object's echo lands within a beamwidth of the **body** -- not of its
    centre, which is a different and wrong question for a target 7 degrees
    wide: the peak lands on whichever part of it is glinting;
  * there is a shadow at all: the band behind the object is far below the
    reverberation either side of it;
  * the shadow's far edge lands where the geometry says, to within the sonar's
    own range resolution;
  * the height read off the shadow brackets the object's.  Where the shadow is
    taken to *start* decides the answer, and the two honest choices bound it:
    the contact's own echo peak sits on its near side, which reads the body too
    tall, and the near end of the dark band is pushed back by the echo's own
    splat, which reads it too short.  A single number here would be a number
    with an uncalibrated bias in it;
  * a loss on the image still carries gradients to the object's pose -- through
    its echo.  **Not** through the shadow: the occlusion mask is a step, so the
    edge that carries the height has no derivative (see
    ``segment_mesh_transmission``).
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import torch

from _common import banner, check, save, setup, timed
from hydropt import (
    azimuth_steering, beamform, cylinder_mesh, line_array_factor,
    make_time_grid, mesh_target, shading_window, target_arrivals,
)
from hydropt.beamform import ArrivalSet
from hydropt.reverb import LambertScattering, reverberation_arrivals
from hydropt.tracer import trace

C = 1500.0
FREQ_KHZ = 100.0
WATER_DEPTH = 30.0
AUV_DEPTH = 18.0                       # 12 m of altitude
PULSE_S = 1.2e-4                       # 9 cm range cell

# The shadow has to be wider than a BEAMWIDTH, not merely wider than the range
# cell.  A beam pointed at a shadow narrower than itself also collects lit
# seabed from either side and fills the hole back in: a 2 m object at 60 m
# subtends 1.9 deg of shadow against this array's 3.6 deg beam, and the band is
# a 3 dB dip rather than a hole, whatever the range resolution.  Shadow width
# over range is the number that decides it, so the same object read easily at
# 30 m is unreadable at 60 m -- which is why survey lines are flown close.
OBJ_RANGE = 35.0                       # horizontal, from the sonar
OBJ_BEARING_DEG = 8.0
OBJ_LENGTH, OBJ_DIAMETER = 4.0, 1.5    # a pipeline section, or a large contact
OBJ_HEADING_DEG = 75.0                 # axis across the line of sight

# A narrow swath on purpose.  The shadow is 2 m across and 6.7 m long -- 13 m^2
# of seabed -- and it can only be read if the seabed around it is painted by
# enough scattering patches to be a background rather than scattered dots.  The
# patch density is rays per square metre of bottom, so a 120 deg swath out to
# 100 m spreads 13,000 rays over 9,800 m^2 and puts about a dozen patches in
# the shadow: the band is empty either way and measures nothing.  Narrowing the
# sector and the range window concentrates the same rays by 5x before a single
# extra one is traced.
SECTOR_DEG = 20.0
N_RX, N_TX = 64, 6
FAN_LO_DEG, FAN_HI_DEG = 10.0, 30.0    # depression angles, looking DOWN
RANGE_NEAR, RANGE_FAR = 20.0, 60.0


def _mills():
    path = Path(__file__).resolve().parent / "13_mills_cross_fls.py"
    spec = importlib.util.spec_from_file_location("_mc13", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def shadow_geometry(height: float, distance: float, altitude: float) -> float:
    """Where a body's shadow ends on a flat bottom -- the idea, in closed form.

    ``altitude`` is the sonar's height above the **local** seabed, which is what
    an altimeter reads and is not the nominal water depth minus the vehicle's
    depth: 0.75 m of bathymetric relief here is 7 percent of the altitude and
    goes straight into the answer.
    """
    return distance * altitude / (altitude - height)


def height_from_shadow(edge: float, contact: float, altitude: float) -> float:
    """Invert it: what an operator does with a measured shadow.

    ``edge`` and ``contact`` are ground ranges -- the far end of the dark band
    and the contact itself, the latter read off its own echo.
    """
    return altitude * (edge - contact) / edge


def true_shadow_edge(world: torch.Tensor, bottom, source_z: float,
                     bearing_deg: float, out_to: float = 40.0) -> float:
    """The shadow's far edge against the **real** bathymetry, for checking.

    The closed form assumes a flat bottom and a body of one height at one
    range.  Neither holds: this seabed has half a metre of relief over the
    shadow, and the body is 4 m long across a 67 degree aspect, so which part
    of it casts the far edge is a question about its silhouette.  Marching the
    grazing ray through every vertex until it meets the true bottom answers
    both, and is what the measured edge is held to.
    """
    b = math.radians(bearing_deg)
    x_v = torch.hypot(world[:, 0], world[:, 1])
    z_v = world[:, 2]
    xs = torch.linspace(float(x_v.min()), float(x_v.max()) + out_to, 4001,
                        dtype=world.dtype)
    bed = bottom.height(torch.stack([xs * math.cos(b), xs * math.sin(b)], dim=-1))
    # Each vertex defines a ray from the source; find where each meets the bed.
    ray = source_z + (z_v.unsqueeze(1) - source_z) * (xs / x_v.unsqueeze(1))
    past = (xs.unsqueeze(0) > x_v.unsqueeze(1)) & (ray >= bed.unsqueeze(0))
    first = torch.where(past.any(dim=1), past.float().argmax(dim=1),
                        torch.zeros(1, dtype=torch.long))
    return float(xs[first].max())


def transmit_fan(n_elev: int, n_azim: int, *, seed: int = 0):
    """A down-looking fan, jittered within each cell.

    Down, not up: the bottom sits 6.8 deg below the horizontal at 100 m and
    25.6 deg at 25 m, so a fan that reaches the seabed over a useful swath has
    to span that, and a fan tilted at the surface never lights the bottom at
    all -- no reverberation, and therefore no shadow to see.
    """
    g = torch.Generator().manual_seed(seed)
    e0, e1 = math.radians(FAN_LO_DEG), math.radians(FAN_HI_DEG)
    a0, a1 = -math.radians(SECTOR_DEG), math.radians(SECTOR_DEG)
    el = torch.linspace(e0, e1, n_elev)
    az = torch.linspace(a0, a1, n_azim)
    E, A = torch.meshgrid(el, az, indexing="ij")
    E, A = E.reshape(-1), A.reshape(-1)
    E = E + (torch.rand(E.shape, generator=g, dtype=E.dtype) - 0.5) * (el[1] - el[0])
    A = A + (torch.rand(A.shape, generator=g, dtype=A.dtype) - 0.5) * (az[1] - az[0])
    dirs = torch.stack([E.cos() * A.cos(), E.cos() * A.sin(), E.sin()], dim=-1)
    tilt = 0.5 * (e0 + e1)
    weights = line_array_factor(torch.sin(E), N_TX, sin_steer=math.sin(tilt))
    return dirs, weights


def place(vertices: torch.Tensor, yaw_deg: float, position) -> torch.Tensor:
    """Body-frame vertices into the world, for the occlusion test."""
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    rot = torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
                       dtype=vertices.dtype)
    return vertices @ rot.T + torch.tensor(position, dtype=vertices.dtype)


def measure_shadow(power, bearings, ranges, *, bearing_deg, echo_range,
                   beams=5, drop_db=8.0, search_m=20.0):
    """The shadow's far edge, read off the image the way an operator would.

    Averaged over a few beams across the contact -- one beam of a single ping
    is speckle, and speckle crosses any fixed threshold on its own.  The
    reference level is the median of the reverberation *just before* the
    contact, because reverberation falls off steeply with range and a level
    taken from the whole image would be a threshold at the wrong range.
    """
    i = int((bearings - bearing_deg).abs().argmin())
    lo, hi = max(0, i - beams // 2), min(len(bearings), i + beams // 2 + 1)
    profile = power[lo:hi].mean(dim=0)

    before = (ranges > echo_range - 12.0) & (ranges < echo_range - 2.0)
    reference = float(profile[before].median())
    dark = profile < reference * 10.0 ** (-drop_db / 10.0)

    # Grow the band out from the darkest cell behind the contact, rather than
    # insisting it starts in the very next one: the contact's own echo is
    # splatted over a pulse length and a beamwidth, so the first cells behind
    # it are filled by the target, not by the seabed, and a band that has to
    # start there measures nothing at all.
    window = (ranges > echo_range) & (ranges < echo_range + search_m)
    deepest = int(torch.where(window, profile,
                              torch.full_like(profile, float("inf"))).argmin())
    lo = hi = deepest
    while hi + 1 < len(ranges) and bool(dark[hi + 1]):
        hi += 1
    while lo - 1 >= 0 and bool(dark[lo - 1]):
        lo -= 1
    return profile, reference, float(ranges[hi]), float(ranges[lo])


def main() -> int:
    setup()
    banner("17 -- a mine-like object on the seabed, measured by its shadow")
    mc = _mills()
    fls = mc._fls

    rx = mc.horizontal_array()
    rx = torch.stack([rx[:, 0], rx[:, 1],
                      torch.full_like(rx[:, 2], AUV_DEPTH)], dim=-1)
    scene, bottom, surface, sediment = fls.build_scene(rx)
    scene.source = torch.tensor([0.0, 0.0, AUV_DEPTH])

    b = math.radians(OBJ_BEARING_DEG)
    height = OBJ_DIAMETER
    # On the REAL seabed, not on the nominal 30 m.  The bathymetry here is
    # 30.7 m under the contact, so placing the body at 30 m would hover it 0.7 m
    # clear -- and a floating body casts the shadow of a taller one, which is
    # exactly the error this example is built to measure.
    xy = torch.tensor([[OBJ_RANGE * math.cos(b), OBJ_RANGE * math.sin(b)]])
    bed = float(scene.bottom.height(xy))
    altitude = bed - AUV_DEPTH
    centre = (float(xy[0, 0]), float(xy[0, 1]), bed - OBJ_DIAMETER / 2.0)
    verts, faces = cylinder_mesh(OBJ_LENGTH, OBJ_DIAMETER / 2.0,
                                 n_axial=14, n_around=256)
    obj = mesh_target(verts, faces, position=centre, yaw=OBJ_HEADING_DEG,
                      n_patches=2, sound_speed=C, learnable=True,
                      facet_chunk=256)
    world = place(verts, OBJ_HEADING_DEG, centre)

    slant = math.hypot(OBJ_RANGE, altitude)
    depression = math.degrees(math.atan2(altitude, OBJ_RANGE))
    flat_edge = shadow_geometry(height, OBJ_RANGE, altitude)
    far_edge = true_shadow_edge(world, scene.bottom, AUV_DEPTH, OBJ_BEARING_DEG)
    shadow_len = far_edge - OBJ_RANGE
    arc = 2 * math.pi * (OBJ_DIAMETER / 2) / 256 / (C / (FREQ_KHZ * 1e3))
    print(f"  {FREQ_KHZ:.0f} kHz, {N_RX} receive x {N_TX} transmit, "
          f"{PULSE_S * C / 2:.2f} m range cell")
    print(f"  AUV at {AUV_DEPTH:.0f} m over a seabed at {bed:.2f} m -- "
          f"{altitude:.2f} m of altitude, fan "
          f"{FAN_LO_DEG:.0f}-{FAN_HI_DEG:.0f} deg down")
    print(f"  object: {OBJ_LENGTH:.1f} x {OBJ_DIAMETER:.1f} m cylinder on the "
          f"bottom at {OBJ_RANGE:.0f} m, bearing {OBJ_BEARING_DEG:+.0f} deg")
    print(f"          {faces.shape[0]} facets ({arc:.2f} wavelengths of arc each)")
    print(f"          seen at {depression:.1f} deg of depression, "
          f"{slant:.1f} m of slant range")
    beamwidth = 2.0 * math.degrees(math.asin(1.0 / (N_RX / 2.0)))
    shadow_deg = math.degrees(OBJ_LENGTH / OBJ_RANGE)
    print(f"  so its shadow should run {shadow_len:.2f} m, "
          f"from {OBJ_RANGE:.1f} to {far_edge:.2f} m on the bottom")
    print(f"  (the flat-bottom closed form says {flat_edge:.2f} m; the "
          f"{abs(far_edge - flat_edge):.2f} m difference is bathymetric relief)")
    print(f"  and it is {shadow_deg:.2f} deg wide against a {beamwidth:.2f} deg "
          f"beam -- a shadow narrower than a beam is filled in from either side")

    banner("ping")
    seabed = LambertScattering(-25.0, learnable=True)
    steer, bearings = azimuth_steering(81, SECTOR_DEG)
    grid = make_time_grid(2.0 * RANGE_NEAR / C, 2.0 * RANGE_FAR / C, 620)
    ranges = grid * C / 2.0

    def ping(occluders, *, n_elev=180, n_azim=420, seed=5):
        dirs, weights = transmit_fan(n_elev, n_azim, seed=seed)
        echo = target_arrivals(scene, obj, dirs, n_rx_rays=360,
                               rx_half_angle_deg=45.0, rx_jitter=1.0,
                               tx_weights=weights, max_arrivals_per_leg=24,
                               generator=torch.Generator().manual_seed(seed))
        solid = (math.radians(2 * SECTOR_DEG)
                 * math.radians(FAN_HI_DEG - FAN_LO_DEG) / dirs.shape[0])
        rev = reverberation_arrivals(
            trace(scene, dirs), dirs, scene.freqs_khz, scattering=seabed,
            solid_angle_per_ray=solid, ray_weights=weights, boundary="bottom",
            surface=scene.surface, bottom=scene.bottom,
            occluders=occluders, generator=torch.Generator().manual_seed(seed + 1))
        both = ArrivalSet(*(None if echo[i] is None or rev[i] is None
                            else torch.cat([echo[i], rev[i]], dim=0)
                            for i in range(len(echo))))
        image = beamform(both, rx, scene.freqs_khz, grid, steer, sigma_t=PULSE_S,
                         shading=shading_window(N_RX, "hamming"), steer_chunk=8,
                         arrival_chunk=2048)
        return image, echo, rev

    with timed("  with the object shadowing the seabed"):
        image, echo, rev = ping([(world, faces)])
    with torch.no_grad():
        with timed("  the same ping, shadowing switched off"):
            flat, _, rev_lit = ping(None)
        # The object's bearing has to be read off its own echo: with the seabed
        # lit, the brightest cell near its range may well be a bright patch of
        # bottom, and then the check is measuring reverberation.
        echo_only = beamform(echo, rx, scene.freqs_khz, grid, steer,
                             sigma_t=PULSE_S,
                             shading=shading_window(N_RX, "hamming"),
                             steer_chunk=8, arrival_chunk=2048)
    print(f"  {echo.n_arrivals} echo arrivals, {rev.n_arrivals} lit patches "
          f"({rev_lit.n_arrivals - rev.n_arrivals} hidden by the object)")

    banner("what the image says")
    power = image.detach()[:, 0]
    echo_slant = slant
    # The contact's own range, from its own echo -- an operator has this.
    echo_profile = echo_only[:, 0].max(dim=0).values
    contact_slant = float(ranges[int(echo_profile.argmax())])
    ground = lambda s: math.sqrt(max(s * s - altitude ** 2, 0.0))

    profile, reference, edge, start = measure_shadow(
        power, bearings, ranges, bearing_deg=OBJ_BEARING_DEG,
        echo_range=contact_slant)
    edge_ground = ground(edge)                    # the image measures SLANT range
    contact_ground = ground(contact_slant)
    measured_len = edge_ground - contact_ground
    # Two readings, from the two defensible starting points.  They bracket the
    # answer: the echo peaks on the contact's NEAR side, several metres in front
    # of the top edge that actually casts the shadow, so it over-reads; the dark
    # band's near end is where the contact's own echo stops filling the cells,
    # which is behind that top edge, so it under-reads.
    from_echo = height_from_shadow(edge_ground, contact_ground, altitude)
    from_band = height_from_shadow(edge_ground, ground(start), altitude)
    recovered = 0.5 * (from_echo + from_band)

    in_shadow = float(profile[(ranges >= start) & (ranges <= edge)].mean())
    beyond = float(profile[(ranges > edge + 2.0)
                           & (ranges < edge + 12.0)].median())
    floor = float(profile[profile > 0].min()) if bool((profile > 0).any()) else 1e-30
    contrast = 10.0 * math.log10(max(beyond, floor) / max(in_shadow, floor))

    peak_i = int(echo_only[:, 0].max(dim=1).values.argmax())
    peak_bearing = float(bearings[peak_i])
    # The body's own angular extent, not its centre bearing.  A 4 m cylinder at
    # 35 m spans 7 degrees, so "within a beamwidth of the centre" would fail a
    # peak sitting squarely on the object's near end -- which is exactly where
    # the specular from a cylinder at this aspect comes from.
    body_az = torch.rad2deg(torch.atan2(world[:, 1], world[:, 0])).detach()
    az_lo, az_hi = float(body_az.min()), float(body_az.max())
    off_body = max(0.0, az_lo - peak_bearing, peak_bearing - az_hi)

    print(f"  echo peaks at bearing {peak_bearing:+.2f} deg; the body spans "
          f"{az_lo:+.2f} to {az_hi:+.2f} deg ({beamwidth:.2f} deg beam)")
    print(f"  contact at {contact_ground:.2f} m ground, from its own echo")
    print(f"  shadow runs to {edge:.2f} m slant = {edge_ground:.2f} m on the "
          f"ground, against {far_edge:.2f} m predicted")
    print(f"  shadow length {measured_len:.2f} m from the echo, "
          f"{edge_ground - ground(start):.2f} m from the dark band")
    print(f"  height reads {from_band:.2f} m .. {from_echo:.2f} m, "
          f"against {height:.2f} m modelled")
    print(f"  the band is {contrast:.1f} dB below the seabed beyond it")

    banner("gradients")
    loss = image.clamp_min(1e-30).log10().mean()
    loss.backward()
    grads = {n: float(p.grad.abs().max()) for n, p in obj.named_parameters()
             if p.grad is not None}
    print("  d(log image)/d(object pose): "
          + ", ".join(f"{n} {v:.3e}" for n, v in grads.items()))
    print("  (through the echo -- the shadow's edge is a step and has none)")

    banner("figure")
    with timed("  draw"):
        # far_edge is a GROUND range; the image is in slant range.
        predicted_slant = math.hypot(far_edge, altitude)
        fig = draw(power, flat.detach()[:, 0], bearings, ranges, profile,
                   reference, start, edge, predicted_slant, contact_slant,
                   min(from_band, from_echo), max(from_band, from_echo),
                   altitude)
        save(fig, "17_seabed_object_shadow.png")

    banner("acceptance")
    cell = float(ranges[1] - ranges[0])
    ok = True
    ok &= check("the echo lands within a beamwidth of the body",
                off_body <= beamwidth,
                f"{off_body:.2f} deg outside a body spanning {az_hi - az_lo:.2f} deg")
    ok &= check("there is a shadow behind it", contrast > 8.0,
                f"{contrast:.1f} dB below the seabed beyond it")
    footprint = OBJ_RANGE * math.radians(beamwidth)
    ok &= check("its far edge is where the geometry says",
                abs(edge_ground - far_edge) < footprint,
                f"{abs(edge_ground - far_edge):.2f} m out, against a "
                f"{footprint:.2f} m beam footprint ({cell:.2f} m range cell)")
    ok &= check("the height read off the shadow brackets the object's",
                min(from_band, from_echo) <= height <= max(from_band, from_echo),
                f"{min(from_band, from_echo):.2f} .. "
                f"{max(from_band, from_echo):.2f} m, true {height:.2f} m")
    ok &= check("the image still carries gradients to the object",
                bool(grads) and max(grads.values()) > 0.0)
    return 0 if ok else 1


def draw(power, lit, bearings, ranges, profile, seabed, start, edge, predicted,
         contact, low, high, altitude):
    import matplotlib.pyplot as plt
    import numpy as np
    from hydropt.plot import plot_fls_sector

    fig = plt.figure(figsize=(15.5, 5.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.15], wspace=0.34)
    # Referenced to the SEABED, not to the image's peak.  The shadow is a hole
    # in the reverberation, and against a reference set by a target 25 dB
    # brighter than the bottom the whole seabed is already black and the hole
    # is invisible -- the picture would show a bright contact on a dark
    # background and say nothing about the thing being measured.
    ref = seabed * 10.0 ** 1.2
    for col, (img, title) in enumerate((
            (lit, "shadowing off: seabed shows through"),
            (power, "shadowing on: the band behind it"))):
        ax = fig.add_subplot(gs[0, col])
        plot_fls_sector(img, bearings, ranges, dynamic_range=22.0, reference=ref,
                        colorbar_label="dB re the seabed", ring_step=10.0,
                        ax=ax, title=title)
        ax.set_xlim(-2.0, 16.0)
        ax.set_ylim(26.0, 50.0)
        ax.set_aspect("equal")

    ax = fig.add_subplot(gs[0, 2])
    r = ranges.numpy()
    db = 10.0 * np.log10(np.maximum(profile.numpy(), seabed * 1e-4) / seabed)
    ax.axvspan(start, edge, color="#dce9f4", zorder=0, label="the dark band")
    ax.plot(r, db, lw=1.0, color="#1b3a5c")
    ax.axhline(0.0, ls=":", color="#888888", lw=1.0)
    ax.axvline(contact, color="#6a4fb0", lw=1.2, ls="-.", label="contact echo")
    ax.axvline(edge, color="#c2452d", lw=1.5, label=f"measured edge {edge:.1f} m")
    ax.axvline(predicted, color="#2e7d32", lw=1.5, ls="--",
               label=f"predicted edge {predicted:.1f} m")
    ax.set_xlim(contact - 8.0, edge + 10.0)
    ax.set_ylim(-34.0, 30.0)
    ax.set_xlabel("slant range (m)")
    ax.set_ylabel("dB re the seabed level")
    ax.set_title(f"height reads {low:.2f} - {high:.2f} m, true {OBJ_DIAMETER:.2f} m")
    ax.legend(fontsize=8, loc="lower right")
    fig.suptitle(f"A {OBJ_LENGTH:.1f} x {OBJ_DIAMETER:.1f} m cylinder on the "
                 f"seabed at {OBJ_RANGE:.0f} m, from an AUV {altitude:.2f} m above "
                 "the bottom.\nThe echo says where it is; the shadow says how "
                 "tall it is.", fontsize=11)
    return fig


if __name__ == "__main__":
    raise SystemExit(main())
