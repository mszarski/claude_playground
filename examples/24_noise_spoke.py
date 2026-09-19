"""A spoke in the picture: what lights a whole bearing, and what does not.

Operators of a forward-looking sonar know the artefact: a bright line from
the sonar out to the edge of the picture along one bearing, the bearing of a
vessel, that comes and goes as the vessel turns.  Two explanations are
offered for it -- a *glint*, a specular flash off the hull, or *emission*,
sound the vessel makes -- and this example builds both at ``examples/21``'s
settings, off 21's boat-hull mesh, to show which one draws a spoke.

The answer is in what a delay-and-sum beamformer does with a signal.  It
sums the elements with the delays of one look direction and reads the sum
out over time; a bearing is a beam, a range is a time.  An echo, however
bright, is a pulse: it is in the time bins of its own range and no others,
and what it does across the picture is spread across BEARING through the
beam pattern's sidelobes -- an arc at its range, not a line along its
bearing.  There is no degeneracy in the beamforming that turns a bright
range bin into a bright bearing.  A spoke needs a signal that is in EVERY
time bin of the beam that points at it, and that is a signal the sonar did
not send: emission from the vessel, arriving continuously from its bearing,
so that every range along that beam reads the same level.  The time-varying
gain then makes it a spoke that BRIGHTENS with range, because the
reverberation it competes with falls off and the emission does not.

Two scenarios, each against the bare picture (21's boat at rest, heading
40 degrees, 58 degrees off the line of sight):

* **emission**: the boat's propeller radiates broadband noise -- 115 dB re
  1 uPa^2/Hz at 1 m at 120 kHz, a cavitating propeller on a 30 m vessel, 150 dB
  across the sonar's 3.3 kHz band -- from a point just aft of the transom of
  the hull mesh, 2 m down.  It reaches the array by every path the method of
  images finds (direct, surface, bottom), one way, through the receive
  beam's own directivity, as a train of pulses the receiver's bandwidth
  wide with random phases: band-limited noise at the emission's received
  level.  The hull is in the way at some aspects.  Each path's first 35 m
  is tested against the hull mesh, and a path the hull blocks is dropped,
  so the spoke depends on where the propeller is seen from: run twice,
  stern towards the sonar (21's heading) and bow-on (heading 162), the hull
  between the propeller and the array.
* **glint**: no emission; the boat turned broadside to the line of sight
  (heading 72), so the flat of its side faces the sonar and the
  physical-optics integral over the hull returns its specular flash.  The
  boat is tens of dB brighter than at 21's aspect, and the picture shows
  what that does: an arc across bearing at the boat's range, from the
  Hamming window's sidelobes, and nothing along its bearing.

Acceptance criteria:
  * with the propeller in view, the boat's bearing reads well above its
    neighbours at every range away from the boat -- a spoke -- and does not
    in the bare picture;
  * bow-on, the hull shadows the propeller and the spoke is much weaker;
  * broadside, the boat's peak is far brighter than at 21's aspect and still
    draws no spoke: the bearing away from the boat reads as its neighbours.

Float32, as ``examples/23``: pictures, not wavelength-scale derivatives.  The
switches of ``examples/21`` (``HYDROPT_FAR``, ``HYDROPT_BOAT``, ...) carry
through, since that module is imported for its settings.
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
    LambertScattering, add_receiver_noise, azimuth_steering, beam_noise_power,
    beam_power_scale, beamform, calibrate, line_array_directivity_db,
    make_time_grid, reverberation_arrivals, shading_window, target_arrivals, trace,
)
from hydropt.beamform import ArrivalSet
from hydropt.eigenray import image_arrivals_batched
from hydropt.mesh import boat_hull_mesh, mesh_target, segment_mesh_transmission

SCENARIO = os.environ.get("HYDROPT_SCENARIO", "all")
if SCENARIO not in ("all", "emission", "bow-on", "glint"):
    raise SystemExit(f"HYDROPT_SCENARIO must be all, emission, bow-on or glint, got {SCENARIO!r}")
SCENARIOS = ("emission", "bow-on", "glint") if SCENARIO == "all" else (SCENARIO,)
CAPTION = {"emission": "the propeller radiating, stern towards the sonar",
           "bow-on": "the propeller radiating, bow-on: the hull in the way",
           "glint": "the boat broadside: a glint, and no emission"}

EMISSION_DB_HZ = 115.0          # radiated noise at 120 kHz, dB re 1 uPa^2/Hz at 1 m
PROPELLER = (-15.6, 0.0, 2.0)   # hull frame: 0.6 m aft of the transom, 2 m down
SHADOW_REACH = 35.0             # metres of each path tested against the hull
BOW_ON_HEADING = 162.0          # bow towards the sonar (bearing -18 + 180)
BROADSIDE_HEADING = 72.0        # the side square to the line of sight


def _ex21():
    path = Path(__file__).resolve().parent / "21_long_range_300m.py"
    spec = importlib.util.spec_from_file_location("_ex21", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def place(vertices: torch.Tensor, yaw_deg: float, position) -> torch.Tensor:
    """Body-frame points into the world, as mesh_target places its mesh."""
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    rot = torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=vertices.dtype)
    return vertices @ rot.T + torch.tensor(position, dtype=vertices.dtype)


def main() -> int:
    setup(double=False)
    banner("24 -- a spoke: emission from the boat against a glint off it")
    ex = _ex21()
    ex15 = ex._ex15()
    C = ex.C
    rx = ex.horizontal_array()
    scene, bottom, sea, sediment = ex.build_scene(rx, learnable=False)
    print(f"  {ex.FREQ_KHZ:.0f} kHz, {ex.N_RX} x {ex.N_TX}, {2 * ex.SECTOR_DEG:.0f} deg "
          f"swath to {ex.FAR:.0f} m; AUV at {ex.AUV_DEPTH:.0f} m in {ex.WATER_DEPTH:.0f} m")

    # ---- 21's boat, at a heading ------------------------------------------ #
    b = math.radians(ex.BOAT_BEARING_DEG)
    head = (ex.BOAT_RANGE * math.cos(b), ex.BOAT_RANGE * math.sin(b), 0.0)
    girth = ex.HULL_BEAM + 2.0 * ex.HULL_DRAUGHT
    verts, faces = boat_hull_mesh(ex.HULL_LENGTH, ex.HULL_BEAM, ex.HULL_DRAUGHT,
                                  n_long=int(round(110 * ex.HULL_LENGTH / 12.0)),
                                  n_around=int(round(34 * girth / (3.2 + 2.0 * 4.0))))

    def make_boat(heading_deg):
        return mesh_target(verts, faces, position=head, yaw=heading_deg, n_patches=6,
                           sound_speed=C, diffuse_db=ex.DIFFUSE_DB, learnable=True,
                           learnable_shape=False, facet_chunk=4096, checkpoint=False)

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
    band_hz = 1.0 / ex.PULSE_S
    noise = float(beam_noise_power(scene.freqs_khz, bandwidth_hz=band_hz,
                                   directivity_db=line_array_directivity_db(ex.N_RX),
                                   wind_speed=ex.WIND))
    solid = (math.radians(2 * ex.SECTOR_DEG)
             * math.radians(ex.ELEV_DEG[1] - ex.ELEV_DEG[0]) / dirs.shape[0])
    seabed = LambertScattering(-27.0, learnable=False)
    span_y = ex.FAR * math.sin(math.radians(ex.SECTOR_DEG)) * 1.02
    x_range = (-0.03 * ex.FAR, 1.02 * ex.FAR)
    pixel_m = (x_range[1] - x_range[0]) / 299.0
    centre = rx.mean(dim=0)

    def echo(target):
        return target_arrivals(
            scene, target, dirs, return_leg="eigenray", n_rx_rays=2000,
            rx_half_angle_deg=45.0, tx_weights=w_tx, tx_pattern=ex.transmit_pattern,
            rx_pattern=rx_beam, max_arrivals_per_leg=24,
            generator=torch.Generator().manual_seed(ex.SEED))

    # ---- the emission ----------------------------------------------------- #
    emission_db = EMISSION_DB_HZ + 10.0 * math.log10(band_hz)     # in the band

    def emission(heading_deg):
        """The propeller's noise at the array: band-limited, continuous, one way.

        Returns the arrivals, and how many of the paths the hull let through.
        """
        prop = place(torch.tensor([PROPELLER]), heading_deg, head).reshape(3)
        hull = place(verts, heading_deg, head)
        paths = image_arrivals_batched(scene, prop.reshape(1, 3), centre.reshape(1, 3),
                                       scene.freqs_khz)[0]
        # the hull's shadow: a path whose first metres cross the hull is gone
        starts = prop.reshape(1, 3).expand(paths.n_arrivals, 3)
        ends = starts + SHADOW_REACH * paths.launch_direction
        clear = segment_mesh_transmission(starts, ends, hull, faces)
        # what the array hears of it: the element's directivity, and the level
        amp = paths.amplitude * rx_beam(paths.direction).reshape(-1, 1) * clear.reshape(-1, 1)
        keep = (amp.detach().max(dim=1).values > 0.0).nonzero().reshape(-1)
        n_clear, n_paths = int(keep.numel()), paths.n_arrivals
        if n_clear == 0:
            return None, 0, n_paths
        amp = amp[keep]
        # A train of pulses, one receiver bandwidth wide, a pulse-width apart,
        # with random phases: band-limited noise.  Its mean power after the
        # beamformer's unit-area envelope and calibrate() is
        # a^2 * 10^(SL/10) * sqrt(pi), so the amplitude that puts the received
        # level at (emission level - path loss) is the path's own pressure
        # ratio, scaled from the sonar's source level to the emission's, over
        # pi^(1/4).
        spacing = ex.PULSE_S
        t = torch.arange(float(grid[0]) - 3 * spacing, float(grid[-1]) + 3 * spacing, spacing)
        n_t, n_p = int(t.shape[0]), n_clear
        level = 10.0 ** ((emission_db - ex.SOURCE_LEVEL_DB) / 20.0) / math.pi ** 0.25
        g = torch.Generator().manual_seed(ex.SEED + 3)
        phase = 2.0 * math.pi * torch.rand(n_t * n_p, generator=g)
        rep = lambda x: x[keep].repeat(n_t, *([1] * (x.ndim - 1)))
        arrivals = ArrivalSet(
            time=t.repeat_interleave(n_p),
            amplitude=(amp * level).repeat(n_t, 1),
            direction=rep(paths.direction), phase=phase,
            distance=rep(paths.distance), path_length=rep(paths.path_length),
            launch_direction=rep(paths.launch_direction))
        return arrivals, n_clear, n_paths

    def ping(*, boat, extra=None):
        result = trace(scene, dirs)
        rev = reverberation_arrivals(
            result, dirs, scene.freqs_khz, scattering=seabed,
            solid_angle_per_ray=solid, ray_weights=w_tx * rx_beam(-dirs),
            boundary="both", surface=sea, bottom=scene.bottom, max_arrivals=ex.PATCHES,
            generator=torch.Generator().manual_seed(ex.SEED + 1))
        parts = [rev, echo(boat)] + ([extra] if extra is not None else [])
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
        return cart, gx, gy, noisy

    banner("the bare picture: sea, seabed, the boat at rest")
    with torch.no_grad(), timed("  ping"):
        bare, gx, gy, bare_polar = ping(boat=make_boat(ex.BOAT_HEADING_DEG))
    X, Y = torch.meshgrid(gx, gy, indexing="xy")
    R = torch.hypot(X, Y)
    B = torch.rad2deg(torch.atan2(Y, X))
    db = lambda t: 10.0 * torch.log10(t.detach().clamp_min(1e-30))
    bare_db = db(bare)
    ext = [float(gx.min()), float(gx.max()), float(gy.min()), float(gy.max())]
    beam_deg = ex.beam_3db_deg(ex.N_RX, shading)
    # the bearing, away from the boat, and its neighbours at the same ranges
    off = (B - ex.BOAT_BEARING_DEG).abs()
    away = (R > ex.NEAR + 15.0) & (R < ex.BOAT_RANGE - 25.0)
    on_bearing = (off < 0.5 * beam_deg) & away
    beside = (off > 3.0 * beam_deg) & (off < 8.0 * beam_deg) & away
    spoke_of = lambda img: float(img[on_bearing].median() - img[beside].median())
    at_boat = (X - head[0]) ** 2 + (Y - head[1]) ** 2 < 20.0 ** 2
    bare_spoke = spoke_of(bare_db)
    bare_peak = float(bare_db[at_boat].max())
    print(f"  the boat's bearing away from the boat reads {bare_spoke:+.1f} dB over its "
          f"neighbours; the boat peaks at {bare_peak:+.1f} dB")

    ok = True
    for name in SCENARIOS:
        banner(f"scenario: {name}")
        if name == "glint":
            heading, extra, n_clear, n_paths = BROADSIDE_HEADING, None, 0, 0
            print(f"  heading {heading:.0f} deg: the side square to the line of sight")
        else:
            heading = ex.BOAT_HEADING_DEG if name == "emission" else BOW_ON_HEADING
            with torch.no_grad(), timed("  the emission's paths"):
                extra, n_clear, n_paths = emission(heading)
            print(f"  heading {heading:.0f} deg; the propeller at {EMISSION_DB_HZ:.0f} dB/Hz "
                  f"({emission_db:.0f} dB in the band): {n_clear} of {n_paths} paths clear "
                  f"the hull" + (f", {extra.n_arrivals} pulses" if extra is not None else ""))
        boat = make_boat(heading)
        with torch.no_grad(), timed("  ping"):
            cart, _, _, polar = ping(boat=boat, extra=extra)
        cart_db = db(cart)
        spoke = spoke_of(cart_db)
        peak = float(cart_db[at_boat].max())
        print(f"  the boat's bearing away from the boat reads {spoke:+.1f} dB over its "
              f"neighbours ({bare_spoke:+.1f} dB bare); the boat peaks at {peak:+.1f} dB "
              f"({bare_peak:+.1f} dB bare)")
        if name == "emission":
            spoke_seen = spoke
            ok &= check("the propeller in view draws a spoke, and the bare picture has none",
                        spoke > 6.0 and bare_spoke < 2.0,
                        f"{spoke:+.1f} dB along the bearing, {bare_spoke:+.1f} dB bare")
        elif name == "bow-on":
            ok &= check("bow-on, the hull shadows the propeller and the spoke fades",
                        spoke < spoke_seen - 6.0 if "emission" in SCENARIOS else spoke < 3.0,
                        f"{spoke:+.1f} dB along the bearing, {n_clear} of {n_paths} paths clear")
        else:
            # The ring: the glint's range bins over the bearings away from the
            # boat, against the ranges beside them -- on the calibrated image
            # BEFORE the display, because the median gain takes each range
            # bin's own median over beams as its reference, and an arc that
            # lifts every beam lifts that reference with it and is divided
            # straight back out.  (A median reference cannot be moved by a
            # target in a few beams; an arc is in all of them.)
            k = int(torch.argmax(polar[:, 0].max(dim=0).values))
            ring = polar[:, 0, max(k - 2, 0):k + 3]
            near = torch.cat([polar[:, 0, max(k - 30, 0):max(k - 10, 0)],
                              polar[:, 0, k + 10:k + 30]], dim=1)
            far_beams = (bearings - ex.BOAT_BEARING_DEG).abs() > 4.0 * beam_deg
            arc = float(10 * torch.log10(ring[far_beams].median() / near[far_beams].median()))
            print(f"  at the glint's range, before the gain, the bearings away from the boat "
                  f"read {arc:+.1f} dB over the ranges beside them: the sidelobes' arc, "
                  f"which the median gain then removes")
            ok &= check("broadside, the glint is far brighter than 21's aspect and draws no spoke",
                        peak > bare_peak + 6.0 and spoke < 2.0,
                        f"peak {peak:+.1f} vs {bare_peak:+.1f} dB; {spoke:+.1f} dB along the bearing")

        # ---- the figure --------------------------------------------------- #
        ref = float(bare_db.max())
        fig, axes = plt.subplots(1, 3, figsize=(19, 6.2))
        for ax, img, title in ((axes[0], bare_db, "the bare picture: 21's boat at rest"),
                               (axes[1], cart_db, CAPTION[name])):
            im = ax.imshow(img.numpy(), origin="lower", extent=ext, vmin=ex.THRESHOLD_DB,
                           vmax=ref, cmap="inferno", aspect="equal")
            ax.set_title(title); ax.set_xlabel("forward (m)"); ax.set_ylabel("across (m)")
        m = axes[2].imshow((cart_db - bare_db).clamp(-15.0, 25.0).numpy(), origin="lower",
                           extent=ext, vmin=-15, vmax=25, cmap="coolwarm", aspect="equal")
        axes[2].set_title(f"{name} minus bare (dB)"); axes[2].set_xlabel("forward (m)")
        fig.colorbar(im, ax=axes[1], fraction=0.04, label="dB re the background at that range")
        fig.colorbar(m, ax=axes[2], fraction=0.04, label="dB")
        for ax in axes:
            ax.plot([0.0, ex.FAR * math.cos(b)], [0.0, ex.FAR * math.sin(b)], "c:", lw=0.8, alpha=0.6)
            ax.plot(head[0], head[1], "c+", ms=10, mew=1.5)
            ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
        fig.suptitle(f"{ex.FREQ_KHZ:.0f} kHz FLS, {2 * ex.SECTOR_DEG:.0f} deg to {ex.FAR:.0f} m, "
                     f"median TVG floored at +{ex.THRESHOLD_DB:.0f} dB: {CAPTION[name]}")
        save(fig, f"24_spoke_{name}.png")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
