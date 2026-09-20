"""A boat under way: the hull moved along a track, a picture per ping.

``examples/21``'s sonar and sea, and its 30 m boat -- but the boat is moving.
It comes in from the port side heading towards the sonar, turns through the
swath and runs off to starboard with its stern to us, at 12 knots, a ping
every few seconds, and every ping is rendered as 21 would render it: the
hull's echo formed on the reverberation's complex beams, the receiver noise
drawn afresh, the median-TVG display at the gain of the first frame, the
picture resampled to metres.  The frames go into a GIF and a contact sheet.
Twice: a **quiet** boat, and the same boat with its propeller
**radiating**, so that the spoke of ``examples/24`` appears as the boat
turns its stern to us and not before.

The spoke is not there whenever the propeller is: a propeller's noise is
shielded forward by its own hull and heard through its own bubble wake
astern, so a vessel bow-on draws no spoke and one quartering away draws a
bright one (:func:`hydropt.propeller_directivity`: 20 dB down at the bow,
10 abeam, a 6 dB notch dead astern, on top of the hull's exact shadow of
the first metres of each path).  The U-turn track runs the boat through
every aspect: the checks below sort the frames by the angle at which the
sonar sees the stern.

What makes a sequence cheap is what made ``examples/22``'s fit cheap.  The
beamformer is linear in the arrivals, so the sea's and seabed's complex
beams are traced and formed once (10 s at 300 m) and each frame costs only
the hull's echo (its outbound leg from the same trace, its return leg by
the method of images) and its beams, a few seconds; the emission is a
one-way solve, cheaper still.  :class:`hydropt.PictureRenderer` holds that
background and :class:`hydropt.Trajectory` gives the poses; the example is
the track, the metrics and the drawing.

Two things are held across the frames on purpose.  The display gain is
taken from the first frame and kept (``display(gain=)``), as a sonar's AGC
would settle rather than hop per ping, so a frame's brightness means what
it meant in the last one.  And the receiver noise is a fresh draw per frame
(``seed + 2 + frame``) while the reverberation is not: the sea is the same
sea between pings a few seconds apart, the electronics are not.

**Construction and assumptions.**  21's sonar, scene, boat, display and
grid (see 21's docstring) in a ``PictureRenderer``; the track a
``Trajectory`` through ``track()``'s waypoints at ``SPEED`` (times ``S``),
sampled at ``N_FRAMES`` times; the hull rebuilt at each pose by ``boat()``;
the propeller at ``PROPELLER`` radiating ``EMISSION_DB_HZ`` through
``emission_arrivals`` with ``propeller_directivity(heading)`` and the hull
as its own occluder over ``SHADOW_REACH``.  Assumptions: as 21's and 24's;
the sea is one realisation across the frames and the sonar does not move
(``examples/29`` is the other case); the display gain is the first
frame's.

Acceptance criteria:
  * quiet, in every frame the hull's echo -- the difference between the
    frame and the bare picture, above threshold, within 40 m of the truth
    -- has its centroid within half the hull's length plus one beam width
    of where the boat is, and its peak within 25 m (the hull is the same
    echo in the radiating run, where the spoke crosses that disc);
  * with the propeller radiating, the propeller's bearing reads more than
    6 dB over its neighbours at ranges short of the boat in every frame
    that sees the stern within 60 degrees, and under 3 dB in every frame
    that sees the bow within 60 degrees (the spoke comes and goes with the
    aspect); quiet, under 2 dB in the median frame;
  * a frame costs less than the background did.

The track and its speed scale with the head's swath (``HYDROPT_SONAR=330``
halves both); ``HYDROPT_SCENARIO`` picks ``quiet``, ``emission`` or ``all``,
``HYDROPT_FRAMES`` the number of frames (16).
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
    LambertScattering, PictureRenderer, Trajectory, azimuth_steering, beam_noise_power,
    emission_arrivals, line_array_directivity_db, make_time_grid, propeller_directivity,
    shading_window,
)
from hydropt.mesh import boat_hull_mesh, mesh_target

SCENARIO = os.environ.get("HYDROPT_SCENARIO", "all")
if SCENARIO not in ("all", "quiet", "emission"):
    raise SystemExit(f"HYDROPT_SCENARIO must be all, quiet or emission, got {SCENARIO!r}")
SCENARIOS = ("quiet", "emission") if SCENARIO == "all" else (SCENARIO,)
CAPTION = {"quiet": "the boat under way, quiet",
           "emission": "the boat under way, its propeller radiating"}
N_FRAMES = int(os.environ.get("HYDROPT_FRAMES", 16))
SPEED = 6.0                     # m/s, 12 knots
EMISSION_DB_HZ = 115.0          # radiated noise at 120 kHz, dB re 1 uPa^2/Hz at 1 m (scaled below)
PROPELLER = (-15.6, 0.0, 2.0)   # hull frame: 0.6 m aft of the transom, 2 m down (as 24)
SHADOW_REACH = 35.0


def _ex21():
    path = Path(__file__).resolve().parent / "21_long_range_300m.py"
    spec = importlib.util.spec_from_file_location("_ex21", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_EX = _ex21()
S = _EX.FAR / 300.0     # the track was laid out for the 120 kHz head's 300 m swath; scale with it
EMISSION_DB_HZ = EMISSION_DB_HZ - 20.0 * math.log10(_EX.FREQ_KHZ / 120.0)


def place(vertices: torch.Tensor, yaw_deg: float, position) -> torch.Tensor:
    """Body-frame points into the world, as mesh_target places its mesh."""
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    rot = torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=vertices.dtype)
    return vertices @ rot.T + torch.tensor(position, dtype=vertices.dtype)


def track() -> Trajectory:
    """In from port heading at the sonar, a U-turn through the swath, off to starboard."""
    f = _EX.FAR
    waypoints = [(0.92 * f, 0.42 * f), (0.75 * f, 0.28 * f), (0.62 * f, 0.12 * f),
                 (0.57 * f, -0.05 * f), (0.62 * f, -0.20 * f), (0.75 * f, -0.32 * f),
                 (0.92 * f, -0.42 * f)]
    return Trajectory.from_waypoints(waypoints, speed=SPEED * S)


def aspect_deg(x, y, heading_deg) -> float:
    """The angle at which the sonar sees the stern: 0 dead astern, 180 bow-on."""
    to_sonar = math.degrees(math.atan2(-y, -x))
    stern = heading_deg + 180.0
    return abs((to_sonar - stern + 180.0) % 360.0 - 180.0)


def main() -> int:
    setup(double=False)
    banner("28 -- a boat under way, a picture per ping")
    ex = _EX
    ex15 = ex._ex15()
    C = ex.C
    rx = ex.horizontal_array()
    scene, bottom, sea, sediment = ex.build_scene(rx, learnable=False)
    print(f"  {ex.FREQ_KHZ:.0f} kHz, {ex.N_RX} x {ex.N_TX}, {2 * ex.SECTOR_DEG:.0f} deg "
          f"swath to {ex.FAR:.0f} m; AUV at {ex.AUV_DEPTH:.0f} m in {ex.WATER_DEPTH:.0f} m")

    # ---- the sonar, as 21 has it, in a renderer --------------------------- #
    dirs, _ = ex.transmit_fan(seed=ex.SEED)
    w_tx = ex.transmit_pattern(dirs)
    tilt = ex.passes_deg()[0]
    rx_beam = lambda d: ex.receive_beam(d, tilt)
    steer, bearings = azimuth_steering(ex.N_BEAMS, ex.SECTOR_DEG)
    grid = make_time_grid(2.0 * ex.NEAR / C, 2.0 * ex.FAR / C, ex.N_BINS)
    rng = grid * C / 2.0
    shading = shading_window(ex.N_RX, "hamming")
    noise = float(beam_noise_power(scene.freqs_khz, bandwidth_hz=1.0 / ex.PULSE_S,
                                   directivity_db=line_array_directivity_db(ex.N_RX),
                                   wind_speed=ex.WIND))
    solid = (math.radians(2 * ex.SECTOR_DEG)
             * math.radians(ex.ELEV_DEG[1] - ex.ELEV_DEG[0]) / dirs.shape[0])
    span_y = ex.FAR * math.sin(math.radians(ex.SECTOR_DEG)) * 1.02
    x_range = (-0.03 * ex.FAR, 1.02 * ex.FAR)
    pixel_m = (x_range[1] - x_range[0]) / 299.0
    gain = []          # the first frame's, then held (see the docstring)

    def display(noisy):
        if not gain:
            gain.append(ex.display_gain(noisy, rng, pixel_m=pixel_m))
        return ex.display(noisy, rng, pixel_m=pixel_m, gain=gain[0])[0]

    def to_cartesian(shown):
        return ex15.to_cartesian(shown, bearings, grid, n_x=300, n_y=300,
                                 x_range=x_range, y_range=(-span_y, span_y))

    renderer = PictureRenderer(
        scene, elements=rx, directions=dirs, tx_weights=w_tx, tx_pattern=ex.transmit_pattern,
        rx_pattern=rx_beam, time_grid=grid, steer=steer, sigma_t=ex.PULSE_S, shading=shading,
        source_level_db=ex.SOURCE_LEVEL_DB, noise_power=noise,
        scattering=LambertScattering(-27.0, learnable=False), solid_angle_per_ray=solid,
        boundary="both", max_arrivals=ex.PATCHES, display=display, to_cartesian=to_cartesian,
        seed=ex.SEED)

    # ---- 21's hull, built at any pose -------------------------------------- #
    girth = ex.HULL_BEAM + 2.0 * ex.HULL_DRAUGHT
    verts, faces = boat_hull_mesh(ex.HULL_LENGTH, ex.HULL_BEAM, ex.HULL_DRAUGHT,
                                  n_long=int(round(110 * ex.HULL_LENGTH / 12.0)),
                                  n_around=int(round(34 * girth / (3.2 + 2.0 * 4.0))))

    def boat(x, y, heading_deg):
        return mesh_target(verts, faces, position=(x, y, 0.0), yaw=heading_deg, n_patches=6,
                           sound_speed=C, diffuse_db=ex.DIFFUSE_DB, learnable=False,
                           learnable_shape=False, facet_chunk=4096, checkpoint=False)

    centre = rx.mean(dim=0)

    def propeller(x, y, heading_deg, frame):
        """What the propeller radiates from this pose, shadowed by its own hull."""
        pos = (x, y, 0.0)
        prop = place(torch.tensor([PROPELLER]), heading_deg, pos).reshape(3)
        arr, n_clear, n_paths = emission_arrivals(
            scene, prop, centre, grid, spectrum_level_db=EMISSION_DB_HZ,
            source_level_db=ex.SOURCE_LEVEL_DB, pulse_s=ex.PULSE_S, rx_pattern=rx_beam,
            pattern=propeller_directivity(heading_deg),
            occluder=(place(verts, heading_deg, pos), faces), shadow_reach=SHADOW_REACH,
            generator=torch.Generator().manual_seed(ex.SEED + 3 + frame))
        clear.append((n_clear, n_paths))
        return arr

    # ---- the track --------------------------------------------------------- #
    traj = track()
    times = torch.linspace(0.0, traj.duration, N_FRAMES).tolist()
    print(f"  {N_FRAMES} frames over {traj.duration:.0f} s at {SPEED * S:.1f} m/s, "
          f"a ping every {times[1] - times[0]:.1f} s; the track from "
          f"({float(traj.positions[0, 0]):.0f}, {float(traj.positions[0, 1]):.0f}) to "
          f"({float(traj.positions[-1, 0]):.0f}, {float(traj.positions[-1, 1]):.0f}) m")

    banner("the bare picture (and the background, once)")
    t0 = time.perf_counter()
    with torch.no_grad():
        bare, gx, gy = renderer.picture([])
    t_back = time.perf_counter() - t0
    print(f"  {renderer.n_reverberation} patches; background + bare picture {t_back:.1f} s")
    X, Y = torch.meshgrid(gx, gy, indexing="xy")
    B = torch.rad2deg(torch.atan2(Y, X))
    R = torch.hypot(X, Y)
    db = lambda t: 10.0 * torch.log10(t.detach().clamp_min(1e-30))
    bare_db = db(bare)
    ref = float(bare_db.max())
    ext = [float(gx.min()), float(gx.max()), float(gy.min()), float(gy.max())]
    beam_deg = ex.beam_3db_deg(ex.N_RX, shading)

    def echo_of(cart, pose):
        """Centroid and peak of the frame's excess over the bare picture near the boat."""
        x, y, _ = pose
        near = (X - x) ** 2 + (Y - y) ** 2 < 40.0 ** 2
        diff = (db(cart) - bare_db).clamp_min(0.0) * near
        lit = diff * (db(cart) > ex.THRESHOLD_DB)
        w = lit.sum()
        if float(w) <= 0.0:
            return float("nan"), float("nan")
        cx, cy = float((lit * X).sum() / w), float((lit * Y).sum() / w)
        k = int(lit.argmax())
        px, py = float(X.reshape(-1)[k]), float(Y.reshape(-1)[k])
        return math.hypot(cx - x, cy - y), math.hypot(px - x, py - y)

    def spoke_of(img, bearing_deg, boat_range):
        """The bearing, short of the boat, over its neighbours at the same ranges."""
        away = (R > ex.NEAR + 15.0) & (R < boat_range - 25.0)
        off = (B - bearing_deg).abs()
        on_bearing = (off < 0.5 * beam_deg) & away
        beside = (off > 3.0 * beam_deg) & (off < 8.0 * beam_deg) & away
        return float(img[on_bearing].median() - img[beside].median())

    ok = True
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for name in SCENARIOS:
        banner(f"scenario: {name}")
        clear = []
        emitters = [propeller] if name == "emission" else []
        frames, poses, cents, peaks, spokes, costs, aspects = [], [], [], [], [], [], []
        last = time.perf_counter()
        for k, (t, pose, (cart, _, _)) in enumerate(
                renderer.sequence(boat, traj, times, emitters=emitters)):
            now = time.perf_counter()          # the generator worked between yields
            costs.append(now - last)
            last = now
            x, y, h = pose
            prop = place(torch.tensor([PROPELLER]), h, (x, y, 0.0)).reshape(3)
            sb = math.degrees(math.atan2(float(prop[1]), float(prop[0])))
            cart_db = db(cart)
            frames.append(cart_db); poses.append(pose)
            c_err, p_err = echo_of(cart, pose)
            cents.append(c_err); peaks.append(p_err)
            spokes.append(spoke_of(cart_db, sb, math.hypot(x, y)))
            aspects.append(aspect_deg(x, y, h))
            note = (f"; {clear[-1][0]} of {clear[-1][1]} emission paths clear"
                    if name == "emission" else "")
            print(f"  t = {t:5.1f} s  boat ({x:6.1f}, {y:7.1f}) m heading {h:6.1f} deg, "
                  f"stern seen at {aspects[-1]:5.1f} deg: echo centroid {c_err:5.1f} m off, "
                  f"peak {p_err:5.1f} m off; spoke {spokes[-1]:+5.1f} dB{note}")
        per_frame = sum(costs) / len(costs)
        tol = 0.5 * ex.HULL_LENGTH + math.radians(beam_deg) * 0.8 * ex.FAR
        print(f"  {len(frames)} frames, {per_frame:.1f} s each; centroid off by "
              f"{max(cents):.1f} m at most (tolerance {tol:.1f} m), peak by {max(peaks):.1f} m; "
              f"spoke median {sorted(spokes)[len(spokes) // 2]:+.1f} dB")
        med_spoke = sorted(spokes)[len(spokes) // 2]
        if name == "emission":
            # the hull is the quiet run's (same pose, same rays): what is
            # checked here is the spoke, which runs through the 40 m disc at
            # +20 dB and would be measured as the "echo" otherwise
            astern = [sp for sp, a in zip(spokes, aspects) if a < 60.0]
            ahead = [sp for sp, a in zip(spokes, aspects) if a > 120.0]
            print(f"  stern within 60 deg in {len(astern)} frames: spoke {min(astern):+.1f} to "
                  f"{max(astern):+.1f} dB; bow within 60 deg in {len(ahead)} frames: "
                  f"{min(ahead):+.1f} to {max(ahead):+.1f} dB")
            ok &= check("radiating: a spoke whenever the stern is towards us, none bow-on",
                        astern and ahead and min(astern) > 6.0 and max(ahead) < 3.0,
                        f"stern-on at least {min(astern):+.1f} dB, bow-on at most {max(ahead):+.1f} dB")
        else:
            ok &= check("quiet: the echo follows the boat in every frame",
                        all(c < tol for c in cents) and all(p < 25.0 for p in peaks),
                        f"centroid within {max(cents):.1f} m, peak within {max(peaks):.1f} m")
            ok &= check("quiet: no spoke", med_spoke < 2.0, f"{med_spoke:+.1f} dB in the median frame")
        ok &= check(f"{name}: a frame costs less than the background",
                    per_frame < t_back, f"{per_frame:.1f} s a frame, {t_back:.1f} s the background")

        # ---- the GIF and the contact sheet --------------------------------- #
        fig, ax = plt.subplots(figsize=(9, 7.5))
        im = ax.imshow(frames[0].numpy(), origin="lower", extent=ext, vmin=ex.THRESHOLD_DB,
                       vmax=ref, cmap="inferno", aspect="equal")
        ax.plot(traj.positions[:, 0].numpy(), traj.positions[:, 1].numpy(), "c:", lw=0.8, alpha=0.6)
        dot, = ax.plot([], [], "c+", ms=12, mew=1.5)
        arrow, = ax.plot([], [], "c-", lw=1.2)
        ax.set_xlabel("forward (m)"); ax.set_ylabel("across (m)")
        ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
        fig.colorbar(im, ax=ax, fraction=0.04, label="dB re the background at that range")
        fig.suptitle(f"{ex.FREQ_KHZ:.0f} kHz FLS, {2 * ex.SECTOR_DEG:.0f} deg to {ex.FAR:.0f} m: "
                     f"{CAPTION[name]}")

        def draw(i):
            im.set_data(frames[i].numpy())
            x, y, h = poses[i]
            dot.set_data([x], [y])
            hx, hy = math.cos(math.radians(h)), math.sin(math.radians(h))
            arrow.set_data([x, x + 0.5 * ex.HULL_LENGTH * hx], [y, y + 0.5 * ex.HULL_LENGTH * hy])
            ax.set_title(f"t = {times[i]:.0f} s: the boat at ({x:.0f}, {y:.0f}) m, "
                         f"heading {h:.0f} deg")
            return im, dot, arrow

        anim = animation.FuncAnimation(fig, draw, frames=len(frames), interval=250, blit=False)
        gif = FIGURE_DIR / f"28_sequence_{name}{ex.TAG}.gif"
        with timed("  gif"):
            anim.save(gif, writer=animation.PillowWriter(fps=4))
        print(f"  wrote {gif}")
        plt.close(fig)

        pick = [int(round(i)) for i in torch.linspace(0, len(frames) - 1, 6).tolist()]
        fig, axes = plt.subplots(2, 3, figsize=(17, 9.5))
        for ax, i in zip(axes.ravel(), pick):
            x, y, h = poses[i]
            im = ax.imshow(frames[i].numpy(), origin="lower", extent=ext, vmin=ex.THRESHOLD_DB,
                           vmax=ref, cmap="inferno", aspect="equal")
            ax.plot(x, y, "c+", ms=10, mew=1.2)
            ax.set_title(f"t = {times[i]:.0f} s, heading {h:.0f} deg")
            ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
        fig.suptitle(f"{ex.FREQ_KHZ:.0f} kHz FLS to {ex.FAR:.0f} m: {CAPTION[name]}, "
                     f"six of {len(frames)} pings")
        save(fig, f"28_sequence_{name}{ex.TAG}.png")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
