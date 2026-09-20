"""A rubble-mound breakwater: why a wall looks like grains of rice.

``examples/23`` put a breakwater in the picture as a vertical caisson face,
and it drew what such a face must draw: a continuous band, 2 m patches well
under the beam's width merging into a line.  A real breakwater seen on a
forward-looking sonar rarely looks like that.  It looks like a row -- or a
scatter -- of bright grains, each a few metres long, lying across the line of
sight, of very unequal brightness: grains of rice.  The boat's hull at 90 m
in ``examples/21`` looked the same, and for the same two reasons.

* **It is discrete.**  A rubble-mound breakwater is a sloping pile of armour
  units -- concrete cubes or tetrapods of 2-4 m, or rock -- on a core of
  quarry run.  Each unit is a body of its own, and a flat face returns its
  specular flash only when it points at the sonar: at 120 kHz that lobe is
  a third of a degree wide, so of a few hundred randomly set units a
  handful flash, most return their faces' sidelobes tens of dB down, and
  every unit adds a diffuse return from its roughness.  The strengths spread
  over 30 dB.  A caisson has one face, and it does not point at the sonar.
* **Every point is a grain.**  The sonar's resolution cell is 0.22 m in range
  (the pulse) and one beam wide across it -- 5 m at 100 m, 14 m at 275 m --
  so a point scatterer is drawn as a dash lying across the line of sight,
  as long as the beam is wide there.  Discrete units at a pitch under the
  beam width become dashes that touch and overlap, with the bright ones
  standing out, and a continuous rough face becomes speckle at the same
  cell size.  Which way the grains lie is the sonar's, not the wall's: along
  the wall where the wall is square to the line of sight (ahead of the
  sonar), across it where the wall runs alongside.

The model, at 21's settings, off the same building blocks: a mound of
``seawall_mesh`` at the armour slope (4:3, 37 degrees) with a diffuse rock
face, occluding the water beyond it, and an armour layer of 3 m concrete
cubes (``box_mesh``) at 4 m pitch along and up the slope, each at a random
orientation, whose faces the physical-optics integral flashes or does not.
Three pictures: 23's caisson alongside, the rubble mound alongside in the
same place, and the rubble mound across the picture ahead, square to the
line of sight, where the grains lie along it.

**Construction and assumptions.**  21's sonar, scene and boat (see 21's
docstring).  What this adds: a rubble mound as a sloped mesh (``MOUND_SLOPE_DEG``,
diffuse ``MOUND_DIFFUSE_DB``) along the wall's line of 23 (``WALL_Y``,
``WALL_FROM``, ``WALL_TO``), with armour units as ``UNIT`` m cubes
(``box_mesh``) on a ``PITCH`` grid jittered by ``JITTER``, each a
``mesh_target`` with ``UNIT_DIFFUSE_DB``; the caisson of 23
(``CAISSON_DIFFUSE_DB``) as the comparison; a second mound dead ahead at ``AHEAD_X`` for the grain
profile, whose grains are found by ``scipy.signal.find_peaks``.
Assumptions: as 21's; the units are rigid cubes with one plane-wave patch
each, so at 330 kHz they are the grains themselves and at 120 kHz the
grains are their interference; no multiple scattering between units.

Acceptance criteria:
  * the rubble band's texture (its 95th percentile over its median, over
    the band's area) exceeds the caisson's by more than 4 dB: grains, not a
    band;
  * ahead, the grains along the wall are as long as the beam is wide there
    (the median grain, a peak 3 dB prominent, within 0.4 to 2 times the
    beamwidth at that range), and there are at least five per 100 m of wall
    -- a grain and its gap take about two beamwidths, so at this range there
    cannot be many more than eight.

Float32, as ``examples/23``.  21's switches carry through.
"""

from __future__ import annotations

import importlib.util
import math
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.signal import find_peaks

from _common import banner, check, save, setup, timed
from hydropt import (
    LambertScattering, add_receiver_noise, azimuth_steering, beam_noise_power,
    beam_power_scale, beamform, box_mesh, calibrate, line_array_directivity_db,
    make_time_grid, reverberation_arrivals, seawall_mesh, shading_window,
    target_arrivals, trace,
)
from hydropt.beamform import ArrivalSet
from hydropt.mesh import boat_hull_mesh, mesh_target

SCENARIO = os.environ.get("HYDROPT_SCENARIO", "all")
if SCENARIO not in ("all", "caisson", "rubble", "ahead"):
    raise SystemExit(f"HYDROPT_SCENARIO must be all, caisson, rubble or ahead, got {SCENARIO!r}")
SCENARIOS = ("caisson", "rubble", "ahead") if SCENARIO == "all" else (SCENARIO,)
CAPTION = {"caisson": "a caisson face alongside (as 23)",
           "rubble": "a rubble mound with armour, alongside",
           "ahead": "the rubble mound ahead, square to the line of sight"}

WALL_Y = 110.0                  # alongside: the toe line, metres to starboard
WALL_FROM, WALL_TO = 20.0, 320.0
AHEAD_X = 100.0                 # ahead: the toe line, metres forward -- the harbour wall at close range
WALL_ABOVE_WATER = 3.0
CAISSON_DIFFUSE_DB = -6.0
MOUND_SLOPE_DEG = 37.0          # 4:3, a single-layer concrete armour slope
MOUND_DIFFUSE_DB = -18.0        # the core, seen between the units at grazing
UNIT = 3.0                      # concrete cubes, 3 m (27 m^3, 65 t)
PITCH = 4.0                     # along and up the slope
JITTER = 0.3                    # placement tolerance, as a fraction of the pitch
UNIT_DIFFUSE_DB = -15.0         # smooth concrete: the faces do the talking
GRAIN_DB = 3.0                  # a grain: a peak of the profile along the wall this prominent


def _ex21():
    path = Path(__file__).resolve().parent / "21_long_range_300m.py"
    spec = importlib.util.spec_from_file_location("_ex21", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_EX = _ex21()
S = _EX.FAR / 300.0     # the scene was laid out for the 120 kHz head's 300 m swath; scale with it
WALL_Y, WALL_FROM, WALL_TO, AHEAD_X = WALL_Y * S, WALL_FROM * S, WALL_TO * S, AHEAD_X * S


def place(vertices: torch.Tensor, yaw_deg: float, position) -> torch.Tensor:
    """Body-frame points into the world, as mesh_target places its mesh."""
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    rot = torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=vertices.dtype)
    return vertices @ rot.T + torch.tensor(position, dtype=vertices.dtype)


def random_rotations(n: int, generator: torch.Generator) -> torch.Tensor:
    """``[n, 3, 3]`` rotations uniform over orientations (unit quaternions)."""
    q = torch.randn(n, 4, generator=generator)
    q = q / q.norm(dim=-1, keepdim=True)
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
        torch.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
        torch.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
    ], dim=-2)


def armour_layer(length: float, height: float, slope_deg: float, *, water_depth: float,
                 unit: float, pitch: float, generator: torch.Generator):
    """Cubes at random orientations on the underwater slope, one mesh.

    The mound's frame: along ``x``, the toe on ``z = 0`` rising to ``-height``,
    the face toward ``+y``.  Units sit a half-diagonal off the face, from just
    above the toe to the waterline, at ``pitch`` along and up the slope.
    """
    a = math.radians(slope_deg)
    normal = torch.tensor([0.0, math.sin(a), -math.cos(a)])        # off the face
    up = torch.tensor([0.0, -math.cos(a), -math.sin(a)])           # up the slope
    wet = water_depth / math.sin(a)                                # slope length under water
    s_up = torch.arange(0.6 * pitch, wet - 0.4 * unit, pitch)
    xs = torch.arange(-length / 2 + pitch / 2, length / 2, pitch)
    X, S = torch.meshgrid(xs, s_up, indexing="ij")
    X = X + JITTER * pitch * (2 * torch.rand(X.shape, generator=generator) - 1)
    S = S + JITTER * pitch * (2 * torch.rand(S.shape, generator=generator) - 1)
    centres = (torch.stack([X, torch.zeros_like(X), torch.zeros_like(X)], -1)
               + S.unsqueeze(-1) * up + 0.5 * unit * math.sqrt(3.0) * 0.9 * normal).reshape(-1, 3)
    n = int(centres.shape[0])
    rot = random_rotations(n, generator)
    v0, f0 = box_mesh((unit, unit, unit))
    verts = torch.einsum("nij,vj->nvi", rot, v0) + centres.unsqueeze(1)    # [n, 8, 3]
    faces = f0.unsqueeze(0) + 8 * torch.arange(n).view(-1, 1, 1)
    return verts.reshape(-1, 3), faces.reshape(-1, 3), n


def main() -> int:
    setup(double=False)
    banner("25 -- a rubble-mound breakwater: grains of rice")
    ex = _EX
    ex15 = ex._ex15()
    C = ex.C
    rx = ex.horizontal_array()
    scene, bottom, sea, sediment = ex.build_scene(rx, learnable=False)
    print(f"  {ex.FREQ_KHZ:.0f} kHz, {ex.N_RX} x {ex.N_TX}, {2 * ex.SECTOR_DEG:.0f} deg "
          f"swath to {ex.FAR:.0f} m; AUV at {ex.AUV_DEPTH:.0f} m in {ex.WATER_DEPTH:.0f} m")

    # ---- 21's boat, at rest ----------------------------------------------- #
    b = math.radians(ex.BOAT_BEARING_DEG)
    head = (ex.BOAT_RANGE * math.cos(b), ex.BOAT_RANGE * math.sin(b), 0.0)
    girth = ex.HULL_BEAM + 2.0 * ex.HULL_DRAUGHT
    verts, faces = boat_hull_mesh(ex.HULL_LENGTH, ex.HULL_BEAM, ex.HULL_DRAUGHT,
                                  n_long=int(round(110 * ex.HULL_LENGTH / 12.0)),
                                  n_around=int(round(34 * girth / (3.2 + 2.0 * 4.0))))

    def make_boat():
        return mesh_target(verts, faces, position=head, yaw=ex.BOAT_HEADING_DEG, n_patches=6,
                           sound_speed=C, diffuse_db=ex.DIFFUSE_DB, learnable=False,
                           learnable_shape=False, facet_chunk=4096, checkpoint=False)

    # ---- the walls, in a frame along x with the water on +y ---------------- #
    height = ex.WATER_DEPTH + WALL_ABOVE_WATER
    length = WALL_TO - WALL_FROM
    c_verts, c_faces = seawall_mesh(length, height, slope_deg=90.0, n_along=150, n_up=4)
    m_verts, m_faces = seawall_mesh(length, height, slope_deg=MOUND_SLOPE_DEG,
                                    n_along=150, n_up=8)
    run = height / math.tan(math.radians(MOUND_SLOPE_DEG))          # crest set-back
    wet_run = ex.WATER_DEPTH / math.tan(math.radians(MOUND_SLOPE_DEG))
    a_verts, a_faces, n_units = armour_layer(length, height, MOUND_SLOPE_DEG,
                                             water_depth=ex.WATER_DEPTH, unit=UNIT,
                                             pitch=PITCH,
                                             generator=torch.Generator().manual_seed(5))
    n_along = int(round(length / PITCH))
    print(f"  the mound: {length:.0f} m at {MOUND_SLOPE_DEG:.0f} deg, {run:.0f} m of set-back, "
          f"{wet_run:.0f} m of it under water; {n_units} cubes of {UNIT:.0f} m at {PITCH:.0f} m "
          f"pitch, {n_units // n_along} rows")

    def wall_targets(kind, yaw, pos):
        if kind == "caisson":
            body = mesh_target(c_verts, c_faces, position=pos, yaw=yaw, n_patches=150,
                               split_axis=0, sound_speed=C, diffuse_db=CAISSON_DIFFUSE_DB,
                               learnable=False, facet_chunk=4096, checkpoint=False)
            return [body], [(place(c_verts, yaw, pos), c_faces)]
        mound = mesh_target(m_verts, m_faces, position=pos, yaw=yaw, n_patches=150,
                            split_axis=0, sound_speed=C, diffuse_db=MOUND_DIFFUSE_DB,
                            learnable=False, facet_chunk=4096, checkpoint=False)
        armour = mesh_target(a_verts, a_faces, position=pos, yaw=yaw, n_patches=n_along,
                             split_axis=0, sound_speed=C, diffuse_db=UNIT_DIFFUSE_DB,
                             learnable=False, facet_chunk=4096, checkpoint=False)
        return [mound, armour], [(place(m_verts, yaw, pos), m_faces)]

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
    beam_deg = ex.beam_3db_deg(ex.N_RX, shading)

    def echo(target):
        return target_arrivals(
            scene, target, dirs, return_leg="eigenray", n_rx_rays=2000,
            rx_half_angle_deg=45.0, tx_weights=w_tx, tx_pattern=ex.transmit_pattern,
            rx_pattern=rx_beam, max_arrivals_per_leg=24,
            generator=torch.Generator().manual_seed(ex.SEED))

    def ping(*, targets, occluders):
        result = trace(scene, dirs)
        rev = reverberation_arrivals(
            result, dirs, scene.freqs_khz, scattering=seabed,
            solid_angle_per_ray=solid, ray_weights=w_tx * rx_beam(-dirs),
            boundary="both", surface=sea, bottom=scene.bottom, max_arrivals=ex.PATCHES,
            occluders=occluders, generator=torch.Generator().manual_seed(ex.SEED + 1))
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
        return cart, gx, gy, sum(p.n_arrivals for p in parts[1:])

    pictures, bands, grains = {}, {}, {}
    for name in SCENARIOS:
        banner(f"scenario: {name}")
        if name == "ahead":
            yaw, pos = 90.0, (AHEAD_X, 0.0, ex.WATER_DEPTH)       # +y (water) -> -x
            targets, occl = wall_targets("rubble", yaw, pos)
            print(f"  the mound across the picture, its toe {AHEAD_X:.0f} m ahead, crest at "
                  f"{AHEAD_X + run:.0f} m; no boat")
        else:
            yaw, pos = 0.0, (0.5 * (WALL_FROM + WALL_TO), -WALL_Y, ex.WATER_DEPTH)
            targets, occl = wall_targets(name, yaw, pos)
            targets = [make_boat()] + targets
            print(f"  the {name} alongside, its toe {WALL_Y:.0f} m to starboard"
                  + (f", crest at {WALL_Y + run:.0f} m" if name == "rubble" else "")
                  + "; 21's boat at rest")
        with torch.no_grad(), timed("  ping"):
            cart, gx, gy, n_echo = ping(targets=targets, occluders=occl)
        print(f"  {n_echo} target arrivals")
        X, Y = torch.meshgrid(gx, gy, indexing="xy")
        R = torch.hypot(X, Y)
        Bdeg = torch.rad2deg(torch.atan2(Y, X))
        infan = (R < 0.98 * ex.FAR) & (Bdeg.abs() < ex.SECTOR_DEG - 2.0)
        if name == "caisson":
            band = ((Y + WALL_Y).abs() < 6.0) & (X > 80.0 * S) & (X < 290.0 * S) & infan
        elif name == "rubble":
            band = (Y < -WALL_Y + 2.0) & (Y > -WALL_Y - wet_run) & (X > 80.0 * S) & (X < 290.0 * S) & infan
        else:
            band = (X > AHEAD_X - 2.0) & (X < AHEAD_X + wet_run) & infan
        db = 10.0 * torch.log10(cart.clamp_min(1e-30))
        # Over the band's area: its texture, the 95th percentile over the median.
        vals = db[band]
        a50, a95 = float(vals.median()), float(vals.quantile(0.95))
        # And as the eye reads the wall: a profile ALONG it of the band's
        # brightest cell across it.  Grains are the profile's peaks, GRAIN_DB
        # prominent over their surroundings; a grain's length is the peak's
        # width at half its prominence, along the wall.
        masked = db.masked_fill(~band, float("-inf"))
        prof = masked.max(dim=1 if name == "ahead" else 0).values     # per y ahead, per x alongside
        prof = prof[torch.isfinite(prof)].numpy()
        peaks, props = find_peaks(prof, prominence=GRAIN_DB, width=1, rel_height=0.5)
        n_grains = int(peaks.shape[0])
        runs = props["widths"] * pixel_m if n_grains else np.zeros(1)
        wall_m = float(prof.shape[0]) * pixel_m
        per_100 = 100.0 * n_grains / wall_m
        print(f"  the band: median {a50:+.1f} dB, 95th percentile {a95:+.1f} dB -- {a95 - a50:.1f} dB "
              f"of texture; along the wall ({wall_m:.0f} m of it in the picture) {n_grains} grains "
              f"{GRAIN_DB:.0f} dB prominent ({per_100:.1f} per 100 m), median length "
              f"{float(np.median(runs)):.1f} m along the wall, the longest {float(runs.max()):.1f} m")
        pictures[name] = (cart, db)
        bands[name] = (a50, a95)
        grains[name] = (n_grains, per_100, float(np.median(runs)))

    ok = True
    if "caisson" in pictures and "rubble" in pictures:
        c_con = bands["caisson"][1] - bands["caisson"][0]
        r_con = bands["rubble"][1] - bands["rubble"][0]
        ok &= check("the rubble's band is grains, not a band: its contrast exceeds the caisson's",
                    r_con > c_con + 4.0, f"{r_con:.1f} dB against {c_con:.1f} dB")
    if "ahead" in pictures:
        n_g, per_100, ext = grains["ahead"]
        beam_m = math.radians(beam_deg) * (AHEAD_X + 0.5 * wet_run)   # at the band's range
        ok &= check("ahead, the grains are the beam's width long and there are many of them",
                    0.4 * beam_m < ext < 2.0 * beam_m and per_100 >= 5.0,
                    f"median grain {ext:.1f} m along the wall against a {beam_m:.1f} m beam; "
                    f"{per_100:.1f} per 100 m")

    # ---- the figure --------------------------------------------------------- #
    names = [n for n in ("caisson", "rubble", "ahead") if n in pictures]
    top = max(float(p[1].max()) for p in pictures.values())
    ext = [float(gx.min()), float(gx.max()), float(gy.min()), float(gy.max())]
    zoom = {"caisson": (60 * S, 300 * S, -175 * S, -85 * S), "rubble": (60 * S, 300 * S, -175 * S, -85 * S),
            "ahead": (60 * S, 160 * S, -120 * S, 120 * S)}
    fig, axes = plt.subplots(2, len(names), figsize=(6.3 * len(names), 12),
                             squeeze=False)
    for j, name in enumerate(names):
        for i in range(2):
            ax = axes[i, j]
            im = ax.imshow(pictures[name][1].numpy(), origin="lower", extent=ext,
                           vmin=ex.THRESHOLD_DB, vmax=top, cmap="inferno", aspect="equal")
            if i == 1:
                x0, x1, y0, y1 = zoom[name]
                ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
                ax.set_title(f"{name}: the band, close up")
            else:
                ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
                ax.set_title(CAPTION[name])
            ax.set_xlabel("forward (m)"); ax.set_ylabel("across (m)")
    fig.colorbar(im, ax=axes[:, -1], fraction=0.03, label="dB re the background at that range")
    fig.suptitle(f"{ex.FREQ_KHZ:.0f} kHz FLS, {2 * ex.SECTOR_DEG:.0f} deg to {ex.FAR:.0f} m, "
                 f"median TVG floored at +{ex.THRESHOLD_DB:.0f} dB: a caisson face and a "
                 f"rubble mound with {UNIT:.0f} m armour cubes")
    save(fig, f"25_rubble_breakwater{ex.TAG}.png")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
