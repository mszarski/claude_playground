"""Forward model: a deep Munk sound channel over 50 km with an azimuthal fan.

Shows the signature behaviour of a deep-water channel: rays launched near the
sound-channel axis are refracted back towards it from both above and below and
never touch either boundary, so they propagate for tens of kilometres with no
reflection loss at all.  Rays launched steeply escape the channel and start
bouncing.

Acceptance criterion: 2,000 rays x 3,000 RK4 steps in under 30 s on CPU.
"""

from __future__ import annotations

import time

import torch

from _common import banner, check, save, setup, timed
from hydropt import (
    ConstantLoss, FlatHeight, MunkProfile, Scene, make_time_grid, octave_bands,
    spherical_fan, vertical_line_array,
)
from hydropt.plot import plot_etc, plot_profile, plot_ray_projections, plot_rays_3d

AXIS_DEPTH = 1300.0
WATER_DEPTH = 5000.0
SOURCE_DEPTH = 1000.0
RANGE = 50_000.0


def build_scene() -> Scene:
    return Scene(
        field=MunkProfile(c1=1500.0, z1=AXIS_DEPTH, B=1300.0, eps=7.37e-3),
        bottom=FlatHeight(WATER_DEPTH),
        surface=FlatHeight(0.0),
        source=(0.0, 0.0, SOURCE_DEPTH),
        receivers=vertical_line_array(RANGE, 0.0, 600.0, 2000.0, 6),
        surface_loss=ConstantLoss(0.5),
        bottom_loss=ConstantLoss(6.0),
        freqs_khz=octave_bands(0.5, 4),  # 0.5-4 kHz: high enough that Thorp absorption
        # separates the bands visibly over 50 km (0.03 to 0.5 dB/km)
        # 20 m steps x 3000 = 60 km of path, enough to cross 50 km of range even
        # along a steeply cycling ray.
        step_size=20.0,
        n_steps=3000,
        max_bounces=40,
    )


def main() -> int:
    setup()
    banner("01 -- forward model: deep Munk channel, 50 km, 3-D azimuthal fan")
    scene = build_scene()

    # 100 elevations x 20 azimuths = 2,000 rays.  The azimuth spread is narrow
    # because the receivers sit on the y = 0 plane; a full 360 deg fan would put
    # most of its rays where nothing is listening.
    # For a source at 1000 m under a 1300 m axis, the channel traps launch
    # angles inside about +/-14 deg; going out to 20 deg means part of the fan
    # escapes and starts reflecting, so the plots show both regimes.
    directions = spherical_fan(100, 20, elev_range_deg=(-20.0, 20.0),
                               azim_range_deg=(-10.0, 10.0))
    print(f"  {directions.shape[0]} rays x {scene.n_steps} RK4 steps "
          f"({scene.max_path_length / 1e3:.0f} km of path each)")

    t0 = time.perf_counter()
    with torch.no_grad():
        result = scene.trace(directions)
    elapsed = time.perf_counter() - t0
    print(f"  trace: {elapsed:.2f} s")

    reached = result.pos[..., 0].max().item()
    no_bounce = int(((result.n_surface + result.n_bottom) == 0).sum())
    print(f"  max range reached {reached / 1e3:.1f} km")
    print(f"  {no_bounce} of {directions.shape[0]} rays never touched a boundary "
          f"(fully refracted channel paths)")
    print(f"  bounces: surface {int(result.n_surface.sum())}, bottom {int(result.n_bottom.sum())}")

    grid = make_time_grid(RANGE / 1520.0, RANGE / 1470.0, 900)
    with torch.no_grad(), timed("render ETC"):
        etc = scene.render(directions, grid, sigma_d=120.0, sigma_t=4e-3,
                           ray_chunk=400)
    print(f"  ETC shape {tuple(etc.shape)} = [receivers, bands, time bins]")

    depth = torch.linspace(0.0, WATER_DEPTH, 400)
    save(plot_profile({"Munk": (scene.field.c_of_z(depth).detach(), depth)},
                      title="Munk profile (axis at 1300 m)"), "01_profile.png")
    save(plot_ray_projections(result, receivers=scene.receivers,
                              source=scene.source_position(), bottom=scene.bottom,
                              stride=6, max_rays=250,
                              title="Munk channel, 2000 rays"), "01_rays.png")
    save(plot_rays_3d(result, receivers=scene.receivers, source=scene.source_position(),
                      stride=8, max_rays=120,
                      title="Munk channel ray fan (3-D)"), "01_rays_3d.png")
    save(plot_etc(etc, grid, freqs_khz=scene.freqs_khz,
                  receiver_labels=[f"z = {z:.0f} m" for z in scene.receivers[:, 2]],
                  title=f"Energy-time curves at {RANGE / 1e3:.0f} km"), "01_etc.png")

    banner("acceptance")
    ok = check("2,000 rays x 3,000 steps under 30 s on CPU", elapsed < 30.0,
               f"{elapsed:.2f} s")
    ok &= check("energy arrives at the array", etc.sum().item() > 0)
    ok &= check("channel paths exist (some rays never bounce)", no_bounce > 0,
                f"{no_bounce} rays")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
