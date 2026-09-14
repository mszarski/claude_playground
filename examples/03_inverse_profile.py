"""Inverse problem 2: recover a piecewise-linear c(z) from a vertical array.

This is ocean acoustic tomography in miniature.  A deep sound-channel profile is
recovered from the energy-time curves on a vertical line array 15 km away,
starting from a nearly featureless initial guess.

**Why deep water and not the shelf.**  Sound speed enters the measurement almost
entirely through arrival *time*, and the cleanest way to read a profile off
arrival times is with rays that turn -- refract back -- rather than bounce.  In a
downward-refracting shelf every ray reaches the array after many seabed
reflections, and a small change to the profile reshuffles which ray bounces
where; the misfit is then chaotic and gradient descent walks into whatever local
minimum is nearest.  Launched inside the channel here, not one of the 100 rays
touches a boundary, and the misfit is smooth enough to descend.

**Why the kernels have to be annealed.**  At 15 km the two profiles put their
arrivals 57 ms apart, while the final time kernel is 3 ms wide.  Started at 3 ms
the predicted and measured Gaussians would not overlap at all, every product
would be ~exp(-180) and the gradient would be exactly zero in floating point.
The fit therefore opens with an 80 ms kernel -- wide enough that the initial
guess still explains some of the measurement -- and tightens geometrically.

**What the data can and cannot constrain.**  The rays sample roughly 320-2650 m
of the water column; above and below that the profile is in the null space of
this experiment, and no amount of optimisation will recover it.  Those knots are
therefore held fixed -- in practice a CTD cast supplies them -- because leaving
them free does not merely fail to recover them, it actively corrupts the rest:
Adam normalises each parameter by its own gradient history, so a knot with
essentially no gradient still moves a full step every iteration.  The accuracy
check below is made over the sampled band, which is the honest statement of what
one array at one range measures.
"""

from __future__ import annotations

import torch

from _common import banner, check, save, setup, timed
from hydropt import (
    ConstantLoss, FlatHeight, PiecewiseLinearProfile, Scene, make_time_grid,
    octave_bands, spherical_fan, vertical_line_array,
)
from hydropt.inverse import fit
from hydropt.plot import plot_etc, plot_fit_history, plot_profile, plot_ray_projections

WATER_DEPTH = 3000.0
KNOT_DEPTHS = [0.0, 400.0, 1000.0, 1800.0, WATER_DEPTH]
TRUE_VALUES = [1540.0, 1508.0, 1496.0, 1502.0, 1520.0]  # channel axis near 1000 m
INITIAL_VALUES = [1540.0, 1516.0, 1504.0, 1495.0, 1520.0]  # channel too weak, axis too deep
RANGE = 15_000.0
SAMPLED_BAND = (320.0, 2650.0)  # depths the ray fan actually illuminates

# 15 km of range means a 1 m/s error in mean sound speed moves arrivals by
# 6.7 ms, so a 3 ms kernel would be demanding sub-m/s accuracy and the misfit
# becomes spiky.  10 ms is matched to what one array at one range resolves.
SIGMA_D, SIGMA_T = 120.0, 1.0e-2

# Knots the ray fan cannot see.  The rays turn between about 320 m and 2650 m,
# so the surface and near-bottom knots sit in this experiment's null space.
# Leaving them free is actively harmful: Adam normalises per parameter, so it
# gives a knot with a near-zero gradient the same step size as a well-determined
# one, and the surface knot wanders tens of m/s while dragging the 0-400 m
# segment the rays *do* sample.  In practice these come from a CTD cast.
FREE_KNOTS = [0.0, 1.0, 1.0, 1.0, 0.0]


def build_scene(values) -> Scene:
    return Scene(
        field=PiecewiseLinearProfile(KNOT_DEPTHS, values),
        bottom=FlatHeight(WATER_DEPTH),
        surface=FlatHeight(0.0),
        source=(0.0, 0.0, 1000.0),
        receivers=vertical_line_array(RANGE, 0.0, 600.0, 1600.0, 6),
        surface_loss=ConstantLoss(0.5, learnable=False),
        bottom_loss=ConstantLoss(5.0, learnable=False),
        freqs_khz=octave_bands(0.2, 2),
        step_size=30.0,
        n_steps=570,
    )


def rms_error(scene: Scene, truth: Scene, depth: torch.Tensor) -> float:
    with torch.no_grad():
        return float((scene.field.c_of_z(depth) - truth.field.c_of_z(depth)).pow(2).mean().sqrt())


def main() -> int:
    setup()
    banner("03 -- inverse: recover a sound-speed profile from a vertical array")

    directions = spherical_fan(100, 1, elev_range_deg=(-9.0, 9.0),
                               azim_range_deg=(0.0, 0.0))
    grid = make_time_grid(9.50, 10.50, 500)
    sampled = torch.linspace(*SAMPLED_BAND, 201)
    full = torch.linspace(0.0, WATER_DEPTH, 301)

    truth = build_scene(TRUE_VALUES)
    with torch.no_grad():
        target = truth.render(directions, grid, sigma_d=SIGMA_D, sigma_t=SIGMA_T)
        traced = truth.trace(directions)
    print(f"  {directions.shape[0]} rays, {int(traced.n_surface.sum() + traced.n_bottom.sum())} "
          f"boundary reflections (pure refraction)")
    print(f"  rays illuminate depths {traced.pos[..., 2].min():.0f}-"
          f"{traced.pos[..., 2].max():.0f} m")

    scene = build_scene(INITIAL_VALUES)
    initial_rms = rms_error(scene, truth, sampled)
    print(f"  true    {TRUE_VALUES}")
    print(f"  initial {INITIAL_VALUES}")
    print(f"  initial RMS error over the sampled band: {initial_rms:.2f} m/s")

    with torch.no_grad():
        initial_etc = scene.render(directions, grid, sigma_d=SIGMA_D, sigma_t=SIGMA_T)

    free = torch.tensor(FREE_KNOTS, dtype=scene.field.values.dtype)
    scene.field.values.register_hook(lambda g, m=free: g * m)

    def smoothness() -> torch.Tensor:
        """Penalise curvature in the knot values.

        Even among the knots the rays do sample, one array at one range leaves a
        null space: different profiles produce the same arrival times for this
        ray set.  A curvature penalty picks the least contrived of them.  The
        The weight matters: at 1e-4 the penalty was ~14% of the converged data
        misfit and the recovered channel came out visibly flatter than the true
        one, which is regularisation bias rather than a resolution limit.  At
        2e-5 it is a few percent -- enough to suppress jagged null-space
        structure, not enough to iron out the channel.
        """
        v = scene.field.values
        return 2e-5 * (v[2:] - 2 * v[1:-1] + v[:-2]).pow(2).mean()

    def keep_physical() -> None:
        scene.field.values.clamp_(1450.0, 1600.0)

    with timed("240 Adam steps"):
        history = fit(
            scene, target, directions, time_grid=grid,
            # lr is in m/s per iteration; it decays so the fit can settle
            # rather than orbit the minimum at a fixed step size.
            n_iters=240, lr=0.35, lr_decay=0.05,
            sigma_d_schedule=SIGMA_D,
            sigma_t_schedule=(8.0e-2, SIGMA_T),
            target_sigma_t=SIGMA_T,
            regulariser=smoothness,
            project=keep_physical,
            log_every=20,
            track={"rms_c": lambda: rms_error(scene, truth, sampled)},
        )

    final_rms = rms_error(scene, truth, sampled)
    final_rms_full = rms_error(scene, truth, full)
    fitted = [round(v, 2) for v in scene.field.values.detach().tolist()]
    print(f"\n  recovered {fitted}")
    print(f"  RMS error over sampled band {SAMPLED_BAND[0]:.0f}-{SAMPLED_BAND[1]:.0f} m: "
          f"{initial_rms:.2f} -> {final_rms:.2f} m/s "
          f"({100 * (1 - final_rms / initial_rms):.1f}% reduction)")
    print(f"  RMS error over the full water column: {final_rms_full:.2f} m/s "
          f"(includes the unsampled surface and near-bottom layers)")

    with torch.no_grad():
        final_etc = scene.render(directions, grid, sigma_d=SIGMA_D, sigma_t=SIGMA_T)

    save(plot_profile(
        {"true": (truth.field.c_of_z(full).detach(), full),
         "initial guess": (build_scene(INITIAL_VALUES).field.c_of_z(full).detach(), full),
         "recovered": (scene.field.c_of_z(full).detach(), full)},
        title="Profile inversion (rays sample 320-2650 m)"), "03_profile.png")
    save(plot_ray_projections(traced, receivers=truth.receivers,
                              source=truth.source_position(), stride=4, max_rays=100,
                              title="Refracted paths, no boundary contact"), "03_rays.png")
    save(plot_fit_history(history, param_names=["field.values"],
                          truth={"field.values": TRUE_VALUES},
                          title="Profile inversion"), "03_history.png")
    save(plot_etc(initial_etc, grid, freqs_khz=scene.freqs_khz, compare=target,
                  compare_label="measured",
                  receiver_labels=[f"z = {z:.0f} m" for z in scene.receivers[:, 2]],
                  title="Before: initial guess vs measurement"), "03_etc_before.png")
    save(plot_etc(final_etc, grid, freqs_khz=scene.freqs_khz, compare=target,
                  compare_label="measured",
                  receiver_labels=[f"z = {z:.0f} m" for z in scene.receivers[:, 2]],
                  title="After: recovered profile vs measurement"), "03_etc_after.png")

    banner("acceptance")
    ok = check("profile RMS error over the sampled band reduced by at least 50%",
               final_rms < 0.50 * initial_rms,
               f"{initial_rms:.2f} -> {final_rms:.2f} m/s")
    ok &= check("loss decreased", history.loss[-1] < history.loss[0],
                f"{history.loss[0]:.3e} -> {history.loss[-1]:.3e}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
