"""Inverse problem 4: localise a source in 3-D from arrival structure.

A 10 km x 10 km x 200 m continental-shelf scene.  The source position is the
only free parameter, and it is recovered from the energy-time curves on an
L-shaped array, starting roughly a kilometre away from the truth.

**Why the array has to be L-shaped.**  Absolute arrival time fixes the *range*
to the array, and the relative timing of the surface- and bottom-reflected
multipath fixes the *depth* -- a vertical aperture is what resolves depth,
because the reflected paths differ from the direct one by an amount that depends
on how deep the source is.  Neither constrains *bearing*: a vertical array alone
sees the same arrival pattern for any source on a circle around it.  Adding a
horizontal leg breaks that azimuthal symmetry, so the vertical leg supplies
range and depth and the horizontal leg supplies bearing.

**Why the kernels are annealed.**  A 1 km position error puts the predicted
arrivals about 0.7 s away from the measured ones, against a 4 ms kernel at
convergence -- no overlap, no gradient.  The fit opens at 250 ms.

Acceptance criterion: converge to within 50 m of the true source.
"""

from __future__ import annotations

import torch

from _common import banner, check, save, setup, timed
from hydropt import (
    ConstantLoss, FlatHeight, PiecewiseLinearProfile, Scene, make_time_grid,
    octave_bands, spherical_fan,
)
from hydropt.inverse import fit
from hydropt.plot import plot_etc, plot_fit_history

WATER_DEPTH = 200.0
TRUE_SOURCE = (3200.0, 2600.0, 75.0)
INITIAL_SOURCE = (3900.0, 3300.0, 130.0)  # ~1.1 km away in 3-D
ARRAY_X, ARRAY_Y = 8000.0, 5000.0

SIGMA_D, SIGMA_T = 90.0, 4.0e-3


def l_array() -> torch.Tensor:
    """Vertical leg (range + depth) plus a horizontal leg (bearing)."""
    z = torch.linspace(40.0, 180.0, 5)
    vertical = torch.stack((torch.full_like(z, ARRAY_X), torch.full_like(z, ARRAY_Y), z), -1)
    y = torch.linspace(ARRAY_Y - 1200.0, ARRAY_Y + 1200.0, 5)
    horizontal = torch.stack((torch.full_like(y, ARRAY_X), y, torch.full_like(y, 100.0)), -1)
    return torch.cat((vertical, horizontal), dim=0)


def build_scene(source, *, learn_source: bool) -> Scene:
    return Scene(
        field=PiecewiseLinearProfile([0.0, 50.0, WATER_DEPTH], [1512.0, 1500.0, 1506.0],
                                     learnable=False),
        bottom=FlatHeight(WATER_DEPTH),
        surface=FlatHeight(0.0),
        source=source,
        receivers=l_array(),
        surface_loss=ConstantLoss(0.5, learnable=False),
        bottom_loss=ConstantLoss(3.5, learnable=False),
        freqs_khz=octave_bands(0.3, 2),
        step_size=30.0,
        n_steps=320,
        domain=(-500.0, 10_500.0, -500.0, 10_500.0),
        learn_source=learn_source,
    )


def position_error(scene: Scene) -> float:
    truth = torch.tensor(TRUE_SOURCE, dtype=scene.source.dtype)
    return float((scene.source_position().detach() - truth).norm())


def main() -> int:
    setup()
    banner("05 -- inverse: localise a source in a 10 km x 10 km x 200 m shelf")

    # The fan has to be wide in azimuth: the source may start on the wrong
    # bearing, and rays that never go towards the array carry no gradient.
    directions = spherical_fan(26, 30, elev_range_deg=(-26.0, 26.0),
                               azim_range_deg=(0.0, 360.0))
    grid = make_time_grid(2.4, 5.2, 500)

    truth = build_scene(TRUE_SOURCE, learn_source=False)
    with torch.no_grad():
        target = truth.render(directions, grid, sigma_d=SIGMA_D, sigma_t=SIGMA_T,
                              ray_chunk=200)

    scene = build_scene(INITIAL_SOURCE, learn_source=True)
    initial_error = position_error(scene)
    print(f"  true source    ({TRUE_SOURCE[0]:.0f}, {TRUE_SOURCE[1]:.0f}, "
          f"{TRUE_SOURCE[2]:.0f}) m")
    print(f"  initial guess  ({INITIAL_SOURCE[0]:.0f}, {INITIAL_SOURCE[1]:.0f}, "
          f"{INITIAL_SOURCE[2]:.0f}) m  -- {initial_error:.0f} m away")
    print(f"  {directions.shape[0]} rays, {scene.receivers.shape[0]} receivers "
          f"(5 vertical + 5 horizontal)")

    with torch.no_grad():
        initial_etc = scene.render(directions, grid, sigma_d=SIGMA_D, sigma_t=SIGMA_T,
                                   ray_chunk=200)

    def keep_in_water() -> None:
        scene.source.data[0].clamp_(0.0, 10_000.0)
        scene.source.data[1].clamp_(0.0, 10_000.0)
        scene.source.data[2].clamp_(5.0, WATER_DEPTH - 5.0)

    with timed("200 Adam steps"):
        history = fit(
            scene, target, directions, time_grid=grid,
            n_iters=200, lr=12.0,  # parameter units are metres
            sigma_d_schedule=(600.0, SIGMA_D),
            sigma_t_schedule=(2.5e-1, SIGMA_T),
            project=keep_in_water,
            ray_chunk_size=200,
            log_every=25,
            track={"error_m": lambda: position_error(scene)},
        )

    final = scene.source_position().detach()
    final_error = position_error(scene)
    print(f"\n  recovered source ({final[0]:.1f}, {final[1]:.1f}, {final[2]:.1f}) m")
    print(f"  position error {initial_error:.0f} -> {final_error:.1f} m")
    print(f"    horizontal {((final[:2] - torch.tensor(TRUE_SOURCE[:2])).norm()):.1f} m, "
          f"depth {abs(final[2] - TRUE_SOURCE[2]):.1f} m")

    with torch.no_grad():
        final_etc = scene.render(directions, grid, sigma_d=SIGMA_D, sigma_t=SIGMA_T,
                                 ray_chunk=200)

    labels = [f"vert z={z:.0f} m" for z in scene.receivers[:5, 2]] + \
             [f"horiz y={y:.0f} m" for y in scene.receivers[5:, 1]]
    save(plot_fit_history(history, param_names=["source"],
                          truth={"source": list(TRUE_SOURCE)},
                          title="Source localisation"), "05_history.png")
    save(plot_etc(initial_etc, grid, freqs_khz=scene.freqs_khz, compare=target,
                  compare_label="measured", receiver_labels=labels,
                  title="Before: initial guess vs measurement"), "05_etc_before.png")
    save(plot_etc(final_etc, grid, freqs_khz=scene.freqs_khz, compare=target,
                  compare_label="measured", receiver_labels=labels,
                  title="After: localised source vs measurement"), "05_etc_after.png")

    banner("acceptance")
    ok = check("source localised to within 50 m", final_error < 50.0,
               f"{final_error:.1f} m (from {initial_error:.0f} m)")
    ok &= check("loss decreased", history.loss[-1] < history.loss[0],
                f"{history.loss[0]:.3e} -> {history.loss[-1]:.3e}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
