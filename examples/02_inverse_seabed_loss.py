"""Inverse problem 1: recover hidden surface and seabed reflection losses.

A shallow-water shelf scene is rendered with known boundary losses to make a
synthetic "measurement".  The losses are then hidden behind a wrong initial
guess and recovered by Adam on the log-domain ETC misfit.

The two parameters are only weakly separable: every path charges some
combination of ``n_surface * L_surface + n_bottom * L_bottom``, so the misfit has
a curved valley rather than a round basin.  What breaks the degeneracy is that
different rays reach the array with different bounce *ratios*, and that a
vertical array sees several of them at once.

Acceptance criterion: both losses to within 0.5 dB in under 100 Adam steps on CPU.
"""

from __future__ import annotations

import torch

from _common import banner, check, save, setup, timed
from hydropt import (
    ConstantLoss, FlatHeight, PiecewiseLinearProfile, Scene, make_time_grid,
    octave_bands, spherical_fan, vertical_line_array,
)
from hydropt.inverse import fit
from hydropt.plot import plot_etc, plot_fit_history

TRUE_SURFACE_LOSS = 0.8
TRUE_BOTTOM_LOSS = 4.5
GUESS_SURFACE_LOSS = 3.0
GUESS_BOTTOM_LOSS = 1.0

WATER_DEPTH = 200.0
SIGMA_D, SIGMA_T = 60.0, 2.0e-3


def build_scene(surface_loss: float, bottom_loss: float) -> Scene:
    return Scene(
        # The profile is known here and held fixed: this example isolates the
        # boundary losses.  Example 03 inverts the profile instead.
        field=PiecewiseLinearProfile([0.0, 60.0, WATER_DEPTH],
                                     [1510.0, 1500.0, 1505.0], learnable=False),
        bottom=FlatHeight(WATER_DEPTH),
        surface=FlatHeight(0.0),
        source=(0.0, 0.0, 50.0),
        receivers=vertical_line_array(4000.0, 0.0, 40.0, 180.0, 4),
        surface_loss=ConstantLoss(surface_loss),
        bottom_loss=ConstantLoss(bottom_loss),
        freqs_khz=octave_bands(0.2, 2),
        step_size=25.0,
        n_steps=260,
    )


def main() -> int:
    setup()
    banner("02 -- inverse: recover surface and seabed reflection loss")

    directions = spherical_fan(140, 1, elev_range_deg=(-30.0, 30.0),
                               azim_range_deg=(0.0, 0.0))
    grid = make_time_grid(2.6, 3.1, 260)

    truth = build_scene(TRUE_SURFACE_LOSS, TRUE_BOTTOM_LOSS)
    with torch.no_grad():
        target = truth.render(directions, grid, sigma_d=SIGMA_D, sigma_t=SIGMA_T)
    print(f"  synthetic measurement: {tuple(target.shape)} "
          f"[receivers, bands, time bins]")
    print(f"  true    L_surface = {TRUE_SURFACE_LOSS:.2f} dB, "
          f"L_bottom = {TRUE_BOTTOM_LOSS:.2f} dB")
    print(f"  initial L_surface = {GUESS_SURFACE_LOSS:.2f} dB, "
          f"L_bottom = {GUESS_BOTTOM_LOSS:.2f} dB")

    scene = build_scene(GUESS_SURFACE_LOSS, GUESS_BOTTOM_LOSS)
    with torch.no_grad():
        initial_etc = scene.render(directions, grid, sigma_d=SIGMA_D, sigma_t=SIGMA_T)

    def err_surface() -> float:
        return abs(scene.surface_loss.loss_db.item() - TRUE_SURFACE_LOSS)

    def err_bottom() -> float:
        return abs(scene.bottom_loss.loss_db.item() - TRUE_BOTTOM_LOSS)

    with timed("90 Adam steps"):
        history = fit(
            scene, target, directions, time_grid=grid,
            n_iters=90, lr=1.0,
            # No annealing needed: the losses do not move the rays at all, so
            # the geometry -- and therefore the overlap with the measurement --
            # is already correct at iteration zero.
            sigma_d_schedule=SIGMA_D, sigma_t_schedule=SIGMA_T,
            log_every=15,
            track={"err_surface_dB": err_surface, "err_bottom_dB": err_bottom},
        )

    fitted_s = scene.surface_loss.loss_db.item()
    fitted_b = scene.bottom_loss.loss_db.item()
    print(f"\n  recovered L_surface = {fitted_s:.4f} dB "
          f"(true {TRUE_SURFACE_LOSS}, error {err_surface():.4f})")
    print(f"  recovered L_bottom  = {fitted_b:.4f} dB "
          f"(true {TRUE_BOTTOM_LOSS}, error {err_bottom():.4f})")

    with torch.no_grad():
        final_etc = scene.render(directions, grid, sigma_d=SIGMA_D, sigma_t=SIGMA_T)

    save(plot_fit_history(history,
                          param_names=["surface_loss.loss_db", "bottom_loss.loss_db"],
                          truth={"surface_loss.loss_db": [TRUE_SURFACE_LOSS],
                                 "bottom_loss.loss_db": [TRUE_BOTTOM_LOSS]},
                          title="Seabed/surface loss inversion"),
         "02_history.png")
    save(plot_etc(initial_etc, grid, freqs_khz=scene.freqs_khz, compare=target,
                  compare_label="measured",
                  receiver_labels=[f"z = {z:.0f} m" for z in scene.receivers[:, 2]],
                  title="Before: initial guess vs measurement"), "02_etc_before.png")
    save(plot_etc(final_etc, grid, freqs_khz=scene.freqs_khz, compare=target,
                  compare_label="measured",
                  receiver_labels=[f"z = {z:.0f} m" for z in scene.receivers[:, 2]],
                  title="After: fitted vs measurement"), "02_etc_after.png")

    banner("acceptance")
    ok = check("L_surface within 0.5 dB", err_surface() < 0.5, f"{err_surface():.4f} dB")
    ok &= check("L_bottom within 0.5 dB", err_bottom() < 0.5, f"{err_bottom():.4f} dB")
    ok &= check("under 100 Adam steps", len(history) < 100, f"{len(history)} steps")
    ok &= check("loss decreased", history.loss[-1] < history.loss[0],
                f"{history.loss[0]:.3e} -> {history.loss[-1]:.3e}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
