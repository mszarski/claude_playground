"""Inverse problem 3: recover a seamount from a horizontal array's ETCs.

The seabed is a learnable bilinear height field.  A Gaussian seamount is hidden
in it, the fit starts from a flat bottom, and the only data are the energy-time
curves on a horizontal line array.

What carries the information is bottom-bounce *timing*: raising the seabed by
``dh`` under a bounce with grazing angle ``theta`` shortens that path by about
``2 dh sin(theta)``, so an 80 m seamount moves its arrivals by tens of
milliseconds.  For that gradient to exist at all, the crossing point of the ray
with the height field has to be differentiable in the node heights -- which is
exactly why :func:`hydropt.boundaries.find_crossing` refines its bisection
bracket with a Newton step instead of returning the bracket itself.

Acceptance criterion: reduce bathymetry RMS error by at least 80% from a flat
initial guess.
"""

from __future__ import annotations

import torch

from _common import banner, check, save, setup, timed
from hydropt import (
    BilinearHeightField, ConstantLoss, FlatHeight, PiecewiseLinearProfile,
    Scene, horizontal_line_array, make_time_grid, octave_bands, spherical_fan,
)
from hydropt.inverse import fit
from hydropt.plot import plot_bathymetry, plot_etc, plot_fit_history

# A deliberately coarse grid: 5 x 4 nodes over 8 km x 3 km.  The number of free
# parameters has to stay comparable to the number of independent arrivals the
# array actually resolves, or the problem is underdetermined however good the
# gradients are.
NX, NY = 5, 4
X0, Y0 = 0.0, -1500.0
DX, DY = 2000.0, 1000.0
FLAT_DEPTH = 200.0
SEAMOUNT_HEIGHT = 80.0
SEAMOUNT_XY = (4000.0, 0.0)
SEAMOUNT_RADIUS = 1800.0

# The measurement's resolution.  SIGMA_T is not a free knob: scanning the misfit
# against seamount amplitude shows a clean monotone bowl down to 15 ms and pure
# noise by 10 ms, because below that the kernel is asking for bathymetry
# accuracy this geometry does not carry.  Annealing past the smooth regime is
# what made an earlier version of this example diverge.
SIGMA_D, SIGMA_T = 180.0, 1.5e-2


def true_heights() -> torch.Tensor:
    """Flat seabed minus a Gaussian seamount (shallower = smaller depth)."""
    x = X0 + DX * torch.arange(NX)
    y = Y0 + DY * torch.arange(NY)
    gy, gx = torch.meshgrid(y, x, indexing="ij")
    r2 = (gx - SEAMOUNT_XY[0]) ** 2 + (gy - SEAMOUNT_XY[1]) ** 2
    return FLAT_DEPTH - SEAMOUNT_HEIGHT * torch.exp(-r2 / (2 * SEAMOUNT_RADIUS**2))


def build_scene(heights: torch.Tensor, *, learnable: bool) -> Scene:
    return Scene(
        field=PiecewiseLinearProfile([0.0, FLAT_DEPTH], [1505.0, 1500.0], learnable=False),
        bottom=BilinearHeightField(heights.clone(), origin=(X0, Y0),
                                   spacing=(DX, DY), learnable=learnable),
        surface=FlatHeight(0.0),
        source=(200.0, 0.0, 30.0),
        receivers=horizontal_line_array(7600.0, -900.0, 7600.0, 900.0, 60.0, 7),
        surface_loss=ConstantLoss(0.5, learnable=False),
        bottom_loss=ConstantLoss(3.5, learnable=False),
        freqs_khz=octave_bands(0.3, 2),
        step_size=30.0,
        n_steps=350,
        max_bounces=30,
    )


def rms(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).pow(2).mean().sqrt())


def main() -> int:
    setup()
    banner("04 -- inverse: recover a seamount height field from an HLA")

    truth_h = true_heights()
    flat_h = torch.full_like(truth_h, FLAT_DEPTH)

    # Azimuth spread matters here: a single vertical plane would only ever
    # sample one line across the seamount.
    directions = spherical_fan(46, 11, elev_range_deg=(-22.0, -2.0),
                               azim_range_deg=(-9.0, 9.0))
    grid = make_time_grid(4.9, 5.6, 340)

    truth = build_scene(truth_h, learnable=False)
    with torch.no_grad():
        target = truth.render(directions, grid, sigma_d=SIGMA_D, sigma_t=SIGMA_T)

    scene = build_scene(flat_h, learnable=True)
    initial_rms = rms(scene.bottom.heights.detach(), truth_h)
    print(f"  grid {NY} x {NX} nodes over "
          f"{(NX - 1) * DX / 1e3:.0f} km x {(NY - 1) * DY / 1e3:.0f} km")
    print(f"  seamount rises {SEAMOUNT_HEIGHT:.0f} m above a {FLAT_DEPTH:.0f} m seabed")
    print(f"  true depths  min {truth_h.min():.1f} m, max {truth_h.max():.1f} m")
    print(f"  initial RMS error {initial_rms:.2f} m (flat guess)")
    print(f"  {directions.shape[0]} rays, {scene.receivers.shape[0]} receivers")

    with torch.no_grad():
        initial_etc = scene.render(directions, grid, sigma_d=SIGMA_D, sigma_t=SIGMA_T)

    def keep_physical() -> None:
        # The seabed must stay below the array and above a plausible floor.
        scene.bottom.heights.clamp_(90.0, 320.0)

    def prior() -> torch.Tensor:
        """Smoothness, plus a light pull towards the flat prior.

        Nodes at the edges of the grid are barely touched by any ray, so the
        data say almost nothing about them.  Adam normalises per parameter, so
        it hands those unconstrained nodes the *same* step size as
        well-determined ones and they wander freely.  The Tikhonov term gives
        them somewhere to sit; the weight is small enough (a few percent of the
        data misfit at the true seabed) not to flatten the seamount itself.
        """
        h = scene.bottom.heights
        dev = h - FLAT_DEPTH
        curv_x = h[:, 2:] - 2 * h[:, 1:-1] + h[:, :-2]
        curv_y = h[2:, :] - 2 * h[1:-1, :] + h[:-2, :]
        return 2e-5 * dev.pow(2).mean() + 2e-5 * (curv_x.pow(2).mean() + curv_y.pow(2).mean())

    with timed("150 Adam steps"):
        history = fit(
            scene, target, directions, time_grid=grid,
            # Without lr_decay this fit reaches ~13 m RMS around iteration 75 and
            # then drifts back out to ~17 m: Adam keeps taking full-size steps
            # whatever the gradient, so it cannot settle into the basin it found.
            n_iters=150, lr=1.5, lr_decay=0.03,
            sigma_d_schedule=SIGMA_D,
            sigma_t_schedule=(4.0e-2, SIGMA_T),
            target_sigma_t=SIGMA_T,
            regulariser=prior,
            project=keep_physical,
            log_every=25,
            track={"rms_h": lambda: rms(scene.bottom.heights.detach(), truth_h)},
        )

    final_rms = rms(scene.bottom.heights.detach(), truth_h)
    reduction = 100.0 * (1.0 - final_rms / initial_rms)
    print(f"\n  bathymetry RMS error: {initial_rms:.2f} -> {final_rms:.2f} m "
          f"({reduction:.1f}% reduction)")
    print(f"  recovered depths min {scene.bottom.heights.min():.1f} m, "
          f"max {scene.bottom.heights.max():.1f} m")

    with torch.no_grad():
        final_etc = scene.render(directions, grid, sigma_d=SIGMA_D, sigma_t=SIGMA_T)

    extent = (X0 / 1e3, (X0 + (NX - 1) * DX) / 1e3,
              Y0 / 1e3, (Y0 + (NY - 1) * DY) / 1e3)
    save(plot_bathymetry({"true": truth_h, "initial (flat)": flat_h,
                          "recovered": scene.bottom.heights.detach()},
                         extent=extent, receivers=scene.receivers,
                         title="Seamount recovery"), "04_bathymetry.png")
    save(plot_fit_history(history, param_names=["bottom.heights"],
                          title="Bathymetry inversion"), "04_history.png")
    save(plot_etc(initial_etc, grid, freqs_khz=scene.freqs_khz, compare=target,
                  compare_label="measured",
                  receiver_labels=[f"y = {y:.0f} m" for y in scene.receivers[:, 1]],
                  title="Before: flat seabed vs measurement"), "04_etc_before.png")
    save(plot_etc(final_etc, grid, freqs_khz=scene.freqs_khz, compare=target,
                  compare_label="measured",
                  receiver_labels=[f"y = {y:.0f} m" for y in scene.receivers[:, 1]],
                  title="After: recovered seabed vs measurement"), "04_etc_after.png")

    banner("acceptance")
    ok = check("bathymetry RMS error reduced by at least 80%",
               final_rms < 0.20 * initial_rms,
               f"{initial_rms:.2f} -> {final_rms:.2f} m ({reduction:.1f}%)")
    ok &= check("loss decreased", history.loss[-1] < history.loss[0],
                f"{history.loss[0]:.3e} -> {history.loss[-1]:.3e}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
