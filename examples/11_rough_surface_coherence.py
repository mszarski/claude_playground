"""What a wind sea does to a specular reflection, and to a 100 kHz sonar's ghosts.

hydropt reflects specularly.  A height field bends the specular direction --
`examples/10` measures that -- but a specular model says nothing about the energy
a rough boundary scatters *out* of that direction, and above a few kHz that is
nearly all of it.

The Eckart correction is one formula: a Gaussian height distribution of RMS
`sigma` spreads the reflected phase by `Gamma = 2 k sigma sin(theta)` radians, and
averaging over that spread costs `(10/ln 10) Gamma^2` dB of coherent energy.  It
is quadratic in frequency, which is the whole story.

The consequence this example exists for: `examples/06` renders a 100 kHz
forward-looking sonar over a *flat* sea surface and flags its surface multipath as
"an optimistic bound on the ghosting it causes".  It is more than optimistic.  At
100 kHz the wavelength is 15 mm, and 2 m/s of wind -- barely a ripple -- raises
22 mm of RMS elevation, so the median surface-bounced path here loses 43 orders of
magnitude.

But not *every* path, and the exception is the interesting part.  `Gamma` is
proportional to `sin(theta)`, so a ray arriving at near-grazing incidence sees a
surface that is effectively flat along its own direction of travel and reflects
coherently however rough that surface is.  There is a cutoff angle, and it has a
closed form: the loss reaches 3 dB at `sin(theta) = 0.831 / (2 k sigma)`, which
for this sea at 100 kHz is 2.5 degrees.  Below that the ghost is real; above about
5 degrees it is gone.  A vehicle at 10 m depth looking at a target 40 m out has
its surface path at roughly 25 degrees, so for *that* geometry the ghost should
indeed not be there -- but the reason is the angle, not the frequency alone.

**Construction and assumptions.**

* *The closed form*: ``coherent_reflection_loss_db`` against
  ``(10 / ln 10) Gamma^2`` at 30 deg grazing and 1 kHz for ``Gamma`` from
  0.1 to 3, and its frequency scaling for a 10 m/s wind sea
  (``wind_sea_rms_height``).
* *The channel*: 50 m of water under a three-knot profile (1510 m/s at the
  surface, 1500 at 25 m, 1498 at the bottom), a pressure-release surface
  without loss and a sand bottom (``sediment_loss``); the source and one
  receiver at 25 m depth, 400 m apart; 600 rays in the vertical plane
  within +/-35 deg (``fan_2d``); 1 m steps, 700 of them, up to 20 bounces;
  bands 0.5, 2 and 8 kHz with 5 cm of chop, and 100 kHz for the winds
  0, 2, 5 and 10 m/s.
* *The measure*: ``roughness_weights`` (Eckart, from the recorded bounces
  and their grazing angles) applied as per-ray weights to ``splat_etc``
  (``sigma_d`` 1 m, ``sigma_t`` 0.2 ms); the 100 kHz survivors are read off
  the surface bounces' grazing angles.
* *Assumptions*: Gaussian surface heights of one RMS, uncorrelated between
  bounces; coherent loss only -- the energy scattered out of the specular
  direction is removed, not redistributed (``reverb.py`` is separate);
  the bottom is smooth.
* *To vary*: ``SURFACE_RMS`` and the winds; the cutoff angle
  (``cutoff_angle_deg``) is the number to watch when moving a sonar's
  geometry, since a surface path within a couple of degrees of grazing
  keeps its ghost at any frequency.

Acceptance criteria:
  * the loss matches `(10/ln 10) Gamma^2` and scales as frequency squared;
  * rays that never touched a boundary are left *exactly* alone;
  * in a multi-band shallow channel the high bands lose monotonically more;
  * at 100 kHz in a 2 m/s breeze the median surface path is annihilated, and
    every path that survives is within a few degrees of grazing -- which is what
    the closed-form cutoff angle predicts.
"""

from __future__ import annotations

import math

import torch

from _common import banner, check, save, setup, timed
from hydropt import (
    ConstantLoss, FlatHeight, IsoProfile, PiecewiseLinearProfile, Scene,
    coherent_reflection_loss_db, make_time_grid, rayleigh_roughness,
    roughness_weights, sediment_loss, splat_etc, trace, wind_sea_rms_height,
)
from hydropt.launch import fan_2d

C = 1500.0
DB_PER_GAMMA2 = 10.0 / math.log(10.0)
WATER_DEPTH = 50.0
SOURCE_DEPTH = 25.0
RX_RANGE = 400.0
BANDS = torch.tensor([0.5, 2.0, 8.0])
SURFACE_RMS = 0.05  # m, a light chop


def shallow_scene(freqs: torch.Tensor) -> Scene:
    return Scene(
        field=PiecewiseLinearProfile([0.0, 25.0, WATER_DEPTH],
                                     [1510.0, 1500.0, 1498.0], learnable=False),
        bottom=FlatHeight(WATER_DEPTH), surface=FlatHeight(0.0),
        source=(0.0, 0.0, SOURCE_DEPTH),
        receivers=torch.tensor([[RX_RANGE, 0.0, SOURCE_DEPTH]]),
        surface_loss=ConstantLoss(0.0, learnable=False, pressure_release=True),
        bottom_loss=sediment_loss("sand", learnable=False),
        freqs_khz=freqs, step_size=1.0, n_steps=700, max_bounces=20,
    )


def cutoff_angle_deg(rms_height: float, freq_khz: float,
                    loss_db: float = 3.0) -> float | None:
    """Grazing angle at which the coherent loss reaches ``loss_db``.

    From ``loss = (10/ln 10) Gamma^2`` and ``Gamma = 2 k sigma sin(theta)``:
    ``sin(theta) = sqrt(loss ln 10 / 10) / (2 k sigma)``.  ``None`` when even
    perpendicular incidence costs less than ``loss_db`` -- a boundary that smooth
    has no cutoff.
    """
    if rms_height <= 0.0:
        return None
    k = 2.0 * math.pi * freq_khz * 1.0e3 / C
    sin_theta = math.sqrt(loss_db / DB_PER_GAMMA2) / (2.0 * k * rms_height)
    return math.degrees(math.asin(sin_theta)) if sin_theta < 1.0 else None


def _surface_grazing(result, scene, weights):
    """Surface-bounce grazing angles (deg), and those of rays keeping >half."""
    from hydropt.tracer import bounce_events

    events = bounce_events(result, surface=scene.surface, bottom=scene.bottom)
    surf = ~events.is_bottom
    graze = torch.rad2deg(events.grazing[surf])
    keep = weights[events.ray[surf]] > 0.5
    return graze.tolist(), graze[keep].tolist()


def main() -> int:
    setup()
    banner("11 -- rough-surface coherence loss")

    # ---- the formula ------------------------------------------------------- #
    banner("the Eckart factor, against its closed form")
    print(f"  {'Gamma':>6s} {'sigma at 30 deg, 1 kHz':>24s} {'loss (dB)':>11s} "
          f"{'4.343 Gamma^2':>14s} {'energy left':>13s}")
    graze30 = torch.tensor([math.radians(30.0)])
    worst = 0.0
    for gamma in (0.1, 0.5, 1.0, 2.0, 3.0):
        k = 2 * math.pi * 1.0e3 / C
        sigma = gamma / (2 * k * math.sin(math.radians(30.0)))
        db = float(coherent_reflection_loss_db(graze30, sigma, 1.0))
        exact = DB_PER_GAMMA2 * gamma * gamma
        worst = max(worst, abs(db - exact))
        print(f"  {gamma:6.1f} {sigma * 1e3:21.2f} mm {db:11.4f} {exact:14.4f} "
              f"{10 ** (-db / 10):13.4e}")
    print(f"  worst absolute disagreement: {worst:.2e} dB")

    banner("quadratic in frequency -- why this cannot be a BoundaryLoss")
    print("  BoundaryLoss carries one scalar per ray, which keeps path memory at")
    print("  O(rays x steps).  Eckart loss is quadratic in frequency, so it is")
    print("  applied after the trace instead, from the bounces already recorded.")
    sigma = wind_sea_rms_height(10.0)
    print(f"\n  a 10 m/s wind sea, RMS elevation {sigma:.3f} m, at 30 deg grazing:")
    ratios = []
    base = float(coherent_reflection_loss_db(graze30, sigma, 0.5))
    for freq in (0.5, 1.0, 2.0, 4.0):
        g = float(rayleigh_roughness(graze30, sigma, freq))
        db = float(coherent_reflection_loss_db(graze30, sigma, freq))
        ratios.append(db / base)
        print(f"    {freq:4.1f} kHz: Gamma = {g:7.3f}, loss = {db:10.2f} dB, "
              f"energy left {10 ** (-db / 10):9.3e}   (x{db / base:6.2f})")
    print("  a single design frequency across 0.5-4 kHz would be wrong by x64.")

    # ---- end to end, multi-band ------------------------------------------- #
    banner(f"a {WATER_DEPTH:.0f} m channel with {SURFACE_RMS * 100:.0f} cm of chop")
    scene = shallow_scene(BANDS)
    directions = fan_2d(600, elev_range_deg=(-35.0, 35.0))
    with torch.no_grad(), timed("  trace"):
        result = trace(scene, directions)
    with torch.no_grad(), timed("  roughness weights"):
        weights = roughness_weights(result, scene.freqs_khz,
                                    surface_rms=SURFACE_RMS,
                                    surface=scene.surface, bottom=scene.bottom)
    never = (result.n_surface + result.n_bottom) == 0
    touched = ~never
    print(f"  {int(never.sum())} of {result.n_rays} rays never touched a boundary; "
          f"their weight is exactly 1: {bool((weights[never] == 1.0).all())}")

    grid = make_time_grid(RX_RANGE / 1520.0, 1.45 * RX_RANGE / 1490.0, 700)
    kw = dict(sigma_d=1.0, sigma_t=2e-4)
    with torch.no_grad():
        smooth = splat_etc(result, scene.receivers, grid, scene.freqs_khz, **kw)
        rough = splat_etc(result, scene.receivers, grid, scene.freqs_khz,
                          ray_weights=weights, **kw)
    print(f"\n  {'band':>8s} {'mean weight (bounced)':>22s} {'received energy':>16s}")
    band_ratios = []
    for i, freq in enumerate(scene.freqs_khz.tolist()):
        ratio = float(rough[0, i].sum() / smooth[0, i].sum())
        band_ratios.append(ratio)
        print(f"  {freq:6.1f} kHz {float(weights[touched, i].mean()):22.4f} "
              f"{10 * math.log10(ratio):+13.2f} dB")

    # ---- the 100 kHz result ------------------------------------------------ #
    banner("at 100 kHz: what happens to example 06's surface ghost")
    print(f"  wavelength at 100 kHz is {C / 1e5 * 1e3:.1f} mm")
    hf = shallow_scene(torch.tensor([100.0]))
    with torch.no_grad():
        hf_result = trace(hf, directions)
    surface_paths = hf_result.n_surface > 0

    print(f"\n  {'wind':>6s} {'RMS elev':>10s} {'cutoff':>8s}  "
          f"{'median surface-path weight':>26s}  {'>20 dB lost':>11s}")
    hundred_khz = {}
    for wind in (0.0, 2.0, 5.0, 10.0):
        rms = wind_sea_rms_height(wind) if wind > 0 else 0.0
        cutoff = cutoff_angle_deg(rms, 100.0)
        with torch.no_grad():
            w = roughness_weights(hf_result, hf.freqs_khz, surface_rms=rms,
                                  surface=hf.surface, bottom=hf.bottom)
        ws = w[surface_paths, 0]
        median = float(ws.median())
        gone = float((ws < 10 ** (-2.0)).double().mean())
        hundred_khz[wind] = (median, gone, float(ws.max()), cutoff)
        cut = "none" if cutoff is None else f"{cutoff:5.2f} d"
        print(f"  {wind:4.1f} m/s {rms * 1e3:8.2f} mm {cut:>8s}  "
              f"{median:26.3e}  {gone * 100:10.1f}%")

    # Which paths survive, and at what angle?
    rms2 = wind_sea_rms_height(2.0)
    with torch.no_grad():
        w2 = roughness_weights(hf_result, hf.freqs_khz, surface_rms=rms2,
                               surface=hf.surface, bottom=hf.bottom)
    graze_deg, survivor_graze = _surface_grazing(hf_result, hf, w2[:, 0])
    print(f"\n  surface bounces span {min(graze_deg):.2f} to {max(graze_deg):.2f} "
          f"deg grazing (median {float(torch.tensor(graze_deg).median()):.1f})")
    print(f"  the 3 dB cutoff for this sea is {cutoff_angle_deg(rms2, 100.0):.2f} deg")
    if survivor_graze:
        print(f"  every ray keeping over half its energy bounces at "
              f"{max(survivor_graze):.2f} deg or shallower")
    print("\n  Loss against grazing angle for this sea, at 100 kHz:")
    for g in (0.5, 1.0, 2.0, 5.0, 10.0, 30.0):
        db = float(coherent_reflection_loss_db(
            torch.tensor([math.radians(g)]), rms2, 100.0))
        print(f"    {g:5.1f} deg: {db:9.2f} dB  (energy left {10 ** (-db / 10):.3e})")

    print("\n  So example 06's flat surface does not merely overstate the ghost: at")
    print("  its geometry -- a vehicle at 10 m looking 40 m out, so a surface path")
    print("  near 25 deg -- there is no coherent surface return at all.  What the")
    print("  angle dependence adds is that a *near-grazing* surface path survives")
    print("  any sea state, so long-range shallow-water propagation keeps its")
    print("  surface bounces even at high frequency.")
    print("\n  And the energy is still there physically: it has been scattered out")
    print("  of the specular direction, and hydropt does not put it back.  reverb.py")
    print("  models boundary backscatter separately and the two are not coupled, so")
    print("  a scene with roughness loss is missing that energy, not redistributing")
    print("  it.")

    save(_plot(grid, smooth, rough, scene, weights, touched), "11_roughness.png")

    banner("acceptance")
    ok = check("loss matches (10/ln 10) Gamma^2", worst < 1e-9,
               f"worst {worst:.1e} dB")
    ok &= check("loss is quadratic in frequency",
                all(abs(r - f ** 2) < 1e-9
                    for r, f in zip(ratios, (1.0, 2.0, 4.0, 8.0))),
                " ".join(f"x{r:.0f}" for r in ratios))
    ok &= check("rays that never bounced are left exactly alone",
                bool((weights[never] == 1.0).all()), f"{int(never.sum())} rays")
    ok &= check("high bands lose monotonically more energy",
                band_ratios[0] > band_ratios[1] > band_ratios[2],
                " > ".join(f"{10 * math.log10(r):+.2f}" for r in band_ratios) + " dB")
    ok &= check("a calm sea costs nothing at all, and has no cutoff angle",
                hundred_khz[0.0][0] == 1.0 and hundred_khz[0.0][3] is None,
                f"median weight {hundred_khz[0.0][0]:.1f}")
    ok &= check("at 100 kHz a 2 m/s breeze annihilates the median surface path",
                hundred_khz[2.0][0] < 1e-20,
                f"median weight {hundred_khz[2.0][0]:.2e}")
    ok &= check("and takes over 20 dB off the great majority of them",
                hundred_khz[2.0][1] > 0.85,
                f"{hundred_khz[2.0][1] * 100:.1f}% of surface paths")
    ok &= check("but near-grazing paths survive, as the cutoff angle predicts",
                bool(survivor_graze)
                and max(survivor_graze) < 3.0 * cutoff_angle_deg(rms2, 100.0),
                f"survivors up to {max(survivor_graze):.2f} deg against a "
                f"{cutoff_angle_deg(rms2, 100.0):.2f} deg cutoff"
                if survivor_graze else "no survivors")
    return 0 if ok else 1


def _plot(grid, smooth, rough, scene, weights, touched):
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    ax = axes[0]
    graze = np.linspace(0.5, 89.5, 200)
    for freq in (0.5, 2.0, 8.0, 100.0):
        db = coherent_reflection_loss_db(
            torch.tensor(np.radians(graze)), SURFACE_RMS, freq).numpy()
        ax.semilogy(graze, np.maximum(db, 1e-4), lw=1.3, label=f"{freq:g} kHz")
    ax.axhline(4.3429, ls=":", c="k", lw=0.9, label=r"$\Gamma=1$ (4.34 dB)")
    ax.set_xlabel("grazing angle (deg)")
    ax.set_ylabel("coherent loss (dB)")
    ax.set_title(f"Eckart loss, {SURFACE_RMS * 100:.0f} cm RMS", fontsize=10)
    ax.grid(alpha=0.3, which="both", lw=0.4)
    ax.legend(fontsize=8)

    ax = axes[1]
    t = grid.detach().numpy() * 1e3
    for i, freq in enumerate(scene.freqs_khz.tolist()):
        s = smooth[0, i].detach().numpy()
        r = rough[0, i].detach().numpy()
        floor = s.max() * 1e-7
        line, = ax.plot(t, 10 * np.log10(np.maximum(s, floor)), lw=0.8, ls="--",
                        alpha=0.7)
        ax.plot(t, 10 * np.log10(np.maximum(r, floor)), lw=1.2,
                color=line.get_color(), label=f"{freq:g} kHz")
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("energy (dB)")
    ax.set_title("ETC: rough (solid) vs smooth (dashed)", fontsize=10)
    ax.grid(alpha=0.3, lw=0.4)
    ax.legend(fontsize=8)

    ax = axes[2]
    for i, freq in enumerate(scene.freqs_khz.tolist()):
        w = weights[touched, i].detach().numpy()
        ax.hist(np.maximum(w, 1e-6), bins=np.logspace(-6, 0, 40), histtype="step",
                lw=1.3, label=f"{freq:g} kHz")
    ax.set_xscale("log")
    ax.set_xlabel("coherent energy factor per ray")
    ax.set_ylabel("rays")
    ax.set_title("Surviving coherent energy,\nrays that touched a boundary",
                 fontsize=10)
    ax.grid(alpha=0.3, lw=0.4)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    raise SystemExit(main())
