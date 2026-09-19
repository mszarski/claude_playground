"""Gaussian beams through a caustic: where 1/s^2 and the ray tube both fail.

A deep sound channel focuses energy into convergence zones, and at the focus the
geometric ray tube collapses to zero area.  `1/s^2` misses the focusing
altogether; the ray tube finds it but then divides by (almost) nothing, so
`hydropt.spreading` has to floor the tube area with `min_jacobian`.  That floor
is an admission, not a model -- the level at the caustic is whatever the floor
was set to.

Gaussian beams fix it properly.  The beam parameter is complex, so `det Q` cannot
vanish where `det Q1` does, and the amplitude is finite everywhere with no clamp
at all.  The bound `|det Q| >= beta^2` is structural, and this example checks it
on a bundle that really does cross a caustic.

Acceptance criteria:
  * the bundle really crosses caustics (signed tube area changes sign);
  * the geometric tube is sitting on its `min_jacobian` floor there, away from
    the source singularity, so its level at the focus is the floor's, not
    physics';
  * the Gaussian beam needs no floor at all, and respects |det Q| >= beta^2;
  * in a homogeneous medium the beam reproduces 1/(s^2 + beta^2) exactly.
"""

from __future__ import annotations

import math

import torch

from _common import banner, check, save, setup, timed
from hydropt import (
    ConstantLoss, FlatHeight, IsoProfile, MunkProfile, Scene, gaussian_beams,
    make_time_grid, octave_bands, splat_etc, spherical_spreading, structured_fan,
    suggest_beam_width, vertical_line_array,
)
from hydropt.spreading import ray_tube

AXIS_DEPTH = 1300.0
WATER_DEPTH = 5000.0
SOURCE_DEPTH = 1000.0
RANGE = 32_000.0
FREQ_KHZ = 0.5
N_ELEV = 40
MIN_JACOBIAN = 1e-3  # `ray_tube`'s default floor, needed here to locate it


def build_scene() -> Scene:
    return Scene(
        field=MunkProfile(c1=1500.0, z1=AXIS_DEPTH, B=1300.0, eps=7.37e-3),
        bottom=FlatHeight(WATER_DEPTH),
        surface=FlatHeight(0.0),
        source=(0.0, 0.0, SOURCE_DEPTH),
        receivers=vertical_line_array(RANGE, 0.0, 800.0, 1800.0, 5),
        surface_loss=ConstantLoss(0.5),
        bottom_loss=ConstantLoss(6.0),
        freqs_khz=octave_bands(FREQ_KHZ, 1),
        step_size=25.0,
        n_steps=1500,
        max_bounces=20,
    )


def homogeneous_check(beta: float) -> float:
    """`det Q = (s + i beta)^2` in a homogeneous medium -- checkable by hand."""
    scene = Scene(
        field=IsoProfile(1500.0, learnable=False),
        surface=FlatHeight(0.0), bottom=FlatHeight(20_000.0),
        surface_loss=ConstantLoss(0.0), bottom_loss=ConstantLoss(0.0),
        source=(0.0, 0.0, 1000.0), step_size=10.0, n_steps=200,
    )
    elev = torch.tensor([0.0, 0.08, -0.13])
    azim = torch.tensor([0.0, 0.4, 1.3])
    with torch.no_grad():
        beams = gaussian_beams(scene, elev, azim, beam_width=beta, freq_khz=FREQ_KHZ)
    s = beams.result.arclen
    exact = 1.0 / (s * s + beta * beta)
    return float(((beams.spreading - exact).abs() / exact).max())


def main() -> int:
    setup()
    banner("08 -- Gaussian beams: a finite amplitude at a caustic")
    scene = build_scene()

    beta = suggest_beam_width(FREQ_KHZ, wavelengths=8.0)
    print(f"  beta = {beta:.1f} m  (8 wavelengths at {FREQ_KHZ} kHz)")

    # A structured fan only because the geometric tube, which we compare against,
    # needs to know which rays are neighbours.  Gaussian beams do not.
    # 3 azimuths is the minimum a central-difference tube can use; the beams
    # would be happy with one ray.  Kept small because five of the beams' six
    # traces are forward-mode dual traces, which cost far more than a plain one.
    directions, elev, azim = structured_fan(N_ELEV, 3, elev_range_deg=(-12.0, 12.0),
                                            azim_range_deg=(-4.0, 4.0))
    print(f"  {directions.shape[0]} rays x {scene.n_steps} steps")

    with torch.no_grad(), timed("geometric tube (1 trace)"):
        result_geo = scene.trace(directions)
        tube = ray_tube(result_geo, elev, azim)
    # `structured_fan` reports its launch-angle *axes*; the beams broadcast them
    # into the same elevation-major per-ray order as `directions`.
    with torch.no_grad(), timed("Gaussian beams (6 traces)"):
        beams = gaussian_beams(scene, elev[:, None], azim[None, :],
                               beam_width=beta, freq_khz=FREQ_KHZ)

    result = beams.result
    live = result.alive > 0
    sph = spherical_spreading(result)

    banner("what each law says")
    # Where is the geometric tube on its floor?  `ray_tube` floors |J| at
    # `min_jacobian * s^2 * cos(e)`, so the test is exact, not a guess -- and it
    # is asked away from the source, where *every* point-source tube is singular
    # and a clamp would prove nothing.
    cos_e = torch.cos(elev)[:, None].expand(N_ELEV, 3).reshape(-1, 1).abs().clamp_min(1e-6)
    arclen = result.arclen.clamp_min(1e-9)
    on_floor = (tube.jacobian.abs() <= MIN_JACOBIAN * arclen**2 * cos_e) & live & tube.valid
    far = live & (arclen > 1000.0)
    n_caustic_rays = int((beams.caustics[:, -1] > 0).sum())
    det_beam = 1.0 / beams.spreading

    print(f"  caustics: {n_caustic_rays} of {directions.shape[0]} rays crossed "
          f"at least one")
    print(f"  geometric tube on its min_jacobian floor: {int(on_floor.sum())} "
          f"vertices, {int((on_floor & far).sum())} of them beyond 1 km from the "
          f"source")
    print(f"  tube / (1/s^2) beyond 1 km: max {float((tube.spreading / sph)[far].max()):.1f} "
          f"-- the floor caps it at {1.0 / MIN_JACOBIAN:.0f}")
    print(f"  Gaussian beam: min |det Q| {float(det_beam[live].min()):.3e} "
          f"(beta^2 = {beta * beta:.1f}), no floor applied")
    print(f"  max spreading: 1/s^2 {float(sph[live].max()):.3e}, "
          f"tube {float(tube.spreading[live].max()):.3e}, "
          f"beam {float(beams.spreading[live].max()):.3e}")
    # Worth reading rather than skipping: at 500 Hz over tens of km the single
    # beam is kilometres wide, far wider than the channel it is in.  That is the
    # formula's own answer, sqrt(c(s^2+beta^2)/(omega beta)), and it is why a
    # real Gaussian-beam field is a *sum* over many narrow beams rather than one
    # wide one.  It is the limit of what this example claims.
    print(f"  beam width: {float(beams.width[live].min()):.2f} m at the source "
          f"to {float(beams.width[live].max()) / 1e3:.2f} km at "
          f"{float(arclen[live].max()) / 1e3:.0f} km of path")

    banner("what it does to the received level")
    grid = make_time_grid(RANGE / 1520.0, RANGE / 1470.0, 700)
    etcs = {}
    for label, spread in (("1/s^2", None), ("ray tube", tube.spreading),
                          ("Gaussian beam", beams.spreading)):
        with torch.no_grad():
            etcs[label] = splat_etc(result, scene.receivers, grid, scene.freqs_khz,
                                    sigma_d=150.0, sigma_t=5e-3, ray_chunk=300,
                                    spreading=spread)
    ref = etcs["1/s^2"].sum(-1)
    for label in ("ray tube", "Gaussian beam"):
        d = 10 * torch.log10((etcs[label].sum(-1) + 1e-300) / (ref + 1e-300))
        print(f"  {label:>14s} vs 1/s^2: "
              + ", ".join(f"{float(v):+.1f} dB" for v in d[:, 0]))

    rel = homogeneous_check(beta)
    print(f"\n  homogeneous medium: max relative error against 1/(s^2+beta^2) "
          f"= {rel:.2e}")

    save(plot_spreading(result, tube, beams, beta), "08_spreading.png")

    from hydropt.plot import plot_etc
    save(plot_etc(etcs["Gaussian beam"], grid, freqs_khz=scene.freqs_khz,
                  compare=etcs["1/s^2"], compare_label="1/s^2",
                  receiver_labels=[f"z = {z:.0f} m" for z in scene.receivers[:, 2]],
                  title="Gaussian beams (solid) vs 1/s^2 (dashed)"),
         "08_etc.png")

    banner("acceptance")
    ok = check("the bundle really crosses caustics", n_caustic_rays > 0,
               f"{n_caustic_rays} of {directions.shape[0]} rays")
    ok &= check("the geometric tube is on its floor at them, not just at the source",
                int((on_floor & far).sum()) > 0,
                f"{int((on_floor & far).sum())} vertices beyond 1 km")
    ok &= check("Gaussian beam needs no clamp",
                float(det_beam[live].min()) >= beta * beta * (1 - 1e-9),
                f"min |det Q| = {float(det_beam[live].min()):.1f} >= {beta * beta:.1f}")
    ok &= check("beam spreading bounded by 1/beta^2",
                float(beams.spreading[live].max()) <= 1.0 / beta**2 * (1 + 1e-9))
    ok &= check("exact in a homogeneous medium", rel < 1e-10, f"{rel:.1e}")
    ok &= check("beams still find the focusing 1/s^2 misses",
                float((beams.spreading[live] / sph[live]).max()) > 3.0,
                f"up to {10 * math.log10(float((beams.spreading[live] / sph[live]).max())):+.1f} dB")
    return 0 if ok else 1


def plot_spreading(result, tube, beams, beta):
    """Spreading along the ray that focuses hardest, on a log axis."""
    import matplotlib.pyplot as plt

    live = result.alive > 0
    focus = int(tube.spreading.where(live, torch.zeros(())).max(dim=1).values.argmax())
    keep = live[focus]
    s = result.arclen[focus][keep].detach() / 1e3

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.semilogy(s, spherical_spreading(result)[focus][keep].detach(),
                label="$1/s^2$", lw=1.2)
    ax.semilogy(s, tube.spreading[focus][keep].detach(),
                label="geometric ray tube", lw=1.2)
    ax.semilogy(s, beams.spreading[focus][keep].detach(),
                label=f"Gaussian beam ($\\beta$ = {beta:.0f} m)", lw=1.8)
    ax.axhline(1.0 / beta**2, ls=":", c="k", lw=0.9,
               label="$1/\\beta^2$ (structural bound)")
    ax.set_xlabel("arclength along ray (km)")
    ax.set_ylabel("spreading factor  (1/m$^2$)")
    ax.set_title(f"Spreading along the most strongly focused ray (#{focus})")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    return fig


if __name__ == "__main__":
    raise SystemExit(main())
