"""A synthesised ocean, and what actually makes it three-dimensional.

Every other example builds its environment by hand, and every one of them is
range-independent: a depth-only sound speed over a flat or single-bump seabed.
A depth-only profile keeps a ray in its launch plane exactly -- that is a theorem,
and `tests/test_snell_2d_limit.py` pins it -- so those scenes are 3-D only in
their bookkeeping.

This one is generated instead, from three statistical specifications, and each
one breaks the launch plane by a different mechanism:

* **a wind sea** from `pierson_moskowitz_surface`, whose sloped facets reflect a
  ray out of plane;
* **a power-law seabed** from `fractal_bathymetry`, likewise;
* **an internal-wave sound-speed field** from `internal_wave_perturbation`, which
  needs no boundary at all: a horizontal sound-speed gradient refracts a ray
  sideways continuously, all along its path.

The experiment is to launch rays in the x-z plane only, at azimuth zero, and
measure how far out of it they get -- separately for each cause.  Zero is the
range-independent answer, so anything else is the 3-D physics doing something.

The refraction case has a closed form to check against: a horizontal gradient
``dc/dy`` bends a ray on a radius ``R = c / |dc/dy|``, so over a path ``L`` it
offsets by ``L^2 / 2R``.  It also turns out to be the *weakest* of the three by
two orders of magnitude -- sloped facets deflect hundreds of metres where
horizontal refraction manages a couple, which is worth knowing before reaching
for a 3-D sound-speed field to explain out-of-plane energy.

**Construction and assumptions.**

* *The base ocean*: a Munk profile with its axis at 1000 m in 2000 m of
  water, the source 300 m deep at the origin; constant losses (0.5 and
  6 dB); one band at 1 kHz; 20 m steps, 900 of them (18 km of path for a
  12 km range), up to 30 bounces.  Flat boundaries are the control.
* *The three generated fields*, all seeded from ``SEED`` and spanning 1.3
  times the range with the source 15 % in from the near edge:
  a wind sea for 12 m/s at eight nodes per peak wavelength
  (``pierson_moskowitz_surface``, RMS set by ``H_s / 4``); a fractal
  seabed of 40 m RMS relief and spectral exponent 3 on 200 m nodes
  (``fractal_bathymetry``); an internal-wave sound-speed perturbation
  from 12 m RMS of heave with a 2 km correlation length on 500 m x 100 m
  nodes (``internal_wave_perturbation``), the seabed and the field
  learnable.
* *The experiment*: 400 rays in the ``x-z`` plane within +/-14 deg
  (``fan_2d`` at azimuth zero), traced under each field alone and all
  together; the out-of-plane measure is ``|y|`` over live vertices.
* *Assumptions*: geometric acoustics at 1 kHz on a sea with 100 m
  wavelengths (a boundary bends the specular direction and nothing else;
  ``examples/11`` is the coherent loss); the internal-wave field is a
  frozen realisation; the check against ``L^2 / 2R`` uses the RMS
  gradient, so the ratio is checked to a factor of three.
* *To vary*: the wind, ``SEABED_RMS`` and ``RMS_DISPLACEMENT`` scale each
  mechanism; keep the grids' extent over the whole fan or a ray leaving a
  grid is clamped to its edge (flat).

Acceptance criteria:
  * the generated fields match the statistics they were specified by;
  * the depth-only control keeps every ray in its launch plane to machine
    precision;
  * each of the three mechanisms takes rays out of plane on its own;
  * the refraction case matches ``L^2 / 2R``, and boundary roughness dominates it;
  * the whole synthesised scene is differentiable in the generated fields.
"""

from __future__ import annotations

import math

import torch

from _common import banner, check, save, setup, timed
from hydropt import (
    ConstantLoss, FlatHeight, MunkProfile, Scene, fractal_bathymetry,
    internal_wave_perturbation, pierson_moskowitz_surface,
    significant_wave_height_pm, trace, wave_number_peak_pm,
)
from hydropt.launch import fan_2d

WIND = 12.0  # m/s at 19.5 m
WATER_DEPTH = 2000.0
SOURCE_DEPTH = 300.0
RANGE = 12_000.0
STEP = 20.0
N_STEPS = 900
N_RAYS = 400
SEED = 11

RMS_DISPLACEMENT = 12.0  # m of internal-wave heave
CORRELATION_LENGTH = 2_000.0  # m
SEABED_RMS = 40.0
SEABED_EXPONENT = 3.0


def gen(offset: int = 0) -> torch.Generator:
    return torch.Generator().manual_seed(SEED + offset)


def build_surface():
    # Sample the peak wavelength properly: eight nodes across it, and enough of
    # them to span the range the rays cover.
    kp = wave_number_peak_pm(WIND)
    dx = 2.0 * math.pi / kp / 8.0
    n = int(math.ceil(1.3 * RANGE / dx)) + 1
    return pierson_moskowitz_surface((n, n), (dx, dx), WIND, origin=(-0.15 * RANGE,
                                     -0.65 * n * dx / 2), generator=gen(1)), dx, n


def build_bottom():
    dx = 200.0
    n = int(math.ceil(1.3 * RANGE / dx)) + 1
    return fractal_bathymetry((n, n), (dx, dx), base_depth=WATER_DEPTH,
                              rms=SEABED_RMS, exponent=SEABED_EXPONENT,
                              origin=(-0.15 * RANGE, -0.5 * n * dx),
                              learnable=True, generator=gen(2)), dx, n


def build_field(base):
    dx = 500.0
    dz = 100.0
    nz = int(WATER_DEPTH / dz) + 1
    n = int(math.ceil(1.3 * RANGE / dx)) + 1
    return internal_wave_perturbation(
        base, (nz, n, n), (dx, dx, dz),
        rms_displacement=RMS_DISPLACEMENT, correlation_length=CORRELATION_LENGTH,
        origin=(-0.15 * RANGE, -0.5 * n * dx, 0.0),
        water_depth=WATER_DEPTH, learnable=True, generator=gen(3)), dx, nz


def make_scene(field, surface, bottom) -> Scene:
    return Scene(
        field=field, surface=surface, bottom=bottom,
        source=(0.0, 0.0, SOURCE_DEPTH),
        surface_loss=ConstantLoss(0.5, learnable=False),
        bottom_loss=ConstantLoss(6.0, learnable=False),
        freqs_khz=torch.tensor([1.0]),
        step_size=STEP, n_steps=N_STEPS, max_bounces=30,
    )


def out_of_plane(result) -> tuple[float, float]:
    """Max and RMS |y| over live vertices, in metres.

    Rays are launched in the x-z plane, so y is zero for any range-independent
    ocean and is a direct measure of the three-dimensional physics.
    """
    live = result.alive > 0
    y = result.pos[..., 1].detach().abs()[live]
    return float(y.max()), float((y * y).mean().sqrt())


def main() -> int:
    setup()
    banner("10 -- a synthesised ocean, and what makes it three-dimensional")

    base = MunkProfile(z1=1000.0, learnable=False)
    surface, surf_dx, surf_n = build_surface()
    bottom, bot_dx, bot_n = build_bottom()
    field, fld_dx, fld_nz = build_field(base)

    hs = significant_wave_height_pm(WIND)
    kp = wave_number_peak_pm(WIND)
    print(f"  wind {WIND:.0f} m/s -> H_s {hs:.2f} m, peak wavelength "
          f"{2 * math.pi / kp:.0f} m, sampled at {surf_dx:.1f} m "
          f"({2 * math.pi / kp / surf_dx:.0f} nodes per wave)")
    print(f"  surface grid {surf_n} x {surf_n}; realised RMS elevation "
          f"{float(surface.heights.std(unbiased=False)):.3f} m (H_s/4 = {hs / 4:.3f})")
    relief_rms = float((bottom.heights.detach() - WATER_DEPTH).std(unbiased=False))
    print(f"  seabed grid {bot_n} x {bot_n} at {bot_dx:.0f} m; relief RMS "
          f"{relief_rms:.2f} m, exponent {SEABED_EXPONENT}")
    dc = field.values.detach()
    print(f"  sound-speed field {fld_nz} x {field.shape[1]} x {field.shape[2]} at "
          f"{fld_dx:.0f} m; internal-wave heave {RMS_DISPLACEMENT:.0f} m RMS")
    print(f"  -> perturbation up to {float(dc.abs().max()):.2f} m/s, "
          f"RMS {float(dc.std(unbiased=False)):.3f} m/s")

    # Rays in the launch plane only: azimuth exactly zero.
    directions = fan_2d(N_RAYS, elev_range_deg=(-14.0, 14.0), azimuth_deg=0.0)
    assert float(directions[:, 1].abs().max()) == 0.0, "fan must be planar"

    flat_surface = FlatHeight(0.0)
    flat_bottom = FlatHeight(WATER_DEPTH)

    banner("out-of-plane deflection, by mechanism")
    print("  rays launched at azimuth 0; y is zero for any range-independent ocean")
    cases = {
        "depth-only control": (base, flat_surface, flat_bottom),
        "wind sea only": (base, surface, flat_bottom),
        "fractal seabed only": (base, flat_surface, bottom),
        "internal waves only": (field, flat_surface, flat_bottom),
        "everything": (field, surface, bottom),
    }
    results, stats = {}, {}
    for label, (f, s, b) in cases.items():
        with torch.no_grad(), timed(f"  {label}"):
            res = trace(make_scene(f, s, b), directions)
        results[label] = res
        stats[label] = out_of_plane(res)

    print(f"\n  {'case':22s} {'max |y|':>12s} {'RMS |y|':>12s}   bounces (surf/bot)")
    for label, res in results.items():
        mx, rms = stats[label]
        print(f"  {label:22s} {mx:12.4g} {rms:12.4g}   "
              f"{int(res.n_surface.sum()):5d} / {int(res.n_bottom.sum()):5d}")

    # The refraction case has a closed form, so it can be checked rather than
    # just observed.  A horizontal gradient dc/dy bends a ray on a radius
    # R = c / |dc/dy|, and over a path L that is a lateral offset of L^2 / 2R.
    banner("the refraction case against its closed form")
    dcdy = float(dc.std(unbiased=False)) / CORRELATION_LENGTH
    radius = 1500.0 / dcdy
    predicted = RANGE * RANGE / (2.0 * radius)
    measured = stats["internal waves only"][0]
    print(f"  RMS |dc/dy| ~ {dcdy:.3e} (m/s)/m over a {CORRELATION_LENGTH:.0f} m "
          f"correlation length")
    print(f"  radius of curvature c/|dc/dy| = {radius / 1e3:.3g} km")
    print(f"  predicted offset over {RANGE / 1e3:.0f} km: L^2/2R = {predicted:.2f} m")
    print(f"  measured max |y|:                          {measured:.2f} m")
    print(f"  ratio {measured / predicted:.2f}")
    print("\n  Worth noting how much weaker this is than either boundary: a sloped")
    print(f"  facet deflects hundreds of metres, horizontal refraction {measured:.0f} m.")
    print("  Horizontal refraction is a real 3-D effect and a small one; out-of-plane")
    print("  energy in shallow water comes overwhelmingly from rough boundaries.")

    banner("differentiability through the generated fields")
    scene = make_scene(field, surface, bottom)
    res = trace(scene, directions)
    res.tau.sum().backward()
    grads = {"internal-wave values": field.values, "seabed heights": bottom.heights}
    for name, param in grads.items():
        ok_g = param.grad is not None and float(param.grad.abs().sum()) > 0.0
        print(f"  d(travel time)/d({name}): "
              f"{'nonzero' if ok_g else 'MISSING'}"
              + (f", |grad| = {float(param.grad.abs().sum()):.3e}" if ok_g else ""))

    save(_plot(surface, bottom, field, results, base), "10_environment.png")

    banner("acceptance")
    ok = check("surface RMS matches H_s / 4",
               abs(float(surface.heights.std(unbiased=False)) - hs / 4) < 1e-9,
               f"{float(surface.heights.std(unbiased=False)):.4f} vs {hs / 4:.4f} m")
    ok &= check("seabed relief RMS matches the specification",
                abs(relief_rms - SEABED_RMS) < 1e-9, f"{relief_rms:.2f} m")
    ok &= check("depth-only control keeps every ray in its launch plane",
                stats["depth-only control"][0] < 1e-9,
                f"max |y| = {stats['depth-only control'][0]:.2e} m")
    for label in ("wind sea only", "fractal seabed only", "internal waves only"):
        ok &= check(f"{label} takes rays out of plane",
                    stats[label][0] > 1.0, f"max |y| = {stats[label][0]:.1f} m")
    ok &= check("refraction offset matches the L^2/2R closed form",
                0.3 < measured / predicted < 3.0,
                f"measured {measured:.2f} m vs predicted {predicted:.2f} m "
                f"(ratio {measured / predicted:.2f})")
    ok &= check("boundary roughness dominates horizontal refraction",
                stats["fractal seabed only"][0] > 10.0 * measured,
                f"{stats['fractal seabed only'][0]:.0f} m against {measured:.1f} m")
    ok &= check("gradients reach both generated fields",
                all(p.grad is not None and float(p.grad.abs().sum()) > 0.0
                    for p in grads.values()))
    return 0 if ok else 1


def _plot(surface, bottom, field, results, base):
    import matplotlib.pyplot as plt
    import numpy as np

    fig = plt.figure(figsize=(13, 8))
    gs = fig.add_gridspec(2, 3, hspace=0.32, wspace=0.28)

    ax = fig.add_subplot(gs[0, 0])
    h = -surface.heights.detach().numpy()
    m = ax.imshow(h[:120, :120], cmap="coolwarm", origin="lower",
                  vmin=-3 * h.std(), vmax=3 * h.std())
    ax.set_title(f"Wind sea, {WIND:.0f} m/s\n"
                 f"$H_s$ = {significant_wave_height_pm(WIND):.2f} m", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(m, ax=ax, label="elevation (m)")

    ax = fig.add_subplot(gs[0, 1])
    b = bottom.heights.detach().numpy()
    m = ax.imshow(b, cmap="terrain_r", origin="lower")
    ax.set_title(f"Power-law seabed\nRMS {SEABED_RMS:.0f} m, "
                 f"$\\gamma$ = {SEABED_EXPONENT}", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(m, ax=ax, label="depth (m)")

    ax = fig.add_subplot(gs[0, 2])
    v = field.values.detach().numpy()
    m = ax.imshow(v[:, v.shape[1] // 2, :], cmap="RdBu_r", origin="upper",
                  aspect="auto")
    ax.set_title("Internal-wave $\\delta c$, x-z slice\n"
                 f"{RMS_DISPLACEMENT:.0f} m heave", fontsize=9)
    ax.set_xlabel("x node"); ax.set_ylabel("depth node")
    fig.colorbar(m, ax=ax, label="$\\delta c$ (m/s)")

    ax = fig.add_subplot(gs[1, :2])
    for label, res in results.items():
        live = res.alive > 0
        x = res.pos[..., 0].detach()
        y = res.pos[..., 1].detach().abs()
        # Largest |y| reached by any ray, as a function of range.
        xs = torch.linspace(0.0, RANGE, 60)
        prof = []
        for lo, hi in zip(xs[:-1], xs[1:]):
            m2 = live & (x >= lo) & (x < hi)
            prof.append(float(y[m2].max()) if int(m2.sum()) else float("nan"))
        ax.semilogy(xs[:-1].numpy() / 1e3, np.maximum(np.array(prof), 1e-16),
                    lw=1.3, label=label)
    ax.set_xlabel("range (km)")
    ax.set_ylabel("max |y| reached (m)")
    ax.set_title("Out-of-plane deflection of rays launched at azimuth zero",
                 fontsize=10)
    ax.grid(alpha=0.3, which="both", lw=0.4)
    ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[1, 2])
    z = torch.linspace(0.0, WATER_DEPTH, 300)
    ax.plot(base.c_of_z(z).detach().numpy(), z.numpy(), lw=1.4, label="background")
    mid = field.values.shape[1] // 2
    for i in (mid, mid + 2):
        zc = torch.arange(field.values.shape[0]) * 100.0
        ax.plot((base.c_of_z(zc) + field.values[:, i, i]).detach().numpy(),
                zc.numpy(), lw=0.9, ls="--", alpha=0.8)
    ax.invert_yaxis()
    ax.set_xlabel("c (m/s)"); ax.set_ylabel("depth (m)")
    ax.set_title("Two perturbed columns\nvs the background", fontsize=9)
    ax.grid(alpha=0.3, lw=0.4)
    ax.legend(fontsize=8)
    return fig


if __name__ == "__main__":
    raise SystemExit(main())
