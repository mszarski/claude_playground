"""Active forward-looking sonar: two-way echoes, beamformed into a bearing-range image.

This is the two stages of the active-sonar path exercised together.

**Stage 1 -- two-way propagation.**  A projector on the vehicle nose insonifies
a sector ahead; a target scatters; the echo returns to a receive array beside
the projector.  :func:`hydropt.active.render_echo` composes the two legs by
convolving their one-way responses, so the whole two-way problem costs two
renders rather than one per incident arrival.

**Stage 2 -- beamforming.**  An energy response cannot be beamformed, so the
coherent path instead extracts *arrivals* -- time, amplitude, direction, and
the phase each bounce imposed -- at the array phase centre, and propagates each
one across the aperture analytically as a plane wave.  That sidesteps the
reason naive coherent splatting fails: ray-fan discretisation puts each arrival
time out by microseconds, which at 100 kHz is whole cycles of phase, but the
*differential* phase across a 233 mm aperture is set by geometry and survives
intact.

The vehicle is at 10 m depth in 30 m of water with a 32-element half-wavelength
array at 100 kHz, looking into a 45 degree sector.

**Construction and assumptions.**

* *Sonar*: 100 kHz; a 32-element horizontal line array along ``y`` at
  half-wavelength spacing (233 mm aperture) on a vehicle at 10 m depth,
  the projector at the array's centre; a Gaussian projector directivity
  14 deg wide in azimuth and 8 deg in elevation (``transmit_shading``),
  applied as a per-ray weight so the sidelobes still illuminate.
* *Environment*: 30 m of water under a three-knot profile (1512 m/s at the
  surface, 1505 at the vehicle, 1503 at the bottom), both boundaries flat;
  the surface pressure-release with 2 dB per bounce (an optimistic stand-in
  for a rough sea at 15 mm wavelength), the bottom a Rayleigh sand
  (1900 kg/m^3, 1650 m/s, 0.8 dB per wavelength); 0.2 m steps, 700 of
  them, up to 4 bounces.
* *Targets*: two point scatterers (``TARGETS``): 0 dB at 35 m on bearing
  -18 deg, 12 m deep; -6 dB at 55 m on +10 deg, 14 m deep.
* *The picture*: a 14,000-ray Fibonacci cone over the 45 deg sector out,
  a return fan from the target back; stage 1 composes the two legs'
  energy responses (``render_echo``, ``sigma_d`` 0.45 m, ``sigma_t`` 30
  us); stage 2 extracts up to 10 arrivals a leg at the array's centre,
  composes them (``compose_arrivals``) and beamforms 541 Hamming-shaded
  beams across the sector (``beamform``) on a 2600-bin time grid.
* *Assumptions*: point targets with an isotropic target strength; specular
  boundaries (surface multipath is a bound on the ghosting a real sea
  gives); each arrival crosses the aperture as a plane wave.
* *To vary*: ``TARGETS`` and ``TS`` for other contacts; ``N_ELEMENTS`` sets
  the beam width; ``shading_window`` the sidelobes (the beam-pattern figure
  compares uniform, Hamming and Blackman).
"""

from __future__ import annotations

import math

import torch

from _common import banner, check, save, setup, timed
from hydropt import (
    ConstantLoss, FlatHeight, PiecewiseLinearProfile, RayleighBottomLoss, Scene,
    make_time_grid,
)
from hydropt.active import (
    PointTarget, _RelocatedScene, compose_arrivals, render_echo, return_fan,
)
from hydropt.beamform import (
    azimuth_steering, beamform, extract_arrivals, shading_window,
)
from hydropt.launch import fibonacci_cone
from hydropt.tracer import trace

C = 1500.0
FREQ_HZ = 100e3
LAMBDA = C / FREQ_HZ
N_ELEMENTS = 32
WATER_DEPTH = 30.0
VEHICLE_DEPTH = 10.0
SECTOR_DEG = 45.0

# Two targets ahead: one close on the port bow, one further out to starboard.
TARGETS = [
    {"bearing": -18.0, "range": 35.0, "depth": 12.0, "ts_db": 0.0},
    {"bearing": 10.0, "range": 55.0, "depth": 14.0, "ts_db": -6.0},
]


def receive_array() -> torch.Tensor:
    """32-element horizontal line array at half-wavelength spacing."""
    y = (torch.arange(N_ELEMENTS, dtype=torch.get_default_dtype())
         - (N_ELEMENTS - 1) / 2) * (LAMBDA / 2)
    return torch.stack((torch.zeros_like(y), y,
                        torch.full_like(y, VEHICLE_DEPTH)), dim=-1)


def build_scene(receivers: torch.Tensor) -> Scene:
    return Scene(
        field=PiecewiseLinearProfile([0.0, VEHICLE_DEPTH, WATER_DEPTH],
                                     [1512.0, 1505.0, 1503.0], learnable=False),
        bottom=FlatHeight(WATER_DEPTH),
        surface=FlatHeight(0.0),
        source=(0.0, 0.0, VEHICLE_DEPTH),
        receivers=receivers,
        # Pressure-release: every surface bounce inverts the echo.  Energy
        # rendering can ignore this; the beamformer cannot.
        # Pressure-release, plus a little loss.  At 100 kHz the sea surface is
        # very rough against a 15 mm wavelength and would scatter far more than
        # this; hydropt models specular reflection only, so treat surface
        # multipath here as an optimistic bound on the ghosting it causes.
        surface_loss=ConstantLoss(2.0, learnable=False, pressure_release=True),
        bottom_loss=RayleighBottomLoss(1900.0, 1650.0, 0.8, learnable=False),
        freqs_khz=torch.tensor([FREQ_HZ / 1e3]),
        step_size=0.2, n_steps=700, max_bounces=4,
    )


def target_position(spec: dict) -> tuple[float, float, float]:
    b = math.radians(spec["bearing"])
    return (spec["range"] * math.cos(b), spec["range"] * math.sin(b), spec["depth"])


def transmit_shading(directions: torch.Tensor) -> torch.Tensor:
    """Projector directivity, applied as a per-ray weight.

    Shading the fan is not the same as narrowing it: the sidelobes still
    illuminate, and a bright target off-axis still returns an echo.  That is
    exactly the effect a detection study needs to see.
    """
    az = torch.atan2(directions[:, 1], directions[:, 0])
    el = torch.asin(directions[:, 2].clamp(-1.0, 1.0))
    return (torch.exp(-0.5 * (az / math.radians(14.0)) ** 2)
            * torch.exp(-0.5 * (el / math.radians(8.0)) ** 2))


def main() -> int:
    setup()
    banner("06 -- active forward-looking sonar: two-way echoes and beamforming")

    elements = receive_array()
    centre = elements.mean(0)
    scene = build_scene(elements)
    aperture = float(elements[:, 1].max() - elements[:, 1].min())
    print(f"  {N_ELEMENTS} elements, {LAMBDA / 2 * 1000:.2f} mm spacing, "
          f"{aperture * 1000:.0f} mm aperture = {aperture / LAMBDA:.1f} wavelengths")
    print(f"  {FREQ_HZ / 1e3:.0f} kHz, {SECTOR_DEG:.0f} deg sector, "
          f"{WATER_DEPTH:.0f} m water, vehicle at {VEHICLE_DEPTH:.0f} m")

    tx_dirs = fibonacci_cone(14000, torch.tensor([1.0, 0.0, 0.0]), SECTOR_DEG)
    tx_w = transmit_shading(tx_dirs)

    # ---- Stage 1: two-way energy response ---------------------------------- #
    banner("stage 1 -- two-way echo (energy)")
    max_range = max(t["range"] for t in TARGETS) + 15.0
    echo_grid = make_time_grid(0.0, 2.0 * max_range / 1490.0, 2600)
    total_etc = None
    for spec in TARGETS:
        target = PointTarget(target_position(spec), spec["ts_db"], learnable=False)
        rx_dirs = return_fan(target, elements, 14000, half_angle_deg=45.0)
        with timed(f"  echo from target at {spec['bearing']:+.0f} deg"):
            result = render_echo(scene, target, tx_dirs, rx_dirs, echo_grid,
                                 sigma_d=0.45, sigma_t=3e-5, tx_weights=tx_w,
                                 ray_chunk=3000)
        total_etc = result.etc if total_etc is None else total_etc + result.etc
        peak_t = echo_grid[result.etc[N_ELEMENTS // 2, 0].argmax()].item()
        print(f"    echo peak at {peak_t * 1000:6.2f} ms -> range "
              f"{peak_t * C / 2:6.2f} m  (true {spec['range']:.1f} m)")

    # ---- Stage 2: coherent arrivals and beamforming ------------------------- #
    banner("stage 2 -- coherent arrivals and beamforming")
    steer, angles = azimuth_steering(541, SECTOR_DEG)
    shading = shading_window(N_ELEMENTS, "hamming")
    image = None
    for spec in TARGETS:
        target = PointTarget(target_position(spec), spec["ts_db"], learnable=False)
        rx_dirs = return_fan(target, elements, 14000, half_angle_deg=45.0)
        with timed(f"  beamformed echo, target at {spec['bearing']:+.0f} deg"):
            inbound = extract_arrivals(
                trace(scene, tx_dirs), target.position, scene.freqs_khz,
                sigma_d=0.45, ray_weights=tx_w, max_arrivals=10)
            outbound = extract_arrivals(
                trace(_RelocatedScene(scene, target.position), rx_dirs), centre,
                scene.freqs_khz, sigma_d=0.45, max_arrivals=10)
            echo = compose_arrivals(inbound, outbound, target, max_arrivals=60)
            power = beamform(echo, elements, scene.freqs_khz, echo_grid, steer,
                             sigma_t=3e-5, shading=shading, steer_chunk=90)
        image = power if image is None else image + power

        flat = power[:, 0].reshape(-1).argmax()
        b_hat = angles[flat // echo_grid.shape[0]].item()
        r_hat = echo_grid[flat % echo_grid.shape[0]].item() * C / 2
        print(f"    detected at bearing {b_hat:+6.2f} deg (true {spec['bearing']:+.1f}), "
              f"range {r_hat:6.2f} m (true {spec['range']:.1f})")
        spec["bearing_hat"], spec["range_hat"] = b_hat, r_hat

    # ---- Plots -------------------------------------------------------------- #
    _plots(echo_grid, angles, image, total_etc, elements, scene, steer, shading)

    banner("acceptance")
    ok = True
    for spec in TARGETS:
        ok &= check(f"target at {spec['bearing']:+.0f} deg localised in bearing",
                    abs(spec["bearing_hat"] - spec["bearing"]) < 1.0,
                    f"error {spec['bearing_hat'] - spec['bearing']:+.2f} deg")
        ok &= check(f"target at {spec['range']:.0f} m localised in range",
                    abs(spec["range_hat"] - spec["range"]) < 0.5,
                    f"error {spec['range_hat'] - spec['range']:+.2f} m")
    ok &= check("two-way energy response is non-zero", float(total_etc.max()) > 0.0)
    return 0 if ok else 1


def _plots(echo_grid, angles, image, total_etc, elements, scene, steer, shading):
    import matplotlib.pyplot as plt
    import numpy as np
    from hydropt.beamform import ArrivalSet

    rng = echo_grid.detach().numpy() * C / 2.0
    img = image[:, 0].detach().numpy()
    img_db = 10.0 * np.log10(np.maximum(img, img.max() * 1e-6) / img.max())

    fig, ax = plt.subplots(figsize=(9, 5.5))
    m = ax.pcolormesh(rng, angles.numpy(), img_db, cmap="inferno",
                      vmin=-40.0, vmax=0.0, shading="auto")
    for spec in TARGETS:
        ax.plot(spec["range"], spec["bearing"], "o", mfc="none", mec="#7fdfff",
                ms=13, mew=1.6)
    # The surface-reflected return arrives later and shows up as a second,
    # weaker detection at a longer apparent range -- a real source of false
    # contacts in shallow water, not an artefact of the renderer.
    for spec in TARGETS:
        pos = target_position(spec)
        direct = math.dist((0.0, 0.0, VEHICLE_DEPTH), pos)
        image_path = math.hypot(math.hypot(pos[0], pos[1]), pos[2] + VEHICLE_DEPTH)
        ax.plot(0.5 * (direct + image_path), spec["bearing"], "x", color="#7fdfff",
                ms=8, mew=1.4)
    ax.set_xlabel("range (m)")
    ax.set_ylabel("bearing (deg)")
    ax.set_title("Beamformed bearing-range image\n"
                 "circles = true targets, crosses = predicted surface-multipath ghosts")
    ax.set_xlim(10, rng.max())
    fig.colorbar(m, ax=ax, label="dB re peak")
    save(fig, "06_bearing_range.png")

    # Beam pattern of the array itself, from a single synthetic plane wave.
    look = torch.tensor([[1.0, 0.0, 0.0]])
    probe = ArrivalSet(torch.tensor([0.02]), torch.ones(1, 1), -look,
                       torch.zeros(1), torch.zeros(1), torch.ones(1))
    pgrid = make_time_grid(0.018, 0.022, 401)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for kind in ("uniform", "hamming", "blackman"):
        w = shading_window(elements.shape[0], kind)
        p = beamform(probe, elements, scene.freqs_khz, pgrid, steer,
                     sigma_t=2e-4, shading=w, steer_chunk=120)
        pat = p.max(dim=-1).values[:, 0]
        ax.plot(angles.numpy(), 10 * np.log10((pat / pat.max()).numpy()), lw=1.2,
                label=kind)
    ax.set_ylim(-70, 3)
    ax.set_xlabel("look bearing (deg)")
    ax.set_ylabel("dB re peak")
    ax.set_title(f"Array beam pattern, {N_ELEMENTS} elements at {FREQ_HZ / 1e3:.0f} kHz")
    ax.grid(alpha=0.25, lw=0.4)
    ax.legend(fontsize=8)
    save(fig, "06_beam_pattern.png")

    # Energy-only two-way response at the centre element, for contrast.
    fig, ax = plt.subplots(figsize=(9, 3.6))
    e = total_etc[elements.shape[0] // 2, 0].detach().numpy()
    ax.plot(rng, 10 * np.log10(np.maximum(e, e.max() * 1e-8)), lw=0.9, color="#1f6f8b")
    for spec in TARGETS:
        ax.axvline(spec["range"], ls="--", lw=0.9, color="#d94801")
    ax.set_xlim(10, rng.max())
    ax.set_xlabel("range (m)")
    ax.set_ylabel("dB re 1")
    ax.set_title("Two-way energy response, centre element (dashed = true ranges)")
    ax.grid(alpha=0.25, lw=0.4)
    save(fig, "06_two_way_etc.png")


if __name__ == "__main__":
    raise SystemExit(main())
