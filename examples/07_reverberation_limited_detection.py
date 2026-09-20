"""Reverberation-limited detection: when is a target actually visible?

Example 06 put targets in a clean ocean, where any echo above the noise floor is
a detection.  Real active sonar is rarely noise-limited -- it is *reverberation*
limited, and a forward-looking geometry is the hard case: the seabed return
arrives at grazing angles smeared across exactly the range window the targets
occupy.

This example renders both and asks the only question that matters: what is the
echo-to-reverberation ratio, and does the target clear it?

Two things make the answer interesting rather than obvious.

**Beamforming buys detection, not just bearing.**  A target is one patch of
sound from one direction; reverberation arrives from every insonified bearing at
once.  Steering a beam rejects the reverberation that arrives from elsewhere, so
the echo-to-reverberation ratio in a beam is better than at a single element by
roughly the array gain.  The run below measures that improvement rather than
asserting it.

**Scattering strength is invertible.**  ``LambertScattering.strength_db`` is a
parameter like any other, so a measured reverberation series can be fitted for
the bottom type.  The last section recovers a hidden value from the ETC alone.

**Construction and assumptions.**

* *Sonar and environment*: 06's -- 100 kHz, 32 elements at half a
  wavelength at 10 m depth in 30 m of water, the same profile, surface and
  Rayleigh sand bottom loss -- with 0.25 m steps, 600 of them, up to 3
  bounces.
* *Reverberation*: a 30,000-ray Fibonacci cone over the 45 deg sector
  (``N_TX``), each bounce on the seabed or the surface a Lambert patch of
  strength -15 dB (``TRUE_BOTTOM_DB``, rock) with the ray's share of the
  cone's solid angle, and up to 4000 patches kept (900 for the beamformed
  pass) from one seeded draw.
* *Target*: a -32 dB point at 45 m on bearing -12 deg, 16 m deep -- small
  against rock on purpose, so the single-element ratio is near zero.
* *The picture*: arrivals composed as in 06 (8 a leg), rendered as energy
  at the centre element (``render_reverberation``, ``sigma_t`` 30 us) and
  beamformed into 181 Hamming beams; the ratio is echo peak over
  reverberation peak within +/-1 m of the target's range.  The inversion
  fits ``LambertScattering.strength_db`` by Adam (60 steps at 0.4 dB) on
  the log-domain reverberation series.
* *Assumptions*: Lambert scattering at every bounce, with one strength for
  both boundaries; the patches are a random subsample, so the level and
  range spread are unbiased and only the speckle is coarse; no absorption
  difference between the paths worth mentioning at 45 m.
* *To vary*: ``TRUE_BOTTOM_DB`` -24 for sand makes it noise-limited (the
  comment on it says by how much); ``TARGET["ts_db"]`` and range set the
  margin; more beams or elements raise the array gain the run measures.
"""

from __future__ import annotations

import math

import torch

from _common import banner, check, save, setup, timed
from hydropt import (
    ConstantLoss, FlatHeight, PiecewiseLinearProfile, RayleighBottomLoss, Scene,
    make_time_grid,
)
from hydropt.active import PointTarget, _RelocatedScene, compose_arrivals, return_fan
from hydropt.beamform import (
    azimuth_steering, beamform, extract_arrivals, shading_window,
)
from hydropt.launch import fibonacci_cone
from hydropt.reverb import (
    LambertScattering, cone_solid_angle, render_reverberation, reverberation_arrivals,
)
from hydropt.tracer import trace

C = 1500.0
FREQ_HZ = 100e3
LAMBDA = C / FREQ_HZ
N_ELEMENTS = 32
WATER_DEPTH = 30.0
VEHICLE_DEPTH = 10.0
SECTOR_DEG = 45.0
N_TX = 30000

# A rock seabed and a small object -- the regime the example is named for.  At
# -24 dB (sand) against a -2 dB target the echo clears reverberation by nearly
# 40 dB even on a single element, which is a noise-limited problem wearing a
# reverberation-limited label.
TRUE_BOTTOM_DB = -15.0  # rock
TARGET = {"bearing": -12.0, "range": 45.0, "depth": 16.0, "ts_db": -32.0}


def receive_array() -> torch.Tensor:
    y = (torch.arange(N_ELEMENTS, dtype=torch.get_default_dtype())
         - (N_ELEMENTS - 1) / 2) * (LAMBDA / 2)
    return torch.stack((torch.zeros_like(y), y,
                        torch.full_like(y, VEHICLE_DEPTH)), dim=-1)


def build_scene(receivers: torch.Tensor) -> Scene:
    return Scene(
        field=PiecewiseLinearProfile([0.0, VEHICLE_DEPTH, WATER_DEPTH],
                                     [1512.0, 1505.0, 1503.0], learnable=False),
        bottom=FlatHeight(WATER_DEPTH), surface=FlatHeight(0.0),
        source=(0.0, 0.0, VEHICLE_DEPTH), receivers=receivers,
        surface_loss=ConstantLoss(2.0, learnable=False, pressure_release=True),
        bottom_loss=RayleighBottomLoss(1900.0, 1650.0, 0.8, learnable=False),
        freqs_khz=torch.tensor([FREQ_HZ / 1e3]),
        step_size=0.25, n_steps=600, max_bounces=3,
    )


def target_position() -> tuple[float, float, float]:
    b = math.radians(TARGET["bearing"])
    return (TARGET["range"] * math.cos(b), TARGET["range"] * math.sin(b),
            TARGET["depth"])


def transmit_shading(directions: torch.Tensor) -> torch.Tensor:
    az = torch.atan2(directions[:, 1], directions[:, 0])
    el = torch.asin(directions[:, 2].clamp(-1.0, 1.0))
    return (torch.exp(-0.5 * (az / math.radians(14.0)) ** 2)
            * torch.exp(-0.5 * (el / math.radians(8.0)) ** 2))


def main() -> int:
    setup()
    banner("07 -- reverberation-limited detection")

    elements = receive_array()
    centre = elements.mean(0)
    scene = build_scene(elements)
    tx_dirs = fibonacci_cone(N_TX, torch.tensor([1.0, 0.0, 0.0]), SECTOR_DEG)
    tx_w = transmit_shading(tx_dirs)
    omega = cone_solid_angle(SECTOR_DEG) / N_TX
    grid = make_time_grid(0.02, 0.13, 2200)
    rng = grid * C / 2.0

    with timed("  trace the transmit fan"):
        tx_result = trace(scene, tx_dirs)

    # ---- reverberation ------------------------------------------------------ #
    banner("seabed reverberation")
    scattering = LambertScattering(TRUE_BOTTOM_DB, learnable=False)
    # A fixed seed so the patch subsample is one reproducible realisation; the
    # inversion below has to fit the same patches it was given, or it is chasing
    # a fresh speckle pattern every iteration.
    def seeded():
        return torch.Generator().manual_seed(12345)

    with timed("  reverberation arrivals"):
        reverb = reverberation_arrivals(
            tx_result, tx_dirs, scene.freqs_khz, scattering=scattering,
            solid_angle_per_ray=omega, ray_weights=tx_w, boundary="both",
            surface=scene.surface, bottom=scene.bottom, max_arrivals=4000,
            generator=seeded())
    reverb_etc = render_reverberation(reverb, grid, sigma_t=3e-5)
    print(f"  {reverb.n_arrivals} scattering patches, "
          f"ranges {reverb.path_length.min() / 2:.1f} to "
          f"{reverb.path_length.max() / 2:.1f} m")

    # ---- target echo -------------------------------------------------------- #
    banner("target echo")
    target = PointTarget(target_position(), TARGET["ts_db"], learnable=False)
    rx_dirs = return_fan(target, elements, 14000, half_angle_deg=45.0)
    with timed("  echo arrivals"):
        inbound = extract_arrivals(tx_result, target.position, scene.freqs_khz,
                                   sigma_d=0.45, ray_weights=tx_w, max_arrivals=8)
        outbound = extract_arrivals(
            trace(_RelocatedScene(scene, target.position), rx_dirs), centre,
            scene.freqs_khz, sigma_d=0.45, max_arrivals=8)
        echo = compose_arrivals(inbound, outbound, target, max_arrivals=40)
    echo_etc = render_reverberation(echo, grid, sigma_t=3e-5)
    print(f"  echo at {echo.time.min() * C / 2:.2f} m "
          f"(true {TARGET['range']:.1f} m), {echo.n_arrivals} arrivals")

    # ---- omnidirectional vs beamformed -------------------------------------- #
    banner("does the target clear the reverberation?")
    gate = (rng - TARGET["range"]).abs() < 1.0
    omni_ratio = _db(echo_etc[0, 0][gate].max() / reverb_etc[0, 0][gate].max())
    print(f"  single element:  echo-to-reverberation {omni_ratio:+6.2f} dB")

    steer, angles = azimuth_steering(181, SECTOR_DEG)
    shading = shading_window(N_ELEMENTS, "hamming")
    # Beamforming cost is linear in patches x beams x elements, so the coherent
    # pass uses a coarser subsample of the reverberation than the energy ETC
    # does.  Random subsampling is unbiased, so the level and range spread
    # survive; only the speckle realisation is coarser.
    reverb_bf_arrivals = reverberation_arrivals(
        tx_result, tx_dirs, scene.freqs_khz, scattering=scattering,
        solid_angle_per_ray=omega, ray_weights=tx_w, boundary="both",
        surface=scene.surface, bottom=scene.bottom, max_arrivals=900,
        generator=seeded())
    with timed("  beamform echo + reverberation"):
        beam_kw = dict(sigma_t=3e-5, shading=shading, steer_chunk=20)
        echo_bf = beamform(echo, elements, scene.freqs_khz, grid, steer, **beam_kw)
        reverb_bf = beamform(reverb_bf_arrivals, elements, scene.freqs_khz, grid,
                             steer, **beam_kw)

    beam = int((angles - TARGET["bearing"]).abs().argmin())
    beam_ratio = _db(echo_bf[beam, 0][gate].max() / reverb_bf[beam, 0][gate].max())
    gain = beam_ratio - omni_ratio
    print(f"  on-target beam:  echo-to-reverberation {beam_ratio:+6.2f} dB")
    print(f"  improvement from beamforming: {gain:+.2f} dB "
          f"(array gain 10log10(N) = {10 * math.log10(N_ELEMENTS):.1f} dB)")

    # ---- invert the bottom type from the reverberation series --------------- #
    banner("recover the bottom scattering strength")
    fitted, history = _fit_bottom(tx_result, tx_dirs, tx_w, scene, omega, grid,
                                  reverb_etc.detach(), seeded)
    print(f"  true {TRUE_BOTTOM_DB:.2f} dB, recovered {fitted:+.2f} dB, "
          f"error {abs(fitted - TRUE_BOTTOM_DB):.3f} dB")

    _plots(rng, angles, grid, reverb_etc, echo_etc, echo_bf, reverb_bf, beam, history)

    banner("acceptance")
    ok = check("beamforming improves echo-to-reverberation", gain > 3.0,
               f"{gain:+.2f} dB")
    ok &= check("target clears reverberation in its beam", beam_ratio > 0.0,
                f"{beam_ratio:+.2f} dB")
    ok &= check("the problem is genuinely reverberation-limited",
                omni_ratio < 12.0,
                f"single element {omni_ratio:+.2f} dB")
    ok &= check("reverberation spans the range window",
                float(reverb.path_length.max() - reverb.path_length.min()) > 60.0)
    ok &= check("bottom scattering strength recovered within 1 dB",
                abs(fitted - TRUE_BOTTOM_DB) < 1.0,
                f"{abs(fitted - TRUE_BOTTOM_DB):.3f} dB")
    return 0 if ok else 1


def _db(x) -> float:
    return 10.0 * math.log10(max(float(x), 1e-300))


def _fit_bottom(tx_result, tx_dirs, tx_w, scene, omega, grid, target_etc, seeded):
    """Fit the Lambert strength to a measured reverberation series.

    Only the scattering strength is free, and it enters the energy linearly, so
    this converges in a handful of steps -- the point is that it is a parameter
    like any other, not that the optimisation is hard.
    """
    model = LambertScattering(-30.0, learnable=True)
    opt = torch.optim.Adam(model.parameters(), lr=0.4)
    history = []
    for _ in range(60):
        opt.zero_grad(set_to_none=True)
        arrivals = reverberation_arrivals(
            tx_result, tx_dirs, scene.freqs_khz, scattering=model,
            solid_angle_per_ray=omega, ray_weights=tx_w, boundary="both",
            surface=scene.surface, bottom=scene.bottom, max_arrivals=4000,
            generator=seeded())
        pred = render_reverberation(arrivals, grid, sigma_t=3e-5)
        eps = 1e-6 * float(target_etc.max())
        loss = ((torch.log10(pred + eps) - torch.log10(target_etc + eps)) ** 2).mean()
        loss.backward()
        opt.step()
        history.append(float(model.strength_db.detach()))
    return float(model.strength_db.detach()), history


def _plots(rng, angles, grid, reverb_etc, echo_etc, echo_bf, reverb_bf, beam, history):
    import matplotlib.pyplot as plt
    import numpy as np

    r = rng.detach().numpy()
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    ref = float(reverb_etc.max())
    ax.plot(r, 10 * np.log10(np.maximum(reverb_etc[0, 0].detach().numpy(), ref * 1e-9) / ref),
            lw=0.8, color="#7f5539", label="reverberation")
    ax.plot(r, 10 * np.log10(np.maximum(echo_etc[0, 0].detach().numpy(), ref * 1e-9) / ref),
            lw=1.2, color="#1f6f8b", label="target echo")
    ax.axvline(TARGET["range"], ls="--", lw=0.9, color="#d94801")
    ax.set_ylabel("dB re reverberation peak")
    ax.set_title("Single element: the target is buried in the seabed return")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, lw=0.4)
    ax.set_ylim(-90, 5)

    rb = reverb_bf[beam, 0].detach().numpy()
    eb = echo_bf[beam, 0].detach().numpy()
    ref2 = max(rb.max(), eb.max())
    ax2.plot(r, 10 * np.log10(np.maximum(rb, ref2 * 1e-9) / ref2), lw=0.8,
             color="#7f5539", label="reverberation")
    ax2.plot(r, 10 * np.log10(np.maximum(eb, ref2 * 1e-9) / ref2), lw=1.2,
             color="#1f6f8b", label="target echo")
    ax2.axvline(TARGET["range"], ls="--", lw=0.9, color="#d94801")
    ax2.set_xlabel("range (m)")
    ax2.set_ylabel("dB re peak")
    ax2.set_title(f"Steered to {TARGET['bearing']:+.0f} deg: "
                  "the beam rejects reverberation from other bearings")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.25, lw=0.4)
    ax2.set_ylim(-90, 5)
    ax2.set_xlim(15, 70)
    fig.tight_layout()
    save(fig, "07_echo_vs_reverberation.png")

    total = (echo_bf + reverb_bf)[:, 0].detach().numpy()
    fig, ax = plt.subplots(figsize=(9, 5))
    db = 10 * np.log10(np.maximum(total, total.max() * 1e-5) / total.max())
    m = ax.pcolormesh(r, angles.numpy(), db, cmap="inferno", vmin=-35, vmax=0,
                      shading="auto")
    ax.plot(TARGET["range"], TARGET["bearing"], "o", mfc="none", mec="#7fdfff",
            ms=14, mew=1.6)
    ax.set_xlim(15, 70)
    ax.set_xlabel("range (m)")
    ax.set_ylabel("bearing (deg)")
    ax.set_title("Beamformed image: target echo on a bed of seabed reverberation")
    fig.colorbar(m, ax=ax, label="dB re peak")
    save(fig, "07_bearing_range_with_reverb.png")

    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(history, lw=1.4, color="#1f6f8b")
    ax.axhline(TRUE_BOTTOM_DB, ls="--", lw=1.0, color="#d94801", label="true")
    ax.set_xlabel("iteration")
    ax.set_ylabel("Lambert strength (dB)")
    ax.set_title("Bottom type recovered from the reverberation series")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, lw=0.4)
    save(fig, "07_bottom_inversion.png")


if __name__ == "__main__":
    raise SystemExit(main())
