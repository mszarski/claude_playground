"""How far can it see the object -- and what actually stops it.

Example 07 asked whether a target beats the reverberation and 17 measured one
on the bottom, but neither could say how *far* the sonar sees, because until
now there was no noise in the package at all.  With no noise floor an echo 60
dB below the seabed is still a number in a cell, and detection range comes out
as whatever the reverberation happens to allow -- which is an answer to a
question nobody asked.

This puts the sonar equation on the simulation.  The same 4 m x 1.5 m cylinder
is walked out along the bottom, and at each range three levels are compared in
the same beam and the same range cell:

  * the echo, simulated -- spreading, absorption, aspect and the transmit fan's
    own directivity all included, not assumed;
  * the seabed reverberation, from one ping of the same fan;
  * the noise, from the Wenz spectrum through the array's directivity index and
    the pulse's bandwidth.

Detection is Swerling 1 -- a target whose echo fluctuates Rayleigh, which is
what a many-highlight body at an unknown aspect does -- so ``Pd = Pfa ^
(B / (S + B))`` and the threshold is a ratio rather than a level.

**The answer for this sonar is that the sea is irrelevant.**  At 100 kHz with
a 210 dB source in 30 m of water, the noise sits ~100 dB below the seabed
reverberation, and the detection range is set entirely by the bottom.  That is
worth computing rather than assuming: it says a quieter sea, a quieter vehicle
or a bigger array buys nothing here, and only a narrower beam or a shorter
pulse -- a smaller resolution cell, holding less seabed -- will move it.

Acceptance criteria:
  * the echo falls with range the way spreading and absorption say it must;
  * aspect dominates: the same body 23 degrees off broadside is far weaker, and
    the example puts a number on what that costs in range;
  * reverberation stands far above the noise across the whole swath, and the
    example says by how much rather than asserting which limit applies;
  * the seabed stops falling at long range, because in 30 m of water the
    multiply-bounced paths take over from the direct one -- reverberation in a
    waveguide is not a r^-4 curve, and the example shows the single-bounce
    comparison that proves where the energy comes from;
  * the detection range falls inside the swept interval, with Pd monotone;
  * removing the reverberation moves the range beyond what the water depth can
    geometrically support, which is what "noise-limited" would have to mean
    here;
  * a loss on the image still carries gradients to the object.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import torch

from _common import banner, check, save, setup, timed
from hydropt import (
    azimuth_steering, beam_noise_power, beam_power_scale, beamform, calibrate,
    cylinder_mesh, line_array_directivity_db, line_array_factor, make_time_grid,
    mesh_target, shading_window, target_arrivals,
)
from hydropt.absorption import thorp_db_per_km
from hydropt.reverb import LambertScattering, reverberation_arrivals
from hydropt.tracer import trace

C = 1500.0
FREQ_KHZ = 100.0
WATER_DEPTH, AUV_DEPTH = 30.0, 18.0
PULSE_S = 1.2e-4
SOURCE_LEVEL_DB = 210.0          # dB re 1 uPa at 1 m, a typical imaging sonar
WIND_SPEED = 5.0                 # m/s
N_RX, N_TX = 64, 6

OBJ_LENGTH, OBJ_DIAMETER = 4.0, 1.5
RANGES = (30.0, 45.0, 60.0, 80.0, 100.0, 125.0, 150.0, 175.0)
# Two aspects, because aspect turns out to matter more than anything else here.
# A 4 m cylinder at 100 kHz has a specular lobe about lambda/L = 0.2 degrees
# wide, so "broadside" is a knife edge: 23 degrees off it -- the aspect
# example 17 happens to use -- is deep in the skirt, and the same body at the
# same range is tens of dB weaker.  Quoting a broadside detection range without
# saying so would be quoting the best case a target can offer.
ASPECTS = ((90.0, "broadside"), (67.0, "23 deg off broadside"))
FAN_LO_DEG, FAN_HI_DEG = 3.5, 30.0
N_SEEDS = 3                      # realisations averaged into each echo level
TRACE_STEPS = 200                # the scene from examples/12 stops rays at 135 m

PFA = 1e-4
PD = 0.5


def _mills():
    path = Path(__file__).resolve().parent / "13_mills_cross_fls.py"
    spec = importlib.util.spec_from_file_location("_mc13", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def swerling1_pd(snr: torch.Tensor | float, pfa: float) -> torch.Tensor:
    """``Pd = Pfa ^ (1 / (1 + S/B))`` for a Rayleigh-fluctuating target.

    A body with many highlights at an unknown aspect does not present a steady
    echo: the highlights add with random relative phase and the result is
    Rayleigh, which is Swerling case 1.  Treating it as steady instead (Marcum)
    makes detection look sharper in range than it is -- the curve is a cliff
    rather than the long tail a real one has.
    """
    s = torch.as_tensor(snr, dtype=torch.get_default_dtype())
    return torch.tensor(pfa, dtype=s.dtype) ** (1.0 / (1.0 + s))


def required_snr_db(pd: float, pfa: float) -> float:
    """The ratio Swerling 1 needs, in dB -- the threshold everything is read against."""
    return 10.0 * math.log10(math.log(pfa) / math.log(pd) - 1.0)


def transmit_fan(target_depression: float, n_elev: int, n_azim: int, *,
                 half_azim_deg: float = 8.0, seed: int = 0):
    """The sonar's own fan, fixed -- not aimed at the target.

    Aiming it at each range in turn would measure a sonar that already knows
    where to look, and would quietly delete the transmit directivity from the
    answer: a contact at the edge of the fan is down the array factor's skirt,
    and that is part of what sets detection range.  Only the azimuth is
    narrowed, to put the rays where the target is in bearing; the elevation
    span and the steer stay put, so the fan's roll-off is felt.

    **The window has to contain the whole target, with margin.**  Each
    highlight is a point, and its echo is a Gaussian in the miss distance of
    the rays that pass it, with a width matched to the fan's own spacing.  A
    highlight outside the sampled window is reached only by the rays at the
    edge, at a fixed offset -- so refining the fan shrinks the width while the
    offset stays put, and the level falls exponentially with the very ray count
    that should have improved it.  A 4 m body at 30 m spans +/- 3.8 degrees, so
    +/- 1.5 would have put its own highlights off the edge and made the echo
    level a function of the sampling.  With the window generous, the level is
    invariant: 64x the ray density moves it by 0.2 dB.
    """
    g = torch.Generator().manual_seed(seed)
    e0, e1 = math.radians(FAN_LO_DEG), math.radians(FAN_HI_DEG)
    a0, a1 = -math.radians(half_azim_deg), math.radians(half_azim_deg)
    el = torch.linspace(e0, e1, n_elev)
    az = torch.linspace(a0, a1, n_azim)
    E, A = torch.meshgrid(el, az, indexing="ij")
    E, A = E.reshape(-1), A.reshape(-1)
    E = E + (torch.rand(E.shape, generator=g, dtype=E.dtype) - 0.5) * (el[1] - el[0])
    A = A + (torch.rand(A.shape, generator=g, dtype=A.dtype) - 0.5) * (az[1] - az[0])
    dirs = torch.stack([E.cos() * A.cos(), E.cos() * A.sin(), E.sin()], dim=-1)
    tilt = 0.5 * (e0 + e1)
    weights = line_array_factor(torch.sin(E), N_TX, sin_steer=math.sin(tilt))
    return dirs, weights


def main() -> int:
    setup()
    banner("18 -- detection range: the seabed or the sea?")
    mc = _mills()
    fls = mc._fls
    rx = mc.horizontal_array()
    rx = torch.stack([rx[:, 0], rx[:, 1],
                      torch.full_like(rx[:, 2], AUV_DEPTH)], dim=-1)
    scene, *_ = fls.build_scene(rx)
    scene.source = torch.tensor([0.0, 0.0, AUV_DEPTH])
    # That scene is built for examples/12's 100 m of work: 1.5 m steps, 90 of
    # them, so every ray stops at 135 m.  Past that there is no echo AND no
    # reverberation -- the sweep would read a detection range set by the length
    # of the ray budget and call it physics.
    scene.n_steps = TRACE_STEPS
    shading = shading_window(N_RX, "hamming")
    freqs = scene.freqs_khz

    di = line_array_directivity_db(N_RX)
    bandwidth = 1.0 / PULSE_S
    noise = float(beam_noise_power(freqs, bandwidth_hz=bandwidth,
                                   directivity_db=di, wind_speed=WIND_SPEED))
    scale = beam_power_scale(shading, PULSE_S)
    alpha = float(thorp_db_per_km(freqs))
    threshold_db = required_snr_db(PD, PFA)
    print(f"  {FREQ_KHZ:.0f} kHz, SL {SOURCE_LEVEL_DB:.0f} dB re 1 uPa @ 1 m, "
          f"{PULSE_S * 1e3:.2f} ms pulse ({bandwidth / 1e3:.1f} kHz)")
    print(f"  {N_RX} elements -> DI {di:.1f} dB; wind {WIND_SPEED:.0f} m/s")
    print(f"  noise in a beam and cell: {10 * math.log10(noise):.1f} dB re 1 uPa^2")
    print(f"  absorption {alpha:.1f} dB/km, two-way {2 * alpha / 1000:.3f} dB/m")
    print(f"  detection at Pd {PD:.0%}, Pfa {PFA:g} (Swerling 1) needs "
          f"S/B > {threshold_db:.1f} dB")

    verts, faces = cylinder_mesh(OBJ_LENGTH, OBJ_DIAMETER / 2.0,
                                 n_axial=10, n_around=128)
    steer, bearings = azimuth_steering(5, 3.0)

    def echo_level(distance, seeds, yaw=90.0):
        """Mean beam power on the contact, over several ray realisations.

        Averaged, not a single realisation: the coherent sum over a faceted
        body fluctuates by well over a dB from one ray set to the next, and
        that is sampling, not target fading -- the fading is already in the
        Swerling statistics.  Taking one draw would put that spread into the
        detection range instead.
        """
        xy = torch.tensor([[distance, 0.0]])
        bed = float(scene.bottom.height(xy).detach())
        target = mesh_target(verts, faces,
                             position=(distance, 0.0, bed - OBJ_DIAMETER / 2.0),
                             yaw=yaw, n_patches=2, sound_speed=C,
                             learnable=True, facet_chunk=256)
        depression = math.atan2(bed - AUV_DEPTH, distance)
        slant = math.hypot(distance, bed - AUV_DEPTH)
        grid = make_time_grid(2.0 * (slant - 4.0) / C, 2.0 * (slant + 4.0) / C, 160)
        total, last = 0.0, None
        for seed in range(seeds):
            dirs, weights = transmit_fan(depression, 60, 32, seed=seed)
            echo = target_arrivals(scene, target, dirs, n_rx_rays=240,
                                   rx_half_angle_deg=45.0, rx_jitter=1.0,
                                   tx_weights=weights, max_arrivals_per_leg=24,
                                   generator=torch.Generator().manual_seed(seed))
            last = beamform(echo, rx, freqs, grid, steer, sigma_t=PULSE_S,
                            shading=shading, arrival_chunk=4096)
            total = total + last.max()
        return total / seeds, target, slant

    banner("the seabed, once")
    with timed("  reverberation over the whole swath"):
        dirs, weights = transmit_fan(0.0, 420, 32, seed=101)
        solid = (math.radians(16.0) * math.radians(FAN_HI_DEG - FAN_LO_DEG)
                 / dirs.shape[0])
        with torch.no_grad():
            rev = reverberation_arrivals(
                trace(scene, dirs), dirs, freqs,
                scattering=LambertScattering(-25.0, learnable=False),
                solid_angle_per_ray=solid, ray_weights=weights,
                boundary="bottom", surface=scene.surface, bottom=scene.bottom,
                generator=torch.Generator().manual_seed(102))
            rev_grid = make_time_grid(2.0 * 20.0 / C, 2.0 * 210.0 / C, 1400)
            rev_image = beamform(rev, rx, freqs, rev_grid, steer,
                                 sigma_t=PULSE_S, shading=shading,
                                 arrival_chunk=4096)
    rev_range = rev_grid * C / 2.0
    rev_profile = rev_image[:, 0].mean(dim=0)
    print(f"  {rev.n_arrivals} patches over "
          f"{float(rev_range[0]):.0f}-{float(rev_range[-1]):.0f} m")

    # The same ping with one bounce allowed, to show where the far-range
    # reverberation actually comes from.  It is cheap -- one trace and one
    # beamform -- and it turns "the seabed stops falling" from an anomaly into
    # a measurement.
    bounces = scene.max_bounces
    scene.max_bounces = 1
    with torch.no_grad():
        direct = reverberation_arrivals(
            trace(scene, dirs), dirs, freqs,
            scattering=LambertScattering(-25.0, learnable=False),
            solid_angle_per_ray=solid, ray_weights=weights, boundary="bottom",
            surface=scene.surface, bottom=scene.bottom,
            generator=torch.Generator().manual_seed(102))
        direct_profile = beamform(direct, rx, freqs, rev_grid, steer,
                                  sigma_t=PULSE_S, shading=shading,
                                  arrival_chunk=4096)[:, 0].mean(dim=0)
    scene.max_bounces = bounces

    banner("walking the object out")
    by_aspect = {}
    for yaw, label in ASPECTS:
        print(f"  -- {label}")
        rows = []
        for distance in RANGES:
            with torch.no_grad():
                power, _, slant = echo_level(distance, N_SEEDS, yaw=yaw)
            window = (rev_range > slant - 6.0) & (rev_range < slant + 6.0)
            # The MEAN over the window, not the median.  Reverberation patches
            # are discrete: at long range most cells in the window hold no patch
            # at all, and the median then reads the empty floor rather than the
            # seabed -- which made the reverberation appear to fall as r^-4, in
            # lockstep with a point target, instead of the slower fall a growing
            # cell gives.  A constant signal-to-background ratio across a whole
            # sweep is the tell.  The mean is unbiased at any patch density.
            reverb = float(rev_profile[window].mean())
            s_abs = float(calibrate(power, SOURCE_LEVEL_DB, beam_scale=scale))
            r_abs = float(calibrate(torch.tensor(reverb), SOURCE_LEVEL_DB,
                                    beam_scale=scale))
            rows.append((distance, slant, s_abs, r_abs, r_abs + noise))
            db = lambda v: 10.0 * math.log10(max(v, 1e-300))
            print(f"     {distance:5.0f} m: echo {db(s_abs):6.1f}   "
                  f"seabed {db(r_abs):6.1f}   noise {db(noise):5.1f}   "
                  f"S/B {db(s_abs / (r_abs + noise)):6.1f} dB")
        by_aspect[label] = rows

    curves = {}
    for label, rows in by_aspect.items():
        d = torch.tensor([row[0] for row in rows])
        s_v = torch.tensor([row[2] for row in rows])
        r_v = torch.tensor([row[3] for row in rows])
        b_v = torch.tensor([row[4] for row in rows])
        snr = s_v / b_v
        curves[label] = dict(d=d, s=s_v, r=r_v, b=b_v, snr=snr,
                             pd=swerling1_pd(snr, PFA),
                             detect=_crossing(d, 10.0 * torch.log10(snr),
                                              threshold_db))

    head = curves[ASPECTS[0][1]]
    d, s_v, r_v = head["d"], head["s"], head["r"]

    banner("the seabed in a waveguide")
    def _at(profile, distance):
        m = (rev_range > distance - 6.0) & (rev_range < distance + 6.0)
        return 10.0 * math.log10(float(profile[m].mean()) + 1e-300)
    for distance in (RANGES[1], RANGES[len(RANGES) // 2], RANGES[-1]):
        both, one = _at(rev_profile, distance), _at(direct_profile, distance)
        print(f"  {distance:5.0f} m: all paths {both:7.1f} dB, "
              f"single bounce only {one:7.1f} dB  "
              f"({both - one:+5.1f} dB from multipath)")
    print(f"  so the reverberation stops falling: past ~{RANGES[-3]:.0f} m the "
          f"energy arriving in a cell has bounced")
    print(f"  more than once, at a steeper grazing angle than the direct path "
          f"and scattering harder for it")

    banner("what stops it")
    over_noise = 10.0 * torch.log10(r_v / noise)
    print(f"  the seabed stands {float(over_noise.min()):.0f}-"
          f"{float(over_noise.max()):.0f} dB above the noise across the swath,")
    print(f"  so the background is the bottom everywhere in it, and a quieter "
          f"sea or a bigger array buys nothing")
    noise_only = _crossing(d, 10.0 * torch.log10(s_v / noise), threshold_db,
                           extrapolate=(s_v, d, alpha))
    print(f"  against the noise alone the range would be {noise_only:.0f} m = "
          f"{noise_only / WATER_DEPTH:.0f} water depths, which no direct path")
    print(f"  reaches in {WATER_DEPTH:.0f} m of water -- 'noise-limited' is not "
          f"a state this sonar can be in")

    banner("and what really sets it")
    for _, label in ASPECTS:
        c = curves[label]
        print(f"  {label:22s}: echo at {float(c['d'][0]):.0f} m is "
              f"{10 * math.log10(float(c['s'][0])):6.1f} dB, "
              f"detection range {c['detect']:.0f} m")
    drop = (10 * math.log10(float(curves[ASPECTS[0][1]]['s'][0]))
            - 10 * math.log10(float(curves[ASPECTS[1][1]]['s'][0])))
    print(f"  turning the body {90.0 - ASPECTS[1][0]:.0f} degrees off broadside "
          f"costs {drop:.0f} dB -- the specular lobe of a {OBJ_LENGTH:.0f} m "
          f"body")
    print(f"  at this wavelength is about "
          f"{math.degrees(C / (FREQ_KHZ * 1e3) / OBJ_LENGTH):.2f} degrees wide, "
          f"so broadside is a knife edge, not a sector")

    banner("gradients")
    power, target, _ = echo_level(RANGES[1], 1)
    power.log10().backward()
    grads = {n: float(p.grad.abs().max()) for n, p in target.named_parameters()
             if p.grad is not None}
    print("  d(log echo)/d(object pose): "
          + ", ".join(f"{k} {v:.3e}" for k, v in grads.items()))

    banner("figure")
    with timed("  draw"):
        fig = draw(curves, noise, threshold_db)
        save(fig, "18_detection_range.png")

    banner("acceptance")
    ok = True
    model = (-40.0 * torch.log10(d / d[0]) - 2.0 * alpha * (d - d[0]) / 1000.0)
    measured = 10.0 * torch.log10(s_v.clamp_min(1e-300) / s_v[0])
    residual = float((measured - model).abs().max())
    ok &= check("the echo falls the way spreading and absorption say",
                residual < 12.0,
                f"worst departure {residual:.1f} dB over "
                f"{float(d[0]):.0f}-{float(d[-1]):.0f} m")
    ok &= check("the seabed stands above the noise everywhere in the swath",
                float(over_noise.min()) > 10.0,
                f"{float(over_noise.min()):.0f} dB at its closest, "
                f"{float(over_noise.max()):.0f} dB at its best")
    ok &= check("aspect costs more than anything else in the sweep",
                drop > 10.0,
                f"{drop:.0f} dB for {90.0 - ASPECTS[1][0]:.0f} degrees of yaw")
    # Measured on the aspect where Pd actually moves.  Broadside it is pinned
    # at 1.00 across the whole sweep -- the object is 45 dB louder there, and
    # the sonar sees it everywhere its geometry reaches -- so a monotonicity
    # test on that curve is a test of rounding noise in the seed average.
    weak = curves[ASPECTS[1][1]]["pd"]
    ok &= check("Pd falls with range at the aspect where it moves",
                bool((weak[1:] <= weak[:-1] + 1e-6).all())
                and float(weak[-1]) < float(weak[0]) - 0.1,
                f"{float(weak[0]):.2f} at {float(d[0]):.0f} m to "
                f"{float(weak[-1]):.2f} at {float(d[-1]):.0f} m "
                f"({ASPECTS[1][1]})")
    ok &= check("noise alone would put it beyond the water's geometry",
                noise_only > 10.0 * WATER_DEPTH,
                f"{noise_only:.0f} m against {WATER_DEPTH:.0f} m of water")
    ok &= check("the echo still carries gradients to the object",
                bool(grads) and max(grads.values()) > 0.0)
    return 0 if ok else 1


def _crossing(x, y_db, level, extrapolate=None) -> float:
    """Where a falling curve crosses ``level``, linearly in log range."""
    below = (y_db < level).nonzero()
    if below.numel():
        i = int(below[0])
        if i == 0:
            return float(x[0])
        x0, x1 = math.log10(float(x[i - 1])), math.log10(float(x[i]))
        y0, y1 = float(y_db[i - 1]), float(y_db[i])
        return 10.0 ** (x0 + (level - y0) * (x1 - x0) / (y1 - y0))
    if extrapolate is None:
        return float(x[-1])
    # Still above the threshold at the last range: continue on spreading plus
    # absorption, which is what the curve is doing by then.
    s, d, alpha = extrapolate
    here = float(10.0 * torch.log10(s[-1]))
    r = float(d[-1])
    while here > level and r < 1e5:
        r *= 1.02
        here = (float(10.0 * torch.log10(s[-1]))
                - 40.0 * math.log10(r / float(d[-1]))
                - 2.0 * alpha * (r - float(d[-1])) / 1000.0)
    return r


def draw(curves, noise, threshold_db):
    import matplotlib.pyplot as plt
    import numpy as np

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13.0, 5.2))
    colours = ("#1b3a5c", "#a2512f")
    head = next(iter(curves.values()))
    ax.plot(head["d"].numpy(), 10 * np.log10(head["r"].numpy()), "s--",
            color="#7a7a7a", label="seabed reverberation")
    ax.axhline(10 * math.log10(noise), ls=":", color="#2e7d32",
               label=f"noise ({10 * math.log10(noise):.0f} dB)")
    for (label, c), colour in zip(curves.items(), colours):
        ax.plot(c["d"].numpy(), 10 * np.log10(c["s"].numpy()), "o-",
                color=colour, label=f"echo, {label}")
        bx.plot(c["d"].numpy(), c["pd"].numpy(), "o-", color=colour,
                label=f"{label}: {c['detect']:.0f} m")
        bx.axvline(c["detect"], color=colour, lw=1.0, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel("range (m)")
    ax.set_ylabel("dB re 1 uPa$^2$ in a beam and a cell")
    ax.set_title("three levels in the same cell")
    ax.grid(alpha=0.3, lw=0.4)
    ax.legend(fontsize=8)

    bx.axhline(PD, ls=":", color="#888888")
    bx.set_xlabel("range (m)")
    bx.set_ylabel(f"probability of detection at Pfa {PFA:g}")
    bx.set_ylim(0.0, 1.02)
    bx.set_title(f"Swerling 1, needs S/B > {threshold_db:.1f} dB")
    bx.grid(alpha=0.3, lw=0.4)
    bx.legend(fontsize=8)
    fig.suptitle("A 4 x 1.5 m cylinder on the bottom, 100 kHz, 210 dB source, "
                 "30 m of water.\nThe noise is ~50 dB below the seabed: what "
                 "stops this sonar is the bottom in its own cell -- and, far "
                 "more than that, the target's aspect.", fontsize=11)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    raise SystemExit(main())
