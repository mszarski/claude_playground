"""Milestone 12: a 3-D seamount, and the error our N x 2-D propagation makes over it.

Our eigenray tier is 2-D: `BellhopProvider` traces (range, depth) only, and over relief it is used N x 2-D
(`along_profile`, one depth profile per bearing from the source, as sim/reverb_relief.py does). It cannot turn
a ray out of its vertical plane, so horizontal refraction by a sloping bed (a ray reflected off a seamount's
flank leaves on another bearing) is missing. This file does not add 3-D propagation; it measures what that
omission costs on one case, against BELLHOP3D (Acoustics Toolbox, GPL-3, run as a separate program:
bellhop3d_oracle.py) in its two modes on the same bed, the same beams and the same receivers: '3' (3-D) and '2'
(N x 2-D, "neglecting horizontal refraction" in its own source, bellhop3D.f90). Their difference is the 3-D
effect alone; ours is compared with both.

The case (ours, chosen so that the summit stands above the source and receiver depth and the flanks are steep):
200 m of 1500 m/s water over a fluid half-space of 1700 m/s, 1.9 g/cm^3, 0.5 dB/wavelength; a Gaussian
seamount (sim/bathymetry.py `seamount`) 180 m high (summit 20 m deep), sigma 500 m (flank slope up to 0.22, 12
degrees), centred 2.5 km north and 0.3 km east of the source; source and receivers 50 m deep; 8 kHz (the flank
array's band centre); incoherent TL on bearings -20 to 36 degrees (from north towards east), 0.1 to 6.5 km.
The incoherent sum is used because the question is where energy goes, not the interference pattern.

Measured here (2026-11-21, Acoustics Toolbox HEAD 8b4682b), with the test's beams and, in brackets, with four
times as many (241 x 721; the 3-D range averages move by 1.1 dB at most, the 3-D minus N x 2-D difference by
0.6 dB): from 0.2 to 1.8 km 3-D and N x 2-D agree to 0.07 dB on every bearing. That is not a level bed (at
1.8 km on the 6 degree bearing the bed is already 64 m shallower): the two modes treat in-plane relief
alike and differ only by rays turned out of their plane, which shows farther out. Averaged over 3.5 to 6.5 km:
on bearings that pass over the flanks (-4 to -2 and 16 to 18 degrees) the 3-D loss is 3.1 to 3.5 dB HIGHER
[3.0 to 3.6], and it is higher on most bearings from -6 to +20 degrees: energy is turned out of them. On
bearings beside the seamount (-20 and 32 degrees) it is 1.1 to 1.2 dB LOWER [1.1 to 1.2], consistent with that
energy arriving there (not shown: -1 dB appears also at -18 and -10 degrees). Over the summit itself
(6 degrees, where the 20 m summit blocks the 50 m paths in both) the
difference is 0.9 dB [0.4]. N x 2-D sees none of it. Our tier against the converged 3-D field: -3.8 dB at -4
and at 16 degrees (too loud), -2.2 dB over the summit, 0.0 and -0.7 dB beside it at -20 and 32 degrees
(against the test's beams: -4.5, -4.5, -3.4, -0.5, -1.2 dB).
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from gpusonar.sim import bathymetry, find_bellhop
from gpusonar.sim.bellhop import BellhopProvider, OceanEnvironment
from bellhop3d_oracle import find_bellhop3d, run_bellhop3d

WATER_DEPTH, WATER, BOTTOM = 200.0, 1500.0, (1700.0, 1.9, 0.5)
SEAMOUNT = dict(depth=WATER_DEPTH, height=180.0, center=[2500.0, 300.0], sigma=500.0)
FREQUENCY, SOURCE_DEPTH, RECEIVER_DEPTH = 8000.0, 50.0, 50.0
BEARINGS = np.arange(-20.0, 36.01, 2.0)
RANGES = np.arange(100.0, 6500.01, 100.0)
# Beams: 121 in elevation over +-30 degrees (the bottom's critical grazing angle is acos(1500/1700) = 28 degrees;
# steeper paths leave through the bed), 361 in azimuth over +-60 degrees; Gaussian beams in Cartesian coordinates
# ('B'). With 241 x 721 beams the range averages below move by 1.1 dB at most, their 3-D minus N x 2-D
# differences by 0.6 dB (measured).
FAN = dict(n_alpha=121, alpha_deg=(-30.0, 30.0), n_beta=361, beta_deg=(-60.0, 60.0), box_km=(7.9, 4.9), beam="B")
GRID_X, GRID_Y = np.arange(-1000.0, 8000.01, 25.0), np.arange(-5000.0, 5000.01, 25.0)
NEAR, FAR = (200.0, 1800.0), (3500.0, 6500.0)
FLANKS, BESIDE = (-4.0, -2.0, 16.0, 18.0), (-20.0, 32.0)
OURS = (-20.0, -4.0, 6.0, 16.0, 32.0)

needs_bellhop3d = pytest.mark.skipif(find_bellhop3d() is None,
                                     reason="Bellhop3D not installed (GPUSONAR_BELLHOP3D; bellhop3d.exe beside the Bellhop executable is found too)")
needs_bellhop = pytest.mark.skipif(find_bellhop() is None, reason="Bellhop not installed (GPUSONAR_BELLHOP)")


def test_seamount_description():
    bed = bathymetry.make_bathymetry({"seamount": SEAMOUNT})
    assert bed.depth(2500.0, 300.0) == pytest.approx(20.0)
    assert bed.depth(2500.0 + 500.0, 300.0) == pytest.approx(200.0 - 180.0 * np.exp(-0.5))
    assert bed.depth(-50e3, 0.0) == pytest.approx(200.0)
    gx, gy = bed.gradient(np.array([2000.0]), np.array([300.0]))
    h = 0.01
    numeric = (bed.depth(2000.0 + h, 300.0) - bed.depth(2000.0 - h, 300.0)) / (2 * h)
    assert gx[0] == pytest.approx(numeric, rel=1e-6) and gy[0] == pytest.approx(0.0, abs=1e-12)
    assert gx[0] < 0.0                                        # shallower towards the summit (depth grows down)
    assert bed.flat_depth is None and bathymetry.seamount(100.0, 0.0, (0, 0), 1.0).flat_depth == 100.0
    with pytest.raises(ValueError):
        bathymetry.seamount(100.0, 100.0, (0, 0), 50.0)       # summit at the surface
    with pytest.raises(KeyError):
        bathymetry.make_bathymetry({"seamount": {"depth": 100.0, "height": 10.0}})   # no default centre or width


def range_average(loss, ranges, window):
    keep = (ranges >= window[0]) & (ranges <= window[1])
    return -10.0 * np.log10(np.mean(10.0 ** (-loss[..., keep] / 10.0), axis=-1))


@pytest.fixture(scope="module")
def bellhop3d_runs():
    bed = bathymetry.make_bathymetry({"seamount": SEAMOUNT})
    out = {}
    for mode in ("3", "2"):
        out[mode] = run_bellhop3d(bed.depth, GRID_X, GRID_Y, WATER_DEPTH, WATER, BOTTOM, FREQUENCY, SOURCE_DEPTH,
                                  [RECEIVER_DEPTH], BEARINGS, RANGES, mode, **FAN)[:, 0, :]
    return out


@needs_bellhop3d
def test_horizontal_refraction_by_the_seamount(bellhop3d_runs, capsys):
    """Bellhop3D against itself (3-D and N x 2-D modes): this checks our Bellhop3D file writers
    (bellhop3d_oracle.py) and `bathymetry.seamount`, and fixes the size of the effect the library misses; it
    does not test library propagation. Margins: the smallest magnitudes measured at either beam count (3.0 dB
    over the flanks, 1.1 dB beside) less 0.6 dB, the most the 3-D minus N x 2-D difference moved between the
    two beam counts: 2.4 dB, taken as 2.0, and 0.5 dB."""
    full, flat = bellhop3d_runs["3"], bellhop3d_runs["2"]
    near = range_average(full, RANGES, NEAR) - range_average(flat, RANGES, NEAR)
    far = range_average(full, RANGES, FAR) - range_average(flat, RANGES, FAR)
    at = {b: far[int(np.argmin(np.abs(BEARINGS - b)))] for b in FLANKS + BESIDE}
    with capsys.disabled():
        print("\n[bellhop3d] 3-D minus N x 2-D, incoherent TL averaged over 3.5 to 6.5 km, by bearing: "
              + ", ".join(f"{b:+.0f} deg {d:+.1f} dB" for b, d in zip(BEARINGS, far))
              + f"; before the seamount max |diff| {np.abs(near).max():.2f} dB")
    assert np.abs(near).max() < 0.3                       # in-plane relief alone: the two modes agree
    assert all(at[b] > 2.0 for b in FLANKS), at           # energy turned out of the bearings over the flanks
    assert all(at[b] < -0.5 for b in BESIDE), at          # more energy beside the seamount
    assert far.max() < 6.0 and far.min() > -3.0


@needs_bellhop3d
@needs_bellhop
def test_our_n_by_2d_tier_over_the_seamount(bellhop3d_runs, capsys):
    """Our tier, one profile per bearing, against Bellhop3D's N x 2-D (the same approximation, another code) and
    its 3-D (the field we miss). Ours is 1.0 to 2.5 dB louder than Bellhop3D's N x 2-D on these bearings with the
    test's beams (0.4 to 1.9 dB with four times as many; largest over the summit, 6 degrees); over a level 200 m
    bed the two codes differ by up to 1.0 dB at 7 km (ours louder, measured: their beams, geometric Gaussian in
    Bellhop3D here and geometric hat, Bellhop's default, in our runs, and their bed interpolation differ). The
    3-D error is the one quantified in the module docstring."""
    bed = bathymetry.make_bathymetry({"seamount": SEAMOUNT})
    env = OceanEnvironment(depth=WATER_DEPTH, sound_speed=WATER, bottom_speed=BOTTOM[0], bottom_density=BOTTOM[1],
                           bottom_attenuation=BOTTOM[2])
    base = BellhopProvider(env, FREQUENCY, max_range=RANGES[-1] + 100.0, range_step=100.0, min_range=100.0,
                           depth_step=20.0, angles=FAN["alpha_deg"])
    assert np.any(np.isclose(base.depths, RECEIVER_DEPTH)) and np.allclose(base.ranges[:len(RANGES)], RANGES)
    profile_ranges = np.arange(0.0, RANGES[-1] + 200.0, 25.0)
    lines, to_flat, to_full = [], [], []
    for b in OURS:
        u = np.array([np.cos(np.radians(b)), np.sin(np.radians(b))])
        provider = base.along_profile(np.column_stack([profile_ranges, bed.depth(profile_ranges * u[0], profile_ranges * u[1])]))
        rays = provider.eigenrays(np.array([0.0, 0.0, SOURCE_DEPTH]),
                                  np.column_stack([RANGES * u[0], RANGES * u[1], np.full_like(RANGES, RECEIVER_DEPTH)]))
        ours = -10.0 * np.log10(np.maximum(np.sum(np.where(rays.valid, np.abs(rays.amplitude) ** 2, 0.0), axis=1), 1e-300))   # no ray: blocked
        i = int(np.argmin(np.abs(BEARINGS - b)))
        mine, flat, full = (range_average(v, RANGES, FAR) for v in (ours, bellhop3d_runs["2"][i], bellhop3d_runs["3"][i]))
        to_flat.append(mine - flat)
        to_full.append(mine - full)
        lines.append(f"{b:+.0f} deg: ours {mine:.1f}, Bellhop3D N x 2-D {flat:.1f}, 3-D {full:.1f} dB")
    to_flat, to_full = np.array(to_flat), np.array(to_full)
    with capsys.disabled():
        print("\n[bellhop3d] incoherent TL averaged over 3.5 to 6.5 km: " + "; ".join(lines))
    # The same approximation in two codes: 2.5 dB measured at most here, and the codes differ by up to 1.0 dB
    # over a level bed; 3.5 dB is the sum.
    assert np.abs(to_flat).max() < 3.5, to_flat
    # CHARACTERISATION OF A KNOWN GAP, not a requirement: our tier has no out-of-plane propagation and is too
    # loud over the flanks (-3.8 dB converged, -4.5 dB with these beams). When 3-D propagation arrives this
    # assertion must be changed to an agreement bound with the 3-D field.
    flanks = [k for k, b in enumerate(OURS) if b in FLANKS]
    assert np.all(to_full[flanks] < -2.0), to_full         # ours is too loud over the flanks: the 3-D error


# --- P4 (sonar-rcq): the opt-in 3-D tier against the same seamount ------------------------------------------------

# The acceptance runs Bellhop3D three times at converged fans (about 7 minutes on a shared 4-core machine: the
# converged oracle 3 min, the provider's two runs 5 min), more than the one minute an oracle test may take in the
# suite, so it runs only when asked: GPUSONAR_SLOW_ORACLES=1 (and the oracle).
slow = pytest.mark.skipif(os.environ.get("GPUSONAR_SLOW_ORACLES", "") in ("", "0"),
                          reason="slow 3-D acceptance (about 8 minutes): set GPUSONAR_SLOW_ORACLES=1 to run it")
ACCEPT_BEARINGS = np.array([-24.0, -4.0, 16.0, 36.0])            # equally spaced for the oracle's writer
ACCEPT_RANGES = np.arange(500.0, 6500.01, 500.0)
ACCEPT_FLANKS = (-4.0, 16.0)                                     # over the flanks (cross-slopes 12.3 and 11.8 degrees)


@slow
@needs_bellhop3d
def test_eigenray3d_tier_over_the_seamount(capsys):
    """The acceptance bound of docs/tasks/PROPAGATION_3D.md P4, revised with D-197: the `eigenray3d` provider
    (hat beams 241 x 181 over +-30 degrees and 120 degrees of azimuth: the default 361 halves the azimuth step and
    doubles the time; measured within 0.42 dB at 361) against Bellhop3D's converged 3-D field (Gaussian beams,
    481 x 361; D-197: 121 is 0.5 to 1.2 dB short) and its N x 2-D field at the same fan, 50 m, four bearings, ranges
    0.5 to 6.5 km every 500 m. Measured 2026-11-26 (averages of the seven ranges from 3.5 to 6.5 km): see the printed
    line; ours minus 3-D -0.01, -0.10, -1.17, +0.01 dB and ours minus N x 2-D -0.45, +3.37, +2.24, -0.71 dB on -24, -4,
    16, 36 degrees; 0.05 dB at most at 0.5 and 1 km; 8 minutes, 223 kB of arrivals for 52 receivers (with bearings -20,
    -7, 6, 19, 32 and 13 ranges: +0.02, -0.04, -0.05, -0.56, -0.05 dB). Bounds (the plan's, docs/tasks/
    PROPAGATION_3D.md section 5, with D-197's converged oracle):
    1. |ours - Bellhop3D 3-D| < 1.5 dB on every bearing (3.5 to 6.5 km averages): 1.17 measured at most; half the
       smallest 3-D effect over the flanks (3.0 dB), so the test cannot pass without it.
    2. ours - Bellhop3D N x 2-D > 2.0 dB over the flanks (-4 and 16 degrees): the energy is turned out of them.
    3. Before the seamount (0.5 and 1.0 km), |ours - Bellhop3D 3-D| < 0.5 dB at each range."""
    from gpusonar.sim.bellhop3d import Bellhop3DProvider
    bed = bathymetry.make_bathymetry({"seamount": SEAMOUNT})
    ref = {mode: run_bellhop3d(bed.depth, GRID_X, GRID_Y, WATER_DEPTH, WATER, BOTTOM, FREQUENCY, SOURCE_DEPTH, [RECEIVER_DEPTH],
                               ACCEPT_BEARINGS, ACCEPT_RANGES, mode, 481, (-30.0, 30.0), 361, (-60.0, 60.0), FAN["box_km"],
                               beam="B")[:, 0, :] for mode in ("3", "2")}
    env = OceanEnvironment(depth=WATER_DEPTH, sound_speed=WATER, bottom_speed=BOTTOM[0], bottom_density=BOTTOM[1],
                           bottom_attenuation=BOTTOM[2])
    provider = Bellhop3DProvider(env, FREQUENCY, max_range=ACCEPT_RANGES[-1] + 100.0, bathymetry={"seamount": SEAMOUNT},
                                 range_step=50.0, depth_step=20.0, n_beta=181)
    pts = np.concatenate([np.column_stack([ACCEPT_RANGES * np.cos(np.radians(b)), ACCEPT_RANGES * np.sin(np.radians(b)),
                                           np.full(len(ACCEPT_RANGES), RECEIVER_DEPTH)]) for b in ACCEPT_BEARINGS])
    es = provider.eigenrays(np.array([0.0, 0.0, SOURCE_DEPTH]), pts)
    ours = (-10.0 * np.log10(np.maximum(np.sum(np.where(es.valid, np.abs(es.amplitude) ** 2, 0.0), axis=1), 1e-300))
            ).reshape(len(ACCEPT_BEARINGS), len(ACCEPT_RANGES))
    to_full = range_average(ours, ACCEPT_RANGES, FAR) - range_average(ref["3"], ACCEPT_RANGES, FAR)
    to_flat = range_average(ours, ACCEPT_RANGES, FAR) - range_average(ref["2"], ACCEPT_RANGES, FAR)
    near = (ours - ref["3"])[:, ACCEPT_RANGES <= 1000.0]
    with capsys.disabled():
        print("\n[eigenray3d] seamount, 3.5 to 6.5 km: " + "; ".join(
            f"{b:+.0f} deg: ours minus 3-D {a:+.2f}, minus N x 2-D {c:+.2f} dB" for b, a, c in zip(ACCEPT_BEARINGS, to_full, to_flat))
              + f"; at 0.5 and 1 km max |ours - 3-D| {np.abs(near).max():.2f} dB; runs {provider.runs}")
    assert np.abs(to_full).max() < 1.5, to_full
    flanks = [k for k, b in enumerate(ACCEPT_BEARINGS) if b in ACCEPT_FLANKS]
    assert np.all(to_flat[flanks] > 2.0), to_flat
    assert np.abs(near).max() < 0.5, near
