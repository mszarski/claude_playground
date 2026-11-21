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
0.6 dB): before the seamount (0.2 to 1.8 km) 3-D and N x 2-D agree to 0.07 dB on every bearing, so the two modes
differ only through the seamount. Averaged over 3.5 to 6.5 km: on bearings that pass over the flanks (-4 to -2
and 16 to 18 degrees) the 3-D loss is 3.1 to 3.5 dB HIGHER [3.0 to 3.6]: energy is turned out of those bearings;
on bearings beside the seamount (-20 and 32 degrees) it is 1.1 to 1.2 dB LOWER [1.1 to 1.2]: the energy turned
out arrives there. Over the summit itself (6 degrees, where the 20 m summit blocks the 50 m paths in both) the
difference is 0.9 dB [0.4]. N x 2-D sees none of it. Our tier against the converged 3-D field: -3.8 dB at -4
and at 16 degrees (too loud), -2.2 dB over the summit, 0.0 and -0.7 dB beside it at -20 and 32 degrees
(against the test's beams: -4.5, -4.5, -3.4, -0.5, -1.2 dB).
"""

from __future__ import annotations

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
    full, flat = bellhop3d_runs["3"], bellhop3d_runs["2"]
    near = range_average(full, RANGES, NEAR) - range_average(flat, RANGES, NEAR)
    far = range_average(full, RANGES, FAR) - range_average(flat, RANGES, FAR)
    at = {b: far[int(np.argmin(np.abs(BEARINGS - b)))] for b in FLANKS + BESIDE}
    with capsys.disabled():
        print("\n[bellhop3d] 3-D minus N x 2-D, incoherent TL averaged over 3.5 to 6.5 km, by bearing: "
              + ", ".join(f"{b:+.0f} deg {d:+.1f} dB" for b, d in zip(BEARINGS, far))
              + f"; before the seamount max |diff| {np.abs(near).max():.2f} dB")
    assert np.abs(near).max() < 0.3                       # the two modes agree where the bed is level
    assert all(at[b] > 2.0 for b in FLANKS), at           # energy turned out of the bearings over the flanks ...
    assert all(at[b] < -0.8 for b in BESIDE), at          # ... arrives beside the seamount
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
    assert np.abs(to_flat).max() < 3.0, to_flat            # the same approximation in two codes
    flanks = [k for k, b in enumerate(OURS) if b in FLANKS]
    assert np.all(to_full[flanks] < -2.0), to_full         # ours is too loud over the flanks: the 3-D error
